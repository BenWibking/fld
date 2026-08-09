#include "DiffusionHierarchy.H"
#include "FLDFieldOps.H"
#include "FLDTest.H"
#include "MLABecAMG.H"

#include <AMReX.H>
#include <AMReX_Arena.H>
#include <AMReX_BoxIterator.H>
#include <AMReX_MFIter.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Reduce.H>

#include <algorithm>

namespace fld_test
{

using namespace amrex;

namespace
{

Array<int, AMREX_SPACEDIM>
nonperiodic ()
{
    return {AMREX_D_DECL(0, 0, 0)};
}

Array<LinOpBCType, AMREX_SPACEDIM>
all_bc (LinOpBCType type)
{
    return {AMREX_D_DECL(type, type, type)};
}

Real
linear_tolerance ()
{
    return (sizeof(Real) == sizeof(float)) ? Real(5.e-5) : Real(2.e-10);
}

void
fill_constant_faces (FaceData& faces, Real value)
{
    for (auto& level : faces) {
        for (auto& face : level) {
            face->setVal(value);
        }
    }
}

struct ReferenceCell
{
    int level = -1;
    int grid = -1;
    IntVect index = IntVect::TheZeroVector();
    int xlo = 0;
    int xhi = 0;
    int ylo = 0;
    int yhi = 0;
    Real x = Real(0);
    Real y = Real(0);
    Real volume = Real(0);
    Real value = Real(0);
    Real acoef = Real(0);
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE Real
operator_test_value (Real x, Real y) noexcept
{
    return Real(0.7) + Real(0.6) * x - Real(0.2) * y +
           Real(0.15) * x * y;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE Real
operator_test_acoef (Real x, Real y) noexcept
{
    return Real(1) + Real(0.1) * x + Real(0.2) * y;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE Real
operator_test_bcoef (Real x, Real y) noexcept
{
    return Real(0.8) + Real(0.3) * x + Real(0.1) * y;
}

Vector<ReferenceCell>
make_reference_cells (DiffusionHierarchy const& hierarchy)
{
    int const nlevels = static_cast<int>(hierarchy.geom.size());
    int const finest_cells =
        hierarchy.geom.back().Domain().length(0);
    Vector<BoxArray> covered(nlevels - 1);
    for (int level = 0; level + 1 < nlevels; ++level) {
        covered[level] = hierarchy.grids[level + 1];
        covered[level].coarsen(hierarchy.ref_ratio[level]);
    }

    Vector<ReferenceCell> result;
    for (int level = 0; level < nlevels; ++level) {
        int const level_cells = hierarchy.geom[level].Domain().length(0);
        AMREX_ALWAYS_ASSERT(finest_cells % level_cells == 0);
        int const scale = finest_cells / level_cells;
        auto const dx = hierarchy.geom[level].CellSizeArray();
        for (int grid = 0; grid < hierarchy.grids[level].size(); ++grid) {
            for (BoxIterator bit(hierarchy.grids[level][grid]); bit.ok();
                 ++bit) {
                IntVect const iv = bit();
                bool is_covered = false;
                if (level + 1 < nlevels) {
                    for (int fine_grid = 0;
                         fine_grid < covered[level].size(); ++fine_grid) {
                        is_covered = is_covered ||
                                     covered[level][fine_grid].contains(iv);
                    }
                }
                if (is_covered) {
                    continue;
                }
                int const xlo = iv[0] * scale;
                int const ylo = iv[1] * scale;
                Real const x = (Real(xlo) + Real(0.5 * scale)) /
                               Real(finest_cells);
                Real const y = (Real(ylo) + Real(0.5 * scale)) /
                               Real(finest_cells);
                result.push_back(ReferenceCell{
                    level, grid, iv, xlo, xlo + scale, ylo, ylo + scale,
                    x, y, dx[0] * dx[1], operator_test_value(x, y),
                    operator_test_acoef(x, y)});
            }
        }
    }
    return result;
}

Real
reference_operator_action (ReferenceCell const& source,
                           Vector<ReferenceCell> const& cells,
                           int finest_cells, Real ascalar, Real bscalar)
{
    Real flux_sum = Real(0);
    for (auto const& target : cells) {
        if (&target == &source) {
            continue;
        }
        int direction = -1;
        int overlap = 0;
        int face_coordinate = 0;
        int transverse_lo = 0;
        int transverse_hi = 0;
        int source_width = 0;
        int target_width = 0;
        if (source.xhi == target.xlo || source.xlo == target.xhi) {
            overlap = std::max(
                0, std::min(source.yhi, target.yhi) -
                       std::max(source.ylo, target.ylo));
            if (overlap > 0) {
                direction = 0;
                face_coordinate =
                    source.xhi == target.xlo ? source.xhi : source.xlo;
                transverse_lo = std::max(source.ylo, target.ylo);
                transverse_hi = std::min(source.yhi, target.yhi);
                source_width = source.xhi - source.xlo;
                target_width = target.xhi - target.xlo;
            }
        } else if (source.yhi == target.ylo ||
                   source.ylo == target.yhi) {
            overlap = std::max(
                0, std::min(source.xhi, target.xhi) -
                       std::max(source.xlo, target.xlo));
            if (overlap > 0) {
                direction = 1;
                face_coordinate =
                    source.yhi == target.ylo ? source.yhi : source.ylo;
                transverse_lo = std::max(source.xlo, target.xlo);
                transverse_hi = std::min(source.xhi, target.xhi);
                source_width = source.yhi - source.ylo;
                target_width = target.yhi - target.ylo;
            }
        }
        if (direction < 0) {
            continue;
        }

        Real const normal = Real(face_coordinate) / Real(finest_cells);
        Real const transverse =
            Real(transverse_lo + transverse_hi) /
            (Real(2) * Real(finest_cells));
        Real const face_x = direction == 0 ? normal : transverse;
        Real const face_y = direction == 0 ? transverse : normal;
        Real const area = Real(overlap) / Real(finest_cells);
        Real const distance = Real(source_width + target_width) /
                              (Real(2) * Real(finest_cells));
        flux_sum += operator_test_bcoef(face_x, face_y) * area / distance *
                    (source.value - target.value);
    }
    return ascalar * source.acoef * source.value +
           bscalar * flux_sum / source.volume;
}

void
fill_operator_test_fields (DiffusionHierarchy const& hierarchy,
                           LevelData& input, LevelData& acoef,
                           FaceData& bcoef)
{
    for (int level = 0; level < static_cast<int>(hierarchy.geom.size());
         ++level) {
        auto const dx = hierarchy.geom[level].CellSizeArray();
        auto const problo = hierarchy.geom[level].ProbLoArray();
        for (MFIter mfi(*input[level]); mfi.isValid(); ++mfi) {
            auto const phi = input[level]->array(mfi);
            auto const a = acoef[level]->array(mfi);
            ParallelFor(mfi.validbox(),
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                Real const x =
                    problo[0] + (Real(i) + Real(0.5)) * dx[0];
                Real const y =
                    problo[1] + (Real(j) + Real(0.5)) * dx[1];
                phi(i, j, k) = operator_test_value(x, y);
                a(i, j, k) = operator_test_acoef(x, y);
            });
        }
        for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
            for (MFIter mfi(*bcoef[level][direction]); mfi.isValid(); ++mfi) {
                auto const b = bcoef[level][direction]->array(mfi);
                ParallelFor(mfi.validbox(),
                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    Real const x = problo[0] +
                        (Real(i) + (direction == 0 ? Real(0) : Real(0.5))) *
                            dx[0];
                    Real const y = problo[1] +
                        (Real(j) + (direction == 1 ? Real(0) : Real(0.5))) *
                            dx[1];
                    b(i, j, k) = operator_test_bcoef(x, y);
                });
            }
        }
    }
}

void
check_operator_action (DiffusionHierarchy const& hierarchy)
{
    Real constexpr ascalar = Real(0.7);
    Real constexpr bscalar = Real(0.4);
    auto input = make_cell_data(hierarchy, 1, 0);
    auto output = make_cell_data(hierarchy, 1, 0);
    auto expected = make_cell_data(hierarchy, 1, 0);
    auto acoef = make_cell_data(hierarchy, 1, 0);
    auto bcoef = make_face_data(hierarchy);
    set_level_data(output, Real(0));
    set_level_data(expected, Real(0));
    fill_operator_test_fields(hierarchy, input, acoef, bcoef);

    auto const cells = make_reference_cells(hierarchy);
    int const finest_cells = hierarchy.geom.back().Domain().length(0);
    LevelData host_expected(hierarchy.geom.size());
    MFInfo const pinned_info = MFInfo().SetArena(The_Pinned_Arena());
    for (int level = 0; level < static_cast<int>(hierarchy.geom.size());
         ++level) {
        host_expected[level] = std::make_unique<MultiFab>(
            hierarchy.grids[level], hierarchy.dmap[level], 1, 0,
            pinned_info);
        host_expected[level]->setVal(Real(0));
    }
    int const myproc = ParallelDescriptor::MyProc();
    for (auto const& cell : cells) {
        if (hierarchy.dmap[cell.level][cell.grid] == myproc) {
            (*host_expected[cell.level])[cell.grid](cell.index) =
                reference_operator_action(cell, cells, finest_cells,
                                          ascalar, bscalar);
        }
    }
    for (int level = 0; level < static_cast<int>(hierarchy.geom.size());
         ++level) {
        MultiFab::Copy(*expected[level], *host_expected[level], 0, 0, 1, 0);
    }

    auto const neumann = all_bc(LinOpBCType::Neumann);
    MLABecLapAMG solver(hierarchy.geom, hierarchy.grids, hierarchy.dmap);
    solver.setup(ascalar, bscalar, get_level_const_ptrs(acoef),
                 get_face_const_ptrs(bcoef), neumann, neumann, {});
    solver.apply(get_level_ptrs(output), get_level_const_ptrs(input));

    auto error = clone_level_data(output);
    lincomb_level_data(error, Real(1), output, Real(-1), expected);
    auto masks = make_composite_masks(hierarchy);
    auto const [error_minimum, error_maximum] =
        composite_minimum_maximum(error, masks);
    auto const [expected_minimum, expected_maximum] =
        composite_minimum_maximum(expected, masks);
    Real const error_norm =
        amrex::max(std::abs(error_minimum), std::abs(error_maximum));
    Real const expected_norm =
        amrex::max(std::abs(expected_minimum), std::abs(expected_maximum));
    Real const tolerance =
        (sizeof(Real) == sizeof(float)) ? Real(2.e-4) : Real(2.e-11);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        error_norm <= tolerance * amrex::max(Real(1), expected_norm),
        "MLABecLapAMG operator action disagrees with the independent "
        "geometric reference");
}

void
check_coarse_fine_ghost_fill ()
{
    auto hierarchy =
        make_centered_patch_hierarchy(8, 2, 4, nonperiodic());
    auto state = make_cell_data(hierarchy, 1, 1);
    for (int level = 0; level < static_cast<int>(state.size()); ++level) {
        auto const dx = hierarchy.geom[level].CellSizeArray();
        auto const problo = hierarchy.geom[level].ProbLoArray();
        for (MFIter mfi(*state[level]); mfi.isValid(); ++mfi) {
            auto const values = state[level]->array(mfi);
            ParallelFor(mfi.validbox(),
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                Real const x =
                    problo[0] + (Real(i) + Real(0.5)) * dx[0];
                Real const y =
                    problo[1] + (Real(j) + Real(0.5)) * dx[1];
                values(i, j, k) = Real(1) + x + Real(2) * y;
            });
        }
    }
    fill_level_ghosts(state, hierarchy);

    int constexpr fine_level = 1;
    auto const dx = hierarchy.geom[fine_level].CellSizeArray();
    auto const problo = hierarchy.geom[fine_level].ProbLoArray();
    ReduceOps<ReduceOpMax> reduce_op;
    ReduceData<Real> reduce_data(reduce_op);
    using Tuple = typename decltype(reduce_data)::Type;
    for (MFIter mfi(*state[fine_level]); mfi.isValid(); ++mfi) {
        auto const values = state[fine_level]->const_array(mfi);
        reduce_op.eval(mfi.growntilebox(1), reduce_data,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept -> Tuple
        {
            Real const x = problo[0] + (Real(i) + Real(0.5)) * dx[0];
            Real const y = problo[1] + (Real(j) + Real(0.5)) * dx[1];
            return {amrex::Math::abs(values(i, j, k) -
                                     (Real(1) + x + Real(2) * y))};
        });
    }
    Real error = amrex::get<0>(reduce_data.value(reduce_op));
    ParallelDescriptor::ReduceRealMax(error);
    Real const limit =
        (sizeof(Real) == sizeof(float)) ? Real(1.e-5) : Real(1.e-12);
    AMREX_ALWAYS_ASSERT(error < limit);
}

void
check_constant_composite (DiffusionHierarchy const& hierarchy,
                          bool require_coarse_fine)
{
    auto solution = make_cell_data(hierarchy, 1, 1);
    auto rhs = make_cell_data(hierarchy, 1, 0);
    auto exact = make_cell_data(hierarchy, 1, 0);
    auto acoef = make_cell_data(hierarchy, 1, 0);
    auto bcoef = make_face_data(hierarchy);
    set_level_data(solution, Real(0));
    set_level_data(rhs, Real(1));
    set_level_data(exact, Real(1));
    set_level_data(acoef, Real(1));
    fill_constant_faces(bcoef, Real(1));

    MLABecLapAMG solver(hierarchy.geom, hierarchy.grids, hierarchy.dmap);
    auto const neumann = all_bc(LinOpBCType::Neumann);
    solver.setup(Real(1), Real(0.01), get_level_const_ptrs(acoef),
                 get_face_const_ptrs(bcoef), neumann, neumann, {});
    auto preconditioned = make_cell_data(hierarchy, 1, 1);
    auto repeated = make_cell_data(hierarchy, 1, 1);
    auto scaled_rhs = clone_level_data(rhs);
    auto scaled_preconditioned = make_cell_data(hierarchy, 1, 1);
    set_level_data(preconditioned, Real(0));
    set_level_data(repeated, Real(0));
    set_level_data(scaled_preconditioned, Real(0));
    solver.precondition(get_level_ptrs(preconditioned),
                        get_level_const_ptrs(rhs));
    solver.precondition(get_level_ptrs(repeated),
                        get_level_const_ptrs(rhs));
    for (auto& field : scaled_rhs) {
        field->mult(Real(2), 0, 1, 0);
    }
    solver.precondition(get_level_ptrs(scaled_preconditioned),
                        get_level_const_ptrs(scaled_rhs));
    auto repeat_error = clone_level_data(repeated);
    lincomb_level_data(repeat_error, Real(1), repeated, Real(-1),
                       preconditioned);
    auto linearity_error = clone_level_data(scaled_preconditioned);
    lincomb_level_data(linearity_error, Real(1), scaled_preconditioned,
                       Real(-2), preconditioned);
    auto preconditioner_masks = make_composite_masks(hierarchy);
    auto const [repeat_minimum, repeat_maximum] =
        composite_minimum_maximum(repeat_error, preconditioner_masks);
    auto const [linearity_minimum, linearity_maximum] =
        composite_minimum_maximum(linearity_error, preconditioner_masks);
    Real const preconditioner_tolerance =
        (sizeof(Real) == sizeof(float)) ? Real(2.e-5) : Real(2.e-12);
    AMREX_ALWAYS_ASSERT(
        amrex::max(std::abs(repeat_minimum), std::abs(repeat_maximum)) <
        preconditioner_tolerance);
    AMREX_ALWAYS_ASSERT(
        amrex::max(std::abs(linearity_minimum),
                   std::abs(linearity_maximum)) <
        preconditioner_tolerance);
    auto const info = solver.solve(get_level_ptrs(solution),
                                   get_level_const_ptrs(rhs),
                                   linear_tolerance(), Real(0));
    amrex::ignore_unused(info);
    auto error = clone_level_data(solution);
    lincomb_level_data(error, Real(1), solution, Real(-1), exact);
    auto masks = make_composite_masks(hierarchy);
    auto const [minimum, maximum] = composite_minimum_maximum(error, masks);
    Real const error_norm = amrex::max(std::abs(minimum), std::abs(maximum));
    Real const limit =
        (sizeof(Real) == sizeof(float)) ? Real(5.e-4) : Real(5.e-9);
    AMREX_ALWAYS_ASSERT(error_norm < limit);
    AMREX_ALWAYS_ASSERT(solver.assembly_diagnostics().global_rows ==
                        composite_cell_count(masks));
    if (ParallelDescriptor::NProcs() <= 4) {
        AMREX_ALWAYS_ASSERT(solver.assembly_diagnostics().local_rows > 0);
    }
    if (require_coarse_fine) {
        AMREX_ALWAYS_ASSERT(
            solver.assembly_diagnostics().coarse_fine_connections > 0);
    }

    set_level_data(solution, Real(0));
    auto const second = solver.solve(get_level_ptrs(solution),
                                     get_level_const_ptrs(rhs),
                                     linear_tolerance(), Real(0));
    amrex::ignore_unused(second);
    AMREX_ALWAYS_ASSERT(solver.assembly_diagnostics().setup_generation == 1);
    solver.setup(Real(1), Real(0.02), get_level_const_ptrs(acoef),
                 get_face_const_ptrs(bcoef), neumann, neumann, {});
    AMREX_ALWAYS_ASSERT(solver.assembly_diagnostics().setup_generation == 2);
}

void
check_dirichlet_linear ()
{
    auto hierarchy = make_uniform_hierarchy(8, 4, nonperiodic());
    auto solution = make_cell_data(hierarchy, 1, 1);
    auto rhs = make_cell_data(hierarchy, 1, 0);
    auto acoef = make_cell_data(hierarchy, 1, 0);
    auto boundary = make_cell_data(hierarchy, 1, 1);
    auto exact = make_cell_data(hierarchy, 1, 0);
    auto bcoef = make_face_data(hierarchy);
    set_level_data(solution, Real(0));
    set_level_data(rhs, Real(0));
    set_level_data(acoef, Real(1));
    fill_constant_faces(bcoef, Real(1));

    auto const dx = hierarchy.geom[0].CellSizeArray();
    for (MFIter mfi(*boundary[0]); mfi.isValid(); ++mfi) {
        auto const bc = boundary[0]->array(mfi);
        Box const box = mfi.fabbox();
        ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            Real const x = amrex::max(
                Real(0), amrex::min(Real(1), (Real(i) + Real(0.5)) * dx[0]));
            Real const y = amrex::max(
                Real(0), amrex::min(Real(1), (Real(j) + Real(0.5)) * dx[1]));
            bc(i, j, k) = Real(1) + x + Real(2) * y;
        });
        auto const ex = exact[0]->array(mfi);
        ParallelFor(mfi.validbox(),
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            Real const x = (Real(i) + Real(0.5)) * dx[0];
            Real const y = (Real(j) + Real(0.5)) * dx[1];
            ex(i, j, k) = Real(1) + x + Real(2) * y;
        });
    }

    auto const dirichlet = all_bc(LinOpBCType::Dirichlet);
    MLABecLapAMG solver(hierarchy.geom, hierarchy.grids, hierarchy.dmap);
    solver.setup(Real(0), Real(1), get_level_const_ptrs(acoef),
                 get_face_const_ptrs(bcoef), dirichlet, dirichlet,
                 get_level_const_ptrs(boundary));
    auto const info = solver.solve(get_level_ptrs(solution),
                                   get_level_const_ptrs(rhs),
                                   linear_tolerance(), Real(0));
    amrex::ignore_unused(info);
    auto error = clone_level_data(solution);
    lincomb_level_data(error, Real(1), solution, Real(-1), exact);
    auto masks = make_composite_masks(hierarchy);
    auto const [minimum, maximum] = composite_minimum_maximum(error, masks);
    Real const error_norm = amrex::max(std::abs(minimum), std::abs(maximum));
    Real const limit =
        (sizeof(Real) == sizeof(float)) ? Real(2.e-3) : Real(2.e-8);
    AMREX_ALWAYS_ASSERT(error_norm < limit);
}

void
check_robin_constant ()
{
    auto hierarchy = make_uniform_hierarchy(8, 4, nonperiodic());
    auto solution = make_cell_data(hierarchy, 1, 1);
    auto rhs = make_cell_data(hierarchy, 1, 0);
    auto acoef = make_cell_data(hierarchy, 1, 0);
    auto robin_a = make_cell_data(hierarchy, 1, 0);
    auto robin_b = make_cell_data(hierarchy, 1, 0);
    auto robin_f = make_cell_data(hierarchy, 1, 0);
    auto exact = make_cell_data(hierarchy, 1, 0);
    auto bcoef = make_face_data(hierarchy);
    set_level_data(solution, Real(0));
    set_level_data(rhs, Real(0));
    set_level_data(acoef, Real(1));
    set_level_data(robin_a, Real(1));
    set_level_data(robin_b, Real(1));
    set_level_data(robin_f, Real(2));
    set_level_data(exact, Real(2));
    fill_constant_faces(bcoef, Real(1));

    auto const robin = all_bc(LinOpBCType::Robin);
    RobinBCData data{get_level_const_ptrs(robin_a),
                     get_level_const_ptrs(robin_b),
                     get_level_const_ptrs(robin_f)};
    auto masks = make_composite_masks(hierarchy);
    Real const limit =
        (sizeof(Real) == sizeof(float)) ? Real(2.e-3) : Real(2.e-8);
    for (auto preconditioner :
         {MLABecPreconditioner::AMG, MLABecPreconditioner::MLMG}) {
        set_level_data(solution, Real(0));
        MLABecLapAMG solver(hierarchy.geom, hierarchy.grids, hierarchy.dmap,
                            {}, preconditioner);
        solver.setup(Real(0), Real(1), get_level_const_ptrs(acoef),
                     get_face_const_ptrs(bcoef), robin, robin, {}, data);
        auto const info = solver.solve(get_level_ptrs(solution),
                                       get_level_const_ptrs(rhs),
                                       linear_tolerance(), Real(0));
        amrex::ignore_unused(info);
        auto error = clone_level_data(solution);
        lincomb_level_data(error, Real(1), solution, Real(-1), exact);
        auto const [minimum, maximum] =
            composite_minimum_maximum(error, masks);
        Real const error_norm =
            amrex::max(std::abs(minimum), std::abs(maximum));
        AMREX_ALWAYS_ASSERT(error_norm < limit);
    }
}

} // namespace

void
run_mlabeclap_amg_checks ()
{
    check_coarse_fine_ghost_fill();
    check_operator_action(
        make_uniform_hierarchy(8, 4, nonperiodic()));
    check_operator_action(
        make_centered_patch_hierarchy(8, 2, 4, nonperiodic()));
    check_operator_action(
        make_centered_patch_hierarchy(8, 4, 4, nonperiodic()));
    check_operator_action(
        make_strip_hierarchy({12, 48}, {{0, 12}, {16, 32}}, 4,
                             nonperiodic()));
    check_operator_action(
        make_strip_hierarchy({8, 16, 32},
                             {{0, 8}, {4, 12}, {12, 20}}, 4,
                             nonperiodic()));
    check_constant_composite(
        make_uniform_hierarchy(8, 4, nonperiodic()), false);
    check_dirichlet_linear();
    check_robin_constant();
    check_constant_composite(
        make_centered_patch_hierarchy(8, 2, 4, nonperiodic()), true);
    check_constant_composite(
        make_strip_hierarchy({8, 32}, {{0, 8}, {12, 20}}, 8,
                             nonperiodic()),
        true);
    check_constant_composite(
        make_strip_hierarchy({12, 48}, {{0, 12}, {16, 32}}, 4,
                             nonperiodic()),
        true);
    amrex::Print() << "MLABecLapAMG operator-action, boundary, lifecycle, "
                      "and hierarchy-shape checks passed\n";
}

} // namespace fld_test
