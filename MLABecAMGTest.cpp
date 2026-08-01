#include "DiffusionHierarchy.H"
#include "FLDFieldOps.H"
#include "FLDTest.H"
#include "MLABecAMG.H"

#include <AMReX.H>
#include <AMReX_MFIter.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Reduce.H>

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
    MLABecLapAMG solver(hierarchy.geom, hierarchy.grids, hierarchy.dmap);
    solver.setup(Real(0), Real(1), get_level_const_ptrs(acoef),
                 get_face_const_ptrs(bcoef), robin, robin, {}, data);
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

} // namespace

void
run_mlabeclap_amg_checks ()
{
    check_coarse_fine_ghost_fill();
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
    amrex::Print() << "MLABecLapAMG operator, boundary, lifecycle, and "
                      "coarse/fine checks passed\n";
}

} // namespace fld_test
