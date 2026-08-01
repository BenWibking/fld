#include "FLDTestCommon.H"

#include <AMReX_MFIter.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Reduce.H>

namespace fld_test
{

using namespace amrex;

GaussianResult
run_gaussian (bool use_amr)
{
    Array<int, AMREX_SPACEDIM> const periodic{
        AMREX_D_DECL(1, 1, 1)};
    DiffusionHierarchy hierarchy =
        use_amr ? make_centered_patch_hierarchy(32, 2, 16, periodic)
                : make_uniform_hierarchy(64, 16, periodic);
    auto masks = make_composite_masks(hierarchy);
    auto state = make_cell_data(hierarchy, 1, 1);
    auto solution = make_cell_data(hierarchy, 1, 1);
    auto rhs = make_cell_data(hierarchy, 1, 0);
    auto acoef = make_cell_data(hierarchy, 1, 0);
    auto bcoef = make_face_data(hierarchy);

    Real constexpr extinction = Real(100);
    Real constexpr diffusion = Real(1) / (Real(3) * extinction);
    Real constexpr initial_time = Real(0.4);
    Real constexpr dt = Real(0.005);
    int constexpr steps = 10;
    Real constexpr pi = Real(3.1415926535897932384626433832795);

    set_level_data(acoef, Real(1));
    for (auto& level : bcoef) {
        for (auto& face : level) {
            face->setVal(diffusion);
        }
    }
    for (int level = 0; level < static_cast<int>(state.size()); ++level) {
        auto const dx = hierarchy.geom[level].CellSizeArray();
        auto const problo = hierarchy.geom[level].ProbLoArray();
        for (MFIter mfi(*state[level]); mfi.isValid(); ++mfi) {
            auto const array = state[level]->array(mfi);
            ParallelFor(mfi.validbox(),
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                Real const x = problo[0] + (Real(i) + Real(0.5)) * dx[0] -
                               Real(0.5);
                Real const y = problo[1] + (Real(j) + Real(0.5)) * dx[1] -
                               Real(0.5);
                Real const radius_squared = x * x + y * y;
                array(i, j, k) =
                    std::exp(-radius_squared /
                             (Real(4) * diffusion * initial_time)) /
                    (Real(4) * pi * diffusion * initial_time);
            });
        }
    }
    average_down_hierarchy(state, hierarchy);
    Real const initial_energy =
        composite_volume_sum(state, hierarchy, masks);

    auto const periodic_bc = Array<LinOpBCType, AMREX_SPACEDIM>{
        AMREX_D_DECL(LinOpBCType::Periodic, LinOpBCType::Periodic,
                     LinOpBCType::Periodic)};
    MLABecLapAMG solver(hierarchy.geom, hierarchy.grids, hierarchy.dmap);
    solver.setup(Real(1), dt, get_level_const_ptrs(acoef),
                 get_face_const_ptrs(bcoef), periodic_bc, periodic_bc, {});

    GaussianResult result;
    result.cells = composite_cell_count(masks);
    record_setup(result.solver, solver);
    for (int step = 0; step < steps; ++step) {
        copy_level_data(rhs, state);
        set_level_data(solution, Real(0));
        auto const info = solver.solve(get_level_ptrs(solution),
                                       get_level_const_ptrs(rhs),
                                       linear_tolerance(), Real(0));
        record_solve(result.solver, info);
        copy_level_data(state, solution);
        average_down_hierarchy(state, hierarchy);
    }

    Real const final_time = initial_time + Real(steps) * dt;
    ReduceOps<ReduceOpSum, ReduceOpSum> reduce_op;
    ReduceData<Real, Real> reduce_data(reduce_op);
    using Tuple = typename decltype(reduce_data)::Type;
    for (int level = 0; level < static_cast<int>(state.size()); ++level) {
        auto const dx = hierarchy.geom[level].CellSizeArray();
        auto const problo = hierarchy.geom[level].ProbLoArray();
        Real const volume = AMREX_D_TERM(dx[0], *dx[1], *dx[2]);
        for (MFIter mfi(*state[level]); mfi.isValid(); ++mfi) {
            auto const array = state[level]->const_array(mfi);
            auto const mask = masks[level].const_array(mfi);
            reduce_op.eval(mfi.validbox(), reduce_data,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept -> Tuple
            {
                Real const x = problo[0] + (Real(i) + Real(0.5)) * dx[0] -
                               Real(0.5);
                Real const y = problo[1] + (Real(j) + Real(0.5)) * dx[1] -
                               Real(0.5);
                Real const reference =
                    std::exp(-(x * x + y * y) /
                             (Real(4) * diffusion * final_time)) /
                    (Real(4) * pi * diffusion * final_time);
                if (mask(i, j, k) == 0) {
                    return {Real(0), Real(0)};
                }
                return {volume * std::abs(array(i, j, k) - reference),
                        volume * std::abs(reference)};
            });
        }
    }
    auto const reductions = reduce_data.value(reduce_op);
    Real absolute_error = amrex::get<0>(reductions);
    Real exact_norm = amrex::get<1>(reductions);
    ParallelDescriptor::ReduceRealSum(absolute_error);
    ParallelDescriptor::ReduceRealSum(exact_norm);
    result.relative_l1_error = absolute_error / exact_norm;
    Real const final_energy = composite_volume_sum(state, hierarchy, masks);
    result.relative_energy_drift =
        std::abs(final_energy - initial_energy) / initial_energy;

    Real const error_limit =
        (sizeof(Real) == sizeof(float)) ? Real(0.12) : Real(0.06);
    Real const conservation_limit =
        (sizeof(Real) == sizeof(float)) ? Real(2.e-3) : Real(2.e-7);
    AMREX_ALWAYS_ASSERT(result.relative_l1_error < error_limit);
    AMREX_ALWAYS_ASSERT(result.relative_energy_drift < conservation_limit);
    return result;
}

} // namespace fld_test
