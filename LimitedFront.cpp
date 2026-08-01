#include "FLDTestCommon.H"

#include <AMReX_MFIter.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Reduce.H>

namespace fld_test
{

using namespace amrex;

namespace
{

Real
front_radius (DiffusionHierarchy const& hierarchy, LevelData const& state,
              Vector<iMultiFab> const& masks, Real threshold,
              Real initial_radius)
{
    ReduceOps<ReduceOpMax> reduce_op;
    ReduceData<Real> reduce_data(reduce_op);
    using Tuple = typename decltype(reduce_data)::Type;
    for (int level = 0; level < static_cast<int>(state.size()); ++level) {
        auto const dx = hierarchy.geom[level].CellSizeArray();
        auto const problo = hierarchy.geom[level].ProbLoArray();
        for (MFIter mfi(*state[level]); mfi.isValid(); ++mfi) {
            auto const e = state[level]->const_array(mfi);
            auto const mask = masks[level].const_array(mfi);
            reduce_op.eval(mfi.validbox(), reduce_data,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept -> Tuple
            {
                if (mask(i, j, k) == 0 || e(i, j, k) <= threshold) {
                    return {initial_radius};
                }
                Real const x = problo[0] + (Real(i) + Real(0.5)) * dx[0] -
                               Real(0.5);
                Real const y = problo[1] + (Real(j) + Real(0.5)) * dx[1] -
                               Real(0.5);
                return {std::sqrt(x * x + y * y)};
            });
        }
    }
    Real result = amrex::get<0>(reduce_data.value(reduce_op));
    ParallelDescriptor::ReduceRealMax(result);
    return result;
}

Real
far_excess (DiffusionHierarchy const& hierarchy, LevelData const& state,
            Vector<iMultiFab> const& masks, Real radius,
            Real ambient_energy)
{
    ReduceOps<ReduceOpMax> reduce_op;
    ReduceData<Real> reduce_data(reduce_op);
    using Tuple = typename decltype(reduce_data)::Type;
    for (int level = 0; level < static_cast<int>(state.size()); ++level) {
        auto const dx = hierarchy.geom[level].CellSizeArray();
        auto const problo = hierarchy.geom[level].ProbLoArray();
        for (MFIter mfi(*state[level]); mfi.isValid(); ++mfi) {
            auto const e = state[level]->const_array(mfi);
            auto const mask = masks[level].const_array(mfi);
            reduce_op.eval(mfi.validbox(), reduce_data,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept -> Tuple
            {
                Real const x = problo[0] + (Real(i) + Real(0.5)) * dx[0] -
                               Real(0.5);
                Real const y = problo[1] + (Real(j) + Real(0.5)) * dx[1] -
                               Real(0.5);
                return {mask(i, j, k) != 0 &&
                                std::sqrt(x * x + y * y) > radius
                            ? e(i, j, k) - ambient_energy
                            : Real(0)};
            });
        }
    }
    Real result = amrex::get<0>(reduce_data.value(reduce_op));
    ParallelDescriptor::ReduceRealMax(result);
    return result;
}

} // namespace

FrontResult
run_limited_front ()
{
    Array<int, AMREX_SPACEDIM> const nonperiodic{
        AMREX_D_DECL(0, 0, 0)};
    DiffusionHierarchy hierarchy =
        make_uniform_hierarchy(64, 16, nonperiodic);
    auto masks = make_composite_masks(hierarchy);
    auto state = make_cell_data(hierarchy, 1, 1);
    auto old_state = make_cell_data(hierarchy, 1, 1);
    auto solution = make_cell_data(hierarchy, 1, 1);
    auto rhs = make_cell_data(hierarchy, 1, 0);
    auto acoef = make_cell_data(hierarchy, 1, 0);
    auto extinction = make_cell_data(hierarchy, 1, 1);
    auto diffusion = make_cell_data(hierarchy, 1, 1);
    auto bcoef = make_face_data(hierarchy);

    Real constexpr source_radius = Real(0.1);
    Real constexpr ambient_energy = Real(1.e-4);
    Real constexpr pi = Real(3.1415926535897932384626433832795);
    auto const dx = hierarchy.geom[0].CellSizeArray();
    auto const problo = hierarchy.geom[0].ProbLoArray();
    for (MFIter mfi(*state[0]); mfi.isValid(); ++mfi) {
        auto const e = state[0]->array(mfi);
        ParallelFor(mfi.validbox(),
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            Real const x = problo[0] + (Real(i) + Real(0.5)) * dx[0] -
                           Real(0.5);
            Real const y = problo[1] + (Real(j) + Real(0.5)) * dx[1] -
                           Real(0.5);
            Real const radius = std::sqrt(x * x + y * y);
            e(i, j, k) =
                radius < source_radius
                    ? ambient_energy +
                          (Real(1) - ambient_energy) * Real(0.5) *
                              (Real(1) + std::cos(pi * radius / source_radius))
                    : ambient_energy;
        });
    }
    set_level_data(acoef, Real(1));
    set_level_data(extinction, Real(0.01));

    PhysicalBoundaryData boundary;
    boundary.lo = {AMREX_D_DECL(LinOpBCType::Neumann,
                                 LinOpBCType::Neumann,
                                 LinOpBCType::Neumann)};
    boundary.hi = boundary.lo;
    auto const neumann = boundary.lo;
    Real constexpr final_time = Real(0.15);
    int constexpr steps = 48;
    Real constexpr dt = final_time / Real(steps);
    Real constexpr relaxation = Real(0.7);
    Real const picard_tolerance =
        (sizeof(Real) == sizeof(float)) ? Real(2.e-4) : Real(2.e-5);
    int constexpr maximum_picard_iterations = 75;

    MLABecLapAMG solver(hierarchy.geom, hierarchy.grids, hierarchy.dmap);
    FrontResult result;
    result.cells = composite_cell_count(masks);
    for (int step = 0; step < steps; ++step) {
        copy_level_data(old_state, state);
        copy_level_data(rhs, old_state);
        bool converged = false;
        int step_iterations = 0;
        for (int iteration = 0; iteration < maximum_picard_iterations;
             ++iteration) {
            compute_diffusion(hierarchy, state, extinction, diffusion,
                              boundary, true);
            fill_face_coefficients(hierarchy, diffusion, nullptr, bcoef,
                                   false);
            solver.setup(Real(1), dt, get_level_const_ptrs(acoef),
                         get_face_const_ptrs(bcoef), neumann, neumann, {});
            record_setup(result.solver, solver);
            set_level_data(solution, Real(0));
            auto const info = solver.solve(get_level_ptrs(solution),
                                           get_level_const_ptrs(rhs),
                                           linear_tolerance(), Real(0));
            record_solve(result.solver, info);
            result.final_picard_change = composite_maximum_relative_change(
                solution, state, masks);
            ++step_iterations;
            ++result.total_picard_iterations;
            if (result.final_picard_change <= picard_tolerance) {
                copy_level_data(state, solution);
                converged = true;
                break;
            }
            lincomb_level_data(state, relaxation, solution,
                               Real(1) - relaxation, state);
        }
        result.maximum_picard_iterations =
            amrex::max(result.maximum_picard_iterations, step_iterations);
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            converged,
            "The limited-front FLD Picard iteration did not converge");
    }

    compute_diffusion(hierarchy, state, extinction, diffusion, boundary, true,
                      &result.maximum_flux_fraction);
    Real const finest_dx = hierarchy.geom.back().CellSize(0);
    result.causal_radius = source_radius + final_time;
    result.front_radius =
        front_radius(hierarchy, state, masks, Real(0.01), source_radius);
    result.far_excess = far_excess(
        hierarchy, state, masks,
        result.causal_radius + Real(2) * finest_dx, ambient_energy);
    auto const [minimum, maximum] =
        composite_minimum_maximum(state, masks);
    result.minimum_energy = minimum;
    result.maximum_energy = maximum;

    auto unlimited_state = make_cell_data(hierarchy, 1, 1);
    auto unlimited_solution = make_cell_data(hierarchy, 1, 1);
    auto unlimited_rhs = make_cell_data(hierarchy, 1, 0);
    auto unlimited_diffusion = make_cell_data(hierarchy, 1, 1);
    auto unlimited_bcoef = make_face_data(hierarchy);
    for (MFIter mfi(*unlimited_state[0]); mfi.isValid(); ++mfi) {
        auto const e = unlimited_state[0]->array(mfi);
        ParallelFor(mfi.validbox(),
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            Real const x = problo[0] + (Real(i) + Real(0.5)) * dx[0] -
                           Real(0.5);
            Real const y = problo[1] + (Real(j) + Real(0.5)) * dx[1] -
                           Real(0.5);
            Real const radius = std::sqrt(x * x + y * y);
            e(i, j, k) =
                radius < source_radius
                    ? ambient_energy +
                          (Real(1) - ambient_energy) * Real(0.5) *
                              (Real(1) + std::cos(pi * radius / source_radius))
                    : ambient_energy;
        });
    }
    set_level_data(unlimited_diffusion, Real(1) / Real(0.03));
    fill_face_coefficients(hierarchy, unlimited_diffusion, nullptr,
                           unlimited_bcoef, false);
    MLABecLapAMG unlimited_solver(hierarchy.geom, hierarchy.grids,
                                  hierarchy.dmap);
    unlimited_solver.setup(Real(1), dt, get_level_const_ptrs(acoef),
                           get_face_const_ptrs(unlimited_bcoef), neumann,
                           neumann, {});
    for (int step = 0; step < steps; ++step) {
        copy_level_data(unlimited_rhs, unlimited_state);
        set_level_data(unlimited_solution, Real(0));
        auto const info = unlimited_solver.solve(
            get_level_ptrs(unlimited_solution),
            get_level_const_ptrs(unlimited_rhs), linear_tolerance(), Real(0));
        amrex::ignore_unused(info);
        copy_level_data(unlimited_state, unlimited_solution);
    }
    result.unlimited_far_excess = far_excess(
        hierarchy, unlimited_state, masks,
        result.causal_radius + Real(2) * finest_dx, ambient_energy);

    Real const causality_tolerance =
        (sizeof(Real) == sizeof(float)) ? Real(2.e-4) : Real(2.e-8);
    AMREX_ALWAYS_ASSERT(result.maximum_flux_fraction <=
                        Real(1) + causality_tolerance);
    AMREX_ALWAYS_ASSERT(result.maximum_flux_fraction > Real(0.95));
    AMREX_ALWAYS_ASSERT(result.front_radius <=
                        result.causal_radius + Real(2) * finest_dx);
    AMREX_ALWAYS_ASSERT(result.front_radius >=
                        result.causal_radius - Real(5) * finest_dx);
    AMREX_ALWAYS_ASSERT(result.minimum_energy >= ambient_energy - Real(2.e-5));
    AMREX_ALWAYS_ASSERT(result.maximum_energy <= Real(1) + Real(2.e-5));
    AMREX_ALWAYS_ASSERT(result.far_excess < Real(0.015));
    AMREX_ALWAYS_ASSERT(result.unlimited_far_excess > Real(0.005));
    AMREX_ALWAYS_ASSERT(result.unlimited_far_excess >
                        Real(5) * result.far_excess);
    return result;
}

} // namespace fld_test
