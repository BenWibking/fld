#include "FLDTest.H"
#include "RadiationMaterial.H"

#include <AMReX_MFIter.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Reduce.H>

#include <cmath>
#include <string>

namespace fld_test
{

using namespace amrex;

namespace
{

Real constexpr initial_radiation_energy = Real(1.e-5);

// E, T, and the atomic number Z: Z = 10 in the rectangular prism spanning
// the middle third of x and y and the middle two-thirds of z, 1 elsewhere.
void
initialize_fields (DiffusionHierarchy const& hierarchy,
                   RadiationMaterialState& state, LevelData& atomic_number)
{
    Real const initial_temperature =
        std::sqrt(std::sqrt(initial_radiation_energy));
    auto const dx = hierarchy.geom[0].CellSizeArray();
    auto const problo = hierarchy.geom[0].ProbLoArray();
    for (MFIter mfi(*state.energy[0]); mfi.isValid(); ++mfi) {
        auto const e = state.energy[0]->array(mfi);
        auto const t = state.temperature[0]->array(mfi);
        auto const z = atomic_number[0]->array(mfi);
        ParallelFor(mfi.validbox(),
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            Real const x = problo[0] + (Real(i) + Real(0.5)) * dx[0];
            Real const y = problo[1] + (Real(j) + Real(0.5)) * dx[1];
            Real const zc = problo[2] + (Real(k) + Real(0.5)) * dx[2];
            bool const inside = x > Real(1) / Real(3) &&
                                x < Real(2) / Real(3) &&
                                y > Real(1) / Real(3) &&
                                y < Real(2) / Real(3) &&
                                zc > Real(1) / Real(6) &&
                                zc < Real(5) / Real(6);
            e(i, j, k) = initial_radiation_energy;
            t(i, j, k) = initial_temperature;
            z(i, j, k) = inside ? Real(10) : Real(1);
        });
    }
}

Long
count_high_z_cells (LevelData const& atomic_number,
                    Vector<iMultiFab> const& masks)
{
    ReduceOps<ReduceOpSum> reduce_op;
    ReduceData<Long> reduce_data(reduce_op);
    using Tuple = typename decltype(reduce_data)::Type;
    for (MFIter mfi(*atomic_number[0]); mfi.isValid(); ++mfi) {
        auto const z = atomic_number[0]->const_array(mfi);
        auto const mask = masks[0].const_array(mfi);
        reduce_op.eval(mfi.validbox(), reduce_data,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept -> Tuple
        {
            return {mask(i,j,k) && z(i,j,k) > Real(1) ? Long(1) : Long(0)};
        });
    }
    Long result = amrex::get<0>(reduce_data.value(reduce_op));
    ParallelDescriptor::ReduceLongSum(result);
    return result;
}

void
write_icase_plotfile (std::string const& name,
                      DiffusionHierarchy const& hierarchy,
                      RadiationMaterialSolver& solver,
                      RadiationMaterialState const& state,
                      LevelData const& atomic_number, Real time)
{
    if (name.empty()) {
        return;
    }
    auto sigma = make_cell_data(hierarchy, 1, 0);
    auto diffusion = make_cell_data(hierarchy, 1, 0);
    solver.fill_cell_coefficients(state, sigma, diffusion);
    auto plot = make_cell_data(hierarchy, 5, 0);
    MultiFab::Copy(*plot[0], *state.energy[0], 0, 0, 1, 0);
    MultiFab::Copy(*plot[0], *state.temperature[0], 0, 1, 1, 0);
    MultiFab::Copy(*plot[0], *atomic_number[0], 0, 2, 1, 0);
    MultiFab::Copy(*plot[0], *sigma[0], 0, 3, 1, 0);
    MultiFab::Copy(*plot[0], *diffusion[0], 0, 4, 1, 0);
    Vector<std::string> const variables{
        "radiation_energy", "material_temperature", "atomic_number",
        "absorption", "radiation_diffusion"};
    WriteMultiLevelPlotfile(name, 1, get_level_const_ptrs(plot), variables,
                            hierarchy.geom, time, Vector<int>(1, 0),
                            hierarchy.ref_ratio);
    amrex::Print() << "Wrote ICASE 2001-12 plotfile " << name << '\n';
}

} // namespace

// ICASE 2001-12, Section 4, extended to the unit cube: radiation enters at
// the x-low Marshak face and leaves at x-high; the y and z faces are
// insulated.  sigma = Z^3/T^3; no material conduction.
ICASE2001Result
run_icase_2001 (int n_cell, int time_steps, Real dt, bool iteration_output,
                std::string const& plotfile_name)
{
    static_assert(AMREX_SPACEDIM == 3);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        n_cell > 0 && n_cell % 6 == 0,
        "icase_n_cell must be positive and divisible by 6 so the "
        "material interfaces align with cell faces");
    AMREX_ALWAYS_ASSERT(time_steps > 0 && dt > Real(0));

    DiffusionHierarchy hierarchy =
        make_uniform_hierarchy(n_cell, 16, {AMREX_D_DECL(0, 0, 0)});

    RadiationMaterialOptions options;
    options.opacity_power = Real(3);
    options.solver = "ne_newton_krylov";
    options.nonlinear_tolerance =
        sizeof(Real) == sizeof(float) ? Real(3.e-4) : Real(2.e-7);
    options.maximum_nonlinear_iterations = 20;
    options.energy_scale_floor = initial_radiation_energy;
    options.temperature_scale_floor =
        std::sqrt(std::sqrt(initial_radiation_energy));
    // The Marshak inflow (incoming 1) bounds E by 4.
    options.maximum_energy = Real(8);
    for (auto& boundary : options.boundary) {
        boundary = RadiationBoundary::Neumann;
    }
    options.boundary[0] = RadiationBoundary::Robin;
    options.boundary[1] = RadiationBoundary::Robin;
    options.incoming[0] = Real(1);
    {
        ParmParse pp;
        pp.query("icase_solver", options.solver);
        pp.query("icase_quasi_newton_max_iter",
                 options.maximum_quasi_newton_iterations);
    }
    RadiationMaterialSolver solver(hierarchy, options);
    auto const& masks = solver.masks();

    RadiationMaterialState state = solver.make_state();
    auto atomic_number = make_cell_data(hierarchy, 1, 1);
    initialize_fields(hierarchy, state, atomic_number);
    auto kappa = make_cell_data(hierarchy, 1, 0);
    for (MFIter mfi(*kappa[0]); mfi.isValid(); ++mfi) {
        auto const z = atomic_number[0]->const_array(mfi);
        auto const k = kappa[0]->array(mfi);
        ParallelFor(mfi.validbox(),
        [=] AMREX_GPU_DEVICE (int i, int j, int kk) noexcept
        {
            k(i, j, kk) = z(i, j, kk) * z(i, j, kk) * z(i, j, kk);
        });
    }
    solver.set_opacity_coefficient(kappa);

    ICASE2001Result result;
    result.nonlinear_solver = options.solver;
    result.cells = composite_cell_count(masks);
    result.high_z_cells = count_high_z_cells(atomic_number, masks);
    result.initial_total_energy = solver.energy_budget(state).total;

    for (int step = 0; step < time_steps; ++step) {
        Real const old_total = solver.energy_budget(state).total;
        int const previous_iterations = solver.counters().nonlinear;
        solver.advance(state, dt);
        // The exchange source is equal and opposite in the E and T
        // equations, so the change in integral(E + T) must equal the net
        // radiation supplied through the two Marshak boundaries.
        auto const budget = solver.energy_budget(state);
        Real const energy_change = budget.total - old_total;
        Real const expected_change = dt * budget.boundary_input;
        Real const balance_error =
            std::abs(energy_change - expected_change) /
            amrex::max(amrex::max(std::abs(energy_change),
                                  std::abs(expected_change)),
                       Real(1.e-30));
        result.maximum_energy_balance_error = amrex::max(
            result.maximum_energy_balance_error, balance_error);
        if (iteration_output) {
            amrex::Print()
                << "ICASE 2001-12 step=" << step + 1 << "/" << time_steps
                << ", time=" << Real(step + 1) * dt
                << ", nonlinear iterations="
                << solver.counters().nonlinear - previous_iterations
                << ", change=" << solver.counters().final_change
                << ", energy-balance error=" << balance_error << '\n';
        }
    }

    auto const& counters = solver.counters();
    result.time_steps = time_steps;
    result.final_time = Real(time_steps) * dt;
    result.final_total_energy = solver.energy_budget(state).total;
    result.final_nonlinear_change = counters.final_change;
    result.maximum_coupled_residual = counters.maximum_coupled_residual;
    result.total_nonlinear_iterations = counters.nonlinear;
    result.maximum_nonlinear_iterations = counters.maximum_nonlinear;
    result.total_newton_krylov_iterations = counters.krylov;
    result.maximum_newton_krylov_iterations = counters.maximum_krylov;
    result.predictor_fallbacks = counters.predictor_fallbacks;
    result.total_local_solves = counters.eliminations;
    result.maximum_local_newton_iterations =
        counters.maximum_local_iterations;
    result.solver = counters.summary;
    auto const [minimum_energy, maximum_energy] =
        composite_minimum_maximum(state.energy, masks);
    auto const [minimum_temperature, maximum_temperature] =
        composite_minimum_maximum(state.temperature, masks);
    result.minimum_radiation_energy = minimum_energy;
    result.maximum_radiation_energy = maximum_energy;
    result.minimum_material_temperature = minimum_temperature;
    result.maximum_material_temperature = maximum_temperature;

    write_icase_plotfile(plotfile_name, hierarchy, solver, state,
                         atomic_number, result.final_time);

    Long const expected_high_z_cells =
        Long(n_cell / 3) * Long(n_cell / 3) * Long(2 * n_cell / 3);
    AMREX_ALWAYS_ASSERT(result.high_z_cells == expected_high_z_cells);
    AMREX_ALWAYS_ASSERT(composite_all_finite(state.energy, masks));
    AMREX_ALWAYS_ASSERT(composite_all_finite(state.temperature, masks));
    AMREX_ALWAYS_ASSERT(result.minimum_radiation_energy > Real(0));
    AMREX_ALWAYS_ASSERT(result.minimum_material_temperature > Real(0));
    AMREX_ALWAYS_ASSERT(result.final_total_energy >
                        result.initial_total_energy);
    Real const balance_tolerance =
        sizeof(Real) == sizeof(float) ? Real(2.e-2) : Real(2.e-4);
    AMREX_ALWAYS_ASSERT(result.maximum_energy_balance_error <
                        balance_tolerance);
    return result;
}

} // namespace fld_test
