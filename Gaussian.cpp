#include "FLDTestCommon.H"

namespace fld_test
{

GaussianResult
run_gaussian (bool use_amr)
{
    auto periodic = periodic_boundary();
    std::array<BoundaryCondition, 4> boundary{periodic, periodic, periodic,
                                              periodic};

    int const nbase = use_amr ? 32 : 64;
    int const ratio = use_amr ? 2 : 1;
    Mesh mesh = make_mesh(
        nbase, ratio, [] (int i, int j, int n) noexcept
        { return i >= n / 4 && i < 3 * n / 4 && j >= n / 4 && j < 3 * n / 4; },
        [] (Real, Real) noexcept { return true; }, boundary);

    Real constexpr extinction_value = Real(100);
    Real const diffusion_value = Real(1) / (Real(3) * extinction_value);
    Real constexpr initial_time = Real(0.4);
    Real constexpr dt = Real(0.005);
    int constexpr steps = 10;
    Real constexpr pi = Real(3.1415926535897932384626433832795);

    auto exact = [&] (Cell const& cell, Real time) -> Real
    {
        Real const x = cell.x - Real(0.5);
        Real const y = cell.y - Real(0.5);
        Real const radius_squared = x * x + y * y;
        return std::exp(-radius_squared / (Real(4) * diffusion_value * time)) /
               (Real(4) * pi * diffusion_value * time);
    };

    Vector<Real> state(mesh.cells.size());
    Vector<Real> extinction(mesh.cells.size(), extinction_value);
    for (Long row = 0; row < static_cast<Long>(mesh.cells.size()); ++row) {
        state[row] = exact(mesh.cells[row], initial_time);
    }
    Real const initial_energy = std::inner_product(
        state.begin(), state.end(), mesh.cells.begin(), Real(0), std::plus<>(),
        [] (Real energy, Cell const& cell) { return energy * cell.volume; });

    auto diffusion = compute_diffusion(mesh, state, extinction, false);
    auto system = assemble_system(mesh, diffusion, state, dt, true);
    AMGGMRESSolver solver(system.matrix);
    GaussianResult result;
    result.cells = static_cast<Long>(mesh.cells.size());
    record_setup(result.solver, solver);

    for (int step = 0; step < steps; ++step) {
        auto solution = solver.solve(state);
        record_solve(result.solver, solution);
        state = std::move(solution.values);
    }

    Real const final_time = initial_time + Real(steps) * dt;
    Real absolute_error = Real(0);
    Real exact_norm = Real(0);
    Real final_energy = Real(0);
    for (Long row = 0; row < static_cast<Long>(mesh.cells.size()); ++row) {
        Real const reference = exact(mesh.cells[row], final_time);
        Real const volume = mesh.cells[row].volume;
        absolute_error += volume * std::abs(state[row] - reference);
        exact_norm += volume * std::abs(reference);
        final_energy += volume * state[row];
    }
    result.relative_l1_error = absolute_error / exact_norm;
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

