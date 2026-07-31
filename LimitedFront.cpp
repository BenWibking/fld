#include "FLDTestCommon.H"

namespace fld_test
{

Mesh
make_front_mesh ()
{
    std::array<BoundaryCondition, 4> boundary{
        reflecting_boundary(), reflecting_boundary(), reflecting_boundary(),
        reflecting_boundary()};
    Real constexpr inner_radius = Real(0.1);
    return make_mesh(
        64, 1, [] (int, int, int) noexcept { return false; },
        [] (Real x, Real y) noexcept
        {
            Real const dx = x - Real(0.5);
            Real const dy = y - Real(0.5);
            return dx * dx + dy * dy > inner_radius * inner_radius;
        },
        boundary, true, dirichlet_boundary(Real(1)));
}

Real
far_excess (Mesh const& mesh, Vector<Real> const& state, Real radius,
            Real ambient_energy)
{
    Real result = Real(0);
    for (Long row = 0; row < static_cast<Long>(mesh.cells.size()); ++row) {
        Real const dx = mesh.cells[row].x - Real(0.5);
        Real const dy = mesh.cells[row].y - Real(0.5);
        if (std::hypot(dx, dy) > radius) {
            result = amrex::max(result, state[row] - ambient_energy);
        }
    }
    return result;
}

Real
front_radius (Mesh const& mesh, Vector<Real> const& state)
{
    Real result = Real(0.1);
    Real constexpr threshold = Real(0.01);
    for (Long row = 0; row < static_cast<Long>(mesh.cells.size()); ++row) {
        if (state[row] > threshold) {
            Real const dx = mesh.cells[row].x - Real(0.5);
            Real const dy = mesh.cells[row].y - Real(0.5);
            result = amrex::max(result, std::hypot(dx, dy));
        }
    }
    return result;
}

FrontResult
run_limited_front ()
{
    Mesh mesh = make_front_mesh();
    Vector<Real> extinction(mesh.cells.size(), Real(0.01));
    Real constexpr ambient_energy = Real(1.e-4);
    Vector<Real> state(mesh.cells.size(), ambient_energy);
    Real constexpr final_time = Real(0.15);
    int constexpr steps = 48;
    Real constexpr dt = final_time / Real(steps);
    Real constexpr relaxation = Real(0.7);
    Real const picard_tolerance =
        (sizeof(Real) == sizeof(float)) ? Real(2.e-4) : Real(2.e-5);
    int constexpr maximum_picard_iterations = 75;

    FrontResult result;
    result.cells = static_cast<Long>(mesh.cells.size());
    for (int step = 0; step < steps; ++step) {
        Vector<Real> const old_state = state;
        bool converged = false;
        int step_iterations = 0;
        for (int iteration = 0; iteration < maximum_picard_iterations;
             ++iteration) {
            auto diffusion = compute_diffusion(mesh, state, extinction, true);
            auto system = assemble_system(mesh, diffusion, old_state, dt, true);
            AMGGMRESSolver solver(system.matrix);
            record_setup(result.solver, solver);
            auto solution = solver.solve(system.rhs);
            record_solve(result.solver, solution);

            result.final_picard_change =
                maximum_relative_change(solution.values, state);
            ++step_iterations;
            ++result.total_picard_iterations;
            if (result.final_picard_change <= picard_tolerance) {
                state = std::move(solution.values);
                converged = true;
                break;
            }
            for (std::size_t row = 0; row < state.size(); ++row) {
                state[row] = relaxation * solution.values[row] +
                             (Real(1) - relaxation) * state[row];
            }
        }
        result.maximum_picard_iterations =
            amrex::max(result.maximum_picard_iterations, step_iterations);
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            converged,
            "The limited-front FLD Picard iteration did not converge");
    }

    compute_diffusion(mesh, state, extinction, true,
                      &result.maximum_flux_fraction);
    result.causal_radius = Real(0.1) + final_time;
    result.front_radius = front_radius(mesh, state);
    result.far_excess =
        far_excess(mesh, state, result.causal_radius + Real(2) * mesh.fine_h,
                   ambient_energy);
    auto const [minimum, maximum] = minimum_maximum(state);
    result.minimum_energy = minimum;
    result.maximum_energy = maximum;

    Vector<Real> unlimited_state(mesh.cells.size(), ambient_energy);
    auto unlimited_diffusion =
        compute_diffusion(mesh, unlimited_state, extinction, false);
    auto unlimited_system =
        assemble_system(mesh, unlimited_diffusion, unlimited_state, dt, true);
    AMGGMRESSolver unlimited_solver(unlimited_system.matrix);
    Vector<Real> unlimited_boundary_rhs(mesh.cells.size());
    for (std::size_t row = 0; row < unlimited_state.size(); ++row) {
        unlimited_boundary_rhs[row] =
            unlimited_system.rhs[row] - unlimited_state[row];
    }
    for (int step = 0; step < steps; ++step) {
        Vector<Real> rhs = unlimited_state;
        for (std::size_t row = 0; row < rhs.size(); ++row) {
            rhs[row] += unlimited_boundary_rhs[row];
        }
        auto solution = unlimited_solver.solve(rhs);
        unlimited_state = std::move(solution.values);
    }
    result.unlimited_far_excess = far_excess(
        mesh, unlimited_state, result.causal_radius + Real(2) * mesh.fine_h,
        ambient_energy);

    Real const causality_tolerance =
        (sizeof(Real) == sizeof(float)) ? Real(2.e-4) : Real(2.e-8);
    AMREX_ALWAYS_ASSERT(result.maximum_flux_fraction <=
                        Real(1) + causality_tolerance);
    AMREX_ALWAYS_ASSERT(result.maximum_flux_fraction > Real(0.95));
    AMREX_ALWAYS_ASSERT(result.front_radius <=
                        result.causal_radius + Real(2) * mesh.fine_h);
    AMREX_ALWAYS_ASSERT(result.front_radius >=
                        result.causal_radius - Real(4) * mesh.fine_h);
    AMREX_ALWAYS_ASSERT(result.minimum_energy >= ambient_energy - Real(2.e-5));
    AMREX_ALWAYS_ASSERT(result.maximum_energy <= Real(1) + Real(2.e-5));
    AMREX_ALWAYS_ASSERT(result.far_excess < Real(0.015));
    AMREX_ALWAYS_ASSERT(result.unlimited_far_excess > Real(0.04));
    return result;
}

} // namespace fld_test

