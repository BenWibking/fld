#include "FLDTest.H"
#include "RadiationMaterial.H"

#include <AMReX_MFIter.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Reduce.H>

#include <cmath>
#include <iomanip>
#include <string>

// Moving-medium test problems for RadiationMaterialSolver on the periodic
// unit cube with a prescribed uniform velocity (vc.velocity, in units of
// c), to O(v/c): an advecting equilibrium pulse (vc.test=pulse) and a steady
// manufactured solution (vc.test=mms).  Opacity sigma = vc.sigma0 *
// T^-vc.opacity_power.  vc.solver selects newton_krylov (default),
// ne_newton_krylov, or ne_quasi_newton.

namespace fld_test
{

using namespace amrex;

namespace
{

Real constexpr pi = Real(3.14159265358979323846);

struct MovingSettings
{
    RadiationMaterialOptions options;
    Real sigma0 = Real(1000);
};

// Defaults: optically thick (tau ~ 1e3-1e4 across the box).  Optically thin,
// limiter-dominated cases need the predictor fallback.
MovingSettings
read_settings (ParmParse& pp)
{
    MovingSettings settings;
    auto& options = settings.options;
    pp.query("sigma0", settings.sigma0);
    pp.query("opacity_power", options.opacity_power);
    Vector<Real> velocity{Real(1.e-3), Real(0), Real(0)};
    Vector<Real> requested;
    if (pp.queryarr("velocity", requested)) { velocity = requested; }
    AMREX_ALWAYS_ASSERT(velocity.size() == 3);
    for (int d = 0; d < 3; ++d) { options.velocity[d] = velocity[d]; }
    int limiter = 1;
    pp.query("limiter", limiter);
    options.limiter = limiter != 0;
    std::string advection = "upwind";
    pp.query("advection", advection);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        advection == "upwind" || advection == "centered",
        "vc.advection must be upwind or centered");
    options.upwind = advection == "upwind";
    for (auto& boundary : options.boundary) {
        boundary = RadiationBoundary::Periodic;
    }
    pp.query("solver", options.solver);
    pp.query("nonlinear_tolerance", options.nonlinear_tolerance);
    pp.query("max_lag_iterations", options.maximum_lag_iterations);
    return settings;
}

LevelData
uniform_field (DiffusionHierarchy const& hierarchy, Real value)
{
    auto field = make_cell_data(hierarchy, 1, 0);
    set_level_data(field, value);
    return field;
}

void
print_counters (std::string const& solver_name,
                RadiationMaterialCounters const& c, double wall)
{
    amrex::Print()
        << "solver=" << solver_name << ", Newton iterations total/max="
        << c.nonlinear << "/" << c.maximum_nonlinear
        << ", Krylov iterations=" << c.krylov
        << ", lag iterations total/max=" << c.lag_iterations << "/"
        << c.maximum_lag_iterations
        << ", predictor fallbacks=" << c.predictor_fallbacks
        << ", eliminations=" << c.eliminations
        << ", max local Newton iterations=" << c.maximum_local_iterations
        << ", AMG setups=" << c.summary.setups
        << ", max coupled residual=" << c.maximum_coupled_residual
        << ", wall=" << wall << " s\n";
}

// ---------------------------------------------------------------------
// Advecting pulse: an equilibrium Gaussian temperature pulse.

void
run_pulse (ParmParse& pp)
{
    int n_cell = 32;
    int steps = 50;
    Real dt = Real(2);
    Real background = Real(0.5);
    Real amplitude = Real(0.5);
    Real width = Real(0.1);
    std::string plotfile;
    pp.query("n_cell", n_cell);
    pp.query("steps", steps);
    pp.query("dt", dt);
    pp.query("background_temperature", background);
    pp.query("pulse_amplitude", amplitude);
    pp.query("pulse_width", width);
    pp.query("plotfile", plotfile);

    auto const hierarchy = make_uniform_hierarchy(
        n_cell, amrex::min(n_cell, 32), {AMREX_D_DECL(1, 1, 1)});
    MovingSettings const settings = read_settings(pp);
    RadiationMaterialSolver solver(hierarchy, settings.options);
    solver.set_opacity_coefficient(uniform_field(hierarchy, settings.sigma0));
    auto const& masks = solver.masks();

    RadiationMaterialState state = solver.make_state();
    auto const dx = hierarchy.geom[0].CellSizeArray();
    for (MFIter mfi(*state.energy[0]); mfi.isValid(); ++mfi) {
        auto const e = state.energy[0]->array(mfi);
        auto const t = state.temperature[0]->array(mfi);
        ParallelFor(mfi.validbox(),
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            Real const x = (Real(i) + Real(0.5)) * dx[0] - Real(0.5);
            Real const y = (Real(j) + Real(0.5)) * dx[1] - Real(0.5);
            Real const z = (Real(k) + Real(0.5)) * dx[2] - Real(0.5);
            Real const temperature = background +
                amplitude * std::exp(-(x * x + y * y + z * z) / (width * width));
            Real const t2 = temperature * temperature;
            t(i, j, k) = temperature;
            e(i, j, k) = t2 * t2;
        });
    }

    // Circular mean of the x position of the excess radiation energy.
    auto centroid = [&] () {
        ReduceOps<ReduceOpSum, ReduceOpSum> reduce_op;
        ReduceData<Real, Real> reduce_data(reduce_op);
        using Tuple = typename decltype(reduce_data)::Type;
        Real const e_background = background * background * background *
                                  background;
        for (MFIter mfi(*state.energy[0]); mfi.isValid(); ++mfi) {
            auto const e = state.energy[0]->const_array(mfi);
            reduce_op.eval(mfi.validbox(), reduce_data,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept -> Tuple
            {
                Real const w = amrex::max(e(i, j, k) - e_background, Real(0));
                Real const angle = Real(2) * pi * (Real(i) + Real(0.5)) * dx[0];
                return {w * std::cos(angle), w * std::sin(angle)};
            });
        }
        auto const sums = reduce_data.value(reduce_op);
        Real r[2] = {amrex::get<0>(sums), amrex::get<1>(sums)};
        ParallelDescriptor::ReduceRealSum(r, 2);
        Real angle = std::atan2(r[1], r[0]);
        if (angle < Real(0)) { angle += Real(2) * pi; }
        return angle / (Real(2) * pi);
    };

    Real const x0 = centroid();
    Real maximum_balance_error = Real(0);
    double const start = amrex::second();
    for (int step = 0; step < steps; ++step) {
        Real const old_total = solver.energy_budget(state).total;
        solver.advance(state, dt);
        // Material advection conserves the total on the periodic domain, so
        // it changes only by the work done on the flow.
        auto const budget = solver.energy_budget(state);
        Real const expected =
            dt * (budget.boundary_input - budget.work + budget.source);
        Real const error = std::abs(budget.total - old_total - expected) /
            amrex::max(std::abs(old_total), Real(1.e-300));
        maximum_balance_error = amrex::max(maximum_balance_error, error);
    }
    double wall = amrex::second() - start;
    ParallelDescriptor::ReduceRealMax(wall);

    Real const time = Real(steps) * dt;
    Real displacement = centroid() - x0;
    displacement -= std::round(displacement);
    auto const& velocity = settings.options.velocity;
    amrex::Print() << std::setprecision(10)
                   << "Moving-medium pulse: n=" << n_cell << ", steps="
                   << steps << ", dt=" << dt << ", sigma0=" << settings.sigma0
                   << ", v=(" << velocity[0] << "," << velocity[1] << ","
                   << velocity[2] << "), v*tau="
                   << velocity[0] * settings.sigma0
                   << ", centroid shift=" << displacement
                   << ", v*t=" << velocity[0] * time
                   << ", max energy-balance error=" << maximum_balance_error
                   << '\n';
    print_counters(settings.options.solver, solver.counters(), wall);
    if (!plotfile.empty()) {
        auto plot = make_cell_data(hierarchy, 2, 0);
        MultiFab::Copy(*plot[0], *state.energy[0], 0, 0, 1, 0);
        MultiFab::Copy(*plot[0], *state.temperature[0], 0, 1, 1, 0);
        WriteMultiLevelPlotfile(plotfile, 1, get_level_const_ptrs(plot),
                                {"radiation_energy", "material_temperature"},
                                hierarchy.geom, time, {0}, {});
    }
    amrex::ignore_unused(masks);
}

// ---------------------------------------------------------------------
// Manufactured solution: steady periodic E and T with uniform velocity,
// no limiter (D = 1/(3 sigma), P = E/3).  One implicit step from the exact
// state, so the time derivative vanishes at the exact solution and the
// error measures the spatial discretization.

struct MMSValues
{
    Real energy;
    Real temperature;
    Real source_energy;
    Real source_temperature;
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE MMSValues
mms_values (Real sigma0, Real power, GpuArray<Real, 3> const& v, Real x,
            Real y, Real z) noexcept
{
    Real const w = Real(2) * pi;
    Real const sx = std::sin(w * x), cx = std::cos(w * x);
    Real const sy = std::sin(w * y), cy = std::cos(w * y);
    Real const sz = std::sin(w * z), cz = std::cos(w * z);
    Real const e = Real(1) + Real(0.3) * sx * cy + Real(0.2) * sz;
    Real const ge[3] = {Real(0.3) * w * cx * cy, Real(-0.3) * w * sx * sy,
                        Real(0.2) * w * cz};
    Real const lap_e = Real(-0.3) * Real(2) * w * w * sx * cy -
                       Real(0.2) * w * w * sz;
    Real const t = Real(1) + Real(0.1) * cx * sz + Real(0.15) * sy;
    Real const gt[3] = {Real(-0.1) * w * sx * sz, Real(0.15) * w * cy,
                        Real(0.1) * w * cx * cz};
    Real const sigma = sigma0 * std::pow(t, -power);
    Real const d = Real(1) / (Real(3) * sigma);
    // grad D = p T^(p-1) grad T / (3 sigma0)
    Real const dd = power * std::pow(t, power - Real(1)) / (Real(3) * sigma0);
    Real grad_d_dot_grad_e = Real(0);
    Real v_dot_ge = Real(0);
    Real v_dot_gt = Real(0);
    Real v_dot_f0 = Real(0);
    for (int i = 0; i < 3; ++i) {
        grad_d_dot_grad_e += dd * gt[i] * ge[i];
        v_dot_ge += v[i] * ge[i];
        v_dot_gt += v[i] * gt[i];
        v_dot_f0 += v[i] * (-d * ge[i]);
    }
    Real const div_f0 = -(grad_d_dot_grad_e + d * lap_e);
    Real const t2 = t * t;
    Real const t4 = t2 * t2;
    Real const se = div_f0 + Real(4) / Real(3) * v_dot_ge -
                    sigma * (t4 - e) - sigma * v_dot_f0;
    Real const st = v_dot_gt - sigma * (e - t4) + Real(2) * sigma * v_dot_f0;
    return {e, t, se, st};
}

void
run_mms (ParmParse& pp)
{
    Vector<int> n_cells{16, 32, 64};
    Vector<int> requested;
    if (pp.queryarr("mms_n_cells", requested)) { n_cells = requested; }
    Real dt = Real(0.1);
    pp.query("mms_dt", dt);
    std::string advection = "upwind";
    pp.query("advection", advection);

    std::string solver_name;
    Vector<Array<Real, 4>> errors; // E max, E L2, T max, T L2
    for (int n : n_cells) {
        auto const hierarchy = make_uniform_hierarchy(
            n, amrex::min(n, 32), {AMREX_D_DECL(1, 1, 1)});
        MovingSettings const settings = read_settings(pp);
        solver_name = settings.options.solver;
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            !settings.options.limiter,
            "The moving-medium MMS requires vc.limiter=0");
        RadiationMaterialSolver solver(hierarchy, settings.options);
        solver.set_opacity_coefficient(
            uniform_field(hierarchy, settings.sigma0));
        auto const& masks = solver.masks();
        RadiationMaterialState state = solver.make_state();
        auto exact_e = make_cell_data(hierarchy, 1, 0);
        auto exact_t = make_cell_data(hierarchy, 1, 0);
        auto source_e = make_cell_data(hierarchy, 1, 0);
        auto source_t = make_cell_data(hierarchy, 1, 0);
        auto const dx = hierarchy.geom[0].CellSizeArray();
        Real const sigma0 = settings.sigma0;
        Real const power = settings.options.opacity_power;
        GpuArray<Real, 3> const v{settings.options.velocity[0],
                                  settings.options.velocity[1],
                                  settings.options.velocity[2]};
        for (MFIter mfi(*exact_e[0]); mfi.isValid(); ++mfi) {
            auto const ee = exact_e[0]->array(mfi);
            auto const et = exact_t[0]->array(mfi);
            auto const se = source_e[0]->array(mfi);
            auto const st = source_t[0]->array(mfi);
            ParallelFor(mfi.validbox(),
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                auto const m = mms_values(sigma0, power, v,
                                          (Real(i) + Real(0.5)) * dx[0],
                                          (Real(j) + Real(0.5)) * dx[1],
                                          (Real(k) + Real(0.5)) * dx[2]);
                ee(i, j, k) = m.energy;
                et(i, j, k) = m.temperature;
                se(i, j, k) = m.source_energy;
                st(i, j, k) = m.source_temperature;
            });
        }
        solver.set_sources(source_e, source_t);
        copy_level_data(state.energy, exact_e);
        copy_level_data(state.temperature, exact_t);
        double const start = amrex::second();
        solver.advance(state, dt);
        double wall = amrex::second() - start;
        ParallelDescriptor::ReduceRealMax(wall);
        Array<Real, 4> err{};
        LevelData* fields[2] = {&state.energy, &state.temperature};
        LevelData const* exact[2] = {&exact_e, &exact_t};
        for (int f = 0; f < 2; ++f) {
            auto diff = make_cell_data(hierarchy, 1, 0);
            lincomb_level_data(diff, Real(1), *fields[f], Real(-1), *exact[f]);
            auto const [minimum, maximum] =
                composite_minimum_maximum(diff, masks);
            err[2 * f] = amrex::max(std::abs(minimum), std::abs(maximum));
            err[2 * f + 1] = std::sqrt(
                composite_weighted_dot(diff, diff, hierarchy, masks));
        }
        errors.push_back(err);
        amrex::Print() << "MMS n=" << n << ": ";
        print_counters(solver_name, solver.counters(), wall);
    }
    amrex::Print() << "\nMoving-medium MMS, solver=" << solver_name
                   << ", advection=" << advection
                   << "\n     n     E max   ord      E L2   ord     T max   ord"
                      "      T L2   ord\n";
    for (int r = 0; r < static_cast<int>(errors.size()); ++r) {
        amrex::Print() << std::setw(6) << n_cells[r];
        for (int q = 0; q < 4; ++q) {
            Real const order = r > 0
                ? std::log2(errors[r - 1][q] / errors[r][q]) : Real(0);
            amrex::Print() << std::scientific << std::setprecision(3)
                           << std::setw(10) << errors[r][q] << std::fixed
                           << std::setprecision(2) << std::setw(6) << order;
        }
        amrex::Print() << '\n';
    }
}

} // namespace

void
run_moving_medium ()
{
    ParmParse pp("vc");
    std::string test = "pulse";
    pp.query("test", test);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(test == "pulse" || test == "mms",
                                     "vc.test must be pulse or mms");
    if (test == "pulse") {
        run_pulse(pp);
    } else {
        run_mms(pp);
    }
}

} // namespace fld_test
