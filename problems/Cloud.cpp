#include "FLDTest.H"
#include "RadiationMaterial.H"
#include "PenroseCloudCenters.H"

#include <AMReX_AsyncOut.H>
#include <AMReX_BLProfiler.H>
#include <AMReX_MFIter.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Reduce.H>

#include <cmath>

namespace fld_test
{

using namespace amrex;

namespace
{

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE Real
semicircle_area_primitive (Real x, Real center, Real radius)
{
    Real const normalized_offset = amrex::max(
        Real(-1), amrex::min(Real(1), (x - center) / radius));
    Real const normalized_height = std::sqrt(amrex::max(
        Real(0), (Real(1) - normalized_offset) *
                     (Real(1) + normalized_offset)));
    return Real(0.5) * radius * radius *
           (normalized_offset * normalized_height +
            std::asin(normalized_offset));
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE Real
circle_rectangle_intersection_area (Real center_x, Real center_y, Real radius,
                                    Real xlo, Real xhi, Real ylo, Real yhi)
{
    Real const nearest_x = amrex::max(xlo, amrex::min(xhi, center_x));
    Real const nearest_y = amrex::max(ylo, amrex::min(yhi, center_y));
    Real const nearest_dx = nearest_x - center_x;
    Real const nearest_dy = nearest_y - center_y;
    if (nearest_dx * nearest_dx + nearest_dy * nearest_dy >=
        radius * radius) {
        return Real(0);
    }

    bool rectangle_inside_circle = true;
    for (int ix = 0; ix < 2; ++ix) {
        Real const x = ix == 0 ? xlo : xhi;
        for (int iy = 0; iy < 2; ++iy) {
            Real const y = iy == 0 ? ylo : yhi;
            Real const dx = x - center_x;
            Real const dy = y - center_y;
            rectangle_inside_circle = rectangle_inside_circle &&
                                      dx * dx + dy * dy <= radius * radius;
        }
    }
    Real const rectangle_area = (xhi - xlo) * (yhi - ylo);
    if (rectangle_inside_circle) {
        return rectangle_area;
    }

    Real const integration_lo = amrex::max(xlo, center_x - radius);
    Real const integration_hi = amrex::min(xhi, center_x + radius);
    if (integration_lo >= integration_hi) {
        return Real(0);
    }
    GpuArray<Real, 6> breakpoints{};
    breakpoints[0] = integration_lo;
    breakpoints[1] = integration_hi;
    int breakpoint_count = 2;
    for (int iy = 0; iy < 2; ++iy) {
        Real const y = iy == 0 ? ylo : yhi;
        Real const offset = std::abs(y - center_y);
        if (offset < radius) {
            Real const dx = std::sqrt(radius * radius - offset * offset);
            for (int side = 0; side < 2; ++side) {
                Real const point = center_x + (side == 0 ? -dx : dx);
                if (point > integration_lo && point < integration_hi) {
                    breakpoints[breakpoint_count++] = point;
                }
            }
        }
    }
    for (int current = 1; current < breakpoint_count; ++current) {
        Real const value = breakpoints[current];
        int position = current;
        while (position > 0 && breakpoints[position - 1] > value) {
            breakpoints[position] = breakpoints[position - 1];
            --position;
        }
        breakpoints[position] = value;
    }

    Real area = Real(0);
    for (int segment = 1; segment < breakpoint_count; ++segment) {
        Real const a = breakpoints[segment - 1];
        Real const b = breakpoints[segment];
        if (a >= b) {
            continue;
        }
        Real const midpoint = Real(0.5) * (a + b);
        Real const midpoint_dx = midpoint - center_x;
        Real const half_height = std::sqrt(amrex::max(
            Real(0), radius * radius - midpoint_dx * midpoint_dx));
        Real const circle_lo = center_y - half_height;
        Real const circle_hi = center_y + half_height;
        if (circle_hi <= ylo || circle_lo >= yhi) {
            continue;
        }
        bool const upper_is_circle = circle_hi < yhi;
        bool const lower_is_circle = circle_lo > ylo;
        Real const constant = (upper_is_circle ? center_y : yhi) -
                              (lower_is_circle ? center_y : ylo);
        int const semicircle_coefficient =
            int(upper_is_circle) + int(lower_is_circle);
        area += constant * (b - a) +
                Real(semicircle_coefficient) *
                    (semicircle_area_primitive(b, center_x, radius) -
                     semicircle_area_primitive(a, center_x, radius));
    }
    return amrex::max(Real(0), amrex::min(rectangle_area, area));
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE Real
cloud_radius ()
{
    Real constexpr spacing = Real(1) / Real(8.5);
    return Real(0.5) * (spacing / Real(1.1));
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE Real
sphere_cell_fraction (Real x, Real y, Real z, Real hx, Real hy, Real hz,
                      GpuArray<Real, penrose_cloud_count> const& centers_x,
                      GpuArray<Real, penrose_cloud_count> const& centers_z)
{
    Real const radius = cloud_radius();
    Real const xlo = x - Real(0.5) * hx;
    Real const xhi = x + Real(0.5) * hx;
    Real const ylo = y - Real(0.5) * hy;
    Real const yhi = y + Real(0.5) * hy;
    Real const zlo = z - Real(0.5) * hz;
    Real const zhi = z + Real(0.5) * hz;
    Real const nearest_y = amrex::max(ylo, amrex::min(yhi, Real(0.5)));
    Real const dy = nearest_y - Real(0.5);
    GpuArray<Real, 4> const nodes{
        Real(0.1834346424956498), Real(0.5255324099163290),
        Real(0.7966664774136267), Real(0.9602898564975363)};
    GpuArray<Real, 4> const weights{
        Real(0.3626837833783620), Real(0.3137066458778873),
        Real(0.2223810344533745), Real(0.1012285362903763)};
    Real volume = Real(0);
    for (int cloud = 0; cloud < penrose_cloud_count; ++cloud) {
        Real const cx = centers_x[cloud];
        Real const cz = centers_z[cloud];
        Real const nearest_x = amrex::max(xlo, amrex::min(xhi, cx));
        Real const nearest_z = amrex::max(zlo, amrex::min(zhi, cz));
        Real const dx = nearest_x - cx;
        Real const dz = nearest_z - cz;
        if (dx * dx + dy * dy + dz * dz >= radius * radius) {
            continue;
        }
        Real const far_x = amrex::max(std::abs(xlo - cx), std::abs(xhi - cx));
        Real const far_y = amrex::max(std::abs(ylo - Real(0.5)),
                                      std::abs(yhi - Real(0.5)));
        Real const far_z = amrex::max(std::abs(zlo - cz), std::abs(zhi - cz));
        if (far_x * far_x + far_y * far_y + far_z * far_z <=
            radius * radius) {
            return Real(1);
        }
        Real const a = amrex::max(zlo, cz - radius);
        Real const b = amrex::min(zhi, cz + radius);
        if (a >= b) {
            continue;
        }
        // Integrate analytic circle/rectangle areas across four z panels.
        for (int panel = 0; panel < 4; ++panel) {
            Real const left = a + (b - a) * Real(panel) / Real(4);
            Real const right = a + (b - a) * Real(panel + 1) / Real(4);
            Real const center = Real(0.5) * (left + right);
            Real const half = Real(0.5) * (right - left);
            for (int point = 0; point < 4; ++point) {
                for (int side = 0; side < 2; ++side) {
                    Real const zz = center + (side == 0 ? -half : half) *
                                                  nodes[point];
                    Real const section_radius = std::sqrt(amrex::max(
                        Real(0), radius * radius - (zz - cz) * (zz - cz)));
                    if (section_radius > Real(0)) {
                        volume += half * weights[point] *
                            circle_rectangle_intersection_area(
                                cx, Real(0.5), section_radius,
                                xlo, xhi, ylo, yhi);
                    }
                }
            }
        }
    }
    return amrex::max(Real(0),
                      amrex::min(Real(1), volume / (hx * hy * hz)));
}
DiffusionHierarchy
make_cloud_hierarchy (bool use_amr, int fine_n)
{
    BL_PROFILE("FLD::cloud_hierarchy");
    Array<int, AMREX_SPACEDIM> const nonperiodic{
        AMREX_D_DECL(0, 0, 0)};
    if (!use_amr) {
        return make_uniform_hierarchy(fine_n, 32, nonperiodic);
    }
    int const nbase = fine_n / 4;
    AMREX_ALWAYS_ASSERT(nbase > 1 && nbase % 8 == 0);
    return make_strip_hierarchy(
        {nbase, fine_n}, {{0, nbase}, {3 * fine_n / 8, 5 * fine_n / 8}},
        32, nonperiodic);
}

void
initialize_cloud_fields (DiffusionHierarchy const& hierarchy,
                         LevelData& state, LevelData& extinction,
                         LevelData& cloud_fraction, Real opacity_contrast)
{
    BL_PROFILE("FLD::cloud_field_initialization");
    // D = lambda / extinction, so the optically thick diffusion coefficient
    // differs between pure cloud and clear cells by opacity_contrast.
    Real constexpr clear_extinction = Real(0.001);
    Real const cloudy_extinction = clear_extinction * opacity_contrast;
    GpuArray<Real, penrose_cloud_count> centers_x{};
    GpuArray<Real, penrose_cloud_count> centers_z{};
    for (int cloud = 0; cloud < penrose_cloud_count; ++cloud) {
        centers_x[cloud] = Real(penrose_cloud_centers_xz[cloud][0]);
        centers_z[cloud] = Real(penrose_cloud_centers_xz[cloud][1]);
    }
    for (int level = 0; level < static_cast<int>(state.size()); ++level) {
        auto const dx = hierarchy.geom[level].CellSizeArray();
        auto const problo = hierarchy.geom[level].ProbLoArray();
        for (MFIter mfi(*state[level]); mfi.isValid(); ++mfi) {
            auto const e = state[level]->array(mfi);
            auto const chi = extinction[level]->array(mfi);
            auto const cloud = cloud_fraction[level]->array(mfi);
            ParallelFor(mfi.validbox(),
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                Real const x = problo[0] + (Real(i) + Real(0.5)) * dx[0];
                Real const y = problo[1] + (Real(j) + Real(0.5)) * dx[1];
                Real const z = problo[2] + (Real(k) + Real(0.5)) * dx[2];
                Real const fraction = sphere_cell_fraction(
                    x, y, z, dx[0], dx[1], dx[2], centers_x, centers_z);
                e(i, j, k) = Real(0.25) + Real(3.5) * y;
                chi(i, j, k) = fraction * cloudy_extinction +
                               (Real(1) - fraction) * clear_extinction;
                cloud(i, j, k) = fraction;
            });
        }
    }
    average_down_hierarchy(state, hierarchy);
    average_down_hierarchy(extinction, hierarchy);
    average_down_hierarchy(cloud_fraction, hierarchy);
}

Long
mixed_cloud_cells (LevelData const& cloud_fraction,
                   Vector<iMultiFab> const& masks)
{
    ReduceOps<ReduceOpSum> reduce_op;
    ReduceData<Long> reduce_data(reduce_op);
    using Tuple = typename decltype(reduce_data)::Type;
    for (int level = 0; level < static_cast<int>(cloud_fraction.size());
         ++level) {
        for (MFIter mfi(*cloud_fraction[level]); mfi.isValid(); ++mfi) {
            auto const fraction = cloud_fraction[level]->const_array(mfi);
            auto const mask = masks[level].const_array(mfi);
            reduce_op.eval(mfi.validbox(), reduce_data,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept -> Tuple
            {
                return {mask(i, j, k) && fraction(i, j, k) > Real(0) &&
                                fraction(i, j, k) < Real(1)
                            ? Long(1)
                            : Long(0)};
            });
        }
    }
    Long result = amrex::get<0>(reduce_data.value(reduce_op));
    ParallelDescriptor::ReduceLongSum(result);
    return result;
}


// T = E^(1/4): the initial state is in radiative equilibrium.
void
initialize_equilibrium_temperature (RadiationMaterialState& state)
{
    for (int level = 0; level < static_cast<int>(state.energy.size());
         ++level) {
        for (MFIter mfi(*state.energy[level]); mfi.isValid(); ++mfi) {
            auto const e = state.energy[level]->const_array(mfi);
            auto const t = state.temperature[level]->array(mfi);
            ParallelFor(mfi.validbox(),
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                t(i, j, k) = std::sqrt(std::sqrt(e(i, j, k)));
            });
        }
    }
}

void
write_cloud_plotfile (std::string const& name,
                      DiffusionHierarchy const& hierarchy,
                      RadiationMaterialSolver& solver,
                      RadiationMaterialState const& state,
                      LevelData& cloud_fraction)
{
    AMREX_ALWAYS_ASSERT(!name.empty());
    auto extinction = make_cell_data(hierarchy, 1, 0);
    auto diffusion = make_cell_data(hierarchy, 1, 0);
    solver.fill_cell_coefficients(state, extinction, diffusion);
    auto radiation_flux = make_cell_data(hierarchy, AMREX_SPACEDIM, 0);
    solver.fill_cell_flux(state, radiation_flux);
    average_down_hierarchy(cloud_fraction, hierarchy);
    auto plot = make_cell_data(hierarchy, 9, 0);
    for (int level = 0; level < static_cast<int>(plot.size()); ++level) {
        MultiFab::Copy(*plot[level], *state.energy[level], 0, 0, 1, 0);
        MultiFab::Copy(*plot[level], *state.temperature[level], 0, 1, 1, 0);
        MultiFab::Copy(*plot[level], *extinction[level], 0, 2, 1, 0);
        MultiFab::Copy(*plot[level], *diffusion[level], 0, 3, 1, 0);
        MultiFab::Copy(*plot[level], *diffusion[level], 0, 4, 1, 0);
        MultiFab::Multiply(*plot[level], *extinction[level], 0, 4, 1, 0);
        MultiFab::Copy(*plot[level], *cloud_fraction[level], 0, 5, 1, 0);
        MultiFab::Copy(*plot[level], *radiation_flux[level], 0, 6,
                       3, 0);
    }
    Vector<std::string> const variables{
        "radiation_energy", "material_temperature", "extinction",
        "diffusion_coefficient", "flux_limiter", "cloud_volume_fraction",
        "radiation_flux_x", "radiation_flux_y", "radiation_flux_z"};
    WriteMultiLevelPlotfile(
        name, static_cast<int>(plot.size()), get_level_const_ptrs(plot),
        variables, hierarchy.geom, Real(0), Vector<int>(plot.size(), 0),
        hierarchy.ref_ratio);
    AsyncOut::Finish();

    PlotFileData written(name);
    AMREX_ALWAYS_ASSERT(written.finestLevel() + 1 ==
                        static_cast<int>(plot.size()));
    AMREX_ALWAYS_ASSERT(written.varNames() == variables);
    for (int level = 0; level < static_cast<int>(plot.size()); ++level) {
        AMREX_ALWAYS_ASSERT(written.probDomain(level) ==
                            hierarchy.geom[level].Domain());
        AMREX_ALWAYS_ASSERT(written.boxArray(level) == hierarchy.grids[level]);
        if (level + 1 < static_cast<int>(plot.size())) {
            AMREX_ALWAYS_ASSERT(written.refRatioVect(level) ==
                                hierarchy.ref_ratio[level]);
        }
        MultiFab reloaded = written.get(level);
        MultiFab difference(hierarchy.grids[level], hierarchy.dmap[level],
                            static_cast<int>(variables.size()), 0);
        difference.ParallelCopy(reloaded, 0, 0,
                                static_cast<int>(variables.size()),
                                IntVect(0), IntVect(0));
        MultiFab::Subtract(difference, *plot[level], 0, 0,
                           static_cast<int>(variables.size()), 0);
        AMREX_ALWAYS_ASSERT(difference.norm0(
                                0, static_cast<int>(variables.size()),
                                IntVect(0)) == Real(0));
    }
    amrex::Print() << "Wrote and verified FLD cloud plotfile " << name << '\n';
}

} // namespace

// The Penrose cloud layer as a radiation-material problem stepped to
// steady state with the unified solver (v = 0): opacity sigma = extinction
// (opacity power 0), equilibrium E = 4 entering at the top (y-high) and 0
// at the bottom through Robin faces 1/2 E + D dE/dn = 1/2 E_eq, insulated
// x and z faces, and the cell Levermore-Pomraning limiter with surface
// opacities on cell faces.  A backward Euler step of dt = 1e10 is the
// steady problem, T = E^(1/4) and -div D grad E = 0, to far below the
// nonlinear tolerance; dt can instead ramp geometrically from
// cloud_dt_initial to cloud_dt_final.
CloudResult
run_cloud (bool use_amr, int fine_n, bool limited, bool iteration_output,
           std::string const& plotfile_name, Real opacity_contrast)
{
    BL_PROFILE("FLD::run_cloud");
    AMREX_ALWAYS_ASSERT(fine_n > 0 && fine_n % 4 == 0);
    AMREX_ALWAYS_ASSERT(std::isfinite(opacity_contrast) &&
                        opacity_contrast >= Real(1));
    Real const radius = cloud_radius();
    for (int first = 0; first < penrose_cloud_count; ++first) {
        Real const x = Real(penrose_cloud_centers_xz[first][0]);
        Real const z = Real(penrose_cloud_centers_xz[first][1]);
        AMREX_ALWAYS_ASSERT(x >= radius && x <= Real(1) - radius &&
                            z >= radius && z <= Real(1) - radius);
        for (int second = first + 1; second < penrose_cloud_count;
             ++second) {
            Real const dx = x - Real(penrose_cloud_centers_xz[second][0]);
            Real const dz = z - Real(penrose_cloud_centers_xz[second][1]);
            AMREX_ALWAYS_ASSERT(dx * dx + dz * dz >
                                Real(4) * radius * radius);
        }
    }
    DiffusionHierarchy hierarchy = make_cloud_hierarchy(use_amr, fine_n);

    Real constexpr beta = Real(0.5);
    Real constexpr top_equilibrium = Real(4);
    Real constexpr incident_marshak_flux = Real(1);
    RadiationMaterialOptions options;
    options.opacity_power = Real(0);
    options.limiter = limited;
    options.diffusion = DiffusionCoefficient::CellLimiterSurfaceOpacity;
    options.robin_b = RobinCoefficient::Diffusion;
    for (auto& boundary : options.boundary) {
        boundary = RadiationBoundary::Neumann;
    }
    options.boundary[2] = RadiationBoundary::Robin;
    options.boundary[3] = RadiationBoundary::Robin;
    options.robin_a[2] = beta;
    options.robin_a[3] = beta;
    options.incoming[3] = beta * top_equilibrium;
    options.solver = "newton_krylov";
    options.nonlinear_tolerance =
        (sizeof(Real) == sizeof(float)) ? Real(2.e-4) : Real(2.e-6);
    options.positive_update = false;
    options.maximum_nonlinear_iterations = 100;
    options.maximum_linear_iterations = 300;
    options.krylov_restart_length = 100;
    options.maximum_line_search_iterations = 24;
    options.centered_difference = false;
    options.linear_verbosity = iteration_output ? 2 : 0;
    // One step long enough to be the steady problem, from the predictor
    // iterations; a geometric ramp is available but much slower.
    options.predict_first = true;
    Real dt = Real(1.e10);
    Real dt_growth = Real(10);
    Real dt_final = Real(1.e10);
    {
        ParmParse pp;
        pp.query("cloud_solver", options.solver);
        pp.query("cloud_predict_first", options.predict_first);
        pp.query("cloud_eisenstat_walker", options.eisenstat_walker);
        pp.query("cloud_dt_initial", dt);
        pp.query("cloud_dt_growth", dt_growth);
        pp.query("cloud_dt_final", dt_final);
    }
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        dt > Real(0) && dt_growth >= Real(1) && dt_final >= dt,
        "cloud_dt_initial > 0, cloud_dt_growth >= 1, and "
        "cloud_dt_final >= cloud_dt_initial are required");
    RadiationMaterialSolver solver(hierarchy, options);
    auto const& masks = solver.masks();

    RadiationMaterialState state = solver.make_state();
    auto extinction = make_cell_data(hierarchy, 1, 1);
    auto cloud_fraction = make_cell_data(hierarchy, 1, 0);
    initialize_cloud_fields(hierarchy, state.energy, extinction,
                            cloud_fraction, opacity_contrast);
    initialize_equilibrium_temperature(state);
    solver.set_opacity_coefficient(extinction);

    CloudResult result;
    result.cells = composite_cell_count(masks);
    result.mixed_cells = mixed_cloud_cells(cloud_fraction, masks);
    Real constexpr pi = Real(3.1415926535897932384626433832795);
    Real const expected_cloudy_volume =
        Real(penrose_cloud_count) * Real(4) / Real(3) * pi *
        cloud_radius() * cloud_radius() * cloud_radius();
    Real const cloudy_volume =
        composite_volume_sum(cloud_fraction, hierarchy, masks);
    result.cloudy_volume_relative_error =
        std::abs(cloudy_volume - expected_cloudy_volume) /
        expected_cloudy_volume;
    Real const cloudy_volume_tolerance =
        (sizeof(Real) == sizeof(float)) ? Real(1.e-3) : Real(1.e-8);
    AMREX_ALWAYS_ASSERT(result.cloudy_volume_relative_error <
                        cloudy_volume_tolerance);
    AMREX_ALWAYS_ASSERT(result.mixed_cells > 0);

    for (int step = 1;; ++step) {
        int const previous_iterations = solver.counters().nonlinear;
        solver.advance(state, dt);
        result.time_steps = step;
        if (iteration_output) {
            amrex::Print()
                << "FLD cloud " << (use_amr ? "AMR" : "uniform")
                << " step=" << step << ", dt=" << dt
                << ", nonlinear iterations="
                << solver.counters().nonlinear - previous_iterations
                << ", change=" << solver.counters().final_change << '\n';
        }
        if (dt >= dt_final) { break; }
        dt = amrex::min(dt * dt_growth, dt_final);
    }

    auto const& counters = solver.counters();
    result.nonlinear_iterations = counters.nonlinear;
    result.total_newton_krylov_iterations = counters.krylov;
    result.maximum_newton_krylov_iterations = counters.maximum_krylov;
    result.predictor_fallbacks = counters.predictor_fallbacks;
    result.final_nonlinear_change = counters.final_change;
    result.final_nonlinear_residual = counters.final_coupled_residual;
    result.solver = counters.summary;

    // Outward fluxes: positive through the bottom, negative (inflow) at the
    // top; at steady state they cancel.
    auto const outflow = solver.boundary_outflow(state);
    Real const bottom_flux = outflow[2];
    Real const top_flux = outflow[3];
    result.transmission = bottom_flux / incident_marshak_flux;
    result.balance_error = std::abs(bottom_flux + top_flux) /
                           amrex::max(std::abs(bottom_flux), Real(1.e-30));
    auto const [minimum, maximum] =
        composite_minimum_maximum(state.energy, masks);
    result.minimum_energy = minimum;
    result.maximum_energy = maximum;
    if (iteration_output) {
        amrex::Print()
            << "FLD cloud " << (use_amr ? "AMR" : "uniform")
            << " steps=" << result.time_steps
            << ", Newton-Krylov iterations=" << result.nonlinear_iterations
            << "/" << result.total_newton_krylov_iterations
            << ", residual=" << result.final_nonlinear_residual
            << ", transmission=" << result.transmission << '\n';
    }
    if (!plotfile_name.empty()) {
        write_cloud_plotfile(plotfile_name, hierarchy, solver, state,
                             cloud_fraction);
    }

    Real const balance_limit =
        (sizeof(Real) == sizeof(float)) ? Real(2.e-3) : Real(2.e-6);
    AMREX_ALWAYS_ASSERT(result.balance_error < balance_limit);
    AMREX_ALWAYS_ASSERT(result.minimum_energy >= Real(-1.e-8));
    AMREX_ALWAYS_ASSERT(result.maximum_energy <= Real(4.01));
    AMREX_ALWAYS_ASSERT(result.transmission > Real(0) &&
                        result.transmission < Real(1));
    return result;
}

} // namespace fld_test
