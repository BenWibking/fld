#include "AndersonMixer.H"
#include "FLDTestCommon.H"

namespace fld_test
{

Real
semicircle_area_primitive (Real x, Real center, Real radius)
{
    Real const normalized_offset =
        std::clamp((x - center) / radius, Real(-1), Real(1));
    Real const normalized_height = std::sqrt(amrex::max(
        Real(0), (Real(1) - normalized_offset) *
                     (Real(1) + normalized_offset)));
    return Real(0.5) * radius * radius *
           (normalized_offset * normalized_height +
            std::asin(normalized_offset));
}

Real
circle_rectangle_intersection_area (Real center_x, Real center_y, Real radius,
                                    Real xlo, Real xhi, Real ylo, Real yhi)
{
    AMREX_ALWAYS_ASSERT(radius > Real(0));
    AMREX_ALWAYS_ASSERT(xlo < xhi && ylo < yhi);

    Real const nearest_x = std::clamp(center_x, xlo, xhi);
    Real const nearest_y = std::clamp(center_y, ylo, yhi);
    Real const nearest_dx = nearest_x - center_x;
    Real const nearest_dy = nearest_y - center_y;
    if (nearest_dx * nearest_dx + nearest_dy * nearest_dy >=
        radius * radius) {
        return Real(0);
    }

    bool rectangle_inside_circle = true;
    for (Real const x : {xlo, xhi}) {
        for (Real const y : {ylo, yhi}) {
            Real const dx = x - center_x;
            Real const dy = y - center_y;
            rectangle_inside_circle =
                rectangle_inside_circle &&
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

    Vector<Real> breakpoints{integration_lo, integration_hi};
    for (Real const y : {ylo, yhi}) {
        Real const offset = std::abs(y - center_y);
        if (offset < radius) {
            Real const dx =
                std::sqrt(radius * radius - offset * offset);
            Real const left = center_x - dx;
            Real const right = center_x + dx;
            if (left > integration_lo && left < integration_hi) {
                breakpoints.push_back(left);
            }
            if (right > integration_lo && right < integration_hi) {
                breakpoints.push_back(right);
            }
        }
    }
    std::sort(breakpoints.begin(), breakpoints.end());

    Real area = Real(0);
    for (std::size_t segment = 1; segment < breakpoints.size(); ++segment) {
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
        Real const constant =
            (upper_is_circle ? center_y : yhi) -
            (lower_is_circle ? center_y : ylo);
        int const semicircle_coefficient =
            int(upper_is_circle) + int(lower_is_circle);
        area += constant * (b - a) +
                Real(semicircle_coefficient) *
                    (semicircle_area_primitive(b, center_x, radius) -
                     semicircle_area_primitive(a, center_x, radius));
    }
    return std::clamp(area, Real(0), rectangle_area);
}

Real
cloud_radius ()
{
    Real constexpr spacing = Real(1) / Real(8.5);
    Real constexpr diameter = spacing / Real(1.1);
    return Real(0.5) * diameter;
}

Real
cloud_volume_fraction (Cell const& cell)
{
    Real constexpr spacing = Real(1) / Real(8.5);
    Real const radius = cloud_radius();
    Real const xlo = cell.x - Real(0.5) * cell.hx;
    Real const xhi = cell.x + Real(0.5) * cell.hx;
    Real const ylo = cell.y - Real(0.5) * cell.hy;
    Real const yhi = cell.y + Real(0.5) * cell.hy;
    Real cloudy_area = Real(0);
    for (int cloud = 0; cloud <= 8; ++cloud) {
        Real const center_x = Real(cloud) * spacing;
        cloudy_area += circle_rectangle_intersection_area(
            center_x, Real(0.5), radius, xlo, xhi, ylo, yhi);
    }
    Real const fraction = cloudy_area / cell.volume;
    AMREX_ALWAYS_ASSERT(fraction >= Real(-1.e-12));
    AMREX_ALWAYS_ASSERT(fraction <= Real(1) + Real(1.e-12));
    return std::clamp(fraction, Real(0), Real(1));
}

CloudMesh
make_cloud_mesh (bool use_amr, int fine_n)
{
    AMREX_ALWAYS_ASSERT(fine_n > 0);
    AMREX_ALWAYS_ASSERT(fine_n % 4 == 0);
    Real constexpr beta = Real(0.5);
    std::array<BoundaryCondition, 4> boundary{
        reflecting_boundary(), reflecting_boundary(),
        marshak_boundary(Real(0), beta), marshak_boundary(Real(4), beta)};

    CloudMesh cloud_mesh;
    if (!use_amr) {
        cloud_mesh.mesh = make_mesh(
            fine_n, 1, [] (int, int, int) noexcept { return false; },
            [] (Real, Real) noexcept { return true; }, boundary);
        cloud_mesh.level_n = {fine_n};
        cloud_mesh.level_y_bounds = {{0, fine_n}};
        return cloud_mesh;
    }

    if (fine_n == 512) {
        // Reproduce the three-level hierarchy used for Fig. 6: a 32^2 base
        // grid, a factor-four level over the middle half, and a second
        // factor-four level over the cloudy middle quarter.
        cloud_mesh.mesh = make_nested_mesh(
            32, Vector<int>{4, 4},
            [] (int level, int, int j, int n) noexcept
            {
                if (level == 0) {
                    return j >= n / 4 && j < 3 * n / 4;
                }
                return level == 1 &&
                       j >= 3 * n / 8 && j < 5 * n / 8;
            },
            [] (Real, Real) noexcept { return true; }, boundary);
        cloud_mesh.level_n = {32, 128, 512};
        cloud_mesh.level_y_bounds = {
            {0, cloud_mesh.level_n[0]},
            {cloud_mesh.level_n[1] / 4,
             3 * cloud_mesh.level_n[1] / 4},
            {3 * cloud_mesh.level_n[2] / 8,
             5 * cloud_mesh.level_n[2] / 8}};
        return cloud_mesh;
    }

    int const nbase = fine_n / 4;
    AMREX_ALWAYS_ASSERT(nbase % 8 == 0);
    cloud_mesh.mesh = make_mesh(
        nbase, 4, [] (int, int j, int n) noexcept
        { return j >= 3 * n / 8 && j < 5 * n / 8; },
        [] (Real, Real) noexcept { return true; }, boundary);
    cloud_mesh.level_n = {nbase, fine_n};
    cloud_mesh.level_y_bounds = {
        {0, nbase}, {3 * fine_n / 8, 5 * fine_n / 8}};
    return cloud_mesh;
}

std::pair<Real, Real>
cloud_boundary_fluxes (Mesh const& mesh, Vector<Real> const& energy,
                       Vector<Real> const& diffusion)
{
    Real bottom_flux = Real(0);
    Real top_flux = Real(0);
    for (Long row = 0; row < static_cast<Long>(mesh.cells.size()); ++row) {
        for (auto const& face : mesh.cells[row].faces) {
            if (face.neighbor >= 0 ||
                face.boundary.kind != BoundaryKind::marshak) {
                continue;
            }
            Real const flux = face_transmissibility(face, row, diffusion) *
                              (energy[row] - face.boundary.value);
            if (face.side == ylo) {
                bottom_flux += flux;
            } else if (face.side == yhi) {
                top_flux += flux;
            }
        }
    }
    return {bottom_flux, top_flux};
}

void
write_cloud_plotfile (std::string const& plotfile_name,
                      CloudMesh const& cloud_mesh,
                      Vector<Real> const& energy,
                      Vector<Real> const& extinction,
                      Vector<Real> const& diffusion)
{
    Mesh const& mesh = cloud_mesh.mesh;
    AMREX_ALWAYS_ASSERT(!plotfile_name.empty());
    AMREX_ALWAYS_ASSERT(energy.size() == mesh.cells.size());
    AMREX_ALWAYS_ASSERT(extinction.size() == energy.size());
    AMREX_ALWAYS_ASSERT(diffusion.size() == energy.size());
    AMREX_ALWAYS_ASSERT(!cloud_mesh.level_n.empty());
    AMREX_ALWAYS_ASSERT(cloud_mesh.level_n.size() ==
                        cloud_mesh.level_y_bounds.size());
    AMREX_ALWAYS_ASSERT(cloud_mesh.level_n.back() == mesh.fine_n);

    int constexpr component_count = 5;
    Vector<std::string> const variable_names{
        "radiation_energy", "extinction", "diffusion_coefficient",
        "flux_limiter", "cloud_volume_fraction"};
    int const level_count = static_cast<int>(cloud_mesh.level_n.size());
    RealBox const physical_domain(
        {AMREX_D_DECL(Real(0), Real(0), Real(0))},
        {AMREX_D_DECL(Real(1), Real(1), Real(1))});
    Array<int, AMREX_SPACEDIM> const is_periodic{
        AMREX_D_DECL(0, 0, 0)};

    Vector<BoxArray> grids(level_count);
    Vector<Geometry> geometries(level_count);
    for (int level = 0; level < level_count; ++level) {
        int const level_n = cloud_mesh.level_n[level];
        auto const [ylo, yhi] = cloud_mesh.level_y_bounds[level];
        AMREX_ALWAYS_ASSERT(level_n > 0);
        AMREX_ALWAYS_ASSERT(ylo >= 0 && ylo < yhi && yhi <= level_n);
        if (level > 0) {
            AMREX_ALWAYS_ASSERT(
                level_n % cloud_mesh.level_n[level - 1] == 0);
        }
        IntVect const domain_lo(AMREX_D_DECL(0, 0, 0));
        IntVect const domain_hi(
            AMREX_D_DECL(level_n - 1, level_n - 1, 0));
        Box const domain(domain_lo, domain_hi);
        IntVect const grid_lo(AMREX_D_DECL(0, ylo, 0));
        IntVect const grid_hi(
            AMREX_D_DECL(level_n - 1, yhi - 1, 0));
        grids[level] = BoxArray(Box(grid_lo, grid_hi));
        grids[level].maxSize(64);
        geometries[level].define(domain, physical_domain,
                                 CoordSys::cartesian, is_periodic);
    }

    Vector<MultiFab> plot_data(level_count);
    for (int level = 0; level < level_count; ++level) {
        DistributionMapping const distribution(grids[level]);
        plot_data[level].define(grids[level], distribution, component_count,
                                0);
        int const level_n = cloud_mesh.level_n[level];
        int const coarsening = mesh.fine_n / level_n;
        Real const inverse_sample_count =
            Real(1) / Real(coarsening * coarsening);
        Vector<Real> host_values(
            static_cast<std::size_t>(level_n) * level_n * component_count,
            Real(0));
        // Covered coarse cells are volume averages of the converged fine
        // solution; unrefined cells reduce to repeated samples of one owner.
        for (int j = 0; j < level_n; ++j) {
            for (int i = 0; i < level_n; ++i) {
                auto const output_index =
                    (static_cast<std::size_t>(j) * level_n + i) *
                    component_count;
                for (int jj = 0; jj < coarsening; ++jj) {
                    for (int ii = 0; ii < coarsening; ++ii) {
                        int const fine_i = i * coarsening + ii;
                        int const fine_j = j * coarsening + jj;
                        Long const row =
                            mesh.owner[static_cast<std::size_t>(fine_j) *
                                           mesh.fine_n +
                                       fine_i];
                        AMREX_ALWAYS_ASSERT(row >= 0);
                        Real const sample_scale = inverse_sample_count;
                        host_values[output_index] +=
                            sample_scale * energy[row];
                        host_values[output_index + 1] +=
                            sample_scale * extinction[row];
                        host_values[output_index + 2] +=
                            sample_scale * diffusion[row];
                        host_values[output_index + 3] +=
                            sample_scale * diffusion[row] * extinction[row];
                        host_values[output_index + 4] +=
                            sample_scale *
                            cloud_volume_fraction(mesh.cells[row]);
                    }
                }
            }
        }

        Gpu::DeviceVector<Real> device_values(host_values.size());
        Gpu::copy(Gpu::hostToDevice, host_values.begin(), host_values.end(),
                  device_values.begin());
        Real const* values = device_values.data();
        for (MFIter mfi(plot_data[level]); mfi.isValid(); ++mfi) {
            Box const& box = mfi.validbox();
            auto const array = plot_data[level].array(mfi);
            ParallelFor(
                box, component_count,
                [=] AMREX_GPU_DEVICE(int i, int j, int k, int component)
                {
                    auto const index =
                        (static_cast<std::size_t>(j) * level_n + i) *
                            component_count +
                        component;
                    array(i, j, k, component) = values[index];
                });
        }
        Gpu::streamSynchronize();
    }

    Vector<IntVect> refinement_ratios;
    refinement_ratios.reserve(level_count - 1);
    for (int level = 0; level + 1 < level_count; ++level) {
        int const refinement_ratio =
            cloud_mesh.level_n[level + 1] / cloud_mesh.level_n[level];
        AMREX_ALWAYS_ASSERT(refinement_ratio > 1);
        refinement_ratios.emplace_back(
            AMREX_D_DECL(refinement_ratio, refinement_ratio,
                         refinement_ratio));
    }
    WriteMultiLevelPlotfile(
        plotfile_name, level_count, GetVecOfConstPtrs(plot_data),
        variable_names, geometries, Real(0), Vector<int>(level_count, 0),
        refinement_ratios);
    amrex::Print() << "Wrote FLD cloud plotfile " << plotfile_name
                   << std::endl;
}

CloudResult
run_cloud (bool use_amr, int fine_n, int anderson_depth, Real anderson_beta,
           bool limited, bool iteration_output,
           std::string const& plotfile_name)
{
    CloudMesh cloud_mesh = make_cloud_mesh(use_amr, fine_n);
    Mesh const& mesh = cloud_mesh.mesh;
    Vector<Real> extinction(mesh.cells.size());
    Vector<Real> state(mesh.cells.size());
    Vector<Real> volume_weights(mesh.cells.size());
    Real cloudy_area = Real(0);
    Long mixed_cells = 0;
    Real constexpr clear_extinction = Real(0.1);
    Real constexpr cloudy_extinction = Real(1000);
    Real constexpr clear_density = Real(1);
    Real constexpr cloudy_density = Real(1);
    for (Long row = 0; row < static_cast<Long>(mesh.cells.size()); ++row) {
        auto const& cell = mesh.cells[row];
        Real const cloudy_fraction = cloud_volume_fraction(cell);
        // Apply the mass-weighted arithmetic Rosseland mean in Fig. 3.  The
        // pure-radiation cloud test specifies no material density contrast,
        // so both densities are normalized to one.
        Real const clear_fraction = Real(1) - cloudy_fraction;
        extinction[row] =
            (cloudy_fraction * cloudy_density * cloudy_extinction +
             clear_fraction * clear_density * clear_extinction) /
            (cloudy_fraction * cloudy_density +
             clear_fraction * clear_density);
        state[row] = Real(0.25) + Real(3.5) * cell.y;
        volume_weights[row] = cell.volume;
        cloudy_area += cloudy_fraction * cell.volume;
        if (cloudy_fraction > Real(0) && cloudy_fraction < Real(1)) {
            ++mixed_cells;
        }
    }

    CloudResult result;
    result.cells = static_cast<Long>(mesh.cells.size());
    result.mixed_cells = mixed_cells;
    Real constexpr pi = Real(3.1415926535897932384626433832795);
    Real const expected_cloudy_area =
        Real(8.5) * pi * cloud_radius() * cloud_radius();
    result.cloudy_area_relative_error =
        std::abs(cloudy_area - expected_cloudy_area) / expected_cloudy_area;
    Real const cloudy_area_tolerance =
        (sizeof(Real) == sizeof(float)) ? Real(2.e-4) : Real(2.e-12);
    AMREX_ALWAYS_ASSERT(result.cloudy_area_relative_error <
                        cloudy_area_tolerance);
    AMREX_ALWAYS_ASSERT(result.mixed_cells > 0);
    Real const nonlinear_tolerance =
        (sizeof(Real) == sizeof(float)) ? Real(2.e-4) : Real(2.e-6);
    int constexpr maximum_nonlinear_iterations = 125;
    AMG<Real>::Options cloud_amg_options;
    cloud_amg_options.priority_seed = 1;
    auto const& incident_boundary = mesh.outer_boundary[yhi];
    AMREX_ALWAYS_ASSERT(incident_boundary.kind == BoundaryKind::marshak);
    AndersonMixer mixer(anderson_depth, anderson_beta,
                         std::move(volume_weights),
                         incident_boundary.value);
    Real const incident_marshak_flux =
        Real(0.5) * incident_boundary.beta * incident_boundary.value;
    AMREX_ALWAYS_ASSERT(incident_marshak_flux > Real(0));

    for (int iteration = 0; iteration < maximum_nonlinear_iterations;
         ++iteration) {
        auto diffusion = compute_diffusion(mesh, state, extinction, limited);
        auto system = assemble_system(mesh, diffusion, {}, Real(0), false,
                                      &extinction);
        AMGGMRESSolver solver(system.matrix, cloud_amg_options);
        record_setup(result.solver, solver);
        auto solution = solver.solve(system.rhs);
        record_solve(result.solver, solution);

        result.final_nonlinear_change =
            maximum_relative_change(solution.values, state);
        ++result.nonlinear_iterations;
        auto const [iteration_bottom_flux, iteration_top_flux] =
            cloud_boundary_fluxes(mesh, solution.values, diffusion);
        amrex::ignore_unused(iteration_top_flux);
        if (iteration_output) {
            amrex::Print()
                << "FLD cloud " << (use_amr ? "AMR" : "uniform")
                << " nonlinear iteration=" << result.nonlinear_iterations
                << ", method="
                << (!limited ? "linear"
                             : (anderson_depth > 0 ? "Anderson" : "Picard"))
                << ", change=" << result.final_nonlinear_change
                << ", transmission="
                << iteration_bottom_flux / incident_marshak_flux
                << ", GMRES iterations=" << solution.iterations
                << ", true relative residual="
                << solution.relative_residual << std::endl;
        }
        if (result.final_nonlinear_change <= nonlinear_tolerance) {
            state = std::move(solution.values);
            break;
        }
        state = mixer.update(state, solution.values);
    }
    result.anderson_steps = mixer.anderson_steps();
    result.anderson_restarts = mixer.restarts();

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        result.final_nonlinear_change <= nonlinear_tolerance,
        "The cloud-layer FLD nonlinear iteration did not converge");

    auto diffusion = compute_diffusion(mesh, state, extinction, limited);
    auto const [bottom_flux, top_flux] =
        cloud_boundary_fluxes(mesh, state, diffusion);
    // beta * (E - E_eq) = c E / 2 - 2 F_inc, so
    // F_inc = beta * E_eq / 2 when beta = c / 2.
    result.transmission = bottom_flux / incident_marshak_flux;
    result.balance_error = std::abs(bottom_flux + top_flux) /
                           amrex::max(std::abs(bottom_flux), Real(1.e-30));
    auto const [minimum, maximum] = minimum_maximum(state);
    result.minimum_energy = minimum;
    result.maximum_energy = maximum;
    if (!plotfile_name.empty()) {
        write_cloud_plotfile(plotfile_name, cloud_mesh, state, extinction,
                             diffusion);
    }

    Real const balance_limit =
        (sizeof(Real) == sizeof(float)) ? Real(2.e-3) : Real(2.e-6);
    AMREX_ALWAYS_ASSERT(result.balance_error < balance_limit);
    AMREX_ALWAYS_ASSERT(result.minimum_energy >= Real(-1.e-8));
    AMREX_ALWAYS_ASSERT(result.maximum_energy <= Real(4.01));
    AMREX_ALWAYS_ASSERT(result.transmission > Real(0));
    AMREX_ALWAYS_ASSERT(result.transmission < Real(1));
    return result;
}

} // namespace fld_test
