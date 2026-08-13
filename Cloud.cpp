#include "FLDTestCommon.H"
#include "NewtonKrylovSolver.H"

#include <AMReX_AsyncOut.H>
#include <AMReX_MFIter.H>
#include <AMReX_ParallelDescriptor.H>
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
cloud_volume_fraction (Real x, Real y, Real hx, Real hy)
{
    Real constexpr spacing = Real(1) / Real(8.5);
    Real const radius = cloud_radius();
    Real const xlo = x - Real(0.5) * hx;
    Real const xhi = x + Real(0.5) * hx;
    Real const ylo = y - Real(0.5) * hy;
    Real const yhi = y + Real(0.5) * hy;
    Real cloudy_area = Real(0);
    for (int cloud = 0; cloud <= 8; ++cloud) {
        cloudy_area += circle_rectangle_intersection_area(
            Real(cloud) * spacing, Real(0.5), radius, xlo, xhi, ylo, yhi);
    }
    return amrex::max(
        Real(0), amrex::min(Real(1), cloudy_area / (hx * hy)));
}

DiffusionHierarchy
make_cloud_hierarchy (bool use_amr, int fine_n)
{
    Array<int, AMREX_SPACEDIM> const nonperiodic{
        AMREX_D_DECL(0, 0, 0)};
    if (!use_amr) {
        return make_uniform_hierarchy(fine_n, 32, nonperiodic);
    }
    if (fine_n == 512) {
        return make_strip_hierarchy(
            {32, 128, 512}, {{0, 32}, {32, 96}, {192, 320}}, 32,
            nonperiodic);
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
                         LevelData& cloud_fraction)
{
    // Extinction (opacity) contrast of 1e6: the diffusion coefficient is
    // D = lambda / extinction, so in the optically-thick limit D jumps from
    // 1/(3*1000) inside the clouds to 1/(3*0.001) in the clear background.
    Real constexpr clear_extinction = Real(0.001);
    Real constexpr cloudy_extinction = Real(1000);
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
                Real const fraction =
                    cloud_volume_fraction(x, y, dx[0], dx[1]);
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

std::pair<Real, Real>
cloud_boundary_fluxes (DiffusionHierarchy const& hierarchy,
                       LevelData const& energy, LevelData const& diffusion,
                       Vector<iMultiFab> const& masks)
{
    Real constexpr beta = Real(0.5);
    Real constexpr bottom_equilibrium = Real(0);
    Real constexpr top_equilibrium = Real(4);
    ReduceOps<ReduceOpSum, ReduceOpSum> reduce_op;
    ReduceData<Real, Real> reduce_data(reduce_op);
    using Tuple = typename decltype(reduce_data)::Type;
    for (int level = 0; level < static_cast<int>(energy.size()); ++level) {
        Box const domain = hierarchy.geom[level].Domain();
        auto const dx = hierarchy.geom[level].CellSizeArray();
        Real const area = dx[0];
        Real const distance = Real(0.5) * dx[1];
        for (MFIter mfi(*energy[level]); mfi.isValid(); ++mfi) {
            auto const e = energy[level]->const_array(mfi);
            auto const d = diffusion[level]->const_array(mfi);
            auto const mask = masks[level].const_array(mfi);
            reduce_op.eval(mfi.validbox(), reduce_data,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept -> Tuple
            {
                if (mask(i, j, k) == 0) {
                    return {Real(0), Real(0)};
                }
                Real bottom = Real(0);
                Real top = Real(0);
                Real const transmissibility =
                    area / (distance / d(i, j, k) + Real(1) / beta);
                if (j == domain.smallEnd(1)) {
                    bottom = transmissibility *
                             (e(i, j, k) - bottom_equilibrium);
                }
                if (j == domain.bigEnd(1)) {
                    top = transmissibility *
                          (e(i, j, k) - top_equilibrium);
                }
                return {bottom, top};
            });
        }
    }
    auto const value = reduce_data.value(reduce_op);
    Real bottom = amrex::get<0>(value);
    Real top = amrex::get<1>(value);
    ParallelDescriptor::ReduceRealSum(bottom);
    ParallelDescriptor::ReduceRealSum(top);
    return {bottom, top};
}

void
fill_robin_data (DiffusionHierarchy const& hierarchy,
                 LevelData const& diffusion, LevelData& robin_a,
                 LevelData& robin_b, LevelData& robin_f)
{
    Real constexpr beta = Real(0.5);
    set_level_data(robin_a, beta);
    copy_level_data(robin_b, diffusion);
    set_level_data(robin_f, Real(0));
    for (int level = 0; level < static_cast<int>(robin_f.size()); ++level) {
        int const top = hierarchy.geom[level].Domain().bigEnd(1);
        for (MFIter mfi(*robin_f[level]); mfi.isValid(); ++mfi) {
            auto const f = robin_f[level]->array(mfi);
            ParallelFor(mfi.validbox(),
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                if (j == top) {
                    f(i, j, k) = beta * Real(4);
                }
            });
        }
    }
}

class CloudNewtonProblem
{
  public:
    using State = LevelData;
    using RT = Real;

    CloudNewtonProblem (
        DiffusionHierarchy const& hierarchy, Vector<iMultiFab> const& masks,
        LevelData& extinction, LevelData const& rhs, LevelData const& acoef,
        PhysicalBoundaryData const& boundary, bool limited,
        MLABecLapAMG& base_solver, SolverSummary& summary)
        : m_hierarchy(hierarchy), m_masks(masks), m_extinction(extinction),
          m_rhs(rhs), m_acoef(acoef), m_boundary(boundary),
          m_limited(limited), m_base_solver(base_solver), m_summary(summary),
          m_trial_solver(hierarchy.geom, hierarchy.grids, hierarchy.dmap),
          m_base_diffusion(make_cell_data(hierarchy, 1, 1)),
          m_trial_diffusion(make_cell_data(hierarchy, 1, 1)),
          m_base_bcoef(make_face_data(hierarchy)),
          m_trial_bcoef(make_face_data(hierarchy)),
          m_base_robin_a(make_cell_data(hierarchy, 1, 0)),
          m_base_robin_b(make_cell_data(hierarchy, 1, 0)),
          m_base_robin_f(make_cell_data(hierarchy, 1, 0)),
          m_trial_robin_a(make_cell_data(hierarchy, 1, 0)),
          m_trial_robin_b(make_cell_data(hierarchy, 1, 0)),
          m_trial_robin_f(make_cell_data(hierarchy, 1, 0))
    {
        m_trial_solver.setMatrixOnly(true);
    }

    void prepare (State const& state)
    {
        auto const [minimum, maximum] =
            composite_minimum_maximum(state, m_masks);
        amrex::ignore_unused(minimum);
        m_state_scale = amrex::max(maximum, Real(1));
        setup(m_base_solver, state, m_base_diffusion, m_base_bcoef,
              m_base_robin_a, m_base_robin_b, m_base_robin_f);
        record_setup(m_summary, m_base_solver);
    }

    Real predict (State& state, Real damping)
    {
        AMREX_ALWAYS_ASSERT(damping > Real(0) && damping <= Real(1));
        setup(m_base_solver, state, m_base_diffusion, m_base_bcoef,
              m_base_robin_a, m_base_robin_b, m_base_robin_f);
        record_setup(m_summary, m_base_solver);
        State prediction = clone_level_data(state);
        auto const info = m_base_solver.solve(
            get_level_ptrs(prediction), get_level_const_ptrs(m_rhs),
            linear_tolerance(), Real(0));
        record_solve(m_summary, info);
        Real const change =
            composite_maximum_relative_change(prediction, state, m_masks);
        lincomb_level_data(state, Real(1) - damping, state, damping,
                           prediction);
        average_down_hierarchy(state, m_hierarchy);
        return change;
    }

    void residual (State const& state, State& output)
    {
        setup(m_trial_solver, state, m_trial_diffusion, m_trial_bcoef,
              m_trial_robin_a, m_trial_robin_b, m_trial_robin_f);
        m_trial_solver.residual(get_level_ptrs(output),
                                get_level_const_ptrs(state),
                                get_level_const_ptrs(m_rhs));
    }

    void linearized_residual (State const& state, State& output)
    {
        setup(m_trial_solver, state, m_trial_diffusion, m_trial_bcoef,
              m_trial_robin_a, m_trial_robin_b, m_trial_robin_f);
        m_trial_solver.residual(get_level_ptrs(output),
                                get_level_const_ptrs(state),
                                get_level_const_ptrs(m_rhs));
    }

    State makeVecRHS () const { return make_cell_data(m_hierarchy, 1, 0); }
    State makeVecLHS () const { return make_cell_data(m_hierarchy, 1, 1); }

    void precondition (State& output, State const& rhs)
    {
        set_level_data(output, Real(0));
        m_base_solver.precondition(get_level_ptrs(output),
                                   get_level_const_ptrs(rhs));
    }

    void assign (State& lhs, State const& rhs) const
    {
        copy_level_data(lhs, rhs);
    }

    Real dotProduct (State const& lhs, State const& rhs) const
    {
        return composite_weighted_dot(lhs, rhs, m_hierarchy, m_masks) /
               (m_state_scale * m_state_scale);
    }

    void increment (State& lhs, State const& rhs, Real scale) const
    {
        saxpy_level_data(lhs, scale, rhs);
    }

    void linComb (State& lhs, Real a, State const& rhs_a, Real b,
                  State const& rhs_b) const
    {
        lincomb_level_data(lhs, a, rhs_a, b, rhs_b);
    }

    Real norm2 (State const& state) const
    {
        return std::sqrt(amrex::max(dotProduct(state, state), Real(0)));
    }

    void scale (State& state, Real factor) const
    {
        for (auto& level : state) {
            level->mult(factor, 0, 1, 0);
        }
    }

    void setToZero (State& state) const { set_level_data(state, Real(0)); }
    State clone (State const& state) const { return clone_level_data(state); }

    void fill_candidate (State& candidate, State const& state,
                         State const& correction, Real step_length) const
    {
        lincomb_level_data(candidate, Real(1), state, step_length,
                           correction);
        average_down_hierarchy(candidate, m_hierarchy);
    }

    bool admissible (State const& state) const
    {
        auto const [minimum, maximum] =
            composite_minimum_maximum(state, m_masks);
        return composite_all_finite(state, m_masks) && minimum >= Real(0) &&
               maximum <= Real(4.01);
    }

    Real relative_change (State const& lhs, State const& rhs) const
    {
        return composite_maximum_relative_change(lhs, rhs, m_masks);
    }

  private:
    void setup (MLABecLapAMG& solver, State const& state,
                LevelData& diffusion, FaceData& bcoef, LevelData& robin_a,
                LevelData& robin_b, LevelData& robin_f)
    {
        State work = clone_level_data(state);
        compute_diffusion(m_hierarchy, work, m_extinction, diffusion,
                          m_boundary, m_limited);
        fill_face_coefficients(m_hierarchy, diffusion, &m_extinction, bcoef,
                               true);
        fill_robin_data(m_hierarchy, diffusion, robin_a, robin_b, robin_f);
        RobinBCData robin{get_level_const_ptrs(robin_a),
                          get_level_const_ptrs(robin_b),
                          get_level_const_ptrs(robin_f)};
        solver.setup(Real(0), Real(1), get_level_const_ptrs(m_acoef),
                     get_face_const_ptrs(bcoef), m_boundary.lo,
                     m_boundary.hi, {}, robin);
    }

    DiffusionHierarchy const& m_hierarchy;
    Vector<iMultiFab> const& m_masks;
    LevelData& m_extinction;
    LevelData const& m_rhs;
    LevelData const& m_acoef;
    PhysicalBoundaryData const& m_boundary;
    bool m_limited;
    MLABecLapAMG& m_base_solver;
    SolverSummary& m_summary;
    MLABecLapAMG m_trial_solver;
    LevelData m_base_diffusion;
    LevelData m_trial_diffusion;
    FaceData m_base_bcoef;
    FaceData m_trial_bcoef;
    LevelData m_base_robin_a;
    LevelData m_base_robin_b;
    LevelData m_base_robin_f;
    LevelData m_trial_robin_a;
    LevelData m_trial_robin_b;
    LevelData m_trial_robin_f;
    Real m_state_scale = Real(1);
};

LevelData
make_cloud_radiation_flux (DiffusionHierarchy const& hierarchy,
                           LevelData& energy, LevelData& extinction,
                           LevelData& diffusion)
{
    static_assert(AMREX_SPACEDIM == 2);
    Real constexpr beta = Real(0.5);
    Real constexpr bottom_equilibrium = Real(0);
    Real constexpr top_equilibrium = Real(4);

    fill_level_ghosts(energy, hierarchy);
    auto face_coefficients = make_face_data(hierarchy);
    fill_face_coefficients(hierarchy, diffusion, &extinction,
                           face_coefficients, true);
    auto face_flux = make_face_data(hierarchy);
    auto radiation_flux = make_cell_data(hierarchy, AMREX_SPACEDIM, 0);

    for (int level = 0; level < static_cast<int>(energy.size()); ++level) {
        Box const domain = hierarchy.geom[level].Domain();
        auto const dlo = amrex::lbound(domain);
        auto const dhi = amrex::ubound(domain);
        auto const dx = hierarchy.geom[level].CellSizeArray();
        for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
            MultiFab& flux = *face_flux[level][direction];
            for (MFIter mfi(flux); mfi.isValid(); ++mfi) {
                auto const e = energy[level]->const_array(mfi);
                auto const d = diffusion[level]->const_array(mfi);
                auto const b =
                    face_coefficients[level][direction]->const_array(mfi);
                auto const f = flux.array(mfi);
                ParallelFor(mfi.validbox(),
                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    int il = i;
                    int jl = j;
                    int ir = i;
                    int jr = j;
                    if (direction == 0) {
                        il = i - 1;
                    } else {
                        jl = j - 1;
                    }
                    bool const left_inside = il >= dlo.x && il <= dhi.x &&
                                             jl >= dlo.y && jl <= dhi.y;
                    bool const right_inside = ir >= dlo.x && ir <= dhi.x &&
                                              jr >= dlo.y && jr <= dhi.y;
                    if (left_inside && right_inside) {
                        f(i, j, k) =
                            -b(i, j, k) *
                            (e(ir, jr, k) - e(il, jl, k)) / dx[direction];
                        return;
                    }

                    // The x boundaries are homogeneous Neumann.  At a y
                    // boundary, reconstruct the Robin flux used by the
                    // linear system and orient it in the positive y direction.
                    if (direction == 0) {
                        f(i, j, k) = Real(0);
                        return;
                    }
                    bool const lower_boundary = !left_inside;
                    int const ic = lower_boundary ? ir : il;
                    int const jc = lower_boundary ? jr : jl;
                    Real const equilibrium = lower_boundary
                                                 ? bottom_equilibrium
                                                 : top_equilibrium;
                    Real const orientation =
                        lower_boundary ? Real(-1) : Real(1);
                    Real const distance = Real(0.5) * dx[direction];
                    f(i, j, k) =
                        orientation * b(i, j, k) * beta *
                        (e(ic, jc, k) - equilibrium) /
                        (d(ic, jc, k) + beta * distance);
                });
            }
        }

        for (MFIter mfi(*radiation_flux[level]); mfi.isValid(); ++mfi) {
            auto const fx = face_flux[level][0]->const_array(mfi);
            auto const fy = face_flux[level][1]->const_array(mfi);
            auto const f = radiation_flux[level]->array(mfi);
            ParallelFor(mfi.validbox(),
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                f(i, j, k, 0) =
                    Real(0.5) * (fx(i, j, k) + fx(i + 1, j, k));
                f(i, j, k, 1) =
                    Real(0.5) * (fy(i, j, k) + fy(i, j + 1, k));
            });
        }
    }
    average_down_hierarchy(radiation_flux, hierarchy);
    return radiation_flux;
}

void
write_cloud_plotfile (std::string const& name,
                      DiffusionHierarchy const& hierarchy, LevelData& state,
                      LevelData& extinction, LevelData& diffusion,
                      LevelData& cloud_fraction)
{
    AMREX_ALWAYS_ASSERT(!name.empty());
    average_down_hierarchy(state, hierarchy);
    average_down_hierarchy(extinction, hierarchy);
    average_down_hierarchy(diffusion, hierarchy);
    average_down_hierarchy(cloud_fraction, hierarchy);
    auto radiation_flux = make_cloud_radiation_flux(
        hierarchy, state, extinction, diffusion);
    auto plot = make_cell_data(hierarchy, 5 + AMREX_SPACEDIM, 0);
    for (int level = 0; level < static_cast<int>(plot.size()); ++level) {
        MultiFab::Copy(*plot[level], *state[level], 0, 0, 1, 0);
        MultiFab::Copy(*plot[level], *extinction[level], 0, 1, 1, 0);
        MultiFab::Copy(*plot[level], *diffusion[level], 0, 2, 1, 0);
        MultiFab::Copy(*plot[level], *cloud_fraction[level], 0, 4, 1, 0);
        MultiFab::Copy(*plot[level], *diffusion[level], 0, 3, 1, 0);
        MultiFab::Multiply(*plot[level], *extinction[level], 0, 3, 1, 0);
        MultiFab::Copy(*plot[level], *radiation_flux[level], 0, 5,
                       AMREX_SPACEDIM, 0);
    }
    Vector<std::string> const variables{
        "radiation_energy", "extinction", "diffusion_coefficient",
        "flux_limiter", "cloud_volume_fraction", "radiation_flux_x",
        "radiation_flux_y"};
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

CloudResult
run_cloud (bool use_amr, int fine_n, bool limited, bool iteration_output,
           std::string const& plotfile_name)
{
    AMREX_ALWAYS_ASSERT(fine_n > 0 && fine_n % 4 == 0);
    DiffusionHierarchy hierarchy = make_cloud_hierarchy(use_amr, fine_n);
    auto masks = make_composite_masks(hierarchy);
    auto state = make_cell_data(hierarchy, 1, 1);
    auto rhs = make_cell_data(hierarchy, 1, 0);
    auto acoef = make_cell_data(hierarchy, 1, 0);
    auto extinction = make_cell_data(hierarchy, 1, 1);
    auto diffusion = make_cell_data(hierarchy, 1, 1);
    auto cloud_fraction = make_cell_data(hierarchy, 1, 0);
    initialize_cloud_fields(hierarchy, state, extinction, cloud_fraction);
    set_level_data(rhs, Real(0));
    set_level_data(acoef, Real(1));

    CloudResult result;
    result.cells = composite_cell_count(masks);
    result.mixed_cells = mixed_cloud_cells(cloud_fraction, masks);
    Real constexpr pi = Real(3.1415926535897932384626433832795);
    Real const expected_cloudy_area =
        Real(8.5) * pi * cloud_radius() * cloud_radius();
    Real const cloudy_area =
        composite_volume_sum(cloud_fraction, hierarchy, masks);
    result.cloudy_area_relative_error =
        std::abs(cloudy_area - expected_cloudy_area) / expected_cloudy_area;
    Real const cloudy_area_tolerance =
        (sizeof(Real) == sizeof(float)) ? Real(2.e-4) : Real(2.e-12);
    AMREX_ALWAYS_ASSERT(result.cloudy_area_relative_error <
                        cloudy_area_tolerance);
    AMREX_ALWAYS_ASSERT(result.mixed_cells > 0);

    PhysicalBoundaryData physical_boundary;
    physical_boundary.lo = {AMREX_D_DECL(LinOpBCType::Neumann,
                                          LinOpBCType::Robin,
                                          LinOpBCType::Neumann)};
    physical_boundary.hi = physical_boundary.lo;
    physical_boundary.lo_value = {AMREX_D_DECL(Real(0), Real(0), Real(0))};
    physical_boundary.hi_value = {AMREX_D_DECL(Real(0), Real(4), Real(0))};
    AMG<Real>::Options options;
    options.priority_seed = 1;
    MLABecLapAMG solver(hierarchy.geom, hierarchy.grids, hierarchy.dmap,
                        options);
    Real const nonlinear_tolerance =
        (sizeof(Real) == sizeof(float)) ? Real(2.e-4) : Real(2.e-6);
    Real constexpr incident_marshak_flux = Real(1);
    CloudNewtonProblem problem(hierarchy, masks, extinction, rhs, acoef,
                               physical_boundary, limited, solver,
                               result.solver);
    for (int predictor = 0; predictor < 20; ++predictor) {
        if (problem.predict(state, Real(0.7)) < Real(1.e-3)) {
            break;
        }
    }
    NewtonKrylovOptions newton_options;
    newton_options.nonlinear_tolerance = nonlinear_tolerance;
    newton_options.maximum_nonlinear_iterations = 100;
    newton_options.maximum_linear_iterations = 300;
    newton_options.krylov_restart_length = 100;
    newton_options.maximum_line_search_iterations = 24;
    newton_options.linear_verbosity = iteration_output ? 2 : 0;
    newton_options.centered_difference = false;
    newton_options.problem_name = "The cloud-layer FLD system";
    NewtonKrylovSolver<CloudNewtonProblem> newton(problem, newton_options);
    auto const nonlinear = newton.solve(state);
    result.final_nonlinear_change = nonlinear.final_change;
    result.final_nonlinear_residual = nonlinear.final_residual_norm;
    result.nonlinear_iterations = nonlinear.nonlinear_iterations;
    result.total_newton_krylov_iterations =
        nonlinear.total_linear_iterations;
    result.maximum_newton_krylov_iterations =
        nonlinear.maximum_linear_iterations;
    result.solver.solves += nonlinear.nonlinear_iterations;
    result.solver.total_iterations += nonlinear.total_linear_iterations;
    result.solver.maximum_iterations = amrex::max(
        result.solver.maximum_iterations,
        nonlinear.maximum_linear_iterations);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        nonlinear.converged,
        "The cloud-layer FLD nonlinear iteration did not converge");

    compute_diffusion(hierarchy, state, extinction, diffusion,
                      physical_boundary, limited);
    auto const [bottom_flux, top_flux] =
        cloud_boundary_fluxes(hierarchy, state, diffusion, masks);
    result.transmission = bottom_flux / incident_marshak_flux;
    result.balance_error = std::abs(bottom_flux + top_flux) /
                           amrex::max(std::abs(bottom_flux), Real(1.e-30));
    auto const [minimum, maximum] =
        composite_minimum_maximum(state, masks);
    result.minimum_energy = minimum;
    result.maximum_energy = maximum;
    if (iteration_output) {
        amrex::Print()
            << "FLD cloud " << (use_amr ? "AMR" : "uniform")
            << " Newton-Krylov iterations=" << result.nonlinear_iterations
            << "/" << result.total_newton_krylov_iterations
            << ", change=" << result.final_nonlinear_change
            << ", residual=" << result.final_nonlinear_residual
            << ", transmission=" << result.transmission << '\n';
    }
    if (!plotfile_name.empty()) {
        write_cloud_plotfile(plotfile_name, hierarchy, state, extinction,
                             diffusion, cloud_fraction);
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
