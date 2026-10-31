#include "RadiationMaterial.H"

#include "MLABecAMG.H"
#include "NewtonKrylovSolver.H"

#include <AMReX_MFIter.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Reduce.H>

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

namespace fld_test
{

using namespace amrex;

namespace
{

Real
linear_tolerance ()
{
    return (sizeof(Real) == sizeof(float)) ? Real(5.e-5) : Real(2.e-10);
}

void
record_setup (SolverSummary& summary, MLABecLapAMG const& solver)
{
    if (summary.preconditioner.empty()) {
        summary.preconditioner = solver.preconditioner_name();
        summary.amg_backend = solver.amg_backend_name();
    }
    AMREX_ALWAYS_ASSERT(summary.preconditioner ==
                        solver.preconditioner_name());
    AMREX_ALWAYS_ASSERT(summary.amg_backend == solver.amg_backend_name());
    ++summary.setups;
    summary.setup_seconds += solver.setup_seconds();
    if (solver.preconditioner() == MLABecPreconditioner::AMG &&
        solver.amg_backend() == MLABecAMGBackend::Native) {
        auto const& diagnostics = solver.diagnostics();
        summary.maximum_levels = amrex::max(
            summary.maximum_levels,
            diagnostics.levels);
        summary.maximum_operator_complexity = amrex::max(
            summary.maximum_operator_complexity,
            diagnostics.operator_complexity);
    }
}

void
record_solve (SolverSummary& summary, SolveInfo const& solution)
{
    ++summary.solves;
    summary.total_iterations += solution.iterations;
    summary.maximum_iterations =
        amrex::max(summary.maximum_iterations, solution.iterations);
    summary.maximum_relative_residual = amrex::max(
        summary.maximum_relative_residual, solution.relative_residual);
    summary.solve_seconds += solution.solve_seconds;
    summary.preconditioner_seconds += solution.preconditioner_seconds;
}

int constexpr rm_periodic = 0;
int constexpr rm_neumann = 1;
int constexpr rm_robin = 2;

int constexpr rm_face_limiter = 0;
int constexpr rm_cell_limiter = 1;
int constexpr rm_cell_limiter_surface = 2;

// Everything the kernels need, copied into device lambdas by value.  dx and
// the domain bounds are those of the level being processed.
struct RMPhysics
{
    Real opacity_power = Real(3);
    Real temperature_floor = Real(1.e-6);
    GpuArray<Real, 3> velocity{};
    GpuArray<Real, 3> dx{};
    Real dt = Real(1);
    bool limiter = true;
    bool moving = false;
    int diffusion = rm_face_limiter;
    bool robin_b_diffusion = false;
    GpuArray<int, 6> boundary{};
    GpuArray<Real, 6> robin_a{};
    GpuArray<Real, 6> incoming{};
    GpuArray<int, 3> domain_lo{};
    GpuArray<int, 3> domain_hi{};
};

// Face diffusion coefficients of one box, by direction.
using RMFaces = GpuArray<Array4<Real const>, 3>;

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE Real
rm_positive (RMPhysics const& p, Real t) noexcept
{
    return amrex::max(t, p.temperature_floor);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE Real
rm_fourth (Real t) noexcept
{
    Real const t2 = t * t;
    return t2 * t2;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE Real
rm_dot (GpuArray<Real, 3> const& a, GpuArray<Real, 3> const& b) noexcept
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

// kappa T^-p, avoiding pow() for small integer p: the cell-local solves
// evaluate this on every iteration.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE Real
rm_opacity (RMPhysics const& p, Real kappa, Real t) noexcept
{
    Real const tp = rm_positive(p, t);
    int const integer_power = static_cast<int>(p.opacity_power);
    if (Real(integer_power) == p.opacity_power && integer_power >= 0 &&
        integer_power <= 8) {
        Real denominator = Real(1);
        for (int n = 0; n < integer_power; ++n) { denominator *= tp; }
        return kappa / denominator;
    }
    return kappa * std::pow(tp, -p.opacity_power);
}

// k = d/dT [sigma (T^4 - E)].  k < 0 where absorption heating raises the
// opacity enough to run away (opacity_power < 0 with E > T^4, or > 4 with
// E < T^4).  Zero below the rm_positive floor, where the residual does not
// depend on T (as in eliminate()).
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE Real
rm_coupling (RMPhysics const& p, Real kappa, Real e, Real t) noexcept
{
    if (!(t > p.temperature_floor)) { return Real(0); }
    Real const tc = t;
    Real const sigma = rm_opacity(p, kappa, tc);
    Real const t3 = tc * tc * tc;
    return -p.opacity_power * sigma / tc * (tc * t3 - e) +
           Real(4) * sigma * t3;
}

// The data of one cell's material equation g(T) = 0 (see
// rm_material_residual); E is clamped at zero.
struct RMCellData
{
    Real kappa;
    Real energy;
    Real work; // v . (lagged flux)
    Real old_temperature;
    Real source;
};

// g(T) = (T - T^n)/dt - sigma (E - T^4) + 2 sigma work - S_T, and dg/dT in
// `slope`.  sigma and T^4 are flat below the rm_positive floor.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE Real
rm_material_residual (RMPhysics const& p, Real inverse_dt,
                      RMCellData const& c, Real temperature,
                      Real& slope) noexcept
{
    Real const tp = rm_positive(p, temperature);
    Real const sigma = rm_opacity(p, c.kappa, temperature);
    Real const dsigma = temperature > p.temperature_floor
        ? -p.opacity_power * sigma / tp : Real(0);
    Real const t4 = rm_fourth(tp);
    Real const dt4 = temperature > p.temperature_floor
        ? Real(4) * tp * tp * tp : Real(0);
    slope = inverse_dt - dsigma * (c.energy - t4) + sigma * dt4 +
            Real(2) * dsigma * c.work;
    return inverse_dt * (temperature - c.old_temperature) -
           sigma * (c.energy - t4) + Real(2) * sigma * c.work - c.source;
}

// dT/dE = -g_E/g_T at the root x, with g_E = -sigma(T).  tau enters the
// reaction coefficient (1 + tau)/dt of the ReducedProblem preconditioner
// and of the quasi-Newton linear solve, which AMG needs positive.  Where
// g_T lies in [-sigma, 0] (runaway opacity) 1 + tau is nonpositive or
// infinite, so drop the derivative there; it is exact everywhere else.
// NaN g_T gives 0.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE Real
rm_temperature_derivative (RMPhysics const& p, Real inverse_dt,
                           RMCellData const& c, Real x) noexcept
{
    Real gp;
    rm_material_residual(p, inverse_dt, c, x, gp);
    Real const sigma = rm_opacity(p, c.kappa, x);
    return gp > Real(0) || gp < -sigma ? sigma / gp : Real(0);
}

// Levermore-Pomraning flux limiter lambda(R).
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE Real
rm_levermore (Real r) noexcept
{
    if (r > Real(1.e8)) { return Real(1) / r; }
    return (Real(2) + r) / (Real(6) + Real(3) * r + r * r);
}

// FaceLimiter: the harmonic mean of each side's 1/(3 sigma + |grad E|/E),
// with the opacity at the mean face temperature.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE Real
rm_face_diffusion (RMPhysics const& p, Real el, Real er, Real tl, Real tr,
                   Real kl, Real kr, Real h) noexcept
{
    Real const tf = Real(0.5) * (rm_positive(p, tl) + rm_positive(p, tr));
    Real const limiter = p.limiter
        ? std::abs(er - el) / h /
              amrex::max(Real(0.5) * (el + er), Real(1.e-30))
        : Real(0);
    Real const dl = Real(1) / (Real(3) * rm_opacity(p, kl, tf) + limiter);
    Real const dr = Real(1) / (Real(3) * rm_opacity(p, kr, tf) + limiter);
    return Real(2) * dl * dr / (dl + dr);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE bool
rm_is_boundary_face (RMPhysics const& p, IntVect const& c, int d,
                     int side) noexcept
{
    if (p.boundary[2 * d + side] == rm_periodic) { return false; }
    return side == 0 ? c[d] == p.domain_lo[d] : c[d] == p.domain_hi[d];
}

// For a ghost cell g outside the domain across exactly one non-periodic
// face, that face (2 d + side) and the boundary cell inside it; -1
// otherwise.  Robin data live in these exterior ghost cells, one value per
// face, so the faces of edge and corner cells are independent.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE int
rm_exterior_face (RMPhysics const& p, IntVect const& g, IntVect& inside) noexcept
{
    int face = -1;
    for (int d = 0; d < 3; ++d) {
        int side = -1;
        if (g[d] < p.domain_lo[d]) { side = 0; }
        if (g[d] > p.domain_hi[d]) { side = 1; }
        if (side < 0) { continue; }
        if (face >= 0 || p.boundary[2 * d + side] == rm_periodic) {
            return -1;
        }
        face = 2 * d + side;
    }
    if (face >= 0) {
        int const d = face / 2;
        inside = g;
        inside[d] = face % 2 == 0 ? p.domain_lo[d] : p.domain_hi[d];
    }
    return face;
}

// Cell lambda(R)/sigma for the cell limiters, R = |grad E|/(sigma E) from
// central differences.  Across a Robin face the outside value is reflected
// about the equilibrium incoming/a; across a Neumann face it is the cell's.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE Real
rm_cell_diffusion (RMPhysics const& p, Array4<Real const> const& e,
                   Array4<Real const> const& t, Array4<Real const> const& k,
                   IntVect const& c) noexcept
{
    Real const sigma = rm_opacity(p, k(c), t(c));
    Real lambda = Real(1) / Real(3);
    if (p.limiter) {
        Real gradient_squared = Real(0);
        for (int d = 0; d < 3; ++d) {
            IntVect const s = IntVect::TheDimensionVector(d);
            Real value[2];
            for (int side = 0; side < 2; ++side) {
                int const face = 2 * d + side;
                if (rm_is_boundary_face(p, c, d, side)) {
                    value[side] = p.boundary[face] == rm_robin
                        ? Real(2) * p.incoming[face] / p.robin_a[face] - e(c)
                        : e(c);
                } else {
                    value[side] = side == 0 ? e(c - s) : e(c + s);
                }
            }
            Real const g = (value[1] - value[0]) / (Real(2) * p.dx[d]);
            gradient_squared += g * g;
        }
        Real const r = std::sqrt(gradient_squared) /
                       (sigma * amrex::max(e(c), Real(1.e-30)));
        lambda = rm_levermore(r);
    }
    return lambda / sigma;
}

// Diffusion coefficient on `face` (normal to d).  At a non-periodic domain
// face the outside cell is replaced by the inside one.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE Real
rm_face_coefficient (RMPhysics const& p, Array4<Real const> const& e,
                     Array4<Real const> const& t, Array4<Real const> const& k,
                     Array4<Real const> const& dcell, IntVect const& face,
                     int d) noexcept
{
    IntVect lo = face - IntVect::TheDimensionVector(d);
    IntVect hi = face;
    bool boundary = false;
    if (p.boundary[2 * d] != rm_periodic) {
        if (face[d] == p.domain_lo[d]) { lo = hi; boundary = true; }
        if (face[d] == p.domain_hi[d] + 1) { hi = lo; boundary = true; }
    }
    if (p.diffusion == rm_face_limiter) {
        return rm_face_diffusion(p, e(lo), e(hi), t(lo), t(hi), k(lo), k(hi),
                                 p.dx[d]);
    }
    Real const dl = dcell(lo);
    Real const dr = dcell(hi);
    if (p.diffusion == rm_cell_limiter || boundary) {
        return Real(2) * dl * dr / (dl + dr);
    }
    Real const sl = rm_opacity(p, k(lo), t(lo));
    Real const sr = rm_opacity(p, k(hi), t(hi));
    Real const arithmetic = Real(0.5) * (sl + sr);
    Real const harmonic =
        sl + sr > Real(0) ? Real(2) * sl * sr / (sl + sr) : Real(0);
    Real const surface = Real(4) / (Real(3) * p.dx[d]);
    Real const face_sigma =
        amrex::min(arithmetic, amrex::max(harmonic, surface));
    // D sigma is the cell's lambda; a transparent cell carries none.
    Real const face_lambda = amrex::max(sl > Real(0) ? dl * sl : Real(0),
                                        sr > Real(0) ? dr * sr : Real(0));
    return face_lambda / amrex::max(face_sigma, Real(1.e-30));
}

// Flux out of the domain through the Robin face (d, side) of a boundary
// cell: a linear profile to the face in a E + b dE/dn = incoming gives
// F_out = D (a E - incoming)/(b + a h/2).  Zero for Neumann.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE Real
rm_robin_outflow (RMPhysics const& p, Real face_diffusion, Real e, Real b,
                  int d, int side) noexcept
{
    int const face = 2 * d + side;
    if (p.boundary[face] != rm_robin) { return Real(0); }
    Real const a = p.robin_a[face];
    return face_diffusion * (a * e - p.incoming[face]) /
           (b + a * Real(0.5) * p.dx[d]);
}

// Comoving flux F0 (positive in +d) on the face of cell c on `side`.  The
// Robin b of a domain face is in the exterior ghost cell across it.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE Real
rm_face_flux (RMPhysics const& p, Array4<Real const> const& e,
              RMFaces const& b, Array4<Real const> const& rb,
              IntVect const& c, int d, int side) noexcept
{
    IntVect const s = IntVect::TheDimensionVector(d);
    if (rm_is_boundary_face(p, c, d, side)) {
        Real const out = rm_robin_outflow(
            p, b[d](side == 0 ? c : c + s), e(c),
            rb(side == 0 ? c - s : c + s), d, side);
        return side == 0 ? -out : out;
    }
    if (side == 0) { return -b[d](c) * (e(c) - e(c - s)) / p.dx[d]; }
    return -b[d](c + s) * (e(c + s) - e(c)) / p.dx[d];
}

// Cell-centered F0: the mean of the two face fluxes in each direction.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE GpuArray<Real, 3>
rm_cell_flux (RMPhysics const& p, Array4<Real const> const& e,
              RMFaces const& b, Array4<Real const> const& rb,
              IntVect const& c) noexcept
{
    GpuArray<Real, 3> flux{};
    for (int d = 0; d < 3; ++d) {
        flux[d] = Real(0.5) * (rm_face_flux(p, e, b, rb, c, d, 0) +
                               rm_face_flux(p, e, b, rb, c, d, 1));
    }
    return flux;
}

// Radiation enthalpy flux v.(E I + P) at a cell center, with the Levermore
// tensor P = E[(1 - f)/2 I + (3f - 1)/2 n n], f = lambda + (lambda R)^2, and
// the lambda(R) of the diffusion coefficient (1/(3 + R) for FaceLimiter).
// Without the limiter f = 1/3 and this is (4/3) v E.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE GpuArray<Real, 3>
rm_enthalpy_flux (RMPhysics const& p, Array4<Real const> const& e,
                  Array4<Real const> const& t, Array4<Real const> const& k,
                  int i, int j, int kk) noexcept
{
    Real const ec = e(i, j, kk);
    Real f = Real(1) / Real(3);
    GpuArray<Real, 3> n{};
    if (p.limiter) {
        Real const g[3] = {
            (e(i + 1, j, kk) - e(i - 1, j, kk)) / (Real(2) * p.dx[0]),
            (e(i, j + 1, kk) - e(i, j - 1, kk)) / (Real(2) * p.dx[1]),
            (e(i, j, kk + 1) - e(i, j, kk - 1)) / (Real(2) * p.dx[2])};
        Real const gmag = std::sqrt(g[0] * g[0] + g[1] * g[1] + g[2] * g[2]);
        Real const r = gmag / (rm_opacity(p, k(i, j, kk), t(i, j, kk)) *
                               amrex::max(ec, Real(1.e-30)));
        Real const lambda = p.diffusion == rm_face_limiter
            ? Real(1) / (Real(3) + r) : rm_levermore(r);
        f = lambda + lambda * lambda * r * r;
        if (gmag > Real(0)) {
            for (int d = 0; d < 3; ++d) { n[d] = g[d] / gmag; }
        }
    }
    Real const vn = rm_dot(p.velocity, n);
    GpuArray<Real, 3> a{};
    for (int d = 0; d < 3; ++d) {
        a[d] = ec * (p.velocity[d] +
                     Real(0.5) * (Real(1) - f) * p.velocity[d] +
                     Real(0.5) * (Real(3) * f - Real(1)) * vn * n[d]);
    }
    return a;
}

// Upwind face value of a cell-centered advective flux component.  (Centered
// values with the explicit forward-Euler material update are unstable once
// radiation diffusion no longer damps grid-scale modes.)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE Real
rm_face_advection (RMPhysics const& p, int d, Real left, Real right) noexcept
{
    return p.velocity[d] >= Real(0) ? left : right;
}

enum class WorkCoupling { Current, Lagged };

LinOpBCType
rm_linop_type (RadiationBoundary boundary)
{
    switch (boundary) {
    case RadiationBoundary::Periodic: return LinOpBCType::Periodic;
    case RadiationBoundary::Neumann: return LinOpBCType::Neumann;
    case RadiationBoundary::Robin: return LinOpBCType::Robin;
    }
    return LinOpBCType::Neumann;
}

int
rm_diffusion_code (DiffusionCoefficient diffusion)
{
    switch (diffusion) {
    case DiffusionCoefficient::FaceLimiter: return rm_face_limiter;
    case DiffusionCoefficient::CellLimiter: return rm_cell_limiter;
    case DiffusionCoefficient::CellLimiterSurfaceOpacity:
        return rm_cell_limiter_surface;
    }
    return rm_face_limiter;
}

RMFaces
rm_faces (FaceData const& bcoef, int level, MFIter const& mfi)
{
    return {bcoef[level][0]->const_array(mfi),
            bcoef[level][1]->const_array(mfi),
            bcoef[level][2]->const_array(mfi)};
}

} // namespace

struct RadiationMaterialSolver::Impl
{
    Impl (DiffusionHierarchy const& a_hierarchy,
          RadiationMaterialOptions a_options)
        : hierarchy(a_hierarchy), options(std::move(a_options)),
          masks_(make_composite_masks(a_hierarchy)),
          solver(a_hierarchy.geom, a_hierarchy.grids, a_hierarchy.dmap),
          kappa(make_cell_data(a_hierarchy, 1, 1)),
          old_energy(make_cell_data(a_hierarchy, 1, 0)),
          old_temperature(make_cell_data(a_hierarchy, 1, 0)),
          source_energy(make_cell_data(a_hierarchy, 1, 0)),
          source_temperature(make_cell_data(a_hierarchy, 1, 0)),
          energy(make_cell_data(a_hierarchy, 1, 2)),
          temperature(make_cell_data(a_hierarchy, 1, 2)),
          eliminated_temperature(make_cell_data(a_hierarchy, 1, 2)),
          cell_diffusion(make_cell_data(a_hierarchy, 1, 1)),
          enthalpy_flux(make_cell_data(a_hierarchy, 3, 1)),
          lagged_flux(make_cell_data(a_hierarchy, 3, 0)),
          residual_energy(make_cell_data(a_hierarchy, 1, 0)),
          residual_temperature(make_cell_data(a_hierarchy, 1, 0)),
          temperature_derivative(make_cell_data(a_hierarchy, 1, 0)),
          acoef(make_cell_data(a_hierarchy, 1, 0)),
          schur_k(make_cell_data(a_hierarchy, 1, 0)),
          schur_sigma(make_cell_data(a_hierarchy, 1, 0)),
          schur_diagonal(make_cell_data(a_hierarchy, 1, 0)),
          robin_a(make_cell_data(a_hierarchy, 1, 1)),
          robin_b(make_cell_data(a_hierarchy, 1, 1)),
          robin_f(make_cell_data(a_hierarchy, 1, 1)),
          robin_zero(make_cell_data(a_hierarchy, 1, 1)),
          zero(make_cell_data(a_hierarchy, 1, 0)),
          bcoef(make_face_data(a_hierarchy))
    {
        // Linear-solve failures fail the step attempt; see attempt_step.
        solver.setAbortOnFailure(false);
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            options.solver == "newton_krylov" ||
                options.solver == "ne_newton_krylov" ||
                options.solver == "ne_quasi_newton",
            "solver must be newton_krylov, ne_newton_krylov, or "
            "ne_quasi_newton");
        for (int d = 0; d < 3; ++d) {
            physics.velocity[d] = options.velocity[d];
            bool const periodic_lo =
                options.boundary[2 * d] == RadiationBoundary::Periodic;
            bool const periodic_hi =
                options.boundary[2 * d + 1] == RadiationBoundary::Periodic;
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                periodic_lo == periodic_hi &&
                    periodic_lo == hierarchy.geom[0].isPeriodic(d),
                "Periodic boundaries must match the geometry on both sides");
            physics.moving = physics.moving || options.velocity[d] != Real(0);
            for (int side = 0; side < 2; ++side) {
                int const face = 2 * d + side;
                RadiationBoundary const type = options.boundary[face];
                physics.boundary[face] =
                    type == RadiationBoundary::Periodic ? rm_periodic
                    : type == RadiationBoundary::Neumann ? rm_neumann
                                                         : rm_robin;
                physics.robin_a[face] = options.robin_a[face];
                physics.incoming[face] = options.incoming[face];
                if (type == RadiationBoundary::Robin) {
                    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                        options.robin_a[face] > Real(0),
                        "Robin a must be positive");
                    has_robin = true;
                }
            }
            lobc[d] = rm_linop_type(options.boundary[2 * d]);
            hibc[d] = rm_linop_type(options.boundary[2 * d + 1]);
        }
        if (physics.moving) {
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                hierarchy.geom[0].isAllPeriodic() && levels() == 1,
                "A nonzero velocity needs a periodic domain and one level");
        }
        physics.opacity_power = options.opacity_power;
        {
            ParmParse pp("radiation_material");
            pp.query("temperature_floor", options.temperature_floor);
        }
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(options.temperature_floor > Real(0),
                                         "temperature_floor must be positive");
        physics.temperature_floor = options.temperature_floor;
        physics.limiter = options.limiter;
        physics.diffusion = rm_diffusion_code(options.diffusion);
        physics.robin_b_diffusion =
            options.robin_b == RobinCoefficient::Diffusion;
        set_level_data(kappa, Real(1));
        set_level_data(source_energy, Real(0));
        set_level_data(source_temperature, Real(0));
        set_level_data(lagged_flux, Real(0));
        set_level_data(acoef, Real(0));
        set_level_data(cell_diffusion, Real(0));
        set_level_data(zero, Real(0));
        set_level_data(robin_zero, Real(0));
        // Each Robin face's a and right-hand side in its exterior ghost
        // cells.
        set_level_data(robin_a, Real(1));
        set_level_data(robin_b, Real(1));
        set_level_data(robin_f, Real(0));
        for (int level = 0; level < levels(); ++level) {
            RMPhysics const p = level_physics(level);
            for (MFIter mfi(*robin_f[level]); mfi.isValid(); ++mfi) {
                auto const a = robin_a[level]->array(mfi);
                auto const f = robin_f[level]->array(mfi);
                ParallelFor(mfi.fabbox(),
                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    IntVect const g(AMREX_D_DECL(i, j, k));
                    IntVect inside;
                    int const face = rm_exterior_face(p, g, inside);
                    if (face >= 0 && p.boundary[face] == rm_robin) {
                        a(g) = p.robin_a[face];
                        f(g) = p.incoming[face];
                    }
                });
            }
        }
    }

    DiffusionHierarchy const& hierarchy;
    RadiationMaterialOptions options;
    Vector<iMultiFab> masks_;
    RMPhysics physics;
    bool has_robin = false;
    Array<LinOpBCType, AMREX_SPACEDIM> lobc{};
    Array<LinOpBCType, AMREX_SPACEDIM> hibc{};
    MLABecLapAMG solver;
    RadiationMaterialCounters counters;
    // coupled_residual_norm at the accepted state, if attempt_step already
    // evaluated it there; negative otherwise.
    Real accepted_coupled_residual = Real(-1);
    // Lag iterations in this step, over all attempts (like step_nonlinear).
    int step_lag = 0;
    // Bumped whenever the loaded state, T(E), the lagged flux, or the
    // residual changes, so a problem can tell whether the work fields
    // still hold its last residual().
    long generation = 0;

    LevelData kappa;
    LevelData old_energy;      // E^n
    LevelData old_temperature; // T^n after explicit material advection
    LevelData source_energy;
    LevelData source_temperature;
    LevelData energy;          // work copies with halos
    LevelData temperature;
    LevelData eliminated_temperature; // T(E), also the warm start
    LevelData cell_diffusion;  // cell D of the cell limiters
    LevelData enthalpy_flux;
    LevelData lagged_flux;
    LevelData residual_energy;
    LevelData residual_temperature;
    LevelData temperature_derivative;
    LevelData acoef;
    LevelData schur_k;
    LevelData schur_sigma;
    LevelData schur_diagonal;
    LevelData robin_a;         // Robin data in the exterior ghost cells
    LevelData robin_b;
    LevelData robin_f;
    LevelData robin_zero;
    LevelData zero;
    FaceData bcoef;            // face D of the loaded state

    [[nodiscard]] int levels () const
    {
        return static_cast<int>(hierarchy.geom.size());
    }

    [[nodiscard]] RMPhysics level_physics (int level) const
    {
        RMPhysics p = physics;
        Box const domain = hierarchy.geom[level].Domain();
        for (int d = 0; d < 3; ++d) {
            p.dx[d] = hierarchy.geom[level].CellSize(d);
            p.domain_lo[d] = domain.smallEnd(d);
            p.domain_hi[d] = domain.bigEnd(d);
        }
        return p;
    }

    Real scale (LevelData const& field, Real floor) const
    {
        return amrex::max(composite_maximum(field, masks_), floor);
    }

    // Copy (E, T) into the halo-carrying work fields (averaged down, with
    // periodic images, coarse-fine interpolation, or constant extrapolation)
    // and evaluate the face diffusion coefficients and Robin data; for a
    // moving medium also the cell enthalpy fluxes with one halo layer for
    // the face upwinding.
    void load_state (LevelData const& e, LevelData const& t)
    {
        copy_level_data(energy, e);
        load_loaded_energy(t);
    }

    // load_state with E already copied into `energy`, as eliminate()
    // leaves it.
    void load_loaded_energy (LevelData const& t)
    {
        ++generation;
        copy_level_data(temperature, t);
        average_down_hierarchy(energy, hierarchy);
        average_down_hierarchy(temperature, hierarchy);
        fill_level_ghosts(energy, hierarchy);
        fill_level_ghosts(temperature, hierarchy);
        update_coefficients();
        if (!physics.moving) { return; }
        RMPhysics const p = level_physics(0);
        for (MFIter mfi(*enthalpy_flux[0]); mfi.isValid(); ++mfi) {
            auto const ea = energy[0]->const_array(mfi);
            auto const ta = temperature[0]->const_array(mfi);
            auto const ka = kappa[0]->const_array(mfi);
            auto const a = enthalpy_flux[0]->array(mfi);
            ParallelFor(mfi.growntilebox(1),
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                auto const flux = rm_enthalpy_flux(p, ea, ta, ka, i, j, k);
                for (int d = 0; d < 3; ++d) { a(i, j, k, d) = flux[d]; }
            });
        }
    }

    void update_coefficients ()
    {
        bool const need_cell_diffusion =
            physics.diffusion != rm_face_limiter || physics.robin_b_diffusion;
        if (need_cell_diffusion) {
            for (int level = 0; level < levels(); ++level) {
                RMPhysics const p = level_physics(level);
                for (MFIter mfi(*cell_diffusion[level]); mfi.isValid(); ++mfi) {
                    auto const e = energy[level]->const_array(mfi);
                    auto const t = temperature[level]->const_array(mfi);
                    auto const k = kappa[level]->const_array(mfi);
                    auto const dc = cell_diffusion[level]->array(mfi);
                    ParallelFor(mfi.validbox(),
                    [=] AMREX_GPU_DEVICE (int i, int j, int kk) noexcept
                    {
                        IntVect const c(AMREX_D_DECL(i, j, kk));
                        dc(c) = rm_cell_diffusion(p, e, t, k, c);
                    });
                }
            }
            fill_level_ghosts(cell_diffusion, hierarchy);
        }
        for (int level = 0; level < levels(); ++level) {
            RMPhysics const p = level_physics(level);
            for (int d = 0; d < 3; ++d) {
                for (MFIter mfi(*bcoef[level][d]); mfi.isValid(); ++mfi) {
                    auto const e = energy[level]->const_array(mfi);
                    auto const t = temperature[level]->const_array(mfi);
                    auto const k = kappa[level]->const_array(mfi);
                    auto const dc = cell_diffusion[level]->const_array(mfi);
                    auto const b = bcoef[level][d]->array(mfi);
                    ParallelFor(mfi.validbox(),
                    [=] AMREX_GPU_DEVICE (int i, int j, int kk) noexcept
                    {
                        IntVect const face(AMREX_D_DECL(i, j, kk));
                        b(face) = rm_face_coefficient(p, e, t, k, dc, face, d);
                    });
                }
            }
            if (!has_robin) { continue; }
            // b of each Robin face from the boundary cell inside it, into
            // the face's exterior ghost cell.  Every box covering a ghost
            // cell computes it (from its own halo), so the copies agree.
            for (MFIter mfi(*robin_b[level]); mfi.isValid(); ++mfi) {
                auto const t = temperature[level]->const_array(mfi);
                auto const k = kappa[level]->const_array(mfi);
                auto const dc = cell_diffusion[level]->const_array(mfi);
                auto const rb = robin_b[level]->array(mfi);
                ParallelFor(mfi.fabbox(),
                [=] AMREX_GPU_DEVICE (int i, int j, int kk) noexcept
                {
                    IntVect const g(AMREX_D_DECL(i, j, kk));
                    IntVect n;
                    int const face = rm_exterior_face(p, g, n);
                    if (face < 0 || p.boundary[face] != rm_robin) { return; }
                    rb(g) = p.robin_b_diffusion
                        ? dc(n)
                        : Real(1) / (Real(6) * rm_opacity(p, k(n), t(n)));
                });
            }
        }
    }

    [[nodiscard]] RobinBCData robin_data (bool homogeneous) const
    {
        if (!has_robin) { return {}; }
        return RobinBCData{get_level_const_ptrs(robin_a),
                           get_level_const_ptrs(robin_b),
                           get_level_const_ptrs(homogeneous ? robin_zero
                                                            : robin_f)};
    }

    // Residuals of both equations at the loaded state, zero on covered
    // cells.  div F0 comes from the composite diffusion operator; the
    // material work term uses the current flux, or the lagged flux.
    void evaluate_residual (WorkCoupling coupling)
    {
        ++generation;
        solver.residualWithoutAssembly(
            get_level_ptrs(residual_energy), get_level_const_ptrs(energy),
            get_level_const_ptrs(zero), Real(0), Real(1),
            get_level_const_ptrs(acoef), get_face_const_ptrs(bcoef), lobc,
            hibc, {}, robin_data(false));
        bool const lagged = coupling == WorkCoupling::Lagged;
        for (int level = 0; level < levels(); ++level) {
            RMPhysics const p = level_physics(level);
            Real const inverse_dt = Real(1) / p.dt;
            for (MFIter mfi(*residual_energy[level]); mfi.isValid(); ++mfi) {
                auto const e = energy[level]->const_array(mfi);
                auto const t = temperature[level]->const_array(mfi);
                auto const k = kappa[level]->const_array(mfi);
                auto const a = enthalpy_flux[level]->const_array(mfi);
                auto const b = rm_faces(bcoef, level, mfi);
                auto const rb = robin_b[level]->const_array(mfi);
                auto const eo = old_energy[level]->const_array(mfi);
                auto const to = old_temperature[level]->const_array(mfi);
                auto const se = source_energy[level]->const_array(mfi);
                auto const st = source_temperature[level]->const_array(mfi);
                auto const lag = lagged_flux[level]->const_array(mfi);
                auto const mask = masks_[level].const_array(mfi);
                auto const re = residual_energy[level]->array(mfi);
                auto const rt = residual_temperature[level]->array(mfi);
                ParallelFor(mfi.validbox(),
                [=] AMREX_GPU_DEVICE (int i, int j, int kk) noexcept
                {
                    IntVect const c(AMREX_D_DECL(i, j, kk));
                    if (!mask(c)) {
                        re(c) = Real(0);
                        rt(c) = Real(0);
                        return;
                    }
                    Real const ec = e(c);
                    Real const tc = t(c);
                    Real divergence = re(c);
                    Real work = Real(0);
                    Real material_work = Real(0);
                    if (p.moving) {
                        auto const flux = rm_cell_flux(p, e, b, rb, c);
                        work = rm_dot(p.velocity, flux);
                        GpuArray<Real, 3> const lagged_flux{
                            lag(i, j, kk, 0), lag(i, j, kk, 1),
                            lag(i, j, kk, 2)};
                        material_work =
                            lagged ? rm_dot(p.velocity, lagged_flux) : work;
                        for (int d = 0; d < 3; ++d) {
                            IntVect const s = IntVect::TheDimensionVector(d);
                            Real const f_lo =
                                rm_face_advection(p, d, a(c - s, d), a(c, d));
                            Real const f_hi =
                                rm_face_advection(p, d, a(c, d), a(c + s, d));
                            divergence += (f_hi - f_lo) / p.dx[d];
                        }
                    }
                    Real const sigma = rm_opacity(p, k(c), tc);
                    Real const t4 = rm_fourth(rm_positive(p, tc));
                    re(c) = inverse_dt * (ec - eo(c)) + divergence -
                            sigma * (t4 - ec) - sigma * work - se(c);
                    rt(c) = inverse_dt * (tc - to(c)) - sigma * (ec - t4) +
                            Real(2) * sigma * material_work - st(c);
                });
            }
        }
    }

    // Store the current cell flux F0 as the lagged work flux.
    void freeze_work_flux ()
    {
        ++generation;
        RMPhysics const p = level_physics(0);
        for (MFIter mfi(*lagged_flux[0]); mfi.isValid(); ++mfi) {
            auto const e = energy[0]->const_array(mfi);
            auto const b = rm_faces(bcoef, 0, mfi);
            auto const rb = robin_b[0]->const_array(mfi);
            auto const lag = lagged_flux[0]->array(mfi);
            ParallelFor(mfi.validbox(),
            [=] AMREX_GPU_DEVICE (int i, int j, int kk) noexcept
            {
                IntVect const c(AMREX_D_DECL(i, j, kk));
                auto const flux = rm_cell_flux(p, e, b, rb, c);
                for (int d = 0; d < 3; ++d) { lag(i, j, kk, d) = flux[d]; }
            });
        }
    }

    // T(E): cell-local material solves for the energy in `e`, with the work
    // term from the lagged flux.  g(T) is solved with a bracketed Newton
    // iteration (analytic slope, bisection fallback), starting from and
    // overwriting eliminated_temperature; optionally returns dT/dE.
    // Finite-difference Jacobian probes can make E negative in cold cells,
    // where g may have no positive root, so the solve absorbs max(E, 0).
    // False when a cell-local solve did not converge (on any rank).
    [[nodiscard]] bool eliminate (LevelData const& e, bool with_derivative)
    {
        BL_PROFILE("FLD::rm_eliminate");
        ++generation;
        copy_level_data(energy, e);
        ++counters.eliminations;
        RMPhysics const p = physics;
        Real const inverse_dt = Real(1) / p.dt;
        int constexpr maximum_iterations = 100;
        // Relative; 1e-13 in double, a few ulps in single precision.
        Real constexpr tolerance = std::max(
            Real(1.e-13), Real(100) * std::numeric_limits<Real>::epsilon());
        ReduceOps<ReduceOpMax, ReduceOpMin> reduce_op;
        ReduceData<int, int> reduce_data(reduce_op);
        using Tuple = typename decltype(reduce_data)::Type;
        for (int level = 0; level < levels(); ++level) {
            for (MFIter mfi(*eliminated_temperature[level]); mfi.isValid();
                 ++mfi) {
                auto const ea = energy[level]->const_array(mfi);
                auto const ka = kappa[level]->const_array(mfi);
                auto const to = old_temperature[level]->const_array(mfi);
                auto const st = source_temperature[level]->const_array(mfi);
                auto const lag = lagged_flux[level]->const_array(mfi);
                auto const out = eliminated_temperature[level]->array(mfi);
                auto const mask = masks_[level].const_array(mfi);
                auto const tau = with_derivative
                    ? temperature_derivative[level]->array(mfi)
                    : Array4<Real>{};
                reduce_op.eval(mfi.validbox(), reduce_data,
                [=] AMREX_GPU_DEVICE (int i, int j, int kk) noexcept -> Tuple
                {
                    GpuArray<Real, 3> const lagged_flux{
                        lag(i, j, kk, 0), lag(i, j, kk, 1), lag(i, j, kk, 2)};
                    RMCellData const cell{
                        ka(i, j, kk), amrex::max(ea(i, j, kk), Real(0)),
                        rm_dot(p.velocity, lagged_flux), to(i, j, kk),
                        st(i, j, kk)};
                    auto evaluate = [&] (Real temperature,
                                         Real& slope) -> Real {
                        return rm_material_residual(p, inverse_dt, cell,
                                                    temperature, slope);
                    };
                    auto g = [&] (Real temperature) -> Real {
                        Real unused;
                        return evaluate(temperature, unused);
                    };
                    Real x = rm_positive(p, out(i, j, kk));
                    Real lo = x;
                    Real hi = x;
                    int expansions = 0;
                    Real g_lo = g(x);
                    Real g_hi = g_lo;
                    if (g_lo <= Real(0)) {
                        while (g_hi < Real(0) && expansions < 200) {
                            lo = hi;
                            g_lo = g_hi;
                            hi *= Real(2);
                            g_hi = g(hi);
                            ++expansions;
                        }
                    } else {
                        while (g_lo > Real(0) && expansions < 200) {
                            hi = lo;
                            g_hi = g_lo;
                            lo *= Real(0.5);
                            g_lo = g(lo);
                            ++expansions;
                        }
                    }
                    // No bracket, or a NaN (which fails both tests).
                    if (!(g_lo <= Real(0) && g_hi >= Real(0))) {
                        if (tau) { tau(i, j, kk) = Real(0); }
                        return {0, mask(i, j, kk) ? 0 : 1};
                    }
                    // After two or more expansions the warm start lies
                    // outside the bracket: start from its nearer end.
                    if (x < lo || x > hi) {
                        x = -g_lo < g_hi ? lo : hi;
                    }
                    int iterations = 0;
                    bool converged = lo == hi;
                    while (!converged && iterations < maximum_iterations) {
                        ++iterations;
                        Real gp;
                        Real const gx = evaluate(x, gp);
                        if (gx == Real(0)) {
                            converged = true;
                            break;
                        }
                        if (gx < Real(0)) { lo = x; } else { hi = x; }
                        if (gp > Real(0)) {
                            Real const step = -gx / gp;
                            if (std::abs(step) <= tolerance * x) {
                                x += step;
                                converged = true;
                                break;
                            }
                            if (x + step > lo && x + step < hi) {
                                x += step;
                                continue;
                            }
                        }
                        x = Real(0.5) * (lo + hi);
                        converged = hi - lo <= tolerance * hi;
                    }
                    out(i, j, kk) = x;
                    if (tau) {
                        tau(i, j, kk) =
                            rm_temperature_derivative(p, inverse_dt, cell, x);
                    }
                    // Covered cells are solved for the warm start, but
                    // do not enter the composite solution.
                    return {iterations,
                            (converged || !mask(i, j, kk)) ? 1 : 0};
                });
            }
        }
        auto const values = reduce_data.value(reduce_op);
        int iterations = amrex::get<0>(values);
        int all_converged = amrex::get<1>(values);
        ParallelDescriptor::ReduceIntMax(iterations);
        ParallelDescriptor::ReduceIntMin(all_converged);
        counters.maximum_local_iterations =
            amrex::max(counters.maximum_local_iterations, iterations);
        return all_converged != 0;
    }

    // eliminate()'s dT/dE without the solve, for an eliminated_temperature
    // that is already T(E) for the E in `energy`.
    void eliminated_derivative ()
    {
        BL_PROFILE("FLD::rm_eliminated_derivative");
        RMPhysics const p = physics;
        Real const inverse_dt = Real(1) / p.dt;
        for (int level = 0; level < levels(); ++level) {
            for (MFIter mfi(*eliminated_temperature[level]); mfi.isValid();
                 ++mfi) {
                auto const ea = energy[level]->const_array(mfi);
                auto const ka = kappa[level]->const_array(mfi);
                auto const to = old_temperature[level]->const_array(mfi);
                auto const st = source_temperature[level]->const_array(mfi);
                auto const lag = lagged_flux[level]->const_array(mfi);
                auto const x = eliminated_temperature[level]->const_array(mfi);
                auto const tau = temperature_derivative[level]->array(mfi);
                ParallelFor(mfi.validbox(),
                [=] AMREX_GPU_DEVICE (int i, int j, int kk) noexcept
                {
                    GpuArray<Real, 3> const lagged_flux{
                        lag(i, j, kk, 0), lag(i, j, kk, 1), lag(i, j, kk, 2)};
                    RMCellData const cell{
                        ka(i, j, kk), amrex::max(ea(i, j, kk), Real(0)),
                        rm_dot(p.velocity, lagged_flux), to(i, j, kk),
                        st(i, j, kk)};
                    tau(i, j, kk) = rm_temperature_derivative(
                        p, inverse_dt, cell, x(i, j, kk));
                });
            }
        }
    }

    // Smallest 1/|k| over uncovered cells with k < 0 (rm_coupling); the
    // largest Real if there are none.
    Real runaway_time (LevelData const& e, LevelData const& t)
    {
        ReduceOps<ReduceOpMin> reduce_op;
        ReduceData<Real> reduce_data(reduce_op);
        using Tuple = typename decltype(reduce_data)::Type;
        Real const none = std::numeric_limits<Real>::max();
        for (int level = 0; level < levels(); ++level) {
            RMPhysics const p = level_physics(level);
            for (MFIter mfi(*e[level]); mfi.isValid(); ++mfi) {
                auto const ea = e[level]->const_array(mfi);
                auto const ta = t[level]->const_array(mfi);
                auto const ka = kappa[level]->const_array(mfi);
                auto const mask = masks_[level].const_array(mfi);
                reduce_op.eval(mfi.validbox(), reduce_data,
                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept -> Tuple
                {
                    Real const coupling = rm_coupling(
                        p, ka(i, j, k), ea(i, j, k), ta(i, j, k));
                    return {mask(i, j, k) && coupling < Real(0)
                                ? Real(-1) / coupling : none};
                });
            }
        }
        Real result = amrex::get<0>(reduce_data.value(reduce_op));
        ParallelDescriptor::ReduceRealMin(result);
        return result;
    }

    // Scaled norm of the full (unlagged) coupled residual at (E, T).
    Real coupled_residual_norm (LevelData const& e, LevelData const& t)
    {
        load_state(e, t);
        return loaded_coupled_residual_norm(e, t);
    }

    // coupled_residual_norm with (e, t) already loaded.
    Real loaded_coupled_residual_norm (LevelData const& e, LevelData const& t)
    {
        evaluate_residual(WorkCoupling::Current);
        return evaluated_coupled_residual_norm(e, t);
    }

    // coupled_residual_norm with the residual at (e, t) already in
    // residual_energy and residual_temperature.
    Real evaluated_coupled_residual_norm (LevelData const& e,
                                          LevelData const& t)
    {
        Real const es = scale(e, options.energy_scale_floor);
        Real const ts = scale(t, options.temperature_scale_floor);
        return std::sqrt(
            composite_weighted_dot(residual_energy, residual_energy,
                                   hierarchy, masks_) / (es * es) +
            composite_weighted_dot(residual_temperature, residual_temperature,
                                   hierarchy, masks_) / (ts * ts));
    }

    // AMG setup with the current acoef and the loaded state's face
    // coefficients.  Robin data are homogeneous: the solver only ever
    // solves for corrections.
    void setup_solver ()
    {
        solver.setup(Real(1), Real(1), get_level_const_ptrs(acoef),
                     get_face_const_ptrs(bcoef), lobc, hibc, {},
                     robin_data(true));
        record_setup(counters.summary, solver);
    }

    // Flux out of the domain through each face, integrated over the face,
    // at the loaded state.
    Array<Real, 6> boundary_outflow ()
    {
        Array<Real, 6> result{};
        for (int level = 0; level < levels(); ++level) {
            RMPhysics const p = level_physics(level);
            Real const volume = p.dx[0] * p.dx[1] * p.dx[2];
            ReduceOps<ReduceOpSum, ReduceOpSum, ReduceOpSum, ReduceOpSum,
                      ReduceOpSum, ReduceOpSum> reduce_op;
            ReduceData<Real, Real, Real, Real, Real, Real> reduce_data(
                reduce_op);
            using Tuple = typename decltype(reduce_data)::Type;
            for (MFIter mfi(*energy[level]); mfi.isValid(); ++mfi) {
                auto const e = energy[level]->const_array(mfi);
                auto const b = rm_faces(bcoef, level, mfi);
                auto const rb = robin_b[level]->const_array(mfi);
                auto const mask = masks_[level].const_array(mfi);
                reduce_op.eval(mfi.validbox(), reduce_data,
                [=] AMREX_GPU_DEVICE (int i, int j, int kk) noexcept -> Tuple
                {
                    IntVect const c(AMREX_D_DECL(i, j, kk));
                    Real out[6] = {};
                    if (mask(c)) {
                        for (int d = 0; d < 3; ++d) {
                            for (int side = 0; side < 2; ++side) {
                                if (!rm_is_boundary_face(p, c, d, side)) {
                                    continue;
                                }
                                Real const flux =
                                    rm_face_flux(p, e, b, rb, c, d, side);
                                out[2 * d + side] = (side == 0 ? -flux : flux) *
                                                    volume / p.dx[d];
                            }
                        }
                    }
                    return {out[0], out[1], out[2], out[3], out[4], out[5]};
                });
            }
            auto const sums = reduce_data.value(reduce_op);
            result[0] += amrex::get<0>(sums);
            result[1] += amrex::get<1>(sums);
            result[2] += amrex::get<2>(sums);
            result[3] += amrex::get<3>(sums);
            result[4] += amrex::get<4>(sums);
            result[5] += amrex::get<5>(sums);
        }
        ParallelDescriptor::ReduceRealSum(result.data(), 6);
        return result;
    }

    void explicit_material_advection (LevelData& t)
    {
        RMPhysics const p = level_physics(0);
        // The explicit update is stable only for dt sum_d |v_d|/dx_d <= 1.
        Real courant = Real(0);
        for (int d = 0; d < 3; ++d) {
            courant += p.dt * std::abs(p.velocity[d]) / p.dx[d];
        }
        if (courant > Real(1)) {
            amrex::Abort("Explicit material advection needs "
                         "dt * sum |v_d|/dx_d <= 1; it is " +
                         std::to_string(courant) + ". Reduce dt.");
        }
        auto work = make_cell_data(hierarchy, 1, 1);
        copy_level_data(work, t);
        work[0]->FillBoundary(hierarchy.geom[0].periodicity());
        for (MFIter mfi(*t[0]); mfi.isValid(); ++mfi) {
            auto const w = work[0]->const_array(mfi);
            auto const out = t[0]->array(mfi);
            ParallelFor(mfi.validbox(),
            [=] AMREX_GPU_DEVICE (int i, int j, int kk) noexcept
            {
                IntVect const c(AMREX_D_DECL(i, j, kk));
                Real divergence = Real(0);
                for (int d = 0; d < 3; ++d) {
                    IntVect const s = IntVect::TheDimensionVector(d);
                    Real const v = p.velocity[d];
                    Real const f_lo = v * rm_face_advection(p, d, w(c - s), w(c));
                    Real const f_hi = v * rm_face_advection(p, d, w(c), w(c + s));
                    divergence += (f_hi - f_lo) / p.dx[d];
                }
                out(c) = w(c) - p.dt * divergence;
            });
        }
    }

    bool attempt_step (RadiationMaterialState& state, bool allow_failure,
                       int& step_nonlinear);
    [[nodiscard]] bool predictors (RadiationMaterialState& state);
    void advance (RadiationMaterialState& state, Real dt);
};

namespace
{

// Newton candidate x exp(s dx/x), or x + s dx when not `positive`.
void
rm_update (LevelData& candidate, LevelData const& state,
           LevelData const& correction, Real step_length, bool positive)
{
    if (!positive) {
        lincomb_level_data(candidate, Real(1), state, step_length,
                           correction);
        return;
    }
    for (int level = 0; level < static_cast<int>(candidate.size()); ++level) {
        for (MFIter mfi(*candidate[level]); mfi.isValid(); ++mfi) {
            auto const next = candidate[level]->array(mfi);
            auto const x = state[level]->const_array(mfi);
            auto const dx = correction[level]->const_array(mfi);
            ParallelFor(mfi.validbox(),
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                Real const exponent = amrex::max(Real(-50), amrex::min(
                    Real(50), step_length * dx(i, j, k) /
                                  amrex::max(x(i, j, k), Real(1.e-30))));
                next(i, j, k) = x(i, j, k) * std::exp(exponent);
            });
        }
    }
}

// Whether every uncovered valid cell on this rank is finite and in
// (0, upper_bound).  NaN and infinity fail both comparisons.  Callers
// combine the ranks with rm_all_ranks.
bool
rm_locally_admissible (LevelData const& field, Vector<iMultiFab> const& masks,
                       Real upper_bound = std::numeric_limits<Real>::max())
{
    ReduceOps<ReduceOpMin> reduce_op;
    ReduceData<int> reduce_data(reduce_op);
    using Tuple = typename decltype(reduce_data)::Type;
    for (int level = 0; level < static_cast<int>(field.size()); ++level) {
        for (MFIter mfi(*field[level]); mfi.isValid(); ++mfi) {
            auto const array = field[level]->const_array(mfi);
            auto const mask = masks[level].const_array(mfi);
            reduce_op.eval(mfi.validbox(), reduce_data,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept -> Tuple
            {
                Real const value = array(i, j, k);
                bool const admissible =
                    value > Real(0) && value < upper_bound;
                return {(!mask(i, j, k) || admissible) ? 1 : 0};
            });
        }
    }
    return amrex::get<0>(reduce_data.value(reduce_op)) != 0;
}

bool
rm_all_ranks (bool local)
{
    int result = local ? 1 : 0;
    ParallelDescriptor::ReduceIntMin(result);
    return result != 0;
}

// max(|E' - E| / E scale, |T' - T| / T scale) over uncovered valid cells,
// each scale the field's maximum (at least `floor`) as in Impl::scale, in
// one pass and one reduction.
Real
rm_scaled_change (LevelData const& next_energy, LevelData const& energy,
                  Real energy_floor, LevelData const& next_temperature,
                  LevelData const& temperature, Real temperature_floor,
                  Vector<iMultiFab> const& masks)
{
    AMREX_ALWAYS_ASSERT(energy.size() == masks.size() &&
                        next_energy.size() == masks.size() &&
                        temperature.size() == masks.size() &&
                        next_temperature.size() == masks.size());
    ReduceOps<ReduceOpMax, ReduceOpMax, ReduceOpMax, ReduceOpMax> reduce_op;
    ReduceData<Real, Real, Real, Real> reduce_data(reduce_op);
    using Tuple = typename decltype(reduce_data)::Type;
    Real const low = std::numeric_limits<Real>::lowest();
    for (int level = 0; level < static_cast<int>(masks.size()); ++level) {
        for (MFIter mfi(*energy[level]); mfi.isValid(); ++mfi) {
            auto const en = next_energy[level]->const_array(mfi);
            auto const e = energy[level]->const_array(mfi);
            auto const tn = next_temperature[level]->const_array(mfi);
            auto const t = temperature[level]->const_array(mfi);
            auto const mask = masks[level].const_array(mfi);
            reduce_op.eval(mfi.validbox(), reduce_data,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept -> Tuple
            {
                if (!mask(i, j, k)) { return {Real(0), low, Real(0), low}; }
                return {amrex::Math::abs(en(i, j, k) - e(i, j, k)), e(i, j, k),
                        amrex::Math::abs(tn(i, j, k) - t(i, j, k)),
                        t(i, j, k)};
            });
        }
    }
    auto const values = reduce_data.value(reduce_op);
    Real result[4] = {amrex::get<0>(values), amrex::get<1>(values),
                      amrex::get<2>(values), amrex::get<3>(values)};
    ParallelDescriptor::ReduceRealMax(result, 4);
    return amrex::max(result[0] / amrex::max(result[1], energy_floor),
                      result[2] / amrex::max(result[3], temperature_floor));
}

// Sum local dot products over ranks in one reduction, then divide by the
// norm scale.
void
rm_reduce_dots (Vector<Real>& dots, Real scale_squared)
{
    if (!dots.empty()) {
        ParallelDescriptor::ReduceRealSum(dots.data(),
                                          static_cast<int>(dots.size()));
    }
    for (Real& value : dots) { value /= scale_squared; }
}

void
rm_scale (LevelData& field, Real a)
{
    for (auto& level : field) { level->mult(a, 0, 1, 0); }
}

using Impl = RadiationMaterialSolver::Impl;

// Nonlinear elimination: Newton-Krylov on R(E) = R_E + R_T with T = T(E)
// (the work term's flux lagged), preconditioned by one AMG cycle of
// (1 + dT/dE)/dt - div D grad.  Advection and work terms are left out of
// the preconditioner.  The reaction coefficient is at least 1/(2 dt), as
// in CoupledProblem: just below g_T = -sigma, 1 + tau approaches zero and
// the AMG row becomes nearly singular.  The quasi-Newton linear solve uses
// the same floor, since a near-zero row there gives a huge, inadmissible
// update.
class ReducedProblem
{
  public:
    using State = LevelData;
    using RT = Real;

    // The norm scale is set from the initial state, because the Newton
    // solver checks the initial residual before the first prepare().
    ReducedProblem (Impl& impl, State const& initial)
        : m_impl(impl),
          m_scale(impl.scale(initial, impl.options.energy_scale_floor))
    {}

    // True when the work fields still hold the last residual(), and it
    // was at `e`: T(E) is in eliminated_temperature, (E, T(E)) is loaded,
    // and the lagged residual is in residual_energy/residual_temperature.
    [[nodiscard]] bool evaluated_at (State const& e) const
    {
        return m_evaluated == m_impl.generation &&
               composite_maximum_difference(e, m_impl.energy,
                                            m_impl.masks_) == Real(0);
    }

    // NewtonKrylovSolver calls this right after residual(e), so only dT/dE
    // is normally left to compute; any other call order falls back to a
    // fresh solve.
    bool prepare (State const& e)
    {
        if (!evaluated_at(e)) { return prepare_unsolved(e); }
        BL_PROFILE("FLD::rm_reduced_prepare");
        m_scale = m_impl.scale(e, m_impl.options.energy_scale_floor);
        m_impl.eliminated_derivative();
        setup_preconditioner();
        return true;
    }

    // prepare() for an E whose T(E) has not been evaluated.  Fails, leaving
    // the preconditioner unset, if a cell-local solve fails: tau is then
    // not written in those cells.
    bool prepare_unsolved (State const& e)
    {
        BL_PROFILE("FLD::rm_reduced_prepare");
        m_scale = m_impl.scale(e, m_impl.options.energy_scale_floor);
        if (!m_impl.eliminate(e, true)) { return false; }
        m_impl.load_loaded_energy(m_impl.eliminated_temperature);
        setup_preconditioner();
        return true;
    }

    // R(E) is undefined where the cell-local solves fail: the residual is
    // then NaN, which the line search rejects.
    void residual (State const& e, State& output)
    {
        m_evaluated = -1;
        if (!m_impl.eliminate(e, false)) {
            set_level_data(output, std::numeric_limits<Real>::quiet_NaN());
            return;
        }
        m_impl.load_loaded_energy(m_impl.eliminated_temperature);
        m_impl.evaluate_residual(WorkCoupling::Lagged);
        m_evaluated = m_impl.generation;
        lincomb_level_data(output, Real(1), m_impl.residual_energy, Real(1),
                           m_impl.residual_temperature);
    }

    // The norm scale, set from the last prepared state.
    [[nodiscard]] Real norm_scale () const { return m_scale; }

    void linearized_residual (State const& e, State& output)
    {
        residual(e, output);
    }

    State makeVecRHS () const { return make_cell_data(m_impl.hierarchy, 1, 0); }
    State makeVecLHS () const { return make_cell_data(m_impl.hierarchy, 1, 1); }

    void precondition (State& output, State const& rhs)
    {
        set_level_data(output, Real(0));
        m_impl.solver.precondition(get_level_ptrs(output),
                                   get_level_const_ptrs(rhs));
    }

    void assign (State& lhs, State const& rhs) const { copy_level_data(lhs, rhs); }
    void copy_state (State& lhs, State const& rhs) const
    {
        copy_level_data_with_ghosts(lhs, rhs);
    }
    Real dotProduct (State const& lhs, State const& rhs) const
    {
        return composite_weighted_dot(lhs, rhs, m_impl.hierarchy,
                                      m_impl.masks_) / (m_scale * m_scale);
    }
    // All of a GMRES orthogonalization's dot products in one reduction.
    void dotProducts (State const& lhs, Vector<State const*> const& basis,
                      Vector<Real>& result) const
    {
        result.resize(basis.size());
        for (std::size_t j = 0; j < basis.size(); ++j) {
            result[j] = composite_weighted_dot(lhs, *basis[j],
                                               m_impl.hierarchy,
                                               m_impl.masks_, true);
        }
        rm_reduce_dots(result, m_scale * m_scale);
    }
    void increment (State& lhs, State const& rhs, Real a) const
    {
        saxpy_level_data(lhs, a, rhs);
    }
    void linComb (State& lhs, Real a, State const& x, Real b,
                  State const& y) const
    {
        lincomb_level_data(lhs, a, x, b, y);
    }
    Real norm2 (State const& x) const
    {
        return std::sqrt(amrex::max(dotProduct(x, x), Real(0)));
    }
    Real norm2Data (State const& x) const
    {
        return composite_weighted_dot(x, x, m_impl.hierarchy, m_impl.masks_);
    }
    Real norm2FromData (Real sum) const
    {
        return std::sqrt(amrex::max(sum / (m_scale * m_scale), Real(0)));
    }
    void scale (State& x, Real a) const { rm_scale(x, a); }
    void setToZero (State& x) const { set_level_data(x, Real(0)); }
    State clone (State const& x) const { return clone_level_data(x); }
    void fill_candidate (State& candidate, State const& state,
                         State const& correction, Real step_length) const
    {
        rm_update(candidate, state, correction, step_length,
                  m_impl.options.positive_update);
        average_down_hierarchy(candidate, m_impl.hierarchy);
    }
    bool admissible (State const& state) const
    {
        return rm_all_ranks(rm_locally_admissible(
            state, m_impl.masks_, m_impl.options.maximum_energy));
    }
    // Measured against the norm scale, like the residual (see
    // Impl::predictors).
    Real relative_change (State const& lhs, State const& rhs) const
    {
        return composite_maximum_difference(lhs, rhs, m_impl.masks_) /
               m_scale;
    }

  private:
    // Reaction coefficient (1 + dT/dE)/dt, then the AMG setup at the
    // loaded state.
    void setup_preconditioner ()
    {
        Real const inverse_dt = Real(1) / m_impl.physics.dt;
        for (int level = 0; level < m_impl.levels(); ++level) {
            for (MFIter mfi(*m_impl.acoef[level]); mfi.isValid(); ++mfi) {
                auto const tau =
                    m_impl.temperature_derivative[level]->const_array(mfi);
                auto const a = m_impl.acoef[level]->array(mfi);
                ParallelFor(mfi.validbox(),
                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    a(i, j, k) = inverse_dt *
                        amrex::max(Real(1) + tau(i, j, k), Real(0.5));
                });
            }
        }
        m_impl.setup_solver();
    }

    Impl& m_impl;
    Real m_scale = Real(1);
    // Impl::generation after the last successful residual(); -1 if none.
    long m_evaluated = -1;
};

// Coupled (E, T) Newton-Krylov.  The preconditioner eliminates the
// cell-local material block linearly (Schur complement, the Fleck factor):
// with k = -dR_E/dT and t_d = dR_T/dT,
//   (1/dt + sigma - k sigma / t_d - div D grad) dE = r_E + k r_T / t_d,
//   dT = (r_T + sigma dE) / t_d.
// k < 0 where absorption heating raises the opacity enough (opacity_power
// < 0 with E > T^4, or > 4 with E < T^4).  The reaction coefficient is
// (1/dt) (1 + sigma / t_d), which is not positive for t_d in [-sigma, 0].
// In cells where t_d falls in [-2 sigma, 0] the preconditioner uses
// max(k, 0) instead, the block inverse of a Jacobian without that
// feedback; elsewhere it keeps the exact k, and the coefficient is at
// least 1/(2 dt).  Newton still uses the true residual.
class CoupledProblem
{
  public:
    using State = RadiationMaterialState;
    using RT = Real;

    // Norm scales from the initial state; see ReducedProblem.
    CoupledProblem (Impl& impl, State const& initial)
        : m_impl(impl),
          m_energy_scale(impl.scale(initial.energy,
                                    impl.options.energy_scale_floor)),
          m_temperature_scale(impl.scale(
              initial.temperature, impl.options.temperature_scale_floor)),
          m_energy_rhs(make_cell_data(impl.hierarchy, 1, 0))
    {}

    bool prepare (State const& state)
    {
        BL_PROFILE("FLD::rm_coupled_prepare");
        m_energy_scale = m_impl.scale(state.energy,
                                      m_impl.options.energy_scale_floor);
        m_temperature_scale = m_impl.scale(
            state.temperature, m_impl.options.temperature_scale_floor);
        // NewtonKrylovSolver calls this right after residual(state), which
        // already loaded the state.
        if (!evaluated_at(state)) {
            m_impl.load_state(state.energy, state.temperature);
        }
        RMPhysics const p = m_impl.physics;
        Real const inverse_dt = Real(1) / p.dt;
        for (int level = 0; level < m_impl.levels(); ++level) {
            for (MFIter mfi(*m_impl.acoef[level]); mfi.isValid(); ++mfi) {
                auto const e = m_impl.energy[level]->const_array(mfi);
                auto const t = m_impl.temperature[level]->const_array(mfi);
                auto const ka = m_impl.kappa[level]->const_array(mfi);
                auto const a = m_impl.acoef[level]->array(mfi);
                auto const kk = m_impl.schur_k[level]->array(mfi);
                auto const ss = m_impl.schur_sigma[level]->array(mfi);
                auto const td = m_impl.schur_diagonal[level]->array(mfi);
                ParallelFor(mfi.validbox(),
                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    Real const sigma =
                        rm_opacity(p, ka(i, j, k), rm_positive(p, t(i, j, k)));
                    // k, clipped where t_d falls in [-2 sigma, 0]; see the
                    // class comment.
                    Real const exact =
                        rm_coupling(p, ka(i, j, k), e(i, j, k), t(i, j, k));
                    Real const exact_diagonal = inverse_dt + exact;
                    Real const coupling =
                        exact_diagonal > Real(0) ||
                                exact_diagonal <= Real(-2) * sigma
                            ? exact
                            : amrex::max(exact, Real(0));
                    Real const diagonal = inverse_dt + coupling;
                    kk(i, j, k) = coupling;
                    ss(i, j, k) = sigma;
                    td(i, j, k) = diagonal;
                    // sigma - k sigma / t_d, without the cancellation that
                    // makes it negative for very large sigma.
                    a(i, j, k) = inverse_dt + sigma * inverse_dt / diagonal;
                });
            }
        }
        m_impl.setup_solver();
        return true;
    }

    // True when residual_energy/residual_temperature still hold the last
    // residual(), and it was at `state`.
    [[nodiscard]] bool evaluated_at (State const& state) const
    {
        return m_evaluated == m_impl.generation &&
               composite_maximum_difference(state.energy, m_impl.energy,
                                            m_impl.masks_) == Real(0) &&
               composite_maximum_difference(state.temperature,
                                            m_impl.temperature,
                                            m_impl.masks_) == Real(0);
    }

    void residual (State const& state, State& output)
    {
        m_impl.load_state(state.energy, state.temperature);
        m_impl.evaluate_residual(WorkCoupling::Current);
        m_evaluated = m_impl.generation;
        copy_level_data(output.energy, m_impl.residual_energy);
        copy_level_data(output.temperature, m_impl.residual_temperature);
    }

    void linearized_residual (State const& state, State& output)
    {
        residual(state, output);
    }

    State makeVecRHS () const
    {
        return {make_cell_data(m_impl.hierarchy, 1, 0),
                make_cell_data(m_impl.hierarchy, 1, 0)};
    }
    State makeVecLHS () const
    {
        return {make_cell_data(m_impl.hierarchy, 1, 1),
                make_cell_data(m_impl.hierarchy, 1, 1)};
    }

    void precondition (State& output, State const& rhs)
    {
        copy_level_data(m_energy_rhs, rhs.energy);
        for (int level = 0; level < m_impl.levels(); ++level) {
            for (MFIter mfi(*m_energy_rhs[level]); mfi.isValid(); ++mfi) {
                auto const re = m_energy_rhs[level]->array(mfi);
                auto const rt = rhs.temperature[level]->const_array(mfi);
                auto const kk = m_impl.schur_k[level]->const_array(mfi);
                auto const td = m_impl.schur_diagonal[level]->const_array(mfi);
                ParallelFor(mfi.validbox(),
                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    re(i, j, k) += kk(i, j, k) * rt(i, j, k) / td(i, j, k);
                });
            }
        }
        set_level_data(output.energy, Real(0));
        m_impl.solver.precondition(get_level_ptrs(output.energy),
                                   get_level_const_ptrs(m_energy_rhs));
        for (int level = 0; level < m_impl.levels(); ++level) {
            for (MFIter mfi(*output.temperature[level]); mfi.isValid();
                 ++mfi) {
                auto const dt_out = output.temperature[level]->array(mfi);
                auto const de = output.energy[level]->const_array(mfi);
                auto const rt = rhs.temperature[level]->const_array(mfi);
                auto const ss = m_impl.schur_sigma[level]->const_array(mfi);
                auto const td = m_impl.schur_diagonal[level]->const_array(mfi);
                ParallelFor(mfi.validbox(),
                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    dt_out(i, j, k) = (rt(i, j, k) + ss(i, j, k) * de(i, j, k)) /
                                      td(i, j, k);
                });
            }
        }
    }

    void assign (State& lhs, State const& rhs) const
    {
        copy_level_data(lhs.energy, rhs.energy);
        copy_level_data(lhs.temperature, rhs.temperature);
    }
    void copy_state (State& lhs, State const& rhs) const
    {
        copy_level_data_with_ghosts(lhs.energy, rhs.energy);
        copy_level_data_with_ghosts(lhs.temperature, rhs.temperature);
    }
    Real dotProduct (State const& lhs, State const& rhs) const
    {
        return composite_weighted_dot(lhs.energy, rhs.energy,
                                      m_impl.hierarchy, m_impl.masks_) /
                   (m_energy_scale * m_energy_scale) +
               composite_weighted_dot(lhs.temperature, rhs.temperature,
                                      m_impl.hierarchy, m_impl.masks_) /
                   (m_temperature_scale * m_temperature_scale);
    }
    // All of a GMRES orthogonalization's dot products in one reduction.
    void dotProducts (State const& lhs, Vector<State const*> const& basis,
                      Vector<Real>& result) const
    {
        Real const ratio = (m_energy_scale * m_energy_scale) /
                           (m_temperature_scale * m_temperature_scale);
        result.resize(basis.size());
        for (std::size_t j = 0; j < basis.size(); ++j) {
            result[j] =
                composite_weighted_dot(lhs.energy, basis[j]->energy,
                                       m_impl.hierarchy, m_impl.masks_,
                                       true) +
                ratio * composite_weighted_dot(lhs.temperature,
                                               basis[j]->temperature,
                                               m_impl.hierarchy,
                                               m_impl.masks_, true);
        }
        rm_reduce_dots(result, m_energy_scale * m_energy_scale);
    }
    void increment (State& lhs, State const& rhs, Real a) const
    {
        saxpy_level_data(lhs.energy, a, rhs.energy);
        saxpy_level_data(lhs.temperature, a, rhs.temperature);
    }
    void linComb (State& lhs, Real a, State const& x, Real b,
                  State const& y) const
    {
        lincomb_level_data(lhs.energy, a, x.energy, b, y.energy);
        lincomb_level_data(lhs.temperature, a, x.temperature, b,
                           y.temperature);
    }
    Real norm2 (State const& x) const
    {
        return std::sqrt(amrex::max(dotProduct(x, x), Real(0)));
    }
    Array<Real, 2> norm2Data (State const& x) const
    {
        return {composite_weighted_dot(x.energy, x.energy, m_impl.hierarchy,
                                       m_impl.masks_),
                composite_weighted_dot(x.temperature, x.temperature,
                                       m_impl.hierarchy, m_impl.masks_)};
    }
    Real norm2FromData (Array<Real, 2> const& sums) const
    {
        return std::sqrt(amrex::max(
            sums[0] / (m_energy_scale * m_energy_scale) +
                sums[1] / (m_temperature_scale * m_temperature_scale),
            Real(0)));
    }
    void scale (State& x, Real a) const
    {
        rm_scale(x.energy, a);
        rm_scale(x.temperature, a);
    }
    void setToZero (State& x) const
    {
        set_level_data(x.energy, Real(0));
        set_level_data(x.temperature, Real(0));
    }
    State clone (State const& x) const
    {
        return {clone_level_data(x.energy), clone_level_data(x.temperature)};
    }
    void fill_candidate (State& candidate, State const& state,
                         State const& correction, Real step_length) const
    {
        rm_update(candidate.energy, state.energy, correction.energy,
                  step_length, m_impl.options.positive_update);
        rm_update(candidate.temperature, state.temperature,
                  correction.temperature, step_length,
                  m_impl.options.positive_update);
        average_down_hierarchy(candidate.energy, m_impl.hierarchy);
        average_down_hierarchy(candidate.temperature, m_impl.hierarchy);
    }
    bool admissible (State const& state) const
    {
        // One global reduction for both fields.
        return rm_all_ranks(
            rm_locally_admissible(state.energy, m_impl.masks_,
                                  m_impl.options.maximum_energy) &&
            rm_locally_admissible(state.temperature, m_impl.masks_));
    }
    Real relative_change (State const& lhs, State const& rhs) const
    {
        return amrex::max(
            composite_maximum_difference(lhs.energy, rhs.energy,
                                         m_impl.masks_) /
                m_energy_scale,
            composite_maximum_difference(lhs.temperature, rhs.temperature,
                                         m_impl.masks_) /
                m_temperature_scale);
    }

  private:
    Impl& m_impl;
    Real m_energy_scale = Real(1);
    Real m_temperature_scale = Real(1);
    LevelData m_energy_rhs; // precondition() scratch
    // Impl::generation after the last residual(); -1 if none.
    long m_evaluated = -1;
};

template <class Result>
void
rm_record_newton (RadiationMaterialCounters& counters, Result const& result,
                  int& step_nonlinear)
{
    step_nonlinear += result.nonlinear_iterations;
    counters.krylov += result.total_linear_iterations;
    counters.maximum_krylov =
        amrex::max(counters.maximum_krylov, result.maximum_linear_iterations);
    counters.final_change = result.final_change;
}

} // namespace

// Damped (0.7) linearized (Picard) iterations, until the relative change
// falls below 1e-3 (at most 20); false if a linear solve fails or the
// result is inadmissible: with sigma and D frozen, a converged solve
// of (1/dt + sigma - div D grad) dE = -R_E, then the diagonal update
// (1/dt + 4 sigma T^3) dT = -R_T at the new E.  Without velocity this is
// E <- [E^n/dt + sigma T^4], T <- linearized emission.  Used to restart a
// step whose Newton iteration failed.
bool
RadiationMaterialSolver::Impl::predictors (RadiationMaterialState& state)
{
    BL_PROFILE("FLD::rm_predictors");
    auto correction = make_cell_data(hierarchy, 1, 1);
    auto rhs = make_cell_data(hierarchy, 1, 0);
    Real const inverse_dt = Real(1) / physics.dt;
    for (int predictor = 0; predictor < 20; ++predictor) {
        load_state(state.energy, state.temperature);
        evaluate_residual(WorkCoupling::Current);
        for (int level = 0; level < levels(); ++level) {
            RMPhysics const p = level_physics(level);
            for (MFIter mfi(*acoef[level]); mfi.isValid(); ++mfi) {
                auto const t = temperature[level]->const_array(mfi);
                auto const ka = kappa[level]->const_array(mfi);
                auto const a = acoef[level]->array(mfi);
                ParallelFor(mfi.validbox(),
                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    a(i, j, k) = inverse_dt + rm_opacity(p, ka(i, j, k),
                                                         t(i, j, k));
                });
            }
        }
        setup_solver();
        copy_level_data(rhs, residual_energy);
        rm_scale(rhs, Real(-1));
        set_level_data(correction, Real(0));
        auto const info = solver.solve(get_level_ptrs(correction),
                                       get_level_const_ptrs(rhs),
                                       linear_tolerance(), Real(0));
        record_solve(counters.summary, info);
        if (!info.converged) { return false; }
        auto next_energy = clone_level_data(state.energy);
        saxpy_level_data(next_energy, Real(1), correction);

        load_state(next_energy, state.temperature);
        evaluate_residual(WorkCoupling::Current);
        auto next_temperature = clone_level_data(state.temperature);
        for (int level = 0; level < levels(); ++level) {
            RMPhysics const p = level_physics(level);
            for (MFIter mfi(*next_temperature[level]); mfi.isValid(); ++mfi) {
                auto const t = temperature[level]->const_array(mfi);
                auto const ka = kappa[level]->const_array(mfi);
                auto const rt = residual_temperature[level]->const_array(mfi);
                auto const out = next_temperature[level]->array(mfi);
                ParallelFor(mfi.validbox(),
                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    Real const tc = rm_positive(p, t(i, j, k));
                    Real const sigma = rm_opacity(p, ka(i, j, k), tc);
                    out(i, j, k) = t(i, j, k) - rt(i, j, k) /
                        (inverse_dt + Real(4) * sigma * tc * tc * tc);
                });
            }
        }

        // Changes are measured against the norm scale, like the residual: a
        // relative change per cell stalls at round-off in cells far below
        // the scale (e.g. ahead of a radiation front).
        Real const change = rm_scaled_change(
            next_energy, state.energy, options.energy_scale_floor,
            next_temperature, state.temperature,
            options.temperature_scale_floor, masks_);
        lincomb_level_data(state.energy, Real(0.3), state.energy, Real(0.7),
                           next_energy);
        lincomb_level_data(state.temperature, Real(0.3), state.temperature,
                           Real(0.7), next_temperature);
        average_down_hierarchy(state.energy, hierarchy);
        average_down_hierarchy(state.temperature, hierarchy);
        if (change < Real(1.e-3)) { break; }
    }
    // A positive update keeps the sign of its start, so Newton cannot
    // recover from a nonpositive predicted state.
    return rm_all_ranks(
        rm_locally_admissible(state.energy, masks_,
                              options.maximum_energy) &&
        rm_locally_admissible(state.temperature, masks_));
}

// One attempt at the implicit step from `state`.  With allow_failure, a
// Newton, lag, quasi-Newton, cell-local material, or linear-solve failure
// returns false instead of aborting.
bool
RadiationMaterialSolver::Impl::attempt_step (RadiationMaterialState& state,
                                             bool allow_failure,
                                             int& step_nonlinear)
{
    Real const tolerance = options.nonlinear_tolerance;
    NewtonKrylovOptions newton_options;
    newton_options.nonlinear_tolerance = tolerance;
    newton_options.maximum_nonlinear_iterations =
        options.maximum_nonlinear_iterations;
    newton_options.maximum_linear_iterations =
        options.maximum_linear_iterations;
    newton_options.krylov_restart_length = options.krylov_restart_length;
    newton_options.maximum_line_search_iterations =
        options.maximum_line_search_iterations;
    newton_options.centered_difference = options.centered_difference;
    newton_options.eisenstat_walker = options.eisenstat_walker;
    newton_options.linear_verbosity = options.linear_verbosity;
    newton_options.problem_name = "The radiation-material system";
    newton_options.abort_on_failure = !allow_failure;
    auto fail = [allow_failure] (char const* message) {
        if (!allow_failure) { amrex::Abort(message); }
        return false;
    };

    if (options.solver == "newton_krylov") {
        CoupledProblem problem(*this, state);
        NewtonKrylovSolver<CoupledProblem> newton(problem, newton_options);
        auto const result = newton.solve(state);
        rm_record_newton(counters, result, step_nonlinear);
        if (!result.converged) {
            return fail("Coupled Newton-Krylov did not converge");
        }
        accepted_coupled_residual = problem.evaluated_at(state)
            ? evaluated_coupled_residual_norm(state.energy, state.temperature)
            : coupled_residual_norm(state.energy, state.temperature);
        return true;
    }

    copy_level_data(eliminated_temperature, state.temperature);
    if (physics.moving) {
        load_state(state.energy, state.temperature);
        freeze_work_flux();
    }

    if (options.solver == "ne_quasi_newton") {
        // Brunner, Haut & Nowak Algorithm 2 (NLMFG-NE) for one group: solve
        // T(E) cell by cell, then one converged linear solve of the
        // frozen-coefficient system, and repeat.  Their Jacobian ignores
        // opacity derivatives; dT/dE here includes d(sigma)/dT.  With a
        // moving medium the work flux is re-frozen each iteration at
        // (E, T(E)), and the right-hand side is the lagged residual there
        // without solving T(E) again: R_T is then not zero, but the
        // iteration count matches a fresh elimination at half the cost.
        // (Freezing at the new E before the elimination instead costs ~15%
        // more iterations and loses round-off energy conservation.)
        auto correction = make_cell_data(hierarchy, 1, 1);
        auto rhs = make_cell_data(hierarchy, 1, 0);
        Real change = std::numeric_limits<Real>::max();
        ReducedProblem problem(*this, state.energy);
        for (int iteration = 0;; ++iteration) {
            if (!problem.prepare_unsolved(state.energy)) {
                return fail("Cell-local material solve did not converge");
            }
            copy_level_data(state.temperature, eliminated_temperature);
            // prepare_unsolved() loaded (E, T(E)).
            Real const unlagged =
                loaded_coupled_residual_norm(state.energy, state.temperature);
            if (unlagged <= tolerance && change <= tolerance) {
                counters.final_change = change;
                accepted_coupled_residual = unlagged;
                return true;
            }
            if (iteration == options.maximum_quasi_newton_iterations) {
                return fail("Quasi-Newton iteration did not converge");
            }
            // Without motion the lagged residual is the current one.
            if (physics.moving) {
                freeze_work_flux();
                evaluate_residual(WorkCoupling::Lagged);
            }
            lincomb_level_data(rhs, Real(1), residual_energy, Real(1),
                               residual_temperature);
            problem.scale(rhs, Real(-1));
            set_level_data(correction, Real(0));
            auto const info = solver.solve(get_level_ptrs(correction),
                                           get_level_const_ptrs(rhs),
                                           linear_tolerance(), Real(0));
            record_solve(counters.summary, info);
            if (!info.converged) {
                return fail("Quasi-Newton linear solve did not converge");
            }
            auto next = clone_level_data(state.energy);
            saxpy_level_data(next, Real(1), correction);
            average_down_hierarchy(next, hierarchy);
            // eliminate() clamps E at zero, so without this check a
            // nonpositive E could be accepted as converged.
            if (!rm_all_ranks(rm_locally_admissible(
                    next, masks_, options.maximum_energy))) {
                return fail("Quasi-Newton update left E inadmissible");
            }
            change =
                composite_maximum_difference(next, state.energy, masks_) /
                problem.norm_scale();
            copy_level_data(state.energy, next);
            ++step_nonlinear;
        }
    }

    // ne_newton_krylov
    if (!physics.moving) {
        ReducedProblem problem(*this, state.energy);
        NewtonKrylovSolver<ReducedProblem> newton(problem, newton_options);
        auto const result = newton.solve(state.energy);
        rm_record_newton(counters, result, step_nonlinear);
        if (!result.converged) {
            return fail("Reduced Newton-Krylov did not converge");
        }
        // The last residual() is normally at the returned E, so T(E) and
        // its residual are already evaluated; without motion the lagged
        // residual is the current one.
        if (!problem.evaluated_at(state.energy)) {
            if (!eliminate(state.energy, false)) {
                return fail("Cell-local material solve did not converge");
            }
            copy_level_data(state.temperature, eliminated_temperature);
            accepted_coupled_residual =
                coupled_residual_norm(state.energy, state.temperature);
            return true;
        }
        copy_level_data(state.temperature, eliminated_temperature);
        accepted_coupled_residual =
            evaluated_coupled_residual_norm(state.energy, state.temperature);
        return true;
    }
    // Lagged work: freeze the work flux, converge the reduced system to a
    // tolerance tied to the unlagged residual, and repeat until the unlagged
    // residual converges.  (The change in E between lags is not a stopping
    // criterion: an inner solve may still move E after the unlagged residual
    // has converged.)
    int lag = 0;
    // Whether eliminated_temperature already holds T(E) for the current E
    // and lagged flux, as the inner solve's last residual() leaves it.
    bool eliminated = false;
    for (;; ++lag) {
        if (!eliminated && !eliminate(state.energy, false)) {
            return fail("Cell-local material solve did not converge");
        }
        copy_level_data(state.temperature, eliminated_temperature);
        Real const unlagged =
            coupled_residual_norm(state.energy, state.temperature);
        if (lag > 0 && unlagged <= tolerance) {
            accepted_coupled_residual = unlagged;
            break;
        }
        if (lag >= options.maximum_lag_iterations) {
            return fail("Lagged work iteration did not converge");
        }
        freeze_work_flux();
        NewtonKrylovOptions inner = newton_options;
        // Tighter than the outer target, or a converged inner residual can
        // stall the lag just above it.
        inner.nonlinear_tolerance =
            Real(0.1) * amrex::max(tolerance, unlagged);
        ReducedProblem problem(*this, state.energy);
        NewtonKrylovSolver<ReducedProblem> newton(problem, inner);
        auto const result = newton.solve(state.energy);
        rm_record_newton(counters, result, step_nonlinear);
        ++step_lag;
        if (!result.converged) {
            return fail("Lagged inner Newton-Krylov did not converge");
        }
        eliminated = problem.evaluated_at(state.energy);
    }
    return true;
}

// Explicit material advection, then the implicit update.  Newton starts
// from the previous step's state; if it fails, the step is retried from the
// predictor iterations' state.  With predict_first the order is reversed:
// the predictors are deterministic, so retrying from them would only repeat
// the failed attempt.  Predictors whose linear solve fails, or that leave
// E or T inadmissible, count as a failed attempt.
void
RadiationMaterialSolver::Impl::advance (RadiationMaterialState& state,
                                        Real dt)
{
    BL_PROFILE("FLD::rm_advance");
    physics.dt = dt;
    average_down_hierarchy(state.energy, hierarchy);
    average_down_hierarchy(state.temperature, hierarchy);
    copy_level_data(old_energy, state.energy);
    if (physics.moving) { explicit_material_advection(state.temperature); }
    copy_level_data(old_temperature, state.temperature);
    // k = sigma T^3 (4 - p + p E/T^4) cannot be negative for 0 <= p <= 4
    // and E >= 0, so skip the sweep and its reduction.
    if (physics.opacity_power < Real(0) || physics.opacity_power > Real(4)) {
        counters.minimum_runaway_time =
            amrex::min(counters.minimum_runaway_time,
                       runaway_time(state.energy, state.temperature));
    }
    RadiationMaterialState const start{clone_level_data(state.energy),
                                       clone_level_data(state.temperature)};
    int step_nonlinear = 0;
    accepted_coupled_residual = Real(-1);
    step_lag = 0;
    for (int attempt = 0; attempt < 2; ++attempt) {
        bool const fallback = attempt == 1;
        if (fallback) {
            copy_level_data(state.energy, start.energy);
            copy_level_data(state.temperature, start.temperature);
            ++counters.predictor_fallbacks;
        }
        if (fallback != options.predict_first && !predictors(state)) {
            if (fallback) {
                amrex::Abort("Predictors failed: a linear solve did not "
                             "converge or E or T is inadmissible");
            }
            continue;
        }
        if (attempt_step(state, !fallback, step_nonlinear)) { break; }
    }
    average_down_hierarchy(state.energy, hierarchy);
    average_down_hierarchy(state.temperature, hierarchy);
    counters.nonlinear += step_nonlinear;
    counters.maximum_nonlinear =
        amrex::max(counters.maximum_nonlinear, step_nonlinear);
    counters.lag_iterations += step_lag;
    counters.maximum_lag_iterations =
        amrex::max(counters.maximum_lag_iterations, step_lag);
    // coupled_residual_norm averages down its work copies, so the
    // average_down above does not change it.
    counters.final_coupled_residual =
        accepted_coupled_residual >= Real(0)
            ? accepted_coupled_residual
            : coupled_residual_norm(state.energy, state.temperature);
    counters.maximum_coupled_residual = amrex::max(
        counters.maximum_coupled_residual, counters.final_coupled_residual);
}

RadiationMaterialSolver::RadiationMaterialSolver (
    DiffusionHierarchy const& hierarchy, RadiationMaterialOptions options)
    : m_impl(std::make_unique<Impl>(hierarchy, std::move(options)))
{}

RadiationMaterialSolver::~RadiationMaterialSolver () = default;

RadiationMaterialState
RadiationMaterialSolver::make_state () const
{
    return {make_cell_data(m_impl->hierarchy, 1, 1),
            make_cell_data(m_impl->hierarchy, 1, 1)};
}

void
RadiationMaterialSolver::set_opacity_coefficient (LevelData const& kappa)
{
    copy_level_data(m_impl->kappa, kappa);
    average_down_hierarchy(m_impl->kappa, m_impl->hierarchy);
    fill_level_ghosts(m_impl->kappa, m_impl->hierarchy);
}

void
RadiationMaterialSolver::set_sources (LevelData const& energy_source,
                                      LevelData const& temperature_source)
{
    copy_level_data(m_impl->source_energy, energy_source);
    copy_level_data(m_impl->source_temperature, temperature_source);
}

void
RadiationMaterialSolver::advance (RadiationMaterialState& state, Real dt)
{
    m_impl->advance(state, dt);
}

Real
RadiationMaterialSolver::runaway_time (RadiationMaterialState const& state)
{
    return m_impl->runaway_time(state.energy, state.temperature);
}

EnergyBudget
RadiationMaterialSolver::energy_budget (RadiationMaterialState const& state)
{
    Impl& s = *m_impl;
    s.load_state(state.energy, state.temperature);
    Real result[3] = {};
    for (int level = 0; level < s.levels(); ++level) {
        RMPhysics const p = s.level_physics(level);
        Real const volume = p.dx[0] * p.dx[1] * p.dx[2];
        ReduceOps<ReduceOpSum, ReduceOpSum, ReduceOpSum> reduce_op;
        ReduceData<Real, Real, Real> reduce_data(reduce_op);
        using Tuple = typename decltype(reduce_data)::Type;
        for (MFIter mfi(*s.energy[level]); mfi.isValid(); ++mfi) {
            auto const e = s.energy[level]->const_array(mfi);
            auto const t = s.temperature[level]->const_array(mfi);
            auto const k = s.kappa[level]->const_array(mfi);
            auto const b = rm_faces(s.bcoef, level, mfi);
            auto const rb = s.robin_b[level]->const_array(mfi);
            auto const se = s.source_energy[level]->const_array(mfi);
            auto const st = s.source_temperature[level]->const_array(mfi);
            auto const mask = s.masks_[level].const_array(mfi);
            reduce_op.eval(mfi.validbox(), reduce_data,
            [=] AMREX_GPU_DEVICE (int i, int j, int kk) noexcept -> Tuple
            {
                IntVect const c(AMREX_D_DECL(i, j, kk));
                if (!mask(c)) { return {Real(0), Real(0), Real(0)}; }
                Real work = Real(0);
                if (p.moving) {
                    auto const flux = rm_cell_flux(p, e, b, rb, c);
                    work = rm_opacity(p, k(c), t(c)) *
                           rm_dot(p.velocity, flux);
                }
                return {(e(c) + t(c)) * volume, work * volume,
                        (se(c) + st(c)) * volume};
            });
        }
        auto const sums = reduce_data.value(reduce_op);
        result[0] += amrex::get<0>(sums);
        result[1] += amrex::get<1>(sums);
        result[2] += amrex::get<2>(sums);
    }
    ParallelDescriptor::ReduceRealSum(result, 3);
    auto const outflow = s.boundary_outflow();
    Real input = Real(0);
    for (Real const value : outflow) { input -= value; }
    return {result[0], input, result[1], result[2]};
}

Array<Real, 2 * AMREX_SPACEDIM>
RadiationMaterialSolver::boundary_outflow (RadiationMaterialState const& state)
{
    m_impl->load_state(state.energy, state.temperature);
    return m_impl->boundary_outflow();
}

void
RadiationMaterialSolver::fill_cell_coefficients (
    RadiationMaterialState const& state, LevelData& sigma,
    LevelData& diffusion)
{
    Impl& s = *m_impl;
    s.load_state(state.energy, state.temperature);
    for (int level = 0; level < s.levels(); ++level) {
        RMPhysics const p = s.level_physics(level);
        for (MFIter mfi(*sigma[level]); mfi.isValid(); ++mfi) {
            auto const e = s.energy[level]->const_array(mfi);
            auto const t = s.temperature[level]->const_array(mfi);
            auto const k = s.kappa[level]->const_array(mfi);
            auto const dc = s.cell_diffusion[level]->const_array(mfi);
            auto const so = sigma[level]->array(mfi);
            auto const dout = diffusion[level]->array(mfi);
            ParallelFor(mfi.validbox(),
            [=] AMREX_GPU_DEVICE (int i, int j, int kk) noexcept
            {
                IntVect const c(AMREX_D_DECL(i, j, kk));
                Real const sigma_value = rm_opacity(p, k(c), t(c));
                so(c) = sigma_value;
                if (p.diffusion != rm_face_limiter) {
                    dout(c) = dc(c);
                    return;
                }
                Real gradient_squared = Real(0);
                for (int d = 0; d < 3; ++d) {
                    IntVect lo = c - IntVect::TheDimensionVector(d);
                    IntVect hi = c + IntVect::TheDimensionVector(d);
                    if (p.boundary[2 * d] != rm_periodic) {
                        if (c[d] == p.domain_lo[d]) { lo = c; }
                        if (c[d] == p.domain_hi[d]) { hi = c; }
                    }
                    // One non-periodic cell across: no gradient.
                    Real const g = hi[d] == lo[d]
                        ? Real(0)
                        : (e(hi) - e(lo)) / (Real(hi[d] - lo[d]) * p.dx[d]);
                    gradient_squared += g * g;
                }
                Real const limiter = p.limiter
                    ? std::sqrt(gradient_squared) /
                          amrex::max(e(c), Real(1.e-30))
                    : Real(0);
                dout(c) = Real(1) / (Real(3) * sigma_value + limiter);
            });
        }
    }
    average_down_hierarchy(sigma, s.hierarchy);
    average_down_hierarchy(diffusion, s.hierarchy);
}

void
RadiationMaterialSolver::fill_cell_flux (RadiationMaterialState const& state,
                                         LevelData& flux)
{
    Impl& s = *m_impl;
    s.load_state(state.energy, state.temperature);
    for (int level = 0; level < s.levels(); ++level) {
        RMPhysics const p = s.level_physics(level);
        for (MFIter mfi(*flux[level]); mfi.isValid(); ++mfi) {
            auto const e = s.energy[level]->const_array(mfi);
            auto const b = rm_faces(s.bcoef, level, mfi);
            auto const rb = s.robin_b[level]->const_array(mfi);
            auto const out = flux[level]->array(mfi);
            ParallelFor(mfi.validbox(),
            [=] AMREX_GPU_DEVICE (int i, int j, int kk) noexcept
            {
                IntVect const c(AMREX_D_DECL(i, j, kk));
                auto const f = rm_cell_flux(p, e, b, rb, c);
                for (int d = 0; d < 3; ++d) { out(i, j, kk, d) = f[d]; }
            });
        }
    }
    average_down_hierarchy(flux, s.hierarchy);
}

RadiationMaterialCounters const&
RadiationMaterialSolver::counters () const
{
    return m_impl->counters;
}

Vector<iMultiFab> const&
RadiationMaterialSolver::masks () const
{
    return m_impl->masks_;
}

} // namespace fld_test
