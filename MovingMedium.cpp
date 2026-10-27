#include "FLDTestCommon.H"
#include "NewtonKrylovSolver.H"

#include <AMReX_MFIter.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Reduce.H>

#include <cmath>
#include <iomanip>
#include <limits>
#include <string>

// Grey flux-limited diffusion coupled to a material energy equation in a
// medium moving with a prescribed uniform velocity, to first order in v/c.
// The four-force is the mixed-frame form of He, Wibking & Krumholz (2024,
// MNRAS 531, 1228), Eqs. (3)-(4), truncated to O(v/c).  In units with
// a_r = c = rho c_v = 1 and v measured in units of c:
//
//   dE/dt + div[F0 + v.(E I + P)] = sigma (T^4 - E) + sigma v.F0
//   dT/dt + div(v T)               = sigma (E - T^4) - 2 sigma v.F0
//
// F0 = -D grad E is the comoving FLD flux and P the Levermore Eddington
// tensor of the limiter.  The material source is c G0 - v.G (internal
// energy); the total E + T loses sigma v.F0, the work done by the radiation
// force on the (prescribed) flow.  Material advection is explicit and
// operator split before the implicit radiation-material update, as for the
// hydrodynamic update in the paper, so the material equation stays local
// to each cell apart from the work term.
//
// The domain is the periodic unit cube on one level.  Solvers
// (vc.solver):
//   newton_krylov        coupled (E, T) JFNK with a linear Schur-complement
//                        (Fleck) preconditioner
//   ne_lagged            nonlinear elimination with the work term's flux
//                        lagged over an outer iteration

namespace fld_test
{

using namespace amrex;

namespace
{

struct MovingPhysics
{
    // Defaults: optically thick (tau ~ 1e3-1e4 across the box).  Optically
    // thin, limiter-dominated cases need the predictor fallback.
    Real sigma0 = Real(1000);
    Real opacity_power = Real(3);
    GpuArray<Real, 3> velocity{};
    GpuArray<Real, 3> dx{};
    Real dt = Real(1);
    bool limiter = true;
    bool upwind = true;
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE Real
mm_positive (Real t) noexcept
{
    return amrex::max(t, Real(1.e-6));
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE Real
mm_fourth (Real t) noexcept
{
    Real const t2 = t * t;
    return t2 * t2;
}

// sigma0 T^-p, avoiding pow() for small integer p: the cell-local solves
// evaluate this on every face of every iteration.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE Real
mm_opacity (MovingPhysics const& p, Real t) noexcept
{
    int const integer_power = static_cast<int>(p.opacity_power);
    if (Real(integer_power) == p.opacity_power && integer_power >= 0 &&
        integer_power <= 8) {
        Real const tp = mm_positive(t);
        Real denominator = Real(1);
        for (int n = 0; n < integer_power; ++n) { denominator *= tp; }
        return p.sigma0 / denominator;
    }
    return p.sigma0 * std::pow(mm_positive(t), -p.opacity_power);
}

// FLD diffusion coefficient on the face between cells L and R.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE Real
mm_face_diffusion (MovingPhysics const& p, Real el, Real er, Real tl, Real tr,
                   Real h) noexcept
{
    Real const sigma = mm_opacity(
        p, Real(0.5) * (mm_positive(tl) + mm_positive(tr)));
    Real const limiter = p.limiter
        ? std::abs(er - el) / h /
              amrex::max(Real(0.5) * (el + er), Real(1.e-30))
        : Real(0);
    return Real(1) / (Real(3) * sigma + limiter);
}

// Comoving flux F0 on the face between cells L and R (positive toward R).
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE Real
mm_face_flux (MovingPhysics const& p, Real el, Real er, Real tl, Real tr,
              Real h) noexcept
{
    return -mm_face_diffusion(p, el, er, tl, tr, h) * (er - el) / h;
}

// Cell-centered F0: the mean of the two face fluxes in each direction, with
// the cell's own temperature replaced by tc.  With tc = t(i,j,k) this is the
// same discrete flux as the divergence, so the work-energy budget closes.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE GpuArray<Real, 3>
mm_cell_flux (MovingPhysics const& p, Array4<Real const> const& e,
              Array4<Real const> const& t, int i, int j, int k,
              Real tc) noexcept
{
    GpuArray<Real, 3> flux{};
    IntVect const c(AMREX_D_DECL(i, j, k));
    Real const ec = e(c);
    for (int d = 0; d < 3; ++d) {
        IntVect const s = IntVect::TheDimensionVector(d);
        Real const lower = mm_face_flux(p, e(c - s), ec, t(c - s), tc, p.dx[d]);
        Real const upper = mm_face_flux(p, ec, e(c + s), tc, t(c + s), p.dx[d]);
        flux[d] = Real(0.5) * (lower + upper);
    }
    return flux;
}

// Radiation enthalpy flux v.(E I + P) at a cell center, with the Levermore
// tensor P = E[(1 - f)/2 I + (3f - 1)/2 n n], f = lambda + (lambda R)^2,
// lambda = 1/(3 + R), consistent with D = lambda/sigma.  Without the
// limiter f = 1/3 and this is (4/3) v E.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE GpuArray<Real, 3>
mm_enthalpy_flux (MovingPhysics const& p, Array4<Real const> const& e,
                  Array4<Real const> const& t, int i, int j, int k) noexcept
{
    Real const ec = e(i, j, k);
    Real f = Real(1) / Real(3);
    GpuArray<Real, 3> n{};
    if (p.limiter) {
        Real const g[3] = {
            (e(i + 1, j, k) - e(i - 1, j, k)) / (Real(2) * p.dx[0]),
            (e(i, j + 1, k) - e(i, j - 1, k)) / (Real(2) * p.dx[1]),
            (e(i, j, k + 1) - e(i, j, k - 1)) / (Real(2) * p.dx[2])};
        Real const gmag = std::sqrt(g[0] * g[0] + g[1] * g[1] + g[2] * g[2]);
        Real const r = gmag / (mm_opacity(p, t(i, j, k)) *
                               amrex::max(ec, Real(1.e-30)));
        Real const lambda = Real(1) / (Real(3) + r);
        f = lambda + lambda * lambda * r * r;
        if (gmag > Real(0)) {
            for (int d = 0; d < 3; ++d) { n[d] = g[d] / gmag; }
        }
    }
    Real const vn = p.velocity[0] * n[0] + p.velocity[1] * n[1] +
                    p.velocity[2] * n[2];
    GpuArray<Real, 3> a{};
    for (int d = 0; d < 3; ++d) {
        a[d] = ec * (p.velocity[d] +
                     Real(0.5) * (Real(1) - f) * p.velocity[d] +
                     Real(0.5) * (Real(3) * f - Real(1)) * vn * n[d]);
    }
    return a;
}

// Face value of a cell-centered advective flux component.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE Real
mm_face_advection (MovingPhysics const& p, int d, Real left,
                   Real right) noexcept
{
    if (!p.upwind) { return Real(0.5) * (left + right); }
    return p.velocity[d] >= Real(0) ? left : right;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE Real
mm_dot (GpuArray<Real, 3> const& a, GpuArray<Real, 3> const& b) noexcept
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

enum class WorkCoupling { Current, Lagged };

struct MovingState
{
    LevelData energy;
    LevelData temperature;
};

// Everything one time step needs, shared by the problem classes.
struct MovingStep
{
    MovingStep (DiffusionHierarchy const& a_hierarchy,
                Vector<iMultiFab> const& a_masks, MovingPhysics a_physics,
                MLABecLapAMG& a_solver, SolverSummary& a_summary)
        : hierarchy(a_hierarchy), masks(a_masks), physics(a_physics),
          solver(a_solver), summary(a_summary),
          old_energy(make_cell_data(a_hierarchy, 1, 0)),
          old_temperature(make_cell_data(a_hierarchy, 1, 0)),
          source_energy(make_cell_data(a_hierarchy, 1, 0)),
          source_temperature(make_cell_data(a_hierarchy, 1, 0)),
          energy(make_cell_data(a_hierarchy, 1, 2)),
          temperature(make_cell_data(a_hierarchy, 1, 2)),
          eliminated_temperature(make_cell_data(a_hierarchy, 1, 2)),
          enthalpy_flux(make_cell_data(a_hierarchy, 3, 1)),
          lagged_flux(make_cell_data(a_hierarchy, 3, 0)),
          residual_energy(make_cell_data(a_hierarchy, 1, 0)),
          residual_temperature(make_cell_data(a_hierarchy, 1, 0)),
          temperature_derivative(make_cell_data(a_hierarchy, 1, 0)),
          acoef(make_cell_data(a_hierarchy, 1, 0)),
          schur_k(make_cell_data(a_hierarchy, 1, 0)),
          schur_sigma(make_cell_data(a_hierarchy, 1, 0)),
          schur_diagonal(make_cell_data(a_hierarchy, 1, 0)),
          bcoef(make_face_data(a_hierarchy))
    {
        set_level_data(source_energy, Real(0));
        set_level_data(source_temperature, Real(0));
        set_level_data(lagged_flux, Real(0));
    }

    DiffusionHierarchy const& hierarchy;
    Vector<iMultiFab> const& masks;
    MovingPhysics physics;
    MLABecLapAMG& solver;
    SolverSummary& summary;
    LevelData old_energy;      // E^n
    LevelData old_temperature; // T^n after explicit material advection
    LevelData source_energy;   // manufactured sources (zero otherwise)
    LevelData source_temperature;
    LevelData energy;          // work copies with periodic halos
    LevelData temperature;
    LevelData eliminated_temperature; // T(E), also the warm start
    LevelData enthalpy_flux;
    LevelData lagged_flux;
    LevelData residual_energy;
    LevelData residual_temperature;
    LevelData temperature_derivative;
    LevelData acoef;
    LevelData schur_k;
    LevelData schur_sigma;
    LevelData schur_diagonal;
    FaceData bcoef;
    int maximum_local_iterations = 0;
    Long eliminations = 0;
};

Periodicity
mm_periodicity (MovingStep const& s)
{
    return s.hierarchy.geom[0].periodicity();
}

// Copy (E, T) into the halo-carrying work fields and evaluate the cell
// enthalpy fluxes, including one halo layer for the face upwinding.
void
mm_load_state (MovingStep& s, LevelData const& energy,
               LevelData const& temperature)
{
    copy_level_data(s.energy, energy);
    copy_level_data(s.temperature, temperature);
    s.energy[0]->FillBoundary(mm_periodicity(s));
    s.temperature[0]->FillBoundary(mm_periodicity(s));
    MovingPhysics const p = s.physics;
    for (MFIter mfi(*s.enthalpy_flux[0]); mfi.isValid(); ++mfi) {
        auto const e = s.energy[0]->const_array(mfi);
        auto const t = s.temperature[0]->const_array(mfi);
        auto const a = s.enthalpy_flux[0]->array(mfi);
        ParallelFor(mfi.growntilebox(1),
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            auto const flux = mm_enthalpy_flux(p, e, t, i, j, k);
            for (int d = 0; d < 3; ++d) { a(i, j, k, d) = flux[d]; }
        });
    }
}

// Residuals of both equations at the loaded state.  The material work term
// uses the current flux, or the lagged flux when coupling is Lagged.
void
mm_evaluate_residual (MovingStep& s, WorkCoupling coupling)
{
    MovingPhysics const p = s.physics;
    Real const inverse_dt = Real(1) / p.dt;
    bool const lagged = coupling == WorkCoupling::Lagged;
    for (MFIter mfi(*s.residual_energy[0]); mfi.isValid(); ++mfi) {
        auto const e = s.energy[0]->const_array(mfi);
        auto const t = s.temperature[0]->const_array(mfi);
        auto const a = s.enthalpy_flux[0]->const_array(mfi);
        auto const eo = s.old_energy[0]->const_array(mfi);
        auto const to = s.old_temperature[0]->const_array(mfi);
        auto const se = s.source_energy[0]->const_array(mfi);
        auto const st = s.source_temperature[0]->const_array(mfi);
        auto const lag = s.lagged_flux[0]->const_array(mfi);
        auto const re = s.residual_energy[0]->array(mfi);
        auto const rt = s.residual_temperature[0]->array(mfi);
        ParallelFor(mfi.validbox(),
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            IntVect const c(AMREX_D_DECL(i, j, k));
            Real const ec = e(c);
            Real const tc = t(c);
            Real divergence = Real(0);
            GpuArray<Real, 3> flux{};
            for (int d = 0; d < 3; ++d) {
                IntVect const sft = IntVect::TheDimensionVector(d);
                IntVect const lo = c - sft;
                IntVect const hi = c + sft;
                Real const f_lo = mm_face_flux(p, e(lo), ec, t(lo), tc, p.dx[d]);
                Real const f_hi = mm_face_flux(p, ec, e(hi), tc, t(hi), p.dx[d]);
                Real const a_lo = mm_face_advection(p, d, a(lo, d), a(c, d));
                Real const a_hi = mm_face_advection(p, d, a(c, d), a(hi, d));
                divergence += ((f_hi + a_hi) - (f_lo + a_lo)) / p.dx[d];
                flux[d] = Real(0.5) * (f_lo + f_hi);
            }
            Real const sigma = mm_opacity(p, tc);
            Real const t4 = mm_fourth(mm_positive(tc));
            Real const work = mm_dot(p.velocity, flux);
            GpuArray<Real, 3> const lagged_flux{
                lag(i, j, k, 0), lag(i, j, k, 1), lag(i, j, k, 2)};
            Real const material_work =
                lagged ? mm_dot(p.velocity, lagged_flux) : work;
            re(c) = inverse_dt * (ec - eo(c)) + divergence -
                    sigma * (t4 - ec) - sigma * work - se(c);
            rt(c) = inverse_dt * (tc - to(c)) - sigma * (ec - t4) +
                    Real(2) * sigma * material_work - st(c);
        });
    }
}

// Store the current cell flux F0 as the lagged work flux.
void
mm_freeze_work_flux (MovingStep& s)
{
    MovingPhysics const p = s.physics;
    for (MFIter mfi(*s.lagged_flux[0]); mfi.isValid(); ++mfi) {
        auto const e = s.energy[0]->const_array(mfi);
        auto const t = s.temperature[0]->const_array(mfi);
        auto const lag = s.lagged_flux[0]->array(mfi);
        ParallelFor(mfi.validbox(),
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            auto const flux = mm_cell_flux(p, e, t, i, j, k, t(i, j, k));
            for (int d = 0; d < 3; ++d) { lag(i, j, k, d) = flux[d]; }
        });
    }
}

// Cell-local material solves for the energy in s.energy, with the work term
// evaluated from the lagged flux.  The material residual g(T) is solved
// with a bracketed Newton iteration (analytic slope, bisection fallback),
// starting from and overwriting s.eliminated_temperature, and optionally
// returns dT/dE.  Returns the largest iteration count, or -1 if any cell
// failed.
int
mm_cell_solves (MovingStep& s, LevelData* derivative)
{
    BL_PROFILE("FLD::vc_cell_solves");
    MovingPhysics const p = s.physics;
    Real const inverse_dt = Real(1) / p.dt;
    bool const want_derivative = derivative != nullptr;
    int constexpr maximum_iterations = 100;
    ReduceOps<ReduceOpMax, ReduceOpMin> reduce_op;
    ReduceData<int, int> reduce_data(reduce_op);
    using Tuple = typename decltype(reduce_data)::Type;
    for (MFIter mfi(*s.eliminated_temperature[0]); mfi.isValid(); ++mfi) {
        auto const e = s.energy[0]->const_array(mfi);
        auto const to = s.old_temperature[0]->const_array(mfi);
        auto const st = s.source_temperature[0]->const_array(mfi);
        auto const lag = s.lagged_flux[0]->const_array(mfi);
        auto const out = s.eliminated_temperature[0]->array(mfi);
        auto const tau = want_derivative
            ? (*derivative)[0]->array(mfi) : Array4<Real>{};
        reduce_op.eval(mfi.validbox(), reduce_data,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept -> Tuple
        {
            Real const ec = e(i, j, k);
            GpuArray<Real, 3> const lagged_flux{
                lag(i, j, k, 0), lag(i, j, k, 1), lag(i, j, k, 2)};
            // g(T) and dg/dT.
            Real const work = mm_dot(p.velocity, lagged_flux);
            auto evaluate = [&] (Real temperature, Real& slope) -> Real {
                Real const tp = mm_positive(temperature);
                Real const sigma = mm_opacity(p, temperature);
                Real const dsigma = temperature > Real(1.e-6)
                    ? -p.opacity_power * sigma / tp : Real(0);
                Real const t4 = mm_fourth(tp);
                Real const dt4 = temperature > Real(1.e-6)
                    ? Real(4) * tp * tp * tp : Real(0);
                slope = inverse_dt - dsigma * (ec - t4) + sigma * dt4 +
                        Real(2) * dsigma * work;
                return inverse_dt * (temperature - to(i, j, k)) -
                       sigma * (ec - t4) + Real(2) * sigma * work -
                       st(i, j, k);
            };
            auto g = [&] (Real temperature) -> Real {
                Real unused;
                return evaluate(temperature, unused);
            };
            Real x = mm_positive(out(i, j, k));
            Real lo = x;
            Real hi = x;
            int expansions = 0;
            if (g(x) <= Real(0)) {
                while (g(hi) < Real(0) && expansions < 200) {
                    lo = hi;
                    hi *= Real(2);
                    ++expansions;
                }
            } else {
                while (g(lo) > Real(0) && expansions < 200) {
                    hi = lo;
                    lo *= Real(0.5);
                    ++expansions;
                }
            }
            if (expansions >= 200) { return {0, 0}; }
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
                    if (std::abs(step) <= Real(1.e-13) * x) {
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
                converged = hi - lo <= Real(1.e-13) * hi;
            }
            out(i, j, k) = x;
            if (want_derivative) {
                // dT/dE = -g_E/g_T with g_E = -sigma(T).
                Real gp;
                evaluate(x, gp);
                tau(i, j, k) = mm_opacity(p, x) / gp;
            }
            return {iterations, converged ? 1 : 0};
        });
    }
    auto const values = reduce_data.value(reduce_op);
    int iterations = amrex::get<0>(values);
    int all_converged = amrex::get<1>(values);
    ParallelDescriptor::ReduceIntMax(iterations);
    ParallelDescriptor::ReduceIntMin(all_converged);
    return all_converged ? iterations : -1;
}

// T(E) for the energy in `energy`, with the lagged work flux.
void
mm_eliminate (MovingStep& s, LevelData const& energy, bool with_derivative)
{
    BL_PROFILE("FLD::vc_eliminate");
    copy_level_data(s.energy, energy);
    s.energy[0]->FillBoundary(mm_periodicity(s));
    ++s.eliminations;
    int const iterations = mm_cell_solves(
        s, with_derivative ? &s.temperature_derivative : nullptr);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        iterations >= 0, "Moving-medium cell-local solve did not converge");
    s.maximum_local_iterations =
        amrex::max(s.maximum_local_iterations, iterations);
}

Real
mm_scale (LevelData const& field, Vector<iMultiFab> const& masks)
{
    return amrex::max(composite_maximum(field, masks), Real(1.e-30));
}

// Scaled norm of the full (unlagged) coupled residual at (E, T).
Real
mm_coupled_residual_norm (MovingStep& s, LevelData const& energy,
                          LevelData const& temperature)
{
    mm_load_state(s, energy, temperature);
    mm_evaluate_residual(s, WorkCoupling::Current);
    Real const es = mm_scale(energy, s.masks);
    Real const ts = mm_scale(temperature, s.masks);
    return std::sqrt(
        composite_weighted_dot(s.residual_energy, s.residual_energy,
                               s.hierarchy, s.masks) / (es * es) +
        composite_weighted_dot(s.residual_temperature,
                               s.residual_temperature, s.hierarchy,
                               s.masks) / (ts * ts));
}

// Face diffusion coefficients of the loaded state.
void
mm_fill_bcoef (MovingStep& s)
{
    MovingPhysics const p = s.physics;
    for (int d = 0; d < 3; ++d) {
        for (MFIter mfi(*s.bcoef[0][d]); mfi.isValid(); ++mfi) {
            auto const e = s.energy[0]->const_array(mfi);
            auto const t = s.temperature[0]->const_array(mfi);
            auto const b = s.bcoef[0][d]->array(mfi);
            ParallelFor(mfi.validbox(),
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                IntVect const c(AMREX_D_DECL(i, j, k));
                IntVect const lo = c - IntVect::TheDimensionVector(d);
                b(c) = mm_face_diffusion(p, e(lo), e(c), t(lo), t(c), p.dx[d]);
            });
        }
    }
}

void
mm_setup_solver (MovingStep& s)
{
    Array<LinOpBCType, AMREX_SPACEDIM> const periodic{
        AMREX_D_DECL(LinOpBCType::Periodic, LinOpBCType::Periodic,
                     LinOpBCType::Periodic)};
    s.solver.setup(Real(1), Real(1), get_level_const_ptrs(s.acoef),
                   get_face_const_ptrs(s.bcoef), periodic, periodic, {});
    record_setup(s.summary, s.solver);
}

void
mm_positive_update (LevelData& candidate, LevelData const& state,
                    LevelData const& correction, Real step_length)
{
    for (MFIter mfi(*candidate[0]); mfi.isValid(); ++mfi) {
        auto const next = candidate[0]->array(mfi);
        auto const x = state[0]->const_array(mfi);
        auto const dx = correction[0]->const_array(mfi);
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

bool
mm_admissible (LevelData const& field, Vector<iMultiFab> const& masks)
{
    auto const [minimum, maximum] = composite_minimum_maximum(field, masks);
    amrex::ignore_unused(maximum);
    return composite_all_finite(field, masks) && minimum > Real(0);
}

// Nonlinear elimination: Newton-Krylov on R(E) = R_E + R_T with T = T(E)
// and the work term's flux lagged, preconditioned by one AMG cycle of
// (1 + dT/dE)/dt - div D grad.  The enthalpy advection and work terms are
// left out of the preconditioner.
class ReducedProblem
{
  public:
    using State = LevelData;
    using RT = Real;

    // The norm scale is set from the initial state, because the Newton
    // solver checks the initial residual before the first prepare().
    ReducedProblem (MovingStep& step, State const& initial)
        : m_step(step), m_scale(mm_scale(initial, step.masks))
    {}

    void prepare (State const& energy)
    {
        BL_PROFILE("FLD::vc_reduced_prepare");
        m_scale = mm_scale(energy, m_step.masks);
        mm_eliminate(m_step, energy, true);
        mm_load_state(m_step, energy, m_step.eliminated_temperature);
        mm_fill_bcoef(m_step);
        Real const inverse_dt = Real(1) / m_step.physics.dt;
        for (MFIter mfi(*m_step.acoef[0]); mfi.isValid(); ++mfi) {
            auto const tau = m_step.temperature_derivative[0]->const_array(mfi);
            auto const a = m_step.acoef[0]->array(mfi);
            ParallelFor(mfi.validbox(),
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                a(i, j, k) = inverse_dt * (Real(1) + tau(i, j, k));
            });
        }
        mm_setup_solver(m_step);
    }

    void residual (State const& energy, State& output)
    {
        mm_eliminate(m_step, energy, false);
        mm_load_state(m_step, energy, m_step.eliminated_temperature);
        mm_evaluate_residual(m_step, WorkCoupling::Lagged);
        lincomb_level_data(output, Real(1), m_step.residual_energy, Real(1),
                           m_step.residual_temperature);
    }

    void linearized_residual (State const& energy, State& output)
    {
        residual(energy, output);
    }

    State makeVecRHS () const { return make_cell_data(m_step.hierarchy, 1, 0); }
    State makeVecLHS () const { return make_cell_data(m_step.hierarchy, 1, 1); }

    void precondition (State& output, State const& rhs)
    {
        set_level_data(output, Real(0));
        m_step.solver.precondition(get_level_ptrs(output),
                                   get_level_const_ptrs(rhs));
    }

    void assign (State& lhs, State const& rhs) const { copy_level_data(lhs, rhs); }
    void copy_state (State& lhs, State const& rhs) const
    {
        copy_level_data_with_ghosts(lhs, rhs);
    }
    Real dotProduct (State const& lhs, State const& rhs) const
    {
        return composite_weighted_dot(lhs, rhs, m_step.hierarchy,
                                      m_step.masks) / (m_scale * m_scale);
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
        return composite_weighted_dot(x, x, m_step.hierarchy, m_step.masks);
    }
    Real norm2FromData (Real sum) const
    {
        return std::sqrt(amrex::max(sum / (m_scale * m_scale), Real(0)));
    }
    void scale (State& x, Real a) const { x[0]->mult(a, 0, 1, 0); }
    void setToZero (State& x) const { set_level_data(x, Real(0)); }
    State clone (State const& x) const { return clone_level_data(x); }
    void fill_candidate (State& candidate, State const& state,
                         State const& correction, Real step_length) const
    {
        mm_positive_update(candidate, state, correction, step_length);
    }
    bool admissible (State const& state) const
    {
        return mm_admissible(state, m_step.masks);
    }
    Real relative_change (State const& lhs, State const& rhs) const
    {
        return composite_maximum_relative_change(lhs, rhs, m_step.masks);
    }

  private:
    MovingStep& m_step;
    Real m_scale = Real(1);
};

// Coupled (E, T) Newton-Krylov.  The preconditioner eliminates the
// cell-local material block linearly (Schur complement, the Fleck factor):
// with k = dR_E/dT and t_d = dR_T/dT,
//   (1/dt + sigma - k sigma / t_d - div D grad) dE = r_E + k r_T / t_d,
//   dT = (r_T + sigma dE) / t_d.
class CoupledProblem
{
  public:
    using State = MovingState;
    using RT = Real;

    // Norm scales from the initial state; see ReducedProblem.
    CoupledProblem (MovingStep& step, State const& initial)
        : m_step(step),
          m_energy_scale(mm_scale(initial.energy, step.masks)),
          m_temperature_scale(mm_scale(initial.temperature, step.masks))
    {}

    void prepare (State const& state)
    {
        BL_PROFILE("FLD::vc_coupled_prepare");
        m_energy_scale = mm_scale(state.energy, m_step.masks);
        m_temperature_scale = mm_scale(state.temperature, m_step.masks);
        mm_load_state(m_step, state.energy, state.temperature);
        mm_fill_bcoef(m_step);
        MovingPhysics const p = m_step.physics;
        Real const inverse_dt = Real(1) / p.dt;
        for (MFIter mfi(*m_step.acoef[0]); mfi.isValid(); ++mfi) {
            auto const e = m_step.energy[0]->const_array(mfi);
            auto const t = m_step.temperature[0]->const_array(mfi);
            auto const a = m_step.acoef[0]->array(mfi);
            auto const kk = m_step.schur_k[0]->array(mfi);
            auto const ss = m_step.schur_sigma[0]->array(mfi);
            auto const td = m_step.schur_diagonal[0]->array(mfi);
            ParallelFor(mfi.validbox(),
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                Real const tc = mm_positive(t(i, j, k));
                Real const sigma = mm_opacity(p, tc);
                Real const dsigma = -p.opacity_power * sigma / tc;
                Real const t3 = tc * tc * tc;
                // k = d/dT [sigma (T^4 - E)]
                Real const coupling = dsigma * (tc * t3 - e(i, j, k)) +
                                      Real(4) * sigma * t3;
                Real const diagonal = inverse_dt + coupling;
                kk(i, j, k) = coupling;
                ss(i, j, k) = sigma;
                td(i, j, k) = diagonal;
                a(i, j, k) = inverse_dt + sigma -
                             coupling * sigma / diagonal;
            });
        }
        mm_setup_solver(m_step);
    }

    void residual (State const& state, State& output)
    {
        mm_load_state(m_step, state.energy, state.temperature);
        mm_evaluate_residual(m_step, WorkCoupling::Current);
        copy_level_data(output.energy, m_step.residual_energy);
        copy_level_data(output.temperature, m_step.residual_temperature);
    }

    void linearized_residual (State const& state, State& output)
    {
        residual(state, output);
    }

    State makeVecRHS () const
    {
        return {make_cell_data(m_step.hierarchy, 1, 0),
                make_cell_data(m_step.hierarchy, 1, 0)};
    }
    State makeVecLHS () const
    {
        return {make_cell_data(m_step.hierarchy, 1, 1),
                make_cell_data(m_step.hierarchy, 1, 1)};
    }

    void precondition (State& output, State const& rhs)
    {
        apply_schur(output, rhs, false);
    }

    // The same Schur-reduced linear system, with the E solve converged
    // (GMRES + AMG) instead of one AMG cycle: a frozen-coefficient Newton
    // step, used by the predictor iterations.
    void solve_correction (State& output, State const& rhs)
    {
        apply_schur(output, rhs, true);
    }

  private:
    void apply_schur (State& output, State const& rhs, bool converged)
    {
        auto energy_rhs = clone_level_data(rhs.energy);
        for (MFIter mfi(*energy_rhs[0]); mfi.isValid(); ++mfi) {
            auto const re = energy_rhs[0]->array(mfi);
            auto const rt = rhs.temperature[0]->const_array(mfi);
            auto const kk = m_step.schur_k[0]->const_array(mfi);
            auto const td = m_step.schur_diagonal[0]->const_array(mfi);
            ParallelFor(mfi.validbox(),
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                re(i, j, k) += kk(i, j, k) * rt(i, j, k) / td(i, j, k);
            });
        }
        set_level_data(output.energy, Real(0));
        if (converged) {
            auto const info = m_step.solver.solve(
                get_level_ptrs(output.energy),
                get_level_const_ptrs(energy_rhs), linear_tolerance(),
                Real(0));
            record_solve(m_step.summary, info);
        } else {
            m_step.solver.precondition(get_level_ptrs(output.energy),
                                       get_level_const_ptrs(energy_rhs));
        }
        for (MFIter mfi(*output.temperature[0]); mfi.isValid(); ++mfi) {
            auto const dt_out = output.temperature[0]->array(mfi);
            auto const de = output.energy[0]->const_array(mfi);
            auto const rt = rhs.temperature[0]->const_array(mfi);
            auto const ss = m_step.schur_sigma[0]->const_array(mfi);
            auto const td = m_step.schur_diagonal[0]->const_array(mfi);
            ParallelFor(mfi.validbox(),
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                dt_out(i, j, k) =
                    (rt(i, j, k) + ss(i, j, k) * de(i, j, k)) / td(i, j, k);
            });
        }
    }

  public:
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
                                      m_step.hierarchy, m_step.masks) /
                   (m_energy_scale * m_energy_scale) +
               composite_weighted_dot(lhs.temperature, rhs.temperature,
                                      m_step.hierarchy, m_step.masks) /
                   (m_temperature_scale * m_temperature_scale);
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
        return {composite_weighted_dot(x.energy, x.energy, m_step.hierarchy,
                                       m_step.masks),
                composite_weighted_dot(x.temperature, x.temperature,
                                       m_step.hierarchy, m_step.masks)};
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
        x.energy[0]->mult(a, 0, 1, 0);
        x.temperature[0]->mult(a, 0, 1, 0);
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
        mm_positive_update(candidate.energy, state.energy, correction.energy,
                           step_length);
        mm_positive_update(candidate.temperature, state.temperature,
                           correction.temperature, step_length);
    }
    bool admissible (State const& state) const
    {
        return mm_admissible(state.energy, m_step.masks) &&
               mm_admissible(state.temperature, m_step.masks);
    }
    Real relative_change (State const& lhs, State const& rhs) const
    {
        return amrex::max(
            composite_maximum_relative_change(lhs.energy, rhs.energy,
                                              m_step.masks),
            composite_maximum_relative_change(lhs.temperature,
                                              rhs.temperature, m_step.masks));
    }

  private:
    MovingStep& m_step;
    Real m_energy_scale = Real(1);
    Real m_temperature_scale = Real(1);
};

// Explicit material advection T* = T - dt div(v T), upwind or centered.
void
mm_advect_material (MovingStep& s, LevelData& temperature)
{
    MovingPhysics const p = s.physics;
    // The explicit update is stable only for dt sum_d |v_d|/dx_d <= 1.
    Real courant = Real(0);
    for (int d = 0; d < 3; ++d) {
        courant += p.dt * std::abs(p.velocity[d]) / p.dx[d];
    }
    if (courant > Real(1)) {
        amrex::Abort("Moving-medium explicit material advection needs "
                     "dt * sum |v_d|/dx_d <= 1; it is " +
                     std::to_string(courant) + ". Reduce vc.dt.");
    }
    auto work = make_cell_data(s.hierarchy, 1, 1);
    copy_level_data(work, temperature);
    work[0]->FillBoundary(mm_periodicity(s));
    for (MFIter mfi(*temperature[0]); mfi.isValid(); ++mfi) {
        auto const t = work[0]->const_array(mfi);
        auto const out = temperature[0]->array(mfi);
        ParallelFor(mfi.validbox(),
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            IntVect const c(AMREX_D_DECL(i, j, k));
            Real divergence = Real(0);
            for (int d = 0; d < 3; ++d) {
                IntVect const sft = IntVect::TheDimensionVector(d);
                Real const v = p.velocity[d];
                Real const f_lo = v * mm_face_advection(p, d, t(c - sft), t(c));
                Real const f_hi = v * mm_face_advection(p, d, t(c), t(c + sft));
                divergence += (f_hi - f_lo) / p.dx[d];
            }
            out(c) = t(c) - p.dt * divergence;
        });
    }
}

// Total radiation + material energy, and the energy removed by work on the
// flow, sigma v.F0, integrated over the domain at the loaded state.
std::pair<Real, Real>
mm_energy_and_work (MovingStep& s, LevelData const& energy,
                    LevelData const& temperature)
{
    mm_load_state(s, energy, temperature);
    MovingPhysics const p = s.physics;
    Real const volume = p.dx[0] * p.dx[1] * p.dx[2];
    ReduceOps<ReduceOpSum, ReduceOpSum> reduce_op;
    ReduceData<Real, Real> reduce_data(reduce_op);
    using Tuple = typename decltype(reduce_data)::Type;
    for (MFIter mfi(*s.energy[0]); mfi.isValid(); ++mfi) {
        auto const e = s.energy[0]->const_array(mfi);
        auto const t = s.temperature[0]->const_array(mfi);
        reduce_op.eval(mfi.validbox(), reduce_data,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept -> Tuple
        {
            auto const flux = mm_cell_flux(p, e, t, i, j, k, t(i, j, k));
            Real const work = mm_opacity(p, t(i, j, k)) *
                              mm_dot(p.velocity, flux);
            return {(e(i, j, k) + t(i, j, k)) * volume, work * volume};
        });
    }
    auto const sums = reduce_data.value(reduce_op);
    Real result[2] = {amrex::get<0>(sums), amrex::get<1>(sums)};
    ParallelDescriptor::ReduceRealSum(result, 2);
    return {result[0], result[1]};
}

struct MovingCounters
{
    int nonlinear = 0;
    int maximum_nonlinear = 0;
    int krylov = 0;
    int lag_iterations = 0;
    int maximum_lag_iterations = 0;
    int predictor_fallbacks = 0;
    Real maximum_coupled_residual = Real(0);
};

// Damped (0.7) frozen-coefficient Newton steps on the full residual, each
// with a converged Schur-reduced linear solve, until the relative change
// falls below 1e-3 (at most 20).  The analog of the ICASE predictor
// iterations; used to restart a step whose Newton iteration failed.
void
mm_predictors (MovingStep& s, MovingState& state)
{
    BL_PROFILE("FLD::vc_predictors");
    for (int predictor = 0; predictor < 20; ++predictor) {
        CoupledProblem problem(s, state);
        problem.prepare(state);
        auto residual = problem.makeVecRHS();
        problem.residual(state, residual);
        problem.scale(residual, Real(-1));
        auto correction = problem.makeVecLHS();
        problem.solve_correction(correction, residual);
        auto candidate = problem.clone(state);
        problem.fill_candidate(candidate, state, correction, Real(0.7));
        Real const change = problem.relative_change(candidate, state);
        problem.assign(state, candidate);
        if (change < Real(1.e-3)) { break; }
    }
}

// One attempt at the implicit step from `state`.  With allow_failure, a
// Newton (or lag) failure returns false instead of aborting.
bool
mm_attempt_step (MovingStep& s, std::string const& solver, Real tolerance,
                 int maximum_lag_iterations, MovingState& state,
                 MovingCounters& counters, int& step_nonlinear,
                 bool allow_failure)
{
    NewtonKrylovOptions options;
    options.nonlinear_tolerance = tolerance;
    options.maximum_nonlinear_iterations = 50;
    options.problem_name = "The moving-medium system";
    options.abort_on_failure = !allow_failure;
    if (solver == "newton_krylov") {
        CoupledProblem problem(s, state);
        NewtonKrylovSolver<CoupledProblem> newton(problem, options);
        auto const result = newton.solve(state);
        step_nonlinear += result.nonlinear_iterations;
        counters.krylov += result.total_linear_iterations;
        if (!result.converged && allow_failure) { return false; }
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            result.converged, "Moving-medium coupled Newton did not converge");
        return true;
    }
    // Lagged work: freeze the work flux, converge the reduced system to a
    // tolerance tied to the unlagged residual, and repeat until the unlagged
    // residual converges.  (The change in E between lags is not a stopping
    // criterion: an inner solve may still move E after the unlagged
    // residual has converged.)
    copy_level_data(s.eliminated_temperature, state.temperature);
    mm_load_state(s, state.energy, state.temperature);
    mm_freeze_work_flux(s);
    int lag = 0;
    for (;; ++lag) {
        mm_eliminate(s, state.energy, false);
        copy_level_data(state.temperature, s.eliminated_temperature);
        Real const unlagged = mm_coupled_residual_norm(
            s, state.energy, state.temperature);
        if (lag > 0 && unlagged <= tolerance) { break; }
        if (lag >= maximum_lag_iterations && allow_failure) { return false; }
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            lag < maximum_lag_iterations,
            "Moving-medium lagged work iteration did not converge");
        mm_freeze_work_flux(s);
        NewtonKrylovOptions inner = options;
        // Tighter than the outer target, or a converged inner residual can
        // stall the lag just above it.
        inner.nonlinear_tolerance =
            Real(0.1) * amrex::max(tolerance, unlagged);
        ReducedProblem problem(s, state.energy);
        NewtonKrylovSolver<ReducedProblem> newton(problem, inner);
        auto const result = newton.solve(state.energy);
        step_nonlinear += result.nonlinear_iterations;
        counters.krylov += result.total_linear_iterations;
        if (!result.converged && allow_failure) { return false; }
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            result.converged,
            "Moving-medium lagged inner Newton did not converge");
    }
    counters.lag_iterations += lag;
    counters.maximum_lag_iterations =
        amrex::max(counters.maximum_lag_iterations, lag);
    return true;
}

// Advance one implicit step from (E, T) (T already advected).  Newton starts
// from the previous step's state; if it fails, the step is retried from the
// predictor iterations' state.
void
mm_implicit_step (MovingStep& s, std::string const& solver,
                  Real tolerance, int maximum_lag_iterations,
                  MovingState& state, MovingCounters& counters)
{
    MovingState const start{clone_level_data(state.energy),
                            clone_level_data(state.temperature)};
    int step_nonlinear = 0;
    for (int attempt = 0; attempt < 2; ++attempt) {
        bool const fallback = attempt == 1;
        if (fallback) {
            copy_level_data(state.energy, start.energy);
            copy_level_data(state.temperature, start.temperature);
            ++counters.predictor_fallbacks;
            mm_predictors(s, state);
        }
        if (mm_attempt_step(s, solver, tolerance, maximum_lag_iterations,
                            state, counters, step_nonlinear, !fallback)) {
            break;
        }
    }
    counters.nonlinear += step_nonlinear;
    counters.maximum_nonlinear =
        amrex::max(counters.maximum_nonlinear, step_nonlinear);
    counters.maximum_coupled_residual = amrex::max(
        counters.maximum_coupled_residual,
        mm_coupled_residual_norm(s, state.energy, state.temperature));
}

MovingPhysics
mm_read_physics (ParmParse& pp, DiffusionHierarchy const& hierarchy, Real dt)
{
    MovingPhysics p;
    pp.query("sigma0", p.sigma0);
    pp.query("opacity_power", p.opacity_power);
    Vector<Real> velocity{Real(1.e-3), Real(0), Real(0)};
    Vector<Real> requested;
    if (pp.queryarr("velocity", requested)) { velocity = requested; }
    AMREX_ALWAYS_ASSERT(velocity.size() == 3);
    for (int d = 0; d < 3; ++d) {
        p.velocity[d] = velocity[d];
        p.dx[d] = hierarchy.geom[0].CellSize(d);
    }
    int limiter = 1;
    pp.query("limiter", limiter);
    p.limiter = limiter != 0;
    std::string advection = "upwind";
    pp.query("advection", advection);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        advection == "upwind" || advection == "centered",
        "vc.advection must be upwind or centered");
    p.upwind = advection == "upwind";
    p.dt = dt;
    return p;
}

void
mm_print_counters (std::string const& solver, MovingStep const& s,
                   MovingCounters const& c, double wall)
{
    amrex::Print()
        << "solver=" << solver << ", Newton iterations total/max="
        << c.nonlinear << "/" << c.maximum_nonlinear
        << ", Krylov iterations=" << c.krylov
        << ", lag iterations total/max=" << c.lag_iterations << "/"
        << c.maximum_lag_iterations
        << ", predictor fallbacks=" << c.predictor_fallbacks
        << ", eliminations=" << s.eliminations
        << ", max local Newton iterations=" << s.maximum_local_iterations
        << ", AMG setups=" << s.summary.setups
        << ", max coupled residual=" << c.maximum_coupled_residual
        << ", wall=" << wall << " s\n";
}

// ---------------------------------------------------------------------
// Advecting pulse.

void
run_pulse (ParmParse& pp, std::string const& solver, Real tolerance,
           int maximum_lag_iterations)
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
    auto const masks = make_composite_masks(hierarchy);
    MovingPhysics const physics = mm_read_physics(pp, hierarchy, dt);
    SolverSummary summary;
    MLABecLapAMG amg(hierarchy.geom, hierarchy.grids, hierarchy.dmap);
    MovingStep s(hierarchy, masks, physics, amg, summary);

    // Equilibrium temperature pulse at the domain center.
    MovingState state{make_cell_data(hierarchy, 1, 1),
                      make_cell_data(hierarchy, 1, 1)};
    for (MFIter mfi(*state.energy[0]); mfi.isValid(); ++mfi) {
        auto const e = state.energy[0]->array(mfi);
        auto const t = state.temperature[0]->array(mfi);
        auto const dx = physics.dx;
        ParallelFor(mfi.validbox(),
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            Real const x = (Real(i) + Real(0.5)) * dx[0] - Real(0.5);
            Real const y = (Real(j) + Real(0.5)) * dx[1] - Real(0.5);
            Real const z = (Real(k) + Real(0.5)) * dx[2] - Real(0.5);
            Real const temperature = background +
                amplitude * std::exp(-(x * x + y * y + z * z) / (width * width));
            t(i, j, k) = temperature;
            e(i, j, k) = mm_fourth(temperature);
        });
    }

    // Circular mean of the x position of the excess radiation energy.
    auto centroid = [&] () {
        ReduceOps<ReduceOpSum, ReduceOpSum> reduce_op;
        ReduceData<Real, Real> reduce_data(reduce_op);
        using Tuple = typename decltype(reduce_data)::Type;
        Real const e_background = mm_fourth(background);
        for (MFIter mfi(*state.energy[0]); mfi.isValid(); ++mfi) {
            auto const e = state.energy[0]->const_array(mfi);
            auto const dx = physics.dx;
            reduce_op.eval(mfi.validbox(), reduce_data,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept -> Tuple
            {
                Real const w = amrex::max(e(i, j, k) - e_background, Real(0));
                Real const angle = Real(2) * Real(3.14159265358979323846) *
                                   (Real(i) + Real(0.5)) * dx[0];
                return {w * std::cos(angle), w * std::sin(angle)};
            });
        }
        auto const sums = reduce_data.value(reduce_op);
        Real r[2] = {amrex::get<0>(sums), amrex::get<1>(sums)};
        ParallelDescriptor::ReduceRealSum(r, 2);
        Real angle = std::atan2(r[1], r[0]);
        if (angle < Real(0)) { angle += Real(2) * Real(3.14159265358979323846); }
        return angle / (Real(2) * Real(3.14159265358979323846));
    };

    MovingCounters counters;
    Real const x0 = centroid();
    Real maximum_balance_error = Real(0);
    double const start = amrex::second();
    for (int step = 0; step < steps; ++step) {
        auto const [old_total, old_work] =
            mm_energy_and_work(s, state.energy, state.temperature);
        amrex::ignore_unused(old_work);
        copy_level_data(s.old_energy, state.energy);
        mm_advect_material(s, state.temperature);
        copy_level_data(s.old_temperature, state.temperature);
        mm_implicit_step(s, solver, tolerance, maximum_lag_iterations, state,
                         counters);
        auto const [new_total, new_work] =
            mm_energy_and_work(s, state.energy, state.temperature);
        // Material advection conserves sum(T) on the periodic domain, so
        // the total changes only by the work done on the flow.
        Real const expected = -dt * new_work;
        Real const change = new_total - old_total;
        Real const error = std::abs(change - expected) /
            amrex::max(std::abs(old_total), Real(1.e-300));
        maximum_balance_error = amrex::max(maximum_balance_error, error);
    }
    double wall = amrex::second() - start;
    ParallelDescriptor::ReduceRealMax(wall);

    Real const time = Real(steps) * dt;
    Real displacement = centroid() - x0;
    displacement -= std::round(displacement);
    Real const advected = physics.velocity[0] * time;
    amrex::Print() << std::setprecision(10)
                   << "Moving-medium pulse: n=" << n_cell << ", steps="
                   << steps << ", dt=" << dt << ", sigma0=" << physics.sigma0
                   << ", v=(" << physics.velocity[0] << ","
                   << physics.velocity[1] << "," << physics.velocity[2]
                   << "), v*tau=" << physics.velocity[0] * physics.sigma0
                   << ", centroid shift=" << displacement
                   << ", v*t=" << advected
                   << ", max energy-balance error=" << maximum_balance_error
                   << '\n';
    mm_print_counters(solver, s, counters, wall);
    if (!plotfile.empty()) {
        auto plot = make_cell_data(hierarchy, 2, 0);
        MultiFab::Copy(*plot[0], *state.energy[0], 0, 0, 1, 0);
        MultiFab::Copy(*plot[0], *state.temperature[0], 0, 1, 1, 0);
        WriteMultiLevelPlotfile(plotfile, 1, get_level_const_ptrs(plot),
                                {"radiation_energy", "material_temperature"},
                                hierarchy.geom, time, {0}, {});
    }
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
mm_mms (MovingPhysics const& p, Real x, Real y, Real z) noexcept
{
    Real const w = Real(2) * Real(3.14159265358979323846);
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
    Real const sigma = p.sigma0 * std::pow(t, -p.opacity_power);
    Real const d = Real(1) / (Real(3) * sigma);
    // grad D = p T^(p-1) grad T / (3 sigma0)
    Real const dd = p.opacity_power * std::pow(t, p.opacity_power - Real(1)) /
                    (Real(3) * p.sigma0);
    Real grad_d_dot_grad_e = Real(0);
    Real v_dot_ge = Real(0);
    Real v_dot_gt = Real(0);
    Real v_dot_f0 = Real(0);
    for (int i = 0; i < 3; ++i) {
        grad_d_dot_grad_e += dd * gt[i] * ge[i];
        v_dot_ge += p.velocity[i] * ge[i];
        v_dot_gt += p.velocity[i] * gt[i];
        v_dot_f0 += p.velocity[i] * (-d * ge[i]);
    }
    Real const div_f0 = -(grad_d_dot_grad_e + d * lap_e);
    Real const t4 = mm_fourth(t);
    Real const se = div_f0 + Real(4) / Real(3) * v_dot_ge -
                    sigma * (t4 - e) - sigma * v_dot_f0;
    Real const st = v_dot_gt - sigma * (e - t4) + Real(2) * sigma * v_dot_f0;
    return {e, t, se, st};
}

void
run_mms (ParmParse& pp, std::string const& solver, Real tolerance,
         int maximum_lag_iterations)
{
    Vector<int> n_cells{16, 32, 64};
    Vector<int> requested;
    if (pp.queryarr("mms_n_cells", requested)) { n_cells = requested; }
    Real dt = Real(0.1);
    pp.query("mms_dt", dt);
    std::string advection = "upwind";
    pp.query("advection", advection);

    Vector<Array<Real, 4>> errors; // E max, E L2, T max, T L2
    for (int n : n_cells) {
        auto const hierarchy = make_uniform_hierarchy(
            n, amrex::min(n, 32), {AMREX_D_DECL(1, 1, 1)});
        auto const masks = make_composite_masks(hierarchy);
        MovingPhysics const physics = mm_read_physics(pp, hierarchy, dt);
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            !physics.limiter, "The moving-medium MMS requires vc.limiter=0");
        SolverSummary summary;
        MLABecLapAMG amg(hierarchy.geom, hierarchy.grids, hierarchy.dmap);
        MovingStep s(hierarchy, masks, physics, amg, summary);
        MovingState state{make_cell_data(hierarchy, 1, 1),
                          make_cell_data(hierarchy, 1, 1)};
        auto exact_e = make_cell_data(hierarchy, 1, 0);
        auto exact_t = make_cell_data(hierarchy, 1, 0);
        for (MFIter mfi(*state.energy[0]); mfi.isValid(); ++mfi) {
            auto const ee = exact_e[0]->array(mfi);
            auto const et = exact_t[0]->array(mfi);
            auto const se = s.source_energy[0]->array(mfi);
            auto const st = s.source_temperature[0]->array(mfi);
            auto const dx = physics.dx;
            ParallelFor(mfi.validbox(),
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                auto const m = mm_mms(physics, (Real(i) + Real(0.5)) * dx[0],
                                      (Real(j) + Real(0.5)) * dx[1],
                                      (Real(k) + Real(0.5)) * dx[2]);
                ee(i, j, k) = m.energy;
                et(i, j, k) = m.temperature;
                se(i, j, k) = m.source_energy;
                st(i, j, k) = m.source_temperature;
            });
        }
        copy_level_data(state.energy, exact_e);
        copy_level_data(state.temperature, exact_t);
        copy_level_data(s.old_energy, exact_e);
        mm_advect_material(s, state.temperature);
        copy_level_data(s.old_temperature, state.temperature);
        MovingCounters counters;
        double const start = amrex::second();
        mm_implicit_step(s, solver, tolerance, maximum_lag_iterations, state,
                         counters);
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
        mm_print_counters(solver, s, counters, wall);
    }
    amrex::Print() << "\nMoving-medium MMS, solver=" << solver
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
    std::string solver = "newton_krylov";
    std::string test = "pulse";
    Real tolerance = Real(1.e-8);
    int maximum_lag_iterations = 50;
    pp.query("solver", solver);
    pp.query("test", test);
    pp.query("nonlinear_tolerance", tolerance);
    pp.query("max_lag_iterations", maximum_lag_iterations);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        solver == "newton_krylov" || solver == "ne_lagged",
        "vc.solver must be newton_krylov or ne_lagged");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(test == "pulse" || test == "mms",
                                     "vc.test must be pulse or mms");
    if (test == "pulse") {
        run_pulse(pp, solver, tolerance, maximum_lag_iterations);
    } else {
        run_mms(pp, solver, tolerance, maximum_lag_iterations);
    }
}

} // namespace fld_test
