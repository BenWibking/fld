#ifndef RADIATION_MATERIAL_H_
#define RADIATION_MATERIAL_H_

#include "DiffusionHierarchy.H"
#include "FLDFieldOps.H"
#include "SolverSummary.H"

#include <AMReX_Array.H>
#include <AMReX_REAL.H>

#include <limits>
#include <memory>
#include <string>

namespace fld_test
{

/** Radiation boundary condition on one face of the domain.  The material
 *  energy has no flux.  Robin: a E + b dE/dn = incoming, with b = 1/(6 sigma)
 *  (Marshak) or b = D; a = 1/4 with b = 1/(6 sigma) is the Marshak
 *  condition. */
enum class RadiationBoundary
{
    Periodic,
    Neumann,
    Robin
};

/** The FLD diffusion coefficient on cell faces. */
enum class DiffusionCoefficient
{
    /** Each side's 1/(3 sigma + |grad E|/E), with the gradient across the
     *  face and sigma at the mean face temperature; harmonic mean. */
    FaceLimiter,
    /** Cell lambda(R)/sigma with the Levermore-Pomraning limiter and
     *  R = |grad E|/(sigma E) from central differences; harmonic mean. */
    CellLimiter,
    /** Cell lambda(R) as for CellLimiter; on interior faces
     *  max(lambda_L, lambda_R) over a face opacity bounded between the
     *  harmonic and arithmetic means and floored at 4/(3 h). */
    CellLimiterSurfaceOpacity
};

/** The b coefficient of Robin boundaries. */
enum class RobinCoefficient
{
    Marshak,  // 1/(6 sigma)
    Diffusion // the cell's D
};

struct RadiationMaterialOptions
{
    /** sigma = kappa T^-opacity_power, with kappa a cell field. */
    amrex::Real opacity_power = amrex::Real(3);
    /** Uniform material velocity in units of c; nonzero needs a periodic
     *  domain and one level. */
    amrex::Array<amrex::Real, AMREX_SPACEDIM> velocity{};
    /** Without the limiter, D = 1/(3 sigma). */
    bool limiter = true;
    DiffusionCoefficient diffusion = DiffusionCoefficient::FaceLimiter;
    /** Upwind (true) or centered (false) advection. */
    bool upwind = true;
    /** Indexed 2*direction + side (side 0 = low). */
    amrex::Array<RadiationBoundary, 2 * AMREX_SPACEDIM> boundary{};
    RobinCoefficient robin_b = RobinCoefficient::Marshak;
    /** Robin a and right-hand side, indexed like `boundary`. */
    amrex::Array<amrex::Real, 2 * AMREX_SPACEDIM> robin_a{
        amrex::Real(0.25), amrex::Real(0.25), amrex::Real(0.25),
        amrex::Real(0.25), amrex::Real(0.25), amrex::Real(0.25)};
    amrex::Array<amrex::Real, 2 * AMREX_SPACEDIM> incoming{};
    /** newton_krylov, ne_newton_krylov, or ne_quasi_newton. */
    std::string solver = "newton_krylov";
    amrex::Real nonlinear_tolerance = amrex::Real(1.e-8);
    int maximum_nonlinear_iterations = 50;
    int maximum_lag_iterations = 50;
    int maximum_quasi_newton_iterations = 200;
    /** Newton candidates x exp(s dx/x), which keep E and T positive
     *  across many decades; additive x + s dx otherwise (with positivity
     *  still required). */
    bool positive_update = true;
    /** Run the predictor iterations before every step's first Newton
     *  attempt, not only after a failure: for steps that start far from
     *  their solution (e.g. one long step to a steady state). */
    bool predict_first = false;
    /** Newton-Krylov GMRES and line-search limits. */
    int maximum_linear_iterations = 100;
    int krylov_restart_length = 30;
    int maximum_line_search_iterations = 14;
    /** Centered (true) or forward finite-difference Jacobian products. */
    bool centered_difference = true;
    /** Newton line-search candidates with E at or above this are rejected;
     *  a bound on E (e.g. from the boundary data) keeps the eliminated
     *  temperature from overshooting on hard steps. */
    amrex::Real maximum_energy = std::numeric_limits<amrex::Real>::max();
    /** Lower bounds of the energy and temperature norm scales. */
    amrex::Real energy_scale_floor = amrex::Real(1.e-30);
    amrex::Real temperature_scale_floor = amrex::Real(1.e-30);
};

struct RadiationMaterialState
{
    LevelData energy;
    LevelData temperature;
};

struct RadiationMaterialCounters
{
    int nonlinear = 0;            // Newton or quasi-Newton iterations
    int maximum_nonlinear = 0;    // per step
    int krylov = 0;               // Newton GMRES iterations
    int maximum_krylov = 0;       // per Newton solve
    int lag_iterations = 0;
    int maximum_lag_iterations = 0;
    int predictor_fallbacks = 0;
    amrex::Long eliminations = 0; // cell-local solve passes
    int maximum_local_iterations = 0;
    amrex::Real final_change = amrex::Real(0);
    amrex::Real maximum_coupled_residual = amrex::Real(0);
    amrex::Real final_coupled_residual = amrex::Real(0);
    SolverSummary summary;        // AMG setups and converged linear solves
};

/** Domain integrals at a state, for energy-balance checks: the change of
 *  `total` over a step of length dt is
 *  dt * (boundary_input - work + source) at the end of the step. */
struct EnergyBudget
{
    amrex::Real total = amrex::Real(0);          // integral of E + T
    amrex::Real boundary_input = amrex::Real(0); // Marshak inflow
    amrex::Real work = amrex::Real(0);           // sigma v.F0 on the flow
    amrex::Real source = amrex::Real(0);         // manufactured sources
};

/**
 * Grey flux-limited diffusion coupled to a material energy equation, with
 * optional uniform velocity to O(v/c), on a composite AMR hierarchy (one
 * level with a velocity).  In units with
 * a_r = c = rho c_v = 1 and v in units of c:
 *
 *   dE/dt + div[F0 + v.(E I + P)] = sigma (T^4 - E) + sigma v.F0 + S_E
 *   dT/dt + div(v T)               = sigma (E - T^4) - 2 sigma v.F0 + S_T
 *
 * with F0 = -D grad E the comoving FLD flux, P the Levermore Eddington
 * tensor of the limiter, and the mixed-frame four-force of He, Wibking &
 * Krumholz (2024).  Material advection is explicit and operator split; the
 * rest is backward Euler.  If Newton fails, a step is retried from damped
 * linearized (Picard) predictor iterations.
 */
class RadiationMaterialSolver
{
  public:
    RadiationMaterialSolver (DiffusionHierarchy const& hierarchy,
                             RadiationMaterialOptions options);
    ~RadiationMaterialSolver ();

    RadiationMaterialSolver (RadiationMaterialSolver const&) = delete;
    RadiationMaterialSolver& operator= (RadiationMaterialSolver const&) =
        delete;

    [[nodiscard]] RadiationMaterialState make_state () const;

    /** kappa in sigma = kappa T^-p (copied). */
    void set_opacity_coefficient (LevelData const& kappa);

    /** Manufactured sources S_E and S_T (copied); zero by default. */
    void set_sources (LevelData const& energy_source,
                      LevelData const& temperature_source);

    /** Advance by dt; aborts if the step cannot be converged. */
    void advance (RadiationMaterialState& state, amrex::Real dt);

    [[nodiscard]] EnergyBudget energy_budget (
        RadiationMaterialState const& state);

    /** Radiation flux out of the domain through each face (indexed like
     *  `boundary`), integrated over the face. */
    [[nodiscard]] amrex::Array<amrex::Real, 2 * AMREX_SPACEDIM>
    boundary_outflow (RadiationMaterialState const& state);

    /** Cell opacity and cell-centered FLD diffusion coefficient, for
     *  plotfiles. */
    void fill_cell_coefficients (RadiationMaterialState const& state,
                                 LevelData& sigma, LevelData& diffusion);

    /** Cell-centered comoving flux F0 (the mean of the face fluxes in each
     *  direction; AMREX_SPACEDIM components), for plotfiles. */
    void fill_cell_flux (RadiationMaterialState const& state,
                         LevelData& flux);

    [[nodiscard]] RadiationMaterialCounters const& counters () const;
    [[nodiscard]] amrex::Vector<amrex::iMultiFab> const& masks () const;

    /** Opaque implementation; public so its helper classes can name it. */
    struct Impl;

  private:
    std::unique_ptr<Impl> m_impl;
};

} // namespace fld_test

#endif
