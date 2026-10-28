#ifndef FLD_TEST_H_
#define FLD_TEST_H_

#include "SolverSummary.H"

#include <AMReX_REAL.H>
#include <AMReX_Vector.H>

#include <string>

namespace fld_test
{

struct CloudResult
{
    amrex::Real transmission = amrex::Real(0);
    amrex::Real balance_error = amrex::Real(0);
    amrex::Real cloudy_volume_relative_error = amrex::Real(0);
    amrex::Real minimum_energy = amrex::Real(0);
    amrex::Real maximum_energy = amrex::Real(0);
    amrex::Real final_nonlinear_change = amrex::Real(0);
    amrex::Real final_nonlinear_residual = amrex::Real(0);
    int time_steps = 0;
    int nonlinear_iterations = 0;
    int total_newton_krylov_iterations = 0;
    int maximum_newton_krylov_iterations = 0;
    int predictor_fallbacks = 0;
    amrex::Long mixed_cells = 0;
    amrex::Long cells = 0;
    SolverSummary solver;
};

struct ICASE2001Result
{
    amrex::Real final_time = amrex::Real(0);
    amrex::Real initial_total_energy = amrex::Real(0);
    amrex::Real final_total_energy = amrex::Real(0);
    amrex::Real minimum_radiation_energy = amrex::Real(0);
    amrex::Real maximum_radiation_energy = amrex::Real(0);
    amrex::Real minimum_material_temperature = amrex::Real(0);
    amrex::Real maximum_material_temperature = amrex::Real(0);
    amrex::Real final_nonlinear_change = amrex::Real(0);
    amrex::Real maximum_coupled_residual = amrex::Real(0);
    amrex::Real maximum_energy_balance_error = amrex::Real(0);
    int time_steps = 0;
    int total_nonlinear_iterations = 0;
    int maximum_nonlinear_iterations = 0;
    int total_newton_krylov_iterations = 0;
    int maximum_newton_krylov_iterations = 0;
    // Newton-Krylov solvers: steps retried from the other starting state
    // (with or without predictor iterations).
    int predictor_fallbacks = 0;
    // Nonlinear elimination only: cell-local temperature solves.
    int maximum_local_newton_iterations = 0;
    amrex::Long total_local_solves = 0;
    std::string nonlinear_solver;
    amrex::Long cells = 0;
    amrex::Long high_z_cells = 0;
    SolverSummary solver;
};

CloudResult run_cloud (bool use_amr, int fine_n, bool limited,
                       bool iteration_output,
                       std::string const& plotfile_name,
                       amrex::Real opacity_contrast = amrex::Real(1.e6));

ICASE2001Result run_icase_2001 (int n_cell, int time_steps,
                                amrex::Real dt, bool iteration_output,
                                std::string const& plotfile_name);

/** Manufactured-solution convergence study of the composite operator on a
 *  two-level hierarchy; resolutions from `mms.n_cells`. */
void run_mms_convergence ();

/** Grey FLD and material energy in a medium with a prescribed uniform
 *  velocity, to O(v/c); parameters under `vc.` (see MovingMedium.cpp). */
void run_moving_medium ();

// Operator-action, boundary, and hierarchy checks of MLABecLapAMG
// (MLABecAMGTest.cpp); aborts on failure.
void run_mlabeclap_amg_checks ();

} // namespace fld_test

#endif
