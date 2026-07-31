#ifndef FLD_TEST_H_
#define FLD_TEST_H_

#include <AMReX_REAL.H>
#include <AMReX_Vector.H>

#include <string>

namespace fld_test
{

struct SolverSummary
{
    int solves = 0;
    int total_iterations = 0;
    int maximum_iterations = 0;
    int maximum_levels = 0;
    amrex::Real maximum_relative_residual = amrex::Real(0);
    double setup_seconds = 0.0;
    double maximum_operator_complexity = 0.0;

    [[nodiscard]] amrex::Real
    average_iterations () const noexcept
    {
        return (solves > 0)
                   ? amrex::Real(total_iterations) / amrex::Real(solves)
                   : amrex::Real(0);
    }
};

struct GaussianResult
{
    amrex::Real relative_l1_error = amrex::Real(0);
    amrex::Real relative_energy_drift = amrex::Real(0);
    amrex::Long cells = 0;
    SolverSummary solver;
};

struct CloudResult
{
    amrex::Real transmission = amrex::Real(0);
    amrex::Real balance_error = amrex::Real(0);
    amrex::Real cloudy_area_relative_error = amrex::Real(0);
    amrex::Real minimum_energy = amrex::Real(0);
    amrex::Real maximum_energy = amrex::Real(0);
    amrex::Real final_nonlinear_change = amrex::Real(0);
    int nonlinear_iterations = 0;
    int anderson_steps = 0;
    int anderson_restarts = 0;
    amrex::Long mixed_cells = 0;
    amrex::Long cells = 0;
    SolverSummary solver;
};

struct FrontResult
{
    amrex::Real front_radius = amrex::Real(0);
    amrex::Real causal_radius = amrex::Real(0);
    amrex::Real far_excess = amrex::Real(0);
    amrex::Real maximum_flux_fraction = amrex::Real(0);
    amrex::Real minimum_energy = amrex::Real(0);
    amrex::Real maximum_energy = amrex::Real(0);
    amrex::Real unlimited_far_excess = amrex::Real(0);
    amrex::Real final_picard_change = amrex::Real(0);
    int total_picard_iterations = 0;
    int maximum_picard_iterations = 0;
    amrex::Long cells = 0;
    SolverSummary solver;
};

GaussianResult run_gaussian (bool use_amr);

CloudResult run_cloud (bool use_amr, int fine_n, int anderson_depth,
                       amrex::Real anderson_beta, bool limited,
                       bool iteration_output,
                       std::string const& plotfile_name);

FrontResult run_limited_front ();

} // namespace fld_test

#endif
