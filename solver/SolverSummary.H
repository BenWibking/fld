#ifndef SOLVER_SUMMARY_H_
#define SOLVER_SUMMARY_H_

#include <AMReX_REAL.H>

#include <string>

namespace fld_test
{

/** Linear-solver statistics accumulated over AMG setups and solves. */
struct SolverSummary
{
    std::string outer_solver;
    std::string preconditioner;
    std::string amg_backend;
    int setups = 0;
    int solves = 0;
    int total_iterations = 0;
    int maximum_iterations = 0;
    int total_preconditioner_applications = 0;
    int maximum_levels = 0;
    amrex::Real maximum_relative_residual = amrex::Real(0);
    double setup_seconds = 0.0;
    double solve_seconds = 0.0;
    double preconditioner_seconds = 0.0;
    double maximum_operator_complexity = 0.0;

    [[nodiscard]] amrex::Real
    average_iterations () const noexcept
    {
        return (solves > 0)
                   ? amrex::Real(total_iterations) / amrex::Real(solves)
                   : amrex::Real(0);
    }
};

} // namespace fld_test

#endif
