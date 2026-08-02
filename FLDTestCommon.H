#ifndef FLD_TEST_COMMON_H_
#define FLD_TEST_COMMON_H_

#include "DiffusionHierarchy.H"
#include "FLDFieldOps.H"
#include "FLDTest.H"
#include "MLABecAMG.H"

#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_PlotFileUtil.H>

#include <algorithm>
#include <cmath>
#include <string>
#include <utility>

namespace fld_test
{
namespace
{

using namespace amrex;

Real
linear_tolerance ()
{
    return (sizeof(Real) == sizeof(float)) ? Real(5.e-5) : Real(2.e-10);
}

void
record_setup (SolverSummary& summary, MLABecLapAMG const& solver)
{
    summary.outer_solver = "gmres";
    if (summary.preconditioner.empty()) {
        summary.preconditioner = solver.preconditioner_name();
        summary.amg_backend = solver.amg_backend_name();
    }
    AMREX_ALWAYS_ASSERT(summary.preconditioner ==
                        solver.preconditioner_name());
    AMREX_ALWAYS_ASSERT(summary.amg_backend == solver.amg_backend_name());
    summary.setup_seconds += solver.setup_seconds();
    if (solver.preconditioner() == MLABecPreconditioner::AMG &&
        solver.amg_backend() == MLABecAMGBackend::Native) {
        auto const& diagnostics = solver.diagnostics();
        summary.maximum_levels = amrex::max(
            summary.maximum_levels,
            static_cast<int>(diagnostics.levels.size()));
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
    summary.total_preconditioner_applications +=
        solution.preconditioner_applications;
    summary.maximum_relative_residual = amrex::max(
        summary.maximum_relative_residual, solution.relative_residual);
    summary.solve_seconds += solution.solve_seconds;
    summary.preconditioner_seconds += solution.preconditioner_seconds;
}

} // namespace
} // namespace fld_test

#endif
