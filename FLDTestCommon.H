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
    auto const& diagnostics = solver.diagnostics();
    summary.setup_seconds += diagnostics.setup_seconds;
    summary.maximum_levels = amrex::max(
        summary.maximum_levels, static_cast<int>(diagnostics.levels.size()));
    summary.maximum_operator_complexity = amrex::max(
        summary.maximum_operator_complexity, diagnostics.operator_complexity);
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
}

} // namespace
} // namespace fld_test

#endif
