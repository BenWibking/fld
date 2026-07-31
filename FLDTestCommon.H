#ifndef FLD_TEST_COMMON_H_
#define FLD_TEST_COMMON_H_

#include "AMGGMRESSolver.H"
#include "FLDLinearSystem.H"
#include "FLDMesh.H"
#include "FLDTest.H"

#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_PlotFileUtil.H>

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <numeric>
#include <string>
#include <utility>

namespace fld_test
{
namespace
{

using namespace amrex;

struct CloudMesh
{
    Mesh mesh;
    Vector<int> level_n;
    Vector<std::pair<int, int>> level_y_bounds;
};

void
record_setup (SolverSummary& summary, AMGGMRESSolver const& solver)
{
    auto const& diagnostics = solver.diagnostics();
    summary.setup_seconds += diagnostics.setup_seconds;
    summary.maximum_levels = amrex::max(
        summary.maximum_levels, static_cast<int>(diagnostics.levels.size()));
    summary.maximum_operator_complexity = amrex::max(
        summary.maximum_operator_complexity, diagnostics.operator_complexity);
}

void
record_solve (SolverSummary& summary, LinearSolution const& solution)
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
