#ifndef NEWTON_KRYLOV_SOLVER_H_
#define NEWTON_KRYLOV_SOLVER_H_

#include <AMReX.H>
#include <AMReX_GMRES.H>

#include <cmath>
#include <limits>
#include <string>
#include <utility>

namespace fld_test
{

struct NewtonKrylovOptions
{
    amrex::Real nonlinear_tolerance = amrex::Real(1.e-7);
    amrex::Real linear_tolerance = amrex::Real(1.e-4);
    amrex::Real armijo_coefficient = amrex::Real(1.e-4);
    int maximum_nonlinear_iterations = 20;
    int maximum_linear_iterations = 100;
    int krylov_restart_length = 30;
    int maximum_line_search_iterations = 14;
    int linear_verbosity = 0;
    bool centered_difference = true;
    std::string problem_name = "nonlinear system";
};

struct NewtonKrylovResult
{
    bool converged = false;
    int nonlinear_iterations = 0;
    int total_linear_iterations = 0;
    int maximum_linear_iterations = 0;
    amrex::Real final_change = std::numeric_limits<amrex::Real>::max();
    amrex::Real final_residual_norm = std::numeric_limits<amrex::Real>::max();
};

/**
 * Jacobian-free Newton--Krylov solver with an Armijo backtracking line search.
 *
 * Problem supplies its state type, residual evaluation, preconditioner, vector
 * algebra, admissibility test, candidate construction, and change metric.
 * `linearized_residual` is separate from `residual` so finite-difference
 * probes cannot overwrite coefficients retained by the preconditioner.
 */
template <typename Problem> class NewtonKrylovSolver
{
  public:
    using State = typename Problem::State;
    using RT = typename Problem::RT;

    explicit NewtonKrylovSolver (Problem& problem,
                                 NewtonKrylovOptions options = {})
        : m_problem(problem), m_options(std::move(options))
    {
        AMREX_ALWAYS_ASSERT(m_options.nonlinear_tolerance > RT(0));
        AMREX_ALWAYS_ASSERT(m_options.linear_tolerance > RT(0));
        AMREX_ALWAYS_ASSERT(m_options.armijo_coefficient > RT(0));
        AMREX_ALWAYS_ASSERT(m_options.maximum_nonlinear_iterations > 0);
        AMREX_ALWAYS_ASSERT(m_options.maximum_linear_iterations > 0);
        AMREX_ALWAYS_ASSERT(m_options.krylov_restart_length > 0);
        AMREX_ALWAYS_ASSERT(m_options.maximum_line_search_iterations > 0);
    }

    [[nodiscard]] NewtonKrylovResult solve (State& state)
    {
        NewtonKrylovResult result;
        State residual = m_problem.makeVecRHS();
        State candidate_residual = m_problem.makeVecRHS();
        m_problem.residual(state, residual);
        result.final_residual_norm = m_problem.norm2(residual);
        if (result.final_residual_norm <= m_options.nonlinear_tolerance) {
            result.final_change = RT(0);
            result.converged = true;
            return result;
        }

        for (int iteration = 0;
             iteration < m_options.maximum_nonlinear_iterations; ++iteration) {
            m_problem.prepare(state);
            LinearizedOperator linearized(m_problem, state, residual,
                                          m_options.centered_difference);
            amrex::GMRES<State, LinearizedOperator> gmres;
            gmres.define(linearized);
            gmres.setRestartLength(m_options.krylov_restart_length);
            gmres.setMaxIters(m_options.maximum_linear_iterations);
            gmres.setVerbose(m_options.linear_verbosity);

            State correction = linearized.makeVecLHS();
            linearized.setToZero(correction);
            State linear_rhs = linearized.makeVecRHS();
            linearized.linComb(linear_rhs, RT(-1), residual, RT(0), residual);
            gmres.solve(correction, linear_rhs, m_options.linear_tolerance,
                        RT(0));
            std::string const linear_failure =
                m_options.problem_name +
                " Newton linear solve did not converge";
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(gmres.getStatus() == 0,
                                             linear_failure.c_str());
            result.total_linear_iterations += gmres.getNumIters();
            result.maximum_linear_iterations = amrex::max(
                result.maximum_linear_iterations, gmres.getNumIters());

            RT const residual_norm = m_problem.norm2(residual);
            bool accepted = false;
            State candidate = m_problem.clone(state);
            auto const try_line_search = [&] () {
                RT step_length = RT(1);
                for (int line_search = 0;
                     line_search < m_options.maximum_line_search_iterations;
                     ++line_search) {
                    m_problem.fill_candidate(candidate, state, correction,
                                             step_length);
                    if (m_problem.admissible(candidate)) {
                        m_problem.residual(candidate, candidate_residual);
                        RT const candidate_norm =
                            m_problem.norm2(candidate_residual);
                        if (candidate_norm <=
                            (RT(1) -
                             m_options.armijo_coefficient * step_length) *
                                residual_norm) {
                            result.final_residual_norm = candidate_norm;
                            return true;
                        }
                    }
                    step_length *= RT(0.5);
                }
                return false;
            };
            accepted = try_line_search();
            if (!accepted) {
                // A noisy Jacobian product or an inexact Krylov solve can
                // produce a non-descent Newton direction.  The frozen-
                // Jacobian preconditioner supplies a robust quasi-Newton
                // fallback while retaining the same globalization test.
                m_problem.precondition(correction, linear_rhs);
                accepted = try_line_search();
            }
            std::string const line_search_failure =
                m_options.problem_name + " Newton line search failed";
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(accepted,
                                             line_search_failure.c_str());

            result.final_change = m_problem.relative_change(candidate, state);
            m_problem.assign(state, candidate);
            m_problem.assign(residual, candidate_residual);
            ++result.nonlinear_iterations;
            if (result.final_change <= m_options.nonlinear_tolerance &&
                result.final_residual_norm <= m_options.nonlinear_tolerance) {
                result.converged = true;
                break;
            }
        }
        return result;
    }

  private:
    class LinearizedOperator
    {
      public:
        using RT = typename Problem::RT;

        LinearizedOperator (Problem& problem, State const& state,
                            State const& residual, bool centered_difference)
            : m_problem(problem), m_state(state), m_residual(residual),
              m_centered_difference(centered_difference)
        {
        }

        State makeVecRHS () const
        {
            return m_problem.makeVecRHS();
        }
        State makeVecLHS () const
        {
            return m_problem.makeVecLHS();
        }

        void apply (State& output, State const& direction)
        {
            RT const direction_norm = norm2(direction);
            if (direction_norm == RT(0)) {
                setToZero(output);
                return;
            }
            RT const epsilon =
                m_centered_difference
                    ? std::cbrt(std::numeric_limits<RT>::epsilon())
                    : std::sqrt(std::numeric_limits<RT>::epsilon());
            RT const step = epsilon * (RT(1) + norm2(m_state)) / direction_norm;
            State trial_plus = m_problem.clone(m_state);
            increment(trial_plus, direction, step);
            m_problem.linearized_residual(trial_plus, output);
            if (m_centered_difference) {
                State trial_minus = m_problem.clone(m_state);
                increment(trial_minus, direction, -step);
                State residual_minus = makeVecRHS();
                m_problem.linearized_residual(trial_minus, residual_minus);
                increment(output, residual_minus, RT(-1));
                scale(output, RT(0.5) / step);
            } else {
                increment(output, m_residual, RT(-1));
                scale(output, RT(1) / step);
            }
        }

        void precond (State& output, State const& rhs)
        {
            m_problem.precondition(output, rhs);
        }

        void assign (State& lhs, State const& rhs) const
        {
            m_problem.assign(lhs, rhs);
        }

        RT dotProduct (State const& lhs, State const& rhs) const
        {
            return m_problem.dotProduct(lhs, rhs);
        }

        void increment (State& lhs, State const& rhs, RT scale) const
        {
            m_problem.increment(lhs, rhs, scale);
        }

        void linComb (State& lhs, RT a, State const& rhs_a, RT b,
                      State const& rhs_b) const
        {
            m_problem.linComb(lhs, a, rhs_a, b, rhs_b);
        }

        RT norm2 (State const& vector) const
        {
            return m_problem.norm2(vector);
        }

        void scale (State& vector, RT factor) const
        {
            m_problem.scale(vector, factor);
        }

        void setToZero (State& vector) const
        {
            m_problem.setToZero(vector);
        }

      private:
        Problem& m_problem;
        State const& m_state;
        State const& m_residual;
        bool m_centered_difference;
    };

    Problem& m_problem;
    NewtonKrylovOptions m_options;
};

} // namespace fld_test

#endif
