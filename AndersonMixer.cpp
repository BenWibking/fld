#include "AndersonMixer.H"

#include <AMReX.H>
#include <AMReX_GMRES_MV.H>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <utility>

namespace fld_test
{

using namespace amrex;

namespace
{

// Adapt a small host dense matrix to the generic AMReX GMRES interface.  This
// reuses GMRES's Arnoldi, Givens rotation, and back-substitution machinery for
// the regularized Anderson normal equations.
class DenseMatrixOperator
{
  public:
    using RT = Real;

    explicit DenseMatrixOperator (Vector<Vector<Real>> matrix)
        : m_matrix(std::move(matrix)),
          m_inverse_diagonal(m_matrix.size(), Real(1))
    {
        AMREX_ALWAYS_ASSERT(!m_matrix.empty());
        for (Long i = 0; i < m_matrix.size(); ++i) {
            AMREX_ALWAYS_ASSERT(m_matrix[i].size() == m_matrix.size());
            AMREX_ALWAYS_ASSERT(m_matrix[i][i] > Real(0));
            m_inverse_diagonal[i] = Real(1) / m_matrix[i][i];
        }
    }

    void
    apply (Vector<Real>& lhs, Vector<Real> const& rhs) const
    {
        AMREX_ALWAYS_ASSERT(rhs.size() == m_matrix.size());
        lhs.assign(m_matrix.size(), Real(0));
        for (Long i = 0; i < m_matrix.size(); ++i) {
            for (Long j = 0; j < m_matrix.size(); ++j) {
                lhs[i] += m_matrix[i][j] * rhs[j];
            }
        }
    }

    static void
    assign (Vector<Real>& lhs, Vector<Real> const& rhs)
    {
        lhs = rhs;
    }

    static Real
    dotProduct (Vector<Real> const& lhs, Vector<Real> const& rhs)
    {
        AMREX_ALWAYS_ASSERT(lhs.size() == rhs.size());
        return std::inner_product(lhs.begin(), lhs.end(), rhs.begin(),
                                  Real(0));
    }

    static void
    increment (Vector<Real>& lhs, Vector<Real> const& rhs, Real scale)
    {
        AMREX_ALWAYS_ASSERT(lhs.size() == rhs.size());
        for (Long i = 0; i < lhs.size(); ++i) {
            lhs[i] += scale * rhs[i];
        }
    }

    static void
    linComb (Vector<Real>& lhs, Real lhs_scale,
             Vector<Real> const& lhs_vector, Real rhs_scale,
             Vector<Real> const& rhs_vector)
    {
        AMREX_ALWAYS_ASSERT(lhs_vector.size() == rhs_vector.size());
        lhs.resize(lhs_vector.size());
        for (Long i = 0; i < lhs.size(); ++i) {
            lhs[i] =
                lhs_scale * lhs_vector[i] + rhs_scale * rhs_vector[i];
        }
    }

    [[nodiscard]] Vector<Real>
    makeVecRHS () const
    {
        return Vector<Real>(m_matrix.size(), Real(0));
    }

    [[nodiscard]] Vector<Real>
    makeVecLHS () const
    {
        return Vector<Real>(m_matrix.size(), Real(0));
    }

    static Real
    norm2 (Vector<Real> const& vector)
    {
        return std::sqrt(dotProduct(vector, vector));
    }

    void
    precond (Vector<Real>& lhs, Vector<Real> const& rhs) const
    {
        AMREX_ALWAYS_ASSERT(rhs.size() == m_inverse_diagonal.size());
        lhs.resize(rhs.size());
        for (Long i = 0; i < rhs.size(); ++i) {
            lhs[i] = m_inverse_diagonal[i] * rhs[i];
        }
    }

    static void
    scale (Vector<Real>& vector, Real factor)
    {
        for (Real& value : vector) {
            value *= factor;
        }
    }

    static void
    setToZero (Vector<Real>& vector)
    {
        std::fill(vector.begin(), vector.end(), Real(0));
    }

  private:
    Vector<Vector<Real>> m_matrix;
    Vector<Real> m_inverse_diagonal;
};

bool
solve_anderson_coefficients (Vector<Vector<Real>> matrix,
                             Vector<Real> const& rhs,
                             Vector<Real>& coefficients)
{
    AMREX_ALWAYS_ASSERT(!matrix.empty());
    AMREX_ALWAYS_ASSERT(matrix.size() == rhs.size());

    DenseMatrixOperator linear_operator(std::move(matrix));
    GMRES<Vector<Real>, DenseMatrixOperator> gmres;
    gmres.define(linear_operator);
    int const dimension = static_cast<int>(rhs.size());
    gmres.setRestartLength(dimension);
    gmres.setMaxIters(2 * dimension);
    coefficients.assign(rhs.size(), Real(0));
    Real const tolerance =
        (sizeof(Real) == sizeof(float)) ? Real(2.e-4) : Real(1.e-10);
    gmres.solve(coefficients, rhs, tolerance, Real(0), 2 * dimension);
    if (gmres.getStatus() != 0) {
        return false;
    }
    return std::all_of(coefficients.begin(), coefficients.end(),
                       [] (Real value) noexcept
                       { return std::isfinite(value); });
}

} // namespace

AndersonMixer::AndersonMixer (
    DiffusionHierarchy const& hierarchy,
    Vector<iMultiFab> const& composite_masks, int depth, Real beta,
    Real upper_bound)
    : m_hierarchy(&hierarchy), m_masks(&composite_masks), m_depth(depth),
      m_beta(beta), m_upper_bound(upper_bound)
{
    AMREX_ALWAYS_ASSERT(m_depth >= 0);
    AMREX_ALWAYS_ASSERT(m_beta > Real(0));
    AMREX_ALWAYS_ASSERT(m_beta <= Real(1));
    AMREX_ALWAYS_ASSERT(m_upper_bound > Real(0));
    AMREX_ALWAYS_ASSERT(m_hierarchy->geom.size() == m_masks->size());
}

void
AndersonMixer::update (LevelData& state, LevelData const& fixed_point)
{
    LevelData residual = clone_level_data(fixed_point);
    saxpy_level_data(residual, Real(-1), state);
    LevelData picard_state = clone_level_data(state);
    saxpy_level_data(picard_state, m_beta, residual);
    if (m_depth == 0) {
        copy_level_data(state, picard_state);
        average_down_hierarchy(state, *m_hierarchy);
        return;
    }

    Real const residual_norm = weighted_norm(residual);
    if (!m_residuals.empty() &&
        residual_norm > Real(2) * weighted_norm(m_residuals.back())) {
        restart(state, residual);
        copy_level_data(state, picard_state);
        average_down_hierarchy(state, *m_hierarchy);
        return;
    }

    m_states.push_back(clone_level_data(state));
    m_residuals.push_back(clone_level_data(residual));
    while (static_cast<int>(m_states.size()) > m_depth + 1) {
        m_states.erase(m_states.begin());
        m_residuals.erase(m_residuals.begin());
    }

    int const difference_count = static_cast<int>(m_states.size()) - 1;
    if (difference_count == 0) {
        copy_level_data(state, picard_state);
        average_down_hierarchy(state, *m_hierarchy);
        return;
    }

    Vector<LevelData> state_differences;
    Vector<LevelData> residual_differences;
    state_differences.reserve(difference_count);
    residual_differences.reserve(difference_count);
    for (int column = 0; column < difference_count; ++column) {
        state_differences.push_back(clone_level_data(m_states[column + 1]));
        saxpy_level_data(state_differences.back(), Real(-1),
                         m_states[column]);
        residual_differences.push_back(
            clone_level_data(m_residuals[column + 1]));
        saxpy_level_data(residual_differences.back(), Real(-1),
                         m_residuals[column]);
    }

    Vector<Vector<Real>> normal_matrix(
        difference_count, Vector<Real>(difference_count));
    Vector<Real> normal_rhs(difference_count);
    // Minimize ||f_k - Delta F gamma|| in the volume-weighted L2 norm.
    Real trace = Real(0);
    for (int i = 0; i < difference_count; ++i) {
        normal_rhs[i] = weighted_dot(residual_differences[i], residual);
        for (int j = 0; j < difference_count; ++j) {
            normal_matrix[i][j] = weighted_dot(
                residual_differences[i], residual_differences[j]);
        }
        trace += normal_matrix[i][i];
    }
    Real const regularization_factor =
        (sizeof(Real) == sizeof(float)) ? Real(1.e-5) : Real(1.e-12);
    Real const regularization =
        regularization_factor *
        amrex::max(trace / Real(difference_count), Real(1.e-30));
    for (int i = 0; i < difference_count; ++i) {
        normal_matrix[i][i] += regularization;
    }

    Vector<Real> coefficients;
    if (!solve_anderson_coefficients(std::move(normal_matrix), normal_rhs,
                                     coefficients)) {
        restart(state, residual);
        copy_level_data(state, picard_state);
        average_down_hierarchy(state, *m_hierarchy);
        return;
    }

    LevelData candidate = clone_level_data(picard_state);
    // Type-II Anderson update:
    // x_{k+1} = x_k + beta f_k - (Delta X + beta Delta F) gamma.
    for (int column = 0; column < difference_count; ++column) {
        LevelData correction = clone_level_data(state_differences[column]);
        saxpy_level_data(correction, m_beta, residual_differences[column]);
        saxpy_level_data(candidate, -coefficients[column], correction);
    }

    LevelData candidate_step = clone_level_data(candidate);
    saxpy_level_data(candidate_step, Real(-1), state);
    LevelData picard_step = clone_level_data(picard_state);
    saxpy_level_data(picard_step, Real(-1), state);
    auto const [minimum, maximum] =
        composite_minimum_maximum(candidate, *m_masks);
    bool const physical = composite_all_finite(candidate, *m_masks) &&
                          minimum >= Real(0) && maximum <= m_upper_bound;
    Real const picard_step_norm = weighted_norm(picard_step);
    Real const candidate_step_norm = weighted_norm(candidate_step);
    // Reject proposals that violate the radiation maximum principle or take a
    // much larger step than the underlying fixed-point update.
    if (!physical || !std::isfinite(candidate_step_norm) ||
        candidate_step_norm >
            Real(10) * amrex::max(picard_step_norm, Real(1.e-30))) {
        restart(state, residual);
        copy_level_data(state, picard_state);
        average_down_hierarchy(state, *m_hierarchy);
        return;
    }

    ++m_anderson_steps;
    copy_level_data(state, candidate);
    average_down_hierarchy(state, *m_hierarchy);
}

int
AndersonMixer::anderson_steps () const noexcept
{
    return m_anderson_steps;
}

int
AndersonMixer::restarts () const noexcept
{
    return m_restarts;
}

Real
AndersonMixer::weighted_dot (LevelData const& lhs,
                             LevelData const& rhs) const
{
    return composite_weighted_dot(lhs, rhs, *m_hierarchy, *m_masks);
}

Real
AndersonMixer::weighted_norm (LevelData const& vector) const
{
    return std::sqrt(amrex::max(weighted_dot(vector, vector), Real(0)));
}

void
AndersonMixer::restart (LevelData const& state, LevelData const& residual)
{
    m_states.clear();
    m_residuals.clear();
    m_states.push_back(clone_level_data(state));
    m_residuals.push_back(clone_level_data(residual));
    ++m_restarts;
}

} // namespace fld_test
