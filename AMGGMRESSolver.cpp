#include "AMGGMRESSolver.H"

#include <AMReX.H>
#include <AMReX_AlgVecUtil.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_SpMV.H>

#include <limits>
#include <vector>

namespace fld_test
{

using namespace amrex;

namespace
{

Real
linear_tolerance ()
{
    return (sizeof(Real) == sizeof(float)) ? Real(5.e-5) : Real(2.e-10);
}

AlgVector<Real>
make_vector (AlgPartition const& partition, Vector<Real> const& values)
{
    AMREX_ALWAYS_ASSERT(static_cast<Long>(values.size()) ==
                        partition.numGlobalRows());
    AlgVector<Real> result(partition);
    Long const begin = result.globalBegin();
    Gpu::PinnedVector<Real> local(result.numLocalRows());
    for (Long i = 0; i < result.numLocalRows(); ++i) {
        local[i] = values[begin + i];
    }
    if (!local.empty()) {
        Gpu::copyAsync(Gpu::hostToDevice, local.begin(), local.end(),
                       result.data());
        Gpu::streamSynchronize();
    }
    return result;
}

Vector<Real>
gather_vector (AlgVector<Real> const& vector)
{
    Long const nlocal = vector.numLocalRows();
    Gpu::PinnedVector<Real> local(nlocal);
    if (nlocal > 0) {
        Gpu::copyAsync(Gpu::deviceToHost, vector.data(), vector.data() + nlocal,
                       local.begin());
        Gpu::streamSynchronize();
    }

    auto const& partition = vector.partition();
    int const nprocs = ParallelDescriptor::NProcs();
    std::vector<int> counts(nprocs);
    std::vector<int> offsets(nprocs);
    for (int rank = 0; rank < nprocs; ++rank) {
        Long const count = partition[rank + 1] - partition[rank];
        AMREX_ALWAYS_ASSERT(count <= std::numeric_limits<int>::max());
        AMREX_ALWAYS_ASSERT(partition[rank] <= std::numeric_limits<int>::max());
        counts[rank] = static_cast<int>(count);
        offsets[rank] = static_cast<int>(partition[rank]);
    }

    Vector<Real> global(partition.numGlobalRows());
    ParallelDescriptor::Gatherv(local.data(), static_cast<int>(nlocal),
                                global.data(), counts, offsets,
                                ParallelDescriptor::IOProcessorNumber());
    ParallelDescriptor::Bcast(global.data(), global.size(),
                              ParallelDescriptor::IOProcessorNumber());
    return global;
}

Real
true_relative_residual (SpMatrix<Real> const& matrix,
                        AlgVector<Real> const& solution,
                        AlgVector<Real> const& rhs)
{
    AlgVector<Real> residual(rhs.partition());
    SpMV(residual, matrix, solution);
    LinComb(residual, Real(1), rhs, Real(-1), residual);
    Real const denominator = amrex::max(rhs.norm2(), Real(1.e-30));
    return residual.norm2() / denominator;
}

} // namespace

AMGGMRESSolver::AMGGMRESSolver (SpMatrix<Real> const& matrix,
                                AMG<Real>::Options options)
    : m_matrix(matrix), m_amg(matrix, options), m_gmres(&matrix)
{
    m_amg.setup();
    m_gmres.setPrecond(
        [this] (AlgVector<Real>& lhs, AlgVector<Real> const& rhs)
        { m_amg.apply(lhs, rhs); });
    m_gmres.getGMRES().setRestartLength(50);
    m_gmres.getGMRES().setMaxIters(500);
}

LinearSolution
AMGGMRESSolver::solve (Vector<Real> const& rhs_values)
{
    AlgVector<Real> rhs = make_vector(m_matrix.partition(), rhs_values);
    AlgVector<Real> solution(m_matrix.partition());
    solution.setVal(Real(0));
    m_gmres.solve(solution, rhs, linear_tolerance(), Real(0));

    auto const& gmres = m_gmres.getGMRES();
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        gmres.getStatus() == 0,
        "GMRES+AMG did not converge in an FLD regression");

    LinearSolution result;
    result.values = gather_vector(solution);
    result.iterations = gmres.getNumIters();
    result.relative_residual = true_relative_residual(m_matrix, solution, rhs);
    AMREX_ALWAYS_ASSERT(result.relative_residual <=
                        Real(5) * linear_tolerance());
    return result;
}

AMG<Real>::Diagnostics const&
AMGGMRESSolver::diagnostics () const noexcept
{
    return m_amg.diagnostics();
}

} // namespace fld_test
