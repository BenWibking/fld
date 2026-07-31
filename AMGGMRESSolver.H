#ifndef AMG_GMRES_SOLVER_H_
#define AMG_GMRES_SOLVER_H_

#include <AMReX_AMG.H>
#include <AMReX_GMRES_MV.H>
#include <AMReX_SpMatrix.H>
#include <AMReX_Vector.H>

namespace fld_test
{

struct LinearSolution
{
    amrex::Vector<amrex::Real> values;
    int iterations = 0;
    amrex::Real relative_residual = amrex::Real(0);
};

class AMGGMRESSolver
{
  public:
    explicit AMGGMRESSolver (
        amrex::SpMatrix<amrex::Real> const& matrix,
        amrex::AMG<amrex::Real>::Options options = {});

    AMGGMRESSolver (AMGGMRESSolver const&) = delete;
    AMGGMRESSolver& operator=(AMGGMRESSolver const&) = delete;

    [[nodiscard]] LinearSolution
    solve (amrex::Vector<amrex::Real> const& rhs_values);

    [[nodiscard]] amrex::AMG<amrex::Real>::Diagnostics const&
    diagnostics () const noexcept;

  private:
    amrex::SpMatrix<amrex::Real> const& m_matrix;
    amrex::AMG<amrex::Real> m_amg;
    amrex::GMRES_MV<amrex::Real> m_gmres;
};

} // namespace fld_test

#endif
