#ifndef ANDERSON_MIXER_H_
#define ANDERSON_MIXER_H_

#include <AMReX_REAL.H>
#include <AMReX_Vector.H>

#include <limits>

namespace fld_test
{

class AndersonMixer
{
  public:
    AndersonMixer (int depth, amrex::Real beta,
                   amrex::Vector<amrex::Real> weights,
                   amrex::Real upper_bound);

    [[nodiscard]] amrex::Vector<amrex::Real>
    update (amrex::Vector<amrex::Real> const& state,
            amrex::Vector<amrex::Real> const& fixed_point);

    [[nodiscard]] int
    anderson_steps () const noexcept;

    [[nodiscard]] int
    restarts () const noexcept;

  private:
    [[nodiscard]] amrex::Real
    weighted_dot (amrex::Vector<amrex::Real> const& lhs,
                  amrex::Vector<amrex::Real> const& rhs) const;

    [[nodiscard]] amrex::Real
    weighted_norm (amrex::Vector<amrex::Real> const& vector) const;

    void restart (amrex::Vector<amrex::Real> const& state,
                  amrex::Vector<amrex::Real> const& residual);

    int m_depth = 0;
    amrex::Real m_beta = amrex::Real(1);
    amrex::Vector<amrex::Real> m_weights;
    amrex::Real m_upper_bound = std::numeric_limits<amrex::Real>::max();
    amrex::Vector<amrex::Vector<amrex::Real>> m_states;
    amrex::Vector<amrex::Vector<amrex::Real>> m_residuals;
    int m_anderson_steps = 0;
    int m_restarts = 0;
};

} // namespace fld_test

#endif
