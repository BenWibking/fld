#ifndef ANDERSON_MIXER_H_
#define ANDERSON_MIXER_H_

#include "DiffusionHierarchy.H"
#include "FLDFieldOps.H"

#include <limits>

namespace fld_test
{

class AndersonMixer
{
  public:
    AndersonMixer (DiffusionHierarchy const& hierarchy,
                   amrex::Vector<amrex::iMultiFab> const& composite_masks,
                   int depth, amrex::Real beta, amrex::Real upper_bound);

    void update (LevelData& state, LevelData const& fixed_point);

    [[nodiscard]] int anderson_steps () const noexcept;
    [[nodiscard]] int restarts () const noexcept;

  private:
    [[nodiscard]] amrex::Real weighted_dot (LevelData const& lhs,
                                            LevelData const& rhs) const;
    [[nodiscard]] amrex::Real weighted_norm (LevelData const& vector) const;
    void restart (LevelData const& state, LevelData const& residual);

    DiffusionHierarchy const* m_hierarchy = nullptr;
    amrex::Vector<amrex::iMultiFab> const* m_masks = nullptr;
    int m_depth = 0;
    amrex::Real m_beta = amrex::Real(1);
    amrex::Real m_upper_bound = std::numeric_limits<amrex::Real>::max();
    amrex::Vector<LevelData> m_states;
    amrex::Vector<LevelData> m_residuals;
    int m_anderson_steps = 0;
    int m_restarts = 0;
};

} // namespace fld_test

#endif
