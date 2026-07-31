#ifndef FLD_LINEAR_SYSTEM_H_
#define FLD_LINEAR_SYSTEM_H_

#include "FLDMesh.H"

#include <AMReX_SpMatrix.H>
#include <AMReX_Vector.H>

#include <utility>

namespace fld_test
{

struct LinearSystem
{
    amrex::SpMatrix<amrex::Real> matrix;
    amrex::Vector<amrex::Real> rhs;
};

[[nodiscard]] amrex::Real face_transmissibility (
    Face const& face, amrex::Long row,
    amrex::Vector<amrex::Real> const& diffusion,
    amrex::Vector<amrex::Real> const* rosseland_extinction = nullptr);

[[nodiscard]] LinearSystem assemble_system (
    Mesh const& mesh, amrex::Vector<amrex::Real> const& diffusion,
    amrex::Vector<amrex::Real> const& old_state, amrex::Real dt,
    bool transient,
    amrex::Vector<amrex::Real> const* rosseland_extinction = nullptr);

amrex::Vector<amrex::Real> compute_diffusion (
    Mesh const& mesh, amrex::Vector<amrex::Real> const& energy,
    amrex::Vector<amrex::Real> const& extinction, bool limited,
    amrex::Real* maximum_flux_fraction = nullptr);

[[nodiscard]] std::pair<amrex::Real, amrex::Real>
minimum_maximum (amrex::Vector<amrex::Real> const& values);

[[nodiscard]] amrex::Real maximum_relative_change (
    amrex::Vector<amrex::Real> const& lhs,
    amrex::Vector<amrex::Real> const& rhs);

} // namespace fld_test

#endif
