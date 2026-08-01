#ifndef FLD_FIELD_OPS_H_
#define FLD_FIELD_OPS_H_

#include "DiffusionHierarchy.H"

#include <AMReX_LO_BCTYPES.H>
#include <AMReX_MultiFab.H>

#include <memory>
#include <utility>

namespace fld_test
{

using LevelData = amrex::Vector<std::unique_ptr<amrex::MultiFab>>;
using FaceData = amrex::Vector<
    amrex::Array<std::unique_ptr<amrex::MultiFab>, AMREX_SPACEDIM>>;

struct PhysicalBoundaryData
{
    amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM> lo;
    amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM> hi;
    amrex::Array<amrex::Real, AMREX_SPACEDIM> lo_value{};
    amrex::Array<amrex::Real, AMREX_SPACEDIM> hi_value{};
};

[[nodiscard]] LevelData make_cell_data (DiffusionHierarchy const& hierarchy,
                                        int ncomp, int nghost);

[[nodiscard]] FaceData make_face_data (DiffusionHierarchy const& hierarchy,
                                       int ncomp = 1, int nghost = 0);

[[nodiscard]] LevelData clone_level_data (LevelData const& source);

void copy_level_data (LevelData& destination, LevelData const& source,
                      int nghost = 0);

void set_level_data (LevelData& data, amrex::Real value);

void lincomb_level_data (LevelData& destination, amrex::Real a,
                         LevelData const& lhs, amrex::Real b,
                         LevelData const& rhs);

void saxpy_level_data (LevelData& destination, amrex::Real a,
                       LevelData const& source);

void fill_level_ghosts (LevelData& data,
                        DiffusionHierarchy const& hierarchy);

void average_down_hierarchy (LevelData& data,
                             DiffusionHierarchy const& hierarchy);

[[nodiscard]] amrex::Vector<amrex::MultiFab*> get_level_ptrs (
    LevelData& data);

[[nodiscard]] amrex::Vector<amrex::MultiFab const*> get_level_const_ptrs (
    LevelData const& data);

[[nodiscard]] amrex::Vector<
    amrex::Array<amrex::MultiFab const*, AMREX_SPACEDIM>>
get_face_const_ptrs (FaceData const& data);

[[nodiscard]] amrex::Real composite_weighted_dot (
    LevelData const& lhs, LevelData const& rhs,
    DiffusionHierarchy const& hierarchy,
    amrex::Vector<amrex::iMultiFab> const& masks);

[[nodiscard]] amrex::Real composite_volume_sum (
    LevelData const& data, DiffusionHierarchy const& hierarchy,
    amrex::Vector<amrex::iMultiFab> const& masks);

[[nodiscard]] std::pair<amrex::Real, amrex::Real> composite_minimum_maximum (
    LevelData const& data, amrex::Vector<amrex::iMultiFab> const& masks);

[[nodiscard]] amrex::Real composite_maximum_relative_change (
    LevelData const& lhs, LevelData const& rhs,
    amrex::Vector<amrex::iMultiFab> const& masks);

[[nodiscard]] bool composite_all_finite (
    LevelData const& data, amrex::Vector<amrex::iMultiFab> const& masks);

void compute_diffusion (DiffusionHierarchy const& hierarchy,
                        LevelData& energy, LevelData& extinction,
                        LevelData& diffusion,
                        PhysicalBoundaryData const& boundary, bool limited,
                        amrex::Real* maximum_flux_fraction = nullptr);

void fill_face_coefficients (DiffusionHierarchy const& hierarchy,
                             LevelData& diffusion, LevelData* extinction,
                             FaceData& face_coefficients,
                             bool use_surface_opacity);

} // namespace fld_test

#endif
