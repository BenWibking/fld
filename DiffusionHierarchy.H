#ifndef DIFFUSION_HIERARCHY_H_
#define DIFFUSION_HIERARCHY_H_

#include <AMReX_Array.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Vector.H>
#include <AMReX_iMultiFab.H>

#include <utility>

namespace fld_test
{

struct DiffusionHierarchy
{
    amrex::Vector<amrex::Geometry> geom;
    amrex::Vector<amrex::BoxArray> grids;
    amrex::Vector<amrex::DistributionMapping> dmap;
    amrex::Vector<amrex::IntVect> ref_ratio;
};

[[nodiscard]] DiffusionHierarchy make_uniform_hierarchy (
    int n_cell, int max_grid_size,
    amrex::Array<int, AMREX_SPACEDIM> const& is_periodic);

[[nodiscard]] DiffusionHierarchy make_centered_patch_hierarchy (
    int n_cell, int refinement_ratio, int max_grid_size,
    amrex::Array<int, AMREX_SPACEDIM> const& is_periodic);

[[nodiscard]] DiffusionHierarchy make_strip_hierarchy (
    amrex::Vector<int> const& level_cells,
    amrex::Vector<std::pair<int, int>> const& level_y_bounds,
    int max_grid_size,
    amrex::Array<int, AMREX_SPACEDIM> const& is_periodic);

void validate_hierarchy (DiffusionHierarchy const& hierarchy);

[[nodiscard]] amrex::Vector<amrex::iMultiFab> make_composite_masks (
    DiffusionHierarchy const& hierarchy, int nghost = 0,
    amrex::MFInfo const& info = amrex::MFInfo());

[[nodiscard]] amrex::Long composite_cell_count (
    amrex::Vector<amrex::iMultiFab> const& masks);

} // namespace fld_test

#endif
