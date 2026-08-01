#include "DiffusionHierarchy.H"

#include <AMReX.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_ParallelDescriptor.H>

namespace fld_test
{

using namespace amrex;

namespace
{

Geometry
make_geometry (int n_cell, Array<int, AMREX_SPACEDIM> const& is_periodic)
{
    IntVect const lo(AMREX_D_DECL(0, 0, 0));
    IntVect const hi(AMREX_D_DECL(n_cell - 1, n_cell - 1, 0));
    RealBox const physical_domain(
        {AMREX_D_DECL(Real(0), Real(0), Real(0))},
        {AMREX_D_DECL(Real(1), Real(1), Real(1))});
    return Geometry(Box(lo, hi), physical_domain, CoordSys::cartesian,
                    is_periodic);
}

BoxArray
split_grid (Box const& box, int max_grid_size)
{
    AMREX_ALWAYS_ASSERT(max_grid_size > 0);
    BoxArray result(box);
    result.maxSize(max_grid_size);
    return result;
}

} // namespace

DiffusionHierarchy
make_uniform_hierarchy (int n_cell, int max_grid_size,
                        Array<int, AMREX_SPACEDIM> const& is_periodic)
{
    AMREX_ALWAYS_ASSERT(n_cell > 1);
    DiffusionHierarchy hierarchy;
    hierarchy.geom.push_back(make_geometry(n_cell, is_periodic));
    hierarchy.grids.push_back(
        split_grid(hierarchy.geom[0].Domain(), max_grid_size));
    hierarchy.dmap.emplace_back(hierarchy.grids[0]);
    validate_hierarchy(hierarchy);
    return hierarchy;
}

DiffusionHierarchy
make_centered_patch_hierarchy (
    int n_cell, int refinement_ratio, int max_grid_size,
    Array<int, AMREX_SPACEDIM> const& is_periodic)
{
    AMREX_ALWAYS_ASSERT(n_cell > 3 && n_cell % 4 == 0);
    AMREX_ALWAYS_ASSERT(refinement_ratio > 1);
    AMREX_ALWAYS_ASSERT(max_grid_size % refinement_ratio == 0);

    DiffusionHierarchy hierarchy;
    hierarchy.geom.push_back(make_geometry(n_cell, is_periodic));
    hierarchy.geom.push_back(
        make_geometry(n_cell * refinement_ratio, is_periodic));
    hierarchy.ref_ratio.emplace_back(refinement_ratio);
    hierarchy.grids.push_back(
        split_grid(hierarchy.geom[0].Domain(), max_grid_size));

    IntVect const coarse_lo(AMREX_D_DECL(n_cell / 4, n_cell / 4, 0));
    IntVect const coarse_hi(
        AMREX_D_DECL(3 * n_cell / 4 - 1, 3 * n_cell / 4 - 1, 0));
    Box const fine_patch = amrex::refine(Box(coarse_lo, coarse_hi),
                                         hierarchy.ref_ratio[0]);
    hierarchy.grids.push_back(split_grid(fine_patch, max_grid_size));
    for (auto const& grids : hierarchy.grids) {
        hierarchy.dmap.emplace_back(grids);
    }
    validate_hierarchy(hierarchy);
    return hierarchy;
}

DiffusionHierarchy
make_strip_hierarchy (
    Vector<int> const& level_cells,
    Vector<std::pair<int, int>> const& level_y_bounds, int max_grid_size,
    Array<int, AMREX_SPACEDIM> const& is_periodic)
{
    AMREX_ALWAYS_ASSERT(!level_cells.empty());
    AMREX_ALWAYS_ASSERT(level_cells.size() == level_y_bounds.size());

    DiffusionHierarchy hierarchy;
    for (int level = 0; level < static_cast<int>(level_cells.size()); ++level) {
        int const n_cell = level_cells[level];
        auto const [ylo, yhi] = level_y_bounds[level];
        AMREX_ALWAYS_ASSERT(n_cell > 1);
        AMREX_ALWAYS_ASSERT(ylo >= 0 && ylo < yhi && yhi <= n_cell);
        hierarchy.geom.push_back(make_geometry(n_cell, is_periodic));
        IntVect const lo(AMREX_D_DECL(0, ylo, 0));
        IntVect const hi(AMREX_D_DECL(n_cell - 1, yhi - 1, 0));
        hierarchy.grids.push_back(split_grid(Box(lo, hi), max_grid_size));
        hierarchy.dmap.emplace_back(hierarchy.grids.back());
        if (level > 0) {
            AMREX_ALWAYS_ASSERT(n_cell % level_cells[level - 1] == 0);
            int const ratio = n_cell / level_cells[level - 1];
            AMREX_ALWAYS_ASSERT(ratio > 1);
            AMREX_ALWAYS_ASSERT(max_grid_size % ratio == 0);
            hierarchy.ref_ratio.emplace_back(ratio);
        }
    }
    validate_hierarchy(hierarchy);
    return hierarchy;
}

void
validate_hierarchy (DiffusionHierarchy const& hierarchy)
{
    int const nlevels = static_cast<int>(hierarchy.geom.size());
    AMREX_ALWAYS_ASSERT(nlevels > 0);
    AMREX_ALWAYS_ASSERT(static_cast<int>(hierarchy.grids.size()) == nlevels);
    AMREX_ALWAYS_ASSERT(static_cast<int>(hierarchy.dmap.size()) == nlevels);
    AMREX_ALWAYS_ASSERT(static_cast<int>(hierarchy.ref_ratio.size()) ==
                        nlevels - 1);

    for (int level = 0; level < nlevels; ++level) {
        AMREX_ALWAYS_ASSERT(hierarchy.grids[level].ixType().cellCentered());
        AMREX_ALWAYS_ASSERT(hierarchy.geom[level].Domain().contains(
            hierarchy.grids[level].minimalBox()));
        if (level + 1 < nlevels) {
            auto const& ratio = hierarchy.ref_ratio[level];
            AMREX_ALWAYS_ASSERT(ratio.allGT(1));
            AMREX_ALWAYS_ASSERT(
                hierarchy.geom[level + 1].Domain() ==
                amrex::refine(hierarchy.geom[level].Domain(), ratio));
            BoxArray coarse_fine = hierarchy.grids[level + 1];
            coarse_fine.coarsen(ratio);
            for (int ibox = 0; ibox < coarse_fine.size(); ++ibox) {
                AMREX_ALWAYS_ASSERT(
                    hierarchy.grids[level].contains(coarse_fine[ibox]));
            }
        }
    }
}

Vector<iMultiFab>
make_composite_masks (DiffusionHierarchy const& hierarchy, int nghost,
                      MFInfo const& info)
{
    validate_hierarchy(hierarchy);
    AMREX_ALWAYS_ASSERT(nghost >= 0);
    int const nlevels = static_cast<int>(hierarchy.geom.size());
    Vector<iMultiFab> masks(nlevels);
    for (int level = 0; level < nlevels; ++level) {
        if (level + 1 < nlevels) {
            masks[level] = amrex::makeFineMask(
                hierarchy.grids[level], hierarchy.dmap[level],
                IntVect(nghost), hierarchy.grids[level + 1],
                hierarchy.ref_ratio[level], hierarchy.geom[level].periodicity(),
                1, 0, info);
        } else {
            masks[level].define(hierarchy.grids[level], hierarchy.dmap[level],
                                1, nghost, info);
            masks[level].setVal(1);
        }
    }
    return masks;
}

Long
composite_cell_count (Vector<iMultiFab> const& masks)
{
    Long count = 0;
    for (auto const& mask : masks) {
        count += mask.sum(0, 0, true);
    }
    ParallelDescriptor::ReduceLongSum(count);
    return count;
}

} // namespace fld_test
