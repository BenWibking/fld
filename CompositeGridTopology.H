#ifndef COMPOSITE_GRID_TOPOLOGY_H_
#define COMPOSITE_GRID_TOPOLOGY_H_

#include <AMReX_AlgPartition.H>
#include <AMReX_Array.H>
#include <AMReX_CSR.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_LO_BCTYPES.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Vector.H>

namespace fld_test
{

/**
 * Immutable geometric and algebraic structure of a cell-centered AMR
 * composite grid.
 *
 * Construction resolves every same-level and coarse/fine neighbor once and
 * assigns each directed face connection a canonical CSR slot.  Numerical
 * assembly therefore only evaluates coefficients and accumulates values into
 * an already sorted, duplicate-free sparsity pattern.
 */
class CompositeGridTopology
{
  public:
    using HostCSR = amrex::CSR<amrex::Real, amrex::Gpu::PinnedVector>;

    struct Cell
    {
        int level = -1;
        int local_grid = -1;
        amrex::IntVect index = amrex::IntVect::TheZeroVector();
        amrex::Long global_row = -1;
        amrex::Long diagonal_slot = -1;
        amrex::Real volume = amrex::Real(0);
    };

    struct BoundaryData
    {
        amrex::Vector<amrex::MultiFab const*> level;
        amrex::Vector<amrex::MultiFab const*> robin_a;
        amrex::Vector<amrex::MultiFab const*> robin_b;
        amrex::Vector<amrex::MultiFab const*> robin_f;
    };

    struct NumericalAssembly
    {
        HostCSR matrix;
        amrex::Gpu::PinnedVector<amrex::Real> boundary_rhs;
        amrex::Long local_coarse_fine_connections = 0;
        amrex::Real local_minimum_diagonal = amrex::Real(0);
        amrex::Real local_maximum_off_diagonal = amrex::Real(0);
    };

    CompositeGridTopology (
        amrex::Vector<amrex::Geometry> geom,
        amrex::Vector<amrex::BoxArray> grids,
        amrex::Vector<amrex::DistributionMapping> dmap);

    [[nodiscard]] int numLevels () const noexcept;
    [[nodiscard]] amrex::Vector<amrex::Geometry> const& geometry () const
        noexcept;
    [[nodiscard]] amrex::Vector<amrex::BoxArray> const& grids () const noexcept;
    [[nodiscard]] amrex::Vector<amrex::IntVect> const& refRatio () const
        noexcept;
    [[nodiscard]] amrex::AlgPartition const& partition () const noexcept;
    [[nodiscard]] amrex::Vector<Cell> const& cells () const noexcept;
    [[nodiscard]] amrex::Long globalRows () const noexcept;
    [[nodiscard]] amrex::Long localRows () const noexcept;

    [[nodiscard]] NumericalAssembly assemble (
        amrex::Real ascalar, amrex::Real bscalar,
        amrex::Vector<amrex::MultiFab const*> const& acoef,
        amrex::Vector<amrex::Array<amrex::MultiFab const*, AMREX_SPACEDIM>>
            const& bcoef,
        amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM> const& lobc,
        amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM> const& hibc,
        BoundaryData const& boundary) const;

  private:
    struct Connection
    {
        amrex::Long local_row = -1;
        amrex::Long matrix_slot = -1;
        amrex::Long column = -1;
        int level = -1;
        int direction = -1;
        amrex::IntVect face = amrex::IntVect::TheZeroVector();
        amrex::Real geometric_weight = amrex::Real(0);
        bool fine_coefficient_on_coarse_layout = false;
        bool coarse_fine = false;
    };

    struct PhysicalFace
    {
        amrex::Long local_row = -1;
        int level = -1;
        int direction = -1;
        int side = 0;
        amrex::IntVect cell = amrex::IntVect::TheZeroVector();
        amrex::IntVect face = amrex::IntVect::TheZeroVector();
        amrex::IntVect exterior = amrex::IntVect::TheZeroVector();
        amrex::Real area = amrex::Real(0);
        amrex::Real distance = amrex::Real(0);
    };

    void validateHierarchy ();
    void buildRowsAndConnections ();
    void buildPattern ();

    amrex::Vector<amrex::Geometry> m_geom;
    amrex::Vector<amrex::BoxArray> m_grids;
    amrex::Vector<amrex::DistributionMapping> m_dmap;
    amrex::Vector<amrex::IntVect> m_ref_ratio;
    amrex::Vector<amrex::BoxArray> m_refined_coarse;
    amrex::AlgPartition m_partition;
    amrex::Vector<Cell> m_cells;
    amrex::Vector<Connection> m_connections;
    amrex::Vector<PhysicalFace> m_physical_faces;
    amrex::Gpu::PinnedVector<amrex::Long> m_row_offset;
    amrex::Gpu::PinnedVector<amrex::Long> m_col_index;
};

} // namespace fld_test

#endif
