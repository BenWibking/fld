#include "CompositeGridTopology.H"

#include <AMReX.H>
#include <AMReX_BaseFab.H>
#include <AMReX_BoxIterator.H>
#include <AMReX_FabArray.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_ParallelDescriptor.H>

#include <algorithm>
#include <limits>
#include <numeric>
#include <utility>

namespace fld_test
{

using namespace amrex;

namespace
{

using LongFabArray = FabArray<BaseFab<Long>>;

MFInfo
topology_host_info ()
{
    return MFInfo().SetArena(The_Pinned_Arena());
}

Vector<Long>
make_partition_rows (Long local_rows)
{
    int const nprocs = ParallelDescriptor::NProcs();
    int const root = ParallelDescriptor::IOProcessorNumber();
    Vector<Long> counts(nprocs, Long(0));
    ParallelDescriptor::Gather(&local_rows, 1, counts.data(), 1, root);
    ParallelDescriptor::Bcast(counts.data(), counts.size(), root);
    Vector<Long> rows(nprocs + 1, Long(0));
    std::partial_sum(counts.begin(), counts.end(), rows.begin() + 1);
    return rows;
}

Real
cell_volume (Geometry const& geometry) noexcept
{
    auto const dx = geometry.CellSizeArray();
    return AMREX_D_TERM(dx[0], *dx[1], *dx[2]);
}

bool
is_robin (LinOpBCType type) noexcept
{
    return type == LinOpBCType::Robin || type == LinOpBCType::Marshak;
}

} // namespace

CompositeGridTopology::CompositeGridTopology (
    Vector<Geometry> geom, Vector<BoxArray> grids,
    Vector<DistributionMapping> dmap)
    : m_geom(std::move(geom)), m_grids(std::move(grids)),
      m_dmap(std::move(dmap))
{
    static_assert(AMREX_SPACEDIM == 2,
                  "CompositeGridTopology currently implements 2-D faces");
    validateHierarchy();
    buildRowsAndConnections();
    buildPattern();
}

void
CompositeGridTopology::validateHierarchy ()
{
    int const nlevels = numLevels();
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        nlevels > 0 && static_cast<int>(m_grids.size()) == nlevels &&
            static_cast<int>(m_dmap.size()) == nlevels,
        "CompositeGridTopology requires matching, nonempty hierarchy vectors");
    m_ref_ratio.resize(nlevels - 1);
    m_refined_coarse.resize(nlevels - 1);
    for (int level = 0; level < nlevels; ++level) {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_grids[level].ixType().cellCentered(),
            "CompositeGridTopology supports only cell-centered grids");
        AMREX_ALWAYS_ASSERT(
            m_geom[level].Domain().contains(m_grids[level].minimalBox()));
        if (level + 1 < nlevels) {
            m_ref_ratio[level] = m_geom[level + 1].Domain().length() /
                                 m_geom[level].Domain().length();
            AMREX_ALWAYS_ASSERT(m_ref_ratio[level].allGT(1));
            AMREX_ALWAYS_ASSERT(
                m_geom[level + 1].Domain() ==
                amrex::refine(m_geom[level].Domain(), m_ref_ratio[level]));
            m_refined_coarse[level] = m_grids[level];
            m_refined_coarse[level].refine(m_ref_ratio[level]);
        }
    }
}

void
CompositeGridTopology::buildRowsAndConnections ()
{
    int const nlevels = numLevels();
    Vector<iMultiFab> active(nlevels);
    Vector<std::unique_ptr<LongFabArray>> row_ids(nlevels);
    Long local_count = 0;
    for (int level = 0; level < nlevels; ++level) {
        if (level + 1 < nlevels) {
            active[level] = amrex::makeFineMask(
                m_grids[level], m_dmap[level], IntVect(2),
                m_grids[level + 1], m_ref_ratio[level],
                m_geom[level].periodicity(), 1, 0, topology_host_info());
        } else {
            active[level].define(m_grids[level], m_dmap[level], 1, 2,
                                 topology_host_info());
            active[level].setVal(1);
        }
        local_count += active[level].sum(0, 0, true);
    }

    m_partition.define(make_partition_rows(local_count));
    int const myproc = ParallelDescriptor::MyProc();
    Long const begin = m_partition[myproc];
    Long next_row = begin;
    for (int level = 0; level < nlevels; ++level) {
        row_ids[level] = std::make_unique<LongFabArray>(
            m_grids[level], m_dmap[level], 1, 2, topology_host_info());
        row_ids[level]->setVal(Long(-1));
        for (MFIter mfi(active[level]); mfi.isValid(); ++mfi) {
            auto const& mask = active[level][mfi];
            auto& rows = (*row_ids[level])[mfi];
            for (BoxIterator bit(mfi.validbox()); bit.ok(); ++bit) {
                IntVect const& iv = bit();
                if (mask(iv) != 0) {
                    rows(iv) = next_row++;
                }
            }
        }
        row_ids[level]->FillBoundary(m_geom[level].periodicity());
    }
    AMREX_ALWAYS_ASSERT(next_row == m_partition[myproc + 1]);

    Vector<std::unique_ptr<LongFabArray>> fine_rows_on_coarse(nlevels - 1);
    Vector<std::unique_ptr<LongFabArray>> coarse_rows_on_fine(nlevels - 1);
    for (int level = 0; level + 1 < nlevels; ++level) {
        fine_rows_on_coarse[level] = std::make_unique<LongFabArray>(
            m_refined_coarse[level], m_dmap[level], 1, 1,
            topology_host_info());
        fine_rows_on_coarse[level]->setVal(Long(-1));
        fine_rows_on_coarse[level]->ParallelCopy(
            *row_ids[level + 1], 0, 0, 1, IntVect(0), IntVect(1),
            m_geom[level + 1].periodicity());

        BoxArray coarsened_fine = m_grids[level + 1];
        coarsened_fine.coarsen(m_ref_ratio[level]);
        coarse_rows_on_fine[level] = std::make_unique<LongFabArray>(
            coarsened_fine, m_dmap[level + 1], 1, 2,
            topology_host_info());
        coarse_rows_on_fine[level]->setVal(Long(-1));
        coarse_rows_on_fine[level]->ParallelCopy(
            *row_ids[level], 0, 0, 1, IntVect(0), IntVect(2),
            m_geom[level].periodicity());
    }
    Gpu::streamSynchronize();

    m_cells.resize(local_count);
    Vector<int> cell_initialized(local_count, 0);
    for (int level = 0; level < nlevels; ++level) {
        Real const volume = cell_volume(m_geom[level]);
        auto const dx = m_geom[level].CellSizeArray();
        Box const domain = m_geom[level].Domain();
        for (MFIter mfi(active[level]); mfi.isValid(); ++mfi) {
            auto const& mask = active[level][mfi];
            auto const& row_fab = (*row_ids[level])[mfi];
            for (BoxIterator bit(mfi.validbox()); bit.ok(); ++bit) {
                IntVect const iv = bit();
                if (mask(iv) == 0) {
                    continue;
                }
                Long const global_row = row_fab(iv);
                Long const local_row = global_row - begin;
                AMREX_ALWAYS_ASSERT(local_row >= 0 && local_row < local_count);
                m_cells[local_row] =
                    Cell{level, mfi.LocalIndex(), iv, global_row, Long(-1),
                         volume};
                cell_initialized[local_row] = 1;

                for (int direction = 0; direction < AMREX_SPACEDIM;
                     ++direction) {
                    Real const area = volume / dx[direction];
                    for (int side : {-1, 1}) {
                        IntVect neighbor = iv;
                        neighbor[direction] += side;
                        IntVect face = iv;
                        if (side > 0) {
                            face[direction] += 1;
                        }
                        bool const outside =
                            neighbor[direction] < domain.smallEnd(direction) ||
                            neighbor[direction] > domain.bigEnd(direction);
                        if (outside && !m_geom[level].isPeriodic(direction)) {
                            IntVect exterior = iv;
                            exterior[direction] += side;
                            m_physical_faces.push_back(PhysicalFace{
                                local_row, level, direction, side, iv,
                                face, exterior, area,
                                Real(0.5) * dx[direction]});
                            continue;
                        }

                        Long const same_level_row = row_fab(neighbor);
                        if (same_level_row >= 0) {
                            m_connections.push_back(Connection{
                                local_row, Long(-1), same_level_row, level,
                                direction, face, area / dx[direction], false,
                                false});
                            continue;
                        }

                        int fine_neighbor_count = 0;
                        if (level + 1 < nlevels) {
                            auto const rr = m_ref_ratio[level];
                            auto const& fine_rows =
                                (*fine_rows_on_coarse[level])[mfi];
                            auto const dxf =
                                m_geom[level + 1].CellSizeArray();
                            Real const fine_volume =
                                cell_volume(m_geom[level + 1]);
                            Real const fine_area =
                                fine_volume / dxf[direction];
                            Real const distance = Real(0.5) *
                                (dx[direction] + dxf[direction]);
                            int const transverse = 1 - direction;
                            for (int offset = 0; offset < rr[transverse];
                                 ++offset) {
                                IntVect fine_cell = IntVect::TheZeroVector();
                                IntVect fine_face = IntVect::TheZeroVector();
                                if (direction == 0) {
                                    fine_cell[1] = iv[1] * rr[1] + offset;
                                    fine_face[1] = fine_cell[1];
                                    if (side < 0) {
                                        fine_cell[0] = iv[0] * rr[0] - 1;
                                        fine_face[0] = iv[0] * rr[0];
                                    } else {
                                        fine_cell[0] = (iv[0] + 1) * rr[0];
                                        fine_face[0] = fine_cell[0];
                                    }
                                } else {
                                    fine_cell[0] = iv[0] * rr[0] + offset;
                                    fine_face[0] = fine_cell[0];
                                    if (side < 0) {
                                        fine_cell[1] = iv[1] * rr[1] - 1;
                                        fine_face[1] = iv[1] * rr[1];
                                    } else {
                                        fine_cell[1] = (iv[1] + 1) * rr[1];
                                        fine_face[1] = fine_cell[1];
                                    }
                                }
                                AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                                    fine_rows.box().contains(fine_cell),
                                    "CompositeGridTopology fine-row metadata "
                                    "does not contain a requested interface cell");
                                Long const fine_row = fine_rows(fine_cell);
                                if (fine_row >= 0) {
                                    m_connections.push_back(Connection{
                                        local_row, Long(-1), fine_row, level,
                                        direction, fine_face,
                                        fine_area / distance, true, true});
                                    ++fine_neighbor_count;
                                }
                            }
                        }
                        if (fine_neighbor_count > 0) {
                            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                                fine_neighbor_count ==
                                    m_ref_ratio[level][1 - direction],
                                "CompositeGridTopology found a partial "
                                "coarse-face refinement");
                            continue;
                        }

                        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                            level > 0,
                            "CompositeGridTopology could not resolve an "
                            "interior composite-grid neighbor");
                        IntVect const coarse_neighbor = amrex::coarsen(
                            neighbor, m_ref_ratio[level - 1]);
                        auto const& coarse_rows =
                            (*coarse_rows_on_fine[level - 1])[mfi];
                        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                            coarse_rows.box().contains(coarse_neighbor),
                            "CompositeGridTopology coarse-row metadata does "
                            "not contain a requested interface cell");
                        Long const coarse_row = coarse_rows(coarse_neighbor);
                        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                            coarse_row >= 0,
                            "CompositeGridTopology is missing a coarse row at "
                            "a coarse/fine interface");
                        Real const coarse_dx =
                            m_geom[level - 1].CellSize(direction);
                        Real const distance = Real(0.5) *
                            (dx[direction] + coarse_dx);
                        m_connections.push_back(Connection{
                            local_row, Long(-1), coarse_row, level, direction,
                            face, area / distance, false, true});
                    }
                }
            }
        }
    }
    AMREX_ALWAYS_ASSERT(
        std::all_of(cell_initialized.begin(), cell_initialized.end(),
                    [] (int value) { return value == 1; }));
}

void
CompositeGridTopology::buildPattern ()
{
    Long const nlocal = localRows();
    Vector<Vector<Long>> columns(nlocal);
    for (Long row = 0; row < nlocal; ++row) {
        columns[row].push_back(m_cells[row].global_row);
    }
    for (auto const& connection : m_connections) {
        columns[connection.local_row].push_back(connection.column);
    }

    m_row_offset.resize(nlocal + 1);
    m_row_offset[0] = 0;
    for (Long row = 0; row < nlocal; ++row) {
        auto& row_columns = columns[row];
        std::sort(row_columns.begin(), row_columns.end());
        row_columns.erase(std::unique(row_columns.begin(), row_columns.end()),
                          row_columns.end());
        for (Long column : row_columns) {
            AMREX_ALWAYS_ASSERT(column >= 0 && column < globalRows());
            m_col_index.push_back(column);
        }
        m_row_offset[row + 1] = static_cast<Long>(m_col_index.size());
    }

    for (Long row = 0; row < nlocal; ++row) {
        auto const first = m_col_index.begin() + m_row_offset[row];
        auto const last = m_col_index.begin() + m_row_offset[row + 1];
        auto const found = std::lower_bound(
            first, last, m_cells[row].global_row);
        AMREX_ALWAYS_ASSERT(found != last &&
                            *found == m_cells[row].global_row);
        m_cells[row].diagonal_slot =
            static_cast<Long>(found - m_col_index.begin());
    }
    for (auto& connection : m_connections) {
        auto const first =
            m_col_index.begin() + m_row_offset[connection.local_row];
        auto const last =
            m_col_index.begin() + m_row_offset[connection.local_row + 1];
        auto const found = std::lower_bound(first, last, connection.column);
        AMREX_ALWAYS_ASSERT(found != last && *found == connection.column);
        connection.matrix_slot =
            static_cast<Long>(found - m_col_index.begin());
    }
}

CompositeGridTopology::NumericalAssembly
CompositeGridTopology::assemble (
    Real ascalar, Real bscalar, Vector<MultiFab const*> const& acoef,
    Vector<Array<MultiFab const*, AMREX_SPACEDIM>> const& bcoef,
    Array<LinOpBCType, AMREX_SPACEDIM> const& lobc,
    Array<LinOpBCType, AMREX_SPACEDIM> const& hibc,
    BoundaryData const& boundary) const
{
    NumericalAssembly result;
    result.matrix.row_offset = m_row_offset;
    result.matrix.col_index = m_col_index;
    result.matrix.mat.resize(m_col_index.size(), Real(0));
    result.matrix.nnz = static_cast<Long>(m_col_index.size());
    result.boundary_rhs.resize(localRows(), Real(0));

    Vector<Array<std::unique_ptr<MultiFab>, AMREX_SPACEDIM>>
        fine_b_on_coarse(numLevels() - 1);
    for (int level = 0; level + 1 < numLevels(); ++level) {
        for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
            BoxArray face_layout = m_refined_coarse[level];
            face_layout.convert(IntVect::TheDimensionVector(direction));
            fine_b_on_coarse[level][direction] = std::make_unique<MultiFab>(
                face_layout, m_dmap[level], 1, 0, topology_host_info());
            fine_b_on_coarse[level][direction]->setVal(Real(0));
            fine_b_on_coarse[level][direction]->ParallelCopy(
                *bcoef[level + 1][direction], 0, 0, 1, IntVect(0),
                IntVect(0), m_geom[level + 1].periodicity());
        }
    }
    Gpu::streamSynchronize();

    Real minimum_diagonal = std::numeric_limits<Real>::max();
    for (Long row = 0; row < localRows(); ++row) {
        Cell const& cell = m_cells[row];
        Real const diagonal = ascalar *
            acoef[cell.level]->atLocalIdx(cell.local_grid)(cell.index) *
            cell.volume;
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            diagonal >= Real(0),
            "CompositeGridTopology assembled a negative reaction diagonal");
        result.matrix.mat[cell.diagonal_slot] += diagonal;
    }

    for (auto const& connection : m_connections) {
        MultiFab const& coefficient_field =
            connection.fine_coefficient_on_coarse_layout
                ? *fine_b_on_coarse[connection.level][connection.direction]
                : *bcoef[connection.level][connection.direction];
        Real const coefficient =
            bscalar *
            coefficient_field.atLocalIdx(
                m_cells[connection.local_row].local_grid)(connection.face) *
            connection.geometric_weight;
        AMREX_ALWAYS_ASSERT(coefficient >= Real(0));
        Cell const& cell = m_cells[connection.local_row];
        result.matrix.mat[cell.diagonal_slot] += coefficient;
        result.matrix.mat[connection.matrix_slot] -= coefficient;
        result.local_coarse_fine_connections += connection.coarse_fine;
    }

    bool const have_level_bc =
        static_cast<int>(boundary.level.size()) == numLevels();
    bool const have_robin =
        static_cast<int>(boundary.robin_a.size()) == numLevels();
    for (auto const& physical : m_physical_faces) {
        LinOpBCType const type = physical.side < 0
                                     ? lobc[physical.direction]
                                     : hibc[physical.direction];
        if (type == LinOpBCType::Neumann) {
            continue;
        }
        Cell const& cell = m_cells[physical.local_row];
        Real const k = bcoef[physical.level][physical.direction]
                           ->atLocalIdx(cell.local_grid)(physical.face);
        if (type == LinOpBCType::Dirichlet) {
            AMREX_ALWAYS_ASSERT(have_level_bc);
            Real const coefficient =
                bscalar * k * physical.area / physical.distance;
            result.matrix.mat[cell.diagonal_slot] += coefficient;
            result.boundary_rhs[physical.local_row] +=
                coefficient * boundary.level[physical.level]
                                  ->atLocalIdx(cell.local_grid)(
                                      physical.exterior);
        } else {
            AMREX_ALWAYS_ASSERT(is_robin(type) && have_robin);
            Real const aa = boundary.robin_a[physical.level]
                                ->atLocalIdx(cell.local_grid)(physical.cell);
            Real const bb = boundary.robin_b[physical.level]
                                ->atLocalIdx(cell.local_grid)(physical.cell);
            Real const ff = boundary.robin_f[physical.level]
                                ->atLocalIdx(cell.local_grid)(physical.cell);
            AMREX_ALWAYS_ASSERT(aa >= Real(0) && bb >= Real(0) &&
                                aa + bb > Real(0));
            Real const scale = bscalar * k * physical.area /
                               (bb + aa * physical.distance);
            result.matrix.mat[cell.diagonal_slot] += scale * aa;
            result.boundary_rhs[physical.local_row] += scale * ff;
        }
    }

    Real maximum_offdiag = std::numeric_limits<Real>::lowest();
    for (Long row = 0; row < localRows(); ++row) {
        Cell const& cell = m_cells[row];
        Real const diagonal = result.matrix.mat[cell.diagonal_slot];
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            diagonal > Real(0),
            "CompositeGridTopology assembled a nonpositive diagonal");
        minimum_diagonal = amrex::min(minimum_diagonal, diagonal);
        for (Long slot = m_row_offset[row]; slot < m_row_offset[row + 1];
             ++slot) {
            if (slot != cell.diagonal_slot) {
                Real const value = result.matrix.mat[slot];
                maximum_offdiag = amrex::max(maximum_offdiag, value);
                AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                    value <= Real(1.e-14),
                    "CompositeGridTopology assembled a positive off-diagonal");
            }
        }
    }
    result.local_minimum_diagonal = minimum_diagonal;
    result.local_maximum_off_diagonal = maximum_offdiag;
    return result;
}

int
CompositeGridTopology::numLevels () const noexcept
{
    return static_cast<int>(m_geom.size());
}

Vector<Geometry> const&
CompositeGridTopology::geometry () const noexcept
{
    return m_geom;
}

Vector<BoxArray> const&
CompositeGridTopology::grids () const noexcept
{
    return m_grids;
}

Vector<IntVect> const&
CompositeGridTopology::refRatio () const noexcept
{
    return m_ref_ratio;
}

AlgPartition const&
CompositeGridTopology::partition () const noexcept
{
    return m_partition;
}

Vector<CompositeGridTopology::Cell> const&
CompositeGridTopology::cells () const noexcept
{
    return m_cells;
}

Long
CompositeGridTopology::globalRows () const noexcept
{
    return m_partition.numGlobalRows();
}

Long
CompositeGridTopology::localRows () const noexcept
{
    int const myproc = ParallelDescriptor::MyProc();
    return m_partition[myproc + 1] - m_partition[myproc];
}

} // namespace fld_test
