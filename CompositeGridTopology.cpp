#include "CompositeGridTopology.H"

#include <AMReX.H>
#include <AMReX_BaseFab.H>
#include <AMReX_BLProfiler.H>
#include <AMReX_BoxIterator.H>
#include <AMReX_FabArray.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParallelContext.H>
#include <AMReX_ParmParse.H>

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

std::unique_ptr<MultiFab>
make_host_buffer (MultiFab const& source, int nghost)
{
    return std::make_unique<MultiFab>(
        source.boxArray(), source.DistributionMap(), source.nComp(), nghost,
        topology_host_info());
}

void
stage_host (MultiFab& destination, MultiFab const& source, int nghost,
            Periodicity const& periodicity, int source_nghost = 0)
{
    destination.setVal(Real(0));
    destination.ParallelCopy(source, 0, 0, source.nComp(),
                             IntVect(source_nghost), IntVect(nghost),
                             periodicity);
}

void
verify_face_transfer (
    Array<MultiFab*, AMREX_SPACEDIM> const& dst,
    Array<MultiFab const*, AMREX_SPACEDIM> const& src,
    Periodicity const& periodicity, char const* phase)
{
    bool matches = true;
    for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
        auto reference = make_host_buffer(*dst[direction], 0);
        stage_host(*reference, *src[direction], 0, periodicity);
        Gpu::streamSynchronize();
        for (MFIter mfi(*reference); mfi.isValid(); ++mfi) {
            auto const& actual = (*dst[direction])[mfi];
            auto const& expected = (*reference)[mfi];
            for (BoxIterator bit(mfi.validbox()); bit.ok(); ++bit) {
                matches = matches && actual(bit(), 0) == expected(bit(), 0);
            }
        }
    }
    ParallelDescriptor::ReduceBoolAnd(matches);
    if (!matches) {
        amrex::Print() << "Packed AMR " << phase
                       << " face transfer differs from ParallelCopy\n";
        ParallelDescriptor::Abort(1, false);
    }
}

} // namespace

void
CompositeGridTopology::FaceTransfer::define (
    Array<MultiFab*, AMREX_SPACEDIM> const& dst,
    Array<MultiFab const*, AMREX_SPACEDIM> const& src,
    Periodicity const& periodicity)
{
    BL_PROFILE(assembly ? "FLD::assembly::fine_b_plan"
                        : "FLD::residual::fine_b_plan");
    AMREX_ALWAYS_ASSERT(!active && local.empty() && sends.empty() && receives.empty());
    for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
        auto const& cpc = dst[direction]->getCPC(
            IntVect(0), *src[direction], IntVect(0), periodicity);
        for (auto const& tag : *cpc.m_LocTags) {
            local.push_back({direction, tag, 0});
        }
        separate_sends += static_cast<Long>(cpc.m_SndTags->size());
        auto append = [direction] (auto& peers, auto const& tags) {
            for (auto const& [rank, copies] : tags) {
                auto& peer = peers[rank];
                for (auto const& tag : copies) {
                    std::size_t const offset = peer.data.size();
                    peer.tags.push_back({direction, tag, offset});
                    peer.data.resize(offset + static_cast<std::size_t>(tag.sbox.numPts()));
                }
            }
        };
        append(sends, *cpc.m_SndTags);
        append(receives, *cpc.m_RcvTags);
    }
}

void
CompositeGridTopology::FaceTransfer::start (
    Array<MultiFab*, AMREX_SPACEDIM> const& dst,
    Array<MultiFab const*, AMREX_SPACEDIM> const& src)
{
    AMREX_ALWAYS_ASSERT(!active);
    active = true;
    // Staged coefficient fields are host-readable even in a GPU build.
    Gpu::streamSynchronize();
#ifdef AMREX_USE_MPI
    int const tag = ParallelDescriptor::SeqNum(); // All ranks, including idle ranks.
    auto const comm = ParallelContext::CommunicatorSub();
    receive_requests.clear();
    send_requests.clear();
    auto post = [tag, comm] (auto& peers, auto& requests, bool receive) {
        for (auto& [rank, peer] : peers) {
            // MPI's count is an int. Ordinary payloads use one message per
            // peer; exceptionally large payloads are split without padding.
            for (std::size_t offset = 0; offset < peer.data.size();) {
                std::size_t const count = std::min(
                    peer.data.size() - offset,
                    static_cast<std::size_t>(std::numeric_limits<int>::max()));
                int const local_rank = ParallelContext::global_to_local_rank(rank);
                auto message = receive
                    ? ParallelDescriptor::Arecv(peer.data.data() + offset, count,
                                                local_rank, tag, comm)
                    : ParallelDescriptor::Asend(peer.data.data() + offset, count,
                                                local_rank, tag, comm);
                requests.push_back(message.req());
                offset += count;
            }
        }
    };
    post(receives, receive_requests, true);
#endif
    {
        BL_PROFILE(assembly ? "FLD::assembly::fine_b_pack"
                            : "FLD::residual::fine_b_pack");
        for (auto& [rank, peer] : sends) {
            amrex::ignore_unused(rank);
            for (auto const& entry : peer.tags) {
                auto const& tag = entry.copy;
                (*src[entry.direction])[tag.srcIndex].copyToMem<RunOn::Host>(
                    tag.sbox, 0, 1, peer.data.data() + entry.offset);
            }
        }
    }
#ifdef AMREX_USE_MPI
    post(sends, send_requests, false);
#endif
    for (auto const& entry : local) {
        auto const& tag = entry.copy;
        (*dst[entry.direction])[tag.dstIndex].copy<RunOn::Host>(
            (*src[entry.direction])[tag.srcIndex], tag.sbox, 0, tag.dbox, 0, 1);
    }
}

void
CompositeGridTopology::FaceTransfer::finish (
    Array<MultiFab*, AMREX_SPACEDIM> const& dst)
{
    AMREX_ALWAYS_ASSERT(active);
#ifdef AMREX_USE_MPI
    {
        BL_PROFILE(assembly ? "FLD::assembly::fine_b_recv_wait"
                            : "FLD::residual::fine_b_recv_wait");
        Vector<MPI_Status> status(receive_requests.size());
        if (!receive_requests.empty()) {
            ParallelDescriptor::Waitall(receive_requests, status);
        }
    }
#endif
    {
        BL_PROFILE(assembly ? "FLD::assembly::fine_b_unpack"
                            : "FLD::residual::fine_b_unpack");
        for (auto const& [rank, peer] : receives) {
            amrex::ignore_unused(rank);
            for (auto const& entry : peer.tags) {
                auto const& tag = entry.copy;
                (*dst[entry.direction])[tag.dstIndex].copyFromMem<RunOn::Host>(
                    tag.dbox, 0, 1, peer.data.data() + entry.offset);
            }
        }
    }
#ifdef AMREX_USE_MPI
    {
        BL_PROFILE(assembly ? "FLD::assembly::fine_b_send_wait"
                            : "FLD::residual::fine_b_send_wait");
        Vector<MPI_Status> status(send_requests.size());
        if (!send_requests.empty()) {
            ParallelDescriptor::Waitall(send_requests, status);
        }
    }
#endif
    active = false;
}

CompositeGridTopology::CompositeGridTopology (
    Vector<Geometry> geom, Vector<BoxArray> grids,
    Vector<DistributionMapping> dmap)
    : m_geom(std::move(geom)), m_grids(std::move(grids)),
      m_dmap(std::move(dmap))
{
    BL_PROFILE("FLD::composite_topology");
    static_assert(AMREX_SPACEDIM == 2 || AMREX_SPACEDIM == 3);
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
                                direction, face, neighbor,
                                area / dx[direction], false,
                                false});
                            continue;
                        }

                        int fine_neighbor_count = 0;
                        int expected_fine_neighbor_count = 0;
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
                            int const transverse_a =
                                (direction + 1) % AMREX_SPACEDIM;
                            int const transverse_b =
                                (direction + 2) % AMREX_SPACEDIM;
                            int const count_b = AMREX_SPACEDIM == 3
                                                    ? rr[transverse_b] : 1;
                            expected_fine_neighbor_count =
                                rr[transverse_a] * count_b;
                            for (int offset_a = 0;
                                 offset_a < rr[transverse_a]; ++offset_a) {
                                for (int offset_b = 0; offset_b < count_b;
                                     ++offset_b) {
                                    IntVect fine_cell = IntVect::TheZeroVector();
                                    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                                        fine_cell[d] = iv[d] * rr[d];
                                    }
                                    fine_cell[transverse_a] += offset_a;
                                    if (AMREX_SPACEDIM == 3) {
                                        fine_cell[transverse_b] += offset_b;
                                    }
                                    fine_cell[direction] = side < 0
                                        ? iv[direction] * rr[direction] - 1
                                        : (iv[direction] + 1) * rr[direction];
                                    IntVect fine_face = fine_cell;
                                    if (side < 0) {
                                        fine_face[direction] += 1;
                                    }
                                    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                                        fine_rows.box().contains(fine_cell),
                                        "CompositeGridTopology fine-row metadata "
                                        "does not contain a requested interface cell");
                                    Long const fine_row = fine_rows(fine_cell);
                                    if (fine_row >= 0) {
                                        m_connections.push_back(Connection{
                                            local_row, Long(-1), fine_row, level,
                                            direction, fine_face, fine_cell,
                                            fine_area / distance, true, true});
                                        ++fine_neighbor_count;
                                    }
                                }
                            }
                        }
                        if (fine_neighbor_count > 0) {
                            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                                fine_neighbor_count ==
                                    expected_fine_neighbor_count,
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
                            face, coarse_neighbor, area / distance, false,
                            true});
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
    BL_PROFILE("FLD::composite_assembly");
    NumericalAssembly result;
    result.matrix.row_offset = m_row_offset;
    result.matrix.col_index = m_col_index;
    result.matrix.mat.resize(m_col_index.size(), Real(0));
    result.matrix.nnz = static_cast<Long>(m_col_index.size());
    result.boundary_rhs.resize(localRows(), Real(0));

    if (!m_assembly_workspace) {
        auto workspace = std::make_unique<AssemblyWorkspace>();
        workspace->fine_b_on_coarse.resize(numLevels() - 1);
        workspace->fine_b_transfer.resize(numLevels() - 1);
        ParmParse pp("mlabeclap_amg");
        pp.query("verify_face_transfers", workspace->verify_face_transfers);
        bool measure_messages = false;
        pp.query("measure_assembly_messages", measure_messages);
        Long counts[3] = {0, 0, 0};
        for (int level = 0; level + 1 < numLevels(); ++level) {
            Array<MultiFab*, AMREX_SPACEDIM> dst;
            for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
                BoxArray face_layout = m_refined_coarse[level];
                face_layout.convert(IntVect::TheDimensionVector(direction));
                auto& field = workspace->fine_b_on_coarse[level][direction];
                field = std::make_unique<MultiFab>(
                    face_layout, m_dmap[level], 1, 0, topology_host_info());
                dst[direction] = field.get();
            }
            auto& transfer = workspace->fine_b_transfer[level];
            transfer.assembly = true;
            transfer.define(dst, bcoef[level + 1], m_geom[level + 1].periodicity());
            counts[0] += transfer.separate_sends;
            for (auto const& [rank, peer] : transfer.sends) {
                amrex::ignore_unused(rank);
                if (!peer.data.empty()) {
                    counts[1] += static_cast<Long>((peer.data.size() - 1) /
                        std::numeric_limits<int>::max() + 1);
                }
                counts[2] += static_cast<Long>(peer.data.size() * sizeof(Real));
            }
        }
        if (measure_messages) {
            ParallelDescriptor::ReduceLongSum(counts, 3);
            amrex::Print() << "FLD assembly face transfers per assembly: separate sends="
                           << counts[0] << ", packed sends=" << counts[1]
                           << ", payload bytes=" << counts[2] << '\n';
        }
        m_assembly_workspace = std::move(workspace);
    }
    auto& workspace = *m_assembly_workspace;
    auto& fine_b_on_coarse = workspace.fine_b_on_coarse;
    for (int level = 0; level + 1 < numLevels(); ++level) {
        BL_PROFILE("FLD::assembly::fine_b_start");
        Array<MultiFab*, AMREX_SPACEDIM> dst;
        for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
            dst[direction] = fine_b_on_coarse[level][direction].get();
            // Uncovered destination faces must stay zero, including after
            // repeated assembly with changed coefficient values.
            dst[direction]->setVal(Real(0));
        }
        workspace.fine_b_transfer[level].start(dst, bcoef[level + 1]);
    }
    for (int level = 0; level + 1 < numLevels(); ++level) {
        Array<MultiFab*, AMREX_SPACEDIM> dst;
        for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
            dst[direction] = fine_b_on_coarse[level][direction].get();
        }
        {
            BL_PROFILE("FLD::assembly::fine_b_finish");
            workspace.fine_b_transfer[level].finish(dst);
        }
        if (workspace.verify_face_transfers) {
            verify_face_transfer(dst, bcoef[level + 1],
                                 m_geom[level + 1].periodicity(), "assembly");
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

Gpu::PinnedVector<Real> const&
CompositeGridTopology::residualWithoutAssembly (
    Real ascalar, Real bscalar, Vector<MultiFab const*> const& state,
    Vector<MultiFab const*> const& rhs,
    Vector<MultiFab const*> const& acoef,
    Vector<Array<MultiFab const*, AMREX_SPACEDIM>> const& bcoef,
    Array<LinOpBCType, AMREX_SPACEDIM> const& lobc,
    Array<LinOpBCType, AMREX_SPACEDIM> const& hibc,
    BoundaryData const& boundary)
{
    BL_PROFILE("FLD::composite_residual_without_assembly");
    int const nlevels = numLevels();
    AMREX_ALWAYS_ASSERT(static_cast<int>(state.size()) == nlevels &&
                        static_cast<int>(rhs.size()) == nlevels &&
                        static_cast<int>(acoef.size()) == nlevels &&
                        static_cast<int>(bcoef.size()) == nlevels);

    if (!m_residual_workspace) {
        auto workspace = std::make_unique<ResidualWorkspace>();
        workspace->host_state.resize(nlevels);
        workspace->host_rhs.resize(nlevels);
        workspace->host_acoef.resize(nlevels);
        workspace->host_level_bc.resize(nlevels);
        workspace->host_robin_a.resize(nlevels);
        workspace->host_robin_b.resize(nlevels);
        workspace->host_robin_f.resize(nlevels);
        workspace->host_bcoef.resize(nlevels);
        workspace->fine_state_on_coarse.resize(nlevels - 1);
        workspace->coarse_state_on_fine.resize(nlevels - 1);
        workspace->fine_b_on_coarse.resize(nlevels - 1);
        workspace->fine_b_transfer.resize(nlevels - 1);
        ParmParse pp("mlabeclap_amg");
        pp.query("measure_residual_messages", workspace->measure_residual_messages);
        pp.query("verify_face_transfers", workspace->verify_face_transfers);
        workspace->result.resize(localRows());
        for (int level = 0; level < nlevels; ++level) {
            workspace->host_state[level] = make_host_buffer(*state[level], 1);
#ifdef AMREX_USE_GPU
            workspace->host_rhs[level] = make_host_buffer(*rhs[level], 0);
            workspace->host_acoef[level] = make_host_buffer(*acoef[level], 0);
            for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
                workspace->host_bcoef[level][direction] =
                    make_host_buffer(*bcoef[level][direction], 0);
            }
#endif
            if (level + 1 < nlevels) {
                workspace->fine_state_on_coarse[level] =
                    std::make_unique<MultiFab>(
                        m_refined_coarse[level], m_dmap[level], 1, 1,
                        topology_host_info());
                BoxArray coarsened_fine = m_grids[level + 1];
                coarsened_fine.coarsen(m_ref_ratio[level]);
                workspace->coarse_state_on_fine[level] =
                    std::make_unique<MultiFab>(
                        coarsened_fine, m_dmap[level + 1], 1, 2,
                        topology_host_info());
                for (int direction = 0; direction < AMREX_SPACEDIM;
                     ++direction) {
                    BoxArray face_layout = m_refined_coarse[level];
                    face_layout.convert(
                        IntVect::TheDimensionVector(direction));
                    workspace->fine_b_on_coarse[level][direction] =
                        std::make_unique<MultiFab>(
                            face_layout, m_dmap[level], 1, 0,
                            topology_host_info());
                }
            }
        }
        m_residual_workspace = std::move(workspace);
    }
    auto& workspace = *m_residual_workspace;
    auto& host_state = workspace.host_state;
    auto& host_rhs = workspace.host_rhs;
    auto& host_acoef = workspace.host_acoef;
    auto& host_level_bc = workspace.host_level_bc;
    auto& host_robin_a = workspace.host_robin_a;
    auto& host_robin_b = workspace.host_robin_b;
    auto& host_robin_f = workspace.host_robin_f;
    auto& host_bcoef = workspace.host_bcoef;
    auto& fine_state_on_coarse = workspace.fine_state_on_coarse;
    auto& coarse_state_on_fine = workspace.coarse_state_on_fine;
    auto& fine_b_on_coarse = workspace.fine_b_on_coarse;
    bool const have_level_bc =
        static_cast<int>(boundary.level.size()) == nlevels;
    bool const have_robin =
        static_cast<int>(boundary.robin_a.size()) == nlevels;
    for (int level = 0; level < nlevels; ++level) {
        auto const periodicity = m_geom[level].periodicity();
        {
            BL_PROFILE("FLD::residual::stage_state");
            stage_host(*host_state[level], *state[level], 1, periodicity);
        }
#ifdef AMREX_USE_GPU
        stage_host(*host_rhs[level], *rhs[level], 0, periodicity);
        stage_host(*host_acoef[level], *acoef[level], 0, periodicity);
        for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
            stage_host(*host_bcoef[level][direction],
                       *bcoef[level][direction], 0, periodicity);
        }
        if (have_level_bc) {
            if (!host_level_bc[level]) {
                host_level_bc[level] =
                    make_host_buffer(*boundary.level[level], 1);
            }
            stage_host(*host_level_bc[level], *boundary.level[level], 1,
                       periodicity, 1);
        }
        if (have_robin) {
            if (!host_robin_a[level]) {
                host_robin_a[level] =
                    make_host_buffer(*boundary.robin_a[level], 0);
                host_robin_b[level] =
                    make_host_buffer(*boundary.robin_b[level], 0);
                host_robin_f[level] =
                    make_host_buffer(*boundary.robin_f[level], 0);
            }
            stage_host(*host_robin_a[level], *boundary.robin_a[level], 0,
                       periodicity);
            stage_host(*host_robin_b[level], *boundary.robin_b[level], 0,
                       periodicity);
            stage_host(*host_robin_f[level], *boundary.robin_f[level], 0,
                       periodicity);
        }
#endif
    }

#ifdef AMREX_USE_GPU
    Vector<MultiFab const*> rhs_values(nlevels);
    Vector<MultiFab const*> acoef_values(nlevels);
    Vector<MultiFab const*> level_bc_values(nlevels);
    Vector<MultiFab const*> robin_a_values(nlevels);
    Vector<MultiFab const*> robin_b_values(nlevels);
    Vector<MultiFab const*> robin_f_values(nlevels);
    Vector<Array<MultiFab const*, AMREX_SPACEDIM>> bcoef_values(nlevels);
    for (int level = 0; level < nlevels; ++level) {
        rhs_values[level] = host_rhs[level].get();
        acoef_values[level] = host_acoef[level].get();
        for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
            bcoef_values[level][direction] =
                host_bcoef[level][direction].get();
        }
        if (have_level_bc) {
            level_bc_values[level] = host_level_bc[level].get();
        }
        if (have_robin) {
            robin_a_values[level] = host_robin_a[level].get();
            robin_b_values[level] = host_robin_b[level].get();
            robin_f_values[level] = host_robin_f[level].get();
        }
    }
#else
    // CPU MultiFabs are host-readable; only state needs a copied halo.
    auto const& rhs_values = rhs;
    auto const& acoef_values = acoef;
    auto const& bcoef_values = bcoef;
    auto const& level_bc_values = boundary.level;
    auto const& robin_a_values = boundary.robin_a;
    auto const& robin_b_values = boundary.robin_b;
    auto const& robin_f_values = boundary.robin_f;
#endif

    if (!workspace.face_plan_ready) {
        Long counts[3] = {0, 0, 0};
        for (int level = 0; level + 1 < nlevels; ++level) {
            Array<MultiFab*, AMREX_SPACEDIM> dst;
            for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
                dst[direction] = fine_b_on_coarse[level][direction].get();
            }
            auto& transfer = workspace.fine_b_transfer[level];
            transfer.define(dst, bcoef_values[level + 1], m_geom[level + 1].periodicity());
            counts[0] += transfer.separate_sends;
            for (auto const& [rank, peer] : transfer.sends) {
                amrex::ignore_unused(rank);
                if (!peer.data.empty()) {
                    counts[1] += static_cast<Long>((peer.data.size() - 1) /
                        std::numeric_limits<int>::max() + 1);
                }
                counts[2] += static_cast<Long>(peer.data.size() * sizeof(Real));
            }
        }
        if (workspace.measure_residual_messages) {
            ParallelDescriptor::ReduceLongSum(counts, 3);
            amrex::Print() << "FLD residual face transfers per evaluation: separate sends="
                           << counts[0] << ", packed sends=" << counts[1]
                           << ", payload bytes=" << counts[2] << '\n';
        }
        workspace.face_plan_ready = true;
    }

    // Start independent coarse/fine transfers before evaluating local rows.
    for (int level = 0; level + 1 < nlevels; ++level) {
        fine_state_on_coarse[level]->setVal(Real(0));
        {
            BL_PROFILE("FLD::residual::fine_state_start");
            fine_state_on_coarse[level]->ParallelCopy_nowait(
                *host_state[level + 1], 0, 0, 1, IntVect(0), IntVect(1),
                m_geom[level + 1].periodicity());
        }

        coarse_state_on_fine[level]->setVal(Real(0));
        {
            BL_PROFILE("FLD::residual::coarse_state_start");
            coarse_state_on_fine[level]->ParallelCopy_nowait(
                *host_state[level], 0, 0, 1, IntVect(0), IntVect(2),
                m_geom[level].periodicity());
        }

        for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
            fine_b_on_coarse[level][direction]->setVal(Real(0));
        }
        {
            BL_PROFILE("FLD::residual::fine_b_start");
            Array<MultiFab*, AMREX_SPACEDIM> dst;
            for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
                dst[direction] = fine_b_on_coarse[level][direction].get();
            }
            workspace.fine_b_transfer[level].start(dst, bcoef_values[level + 1]);
        }
    }

    auto& result = workspace.result;
    {
        BL_PROFILE("FLD::residual::local_rows");
        for (Long row = 0; row < localRows(); ++row) {
            Cell const& cell = m_cells[row];
            Real const value =
                host_state[cell.level]->atLocalIdx(cell.local_grid)(cell.index);
            result[row] = ascalar *
                acoef_values[cell.level]->atLocalIdx(cell.local_grid)(cell.index) *
                value * cell.volume -
                rhs_values[cell.level]->atLocalIdx(cell.local_grid)(cell.index) *
                    cell.volume;
        }
    }

    for (int level = 0; level + 1 < nlevels; ++level) {
        {
            BL_PROFILE("FLD::residual::fine_state_finish");
            fine_state_on_coarse[level]->ParallelCopy_finish();
        }
        {
            BL_PROFILE("FLD::residual::coarse_state_finish");
            coarse_state_on_fine[level]->ParallelCopy_finish();
        }
        {
            BL_PROFILE("FLD::residual::fine_b_finish");
            Array<MultiFab*, AMREX_SPACEDIM> dst;
            for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
                dst[direction] = fine_b_on_coarse[level][direction].get();
            }
            workspace.fine_b_transfer[level].finish(dst);
        }
        if (workspace.verify_face_transfers) {
            Array<MultiFab*, AMREX_SPACEDIM> dst;
            for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
                dst[direction] = fine_b_on_coarse[level][direction].get();
            }
            verify_face_transfer(dst, bcoef_values[level + 1],
                                 m_geom[level + 1].periodicity(), "residual");
        }
    }
    Gpu::streamSynchronize();
    for (auto const& connection : m_connections) {
        Cell const& cell = m_cells[connection.local_row];
        MultiFab const& coefficient_field =
            connection.fine_coefficient_on_coarse_layout
                ? *fine_b_on_coarse[connection.level][connection.direction]
                : *bcoef_values[connection.level][connection.direction];
        Real const coefficient = bscalar *
            coefficient_field.atLocalIdx(cell.local_grid)(connection.face) *
            connection.geometric_weight;
        Real const source =
            host_state[cell.level]->atLocalIdx(cell.local_grid)(cell.index);
        Real target;
        if (!connection.coarse_fine) {
            target = host_state[cell.level]
                         ->atLocalIdx(cell.local_grid)(connection.target);
        } else if (connection.fine_coefficient_on_coarse_layout) {
            target = fine_state_on_coarse[connection.level]
                         ->atLocalIdx(cell.local_grid)(connection.target);
        } else {
            target = coarse_state_on_fine[connection.level - 1]
                         ->atLocalIdx(cell.local_grid)(connection.target);
        }
        result[connection.local_row] += coefficient * (source - target);
    }

    for (auto const& physical : m_physical_faces) {
        LinOpBCType const type = physical.side < 0
                                     ? lobc[physical.direction]
                                     : hibc[physical.direction];
        if (type == LinOpBCType::Neumann) { continue; }
        Cell const& cell = m_cells[physical.local_row];
        Real const k = bcoef_values[physical.level][physical.direction]
                           ->atLocalIdx(cell.local_grid)(physical.face);
        Real const value =
            host_state[cell.level]->atLocalIdx(cell.local_grid)(cell.index);
        if (type == LinOpBCType::Dirichlet) {
            AMREX_ALWAYS_ASSERT(have_level_bc);
            Real const coefficient =
                bscalar * k * physical.area / physical.distance;
            Real const exterior = level_bc_values[physical.level]
                                      ->atLocalIdx(cell.local_grid)(
                                          physical.exterior);
            result[physical.local_row] += coefficient * (value - exterior);
        } else {
            AMREX_ALWAYS_ASSERT(is_robin(type) && have_robin);
            Real const aa = robin_a_values[physical.level]
                                ->atLocalIdx(cell.local_grid)(physical.cell);
            Real const bb = robin_b_values[physical.level]
                                ->atLocalIdx(cell.local_grid)(physical.cell);
            Real const ff = robin_f_values[physical.level]
                                ->atLocalIdx(cell.local_grid)(physical.cell);
            AMREX_ALWAYS_ASSERT(aa >= Real(0) && bb >= Real(0) &&
                                aa + bb > Real(0));
            Real const scale = bscalar * k * physical.area /
                               (bb + aa * physical.distance);
            result[physical.local_row] += scale * (aa * value - ff);
        }
    }
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

Vector<DistributionMapping> const&
CompositeGridTopology::dmap () const noexcept
{
    return m_dmap;
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
