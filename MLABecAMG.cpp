#include "MLABecAMG.H"

#include <AMReX.H>
#include <AMReX_AlgPartition.H>
#include <AMReX_AlgVecUtil.H>
#include <AMReX_Arena.H>
#include <AMReX_BaseFab.H>
#include <AMReX_BoxIterator.H>
#include <AMReX_FabArray.H>
#include <AMReX_GMRES_MV.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_SpMV.H>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <utility>

namespace fld_test
{

using namespace amrex;

namespace
{

using Entry = std::pair<Long, Real>;
using Entries = Vector<Entry>;
using LongFabArray = FabArray<BaseFab<Long>>;

MFInfo
host_info ()
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

std::unique_ptr<MultiFab>
stage_multifab (MultiFab const& source, int nghost,
                Periodicity const& periodicity)
{
    AMREX_ALWAYS_ASSERT(nghost >= 0 && nghost <= source.nGrow());
    auto result = std::make_unique<MultiFab>(
        source.boxArray(), source.DistributionMap(), source.nComp(), nghost,
        host_info());
    result->setVal(Real(0));
    result->ParallelCopy(source, 0, 0, source.nComp(), IntVect(nghost),
                         IntVect(nghost), periodicity);
    Gpu::streamSynchronize();
    return result;
}

void
add_entry (Entries& entries, Long column, Real value)
{
    AMREX_ALWAYS_ASSERT(column >= 0);
    if (value != Real(0)) {
        entries.emplace_back(column, value);
    }
}

bool
is_robin (LinOpBCType type) noexcept
{
    return type == LinOpBCType::Robin || type == LinOpBCType::Marshak;
}

Real
cell_volume (Geometry const& geometry) noexcept
{
    auto const dx = geometry.CellSizeArray();
    return AMREX_D_TERM(dx[0], *dx[1], *dx[2]);
}

} // namespace

struct MLABecLapAMG::Impl
{
    Impl (Vector<Geometry> a_geom, Vector<BoxArray> a_grids,
          Vector<DistributionMapping> a_dmap, AMG<Real>::Options a_options,
          std::string prefix)
        : geom(std::move(a_geom)), grids(std::move(a_grids)),
          dmap(std::move(a_dmap)), options(std::move(a_options)),
          parmparse_prefix(std::move(prefix))
    {
        validate_hierarchy();
        ParmParse pp(parmparse_prefix);
        pp.query("verbose", verbose);
        pp.query("max_iter", max_iter);
        pp.query("restart_length", restart_length);
        AMREX_ALWAYS_ASSERT(max_iter > 0 && restart_length > 0);
        build_row_numbering();
    }

    void validate_hierarchy ()
    {
        nlevels = static_cast<int>(geom.size());
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            nlevels > 0 && static_cast<int>(grids.size()) == nlevels &&
                static_cast<int>(dmap.size()) == nlevels,
            "MLABecLapAMG requires matching, nonempty hierarchy vectors");
        ref_ratio.resize(nlevels - 1);
        for (int level = 0; level < nlevels; ++level) {
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                grids[level].ixType().cellCentered(),
                "MLABecLapAMG supports only cell-centered grids");
            AMREX_ALWAYS_ASSERT(
                geom[level].Domain().contains(grids[level].minimalBox()));
            if (level + 1 < nlevels) {
                ref_ratio[level] = geom[level + 1].Domain().length() /
                                   geom[level].Domain().length();
                AMREX_ALWAYS_ASSERT(ref_ratio[level].allGT(1));
                AMREX_ALWAYS_ASSERT(
                    geom[level + 1].Domain() ==
                    amrex::refine(geom[level].Domain(), ref_ratio[level]));
            }
        }
    }

    void build_row_numbering ()
    {
        active.resize(nlevels);
        row_ids.resize(nlevels);
        Long local_count = 0;
        for (int level = 0; level < nlevels; ++level) {
            if (level + 1 < nlevels) {
                active[level] = amrex::makeFineMask(
                    grids[level], dmap[level], IntVect(2), grids[level + 1],
                    ref_ratio[level], geom[level].periodicity(), 1, 0,
                    host_info());
            } else {
                active[level].define(grids[level], dmap[level], 1, 2,
                                     host_info());
                active[level].setVal(1);
            }
            local_count += active[level].sum(0, 0, true);
        }
        partition.define(make_partition_rows(local_count));
        int const myproc = ParallelDescriptor::MyProc();
        Long next_row = partition[myproc];
        for (int level = 0; level < nlevels; ++level) {
            row_ids[level] = std::make_unique<LongFabArray>(
                grids[level], dmap[level], 1, 2, host_info());
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
            row_ids[level]->FillBoundary(geom[level].periodicity());
        }
        AMREX_ALWAYS_ASSERT(next_row == partition[myproc + 1]);
        assembly.local_rows = local_count;
        assembly.global_rows = partition.numGlobalRows();
    }

    void validate_setup_inputs (
        Vector<MultiFab const*> const& acoef,
        Vector<Array<MultiFab const*, AMREX_SPACEDIM>> const& bcoef,
        Array<LinOpBCType, AMREX_SPACEDIM> const& lobc,
        Array<LinOpBCType, AMREX_SPACEDIM> const& hibc,
        Vector<MultiFab const*> const& level_bc,
        RobinBCData const& robin) const
    {
        AMREX_ALWAYS_ASSERT(static_cast<int>(acoef.size()) == nlevels);
        AMREX_ALWAYS_ASSERT(static_cast<int>(bcoef.size()) == nlevels);
        for (int level = 0; level < nlevels; ++level) {
            AMREX_ALWAYS_ASSERT(acoef[level] != nullptr);
            AMREX_ALWAYS_ASSERT(acoef[level]->boxArray() == grids[level]);
            AMREX_ALWAYS_ASSERT(acoef[level]->nComp() == 1);
            for (int direction = 0; direction < AMREX_SPACEDIM;
                 ++direction) {
                AMREX_ALWAYS_ASSERT(bcoef[level][direction] != nullptr);
                AMREX_ALWAYS_ASSERT(bcoef[level][direction]->nComp() == 1);
                AMREX_ALWAYS_ASSERT(
                    bcoef[level][direction]->ixType().nodeCentered(direction));
            }
        }
        bool needs_dirichlet = false;
        bool needs_robin = false;
        for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
            if (geom[0].isPeriodic(direction)) {
                AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                    lobc[direction] == LinOpBCType::Periodic &&
                        hibc[direction] == LinOpBCType::Periodic,
                    "Periodic geometry requires periodic MLABecLapAMG BCs");
                continue;
            }
            for (LinOpBCType type : {lobc[direction], hibc[direction]}) {
                bool const supported = type == LinOpBCType::Dirichlet ||
                                       type == LinOpBCType::Neumann ||
                                       is_robin(type);
                AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                    supported,
                    "MLABecLapAMG supports Dirichlet, homogeneous Neumann, "
                    "Robin/Marshak, and periodic boundaries only");
                needs_dirichlet = needs_dirichlet ||
                                  type == LinOpBCType::Dirichlet;
                needs_robin = needs_robin || is_robin(type);
            }
        }
        if (needs_dirichlet) {
            AMREX_ALWAYS_ASSERT(static_cast<int>(level_bc.size()) == nlevels);
            for (int level = 0; level < nlevels; ++level) {
                AMREX_ALWAYS_ASSERT(level_bc[level] != nullptr &&
                                    level_bc[level]->nComp() == 1 &&
                                    level_bc[level]->nGrow() >= 1);
            }
        }
        if (needs_robin) {
            AMREX_ALWAYS_ASSERT(static_cast<int>(robin.a.size()) == nlevels &&
                                static_cast<int>(robin.b.size()) == nlevels &&
                                static_cast<int>(robin.f.size()) == nlevels);
            for (int level = 0; level < nlevels; ++level) {
                AMREX_ALWAYS_ASSERT(robin.a[level] != nullptr &&
                                    robin.b[level] != nullptr &&
                                    robin.f[level] != nullptr);
                AMREX_ALWAYS_ASSERT(robin.a[level]->nComp() == 1 &&
                                    robin.b[level]->nComp() == 1 &&
                                    robin.f[level]->nComp() == 1);
            }
        }
    }

    void assemble (
        Real ascalar, Real bscalar,
        Vector<MultiFab const*> const& acoef_input,
        Vector<Array<MultiFab const*, AMREX_SPACEDIM>> const& bcoef_input,
        Array<LinOpBCType, AMREX_SPACEDIM> const& lobc_input,
        Array<LinOpBCType, AMREX_SPACEDIM> const& hibc_input,
        Vector<MultiFab const*> const& level_bc_input,
        RobinBCData const& robin_input)
    {
        validate_setup_inputs(acoef_input, bcoef_input, lobc_input, hibc_input,
                              level_bc_input, robin_input);
        AMREX_ALWAYS_ASSERT(ascalar >= Real(0) && bscalar >= Real(0));
        lobc = lobc_input;
        hibc = hibc_input;

        Vector<std::unique_ptr<MultiFab>> acoef(nlevels);
        Vector<Array<std::unique_ptr<MultiFab>, AMREX_SPACEDIM>> bcoef(nlevels);
        Vector<std::unique_ptr<MultiFab>> level_bc;
        Vector<std::unique_ptr<MultiFab>> robin_a;
        Vector<std::unique_ptr<MultiFab>> robin_b;
        Vector<std::unique_ptr<MultiFab>> robin_f;
        bool have_level_bc = static_cast<int>(level_bc_input.size()) == nlevels;
        bool have_robin = static_cast<int>(robin_input.a.size()) == nlevels;
        if (have_level_bc) {
            level_bc.resize(nlevels);
        }
        if (have_robin) {
            robin_a.resize(nlevels);
            robin_b.resize(nlevels);
            robin_f.resize(nlevels);
        }
        for (int level = 0; level < nlevels; ++level) {
            acoef[level] = stage_multifab(*acoef_input[level], 0,
                                         geom[level].periodicity());
            for (int direction = 0; direction < AMREX_SPACEDIM;
                 ++direction) {
                bcoef[level][direction] = stage_multifab(
                    *bcoef_input[level][direction], 0,
                    geom[level].periodicity());
            }
            if (have_level_bc) {
                level_bc[level] = stage_multifab(
                    *level_bc_input[level], 1, geom[level].periodicity());
            }
            if (have_robin) {
                robin_a[level] = stage_multifab(
                    *robin_input.a[level], 0, geom[level].periodicity());
                robin_b[level] = stage_multifab(
                    *robin_input.b[level], 0, geom[level].periodicity());
                robin_f[level] = stage_multifab(
                    *robin_input.f[level], 0, geom[level].periodicity());
            }
        }

        Vector<std::unique_ptr<LongFabArray>> fine_rows_on_coarse(nlevels - 1);
        Vector<Array<std::unique_ptr<MultiFab>, AMREX_SPACEDIM>>
            fine_b_on_coarse(nlevels - 1);
        Vector<std::unique_ptr<LongFabArray>> coarse_rows_on_fine(nlevels - 1);
        for (int level = 0; level + 1 < nlevels; ++level) {
            BoxArray refined_coarse = grids[level];
            refined_coarse.refine(ref_ratio[level]);
            // A fine neighbor can lie just beyond the refined image of the
            // current coarse grid when the coarse/fine interface coincides
            // with a coarse grid boundary.
            fine_rows_on_coarse[level] = std::make_unique<LongFabArray>(
                refined_coarse, dmap[level], 1, 1, host_info());
            fine_rows_on_coarse[level]->setVal(Long(-1));
            fine_rows_on_coarse[level]->ParallelCopy(
                *row_ids[level + 1], 0, 0, 1, IntVect(0), IntVect(1),
                geom[level + 1].periodicity());
            for (int direction = 0; direction < AMREX_SPACEDIM;
                 ++direction) {
                BoxArray face_layout = refined_coarse;
                face_layout.convert(IntVect::TheDimensionVector(direction));
                fine_b_on_coarse[level][direction] =
                    std::make_unique<MultiFab>(face_layout, dmap[level], 1, 0,
                                               host_info());
                fine_b_on_coarse[level][direction]->setVal(Real(0));
                fine_b_on_coarse[level][direction]->ParallelCopy(
                    *bcoef[level + 1][direction], 0, 0, 1, IntVect(0),
                    IntVect(0), geom[level + 1].periodicity());
            }

            BoxArray coarsened_fine = grids[level + 1];
            coarsened_fine.coarsen(ref_ratio[level]);
            coarse_rows_on_fine[level] = std::make_unique<LongFabArray>(
                coarsened_fine, dmap[level + 1], 1, 2, host_info());
            coarse_rows_on_fine[level]->setVal(Long(-1));
            coarse_rows_on_fine[level]->ParallelCopy(
                *row_ids[level], 0, 0, 1, IntVect(0), IntVect(2),
                geom[level].periodicity());
        }
        Gpu::streamSynchronize();

        int const myproc = ParallelDescriptor::MyProc();
        Long const begin = partition[myproc];
        Long const nlocal = partition[myproc + 1] - begin;
        Vector<Entries> rows(nlocal);
        boundary_rhs.assign(nlocal, Real(0));
        Long local_cf_connections = 0;
        Real local_minimum_diagonal = std::numeric_limits<Real>::max();
        Real local_maximum_offdiag = std::numeric_limits<Real>::lowest();

        for (int level = 0; level < nlevels; ++level) {
            Real const volume = cell_volume(geom[level]);
            auto const dx = geom[level].CellSizeArray();
            Box const domain = geom[level].Domain();
            for (MFIter mfi(active[level]); mfi.isValid(); ++mfi) {
                auto const& mask = active[level][mfi];
                auto const& row_fab = (*row_ids[level])[mfi];
                auto const a = acoef[level]->const_array(mfi);
                GpuArray<Array4<Real const>, AMREX_SPACEDIM> face_b;
                for (int direction = 0; direction < AMREX_SPACEDIM;
                     ++direction) {
                    face_b[direction] =
                        bcoef[level][direction]->const_array(mfi);
                }
                Array4<Real const> bc;
                Array4<Real const> ra;
                Array4<Real const> rb;
                Array4<Real const> rf;
                if (have_level_bc) {
                    bc = level_bc[level]->const_array(mfi);
                }
                if (have_robin) {
                    ra = robin_a[level]->const_array(mfi);
                    rb = robin_b[level]->const_array(mfi);
                    rf = robin_f[level]->const_array(mfi);
                }

                for (BoxIterator bit(mfi.validbox()); bit.ok(); ++bit) {
                    IntVect const iv = bit();
                    if (mask(iv) == 0) {
                        continue;
                    }
                    Long const global_row = row_fab(iv);
                    AMREX_ALWAYS_ASSERT(global_row >= begin &&
                                        global_row < begin + nlocal);
                    Long const local_row = global_row - begin;
                    Real diagonal = ascalar * a(iv) * volume;

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
                                neighbor[direction] <
                                    domain.smallEnd(direction) ||
                                neighbor[direction] >
                                    domain.bigEnd(direction);
                            if (outside && !geom[level].isPeriodic(direction)) {
                                LinOpBCType const type =
                                    side < 0 ? lobc[direction] : hibc[direction];
                                if (type == LinOpBCType::Neumann) {
                                    continue;
                                }
                                Real const k = face_b[direction](face);
                                Real const distance = Real(0.5) * dx[direction];
                                if (type == LinOpBCType::Dirichlet) {
                                    Real const coefficient =
                                        bscalar * k * area / distance;
                                    diagonal += coefficient;
                                    IntVect exterior = iv;
                                    exterior[direction] += side;
                                    boundary_rhs[local_row] +=
                                        coefficient * bc(exterior);
                                } else {
                                    AMREX_ALWAYS_ASSERT(is_robin(type));
                                    Real const aa = ra(iv);
                                    Real const bb = rb(iv);
                                    Real const ff = rf(iv);
                                    AMREX_ALWAYS_ASSERT(aa >= Real(0) &&
                                                        bb >= Real(0) &&
                                                        aa + bb > Real(0));
                                    Real const denominator = bb + aa * distance;
                                    Real const scale =
                                        bscalar * k * area / denominator;
                                    diagonal += scale * aa;
                                    boundary_rhs[local_row] += scale * ff;
                                }
                                continue;
                            }

                            Long neighbor_row = row_fab(neighbor);
                            if (neighbor_row >= 0) {
                                Real const coefficient =
                                    bscalar * face_b[direction](face) * area /
                                    dx[direction];
                                diagonal += coefficient;
                                add_entry(rows[local_row], neighbor_row,
                                          -coefficient);
                                continue;
                            }

                            bool connected_to_fine = false;
                            if (level + 1 < nlevels) {
                                auto const rr = ref_ratio[level];
                                auto const& fine_rows =
                                    (*fine_rows_on_coarse[level])[mfi];
                                auto const fine_b =
                                    fine_b_on_coarse[level][direction]
                                        ->const_array(mfi);
                                auto const dxf =
                                    geom[level + 1].CellSizeArray();
                                Real const fine_volume =
                                    cell_volume(geom[level + 1]);
                                Real const fine_area =
                                    fine_volume / dxf[direction];
                                Real const distance =
                                    Real(0.5) *
                                    (dx[direction] + dxf[direction]);
                                int const transverse = 1 - direction;
                                for (int offset = 0; offset < rr[transverse];
                                     ++offset) {
                                    IntVect fine_cell;
                                    IntVect fine_face;
                                    if (direction == 0) {
                                        fine_cell[1] =
                                            iv[1] * rr[1] + offset;
                                        fine_face[1] = fine_cell[1];
                                        if (side < 0) {
                                            fine_cell[0] = iv[0] * rr[0] - 1;
                                            fine_face[0] = iv[0] * rr[0];
                                        } else {
                                            fine_cell[0] =
                                                (iv[0] + 1) * rr[0];
                                            fine_face[0] = fine_cell[0];
                                        }
                                    } else {
                                        fine_cell[0] =
                                            iv[0] * rr[0] + offset;
                                        fine_face[0] = fine_cell[0];
                                        if (side < 0) {
                                            fine_cell[1] = iv[1] * rr[1] - 1;
                                            fine_face[1] = iv[1] * rr[1];
                                        } else {
                                            fine_cell[1] =
                                                (iv[1] + 1) * rr[1];
                                            fine_face[1] = fine_cell[1];
                                        }
                                    }
                                    Long const fine_row = fine_rows(fine_cell);
                                    if (fine_row >= 0) {
                                        Real const coefficient =
                                            bscalar * fine_b(fine_face) *
                                            fine_area / distance;
                                        diagonal += coefficient;
                                        add_entry(rows[local_row], fine_row,
                                                  -coefficient);
                                        connected_to_fine = true;
                                        ++local_cf_connections;
                                    }
                                }
                            }
                            if (connected_to_fine) {
                                continue;
                            }

                            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                                level > 0,
                                "MLABecLapAMG could not resolve an interior "
                                "composite-grid neighbor");
                            auto const rr = ref_ratio[level - 1];
                            IntVect coarse_neighbor = amrex::coarsen(neighbor, rr);
                            Long const coarse_row =
                                (*coarse_rows_on_fine[level - 1])[mfi](
                                    coarse_neighbor);
                            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                                coarse_row >= 0,
                                "MLABecLapAMG missing coarse row metadata at "
                                "a coarse/fine interface");
                            Real const coarse_dx =
                                geom[level - 1].CellSize(direction);
                            Real const distance =
                                Real(0.5) * (dx[direction] + coarse_dx);
                            Real const coefficient =
                                bscalar * face_b[direction](face) * area /
                                distance;
                            diagonal += coefficient;
                            add_entry(rows[local_row], coarse_row,
                                      -coefficient);
                            ++local_cf_connections;
                        }
                    }
                    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                        diagonal > Real(0),
                        "MLABecLapAMG assembled a nonpositive diagonal");
                    add_entry(rows[local_row], global_row, diagonal);
                    local_minimum_diagonal =
                        amrex::min(local_minimum_diagonal, diagonal);
                }
            }
        }

        using HostCSR = CSR<Real, Gpu::PinnedVector>;
        using DeviceCSR = SpMatrix<Real>::csr_type;
        HostCSR host;
        host.row_offset.resize(nlocal + 1);
        host.row_offset[0] = 0;
        for (Long local_row = 0; local_row < nlocal; ++local_row) {
            auto& entries = rows[local_row];
            std::sort(entries.begin(), entries.end(),
                      [] (Entry const& lhs, Entry const& rhs)
                      { return lhs.first < rhs.first; });
            bool have_diagonal = false;
            for (std::size_t p = 0; p < entries.size();) {
                Long const column = entries[p].first;
                Real value = Real(0);
                do {
                    value += entries[p].second;
                    ++p;
                } while (p < entries.size() && entries[p].first == column);
                AMREX_ALWAYS_ASSERT(column >= 0 &&
                                    column < partition.numGlobalRows());
                if (value != Real(0)) {
                    host.col_index.push_back(column);
                    host.mat.push_back(value);
                    if (column == begin + local_row) {
                        have_diagonal = true;
                    } else {
                        local_maximum_offdiag =
                            amrex::max(local_maximum_offdiag, value);
                        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                            value <= Real(1.e-14),
                            "MLABecLapAMG assembled a positive off-diagonal");
                    }
                }
            }
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                have_diagonal,
                "MLABecLapAMG assembled an empty row without a diagonal");
            host.row_offset[local_row + 1] =
                static_cast<Long>(host.mat.size());
        }
        host.nnz = static_cast<Long>(host.mat.size());
        DeviceCSR device;
        duplicateCSR(Gpu::hostToDevice, device, host);
        Gpu::streamSynchronize();

        gmres.reset();
        amg.reset();
        matrix.reset();
        matrix = std::make_unique<SpMatrix<Real>>(partition, partition,
                                                  std::move(device));
        amg = std::make_unique<AMG<Real>>(*matrix, options);
        amg->setup();
        gmres = std::make_unique<GMRES_MV<Real>>(matrix.get());
        gmres->setPrecond([this] (AlgVector<Real>& lhs,
                                  AlgVector<Real> const& rhs)
        { amg->apply(lhs, rhs); });
        gmres->getGMRES().setRestartLength(restart_length);
        gmres->getGMRES().setMaxIters(max_iter);
        gmres->setVerbose(verbose);

        assembly.coarse_fine_connections = local_cf_connections;
        ParallelDescriptor::ReduceLongSum(assembly.coarse_fine_connections);
        assembly.minimum_diagonal = local_minimum_diagonal;
        assembly.maximum_off_diagonal = local_maximum_offdiag;
        ParallelDescriptor::ReduceRealMin(assembly.minimum_diagonal);
        ParallelDescriptor::ReduceRealMax(assembly.maximum_off_diagonal);
        ++assembly.setup_generation;
    }

    SolveInfo solve (Vector<MultiFab*> const& solution,
                     Vector<MultiFab const*> const& rhs,
                     Real relative_tolerance, Real absolute_tolerance)
    {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            matrix != nullptr && amg != nullptr && gmres != nullptr,
            "MLABecLapAMG::setup must be called before solve");
        AMREX_ALWAYS_ASSERT(static_cast<int>(solution.size()) == nlevels &&
                            static_cast<int>(rhs.size()) == nlevels);
        AMREX_ALWAYS_ASSERT(relative_tolerance >= Real(0) &&
                            absolute_tolerance >= Real(0));

        Vector<std::unique_ptr<MultiFab>> host_solution(nlevels);
        Vector<std::unique_ptr<MultiFab>> host_rhs(nlevels);
        for (int level = 0; level < nlevels; ++level) {
            AMREX_ALWAYS_ASSERT(solution[level] != nullptr &&
                                rhs[level] != nullptr);
            AMREX_ALWAYS_ASSERT(solution[level]->boxArray() == grids[level] &&
                                rhs[level]->boxArray() == grids[level]);
            AMREX_ALWAYS_ASSERT(solution[level]->nComp() == 1 &&
                                rhs[level]->nComp() == 1);
            AMREX_ALWAYS_ASSERT(rhs[level]->nGrow() == 0);
            host_solution[level] = stage_multifab(
                *solution[level], 0, geom[level].periodicity());
            host_rhs[level] =
                stage_multifab(*rhs[level], 0, geom[level].periodicity());
        }

        int const myproc = ParallelDescriptor::MyProc();
        Long const begin = partition[myproc];
        Long const nlocal = partition[myproc + 1] - begin;
        Gpu::PinnedVector<Real> local_solution(nlocal, Real(0));
        Gpu::PinnedVector<Real> local_rhs(nlocal, Real(0));
        for (int level = 0; level < nlevels; ++level) {
            Real const volume = cell_volume(geom[level]);
            for (MFIter mfi(active[level]); mfi.isValid(); ++mfi) {
                auto const& mask = active[level][mfi];
                auto const& rows = (*row_ids[level])[mfi];
                auto const& x = (*host_solution[level])[mfi];
                auto const& b = (*host_rhs[level])[mfi];
                for (BoxIterator bit(mfi.validbox()); bit.ok(); ++bit) {
                    IntVect const& iv = bit();
                    if (mask(iv) != 0) {
                        Long const local = rows(iv) - begin;
                        AMREX_ALWAYS_ASSERT(local >= 0 && local < nlocal);
                        local_solution[local] = x(iv);
                        local_rhs[local] = volume * b(iv) + boundary_rhs[local];
                    }
                }
            }
        }

        AlgVector<Real> algebra_solution(partition);
        AlgVector<Real> algebra_rhs(partition);
        if (nlocal > 0) {
            Gpu::copyAsync(Gpu::hostToDevice, local_solution.begin(),
                           local_solution.end(), algebra_solution.data());
            Gpu::copyAsync(Gpu::hostToDevice, local_rhs.begin(),
                           local_rhs.end(), algebra_rhs.data());
            Gpu::streamSynchronize();
        }
        gmres->solve(algebra_solution, algebra_rhs, relative_tolerance,
                     absolute_tolerance);
        auto const& solver = gmres->getGMRES();
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            solver.getStatus() == 0,
            "GMRES+AMG did not converge in MLABecLapAMG::solve");

        AlgVector<Real> residual(partition);
        SpMV(residual, *matrix, algebra_solution);
        LinComb(residual, Real(1), algebra_rhs, Real(-1), residual);
        SolveInfo info;
        info.iterations = solver.getNumIters();
        info.absolute_residual = residual.norm2();
        Real const rhs_norm = algebra_rhs.norm2();
        info.relative_residual =
            info.absolute_residual / amrex::max(rhs_norm, Real(1.e-30));
        Real const target =
            amrex::max(absolute_tolerance, relative_tolerance * rhs_norm);
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            info.absolute_residual <= Real(5) *
                                          amrex::max(target, Real(1.e-30)),
            "MLABecLapAMG true residual exceeds the requested tolerance");

        if (nlocal > 0) {
            Gpu::copyAsync(Gpu::deviceToHost, algebra_solution.data(),
                           algebra_solution.data() + nlocal,
                           local_solution.begin());
            Gpu::streamSynchronize();
        }
        for (int level = 0; level < nlevels; ++level) {
            for (MFIter mfi(active[level]); mfi.isValid(); ++mfi) {
                auto const& mask = active[level][mfi];
                auto const& rows = (*row_ids[level])[mfi];
                auto& x = (*host_solution[level])[mfi];
                for (BoxIterator bit(mfi.validbox()); bit.ok(); ++bit) {
                    IntVect const& iv = bit();
                    if (mask(iv) != 0) {
                        x(iv) = local_solution[rows(iv) - begin];
                    }
                }
            }
            MultiFab::Copy(*solution[level], *host_solution[level], 0, 0, 1,
                           0);
        }
        for (int level = nlevels - 2; level >= 0; --level) {
            amrex::average_down(*solution[level + 1], *solution[level],
                                geom[level + 1], geom[level], 0, 1,
                                ref_ratio[level]);
        }
        for (int level = 0; level < nlevels; ++level) {
            solution[level]->FillBoundary(geom[level].periodicity());
        }
        return info;
    }

    Vector<Geometry> geom;
    Vector<BoxArray> grids;
    Vector<DistributionMapping> dmap;
    Vector<IntVect> ref_ratio;
    int nlevels = 0;
    AMG<Real>::Options options;
    std::string parmparse_prefix;
    int verbose = 0;
    int max_iter = 500;
    int restart_length = 50;
    Array<LinOpBCType, AMREX_SPACEDIM> lobc;
    Array<LinOpBCType, AMREX_SPACEDIM> hibc;

    Vector<iMultiFab> active;
    Vector<std::unique_ptr<LongFabArray>> row_ids;
    AlgPartition partition;
    Vector<Real> boundary_rhs;
    std::unique_ptr<SpMatrix<Real>> matrix;
    std::unique_ptr<AMG<Real>> amg;
    std::unique_ptr<GMRES_MV<Real>> gmres;
    MLABecAssemblyDiagnostics assembly;
};

MLABecLapAMG::MLABecLapAMG (
    Vector<Geometry> geom, Vector<BoxArray> grids,
    Vector<DistributionMapping> dmap, AMG<Real>::Options options,
    std::string parmparse_prefix)
    : m_impl(std::make_unique<Impl>(
          std::move(geom), std::move(grids), std::move(dmap),
          std::move(options), std::move(parmparse_prefix)))
{}

MLABecLapAMG::~MLABecLapAMG () = default;

void
MLABecLapAMG::setVerbose (int value)
{
    m_impl->verbose = value;
    if (m_impl->gmres) {
        m_impl->gmres->setVerbose(value);
    }
}

void
MLABecLapAMG::setMaxIter (int value)
{
    AMREX_ALWAYS_ASSERT(value > 0);
    m_impl->max_iter = value;
    if (m_impl->gmres) {
        m_impl->gmres->getGMRES().setMaxIters(value);
    }
}

void
MLABecLapAMG::setup (
    Real ascalar, Real bscalar, Vector<MultiFab const*> const& acoef,
    Vector<Array<MultiFab const*, AMREX_SPACEDIM>> const& bcoef,
    Array<LinOpBCType, AMREX_SPACEDIM> const& lobc,
    Array<LinOpBCType, AMREX_SPACEDIM> const& hibc,
    Vector<MultiFab const*> const& level_bc_data,
    RobinBCData const& robin_bc_data)
{
    m_impl->assemble(ascalar, bscalar, acoef, bcoef, lobc, hibc,
                     level_bc_data, robin_bc_data);
}

SolveInfo
MLABecLapAMG::solve (Vector<MultiFab*> const& solution,
                    Vector<MultiFab const*> const& rhs,
                    Real relative_tolerance, Real absolute_tolerance)
{
    return m_impl->solve(solution, rhs, relative_tolerance,
                         absolute_tolerance);
}

AMG<Real>::Diagnostics const&
MLABecLapAMG::diagnostics () const noexcept
{
    AMREX_ALWAYS_ASSERT(m_impl->amg != nullptr);
    return m_impl->amg->diagnostics();
}

MLABecAssemblyDiagnostics const&
MLABecLapAMG::assembly_diagnostics () const noexcept
{
    return m_impl->assembly;
}

} // namespace fld_test
