#include "MLABecAMG.H"
#include "CompositeGridTopology.H"

#include <AMReX.H>
#include <AMReX_AlgVecUtil.H>
#include <AMReX_Arena.H>
#include <AMReX_GMRES_MV.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_SpMV.H>

#include <cmath>
#include <utility>

namespace fld_test
{

using namespace amrex;

namespace
{

MFInfo
host_info ()
{
    return MFInfo().SetArena(The_Pinned_Arena());
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

bool
is_robin (LinOpBCType type) noexcept
{
    return type == LinOpBCType::Robin || type == LinOpBCType::Marshak;
}

} // namespace

struct MLABecLapAMG::Impl
{
    Impl (Vector<Geometry> geom, Vector<BoxArray> grids,
          Vector<DistributionMapping> dmap, AMG<Real>::Options a_options,
          std::string prefix)
        : topology(std::move(geom), std::move(grids), std::move(dmap)),
          options(std::move(a_options)), parmparse_prefix(std::move(prefix))
    {
        ParmParse pp(parmparse_prefix);
        pp.query("verbose", verbose);
        pp.query("max_iter", max_iter);
        pp.query("restart_length", restart_length);
        AMREX_ALWAYS_ASSERT(max_iter > 0 && restart_length > 0);
        assembly.local_rows = topology.localRows();
        assembly.global_rows = topology.globalRows();
    }

    void validate_setup_inputs (
        Vector<MultiFab const*> const& acoef,
        Vector<Array<MultiFab const*, AMREX_SPACEDIM>> const& bcoef,
        Array<LinOpBCType, AMREX_SPACEDIM> const& lobc,
        Array<LinOpBCType, AMREX_SPACEDIM> const& hibc,
        Vector<MultiFab const*> const& level_bc,
        RobinBCData const& robin) const
    {
        int const nlevels = topology.numLevels();
        auto const& geom = topology.geometry();
        auto const& grids = topology.grids();
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
                BoxArray expected = grids[level];
                expected.convert(IntVect::TheDimensionVector(direction));
                AMREX_ALWAYS_ASSERT(
                    bcoef[level][direction]->boxArray() == expected);
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
                                    level_bc[level]->boxArray() == grids[level] &&
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
                AMREX_ALWAYS_ASSERT(
                    robin.a[level]->boxArray() == grids[level] &&
                    robin.b[level]->boxArray() == grids[level] &&
                    robin.f[level]->boxArray() == grids[level]);
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
        int const nlevels = topology.numLevels();
        auto const& geom = topology.geometry();

        Vector<std::unique_ptr<MultiFab>> acoef(nlevels);
        Vector<Array<std::unique_ptr<MultiFab>, AMREX_SPACEDIM>> bcoef(nlevels);
        Vector<std::unique_ptr<MultiFab>> level_bc;
        Vector<std::unique_ptr<MultiFab>> robin_a;
        Vector<std::unique_ptr<MultiFab>> robin_b;
        Vector<std::unique_ptr<MultiFab>> robin_f;
        bool const have_level_bc =
            static_cast<int>(level_bc_input.size()) == nlevels;
        bool const have_robin =
            static_cast<int>(robin_input.a.size()) == nlevels;
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

        Vector<MultiFab const*> acoef_ptrs(nlevels);
        Vector<Array<MultiFab const*, AMREX_SPACEDIM>> bcoef_ptrs(nlevels);
        CompositeGridTopology::BoundaryData boundary;
        if (have_level_bc) {
            boundary.level.resize(nlevels);
        }
        if (have_robin) {
            boundary.robin_a.resize(nlevels);
            boundary.robin_b.resize(nlevels);
            boundary.robin_f.resize(nlevels);
        }
        for (int level = 0; level < nlevels; ++level) {
            acoef_ptrs[level] = acoef[level].get();
            for (int direction = 0; direction < AMREX_SPACEDIM;
                 ++direction) {
                bcoef_ptrs[level][direction] =
                    bcoef[level][direction].get();
            }
            if (have_level_bc) {
                boundary.level[level] = level_bc[level].get();
            }
            if (have_robin) {
                boundary.robin_a[level] = robin_a[level].get();
                boundary.robin_b[level] = robin_b[level].get();
                boundary.robin_f[level] = robin_f[level].get();
            }
        }

        auto numerical = topology.assemble(
            ascalar, bscalar, acoef_ptrs, bcoef_ptrs, lobc_input, hibc_input,
            boundary);
        boundary_rhs = std::move(numerical.boundary_rhs);
        using DeviceCSR = SpMatrix<Real>::csr_type;
        DeviceCSR device;
        duplicateCSR(Gpu::hostToDevice, device, numerical.matrix);
        Gpu::streamSynchronize();

        gmres.reset();
        amg.reset();
        matrix.reset();
        matrix = std::make_unique<SpMatrix<Real>>(
            topology.partition(), topology.partition(), std::move(device));
        amg = std::make_unique<AMG<Real>>(*matrix, options);
        amg->setup();
        gmres = std::make_unique<GMRES_MV<Real>>(matrix.get());
        gmres->setPrecond([this] (AlgVector<Real>& lhs,
                                  AlgVector<Real> const& rhs)
        { amg->apply(lhs, rhs); });
        gmres->getGMRES().setRestartLength(restart_length);
        gmres->getGMRES().setMaxIters(max_iter);
        gmres->setVerbose(verbose);

        assembly.coarse_fine_connections =
            numerical.local_coarse_fine_connections;
        ParallelDescriptor::ReduceLongSum(assembly.coarse_fine_connections);
        assembly.minimum_diagonal = numerical.local_minimum_diagonal;
        assembly.maximum_off_diagonal =
            numerical.local_maximum_off_diagonal;
        ParallelDescriptor::ReduceRealMin(assembly.minimum_diagonal);
        ParallelDescriptor::ReduceRealMax(assembly.maximum_off_diagonal);
        ++assembly.setup_generation;
    }

    void validate_field_vectors (Vector<MultiFab*> const& output,
                                 Vector<MultiFab const*> const& input) const
    {
        int const nlevels = topology.numLevels();
        auto const& grids = topology.grids();
        AMREX_ALWAYS_ASSERT(static_cast<int>(output.size()) == nlevels &&
                            static_cast<int>(input.size()) == nlevels);
        for (int level = 0; level < nlevels; ++level) {
            AMREX_ALWAYS_ASSERT(output[level] != nullptr &&
                                input[level] != nullptr);
            AMREX_ALWAYS_ASSERT(output[level]->boxArray() == grids[level] &&
                                input[level]->boxArray() == grids[level]);
            AMREX_ALWAYS_ASSERT(output[level]->nComp() == 1 &&
                                input[level]->nComp() == 1);
        }
    }

    SolveInfo solve (Vector<MultiFab*> const& solution,
                     Vector<MultiFab const*> const& rhs,
                     Real relative_tolerance, Real absolute_tolerance)
    {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            matrix != nullptr && amg != nullptr && gmres != nullptr,
            "MLABecLapAMG::setup must be called before solve");
        validate_field_vectors(solution, rhs);
        AMREX_ALWAYS_ASSERT(relative_tolerance >= Real(0) &&
                            absolute_tolerance >= Real(0));

        int const nlevels = topology.numLevels();
        auto const& geom = topology.geometry();
        auto const& cells = topology.cells();
        auto const& partition = topology.partition();
        Long const nlocal = topology.localRows();
        Vector<std::unique_ptr<MultiFab>> host_solution(nlevels);
        Vector<std::unique_ptr<MultiFab>> host_rhs(nlevels);
        for (int level = 0; level < nlevels; ++level) {
            AMREX_ALWAYS_ASSERT(rhs[level]->nGrow() == 0);
            host_solution[level] = stage_multifab(
                *solution[level], 0, geom[level].periodicity());
            host_rhs[level] =
                stage_multifab(*rhs[level], 0, geom[level].periodicity());
        }

        Gpu::PinnedVector<Real> local_solution(nlocal, Real(0));
        Gpu::PinnedVector<Real> local_rhs(nlocal, Real(0));
        for (Long row = 0; row < nlocal; ++row) {
            auto const& cell = cells[row];
            local_solution[row] =
                host_solution[cell.level]->atLocalIdx(cell.local_grid)(
                    cell.index);
            local_rhs[row] = cell.volume *
                                 host_rhs[cell.level]
                                     ->atLocalIdx(cell.local_grid)(cell.index) +
                             boundary_rhs[row];
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
        for (Long row = 0; row < nlocal; ++row) {
            auto const& cell = cells[row];
            host_solution[cell.level]->atLocalIdx(cell.local_grid)(
                cell.index) = local_solution[row];
        }
        for (int level = 0; level < nlevels; ++level) {
            MultiFab::Copy(*solution[level], *host_solution[level], 0, 0, 1,
                           0);
        }
        for (int level = nlevels - 2; level >= 0; --level) {
            amrex::average_down(*solution[level + 1], *solution[level],
                                geom[level + 1], geom[level], 0, 1,
                                topology.refRatio()[level]);
        }
        for (int level = 0; level < nlevels; ++level) {
            solution[level]->FillBoundary(geom[level].periodicity());
        }
        return info;
    }

    void apply (Vector<MultiFab*> const& output,
                Vector<MultiFab const*> const& input) const
    {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            matrix != nullptr,
            "MLABecLapAMG::setup must be called before apply");
        validate_field_vectors(output, input);

        int const nlevels = topology.numLevels();
        auto const& geom = topology.geometry();
        auto const& cells = topology.cells();
        auto const& partition = topology.partition();
        Long const nlocal = topology.localRows();
        Vector<std::unique_ptr<MultiFab>> host_input(nlevels);
        Vector<std::unique_ptr<MultiFab>> host_output(nlevels);
        for (int level = 0; level < nlevels; ++level) {
            host_input[level] = stage_multifab(
                *input[level], 0, geom[level].periodicity());
            host_output[level] = stage_multifab(
                *output[level], 0, geom[level].periodicity());
        }

        Gpu::PinnedVector<Real> local_input(nlocal, Real(0));
        Gpu::PinnedVector<Real> local_output(nlocal, Real(0));
        for (Long row = 0; row < nlocal; ++row) {
            auto const& cell = cells[row];
            local_input[row] =
                host_input[cell.level]->atLocalIdx(cell.local_grid)(
                    cell.index);
        }
        AlgVector<Real> algebra_input(partition);
        AlgVector<Real> algebra_output(partition);
        if (nlocal > 0) {
            Gpu::copyAsync(Gpu::hostToDevice, local_input.begin(),
                           local_input.end(), algebra_input.data());
            Gpu::streamSynchronize();
        }
        SpMV(algebra_output, *matrix, algebra_input);
        if (nlocal > 0) {
            Gpu::copyAsync(Gpu::deviceToHost, algebra_output.data(),
                           algebra_output.data() + nlocal,
                           local_output.begin());
            Gpu::streamSynchronize();
        }
        for (Long row = 0; row < nlocal; ++row) {
            auto const& cell = cells[row];
            host_output[cell.level]->atLocalIdx(cell.local_grid)(cell.index) =
                local_output[row] / cell.volume;
        }
        for (int level = 0; level < nlevels; ++level) {
            MultiFab::Copy(*output[level], *host_output[level], 0, 0, 1, 0);
        }
        for (int level = nlevels - 2; level >= 0; --level) {
            amrex::average_down(*output[level + 1], *output[level],
                                geom[level + 1], geom[level], 0, 1,
                                topology.refRatio()[level]);
        }
        for (int level = 0; level < nlevels; ++level) {
            output[level]->FillBoundary(geom[level].periodicity());
        }
    }

    CompositeGridTopology topology;
    AMG<Real>::Options options;
    std::string parmparse_prefix;
    int verbose = 0;
    int max_iter = 500;
    int restart_length = 50;

    Gpu::PinnedVector<Real> boundary_rhs;
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

void
MLABecLapAMG::apply (Vector<MultiFab*> const& output,
                     Vector<MultiFab const*> const& input) const
{
    m_impl->apply(output, input);
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
