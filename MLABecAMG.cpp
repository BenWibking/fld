#include "MLABecAMG.H"
#include "CompositeGridTopology.H"

#include <AMReX.H>
#include <AMReX_AlgVecUtil.H>
#include <AMReX_Arena.H>
#include <AMReX_GMRES_MV.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_MLABecLaplacian.H>
#include <AMReX_MLMG.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_SpGEMM.H>
#include <AMReX_SpMV.H>

#ifdef AMREX_USE_HYPRE
#include <HYPRE.h>
#include <HYPRE_IJ_mv.h>
#include <HYPRE_parcsr_ls.h>
#endif

#include <algorithm>
#include <cmath>
#include <cctype>
#include <limits>
#include <type_traits>
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

std::unique_ptr<MultiFab>
stage_robin_coefficient (MultiFab const& source, Geometry const& geom)
{
    auto result = std::make_unique<MultiFab>(
        source.boxArray(), source.DistributionMap(), source.nComp(), 1);
    result->setVal(Real(0));
    result->ParallelCopy(source, 0, 0, source.nComp(), IntVect(0), IntVect(1),
                         geom.periodicity());
    Box const domain = geom.Domain();
    for (MFIter mfi(*result); mfi.isValid(); ++mfi) {
        Box const exterior = grow(mfi.validbox(), 1) & mfi.fabbox();
        auto const values = result->array(mfi);
        ParallelFor(exterior, source.nComp(),
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {
            IntVect const cell{
                AMREX_D_DECL(amrex::max(domain.smallEnd(0),
                                        amrex::min(domain.bigEnd(0), i)),
                             amrex::max(domain.smallEnd(1),
                                        amrex::min(domain.bigEnd(1), j)),
                             amrex::max(domain.smallEnd(2),
                                        amrex::min(domain.bigEnd(2), k)))};
            if (!domain.contains(IntVect{AMREX_D_DECL(i, j, k)})) {
                values(i, j, k, n) = values(cell, n);
            }
        });
    }
    Gpu::streamSynchronize();
    return result;
}

bool
is_robin (LinOpBCType type) noexcept
{
    return type == LinOpBCType::Robin || type == LinOpBCType::Marshak;
}

MLABecPreconditioner
parse_preconditioner (std::string name)
{
    std::transform(name.begin(), name.end(), name.begin(),
                   [] (unsigned char c) { return std::tolower(c); });
    if (name == "amg") {
        return MLABecPreconditioner::AMG;
    }
    if (name == "mlmg") {
        return MLABecPreconditioner::MLMG;
    }
    amrex::Abort("Unknown MLABecLapAMG preconditioner '" + name +
                 "'; expected amg or mlmg");
    return MLABecPreconditioner::AMG;
}

char const*
preconditioner_name (MLABecPreconditioner preconditioner) noexcept
{
    switch (preconditioner) {
    case MLABecPreconditioner::AMG: return "amg";
    case MLABecPreconditioner::MLMG: return "mlmg";
    }
    return "unknown";
}

MLABecAMGBackend
parse_amg_backend (std::string name)
{
    std::transform(name.begin(), name.end(), name.begin(),
                   [] (unsigned char c) { return std::tolower(c); });
    if (name == "native" || name == "amg") {
        return MLABecAMGBackend::Native;
    }
    if (name == "boomeramg" || name == "hypre") {
        return MLABecAMGBackend::BoomerAMG;
    }
    amrex::Abort("Unknown MLABecLapAMG AMG backend '" + name +
                 "'; expected native or boomeramg");
    return MLABecAMGBackend::Native;
}

char const*
amg_backend_name (MLABecAMGBackend backend) noexcept
{
    switch (backend) {
    case MLABecAMGBackend::Native: return "native";
    case MLABecAMGBackend::BoomerAMG: return "boomeramg";
    }
    return "unknown";
}

#ifdef AMREX_USE_HYPRE
class BoomerAMGPreconditioner
{
  public:
    BoomerAMGPreconditioner (SpMatrix<Real> const& matrix,
                             AMG<Real>::Options const& options,
                             int verbose)
        : partition(matrix.partition()), nlocal(matrix.numLocalRows()),
          row_ids(nlocal), rhs_values(nlocal), solution_values(nlocal)
    {
        static_assert(std::is_same_v<Real, HYPRE_Real>);
        AMREX_ALWAYS_ASSERT(
            nlocal <= std::numeric_limits<HYPRE_Int>::max());
        auto const begin = matrix.globalRowBegin();
        auto const end = matrix.globalRowEnd();
        auto const ilower = static_cast<HYPRE_BigInt>(begin);
        auto const iupper = static_cast<HYPRE_BigInt>(end - 1);
        auto const comm = ParallelContext::CommunicatorSub();

        auto rows = SpGEMMHelper<Real, DefaultAllocator>::
            copy_local_global_csr(matrix);
        AMREX_ALWAYS_ASSERT(
            rows.nnz <= std::numeric_limits<HYPRE_Int>::max());
        Gpu::DeviceVector<Long> offsets(rows.row_offset.size());
        Gpu::DeviceVector<Long> source_columns(rows.col_index.size());
        Gpu::DeviceVector<Real> source_values(rows.mat.size());
        if (!rows.row_offset.empty()) {
            Gpu::copyAsync(Gpu::hostToDevice, rows.row_offset.begin(),
                           rows.row_offset.end(), offsets.begin());
        }
        if (rows.nnz > 0) {
            Gpu::copyAsync(Gpu::hostToDevice, rows.col_index.begin(),
                           rows.col_index.end(), source_columns.begin());
            Gpu::copyAsync(Gpu::hostToDevice, rows.mat.begin(), rows.mat.end(),
                           source_values.begin());
        }
        Gpu::DeviceVector<HYPRE_Int> row_nnz(nlocal);
        Gpu::DeviceVector<HYPRE_BigInt> columns(rows.nnz);
        Gpu::DeviceVector<HYPRE_Complex> values(rows.nnz);
        auto const* poffsets = offsets.data();
        auto const* psource_columns = source_columns.data();
        auto const* psource_values = source_values.data();
        auto* prows = row_ids.data();
        auto* pncols = row_nnz.data();
        auto* pcolumns = columns.data();
        auto* pvalues = values.data();
        ParallelFor(nlocal,
        [=] AMREX_GPU_DEVICE (Long i) noexcept
        {
            prows[i] = static_cast<HYPRE_BigInt>(begin + i);
            pncols[i] =
                static_cast<HYPRE_Int>(poffsets[i + 1] - poffsets[i]);
        });
        ParallelFor(rows.nnz,
        [=] AMREX_GPU_DEVICE (Long i) noexcept
        {
            pcolumns[i] =
                static_cast<HYPRE_BigInt>(psource_columns[i]);
            pvalues[i] = static_cast<HYPRE_Complex>(psource_values[i]);
        });
        Gpu::streamSynchronize();

        AMREX_ALWAYS_ASSERT(HYPRE_IJMatrixCreate(
                                comm, ilower, iupper, ilower, iupper,
                                &ij_matrix) == 0);
        AMREX_ALWAYS_ASSERT(
            HYPRE_IJMatrixSetObjectType(ij_matrix, HYPRE_PARCSR) == 0);
        AMREX_ALWAYS_ASSERT(HYPRE_IJMatrixInitialize(ij_matrix) == 0);
        AMREX_ALWAYS_ASSERT(HYPRE_IJMatrixSetValues(
                                ij_matrix, static_cast<HYPRE_Int>(nlocal),
                                row_nnz.data(), row_ids.data(), columns.data(),
                                values.data()) == 0);
        AMREX_ALWAYS_ASSERT(HYPRE_IJMatrixAssemble(ij_matrix) == 0);

        define_vector(comm, ilower, iupper, ij_rhs);
        define_vector(comm, ilower, iupper, ij_solution);
        AMREX_ALWAYS_ASSERT(HYPRE_IJMatrixGetObject(
                                ij_matrix,
                                reinterpret_cast<void**>(&par_matrix)) == 0);
        AMREX_ALWAYS_ASSERT(HYPRE_IJVectorGetObject(
                                ij_rhs,
                                reinterpret_cast<void**>(&par_rhs)) == 0);
        AMREX_ALWAYS_ASSERT(HYPRE_IJVectorGetObject(
                                ij_solution,
                                reinterpret_cast<void**>(&par_solution)) == 0);

        AMREX_ALWAYS_ASSERT(HYPRE_BoomerAMGCreate(&solver) == 0);
        HYPRE_BoomerAMGSetStrongThreshold(solver, options.strong_threshold);
        HYPRE_BoomerAMGSetMaxRowSum(solver, options.max_row_sum);
        HYPRE_BoomerAMGSetCoarsenType(solver, 8);
        HYPRE_BoomerAMGSetInterpType(solver, 6);
        HYPRE_BoomerAMGSetPMaxElmts(solver, options.max_interp_elements);
        HYPRE_BoomerAMGSetTruncFactor(solver, HYPRE_Real(0));
        HYPRE_BoomerAMGSetAggNumLevels(solver, 0);
        HYPRE_BoomerAMGSetCycleType(solver, 1);
        HYPRE_BoomerAMGSetRelaxOrder(solver, 0);
        HYPRE_BoomerAMGSetCycleRelaxType(solver, 16, 1);
        HYPRE_BoomerAMGSetCycleRelaxType(solver, 16, 2);
        HYPRE_BoomerAMGSetCycleRelaxType(solver, 9, 3);
        HYPRE_BoomerAMGSetChebyOrder(solver, options.chebyshev_order);
        HYPRE_BoomerAMGSetChebyFraction(
            solver, options.chebyshev_fraction);
        HYPRE_BoomerAMGSetChebyScale(
            solver, options.chebyshev_scale ? 1 : 0);
        HYPRE_BoomerAMGSetChebyVariant(
            solver, options.chebyshev_variant);
        HYPRE_BoomerAMGSetChebyEigEst(
            solver, options.chebyshev_eigenvalue_iterations);
        HYPRE_BoomerAMGSetCycleNumSweeps(solver, options.pre_sweeps, 1);
        HYPRE_BoomerAMGSetCycleNumSweeps(solver, options.post_sweeps, 2);
        HYPRE_BoomerAMGSetCycleNumSweeps(solver, 1, 3);
        HYPRE_BoomerAMGSetMaxCoarseSize(solver, options.max_coarse_size);
        HYPRE_BoomerAMGSetMaxLevels(solver, options.max_levels);
        HYPRE_BoomerAMGSetKeepTranspose(solver, 1);
        HYPRE_BoomerAMGSetTol(solver, HYPRE_Real(0));
        HYPRE_BoomerAMGSetMinIter(solver, 1);
        HYPRE_BoomerAMGSetMaxIter(solver, 1);
        HYPRE_BoomerAMGSetPrintLevel(solver, verbose > 0 ? 2 : 0);
        HYPRE_BoomerAMGSetLogging(solver, 1);
        AMREX_ALWAYS_ASSERT(
            HYPRE_BoomerAMGSetup(solver, par_matrix, par_rhs,
                                 par_solution) == 0);
    }

    ~BoomerAMGPreconditioner ()
    {
        if (solver != nullptr) {
            HYPRE_BoomerAMGDestroy(solver);
        }
        if (ij_solution != nullptr) {
            HYPRE_IJVectorDestroy(ij_solution);
        }
        if (ij_rhs != nullptr) {
            HYPRE_IJVectorDestroy(ij_rhs);
        }
        if (ij_matrix != nullptr) {
            HYPRE_IJMatrixDestroy(ij_matrix);
        }
    }

    void apply (AlgVector<Real>& output, AlgVector<Real> const& input)
    {
        AMREX_ALWAYS_ASSERT(output.partition() == partition &&
                            input.partition() == partition);
        auto const* pinput = input.data();
        auto* prhs = rhs_values.data();
        ParallelFor(nlocal,
        [=] AMREX_GPU_DEVICE (Long i) noexcept
        {
            prhs[i] = static_cast<HYPRE_Complex>(pinput[i]);
        });
        Gpu::streamSynchronize();

        AMREX_ALWAYS_ASSERT(HYPRE_IJVectorInitialize(ij_rhs) == 0);
        AMREX_ALWAYS_ASSERT(HYPRE_IJVectorSetValues(
                                ij_rhs, static_cast<HYPRE_Int>(nlocal),
                                row_ids.data(), rhs_values.data()) == 0);
        AMREX_ALWAYS_ASSERT(HYPRE_IJVectorAssemble(ij_rhs) == 0);
        AMREX_ALWAYS_ASSERT(HYPRE_IJVectorInitialize(ij_solution) == 0);
        AMREX_ALWAYS_ASSERT(HYPRE_IJVectorSetConstantValues(
                                ij_solution, HYPRE_Complex(0)) == 0);
        AMREX_ALWAYS_ASSERT(HYPRE_IJVectorAssemble(ij_solution) == 0);
        AMREX_ALWAYS_ASSERT(
            HYPRE_BoomerAMGSolve(solver, par_matrix, par_rhs,
                                 par_solution) == 0);
        AMREX_ALWAYS_ASSERT(HYPRE_IJVectorGetValues(
                                ij_solution,
                                static_cast<HYPRE_Int>(nlocal),
                                row_ids.data(), solution_values.data()) == 0);
        Gpu::hypreSynchronize();
        auto const* psolution = solution_values.data();
        auto* poutput = output.data();
        ParallelFor(nlocal,
        [=] AMREX_GPU_DEVICE (Long i) noexcept
        {
            poutput[i] = static_cast<Real>(psolution[i]);
        });
        Gpu::streamSynchronize();
    }

    void setVerbose (int verbose)
    {
        HYPRE_BoomerAMGSetPrintLevel(solver, verbose > 0 ? 2 : 0);
    }

  private:
    static void define_vector (MPI_Comm comm, HYPRE_BigInt ilower,
                               HYPRE_BigInt iupper, HYPRE_IJVector& vector)
    {
        AMREX_ALWAYS_ASSERT(
            HYPRE_IJVectorCreate(comm, ilower, iupper, &vector) == 0);
        AMREX_ALWAYS_ASSERT(
            HYPRE_IJVectorSetObjectType(vector, HYPRE_PARCSR) == 0);
        AMREX_ALWAYS_ASSERT(HYPRE_IJVectorInitialize(vector) == 0);
        AMREX_ALWAYS_ASSERT(
            HYPRE_IJVectorSetConstantValues(vector, HYPRE_Complex(0)) == 0);
        AMREX_ALWAYS_ASSERT(HYPRE_IJVectorAssemble(vector) == 0);
    }

    AlgPartition partition;
    Long nlocal = 0;
    Gpu::DeviceVector<HYPRE_BigInt> row_ids;
    Gpu::DeviceVector<HYPRE_Complex> rhs_values;
    Gpu::DeviceVector<HYPRE_Complex> solution_values;
    HYPRE_IJMatrix ij_matrix = nullptr;
    HYPRE_IJVector ij_rhs = nullptr;
    HYPRE_IJVector ij_solution = nullptr;
    HYPRE_ParCSRMatrix par_matrix = nullptr;
    HYPRE_ParVector par_rhs = nullptr;
    HYPRE_ParVector par_solution = nullptr;
    HYPRE_Solver solver = nullptr;
};
#endif

} // namespace

struct MLABecLapAMG::Impl
{
    Impl (Vector<Geometry> geom, Vector<BoxArray> grids,
          Vector<DistributionMapping> dmap, AMG<Real>::Options a_options,
          MLABecPreconditioner a_preconditioner,
          MLABecAMGBackend a_amg_backend, bool query_configuration,
          std::string prefix)
        : topology(std::move(geom), std::move(grids), std::move(dmap)),
          options(std::move(a_options)),
          selected_preconditioner(a_preconditioner),
          selected_amg_backend(a_amg_backend),
          parmparse_prefix(std::move(prefix))
    {
        ParmParse pp(parmparse_prefix);
        pp.query("verbose", verbose);
        pp.query("max_iter", max_iter);
        pp.query("restart_length", restart_length);
        if (query_configuration) {
            std::string name =
                fld_test::preconditioner_name(selected_preconditioner);
            pp.query("preconditioner", name);
            selected_preconditioner = parse_preconditioner(std::move(name));
            std::string backend =
                fld_test::amg_backend_name(selected_amg_backend);
            pp.query("amg_backend", backend);
            selected_amg_backend = parse_amg_backend(std::move(backend));
        }
        AMREX_ALWAYS_ASSERT(max_iter > 0 && restart_length > 0);
#ifndef AMREX_USE_HYPRE
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            selected_preconditioner != MLABecPreconditioner::AMG ||
                selected_amg_backend != MLABecAMGBackend::BoomerAMG,
            "The BoomerAMG backend requires an AMReX build with "
            "USE_HYPRE=TRUE");
#endif
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

    void define_mlmg_fields ()
    {
        int const nlevels = topology.numLevels();
        auto const& grids = topology.grids();
        auto const& dmap = topology.dmap();
        mlmg_solution.resize(nlevels);
        mlmg_rhs.resize(nlevels);
        host_mlmg_solution.resize(nlevels);
        host_mlmg_rhs.resize(nlevels);
        for (int level = 0; level < nlevels; ++level) {
            mlmg_solution[level] =
                std::make_unique<MultiFab>(grids[level], dmap[level], 1, 1);
            mlmg_rhs[level] =
                std::make_unique<MultiFab>(grids[level], dmap[level], 1, 0);
            host_mlmg_solution[level] = std::make_unique<MultiFab>(
                grids[level], dmap[level], 1, 0, host_info());
            host_mlmg_rhs[level] = std::make_unique<MultiFab>(
                grids[level], dmap[level], 1, 0, host_info());
        }
        local_mlmg_input.resize(topology.localRows());
        local_mlmg_output.resize(topology.localRows());
    }

    void setup_mlmg_preconditioner (
        Real ascalar, Real bscalar,
        Vector<MultiFab const*> const& acoef,
        Vector<Array<MultiFab const*, AMREX_SPACEDIM>> const& bcoef,
        Array<LinOpBCType, AMREX_SPACEDIM> const& lobc,
        Array<LinOpBCType, AMREX_SPACEDIM> const& hibc,
        CompositeGridTopology::BoundaryData const& boundary)
    {
        define_mlmg_fields();
        mlmg_operator = std::make_unique<MLABecLaplacian>(
            topology.geometry(), topology.grids(), topology.dmap());
        mlmg_operator->setDomainBC(lobc, hibc);
        mlmg_operator->setScalars(ascalar, bscalar);

        Vector<std::unique_ptr<MultiFab>> staged_robin_a;
        Vector<std::unique_ptr<MultiFab>> staged_robin_b;
        Vector<std::unique_ptr<MultiFab>> homogeneous_robin_f;
        if (!boundary.robin_a.empty()) {
            staged_robin_a.resize(topology.numLevels());
            staged_robin_b.resize(topology.numLevels());
            homogeneous_robin_f.resize(topology.numLevels());
            for (int level = 0; level < topology.numLevels(); ++level) {
                staged_robin_a[level] = stage_robin_coefficient(
                    *boundary.robin_a[level], topology.geometry()[level]);
                staged_robin_b[level] = stage_robin_coefficient(
                    *boundary.robin_b[level], topology.geometry()[level]);
                homogeneous_robin_f[level] = std::make_unique<MultiFab>(
                    topology.grids()[level], topology.dmap()[level], 1, 1);
                homogeneous_robin_f[level]->setVal(Real(0));
            }
        }
        for (int level = 0; level < topology.numLevels(); ++level) {
            mlmg_operator->setACoeffs(level, *acoef[level]);
            mlmg_operator->setBCoeffs(level, bcoef[level]);
            MultiFab const* robin_a = boundary.robin_a.empty()
                                            ? nullptr
                                            : staged_robin_a[level].get();
            MultiFab const* robin_b = boundary.robin_b.empty()
                                            ? nullptr
                                            : staged_robin_b[level].get();
            MultiFab const* robin_f = homogeneous_robin_f.empty()
                                            ? nullptr
                                            : homogeneous_robin_f[level].get();
            mlmg_operator->setLevelBC(level, nullptr, robin_a, robin_b,
                                      robin_f);
        }
        Gpu::streamSynchronize();
        mlmg = std::make_unique<MLMG>(*mlmg_operator);
        mlmg->setVerbose(verbose);
        mlmg->setFixedIter(1);
        mlmg->setBottomSolver(BottomSolver::smoother);
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
        double const setup_start = amrex::second();
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
        mlmg.reset();
        mlmg_operator.reset();
#ifdef AMREX_USE_HYPRE
        boomeramg.reset();
#endif
        matrix.reset();
        matrix = std::make_unique<SpMatrix<Real>>(
            topology.partition(), topology.partition(), std::move(device));
        if (selected_preconditioner == MLABecPreconditioner::AMG) {
            if (selected_amg_backend == MLABecAMGBackend::Native) {
                amg = std::make_unique<AMG<Real>>(*matrix, options);
                amg->setup();
            }
#ifdef AMREX_USE_HYPRE
            else {
                boomeramg = std::make_unique<BoomerAMGPreconditioner>(
                    *matrix, options, verbose);
            }
#endif
        } else {
            setup_mlmg_preconditioner(ascalar, bscalar, acoef_ptrs,
                                      bcoef_ptrs, lobc_input, hibc_input,
                                      boundary);
        }
        gmres = std::make_unique<GMRES_MV<Real>>(matrix.get());
        gmres->setPrecond(
            [this] (AlgVector<Real>& lhs, AlgVector<Real> const& rhs)
            { apply_preconditioner(lhs, rhs); });
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
        last_setup_seconds = amrex::second() - setup_start;
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

    void apply_mlmg_preconditioner (AlgVector<Real>& lhs,
                                    AlgVector<Real> const& rhs)
    {
        Long const nlocal = topology.localRows();
        if (nlocal > 0) {
            Gpu::copyAsync(Gpu::deviceToHost, rhs.data(), rhs.data() + nlocal,
                           local_mlmg_input.begin());
            Gpu::streamSynchronize();
        }
        for (auto& field : host_mlmg_rhs) {
            field->setVal(Real(0));
        }
        auto const& cells = topology.cells();
        for (Long row = 0; row < nlocal; ++row) {
            auto const& cell = cells[row];
            host_mlmg_rhs[cell.level]
                ->atLocalIdx(cell.local_grid)(cell.index) =
                local_mlmg_input[row] / cell.volume;
        }

        int const nlevels = topology.numLevels();
        Vector<MultiFab*> solution(nlevels);
        Vector<MultiFab const*> field_rhs(nlevels);
        for (int level = 0; level < nlevels; ++level) {
            MultiFab::Copy(*mlmg_rhs[level], *host_mlmg_rhs[level], 0, 0, 1,
                           0);
            mlmg_solution[level]->setVal(Real(0));
            solution[level] = mlmg_solution[level].get();
            field_rhs[level] = mlmg_rhs[level].get();
        }
        mlmg->solve(solution, field_rhs, Real(0), Real(0));

        for (int level = 0; level < nlevels; ++level) {
            MultiFab::Copy(*host_mlmg_solution[level], *mlmg_solution[level],
                           0, 0, 1, 0);
        }
        Gpu::streamSynchronize();
        for (Long row = 0; row < nlocal; ++row) {
            auto const& cell = cells[row];
            local_mlmg_output[row] =
                host_mlmg_solution[cell.level]
                    ->atLocalIdx(cell.local_grid)(cell.index);
        }
        if (nlocal > 0) {
            Gpu::copyAsync(Gpu::hostToDevice, local_mlmg_output.begin(),
                           local_mlmg_output.end(), lhs.data());
            Gpu::streamSynchronize();
        }
    }

    void apply_preconditioner (AlgVector<Real>& lhs,
                               AlgVector<Real> const& rhs)
    {
        double const start = amrex::second();
        ++current_preconditioner_applications;
        if (selected_preconditioner == MLABecPreconditioner::AMG) {
            if (selected_amg_backend == MLABecAMGBackend::Native) {
                amg->apply(lhs, rhs);
            }
#ifdef AMREX_USE_HYPRE
            else {
                boomeramg->apply(lhs, rhs);
            }
#endif
        } else {
            apply_mlmg_preconditioner(lhs, rhs);
        }
        current_preconditioner_seconds += amrex::second() - start;
    }

    SolveInfo solve (Vector<MultiFab*> const& solution,
                     Vector<MultiFab const*> const& rhs,
                     Real relative_tolerance, Real absolute_tolerance)
    {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            matrix != nullptr && gmres != nullptr,
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
        current_preconditioner_applications = 0;
        current_preconditioner_seconds = 0.0;
        double const solve_start = amrex::second();
        gmres->solve(algebra_solution, algebra_rhs, relative_tolerance,
                     absolute_tolerance);
        double const solve_seconds = amrex::second() - solve_start;
        auto const& solver = gmres->getGMRES();
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            solver.getStatus() == 0,
            "GMRES+multigrid did not converge in MLABecLapAMG::solve");

        AlgVector<Real> residual(partition);
        SpMV(residual, *matrix, algebra_solution);
        LinComb(residual, Real(1), algebra_rhs, Real(-1), residual);
        SolveInfo info;
        info.iterations = solver.getNumIters();
        info.preconditioner_applications =
            current_preconditioner_applications;
        info.solve_seconds = solve_seconds;
        info.preconditioner_seconds = current_preconditioner_seconds;
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

    void precondition (Vector<MultiFab*> const& output,
                       Vector<MultiFab const*> const& rhs)
    {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            matrix != nullptr && gmres != nullptr,
            "MLABecLapAMG::setup must be called before precondition");
        validate_field_vectors(output, rhs);

        int const nlevels = topology.numLevels();
        auto const& geom = topology.geometry();
        auto const& cells = topology.cells();
        auto const& partition = topology.partition();
        Long const nlocal = topology.localRows();
        Vector<std::unique_ptr<MultiFab>> host_rhs(nlevels);
        Vector<std::unique_ptr<MultiFab>> host_output(nlevels);
        for (int level = 0; level < nlevels; ++level) {
            AMREX_ALWAYS_ASSERT(rhs[level]->nGrow() == 0);
            host_rhs[level] =
                stage_multifab(*rhs[level], 0, geom[level].periodicity());
            host_output[level] =
                stage_multifab(*output[level], 0, geom[level].periodicity());
        }

        Gpu::PinnedVector<Real> local_rhs(nlocal, Real(0));
        Gpu::PinnedVector<Real> local_output(nlocal, Real(0));
        for (Long row = 0; row < nlocal; ++row) {
            auto const& cell = cells[row];
            local_rhs[row] =
                cell.volume * host_rhs[cell.level]
                                  ->atLocalIdx(cell.local_grid)(cell.index);
        }
        AlgVector<Real> algebra_rhs(partition);
        AlgVector<Real> algebra_output(partition);
        if (nlocal > 0) {
            Gpu::copyAsync(Gpu::hostToDevice, local_rhs.begin(),
                           local_rhs.end(), algebra_rhs.data());
            Gpu::streamSynchronize();
        }
        apply_preconditioner(algebra_output, algebra_rhs);
        if (nlocal > 0) {
            Gpu::copyAsync(Gpu::deviceToHost, algebra_output.data(),
                           algebra_output.data() + nlocal,
                           local_output.begin());
            Gpu::streamSynchronize();
        }
        for (Long row = 0; row < nlocal; ++row) {
            auto const& cell = cells[row];
            host_output[cell.level]->atLocalIdx(cell.local_grid)(cell.index) =
                local_output[row];
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
    MLABecPreconditioner selected_preconditioner =
        MLABecPreconditioner::AMG;
    MLABecAMGBackend selected_amg_backend = MLABecAMGBackend::Native;
    std::string parmparse_prefix;
    int verbose = 0;
    int max_iter = 500;
    int restart_length = 50;
    double last_setup_seconds = 0.0;
    int current_preconditioner_applications = 0;
    double current_preconditioner_seconds = 0.0;

    Gpu::PinnedVector<Real> boundary_rhs;
    std::unique_ptr<SpMatrix<Real>> matrix;
    std::unique_ptr<AMG<Real>> amg;
    std::unique_ptr<MLABecLaplacian> mlmg_operator;
    std::unique_ptr<MLMG> mlmg;
#ifdef AMREX_USE_HYPRE
    std::unique_ptr<BoomerAMGPreconditioner> boomeramg;
#endif
    std::unique_ptr<GMRES_MV<Real>> gmres;
    Vector<std::unique_ptr<MultiFab>> mlmg_solution;
    Vector<std::unique_ptr<MultiFab>> mlmg_rhs;
    Vector<std::unique_ptr<MultiFab>> host_mlmg_solution;
    Vector<std::unique_ptr<MultiFab>> host_mlmg_rhs;
    Gpu::PinnedVector<Real> local_mlmg_input;
    Gpu::PinnedVector<Real> local_mlmg_output;
    MLABecAssemblyDiagnostics assembly;
};

MLABecLapAMG::MLABecLapAMG (
    Vector<Geometry> geom, Vector<BoxArray> grids,
    Vector<DistributionMapping> dmap, AMG<Real>::Options options,
    std::string parmparse_prefix)
    : m_impl(std::make_unique<Impl>(
          std::move(geom), std::move(grids), std::move(dmap),
          std::move(options), MLABecPreconditioner::AMG,
          MLABecAMGBackend::Native, true,
          std::move(parmparse_prefix)))
{}

MLABecLapAMG::MLABecLapAMG (
    Vector<Geometry> geom, Vector<BoxArray> grids,
    Vector<DistributionMapping> dmap, AMG<Real>::Options options,
    MLABecPreconditioner preconditioner, MLABecAMGBackend amg_backend,
    std::string parmparse_prefix)
    : m_impl(std::make_unique<Impl>(
          std::move(geom), std::move(grids), std::move(dmap),
          std::move(options), preconditioner, amg_backend, false,
          std::move(parmparse_prefix)))
{}

MLABecLapAMG::~MLABecLapAMG () = default;

void
MLABecLapAMG::setVerbose (int value)
{
    m_impl->verbose = value;
    if (m_impl->gmres) {
        m_impl->gmres->setVerbose(value);
    }
    if (m_impl->mlmg) {
        m_impl->mlmg->setVerbose(value);
    }
#ifdef AMREX_USE_HYPRE
    if (m_impl->boomeramg) {
        m_impl->boomeramg->setVerbose(value);
    }
#endif
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
MLABecLapAMG::precondition (Vector<MultiFab*> const& output,
                            Vector<MultiFab const*> const& rhs)
{
    m_impl->precondition(output, rhs);
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
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_impl->amg != nullptr,
        "Native AMG hierarchy diagnostics require preconditioner=amg and "
        "amg_backend=native");
    return m_impl->amg->diagnostics();
}

MLABecAssemblyDiagnostics const&
MLABecLapAMG::assembly_diagnostics () const noexcept
{
    return m_impl->assembly;
}

MLABecPreconditioner
MLABecLapAMG::preconditioner () const noexcept
{
    return m_impl->selected_preconditioner;
}

char const*
MLABecLapAMG::preconditioner_name () const noexcept
{
    return fld_test::preconditioner_name(m_impl->selected_preconditioner);
}

MLABecAMGBackend
MLABecLapAMG::amg_backend () const noexcept
{
    return m_impl->selected_amg_backend;
}

char const*
MLABecLapAMG::amg_backend_name () const noexcept
{
    return m_impl->selected_preconditioner == MLABecPreconditioner::AMG
               ? fld_test::amg_backend_name(m_impl->selected_amg_backend)
               : "n/a";
}

double
MLABecLapAMG::setup_seconds () const noexcept
{
    return m_impl->last_setup_seconds;
}

} // namespace fld_test
