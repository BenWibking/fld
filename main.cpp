#include <AMReX.H>
#include <AMReX_AMG.H>
#include <AMReX_AlgVecUtil.H>
#include <AMReX_GMRES_MV.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_SpMV.H>

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

using namespace amrex;

namespace
{

using Entry = std::pair<Long, Real>;
using Entries = Vector<Entry>;

constexpr int xlo = 0;
constexpr int xhi = 1;
constexpr int ylo = 2;
constexpr int yhi = 3;

enum class BoundaryKind { periodic, reflecting, dirichlet, marshak };

struct BoundaryCondition
{
    BoundaryKind kind = BoundaryKind::reflecting;
    Real value = Real(0);
    Real beta = Real(0);
};

struct Face
{
    Long neighbor = -1;
    Real area = Real(0);
    Real self_distance = Real(0);
    Real neighbor_distance = Real(0);
    int normal_x = 0;
    int normal_y = 0;
    int side = -1;
    BoundaryCondition boundary;
};

struct Cell
{
    Long id = -1;
    int ilo = 0;
    int ihi = 0;
    int jlo = 0;
    int jhi = 0;
    Real x = Real(0);
    Real y = Real(0);
    Real hx = Real(0);
    Real hy = Real(0);
    Real volume = Real(0);
    Vector<Face> faces;
};

struct Mesh
{
    int fine_n = 0;
    Real fine_h = Real(0);
    Vector<Cell> cells;
    Vector<Long> owner;
    std::array<BoundaryCondition, 4> outer_boundary;
    BoundaryCondition hole_boundary;
    bool has_hole = false;
};

struct CloudMesh
{
    Mesh mesh;
    Vector<int> level_n;
    Vector<std::pair<int, int>> level_y_bounds;
};

struct LinearSolution
{
    Vector<Real> values;
    int iterations = 0;
    Real relative_residual = Real(0);
};

struct SolverSummary
{
    int solves = 0;
    int total_iterations = 0;
    int maximum_iterations = 0;
    int maximum_levels = 0;
    Real maximum_relative_residual = Real(0);
    double setup_seconds = 0.0;
    double maximum_operator_complexity = 0.0;

    [[nodiscard]] Real
    average_iterations () const noexcept
    {
        return (solves > 0) ? Real(total_iterations) / Real(solves) : Real(0);
    }
};

struct GaussianResult
{
    Real relative_l1_error = Real(0);
    Real relative_energy_drift = Real(0);
    Long cells = 0;
    SolverSummary solver;
};

struct CloudResult
{
    Real transmission = Real(0);
    Real balance_error = Real(0);
    Real cloudy_area_relative_error = Real(0);
    Real minimum_energy = Real(0);
    Real maximum_energy = Real(0);
    Real final_nonlinear_change = Real(0);
    int nonlinear_iterations = 0;
    int anderson_steps = 0;
    int anderson_restarts = 0;
    Long mixed_cells = 0;
    Long cells = 0;
    SolverSummary solver;
};

struct FrontResult
{
    Real front_radius = Real(0);
    Real causal_radius = Real(0);
    Real far_excess = Real(0);
    Real maximum_flux_fraction = Real(0);
    Real minimum_energy = Real(0);
    Real maximum_energy = Real(0);
    Real unlimited_far_excess = Real(0);
    Real final_picard_change = Real(0);
    int total_picard_iterations = 0;
    int maximum_picard_iterations = 0;
    Long cells = 0;
    SolverSummary solver;
};

Real
linear_tolerance ()
{
    return (sizeof(Real) == sizeof(float)) ? Real(5.e-5) : Real(2.e-10);
}

BoundaryCondition
periodic_boundary ()
{
    return {BoundaryKind::periodic, Real(0), Real(0)};
}

BoundaryCondition
reflecting_boundary ()
{
    return {BoundaryKind::reflecting, Real(0), Real(0)};
}

BoundaryCondition
dirichlet_boundary (Real value)
{
    return {BoundaryKind::dirichlet, value, Real(0)};
}

BoundaryCondition
marshak_boundary (Real equilibrium_energy, Real beta)
{
    return {BoundaryKind::marshak, equilibrium_energy, beta};
}

template <typename RefinePredicate, typename ActivePredicate>
Mesh
make_nested_mesh (int nbase, Vector<int> const& refinement_ratios,
                  RefinePredicate&& refine, ActivePredicate&& active,
                  std::array<BoundaryCondition, 4> outer_boundary,
                  bool has_hole = false,
                  BoundaryCondition hole_boundary = {})
{
    AMREX_ALWAYS_ASSERT(nbase > 1);

    Mesh mesh;
    mesh.fine_n = nbase;
    for (int const refinement_ratio : refinement_ratios) {
        AMREX_ALWAYS_ASSERT(refinement_ratio > 0);
        mesh.fine_n *= refinement_ratio;
    }
    mesh.fine_h = Real(1) / Real(mesh.fine_n);
    mesh.owner.resize(static_cast<std::size_t>(mesh.fine_n) * mesh.fine_n,
                      Long(-1));
    mesh.outer_boundary = outer_boundary;
    mesh.has_hole = has_hole;
    mesh.hole_boundary = hole_boundary;

    // Represent composite cells on one finest-level integer lattice so every
    // coarse-fine face can be split into conservative subfaces.
    auto owner_index = [&] (int i, int j) -> std::size_t
    { return static_cast<std::size_t>(j) * mesh.fine_n + i; };

    auto add_cell = [&] (int ilo, int ihi, int jlo, int jhi)
    {
        Cell cell;
        cell.id = static_cast<Long>(mesh.cells.size());
        cell.ilo = ilo;
        cell.ihi = ihi;
        cell.jlo = jlo;
        cell.jhi = jhi;
        cell.hx = Real(ihi - ilo) * mesh.fine_h;
        cell.hy = Real(jhi - jlo) * mesh.fine_h;
        cell.x = Real(0.5) * Real(ilo + ihi) * mesh.fine_h;
        cell.y = Real(0.5) * Real(jlo + jhi) * mesh.fine_h;
        cell.volume = cell.hx * cell.hy;
        mesh.cells.push_back(cell);

        for (int j = jlo; j < jhi; ++j) {
            for (int i = ilo; i < ihi; ++i) {
                auto const index = owner_index(i, j);
                AMREX_ALWAYS_ASSERT(mesh.owner[index] == Long(-1));
                mesh.owner[index] = cell.id;
            }
        }
    };

    auto add_level_cell = [&] (auto&& self, int level, int i, int j,
                               int level_n) -> void
    {
        bool const refined =
            level < static_cast<int>(refinement_ratios.size()) &&
            refinement_ratios[level] > 1 &&
            refine(level, i, j, level_n);
        if (refined) {
            int const refinement_ratio = refinement_ratios[level];
            for (int jj = 0; jj < refinement_ratio; ++jj) {
                for (int ii = 0; ii < refinement_ratio; ++ii) {
                    self(self, level + 1, i * refinement_ratio + ii,
                         j * refinement_ratio + jj,
                         level_n * refinement_ratio);
                }
            }
            return;
        }

        AMREX_ALWAYS_ASSERT(mesh.fine_n % level_n == 0);
        int const scale = mesh.fine_n / level_n;
        int const ilo = i * scale;
        int const ihi = (i + 1) * scale;
        int const jlo = j * scale;
        int const jhi = (j + 1) * scale;
        Real const x = Real(0.5) * Real(ilo + ihi) * mesh.fine_h;
        Real const y = Real(0.5) * Real(jlo + jhi) * mesh.fine_h;
        if (active(x, y)) {
            add_cell(ilo, ihi, jlo, jhi);
        }
    };

    for (int j = 0; j < nbase; ++j) {
        for (int i = 0; i < nbase; ++i) {
            add_level_cell(add_level_cell, 0, i, j, nbase);
        }
    }

    auto add_face = [&] (Cell& cell, int ni, int nj, int side, int normal_x,
                         int normal_y, Real area)
    {
        bool outside =
            ni < 0 || ni >= mesh.fine_n || nj < 0 || nj >= mesh.fine_n;
        BoundaryCondition boundary;

        if (outside) {
            boundary = mesh.outer_boundary[side];
            if (boundary.kind == BoundaryKind::periodic) {
                if (ni < 0) {
                    ni += mesh.fine_n;
                } else if (ni >= mesh.fine_n) {
                    ni -= mesh.fine_n;
                }
                if (nj < 0) {
                    nj += mesh.fine_n;
                } else if (nj >= mesh.fine_n) {
                    nj -= mesh.fine_n;
                }
                outside = false;
            }
        }

        Face face;
        face.area = area;
        face.self_distance =
            (normal_x != 0) ? Real(0.5) * cell.hx : Real(0.5) * cell.hy;
        face.normal_x = normal_x;
        face.normal_y = normal_y;
        face.side = side;

        if (!outside) {
            Long const neighbor = mesh.owner[owner_index(ni, nj)];
            if (neighbor >= 0) {
                if (neighbor == cell.id) {
                    return;
                }
                face.neighbor = neighbor;
                auto const& other = mesh.cells[neighbor];
                face.neighbor_distance = (normal_x != 0) ? Real(0.5) * other.hx
                                                         : Real(0.5) * other.hy;
            } else {
                face.boundary =
                    mesh.has_hole ? mesh.hole_boundary : reflecting_boundary();
            }
        } else {
            face.boundary = boundary;
        }
        cell.faces.push_back(face);
    };

    for (auto& cell : mesh.cells) {
        for (int j = cell.jlo; j < cell.jhi; ++j) {
            add_face(cell, cell.ilo - 1, j, xlo, -1, 0, mesh.fine_h);
            add_face(cell, cell.ihi, j, xhi, 1, 0, mesh.fine_h);
        }
        for (int i = cell.ilo; i < cell.ihi; ++i) {
            add_face(cell, i, cell.jlo - 1, ylo, 0, -1, mesh.fine_h);
            add_face(cell, i, cell.jhi, yhi, 0, 1, mesh.fine_h);
        }
    }

    AMREX_ALWAYS_ASSERT(mesh.cells.size() > 9);
    return mesh;
}

template <typename RefinePredicate, typename ActivePredicate>
Mesh
make_mesh (int nbase, int refinement_ratio, RefinePredicate&& refine,
           ActivePredicate&& active,
           std::array<BoundaryCondition, 4> outer_boundary,
           bool has_hole = false, BoundaryCondition hole_boundary = {})
{
    return make_nested_mesh(
        nbase, Vector<int>{refinement_ratio},
        [&] (int, int i, int j, int n) noexcept
        { return refine(i, j, n); },
        std::forward<ActivePredicate>(active), outer_boundary, has_hole,
        hole_boundary);
}

template <typename F>
SpMatrix<Real>
make_matrix (AlgPartition const& partition, F&& make_row)
{
    using host_csr_type = CSR<Real, Gpu::PinnedVector>;
    using csr_type = SpMatrix<Real>::csr_type;

    int const myproc = ParallelDescriptor::MyProc();
    Long const begin = partition[myproc];
    Long const end = partition[myproc + 1];
    Long const nlocal = end - begin;
    host_csr_type host;
    host.row_offset.resize(nlocal + 1);
    host.row_offset[0] = 0;

    for (Long i = 0; i < nlocal; ++i) {
        Entries entries = make_row(begin + i);
        std::sort(entries.begin(), entries.end(),
                  [] (Entry const& lhs, Entry const& rhs)
                  { return lhs.first < rhs.first; });
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
            }
        }
        host.row_offset[i + 1] = static_cast<Long>(host.mat.size());
    }
    host.nnz = static_cast<Long>(host.mat.size());

    csr_type device;
    duplicateCSR(Gpu::hostToDevice, device, host);
    Gpu::streamSynchronize();
    return SpMatrix<Real>(partition, partition, std::move(device));
}

AlgVector<Real>
make_vector (AlgPartition const& partition, Vector<Real> const& values)
{
    AMREX_ALWAYS_ASSERT(static_cast<Long>(values.size()) ==
                        partition.numGlobalRows());
    AlgVector<Real> result(partition);
    Long const begin = result.globalBegin();
    Gpu::PinnedVector<Real> local(result.numLocalRows());
    for (Long i = 0; i < result.numLocalRows(); ++i) {
        local[i] = values[begin + i];
    }
    if (!local.empty()) {
        Gpu::copyAsync(Gpu::hostToDevice, local.begin(), local.end(),
                       result.data());
        Gpu::streamSynchronize();
    }
    return result;
}

Vector<Real>
gather_vector (AlgVector<Real> const& vector)
{
    Long const nlocal = vector.numLocalRows();
    Gpu::PinnedVector<Real> local(nlocal);
    if (nlocal > 0) {
        Gpu::copyAsync(Gpu::deviceToHost, vector.data(), vector.data() + nlocal,
                       local.begin());
        Gpu::streamSynchronize();
    }

    auto const& partition = vector.partition();
    int const nprocs = ParallelDescriptor::NProcs();
    std::vector<int> counts(nprocs);
    std::vector<int> offsets(nprocs);
    for (int rank = 0; rank < nprocs; ++rank) {
        Long const count = partition[rank + 1] - partition[rank];
        AMREX_ALWAYS_ASSERT(count <= std::numeric_limits<int>::max());
        AMREX_ALWAYS_ASSERT(partition[rank] <= std::numeric_limits<int>::max());
        counts[rank] = static_cast<int>(count);
        offsets[rank] = static_cast<int>(partition[rank]);
    }

    Vector<Real> global(partition.numGlobalRows());
    ParallelDescriptor::Gatherv(local.data(), static_cast<int>(nlocal),
                                global.data(), counts, offsets,
                                ParallelDescriptor::IOProcessorNumber());
    ParallelDescriptor::Bcast(global.data(), global.size(),
                              ParallelDescriptor::IOProcessorNumber());
    return global;
}

Real
true_relative_residual (SpMatrix<Real> const& matrix,
                        AlgVector<Real> const& solution,
                        AlgVector<Real> const& rhs)
{
    AlgVector<Real> residual(rhs.partition());
    SpMV(residual, matrix, solution);
    LinComb(residual, Real(1), rhs, Real(-1), residual);
    Real const denominator = amrex::max(rhs.norm2(), Real(1.e-30));
    return residual.norm2() / denominator;
}

class AMGGMRESSolver
{
  public:
    explicit AMGGMRESSolver (SpMatrix<Real> const& matrix,
                             AMG<Real>::Options options = {})
        : m_matrix(matrix), m_amg(matrix, options), m_gmres(&matrix)
    {
        m_amg.setup();
        m_gmres.setPrecond(
            [this] (AlgVector<Real>& lhs, AlgVector<Real> const& rhs)
            { m_amg.apply(lhs, rhs); });
        m_gmres.getGMRES().setRestartLength(50);
        m_gmres.getGMRES().setMaxIters(500);
    }

    AMGGMRESSolver (AMGGMRESSolver const&) = delete;
    AMGGMRESSolver& operator=(AMGGMRESSolver const&) = delete;

    [[nodiscard]] LinearSolution
    solve (Vector<Real> const& rhs_values)
    {
        AlgVector<Real> rhs = make_vector(m_matrix.partition(), rhs_values);
        AlgVector<Real> solution(m_matrix.partition());
        solution.setVal(Real(0));
        m_gmres.solve(solution, rhs, linear_tolerance(), Real(0));

        auto const& gmres = m_gmres.getGMRES();
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            gmres.getStatus() == 0,
            "GMRES+AMG did not converge in an FLD regression");

        LinearSolution result;
        result.values = gather_vector(solution);
        result.iterations = gmres.getNumIters();
        result.relative_residual =
            true_relative_residual(m_matrix, solution, rhs);
        AMREX_ALWAYS_ASSERT(result.relative_residual <=
                            Real(5) * linear_tolerance());
        return result;
    }

    [[nodiscard]] AMG<Real>::Diagnostics const&
    diagnostics () const noexcept
    {
        return m_amg.diagnostics();
    }

  private:
    SpMatrix<Real> const& m_matrix;
    AMG<Real> m_amg;
    GMRES_MV<Real> m_gmres;
};

void
record_setup (SolverSummary& summary, AMGGMRESSolver const& solver)
{
    auto const& diagnostics = solver.diagnostics();
    summary.setup_seconds += diagnostics.setup_seconds;
    summary.maximum_levels = amrex::max(
        summary.maximum_levels, static_cast<int>(diagnostics.levels.size()));
    summary.maximum_operator_complexity = amrex::max(
        summary.maximum_operator_complexity, diagnostics.operator_complexity);
}

void
record_solve (SolverSummary& summary, LinearSolution const& solution)
{
    ++summary.solves;
    summary.total_iterations += solution.iterations;
    summary.maximum_iterations =
        amrex::max(summary.maximum_iterations, solution.iterations);
    summary.maximum_relative_residual = amrex::max(
        summary.maximum_relative_residual, solution.relative_residual);
}

Real
face_transmissibility (Face const& face, Long row,
                       Vector<Real> const& diffusion,
                       Vector<Real> const* rosseland_extinction = nullptr)
{
    Real const dself = diffusion[row];
    AMREX_ALWAYS_ASSERT(dself > Real(0));
    if (face.neighbor >= 0) {
        Real const dother = diffusion[face.neighbor];
        AMREX_ALWAYS_ASSERT(dother > Real(0));

        if (rosseland_extinction != nullptr) {
            auto const& extinction = *rosseland_extinction;
            AMREX_ALWAYS_ASSERT(extinction.size() == diffusion.size());
            Real const chi_self = extinction[row];
            Real const chi_other = extinction[face.neighbor];
            AMREX_ALWAYS_ASSERT(chi_self > Real(0));
            AMREX_ALWAYS_ASSERT(chi_other > Real(0));

            // Howell & Greenough Fig. 4: average the Rosseland extinction to
            // the face using the surface-flux formula.  The limiter remains
            // cell centered, so average lambda = D chi in series separately.
            Real const center_distance =
                face.self_distance + face.neighbor_distance;
            Real const arithmetic_mean =
                Real(0.5) * (chi_self + chi_other);
            Real const harmonic_mean =
                Real(2) * chi_self * chi_other / (chi_self + chi_other);
            Real const surface_scale =
                Real(4) / (Real(3) * center_distance);
            Real const face_extinction =
                amrex::min(arithmetic_mean,
                           amrex::max(harmonic_mean, surface_scale));
            Real const lambda_self = dself * chi_self;
            Real const lambda_other = dother * chi_other;
            // Use the less restrictive of the two cell-centered limiters at
            // the face with the Howell-Greenough surface opacity.
            Real const face_limiter =
                amrex::max(lambda_self, lambda_other);
            return face.area * face_limiter /
                   (center_distance * face_extinction);
        }

        // The two half-cell diffusion resistances are in series.  This gives
        // the harmonic face coefficient and preserves flux across AMR
        // interfaces.
        return face.area /
               (face.self_distance / dself + face.neighbor_distance / dother);
    }

    switch (face.boundary.kind) {
    case BoundaryKind::reflecting:
        return Real(0);
    case BoundaryKind::dirichlet:
        return face.area * dself / face.self_distance;
    case BoundaryKind::marshak:
        AMREX_ALWAYS_ASSERT(face.boundary.beta > Real(0));
        return face.area /
               (face.self_distance / dself + Real(1) / face.boundary.beta);
    case BoundaryKind::periodic:
        amrex::Abort("Unresolved periodic FLD boundary face");
        return Real(0);
    }
    return Real(0);
}

struct LinearSystem
{
    SpMatrix<Real> matrix;
    Vector<Real> rhs;
};

LinearSystem
assemble_system (Mesh const& mesh, Vector<Real> const& diffusion,
                 Vector<Real> const& old_state, Real dt, bool transient,
                 Vector<Real> const* rosseland_extinction = nullptr)
{
    Long const n = static_cast<Long>(mesh.cells.size());
    AMREX_ALWAYS_ASSERT(static_cast<Long>(diffusion.size()) == n);
    if (rosseland_extinction != nullptr) {
        AMREX_ALWAYS_ASSERT(
            static_cast<Long>(rosseland_extinction->size()) == n);
    }
    if (transient) {
        AMREX_ALWAYS_ASSERT(static_cast<Long>(old_state.size()) == n);
        AMREX_ALWAYS_ASSERT(dt > Real(0));
    }

    Vector<Entries> rows(n);
    Vector<Real> rhs(n, Real(0));
    for (Long row = 0; row < n; ++row) {
        auto const& cell = mesh.cells[row];
        Real diagonal = transient ? Real(1) : Real(0);
        rhs[row] = transient ? old_state[row] : Real(0);
        Real const scale = (transient ? dt : Real(1)) / cell.volume;

        for (auto const& face : cell.faces) {
            Real const transmissibility =
                face_transmissibility(face, row, diffusion,
                                      rosseland_extinction);
            if (transmissibility == Real(0)) {
                continue;
            }
            Real const coefficient = scale * transmissibility;
            diagonal += coefficient;
            if (face.neighbor >= 0) {
                rows[row].emplace_back(face.neighbor, -coefficient);
            } else {
                rhs[row] += coefficient * face.boundary.value;
            }
        }
        AMREX_ALWAYS_ASSERT(diagonal > Real(0));
        rows[row].emplace_back(row, diagonal);
    }

    AlgPartition partition(n);
    auto matrix =
        make_matrix(partition, [&] (Long row) -> Entries { return rows[row]; });
    return {std::move(matrix), std::move(rhs)};
}

Vector<std::array<Real, 2>>
cell_gradients (Mesh const& mesh, Vector<Real> const& energy)
{
    AMREX_ALWAYS_ASSERT(energy.size() == mesh.cells.size());
    Vector<std::array<Real, 2>> gradient(mesh.cells.size(), {Real(0), Real(0)});

    for (Long row = 0; row < static_cast<Long>(mesh.cells.size()); ++row) {
        auto const& cell = mesh.cells[row];
        for (auto const& face : cell.faces) {
            Real face_energy = energy[row];
            if (face.neighbor >= 0) {
                face_energy = (face.neighbor_distance * energy[row] +
                               face.self_distance * energy[face.neighbor]) /
                              (face.self_distance + face.neighbor_distance);
            } else if (face.boundary.kind == BoundaryKind::dirichlet ||
                       face.boundary.kind == BoundaryKind::marshak) {
                face_energy = face.boundary.value;
            }
            gradient[row][0] +=
                Real(face.normal_x) * face.area * face_energy / cell.volume;
            gradient[row][1] +=
                Real(face.normal_y) * face.area * face_energy / cell.volume;
        }
    }
    return gradient;
}

Real
levermore_pomraning_limiter (Real r)
{
    AMREX_ALWAYS_ASSERT(r >= Real(0));
    if (r > Real(1.e8)) {
        return Real(1) / r;
    }
    return (Real(2) + r) / (Real(6) + Real(3) * r + r * r);
}

Vector<Real>
compute_diffusion (Mesh const& mesh, Vector<Real> const& energy,
                   Vector<Real> const& extinction, bool limited,
                   Real* maximum_flux_fraction = nullptr)
{
    AMREX_ALWAYS_ASSERT(energy.size() == mesh.cells.size());
    AMREX_ALWAYS_ASSERT(extinction.size() == mesh.cells.size());
    auto const gradient = cell_gradients(mesh, energy);
    Vector<Real> diffusion(mesh.cells.size());
    Real maximum_fraction = Real(0);

    for (Long row = 0; row < static_cast<Long>(mesh.cells.size()); ++row) {
        Real const chi = extinction[row];
        Real const e = amrex::max(energy[row], Real(1.e-30));
        AMREX_ALWAYS_ASSERT(chi > Real(0));
        Real const magnitude = std::hypot(gradient[row][0], gradient[row][1]);
        Real const r = magnitude / (chi * e);
        Real const lambda =
            limited ? levermore_pomraning_limiter(r) : Real(1) / Real(3);
        // Radiation speed is normalized to one, so lambda*R is |F|/(c E).
        diffusion[row] = lambda / chi;
        maximum_fraction = amrex::max(maximum_fraction, lambda * r);
    }
    if (maximum_flux_fraction != nullptr) {
        *maximum_flux_fraction = maximum_fraction;
    }
    return diffusion;
}

std::pair<Real, Real>
minimum_maximum (Vector<Real> const& values)
{
    auto const [minimum, maximum] =
        std::minmax_element(values.begin(), values.end());
    return {*minimum, *maximum};
}

Real
maximum_relative_change (Vector<Real> const& lhs, Vector<Real> const& rhs)
{
    AMREX_ALWAYS_ASSERT(lhs.size() == rhs.size());
    Real result = Real(0);
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        Real const scale = amrex::max(
            amrex::max(std::abs(lhs[i]), std::abs(rhs[i])), Real(1.e-12));
        result = amrex::max(result, std::abs(lhs[i] - rhs[i]) / scale);
    }
    return result;
}

// Adapt a small host dense matrix to the generic AMReX GMRES interface.  This
// reuses GMRES's Arnoldi, Givens rotation, and back-substitution machinery for
// the regularized Anderson normal equations.
class DenseMatrixOperator
{
  public:
    using RT = Real;

    explicit DenseMatrixOperator (Vector<Vector<Real>> matrix)
        : m_matrix(std::move(matrix)),
          m_inverse_diagonal(m_matrix.size(), Real(1))
    {
        AMREX_ALWAYS_ASSERT(!m_matrix.empty());
        for (std::size_t i = 0; i < m_matrix.size(); ++i) {
            AMREX_ALWAYS_ASSERT(m_matrix[i].size() == m_matrix.size());
            AMREX_ALWAYS_ASSERT(m_matrix[i][i] > Real(0));
            m_inverse_diagonal[i] = Real(1) / m_matrix[i][i];
        }
    }

    void
    apply (Vector<Real>& lhs, Vector<Real> const& rhs) const
    {
        AMREX_ALWAYS_ASSERT(rhs.size() == m_matrix.size());
        lhs.assign(m_matrix.size(), Real(0));
        for (std::size_t i = 0; i < m_matrix.size(); ++i) {
            for (std::size_t j = 0; j < m_matrix.size(); ++j) {
                lhs[i] += m_matrix[i][j] * rhs[j];
            }
        }
    }

    static void
    assign (Vector<Real>& lhs, Vector<Real> const& rhs)
    {
        lhs = rhs;
    }

    static Real
    dotProduct (Vector<Real> const& lhs, Vector<Real> const& rhs)
    {
        AMREX_ALWAYS_ASSERT(lhs.size() == rhs.size());
        return std::inner_product(lhs.begin(), lhs.end(), rhs.begin(),
                                  Real(0));
    }

    static void
    increment (Vector<Real>& lhs, Vector<Real> const& rhs, Real scale)
    {
        AMREX_ALWAYS_ASSERT(lhs.size() == rhs.size());
        for (std::size_t i = 0; i < lhs.size(); ++i) {
            lhs[i] += scale * rhs[i];
        }
    }

    static void
    linComb (Vector<Real>& lhs, Real lhs_scale,
             Vector<Real> const& lhs_vector, Real rhs_scale,
             Vector<Real> const& rhs_vector)
    {
        AMREX_ALWAYS_ASSERT(lhs_vector.size() == rhs_vector.size());
        lhs.resize(lhs_vector.size());
        for (std::size_t i = 0; i < lhs.size(); ++i) {
            lhs[i] =
                lhs_scale * lhs_vector[i] + rhs_scale * rhs_vector[i];
        }
    }

    [[nodiscard]] Vector<Real>
    makeVecRHS () const
    {
        return Vector<Real>(m_matrix.size(), Real(0));
    }

    [[nodiscard]] Vector<Real>
    makeVecLHS () const
    {
        return Vector<Real>(m_matrix.size(), Real(0));
    }

    static Real
    norm2 (Vector<Real> const& vector)
    {
        return std::sqrt(dotProduct(vector, vector));
    }

    void
    precond (Vector<Real>& lhs, Vector<Real> const& rhs) const
    {
        AMREX_ALWAYS_ASSERT(rhs.size() == m_inverse_diagonal.size());
        lhs.resize(rhs.size());
        for (std::size_t i = 0; i < rhs.size(); ++i) {
            lhs[i] = m_inverse_diagonal[i] * rhs[i];
        }
    }

    static void
    scale (Vector<Real>& vector, Real factor)
    {
        for (Real& value : vector) {
            value *= factor;
        }
    }

    static void
    setToZero (Vector<Real>& vector)
    {
        std::fill(vector.begin(), vector.end(), Real(0));
    }

  private:
    Vector<Vector<Real>> m_matrix;
    Vector<Real> m_inverse_diagonal;
};

bool
solve_anderson_coefficients (Vector<Vector<Real>> matrix,
                             Vector<Real> const& rhs,
                             Vector<Real>& coefficients)
{
    AMREX_ALWAYS_ASSERT(!matrix.empty());
    AMREX_ALWAYS_ASSERT(matrix.size() == rhs.size());

    DenseMatrixOperator linear_operator(std::move(matrix));
    GMRES<Vector<Real>, DenseMatrixOperator> gmres;
    gmres.define(linear_operator);
    int const dimension = static_cast<int>(rhs.size());
    gmres.setRestartLength(dimension);
    gmres.setMaxIters(2 * dimension);
    coefficients.assign(rhs.size(), Real(0));
    Real const tolerance =
        (sizeof(Real) == sizeof(float)) ? Real(2.e-4) : Real(1.e-10);
    gmres.solve(coefficients, rhs, tolerance, Real(0), 2 * dimension);
    if (gmres.getStatus() != 0) {
        return false;
    }
    return std::all_of(coefficients.begin(), coefficients.end(),
                       [] (Real value) noexcept
                       { return std::isfinite(value); });
}

class AndersonMixer
{
  public:
    AndersonMixer (int depth, Real beta, Vector<Real> weights,
                   Real upper_bound)
        : m_depth(depth), m_beta(beta),
          m_weights(std::move(weights)), m_upper_bound(upper_bound)
    {
        AMREX_ALWAYS_ASSERT(m_depth >= 0);
        AMREX_ALWAYS_ASSERT(m_beta > Real(0));
        AMREX_ALWAYS_ASSERT(m_beta <= Real(1));
        AMREX_ALWAYS_ASSERT(m_upper_bound > Real(0));
        AMREX_ALWAYS_ASSERT(!m_weights.empty());
        AMREX_ALWAYS_ASSERT(std::all_of(
            m_weights.begin(), m_weights.end(),
            [] (Real weight) noexcept { return weight > Real(0); }));
    }

    [[nodiscard]] Vector<Real>
    update (Vector<Real> const& state, Vector<Real> const& fixed_point)
    {
        AMREX_ALWAYS_ASSERT(state.size() == m_weights.size());
        AMREX_ALWAYS_ASSERT(fixed_point.size() == state.size());

        Vector<Real> residual(state.size());
        Vector<Real> picard_state(state.size());
        for (std::size_t i = 0; i < state.size(); ++i) {
            residual[i] = fixed_point[i] - state[i];
            picard_state[i] = state[i] + m_beta * residual[i];
        }
        if (m_depth == 0) {
            return picard_state;
        }

        Real const residual_norm = weighted_norm(residual);
        if (!m_residuals.empty() &&
            residual_norm > Real(2) * weighted_norm(m_residuals.back())) {
            restart(state, residual);
            return picard_state;
        }

        m_states.push_back(state);
        m_residuals.push_back(residual);
        while (static_cast<int>(m_states.size()) > m_depth + 1) {
            m_states.erase(m_states.begin());
            m_residuals.erase(m_residuals.begin());
        }

        int const difference_count =
            static_cast<int>(m_states.size()) - 1;
        if (difference_count == 0) {
            return picard_state;
        }

        Vector<Vector<Real>> state_differences(
            difference_count, Vector<Real>(state.size()));
        Vector<Vector<Real>> residual_differences(
            difference_count, Vector<Real>(state.size()));
        for (int column = 0; column < difference_count; ++column) {
            for (std::size_t i = 0; i < state.size(); ++i) {
                state_differences[column][i] =
                    m_states[column + 1][i] - m_states[column][i];
                residual_differences[column][i] =
                    m_residuals[column + 1][i] - m_residuals[column][i];
            }
        }

        Vector<Vector<Real>> normal_matrix(
            difference_count, Vector<Real>(difference_count));
        Vector<Real> normal_rhs(difference_count);
        // Minimize ||f_k - Delta F gamma|| in the volume-weighted L2 norm.
        Real trace = Real(0);
        for (int i = 0; i < difference_count; ++i) {
            normal_rhs[i] = weighted_dot(residual_differences[i], residual);
            for (int j = 0; j < difference_count; ++j) {
                normal_matrix[i][j] = weighted_dot(
                    residual_differences[i], residual_differences[j]);
            }
            trace += normal_matrix[i][i];
        }
        Real const regularization_factor =
            (sizeof(Real) == sizeof(float)) ? Real(1.e-5) : Real(1.e-12);
        Real const regularization =
            regularization_factor *
            amrex::max(trace / Real(difference_count), Real(1.e-30));
        for (int i = 0; i < difference_count; ++i) {
            normal_matrix[i][i] += regularization;
        }

        Vector<Real> coefficients;
        if (!solve_anderson_coefficients(std::move(normal_matrix), normal_rhs,
                                         coefficients)) {
            restart(state, residual);
            return picard_state;
        }

        Vector<Real> candidate = picard_state;
        // Type-II Anderson update:
        // x_{k+1} = x_k + beta f_k - (Delta X + beta Delta F) gamma.
        for (int column = 0; column < difference_count; ++column) {
            for (std::size_t i = 0; i < candidate.size(); ++i) {
                candidate[i] -= coefficients[column] *
                                (state_differences[column][i] +
                                 m_beta * residual_differences[column][i]);
            }
        }

        Vector<Real> candidate_step(state.size());
        Vector<Real> picard_step(state.size());
        bool physical = true;
        for (std::size_t i = 0; i < state.size(); ++i) {
            candidate_step[i] = candidate[i] - state[i];
            picard_step[i] = picard_state[i] - state[i];
            physical = physical && std::isfinite(candidate[i]) &&
                       candidate[i] >= Real(0) &&
                       candidate[i] <= m_upper_bound;
        }
        Real const picard_step_norm = weighted_norm(picard_step);
        Real const candidate_step_norm = weighted_norm(candidate_step);
        // Reject proposals that violate the radiation maximum principle or
        // take a much larger step than the underlying fixed-point update.
        if (!physical || !std::isfinite(candidate_step_norm) ||
            candidate_step_norm >
                Real(10) * amrex::max(picard_step_norm, Real(1.e-30))) {
            restart(state, residual);
            return picard_state;
        }

        ++m_anderson_steps;
        return candidate;
    }

    [[nodiscard]] int
    anderson_steps () const noexcept
    {
        return m_anderson_steps;
    }

    [[nodiscard]] int
    restarts () const noexcept
    {
        return m_restarts;
    }

  private:
    [[nodiscard]] Real
    weighted_dot (Vector<Real> const& lhs, Vector<Real> const& rhs) const
    {
        AMREX_ALWAYS_ASSERT(lhs.size() == m_weights.size());
        AMREX_ALWAYS_ASSERT(rhs.size() == lhs.size());
        Real result = Real(0);
        for (std::size_t i = 0; i < lhs.size(); ++i) {
            result += m_weights[i] * lhs[i] * rhs[i];
        }
        return result;
    }

    [[nodiscard]] Real
    weighted_norm (Vector<Real> const& vector) const
    {
        return std::sqrt(amrex::max(weighted_dot(vector, vector), Real(0)));
    }

    void
    restart (Vector<Real> const& state, Vector<Real> const& residual)
    {
        m_states.clear();
        m_residuals.clear();
        m_states.push_back(state);
        m_residuals.push_back(residual);
        ++m_restarts;
    }

    int m_depth = 0;
    Real m_beta = Real(1);
    Vector<Real> m_weights;
    Real m_upper_bound = std::numeric_limits<Real>::max();
    Vector<Vector<Real>> m_states;
    Vector<Vector<Real>> m_residuals;
    int m_anderson_steps = 0;
    int m_restarts = 0;
};

GaussianResult
run_gaussian (bool use_amr)
{
    auto periodic = periodic_boundary();
    std::array<BoundaryCondition, 4> boundary{periodic, periodic, periodic,
                                              periodic};

    int const nbase = use_amr ? 32 : 64;
    int const ratio = use_amr ? 2 : 1;
    Mesh mesh = make_mesh(
        nbase, ratio, [] (int i, int j, int n) noexcept
        { return i >= n / 4 && i < 3 * n / 4 && j >= n / 4 && j < 3 * n / 4; },
        [] (Real, Real) noexcept { return true; }, boundary);

    Real constexpr extinction_value = Real(100);
    Real const diffusion_value = Real(1) / (Real(3) * extinction_value);
    Real constexpr initial_time = Real(0.4);
    Real constexpr dt = Real(0.005);
    int constexpr steps = 10;
    Real constexpr pi = Real(3.1415926535897932384626433832795);

    auto exact = [&] (Cell const& cell, Real time) -> Real
    {
        Real const x = cell.x - Real(0.5);
        Real const y = cell.y - Real(0.5);
        Real const radius_squared = x * x + y * y;
        return std::exp(-radius_squared / (Real(4) * diffusion_value * time)) /
               (Real(4) * pi * diffusion_value * time);
    };

    Vector<Real> state(mesh.cells.size());
    Vector<Real> extinction(mesh.cells.size(), extinction_value);
    for (Long row = 0; row < static_cast<Long>(mesh.cells.size()); ++row) {
        state[row] = exact(mesh.cells[row], initial_time);
    }
    Real const initial_energy = std::inner_product(
        state.begin(), state.end(), mesh.cells.begin(), Real(0), std::plus<>(),
        [] (Real energy, Cell const& cell) { return energy * cell.volume; });

    auto diffusion = compute_diffusion(mesh, state, extinction, false);
    auto system = assemble_system(mesh, diffusion, state, dt, true);
    AMGGMRESSolver solver(system.matrix);
    GaussianResult result;
    result.cells = static_cast<Long>(mesh.cells.size());
    record_setup(result.solver, solver);

    for (int step = 0; step < steps; ++step) {
        auto solution = solver.solve(state);
        record_solve(result.solver, solution);
        state = std::move(solution.values);
    }

    Real const final_time = initial_time + Real(steps) * dt;
    Real absolute_error = Real(0);
    Real exact_norm = Real(0);
    Real final_energy = Real(0);
    for (Long row = 0; row < static_cast<Long>(mesh.cells.size()); ++row) {
        Real const reference = exact(mesh.cells[row], final_time);
        Real const volume = mesh.cells[row].volume;
        absolute_error += volume * std::abs(state[row] - reference);
        exact_norm += volume * std::abs(reference);
        final_energy += volume * state[row];
    }
    result.relative_l1_error = absolute_error / exact_norm;
    result.relative_energy_drift =
        std::abs(final_energy - initial_energy) / initial_energy;

    Real const error_limit =
        (sizeof(Real) == sizeof(float)) ? Real(0.12) : Real(0.06);
    Real const conservation_limit =
        (sizeof(Real) == sizeof(float)) ? Real(2.e-3) : Real(2.e-7);
    AMREX_ALWAYS_ASSERT(result.relative_l1_error < error_limit);
    AMREX_ALWAYS_ASSERT(result.relative_energy_drift < conservation_limit);
    return result;
}

Real
semicircle_area_primitive (Real x, Real center, Real radius)
{
    Real const offset = std::clamp(x - center, -radius, radius);
    Real const height =
        std::sqrt(amrex::max(Real(0), radius * radius - offset * offset));
    return Real(0.5) *
           (offset * height +
            radius * radius * std::asin(offset / radius));
}

Real
circle_rectangle_intersection_area (Real center_x, Real center_y, Real radius,
                                    Real xlo, Real xhi, Real ylo, Real yhi)
{
    AMREX_ALWAYS_ASSERT(radius > Real(0));
    AMREX_ALWAYS_ASSERT(xlo < xhi && ylo < yhi);

    Real const nearest_x = std::clamp(center_x, xlo, xhi);
    Real const nearest_y = std::clamp(center_y, ylo, yhi);
    Real const nearest_dx = nearest_x - center_x;
    Real const nearest_dy = nearest_y - center_y;
    if (nearest_dx * nearest_dx + nearest_dy * nearest_dy >=
        radius * radius) {
        return Real(0);
    }

    bool rectangle_inside_circle = true;
    for (Real const x : {xlo, xhi}) {
        for (Real const y : {ylo, yhi}) {
            Real const dx = x - center_x;
            Real const dy = y - center_y;
            rectangle_inside_circle =
                rectangle_inside_circle &&
                dx * dx + dy * dy <= radius * radius;
        }
    }
    Real const rectangle_area = (xhi - xlo) * (yhi - ylo);
    if (rectangle_inside_circle) {
        return rectangle_area;
    }

    Real const integration_lo = amrex::max(xlo, center_x - radius);
    Real const integration_hi = amrex::min(xhi, center_x + radius);
    if (integration_lo >= integration_hi) {
        return Real(0);
    }

    Vector<Real> breakpoints{integration_lo, integration_hi};
    for (Real const y : {ylo, yhi}) {
        Real const offset = std::abs(y - center_y);
        if (offset < radius) {
            Real const dx =
                std::sqrt(radius * radius - offset * offset);
            Real const left = center_x - dx;
            Real const right = center_x + dx;
            if (left > integration_lo && left < integration_hi) {
                breakpoints.push_back(left);
            }
            if (right > integration_lo && right < integration_hi) {
                breakpoints.push_back(right);
            }
        }
    }
    std::sort(breakpoints.begin(), breakpoints.end());

    Real area = Real(0);
    for (std::size_t segment = 1; segment < breakpoints.size(); ++segment) {
        Real const a = breakpoints[segment - 1];
        Real const b = breakpoints[segment];
        if (a >= b) {
            continue;
        }

        Real const midpoint = Real(0.5) * (a + b);
        Real const midpoint_dx = midpoint - center_x;
        Real const half_height = std::sqrt(amrex::max(
            Real(0), radius * radius - midpoint_dx * midpoint_dx));
        Real const circle_lo = center_y - half_height;
        Real const circle_hi = center_y + half_height;
        if (circle_hi <= ylo || circle_lo >= yhi) {
            continue;
        }

        bool const upper_is_circle = circle_hi < yhi;
        bool const lower_is_circle = circle_lo > ylo;
        Real const constant =
            (upper_is_circle ? center_y : yhi) -
            (lower_is_circle ? center_y : ylo);
        int const semicircle_coefficient =
            int(upper_is_circle) + int(lower_is_circle);
        area += constant * (b - a) +
                Real(semicircle_coefficient) *
                    (semicircle_area_primitive(b, center_x, radius) -
                     semicircle_area_primitive(a, center_x, radius));
    }
    return std::clamp(area, Real(0), rectangle_area);
}

Real
cloud_radius ()
{
    Real constexpr spacing = Real(1) / Real(8.5);
    Real constexpr diameter = spacing / Real(1.1);
    return Real(0.5) * diameter;
}

Real
cloud_volume_fraction (Cell const& cell)
{
    Real constexpr spacing = Real(1) / Real(8.5);
    Real const radius = cloud_radius();
    Real const xlo = cell.x - Real(0.5) * cell.hx;
    Real const xhi = cell.x + Real(0.5) * cell.hx;
    Real const ylo = cell.y - Real(0.5) * cell.hy;
    Real const yhi = cell.y + Real(0.5) * cell.hy;
    Real cloudy_area = Real(0);
    for (int cloud = 0; cloud <= 8; ++cloud) {
        Real const center_x = Real(cloud) * spacing;
        cloudy_area += circle_rectangle_intersection_area(
            center_x, Real(0.5), radius, xlo, xhi, ylo, yhi);
    }
    Real const fraction = cloudy_area / cell.volume;
    AMREX_ALWAYS_ASSERT(fraction >= Real(-1.e-12));
    AMREX_ALWAYS_ASSERT(fraction <= Real(1) + Real(1.e-12));
    return std::clamp(fraction, Real(0), Real(1));
}

CloudMesh
make_cloud_mesh (bool use_amr, int fine_n)
{
    AMREX_ALWAYS_ASSERT(fine_n > 0);
    AMREX_ALWAYS_ASSERT(fine_n % 4 == 0);
    Real constexpr beta = Real(0.5);
    std::array<BoundaryCondition, 4> boundary{
        reflecting_boundary(), reflecting_boundary(),
        marshak_boundary(Real(0), beta), marshak_boundary(Real(4), beta)};

    CloudMesh cloud_mesh;
    if (!use_amr) {
        cloud_mesh.mesh = make_mesh(
            fine_n, 1, [] (int, int, int) noexcept { return false; },
            [] (Real, Real) noexcept { return true; }, boundary);
        cloud_mesh.level_n = {fine_n};
        cloud_mesh.level_y_bounds = {{0, fine_n}};
        return cloud_mesh;
    }

    if (fine_n == 512) {
        // Reproduce the three-level hierarchy used for Fig. 6: a 32^2 base
        // grid, a factor-four level over the middle half, and a second
        // factor-four level over the cloudy middle quarter.
        cloud_mesh.mesh = make_nested_mesh(
            32, Vector<int>{4, 4},
            [] (int level, int, int j, int n) noexcept
            {
                if (level == 0) {
                    return j >= n / 4 && j < 3 * n / 4;
                }
                return level == 1 &&
                       j >= 3 * n / 8 && j < 5 * n / 8;
            },
            [] (Real, Real) noexcept { return true; }, boundary);
        cloud_mesh.level_n = {32, 128, 512};
        cloud_mesh.level_y_bounds = {
            {0, cloud_mesh.level_n[0]},
            {cloud_mesh.level_n[1] / 4,
             3 * cloud_mesh.level_n[1] / 4},
            {3 * cloud_mesh.level_n[2] / 8,
             5 * cloud_mesh.level_n[2] / 8}};
        return cloud_mesh;
    }

    int const nbase = fine_n / 4;
    AMREX_ALWAYS_ASSERT(nbase % 8 == 0);
    cloud_mesh.mesh = make_mesh(
        nbase, 4, [] (int, int j, int n) noexcept
        { return j >= 3 * n / 8 && j < 5 * n / 8; },
        [] (Real, Real) noexcept { return true; }, boundary);
    cloud_mesh.level_n = {nbase, fine_n};
    cloud_mesh.level_y_bounds = {
        {0, nbase}, {3 * fine_n / 8, 5 * fine_n / 8}};
    return cloud_mesh;
}

std::pair<Real, Real>
cloud_boundary_fluxes (Mesh const& mesh, Vector<Real> const& energy,
                       Vector<Real> const& diffusion)
{
    Real bottom_flux = Real(0);
    Real top_flux = Real(0);
    for (Long row = 0; row < static_cast<Long>(mesh.cells.size()); ++row) {
        for (auto const& face : mesh.cells[row].faces) {
            if (face.neighbor >= 0 ||
                face.boundary.kind != BoundaryKind::marshak) {
                continue;
            }
            Real const flux = face_transmissibility(face, row, diffusion) *
                              (energy[row] - face.boundary.value);
            if (face.side == ylo) {
                bottom_flux += flux;
            } else if (face.side == yhi) {
                top_flux += flux;
            }
        }
    }
    return {bottom_flux, top_flux};
}

void
write_cloud_plotfile (std::string const& plotfile_name,
                      CloudMesh const& cloud_mesh,
                      Vector<Real> const& energy,
                      Vector<Real> const& extinction,
                      Vector<Real> const& diffusion)
{
    Mesh const& mesh = cloud_mesh.mesh;
    AMREX_ALWAYS_ASSERT(!plotfile_name.empty());
    AMREX_ALWAYS_ASSERT(energy.size() == mesh.cells.size());
    AMREX_ALWAYS_ASSERT(extinction.size() == energy.size());
    AMREX_ALWAYS_ASSERT(diffusion.size() == energy.size());
    AMREX_ALWAYS_ASSERT(!cloud_mesh.level_n.empty());
    AMREX_ALWAYS_ASSERT(cloud_mesh.level_n.size() ==
                        cloud_mesh.level_y_bounds.size());
    AMREX_ALWAYS_ASSERT(cloud_mesh.level_n.back() == mesh.fine_n);

    int constexpr component_count = 5;
    Vector<std::string> const variable_names{
        "radiation_energy", "extinction", "diffusion_coefficient",
        "flux_limiter", "cloud_volume_fraction"};
    int const level_count = static_cast<int>(cloud_mesh.level_n.size());
    RealBox const physical_domain(
        {AMREX_D_DECL(Real(0), Real(0), Real(0))},
        {AMREX_D_DECL(Real(1), Real(1), Real(1))});
    Array<int, AMREX_SPACEDIM> const is_periodic{
        AMREX_D_DECL(0, 0, 0)};

    Vector<BoxArray> grids(level_count);
    Vector<Geometry> geometries(level_count);
    for (int level = 0; level < level_count; ++level) {
        int const level_n = cloud_mesh.level_n[level];
        auto const [ylo, yhi] = cloud_mesh.level_y_bounds[level];
        AMREX_ALWAYS_ASSERT(level_n > 0);
        AMREX_ALWAYS_ASSERT(ylo >= 0 && ylo < yhi && yhi <= level_n);
        if (level > 0) {
            AMREX_ALWAYS_ASSERT(
                level_n % cloud_mesh.level_n[level - 1] == 0);
        }
        IntVect const domain_lo(AMREX_D_DECL(0, 0, 0));
        IntVect const domain_hi(
            AMREX_D_DECL(level_n - 1, level_n - 1, 0));
        Box const domain(domain_lo, domain_hi);
        IntVect const grid_lo(AMREX_D_DECL(0, ylo, 0));
        IntVect const grid_hi(
            AMREX_D_DECL(level_n - 1, yhi - 1, 0));
        grids[level] = BoxArray(Box(grid_lo, grid_hi));
        grids[level].maxSize(64);
        geometries[level].define(domain, physical_domain,
                                 CoordSys::cartesian, is_periodic);
    }

    Vector<MultiFab> plot_data(level_count);
    for (int level = 0; level < level_count; ++level) {
        DistributionMapping const distribution(grids[level]);
        plot_data[level].define(grids[level], distribution, component_count,
                                0);
        int const level_n = cloud_mesh.level_n[level];
        int const coarsening = mesh.fine_n / level_n;
        Real const inverse_sample_count =
            Real(1) / Real(coarsening * coarsening);
        Vector<Real> host_values(
            static_cast<std::size_t>(level_n) * level_n * component_count,
            Real(0));
        // Covered coarse cells are volume averages of the converged fine
        // solution; unrefined cells reduce to repeated samples of one owner.
        for (int j = 0; j < level_n; ++j) {
            for (int i = 0; i < level_n; ++i) {
                auto const output_index =
                    (static_cast<std::size_t>(j) * level_n + i) *
                    component_count;
                for (int jj = 0; jj < coarsening; ++jj) {
                    for (int ii = 0; ii < coarsening; ++ii) {
                        int const fine_i = i * coarsening + ii;
                        int const fine_j = j * coarsening + jj;
                        Long const row =
                            mesh.owner[static_cast<std::size_t>(fine_j) *
                                           mesh.fine_n +
                                       fine_i];
                        AMREX_ALWAYS_ASSERT(row >= 0);
                        Real const sample_scale = inverse_sample_count;
                        host_values[output_index] +=
                            sample_scale * energy[row];
                        host_values[output_index + 1] +=
                            sample_scale * extinction[row];
                        host_values[output_index + 2] +=
                            sample_scale * diffusion[row];
                        host_values[output_index + 3] +=
                            sample_scale * diffusion[row] * extinction[row];
                        host_values[output_index + 4] +=
                            sample_scale *
                            cloud_volume_fraction(mesh.cells[row]);
                    }
                }
            }
        }

        Gpu::DeviceVector<Real> device_values(host_values.size());
        Gpu::copy(Gpu::hostToDevice, host_values.begin(), host_values.end(),
                  device_values.begin());
        Real const* values = device_values.data();
        for (MFIter mfi(plot_data[level]); mfi.isValid(); ++mfi) {
            Box const& box = mfi.validbox();
            auto const array = plot_data[level].array(mfi);
            ParallelFor(
                box, component_count,
                [=] AMREX_GPU_DEVICE(int i, int j, int k, int component)
                {
                    auto const index =
                        (static_cast<std::size_t>(j) * level_n + i) *
                            component_count +
                        component;
                    array(i, j, k, component) = values[index];
                });
        }
        Gpu::streamSynchronize();
    }

    Vector<IntVect> refinement_ratios;
    refinement_ratios.reserve(level_count - 1);
    for (int level = 0; level + 1 < level_count; ++level) {
        int const refinement_ratio =
            cloud_mesh.level_n[level + 1] / cloud_mesh.level_n[level];
        AMREX_ALWAYS_ASSERT(refinement_ratio > 1);
        refinement_ratios.emplace_back(
            AMREX_D_DECL(refinement_ratio, refinement_ratio,
                         refinement_ratio));
    }
    WriteMultiLevelPlotfile(
        plotfile_name, level_count, GetVecOfConstPtrs(plot_data),
        variable_names, geometries, Real(0), Vector<int>(level_count, 0),
        refinement_ratios);
    amrex::Print() << "Wrote FLD cloud plotfile " << plotfile_name
                   << std::endl;
}

CloudResult
run_cloud (bool use_amr, int fine_n, int anderson_depth, Real anderson_beta,
           bool limited, bool iteration_output,
           std::string const& plotfile_name)
{
    CloudMesh cloud_mesh = make_cloud_mesh(use_amr, fine_n);
    Mesh const& mesh = cloud_mesh.mesh;
    Vector<Real> extinction(mesh.cells.size());
    Vector<Real> state(mesh.cells.size());
    Vector<Real> volume_weights(mesh.cells.size());
    Real cloudy_area = Real(0);
    Long mixed_cells = 0;
    Real constexpr clear_extinction = Real(0.1);
    Real constexpr cloudy_extinction = Real(1000);
    Real constexpr clear_density = Real(1);
    Real constexpr cloudy_density = Real(1);
    for (Long row = 0; row < static_cast<Long>(mesh.cells.size()); ++row) {
        auto const& cell = mesh.cells[row];
        Real const cloudy_fraction = cloud_volume_fraction(cell);
        // Apply the mass-weighted arithmetic Rosseland mean in Fig. 3.  The
        // pure-radiation cloud test specifies no material density contrast,
        // so both densities are normalized to one.
        Real const clear_fraction = Real(1) - cloudy_fraction;
        extinction[row] =
            (cloudy_fraction * cloudy_density * cloudy_extinction +
             clear_fraction * clear_density * clear_extinction) /
            (cloudy_fraction * cloudy_density +
             clear_fraction * clear_density);
        state[row] = Real(0.25) + Real(3.5) * cell.y;
        volume_weights[row] = cell.volume;
        cloudy_area += cloudy_fraction * cell.volume;
        if (cloudy_fraction > Real(0) && cloudy_fraction < Real(1)) {
            ++mixed_cells;
        }
    }

    CloudResult result;
    result.cells = static_cast<Long>(mesh.cells.size());
    result.mixed_cells = mixed_cells;
    Real constexpr pi = Real(3.1415926535897932384626433832795);
    Real const expected_cloudy_area =
        Real(8.5) * pi * cloud_radius() * cloud_radius();
    result.cloudy_area_relative_error =
        std::abs(cloudy_area - expected_cloudy_area) / expected_cloudy_area;
    Real const cloudy_area_tolerance =
        (sizeof(Real) == sizeof(float)) ? Real(2.e-4) : Real(2.e-12);
    AMREX_ALWAYS_ASSERT(result.cloudy_area_relative_error <
                        cloudy_area_tolerance);
    AMREX_ALWAYS_ASSERT(result.mixed_cells > 0);
    Real const nonlinear_tolerance =
        (sizeof(Real) == sizeof(float)) ? Real(2.e-4) : Real(2.e-6);
    int constexpr maximum_nonlinear_iterations = 125;
    AMG<Real>::Options cloud_amg_options;
    cloud_amg_options.priority_seed = 1;
    auto const& incident_boundary = mesh.outer_boundary[yhi];
    AMREX_ALWAYS_ASSERT(incident_boundary.kind == BoundaryKind::marshak);
    AndersonMixer mixer(anderson_depth, anderson_beta,
                         std::move(volume_weights),
                         incident_boundary.value);
    Real const incident_marshak_flux =
        Real(0.5) * incident_boundary.beta * incident_boundary.value;
    AMREX_ALWAYS_ASSERT(incident_marshak_flux > Real(0));

    for (int iteration = 0; iteration < maximum_nonlinear_iterations;
         ++iteration) {
        auto diffusion = compute_diffusion(mesh, state, extinction, limited);
        auto system = assemble_system(mesh, diffusion, {}, Real(0), false,
                                      &extinction);
        AMGGMRESSolver solver(system.matrix, cloud_amg_options);
        record_setup(result.solver, solver);
        auto solution = solver.solve(system.rhs);
        record_solve(result.solver, solution);

        result.final_nonlinear_change =
            maximum_relative_change(solution.values, state);
        ++result.nonlinear_iterations;
        auto const [iteration_bottom_flux, iteration_top_flux] =
            cloud_boundary_fluxes(mesh, solution.values, diffusion);
        amrex::ignore_unused(iteration_top_flux);
        if (iteration_output) {
            amrex::Print()
                << "FLD cloud " << (use_amr ? "AMR" : "uniform")
                << " nonlinear iteration=" << result.nonlinear_iterations
                << ", method="
                << (!limited ? "linear"
                             : (anderson_depth > 0 ? "Anderson" : "Picard"))
                << ", change=" << result.final_nonlinear_change
                << ", transmission="
                << iteration_bottom_flux / incident_marshak_flux
                << ", GMRES iterations=" << solution.iterations
                << ", true relative residual="
                << solution.relative_residual << std::endl;
        }
        if (result.final_nonlinear_change <= nonlinear_tolerance) {
            state = std::move(solution.values);
            break;
        }
        state = mixer.update(state, solution.values);
    }
    result.anderson_steps = mixer.anderson_steps();
    result.anderson_restarts = mixer.restarts();

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        result.final_nonlinear_change <= nonlinear_tolerance,
        "The cloud-layer FLD nonlinear iteration did not converge");

    auto diffusion = compute_diffusion(mesh, state, extinction, limited);
    auto const [bottom_flux, top_flux] =
        cloud_boundary_fluxes(mesh, state, diffusion);
    // beta * (E - E_eq) = c E / 2 - 2 F_inc, so
    // F_inc = beta * E_eq / 2 when beta = c / 2.
    result.transmission = bottom_flux / incident_marshak_flux;
    result.balance_error = std::abs(bottom_flux + top_flux) /
                           amrex::max(std::abs(bottom_flux), Real(1.e-30));
    auto const [minimum, maximum] = minimum_maximum(state);
    result.minimum_energy = minimum;
    result.maximum_energy = maximum;
    if (!plotfile_name.empty()) {
        write_cloud_plotfile(plotfile_name, cloud_mesh, state, extinction,
                             diffusion);
    }

    Real const balance_limit =
        (sizeof(Real) == sizeof(float)) ? Real(2.e-3) : Real(2.e-6);
    AMREX_ALWAYS_ASSERT(result.balance_error < balance_limit);
    AMREX_ALWAYS_ASSERT(result.minimum_energy >= Real(-1.e-8));
    AMREX_ALWAYS_ASSERT(result.maximum_energy <= Real(4.01));
    AMREX_ALWAYS_ASSERT(result.transmission > Real(0));
    AMREX_ALWAYS_ASSERT(result.transmission < Real(1));
    return result;
}

Mesh
make_front_mesh ()
{
    std::array<BoundaryCondition, 4> boundary{
        reflecting_boundary(), reflecting_boundary(), reflecting_boundary(),
        reflecting_boundary()};
    Real constexpr inner_radius = Real(0.1);
    return make_mesh(
        64, 1, [] (int, int, int) noexcept { return false; },
        [] (Real x, Real y) noexcept
        {
            Real const dx = x - Real(0.5);
            Real const dy = y - Real(0.5);
            return dx * dx + dy * dy > inner_radius * inner_radius;
        },
        boundary, true, dirichlet_boundary(Real(1)));
}

Real
far_excess (Mesh const& mesh, Vector<Real> const& state, Real radius,
            Real ambient_energy)
{
    Real result = Real(0);
    for (Long row = 0; row < static_cast<Long>(mesh.cells.size()); ++row) {
        Real const dx = mesh.cells[row].x - Real(0.5);
        Real const dy = mesh.cells[row].y - Real(0.5);
        if (std::hypot(dx, dy) > radius) {
            result = amrex::max(result, state[row] - ambient_energy);
        }
    }
    return result;
}

Real
front_radius (Mesh const& mesh, Vector<Real> const& state)
{
    Real result = Real(0.1);
    Real constexpr threshold = Real(0.01);
    for (Long row = 0; row < static_cast<Long>(mesh.cells.size()); ++row) {
        if (state[row] > threshold) {
            Real const dx = mesh.cells[row].x - Real(0.5);
            Real const dy = mesh.cells[row].y - Real(0.5);
            result = amrex::max(result, std::hypot(dx, dy));
        }
    }
    return result;
}

FrontResult
run_limited_front ()
{
    Mesh mesh = make_front_mesh();
    Vector<Real> extinction(mesh.cells.size(), Real(0.01));
    Real constexpr ambient_energy = Real(1.e-4);
    Vector<Real> state(mesh.cells.size(), ambient_energy);
    Real constexpr final_time = Real(0.15);
    int constexpr steps = 48;
    Real constexpr dt = final_time / Real(steps);
    Real constexpr relaxation = Real(0.7);
    Real const picard_tolerance =
        (sizeof(Real) == sizeof(float)) ? Real(2.e-4) : Real(2.e-5);
    int constexpr maximum_picard_iterations = 75;

    FrontResult result;
    result.cells = static_cast<Long>(mesh.cells.size());
    for (int step = 0; step < steps; ++step) {
        Vector<Real> const old_state = state;
        bool converged = false;
        int step_iterations = 0;
        for (int iteration = 0; iteration < maximum_picard_iterations;
             ++iteration) {
            auto diffusion = compute_diffusion(mesh, state, extinction, true);
            auto system = assemble_system(mesh, diffusion, old_state, dt, true);
            AMGGMRESSolver solver(system.matrix);
            record_setup(result.solver, solver);
            auto solution = solver.solve(system.rhs);
            record_solve(result.solver, solution);

            result.final_picard_change =
                maximum_relative_change(solution.values, state);
            ++step_iterations;
            ++result.total_picard_iterations;
            if (result.final_picard_change <= picard_tolerance) {
                state = std::move(solution.values);
                converged = true;
                break;
            }
            for (std::size_t row = 0; row < state.size(); ++row) {
                state[row] = relaxation * solution.values[row] +
                             (Real(1) - relaxation) * state[row];
            }
        }
        result.maximum_picard_iterations =
            amrex::max(result.maximum_picard_iterations, step_iterations);
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            converged,
            "The limited-front FLD Picard iteration did not converge");
    }

    compute_diffusion(mesh, state, extinction, true,
                      &result.maximum_flux_fraction);
    result.causal_radius = Real(0.1) + final_time;
    result.front_radius = front_radius(mesh, state);
    result.far_excess =
        far_excess(mesh, state, result.causal_radius + Real(2) * mesh.fine_h,
                   ambient_energy);
    auto const [minimum, maximum] = minimum_maximum(state);
    result.minimum_energy = minimum;
    result.maximum_energy = maximum;

    Vector<Real> unlimited_state(mesh.cells.size(), ambient_energy);
    auto unlimited_diffusion =
        compute_diffusion(mesh, unlimited_state, extinction, false);
    auto unlimited_system =
        assemble_system(mesh, unlimited_diffusion, unlimited_state, dt, true);
    AMGGMRESSolver unlimited_solver(unlimited_system.matrix);
    Vector<Real> unlimited_boundary_rhs(mesh.cells.size());
    for (std::size_t row = 0; row < unlimited_state.size(); ++row) {
        unlimited_boundary_rhs[row] =
            unlimited_system.rhs[row] - unlimited_state[row];
    }
    for (int step = 0; step < steps; ++step) {
        Vector<Real> rhs = unlimited_state;
        for (std::size_t row = 0; row < rhs.size(); ++row) {
            rhs[row] += unlimited_boundary_rhs[row];
        }
        auto solution = unlimited_solver.solve(rhs);
        unlimited_state = std::move(solution.values);
    }
    result.unlimited_far_excess = far_excess(
        mesh, unlimited_state, result.causal_radius + Real(2) * mesh.fine_h,
        ambient_energy);

    Real const causality_tolerance =
        (sizeof(Real) == sizeof(float)) ? Real(2.e-4) : Real(2.e-8);
    AMREX_ALWAYS_ASSERT(result.maximum_flux_fraction <=
                        Real(1) + causality_tolerance);
    AMREX_ALWAYS_ASSERT(result.maximum_flux_fraction > Real(0.95));
    AMREX_ALWAYS_ASSERT(result.front_radius <=
                        result.causal_radius + Real(2) * mesh.fine_h);
    AMREX_ALWAYS_ASSERT(result.front_radius >=
                        result.causal_radius - Real(4) * mesh.fine_h);
    AMREX_ALWAYS_ASSERT(result.minimum_energy >= ambient_energy - Real(2.e-5));
    AMREX_ALWAYS_ASSERT(result.maximum_energy <= Real(1) + Real(2.e-5));
    AMREX_ALWAYS_ASSERT(result.far_excess < Real(0.015));
    AMREX_ALWAYS_ASSERT(result.unlimited_far_excess > Real(0.04));
    return result;
}

void
print_solver_summary (SolverSummary const& solver)
{
    amrex::Print() << "solves=" << solver.solves
                   << ", GMRES avg/max=" << solver.average_iterations() << "/"
                   << solver.maximum_iterations
                   << ", max true relative residual="
                   << solver.maximum_relative_residual
                   << ", max AMG levels=" << solver.maximum_levels
                   << ", max operator complexity="
                   << solver.maximum_operator_complexity
                   << ", aggregate setup=" << solver.setup_seconds << " s";
}

} // namespace

int
main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    {
        static_assert(AMREX_SPACEDIM == 2);

        int cloud_anderson_depth = 7;
        int cloud_fine_n = 128;
        Real cloud_anderson_beta = Real(1);
        int cloud_iteration_output = 1;
        int cloud_flux_limiter = 1;
        int cloud_only = 0;
        std::string cloud_case = "both";
        std::string cloud_plotfile_prefix;
        {
            ParmParse pp;
            pp.query("cloud_anderson_depth", cloud_anderson_depth);
            pp.query("cloud_fine_n", cloud_fine_n);
            pp.query("cloud_anderson_beta", cloud_anderson_beta);
            pp.query("cloud_iteration_output", cloud_iteration_output);
            pp.query("cloud_flux_limiter", cloud_flux_limiter);
            pp.query("cloud_only", cloud_only);
            pp.query("cloud_case", cloud_case);
            pp.query("cloud_plotfile_prefix", cloud_plotfile_prefix);
        }

        if (cloud_only != 0) {
            if (cloud_case != "both" && cloud_case != "uniform" &&
                cloud_case != "amr") {
                amrex::Abort(
                    "cloud_case must be one of: both, uniform, or amr");
            }

            auto const run_selected_cloud = [&] (bool use_amr) {
                return run_cloud(
                    use_amr, cloud_fine_n, cloud_anderson_depth,
                    cloud_anderson_beta, cloud_flux_limiter != 0,
                    cloud_iteration_output != 0,
                    cloud_plotfile_prefix.empty()
                        ? std::string()
                        : cloud_plotfile_prefix +
                              (use_amr ? "_amr" : "_uniform"));
            };

            if (cloud_case == "uniform") {
                auto const cloud_uniform = run_selected_cloud(false);
                amrex::Print()
                    << "FLD cloud Anderson benchmark: depth="
                    << cloud_anderson_depth
                    << ", fine_n=" << cloud_fine_n
                    << ", beta=" << cloud_anderson_beta
                    << ", limiter="
                    << (cloud_flux_limiter != 0 ? "on" : "off")
                    << ", uniform transmission/iterations/Anderson/restarts="
                    << cloud_uniform.transmission << "/"
                    << cloud_uniform.nonlinear_iterations << "/"
                    << cloud_uniform.anderson_steps << "/"
                    << cloud_uniform.anderson_restarts << std::endl;
                amrex::Finalize();
                return 0;
            }

            if (cloud_case == "amr") {
                auto const cloud_amr = run_selected_cloud(true);
                amrex::Print()
                    << "FLD cloud Anderson benchmark: depth="
                    << cloud_anderson_depth
                    << ", fine_n=" << cloud_fine_n
                    << ", beta=" << cloud_anderson_beta
                    << ", limiter="
                    << (cloud_flux_limiter != 0 ? "on" : "off")
                    << ", AMR transmission/iterations/Anderson/restarts="
                    << cloud_amr.transmission << "/"
                    << cloud_amr.nonlinear_iterations << "/"
                    << cloud_amr.anderson_steps << "/"
                    << cloud_amr.anderson_restarts << std::endl;
                amrex::Finalize();
                return 0;
            }

            auto const cloud_uniform =
                run_selected_cloud(false);
            auto const cloud_amr = run_selected_cloud(true);
            amrex::Print()
                << "FLD cloud Anderson benchmark: depth="
                << cloud_anderson_depth
                << ", fine_n=" << cloud_fine_n
                << ", beta=" << cloud_anderson_beta
                << ", limiter="
                << (cloud_flux_limiter != 0 ? "on" : "off")
                << ", uniform transmission/iterations/Anderson/restarts="
                << cloud_uniform.transmission << "/"
                << cloud_uniform.nonlinear_iterations << "/"
                << cloud_uniform.anderson_steps << "/"
                << cloud_uniform.anderson_restarts
                << ", AMR transmission/iterations/Anderson/restarts="
                << cloud_amr.transmission << "/"
                << cloud_amr.nonlinear_iterations << "/"
                << cloud_amr.anderson_steps << "/"
                << cloud_amr.anderson_restarts << std::endl;
            amrex::Finalize();
            return 0;
        }

        auto const gaussian_uniform = run_gaussian(false);
        amrex::Print() << "FLD Gaussian uniform: cells="
                       << gaussian_uniform.cells << ", relative L1 error="
                       << gaussian_uniform.relative_l1_error
                       << ", relative energy drift="
                       << gaussian_uniform.relative_energy_drift << ", ";
        print_solver_summary(gaussian_uniform.solver);
        amrex::Print() << '\n';

        auto const gaussian_amr = run_gaussian(true);
        amrex::Print() << "FLD Gaussian AMR: cells=" << gaussian_amr.cells
                       << ", relative L1 error="
                       << gaussian_amr.relative_l1_error
                       << ", relative energy drift="
                       << gaussian_amr.relative_energy_drift << ", ";
        print_solver_summary(gaussian_amr.solver);
        amrex::Print() << '\n';
        AMREX_ALWAYS_ASSERT(gaussian_amr.relative_l1_error <=
                            Real(2) * gaussian_uniform.relative_l1_error);

        auto const cloud_uniform =
            run_cloud(false, cloud_fine_n, cloud_anderson_depth,
                      cloud_anderson_beta, true,
                      cloud_iteration_output != 0, std::string());
        amrex::Print() << "FLD cloud uniform: cells=" << cloud_uniform.cells
                       << ", transmission=" << cloud_uniform.transmission
                       << ", balance error=" << cloud_uniform.balance_error
                       << ", mixed cells/cloud area error="
                       << cloud_uniform.mixed_cells << "/"
                       << cloud_uniform.cloudy_area_relative_error
                       << ", E range=[" << cloud_uniform.minimum_energy << ","
                       << cloud_uniform.maximum_energy << "]"
                       << ", nonlinear iterations/change="
                       << cloud_uniform.nonlinear_iterations << "/"
                       << cloud_uniform.final_nonlinear_change
                       << ", Anderson steps/restarts="
                       << cloud_uniform.anderson_steps << "/"
                       << cloud_uniform.anderson_restarts << ", ";
        print_solver_summary(cloud_uniform.solver);
        amrex::Print() << '\n';

        auto const cloud_amr =
            run_cloud(true, cloud_fine_n, cloud_anderson_depth,
                      cloud_anderson_beta, true,
                      cloud_iteration_output != 0, std::string());
        amrex::Print() << "FLD cloud AMR: cells=" << cloud_amr.cells
                       << ", transmission=" << cloud_amr.transmission
                       << ", balance error=" << cloud_amr.balance_error
                       << ", mixed cells/cloud area error="
                       << cloud_amr.mixed_cells << "/"
                       << cloud_amr.cloudy_area_relative_error
                       << ", E range=[" << cloud_amr.minimum_energy << ","
                       << cloud_amr.maximum_energy << "]"
                       << ", nonlinear iterations/change="
                       << cloud_amr.nonlinear_iterations << "/"
                       << cloud_amr.final_nonlinear_change
                       << ", Anderson steps/restarts="
                       << cloud_amr.anderson_steps << "/"
                       << cloud_amr.anderson_restarts << ", ";
        print_solver_summary(cloud_amr.solver);
        amrex::Print() << '\n';

        Real const transmission_difference =
            std::abs(cloud_amr.transmission - cloud_uniform.transmission) /
            cloud_uniform.transmission;
        amrex::Print() << "FLD cloud AMR/fine transmission difference="
                       << transmission_difference << '\n';
        AMREX_ALWAYS_ASSERT(transmission_difference < Real(0.12));

        auto const front = run_limited_front();
        amrex::Print() << "FLD limited front: cells=" << front.cells
                       << ", front/causal radius=" << front.front_radius << "/"
                       << front.causal_radius
                       << ", far excess=" << front.far_excess
                       << ", unlimited far excess="
                       << front.unlimited_far_excess
                       << ", max |F|/(cE)=" << front.maximum_flux_fraction
                       << ", E range=[" << front.minimum_energy << ","
                       << front.maximum_energy << "]"
                       << ", Picard total/max/change="
                       << front.total_picard_iterations << "/"
                       << front.maximum_picard_iterations << "/"
                       << front.final_picard_change << ", ";
        print_solver_summary(front.solver);
        amrex::Print() << '\n';

        amrex::Print() << "2-D scattering-only FLD Gaussian, cloud-layer, "
                       << "and limited-front GMRES+AMG tests passed\n";
    }
    amrex::Finalize();
}
