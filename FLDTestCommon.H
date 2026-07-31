#ifndef FLD_TEST_COMMON_H_
#define FLD_TEST_COMMON_H_

#include "FLDTest.H"

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


namespace fld_test
{
namespace
{

using namespace amrex;

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

} // namespace
} // namespace fld_test

#endif
