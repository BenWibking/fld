#include "FLDLinearSystem.H"

#include <AMReX.H>
#include <AMReX_AlgPartition.H>
#include <AMReX_ParallelDescriptor.H>

#include <algorithm>
#include <array>
#include <cmath>
#include <utility>

namespace fld_test
{

using namespace amrex;

namespace
{

using Entry = std::pair<Long, Real>;
using Entries = Vector<Entry>;

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

} // namespace

Real
face_transmissibility (Face const& face, Long row,
                       Vector<Real> const& diffusion,
                       Vector<Real> const* rosseland_extinction)
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
            Real const arithmetic_mean = Real(0.5) * (chi_self + chi_other);
            Real const harmonic_mean =
                Real(2) * chi_self * chi_other / (chi_self + chi_other);
            Real const surface_scale = Real(4) / (Real(3) * center_distance);
            Real const face_extinction =
                amrex::min(arithmetic_mean,
                           amrex::max(harmonic_mean, surface_scale));
            Real const lambda_self = dself * chi_self;
            Real const lambda_other = dother * chi_other;
            // Use the less restrictive of the two cell-centered limiters at
            // the face with the Howell-Greenough surface opacity.
            Real const face_limiter = amrex::max(lambda_self, lambda_other);
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

LinearSystem
assemble_system (Mesh const& mesh, Vector<Real> const& diffusion,
                 Vector<Real> const& old_state, Real dt, bool transient,
                 Vector<Real> const* rosseland_extinction)
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

Vector<Real>
compute_diffusion (Mesh const& mesh, Vector<Real> const& energy,
                   Vector<Real> const& extinction, bool limited,
                   Real* maximum_flux_fraction)
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

} // namespace fld_test
