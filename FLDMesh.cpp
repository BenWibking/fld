#include "FLDMesh.H"

#include <AMReX.H>

#include <cstddef>

namespace fld_test
{

using namespace amrex;

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

Mesh
make_nested_mesh (int nbase, Vector<int> const& refinement_ratios,
                  RefinePredicate const& refine,
                  ActivePredicate const& active,
                  std::array<BoundaryCondition, 4> outer_boundary,
                  bool has_hole, BoundaryCondition hole_boundary)
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
            refinement_ratios[level] > 1 && refine(level, i, j, level_n);
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

Mesh
make_mesh (int nbase, int refinement_ratio,
           SimpleRefinePredicate const& refine,
           ActivePredicate const& active,
           std::array<BoundaryCondition, 4> outer_boundary, bool has_hole,
           BoundaryCondition hole_boundary)
{
    return make_nested_mesh(
        nbase, Vector<int>{refinement_ratio},
        [&] (int, int i, int j, int n) { return refine(i, j, n); }, active,
        outer_boundary, has_hole, hole_boundary);
}

} // namespace fld_test
