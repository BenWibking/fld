#ifndef FLD_MESH_H_
#define FLD_MESH_H_

#include <AMReX_REAL.H>
#include <AMReX_Vector.H>

#include <array>
#include <functional>
#include <utility>

namespace fld_test
{

inline constexpr int xlo = 0;
inline constexpr int xhi = 1;
inline constexpr int ylo = 2;
inline constexpr int yhi = 3;

enum class BoundaryKind { periodic, reflecting, dirichlet, marshak };

struct BoundaryCondition
{
    BoundaryKind kind = BoundaryKind::reflecting;
    amrex::Real value = amrex::Real(0);
    amrex::Real beta = amrex::Real(0);
};

struct Face
{
    amrex::Long neighbor = -1;
    amrex::Real area = amrex::Real(0);
    amrex::Real self_distance = amrex::Real(0);
    amrex::Real neighbor_distance = amrex::Real(0);
    int normal_x = 0;
    int normal_y = 0;
    int side = -1;
    BoundaryCondition boundary;
};

struct Cell
{
    amrex::Long id = -1;
    int ilo = 0;
    int ihi = 0;
    int jlo = 0;
    int jhi = 0;
    amrex::Real x = amrex::Real(0);
    amrex::Real y = amrex::Real(0);
    amrex::Real hx = amrex::Real(0);
    amrex::Real hy = amrex::Real(0);
    amrex::Real volume = amrex::Real(0);
    amrex::Vector<Face> faces;
};

struct Mesh
{
    int fine_n = 0;
    amrex::Real fine_h = amrex::Real(0);
    amrex::Vector<Cell> cells;
    amrex::Vector<amrex::Long> owner;
    std::array<BoundaryCondition, 4> outer_boundary;
    BoundaryCondition hole_boundary;
    bool has_hole = false;
};

using RefinePredicate = std::function<bool(int, int, int, int)>;
using SimpleRefinePredicate = std::function<bool(int, int, int)>;
using ActivePredicate = std::function<bool(amrex::Real, amrex::Real)>;

[[nodiscard]] BoundaryCondition periodic_boundary ();
[[nodiscard]] BoundaryCondition reflecting_boundary ();
[[nodiscard]] BoundaryCondition dirichlet_boundary (amrex::Real value);
[[nodiscard]] BoundaryCondition marshak_boundary (
    amrex::Real equilibrium_energy, amrex::Real beta);

[[nodiscard]] Mesh make_nested_mesh (
    int nbase, amrex::Vector<int> const& refinement_ratios,
    RefinePredicate const& refine, ActivePredicate const& active,
    std::array<BoundaryCondition, 4> outer_boundary, bool has_hole = false,
    BoundaryCondition hole_boundary = {});

[[nodiscard]] Mesh make_mesh (
    int nbase, int refinement_ratio, SimpleRefinePredicate const& refine,
    ActivePredicate const& active,
    std::array<BoundaryCondition, 4> outer_boundary, bool has_hole = false,
    BoundaryCondition hole_boundary = {});

} // namespace fld_test

#endif
