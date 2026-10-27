#include "FLDFieldOps.H"
#include "FLDTest.H"
#include "MLABecAMG.H"

#include <AMReX_GpuQualifiers.H>
#include <AMReX_MFParallelFor.H>
#include <AMReX_ParReduce.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>

#include <cmath>
#include <iomanip>
#include <string>

namespace fld_test
{

using namespace amrex;

namespace
{

// phi - div(b grad phi) = f on the unit cube with exact Dirichlet data.
// phi is asymmetric about the refined patch, so every coarse/fine face sees
// nonzero normal and transverse gradients.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE Real
mms_phi (Real x, Real y, Real z) noexcept
{
    return std::sin(Real(2) * x + Real(0.3)) *
           std::cos(Real(1.7) * y - Real(0.2)) * std::exp(Real(0.8) * z);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE Real
mms_b (Real x, Real y, Real z) noexcept
{
    return Real(1) + Real(0.4) * x * y + Real(0.3) * z * z;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE Real
mms_rhs (Real x, Real y, Real z) noexcept
{
    Real const sx = std::sin(Real(2) * x + Real(0.3));
    Real const cx = std::cos(Real(2) * x + Real(0.3));
    Real const sy = std::sin(Real(1.7) * y - Real(0.2));
    Real const cy = std::cos(Real(1.7) * y - Real(0.2));
    Real const ez = std::exp(Real(0.8) * z);
    Real const phi = sx * cy * ez;
    Real const phi_x = Real(2) * cx * cy * ez;
    Real const phi_y = Real(-1.7) * sx * sy * ez;
    Real const phi_z = Real(0.8) * phi;
    Real const laplacian = Real(-6.25) * phi;
    Real const div_flux = Real(0.4) * y * phi_x + Real(0.4) * x * phi_y +
                          Real(0.6) * z * phi_z +
                          mms_b(x, y, z) * laplacian;
    return phi - div_flux;
}

// Cell classes for the error report.  Physical-boundary cells are separated
// because both stencils share their first-order one-sided boundary flux.
enum Band : int { covered = -1, interior = 0, fine_interface = 1,
                  coarse_interface = 2, physical_boundary = 3, num_bands = 4 };

struct BandStats
{
    Real max_error[num_bands] = {};
    Real max_tau[num_bands] = {};
    Real l2_error = Real(0);
    Real max_error_all = Real(0);
};

BandStats
measure (DiffusionHierarchy const& hierarchy, LevelData const& error,
         LevelData const& tau)
{
    // The hierarchy has one fine patch, refine(Pc, 2) with Pc the middle
    // half of the coarse domain.
    int const n = hierarchy.geom[0].Domain().length(0);
    Box const coarse_patch(IntVect(n / 4), IntVect(3 * n / 4 - 1));
    Box const fine_patch = amrex::refine(coarse_patch, 2);
    Box const fine_inner = amrex::grow(fine_patch, -1);
    Box const coarse_ring = amrex::grow(coarse_patch, 1);
    Box const coarse_inner = amrex::grow(hierarchy.geom[0].Domain(), -1);

    BandStats stats;
    Real sum_sq = Real(0);
    for (int level = 0; level < 2; ++level) {
        Real const volume = AMREX_D_TERM(hierarchy.geom[level].CellSize(0),
                                         *hierarchy.geom[level].CellSize(1),
                                         *hierarchy.geom[level].CellSize(2));
        auto const& err = *error[level];
        auto const& res = *tau[level];
        auto const e = err.const_arrays();
        auto const t = res.const_arrays();
        auto band_of = [=] AMREX_GPU_DEVICE (IntVect const& iv) noexcept
        {
            if (level == 1) {
                return fine_inner.contains(iv) ? int(interior)
                                               : int(fine_interface);
            }
            if (coarse_patch.contains(iv)) { return int(covered); }
            if (coarse_ring.contains(iv)) { return int(coarse_interface); }
            if (!coarse_inner.contains(iv)) { return int(physical_boundary); }
            return int(interior);
        };
        for (int b = 0; b < num_bands; ++b) {
            auto const r = ParReduce(
                TypeList<ReduceOpMax, ReduceOpMax, ReduceOpSum>{},
                TypeList<Real, Real, Real>{}, err,
                [=] AMREX_GPU_DEVICE (int box, int i, int j, int k) noexcept
                    -> GpuTuple<Real, Real, Real>
                {
                    if (band_of(IntVect(AMREX_D_DECL(i, j, k))) != b) {
                        return {Real(0), Real(0), Real(0)};
                    }
                    Real const ae = std::abs(e[box](i, j, k));
                    return {ae, std::abs(t[box](i, j, k)), ae * ae * volume};
                });
            stats.max_error[b] = amrex::max(stats.max_error[b],
                                            amrex::get<0>(r));
            stats.max_tau[b] = amrex::max(stats.max_tau[b], amrex::get<1>(r));
            sum_sq += amrex::get<2>(r);
        }
    }
    ParallelDescriptor::ReduceRealMax(stats.max_error, num_bands);
    ParallelDescriptor::ReduceRealMax(stats.max_tau, num_bands);
    ParallelDescriptor::ReduceRealSum(sum_sq);
    stats.l2_error = std::sqrt(sum_sq);
    for (Real value : stats.max_error) {
        stats.max_error_all = amrex::max(stats.max_error_all, value);
    }
    return stats;
}

BandStats
run_mms_case (int n_cell)
{
    int const max_grid_size = amrex::min(16, n_cell / 2);
    auto const hierarchy = make_centered_patch_hierarchy(
        n_cell, 2, max_grid_size, {AMREX_D_DECL(0, 0, 0)});
    int const nlevels = static_cast<int>(hierarchy.geom.size());

    auto acoef = make_cell_data(hierarchy, 1, 0);
    auto rhs = make_cell_data(hierarchy, 1, 0);
    auto exact = make_cell_data(hierarchy, 1, 0);
    auto level_bc = make_cell_data(hierarchy, 1, 1);
    auto solution = make_cell_data(hierarchy, 1, 1);
    auto tau = make_cell_data(hierarchy, 1, 0);
    auto bcoef = make_face_data(hierarchy);
    set_level_data(acoef, Real(1));
    set_level_data(solution, Real(0));

    for (int level = 0; level < nlevels; ++level) {
        auto const dx = hierarchy.geom[level].CellSizeArray();
        auto const domain = hierarchy.geom[level].Domain();
        auto const f = rhs[level]->arrays();
        auto const u = exact[level]->arrays();
        ParallelFor(*rhs[level],
        [=] AMREX_GPU_DEVICE (int box, int i, int j, int k) noexcept
        {
            Real const x = (Real(i) + Real(0.5)) * dx[0];
            Real const y = (Real(j) + Real(0.5)) * dx[1];
            Real const z = (Real(k) + Real(0.5)) * dx[2];
            f[box](i, j, k) = mms_rhs(x, y, z);
            u[box](i, j, k) = mms_phi(x, y, z);
        });
        // Dirichlet values live in the exterior ghost cells and are the
        // boundary-face values (the face is half a cell away).
        auto const g = level_bc[level]->arrays();
        ParallelFor(*level_bc[level], IntVect(1),
        [=] AMREX_GPU_DEVICE (int box, int i, int j, int k) noexcept
        {
            int const ii[3] = {i, j, k};
            Real xyz[3];
            for (int d = 0; d < 3; ++d) {
                xyz[d] = (Real(ii[d]) + Real(0.5)) * dx[d];
                if (ii[d] < domain.smallEnd(d)) { xyz[d] = Real(0); }
                if (ii[d] > domain.bigEnd(d)) { xyz[d] = Real(1); }
            }
            g[box](i, j, k) = mms_phi(xyz[0], xyz[1], xyz[2]);
        });
        for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
            auto const b = bcoef[level][direction]->arrays();
            ParallelFor(*bcoef[level][direction],
            [=] AMREX_GPU_DEVICE (int box, int i, int j, int k) noexcept
            {
                Real const x = (Real(i) + Real(direction == 0 ? 0 : 0.5)) * dx[0];
                Real const y = (Real(j) + Real(direction == 1 ? 0 : 0.5)) * dx[1];
                Real const z = (Real(k) + Real(direction == 2 ? 0 : 0.5)) * dx[2];
                b[box](i, j, k) = mms_b(x, y, z);
            });
        }
    }
    Gpu::streamSynchronize();

    Array<LinOpBCType, AMREX_SPACEDIM> const dirichlet{
        AMREX_D_DECL(LinOpBCType::Dirichlet, LinOpBCType::Dirichlet,
                     LinOpBCType::Dirichlet)};
    MLABecLapAMG solver(hierarchy.geom, hierarchy.grids, hierarchy.dmap);
    solver.setMaxIter(1000);
    solver.setup(Real(1), Real(1), get_level_const_ptrs(acoef),
                 get_face_const_ptrs(bcoef), dirichlet, dirichlet,
                 get_level_const_ptrs(level_bc));

    // Local truncation error: the discrete operator applied to the exact
    // cell-center values, minus the right-hand side.
    solver.residual(get_level_ptrs(tau), get_level_const_ptrs(exact),
                    get_level_const_ptrs(rhs));

    auto const info = solver.solve(get_level_ptrs(solution),
                                   get_level_const_ptrs(rhs), Real(1.e-12),
                                   Real(0));
    amrex::Print() << "MMS n_cell=" << n_cell << ": GMRES iterations="
                   << info.iterations << ", relative residual="
                   << info.relative_residual << '\n';

    auto error = make_cell_data(hierarchy, 1, 0);
    for (int level = 0; level < nlevels; ++level) {
        MultiFab::LinComb(*error[level], Real(1), *solution[level], 0,
                          Real(-1), *exact[level], 0, 0, 1, 0);
    }
    return measure(hierarchy, error, tau);
}

} // namespace

void
run_mms_convergence ()
{
    Vector<int> n_cells{16, 32, 64};
    // queryarr never shrinks its argument, so query into an empty vector.
    Vector<int> requested;
    if (ParmParse("mms").queryarr("n_cells", requested)) {
        n_cells = requested;
    }
    std::string stencil = "hypre";
    ParmParse("mlabeclap_amg").query("cf_stencil", stencil);

    Vector<BandStats> stats;
    for (int n : n_cells) { stats.push_back(run_mms_case(n)); }

    auto order = [] (Real coarse, Real fine) {
        return (coarse > Real(0) && fine > Real(0))
            ? std::log2(coarse / fine) : Real(0);
    };
    char const* band_names[num_bands] = {"interior", "fine-iface",
                                         "coarse-iface", "phys-bdry"};
    auto print_row = [&] (char const* label, auto const& value) {
        amrex::Print() << "  " << std::left << std::setw(20) << label
                       << std::right;
        for (int r = 0; r < static_cast<int>(stats.size()); ++r) {
            Real const v = value(stats[r]);
            amrex::Print() << std::scientific << std::setprecision(3)
                           << std::setw(12) << v;
            if (r > 0) {
                amrex::Print() << " (" << std::fixed << std::setprecision(2)
                               << std::setw(5)
                               << order(value(stats[r - 1]), v) << ")";
            } else {
                amrex::Print() << std::string(8, ' ');
            }
        }
        amrex::Print() << '\n';
    };
    amrex::Print() << "\nMMS convergence, cf_stencil=" << stencil
                   << " (two levels, ratio 2, fine patch [1/4,3/4]^3);"
                   << " observed order vs previous column in parentheses\n"
                   << "  " << std::left << std::setw(20) << "coarse n_cell"
                   << std::right;
    for (int n : n_cells) {
        amrex::Print() << std::setw(12) << n << std::string(8, ' ');
    }
    amrex::Print() << '\n';
    print_row("error L2", [] (BandStats const& s) { return s.l2_error; });
    print_row("error max", [] (BandStats const& s) { return s.max_error_all; });
    for (int b = 0; b < num_bands; ++b) {
        std::string const label = std::string("error max ") + band_names[b];
        print_row(label.c_str(),
                  [b] (BandStats const& s) { return s.max_error[b]; });
    }
    for (int b = 0; b < num_bands; ++b) {
        std::string const label = std::string("tau max ") + band_names[b];
        print_row(label.c_str(),
                  [b] (BandStats const& s) { return s.max_tau[b]; });
    }
}

} // namespace fld_test
