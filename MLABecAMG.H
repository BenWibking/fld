#ifndef MLABEC_AMG_H_
#define MLABEC_AMG_H_

#include <AMReX_AMG.H>
#include <AMReX_Array.H>
#include <AMReX_Geometry.H>
#include <AMReX_LO_BCTYPES.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Vector.H>

#include <memory>
#include <string>

namespace fld_test
{

/** Cell-centered data for `a phi + b dphi/dn = f` Robin boundaries. */
struct RobinBCData
{
    amrex::Vector<amrex::MultiFab const*> a;
    amrex::Vector<amrex::MultiFab const*> b;
    amrex::Vector<amrex::MultiFab const*> f;
};

struct SolveInfo
{
    int iterations = 0;
    amrex::Real relative_residual = amrex::Real(0);
    amrex::Real absolute_residual = amrex::Real(0);
};

struct MLABecAssemblyDiagnostics
{
    amrex::Long global_rows = 0;
    amrex::Long local_rows = 0;
    amrex::Long coarse_fine_connections = 0;
    amrex::Real minimum_diagonal = amrex::Real(0);
    amrex::Real maximum_off_diagonal = amrex::Real(0);
    int setup_generation = 0;
};

/**
 * Cell-centered, single-component, multilevel ABec solver using AMReX's
 * native `SpMatrix`, `GMRES_MV`, and `AMG` algebra.
 *
 * The operator is
 * `ascalar * acoef * phi - bscalar * div(bcoef * grad(phi))`.
 * `acoef` is cell centered and `bcoef` is face centered.  Solution fields may
 * have ghosts; right-hand sides must have none.  `setup` copies all data it
 * needs, so coefficient and boundary fields need only remain valid for that
 * call.  A later `setup` replaces the matrix and preconditioner, while any
 * number of `solve` calls reuse the current setup.
 *
 * Dirichlet values are read from the exterior ghost cells of
 * `level_bc_data`.  Homogeneous Neumann needs no data.  Robin and Marshak
 * boundaries use `RobinBCData` with the convention
 * `a * phi + b * dphi/dn = f`.
 */
class MLABecLapAMG
{
  public:
    MLABecLapAMG (
        amrex::Vector<amrex::Geometry> geom,
        amrex::Vector<amrex::BoxArray> grids,
        amrex::Vector<amrex::DistributionMapping> dmap,
        amrex::AMG<amrex::Real>::Options options = {},
        std::string parmparse_prefix = "mlabeclap_amg");

    ~MLABecLapAMG ();

    MLABecLapAMG (MLABecLapAMG const&) = delete;
    MLABecLapAMG& operator=(MLABecLapAMG const&) = delete;
    MLABecLapAMG (MLABecLapAMG&&) = delete;
    MLABecLapAMG& operator=(MLABecLapAMG&&) = delete;

    void setVerbose (int value);
    void setMaxIter (int value);

    void setup (
        amrex::Real ascalar, amrex::Real bscalar,
        amrex::Vector<amrex::MultiFab const*> const& acoef,
        amrex::Vector<
            amrex::Array<amrex::MultiFab const*, AMREX_SPACEDIM>> const& bcoef,
        amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM> const& lobc,
        amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM> const& hibc,
        amrex::Vector<amrex::MultiFab const*> const& level_bc_data,
        RobinBCData const& robin_bc_data = {});

    [[nodiscard]] SolveInfo solve (
        amrex::Vector<amrex::MultiFab*> const& solution,
        amrex::Vector<amrex::MultiFab const*> const& rhs,
        amrex::Real relative_tolerance, amrex::Real absolute_tolerance);

    [[nodiscard]] amrex::AMG<amrex::Real>::Diagnostics const& diagnostics ()
        const noexcept;

    [[nodiscard]] MLABecAssemblyDiagnostics const& assembly_diagnostics ()
        const noexcept;

  private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace fld_test

#endif
