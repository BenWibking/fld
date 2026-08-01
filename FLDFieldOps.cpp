#include "FLDFieldOps.H"

#include <AMReX.H>
#include <AMReX_BC_TYPES.H>
#include <AMReX_BCRec.H>
#include <AMReX_FillPatchUtil.H>
#include <AMReX_Interpolater.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Reduce.H>

#include <cmath>
#include <limits>

namespace fld_test
{

using namespace amrex;

namespace
{

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE Real
limiter (Real r) noexcept
{
    if (r > Real(1.e8)) {
        return Real(1) / r;
    }
    return (Real(2) + r) / (Real(6) + Real(3) * r + r * r);
}

void
assert_compatible (LevelData const& lhs, LevelData const& rhs)
{
    AMREX_ALWAYS_ASSERT(lhs.size() == rhs.size());
    for (int level = 0; level < static_cast<int>(lhs.size()); ++level) {
        AMREX_ALWAYS_ASSERT(lhs[level] != nullptr && rhs[level] != nullptr);
        AMREX_ALWAYS_ASSERT(lhs[level]->boxArray() == rhs[level]->boxArray());
        AMREX_ALWAYS_ASSERT(lhs[level]->DistributionMap() ==
                            rhs[level]->DistributionMap());
        AMREX_ALWAYS_ASSERT(lhs[level]->nComp() == rhs[level]->nComp());
    }
}

Real
cell_volume (Geometry const& geometry) noexcept
{
    auto const dx = geometry.CellSizeArray();
    return AMREX_D_TERM(dx[0], *dx[1], *dx[2]);
}

} // namespace

LevelData
make_cell_data (DiffusionHierarchy const& hierarchy, int ncomp, int nghost)
{
    AMREX_ALWAYS_ASSERT(ncomp > 0 && nghost >= 0);
    LevelData result(hierarchy.geom.size());
    for (int level = 0; level < static_cast<int>(result.size()); ++level) {
        result[level] = std::make_unique<MultiFab>(
            hierarchy.grids[level], hierarchy.dmap[level], ncomp, nghost);
    }
    return result;
}

FaceData
make_face_data (DiffusionHierarchy const& hierarchy, int ncomp, int nghost)
{
    AMREX_ALWAYS_ASSERT(ncomp > 0 && nghost >= 0);
    FaceData result(hierarchy.geom.size());
    for (int level = 0; level < static_cast<int>(result.size()); ++level) {
        for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
            BoxArray face_grids = hierarchy.grids[level];
            face_grids.convert(IntVect::TheDimensionVector(direction));
            result[level][direction] = std::make_unique<MultiFab>(
                face_grids, hierarchy.dmap[level], ncomp, nghost);
        }
    }
    return result;
}

LevelData
clone_level_data (LevelData const& source)
{
    LevelData result(source.size());
    for (int level = 0; level < static_cast<int>(source.size()); ++level) {
        AMREX_ALWAYS_ASSERT(source[level] != nullptr);
        result[level] = std::make_unique<MultiFab>(
            source[level]->boxArray(), source[level]->DistributionMap(),
            source[level]->nComp(), source[level]->nGrowVect());
        result[level]->setVal(Real(0));
        MultiFab::Copy(*result[level], *source[level], 0, 0,
                       source[level]->nComp(), source[level]->nGrow());
    }
    return result;
}

void
copy_level_data (LevelData& destination, LevelData const& source, int nghost)
{
    assert_compatible(destination, source);
    for (int level = 0; level < static_cast<int>(source.size()); ++level) {
        AMREX_ALWAYS_ASSERT(nghost <= destination[level]->nGrow() &&
                            nghost <= source[level]->nGrow());
        MultiFab::Copy(*destination[level], *source[level], 0, 0,
                       source[level]->nComp(), nghost);
    }
}

void
set_level_data (LevelData& data, Real value)
{
    for (auto& field : data) {
        AMREX_ALWAYS_ASSERT(field != nullptr);
        field->setVal(value);
    }
}

void
lincomb_level_data (LevelData& destination, Real a, LevelData const& lhs,
                    Real b, LevelData const& rhs)
{
    assert_compatible(destination, lhs);
    assert_compatible(lhs, rhs);
    for (int level = 0; level < static_cast<int>(lhs.size()); ++level) {
        MultiFab::LinComb(*destination[level], a, *lhs[level], 0, b,
                          *rhs[level], 0, 0, lhs[level]->nComp(), 0);
    }
}

void
saxpy_level_data (LevelData& destination, Real a, LevelData const& source)
{
    assert_compatible(destination, source);
    for (int level = 0; level < static_cast<int>(source.size()); ++level) {
        MultiFab::Saxpy(*destination[level], a, *source[level], 0, 0,
                        source[level]->nComp(), 0);
    }
}

void
fill_level_ghosts (LevelData& data, DiffusionHierarchy const& hierarchy)
{
    AMREX_ALWAYS_ASSERT(data.size() == hierarchy.geom.size());
    for (int level = 0; level < static_cast<int>(data.size()); ++level) {
        MultiFab& field = *data[level];
        int const nghost = field.nGrow();
        if (nghost == 0) {
            continue;
        }
        for (MFIter mfi(field); mfi.isValid(); ++mfi) {
            Box const valid = mfi.validbox();
            Box const grown = mfi.fabbox();
            auto const array = field.array(mfi);
            auto const lo = amrex::lbound(valid);
            auto const hi = amrex::ubound(valid);
            ParallelFor(grown, field.nComp(),
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
            {
                if (!valid.contains(IntVect(AMREX_D_DECL(i, j, k)))) {
                    int const ii = amrex::max(lo.x, amrex::min(i, hi.x));
                    int const jj = amrex::max(lo.y, amrex::min(j, hi.y));
#if (AMREX_SPACEDIM == 3)
                    int const kk = amrex::max(lo.z, amrex::min(k, hi.z));
#else
                    int const kk = k;
#endif
                    array(i, j, k, n) = array(ii, jj, kk, n);
                }
            });
        }
        field.FillBoundary(hierarchy.geom[level].periodicity());
        if (level > 0) {
            Vector<MultiFab*> coarse_source{data[level - 1].get()};
            Vector<MultiFab*> fine_source{data[level].get()};
            Vector<Real> const times{Real(0)};
            Vector<BCRec> boundary_records(field.nComp());
            for (auto& record : boundary_records) {
                for (int direction = 0; direction < AMREX_SPACEDIM;
                     ++direction) {
                    int const type =
                        hierarchy.geom[level].isPeriodic(direction)
                            ? BCType::int_dir
                            : BCType::foextrap;
                    record.setLo(direction, type);
                    record.setHi(direction, type);
                }
            }
            FillPatchTwoLevels(
                field, IntVect(nghost), IntVect(0), Real(0), coarse_source,
                times, fine_source, times, 0, 0, field.nComp(),
                hierarchy.geom[level - 1], hierarchy.geom[level],
                hierarchy.ref_ratio[level - 1], &cell_cons_interp,
                boundary_records, 0);
        }
    }
}

void
average_down_hierarchy (LevelData& data,
                        DiffusionHierarchy const& hierarchy)
{
    AMREX_ALWAYS_ASSERT(data.size() == hierarchy.geom.size());
    for (int level = static_cast<int>(data.size()) - 2; level >= 0; --level) {
        amrex::average_down(*data[level + 1], *data[level],
                            hierarchy.geom[level + 1], hierarchy.geom[level],
                            0, data[level]->nComp(),
                            hierarchy.ref_ratio[level]);
    }
}

Vector<MultiFab*>
get_level_ptrs (LevelData& data)
{
    Vector<MultiFab*> result;
    result.reserve(data.size());
    for (auto& field : data) {
        result.push_back(field.get());
    }
    return result;
}

Vector<MultiFab const*>
get_level_const_ptrs (LevelData const& data)
{
    Vector<MultiFab const*> result;
    result.reserve(data.size());
    for (auto const& field : data) {
        result.push_back(field.get());
    }
    return result;
}

Vector<Array<MultiFab const*, AMREX_SPACEDIM>>
get_face_const_ptrs (FaceData const& data)
{
    Vector<Array<MultiFab const*, AMREX_SPACEDIM>> result(data.size());
    for (int level = 0; level < static_cast<int>(data.size()); ++level) {
        for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
            result[level][direction] = data[level][direction].get();
        }
    }
    return result;
}

Real
composite_weighted_dot (LevelData const& lhs, LevelData const& rhs,
                        DiffusionHierarchy const& hierarchy,
                        Vector<iMultiFab> const& masks)
{
    assert_compatible(lhs, rhs);
    AMREX_ALWAYS_ASSERT(lhs.size() == masks.size());
    ReduceOps<ReduceOpSum> reduce_op;
    ReduceData<Real> reduce_data(reduce_op);
    using Tuple = typename decltype(reduce_data)::Type;
    for (int level = 0; level < static_cast<int>(lhs.size()); ++level) {
        auto const volume = cell_volume(hierarchy.geom[level]);
        for (MFIter mfi(*lhs[level]); mfi.isValid(); ++mfi) {
            Box const& box = mfi.validbox();
            auto const x = lhs[level]->const_array(mfi);
            auto const y = rhs[level]->const_array(mfi);
            auto const mask = masks[level].const_array(mfi);
            reduce_op.eval(box, reduce_data,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept -> Tuple
            {
                return {mask(i, j, k) ? volume * x(i, j, k) * y(i, j, k)
                                      : Real(0)};
            });
        }
    }
    Real result = amrex::get<0>(reduce_data.value(reduce_op));
    ParallelDescriptor::ReduceRealSum(result);
    return result;
}

Real
composite_volume_sum (LevelData const& data,
                      DiffusionHierarchy const& hierarchy,
                      Vector<iMultiFab> const& masks)
{
    AMREX_ALWAYS_ASSERT(data.size() == masks.size());
    ReduceOps<ReduceOpSum> reduce_op;
    ReduceData<Real> reduce_data(reduce_op);
    using Tuple = typename decltype(reduce_data)::Type;
    for (int level = 0; level < static_cast<int>(data.size()); ++level) {
        Real const volume = cell_volume(hierarchy.geom[level]);
        for (MFIter mfi(*data[level]); mfi.isValid(); ++mfi) {
            auto const array = data[level]->const_array(mfi);
            auto const mask = masks[level].const_array(mfi);
            reduce_op.eval(mfi.validbox(), reduce_data,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept -> Tuple
            {
                return {mask(i, j, k) ? volume * array(i, j, k) : Real(0)};
            });
        }
    }
    Real result = amrex::get<0>(reduce_data.value(reduce_op));
    ParallelDescriptor::ReduceRealSum(result);
    return result;
}

std::pair<Real, Real>
composite_minimum_maximum (LevelData const& data,
                           Vector<iMultiFab> const& masks)
{
    AMREX_ALWAYS_ASSERT(data.size() == masks.size());
    ReduceOps<ReduceOpMin, ReduceOpMax> reduce_op;
    ReduceData<Real, Real> reduce_data(reduce_op);
    using Tuple = typename decltype(reduce_data)::Type;
    Real const high = std::numeric_limits<Real>::max();
    Real const low = std::numeric_limits<Real>::lowest();
    for (int level = 0; level < static_cast<int>(data.size()); ++level) {
        for (MFIter mfi(*data[level]); mfi.isValid(); ++mfi) {
            auto const array = data[level]->const_array(mfi);
            auto const mask = masks[level].const_array(mfi);
            reduce_op.eval(mfi.validbox(), reduce_data,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept -> Tuple
            {
                return mask(i, j, k) ? Tuple{array(i, j, k), array(i, j, k)}
                                     : Tuple{high, low};
            });
        }
    }
    auto const value = reduce_data.value(reduce_op);
    Real minimum = amrex::get<0>(value);
    Real maximum = amrex::get<1>(value);
    ParallelDescriptor::ReduceRealMin(minimum);
    ParallelDescriptor::ReduceRealMax(maximum);
    return {minimum, maximum};
}

Real
composite_maximum_relative_change (LevelData const& lhs, LevelData const& rhs,
                                   Vector<iMultiFab> const& masks)
{
    assert_compatible(lhs, rhs);
    AMREX_ALWAYS_ASSERT(lhs.size() == masks.size());
    ReduceOps<ReduceOpMax> reduce_op;
    ReduceData<Real> reduce_data(reduce_op);
    using Tuple = typename decltype(reduce_data)::Type;
    for (int level = 0; level < static_cast<int>(lhs.size()); ++level) {
        for (MFIter mfi(*lhs[level]); mfi.isValid(); ++mfi) {
            auto const x = lhs[level]->const_array(mfi);
            auto const y = rhs[level]->const_array(mfi);
            auto const mask = masks[level].const_array(mfi);
            reduce_op.eval(mfi.validbox(), reduce_data,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept -> Tuple
            {
                Real const scale = amrex::max(
                    amrex::max(amrex::Math::abs(x(i, j, k)),
                               amrex::Math::abs(y(i, j, k))),
                    Real(1.e-12));
                return {mask(i, j, k)
                            ? amrex::Math::abs(x(i, j, k) - y(i, j, k)) /
                                  scale
                            : Real(0)};
            });
        }
    }
    Real result = amrex::get<0>(reduce_data.value(reduce_op));
    ParallelDescriptor::ReduceRealMax(result);
    return result;
}

bool
composite_all_finite (LevelData const& data, Vector<iMultiFab> const& masks)
{
    AMREX_ALWAYS_ASSERT(data.size() == masks.size());
    ReduceOps<ReduceOpMin> reduce_op;
    ReduceData<int> reduce_data(reduce_op);
    using Tuple = typename decltype(reduce_data)::Type;
    for (int level = 0; level < static_cast<int>(data.size()); ++level) {
        for (MFIter mfi(*data[level]); mfi.isValid(); ++mfi) {
            auto const array = data[level]->const_array(mfi);
            auto const mask = masks[level].const_array(mfi);
            reduce_op.eval(mfi.validbox(), reduce_data,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept -> Tuple
            {
                Real const value = array(i, j, k);
                bool const finite = value == value &&
                                    amrex::Math::abs(value) <
                                        std::numeric_limits<Real>::max();
                return {(!mask(i, j, k) || finite) ? 1 : 0};
            });
        }
    }
    int result = amrex::get<0>(reduce_data.value(reduce_op));
    ParallelDescriptor::ReduceIntMin(result);
    return result != 0;
}

void
compute_diffusion (DiffusionHierarchy const& hierarchy, LevelData& energy,
                   LevelData& extinction, LevelData& diffusion,
                   PhysicalBoundaryData const& boundary, bool limited,
                   Real* maximum_flux_fraction)
{
    assert_compatible(energy, extinction);
    assert_compatible(energy, diffusion);
    fill_level_ghosts(energy, hierarchy);
    fill_level_ghosts(extinction, hierarchy);

    ReduceOps<ReduceOpMax> reduce_op;
    ReduceData<Real> reduce_data(reduce_op);
    using Tuple = typename decltype(reduce_data)::Type;
    for (int level = 0; level < static_cast<int>(energy.size()); ++level) {
        auto const dx = hierarchy.geom[level].CellSizeArray();
        Box const domain = hierarchy.geom[level].Domain();
        auto const dlo = amrex::lbound(domain);
        auto const dhi = amrex::ubound(domain);
        auto const periodic = hierarchy.geom[level].isPeriodicArray();
        for (MFIter mfi(*energy[level]); mfi.isValid(); ++mfi) {
            auto const e = energy[level]->const_array(mfi);
            auto const chi = extinction[level]->const_array(mfi);
            auto const d = diffusion[level]->array(mfi);
            auto const lo_type = boundary.lo;
            auto const hi_type = boundary.hi;
            auto const lo_value = boundary.lo_value;
            auto const hi_value = boundary.hi_value;
            Box const& box = mfi.validbox();
            reduce_op.eval(box, reduce_data,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept -> Tuple
            {
                Real emx;
                if (i > dlo.x || periodic[0]) {
                    emx = e(i - 1, j, k);
                } else if (lo_type[0] == LinOpBCType::Dirichlet ||
                           lo_type[0] == LinOpBCType::Robin ||
                           lo_type[0] == LinOpBCType::Marshak) {
                    emx = Real(2) * lo_value[0] - e(i, j, k);
                } else {
                    emx = e(i, j, k);
                }
                Real epx;
                if (i < dhi.x || periodic[0]) {
                    epx = e(i + 1, j, k);
                } else if (hi_type[0] == LinOpBCType::Dirichlet ||
                           hi_type[0] == LinOpBCType::Robin ||
                           hi_type[0] == LinOpBCType::Marshak) {
                    epx = Real(2) * hi_value[0] - e(i, j, k);
                } else {
                    epx = e(i, j, k);
                }
                Real emy;
                if (j > dlo.y || periodic[1]) {
                    emy = e(i, j - 1, k);
                } else if (lo_type[1] == LinOpBCType::Dirichlet ||
                           lo_type[1] == LinOpBCType::Robin ||
                           lo_type[1] == LinOpBCType::Marshak) {
                    emy = Real(2) * lo_value[1] - e(i, j, k);
                } else {
                    emy = e(i, j, k);
                }
                Real epy;
                if (j < dhi.y || periodic[1]) {
                    epy = e(i, j + 1, k);
                } else if (hi_type[1] == LinOpBCType::Dirichlet ||
                           hi_type[1] == LinOpBCType::Robin ||
                           hi_type[1] == LinOpBCType::Marshak) {
                    epy = Real(2) * hi_value[1] - e(i, j, k);
                } else {
                    epy = e(i, j, k);
                }
                Real const gx = (epx - emx) / (Real(2) * dx[0]);
                Real const gy = (epy - emy) / (Real(2) * dx[1]);
                Real const extinction_value = chi(i, j, k);
                Real const energy_value = amrex::max(e(i, j, k), Real(1.e-30));
                Real const r = std::sqrt(gx * gx + gy * gy) /
                               (extinction_value * energy_value);
                Real const lambda = limited ? limiter(r) : Real(1) / Real(3);
                d(i, j, k) = lambda / extinction_value;
                return {lambda * r};
            });
        }
    }
    if (maximum_flux_fraction != nullptr) {
        *maximum_flux_fraction =
            amrex::get<0>(reduce_data.value(reduce_op));
        ParallelDescriptor::ReduceRealMax(*maximum_flux_fraction);
    }
}

void
fill_face_coefficients (DiffusionHierarchy const& hierarchy,
                        LevelData& diffusion, LevelData* extinction,
                        FaceData& face_coefficients,
                        bool use_surface_opacity)
{
    AMREX_ALWAYS_ASSERT(diffusion.size() == face_coefficients.size());
    if (use_surface_opacity) {
        AMREX_ALWAYS_ASSERT(extinction != nullptr);
        assert_compatible(diffusion, *extinction);
    }
    fill_level_ghosts(diffusion, hierarchy);
    if (extinction != nullptr) {
        fill_level_ghosts(*extinction, hierarchy);
    }

    for (int level = 0; level < static_cast<int>(diffusion.size()); ++level) {
        Box const domain = hierarchy.geom[level].Domain();
        auto const dlo = amrex::lbound(domain);
        auto const dhi = amrex::ubound(domain);
        auto const dx = hierarchy.geom[level].CellSizeArray();
        for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
            MultiFab& face = *face_coefficients[level][direction];
            for (MFIter mfi(face); mfi.isValid(); ++mfi) {
                auto const d = diffusion[level]->const_array(mfi);
                Array4<Real const> chi;
                if (extinction != nullptr) {
                    chi = (*extinction)[level]->const_array(mfi);
                }
                auto const b = face.array(mfi);
                Box const& box = mfi.validbox();
                ParallelFor(box,
                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    int il = i;
                    int jl = j;
                    int ir = i;
                    int jr = j;
                    if (direction == 0) {
                        il = i - 1;
                    } else {
                        jl = j - 1;
                    }
                    bool const left_inside = il >= dlo.x && il <= dhi.x &&
                                             jl >= dlo.y && jl <= dhi.y;
                    bool const right_inside = ir >= dlo.x && ir <= dhi.x &&
                                              jr >= dlo.y && jr <= dhi.y;
                    if (!left_inside) {
                        il = ir;
                        jl = jr;
                    }
                    if (!right_inside) {
                        ir = il;
                        jr = jl;
                    }
                    Real const dl = d(il, jl, k);
                    Real const dr = d(ir, jr, k);
                    if (!use_surface_opacity || !left_inside ||
                        !right_inside) {
                        b(i, j, k) = Real(2) * dl * dr / (dl + dr);
                        return;
                    }
                    Real const chil = chi(il, jl, k);
                    Real const chir = chi(ir, jr, k);
                    Real const arithmetic = Real(0.5) * (chil + chir);
                    Real const harmonic =
                        Real(2) * chil * chir / (chil + chir);
                    Real const surface = Real(4) / (Real(3) * dx[direction]);
                    Real const face_extinction = amrex::min(
                        arithmetic, amrex::max(harmonic, surface));
                    Real const face_limiter =
                        amrex::max(dl * chil, dr * chir);
                    b(i, j, k) = face_limiter / face_extinction;
                });
            }
        }
    }
}

} // namespace fld_test
