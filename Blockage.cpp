#include "FLDTestCommon.H"
#include "NewtonKrylovSolver.H"

#include <AMReX_GMRES.H>
#include <AMReX_MFIter.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Reduce.H>

#include <cmath>
#include <limits>
#include <string>

namespace fld_test
{

using namespace amrex;

namespace
{

Real constexpr initial_radiation_energy = Real(1.e-5);
Real constexpr material_conductivity = Real(0.005);

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE Real
positive_temperature (Real temperature) noexcept
{
    return amrex::max(temperature, Real(1.e-6));
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE Real
cube (Real value) noexcept
{
    return value * value * value;
}

void
initialize_fields (DiffusionHierarchy const& hierarchy, LevelData& energy,
                   LevelData& temperature, LevelData& atomic_number,
                   LevelData& sigma)
{
    Real const initial_temperature =
        std::sqrt(std::sqrt(initial_radiation_energy));
    for (int level = 0; level < static_cast<int>(energy.size()); ++level) {
        auto const dx = hierarchy.geom[level].CellSizeArray();
        auto const problo = hierarchy.geom[level].ProbLoArray();
        for (MFIter mfi(*energy[level]); mfi.isValid(); ++mfi) {
            auto const e = energy[level]->array(mfi);
            auto const t = temperature[level]->array(mfi);
            auto const z = atomic_number[level]->array(mfi);
            auto const absorption = sigma[level]->array(mfi);
            ParallelFor(mfi.validbox(),
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                Real const x = problo[0] + (Real(i) + Real(0.5)) * dx[0];
                Real const y = problo[1] + (Real(j) + Real(0.5)) * dx[1];
                Real const atomic =
                    x > Real(1) / Real(3) && x < Real(2) / Real(3) &&
                            y > Real(1) / Real(3) && y < Real(2) / Real(3)
                        ? Real(10)
                        : Real(1);
                e(i, j, k) = initial_radiation_energy;
                t(i, j, k) = initial_temperature;
                z(i, j, k) = atomic;
                absorption(i, j, k) =
                    cube(atomic) / cube(initial_temperature);
            });
        }
    }
}

void
update_absorption_and_cell_diffusion (
    DiffusionHierarchy const& hierarchy, LevelData& energy,
    LevelData& temperature, LevelData& atomic_number, LevelData& sigma,
    LevelData& radiation_diffusion, LevelData& material_diffusion)
{
    fill_level_ghosts(energy, hierarchy);
    fill_level_ghosts(temperature, hierarchy);
    fill_level_ghosts(atomic_number, hierarchy);
    for (int level = 0; level < static_cast<int>(energy.size()); ++level) {
        auto const dx = hierarchy.geom[level].CellSizeArray();
        Box const domain = hierarchy.geom[level].Domain();
        auto const dlo = amrex::lbound(domain);
        auto const dhi = amrex::ubound(domain);
        for (MFIter mfi(*energy[level]); mfi.isValid(); ++mfi) {
            auto const e = energy[level]->const_array(mfi);
            auto const t = temperature[level]->const_array(mfi);
            auto const z = atomic_number[level]->const_array(mfi);
            auto const absorption = sigma[level]->array(mfi);
            auto const dr = radiation_diffusion[level]->array(mfi);
            auto const dt = material_diffusion[level]->array(mfi);
            ParallelFor(mfi.validbox(),
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                Real const temperature_value =
                    positive_temperature(t(i, j, k));
                Real const sigma_value = cube(z(i, j, k)) /
                                         cube(temperature_value);
                int const im = amrex::max(i - 1, dlo.x);
                int const ip = amrex::min(i + 1, dhi.x);
                int const jm = amrex::max(j - 1, dlo.y);
                int const jp = amrex::min(j + 1, dhi.y);
                Real const gx = (e(ip, j, k) - e(im, j, k)) /
                                (Real(ip - im) * dx[0]);
                Real const gy = (e(i, jp, k) - e(i, jm, k)) /
                                (Real(jp - jm) * dx[1]);
                Real const limiter_term =
                    std::sqrt(gx * gx + gy * gy) /
                    amrex::max(e(i, j, k), Real(1.e-30));
                absorption(i, j, k) = sigma_value;
                dr(i, j, k) = Real(1) /
                            (Real(3) * sigma_value + limiter_term);
                dt(i, j, k) = material_conductivity *
                            temperature_value * temperature_value *
                            std::sqrt(temperature_value);
            });
        }
    }
}

void
fill_paper_face_coefficients (
    DiffusionHierarchy const& hierarchy, LevelData& energy,
    LevelData& temperature, LevelData& atomic_number,
    FaceData& radiation_bcoef, FaceData& material_bcoef)
{
    fill_level_ghosts(energy, hierarchy);
    fill_level_ghosts(temperature, hierarchy);
    fill_level_ghosts(atomic_number, hierarchy);
    for (int level = 0; level < static_cast<int>(energy.size()); ++level) {
        Box const domain = hierarchy.geom[level].Domain();
        auto const dlo = amrex::lbound(domain);
        auto const dhi = amrex::ubound(domain);
        auto const dx = hierarchy.geom[level].CellSizeArray();
        for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
            MultiFab& radiation_face = *radiation_bcoef[level][direction];
            MultiFab& material_face = *material_bcoef[level][direction];
            for (MFIter mfi(radiation_face); mfi.isValid(); ++mfi) {
                auto const e = energy[level]->const_array(mfi);
                auto const t = temperature[level]->const_array(mfi);
                auto const z = atomic_number[level]->const_array(mfi);
                auto const dr = radiation_face.array(mfi);
                auto const dt = material_face.array(mfi);
                ParallelFor(mfi.validbox(),
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
                    Real const tl = positive_temperature(t(il,jl,k));
                    Real const tr = positive_temperature(t(ir,jr,k));
                    Real const face_temperature = Real(0.5) * (tl + tr);
                    Real const face_energy = amrex::max(
                        Real(0.5) * (e(il,jl,k) + e(ir,jr,k)), Real(1.e-30));
                    Real const gradient =
                        left_inside && right_inside
                            ? std::abs(e(ir,jr,k) - e(il,jl,k)) /
                                  dx[direction]
                            : Real(0);
                    Real const limiter_term = gradient / face_energy;
                    Real const sigma_left =
                        cube(z(il,jl,k)) / cube(face_temperature);
                    Real const sigma_right =
                        cube(z(ir,jr,k)) / cube(face_temperature);
                    Real const dl = Real(1) /
                        (Real(3) * sigma_left + limiter_term);
                    Real const dright = Real(1) /
                        (Real(3) * sigma_right + limiter_term);
                    dr(i,j,k) = Real(2) * dl * dright / (dl + dright);
                    dt(i,j,k) = material_conductivity *
                        face_temperature * face_temperature *
                        std::sqrt(face_temperature);
                });
            }
        }
    }
}

void
fill_radiation_robin_data (DiffusionHierarchy const& hierarchy,
                           LevelData const& sigma, LevelData& robin_a,
                           LevelData& robin_b, LevelData& robin_f)
{
    set_level_data(robin_a, Real(0.25));
    set_level_data(robin_f, Real(0));
    for (int level = 0; level < static_cast<int>(sigma.size()); ++level) {
        int const inlet = hierarchy.geom[level].Domain().smallEnd(0);
        for (MFIter mfi(*sigma[level]); mfi.isValid(); ++mfi) {
            auto const absorption = sigma[level]->const_array(mfi);
            auto const b = robin_b[level]->array(mfi);
            auto const f = robin_f[level]->array(mfi);
            ParallelFor(mfi.validbox(),
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                b(i,j,k) = Real(1) / (Real(6) * absorption(i,j,k));
                if (i == inlet) {
                    f(i,j,k) = Real(1);
                }
            });
        }
    }
}

void
fill_linearized_systems (LevelData const& old_energy,
                         LevelData const& old_temperature,
                         LevelData const& energy_iterate,
                         LevelData const& temperature_iterate,
                         LevelData const& sigma, Real dt,
                         LevelData& energy_acoef, LevelData& energy_rhs,
                         LevelData& temperature_acoef,
                         LevelData& temperature_rhs)
{
    Real const inverse_dt = Real(1) / dt;
    for (int level = 0; level < static_cast<int>(sigma.size()); ++level) {
        for (MFIter mfi(*sigma[level]); mfi.isValid(); ++mfi) {
            auto const eo = old_energy[level]->const_array(mfi);
            auto const to = old_temperature[level]->const_array(mfi);
            auto const e = energy_iterate[level]->const_array(mfi);
            auto const t = temperature_iterate[level]->const_array(mfi);
            auto const absorption = sigma[level]->const_array(mfi);
            auto const ea = energy_acoef[level]->array(mfi);
            auto const erhs = energy_rhs[level]->array(mfi);
            auto const ta = temperature_acoef[level]->array(mfi);
            auto const trhs = temperature_rhs[level]->array(mfi);
            ParallelFor(mfi.validbox(),
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                Real const temperature_value = positive_temperature(t(i,j,k));
                Real const t3 = cube(temperature_value);
                Real const t4 = temperature_value * t3;
                Real const sigma_value = absorption(i,j,k);
                ea(i,j,k) = inverse_dt + sigma_value;
                erhs(i,j,k) = inverse_dt * eo(i,j,k) + sigma_value * t4;
                ta(i,j,k) = inverse_dt + Real(4) * sigma_value * t3;
                trhs(i,j,k) = inverse_dt * to(i,j,k) +
                              sigma_value * e(i,j,k) +
                              Real(3) * sigma_value * t4;
            });
        }
    }
}

struct CoupledVector
{
    LevelData energy;
    LevelData temperature;
};

CoupledVector
make_coupled_vector (DiffusionHierarchy const& hierarchy, int nghost)
{
    return {make_cell_data(hierarchy, 1, nghost),
            make_cell_data(hierarchy, 1, nghost)};
}

CoupledVector
clone_coupled_vector (CoupledVector const& source)
{
    return {clone_level_data(source.energy),
            clone_level_data(source.temperature)};
}

void
copy_coupled_vector (CoupledVector& destination,
                     CoupledVector const& source)
{
    copy_level_data(destination.energy, source.energy);
    copy_level_data(destination.temperature, source.temperature);
}

void
set_coupled_vector (CoupledVector& vector, Real value)
{
    set_level_data(vector.energy, value);
    set_level_data(vector.temperature, value);
}

void
scale_coupled_vector (CoupledVector& vector, Real scale)
{
    for (auto& field : vector.energy) {
        field->mult(scale, 0, 1, 0);
    }
    for (auto& field : vector.temperature) {
        field->mult(scale, 0, 1, 0);
    }
}

void
increment_coupled_vector (CoupledVector& destination,
                          CoupledVector const& source, Real scale)
{
    saxpy_level_data(destination.energy, scale, source.energy);
    saxpy_level_data(destination.temperature, scale, source.temperature);
}

void
lincomb_coupled_vector (CoupledVector& destination, Real a,
                        CoupledVector const& lhs, Real b,
                        CoupledVector const& rhs)
{
    lincomb_level_data(destination.energy, a, lhs.energy, b, rhs.energy);
    lincomb_level_data(destination.temperature, a, lhs.temperature, b,
                       rhs.temperature);
}

Real
coupled_dot (CoupledVector const& lhs, CoupledVector const& rhs,
             DiffusionHierarchy const& hierarchy,
             Vector<iMultiFab> const& masks)
{
    return composite_weighted_dot(lhs.energy, rhs.energy, hierarchy, masks) +
           composite_weighted_dot(lhs.temperature, rhs.temperature,
                                  hierarchy, masks);
}

Real
coupled_norm (CoupledVector const& vector,
              DiffusionHierarchy const& hierarchy,
              Vector<iMultiFab> const& masks)
{
    return std::sqrt(amrex::max(coupled_dot(vector, vector, hierarchy, masks),
                                Real(0)));
}

bool
coupled_positive_finite (CoupledVector const& vector,
                         Vector<iMultiFab> const& masks)
{
    auto const [minimum_energy, maximum_energy] =
        composite_minimum_maximum(vector.energy, masks);
    auto const [minimum_temperature, maximum_temperature] =
        composite_minimum_maximum(vector.temperature, masks);
    return composite_all_finite(vector.energy, masks) &&
           composite_all_finite(vector.temperature, masks) &&
           minimum_energy > Real(0) && minimum_temperature > Real(0) &&
           maximum_energy < Real(8) && maximum_temperature < Real(3);
}

void
fill_positive_newton_candidate (CoupledVector& candidate,
                                CoupledVector const& state,
                                CoupledVector const& correction,
                                Real step_length)
{
    for (int level = 0; level < static_cast<int>(state.energy.size()); ++level) {
        for (MFIter mfi(*candidate.energy[level]); mfi.isValid(); ++mfi) {
            auto const next_energy = candidate.energy[level]->array(mfi);
            auto const next_temperature =
                candidate.temperature[level]->array(mfi);
            auto const energy = state.energy[level]->const_array(mfi);
            auto const temperature =
                state.temperature[level]->const_array(mfi);
            auto const delta_energy =
                correction.energy[level]->const_array(mfi);
            auto const delta_temperature =
                correction.temperature[level]->const_array(mfi);
            ParallelFor(mfi.validbox(),
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                Real const energy_exponent = amrex::max(
                    Real(-50), amrex::min(
                        Real(50), step_length * delta_energy(i,j,k) /
                            amrex::max(energy(i,j,k), Real(1.e-30))));
                Real const temperature_exponent = amrex::max(
                    Real(-50), amrex::min(
                        Real(50), step_length * delta_temperature(i,j,k) /
                            amrex::max(temperature(i,j,k), Real(1.e-30))));
                next_energy(i,j,k) =
                    energy(i,j,k) * std::exp(energy_exponent);
                next_temperature(i,j,k) =
                    temperature(i,j,k) * std::exp(temperature_exponent);
            });
        }
    }
}

void
fill_coupled_residual (
    DiffusionHierarchy const& hierarchy, Vector<iMultiFab> const& masks,
    CoupledVector const& state, LevelData const& old_energy,
    LevelData const& old_temperature, LevelData& atomic_number, Real dt,
    CoupledVector& residual, LevelData& sigma,
    LevelData& radiation_diffusion, LevelData& material_diffusion,
    FaceData& radiation_bcoef, FaceData& material_bcoef)
{
    AMREX_ALWAYS_ASSERT(hierarchy.geom.size() == 1);
    CoupledVector work = clone_coupled_vector(state);
    update_absorption_and_cell_diffusion(
        hierarchy, work.energy, work.temperature, atomic_number, sigma,
        radiation_diffusion, material_diffusion);
    fill_paper_face_coefficients(
        hierarchy, work.energy, work.temperature, atomic_number,
        radiation_bcoef, material_bcoef);

    Geometry const& geometry = hierarchy.geom[0];
    Box const domain = geometry.Domain();
    auto const dlo = amrex::lbound(domain);
    auto const dhi = amrex::ubound(domain);
    auto const dx = geometry.CellSizeArray();
    Real const inverse_dt = Real(1) / dt;
    iMultiFab const& mask = masks[0];
    for (MFIter mfi(*residual.energy[0]); mfi.isValid(); ++mfi) {
        Box const& box = mfi.validbox();
        auto const e = work.energy[0]->const_array(mfi);
        auto const t = work.temperature[0]->const_array(mfi);
        auto const eo = old_energy[0]->const_array(mfi);
        auto const to = old_temperature[0]->const_array(mfi);
        auto const absorption = sigma[0]->const_array(mfi);
        auto const drx = radiation_bcoef[0][0]->const_array(mfi);
        auto const dry = radiation_bcoef[0][1]->const_array(mfi);
        auto const dtx = material_bcoef[0][0]->const_array(mfi);
        auto const dty = material_bcoef[0][1]->const_array(mfi);
        auto const re = residual.energy[0]->array(mfi);
        auto const rt = residual.temperature[0]->array(mfi);
        auto const active = mask.const_array(mfi);
        ParallelFor(box,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            if (!active(i, j, k)) {
                re(i, j, k) = Real(0);
                rt(i, j, k) = Real(0);
                return;
            }
            Real const temperature_value = positive_temperature(t(i,j,k));
            Real const t4 = temperature_value * cube(temperature_value);
            Real const exchange = absorption(i,j,k) * (e(i,j,k) - t4);
            Real radiation = inverse_dt * (e(i,j,k) - eo(i,j,k)) + exchange;
            Real material = inverse_dt * (t(i,j,k) - to(i,j,k)) - exchange;

            if (i > dlo.x) {
                radiation += drx(i,j,k) * (e(i,j,k) - e(i-1,j,k)) /
                             (dx[0] * dx[0]);
                material += dtx(i,j,k) * (t(i,j,k) - t(i-1,j,k)) /
                            (dx[0] * dx[0]);
            } else {
                Real const aa = Real(0.25);
                Real const bb = Real(1) / (Real(6) * absorption(i,j,k));
                radiation += drx(i,j,k) * (aa * e(i,j,k) - Real(1)) /
                             ((bb + aa * Real(0.5) * dx[0]) * dx[0]);
            }
            if (i < dhi.x) {
                radiation += drx(i+1,j,k) * (e(i,j,k) - e(i+1,j,k)) /
                             (dx[0] * dx[0]);
                material += dtx(i+1,j,k) * (t(i,j,k) - t(i+1,j,k)) /
                            (dx[0] * dx[0]);
            } else {
                Real const aa = Real(0.25);
                Real const bb = Real(1) / (Real(6) * absorption(i,j,k));
                radiation += drx(i+1,j,k) * aa * e(i,j,k) /
                             ((bb + aa * Real(0.5) * dx[0]) * dx[0]);
            }
            if (j > dlo.y) {
                radiation += dry(i,j,k) * (e(i,j,k) - e(i,j-1,k)) /
                             (dx[1] * dx[1]);
                material += dty(i,j,k) * (t(i,j,k) - t(i,j-1,k)) /
                            (dx[1] * dx[1]);
            }
            if (j < dhi.y) {
                radiation += dry(i,j+1,k) * (e(i,j,k) - e(i,j+1,k)) /
                             (dx[1] * dx[1]);
                material += dty(i,j+1,k) * (t(i,j,k) - t(i,j+1,k)) /
                            (dx[1] * dx[1]);
            }
            re(i,j,k) = radiation;
            rt(i,j,k) = material;
        });
    }
}

void
fill_newton_preconditioner_coefficients (
    CoupledVector const& state, LevelData const& sigma, Real dt,
    LevelData& energy_acoef, LevelData& temperature_acoef)
{
    Real const inverse_dt = Real(1) / dt;
    for (int level = 0; level < static_cast<int>(sigma.size()); ++level) {
        for (MFIter mfi(*sigma[level]); mfi.isValid(); ++mfi) {
            auto const e = state.energy[level]->const_array(mfi);
            auto const t = state.temperature[level]->const_array(mfi);
            auto const absorption = sigma[level]->const_array(mfi);
            auto const ea = energy_acoef[level]->array(mfi);
            auto const ta = temperature_acoef[level]->array(mfi);
            ParallelFor(mfi.validbox(),
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                Real const temperature_value = positive_temperature(t(i,j,k));
                Real const sigma_value = absorption(i,j,k);
                ea(i,j,k) = inverse_dt + sigma_value;
                ta(i,j,k) = inverse_dt + sigma_value *
                    (cube(temperature_value) +
                     Real(3) * e(i,j,k) / temperature_value);
            });
        }
    }
}

void
add_exchange_coupling (LevelData& temperature_rhs,
                       LevelData const& energy_correction,
                       LevelData const& sigma)
{
    for (int level = 0; level < static_cast<int>(sigma.size()); ++level) {
        for (MFIter mfi(*temperature_rhs[level]); mfi.isValid(); ++mfi) {
            auto const rhs = temperature_rhs[level]->array(mfi);
            auto const correction = energy_correction[level]->const_array(mfi);
            auto const absorption = sigma[level]->const_array(mfi);
            ParallelFor(mfi.validbox(),
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                rhs(i,j,k) += absorption(i,j,k) * correction(i,j,k);
            });
        }
    }
}

class CoupledNewtonProblem
{
  public:
    using State = CoupledVector;
    using RT = Real;

    CoupledNewtonProblem (
        DiffusionHierarchy const& hierarchy, Vector<iMultiFab> const& masks,
        LevelData const& old_energy, LevelData const& old_temperature,
        LevelData& atomic_number, LevelData& sigma, Real dt,
        MLABecLapAMG& radiation_solver, MLABecLapAMG& material_solver,
        SolverSummary& summary, LevelData& energy_acoef,
        LevelData& temperature_acoef, FaceData& radiation_bcoef,
        FaceData& material_bcoef, LevelData& robin_a, LevelData& robin_b,
        LevelData& robin_f,
        Array<LinOpBCType, AMREX_SPACEDIM> radiation_lo,
        Array<LinOpBCType, AMREX_SPACEDIM> radiation_hi,
        Array<LinOpBCType, AMREX_SPACEDIM> insulated)
        : m_hierarchy(hierarchy), m_masks(masks), m_old_energy(old_energy),
          m_old_temperature(old_temperature), m_atomic_number(atomic_number),
          m_sigma(sigma), m_dt(dt),
          m_radiation_solver(radiation_solver),
          m_material_solver(material_solver), m_summary(summary),
          m_energy_acoef(energy_acoef),
          m_temperature_acoef(temperature_acoef),
          m_radiation_bcoef(radiation_bcoef),
          m_material_bcoef(material_bcoef), m_robin_a(robin_a),
          m_robin_b(robin_b), m_robin_f(robin_f),
          m_radiation_lo(radiation_lo), m_radiation_hi(radiation_hi),
          m_insulated(insulated),
          m_trial_sigma(make_cell_data(hierarchy, 1, 1)),
          m_trial_radiation_diffusion(make_cell_data(hierarchy, 1, 1)),
          m_trial_material_diffusion(make_cell_data(hierarchy, 1, 1)),
          m_trial_radiation_bcoef(make_face_data(hierarchy)),
          m_trial_material_bcoef(make_face_data(hierarchy))
    {}

    void prepare (State const& state)
    {
        auto const [minimum_energy, maximum_energy] =
            composite_minimum_maximum(state.energy, m_masks);
        auto const [minimum_temperature, maximum_temperature] =
            composite_minimum_maximum(state.temperature, m_masks);
        amrex::ignore_unused(minimum_energy, minimum_temperature);
        m_energy_scale = amrex::max(maximum_energy, initial_radiation_energy);
        m_temperature_scale = amrex::max(
            maximum_temperature,
            std::sqrt(std::sqrt(initial_radiation_energy)));

        fill_newton_preconditioner_coefficients(
            state, m_sigma, m_dt, m_energy_acoef, m_temperature_acoef);
        fill_radiation_robin_data(
            m_hierarchy, m_sigma, m_robin_a, m_robin_b, m_robin_f);
        RobinBCData robin{get_level_const_ptrs(m_robin_a),
                          get_level_const_ptrs(m_robin_b),
                          get_level_const_ptrs(m_robin_f)};
        m_radiation_solver.setup(
            Real(1), Real(1), get_level_const_ptrs(m_energy_acoef),
            get_face_const_ptrs(m_radiation_bcoef), m_radiation_lo,
            m_radiation_hi, {}, robin);
        record_setup(m_summary, m_radiation_solver);
        m_material_solver.setup(
            Real(1), Real(1), get_level_const_ptrs(m_temperature_acoef),
            get_face_const_ptrs(m_material_bcoef), m_insulated, m_insulated,
            {});
        record_setup(m_summary, m_material_solver);
    }

    void residual (State const& state, State& output)
    {
        fill_coupled_residual(
            m_hierarchy, m_masks, state, m_old_energy, m_old_temperature,
            m_atomic_number, m_dt, output, m_sigma,
            m_trial_radiation_diffusion, m_trial_material_diffusion,
            m_radiation_bcoef, m_material_bcoef);
    }

    void linearized_residual (State const& state, State& output)
    {
        fill_coupled_residual(
            m_hierarchy, m_masks, state, m_old_energy, m_old_temperature,
            m_atomic_number, m_dt, output, m_trial_sigma,
            m_trial_radiation_diffusion, m_trial_material_diffusion,
            m_trial_radiation_bcoef, m_trial_material_bcoef);
    }

    CoupledVector makeVecRHS () const
    {
        return make_coupled_vector(m_hierarchy, 0);
    }

    CoupledVector makeVecLHS () const
    {
        return make_coupled_vector(m_hierarchy, 1);
    }

    void precondition (CoupledVector& output, CoupledVector const& rhs)
    {
        set_coupled_vector(output, Real(0));
        m_radiation_solver.precondition(
            get_level_ptrs(output.energy), get_level_const_ptrs(rhs.energy));
        LevelData temperature_rhs = clone_level_data(rhs.temperature);
        add_exchange_coupling(temperature_rhs, output.energy, m_sigma);
        m_material_solver.precondition(
            get_level_ptrs(output.temperature),
            get_level_const_ptrs(temperature_rhs));
    }

    void assign (CoupledVector& lhs, CoupledVector const& rhs) const
    {
        copy_coupled_vector(lhs, rhs);
    }

    Real dotProduct (CoupledVector const& lhs,
                     CoupledVector const& rhs) const
    {
        return composite_weighted_dot(lhs.energy, rhs.energy, m_hierarchy,
                                      m_masks) /
                   (m_energy_scale * m_energy_scale) +
               composite_weighted_dot(lhs.temperature, rhs.temperature,
                                      m_hierarchy, m_masks) /
                   (m_temperature_scale * m_temperature_scale);
    }

    void increment (CoupledVector& lhs, CoupledVector const& rhs,
                    Real scale) const
    {
        increment_coupled_vector(lhs, rhs, scale);
    }

    void linComb (CoupledVector& lhs, Real a, CoupledVector const& rhs_a,
                  Real b, CoupledVector const& rhs_b) const
    {
        lincomb_coupled_vector(lhs, a, rhs_a, b, rhs_b);
    }

    Real norm2 (CoupledVector const& vector) const
    {
        return std::sqrt(amrex::max(dotProduct(vector, vector), Real(0)));
    }

    void scale (CoupledVector& vector, Real factor) const
    {
        scale_coupled_vector(vector, factor);
    }

    void setToZero (CoupledVector& vector) const
    {
        set_coupled_vector(vector, Real(0));
    }

    CoupledVector clone (CoupledVector const& state) const
    {
        return clone_coupled_vector(state);
    }

    void fill_candidate (CoupledVector& candidate,
                         CoupledVector const& state,
                         CoupledVector const& correction,
                         Real step_length) const
    {
        fill_positive_newton_candidate(candidate, state, correction,
                                       step_length);
    }

    bool admissible (CoupledVector const& state) const
    {
        return coupled_positive_finite(state, m_masks);
    }

    Real relative_change (CoupledVector const& lhs,
                          CoupledVector const& rhs) const
    {
        return amrex::max(
            composite_maximum_relative_change(lhs.energy, rhs.energy, m_masks),
            composite_maximum_relative_change(lhs.temperature,
                                              rhs.temperature, m_masks));
    }

  private:
    DiffusionHierarchy const& m_hierarchy;
    Vector<iMultiFab> const& m_masks;
    LevelData const& m_old_energy;
    LevelData const& m_old_temperature;
    LevelData& m_atomic_number;
    LevelData& m_sigma;
    Real m_dt;
    MLABecLapAMG& m_radiation_solver;
    MLABecLapAMG& m_material_solver;
    SolverSummary& m_summary;
    LevelData& m_energy_acoef;
    LevelData& m_temperature_acoef;
    FaceData& m_radiation_bcoef;
    FaceData& m_material_bcoef;
    LevelData& m_robin_a;
    LevelData& m_robin_b;
    LevelData& m_robin_f;
    Array<LinOpBCType, AMREX_SPACEDIM> m_radiation_lo;
    Array<LinOpBCType, AMREX_SPACEDIM> m_radiation_hi;
    Array<LinOpBCType, AMREX_SPACEDIM> m_insulated;
    LevelData m_trial_sigma;
    LevelData m_trial_radiation_diffusion;
    LevelData m_trial_material_diffusion;
    FaceData m_trial_radiation_bcoef;
    FaceData m_trial_material_bcoef;
    Real m_energy_scale = Real(1);
    Real m_temperature_scale = Real(1);
};

Real
total_energy (LevelData const& energy, LevelData const& temperature,
              DiffusionHierarchy const& hierarchy,
              Vector<iMultiFab> const& masks)
{
    auto total = clone_level_data(energy);
    saxpy_level_data(total, Real(1), temperature);
    return composite_volume_sum(total, hierarchy, masks);
}

Real
radiation_boundary_input (DiffusionHierarchy const& hierarchy,
                          LevelData const& energy, LevelData const& sigma,
                          FaceData const& radiation_bcoef)
{
    ReduceOps<ReduceOpSum> reduce_op;
    ReduceData<Real> reduce_data(reduce_op);
    using Tuple = typename decltype(reduce_data)::Type;
    for (int level = 0; level < static_cast<int>(energy.size()); ++level) {
        Box const domain = hierarchy.geom[level].Domain();
        int const inlet = domain.smallEnd(0);
        int const outlet = domain.bigEnd(0);
        auto const dx = hierarchy.geom[level].CellSizeArray();
        Real const area = dx[1];
        Real const distance = Real(0.5) * dx[0];
        for (MFIter mfi(*energy[level]); mfi.isValid(); ++mfi) {
            auto const e = energy[level]->const_array(mfi);
            auto const absorption = sigma[level]->const_array(mfi);
            auto const face = radiation_bcoef[level][0]->const_array(mfi);
            reduce_op.eval(mfi.validbox(), reduce_data,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept -> Tuple
            {
                Real input = Real(0);
                Real const aa = Real(0.25);
                Real const bb = Real(1) /
                                (Real(6) * absorption(i,j,k));
                if (i == inlet) {
                    input += face(i,j,k) * area *
                             (Real(1) - aa * e(i,j,k)) /
                             (bb + aa * distance);
                }
                if (i == outlet) {
                    input += face(i + 1,j,k) * area *
                             (Real(0) - aa * e(i,j,k)) /
                             (bb + aa * distance);
                }
                return {input};
            });
        }
    }
    Real result = amrex::get<0>(reduce_data.value(reduce_op));
    ParallelDescriptor::ReduceRealSum(result);
    return result;
}

Long
count_high_z_cells (LevelData const& atomic_number,
                    Vector<iMultiFab> const& masks)
{
    ReduceOps<ReduceOpSum> reduce_op;
    ReduceData<Long> reduce_data(reduce_op);
    using Tuple = typename decltype(reduce_data)::Type;
    for (int level = 0; level < static_cast<int>(atomic_number.size());
         ++level) {
        for (MFIter mfi(*atomic_number[level]); mfi.isValid(); ++mfi) {
            auto const z = atomic_number[level]->const_array(mfi);
            auto const mask = masks[level].const_array(mfi);
            reduce_op.eval(mfi.validbox(), reduce_data,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept -> Tuple
            {
                return {mask(i,j,k) && z(i,j,k) > Real(1) ? Long(1)
                                                          : Long(0)};
            });
        }
    }
    Long result = amrex::get<0>(reduce_data.value(reduce_op));
    ParallelDescriptor::ReduceLongSum(result);
    return result;
}

void
write_icase_plotfile (std::string const& name,
                      DiffusionHierarchy const& hierarchy,
                      LevelData const& energy, LevelData const& temperature,
                      LevelData const& atomic_number, LevelData const& sigma,
                      LevelData const& radiation_diffusion,
                      LevelData const& material_diffusion, Real time)
{
    if (name.empty()) {
        return;
    }
    auto plot = make_cell_data(hierarchy, 6, 0);
    for (int level = 0; level < static_cast<int>(plot.size()); ++level) {
        MultiFab::Copy(*plot[level], *energy[level], 0, 0, 1, 0);
        MultiFab::Copy(*plot[level], *temperature[level], 0, 1, 1, 0);
        MultiFab::Copy(*plot[level], *atomic_number[level], 0, 2, 1, 0);
        MultiFab::Copy(*plot[level], *sigma[level], 0, 3, 1, 0);
        MultiFab::Copy(*plot[level], *radiation_diffusion[level], 0, 4, 1, 0);
        MultiFab::Copy(*plot[level], *material_diffusion[level], 0, 5, 1, 0);
    }
    Vector<std::string> const variables{
        "radiation_energy", "material_temperature", "atomic_number",
        "absorption", "radiation_diffusion", "material_diffusion"};
    WriteMultiLevelPlotfile(name, static_cast<int>(plot.size()),
                            get_level_const_ptrs(plot), variables,
                            hierarchy.geom, time,
                            Vector<int>(plot.size(), 0), hierarchy.ref_ratio);
    amrex::Print() << "Wrote ICASE 2001-12 plotfile " << name << '\n';
}

} // namespace

ICASE2001Result
run_icase_2001 (int n_cell, int time_steps, Real dt, bool iteration_output,
                std::string const& plotfile_name)
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        n_cell > 0 && n_cell % 3 == 0,
        "icase_n_cell must be positive and divisible by 3 so the material "
        "interfaces align with cell faces");
    AMREX_ALWAYS_ASSERT(time_steps > 0 && dt > Real(0));

    Array<int, AMREX_SPACEDIM> const nonperiodic{
        AMREX_D_DECL(0, 0, 0)};
    DiffusionHierarchy hierarchy =
        make_uniform_hierarchy(n_cell, 32, nonperiodic);
    auto masks = make_composite_masks(hierarchy);
    CoupledVector state = make_coupled_vector(hierarchy, 1);
    auto old_energy = make_cell_data(hierarchy, 1, 1);
    auto old_temperature = make_cell_data(hierarchy, 1, 1);
    CoupledVector predictor_solution = make_coupled_vector(hierarchy, 1);
    auto atomic_number = make_cell_data(hierarchy, 1, 1);
    auto sigma = make_cell_data(hierarchy, 1, 1);
    auto radiation_diffusion = make_cell_data(hierarchy, 1, 1);
    auto material_diffusion = make_cell_data(hierarchy, 1, 1);
    auto energy_acoef = make_cell_data(hierarchy, 1, 0);
    auto temperature_acoef = make_cell_data(hierarchy, 1, 0);
    auto energy_rhs = make_cell_data(hierarchy, 1, 0);
    auto temperature_rhs = make_cell_data(hierarchy, 1, 0);
    auto radiation_bcoef = make_face_data(hierarchy);
    auto material_bcoef = make_face_data(hierarchy);
    auto robin_a = make_cell_data(hierarchy, 1, 0);
    auto robin_b = make_cell_data(hierarchy, 1, 0);
    auto robin_f = make_cell_data(hierarchy, 1, 0);

    initialize_fields(hierarchy, state.energy, state.temperature,
                      atomic_number, sigma);
    ICASE2001Result result;
    result.cells = composite_cell_count(masks);
    result.high_z_cells = count_high_z_cells(atomic_number, masks);
    result.initial_total_energy =
        total_energy(state.energy, state.temperature, hierarchy, masks);

    Array<LinOpBCType, AMREX_SPACEDIM> const radiation_lo{
        AMREX_D_DECL(LinOpBCType::Robin, LinOpBCType::Neumann,
                     LinOpBCType::Neumann)};
    Array<LinOpBCType, AMREX_SPACEDIM> const radiation_hi = radiation_lo;
    Array<LinOpBCType, AMREX_SPACEDIM> const insulated{
        AMREX_D_DECL(LinOpBCType::Neumann, LinOpBCType::Neumann,
                     LinOpBCType::Neumann)};

    MLABecLapAMG radiation_solver(hierarchy.geom, hierarchy.grids,
                                  hierarchy.dmap);
    MLABecLapAMG material_solver(hierarchy.geom, hierarchy.grids,
                                 hierarchy.dmap);
    Real const nonlinear_tolerance =
        sizeof(Real) == sizeof(float) ? Real(3.e-4) : Real(2.e-7);
    int constexpr maximum_nonlinear_iterations = 20;

    for (int step = 0; step < time_steps; ++step) {
        copy_level_data(old_energy, state.energy);
        copy_level_data(old_temperature, state.temperature);
        Real const old_total =
            total_energy(old_energy, old_temperature, hierarchy, masks);
        for (int predictor = 0; predictor < 20; ++predictor) {
            update_absorption_and_cell_diffusion(
                hierarchy, state.energy, state.temperature, atomic_number,
                sigma, radiation_diffusion, material_diffusion);
            fill_paper_face_coefficients(
                hierarchy, state.energy, state.temperature, atomic_number,
                radiation_bcoef, material_bcoef);
            fill_radiation_robin_data(hierarchy, sigma, robin_a, robin_b,
                                      robin_f);
            fill_linearized_systems(
                old_energy, old_temperature, state.energy, state.temperature,
                sigma, dt, energy_acoef, energy_rhs, temperature_acoef,
                temperature_rhs);
            RobinBCData predictor_robin{
                get_level_const_ptrs(robin_a), get_level_const_ptrs(robin_b),
                get_level_const_ptrs(robin_f)};
            radiation_solver.setup(
                Real(1), Real(1), get_level_const_ptrs(energy_acoef),
                get_face_const_ptrs(radiation_bcoef), radiation_lo,
                radiation_hi, {}, predictor_robin);
            record_setup(result.solver, radiation_solver);
            copy_level_data(predictor_solution.energy, state.energy);
            auto const predictor_energy_info = radiation_solver.solve(
                get_level_ptrs(predictor_solution.energy),
                get_level_const_ptrs(energy_rhs), linear_tolerance(), Real(0));
            record_solve(result.solver, predictor_energy_info);
            fill_linearized_systems(
                old_energy, old_temperature, predictor_solution.energy,
                state.temperature, sigma, dt, energy_acoef, energy_rhs,
                temperature_acoef, temperature_rhs);
            material_solver.setup(
                Real(1), Real(1), get_level_const_ptrs(temperature_acoef),
                get_face_const_ptrs(material_bcoef), insulated, insulated,
                {});
            record_setup(result.solver, material_solver);
            copy_level_data(predictor_solution.temperature,
                            state.temperature);
            auto const predictor_temperature_info = material_solver.solve(
                get_level_ptrs(predictor_solution.temperature),
                get_level_const_ptrs(temperature_rhs), linear_tolerance(),
                Real(0));
            record_solve(result.solver, predictor_temperature_info);
            Real const predictor_change = amrex::max(
                composite_maximum_relative_change(
                    predictor_solution.energy, state.energy, masks),
                composite_maximum_relative_change(
                    predictor_solution.temperature, state.temperature,
                    masks));
            lincomb_level_data(state.energy, Real(0.3), state.energy,
                               Real(0.7), predictor_solution.energy);
            lincomb_level_data(state.temperature, Real(0.3),
                               state.temperature, Real(0.7),
                               predictor_solution.temperature);
            if (predictor_change < Real(1.e-3)) {
                break;
            }
        }
        CoupledNewtonProblem problem(
            hierarchy, masks, old_energy, old_temperature, atomic_number,
            sigma, dt, radiation_solver, material_solver, result.solver,
            energy_acoef, temperature_acoef, radiation_bcoef, material_bcoef,
            robin_a, robin_b, robin_f, radiation_lo, radiation_hi,
            insulated);
        NewtonKrylovOptions options;
        options.nonlinear_tolerance = nonlinear_tolerance;
        options.maximum_nonlinear_iterations =
            maximum_nonlinear_iterations;
        options.problem_name = "The ICASE 2001-12 coupled system";
        NewtonKrylovSolver<CoupledNewtonProblem> newton(problem, options);
        auto const solve = newton.solve(state);
        int const step_iterations = solve.nonlinear_iterations;
        result.final_nonlinear_change = solve.final_change;
        result.total_nonlinear_iterations += solve.nonlinear_iterations;
        result.total_newton_krylov_iterations +=
            solve.total_linear_iterations;
        result.maximum_newton_krylov_iterations = amrex::max(
            result.maximum_newton_krylov_iterations,
            solve.maximum_linear_iterations);
        result.maximum_coupled_residual = amrex::max(
            result.maximum_coupled_residual, solve.final_residual_norm);
        result.maximum_nonlinear_iterations = amrex::max(
            result.maximum_nonlinear_iterations, step_iterations);
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            solve.converged,
            "The ICASE 2001-12 coupled nonlinear iteration did not converge");

        // The exchange source is equal and opposite in the E and T
        // equations.  Therefore the change in integral(E+T) must equal the
        // net radiation supplied through the two Marshak boundaries.
        update_absorption_and_cell_diffusion(
            hierarchy, state.energy, state.temperature, atomic_number, sigma,
            radiation_diffusion, material_diffusion);
        fill_paper_face_coefficients(
            hierarchy, state.energy, state.temperature, atomic_number,
            radiation_bcoef, material_bcoef);
        Real const new_total =
            total_energy(state.energy, state.temperature, hierarchy, masks);
        Real const expected_change = dt * radiation_boundary_input(
            hierarchy, state.energy, sigma, radiation_bcoef);
        Real const energy_change = new_total - old_total;
        Real const balance_error =
            std::abs(energy_change - expected_change) /
            amrex::max(amrex::max(std::abs(energy_change),
                                  std::abs(expected_change)),
                       Real(1.e-30));
        result.maximum_energy_balance_error = amrex::max(
            result.maximum_energy_balance_error, balance_error);
        if (iteration_output) {
            amrex::Print()
                << "ICASE 2001-12 step=" << step + 1 << "/" << time_steps
                << ", time=" << Real(step + 1) * dt
                << ", nonlinear iterations=" << step_iterations
                << ", change=" << result.final_nonlinear_change
                << ", energy-balance error=" << balance_error << '\n';
        }
    }

    update_absorption_and_cell_diffusion(
        hierarchy, state.energy, state.temperature, atomic_number, sigma,
        radiation_diffusion, material_diffusion);
    result.time_steps = time_steps;
    result.final_time = Real(time_steps) * dt;
    result.final_total_energy =
        total_energy(state.energy, state.temperature, hierarchy, masks);
    auto const [minimum_energy, maximum_energy] =
        composite_minimum_maximum(state.energy, masks);
    auto const [minimum_temperature, maximum_temperature] =
        composite_minimum_maximum(state.temperature, masks);
    result.minimum_radiation_energy = minimum_energy;
    result.maximum_radiation_energy = maximum_energy;
    result.minimum_material_temperature = minimum_temperature;
    result.maximum_material_temperature = maximum_temperature;

    write_icase_plotfile(plotfile_name, hierarchy, state.energy,
                         state.temperature,
                         atomic_number, sigma, radiation_diffusion,
                         material_diffusion, result.final_time);

    AMREX_ALWAYS_ASSERT(result.high_z_cells ==
                        Long(n_cell / 3) * Long(n_cell / 3));
    AMREX_ALWAYS_ASSERT(composite_all_finite(state.energy, masks));
    AMREX_ALWAYS_ASSERT(composite_all_finite(state.temperature, masks));
    AMREX_ALWAYS_ASSERT(result.minimum_radiation_energy > Real(0));
    AMREX_ALWAYS_ASSERT(result.minimum_material_temperature > Real(0));
    AMREX_ALWAYS_ASSERT(result.final_total_energy >
                        result.initial_total_energy);
    Real const balance_tolerance =
        sizeof(Real) == sizeof(float) ? Real(2.e-2) : Real(2.e-4);
    AMREX_ALWAYS_ASSERT(result.maximum_energy_balance_error <
                        balance_tolerance);
    return result;
}

} // namespace fld_test
