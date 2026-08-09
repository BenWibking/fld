#include "FLDTest.H"

#include <AMReX.H>
#include <AMReX_ParmParse.H>

#include <cmath>
#include <string>
#include <utility>

using namespace amrex;
using namespace fld_test;

namespace
{

void
print_solver_summary (SolverSummary const& solver)
{
    amrex::Print() << "outer solver=" << solver.outer_solver
                   << ", preconditioner=" << solver.preconditioner
                   << ", AMG backend=" << solver.amg_backend
                   << ", solves=" << solver.solves
                   << ", linear iterations total/avg/max="
                   << solver.total_iterations << "/"
                   << solver.average_iterations() << "/"
                   << solver.maximum_iterations
                   << ", preconditioner applications="
                   << solver.total_preconditioner_applications
                   << ", max true relative residual="
                   << solver.maximum_relative_residual
                   << ", max native AMG levels=" << solver.maximum_levels
                   << ", max operator complexity="
                   << solver.maximum_operator_complexity
                   << ", aggregate setup=" << solver.setup_seconds << " s"
                   << ", aggregate solve=" << solver.solve_seconds << " s"
                   << " (preconditioner=" << solver.preconditioner_seconds
                   << " s)";
}
} // namespace

int
main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    {
        static_assert(AMREX_SPACEDIM == 2);

        run_mlabeclap_amg_checks();

        int cloud_fine_n = 128;
        int cloud_iteration_output = 1;
        int cloud_flux_limiter = 1;
        int cloud_only = 0;
        int front_only = 0;
        int icase_only = 0;
        int icase_n_cell = 87;
        int icase_steps = 1000;
        int icase_iteration_output = 1;
        int icase_write_plotfile = 1;
        Real icase_dt = Real(0.01);
        std::string icase_plotfile = "plt_icase_2001";
        int solver_checks_only = 0;
        std::string cloud_case = "both";
        std::string cloud_plotfile_prefix = "plt";
        {
            ParmParse pp;
            pp.query("cloud_fine_n", cloud_fine_n);
            pp.query("cloud_iteration_output", cloud_iteration_output);
            pp.query("cloud_flux_limiter", cloud_flux_limiter);
            pp.query("cloud_only", cloud_only);
            pp.query("front_only", front_only);
            pp.query("icase_only", icase_only);
            pp.query("icase_n_cell", icase_n_cell);
            pp.query("icase_steps", icase_steps);
            pp.query("icase_dt", icase_dt);
            pp.query("icase_iteration_output", icase_iteration_output);
            pp.query("icase_write_plotfile", icase_write_plotfile);
            pp.query("icase_plotfile", icase_plotfile);
            pp.query("solver_checks_only", solver_checks_only);
            pp.query("cloud_case", cloud_case);
            pp.query("cloud_plotfile_prefix", cloud_plotfile_prefix);
        }

        if (solver_checks_only != 0) {
            amrex::Finalize();
            return 0;
        }

        if (icase_only != 0) {
            auto const icase = run_icase_2001(
                icase_n_cell, icase_steps, icase_dt,
                icase_iteration_output != 0,
                icase_write_plotfile != 0 ? icase_plotfile : std::string());
            amrex::Print()
                << "ICASE 2001-12 nonequilibrium radiation diffusion: "
                << "cells/high-z cells=" << icase.cells << "/"
                << icase.high_z_cells << ", steps/final time="
                << icase.time_steps << "/" << icase.final_time
                << ", total energy initial/final="
                << icase.initial_total_energy << "/"
                << icase.final_total_energy << ", E range=["
                << icase.minimum_radiation_energy << ","
                << icase.maximum_radiation_energy << "]"
                << ", T range=[" << icase.minimum_material_temperature
                << "," << icase.maximum_material_temperature << "]"
                << ", nonlinear iterations total/max/change="
                << icase.total_nonlinear_iterations << "/"
                << icase.maximum_nonlinear_iterations << "/"
                << icase.final_nonlinear_change
                << ", max coupled residual="
                << icase.maximum_coupled_residual
                << ", max step energy-balance error="
                << icase.maximum_energy_balance_error
                << ", Newton-Krylov iterations total/max="
                << icase.total_newton_krylov_iterations << "/"
                << icase.maximum_newton_krylov_iterations << ", ";
            print_solver_summary(icase.solver);
            amrex::Print() << '\n';
            amrex::Finalize();
            return 0;
        }

        if (front_only != 0) {
            using Configuration =
                std::pair<MLABecPreconditioner, MLABecAMGBackend>;
            Vector<Configuration> configurations{
                {MLABecPreconditioner::AMG, MLABecAMGBackend::Native},
                {MLABecPreconditioner::MLMG, MLABecAMGBackend::Native}};
#ifdef AMREX_USE_HYPRE
            configurations.emplace_back(MLABecPreconditioner::AMG,
                                        MLABecAMGBackend::BoomerAMG);
#endif
            for (auto const& [preconditioner, amg_backend] : configurations) {
                auto const front =
                    run_limited_front(preconditioner, amg_backend);
                amrex::Print()
                    << "FLD limited-front preconditioner comparison: cells="
                    << front.cells << ", front/causal radius="
                    << front.front_radius << "/" << front.causal_radius
                    << ", far excess=" << front.far_excess
                    << ", unlimited far excess="
                    << front.unlimited_far_excess
                    << ", max |F|/(cE)=" << front.maximum_flux_fraction
                    << ", E range=[" << front.minimum_energy << ","
                    << front.maximum_energy << "]"
                    << ", nonlinear iterations total/max="
                    << front.total_picard_iterations << "/"
                    << front.maximum_picard_iterations
                    << ", final nonlinear fixed-point residual="
                    << front.final_picard_change << ", ";
                print_solver_summary(front.solver);
                amrex::Print() << '\n';
            }
#ifndef AMREX_USE_HYPRE
            amrex::Print()
                << "FLD limited-front preconditioner comparison: "
                << "outer solver=gmres, preconditioner=amg, "
                << "AMG backend=boomeramg, unavailable "
                << "(rebuild with USE_HYPRE=TRUE)\n";
#endif
            amrex::Finalize();
            return 0;
        }

        if (cloud_only != 0) {
            if (cloud_case != "both" && cloud_case != "uniform" &&
                cloud_case != "amr") {
                amrex::Abort(
                    "cloud_case must be one of: both, uniform, or amr");
            }

            auto const run_selected_cloud = [&] (bool use_amr) {
                return run_cloud(
                    use_amr, cloud_fine_n, cloud_flux_limiter != 0,
                    cloud_iteration_output != 0,
                    cloud_plotfile_prefix.empty()
                        ? std::string()
                        : cloud_plotfile_prefix +
                              (use_amr ? "_amr" : "_uniform"));
            };

            if (cloud_case == "uniform") {
                auto const cloud_uniform = run_selected_cloud(false);
                amrex::Print()
                    << "FLD cloud Newton-Krylov benchmark: fine_n="
                    << cloud_fine_n
                    << ", limiter="
                    << (cloud_flux_limiter != 0 ? "on" : "off")
                    << ", uniform transmission/Newton/Krylov iterations="
                    << cloud_uniform.transmission << "/"
                    << cloud_uniform.nonlinear_iterations << "/"
                    << cloud_uniform.total_newton_krylov_iterations
                    << std::endl;
                amrex::Finalize();
                return 0;
            }

            if (cloud_case == "amr") {
                auto const cloud_amr = run_selected_cloud(true);
                amrex::Print()
                    << "FLD cloud Newton-Krylov benchmark: fine_n="
                    << cloud_fine_n
                    << ", limiter="
                    << (cloud_flux_limiter != 0 ? "on" : "off")
                    << ", AMR transmission/Newton/Krylov iterations="
                    << cloud_amr.transmission << "/"
                    << cloud_amr.nonlinear_iterations << "/"
                    << cloud_amr.total_newton_krylov_iterations
                    << std::endl;
                amrex::Finalize();
                return 0;
            }

            auto const cloud_uniform =
                run_selected_cloud(false);
            auto const cloud_amr = run_selected_cloud(true);
            amrex::Print()
                << "FLD cloud Newton-Krylov benchmark: fine_n="
                << cloud_fine_n
                << ", limiter="
                << (cloud_flux_limiter != 0 ? "on" : "off")
                << ", uniform transmission/Newton/Krylov iterations="
                << cloud_uniform.transmission << "/"
                << cloud_uniform.nonlinear_iterations << "/"
                << cloud_uniform.total_newton_krylov_iterations
                << ", AMR transmission/Newton/Krylov iterations="
                << cloud_amr.transmission << "/"
                << cloud_amr.nonlinear_iterations << "/"
                << cloud_amr.total_newton_krylov_iterations << std::endl;
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
            run_cloud(false, cloud_fine_n, true,
                      cloud_iteration_output != 0, std::string());
        amrex::Print() << "FLD cloud uniform: cells=" << cloud_uniform.cells
                       << ", transmission=" << cloud_uniform.transmission
                       << ", balance error=" << cloud_uniform.balance_error
                       << ", mixed cells/cloud area error="
                       << cloud_uniform.mixed_cells << "/"
                       << cloud_uniform.cloudy_area_relative_error
                       << ", E range=[" << cloud_uniform.minimum_energy << ","
                       << cloud_uniform.maximum_energy << "]"
                       << ", Newton iterations/change/residual="
                       << cloud_uniform.nonlinear_iterations << "/"
                       << cloud_uniform.final_nonlinear_change << "/"
                       << cloud_uniform.final_nonlinear_residual
                       << ", Newton-Krylov iterations total/max="
                       << cloud_uniform.total_newton_krylov_iterations << "/"
                       << cloud_uniform.maximum_newton_krylov_iterations
                       << ", ";
        print_solver_summary(cloud_uniform.solver);
        amrex::Print() << '\n';

        auto const cloud_amr =
            run_cloud(true, cloud_fine_n, true,
                      cloud_iteration_output != 0, std::string());
        amrex::Print() << "FLD cloud AMR: cells=" << cloud_amr.cells
                       << ", transmission=" << cloud_amr.transmission
                       << ", balance error=" << cloud_amr.balance_error
                       << ", mixed cells/cloud area error="
                       << cloud_amr.mixed_cells << "/"
                       << cloud_amr.cloudy_area_relative_error
                       << ", E range=[" << cloud_amr.minimum_energy << ","
                       << cloud_amr.maximum_energy << "]"
                       << ", Newton iterations/change/residual="
                       << cloud_amr.nonlinear_iterations << "/"
                       << cloud_amr.final_nonlinear_change << "/"
                       << cloud_amr.final_nonlinear_residual
                       << ", Newton-Krylov iterations total/max="
                       << cloud_amr.total_newton_krylov_iterations << "/"
                       << cloud_amr.maximum_newton_krylov_iterations << ", ";
        print_solver_summary(cloud_amr.solver);
        amrex::Print() << '\n';

        Real const transmission_difference =
            std::abs(cloud_amr.transmission - cloud_uniform.transmission) /
            cloud_uniform.transmission;
        amrex::Print() << "FLD cloud AMR/fine transmission difference="
                       << transmission_difference << '\n';
        AMREX_ALWAYS_ASSERT(transmission_difference < Real(0.12));

        // auto const front = run_limited_front();
        // amrex::Print() << "FLD limited front: cells=" << front.cells
        //                << ", front/causal radius=" << front.front_radius << "/"
        //                << front.causal_radius
        //                << ", far excess=" << front.far_excess
        //                << ", unlimited far excess="
        //                << front.unlimited_far_excess
        //                << ", max |F|/(cE)=" << front.maximum_flux_fraction
        //                << ", E range=[" << front.minimum_energy << ","
        //                << front.maximum_energy << "]"
        //                << ", Picard total/max/change="
        //                << front.total_picard_iterations << "/"
        //                << front.maximum_picard_iterations << "/"
        //                << front.final_picard_change << ", ";
        // print_solver_summary(front.solver);
        // amrex::Print() << '\n';

        amrex::Print() << "2-D scattering-only FLD Gaussian and cloud-layer "
                       << "GMRES+multigrid tests passed\n";
    }
    amrex::Finalize();
}
