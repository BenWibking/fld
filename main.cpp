#include "FLDTest.H"

#include <AMReX.H>
#include <AMReX_ParmParse.H>

#include <cmath>
#include <string>

using namespace amrex;
using namespace fld_test;

namespace
{

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
