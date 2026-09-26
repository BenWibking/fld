#include "FLDTest.H"

#include <AMReX.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>

#include <cmath>
#include <string>

using namespace amrex;
using namespace fld_test;

int
main (int argc, char* argv[])
{
    static_assert(AMREX_SPACEDIM == 3);
    amrex::Initialize(argc, argv);
    {
        int fine_n = 64;
        int iteration_output = 0;
        int flux_limiter = 1;
        int write_plotfile = 0;
        int icase_only = 0;
        int icase_n_cell = 24;
        int icase_steps = 1000;
        int icase_iteration_output = 1;
        int icase_write_plotfile = 1;
        Real icase_dt = Real(0.01);
        std::string icase_plotfile = "plt_icase_2001_3d";
        std::string cloud_case = "amr";
        std::string plotfile_prefix = "plt_cloud3d";
        ParmParse pp;
        pp.query("cloud_fine_n", fine_n);
        pp.query("cloud_iteration_output", iteration_output);
        pp.query("cloud_flux_limiter", flux_limiter);
        pp.query("cloud_write_plotfile", write_plotfile);
        pp.query("cloud_case", cloud_case);
        pp.query("cloud_plotfile_prefix", plotfile_prefix);
        pp.query("icase_only", icase_only);
        pp.query("icase_n_cell", icase_n_cell);
        pp.query("icase_steps", icase_steps);
        pp.query("icase_dt", icase_dt);
        pp.query("icase_iteration_output", icase_iteration_output);
        pp.query("icase_write_plotfile", icase_write_plotfile);
        pp.query("icase_plotfile", icase_plotfile);

        if (icase_only != 0) {
            double const start = amrex::second();
            ICASE2001Result const result = run_icase_2001(
                icase_n_cell, icase_steps, icase_dt,
                icase_iteration_output != 0,
                icase_write_plotfile != 0 ? icase_plotfile : std::string());
            double wall_seconds = amrex::second() - start;
            ParallelDescriptor::ReduceRealMax(wall_seconds);
            amrex::Print()
                << "ICASE 2001-12 3-D rectangular prism: cells/high-Z cells="
                << result.cells << "/" << result.high_z_cells
                << ", steps/final time=" << result.time_steps << "/"
                << result.final_time
                << ", total energy initial/final="
                << result.initial_total_energy << "/"
                << result.final_total_energy << ", E range=["
                << result.minimum_radiation_energy << ","
                << result.maximum_radiation_energy << "]"
                << ", T range=[" << result.minimum_material_temperature
                << "," << result.maximum_material_temperature << "]"
                << ", nonlinear iterations total/max/change="
                << result.total_nonlinear_iterations << "/"
                << result.maximum_nonlinear_iterations << "/"
                << result.final_nonlinear_change
                << ", max coupled residual="
                << result.maximum_coupled_residual
                << ", max step energy-balance error="
                << result.maximum_energy_balance_error
                << ", Newton-Krylov iterations total/max="
                << result.total_newton_krylov_iterations << "/"
                << result.maximum_newton_krylov_iterations
                << ", whole-case wall=" << wall_seconds << " s"
                << ", preconditioner=" << result.solver.preconditioner
                << ", AMG backend=" << result.solver.amg_backend
                << ", linear iterations=" << result.solver.total_iterations
                << ", aggregate setup=" << result.solver.setup_seconds
                << " s, aggregate solve=" << result.solver.solve_seconds
                << " s\n";
            amrex::Finalize();
            return 0;
        }

        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            cloud_case == "amr" || cloud_case == "uniform" ||
                cloud_case == "both",
            "cloud_case must be amr, uniform, or both");

        auto run_case = [&] (bool use_amr) {
            std::string plotfile;
            if (write_plotfile != 0) {
                plotfile = plotfile_prefix +
                           (use_amr ? "_amr" : "_uniform");
            }
            double const start = amrex::second();
            CloudResult const result = run_cloud(
                use_amr, fine_n, flux_limiter != 0,
                iteration_output != 0,
                plotfile);
            double wall_seconds = amrex::second() - start;
            ParallelDescriptor::ReduceRealMax(wall_seconds);
            amrex::Print()
                << "FLD 3-D Penrose cloud: case="
                << (use_amr ? "amr" : "uniform")
                << ", fine_n=" << fine_n
                << ", cells=" << result.cells
                << ", limiter=" << (flux_limiter != 0 ? "on" : "off")
                << ", mixed cells=" << result.mixed_cells
                << ", cloudy volume relative error="
                << result.cloudy_volume_relative_error
                << ", transmission=" << result.transmission
                << ", balance error=" << result.balance_error
                << ", Newton/Krylov iterations="
                << result.nonlinear_iterations << "/"
                << result.total_newton_krylov_iterations
                << ", nonlinear residual="
                << result.final_nonlinear_residual
                << ", E range=[" << result.minimum_energy << ","
                << result.maximum_energy << "]"
                << ", whole-case wall=" << wall_seconds << " s"
                << ", preconditioner=" << result.solver.preconditioner
                << ", AMG backend=" << result.solver.amg_backend
                << ", linear iterations="
                << result.solver.total_iterations
                << ", max true relative residual="
                << result.solver.maximum_relative_residual
                << ", aggregate setup=" << result.solver.setup_seconds
                << " s, aggregate solve=" << result.solver.solve_seconds
                << " s\n";
            return result;
        };

        if (cloud_case == "uniform") {
            run_case(false);
        } else if (cloud_case == "amr") {
            run_case(true);
        } else {
            CloudResult const uniform = run_case(false);
            CloudResult const amr = run_case(true);
            amrex::Print() << "FLD 3-D Penrose cloud AMR/uniform "
                              "transmission difference="
                           << std::abs(amr.transmission -
                                       uniform.transmission) /
                                  uniform.transmission << '\n';
        }
    }
    amrex::Finalize();
    return 0;
}
