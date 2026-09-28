#!/bin/bash
#SBATCH --account=ast236
#SBATCH --job-name=fld-gpu-ws-064
#SBATCH --output=%x-%j.out
#SBATCH --time=00:05:00
#SBATCH --partition=batch
#SBATCH --nodes=64
#SBATCH --ntasks=512
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --threads-per-core=1
#SBATCH --gpus-per-node=8

set -euo pipefail

# Submit from the repository root or scaling/. Inherit the build's modules.
submit_dir="${SLURM_SUBMIT_DIR:-$PWD}"
repo_dir="${submit_dir}"
if [[ ! -f "${repo_dir}/GNUmakefile" ]]; then
    repo_dir="${submit_dir}/.."
fi
executable="${FLD_GPU_EXECUTABLE:-${repo_dir}/main3d.hip.${CRAY_CPU_TARGET:-x86-trento}.TPROF.MPI.HIP.ex}"

if [[ ! -x "${executable}" ]]; then
    echo "error: executable not found: ${executable}" >&2
    exit 1
fi

# Slurm supplies each rank's GPU binding; discard inherited visibility masks.
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS=1
export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_OFI_CXI_COUNTER_REPORT=2

srun --nodes="${SLURM_NNODES}" \
    --ntasks="${SLURM_NTASKS}" \
    --ntasks-per-node=8 \
    --cpus-per-task=7 \
    --cpu-bind=cores \
    --threads-per-core=1 \
    --distribution=block:cyclic \
    --gpus-per-task=1 \
    --gpu-bind=closest \
    "${executable}" \
    amrex.use_gpu_aware_mpi=1 \
    cloud_case=amr \
    cloud_fine_n=1024 \
    cloud_opacity_contrast=1e5 \
    cloud_flux_limiter=0 \
    cloud_predictor_steps=0 \
    cloud_eisenstat_walker=1 \
    cloud_write_plotfile=0 \
    cloud_iteration_output=1 \
    tiny_profiler.output_file=stdout \
    tiny_profiler.print_threshold=0.1 \
    mlabeclap_amg.measure_setup_messages=1 \
    mlabeclap_amg.measure_residual_messages=1 \
    mlabeclap_amg.measure_assembly_messages=1 \
    mlabeclap_amg.max_interp_elements=4 \
    mlabeclap_amg.true_residual_factor=10
