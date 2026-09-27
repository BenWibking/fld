#!/bin/bash
#SBATCH --account=ast236
#SBATCH --job-name=fld-ws-008
#SBATCH --output=%x-%j.out
#SBATCH --time=00:30:00
#SBATCH --partition=batch
#SBATCH --nodes=8
#SBATCH --ntasks=448
#SBATCH --ntasks-per-node=56
#SBATCH --cpus-per-task=1
#SBATCH --threads-per-core=1

set -euo pipefail

executable="../main3d.gnu.x86-trento.TPROF.MPI.ex"

if [[ ! -x "${executable}" ]]; then
    echo "error: executable not found: ${executable}" >&2
    exit 1
fi

export OMP_NUM_THREADS=1

srun --nodes="${SLURM_NNODES}" \
    --ntasks="${SLURM_NTASKS}" \
    --ntasks-per-node=56 \
    --cpus-per-task=1 \
    --cpu-bind=threads \
    --threads-per-core=1 \
    --distribution=block:cyclic \
    "${executable}" \
    cloud_case=amr \
    cloud_fine_n=512 \
    cloud_opacity_contrast=1e5 \
    cloud_write_plotfile=0 \
    cloud_iteration_output=1 \
    tiny_profiler.output_file=stdout \
    tiny_profiler.print_threshold=0.1 \
    mlabeclap_amg.measure_setup_messages=1 \
    mlabeclap_amg.true_residual_factor=6
