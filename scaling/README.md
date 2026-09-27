# AMR-cloud weak scaling on Frontier

These jobs hold the AMR-cloud work per node constant while scaling the cubic
mesh by a factor of two in each direction. Each case uses 56 MPI ranks per
node, matching Frontier's default 56 allocatable CPU cores.

| Script | Nodes | MPI ranks | `cloud_fine_n` | Composite AMR cells |
| --- | ---: | ---: | ---: | ---: |
| `weak_001.slurm` | 1 | 56 | 256 | 4,390,912 |
| `weak_008.slurm` | 8 | 448 | 512 | 35,127,296 |
| `weak_064.slurm` | 64 | 3,584 | 1,024 | 281,018,368 |
| `weak_512.slurm` | 512 | 28,672 | 2,048 | 2,248,146,944 |

The cell count grows by exactly 8x with every 8x increase in nodes, retaining
4,390,912 composite cells per node. Plotfiles are disabled, and iteration
output is enabled for every run.
All four jobs set `cloud_opacity_contrast=1e5`: clear-cell extinction stays
at `0.001`, and pure-cloud extinction is `100`. The default for other cloud
runs remains a contrast of `1e6`.
All four jobs set `cloud_predictor_steps=13`, the lowest median whole-case
time in the local eight-rank `cloud_fine_n=128` sweep. The best count at the
Frontier scaling resolutions has not yet been measured.
All four jobs enable `cloud_eisenstat_walker=1` for the cloud Newton solve.
This uses Eisenstat--Walker Choice 2 for the GMRES relative tolerance, with
initial value 0.5, gamma 0.9, exponent 2, upper cap 0.9, the published 0.1
safeguard threshold, and a `1e-4` lower bound to avoid requesting accuracy
below the previous fixed tolerance from finite-difference Jacobian products.
Omitting the option retains the fixed `1e-4` tolerance. Local two-rank
`fine_n=64` and eight-rank `fine_n=128` AMR cases at contrast `1e5` both
converged with unchanged reported transmission and fewer Krylov iterations;
the Frontier-scale effect remains to be measured.
On the local LLVM/Open MPI build, matching AMR runs with two ranks at
`fine_n=64` took 6.20 s and 10/281 Newton/Krylov iterations with fixed
tolerance, versus 4.89 s and 10/83 with adaptive tolerance. Eight-rank
`fine_n=128` runs took 24.33 s and 11/298 versus 18.07 s and 13/107.
These are whole-case times from one final-build run per setting; local timing
varies, while the Krylov counts and reported transmission were repeatable.
The jobs require the GNU TinyProfiler executable and print the profile summary
to the job log at AMReX finalization. Timings include profiling and iteration
logging overhead.
The TinyProfiler summary now separates each AMG V-cycle level into residual,
restriction, prolongation, and pre/post-smoothing time. It also separates
`SpMatrix::finishComm_mv` into receive wait, unpack, send wait, and cleanup;
the cleanup timer includes GPU synchronization and buffer release. These
timers are nested, so their inclusive totals must not be added to the enclosing
V-cycle or SpMV totals.
All four jobs set `MPICH_OFI_CXI_COUNTER_REPORT=2`, which prints a summary of
Cassini counters at `MPI_Finalize`. The counters span the whole MPI run, not
individual AMG levels or solver phases. Compare pause cycles, PCIe blocked
cycles per packet, message-matching overflow, and retry counters alongside the
per-level timers. The counter summary does not include MPI collective traffic
attribution or explain a particular wait on its own.

The native AMG hierarchy and smoother controls can be overridden on the
command line with `mlabeclap_amg.strong_threshold`,
`mlabeclap_amg.max_interp_elements`, `mlabeclap_amg.max_levels`,
`mlabeclap_amg.max_coarse_size`, `mlabeclap_amg.pre_sweeps`,
`mlabeclap_amg.post_sweeps`, and `mlabeclap_amg.chebyshev_order`.
Their defaults are `0.25`, `4`, `25`, `9`, `1`, `1`, and `4`, respectively.
Set only one control per comparison and retain the same residual tolerance.
All four scaling jobs explicitly set `mlabeclap_amg.max_interp_elements=4`.
The cap-6 trial increased whole-case time at both 56 and 448 ranks, so it is
not part of the active scaling configuration.
Native AMG message accounting is enabled for every run. Each setup prints
rank-summed point-to-point send counts and payload bytes by level and phase,
plus logical collective-call counts; messages inside MPI collectives are not
counted.
All four jobs set `mlabeclap_amg.true_residual_factor=8`. With the
double-precision relative linear tolerance of `2e-10`, the checked true
residual limit is `1.6e-9` relative to the right-hand-side norm. Other runs
retain the solver default factor of 5.

Build the MPI executable from the repository root, then submit the cases:

```sh
gmake -j8 TINY_PROFILE=TRUE PROFILE=FALSE
sbatch scaling/weak_001.slurm
sbatch scaling/weak_008.slurm
sbatch scaling/weak_064.slurm
sbatch scaling/weak_512.slurm
```

Each job writes `%x-%j.out` in the directory from which `sbatch` is invoked.
