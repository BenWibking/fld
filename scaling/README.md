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
The jobs require the GNU TinyProfiler executable and print the profile summary
to the job log at AMReX finalization. Timings include profiling and iteration
logging overhead.

The native AMG hierarchy and smoother controls can be overridden on the
command line with `mlabeclap_amg.strong_threshold`,
`mlabeclap_amg.max_interp_elements`, `mlabeclap_amg.max_levels`,
`mlabeclap_amg.max_coarse_size`, `mlabeclap_amg.pre_sweeps`,
`mlabeclap_amg.post_sweeps`, and `mlabeclap_amg.chebyshev_order`.
Their defaults are `0.25`, `4`, `25`, `9`, `1`, `1`, and `4`, respectively.
Set only one control per comparison and retain the same residual tolerance.
All four scaling jobs set `mlabeclap_amg.max_interp_elements=6`.
Native AMG message accounting is enabled for every run. Each setup prints
rank-summed point-to-point send counts and payload bytes by level and phase,
plus logical collective-call counts; messages inside MPI collectives are not
counted.
All four jobs set `mlabeclap_amg.true_residual_factor=6`. With the
double-precision relative linear tolerance of `2e-10`, the checked true
residual limit is `1.2e-9` relative to the right-hand-side norm. Other runs
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
