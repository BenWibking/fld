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
4,390,912 composite cells per node. Plotfiles and per-iteration output are
disabled so filesystem traffic does not distort the timing.

Build the MPI executable from the repository root, then submit the cases:

```sh
gmake -j8
sbatch scaling/weak_001.slurm
sbatch scaling/weak_008.slurm
sbatch scaling/weak_064.slurm
sbatch scaling/weak_512.slurm
```

Each job writes `%x-%j.out` in the directory from which `sbatch` is invoked.
