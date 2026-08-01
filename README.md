## FLD

Build:
```
git submodule update --init
gmake -j8
```

Run:
```
mpirun -np 8 ./main2d.gnu.MPI.ex cloud_fine_n=1024 cloud_only=1
```

Plot:
![Clouds contour plot](clouds.png)
