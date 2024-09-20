# Symmetric Dambreak

This is an OpenMPI and OpenMP Hybrid programming exercise. You are given 4 MPI processes and 4 OpenMP threads per process to achieve a speedup of at least 10.


## Thread binding / Thread affinity

For this exercise you can control thread binding settings of OpenMP and OpenMPI using special comments (see Makefile sed commands).

The submission server defaults were changed to something suboptimal, so you can't pass the assignment without changing some options to more reasonable settings.
There are multiple valid strategies.

Your submission will be started like this:

`bash -c 'env $(cat ./student_submission.env | xargs) mpirun -np 4 $(cat ./student_default_mpirun.opt | xargs) $(cat ./student_mpirun.opt | xargs) --report-bindings ./run_submission.sh'`

### OpenMPI

For OpenMPI you can only change the settings of `--rank-by` `--map-by` and `--bind-to` like this:

```
//!mpirun_option --rank-by object
//!mpirun_option --map-by object
//!mpirun_option --bind-to object
```

Replace object with your binding strategy.

### OpenMP

OpenMP can be controlled via environment variables like in previous assignments

```
//!submission_env ENV_VARIABLE=something
```

Hint: OMP_DISPLAY_ENV will only be printed if OMP is used in the code.