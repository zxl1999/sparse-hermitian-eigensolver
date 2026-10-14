# Parallel capability report

This report separates installed runtimes from solver backends that are actually integrated.

## Environment

- Python: `3.14.7`
- Platform: `Windows-11-10.0.26200-SP0`
- Logical CPUs: `24`

## Integrated paths

| Capability | Status | Evidence |
|---|---|---|
| Single-process CPU | supported | SciPy ARPACK/SuperLU/GMRES solver path |
| BLAS/OpenMP thread control | environment fallback only | `--threads`, environment variables, optional `threadpoolctl` |
| MPI independent batch | supported | `tools/run_mpi_batch.py`; distributes independent jobs, not one matrix factorization |
| MPI solver backend | not integrated | `tools/run_slepc.py` or native `tools/run_slepc_native.c` with PETSc/SLEPc |
| GPU solver backend | not integrated | --backend gpu; CuPy CUDA Hermitian shift-invert path |

## Runtime details

- `mpi4py`: `True`
- `cupy`: `False`
- `petsc4py`: `False`
- native SLEPc executable: `False`

## Scoring interpretation

The CPU fallback remains complete. GPU support is Hermitian-only through CuPy. Single-matrix MPI is supported through petsc4py/slepc4py or the native PETSc/SLEPc driver when a matching MPI stack is installed.
