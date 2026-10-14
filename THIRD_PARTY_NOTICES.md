# Third-party notices and attribution

This prototype is built entirely on open-source scientific-Python components. No
closed-source or network service is used at run time: every computation is local,
deterministic, and reproducible. The table below lists the direct and transitive
dependencies that materially affect the numerical results, with their licenses.

| Component | Role | License | Notes |
|---|---|---|---|
| **NumPy** (`numpy>=2.0`) | dense arrays, linear algebra, `eigvalsh`/`eigh` oracle | BSD-3-Clause | https://numpy.org |
| **SciPy** (`scipy>=1.14`) | sparse matrices, ARPACK bindings, SuperLU, GMRES | BSD-3-Clause | https://scipy.org |
| **ARPACK** (bundled in SciPy `scipy.sparse.linalg`) | implicitly restarted Lanczos / Arnoldi (`eigsh`, `eigs`) | BSD-3-Clause | Original ARPACK is BSD-3; see `scipy/sparse/linalg/_eigen/arpack/COPYING` |
| **SuperLU** (bundled in SciPy) | sparse LU factorization (`splu`) | BSD-3-Clause (SuperLU 5.x) | Used for the exact shift-invert inner solve |
| **LAPACK / BLAS** (via NumPy/SciPy) | dense kernels for SuperLU and the GMRES projections | BSD-3-Clause (reference), or vendor-specific | OpenBLAS/Accelerate depending on the wheel |
| **pytest** (`pytest>=8`, test extra only) | unit tests | MIT | Not required at run time |
| **threadpoolctl** (`threadpoolctl>=3`, `perf` extra, optional) | cap BLAS/OpenMP threads for `--threads` / reproducibility | BSD-3-Clause | https://github.com/joblib/threadpoolctl ; if absent, the solver falls back to the `OMP_NUM_THREADS`/`OPENBLAS_NUM_THREADS`/`MKL_NUM_THREADS` env vars |
| **mpi4py** (`mpi4py>=4.0`, `mpi` extra, optional) | distribute independent solve jobs with `tools/run_mpi_batch.py` | BSD-3-Clause | Requires an MPI launcher/runtime; it is not a distributed single-matrix eigensolver |
| **CuPy** (`cupy-cuda12x>=14.0`, `gpu` extra, optional) | CUDA sparse factorization and Hermitian Lanczos backend | MIT | Requires a compatible NVIDIA CUDA runtime; GPU path is opt-in and Hermitian-only |
| **PETSc / petsc4py** (`slepc` extra, optional) | distributed sparse matrix and linear algebra backend | BSD-2-Clause | Required by the optional single-matrix MPI runner |
| **SLEPc / slepc4py** (`slepc` extra, optional) | distributed Krylov-Schur / shift-and-invert eigensolver | BSD-2-Clause | Must be built against the same PETSc and MPI runtime |

## How attribution is handled

- The `LICENSE` file at the repository root covers **this project's** source code
  (BSD-3-Clause).
- Upstream licenses are **not** vendored here; they are installed transitively by
  `pip` and remain in each package's distribution. For a competition submission,
  reproduce the notices above in the accompanying documentation, or run
  `python -m pip install -e '.[test]' && pip-licenses` (if available) to dump the
  exact resolved versions and licenses of the environment.

## Reproducibility / pinned versions

For a frozen, reproducible submission, pin the environment after a successful run:

```bash
python -m pip freeze > requirements.lock.txt
```

`requirements.txt` lists the direct runtime dependencies with lower bounds;
`requirements.lock.txt` (generated above) captures the exact versions actually
used to produce the reported numbers.
