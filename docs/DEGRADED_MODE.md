# Degraded mode: CPU fallback, optional GPU, low-memory operation

The competition environment (约束 d) may not offer GPUs, may cap memory, and may
run headless. This document records how the solver behaves when resources are
scarce and which knobs to turn. The CPU path remains complete and is the required
fallback; an optional CuPy/CUDA path is available for Hermitian problems.

## Baseline capabilities

- **CPU fallback.** ARPACK (Lanczos/Arnoldi) and SuperLU are the default CPU routines;
  results are bit-for-bit reproducible for a fixed BLAS and thread count. The GPU
  path is opt-in and does not replace the CPU fallback.
- **Headless.** No display, window, or service is needed. The CLI writes files and
  prints JSON.
- **Single matrix process.** The default CPU solve is rank-local. A true distributed
  single-matrix path is available through `tools/run_slepc.py` when a matching
  PETSc/SLEPc/MPI stack is installed. Independent jobs can also be distributed with
  `tools/run_mpi_batch.py`; parallelism inside one CPU job comes only from the
  BLAS/LAPACK threads (controlled via `--threads N` / `set_threads()`, which also
  export `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`). Measured
  scaling is poor (see [`../results/SPEEDUP.md`](../results/SPEEDUP.md)): pinning
  `--threads 1` maximizes reproducibility and avoids oversubscription.
- **Capability reporting.** Run `make capabilities` before submission. The report
  distinguishes integrated solver backends from optional packages installed in the
  environment; `mpi4py` alone does not provide MPI eigensolver support.

## Degradation ladder (most → least memory)

When peak memory is the binding constraint, switch strategies in this order. Each
step trades some inner-solve accuracy/time for memory.

1. **Exact LU (default).** `--inner-solver direct`. Highest accuracy, highest
   memory. Fill-in can be hundreds of times `nnz(A)` on 3-D EM matrices.
2. **Reorder first.** `--reorder rcm` (and/or `--permc-spec MMD_AT_PLUS_A`). Free
   win when it helps: reduces bandwidth/fill before factorizing, spectrum unchanged.
3. **ILU + GMRES.** `--inner-solver ilu --ilu-fill-factor 20`. Replaces the exact
   factorization with an incomplete one plus an iterative inner solve. On the public
   matrices this cut fill-in from ~148× to ~1.6× (`CEMW__vfem`) and from ~247× to
   ~19× (`Um__2cubes_sphere`). If it reports `Factor is exactly singular`, raise
   `--ilu-fill-factor` (20 → 50) — it is usually a drop-tolerance zero pivot, not a
   singular shift.
4. **Larger drop tolerance.** `--ilu-drop-tol 1e-2` … `1e-1` shrinks the ILU factor
   further at the cost of more GMRES iterations. Use only if the residual tolerance
   is still met.
5. **Reduce `k` / `ncv`.** Ask for fewer eigenpairs and/or a smaller Krylov subspace
   (`--ncv`). Memory in ARPACK scales with `n * ncv`.

### Hard memory budget

On Linux, cap the process with `ulimit -v` (or `RLIMIT_AS`) before launching. On
**macOS `RLIMIT_AS` cannot be set to a finite value** (`setrlimit` raises
`ValueError: current limit exceeds maximum limit`), so `tools/run_fault_scenarios.py`
enforces the budget by monitoring the child's RSS from the parent and killing it on
breach — see the `memory_limited` scenario. Use the same pattern in production:
run the solve as a child process and watch its RSS.

On Windows, `tools/run_fault_scenarios.py` reads child-process working-set memory
through `GetProcessMemoryInfo`; on Linux/macOS it uses the platform process tools.

## What is *not* degraded silently

- **Correctness is never traded away silently.** Every returned eigenpair is
  residual-checked against the **original** matrix; if the check fails the solve
  raises rather than returning an unverified answer.
- **The effective shift and strategy are always recorded** in the diagnostics JSON
  (`effective_shift`, `strategy`, `factor_kind`, `factor_nnz`, `n_attempts`), so a
  degraded run is auditable after the fact.

## Recommended environment variables

```bash
export OMP_NUM_THREADS=4          # or 1 for maximum reproducibility
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONHASHSEED=0
```

For a fully deterministic submission, pin the thread count and the BLAS vendor, and
record them alongside the results.
