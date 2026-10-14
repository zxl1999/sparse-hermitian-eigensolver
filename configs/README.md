# Example configurations

JSON configuration samples for the most common regimes. A config file supplies
defaults; **explicit CLI flags always win**. Pass one with `--config`:

```bash
sparse-eigensolve data/raw/CEMW__vfem.npz --output results/vfem.npz \
  --config configs/memory_limited_ilu.json
```

To see the fully-resolved settings (defaults + config + CLI) without solving:

```bash
sparse-eigensolve any.npz --output /dev/null --config configs/default.json \
  --dump-config /tmp/resolved.json
```

| File | When to use | Key knobs |
|---|---|---|
| `default.json` | Sanity run / unknown matrix | `mode: auto`, exact LU |
| `hermitian_direct.json` | Physics guarantees `A = A^H` (lossless) | `mode: hermitian`, RCM + LU |
| `general_lossy_pml.json` | Complex-symmetric `A = A^T` from lossy/PML media (non-Hermitian) | `mode: general`, RCM + ILU |
| `memory_limited_ilu.json` | Sparse storage small but LU fill-in dominates memory | `inner_solver: ilu`, `ilu_fill_factor: 20` |
| `adaptive_ilu.json` | ILU parameters unknown or matrix-dependent | ilu_adaptive: true, ilu_memory_budget_mb |
| `known_band_adaptive.json` | Target frequency band `[lo, hi]` is known | `target_band`, `adaptive_shift` |
| `near_singular_probes.json` | `sigma` sits on/near an eigenvalue (singular `A - sigma I`) | `shift_probes`, `adaptive_shift` |
| `gpu_hermitian.json` | CUDA device is available and the matrix is Hermitian | `backend: gpu`, `mode: hermitian`, `ncv` |

For a generalized Hermitian problem `A x = lambda B x`, pass the mass matrix on the
CLI or in JSON configuration:

```bash
sparse-eigensolve A.npz --mass-matrix B.npz --output results/generalized.npz \
  --mode hermitian --sigma 1.25 -k 6
```

The Hermitian path validates the symmetry of `B` and factorizes `A - sigma B`
without forming a dense inverse.

The optional GPU path uses `backend: gpu` and requires the `gpu` extra. It supports
Hermitian standard problems and diagonal positive-definite generalized problems
with direct CUDA sparse factorization; non-Hermitian, non-diagonal-`B`, and ILU
cases remain on the CPU backend.

## Key concepts

- **`mode`**: `auto` (default) measures `||A - A^H|| / ||A||` and routes to the
  Hermitian (`eigsh`) or general (`eigs`) path. `hermitian`/`general` force it. This
  is a **correctness** choice: for a complex-symmetric (non-Hermitian) matrix the two
  paths return different spectra. Confirm the official convention with the organizers.
- **`shift_probes`** are **absolute offsets from `sigma`**, tried in order, before the
  blind retries. E.g. with `sigma = 0`, `[0.001, 0.01, 0.1]` tries shifts
  0.001, 0.01, 0.1.
- **`target_band`** is an absolute interval `[lo, hi]`; candidate shifts walk from the
  band centre toward the edges so a near-singular centre does not abort the solve.
- **`adaptive_shift`** adds a geometric backoff (`step, 2·step, 5·step, 10·step, …`) so
  a near-singular region is escaped decisively rather than by tiny steps.
- **`inner_solver: ilu`** replaces the exact LU with ILU + GMRES. Add `ilu_adaptive: true` to probe a bounded parameter ladder and choose the lowest predicted work within the memory budget. If it reports
  `Factor is exactly singular`, raise `ilu_fill_factor` (20 → 50): it is usually a
  drop-tolerance zero pivot, not a singular shift.


