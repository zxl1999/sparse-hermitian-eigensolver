# Sparse Eigenvalue Solver (Hermitian and general)

A reproducible Python prototype for computing eigenpairs of large sparse matrices near a requested target shift. It supports two paths:

- **Hermitian** (real symmetric or complex Hermitian): SciPy's ARPACK-backed `eigsh` in shift-invert mode.
- **General** (non-Hermitian, e.g. complex symmetric from lossy/PML media): SciPy's ARPACK-backed `eigs` in shift-invert mode (bi-orthogonal, `A^H`-based residuals).

By default (`--mode auto`) the solver measures the relative asymmetry `||A - A^H|| / ||A||` and routes the matrix automatically. Both paths keep matrix operations sparse.

> **Open question for the organizers / 待官方确认**: it is not yet confirmed whether the competition matrices are **complex-Hermitian** (`A = A^H`) or **complex-symmetric** (`A = A^T`, which arises from lossy/PML media and is *not* Hermitian). This choice changes the algorithm (Hermitian ARPACK vs. non-Hermitian bi-orthogonal ARPACK) and the meaning of the eigenpairs. We have implemented **both** and verified they give different spectra on a real non-Hermitian waveguide matrix, so the convention must be pinned down before final submission. Until confirmed, keep `--mode auto` and report `matrix_class` / `resolved_mode` in the diagnostics for every run.


## Setup

The project requires **Python >= 3.10** (declared in `pyproject.toml`). On this machine the system `python3` is 3.9.6 and cannot install the package, so use an interpreter that is 3.10+ (here, the Anaconda `python`, 3.14.6). Create an isolated environment and install the project, including test tools:

```bash
cd path/to/project
python -m venv .venv            # use a Python >= 3.10 interpreter
source .venv/bin/activate
python -m pip install -e '.[test]'
```

If you prefer not to create a venv, install into the active (3.10+) environment directly:

```bash
cd path/to/project
python -m pip install -e '.[test]'
```

This installs two commands: `sparse-eigensolve` and `sparse-eigen-benchmark`. To install dependencies without installing the package, use `python -m pip install -r requirements.txt` and run with `PYTHONPATH=src python -m ...` as shown below.

For reliable `--threads` control (via `threadpoolctl`), install the optional `perf` extra: `python -m pip install -e '.[test,perf]'`. Without it, `--threads` still exports the standard `OMP_NUM_THREADS`/`OPENBLAS_NUM_THREADS`/`MKL_NUM_THREADS` variables.

## Input and execution

- Dense input: a square two-dimensional NumPy array saved as `.npy`.
- Sparse input: a SciPy sparse matrix saved with `scipy.sparse.save_npz` as `.npz`.
- Interoperable electromagnetic input: Matrix Market `.mtx`/`.mm` (including gzip-compressed files) is converted to CSR on load; use `tools/convert_em_matrix.py` to normalize it to `.npz`.
- The matrix must be finite and square. In `--mode auto` (default) a matrix with relative asymmetry `||A - A^H||/||A|| <= --hermitian-tol` (default `1e-12`) uses the Hermitian `eigsh` path and is symmetrized to `(A + A^H)/2`; otherwise the general `eigs` path is used on the matrix as-is. Dense `.npy` input is converted to CSR on load.

Run the installed CLI:

```bash
sparse-eigensolve matrix.npz --output results/eigenpairs.npz -k 6 --sigma 1.25
```

Force a path explicitly when the physical convention is known:

```bash
sparse-eigensolve matrix.npz --output out.npz -k 6 --sigma 0.0 --mode hermitian   # A = A^H
sparse-eigensolve matrix.npz --output out.npz -k 6 --sigma 0.0 --mode general     # complex-symmetric / non-Hermitian
```

Or run directly from the source tree:

```bash
PYTHONPATH=src python -m sparse_eigensolver matrix.npz --output results/eigenpairs.npz -k 6 --sigma 1.25
```

The result archive contains `eigenvalues`, `eigenvectors` (columns correspond to eigenvalues), and normalized `residuals`. A JSON diagnostics file with the same base name records attempted shifts, elapsed time, matrix size and nonzero count, and residual quality.

### Official data and output contract

The current competition PDF does **not** include an official matrix package, fixed
filenames, fixed archive keys, or a prescribed generalized-eigenvalue `B`-matrix
interface. It only requires NumPy/SciPy-compatible input and output. Therefore:

- `data/raw/` contains public SuiteSparse replacement benchmarks, not official
  competition data; the final submission must identify them as substitutes.
- Accepted input is a two-dimensional square `.npy` array, a SciPy sparse `.npz`
  created by `scipy.sparse.save_npz`, or a Matrix Market `.mtx/.mm` file. The solver keeps the computation sparse after
  loading; `.npy` is converted to CSR only at the input boundary.
- The combined result archive is a NumPy `.npz` with these required fields:
  `eigenvalues` with shape `(k,)`, `eigenvectors` with shape `(n, k)`, and
  `residuals` with shape `(k,)`. The adjacent `.json` file is diagnostic metadata,
  not a substitute for the numerical result.
- To provide separate NumPy files when an external evaluator asks for them, run:

  ```bash
  python tools/export_results.py results/eigenpairs.npz --out-dir results/exported
  python tools/export_results.py results/eigenpairs.npz --out-dir results/exported --scipy-vectors
  ```

  This writes `*_eigenvalues.npy`, `*_eigenvectors.npy`, and `*_residuals.npy`;
  `--scipy-vectors` additionally writes `*_eigenvectors_scipy.npz` using
  `scipy.sparse.save_npz`. The default dense NumPy vector is the canonical output
  because computed eigenvectors are generally dense; the SciPy sparse form is an
  interoperability export, not a claim that the vectors are sparse.
  The optional generalized problem `A x = lambda B x` is supported with
  `--mass-matrix B.npz`; the Hermitian path requires `B` to be Hermitian and uses
  sparse shift-invert on `A - sigma B`. The diagnostic JSON records
  `problem_type` and mass-matrix metadata. The official file naming, units, and
  frequency-to-`sigma` mapping still need confirmation from the organizer.
- If the organizer supplies a different schema or a domain-specific electromagnetic
  format, treat that as a submission gate and add an explicit adapter before claiming
  compatibility. Proprietary solver exports still require a sample file and schema from
  the organizer; no undocumented binary format is guessed.

### GPU backend

The optional CuPy backend runs Hermitian standard problems, and generalized problems
whose positive-definite `B` is diagonal, on one CUDA device. It uses sparse CUDA
factorization of the transformed shift-invert operator and CuPy's Lanczos path:

```bash
python -m pip install -e '.[gpu]'
sparse-eigensolve matrix.npz --output results/gpu.npz --backend gpu \
  --mode hermitian --ncv 32 -k 6 --sigma 0.0
```

On machines whose GPU compute capability is newer than the CuPy wheel's bundled
binary targets, set `CUPY_COMPILE_WITH_PTX=1`; if CUDA is installed outside the
default search path, also set `CUDA_PATH` before launching the command.

GPU diagnostics include `backend`, `gpu_name`, and residual verification. The GPU path
does not currently implement non-Hermitian Arnoldi or ILU; dense/non-diagonal `B`
also remains on CPU.

Useful options: `--mode`, `--hermitian-tol`, `--tol`, `--residual-tol`, `--maxiter`, `--ncv`, `--retries`, and `--shift-step`. If factorization at the requested shift fails or returns eigenpairs that fail residual validation, the solver tries nearby shifts. Results are sorted by distance to the originally requested shift and verified against the original matrix.

### Triangular storage and thread control

Some electromagnetic packages store only one triangle of a symmetric/Hermitian matrix. `--triangular` rebuilds the full matrix on load:

```bash
sparse-eigensolve lower.npz --output out.npz --triangular auto   # strict triangle -> Hermitian; else symmetric
sparse-eigensolve lower.npz --output out.npz --triangular hermitian   # mirror with conjugation:  tril(A,0) + tril(A,-1)^H
sparse-eigensolve lower.npz --output out.npz --triangular symmetric   # mirror without conjugation: tril(A,0) + tril(A,-1)^T
```

`auto` uses the diagonal to decide: a real matrix (or one with a real diagonal) is treated as Hermitian (conjugate mirror); a complex matrix with an imaginary diagonal is treated as complex-symmetric (plain transpose). The conjugation distinction is exactly the Hermitian vs. complex-symmetric (lossy/PML) split, so pick explicitly when the convention is known.

`--threads N` caps the BLAS/OpenMP pool for the inner LU/GMRES (via `threadpoolctl` plus the standard env vars), which makes runs reproducible and enables the speedup study:

```bash
sparse-eigensolve matrix.npz --output out.npz --threads 4 -k 6 --sigma 0.0   # pin to 4 threads
make speedup                                                                  # -> results/SPEEDUP.md
```

`--threads 0` (or omitting the flag) leaves the BLAS to choose. See [`docs/DESIGN.md`](docs/DESIGN.md) §6 and [`results/SPEEDUP.md`](results/SPEEDUP.md) for measured scaling.

### Generalized eigenvalue input

For a generalized Hermitian problem `A x = lambda B x`, pass `B` with
`--mass-matrix`. The solver factorizes the sparse operator `A - sigma B`, passes
`B` to ARPACK, and validates the returned generalized residual
`||A x - lambda B x|| / ((||A|| + |lambda| ||B||) ||x||)`:

```bash
sparse-eigensolve A.npz --mass-matrix B.npz --output results/generalized.npz \
  --mode hermitian --sigma 1.25 -k 6
```

The JSON diagnostics identify the run as `problem_type: generalized` and include
the shape, nonzero count, and dtype of `B`. Use `--mass-triangular` when the mass
matrix is stored as one triangle. The official filename, units, and mapping from
physical frequency to `sigma` still require confirmation from the organizer.

### Config files, known bands, and adaptive shifts

Every run can be described by a JSON config file; explicit CLI flags always win over the file, and `--dump-config` prints the fully-resolved settings without solving. Sample configs live in [`configs/`](configs/README.md).

```bash
# Reuse a tuned configuration (memory-limited ILU), then override k on the CLI
sparse-eigensolve matrix.npz --output out.npz --config configs/memory_limited_ilu.json -k 8

# Show the resolved settings (defaults + config + CLI) and exit
sparse-eigensolve matrix.npz --output out.npz --config configs/default.json --dump-config /tmp/resolved.json
```

When the target frequency band is known, tell the solver so it can walk candidate shifts from the band centre outward instead of aborting on a near-singular centre. When `sigma` itself sits on/near an eigenvalue, give explicit probe offsets and/or enable the adaptive geometric backoff:

```bash
# Known band [1.0, 2.0]: candidate shifts start at the band centre
sparse-eigensolve matrix.npz --output out.npz --target-band 1.0 2.0 --adaptive-shift

# sigma near a singular point: try absolute offsets 1e-3, 1e-2, 0.1, then back off geometrically
sparse-eigensolve matrix.npz --output out.npz --sigma 0 --shift-probes 1e-3 1e-2 0.1 --adaptive-shift
```

`--shift-probes` are **absolute offsets from `sigma`**, tried in order. `--adaptive-shift` adds a geometric backoff (`step, 2*step, 5*step, 10*step, 20*step, 50*step, 100*step`, both signs) so a near-singular region is escaped decisively rather than by tiny steps.

### Accuracy and iteration metrics

Each run's diagnostics JSON records the iteration counts and, when a reference is supplied, the accuracy of the returned subspace:

- `arpack_iterations` — outer Arnoldi/Lanczos applications of the shift-invert operator.
- `inner_matvecs` — applications of `(A - sigma I)`: one per LU solve (direct), or one per GMRES iteration (ILU).
- `factor_kind` / `factor_nnz` — `LU` or `ILU`, and the factorization's nonzero count (fill-in).
- `n_attempts`, `target_band`, `adaptive_shift`, `effective_shift` — which shift finally succeeded and why.

To quantify correctness against a trusted reference (e.g. a direct-LU run), pass its result file with `--reference`. The solver reports the **principal angles** (degrees) between the two subspaces and the max eigenvalue error:

```bash
sparse-eigensolve matrix.npz --output out.npz --reference results/direct_reference.npz
```

This prints a `subspace_angle` block (`max_angle_deg`, `mean_angle_deg`, `angles_deg`, `match`, `max_eigenvalue_error`). Matching is by **value** on the Hermitian path and by **direction** on the general path. Measured example: an ILU+GMRES run vs. an exact-LU reference on the same matrix agrees to a max subspace angle of `8.5e-7` degrees with max eigenvalue error `2.2e-18` — i.e. the fast path reproduces the exact spectrum to machine precision.

### Reducing factorization fill-in (memory/time)

Shift-invert factors `A - sigma I`; on real EM matrices the sparse-LU **fill-in**, not `n`, dominates cost. Three independent knobs reduce it, all preserving the spectrum:

```bash
# Reorder the matrix (reverse Cuthill-McKee) before factorizing
sparse-eigensolve matrix.npz --output out.npz -k 6 --sigma 0.0 --reorder rcm

# Change SuperLU's column ordering (default COLAMD; MMD_* often smaller)
sparse-eigensolve matrix.npz --output out.npz -k 6 --sigma 0.0 --permc-spec MMD_AT_PLUS_A

# Replace the exact LU with ILU + GMRES (much lower memory, iterative inner solve)
sparse-eigensolve matrix.npz --output out.npz -k 6 --sigma 0.0 --inner-solver ilu --ilu-drop-tol 1e-4
```

Options: `--permc-spec {NATURAL,MMD_AT_PLUS_A,MMD_ATA,COLAMD}`, `--reorder {none,rcm}`, `--inner-solver {direct,ilu}`, `--ilu-drop-tol`, `--ilu-fill-factor`, `--gmres-restart`, `--gmres-rtol`. With `--reorder rcm` the solver factors `P A P^T` and maps eigenvectors back, so residuals are always checked against the original matrix. See `results/STRATEGY_COMPARISON.md` for measured fill-in and runtimes.

Measured on real matrices, the payoff is large and matrix-dependent (fill ratio = factor nnz / nnz(A)):

- `Oberwolfach__gyro` (n=17k): `MMD_AT_PLUS_A` **3.05x** vs `COLAMD` 6.63x vs `NATURAL` 24.72x.
- `Um__2cubes_sphere` (n=101k): exact LU is 247x fill (~406M nonzeros, ~5.5 min, ~4.1 GB); `--inner-solver ilu --ilu-fill-factor 20` cuts the factor to **18.8x** (~33 s on an idle machine), but GMRES convergence on this ill-conditioned matrix can need a much longer timeout under load.
- `CEMW__vfem` (n=93k, complex): exact LU is 148x (~212M nonzeros, ~500 s); ILU cuts it to **1.56x** (~6 s).

Two practical notes: (1) ordering choice is matrix-dependent — run a cheap trial before committing; never ship `NATURAL` as a default. (2) If ILU fails with `Factor is exactly singular`, it is usually a drop-tolerance zero pivot, not a singular shift — raise `--ilu-fill-factor` (10 -> 20 -> 50) first.


## Make targets

Common tasks are wrapped in a [`Makefile`](Makefile) (`make help` lists them all):

```bash
make install            # pip install -e '.[test]'
make test               # pytest -q
make bench              # synthetic benchmark
make strategies         # fill-in / reordering comparison  -> results/STRATEGY_COMPARISON.md
make fault-scenarios    # 故障场景表 runner               -> results/fault_scenarios.json
make accuracy           # subspace-angle accuracy report   -> results/ACCURACY.md
make speedup            # BLAS/OpenMP thread scaling       -> results/SPEEDUP.md
make reports            # render the markdown reports
make fetch-data         # download public SuiteSparse matrices into data/raw/
```

`make fault-scenarios` writes the raw `results/fault_scenarios.json`; `make fault-report` renders it into `results/FAULT_SCENARIOS.md` (精度/性能/迭代次数/处置建议 per scenario); `make accuracy` writes `results/ACCURACY.md`; `make speedup` writes `results/SPEEDUP.md` (加速比对比表).

## Tests

```bash
python -m pip install pytest
PYTHONPATH=src python -m pytest -q
```

Small test cases compare with dense `eigvalsh` as a correctness oracle only. Dense conversion is not used in the solver path.

Run deterministic synthetic benchmarks (no dense conversion):

```bash
PYTHONPATH=src python -m sparse_eigensolver.benchmark --n 10000 --case indefinite -k 6 --seed 2026 --output results/benchmark.json --save-matrix results/indefinite.npz
```

Available cases are `indefinite`, `clustered`, `repeated`, `near_shift`, `low_freq_breakdown`, `memory_limited`, `memory_limited_3d`, `complex_hermitian`, and `complex_symmetric`. The benchmark reports elapsed time, eigenvalues, residuals, orthogonality error, solver attempts, and the process **peak RSS** (`peak_rss_mb`). Increase `--n` gradually on the target machine; runtime and factorization fill-in are matrix and platform dependent.

## Public test data

The competition provides no official matrices. `tools/fetch_matrices.py` downloads freely available electromagnetic matrices from the SuiteSparse Matrix Collection, converts them to `.npz`, and saves them under `data/raw/`. See [data/README.md](data/README.md) for the catalog, measured runtimes, and caveats.

```bash
python tools/fetch_matrices.py --list
python tools/fetch_matrices.py --out data/raw                      # Hermitian matrices only
python tools/fetch_matrices.py --out data/raw --names FEMLAB/waveguide3D --keep-general   # keep a non-Hermitian one for the general path
```

## Limits and engineering notes

- `eigsh` requires `1 <= k < n`. For very small matrices or requests for most of the spectrum, use a separate dense workflow only when explicitly allowed by the application.
- Shift-invert computes solves involving `A - sigma I`. Sparse LU fill-in can dominate memory and runtime; matrix dimension alone does not guarantee a fit within 16 or 32 GB. Monitor process peak RSS on representative matrices. On real matrices this is severe: a 101k-order real-symmetric problem took ~397 s / 4.4 GB, and a 93k-order complex-Hermitian FEM problem took ~440 s / 4.0 GB, whereas synthetic 100k tridiagonal cases finish in ~0.3 s / 0.18 GB.
- **Hermitian vs. general is a correctness issue, not a style choice.** For a non-Hermitian complex-symmetric matrix the two paths return different spectra. Example (`FEMLAB/waveguide3D`, n=21,036, relative asymmetry 0.179, sigma=0.0): general path `[-3.8e-4+2.1e-6j, -1.7e-4+4.2e-5j, ...]` vs. forced-Hermitian path `[-1.8e-4, 8.4e-5, 4.0e-4, 5.3e-4]` (all real) — the spectra do not overlap. Confirm the convention with the organizers.
- Nearby-shift retries are a robustness mechanism, not a substitute for singular-system treatment. The requested shift and the effective factorization shift are both recorded. Candidate eigenpairs are ranked and residual-checked against the original matrix, but a finite `k` solve cannot prove that no unreturned eigenvalue lies closer to the target.
- The default backend is single-process CPU, with BLAS/OpenMP thread control and a
  complete CPU fallback. `--backend gpu` adds a validated CuPy Hermitian path on CUDA;
  non-Hermitian and ILU cases remain on CPU. `tools/run_mpi_batch.py` distributes
  independent jobs, while `tools/run_slepc.py` is the optional true distributed
  single-matrix path when PETSc/SLEPc is installed. `make capabilities` records which
  of these optional runtimes are actually usable. Measured BLAS/OpenMP thread scaling is poor
  (`make speedup` → [`results/SPEEDUP.md`](results/SPEEDUP.md)): the sparse
  shift-invert workload is memory-bandwidth-bound, so reducing LU fill-in (matrix
  reordering, preconditioned iterative inner solves) — not more threads — is the
  main lever for the competition's memory/time targets. See [`docs/DESIGN.md`](docs/DESIGN.md)
  and [`docs/SUBMISSION_AUDIT.md`](docs/SUBMISSION_AUDIT.md).
- ARPACK/SciPy licensing and dependency versions should be documented with the final competition submission; no closed-source service is used.

## Fault scenarios (故障场景表)

`tools/run_fault_scenarios.py` exercises the failure modes the competition asks about
(near-singular shift, low-frequency breakdown, repeated/clustered eigenvalues,
**memory-limited** runs, and real ill-conditioned matrices) and writes
`results/fault_scenarios.json`. `results/FAULT_SCENARIOS.md` tabulates, per scenario, the
**accuracy / performance / iteration counts / mitigation (处置建议)**.

```bash
python -W ignore tools/run_fault_scenarios.py                  # all scenarios
python -W ignore tools/run_fault_scenarios.py --names memory_limited near_shift
python -W ignore tools/run_fault_scenarios.py --merge          # merge into the existing JSON
```

The **memory-limited** scenario is enforced by a parent-side RSS monitor: macOS cannot set a
finite `RLIMIT_AS`, so the runner polls the child's resident set size and kills it once it
crosses the budget. It runs a 3D 7-point Laplacian `A = L + I` (n = 40^3 = 64,000, only
~4.4e5 nonzeros) under a 1 GB budget: exact LU fill-in is ~1500x (~1.7 GB), so LU is killed
while ILU+GMRES finishes inside the budget (~0.29 GB, resid ~5e-12) — a concrete
memory/accuracy trade-off.

## Documentation and licensing

- [`docs/DESIGN.md`](docs/DESIGN.md) — 方案说明文档：算法原理、复杂度与内存分析、收敛性、预处理、并行方案、评测结果、求解边界、诊断流程、安全设计。
- [`docs/DEGRADED_MODE.md`](docs/DEGRADED_MODE.md) — what happens when a run cannot meet the
  requested residual/accuracy (retry ladder, degraded results, when to switch strategy).
- [`docs/SUBMISSION_AUDIT.md`](docs/SUBMISSION_AUDIT.md) — 对照题目 PDF 的数据、输出、并行能力与提交风险清单。
- [`SECURITY.md`](SECURITY.md) — sensitive-data handling; large downloaded matrices and bulky
  run artefacts are git-ignored.
- [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) — dependency licenses (NumPy, SciPy/ARPACK, SuperLU).
- [`LICENSE`](LICENSE) — BSD-3-Clause.
- [`configs/README.md`](configs/README.md) — annotated sample configurations.

### Interfaces and permissions actually used

This prototype calls only in-process, open-source numerical libraries — no external
service, no proprietary solver, and no privileged operation:

- **Sparse eigenproblem**: SciPy's ARPACK bindings `scipy.sparse.linalg.eigsh`
  (Hermitian Lanczos) and `eigs` (general Arnoldi).
- **Sparse direct solve**: SuperLU via `scipy.sparse.linalg.splu` (real and complex).
- **Iterative inner solve**: `scipy.sparse.linalg.spilu` (incomplete LU) + `gmres`
  (with `LinearOperator`).
- **Reordering**: `scipy.sparse.csgraph.reverse_cuthill_mckee`; SuperLU column
  orderings (`NATURAL`/`MMD_AT_PLUS_A`/`MMD_ATA`/`COLAMD`).
- **Dense kernels**: LAPACK/BLAS through NumPy and SciPy.
- **Threads**: BLAS/OpenMP pools capped via `threadpoolctl` (optional) and the
  standard `OMP_NUM_THREADS`/`OPENBLAS_NUM_THREADS`/`MKL_NUM_THREADS` variables.
- **Capability audit**: `make capabilities` reports integrated CPU/thread/GPU paths
  separately from merely installed MPI/PETSc/SLEPc runtimes.
- **MPI batch**: install `python -m pip install -e '.[mpi]'`, then run
  `mpiexec -n 2 python tools/run_mpi_batch.py --jobs configs/mpi_batch_example.json \
  --output results/mpi_batch.json`. This distributes independent jobs only; it does
  not partition one sparse matrix across ranks.
- **Distributed single matrix**: install a matching PETSc/SLEPc + MPI stack, then run
  `mpiexec -n 4 python tools/run_slepc.py matrix.npz --output results/slepc.npz -k 6 --sigma 0.0`.
  On Linux this is the Python binding path. On Windows, MSYS2 UCRT64 provides a native
  PETSc/SLEPc driver: convert the input with `tools/convert_em_matrix.py`, build with
  `make slepc-native-build SLEPC_PKG=slepc-dmo SLEPC_EXE_SUFFIX=.exe`, then run
  `mpiexec -n 4 tools/run_slepc_native.exe matrix.mtx results/slepc_native.json 4 0.0
  1e-10`. Add the MSYS2 UCRT64 and active MPI DLL directories to `PATH`. The driver
  accepts real symmetric Matrix Market input, distributes one matrix by rows, and records
  rank count, convergence, elapsed time, eigenvalues, and residuals.

Permissions:

- **Filesystem**: reads a user-specified matrix (`.npy`/`.npz`/`.mtx`) and writes results
  (`.npz` + `.json`) to user-specified paths. Nothing else is touched.
- **Network**: **not used at solve time.** The only network access is the optional
  `tools/fetch_matrices.py`, which downloads public SuiteSparse matrices when the
  user explicitly runs it; the solver itself runs fully offline.
- **Privileges**: none. No `sudo`, no daemons, no listening sockets, and no driver installation or modification.
- **Determinism**: no randomness in the solve path (benchmark matrices are seeded).
  With a fixed BLAS and thread count, results are bit-for-bit reproducible.


### Adaptive ILU selection

Use --inner-solver ilu --ilu-adaptive --ilu-memory-budget-mb 1024 to run bounded GMRES probes and select the ILU rung with the lowest predicted inner work. The full decision trace is written to the diagnostics JSON. make adaptive-ilu regenerates the offline benchmark artifacts.

