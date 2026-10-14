# Benchmark Run Report

Environment: macOS, Anaconda Python 3.14.6, NumPy 2.4.6, SciPy 1.18.0. Single process, CPU only. Matrices are the deterministic synthetic matrices from `sparse_eigensolver.benchmark`; all are tridiagonal (or diagonal for `near_shift`), so sparse LU fill-in is minimal and the memory numbers below are a lower bound, not representative of real electromagnetic matrices.

## Results

| Case | n | k | sigma | Wall time (s) | Max rel. residual | Peak RSS (MB) | Shifts tried |
| --- | --- | --- | --- | --- | --- | --- | --- |
| indefinite | 100000 | 6 | 1.0 | 0.25 | 6.1e-19 | 184 | 1 |
| clustered | 100000 | 6 | 0.0 | 0.28 | 7.5e-19 | 181 | 1 |
| repeated | 100000 | 6 | 0.0 | 9.39 | 2.7e-14 | 183 | 1 |
| near_shift | 100000 | 6 | 0.0 | 0.17 | 4.7e-19 | 168 | 2 (first shift singular) |
| near_shift | 20000 | 6 | 0.0 | 0.09 | 2.5e-18 | n/a | 2 (first shift singular) |

## Observations

- The `near_shift` case deliberately places an eigenvalue exactly at the requested shift. The shift-invert factorization fails with "Factor is exactly singular", and the retry logic moves the shift by a scaled offset and succeeds. Recorded eigenvalues still include the exact zero and the 1e-8 eigenvalue.
- The `repeated` case with clustered repeated eigenvalues is markedly slower (9.4 s, 76 s user time) than the others. ARPACK needs many more iterations to resolve degenerate eigenvalues; this is the expected weak point of a plain shift-invert Krylov method. Clustered spectra need a block or subspace-aware method to be efficient.
- Peak RSS stays under 200 MB, but this is because the synthetic matrices are tridiagonal. Real electromagnetic stiffness matrices have far more fill-in under sparse LU, so memory must be measured on the actual competition matrices.

## CLI end-to-end

`sparse-eigensolve results/near_shift_20000.npz --output results/eigenpairs.npz -k 6 --sigma 0.0` produced:

- `eigenpairs.npz` with `eigenvalues` shape `(6,)`, `eigenvectors` shape `(20000, 6)`, `residuals` shape `(6,)`.
- Max relative residual 6.8e-19.
- `eigenpairs.json` diagnostics with the solver identity, attempts, and timing.

All 7 unit tests pass.
