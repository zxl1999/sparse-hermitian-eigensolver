# Adaptive ILU benchmark

Deterministic offline benchmark for bounded-probe ILU parameter selection.
The adaptive path minimizes predicted inner work under a 256 MB approximate factor budget.

| Case | n | Path | Status | Residual | Factor nnz | Plan seconds | Selected rung |
|---|---:|---|---|---:|---:|---:|---|
| clustered | 400 | fixed | ok | 2.4149131894618948e-12 | 1600 | 0.000 | — |
| clustered | 400 | adaptive | ok | 1.720672234053035e-13 | 1602 | 0.005 | ff=50.0, tol=1e-06 |
| complex_symmetric | 300 | fixed | ok | 3.444739997663382e-17 | 1200 | 0.000 | — |
| complex_symmetric | 300 | adaptive | ok | 6.224190132195758e-17 | 1200 | 0.003 | ff=5.0, tol=0.0001 |
| memory_limited | 900 | fixed | ok | 6.943218357994344e-13 | 30098 | 0.000 | — |
| memory_limited | 900 | adaptive | ok | 5.951409990913804e-13 | 30564 | 0.013 | ff=20.0, tol=1e-05 |

The JSON artifact records the full rung trace, including skipped, unconverged, and selected candidates.
