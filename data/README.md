# Public Electromagnetic Sparse Matrix Data

The competition PDF does not ship official matrices. These public, freely redistributable matrices come from the SuiteSparse Matrix Collection (https://sparse.tamu.edu), which aggregates academic test sets. `tools/fetch_matrices.py` downloads and converts them to SciPy `.npz`.

**Hermitian convention caveat:** most entries here are real-symmetric (hence Hermitian) or complex-Hermitian, which the solver handles directly. Real lossy/PML electromagnetic problems however produce **complex-symmetric (non-Hermitian)** matrices; `FEMLAB/waveguide3D` is kept as one of those to exercise the general path. Whether the competition uses complex-Hermitian or complex-symmetric matrices must be confirmed with the organizers — see the top-level README.

## Download

```bash
python tools/fetch_matrices.py --list                                          # show the curated catalog
python tools/fetch_matrices.py --out data/raw                                  # download all Hermitian ones
python tools/fetch_matrices.py --out data/raw --names Bai/dw8192 --symmetrize # optionally symmetrize a non-symmetric one
python tools/fetch_matrices.py --out data/raw --names FEMLAB/waveguide3D --keep-general  # keep a non-Hermitian one as-is
```

## Included matrices

| File | n | nnz | Source | Type |
| --- | --- | --- | --- | --- |
| `Bai__mhdb416.npz` | 416 | 2,312 | Bai | MHD electromagnetics, real symmetric |
| `Bai__bfwb782.npz` | 782 | 5,982 | Bai | Electromagnetics, real symmetric, indefinite |
| `Bai__mhd4800b.npz` | 4,800 | 27,520 | Bai | MHD electromagnetics, real symmetric |
| `Oberwolfach__gyro.npz` | 17,361 | 1,021,159 | Oberwolfach | Gyroscopic / model reduction, real symmetric indefinite |
| `FEMLAB__waveguide3D.npz` | 21,036 | 303,468 | FEMLAB | Waveguide, complex **general (non-Hermitian)**, asymmetry 0.179 |
| `Li__pli.npz` | 22,695 | 1,350,309 | Li | Electromagnetics (edge-element / curl-curl), real symmetric indefinite |
| `CEMW__vfem.npz` | 93,476 | 1,434,636 | CEMW | Vector FEM electromagnetics, **complex128 Hermitian**, ~100k |
| `Um__2cubes_sphere.npz` | 101,492 | 1,647,264 | Um | Electromagnetic scattering, real symmetric, ~100k |

Skipped / not stored as Hermitian: `Bai/dw8192` (relative asymmetry 6e-3), `TOKAMAK/utm5940` (relative asymmetry 0.93, nonsymmetric formulation). Use `--symmetrize` only if a symmetrized variant is an acceptable test model, and document it.

## Solver run

`sparse-eigensolve data/raw/<file>.npz --output results/real_<file>.npz -k 6 --sigma <s>`, measured with `/usr/bin/time -l`.

| Matrix | n | Mode | sigma | Elapsed (s) | Max rel. residual | Peak RSS (GB) | Shifts |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Bai__mhdb416 | 416 | hermitian | 0.0 | 0.00 | 1.0e-20 | 0.06 | 1 |
| Bai__bfwb782 | 782 | hermitian | 0.0 | 0.01 | 2.7e-15 | 0.06 | 1 |
| Bai__mhd4800b | 4800 | hermitian | 0.0 | 0.03 | 5.0e-24 | 0.07 | 1 |
| Oberwolfach__gyro | 17361 | hermitian | 0.0 | 0.60 | 9.1e-18 | 0.24 | 1 |
| FEMLAB__waveguide3D | 21036 | general | 0.0 | 6.26 | 4.8e-16 | 0.30 | 1 |
| Li__pli | 22695 | hermitian | 0.0 | 52.1 | 1.5e-16 | 1.25 | 2 |
| CEMW__vfem | 93476 | hermitian (complex) | 0.1 | 467.9 | 2.5e-12 | 4.04 | 1 |
| Um__2cubes_sphere | 101492 | hermitian | 100.0 | 409.4 | 2.4e-24 | 4.37 | 1 |

> Values exported from `results/path_comparison.json` (single reproducible run via `python tools/run_public_suite.py`); timings/residuals vary slightly run-to-run. Peak RSS is from a separate `/usr/bin/time -l` run on the same matrix.

## Notes

- **Scale is dominated by LU fill-in, not `n`.** The 100k synthetic tridiagonal benchmarks run in ~0.3 s / 0.18 GB, but the real 100k matrices take ~400–440 s and ~4.0–4.4 GB. A 22,695-order curl-curl matrix (`Li__pli`) already takes 51.6 s / 1.25 GB. Do not extrapolate linearly to 100k.
- `Li__pli` needed a second shift (first factorization at sigma=0 failed), which exercises the retry path on real data.
- `CEMW__vfem` is complex128 Hermitian (imaginary part of the diagonal is 0) and confirms the Hermitian path works on complex data at ~100k. `Um__2cubes_sphere` is real symmetric with a wide diagonal scale (9.7e1 to 2.5e10), so its sigma must be chosen with unit awareness.
- `FEMLAB__waveguide3D` is the non-Hermitian control: on it the general (`eigs`) and forced-Hermitian (`eigsh`) paths return **different spectra**, which is why the matrix class must be confirmed before submission.
- These matrices are a starting point for scale and fill-in behavior; they are not the exact competition setup.
- License/attribution: matrices are redistributed by SuiteSparse under their original academic terms. Record the source collection and matrix id in any competition submission.
