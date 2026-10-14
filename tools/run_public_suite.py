#!/usr/bin/env python3
"""Run every downloaded matrix through the solver and summarise both code paths.

Produces results/path_comparison.json plus a printed table. Intended for the
competition write-up: it records, per matrix, the detected class, the resolved
solver path, wall time, residual and whether complex eigenvalues appeared.
"""

from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
from scipy import sparse

from sparse_eigensolver.solver import SolverConfig, classify_matrix, solve

# (matrix stem, target shift). Shifts are chosen from each matrix's diagonal scale.
CASES: list[tuple[str, complex]] = [
    ("Bai__mhdb416", 0.0),
    ("Bai__bfwb782", 0.0),
    ("Bai__mhd4800b", 0.0),
    ("Oberwolfach__gyro", 0.0),
    ("FEMLAB__waveguide3D", 0.0),
    ("Li__pli", 0.0),
    ("CEMW__vfem", 0.1),
    ("Um__2cubes_sphere", 100.0),
]


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    raw = repo / "data" / "raw"
    results = repo / "results"
    results.mkdir(parents=True, exist_ok=True)

    runs: list[dict[str, object]] = []
    for name, sigma in CASES:
        path = raw / f"{name}.npz"
        if not path.exists():
            print(f"skip {name}: {path} not found")
            continue
        matrix = sparse.load_npz(path)
        matrix_class = classify_matrix(matrix)
        start = time.perf_counter()
        values, _, residuals, diagnostics = solve(matrix, SolverConfig(k=6, sigma=sigma))
        values = np.asarray(values)
        runs.append(
            {
                "matrix": name,
                "n": int(matrix.shape[0]),
                "nnz": int(matrix.nnz),
                "dtype": str(matrix.dtype),
                "matrix_class": matrix_class,
                "resolved_mode": diagnostics["resolved_mode"],
                "sigma": sigma if not isinstance(sigma, complex) else [sigma.real, sigma.imag],
                "elapsed_seconds": float(diagnostics["elapsed_seconds"]),
                "wall_seconds": time.perf_counter() - start,
                "max_relative_residual": float(diagnostics["max_relative_residual"]),
                "shifts_attempted": len(diagnostics["attempts"]),
                "complex_eigenvalues": bool(np.abs(values.imag).max() > 1e-9) if np.iscomplexobj(values) else False,
            }
        )

    (results / "path_comparison.json").write_text(
        json.dumps({"runs": runs}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    header = f"{'matrix':24s}{'n':>8}{'class':>10}{'mode':>10}{'t(s)':>9}{'resid':>10}{'cx':>4}"
    print(header)
    print("-" * len(header))
    for run in runs:
        print(
            f"{run['matrix']:24s}{run['n']:>8}{run['matrix_class']:>10}{run['resolved_mode']:>10}"
            f"{run['elapsed_seconds']:>9.2f}{run['max_relative_residual']:>10.1e}"
            f"{'Y' if run['complex_eigenvalues'] else 'n':>4}"
        )


if __name__ == "__main__":
    main()
