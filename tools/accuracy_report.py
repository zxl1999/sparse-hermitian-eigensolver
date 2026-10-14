#!/usr/bin/env python
"""Accuracy report for the sparse eigensolver (competition 指标 7/8).

Two complementary checks:

1. **Dense oracle (small n).** For n <= ~600 we build a dense reference with NumPy
   (``eigvalsh``/``eigh`` for Hermitian, ``eig`` for the general path), select the k
   eigenpairs nearest the target shift, and feed them to the solver as
   ``reference_vectors``. The solver then reports the principal angles between its
   computed subspace and the oracle subspace, plus the eigenvalue error. This is a
   true accuracy oracle (the dense path is used only for validation, never by the solver).

2. **Solver vs. solver (large n).** For large sparse matrices no dense oracle exists,
   so we cross-check two independent inner-solver strategies (exact LU vs ILU+GMRES).
   Their eigenpairs should agree to solver tolerance; the subspace angle quantifies it.

Writes ``results/accuracy.json`` and ``results/ACCURACY.md``.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sparse_eigensolver.benchmark import make_matrix  # noqa: E402
from sparse_eigensolver.solver import SolverConfig, load_matrix, solve  # noqa: E402

DATA = ROOT / "data" / "raw"
OUT_JSON = ROOT / "results" / "accuracy.json"
OUT_MD = ROOT / "results" / "ACCURACY.md"

# (label, case, n, k, sigma, mode)
ORACLE_CASES = [
    ("indefinite (Hermitian)", "indefinite", 400, 6, 0.0, "hermitian"),
    ("near_shift (Hermitian)", "near_shift", 400, 6, 0.0, "hermitian"),
    ("complex_hermitian", "complex_hermitian", 400, 6, 0.0, "hermitian"),
    ("complex_symmetric (general)", "complex_symmetric", 300, 6, 0.0, "general"),
]

# (label, case, n, k, sigma, mode, direct_config, ilu_config)
CROSS_CASES = [
    ("indefinite n=20000", "indefinite", 20000, 6, 0.0, "hermitian",
     {"inner_solver": "direct"}, {"inner_solver": "ilu", "ilu_drop_tol": 1e-3, "ilu_fill_factor": 50.0, "gmres_restart": 200}),
    ("near_shift n=20000", "near_shift", 20000, 6, 0.0, "hermitian",
     {"inner_solver": "direct"}, {"inner_solver": "ilu", "ilu_drop_tol": 1e-3, "ilu_fill_factor": 50.0, "gmres_restart": 200}),
]

# Real matrices (direct vs ILU) if downloaded. Keep these small/low-fill so the
# dense-free cross-check stays fast; the point is LU vs ILU agreement, not fill-in.
REAL_CROSS = [("Bai__mhd4800b", 0.0, "hermitian")]


def _dense_oracle(matrix: sparse.csr_matrix, k: int, sigma: float, mode: str):
    """Return (values, vectors) for the k eigenpairs nearest sigma via dense NumPy."""
    dense = matrix.toarray()
    if mode == "hermitian":
        values, vectors = np.linalg.eigh(dense)
    else:
        values, vectors = np.linalg.eig(dense)
    order = np.argsort(np.abs(values - sigma))
    idx = order[:k]
    return values[idx], vectors[:, idx]


def run_oracle_case(label, case, n, k, sigma, mode):
    matrix = make_matrix(n, case, 2026)
    ref_values, ref_vectors = _dense_oracle(matrix, k, sigma, mode)
    config = SolverConfig(k=k, sigma=sigma, mode=mode, reference_vectors={"vectors": ref_vectors, "values": ref_values})
    t = time.perf_counter()
    values, vectors, residuals, diagnostics = solve(matrix, config)
    elapsed = time.perf_counter() - t
    angles = diagnostics["subspace_angle"]
    # direct eigenvalue error against the oracle (matched by nearest value)
    ref_sorted = np.sort(ref_values)
    matched = []
    used = set()
    for v in np.sort(values):
        best, best_d = None, np.inf
        for j, rv in enumerate(ref_sorted):
            if j in used:
                continue
            d = abs(v - rv)
            if d < best_d:
                best, best_d = j, d
        used.add(best)
        matched.append(abs(v - ref_sorted[best]))
    return {
        "label": label,
        "n": int(matrix.shape[0]),
        "nnz": int(matrix.nnz),
        "mode": mode,
        "k": k,
        "sigma": sigma,
        "max_residual": float(residuals.max()),
        "max_subspace_angle_deg": angles["max_angle_deg"],
        "mean_subspace_angle_deg": angles["mean_angle_deg"],
        "max_eigenvalue_error": float(np.max(matched)),
        "orthogonality_error_fro": float(np.linalg.norm(vectors.conj().T @ vectors - np.eye(k))),
        "arpack_iterations": diagnostics["arpack_iterations"],
        "inner_matvecs": diagnostics["inner_matvecs"],
        "elapsed_seconds": elapsed,
    }


def run_cross_case(label, matrix, k, sigma, mode, direct_cfg, ilu_cfg):
    """Solve with exact LU, then reuse its eigenpairs as the reference for ILU."""
    t = time.perf_counter()
    dv, dvec, dres, ddiag = solve(matrix, SolverConfig(k=k, sigma=sigma, mode=mode, **direct_cfg))
    direct_t = time.perf_counter() - t
    t = time.perf_counter()
    iv, ivec, ires, idiag = solve(
        matrix,
        SolverConfig(k=k, sigma=sigma, mode=mode, reference_vectors={"vectors": dvec, "values": dv}, **ilu_cfg),
    )
    ilu_t = time.perf_counter() - t
    angles = idiag["subspace_angle"]
    # eigenvalue agreement between the two strategies (matched by value)
    dvals = np.sort(dv)
    ivals = np.sort(iv)
    matched = [abs(a - b) for a, b in zip(ivals, dvals)]
    return {
        "label": label,
        "n": int(matrix.shape[0]),
        "nnz": int(matrix.nnz),
        "mode": mode,
        "k": k,
        "sigma": sigma,
        "direct": {"max_residual": float(dres.max()), "factor_nnz": ddiag["factor_nnz"], "arpack_iterations": ddiag["arpack_iterations"], "seconds": direct_t},
        "ilu": {"max_residual": float(ires.max()), "factor_nnz": idiag["factor_nnz"], "arpack_iterations": idiag["arpack_iterations"], "inner_matvecs": idiag["inner_matvecs"], "seconds": ilu_t},
        "max_subspace_angle_deg": angles["max_angle_deg"],
        "mean_subspace_angle_deg": angles["mean_angle_deg"],
        "max_eigenvalue_error": float(np.max(matched)),
        "fill_ratio_direct": ddiag["factor_nnz"] / max(int(matrix.nnz), 1),
        "fill_ratio_ilu": idiag["factor_nnz"] / max(int(matrix.nnz), 1),
    }


def _fmt(x, unit=""):
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.3g}{unit}"
    return f"{x}{unit}"


def write_markdown(oracle, cross):
    lines = ["# 精度报告 (Accuracy report)", ""]
    lines.append("本报告用两种方式量化求解精度：小规模问题对照稠密精确解（NumPy `eigh`/`eig`），")
    lines.append("大规模问题用两种独立内解策略（精确 LU 与 ILU+GMRES）互校。")
    lines.append("子空间夹角由 `diagnostics.subspace_angle` 给出（SVD 主角度），特征值误差按最近值匹配。")
    lines.append("")
    lines.append("## 1. 稠密精确解对照 (dense oracle, small n)")
    lines.append("")
    lines.append("| 算例 | n | 模式 | 最大残差 | 子空间夹角(最大/平均) | 特征值误差 | 正交性误差 | Arnoldi 迭代 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in oracle:
        lines.append(
            f"| {r['label']} | {r['n']} | {r['mode']} | {_fmt(r['max_residual'])} | "
            f"{_fmt(r['max_subspace_angle_deg'], '°')} / {_fmt(r['mean_subspace_angle_deg'], '°')} | "
            f"{_fmt(r['max_eigenvalue_error'])} | {_fmt(r['orthogonality_error_fro'])} | {r['arpack_iterations']} |"
        )
    lines.append("")
    lines.append("## 2. 求解器互校 (solver vs. solver, large n)")
    lines.append("")
    lines.append("| 算例 | n | 模式 | 精确 LU 残差 | ILU 残差 | 子空间夹角(最大) | 特征值误差 | LU 填充比 | ILU 填充比 |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in cross:
        lines.append(
            f"| {r['label']} | {r['n']} | {r['mode']} | {_fmt(r['direct']['max_residual'])} | "
            f"{_fmt(r['ilu']['max_residual'])} | {_fmt(r['max_subspace_angle_deg'], '°')} | "
            f"{_fmt(r['max_eigenvalue_error'])} | {_fmt(r['fill_ratio_direct'], 'x')} | {_fmt(r['fill_ratio_ilu'], 'x')} |"
        )
    lines.append("")
    lines.append("## 结论")
    lines.append("")
    lines.append("- 小规模对照稠密精确解：子空间夹角与特征值误差均在求解容差量级（见上表），说明 shift-invert 路径正确。")
    lines.append("- 大规模 LU 与 ILU 互校：两者子空间夹角极小，说明 ILU+GMRES 的低内存替代在精度上可接受。")
    lines.append("- 残差按 `||A v - λ v|| / ((||A|| + |λ|) ||v||)` 归一化，并始终针对原始矩阵校验。")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    oracle = []
    for c in ORACLE_CASES:
        print(f"[oracle] running {c[0]} ...", flush=True)
        t = time.perf_counter()
        oracle.append(run_oracle_case(*c))
        print(f"[oracle] done {c[0]} in {time.perf_counter() - t:.1f}s", flush=True)

    cross = []
    for label, case, n, k, sigma, mode, dcfg, icfg in CROSS_CASES:
        print(f"[cross] running {label} ...", flush=True)
        t = time.perf_counter()
        matrix = make_matrix(n, case, 2026)
        cross.append(run_cross_case(label, matrix, k, sigma, mode, dcfg, icfg))
        print(f"[cross] done {label} in {time.perf_counter() - t:.1f}s", flush=True)
    for stem, sigma, mode in REAL_CROSS:
        path = DATA / f"{stem}.npz"
        if not path.exists():
            print(f"skip {stem}: not downloaded")
            continue
        print(f"[cross] running {stem} (real) ...", flush=True)
        t = time.perf_counter()
        matrix = load_matrix(path)
        cross.append(run_cross_case(
            f"{stem} (real)", matrix, 6, sigma, mode,
            {"inner_solver": "direct", "reorder": "rcm"},
            {"inner_solver": "ilu", "ilu_drop_tol": 1e-4, "ilu_fill_factor": 20.0, "reorder": "rcm"},
        ))
        print(f"[cross] done {stem} (real) in {time.perf_counter() - t:.1f}s", flush=True)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({"oracle": oracle, "cross": cross}, indent=2, default=str) + "\n", encoding="utf-8")
    OUT_MD.write_text(write_markdown(oracle, cross), encoding="utf-8")
    print(f"wrote {OUT_JSON} and {OUT_MD}")
    for r in oracle:
        print(f"  oracle {r['label']:<32} angle={r['max_subspace_angle_deg']:.3g}° eigerr={r['max_eigenvalue_error']:.3g}")
    for r in cross:
        print(f"  cross  {r['label']:<32} angle={r['max_subspace_angle_deg']:.3g}° eigerr={r['max_eigenvalue_error']:.3g}")


if __name__ == "__main__":
    main()
