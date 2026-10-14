#!/usr/bin/env python3
"""Compare shift-invert strategies (ordering / reorder / inner solver) on real matrices.

Two sub-steps run per (matrix, strategy), each in its own isolated subprocess with a
wall-clock timeout, so pathological cases are recorded as "timeout" instead of
blocking the suite:

  * factor  - factor (A - sigma I) once and report the nnz of the L/U factors
              (sparse LU for "direct", ILU for "ilu"). This is cheap relative to a
              full eigensolve, so fill-in data survives even when the solve is slow.
  * solve   - run the actual shift-invert ARPACK eigensolve and verify residuals.

Because the two steps are independent, a catastrophic-fill matrix still yields the
factorization numbers. Writes results/strategy_comparison.json.

Usage:
    python tools/benchmark_strategies.py                          # default matrices
    python tools/benchmark_strategies.py --names Oberwolfach__gyro Bai__mhd4800b
    python tools/benchmark_strategies.py --steps factor           # skip the solves
    python tools/benchmark_strategies.py --timeout 300            # per-step seconds
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spilu, splu

from sparse_eigensolver.solver import (
    SolverConfig,
    _apply_permutation,
    _rcm_permutation,
    solve,
)

# (stem, shift). Shifts kept non-singular and near the spectrum.
DEFAULT_CASES: list[tuple[str, complex]] = [
    ("Bai__mhd4800b", 0.0),
    ("Oberwolfach__gyro", 0.0),
    ("FEMLAB__waveguide3D", 0.0),
    ("Li__pli", 0.0),
    ("CEMW__vfem", 0.1),
    ("Um__2cubes_sphere", 100.0),
]

STRATEGIES: dict[str, SolverConfig] = {
    "direct-COLAMD": SolverConfig(k=6, permc_spec=None, reorder=None, inner_solver="direct"),
    "direct-NATURAL": SolverConfig(k=6, permc_spec="NATURAL", reorder=None, inner_solver="direct"),
    "direct-MMD_AT_PLUS_A": SolverConfig(k=6, permc_spec="MMD_AT_PLUS_A", reorder=None, inner_solver="direct"),
    "direct-MMD_ATA": SolverConfig(k=6, permc_spec="MMD_ATA", reorder=None, inner_solver="direct"),
    "direct-RCM": SolverConfig(k=6, permc_spec=None, reorder="rcm", inner_solver="direct"),
    "ilu-GMRES": SolverConfig(k=6, permc_spec=None, reorder=None, inner_solver="ilu"),
    "ilu-GMRES-RCM": SolverConfig(k=6, permc_spec=None, reorder="rcm", inner_solver="ilu"),
    # Looser ILU: fill_factor=20 rescues the "exactly singular" drop-tolerance
    # failures seen with the default fill_factor=10 on harder complex matrices.
    "ilu-GMRES-ff20": SolverConfig(k=6, permc_spec=None, reorder=None, inner_solver="ilu", ilu_fill_factor=20.0),
    "ilu-GMRES-ff20-RCM": SolverConfig(k=6, permc_spec=None, reorder="rcm", inner_solver="ilu", ilu_fill_factor=20.0),
    # Toughest case (Li__pli): needs both a higher fill_factor and a tighter drop_tol.
    "ilu-GMRES-ff20-tol6": SolverConfig(k=6, permc_spec=None, reorder=None, inner_solver="ilu",
                                        ilu_fill_factor=20.0, ilu_drop_tol=1e-6),
}

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"


def _c(value: complex) -> object:
    value = complex(value)
    return float(value.real) if value.imag == 0 else [float(value.real), float(value.imag)]


def _shifted(matrix: sparse.spmatrix, sigma: complex, reorder: str | None) -> sparse.csc_matrix:
    work = _apply_permutation(matrix, _rcm_permutation(matrix))[0] if reorder == "rcm" else matrix
    dtype = np.result_type(work.dtype, np.asarray(sigma).dtype)
    return (work.astype(dtype) - sigma * sparse.identity(work.shape[0], dtype=dtype)).tocsc()


def factor_stats(matrix: sparse.csr_matrix, config: SolverConfig, sigma: complex) -> dict[str, object]:
    """Factor (A - sigma I) once and report fill-in without running ARPACK."""
    shifted = _shifted(matrix, sigma, config.reorder)
    kind = "ILU" if config.inner_solver == "ilu" else "LU"
    start = time.perf_counter()
    try:
        if config.inner_solver == "ilu":
            factor = spilu(
                shifted,
                drop_tol=config.ilu_drop_tol,
                fill_factor=config.ilu_fill_factor,
            )
            nnz = int(factor.L.nnz + factor.U.nnz)
        else:
            factor = splu(shifted, permc_spec=config.permc_spec or "COLAMD")
            nnz = int(factor.nnz)
    except RuntimeError as exc:  # singular or structurally rank-deficient at this shift
        return {"kind": kind, "factor_nnz": -1, "fill_ratio": -1.0, "factor_seconds": -1.0, "error": str(exc)}
    return {
        "kind": kind,
        "factor_nnz": nnz,
        "fill_ratio": float(nnz / max(matrix.nnz, 1)),
        "factor_seconds": float(time.perf_counter() - start),
    }


def solve_stats(name: str, strategy: str, sigma: complex) -> dict[str, object]:
    """Run the real eigensolve and report convergence/residual diagnostics."""
    matrix = sparse.load_npz(RAW / f"{name}.npz")
    config = replace(STRATEGIES[strategy], sigma=sigma)
    record: dict[str, object] = {"matrix": name, "strategy": strategy, "sigma": _c(sigma)}
    start = time.perf_counter()
    try:
        values, _, residuals, diagnostics = solve(matrix, config)
        record.update(
            status="ok",
            wall_seconds=float(time.perf_counter() - start),
            solver_seconds=float(diagnostics["elapsed_seconds"]),
            max_relative_residual=float(residuals.max()),
            resolved_mode=diagnostics["resolved_mode"],
            shifts_attempted=len(diagnostics["attempts"]),
            eigenvalues=[_c(v) for v in values.tolist()],
        )
    except Exception as exc:  # noqa: BLE001 - report, keep going
        record.update(status="error", error=str(exc), wall_seconds=float(time.perf_counter() - start))
    return record


def _run_step(step: str, name: str, strategy: str, sigma: complex) -> dict[str, object]:
    matrix = sparse.load_npz(RAW / f"{name}.npz")
    if step == "factor":
        stats = factor_stats(matrix, replace(STRATEGIES[strategy], sigma=sigma), sigma)
        return {"matrix": name, "strategy": strategy, "sigma": _c(sigma), **stats}
    return solve_stats(name, strategy, sigma)


def _fmt(step: str, record: dict[str, object]) -> str:
    if step == "factor":
        if record.get("factor_nnz", -1) > 0:
            return (f"  factor {record['strategy']:22s} {record['kind']:3s} "
                    f"nnz={record['factor_nnz']:>10d} fill={record['fill_ratio']:6.2f}x "
                    f"t={record['factor_seconds']:7.2f}s")
        return f"  factor {record['strategy']:22s} FAILED {record.get('error', '')[:60]}"
    line = f"  solve  {record['strategy']:22s} {record.get('status','?'):7s} wall={record['wall_seconds']:8.2f}s"
    if record.get("status") == "ok":
        line += f" resid={record['max_relative_residual']:.1e} shifts={record['shifts_attempted']}"
    elif record.get("error"):
        line += f" ({record['error'][:60]})"
    return line


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--names", nargs="*", help="Matrix stems in data/raw; default is the built-in list")
    parser.add_argument("--strategies", nargs="*", default=list(STRATEGIES), choices=list(STRATEGIES))
    parser.add_argument("--steps", nargs="*", default=["factor", "solve"], choices=["factor", "solve"])
    parser.add_argument("--timeout", type=float, default=300.0, help="Per-step wall-clock limit in seconds")
    parser.add_argument("--shift", action="append", default=[], metavar="NAME=VALUE",
                        help="Override the shift for a matrix, e.g. --shift Li__pli=0.1 (repeatable)")
    parser.add_argument("--merge", action="store_true",
                        help="Merge into an existing --out file instead of overwriting it")
    parser.add_argument("--out", type=Path, default=Path("results/strategy_comparison.json"))
    parser.add_argument("--run-step", nargs=4, metavar=("STEP", "NAME", "STRATEGY", "SIGMA"), help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.run_step:
        step, name, strategy, sigma_text = args.run_step
        sigma: complex = complex(sigma_text)
        sigma = sigma.real if sigma.imag == 0 else sigma
        print(json.dumps(_run_step(step, name, strategy, sigma)))
        return

    cases = (
        [(name, dict(DEFAULT_CASES).get(name, 0.0)) for name in args.names]
        if args.names
        else DEFAULT_CASES
    )
    overrides = dict(pair.split("=", 1) for pair in args.shift)
    cases = [(name, complex(overrides.get(name, sigma))) for name, sigma in cases]

    results: dict[str, dict[str, list[dict[str, object]]]] = {}
    if args.merge and args.out.exists():
        results = json.loads(args.out.read_text(encoding="utf-8"))
    for name, sigma in cases:
        if not (RAW / f"{name}.npz").exists():
            print(f"skip {name}: data/raw/{name}.npz not found")
            continue
        matrix = sparse.load_npz(RAW / f"{name}.npz")
        print(f"\n=== {name}  n={matrix.shape[0]} nnz={matrix.nnz} dtype={matrix.dtype} sigma={sigma} ===", flush=True)
        results.setdefault(name, {})
        for strategy in args.strategies:
            print(f" [{strategy}]", flush=True)
            for step in args.steps:
                command = [sys.executable, __file__, "--run-step", step, name, strategy, str(sigma)]
                start = time.perf_counter()
                try:
                    completed = subprocess.run(command, capture_output=True, text=True, timeout=args.timeout)
                    payload = completed.stdout.strip().splitlines()[-1] if completed.stdout.strip() else ""
                    record = json.loads(payload) if payload else {"status": "error",
                                                                  "error": completed.stderr.strip()[-300:]}
                except subprocess.TimeoutExpired:
                    record = {"status": "timeout", "factor_nnz": -1, "fill_ratio": -1.0,
                              "wall_seconds": float(time.perf_counter() - start),
                              "error": f"exceeded {args.timeout:g}s per-step limit"}
                record.setdefault("matrix", name)
                record.setdefault("strategy", strategy)
                results.setdefault(name, {}).setdefault(strategy, []).append(record)
                print(_fmt(step, record), flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    total = sum(len(v) for m in results.values() for v in m.values())
    print(f"\nwrote {args.out} ({len(results)} matrices, {total} records)")


if __name__ == "__main__":
    main()
