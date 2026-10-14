#!/usr/bin/env python
"""Run the electromagnetic fault-scenario suite and emit a machine-readable table.

Covers the scenarios required by the competition PDF (交付要求 3e):

1. ``near_shift``          -- target shift sits exactly on an eigenvalue (near-singular A - sigma I).
2. ``low_freq_breakdown``  -- a dense cluster of eigenvalues within ~1e-6 of zero (低频击穿).
3. ``repeated``            -- degenerate / repeated eigenvalues.
4. ``clustered``           -- tightly clustered spectrum.
5. ``memory_limited``      -- run a 3D 7-point Laplacian (n = 40^3) under a hard RSS budget:
                              exact LU exceeds the budget (killed) while ILU + GMRES stays inside it (内存受限).
6. real matrices           -- ``Li__pli`` (near-singular shift), ``Um__2cubes_sphere`` and
                              ``CEMW__vfem`` (large fill-in, ILU rescue).

Each (scenario x strategy) runs in its own subprocess so peak RSS is isolated and a slow or
memory-hungry case cannot take down the whole suite. Because macOS cannot set ``RLIMIT_AS``
(``setrlimit`` raises ``ValueError: current limit exceeds maximum limit`` for any finite value),
the memory budget is enforced by a **parent-side RSS monitor** that polls the child's resident
set size and kills it once it exceeds ``memory_budget_mb``. Output JSON feeds
results/FAULT_SCENARIOS.md.

Usage:
    python tools/run_fault_scenarios.py                 # run everything, write results/fault_scenarios.json
    python tools/run_fault_scenarios.py --names near_shift memory_limited
    python tools/run_fault_scenarios.py --timeout 300
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "results" / "fault_scenarios.json"
DATA = ROOT / "data" / "raw"


# Each scenario: a synthetic benchmark case or a real matrix, plus the strategy sweep to try.
# ``strategies`` entries are kwargs for SolverConfig (name -> value); ``memory_budget_mb`` caps
# the child's peak RSS (parent-enforced), and ``timeout`` overrides the per-strategy wall clock.
SCENARIOS: list[dict[str, Any]] = [
    {
        "name": "near_shift",
        "case": "near_shift",
        "n": 20000,
        "k": 6,
        "sigma": 0.0,
        "description": "目标位移正好落在特征值上 (A - sigma I 精确奇异)",
        "strategies": [
            {"label": "blind-retry", "config": {}},
            {"label": "adaptive-probes", "config": {"shift_probes": (1e-3, 1e-2, 0.1), "adaptive_shift": True}},
        ],
    },
    {
        "name": "low_freq_breakdown",
        "case": "low_freq_breakdown",
        "n": 20000,
        "k": 6,
        "sigma": 0.0,
        "description": "零附近密集谱 (低频击穿): 8 个特征值挤在 [-1e-6, 1e-6]",
        "strategies": [
            {"label": "plain", "config": {}},
            {"label": "target-band", "config": {"target_band": (0.0, 0.01), "adaptive_shift": True}},
        ],
    },
    {
        "name": "repeated",
        "case": "repeated",
        "n": 20000,
        "k": 6,
        "sigma": 0.0,
        "description": "重特征值 (简并谱)",
        "strategies": [
            {"label": "plain", "config": {}},
            {"label": "block-ncv", "config": {"ncv": 60}},
        ],
    },
    {
        "name": "clustered",
        "case": "clustered",
        "n": 20000,
        "k": 6,
        "sigma": 0.0,
        "description": "紧致谱簇",
        "strategies": [
            {"label": "plain", "config": {}},
        ],
    },
    {
        "name": "memory_limited",
        "n": 64000,
        "case": "memory_limited_3d",
        "k": 6,
        "sigma": -1.0,
        "description": "内存受限 (1GB 预算): 3D 7点 Laplacian A = L + I (n = 40^3 = 64000, nnz(A) ~4.4e5), 精确 LU 填充 ~1500x (~1.7GB) 超预算被终止, ILU+GMRES 降级成功 (~0.28GB)",
        "strategies": [
            {"label": "direct-LU (budget)", "config": {}, "memory_budget_mb": 1024, "timeout": 300},
            {"label": "ilu-gmres (budget)", "config": {"inner_solver": "ilu", "ilu_fill_factor": 20.0}, "memory_budget_mb": 1024, "timeout": 300},
        ],
    },
    {
        "name": "real_near_singular_Li__pli",
        "matrix": "Li__pli.npz",
        "k": 6,
        "sigma": 0.0,
        "description": "真实电磁矩阵 Li__pli, sigma=0 精确奇异",
        "strategies": [
            {"label": "adaptive-probes", "config": {"shift_probes": (0.05, 0.1), "adaptive_shift": True}},
        ],
    },
    {
        "name": "real_fill_Um__2cubes_sphere",
        "matrix": "Um__2cubes_sphere.npz",
        "k": 6,
        "sigma": 100.0,
        "description": "真实电磁散射矩阵 ~10 万阶, LU 填充 ~406M nnz (247x), ILU 在 600s 内未收敛 (超时)",
        "strategies": [
            {"label": "direct-RCM", "config": {"reorder": "rcm"}, "timeout": 600},
            {"label": "ilu-gmres", "config": {"inner_solver": "ilu", "ilu_fill_factor": 20.0, "reorder": "rcm"}, "timeout": 600},
        ],
    },
    {
        "name": "real_fill_CEMW__vfem",
        "matrix": "CEMW__vfem.npz",
        "k": 6,
        "sigma": 0.1,
        "description": "真实复厄密 FEM 矩阵 ~9 万阶, LU 填充 ~148x, ILU 降到 ~1.56x",
        "strategies": [
            {"label": "ilu-gmres", "config": {"inner_solver": "ilu", "reorder": "rcm"}, "timeout": 600},
        ],
    },
]


def _peak_rss_mb() -> float:
    from sparse_eigensolver.benchmark import _peak_rss_mb as peak_rss_mb

    return peak_rss_mb()


def _build_matrix(scenario: dict[str, Any]):
    from sparse_eigensolver.benchmark import make_matrix

    if "matrix" in scenario:
        from sparse_eigensolver.solver import load_matrix

        return load_matrix(DATA / scenario["matrix"])
    return make_matrix(scenario["n"], scenario["case"], scenario.get("seed", 2026))


def run_one(scenario: dict[str, Any], strategy: dict[str, Any]) -> dict[str, Any]:
    """Run a single (scenario, strategy) in this process and return its record."""
    from sparse_eigensolver.solver import SolverConfig, solve

    matrix = _build_matrix(scenario)
    config = SolverConfig(k=scenario["k"], sigma=scenario["sigma"], **strategy["config"])
    record: dict[str, Any] = {
        "scenario": scenario["name"],
        "description": scenario["description"],
        "strategy": strategy["label"],
        "n": int(matrix.shape[0]),
        "nnz": int(matrix.nnz),
        "sigma": scenario["sigma"],
        "memory_budget_mb": strategy.get("memory_budget_mb"),
    }
    started = time.perf_counter()
    try:
        values, vectors, residuals, diagnostics = solve(matrix, config)
        record.update({
            "status": "ok",
            "elapsed_seconds": time.perf_counter() - started,
            "peak_rss_mb": _peak_rss_mb(),
            "max_relative_residual": float(residuals.max()),
            "n_attempts": diagnostics["n_attempts"],
            "effective_shift": diagnostics["effective_shift"],
            "arpack_iterations": diagnostics["arpack_iterations"],
            "inner_matvecs": diagnostics["inner_matvecs"],
            "factor_kind": diagnostics["factor_kind"],
            "factor_nnz": diagnostics["factor_nnz"],
            "eigenvalues": [complex(v) if not np.isrealobj(v) else float(v) for v in values.tolist()],
        })
    except MemoryError as exc:
        record.update({
            "status": "memory_exceeded",
            "elapsed_seconds": time.perf_counter() - started,
            "peak_rss_mb": _peak_rss_mb(),
            "error": f"{type(exc).__name__}: {exc}" or "MemoryError",
        })
    except Exception as exc:  # noqa: BLE001 - record any failure for the table
        record.update({
            "status": "error",
            "elapsed_seconds": time.perf_counter() - started,
            "peak_rss_mb": _peak_rss_mb(),
            "error": f"{type(exc).__name__}: {exc}",
        })
    return record


def _child_rss_mb(pid: int) -> float:
    """Return a live child process RSS in MB on Windows, macOS, and Linux."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(ProcessMemoryCounters),
            wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL

        process_query_information = 0x0400
        process_vm_read = 0x0010
        handle = kernel32.OpenProcess(process_query_information | process_vm_read, False, pid)
        if not handle:
            return 0.0
        try:
            counters = ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            if psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
                return counters.WorkingSetSize / (1024 * 1024)
        finally:
            kernel32.CloseHandle(handle)
        return 0.0

    try:
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
        return int(out) / 1024.0 if out else 0.0
    except (ValueError, OSError):
        return 0.0


def _run_isolated(scenario: dict[str, Any], strategy: dict[str, Any], timeout: float) -> dict[str, Any]:
    """Run one (scenario, strategy) in a subprocess with a wall-clock and RSS-budget guard.

    macOS cannot set ``RLIMIT_AS``, so the memory budget is enforced by polling the child's
    RSS from the parent and killing the child once it crosses the budget.
    """
    payload = {"scenario": scenario, "strategy": strategy}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(payload, fh, default=str)
        payload_path = fh.name

    budget = strategy.get("memory_budget_mb")
    peak = {"mb": 0.0, "exceeded": False}
    proc = subprocess.Popen(
        [sys.executable, "-W", "ignore", __file__, "--run-one", payload_path],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )

    def monitor() -> None:
        while proc.poll() is None:
            rss = _child_rss_mb(proc.pid)
            if rss > peak["mb"]:
                peak["mb"] = rss
            if budget and rss > budget:
                peak["exceeded"] = True
                try:
                    proc.kill()
                except OSError:
                    pass
                return
            time.sleep(0.1)

    watcher = threading.Thread(target=monitor, daemon=True)
    watcher.start()
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate()
        watcher.join(timeout=1)
        os.unlink(payload_path)
        return {
            "scenario": scenario["name"], "description": scenario["description"],
            "strategy": strategy["label"], "status": "timeout",
            "timeout_seconds": timeout, "peak_rss_mb": peak["mb"],
            "memory_budget_mb": budget,
        }
    watcher.join(timeout=1)
    os.unlink(payload_path)

    if peak["exceeded"]:
        return {
            "scenario": scenario["name"], "description": scenario["description"],
            "strategy": strategy["label"], "status": "memory_exceeded",
            "peak_rss_mb": peak["mb"], "memory_budget_mb": budget,
            "error": f"peak RSS {peak['mb']:.0f} MB exceeded budget {budget} MB (killed)",
        }
    if proc.returncode != 0:
        return {
            "scenario": scenario["name"], "description": scenario["description"],
            "strategy": strategy["label"], "status": "error",
            "peak_rss_mb": peak["mb"], "memory_budget_mb": budget,
            "error": (stderr or stdout or "subprocess failed").strip().splitlines()[-1],
        }
    record = json.loads(stdout.strip().splitlines()[-1])
    # Trust the child's own ru_maxrss if the poll missed the peak.
    record["peak_rss_mb"] = max(record.get("peak_rss_mb", 0.0), peak["mb"])
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--names", nargs="*", help="Only run these scenario names")
    parser.add_argument("--timeout", type=float, default=600.0, help="Per (scenario, strategy) wall-clock timeout")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--merge", action="store_true", help="Merge into an existing results file")
    parser.add_argument("--run-one", help=argparse.SUPPRESS)  # internal: payload path
    args = parser.parse_args()

    if args.run_one:
        payload = json.loads(Path(args.run_one).read_text())
        record = run_one(payload["scenario"], payload["strategy"])
        print(json.dumps(record, default=str))
        return

    selected = [s for s in SCENARIOS if not args.names or s["name"] in args.names]
    records: list[dict[str, Any]] = []
    for scenario in selected:
        print(f"\n=== {scenario['name']} :: {scenario['description']} ===", flush=True)
        for strategy in scenario["strategies"]:
            timeout = float(strategy.get("timeout", args.timeout))
            rec = _run_isolated(scenario, strategy, timeout)
            records.append(rec)
            _print_record(rec)
            _flush(records, args)

    print(f"\nwrote {args.out} ({len(records)} records)")


def _flush(records: list[dict[str, Any]], args: argparse.Namespace) -> None:
    if args.merge and args.out.exists():
        existing = json.loads(args.out.read_text())
        merged: dict[str, Any] = {(r["scenario"], r["strategy"]): r for r in existing}
        for r in records:
            merged[(r["scenario"], r["strategy"])] = r
        records_to_write = list(merged.values())
    else:
        records_to_write = records
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(records_to_write, indent=2, default=str) + "\n", encoding="utf-8")


def _fmt(value: Any, unit: str = "") -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if value != value:
            return "nan"
        return f"{value:.3g}{unit}"
    return f"{value}{unit}"


def _print_record(rec: dict[str, Any]) -> None:
    status = rec.get("status")
    line = f"  [{rec['strategy']:<18}] {status:<16}"
    if status == "ok":
        line += (
            f" resid={_fmt(rec['max_relative_residual']):<10} "
            f"t={_fmt(rec['elapsed_seconds'], 's'):<9} "
            f"rss={_fmt(rec['peak_rss_mb'], 'MB'):<10} "
            f"it={rec['arpack_iterations']:<4} factor={rec['factor_kind']}({rec['factor_nnz']}) "
            f"attempts={rec['n_attempts']}"
        )
    else:
        line += f" rss={_fmt(rec.get('peak_rss_mb'), 'MB')} {rec.get('error', rec.get('timeout_seconds', ''))}"
    print(line, flush=True)


if __name__ == "__main__":
    main()
