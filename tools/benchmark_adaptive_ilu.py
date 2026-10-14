from __future__ import annotations

import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sparse_eigensolver.benchmark import make_matrix  # noqa: E402
from sparse_eigensolver.solver import SolverConfig, solve  # noqa: E402

OUT_JSON = ROOT / "results" / "adaptive_ilu.json"
OUT_MD = ROOT / "results" / "ADAPTIVE_ILU.md"
CASES = (("clustered", 400, 0.0), ("complex_symmetric", 300, 0.0), ("memory_limited", 900, 0.0))


def run(case: str, n: int, sigma: float, adaptive: bool) -> dict:
    matrix = make_matrix(n, case, 2026)
    config = SolverConfig(
        k=4,
        sigma=sigma,
        mode="auto",
        reorder="rcm",
        inner_solver="ilu",
        ilu_adaptive=adaptive,
        ilu_memory_budget_mb=256.0 if adaptive else None,
        ilu_fill_factor=20.0,
        ilu_drop_tol=1e-4,
        gmres_rtol=1e-8,
        residual_tol=1e-5,
        ilu_probe_maxiter=3,
        ilu_probe_time_budget_s=5.0,
    )
    started = time.perf_counter()
    try:
        _, _, residuals, diagnostics = solve(matrix, config)
        plan = diagnostics.get("ilu_plan")
        return {
            "case": case,
            "n": n,
            "adaptive": adaptive,
            "status": "ok",
            "wall_seconds": time.perf_counter() - started,
            "max_residual": float(residuals.max()),
            "factor_nnz": diagnostics.get("factor_nnz"),
            "selected": plan.get("selected") if plan else None,
            "plan_seconds": plan.get("elapsed_seconds") if plan else 0.0,
            "trace_status": [record["status"] for record in plan.get("trace", [])] if plan else [],
        }
    except Exception as exc:  # noqa: BLE001 - preserve benchmark failure details
        return {
            "case": case,
            "n": n,
            "adaptive": adaptive,
            "status": "error",
            "wall_seconds": time.perf_counter() - started,
            "error": f"{type(exc).__name__}: {exc}",
        }


def render(records: list[dict]) -> str:
    lines = [
        "# Adaptive ILU benchmark",
        "",
        "Deterministic offline benchmark for bounded-probe ILU parameter selection.",
        "The adaptive path minimizes predicted inner work under a 256 MB approximate factor budget.",
        "",
        "| Case | n | Path | Status | Residual | Factor nnz | Plan seconds | Selected rung |",
        "|---|---:|---|---|---:|---:|---:|---|",
    ]
    for record in records:
        selected = record.get("selected") or {}
        rung = f"ff={selected.get('fill_factor')}, tol={selected.get('drop_tol')}" if selected else "—"
        lines.append(
            f"| {record['case']} | {record['n']} | {'adaptive' if record['adaptive'] else 'fixed'} | "
            f"{record['status']} | {record.get('max_residual', '—')} | {record.get('factor_nnz', '—')} | "
            f"{record.get('plan_seconds', 0.0):.3f} | {rung} |"
        )
    lines += [
        "",
        "The JSON artifact records the full rung trace, including skipped, unconverged, and selected candidates.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    records = []
    for case, n, sigma in CASES:
        records.append(run(case, n, sigma, adaptive=False))
        records.append(run(case, n, sigma, adaptive=True))
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    OUT_MD.write_text(render(records), encoding="utf-8")
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    main()
