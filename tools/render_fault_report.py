#!/usr/bin/env python
"""Render ``results/fault_scenarios.json`` into the 故障场景表 (``results/FAULT_SCENARIOS.md``).

The competition (交付要求 3e) asks for the failure scenarios we handle and how we handle
them. ``tools/run_fault_scenarios.py`` produces the machine-readable records; this script
turns them into a human-readable table with, per scenario: **精度 / 性能 / 迭代次数 /
处置建议** (accuracy / performance / iteration counts / mitigation).

Usage:
    python tools/render_fault_report.py                    # reads results/fault_scenarios.json
    python tools/render_fault_report.py --in results/fault_scenarios.json --out results/FAULT_SCENARIOS.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IN = ROOT / "results" / "fault_scenarios.json"
DEFAULT_OUT = ROOT / "results" / "FAULT_SCENARIOS.md"

# Human-readable title + the mitigation we recommend, per scenario name.
SCENARIO_INFO: dict[str, dict[str, str]] = {
    "near_shift": {
        "title": "目标位移落在特征值上 (A − σI 精确奇异)",
        "mitigation": "探测位移 `--shift-probes` + `--adaptive-shift`；不要用盲重试的小步长。",
    },
    "low_freq_breakdown": {
        "title": "低频击穿：零附近密集谱",
        "mitigation": "已知目标频段时用 `--target-band`，候选位移从频段中心向外走，避开奇异中心。",
    },
    "repeated": {
        "title": "重特征值 (简并谱)",
        "mitigation": "加大子空间维数 `--ncv`（块 Krylov），避免 ARPACK 只收敛到简并子空间的一部分。",
    },
    "clustered": {
        "title": "紧致谱簇",
        "mitigation": "收紧 `--residual-tol` 并适度加大 `--ncv`；必要时提高 `--maxiter`。",
    },
    "memory_limited": {
        "title": "内存受限 (硬 RSS 预算)",
        "mitigation": "精确 LU 超预算时改用 `--inner-solver ilu`（+ `--reorder rcm`）降级；内存预算由父进程 RSS 监控强制。",
    },
    "real_near_singular_Li__pli": {
        "title": "真实矩阵 σ=0 精确奇异 (Li__pli)",
        "mitigation": "`--shift-probes` 给出离开奇点的绝对偏移 + `--adaptive-shift`。",
    },
    "real_fill_Um__2cubes_sphere": {
        "title": "真实散射矩阵填充爆炸 (Um__2cubes_sphere)",
        "mitigation": "精确 LU 填充 ~250x；改 `--inner-solver ilu --ilu-fill-factor 20` 降到 ~19x。",
    },
    "real_fill_CEMW__vfem": {
        "title": "真实复厄密 FEM 矩阵填充爆炸 (CEMW__vfem)",
        "mitigation": "精确 LU 填充 ~148x；改 `--inner-solver ilu` 降到 ~1.6x。",
    },
}

STATUS_LABEL = {
    "ok": "✅ ok",
    "timeout": "⏱ timeout",
    "memory_exceeded": "⛔ memory",
    "error": "✗ error",
}


def _fmt(value: Any, unit: str = "", digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        if value != value:
            return "nan"
        if value == 0:
            return f"0{unit}"
        if abs(value) < 1e-3 or abs(value) >= 1e5:
            return f"{value:.{digits}e}{unit}"
        return f"{value:.{digits}g}{unit}"
    return f"{value}{unit}"


def _row(rec: dict[str, Any]) -> str:
    status = rec.get("status", "?")
    label = STATUS_LABEL.get(status, status)
    resid = _fmt(rec.get("max_relative_residual"))
    seconds = _fmt(rec.get("elapsed_seconds"), "s")
    rss = _fmt(rec.get("peak_rss_mb"), " MB")
    budget = rec.get("memory_budget_mb")
    rss_cell = rss if not budget else f"{rss} / {budget} MB"
    iters = _fmt(rec.get("arpack_iterations"))
    inner = _fmt(rec.get("inner_matvecs"))
    factor = rec.get("factor_kind") or "—"
    nnz = rec.get("factor_nnz")
    factor_cell = f"{factor} ({nnz:,})" if nnz else factor
    note = rec.get("error") or (f"timeout {rec.get('timeout_seconds')}s" if status == "timeout" else "—")
    return (
        f"| {rec.get('strategy', '?')} | {label} | {resid} | {seconds} | {rss_cell} | "
        f"{iters} | {inner} | {factor_cell} | {note} |"
    )


def write_markdown(records: list[dict[str, Any]]) -> str:
    by_scenario: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        by_scenario.setdefault(rec["scenario"], []).append(rec)

    lines: list[str] = []
    lines.append("# 故障场景表 / Fault-scenario report")
    lines.append("")
    lines.append("赛题交付要求 3e 的失败场景与处置。数据由 `tools/run_fault_scenarios.py` 在**独立子进程**中实测，")
    lines.append("带 wall-clock 超时与**父进程 RSS 内存预算**（macOS 无法设 `RLIMIT_AS`，故轮询子进程 RSS 超限即终止）。")
    lines.append("每格含义：**精度**=相对残差 `‖Av−λv‖/((‖A‖+|λ|)‖v‖)`；**性能**=墙钟时间；**迭代次数**=ARPACK 外层迭代 / 内层 matvec；")
    lines.append("**处置建议**见每节末。原始记录：[`fault_scenarios.json`](fault_scenarios.json)。")
    lines.append("")

    for name, recs in by_scenario.items():
        info = SCENARIO_INFO.get(name, {})
        title = info.get("title", name)
        desc = recs[0].get("description", "")
        lines.append(f"## {name} — {title}")
        lines.append("")
        if desc:
            lines.append(f"*{desc}*")
            lines.append("")
        lines.append("| 策略 | 状态 | 最大残差 | 时间 | 峰值 RSS | Arnoldi 迭代 | 内层 matvec | 因子 (nnz) | 备注 |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for rec in recs:
            lines.append(_row(rec))
        lines.append("")
        mitigation = info.get("mitigation")
        if mitigation:
            lines.append(f"**处置建议**：{mitigation}")
            lines.append("")

    lines.append("## 图例")
    lines.append("")
    lines.append("- ✅ ok：在给定预算/超时内收敛，残差达标。")
    lines.append("- ⏱ timeout：超过 wall-clock 超时（多为精确 LU 填充过大导致分解过慢）。")
    lines.append("- ⛔ memory：峰值 RSS 超过内存预算，被父进程监控终止。")
    lines.append("- ✗ error：求解器抛错（例如 ILU 丢元产生零主元 `Factor is exactly singular`）。")
    lines.append("")
    lines.append("> 说明：`memory_limited` 的预算与超时可按机器调整（`SCENARIOS` 中的 `memory_budget_mb` / `timeout`）。")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="src", type=Path, default=DEFAULT_IN, help="Input JSON (from run_fault_scenarios.py)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output markdown path")
    args = parser.parse_args()

    if not args.src.exists():
        raise SystemExit(f"missing {args.src}; run `python tools/run_fault_scenarios.py` first")
    records = json.loads(args.src.read_text(encoding="utf-8"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(write_markdown(records), encoding="utf-8")
    print(f"wrote {args.out} ({len(records)} records, {len({r['scenario'] for r in records})} scenarios)")


if __name__ == "__main__":
    main()
