#!/usr/bin/env python
"""Parallel-speedup benchmark (competition 指标 4-性能: 并行加速是否有效).

The solver's inner work is dominated by (a) sparse matrix-vector products inside
ARPACK and (b) the sparse LU / ILU factorization plus triangular solves. Both route
their dense kernels through the BLAS, so capping the BLAS/OpenMP thread count lets us
measure whether extra threads actually buy wall-clock time on this workload.

We run the same solve at 1, 2, 4, 8, ... threads and report:

* wall-clock time per thread count,
* speedup ``S(p) = T(1) / T(p)`` and parallel efficiency ``S(p) / p``,
* peak RSS (to show memory cost of the factorization is unaffected by threads).

Writes ``results/speedup.json`` and ``results/SPEEDUP.md``.

Note: on this 10-core / 16 GB machine the load average is chronically high (other
processes), so absolute timings are noisy. We take the best of ``--repeats`` runs and
report the median as well; treat small speedups as "BLAS-bound, memory-bandwidth
limited" rather than a hard limit of the method.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys
import time

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sparse_eigensolver.benchmark import _peak_rss_mb, make_matrix  # noqa: E402
from sparse_eigensolver.solver import SolverConfig, load_matrix, set_threads, solve  # noqa: E402

OUT_JSON = ROOT / "results" / "speedup.json"
OUT_MD = ROOT / "results" / "SPEEDUP.md"

DEFAULT_THREADS = (1, 2, 4, 8)


def _run_once(matrix, k: int, sigma: float, mode: str, inner: str, threads: int) -> dict:
    """Solve once with the BLAS/OpenMP pool capped at ``threads`` threads."""
    set_threads(threads)
    config = SolverConfig(k=k, sigma=sigma, mode=mode, inner_solver=inner)
    t = time.perf_counter()
    values, vectors, residuals, diagnostics = solve(matrix, config)
    elapsed = time.perf_counter() - t
    return {
        "threads": threads,
        "seconds": elapsed,
        "peak_rss_mb": _peak_rss_mb(),
        "max_residual": float(np.max(residuals)),
        "arpack_iterations": diagnostics["arpack_iterations"],
        "inner_matvecs": diagnostics["inner_matvecs"],
        "factor_nnz": diagnostics["factor_nnz"],
        "eigenvalues": [float(v.real) if np.isrealobj(v) else complex(v) for v in values],
    }


def benchmark(matrix, k: int, sigma: float, mode: str, inner: str, threads_seq, repeats: int) -> list[dict]:
    records = []
    for threads in threads_seq:
        runs = []
        for _ in range(repeats):
            runs.append(_run_once(matrix, k, sigma, mode, inner, threads))
        seconds = [r["seconds"] for r in runs]
        best = min(runs, key=lambda r: r["seconds"])
        best = dict(best)
        best["seconds"] = min(seconds)
        best["seconds_median"] = statistics.median(seconds)
        best["repeats"] = repeats
        records.append(best)
        print(
            f"[speedup] threads={threads:>2}  best={min(seconds):.3f}s  median={statistics.median(seconds):.3f}s  "
            f"resid={best['max_residual']:.2e}",
            flush=True,
        )
    return records


def _attach_speedup(records: list[dict]) -> None:
    base = records[0]["seconds"]
    for r in records:
        p = r["threads"]
        r["speedup"] = base / r["seconds"] if r["seconds"] > 0 else float("nan")
        r["speedup_median"] = base / r["seconds_median"] if r["seconds_median"] > 0 else float("nan")
        r["efficiency"] = r["speedup"] / p if p > 0 else float("nan")
        r["efficiency_median"] = r["speedup_median"] / p if p > 0 else float("nan")


def _fmt(x, unit="", nd=3):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    if isinstance(x, float):
        return f"{x:.{nd}g}{unit}"
    return f"{x}{unit}"


def write_markdown(meta: dict, records: list[dict]) -> str:
    lines = ["# 并行加速报告 (Parallel speedup)", ""]
    lines.append("衡量 BLAS/OpenMP 线程数对端到端求解墙钟时间的影响（`--threads` / `set_threads()`）。")
    lines.append("每个线程数取 `--repeats` 次运行，表中同时给出最好值与中位数，并报告加速比 `S(p)=T(1)/T(p)` 与并行效率 `S(p)/p`。")
    lines.append("")
    lines.append(f"- 矩阵：`{meta['matrix_label']}`  n={meta['n']}  nnz={meta['nnz']}")
    lines.append(f"- 求解：k={meta['k']}  sigma={meta['sigma']}  mode={meta['mode']}  inner={meta['inner_solver']}")
    lines.append(f"- 机器：{meta['cpus']} 逻辑核；线程序列 {list(meta['threads'])}；repeats={meta['repeats']}")
    lines.append("")
    lines.append("| 线程 p | 最好 T(p)/s | 中位 T(p)/s | 加速比 S(p) | 效率 S(p)/p | 峰值内存/MB | 残差 |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in records:
        lines.append(
            f"| {r['threads']} | {_fmt(r['seconds'])} | {_fmt(r['seconds_median'])} | "
            f"{_fmt(r['speedup'], 'x', 3)} | {_fmt(r['efficiency'], '', 3)} | "
            f"{_fmt(r['peak_rss_mb'], '', 4)} | {_fmt(r['max_residual'])} |"
        )
    lines.append("")
    peak = max(records, key=lambda r: r["speedup"])
    lines.append("## 结论")
    lines.append("")
    if peak["speedup"] <= 1.05:
        lines.append(
            "- **并行加速在本工作负载上基本无效**：最好成绩出现在 p=%d（S=%s），更多线程反而略慢。"
            % (peak["threads"], _fmt(peak["speedup"], "x", 3))
        )
        lines.append(
            "- 原因：shift-invert 的主成本是稀疏 LU 分解与稀疏三角回代，二者是**内存带宽受限**且高度串行"
            "（依赖链、超节点之间的同步），BLAS 多线程几乎帮不上忙；线程越多，同步/调度开销越大。"
        )
        lines.append(
            "- 这正面回答了「并行加速是否有效」：对**稀疏**特征值问题，答案是「基本无效」，加速应来自**算法**"
            "（重排/ILU 降填充、并行重排）而非线程数。线程旋钮的价值在于**可复现性**（固定 `--threads 1`）"
            "与避免超订（`--threads` 不超过物理核）。"
        )
    else:
        lines.append(
            "- 最好加速出现在 p=%d：S=%s，效率 %s。" % (peak["threads"], _fmt(peak["speedup"], "x", 3), _fmt(peak["efficiency"], "", 3))
        )
        lines.append("- 稀疏矩阵-向量乘与稀疏三角求解受内存带宽限制，多线程收益通常有限；密集 BLAS 部分才明显受益。")
    lines.append("- 峰值内存与线程数基本无关（分解填充主导），因此并行不改变内存预算结论。")
    lines.append("- 线程数通过 `--threads` 暴露给用户；`set_threads(0)` 表示不限制（交还 BLAS 自行决定）。")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure BLAS/OpenMP thread scaling of the eigensolver.")
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--matrix", type=Path, help="Path to a SciPy .npz sparse matrix (real data)")
    src.add_argument("--case", help="Synthetic benchmark case (see benchmark.make_matrix)")
    parser.add_argument("--n", type=int, default=40000, help="Synthetic dimension when --case is used")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("-k", type=int, default=6)
    parser.add_argument("--sigma", type=float, default=0.0)
    parser.add_argument("--mode", choices=("auto", "hermitian", "general"), default="auto")
    parser.add_argument("--inner", choices=("direct", "ilu"), default="direct")
    parser.add_argument("--threads", type=int, nargs="+", default=list(DEFAULT_THREADS))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=OUT_JSON)
    parser.add_argument("--markdown", type=Path, default=OUT_MD)
    args = parser.parse_args()

    if args.matrix:
        matrix = load_matrix(args.matrix)
        label = args.matrix.name
    else:
        case = args.case or "indefinite"
        matrix = make_matrix(args.n, case, args.seed)
        label = f"{case} (n={args.n}, seed={args.seed})"

    threads_seq = sorted({int(t) for t in args.threads if int(t) > 0}) or [1]
    records = benchmark(matrix, args.k, args.sigma, args.mode, args.inner, threads_seq, args.repeats)
    _attach_speedup(records)

    meta = {
        "matrix_label": label,
        "n": int(matrix.shape[0]),
        "nnz": int(matrix.nnz),
        "k": args.k,
        "sigma": args.sigma,
        "mode": args.mode,
        "inner_solver": args.inner,
        "cpus": __import__("os").cpu_count(),
        "threads": threads_seq,
        "repeats": args.repeats,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"meta": meta, "records": records}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(write_markdown(meta, records), encoding="utf-8")
    print(f"wrote {args.output} and {args.markdown}", flush=True)


if __name__ == "__main__":
    main()
