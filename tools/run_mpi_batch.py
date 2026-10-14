#!/usr/bin/env python
"""Distribute independent sparse-eigenvalue jobs across MPI ranks.

This is an embarrassingly parallel batch backend for independent matrices or
target shifts. It does not claim to distribute one matrix factorization across
ranks; that requires a distributed eigensolver such as PETSc/SLEPc.

Job file format:

[
  {
    "matrix": "data/raw/A.npz",
    "output": "results/mpi_A.npz",
    "config": {"k": 6, "sigma": 0.0, "mode": "auto"}
  }
]
"""

from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parent.parent


def _load_solver():
    sys.path.insert(0, str(ROOT / "src"))
    from sparse_eigensolver.solver import SolverConfig, load_matrix, save_results, set_threads, solve

    return SolverConfig, load_matrix, save_results, set_threads, solve


def _config_from_job(config_data: dict[str, Any], matrix_path: Path, jobs_path: Path):
    SolverConfig, load_matrix, _, _, _ = _load_solver()
    names = {item.name for item in fields(SolverConfig)}
    values = {key: value for key, value in config_data.items() if key in names}
    if "shift_probes" in values:
        values["shift_probes"] = tuple(values["shift_probes"])
    if "target_band" in values and values["target_band"] is not None:
        values["target_band"] = tuple(values["target_band"])
    if "reorder" in values and values["reorder"] == "none":
        values["reorder"] = None
    mass_path = values.pop("mass_matrix", None)
    if mass_path:
        mass_path = Path(mass_path)
        if not mass_path.is_absolute():
            candidates = (
                ROOT / mass_path,
                jobs_path.parent / mass_path,
                matrix_path.parent / mass_path,
            )
            mass_path = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
        values["mass_matrix"] = load_matrix(mass_path, triangular=values.pop("mass_triangular", "none"))
    else:
        values.pop("mass_triangular", None)
    return SolverConfig(**values)


def _run_job(job: dict[str, Any], jobs_path: Path) -> dict[str, Any]:
    SolverConfig, load_matrix, save_results, set_threads, solve = _load_solver()
    matrix_path = Path(job["matrix"])
    output_path = Path(job["output"])
    if not matrix_path.is_absolute():
        matrix_path = ROOT / matrix_path
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    config_data = dict(job.get("config", {}))
    if "threads" in config_data:
        set_threads(int(config_data.pop("threads")))
    config = _config_from_job(config_data, matrix_path, jobs_path)
    started = time.perf_counter()
    try:
        values, vectors, residuals, diagnostics = solve(load_matrix(matrix_path, triangular=job.get("triangular", "none")), config)
        save_results(output_path, values, vectors, residuals, diagnostics)
        return {
            "matrix": str(matrix_path),
            "output": str(output_path),
            "status": "ok",
            "elapsed_seconds": time.perf_counter() - started,
            "max_relative_residual": float(residuals.max()),
            "problem_type": diagnostics["problem_type"],
            "resolved_mode": diagnostics["resolved_mode"],
            "effective_shift": diagnostics["effective_shift"],
        }
    except Exception as exc:  # noqa: BLE001 - batch records must continue across jobs
        return {
            "matrix": str(matrix_path),
            "output": str(output_path),
            "status": "error",
            "elapsed_seconds": time.perf_counter() - started,
            "error": f"{type(exc).__name__}: {exc}",
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=Path, required=True, help="JSON list of independent solve jobs")
    parser.add_argument("--output", type=Path, required=True, help="Rank-0 summary JSON")
    args = parser.parse_args()

    try:
        from mpi4py import MPI
    except ImportError as exc:
        raise SystemExit("run_mpi_batch.py requires mpi4py and an MPI launcher such as mpiexec") from exc

    communicator = MPI.COMM_WORLD
    rank = communicator.Get_rank()
    size = communicator.Get_size()
    jobs = None
    if rank == 0:
        payload = json.loads(args.jobs.read_text(encoding="utf-8-sig"))
        jobs = payload["jobs"] if isinstance(payload, dict) else payload
        if not isinstance(jobs, list) or not all(isinstance(job, dict) for job in jobs):
            raise ValueError("jobs JSON must be a list of objects or an object with a 'jobs' list")
    jobs = communicator.bcast(jobs, root=0)
    jobs_path = args.jobs if args.jobs.is_absolute() else (Path.cwd() / args.jobs).resolve()
    local_records = [_run_job(job, jobs_path) for job in jobs[rank::size]]
    gathered = communicator.gather(local_records, root=0)
    if rank == 0:
        records = [record for group in gathered for record in group]
        records.sort(key=lambda record: record["matrix"])
        report = {
            "backend": "mpi-independent-batch",
            "distributed_single_matrix": False,
            "mpi_size": size,
            "job_count": len(jobs),
            "records": records,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {args.output} with {len(records)} job records")


if __name__ == "__main__":
    main()
