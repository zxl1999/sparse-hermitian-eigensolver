from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, gmres, spilu


DEFAULT_ILU_LADDER = (
    (1.0, 1e-2),
    (2.0, 1e-3),
    (5.0, 1e-4),
    (10.0, 1e-4),
    (20.0, 1e-5),
    (50.0, 1e-6),
)


class ProbeTimeout(RuntimeError):
    pass


@dataclass
class ILUPlan:
    factor: Any
    selected: dict[str, Any]
    trace: list[dict[str, Any]]
    elapsed_seconds: float


def memory_estimate_mb(nnz: int, dtype: np.dtype) -> float:
    value_bytes = np.dtype(dtype).itemsize
    index_bytes = np.dtype(np.int64).itemsize
    return float(nnz * (value_bytes + 2 * index_bytes) * 1.25 / (1024 * 1024))


def _probe(shifted: sparse.csc_matrix, ilu: Any, restart: int, maxiter: int, rtol: float, time_budget_s: float, dtype: np.dtype) -> tuple[int, float]:
    rng = np.random.default_rng(20261002)
    rhs = rng.standard_normal(shifted.shape[0])
    if np.issubdtype(np.dtype(dtype), np.complexfloating):
        rhs = rhs + 1j * rng.standard_normal(shifted.shape[0])
    rhs = np.asarray(rhs, dtype=dtype)
    rhs /= max(float(np.linalg.norm(rhs)), 1.0)
    deadline = time.perf_counter() + time_budget_s
    calls = 0

    def matvec(vector):
        nonlocal calls
        calls += 1
        if time.perf_counter() > deadline:
            raise ProbeTimeout("ILU probe exceeded its time budget")
        return shifted @ vector

    operator = LinearOperator(shifted.shape, matvec=matvec, dtype=dtype)
    preconditioner = LinearOperator(shifted.shape, matvec=ilu.solve, dtype=dtype)
    started = time.perf_counter()
    _, info = gmres(operator, rhs, M=preconditioner, restart=min(restart, 100), maxiter=maxiter, rtol=rtol, atol=0.0)
    elapsed = time.perf_counter() - started
    if info != 0:
        raise RuntimeError(f"ILU probe did not converge (info={info})")
    return calls, elapsed


def plan_ilu_parameters(
    matrix: sparse.csr_matrix,
    shift: complex,
    *,
    mass_matrix: sparse.csr_matrix | None = None,
    memory_budget_mb: float | None,
    permc_spec: str | None,
    gmres_restart: int,
    gmres_rtol: float,
    probe_maxiter: int,
    probe_time_budget_s: float,
) -> ILUPlan:
    dtype = np.result_type(matrix.dtype, np.asarray(shift).dtype)
    if mass_matrix is None:
        mass = sparse.identity(matrix.shape[0], dtype=dtype, format="csr")
    else:
        dtype = np.result_type(dtype, mass_matrix.dtype)
        mass = mass_matrix.astype(dtype)
    shifted = (matrix.astype(dtype) - shift * mass).tocsc()
    trace: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    started = time.perf_counter()
    for fill_factor, drop_tol in DEFAULT_ILU_LADDER:
        predicted_nnz = int(np.ceil(fill_factor * max(matrix.nnz, 1)))
        record: dict[str, Any] = {"fill_factor": fill_factor, "drop_tol": drop_tol, "predicted_nnz": predicted_nnz, "predicted_memory_mb": memory_estimate_mb(predicted_nnz, dtype)}
        if memory_budget_mb is not None and record["predicted_memory_mb"] > memory_budget_mb:
            record["status"] = "skipped_memory_budget"
            trace.append(record)
            continue
        try:
            ilu = spilu(shifted, drop_tol=drop_tol, fill_factor=fill_factor, permc_spec=permc_spec or "COLAMD")
            factor_nnz = int(ilu.L.nnz + ilu.U.nnz)
            actual_mb = memory_estimate_mb(factor_nnz, dtype)
            record.update({"factor_nnz": factor_nnz, "factor_memory_mb": actual_mb})
            if memory_budget_mb is not None and actual_mb > memory_budget_mb:
                record["status"] = "skipped_memory_budget_actual"
                trace.append(record)
                continue
            probe_iters, probe_seconds = _probe(shifted, ilu, gmres_restart, probe_maxiter, gmres_rtol, probe_time_budget_s, dtype)
            work = float((matrix.nnz + factor_nnz) * probe_iters)
            record.update({"status": "ok", "gmres_iters": probe_iters, "probe_seconds": probe_seconds, "predicted_work": work})
            trace.append(record)
            if selected is None or work < selected["predicted_work"]:
                selected = {"factor": ilu, "fill_factor": fill_factor, "drop_tol": drop_tol, "factor_nnz": factor_nnz, "factor_memory_mb": actual_mb, "gmres_iters": probe_iters, "predicted_work": work}
        except (ProbeTimeout, RuntimeError, ValueError) as exc:
            record.update({"status": "unconverged", "error": str(exc)})
            trace.append(record)
    if selected is None:
        raise RuntimeError(f"adaptive ILU found no feasible rung; trace={trace}")
    chosen = {key: value for key, value in selected.items() if key != "factor"}
    return ILUPlan(selected=chosen, factor=selected["factor"], trace=trace, elapsed_seconds=time.perf_counter() - started)
