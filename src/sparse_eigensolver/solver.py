from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import os
import time
from typing import Any

import numpy as np
from scipy import sparse
from scipy.io import mmread
from scipy.sparse.csgraph import reverse_cuthill_mckee
from scipy.sparse.linalg import (
    ArpackNoConvergence,
    LinearOperator,
    eigs,
    eigsh,
    gmres,
    spilu,
    splu,
)

from .adaptive import ILUPlan, plan_ilu_parameters

PERMC_SPECS = (None, "NATURAL", "MMD_AT_PLUS_A", "MMD_ATA", "COLAMD")
REORDERS = (None, "rcm")
INNER_SOLVERS = ("direct", "ilu")


def set_threads(n: int) -> None:
    """Pin BLAS/OpenMP thread counts (must be set before the numeric libraries spin up).

    Uses :func:`threadpoolctl.threadpool_limits` when available (the reliable way to
    cap an already-imported OpenBLAS/MKL), and also exports the usual environment
    variables for child processes. ``n <= 0`` means "no limit / let the BLAS decide".
    """
    if n <= 0:
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ.pop(var, None)
        return
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = str(n)
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(n)
    except Exception:  # noqa: BLE001 - optional dependency; env vars are the fallback
        pass


@dataclass(frozen=True)
class SolverConfig:
    k: int = 6
    sigma: float | complex = 0.0
    tol: float = 1e-10
    maxiter: int | None = None
    ncv: int | None = None
    retries: int = 3
    shift_step: float | None = None
    residual_tol: float = 1e-8
    mode: str = "auto"  # "auto" | "hermitian" | "general"
    backend: str = "cpu"  # "cpu" | "gpu"
    hermitian_tol: float = 1e-12  # relative ||A - A^H|| threshold for classification
    # --- fill-in / factorization controls (shift-invert inner solve) ---
    permc_spec: str | None = None  # SuperLU column ordering: None (=COLAMD) | NATURAL | MMD_AT_PLUS_A | MMD_ATA | COLAMD
    reorder: str | None = None  # None | "rcm" (reverse Cuthill-McKee bandwidth reduction)
    inner_solver: str = "direct"  # "direct" (sparse LU) | "ilu" (incomplete LU + GMRES)
    ilu_drop_tol: float = 1e-4
    ilu_fill_factor: float = 10.0
    gmres_restart: int = 100
    gmres_rtol: float = 1e-8
    ilu_adaptive: bool = False
    ilu_memory_budget_mb: float | None = None
    ilu_probe_maxiter: int = 3
    ilu_probe_time_budget_s: float = 10.0
    # --- adaptive shift / target-band controls ---
    # Ordered probe shifts (absolute offsets from sigma) tried before the blind +/- shift_step
    # retries. Useful when the target band is known: e.g. (1e-3, 1e-2, 1e-1, 0.5, 1.0).
    shift_probes: tuple[float, ...] = ()
    # Known target frequency band [lo, hi] (absolute). When set, candidate shifts walk from
    # the band centre toward the edges so a near-singular centre does not abort the solve.
    target_band: tuple[float, float] | None = None
    # Adaptive shift: use a geometric backoff of the shift step (step, 2*step, 5*step, ...)
    # so a near-singular region is escaped decisively rather than by one tiny step at a time.
    adaptive_shift: bool = False
    # Optional mass matrix for the generalized problem A x = lambda B x.
    mass_matrix: Any = None
    # Optional reference subspace used only to report the subspace angle (accuracy metric).
    reference_vectors: Any = None


def _json_scalar(value: Any) -> Any:
    """Make a possibly-complex shift JSON serializable (real part if purely real)."""
    if isinstance(value, complex):
        return float(value.real) if value.imag == 0 else [float(value.real), float(value.imag)]
    return float(value)


def _restore_triangular(matrix: sparse.csr_matrix, mode: str) -> sparse.csr_matrix:
    """Recover a full matrix from a stored triangle.

    The competition brief requires reading triangular matrices and restoring the
    full operator on demand. ``mode`` is ``"auto"``, ``"none"``, ``"hermitian"`` or
    ``"symmetric"``.

    A stored triangle cannot, by itself, tell Hermitian from complex-symmetric
    storage (both may store the same lower triangle). The one reliable signal is the
    diagonal: a Hermitian matrix has a *real* diagonal, whereas a complex-symmetric
    (lossy/PML) matrix generally has a complex one. ``auto`` therefore mirrors with
    conjugation (Hermitian) when the matrix is real or its stored diagonal is real,
    and mirrors without conjugation (symmetric) when the diagonal carries an
    imaginary part. Force ``hermitian``/``symmetric`` when the convention is known.
    """
    matrix = sparse.csr_matrix(matrix)
    if mode == "none":
        return matrix
    if mode not in ("auto", "hermitian", "symmetric"):
        raise ValueError("triangular must be one of: auto, none, hermitian, symmetric")
    if mode == "auto":
        if not np.iscomplexobj(matrix.data):
            mode = "symmetric"  # real: conjugation is a no-op either way
        else:
            diag = matrix.diagonal()
            has_imag_diag = bool(np.any(np.abs(np.asarray(diag).imag) > 0))
            mode = "symmetric" if has_imag_diag else "hermitian"
    if mode == "hermitian":
        return (sparse.tril(matrix, k=0) + sparse.tril(matrix, k=-1).getH()).tocsr()
    # symmetric: mirror the stored triangle without conjugation
    return (sparse.tril(matrix, k=0) + sparse.tril(matrix, k=-1).T).tocsr()


def load_matrix(path: str | Path, triangular: str = "none") -> sparse.csr_matrix:
    """Load .npy dense arrays or .npz sparse matrices and store them as CSR.

    ``triangular`` controls recovery of a full matrix from a stored triangle
    (``"none"`` keeps the input as-is; ``"auto"``/``"hermitian"``/``"symmetric"``
    restore it — see :func:`_restore_triangular`).
    """
    path = Path(path)
    if path.suffix == ".npy":
        matrix = np.load(path, allow_pickle=False)
        if matrix.ndim != 2:
            raise ValueError(".npy input must contain a two-dimensional matrix")
        matrix = sparse.csr_matrix(matrix)
    elif path.suffix == ".npz":
        try:
            matrix = sparse.load_npz(path).tocsr()
        except (ValueError, KeyError) as exc:
            raise ValueError(".npz input must be a SciPy sparse matrix saved by save_npz") from exc
    elif path.suffix.lower() in ('.mtx', '.mm') or ''.join(path.suffixes).lower().endswith(('.mtx.gz', '.mm.gz')):
        try:
            matrix = sparse.csr_matrix(mmread(path))
        except (OSError, ValueError) as exc:
            raise ValueError('Matrix Market input could not be read') from exc
    else:
        raise ValueError('input must be .npy, SciPy .npz, or Matrix Market .mtx/.mm')
    if triangular != "none":
        matrix = _restore_triangular(matrix, triangular)
    return validate_matrix(matrix)


def _relative_asymmetry(matrix: sparse.csr_matrix) -> float:
    """Relative Hermitian asymmetry ||A - A^H|| / ||A||."""
    delta = matrix - matrix.getH()
    scale = max(float(sparse.linalg.norm(matrix)), 1.0)
    return 0.0 if delta.nnz == 0 else float(sparse.linalg.norm(delta)) / scale


def classify_matrix(matrix: Any, hermitian_tol: float = 1e-12) -> str:
    """Return "hermitian" or "general" based on relative asymmetry."""
    matrix = sparse.csr_matrix(np.asarray(matrix)) if not sparse.issparse(matrix) else matrix.tocsr()
    return "hermitian" if _relative_asymmetry(matrix) <= hermitian_tol else "general"


def validate_matrix(matrix: Any) -> sparse.csr_matrix:
    if not sparse.issparse(matrix):
        matrix = sparse.csr_matrix(np.asarray(matrix))
    matrix = matrix.tocsr()
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")
    if matrix.shape[0] < 2:
        raise ValueError("matrix dimension must be at least 2")
    if not np.issubdtype(matrix.dtype, np.number):
        raise ValueError("matrix values must be numeric")
    if not np.all(np.isfinite(matrix.data)):
        raise ValueError("matrix values must be finite")
    return matrix


def _enforce_hermitian(matrix: sparse.csr_matrix, hermitian_tol: float) -> sparse.csr_matrix:
    scale = max(float(sparse.linalg.norm(matrix)), 1.0)
    delta = matrix - matrix.getH()
    if delta.nnz and float(sparse.linalg.norm(delta)) > hermitian_tol * scale:
        raise ValueError(f"matrix must be Hermitian within relative tolerance {hermitian_tol:g}")
    return ((matrix + matrix.getH()) * 0.5).tocsr()


def _residuals(
    matrix: sparse.csr_matrix,
    values: np.ndarray,
    vectors: np.ndarray,
    mass_matrix: sparse.csr_matrix | None = None,
) -> np.ndarray:
    av = matrix @ vectors
    rhs = vectors if mass_matrix is None else mass_matrix @ vectors
    numer = np.linalg.norm(av - rhs * values[np.newaxis, :], axis=0)
    norm_a = float(sparse.linalg.norm(matrix))
    norm_b = 1.0 if mass_matrix is None else float(sparse.linalg.norm(mass_matrix))
    denom = (norm_a + np.abs(values) * norm_b) * np.linalg.norm(vectors, axis=0)
    return numer / np.maximum(denom, np.finfo(float).tiny)


def _subspace_angle(
    vectors: np.ndarray,
    reference: np.ndarray,
    mode: str,
    values: np.ndarray | None = None,
    reference_values: np.ndarray | None = None,
) -> dict[str, Any]:
    """Principal angles (degrees) between two subspaces, plus an eigenvalue error.

    For the Hermitian path the two subspaces are matched by *value* (eigenvalue);
    for the general path they are matched by *direction* (largest overlap).
    """
    def _orthonormalize(block: np.ndarray) -> np.ndarray:
        q, _ = np.linalg.qr(block)
        return q

    ref = np.asarray(reference)
    if ref.shape[0] != vectors.shape[0]:
        raise ValueError("reference_vectors must have the same number of rows as the matrix")
    qv = _orthonormalize(vectors)
    qr = _orthonormalize(ref)
    overlap = qv.conj().T @ qr
    singular_values = np.linalg.svd(overlap, compute_uv=False)
    singular_values = np.clip(singular_values, 0.0, 1.0)
    angles = np.degrees(np.arccos(singular_values))
    result: dict[str, Any] = {
        "max_angle_deg": float(angles.max()),
        "mean_angle_deg": float(angles.mean()),
        "angles_deg": [float(a) for a in angles],
        "match": mode,
    }
    if values is not None and reference_values is not None:
        if mode == "value":
            order = np.argsort(np.abs(np.asarray(reference_values)))
            ref_sorted = np.asarray(reference_values)[order]
            vals = np.asarray(values)
            used = set()
            matched = []
            for v in vals:
                best, best_d = None, np.inf
                for j, rv in enumerate(ref_sorted):
                    if j in used:
                        continue
                    d = abs(v - rv)
                    if d < best_d:
                        best, best_d = j, d
                used.add(best)
                matched.append(abs(v - ref_sorted[best]))
            result["max_eigenvalue_error"] = float(np.max(matched)) if matched else 0.0
        else:
            result["max_eigenvalue_error"] = float(
                np.min(np.abs(np.asarray(values)[:, None] - np.asarray(reference_values)[None, :]))
            ) if values.size and np.asarray(reference_values).size else 0.0
    return result


def _apply_permutation(matrix: sparse.csr_matrix, perm: np.ndarray) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Return (P A P^T, inv_perm) where (P x)[i] = x[perm[i]]."""
    perm = np.asarray(perm)
    inv_perm = np.empty_like(perm)
    inv_perm[perm] = np.arange(perm.size)
    return matrix[perm][:, perm].tocsr(), inv_perm


def _rcm_permutation(matrix: sparse.csr_matrix) -> np.ndarray:
    """Reverse Cuthill-McKee ordering of the (symmetrised) sparsity pattern."""
    pattern = matrix.copy()
    pattern.data = np.ones_like(pattern.data, dtype=np.int8)
    pattern = (pattern + pattern.T).tocsr()
    pattern.data[:] = 1
    return reverse_cuthill_mckee(pattern, symmetric_mode=True)


def _candidate_offsets(config: SolverConfig, step: float) -> list[float]:
    """Ordered offsets (from sigma) to try: sigma, target-band walk, probes, then backoff.

    ``sigma`` is always tried first so an exactly-solvable requested shift is used as-is.
    When ``target_band`` is given, candidates walk from the band centre outward so a
    near-singular centre does not abort the solve. Explicit ``shift_probes`` are appended,
    then the retry offsets: a plain +/- ``step`` ladder, or (with ``adaptive_shift``) a
    geometric backoff that escapes a near-singular region faster. All values are
    de-duplicated while preserving order.
    """
    offsets: list[float] = [0.0]

    def add(value: float) -> None:
        if value not in offsets:
            offsets.append(value)

    if config.target_band is not None:
        lo, hi = float(config.target_band[0]), float(config.target_band[1])
        if hi < lo:
            lo, hi = hi, lo
        centre = 0.5 * (lo + hi)
        half = 0.5 * (hi - lo)
        # centre first (relative to sigma), then successively closer to the two edges.
        add(centre - config.sigma)
        for frac in (0.25, 0.5, 0.75, 1.0):
            add(centre - frac * half - config.sigma)
            add(centre + frac * half - config.sigma)
    for probe in config.shift_probes:
        add(float(probe))
    if config.adaptive_shift:
        for factor in (1, 2, 5, 10, 20, 50, 100):
            for sign in (1, -1):
                add(sign * factor * step)
    else:
        for attempt in range(1, config.retries + 1):
            add(((attempt + 1) // 2) * step * (1 if attempt % 2 else -1))
    return offsets


def _build_inner_solver(
    matrix: sparse.csr_matrix,
    config: SolverConfig,
    shift: complex,
    mass_matrix: sparse.csr_matrix | None = None,
) -> tuple[LinearOperator, str, dict[str, Any]]:
    """Build the shift-invert inner operator (A - shift B)^-1 with the chosen factorization.

    Returns the operator, a human-readable description, and a mutable ``counts`` dict
    used to report iteration counts: ``arpack_applies`` counts how many times the
    outer eigensolver applies the shift-invert operator (one per Arnoldi/Lanczos
    iteration in shift-invert mode), and ``inner_matvecs`` counts applications of
    ``(A - shift I)`` (one per LU solve, or one per GMRES iteration for ILU).
    """
    dtype = np.result_type(matrix.dtype, np.asarray(shift).dtype)
    if mass_matrix is None:
        mass = sparse.identity(matrix.shape[0], dtype=dtype, format="csr")
    else:
        dtype = np.result_type(dtype, mass_matrix.dtype)
        mass = mass_matrix.astype(dtype)
    shifted = (matrix.astype(dtype) - shift * mass).tocsc()
    counts: dict[str, Any] = {"arpack_applies": 0, "inner_matvecs": 0}

    ilu_plan: ILUPlan | None = None
    if config.inner_solver == "ilu":
        if config.ilu_adaptive:
            ilu_plan = plan_ilu_parameters(
                matrix,
                shift,
                mass_matrix=mass_matrix,
                memory_budget_mb=config.ilu_memory_budget_mb,
                permc_spec=config.permc_spec,
                gmres_restart=config.gmres_restart,
                gmres_rtol=config.gmres_rtol,
                probe_maxiter=config.ilu_probe_maxiter,
                probe_time_budget_s=config.ilu_probe_time_budget_s,
            )
            ilu = ilu_plan.factor
            drop_tol = ilu_plan.selected["drop_tol"]
            fill_factor = ilu_plan.selected["fill_factor"]
        else:
            ilu = spilu(
                shifted,
                drop_tol=config.ilu_drop_tol,
                fill_factor=config.ilu_fill_factor,
                permc_spec=config.permc_spec or "COLAMD",
            )
            drop_tol = config.ilu_drop_tol
            fill_factor = config.ilu_fill_factor
        preconditioner = LinearOperator(matrix.shape, matvec=ilu.solve, dtype=dtype)
        factor_nnz = int(ilu.L.nnz + ilu.U.nnz)

        def _shifted_matvec(x: np.ndarray) -> np.ndarray:
            counts["inner_matvecs"] += 1
            return shifted @ x

        shifted_op = LinearOperator(matrix.shape, matvec=_shifted_matvec, dtype=dtype)

        def solve(x: np.ndarray) -> np.ndarray:
            counts["arpack_applies"] += 1
            solution, info = gmres(
                shifted_op, x, M=preconditioner, restart=config.gmres_restart,
                rtol=config.gmres_rtol, atol=0.0,
            )
            if info != 0:
                raise RuntimeError(f"inner GMRES failed to converge (info={info})")
            return solution

        operator = LinearOperator(matrix.shape, matvec=solve, dtype=dtype)
        description = (
            f"ILU(drop_tol={drop_tol:g}, fill_factor={fill_factor:g}, "
            f"permc_spec={config.permc_spec or 'COLAMD'}) + GMRES(restart={config.gmres_restart}, rtol={config.gmres_rtol:g})"
        )
        counts["factor_nnz"] = factor_nnz
        counts["factor_kind"] = "ILU"
        counts["ilu_adaptive"] = bool(config.ilu_adaptive)
        if ilu_plan is not None:
            counts["ilu_plan"] = {
                "selected": ilu_plan.selected,
                "trace": ilu_plan.trace,
                "elapsed_seconds": ilu_plan.elapsed_seconds,
            }
        return operator, description, counts

    lu = splu(shifted, permc_spec=config.permc_spec or "COLAMD")
    factor_nnz = int(lu.L.nnz + lu.U.nnz)

    def _lu_solve(x: np.ndarray) -> np.ndarray:
        counts["arpack_applies"] += 1
        counts["inner_matvecs"] += 1
        return lu.solve(x)

    operator = LinearOperator(matrix.shape, matvec=_lu_solve, dtype=dtype)
    counts["factor_nnz"] = factor_nnz
    counts["factor_kind"] = "LU"
    return operator, f"sparse LU (permc_spec={config.permc_spec or 'COLAMD'})", counts


def _solve_once(
    matrix: sparse.csr_matrix,
    config: SolverConfig,
    shift: complex,
    solver: Any,
    inner: LinearOperator | None = None,
    mass_matrix: sparse.csr_matrix | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Run one ARPACK solve and return (values, vectors, arpack_diagnostics)."""
    if inner is None:
        values, vectors = solver(
            matrix,
            k=config.k,
            M=mass_matrix,
            sigma=shift,
            which="LM",
            tol=config.tol,
            maxiter=config.maxiter,
            ncv=config.ncv,
            return_eigenvectors=True,
        )
        return values, vectors, {}
    if mass_matrix is not None:
        values, vectors = solver(
            matrix,
            k=config.k,
            M=mass_matrix,
            sigma=shift,
            OPinv=inner,
            which="LM",
            tol=config.tol,
            maxiter=config.maxiter,
            ncv=config.ncv,
            return_eigenvectors=True,
        )
        return values, vectors, {}

    # Standard shift-invert: ARPACK solves the inverse operator
    # (A - shift I)^-1, so its eigenvalues are mu = 1 / (lambda - shift).
    mu, vectors = solver(
        inner,
        k=config.k,
        which="LM",
        tol=config.tol,
        maxiter=config.maxiter,
        ncv=config.ncv,
        return_eigenvectors=True,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        values = shift + 1.0 / mu
    return values, vectors, {}


def solve(matrix: Any, config: SolverConfig = SolverConfig()) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    if config.backend == 'gpu':
        from .gpu import solve_gpu

        return solve_gpu(matrix, config)
    if config.backend != 'cpu':
        raise ValueError("backend must be 'cpu' or 'gpu'")
    raw = validate_matrix(matrix)
    raw_mass = None if config.mass_matrix is None else validate_matrix(config.mass_matrix)
    n = raw.shape[0]
    if config.mode not in ("auto", "hermitian", "general"):
        raise ValueError("mode must be 'auto', 'hermitian', or 'general'")
    if config.k < 1 or config.k >= n:
        raise ValueError(f"k must satisfy 1 <= k < matrix dimension ({n})")
    if raw_mass is not None and raw_mass.shape != raw.shape:
        raise ValueError(f"mass_matrix must have shape {raw.shape}, got {raw_mass.shape}")
    if config.tol <= 0 or config.residual_tol <= 0 or config.retries < 0:
        raise ValueError("tol and residual_tol must be positive; retries cannot be negative")
    if config.maxiter is not None and config.maxiter < 1:
        raise ValueError("maxiter must be positive when specified")
    if config.shift_step is not None and config.shift_step <= 0:
        raise ValueError("shift_step must be positive when specified")
    if config.ncv is not None and not (config.k < config.ncv <= n):
        raise ValueError("ncv must satisfy k < ncv <= matrix dimension")
    if config.permc_spec not in PERMC_SPECS:
        raise ValueError(f"permc_spec must be one of {PERMC_SPECS}")
    if config.reorder not in REORDERS:
        raise ValueError(f"reorder must be one of {REORDERS}")
    if config.inner_solver not in INNER_SOLVERS:
        raise ValueError(f"inner_solver must be one of {INNER_SOLVERS}")
    if config.ilu_drop_tol <= 0 or config.ilu_fill_factor <= 0:
        raise ValueError("ILU drop tolerance and fill factor must be positive")
    if config.gmres_restart < 1 or config.gmres_rtol <= 0:
        raise ValueError("GMRES restart must be positive and rtol must be positive")
    if config.ilu_memory_budget_mb is not None and config.ilu_memory_budget_mb <= 0:
        raise ValueError("ILU memory budget must be positive when specified")
    if config.ilu_probe_maxiter < 1 or config.ilu_probe_time_budget_s <= 0:
        raise ValueError("ILU probe limits must be positive")
    if config.ilu_adaptive and config.inner_solver != "ilu":
        raise ValueError("ilu_adaptive requires inner_solver='ilu'")
    if config.target_band is not None:
        lo, hi = config.target_band
        if not (np.isfinite(lo) and np.isfinite(hi)):
            raise ValueError("target_band bounds must be finite")

    detected = classify_matrix(raw, config.hermitian_tol)
    resolved = detected if config.mode == "auto" else config.mode
    if resolved == "hermitian":
        matrix = _enforce_hermitian(raw, max(config.hermitian_tol, 1e-12))
        mass_matrix = None if raw_mass is None else _enforce_hermitian(raw_mass, max(config.hermitian_tol, 1e-12))
        solver = eigsh
        solver_name = "scipy.sparse.linalg.eigsh (ARPACK), shift-invert, Hermitian"
    else:
        matrix = raw
        mass_matrix = raw_mass
        solver = eigs
        solver_name = "scipy.sparse.linalg.eigs (ARPACK), shift-invert, non-Hermitian (bi-orthogonal)"

    if mass_matrix is not None:
        dtype = np.result_type(matrix.dtype, mass_matrix.dtype)
        matrix = matrix.astype(dtype)
        mass_matrix = mass_matrix.astype(dtype)

    # Optional bandwidth/fill-in reduction: work on B = P A P^T with an RCM permutation,
    # then map eigenvectors back with x = P^T y.
    perm: np.ndarray | None = None
    if config.reorder == "rcm":
        perm = _rcm_permutation(matrix)
        matrix, _ = _apply_permutation(matrix, perm)
        if mass_matrix is not None:
            mass_matrix, _ = _apply_permutation(mass_matrix, perm)

    started = time.perf_counter()
    norm_a = max(float(sparse.linalg.norm(matrix)), 1.0)
    step = config.shift_step if config.shift_step is not None else max(1e-8 * norm_a, 1e-12)

    # Ordered candidate shifts: the requested sigma first, then explicit target-band
    # probes (absolute offsets from sigma), then the blind/geometric backoff retries.
    offsets = _candidate_offsets(config, step)

    attempts: list[dict[str, Any]] = []
    last_error: Exception | None = None
    counts: dict[str, Any] = {}
    for offset in offsets:
        shift = config.sigma + offset
        try:
            inner, inner_description, counts = _build_inner_solver(matrix, config, shift, mass_matrix)
            values, vectors, _ = _solve_once(matrix, config, shift, solver, inner, mass_matrix)
            if perm is not None:
                vectors = vectors[np.argsort(perm), :]
            residual_matrix = raw if perm is not None else matrix
            residual_mass = raw_mass if perm is not None else mass_matrix
            residuals = _residuals(residual_matrix, values, vectors, residual_mass)
            attempts.append({
                "shift": _json_scalar(shift),
                "status": "ok",
                "max_residual": float(residuals.max()),
                "inner_solver": inner_description,
                "arpack_iterations": int(counts.get("arpack_applies", 0)),
                "inner_matvecs": int(counts.get("inner_matvecs", 0)),
                "factor_kind": counts.get("factor_kind"),
                "factor_nnz": counts.get("factor_nnz"),
                "ilu_adaptive": counts.get("ilu_adaptive", False),
                "ilu_plan": counts.get("ilu_plan"),
            })
            if np.all(residuals <= config.residual_tol):
                order = np.argsort(np.abs(values - config.sigma), kind="stable")
                values, vectors, residuals = values[order], vectors[:, order], residuals[order]
                break
            last_error = RuntimeError("computed eigenpairs failed the requested residual tolerance")
            attempts[-1]["status"] = "residual_too_large"
        except (ArpackNoConvergence, RuntimeError, ValueError) as exc:
            last_error = exc
            attempts.append({"shift": _json_scalar(shift), "status": "error", "error": str(exc)})
    else:
        raise RuntimeError(f"eigensolver did not produce verified eigenpairs: {last_error}; attempts={attempts}")

    diagnostics = {
        "backend": "cpu",
        "matrix_shape": list(matrix.shape),
        "matrix_nnz": int(matrix.nnz),
        "matrix_dtype": str(matrix.dtype),
        "problem_type": "generalized" if raw_mass is not None else "standard",
        "mass_matrix_shape": list(raw_mass.shape) if raw_mass is not None else None,
        "mass_matrix_nnz": int(raw_mass.nnz) if raw_mass is not None else None,
        "mass_matrix_dtype": str(raw_mass.dtype) if raw_mass is not None else None,
        "matrix_class": detected,
        "resolved_mode": resolved,
        "k": config.k,
        "requested_sigma": _json_scalar(config.sigma),
        "effective_shift": attempts[-1]["shift"],
        "attempts": attempts,
        "n_attempts": len(attempts),
        "target_band": list(config.target_band) if config.target_band is not None else None,
        "adaptive_shift": bool(config.adaptive_shift),
        "arpack_iterations": int(attempts[-1].get("arpack_iterations", 0)),
        "inner_matvecs": int(attempts[-1].get("inner_matvecs", 0)),
        "factor_kind": attempts[-1].get("factor_kind"),
        "factor_nnz": attempts[-1].get("factor_nnz"),
        "ilu_adaptive": bool(config.ilu_adaptive),
        "ilu_plan": attempts[-1].get("ilu_plan"),
        "max_relative_residual": float(residuals.max()),
        "elapsed_seconds": time.perf_counter() - started,
        "solver": solver_name,
        "strategy": {
            "permc_spec": config.permc_spec or "COLAMD",
            "reorder": config.reorder,
            "inner_solver": config.inner_solver,
            "inner_description": attempts[-1].get("inner_solver"),
        },
        "sparse_factorization_memory_note": "Shift-invert may incur substantial sparse LU fill-in; peak memory depends on matrix structure.",
    }
    if config.reference_vectors is not None:
        match_mode = "value" if resolved == "hermitian" else "direction"
        ref = config.reference_vectors
        ref_values = None
        if isinstance(ref, dict):
            ref_values = ref.get("values")
            ref = ref.get("vectors")
        diagnostics["subspace_angle"] = _subspace_angle(
            vectors, np.asarray(ref), match_mode, values=values, reference_values=ref_values
        )
    return values, vectors, residuals, diagnostics


def save_results(output: str | Path, values: np.ndarray, vectors: np.ndarray, residuals: np.ndarray, diagnostics: dict[str, Any]) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, eigenvalues=values, eigenvectors=vectors, residuals=residuals)
    diagnostics_path = output.with_suffix(".json")
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")





