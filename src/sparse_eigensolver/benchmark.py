from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

try:
    import resource
except ImportError:
    resource = None
from typing import Any

import numpy as np
from scipy import sparse

from .solver import SolverConfig, save_results, solve


def _peak_rss_mb() -> float:
    """Return this process peak resident memory in MB across major platforms."""
    if resource is not None:
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return usage / (1024 * 1024) if sys.platform == "darwin" else usage / 1024.0
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

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        psapi = ctypes.WinDLL('psapi')
        kernel32 = ctypes.WinDLL('kernel32')
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE, ctypes.POINTER(ProcessMemoryCounters), wintypes.DWORD
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        process = kernel32.GetCurrentProcess()
        ok = psapi.GetProcessMemoryInfo(process, ctypes.byref(counters), counters.cb)
        if ok:
            return counters.PeakWorkingSetSize / (1024 * 1024)
    return 0.0


def make_matrix(n: int, case: str, seed: int) -> sparse.csr_matrix:
    """Create deterministic sparse benchmark matrices.

    ``indefinite``/``clustered``/``repeated``/``near_shift`` are real symmetric.
    ``complex_hermitian`` is a complex Hermitian matrix and ``complex_symmetric``
    is a complex-symmetric (non-Hermitian) matrix, the lossy/PML electromagnetic case.

    Fault-scenario cases used by ``tools/run_fault_scenarios.py``:
    ``low_freq_breakdown`` places a dense cluster of eigenvalues within ~1e-6 of zero
    (low-frequency breakdown); ``memory_limited`` is a 2D 5-point Laplacian whose
    sparse-LU fill-in dominates memory even though it is sparsely stored, and
    ``memory_limited_3d`` is the harder 3D 7-point Laplacian variant (n = m^3) whose
    sparse-LU fill-in is far larger and reliably blows past a 1 GB budget.
    """
    if n < 8:
        raise ValueError("benchmark dimension must be at least 8")
    rng = np.random.default_rng(seed)
    if case == "indefinite":
        diagonal = np.linspace(-1.0, 1.0, n)
        diagonal[n // 2 - 1:n // 2 + 1] = (-1e-4, 1e-4)
    elif case == "clustered":
        diagonal = np.linspace(-2.0, 2.0, n)
        center = n // 2
        diagonal[center - 2:center + 2] = (-1e-5, -5e-6, 5e-6, 1e-5)
    elif case == "repeated":
        diagonal = np.resize(np.array([-1.0, -0.1, -0.1, 0.1, 0.1, 1.0]), n)
    elif case == "near_shift":
        diagonal = np.linspace(-2.0, 2.0, n)
        diagonal[n // 2] = 0.0
        diagonal[n // 2 + 1] = 1e-8
    elif case == "low_freq_breakdown":
        # Many eigenvalues squeezed into a tiny window around zero (low-frequency
        # breakdown in EM FEM). Dense cluster of half-width 8 eigenvalues in [-1e-6, 1e-6].
        diagonal = np.linspace(-2.0, 2.0, n)
        center = n // 2
        half = min(8, center - 1)
        diagonal[center - half:center + half] = np.linspace(-1e-6, 1e-6, 2 * half)
    elif case == "memory_limited":
        # 2D 5-point Laplacian on an m x m grid (n = m^2), shifted to be indefinite.
        # Sparsely stored (5 nnz/row) yet its sparse LU fill-in grows like the front
        # width, so it stresses the memory budget rather than the nnz count.
        m = int(round(np.sqrt(n)))
        if m * m != n:
            raise ValueError("memory_limited requires n to be a perfect square")
        main = 4.0 * np.ones(m * m)
        east = -1.0 * np.ones(m * m - 1)
        east[m - 1::m] = 0.0  # remove the row-wrap coupling
        north = -1.0 * np.ones(m * (m - 1))
        laplacian = sparse.diags([east, main, east, north, north], [-1, 0, 1, -m, m], format="csr")
        return (laplacian - 3.99 * sparse.identity(m * m, format="csr")).tocsr()
    elif case == "memory_limited_3d":
        # 3D 7-point Laplacian on an m x m x m grid (n = m^3), shifted to A = L + I.
        # 7 nnz/row, yet sparse-LU fill-in is ~1500x nnz(A) (tens of GB for m=40),
        # so it decisively stresses the memory budget while ILU stays small.
        m = int(round(n ** (1.0 / 3.0)))
        if m * m * m != n:
            raise ValueError("memory_limited_3d requires n to be a perfect cube")
        base = sparse.diags(
            [-1.0 * np.ones(m - 1), 2.0 * np.ones(m), -1.0 * np.ones(m - 1)],
            [-1, 0, 1], format="csr",
        )
        eye = sparse.identity(m, format="csr")
        laplacian = (
            sparse.kron(eye, sparse.kron(eye, base))
            + sparse.kron(eye, sparse.kron(base, eye))
            + sparse.kron(base, sparse.kron(eye, eye))
        ).tocsr()
        return (laplacian + sparse.identity(m * m * m, format="csr")).tocsr()
    elif case == "complex_hermitian":
        # Hermitian: real diagonal, complex off-diagonal that is conjugate-symmetric.
        diagonal = np.linspace(-1.0, 1.0, n).astype(complex)
        diagonal[n // 2 - 1:n // 2 + 1] = (-1e-4, 1e-4)
        rows = np.arange(n - 1)
        cols = rows + 1
        weights = rng.uniform(-0.02, 0.02, size=n - 1) + 1j * rng.uniform(-0.02, 0.02, size=n - 1)
        upper = sparse.coo_matrix((weights, (rows, cols)), shape=(n, n)).tocsr()
        return (sparse.diags(diagonal, format="csr") + upper + upper.getH()).tocsr()
    elif case == "complex_symmetric":
        diagonal = (np.linspace(-1.0, 1.0, n) + 0.05j * rng.standard_normal(n)).astype(complex)
        diagonal[n // 2 - 1:n // 2 + 1] = (-1e-4, 1e-4)
        rows = np.arange(n - 1)
        cols = rows + 1
        weights = rng.uniform(-0.02, 0.02, size=n - 1).astype(complex)
        upper = sparse.coo_matrix((weights, (rows, cols)), shape=(n, n)).tocsr()
        return (sparse.diags(diagonal, format="csr") + upper + upper.T).tocsr()
    else:
        raise ValueError(f"unknown benchmark case: {case}")

    if case == "near_shift":
        return sparse.diags(diagonal, format="csr")

    rows = np.arange(n - 1)
    cols = rows + 1
    weights = rng.uniform(-0.02, 0.02, size=n - 1)
    off_diagonal = sparse.coo_matrix(
        (np.concatenate((weights, weights)), (np.concatenate((rows, cols)), np.concatenate((cols, rows)))),
        shape=(n, n),
    )
    return (sparse.diags(diagonal, format="csr") + off_diagonal.tocsr()).tocsr()


def run_benchmark(n: int, case: str, k: int, sigma: float, seed: int, retries: int) -> dict[str, Any]:
    matrix = make_matrix(n, case, seed)
    start = time.perf_counter()
    values, vectors, residuals, diagnostics = solve(
        matrix,
        SolverConfig(k=k, sigma=sigma, retries=retries),
    )
    diagnostics.update({
        "benchmark_case": case,
        "seed": seed,
        "wall_seconds": time.perf_counter() - start,
        "peak_rss_mb": _peak_rss_mb(),
        "eigenvalues": values.tolist(),
        "residuals": residuals.tolist(),
        "orthogonality_error_fro": float(np.linalg.norm(vectors.conj().T @ vectors - np.eye(k))),
    })
    return diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reproducible sparse eigensolver synthetic benchmarks.")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--case", choices=("indefinite", "clustered", "repeated", "near_shift", "low_freq_breakdown", "memory_limited", "memory_limited_3d", "complex_hermitian", "complex_symmetric"), default="indefinite")
    parser.add_argument("-k", type=int, default=6)
    parser.add_argument("--sigma", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--output", type=Path, help="Optional JSON diagnostics output path")
    parser.add_argument("--save-matrix", type=Path, help="Optional path to save the generated SciPy sparse matrix")
    args = parser.parse_args()

    matrix = make_matrix(args.n, args.case, args.seed)
    if args.save_matrix:
        args.save_matrix.parent.mkdir(parents=True, exist_ok=True)
        sparse.save_npz(args.save_matrix, matrix)
    start = time.perf_counter()
    values, vectors, residuals, diagnostics = solve(matrix, SolverConfig(k=args.k, sigma=args.sigma, retries=args.retries))
    diagnostics.update({
        "benchmark_case": args.case,
        "seed": args.seed,
        "wall_seconds": time.perf_counter() - start,
        "peak_rss_mb": _peak_rss_mb(),
        "eigenvalues": values.tolist(),
        "residuals": residuals.tolist(),
        "orthogonality_error_fro": float(np.linalg.norm(vectors.conj().T @ vectors - np.eye(args.k))),
    })
    rendered = json.dumps(diagnostics, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()

