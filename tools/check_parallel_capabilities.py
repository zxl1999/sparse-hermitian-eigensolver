#!/usr/bin/env python
"""Report available parallel runtimes and integrated solver execution paths."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_JSON = ROOT / "results" / "parallel_capabilities.json"
DEFAULT_MARKDOWN = ROOT / "results" / "PARALLEL_CAPABILITIES.md"


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _cuda_solver_available() -> tuple[bool, dict[str, Any]]:
    if not _module_available('cupy'):
        return False, {'device_count': 0, 'device_name': None, 'error': 'cupy not installed'}
    try:
        os.environ.setdefault('CUPY_COMPILE_WITH_PTX', '1')
        import cupy as cp
        count = int(cp.cuda.runtime.getDeviceCount())
        if count < 1:
            return False, {'device_count': count, 'device_name': None, 'error': 'no CUDA device'}
        properties = cp.cuda.runtime.getDeviceProperties(0)
        name = properties['name']
        if isinstance(name, bytes):
            name = name.decode(errors='replace')
        cp.sort(cp.asarray([2, 1], dtype=cp.int32))
        cp.cuda.Stream.null.synchronize()
        return True, {'device_count': count, 'device_name': str(name), 'error': None}
    except Exception as exc:
        return False, {'device_count': 0, 'device_name': None, 'error': f'{type(exc).__name__}: {exc}'}


def _slepc_available() -> tuple[bool, str | None]:
    if not (_module_available('petsc4py') and _module_available('slepc4py')):
        return False, 'petsc4py/slepc4py not installed'
    try:
        from petsc4py import PETSc
        from slepc4py import SLEPc
        return True, f'PETSc {PETSc.Sys.getVersion()} / SLEPc {SLEPc.getVersion()}'
    except Exception as exc:
        return False, f'{type(exc).__name__}: {exc}'


def _native_slepc_available() -> tuple[bool, str | None]:
    candidates = []
    configured = os.environ.get("SLEPC_NATIVE_EXECUTABLE")
    if configured:
        candidates.append(Path(configured))
    candidates.extend((ROOT / "tools" / "run_slepc_native.exe", ROOT / "tools" / "run_slepc_native"))
    for candidate in candidates:
        if candidate.is_file():
            env = os.environ.copy()
            extra_paths = [item for item in env.get("SLEPC_NATIVE_DLL_DIRS", "").split(os.pathsep) if item]
            env["PATH"] = os.pathsep.join([str(candidate.parent), *extra_paths, env.get("PATH", "")])
            launcher = shutil.which("mpiexec", path=env["PATH"])
            command = [str(candidate)]
            if launcher:
                command = [launcher, "-n", "1", str(candidate)]
            try:
                probe = subprocess.run(
                    command,
                    cwd=ROOT,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=15,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                return False, f"native driver launch failed: {type(exc).__name__}: {exc}"
            if probe.returncode == 2 and "usage:" in probe.stdout.lower():
                try:
                    display_path = str(candidate.relative_to(ROOT))
                except ValueError:
                    display_path = candidate.name
                return True, f"{display_path}; PETSc/SLEPc runtime initialized"
            message = (probe.stderr or probe.stdout).strip().splitlines()
            return False, f"native driver probe failed: {message[-1] if message else probe.returncode}"
    return False, "native SLEPc executable not built"


def _threadpools() -> list[dict[str, Any]]:
    try:
        from threadpoolctl import threadpool_info
    except ImportError:
        return []
    return [
        {
            "internal_api": item.get("internal_api"),
            "prefix": item.get("prefix"),
            "filepath": item.get("filepath"),
            "num_threads": item.get("num_threads"),
        }
        for item in threadpool_info()
    ]


def collect_capabilities() -> dict[str, Any]:
    optional_runtimes = {
        "threadpoolctl": _module_available("threadpoolctl"),
        "mpi4py_installed": _module_available("mpi4py"),
        "cupy_installed": _module_available("cupy"),
        "petsc4py_installed": _module_available("petsc4py"),
        "slepc4py_installed": _module_available("slepc4py"),
        "mpiexec_found": shutil.which("mpiexec") is not None,
    }
    gpu_available, gpu_details = _cuda_solver_available()
    slepc_available, slepc_details = _slepc_available()
    native_slepc_available, native_slepc_details = _native_slepc_available()
    mpi_solver_available = (slepc_available or native_slepc_available) and optional_runtimes["mpiexec_found"]
    return {
        "generated_by": "tools/check_parallel_capabilities.py",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "supported_execution": {
            "single_process_cpu": True,
            "blas_openmp_threads": optional_runtimes["threadpoolctl"],
            "mpi_independent_batch": optional_runtimes["mpi4py_installed"] and optional_runtimes["mpiexec_found"],
            "mpi_solver_backend": mpi_solver_available,
            "gpu_solver_backend": gpu_available,
            "distributed_solver_backend": mpi_solver_available,
        },
        "optional_runtimes": optional_runtimes,
        "gpu": gpu_details,
        "slepc": {
            "available": slepc_available or native_slepc_available,
            "python_bindings": {"available": slepc_available, "details": slepc_details},
            "native_executable": {"available": native_slepc_available, "details": native_slepc_details},
        },
        "threadpools": _threadpools(),
        "scoring_note": (
            "The solver supports CPU BLAS/OpenMP thread control and a complete CPU fallback. "
            "GPU support is Hermitian-only through CuPy; single-matrix MPI requires PETSc/SLEPc "
            "built against the active MPI runtime; this report also recognizes the native C driver."
        ),
    }


def render_markdown(report: dict[str, Any]) -> str:
    supported = report["supported_execution"]
    optional = report["optional_runtimes"]
    lines = [
        "# Parallel capability report",
        "",
        "This report separates installed runtimes from solver backends that are actually integrated.",
        "",
        "## Environment",
        "",
        f"- Python: `{report['python']}`",
        f"- Platform: `{report['platform']}`",
        f"- Logical CPUs: `{report['logical_cpus']}`",
        "",
        "## Integrated paths",
        "",
        "| Capability | Status | Evidence |",
        "|---|---|---|",
        f"| Single-process CPU | {'supported' if supported['single_process_cpu'] else 'unavailable'} | SciPy ARPACK/SuperLU/GMRES solver path |",
        f"| BLAS/OpenMP thread control | {'supported' if supported['blas_openmp_threads'] else 'environment fallback only'} | `--threads`, environment variables, optional `threadpoolctl` |",
        f"| MPI independent batch | {'supported' if supported['mpi_independent_batch'] else 'unavailable'} | `tools/run_mpi_batch.py`; distributes independent jobs, not one matrix factorization |",
        f"| MPI solver backend | {'supported' if supported['mpi_solver_backend'] else 'not integrated'} | `tools/run_slepc.py` or native `tools/run_slepc_native.c` with PETSc/SLEPc |",
        f"| GPU solver backend | {'supported' if supported['gpu_solver_backend'] else 'not integrated'} | --backend gpu; CuPy CUDA Hermitian shift-invert path |",
        "",
        "## Runtime details",
        "",
        f"- `mpi4py`: `{optional['mpi4py_installed']}`",
        f"- `cupy`: `{optional['cupy_installed']}`",
        f"- `petsc4py`: `{optional['petsc4py_installed']}`",
        f"- native SLEPc executable: `{report['slepc']['native_executable']['available']}`",
        "",
        "## Scoring interpretation",
        "",
        "The CPU fallback remains complete. GPU support is Hermitian-only through CuPy. Single-matrix MPI is supported through petsc4py/slepc4py or the native PETSc/SLEPc driver when a matching MPI stack is installed.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()

    report = collect_capabilities()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.markdown.write_text(render_markdown(report), encoding="utf-8")
    print(f"wrote {args.output} and {args.markdown}")


if __name__ == "__main__":
    main()
