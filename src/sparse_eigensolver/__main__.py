from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .solver import SolverConfig, load_matrix, save_results, solve


# Canonical defaults, applied when neither the CLI nor a --config file supplies a value.
DEFAULTS: dict = {
    "k": 6,
    "sigma": 0.0,
    "tol": 1e-10,
    "residual_tol": 1e-8,
    "maxiter": None,
    "ncv": None,
    "retries": 3,
    "shift_step": None,
    "mode": "auto",
    "backend": "cpu",
    "hermitian_tol": 1e-12,
    "permc_spec": None,
    "reorder": "none",
    "inner_solver": "direct",
    "ilu_drop_tol": 1e-4,
    "ilu_fill_factor": 10.0,
    "gmres_restart": 100,
    "gmres_rtol": 1e-8,
    "ilu_adaptive": False,
    "ilu_memory_budget_mb": None,
    "ilu_probe_maxiter": 3,
    "ilu_probe_time_budget_s": 10.0,
    "shift_probes": (),
    "target_band": None,
    "adaptive_shift": False,
    "mass_matrix": None,
    "mass_triangular": "none",
}

# Config keys whose JSON representation is a list that must become a tuple.
_TUPLE_KEYS = ("shift_probes", "target_band")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Find sparse eigenvalue problems nearest a target shift (Hermitian or general).",
    )
    parser.add_argument("matrix", help="Input .npy, SciPy .npz, or Matrix Market .mtx/.mm matrix")
    parser.add_argument("--output", required=True, help="Output .npz path (diagnostics JSON uses the same stem)")
    parser.add_argument("-k", type=int, default=None, help="Number of eigenpairs (default 6)")
    parser.add_argument("--sigma", type=float, default=None, help="Target shift (default 0.0)")
    parser.add_argument("--tol", type=float, default=None)
    parser.add_argument("--residual-tol", type=float, default=None)
    parser.add_argument("--maxiter", type=int, default=None)
    parser.add_argument("--ncv", type=int, default=None)
    parser.add_argument("--retries", type=int, default=None)
    parser.add_argument("--shift-step", type=float, default=None, help="Absolute offset for retry shifts; default scales with ||A||")
    parser.add_argument("--mode", choices=("auto", "hermitian", "general"), default=None,
                        help="auto: detect Hermitian and use eigsh, else eigs; hermitian/general force the path")
    parser.add_argument("--backend", choices=("cpu", "gpu"), default=None,
                        help="Execution backend; gpu requires cupy-cuda12x and a CUDA device")
    parser.add_argument("--hermitian-tol", type=float, default=None,
                        help="Relative ||A - A^H||/||A|| below which a matrix counts as Hermitian in auto mode")
    parser.add_argument("--permc-spec", choices=("NATURAL", "MMD_AT_PLUS_A", "MMD_ATA", "COLAMD"), default=None,
                        help="SuperLU column ordering for the inner LU (default COLAMD); try NATURAL/MMD to reduce fill-in")
    parser.add_argument("--reorder", choices=("none", "rcm"), default=None,
                        help="Apply a reverse Cuthill-McKee permutation to reduce bandwidth/fill-in before factorizing")
    parser.add_argument("--inner-solver", choices=("direct", "ilu"), default=None,
                        help="direct: sparse LU; ilu: incomplete LU + GMRES (lower memory, iterative)")
    parser.add_argument("--ilu-drop-tol", type=float, default=None, help="ILU drop tolerance (inner-solver=ilu)")
    parser.add_argument("--ilu-fill-factor", type=float, default=None, help="ILU fill factor (inner-solver=ilu)")
    parser.add_argument("--gmres-restart", type=int, default=None, help="GMRES restart (inner-solver=ilu)")
    parser.add_argument("--gmres-rtol", type=float, default=None, help="GMRES relative tolerance (inner-solver=ilu)")
    parser.add_argument("--ilu-adaptive", action="store_true", default=None, help="Select ILU parameters with bounded GMRES probes")
    parser.add_argument("--ilu-memory-budget-mb", type=float, default=None, help="Approximate ILU factor memory budget in MB")
    parser.add_argument("--ilu-probe-maxiter", type=int, default=None, help="Maximum GMRES restart cycles per ILU probe")
    parser.add_argument("--ilu-probe-time-budget-s", type=float, default=None, help="Wall-clock limit per ILU probe")
    parser.add_argument("--shift-probes", type=float, nargs="*", default=None,
                        help="Target-band probe shifts (absolute offsets from sigma) tried in order, e.g. --shift-probes 1e-3 1e-2 0.1 1.0")
    parser.add_argument("--target-band", type=float, nargs=2, metavar=("LO", "HI"), default=None,
                        help="Known target frequency band [LO, HI]; candidate shifts walk from the band centre outward")
    parser.add_argument("--adaptive-shift", action="store_true", default=None,
                        help="Escape a near-singular region with a geometric shift backoff (step, 2*step, 5*step, ...)")
    parser.add_argument("--reference", help="Optional .npz with reference eigenvectors (key 'eigenvectors', optional 'eigenvalues') to report the subspace angle")
    parser.add_argument("--triangular", choices=("none", "auto", "hermitian", "symmetric"), default="none",
                        help="Recover a full matrix from a stored triangle (auto: strict-triangle -> Hermitian, else symmetric)")
    parser.add_argument("--mass-matrix", default=None,
                        help="Optional generalized-eigenvalue mass matrix B for A x = lambda B x")
    parser.add_argument("--mass-triangular", choices=("none", "auto", "hermitian", "symmetric"), default=None,
                        help="Triangle recovery mode for --mass-matrix; defaults to none")
    parser.add_argument("--threads", type=int, default=None,
                        help="Pin BLAS/OpenMP thread count for the inner LU/GMRES (reproducibility + scaling tests); default lets the BLAS choose")
    parser.add_argument("--config", type=Path,
                        help="Optional JSON config file; its keys are defaults and explicit CLI flags override them")
    parser.add_argument("--dump-config", type=Path,
                        help="Write the fully-resolved configuration (defaults + config file + CLI) to this JSON path and exit")
    return parser


def resolve_config(args: argparse.Namespace) -> dict:
    """Merge DEFAULTS <- --config JSON <- explicit CLI flags (CLI wins)."""
    resolved = dict(DEFAULTS)
    if args.config:
        file_cfg = json.loads(Path(args.config).read_text(encoding="utf-8-sig"))
        file_cfg = {key: value for key, value in file_cfg.items() if not str(key).startswith("_")}
        unknown = set(file_cfg) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        resolved.update(file_cfg)
    for key in DEFAULTS:
        value = getattr(args, key.replace("-", "_"), None)
        if value is not None:
            resolved[key] = value
    for key in _TUPLE_KEYS:
        if resolved.get(key) is not None:
            resolved[key] = tuple(resolved[key])
    return resolved


def main() -> None:
    args = build_parser().parse_args()
    resolved = resolve_config(args)

    if args.threads is not None:
        from .solver import set_threads

        set_threads(args.threads)

    if args.dump_config:
        args.dump_config.parent.mkdir(parents=True, exist_ok=True)
        args.dump_config.write_text(json.dumps(resolved, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {args.dump_config}")
        return

    reference = None
    if args.reference:
        with np.load(args.reference) as data:
            reference = {
                "vectors": data["eigenvectors"],
                "values": data["eigenvalues"] if "eigenvalues" in data else None,
            }
    config = SolverConfig(
        k=resolved["k"], sigma=resolved["sigma"], tol=resolved["tol"], maxiter=resolved["maxiter"],
        ncv=resolved["ncv"], retries=resolved["retries"], shift_step=resolved["shift_step"],
        residual_tol=resolved["residual_tol"], mode=resolved["mode"], backend=resolved["backend"], hermitian_tol=resolved["hermitian_tol"],
        permc_spec=resolved["permc_spec"], reorder=None if resolved["reorder"] == "none" else resolved["reorder"],
        inner_solver=resolved["inner_solver"], ilu_drop_tol=resolved["ilu_drop_tol"],
        ilu_fill_factor=resolved["ilu_fill_factor"], gmres_restart=resolved["gmres_restart"],
        gmres_rtol=resolved["gmres_rtol"], ilu_adaptive=bool(resolved["ilu_adaptive"]),
        ilu_memory_budget_mb=resolved["ilu_memory_budget_mb"], ilu_probe_maxiter=resolved["ilu_probe_maxiter"],
        ilu_probe_time_budget_s=resolved["ilu_probe_time_budget_s"], shift_probes=tuple(resolved["shift_probes"] or ()),
        mass_matrix=load_matrix(
            resolved["mass_matrix"],
            triangular=resolved["mass_triangular"],
        ) if resolved["mass_matrix"] else None,
        reference_vectors=reference,
        target_band=tuple(resolved["target_band"]) if resolved["target_band"] else None,
        adaptive_shift=bool(resolved["adaptive_shift"]),
    )
    values, vectors, residuals, diagnostics = solve(load_matrix(args.matrix, triangular=args.triangular), config)
    save_results(args.output, values, vectors, residuals, diagnostics)
    summary = {
        "eigenvalues": [c if not np.issubdtype(np.asarray(c).dtype, np.complexfloating) or np.asarray(c).imag != 0 else float(np.asarray(c).real) for c in values.tolist()],
        "max_relative_residual": float(residuals.max()),
        "matrix_class": diagnostics["matrix_class"],
        "resolved_mode": diagnostics["resolved_mode"],
        "n_attempts": diagnostics["n_attempts"],
        "arpack_iterations": diagnostics["arpack_iterations"],
        "inner_matvecs": diagnostics["inner_matvecs"],
        "subspace_angle": diagnostics.get("subspace_angle"),
        "strategy": diagnostics["strategy"],
        "diagnostics": str(args.output).rsplit(".", 1)[0] + ".json",
    }
    print(json.dumps(summary, indent=2, default=lambda o: [o.real, o.imag] if isinstance(o, complex) else str(o)))


if __name__ == "__main__":
    main()




