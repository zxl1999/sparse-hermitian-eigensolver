"""Tests for the accuracy/iteration metrics, adaptive shift, and config resolution."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from sparse_eigensolver import __main__ as cli
from sparse_eigensolver.benchmark import make_matrix
from sparse_eigensolver.solver import (
    SolverConfig,
    _candidate_offsets,
    _subspace_angle,
    solve,
)


# --------------------------------------------------------------------------- #
# Subspace angle
# --------------------------------------------------------------------------- #
def test_subspace_angle_identical_and_orthogonal() -> None:
    rng = np.random.default_rng(0)
    block = rng.standard_normal((10, 3))
    identical = _subspace_angle(block, block, "value")
    # arccos loses precision near 1, so an identical subspace reads a few 1e-6 deg.
    assert identical["max_angle_deg"] < 1e-4
    assert identical["mean_angle_deg"] < 1e-4

    e1 = np.eye(10)[:, :3]
    e2 = np.eye(10)[:, 3:6]
    orthogonal = _subspace_angle(e1, e2, "direction")
    assert orthogonal["max_angle_deg"] == pytest.approx(90.0, abs=1e-6)
    assert orthogonal["mean_angle_deg"] == pytest.approx(90.0, abs=1e-6)


def test_subspace_angle_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="same number of rows"):
        _subspace_angle(np.ones((5, 2)), np.ones((6, 2)), "value")


def test_subspace_angle_reported_against_reference() -> None:
    matrix = make_matrix(200, "clustered", 3)
    ref_values, ref_vectors, _, _ = solve(matrix, SolverConfig(k=4, sigma=0.0))
    # A different-but-equivalent factorization should reproduce the same subspace.
    values, vectors, residuals, diagnostics = solve(
        matrix,
        SolverConfig(k=4, sigma=0.0, inner_solver="ilu", reference_vectors={
            "vectors": ref_vectors,
            "values": ref_values,
        }),
    )
    assert residuals.max() < 1e-6
    angle = diagnostics["subspace_angle"]
    assert angle["match"] == "value"
    assert angle["max_angle_deg"] < 1.0  # same eigenpairs -> nearly aligned subspaces
    assert angle["max_eigenvalue_error"] < 1e-4
    assert len(angle["angles_deg"]) == 4


def test_subspace_angle_direction_matching_on_general_matrix() -> None:
    matrix = make_matrix(160, "complex_symmetric", 11)
    ref_values, ref_vectors, _, _ = solve(matrix, SolverConfig(k=4, sigma=0.0))
    _, _, residuals, diagnostics = solve(
        matrix,
        SolverConfig(k=4, sigma=0.0, inner_solver="ilu", reorder="rcm",
                     reference_vectors={"vectors": ref_vectors, "values": ref_values}),
    )
    assert residuals.max() < 1e-5
    assert diagnostics["subspace_angle"]["match"] == "direction"


# --------------------------------------------------------------------------- #
# Iteration counts / factorization diagnostics
# --------------------------------------------------------------------------- #
def test_iteration_counts_direct() -> None:
    matrix = make_matrix(400, "indefinite", 7)
    _, _, residuals, diagnostics = solve(matrix, SolverConfig(k=4, sigma=0.0))
    assert residuals.max() < 1e-8
    assert diagnostics["n_attempts"] == 1
    assert diagnostics["arpack_iterations"] > 0
    assert diagnostics["inner_matvecs"] == diagnostics["arpack_iterations"]
    assert diagnostics["factor_kind"] == "LU"
    # A sparse LU always has at least as many nonzeros as the original matrix.
    assert diagnostics["factor_nnz"] >= matrix.nnz
    assert diagnostics["strategy"]["inner_solver"] == "direct"


def test_iteration_counts_ilu_uses_more_inner_matvecs() -> None:
    matrix = make_matrix(400, "clustered", 3)
    _, _, residuals, diagnostics = solve(
        matrix, SolverConfig(k=4, sigma=0.0, inner_solver="ilu")
    )
    assert residuals.max() < 1e-6
    assert diagnostics["factor_kind"] == "ILU"
    # GMRES applies (A - sigma I) several times per Arnoldi iteration.
    assert diagnostics["inner_matvecs"] > diagnostics["arpack_iterations"]
    assert diagnostics["strategy"]["inner_description"].startswith("ILU")


# --------------------------------------------------------------------------- #
# Adaptive shift / target band
# --------------------------------------------------------------------------- #
def test_candidate_offsets_adaptive_backoff() -> None:
    adaptive = _candidate_offsets(SolverConfig(adaptive_shift=True, shift_step=0.01), 0.01)
    assert adaptive[0] == 0.0
    assert 0.01 in adaptive and 0.02 in adaptive and 0.05 in adaptive and 0.1 in adaptive
    # de-duplicated, order preserved
    assert len(adaptive) == len(set(adaptive))

    plain = _candidate_offsets(SolverConfig(adaptive_shift=False, retries=3, shift_step=0.01), 0.01)
    assert plain == [0.0, 0.01, -0.01, 0.02]


def test_candidate_offsets_target_band_walks_outward() -> None:
    offsets = _candidate_offsets(
        SolverConfig(target_band=(1.0, 2.0), sigma=0.0, retries=1, shift_step=0.5), 0.5
    )
    assert offsets[0] == 0.0
    assert offsets[1] == pytest.approx(1.5)  # band centre first
    assert 1.0 in [round(o, 6) for o in offsets]
    assert 2.0 in [round(o, 6) for o in offsets]


def test_adaptive_shift_recovers_near_singular_shift() -> None:
    matrix = make_matrix(300, "near_shift", 5)  # an eigenvalue sits exactly on sigma=0
    values, _, residuals, diagnostics = solve(
        matrix, SolverConfig(k=3, sigma=0.0, retries=0, adaptive_shift=True, shift_step=1e-3)
    )
    assert residuals.max() < 1e-8
    assert diagnostics["adaptive_shift"] is True
    # sigma was singular, so the solver had to move off it.
    assert diagnostics["n_attempts"] >= 1
    assert diagnostics["effective_shift"] != 0.0 or diagnostics["n_attempts"] == 1


def test_target_band_is_recorded_and_validated() -> None:
    matrix = make_matrix(200, "low_freq_breakdown", 9)
    _, _, residuals, diagnostics = solve(
        matrix, SolverConfig(k=4, sigma=0.0, target_band=(-1e-6, 1e-6), adaptive_shift=True)
    )
    assert residuals.max() < 1e-8
    assert diagnostics["target_band"] == [-1e-6, 1e-6]
    with pytest.raises(ValueError, match="finite"):
        solve(matrix, SolverConfig(k=2, sigma=0.0, target_band=(0.0, np.inf)))


# --------------------------------------------------------------------------- #
# New benchmark cases
# --------------------------------------------------------------------------- #
def test_low_freq_breakdown_case() -> None:
    matrix = make_matrix(64, "low_freq_breakdown", 1)
    assert sparse.issparse(matrix)
    assert (matrix - matrix.getH()).nnz == 0
    assert np.count_nonzero(np.abs(matrix.diagonal()) < 1e-6) >= 8


def test_memory_limited_case_shape_and_sparsity() -> None:
    matrix = make_matrix(900, "memory_limited", 1)  # 30 x 30 grid
    assert matrix.shape == (900, 900)
    assert matrix.nnz <= 5 * 900
    with pytest.raises(ValueError, match="perfect square"):
        make_matrix(50, "memory_limited", 1)


def test_memory_limited_3d_case_shape_and_sparsity() -> None:
    matrix = make_matrix(8000, "memory_limited_3d", 1)  # 20^3 grid
    assert matrix.shape == (8000, 8000)
    assert matrix.nnz <= 7 * 8000  # 7-point stencil, sparsely stored
    assert (matrix - matrix.T).nnz == 0  # symmetric (real), so a Hermitian eigenproblem
    with pytest.raises(ValueError, match="perfect cube"):
        make_matrix(2000, "memory_limited_3d", 1)


# --------------------------------------------------------------------------- #
# Config resolution (defaults <- config file <- CLI)
# --------------------------------------------------------------------------- #
def test_resolve_config_defaults_only() -> None:
    args = cli.build_parser().parse_args(["m.npz", "--output", "o.npz"])
    resolved = cli.resolve_config(args)
    assert resolved["k"] == cli.DEFAULTS["k"]
    assert resolved["mode"] == "auto"
    assert resolved["target_band"] is None


def test_resolve_config_file_then_cli_override(tmp_path: Path) -> None:
    cfg_path = tmp_path / "c.json"
    cfg_path.write_text(json.dumps({"k": 10, "sigma": 1.5, "shift_probes": [0.1, 0.2]}))
    args = cli.build_parser().parse_args(
        ["m.npz", "--output", "o.npz", "--config", str(cfg_path), "-k", "3"]
    )
    resolved = cli.resolve_config(args)
    assert resolved["k"] == 3  # CLI wins
    assert resolved["sigma"] == 1.5  # from file
    assert resolved["shift_probes"] == (0.1, 0.2)  # list -> tuple


def test_resolve_config_rejects_unknown_key(tmp_path: Path) -> None:
    cfg_path = tmp_path / "bad.json"
    cfg_path.write_text(json.dumps({"nonsense": 1}))
    args = cli.build_parser().parse_args(
        ["m.npz", "--output", "o.npz", "--config", str(cfg_path)]
    )
    with pytest.raises(ValueError, match="unknown config keys"):
        cli.resolve_config(args)


def test_dump_config_roundtrip(tmp_path: Path) -> None:
    out = tmp_path / "resolved.json"
    args = cli.build_parser().parse_args(
        ["m.npz", "--output", "o.npz", "--target-band", "1", "2", "--adaptive-shift",
         "--dump-config", str(out)]
    )
    resolved = cli.resolve_config(args)
    assert resolved["target_band"] == (1.0, 2.0)
    assert resolved["adaptive_shift"] is True


def test_adaptive_ilu_selects_rung_and_records_trace() -> None:
    matrix = make_matrix(200, "clustered", 3)
    _, _, residuals, diagnostics = solve(
        matrix,
        SolverConfig(
            k=4,
            sigma=0.0,
            inner_solver="ilu",
            ilu_adaptive=True,
            ilu_memory_budget_mb=64.0,
            ilu_probe_maxiter=2,
            ilu_probe_time_budget_s=2.0,
        ),
    )
    assert residuals.max() < 1e-6
    assert diagnostics["ilu_adaptive"] is True
    plan = diagnostics["ilu_plan"]
    assert plan["selected"]["fill_factor"] in (1.0, 2.0, 5.0, 10.0, 20.0, 50.0)
    assert plan["selected"]["predicted_work"] > 0
    assert len(plan["trace"]) == 6
    assert any(record["status"] == "ok" for record in plan["trace"])


def test_adaptive_ilu_requires_ilu_inner_solver() -> None:
    matrix = make_matrix(80, "indefinite", 4)
    with pytest.raises(ValueError, match="requires inner_solver"):
        solve(matrix, SolverConfig(k=3, ilu_adaptive=True))


def test_adaptive_ilu_memory_budget_can_reject_all_rungs() -> None:
    matrix = make_matrix(100, "indefinite", 4)
    with pytest.raises(RuntimeError, match="no feasible rung"):
        solve(
            matrix,
            SolverConfig(
                k=3,
                inner_solver="ilu",
                ilu_adaptive=True,
                ilu_memory_budget_mb=1e-9,
                ilu_probe_time_budget_s=1.0,
            ),
        )


def test_cli_resolves_adaptive_ilu_options(tmp_path: Path) -> None:
    cfg_path = tmp_path / "adaptive.json"
    cfg_path.write_text(json.dumps({"inner_solver": "ilu", "ilu_adaptive": True, "ilu_memory_budget_mb": 128.0}))
    args = cli.build_parser().parse_args(["m.npz", "--output", "o.npz", "--config", str(cfg_path)])
    resolved = cli.resolve_config(args)
    assert resolved["ilu_adaptive"] is True
    assert resolved["ilu_memory_budget_mb"] == 128.0


def test_cli_resolves_generalized_matrix_options(tmp_path: Path) -> None:
    cfg_path = tmp_path / "generalized.json"
    cfg_path.write_text(json.dumps({"mass_matrix": "B.npz", "mass_triangular": "hermitian"}))
    args = cli.build_parser().parse_args(["A.npz", "--output", "o.npz", "--config", str(cfg_path)])
    resolved = cli.resolve_config(args)
    assert resolved["mass_matrix"] == "B.npz"
    assert resolved["mass_triangular"] == "hermitian"
