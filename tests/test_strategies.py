"""Tests for the fill-in reduction strategies: reordering, permc_spec, ILU+GMRES."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from sparse_eigensolver.benchmark import make_matrix
from sparse_eigensolver.solver import SolverConfig, solve


def _reference(matrix: sparse.csr_matrix, k: int = 4, sigma: float = 0.0) -> np.ndarray:
    values, _, residuals, _ = solve(matrix, SolverConfig(k=k, sigma=sigma))
    assert residuals.max() < 1e-8
    return np.sort_complex(values)


@pytest.mark.parametrize("reorder", [None, "rcm"])
@pytest.mark.parametrize("permc_spec", [None, "NATURAL", "MMD_AT_PLUS_A", "MMD_ATA", "COLAMD"])
def test_reorder_and_permc_preserve_spectrum(reorder: str | None, permc_spec: str | None) -> None:
    matrix = make_matrix(200, "indefinite", 7)
    expected = _reference(matrix)
    values, _, residuals, diagnostics = solve(
        matrix, SolverConfig(k=4, sigma=0.0, reorder=reorder, permc_spec=permc_spec)
    )
    assert residuals.max() < 1e-8
    assert np.allclose(np.sort_complex(values), expected, rtol=1e-6, atol=1e-9)
    assert diagnostics["strategy"]["reorder"] == reorder
    assert diagnostics["strategy"]["permc_spec"] == (permc_spec or "COLAMD")


@pytest.mark.parametrize("reorder", [None, "rcm"])
def test_ilu_gmres_inner_solver_matches_direct(reorder: str | None) -> None:
    matrix = make_matrix(200, "clustered", 3)
    expected = _reference(matrix)
    values, _, residuals, diagnostics = solve(
        matrix, SolverConfig(k=4, sigma=0.0, inner_solver="ilu", reorder=reorder)
    )
    assert residuals.max() < 1e-6
    assert np.allclose(np.sort_complex(values), expected, rtol=1e-5, atol=1e-7)
    assert diagnostics["strategy"]["inner_solver"] == "ilu"
    assert "ILU" in diagnostics["strategy"]["inner_description"]


def test_ilu_gmres_on_complex_non_hermitian() -> None:
    matrix = make_matrix(160, "complex_symmetric", 11)
    direct = solve(matrix, SolverConfig(k=4, sigma=0.0))[0]
    iterative = solve(matrix, SolverConfig(k=4, sigma=0.0, inner_solver="ilu", reorder="rcm"))
    values, _, residuals, diagnostics = iterative
    assert residuals.max() < 1e-5
    assert diagnostics["resolved_mode"] == "general"
    assert np.allclose(
        np.sort_complex(values), np.sort_complex(direct), rtol=1e-4, atol=1e-6
    )


def test_rcm_reorders_and_maps_back() -> None:
    """RCM must permute internally yet return eigenvectors in the original ordering."""
    matrix = make_matrix(120, "indefinite", 2)
    values, vectors, residuals, diagnostics = solve(matrix, SolverConfig(k=3, sigma=0.0, reorder="rcm"))
    assert residuals.max() < 1e-8
    # residuals are measured against the *original* matrix, so this validates the back-permutation.
    direct_values = solve(matrix, SolverConfig(k=3, sigma=0.0))[0]
    assert np.allclose(np.sort_complex(values), np.sort_complex(direct_values), rtol=1e-6, atol=1e-9)


def test_invalid_strategy_rejected() -> None:
    matrix = make_matrix(20, "indefinite", 1)
    with pytest.raises(ValueError):
        solve(matrix, SolverConfig(k=2, sigma=0.0, reorder="bogus"))
    with pytest.raises(ValueError):
        solve(matrix, SolverConfig(k=2, sigma=0.0, permc_spec="bogus"))
    with pytest.raises(ValueError):
        solve(matrix, SolverConfig(k=2, sigma=0.0, inner_solver="bogus"))
