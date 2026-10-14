"""Tests for triangular-matrix restoration (on-demand) and thread control."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sparse_eigensolver.solver import (  # noqa: E402
    SolverConfig,
    _restore_triangular,
    load_matrix,
    set_threads,
    solve,
)


# --------------------------------------------------------------------------- #
# _restore_triangular
# --------------------------------------------------------------------------- #
def test_restore_strict_lower_triangle_is_hermitian() -> None:
    rng = np.random.default_rng(0)
    n = 8
    # strict lower triangle (no diagonal) of a complex Hermitian matrix
    full = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    full = full + full.conj().T
    lower = sparse.tril(sparse.csr_matrix(full), k=-1)
    restored = _restore_triangular(lower, "auto")
    # auto must detect strict-triangle storage and mirror with conjugation
    assert (restored - restored.getH()).nnz == 0
    # the strict lower part is preserved and the diagonal is (still) zero
    assert np.allclose(restored.toarray()[np.tril_indices(n, -1)], full[np.tril_indices(n, -1)])
    assert np.allclose(restored.diagonal(), 0.0)


def test_restore_lower_triangle_with_diagonal_is_symmetric() -> None:
    rng = np.random.default_rng(1)
    n = 8
    full = rng.standard_normal((n, n))
    full = full + full.T  # real symmetric
    lower = sparse.tril(sparse.csr_matrix(full), k=0)
    restored = _restore_triangular(lower, "auto")
    assert (restored - restored.T).nnz == 0
    assert np.allclose(restored.toarray(), full)


def test_restore_explicit_modes_differ_for_complex() -> None:
    rng = np.random.default_rng(2)
    n = 6
    lower = sparse.tril(sparse.csr_matrix(rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))), k=-1)
    hermitian = _restore_triangular(lower, "hermitian")
    symmetric = _restore_triangular(lower, "symmetric")
    # strict-triangle storage -> real (zero) diagonal, so the Hermitian mirror is Hermitian
    assert (hermitian - hermitian.getH()).nnz == 0
    assert (symmetric - symmetric.T).nnz == 0
    # For complex off-diagonal data the two conventions genuinely differ.
    assert (hermitian - symmetric).nnz > 0


def test_restore_auto_uses_real_diagonal_for_hermitian() -> None:
    # A stored complex triangle with a *real* diagonal -> Hermitian mirror.
    rng = np.random.default_rng(7)
    n = 6
    lower = sparse.tril(
        sparse.csr_matrix(rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))), k=0
    ).tolil()
    lower.setdiag(np.arange(1.0, n + 1))  # real diagonal
    lower = lower.tocsr()
    restored = _restore_triangular(lower, "auto")
    assert (restored - restored.getH()).nnz == 0


def test_restore_auto_uses_complex_diagonal_for_symmetric() -> None:
    # A stored complex triangle with an *imaginary* diagonal -> symmetric mirror.
    rng = np.random.default_rng(8)
    n = 6
    lower = sparse.tril(
        sparse.csr_matrix(rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))), k=0
    ).tolil()
    lower.setdiag(np.arange(1.0, n + 1) + 0.5j)  # complex diagonal
    lower = lower.tocsr()
    restored = _restore_triangular(lower, "auto")
    assert (restored - restored.T).nnz == 0
    assert (restored - restored.getH()).nnz > 0


def test_restore_none_is_identity_and_bad_mode_rejected() -> None:
    lower = sparse.tril(sparse.random(6, 6, density=0.5, random_state=3, format="csr"), k=-1)
    assert (_restore_triangular(lower, "none") != lower).nnz == 0
    with pytest.raises(ValueError, match="triangular"):
        _restore_triangular(lower, "bogus")


# --------------------------------------------------------------------------- #
# load_matrix(triangular=...) round-trip and end-to-end solve
# --------------------------------------------------------------------------- #
def test_load_matrix_triangular_roundtrip_hermitian(tmp_path: Path) -> None:
    rng = np.random.default_rng(4)
    n = 12
    full = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    full = sparse.csr_matrix(full + full.conj().T)  # Hermitian -> real diagonal
    stored = sparse.tril(full, k=0).tocsr()
    path = tmp_path / "lower.npz"
    sparse.save_npz(path, stored)

    restored = load_matrix(path, triangular="auto")
    assert np.allclose(restored.toarray(), full.toarray())
    # loading without the flag leaves the stored triangle untouched
    plain = load_matrix(path)
    assert (plain - stored).nnz == 0


def test_triangular_solve_matches_full_matrix(tmp_path: Path) -> None:
    # A Hermitian matrix with a few isolated eigenvalues, stored as its lower triangle.
    diagonal = np.array([-8.0, -3.0, -1.0, 0.5, 2.0, 6.0, 9.0, 15.0], dtype=complex)
    n = diagonal.size
    rng = np.random.default_rng(5)
    upper = sparse.triu(sparse.random(n, n, density=0.2, random_state=6, format="csr"), k=1).astype(complex)
    full = (sparse.diags(diagonal) + upper + upper.getH()).tocsr()
    path = tmp_path / "lower.npz"
    sparse.save_npz(path, sparse.tril(full, k=0).tocsr())

    restored = load_matrix(path, triangular="hermitian")
    assert np.allclose(restored.toarray(), full.toarray())
    v_full, _, r_full, _ = solve(full, SolverConfig(k=3, sigma=0.5))
    v_rest, _, r_rest, _ = solve(restored, SolverConfig(k=3, sigma=0.5))
    assert np.allclose(np.sort(v_full.real), np.sort(v_rest.real), atol=1e-6)
    assert r_rest.max() < 1e-8 and r_full.max() < 1e-8


# --------------------------------------------------------------------------- #
# set_threads
# --------------------------------------------------------------------------- #
def test_set_threads_sets_and_clears_env(monkeypatch) -> None:
    import os

    set_threads(3)
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        assert os.environ[var] == "3"
    set_threads(0)
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        assert var not in os.environ
