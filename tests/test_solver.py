from pathlib import Path
import sys

import numpy as np
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sparse_eigensolver.solver import SolverConfig, load_matrix, solve


def check_against_dense(matrix: sparse.csr_matrix, sigma: float, k: int) -> None:
    values, vectors, residuals, _ = solve(matrix, SolverConfig(k=k, sigma=sigma, retries=4))
    expected = np.linalg.eigvalsh(matrix.toarray())
    expected = expected[np.argsort(np.abs(expected - sigma))[:k]]
    assert np.allclose(np.sort(values), np.sort(expected), atol=1e-8, rtol=1e-8)
    assert residuals.max() < 1e-8
    assert np.allclose(np.linalg.norm(vectors, axis=0), 1.0)


def test_indefinite_and_dense_oracle() -> None:
    diagonal = np.array([-9.0, -5.0, -1.0, 0.25, 2.0, 7.0, 11.0])
    check_against_dense(sparse.diags(diagonal, format="csr"), sigma=0.1, k=3)


def test_clustered_zero_and_repeated_eigenvalues() -> None:
    diagonal = np.array([-2.0, -0.2, -0.01, -0.01, 0.01, 0.01, 0.2, 3.0])
    check_against_dense(sparse.diags(diagonal, format="csr"), sigma=0.0, k=4)


def test_exact_shift_retries() -> None:
    matrix = sparse.diags(np.array([-3.0, -1.0, 0.0, 0.5, 2.0, 4.0]), format="csr")
    values, _, residuals, diagnostics = solve(matrix, SolverConfig(k=2, sigma=0.0, retries=4, shift_step=1e-5))
    expected = np.array([0.0, 0.5])
    assert np.allclose(values, expected, atol=1e-8)
    assert residuals.max() < 1e-8
    assert len(diagnostics["attempts"]) >= 2


def test_sparse_npz_roundtrip(tmp_path: Path) -> None:
    original = sparse.diags(np.arange(1.0, 7.0), format="csr")
    path = tmp_path / "matrix.npz"
    sparse.save_npz(path, original)
    loaded = load_matrix(path)
    assert sparse.isspmatrix_csr(loaded)
    assert (loaded != original).nnz == 0
