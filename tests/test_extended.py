from pathlib import Path
import sys

import numpy as np
import pytest
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sparse_eigensolver.solver import SolverConfig, load_matrix, save_results, solve
from sparse_eigensolver.benchmark import make_matrix


def test_complex_hermitian_dense_npy_roundtrip(tmp_path: Path) -> None:
    matrix = np.array([[1.0, 1.0j, 0.0], [-1.0j, 2.0, 0.0], [0.0, 0.0, 4.0]], dtype=complex)
    input_path = tmp_path / "matrix.npy"
    np.save(input_path, matrix)
    loaded = load_matrix(input_path)
    values, vectors, residuals, diagnostics = solve(loaded, SolverConfig(k=1, sigma=1.5))
    output_path = tmp_path / "result.npz"
    save_results(output_path, values, vectors, residuals, diagnostics)
    with np.load(output_path) as result:
        assert set(result.files) == {"eigenvalues", "eigenvectors", "residuals"}
        assert result["eigenvectors"].shape == (3, 1)
        assert result["residuals"].max() < 1e-8
    assert (tmp_path / "result.json").exists()


def test_rejects_non_hermitian_and_bad_config() -> None:
    with pytest.raises(ValueError, match="Hermitian"):
        solve(np.array([[1.0, 2.0], [0.0, 1.0]]), SolverConfig(k=1, mode="hermitian"))
    with pytest.raises(ValueError, match="shift_step"):
        solve(sparse.eye(4, format="csr"), SolverConfig(k=1, shift_step=0))


def test_auto_detects_and_routes_general_matrix() -> None:
    # Real skew-symmetric block-diagonal: non-Hermitian, eigenvalues +-1j, +-2j.
    general = sparse.block_diag(
        (np.array([[0.0, 1.0], [-1.0, 0.0]]), np.array([[0.0, 2.0], [-2.0, 0.0]])), format="csr"
    )
    values, vectors, residuals, diagnostics = solve(general, SolverConfig(k=2, sigma=0.0))
    assert diagnostics["matrix_class"] == "general"
    assert diagnostics["resolved_mode"] == "general"
    assert np.allclose(np.sort_complex(values), [-1.0j, 1.0j], atol=1e-8)
    assert residuals.max() < 1e-8


def test_complex_non_hermitian_matches_dense_oracle() -> None:
    rng = np.random.default_rng(3)
    dense = rng.standard_normal((6, 6)) + 1j * rng.standard_normal((6, 6))
    dense = sparse.csr_matrix(dense)  # complex symmetric is not Hermitian
    sigma = -0.5 + 0.25j
    values, _, residuals, diagnostics = solve(dense, SolverConfig(k=2, sigma=sigma))
    assert diagnostics["resolved_mode"] == "general"
    expected = np.linalg.eigvals(dense.toarray())
    expected = expected[np.argsort(np.abs(expected - sigma))[:2]]
    assert np.allclose(np.sort_complex(values), np.sort_complex(expected), atol=1e-6)
    assert residuals.max() < 1e-8


def test_hermitian_mode_still_matches_dense_oracle() -> None:
    diagonal = np.array([-9.0, -5.0, -1.0, 0.25, 2.0, 7.0, 11.0])
    values, _, residuals, diagnostics = solve(sparse.diags(diagonal, format="csr"), SolverConfig(k=3, sigma=0.1))
    expected = np.sort(np.abs(diagonal - 0.1))[:3]
    assert np.allclose(np.sort(np.abs(values - 0.1)), expected, atol=1e-8)
    assert diagnostics["resolved_mode"] == "hermitian"
    assert residuals.max() < 1e-8


def test_generalized_hermitian_problem_with_mass_matrix() -> None:
    matrix = sparse.diags([1.0, 2.0, 3.0, 4.0], format="csr")
    mass = sparse.diags([1.0, 1.0, 2.0, 2.0], format="csr")
    values, vectors, residuals, diagnostics = solve(
        matrix,
        SolverConfig(k=2, sigma=1.4, mass_matrix=mass, reorder="rcm"),
    )
    assert np.allclose(np.sort(values), [1.0, 1.5], atol=1e-8)
    assert vectors.shape == (4, 2)
    assert residuals.max() < 1e-8
    assert diagnostics["problem_type"] == "generalized"
    assert diagnostics["mass_matrix_shape"] == [4, 4]
    assert diagnostics["mass_matrix_nnz"] == 4


def test_benchmark_matrices_are_sparse_hermitian_and_repeatable() -> None:
    first = make_matrix(32, "clustered", 17)
    second = make_matrix(32, "clustered", 17)
    near_shift = make_matrix(32, "near_shift", 17)
    assert sparse.issparse(first)
    assert (first - first.getH()).nnz == 0
    assert (first != second).nnz == 0
    assert np.count_nonzero(near_shift.diagonal() == 0) == 1
    assert np.count_nonzero(np.abs(near_shift.diagonal()) == 1e-8) == 1
    assert first.nnz < 5 * first.shape[0]


def test_complex_benchmark_cases() -> None:
    hermitian = make_matrix(32, "complex_hermitian", 5)
    symmetric = make_matrix(32, "complex_symmetric", 5)
    assert hermitian.dtype == np.complex128
    assert (hermitian - hermitian.getH()).nnz == 0
    # complex symmetric is symmetric, but not Hermitian
    assert (symmetric - symmetric.T).nnz == 0
    assert (symmetric - symmetric.getH()).nnz > 0
    values_h, _, resid_h, diag_h = solve(hermitian, SolverConfig(k=2, sigma=0.0))
    values_s, _, resid_s, diag_s = solve(symmetric, SolverConfig(k=2, sigma=0.0))
    assert diag_h["resolved_mode"] == "hermitian"
    assert diag_s["resolved_mode"] == "general"
    assert resid_h.max() < 1e-8 and resid_s.max() < 1e-8
    # Hermitian path returns (nearly) real eigenvalues; the general path may return complex ones.
    assert np.abs(values_h.imag).max() < 1e-8
def test_generalized_problem_rejects_bad_mass_shape() -> None:
    matrix = sparse.eye(4, format='csr')
    mass = sparse.eye(3, format='csr')
    with pytest.raises(ValueError, match='mass_matrix must have shape'):
        solve(matrix, SolverConfig(k=2, mass_matrix=mass))
def test_matrix_market_roundtrip(tmp_path: Path) -> None:
    from scipy.io import mmwrite

    matrix = sparse.diags([-2.0, 0.5, 3.0, 7.0], format='csr')
    path = tmp_path / 'matrix.mtx'
    mmwrite(path, matrix)
    loaded = load_matrix(path)
    assert (loaded != matrix).nnz == 0
