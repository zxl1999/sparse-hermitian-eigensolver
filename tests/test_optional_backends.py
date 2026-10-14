from __future__ import annotations

import os
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from sparse_eigensolver.solver import SolverConfig, solve  # noqa: E402


def _gpu_ready() -> bool:
    if os.environ.get('RUN_GPU_TESTS') != '1':
        return False
    try:
        import cupy as cp
        return int(cp.cuda.runtime.getDeviceCount()) > 0
    except Exception:
        return False


@pytest.mark.skipif(not _gpu_ready(), reason='set RUN_GPU_TESTS=1 with a CUDA device to run GPU tests')
def test_gpu_hermitian_shift_invert_residual() -> None:
    diagonal = sparse.diags(np.arange(1.0, 11.0), format='csr')
    values, _, residuals, diagnostics = solve(
        diagonal,
        SolverConfig(k=2, ncv=5, sigma=4.4, mode='hermitian', backend='gpu'),
    )
    assert np.allclose(np.sort(values), [4.0, 5.0], atol=1e-7)
    assert residuals.max() < 1e-8
    assert diagnostics['backend'] == 'cupy-gpu'


@pytest.mark.skipif(not _gpu_ready(), reason='set RUN_GPU_TESTS=1 with a CUDA device to run GPU tests')
def test_gpu_generalized_hermitian_residual() -> None:
    matrix = sparse.diags([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], format='csr')
    mass = sparse.diags([1.0, 1.0, 2.0, 2.0, 3.0, 3.0], format='csr')
    values, _, residuals, diagnostics = solve(
        matrix,
        SolverConfig(k=2, ncv=5, sigma=1.4, mode='hermitian', backend='gpu', mass_matrix=mass),
    )
    assert np.allclose(np.sort(values), [1.5, 5.0 / 3.0], atol=1e-7)
    assert residuals.max() < 1e-8
    assert diagnostics['problem_type'] == 'generalized'
