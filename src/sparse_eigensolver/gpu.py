'''Optional CuPy/CUDA backend for Hermitian shift-invert problems.'''

from __future__ import annotations

import time
import os
from typing import Any

import numpy as np
from scipy import sparse

from .solver import (
    SolverConfig,
    _apply_permutation,
    _candidate_offsets,
    _enforce_hermitian,
    _json_scalar,
    _rcm_permutation,
    _residuals,
    classify_matrix,
    validate_matrix,
)


def _import_cupy():
    os.environ.setdefault('CUPY_COMPILE_WITH_PTX', '1')
    try:
        import cupy as cp
        import cupyx.scipy.sparse as cpsparse
        import cupyx.scipy.sparse.linalg as cplinalg
    except ImportError as exc:
        raise RuntimeError(
            'GPU backend requires cupy-cuda12x; install the optional gpu dependency'
        ) from exc
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError('GPU backend requires at least one CUDA device')
    except Exception as exc:
        raise RuntimeError(f'CUDA device is unavailable: {exc}') from exc
    return cp, cpsparse, cplinalg


def _to_cupy_csr(matrix: sparse.csr_matrix, cp: Any, cpsparse: Any):
    if matrix.nnz >= np.iinfo(np.int32).max:
        raise ValueError('GPU backend requires CSR indices that fit in int32')
    return cpsparse.csr_matrix(
        (
            cp.asarray(matrix.data),
            cp.asarray(matrix.indices, dtype=cp.int32),
            cp.asarray(matrix.indptr, dtype=cp.int32),
        ),
        shape=matrix.shape,
    )


def _gpu_ncv(config: SolverConfig, n: int) -> int:
    ncv = config.ncv
    if ncv is None:
        ncv = min(n - 1, max(config.k + 2, 2 * config.k + 2))
    if not (config.k + 1 < ncv < n):
        raise ValueError(
            'CuPy eigsh requires k + 1 < ncv < matrix dimension; '
            'set a smaller k or an explicit valid ncv'
        )
    return ncv


def _solve_shift(
    matrix: sparse.csr_matrix,
    mass_matrix: sparse.csr_matrix | None,
    config: SolverConfig,
    shift: float,
    cp: Any,
    cpsparse: Any,
    cplinalg: Any,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    gpu_matrix = _to_cupy_csr(matrix, cp, cpsparse)
    gpu_mass = None if mass_matrix is None else _to_cupy_csr(mass_matrix, cp, cpsparse)
    if gpu_mass is None:
        shifted = gpu_matrix - shift * cpsparse.eye(matrix.shape[0], dtype=gpu_matrix.dtype, format='csr')
    else:
        shifted = gpu_matrix - shift * gpu_mass
    factor = cplinalg.factorized(shifted.tocsc())

    def apply(vector):
        rhs = vector if gpu_mass is None else gpu_mass @ vector
        return factor(rhs)

    operator = cplinalg.LinearOperator(matrix.shape, matvec=apply, dtype=gpu_matrix.dtype)
    mu, vectors = cplinalg.eigsh(
        operator,
        k=config.k,
        which='LM',
        tol=config.tol,
        maxiter=config.maxiter,
        ncv=_gpu_ncv(config, matrix.shape[0]),
        return_eigenvectors=True,
    )
    with np.errstate(divide='ignore', invalid='ignore'):
        values = shift + 1.0 / cp.asnumpy(mu)
    return values, cp.asnumpy(vectors), {
        'factor_kind': 'CUDA sparse direct',
        'factor_nnz': None,
        'arpack_iterations': None,
        'inner_matvecs': None,
    }


def solve_gpu(matrix: Any, config: SolverConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    '''Solve a Hermitian standard or generalized problem on one CUDA device.

    The backend uses CuPy sparse factorization for ``A - sigma B`` and CuPy's
    thick-restart Lanczos implementation on the resulting shift-invert operator.
    It intentionally supports Hermitian problems only; CuPy 14 does not expose
    a compatible sparse general Arnoldi ``eigs`` path.
    '''
    cp, cpsparse, cplinalg = _import_cupy()
    raw = validate_matrix(matrix)
    raw_mass = None if config.mass_matrix is None else validate_matrix(config.mass_matrix)
    if config.k < 1 or config.k >= raw.shape[0]:
        raise ValueError('invalid k for GPU matrix dimension')
    if raw_mass is not None and raw_mass.shape != raw.shape:
        raise ValueError(f'mass_matrix must have shape {raw.shape}, got {raw_mass.shape}')
    if config.mode == 'general' or (config.mode == 'auto' and classify_matrix(raw, config.hermitian_tol) != 'hermitian'):
        raise ValueError('GPU backend currently supports Hermitian matrices only')
    if config.inner_solver != 'direct':
        raise ValueError('GPU backend currently supports inner_solver=direct only')
    if np.iscomplexobj(config.sigma) and abs(complex(config.sigma).imag) > 0:
        raise ValueError('GPU Hermitian backend requires a real sigma')

    matrix_work = _enforce_hermitian(raw, max(config.hermitian_tol, 1e-12))
    mass_work = None if raw_mass is None else _enforce_hermitian(raw_mass, max(config.hermitian_tol, 1e-12))
    perm = None
    if config.reorder == 'rcm':
        perm = _rcm_permutation(matrix_work)
        matrix_work, _ = _apply_permutation(matrix_work, perm)
        if mass_work is not None:
            mass_work, _ = _apply_permutation(mass_work, perm)

    diagonal_scale = None
    if mass_work is not None:
        diagonal = np.asarray(mass_work.diagonal())
        if (mass_work - sparse.diags(diagonal, format='csr')).nnz != 0:
            raise ValueError('GPU generalized backend currently requires diagonal positive-definite B')
        if np.iscomplexobj(diagonal) and np.any(np.abs(diagonal.imag) > 1e-12):
            raise ValueError('GPU generalized backend requires a real positive-definite diagonal B')
        diagonal = diagonal.real
        if np.any(diagonal <= 0):
            raise ValueError('GPU generalized backend requires a positive-definite diagonal B')
        diagonal_scale = 1.0 / np.sqrt(diagonal)
        scale_matrix = sparse.diags(diagonal_scale, format='csr')
        matrix_work = (scale_matrix @ matrix_work @ scale_matrix).tocsr()
        mass_work = None

    started = time.perf_counter()
    norm_a = max(float(sparse.linalg.norm(matrix_work)), 1.0)
    step = config.shift_step if config.shift_step is not None else max(1e-8 * norm_a, 1e-12)
    attempts: list[dict[str, Any]] = []
    last_error: Exception | None = None
    for offset in _candidate_offsets(config, step):
        shift = float(np.real(config.sigma + offset))
        try:
            values, vectors, counts = _solve_shift(
                matrix_work, mass_work, config, shift, cp, cpsparse, cplinalg
            )
            if diagonal_scale is not None:
                vectors = diagonal_scale[:, np.newaxis] * vectors
            if perm is not None:
                vectors = vectors[np.argsort(perm), :]
            residuals = _residuals(raw, values, vectors, raw_mass)
            attempts.append({
                'shift': _json_scalar(shift),
                'status': 'ok',
                'max_residual': float(residuals.max()),
                **counts,
            })
            if np.all(residuals <= config.residual_tol):
                order = np.argsort(np.abs(values - config.sigma), kind='stable')
                values, vectors, residuals = values[order], vectors[:, order], residuals[order]
                break
            last_error = RuntimeError('computed GPU eigenpairs failed the requested residual tolerance')
            attempts[-1]['status'] = 'residual_too_large'
        except Exception as exc:
            last_error = exc
            attempts.append({'shift': _json_scalar(shift), 'status': 'error', 'error': str(exc)})
    else:
        raise RuntimeError(f'GPU eigensolver did not produce verified eigenpairs: {last_error}; attempts={attempts}')

    device = cp.cuda.Device()
    properties = cp.cuda.runtime.getDeviceProperties(device.id)
    gpu_name = properties['name']
    if isinstance(gpu_name, bytes):
        gpu_name = gpu_name.decode(errors='replace')
    diagnostics = {
        'backend': 'cupy-gpu',
        'gpu_device': int(device.id),
        'gpu_name': str(gpu_name),
        'matrix_shape': list(raw.shape),
        'matrix_nnz': int(raw.nnz),
        'matrix_dtype': str(raw.dtype),
        'problem_type': 'generalized' if raw_mass is not None else 'standard',
        'mass_matrix_shape': list(raw_mass.shape) if raw_mass is not None else None,
        'mass_matrix_nnz': int(raw_mass.nnz) if raw_mass is not None else None,
        'mass_matrix_dtype': str(raw_mass.dtype) if raw_mass is not None else None,
        'matrix_class': 'hermitian',
        'resolved_mode': 'hermitian',
        'k': config.k,
        'requested_sigma': _json_scalar(config.sigma),
        'effective_shift': attempts[-1]['shift'],
        'attempts': attempts,
        'n_attempts': len(attempts),
        'arpack_iterations': attempts[-1].get('arpack_iterations'),
        'inner_matvecs': attempts[-1].get('inner_matvecs'),
        'factor_kind': attempts[-1].get('factor_kind'),
        'factor_nnz': attempts[-1].get('factor_nnz'),
        'max_relative_residual': float(residuals.max()),
        'elapsed_seconds': time.perf_counter() - started,
        'solver': 'CuPy eigsh, CUDA sparse factorization, shift-invert Lanczos',
        'strategy': {'inner_solver': 'CUDA sparse direct', 'reorder': config.reorder},
    }
    return values, vectors, residuals, diagnostics
