#!/usr/bin/env python
'''Run one sparse eigenproblem with a distributed PETSc/SLEPc backend.

This is the optional single-matrix MPI backend. It requires a matching MPI
installation plus petsc4py and slepc4py built with the same MPI implementation.
The root rank stages the input file and distributes CSR row blocks; matrix-vector
products, spectral transformation, and Krylov iterations are distributed by
PETSc/SLEPc. The root rank writes the normal project result archive.
'''

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from sparse_eigensolver.solver import _residuals, load_matrix, save_results  # noqa: E402


def _imports():
    try:
        from mpi4py import MPI
        import petsc4py
        petsc4py.init(sys.argv)
        from petsc4py import PETSc
        from slepc4py import SLEPc
    except ImportError as exc:
        raise SystemExit(
            'SLEPc backend requires mpi4py, petsc4py, and slepc4py built against the same MPI runtime'
        ) from exc
    return MPI, PETSc, SLEPc


def _row_partition(n: int, size: int, rank: int) -> tuple[int, int]:
    start = (n * rank) // size
    stop = (n * (rank + 1)) // size
    return start, stop


def _scatter_csr(matrix, communicator, rank: int, size: int):
    if rank == 0:
        n = matrix.shape[0]
        payload = []
        for worker in range(size):
            start, stop = _row_partition(n, size, worker)
            block = matrix[start:stop].tocsr()
            payload.append((start, stop, block.indptr.astype(np.int32), block.indices.astype(np.int32), block.data))
    else:
        n = None
        payload = None
    n = communicator.bcast(n, root=0)
    start, stop, indptr, indices, data = communicator.scatter(payload, root=0)
    return n, start, stop, indptr, indices, data


def _petsc_matrix(PETSc, n: int, start: int, stop: int, indptr, indices, data):
    local_rows = stop - start
    matrix = PETSc.Mat().createAIJ(
        size=((local_rows, n), (local_rows, n)),
        csr=(indptr, indices, data),
        comm=PETSc.COMM_WORLD,
    )
    matrix.assemble()
    return matrix


def _gather_vector(vector, communicator, start: int, stop: int, n: int):
    local = np.asarray(vector.getArray(readonly=True)).copy()
    pieces = communicator.gather((start, stop, local), root=0)
    if communicator.Get_rank() != 0:
        return None
    result = np.empty(n, dtype=local.dtype)
    for piece_start, piece_stop, piece in pieces:
        result[piece_start:piece_stop] = piece
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('matrix', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('-k', type=int, default=6)
    parser.add_argument('--sigma', type=float, default=0.0)
    parser.add_argument('--tol', type=float, default=1e-10)
    parser.add_argument('--maxiter', type=int, default=None)
    parser.add_argument('--ncv', type=int, default=None)
    parser.add_argument('--residual-tol', type=float, default=1e-8)
    parser.add_argument('--mode', choices=('hermitian', 'general'), default='hermitian')
    parser.add_argument('--triangular', choices=('none', 'auto', 'hermitian', 'symmetric'), default='none')
    parser.add_argument('--mass-matrix', type=Path, default=None)
    parser.add_argument('--mass-triangular', choices=('none', 'auto', 'hermitian', 'symmetric'), default='none')
    args = parser.parse_args()
    MPI, PETSc, SLEPc = _imports()
    communicator = MPI.COMM_WORLD
    rank = communicator.Get_rank()
    size = communicator.Get_size()

    matrix = load_matrix(args.matrix, triangular=args.triangular) if rank == 0 else None
    mass = load_matrix(args.mass_matrix, triangular=args.mass_triangular) if rank == 0 and args.mass_matrix else None
    if rank == 0 and mass is not None and mass.shape != matrix.shape:
        raise ValueError(f'mass matrix must have shape {matrix.shape}, got {mass.shape}')
    n, start, stop, indptr, indices, data = _scatter_csr(matrix, communicator, rank, size)
    if rank == 0 and mass is not None:
        mass_payload = []
        for worker in range(size):
            row_start, row_stop = _row_partition(n, size, worker)
            block = mass[row_start:row_stop].tocsr()
            mass_payload.append((block.indptr.astype(np.int32), block.indices.astype(np.int32), block.data))
    else:
        mass_payload = None
    mass_parts = communicator.scatter(mass_payload, root=0)
    A = _petsc_matrix(PETSc, n, start, stop, indptr, indices, data)
    B = None
    if args.mass_matrix:
        B = _petsc_matrix(PETSc, n, start, stop, mass_parts[0], mass_parts[1], mass_parts[2])

    eps = SLEPc.EPS().create(comm=PETSc.COMM_WORLD)
    if B is None:
        eps.setOperators(A)
        eps.setProblemType(SLEPc.EPS.ProblemType.HEP if args.mode == 'hermitian' else SLEPc.EPS.ProblemType.NHEP)
    else:
        eps.setOperators(A, B)
        eps.setProblemType(SLEPc.EPS.ProblemType.GHEP if args.mode == 'hermitian' else SLEPc.EPS.ProblemType.GNHEP)
    if args.ncv is None:
        eps.setDimensions(nev=args.k)
    else:
        eps.setDimensions(nev=args.k, ncv=args.ncv)
    if args.maxiter is None:
        eps.setTolerances(tol=args.tol)
    else:
        eps.setTolerances(tol=args.tol, max_it=args.maxiter)
    eps.setTarget(args.sigma)
    eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
    st = eps.getST()
    st.setType(SLEPc.ST.Type.SINVERT)
    st.setShift(args.sigma)
    eps.setFromOptions()
    eps.solve()

    converged = eps.getConverged()
    if converged < args.k:
        raise RuntimeError(f'SLEPc converged only {converged} of requested {args.k} eigenpairs')
    xr, xi = A.createVecs()
    values = []
    vectors = []
    for index in range(args.k):
        value = eps.getEigenpair(index, xr, xi)
        values.append(value)
        vectors.append(_gather_vector(xr, communicator, start, stop, n))
    if rank != 0:
        return
    values = np.asarray(values)
    vectors = np.column_stack(vectors)
    residuals = _residuals(matrix, values, vectors, mass)
    if np.any(residuals > args.residual_tol):
        raise RuntimeError(f'SLEPc result failed residual tolerance: {residuals.max():.3e}')
    diagnostics = {
        'backend': 'slepc-mpi',
        'distributed_single_matrix': True,
        'mpi_size': size,
        'matrix_shape': list(matrix.shape),
        'matrix_nnz': int(matrix.nnz),
        'problem_type': 'generalized' if mass is not None else 'standard',
        'mass_matrix_shape': list(mass.shape) if mass is not None else None,
        'mass_matrix_nnz': int(mass.nnz) if mass is not None else None,
        'resolved_mode': args.mode,
        'k': args.k,
        'requested_sigma': args.sigma,
        'effective_shift': args.sigma,
        'converged': int(converged),
        'max_relative_residual': float(residuals.max()),
        'solver': 'SLEPc EPS with PETSc distributed AIJ and shift-and-invert ST',
    }
    save_results(args.output, values, vectors, residuals, diagnostics)
    print(json.dumps(diagnostics, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
