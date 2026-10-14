#!/usr/bin/env python
'''Convert a common electromagnetic matrix exchange file to SciPy .npz.'''

from __future__ import annotations

import argparse
from pathlib import Path

from scipy import sparse
from scipy.io import mmread


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, help='Matrix Market .mtx/.mm file, optionally gzip-compressed')
    parser.add_argument('--output', type=Path, required=True, help='SciPy sparse .npz output')
    parser.add_argument('--triangular', choices=('none', 'auto', 'hermitian', 'symmetric'), default='none')
    args = parser.parse_args()

    from sparse_eigensolver.solver import _restore_triangular, validate_matrix

    matrix = sparse.csr_matrix(mmread(args.input))
    if args.triangular != 'none':
        matrix = _restore_triangular(matrix, args.triangular)
    matrix = validate_matrix(matrix)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(args.output, matrix)
    print(f'wrote {args.output} shape={matrix.shape} nnz={matrix.nnz} dtype={matrix.dtype}')


if __name__ == '__main__':
    main()
