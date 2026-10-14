#!/usr/bin/env python
"""Export the combined result archive into separate NumPy arrays."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy import sparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path, help="Combined solver result archive (.npz)")
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory for separate .npy files")
    parser.add_argument("--prefix", default=None, help="Output filename prefix; defaults to the archive stem")
    parser.add_argument("--scipy-vectors", action="store_true", help="Also write eigenvectors as a SciPy sparse .npz")
    args = parser.parse_args()

    prefix = args.prefix or args.result.stem
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with np.load(args.result, allow_pickle=False) as result:
        required = {"eigenvalues", "eigenvectors", "residuals"}
        missing = required - set(result.files)
        if missing:
            raise ValueError(f"result archive is missing fields: {sorted(missing)}")
        for field in ("eigenvalues", "eigenvectors", "residuals"):
            np.save(args.out_dir / f"{prefix}_{field}.npy", result[field])
        if args.scipy_vectors:
            sparse.save_npz(args.out_dir / f"{prefix}_eigenvectors_scipy.npz", sparse.csr_matrix(result["eigenvectors"]))

    print(f"wrote separate NumPy arrays under {args.out_dir}")


if __name__ == "__main__":
    main()
