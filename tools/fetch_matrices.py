#!/usr/bin/env python3
"""Download public electromagnetic sparse matrices and convert them to .npz.

Sources: SuiteSparse Matrix Collection (https://sparse.tamu.edu), which aggregates
matrices from public academic collections. Only real-symmetric (hence Hermitian)
matrices are selected so they fit the target-shift Hermitian eigensolver.

Usage:
    python tools/fetch_matrices.py --list
    python tools/fetch_matrices.py --out data/raw --names Bai/mhdb416 Bai/mhd4800b
"""

from __future__ import annotations

import argparse
from pathlib import Path
import tarfile
import tempfile
import urllib.request

import numpy as np
from scipy import sparse
from scipy.io import mmread

BASE_URL = "https://sparse.tamu.edu/MM"

CATALOG: dict[str, dict[str, object]] = {
    "Bai/bfwb782": {"n": 782, "nnz": 5982, "note": "electromagnetics, real symmetric, indefinite"},
    "Bai/mhdb416": {"n": 416, "nnz": 2312, "note": "electromagnetics (MHD), real symmetric"},
    "Bai/mhd4800b": {"n": 4800, "nnz": 27520, "note": "electromagnetics (MHD), real symmetric"},
    "Bai/dw8192": {"n": 8192, "nnz": 41746, "note": "electromagnetics, real symmetric, indefinite"},
    "TOKAMAK/utm5940": {"n": 5940, "nnz": 83842, "note": "electromagnetics (tokamak), real symmetric, indefinite"},
    "Oberwolfach/gyro": {"n": 17361, "nnz": 1021159, "note": "model reduction / gyroscope, real symmetric, indefinite"},
    "Li/pli": {"n": 22695, "nnz": 1350309, "note": "electromagnetics, real symmetric, indefinite"},
    "FEMLAB/waveguide3D": {"n": 21036, "nnz": 303468, "note": "waveguide, complex general (non-Hermitian)"},
    "CEMW/vfem": {"n": 93476, "nnz": 1434636, "note": "vector FEM electromagnetics, complex Hermitian, ~100k"},
    "Um/2cubes_sphere": {"n": 101492, "nnz": 1647264, "note": "electromagnetic scattering, real symmetric, ~100k"},
    "Guettel/TEM152078": {"n": 152078, "nnz": 6459326, "note": "electromagnetics, real symmetric, ~150k"},
    "Dziekonski/dielFilterV3clx": {"n": 420408, "nnz": 32886208, "note": "dielectric filter, complex symmetric (non-Hermitian), lossy/PML EM"},
}


def download_matrix(group_and_name: str, destination: Path, symmetrize: bool = False, hermitian_tol: float = 0.0, keep_general: bool = False) -> Path:
    url = f"{BASE_URL}/{group_and_name}.tar.gz"
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (matrix-fetch script)"})
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "matrix.tar.gz"
        with urllib.request.urlopen(request, timeout=180) as response, open(archive, "wb") as handle:
            handle.write(response.read())
        with tarfile.open(archive, "r:gz") as tar:
            members = [m for m in tar.getmembers() if m.name.endswith(".mtx")]
            if not members:
                raise RuntimeError(f"no .mtx file inside {url}")
            tar.extract(members[0], path=tmp, filter="data")
            matrix = mmread(Path(tmp) / members[0].name)
    matrix = sparse.coo_matrix(matrix).tocsr()
    delta = matrix - matrix.getH()
    if delta.nnz:
        scale = max(float(np.linalg.norm(matrix.data)), 1.0)
        relative = float(np.linalg.norm(delta.data)) / scale
        if relative <= hermitian_tol:
            print(f"  note: {group_and_name} is Hermitian within {hermitian_tol:.1e} (relative asymmetry {relative:.3e}); enforcing exact Hermitian form")
            matrix = ((matrix + matrix.getH()) * 0.5).tocsr()
        elif keep_general:
            print(f"  note: keeping {group_and_name} as a general (non-Hermitian) matrix (relative asymmetry {relative:.3e})")
        elif not symmetrize:
            raise RuntimeError(
                f"{group_and_name} is not Hermitian (relative asymmetry {relative:.3e}); "
                "pass --keep-general to store it as-is for the non-Hermitian solver, "
                "--symmetrize to force (A + A^H) / 2, or --hermitian-tol to accept near-Hermitian input"
            )
        else:
            print(f"  warning: symmetrizing {group_and_name} (relative asymmetry {relative:.3e})")
            matrix = ((matrix + matrix.getH()) * 0.5).tocsr()
    destination.parent.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(destination, matrix)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch public electromagnetic sparse matrices.")
    parser.add_argument("--out", type=Path, default=Path("data/raw"))
    parser.add_argument("--names", nargs="*", help="Matrix ids like Bai/mhdb416")
    parser.add_argument("--symmetrize", action="store_true", help="Force (A + A^H) / 2 when a matrix is not exactly Hermitian")
    parser.add_argument("--hermitian-tol", type=float, default=0.0, help="Accept and exactly symmetrize matrices Hermitian within this relative tolerance")
    parser.add_argument("--keep-general", action="store_true", help="Store non-Hermitian matrices as-is for the non-Hermitian solver path")
    parser.add_argument("--list", action="store_true", help="List the curated catalog")
    args = parser.parse_args()

    if args.list:
        for name, meta in CATALOG.items():
            print(f"{name:24s} n={meta['n']:>7} nnz={meta['nnz']:>9}  {meta['note']}")
        return

    names = args.names or list(CATALOG)
    failures: list[str] = []
    for name in names:
        if name not in CATALOG:
            failures.append(name)
            print(f"skip {name}: not in the curated catalog (see --list)")
            continue
        target = args.out / (name.replace("/", "__") + ".npz")
        print(f"downloading {name} -> {target}")
        try:
            download_matrix(name, target, symmetrize=args.symmetrize, hermitian_tol=args.hermitian_tol, keep_general=args.keep_general)
        except (RuntimeError, OSError) as exc:
            failures.append(name)
            print(f"  failed: {exc}")
            continue
        print(f"  saved {target} ({target.stat().st_size / 1e6:.1f} MB)")
    if failures:
        print(f"\n{len(failures)} matrix(es) not saved: {', '.join(failures)}")


if __name__ == "__main__":
    main()
