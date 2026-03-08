#!/usr/bin/env python3
"""
Compute the log-likelihood for a single gene family using the current
fixed-point implementation (E_step loop + Pi_fixed_point).

Example:
  python scripts/compute_likelihood.py \
    --species /home/enzo/Documents/git/WP2/gpurec/tests/data/test_mixed_200/sp.nwk \
    --genes   /home/enzo/Documents/git/WP2/gpurec/tests/data/test_mixed_200/g.nwk \
    --dtype float64 --device cpu

Notes:
  - By default, uses the first sample (index 0) from GeneDataset.
  - Uses the theta stored in the dataset for that family.
  - Runs a simple Picard iteration for E (via E_step) and then Pi_fixed_point.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import math
import sys
from typing import List

import torch


def _resolve_dtype(name: str) -> torch.dtype:
    attr = name.lower()
    if not hasattr(torch, attr):
        raise ValueError(f"Unsupported dtype '{name}'")
    dtype = getattr(torch, attr)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Attribute '{name}' is not a torch dtype")
    return dtype


def _resolve_device(name: str | None) -> torch.device:
    if name is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute log-likelihood for one gene family.")
    parser.add_argument("--species", required=True, help="Path to species tree (Newick)")
    parser.add_argument(
        "--genes",
        required=True,
        nargs="+",
        help="One or more gene tree paths (Newick). First item used by default.",
    )
    parser.add_argument("--index", type=int, default=0, help="Dataset index to use (default 0)")
    parser.add_argument("--dtype", type=str, default="float64", help="torch dtype, e.g. float32/float64")
    parser.add_argument("--device", type=str, default=None, help="cuda/cpu (default: auto)")
    parser.add_argument("--genewise", action="store_true", help="Enable genewise parameters")
    parser.add_argument("--specieswise", action="store_true", help="Enable specieswise parameters")
    parser.add_argument("--pairwise", action="store_true", help="Enable pairwise transfer coefficients")
    parser.add_argument("--max-iters-E", type=int, default=2000)
    parser.add_argument("--tol-E", type=float, default=1e-9)
    parser.add_argument("--max-iters-Pi", type=int, default=2000)
    parser.add_argument("--tol-Pi", type=float, default=1e-9)

    args = parser.parse_args()

    # ensure src/ on sys.path to avoid importing src/__init__.py side effects
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from core.model import GeneDataset

    device = _resolve_device(args.device)
    dtype = _resolve_dtype(args.dtype)

    # Build dataset (kept small/simple per user request)
    dataset = GeneDataset(
        species_tree_path=args.species,
        gene_tree_paths=args.genes,
        genewise=args.genewise,
        specieswise=args.specieswise,
        pairwise=args.pairwise,
        dtype=dtype,
        device=device,
    )

    if not (0 <= args.index < len(dataset)):
        raise IndexError(f"--index {args.index} out of range [0, {len(dataset)-1}]")

    res = dataset.compute_likelihood(
        idx=args.index,
        max_iters_E=args.max_iters_E,
        tol_E=args.tol_E,
        max_iters_Pi=args.max_iters_Pi,
        tol_Pi=args.tol_Pi,
        device=device,
        dtype=dtype,
    )
    print(f"log-likelihood: {res['log_likelihood']:.12g}")


if __name__ == "__main__":
    main()
