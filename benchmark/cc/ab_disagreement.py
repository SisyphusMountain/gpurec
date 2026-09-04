"""Who are the families where the two fits disagree, and are they the uninformative ones?

`compare_fit_thetas.py` says how big the per-family likelihood gap between two fits is. This says
WHICH families carry it: it splits the families by whether their gap exceeds `--threshold` bits and
reports, for each side of that split, how many cover fewer than `--min-species` species and how many
ended with a rate pinned to the edge of the allowed rate box in either fit.

The point is to separate two very different situations. If the families that disagree are the ones
with almost no data -- a handful of sequences from two or three organisms -- then the two fits are
landing in different, nearly equally good places on a flat likelihood surface, which is a property
of the data, not a regression. If instead the disagreement sits on ordinary, well-covered families,
that is a real difference in the optimizer.

Usage:
  python benchmark/cc/ab_disagreement.py --cmp CMP_NLL.pt --left LEFT.pt --right RIGHT.pt \
      --threshold 0.01 --min-species 4 --min-rate 1e-6 --max-rate 2.0 --edge-tolerance 1e-3
"""
from __future__ import annotations

import argparse
import math
import re
import sys

import torch

_SPECIES = re.compile(r"[(,]([A-Za-z][A-Za-z0-9]*)_")


def _covered_species(path: str) -> int:
    with open(path) as handle:
        first = handle.readline()
        line = handle.readline() if first.startswith("#") else first
    return len(set(_SPECIES.findall(line)))


def _pinned(theta: torch.Tensor, lo: float, hi: float, tolerance: float) -> torch.Tensor:
    return ((theta <= lo + tolerance) | (theta >= hi - tolerance)).any(dim=1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmp", required=True, help="the .cmp_nll.pt compare_fit_thetas.py wrote")
    parser.add_argument("--left", required=True, help="left run_genewise .pt (holds the paths)")
    parser.add_argument("--right", required=True)
    parser.add_argument("--threshold", required=True, type=float, help="bits; above this = 'disagrees'")
    parser.add_argument("--min-species", required=True, type=int)
    parser.add_argument("--min-rate", required=True, type=float)
    parser.add_argument("--max-rate", required=True, type=float)
    parser.add_argument("--edge-tolerance", required=True, type=float)
    args = parser.parse_args()

    cmp_payload = torch.load(args.cmp, map_location="cpu")
    difference = (cmp_payload["right_nll"].double() - cmp_payload["left_nll"].double())
    left = torch.load(args.left, map_location="cpu")
    right = torch.load(args.right, map_location="cpu")
    paths = [str(p) for p in left["paths"]]
    by_path = {str(p): i for i, p in enumerate(right["paths"])}
    right_theta = torch.stack([right["theta"][by_path[p]] for p in paths]).double()
    left_theta = left["theta"].double()

    lo, hi = math.log2(args.min_rate), math.log2(args.max_rate)
    pinned_either = (_pinned(left_theta, lo, hi, args.edge_tolerance)
                     | _pinned(right_theta, lo, hi, args.edge_tolerance))
    species = torch.tensor([_covered_species(p) for p in paths])
    thin = species < args.min_species
    disagrees = difference.abs() > args.threshold

    total = len(paths)
    print(f"[split] {total} families, threshold {args.threshold} bits", flush=True)
    for label, mask in (("disagree", disagrees), ("agree", ~disagrees)):
        n = int(mask.sum())
        if n == 0:
            continue
        share_thin = int((mask & thin).sum())
        share_pinned = int((mask & pinned_either).sum())
        median_species = float(species[mask].double().median())
        print(f"[split] {label}: {n} families ({100.0 * n / total:.1f}%) -- "
              f"{share_thin} ({100.0 * share_thin / n:.1f}%) cover < {args.min_species} species, "
              f"{share_pinned} ({100.0 * share_pinned / n:.1f}%) have a rate pinned to a bound in "
              f"at least one fit, median covered species {median_species:.0f}", flush=True)
    print(f"[split] the {int(disagrees.sum())} disagreeing families carry "
          f"{float(difference[disagrees].sum()):+.3f} bits of the "
          f"{float(difference.sum()):+.3f} bit total", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
