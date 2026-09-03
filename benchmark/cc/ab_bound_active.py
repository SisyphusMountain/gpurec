"""Which families sit on a rate bound, and do they coincide with the tiny families?

`fit_genewise`'s certificate counts how many families ended with at least one of their three rates
pinned to the edge of the allowed rate box ([1e-6, 2.0] for the genewise recipe). A large count is
not by itself a problem -- a family with very few species carries almost no information about, say,
the transfer rate, so the fit walks that rate to the box edge and stops. This script says how much
of the count is explained that way, by crossing the pinned families with the families whose
representative gene tree covers fewer than `--min-species` species.

Usage:
  python benchmark/cc/ab_bound_active.py --theta FIT.pt --min-species 4 --min-rate 1e-6 \
      --max-rate 2.0 --edge-tolerance 1e-3
`--edge-tolerance` is in log2-rate units: a rate counts as pinned when it is within that many log2
units of a bound, which is the same `bound_active_eps` idea the fit's own certificate uses.
"""
from __future__ import annotations

import argparse
import math
import os
import re
import sys

import torch

_SPECIES = re.compile(r"[(,]([A-Za-z][A-Za-z0-9]*)_")


def _covered_species(path: str) -> int:
    with open(path) as handle:
        first = handle.readline()
        line = handle.readline() if first.startswith("#") else first
    return len(set(_SPECIES.findall(line)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--theta", required=True)
    parser.add_argument("--min-species", required=True, type=int)
    parser.add_argument("--min-rate", required=True, type=float)
    parser.add_argument("--max-rate", required=True, type=float)
    parser.add_argument("--edge-tolerance", required=True, type=float)
    args = parser.parse_args()

    payload = torch.load(args.theta, map_location="cpu")
    theta = payload["theta"].double()
    paths = [str(p) for p in payload["paths"]]
    lo, hi = math.log2(args.min_rate), math.log2(args.max_rate)
    at_low = (theta <= lo + args.edge_tolerance).any(dim=1)
    at_high = (theta >= hi - args.edge_tolerance).any(dim=1)
    pinned = at_low | at_high
    thin = torch.tensor([_covered_species(p) < args.min_species for p in paths])

    n = len(paths)
    print(f"[bounds] {os.path.basename(args.theta)}: {n} families", flush=True)
    print(f"[bounds] pinned to a rate bound: {int(pinned.sum())} "
          f"(at the low bound {int(at_low.sum())}, at the high bound {int(at_high.sum())})", flush=True)
    print(f"[bounds] fewer than {args.min_species} covered species: {int(thin.sum())}", flush=True)
    print(f"[bounds] pinned AND tiny: {int((pinned & thin).sum())}   "
          f"pinned but NOT tiny: {int((pinned & ~thin).sum())}   "
          f"tiny but NOT pinned: {int((~pinned & thin).sum())}", flush=True)
    for column, name in enumerate(("D", "L", "T")):
        low = int((theta[:, column] <= lo + args.edge_tolerance).sum())
        high = int((theta[:, column] >= hi - args.edge_tolerance).sum())
        print(f"[bounds]   {name}: {low} at the low bound, {high} at the high bound", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
