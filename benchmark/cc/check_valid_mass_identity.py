"""Check, on the real species tree, that the valid transfer mass splits into two non-negative sums.

The forward kernels build a donor's valid receiver mass as
``total_receiver_mass - excluded_ancestor_mass``: two nearly equal numbers whose difference is
what we want. This checks the algebraic replacement that has no subtraction at all. With the
depth-first interval numbering (``start`` is a permutation of ``0..S-1``, every node's subtree
occupies ``[start, end)``), a node ``a`` is an ancestor-or-self of ``s`` exactly when
``start[a] <= start[s] < end[a]``, so the NON-ancestors split into two disjoint groups:

    already closed   ``end[a] <= start[s]``
    not yet opened   ``start[a] > start[s]``

and the valid mass is the sum of those two groups -- non-negative terms only.

CPU only; no GPU needed. Usage:
  python benchmark/cc/check_valid_mass_identity.py --species S.nwk --families LIST --trials 5
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import torch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--species", required=True)
    parser.add_argument("--families", required=True)
    parser.add_argument("--trials", required=True, type=int)
    args = parser.parse_args()

    from gpurec.core.scheduling.batching import preprocess_dataset

    paths = [
        line.strip() for line in open(args.families) if line.strip() and not line.startswith("#")
    ][:2]
    helpers = preprocess_dataset(args.species, paths, clade_budget=315000)["species"]
    S = int(helpers["S"])
    parent = np.asarray(helpers["sp_parent"], dtype=np.int64)
    start = np.asarray(helpers["sp_subtree_start"], dtype=np.int64)
    end = np.asarray(helpers["sp_subtree_end"], dtype=np.int64)
    print(f"S={S}  start is a permutation: {sorted(start.tolist()) == list(range(S))}  "
          f"end in [{end.min()}, {end.max()}]")

    # Host tables the kernel would carry.
    euler_node = np.empty(S, dtype=np.int64)
    euler_node[start] = np.arange(S)                       # node sitting at each start position
    reverse_euler = euler_node[::-1].copy()                # so a forward scan gives a suffix sum
    end_order = np.argsort(end, kind="stable")             # nodes ordered by where their subtree ends
    close_count = np.searchsorted(end[end_order], np.arange(S), side="right")

    generator = np.random.default_rng(0)
    worst = 0.0
    for trial in range(args.trials):
        mass = generator.random(S) * np.exp(generator.normal(scale=6.0, size=S))
        # Reference: for each s, sum over every node that is NOT s and NOT an ancestor of s.
        reference = np.empty(S)
        for s in range(S):
            chain = []
            node = s
            while node >= 0:
                chain.append(node)
                node = parent[node]
            reference[s] = mass.sum() - mass[chain].sum()
        # Replacement: two disjoint non-negative groups.
        not_yet_open = np.concatenate(([0.0], np.cumsum(mass[reverse_euler])))  # exclusive prefix
        already_closed = np.concatenate(([0.0], np.cumsum(mass[end_order])))    # exclusive prefix
        candidate = (
            not_yet_open[S - 1 - start] + already_closed[close_count[start]]
        )
        relative = np.abs(candidate - reference) / np.maximum(np.abs(reference), 1e-300)
        worst = max(worst, float(relative.max()))
        print(f"trial {trial}: max relative difference {relative.max():.3e}")
    print(f"worst over {args.trials} trials: {worst:.3e}")
    return 0 if worst < 1e-9 else 1


if __name__ == "__main__":
    sys.exit(main())
