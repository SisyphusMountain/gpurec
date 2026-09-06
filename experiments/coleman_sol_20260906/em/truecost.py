"""Recover the Newton gradient pass count and clade-weighted cost from one fit log."""
from __future__ import annotations

import argparse
import json
import re


CHECK = re.compile(r"\[pi(\d+) it(\d+)\] live=(\d+) .*clades: live ([0-9.]+)M of ([0-9.]+)M in model")
TIER = re.compile(r"\[fit_genewise\] tier pi=(\d+): (\d+) families")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True)
    args = parser.parse_args()
    tiers, current = [], None
    for line in open(args.log, errors="replace"):
        if TIER.search(line):
            current = []
            tiers.append(current)
        match = CHECK.search(line)
        if match and current is not None:
            current.append({"it": int(match.group(2)), "live": float(match.group(4)), "model": float(match.group(5))})
    if not tiers or not tiers[0]:
        raise ValueError("no fit tier/check records found")
    total = tiers[0][0]["model"]
    fractions = []
    for checks in tiers:
        if not checks:
            continue
        by_it = {row["it"]: row for row in checks}
        for iteration in range(max(by_it) + 1):
            next_check = min((key for key in by_it if key >= iteration), default=max(by_it))
            # A frozen family continues to cost gradient work until the model is
            # actually re-planned. Therefore use the resident MODEL clades, not
            # the smaller logical-live count printed on the same line.
            fractions.append(by_it[next_check]["model"] / total)
    print(json.dumps({
        "newton_gradient_passes": len(fractions),
        "newton_clade_weighted": sum(fractions),
        "fractions": fractions,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
