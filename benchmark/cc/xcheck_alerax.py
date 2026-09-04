"""Cross-check a gpurec fit's per-family log-likelihood against AleRax's reported per-family values.

AleRax writes one line per family, `<family> <logL in nats>` (a negative number: the higher, the
better). gpurec scores a family as an NLL in bits (log2), so the comparison converts
`logL_nats = -nll_bits * ln 2` and keeps only the families that appear in BOTH sets.

Reported: how many families matched, the Pearson correlation of the two per-family log-likelihood
vectors, the mean signed difference (gpurec minus AleRax, in nats per family -- positive means
gpurec assigns the family a HIGHER likelihood), the mean and max absolute difference, and the two
totals so the head-to-head margin can be read directly.

This is CPU-only; it reads tensors already written by `score_per_family.py` or
`compare_fit_thetas.py` and never touches a GPU.

Usage:
  python benchmark/cc/xcheck_alerax.py --label new1055 \
      --nll-source scored.pt --nll-key nll_bits --paths-source scored.pt \
      --name-rule family-dir --alerax-likelihoods alerax.txt --out out.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import torch

_LN2 = math.log(2.0)


def _family_name(path: str, rule: str) -> str:
    """`basename-noext`: archaea, `<dir>/fix_1000_c60_lg_combined.ale` -> `fix_1000_c60_lg_combined`.
    `family-dir`: HOGENOM, `<root>/<FAMILY>/gene_trees/<tree>.newick` -> `<FAMILY>`."""
    if rule == "basename-noext":
        return os.path.splitext(os.path.basename(path))[0]
    return os.path.basename(os.path.dirname(os.path.dirname(path)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--nll-source", required=True, help=".pt holding the per-family NLL vector")
    parser.add_argument("--nll-key", required=True, help="tensor key inside --nll-source")
    parser.add_argument("--paths-source", required=True, help=".pt holding the 'paths' list")
    parser.add_argument("--name-rule", required=True, choices=("basename-noext", "family-dir"))
    parser.add_argument("--alerax-likelihoods", required=True, help="'<family> <logL nats>' per line")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    nll_bits = torch.load(args.nll_source, map_location="cpu")[args.nll_key].double()
    paths = [str(p) for p in torch.load(args.paths_source, map_location="cpu")["paths"]]
    if len(paths) != nll_bits.numel():
        raise ValueError(f"{len(paths)} paths but {nll_bits.numel()} NLL entries")

    alerax = {}
    for line in open(args.alerax_likelihoods):
        parts = line.split()
        if len(parts) >= 2:
            alerax[parts[0]] = float(parts[-1])

    gpurec_logl, alerax_logl, matched_names, unmatched = [], [], [], []
    for index, path in enumerate(paths):
        name = _family_name(path, args.name_rule)
        if name in alerax:
            matched_names.append(name)
            gpurec_logl.append(-float(nll_bits[index]) * _LN2)
            alerax_logl.append(alerax[name])
        else:
            unmatched.append(name)

    g = torch.tensor(gpurec_logl, dtype=torch.float64)
    a = torch.tensor(alerax_logl, dtype=torch.float64)
    d = g - a
    gc, ac = g - g.mean(), a - a.mean()
    pearson = float((gc * ac).sum() / (gc.norm() * ac.norm()))
    worst = int(d.abs().argmax())
    summary = {
        "label": args.label,
        "n_gpurec_families": len(paths),
        "n_alerax_families": len(alerax),
        "n_matched": int(g.numel()),
        "n_gpurec_not_in_alerax": len(unmatched),
        "pearson_r": pearson,
        "mean_signed_diff_nats": float(d.mean()),
        "median_signed_diff_nats": float(d.median()),
        "mean_abs_diff_nats": float(d.abs().mean()),
        "max_abs_diff_nats": float(d.abs().max()),
        "max_abs_diff_family": matched_names[worst],
        "gpurec_total_nll_nats": float(-g.sum()),
        "alerax_total_nll_nats": float(-a.sum()),
        "gpurec_minus_alerax_total_logl_nats": float(d.sum()),
    }
    for key, value in summary.items():
        print(f"[xcheck {args.label}] {key} = {value}", flush=True)
    with open(args.out, "w") as handle:
        json.dump(summary, handle, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
