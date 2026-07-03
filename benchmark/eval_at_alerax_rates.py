#!/usr/bin/env python
"""Evaluate gpurec per-family logL at AleRax's fitted GLOBAL rates and report fidelity.

Reads an AleRax run directory (model_parameters.txt + per_fam_likelihoods.txt), runs
gpurec at those rates over the given families, and prints matched count, mean/RMSE/max
|Δ| (nats) and Pearson r.
"""
import argparse
import sys
from pathlib import Path


def build_parser():
    p = argparse.ArgumentParser(description="gpurec-vs-AleRax global fidelity")
    p.add_argument("--species", required=True)
    p.add_argument("--gene", required=True, nargs="+")
    p.add_argument("--alerax-dir", required=True,
                   help="AleRax output dir with model_parameters/ and per_fam_likelihoods.txt")
    p.add_argument("--device", default="cuda")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    from gpurec.bench.alerax_io import (parse_alerax_likelihoods, parse_alerax_parameters,
                                        global_rates)
    from gpurec.bench.fidelity import reconcile_at_alerax_rates, compare
    d = Path(args.alerax_dir)
    rates = global_rates(parse_alerax_parameters(d / "model_parameters" / "model_parameters.txt"))
    alerax_ll = parse_alerax_likelihoods(d / "per_fam_likelihoods.txt")
    gpurec_ll = {}
    for g in args.gene:
        gpurec_ll.update(reconcile_at_alerax_rates(args.species, [g], rates, device=args.device))
    rep = compare(gpurec_ll, alerax_ll)
    print(f"matched={rep.n_matched} mean|Δ|={rep.mean_abs_delta:.3e} rmse={rep.rmse:.3e} "
          f"max|Δ|={rep.max_abs_delta:.3e} pearson_r={rep.pearson_r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
