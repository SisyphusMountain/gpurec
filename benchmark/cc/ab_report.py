"""Turn this task's result JSONs into the markdown tables of AB_ARCHAEA_HOGENOM.md.

Reads the files `run_genewise.py`, `run_global.py` and `xcheck_alerax.py` wrote into
`benchmark/cc/results/` and prints three tables: the genewise fits, the global fits, and the AleRax
cross-checks. Printing rather than writing keeps the prose of the report under human control -- the
numbers are pasted in from here so none of them is retyped by hand.

Usage: python benchmark/cc/ab_report.py --results-dir benchmark/cc/results
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def _load(directory: str, tag: str):
    path = os.path.join(directory, f"{tag}.json")
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        return json.load(handle)


def _cert(summary: dict, key: str) -> str:
    value = summary.get("certificate", {}).get(key)
    return "-" if value is None else str(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    args = parser.parse_args()
    d = args.results_dir

    genewise_rows = [
        ("archaea 5446 fam", "arch_old_gw", "arch_new_gw"),
        ("HOGENOM 1055 fam", "hog1055_old_gw", "hog1055_new_gw"),
        ("HOGENOM 10869 fam", None, "hogfull_new_gw"),
    ]
    print("### Genewise fits\n")
    print("| dataset | code | families | wall (s) | NLL (bits) | NLL (nats) | converged | "
          "bound-active | unconverged | max |Pg| | peak GiB |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for label, old_tag, new_tag in genewise_rows:
        for code, tag in (("old", old_tag), ("new", new_tag)):
            if tag is None:
                continue
            s = _load(d, tag)
            if s is None:
                print(f"| {label} | {code} | (not run) | | | | | | | | |")
                continue
            print(f"| {label} | {code} | {s['n_families']} | {s['wall_s']:.1f} | "
                  f"{s['nll_bits']:.3f} | {s['nll_nats']:.3f} | {_cert(s, 'converged')} | "
                  f"{_cert(s, 'bound_active')} | {_cert(s, 'unconverged')} | "
                  f"{float(s['certificate']['pg_max']):.3g} | {s['peak_gib']:.2f} |")

    print("\n### Global fits (one shared D/L/T for every family)\n")
    print("| dataset | code | families | wall (s) | D | L | T | NLL (bits) | NLL (nats) | "
          "steps | max |Pg| |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for label, old_tag, new_tag in (("archaea 5446 fam", "arch_old_gl", "arch_new_gl"),
                                    ("HOGENOM 1055 fam", "hog1055_old_gl", "hog1055_new_gl")):
        for code, tag in (("old", old_tag), ("new", new_tag)):
            s = _load(d, tag)
            if s is None:
                print(f"| {label} | {code} | (not run) | | | | | | | | |")
                continue
            print(f"| {label} | {code} | {s['n_families']} | {s['wall_s']:.1f} | "
                  f"{s['rate_D']:.6g} | {s['rate_L']:.6g} | {s['rate_T']:.6g} | "
                  f"{s['nll_bits']:.3f} | {s['nll_nats']:.3f} | {s['n_steps']} | "
                  f"{float(s['gnorm']):.3g} |")

    print("\n### AleRax cross-check (per-family log-likelihood, nats)\n")
    print("| fit | matched families | Pearson r | mean gpurec-AleRax | median | mean abs | "
          "max abs | gpurec total NLL | AleRax total NLL |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for tag in ("hog1055_old_gw", "hog1055_new_gw", "hogfull_new_gw"):
        path = os.path.join(d, f"xcheck_{tag}.json")
        if not os.path.exists(path):
            print(f"| {tag} | (not run) | | | | | | | |")
            continue
        with open(path) as handle:
            x = json.load(handle)
        print(f"| {x['label']} | {x['n_matched']} | {x['pearson_r']:.6f} | "
              f"{x['mean_signed_diff_nats']:+.4f} | {x['median_signed_diff_nats']:+.4f} | "
              f"{x['mean_abs_diff_nats']:.4f} | {x['max_abs_diff_nats']:.4f} | "
              f"{x['gpurec_total_nll_nats']:.0f} | {x['alerax_total_nll_nats']:.0f} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
