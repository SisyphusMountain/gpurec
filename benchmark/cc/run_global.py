"""Timed global-mode (one shared D/L/T rate for all families) fit driver.

Deliberately calls `gpurec.fit.dtl_fit.fit_dtl(..., "global")` with NOTHING but the species tree,
the family list and the device, so whichever checkout is on PYTHONPATH runs its OWN default recipe.
That is the point of this driver: the old and the new code must be compared at their own defaults,
and passing a config here would force one checkout's settings onto the other.

Writes <out_dir>/<tag>.json with the fitted rates (D, L, T -- linear, relative to speciation), the
total NLL in bits and nats, the final projected-gradient size and the wall time, and
<out_dir>/<tag>.pt with the fitted 3-vector theta (log2 rates) and the family paths.

Usage:
  python benchmark/cc/run_global.py --species S.nwk --families LIST.txt --limit 0 \
      --out-dir DIR --tag NAME
`--limit 0` means all families in the list; `--limit N` keeps the first N.
"""
from __future__ import annotations

import argparse
import builtins
import json
import os
import sys
import time

import torch

REPO_LEVEL_PRINT = builtins.print
_T0 = time.perf_counter()


def _timed_print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    REPO_LEVEL_PRINT(f"[{time.perf_counter() - _T0:9.2f}s]", *args, **kwargs)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--species", required=True)
    parser.add_argument("--families", required=True, help="listfile: one gene-tree path per line")
    parser.add_argument("--limit", required=True, type=int, help="0 = all families; N = first N")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()

    paths = [line.strip() for line in open(args.families)
             if line.strip() and not line.startswith("#")]
    if args.limit > 0:
        paths = paths[: args.limit]
    builtins.print = _timed_print
    print(f"[driver] global mode, {len(paths)} families, species={args.species}, "
          f"torch {torch.__version__}, gpu={torch.cuda.get_device_name(0)}")

    from gpurec.fit.dtl_fit import fit_dtl

    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    res = fit_dtl(args.species, paths, "global", device="cuda", verbose=True)
    wall = time.perf_counter() - start
    peak_gib = torch.cuda.max_memory_allocated() / 2**30
    rates = [float(x) for x in res["rates"].tolist()]
    summary = {
        "tag": args.tag, "mode": "global", "n_families": res["n_families"], "wall_s": wall,
        "nll_bits": res["nll_bits"], "nll_nats": res["nll_nats"],
        "theta_log2": [float(x) for x in res["theta"].tolist()],
        "rate_D": rates[0], "rate_L": rates[1], "rate_T": rates[2],
        "gnorm": res["gnorm"], "n_steps": res["n_steps"], "peak_gib": peak_gib,
    }
    print(f"[driver] DONE wall={wall:.1f}s nll_bits={res['nll_bits']:.3f} "
          f"nll_nats={res['nll_nats']:.3f} D={rates[0]:.6g} L={rates[1]:.6g} T={rates[2]:.6g} "
          f"steps={res['n_steps']} |Pg|={res['gnorm']:.3e} peak={peak_gib:.2f}GiB")
    os.makedirs(args.out_dir, exist_ok=True)
    with open(f"{args.out_dir}/{args.tag}.json", "w") as handle:
        json.dump(summary, handle, indent=2)
    torch.save({"theta": res["theta"], "paths": paths}, f"{args.out_dir}/{args.tag}.pt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
