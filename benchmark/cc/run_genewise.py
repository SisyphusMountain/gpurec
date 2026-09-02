"""Timed genewise fit driver for the Coleman 1007-leaf / 5124-family benchmark on the cluster.

Runs exactly the production recipe (`gpurec.fit.dtl_fit.fit_dtl`, mode="genewise", certify on) with
verbose progress, prefixes every printed line with elapsed seconds, and writes:
  <out_dir>/<tag>.json   headline numbers (NLL bits/nats, wall time, steps, builds, certificate)
  <out_dir>/<tag>.pt     fitted theta [F,3] (log2 rates) + the family paths, for likelihood comparison

Usage:
  python benchmark/cc/run_genewise.py --species S.nwk --families LIST.txt --limit 0 --out-dir DIR --tag NAME
`--limit 0` means all families in the list; `--limit N` keeps the first N (for smoke/profiling runs).
"""
from __future__ import annotations

import argparse
import builtins
import json
import sys
import time

import torch

REPO_LEVEL_PRINT = builtins.print
_T0 = time.perf_counter()


def _timed_print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    REPO_LEVEL_PRINT(f"[{time.perf_counter() - _T0:9.2f}s]", *args, **kwargs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", required=True)
    ap.add_argument("--families", required=True, help="listfile: one .ale path per line")
    ap.add_argument("--limit", required=True, type=int, help="0 = all families; N = first N of the list")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    paths = [ln.strip() for ln in open(args.families) if ln.strip() and not ln.startswith("#")]
    if args.limit > 0:
        paths = paths[: args.limit]
    builtins.print = _timed_print
    print(f"[driver] {len(paths)} families, species={args.species}, torch {torch.__version__}, "
          f"gpu={torch.cuda.get_device_name(0)}")

    from gpurec.fit.dtl_fit import fit_dtl

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    res = fit_dtl(args.species, paths, "genewise", device="cuda", verbose=True)
    wall = time.perf_counter() - t0
    peak_gib = torch.cuda.max_memory_allocated() / 2**30
    g = res["genewise_result"]
    summary = {
        "tag": args.tag, "n_families": res["n_families"], "wall_s": wall,
        "nll_bits": res["nll_bits"], "nll_nats": res["nll_nats"],
        "opt_seconds": g["opt_seconds"], "n_steps": g["n_steps"], "n_builds": g["n_builds"],
        # rebatching cost split: candidate verification vs re-planning the model over the survivors
        "n_verify_builds": g["n_verify_builds"], "verify_seconds": g["verify_seconds"],
        "n_rebuilds": g["n_rebuilds"], "rebuild_seconds": g["rebuild_seconds"],
        "history": g["history"], "peak_gib": peak_gib,
        "certificate": {k: g[k] for k in ("converged", "interior_pd", "bound_active", "unconverged",
                                           "premature_drops", "pg_max") if k in g},
    }
    print(f"[driver] DONE wall={wall:.1f}s nll_bits={res['nll_bits']:.3f} nll_nats={res['nll_nats']:.3f} "
          f"steps={g['n_steps']} builds={g['n_builds']} peak={peak_gib:.2f}GiB "
          f"verify={g['n_verify_builds']}x/{g['verify_seconds']:.0f}s "
          f"rebuild={g['n_rebuilds']}x/{g['rebuild_seconds']:.0f}s cert={summary['certificate']}")
    import os
    os.makedirs(args.out_dir, exist_ok=True)
    with open(f"{args.out_dir}/{args.tag}.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    torch.save({"theta": res["theta"], "paths": paths}, f"{args.out_dir}/{args.tag}.pt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
