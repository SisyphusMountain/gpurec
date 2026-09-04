"""Run the genewise recipe with explicit knob overrides and report phase times, steps, NLL, certified.

Usage: recipe_experiment.py --species S --families LIST --limit 200 --clade-budget 100000 --tag t \
          --trust 2.0 --adam-steps 5 --hessian-refresh 15 --init-curvature adam_bfgs --out-dir DIR
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time

import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", required=True); ap.add_argument("--families", required=True)
    ap.add_argument("--limit", required=True, type=int); ap.add_argument("--clade-budget", required=True, type=int)
    ap.add_argument("--tag", required=True); ap.add_argument("--out-dir", required=True)
    ap.add_argument("--trust", required=True, type=float); ap.add_argument("--adam-steps", required=True, type=int)
    ap.add_argument("--hessian-refresh", required=True, type=int)
    ap.add_argument("--init-curvature", required=True, choices=("adam_bfgs", "exact"))
    ap.add_argument("--mu", required=True, type=float); ap.add_argument("--trust-max", required=True, type=float)
    args = ap.parse_args()
    from gpurec.fit.genewise_fit import fit_genewise
    paths = [ln.strip() for ln in open(args.families) if ln.strip() and not ln.startswith("#")][: args.limit]
    t0 = time.perf_counter()
    res = fit_genewise(
        args.species, paths, device="cuda", dtype=None, certify=True, certify_curvature=False,
        min_drop=32, rebuild_frac=0.25, hessian_refresh=args.hessian_refresh,
        init_curvature=args.init_curvature, solver_options=None, config=None, verbose=True,
        init_log2_rates=(math.log2(0.01), math.log2(0.1), math.log2(0.01)),
        clade_budget=(None if args.clade_budget == 0 else args.clade_budget),
        stall_patience=120, trust=args.trust, adam_steps=args.adam_steps, mu=args.mu, trust_max=args.trust_max,
    )
    wall = time.perf_counter() - t0
    cert = {k: res.get(k) for k in ("converged", "bound_active", "unconverged", "interior_pd")}
    summary = dict(tag=args.tag, wall_s=wall, nll_bits=float(res["loss_bits"]), n_steps=int(res["n_steps"]),
                   n_builds=int(res["n_builds"]), adam_s=res["adam_seconds"], hessian_s=res["hessian_seconds"],
                   n_hessians=res["n_hessians"], newton_grad_s=res["newton_grad_seconds"],
                   verify_s=res.get("verify_seconds"), certificate={k: (int(v) if v is not None else None) for k, v in cert.items()},
                   trust=args.trust, adam_steps=args.adam_steps, hessian_refresh=args.hessian_refresh,
                   init_curvature=args.init_curvature, mu=args.mu, trust_max=args.trust_max)
    print("[recipe] " + json.dumps(summary, default=str), flush=True)
    torch.save({"theta": res["theta"].detach().cpu(), "paths": paths}, f"{args.out_dir}/{args.tag}.pt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
