"""Run the genewise recipe with explicit knob overrides and report phase times, steps, NLL, certified.

Usage: recipe_experiment.py --species S --families LIST --limit 200 (0 = all) --clade-budget 100000 --tag t \
          --trust 2.0 --adam-steps 5 --hessian-refresh 15 --init-curvature adam_bfgs \
          --curvature-update bfgs --dtype float32 --out-dir DIR
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
    ap.add_argument("--curvature-update", required=True, choices=("bfgs", "sr1", "multisecant"))
    ap.add_argument("--mu", required=True, type=float); ap.add_argument("--trust-max", required=True, type=float); ap.add_argument("--adam-lr", required=True, type=float)
    ap.add_argument("--rebuild-frac", required=True, type=float); ap.add_argument("--check-every", required=True, type=int)
    ap.add_argument("--tol", required=True, type=float)
    # Model dtype. float32 is the production precision; float64 is a measurement mode -- it makes the
    # gradient exact to ~1e-16 instead of ~1e-3, which is what tells apart "the end game is limited by
    # float32 gradient noise" from "the end game is limited by the quality of the carried curvature".
    ap.add_argument("--dtype", required=True, choices=("float32", "float64"))
    # Round-5 step experiments (see fit_genewise's documentation). Each has an explicit "off" value
    # that reproduces the measured recipe bit for bit: 1.0 / quadratic / 0.0 / 0.0.
    ap.add_argument("--step-extrapolation", required=True, type=float)
    ap.add_argument("--step-model", required=True, choices=("quadratic", "rate_affine"))
    ap.add_argument("--stop-nll-bits", required=True, type=float)
    ap.add_argument("--approach-pruning-threshold", required=True, type=float)
    # Round-6 experiments (see fit_genewise's documentation). The off values that reproduce the
    # measured recipe bit for bit are: --stuck-from 0 --stuck-max-frac 0.0 (no targeted exact
    # Hessian), --stage-freeze-t 0 --stage-d-only 0 (no coordinate staging), and the ratio test's
    # own four measured numbers 0.25 / 0.75 / 0.5 / 0.05.
    ap.add_argument("--stuck-from", required=True, type=int,
                    help="Newton step from which stuck live families may get a targeted exact Hessian")
    ap.add_argument("--stuck-max-frac", required=True, type=float,
                    help="largest share of the live model's clades a targeted Hessian may cover (0 = off)")
    ap.add_argument("--stage-freeze-t", required=True, type=int,
                    help="Newton steps during which the transfer rate is held fixed")
    ap.add_argument("--stage-d-only", required=True, type=int,
                    help="Newton steps during which only the duplication rate moves")
    ap.add_argument("--trust-shrink", required=True, type=float,
                    help="trust-radius multiplier after a step the ratio test judged badly (0.25 = today)")
    ap.add_argument("--trust-grow-ratio", required=True, type=float,
                    help="actual/predicted decrease above which a radius-capped step doubles the radius (0.75 = today)")
    ap.add_argument("--trust-radius-min", required=True, type=float,
                    help="floor on the trust radius, in log2 rate units (0.5 = today)")
    ap.add_argument("--trust-min-predicted-bits", required=True, type=float,
                    help="predicted decrease below which a step is too small to judge, in bits (0.05 = today)")
    args = ap.parse_args()
    from gpurec.fit.genewise_fit import fit_genewise
    paths = [ln.strip() for ln in open(args.families) if ln.strip() and not ln.startswith("#")]
    if args.limit > 0:   # 0 = every family in the list, as in run_genewise.py
        paths = paths[: args.limit]
    t0 = time.perf_counter()
    res = fit_genewise(
        args.species, paths, device="cuda", dtype=args.dtype, certify=True, certify_curvature=False,
        min_drop=32, rebuild_frac=args.rebuild_frac, check_every=args.check_every, hessian_refresh=args.hessian_refresh, tol=args.tol,
        init_curvature=args.init_curvature, curvature_update=args.curvature_update,
        solver_options=None, config=None, verbose=True,
        init_log2_rates=(math.log2(0.01), math.log2(0.1), math.log2(0.01)),
        clade_budget=(None if args.clade_budget == 0 else args.clade_budget),
        stall_patience=120, trust=args.trust, adam_steps=args.adam_steps, mu=args.mu, trust_max=args.trust_max, adam_lr=args.adam_lr,
        step_extrapolation=args.step_extrapolation, step_model=args.step_model,
        stop_nll_bits=args.stop_nll_bits, approach_pruning_threshold=args.approach_pruning_threshold,
        targeted_hessian=(args.stuck_from, args.stuck_max_frac),
        coordinate_staging=(args.stage_freeze_t, args.stage_d_only),
        trust_test=(args.trust_shrink, args.trust_grow_ratio,
                    args.trust_radius_min, args.trust_min_predicted_bits),
    )
    wall = time.perf_counter() - t0
    cert = {k: res.get(k) for k in ("converged", "bound_active", "unconverged", "interior_pd")}
    summary = dict(tag=args.tag, wall_s=wall, nll_bits=float(res["loss_bits"]), n_steps=int(res["n_steps"]),
                   n_builds=int(res["n_builds"]), adam_s=res["adam_seconds"], hessian_s=res["hessian_seconds"],
                   n_hessians=res["n_hessians"], newton_grad_s=res["newton_grad_seconds"],
                   verify_s=res.get("verify_seconds"), certificate={k: (int(v) if v is not None else None) for k, v in cert.items()},
                   trust=args.trust, adam_steps=args.adam_steps, hessian_refresh=args.hessian_refresh,
                   init_curvature=args.init_curvature, curvature_update=args.curvature_update,
                   mu=args.mu, trust_max=args.trust_max, dtype=args.dtype,
                   step_extrapolation=args.step_extrapolation, step_model=args.step_model,
                   stop_nll_bits=args.stop_nll_bits, approach_pruning_threshold=args.approach_pruning_threshold,
                   n_nll_stopped=int(res["n_nll_stopped"]), approach_end_it=res["approach_end_it"],
                   targeted_hessian=(args.stuck_from, args.stuck_max_frac),
                   coordinate_staging=(args.stage_freeze_t, args.stage_d_only),
                   trust_test=(args.trust_shrink, args.trust_grow_ratio,
                               args.trust_radius_min, args.trust_min_predicted_bits),
                   n_targeted_hessians=int(res["n_targeted_hessians"]),
                   targeted_hessian_families=int(res["targeted_hessian_families"]),
                   targeted_hessian_s=res["targeted_hessian_seconds"])
    print("[recipe] " + json.dumps(summary, default=str), flush=True)
    torch.save({"theta": res["theta"].detach().cpu(), "paths": paths}, f"{args.out_dir}/{args.tag}.pt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
