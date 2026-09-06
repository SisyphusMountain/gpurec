"""Gradient time at a flat theta (-6) vs at a fitted theta (from a run_genewise .pt), same families.

Usage: python benchmark/cc/test_grad_theta.py --species S --families LIST --limit N --clade-budget B --theta-pt FILE
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", required=True)
    ap.add_argument("--families", required=True)
    ap.add_argument("--limit", required=True, type=int)
    ap.add_argument("--clade-budget", required=True, type=int)
    ap.add_argument("--theta-pt", required=True)
    args = ap.parse_args()
    paths = [ln.strip() for ln in open(args.families) if ln.strip() and not ln.startswith("#")]
    if args.limit > 0:
        paths = paths[: args.limit]
    saved = torch.load(args.theta_pt, map_location="cpu")
    pos = {p: i for i, p in enumerate(saved["paths"])}
    theta_fit = torch.stack([saved["theta"][pos[p]] for p in paths]).float().cuda()

    from gpurec.api import _execution
    from gpurec.api.model import GeneReconModel
    from gpurec.api.solver_options import SolverOptions
    from gpurec.fit.genewise_fit import _BASE_SOLVER

    so = SolverOptions(**{**_BASE_SOLVER, "pi_iters": 16, "neumann_terms": 16})
    m = GeneReconModel(args.species, paths, mode="genewise", device="cuda", dtype=torch.float32,
                       solver_options=so, clade_budget=args.clade_budget)
    m.receiver_weights.requires_grad_(False)
    per_batch = []
    orig = _execution.evaluate_static_loss_vector_grad

    def timed(static, *a, **k):
        torch.cuda.synchronize(); t = time.perf_counter()
        out = orig(static, *a, **k)
        torch.cuda.synchronize(); per_batch.append(time.perf_counter() - t)
        return out

    _execution.evaluate_static_loss_vector_grad = timed

    def run(theta, label):
        m.genewise_loss_vector_and_grad(theta=theta, need_grad=True)  # warm-up
        for rep in range(2):
            per_batch.clear(); torch.cuda.synchronize(); t = time.perf_counter()
            m.genewise_loss_vector_and_grad(theta=theta, need_grad=True); torch.cuda.synchronize()
            tot = time.perf_counter() - t; pb = sorted(per_batch)
            print(f"[theta] {label} grad {rep}: {tot:.1f}s over {len(pb)} batches; per-batch min/med/max = "
                  f"{pb[0]:.2f}/{statistics.median(pb):.2f}/{pb[-1]:.2f}s", flush=True)

    theta_flat = torch.full_like(theta_fit, -6.0)
    print(f"[theta] families={len(paths)} batches={len(m.batch_statics)}; fitted theta: min {theta_fit.min():.2f} max {theta_fit.max():.2f} "
          f"frac rows with any rate > 2^-2: {(theta_fit.max(dim=1).values > -2).float().mean():.3f}", flush=True)
    run(theta_flat, "flat(-6)")
    run(theta_fit, "fitted  ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
