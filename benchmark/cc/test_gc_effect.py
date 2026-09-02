"""Does Python's cyclic garbage collector slow the gradient loop at scale?

Builds a model over N families (tens of millions of small Python objects live in the parsed payload),
times gradients with the collector ENABLED, then DISABLED, then with the payload FROZEN (gc.freeze),
in the same process. GPU required.
Usage: python benchmark/cc/test_gc_effect.py --species S --families LIST --limit N --clade-budget B
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time

import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", required=True)
    ap.add_argument("--families", required=True)
    ap.add_argument("--limit", required=True, type=int)
    ap.add_argument("--clade-budget", required=True, type=int)
    args = ap.parse_args()
    paths = [ln.strip() for ln in open(args.families) if ln.strip() and not ln.startswith("#")]
    if args.limit > 0:
        paths = paths[: args.limit]

    from gpurec.api.model import GeneReconModel
    from gpurec.api.solver_options import SolverOptions
    from gpurec.fit.genewise_fit import _BASE_SOLVER

    os.environ["GPUREC_WARM_ADJOINT"] = "1"
    so = SolverOptions(**{**_BASE_SOLVER, "pi_iters": 16, "neumann_terms": 16})
    t0 = time.perf_counter()
    m = GeneReconModel(args.species, paths, mode="genewise", device="cuda", dtype=torch.float32,
                       solver_options=so, clade_budget=args.clade_budget)
    m.receiver_weights.requires_grad_(False)
    print(f"[gc] build {time.perf_counter() - t0:.1f}s, {len(paths)} families, {len(m.batch_statics)} batches, "
          f"gc tracked objects={len(gc.get_objects()):,}", flush=True)
    theta = torch.full((len(paths), 3), -6.0, device="cuda")

    def grad_time():
        torch.cuda.synchronize(); t = time.perf_counter()
        m.genewise_loss_vector_and_grad(theta=theta, need_grad=True)
        torch.cuda.synchronize(); return time.perf_counter() - t

    grad_time()  # warm-up / compile
    gc.collect()
    c0 = gc.get_stats()
    en = [grad_time() for _ in range(2)]
    c1 = gc.get_stats()
    print(f"[gc] ENABLED : {en}  collections during: {[ (b['collections']-a['collections']) for a,b in zip(c0,c1)]}", flush=True)
    gc.disable()
    dis = [grad_time() for _ in range(2)]
    print(f"[gc] DISABLED: {dis}", flush=True)
    gc.enable(); gc.collect(); gc.freeze()
    fr = [grad_time() for _ in range(2)]
    print(f"[gc] FROZEN  : {fr}  (frozen objects={gc.get_freeze_count():,})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
