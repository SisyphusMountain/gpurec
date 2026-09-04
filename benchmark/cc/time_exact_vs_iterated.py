"""Wall time of one forward and one gradient: iterated paths (log forward, series adjoint) vs exact.

Usage: time_exact_vs_iterated.py --species S --families LIST --limit 200 --reps 3 --dtype float32 --clade-budget 100000 --pi-iters 16 --neumann-terms 16
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time

import torch


def _build(species, paths, forward, adjoint, dtype, clade_budget, pi_iters, neumann_terms):
    from gpurec.api.model import GeneReconModel
    from gpurec.api.solver_options import SolverOptions
    from gpurec.fit.genewise_fit import _BASE_SOLVER
    so = SolverOptions(**{**_BASE_SOLVER, "forward_self_loop": forward, "adjoint_self_loop": adjoint,
                          "pi_iters": pi_iters, "neumann_terms": neumann_terms})
    m = GeneReconModel(species, paths, mode="genewise", device="cuda", dtype=dtype,
                       solver_options=so, clade_budget=clade_budget)
    m.receiver_weights.requires_grad_(False)
    return m, so


def _time(fn, reps):
    fn(); torch.cuda.synchronize()
    out = []
    for _ in range(reps):
        torch.cuda.synchronize(); t = time.perf_counter(); fn(); torch.cuda.synchronize()
        out.append(time.perf_counter() - t)
    return statistics.median(out), min(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", required=True); ap.add_argument("--families", required=True)
    ap.add_argument("--limit", required=True, type=int); ap.add_argument("--reps", required=True, type=int)
    ap.add_argument("--dtype", required=True, choices=("float32", "float64"))
    ap.add_argument("--clade-budget", required=True, type=int)
    ap.add_argument("--pi-iters", required=True, type=int); ap.add_argument("--neumann-terms", required=True, type=int)
    args = ap.parse_args()
    dtype = getattr(torch, args.dtype)
    paths = [ln.strip() for ln in open(args.families) if ln.strip() and not ln.startswith("#")][: args.limit]
    for forward, adjoint in (("log", "series"), ("exact", "exact")):
        m, so = _build(args.species, paths, forward, adjoint, dtype, args.clade_budget, args.pi_iters, args.neumann_terms)
        theta = torch.tensor([-6.0, -3.0, -6.0], dtype=dtype, device="cuda").repeat(len(paths), 1).contiguous()
        fwd_med, fwd_min = _time(lambda: m.genewise_loss_vector_and_grad(theta=theta, need_grad=False), args.reps)
        grad_med, grad_min = _time(lambda: m.genewise_loss_vector_and_grad(theta=theta, need_grad=True), args.reps)
        nll = float(m.genewise_loss_vector_and_grad(theta=theta, need_grad=False)[0].double().sum())
        print(f"[time] {forward:5s}/{adjoint:6s} {args.dtype} families={len(paths)} batches={len(m.batch_statics)} "
              f"pi_iters={so.pi_iters} neumann_terms={so.neumann_terms}: forward median {fwd_med*1e3:8.1f} ms (min {fwd_min*1e3:8.1f}), "
              f"forward+gradient median {grad_med*1e3:8.1f} ms (min {grad_min*1e3:8.1f}), NLL {nll:.4f}", flush=True)
        del m; torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
