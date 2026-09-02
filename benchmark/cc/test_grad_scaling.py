"""Per-batch gradient timing at scale, warm vs cold adjoint, with CUDA allocator counters.

Usage: python benchmark/cc/test_grad_scaling.py --species S --families LIST --limit N --clade-budget B --warm 0|1
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
    ap.add_argument("--warm", required=True, type=int)
    args = ap.parse_args()
    paths = [ln.strip() for ln in open(args.families) if ln.strip() and not ln.startswith("#")]
    if args.limit > 0:
        paths = paths[: args.limit]

    from gpurec.api import _execution
    from gpurec.api.model import GeneReconModel
    from gpurec.api.solver_options import SolverOptions
    from gpurec.fit.genewise_fit import _BASE_SOLVER

    if args.warm:
        os.environ["GPUREC_WARM_ADJOINT"] = "1"
    else:
        os.environ.pop("GPUREC_WARM_ADJOINT", None)
    so = SolverOptions(**{**_BASE_SOLVER, "pi_iters": 16, "neumann_terms": 16})
    t0 = time.perf_counter()
    m = GeneReconModel(args.species, paths, mode="genewise", device="cuda", dtype=torch.float32,
                       solver_options=so, clade_budget=args.clade_budget)
    m.receiver_weights.requires_grad_(False)
    print(f"[scal] build {time.perf_counter() - t0:.1f}s  families={len(paths)} batches={len(m.batch_statics)} "
          f"warm_env={bool(os.environ.get('GPUREC_WARM_ADJOINT'))} warm_ok={getattr(m, 'warm_adjoint_ok', None)}", flush=True)

    per_batch = []
    orig = _execution.evaluate_static_loss_vector_grad

    def timed(static, *a, **k):
        torch.cuda.synchronize(); t = time.perf_counter()
        out = orig(static, *a, **k)
        torch.cuda.synchronize(); per_batch.append(time.perf_counter() - t)
        return out

    _execution.evaluate_static_loss_vector_grad = timed
    theta = torch.full((len(paths), 3), -6.0, device="cuda")

    def stats():
        s = torch.cuda.memory_stats()
        return {k: s.get(k, -1) for k in ("num_alloc_retries", "num_device_alloc", "num_device_free")} | \
               {"reserved_gib": round(s.get("reserved_bytes.all.current", 0) / 2**30, 1),
                "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 1)}

    torch.cuda.synchronize(); t = time.perf_counter()
    m.genewise_loss_vector_and_grad(theta=theta, need_grad=True); torch.cuda.synchronize()
    print(f"[scal] warm-up grad {time.perf_counter() - t:.1f}s  {stats()}", flush=True)
    for rep in range(2):
        per_batch.clear(); s0 = stats()
        torch.cuda.synchronize(); t = time.perf_counter()
        m.genewise_loss_vector_and_grad(theta=theta, need_grad=True); torch.cuda.synchronize()
        tot = time.perf_counter() - t; s1 = stats()
        pb = sorted(per_batch)
        print(f"[scal] grad {rep}: {tot:.1f}s over {len(pb)} batches; per-batch min/med/max = "
              f"{pb[0]:.2f}/{statistics.median(pb):.2f}/{pb[-1]:.2f}s; "
              f"device_alloc +{s1['num_device_alloc'] - s0['num_device_alloc']} device_free +{s1['num_device_free'] - s0['num_device_free']} "
              f"retries +{s1['num_alloc_retries'] - s0['num_alloc_retries']} reserved={s1['reserved_gib']}GiB peak={s1['peak_gib']}GiB", flush=True)
        if len(pb) >= 8:
            print(f"[scal]   slowest 5 batches: {[round(x, 2) for x in pb[-5:]]}  first 5 in order: {[round(x, 2) for x in per_batch[:5]]}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
