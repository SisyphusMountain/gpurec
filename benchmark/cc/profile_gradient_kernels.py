"""Per-kernel GPU time of one forward and of one forward+gradient (torch.profiler, CUDA kernels only).

Usage: profile_gradient_kernels.py --species S --families LIST --limit 200 --clade-budget 100000 --top 14
"""
from __future__ import annotations

import argparse
import sys

import torch
from torch.profiler import ProfilerActivity, profile


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", required=True); ap.add_argument("--families", required=True)
    ap.add_argument("--limit", required=True, type=int); ap.add_argument("--clade-budget", required=True, type=int)
    ap.add_argument("--top", required=True, type=int)
    args = ap.parse_args()
    from gpurec.api.model import GeneReconModel
    from gpurec.api.solver_options import SolverOptions
    from gpurec.fit.genewise_fit import _BASE_SOLVER
    paths = [ln.strip() for ln in open(args.families) if ln.strip() and not ln.startswith("#")][: args.limit]
    so = SolverOptions(**_BASE_SOLVER)
    m = GeneReconModel(args.species, paths, mode="genewise", device="cuda", dtype=torch.float32,
                       solver_options=so, clade_budget=args.clade_budget)
    m.receiver_weights.requires_grad_(False)
    theta = torch.tensor([-6.0, -3.0, -6.0], dtype=torch.float32, device="cuda").repeat(len(paths), 1).contiguous()
    for need_grad in (False, True):
        m.genewise_loss_vector_and_grad(theta=theta, need_grad=need_grad); torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            m.genewise_loss_vector_and_grad(theta=theta, need_grad=need_grad); torch.cuda.synchronize()
        rows = [e for e in prof.key_averages() if e.device_time_total > 0]
        total = sum(e.device_time_total for e in rows)
        rows.sort(key=lambda e: e.device_time_total, reverse=True)
        label = "forward+gradient" if need_grad else "forward only"
        print(f"[prof] {label}: total GPU time {total/1e3:.0f} ms over {len(rows)} distinct kernels", flush=True)
        for e in rows[: args.top]:
            print(f"[prof]   {100*e.device_time_total/total:5.1f}%  {e.device_time_total/1e3:8.1f} ms  {e.count:6d} launches  {e.key[:90]}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
