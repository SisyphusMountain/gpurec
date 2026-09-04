"""Per-launch duration of the exact forward solve versus the wave's row count (torch.profiler).

Fits duration ~ floor + slope * rows over all wave launches of one forward, to separate the fixed
per-launch latency (barriers, level walks, launch) from the per-row work.
Usage: launch_floor.py --species S --families LIST --limit 200 --clade-budget 100000
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
    args = ap.parse_args()
    from gpurec.api.model import GeneReconModel
    from gpurec.api.solver_options import SolverOptions
    from gpurec.fit.genewise_fit import _BASE_SOLVER
    paths = [ln.strip() for ln in open(args.families) if ln.strip() and not ln.startswith("#")][: args.limit]
    so = SolverOptions(**{**_BASE_SOLVER, "forward_self_loop": "exact", "adjoint_self_loop": "exact"})
    m = GeneReconModel(args.species, paths, mode="genewise", device="cuda", dtype=torch.float32,
                       solver_options=so, clade_budget=args.clade_budget)
    m.receiver_weights.requires_grad_(False)
    theta = torch.tensor([-6.0, -3.0, -6.0], dtype=torch.float32, device="cuda").repeat(len(paths), 1).contiguous()
    rows_in_order = [int(meta["W"]) for st in m.batch_statics for meta in st.wave_layout["wave_metas"]]
    m.genewise_loss_vector_and_grad(theta=theta, need_grad=False); torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        m.genewise_loss_vector_and_grad(theta=theta, need_grad=False); torch.cuda.synchronize()
    for name in ("_exact_tree_pi_self_loop_kernel", "_stage_multiple_gene_split_event_reduction_kernel", "_update_reconciliation_likelihood_kernel"):
        ev = sorted([e for e in prof.events() if e.name.startswith(name) and e.device_type.name == "CUDA"], key=lambda e: e.time_range.start)
        d = torch.tensor([e.time_range.elapsed_us() for e in ev], dtype=torch.float64)
        if name == "_exact_tree_pi_self_loop_kernel":
            assert len(ev) == len(rows_in_order), (len(ev), len(rows_in_order))
            W = torch.tensor(rows_in_order, dtype=torch.float64)
            X = torch.stack([torch.ones_like(W), W], dim=1)
            coef = torch.linalg.lstsq(X, d.unsqueeze(1)).solution.flatten()
            print(f"[floor] {name}: {len(ev)} launches, total {d.sum()/1e3:.1f} ms; rows per wave: median {W.median():.0f}, "
                  f"mean {W.mean():.0f}, max {W.max():.0f}; waves with <=32 rows: {(W<=32).float().mean()*100:.0f}% carrying "
                  f"{(d[W<=32].sum()/d.sum())*100:.0f}% of the time; <=128 rows: {(W<=128).float().mean()*100:.0f}% / {(d[W<=128].sum()/d.sum())*100:.0f}% of time", flush=True)
            print(f"[floor]   fit duration = {coef[0]:.1f} us + {coef[1]:.3f} us/row; floor share of total = {coef[0]*len(ev)/d.sum()*100:.0f}%", flush=True)
            for lo, hi in ((1, 8), (9, 32), (33, 128), (129, 512), (513, 4096), (4097, 10**9)):
                sel = (W >= lo) & (W <= hi)
                if sel.any():
                    print(f"[floor]   rows {lo}-{hi}: {int(sel.sum()):5d} launches, median {d[sel].median():7.1f} us, mean {d[sel].mean():7.1f} us", flush=True)
        else:
            print(f"[floor] {name}: {len(ev)} launches, total {d.sum()/1e3:.1f} ms, median {d.median():.1f} us, min {d.min():.1f} us", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
