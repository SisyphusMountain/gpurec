"""What the exact path's range fallback costs when it never fires.

The forward decides per wave whether any clade row exceeded what one row scale can hold. With
``forward.set_exact_range_fallback_decision("sync")`` it reads that wave's count back to the host
and launches nothing when it is zero -- one device-to-host copy per wave, ~12,000 of them at full
scale. With ``"always"`` it skips the read and launches the masked sweeps regardless, which return
immediately on every row: ``pi_iters`` empty launches per wave instead of the copy.

Run this in a checkout WITHOUT the fallback too (the script degrades to a single "baseline" row
there) to get the real before/after: the cost of the machinery is the gap between that row and the
better of the two decisions.

Usage:
  python benchmark/cc/test_exact_range_cost.py --species S --families LIST --limit 500 \
      --clade-budget 315000 --pi-iters 16 --neumann-terms 16 --fitted-theta T.pt \
      --theta -6.0 --reps 3
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_exact_forward import _fitted_theta  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--species", required=True)
    parser.add_argument("--families", required=True)
    parser.add_argument("--limit", required=True, type=int)
    parser.add_argument("--clade-budget", required=True, type=int)
    parser.add_argument("--pi-iters", required=True, type=int)
    parser.add_argument("--neumann-terms", required=True, type=int)
    parser.add_argument("--fitted-theta", required=True)
    parser.add_argument("--theta", required=True, type=float)
    parser.add_argument("--reps", required=True, type=int)
    args = parser.parse_args()

    from gpurec.api.model import GeneReconModel
    from gpurec.api.solver_options import SolverOptions
    from gpurec.core.inference import forward as forward_module
    from gpurec.fit.genewise_fit import _BASE_SOLVER

    paths = [
        line.strip() for line in open(args.families) if line.strip() and not line.startswith("#")
    ][: args.limit]
    options = SolverOptions(**{
        **_BASE_SOLVER,
        "pi_iters": args.pi_iters,
        "neumann_terms": args.neumann_terms,
        "forward_self_loop": "exact",
        "adjoint_self_loop": "exact",
    })
    start = time.perf_counter()
    model = GeneReconModel(
        args.species, paths, mode="genewise", device="cuda", dtype=torch.float32,
        solver_options=options, clade_budget=args.clade_budget,
    )
    model.receiver_weights.requires_grad_(False)
    waves = sum(len(s.wave_layout["wave_metas"]) for s in model.batch_statics)
    print(
        f"[cost] build {time.perf_counter() - start:.1f}s families={len(paths)} "
        f"batches={len(model.batch_statics)} waves={waves} "
        f"S={int(model.species_helpers['S'])}",
        flush=True,
    )

    thetas = {
        "fitted": _fitted_theta(args.fitted_theta, paths, "cuda", torch.float32),
        f"flat{args.theta:g}": torch.full(
            (len(paths), 3), args.theta, device="cuda", dtype=torch.float32
        ),
    }
    switch = getattr(forward_module, "set_exact_range_fallback_decision", None)
    decisions = ("sync", "always") if switch is not None else ("baseline (no fallback)",)
    for decision in decisions:
        if switch is not None:
            switch(decision)
        for rate_label, theta in thetas.items():
            model.genewise_loss_vector_and_grad(theta=theta, need_grad=True)
            torch.cuda.synchronize()
            samples = []
            for _ in range(args.reps):
                torch.cuda.synchronize()
                started = time.perf_counter()
                model.genewise_loss_vector_and_grad(theta=theta, need_grad=True)
                torch.cuda.synchronize()
                samples.append(time.perf_counter() - started)
            flagged = sum(
                int(getattr(s.pi_forward_state, "wide_row_total", 0)) for s in model.batch_statics
            )
            print(
                f"[cost] decision={decision:<24} {rate_label:<8} loss+grad "
                f"mean {statistics.mean(samples):.4f}s  min {min(samples):.4f}s  "
                f"flagged rows={flagged}  samples {[round(x, 4) for x in samples]}",
                flush=True,
            )
    if switch is not None:
        switch("sync")
    return 0


if __name__ == "__main__":
    sys.exit(main())
