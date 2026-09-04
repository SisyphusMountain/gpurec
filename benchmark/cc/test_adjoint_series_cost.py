"""What the exact adjoint's Neumann-series fallback costs when it never fires.

The exact transposed solve hands a clade row to the Neumann series only when its elimination is
badly conditioned, which almost never happens. With
``wave_backward.set_adjoint_series_spill_decision("sync")`` the host reads that wave's spill count
back and launches the series only when it is nonzero -- one 4-byte device-to-host copy per wave.
With ``"always"`` it skips the read and launches the series regardless; every program returns on
its first load, so that is one empty launch per wave instead of the copy.

This is the adjoint's copy of the same question the forward answers in
``benchmark/cc/test_exact_range_cost.py``.

Usage:
  python benchmark/cc/test_adjoint_series_cost.py --species S --families LIST --limit 200 \
      --clade-budget 100000 --pi-iters 16 --neumann-terms 16 --theta -6.0 --reps 5
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time

import torch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--species", required=True)
    parser.add_argument("--families", required=True)
    parser.add_argument("--limit", required=True, type=int)
    parser.add_argument("--clade-budget", required=True, type=int)
    parser.add_argument("--pi-iters", required=True, type=int)
    parser.add_argument("--neumann-terms", required=True, type=int)
    parser.add_argument("--theta", required=True, type=float)
    parser.add_argument("--reps", required=True, type=int)
    args = parser.parse_args()

    from gpurec.api.model import GeneReconModel
    from gpurec.api.solver_options import SolverOptions
    from gpurec.core.kernels import wave_backward
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
    model = GeneReconModel(
        args.species, paths, mode="genewise", device="cuda", dtype=torch.float32,
        solver_options=options, clade_budget=args.clade_budget,
    )
    model.receiver_weights.requires_grad_(False)
    waves = sum(len(s.wave_layout["wave_metas"]) for s in model.batch_statics)
    print(
        f"[cost] families={len(paths)} batches={len(model.batch_statics)} waves={waves} "
        f"S={int(model.species_helpers['S'])}",
        flush=True,
    )
    theta = torch.full((len(paths), 3), args.theta, device="cuda", dtype=torch.float32)

    # The two decisions are measured INTERLEAVED, one gradient each, turn by turn. Another
    # process sharing the card makes any single timing unusable, and its load drifts over
    # minutes, so measuring one decision to completion and then the other compares them at two
    # different loads. Alternating gives both the same conditions, and the per-rep difference is
    # then the quantity of interest.
    decisions = list(wave_backward.ADJOINT_SERIES_SPILL_DECISIONS)
    samples = {name: [] for name in decisions}
    for name in decisions:
        wave_backward.set_adjoint_series_spill_decision(name)
        model.genewise_loss_vector_and_grad(theta=theta, need_grad=True)
    torch.cuda.synchronize()
    for _ in range(args.reps):
        for name in decisions:
            wave_backward.set_adjoint_series_spill_decision(name)
            torch.cuda.synchronize()
            started = time.perf_counter()
            model.genewise_loss_vector_and_grad(theta=theta, need_grad=True)
            torch.cuda.synchronize()
            samples[name].append(time.perf_counter() - started)
    for name in decisions:
        print(
            f"[cost] decision={name:<8} loss+grad median {statistics.median(samples[name]):.4f}s  "
            f"min {min(samples[name]):.4f}s  samples {[round(x, 4) for x in samples[name]]}",
            flush=True,
        )
    paired = [a - b for a, b in zip(samples[decisions[0]], samples[decisions[1]])]
    print(
        f"[cost] paired {decisions[0]} - {decisions[1]} per rep: "
        f"median {statistics.median(paired):+.4f}s  {[round(x, 4) for x in paired]}",
        flush=True,
    )
    wave_backward.set_adjoint_series_spill_decision("always")
    return 0


if __name__ == "__main__":
    sys.exit(main())
