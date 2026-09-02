"""Sweep the rate box and report where the linear self-loop stops matching the log one.

The fused linear self-loop holds one clade row as linear values against a single per-row scale,
so a lane more than ~126 binary orders below that scale is zero, where the log path keeps it as a
very negative number. Whether that ever matters depends on the rates: this walks the log2-rate box
the genewise fit uses, runs the forward both ways on the same model, and reports, per corner,

  * how many Pi rows come out entirely -inf (the shape a linear-space underflow takes),
  * the largest absolute Pi disagreement inside a 60-log2 window of each row's maximum,
  * whether the per-family NLL and gradient stay finite.

Usage:
  python benchmark/cc/test_linear_extreme_theta.py --species S --families LIST --limit N \
      --clade-budget B --pi-iters 16 --neumann-terms 16 --window 60 --grid -19.9,-12,-6,-2,0,1
"""
from __future__ import annotations

import argparse
import itertools
import sys

import torch


def _forward(static, theta_static, receiver_weights, mode):
    from gpurec.core.inference.solver import solve_resident_e_pi

    static.solver_options.forward_self_loop = mode
    with torch.no_grad():
        result = solve_resident_e_pi(
            static, theta_static, receiver_weights,
            warm_start_E=None, pi_iters=None, pi_residual_out=None,
        )
    pi_wave = result[5]
    absolute = pi_wave.to(static.pi_forward_state.pi_offset.dtype) + \
        static.pi_forward_state.pi_offset.unsqueeze(1)
    dead_rows = int((~torch.isfinite(pi_wave)).all(dim=1).sum())
    root_rows = result[4]
    nonfinite_roots = int((~torch.isfinite(root_rows)).sum())
    del result, pi_wave
    return absolute, dead_rows, nonfinite_roots


def _windowed_difference(reference, candidate, window):
    finite_reference = torch.isfinite(reference)
    row_max = torch.where(
        finite_reference, reference, torch.full_like(reference, -float("inf"))
    ).amax(dim=1, keepdim=True)
    inside = finite_reference & torch.isfinite(row_max) & (reference >= row_max - window)
    both = inside & torch.isfinite(candidate)
    if not bool(both.any()):
        return float("nan"), int((inside & ~torch.isfinite(candidate)).sum())
    difference = (candidate - reference).abs()
    return (
        float(difference[both].max()),
        int((inside & ~torch.isfinite(candidate)).sum()),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--species", required=True)
    parser.add_argument("--families", required=True)
    parser.add_argument("--limit", required=True, type=int)
    parser.add_argument("--clade-budget", required=True, type=int)
    parser.add_argument("--pi-iters", required=True, type=int)
    parser.add_argument("--neumann-terms", required=True, type=int)
    parser.add_argument("--window", required=True, type=float)
    parser.add_argument("--grid", required=True, help="comma-separated log2 rates to sweep")
    parser.add_argument("--dtype", required=True, choices=("float32", "float64"))
    args = parser.parse_args()

    from gpurec.api.model import GeneReconModel
    from gpurec.api.solver_options import SolverOptions
    from gpurec.fit.genewise_fit import _BASE_SOLVER

    paths = [
        line.strip() for line in open(args.families) if line.strip() and not line.startswith("#")
    ][: args.limit]
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    options = SolverOptions(
        **{**_BASE_SOLVER, "pi_iters": args.pi_iters, "neumann_terms": args.neumann_terms,
           "forward_self_loop": "log"}
    )
    model = GeneReconModel(
        args.species, paths, mode="genewise", device="cuda", dtype=dtype,
        solver_options=options, clade_budget=args.clade_budget,
    )
    model.receiver_weights.requires_grad_(False)
    receiver_weights = model.receiver_weights.detach()
    print(f"[grid] {len(paths)} families, {len(model.batch_statics)} batches, "
          f"S={int(model.species_helpers['S'])}", flush=True)

    grid = [float(value) for value in args.grid.split(",")]
    worst = []
    for duplication, transfer, loss in itertools.product(grid, repeat=3):
        theta = torch.tensor(
            [duplication, transfer, loss], device="cuda", dtype=dtype
        ).expand(len(paths), 3).contiguous()
        rows = []
        for static in model.batch_statics:
            theta_static = model._theta_for_static(static, theta)
            reference, dead_log, roots_log = _forward(static, theta_static, receiver_weights, "log")
            candidate, dead_lin, roots_lin = _forward(
                static, theta_static, receiver_weights, "linear"
            )
            difference, lost = _windowed_difference(reference, candidate, args.window)
            rows.append((dead_log, dead_lin, roots_log, roots_lin, difference, lost))
            del reference, candidate
            torch.cuda.empty_cache()
        dead_log = sum(r[0] for r in rows); dead_lin = sum(r[1] for r in rows)
        roots_log = sum(r[2] for r in rows); roots_lin = sum(r[3] for r in rows)
        difference = max(r[4] for r in rows); lost = sum(r[5] for r in rows)
        flag = "  <-- LINEAR DIVERGES" if (dead_lin > dead_log or roots_lin > roots_log
                                          or lost > 0 or not difference == difference
                                          or difference > 1.0) else ""
        print(f"[grid] theta D={duplication:6.2f} T={transfer:6.2f} L={loss:6.2f} | "
              f"all--inf Pi rows log={dead_log} linear={dead_lin} | "
              f"non-finite root rows log={roots_log} linear={roots_lin} | "
              f"max|dPi| in window={difference:.3e} | in-window lanes lost by linear={lost}{flag}",
              flush=True)
        if flag:
            worst.append((duplication, transfer, loss))
    print(f"[grid] corners where the linear path diverges: {worst}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
