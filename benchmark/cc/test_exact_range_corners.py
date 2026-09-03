"""Does the exact path, with its range fallback, hold up across the rate box like the log path?

The exact and linear self-loops keep ONE scale per clade row, so a species lane far enough below
that scale is an exact zero in float32 where the log path still carries it. ``exact_range_log2``
is meant to close that gap: the forward hands such a row to the log-space sweeps, and its adjoint
and tangent to the Neumann series and the tangent sweeps.

This walks the log2-rate box the genewise fit uses and, at every corner, scores the float32 exact
path AND the float32 log path against the SAME float64 oracle -- the log path run in float64,
which has the range for anything the box produces. The exact path passes a corner when it is no
worse than the log path is at that corner, since the log path is what it must match.

Per corner it reports, for each of the two float32 paths:
  * the largest absolute Pi disagreement with the oracle inside a window of each row's maximum,
  * how many in-window lanes the oracle resolves that the path lost to -inf,
  * the per-family NLL and gradient disagreement,
and, for the exact path, how many clade rows its range check flagged.

Usage:
  python benchmark/cc/test_exact_range_corners.py --species S --families LIST --limit 20 \
      --clade-budget 315000 --pi-iters 16 --neumann-terms 16 --window 60 \
      --grid -19.9,-6,1 --exact-range-log2 100
"""
from __future__ import annotations

import argparse
import itertools
import sys

import torch


def _build(species, paths, clade_budget, pi_iters, neumann_terms, forward, adjoint, dtype,
           exact_range_log2):
    from gpurec.api.model import GeneReconModel
    from gpurec.api.solver_options import SolverOptions
    from gpurec.fit.genewise_fit import _BASE_SOLVER

    options = SolverOptions(**{
        **_BASE_SOLVER,
        "pi_iters": pi_iters,
        "neumann_terms": neumann_terms,
        "forward_self_loop": forward,
        "adjoint_self_loop": adjoint,
        "exact_range_log2": exact_range_log2,
    })
    model = GeneReconModel(
        species, paths, mode="genewise", device="cuda", dtype=dtype,
        solver_options=options, clade_budget=clade_budget,
    )
    model.receiver_weights.requires_grad_(False)
    return model


def _forward_rows(model, theta):
    """Absolute log2 Pi per batch, plus how many rows the range check flagged."""
    from gpurec.core.inference.solver import solve_resident_e_pi

    receiver_weights = model.receiver_weights.detach()
    rows = []
    flagged = 0
    for static in model.batch_statics:
        theta_static = model._theta_for_static(static, theta)
        with torch.no_grad():
            result = solve_resident_e_pi(
                static, theta_static, receiver_weights, warm_start_E=None, pi_iters=None,
                pi_residual_out=None,
            )
        state = static.pi_forward_state
        rows.append((result[5].to(state.pi_offset.dtype) + state.pi_offset.unsqueeze(1)).clone())
        flagged += int(state.wide_row_total)
        del result
        torch.cuda.empty_cache()
    return rows, flagged


def _score(oracle_rows, candidate_rows, window):
    """Worst in-window disagreement with the oracle, and in-window lanes lost to -inf."""
    worst = 0.0
    lost = 0
    for oracle, candidate in zip(oracle_rows, candidate_rows):
        oracle = oracle.to(candidate.dtype)
        finite = torch.isfinite(oracle)
        row_max = torch.where(finite, oracle, torch.full_like(oracle, -float("inf")))
        row_max = row_max.amax(dim=1, keepdim=True)
        inside = finite & torch.isfinite(row_max) & (oracle >= row_max - window)
        both = inside & torch.isfinite(candidate)
        if bool(both.any()):
            worst = max(worst, float((candidate[both] - oracle[both]).abs().max()))
        lost += int((inside & ~torch.isfinite(candidate)).sum())
    return worst, lost


def _loss_and_grad(model, theta):
    loss, grad, _ = model.genewise_loss_vector_and_grad(theta=theta, need_grad=True)
    torch.cuda.synchronize()
    return loss.detach().double().clone(), grad.detach().double().clone()


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
    parser.add_argument("--exact-range-log2", required=True, type=float)
    args = parser.parse_args()

    paths = [
        line.strip() for line in open(args.families) if line.strip() and not line.startswith("#")
    ][: args.limit]

    common = (args.species, paths, args.clade_budget, args.pi_iters, args.neumann_terms)
    oracle_model = _build(*common, "log", "series", torch.float64, args.exact_range_log2)
    log_model = _build(*common, "log", "series", torch.float32, args.exact_range_log2)
    exact_model = _build(*common, "exact", "exact", torch.float32, args.exact_range_log2)
    print(
        f"[grid] {len(paths)} families, {len(exact_model.batch_statics)} batches, "
        f"S={int(exact_model.species_helpers['S'])}, window={args.window} log2, "
        f"exact_range_log2={args.exact_range_log2}",
        flush=True,
    )

    grid = [float(value) for value in args.grid.split(",")]
    failures = []
    for duplication, loss_rate, transfer in itertools.product(grid, repeat=3):
        def theta_for(model):
            return torch.tensor(
                [duplication, loss_rate, transfer],
                device="cuda", dtype=model.theta.dtype,
            ).expand(len(paths), 3).contiguous()

        oracle_rows, _ = _forward_rows(oracle_model, theta_for(oracle_model))
        oracle_loss, oracle_grad = _loss_and_grad(oracle_model, theta_for(oracle_model))

        report = []
        for label, model in (("log", log_model), ("exact", exact_model)):
            rows, flagged = _forward_rows(model, theta_for(model))
            worst, lost = _score(oracle_rows, rows, args.window)
            model_loss, model_grad = _loss_and_grad(model, theta_for(model))
            report.append({
                "label": label,
                "pi": worst,
                "lost": lost,
                "flagged": flagged,
                "nll": float((model_loss - oracle_loss).abs().max()),
                "grad": float((model_grad - oracle_grad).abs().max()),
            })
            del rows
            torch.cuda.empty_cache()
        del oracle_rows
        torch.cuda.empty_cache()

        log_row, exact_row = report
        # The log path is the standard the exact path has to meet, so the comparison is relative
        # to it, with a little slack for the two paths' different arithmetic order.
        diverged = (
            exact_row["lost"] > log_row["lost"]
            or exact_row["pi"] > max(10.0 * log_row["pi"], 1e-2)
            or exact_row["nll"] > max(10.0 * log_row["nll"], 1e-2)
            or not exact_row["pi"] == exact_row["pi"]
        )
        flag = "  <-- EXACT WORSE THAN LOG" if diverged else ""
        print(
            f"[grid] D={duplication:6.2f} L={loss_rate:6.2f} T={transfer:6.2f} | "
            f"flagged rows={exact_row['flagged']:6d} | "
            f"max|dPi| vs fp64 log={log_row['pi']:.3e} exact={exact_row['pi']:.3e} | "
            f"lanes lost log={log_row['lost']} exact={exact_row['lost']} | "
            f"max|dNLL| log={log_row['nll']:.3e} exact={exact_row['nll']:.3e} | "
            f"max|dgrad| log={log_row['grad']:.3e} exact={exact_row['grad']:.3e}{flag}",
            flush=True,
        )
        if diverged:
            failures.append((duplication, loss_rate, transfer))
    print(f"[grid] corners where the exact path is worse than the log path: {failures}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
