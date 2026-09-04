"""Replay one dumped failing batch (see gpurec/api/_failure_dump.py) in isolation.

Rebuilds a model over just that batch's families at the rates it failed at, then:
  * runs the forward with forward_self_loop="log" and "exact" and reports every non-finite
    tensor, the count of clade rows that came out entirely -inf, and the largest absolute Pi
    disagreement inside a window of each row's maximum;
  * runs the per-family loss and gradient in both modes;
  * optionally runs the 3-probe analytic Hessian, which is where the fit died.

Usage:
  python benchmark/cc/replay_failing_batch.py --dump PATH --clade-budget B --window 60 \
      --run-hessian 1
"""
from __future__ import annotations

import argparse
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
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump", required=True)
    parser.add_argument("--clade-budget", required=True, type=int)
    parser.add_argument("--window", required=True, type=float)
    parser.add_argument("--run-hessian", required=True, type=int)
    parser.add_argument("--dtype", required=True, choices=("float32", "float64"))
    parser.add_argument("--oracle", required=True, type=int,
                        help="1: also solve in float64 and score both float32 paths against it")
    args = parser.parse_args()

    payload = torch.load(args.dump, weights_only=False)
    print(f"[replay] {payload['reason']}: {len(payload['family_paths'])} families, "
          f"extra={ {k: v for k, v in payload.items() if k in ('pi_iters', 'probe')} }", flush=True)

    from gpurec.api import _failure_dump
    from gpurec.api.model import GeneReconModel
    from gpurec.api.solver_options import SolverOptions

    options = SolverOptions(**payload["solver_options"])
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    theta = payload["theta_batch"].cuda().to(dtype)
    print(f"[replay] theta min/max per column = "
          f"{[round(float(v), 4) for v in theta.amin(dim=0)]} / "
          f"{[round(float(v), 4) for v in theta.amax(dim=0)]}", flush=True)
    model = GeneReconModel(
        payload["species_tree"], payload["family_paths"], mode="genewise", device="cuda",
        dtype=dtype, solver_options=options, clade_budget=args.clade_budget,
    )
    model.receiver_weights.requires_grad_(False)
    receiver_weights = model.receiver_weights.detach()

    for index, static in enumerate(model.batch_statics):
        theta_static = model._theta_for_static(static, theta)
        print(f"[replay] batch {index} ({len(static.family_indices)} families):", flush=True)
        print(_failure_dump.describe_forward_state(static, theta_static, receiver_weights),
              flush=True)
        absolute = {}
        offsets = {}
        pibar_absolute = {}
        for mode in ("log", "exact"):
            result = _forward(static, theta_static, receiver_weights, mode)
            state = static.pi_forward_state
            absolute[mode] = result[5].to(state.pi_offset.dtype) + state.pi_offset.unsqueeze(1)
            offsets[mode] = state.pi_offset.clone()
            pibar_absolute[mode] = (
                result[6].to(state.pibar_offset.dtype) + state.pibar_offset.unsqueeze(1)
            )
            del result
            torch.cuda.empty_cache()
        # Pibar is mt * (total receiver mass - the lane's ancestor mass): a lane that is -inf in
        # one path and finite in the other is that subtraction cancelling to zero or below, which
        # is what makes the second-order kernel's 1/valid_receiver_mass blow up.
        for mode in ("log", "exact"):
            other = "exact" if mode == "log" else "log"
            only_here = (
                torch.isfinite(pibar_absolute[mode]) & ~torch.isfinite(pibar_absolute[other])
            )
            print(f"[replay] batch {index}: Pibar lanes finite in {mode} but not in {other}: "
                  f"{int(only_here.sum())}", flush=True)
        both_pibar = torch.isfinite(pibar_absolute["log"]) & torch.isfinite(pibar_absolute["exact"])
        if bool(both_pibar.any()):
            print(f"[replay] batch {index}: max|dPibar| where both finite = "
                  f"{float((pibar_absolute['exact'] - pibar_absolute['log']).abs()[both_pibar].max()):.3e} log2",
                  flush=True)
        del pibar_absolute
        torch.cuda.empty_cache()
        reference, candidate = absolute["log"], absolute["exact"]
        finite_reference = torch.isfinite(reference)
        row_max = torch.where(
            finite_reference, reference, torch.full_like(reference, -float("inf"))
        ).amax(dim=1, keepdim=True)
        inside = finite_reference & torch.isfinite(row_max) & (reference >= row_max - args.window)
        both = inside & torch.isfinite(candidate)
        lost = int((inside & ~torch.isfinite(candidate)).sum())
        worst = float((candidate - reference).abs()[both].max()) if bool(both.any()) else float("nan")
        print(f"[replay] batch {index}: max|dPi| in window = {worst:.3e} log2, "
              f"in-window lanes the exact path lost = {lost}, "
              f"max|log2 Pi| = {float(reference[finite_reference].abs().max()):.4g}", flush=True)

        # Which rows, and is it the row gauge or the row's dynamic range? ``Pi_offset`` under the
        # exact path IS the scale the kernel exponentiated against, so comparing it with the log
        # path's own row maximum says directly whether the scale is the problem.
        row_difference = torch.where(
            both, (candidate - reference).abs(), torch.zeros_like(reference)
        ).amax(dim=1)
        worst_rows = torch.argsort(row_difference, descending=True)[:8]
        log_offset = offsets["log"]
        exact_offset = offsets["exact"]
        finite_row_max = row_max.squeeze(1)
        print(f"[replay] batch {index}: exact Pi_offset minus the log path's own row maximum: "
              f"max {float((exact_offset - finite_row_max).max()):.4g}, "
              f"min {float((exact_offset - finite_row_max).min()):.4g}", flush=True)
        for row in worst_rows.tolist():
            reference_row = reference[row]
            finite_lanes = torch.isfinite(reference_row)
            span = (
                float(reference_row[finite_lanes].max() - reference_row[finite_lanes].min())
                if bool(finite_lanes.any()) else float("nan")
            )
            print(f"[replay]   row {row}: max|dPi| {float(row_difference[row]):.3e} | "
                  f"log row max {float(finite_row_max[row]):.6g}, log offset "
                  f"{float(log_offset[row]):.6g}, exact offset {float(exact_offset[row]):.6g} | "
                  f"log finite lanes {int(finite_lanes.sum())}, exact finite lanes "
                  f"{int(torch.isfinite(candidate[row]).sum())}, log value span {span:.4g} log2",
                  flush=True)
        del reference, candidate, absolute
        torch.cuda.empty_cache()

    if args.oracle:
        # The float32 log path is not a correctness reference here: its valid-receiver-mass
        # subtraction is the thing under suspicion. Score BOTH float32 paths against the float64
        # log solve, which docs/centered_state_contract.md names as the correctness oracle.
        oracle_options = SolverOptions(**payload["solver_options"])
        oracle_options.forward_self_loop = "log"
        oracle_model = GeneReconModel(
            payload["species_tree"], payload["family_paths"], mode="genewise", device="cuda",
            dtype=torch.float64, solver_options=oracle_options, clade_budget=args.clade_budget,
        )
        oracle_model.receiver_weights.requires_grad_(False)
        oracle_theta = payload["theta_batch"].cuda().to(torch.float64)
        for index, static in enumerate(oracle_model.batch_statics):
            oracle_static_theta = oracle_model._theta_for_static(static, oracle_theta)
            result = _forward(
                static, oracle_static_theta, oracle_model.receiver_weights.detach(), "log"
            )
            oracle_absolute = (
                result[5].to(static.pi_forward_state.pi_offset.dtype)
                + static.pi_forward_state.pi_offset.unsqueeze(1)
            )
            del result
            torch.cuda.empty_cache()
            finite_oracle = torch.isfinite(oracle_absolute)
            oracle_row_max = torch.where(
                finite_oracle, oracle_absolute,
                torch.full_like(oracle_absolute, -float("inf")),
            ).amax(dim=1, keepdim=True)
            inside = finite_oracle & torch.isfinite(oracle_row_max) & (
                oracle_absolute >= oracle_row_max - args.window
            )
            for mode in ("log", "exact"):
                static32 = model.batch_statics[index]
                candidate_result = _forward(
                    static32, model._theta_for_static(static32, theta), receiver_weights, mode
                )
                candidate = (
                    candidate_result[5].to(static32.pi_forward_state.pi_offset.dtype)
                    + static32.pi_forward_state.pi_offset.unsqueeze(1)
                )
                del candidate_result
                torch.cuda.empty_cache()
                both = inside & torch.isfinite(candidate)
                worst_here = (
                    float((candidate.to(torch.float64) - oracle_absolute).abs()[both].max())
                    if bool(both.any()) else float("nan")
                )
                print(f"[replay] batch {index}: float32 {mode} vs the float64 oracle: "
                      f"max|dPi| in window = {worst_here:.3e} log2, "
                      f"in-window lanes it lost = {int((inside & ~torch.isfinite(candidate)).sum())}",
                      flush=True)
                del candidate
                torch.cuda.empty_cache()
            del oracle_absolute, inside, finite_oracle, oracle_row_max
            torch.cuda.empty_cache()
        del oracle_model
        torch.cuda.empty_cache()

    for mode in ("log", "exact"):
        model.configure_solver(forward_self_loop=mode)
        loss, grad, _ = model.genewise_loss_vector_and_grad(theta=theta, need_grad=True)
        print(f"[replay] {mode}: NLL sum {float(loss.sum()):.4f} bits, "
              f"non-finite losses {int((~torch.isfinite(loss)).sum())}, "
              f"non-finite gradients {int((~torch.isfinite(grad)).sum())}", flush=True)

    if args.run_hessian:
        from gpurec.fit.genewise_fit import _analytic_hessian

        for mode in ("log", "exact"):
            model.configure_solver(forward_self_loop=mode)
            try:
                curvature = _analytic_hessian(
                    model, theta, int(options.pi_iters), payload["species_tree"],
                    payload["family_paths"],
                )
                print(f"[replay] {mode}: analytic Hessian ok, non-finite entries "
                      f"{int((~torch.isfinite(curvature)).sum())}", flush=True)
            except Exception as failure:  # noqa: BLE001 - reporting which mode fails is the point
                print(f"[replay] {mode}: analytic Hessian raised "
                      f"{type(failure).__name__}: {failure}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
