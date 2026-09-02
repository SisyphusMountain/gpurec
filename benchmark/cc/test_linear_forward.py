"""Equivalence and timing checks for the fused linear-space Pi self-loop.

Compares ``SolverOptions.forward_self_loop`` "linear" against the original "log" path on the
same model, same theta, same wave layout:

  A. per-entry absolute Pi (residual + offset), per-family NLL, and per-family gradients,
     with the linear path's early exit active (``pi_linear_tol`` as given);
  B. the same three with ``pi_linear_tol = 0``, i.e. the pure linear-space arithmetic run for
     exactly as many iterations as the log path;
  C. wall time of one loss+gradient call at a larger family count, log vs linear, plus the mean
     number of self-loop iterations the linear kernel actually took per clade row.

Usage:
  python benchmark/cc/test_linear_forward.py --species S --families LIST \
      --limit-compare 100 --limit-time 500 --clade-budget 315000 \
      --pi-iters 16 --neumann-terms 16 --theta -6.0 --window 60 --reps 3
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time

import torch


def _abs_pi(pi_residual: torch.Tensor, pi_offset: torch.Tensor) -> torch.Tensor:
    """Represented absolute log2 Pi, in the offset (accumulator) dtype."""
    return pi_residual.to(pi_offset.dtype) + pi_offset.unsqueeze(1)


def _forward_abs_pi(static, theta_static, receiver_weights):
    from gpurec.core.inference.solver import solve_resident_e_pi

    with torch.no_grad():
        result = solve_resident_e_pi(
            static, theta_static, receiver_weights, warm_start_E=None, pi_iters=None,
            pi_residual_out=None,
        )
    pi_wave = result[5]
    state = static.pi_forward_state
    out = _abs_pi(pi_wave, state.pi_offset)
    del result, pi_wave
    return out


def _compare_pi(reference: torch.Tensor, candidate: torch.Tensor, window: float, eps: float) -> dict:
    """Row-max-windowed comparison of two absolute log2 Pi matrices."""
    finite_reference = torch.isfinite(reference)
    row_max = torch.where(finite_reference, reference, torch.full_like(reference, -float("inf")))
    row_max = row_max.amax(dim=1, keepdim=True)
    in_window = finite_reference & torch.isfinite(row_max) & (reference >= row_max - window)
    both_finite = finite_reference & torch.isfinite(candidate)
    difference = (candidate - reference).abs()
    windowed = in_window & both_finite
    total = reference.numel()
    largest_magnitude = float(reference[finite_reference].abs().max().item()) if bool(finite_reference.any()) else 0.0
    return {
        "entries": total,
        # Both paths carry the row's log2 values through fp32 frame shifts of this magnitude,
        # so this times fp32's 2**-24 is the floor on any log-vs-linear disagreement.
        "max_abs_log2_value": largest_magnitude,
        "frame_resolution": largest_magnitude * eps,
        "finite_frac": float(finite_reference.sum().item()) / total,
        "in_window_frac": float(in_window.sum().item()) / total,
        "below_window_frac": float((finite_reference & ~in_window).sum().item()) / total,
        "max_abs_diff_in_window": float(difference[windowed].max().item()) if bool(windowed.any()) else 0.0,
        "max_abs_diff_all_finite": float(difference[both_finite].max().item()) if bool(both_finite.any()) else 0.0,
        "finite_mismatch": int((finite_reference ^ torch.isfinite(candidate)).sum().item()),
    }


def _loss_and_grad(model, theta):
    loss_vector, grad_theta, _grad_receiver = model.genewise_loss_vector_and_grad(
        theta=theta, need_grad=True
    )
    return loss_vector.detach().clone(), grad_theta.detach().clone()


def _build(species, paths, clade_budget, pi_iters, neumann_terms, mode, tol, dtype):
    from gpurec.api.model import GeneReconModel
    from gpurec.api.solver_options import SolverOptions
    from gpurec.fit.genewise_fit import _BASE_SOLVER

    options = SolverOptions(
        **{
            **_BASE_SOLVER,
            "pi_iters": pi_iters,
            "neumann_terms": neumann_terms,
            "forward_self_loop": mode,
            "pi_linear_tol": tol,
        }
    )
    model = GeneReconModel(
        species, paths, mode="genewise", device="cuda", dtype=dtype,
        solver_options=options, clade_budget=clade_budget,
    )
    model.receiver_weights.requires_grad_(False)
    return model


def _wave_shape(model):
    """Largest wave row count and the linear working buffer that implies."""
    species_count = int(model.species_helpers["S"])
    rows = max(
        int(meta["W"])
        for static in model.batch_statics
        for meta in static.wave_layout["wave_metas"]
    )
    element_bytes = torch.finfo(model.theta.dtype).bits // 8
    return rows, 2 * rows * species_count * element_bytes / 2**30


def _install_iteration_probe():
    """Wrap ``pi_wave_forward`` so each call records the per-row self-loop iteration counts."""
    from gpurec.core.inference import solver as solver_module

    original = solver_module.pi_wave_forward
    counts: list[torch.Tensor] = []

    def probed(**kwargs):
        rows = int(kwargs["wave_layout"]["leaf_species_index"].numel())
        used = torch.zeros(rows, device=kwargs["e"].device, dtype=torch.int32)
        kwargs["linear_iterations_out"] = used
        result = original(**kwargs)
        counts.append(used)
        return result

    solver_module.pi_wave_forward = probed
    return original, counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--species", required=True)
    parser.add_argument("--families", required=True)
    parser.add_argument("--limit-compare", required=True, type=int)
    parser.add_argument("--limit-time", required=True, type=int)
    parser.add_argument("--clade-budget", required=True, type=int)
    parser.add_argument("--pi-iters", required=True, type=int)
    parser.add_argument("--neumann-terms", required=True, type=int)
    parser.add_argument("--theta", required=True, type=float)
    parser.add_argument("--window", required=True, type=float)
    parser.add_argument("--reps", required=True, type=int)
    parser.add_argument("--dtype", required=True, choices=("float32", "float64"))
    args = parser.parse_args()
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    eps = float(torch.finfo(dtype).eps)

    all_paths = [
        line.strip() for line in open(args.families) if line.strip() and not line.startswith("#")
    ]

    # ---------------- A and B: equivalence on --limit-compare families ----------------
    paths = all_paths[: args.limit_compare]
    build_start = time.perf_counter()
    model = _build(
        args.species, paths, args.clade_budget, args.pi_iters, args.neumann_terms, "log", 0.0,
        dtype,
    )
    wave_rows, scratch_gib = _wave_shape(model)
    print(
        f"[cmp] build {time.perf_counter() - build_start:.1f}s families={len(paths)} "
        f"batches={len(model.batch_statics)} S={int(model.species_helpers['S'])} "
        f"max_ancestor_depth={int(model.species_helpers['max_ancestor_depth'])} "
        f"dtype={args.dtype} max_wave_rows={wave_rows} linear_buffer={scratch_gib:.2f} GiB",
        flush=True,
    )
    theta = torch.full((len(paths), 3), args.theta, device="cuda", dtype=dtype)
    receiver_weights = model.receiver_weights.detach()

    reference_rows = []
    for static in model.batch_statics:
        theta_static = model._theta_for_static(static, theta)
        reference_rows.append(_forward_abs_pi(static, theta_static, receiver_weights))
    reference_loss, reference_grad = _loss_and_grad(model, theta)
    repeat_loss, repeat_grad = _loss_and_grad(model, theta)
    grad_noise = float((repeat_grad - reference_grad).abs().max().item())
    loss_noise = float((repeat_loss - reference_loss).abs().max().item())
    print(
        f"[cmp] log-vs-log run-to-run noise: grad max|diff| = {grad_noise:.3e}  "
        f"NLL max|diff| = {loss_noise:.3e} bits",
        flush=True,
    )

    for label, tol in (("A tol=default", None), ("B tol=0", 0.0)):
        model.configure_solver(
            forward_self_loop="linear",
            pi_linear_tol=(1e-6 if tol is None else tol),
        )
        for index, static in enumerate(model.batch_statics):
            theta_static = model._theta_for_static(static, theta)
            candidate = _forward_abs_pi(static, theta_static, receiver_weights)
            stats = _compare_pi(reference_rows[index], candidate, args.window, eps)
            del candidate
            torch.cuda.empty_cache()
            print(
                f"[{label}] batch {index}: rows*species={stats['entries']} "
                f"finite={stats['finite_frac']:.4f} in_window={stats['in_window_frac']:.4f} "
                f"below_window={stats['below_window_frac']:.4f} "
                f"max|dPi| in window = {stats['max_abs_diff_in_window']:.3e} log2 "
                f"(all finite: {stats['max_abs_diff_all_finite']:.3e}) "
                f"finite_mismatch={stats['finite_mismatch']} "
                f"| max|log2 Pi| = {stats['max_abs_log2_value']:.4g}, "
                f"model-dtype frame resolution there = {stats['frame_resolution']:.3e}",
                flush=True,
            )
        linear_loss, linear_grad = _loss_and_grad(model, theta)
        print(
            f"[{label}] NLL max|diff| = {float((linear_loss - reference_loss).abs().max().item()):.3e} bits "
            f"(sum log {float(reference_loss.sum().item()):.4f} vs linear "
            f"{float(linear_loss.sum().item()):.4f} bits)",
            flush=True,
        )
        print(
            f"[{label}] grad max|diff| = {float((linear_grad - reference_grad).abs().max().item()):.3e} "
            f"(log-vs-log atomics noise {grad_noise:.3e}); "
            f"grad max|value| = {float(reference_grad.abs().max().item()):.3e}",
            flush=True,
        )
        model.configure_solver(forward_self_loop="log")

    del reference_rows, model
    torch.cuda.empty_cache()

    # ---------------- C: timing on --limit-time families ----------------
    paths = all_paths[: args.limit_time]
    timings = {}
    iteration_report = ""
    for mode in ("log", "linear"):
        build_start = time.perf_counter()
        model = _build(
            args.species, paths, args.clade_budget, args.pi_iters, args.neumann_terms, mode, 1e-6,
            dtype,
        )
        build_seconds = time.perf_counter() - build_start
        wave_rows, scratch_gib = _wave_shape(model)
        theta = torch.full((len(paths), 3), args.theta, device="cuda", dtype=dtype)
        model.genewise_loss_vector_and_grad(theta=theta, need_grad=True)
        torch.cuda.synchronize()
        samples = []
        for _ in range(args.reps):
            torch.cuda.synchronize()
            start = time.perf_counter()
            model.genewise_loss_vector_and_grad(theta=theta, need_grad=True)
            torch.cuda.synchronize()
            samples.append(time.perf_counter() - start)
        timings[mode] = samples
        print(
            f"[time] {mode}: build {build_seconds:.1f}s  max_wave_rows={wave_rows} "
            f"linear_buffer={scratch_gib:.2f} GiB  peak={torch.cuda.max_memory_allocated() / 2**30:.1f} GiB  loss+grad "
            f"mean {statistics.mean(samples):.3f}s  min {min(samples):.3f}s  "
            f"samples {[round(x, 3) for x in samples]}",
            flush=True,
        )
        if mode == "linear":
            original, counts = _install_iteration_probe()
            model.genewise_loss_vector_and_grad(theta=theta, need_grad=True)
            torch.cuda.synchronize()
            from gpurec.core.inference import solver as solver_module

            solver_module.pi_wave_forward = original
            used = torch.cat([c.reshape(-1) for c in counts]).float()
            active = used[used > 0]
            iteration_report = (
                f"[time] linear self-loop iterations per clade row over {int(active.numel())} rows: "
                f"mean {float(active.mean().item()):.2f}  median {float(active.median().item()):.0f}  "
                f"max {int(active.max().item())}  (cap = pi_iters - prologue)"
            )
            print(iteration_report, flush=True)
        del model
        torch.cuda.empty_cache()

    speedup = statistics.mean(timings["log"]) / statistics.mean(timings["linear"])
    print(f"[time] linear speedup on one loss+gradient call: {speedup:.2f}x", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
