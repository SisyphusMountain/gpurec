"""Correctness and timing checks for the EXACT tree solve of the forward Pi self-loop.

``SolverOptions.forward_self_loop = "exact"`` replaces the truncated Jacobi iteration by an
elimination on the species tree, so its answer is the CONVERGED fixed point, not an iterate. It is
therefore compared against the log-space path run to convergence (``pi_iters`` given by
``--reference-pi-iters``), not against the production 16-iteration path -- which is measured
against the same reference here too, to show how far from converged the truncated iteration is.

  A. per-entry absolute Pi (residual + offset, entries within ``--window`` log2 units of their row
     maximum) and per-family NLL, for "exact" and for "linear" at ``--pi-iters``;
  B. per-family gradients of "exact" against the same reference, sized against the run-to-run
     atomics noise of two identical reference calls;
  C. wall time of one loss+gradient call at ``--limit-time`` families, "exact" vs "linear" vs
     "log", mean of ``--reps`` warm calls;
  D. exact-solve guard trips: elimination pivots that were not positive, and rows whose transfer
     loop gain reached 1. Both should be zero.

Every check runs at two rate settings: the fitted theta of a real run (``--fitted-theta``, a
``run_genewise.py`` .pt keyed by family path) and a flat ``--theta`` in log2 units.

Usage:
  python benchmark/cc/test_exact_forward.py --species S --families LIST \
      --fitted-theta $CC_RUNS/results/full_v3.pt --limit-compare 100 --limit-time 500 \
      --clade-budget 315000 --pi-iters 16 --reference-pi-iters 256 --neumann-terms 16 \
      --theta -6.0 --window 60 --reps 3 --dtype float32
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_linear_forward import _compare_pi, _forward_abs_pi  # noqa: E402


def _build(species, paths, clade_budget, pi_iters, neumann_terms, mode, dtype):
    from gpurec.api.model import GeneReconModel
    from gpurec.api.solver_options import SolverOptions
    from gpurec.fit.genewise_fit import _BASE_SOLVER

    options = SolverOptions(
        **{
            **_BASE_SOLVER,
            "pi_iters": pi_iters,
            "neumann_terms": neumann_terms,
            "forward_self_loop": mode,
        }
    )
    model = GeneReconModel(
        species, paths, mode="genewise", device="cuda", dtype=dtype,
        solver_options=options, clade_budget=clade_budget,
    )
    model.receiver_weights.requires_grad_(False)
    return model


def _loss_and_grad(model, theta):
    loss_vector, grad_theta, _grad_receiver = model.genewise_loss_vector_and_grad(
        theta=theta, need_grad=True
    )
    return loss_vector.detach().clone(), grad_theta.detach().clone()


def _fitted_theta(path, family_paths, device, dtype):
    """Rows of a ``run_genewise.py`` .pt reordered to ``family_paths``, matched by file path."""
    payload = torch.load(path, map_location="cpu")
    fitted = payload["theta"]
    by_path = {str(p): index for index, p in enumerate(payload["paths"])}
    rows = []
    for family_path in family_paths:
        key = str(family_path)
        if key not in by_path:
            raise KeyError(f"{path} has no fitted theta for family {key}")
        rows.append(fitted[by_path[key]])
    return torch.stack(rows).to(device=device, dtype=dtype).contiguous()


def _install_guard_probe():
    """Wrap ``pi_wave_forward`` so each call records the exact solve's per-row guard trips."""
    from gpurec.core.inference import solver as solver_module

    original = solver_module.pi_wave_forward
    trips: list[torch.Tensor] = []

    def probed(**kwargs):
        rows = int(kwargs["wave_layout"]["leaf_species_index"].numel())
        counts = torch.zeros((rows, 2), device=kwargs["e"].device, dtype=torch.int32)
        kwargs["exact_guard_trips_out"] = counts
        result = original(**kwargs)
        trips.append(counts)
        return result

    solver_module.pi_wave_forward = probed
    return original, trips


def _report_guard_trips(label, trips):
    stacked = torch.cat([t for t in trips], dim=0)
    pivot_trips = int(stacked[:, 0].sum().item())
    denominator_trips = int(stacked[:, 1].sum().item())
    rows_with_pivot_trip = int((stacked[:, 0] > 0).sum().item())
    print(
        f"[{label}] guard trips over {int(stacked.shape[0])} clade rows: "
        f"non-positive elimination pivots = {pivot_trips} (in {rows_with_pivot_trip} rows), "
        f"non-positive 1 - loop gain = {denominator_trips} rows",
        flush=True,
    )
    return pivot_trips + denominator_trips


def _compare_against_reference(model, theta, receiver_weights, reference_rows, reference_loss,
                               reference_grad, grad_noise, mode, pi_iters, window, eps, label,
                               with_grad, collect_guards):
    from gpurec.core.inference import solver as solver_module

    model.configure_solver(forward_self_loop=mode, pi_iters=pi_iters)
    original = None
    trips = None
    if collect_guards:
        original, trips = _install_guard_probe()
    worst_in_window = 0.0
    worst_all = 0.0
    finite_mismatch = 0
    for index, static in enumerate(model.batch_statics):
        theta_static = model._theta_for_static(static, theta)
        candidate = _forward_abs_pi(static, theta_static, receiver_weights)
        stats = _compare_pi(reference_rows[index], candidate, window, eps)
        del candidate
        torch.cuda.empty_cache()
        worst_in_window = max(worst_in_window, stats["max_abs_diff_in_window"])
        worst_all = max(worst_all, stats["max_abs_diff_all_finite"])
        finite_mismatch += stats["finite_mismatch"]
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
    loss, grad = _loss_and_grad(model, theta)
    print(
        f"[{label}] SUMMARY max|dPi| in window = {worst_in_window:.3e} log2  "
        f"(all finite {worst_all:.3e})  finite_mismatch={finite_mismatch}",
        flush=True,
    )
    print(
        f"[{label}] NLL max|diff| = {float((loss - reference_loss).abs().max().item()):.3e} bits  "
        f"(sum reference {float(reference_loss.sum().item()):.4f} vs {float(loss.sum().item()):.4f} bits, "
        f"total diff {float((loss.sum() - reference_loss.sum()).item()):.4e} bits)",
        flush=True,
    )
    if with_grad:
        print(
            f"[{label}] grad max|diff| = {float((grad - reference_grad).abs().max().item()):.3e}  "
            f"(reference-vs-reference atomics noise {grad_noise:.3e}; "
            f"grad max|value| {float(reference_grad.abs().max().item()):.3e})",
            flush=True,
        )
    if collect_guards:
        solver_module.pi_wave_forward = original
        _report_guard_trips(label, trips)
    model.configure_solver(forward_self_loop="log")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--species", required=True)
    parser.add_argument("--families", required=True)
    parser.add_argument("--fitted-theta", required=True,
                        help="run_genewise.py .pt with {'theta': [F,3], 'paths': [...]}")
    parser.add_argument("--limit-compare", required=True, type=int)
    parser.add_argument("--limit-time", required=True, type=int)
    parser.add_argument("--clade-budget", required=True, type=int)
    parser.add_argument("--pi-iters", required=True, type=int,
                        help="production self-loop iteration cap, used for 'linear' and 'log'")
    parser.add_argument("--reference-pi-iters", required=True, type=int,
                        help="iteration cap for the CONVERGED log-space reference")
    parser.add_argument("--neumann-terms", required=True, type=int)
    parser.add_argument("--theta", required=True, type=float,
                        help="flat log2 rate for the second rate setting")
    parser.add_argument("--window", required=True, type=float)
    parser.add_argument("--reps", required=True, type=int)
    parser.add_argument("--dtype", required=True, choices=("float32", "float64"))
    args = parser.parse_args()
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    eps = float(torch.finfo(dtype).eps)

    all_paths = [
        line.strip() for line in open(args.families) if line.strip() and not line.startswith("#")
    ]

    # ---------------- A, B, D: correctness on --limit-compare families ----------------
    paths = all_paths[: args.limit_compare]
    build_start = time.perf_counter()
    model = _build(
        args.species, paths, args.clade_budget, args.reference_pi_iters, args.neumann_terms,
        "log", dtype,
    )
    species_count = int(model.species_helpers["S"])
    wave_rows = max(
        int(meta["W"])
        for static in model.batch_statics
        for meta in static.wave_layout["wave_metas"]
    )
    element_bytes = torch.finfo(dtype).bits // 8
    from gpurec.core.kernels.pi_forward import _EXACT_TREE_SCRATCH_SLOTS

    n_levels = int(model.species_helpers["compact_level_ptr"].numel()) - 1
    print(
        f"[cmp] build {time.perf_counter() - build_start:.1f}s families={len(paths)} "
        f"batches={len(model.batch_statics)} S={species_count} "
        f"max_ancestor_depth={int(model.species_helpers['max_ancestor_depth'])} "
        f"species_tree_levels={n_levels} dtype={args.dtype} max_wave_rows={wave_rows} "
        f"linear_buffer={2 * wave_rows * species_count * element_bytes / 2**30:.2f} GiB "
        f"exact_buffer="
        f"{_EXACT_TREE_SCRATCH_SLOTS * wave_rows * species_count * element_bytes / 2**30:.2f} GiB",
        flush=True,
    )
    receiver_weights = model.receiver_weights.detach()
    thetas = {
        "fitted": _fitted_theta(args.fitted_theta, paths, "cuda", dtype),
        f"flat{args.theta:g}": torch.full((len(paths), 3), args.theta, device="cuda", dtype=dtype),
    }

    for rate_label, theta in thetas.items():
        print(f"[cmp] ==== rate setting: {rate_label} "
              f"(theta rows min {float(theta.min().item()):.3f} max {float(theta.max().item()):.3f}) ====",
              flush=True)
        model.configure_solver(forward_self_loop="log", pi_iters=args.reference_pi_iters)
        reference_rows = []
        for static in model.batch_statics:
            theta_static = model._theta_for_static(static, theta)
            reference_rows.append(_forward_abs_pi(static, theta_static, receiver_weights))
        reference_loss, reference_grad = _loss_and_grad(model, theta)
        repeat_loss, repeat_grad = _loss_and_grad(model, theta)
        grad_noise = float((repeat_grad - reference_grad).abs().max().item())
        loss_noise = float((repeat_loss - reference_loss).abs().max().item())
        print(
            f"[cmp/{rate_label}] reference = log, pi_iters={args.reference_pi_iters}; "
            f"run-to-run noise: grad max|diff| = {grad_noise:.3e}  "
            f"NLL max|diff| = {loss_noise:.3e} bits",
            flush=True,
        )
        _compare_against_reference(
            model, theta, receiver_weights, reference_rows, reference_loss, reference_grad,
            grad_noise, "exact", args.pi_iters, args.window, eps,
            f"A/B/D exact {rate_label}", with_grad=True, collect_guards=True,
        )
        _compare_against_reference(
            model, theta, receiver_weights, reference_rows, reference_loss, reference_grad,
            grad_noise, "linear", args.pi_iters, args.window, eps,
            f"A linear(pi_iters={args.pi_iters}) {rate_label}", with_grad=True, collect_guards=False,
        )
        del reference_rows
        torch.cuda.empty_cache()

    del model, thetas
    torch.cuda.empty_cache()

    # ---------------- C: timing on --limit-time families ----------------
    paths = all_paths[: args.limit_time]
    timings: dict[tuple[str, str], list[float]] = {}
    for mode in ("log", "linear", "exact"):
        build_start = time.perf_counter()
        model = _build(
            args.species, paths, args.clade_budget, args.pi_iters, args.neumann_terms, mode, dtype,
        )
        build_seconds = time.perf_counter() - build_start
        rate_thetas = {
            "fitted": _fitted_theta(args.fitted_theta, paths, "cuda", dtype),
            f"flat{args.theta:g}": torch.full(
                (len(paths), 3), args.theta, device="cuda", dtype=dtype
            ),
        }
        for rate_label, theta in rate_thetas.items():
            model.genewise_loss_vector_and_grad(theta=theta, need_grad=True)
            torch.cuda.synchronize()
            samples = []
            for _ in range(args.reps):
                torch.cuda.synchronize()
                start = time.perf_counter()
                model.genewise_loss_vector_and_grad(theta=theta, need_grad=True)
                torch.cuda.synchronize()
                samples.append(time.perf_counter() - start)
            timings[(mode, rate_label)] = samples
            print(
                f"[time] {mode} / {rate_label}: build {build_seconds:.1f}s  "
                f"peak={torch.cuda.max_memory_allocated() / 2**30:.1f} GiB  loss+grad "
                f"mean {statistics.mean(samples):.3f}s  min {min(samples):.3f}s  "
                f"samples {[round(x, 3) for x in samples]}",
                flush=True,
            )
        del model, rate_thetas
        torch.cuda.empty_cache()

    for rate_label in {key[1] for key in timings}:
        base = statistics.mean(timings[("log", rate_label)])
        for mode in ("linear", "exact"):
            mean = statistics.mean(timings[(mode, rate_label)])
            print(
                f"[time] {rate_label}: {mode} is {base / mean:.2f}x the log path "
                f"({mean:.3f}s vs {base:.3f}s)",
                flush=True,
            )
        linear_mean = statistics.mean(timings[("linear", rate_label)])
        exact_mean = statistics.mean(timings[("exact", rate_label)])
        print(
            f"[time] {rate_label}: exact is {linear_mean / exact_mean:.2f}x the linear path "
            f"({exact_mean:.3f}s vs {linear_mean:.3f}s)",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
