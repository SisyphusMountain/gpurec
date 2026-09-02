"""Correctness and timing checks for the EXACT tree solve of the wave-tangent self-loop.

The Hessian probes push a tangent through the same wave self-loop the primal solves, with a
different right-hand side. Today that runs ``tangent_self_iters`` fixed Jacobi sweeps (the fit
passes its current ``pi_iters``, 16 or 64). ``SolverOptions.adjoint_self_loop = "exact"`` -- the
same switch that turns on the exact adjoint, because the tangent and the adjoint are the two sides
of one operator -- instead eliminates the system on the species tree, so its answer is the LIMIT of
those sweeps rather than a truncation of them.

  A. per-family 3x3 analytic Hessians at ``--limit-compare`` families, "exact" and the production
     sweep count against the sweeps run to convergence (``--reference-iters``), sized against the
     run-to-run atomics noise of two identical reference calls.
  B. wall time of one 3-probe analytic Hessian at ``--limit-time`` families, reference vs
     production sweeps vs exact.
  C. exact-solve guard trips: elimination pivots that were not positive, and rows whose donor
     tangent loop gain reached 1. Both should be zero.

Every check runs at two rate settings: the fitted theta of a real run (``--fitted-theta``) and a
flat ``--theta`` in log2 units.

Usage:
  python benchmark/cc/test_exact_tangent.py --species S --families LIST \
      --fitted-theta $CC_RUNS/results/full_v3.pt --limit-compare 100 --limit-time 500 \
      --clade-budget 315000 --pi-iters 16 --reference-iters 256 --neumann-terms 16 \
      --theta -6.0 --reps 3 --forward-self-loop exact
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


def _build(species, paths, clade_budget, pi_iters, neumann_terms,
           forward_self_loop, adjoint_self_loop, dtype):
    from gpurec.api.model import GeneReconModel
    from gpurec.api.solver_options import SolverOptions
    from gpurec.fit.genewise_fit import _BASE_SOLVER

    options = SolverOptions(**{
        **_BASE_SOLVER,
        "pi_iters": pi_iters,
        "neumann_terms": neumann_terms,
        "forward_self_loop": forward_self_loop,
        "adjoint_self_loop": adjoint_self_loop,
    })
    model = GeneReconModel(
        species, paths, mode="genewise", device="cuda", dtype=dtype,
        solver_options=options, clade_budget=clade_budget,
    )
    model.receiver_weights.requires_grad_(False)
    return model


def _hessian(model, theta, species, paths, self_iters):
    """The fit's own per-family [G, 3, 3] curvature, at a given tangent sweep count."""
    from gpurec.fit.genewise_fit import _analytic_hessian

    out = _analytic_hessian(model, theta, int(self_iters), species, list(paths))
    torch.cuda.synchronize()
    return out.detach().clone()


def _compare(label, reference, candidate):
    difference = (candidate - reference).abs()
    scale = reference.abs().amax()
    print(
        f"[{label}] max|dH| = {float(difference.max().item()):.4e} "
        f"(max|H| = {float(scale.item()):.4e}, relative "
        f"{float((difference.max() / scale.clamp_min(torch.finfo(reference.dtype).tiny)).item()):.4e}); "
        f"mean|dH| = {float(difference.mean().item()):.4e}; "
        f"non-finite in candidate: {int((~torch.isfinite(candidate)).sum().item())}",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--species", required=True)
    parser.add_argument("--families", required=True)
    parser.add_argument("--fitted-theta", required=True)
    parser.add_argument("--limit-compare", required=True, type=int)
    parser.add_argument("--limit-time", required=True, type=int)
    parser.add_argument("--clade-budget", required=True, type=int)
    parser.add_argument("--pi-iters", required=True, type=int,
                        help="production tangent sweep count (the fit passes its pi tier)")
    parser.add_argument("--reference-iters", required=True, type=int,
                        help="tangent sweep count for the CONVERGED reference")
    parser.add_argument("--neumann-terms", required=True, type=int)
    parser.add_argument("--theta", required=True, type=float)
    parser.add_argument("--reps", required=True, type=int)
    parser.add_argument("--forward-self-loop", required=True, choices=("log", "linear", "exact"))
    parser.add_argument("--dtype", required=True, choices=("float32", "float64"),
                        help="float64 is the control: it separates the elimination's arithmetic "
                             "from float32's own resolution")
    parser.add_argument("--skip-timing", required=True, type=int,
                        help="1 to stop after the correctness section")
    args = parser.parse_args()
    dtype = torch.float32 if args.dtype == "float32" else torch.float64

    from gpurec.core.kernels import wave_tangent

    all_paths = [
        line.strip() for line in open(args.families) if line.strip() and not line.startswith("#")
    ]

    # ---------------- A, C: correctness on --limit-compare families ----------------
    paths = all_paths[: args.limit_compare]
    build_start = time.perf_counter()
    model = _build(
        args.species, paths, args.clade_budget, args.pi_iters, args.neumann_terms,
        args.forward_self_loop, "series", dtype,
    )
    height = model.species_helpers["sp_height"]
    print(
        f"[cmp] build {time.perf_counter() - build_start:.1f}s families={len(paths)} "
        f"batches={len(model.batch_statics)} S={int(model.species_helpers['S'])} "
        f"species_tree_height={int(height.max().item())} forward={args.forward_self_loop} "
        f"dtype={args.dtype} reference=sweeps@{args.reference_iters}",
        flush=True,
    )
    thetas = {
        "fitted": _fitted_theta(args.fitted_theta, paths, "cuda", dtype),
        f"flat{args.theta:g}": torch.full((len(paths), 3), args.theta, device="cuda", dtype=dtype),
    }

    for rate_label, theta in thetas.items():
        print(f"[cmp] ==== rate setting: {rate_label} ====", flush=True)
        model.configure_solver(adjoint_self_loop="series")
        reference = _hessian(model, theta, args.species, paths, args.reference_iters)
        repeat = _hessian(model, theta, args.species, paths, args.reference_iters)
        _compare(f"A {rate_label} reference-vs-reference", reference, repeat)

        wave_tangent.set_exact_tangent_guard_trip_collection(True)
        model.configure_solver(adjoint_self_loop="exact")
        exact = _hessian(model, theta, args.species, paths, args.pi_iters)
        trips = torch.cat([t for t in wave_tangent._EXACT_TANGENT_GUARD_TRIPS], dim=0)
        margin = trips[:, 2]
        print(
            f"[C {rate_label}] guard trips over {int(trips.shape[0])} clade-row solves: "
            f"non-positive elimination pivots = {int(trips[:, 0].sum().item())} "
            f"(in {int((trips[:, 0] > 0).sum().item())} rows), "
            f"non-positive 1 - loop gain = {int(trips[:, 1].sum().item())} rows; "
            f"closure margin 1 - M1: min {float(margin.min().item()):.4e} "
            f"1st-percentile {float(margin.quantile(0.01).item()):.4e} "
            f"median {float(margin.median().item()):.4e} "
            f"(worst row loses about {-torch.log10(margin.min().abs().clamp_min(1e-30)).item():.1f} "
            f"decimal digits to it)",
            flush=True,
        )
        wave_tangent.set_exact_tangent_guard_trip_collection(False)
        _compare(f"A {rate_label} exact", reference, exact)

        model.configure_solver(adjoint_self_loop="series")
        production = _hessian(model, theta, args.species, paths, args.pi_iters)
        _compare(f"A {rate_label} sweeps@{args.pi_iters}", reference, production)
        del reference, repeat, exact, production
        torch.cuda.empty_cache()

    del model, thetas
    torch.cuda.empty_cache()
    if args.skip_timing:
        return 0

    # ---------------- B: timing on --limit-time families ----------------
    paths = all_paths[: args.limit_time]
    timings: dict[tuple[str, str], list[float]] = {}
    modes = (
        (f"sweeps@{args.reference_iters}", "series", args.reference_iters),
        (f"sweeps@{args.pi_iters}", "series", args.pi_iters),
        ("exact", "exact", args.pi_iters),
    )
    for label, mode, iters in modes:
        build_start = time.perf_counter()
        model = _build(
            args.species, paths, args.clade_budget, args.pi_iters, args.neumann_terms,
            args.forward_self_loop, mode, dtype,
        )
        build_seconds = time.perf_counter() - build_start
        rate_thetas = {
            "fitted": _fitted_theta(args.fitted_theta, paths, "cuda", dtype),
            f"flat{args.theta:g}": torch.full(
                (len(paths), 3), args.theta, device="cuda", dtype=dtype
            ),
        }
        for rate_label, theta in rate_thetas.items():
            _hessian(model, theta, args.species, paths, iters)
            samples = []
            for _ in range(args.reps):
                torch.cuda.synchronize()
                start = time.perf_counter()
                _hessian(model, theta, args.species, paths, iters)
                samples.append(time.perf_counter() - start)
            timings[(label, rate_label)] = samples
            print(
                f"[time] {label} / {rate_label}: build {build_seconds:.1f}s  "
                f"peak={torch.cuda.max_memory_allocated() / 2**30:.1f} GiB  3-probe Hessian "
                f"mean {statistics.mean(samples):.3f}s  min {min(samples):.3f}s  "
                f"samples {[round(x, 3) for x in samples]}",
                flush=True,
            )
        del model, rate_thetas
        torch.cuda.empty_cache()

    for rate_label in sorted({key[1] for key in timings}):
        base_label = f"sweeps@{args.pi_iters}"
        base = statistics.mean(timings[(base_label, rate_label)])
        for label, _mode, _iters in modes:
            if label == base_label:
                continue
            mean = statistics.mean(timings[(label, rate_label)])
            print(
                f"[time] {rate_label}: {label} is {base / mean:.2f}x the production "
                f"{base_label} path ({mean:.3f}s vs {base:.3f}s)",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
