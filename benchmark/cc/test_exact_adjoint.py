"""Correctness and timing checks for the EXACT tree solve of the wave-backward adjoint.

``SolverOptions.adjoint_self_loop = "exact"`` replaces the Neumann series by an elimination on
the species tree, so its answer is the series' LIMIT, not a truncation of it. It is therefore
compared against the series run to convergence (``--reference-terms`` terms, early exit off),
not against the production 16-term path -- which is measured against the same reference here too,
to show how far from converged the truncated series is.

  A. per-wave adjoint. One gradient pass is spied on to capture the arguments of the largest leaf
     wave and the largest gene-split wave; each is then re-solved from IDENTICAL inputs under the
     reference, under "exact", and under the production series, so the comparison isolates the
     solve from everything upstream of it.
  B. full gradient at ``--limit-compare`` families: "exact" and the production series against the
     converged reference, sized against the run-to-run atomics noise of two identical reference
     calls.
  C. wall time of one loss+gradient at ``--limit-time`` families, reference vs series vs exact,
     mean of ``--reps`` warm calls.
  D. exact-solve guard trips: elimination pivots that were not positive, and rows whose transfer
     loop gain reached 1. Both should be zero.

Every check runs at two rate settings: the fitted theta of a real run (``--fitted-theta``) and a
flat ``--theta`` in log2 units.

Usage:
  python benchmark/cc/test_exact_adjoint.py --species S --families LIST \
      --fitted-theta $CC_RUNS/results/full_v3.pt --limit-compare 100 --limit-time 500 \
      --clade-budget 315000 --pi-iters 16 --neumann-terms 16 --reference-terms 256 \
      --neumann-term-tol 1e-7 --theta -6.0 --reps 3 --forward-self-loop exact
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

# The only arguments the wave self-loop solve writes into; everything else it reads. A replay has
# to hand each run its own copy of these, and of ``rhs``, which later waves keep adding into.
_MUTATED_KWARGS = ("grad_receiver_log_probs", "self_loop_grad_targets")
_RHS_POSITION = 6


def _build(species, paths, clade_budget, pi_iters, neumann_terms, neumann_term_tol,
           forward_self_loop, adjoint_self_loop, dtype):
    from gpurec.api.model import GeneReconModel
    from gpurec.api.solver_options import SolverOptions
    from gpurec.fit.genewise_fit import _BASE_SOLVER

    options = SolverOptions(**{
        **_BASE_SOLVER,
        "pi_iters": pi_iters,
        "neumann_terms": neumann_terms,
        "neumann_term_tol": neumann_term_tol,
        "forward_self_loop": forward_self_loop,
        "adjoint_self_loop": adjoint_self_loop,
    })
    model = GeneReconModel(
        species, paths, mode="genewise", device="cuda", dtype=dtype,
        solver_options=options, clade_budget=clade_budget,
    )
    model.receiver_weights.requires_grad_(False)
    return model


def _copy_of(value):
    if torch.is_tensor(value):
        return value.clone()
    if isinstance(value, (tuple, list)):
        return type(value)(_copy_of(item) for item in value)
    return value


def _capture_widest_waves(model, theta, wave_backward):
    """Arguments of the widest leaf wave and the widest gene-split wave of one gradient pass."""
    original = wave_backward._solve_reconciliation_wave_vjp_2d
    widest: dict[str, dict] = {}

    def spy(*args, **kwargs):
        width = int(kwargs["W"]) if "W" in kwargs else int(args[3])
        # args[5] is gene_split_log_likelihood: present exactly on the split waves.
        kind = "split" if args[5] is not None else "leaf"
        if width > widest.get(kind, {}).get("W", -1):
            captured = list(args)
            captured[_RHS_POSITION] = _copy_of(captured[_RHS_POSITION])
            widest[kind] = {"W": width, "args": tuple(captured), "kwargs": dict(kwargs)}
        return original(*args, **kwargs)

    wave_backward._solve_reconciliation_wave_vjp_2d = spy
    try:
        model.genewise_loss_vector_and_grad(theta=theta, need_grad=True)
        torch.cuda.synchronize()
    finally:
        wave_backward._solve_reconciliation_wave_vjp_2d = original
    return original, widest


def _replay(original, captured, *, adjoint_self_loop, neumann_terms, neumann_term_tol):
    """Re-solve one captured wave under a different adjoint setting, inputs untouched."""
    args = list(captured["args"])
    args[_RHS_POSITION] = _copy_of(args[_RHS_POSITION])
    kwargs = dict(captured["kwargs"])
    for name in _MUTATED_KWARGS:
        if kwargs.get(name) is not None:
            kwargs[name] = _copy_of(kwargs[name])
    kwargs["adjoint_self_loop"] = adjoint_self_loop
    kwargs["neumann_terms"] = int(neumann_terms)
    kwargs["neumann_term_tol"] = float(neumann_term_tol)
    out = original(*args, **kwargs)
    torch.cuda.synchronize()
    adjoint = out[0].clone()
    active = kwargs.get("active_mask")
    if active is not None:
        row_active = active.reshape(active.shape[0], -1).ne(0).any(dim=1)
        adjoint = adjoint[row_active]
    return adjoint


def _compare_adjoint(label, reference, candidate):
    difference = (candidate - reference).abs()
    scale = reference.abs().amax()
    print(
        f"[{label}] rows*species={reference.numel()} "
        f"max|dv| = {float(difference.max().item()):.4e} "
        f"(max|v| = {float(scale.item()):.4e}, "
        f"relative {float((difference.max() / scale.clamp_min(torch.finfo(reference.dtype).tiny)).item()):.4e}); "
        f"mean|dv| = {float(difference.mean().item()):.4e}; "
        f"non-finite in candidate: {int((~torch.isfinite(candidate)).sum().item())}",
        flush=True,
    )


def _loss_and_grad(model, theta):
    loss_vector, grad_theta, _ = model.genewise_loss_vector_and_grad(theta=theta, need_grad=True)
    torch.cuda.synchronize()
    return loss_vector.detach().clone(), grad_theta.detach().clone()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--species", required=True)
    parser.add_argument("--families", required=True)
    parser.add_argument("--fitted-theta", required=True)
    parser.add_argument("--limit-compare", required=True, type=int)
    parser.add_argument("--limit-time", required=True, type=int)
    parser.add_argument("--clade-budget", required=True, type=int)
    parser.add_argument("--pi-iters", required=True, type=int)
    parser.add_argument("--neumann-terms", required=True, type=int,
                        help="production Neumann cap, used for the 'series' comparison and timing")
    parser.add_argument("--reference-terms", required=True, type=int,
                        help="Neumann cap for the CONVERGED reference (early exit forced off)")
    parser.add_argument("--neumann-term-tol", required=True, type=float)
    parser.add_argument("--theta", required=True, type=float)
    parser.add_argument("--reps", required=True, type=int)
    parser.add_argument("--forward-self-loop", required=True, choices=("log", "linear", "exact"))
    args = parser.parse_args()
    dtype = torch.float32

    from gpurec.core.kernels import wave_backward

    all_paths = [
        line.strip() for line in open(args.families) if line.strip() and not line.startswith("#")
    ]

    # ---------------- A, B, D: correctness on --limit-compare families ----------------
    paths = all_paths[: args.limit_compare]
    build_start = time.perf_counter()
    model = _build(
        args.species, paths, args.clade_budget, args.pi_iters, args.reference_terms, 0.0,
        args.forward_self_loop, "series", dtype,
    )
    print(
        f"[cmp] build {time.perf_counter() - build_start:.1f}s families={len(paths)} "
        f"batches={len(model.batch_statics)} S={int(model.species_helpers['S'])} "
        f"species_tree_levels={int(model.species_helpers['compact_level_ptr'].numel()) - 1} "
        f"forward={args.forward_self_loop} reference=series@{args.reference_terms} tol=0",
        flush=True,
    )
    thetas = {
        "fitted": _fitted_theta(args.fitted_theta, paths, "cuda", dtype),
        f"flat{args.theta:g}": torch.full((len(paths), 3), args.theta, device="cuda", dtype=dtype),
    }

    for rate_label, theta in thetas.items():
        print(f"[cmp] ==== rate setting: {rate_label} ====", flush=True)

        # --- A: one wave at a time, replayed from identical inputs.
        original, widest = _capture_widest_waves(model, theta, wave_backward)
        for kind, captured in sorted(widest.items()):
            reference = _replay(
                original, captured, adjoint_self_loop="series",
                neumann_terms=args.reference_terms, neumann_term_tol=0.0,
            )
            repeat = _replay(
                original, captured, adjoint_self_loop="series",
                neumann_terms=args.reference_terms, neumann_term_tol=0.0,
            )
            _compare_adjoint(f"A {kind} wave W={captured['W']} {rate_label} reference-vs-reference",
                             reference, repeat)
            exact = _replay(
                original, captured, adjoint_self_loop="exact",
                neumann_terms=args.neumann_terms, neumann_term_tol=args.neumann_term_tol,
            )
            _compare_adjoint(f"A {kind} wave W={captured['W']} {rate_label} exact", reference, exact)
            series = _replay(
                original, captured, adjoint_self_loop="series",
                neumann_terms=args.neumann_terms, neumann_term_tol=args.neumann_term_tol,
            )
            _compare_adjoint(
                f"A {kind} wave W={captured['W']} {rate_label} series@{args.neumann_terms}",
                reference, series,
            )
            del reference, repeat, exact, series
            torch.cuda.empty_cache()
        del widest
        torch.cuda.empty_cache()

        # --- B: the whole gradient.
        model.configure_solver(
            adjoint_self_loop="series", neumann_terms=args.reference_terms, neumann_term_tol=0.0
        )
        reference_loss, reference_grad = _loss_and_grad(model, theta)
        repeat_loss, repeat_grad = _loss_and_grad(model, theta)
        grad_noise = float((repeat_grad - reference_grad).abs().max().item())
        print(
            f"[B {rate_label}] reference-vs-reference atomics noise: grad max|diff| = "
            f"{grad_noise:.3e}  NLL max|diff| = "
            f"{float((repeat_loss - reference_loss).abs().max().item()):.3e} bits",
            flush=True,
        )

        # --- D: guard trips, collected over the exact run below.
        wave_backward.set_exact_adjoint_guard_trip_collection(True)
        model.configure_solver(
            adjoint_self_loop="exact", neumann_terms=args.neumann_terms,
            neumann_term_tol=args.neumann_term_tol,
        )
        exact_loss, exact_grad = _loss_and_grad(model, theta)
        trips = torch.cat([t for t in wave_backward._EXACT_ADJOINT_GUARD_TRIPS], dim=0)
        print(
            f"[D {rate_label}] guard trips over {int(trips.shape[0])} clade rows: "
            f"non-positive elimination pivots = {int(trips[:, 0].sum().item())} "
            f"(in {int((trips[:, 0] > 0).sum().item())} rows), "
            f"non-positive 1 - loop gain = {int(trips[:, 1].sum().item())} rows",
            flush=True,
        )
        wave_backward.set_exact_adjoint_guard_trip_collection(False)
        print(
            f"[B {rate_label}] exact:  grad max|diff| = "
            f"{float((exact_grad - reference_grad).abs().max().item()):.3e}  "
            f"NLL max|diff| = {float((exact_loss - reference_loss).abs().max().item()):.3e} bits  "
            f"(grad max|value| {float(reference_grad.abs().max().item()):.3e})",
            flush=True,
        )

        model.configure_solver(
            adjoint_self_loop="series", neumann_terms=args.neumann_terms,
            neumann_term_tol=args.neumann_term_tol,
        )
        series_loss, series_grad = _loss_and_grad(model, theta)
        print(
            f"[B {rate_label}] series@{args.neumann_terms}: grad max|diff| = "
            f"{float((series_grad - reference_grad).abs().max().item()):.3e}  "
            f"NLL max|diff| = {float((series_loss - reference_loss).abs().max().item()):.3e} bits",
            flush=True,
        )
        model.configure_solver(
            adjoint_self_loop="series", neumann_terms=args.reference_terms, neumann_term_tol=0.0
        )

    del model, thetas
    torch.cuda.empty_cache()

    # ---------------- C: timing on --limit-time families ----------------
    paths = all_paths[: args.limit_time]
    timings: dict[tuple[str, str], list[float]] = {}
    modes = (
        (f"series@{args.reference_terms}", "series", args.reference_terms, 0.0),
        (f"series@{args.neumann_terms}", "series", args.neumann_terms, args.neumann_term_tol),
        ("exact", "exact", args.neumann_terms, args.neumann_term_tol),
    )
    for label, mode, terms, tol in modes:
        build_start = time.perf_counter()
        model = _build(
            args.species, paths, args.clade_budget, args.pi_iters, terms, tol,
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
            model.genewise_loss_vector_and_grad(theta=theta, need_grad=True)
            torch.cuda.synchronize()
            samples = []
            for _ in range(args.reps):
                torch.cuda.synchronize()
                start = time.perf_counter()
                model.genewise_loss_vector_and_grad(theta=theta, need_grad=True)
                torch.cuda.synchronize()
                samples.append(time.perf_counter() - start)
            timings[(label, rate_label)] = samples
            print(
                f"[time] {label} / {rate_label}: build {build_seconds:.1f}s  "
                f"peak={torch.cuda.max_memory_allocated() / 2**30:.1f} GiB  loss+grad "
                f"mean {statistics.mean(samples):.3f}s  min {min(samples):.3f}s  "
                f"samples {[round(x, 3) for x in samples]}",
                flush=True,
            )
        del model, rate_thetas
        torch.cuda.empty_cache()

    for rate_label in sorted({key[1] for key in timings}):
        base_label = f"series@{args.neumann_terms}"
        base = statistics.mean(timings[(base_label, rate_label)])
        for label, _mode, _terms, _tol in modes:
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
