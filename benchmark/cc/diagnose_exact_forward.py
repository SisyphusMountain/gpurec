"""Where "exact" and the log-space forward disagree, which one is wrong?

``test_exact_forward.py`` reports, at the fitted theta, a small number of Pi entries where the
exact tree solve and the log path at ``pi_iters=256`` differ by over 10 log2 units even though
the NLL and the gradient agree. Three things could explain that: the log path is still moving at
256 iterations (so the "reference" is not the fixed point), the exact solve is wrong, or neither
linear-space path can represent those entries. This script separates them by making the log path
itself the moving part: it compares the log path at ``--reference-pi-iters`` and at
``--long-pi-iters`` (a multiple of it) against each other, and both against "exact" and "linear",
on the same batch, same theta.

For each pair it prints the number of entries past a few log2-distance thresholds and the full
profile of the single worst-disagreeing entry (its row, its species, its row maximum and its value
under every path), so a disagreement can be read as "this lane, this far below its row's maximum".

What it established (100 families, fitted theta, batch 0, 6.2e8 entries, H100):

  * The reference IS converged. log@256 against log@2048 differ by at most 2.1e-2 log2, with 9
    entries past 1e-3 and none past 1e-2.
  * The exact solve IS the fixed point. Re-run at ``--dtype float64``, it matches log@512 to
    2.5e-7 log2 with nothing past 1e-3 -- a thousand times closer than the truncated linear path
    at the same precision (2.5e-4). The elimination is right; only its float32 arithmetic is not.
  * What is left in float32 belongs to the representation, not to either solver. Both linear-space
    paths rescale a row so its largest entry is 1, so a lane whose share of the row's transfer
    mass is under float32 roundoff gets the roundoff instead of its value -- and comes out too
    LARGE. On the worst entry the two paths land on the SAME wrong value (-6281.3 and -6281.4
    against the log path's -6293.9, a lane 36.6 log2 below its row maximum). Only the log path
    is free of that floor.

Usage:
  python benchmark/cc/diagnose_exact_forward.py --species S --families LIST \
      --fitted-theta $CC_RUNS/results/full_v3.pt --limit 100 --batch 0 \
      --clade-budget 315000 --pi-iters 16 --reference-pi-iters 256 --long-pi-iters 2048 \
      --neumann-terms 16 --window 60 --dtype float32
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_exact_forward import _build, _fitted_theta  # noqa: E402
from test_linear_forward import _forward_abs_pi  # noqa: E402

_THRESHOLDS = (1e-3, 1e-2, 1e-1, 1.0, 5.0)


def _pair_report(label, reference, candidate, window):
    """Windowed distance between two absolute log2 Pi matrices, with the worst entry located."""
    finite_reference = torch.isfinite(reference)
    row_max = torch.where(
        finite_reference, reference, torch.full_like(reference, -float("inf"))
    ).amax(dim=1, keepdim=True)
    in_window = finite_reference & torch.isfinite(row_max) & (reference >= row_max - window)
    both_finite = in_window & torch.isfinite(candidate)
    difference = torch.where(
        both_finite, (candidate - reference).abs(), torch.zeros_like(reference)
    )
    counts = " ".join(
        f">{threshold:g}: {int((difference > threshold).sum().item())}"
        for threshold in _THRESHOLDS
    )
    mismatch = int((in_window & ~torch.isfinite(candidate)).sum().item())
    worst = int(difference.argmax().item())
    row, species = divmod(worst, int(reference.shape[1]))
    print(
        f"[pair] {label}: max|d| = {float(difference.max().item()):.4e} log2 over "
        f"{int(in_window.sum().item())} in-window entries; entries past "
        f"{counts}; in-window entries that went to -inf: {mismatch}",
        flush=True,
    )
    return row, species


def _entry_profile(name_to_rows, row, species):
    row_max = {
        name: float(
            torch.where(
                torch.isfinite(rows[row]), rows[row], torch.full_like(rows[row], -float("inf"))
            ).max().item()
        )
        for name, rows in name_to_rows.items()
    }
    print(f"[entry] clade row {row}, species {species}:", flush=True)
    for name, rows in name_to_rows.items():
        value = float(rows[row, species].item())
        print(
            f"[entry]   {name:>14}: log2 Pi = {value:.6f}  "
            f"(row max {row_max[name]:.6f}, {row_max[name] - value:.3f} log2 below it)",
            flush=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--species", required=True)
    parser.add_argument("--families", required=True)
    parser.add_argument("--fitted-theta", required=True)
    parser.add_argument("--limit", required=True, type=int)
    parser.add_argument("--batch", required=True, type=int)
    parser.add_argument("--clade-budget", required=True, type=int)
    parser.add_argument("--pi-iters", required=True, type=int)
    parser.add_argument("--reference-pi-iters", required=True, type=int)
    parser.add_argument("--long-pi-iters", required=True, type=int)
    parser.add_argument("--neumann-terms", required=True, type=int)
    parser.add_argument("--window", required=True, type=float)
    parser.add_argument("--dtype", required=True, choices=("float32", "float64"))
    args = parser.parse_args()
    dtype = torch.float32 if args.dtype == "float32" else torch.float64

    paths = [
        line.strip() for line in open(args.families) if line.strip() and not line.startswith("#")
    ][: args.limit]
    start = time.perf_counter()
    model = _build(
        args.species, paths, args.clade_budget, args.long_pi_iters, args.neumann_terms, "log",
        dtype,
    )
    static = model.batch_statics[args.batch]
    theta = _fitted_theta(args.fitted_theta, paths, "cuda", dtype)
    theta_static = model._theta_for_static(static, theta)
    receiver_weights = model.receiver_weights.detach()
    print(
        f"[diag] build {time.perf_counter() - start:.1f}s families={len(paths)} "
        f"batch {args.batch} of {len(model.batch_statics)} "
        f"clade rows={int(static.wave_layout['leaf_species_index'].numel())} "
        f"S={int(model.species_helpers['S'])} dtype={args.dtype}",
        flush=True,
    )

    rows = {}
    for name, mode, pi_iters in (
        (f"log@{args.long_pi_iters}", "log", args.long_pi_iters),
        (f"log@{args.reference_pi_iters}", "log", args.reference_pi_iters),
        (f"linear@{args.pi_iters}", "linear", args.pi_iters),
        ("exact", "exact", args.pi_iters),
    ):
        model.configure_solver(forward_self_loop=mode, pi_iters=pi_iters)
        solve_start = time.perf_counter()
        rows[name] = _forward_abs_pi(static, theta_static, receiver_weights)
        print(
            f"[diag] {name}: forward solve {time.perf_counter() - solve_start:.2f}s",
            flush=True,
        )

    long_name = f"log@{args.long_pi_iters}"
    reference = rows[long_name]
    worst_entries = []
    for name in rows:
        if name == long_name:
            continue
        worst_entries.append(
            (name, *_pair_report(f"{name} vs {long_name}", reference, rows[name], args.window))
        )
    for name, row, species in worst_entries:
        print(f"[entry] --- worst entry for {name} vs {long_name} ---", flush=True)
        _entry_profile(rows, row, species)
    return 0


if __name__ == "__main__":
    sys.exit(main())
