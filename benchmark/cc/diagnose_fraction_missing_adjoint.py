"""Why does a gradient with ``fraction_missing`` set raise "E-adjoint Neumann series failed"?

The gradient needs one linear solve that the self-loop kernels have nothing to do with: the
EXTINCTION-ADJOINT solve ``(I - J) x = b``, where ``J`` is the Jacobian of the extinction-probability
step. gpurec solves it by summing the Neumann series ``x = b + Jb + J^2 b + ...``, which is only
valid, and only FAST, when ``J`` shrinks a vector -- when its spectral radius is below 1. The code
stops after ``e_adjoint_max_iter`` terms and raises if the last term is still large.

A run of benchmark/cc/test_weighted_equiv.py on 100 Coleman families raised exactly that, in
float64, with ``fraction_missing`` on 610 of the 2013 species at values up to 0.4998:

    E-adjoint Neumann series failed to converge at conservative relative residual 6.069e-03
    after 512 terms (target 1.000e-12, dtype torch.float64)

Two very different things produce that message and they need opposite fixes:
  * ``J`` does not shrink at all (spectral radius at or above 1) -- the series is the WRONG method
    and no iteration budget saves it;
  * ``J`` shrinks, but only just -- the series is right and simply needs more terms.

This script tells them apart by measuring the shrink factor directly. Each Neumann term is
``J`` applied to the previous one, so the ratio of consecutive term norms IS the operator's
asymptotic shrink factor per term (its spectral radius). The script records every term norm of every
extinction-adjoint solve in one gradient call, at a sweep of ``fraction_missing`` levels, and reports
that ratio along with how many terms the requested tolerance would actually need.

Usage:
  python benchmark/cc/diagnose_fraction_missing_adjoint.py --species S --families LIST \
      --fitted-theta $CC_RUNS/results/full_v3.pt --limit 100 --clade-budget 315000 \
      --missing-levels 0.0 0.1 0.2 0.3 0.5 --missing-leaf-fraction 0.3 \
      --e-adjoint-max-iter 4096 --seed 0
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_weighted_equiv import (  # noqa: E402
    _random_fraction_missing, _random_weights, _solver_options, _theta,
)


def _install_term_norm_recorder():
    """Replace the extinction-adjoint solve with one that records every term norm it computes.

    The replacement runs the identical recurrence -- ``term <- J term``, ``x <- x + term`` -- so the
    answer is the same; it just never gives up and keeps the whole norm history. It is installed on
    the module the caller looks the name up on, which is the same module it is defined in.
    """
    from gpurec.api import _implicit_grad

    original = _implicit_grad._neumann_e_adjoint
    history: list[list[float]] = []

    @torch.no_grad()
    def recording(Av, b, *, max_iter, tol=None):
        norm_b = float(torch.linalg.vector_norm(b.reshape(-1)))
        if norm_b == 0.0:
            history.append([])
            return b.clone()
        x = b.clone()
        term = b.clone()
        norms = []
        for _ in range(int(max_iter)):
            term = term - Av(term)
            x = x + term
            value = float(torch.linalg.vector_norm(term.reshape(-1))) / norm_b
            norms.append(value)
            # Once a term is not a finite number there is nothing left to measure, and every later
            # term inherits the same NaN, so stop here and let the caller see where it happened.
            if not math.isfinite(value):
                break
        history.append(norms)
        return x

    _implicit_grad._neumann_e_adjoint = recording
    return original, history


# A term this small can no longer move a float64 sum, so the series has finished whether or not it
# literally reaches zero. Not a setting: it is float64's unit roundoff (2.22e-16) rounded down.
_DONE = 1e-16


def _terms_until_done(norms):
    """Index of the first term at or below the float64 roundoff, or None if it never gets there."""
    for index, value in enumerate(norms):
        if value <= _DONE:
            return index + 1
    return None


def _shrink_factor(norms):
    """Geometric shrink per term, measured over the tail of the terms that are still meaningful.

    The first terms are dominated by whichever directions the right-hand side happens to excite; the
    asymptotic rate is what the tail shows. Terms that have already underflowed to zero carry no
    rate information, so the tail is taken from the part of the history before that point.
    """
    useful = [value for value in norms if value > 0.0]
    if len(useful) < 8:
        return float("nan")
    start = len(useful) - max(4, len(useful) // 4)
    first, last = useful[start], useful[-1]
    steps = len(useful) - 1 - start
    if first <= 0.0 or last <= 0.0 or steps < 1:
        return float("nan")
    return (last / first) ** (1.0 / steps)


def _terms_needed(shrink, target):
    if not (0.0 < shrink < 1.0):
        return float("inf")
    return math.log(target) / math.log(shrink)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--species", required=True)
    parser.add_argument("--families", required=True)
    parser.add_argument("--fitted-theta", required=True)
    parser.add_argument("--limit", required=True, type=int)
    parser.add_argument("--clade-budget", required=True, type=int)
    parser.add_argument("--missing-levels", required=True, type=float, nargs="+",
                        help="upper end of the random fraction_missing values; 0 means none at all")
    parser.add_argument("--missing-leaf-fraction", required=True, type=float)
    parser.add_argument("--e-adjoint-max-iter", required=True, type=int,
                        help="how many Neumann terms the recorder runs (it never raises)")
    parser.add_argument("--receiver-scale", required=True, type=float)
    parser.add_argument("--origination-scale", required=True, type=float)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--per-family", required=True, type=int, choices=(0, 1),
                        help="1 to also rebuild the model one family at a time at the LAST level "
                             "and name the families whose own solve diverges. A batch's solve "
                             "covers many families at once, so this is what turns 'batch 1 blew "
                             "up' into 'these gene families blow up'.")
    args = parser.parse_args()

    from gpurec.api._execution import stream_genewise_loss_vector_grad
    from gpurec.api.model import GeneReconModel

    all_paths = [
        line.strip() for line in open(args.families) if line.strip() and not line.startswith("#")
    ]
    paths = all_paths[: args.limit]

    original, history = _install_term_norm_recorder()
    try:
        for level in args.missing_levels:
            missing = (
                None if level <= 0.0
                else _random_fraction_missing(
                    args.species, args.missing_leaf_fraction, level, args.seed
                )
            )
            options = _solver_options("exact", 64, 64, torch.float64, args.e_adjoint_max_iter)
            model = GeneReconModel(
                args.species, paths, mode="genewise", device="cuda", dtype=torch.float64,
                solver_options=options, clade_budget=args.clade_budget,
                fraction_missing=missing,
            )
            species_count = int(model.species_helpers["S"])
            receiver = _random_weights(
                species_count, args.receiver_scale, args.seed, torch.float64).cuda()
            origination = _random_weights(
                species_count, args.origination_scale, args.seed + 1, torch.float64).cuda()
            theta = _theta(args.fitted_theta, paths, torch.float64)

            # Does the FORWARD extinction fixed point converge at this level? The Neumann series is
            # justified by "the forward E fixed point converges, so its Jacobian is a contraction",
            # so it matters whether the first half of that sentence still holds. Running the forward
            # at two very different iteration caps and comparing the per-family likelihood answers
            # it: if the answer stops moving, the fixed point was reached.
            history.clear()
            model.configure_solver(e_max_iter=128)
            short = stream_genewise_loss_vector_grad(
                model.batch_statics, theta, receiver, origination,
                need_grad=False, update_warm_starts=False, need_origination_grad=False,
            )[0]
            model.configure_solver(e_max_iter=4096)
            long = stream_genewise_loss_vector_grad(
                model.batch_statics, theta, receiver, origination,
                need_grad=False, update_warm_starts=False, need_origination_grad=False,
            )[0]
            forward_move = float((long - short).abs().max())
            model.configure_solver(e_max_iter=512)

            history.clear()
            start = time.perf_counter()
            stream_genewise_loss_vector_grad(
                model.batch_statics, theta, receiver, origination,
                need_grad=True, update_warm_starts=False, need_origination_grad=True,
            )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            described = "none" if missing is None else (
                f"{len(missing)} species, values up to {max(missing.values()):.4f}"
            )
            print(f"\n[level {level:g}] fraction_missing: {described}   "
                  f"gradient took {elapsed:.1f}s over {len(history)} extinction-adjoint solves",
                  flush=True)
            print(
                f"  forward check: raising the extinction-step cap from 128 to 4096 iterations "
                f"moves the largest per-family likelihood by {forward_move:.4e} bits "
                + ("-- the forward fixed point IS converged here"
                   if forward_move < 1e-9 else "-- the forward has NOT settled here"),
                flush=True,
            )
            for index, norms in enumerate(history):
                if not norms:
                    print(f"  solve {index}: right-hand side was zero, nothing to solve", flush=True)
                    continue
                finite = [value for value in norms if math.isfinite(value)]
                smallest = min(finite) if finite else float("nan")
                smallest_at = finite.index(smallest) + 1 if finite else -1
                done = _terms_until_done(norms)
                shrink = _shrink_factor(norms)
                verdict = (
                    f"reached float64 roundoff after {done} terms" if done is not None
                    else (
                        f"went to {norms[-1]} at term {len(norms)}"
                        if norms and not math.isfinite(norms[-1])
                        else f"still {norms[-1]:.4e} after {len(norms)} terms"
                    )
                )
                print(f"  solve {index}: {verdict}", flush=True)
                print(
                    f"    smallest term seen {smallest:.4e} of the right-hand side, at term "
                    f"{smallest_at}; shrink per term over the tail = {shrink:.6f}"
                    + (f"; terms needed for 1e-12 = {_terms_needed(shrink, 1e-12):.0f}"
                       if 0.0 < shrink < 1.0 else ""),
                    flush=True,
                )
                # The whole shape of the sequence: a healthy solve falls monotonically; one that
                # turns around and climbs is an operator that grows a vector rather than shrinking it.
                probe = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
                shown = "  ".join(
                    f"{i + 1}:{norms[i]:.3e}" for i in probe if i < len(norms)
                )
                print(f"    term norms: {shown}", flush=True)
            del model, theta, receiver, origination
            torch.cuda.empty_cache()

        if args.per_family:
            level = args.missing_levels[-1]
            missing = _random_fraction_missing(
                args.species, args.missing_leaf_fraction, level, args.seed
            )
            print(f"\n[per-family at level {level:g}] one model per family, "
                  f"{len(paths)} families", flush=True)
            diverged = []
            for family_index, path in enumerate(paths):
                options = _solver_options("exact", 64, 64, torch.float64, 200)
                model = GeneReconModel(
                    args.species, [path], mode="genewise", device="cuda", dtype=torch.float64,
                    solver_options=options, clade_budget=args.clade_budget,
                    fraction_missing=missing,
                )
                species_count = int(model.species_helpers["S"])
                receiver = _random_weights(
                    species_count, args.receiver_scale, args.seed, torch.float64).cuda()
                origination = _random_weights(
                    species_count, args.origination_scale, args.seed + 1, torch.float64).cuda()
                theta = _theta(args.fitted_theta, [path], torch.float64)
                history.clear()
                stream_genewise_loss_vector_grad(
                    model.batch_statics, theta, receiver, origination,
                    need_grad=True, update_warm_starts=False, need_origination_grad=True,
                )
                torch.cuda.synchronize()
                worst = max(
                    (max(norms) for norms in history if norms), default=0.0
                )
                if not math.isfinite(worst) or worst > 1.0:
                    diverged.append((family_index, path, worst,
                                     [round(float(t), 4) for t in theta[0].tolist()]))
                del model, theta, receiver, origination
                torch.cuda.empty_cache()
            print(f"[per-family] {len(diverged)} of {len(paths)} families diverge on their own:",
                  flush=True)
            for family_index, path, worst, rates in diverged:
                print(f"   family {family_index}: largest term {worst:.4e}  "
                      f"theta (log2 rates D,T,L) = {rates}  {path}", flush=True)
    finally:
        from gpurec.api import _implicit_grad

        _implicit_grad._neumann_e_adjoint = original
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
