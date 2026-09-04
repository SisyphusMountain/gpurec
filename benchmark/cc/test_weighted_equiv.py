"""Do the EXACT tree solves still equal the iterated ones when the weights are NOT uniform?

``SolverOptions.forward_self_loop="exact"`` and ``adjoint_self_loop="exact"`` replace an iteration
by a direct elimination on the species tree, so their answer is what the iteration CONVERGES to.
That claim was checked at scale only with uniform transfer receiver weights (``receiver_weights``
all zero), which is what the genewise recipe uses. This script checks it with the weights turned on:

  * ``receiver_weights``    -- per-species logits shaping WHO receives a transfer. All-zero means
                              every species is an equally likely recipient; a non-zero vector makes
                              the transfer receiver distribution non-uniform and switches the
                              kernels onto their weighted code path.
  * ``origination_weights`` -- per-species logits for WHERE a gene family starts. All-zero means the
                              uniform origination prior. These enter only the NLL head, but the
                              backward pass through that head feeds the same adjoint solve.
  * ``fraction_missing``    -- per-leaf probability that a gene present in a species is not observed
                              there. It changes the leaf boundary of both the E recurrence and the
                              Pi recurrence, i.e. the very system the exact solve eliminates.

Five quantities are compared, all at once, between the two solver paths:

  1. per-family NLL                                       [G]      bits
  2. gradient w.r.t. theta (the 3 per-family DTL rates)   [G,3]
  3. gradient w.r.t. receiver_weights                     [S]
  4. gradient w.r.t. origination_weights                  [S]
  5. the genewise 3x3 curvature blocks                    [G,3,3]  (gpurec.fit.genewise_fit._analytic_hessian)

``exact``    = forward_self_loop "exact" + adjoint_self_loop "exact".  The Hessian-probe tangent
               follows ``adjoint_self_loop`` (see gpurec/solver/hvp/forward_tangent.py), so this
               setting also selects the exact tangent.
``iterated`` = forward_self_loop "log" at ``--pi-iters`` + adjoint_self_loop "series" at
               ``--neumann-terms`` with the early exit switched OFF (``neumann_term_tol=0``), and the
               tangent iterated the same number of times.  Run at 256/256 this is the converged
               reference, not the production 16/16 truncation.

In float64 the two must agree to ~1e-11 relative.  The script then repeats the pair in float32 and
measures BOTH against the float64 exact answer, so the report can say which of the two float32 paths
is closer to the truth rather than only that they differ from each other.

Adjoint pruning is switched OFF on both sides (``use_adjoint_pruning=False``): it drops adjoint
contributions below a threshold and is a separate approximation from the self-loop solve, so leaving
it on would blur what this script is measuring.  The E-adjoint tolerance is left at ``None`` so it
scales with the dtype (1e-6 in fp32, 1e-12 in fp64); the fitted 1e-7 of the genewise recipe is a
float32 number and would cap float64 six orders of magnitude early.

Usage:
  python benchmark/cc/test_weighted_equiv.py --species $CC_SPECIES --families $CC_FAMILIES \
      --fitted-theta $CC_RUNS/results/full_v3.pt --limit 100 --clade-budget 315000 \
      --pi-iters 256 --neumann-terms 256 --receiver-scale 0.6 --origination-scale 0.5 \
      --missing-leaf-fraction 0.3 --missing-max 0.5 --seed 0 --e-adjoint-max-iter 512 \
      --json-out out.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_exact_forward import _fitted_theta  # noqa: E402


# Both solver paths share everything except the self-loop implementations. Pruning off and the
# dtype-relative E-adjoint tolerance are stated here rather than inherited so the comparison is
# about the self-loop and nothing else.
_SHARED_SOLVER = dict(
    e_max_iter=512,
    e_adjoint_tol=None,
    adjoint_pruning_threshold=1e-6,
    use_adjoint_pruning=False,
    pibar_side_threshold=0.0,
)

# Convergence target for the extinction-probability fixed point, per model dtype. It has to sit just
# above the dtype's own resolution: 1e-8 is about 84x float32's eps (1.19e-7), which the iteration
# can actually reach, and 1e-15 about 4.5x float64's eps (2.22e-16). One number cannot serve both --
# float32 would never reach 1e-15 and would burn all 512 iterations on every call.
_E_TOL_BY_DTYPE = {torch.float32: 1e-8, torch.float64: 1e-15}


def _solver_options(mode, pi_iters, neumann_terms, dtype, e_adjoint_max_iter):
    from gpurec.api.solver_options import SolverOptions

    if mode == "exact":
        forward, adjoint = "exact", "exact"
    elif mode == "iterated":
        forward, adjoint = "log", "series"
    else:
        raise ValueError(f"unknown mode {mode!r}")
    return SolverOptions(
        **{
            **_SHARED_SOLVER,
            "e_tol": _E_TOL_BY_DTYPE[dtype],
            "e_adjoint_max_iter": int(e_adjoint_max_iter),
            "pi_iters": int(pi_iters),
            "neumann_terms": int(neumann_terms),
            # 0.0 disables the Neumann early exit, so the reference really runs every term.
            "neumann_term_tol": 0.0,
            "forward_self_loop": forward,
            "adjoint_self_loop": adjoint,
        }
    )


def _theta(source, paths, dtype):
    """Per-family rates [G,3] in log2 units.

    ``source`` is either a ``run_genewise.py`` .pt keyed by family path (the realistic setting: a
    real fit's rates, which is where the solves are actually used), or the literal ``flat:<value>``
    to give every family the same log2 rate -- used by the small-fixture smoke test, which has no
    fitted run to read.
    """
    if source.startswith("flat:"):
        return torch.full(
            (len(paths), 3), float(source.split(":", 1)[1]), device="cuda", dtype=dtype
        )
    return _fitted_theta(source, paths, "cuda", dtype)


def _random_weights(count, scale, seed, dtype):
    """Centered random logits [count], reproducible across dtypes and processes.

    Centered because both receiver_weights and origination_weights enter through a softmax over the
    S species, which is invariant to a constant shift: centering removes that null direction so the
    gradients being compared are the ones the optimizer actually sees.
    """
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    raw = torch.randn(int(count), generator=generator, dtype=torch.float64)
    raw = raw - raw.mean()
    return (float(scale) * raw).to(dtype=dtype)


def _random_fraction_missing(species_tree, leaf_fraction, maximum, seed):
    """{species_name: fraction} on a random subset of the species-tree leaves."""
    from gpurec.core.scheduling.batching import species_name_to_index

    names = sorted(species_name_to_index(str(species_tree)).keys())
    generator = torch.Generator(device="cpu").manual_seed(int(seed) + 7919)
    picked = torch.rand(len(names), generator=generator, dtype=torch.float64) < float(leaf_fraction)
    values = torch.rand(len(names), generator=generator, dtype=torch.float64) * float(maximum)
    return {
        name: float(values[index])
        for index, name in enumerate(names)
        if bool(picked[index])
    }


def _build(species, paths, clade_budget, dtype, options, fraction_missing, parsed, indices):
    from gpurec.api.model import GeneReconModel

    return GeneReconModel(
        species,
        paths,
        mode="genewise",
        device="cuda",
        dtype=dtype,
        solver_options=options,
        clade_budget=clade_budget,
        fraction_missing=fraction_missing,
        parsed_families=parsed,
        family_indices=indices,
    )


def _value_and_grads(model, theta, receiver_weights, origination_weights):
    """per-family NLL [G], d/dtheta [G,3], d/dreceiver_weights [S], d/dorigination_weights [S]."""
    from gpurec.api._execution import stream_genewise_loss_vector_grad

    loss, grad_theta, grad_receiver, grad_origination = stream_genewise_loss_vector_grad(
        model.batch_statics,
        theta,
        receiver_weights,
        origination_weights,
        need_grad=True,
        update_warm_starts=False,
        need_origination_grad=True,
    )
    torch.cuda.synchronize()
    return (
        loss.detach().clone(),
        grad_theta.detach().clone(),
        grad_receiver.detach().clone(),
        grad_origination.detach().clone(),
    )


def _hessian(model, theta, receiver_weights, tangent_iters):
    """The genewise [G,3,3] curvature blocks over every family in the model.

    ``_analytic_hessian`` reads the model's own ``receiver_weights`` buffer, so it is written first,
    and it indexes ``theta`` by each batch's absolute family ids -- so ``theta`` has to cover every
    family the model holds, not a prefix of them. ``tangent_self_iters`` is only consulted when the
    tangent iterates; the exact tangent ignores it. Note that ``_analytic_hessian`` never passes
    origination weights down, so these blocks are curvature under the UNIFORM origination prior on
    both sides -- which is exactly what the production genewise fit computes.
    """
    from gpurec.fit.genewise_fit import _analytic_hessian

    with torch.no_grad():
        model.receiver_weights.copy_(receiver_weights)
    hessian = _analytic_hessian(model, theta, int(tangent_iters), None, None)
    torch.cuda.synchronize()
    return hessian.detach().clone()


def _difference(name, reference, candidate):
    """max |candidate - reference| in absolute terms and relative to the reference's own scale."""
    reference64 = reference.to(torch.float64)
    candidate64 = candidate.to(torch.float64)
    absolute = (candidate64 - reference64).abs()
    scale = reference64.abs().amax()
    denominator = float(scale) if float(scale) > 0.0 else 1.0
    non_finite = int((~torch.isfinite(candidate64)).sum())
    return {
        "name": name,
        "count": int(reference64.numel()),
        "max_abs": float(absolute.max()),
        "mean_abs": float(absolute.mean()),
        "max_rel": float(absolute.max()) / denominator,
        "ref_max_abs": float(scale),
        "non_finite": non_finite,
    }


def _print_row(label, row):
    print(
        f"[{label}] {row['name']:<22} n={row['count']:<8} "
        f"max|diff| = {row['max_abs']:.4e}  mean|diff| = {row['mean_abs']:.4e}  "
        f"max|diff|/max|ref| = {row['max_rel']:.4e}  (max|ref| = {row['ref_max_abs']:.4e})"
        + (f"  NON-FINITE={row['non_finite']}" if row["non_finite"] else ""),
        flush=True,
    )


def _run_one(species, paths, clade_budget, dtype, mode, fraction_missing, parsed, indices,
             e_adjoint_max_iter,
             theta_source, receiver_scale, origination_scale, seed, pi_iters, neumann_terms,
             label):
    """Build a model in one dtype/mode, evaluate the five quantities, tear it down."""
    options = _solver_options(mode, pi_iters, neumann_terms, dtype, e_adjoint_max_iter)
    build_start = time.perf_counter()
    model = _build(species, paths, clade_budget, dtype, options, fraction_missing, parsed, indices)
    build_seconds = time.perf_counter() - build_start
    species_count = int(model.species_helpers["S"])

    theta = _theta(theta_source, paths, dtype)
    receiver = _random_weights(species_count, receiver_scale, seed, dtype).cuda()
    origination = _random_weights(species_count, origination_scale, seed + 1, dtype).cuda()

    evaluate_start = time.perf_counter()
    loss, grad_theta, grad_receiver, grad_origination = _value_and_grads(
        model, theta, receiver, origination
    )
    evaluate_seconds = time.perf_counter() - evaluate_start

    hessian_start = time.perf_counter()
    hessian = _hessian(model, theta, receiver, pi_iters)
    hessian_seconds = time.perf_counter() - hessian_start

    print(
        f"[{label}] build {build_seconds:.1f}s  loss+grad {evaluate_seconds:.2f}s  "
        f"hessian({hessian.shape[0]} families) {hessian_seconds:.2f}s  "
        f"S={species_count} batches={len(model.batch_statics)} "
        f"peak={torch.cuda.max_memory_allocated() / 2**30:.1f} GiB  "
        f"NLL sum {float(loss.to(torch.float64).sum()):.9f} bits",
        flush=True,
    )
    result = {
        "nll": loss,
        "grad_theta": grad_theta,
        "grad_receiver": grad_receiver,
        "grad_origination": grad_origination,
        "hessian": hessian,
    }
    del model, theta, receiver, origination
    torch.cuda.empty_cache()
    return result


_QUANTITIES = ("nll", "grad_theta", "grad_receiver", "grad_origination", "hessian")


def _compare(label, reference, candidate):
    rows = [_difference(name, reference[name], candidate[name]) for name in _QUANTITIES]
    for row in rows:
        _print_row(label, row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--species", required=True)
    parser.add_argument("--families", required=True)
    parser.add_argument("--fitted-theta", required=True,
                        help="run_genewise.py .pt with {'theta': [F,3], 'paths': [...]}, "
                             "or 'flat:<log2 rate>' to give every family the same rate")
    parser.add_argument("--limit", required=True, type=int,
                        help="how many families to compare on")
    parser.add_argument("--clade-budget", required=True, type=int)
    parser.add_argument("--pi-iters", required=True, type=int,
                        help="forward log-space iterations for the ITERATED path (256 = converged)")
    parser.add_argument("--neumann-terms", required=True, type=int,
                        help="adjoint series terms for the ITERATED path (256 = converged)")
    parser.add_argument("--receiver-scale", required=True, type=float,
                        help="standard deviation of the random receiver_weights logits")
    parser.add_argument("--origination-scale", required=True, type=float,
                        help="standard deviation of the random origination_weights logits")
    parser.add_argument("--missing-leaf-fraction", required=True, type=float,
                        help="fraction of species-tree leaves given a non-zero fraction_missing")
    parser.add_argument("--missing-max", required=True, type=float,
                        help="upper end of the random fraction_missing values")
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--e-adjoint-max-iter", required=True, type=int,
                        help="Neumann terms allowed for the extinction-adjoint linear solve. This "
                             "is NOT the wave self-loop being compared -- it is the separate solve "
                             "the gradient needs, and it is what a large fraction_missing makes "
                             "slow (see benchmark/cc/diagnose_fraction_missing_adjoint.py).")
    parser.add_argument("--json-out", required=True,
                        help="where the numbers are written; '-' to skip")
    args = parser.parse_args()

    from gpurec.core.scheduling.batching import parse_families

    all_paths = [
        line.strip() for line in open(args.families) if line.strip() and not line.startswith("#")
    ]
    paths = all_paths[: args.limit]
    parse_start = time.perf_counter()
    parsed = parse_families(args.species, paths)
    indices = list(range(len(paths)))
    print(
        f"[setup] parsed {len(paths)} families in {time.perf_counter() - parse_start:.1f}s; "
        f"receiver logits sd {args.receiver_scale}, origination logits sd "
        f"{args.origination_scale}, seed {args.seed}",
        flush=True,
    )

    missing = _random_fraction_missing(
        args.species, args.missing_leaf_fraction, args.missing_max, args.seed
    )
    print(
        f"[setup] fraction_missing on {len(missing)} species, values in "
        f"[{min(missing.values()):.4f}, {max(missing.values()):.4f}]"
        if missing else "[setup] fraction_missing dict came out empty",
        flush=True,
    )

    # Three weight settings, each run in both dtypes and both solver paths.
    #   uniform  -- the control: what was already validated at scale.
    #   weighted -- non-uniform receiver AND origination logits, no missing data.
    #   weighted+missing -- the same plus a random per-leaf fraction_missing.
    settings = [
        ("uniform", 0.0, 0.0, None),
        ("weighted", args.receiver_scale, args.origination_scale, None),
        ("weighted+missing", args.receiver_scale, args.origination_scale, missing),
    ]

    report = {
        "families": len(paths),
        "pi_iters": args.pi_iters,
        "neumann_terms": args.neumann_terms,
        "seed": args.seed,
        "receiver_scale": args.receiver_scale,
        "origination_scale": args.origination_scale,
        "fraction_missing_species": len(missing),
        "settings": {},
    }

    for setting_name, receiver_scale, origination_scale, fraction_missing in settings:
        print(f"\n[==== setting: {setting_name} ====]", flush=True)
        runs = {}
        failures = {}
        for dtype_name, dtype in (("float64", torch.float64), ("float32", torch.float32)):
            for mode in ("exact", "iterated"):
                label = f"{setting_name}/{dtype_name}/{mode}"
                # A solve that cannot be computed is a result, not a reason to lose the other
                # settings: record why and carry on, so one broken configuration does not take the
                # whole table down with it.
                try:
                    runs[(dtype_name, mode)] = _run_one(
                        args.species, paths, args.clade_budget, dtype, mode, fraction_missing,
                        parsed, indices, args.e_adjoint_max_iter,
                        args.fitted_theta, receiver_scale, origination_scale,
                        args.seed, args.pi_iters, args.neumann_terms, label,
                    )
                except RuntimeError as error:
                    failures[(dtype_name, mode)] = str(error)
                    print(f"[{label}] FAILED: {error}", flush=True)
                    torch.cuda.empty_cache()

        if failures:
            report["settings"][setting_name] = {"failures": {
                f"{dtype_name}/{mode}": message for (dtype_name, mode), message in failures.items()
            }}
            print(
                f"\n-- {setting_name}: {len(failures)} of the 4 runs could not be computed, so "
                f"there is nothing to compare here. What failed:", flush=True,
            )
            for (dtype_name, mode), message in sorted(failures.items()):
                print(f"   {dtype_name}/{mode}: {message}", flush=True)
            del runs, failures
            torch.cuda.empty_cache()
            continue

        entry = {}
        print(f"\n-- {setting_name}: float64 exact vs float64 iterated "
              f"(the equivalence claim; expect ~1e-11 relative) --", flush=True)
        entry["fp64_exact_vs_fp64_iterated"] = _compare(
            f"{setting_name} fp64", runs[("float64", "exact")], runs[("float64", "iterated")]
        )

        oracle = runs[("float64", "exact")]
        print(f"\n-- {setting_name}: float32 exact vs the float64 exact oracle --", flush=True)
        entry["fp32_exact_vs_oracle"] = _compare(
            f"{setting_name} fp32-exact", oracle, runs[("float32", "exact")]
        )
        print(f"\n-- {setting_name}: float32 iterated vs the float64 exact oracle --", flush=True)
        entry["fp32_iterated_vs_oracle"] = _compare(
            f"{setting_name} fp32-iter", oracle, runs[("float32", "iterated")]
        )

        print(f"\n-- {setting_name}: which float32 path is closer to the float64 oracle --",
              flush=True)
        for exact_row, iterated_row in zip(
            entry["fp32_exact_vs_oracle"], entry["fp32_iterated_vs_oracle"]
        ):
            winner = "exact" if exact_row["max_abs"] <= iterated_row["max_abs"] else "iterated"
            ratio = iterated_row["max_abs"] / exact_row["max_abs"] if exact_row["max_abs"] > 0 else float("inf")
            print(
                f"[{setting_name} fp32] {exact_row['name']:<22} exact {exact_row['max_abs']:.4e} vs "
                f"iterated {iterated_row['max_abs']:.4e} -> {winner} is closer "
                f"(iterated/exact = {ratio:.3g})",
                flush=True,
            )
        report["settings"][setting_name] = entry
        del runs, oracle
        torch.cuda.empty_cache()

    if args.json_out != "-":
        with open(args.json_out, "w") as handle:
            json.dump(report, handle, indent=2)
        print(f"\n[out] wrote {args.json_out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
