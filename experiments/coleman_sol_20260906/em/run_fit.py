"""Production-faithful Coleman subset fit with zero to three EM warm steps.

The baseline is the current ``fit_dtl`` genewise recipe.  EM variants replace the
three Adam passes with exact boxed M-steps and then enter the unchanged production
BFGS/Newton fit.  The prototype builds a separate model for the warm passes; both
the observed wall cost and the reusable-kernel phase cost are reported explicitly.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import math
import time

import torch

from counts_hook import counts_and_gradient
from mstep import HI, LO, complete_information, m_step
from gpurec.api.model import GeneReconModel
from gpurec.config import GpurecConfig
from gpurec.config.memory import MemoryOptions
from gpurec.core.memory_policy import clade_budget_for_device
from gpurec.core.scheduling.batching import DEFAULT_CLADE_BUDGET, parse_families
from gpurec.fit.genewise_fit import TRUST_TEST_OFF, _bfgs_update, fit_genewise


COMMON_START = (math.log2(0.01), math.log2(0.1), math.log2(0.01))
_ORIGINAL_GRADIENT_METHOD = GeneReconModel.genewise_loss_vector_and_grad
_TRACE_PHASE = ["unclassified"]
_GRADIENT_TRACE: list[dict] = []


def read_paths(path: str, limit: int) -> list[str]:
    with open(path) as handle:
        rows = [line.strip() for line in handle if line.strip() and not line.startswith("#")]
    return rows if limit == 0 else rows[:limit]


def _traced_gradient_method(self, *args, **kwargs):
    """Record every actual gradient-model call, including verification subsets."""
    torch.cuda.synchronize()
    started = time.perf_counter()
    result = _ORIGINAL_GRADIENT_METHOD(self, *args, **kwargs)
    torch.cuda.synchronize()
    solver = self.solver_options
    _GRADIENT_TRACE.append({
        "phase": _TRACE_PHASE[0],
        "families": len(self.families),
        "clades": sum(int(family["C"]) for family in self.families),
        "pi_iters": int(solver.pi_iters),
        "neumann_terms": int(solver.neumann_terms),
        "need_grad": bool(kwargs.get("need_grad", True)),
        "seconds": time.perf_counter() - started,
    })
    return result


def start_gradient_trace() -> None:
    _GRADIENT_TRACE.clear()
    GeneReconModel.genewise_loss_vector_and_grad = _traced_gradient_method


def stop_gradient_trace() -> None:
    GeneReconModel.genewise_loss_vector_and_grad = _ORIGINAL_GRADIENT_METHOD


def derive_budget(parsed, n_families: int, dtype: torch.dtype, device: torch.device) -> int:
    meta = parsed.families(list(range(n_families)))
    budget, detail = clade_budget_for_device(
        total_clades=sum(int(row["C"]) for row in meta),
        total_splits=sum(int(row["N_splits"]) for row in meta),
        S=int(parsed.species()["S"]), dtype=dtype, device=device,
        fixed_clade_budget=DEFAULT_CLADE_BUDGET,
        scratch_tensors=MemoryOptions().scratch_tensors,
    )
    print(f"[emfit] derived clade budget {budget:,}; automatic={detail['automatic']}", flush=True)
    return int(budget)


def free_at_both(theta0: torch.Tensor, theta1: torch.Tensor) -> torch.Tensor:
    eps = 1e-6
    return (((theta0 > LO + eps) & (theta0 < HI - eps))
            & ((theta1 > LO + eps) & (theta1 < HI - eps))).to(theta0.dtype)


def warm_em(
    species: str,
    paths: list[str],
    parsed,
    budget: int,
    steps: int,
    seed: str,
    config: GpurecConfig,
) -> tuple[torch.Tensor, torch.Tensor, dict, list[dict]]:
    solver = replace(config.solver, pi_iters=16, neumann_terms=16)
    t_build = time.perf_counter()
    model = GeneReconModel(
        species, paths, mode="genewise", device="cuda", dtype=torch.float32,
        solver_options=solver, config=config, clade_budget=budget,
        parsed_families=parsed, family_indices=list(range(len(paths))),
    )
    model.receiver_weights.requires_grad_(False)
    torch.cuda.synchronize()
    build_seconds = time.perf_counter() - t_build
    theta = torch.tensor(COMMON_START, dtype=torch.float64).reshape(1, 3).repeat(len(paths), 1)
    curvature = None
    previous_theta = previous_gradient = None
    traces: list[dict] = []
    pass_seconds = 0.0
    for step in range(steps):
        _TRACE_PHASE[0] = "warm_count"
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        nll, gradient, counts = counts_and_gradient(model, theta)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        pass_seconds += elapsed
        next_theta, status, kkt = m_step(counts)
        info = complete_information(theta, counts)
        endpoint_info = complete_information(next_theta, counts)
        if curvature is None:
            curvature = endpoint_info if seed in ("ic_endpoint", "endpoint_scaled_secants") else info
        elif seed == "ic_latest":
            curvature = info
        elif seed in ("ic_endpoint", "endpoint_scaled_secants"):
            curvature = endpoint_info
        if seed == "ic_first_secants" and previous_theta is not None:
            curvature = _bfgs_update(
                curvature,
                theta - previous_theta,
                gradient - previous_gradient,
                free_at_both(previous_theta, theta),
            )
        if seed == "endpoint_scaled_secants" and previous_theta is not None:
            step_pair = theta - previous_theta
            grad_pair = gradient - previous_gradient
            i_step = torch.einsum("gij,gj->gi", curvature, step_pair)
            sy = (step_pair * grad_pair).sum(dim=1)
            s_is = (step_pair * i_step).sum(dim=1)
            valid = (sy > 0.0) & (s_is > 0.0) & torch.isfinite(sy) & torch.isfinite(s_is)
            scale = torch.where(valid, sy / s_is, torch.ones_like(sy))
            curvature = curvature * scale[:, None, None]
            curvature = _bfgs_update(
                curvature, step_pair, grad_pair, free_at_both(previous_theta, theta),
            )
        traces.append({
            "step": step + 1,
            "seconds": elapsed,
            "nll_bits": float(nll.sum()),
            "pg_max": float(gradient.abs().amax()),
            "step_median": float((next_theta - theta).norm(dim=1).median()),
            "step_p90": float((next_theta - theta).norm(dim=1).quantile(0.9)),
            "pinned": int((status != 0).sum()),
            "kkt_residual_max": float(kkt.max()),
        })
        print(f"[emfit] warm {traces[-1]}", flush=True)
        previous_theta, previous_gradient = theta, gradient
        theta = next_theta
    del model
    torch.cuda.empty_cache()
    assert curvature is not None
    return theta.float(), curvature.float(), {
        "build_seconds": build_seconds,
        "pass_seconds": pass_seconds,
        "prototype_warm_seconds": build_seconds + pass_seconds,
    }, traces


def accurate_nll_vector(
    species: str, paths: list[str], parsed, budget: int, theta: torch.Tensor, config: GpurecConfig,
) -> tuple[torch.Tensor, float]:
    solver = replace(config.solver, pi_iters=64, neumann_terms=64)
    t0 = time.perf_counter()
    model = GeneReconModel(
        species, paths, mode="genewise", device="cuda", dtype=torch.float32,
        solver_options=solver, config=config, clade_budget=budget,
        parsed_families=parsed, family_indices=list(range(len(paths))),
    )
    model.receiver_weights.requires_grad_(False)
    with torch.no_grad():
        values = model.genewise_loss_vector(theta=theta.to(device="cuda", dtype=torch.float32)).double().cpu()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    del model
    torch.cuda.empty_cache()
    return values, elapsed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--species", required=True)
    parser.add_argument("--families", required=True)
    parser.add_argument("--limit", required=True, type=int)
    parser.add_argument("--clade-budget", required=True, type=int)
    parser.add_argument("--em-steps", required=True, type=int, choices=(0, 1, 2, 3))
    parser.add_argument(
        "--seed", required=True,
        choices=("baseline", "ic_latest", "ic_first_secants", "ic_endpoint", "endpoint_scaled_secants"),
    )
    parser.add_argument("--tag", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if (args.em_steps == 0) != (args.seed == "baseline"):
        raise ValueError("em-steps=0 requires seed=baseline; EM steps require an Ic seed")

    config = GpurecConfig.genewise_reference()
    paths = read_paths(args.families, args.limit)
    parse_t0 = time.perf_counter()
    parsed = parse_families(args.species, paths)
    parse_seconds = time.perf_counter() - parse_t0
    budget = args.clade_budget or derive_budget(parsed, len(paths), torch.float32, torch.device("cuda"))
    initial_clades = sum(int(row["C"]) for row in parsed.families(list(range(len(paths)))))
    print(f"[emfit] BEGIN {args.tag}: F={len(paths)} em_steps={args.em_steps} seed={args.seed} budget={budget}", flush=True)

    warm_summary = {"build_seconds": 0.0, "pass_seconds": 0.0, "prototype_warm_seconds": 0.0}
    warm_trace: list[dict] = []
    start_gradient_trace()
    try:
        if args.em_steps:
            theta0, curvature0, warm_summary, warm_trace = warm_em(
                args.species, paths, parsed, budget, args.em_steps, args.seed, config,
            )
            adam_steps = 0
        else:
            theta0, curvature0, adam_steps = COMMON_START, "adam_bfgs", 3

        _TRACE_PHASE[0] = "production_fit"
        torch.cuda.synchronize()
        fit_t0 = time.perf_counter()
        result = fit_genewise(
            args.species, paths, device="cuda", dtype="float32",
            adam_steps=adam_steps, adam_lr=1.0,
            clade_budget=budget, tol=1e-3, max_iter=120, check_every=2,
            min_drop=32, rebuild_frac=0.25, hessian_refresh=15,
            init_curvature=curvature0, curvature_update="bfgs",
            trust=2.0, trust_max=8.0, mu=1e-4,
            certify=True, certify_curvature=False,
            init_log2_rates=theta0, stall_patience=120,
            step_extrapolation=1.0, step_model="quadratic",
            stop_nll_bits=0.0, approach_pruning_threshold=0.0,
            targeted_hessian=(0, 0.0), coordinate_staging=(0, 0), trust_test=TRUST_TEST_OFF,
            solver_options=None, config=config, verbose=True,
        )
        torch.cuda.synchronize()
        fit_wall = time.perf_counter() - fit_t0
    finally:
        stop_gradient_trace()
    theta_final = result["theta"].detach().float().cpu()
    nll_vector, nll_vector_seconds = accurate_nll_vector(
        args.species, paths, parsed, budget, theta_final, config,
    )
    row = {
        "tag": args.tag,
        "n_families": len(paths),
        "em_steps": args.em_steps,
        "seed": args.seed,
        "clade_budget": budget,
        "parse_seconds": parse_seconds,
        **warm_summary,
        "fit_wall_seconds": fit_wall,
        "algorithm_wall_seconds": warm_summary["pass_seconds"] + fit_wall,
        "prototype_wall_seconds": parse_seconds + warm_summary["prototype_warm_seconds"] + fit_wall,
        "nll_vector_seconds": nll_vector_seconds,
        "nll_bits_fit": float(result["loss_bits"]),
        "nll_bits_vector": float(nll_vector.sum()),
        "n_steps": int(result["n_steps"]),
        "n_hessians": int(result["n_hessians"]),
        "n_rebuilds": int(result["n_rebuilds"]),
        "converged": int(result["converged"]),
        "unconverged": int(result["unconverged"]),
        "bound_active": int(result["bound_active"]),
        "pg_max": float(result["pg_max"]),
        "adam_seconds": float(result["adam_seconds"]),
        "newton_grad_seconds": float(result["newton_grad_seconds"]),
        "hessian_seconds": float(result["hessian_seconds"]),
        "rebuild_seconds": float(result["rebuild_seconds"]),
        "certify_seconds": float(result["certify_seconds"]),
        "gradient_calls": len(_GRADIENT_TRACE),
        "gradient_clades_total": sum(record["clades"] for record in _GRADIENT_TRACE),
        "gradient_clade_equivalents": sum(record["clades"] for record in _GRADIENT_TRACE) / initial_clades,
        "gradient_seconds_traced": sum(record["seconds"] for record in _GRADIENT_TRACE),
        "warm_count_calls": sum(record["phase"] == "warm_count" for record in _GRADIENT_TRACE),
        "fit_gradient_calls": sum(record["phase"] == "production_fit" for record in _GRADIENT_TRACE),
        "config_solver": asdict(config.solver),
        "config_precision": asdict(config.precision),
    }
    torch.save({
        "row": row, "warm_trace": warm_trace, "gradient_trace": _GRADIENT_TRACE, "paths": paths,
        "theta": theta_final, "nll_per_family": nll_vector,
        "curvature": result["curvature"].detach().float().cpu(),
    }, args.out)
    print("[emfit] SUMMARY " + json.dumps(row), flush=True)
    print(f"[emfit] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
