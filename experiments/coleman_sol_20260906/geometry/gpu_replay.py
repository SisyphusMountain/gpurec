"""Matched GPU replay of log-rate and hierarchical BFGS after the same Adam warm-up.

This is an experiment driver, not a production implementation.  It keeps the current
GPURec likelihood and pruning configuration, performs one shared three-gradient Adam
warm-up, then runs independent actual-NLL trust-region replays for:

* ``log2``: native log-rate BB plus BFGS warm-up seed (matched control);
* ``hier_raw``: hierarchical BFGS seeded by the diagonal complete information;
* ``hier_scaled``: the same diagonal multiplied by a scalar warm-up secant ratio.

Expected counts are extracted once at the post-Adam point by the portable sibling
experiment hook.  Every subsequent round is one ordinary gradient pass.  The script
records the clades in every model actually evaluated, including frozen families that
have not yet triggered a rebuild.  Rejected trials therefore remain charged work.
The final whole-population pass is an explicit projected-gradient certificate.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import sys
import time

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
EM_DIR = REPO_ROOT / "experiments" / "coleman_sol_20260906" / "em"
sys.path.insert(0, str(EM_DIR))

from counts_hook import counts_and_gradient  # noqa: E402
from gpurec.api.model import GeneReconModel  # noqa: E402
from gpurec.config import GpurecConfig  # noqa: E402
from gpurec.config.memory import MemoryOptions  # noqa: E402
from gpurec.core.memory_policy import clade_budget_for_device  # noqa: E402
from gpurec.core.scheduling.batching import DEFAULT_CLADE_BUDGET, parse_families  # noqa: E402
from gpurec.fit.genewise_fit import _analytic_hessian_blocks  # noqa: E402

from geometry_cpu import (  # noqa: E402
    BFGS_FLOOR,
    BOUND_EPS,
    HI,
    LO,
    bfgs_update,
    complete_information,
    inverse_jacobian,
    phi_to_theta,
    projected_gradient_max,
    theta_to_phi,
    transform_gradient,
    transform_hessian,
)


COMMON_START = (math.log2(0.01), math.log2(0.1), math.log2(0.01))
TRUST_SHRINK = 0.25
TRUST_GROW_RATIO = 0.75
TRUST_RADIUS_MIN = 0.5
TRUST_MIN_PREDICTED_BITS = 0.05
NLL_ACCEPTANCE_TOLERANCE = 0.005
CURVATURE_FLOOR = 1.0e-4
TOLERANCE = 1.0e-3


def read_paths(path: Path, limit: int) -> list[str]:
    with path.open() as handle:
        rows = [line.strip() for line in handle if line.strip() and not line.startswith("#")]
    if len(rows) < limit:
        raise ValueError(f"requested {limit} families but {path} contains {len(rows)}")
    return rows[:limit]


def derive_budget(parsed, n_families: int, device: torch.device) -> int:
    metadata = parsed.families(list(range(n_families)))
    budget, detail = clade_budget_for_device(
        total_clades=sum(int(row["C"]) for row in metadata),
        total_splits=sum(int(row["N_splits"]) for row in metadata),
        S=int(parsed.species()["S"]),
        dtype=torch.float32,
        device=device,
        fixed_clade_budget=DEFAULT_CLADE_BUDGET,
        scratch_tensors=MemoryOptions().scratch_tensors,
    )
    print(f"[hier] derived clade budget {int(budget):,}; automatic={detail['automatic']}", flush=True)
    return int(budget)


def build_model(
    species: str,
    all_paths: list[str],
    parsed,
    family_indices: torch.Tensor,
    clade_budget: int,
    config: GpurecConfig,
) -> GeneReconModel:
    indices = [int(index) for index in family_indices.tolist()]
    solver = replace(config.solver, pi_iters=16, neumann_terms=16)
    model = GeneReconModel(
        species,
        [all_paths[index] for index in indices],
        mode="genewise",
        device="cuda",
        dtype=torch.float32,
        solver_options=solver,
        config=config,
        clade_budget=clade_budget,
        parsed_families=parsed,
        family_indices=indices,
    )
    model.receiver_weights.requires_grad_(False)
    return model


def model_clades(model: GeneReconModel) -> torch.Tensor:
    return torch.tensor([int(family["C"]) for family in model.families], dtype=torch.float64)


def timed_gradient(model: GeneReconModel, theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float]:
    torch.cuda.synchronize()
    start = time.perf_counter()
    values, gradient, _ = model.genewise_loss_vector_and_grad(
        theta=theta.to(device="cuda", dtype=torch.float32).contiguous(), need_grad=True,
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return values.detach().double().cpu(), gradient.detach().double().cpu(), elapsed


def timed_counts(
    model: GeneReconModel, theta: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    torch.cuda.synchronize()
    start = time.perf_counter()
    values, gradient, counts = counts_and_gradient(model, theta)
    torch.cuda.synchronize()
    return values, gradient, counts, time.perf_counter() - start


def clamp_theta(theta: torch.Tensor) -> torch.Tensor:
    return torch.minimum(
        torch.maximum(theta, torch.full_like(theta, LO)), torch.full_like(theta, HI)
    )


def free_theta(theta: torch.Tensor, gradient: torch.Tensor) -> torch.Tensor:
    fixed = ((theta <= LO + BOUND_EPS) & (gradient > 0.0)) | (
        (theta >= HI - BOUND_EPS) & (gradient < 0.0)
    )
    return (~fixed).to(theta.dtype)


def map_forward(kind: str, theta: torch.Tensor) -> torch.Tensor:
    return theta if kind == "log2" else theta_to_phi(theta)


def map_inverse(kind: str, phi: torch.Tensor) -> torch.Tensor:
    return phi if kind == "log2" else phi_to_theta(phi)


def map_jacobian(kind: str, phi: torch.Tensor) -> torch.Tensor:
    if kind == "log2":
        return torch.eye(3, dtype=phi.dtype).expand(phi.shape[0], 3, 3)
    return inverse_jacobian(phi)


def map_gradient(kind: str, theta: torch.Tensor, gradient_theta: torch.Tensor) -> torch.Tensor:
    if kind == "log2":
        return gradient_theta
    return transform_gradient(theta_to_phi(theta), gradient_theta)


def projected_step(
    kind: str,
    theta: torch.Tensor,
    gradient_theta: torch.Tensor,
    curvature: torch.Tensor,
    radius: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Coordinate-native step projected through the original theta bounds.

    For log rates, currently active theta coordinates are removed exactly.  The
    hierarchical map is coupled, so its trial is mapped to theta, box-projected,
    and mapped back before prediction; actual NLL decides acceptance next round.
    """
    phi = map_forward(kind, theta)
    jacobian = map_jacobian(kind, phi)
    gradient = map_gradient(kind, theta, gradient_theta)
    if kind == "log2":
        free = free_theta(theta, gradient_theta)
        gradient = gradient * free
        curvature_work = curvature * free[:, :, None] * free[:, None, :] + torch.diag_embed(1.0 - free)
    else:
        curvature_work = curvature
    eigenvalues, eigenvectors = torch.linalg.eigh(curvature_work)
    gradient_eigen = torch.einsum("fji,fj->fi", eigenvectors, gradient)
    theta_scale = torch.einsum("fki,fij->fkj", jacobian, eigenvectors).norm(dim=1)
    adjusted = torch.maximum(
        torch.maximum(eigenvalues, CURVATURE_FLOOR * theta_scale.square()),
        gradient_eigen.abs() * theta_scale / radius[:, None],
    )
    direction = -torch.einsum("fij,fj->fi", eigenvectors, gradient_eigen / adjusted)

    def land(alpha: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        theta_raw = map_inverse(kind, phi + alpha[:, None] * direction)
        theta_bounded = clamp_theta(theta_raw)
        return theta_bounded, map_forward(kind, theta_bounded)

    theta_full, _ = land(torch.ones_like(radius))
    capped = (theta_full - theta).norm(dim=-1) > radius
    lower = torch.zeros_like(radius)
    upper = torch.ones_like(radius)
    for _ in range(50):
        middle = 0.5 * (lower + upper)
        theta_middle, _ = land(middle)
        over = (theta_middle - theta).norm(dim=-1) > radius
        upper = torch.where(over, middle, upper)
        lower = torch.where(over, lower, middle)
    alpha = torch.where(capped, lower, torch.ones_like(lower))
    theta_new, phi_new = land(alpha)
    applied = phi_new - phi
    predicted = -(
        (gradient * applied).sum(dim=-1)
        + 0.5 * torch.einsum("fi,fij,fj->f", applied, curvature_work, applied)
    )
    return theta_new, capped, predicted


def warmup(
    model: GeneReconModel,
    n_families: int,
    adam_steps: int,
    adam_lr: float,
) -> tuple[torch.Tensor, list[dict], list[tuple[torch.Tensor, torch.Tensor]], float]:
    theta_leaf = torch.tensor(COMMON_START, device="cuda", dtype=torch.float32).reshape(1, 3)
    theta_leaf = theta_leaf.repeat(n_families, 1).requires_grad_(True)
    optimizer = torch.optim.Adam([theta_leaf], lr=adam_lr)
    observations: list[tuple[torch.Tensor, torch.Tensor]] = []
    trace: list[dict] = []
    elapsed_total = 0.0
    for iteration in range(adam_steps):
        values, gradient, elapsed = timed_gradient(model, theta_leaf.detach())
        elapsed_total += elapsed
        theta_cpu = theta_leaf.detach().double().cpu().clone()
        observations.append((theta_cpu, gradient.clone()))
        trace.append({
            "iteration": iteration,
            "seconds": elapsed,
            "nll_bits": float(values.sum()),
            "pg_max": float(projected_gradient_max(theta_cpu, gradient).max()),
        })
        theta_leaf.grad = gradient.to(device="cuda", dtype=torch.float32)
        torch.nn.utils.clip_grad_norm_(theta_leaf, 10.0)
        with torch.no_grad():
            theta_leaf.grad *= free_theta(theta_leaf.detach().double().cpu(), gradient).to("cuda", torch.float32)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            theta_leaf.copy_(clamp_theta(theta_leaf))
        print(f"[hier] Adam {trace[-1]}", flush=True)
    return theta_leaf.detach().double().cpu(), trace, observations, elapsed_total


def log_seed_from_warmup(observations: list[tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
    pairs = [
        (observations[index][0] - observations[index - 1][0],
         observations[index][1] - observations[index - 1][1])
        for index in range(1, len(observations))
    ]
    last_step, last_change = pairs[-1]
    sy = (last_step * last_change).sum(dim=-1)
    good = sy > BFGS_FLOOR * last_step.norm(dim=-1) * last_change.norm(dim=-1)
    scale = torch.where(
        good,
        last_change.square().sum(dim=-1) / torch.where(good, sy, torch.ones_like(sy)),
        torch.ones_like(sy),
    )
    curvature = scale[:, None, None] * torch.eye(3).expand(last_step.shape[0], 3, 3)
    for step, change in pairs:
        curvature, _ = bfgs_update(curvature, step, change)
    return curvature


def hierarchy_seed(
    variant: str,
    theta_adam: torch.Tensor,
    gradient_adam: torch.Tensor,
    counts_adam: torch.Tensor,
    first_observation: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, dict]:
    phi_adam = theta_to_phi(theta_adam)
    information = complete_information(phi_adam, counts_adam)
    metadata = {"scale_median": 1.0, "scale_p90": 1.0, "scale_valid": theta_adam.shape[0]}
    if variant == "hier_raw":
        return information, metadata
    theta_start, gradient_start = first_observation
    step = phi_adam - theta_to_phi(theta_start)
    change = transform_gradient(phi_adam, gradient_adam) - transform_gradient(
        theta_to_phi(theta_start), gradient_start
    )
    numerator = (step * change).sum(dim=-1)
    denominator = (step * torch.einsum("fij,fj->fi", information, step)).sum(dim=-1)
    valid = (numerator > 0.0) & (denominator > 0.0)
    scale = torch.where(valid, numerator / denominator, torch.ones_like(numerator))
    scale = torch.minimum(torch.maximum(scale, torch.full_like(scale, 1.0e-3)), torch.full_like(scale, 1.0e3))
    metadata = {
        "scale_median": float(scale.median()),
        "scale_p90": float(scale.quantile(0.9)),
        "scale_valid": int(valid.sum()),
    }
    return scale[:, None, None] * information, metadata


def replay_one(
    variant: str,
    initial_theta: torch.Tensor,
    initial_nll: torch.Tensor,
    initial_gradient: torch.Tensor,
    initial_curvature: torch.Tensor,
    species: str,
    paths: list[str],
    parsed,
    clade_budget: int,
    config: GpurecConfig,
    all_clades: torch.Tensor,
    rounds: int,
    rebuild_fraction: float,
    trust: float,
    trust_max: float,
    hessian_refresh: int,
    hessian_pass_equivalent: float,
    initial_pass_seconds: float,
    warmup_seconds: float,
) -> tuple[dict, torch.Tensor]:
    kind = "log2" if variant == "log2" else "hier"
    n_families = initial_theta.shape[0]
    theta, nll, gradient, curvature = (
        initial_theta.clone(), initial_nll.clone(), initial_gradient.clone(), initial_curvature.clone()
    )
    radius = torch.full((n_families,), trust)
    frozen = projected_gradient_max(theta, gradient) < TOLERANCE
    total_clades = float(all_clades.sum())
    charged_clades = (3.0 + 1.0) * total_clades  # shared Adam plus this run's initial evaluation
    gradient_seconds = warmup_seconds + initial_pass_seconds
    build_seconds = 0.0
    rejected_total = 0
    hessian_seconds = 0.0
    hessian_refreshes = 0
    rows = torch.arange(n_families)
    model = build_model(species, paths, parsed, rows, clade_budget, config)
    clades_in_model = model_clades(model)
    trial, capped, predicted = projected_step(kind, theta, gradient, curvature, radius)
    trial = torch.where(frozen[:, None], theta, trial)
    trace = [{
        "evaluation": 0,
        "frozen": int(frozen.sum()),
        "live_clade_fraction": float(all_clades[~frozen].sum() / all_clades.sum()),
        "pg_max": float(projected_gradient_max(theta, gradient).max()),
        "nll_bits": float(nll.sum()),
        "charged_clade_passes": charged_clades / total_clades,
    }]

    for evaluation in range(1, rounds + 1):
        values_rows, gradient_rows, elapsed = timed_gradient(model, trial[rows])
        gradient_seconds += elapsed
        charged_clades += float(clades_in_model.sum())
        old_theta = theta[rows].clone()
        old_gradient = gradient[rows].clone()
        old_nll = nll[rows].clone()
        live_rows = ~frozen[rows]
        actual = old_nll - values_rows
        pending = live_rows & (predicted[rows] > TRUST_MIN_PREDICTED_BITS)
        ratio = torch.where(
            pending,
            actual / torch.where(pending, predicted[rows], torch.ones_like(actual)),
            torch.ones_like(actual),
        )
        current_radius = radius[rows]
        current_radius = torch.where(
            pending & (ratio < 0.25),
            torch.maximum(TRUST_SHRINK * current_radius, torch.full_like(current_radius, TRUST_RADIUS_MIN)),
            current_radius,
        )
        current_radius = torch.where(
            pending & (ratio > TRUST_GROW_RATIO) & capped[rows],
            torch.minimum(2.0 * current_radius, torch.full_like(current_radius, trust_max)),
            current_radius,
        )
        radius[rows] = current_radius
        # A nonlinear coupled projection is not covered by the local quadratic model.
        # Always require measured-NLL acceptance for it, even when the predicted
        # decrease is too small for a meaningful trust ratio.  The log-rate control
        # retains production's exact pending-test rule.
        if kind == "hier":
            reject = live_rows & (actual < -NLL_ACCEPTANCE_TOLERANCE)
            current_radius = torch.where(
                reject,
                torch.maximum(TRUST_SHRINK * current_radius,
                              torch.full_like(current_radius, TRUST_RADIUS_MIN)),
                current_radius,
            )
            radius[rows] = current_radius
        else:
            reject = pending & (actual < -TRUST_MIN_PREDICTED_BITS)
        rejected_total += int(reject.sum())

        evaluated_theta = trial[rows]
        old_phi = map_forward(kind, old_theta)
        evaluated_phi = map_forward(kind, evaluated_theta)
        old_gradient_phi = map_gradient(kind, old_theta, old_gradient)
        evaluated_gradient_phi = map_gradient(kind, evaluated_theta, gradient_rows)
        moved = live_rows.to(torch.float64)
        updated, _ = bfgs_update(
            curvature[rows],
            (evaluated_phi - old_phi) * moved[:, None],
            (evaluated_gradient_phi - old_gradient_phi) * moved[:, None],
        )
        curvature[rows] = updated
        accept = live_rows & ~reject
        theta[rows] = torch.where(accept[:, None], evaluated_theta, old_theta)
        gradient[rows] = torch.where(accept[:, None], gradient_rows, old_gradient)
        nll[rows] = torch.where(accept, values_rows, old_nll)

        if evaluation % 2 == 0:
            frozen |= projected_gradient_max(theta, gradient) < TOLERANCE
        live_clade_fraction = float(all_clades[~frozen].sum() / all_clades.sum())
        record = {
            "evaluation": evaluation,
            "seconds": elapsed,
            "model_families": int(rows.numel()),
            "model_clades": float(clades_in_model.sum()),
            "charged_clade_passes": charged_clades / total_clades,
            "frozen": int(frozen.sum()),
            "live_clade_fraction": live_clade_fraction,
            "nll_bits": float(nll.sum()),
            "pg_max": float(projected_gradient_max(theta, gradient).max()),
            "rejected": int(reject.sum()),
            "uphill": int((actual[live_rows] < 0.0).sum()),
            "radius_median": float(radius[~frozen].median()) if bool((~frozen).any()) else 0.0,
        }
        trace.append(record)
        print(f"[hier] {variant} {record}", flush=True)
        if bool(frozen.all()):
            break

        frozen_in_model = frozen[rows]
        if float(clades_in_model[frozen_in_model].sum()) >= rebuild_fraction * float(clades_in_model.sum()):
            rows = (~frozen).nonzero(as_tuple=True)[0]
            del model
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            build_start = time.perf_counter()
            model = build_model(species, paths, parsed, rows, clade_budget, config)
            torch.cuda.synchronize()
            build_seconds += time.perf_counter() - build_start
            clades_in_model = model_clades(model)

        if hessian_refresh > 0 and hessian_refreshes == 0 and evaluation >= hessian_refresh:
            torch.cuda.synchronize()
            hessian_start = time.perf_counter()
            hessian_theta, measured = _analytic_hessian_blocks(
                model,
                theta[rows].to(device="cuda", dtype=torch.float32).contiguous(),
                16,
                species,
                [paths[int(index)] for index in rows.tolist()],
                skip_batches_that_do_not_fit=True,
            )
            torch.cuda.synchronize()
            hessian_elapsed = time.perf_counter() - hessian_start
            hessian_seconds += hessian_elapsed
            hessian_refreshes += 1
            hessian_theta = hessian_theta.detach().double().cpu()
            measured = measured.detach().cpu()
            if kind == "log2":
                refreshed = hessian_theta
            else:
                refreshed = transform_hessian(
                    theta_to_phi(theta[rows]), gradient[rows], hessian_theta
                )
            curvature[rows] = torch.where(
                measured[:, None, None], refreshed, curvature[rows]
            )
            charged_clades += hessian_pass_equivalent * float(clades_in_model[measured].sum())
            print(
                f"[hier] {variant} exact refresh {hessian_refreshes}: "
                f"{int(measured.sum())}/{rows.numel()} families, {hessian_elapsed:.3f}s, "
                f"charged passes {charged_clades / total_clades:.3f}", flush=True,
            )

        trial_all, capped_all, predicted_all = projected_step(kind, theta, gradient, curvature, radius)
        trial = torch.where(frozen[:, None], theta, trial_all)
        capped = capped_all & ~frozen
        predicted = torch.where(frozen, torch.zeros_like(predicted_all), predicted_all)

    del model
    torch.cuda.empty_cache()
    summary = {
        "variant": variant,
        "rounds_run": len(trace) - 1,
        "frozen_before_certificate": int(frozen.sum()),
        "rejected_total": rejected_total,
        "charged_clade_passes_before_certificate": charged_clades / total_clades,
        "gradient_seconds_before_certificate": gradient_seconds,
        "rebuild_seconds": build_seconds,
        "hessian_seconds": hessian_seconds,
        "hessian_refreshes": hessian_refreshes,
        "trace": trace,
    }
    return summary, theta


def certificate(
    species: str,
    paths: list[str],
    parsed,
    clade_budget: int,
    config: GpurecConfig,
    theta: torch.Tensor,
) -> tuple[dict, torch.Tensor]:
    rows = torch.arange(theta.shape[0])
    model = build_model(species, paths, parsed, rows, clade_budget, config)
    values, gradient, elapsed = timed_gradient(model, theta)
    pg = projected_gradient_max(theta, gradient)
    result = {
        "seconds": elapsed,
        "nll_bits": float(values.sum()),
        "certified": int((pg < TOLERANCE).sum()),
        "pg_max": float(pg.max()),
        "pg_median": float(pg.median()),
    }
    del model
    torch.cuda.empty_cache()
    return result, values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--species", required=True)
    parser.add_argument("--families", required=True, type=Path)
    parser.add_argument("--limit", required=True, type=int)
    parser.add_argument("--clade-budget", required=True, type=int, help="0 derives it for this GPU")
    parser.add_argument("--rounds", required=True, type=int)
    parser.add_argument("--runs", required=True, help="comma list: log2,hier_raw,hier_scaled")
    parser.add_argument("--rebuild-fraction", required=True, type=float)
    parser.add_argument("--trust", required=True, type=float)
    parser.add_argument("--trust-max", required=True, type=float)
    parser.add_argument("--hessian-refresh", required=True, type=int, help="0 disables exact refresh")
    parser.add_argument("--hessian-pass-equivalent", required=True, type=float)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    variants = arguments.runs.split(",")
    allowed = {"log2", "hier_raw", "hier_scaled"}
    if not variants or any(variant not in allowed for variant in variants):
        raise ValueError(f"runs must be a comma list from {sorted(allowed)}")

    device = torch.device("cuda")
    config = GpurecConfig.genewise_reference()
    paths = read_paths(arguments.families, arguments.limit)
    parsed = parse_families(arguments.species, paths)
    clade_budget = arguments.clade_budget or derive_budget(parsed, len(paths), device)
    all_rows = torch.arange(len(paths))
    warm_model = build_model(arguments.species, paths, parsed, all_rows, clade_budget, config)
    all_clades = model_clades(warm_model)
    theta_adam, adam_trace, observations, warmup_seconds = warmup(
        warm_model, len(paths), adam_steps=3, adam_lr=1.0,
    )

    ordinary_nll, ordinary_gradient, ordinary_seconds = timed_gradient(warm_model, theta_adam)
    count_nll, count_gradient_theta, counts, count_seconds = timed_counts(warm_model, theta_adam)
    gradient_disagreement = float((ordinary_gradient - count_gradient_theta).abs().max())
    nll_disagreement = float((ordinary_nll - count_nll).abs().max())
    print(
        f"[hier] post-Adam ordinary {ordinary_seconds:.3f}s, counts {count_seconds:.3f}s; "
        f"max gradient disagreement {gradient_disagreement:.3e}, NLL {nll_disagreement:.3e}", flush=True,
    )
    del warm_model
    torch.cuda.empty_cache()

    summaries = {}
    final_thetas = {}
    final_nll_vectors = {}
    for variant in variants:
        if variant == "log2":
            curvature = log_seed_from_warmup(observations)
            seed_metadata = {}
            initial_nll, initial_gradient = ordinary_nll, ordinary_gradient
            initial_seconds = ordinary_seconds
        else:
            curvature, seed_metadata = hierarchy_seed(
                variant, theta_adam, count_gradient_theta, counts, observations[0]
            )
            initial_nll, initial_gradient = count_nll, count_gradient_theta
            initial_seconds = count_seconds
        summary, theta_final = replay_one(
            variant,
            theta_adam,
            initial_nll,
            initial_gradient,
            curvature,
            arguments.species,
            paths,
            parsed,
            clade_budget,
            config,
            all_clades,
            arguments.rounds,
            arguments.rebuild_fraction,
            arguments.trust,
            arguments.trust_max,
            arguments.hessian_refresh,
            arguments.hessian_pass_equivalent,
            initial_seconds,
            warmup_seconds,
        )
        cert, nll_vector = certificate(
            arguments.species, paths, parsed, clade_budget, config, theta_final
        )
        summary["seed"] = seed_metadata
        summary["certificate"] = cert
        summary["charged_clade_passes_with_certificate"] = (
            summary["charged_clade_passes_before_certificate"] + 1.0
        )
        summary["gradient_seconds_with_certificate"] = (
            summary["gradient_seconds_before_certificate"] + cert["seconds"]
        )
        summaries[variant] = summary
        final_thetas[variant] = theta_final.float()
        final_nll_vectors[variant] = nll_vector
        print("[hier] SUMMARY " + json.dumps(summary), flush=True)

    output = {
        "arguments": vars(arguments) | {"families": str(arguments.families), "output": str(arguments.output)},
        "clade_budget": clade_budget,
        "total_clades": float(all_clades.sum()),
        "adam_trace": adam_trace,
        "warmup_seconds": warmup_seconds,
        "post_adam_ordinary_seconds": ordinary_seconds,
        "post_adam_count_seconds": count_seconds,
        "post_adam_gradient_disagreement_max": gradient_disagreement,
        "post_adam_nll_disagreement_max": nll_disagreement,
        "summaries": summaries,
        "theta": final_thetas,
        "nll_per_family": final_nll_vectors,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, arguments.output)
    json_path = arguments.output.with_suffix(".json")
    json_path.write_text(json.dumps({key: value for key, value in output.items() if key not in {"theta", "nll_per_family"}}, indent=2, default=str) + "\n")
    print(f"[hier] wrote {arguments.output} and {json_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
