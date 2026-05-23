"""Legacy checkout-local HOGENOM experiment launcher.

Prefer the installed ``gpurec optimize`` workflow for supported production
runs.  This script is retained for reproducing historical HOGENOM experiments
with experiment-specific optimizer and reporting logic.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from gpurec import GeneReconModel  # noqa: E402
from gpurec.optimization import BatchedLBFGS  # noqa: E402


HOGENOM_DIR = REPO / "tests" / "data" / "HOGENOM" / "hogenom"
FAMILIES_FILE = HOGENOM_DIR / "hogenom_families.local.txt"
ALERAX_OUTPUT = HOGENOM_DIR / "output_alerax_corrected"
INFERRED_SPECIES_TREE = ALERAX_OUTPUT / "species_trees" / "inferred_species_tree.newick"
SPECIES_TREE = (
    INFERRED_SPECIES_TREE
    if INFERRED_SPECIES_TREE.exists()
    else HOGENOM_DIR / "hogenom_S.tree"
)
OUT_DIR = HOGENOM_DIR / "output_gpurec_fast_ccp_opt"

LN2 = math.log(2.0)
RATE_QUANTILES = (0.0, 0.05, 0.5, 0.95, 1.0)


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def theta_logits(theta: torch.Tensor) -> torch.Tensor:
    theta2 = theta.reshape(-1, 3)
    zeros = theta2.new_zeros((theta2.shape[0], 1))
    return torch.cat((zeros, theta2), dim=1) * LN2


def pS_values(theta: torch.Tensor) -> torch.Tensor:
    return torch.softmax(theta_logits(theta), dim=1)[:, 0]


def regularization_vector(theta: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    theta2 = theta.reshape(-1, 3)
    if args.regularization == "none" or args.regularization_weight == 0.0:
        return theta2.new_zeros((theta2.shape[0],))

    if args.regularization == "square-theta":
        center = theta2.new_tensor(args.regularization_center)
        penalty = (theta2 - center).square().sum(dim=1)
    elif args.regularization == "gaussian-theta":
        center = theta2.new_tensor(args.regularization_center)
        std = theta2.new_tensor(args.regularization_std)
        penalty = 0.5 * ((theta2 - center) / std).square().sum(dim=1)
    elif args.regularization == "beta-ps":
        logits = theta_logits(theta)
        log_probs = torch.log_softmax(logits, dim=1)
        log_pS = log_probs[:, 0] / LN2
        log_not_pS = torch.logsumexp(logits[:, 1:], dim=1) / LN2 - torch.logsumexp(
            logits,
            dim=1,
        ) / LN2
        penalty = -(
            (args.beta_ps_alpha - 1.0) * log_pS
            + (args.beta_ps_beta - 1.0) * log_not_pS
        )
    else:
        raise ValueError(f"unknown regularization {args.regularization!r}")

    return penalty * args.regularization_weight


def regularization(theta: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    return regularization_vector(theta, args).sum()


def regularization_grad(theta: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    if args.regularization == "none" or args.regularization_weight == 0.0:
        return torch.zeros_like(theta)
    penalty = regularization_vector(theta, args).sum()
    (grad,) = torch.autograd.grad(penalty, theta, retain_graph=False, create_graph=False)
    return grad.detach()


def rate_summary(theta: torch.Tensor) -> dict[str, dict[str, float]]:
    theta2 = theta.detach().reshape(-1, 3)
    rates = torch.exp2(theta2).float().cpu()
    pS = pS_values(theta.detach()).float().cpu()
    out: dict[str, dict[str, float]] = {}
    for name, column in (("D", 0), ("T", 2), ("L", 1)):
        values = rates[:, column]
        qs = torch.quantile(values, torch.tensor(RATE_QUANTILES))
        out[name] = {
            "min": float(qs[0]),
            "p05": float(qs[1]),
            "median": float(qs[2]),
            "p95": float(qs[3]),
            "max": float(qs[4]),
            "mean": float(values.mean()),
        }
    qs = torch.quantile(pS, torch.tensor(RATE_QUANTILES))
    out["pS"] = {
        "min": float(qs[0]),
        "p05": float(qs[1]),
        "median": float(qs[2]),
        "p95": float(qs[3]),
        "max": float(qs[4]),
        "mean": float(pS.mean()),
    }
    return out


def format_rate_summary(summary: dict[str, dict[str, float]]) -> str:
    return " ".join(
        f"{name}[min={vals['min']:.3g} med={vals['median']:.3g} "
        f"p95={vals['p95']:.3g} max={vals['max']:.3g}]"
        for name, vals in summary.items()
    )


def evaluate(
    model: GeneReconModel,
    args: argparse.Namespace,
    *,
    zero_grad: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    if zero_grad:
        model.theta.grad = None
    synchronize()
    start = time.perf_counter()
    data_loss = model.full_loss()
    penalty = regularization(model.theta, args)
    objective = data_loss + penalty
    objective.backward()
    synchronize()
    elapsed = time.perf_counter() - start
    if model.theta.grad is None:
        raise RuntimeError("missing theta gradient")
    grad = model.theta.grad.detach()
    metrics = {
        "data_nll_bits": float(data_loss.detach().cpu()),
        "regularization_bits": float(penalty.detach().cpu()),
        "objective_bits": float(objective.detach().cpu()),
        "log_likelihood_bits": float(-data_loss.detach().cpu()),
        "grad_inf": float(grad.abs().amax().cpu()),
        "grad_norm": float(torch.linalg.vector_norm(grad).cpu()),
        "eval_s": elapsed,
    }
    metrics.update(gradient_constraint_metrics(model.theta.detach(), grad, args))
    metrics.update(solver_iteration_metrics(model))
    return objective, metrics


def clamp_rates(model: GeneReconModel, args: argparse.Namespace) -> None:
    model.clamp_theta_(min_rate=args.min_rate, max_rate=args.max_rate)


def tensor_is_finite(value: torch.Tensor) -> bool:
    return bool(torch.isfinite(value).all().item())


def metrics_are_finite(metrics: dict[str, float]) -> bool:
    return all(math.isfinite(float(value)) for value in metrics.values())


def _add_iteration_distribution(
    metrics: dict[str, float],
    prefix: str,
    values: list[int],
) -> None:
    if not values:
        return
    metrics[f"{prefix}_min"] = float(min(values))
    metrics[f"{prefix}_max"] = float(max(values))
    metrics[f"{prefix}_mean"] = float(sum(values) / len(values))
    for value, count in sorted(Counter(values).items()):
        metrics[f"{prefix}_count_{value}"] = float(count)


def solver_iteration_metrics(model: GeneReconModel) -> dict[str, float]:
    records = model.solver_stat_records()
    if not records:
        return {}

    e_iters = [int(stats["E_iterations"]) for stats in records]
    pi_iters: list[int] = []
    for stats in records:
        wave_iterations = stats.get("Pi_wave_iterations")
        if wave_iterations:
            pi_iters.extend(int(value) for value in wave_iterations)
        else:
            pi_iters.append(int(stats["Pi_max_iterations"]))
    pi_converged = sum(float(stats["Pi_converged_waves"]) for stats in records)
    pi_waves = sum(float(stats["Pi_wave_count"]) for stats in records)
    metrics: dict[str, float] = {
        "solver_Pi_converged_waves": pi_converged,
        "solver_Pi_wave_count": pi_waves,
    }
    _add_iteration_distribution(metrics, "solver_E_iterations", e_iters)
    _add_iteration_distribution(metrics, "solver_Pi_iterations", pi_iters)
    neumann_terms = [
        int(stats["Neumann_terms"])
        for stats in records
        if "Neumann_terms" in stats
    ]
    _add_iteration_distribution(metrics, "solver_Neumann_terms", neumann_terms)
    e_adjoint_iters = [
        int(stats["E_adjoint_iterations"])
        for stats in records
        if "E_adjoint_iterations" in stats
    ]
    _add_iteration_distribution(
        metrics,
        "solver_E_adjoint_iterations",
        e_adjoint_iters,
    )
    return metrics


def restore_theta(model: GeneReconModel, theta: torch.Tensor) -> None:
    with torch.no_grad():
        model.theta.copy_(theta)
        model.theta.grad = None
    model.clear()


def optimizer_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def tensor_cosine(
    previous: torch.Tensor | None,
    current: torch.Tensor | None,
) -> float | None:
    if previous is None or current is None:
        return None
    if not tensor_is_finite(previous) or not tensor_is_finite(current):
        return None
    previous_flat = previous.detach().reshape(-1)
    current_flat = current.detach().reshape(-1)
    denom = torch.linalg.vector_norm(previous_flat) * torch.linalg.vector_norm(current_flat)
    if float(denom.cpu()) == 0.0:
        return None
    return float(torch.dot(previous_flat, current_flat).div(denom).cpu())


def sign_flip_fraction(
    previous: torch.Tensor | None,
    current: torch.Tensor | None,
) -> float | None:
    if previous is None or current is None:
        return None
    if not tensor_is_finite(previous) or not tensor_is_finite(current):
        return None
    active = (previous != 0) & (current != 0)
    active_count = int(active.sum().cpu())
    if active_count == 0:
        return None
    flips = (previous.sign() != current.sign()) & active
    return float(flips.sum().cpu()) / active_count


def maybe_decay_adam_lr_for_oscillation(
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    *,
    previous_grad: torch.Tensor | None,
    current_grad: torch.Tensor | None,
    previous_step: torch.Tensor | None,
    current_step: torch.Tensor | None,
    cooldown_remaining: int,
) -> tuple[dict[str, Any], int]:
    mode = args.adam_oscillation_detection
    old_lr = optimizer_lr(optimizer)
    metrics: dict[str, Any] = {
        "adam_lr": old_lr,
        "adam_lr_reduced": False,
        "adam_oscillating": False,
        "adam_oscillation_cooldown": cooldown_remaining,
    }
    if mode == "off":
        return metrics, cooldown_remaining

    grad_cos = None
    grad_flips = None
    step_cos = None
    step_flips = None
    reasons: list[str] = []

    if mode in {"gradient", "both"}:
        grad_cos = tensor_cosine(previous_grad, current_grad)
        grad_flips = sign_flip_fraction(previous_grad, current_grad)
        metrics["adam_grad_cosine"] = grad_cos
        metrics["adam_grad_flip_fraction"] = grad_flips
        if grad_cos is not None and grad_cos <= args.adam_oscillation_cos_threshold:
            reasons.append(f"gradient_cosine={grad_cos:.3g}")
        if (
            grad_flips is not None
            and grad_flips >= args.adam_oscillation_flip_fraction
        ):
            reasons.append(f"gradient_flips={grad_flips:.3g}")

    if mode in {"parameters", "both"}:
        step_cos = tensor_cosine(previous_step, current_step)
        step_flips = sign_flip_fraction(previous_step, current_step)
        metrics["adam_step_cosine"] = step_cos
        metrics["adam_step_flip_fraction"] = step_flips
        if step_cos is not None and step_cos <= args.adam_oscillation_cos_threshold:
            reasons.append(f"step_cosine={step_cos:.3g}")
        if (
            step_flips is not None
            and step_flips >= args.adam_oscillation_flip_fraction
        ):
            reasons.append(f"step_flips={step_flips:.3g}")

    if cooldown_remaining > 0:
        cooldown_remaining -= 1
        metrics["adam_oscillation_cooldown"] = cooldown_remaining
        if reasons:
            metrics["adam_oscillating"] = True
            metrics["adam_oscillation_reason"] = ",".join(reasons)
        return metrics, cooldown_remaining

    if not reasons:
        return metrics, cooldown_remaining

    metrics["adam_oscillating"] = True
    metrics["adam_oscillation_reason"] = ",".join(reasons)
    new_lr = max(args.adam_min_lr, old_lr * args.adam_oscillation_lr_decay)
    if new_lr < old_lr:
        set_optimizer_lr(optimizer, new_lr)
        cooldown_remaining = max(0, args.adam_oscillation_cooldown)
        metrics["adam_lr"] = new_lr
        metrics["adam_lr_previous"] = old_lr
        metrics["adam_lr_reduced"] = True
        metrics["adam_oscillation_cooldown"] = cooldown_remaining
    else:
        metrics["adam_lr_at_min"] = True
    return metrics, cooldown_remaining


def projected_gradient(
    theta: torch.Tensor,
    grad: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    lower = math.log2(args.min_rate)
    upper = math.log2(args.max_rate)
    bound_tol = 1e-6
    blocked_lower = (theta <= lower + bound_tol) & (grad > 0)
    blocked_upper = (theta >= upper - bound_tol) & (grad < 0)
    return torch.where(blocked_lower | blocked_upper, torch.zeros_like(grad), grad)


def gradient_constraint_metrics(
    theta: torch.Tensor,
    grad: torch.Tensor,
    args: argparse.Namespace,
) -> dict[str, float]:
    projected = projected_gradient(theta, grad, args)
    lower = math.log2(args.min_rate)
    upper = math.log2(args.max_rate)
    bound_tol = 1e-6
    lower_bound = theta <= lower + bound_tol
    upper_bound = theta >= upper - bound_tol
    return {
        "projected_grad_inf": float(projected.abs().amax().cpu()),
        "projected_grad_norm": float(torch.linalg.vector_norm(projected).cpu()),
        "lower_bound_entries": float(lower_bound.sum().cpu()),
        "upper_bound_entries": float(upper_bound.sum().cpu()),
    }


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def _format_iteration_hist(row: dict[str, Any], prefix: str) -> str:
    marker = f"{prefix}_count_"
    pairs: list[tuple[int, int]] = []
    for key, value in row.items():
        if key.startswith(marker):
            pairs.append((int(key[len(marker):]), int(float(value))))
    return ",".join(f"{iteration}:{count}" for iteration, count in sorted(pairs))


def log_row(row: dict[str, Any], summary: dict[str, dict[str, float]]) -> None:
    delta = row.get("delta_objective_bits")
    delta_text = "nan" if delta is None else f"{float(delta):.6g}"
    bfgs_text = ""
    if "accepted_step" in row:
        bfgs_text += f" accepted_step={bool(row['accepted_step'])}"
    if "reject_reason" in row:
        bfgs_text += f" reject_reason={row['reject_reason']}"
    if "accepted_rows" in row:
        bfgs_text += (
            f" accepted_rows={int(row['accepted_rows'])} "
            f"accepted_fraction={row['accepted_fraction']:.3f} "
            f"grad_evals={int(row.get('batched_bfgs_grad_evals', 0))} "
            f"loss_evals={int(row.get('batched_bfgs_loss_evals', 0))}"
        )
    adam_text = ""
    if "adam_lr" in row:
        adam_text += f" adam_lr={float(row['adam_lr']):.6g}"
    if row.get("adam_lr_reduced"):
        adam_text += (
            f" adam_lr_reduced=True"
            f" reason={row.get('adam_oscillation_reason', 'oscillation')}"
        )
    elif row.get("adam_oscillating"):
        adam_text += (
            f" adam_oscillating=True"
            f" reason={row.get('adam_oscillation_reason', 'oscillation')}"
        )
    solver_text = ""
    if "solver_E_iterations_max" in row:
        e_hist = _format_iteration_hist(row, "solver_E_iterations")
        pi_hist = _format_iteration_hist(row, "solver_Pi_iterations")
        if e_hist:
            solver_text += f" solver_E_iter_hist={e_hist}"
        else:
            solver_text += f" solver_E_iter_max={row['solver_E_iterations_max']:.0f}"
        if pi_hist:
            solver_text += f" solver_Pi_iter_hist={pi_hist}"
        else:
            solver_text += f" solver_Pi_iter_max={row['solver_Pi_iterations_max']:.0f}"
        wave_count = row.get("solver_Pi_wave_count", 0.0)
        if wave_count:
            solver_text += (
                f" solver_Pi_converged="
                f"{row['solver_Pi_converged_waves']:.0f}/{wave_count:.0f}"
            )
        neumann_hist = _format_iteration_hist(row, "solver_Neumann_terms")
        if neumann_hist:
            solver_text += f" solver_Neumann_terms_hist={neumann_hist}"
        elif "solver_Neumann_terms_max" in row:
            solver_text += f" solver_Neumann_terms_max={row['solver_Neumann_terms_max']:.0f}"
        e_adj_hist = _format_iteration_hist(row, "solver_E_adjoint_iterations")
        if e_adj_hist:
            solver_text += f" solver_E_adj_iter_hist={e_adj_hist}"
        elif "solver_E_adjoint_iterations_max" in row:
            solver_text += f" solver_E_adj_iter_max={row['solver_E_adjoint_iterations_max']:.0f}"
    print(
        f"phase={row['phase']} iter={row['iteration']:04d} "
        f"objective_bits={row['objective_bits']:.6f} "
        f"data_nll_bits={row['data_nll_bits']:.6f} "
        f"delta_objective_bits={delta_text} "
        f"grad_inf={row['grad_inf']:.6g} grad_norm={row['grad_norm']:.6g} "
        f"projected_grad_inf={row.get('projected_grad_inf', row['grad_inf']):.6g} "
        f"theta_step_inf={row['theta_step_inf']:.3g} "
        f"eval_s={row['eval_s']:.3f} step_s={row['step_s']:.3f} "
        f"closure_evals={row.get('closure_evals', 1)}"
        f"{bfgs_text}{adam_text}{solver_text}",
        flush=True,
    )
    print("  " + format_rate_summary(summary), flush=True)


def termination_status(
    row: dict[str, Any],
    *,
    stable_loss_steps: int,
    args: argparse.Namespace,
) -> tuple[str, str] | None:
    optimality = row.get("projected_grad_inf", row["grad_inf"])
    if optimality <= args.grad_inf_tol:
        return "converged", "projected_gradient_tolerance"
    if row["iteration"] < args.min_steps:
        return None
    if row["theta_step_inf"] <= args.theta_step_tol and row["iteration"] >= args.min_steps:
        return "stalled", "theta_step_tolerance_high_gradient"
    if (
        stable_loss_steps >= args.loss_patience
        and row["iteration"] >= args.min_steps
    ):
        return "stalled", "loss_change_patience_high_gradient"
    return None


def make_batched_bfgs(model: GeneReconModel, args: argparse.Namespace) -> BatchedLBFGS:
    return BatchedLBFGS(
        [model.theta],
        lr=args.lbfgs_lr,
        max_iter=1,
        history_size=args.lbfgs_history_size,
        lower_bound=math.log2(args.min_rate),
        upper_bound=math.log2(args.max_rate),
    )


def first_order_step(
    model: GeneReconModel,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], float, torch.Tensor | None, torch.Tensor]:
    theta_before = model.theta.detach().clone()
    _objective, metrics = evaluate(model, args)
    if model.theta.grad is None:
        raise RuntimeError("missing theta gradient")
    grad_snapshot = model.theta.grad.detach().clone()
    if not metrics_are_finite(metrics) or not tensor_is_finite(grad_snapshot):
        model.theta.grad = None
        model.clear()
        metrics["accepted_step"] = 0.0
        metrics["reject_reason"] = "nonfinite_first_order_gradient"
        return metrics, 0.0, grad_snapshot, torch.zeros_like(theta_before)

    optimizer.step()
    synchronize()
    clamp_rates(model, args)
    step_snapshot = model.theta.detach() - theta_before
    if not tensor_is_finite(step_snapshot) or not tensor_is_finite(model.theta):
        optimizer.state.pop(model.theta, None)
        restore_theta(model, theta_before)
        metrics["accepted_step"] = 0.0
        metrics["reject_reason"] = "nonfinite_first_order_step"
        return metrics, 0.0, grad_snapshot, torch.zeros_like(theta_before)

    theta_step = float(step_snapshot.abs().amax().cpu())
    model.clear()
    return metrics, theta_step, grad_snapshot, step_snapshot.detach().clone()


def lbfgs_step(
    model: GeneReconModel,
    optimizer: torch.optim.LBFGS,
    args: argparse.Namespace,
) -> tuple[dict[str, float], float, int]:
    theta_before = model.theta.detach().clone()
    closure_evals = 0
    start_metrics: dict[str, float] | None = None
    last_metrics: dict[str, float] | None = None
    saw_nonfinite_trial = False

    def reject_step(reason: str) -> tuple[dict[str, float], float, int]:
        optimizer.state.pop(model.theta, None)
        restore_theta(model, theta_before)
        if start_metrics is None:
            _objective, metrics = evaluate(model, args)
            model.theta.grad = None
            model.clear()
        else:
            metrics = dict(start_metrics)
        metrics["accepted_step"] = 0.0
        metrics["reject_reason"] = reason
        metrics["lbfgs_nonfinite_trial"] = float(saw_nonfinite_trial)
        return metrics, 0.0, closure_evals

    def closure() -> torch.Tensor:
        nonlocal closure_evals, last_metrics, start_metrics, saw_nonfinite_trial
        closure_evals += 1
        with torch.no_grad():
            if not tensor_is_finite(model.theta):
                saw_nonfinite_trial = True
                model.theta.copy_(theta_before)
            clamp_rates(model, args)
        optimizer.zero_grad(set_to_none=True)
        objective, metrics = evaluate(model, args, zero_grad=False)
        if start_metrics is None:
            start_metrics = dict(metrics)
        if not tensor_is_finite(objective) or not metrics_are_finite(metrics):
            saw_nonfinite_trial = True
        last_metrics = metrics
        return objective

    try:
        optimizer.step(closure)
    except RuntimeError:
        return reject_step("lbfgs_runtime_error")
    synchronize()
    if start_metrics is None or last_metrics is None:
        raise RuntimeError("LBFGS closure was not called")
    if saw_nonfinite_trial or not tensor_is_finite(model.theta):
        return reject_step("nonfinite_lbfgs_trial")

    clamp_rates(model, args)
    if not tensor_is_finite(model.theta):
        return reject_step("nonfinite_projected_theta")
    _objective, final_metrics = evaluate(model, args)
    model.theta.grad = None
    if (
        not metrics_are_finite(final_metrics)
        or final_metrics["objective_bits"] > start_metrics["objective_bits"]
    ):
        return reject_step("post_clamp_objective_increase_or_nonfinite")

    theta_step = float((model.theta.detach() - theta_before).abs().amax().cpu())
    final_metrics["accepted_step"] = 1.0
    final_metrics["lbfgs_nonfinite_trial"] = float(saw_nonfinite_trial)
    model.clear()
    return final_metrics, theta_step, closure_evals


def genewise_data_loss_vector(
    model: GeneReconModel,
    *,
    need_grad: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if model.theta.ndim != 2 or model.theta.shape[1] != 3:
        raise ValueError("batched BFGS expects genewise theta with shape [G, 3]")

    return model.full_genewise_nll_and_grad(need_grad=need_grad)


def genewise_objective_vector(
    model: GeneReconModel,
    args: argparse.Namespace,
    *,
    need_grad: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    data_loss, data_grad = genewise_data_loss_vector(model, need_grad=need_grad)
    reg_vec = regularization_vector(model.theta, args)
    objective = data_loss + reg_vec.detach()

    if need_grad:
        if data_grad is None:
            raise RuntimeError("internal error: missing genewise data gradient")
        model.theta.grad = data_grad + regularization_grad(model.theta, args)

    return objective.detach(), data_loss.detach(), reg_vec.detach()


def batched_bfgs_step(
    model: GeneReconModel,
    optimizer: BatchedLBFGS,
    args: argparse.Namespace,
) -> tuple[dict[str, float], float, int]:
    theta_before = model.theta.detach().clone()
    grad_evals = 0
    loss_evals = 0

    def closure() -> torch.Tensor:
        nonlocal grad_evals
        grad_evals += 1
        optimizer.zero_grad(set_to_none=True)
        objective_vec, _data_vec, _reg_vec = genewise_objective_vector(
            model,
            args,
            need_grad=True,
        )
        return objective_vec

    def loss_closure() -> torch.Tensor:
        nonlocal loss_evals
        loss_evals += 1
        objective_vec, _data_vec, _reg_vec = genewise_objective_vector(
            model,
            args,
            need_grad=False,
        )
        return objective_vec

    synchronize()
    start = time.perf_counter()
    optimizer.step(closure, loss_closure=loss_closure)
    synchronize()
    eval_elapsed = time.perf_counter() - start
    clamp_rates(model, args)

    objective_vec, data_vec, reg_vec = genewise_objective_vector(
        model,
        args,
        need_grad=False,
    )
    state = optimizer.state[model.theta]
    grad = state.get("last_grad")
    if grad is None:
        raise RuntimeError("BatchedLBFGS did not report a final gradient")
    grad = grad.reshape_as(model.theta).detach()

    theta_step = float((model.theta.detach() - theta_before).abs().amax().cpu())
    accepted = state.get("last_accepted")
    alpha = state.get("last_alpha")
    metrics = {
        "data_nll_bits": float(data_vec.sum().cpu()),
        "regularization_bits": float(reg_vec.sum().cpu()),
        "objective_bits": float(objective_vec.sum().cpu()),
        "log_likelihood_bits": float(-data_vec.sum().cpu()),
        "grad_inf": float(grad.abs().amax().cpu()),
        "grad_norm": float(torch.linalg.vector_norm(grad).cpu()),
        "eval_s": eval_elapsed,
        "batched_bfgs_grad_evals": float(grad_evals),
        "batched_bfgs_loss_evals": float(loss_evals),
    }
    if accepted is not None:
        accepted_cpu = accepted.detach().cpu()
        metrics["accepted_rows"] = float(accepted_cpu.sum())
        metrics["accepted_fraction"] = float(accepted_cpu.float().mean())
    if alpha is not None:
        alpha_cpu = alpha.detach().cpu()
        metrics["alpha_mean"] = float(alpha_cpu.mean())
        metrics["alpha_max"] = float(alpha_cpu.max())
    model.theta.grad = None
    model.clear()
    return metrics, theta_step, grad_evals + loss_evals


def build_model(args: argparse.Namespace) -> GeneReconModel:
    return GeneReconModel.from_alerax_families(
        str(args.species_tree),
        args.families_file,
        mode=args.mode,
        start=0,
        max_families=args.max_families,
        device=args.device,
        dtype=torch.float32,
        theta_init_rates=(args.init_d, args.init_l, args.init_t),
        fixed_iters_E=args.fixed_iters_e,
        fixed_iters_Pi=args.fixed_iters_pi,
        neumann_terms=args.neumann_terms,
        adaptive_iters=args.adaptive_iterations,
        convergence_check_interval=args.convergence_check_interval,
        e_logsumexp_tol=args.e_logsumexp_tol,
        pi_max_diff_tol=args.pi_max_diff_tol,
        gradient_change_tol=args.gradient_change_tol,
        gradient_change_rtol=args.gradient_change_rtol,
        use_pruning=True,
        family_chunk_size=args.family_chunk_size,
        clade_budget=args.clade_budget,
        batch_packing=args.batch_packing,
        max_wave_size=args.max_wave_size,
        lazy_preprocess=True,
        prefetch_batches="all",
    )


def prepare_all_batches(model: GeneReconModel) -> None:
    model.materialize_batches()
    model.select_batch(0)
    model.clear()


def parameter_row_names(
    model: GeneReconModel,
    species_names: list[str],
    args: argparse.Namespace,
) -> list[str]:
    row_count = int(model.theta.detach().reshape(-1, 3).shape[0])
    if args.mode == "specieswise":
        return species_names[:row_count]
    if args.mode == "genewise":
        names = [f"family_{i}" for i in range(row_count)]
        for meta in model.batch_metadata:
            for idx, name in zip(meta.family_indices, meta.family_names):
                if 0 <= idx < row_count:
                    names[idx] = name
        return names
    return ["global"]


def write_outputs(
    model: GeneReconModel,
    species_names: list[str],
    out_dir: Path,
    args: argparse.Namespace,
    final_metrics: dict[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.theta.detach().cpu(), out_dir / "theta_final.pt")
    with (out_dir / "final_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(final_metrics, handle, indent=2, sort_keys=True)
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    with (out_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)

    theta = model.theta.detach().reshape(-1, 3).cpu()
    rates = torch.exp2(theta)
    ps = pS_values(theta).cpu()
    names = parameter_row_names(model, species_names, args)
    with (out_dir / "rates_final.tsv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["row", "name", "D", "T", "L", "pS", "theta_D", "theta_T", "theta_L"])
        for i, name in enumerate(names):
            writer.writerow(
                [
                    i,
                    name,
                    float(rates[i, 0]),
                    float(rates[i, 2]),
                    float(rates[i, 1]),
                    float(ps[i]),
                    float(theta[i, 0]),
                    float(theta[i, 2]),
                    float(theta[i, 1]),
                ]
            )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fast exact optimization of the HOGENOM CCP likelihood."
    )
    parser.add_argument(
        "--optimizer",
        choices=(
            "adagrad",
            "adam",
            "lbfgs",
            "adagrad-lbfgs",
            "batched-bfgs",
            "adagrad-batched-bfgs",
        ),
        default="adagrad",
    )
    parser.add_argument("--steps", type=int, default=200, help="Maximum optimization steps after any warmup phase.")
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=20,
        help="Adagrad warmup steps for --optimizer adagrad-lbfgs/adagrad-batched-bfgs.",
    )
    parser.add_argument("--lr", type=float, default=1.0, help="Learning rate for Adagrad/Adam or LBFGS if no separate LBFGS lr is given.")
    parser.add_argument(
        "--adam-oscillation-detection",
        choices=("off", "gradient", "parameters", "both"),
        default="both",
        help="For Adam, decay the learning rate when successive gradients and/or parameter updates reverse direction.",
    )
    parser.add_argument(
        "--adam-oscillation-lr-decay",
        type=float,
        default=0.5,
        help="Multiplicative Adam learning-rate decay applied after oscillation is detected.",
    )
    parser.add_argument(
        "--adam-min-lr",
        type=float,
        default=1e-6,
        help="Lower bound for automatic Adam learning-rate decay.",
    )
    parser.add_argument(
        "--adam-oscillation-cos-threshold",
        type=float,
        default=-0.25,
        help="Decay Adam lr when the cosine between successive gradients/updates is at or below this value.",
    )
    parser.add_argument(
        "--adam-oscillation-flip-fraction",
        type=float,
        default=0.5,
        help="Decay Adam lr when at least this fraction of gradient/update entries flips sign.",
    )
    parser.add_argument(
        "--adam-oscillation-cooldown",
        type=int,
        default=5,
        help="Number of optimizer steps to wait before another automatic Adam lr decay.",
    )
    parser.add_argument("--lbfgs-lr", type=float, default=0.1)
    parser.add_argument("--lbfgs-history-size", type=int, default=10)
    parser.add_argument("--lbfgs-line-search", choices=("none", "strong_wolfe"), default="none")
    parser.add_argument("--grad-inf-tol", type=float, default=1e-3)
    parser.add_argument("--loss-change-tol", type=float, default=1e-5)
    parser.add_argument("--theta-step-tol", type=float, default=1e-6)
    parser.add_argument("--loss-patience", type=int, default=5)
    parser.add_argument("--min-steps", type=int, default=5)

    parser.add_argument("--mode", choices=("global", "specieswise", "genewise"), default="specieswise")
    parser.add_argument("--max-families", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--species-tree", type=Path, default=SPECIES_TREE)
    parser.add_argument("--families-file", type=Path, default=FAMILIES_FILE)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)

    parser.add_argument("--family-chunk-size", type=int, default=0)
    parser.add_argument("--clade-budget", type=int, default=600000)
    parser.add_argument("--batch-packing", choices=("sequential", "clade_first_fit", "depth_first_fit"), default="depth_first_fit")
    parser.add_argument("--max-wave-size", type=int, default=8192)
    parser.add_argument("--fixed-iters-e", type=int, default=6)
    parser.add_argument("--fixed-iters-pi", type=int, default=6)
    parser.add_argument("--neumann-terms", type=int, default=6)
    parser.add_argument(
        "--adaptive-iterations",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Check E/Pi/gradient convergence instead of always using the "
            "iteration counts exactly. The fixed iteration arguments become "
            "maximum iteration counts."
        ),
    )
    parser.add_argument("--convergence-check-interval", type=int, default=4)
    parser.add_argument("--e-logsumexp-tol", type=float, default=1e-5)
    parser.add_argument("--pi-max-diff-tol", type=float, default=1e-5)
    parser.add_argument("--gradient-change-tol", type=float, default=1e-4)
    parser.add_argument("--gradient-change-rtol", type=float, default=1e-4)

    parser.add_argument("--init-d", type=float, default=0.05)
    parser.add_argument("--init-l", type=float, default=0.05)
    parser.add_argument("--init-t", type=float, default=0.05)
    parser.add_argument("--min-rate", type=float, default=1e-10)
    parser.add_argument("--max-rate", type=float, default=100.0)

    parser.add_argument("--regularization", choices=("none", "square-theta", "gaussian-theta", "beta-ps"), default="none")
    parser.add_argument("--regularization-weight", type=float, default=1.0)
    parser.add_argument("--regularization-center", type=float, default=0.0)
    parser.add_argument("--regularization-std", type=float, default=0.5)
    parser.add_argument("--beta-ps-alpha", type=float, default=4.0)
    parser.add_argument("--beta-ps-beta", type=float, default=1.0)
    parser.add_argument("--prepare-all-batches", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(argv)


def run_phase(
    *,
    model: GeneReconModel,
    args: argparse.Namespace,
    phase: str,
    optimizer: torch.optim.Optimizer,
    start_iteration: int,
    max_steps: int,
    history_path: Path,
) -> tuple[int, dict[str, Any]]:
    previous_objective: float | None = None
    stable_loss_steps = 0
    final_metrics: dict[str, Any] = {}
    previous_adam_grad: torch.Tensor | None = None
    previous_adam_step: torch.Tensor | None = None
    adam_cooldown_remaining = 0

    for local_step in range(max_steps):
        step_start = time.perf_counter()
        optimizer_metrics: dict[str, Any] = {}
        if isinstance(optimizer, torch.optim.LBFGS):
            metrics, theta_step, closure_evals = lbfgs_step(model, optimizer, args)
        elif isinstance(optimizer, BatchedLBFGS):
            metrics, theta_step, closure_evals = batched_bfgs_step(model, optimizer, args)
        else:
            metrics, theta_step, grad_snapshot, step_snapshot = first_order_step(
                model,
                optimizer,
                args,
            )
            closure_evals = 1
            if isinstance(optimizer, torch.optim.Adam):
                optimizer_metrics, adam_cooldown_remaining = (
                    maybe_decay_adam_lr_for_oscillation(
                        optimizer,
                        args,
                        previous_grad=previous_adam_grad,
                        current_grad=grad_snapshot,
                        previous_step=previous_adam_step,
                        current_step=step_snapshot,
                        cooldown_remaining=adam_cooldown_remaining,
                    )
                )
                previous_adam_grad = grad_snapshot.detach().clone()
                previous_adam_step = step_snapshot.detach().clone()
        step_s = time.perf_counter() - step_start

        objective = metrics["objective_bits"]
        delta = None if previous_objective is None else previous_objective - objective
        if delta is not None and abs(delta) <= args.loss_change_tol:
            stable_loss_steps += 1
        else:
            stable_loss_steps = 0
        previous_objective = objective

        row: dict[str, Any] = {
            "phase": phase,
            "iteration": start_iteration + local_step,
            "phase_iteration": local_step,
            "theta_step_inf": theta_step,
            "step_s": step_s,
            "closure_evals": closure_evals,
            "delta_objective_bits": delta,
            **metrics,
            **optimizer_metrics,
        }
        summary = rate_summary(model.theta)
        row["rates"] = summary
        append_jsonl(history_path, row)
        log_row(row, summary)
        final_metrics = metrics

        status = termination_status(row, stable_loss_steps=stable_loss_steps, args=args)
        if status is not None:
            event, reason = status
            row = {
                "event": event,
                "reason": reason,
                "phase": phase,
                "iteration": start_iteration + local_step,
                "stable_loss_steps": stable_loss_steps,
                "grad_inf": metrics["grad_inf"],
                "grad_inf_tol": args.grad_inf_tol,
                "projected_grad_inf": metrics.get("projected_grad_inf", metrics["grad_inf"]),
                "theta_step_inf": theta_step,
                "theta_step_tol": args.theta_step_tol,
            }
            append_jsonl(history_path, row)
            print(json.dumps(row, sort_keys=True), flush=True)
            return start_iteration + local_step + 1, final_metrics

    return start_iteration + max_steps, final_metrics


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if args.fixed_iters_pi % 2 != 0:
        raise ValueError("--fixed-iters-pi must be even")
    if args.max_rate <= args.min_rate:
        raise ValueError("--max-rate must be greater than --min-rate")
    if args.adaptive_iterations and args.convergence_check_interval % 2 != 0:
        raise ValueError("--convergence-check-interval must be even in adaptive mode")
    if not (0.0 < args.adam_oscillation_lr_decay < 1.0):
        raise ValueError("--adam-oscillation-lr-decay must be in (0, 1)")
    if args.adam_min_lr <= 0.0:
        raise ValueError("--adam-min-lr must be positive")
    if not (-1.0 <= args.adam_oscillation_cos_threshold <= 1.0):
        raise ValueError("--adam-oscillation-cos-threshold must be in [-1, 1]")
    if not (0.0 <= args.adam_oscillation_flip_fraction <= 1.0):
        raise ValueError("--adam-oscillation-flip-fraction must be in [0, 1]")
    if args.adam_oscillation_cooldown < 0:
        raise ValueError("--adam-oscillation-cooldown must be non-negative")
    if args.optimizer in {"batched-bfgs", "adagrad-batched-bfgs"} and args.mode != "genewise":
        raise ValueError("batched BFGS optimizers require --mode genewise")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    history_path = args.out_dir / "history.jsonl"
    if history_path.exists():
        history_path.unlink()

    print("fast_hogenom_ccp_optimizer", flush=True)
    print(f"species_tree={args.species_tree}", flush=True)
    print(f"families_file={args.families_file}", flush=True)
    print(
        "layout="
        f"chunk_size={args.family_chunk_size} clade_budget={args.clade_budget} "
        f"packing={args.batch_packing} max_wave_size={args.max_wave_size}",
        flush=True,
    )
    print(
        "solver="
        f"fixed_iters_e={args.fixed_iters_e} fixed_iters_pi={args.fixed_iters_pi} "
        f"neumann_terms={args.neumann_terms} adaptive={args.adaptive_iterations} "
        f"check_interval={args.convergence_check_interval}",
        flush=True,
    )
    if args.adaptive_iterations:
        print(
            "solver_tolerances="
            f"e_logsumexp={args.e_logsumexp_tol} "
            f"pi_max_diff={args.pi_max_diff_tol} "
            f"gradient_change={args.gradient_change_tol} "
            f"gradient_change_rtol={args.gradient_change_rtol}",
            flush=True,
        )
    print(
        f"mode={args.mode} optimizer={args.optimizer} lr={args.lr} lbfgs_lr={args.lbfgs_lr} "
        f"lbfgs_line_search={args.lbfgs_line_search} max_rate={args.max_rate} "
        "origination=uniform",
        flush=True,
    )
    if args.optimizer == "adam":
        print(
            "adam_oscillation="
            f"detection={args.adam_oscillation_detection} "
            f"decay={args.adam_oscillation_lr_decay} min_lr={args.adam_min_lr} "
            f"cos_threshold={args.adam_oscillation_cos_threshold} "
            f"flip_fraction={args.adam_oscillation_flip_fraction} "
            f"cooldown={args.adam_oscillation_cooldown}",
            flush=True,
        )

    build_start = time.perf_counter()
    model = build_model(args)
    species_names = model.species_names
    if args.prepare_all_batches:
        prepare_all_batches(model)
    synchronize()
    print(
        f"build_s={time.perf_counter() - build_start:.3f} "
        f"families={sum(m.family_count for m in model.batch_metadata)} "
        f"species={model.n_species} batches={len(model.batch_metadata)} "
        f"waves={sum(m.wave_count for m in model.batch_metadata)}",
        flush=True,
    )

    try:
        print("warmup_exact_full_eval", flush=True)
        _, initial = evaluate(model, args)
        model.theta.grad = None
        model.clear()
        print(
            f"initial objective_bits={initial['objective_bits']:.6f} "
            f"data_nll_bits={initial['data_nll_bits']:.6f} "
            f"grad_inf={initial['grad_inf']:.6g} eval_s={initial['eval_s']:.3f}",
            flush=True,
        )

        iteration = 0
        final_metrics = initial
        if args.optimizer in {"adagrad-lbfgs", "adagrad-batched-bfgs"}:
            warm_optimizer = torch.optim.Adagrad(
                [model.theta],
                lr=args.lr,
                eps=1e-10,
            )
            iteration, final_metrics = run_phase(
                model=model,
                args=args,
                phase="adagrad",
                optimizer=warm_optimizer,
                start_iteration=iteration,
                max_steps=args.warmup_steps,
                history_path=history_path,
            )
            if args.optimizer == "adagrad-batched-bfgs":
                lbfgs_optimizer: torch.optim.Optimizer = make_batched_bfgs(model, args)
                phase = "batched-bfgs"
            else:
                lbfgs_optimizer = torch.optim.LBFGS(
                    [model.theta],
                    lr=args.lbfgs_lr,
                    max_iter=1,
                    max_eval=20,
                    history_size=args.lbfgs_history_size,
                    line_search_fn=None if args.lbfgs_line_search == "none" else args.lbfgs_line_search,
                )
                phase = "lbfgs"
            iteration, final_metrics = run_phase(
                model=model,
                args=args,
                phase=phase,
                optimizer=lbfgs_optimizer,
                start_iteration=iteration,
                max_steps=args.steps,
                history_path=history_path,
            )
        else:
            if args.optimizer == "adagrad":
                optimizer: torch.optim.Optimizer = torch.optim.Adagrad(
                    [model.theta],
                    lr=args.lr,
                    eps=1e-10,
                )
            elif args.optimizer == "adam":
                optimizer = torch.optim.Adam([model.theta], lr=args.lr)
            elif args.optimizer == "batched-bfgs":
                optimizer = make_batched_bfgs(model, args)
            else:
                optimizer = torch.optim.LBFGS(
                    [model.theta],
                    lr=args.lbfgs_lr,
                    max_iter=1,
                    max_eval=20,
                    history_size=args.lbfgs_history_size,
                    line_search_fn=None if args.lbfgs_line_search == "none" else args.lbfgs_line_search,
                )
            iteration, final_metrics = run_phase(
                model=model,
                args=args,
                phase=args.optimizer,
                optimizer=optimizer,
                start_iteration=iteration,
                max_steps=args.steps,
                history_path=history_path,
            )

        print("final_exact_full_eval", flush=True)
        _, final_metrics = evaluate(model, args)
        model.theta.grad = None
        model.clear()
        final_metrics["total_iterations"] = float(iteration)
        print(
            f"final objective_bits={final_metrics['objective_bits']:.6f} "
            f"data_nll_bits={final_metrics['data_nll_bits']:.6f} "
            f"grad_inf={final_metrics['grad_inf']:.6g} eval_s={final_metrics['eval_s']:.3f}",
            flush=True,
        )
        write_outputs(model, species_names, args.out_dir, args, final_metrics)
        print(f"wrote_outputs={args.out_dir}", flush=True)
    finally:
        model.close()


if __name__ == "__main__":
    main()
