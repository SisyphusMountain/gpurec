"""Private optimizer-step metric helpers.

This module centralizes row-level optimizer telemetry for step execution. It is
not a public workflow API surface, and it intentionally avoids optimizer
stepping, solver configuration, theta mutation, and cache side effects.
"""

from __future__ import annotations

from typing import Any

import torch

from gpurec.api._theta_constraints import tensor_inf_norm

from .diagnostics import parameter_stats, solver_stats, tensor_stats


def _cached_scalar_loss_metrics(
    metrics: dict[str, Any],
    model: Any,
    loss: torch.Tensor,
) -> dict[str, Any]:
    updated = dict(metrics)
    updated["likelihood/data_nll_bits"] = float(loss.detach().cpu())
    updated["likelihood/log_likelihood_bits"] = float(-loss.detach().cpu())
    if model.theta.grad is not None:
        updated.update(tensor_stats("grad", model.theta.grad))
        updated.update(parameter_stats(model.theta))
        updated.update(solver_stats(model))
    return updated


def _cached_genewise_loss_metrics(
    metrics: dict[str, Any],
    model: Any,
    loss_vec: torch.Tensor,
) -> dict[str, Any]:
    updated = dict(metrics)
    loss_total = loss_vec.sum().detach().cpu()
    updated["likelihood/data_nll_bits"] = float(loss_total)
    updated["likelihood/log_likelihood_bits"] = float(-loss_total)
    updated.update(tensor_stats("grad", model.theta.grad))
    updated.update(parameter_stats(model.theta))
    updated.update(solver_stats(model))
    return updated


def _projected_grad_inf_from_optimizer_state(
    *,
    opt_state: dict[str, Any],
    evaluation: Any,
    model: Any,
    lower_bound: float,
    upper_bound: float,
) -> float:
    projected_grad = opt_state.get("last_projected_grad")
    if torch.is_tensor(projected_grad):
        return tensor_inf_norm(projected_grad)
    _, projected_inf = evaluation.projected_grad_inf(
        model,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )
    return projected_inf


def _projected_optimizer_step_metrics(
    *,
    metric_prefix: str,
    opt_state: dict[str, Any],
    projected_loss_evals: int,
    theta_step: float,
    projected_grad_inf: float,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "grad/projected_inf": projected_grad_inf,
        f"optimizer/{metric_prefix}_grad_evals": float(
            int(opt_state.get("last_grad_evals", 0))
        ),
        f"optimizer/{metric_prefix}_loss_evals": float(
            int(opt_state.get("last_loss_evals", projected_loss_evals))
        ),
        f"optimizer/{metric_prefix}_accepted": bool(
            opt_state.get("last_accepted", False)
        ),
        f"optimizer/{metric_prefix}_alpha": float(
            opt_state.get("last_alpha", 0.0)
        ),
        f"optimizer/{metric_prefix}_step_inf": float(
            opt_state.get("last_step_inf", theta_step)
        ),
        f"optimizer/{metric_prefix}_line_search_decrease": float(
            opt_state.get("last_line_search_decrease", 0.0)
        ),
        f"optimizer/{metric_prefix}_armijo_required_decrease": float(
            opt_state.get("last_armijo_required_decrease", 0.0)
        ),
        f"optimizer/{metric_prefix}_fallback_attempted": bool(
            opt_state.get("last_fallback_attempted", False)
        ),
        f"optimizer/{metric_prefix}_fallback_used": bool(
            opt_state.get("last_fallback_used", False)
        ),
        f"optimizer/{metric_prefix}_fallback_alpha": float(
            opt_state.get("last_fallback_alpha", 0.0)
        ),
        f"optimizer/{metric_prefix}_fallback_loss_evals": float(
            int(opt_state.get("last_fallback_loss_evals", 0))
        ),
        f"optimizer/{metric_prefix}_fallback_budget_exhausted": bool(
            opt_state.get("last_fallback_budget_exhausted", False)
        ),
        f"optimizer/{metric_prefix}_fallback_reason": str(
            opt_state.get("last_fallback_reason", "none")
        ),
        f"optimizer/{metric_prefix}_high_kkt_stall_count": float(
            int(opt_state.get("last_high_kkt_stall_count", 0))
        ),
        f"optimizer/{metric_prefix}_history_cleared_for_fallback": bool(
            opt_state.get("last_history_cleared_for_fallback", False)
        ),
    }
    direction_kind = opt_state.get("last_direction_kind")
    if direction_kind is not None:
        metrics[f"optimizer/{metric_prefix}_direction_kind"] = str(direction_kind)
    fallback_max_loss_evals = opt_state.get("last_fallback_max_loss_evals")
    if fallback_max_loss_evals is not None:
        metrics[f"optimizer/{metric_prefix}_fallback_max_loss_evals"] = float(
            int(fallback_max_loss_evals)
        )
    return metrics


def _batched_lbfgs_step_metrics(
    *,
    opt_state: dict[str, Any],
    batched_grad_evals: int,
    batched_loss_evals: int,
    reused_optimizer_gradient: bool,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "optimizer/batched_lbfgs_grad_evals": float(batched_grad_evals),
        "optimizer/batched_lbfgs_loss_evals": float(batched_loss_evals),
        "optimizer/batched_lbfgs_reused_gradient": bool(reused_optimizer_gradient),
        "optimizer/batched_lbfgs_inner_iters": float(
            int(opt_state.get("last_n_iter", 0))
        ),
    }
    accepted = opt_state.get("last_accepted")
    if torch.is_tensor(accepted):
        accepted_f = accepted.detach().to(dtype=torch.float32)
        metrics["optimizer/batched_lbfgs_accepted_rows"] = float(
            accepted_f.sum().cpu()
        )
        metrics["optimizer/batched_lbfgs_accepted_fraction"] = float(
            accepted_f.mean().cpu()
        )
    alpha = opt_state.get("last_alpha")
    if torch.is_tensor(alpha):
        alpha_cpu = alpha.detach().cpu()
        metrics["optimizer/batched_lbfgs_alpha_mean"] = float(alpha_cpu.mean())
        metrics["optimizer/batched_lbfgs_alpha_max"] = float(alpha_cpu.max())
    return metrics


def _hessian_sgd_budget_metrics(
    *,
    config: Any,
    solver: Any,
    model: Any,
    active_solver_stage: str,
    hessian_sgd_validation_step: bool,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "optimizer/hessian_sgd_validation_step": hessian_sgd_validation_step,
    }
    if hessian_sgd_validation_step:
        metrics["optimizer/hessian_sgd_solver_budget"] = "validation"
        metrics["optimizer/hessian_sgd_active_fixed_iters_pi"] = float(
            config.hessian_sgd_validation_fixed_iters_pi
        )
        metrics["optimizer/hessian_sgd_active_neumann_terms"] = float(
            config.hessian_sgd_validation_neumann_terms
        )
    elif active_solver_stage == "warmup":
        metrics["optimizer/hessian_sgd_solver_budget"] = "warmup"
        active_fixed_iters_pi = solver.hessian_sgd_warmup_iters(model)
        metrics["optimizer/hessian_sgd_active_fixed_iters_pi"] = float(
            active_fixed_iters_pi
        )
        metrics["optimizer/hessian_sgd_active_neumann_terms"] = float(
            active_fixed_iters_pi
        )
    else:
        metrics["optimizer/hessian_sgd_solver_budget"] = "normal"
        metrics["optimizer/hessian_sgd_active_fixed_iters_pi"] = float(
            config.hessian_sgd_normal_fixed_iters_pi
            if config.hessian_sgd_normal_fixed_iters_pi is not None
            else config.fixed_iters_pi
        )
        metrics["optimizer/hessian_sgd_active_neumann_terms"] = float(
            config.hessian_sgd_normal_neumann_terms
            if config.hessian_sgd_normal_neumann_terms is not None
            else config.neumann_terms
        )
    return metrics


def _adagrad_restart_step_metrics(
    *,
    active_phase: Any,
    step: int,
) -> dict[str, Any]:
    phase_step = step - active_phase.start_step
    return {
        "optimizer/adagrad_restart_phase": active_phase.name,
        "optimizer/adagrad_restart_phase_index": int(active_phase.index),
        "optimizer/adagrad_restart_phase_step": int(phase_step),
        "optimizer/adagrad_restart_phase_steps": int(active_phase.phase.steps),
        "optimizer/adagrad_restart_budget": int(active_phase.phase.budget),
        "optimizer/adagrad_restart_fixed_iters_E": int(
            active_phase.phase.fixed_iters_e
        ),
        "optimizer/adagrad_restart_fixed_iters_Pi": int(
            active_phase.phase.fixed_iters_pi
        ),
        "optimizer/adagrad_restart_neumann_terms": int(
            active_phase.phase.neumann_terms
        ),
        "optimizer/adagrad_restart_lr": float(active_phase.phase.lr),
        "optimizer/adagrad_restart_restarted": phase_step == 0,
    }
