from __future__ import annotations

from typing import Any

import torch

from gpurec.api._theta_constraints import finite_theta_rate_bounds_log2
from gpurec.api.autograd import _clear_pi_adjoint_runtime_cache
from gpurec.api.model import GeneReconModel

from ._evaluation import EvaluationOps
from ._runtime_helpers import (
    _commit_pi_adjoint_pending_caches,
    _discard_pi_adjoint_pending_caches,
    _is_finite_tensor,
)


class _NonfiniteParameterUpdate(RuntimeError):
    """Internal sentinel for optimizer updates that corrupt theta."""


def _set_model_theta(model: GeneReconModel, theta: torch.Tensor) -> None:
    with torch.no_grad():
        model.theta.copy_(theta)


def _restore_theta_if_nonfinite_update(
    model: GeneReconModel,
    theta_before: torch.Tensor,
) -> bool:
    if _is_finite_tensor(model.theta):
        return False
    _set_model_theta(model, theta_before)
    model.theta.grad = None
    return True


def _active_adam_step(
    model: GeneReconModel,
    optimizer: torch.optim.Optimizer,
    *,
    evaluation: EvaluationOps,
    config: Any,
    solver_stage: str,
    theta_before: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any], int]:
    pi_adjoint_pending_discards = _discard_pi_adjoint_pending_caches(model)
    optimizer.zero_grad(set_to_none=True)
    _ = evaluation.evaluate_active_genewise_vector_grad_at_current_theta(
        model,
        solver_stage=solver_stage,
    )
    optimizer.step()
    with torch.no_grad():
        model.clamp_theta_(config.min_rate, config.max_rate)
    if _restore_theta_if_nonfinite_update(model, theta_before):
        _discard_pi_adjoint_pending_caches(model)
        raise _NonfiniteParameterUpdate()
    loss_vec, _grad, metrics = (
        evaluation.evaluate_active_genewise_vector_grad_at_current_theta(
            model,
            solver_stage=solver_stage,
        )
    )
    lower_bound, upper_bound = finite_theta_rate_bounds_log2(
        config.min_rate,
        config.max_rate,
    )
    _, projected_grad_inf = evaluation.projected_grad_inf(
        model,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )
    metrics["grad/projected_inf"] = projected_grad_inf
    metrics["solver/pi_adjoint_pending_cache_commits"] = float(
        _commit_pi_adjoint_pending_caches(model)
    )
    metrics["solver/pi_adjoint_pending_cache_discards"] = float(
        pi_adjoint_pending_discards
    )
    return loss_vec, metrics, 2


def _clear_solver_runtime_state_preserving_pi_cache(model: GeneReconModel) -> None:
    statics = getattr(model, "cached_static_states", None)
    if statics is None:
        model.clear()
        return
    for static in list(statics):
        if hasattr(static, "warm_E"):
            static.warm_E = None
        _clear_pi_adjoint_runtime_cache(static)
        if hasattr(static, "last_solver_stats"):
            static.last_solver_stats = None
