from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from gpurec.api._theta_constraints import (
    finite_theta_rate_bounds_log2,
    projected_theta_gradient_and_free,
    tensor_inf_norm,
)
from gpurec.api.model import GeneReconModel

_FD_NEWTON_CURVATURE_EPS = 1e-12


@dataclass
class _FDNewtonHessianState:
    batch_index: int
    solver_stage: str
    family_indices: tuple[int, ...]
    hessian: torch.Tensor
    active_theta: torch.Tensor
    active_grad: torch.Tensor
    active_loss: torch.Tensor
    updates_since_refresh: int = 0


def _fd_newton_state_matches(
    runtime: Any,
    model: GeneReconModel,
    state: _FDNewtonHessianState | None,
    *,
    solver_stage: str,
) -> bool:
    if state is None:
        return False
    if state.batch_index != int(model.current_batch_index):
        return False
    if state.solver_stage != solver_stage:
        return False
    family_indices = tuple(int(index) for index in model.current_batch_metadata.family_indices)
    if state.family_indices != family_indices:
        return False
    idx = runtime.active_batch_indices(model)
    active_theta = model.theta.detach().index_select(0, idx)
    return torch.equal(active_theta, state.active_theta)


def _bfgs_update_fd_newton_hessian(
    *,
    state: _FDNewtonHessianState,
    active_theta: torch.Tensor,
    active_grad: torch.Tensor,
    active_loss: torch.Tensor,
    accepted: torch.Tensor,
    free_before: torch.Tensor,
    free_after: torch.Tensor,
) -> tuple[_FDNewtonHessianState, torch.Tensor]:
    old_hessian = state.hessian.detach()
    s = active_theta - state.active_theta
    y = active_grad - state.active_grad
    bs = torch.bmm(old_hessian, s.unsqueeze(-1)).squeeze(-1)
    sbs = (s * bs).sum(dim=1)
    ys = (y * s).sum(dim=1)
    finite = (
        torch.isfinite(s).all(dim=1)
        & torch.isfinite(y).all(dim=1)
        & torch.isfinite(bs).all(dim=1)
        & torch.isfinite(sbs)
        & torch.isfinite(ys)
    )
    active_set_same = (free_before == free_after).all(dim=1)
    moved = s.abs().amax(dim=1) > _FD_NEWTON_CURVATURE_EPS
    valid_update = (
        accepted
        & moved
        & finite
        & active_set_same
        & (ys > _FD_NEWTON_CURVATURE_EPS)
        & (sbs > _FD_NEWTON_CURVATURE_EPS)
    )
    safe_sbs = sbs.abs().clamp_min(_FD_NEWTON_CURVATURE_EPS)
    safe_ys = ys.abs().clamp_min(_FD_NEWTON_CURVATURE_EPS)
    bfgs_hessian = (
        old_hessian
        - torch.einsum("bi,bj->bij", bs, bs) / safe_sbs[:, None, None]
        + torch.einsum("bi,bj->bij", y, y) / safe_ys[:, None, None]
    )
    bfgs_hessian = 0.5 * (bfgs_hessian + bfgs_hessian.transpose(1, 2))
    hessian = torch.where(
        valid_update[:, None, None],
        bfgs_hessian,
        old_hessian,
    )
    new_state = _FDNewtonHessianState(
        batch_index=state.batch_index,
        solver_stage=state.solver_stage,
        family_indices=state.family_indices,
        hessian=hessian.detach().clone(),
        active_theta=active_theta.detach().clone(),
        active_grad=active_grad.detach().clone(),
        active_loss=active_loss.detach().clone(),
        updates_since_refresh=state.updates_since_refresh + 1,
    )
    return new_state, valid_update


def _refresh_fd_newton_hessian_state(
    runtime: Any,
    model: GeneReconModel,
    *,
    solver_stage: str,
    baseline_state: _FDNewtonHessianState | None = None,
) -> tuple[_FDNewtonHessianState, dict[str, Any], int]:
    config = runtime.config
    idx = runtime.active_batch_indices(model)
    theta0 = model.theta.detach().clone()
    lower_bound, upper_bound = finite_theta_rate_bounds_log2(
        config.min_rate,
        config.max_rate,
    )
    eps = float(config.fd_hessian_epsilon)

    if baseline_state is None:
        loss0, grad0, metrics0 = (
            runtime.evaluate_active_genewise_vector_grad_at_current_theta(
                model,
                solver_stage=solver_stage,
            )
        )
        grad_evals = 1
        active_grad0 = grad0.index_select(0, idx)
        active_loss0 = loss0.index_select(0, idx)
    else:
        active_grad0 = baseline_state.active_grad.detach().clone()
        active_loss0 = baseline_state.active_loss.detach().clone()
        metrics0 = {
            "grad/inf": (
                float(active_grad0.detach().abs().amax().cpu())
                if active_grad0.numel()
                else 0.0
            ),
        }
        grad_evals = 0

    active_theta0 = theta0.index_select(0, idx)
    rows, cols = active_grad0.shape
    if cols != 3:
        raise RuntimeError(
            "Hessian-conditioned genewise optimization expects three D/T/L "
            "parameters per family; "
            f"got {cols}"
        )

    projected_grad, _free = projected_theta_gradient_and_free(
        active_theta0,
        active_grad0,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )

    hessian = torch.zeros(
        (rows, cols, cols),
        device=model.theta.device,
        dtype=model.theta.dtype,
    )
    for col in range(cols):
        plus = theta0.clone()
        minus = theta0.clone()
        plus_active = torch.clamp(
            active_theta0[:, col] + eps,
            min=lower_bound,
            max=upper_bound,
        )
        minus_active = torch.clamp(
            active_theta0[:, col] - eps,
            min=lower_bound,
            max=upper_bound,
        )
        plus_rows = active_theta0.clone()
        minus_rows = active_theta0.clone()
        plus_rows[:, col] = plus_active
        minus_rows[:, col] = minus_active
        plus.index_copy_(0, idx, plus_rows)
        minus.index_copy_(0, idx, minus_rows)
        runtime.set_model_theta(model, plus)
        _plus_loss, plus_grad, _plus_metrics = (
            runtime.evaluate_active_genewise_vector_grad_at_current_theta(
                model,
                solver_stage=solver_stage,
            )
        )
        runtime.set_model_theta(model, minus)
        _minus_loss, minus_grad, _minus_metrics = (
            runtime.evaluate_active_genewise_vector_grad_at_current_theta(
                model,
                solver_stage=solver_stage,
            )
        )
        grad_evals += 2
        denom = (plus_active - minus_active).abs().clamp_min(1e-12)
        hessian[:, :, col] = (
            plus_grad.index_select(0, idx) - minus_grad.index_select(0, idx)
        ) / denom[:, None]

    runtime.set_model_theta(model, theta0)
    hessian = 0.5 * (hessian + hessian.transpose(1, 2))
    state = _FDNewtonHessianState(
        batch_index=int(model.current_batch_index),
        solver_stage=solver_stage,
        family_indices=tuple(
            int(index) for index in model.current_batch_metadata.family_indices
        ),
        hessian=hessian.detach().clone(),
        active_theta=active_theta0.detach().clone(),
        active_grad=active_grad0.detach().clone(),
        active_loss=active_loss0.detach().clone(),
        updates_since_refresh=0,
    )
    metrics0 = dict(metrics0)
    metrics0["grad/projected_inf"] = tensor_inf_norm(projected_grad)
    return state, metrics0, grad_evals
