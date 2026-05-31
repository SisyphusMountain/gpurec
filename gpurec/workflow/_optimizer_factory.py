from __future__ import annotations

import math

import torch

from gpurec.api.model import GeneReconModel

from ._phase import _is_adagrad_restart_phase
from .config import RunConfig


def _bounded_optimizer_bounds(config: RunConfig) -> tuple[float, float]:
    return (
        math.log2(config.min_rate),
        math.log2(config.max_rate),
    )


def _lbfgs_common_kwargs(config: RunConfig) -> dict[str, object]:
    return {
        "lr": float(config.lbfgs_lr),
        "max_iter": int(config.lbfgs_max_iter),
        "history_size": int(config.lbfgs_history_size),
        "max_ls": int(config.lbfgs_max_ls),
        "tolerance_grad": 0.0,
    }


def _make_optimizer(
    config: RunConfig,
    model: GeneReconModel,
    phase: str,
) -> torch.optim.Optimizer:
    params = [model.theta]
    if phase == "adam":
        return torch.optim.Adam(params, lr=config.lr)
    if phase == "adagrad" or _is_adagrad_restart_phase(phase):
        return torch.optim.Adagrad(params, lr=config.lr, eps=1e-10)
    if phase == "projected-sgd":
        return torch.optim.SGD(params, lr=config.lr)
    if phase == "batched-lbfgs":
        from gpurec.optimization import BatchedLBFGS

        lower_bound, upper_bound = _bounded_optimizer_bounds(config)
        return BatchedLBFGS(
            params,
            **_lbfgs_common_kwargs(config),
            line_search_fn=(
                "strong_wolfe"
                if config.lbfgs_line_search == "strong_wolfe"
                else "armijo"
            ),
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )
    if phase == "adam-fd-newton":
        return torch.optim.Adam(params, lr=config.lr)
    if phase == "hessian-sgd":
        return torch.optim.SGD(params, lr=config.lr)
    if phase == "projected-lbfgs":
        from gpurec.optimization import ProjectedLBFGS

        lower_bound, upper_bound = _bounded_optimizer_bounds(config)
        return ProjectedLBFGS(
            params,
            **_lbfgs_common_kwargs(config),
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )
    if phase == "lbfgsb":
        from gpurec.optimization import LBFGSB

        lower_bound, upper_bound = _bounded_optimizer_bounds(config)
        return LBFGSB(
            params,
            **_lbfgs_common_kwargs(config),
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            active_tol=1e-7,
            fallback_max_ls=config.lbfgs_max_ls,
            fallback_max_coordinates=config.lbfgsb_fallback_max_coordinates,
            fallback_max_loss_evals=config.lbfgsb_fallback_max_loss_evals,
            fallback_resolution_competition_factor=(
                config.lbfgsb_fallback_resolution_competition_factor
            ),
        )
    if phase == "lbfgs":
        return torch.optim.LBFGS(
            params,
            lr=config.lbfgs_lr,
            max_iter=config.lbfgs_max_iter,
            history_size=config.lbfgs_history_size,
            line_search_fn=(
                None if config.lbfgs_line_search == "none" else config.lbfgs_line_search
            ),
        )
    raise ValueError(f"unknown optimizer phase {phase!r}")


def _refresh_optimizer_runtime_options(
    optimizer: torch.optim.Optimizer,
    phase: str | None,
    config: RunConfig,
) -> None:
    if phase not in {"projected-lbfgs", "lbfgsb"}:
        return
    if not optimizer.param_groups:
        return
    group = optimizer.param_groups[0]
    lower_bound, upper_bound = _bounded_optimizer_bounds(config)
    group["lr"] = float(config.lbfgs_lr)
    group["max_iter"] = int(config.lbfgs_max_iter)
    group["history_size"] = int(config.lbfgs_history_size)
    group["max_ls"] = int(config.lbfgs_max_ls)
    group["lower_bound"] = lower_bound
    group["upper_bound"] = upper_bound
    if phase == "lbfgsb":
        group["fallback_max_ls"] = int(config.lbfgs_max_ls)
        group["fallback_max_coordinates"] = int(config.lbfgsb_fallback_max_coordinates)
        group["fallback_max_loss_evals"] = (
            None
            if config.lbfgsb_fallback_max_loss_evals is None
            else int(config.lbfgsb_fallback_max_loss_evals)
        )
        group["fallback_resolution_competition_factor"] = float(
            config.lbfgsb_fallback_resolution_competition_factor
        )
