"""Internal theta bound helpers shared by API and workflow code.

This module owns rate-bound validation, base-2 log-rate conversion, in-place
theta clamping, and projected-gradient maps for bounded theta optimizers. It
is support code for ``gpurec.api`` and ``gpurec.workflow``, not a standalone
public import surface.
"""

from __future__ import annotations

import math

import torch

from ._validation import positive_float


def theta_rate_bounds_log2(
    min_rate: float,
    max_rate: float | None = None,
) -> tuple[float, float | None]:
    min_rate = positive_float("min_rate", min_rate)
    if max_rate is not None:
        max_rate = positive_float("max_rate", max_rate)
    if max_rate is not None and max_rate < min_rate:
        raise ValueError("max_rate must be greater than or equal to min_rate")
    return math.log2(min_rate), None if max_rate is None else math.log2(max_rate)


def finite_theta_rate_bounds_log2(
    min_rate: float,
    max_rate: float,
) -> tuple[float, float]:
    lower_bound, upper_bound = theta_rate_bounds_log2(min_rate, max_rate)
    if upper_bound is None:
        raise ValueError("max_rate must be provided")
    return lower_bound, upper_bound


def clamp_theta_rates_(
    theta: torch.Tensor,
    *,
    min_rate: float = 1e-10,
    max_rate: float | None = None,
) -> None:
    lower_bound, upper_bound = theta_rate_bounds_log2(min_rate, max_rate)
    with torch.no_grad():
        theta.clamp_(min=lower_bound, max=upper_bound)


def projected_theta_gradient(
    theta: torch.Tensor,
    grad: torch.Tensor,
    *,
    lower_bound: float,
    upper_bound: float,
) -> torch.Tensor:
    theta = theta.detach()
    grad = grad.detach()
    return theta - torch.clamp(theta - grad, min=lower_bound, max=upper_bound)


def tensor_inf_norm(value: torch.Tensor) -> float:
    return float(value.detach().abs().amax().cpu()) if value.numel() else 0.0


def projected_theta_gradient_inf(
    theta: torch.Tensor,
    grad: torch.Tensor,
    *,
    lower_bound: float,
    upper_bound: float,
) -> tuple[torch.Tensor, float]:
    projected = projected_theta_gradient(
        theta,
        grad,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )
    return projected, tensor_inf_norm(projected)


def projected_theta_gradient_and_free(
    theta: torch.Tensor,
    grad: torch.Tensor,
    *,
    lower_bound: float,
    upper_bound: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    projected = projected_theta_gradient(
        theta,
        grad,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )
    return projected, projected.abs() > 0
