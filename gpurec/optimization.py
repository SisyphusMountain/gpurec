import math

import torch


def log2_rate_bounds(
    min_rate: float = 1e-10,
    max_rate: float | None = None,
) -> tuple[float, float | None]:
    if isinstance(min_rate, bool):
        raise TypeError("min_rate must be a positive finite float, not bool")
    min_rate = float(min_rate)
    if not math.isfinite(min_rate) or min_rate <= 0.0:
        raise ValueError("min_rate must be a positive finite float")
    if max_rate is None:
        return math.log2(min_rate), None
    if isinstance(max_rate, bool):
        raise TypeError("max_rate must be a positive finite float, not bool")
    max_rate = float(max_rate)
    if not math.isfinite(max_rate) or max_rate <= 0.0:
        raise ValueError("max_rate must be a positive finite float")
    if max_rate < min_rate:
        raise ValueError("max_rate must be greater than or equal to min_rate")
    return math.log2(min_rate), math.log2(max_rate)


@torch.no_grad()
def project_rate_gradient_(
    theta: torch.Tensor,
    grad: torch.Tensor | None = None,
    *,
    min_rate: float = 1e-10,
    max_rate: float | None = None,
) -> torch.Tensor:
    """Project a log2-rate gradient against active min/max rate constraints.

    ``theta`` is expected to store log2 rates. If ``grad`` is omitted,
    ``theta.grad`` is modified in place.
    """
    if grad is None:
        grad = theta.grad
    if grad is None:
        raise RuntimeError("missing gradient to project")

    lower_bound, upper_bound = log2_rate_bounds(min_rate, max_rate)
    theta_detached = theta.detach()
    grad[(theta_detached <= lower_bound) & (grad > 0)] = 0
    if upper_bound is not None:
        grad[(theta_detached >= upper_bound) & (grad < 0)] = 0
    return grad


@torch.no_grad()
def clamp_log_rate_(
    theta: torch.Tensor,
    *,
    min_rate: float = 1e-10,
    max_rate: float | None = None,
) -> torch.Tensor:
    """Project log2-rate parameters into natural-rate bounds in place."""
    lower_bound, upper_bound = log2_rate_bounds(min_rate, max_rate)
    theta.clamp_(min=lower_bound, max=upper_bound)
    return theta
