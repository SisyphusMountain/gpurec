import dataclasses
import math

import torch

from gpurec.batched_lbfgs import BatchedLBFGS as BatchedLBFGS
from gpurec.config.rates import RateBounds

_UNSET = object()  # distinguishes "kwarg not passed" from max_rate's legitimate None ("no cap")


def resolve_rate_bounds(
    bounds: RateBounds | None = None,
    *,
    min_rate=_UNSET,
    max_rate=_UNSET,
) -> RateBounds:
    """Resolve a ``RateBounds`` from an optional base ``bounds`` plus explicit ``min_rate``/
    ``max_rate`` kwarg overrides (the deprecation shim for the old individual kwargs): an override
    left at the ``_UNSET`` sentinel falls back to ``bounds`` (or ``RateBounds()`` if ``bounds`` is
    also ``None``); an explicitly-passed override (including ``max_rate=None``, meaning "no cap")
    replaces that field. Used by ``log2_rate_bounds``/``project_rate_gradient_``/
    ``clamp_log_rate_`` so existing callers passing e.g. ``min_rate=1e-6, max_rate=2.0`` explicitly
    keep working unchanged."""
    base = bounds if bounds is not None else RateBounds()
    overrides = {}
    if min_rate is not _UNSET:
        overrides["min_rate"] = min_rate
    if max_rate is not _UNSET:
        overrides["max_rate"] = max_rate
    return dataclasses.replace(base, **overrides) if overrides else base


def log2_rate_bounds(
    min_rate: float = _UNSET,
    max_rate: float | None = _UNSET,
    *,
    bounds: RateBounds | None = None,
) -> tuple[float, float | None]:
    resolved = resolve_rate_bounds(bounds, min_rate=min_rate, max_rate=max_rate)
    min_rate, max_rate = resolved.min_rate, resolved.max_rate
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
    min_rate: float = _UNSET,
    max_rate: float | None = _UNSET,
    bounds: RateBounds | None = None,
) -> torch.Tensor:
    """Project a log2-rate gradient against active min/max rate constraints.

    ``theta`` is expected to store log2 rates. If ``grad`` is omitted,
    ``theta.grad`` is modified in place.
    """
    if grad is None:
        grad = theta.grad
    if grad is None:
        raise RuntimeError("missing gradient to project")

    lower_bound, upper_bound = log2_rate_bounds(min_rate, max_rate, bounds=bounds)
    theta_detached = theta.detach()
    grad[(theta_detached <= lower_bound) & (grad > 0)] = 0
    if upper_bound is not None:
        grad[(theta_detached >= upper_bound) & (grad < 0)] = 0
    return grad


@torch.no_grad()
def clamp_log_rate_(
    theta: torch.Tensor,
    *,
    min_rate: float = _UNSET,
    max_rate: float | None = _UNSET,
    bounds: RateBounds | None = None,
) -> torch.Tensor:
    """Project log2-rate parameters into natural-rate bounds in place."""
    lower_bound, upper_bound = log2_rate_bounds(min_rate, max_rate, bounds=bounds)
    theta.clamp_(min=lower_bound, max=upper_bound)
    return theta
