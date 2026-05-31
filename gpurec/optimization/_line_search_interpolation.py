"""Internal line-search interpolation helpers.

This module is private optimization support, not a public import surface. It
owns pure tensor interpolation/clamping used by BatchedLBFGS line search only;
optimizer state, closure evaluation, and history updates stay in
``batched_lbfgs``.
"""
import torch
from torch import Tensor


def _clamp_tensor(value: Tensor, lower: Tensor, upper: Tensor) -> Tensor:
    return torch.minimum(torch.maximum(value, lower), upper)


def _cubic_interpolate(
    x1: Tensor,
    f1: Tensor,
    g1: Tensor,
    x2: Tensor,
    f2: Tensor,
    g2: Tensor,
    *,
    bounds: tuple[Tensor, Tensor] | None = None,
) -> Tensor:
    """Vectorized port of ``torch.optim.lbfgs._cubic_interpolate``."""
    if bounds is None:
        xmin_bound = torch.minimum(x1, x2)
        xmax_bound = torch.maximum(x1, x2)
    else:
        xmin_bound, xmax_bound = bounds

    x_diff = x1 - x2
    d1 = g1 + g2 - 3 * (f1 - f2) / x_diff
    d2_square = d1.square() - g1 * g2
    d2 = d2_square.clamp_min(0).sqrt()

    denom_forward = g2 - g1 + 2 * d2
    denom_reverse = g1 - g2 + 2 * d2
    min_forward = x2 - (x2 - x1) * ((g2 + d2 - d1) / denom_forward)
    min_reverse = x1 - (x1 - x2) * ((g1 + d2 - d1) / denom_reverse)
    min_pos = torch.where(x1 <= x2, min_forward, min_reverse)

    midpoint = (xmin_bound + xmax_bound) / 2.0
    denom = torch.where(x1 <= x2, denom_forward, denom_reverse)
    valid = (
        (d2_square >= 0)
        & torch.isfinite(min_pos)
        & torch.isfinite(denom)
        & torch.isfinite(x_diff)
        & (denom.abs() > torch.finfo(x1.dtype).eps)
        & (x_diff.abs() > torch.finfo(x1.dtype).eps)
    )
    return _clamp_tensor(torch.where(valid, min_pos, midpoint), xmin_bound, xmax_bound)
