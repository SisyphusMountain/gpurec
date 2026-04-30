"""Log2-space utility functions.

All log-space tensors in gpurec use base-2 logarithms (log2).
This module provides base-2 equivalents of torch.logsumexp,
torch.logaddexp, and torch.log_softmax.
"""

import math
import os
import torch

_INV_LN2 = 1.0 / math.log(2.0)  # = log2(e)

_NEG_INF = float('-inf')


def _debug_nan_checks_enabled() -> bool:
    """Return True when expensive NaN scans are enabled for debugging.

    Set GPUREC_DEBUG_NAN_CHECKS to one of: 1, true, yes, on.
    Defaults to False for performance.
    """
    v = os.getenv("GPUREC_DEBUG_NAN_CHECKS", "").strip().lower()
    return v in {"1", "true", "yes", "on"}


_neg_inf_sentinel_cache: dict = {}


def _safe_log2_internal(x: torch.Tensor) -> torch.Tensor:
    """log2(x) with zero gradient at x<=0 (returns -inf there).

    Avoids the standard log2(0)=-inf whose gradient 1/(0*ln2)=inf causes
    0*inf=NaN when upstream gradient is zero.

    In debug mode (GPUREC_DEBUG_NAN_CHECKS=1), raises ValueError if x contains
    any NaN. Disabled by default to avoid full-tensor scans on hot paths.
    """
    if _debug_nan_checks_enabled() and torch.isnan(x).any():
        raise ValueError(f"_safe_log2_internal received NaN input (shape={tuple(x.shape)}, dtype={x.dtype})")
    key = (x.device.type, getattr(x.device, 'index', None), x.dtype)
    sentinel = _neg_inf_sentinel_cache.get(key)
    if sentinel is None:
        sentinel = torch.tensor(_NEG_INF, device=x.device, dtype=x.dtype)
        _neg_inf_sentinel_cache[key] = sentinel
    pos = x > 0
    safe_x = torch.where(pos, x, torch.ones_like(x))
    return torch.where(pos, torch.log2(safe_x), sentinel.expand_as(x))


def logsumexp2(x: torch.Tensor, dim: int, keepdim: bool = False) -> torch.Tensor:
    """log2(sum(2^x, dim)) with max-subtract stabilization."""
    m = x.max(dim=dim, keepdim=True).values
    # When all entries are -inf, m=-inf. exp2(x - (-inf)) = NaN.
    # Replace -inf with 0 so exp2(x - 0) = exp2(-inf) = 0, giving result -inf via + m.
    m_safe = torch.where(m == _NEG_INF, torch.zeros_like(m), m)
    s = torch.exp2(x - m_safe).sum(dim=dim, keepdim=True)
    result = _safe_log2_internal(s) + m
    if not keepdim:
        result = result.squeeze(dim)
    return result


def logaddexp2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """log2(2^a + 2^b) — base-2 logaddexp."""
    m = torch.maximum(a, b)
    m_safe = torch.where(m == _NEG_INF, torch.zeros_like(m), m)
    return _safe_log2_internal(torch.exp2(a - m_safe) + torch.exp2(b - m_safe)) + m


class _Log2Softmax(torch.autograd.Function):
    """Custom autograd for base-2 log-softmax with stable backward.

    Forward: y_i = x_i - log2(sum(2^x_j))
    Backward: grad_x_i = grad_y_i - sum_j(grad_y_j * 2^y_j)

    The standard autograd path computes 2^(x_i - max) as an intermediate,
    which underflows to 0 for very negative x_i (e.g. -33 in fp32).
    The subsequent log2(0) = -inf produces NaN gradients.

    This custom backward uses 2^y_j (the softmax probability) directly.
    When y_j is very negative, 2^y_j ≈ 0 and contributes nothing to the sum,
    which is correct — tiny-probability components don't affect the gradient.
    """

    @staticmethod
    def forward(ctx, x, dim):
        y = x - logsumexp2(x, dim=dim, keepdim=True)
        ctx.save_for_backward(y)
        ctx.dim = dim
        return y

    @staticmethod
    def backward(ctx, grad_output):
        y, = ctx.saved_tensors
        # softmax probabilities from the log2-space output
        p = torch.exp2(y)  # safe: y <= 0, so 0 <= p <= 1
        # Jacobian: dy_i/dx_j = δ_ij - p_j
        # VJP: grad_x_j = grad_y_j - p_j * sum_i(grad_y_i)
        grad_x = grad_output - p * grad_output.sum(dim=ctx.dim, keepdim=True)
        return grad_x, None  # None for dim


def log2_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """log2(softmax(x, dim)) in base-2: x_i - log2(sum(2^x_j, dim)).

    Input and output are both in log2-space. Uses exp2/log2 internally
    (not exp/log) so that theta in log2-space produces correct probabilities.

    Custom backward avoids NaN for very negative inputs (e.g. theta = -33 in fp32).
    """
    return _Log2Softmax.apply(x, dim)
