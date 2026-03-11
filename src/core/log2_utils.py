"""Log2-space utility functions.

All log-space tensors in gpurec use base-2 logarithms (log2).
This module provides base-2 equivalents of torch.logsumexp,
torch.logaddexp, and torch.log_softmax.
"""

import math
import torch

_INV_LN2 = 1.0 / math.log(2.0)  # = log2(e)


def logsumexp2(x: torch.Tensor, dim: int, keepdim: bool = False) -> torch.Tensor:
    """log2(sum(2^x, dim)) with max-subtract stabilization."""
    m = x.max(dim=dim, keepdim=True).values
    # When all entries are -inf, m=-inf. exp2(x - (-inf)) = NaN.
    # Replace -inf with 0 so exp2(x - 0) = exp2(-inf) = 0, giving result -inf via + m.
    m_safe = torch.where(m == float('-inf'), torch.zeros_like(m), m)
    s = torch.exp2(x - m_safe).sum(dim=dim, keepdim=True)
    result = torch.log2(s) + m
    if not keepdim:
        result = result.squeeze(dim)
    return result


def logaddexp2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """log2(2^a + 2^b) — base-2 logaddexp."""
    m = torch.maximum(a, b)
    m_safe = torch.where(m == float('-inf'), torch.zeros_like(m), m)
    return torch.log2(torch.exp2(a - m_safe) + torch.exp2(b - m_safe)) + m


def log2_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """log2(softmax(x, dim)) — log-softmax outputting log2 probabilities."""
    return torch.log_softmax(x, dim=dim) * _INV_LN2
