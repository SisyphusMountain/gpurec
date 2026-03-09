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
    # When all entries are -inf, m=-inf. exp2(x - (-inf)) = exp2(inf) = inf.
    # Clamp m to avoid that; result will still be -inf via the + m at the end.
    s = torch.exp2(x - m.clamp(min=-1e30)).sum(dim=dim, keepdim=True)
    result = torch.log2(s) + m
    if not keepdim:
        result = result.squeeze(dim)
    return result


def logaddexp2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """log2(2^a + 2^b) — base-2 logaddexp."""
    m = torch.maximum(a, b)
    return torch.log2(torch.exp2(a - m.clamp(min=-1e30)) + torch.exp2(b - m.clamp(min=-1e30))) + m


def log2_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """log2(softmax(x, dim)) — log-softmax outputting log2 probabilities."""
    return torch.log_softmax(x, dim=dim) * _INV_LN2
