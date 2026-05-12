"""Shared helpers used by forward.py, backward.py, and likelihood.py.

Centralises utilities that were previously copy-pasted across modules.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch


NEG_INF = float("-inf")


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------

def _safe_exp2_ratio(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``2^(a-b)`` safely: returns 0 when *a* = -inf."""
    neg_inf_a = a == NEG_INF
    a_safe = torch.where(neg_inf_a, torch.zeros_like(a), a)
    b_safe = torch.where(neg_inf_a, torch.zeros_like(b), b)
    return torch.where(neg_inf_a, torch.zeros_like(a), torch.exp2(a_safe - b_safe))


# ---------------------------------------------------------------------------
# NVTX profiling ranges
# ---------------------------------------------------------------------------

@contextmanager
def _nvtx_range(name: str):
    nvtx = getattr(getattr(torch, "cuda", None), "nvtx", None)
    if nvtx is not None and hasattr(nvtx, "range"):
        try:
            range_ctx = nvtx.range(name)
        except Exception:
            range_ctx = None
        if range_ctx is not None:
            with range_ctx:
                yield
            return
    pushed = False
    if nvtx is not None and hasattr(nvtx, "range_push"):
        try:
            nvtx.range_push(name)
            pushed = True
        except Exception:
            pushed = False
    try:
        yield
    finally:
        if pushed and hasattr(nvtx, "range_pop"):
            try:
                nvtx.range_pop()
            except Exception:
                pass
