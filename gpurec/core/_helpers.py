"""Shared helpers used by forward.py, backward.py, legacy.py, and likelihood.py.

Centralises utilities that were previously copy-pasted across modules.
"""

from __future__ import annotations

import inspect
import os
from contextlib import contextmanager

import torch

from .log2_utils import logsumexp2


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
# Segmented logsumexp (CPU fallback + CUDA dispatch)
# ---------------------------------------------------------------------------

def _seg_logsumexp_host(x: torch.Tensor, ptr: torch.Tensor) -> torch.Tensor:
    """CPU fallback for segmented logsumexp; uses Triton kernel when CUDA is available."""
    from .kernels.scatter_lse import seg_logsumexp

    if x.is_cuda and ptr.is_cuda:
        return seg_logsumexp(x, ptr)

    num_segs = int(ptr.numel()) - 1
    out = []
    for i in range(num_segs):
        s = int(ptr[i].item())
        e = int(ptr[i + 1].item())
        if e > s:
            out.append(logsumexp2(x[s:e], dim=0))
        else:
            out.append(torch.full_like(x[0], NEG_INF))
    return (
        torch.stack(out, dim=0)
        if out
        else torch.empty((0, *x.shape[1:]), device=x.device, dtype=x.dtype)
    )


# ---------------------------------------------------------------------------
# NVTX profiling ranges
# ---------------------------------------------------------------------------

@contextmanager
def _nvtx_range(name: str):
    nvtx = getattr(getattr(torch, "cuda", None), "nvtx", None)
    if nvtx is not None and hasattr(nvtx, "range"):
        try:
            with nvtx.range(name):
                yield
            return
        except Exception:
            pass
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


@contextmanager
def _nvtx_here(name: str):
    """NVTX range whose label includes caller file:line for easy mapping."""
    try:
        frame = inspect.currentframe().f_back
        info = inspect.getframeinfo(frame, context=0)
        base = os.path.basename(info.filename)
        label = f"{name} [{base}:{info.lineno}]"
    except Exception:
        label = name
    record_function = None
    try:
        record_function = getattr(
            getattr(torch, "autograd", None).profiler, "record_function", None
        )
    except Exception:
        record_function = None
    if record_function is not None:
        with record_function(label):
            with _nvtx_range(label):
                yield
    else:
        with _nvtx_range(label):
            yield
