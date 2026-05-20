"""Shared helpers used by forward.py, backward.py, and likelihood.py.

Centralises utilities that were previously copy-pasted across modules.
"""

from __future__ import annotations

from contextlib import contextmanager
import os

import torch


NEG_INF = float("-inf")
_FALSE_ENV_TOKENS = frozenset(("", "0", "false", "no", "off"))
_REQUIRED_ENV_TOKENS = frozenset(("1", "true", "yes", "on", "force", "required"))


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
# Environment option helpers
# ---------------------------------------------------------------------------

def _env_flag_enabled(name: str, default: str = "1") -> bool:
    """Return whether an environment flag is enabled using gpurec truth rules."""
    return os.environ.get(name, default).strip().lower() not in _FALSE_ENV_TOKENS


def _env_mode_enabled_required(
    name: str,
    default: str = "auto",
) -> tuple[str, bool, bool]:
    """Return normalized mode plus enabled/required flags for CUDA path toggles."""
    mode = os.environ.get(name, default).strip().lower()
    return (
        mode,
        mode not in _FALSE_ENV_TOKENS,
        mode in _REQUIRED_ENV_TOKENS,
    )


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
            enter_range = getattr(range_ctx, "__enter__", None)
            exit_range = getattr(range_ctx, "__exit__", None)
            if enter_range is not None and exit_range is not None:
                try:
                    enter_range()
                except Exception:
                    pass
                else:
                    try:
                        yield
                    except BaseException as exc:
                        try:
                            exit_range(type(exc), exc, exc.__traceback__)
                        except Exception:
                            pass
                        raise
                    else:
                        try:
                            exit_range(None, None, None)
                        except Exception:
                            pass
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
