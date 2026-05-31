"""Internal row-wise history helpers for batched L-BFGS.

This module is private optimization support for ``BatchedLBFGS`` and not a
public import surface.  Line-search behavior remains in ``batched_lbfgs``.
"""

from typing import Any

import torch
from torch import Tensor


def _row_dot(a: Tensor, b: Tensor) -> Tensor:
    return (a * b).sum(dim=1)


class BatchedLBFGSHistoryMixin:
    """Private two-loop direction and curvature-history methods."""

    def _direction(
        self,
        flat_grad: Tensor,
        old_dirs: list[Tensor],
        old_stps: list[Tensor],
        ro: list[Tensor],
        H_diag: Tensor,
    ) -> Tensor:
        if not old_dirs:
            return -flat_grad

        q = flat_grad.clone()
        alphas: list[Tensor] = []
        for y_k, s_k, ro_k in zip(reversed(old_dirs), reversed(old_stps), reversed(ro)):
            alpha = _row_dot(s_k, q) * ro_k
            q = q - alpha[:, None] * y_k
            alphas.append(alpha)

        r = H_diag[:, None] * q
        for y_k, s_k, ro_k, alpha in zip(old_dirs, old_stps, ro, reversed(alphas)):
            beta = _row_dot(y_k, r) * ro_k
            r = r + (alpha - beta)[:, None] * s_k
        return -r

    def _append_history(
        self,
        state: dict[str, Any],
        s_k: Tensor,
        y_k: Tensor,
        active: Tensor,
        history_size: int,
        tolerance_change: float,
    ) -> None:
        old_dirs: list[Tensor] = state["old_dirs"]
        old_stps: list[Tensor] = state["old_stps"]
        ro: list[Tensor] = state["ro"]
        H_diag: Tensor = state["H_diag"]

        ys = _row_dot(y_k, s_k)
        yy = _row_dot(y_k, y_k)
        step_norm = s_k.abs().amax(dim=1)
        valid = (
            active
            & torch.isfinite(ys)
            & torch.isfinite(yy)
            & (ys > 1e-10)
            & (yy > 1e-30)
            & (step_norm > tolerance_change)
        )
        if not bool(valid.any()):
            return

        if len(old_dirs) == history_size:
            old_dirs.pop(0)
            old_stps.pop(0)
            ro.pop(0)

        old_dirs.append(torch.where(valid[:, None], y_k, torch.zeros_like(y_k)))
        old_stps.append(torch.where(valid[:, None], s_k, torch.zeros_like(s_k)))
        ro.append(torch.where(valid, 1.0 / ys.clamp_min(1e-30), torch.zeros_like(ys)))
        state["H_diag"] = torch.where(valid, ys / yy.clamp_min(1e-30), H_diag)
