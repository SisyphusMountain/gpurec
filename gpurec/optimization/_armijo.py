"""Shared scalar Armijo line-search arithmetic."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def _armijo_threshold(loss: Tensor, trial_gtd: Tensor, c1: float) -> float:
    loss_value = float(loss.detach().cpu())
    gtd_value = float(trial_gtd.detach().cpu())
    armijo_value = loss_value + c1 * gtd_value
    strict_value = math.nextafter(loss_value, -math.inf)
    return min(armijo_value, strict_value)


def armijo_accepts(
    trial_loss: Tensor,
    loss: Tensor,
    trial_gtd: Tensor,
    c1: float,
) -> bool:
    """Return whether a scalar trial loss satisfies strict Armijo decrease."""
    if (
        not torch.isfinite(trial_loss)
        or not torch.isfinite(loss)
        or not torch.isfinite(trial_gtd)
    ):
        return False
    trial_value = float(trial_loss.detach().cpu())
    return trial_value <= _armijo_threshold(loss, trial_gtd, c1)


def armijo_required_decrease(loss: Tensor, trial_gtd: Tensor, c1: float) -> float:
    """Return the minimum decrease implied by strict Armijo arithmetic."""
    loss_value = float(loss.detach().cpu())
    return max(0.0, loss_value - _armijo_threshold(loss, trial_gtd, c1))


class ScalarArmijoMixin:
    """Private scalar-optimizer Armijo methods."""

    def _armijo_accepts(
        self,
        *,
        trial_loss: Tensor,
        loss: Tensor,
        trial_gtd: Tensor,
        c1: float,
    ) -> bool:
        return armijo_accepts(trial_loss, loss, trial_gtd, c1)

    def _armijo_required_decrease(
        self,
        *,
        loss: Tensor,
        trial_gtd: Tensor,
        c1: float,
    ) -> float:
        return armijo_required_decrease(loss, trial_gtd, c1)
