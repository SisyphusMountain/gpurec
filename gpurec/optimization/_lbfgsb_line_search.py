"""Internal scalar line-search helpers for L-BFGS-B."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor


LossClosure = Callable[[], Tensor]


@dataclass(frozen=True)
class _LineSearchResult:
    accepted: bool
    flat: Tensor
    loss: Tensor
    alpha: float
    delta: Tensor
    directional_derivative: float
    step_inf: float
    decrease: float
    loss_evals: int
    next_alpha: float
    armijo_required_decrease: float


_LineSearchResult.__module__ = "gpurec.optimization.lbfgsb"


class LBFGSBLineSearchMixin:
    """Private scalar backtracking line search for ``LBFGSB``."""

    def _backtracking_line_search(
        self,
        closure: LossClosure,
        loss_closure: LossClosure | None,
        *,
        flat: Tensor,
        loss: Tensor,
        grad: Tensor,
        direction: Tensor,
        initial_alpha: float,
        max_ls: int,
        c1: float,
        shrink: float,
        tolerance_change: float,
        lower_bound: float | Tensor | None,
        upper_bound: float | Tensor | None,
    ) -> _LineSearchResult:
        accepted_flat = flat.clone()
        accepted_loss = loss
        accepted_alpha = 0.0
        accepted_delta = torch.zeros_like(flat)
        accepted_gtd = 0.0
        accepted_step_inf = 0.0
        evals = 0
        loss_decrease = 0.0
        required_decrease = 0.0
        alpha = float(initial_alpha)

        for _probe in range(max_ls):
            trial_flat = self._project_flat(
                flat + alpha * direction,
                lower_bound,
                upper_bound,
            )
            delta = trial_flat - flat
            step_inf = float(delta.abs().amax().cpu()) if delta.numel() else 0.0
            if step_inf <= tolerance_change:
                alpha *= shrink
                continue
            trial_gtd = torch.dot(grad, delta)
            if (
                (not torch.isfinite(trial_gtd))
                or float(trial_gtd.detach().cpu()) >= -1e-12
            ):
                alpha *= shrink
                continue
            self._set_flat_param(trial_flat)
            trial_loss = self._evaluate_loss(closure, loss_closure)
            evals += 1
            loss_value = float(loss.detach().cpu())
            trial_value = float(trial_loss.detach().cpu())
            required_decrease = self._armijo_required_decrease(
                loss=loss,
                trial_gtd=trial_gtd,
                c1=c1,
            )
            loss_decrease = loss_value - trial_value
            if self._armijo_accepts(
                trial_loss=trial_loss,
                loss=loss,
                trial_gtd=trial_gtd,
                c1=c1,
            ):
                accepted_flat = trial_flat.detach().clone()
                accepted_loss = trial_loss.detach().clone()
                accepted_alpha = alpha
                accepted_delta = delta.detach().clone()
                accepted_gtd = float(trial_gtd.detach().cpu())
                accepted_step_inf = step_inf
                self._set_flat_param(flat)
                return _LineSearchResult(
                    accepted=True,
                    flat=accepted_flat,
                    loss=accepted_loss,
                    alpha=accepted_alpha,
                    delta=accepted_delta,
                    directional_derivative=accepted_gtd,
                    step_inf=accepted_step_inf,
                    decrease=loss_decrease,
                    loss_evals=evals,
                    next_alpha=alpha * shrink,
                    armijo_required_decrease=required_decrease,
                )
            alpha *= shrink

        self._set_flat_param(flat)
        return _LineSearchResult(
            accepted=False,
            flat=accepted_flat,
            loss=accepted_loss,
            alpha=accepted_alpha,
            delta=accepted_delta,
            directional_derivative=accepted_gtd,
            step_inf=accepted_step_inf,
            decrease=loss_decrease,
            loss_evals=evals,
            next_alpha=alpha,
            armijo_required_decrease=required_decrease,
        )
