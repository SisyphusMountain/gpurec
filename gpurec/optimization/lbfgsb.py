"""PyTorch L-BFGS-B optimizer for one dense bounded parameter tensor."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor
from torch.optim import Optimizer

from ._armijo import ScalarArmijoMixin
from ._bounds import BoxBoundsMixin
from ._closures import ScalarClosureMixin
from ._lbfgsb_fallbacks import LBFGSBFallbackMixin
from ._lbfgsb_line_search import (
    LBFGSBLineSearchMixin,
    _LineSearchResult as _LineSearchResult,
)
from ._lbfgsb_subspace import LBFGSBSubspaceMixin


LossClosure = Callable[[], Tensor]


class LBFGSB(
    LBFGSBFallbackMixin,
    LBFGSBLineSearchMixin,
    LBFGSBSubspaceMixin,
    BoxBoundsMixin,
    ScalarClosureMixin,
    ScalarArmijoMixin,
    Optimizer,
):
    """Limited-memory BFGS with box constraints.

    This is a PyTorch implementation of the L-BFGS-B structure described by
    Byrd, Lu, Nocedal, and Zhu: it computes a generalized Cauchy point along
    the projected gradient path, then attempts a reduced free-subspace step.
    The implementation is scoped to gpurec's single dense parameter tensor and
    uses loss-only Armijo probes for line search.
    """

    _optimizer_name = "LBFGSB"
    _bounds_scalar_to_flat = True

    def __init__(
        self,
        params,
        *,
        lr: float = 1.0,
        max_iter: int = 1,
        history_size: int = 10,
        max_ls: int = 20,
        c1: float = 1e-4,
        shrink: float = 0.5,
        tolerance_grad: float = 1e-7,
        tolerance_change: float = 1e-12,
        lower_bound: float | Tensor | None = None,
        upper_bound: float | Tensor | None = None,
        active_tol: float = 1e-10,
        cg_max_iter: int | None = None,
        cg_tol: float = 1e-4,
        fallback_max_ls: int | None = None,
        fallback_max_coordinates: int = 16,
        fallback_max_loss_evals: int | None = None,
        fallback_resolution_competition_factor: float = 0.0,
    ) -> None:
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {max_iter}")
        if history_size < 1:
            raise ValueError(f"history_size must be >= 1, got {history_size}")
        if max_ls < 1:
            raise ValueError(f"max_ls must be >= 1, got {max_ls}")
        if not (0.0 < c1 < 1.0):
            raise ValueError(f"c1 must be in (0, 1), got {c1}")
        if not (0.0 < shrink < 1.0):
            raise ValueError(f"shrink must be in (0, 1), got {shrink}")
        if tolerance_grad < 0.0 or tolerance_change < 0.0:
            raise ValueError("tolerances must be non-negative")
        if active_tol < 0.0:
            raise ValueError("active_tol must be non-negative")
        if cg_max_iter is not None and cg_max_iter < 1:
            raise ValueError("cg_max_iter must be positive when provided")
        if cg_tol < 0.0:
            raise ValueError("cg_tol must be non-negative")
        if fallback_max_ls is not None and fallback_max_ls < 1:
            raise ValueError("fallback_max_ls must be positive when provided")
        if fallback_max_coordinates < 0:
            raise ValueError("fallback_max_coordinates must be non-negative")
        if fallback_max_loss_evals is not None and fallback_max_loss_evals < 1:
            raise ValueError(
                "fallback_max_loss_evals must be positive when provided"
            )
        if fallback_resolution_competition_factor < 0.0:
            raise ValueError(
                "fallback_resolution_competition_factor must be non-negative"
            )

        defaults = {
            "lr": float(lr),
            "max_iter": int(max_iter),
            "history_size": int(history_size),
            "max_ls": int(max_ls),
            "c1": float(c1),
            "shrink": float(shrink),
            "tolerance_grad": float(tolerance_grad),
            "tolerance_change": float(tolerance_change),
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "active_tol": float(active_tol),
            "cg_max_iter": cg_max_iter,
            "cg_tol": float(cg_tol),
            "fallback_max_ls": (
                max(32, int(max_ls))
                if fallback_max_ls is None
                else int(fallback_max_ls)
            ),
            "fallback_max_coordinates": int(fallback_max_coordinates),
            "fallback_max_loss_evals": (
                None
                if fallback_max_loss_evals is None
                else int(fallback_max_loss_evals)
            ),
            "fallback_resolution_competition_factor": float(
                fallback_resolution_competition_factor
            ),
        }
        super().__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError("LBFGSB supports exactly one parameter group")
        self._params = self.param_groups[0]["params"]
        if len(self._params) != 1:
            raise ValueError("LBFGSB supports exactly one parameter tensor")
        p = self._params[0]
        if p.ndim < 1:
            raise ValueError("LBFGSB parameter must have at least one dimension")
        if p.is_sparse:
            raise TypeError("LBFGSB parameter must be dense")
        if torch.is_complex(p):
            raise TypeError("LBFGSB only supports real-valued parameters")
        self._param = p

    def _flat_param(self) -> Tensor:
        return self._param.detach().reshape(-1)

    def _set_flat_param(self, flat: Tensor) -> None:
        with torch.no_grad():
            self._param.copy_(flat.reshape_as(self._param))

    @torch.no_grad()
    def _store_last_state(
        self,
        *,
        loss: Tensor,
        grad: Tensor,
        projected_grad: Tensor,
        grad_evals: int,
        loss_evals: int,
        accepted: bool,
        alpha: float,
        step_inf: float,
        directional_derivative: float,
        direction_kind: str,
        line_search_decrease: float,
        armijo_required_decrease: float,
        fallback_attempted: bool,
        fallback_used: bool,
        fallback_alpha: float,
        fallback_loss_evals: int,
        fallback_max_loss_evals: int | None,
        fallback_budget_exhausted: bool,
        fallback_reason: str,
        high_kkt_stall_count: int,
        history_cleared_for_fallback: bool,
    ) -> None:
        state = self.state[self._param]
        state["last_loss"] = loss.detach().reshape(())
        state["last_grad"] = grad.detach().clone()
        state["last_projected_grad"] = projected_grad.detach().clone()
        state["last_grad_evals"] = int(grad_evals)
        state["last_loss_evals"] = int(loss_evals)
        state["last_accepted"] = bool(accepted)
        state["last_alpha"] = float(alpha)
        state["last_step_inf"] = float(step_inf)
        state["last_directional_derivative"] = float(directional_derivative)
        state["last_direction_kind"] = str(direction_kind)
        state["last_line_search_decrease"] = float(line_search_decrease)
        state["last_armijo_required_decrease"] = float(armijo_required_decrease)
        state["last_fallback_attempted"] = bool(fallback_attempted)
        state["last_fallback_used"] = bool(fallback_used)
        state["last_fallback_alpha"] = float(fallback_alpha)
        state["last_fallback_loss_evals"] = int(fallback_loss_evals)
        state["last_fallback_max_loss_evals"] = (
            None
            if fallback_max_loss_evals is None
            else int(fallback_max_loss_evals)
        )
        state["last_fallback_budget_exhausted"] = bool(fallback_budget_exhausted)
        state["last_fallback_reason"] = str(fallback_reason)
        state["last_high_kkt_stall_count"] = int(high_kkt_stall_count)
        state["last_history_cleared_for_fallback"] = bool(history_cleared_for_fallback)

    def step(
        self,
        closure: LossClosure,
        *,
        loss_closure: LossClosure | None = None,
    ) -> Tensor:
        group = self.param_groups[0]
        lr = float(group["lr"])
        max_iter = int(group["max_iter"])
        history_size = int(group["history_size"])
        max_ls = int(group["max_ls"])
        fallback_max_ls = int(group["fallback_max_ls"])
        fallback_max_coordinates = int(group.get("fallback_max_coordinates", 16))
        fallback_max_loss_evals_raw = group.get("fallback_max_loss_evals")
        fallback_max_loss_evals = (
            None
            if fallback_max_loss_evals_raw is None
            else int(fallback_max_loss_evals_raw)
        )
        fallback_resolution_competition_factor = float(
            group.get("fallback_resolution_competition_factor", 0.0)
        )
        c1 = float(group["c1"])
        shrink = float(group["shrink"])
        tolerance_grad = float(group["tolerance_grad"])
        tolerance_change = float(group["tolerance_change"])
        lower_bound = group["lower_bound"]
        upper_bound = group["upper_bound"]
        active_tol = float(group["active_tol"])
        cg_max_iter = group["cg_max_iter"]
        cg_tol = float(group["cg_tol"])

        state = self.state[self._param]
        old_dirs = state.setdefault("old_dirs", [])
        old_stps = state.setdefault("old_stps", [])

        initial_flat = self._flat_param().clone()
        projected_initial = self._project_flat(initial_flat, lower_bound, upper_bound)
        if not torch.equal(initial_flat, projected_initial):
            self._set_flat_param(projected_initial)

        loss, grad = self._evaluate_with_grad(closure)
        grad_evals = 1
        loss_evals = 0
        flat = self._flat_param().clone()
        projected_grad = self._projected_gradient(flat, grad, lower_bound, upper_bound)
        accepted_any = False
        last_alpha = 0.0
        last_step_inf = 0.0
        last_gtd = 0.0
        last_kind = "none"
        last_line_search_decrease = 0.0
        last_armijo_required_decrease = 0.0
        fallback_attempted = False
        fallback_used = False
        fallback_alpha = 0.0
        fallback_loss_evals = 0
        fallback_reason = "none"
        history_cleared_for_fallback = False
        high_kkt_stall_count = int(state.get("consecutive_high_kkt_stalls", 0))

        for _ in range(max_iter):
            projected_grad_inf = (
                float(projected_grad.abs().amax().cpu())
                if projected_grad.numel()
                else 0.0
            )
            if projected_grad_inf <= tolerance_grad:
                break

            force_fallback = high_kkt_stall_count >= 2
            initial_alpha = lr
            if force_fallback:
                if old_dirs:
                    old_dirs.clear()
                    old_stps.clear()
                history_cleared_for_fallback = True
                fallback_attempted = True
                fallback_reason = "high_kkt_tiny_progress"
                direction = self._projected_gradient_direction(
                    flat,
                    projected_grad,
                    lower_bound,
                    upper_bound,
                )
                direction_kind = "projected_gradient_fallback"
                initial_alpha = self._adaptive_projected_gradient_alpha(
                    state,
                    lr=lr,
                    shrink=shrink,
                )
            else:
                direction, direction_kind = self._candidate_direction(
                    flat,
                    grad,
                    lower_bound,
                    upper_bound,
                    old_dirs,
                    old_stps,
                    active_tol=active_tol,
                    cg_max_iter=cg_max_iter,
                    cg_tol=cg_tol,
                )

            if (not torch.isfinite(direction).all()) or not bool((direction != 0).any()):
                break

            gtd = torch.dot(grad, direction)
            if (
                (not torch.isfinite(gtd))
                or float(gtd.detach().cpu()) >= -1e-12
            ) and not force_fallback:
                if old_dirs:
                    old_dirs.clear()
                    old_stps.clear()
                direction, direction_kind = self._candidate_direction(
                    flat,
                    grad,
                    lower_bound,
                    upper_bound,
                    old_dirs,
                    old_stps,
                    active_tol=active_tol,
                    cg_max_iter=cg_max_iter,
                    cg_tol=cg_tol,
                )
                gtd = torch.dot(grad, direction)
            if (not torch.isfinite(gtd)) or float(gtd.detach().cpu()) >= -1e-12:
                break

            step_max_ls = fallback_max_ls if force_fallback else max_ls
            if force_fallback:
                remaining = self._remaining_loss_eval_budget(
                    fallback_max_loss_evals,
                    fallback_loss_evals,
                )
                step_max_ls = (
                    fallback_max_ls
                    if remaining is None
                    else min(fallback_max_ls, remaining)
                )
                if step_max_ls <= 0:
                    break

            search = self._backtracking_line_search(
                closure=closure,
                loss_closure=loss_closure,
                flat=flat,
                loss=loss,
                grad=grad,
                direction=direction,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                initial_alpha=initial_alpha,
                max_ls=step_max_ls,
                c1=c1,
                shrink=shrink,
                tolerance_change=tolerance_change,
            )
            loss_evals += search.loss_evals
            if force_fallback:
                fallback_loss_evals += search.loss_evals

            if force_fallback:
                search, direction_kind, extra_loss_evals = (
                    self._compete_projected_gradient_fallbacks(
                        closure=closure,
                        loss_closure=loss_closure,
                        state=state,
                        flat=flat,
                        loss=loss,
                        grad=grad,
                        projected_grad=projected_grad,
                        lower_bound=lower_bound,
                        upper_bound=upper_bound,
                        current_search=search,
                        current_kind=direction_kind,
                        lr=lr,
                        max_ls=fallback_max_ls,
                        max_coordinates=fallback_max_coordinates,
                        max_loss_evals=self._remaining_loss_eval_budget(
                            fallback_max_loss_evals,
                            fallback_loss_evals,
                        ),
                        resolution_competition_factor=(
                            fallback_resolution_competition_factor
                        ),
                        c1=c1,
                        shrink=shrink,
                        tolerance_change=tolerance_change,
                    )
                )
                loss_evals += extra_loss_evals
                fallback_loss_evals += extra_loss_evals

            if not search.accepted and not force_fallback:
                if old_dirs:
                    old_dirs.clear()
                    old_stps.clear()
                history_cleared_for_fallback = True
                fallback_attempted = True
                fallback_reason = "line_search_failed"
                direction = self._projected_gradient_direction(
                    flat,
                    projected_grad,
                    lower_bound,
                    upper_bound,
                )
                direction_kind = "projected_gradient_fallback"
                if (not torch.isfinite(direction).all()) or not bool((direction != 0).any()):
                    break
                gtd = torch.dot(grad, direction)
                if (not torch.isfinite(gtd)) or float(gtd.detach().cpu()) >= -1e-12:
                    break
                remaining = self._remaining_loss_eval_budget(
                    fallback_max_loss_evals,
                    fallback_loss_evals,
                )
                fallback_line_search_max_ls = (
                    fallback_max_ls
                    if remaining is None
                    else min(fallback_max_ls, remaining)
                )
                if fallback_line_search_max_ls <= 0:
                    break
                search = self._backtracking_line_search(
                    closure=closure,
                    loss_closure=loss_closure,
                    flat=flat,
                    loss=loss,
                    grad=grad,
                    direction=direction,
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                    initial_alpha=self._adaptive_projected_gradient_alpha(
                        state,
                        lr=lr,
                        shrink=shrink,
                        upper_alpha=search.next_alpha,
                    ),
                    max_ls=fallback_line_search_max_ls,
                    c1=c1,
                    shrink=shrink,
                    tolerance_change=tolerance_change,
                )
                loss_evals += search.loss_evals
                fallback_loss_evals += search.loss_evals
                search, direction_kind, extra_loss_evals = (
                    self._compete_projected_gradient_fallbacks(
                        closure=closure,
                        loss_closure=loss_closure,
                        state=state,
                        flat=flat,
                        loss=loss,
                        grad=grad,
                        projected_grad=projected_grad,
                        lower_bound=lower_bound,
                        upper_bound=upper_bound,
                        current_search=search,
                        current_kind=direction_kind,
                        lr=lr,
                        max_ls=fallback_max_ls,
                        max_coordinates=fallback_max_coordinates,
                        max_loss_evals=self._remaining_loss_eval_budget(
                            fallback_max_loss_evals,
                            fallback_loss_evals,
                        ),
                        resolution_competition_factor=(
                            fallback_resolution_competition_factor
                        ),
                        c1=c1,
                        shrink=shrink,
                        tolerance_change=tolerance_change,
                    )
                )
                loss_evals += extra_loss_evals
                fallback_loss_evals += extra_loss_evals

            if not search.accepted:
                if projected_grad_inf > tolerance_grad:
                    high_kkt_stall_count += 1
                    state["consecutive_high_kkt_stalls"] = high_kkt_stall_count
                break

            if direction_kind in {
                "projected_gradient_fallback",
                "projected_gradient_sign_fallback",
            } or direction_kind.startswith(
                ("projected_gradient_top", "projected_gradient_coord")
            ):
                fallback_used = True
                fallback_alpha = search.alpha
                state["last_projected_gradient_fallback_alpha"] = search.alpha

            self._set_flat_param(search.flat)
            new_loss, new_grad = self._evaluate_with_grad(closure)
            grad_evals += 1
            new_flat = self._flat_param().clone()
            new_projected_grad = self._projected_gradient(
                new_flat,
                new_grad,
                lower_bound,
                upper_bound,
            )
            s = new_flat - flat
            y = new_grad - grad
            sy = torch.dot(s, y)
            yy = torch.dot(y, y)
            finite_update = bool(
                (
                    torch.isfinite(s).all()
                    & torch.isfinite(y).all()
                    & torch.isfinite(sy)
                    & torch.isfinite(yy)
                ).detach().cpu()
            )
            if finite_update and bool((sy > torch.finfo(s.dtype).eps * yy).detach().cpu()):
                old_dirs.append(s.detach().clone())
                old_stps.append(y.detach().clone())
                if len(old_dirs) > history_size:
                    old_dirs.pop(0)
                    old_stps.pop(0)

            loss = new_loss.detach() if torch.isfinite(new_loss) else search.loss
            grad = new_grad.detach()
            flat = new_flat.detach().clone()
            projected_grad = new_projected_grad.detach()
            accepted_any = True
            last_alpha = search.alpha
            last_gtd = search.directional_derivative
            last_kind = direction_kind
            last_step_inf = search.step_inf
            last_line_search_decrease = search.decrease
            last_armijo_required_decrease = search.armijo_required_decrease

            new_projected_grad_inf = (
                float(projected_grad.abs().amax().cpu())
                if projected_grad.numel()
                else 0.0
            )
            if (
                new_projected_grad_inf > tolerance_grad
                and self._tiny_progress(
                    flat=flat,
                    loss=loss,
                    step_inf=search.step_inf,
                    decrease=search.decrease,
                    tolerance_change=tolerance_change,
                )
            ):
                high_kkt_stall_count += 1
            else:
                high_kkt_stall_count = 0
            state["consecutive_high_kkt_stalls"] = high_kkt_stall_count

            if float(search.delta.abs().amax().detach().cpu()) <= tolerance_change:
                break

        if not accepted_any:
            self._set_flat_param(flat)
        fallback_budget_exhausted = (
            fallback_max_loss_evals is not None
            and fallback_loss_evals >= fallback_max_loss_evals
        )

        self._store_last_state(
            loss=loss,
            grad=grad,
            projected_grad=projected_grad,
            grad_evals=grad_evals,
            loss_evals=loss_evals,
            accepted=accepted_any,
            alpha=last_alpha,
            step_inf=last_step_inf,
            directional_derivative=last_gtd,
            direction_kind=last_kind,
            line_search_decrease=last_line_search_decrease,
            armijo_required_decrease=last_armijo_required_decrease,
            fallback_attempted=fallback_attempted,
            fallback_used=fallback_used,
            fallback_alpha=fallback_alpha,
            fallback_loss_evals=fallback_loss_evals,
            fallback_max_loss_evals=fallback_max_loss_evals,
            fallback_budget_exhausted=fallback_budget_exhausted,
            fallback_reason=fallback_reason,
            high_kkt_stall_count=high_kkt_stall_count,
            history_cleared_for_fallback=history_cleared_for_fallback,
        )
        state["n_iter"] = int(state.get("n_iter", 0)) + int(max_iter)
        return loss
