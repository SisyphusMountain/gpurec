"""Internal projected-gradient fallback helpers for scalar L-BFGS-B.

This module is private optimization support for ``LBFGSB`` and not a public
import surface.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from .lbfgsb import _LineSearchResult


LossClosure = Callable[[], Tensor]


class LBFGSBFallbackMixin:
    """Private projected-gradient fallback/search methods for ``LBFGSB``."""

    def _projected_gradient_direction(
        self,
        flat: Tensor,
        projected_grad: Tensor,
        lower_bound: float | Tensor | None,
        upper_bound: float | Tensor | None,
    ) -> Tensor:
        direction = -projected_grad
        return self._project_flat(flat + direction, lower_bound, upper_bound) - flat

    def _projected_gradient_sign_direction(
        self,
        flat: Tensor,
        projected_grad: Tensor,
        lower_bound: float | Tensor | None,
        upper_bound: float | Tensor | None,
    ) -> Tensor:
        direction = -projected_grad.sign()
        return self._project_flat(flat + direction, lower_bound, upper_bound) - flat

    def _projected_gradient_topk_sign_direction(
        self,
        flat: Tensor,
        projected_grad: Tensor,
        lower_bound: float | Tensor | None,
        upper_bound: float | Tensor | None,
        *,
        k: int,
    ) -> Tensor:
        if projected_grad.numel() == 0 or k <= 0:
            return torch.zeros_like(projected_grad)
        count = min(int(k), int(projected_grad.numel()))
        values = projected_grad.abs()
        if not bool((values > 0).any().detach().cpu()):
            return torch.zeros_like(projected_grad)
        topk = torch.topk(values, count).indices
        direction = torch.zeros_like(projected_grad)
        direction[topk] = -projected_grad[topk].sign()
        return self._project_flat(flat + direction, lower_bound, upper_bound) - flat

    def _topk_sign_fallback_sizes(self, numel: int) -> tuple[int, ...]:
        sizes: list[int] = []
        for size in (1, 2, 3, 4, 5, 8, 13, 20, 50, 200, 500):
            if size > numel:
                continue
            if size not in sizes:
                sizes.append(size)
        return tuple(sizes)

    def _topk_sign_fallback_search(
        self,
        *,
        closure: LossClosure,
        loss_closure: LossClosure | None,
        flat: Tensor,
        loss: Tensor,
        grad: Tensor,
        projected_grad: Tensor,
        lower_bound: float | Tensor | None,
        upper_bound: float | Tensor | None,
        initial_alpha: float,
        max_ls: int,
        c1: float,
        shrink: float,
        tolerance_change: float,
        topk_sizes: tuple[int, ...] | None = None,
        competitive: bool = True,
        max_loss_evals: int | None = None,
    ) -> tuple[_LineSearchResult | None, str, int]:
        if max_loss_evals is not None and max_loss_evals <= 0:
            return None, "none", 0
        if topk_sizes is None:
            topk_sizes = tuple(
                size
                for size in self._topk_sign_fallback_sizes(int(projected_grad.numel()))
                if size <= 5
            )
        total_loss_evals = 0

        def probe(topk_size: int, *, probe_max_ls: int) -> _LineSearchResult | None:
            direction = self._projected_gradient_topk_sign_direction(
                flat,
                projected_grad,
                lower_bound,
                upper_bound,
                k=topk_size,
            )
            if (not torch.isfinite(direction).all()) or not bool(
                (direction != 0).any()
            ):
                return None
            gtd = torch.dot(grad, direction)
            if (not torch.isfinite(gtd)) or float(gtd.detach().cpu()) >= -1e-12:
                return None
            return self._backtracking_line_search(
                closure=closure,
                loss_closure=loss_closure,
                flat=flat,
                loss=loss,
                grad=grad,
                direction=direction,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                initial_alpha=max(float(initial_alpha), 4.0),
                max_ls=probe_max_ls,
                c1=c1,
                shrink=shrink,
                tolerance_change=tolerance_change,
            )

        best_search: _LineSearchResult | None = None
        best_kind = "none"
        for topk_size in topk_sizes:
            remaining = self._remaining_loss_eval_budget(
                max_loss_evals,
                total_loss_evals,
            )
            if remaining is not None and remaining <= 0:
                break
            probe_max_ls = max_ls if remaining is None else min(max_ls, remaining)
            if probe_max_ls <= 0:
                break
            search = probe(topk_size, probe_max_ls=probe_max_ls)
            if search is None:
                continue
            total_loss_evals += search.loss_evals
            if search.accepted:
                kind = f"projected_gradient_top{topk_size}_sign_fallback"
                if not competitive:
                    return search, kind, total_loss_evals
                if best_search is None or search.decrease > best_search.decrease:
                    best_search = search
                    best_kind = kind
        if best_search is not None:
            return best_search, best_kind, total_loss_evals

        return None, "none", total_loss_evals

    def _coordinate_sign_fallback_search(
        self,
        *,
        closure: LossClosure,
        loss_closure: LossClosure | None,
        flat: Tensor,
        loss: Tensor,
        grad: Tensor,
        projected_grad: Tensor,
        lower_bound: float | Tensor | None,
        upper_bound: float | Tensor | None,
        initial_alpha: float,
        max_coordinates: int = 16,
        max_ls: int,
        c1: float,
        shrink: float,
        tolerance_change: float,
        max_loss_evals: int | None = None,
    ) -> tuple[_LineSearchResult | None, str, int]:
        if max_loss_evals is not None and max_loss_evals <= 0:
            return None, "none", 0
        if projected_grad.numel() == 0 or max_coordinates <= 0:
            return None, "none", 0
        values = projected_grad.abs()
        if not bool((values > 0).any().detach().cpu()):
            return None, "none", 0
        count = min(int(max_coordinates), int(projected_grad.numel()))
        order = torch.topk(values, count).indices
        total_loss_evals = 0
        coordinate_max_ls = min(max_ls, 8)
        meaningful_decrease = max(1.0, 16.0 * self._loss_resolution(loss))
        best_overall: _LineSearchResult | None = None
        best_overall_kind = "none"
        directions: list[tuple[int, Tensor]] = []

        for rank, index in enumerate(order.tolist(), start=1):
            direction = torch.zeros_like(projected_grad)
            sign = projected_grad[index].sign()
            if float(sign.detach().cpu()) == 0.0:
                continue
            direction[index] = -sign
            direction = self._project_flat(
                flat + direction,
                lower_bound,
                upper_bound,
            ) - flat
            if (not torch.isfinite(direction).all()) or not bool(
                (direction != 0).any()
            ):
                continue
            gtd = torch.dot(grad, direction)
            if (not torch.isfinite(gtd)) or float(gtd.detach().cpu()) >= -1e-12:
                continue
            directions.append((rank, direction))

        alpha = max(float(initial_alpha), 4.0)
        for _ in range(coordinate_max_ls):
            remaining = self._remaining_loss_eval_budget(
                max_loss_evals,
                total_loss_evals,
            )
            if remaining is not None and remaining <= 0:
                break
            best_at_alpha: _LineSearchResult | None = None
            best_at_alpha_kind = "none"
            for rank, direction in directions:
                remaining = self._remaining_loss_eval_budget(
                    max_loss_evals,
                    total_loss_evals,
                )
                if remaining is not None and remaining <= 0:
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
                    initial_alpha=alpha,
                    max_ls=1,
                    c1=c1,
                    shrink=shrink,
                    tolerance_change=tolerance_change,
                )
                total_loss_evals += search.loss_evals
                if search.accepted and (
                    best_at_alpha is None or search.decrease > best_at_alpha.decrease
                ):
                    best_at_alpha = search
                    best_at_alpha_kind = f"projected_gradient_coord{rank}_sign_fallback"
            if best_at_alpha is not None:
                if (
                    best_overall is None
                    or best_at_alpha.decrease > best_overall.decrease
                ):
                    best_overall = best_at_alpha
                    best_overall_kind = best_at_alpha_kind
                if best_at_alpha.decrease >= meaningful_decrease:
                    return best_at_alpha, best_at_alpha_kind, total_loss_evals
            alpha *= shrink

        if best_overall is not None:
            return best_overall, best_overall_kind, total_loss_evals
        return None, "none", total_loss_evals

    @staticmethod
    def _remaining_loss_eval_budget(
        max_loss_evals: int | None,
        used_loss_evals: int,
    ) -> int | None:
        if max_loss_evals is None:
            return None
        return max(0, int(max_loss_evals) - int(used_loss_evals))

    def _loss_resolution(self, loss: Tensor) -> float:
        if torch.is_floating_point(loss):
            eps = torch.finfo(loss.dtype).eps
        else:
            eps = torch.finfo(torch.float64).eps
        loss_value = abs(float(loss.detach().cpu()))
        return float(eps) * max(loss_value, 1.0)

    def _tiny_progress(
        self,
        *,
        flat: Tensor,
        loss: Tensor,
        step_inf: float,
        decrease: float,
        tolerance_change: float,
    ) -> bool:
        flat_scale = (
            max(float(flat.detach().abs().amax().cpu()), 1.0)
            if flat.numel()
            else 1.0
        )
        step_floor = max(
            tolerance_change,
            math.sqrt(torch.finfo(flat.dtype).eps) * flat_scale,
        )
        decrease_floor = max(tolerance_change, self._loss_resolution(loss))
        return step_inf <= step_floor or decrease <= decrease_floor

    def _fallback_needs_competition(
        self,
        search: _LineSearchResult,
        *,
        flat: Tensor,
        loss: Tensor,
        tolerance_change: float,
        resolution_competition_factor: float,
    ) -> bool:
        # At large fp32 likelihood scales, a nominally positive fallback decrease can
        # still sit in the evaluation-resolution band. Spend the fallback budget on
        # sign/coordinate competitors before accepting that as a meaningful escape.
        meaningful_decrease = max(
            tolerance_change,
            max(0.0, resolution_competition_factor) * self._loss_resolution(loss),
        )
        return (
            (not search.accepted)
            or search.decrease <= meaningful_decrease
            or self._tiny_progress(
                flat=flat,
                loss=loss,
                step_inf=search.step_inf,
                decrease=search.decrease,
                tolerance_change=tolerance_change,
            )
        )

    def _compete_projected_gradient_fallbacks(
        self,
        *,
        closure: LossClosure,
        loss_closure: LossClosure | None,
        state: dict,
        flat: Tensor,
        loss: Tensor,
        grad: Tensor,
        projected_grad: Tensor,
        lower_bound: float | Tensor | None,
        upper_bound: float | Tensor | None,
        current_search: _LineSearchResult,
        current_kind: str,
        lr: float,
        max_ls: int,
        c1: float,
        shrink: float,
        tolerance_change: float,
        max_coordinates: int = 16,
        max_loss_evals: int | None = None,
        resolution_competition_factor: float = 0.0,
    ) -> tuple[_LineSearchResult, str, int]:
        best_search = current_search
        best_kind = current_kind
        total_loss_evals = 0
        alternative_max_ls = min(max_ls, 8)

        def remaining_budget() -> int | None:
            return self._remaining_loss_eval_budget(
                max_loss_evals,
                total_loss_evals,
            )

        def capped_max_ls(value: int) -> int:
            remaining = remaining_budget()
            return int(value) if remaining is None else min(int(value), remaining)

        if not self._fallback_needs_competition(
            best_search,
            flat=flat,
            loss=loss,
            tolerance_change=tolerance_change,
            resolution_competition_factor=resolution_competition_factor,
        ):
            return best_search, best_kind, total_loss_evals

        direction = self._projected_gradient_sign_direction(
            flat,
            projected_grad,
            lower_bound,
            upper_bound,
        )
        if torch.isfinite(direction).all() and bool((direction != 0).any()):
            gtd = torch.dot(grad, direction)
            if torch.isfinite(gtd) and float(gtd.detach().cpu()) < -1e-12:
                sign_max_ls = capped_max_ls(alternative_max_ls)
                if sign_max_ls > 0:
                    sign_search = self._backtracking_line_search(
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
                        ),
                        max_ls=sign_max_ls,
                        c1=c1,
                        shrink=shrink,
                        tolerance_change=tolerance_change,
                    )
                    total_loss_evals += sign_search.loss_evals
                    if sign_search.accepted and (
                        not best_search.accepted
                        or sign_search.decrease > best_search.decrease
                    ):
                        best_search = sign_search
                        best_kind = "projected_gradient_sign_fallback"

        if self._fallback_needs_competition(
            best_search,
            flat=flat,
            loss=loss,
            tolerance_change=tolerance_change,
            resolution_competition_factor=resolution_competition_factor,
        ):
            topk_sizes = self._topk_sign_fallback_sizes(int(projected_grad.numel()))
            topk_search, topk_kind, topk_loss_evals = (
                self._topk_sign_fallback_search(
                    closure=closure,
                    loss_closure=loss_closure,
                    flat=flat,
                    loss=loss,
                    grad=grad,
                    projected_grad=projected_grad,
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                    initial_alpha=lr,
                    max_ls=alternative_max_ls,
                    c1=c1,
                    shrink=shrink,
                    tolerance_change=tolerance_change,
                    max_loss_evals=remaining_budget(),
                )
            )
            total_loss_evals += topk_loss_evals
            if topk_search is not None and (
                not best_search.accepted
                or topk_search.decrease > best_search.decrease
            ):
                best_search = topk_search
                best_kind = topk_kind

            if self._fallback_needs_competition(
                best_search,
                flat=flat,
                loss=loss,
                tolerance_change=tolerance_change,
                resolution_competition_factor=resolution_competition_factor,
            ):
                coord_search, coord_kind, coord_loss_evals = (
                    self._coordinate_sign_fallback_search(
                        closure=closure,
                        loss_closure=loss_closure,
                        flat=flat,
                        loss=loss,
                        grad=grad,
                        projected_grad=projected_grad,
                        lower_bound=lower_bound,
                        upper_bound=upper_bound,
                        initial_alpha=lr,
                        max_coordinates=max_coordinates,
                        max_ls=alternative_max_ls,
                        c1=c1,
                        shrink=shrink,
                        tolerance_change=tolerance_change,
                        max_loss_evals=remaining_budget(),
                    )
                )
                total_loss_evals += coord_loss_evals
                if coord_search is not None and (
                    not best_search.accepted
                    or coord_search.decrease > best_search.decrease
                ):
                    best_search = coord_search
                    best_kind = coord_kind

            if not best_search.accepted:
                large_topk_sizes = tuple(size for size in topk_sizes if size > 5)
                topk_search, topk_kind, topk_loss_evals = (
                    self._topk_sign_fallback_search(
                        closure=closure,
                        loss_closure=loss_closure,
                        flat=flat,
                        loss=loss,
                        grad=grad,
                        projected_grad=projected_grad,
                        lower_bound=lower_bound,
                        upper_bound=upper_bound,
                        initial_alpha=lr,
                        max_ls=alternative_max_ls,
                        c1=c1,
                        shrink=shrink,
                        tolerance_change=tolerance_change,
                        topk_sizes=large_topk_sizes,
                        competitive=False,
                        max_loss_evals=remaining_budget(),
                    )
                )
                total_loss_evals += topk_loss_evals
                if topk_search is not None:
                    best_search = topk_search
                    best_kind = topk_kind

        return best_search, best_kind, total_loss_evals

    def _adaptive_projected_gradient_alpha(
        self,
        state: dict,
        *,
        lr: float,
        shrink: float,
        upper_alpha: float | None = None,
    ) -> float:
        previous = float(state.get("last_projected_gradient_fallback_alpha", 0.0))
        if previous > 0.0 and math.isfinite(previous):
            alpha = min(lr, previous / shrink)
        else:
            alpha = lr
        if upper_alpha is not None and upper_alpha > 0.0 and math.isfinite(upper_alpha):
            alpha = min(alpha, upper_alpha)
        return max(alpha, torch.finfo(torch.float64).tiny)
