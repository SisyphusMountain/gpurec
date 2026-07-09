"""Row-wise batched L-BFGS for independent objectives.

``BatchedLBFGS`` is meant for tensors such as genewise DTL parameters where
``theta.shape == [G, ...]`` and row ``g`` only affects objective ``f_g``.  It
keeps one L-BFGS history per row while evaluating all rows in one batched
closure.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor
from torch.optim import Optimizer


LossClosure = Callable[[], Tensor]
Bound = float | Tensor | None


def _row_dot(a: Tensor, b: Tensor) -> Tensor:
    return (a * b).sum(dim=1)


def _clamp_tensor(value: Tensor, lower: Tensor, upper: Tensor) -> Tensor:
    return torch.minimum(torch.maximum(value, lower), upper)


def _cubic_interpolate(
    x1: Tensor,
    f1: Tensor,
    g1: Tensor,
    x2: Tensor,
    f2: Tensor,
    g2: Tensor,
    *,
    bounds: tuple[Tensor, Tensor] | None = None,
) -> Tensor:
    """Vectorized port of ``torch.optim.lbfgs._cubic_interpolate``."""
    if bounds is None:
        xmin_bound = torch.minimum(x1, x2)
        xmax_bound = torch.maximum(x1, x2)
    else:
        xmin_bound, xmax_bound = bounds

    x_diff = x1 - x2
    d1 = g1 + g2 - 3 * (f1 - f2) / x_diff
    d2_square = d1.square() - g1 * g2
    d2 = d2_square.clamp_min(0).sqrt()

    denom_forward = g2 - g1 + 2 * d2
    denom_reverse = g1 - g2 + 2 * d2
    min_forward = x2 - (x2 - x1) * ((g2 + d2 - d1) / denom_forward)
    min_reverse = x1 - (x1 - x2) * ((g1 + d2 - d1) / denom_reverse)
    min_pos = torch.where(x1 <= x2, min_forward, min_reverse)

    midpoint = (xmin_bound + xmax_bound) / 2.0
    denom = torch.where(x1 <= x2, denom_forward, denom_reverse)
    eps = torch.finfo(x1.dtype).eps
    valid = (
        (d2_square >= 0)
        & torch.isfinite(min_pos)
        & torch.isfinite(denom)
        & torch.isfinite(x_diff)
        & (denom.abs() > eps)
        & (x_diff.abs() > eps)
    )
    return _clamp_tensor(torch.where(valid, min_pos, midpoint), xmin_bound, xmax_bound)


def bound_for_flat(
    bound: Bound,
    flat: Tensor,
    parameter_shape: torch.Size | tuple[int, ...],
    *,
    broadcast_to_flat: bool = False,
) -> Tensor | None:
    if bound is None:
        return None
    if torch.is_tensor(bound):
        bound_tensor = bound.detach().to(device=flat.device, dtype=flat.dtype)
    else:
        bound_tensor = torch.as_tensor(bound, device=flat.device, dtype=flat.dtype)
    if bound_tensor.ndim == 0:
        return bound_tensor
    if tuple(bound_tensor.shape) == tuple(flat.shape):
        return bound_tensor
    parameter_shape = tuple(parameter_shape)
    if tuple(bound_tensor.shape) == parameter_shape:
        return bound_tensor.reshape_as(flat)
    if broadcast_to_flat:
        try:
            return torch.broadcast_to(bound_tensor, parameter_shape).reshape_as(flat)
        except RuntimeError:
            return torch.broadcast_to(bound_tensor, flat.shape)
    return torch.broadcast_to(bound_tensor, parameter_shape).reshape_as(flat)


def bounds_for_flat(
    flat: Tensor,
    lower_bound: Bound,
    upper_bound: Bound,
    parameter_shape: torch.Size | tuple[int, ...],
    *,
    broadcast_to_flat: bool = False,
) -> tuple[Tensor | None, Tensor | None]:
    lower = bound_for_flat(
        lower_bound,
        flat,
        parameter_shape,
        broadcast_to_flat=broadcast_to_flat,
    )
    upper = bound_for_flat(
        upper_bound,
        flat,
        parameter_shape,
        broadcast_to_flat=broadcast_to_flat,
    )
    if lower is not None and upper is not None and bool((lower > upper).any()):
        raise ValueError("lower_bound must be <= upper_bound")
    return lower, upper


def project_flat(
    flat: Tensor,
    lower_bound: Bound,
    upper_bound: Bound,
    parameter_shape: torch.Size | tuple[int, ...],
    *,
    broadcast_to_flat: bool = False,
) -> Tensor:
    lower, upper = bounds_for_flat(
        flat,
        lower_bound,
        upper_bound,
        parameter_shape,
        broadcast_to_flat=broadcast_to_flat,
    )
    projected = flat
    if lower is not None:
        projected = torch.maximum(projected, lower)
    if upper is not None:
        projected = torch.minimum(projected, upper)
    return projected


def projected_gradient(
    flat: Tensor,
    grad: Tensor,
    lower_bound: Bound,
    upper_bound: Bound,
    parameter_shape: torch.Size | tuple[int, ...],
    *,
    broadcast_to_flat: bool = False,
) -> Tensor:
    return flat - project_flat(
        flat - grad,
        lower_bound,
        upper_bound,
        parameter_shape,
        broadcast_to_flat=broadcast_to_flat,
    )


def feasible_direction(
    flat: Tensor,
    direction: Tensor,
    lower_bound: Bound,
    upper_bound: Bound,
    parameter_shape: torch.Size | tuple[int, ...],
    *,
    broadcast_to_flat: bool = False,
) -> Tensor:
    lower, upper = bounds_for_flat(
        flat,
        lower_bound,
        upper_bound,
        parameter_shape,
        broadcast_to_flat=broadcast_to_flat,
    )
    feasible = torch.ones_like(direction, dtype=torch.bool)
    if lower is not None:
        feasible = feasible & ((flat > lower) | (direction >= 0))
    if upper is not None:
        feasible = feasible & ((flat < upper) | (direction <= 0))
    return torch.where(feasible, direction, torch.zeros_like(direction))


def loss_vector_tensor(loss: object, batch_size: int, owner: str) -> Tensor:
    if not torch.is_tensor(loss):
        raise TypeError(f"{owner} closure must return a Tensor")
    if loss.numel() != batch_size:
        raise ValueError(
            f"{owner} closure must return one loss per parameter row; "
            f"got shape {tuple(loss.shape)} for batch size {batch_size}"
        )
    return loss.detach().reshape(batch_size)


def flat_grad(
    param: Tensor,
    flat_like: Tensor,
    owner: str,
    *,
    row_batch_size: int | None = None,
) -> Tensor:
    grad = param.grad
    if grad is None:
        return torch.zeros_like(flat_like)
    if grad.is_sparse:
        grad = grad.to_dense()
    if torch.is_complex(grad):
        raise TypeError(f"{owner} only supports real-valued gradients")
    if row_batch_size is not None:
        return grad.detach().reshape(int(row_batch_size), -1)
    return grad.detach().reshape_as(flat_like)


class _BoxBoundsMixin:
    _bounds_broadcast_to_flat = False

    def _bounds_parameter_shape(self) -> torch.Size:
        return self._param.shape

    def _project_flat(
        self,
        flat: Tensor,
        lower_bound: Bound,
        upper_bound: Bound,
    ) -> Tensor:
        return project_flat(
            flat,
            lower_bound,
            upper_bound,
            self._bounds_parameter_shape(),
            broadcast_to_flat=self._bounds_broadcast_to_flat,
        )

    def _projected_gradient(
        self,
        flat: Tensor,
        grad: Tensor,
        lower_bound: Bound,
        upper_bound: Bound,
    ) -> Tensor:
        return projected_gradient(
            flat,
            grad,
            lower_bound,
            upper_bound,
            self._bounds_parameter_shape(),
            broadcast_to_flat=self._bounds_broadcast_to_flat,
        )

    def _feasible_direction(
        self,
        flat: Tensor,
        direction: Tensor,
        lower_bound: Bound,
        upper_bound: Bound,
    ) -> Tensor:
        return feasible_direction(
            flat,
            direction,
            lower_bound,
            upper_bound,
            self._bounds_parameter_shape(),
            broadcast_to_flat=self._bounds_broadcast_to_flat,
        )


class _VectorClosureMixin:
    _optimizer_name: str

    def _evaluate_with_grad(self, closure: LossClosure) -> tuple[Tensor, Tensor]:
        with torch.enable_grad():
            loss = closure()
        return (
            loss_vector_tensor(loss, self._batch_size(), self._optimizer_name),
            flat_grad(
                self._param,
                self._flat_param(),
                self._optimizer_name,
                row_batch_size=self._batch_size(),
            ),
        )

    def _evaluate_loss(
        self,
        closure: LossClosure,
        loss_closure: LossClosure | None,
    ) -> Tensor:
        if loss_closure is None:
            with torch.enable_grad():
                loss = closure()
        else:
            with torch.no_grad():
                loss = loss_closure()
        return loss_vector_tensor(loss, self._batch_size(), self._optimizer_name)


class _BatchedLBFGSHistoryMixin:
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
        h_diag: Tensor = state["H_diag"]

        ys = _row_dot(y_k, s_k)
        yy = _row_dot(y_k, y_k)  # ||y||^2
        ss = _row_dot(s_k, s_k)  # ||s||^2
        step_norm = s_k.abs().amax(dim=1)
        # Scale-invariant curvature condition: cos(s, y) = ys / (||s|| ||y||) > eps_rel.
        # Dimensionless, so acceptance is invariant to each row's objective/parameter scaling --
        # critical here, where rows are independent problems of differing magnitude and fp32 is used
        # for the large ones. Replaces the absolute ``ys > 1e-10`` inherited from torch.optim.LBFGS,
        # which is only calibrated for a well-scaled fp64 single problem. Subsumes the old
        # ``yy > 1e-30`` guard: acceptance implies ys > 0 and y != 0.
        eps_rel = float(torch.finfo(ys.dtype).eps) ** 0.5
        valid = (
            active
            & torch.isfinite(ys)
            & torch.isfinite(yy)
            & torch.isfinite(ss)
            & (ys > eps_rel * (ss * yy).sqrt())
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
        # ``valid`` already guarantees ys > 1e-10 and yy > 1e-30 (curvature condition above), so the
        # denominators are safe on the selected rows. torch.where evaluates BOTH branches, so guard
        # only the discarded (invalid) rows against 1/0 with the mask itself -- no epsilon floor.
        safe_ys = torch.where(valid, ys, torch.ones_like(ys))
        safe_yy = torch.where(valid, yy, torch.ones_like(yy))
        ro.append(torch.where(valid, 1.0 / safe_ys, torch.zeros_like(ys)))
        state["H_diag"] = torch.where(valid, ys / safe_yy, h_diag)


class _BatchedLBFGSStrongWolfeMixin:
    def _evaluate_trial_with_grad(
        self,
        closure: LossClosure,
        *,
        start_flat: Tensor,
        direction: Tensor,
        alpha: Tensor,
        evaluate: Tensor,
        keep_flat: Tensor,
        lower_bound: Bound,
        upper_bound: Bound,
    ) -> tuple[Tensor, Tensor, Tensor]:
        trial_flat = self._project_flat(
            start_flat + alpha[:, None] * direction,
            lower_bound,
            upper_bound,
        )
        candidate_flat = torch.where(evaluate[:, None], trial_flat, keep_flat)
        self._set_flat_param(candidate_flat)
        trial_loss, trial_grad = self._evaluate_with_grad(closure)
        return trial_loss, trial_grad, trial_flat

    def _strong_wolfe(
        self,
        closure: LossClosure,
        *,
        start_flat: Tensor,
        direction: Tensor,
        start_loss: Tensor,
        start_grad: Tensor,
        gtd: Tensor,
        alpha: Tensor,
        active: Tensor,
        c1: float,
        c2: float,
        tolerance_change: float,
        max_ls: int,
        remaining_eval_budget: int,
        lower_bound: Bound,
        upper_bound: Bound,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, int]:
        batch_size = start_flat.shape[0]
        device = start_flat.device
        dtype = start_flat.dtype
        zeros = torch.zeros(batch_size, device=device, dtype=dtype)
        false = torch.zeros(batch_size, device=device, dtype=torch.bool)

        final_alpha = zeros.clone()
        final_loss = start_loss.clone()
        final_grad = start_grad.clone()
        if remaining_eval_budget <= 0 or not bool(active.any()):
            return start_flat, final_loss, final_grad, final_alpha, false, 0

        trial_loss, trial_grad, trial_flat = self._evaluate_trial_with_grad(
            closure,
            start_flat=start_flat,
            direction=direction,
            alpha=alpha,
            evaluate=active,
            keep_flat=start_flat,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )
        evals = 1
        loss_new = torch.where(active, trial_loss, start_loss)
        grad_new = torch.where(active[:, None], trial_grad, start_grad)
        trial_direction = self._feasible_direction(
            trial_flat,
            direction,
            lower_bound,
            upper_bound,
        )
        gtd_new = torch.where(active, _row_dot(grad_new, trial_direction), gtd)

        t = alpha.clone()
        t_prev = zeros.clone()
        f_prev = start_loss.clone()
        g_prev = start_grad.clone(memory_format=torch.contiguous_format)
        gtd_prev = gtd.clone()

        bracket_t0 = zeros.clone()
        bracket_t1 = t.clone()
        bracket_f0 = start_loss.clone()
        bracket_f1 = loss_new.clone()
        bracket_g0 = start_grad.clone(memory_format=torch.contiguous_format)
        bracket_g1 = grad_new.clone(memory_format=torch.contiguous_format)
        bracket_gtd0 = gtd.clone()
        bracket_gtd1 = gtd_new.clone()

        done = false.clone()
        bracketed = false.clone()
        searching = active.clone()
        ls_iter = torch.zeros(batch_size, device=device, dtype=torch.long)

        while bool(searching.any()):
            exhausted = searching & (ls_iter >= max_ls)
            if bool(exhausted.any()):
                bracket_t0 = torch.where(exhausted, zeros, bracket_t0)
                bracket_t1 = torch.where(exhausted, t, bracket_t1)
                bracket_f0 = torch.where(exhausted, start_loss, bracket_f0)
                bracket_f1 = torch.where(exhausted, loss_new, bracket_f1)
                bracket_g0 = torch.where(exhausted[:, None], start_grad, bracket_g0)
                bracket_g1 = torch.where(exhausted[:, None], grad_new, bracket_g1)
                bracket_gtd0 = torch.where(exhausted, gtd, bracket_gtd0)
                bracket_gtd1 = torch.where(exhausted, gtd_new, bracket_gtd1)
                bracketed |= exhausted
                searching = searching & ~exhausted
                if not bool(searching.any()):
                    break

            armijo = start_loss + c1 * t * gtd
            previous_worse = (ls_iter > 1) & (loss_new >= f_prev)
            bracket_now = searching & (
                ~torch.isfinite(loss_new)
                | ~torch.isfinite(gtd_new)
                | (loss_new > armijo)
                | previous_worse
            )
            if bool(bracket_now.any()):
                bracket_t0 = torch.where(bracket_now, t_prev, bracket_t0)
                bracket_t1 = torch.where(bracket_now, t, bracket_t1)
                bracket_f0 = torch.where(bracket_now, f_prev, bracket_f0)
                bracket_f1 = torch.where(bracket_now, loss_new, bracket_f1)
                bracket_g0 = torch.where(bracket_now[:, None], g_prev, bracket_g0)
                bracket_g1 = torch.where(bracket_now[:, None], grad_new, bracket_g1)
                bracket_gtd0 = torch.where(bracket_now, gtd_prev, bracket_gtd0)
                bracket_gtd1 = torch.where(bracket_now, gtd_new, bracket_gtd1)
                bracketed |= bracket_now

            remaining = searching & ~bracket_now
            done_now = remaining & (gtd_new.abs() <= -c2 * gtd)
            if bool(done_now.any()):
                final_alpha = torch.where(done_now, t, final_alpha)
                final_loss = torch.where(done_now, loss_new, final_loss)
                final_grad = torch.where(done_now[:, None], grad_new, final_grad)
                done |= done_now

            remaining = remaining & ~done_now
            bracket_turn = remaining & (gtd_new >= 0)
            if bool(bracket_turn.any()):
                bracket_t0 = torch.where(bracket_turn, t_prev, bracket_t0)
                bracket_t1 = torch.where(bracket_turn, t, bracket_t1)
                bracket_f0 = torch.where(bracket_turn, f_prev, bracket_f0)
                bracket_f1 = torch.where(bracket_turn, loss_new, bracket_f1)
                bracket_g0 = torch.where(bracket_turn[:, None], g_prev, bracket_g0)
                bracket_g1 = torch.where(bracket_turn[:, None], grad_new, bracket_g1)
                bracket_gtd0 = torch.where(bracket_turn, gtd_prev, bracket_gtd0)
                bracket_gtd1 = torch.where(bracket_turn, gtd_new, bracket_gtd1)
                bracketed |= bracket_turn

            searching = remaining & ~bracket_turn
            if not bool(searching.any()) or evals >= remaining_eval_budget:
                break

            min_step = t + 0.01 * (t - t_prev)
            max_step = t * 10.0
            next_t = _cubic_interpolate(
                t_prev,
                f_prev,
                gtd_prev,
                t,
                loss_new,
                gtd_new,
                bounds=(min_step, max_step),
            )

            old_t = t.clone()
            t_prev = torch.where(searching, old_t, t_prev)
            f_prev = torch.where(searching, loss_new, f_prev)
            g_prev = torch.where(searching[:, None], grad_new, g_prev)
            gtd_prev = torch.where(searching, gtd_new, gtd_prev)
            t = torch.where(searching, next_t, t)

            trial_loss, trial_grad, trial_flat = self._evaluate_trial_with_grad(
                closure,
                start_flat=start_flat,
                direction=direction,
                alpha=t,
                evaluate=searching,
                keep_flat=start_flat,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
            )
            evals += 1
            loss_new = torch.where(searching, trial_loss, loss_new)
            grad_new = torch.where(searching[:, None], trial_grad, grad_new)
            trial_direction = self._feasible_direction(
                trial_flat,
                direction,
                lower_bound,
                upper_bound,
            )
            gtd_new = torch.where(
                searching,
                _row_dot(grad_new, trial_direction),
                gtd_new,
            )
            ls_iter = torch.where(searching, ls_iter + 1, ls_iter)

        if bool(searching.any()):
            bracket_t0 = torch.where(searching, zeros, bracket_t0)
            bracket_t1 = torch.where(searching, t, bracket_t1)
            bracket_f0 = torch.where(searching, start_loss, bracket_f0)
            bracket_f1 = torch.where(searching, loss_new, bracket_f1)
            bracket_g0 = torch.where(searching[:, None], start_grad, bracket_g0)
            bracket_g1 = torch.where(searching[:, None], grad_new, bracket_g1)
            bracket_gtd0 = torch.where(searching, gtd, bracket_gtd0)
            bracket_gtd1 = torch.where(searching, gtd_new, bracket_gtd1)
            bracketed |= searching

        low_is_0 = bracket_f0 <= bracket_f1
        low_t = torch.where(low_is_0, bracket_t0, bracket_t1)
        high_t = torch.where(low_is_0, bracket_t1, bracket_t0)
        low_f = torch.where(low_is_0, bracket_f0, bracket_f1)
        high_f = torch.where(low_is_0, bracket_f1, bracket_f0)
        low_g = torch.where(low_is_0[:, None], bracket_g0, bracket_g1)
        high_g = torch.where(low_is_0[:, None], bracket_g1, bracket_g0)
        low_gtd = torch.where(low_is_0, bracket_gtd0, bracket_gtd1)
        high_gtd = torch.where(low_is_0, bracket_gtd1, bracket_gtd0)

        zooming = bracketed & ~done
        insuf_progress = false.clone()
        d_norm = direction.abs().amax(dim=1)

        while bool(zooming.any()) and evals < remaining_eval_budget:
            zooming = zooming & (ls_iter < max_ls)
            if not bool(zooming.any()):
                break
            bracket_width = (high_t - low_t).abs()
            too_small = zooming & (bracket_width * d_norm < tolerance_change)
            zoom_eval = zooming & ~too_small
            if not bool(zoom_eval.any()):
                break

            trial_t = _cubic_interpolate(low_t, low_f, low_gtd, high_t, high_f, high_gtd)

            min_bracket = torch.minimum(low_t, high_t)
            max_bracket = torch.maximum(low_t, high_t)
            eps = 0.1 * (max_bracket - min_bracket)
            near_boundary = zoom_eval & (
                torch.minimum(max_bracket - trial_t, trial_t - min_bracket) < eps
            )
            force_progress = near_boundary & (
                insuf_progress | (trial_t >= max_bracket) | (trial_t <= min_bracket)
            )
            closer_to_max = (trial_t - max_bracket).abs() < (
                trial_t - min_bracket
            ).abs()
            forced_t = torch.where(closer_to_max, max_bracket - eps, min_bracket + eps)
            trial_t = torch.where(force_progress, forced_t, trial_t)
            insuf_progress = torch.where(
                zoom_eval,
                near_boundary & ~force_progress,
                insuf_progress,
            )

            keep_flat = self._project_flat(
                start_flat + low_t[:, None] * direction,
                lower_bound,
                upper_bound,
            )
            trial_loss, trial_grad, trial_flat = self._evaluate_trial_with_grad(
                closure,
                start_flat=start_flat,
                direction=direction,
                alpha=trial_t,
                evaluate=zoom_eval,
                keep_flat=keep_flat,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
            )
            evals += 1
            ls_iter = torch.where(zoom_eval, ls_iter + 1, ls_iter)
            trial_direction = self._feasible_direction(
                trial_flat,
                direction,
                lower_bound,
                upper_bound,
            )
            trial_gtd = _row_dot(trial_grad, trial_direction)

            high_from_trial = zoom_eval & (
                ~torch.isfinite(trial_loss)
                | ~torch.isfinite(trial_gtd)
                | (trial_loss > start_loss + c1 * trial_t * gtd)
                | (trial_loss >= low_f)
            )
            if bool(high_from_trial.any()):
                cand_high_t = torch.where(high_from_trial, trial_t, high_t)
                cand_high_f = torch.where(high_from_trial, trial_loss, high_f)
                cand_high_g = torch.where(high_from_trial[:, None], trial_grad, high_g)
                cand_high_gtd = torch.where(high_from_trial, trial_gtd, high_gtd)
                swap = high_from_trial & (cand_high_f < low_f)
                old_low_t = low_t.clone()
                old_low_f = low_f.clone()
                old_low_g = low_g.clone(memory_format=torch.contiguous_format)
                old_low_gtd = low_gtd.clone()
                low_t = torch.where(swap, cand_high_t, low_t)
                low_f = torch.where(swap, cand_high_f, low_f)
                low_g = torch.where(swap[:, None], cand_high_g, low_g)
                low_gtd = torch.where(swap, cand_high_gtd, low_gtd)
                high_t = torch.where(
                    high_from_trial,
                    torch.where(swap, old_low_t, cand_high_t),
                    high_t,
                )
                high_f = torch.where(
                    high_from_trial,
                    torch.where(swap, old_low_f, cand_high_f),
                    high_f,
                )
                high_g = torch.where(
                    high_from_trial[:, None],
                    torch.where(swap[:, None], old_low_g, cand_high_g),
                    high_g,
                )
                high_gtd = torch.where(
                    high_from_trial,
                    torch.where(swap, old_low_gtd, cand_high_gtd),
                    high_gtd,
                )

            new_low = zoom_eval & ~high_from_trial
            wolfe_done = new_low & (trial_gtd.abs() <= -c2 * gtd)
            turn_high = new_low & ~wolfe_done & (
                trial_gtd * (high_t - low_t) >= 0
            )
            if bool(turn_high.any()):
                high_t = torch.where(turn_high, low_t, high_t)
                high_f = torch.where(turn_high, low_f, high_f)
                high_g = torch.where(turn_high[:, None], low_g, high_g)
                high_gtd = torch.where(turn_high, low_gtd, high_gtd)
            if bool(new_low.any()):
                low_t = torch.where(new_low, trial_t, low_t)
                low_f = torch.where(new_low, trial_loss, low_f)
                low_g = torch.where(new_low[:, None], trial_grad, low_g)
                low_gtd = torch.where(new_low, trial_gtd, low_gtd)

            zooming = zooming & ~too_small & ~wolfe_done

        if bool(bracketed.any()):
            final_alpha = torch.where(bracketed, low_t, final_alpha)
            final_loss = torch.where(bracketed, low_f, final_loss)
            final_grad = torch.where(bracketed[:, None], low_g, final_grad)

        accepted = active & torch.isfinite(final_loss) & (final_alpha > 0)
        final_flat = self._project_flat(
            start_flat + final_alpha[:, None] * direction,
            lower_bound,
            upper_bound,
        )
        final_flat = torch.where(accepted[:, None], final_flat, start_flat)
        final_loss = torch.where(accepted, final_loss, start_loss)
        final_grad = torch.where(accepted[:, None], final_grad, start_grad)
        final_alpha = torch.where(accepted, final_alpha, zeros)
        return final_flat, final_loss, final_grad, final_alpha, accepted, evals


class BatchedLBFGS(
    _BatchedLBFGSHistoryMixin,
    _BatchedLBFGSStrongWolfeMixin,
    _BoxBoundsMixin,
    _VectorClosureMixin,
    Optimizer,
):
    """Limited-memory BFGS with independent state along dimension 0."""

    _optimizer_name = "BatchedLBFGS"
    _bounds_broadcast_to_flat = True

    def __init__(
        self,
        params,
        *,
        lr: float = 1.0,
        max_iter: int = 1,
        max_eval: int | None = None,
        tolerance_grad: float = 1e-7,
        tolerance_change: float = 1e-9,
        history_size: int = 10,
        max_ls: int = 20,
        c1: float = 1e-4,
        c2: float = 0.9,
        shrink: float = 0.5,
        line_search_fn: str = "armijo",
        lower_bound: Bound = None,
        upper_bound: Bound = None,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {max_iter}")
        if max_eval is not None and max_eval < 1:
            raise ValueError(f"max_eval must be >= 1, got {max_eval}")
        if history_size < 1:
            raise ValueError(f"history_size must be >= 1, got {history_size}")
        if max_ls < 1:
            raise ValueError(f"max_ls must be >= 1, got {max_ls}")
        if not (0.0 < c1 < 1.0):
            raise ValueError(f"c1 must be in (0, 1), got {c1}")
        if not (0.0 < c2 < 1.0):
            raise ValueError(f"c2 must be in (0, 1), got {c2}")
        if not (0.0 < shrink < 1.0):
            raise ValueError(f"shrink must be in (0, 1), got {shrink}")
        if line_search_fn not in {"armijo", "strong_wolfe"}:
            raise ValueError("line_search_fn must be 'armijo' or 'strong_wolfe'")

        defaults = {
            "lr": float(lr),
            "max_iter": int(max_iter),
            "max_eval": int(max_eval) if max_eval is not None else None,
            "tolerance_grad": float(tolerance_grad),
            "tolerance_change": float(tolerance_change),
            "history_size": int(history_size),
            "max_ls": int(max_ls),
            "c1": float(c1),
            "c2": float(c2),
            "shrink": float(shrink),
            "line_search_fn": line_search_fn,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
        }
        super().__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError("BatchedLBFGS supports exactly one parameter group")
        self._params = self.param_groups[0]["params"]
        if len(self._params) != 1:
            raise ValueError("BatchedLBFGS supports exactly one parameter tensor")

        param = self._params[0]
        if param.ndim < 1:
            raise ValueError("BatchedLBFGS parameter must have a batch dimension")
        if torch.is_complex(param):
            raise TypeError("BatchedLBFGS only supports real-valued parameters")
        if param.is_sparse:
            raise TypeError("BatchedLBFGS parameter must be dense")
        self._param = param

    def _batch_size(self) -> int:
        return int(self._param.shape[0])

    def _flat_param(self) -> Tensor:
        return self._param.detach().reshape(self._batch_size(), -1)

    def _set_flat_param(self, flat: Tensor) -> None:
        with torch.no_grad():
            self._param.copy_(flat.reshape_as(self._param))

    @torch.no_grad()
    def step(  # type: ignore[override]
        self,
        closure: LossClosure,
        *,
        loss_closure: LossClosure | None = None,
    ) -> Tensor:
        """Perform one batched L-BFGS step and return final per-row losses.

        ``closure`` must zero gradients, compute a loss vector with shape
        ``[B]``, populate the parameter gradient, and return that vector. Most
        callers can use ``loss_vec.sum().backward()``. ``loss_closure`` is
        optional and should return the same vector without gradients; it is used
        for cheaper Armijo probes.
        """
        closure = torch.enable_grad()(closure)
        group = self.param_groups[0]
        lr = float(group["lr"])
        max_iter = int(group["max_iter"])
        max_eval = group["max_eval"]
        tolerance_grad = float(group["tolerance_grad"])
        tolerance_change = float(group["tolerance_change"])
        history_size = int(group["history_size"])
        max_ls = int(group["max_ls"])
        c1 = float(group["c1"])
        c2 = float(group["c2"])
        shrink = float(group["shrink"])
        line_search_fn = group["line_search_fn"]
        lower_bound = group["lower_bound"]
        upper_bound = group["upper_bound"]

        if max_eval is None:
            max_eval = max_iter * (max_ls + 1) + 1
        max_eval = int(max_eval)

        state = self.state[self._param]
        batch_size = self._batch_size()
        device = self._param.device
        dtype = self._param.dtype
        state.setdefault("func_evals", 0)
        state.setdefault("n_iter", 0)
        state.setdefault("old_dirs", [])
        state.setdefault("old_stps", [])
        state.setdefault("ro", [])
        state.setdefault("H_diag", torch.ones(batch_size, device=device, dtype=dtype))

        initial_flat = self._flat_param()
        projected_initial = self._project_flat(initial_flat, lower_bound, upper_bound)
        if not torch.equal(projected_initial, initial_flat):
            self._set_flat_param(projected_initial)

        loss, flat_grad = self._evaluate_with_grad(closure)
        func_evals = 1
        state["func_evals"] += 1
        flat_param = self._flat_param().clone()
        projected_grad = self._projected_gradient(
            flat_param,
            flat_grad,
            lower_bound,
            upper_bound,
        )

        accepted_total = torch.zeros(batch_size, device=device, dtype=torch.bool)
        final_alpha = torch.zeros(batch_size, device=device, dtype=dtype)
        n_iter = 0

        while n_iter < max_iter and func_evals < max_eval:
            n_iter += 1
            state["n_iter"] += 1

            finite_grad = torch.isfinite(flat_grad).all(dim=1)
            finite_loss = torch.isfinite(loss)
            projected_grad = self._projected_gradient(
                flat_param,
                flat_grad,
                lower_bound,
                upper_bound,
            )
            grad_norm = projected_grad.abs().amax(dim=1)
            active = finite_loss & finite_grad & (grad_norm > tolerance_grad)
            if not bool(active.any()):
                break

            direction = self._direction(
                projected_grad,
                state["old_dirs"],
                state["old_stps"],
                state["ro"],
                state["H_diag"],
            )
            direction = self._feasible_direction(
                flat_param,
                direction,
                lower_bound,
                upper_bound,
            )
            direction = torch.where(active[:, None], direction, torch.zeros_like(direction))

            gtd = _row_dot(flat_grad, direction)
            bad_dir = active & (~torch.isfinite(gtd) | (gtd >= -tolerance_change))
            if bool(bad_dir.any()):
                fallback_direction = self._feasible_direction(
                    flat_param,
                    -projected_grad,
                    lower_bound,
                    upper_bound,
                )
                direction = torch.where(bad_dir[:, None], fallback_direction, direction)
                gtd = _row_dot(flat_grad, direction)

            active = active & torch.isfinite(gtd) & (gtd < -tolerance_change)
            if not bool(active.any()):
                break

            if state["n_iter"] == 1:
                grad_l1 = projected_grad.abs().sum(dim=1)
                # Active rows have a nonzero projected gradient (gtd < 0 was required above), so
                # grad_l1 > 0 there; guard only the inactive rows (alpha discarded at the
                # ``where(active, ...)`` below) against 1/0 with the mask -- no epsilon floor.
                safe_l1 = torch.where(active, grad_l1, torch.ones_like(grad_l1))
                alpha = torch.minimum(torch.ones_like(grad_l1), 1.0 / safe_l1) * lr
            else:
                alpha = torch.full((batch_size,), lr, device=device, dtype=dtype)
            alpha = torch.where(active, alpha, torch.zeros_like(alpha))

            start_flat = flat_param.clone()
            start_loss = loss.clone()
            start_grad = flat_grad.clone()
            start_projected_grad = projected_grad.clone()
            accepted = torch.zeros(batch_size, device=device, dtype=torch.bool)

            if line_search_fn == "strong_wolfe":
                (
                    accepted_flat,
                    accepted_loss,
                    accepted_grad,
                    alpha,
                    accepted,
                    ls_evals,
                ) = self._strong_wolfe(
                    closure,
                    start_flat=start_flat,
                    direction=direction,
                    start_loss=start_loss,
                    start_grad=start_grad,
                    gtd=gtd,
                    alpha=alpha,
                    active=active,
                    c1=c1,
                    c2=c2,
                    tolerance_change=tolerance_change,
                    max_ls=max_ls,
                    remaining_eval_budget=max_eval - func_evals,
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                )
                func_evals += ls_evals
                state["func_evals"] += ls_evals
                final_alpha = torch.where(active & accepted, alpha, final_alpha)
                self._set_flat_param(accepted_flat)
                loss = torch.where(active, accepted_loss, start_loss)
                flat_grad = torch.where(active[:, None], accepted_grad, start_grad)
                refreshed_grad = True
            else:
                accepted_flat = start_flat.clone()
                accepted_loss = start_loss.clone()

                searching = active.clone()
                for _ in range(max_ls):
                    if func_evals >= max_eval:
                        break
                    trial_flat = self._project_flat(
                        start_flat + alpha[:, None] * direction,
                        lower_bound,
                        upper_bound,
                    )
                    candidate_flat = torch.where(
                        searching[:, None],
                        trial_flat,
                        accepted_flat,
                    )
                    self._set_flat_param(candidate_flat)
                    trial_loss = self._evaluate_loss(closure, loss_closure)
                    func_evals += 1
                    state["func_evals"] += 1

                    armijo_rhs = start_loss + c1 * alpha * gtd
                    ok = searching & torch.isfinite(trial_loss) & (trial_loss <= armijo_rhs)
                    if bool(ok.any()):
                        accepted = accepted | ok
                        accepted_flat = torch.where(ok[:, None], trial_flat, accepted_flat)
                        accepted_loss = torch.where(ok, trial_loss, accepted_loss)

                    searching = active & ~accepted
                    if not bool(searching.any()):
                        break
                    alpha = torch.where(searching, alpha * shrink, alpha)

                accepted_flat = torch.where(
                    active[:, None] & accepted[:, None],
                    accepted_flat,
                    start_flat,
                )
                final_alpha = torch.where(active & accepted, alpha, final_alpha)
                self._set_flat_param(accepted_flat)
                accepted_loss = torch.where(accepted, accepted_loss, start_loss)

                refreshed_grad = func_evals < max_eval
                if refreshed_grad:
                    loss, flat_grad = self._evaluate_with_grad(closure)
                    func_evals += 1
                    state["func_evals"] += 1
                    loss = torch.where(active, loss, accepted_loss)
                else:
                    loss = torch.where(active, accepted_loss, start_loss)
                    flat_grad = start_grad

            accepted_total = accepted_total | accepted
            new_flat = self._flat_param().clone()
            s_k = new_flat - start_flat
            projected_grad = self._projected_gradient(
                new_flat,
                flat_grad,
                lower_bound,
                upper_bound,
            )
            if refreshed_grad:
                free_s_k = self._feasible_direction(
                    new_flat,
                    s_k,
                    lower_bound,
                    upper_bound,
                )
                y_k = projected_grad - start_projected_grad
                self._append_history(
                    state,
                    free_s_k,
                    y_k,
                    active & accepted,
                    history_size,
                    tolerance_change,
                )

            flat_param = new_flat

            step_norm = s_k.abs().amax(dim=1)
            loss_change = (loss - start_loss).abs()
            still_progressing = active & accepted & (
                (step_norm > tolerance_change) | (loss_change > tolerance_change)
            )
            if not bool(still_progressing.any()):
                break

        state["last_loss"] = loss.detach()
        state["last_grad"] = flat_grad.detach()
        state["last_projected_grad"] = projected_grad.detach()
        state["last_accepted"] = accepted_total.detach()
        state["last_alpha"] = final_alpha.detach()
        state["last_n_iter"] = n_iter
        return loss.detach()


__all__ = ["BatchedLBFGS"]
