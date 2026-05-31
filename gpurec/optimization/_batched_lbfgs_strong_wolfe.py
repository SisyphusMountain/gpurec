"""Internal vectorized strong-Wolfe helpers for batched L-BFGS.

This module is private optimization support for ``BatchedLBFGS`` and not a
public import surface.  Optimizer state, Armijo search, closure budgeting, and
history updates stay in ``batched_lbfgs``.
"""
from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

from ._batched_lbfgs_history import _row_dot
from ._line_search_interpolation import _cubic_interpolate


LossClosure = Callable[[], Tensor]


class BatchedLBFGSStrongWolfeMixin:
    """Private vectorized strong-Wolfe methods for batched L-BFGS."""

    def _evaluate_trial_with_grad(
        self,
        closure: LossClosure,
        *,
        start_flat: Tensor,
        direction: Tensor,
        alpha: Tensor,
        evaluate: Tensor,
        keep_flat: Tensor,
        lower_bound: float | Tensor | None,
        upper_bound: float | Tensor | None,
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
        lower_bound: float | Tensor | None,
        upper_bound: float | Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, int]:
        B = start_flat.shape[0]
        device = start_flat.device
        dtype = start_flat.dtype
        zeros = torch.zeros(B, device=device, dtype=dtype)
        false = torch.zeros(B, device=device, dtype=torch.bool)

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
        ls_iter = torch.zeros(B, device=device, dtype=torch.long)

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
            searching = false.clone()

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

        while (
            bool(zooming.any())
            and evals < remaining_eval_budget
        ):
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
