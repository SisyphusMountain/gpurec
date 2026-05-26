"""PyTorch L-BFGS-B optimizer for one dense bounded parameter tensor."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.optim import Optimizer


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


class LBFGSB(Optimizer):
    """Limited-memory BFGS with box constraints.

    This is a PyTorch implementation of the L-BFGS-B structure described by
    Byrd, Lu, Nocedal, and Zhu: it computes a generalized Cauchy point along
    the projected gradient path, then attempts a reduced free-subspace step.
    The implementation is scoped to gpurec's single dense parameter tensor and
    uses loss-only Armijo probes for line search.
    """

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

    def _bound_for_flat(
        self,
        bound: float | Tensor | None,
        flat: Tensor,
    ) -> Tensor | None:
        if bound is None:
            return None
        if torch.is_tensor(bound):
            bound_tensor = bound.detach().to(device=flat.device, dtype=flat.dtype)
        else:
            bound_tensor = torch.as_tensor(bound, device=flat.device, dtype=flat.dtype)
        if bound_tensor.ndim == 0:
            return torch.full_like(flat, bound_tensor)
        if tuple(bound_tensor.shape) == tuple(flat.shape):
            return bound_tensor
        if tuple(bound_tensor.shape) == tuple(self._param.shape):
            return bound_tensor.reshape_as(flat)
        return torch.broadcast_to(bound_tensor, self._param.shape).reshape_as(flat)

    def _bounds_for_flat(
        self,
        flat: Tensor,
        lower_bound: float | Tensor | None,
        upper_bound: float | Tensor | None,
    ) -> tuple[Tensor | None, Tensor | None]:
        lower = self._bound_for_flat(lower_bound, flat)
        upper = self._bound_for_flat(upper_bound, flat)
        if lower is not None and upper is not None and bool((lower > upper).any()):
            raise ValueError("lower_bound must be <= upper_bound")
        return lower, upper

    def _project_flat(
        self,
        flat: Tensor,
        lower_bound: float | Tensor | None,
        upper_bound: float | Tensor | None,
    ) -> Tensor:
        lower, upper = self._bounds_for_flat(flat, lower_bound, upper_bound)
        projected = flat
        if lower is not None:
            projected = torch.maximum(projected, lower)
        if upper is not None:
            projected = torch.minimum(projected, upper)
        return projected

    def _projected_gradient(
        self,
        flat: Tensor,
        grad: Tensor,
        lower_bound: float | Tensor | None,
        upper_bound: float | Tensor | None,
    ) -> Tensor:
        return flat - self._project_flat(flat - grad, lower_bound, upper_bound)

    def _active_mask(
        self,
        flat: Tensor,
        grad: Tensor,
        lower_bound: float | Tensor | None,
        upper_bound: float | Tensor | None,
        active_tol: float,
    ) -> Tensor:
        lower, upper = self._bounds_for_flat(flat, lower_bound, upper_bound)
        active = torch.zeros_like(flat, dtype=torch.bool)
        if lower is not None:
            active = active | ((flat <= lower + active_tol) & (grad >= 0))
        if upper is not None:
            active = active | ((flat >= upper - active_tol) & (grad <= 0))
        return active

    def _gather_flat_grad(self) -> Tensor:
        grad = self._param.grad
        if grad is None:
            return torch.zeros_like(self._flat_param())
        if grad.is_sparse:
            grad = grad.to_dense()
        if torch.is_complex(grad):
            raise TypeError("LBFGSB only supports real-valued gradients")
        return grad.detach().reshape(-1)

    def _evaluate_with_grad(self, closure: LossClosure) -> tuple[Tensor, Tensor]:
        with torch.enable_grad():
            loss = closure()
        if not torch.is_tensor(loss) or loss.numel() != 1:
            raise ValueError("LBFGSB closure must return a scalar Tensor")
        return loss.detach().reshape(()), self._gather_flat_grad()

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
        if not torch.is_tensor(loss) or loss.numel() != 1:
            raise ValueError("LBFGSB loss closure must return a scalar Tensor")
        return loss.detach().reshape(())

    def _armijo_accepts(
        self,
        *,
        trial_loss: Tensor,
        loss: Tensor,
        trial_gtd: Tensor,
        c1: float,
    ) -> bool:
        if (
            not torch.isfinite(trial_loss)
            or not torch.isfinite(loss)
            or not torch.isfinite(trial_gtd)
        ):
            return False
        trial_value = float(trial_loss.detach().cpu())
        loss_value = float(loss.detach().cpu())
        gtd_value = float(trial_gtd.detach().cpu())
        armijo_value = loss_value + c1 * gtd_value
        threshold = min(armijo_value, math.nextafter(loss_value, -math.inf))
        return trial_value <= threshold

    def _armijo_required_decrease(
        self,
        *,
        loss: Tensor,
        trial_gtd: Tensor,
        c1: float,
    ) -> float:
        loss_value = float(loss.detach().cpu())
        gtd_value = float(trial_gtd.detach().cpu())
        armijo_threshold = loss_value + c1 * gtd_value
        strict_threshold = math.nextafter(loss_value, -math.inf)
        threshold = min(armijo_threshold, strict_threshold)
        return max(0.0, loss_value - threshold)

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

    def _history_matrices(
        self,
        old_dirs: list[Tensor],
        old_stps: list[Tensor],
    ) -> tuple[Tensor | None, Tensor | None]:
        if not old_dirs:
            return None, None
        return torch.stack(old_dirs, dim=1), torch.stack(old_stps, dim=1)

    def _theta(self, old_dirs: list[Tensor], old_stps: list[Tensor]) -> float:
        if not old_dirs:
            return 1.0
        s = old_dirs[-1]
        y = old_stps[-1]
        sy = torch.dot(s, y)
        ss = torch.dot(s, s).clamp_min(torch.finfo(s.dtype).eps)
        if not torch.isfinite(sy) or float(sy.detach().cpu()) <= 0.0:
            return 1.0
        return float((torch.dot(y, y) / sy.clamp_min(torch.finfo(y.dtype).eps)).detach().cpu())

    def _b_matvec(
        self,
        vec: Tensor,
        old_dirs: list[Tensor],
        old_stps: list[Tensor],
        theta: float,
    ) -> Tensor:
        s_mat, y_mat = self._history_matrices(old_dirs, old_stps)
        if s_mat is None or y_mat is None:
            return theta * vec
        m = s_mat.shape[1]
        sy = s_mat.T @ y_mat
        ss = s_mat.T @ s_mat
        lower_l = torch.tril(sy, diagonal=-1)
        diag_d = torch.diag(torch.diag(sy))
        top = torch.cat((theta * ss, lower_l), dim=1)
        bottom = torch.cat((lower_l.T, -diag_d), dim=1)
        k_mat = torch.cat((top, bottom), dim=0)
        w_mat = torch.cat((theta * s_mat, y_mat), dim=1)
        rhs = w_mat.T @ vec
        try:
            sol = torch.linalg.solve(k_mat, rhs)
        except RuntimeError:
            return theta * vec
        return theta * vec - w_mat @ sol

    def _reduced_cg_direction(
        self,
        flat: Tensor,
        grad: Tensor,
        free: Tensor,
        old_dirs: list[Tensor],
        old_stps: list[Tensor],
        theta: float,
        *,
        max_iter: int | None,
        tol: float,
    ) -> Tensor:
        nfree = int(free.sum().detach().cpu())
        if nfree == 0:
            return torch.zeros_like(flat)
        rhs = torch.where(free, -grad, torch.zeros_like(grad))
        x = torch.zeros_like(rhs)
        r = rhs.clone()
        p = r.clone()
        rs_old = torch.dot(r, r)
        if float(rs_old.detach().cpu()) <= 0.0:
            return x
        limit = min(nfree, 50 if max_iter is None else int(max_iter))
        threshold = tol * tol * float(rs_old.detach().cpu())
        for _ in range(limit):
            bp = torch.where(
                free,
                self._b_matvec(p, old_dirs, old_stps, theta),
                torch.zeros_like(p),
            )
            denom = torch.dot(p, bp)
            if (not torch.isfinite(denom)) or float(denom.detach().cpu()) <= 1e-20:
                break
            alpha = rs_old / denom
            x = x + alpha * p
            r = r - alpha * bp
            rs_new = torch.dot(r, r)
            if float(rs_new.detach().cpu()) <= threshold:
                break
            beta = rs_new / rs_old
            p = r + beta * p
            rs_old = rs_new
        return torch.where(free, x, torch.zeros_like(x))

    def _generalized_cauchy_point(
        self,
        flat: Tensor,
        grad: Tensor,
        lower_bound: float | Tensor | None,
        upper_bound: float | Tensor | None,
        old_dirs: list[Tensor],
        old_stps: list[Tensor],
        theta: float,
        active_tol: float,
    ) -> Tensor:
        lower, upper = self._bounds_for_flat(flat, lower_bound, upper_bound)
        direction = -grad.clone()
        if lower is not None:
            at_lower = flat <= lower + active_tol
            direction = torch.where(
                at_lower & (direction < 0),
                torch.zeros_like(direction),
                direction,
            )
        if upper is not None:
            at_upper = flat >= upper - active_tol
            direction = torch.where(
                at_upper & (direction > 0),
                torch.zeros_like(direction),
                direction,
            )
        if not bool((direction != 0).any()):
            return flat.clone()

        inf = torch.tensor(float("inf"), device=flat.device, dtype=flat.dtype)
        breaks = torch.full_like(flat, inf)
        if lower is not None:
            to_lower = direction < 0
            breaks = torch.where(to_lower, (lower - flat) / direction, breaks)
        if upper is not None:
            to_upper = direction > 0
            breaks = torch.where(to_upper, (upper - flat) / direction, breaks)
        breaks = torch.where((breaks > 0) & torch.isfinite(breaks), breaks, inf)
        order = torch.argsort(breaks)

        z = torch.zeros_like(flat)
        d = direction.clone()
        prev_t = torch.zeros((), device=flat.device, dtype=flat.dtype)
        finite_breaks = breaks[order]
        finite_count = int(torch.isfinite(finite_breaks).sum().detach().cpu())

        for position in range(finite_count + 1):
            next_t = (
                finite_breaks[position]
                if position < finite_count
                else torch.tensor(float("inf"), device=flat.device, dtype=flat.dtype)
            )
            bd = self._b_matvec(d, old_dirs, old_stps, theta)
            qprime = torch.dot(grad, d) + torch.dot(z, bd)
            qcurv = torch.dot(d, bd)
            if torch.isfinite(qprime) and float(qprime.detach().cpu()) >= 0.0:
                return self._project_flat(flat + z, lower_bound, upper_bound)
            if (
                torch.isfinite(qcurv)
                and float(qcurv.detach().cpu()) > 1e-20
                and torch.isfinite(qprime)
            ):
                dt_star = -qprime / qcurv
                if float(dt_star.detach().cpu()) >= 0.0:
                    interval = next_t - prev_t
                    if (not torch.isfinite(interval)) or bool(
                        (dt_star <= interval).detach().cpu()
                    ):
                        z = z + dt_star.clamp_min(0.0) * d
                        return self._project_flat(flat + z, lower_bound, upper_bound)
            if position >= finite_count:
                return self._project_flat(flat + z, lower_bound, upper_bound)
            dt = next_t - prev_t
            z = z + dt * d
            idx = int(order[position].detach().cpu())
            d[idx] = 0.0
            prev_t = next_t
        return self._project_flat(flat + z, lower_bound, upper_bound)

    def _candidate_direction(
        self,
        flat: Tensor,
        grad: Tensor,
        lower_bound: float | Tensor | None,
        upper_bound: float | Tensor | None,
        old_dirs: list[Tensor],
        old_stps: list[Tensor],
        *,
        active_tol: float,
        cg_max_iter: int | None,
        cg_tol: float,
    ) -> tuple[Tensor, str]:
        theta = self._theta(old_dirs, old_stps)
        cauchy = self._generalized_cauchy_point(
            flat,
            grad,
            lower_bound,
            upper_bound,
            old_dirs,
            old_stps,
            theta,
            active_tol,
        )
        cauchy_step = cauchy - flat
        active = self._active_mask(cauchy, grad, lower_bound, upper_bound, active_tol)
        free = ~active
        reduced_grad = grad + self._b_matvec(cauchy_step, old_dirs, old_stps, theta)
        subspace_step = self._reduced_cg_direction(
            cauchy,
            reduced_grad,
            free,
            old_dirs,
            old_stps,
            theta,
            max_iter=cg_max_iter,
            tol=cg_tol,
        )
        candidate = self._limit_subspace_step(
            cauchy,
            subspace_step,
            lower_bound,
            upper_bound,
        )
        delta = candidate - flat
        gtd = torch.dot(grad, delta)
        if torch.isfinite(gtd) and float(gtd.detach().cpu()) < -1e-12:
            return delta, "subspace"
        gtd_cauchy = torch.dot(grad, cauchy_step)
        if torch.isfinite(gtd_cauchy) and float(gtd_cauchy.detach().cpu()) < -1e-12:
            return cauchy_step, "cauchy"
        projected_grad = self._projected_gradient(
            flat,
            grad,
            lower_bound,
            upper_bound,
        )
        fallback = -projected_grad
        fallback = self._project_flat(flat + fallback, lower_bound, upper_bound) - flat
        return fallback, "projected_gradient"

    def _limit_subspace_step(
        self,
        flat: Tensor,
        step: Tensor,
        lower_bound: float | Tensor | None,
        upper_bound: float | Tensor | None,
    ) -> Tensor:
        lower, upper = self._bounds_for_flat(flat, lower_bound, upper_bound)
        alpha = torch.ones((), device=flat.device, dtype=flat.dtype)
        if lower is not None:
            to_lower = step < 0
            lower_ratio = (lower - flat) / step
            valid = to_lower & torch.isfinite(lower_ratio) & (lower_ratio >= 0)
            if bool(valid.any().detach().cpu()):
                alpha = torch.minimum(alpha, lower_ratio[valid].amin())
        if upper is not None:
            to_upper = step > 0
            upper_ratio = (upper - flat) / step
            valid = to_upper & torch.isfinite(upper_ratio) & (upper_ratio >= 0)
            if bool(valid.any().detach().cpu()):
                alpha = torch.minimum(alpha, upper_ratio[valid].amin())
        alpha = alpha.clamp(min=0.0, max=1.0)
        return self._project_flat(flat + alpha * step, lower_bound, upper_bound)

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
    ) -> bool:
        return (not search.accepted) or self._tiny_progress(
            flat=flat,
            loss=loss,
            step_inf=search.step_inf,
            decrease=search.decrease,
            tolerance_change=tolerance_change,
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
        ):
            topk_sizes = self._topk_sign_fallback_sizes(int(projected_grad.numel()))
            remaining = remaining_budget()
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
                    max_loss_evals=remaining,
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
