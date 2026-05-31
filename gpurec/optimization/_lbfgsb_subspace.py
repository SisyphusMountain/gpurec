"""Internal Cauchy/subspace direction helpers for scalar L-BFGS-B.

This module is private optimization support for ``LBFGSB`` and not a public
import surface.
"""

import torch
from torch import Tensor


class LBFGSBSubspaceMixin:
    """Private generalized-Cauchy and free-subspace methods for ``LBFGSB``."""

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
        if not torch.isfinite(sy) or float(sy.detach().cpu()) <= 0.0:
            return 1.0
        theta = torch.dot(y, y) / sy.clamp_min(torch.finfo(y.dtype).eps)
        return float(theta.detach().cpu())

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
