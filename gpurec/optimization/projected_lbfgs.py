"""Single-objective projected L-BFGS with loss-only Armijo probes."""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
from torch import Tensor
from torch.optim import Optimizer


LossClosure = Callable[[], Tensor]


class ProjectedLBFGS(Optimizer):
    """Limited-memory BFGS for one dense parameter tensor with box projection.

    This is intentionally smaller than a full L-BFGS-B implementation. It uses
    L-BFGS two-loop directions on free coordinates, projects trial points to the
    configured bounds, and tests Armijo decrease with an optional loss-only
    closure. The final accepted point is evaluated once with gradients so the
    workflow can reuse the current loss and gradient.
    """

    def __init__(
        self,
        params,
        *,
        lr: float = 1.0,
        max_iter: int = 1,
        history_size: int = 10,
        max_ls: int = 8,
        c1: float = 1e-4,
        shrink: float = 0.5,
        tolerance_grad: float = 1e-7,
        tolerance_change: float = 1e-9,
        lower_bound: float | Tensor | None = None,
        upper_bound: float | Tensor | None = None,
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
        if tolerance_grad < 0.0:
            raise ValueError("tolerance_grad must be non-negative")
        if tolerance_change < 0.0:
            raise ValueError("tolerance_change must be non-negative")

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
        }
        super().__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError("ProjectedLBFGS supports exactly one parameter group")
        self._params = self.param_groups[0]["params"]
        if len(self._params) != 1:
            raise ValueError("ProjectedLBFGS supports exactly one parameter tensor")
        p = self._params[0]
        if p.ndim < 1:
            raise ValueError("ProjectedLBFGS parameter must have at least one dimension")
        if p.is_sparse:
            raise TypeError("ProjectedLBFGS parameter must be dense")
        if torch.is_complex(p):
            raise TypeError("ProjectedLBFGS only supports real-valued parameters")
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
            return bound_tensor
        if tuple(bound_tensor.shape) == tuple(flat.shape):
            return bound_tensor
        if tuple(bound_tensor.shape) == tuple(self._param.shape):
            return bound_tensor.reshape_as(flat)
        try:
            return torch.broadcast_to(bound_tensor, self._param.shape).reshape_as(flat)
        except RuntimeError:
            return torch.broadcast_to(bound_tensor, flat.shape)

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

    def _feasible_direction(
        self,
        flat: Tensor,
        direction: Tensor,
        lower_bound: float | Tensor | None,
        upper_bound: float | Tensor | None,
    ) -> Tensor:
        lower, upper = self._bounds_for_flat(flat, lower_bound, upper_bound)
        feasible = torch.ones_like(direction, dtype=torch.bool)
        if lower is not None:
            feasible = feasible & ((flat > lower) | (direction >= 0))
        if upper is not None:
            feasible = feasible & ((flat < upper) | (direction <= 0))
        return torch.where(feasible, direction, torch.zeros_like(direction))

    def _gather_flat_grad(self) -> Tensor:
        grad = self._param.grad
        if grad is None:
            return torch.zeros_like(self._flat_param())
        if grad.is_sparse:
            grad = grad.to_dense()
        if torch.is_complex(grad):
            raise TypeError("ProjectedLBFGS only supports real-valued gradients")
        return grad.detach().reshape(-1)

    def _evaluate_with_grad(self, closure: LossClosure) -> tuple[Tensor, Tensor]:
        with torch.enable_grad():
            loss = closure()
        if not torch.is_tensor(loss) or loss.numel() != 1:
            raise ValueError("ProjectedLBFGS closure must return a scalar Tensor")
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
            raise ValueError("ProjectedLBFGS loss closure must return a scalar Tensor")
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

    def _two_loop_direction(
        self,
        projected_grad: Tensor,
        old_dirs: list[Tensor],
        old_stps: list[Tensor],
        ro: list[Tensor],
    ) -> Tensor:
        if not old_dirs:
            return -projected_grad
        q = projected_grad.clone()
        alphas: list[Tensor] = []
        for s, y, rho in zip(reversed(old_dirs), reversed(old_stps), reversed(ro)):
            alpha = rho * torch.dot(s, q)
            q = q - alpha * y
            alphas.append(alpha)
        last_s = old_dirs[-1]
        last_y = old_stps[-1]
        yy = torch.dot(last_y, last_y).clamp_min(torch.finfo(q.dtype).eps)
        gamma = torch.dot(last_s, last_y) / yy
        r = gamma * q
        for s, y, rho, alpha in zip(old_dirs, old_stps, ro, reversed(alphas)):
            beta = rho * torch.dot(y, r)
            r = r + s * (alpha - beta)
        return -r

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
        c1 = float(group["c1"])
        shrink = float(group["shrink"])
        tolerance_grad = float(group["tolerance_grad"])
        tolerance_change = float(group["tolerance_change"])
        lower_bound = group["lower_bound"]
        upper_bound = group["upper_bound"]

        state = self.state[self._param]
        old_dirs = state.setdefault("old_dirs", [])
        old_stps = state.setdefault("old_stps", [])
        ro = state.setdefault("ro", [])

        initial_flat = self._flat_param().clone()
        projected_initial = self._project_flat(initial_flat, lower_bound, upper_bound)
        if not torch.equal(initial_flat, projected_initial):
            self._set_flat_param(projected_initial)

        loss, grad = self._evaluate_with_grad(closure)
        grad_evals = 1
        loss_evals = 0
        flat = self._flat_param().clone()
        projected_grad = self._projected_gradient(
            flat,
            grad,
            lower_bound,
            upper_bound,
        )
        accepted_any = False
        last_alpha = 0.0
        last_step_inf = 0.0
        last_gtd = 0.0

        for _ in range(max_iter):
            projected_grad_inf = (
                float(projected_grad.abs().amax().cpu())
                if projected_grad.numel()
                else 0.0
            )
            if projected_grad_inf <= tolerance_grad:
                break

            direction = self._two_loop_direction(projected_grad, old_dirs, old_stps, ro)
            direction = self._feasible_direction(
                flat,
                direction,
                lower_bound,
                upper_bound,
            )
            gtd = torch.dot(projected_grad, direction)
            if (not torch.isfinite(gtd)) or float(gtd.cpu()) >= -1e-12:
                direction = self._feasible_direction(
                    flat,
                    -projected_grad,
                    lower_bound,
                    upper_bound,
                )
                gtd = torch.dot(projected_grad, direction)
            if (not torch.isfinite(gtd)) or float(gtd.cpu()) >= -1e-12:
                break

            accepted_flat = flat.clone()
            accepted_loss = loss
            accepted_alpha = 0.0
            alpha = lr
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
                trial_gtd = torch.dot(projected_grad, delta)
                if (not torch.isfinite(trial_gtd)) or float(trial_gtd.cpu()) >= -1e-12:
                    alpha *= shrink
                    continue
                self._set_flat_param(trial_flat)
                trial_loss = self._evaluate_loss(closure, loss_closure)
                loss_evals += 1
                if self._armijo_accepts(
                    trial_loss=trial_loss,
                    loss=loss,
                    trial_gtd=trial_gtd,
                    c1=c1,
                    ):
                    accepted_flat = trial_flat.detach().clone()
                    accepted_loss = trial_loss.detach().clone()
                    accepted_alpha = alpha
                    last_gtd = float(trial_gtd.detach().cpu())
                    last_step_inf = step_inf
                    break
                alpha *= shrink
            else:
                self._set_flat_param(flat)
                break

            if accepted_alpha == 0.0:
                self._set_flat_param(flat)
                break

            self._set_flat_param(accepted_flat)
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
            y = new_projected_grad - projected_grad
            ys = torch.dot(y, s)
            finite_update = bool(
                (
                    torch.isfinite(s).all()
                    & torch.isfinite(y).all()
                    & torch.isfinite(ys)
                ).detach().cpu()
            )
            min_curvature = 1e-10 * s.norm().clamp_min(1.0) * y.norm().clamp_min(1.0)
            if finite_update and bool((ys > min_curvature).detach().cpu()):
                old_dirs.append(s.detach().clone())
                old_stps.append(y.detach().clone())
                ro.append((1.0 / ys).detach().clone())
                if len(old_dirs) > history_size:
                    old_dirs.pop(0)
                    old_stps.pop(0)
                    ro.pop(0)

            loss = new_loss.detach() if torch.isfinite(new_loss) else accepted_loss
            grad = new_grad.detach()
            flat = new_flat.detach().clone()
            projected_grad = new_projected_grad.detach()
            accepted_any = True
            last_alpha = accepted_alpha

        if not accepted_any:
            self._set_flat_param(flat)

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
        )
        state["n_iter"] = int(state.get("n_iter", 0)) + int(max_iter)
        return loss
