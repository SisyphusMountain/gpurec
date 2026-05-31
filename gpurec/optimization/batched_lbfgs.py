"""Row-wise batched L-BFGS for independent objectives.

This optimizer is meant for tensors such as genewise DTL parameters where
``theta.shape == [G, ...]`` and row ``g`` only affects objective ``f_g``.
It keeps one L-BFGS history per row while still evaluating all rows in one
batched closure.
"""
from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor
from torch.optim import Optimizer

from ._batched_lbfgs_history import BatchedLBFGSHistoryMixin, _row_dot
from ._batched_lbfgs_strong_wolfe import BatchedLBFGSStrongWolfeMixin
from ._bounds import BoxBoundsMixin
from ._closures import VectorClosureMixin


LossClosure = Callable[[], Tensor]


class BatchedLBFGS(
    BatchedLBFGSHistoryMixin,
    BatchedLBFGSStrongWolfeMixin,
    BoxBoundsMixin,
    VectorClosureMixin,
    Optimizer,
):
    """Limited-memory BFGS with independent state along dimension 0.

    Parameters
    ----------
    params:
        Iterable containing exactly one dense real parameter tensor with shape
        ``[B, ...]``. Row ``i`` is optimized against loss ``loss_vec[i]``.
    lr:
        Initial step-size multiplier.
    max_iter:
        Maximum L-BFGS iterations performed by one :meth:`step` call.
    max_eval:
        Maximum closure evaluations allowed in one :meth:`step` call.  ``None``
        uses the internal budget ``max_iter * (max_ls + 1) + 1``.  Tight budgets
        can stop before a refreshed gradient is available; in that case the
        accepted probed loss is returned with the current parameter state.
    tolerance_grad:
        Stop when every row's flattened gradient infinity norm is at or below
        this threshold.
    tolerance_change:
        Stop when every active row's maximum parameter change is at or below
        this threshold.
    history_size:
        Number of ``s/y`` curvature pairs retained per row.
    max_ls:
        Maximum Armijo backtracking probes per L-BFGS iteration.
    c1:
        Armijo sufficient-decrease constant.
    c2:
        Strong-Wolfe curvature constant. Used only when
        ``line_search_fn="strong_wolfe"``.
    shrink:
        Multiplicative step-size shrink factor for failed Armijo probes.
    line_search_fn:
        ``"armijo"`` for the original masked row-wise Armijo backtracking or
        ``"strong_wolfe"`` for a vectorized port of PyTorch's scalar
        strong-Wolfe bracket/zoom search.
    lower_bound:
        Optional scalar lower bound applied to the parameter values after every
        candidate step. For gpurec rates in log2-space this is
        ``math.log2(min_rate)``.
    upper_bound:
        Optional scalar upper bound applied after every candidate step. For
        ALERax-style DTL rate comparisons in log2-space this is typically
        ``math.log2(2.0)``.

    Notes
    -----
    Bound handling uses L-BFGS-B-style projected gradients: coordinates at a
    lower bound with positive gradient, or at an upper bound with negative
    gradient, are treated as inactive for convergence and search directions.
    """

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
        lower_bound: float | Tensor | None = None,
        upper_bound: float | Tensor | None = None,
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

        p = self._params[0]
        if p.ndim < 1:
            raise ValueError("BatchedLBFGS parameter must have a batch dimension")
        if torch.is_complex(p):
            raise TypeError("BatchedLBFGS only supports real-valued parameters")
        if p.is_sparse:
            raise TypeError("BatchedLBFGS parameter must be dense")
        self._param = p

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
        ``[B]``, populate the parameter gradient, and return the vector. Most
        callers can use ``loss_vec.sum().backward()``; streaming callers may
        assign ``param.grad`` directly. ``loss_closure`` is optional and should
        return the same vector without gradients; it is used for cheaper Armijo
        line-search probes.
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
        B = self._batch_size()
        device = self._param.device
        dtype = self._param.dtype
        state.setdefault("func_evals", 0)
        state.setdefault("n_iter", 0)
        state.setdefault("old_dirs", [])
        state.setdefault("old_stps", [])
        state.setdefault("ro", [])
        state.setdefault("H_diag", torch.ones(B, device=device, dtype=dtype))

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

        accepted_total = torch.zeros(B, device=device, dtype=torch.bool)
        final_alpha = torch.zeros(B, device=device, dtype=dtype)
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

            H_diag = state["H_diag"]
            direction = self._direction(
                projected_grad,
                state["old_dirs"],
                state["old_stps"],
                state["ro"],
                H_diag,
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
                grad_l1 = projected_grad.abs().sum(dim=1).clamp_min(1e-30)
                alpha = torch.minimum(torch.ones_like(grad_l1), 1.0 / grad_l1) * lr
            else:
                alpha = torch.full((B,), lr, device=device, dtype=dtype)
            alpha = torch.where(active, alpha, torch.zeros_like(alpha))

            start_flat = flat_param.clone()
            start_loss = loss.clone()
            start_grad = flat_grad.clone()
            start_projected_grad = projected_grad.clone()
            accepted = torch.zeros(B, device=device, dtype=torch.bool)

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
