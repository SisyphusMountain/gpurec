from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from gpurec.api.autograd import _discard_pi_adjoint_pending_cache
from gpurec.api.model import GeneReconModel

from .config import RunConfig
from .diagnostics import parameter_stats, solver_stats, tensor_stats
from .model_factory import build_alerax_workflow_model
from ._runtime_helpers import (
    _cached_static_states,
    _clear_cached_solver_runtime_state,
    _clear_cuda_allocator_cache_if_needed,
    _discard_pi_adjoint_pending_caches,
    _drop_cached_static_states_if_needed,
    _is_single_value_tensor,
    _tensor_shape,
)


def _validate_genewise_optimizer_loss_vector(
    model: GeneReconModel,
    loss_vec: object,
    *,
    label: str,
) -> torch.Tensor:
    if not torch.is_tensor(loss_vec):
        raise RuntimeError(f"{label} did not return a tensor loss vector")
    if loss_vec.ndim != 1:
        raise RuntimeError(
            f"{label} returned loss vector with shape {_tensor_shape(loss_vec)}, "
            "expected a one-dimensional tensor"
        )
    expected_family_count = getattr(model, "n_families", None)
    if (
        expected_family_count is not None
        and loss_vec.numel() != int(expected_family_count)
    ):
        raise RuntimeError(
            f"{label} returned {loss_vec.numel()} loss values for "
            f"{int(expected_family_count)} families"
        )
    return loss_vec


def _uses_staged_pi_adjoint_cache(model: GeneReconModel) -> bool:
    return any(
        bool(getattr(static, "pi_adjoint_warmstart", False))
        and getattr(static, "pi_adjoint_cache_update_mode", "immediate") == "stage"
        for static in _cached_static_states(model)
)


def _clear_solver_runtime_state_preserving_pi_cache(model: GeneReconModel) -> None:
    if not _uses_staged_pi_adjoint_cache(model):
        model.clear()
        return
    for static in _cached_static_states(model):
        if hasattr(static, "warm_E"):
            static.warm_E = None
        _discard_pi_adjoint_pending_cache(static)
        if hasattr(static, "last_solver_stats"):
            static.last_solver_stats = None


def _is_memory_retryable_runtime_error(exc: RuntimeError) -> bool:
    if isinstance(exc, torch.OutOfMemoryError):
        return True
    message = str(exc).lower()
    return (
        "out of memory" in message
        or "memory budget" in message
        or "estimated scratch" in message
        or ("scratch" in message and "budget" in message)
    )


@dataclass
class EvaluationOps:
    config: RunConfig

    def evaluate_and_backward(self, model: GeneReconModel) -> tuple[torch.Tensor, dict[str, Any]]:
        model.theta.grad = None
        loss = model.full_loss()
        if not _is_single_value_tensor(loss):
            raise RuntimeError("optimizer evaluation did not return a scalar loss")
        loss = loss.reshape(())
        loss.backward()
        grad = model.theta.grad
        if grad is None:
            raise RuntimeError("optimizer evaluation did not produce theta gradients")
        if not torch.is_tensor(grad):
            raise RuntimeError("optimizer evaluation did not return tensor gradients")
        if _tensor_shape(grad) != _tensor_shape(model.theta):
            raise RuntimeError(
                "optimizer evaluation returned gradient shape "
                f"{_tensor_shape(grad)}, expected theta shape {_tensor_shape(model.theta)}"
            )
        row: dict[str, Any] = {
            "likelihood/data_nll_bits": float(loss.detach().cpu()),
            "likelihood/log_likelihood_bits": float(-loss.detach().cpu()),
        }
        row.update(tensor_stats("grad", grad))
        row.update(parameter_stats(model.theta))
        row.update(solver_stats(model))
        return loss, row

    def evaluate_loss_only(self, model: GeneReconModel) -> torch.Tensor:
        with torch.no_grad():
            full_loss_for_theta = getattr(model, "full_loss_for_theta", None)
            if callable(full_loss_for_theta):
                loss = full_loss_for_theta(model.theta.detach())
            else:
                loss = model.full_loss()
        if not torch.is_tensor(loss) or loss.numel() != 1:
            raise RuntimeError("loss-only optimizer probe did not return a scalar loss")
        return loss.detach().reshape(())

    def evaluate_loss_only_probe(self, model: GeneReconModel) -> torch.Tensor:
        _clear_solver_runtime_state_preserving_pi_cache(model)
        try:
            return self.evaluate_loss_only(model)
        finally:
            _clear_solver_runtime_state_preserving_pi_cache(model)

    def evaluate_genewise_vector_and_grad(
        self,
        model: GeneReconModel,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        model.theta.grad = None
        loss_vec, grad = model.full_genewise_nll_and_grad(need_grad=True)
        loss_vec = _validate_genewise_optimizer_loss_vector(
            model,
            loss_vec,
            label="genewise optimizer evaluation",
        )
        if grad is None:
            raise RuntimeError("genewise optimizer evaluation did not produce gradients")
        if not torch.is_tensor(grad):
            raise RuntimeError(
                "genewise optimizer evaluation did not return tensor gradients"
            )
        grad = grad.detach().to(
            device=model.theta.device,
            dtype=model.theta.dtype,
        )
        if _tensor_shape(grad) != _tensor_shape(model.theta):
            raise RuntimeError(
                "genewise optimizer evaluation returned gradient shape "
                f"{_tensor_shape(grad)}, expected theta shape {_tensor_shape(model.theta)}"
            )
        model.theta.grad = grad
        loss = loss_vec.sum()
        row: dict[str, Any] = {
            "likelihood/data_nll_bits": float(loss.detach().cpu()),
            "likelihood/log_likelihood_bits": float(-loss.detach().cpu()),
        }
        row.update(tensor_stats("grad", model.theta.grad))
        row.update(parameter_stats(model.theta))
        row.update(solver_stats(model))
        return loss_vec.detach(), row

    def evaluate_genewise_loss_vector(self, model: GeneReconModel) -> torch.Tensor:
        loss_vec, _grad = model.full_genewise_nll_and_grad(need_grad=False)
        loss_vec = _validate_genewise_optimizer_loss_vector(
            model,
            loss_vec,
            label="genewise loss-only optimizer probe",
        )
        return loss_vec.detach()

    def _final_eval_fallback_clade_budgets(self) -> list[int]:
        current = self.config.clade_budget
        candidates: list[int] = []
        if current is not None:
            candidates.extend(
                max(1, int(current) // divisor)
                for divisor in (2, 5, 10, 20)
                if int(current) // divisor > 0
            )
        candidates.extend([100_000, 50_000, 25_000])
        seen: set[int] = set()
        out: list[int] = []
        for budget in candidates:
            if current is not None and budget >= int(current):
                continue
            if budget in seen:
                continue
            seen.add(budget)
            out.append(budget)
        return out

    def evaluate_genewise_vector_and_grad_with_memory_fallback(
        self,
        model: GeneReconModel,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        try:
            return self.evaluate_genewise_vector_and_grad(model)
        except RuntimeError as original_exc:
            if not _is_memory_retryable_runtime_error(original_exc):
                raise
            _drop_cached_static_states_if_needed(model)
            try:
                loss_vec, metrics = self.evaluate_genewise_vector_and_grad(model)
                metrics = dict(metrics)
                metrics["optimizer/final_eval_source"] = (
                    "recomputed_after_cache_drop"
                )
                metrics["optimizer/final_eval_fallback_reason"] = (
                    f"{type(original_exc).__name__}: {original_exc}"
                )
                return loss_vec, metrics
            except RuntimeError as retry_exc:
                if not _is_memory_retryable_runtime_error(retry_exc):
                    raise
                _drop_cached_static_states_if_needed(model)
            budgets = self._final_eval_fallback_clade_budgets()
            if not budgets:
                raise
            fallback_errors: list[str] = []
            for budget in budgets:
                fallback_model: GeneReconModel | None = None
                try:
                    fallback_data = self.config.to_dict()
                    fallback_data["clade_budget"] = budget
                    fallback_config = RunConfig.from_dict(fallback_data)
                    fallback_model = build_alerax_workflow_model(
                        fallback_config,
                        prefetch_batches=1,
                    )
                    with torch.no_grad():
                        fallback_model.theta.copy_(
                            model.theta.detach().to(
                                device=fallback_model.theta.device,
                                dtype=fallback_model.theta.dtype,
                            )
                        )
                    loss_vec, metrics = self.evaluate_genewise_vector_and_grad(
                        fallback_model
                    )
                    if fallback_model.theta.grad is None:
                        raise RuntimeError("fallback final eval did not produce gradients")
                    model.theta.grad = fallback_model.theta.grad.detach().to(
                        device=model.theta.device,
                        dtype=model.theta.dtype,
                    )
                    metrics = dict(metrics)
                    metrics["optimizer/final_eval_source"] = (
                        "fallback_clade_budget"
                    )
                    metrics["optimizer/final_eval_fallback_clade_budget"] = float(
                        budget
                    )
                    metrics["optimizer/final_eval_fallback_reason"] = (
                        f"{type(original_exc).__name__}: {original_exc}"
                    )
                    metrics.update(tensor_stats("grad", model.theta.grad))
                    metrics.update(parameter_stats(model.theta))
                    return loss_vec.detach().to(
                        device=model.theta.device,
                        dtype=model.theta.dtype,
                    ), metrics
                except RuntimeError as fallback_exc:
                    if not _is_memory_retryable_runtime_error(fallback_exc):
                        raise
                    fallback_errors.append(
                        f"clade_budget={budget}: "
                        f"{type(fallback_exc).__name__}: {fallback_exc}"
                    )
                finally:
                    if fallback_model is not None:
                        fallback_model.close()
                    _clear_cuda_allocator_cache_if_needed(model)
            raise RuntimeError(
                "final genewise evaluation failed in the resident layout and all "
                "smaller-clade fallbacks failed; original error: "
                f"{type(original_exc).__name__}: {original_exc}; fallbacks: "
                + "; ".join(fallback_errors)
            ) from original_exc

    def _active_batch_indices(self, model: GeneReconModel) -> torch.Tensor:
        indices = getattr(model.current_batch_metadata, "family_indices")
        return torch.as_tensor(
            indices,
            dtype=torch.long,
            device=model.theta.device,
        )

    def evaluate_active_genewise_vector_and_grad(
        self,
        model: GeneReconModel,
        *,
        solver_stage: str,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        model.theta.grad = None
        local_loss_vec = model.nll_per_family()
        local_loss_vec.sum().backward()
        idx = self._active_batch_indices(model)
        self.zero_inactive_batch_grad(model, idx)
        loss_vec = self.full_vector_from_active_batch(model, local_loss_vec)
        return loss_vec, self.active_batch_metrics(
            model,
            loss_vec=loss_vec,
            solver_stage=solver_stage,
        )

    def evaluate_active_genewise_loss_vector(self, model: GeneReconModel) -> torch.Tensor:
        local_loss_vec = model.nll_per_family()
        return self.full_vector_from_active_batch(model, local_loss_vec)

    def evaluate_genewise_loss_vector_probe(
        self,
        model: GeneReconModel,
        *,
        active_batch: bool,
    ) -> torch.Tensor:
        _clear_solver_runtime_state_preserving_pi_cache(model)
        try:
            with torch.no_grad():
                if active_batch:
                    return self.evaluate_active_genewise_loss_vector(model)
                return self.evaluate_genewise_loss_vector(model)
        finally:
            _clear_solver_runtime_state_preserving_pi_cache(model)

    def projected_grad_inf(
        self,
        model: GeneReconModel,
        *,
        lower_bound: float,
        upper_bound: float,
    ) -> tuple[torch.Tensor, float]:
        grad = model.theta.grad
        if grad is None:
            raise RuntimeError("projected gradient requested before gradient evaluation")
        theta = model.theta.detach()
        grad = grad.detach()
        projected = theta - torch.clamp(theta - grad, min=lower_bound, max=upper_bound)
        projected_inf = float(projected.detach().abs().amax().cpu()) if projected.numel() else 0.0
        return projected, projected_inf

    def evaluate_active_genewise_vector_grad_at_current_theta(
        self,
        model: GeneReconModel,
        *,
        solver_stage: str,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        _clear_solver_runtime_state_preserving_pi_cache(model)
        loss_vec, metrics = self.evaluate_active_genewise_vector_and_grad(
            model,
            solver_stage=solver_stage,
        )
        if model.theta.grad is None:
            raise RuntimeError("active genewise evaluation did not produce gradients")
        return loss_vec.detach(), model.theta.grad.detach().clone(), metrics

    def active_batch_metrics(
        self,
        model: GeneReconModel,
        *,
        loss_vec: torch.Tensor,
        solver_stage: str,
    ) -> dict[str, Any]:
        metadata = model.current_batch_metadata
        family_indices = tuple(int(idx) for idx in metadata.family_indices)
        loss = loss_vec.sum()
        row: dict[str, Any] = {
            "likelihood/data_nll_bits": float(loss.detach().cpu()),
            "likelihood/log_likelihood_bits": float(-loss.detach().cpu()),
            "optimizer/objective_scope": "active_batch",
            "optimizer/batch_index": int(model.current_batch_index),
            "optimizer/batch_family_count": int(len(family_indices)),
            "optimizer/solver_stage": solver_stage,
        }
        if family_indices:
            row["optimizer/batch_family_first"] = int(min(family_indices))
            row["optimizer/batch_family_last"] = int(max(family_indices))
        row.update(tensor_stats("grad", model.theta.grad))
        row.update(parameter_stats(model.theta))
        row.update(solver_stats(model))
        return row

    def full_vector_from_active_batch(
        self,
        model: GeneReconModel,
        active_values: torch.Tensor,
    ) -> torch.Tensor:
        idx = self._active_batch_indices(model)
        if not torch.is_tensor(active_values):
            raise RuntimeError(
                "active genewise objective did not return a tensor loss vector"
            )
        expected_shape = (int(idx.numel()),)
        if _tensor_shape(active_values) != expected_shape:
            raise RuntimeError(
                "active genewise objective returned loss vector shape "
                f"{_tensor_shape(active_values)}, expected {expected_shape}"
            )
        values = active_values.detach().to(
            device=model.theta.device,
            dtype=model.theta.dtype,
        )
        full = torch.zeros(
            (int(model.n_families),),
            device=model.theta.device,
            dtype=model.theta.dtype,
        )
        full.index_copy_(0, idx, values)
        return full

    def zero_inactive_batch_grad(
        self,
        model: GeneReconModel,
        idx: torch.Tensor,
    ) -> None:
        grad = model.theta.grad
        if grad is None:
            raise RuntimeError("active genewise optimizer evaluation did not produce gradients")
        mask = torch.zeros(
            (int(model.n_families),),
            device=grad.device,
            dtype=torch.bool,
        )
        mask.index_fill_(0, idx.to(device=grad.device), True)
        grad = grad.detach().clone()
        grad[~mask] = 0
        model.theta.grad = grad
