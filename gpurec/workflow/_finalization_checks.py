from __future__ import annotations

from typing import Any

import torch

from gpurec.api.model import GeneReconModel

from ._evaluation import (
    EvaluationOps,
    _is_memory_retryable_runtime_error,
)
from ._runtime_helpers import (
    _clear_cached_solver_runtime_state,
    _clear_cuda_allocator_cache_if_needed,
    _drop_cached_static_states_if_needed,
    _is_finite_tensor,
    _is_single_value_tensor,
    _tensor_shape,
)
from ._solver_stage import SolverStageController
from .config import RunConfig
from .diagnostics import tensor_stats
from .model_factory import build_alerax_workflow_model


def _evaluate_final_check_genewise_with_memory_fallback(
    config: RunConfig,
    solver: SolverStageController,
    model: GeneReconModel,
    *,
    check_iters: int,
    evaluation: EvaluationOps,
    original_exc: RuntimeError,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    budgets = evaluation._final_eval_fallback_clade_budgets()
    if not budgets:
        raise original_exc
    _clear_cuda_allocator_cache_if_needed(model)
    fallback_errors: list[str] = []
    for budget in budgets:
        fallback_model: GeneReconModel | None = None
        try:
            fallback_data = config.to_dict()
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
            configure_solver = getattr(fallback_model, "configure_solver_iterations", None)
            if callable(configure_solver):
                configure_solver(
                    fixed_iters_E=solver.final_check_fixed_iters_E(check_iters),
                    fixed_iters_Pi=check_iters,
                    neumann_terms=check_iters,
                )
            loss_vec, metrics = evaluation.evaluate_genewise_vector_and_grad(
                fallback_model
            )
            if fallback_model.theta.grad is None:
                raise RuntimeError("fallback final check did not produce gradients")
            grad = fallback_model.theta.grad.detach().to(
                device=model.theta.device,
                dtype=model.theta.dtype,
            )
            metrics = dict(metrics)
            metrics["optimizer/final_check_source"] = "fallback_clade_budget"
            metrics["optimizer/final_check_fallback_clade_budget"] = float(budget)
            metrics["optimizer/final_check_reason"] = (
                f"{type(original_exc).__name__}: {original_exc}"
            )
            metrics["optimizer/final_check_fallback_reason"] = (
                f"{type(original_exc).__name__}: {original_exc}"
            )
            metrics.update(tensor_stats("grad", grad))
            return (
                loss_vec.detach().to(
                    device=model.theta.device,
                    dtype=model.theta.dtype,
                ),
                grad,
                metrics,
            )
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
        "final iteration check failed in the resident layout and all "
        "smaller-clade fallbacks failed; original error: "
        f"{type(original_exc).__name__}: {original_exc}; fallbacks: "
        + "; ".join(fallback_errors)
    ) from original_exc


def _evaluate_final_iteration_check(
    config: RunConfig,
    *,
    solver: SolverStageController,
    evaluation: EvaluationOps,
    model: GeneReconModel,
    baseline_loss: torch.Tensor,
    baseline_grad: torch.Tensor,
    baseline_at_check_iters: bool = False,
) -> dict[str, Any]:
    check_iters = solver.final_iteration_check_iters()
    check_iters_E = solver.final_check_fixed_iters_E(check_iters)
    configure_solver = getattr(model, "configure_solver_iterations", None)
    if not callable(configure_solver):
        return {
            "optimizer/final_check_status": "skipped",
            "optimizer/final_check_source": "not_evaluated",
            "optimizer/final_check_reason": (
                "model_has_no_solver_iteration_controls"
            ),
            "optimizer/final_check_iters": check_iters,
            "optimizer/final_check_iters_E": check_iters_E,
        }
    if check_iters <= 0:
        return {
            "optimizer/final_check_status": "disabled",
            "optimizer/final_check_source": "not_evaluated",
            "optimizer/final_check_reason": "final_check_iters_disabled",
            "optimizer/final_check_iters": 0,
            "optimizer/final_check_iters_E": 0,
        }

    metrics: dict[str, Any] = {
        "optimizer/final_check_status": "failed",
        "optimizer/final_check_source": "configured_solver_budget",
        "optimizer/final_check_iters": check_iters,
        "optimizer/final_check_iters_E": check_iters_E,
        "optimizer/final_check_evals": 1,
    }
    if not _is_single_value_tensor(baseline_loss):
        metrics["optimizer/final_check_reason"] = "baseline_loss_not_scalar"
        return metrics
    if not torch.is_tensor(baseline_grad):
        metrics["optimizer/final_check_reason"] = "baseline_gradient_not_tensor"
        return metrics

    baseline_grad = baseline_grad.detach().clone()
    baseline_grad_shape = _tensor_shape(baseline_grad)
    theta_shape = _tensor_shape(model.theta)
    if baseline_grad_shape != theta_shape:
        metrics["optimizer/final_check_reason"] = (
            "baseline_gradient_shape_mismatch: baseline gradient shape "
            f"{baseline_grad_shape} does not match theta shape {theta_shape}"
        )
        return metrics

    baseline_loss_bits = float(baseline_loss.detach().reshape(()).cpu())
    baseline_grad_inf = (
        float(baseline_grad.detach().abs().amax().cpu())
        if baseline_grad.numel()
        else 0.0
    )
    if baseline_at_check_iters:
        return {
            "optimizer/final_check_status": "baseline",
            "optimizer/final_check_iters": check_iters,
            "optimizer/final_check_iters_E": check_iters_E,
            "optimizer/final_check_evals": 0,
            "optimizer/final_check_loss_abs_delta_bits": 0.0,
            "optimizer/final_check_grad_max_abs_delta": 0.0,
            "optimizer/final_check_baseline_grad_inf": baseline_grad_inf,
            "optimizer/final_check_grad_inf": baseline_grad_inf,
        }

    try:
        _clear_cuda_allocator_cache_if_needed(model)
        _clear_cached_solver_runtime_state(model)
        configure_solver(
            fixed_iters_E=check_iters_E,
            fixed_iters_Pi=check_iters,
            neumann_terms=check_iters,
        )
        if config.mode == "genewise" and callable(
            getattr(model, "full_genewise_nll_and_grad", None)
        ):
            try:
                check_loss_vec, _check_metrics = (
                    evaluation.evaluate_genewise_vector_and_grad(model)
                )
                check_grad = model.theta.grad
            except RuntimeError as check_exc:
                if not _is_memory_retryable_runtime_error(check_exc):
                    raise
                _drop_cached_static_states_if_needed(model)
                configure_solver(
                    fixed_iters_E=check_iters_E,
                    fixed_iters_Pi=check_iters,
                    neumann_terms=check_iters,
                )
                try:
                    check_loss_vec, _check_metrics = (
                        evaluation.evaluate_genewise_vector_and_grad(model)
                    )
                    check_grad = model.theta.grad
                    metrics.update(
                        {
                            "optimizer/final_check_source": (
                                "recomputed_after_cache_drop"
                            ),
                            "optimizer/final_check_reason": (
                                f"{type(check_exc).__name__}: {check_exc}"
                            ),
                            "optimizer/final_check_fallback_reason": (
                                f"{type(check_exc).__name__}: {check_exc}"
                            ),
                        }
                    )
                except RuntimeError as retry_exc:
                    if not _is_memory_retryable_runtime_error(retry_exc):
                        raise
                    check_loss_vec, check_grad, fallback_metrics = (
                        _evaluate_final_check_genewise_with_memory_fallback(
                            config,
                            solver=solver,
                            model=model,
                            check_iters=check_iters,
                            evaluation=evaluation,
                            original_exc=check_exc,
                        )
                    )
                    metrics.update(fallback_metrics)
        else:
            check_loss_vec, _check_metrics = evaluation.evaluate_and_backward(model)
            check_grad = model.theta.grad

        if not _is_single_value_tensor(check_loss_vec.sum()):
            metrics["optimizer/final_check_reason"] = "check_loss_not_scalar"
            return metrics
        check_loss = check_loss_vec.sum().detach().reshape(())
        if check_grad is None:
            metrics["optimizer/final_check_reason"] = "missing_check_gradient"
            return metrics
        if not torch.is_tensor(check_grad):
            metrics["optimizer/final_check_reason"] = "check_gradient_not_tensor"
            return metrics
        check_grad = check_grad.detach()
        if _tensor_shape(check_grad) != baseline_grad_shape:
            metrics["optimizer/final_check_reason"] = (
                "gradient_shape_mismatch: check gradient shape "
                f"{_tensor_shape(check_grad)} does not match baseline gradient "
                f"shape {baseline_grad_shape}"
            )
            return metrics
        check_failed = (
            not torch.isfinite(check_loss).item()
            or not _is_finite_tensor(check_grad)
        )
        if check_failed:
            metrics["optimizer/final_check_reason"] = (
                "nonfinite_objective_or_gradient"
            )
            return metrics

        check_loss_bits = float(check_loss.detach().cpu())
        grad_delta = (check_grad - baseline_grad).detach()
        grad_delta_inf = (
            float(grad_delta.abs().amax().cpu()) if grad_delta.numel() else 0.0
        )
        check_grad_inf = (
            float(check_grad.abs().amax().cpu()) if check_grad.numel() else 0.0
        )
        grad_scale = max(baseline_grad_inf, check_grad_inf, 1.0)
        metrics.update(
            {
                "optimizer/final_check_status": "ok",
                "optimizer/final_check_loss_bits": check_loss_bits,
                "optimizer/final_check_loss_delta_bits": (
                    check_loss_bits - baseline_loss_bits
                ),
                "optimizer/final_check_loss_abs_delta_bits": abs(
                    check_loss_bits - baseline_loss_bits
                ),
                "optimizer/final_check_grad_inf": check_grad_inf,
                "optimizer/final_check_grad_baseline_inf": baseline_grad_inf,
                "optimizer/final_check_grad_max_abs_delta": grad_delta_inf,
                "optimizer/final_check_grad_rel_inf_delta": (
                    grad_delta_inf / grad_scale
                ),
            }
        )
        return metrics
    except Exception as exc:  # pragma: no cover - defensive diagnostic path
        metrics["optimizer/final_check_reason"] = (
            f"{type(exc).__name__}: {exc}"
        )
        return metrics
    finally:
        solver.configure_stage(model, "full")
        model.theta.grad = baseline_grad
        _clear_cached_solver_runtime_state(model)
