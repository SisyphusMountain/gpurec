from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any

from gpurec._validation import finite_float


def _optional_result_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    try:
        return finite_float("summary value", value)
    except ValueError:
        return None


def _optional_result_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Integral):
        return None
    return int(value)


def _optional_result_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _optional_result_bool(value: object) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    return None


def _optional_result_path(value: object) -> Path | None:
    text = _optional_result_text(value)
    return None if text is None else Path(text)


def _optional_result_text_tuple(value: object) -> tuple[str, ...] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        return None
    return tuple(str(item) for item in value)


def _or_default(value: Any, default: Any) -> Any:
    return default if value is None else value


@dataclass
class OptimizationResult:
    out_dir: Path
    status: str
    reason: str
    final_nll_bits: float
    final_grad_inf: float
    best_nll_bits: float | None
    best_step: int | None
    steps_completed: int
    elapsed_s: float | None = None
    mode: str | None = None
    optimizer: str | None = None
    mode_default_optimizer: str | None = None
    uses_mode_default_optimizer: bool | None = None
    uses_production_default_optimizer_settings: bool | None = None
    production_default_optimizer_setting_mismatches: tuple[str, ...] | None = None
    uses_production_default_route: bool | None = None
    production_default_route_mismatches: tuple[str, ...] | None = None
    families: int | None = None
    species: int | None = None
    batches: int | None = None
    batch_packing: str | None = None
    family_chunk_size: int | None = None
    clade_budget: int | None = None
    fixed_iters_e: int | None = None
    fixed_iters_pi: int | None = None
    neumann_terms: int | None = None
    objective: str | None = None
    gradient_route: str | None = None
    rate_parameterization: str | None = None
    production_default_basis: str | None = None
    configured_steps: int | None = None
    optimizer_step_cap: int | None = None
    optimizer_step_cap_reason: str | None = None
    final_check_iters: int | None = None
    final_check_iters_e: int | None = None
    solver_warmup_iters: int | None = None
    fd_adam_warmup_steps: int | None = None
    fd_hessian_refresh_steps: int | None = None
    hessian_sgd_normal_fixed_iters_pi: int | None = None
    hessian_sgd_normal_neumann_terms: int | None = None
    hessian_sgd_pi_adjoint_warmstart: bool | None = None
    pi_fixed_point_relaxation: float | None = None
    hessian_sgd_validation_interval: int | None = None
    hessian_sgd_validation_fixed_iters_pi: int | None = None
    hessian_sgd_validation_neumann_terms: int | None = None
    adagrad_restart_schedule: str | None = None
    adagrad_restart_total_steps: int | None = None
    adagrad_restart_final_check_iters: int | None = None
    final_projected_grad_inf: float | None = None
    sampling_checkpoint: Path | None = None
    final_log_likelihood_bits: float | None = None
    best_log_likelihood_bits: float | None = None
    final_check_status: str | None = None
    final_check_source: str | None = None
    final_check_reason: str | None = None
    final_check_fallback_clade_budget: float | None = None
    final_check_loss_abs_delta_bits: float | None = None
    final_check_grad_max_abs_delta: float | None = None
    final_check_grad_rel_inf_delta: float | None = None
    final_solver_e_adjoint_failed_batches: int | None = None
    final_solver_e_adjoint_success_batches: int | None = None
    final_solver_e_adjoint_rel_res_max: float | None = None


_RESULT_REQUIRED_FIELDS = {
    "steps_completed": _optional_result_int,
}


_RESULT_FIELDS = {
    "status": _optional_result_text,
    "reason": _optional_result_text,
    "elapsed_s": _optional_result_float,
    "mode": _optional_result_text,
    "optimizer": _optional_result_text,
    "mode_default_optimizer": _optional_result_text,
    "uses_mode_default_optimizer": _optional_result_bool,
    "uses_production_default_optimizer_settings": _optional_result_bool,
    "production_default_optimizer_setting_mismatches": _optional_result_text_tuple,
    "uses_production_default_route": _optional_result_bool,
    "production_default_route_mismatches": _optional_result_text_tuple,
    "families": _optional_result_int,
    "species": _optional_result_int,
    "batches": _optional_result_int,
    "batch_packing": _optional_result_text,
    "family_chunk_size": _optional_result_int,
    "clade_budget": _optional_result_int,
    "fixed_iters_e": _optional_result_int,
    "fixed_iters_pi": _optional_result_int,
    "neumann_terms": _optional_result_int,
    "objective": _optional_result_text,
    "gradient_route": _optional_result_text,
    "rate_parameterization": _optional_result_text,
    "production_default_basis": _optional_result_text,
    "configured_steps": _optional_result_int,
    "optimizer_step_cap": _optional_result_int,
    "optimizer_step_cap_reason": _optional_result_text,
    "final_check_iters": _optional_result_int,
    "final_check_iters_e": _optional_result_int,
    "solver_warmup_iters": _optional_result_int,
    "fd_adam_warmup_steps": _optional_result_int,
    "fd_hessian_refresh_steps": _optional_result_int,
    "hessian_sgd_normal_fixed_iters_pi": _optional_result_int,
    "hessian_sgd_normal_neumann_terms": _optional_result_int,
    "hessian_sgd_pi_adjoint_warmstart": _optional_result_bool,
    "pi_fixed_point_relaxation": _optional_result_float,
    "hessian_sgd_validation_interval": _optional_result_int,
    "hessian_sgd_validation_fixed_iters_pi": _optional_result_int,
    "hessian_sgd_validation_neumann_terms": _optional_result_int,
    "adagrad_restart_schedule": _optional_result_text,
    "adagrad_restart_total_steps": _optional_result_int,
    "adagrad_restart_final_check_iters": _optional_result_int,
    "final_projected_grad_inf": _optional_result_float,
    "sampling_checkpoint": _optional_result_path,
    "final_log_likelihood_bits": _optional_result_float,
    "best_log_likelihood_bits": _optional_result_float,
    "final_check_status": _optional_result_text,
    "final_check_source": _optional_result_text,
    "final_check_reason": _optional_result_text,
    "final_check_fallback_clade_budget": _optional_result_float,
    "final_check_loss_abs_delta_bits": _optional_result_float,
    "final_check_grad_max_abs_delta": _optional_result_float,
    "final_check_grad_rel_inf_delta": _optional_result_float,
    "final_solver_e_adjoint_failed_batches": _optional_result_int,
    "final_solver_e_adjoint_success_batches": _optional_result_int,
    "final_solver_e_adjoint_rel_res_max": _optional_result_float,
    "best_nll_bits": _optional_result_float,
    "best_step": _optional_result_int,
    "final_nll_bits": _optional_result_float,
    "final_grad_inf": _optional_result_float,
}


def optimization_result_from_summary(
    out_dir: Path,
    summary: dict[str, Any],
) -> OptimizationResult:
    result_values: dict[str, Any] = {}
    for field_name, convert in _RESULT_FIELDS.items():
        if field_name in {"status", "reason"}:
            continue
        value = convert(summary.get(field_name))
        if field_name == "final_nll_bits":
            value = _or_default(value, math.nan)
        if field_name == "final_grad_inf":
            value = _or_default(value, math.inf)
        result_values[field_name] = value

    for field_name, convert in _RESULT_REQUIRED_FIELDS.items():
        value = convert(summary.get(field_name))
        if value is None:
            raise ValueError(f"summary {field_name} must be a JSON integer")
        result_values[field_name] = value

    return OptimizationResult(
        out_dir=out_dir,
        status=str(summary.get("status", "")),
        reason=str(summary.get("reason", "")),
        **result_values,
    )
