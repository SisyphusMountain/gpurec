"""Private workflow config IO and scalar schema helpers.

Public callers should keep importing loader helpers from
``gpurec.workflow.config``. This module owns only JSON/path handling and the
flat ``RunConfig`` scalar schema; it is not a public workflow shortcut.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


_JSON_INT_FIELDS = {
    "start",
    "max_families",
    "clade_budget",
    "max_wave_size",
    "small_family_max_leaves",
    "fixed_iters_e",
    "max_iters_e",
    "fixed_iters_pi",
    "neumann_terms",
    "final_check_iters",
    "solver_warmup_iters",
    "solver_warmup_loss_patience",
    "convergence_check_interval",
    "preprocess_cpu_cores",
    "steps",
    "adam_warmup_steps",
    "fd_adam_warmup_steps",
    "fd_hessian_refresh_steps",
    "hessian_sgd_normal_fixed_iters_pi",
    "hessian_sgd_normal_neumann_terms",
    "hessian_sgd_validation_interval",
    "hessian_sgd_validation_fixed_iters_pi",
    "hessian_sgd_validation_neumann_terms",
    "adagrad_restart_final_check_iters",
    "adagrad_restart_phase_loss_patience",
    "adaptive_rebatch_check_interval",
    "adaptive_rebatch_min_remaining_families",
    "lbfgs_history_size",
    "lbfgs_max_iter",
    "lbfgs_max_ls",
    "lbfgsb_high_kkt_stop_patience",
    "lbfgsb_high_kkt_stop_min_fallbacks",
    "lbfgsb_fallback_max_coordinates",
    "lbfgsb_fallback_max_loss_evals",
    "lbfgsb_best_retry_attempts",
    "loss_patience",
    "best_likelihood_patience",
    "checkpoint_every",
    "log_every",
}
_JSON_FLOAT_FIELDS = {
    "tol_e",
    "e_logsumexp_tol",
    "pi_max_diff_tol",
    "gradient_change_tol",
    "gradient_change_rtol",
    "theta_init_d",
    "theta_init_l",
    "theta_init_t",
    "min_rate",
    "max_rate",
    "lr",
    "lbfgs_lr",
    "lbfgsb_fallback_resolution_competition_factor",
    "fd_hessian_epsilon",
    "fd_newton_damping",
    "adaptive_rebatch_fraction",
    "loss_change_tol",
    "best_likelihood_min_delta",
    "projected_grad_tol",
    "projected_lbfgs_min_lr",
    "pi_fixed_point_relaxation",
}
_JSON_BOOL_FIELDS = {
    "adaptive_iters",
    "adaptive_neumann_terms",
    "adaptive_rebatch",
    "hessian_sgd_pi_adjoint_warmstart",
    "lbfgsb_loss_schedule_force_fallback",
    "loss_stop_projected_grad_gate",
}
_RUN_CONFIG_REQUIRED_PATH_FIELDS = ("species_tree", "families_file", "out_dir")
_RUN_CONFIG_PATH_FIELDS = _RUN_CONFIG_REQUIRED_PATH_FIELDS + (
    "resume_from",
)
_RUN_CONFIG_LEGACY_FIELDS = frozenset(
    {
        "fd_newton_max_step",
        "grad_inf_tol",
        "hessian_sgd_polish_max_steps",
        "hessian_sgd_polish_refresh_steps",
        "hessian_sgd_polish_max_ls",
        "solver_warmup_grad_inf_tol",
    }
)


def _validate_json_scalar_types(data: dict[str, Any]) -> None:
    for name in _JSON_INT_FIELDS:
        if name not in data or data[name] is None:
            continue
        if isinstance(data[name], bool) or not isinstance(data[name], int):
            raise ValueError(f"{name} must be an integer")
    for name in _JSON_FLOAT_FIELDS:
        if name not in data:
            continue
        if isinstance(data[name], bool) or not isinstance(data[name], (int, float)):
            raise ValueError(f"{name} must be a number")
    for name in _JSON_BOOL_FIELDS:
        if name not in data:
            continue
        if not isinstance(data[name], bool):
            raise ValueError(f"{name} must be true or false")
    if (
        "device" in data
        and data["device"] is not None
        and not isinstance(data["device"], str)
    ):
        raise ValueError("device must be a device string")
    if (
        "mode" in data
        and data["mode"] is not None
        and not isinstance(data["mode"], str)
    ):
        raise ValueError("mode must be 'global', 'specieswise', or 'genewise'")
    for name in _RUN_CONFIG_PATH_FIELDS:
        if name not in data or data[name] is None:
            continue
        if not isinstance(data[name], (str, Path)):
            raise ValueError(f"{name} must be a path string")


def _resolve_run_config_path_fields(
    data: dict[str, Any],
    *,
    base_dir: Path,
) -> dict[str, Any]:
    resolved = dict(data)
    for name in _RUN_CONFIG_PATH_FIELDS:
        value = resolved.get(name)
        if value is None or not isinstance(value, (str, Path)):
            continue
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = base_dir / path
        resolved[name] = path.resolve()
    return resolved


def _reject_json_constant(constant: str) -> None:
    raise ValueError(f"invalid JSON numeric constant {constant}")


def load_json_object(path: str | Path, *, description: str = "config") -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        detail = exc.strerror or str(exc)
        raise ValueError(f"could not read {description} {path}: {detail}") from exc
    try:
        data = json.loads(text, parse_constant=_reject_json_constant)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON {description} {path}: {exc.msg}") from exc
    except ValueError as exc:
        raise ValueError(f"invalid JSON {description} {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{description} {path} must contain a JSON object")
    return data


def load_json_object_text(
    text: str,
    *,
    description: str = "config",
) -> dict[str, Any]:
    try:
        data = json.loads(text, parse_constant=_reject_json_constant)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON {description}: {exc.msg}") from exc
    except ValueError as exc:
        raise ValueError(f"invalid JSON {description}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{description} must contain a JSON object")
    return data


def load_run_config_data(path: str | Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    data = load_json_object(path)
    return _resolve_run_config_path_fields(
        data,
        base_dir=path.parent,
    )


def load_run_config_text(
    text: str,
    *,
    base_dir: str | Path,
    description: str = "config",
) -> dict[str, Any]:
    data = load_json_object_text(text, description=description)
    return _resolve_run_config_path_fields(
        data,
        base_dir=Path(base_dir).expanduser().resolve(),
    )
