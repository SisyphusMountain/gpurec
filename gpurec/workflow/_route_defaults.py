"""Internal production route/default policy helpers.

This private module owns route-audit constants and pure route dictionary
checks.  Public callers should keep importing the facade wrappers from
``gpurec.workflow.config``; this module is not a public workflow shortcut.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Any, Callable

from gpurec._validation import finite_float
from gpurec.core.batch_planning import (
    normalize_batch_packing as _normalize_batch_packing,
)


DEFAULT_ADAGRAD_RESTART_SCHEDULE = "8:1.0:60,16:0.5:35,32:0.5:30"
DEFAULT_ADAGRAD_RESTART_FINAL_CHECK_ITERS = 128
DEFAULT_CLADE_BUDGET = 315_000
MODE_DEFAULT_OPTIMIZERS = {
    "genewise": "hessian-sgd",
    "specieswise": "adagrad-restarts",
    "global": "adam",
}
RUN_CONFIG_MODES = frozenset(MODE_DEFAULT_OPTIMIZERS)

_PRODUCTION_DEFAULT_GENEWISE_OPTIMIZER_SETTINGS = {
    "final_check_iters": 32,
    "final_check_iters_e": None,
    "solver_warmup_iters": 4,
    "fd_adam_warmup_steps": 3,
    "fd_hessian_refresh_steps": 16,
    "hessian_sgd_normal_fixed_iters_pi": None,
    "hessian_sgd_normal_neumann_terms": None,
    "hessian_sgd_pi_adjoint_warmstart": False,
    "pi_fixed_point_relaxation": 1.0,
    "hessian_sgd_validation_interval": 0,
    "hessian_sgd_validation_fixed_iters_pi": None,
    "hessian_sgd_validation_neumann_terms": None,
}
_PRODUCTION_DEFAULT_OPTIMIZER_CONFIG_FIELDS = {
    "genewise": tuple(
        name
        for name in _PRODUCTION_DEFAULT_GENEWISE_OPTIMIZER_SETTINGS
        if name != "final_check_iters_e"
    ),
    "specieswise": (
        "adagrad_restart_schedule",
        "adagrad_restart_final_check_iters",
    ),
    "global": (),
}
_PRODUCTION_DEFAULT_ROUTE_CONTRACT = {
    "objective": "negative_log_likelihood_bits",
    "gradient_route": "implicit_first_order_adjoint",
    "rate_parameterization": "base2_log_dlt_rates",
    "production_default_basis": "hogenom_and_test_trees_1000",
}
_PRODUCTION_DEFAULT_BATCH_ROUTE_SETTINGS = {
    "batch_packing": "depth_first_fit",
    "family_chunk_size": 0,
    "clade_budget": DEFAULT_CLADE_BUDGET,
}
_PRODUCTION_ROUTE_STEP_CAP_FIELDS = (
    "configured_steps",
    "optimizer_step_cap",
    "optimizer_step_cap_reason",
)

NormalizeAdagradSchedule = Callable[[str], str]
AdagradTotalSteps = Callable[[str], int]
EffectiveConfigInt = Callable[[Any], int]
EffectiveConfigOptionalInt = Callable[[Any], int | None]
EffectiveStepCap = Callable[[Any], tuple[int, str]]
Jsonable = Callable[[Any], Any]


@dataclass(frozen=True)
class SpecieswiseRouteDefaults:
    normalized_adagrad_restart_schedule: str
    adagrad_restart_total_steps: int
    adagrad_restart_final_check_iters: int = DEFAULT_ADAGRAD_RESTART_FINAL_CHECK_ITERS


def normalize_mode_name(mode: str) -> str:
    if not isinstance(mode, str):
        raise ValueError("mode must be 'global', 'specieswise', or 'genewise'")
    normalized = mode.strip().lower()
    if normalized not in RUN_CONFIG_MODES:
        raise ValueError("mode must be 'global', 'specieswise', or 'genewise'")
    return normalized


def default_optimizer_for_mode(mode: str) -> str:
    return MODE_DEFAULT_OPTIMIZERS[normalize_mode_name(mode)]


def normalize_optimizer_name(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError(
            "optimizer must be auto, adam, adagrad, projected-sgd, lbfgs, "
            "adam-lbfgs, projected-lbfgs, lbfgsb, batched-lbfgs, "
            "adam-fd-newton, hessian-sgd, adagrad-restarts, or "
            "adagrad-restarts-lbfgsb"
        )
    return value.strip().lower().replace("_", "-")


def normalize_optimizer_for_mode(mode: str, value: str) -> str:
    mode = normalize_mode_name(mode)
    normalized = normalize_optimizer_name(value)
    if normalized == "auto":
        return default_optimizer_for_mode(mode)
    return normalized


def production_default_route_contract() -> dict[str, Any]:
    """Return the shipped likelihood/gradient route contract fields."""
    return dict(_PRODUCTION_DEFAULT_ROUTE_CONTRACT)


def production_default_route_contract_fields() -> tuple[str, ...]:
    """Return the required shipped likelihood/gradient route field names."""
    return tuple(_PRODUCTION_DEFAULT_ROUTE_CONTRACT)


def production_default_optimizer_config_overrides(mode: str) -> dict[str, Any]:
    """Return RunConfig overrides for the shipped optimizer profile."""
    mode_text = normalize_mode_name(mode)
    if mode_text == "specieswise":
        settings = {
            "adagrad_restart_schedule": DEFAULT_ADAGRAD_RESTART_SCHEDULE,
            "adagrad_restart_final_check_iters": (
                DEFAULT_ADAGRAD_RESTART_FINAL_CHECK_ITERS
            ),
        }
    else:
        settings = _production_default_optimizer_expected_settings(
            mode_text,
            specieswise_defaults=None,
        )
    return {
        name: settings[name]
        for name in _PRODUCTION_DEFAULT_OPTIMIZER_CONFIG_FIELDS[mode_text]
    }


def production_default_optimizer_setting_mismatches_from_route(
    route: dict[str, Any],
    *,
    specieswise_defaults: SpecieswiseRouteDefaults,
    normalize_adagrad_restart_schedule: NormalizeAdagradSchedule,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return ``(missing, mismatched)`` audit fields for a route dictionary."""
    missing: list[str] = []
    mismatched: list[str] = []
    mode = route.get("mode")
    optimizer = route.get("optimizer")
    if mode is None:
        missing.append("mode")
    if optimizer is None:
        missing.append("optimizer")
    if missing:
        return tuple(missing), tuple(mismatched)
    try:
        mode_text = normalize_mode_name(str(mode))
        mode_default_optimizer = default_optimizer_for_mode(mode_text)
    except ValueError:
        mismatched.append("mode")
        return tuple(missing), tuple(mismatched)
    try:
        if not isinstance(optimizer, str):
            raise ValueError("optimizer must be a string")
        optimizer_text = normalize_optimizer_for_mode(mode_text, optimizer)
    except ValueError:
        mismatched.append("optimizer")
        return tuple(missing), tuple(mismatched)
    if optimizer_text != mode_default_optimizer:
        mismatched.append("optimizer")
        return tuple(missing), tuple(mismatched)
    if mode_text == "global":
        mismatched.append("mode")
        return tuple(missing), tuple(mismatched)
    _append_route_step_cap_evidence_mismatches(
        route,
        mode=mode_text,
        missing=missing,
        mismatched=mismatched,
    )
    expected_settings = _production_default_optimizer_expected_settings(
        mode_text,
        specieswise_defaults=specieswise_defaults,
    )
    for name, expected in expected_settings.items():
        if name not in route:
            if name not in missing:
                missing.append(name)
            continue
        if not _route_setting_matches(
            name,
            route[name],
            expected,
            normalize_adagrad_restart_schedule=normalize_adagrad_restart_schedule,
        ):
            if name not in mismatched:
                mismatched.append(name)
    return tuple(missing), tuple(mismatched)


def production_default_route_mismatches_from_route(
    route: dict[str, Any],
    *,
    specieswise_defaults: SpecieswiseRouteDefaults,
    normalize_adagrad_restart_schedule: NormalizeAdagradSchedule,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return missing/mismatched fields for the shipped route.

    The route includes likelihood/gradient fields, resident batch route fields,
    and optimizer evidence.
    """
    missing: list[str] = []
    mismatched: list[str] = []
    for name, expected in _PRODUCTION_DEFAULT_ROUTE_CONTRACT.items():
        if name not in route:
            missing.append(name)
        elif route[name] != expected:
            mismatched.append(name)
    for (
        batch_name,
        batch_expected,
    ) in _PRODUCTION_DEFAULT_BATCH_ROUTE_SETTINGS.items():
        if batch_name not in route:
            missing.append(batch_name)
        elif not _route_setting_matches(
            batch_name,
            route[batch_name],
            batch_expected,
            normalize_adagrad_restart_schedule=normalize_adagrad_restart_schedule,
        ):
            mismatched.append(batch_name)
    setting_missing, setting_mismatches = (
        production_default_optimizer_setting_mismatches_from_route(
            route,
            specieswise_defaults=specieswise_defaults,
            normalize_adagrad_restart_schedule=normalize_adagrad_restart_schedule,
        )
    )
    missing.extend(setting_missing)
    mismatched.extend(setting_mismatches)
    return tuple(missing), tuple(mismatched)


def production_default_optimizer_setting_mismatches(
    config: Any,
    *,
    effective_optimizer_step_cap: EffectiveStepCap,
    effective_final_check_iters: EffectiveConfigInt,
    effective_final_check_iters_e: EffectiveConfigOptionalInt,
    adagrad_restart_schedule_total_steps: AdagradTotalSteps,
    specieswise_defaults: SpecieswiseRouteDefaults,
    normalize_adagrad_restart_schedule: NormalizeAdagradSchedule,
) -> tuple[str, ...]:
    optimizer_step_cap, optimizer_step_cap_reason = effective_optimizer_step_cap(config)
    route = {
        "mode": config.mode,
        "optimizer": config.optimizer,
        "configured_steps": config.steps,
        "final_check_iters": effective_final_check_iters(config),
        "final_check_iters_e": effective_final_check_iters_e(config),
        "optimizer_step_cap": optimizer_step_cap,
        "optimizer_step_cap_reason": optimizer_step_cap_reason,
        "solver_warmup_iters": config.solver_warmup_iters,
        "fd_adam_warmup_steps": config.fd_adam_warmup_steps,
        "fd_hessian_refresh_steps": config.fd_hessian_refresh_steps,
        "hessian_sgd_normal_fixed_iters_pi": (config.hessian_sgd_normal_fixed_iters_pi),
        "hessian_sgd_normal_neumann_terms": config.hessian_sgd_normal_neumann_terms,
        "hessian_sgd_pi_adjoint_warmstart": (config.hessian_sgd_pi_adjoint_warmstart),
        "pi_fixed_point_relaxation": config.pi_fixed_point_relaxation,
        "hessian_sgd_validation_interval": config.hessian_sgd_validation_interval,
        "hessian_sgd_validation_fixed_iters_pi": (
            config.hessian_sgd_validation_fixed_iters_pi
        ),
        "hessian_sgd_validation_neumann_terms": (
            config.hessian_sgd_validation_neumann_terms
        ),
        "adagrad_restart_schedule": config.adagrad_restart_schedule,
        "adagrad_restart_total_steps": adagrad_restart_schedule_total_steps(
            config.adagrad_restart_schedule
        ),
        "adagrad_restart_final_check_iters": (config.adagrad_restart_final_check_iters),
    }
    missing, mismatches = production_default_optimizer_setting_mismatches_from_route(
        route,
        specieswise_defaults=specieswise_defaults,
        normalize_adagrad_restart_schedule=normalize_adagrad_restart_schedule,
    )
    if missing:
        raise RuntimeError(
            "internal error: complete RunConfig route is missing production "
            f"default optimizer setting field(s): {', '.join(missing)}"
        )
    return mismatches


def effective_route_metadata(
    config: Any,
    *,
    effective_optimizer_step_cap: EffectiveStepCap,
    effective_final_check_iters: EffectiveConfigInt,
    effective_final_check_iters_e: EffectiveConfigOptionalInt,
    adagrad_restart_schedule_total_steps: AdagradTotalSteps,
    specieswise_defaults: SpecieswiseRouteDefaults,
    normalize_adagrad_restart_schedule: NormalizeAdagradSchedule,
    jsonable: Jsonable,
) -> dict[str, Any]:
    optimizer_step_cap, optimizer_step_cap_reason = effective_optimizer_step_cap(config)
    mode_default_optimizer = default_optimizer_for_mode(config.mode)
    default_setting_mismatches = production_default_optimizer_setting_mismatches(
        config,
        effective_optimizer_step_cap=effective_optimizer_step_cap,
        effective_final_check_iters=effective_final_check_iters,
        effective_final_check_iters_e=effective_final_check_iters_e,
        adagrad_restart_schedule_total_steps=adagrad_restart_schedule_total_steps,
        specieswise_defaults=specieswise_defaults,
        normalize_adagrad_restart_schedule=normalize_adagrad_restart_schedule,
    )
    route: dict[str, Any] = {
        **_PRODUCTION_DEFAULT_ROUTE_CONTRACT,
        "mode": config.mode,
        "optimizer": config.optimizer,
        "mode_default_optimizer": mode_default_optimizer,
        "uses_mode_default_optimizer": config.optimizer == mode_default_optimizer,
        "uses_production_default_optimizer_settings": (
            len(default_setting_mismatches) == 0
        ),
        "production_default_optimizer_setting_mismatches": list(
            default_setting_mismatches
        ),
        "batch_packing": config.batch_packing,
        "family_chunk_size": config.family_chunk_size,
        "clade_budget": config.clade_budget,
        "fixed_iters_e": config.fixed_iters_e,
        "fixed_iters_pi": config.fixed_iters_pi,
        "neumann_terms": config.neumann_terms,
        "final_check_iters": effective_final_check_iters(config),
        "final_check_iters_e": effective_final_check_iters_e(config),
        "configured_steps": config.steps,
        "optimizer_step_cap": optimizer_step_cap,
        "optimizer_step_cap_reason": optimizer_step_cap_reason,
    }
    if config.optimizer == "hessian-sgd":
        route.update(
            {
                "solver_warmup_iters": config.solver_warmup_iters,
                "fd_adam_warmup_steps": config.fd_adam_warmup_steps,
                "fd_hessian_refresh_steps": config.fd_hessian_refresh_steps,
                "hessian_sgd_normal_fixed_iters_pi": (
                    config.hessian_sgd_normal_fixed_iters_pi
                ),
                "hessian_sgd_normal_neumann_terms": (
                    config.hessian_sgd_normal_neumann_terms
                ),
                "hessian_sgd_pi_adjoint_warmstart": (
                    config.hessian_sgd_pi_adjoint_warmstart
                ),
                "pi_fixed_point_relaxation": config.pi_fixed_point_relaxation,
                "hessian_sgd_validation_interval": (
                    config.hessian_sgd_validation_interval
                ),
                "hessian_sgd_validation_fixed_iters_pi": (
                    config.hessian_sgd_validation_fixed_iters_pi
                ),
                "hessian_sgd_validation_neumann_terms": (
                    config.hessian_sgd_validation_neumann_terms
                ),
            }
        )
    elif config.optimizer == "adagrad-restarts":
        route.update(
            {
                "adagrad_restart_schedule": config.adagrad_restart_schedule,
                "adagrad_restart_total_steps": (
                    adagrad_restart_schedule_total_steps(
                        config.adagrad_restart_schedule
                    )
                ),
                "adagrad_restart_final_check_iters": (
                    config.adagrad_restart_final_check_iters
                ),
            }
        )
    default_route_missing, default_route_mismatches = (
        production_default_route_mismatches_from_route(
            route,
            specieswise_defaults=specieswise_defaults,
            normalize_adagrad_restart_schedule=normalize_adagrad_restart_schedule,
        )
    )
    if default_route_missing:
        raise RuntimeError(
            "internal error: complete RunConfig route is missing production "
            f"default route field(s): {', '.join(default_route_missing)}"
        )
    route["uses_production_default_route"] = len(default_route_mismatches) == 0
    route["production_default_route_mismatches"] = list(default_route_mismatches)
    return jsonable(route)


def _production_default_optimizer_expected_settings(
    mode: str,
    *,
    specieswise_defaults: SpecieswiseRouteDefaults | None,
) -> dict[str, Any]:
    if mode == "genewise":
        return _PRODUCTION_DEFAULT_GENEWISE_OPTIMIZER_SETTINGS
    if mode == "specieswise":
        if specieswise_defaults is None:
            raise ValueError("specieswise route defaults are required")
        return {
            "final_check_iters": (
                specieswise_defaults.adagrad_restart_final_check_iters
            ),
            "final_check_iters_e": (
                specieswise_defaults.adagrad_restart_final_check_iters
            ),
            "optimizer_step_cap": specieswise_defaults.adagrad_restart_total_steps,
            "optimizer_step_cap_reason": "adagrad_restart_schedule",
            "adagrad_restart_schedule": (
                specieswise_defaults.normalized_adagrad_restart_schedule
            ),
            "adagrad_restart_total_steps": (
                specieswise_defaults.adagrad_restart_total_steps
            ),
            "adagrad_restart_final_check_iters": (
                specieswise_defaults.adagrad_restart_final_check_iters
            ),
        }
    if mode == "global":
        return {}
    raise ValueError("mode must be 'global', 'specieswise', or 'genewise'")


def _route_setting_matches(
    name: str,
    actual: Any,
    expected: Any,
    *,
    normalize_adagrad_restart_schedule: NormalizeAdagradSchedule,
) -> bool:
    if name == "adagrad_restart_schedule" and actual is not None:
        try:
            actual = normalize_adagrad_restart_schedule(str(actual))
        except ValueError:
            return False
    if name == "batch_packing" and actual is not None:
        try:
            actual = _normalize_batch_packing(str(actual))
        except ValueError:
            return False
    if expected is None:
        return actual is None
    if isinstance(expected, bool):
        return isinstance(actual, bool) and actual is expected
    if isinstance(expected, int):
        return (
            not isinstance(actual, bool)
            and isinstance(actual, int)
            and actual == expected
        )
    if isinstance(expected, float):
        if isinstance(actual, bool) or not isinstance(actual, Real):
            return False
        try:
            if isinstance(actual, str):
                actual = float(actual)
            return finite_float(name, actual) == expected
        except ValueError:
            return False
    return actual == expected


def _append_route_step_cap_evidence_mismatches(
    route: dict[str, Any],
    *,
    mode: str,
    missing: list[str],
    mismatched: list[str],
) -> None:
    def append_mismatch(name: str) -> None:
        if name not in mismatched:
            mismatched.append(name)

    def append_missing(name: str) -> None:
        if name not in missing:
            missing.append(name)

    typed_ints: dict[str, int] = {}
    for name in _PRODUCTION_ROUTE_STEP_CAP_FIELDS[:2]:
        if name not in route:
            append_missing(name)
            continue
        value = route[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            append_mismatch(name)
            continue
        typed_ints[name] = int(value)
    reason_name = _PRODUCTION_ROUTE_STEP_CAP_FIELDS[2]
    if reason_name not in route:
        append_missing(reason_name)
        reason: str | None = None
    else:
        reason_value = route[reason_name]
        reason = reason_value if isinstance(reason_value, str) else None
        if reason not in {"configured_steps", "adagrad_restart_schedule"}:
            append_mismatch(reason_name)
    if set(typed_ints) != {"configured_steps", "optimizer_step_cap"} or reason is None:
        return
    configured_steps = typed_ints["configured_steps"]
    optimizer_step_cap = typed_ints["optimizer_step_cap"]
    if reason == "configured_steps":
        if optimizer_step_cap != configured_steps:
            append_mismatch("optimizer_step_cap")
    elif mode != "specieswise":
        append_mismatch(reason_name)
    elif configured_steps < optimizer_step_cap:
        append_mismatch("configured_steps")
