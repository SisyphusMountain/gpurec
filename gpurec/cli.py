from __future__ import annotations

import argparse
import json
import math
from numbers import Integral, Real
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from gpurec.core.batch_planning import (
    normalize_batch_packing,
    normalize_family_chunk_size,
)


_EXPECTED_WORKFLOW_ERRORS = (ValueError, OSError, RuntimeError)
_RAW_THETA_CHECKPOINT_ERROR = "must contain a dictionary payload"
_CUDA_BACKWARD_MIN_SPECIES_NODES_EXCLUSIVE = 256
_MODE_DEFAULT_OPTIMIZER_HELP = (
    "Fail unless the resolved optimizer matches the mode default optimizer "
    "for the selected mode."
)
_PRODUCTION_DEFAULT_ROUTE_HELP = (
    "Fail unless the objective, likelihood/gradient route, rate parameterization, "
    "resident batch settings, resolved optimizer, final_check_iters_e evidence, "
    "and optimizer-specific settings match the full shipped "
    "HOGENOM/test_trees_1000 likelihood/gradient, resident batch, and optimizer route."
)
_MODE_DEFAULT_OPTIMIZER_CONFIG_ACTION = (
    "use optimizer=auto or the mode default optimizer"
)
_PRODUCTION_DEFAULT_ROUTE_CONFIG_ACTION = (
    "use optimizer=auto and omit route overrides so the shipped "
    "likelihood/gradient, resident batch, and optimizer defaults apply"
)
_MODE_DEFAULT_OPTIMIZER_ARTIFACT_ACTION = "expected the mode default optimizer route"
_PRODUCTION_DEFAULT_ROUTE_ARTIFACT_ACTION = (
    "expected the shipped likelihood/gradient, resident batch, and optimizer route"
)
_ProductionRouteEvidence = tuple[
    dict[str, Any],
    tuple[str, ...],
    tuple[str, ...],
]
_SAFE_STATUS_TEXT_CHARS = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789"
    "._:/+-,"
)


def _run_config_cli_override_fields() -> tuple[str, ...]:
    from dataclasses import fields

    from gpurec.workflow.config import RunConfig

    return tuple(field.name for field in fields(RunConfig))


def _sampling_error_message(exc: BaseException) -> str:
    message = str(exc)
    if _RAW_THETA_CHECKPOINT_ERROR in message:
        return (
            f"{message}; --checkpoint must point to an optimization checkpoint "
            "such as checkpoints/best.pt or checkpoints/latest.pt, not "
            "theta_final.pt"
        )
    return message


def _exit_runtime_error(parser: argparse.ArgumentParser, message: str) -> None:
    parser.exit(status=1, message=f"error: {message}\n")


def _exit_unless_final_check_ok(
    parser: argparse.ArgumentParser,
    status: object,
    *,
    subject: str,
    action: str | None = None,
) -> None:
    if status == "ok":
        return
    message = f"{subject} final check status is {status!r}; expected 'ok'"
    if action is not None:
        message = f"{message}; {action}"
    parser.exit(status=1, message=f"{message}\n")


def _add_require_mode_default_optimizer_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--require-mode-default-optimizer",
        action="store_true",
        help=_MODE_DEFAULT_OPTIMIZER_HELP,
    )


def _add_require_production_default_route_arg(
    parser: argparse.ArgumentParser,
) -> None:
    parser.add_argument(
        "--require-production-default-route",
        action="store_true",
        help=_PRODUCTION_DEFAULT_ROUTE_HELP,
    )


def _route_with_mode_default_audit_fields(route: dict[str, Any]) -> dict[str, Any]:
    audited = dict(route)
    mode = audited.get("mode")
    optimizer = audited.get("optimizer")
    mode_default_optimizer: str | None = None
    optimizer_text: str | None = None
    if mode is not None:
        try:
            from gpurec.workflow.config import (
                default_optimizer_for_mode,
                normalize_mode_name,
                normalize_optimizer_for_mode,
            )

            mode_text = normalize_mode_name(str(mode))
            mode_default_optimizer = default_optimizer_for_mode(mode_text)
            audited["mode"] = mode_text
            if optimizer is not None:
                optimizer_text = normalize_optimizer_for_mode(mode_text, optimizer)
                audited["optimizer"] = optimizer_text
        except _EXPECTED_WORKFLOW_ERRORS:
            pass
    audited["mode_default_optimizer"] = mode_default_optimizer
    if mode is None or optimizer is None or mode_default_optimizer is None:
        audited["uses_mode_default_optimizer"] = None
    elif optimizer_text is None:
        audited["uses_mode_default_optimizer"] = False
    else:
        audited["uses_mode_default_optimizer"] = (
            optimizer_text == mode_default_optimizer
        )
    return audited


def _route_with_production_default_audit_fields(
    route: dict[str, Any],
) -> dict[str, Any]:
    audited, _evidence = _route_with_production_default_audit_evidence(route)
    return audited


def _route_with_production_default_audit_evidence(
    route: dict[str, Any],
) -> tuple[dict[str, Any], _ProductionRouteEvidence]:
    evidence = _production_default_route_evidence(route)
    return _route_with_production_default_evidence_fields(*evidence), evidence


def _route_with_production_default_evidence_fields(
    audited: dict[str, Any],
    missing: tuple[str, ...],
    mismatches: tuple[str, ...],
) -> dict[str, Any]:
    audited = dict(audited)
    if not missing:
        audited["production_default_route_mismatches"] = list(mismatches)
        audited["uses_production_default_route"] = len(mismatches) == 0
    else:
        audited["production_default_route_mismatches"] = None
        audited["uses_production_default_route"] = None
    return audited


def _production_default_optimizer_settings_evidence(
    route: dict[str, Any],
) -> tuple[dict[str, Any], tuple[str, ...], tuple[str, ...]]:
    audited = _route_with_mode_default_audit_fields(route)
    try:
        from gpurec.workflow.config import (
            production_default_optimizer_setting_mismatches_from_route,
        )

        missing, mismatches = (
            production_default_optimizer_setting_mismatches_from_route(audited)
        )
    except _EXPECTED_WORKFLOW_ERRORS:
        return audited, ("mode", "optimizer"), ()
    if not missing:
        audited["production_default_optimizer_setting_mismatches"] = list(mismatches)
        audited["uses_production_default_optimizer_settings"] = (
            len(mismatches) == 0
        )
    else:
        audited["production_default_optimizer_setting_mismatches"] = None
        audited["uses_production_default_optimizer_settings"] = None
    return audited, tuple(missing), tuple(mismatches)


def _production_default_route_evidence(
    route: dict[str, Any],
) -> tuple[dict[str, Any], tuple[str, ...], tuple[str, ...]]:
    audited, _missing, _mismatches = _production_default_optimizer_settings_evidence(
        route
    )
    from gpurec.workflow.config import (
        production_default_route_contract_fields,
        production_default_route_mismatches_from_route,
    )

    try:
        missing, mismatches = production_default_route_mismatches_from_route(audited)
    except _EXPECTED_WORKFLOW_ERRORS:
        return audited, (
            *production_default_route_contract_fields(),
            "mode",
            "optimizer",
        ), ()
    return audited, tuple(missing), tuple(mismatches)


def _mode_default_optimizer_gate_message(
    subject: str,
    route: dict[str, Any],
    *,
    action: str | None = None,
) -> str:
    return _mode_default_optimizer_gate_message_from_audited(
        subject,
        _route_with_mode_default_audit_fields(route),
        action=action,
    )


def _mode_default_optimizer_gate_message_from_audited(
    subject: str,
    audited: dict[str, Any],
    *,
    action: str | None = None,
) -> str:
    missing = [
        name
        for name in ("mode", "optimizer", "mode_default_optimizer")
        if audited.get(name) is None
    ]
    if missing:
        message = (
            f"{subject} mode default optimizer evidence is incomplete; "
            f"missing {', '.join(missing)}"
        )
        if audited.get("mode") is not None or audited.get("optimizer") is not None:
            message = (
                f"{message} (mode={audited.get('mode')!r}, "
                f"optimizer={audited.get('optimizer')!r})"
            )
    else:
        message = (
            f"{subject} optimizer is {audited.get('optimizer')!r}; expected mode "
            f"default {audited.get('mode_default_optimizer')!r} for mode "
            f"{audited.get('mode')!r}"
        )
    if action is not None:
        message = f"{message}; {action}"
    return message


def _production_default_route_gate_message(
    subject: str,
    route: dict[str, Any],
    *,
    action: str | None = None,
) -> str:
    audited, missing, mismatches = _production_default_route_evidence(route)
    return _production_default_route_gate_message_from_evidence(
        subject,
        audited,
        missing,
        mismatches,
        action=action,
    )


def _production_default_route_gate_message_from_evidence(
    subject: str,
    audited: dict[str, Any],
    missing: tuple[str, ...],
    mismatches: tuple[str, ...],
    *,
    action: str | None = None,
) -> str:
    if missing:
        message = (
            f"{subject} production default route evidence is incomplete; "
            f"missing {', '.join(missing)}"
        )
    else:
        mismatch_text = ", ".join(str(item) for item in mismatches) or "none"
        message = (
            f"{subject} production default route fields differ for mode "
            f"{audited.get('mode')!r}: {mismatch_text}"
        )
    if action is not None:
        message = f"{message}; {action}"
    return message


def _require_config_mode_default_optimizer(
    parser: argparse.ArgumentParser,
    config: Any,
    *,
    route: dict[str, Any] | None = None,
) -> None:
    route = _config_route_metadata(config) if route is None else route
    if route.get("uses_mode_default_optimizer") is not True:
        parser.error(
            _mode_default_optimizer_gate_message(
                "config",
                route,
                action=_MODE_DEFAULT_OPTIMIZER_CONFIG_ACTION,
            )
        )


def _require_config_production_default_route(
    parser: argparse.ArgumentParser,
    config: Any,
    *,
    route: dict[str, Any] | None = None,
) -> None:
    route = _config_route_metadata(config) if route is None else route
    if route.get("uses_production_default_route") is not True:
        parser.error(
            _production_default_route_gate_message(
                "config",
                route,
                action=_PRODUCTION_DEFAULT_ROUTE_CONFIG_ACTION,
            )
        )


def _config_route_metadata(config: Any) -> dict[str, Any]:
    from gpurec.workflow.config import effective_route_metadata

    return effective_route_metadata(config)


def _require_config_route_gates(
    parser: argparse.ArgumentParser,
    config: Any,
    *,
    require_mode_default_optimizer: bool,
    require_production_default_route: bool,
) -> dict[str, Any] | None:
    if not require_mode_default_optimizer and not require_production_default_route:
        return None
    route = _config_route_metadata(config)
    if require_mode_default_optimizer:
        _require_config_mode_default_optimizer(parser, config, route=route)
    if require_production_default_route:
        _require_config_production_default_route(parser, config, route=route)
    return route


def _exit_unless_mode_default_optimizer(
    parser: argparse.ArgumentParser,
    route: dict[str, Any],
    *,
    subject: str,
    audited_route: dict[str, Any] | None = None,
) -> None:
    audited = (
        _route_with_mode_default_audit_fields(route)
        if audited_route is None
        else audited_route
    )
    if audited.get("uses_mode_default_optimizer") is True:
        return
    parser.exit(
        status=1,
        message=(
            _mode_default_optimizer_gate_message_from_audited(
                subject,
                audited,
                action=_MODE_DEFAULT_OPTIMIZER_ARTIFACT_ACTION,
            )
            + "\n"
        ),
    )


def _exit_unless_production_default_route(
    parser: argparse.ArgumentParser,
    route: dict[str, Any],
    *,
    subject: str,
    production_route_evidence: _ProductionRouteEvidence | None = None,
) -> None:
    audited, missing, mismatches = (
        _production_default_route_evidence(route)
        if production_route_evidence is None
        else production_route_evidence
    )
    if not missing and not mismatches:
        return
    parser.exit(
        status=1,
        message=(
            _production_default_route_gate_message_from_evidence(
                subject,
                audited,
                missing,
                mismatches,
                action=_PRODUCTION_DEFAULT_ROUTE_ARTIFACT_ACTION,
            )
            + "\n"
        ),
    )


def _optional_metric_text(name: str, value: object) -> str:
    if value is None:
        return f"{name}=null"
    if isinstance(value, bool) or not isinstance(value, Real):
        return f"{name}=null"
    numeric = float(value)
    if math.isnan(numeric):
        return f"{name}=null"
    if math.isinf(numeric):
        return f"{name}={'inf' if numeric > 0.0 else '-inf'}"
    return f"{name}={numeric:.6f}"


def _optional_int_text(name: str, value: object) -> str:
    if value is None:
        return f"{name}=null"
    if isinstance(value, bool) or not isinstance(value, Integral):
        return f"{name}=null"
    return f"{name}={int(value)}"


def _optional_text(name: str, value: object) -> str:
    if value is None:
        return f"{name}=null"
    text = str(value)
    if not text:
        return f"{name}=null"
    if any(char not in _SAFE_STATUS_TEXT_CHARS for char in text):
        text = json.dumps(text, ensure_ascii=True).replace(" ", "\\u0020")
    return f"{name}={text}"


def _optional_bool_text(name: str, value: object) -> str:
    if value is None:
        return f"{name}=null"
    if isinstance(value, bool):
        return f"{name}={'true' if value else 'false'}"
    return f"{name}=null"


def _optional_list_text(name: str, value: object, *, empty_text: str = "none") -> str:
    if value is None:
        return f"{name}=null"
    if not isinstance(value, (list, tuple)):
        return f"{name}=null"
    if not value:
        return f"{name}={empty_text}"
    return _optional_text(name, ",".join(str(item) for item in value))


def _log_likelihood_from_result(
    result: Any,
    *,
    nll_attr: str,
    log_likelihood_attr: str,
    final_metric: bool = False,
) -> float | None:
    explicit = getattr(result, log_likelihood_attr, None)
    if explicit is not None:
        if isinstance(explicit, bool) or not isinstance(explicit, Real):
            return None
        return float(explicit)
    if final_metric and getattr(result, "status", None) == "failed":
        return None
    nll_value = getattr(result, nll_attr, None)
    if nll_value is None:
        return None
    if isinstance(nll_value, bool) or not isinstance(nll_value, Real):
        return None
    nll = float(nll_value)
    if not math.isfinite(nll):
        return None
    return -nll


def _optimization_result_text(result: Any) -> str:
    final_log_likelihood = _log_likelihood_from_result(
        result,
        nll_attr="final_nll_bits",
        log_likelihood_attr="final_log_likelihood_bits",
        final_metric=True,
    )
    best_log_likelihood = _log_likelihood_from_result(
        result,
        nll_attr="best_nll_bits",
        log_likelihood_attr="best_log_likelihood_bits",
    )
    return " ".join(
        [
            _optional_text("status", getattr(result, "status", None)),
            _optional_text("reason", getattr(result, "reason", None)),
            _optional_text("mode", getattr(result, "mode", None)),
            _optional_text("optimizer", getattr(result, "optimizer", None)),
            _optional_text(
                "mode_default_optimizer",
                getattr(result, "mode_default_optimizer", None),
            ),
            _optional_bool_text(
                "uses_mode_default_optimizer",
                getattr(result, "uses_mode_default_optimizer", None),
            ),
            _optional_bool_text(
                "uses_production_default_optimizer_settings",
                getattr(
                    result,
                    "uses_production_default_optimizer_settings",
                    None,
                ),
            ),
            _optional_list_text(
                "production_default_optimizer_setting_mismatches",
                getattr(
                    result,
                    "production_default_optimizer_setting_mismatches",
                    None,
                ),
            ),
            _optional_bool_text(
                "uses_production_default_route",
                getattr(result, "uses_production_default_route", None),
            ),
            _optional_list_text(
                "production_default_route_mismatches",
                getattr(result, "production_default_route_mismatches", None),
            ),
            _optional_int_text("families", getattr(result, "families", None)),
            _optional_int_text("species", getattr(result, "species", None)),
            _optional_int_text("batches", getattr(result, "batches", None)),
            _optional_text("batch_packing", getattr(result, "batch_packing", None)),
            _optional_int_text(
                "family_chunk_size",
                getattr(result, "family_chunk_size", None),
            ),
            _optional_int_text(
                "clade_budget",
                getattr(result, "clade_budget", None),
            ),
            _optional_int_text(
                "fixed_iters_e",
                getattr(result, "fixed_iters_e", None),
            ),
            _optional_int_text(
                "fixed_iters_pi",
                getattr(result, "fixed_iters_pi", None),
            ),
            _optional_int_text(
                "neumann_terms",
                getattr(result, "neumann_terms", None),
            ),
            _optional_text("objective", getattr(result, "objective", None)),
            _optional_text(
                "gradient_route",
                getattr(result, "gradient_route", None),
            ),
            _optional_text(
                "rate_parameterization",
                getattr(result, "rate_parameterization", None),
            ),
            _optional_text(
                "production_default_basis",
                getattr(result, "production_default_basis", None),
            ),
            _optional_int_text(
                "configured_steps",
                getattr(result, "configured_steps", None),
            ),
            _optional_int_text(
                "optimizer_step_cap",
                getattr(result, "optimizer_step_cap", None),
            ),
            _optional_text(
                "optimizer_step_cap_reason",
                getattr(result, "optimizer_step_cap_reason", None),
            ),
            _optional_int_text(
                "final_check_iters",
                getattr(result, "final_check_iters", None),
            ),
            _optional_int_text(
                "final_check_iters_e",
                getattr(result, "final_check_iters_e", None),
            ),
            *(
                [
                    _optional_int_text(
                        "solver_warmup_iters",
                        getattr(result, "solver_warmup_iters", None),
                    ),
                    _optional_int_text(
                        "fd_adam_warmup_steps",
                        getattr(result, "fd_adam_warmup_steps", None),
                    ),
                    _optional_int_text(
                        "fd_hessian_refresh_steps",
                        getattr(result, "fd_hessian_refresh_steps", None),
                    ),
                    _optional_int_text(
                        "hessian_sgd_normal_fixed_iters_pi",
                        getattr(
                            result,
                            "hessian_sgd_normal_fixed_iters_pi",
                            None,
                        ),
                    ),
                    _optional_int_text(
                        "hessian_sgd_normal_neumann_terms",
                        getattr(
                            result,
                            "hessian_sgd_normal_neumann_terms",
                            None,
                        ),
                    ),
                    _optional_bool_text(
                        "hessian_sgd_pi_adjoint_warmstart",
                        getattr(
                            result,
                            "hessian_sgd_pi_adjoint_warmstart",
                            None,
                        ),
                    ),
                    _optional_metric_text(
                        "pi_fixed_point_relaxation",
                        getattr(result, "pi_fixed_point_relaxation", None),
                    ),
                    _optional_int_text(
                        "hessian_sgd_validation_interval",
                        getattr(result, "hessian_sgd_validation_interval", None),
                    ),
                    _optional_int_text(
                        "hessian_sgd_validation_fixed_iters_pi",
                        getattr(
                            result,
                            "hessian_sgd_validation_fixed_iters_pi",
                            None,
                        ),
                    ),
                    _optional_int_text(
                        "hessian_sgd_validation_neumann_terms",
                        getattr(
                            result,
                            "hessian_sgd_validation_neumann_terms",
                            None,
                        ),
                    ),
                ]
                if getattr(result, "optimizer", None) == "hessian-sgd"
                else []
            ),
            *(
                [
                    _optional_text(
                        "adagrad_restart_schedule",
                        getattr(result, "adagrad_restart_schedule", None),
                    ),
                    _optional_int_text(
                        "adagrad_restart_total_steps",
                        getattr(result, "adagrad_restart_total_steps", None),
                    ),
                    _optional_int_text(
                        "adagrad_restart_final_check_iters",
                        getattr(
                            result,
                            "adagrad_restart_final_check_iters",
                            None,
                        ),
                    ),
                ]
                if getattr(result, "optimizer", None) == "adagrad-restarts"
                else []
            ),
            _optional_int_text(
                "steps_completed",
                getattr(result, "steps_completed", None),
            ),
            _optional_metric_text(
                "elapsed_s",
                getattr(result, "elapsed_s", None),
            ),
            _optional_int_text(
                "best_step",
                getattr(result, "best_step", None),
            ),
            _optional_text(
                "sampling_checkpoint",
                getattr(result, "sampling_checkpoint", None),
            ),
            _optional_metric_text(
                "final_nll_bits",
                getattr(result, "final_nll_bits", None),
            ),
            _optional_metric_text(
                "final_log_likelihood_bits",
                final_log_likelihood,
            ),
            _optional_metric_text(
                "final_grad_inf",
                getattr(result, "final_grad_inf", None),
            ),
            _optional_metric_text(
                "final_projected_grad_inf",
                getattr(result, "final_projected_grad_inf", None),
            ),
            _optional_metric_text(
                "best_nll_bits",
                getattr(result, "best_nll_bits", None),
            ),
            _optional_metric_text(
                "best_log_likelihood_bits",
                best_log_likelihood,
            ),
            _optional_text(
                "final_check_status",
                getattr(result, "final_check_status", None),
            ),
            _optional_text(
                "final_check_source",
                getattr(result, "final_check_source", None),
            ),
            _optional_text(
                "final_check_reason",
                getattr(result, "final_check_reason", None),
            ),
            _optional_metric_text(
                "final_check_fallback_clade_budget",
                getattr(result, "final_check_fallback_clade_budget", None),
            ),
            _optional_metric_text(
                "final_check_loss_abs_delta_bits",
                getattr(result, "final_check_loss_abs_delta_bits", None),
            ),
            _optional_metric_text(
                "final_check_grad_max_abs_delta",
                getattr(result, "final_check_grad_max_abs_delta", None),
            ),
            _optional_metric_text(
                "final_check_grad_rel_inf_delta",
                getattr(result, "final_check_grad_rel_inf_delta", None),
            ),
            _optional_int_text(
                "final_solver_e_adjoint_failed_batches",
                getattr(result, "final_solver_e_adjoint_failed_batches", None),
            ),
            _optional_int_text(
                "final_solver_e_adjoint_success_batches",
                getattr(result, "final_solver_e_adjoint_success_batches", None),
            ),
            _optional_metric_text(
                "final_solver_e_adjoint_rel_res_max",
                getattr(result, "final_solver_e_adjoint_rel_res_max", None),
            ),
        ]
    )


def _summary_info_text(
    summary: Path,
    payload: dict[str, Any],
    *,
    audited_payload: dict[str, Any] | None = None,
) -> str:
    payload = (
        _route_with_production_default_audit_fields(payload)
        if audited_payload is None
        else audited_payload
    )
    return (
        f"{_optional_text('summary', summary)} "
        f"{_optimization_result_text(SimpleNamespace(**payload))}"
    )


def optimize(config: Any) -> Any:
    from gpurec.workflow.optimize import optimize as _optimize

    return _optimize(config)


def sample(config: Any) -> Any:
    from gpurec.workflow.sampling import sample as _sample

    return _sample(config)


def _chunk_size(value: str) -> int:
    try:
        return int(normalize_family_chunk_size(value))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _dtype_name(value: str) -> str:
    from gpurec.workflow.config import dtype_name_from_name

    try:
        return dtype_name_from_name(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _mode_name(value: str) -> str:
    from gpurec.workflow.config import normalize_mode_name

    try:
        return normalize_mode_name(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _optimizer_name(value: str) -> str:
    from gpurec.workflow.config import normalize_optimizer_name

    try:
        return normalize_optimizer_name(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _batch_packing(value: str) -> str:
    try:
        return normalize_batch_packing(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _config_data(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    if path.suffix.lower() in {".yaml", ".yml"}:
        raise ValueError(
            "--config currently expects a flat JSON RunConfig file; "
            "Hydra-style YAML configs must be converted to JSON or passed as "
            "explicit CLI flags"
        )
    from gpurec.workflow.config import load_run_config_data

    return load_run_config_data(path)


def _set_if_present(data: dict[str, Any], args: argparse.Namespace, name: str) -> None:
    value = getattr(args, name, None)
    if value is not None:
        data[name] = value


def _validate_run_config_input_paths(config: RunConfig) -> None:
    for option, path in (
        ("--species-tree", config.species_tree),
        ("--families-file", config.families_file),
    ):
        if not path.is_file():
            raise ValueError(f"{option} path does not exist or is not a file: {path}")
    if config.resume_from is not None and not config.resume_from.is_file():
        raise ValueError(
            "--resume-from path does not exist or is not a file: "
            f"{config.resume_from}"
        )


def _validate_run_config_family_references(config: RunConfig) -> dict[str, int]:
    from gpurec.core.model import parse_alerax_family_file

    family_names, tree_paths, leaf_species_maps = parse_alerax_family_file(
        config.families_file,
        start=config.start,
        max_families=config.max_families,
    )
    missing: list[tuple[str, Path]] = []
    gene_tree_files = 0
    for family, paths in zip(family_names, tree_paths):
        for raw_path in paths:
            gene_tree_files += 1
            path = Path(raw_path)
            if not path.is_file():
                missing.append((family, path))
    if missing:
        preview = "; ".join(
            f"{family}: {path}" for family, path in missing[:5]
        )
        suffix = "" if len(missing) <= 5 else f"; ... {len(missing) - 5} more"
        raise ValueError(
            "AleRax family file references missing gene-tree path(s): "
            f"{preview}{suffix}"
        )
    return {
        "families": len(family_names),
        "gene_tree_files": gene_tree_files,
        "mapped_families": sum(1 for mapping in leaf_species_maps if mapping),
    }


def _validate_run_config_preprocess(config: RunConfig) -> dict[str, int]:
    import torch

    from gpurec.core.model import GeneDataset, parse_alerax_family_file

    family_names, tree_paths, leaf_species_maps = parse_alerax_family_file(
        config.families_file,
        start=config.start,
        max_families=config.max_families,
    )
    dataset = GeneDataset(
        config.species_tree,
        tree_paths,
        genewise=config.mode == "genewise",
        specieswise=config.mode == "specieswise",
        dtype=torch.float32,
        device="cpu",
        preprocess_cpu_cores=config.preprocess_cpu_cores,
        family_names=family_names,
        leaf_species_maps=leaf_species_maps,
    )
    return {
        "preprocessed_families": int(dataset.num_families),
        "preprocessed_species_nodes": int(dataset.S),
    }


def _cuda_backward_readiness(species_nodes: int) -> dict[str, object]:
    ready = species_nodes > _CUDA_BACKWARD_MIN_SPECIES_NODES_EXCLUSIVE
    return {
        "cuda_backward_ready": ready,
        "cuda_backward_ready_reason": None if ready else "requires_s_gt_256",
    }


def _validate_sampling_checkpoint_path(checkpoint: Path) -> None:
    path = checkpoint.expanduser().resolve()
    if not path.is_file():
        raise ValueError(
            f"--checkpoint path does not exist or is not a file: {path}"
        )


def _validate_summary_path(summary: Path) -> None:
    path = summary.expanduser().resolve()
    if not path.is_file():
        raise ValueError(
            f"--summary path does not exist or is not a file: {path}"
        )


def _partial_route_metadata_from_config_data(
    config_data: dict[str, Any],
) -> dict[str, Any]:
    route, _evidence = _partial_route_metadata_evidence_from_config_data(config_data)
    return route


def _partial_route_metadata_evidence_from_config_data(
    config_data: dict[str, Any],
) -> tuple[dict[str, Any], _ProductionRouteEvidence]:
    route = {
        key: config_data[key]
        for key in ("mode", "optimizer")
        if config_data.get(key) is not None
    }
    return _route_with_production_default_audit_evidence(route)


def _checkpoint_route_metadata(payload: dict[str, Any]) -> tuple[dict[str, Any], str]:
    route, route_source, _evidence = _checkpoint_route_metadata_evidence(payload)
    return route, route_source


def _checkpoint_route_metadata_evidence(
    payload: dict[str, Any],
) -> tuple[dict[str, Any], str, _ProductionRouteEvidence | None]:
    route = payload.get("route_metadata")
    if isinstance(route, dict):
        audited, evidence = _route_with_production_default_audit_evidence(route)
        return audited, "checkpoint", evidence
    config_data = payload.get("config")
    if not isinstance(config_data, dict):
        return {}, "missing", None
    try:
        from gpurec.workflow.config import RunConfig, effective_route_metadata

        audited, evidence = _route_with_production_default_audit_evidence(
            effective_route_metadata(RunConfig.from_dict(config_data))
        )
        return audited, "config", evidence
    except _EXPECTED_WORKFLOW_ERRORS:
        partial_route, partial_evidence = (
            _partial_route_metadata_evidence_from_config_data(config_data)
        )
        if partial_route:
            return partial_route, "config", partial_evidence
        return {}, "missing", None


def _route_int_text(name: str, route: dict[str, Any], *, none_text: str = "null") -> str:
    value = route.get(name)
    if value is None:
        return f"{name}={none_text}"
    return _optional_int_text(name, value)


def _route_metadata_text(
    route: dict[str, Any],
    *,
    production_route_evidence: _ProductionRouteEvidence | None = None,
) -> str:
    route = (
        _route_with_production_default_audit_fields(route)
        if production_route_evidence is None
        else _route_with_production_default_evidence_fields(
            production_route_evidence[0],
            production_route_evidence[1],
            production_route_evidence[2],
        )
    )
    fields = [
        _optional_text("objective", route.get("objective")),
        _optional_text("gradient_route", route.get("gradient_route")),
        _optional_text(
            "rate_parameterization",
            route.get("rate_parameterization"),
        ),
        _optional_text(
            "production_default_basis",
            route.get("production_default_basis"),
        ),
        _optional_text(
            "mode_default_optimizer",
            route.get("mode_default_optimizer"),
        ),
        _optional_bool_text(
            "uses_mode_default_optimizer",
            route.get("uses_mode_default_optimizer"),
        ),
        _optional_bool_text(
            "uses_production_default_optimizer_settings",
            route.get("uses_production_default_optimizer_settings"),
        ),
        _optional_list_text(
            "production_default_optimizer_setting_mismatches",
            route.get("production_default_optimizer_setting_mismatches"),
        ),
        _optional_bool_text(
            "uses_production_default_route",
            route.get("uses_production_default_route"),
        ),
        _optional_list_text(
            "production_default_route_mismatches",
            route.get("production_default_route_mismatches"),
        ),
        _optional_text("batch_packing", route.get("batch_packing")),
        _route_int_text("family_chunk_size", route),
        _route_int_text("clade_budget", route, none_text="none"),
        _route_int_text("fixed_iters_e", route, none_text="adaptive"),
        _route_int_text("fixed_iters_pi", route),
        _route_int_text("neumann_terms", route),
        _route_int_text("final_check_iters", route),
        _route_int_text("final_check_iters_e", route),
        _route_int_text("configured_steps", route),
        _route_int_text("optimizer_step_cap", route),
        _optional_text(
            "optimizer_step_cap_reason",
            route.get("optimizer_step_cap_reason"),
        ),
    ]
    if route.get("optimizer") == "hessian-sgd":
        fields.extend(
            [
                _route_int_text("solver_warmup_iters", route),
                _route_int_text("fd_adam_warmup_steps", route),
                _route_int_text("fd_hessian_refresh_steps", route),
                _route_int_text(
                    "hessian_sgd_normal_fixed_iters_pi",
                    route,
                    none_text="full",
                ),
                _route_int_text(
                    "hessian_sgd_normal_neumann_terms",
                    route,
                    none_text="full",
                ),
                _optional_bool_text(
                    "hessian_sgd_pi_adjoint_warmstart",
                    route.get("hessian_sgd_pi_adjoint_warmstart"),
                ),
                _optional_metric_text(
                    "pi_fixed_point_relaxation",
                    route.get("pi_fixed_point_relaxation"),
                ),
                _route_int_text("hessian_sgd_validation_interval", route),
                _route_int_text(
                    "hessian_sgd_validation_fixed_iters_pi",
                    route,
                    none_text="configured",
                ),
                _route_int_text(
                    "hessian_sgd_validation_neumann_terms",
                    route,
                    none_text="configured",
                ),
            ]
        )
    elif route.get("optimizer") == "adagrad-restarts":
        fields.extend(
            [
                _optional_text(
                    "adagrad_restart_schedule",
                    route.get("adagrad_restart_schedule"),
                ),
                _route_int_text("adagrad_restart_total_steps", route),
                _route_int_text("adagrad_restart_final_check_iters", route),
            ]
        )
    return " ".join(fields)


def _checkpoint_info_text(
    checkpoint: Path,
    payload: dict[str, Any],
    *,
    route_metadata: tuple[dict[str, Any], str] | None = None,
    production_route_evidence: _ProductionRouteEvidence | None = None,
) -> str:
    if route_metadata is None:
        route, route_source, reconstructed_evidence = (
            _checkpoint_route_metadata_evidence(payload)
        )
        if production_route_evidence is None:
            production_route_evidence = reconstructed_evidence
    else:
        route, route_source = route_metadata
    config_data = payload.get("config")
    if not isinstance(config_data, dict):
        config_data = {}
    status = payload.get("status")
    if not isinstance(status, dict):
        status = {}
    last_row = payload.get("last_row")
    if not isinstance(last_row, dict):
        last_row = {}
    family_names = payload.get("family_names")
    species_names = payload.get("species_names")
    return " ".join(
        [
            _optional_text("checkpoint", checkpoint),
            _optional_int_text("version", payload.get("version")),
            _optional_int_text("step", payload.get("step")),
            _optional_int_text("next_step", payload.get("next_step")),
            _optional_text("status", status.get("status")),
            _optional_text("reason", status.get("reason")),
            _optional_text("mode", route.get("mode", config_data.get("mode"))),
            _optional_text(
                "optimizer",
                route.get("optimizer", config_data.get("optimizer")),
            ),
            _route_metadata_text(
                route,
                production_route_evidence=production_route_evidence,
            ),
            _optional_text("route_metadata_source", route_source),
            _optional_text("optimizer_phase", payload.get("optimizer_phase")),
            _optional_text("last_phase", last_row.get("optimizer/phase")),
            _optional_int_text(
                "families",
                None if not isinstance(family_names, list) else len(family_names),
            ),
            _optional_int_text(
                "species",
                None if not isinstance(species_names, list) else len(species_names),
            ),
            _optional_int_text("best_step", status.get("best_step")),
            _optional_metric_text("best_nll_bits", status.get("best_nll_bits")),
            _optional_metric_text(
                "last_nll_bits",
                last_row.get("likelihood/data_nll_bits"),
            ),
            _optional_metric_text(
                "last_log_likelihood_bits",
                last_row.get("likelihood/log_likelihood_bits"),
            ),
            _optional_metric_text("last_grad_inf", last_row.get("grad/inf")),
            _optional_metric_text(
                "last_projected_grad_inf",
                last_row.get("grad/projected_inf"),
            ),
            _optional_int_text(
                "last_final_check_iters",
                last_row.get("optimizer/final_check_iters"),
            ),
            _optional_int_text(
                "last_final_check_iters_e",
                last_row.get("optimizer/final_check_iters_E"),
            ),
            _optional_text(
                "last_final_check_status",
                last_row.get("optimizer/final_check_status"),
            ),
            _optional_text(
                "last_final_check_source",
                last_row.get("optimizer/final_check_source"),
            ),
            _optional_text(
                "last_final_check_reason",
                last_row.get("optimizer/final_check_reason"),
            ),
            _optional_metric_text(
                "last_final_check_fallback_clade_budget",
                last_row.get("optimizer/final_check_fallback_clade_budget"),
            ),
            _optional_metric_text(
                "last_final_check_loss_abs_delta_bits",
                last_row.get("optimizer/final_check_loss_abs_delta_bits"),
            ),
            _optional_metric_text(
                "last_final_check_grad_max_abs_delta",
                last_row.get("optimizer/final_check_grad_max_abs_delta"),
            ),
            _optional_metric_text(
                "last_final_check_grad_rel_inf_delta",
                last_row.get("optimizer/final_check_grad_rel_inf_delta"),
            ),
            _optional_int_text(
                "last_solver_e_adjoint_failed_batches",
                last_row.get("solver/e_adjoint_failed_batches"),
            ),
            _optional_int_text(
                "last_solver_e_adjoint_success_batches",
                last_row.get("solver/e_adjoint_success_batches"),
            ),
            _optional_metric_text(
                "last_solver_e_adjoint_rel_res_max",
                last_row.get("solver/e_adjoint_rel_res_max"),
            ),
        ]
    )


def _checkpoint_final_check_status(payload: dict[str, Any]) -> object:
    last_row = payload.get("last_row")
    if not isinstance(last_row, dict):
        return None
    return last_row.get("optimizer/final_check_status")


def _run_config_from_args(
    args: argparse.Namespace,
    *,
    validate_input_paths: bool = True,
) -> RunConfig:
    data = _config_data(args.config)
    from gpurec.workflow.config import RunConfig

    for name in _run_config_cli_override_fields():
        _set_if_present(data, args, name)
    missing = [
        name
        for name in ("species_tree", "families_file", "out_dir")
        if data.get(name) is None
    ]
    if missing:
        raise ValueError(f"missing required optimize option(s): {', '.join(missing)}")
    config = RunConfig.from_dict(data)
    if validate_input_paths:
        _validate_run_config_input_paths(config)
    return config


def _preflight_run_config(
    config: RunConfig,
    *,
    check_preprocess: bool = False,
) -> dict[str, object]:
    summary = _validate_run_config_family_references(config)
    if check_preprocess:
        preprocess_summary = _validate_run_config_preprocess(config)
        summary.update(preprocess_summary)
        summary.update(
            _cuda_backward_readiness(preprocess_summary["preprocessed_species_nodes"])
        )
    return summary


def _sampling_config_from_args(
    args: argparse.Namespace,
    checkpoint: Path,
) -> SamplingConfig:
    from gpurec.workflow.config import SamplingConfig

    return SamplingConfig(
        checkpoint=checkpoint,
        out_dir=args.sample_out_dir,
        samples=args.samples,
        seed=args.seed,
        family_start=args.family_start,
        max_families=args.sample_max_families,
        max_events=args.max_events,
        backtrack_binary=args.backtrack_binary,
    )


def _ensure_backtracking_available(backtrack_binary: Path | None) -> None:
    from gpurec.backtracking import ensure_backtracking_available

    ensure_backtracking_available(backtrack_binary)


def _ensure_preprocessing_available(preprocess_native_lib: Path | None) -> Path:
    from gpurec.core.preprocess_rust import ensure_native_preprocessing_available

    return ensure_native_preprocessing_available(
        preprocess_native_lib=preprocess_native_lib,
    )


def _validate_run_sampling_args(args: argparse.Namespace, run_config: RunConfig) -> None:
    _sampling_config_from_args(
        args,
        run_config.out_dir / "checkpoints" / "sampling-argument-validation.pt",
    )


def _config_template_data(args: argparse.Namespace) -> dict[str, Any]:
    from gpurec.workflow.config import (
        DEFAULT_CLADE_BUDGET,
        production_default_optimizer_config_overrides,
    )

    data: dict[str, Any] = {
        "species_tree": str(args.species_tree),
        "families_file": str(args.families_file),
        "out_dir": str(args.out_dir),
        "mode": args.mode,
        "device": args.device,
        "dtype": "float32",
        "optimizer": "auto",
        "family_chunk_size": 0,
        "batch_packing": "depth_first_fit",
        "clade_budget": DEFAULT_CLADE_BUDGET,
        "fixed_iters_pi": 16,
        "neumann_terms": 16,
        "steps": 5000,
        "log_every": 1,
        "checkpoint_every": 1,
    }
    data.update(production_default_optimizer_config_overrides(args.mode))
    return data


def _write_config_template(args: argparse.Namespace) -> Path | None:
    text = json.dumps(_config_template_data(args), indent=2) + "\n"
    output = args.output
    if output is None:
        print(text, end="", flush=True)
        return None
    output = output.expanduser().resolve()
    if output.exists() and not args.force:
        raise ValueError(
            f"output config already exists: {output}; use --force to overwrite"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    return output


def _validate_config_route_text(
    config: RunConfig,
    *,
    route_metadata: dict[str, Any] | None = None,
) -> str:
    route = _config_route_metadata(config) if route_metadata is None else route_metadata
    return _route_metadata_text(route)


def _add_run_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        help=(
            "Flat JSON RunConfig file; relative config paths resolve from the "
            "config file, and explicit CLI flags override matching fields."
        ),
    )
    parser.add_argument(
        "--species-tree",
        type=Path,
        help="Species tree Newick path. Required unless supplied by --config.",
    )
    parser.add_argument(
        "--families-file",
        type=Path,
        help="AleRax-style family list. Required unless supplied by --config.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        help=(
            "Output directory for checkpoints, logs, and rates. Required "
            "unless supplied by --config."
        ),
    )
    parser.add_argument(
        "--mode",
        type=_mode_name,
        choices=("genewise", "global", "specieswise"),
        help="Parameter sharing mode. Workflow default: genewise.",
    )
    parser.add_argument(
        "--device",
        help="Torch device for production optimization. Workflow default: cuda.",
    )
    parser.add_argument(
        "--dtype",
        type=_dtype_name,
        metavar="{float32,float64}",
        help=(
            "Floating-point dtype; aliases include fp32/single and "
            "fp64/double. Workflow default: float32."
        ),
    )
    parser.add_argument("--start", type=int, help="First family index to load.")
    parser.add_argument(
        "--max-families",
        type=int,
        help="Maximum number of families to load.",
    )
    parser.add_argument(
        "--preprocess-cpu-cores",
        type=int,
        help=(
            "Worker thread count for CPU preprocessing. Workflow default uses "
            "Rust preprocessing's runtime default."
        ),
    )
    parser.add_argument(
        "--family-chunk-size",
        type=_chunk_size,
        help=(
            "Families per resident batch; use 0/all/none/null for one "
            "resident batch."
        ),
    )
    parser.add_argument(
        "--clade-budget",
        type=int,
        help=(
            "Clade budget for non-sequential resident-batch packing. "
            "Workflow default: 315000."
        ),
    )
    parser.add_argument(
        "--batch-packing",
        type=_batch_packing,
        metavar="{sequential,clade_first_fit,depth_first_fit}",
        help=(
            "Resident-batch packing policy; aliases include "
            "contiguous/input_order, ffd/clade_ffd, and "
            "depth_ffd/wave_first_fit. Workflow default: depth_first_fit."
        ),
    )
    parser.add_argument(
        "--max-wave-size",
        type=int,
        help="Maximum clades scheduled into one resident wave.",
    )
    parser.add_argument(
        "--small-family-max-leaves",
        type=int,
        help=(
            "Plan families with at most this many leaves before larger "
            "families; use 0 to disable. Workflow default: 0."
        ),
    )
    parser.add_argument(
        "--adaptive-rebatch",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable adaptive resident-batch rebuilding for supported genewise runs.",
    )
    parser.add_argument(
        "--adaptive-rebatch-fraction",
        type=float,
        help="Fraction threshold used by adaptive resident-batch rebuilding.",
    )
    parser.add_argument(
        "--adaptive-rebatch-check-interval",
        type=int,
        help="Step interval for adaptive resident-batch rebuilding checks.",
    )
    parser.add_argument(
        "--adaptive-rebatch-min-remaining-families",
        type=int,
        help="Minimum remaining families before adaptive rebatching can run.",
    )
    parser.add_argument(
        "--fixed-iters-e",
        type=int,
        help=(
            "Fixed E iterations per solve. In specieswise mode, fixed Pi "
            "budgets above 16 force E to be at least the Pi budget."
        ),
    )
    parser.add_argument("--max-iters-e", type=int, help="Maximum adaptive E iterations.")
    parser.add_argument("--tol-e", type=float, help="E fixed-point convergence tolerance.")
    parser.add_argument("--fixed-iters-pi", type=int, help="Fixed Pi iterations per solve.")
    parser.add_argument(
        "--neumann-terms",
        type=int,
        help="Terms for implicit-gradient Neumann series.",
    )
    parser.add_argument(
        "--solver-warmup-iters",
        type=int,
        help=(
            "Initial fixed solver budget for supported genewise active-batch "
            "optimizers and specieswise runs whose full Pi budget is larger; "
            "hessian-sgd keeps E at --fixed-iters-e and uses this only for "
            "Pi/Neumann. Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--final-check-iters",
        type=int,
        help=(
            "Final validation solver budget used only to compare the final loss "
            "and gradient against the configured full budget. Specieswise mode "
            "also uses this for fixed E iterations; use 0 to disable."
        ),
    )
    parser.add_argument(
        "--solver-warmup-grad-inf-tol",
        type=float,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--solver-warmup-loss-patience",
        type=int,
        help=(
            "Switch genewise active-batch optimizers from warmup to full "
            "solvers after this many flat warmup steps."
        ),
    )
    parser.add_argument(
        "--adaptive-iters",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable adaptive E/Pi solver iteration stopping.",
    )
    parser.add_argument(
        "--adaptive-neumann-terms",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Disabled compatibility flag; enabling it is rejected because the "
            "adaptive Neumann path is not part of the supported production "
            "optimization route."
        ),
    )
    parser.add_argument(
        "--convergence-check-interval",
        type=int,
        help="Iteration interval for adaptive solver convergence checks.",
    )
    parser.add_argument(
        "--e-logsumexp-tol",
        type=float,
        help="E logsumexp convergence tolerance.",
    )
    parser.add_argument(
        "--pi-max-diff-tol",
        type=float,
        help="Pi max-difference convergence tolerance.",
    )
    parser.add_argument(
        "--gradient-change-tol",
        type=float,
        help="Absolute gradient-change tolerance.",
    )
    parser.add_argument(
        "--gradient-change-rtol",
        type=float,
        help="Relative gradient-change tolerance.",
    )
    parser.add_argument("--theta-init-d", type=float, help="Initial duplication rate.")
    parser.add_argument("--theta-init-l", type=float, help="Initial loss rate.")
    parser.add_argument("--theta-init-t", type=float, help="Initial transfer rate.")
    parser.add_argument(
        "--min-rate",
        type=float,
        help="Minimum allowed D/L/T rate; defaults to 2^-30.",
    )
    parser.add_argument(
        "--max-rate",
        type=float,
        help="Maximum allowed D/L/T rate; defaults to 2.",
    )
    parser.add_argument(
        "--optimizer",
        type=_optimizer_name,
        choices=(
            "auto",
            "adam",
            "adagrad",
            "projected-sgd",
            "lbfgs",
            "adam-lbfgs",
            "projected-lbfgs",
            "lbfgsb",
            "batched-lbfgs",
            "adam-fd-newton",
            "hessian-sgd",
            "adagrad-restarts",
        ),
        help=(
            "Optimizer schedule. auto uses hessian-sgd for genewise mode, "
            "adagrad-restarts for specieswise mode, and adam otherwise."
        ),
    )
    parser.add_argument("--steps", type=int, help="Maximum optimization steps.")
    parser.add_argument(
        "--lr",
        type=float,
        help="Adam/Adagrad learning rate or hessian-sgd preconditioned step scale.",
    )
    parser.add_argument(
        "--adam-warmup-steps",
        type=int,
        help="Adam steps before LBFGS in adam-lbfgs mode.",
    )
    parser.add_argument(
        "--fd-adam-warmup-steps",
        type=int,
        help="Adam steps per resident batch before Hessian-conditioned genewise updates.",
    )
    parser.add_argument(
        "--fd-hessian-refresh-steps",
        type=int,
        help="Hessian-conditioned genewise steps between full finite-difference Hessian refreshes.",
    )
    parser.add_argument(
        "--hessian-sgd-normal-fixed-iters-pi",
        type=int,
        help=(
            "Optional Pi iteration budget for hessian-sgd full-stage steps."
        ),
    )
    parser.add_argument(
        "--hessian-sgd-normal-neumann-terms",
        type=int,
        help=(
            "Optional Neumann iteration budget for hessian-sgd full-stage steps."
        ),
    )
    parser.add_argument(
        "--hessian-sgd-pi-adjoint-warmstart",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Enable the experimental staged Pi-adjoint warm-start cache for "
            "genewise hessian-sgd runs. Workflow default: disabled."
        ),
    )
    parser.add_argument(
        "--pi-fixed-point-relaxation",
        type=float,
        help=(
            "Experimental Pi-adjoint fixed-point relaxation factor for "
            "warm-started genewise hessian-sgd runs. Workflow default: 1.0."
        ),
    )
    parser.add_argument(
        "--hessian-sgd-validation-interval",
        type=int,
        help=(
            "Full-stage hessian-sgd cadence for high-budget validation "
            "gradient steps; 0 disables periodic validation."
        ),
    )
    parser.add_argument(
        "--hessian-sgd-validation-fixed-iters-pi",
        type=int,
        help=(
            "Optional Pi iteration budget for periodic hessian-sgd validation "
            "gradient steps."
        ),
    )
    parser.add_argument(
        "--hessian-sgd-validation-neumann-terms",
        type=int,
        help=(
            "Optional Neumann budget for periodic hessian-sgd validation "
            "gradient steps."
        ),
    )
    parser.add_argument(
        "--adagrad-restart-schedule",
        help=(
            "Specieswise adagrad-restarts phase schedule as "
            "budget:lr:steps or E/Pi[/Neumann]:lr:steps entries, for "
            "example 8/4:1.0:60,16:0.5:35."
        ),
    )
    parser.add_argument(
        "--adagrad-restart-final-check-iters",
        type=int,
        help=(
            "Final specieswise validation budget for adagrad-restarts; "
            "workflow default: 128."
        ),
    )
    parser.add_argument("--lbfgs-lr", type=float, help="LBFGS learning rate.")
    parser.add_argument("--lbfgs-history-size", type=int, help="LBFGS history size.")
    parser.add_argument("--lbfgs-max-iter", type=int, help="LBFGS inner iterations per step.")
    parser.add_argument("--lbfgs-max-ls", type=int, help="Batched LBFGS line-search probes.")
    parser.add_argument(
        "--lbfgs-line-search",
        choices=("none", "strong_wolfe"),
        help="LBFGS line-search mode.",
    )
    parser.add_argument(
        "--fd-hessian-epsilon",
        type=float,
        help="Finite-difference epsilon for Hessian-conditioned genewise probes.",
    )
    parser.add_argument(
        "--fd-newton-damping",
        type=float,
        help="Diagonal damping added to finite-difference Hessians.",
    )
    parser.add_argument(
        "--grad-inf-tol",
        type=float,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--loss-change-tol",
        type=float,
        help=(
            "Loss-change stopping tolerance; genewise active-batch optimizers "
            "apply this per active family."
        ),
    )
    parser.add_argument(
        "--projected-grad-tol",
        type=float,
        help=(
            "Projected-gradient infinity-norm tolerance for projected optimizers; "
            "projected-lbfgs/lbfgsb keep optimizing instead of stopping "
            "while this is exceeded."
        ),
    )
    parser.add_argument(
        "--projected-lbfgs-min-lr",
        type=float,
        help="Minimum projected-lbfgs base learning rate after automatic backoff.",
    )
    parser.add_argument(
        "--loss-patience",
        type=int,
        help="Consecutive small-loss-change steps before stopping.",
    )
    parser.add_argument(
        "--best-likelihood-patience",
        type=int,
        help="Steps without best-likelihood improvement before stopping.",
    )
    parser.add_argument(
        "--best-likelihood-min-delta",
        type=float,
        help="Minimum best-likelihood improvement to reset patience.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        help="Checkpoint interval in optimization steps; 0 disables periodic checkpoints.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        help="Console progress print interval in optimization steps; history is recorded every step.",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        help="Resume optimization state from an existing checkpoint.",
    )


def _add_backtrack_binary_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--backtrack-binary",
        type=Path,
        help=(
            "Rust backtracking binary. Installed sampling requires this or "
            "GPUREC_BACKTRACK_BIN; source trees can fall back to cargo when "
            "a Rust toolchain is present."
        ),
    )


def _add_preprocess_native_lib_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--preprocess-native-lib",
        type=Path,
        help=(
            "Native Rust preprocessing extension. Installed workflow "
            "preprocessing requires this or GPUREC_PREPROCESS_NATIVE_LIB; source "
            "trees can fall back to Cargo when a Rust toolchain is present."
        ),
    )


def _add_sampling_args(
    parser: argparse.ArgumentParser,
    *,
    checkpoint_required: bool,
    include_checkpoint: bool = True,
) -> None:
    if include_checkpoint:
        parser.add_argument(
            "--checkpoint",
            type=Path,
            required=checkpoint_required,
            help=(
                "Optimization checkpoint to sample, usually checkpoints/best.pt "
                "or checkpoints/latest.pt; theta_final.pt is only a raw tensor "
                "export."
            ),
        )
    parser.add_argument(
        "--sample-out-dir",
        "--sampling-out-dir",
        dest="sample_out_dir",
        type=Path,
        help="Sampling output directory. Defaults under the checkpoint run directory.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Samples per selected family.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for backtracking samples.",
    )
    parser.add_argument(
        "--family-start",
        type=int,
        default=0,
        help="First family index to sample.",
    )
    parser.add_argument(
        "--sample-max-families",
        dest="sample_max_families",
        type=int,
        help="Maximum number of families to sample.",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=100_000,
        help="Maximum events per sampled reconciliation.",
    )
    _add_backtrack_binary_arg(parser)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gpurec",
        description="Optimize D/T/L reconciliation likelihoods and sample RecPhyloXML scenarios.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    optimize_parser = sub.add_parser(
        "optimize",
        help="Optimize D/T/L likelihood parameters.",
        description="Optimize D/T/L likelihood parameters from AleRax-style family inputs.",
    )
    _add_run_config_args(optimize_parser)
    _add_require_mode_default_optimizer_arg(optimize_parser)
    _add_require_production_default_route_arg(optimize_parser)
    optimize_parser.add_argument(
        "--require-converged",
        action="store_true",
        help=(
            "After printing the optimization status, exit with status 1 unless "
            "the status is converged."
        ),
    )
    optimize_parser.add_argument(
        "--require-final-check-ok",
        action="store_true",
        help=(
            "After printing the optimization status, exit with status 1 unless "
            "final_check_status is ok."
        ),
    )
    optimize_parser.set_defaults(_command_parser=optimize_parser)

    validate_parser = sub.add_parser(
        "validate-config",
        help="Validate an optimization config without CUDA.",
        description=(
            "Validate a flat RunConfig JSON file or equivalent CLI flags, "
            "including AleRax family references, without constructing the "
            "CUDA likelihood model."
        ),
    )
    _add_run_config_args(validate_parser)
    _add_require_mode_default_optimizer_arg(validate_parser)
    _add_require_production_default_route_arg(validate_parser)
    validate_parser.add_argument(
        "--check-preprocess",
        action="store_true",
        help=(
            "Also run CPU preprocessing with the retained Rust parser to check "
            "selected Newick trees and leaf/species mappings, then report "
            "whether the species-node count passes the CUDA backward S > 256 gate."
        ),
    )
    validate_parser.add_argument(
        "--require-cuda-backward-ready",
        action="store_true",
        help=(
            "With --check-preprocess, fail unless the species-node count passes "
            "the retained CUDA backward S > 256 gate."
        ),
    )
    validate_parser.set_defaults(_command_parser=validate_parser)

    sample_parser = sub.add_parser(
        "sample",
        help="Sample RecPhyloXML scenarios from a checkpoint.",
        description="Sample RecPhyloXML scenarios from a gpurec optimization checkpoint.",
    )
    _add_sampling_args(sample_parser, checkpoint_required=True)
    _add_require_mode_default_optimizer_arg(sample_parser)
    _add_require_production_default_route_arg(sample_parser)
    sample_parser.set_defaults(_command_parser=sample_parser)

    run_parser = sub.add_parser(
        "run",
        help="Optimize, then sample from the best checkpoint.",
        description="Run optimization, then sample from the best or latest checkpoint it produced.",
    )
    _add_run_config_args(run_parser)
    _add_sampling_args(run_parser, checkpoint_required=False, include_checkpoint=False)
    _add_require_mode_default_optimizer_arg(run_parser)
    _add_require_production_default_route_arg(run_parser)
    run_parser.add_argument("--checkpoint", type=Path, help=argparse.SUPPRESS)
    run_parser.add_argument(
        "--require-converged",
        action="store_true",
        help=(
            "After optimization, print the optimization status and exit before "
            "sampling unless the status is converged."
        ),
    )
    run_parser.add_argument(
        "--require-final-check-ok",
        action="store_true",
        help=(
            "After optimization, print the optimization status and exit before "
            "sampling unless final_check_status is ok."
        ),
    )
    run_parser.set_defaults(_command_parser=run_parser)

    backtrack_check_parser = sub.add_parser(
        "backtrack-check",
        help="Check Rust backtracking command availability.",
        description=(
            "Validate the Rust backtracking binary or source-tree cargo fallback "
            "by running --help without loading a checkpoint."
        ),
    )
    _add_backtrack_binary_arg(backtrack_check_parser)
    backtrack_check_parser.set_defaults(_command_parser=backtrack_check_parser)

    preprocess_check_parser = sub.add_parser(
        "preprocess-check",
        help="Check Rust preprocessing native extension availability.",
        description=(
            "Validate the Rust preprocessing native extension or source-tree "
            "Cargo build fallback without reading dataset files."
        ),
    )
    _add_preprocess_native_lib_arg(preprocess_check_parser)
    preprocess_check_parser.set_defaults(_command_parser=preprocess_check_parser)

    checkpoint_info_parser = sub.add_parser(
        "checkpoint-info",
        help="Print optimization checkpoint status and route metadata.",
        description=(
            "Safely inspect a gpurec optimization checkpoint without building "
            "the CUDA likelihood model."
        ),
    )
    checkpoint_info_parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Optimization checkpoint to inspect, usually checkpoints/best.pt or latest.pt.",
    )
    checkpoint_info_parser.add_argument(
        "--require-final-check-ok",
        action="store_true",
        help=(
            "Exit with status 1 after printing checkpoint info unless the "
            "checkpoint last row has optimizer/final_check_status ok."
        ),
    )
    _add_require_mode_default_optimizer_arg(checkpoint_info_parser)
    _add_require_production_default_route_arg(checkpoint_info_parser)
    checkpoint_info_parser.set_defaults(_command_parser=checkpoint_info_parser)

    summary_info_parser = sub.add_parser(
        "summary-info",
        help="Print optimization summary status and route metadata.",
        description=(
            "Inspect a gpurec optimization summary.json file without building "
            "the CUDA likelihood model."
        ),
    )
    summary_info_parser.add_argument(
        "--summary",
        type=Path,
        required=True,
        help="Optimization summary.json file to inspect.",
    )
    summary_info_parser.add_argument(
        "--require-converged",
        action="store_true",
        help=(
            "Exit with status 1 after printing the summary unless "
            "summary.status is converged."
        ),
    )
    summary_info_parser.add_argument(
        "--require-final-check-ok",
        action="store_true",
        help=(
            "Exit with status 1 after printing the summary unless "
            "summary.final_check_status is ok."
        ),
    )
    _add_require_mode_default_optimizer_arg(summary_info_parser)
    _add_require_production_default_route_arg(summary_info_parser)
    summary_info_parser.set_defaults(_command_parser=summary_info_parser)

    template_parser = sub.add_parser(
        "config-template",
        help="Print or write a flat JSON RunConfig template.",
        description=(
            "Print or write a flat JSON RunConfig template. Genewise and "
            "specieswise templates are production-route starters; global "
            "remains a mode-default Adam diagnostic outside "
            "--require-production-default-route."
        ),
    )
    template_parser.add_argument(
        "--mode",
        type=_mode_name,
        choices=("genewise", "specieswise", "global"),
        default="genewise",
        help=(
            "Template parameter-sharing mode. Genewise/specieswise are "
            "production-route starters; global is a diagnostic Adam template. "
            "Default: genewise."
        ),
    )
    template_parser.add_argument(
        "--species-tree",
        default="S.tree",
        help="Species tree path to place in the template.",
    )
    template_parser.add_argument(
        "--families-file",
        default="families.txt",
        help="AleRax [FAMILIES] path to place in the template.",
    )
    template_parser.add_argument(
        "--out-dir",
        default="output_gpurec",
        help="Output directory to place in the template.",
    )
    template_parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device to place in the template. Default: cuda.",
    )
    template_parser.add_argument(
        "--output",
        type=Path,
        help="Write the template to this path instead of stdout.",
    )
    template_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite --output if it already exists.",
    )
    template_parser.set_defaults(_command_parser=template_parser)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    command_parser = getattr(args, "_command_parser", parser)
    if args.command == "config-template":
        try:
            output = _write_config_template(args)
        except _EXPECTED_WORKFLOW_ERRORS as exc:
            command_parser.error(str(exc))
        if output is not None:
            print(_optional_text("config_template", output), flush=True)
        return
    if args.command == "optimize":
        try:
            config = _run_config_from_args(args, validate_input_paths=False)
            _require_config_route_gates(
                command_parser,
                config,
                require_mode_default_optimizer=args.require_mode_default_optimizer,
                require_production_default_route=args.require_production_default_route,
            )
            _validate_run_config_input_paths(config)
            _preflight_run_config(config)
        except _EXPECTED_WORKFLOW_ERRORS as exc:
            command_parser.error(str(exc))
        try:
            result = optimize(config)
        except _EXPECTED_WORKFLOW_ERRORS as exc:
            _exit_runtime_error(command_parser, str(exc))
        print(
            f"{_optimization_result_text(result)} "
            f"{_optional_text('out_dir', result.out_dir)}",
            flush=True,
        )
        if result.status == "failed":
            command_parser.exit(status=1)
        if args.require_converged and result.status != "converged":
            command_parser.exit(
                status=1,
                message=(
                    "optimization status is "
                    f"{result.status!r}; expected 'converged'"
                    "\n"
                ),
            )
        if args.require_final_check_ok:
            _exit_unless_final_check_ok(
                command_parser,
                getattr(result, "final_check_status", None),
                subject="optimization",
            )
        return
    if args.command == "validate-config":
        if args.require_cuda_backward_ready and not args.check_preprocess:
            command_parser.error(
                "--require-cuda-backward-ready requires --check-preprocess"
            )
        try:
            config = _run_config_from_args(args, validate_input_paths=False)
            route_metadata = _require_config_route_gates(
                command_parser,
                config,
                require_mode_default_optimizer=args.require_mode_default_optimizer,
                require_production_default_route=args.require_production_default_route,
            )
            _validate_run_config_input_paths(config)
            summary = _preflight_run_config(
                config,
                check_preprocess=args.check_preprocess,
            )
        except _EXPECTED_WORKFLOW_ERRORS as exc:
            command_parser.error(str(exc))
        preprocess_text = ""
        if args.check_preprocess:
            cuda_backward_ready = (
                "true" if summary["cuda_backward_ready"] else "false"
            )
            cuda_backward_reason = _optional_text(
                "cuda_backward_ready_reason",
                summary.get("cuda_backward_ready_reason"),
            )
            preprocess_text = (
                f" preprocess_checked=true"
                f" preprocessed_families={summary['preprocessed_families']}"
                f" preprocessed_species_nodes={summary['preprocessed_species_nodes']}"
                f" cuda_backward_ready={cuda_backward_ready}"
                f" {cuda_backward_reason}"
            )
            if (
                args.require_cuda_backward_ready
                and not summary["cuda_backward_ready"]
            ):
                command_parser.error(
                    "cuda_backward_ready=false "
                    f"{cuda_backward_reason}; retained CUDA backward requires "
                    "more than 256 postorder species nodes"
                )
        print(
            "valid_config=true "
            f"mode={config.mode} optimizer={config.optimizer} "
            f"families={summary['families']} "
            f"gene_tree_files={summary['gene_tree_files']} "
            f"mapped_families={summary['mapped_families']} "
            f"{_validate_config_route_text(config, route_metadata=route_metadata)} "
            f"device={config.device} {_optional_text('out_dir', config.out_dir)}"
            f"{preprocess_text}",
            flush=True,
        )
        return
    if args.command == "sample":
        try:
            sampling_config = _sampling_config_from_args(args, args.checkpoint)
            _validate_sampling_checkpoint_path(sampling_config.checkpoint)
        except _EXPECTED_WORKFLOW_ERRORS as exc:
            command_parser.error(_sampling_error_message(exc))
        checkpoint_gate_route: dict[str, Any] | None = None
        checkpoint_gate_evidence: _ProductionRouteEvidence | None = None
        if args.require_mode_default_optimizer or args.require_production_default_route:
            try:
                from gpurec.workflow.checkpoint import load_checkpoint

                payload = load_checkpoint(sampling_config.checkpoint)
            except _EXPECTED_WORKFLOW_ERRORS as exc:
                _exit_runtime_error(command_parser, _sampling_error_message(exc))
            checkpoint_gate_route, _route_source, checkpoint_gate_evidence = (
                _checkpoint_route_metadata_evidence(payload)
            )
        if args.require_mode_default_optimizer:
            _exit_unless_mode_default_optimizer(
                command_parser,
                checkpoint_gate_route or {},
                subject="checkpoint",
                audited_route=(
                    checkpoint_gate_route
                    if checkpoint_gate_evidence is not None
                    else None
                ),
            )
        if args.require_production_default_route:
            _exit_unless_production_default_route(
                command_parser,
                checkpoint_gate_route or {},
                subject="checkpoint",
                production_route_evidence=checkpoint_gate_evidence,
            )
        try:
            result = sample(sampling_config)
        except _EXPECTED_WORKFLOW_ERRORS as exc:
            _exit_runtime_error(command_parser, _sampling_error_message(exc))
        print(
            f"sampled_families={result.families_sampled} "
            f"samples={result.samples_per_family} xml={result.xml_files} "
            f"{_optional_text('out_dir', result.out_dir)}",
            flush=True,
        )
        return
    if args.command == "checkpoint-info":
        try:
            checkpoint = args.checkpoint.expanduser().resolve()
            _validate_sampling_checkpoint_path(checkpoint)
        except _EXPECTED_WORKFLOW_ERRORS as exc:
            command_parser.error(str(exc))
        try:
            from gpurec.workflow.checkpoint import load_checkpoint

            payload = load_checkpoint(checkpoint)
        except _EXPECTED_WORKFLOW_ERRORS as exc:
            _exit_runtime_error(command_parser, _sampling_error_message(exc))
        route_gate_required = (
            args.require_mode_default_optimizer
            or args.require_production_default_route
        )
        route_metadata: tuple[dict[str, Any], str] | None = None
        checkpoint_route_evidence: _ProductionRouteEvidence | None = None
        if route_gate_required:
            route, route_source, checkpoint_route_evidence = (
                _checkpoint_route_metadata_evidence(payload)
            )
            route_metadata = (route, route_source)
        print(
            _checkpoint_info_text(
                checkpoint,
                payload,
                route_metadata=route_metadata,
                production_route_evidence=checkpoint_route_evidence,
            ),
            flush=True,
        )
        if args.require_mode_default_optimizer:
            route, _route_source = route_metadata or _checkpoint_route_metadata(payload)
            _exit_unless_mode_default_optimizer(
                command_parser,
                route,
                subject="checkpoint",
                audited_route=(
                    route if checkpoint_route_evidence is not None else None
                ),
            )
        if args.require_production_default_route:
            route, _route_source = route_metadata or _checkpoint_route_metadata(payload)
            _exit_unless_production_default_route(
                command_parser,
                route,
                subject="checkpoint",
                production_route_evidence=checkpoint_route_evidence,
            )
        if args.require_final_check_ok:
            _exit_unless_final_check_ok(
                command_parser,
                _checkpoint_final_check_status(payload),
                subject="checkpoint",
            )
        return
    if args.command == "summary-info":
        try:
            summary = args.summary.expanduser().resolve()
            _validate_summary_path(summary)
            from gpurec.workflow.config import load_json_object

            payload = load_json_object(summary, description="summary")
        except _EXPECTED_WORKFLOW_ERRORS as exc:
            command_parser.error(str(exc))
        route_gate_required = (
            args.require_mode_default_optimizer
            or args.require_production_default_route
        )
        production_route_evidence = (
            _production_default_route_evidence(payload) if route_gate_required else None
        )
        audited_payload = (
            _route_with_production_default_evidence_fields(
                production_route_evidence[0],
                production_route_evidence[1],
                production_route_evidence[2],
            )
            if production_route_evidence is not None
            else None
        )
        gate_payload = audited_payload if audited_payload is not None else payload
        print(
            _summary_info_text(summary, payload, audited_payload=audited_payload),
            flush=True,
        )
        if args.require_mode_default_optimizer:
            _exit_unless_mode_default_optimizer(
                command_parser,
                gate_payload,
                subject="summary",
                audited_route=audited_payload,
            )
        if args.require_production_default_route:
            _exit_unless_production_default_route(
                command_parser,
                gate_payload,
                subject="summary",
                production_route_evidence=production_route_evidence,
            )
        if args.require_converged and payload.get("status") != "converged":
            command_parser.exit(
                status=1,
                message=(
                    "summary status is "
                    f"{payload.get('status')!r}; expected 'converged'"
                    "\n"
                ),
            )
        if args.require_final_check_ok:
            _exit_unless_final_check_ok(
                command_parser,
                payload.get("final_check_status"),
                subject="summary",
            )
        return
    if args.command == "backtrack-check":
        try:
            _ensure_backtracking_available(args.backtrack_binary)
        except _EXPECTED_WORKFLOW_ERRORS as exc:
            _exit_runtime_error(command_parser, str(exc))
        print("backtracking_available=true", flush=True)
        return
    if args.command == "preprocess-check":
        try:
            preprocess_native_lib = _ensure_preprocessing_available(
                args.preprocess_native_lib,
            )
        except _EXPECTED_WORKFLOW_ERRORS as exc:
            _exit_runtime_error(command_parser, str(exc))
        print(
            "preprocessing_available=true "
            f"{_optional_text('preprocess_native_lib', preprocess_native_lib)}",
            flush=True,
        )
        return
    if args.command == "run":
        if args.checkpoint is not None:
            command_parser.error(
                "gpurec run samples from the checkpoint produced by this optimization; "
                "use gpurec sample --checkpoint to sample an existing checkpoint, or "
                "--resume-from to resume optimization"
            )
        try:
            run_config = _run_config_from_args(args, validate_input_paths=False)
            _require_config_route_gates(
                command_parser,
                run_config,
                require_mode_default_optimizer=args.require_mode_default_optimizer,
                require_production_default_route=args.require_production_default_route,
            )
            _validate_run_config_input_paths(run_config)
            _preflight_run_config(run_config)
            _validate_run_sampling_args(args, run_config)
        except _EXPECTED_WORKFLOW_ERRORS as exc:
            command_parser.error(str(exc))
        try:
            _ensure_backtracking_available(args.backtrack_binary)
        except _EXPECTED_WORKFLOW_ERRORS as exc:
            _exit_runtime_error(command_parser, str(exc))
        try:
            opt_result = optimize(run_config)
        except _EXPECTED_WORKFLOW_ERRORS as exc:
            _exit_runtime_error(command_parser, str(exc))
        if opt_result.status == "failed":
            print(
                f"{_optimization_result_text(opt_result)} "
                f"{_optional_text('out_dir', run_config.out_dir)}",
                flush=True,
            )
            command_parser.exit(
                status=1,
                message=(
                    "optimization failed; refusing to sample from a failed run "
                    f"({opt_result.reason})"
                    "\n"
                ),
            )
        if args.require_converged and opt_result.status != "converged":
            print(
                f"{_optimization_result_text(opt_result)} "
                f"{_optional_text('out_dir', run_config.out_dir)}",
                flush=True,
            )
            command_parser.exit(
                status=1,
                message=(
                    "optimization status is "
                    f"{opt_result.status!r}; expected 'converged'; "
                    "refusing to sample\n"
                ),
            )
        if args.require_final_check_ok and getattr(
            opt_result,
            "final_check_status",
            None,
        ) != "ok":
            print(
                f"{_optimization_result_text(opt_result)} "
                f"{_optional_text('out_dir', run_config.out_dir)}",
                flush=True,
            )
            _exit_unless_final_check_ok(
                command_parser,
                getattr(opt_result, "final_check_status", None),
                subject="optimization",
                action="refusing to sample",
            )
        checkpoint = getattr(opt_result, "sampling_checkpoint", None)
        if checkpoint is None:
            checkpoint = run_config.out_dir / "checkpoints" / "best.pt"
            if not checkpoint.exists():
                checkpoint = run_config.out_dir / "checkpoints" / "latest.pt"
        else:
            checkpoint = Path(checkpoint)
        if not checkpoint.is_file():
            _exit_runtime_error(
                command_parser,
                "optimization completed but no sampling checkpoint was found "
                f"at {checkpoint}",
            )
        try:
            sampling_config = _sampling_config_from_args(args, checkpoint)
        except _EXPECTED_WORKFLOW_ERRORS as exc:
            command_parser.error(_sampling_error_message(exc))
        try:
            sampling_result = sample(sampling_config)
        except _EXPECTED_WORKFLOW_ERRORS as exc:
            _exit_runtime_error(command_parser, _sampling_error_message(exc))
        print(
            f"{_optimization_result_text(opt_result)} "
            f"sampled_families={sampling_result.families_sampled} "
            f"samples={sampling_result.samples_per_family} "
            f"xml={sampling_result.xml_files} "
            f"{_optional_text('out_dir', run_config.out_dir)}",
            f"{_optional_text('sample_out_dir', sampling_result.out_dir)}",
            flush=True,
        )
        return
    parser.error(f"unknown command {args.command!r}")


if __name__ == "__main__":
    main()
