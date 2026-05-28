from __future__ import annotations

import argparse
import inspect
import os
import json
import math
import sys
import shutil
from numbers import Integral, Real
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from gpurec.workflow.config import RunConfig, SamplingConfig

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


def _ensure_json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_ensure_json_ready(item) for item in value]
    if isinstance(value, set):
        return [_ensure_json_ready(item) for item in sorted(value)]
    if isinstance(value, dict):
        return {str(key): _ensure_json_ready(item) for key, item in value.items()}
    return value


def _add_json_output_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a machine-readable JSON report instead of a status text line.",
    )


def _doctor_torch_readiness() -> dict[str, Any]:
    check: dict[str, Any] = {"ok": False}
    try:
        import torch

        check["ok"] = True
        check["version"] = torch.__version__
        check["cuda_build"] = torch.version.cuda
        check["cuda_available"] = bool(torch.cuda.is_available())
        check["cuda_device_count"] = int(torch.cuda.device_count())
        if check["cuda_available"] and check["cuda_device_count"] > 0:
            current = int(torch.cuda.current_device())
            check["cuda_current_device"] = current
            check["cuda_current_device_name"] = torch.cuda.get_device_name(current)
    except Exception as exc:  # pragma: no cover - exercised via monkeypatch in tests
        check["error"] = str(exc)
    return check


def _doctor_triton_readiness() -> dict[str, Any]:
    check: dict[str, Any] = {"ok": False}
    try:
        import triton as triton

        check["ok"] = True
        check["version"] = getattr(triton, "__version__", None)
    except Exception as exc:
        check["error"] = str(exc)
    return check


def _backtrack_command_path(backtrack_binary: Path | None) -> str | None:
    from gpurec import backtracking
    from gpurec.backtracking import _backtrack_command, _is_cargo_fallback_command

    try:
        command = _backtrack_command(
            cargo_manifest=backtracking._BACKTRACK_MANIFEST,
            backtrack_binary=backtrack_binary,
        )
    except Exception:
        return None
    if not command:
        return None
    if command[0] == "cargo":
        if _is_cargo_fallback_command(command):
            marker = "--manifest-path"
            if marker not in command:
                return None
            manifest_index = command.index(marker)
            if manifest_index + 1 >= len(command):
                return None
            return str(Path(command[manifest_index + 1]).expanduser().resolve())
        return None

    path = Path(command[0]).expanduser()
    if path.is_absolute() or os.path.dirname(command[0]) not in ("", os.curdir):
        return str(path.resolve())
    resolved = shutil.which(command[0])
    return resolved


def _add_version_contract_fields(
    check: dict[str, Any],
    *,
    expected_version: str | None,
    package_version: str,
) -> dict[str, Any]:
    check["expected_version"] = expected_version
    check["package_version"] = package_version
    if expected_version is None:
        check["version_compatible"] = None
        return check
    check["version_compatible"] = expected_version == package_version
    if check.get("ok") and not check["version_compatible"]:
        check["ok"] = False
        check["error"] = (
            "incompatible native artifact contract version: "
            f"expected {expected_version!r}, package version is {package_version!r}"
        )
    return check


def _doctor_preprocessing_readiness(
    preprocess_native_lib: Path | None,
    package_version: str,
) -> dict[str, Any]:
    check: dict[str, Any] = {"ok": False}
    from gpurec.core import preprocess_rust

    expected_version = preprocess_rust._manifest_version()
    try:
        path = _ensure_preprocessing_available(preprocess_native_lib)
        check["ok"] = True
        check["path"] = str(path)
        _add_version_contract_fields(
            check,
            expected_version=expected_version,
            package_version=package_version,
        )
    except Exception as exc:
        check["error"] = str(exc)
        _add_version_contract_fields(
            check=check,
            expected_version=expected_version,
            package_version=package_version,
        )
    return check


def _doctor_backtracking_readiness(
    backtrack_binary: Path | None,
    package_version: str,
) -> dict[str, Any]:
    check: dict[str, Any] = {"ok": False}
    from gpurec import backtracking

    expected_version = backtracking._manifest_version()
    try:
        _ensure_backtracking_available(backtrack_binary)
        check["ok"] = True
        check["path"] = _backtrack_command_path(backtrack_binary)
        if backtrack_binary is not None and check["path"] is None:
            check["path"] = str(backtrack_binary)
        _add_version_contract_fields(
            check,
            expected_version=expected_version,
            package_version=package_version,
        )
    except Exception as exc:
        check["error"] = str(exc)
        _add_version_contract_fields(
            check,
            expected_version=expected_version,
            package_version=package_version,
        )
    return check


def _doctor_output_dir_readiness(out_dir: Path | None) -> dict[str, Any]:
    check: dict[str, Any] = {"ok": False}
    target = Path.cwd() if out_dir is None else out_dir.expanduser().resolve()
    check["path"] = str(target)
    try:
        import tempfile

        target.mkdir(parents=True, exist_ok=True)
        if not target.is_dir():
            check["error"] = "resolved path is not a directory"
            return check
        fd, probe = tempfile.mkstemp(prefix=".gpurec-doctor-", dir=str(target))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write("ok")
        finally:
            os.unlink(probe)
        check["ok"] = True
    except Exception as exc:
        check["error"] = str(exc)
    return check


def _doctor_readiness_report(
    out_dir: Path | None,
    preprocess_native_lib: Path | None,
    backtrack_binary: Path | None,
) -> dict[str, Any]:
    from gpurec import __version__

    checks: dict[str, Any] = {
        "python": {
            "ok": True,
            "version": sys.version,
            "platform": sys.platform,
            "executable": sys.executable,
            "implementation": sys.implementation.name,
        },
        "torch": _doctor_torch_readiness(),
        "triton": _doctor_triton_readiness(),
        "preprocess": _doctor_preprocessing_readiness(
            preprocess_native_lib,
            package_version=__version__,
        ),
        "backtracking": _doctor_backtracking_readiness(
            backtrack_binary,
            package_version=__version__,
        ),
        "out_dir": _doctor_output_dir_readiness(out_dir),
    }
    ready = all(check["ok"] for check in checks.values())
    return {
        "ready": ready,
        "package_version": __version__,
        "checks": checks,
    }


def _doctor_readiness_text(report: dict[str, Any]) -> str:
    checks = report["checks"]
    torch_check = checks["torch"]
    triton_check = checks["triton"]
    preprocess_check = checks["preprocess"]
    backtrack_check = checks["backtracking"]
    out_dir_check = checks["out_dir"]
    python_check = checks["python"]
    return " ".join(
        [
            f"doctor_ready={'true' if report['ready'] else 'false'}",
            _optional_text("package_version", report.get("package_version")),
            _optional_text("python_version", python_check.get("version")),
            _optional_text("python_platform", python_check.get("platform")),
            _optional_text("python_executable", python_check.get("executable")),
            _optional_text("torch_version", torch_check.get("version")),
            _optional_bool_text("torch_cuda_available", torch_check.get("cuda_available")),
            _optional_int_text("torch_cuda_devices", torch_check.get("cuda_device_count")),
            _optional_text("torch_cuda_build", torch_check.get("cuda_build")),
            _optional_text(
                "torch_error",
                torch_check.get("error") if not torch_check.get("ok") else None,
            ),
            _optional_text("triton_version", triton_check.get("version")),
            _optional_text(
                "triton_error",
                triton_check.get("error") if not triton_check.get("ok") else None,
            ),
            _optional_bool_text(
                "preprocess_available",
                preprocess_check.get("ok"),
            ),
            _optional_text("preprocess_path", preprocess_check.get("path")),
            _optional_text(
                "preprocess_error",
                preprocess_check.get("error") if not preprocess_check.get("ok") else None,
            ),
            _optional_text(
                "preprocess_expected_version",
                preprocess_check.get("expected_version"),
            ),
            _optional_text(
                "preprocess_package_version",
                preprocess_check.get("package_version"),
            ),
            _optional_bool_text(
                "preprocess_version_compatible",
                preprocess_check.get("version_compatible"),
            ),
            _optional_bool_text(
                "backtracking_available",
                backtrack_check.get("ok"),
            ),
            _optional_text("backtracking_path", backtrack_check.get("path")),
            _optional_text(
                "backtracking_error",
                backtrack_check.get("error") if not backtrack_check.get("ok") else None,
            ),
            _optional_text(
                "backtracking_expected_version",
                backtrack_check.get("expected_version"),
            ),
            _optional_text(
                "backtracking_package_version",
                backtrack_check.get("package_version"),
            ),
            _optional_bool_text(
                "backtracking_version_compatible",
                backtrack_check.get("version_compatible"),
            ),
            _optional_bool_text("out_dir_writable", out_dir_check.get("ok")),
            _optional_text("out_dir", out_dir_check.get("path")),
            _optional_text(
                "out_dir_error",
                out_dir_check.get("error") if not out_dir_check.get("ok") else None,
            ),
        ]
    )


def _run_config_cli_override_fields() -> tuple[str, ...]:
    from dataclasses import fields

    from gpurec.workflow.config import RunConfig

    return tuple(field.name for field in fields(RunConfig))


def _sampling_error_message(exc: BaseException) -> str:
    message = str(exc)
    if _RAW_THETA_CHECKPOINT_ERROR in message:
        return _with_suggestion(
            f"{message}; --checkpoint must point to an optimization checkpoint "
            "such as checkpoints/best.pt or checkpoints/latest.pt, not "
            "theta_final.pt",
            "run optimize first (or use an existing run directory) and pass checkpoints/best.pt or checkpoints/latest.pt to --checkpoint",
        )
    return message


def _exit_runtime_error(parser: argparse.ArgumentParser, message: str) -> None:
    parser.exit(status=1, message=f"error: {message}\n")


def _with_suggestion(message: str, suggestion: str) -> str:
    suggestion_text = suggestion.strip()
    if not suggestion_text:
        return message
    return f"{message}; suggestion: {suggestion_text}"


def _exit_unless_final_check_ok(
    parser: argparse.ArgumentParser,
    status: object,
    *,
    subject: str,
    action: str | None = None,
) -> None:
    if status == "ok":
        return
    message = _with_suggestion(
        f"{subject} final check status is {status!r}; expected 'ok'",
        "inspect final-check diagnostics via summary-info/checkpoint-info and rerun with adjusted solver/optimizer settings",
    )
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
    return _with_suggestion(
        message,
        "use optimizer=auto or the documented mode default optimizer for the selected mode",
    )


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
    return _with_suggestion(
        message,
        "use optimizer=auto and remove route/optimizer override fields to restore the shipped production default route",
    )


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


def optimize(
    config: Any,
    *,
    command_argv: tuple[str, ...] | list[str] | None = None,
) -> Any:
    from gpurec.workflow.optimize import optimize as _optimize

    if command_argv is None:
        return _optimize(config)
    return _optimize(config, command_argv=command_argv)


def _run_optimize_command(config: Any, command_argv: list[str]) -> Any:
    """Run optimization while preserving test monkeypatch compatibility."""
    try:
        parameters = inspect.signature(optimize).parameters
    except TypeError:
        return optimize(config)

    accepts_var_keyword = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    has_command_argv = "command_argv" in parameters
    if accepts_var_keyword or has_command_argv:
        return optimize(config, command_argv=command_argv)
    return optimize(config)


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
    if str(path) == "-":
        from gpurec.workflow.config import load_run_config_text

        return load_run_config_text(
            sys.stdin.read(),
            base_dir=Path.cwd(),
            description="config <stdin>",
        )
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
            raise ValueError(
                _with_suggestion(
                    f"{option} path does not exist or is not a file: {path}",
                    f"verify the path or regenerate a template with `gpurec config-template` and update {option}",
                )
            )
    if config.resume_from is not None and not config.resume_from.is_file():
        raise ValueError(
            _with_suggestion(
                "--resume-from path does not exist or is not a file: "
                f"{config.resume_from}",
                "use a checkpoint path such as output_gpurec/checkpoints/latest.pt",
            )
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
            _with_suggestion(
                "AleRax family file references missing gene-tree path(s): "
                f"{preview}{suffix}",
                "fix starting_gene_tree entries in the [FAMILIES] file so each referenced path exists",
            )
        )
    return {
        "families": len(family_names),
        "gene_tree_files": gene_tree_files,
        "mapped_families": sum(1 for mapping in leaf_species_maps if mapping),
    }


def _input_issue(
    code: str,
    *,
    family: str | None = None,
    index: int | None = None,
    path: str | None = None,
    gene: str | None = None,
    species: str | None = None,
    message: str,
    action: str,
) -> dict[str, Any]:
    issue: dict[str, Any] = {
        "code": code,
        "message": message,
        "action": action,
    }
    if family is not None:
        issue["family"] = family
    if index is not None:
        issue["index"] = index
    if path is not None:
        issue["path"] = path
    if gene is not None:
        issue["gene"] = gene
    if species is not None:
        issue["species"] = species
    return issue


def _summarize_alerax_family_inputs(config: RunConfig) -> dict[str, Any]:
    from gpurec.core.model import (
        parse_alerax_family_file_with_paths,
        parse_alerax_mapping_file,
    )

    summary: dict[str, Any] = {
        "valid_inputs": True,
        "preprocess_checked": False,
        "species_tree": str(config.species_tree),
        "families_file": str(config.families_file),
        "start": config.start,
        "max_families": config.max_families,
        "families": 0,
        "mapped_families": 0,
        "gene_tree_files": 0,
        "issues": [],
        "family_reports": [],
        "family_names": [],
        "tree_paths": [],
        "mapping_paths": [],
        "parsed_leaf_species_maps": [],
    }

    if not config.species_tree.is_file():
        summary["valid_inputs"] = False
        summary["issues"].append(
            _input_issue(
                "missing_species_tree",
                path=str(config.species_tree),
                message=(
                    "species-tree path does not exist or is not a file: "
                    f"{config.species_tree}"
                ),
                action=(
                    "Create the species tree file or fix --species-tree to "
                    "point at an existing file."
                ),
            )
        )

    try:
        family_names, tree_paths, mapping_paths = parse_alerax_family_file_with_paths(
            config.families_file,
            start=config.start,
            max_families=config.max_families,
        )
    except _EXPECTED_WORKFLOW_ERRORS as exc:
        summary["valid_inputs"] = False
        summary["issues"].append(
            _input_issue(
                "families_file_parse_error",
                path=str(config.families_file),
                message=str(exc),
                action="Fix the families file syntax and retry.",
            )
        )
        return summary

    summary["family_names"] = family_names
    summary["tree_paths"] = tree_paths
    summary["mapping_paths"] = mapping_paths

    for family_index, (name, paths, mapping_path) in enumerate(
        zip(family_names, tree_paths, mapping_paths)
    ):
        family: dict[str, Any] = {
            "index": family_index,
            "name": name,
            "status": "ok",
            "issues": [],
            "tree_paths": list(paths),
            "tree_path_count": len(paths),
            "mapping_path": mapping_path,
            "mapping_size": 0,
        }
        mapping: dict[str, str] = {}
        if mapping_path is not None:
            mapping_file = Path(mapping_path)
            if not mapping_file.is_file():
                family["status"] = "error"
                family["issues"].append(
                    _input_issue(
                        "missing_mapping_file",
                        family=name,
                        index=family_index,
                        path=str(mapping_file),
                        message=(
                            "mapping file path does not exist or is not a "
                            f"file for family {name!r}: {mapping_file}"
                        ),
                        action="Create the mapping file or remove mapping= from this family.",
                    )
                )
            else:
                try:
                    mapping = parse_alerax_mapping_file(mapping_file)
                except _EXPECTED_WORKFLOW_ERRORS as exc:
                    family["status"] = "error"
                    family["issues"].append(
                        _input_issue(
                            "invalid_mapping_file",
                            family=name,
                            index=family_index,
                            path=str(mapping_file),
                            message=f"invalid mapping file for family {name!r}: {exc}",
                            action="Fix the mapping file format and retry.",
                        )
                    )
                else:
                    family["mapping_size"] = len(mapping)
        summary["parsed_leaf_species_maps"].append(mapping)

        for raw_path in paths:
            path = Path(raw_path)
            summary["gene_tree_files"] += 1
            if not path.is_file():
                family["status"] = "error"
                family["issues"].append(
                    _input_issue(
                        "missing_gene_tree",
                        family=name,
                        index=family_index,
                        path=str(path),
                        message=(
                            "gene-tree file path does not exist or is not a "
                            f"file for family {name!r}: {path}"
                        ),
                        action=(
                            "Add this file or fix the family tree path in the"
                            " families file."
                        ),
                    )
                )

        if family["status"] == "error":
            summary["valid_inputs"] = False
        summary["family_reports"].append(family)

    summary["families"] = len(family_names)
    summary["mapped_families"] = sum(
        1 for path in mapping_paths if path is not None
    )

    summary["issues"] = [
        *summary["issues"],
        *[
            issue
            for family in summary["family_reports"]
            for issue in family["issues"]
        ],
    ]

    return summary


def _validate_run_config_preprocess_inputs(config: RunConfig, summary: dict[str, Any]) -> dict[str, Any]:
    import torch

    from gpurec.core.model import GeneDataset

    if not summary.get("valid_inputs", True):
        return summary

    family_names = summary["family_names"]
    tree_paths = summary["tree_paths"]
    leaf_species_maps = summary["parsed_leaf_species_maps"]
    report = summary
    try:
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
    except _EXPECTED_WORKFLOW_ERRORS as exc:
        report["preprocess_checked"] = True
        report["preprocess_error"] = str(exc)
        report["valid_inputs"] = False
        for family in report["family_reports"]:
            if family["status"] == "error":
                continue
            family_index = int(family["index"])
            family_name = family["name"]
            try:
                _ = GeneDataset(
                    config.species_tree,
                    [family["tree_paths"]],
                    genewise=config.mode == "genewise",
                    specieswise=config.mode == "specieswise",
                    dtype=torch.float32,
                    device="cpu",
                    preprocess_cpu_cores=config.preprocess_cpu_cores,
                    family_names=[family_name],
                    leaf_species_maps=[
                        report["parsed_leaf_species_maps"][family_index]
                    ],
                )
            except _EXPECTED_WORKFLOW_ERRORS as family_exc:
                family["status"] = "error"
                family["issues"].append(
                    _input_issue(
                        "preprocess_error",
                        family=family_name,
                        index=family_index,
                        path=(
                            family["tree_paths"][0]
                            if family["tree_paths"]
                            else None
                        ),
                        message=(
                            f"preprocessing failed for family {family_name!r}: "
                            f"{family_exc}"
                        ),
                        action="Fix this family data in the families file.",
                    )
                )
                report["issues"].append(family["issues"][-1])
        return report

    report["preprocess_checked"] = True
    report["preprocessed_families"] = int(dataset.num_families)
    report["preprocessed_species_nodes"] = int(dataset.S)
    report["cuda_backward_ready"] = int(dataset.S) > _CUDA_BACKWARD_MIN_SPECIES_NODES_EXCLUSIVE
    report["cuda_backward_ready_reason"] = (
        None
        if report["preprocessed_species_nodes"]
        > _CUDA_BACKWARD_MIN_SPECIES_NODES_EXCLUSIVE
        else "requires_s_gt_256"
    )
    report["species_names"] = [str(name) for name in dataset.species_helpers["names"]]

    species_set = set(report["species_names"])
    for family, mapping in zip(
        report["family_reports"], report["parsed_leaf_species_maps"]
    ):
        if not mapping:
            continue
        unknown = sorted({species for species in mapping.values() if species not in species_set})
        if unknown:
            family["status"] = "error"
            family["issues"].append(
                _input_issue(
                    "unknown_mapping_species",
                    family=family.get("name"),
                    index=family.get("index"),
                    message=(
                        f"mapping references unknown species for family "
                        f"{family.get('name')!r}: {', '.join(unknown)}"
                    ),
                    action=(
                        "Update the mapping file to use only species present "
                        "in the species tree."
                    ),
                )
            )
            report["valid_inputs"] = False

    return report


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
            _with_suggestion(
                f"--checkpoint path does not exist or is not a file: {path}",
                "use an optimization checkpoint such as output_gpurec/checkpoints/latest.pt",
            )
        )


def _validate_summary_path(summary: Path) -> None:
    path = summary.expanduser().resolve()
    if not path.is_file():
        raise ValueError(
            _with_suggestion(
                f"--summary path does not exist or is not a file: {path}",
                "point to output_gpurec/summary.json from a completed or in-progress run directory",
            )
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
    data = _resolved_run_config_data_from_args(args)
    from gpurec.workflow.config import RunConfig

    config = RunConfig.from_dict(data)
    if validate_input_paths:
        _validate_run_config_input_paths(config)
    return config


def _resolved_run_config_data_from_args(args: argparse.Namespace) -> dict[str, Any]:
    data = _config_data(args.config)
    for name in _run_config_cli_override_fields():
        _set_if_present(data, args, name)
    missing = [
        name
        for name in ("species_tree", "families_file", "out_dir")
        if data.get(name) is None
    ]
    if missing:
        raise ValueError(f"missing required optimize option(s): {', '.join(missing)}")
    return data


def _run_config_explanation(
    config: RunConfig,
    *,
    raw_config_data: dict[str, Any],
    route_metadata: dict[str, Any],
) -> dict[str, Any]:
    provided_fields = sorted(str(name) for name in raw_config_data)
    effective = config.to_dict()
    inferred_defaults = sorted(
        name for name in effective if name not in raw_config_data
    )

    optimizer_source = "explicit"
    if "optimizer" not in raw_config_data:
        optimizer_source = "default"
    elif str(raw_config_data.get("optimizer", "")).strip().lower() == "auto":
        optimizer_source = "mode-default-from-auto"

    return {
        "provided_fields": provided_fields,
        "inferred_default_fields": inferred_defaults,
        "optimizer_resolution": {
            "source": optimizer_source,
            "mode": config.mode,
            "effective_optimizer": config.optimizer,
            "mode_default_optimizer": route_metadata.get("mode_default_optimizer"),
            "uses_mode_default_optimizer": route_metadata.get(
                "uses_mode_default_optimizer"
            ),
        },
        "route_resolution": {
            "uses_production_default_route": route_metadata.get(
                "uses_production_default_route"
            ),
            "production_default_basis": route_metadata.get(
                "production_default_basis"
            ),
            "batch_packing": route_metadata.get("batch_packing"),
            "family_chunk_size": route_metadata.get("family_chunk_size"),
            "clade_budget": route_metadata.get("clade_budget"),
        },
        "effective_config": _ensure_json_ready(effective),
    }


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
    if args.mode == "specieswise":
        data["adagrad_restart_phase_loss_patience"] = 0
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


def _workflow_dry_run_text(
    *,
    command: str,
    config: RunConfig,
    summary: dict[str, object],
    route_metadata: dict[str, Any],
) -> str:
    cuda_ready = summary.get("cuda_backward_ready")
    return (
        f"dry_run=true command={command} "
        f"mode={config.mode} optimizer={config.optimizer} "
        f"families={summary.get('families', 0)} "
        f"gene_tree_files={summary.get('gene_tree_files', 0)} "
        f"mapped_families={summary.get('mapped_families', 0)} "
        f"preprocessed_families={summary.get('preprocessed_families', 0)} "
        f"preprocessed_species_nodes={summary.get('preprocessed_species_nodes', 0)} "
        f"estimated_memory_bytes={_dry_run_memory_estimate_bytes(summary, route_metadata)} "
        f"cuda_backward_ready={str(cuda_ready).lower() if isinstance(cuda_ready, bool) else 'null'} "
        f"{_optional_text('cuda_backward_ready_reason', summary.get('cuda_backward_ready_reason'))} "
        f"{_validate_config_route_text(config, route_metadata=route_metadata)} "
        f"device={config.device} {_optional_text('out_dir', config.out_dir)}"
    )


def _dry_run_memory_estimate_bytes(
    summary: dict[str, object],
    route_metadata: dict[str, Any],
) -> int:
    families = int(summary.get("preprocessed_families", 0) or 0)
    species_nodes = int(summary.get("preprocessed_species_nodes", 0) or 0)
    if families <= 0 or species_nodes <= 0:
        return 0
    chunk_size = int(route_metadata.get("family_chunk_size", 0) or 0)
    active_families = families if chunk_size <= 0 else min(families, chunk_size)
    bytes_per_species_node = 256
    return active_families * species_nodes * bytes_per_species_node


def _add_run_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        help=(
            "relative config paths resolve from the config file. "
            "Use '-' as --config to read JSON from stdin; in that mode, "
            "relative paths resolve from stdin's current directory. "
            "Explicit CLI flags override matching fields. "
            "Flat JSON RunConfig file may also be provided."
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
            "adagrad-restarts-lbfgsb",
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
            "example 8/4:1.0:60,16:0.5:35. Also controls the Adagrad "
            "prefix of adagrad-restarts-lbfgsb."
        ),
    )
    parser.add_argument(
        "--adagrad-restart-final-check-iters",
        type=int,
        help=(
            "Final specieswise validation budget for adagrad-restarts and "
            "adagrad-restarts-lbfgsb; "
            "workflow default: 128."
        ),
    )
    parser.add_argument(
        "--adagrad-restart-phase-loss-patience",
        type=int,
        help=(
            "For specieswise adagrad-restarts, advance to the next restart "
            "phase after this many flat-loss steps; 0 keeps fixed phase lengths. "
            "The same rule controls the Adagrad prefix of "
            "adagrad-restarts-lbfgsb."
        ),
    )
    parser.add_argument("--lbfgs-lr", type=float, help="LBFGS learning rate.")
    parser.add_argument("--lbfgs-history-size", type=int, help="LBFGS history size.")
    parser.add_argument("--lbfgs-max-iter", type=int, help="LBFGS inner iterations per step.")
    parser.add_argument("--lbfgs-max-ls", type=int, help="Batched LBFGS line-search probes.")
    parser.add_argument(
        "--lbfgsb-high-kkt-stop-patience",
        type=int,
        help=(
            "For lbfgsb, stop after this many consecutive high-KKT "
            "tiny-progress rows; 0 disables the stop."
        ),
    )
    parser.add_argument(
        "--lbfgsb-high-kkt-stop-min-fallbacks",
        type=int,
        help=(
            "Minimum accepted lbfgsb fallback rows before the high-KKT stop can "
            "trigger."
        ),
    )
    parser.add_argument(
        "--lbfgsb-fallback-max-coordinates",
        type=int,
        help=(
            "Maximum coordinate sign-fallback candidates for lbfgsb fallback "
            "competition; 0 disables coordinate fallback."
        ),
    )
    parser.add_argument(
        "--lbfgsb-fallback-max-loss-evals",
        type=int,
        help=(
            "Optional per-step loss-only evaluation budget for lbfgsb fallback "
            "line searches and fallback competition."
        ),
    )
    parser.add_argument(
        "--lbfgsb-fallback-resolution-competition-factor",
        type=float,
        help=(
            "For lbfgsb fallback competition, also challenge accepted fallback "
            "moves whose decrease is at most this multiple of the fp loss "
            "resolution; 0 keeps only the ordinary tiny-progress trigger."
        ),
    )
    parser.add_argument(
        "--lbfgsb-best-retry-attempts",
        type=int,
        help=(
            "For lbfgsb, reload the best checkpoint this many times when a "
            "terminal plateau is reached, preserving serialized LBFGS-B state."
        ),
    )
    parser.add_argument(
        "--lbfgsb-loss-change-tol-schedule",
        help=(
            "Optional lbfgsb loss-stop schedule as "
            "loss_change_tol:loss_patience entries, for example 0.25:2,0.1:2."
        ),
    )
    parser.add_argument(
        "--lbfgsb-loss-schedule-force-fallback",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "When an lbfgsb loss-stop schedule advances, force the next row to "
            "start from the projected-gradient fallback."
        ),
    )
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
        "--loss-stop-projected-grad-gate",
        dest="loss_stop_projected_grad_gate",
        action="store_true",
        default=None,
        help=(
            "Require projected-lbfgs/lbfgsb to pass --projected-grad-tol before "
            "loss-change patience can stop the run."
        ),
    )
    parser.add_argument(
        "--no-loss-stop-projected-grad-gate",
        dest="loss_stop_projected_grad_gate",
        action="store_false",
        default=None,
        help=(
            "Allow loss-change patience to stop projected-lbfgs/lbfgsb even when "
            "the projected-gradient diagnostic is still above tolerance."
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
    optimize_parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate config, routes, and CPU preprocessing only; print "
            "estimated workflow readiness/counts without running optimization."
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
    _add_json_output_arg(validate_parser)
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
    validate_parser.add_argument(
        "--explain-config",
        action="store_true",
        help=(
            "Include effective-config defaults and route/optimizer resolution "
            "details to explain why selected defaults were chosen."
        ),
    )
    validate_parser.set_defaults(_command_parser=validate_parser)

    validate_inputs_parser = sub.add_parser(
        "validate-inputs",
        help="Validate AleRax input files and references without CUDA.",
        description=(
            "Validate species tree and AleRax family declarations, optionally "+
            "running CPU preprocessing to validate Newick parsing and mapping "
            "coverage."
        ),
    )
    validate_inputs_parser.add_argument(
        "--species-tree",
        type=Path,
        required=True,
        help="Species-tree Newick path.",
    )
    validate_inputs_parser.add_argument(
        "--families-file",
        type=Path,
        required=True,
        help="AleRax [FAMILIES] file path.",
    )
    validate_inputs_parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="First family index to validate.",
    )
    validate_inputs_parser.add_argument(
        "--max-families",
        type=int,
        help="Maximum number of families to validate.",
    )
    validate_inputs_parser.add_argument(
        "--mode",
        type=_mode_name,
        choices=("genewise", "global", "specieswise"),
        default="genewise",
        help=(
            "Parameter-sharing mode used during preprocessing. Workflow "
            "default: genewise."
        ),
    )
    validate_inputs_parser.add_argument(
        "--preprocess-cpu-cores",
        type=int,
        help=(
            "Worker thread count for CPU preprocessing. Workflow default uses "
            "Rust preprocessing's runtime default."
        ),
    )
    _add_json_output_arg(validate_inputs_parser)
    validate_inputs_parser.add_argument(
        "--check-preprocess",
        action="store_true",
        help=(
            "Also run CPU preprocessing with the retained Rust parser to check "
            "selected Newick trees and leaf/species mappings, then report "
            "whether the species-node count passes the CUDA backward S > 256 gate."
        ),
    )
    validate_inputs_parser.add_argument(
        "--require-cuda-backward-ready",
        action="store_true",
        help=(
            "With --check-preprocess, fail unless the species-node count "
            "passes the retained CUDA backward S > 256 gate."
        ),
    )
    validate_inputs_parser.set_defaults(_command_parser=validate_inputs_parser)

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
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate config, routes, sampling args, and CPU preprocessing "
            "only; print estimated workflow readiness/counts without running "
            "optimization or sampling."
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
    _add_json_output_arg(backtrack_check_parser)
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
    _add_json_output_arg(preprocess_check_parser)
    _add_preprocess_native_lib_arg(preprocess_check_parser)
    preprocess_check_parser.set_defaults(_command_parser=preprocess_check_parser)

    doctor_parser = sub.add_parser(
        "doctor",
        help="Print workflow readiness checks before running optimization.",
        description=(
            "Collect installation and runtime readiness for Python runtime, "
            "PyTorch, Triton, native preprocessing, backtracking binary, "
            "and a writable output directory."
        ),
    )
    _add_json_output_arg(doctor_parser)
    _add_preprocess_native_lib_arg(doctor_parser)
    _add_backtrack_binary_arg(doctor_parser)
    doctor_parser.add_argument(
        "--out-dir",
        type=Path,
        help="Directory to probe for writable tempfile checks when validating out-dir readiness.",
    )
    doctor_parser.set_defaults(_command_parser=doctor_parser)

    checkpoint_info_parser = sub.add_parser(
        "checkpoint-info",
        help="Print optimization checkpoint status and route metadata.",
        description=(
            "Safely inspect a gpurec optimization checkpoint without building "
            "the CUDA likelihood model."
        ),
    )
    _add_json_output_arg(checkpoint_info_parser)
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
    _add_json_output_arg(summary_info_parser)
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
    invocation_argv = list(sys.argv[1:] if argv is None else argv)
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
            route_metadata = _require_config_route_gates(
                command_parser,
                config,
                require_mode_default_optimizer=args.require_mode_default_optimizer,
                require_production_default_route=args.require_production_default_route,
            )
            _validate_run_config_input_paths(config)
            summary = _preflight_run_config(
                config,
                check_preprocess=args.dry_run,
            )
        except _EXPECTED_WORKFLOW_ERRORS as exc:
            command_parser.error(str(exc))
        route_metadata_for_report = (
            route_metadata
            if route_metadata is not None
            else _config_route_metadata(config)
        )
        if args.dry_run:
            print(
                _workflow_dry_run_text(
                    command="optimize",
                    config=config,
                    summary=summary,
                    route_metadata=route_metadata_for_report,
                ),
                flush=True,
            )
            return
        try:
            result = _run_optimize_command(config, invocation_argv)
        except _EXPECTED_WORKFLOW_ERRORS as exc:
            _exit_runtime_error(
                command_parser,
                _with_suggestion(
                    str(exc),
                    "inspect optimize diagnostics, then retry optimize or run validate-config --check-preprocess to isolate input/native setup failures",
                ),
            )
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
                    _with_suggestion(
                        "optimization status is "
                        f"{result.status!r}; expected 'converged'",
                        "inspect summary/checkpoint diagnostics and resume with higher steps or adjusted optimizer settings",
                    )
                    + "\n"
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
                _with_suggestion(
                    "--require-cuda-backward-ready requires --check-preprocess",
                    "add --check-preprocess to run CPU preprocessing before enforcing CUDA backward readiness",
                )
            )
        try:
            raw_config_data = _resolved_run_config_data_from_args(args)
            from gpurec.workflow.config import RunConfig

            config = RunConfig.from_dict(raw_config_data)
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
        route_metadata_for_report = (
            route_metadata
            if route_metadata is not None
            else _config_route_metadata(config)
        )
        explanation = (
            _run_config_explanation(
                config,
                raw_config_data=raw_config_data,
                route_metadata=route_metadata_for_report,
            )
            if args.explain_config
            else None
        )
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
                    _with_suggestion(
                        "cuda_backward_ready=false "
                        f"{cuda_backward_reason}; retained CUDA backward requires "
                        "more than 256 postorder species nodes",
                        "rerun without --require-cuda-backward-ready to inspect config and preprocessing results before enforcing the CUDA backward gate",
                    )
                )
        if args.json:
            payload: dict[str, Any] = {
                "valid_config": True,
                "mode": config.mode,
                "optimizer": config.optimizer,
                "families": summary["families"],
                "gene_tree_files": summary["gene_tree_files"],
                "mapped_families": summary["mapped_families"],
                "device": config.device,
                "out_dir": config.out_dir,
            }
            if args.check_preprocess:
                payload["preprocess_checked"] = True
                payload["preprocessed_families"] = summary[
                    "preprocessed_families"
                ]
                payload["preprocessed_species_nodes"] = summary[
                    "preprocessed_species_nodes"
                ]
                payload["cuda_backward_ready"] = summary["cuda_backward_ready"]
                payload["cuda_backward_ready_reason"] = summary[
                    "cuda_backward_ready_reason"
                ]
            else:
                payload["preprocess_checked"] = False
            payload["route"] = _ensure_json_ready(route_metadata_for_report)
            if explanation is not None:
                payload["explain_config"] = explanation
            print(json.dumps(_ensure_json_ready(payload), indent=2), flush=True)
        else:
            explain_text = ""
            if explanation is not None:
                explain_text = (
                    " explain_config=true "
                    f"optimizer_source={explanation['optimizer_resolution']['source']} "
                    f"default_fields={len(explanation['inferred_default_fields'])}"
                )
            print(
                "valid_config=true "
                f"mode={config.mode} optimizer={config.optimizer} "
                f"families={summary['families']} "
                f"gene_tree_files={summary['gene_tree_files']} "
                f"mapped_families={summary['mapped_families']} "
                f"{_validate_config_route_text(config, route_metadata=route_metadata)} "
                f"device={config.device} {_optional_text('out_dir', config.out_dir)}"
                f"{preprocess_text}{explain_text}",
                flush=True,
            )
        return
    if args.command == "validate-inputs":
        if args.require_cuda_backward_ready and not args.check_preprocess:
            command_parser.error(
                _with_suggestion(
                    "--require-cuda-backward-ready requires --check-preprocess",
                    "add --check-preprocess to run CPU preprocessing before enforcing CUDA backward readiness",
                )
            )
        config = SimpleNamespace(
            species_tree=args.species_tree.expanduser().resolve(),
            families_file=args.families_file.expanduser().resolve(),
            start=args.start,
            max_families=args.max_families,
            mode=args.mode,
            preprocess_cpu_cores=args.preprocess_cpu_cores,
        )
        try:
            summary = _summarize_alerax_family_inputs(config)
            if args.check_preprocess:
                summary = _validate_run_config_preprocess_inputs(config, summary)
        except _EXPECTED_WORKFLOW_ERRORS as exc:
            command_parser.error(str(exc))
        preprocess_text = ""
        if args.check_preprocess:
            if args.require_cuda_backward_ready and not summary["valid_inputs"]:
                command_parser.error(
                    _with_suggestion(
                        "input validation failed; fix input issues before checking "
                        "CUDA backward readiness",
                        "rerun validate-inputs --json to review issue codes and actions, then fix input contracts before enabling --require-cuda-backward-ready",
                    )
                )
            preprocess_text = (
                f" preprocess_checked={summary.get('preprocess_checked', False)}"
                f" preprocessed_families={summary.get('preprocessed_families', 0)}"
                f" preprocessed_species_nodes={summary.get('preprocessed_species_nodes', 0)}"
                f" cuda_backward_ready={summary.get('cuda_backward_ready', False)}"
                f" {_optional_text('cuda_backward_ready_reason', summary.get('cuda_backward_ready_reason'))}"
            )
            if args.require_cuda_backward_ready and (
                not summary.get("cuda_backward_ready", False)
            ):
                reason = (
                    summary.get("cuda_backward_ready_reason")
                    or summary.get("preprocess_error")
                )
                command_parser.error(
                    _with_suggestion(
                        "cuda_backward_ready=false "
                        f"{_optional_text('cuda_backward_ready_reason', reason)}; "
                        "retained CUDA backward requires more than 256 postorder species nodes",
                        "rerun without --require-cuda-backward-ready to inspect input validation and preprocessing details before enforcing the CUDA backward gate",
                    )
                )
        if args.json:
            print(json.dumps(_ensure_json_ready(summary), indent=2), flush=True)
        else:
            print(
                f"valid_inputs={str(summary['valid_inputs']).lower()} "
                f"families={summary['families']} "
                f"gene_tree_files={summary['gene_tree_files']} "
                f"mapped_families={summary['mapped_families']} "
                f"issues={len(summary['issues'])} "
                f"mode={config.mode}"
                f"{preprocess_text}",
                flush=True,
            )
        if not summary["valid_inputs"]:
            command_parser.exit(status=1)
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
        route_for_report, route_source_for_report, checkpoint_route_evidence = (
            _checkpoint_route_metadata_evidence(payload)
        )
        status = payload.get("status")
        if not isinstance(status, dict):
            status = {}
        last_row = payload.get("last_row")
        if not isinstance(last_row, dict):
            last_row = {}
        config_data = payload.get("config")
        if not isinstance(config_data, dict):
            config_data = {}
        family_names = payload.get("family_names")
        species_names = payload.get("species_names")
        if args.json:
            print(
                json.dumps(
                    _ensure_json_ready(
                        {
                            "checkpoint": checkpoint,
                            "version": payload.get("version"),
                            "step": payload.get("step"),
                            "next_step": payload.get("next_step"),
                            "status": {
                                "status": status.get("status"),
                                "reason": status.get("reason"),
                            },
                            "mode": route_for_report.get(
                                "mode", config_data.get("mode")
                            ),
                            "optimizer": route_for_report.get(
                                "optimizer", config_data.get("optimizer")
                            ),
                            "route": route_for_report,
                            "route_metadata_source": route_source_for_report,
                            "optimizer_phase": payload.get("optimizer_phase"),
                            "last_phase": last_row.get("optimizer/phase"),
                            "families": None
                            if not isinstance(family_names, list)
                            else len(family_names),
                            "species": None
                            if not isinstance(species_names, list)
                            else len(species_names),
                            "best_step": status.get("best_step"),
                            "best_nll_bits": status.get("best_nll_bits"),
                            "last_nll_bits": last_row.get("likelihood/data_nll_bits"),
                            "last_log_likelihood_bits": last_row.get(
                                "likelihood/log_likelihood_bits"
                            ),
                            "last_grad_inf": last_row.get("grad/inf"),
                            "last_projected_grad_inf": last_row.get(
                                "grad/projected_inf"
                            ),
                            "last_final_check_iters": last_row.get(
                                "optimizer/final_check_iters"
                            ),
                            "last_final_check_status": last_row.get(
                                "optimizer/final_check_status"
                            ),
                            "last_final_check_source": last_row.get(
                                "optimizer/final_check_source"
                            ),
                            "last_final_check_reason": last_row.get(
                                "optimizer/final_check_reason"
                            ),
                            "last_final_check_fallback_clade_budget": last_row.get(
                                "optimizer/final_check_fallback_clade_budget"
                            ),
                            "last_final_check_loss_abs_delta_bits": last_row.get(
                                "optimizer/final_check_loss_abs_delta_bits"
                            ),
                            "last_final_check_grad_max_abs_delta": last_row.get(
                                "optimizer/final_check_grad_max_abs_delta"
                            ),
                            "last_final_check_grad_rel_inf_delta": last_row.get(
                                "optimizer/final_check_grad_rel_inf_delta"
                            ),
                            "last_solver_e_adjoint_failed_batches": last_row.get(
                                "solver/e_adjoint_failed_batches"
                            ),
                            "last_solver_e_adjoint_success_batches": last_row.get(
                                "solver/e_adjoint_success_batches"
                            ),
                            "last_solver_e_adjoint_rel_res_max": last_row.get(
                                "solver/e_adjoint_rel_res_max"
                            ),
                        }
                    ),
                    indent=2,
                ),
                flush=True,
            )
        else:
            print(
                _checkpoint_info_text(
                    checkpoint,
                    payload,
                    route_metadata=(route_for_report, route_source_for_report),
                    production_route_evidence=checkpoint_route_evidence,
                ),
                flush=True,
            )
        if args.require_mode_default_optimizer:
            route, _route_source = route_for_report, route_source_for_report
            _exit_unless_mode_default_optimizer(
                command_parser,
                route,
                subject="checkpoint",
                audited_route=(
                    route if checkpoint_route_evidence is not None else None
                ),
            )
        if args.require_production_default_route:
            route, _route_source = route_for_report, route_source_for_report
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
        if args.json:
            print(
                json.dumps(
                    _ensure_json_ready(
                        {
                            "summary": summary,
                            "status": payload.get("status"),
                            "reason": payload.get("reason"),
                            "mode": payload.get("mode"),
                            "optimizer": payload.get("optimizer"),
                            "mode_default_optimizer": payload.get(
                                "mode_default_optimizer"
                            ),
                            "uses_mode_default_optimizer": payload.get(
                                "uses_mode_default_optimizer"
                            ),
                            "uses_production_default_optimizer_settings": payload.get(
                                "uses_production_default_optimizer_settings"
                            ),
                            "production_default_optimizer_setting_mismatches": payload.get(
                                "production_default_optimizer_setting_mismatches"
                            ),
                            "uses_production_default_route": payload.get(
                                "uses_production_default_route"
                            ),
                            "production_default_route_mismatches": payload.get(
                                "production_default_route_mismatches"
                            ),
                            "families": payload.get("families"),
                            "species": payload.get("species"),
                            "batches": payload.get("batches"),
                            "batch_packing": payload.get("batch_packing"),
                            "family_chunk_size": payload.get("family_chunk_size"),
                            "clade_budget": payload.get("clade_budget"),
                            "fixed_iters_e": payload.get("fixed_iters_e"),
                            "fixed_iters_pi": payload.get("fixed_iters_pi"),
                            "neumann_terms": payload.get("neumann_terms"),
                            "objective": payload.get("objective"),
                            "gradient_route": payload.get("gradient_route"),
                            "rate_parameterization": payload.get(
                                "rate_parameterization"
                            ),
                            "production_default_basis": payload.get(
                                "production_default_basis"
                            ),
                            "configured_steps": payload.get("configured_steps"),
                            "optimizer_step_cap": payload.get("optimizer_step_cap"),
                            "optimizer_step_cap_reason": payload.get(
                                "optimizer_step_cap_reason"
                            ),
                            "final_check_iters": payload.get("final_check_iters"),
                            "final_check_iters_e": payload.get("final_check_iters_e"),
                            "solver_warmup_iters": payload.get("solver_warmup_iters"),
                            "fd_adam_warmup_steps": payload.get("fd_adam_warmup_steps"),
                            "fd_hessian_refresh_steps": payload.get(
                                "fd_hessian_refresh_steps"
                            ),
                            "hessian_sgd_normal_fixed_iters_pi": payload.get(
                                "hessian_sgd_normal_fixed_iters_pi"
                            ),
                            "hessian_sgd_normal_neumann_terms": payload.get(
                                "hessian_sgd_normal_neumann_terms"
                            ),
                            "hessian_sgd_pi_adjoint_warmstart": payload.get(
                                "hessian_sgd_pi_adjoint_warmstart"
                            ),
                            "pi_fixed_point_relaxation": payload.get(
                                "pi_fixed_point_relaxation"
                            ),
                            "hessian_sgd_validation_interval": payload.get(
                                "hessian_sgd_validation_interval"
                            ),
                            "hessian_sgd_validation_fixed_iters_pi": payload.get(
                                "hessian_sgd_validation_fixed_iters_pi"
                            ),
                            "hessian_sgd_validation_neumann_terms": payload.get(
                                "hessian_sgd_validation_neumann_terms"
                            ),
                            "adagrad_restart_schedule": payload.get(
                                "adagrad_restart_schedule"
                            ),
                            "adagrad_restart_total_steps": payload.get(
                                "adagrad_restart_total_steps"
                            ),
                            "adagrad_restart_final_check_iters": payload.get(
                                "adagrad_restart_final_check_iters"
                            ),
                            "steps_completed": payload.get("steps_completed"),
                            "elapsed_s": payload.get("elapsed_s"),
                            "best_step": payload.get("best_step"),
                            "final_nll_bits": payload.get("final_nll_bits"),
                            "final_log_likelihood_bits": payload.get(
                                "final_log_likelihood_bits"
                            ),
                            "final_grad_inf": payload.get("final_grad_inf"),
                            "final_projected_grad_inf": payload.get(
                                "final_projected_grad_inf"
                            ),
                            "best_nll_bits": payload.get("best_nll_bits"),
                            "best_log_likelihood_bits": payload.get(
                                "best_log_likelihood_bits"
                            ),
                            "final_check_status": payload.get("final_check_status"),
                            "final_check_source": payload.get("final_check_source"),
                            "final_check_reason": payload.get("final_check_reason"),
                            "final_check_fallback_clade_budget": payload.get(
                                "final_check_fallback_clade_budget"
                            ),
                            "final_check_loss_abs_delta_bits": payload.get(
                                "final_check_loss_abs_delta_bits"
                            ),
                            "final_check_grad_max_abs_delta": payload.get(
                                "final_check_grad_max_abs_delta"
                            ),
                            "final_check_grad_rel_inf_delta": payload.get(
                                "final_check_grad_rel_inf_delta"
                            ),
                            "final_solver_e_adjoint_failed_batches": payload.get(
                                "final_solver_e_adjoint_failed_batches"
                            ),
                            "final_solver_e_adjoint_success_batches": payload.get(
                                "final_solver_e_adjoint_success_batches"
                            ),
                            "final_solver_e_adjoint_rel_res_max": payload.get(
                                "final_solver_e_adjoint_rel_res_max"
                            ),
                            "route": gate_payload.get("route", {}),
                        }
                    ),
                    indent=2,
                ),
                flush=True,
            )
        else:
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
                    _with_suggestion(
                        "summary status is "
                        f"{payload.get('status')!r}; expected 'converged'",
                        "review summary status/reason and resume optimization before enforcing converged-only gates",
                    )
                    + "\n"
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
        from gpurec import __version__ as package_version

        backtrack_payload = _doctor_backtracking_readiness(
            args.backtrack_binary,
            package_version=package_version,
        )
        if not backtrack_payload.get("ok"):
            _exit_runtime_error(
                command_parser,
                _with_suggestion(
                    str(backtrack_payload.get("error")),
                    "install or point to a compatible backtracking artifact via --backtrack-binary/GPUREC_BACKTRACK_BIN, then rerun backtrack-check",
                ),
            )
        payload = {
            "backtracking_available": True,
            "backtrack_binary": (
                str(args.backtrack_binary)
                if args.backtrack_binary is not None
                else None
            ),
            "expected_version": backtrack_payload.get("expected_version"),
            "package_version": backtrack_payload.get("package_version"),
            "version_compatible": backtrack_payload.get("version_compatible"),
        }
        if backtrack_payload.get("path") is not None:
            payload["backtrack_binary"] = backtrack_payload.get("path")
        if args.json:
            print(json.dumps(payload, indent=2), flush=True)
        else:
            print("backtracking_available=true", flush=True)
        return
    if args.command == "preprocess-check":
        from gpurec import __version__ as package_version

        preprocess_payload = _doctor_preprocessing_readiness(
            args.preprocess_native_lib,
            package_version=package_version,
        )
        if not preprocess_payload.get("ok"):
            _exit_runtime_error(
                command_parser,
                _with_suggestion(
                    str(preprocess_payload.get("error")),
                    "install or point to a compatible preprocessing native library via --preprocess-native-lib/GPUREC_PREPROCESS_NATIVE_LIB, then rerun preprocess-check",
                ),
            )
        preprocess_native_lib = preprocess_payload.get("path")
        payload = {
            "preprocessing_available": True,
            "preprocess_native_lib": str(preprocess_native_lib),
            "expected_version": preprocess_payload.get("expected_version"),
            "package_version": preprocess_payload.get("package_version"),
            "version_compatible": preprocess_payload.get("version_compatible"),
        }
        if args.json:
            print(json.dumps(payload, indent=2), flush=True)
        else:
            print(
                "preprocessing_available=true "
                f"{_optional_text('preprocess_native_lib', preprocess_native_lib)}",
                flush=True,
            )
        return

    if args.command == "doctor":
        report = _doctor_readiness_report(
            args.out_dir,
            args.preprocess_native_lib,
            args.backtrack_binary,
        )
        if args.json:
            print(json.dumps(_ensure_json_ready(report), indent=2), flush=True)
        else:
            print(_doctor_readiness_text(report), flush=True)
        if not report["ready"]:
            command_parser.exit(status=1)
        return
    if args.command == "run":
        if args.checkpoint is not None:
            command_parser.error(
                _with_suggestion(
                    "gpurec run samples from the checkpoint produced by this optimization; "
                    "use gpurec sample --checkpoint to sample an existing checkpoint, or "
                    "--resume-from to resume optimization",
                    "remove --checkpoint from run; use run for optimize+sample, sample --checkpoint for sampling-only, or optimize --resume-from for resume-only",
                )
            )
        try:
            run_config = _run_config_from_args(args, validate_input_paths=False)
            route_metadata = _require_config_route_gates(
                command_parser,
                run_config,
                require_mode_default_optimizer=args.require_mode_default_optimizer,
                require_production_default_route=args.require_production_default_route,
            )
            _validate_run_config_input_paths(run_config)
            summary = _preflight_run_config(
                run_config,
                check_preprocess=args.dry_run,
            )
            _validate_run_sampling_args(args, run_config)
        except _EXPECTED_WORKFLOW_ERRORS as exc:
            command_parser.error(str(exc))
        route_metadata_for_report = (
            route_metadata
            if route_metadata is not None
            else _config_route_metadata(run_config)
        )
        if args.dry_run:
            print(
                _workflow_dry_run_text(
                    command="run",
                    config=run_config,
                    summary=summary,
                    route_metadata=route_metadata_for_report,
                ),
                flush=True,
            )
            return
        try:
            _ensure_backtracking_available(args.backtrack_binary)
        except _EXPECTED_WORKFLOW_ERRORS as exc:
            _exit_runtime_error(
                command_parser,
                _with_suggestion(
                    str(exc),
                    "install or point to a compatible backtracking artifact via --backtrack-binary/GPUREC_BACKTRACK_BIN, run backtrack-check, then rerun run",
                ),
            )
        try:
            opt_result = _run_optimize_command(run_config, invocation_argv)
        except _EXPECTED_WORKFLOW_ERRORS as exc:
            _exit_runtime_error(
                command_parser,
                _with_suggestion(
                    str(exc),
                    "inspect optimize diagnostics, then retry run or use optimize first to isolate optimization failures before sampling",
                ),
            )
        if opt_result.status == "failed":
            print(
                f"{_optimization_result_text(opt_result)} "
                f"{_optional_text('out_dir', run_config.out_dir)}",
                flush=True,
            )
            command_parser.exit(
                status=1,
                message=(
                    _with_suggestion(
                        "optimization failed; refusing to sample from a failed run "
                        f"({opt_result.reason})",
                        "inspect summary/checkpoint diagnostics, fix the failure cause, then resume or rerun optimize before sampling",
                    )
                    + "\n"
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
                    _with_suggestion(
                        "optimization status is "
                        f"{opt_result.status!r}; expected 'converged'; "
                        "refusing to sample",
                        "resume or rerun optimization until converged before invoking run-level sampling gates",
                    )
                    + "\n"
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
                _with_suggestion(
                    "optimization completed but no sampling checkpoint was found "
                    f"at {checkpoint}",
                    "resume optimization to produce checkpoints/latest.pt or checkpoints/best.pt, then rerun run or invoke sample --checkpoint explicitly",
                ),
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
