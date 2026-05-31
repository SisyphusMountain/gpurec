from __future__ import annotations

import argparse
import inspect
import os
import json
import math
import sys
import shutil
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Mapping

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


def _emit_readiness_check(
    parser: argparse.ArgumentParser,
    *,
    payload: dict[str, Any],
    json_output: bool,
    text_success: str,
    suggestion: str,
) -> None:
    if "ok" in payload and payload.get("ok") is False:
        _suggested_exit_error(
            parser,
            str(payload.get("error")),
            suggestion,
        )
    if json_output:
        print(json.dumps(_ensure_json_ready(payload), indent=2), flush=True)
    else:
        print(text_success, flush=True)


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


_SUGGEST_RUN_OPTIMIZE_FAILURE = (
    "inspect optimize diagnostics, then retry optimize or run validate-config --check-preprocess to isolate input/native setup failures"
)
_SUGGEST_RUN_OPTIMIZE_PREP = (
    "inspect optimize diagnostics, then retry run or use optimize first to isolate optimization failures before sampling"
)
_SUGGEST_RUN_BACKTRACK_PREP = (
    "install or point to a compatible backtracking artifact via --backtrack-binary/GPUREC_BACKTRACK_BIN, "
    "run backtrack-check, then rerun run"
)
_SUGGEST_MISSING_SAMPLE_CHECKPOINT = (
    "resume optimization to produce checkpoints/latest.pt or checkpoints/best.pt, then rerun run or invoke sample --checkpoint explicitly"
)
_SUGGEST_REQUIRE_CUDA_BACKWARD_READY = (
    "add --check-preprocess to run CPU preprocessing before enforcing CUDA backward readiness"
)
_SUGGEST_VALIDATE_CUDA_BACKWARD = (
    "rerun without --require-cuda-backward-ready to inspect config and preprocessing results before enforcing the CUDA backward gate"
)


def _suggested_command_error(
    parser: argparse.ArgumentParser,
    message: str,
    suggestion: str,
) -> None:
    parser.error(_with_suggestion(message, suggestion))


def _suggested_exit_error(
    parser: argparse.ArgumentParser,
    error: Exception | str,
    suggestion: str,
) -> None:
    _exit_runtime_error(parser, _with_suggestion(str(error), suggestion))


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


def _require_route_uses_mode_default_optimizer(
    parser: argparse.ArgumentParser,
    route: dict[str, Any],
    *,
    subject: str = "config",
) -> None:
    if route.get("uses_mode_default_optimizer") is not True:
        parser.error(
            _mode_default_optimizer_gate_message(
                subject,
                route,
                action=_MODE_DEFAULT_OPTIMIZER_CONFIG_ACTION,
            )
        )


def _require_route_uses_production_default_route(
    parser: argparse.ArgumentParser,
    route: dict[str, Any],
    *,
    subject: str = "config",
) -> None:
    if route.get("uses_production_default_route") is not True:
        parser.error(
            _production_default_route_gate_message(
                subject,
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
        _require_route_uses_mode_default_optimizer(
            parser,
            route,
            subject="config",
        )
    if require_production_default_route:
        _require_route_uses_production_default_route(
            parser,
            route,
            subject="config",
        )
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


def _result_field_specs_for_optimizer(
    optimizer: object,
) -> tuple[tuple[str, str], ...]:
    base = (
        ("status", "text"),
        ("reason", "text"),
        ("mode", "text"),
        ("optimizer", "text"),
        ("mode_default_optimizer", "text"),
        ("uses_mode_default_optimizer", "bool"),
        ("uses_production_default_optimizer_settings", "bool"),
        ("production_default_optimizer_setting_mismatches", "list"),
        ("uses_production_default_route", "bool"),
        ("production_default_route_mismatches", "list"),
        ("families", "int"),
        ("species", "int"),
        ("batches", "int"),
        ("batch_packing", "text"),
        ("family_chunk_size", "int"),
        ("clade_budget", "int"),
        ("fixed_iters_e", "int"),
        ("fixed_iters_pi", "int"),
        ("neumann_terms", "int"),
        ("objective", "text"),
        ("gradient_route", "text"),
        ("rate_parameterization", "text"),
        ("production_default_basis", "text"),
        ("configured_steps", "int"),
        ("optimizer_step_cap", "int"),
        ("optimizer_step_cap_reason", "text"),
        ("final_check_iters", "int"),
        ("final_check_iters_e", "int"),
        ("steps_completed", "int"),
        ("elapsed_s", "metric"),
        ("best_step", "int"),
        ("sampling_checkpoint", "text"),
        ("final_nll_bits", "metric"),
        ("final_log_likelihood_bits", "metric"),
        ("final_grad_inf", "metric"),
        ("final_projected_grad_inf", "metric"),
        ("best_nll_bits", "metric"),
        ("best_log_likelihood_bits", "metric"),
        ("final_check_status", "text"),
        ("final_check_source", "text"),
        ("final_check_reason", "text"),
        ("final_check_fallback_clade_budget", "metric"),
        ("final_check_loss_abs_delta_bits", "metric"),
        ("final_check_grad_max_abs_delta", "metric"),
        ("final_check_grad_rel_inf_delta", "metric"),
        ("final_solver_e_adjoint_failed_batches", "int"),
        ("final_solver_e_adjoint_success_batches", "int"),
        ("final_solver_e_adjoint_rel_res_max", "metric"),
    )
    if optimizer == "hessian-sgd":
        return (
            *base,
            ("solver_warmup_iters", "int"),
            ("fd_adam_warmup_steps", "int"),
            ("fd_hessian_refresh_steps", "int"),
            ("hessian_sgd_normal_fixed_iters_pi", "int"),
            ("hessian_sgd_normal_neumann_terms", "int"),
            ("hessian_sgd_pi_adjoint_warmstart", "bool"),
            ("pi_fixed_point_relaxation", "metric"),
            ("hessian_sgd_validation_interval", "int"),
            ("hessian_sgd_validation_fixed_iters_pi", "int"),
            ("hessian_sgd_validation_neumann_terms", "int"),
        )
    if optimizer == "adagrad-restarts":
        return (
            *base,
            ("adagrad_restart_schedule", "text"),
            ("adagrad_restart_total_steps", "int"),
            ("adagrad_restart_final_check_iters", "int"),
        )
    return base


def _result_field_value(source: Any, name: str) -> Any:
    if isinstance(source, Mapping):
        return source.get(name)
    return getattr(source, name, None)


def _format_field_value(name: str, kind: str, value: object) -> str:
    if kind == "text":
        return _optional_text(name, value)
    if kind == "bool":
        return _optional_bool_text(name, value)
    if kind == "int":
        return _optional_int_text(name, value)
    if kind == "metric":
        return _optional_metric_text(name, value)
    if kind == "list":
        return _optional_list_text(name, value)
    raise ValueError(f"unsupported field kind: {kind!r}")


def _status_text_from_fields(
    source: Any,
    fields: tuple[tuple[str, str], ...],
    *,
    overrides: Mapping[str, Any] | None = None,
) -> str:
    extra = {} if overrides is None else overrides
    return " ".join(
        _format_field_value(
            name,
            kind,
            extra.get(name, _result_field_value(source, name)),
        )
        for name, kind in fields
    )


def _json_from_fields(
    source: Any,
    fields: tuple[tuple[str, str], ...],
    *,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    extra = {} if overrides is None else overrides
    return {
        name: _ensure_json_ready(
            extra.get(name, _result_field_value(source, name))
        )
        for name, kind in fields
    }


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
    fields = _result_field_specs_for_optimizer(
        _result_field_value(result, "optimizer"),
    )
    return _status_text_from_fields(
        result,
        fields,
        overrides={
            "final_log_likelihood_bits": final_log_likelihood,
            "best_log_likelihood_bits": best_log_likelihood,
        },
    )


def _summary_info_text(
    summary: Path,
    payload: dict[str, Any],
    *,
    audited_payload: dict[str, Any] | None = None,
) -> str:
    route_payload = (
        _route_with_production_default_audit_fields(payload)
        if audited_payload is None
        else audited_payload
    )
    status_text = _status_text_from_fields(
        route_payload,
        _result_field_specs_for_optimizer(
            _result_field_value(route_payload, "optimizer")
        ),
    )
    return f"{_optional_text('summary', summary)} {status_text}"


def _summary_info_payload(
    summary: Path,
    payload: dict[str, Any],
    route: dict[str, Any],
    *,
    audited_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source = payload if audited_payload is None else audited_payload
    return {
        "summary": summary,
        **_json_from_fields(
            source,
            _result_field_specs_for_optimizer(
                _result_field_value(source, "optimizer")
            ),
        ),
        "route": route,
    }


def _summary_info_route_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    route = payload.get("route", {})
    if not isinstance(route, dict):
        route = {}
    metadata = dict(route)
    base = {
        "mode": payload.get("mode"),
        "optimizer": payload.get("optimizer"),
        "mode_default_optimizer": payload.get("mode_default_optimizer"),
        "uses_mode_default_optimizer": payload.get("uses_mode_default_optimizer"),
        "uses_production_default_optimizer_settings": payload.get(
            "uses_production_default_optimizer_settings"
        ),
        "production_default_optimizer_setting_mismatches": payload.get(
            "production_default_optimizer_setting_mismatches"
        ),
        "uses_production_default_route": payload.get("uses_production_default_route"),
        "production_default_route_mismatches": payload.get(
            "production_default_route_mismatches"
        ),
        "objective": payload.get("objective"),
        "gradient_route": payload.get("gradient_route"),
        "rate_parameterization": payload.get("rate_parameterization"),
        "production_default_basis": payload.get("production_default_basis"),
        "batch_packing": payload.get("batch_packing"),
        "family_chunk_size": payload.get("family_chunk_size"),
        "clade_budget": payload.get("clade_budget"),
        "fixed_iters_e": payload.get("fixed_iters_e"),
        "fixed_iters_pi": payload.get("fixed_iters_pi"),
        "neumann_terms": payload.get("neumann_terms"),
        "final_check_iters": payload.get("final_check_iters"),
        "final_check_iters_e": payload.get("final_check_iters_e"),
        "configured_steps": payload.get("configured_steps"),
        "optimizer_step_cap": payload.get("optimizer_step_cap"),
        "optimizer_step_cap_reason": payload.get("optimizer_step_cap_reason"),
        "solver_warmup_iters": payload.get("solver_warmup_iters"),
        "fd_adam_warmup_steps": payload.get("fd_adam_warmup_steps"),
        "fd_hessian_refresh_steps": payload.get("fd_hessian_refresh_steps"),
        "hessian_sgd_normal_fixed_iters_pi": payload.get(
            "hessian_sgd_normal_fixed_iters_pi"
        ),
        "hessian_sgd_normal_neumann_terms": payload.get(
            "hessian_sgd_normal_neumann_terms"
        ),
        "hessian_sgd_pi_adjoint_warmstart": payload.get(
            "hessian_sgd_pi_adjoint_warmstart"
        ),
        "pi_fixed_point_relaxation": payload.get("pi_fixed_point_relaxation"),
        "hessian_sgd_validation_interval": payload.get(
            "hessian_sgd_validation_interval"
        ),
        "hessian_sgd_validation_fixed_iters_pi": payload.get(
            "hessian_sgd_validation_fixed_iters_pi"
        ),
        "hessian_sgd_validation_neumann_terms": payload.get(
            "hessian_sgd_validation_neumann_terms"
        ),
        "adagrad_restart_schedule": payload.get("adagrad_restart_schedule"),
        "adagrad_restart_total_steps": payload.get("adagrad_restart_total_steps"),
        "adagrad_restart_final_check_iters": payload.get(
            "adagrad_restart_final_check_iters"
        ),
    }
    for key, value in base.items():
        if key not in metadata and key in payload:
            metadata[key] = value
    return metadata


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


def _validate_existing_file(
    path: Path,
    *,
    option: str,
    suggestion: str,
) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(
            _with_suggestion(
                f"{option} path does not exist or is not a file: {resolved}",
                suggestion,
            )
        )
    return resolved


def _validate_run_config_family_references(config: RunConfig) -> dict[str, int]:
    summary = _summarize_alerax_family_inputs(config)
    if not summary["valid_inputs"]:
        raise ValueError(_compact_input_error(summary))
    return {
        "families": int(summary["families"]),
        "gene_tree_files": int(summary["gene_tree_files"]),
        "mapped_families": int(summary["mapped_families"]),
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
                            "missing gene-tree path; file does not exist or is not a "
                            f"file for family {name!r}: {path}"
                        ),
                        action=(
                            "Add this file or fix starting_gene_tree entries in the"
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


def _compact_input_error(summary: dict[str, Any]) -> str:
    issues = summary.get("issues")
    if not isinstance(issues, list) or not issues:
        return (
            "input validation failed; fix input issues then rerun with valid files"
        )
    first = issues[0]
    if not isinstance(first, dict):
        return "input validation failed; fix input issues then rerun with valid files"
    message = first.get("message")
    if message is None:
        message = "input validation failed; fix input issues"
    action = first.get("action")
    if action:
        return _with_suggestion(message, str(action))
    return str(message)


def _build_cpu_preprocess_dataset(
    config: RunConfig,
    tree_paths: list[object],
    *,
    family_names: list[object],
    leaf_species_maps: list[object],
) -> object:
    import torch

    from gpurec.core.model import GeneDataset

    return GeneDataset(
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


def _validate_run_config_preprocess_inputs(config: RunConfig, summary: dict[str, Any]) -> dict[str, Any]:

    if not summary.get("valid_inputs", True):
        return summary

    family_names = summary["family_names"]
    tree_paths = summary["tree_paths"]
    leaf_species_maps = summary["parsed_leaf_species_maps"]
    report = summary
    try:
        dataset = _build_cpu_preprocess_dataset(
            config,
            tree_paths,
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
                _ = _build_cpu_preprocess_dataset(
                    config,
                    [family["tree_paths"]],
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
    summary = _summarize_alerax_family_inputs(config)
    if not summary["valid_inputs"]:
        raise ValueError(_compact_input_error(summary))
    validated = _validate_run_config_preprocess_inputs(config, summary)
    if not validated.get("valid_inputs", True):
        raise ValueError(_compact_input_error(validated))
    return {
        "preprocessed_families": int(validated["preprocessed_families"]),
        "preprocessed_species_nodes": int(validated["preprocessed_species_nodes"]),
        "cuda_backward_ready": validated["cuda_backward_ready"],
        "cuda_backward_ready_reason": validated["cuda_backward_ready_reason"],
    }


def _cuda_backward_readiness(species_nodes: int) -> dict[str, object]:
    ready = species_nodes > _CUDA_BACKWARD_MIN_SPECIES_NODES_EXCLUSIVE
    return {
        "cuda_backward_ready": ready,
        "cuda_backward_ready_reason": None if ready else "requires_s_gt_256",
    }


def _validate_sampling_checkpoint_path(checkpoint: Path) -> None:
    _validate_existing_file(
        checkpoint,
        option="--checkpoint",
        suggestion="use an optimization checkpoint such as output_gpurec/checkpoints/latest.pt",
    )


def _validate_summary_path(summary: Path) -> None:
    _validate_existing_file(
        summary,
        option="--summary",
        suggestion=(
            "point to output_gpurec/summary.json from a completed or in-progress "
            "run directory"
        ),
    )


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
        route = {
            key: config_data[key]
            for key in ("mode", "optimizer")
            if config_data.get(key) is not None
        }
        if route:
            partial_audited, partial_evidence = (
                _route_with_production_default_audit_evidence(route)
            )
            return partial_audited, "config", partial_evidence
        return {}, "missing", None


def _load_checkpoint_for_route_gates(
    parser: argparse.ArgumentParser,
    checkpoint: Path,
    *,
    require_mode_default_optimizer: bool,
    require_production_default_route: bool,
    apply_gates: bool = True,
) -> tuple[dict[str, Any], dict[str, Any], str, _ProductionRouteEvidence | None]:
    from gpurec.workflow.checkpoint import load_checkpoint

    payload = load_checkpoint(checkpoint)
    route, route_source, evidence = _checkpoint_route_metadata_evidence(payload)
    if apply_gates and require_mode_default_optimizer:
        _exit_unless_mode_default_optimizer(
            parser,
            route,
            subject="checkpoint",
            audited_route=evidence[0] if evidence is not None else None,
        )
    if apply_gates and require_production_default_route:
        _exit_unless_production_default_route(
            parser,
            route,
            subject="checkpoint",
            production_route_evidence=evidence,
        )
    return payload, route, route_source, evidence


def _apply_checkpoint_route_gates(
    parser: argparse.ArgumentParser,
    route: dict[str, Any],
    subject: str,
    *,
    require_mode_default_optimizer: bool,
    require_production_default_route: bool,
    evidence: _ProductionRouteEvidence | None,
) -> None:
    if require_mode_default_optimizer:
        _exit_unless_mode_default_optimizer(
            parser,
            route,
            subject=subject,
            audited_route=evidence[0] if evidence is not None else None,
        )
    if require_production_default_route:
        _exit_unless_production_default_route(
            parser,
            route,
            subject=subject,
            production_route_evidence=evidence,
        )


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


@dataclass
class PreparedRunConfig:
    config: RunConfig
    raw_config_data: dict[str, Any]
    summary: dict[str, object]
    route_metadata: dict[str, Any]


def _prepare_run_config_command(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    *,
    check_preprocess: bool = False,
) -> PreparedRunConfig:
    raw_config_data = _resolved_run_config_data_from_args(args)
    from gpurec.workflow.config import RunConfig

    config = RunConfig.from_dict(raw_config_data)
    route_metadata = _require_config_route_gates(
        parser,
        config,
        require_mode_default_optimizer=args.require_mode_default_optimizer,
        require_production_default_route=args.require_production_default_route,
    )
    _validate_run_config_input_paths(config)
    summary = _preflight_run_config(config, check_preprocess=check_preprocess)
    if route_metadata is None:
        route_metadata = _config_route_metadata(config)
    return PreparedRunConfig(
        config=config,
        raw_config_data=raw_config_data,
        summary=summary,
        route_metadata=route_metadata,
    )


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


# Export all public and private helper symbols for module re-export compatibility.
__all__ = [name for name in globals().keys() if not name.startswith("__")]
