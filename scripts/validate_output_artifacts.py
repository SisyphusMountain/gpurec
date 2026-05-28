#!/usr/bin/env python3
"""Validate gpurec output artifacts for structural and numeric sanity.

The validator checks a subset of invariants that can be verified without
constructing CUDA models:

- summary.json: required core fields, numeric values, and route metadata shape.
- checkpoint payloads: delegated to gpurec.workflow.checkpoint.load_checkpoint().
- history.jsonl: JSON lines parsing and finite numeric checks.
- rate/per-family TSV tables: row width and finite numeric values.
- run_manifest.json: minimal schema and required nested fields.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from numbers import Integral
from typing import Any
import sys

from gpurec.workflow.checkpoint import load_checkpoint


def _is_finite_number(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number)


def _is_nonnegative_int(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, Integral):
        return False
    return value >= 0


def _append_issue(issues: list[str], artifact: str, detail: str) -> None:
    issues.append(f"[{artifact}] {detail}")


def _validate_summary(path: Path, issues: list[str]) -> None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        _append_issue(issues, path.name, f"invalid JSON: {exc}")
        return

    if not isinstance(payload, dict):
        _append_issue(issues, path.name, f"expected JSON object, got {type(payload).__name__}")
        return

    def required_text(field: str) -> None:
        value = payload.get(field)
        if not isinstance(value, str):
            _append_issue(
                issues,
                path.name,
                f"{field!r} must be a string, got {type(value).__name__}",
            )

    def required_number(field: str, *, allow_none: bool = False) -> None:
        value = payload.get(field)
        if value is None:
            if not allow_none:
                _append_issue(issues, path.name, f"{field!r} missing")
            return
        if not _is_finite_number(value):
            _append_issue(
                issues,
                path.name,
                f"{field!r} must be finite numeric, got {value!r}",
            )

    def optional_number(field: str) -> None:
        if field in payload:
            required_number(field, allow_none=True)

    def required_int(field: str, *, allow_none: bool = False) -> None:
        value = payload.get(field)
        if value is None:
            if not allow_none:
                _append_issue(issues, path.name, f"{field!r} missing")
            return
        if not _is_nonnegative_int(value):
            _append_issue(
                issues,
                path.name,
                f"{field!r} must be a non-negative integer, got {value!r}",
            )

    def optional_int(field: str, *, allow_none: bool = True) -> None:
        value = payload.get(field)
        if value is None:
            return
        if allow_none and value is None:
            return
        if not _is_nonnegative_int(value):
            _append_issue(
                issues,
                path.name,
                f"{field!r} must be a non-negative integer when present, got {value!r}",
            )

    required_text("status")
    required_text("reason")
    required_text("mode")
    required_text("optimizer")
    required_int("families")
    required_int("species")
    required_int("batches")
    required_int("steps_completed")

    required_number("final_nll_bits", allow_none=True)
    required_number("final_grad_inf", allow_none=True)
    optional_number("final_log_likelihood_bits")
    optional_number("best_nll_bits")
    optional_number("final_projected_grad_inf")
    optional_number("best_log_likelihood_bits")
    optional_number("elapsed_s")
    optional_number("final_check_iters_e")
    optional_number("final_check_loss_abs_delta_bits")
    optional_number("final_check_grad_max_abs_delta")
    optional_number("final_check_grad_rel_inf_delta")
    optional_number("final_solver_e_adjoint_rel_res_max")

    optional_int("best_step")
    optional_int("configured_steps")
    optional_int("optimizer_step_cap")
    optional_int("final_check_iters")
    optional_int("family_chunk_size")
    optional_int("fixed_iters_e")
    optional_int("fixed_iters_pi")
    optional_int("neumann_terms")
    optional_int("clade_budget", allow_none=True)
    optional_int("solver_warmup_iters")
    optional_int("fd_adam_warmup_steps")
    optional_int("fd_hessian_refresh_steps")
    optional_int("hessian_sgd_normal_fixed_iters_pi")
    optional_int("hessian_sgd_normal_neumann_terms")
    optional_int("hessian_sgd_validation_interval")
    optional_int("hessian_sgd_validation_fixed_iters_pi")
    optional_int("hessian_sgd_validation_neumann_terms")
    optional_int("adagrad_restart_total_steps")
    optional_int("adagrad_restart_final_check_iters")
    optional_int("final_solver_e_adjoint_failed_batches")
    optional_int("final_solver_e_adjoint_success_batches")

    if payload.get("mode_default_optimizer") is not None and not isinstance(
        payload.get("mode_default_optimizer"), str
    ):
        _append_issue(
            issues,
            path.name,
            "mode_default_optimizer must be a string when present",
        )
    if payload.get("uses_mode_default_optimizer") is not None and not isinstance(
        payload.get("uses_mode_default_optimizer"), bool
    ):
        _append_issue(
            issues,
            path.name,
            "uses_mode_default_optimizer must be boolean when present",
        )
    if payload.get("uses_production_default_optimizer_settings") is not None and not isinstance(
        payload.get("uses_production_default_optimizer_settings"), bool
    ):
        _append_issue(
            issues,
            path.name,
            "uses_production_default_optimizer_settings must be boolean when present",
        )
    if payload.get("uses_production_default_route") is not None and not isinstance(
        payload.get("uses_production_default_route"), bool
    ):
        _append_issue(
            issues,
            path.name,
            "uses_production_default_route must be boolean when present",
        )

    for field in (
        "mode_default_optimizer",
        "objective",
        "gradient_route",
        "rate_parameterization",
        "production_default_basis",
        "batch_packing",
    ):
        value = payload.get(field)
        if value is not None and not isinstance(value, str):
            _append_issue(
                issues,
                path.name,
                f"{field!r} must be a string when present",
            )

    sampling_checkpoint = payload.get("sampling_checkpoint")
    if sampling_checkpoint is not None and not isinstance(
        sampling_checkpoint, str
    ):
        _append_issue(
            issues,
            path.name,
            "sampling_checkpoint must be a string path or null",
        )

    mismatches = payload.get("production_default_route_mismatches")
    if mismatches is not None and not isinstance(mismatches, list):
        _append_issue(
            issues,
            path.name,
            "production_default_route_mismatches must be list or null",
        )
    optimizer_mismatches = payload.get("production_default_optimizer_setting_mismatches")
    if (
        optimizer_mismatches is not None
        and not isinstance(optimizer_mismatches, list)
    ):
        _append_issue(
            issues,
            path.name,
            "production_default_optimizer_setting_mismatches must be list or null",
        )


def _validate_history_row(value: Any, path: Path, line_no: int, issues: list[str]) -> None:
    if not isinstance(value, dict):
        _append_issue(
            issues,
            path.name,
            f"history.jsonl line {line_no}: expected JSON object, got {type(value).__name__}",
        )
        return

    step = value.get("step")
    if step is None:
        _append_issue(
            issues,
            path.name,
            f"history.jsonl line {line_no}: missing required key 'step'",
        )
    elif not _is_nonnegative_int(step):
        _append_issue(
            issues,
            path.name,
            f"history.jsonl line {line_no}: 'step' must be a non-negative integer",
        )

    def _check_numeric(row: Any, at: str) -> None:
        if isinstance(row, dict):
            for key, nested in row.items():
                _check_numeric(nested, at + f".{key}")
            return
        if isinstance(row, (list, tuple)):
            for index, nested in enumerate(row):
                _check_numeric(nested, f"{at}[{index}]")
            return
        if isinstance(row, bool):
            _append_issue(
                issues,
                path.name,
                f"history.jsonl line {line_no}{at}: boolean is invalid metric value",
            )
            return
        if isinstance(row, (int, float)):
            if not math.isfinite(float(row)):
                _append_issue(
                    issues,
                    path.name,
                    f"history.jsonl line {line_no}{at}: non-finite numeric value {row!r}",
                )
            return
        # Non-numeric values are allowed (e.g. metadata strings).

    _check_numeric(value, "")


def _validate_history(path: Path, issues: list[str]) -> None:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        _append_issue(issues, path.name, f"failed to read file: {exc}")
        return

    if not lines:
        _append_issue(issues, path.name, "history.jsonl is empty")
        return

    previous_step: int | None = None
    for line_no, line in enumerate(lines, start=1):
        if not line.strip():
            _append_issue(
                issues,
                path.name,
                f"history.jsonl line {line_no}: empty line",
            )
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            _append_issue(
                issues,
                path.name,
                f"history.jsonl line {line_no}: invalid JSON ({exc})",
            )
            continue
        _validate_history_row(payload, path, line_no, issues)
        step = payload.get("step") if isinstance(payload, dict) else None
        if _is_nonnegative_int(step):
            current_step = int(step)
            if previous_step is not None and current_step < previous_step:
                _append_issue(
                    issues,
                    path.name,
                    "history.jsonl line "
                    f"{line_no}: step {current_step} is smaller than prior step "
                    f"{previous_step}",
                )
            previous_step = current_step


def _validate_checkpoint(path: Path, issues: list[str]) -> None:
    try:
        load_checkpoint(path)
    except Exception as exc:
        _append_issue(
            issues,
            path.name,
            f"checkpoint load failed: {exc}",
        )


def _validate_tsv(path: Path, issues: list[str]) -> None:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.reader(handle, delimiter="\t"))
    except OSError as exc:
        _append_issue(issues, path.name, f"failed to read file: {exc}")
        return

    if not rows:
        _append_issue(issues, path.name, "TSV is empty")
        return

    header = rows[0]
    if not header:
        _append_issue(issues, path.name, "TSV header row is empty")
        return

    id_like_columns = {"row", "name", "family"}
    for row_idx, row in enumerate(rows[1:], start=2):
        if len(row) != len(header):
            _append_issue(
                issues,
                path.name,
                f"TSV row {row_idx} has {len(row)} columns but header has {len(header)}",
            )
            continue
        for column_idx, (column, value) in enumerate(zip(header, row), start=1):
            if column in id_like_columns:
                continue
            try:
                number = float(value)
            except ValueError:
                _append_issue(
                    issues,
                    path.name,
                    f"TSV row {row_idx}, column {column_idx} ({column!r}) is not numeric: {value!r}",
                )
                continue
            if not math.isfinite(number):
                _append_issue(
                    issues,
                    path.name,
                    f"TSV row {row_idx}, column {column_idx} ({column!r}) is not finite: {value!r}",
                )


def _validate_run_manifest(path: Path, issues: list[str]) -> None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        _append_issue(issues, path.name, f"invalid JSON: {exc}")
        return

    if not isinstance(payload, dict):
        _append_issue(
            issues,
            path.name,
            f"expected JSON object, got {type(payload).__name__}",
        )
        return

    if payload.get("schema_version") != 1:
        _append_issue(issues, path.name, "schema_version must be 1")
    if payload.get("schema_name") != "gpurec optimization run manifest":
        _append_issue(
            issues,
            path.name,
            "schema_name must be 'gpurec optimization run manifest'",
        )

    for field in ("out_dir", "command", "command_argv", "runtime", "route", "optimization", "run_config", "reproducibility"):
        if field not in payload:
            _append_issue(issues, path.name, f"missing required field {field!r}")
            continue

    if "out_dir" in payload and not isinstance(payload["out_dir"], str):
        _append_issue(issues, path.name, "out_dir must be a string")
    if payload.get("command") is not None and not isinstance(payload["command"], str):
        _append_issue(issues, path.name, "command must be a string when present")
    command_argv = payload.get("command_argv")
    if command_argv is not None:
        if not isinstance(command_argv, list) or any(
            not isinstance(item, str) for item in command_argv
        ):
            _append_issue(
                issues,
                path.name,
                "command_argv must be a list of strings when present",
            )
        elif len(command_argv) == 0:
            _append_issue(
                issues,
                path.name,
                "command_argv must not be empty when present",
            )

    run_config = payload.get("run_config")
    if isinstance(run_config, dict):
        for key in ("path", "hash_sha256", "version"):
            if key not in run_config:
                _append_issue(
                    issues,
                    path.name,
                    f"run_config missing {key!r}",
                )
            elif not isinstance(run_config[key], str):
                _append_issue(
                    issues,
                    path.name,
                    f"run_config[{key!r}] must be a string",
                )
    if payload.get("runtime") is not None and not isinstance(payload.get("runtime"), dict):
        _append_issue(issues, path.name, "runtime must be an object")

    optimization = payload.get("optimization")
    if optimization is not None and not isinstance(optimization, dict):
        _append_issue(issues, path.name, "optimization must be an object")
    if isinstance(optimization, dict):
        for key in ("mode", "optimizer", "status", "reason"):
            if key not in optimization:
                _append_issue(
                    issues,
                    path.name,
                    f"optimization missing {key!r}",
                )
            elif not isinstance(optimization[key], str):
                _append_issue(
                    issues,
                    path.name,
                    f"optimization[{key!r}] must be a string",
                )
        if "steps_completed" in optimization and not isinstance(
            optimization["steps_completed"], int
        ):
            _append_issue(
                issues,
                path.name,
                "optimization[steps_completed] must be an integer",
            )
    else:
        _append_issue(issues, path.name, "optimization must be an object")

    route = payload.get("route")
    if route is not None and not isinstance(route, dict):
        _append_issue(issues, path.name, "route must be an object")

    run_config = payload.get("run_config")
    if run_config is not None and not isinstance(run_config, dict):
        _append_issue(issues, path.name, "run_config must be an object")

    reproducibility = payload.get("reproducibility")
    if reproducibility is not None and not isinstance(
        reproducibility, dict
    ):
        _append_issue(issues, path.name, "reproducibility must be an object")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", action="append", type=Path, help="Validate summary.json")
    parser.add_argument(
        "--history", action="append", type=Path, help="Validate history.jsonl"
    )
    parser.add_argument(
        "--checkpoint", action="append", type=Path, help="Validate checkpoint .pt"
    )
    parser.add_argument("--tsv", action="append", type=Path, help="Validate a TSV artifact")
    parser.add_argument(
        "--run-manifest",
        action="append",
        type=Path,
        help="Validate run_manifest.json",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON report")
    return parser.parse_args()


def _load_json_object(path: Path, issues: list[str]) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _append_issue(issues, path.name, f"failed to parse JSON for consistency checks: {exc}")
        return None
    if not isinstance(payload, dict):
        _append_issue(
            issues,
            path.name,
            f"expected JSON object for consistency checks, got {type(payload).__name__}",
        )
        return None
    return payload


def _load_history_max_step(path: Path, issues: list[str]) -> int | None:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        _append_issue(
            issues,
            path.name,
            f"failed to read history for consistency checks: {exc}",
        )
        return None

    max_step: int | None = None
    for line in lines:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            step = row.get("step")
            if _is_nonnegative_int(step):
                current_step = int(step)
                max_step = current_step if max_step is None else max(max_step, current_step)
    return max_step


def _cross_validate_group(
    *,
    summary_path: Path | None,
    history_path: Path | None,
    run_manifest_path: Path | None,
    issues: list[str],
) -> None:
    summary_payload: dict[str, Any] | None = None
    manifest_payload: dict[str, Any] | None = None

    if summary_path is not None:
        summary_payload = _load_json_object(summary_path, issues)
    if run_manifest_path is not None:
        manifest_payload = _load_json_object(run_manifest_path, issues)

    if summary_payload is not None and history_path is not None:
        max_step = _load_history_max_step(history_path, issues)
        steps_completed = summary_payload.get("steps_completed")
        if _is_nonnegative_int(steps_completed) and max_step is not None and int(steps_completed) != max_step:
            _append_issue(
                issues,
                summary_path.name,
                "steps_completed mismatch vs history max step: "
                f"{steps_completed} != {max_step}",
            )

    if summary_payload is None or manifest_payload is None:
        return

    optimization = manifest_payload.get("optimization")
    if not isinstance(optimization, dict):
        return

    for key in ("mode", "optimizer", "status", "reason", "steps_completed"):
        if key in summary_payload and key in optimization:
            summary_value = summary_payload[key]
            manifest_value = optimization[key]
            if summary_value != manifest_value:
                _append_issue(
                    issues,
                    summary_path.name,
                    f"{key} mismatch vs run_manifest optimization: "
                    f"{summary_value!r} != {manifest_value!r}",
                )


def main() -> int:
    args = _parse_args()

    requests = [
        (args.summary, _validate_summary),
        (args.history, _validate_history),
        (args.checkpoint, _validate_checkpoint),
        (args.tsv, _validate_tsv),
        (args.run_manifest, _validate_run_manifest),
    ]

    selected = []
    for paths, _validator in requests:
        selected.extend(paths or [])

    if not selected:
        print(
            "nothing to validate; pass at least one of --summary, --history, --checkpoint, --tsv, --run-manifest",
            file=sys.stderr,
        )
        return 2

    issues: list[str] = []
    for paths, validator in requests:
        for path in paths or []:
            resolved = path.expanduser().resolve()
            if not resolved.is_file():
                _append_issue(
                    issues,
                    resolved.name,
                    f"file does not exist: {resolved}",
                )
                continue
            try:
                validator(resolved, issues)
            except Exception as exc:
                _append_issue(
                    issues,
                    resolved.name,
                    f"unexpected validation error: {exc}",
                )

    summary_by_dir = {path.expanduser().resolve().parent: path.expanduser().resolve() for path in args.summary or []}
    history_by_dir = {path.expanduser().resolve().parent: path.expanduser().resolve() for path in args.history or []}
    manifest_by_dir = {path.expanduser().resolve().parent: path.expanduser().resolve() for path in args.run_manifest or []}
    all_dirs = set(summary_by_dir) | set(history_by_dir) | set(manifest_by_dir)
    for out_dir in all_dirs:
        _cross_validate_group(
            summary_path=summary_by_dir.get(out_dir),
            history_path=history_by_dir.get(out_dir),
            run_manifest_path=manifest_by_dir.get(out_dir),
            issues=issues,
        )

    if args.json:
        print(
            json.dumps(
                {
                    "valid": len(issues) == 0,
                    "issues": issues,
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        if not issues:
            print("all requested artifacts passed validation")
        for issue in issues:
            print(issue, file=sys.stderr)

    return 0 if not issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
