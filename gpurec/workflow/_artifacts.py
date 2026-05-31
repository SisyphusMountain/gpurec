from __future__ import annotations

from pathlib import Path
from typing import Any

import csv
import datetime
import hashlib
import json
import os
import platform
import sys

import torch

from gpurec.api.model import GeneReconModel

from ._artifact_publish import (
    StagedArtifact,
    create_artifact_temp_dir,
    publish_staged_artifacts,
)
from ._cleanup import cleanup_stage, cleanup_stage_after_error
from ._metadata import model_family_names, model_species_names
from .config import RunConfig
from .diagnostics import (
    json_dumps_strict,
    rates_and_survival_probability,
    write_csv,
    write_json_strict,
)


_FINAL_CHECK_SUMMARY_FIELDS = (
    ("optimizer/final_check_iters_E", "final_check_iters_e"),
    ("optimizer/final_check_status", "final_check_status"),
    ("optimizer/final_check_source", "final_check_source"),
    ("optimizer/final_check_reason", "final_check_reason"),
    (
        "optimizer/final_check_fallback_clade_budget",
        "final_check_fallback_clade_budget",
    ),
    ("optimizer/final_check_loss_abs_delta_bits", "final_check_loss_abs_delta_bits"),
    ("optimizer/final_check_grad_max_abs_delta", "final_check_grad_max_abs_delta"),
    ("optimizer/final_check_grad_rel_inf_delta", "final_check_grad_rel_inf_delta"),
)

_FINAL_SOLVER_SUMMARY_FIELDS = (
    ("solver/e_adjoint_failed_batches", "final_solver_e_adjoint_failed_batches"),
    ("solver/e_adjoint_success_batches", "final_solver_e_adjoint_success_batches"),
    ("solver/e_adjoint_rel_res_max", "final_solver_e_adjoint_rel_res_max"),
)

_FINAL_ARTIFACT_FILES = (
    "history.jsonl",
    "rates_final.tsv",
    "per_fam_likelihoods.tsv",
    "theta_final.pt",
    "optimization_history.csv",
    "summary.json",
    "run_manifest.json",
)
_RUN_CONFIG_ARTIFACT_FILE = "run_config.json"
_RUN_MANIFEST_ARTIFACT_FILE = "run_manifest.json"
_TORCH_SEED_ENV = "GPUREC_TORCH_SEED"
_MAX_TORCH_SEED = (1 << 63) - 1


def _run_manifest_native_artifacts_info() -> dict[str, Any]:
    from gpurec.core import preprocess_rust
    from gpurec import backtracking

    preprocess_version = preprocess_rust._manifest_version()
    backtrack_version = backtracking._manifest_version()

    preprocess_path = os.environ.get("GPUREC_PREPROCESS_NATIVE_LIB")
    backtrack_path = os.environ.get("GPUREC_BACKTRACK_BIN")

    return {
        "preprocess": {
            "path": (
                str(Path(preprocess_path).expanduser().resolve())
                if preprocess_path is not None
                else None
            ),
            "manifest_version": preprocess_version,
        },
        "backtracking": {
            "path": (
                str(Path(backtrack_path).expanduser().resolve())
                if backtrack_path is not None
                else None
            ),
            "manifest_version": backtrack_version,
        },
    }


def _final_check_summary_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        summary_key: metrics[metric_key]
        for metric_key, summary_key in _FINAL_CHECK_SUMMARY_FIELDS
        if metric_key in metrics
    }


def _final_solver_summary_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        summary_key: metrics[metric_key]
        for metric_key, summary_key in _FINAL_SOLVER_SUMMARY_FIELDS
        if metric_key in metrics
    }


def _readiness_text(value: Any, fallback: Any) -> Any:
    return fallback if value is None else value


def _run_manifest_runtime_info(start_time_s: float) -> dict[str, Any]:
    created_utc = datetime.datetime.now(datetime.timezone.utc)
    try:
        import torch
    except Exception:  # pragma: no cover - exercised in integration and guarded by monkeypatches
        torch = None

    try:
        import triton
    except Exception:  # pragma: no cover - exercised in integration and guarded by monkeypatches
        triton = None

    artifact_paths = {
        "preprocess_native_lib": os.environ.get("GPUREC_PREPROCESS_NATIVE_LIB"),
        "backtrack_binary": os.environ.get("GPUREC_BACKTRACK_BIN"),
    }
    artifact_readiness = {
        name: None if path is None else str(path)
        for name, path in artifact_paths.items()
    }

    torch_info: dict[str, Any] = {
        "version": _readiness_text(torch.__version__, None) if torch is not None else None,
        "cuda": _readiness_text(torch.version.cuda if torch is not None else None, None),
        "cuda_available": (
            None if torch is None else bool(torch.cuda.is_available())
        ),
        "devices": None if torch is None else int(torch.cuda.device_count()),
        "current_device": (
            None
            if torch is None
            or not torch.cuda.is_available()
            else int(torch.cuda.current_device())
        ),
    }
    if torch_info["current_device"] is not None:
        torch_info["current_device_name"] = _readiness_text(
            torch.cuda.get_device_name(torch.cuda.current_device()),
            "",
        )

    triton_info: dict[str, Any] = {
        "version": getattr(triton, "__version__", None)
        if triton is not None
        else None,
    }
    uname = platform.uname()

    return {
        "created_utc": created_utc.isoformat().replace("+00:00", "Z"),
        "package_version": __import__("gpurec", fromlist=["__version__"]).__version__,
        "started_s": float(start_time_s),
        "started_dt_utc": datetime.datetime.fromtimestamp(
            start_time_s,
            tz=datetime.timezone.utc,
        ).isoformat().replace("+00:00", "Z"),
        "platform": {
            "name": uname.system,
            "node": uname.node,
            "release": uname.release,
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": getattr(platform, "python_executable", lambda: sys.executable)(),
        },
        "torch": torch_info,
        "triton": triton_info,
        "native_artifacts": {
            **artifact_readiness,
            **_run_manifest_native_artifacts_info(),
        },
    }


def _run_manifest_hash(config: RunConfig) -> str:
    payload = json.dumps(config.to_dict(), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _runtime_seed_context_from_environment() -> dict[str, Any]:
    raw = os.environ.get(_TORCH_SEED_ENV)
    if raw is None:
        return {
            "seeded": False,
            "seed_source": None,
            "requested_torch_seed": None,
        }
    value = raw.strip()
    if not value:
        raise ValueError(f"{_TORCH_SEED_ENV} must be a non-empty integer")
    try:
        seed = int(value, 10)
    except ValueError as exc:
        raise ValueError(f"{_TORCH_SEED_ENV} must be an integer") from exc
    if seed < 0:
        raise ValueError(f"{_TORCH_SEED_ENV} must be non-negative")
    if seed > _MAX_TORCH_SEED:
        raise ValueError(f"{_TORCH_SEED_ENV} must be <= {_MAX_TORCH_SEED}")
    torch.manual_seed(seed)
    return {
        "seeded": True,
        "seed_source": _TORCH_SEED_ENV,
        "requested_torch_seed": seed,
    }


def _build_run_manifest(
    config: RunConfig,
    *,
    command: str | None,
    command_argv: tuple[str, ...] | list[str] | None,
    route_metadata: dict[str, Any],
    summary: dict[str, Any],
    started_wall_time: float,
    elapsed_wall_s: float,
    runtime_seed_context: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "schema_name": "gpurec optimization run manifest",
        "out_dir": str(config.out_dir),
        "command": command,
        "command_argv": None if command_argv is None else list(command_argv),
        "run_config": {
            "path": str(config.out_dir / _RUN_CONFIG_ARTIFACT_FILE),
            "hash_sha256": _run_manifest_hash(config),
            "version": "1",
        },
        "runtime": _run_manifest_runtime_info(started_wall_time),
        "elapsed_s": elapsed_wall_s,
        "route": route_metadata,
        "optimization": {
            "mode": summary.get("mode"),
            "optimizer": summary.get("optimizer"),
            "status": summary.get("status"),
            "reason": summary.get("reason"),
            "steps_completed": summary.get("steps_completed"),
            "families": summary.get("families"),
            "species": summary.get("species"),
            "batches": summary.get("batches"),
            "final_nll_bits": summary.get("final_nll_bits"),
            "final_log_likelihood_bits": summary.get("final_log_likelihood_bits"),
            "best_nll_bits": summary.get("best_nll_bits"),
            "sampling_checkpoint": summary.get("sampling_checkpoint"),
            "final_check_status": summary.get("final_check_status"),
        },
        "reproducibility": {
            "torch_seed": int(torch.initial_seed()),
            "seeded": bool(runtime_seed_context.get("seeded", False)),
            "seed_source": runtime_seed_context.get("seed_source"),
            "requested_torch_seed": runtime_seed_context.get("requested_torch_seed"),
        },
        "selections": {
            "families": summary.get("families"),
            "species": summary.get("species"),
        },
    }


def _write_history_jsonl_with_final_row(
    path: Path,
    current_history_path: Path,
    final_row: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out_handle:
        if current_history_path.is_file():
            existing = current_history_path.read_text(encoding="utf-8")
            out_handle.write(existing)
            if existing and not existing.endswith("\n"):
                out_handle.write("\n")
        out_handle.write(json_dumps_strict(final_row, sort_keys=True) + "\n")


def _final_artifact_paths(out_dir: Path) -> list[Path]:
    return [
        path
        for name in _FINAL_ARTIFACT_FILES
        if (path := out_dir / name).is_file()
    ]


def _clear_final_artifacts(out_dir: Path) -> None:
    for path in _final_artifact_paths(out_dir):
        path.unlink()


def _publish_final_artifacts(
    out_dir: Path,
    staged_outputs: list[StagedArtifact],
) -> None:
    publish_staged_artifacts(
        base_dir=out_dir,
        staged_outputs=staged_outputs,
        current_paths=_final_artifact_paths(out_dir),
        backup_prefix=".gpurec-optimization-backup-",
        clear_current=lambda: _clear_final_artifacts(out_dir),
    )


def _parameter_labels(
    model: GeneReconModel,
    mode: str,
    *,
    theta_rows: int,
) -> list[str]:
    if mode == "genewise":
        labels = model_family_names(model)
        label_kind = "family"
    elif mode == "specieswise":
        labels = model_species_names(model)
        label_kind = "species"
    elif mode == "global":
        if theta_rows != 1:
            raise RuntimeError(
                f"global rate table has {theta_rows} theta rows; expected 1"
            )
        return ["global"]
    else:
        raise RuntimeError(f"unsupported rate-table mode {mode!r}")
    label_count = len(labels)
    if label_count < theta_rows:
        raise RuntimeError(
            f"{mode} rate table has {theta_rows} theta rows but only "
            f"{label_count} {label_kind} labels"
        )
    if label_count > theta_rows:
        raise RuntimeError(
            f"{mode} rate table has {theta_rows} theta rows but "
            f"{label_count} {label_kind} labels; expected one theta row per "
            f"{label_kind}"
        )
    return labels


def _rate_table_theta(model: GeneReconModel, mode: str) -> torch.Tensor:
    theta = model.theta.detach()
    theta_shape = tuple(int(dim) for dim in theta.shape)
    if mode == "global":
        if theta_shape == (3,):
            theta = theta.reshape(1, 3)
        elif theta.ndim == 2 and int(theta.shape[1]) == 3:
            pass
        else:
            raise RuntimeError(
                "global rate table theta has shape "
                f"{theta_shape}; expected (3,) or a two-dimensional [rows, 3] tensor"
            )
    elif mode in {"genewise", "specieswise"}:
        if theta.ndim != 2 or int(theta.shape[1]) != 3:
            raise RuntimeError(
                f"{mode} rate table theta has shape {theta_shape}; "
                "expected a two-dimensional [rows, 3] tensor"
            )
    else:
        raise RuntimeError(f"unsupported rate-table mode {mode!r}")
    theta = theta.to(device="cpu", dtype=torch.float64)
    if not bool(torch.isfinite(theta).all().item()):
        raise RuntimeError(f"{mode} rate table theta contains nonfinite values")
    return theta


def _write_rate_table(path: Path, model: GeneReconModel, mode: str) -> None:
    theta = _rate_table_theta(model, mode)
    labels = _parameter_labels(model, mode, theta_rows=int(theta.shape[0]))
    rates, p_s = rates_and_survival_probability(theta)
    if not bool(torch.isfinite(rates).all().item()) or not bool(
        torch.isfinite(p_s).all().item()
    ):
        raise RuntimeError(f"{mode} rate table contains nonfinite rates")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            (
                "row",
                "name",
                "D",
                "T",
                "L",
                "pS",
                "theta_D",
                "theta_T",
                "theta_L",
            )
        )
        for row, label in enumerate(labels):
            theta_row = 0 if theta.shape[0] == 1 else row
            writer.writerow(
                (
                    row,
                    label,
                    float(rates[theta_row, 0]),
                    float(rates[theta_row, 2]),
                    float(rates[theta_row, 1]),
                    float(p_s[theta_row]),
                    float(theta[theta_row, 0]),
                    float(theta[theta_row, 2]),
                    float(theta[theta_row, 1]),
                )
            )


@torch.no_grad()
def _per_family_nll(
    model: GeneReconModel,
    values: torch.Tensor | None = None,
) -> list[tuple[str, float]]:
    if values is None:
        values = model.full_nll_per_family()
    family_names = model_family_names(model)
    if not torch.is_tensor(values):
        raise RuntimeError("per-family likelihood vector must be a tensor")
    expected_shape = (len(family_names),)
    actual_shape = tuple(int(dim) for dim in values.shape)
    if actual_shape != expected_shape:
        raise RuntimeError(
            "per-family likelihood vector has shape "
            f"{actual_shape}, expected {expected_shape}"
        )
    values = values.detach().cpu()
    if not bool(torch.isfinite(values).all().item()):
        raise RuntimeError("per-family likelihood vector contains nonfinite values")
    return list(zip(family_names, values.tolist()))


def _write_per_family_likelihoods(
    path: Path,
    model: GeneReconModel,
    values: torch.Tensor | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("family", "nll_bits", "log_likelihood_bits"))
        for family, nll in _per_family_nll(model, values):
            writer.writerow((family, f"{nll:.12g}", f"{-nll:.12g}"))


def _write_final_artifacts(
    config: RunConfig,
    *,
    model: GeneReconModel,
    history: list[dict[str, Any]],
    final_row: dict[str, Any],
    summary: dict[str, Any],
    run_manifest: dict[str, Any] | None = None,
    history_jsonl: Path,
    per_family_nll: torch.Tensor | None = None,
    include_per_family_likelihoods: bool = True,
) -> None:
    stage_dir: Path | None = None
    try:
        stage_dir = create_artifact_temp_dir(
            config.out_dir,
            prefix=".gpurec-optimization-stage-",
        )
        staged_outputs: list[StagedArtifact] = []

        history_stage_path = stage_dir / "history.jsonl"
        _write_history_jsonl_with_final_row(
            history_stage_path,
            history_jsonl,
            final_row,
        )
        history_jsonl_output = (history_stage_path, history_jsonl)

        rates_stage_path = stage_dir / "rates_final.tsv"
        _write_rate_table(rates_stage_path, model, config.mode)
        staged_outputs.append((rates_stage_path, config.out_dir / "rates_final.tsv"))

        if config.mode == "genewise" and include_per_family_likelihoods:
            per_family_stage_path = stage_dir / "per_fam_likelihoods.tsv"
            _write_per_family_likelihoods(
                per_family_stage_path,
                model,
                per_family_nll,
            )
            staged_outputs.append(
                (
                    per_family_stage_path,
                    config.out_dir / "per_fam_likelihoods.tsv",
                )
            )

        theta_stage_path = stage_dir / "theta_final.pt"
        torch.save(model.theta.detach().cpu(), theta_stage_path)
        staged_outputs.append((theta_stage_path, config.out_dir / "theta_final.pt"))

        history_csv_stage_path = stage_dir / "optimization_history.csv"
        write_csv(history_csv_stage_path, history)
        staged_outputs.append(
            (history_csv_stage_path, config.out_dir / "optimization_history.csv")
        )

        summary_stage_path = stage_dir / "summary.json"
        write_json_strict(summary_stage_path, summary)
        staged_outputs.append(history_jsonl_output)
        staged_outputs.append((summary_stage_path, config.out_dir / "summary.json"))
        if run_manifest is not None:
            manifest_stage_path = stage_dir / _RUN_MANIFEST_ARTIFACT_FILE
            write_json_strict(manifest_stage_path, run_manifest)
            staged_outputs.append(
                (
                    manifest_stage_path,
                    config.out_dir / _RUN_MANIFEST_ARTIFACT_FILE,
                )
            )

        _publish_final_artifacts(config.out_dir, staged_outputs)
    except BaseException as exc:
        cleanup_stage_after_error(stage_dir, exc)
        raise
    else:
        cleanup_stage(stage_dir)
