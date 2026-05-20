from __future__ import annotations

from pathlib import Path
from typing import Any

from ._artifact_publish import cleanup_artifact_temp_dir


def cleanup_stage(path: Path | None) -> None:
    cleanup_error = cleanup_artifact_temp_dir(path)
    if cleanup_error is not None:
        raise cleanup_error


def cleanup_stage_after_error(
    path: Path | None,
    primary_error: BaseException,
) -> None:
    cleanup_error = cleanup_artifact_temp_dir(path)
    if cleanup_error is not None:
        raise primary_error from cleanup_error


def close_model_after_error(model: Any, primary_error: BaseException) -> None:
    try:
        model.close()
    except Exception as close_error:
        raise primary_error from close_error


def cleanup_stage_and_close_model_after_error(
    *,
    stage_dir: Path | None,
    model: Any,
    primary_error: BaseException,
) -> None:
    cleanup_error = cleanup_artifact_temp_dir(stage_dir)
    close_model_after_error(model, primary_error)
    if cleanup_error is not None:
        raise primary_error from cleanup_error
