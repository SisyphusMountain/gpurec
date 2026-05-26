"""Internal staged-artifact publishing support for workflow outputs.

This module is shared by optimization and sampling to publish generated files
with backup/rollback behavior. It is not a public workflow shortcut; callers
outside workflow internals should inspect final artifacts rather than depend on
the staging implementation.
"""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Callable, Iterable
from pathlib import Path


StagedArtifact = tuple[Path, Path]
_MovedArtifact = tuple[Path, Path]


def create_artifact_temp_dir(parent_dir: Path, *, prefix: str) -> Path:
    parent_dir.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=prefix, dir=parent_dir))


def cleanup_artifact_temp_dir(path: Path | None) -> OSError | None:
    if path is None:
        return None
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        return None
    except OSError as exc:
        return exc
    return None


def _restore_backup(moved: list[_MovedArtifact]) -> None:
    for backup_path, final_path in reversed(moved):
        if backup_path.is_file():
            final_path.parent.mkdir(parents=True, exist_ok=True)
            backup_path.replace(final_path)


def _move_to_backup(
    *,
    base_dir: Path,
    current_paths: Iterable[Path],
    backup_dir: Path,
) -> list[_MovedArtifact]:
    moved: list[_MovedArtifact] = []
    try:
        for final_path in current_paths:
            if not final_path.is_file():
                continue
            backup_path = backup_dir / final_path.relative_to(base_dir)
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            final_path.replace(backup_path)
            moved.append((backup_path, final_path))
    except BaseException as exc:
        restore_error: BaseException | None = None
        for backup_path, final_path in reversed(moved):
            try:
                if backup_path.is_file():
                    final_path.parent.mkdir(parents=True, exist_ok=True)
                    backup_path.replace(final_path)
            except BaseException as rollback_error:
                if restore_error is None:
                    restore_error = rollback_error
        if restore_error is not None:
            raise exc from restore_error
        raise
    return moved


def publish_staged_artifacts(
    *,
    base_dir: Path,
    staged_outputs: list[StagedArtifact],
    current_paths: Iterable[Path],
    backup_prefix: str,
    clear_current: Callable[[], None],
) -> None:
    backup_dir = create_artifact_temp_dir(base_dir, prefix=backup_prefix)
    try:
        moved = _move_to_backup(
            base_dir=base_dir,
            current_paths=list(current_paths),
            backup_dir=backup_dir,
        )
    except BaseException:
        shutil.rmtree(backup_dir, ignore_errors=True)
        raise

    try:
        for staged_path, final_path in staged_outputs:
            final_path.parent.mkdir(parents=True, exist_ok=True)
            staged_path.replace(final_path)
    except BaseException as exc:
        cleanup_error: BaseException | None = None
        try:
            clear_current()
        except BaseException as clear_error:
            cleanup_error = clear_error
        try:
            _restore_backup(moved)
        except BaseException as restore_error:
            if cleanup_error is None:
                cleanup_error = restore_error
        shutil.rmtree(backup_dir, ignore_errors=True)
        if cleanup_error is not None:
            raise exc from cleanup_error
        raise
    shutil.rmtree(backup_dir, ignore_errors=True)
