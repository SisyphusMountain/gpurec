from __future__ import annotations

from pathlib import Path

import pytest

import gpurec.workflow._cleanup as workflow_cleanup
from gpurec.workflow._artifact_publish import (
    cleanup_artifact_temp_dir,
    create_artifact_temp_dir,
    publish_staged_artifacts,
)


def _unlink_existing(paths: list[Path]) -> None:
    for path in paths:
        if path.is_file():
            path.unlink()


def test_artifact_temp_dir_cleanup_removes_private_tree(tmp_path: Path):
    stage_dir = create_artifact_temp_dir(tmp_path, prefix=".gpurec-test-stage-")
    (stage_dir / "nested").mkdir()
    (stage_dir / "nested" / "artifact.txt").write_text("staged", encoding="utf-8")

    assert cleanup_artifact_temp_dir(stage_dir) is None
    assert not stage_dir.exists()
    assert cleanup_artifact_temp_dir(None) is None


def test_cleanup_stage_after_error_chains_cleanup_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    primary = RuntimeError("primary failure")
    cleanup_failure = OSError("cleanup failure")

    monkeypatch.setattr(
        workflow_cleanup,
        "cleanup_artifact_temp_dir",
        lambda path: cleanup_failure,
    )

    with pytest.raises(RuntimeError, match="primary failure") as excinfo:
        workflow_cleanup.cleanup_stage_after_error(tmp_path, primary)

    assert excinfo.value is primary
    assert excinfo.value.__cause__ is cleanup_failure


def test_cleanup_stage_and_close_model_after_error_prefers_close_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    class FailingCloseModel:
        def __init__(self):
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1
            raise RuntimeError("close failure")

    primary = RuntimeError("primary failure")
    cleanup_failure = OSError("cleanup failure")
    model = FailingCloseModel()

    monkeypatch.setattr(
        workflow_cleanup,
        "cleanup_artifact_temp_dir",
        lambda path: cleanup_failure,
    )

    with pytest.raises(RuntimeError, match="primary failure") as excinfo:
        workflow_cleanup.cleanup_stage_and_close_model_after_error(
            stage_dir=tmp_path,
            model=model,
            primary_error=primary,
        )

    assert model.close_calls == 1
    assert excinfo.value is primary
    assert isinstance(excinfo.value.__cause__, RuntimeError)
    assert str(excinfo.value.__cause__) == "close failure"


def test_publish_staged_artifacts_replaces_nested_managed_outputs(
    tmp_path: Path,
):
    base_dir = tmp_path / "artifacts"
    final_report = base_dir / "nested" / "report.txt"
    final_added = base_dir / "nested" / "added.txt"
    manual_file = base_dir / "manual.keep"
    final_report.parent.mkdir(parents=True)
    final_report.write_text("old report", encoding="utf-8")
    manual_file.write_text("manual", encoding="utf-8")

    stage_dir = create_artifact_temp_dir(base_dir, prefix=".gpurec-test-stage-")
    staged_report = stage_dir / "nested" / "report.txt"
    staged_added = stage_dir / "nested" / "added.txt"
    staged_report.parent.mkdir(parents=True)
    staged_report.write_text("new report", encoding="utf-8")
    staged_added.write_text("new added", encoding="utf-8")
    managed_paths = [final_report, final_added]

    publish_staged_artifacts(
        base_dir=base_dir,
        staged_outputs=[
            (staged_report, final_report),
            (staged_added, final_added),
        ],
        current_paths=managed_paths,
        backup_prefix=".gpurec-test-backup-",
        clear_current=lambda: _unlink_existing(managed_paths),
    )

    assert final_report.read_text(encoding="utf-8") == "new report"
    assert final_added.read_text(encoding="utf-8") == "new added"
    assert manual_file.read_text(encoding="utf-8") == "manual"
    assert list(base_dir.glob(".gpurec-test-backup-*")) == []
    assert cleanup_artifact_temp_dir(stage_dir) is None


def test_publish_staged_artifacts_restores_backup_after_mid_publish_failure(
    tmp_path: Path,
):
    base_dir = tmp_path / "artifacts"
    final_report = base_dir / "nested" / "report.txt"
    final_summary = base_dir / "summary.json"
    final_report.parent.mkdir(parents=True)
    final_report.write_text("old report", encoding="utf-8")
    final_summary.write_text("old summary", encoding="utf-8")

    stage_dir = create_artifact_temp_dir(base_dir, prefix=".gpurec-test-stage-")
    staged_report = stage_dir / "nested" / "report.txt"
    missing_summary = stage_dir / "summary.json"
    staged_report.parent.mkdir(parents=True)
    staged_report.write_text("new report", encoding="utf-8")
    managed_paths = [final_report, final_summary]
    clear_calls: list[str] = []

    def clear_current() -> None:
        clear_calls.append("clear")
        _unlink_existing(managed_paths)

    with pytest.raises(FileNotFoundError):
        publish_staged_artifacts(
            base_dir=base_dir,
            staged_outputs=[
                (staged_report, final_report),
                (missing_summary, final_summary),
            ],
            current_paths=managed_paths,
            backup_prefix=".gpurec-test-backup-",
            clear_current=clear_current,
        )

    assert clear_calls == ["clear"]
    assert final_report.read_text(encoding="utf-8") == "old report"
    assert final_summary.read_text(encoding="utf-8") == "old summary"
    assert list(base_dir.glob(".gpurec-test-backup-*")) == []
    assert cleanup_artifact_temp_dir(stage_dir) is None
