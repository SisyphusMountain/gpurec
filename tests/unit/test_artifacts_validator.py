from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch

from gpurec.workflow.checkpoint import save_checkpoint
from gpurec.workflow.config import RunConfig

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "validate_output_artifacts.py"


def _run_validator(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )


def test_validate_summary_accepts_nullable_objective_metrics(tmp_path: Path) -> None:
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "status": "failed",
                "reason": "nonfinite_objective_or_gradient",
                "mode": "genewise",
                "optimizer": "hessian-sgd",
                "families": 1,
                "species": 1,
                "batches": 1,
                "steps_completed": 12,
                "final_nll_bits": None,
                "final_grad_inf": None,
                "final_projected_grad_inf": 0.0,
                "uses_mode_default_optimizer": True,
                "uses_production_default_optimizer_settings": True,
                "uses_production_default_route": True,
                "production_default_optimizer_setting_mismatches": [],
                "production_default_route_mismatches": [],
                "mode_default_optimizer": "hessian-sgd",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    result = _run_validator("--summary", str(summary), "--json")
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert report["valid"]
    assert report["issues"] == []


def test_validate_summary_rejects_invalid_fields(tmp_path: Path) -> None:
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "status": "failed",
                "reason": "nonfinite_objective_or_gradient",
                "mode": "genewise",
                "optimizer": "hessian-sgd",
                "families": 1,
                "species": 1,
                "batches": 1,
                "steps_completed": 12,
                "final_nll_bits": "bad",
                "final_grad_inf": 0.5,
                "uses_production_default_route": "yes",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    result = _run_validator("--summary", str(summary), "--json")
    assert result.returncode == 1
    report = json.loads(result.stdout)
    assert not report["valid"]
    assert any("final_nll_bits" in issue for issue in report["issues"])
    assert any(
        "uses_production_default_route must be boolean" in issue
        for issue in report["issues"]
    )


def test_validate_history_rejects_nonfinite_values(tmp_path: Path) -> None:
    history = tmp_path / "history.jsonl"
    history.write_text(
        "\n".join(
            [
                json.dumps({"step": 0, "value": 1.0}),
                json.dumps({"step": 1, "value": float("nan")}),
            ]
        ),
        encoding="utf-8",
    )

    result = _run_validator("--history", str(history), "--json")
    assert result.returncode == 1
    report = json.loads(result.stdout)
    assert not report["valid"]
    assert any("non-finite numeric" in issue for issue in report["issues"])


def test_validate_history_rejects_non_monotonic_step_sequence(tmp_path: Path) -> None:
    history = tmp_path / "history.jsonl"
    history.write_text(
        "\n".join(
            [
                json.dumps({"step": 0, "value": 1.0}),
                json.dumps({"step": 3, "value": 0.5}),
                json.dumps({"step": 2, "value": 0.4}),
            ]
        ),
        encoding="utf-8",
    )

    result = _run_validator("--history", str(history), "--json")
    assert result.returncode == 1
    report = json.loads(result.stdout)
    assert not report["valid"]
    assert any("smaller than prior step" in issue for issue in report["issues"])


def test_validate_tsv_rejects_non_numeric_values(tmp_path: Path) -> None:
    rates = tmp_path / "rates.tsv"
    rates.write_text("name\tD\tL\nrow\t1.0\tnot-number\n", encoding="utf-8")

    result = _run_validator("--tsv", str(rates), "--json")
    assert result.returncode == 1
    report = json.loads(result.stdout)
    assert not report["valid"]
    assert any("not numeric" in issue for issue in report["issues"])


def test_validate_run_manifest_requires_required_fields(tmp_path: Path) -> None:
    manifest = tmp_path / "run_manifest.json"
    payload = {
        "schema_version": 1,
        "schema_name": "gpurec optimization run manifest",
        "out_dir": str(tmp_path / "output"),
        "command": "gpurec optimize",
        "command_argv": ["python", "-m", "gpurec"],
        "runtime": {
            "python": "3.12",
        },
        "route": {
            "mode": "genewise",
            "optimizer": "hessian-sgd",
        },
        "optimization": {
            "mode": "genewise",
            "optimizer": "hessian-sgd",
            "status": "converged",
            "reason": "max_steps",
            "steps_completed": 10,
            "families": 3,
            "species": 2,
            "batches": 1,
            "final_nll_bits": 5.0,
            "final_log_likelihood_bits": -5.0,
            "best_nll_bits": 4.5,
            "sampling_checkpoint": "checkpoints/latest.pt",
            "final_check_status": "ok",
        },
        "run_config": {
            "path": "run_config.json",
            "hash_sha256": "deadbeef",
            "version": "1",
        },
        "reproducibility": {
            "torch_seed": 123,
            "seeded": True,
        },
    }
    manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    valid = _run_validator("--run-manifest", str(manifest), "--json")
    assert valid.returncode == 0
    report = json.loads(valid.stdout)
    assert report["valid"]

    broken = payload.copy()
    del broken["run_config"]
    bad_manifest = tmp_path / "bad_run_manifest.json"
    bad_manifest.write_text(json.dumps(broken, indent=2), encoding="utf-8")

    invalid = _run_validator("--run-manifest", str(bad_manifest), "--json")
    assert invalid.returncode == 1
    invalid_report = json.loads(invalid.stdout)
    assert not invalid_report["valid"]
    assert any("missing required field 'run_config'" in issue for issue in invalid_report["issues"])


def test_validate_run_manifest_rejects_empty_command_argv(tmp_path: Path) -> None:
    manifest = tmp_path / "run_manifest.json"
    payload = {
        "schema_version": 1,
        "schema_name": "gpurec optimization run manifest",
        "out_dir": str(tmp_path / "output"),
        "command": "gpurec optimize",
        "command_argv": [],
        "runtime": {"python": "3.12"},
        "route": {"mode": "genewise", "optimizer": "hessian-sgd"},
        "optimization": {
            "mode": "genewise",
            "optimizer": "hessian-sgd",
            "status": "converged",
            "reason": "max_steps",
            "steps_completed": 1,
        },
        "run_config": {
            "path": "run_config.json",
            "hash_sha256": "deadbeef",
            "version": "1",
        },
        "reproducibility": {"seeded": True},
    }
    manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    result = _run_validator("--run-manifest", str(manifest), "--json")
    assert result.returncode == 1
    report = json.loads(result.stdout)
    assert not report["valid"]
    assert any("command_argv must not be empty" in issue for issue in report["issues"])


def test_validate_checkpoint_accepts_valid_checkpoint_and_rejects_invalid_payload(tmp_path: Path) -> None:
    model = type("model", (), {"theta": torch.tensor([1.0, 2.0], dtype=torch.float32)})()
    config = RunConfig(
        species_tree=tmp_path / "species.nwk",
        families_file=tmp_path / "families.txt",
        out_dir=tmp_path / "out",
    )
    (tmp_path / "species.nwk").write_text("(A,B);", encoding="utf-8")
    (tmp_path / "families.txt").write_text("f1\n", encoding="utf-8")

    valid_checkpoint = tmp_path / "checkpoint.pt"
    save_checkpoint(
        valid_checkpoint,
        config=config,
        model=model,
        optimizer=None,
        step=1,
        status={"status": "converged"},
        row={"step": 1},
        next_step=2,
    )

    valid = _run_validator("--checkpoint", str(valid_checkpoint), "--json")
    assert valid.returncode == 0
    assert json.loads(valid.stdout)["valid"]

    broken = tmp_path / "broken_checkpoint.pt"
    torch.save({"version": 1}, broken)

    invalid = _run_validator("--checkpoint", str(broken), "--json")
    assert invalid.returncode == 1
    assert any(
        "missing key(s)" in issue for issue in json.loads(invalid.stdout)["issues"]
    )


def test_validate_history_rejects_empty_file(tmp_path: Path) -> None:
    history = tmp_path / "history.jsonl"
    history.write_text("", encoding="utf-8")
    result = _run_validator("--history", str(history), "--json")
    assert result.returncode == 1
    issues = json.loads(result.stdout)["issues"]
    assert any("is empty" in issue for issue in issues)
