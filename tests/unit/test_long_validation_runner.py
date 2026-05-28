from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "run_long_validation.py"


def _write_fake_gpurec(path: Path) -> None:
    path.write_text(
        """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

argv = sys.argv[1:]
if not argv:
    raise SystemExit(2)
cmd = argv[0]
if cmd == "doctor":
    print(json.dumps({"ready": True, "checks": {"python": {"ok": True}}}))
    raise SystemExit(0)
if cmd == "validate-config":
    raise SystemExit(0)
if cmd == "optimize":
    config = Path(argv[argv.index("--config") + 1])
    payload = json.loads(config.read_text(encoding="utf-8"))
    out_dir = Path(payload["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "converged",
        "reason": "max_steps",
        "mode": "genewise",
        "optimizer": "hessian-sgd",
        "families": 2,
        "species": 261,
        "batches": 1,
        "steps_completed": 1,
        "final_nll_bits": 12.5,
        "final_grad_inf": 0.01,
        "elapsed_s": 10.0,
        "uses_mode_default_optimizer": True,
        "uses_production_default_route": True,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (out_dir / "history.jsonl").write_text(json.dumps({"step": 0, "loss": 1.0}) + "\\n", encoding="utf-8")
    (out_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "schema_name": "gpurec optimization run manifest",
                "route": {"mode": "genewise"},
            }
        ),
        encoding="utf-8",
    )
    (out_dir / "rates_final.tsv").write_text("name\\tD\\tT\\tL\\nrow\\t1\\t1\\t1\\n", encoding="utf-8")
    (out_dir / "per_fam_likelihoods.tsv").write_text("family\\tnll_bits\\nf1\\t1\\n", encoding="utf-8")
    for ckpt in ("best.pt", "latest.pt"):
        (out_dir / "checkpoints" / ckpt).write_text("fake", encoding="utf-8")
    raise SystemExit(0)
if cmd == "summary-info":
    raise SystemExit(0)
if cmd == "sample":
    ckpt = Path(argv[argv.index("--checkpoint") + 1])
    out_dir = Path(argv[argv.index("--sample-out-dir") + 1])
    recon = out_dir / "reconciliations"
    recon.mkdir(parents=True, exist_ok=True)
    summary = {
        "families_sampled": 2,
        "samples_per_family": 2,
        "xml_files": 4,
        "checkpoint": str(ckpt),
        "out_dir": str(out_dir),
    }
    (recon / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    raise SystemExit(0)
raise SystemExit(2)
""",
        encoding="utf-8",
    )
    path.chmod(0o755)


def test_long_validation_runner_writes_report(tmp_path: Path) -> None:
    fake = tmp_path / "gpurec"
    _write_fake_gpurec(fake)
    out_dir = tmp_path / "out"
    config = tmp_path / "run.json"
    config.write_text(json.dumps({"out_dir": str(out_dir)}), encoding="utf-8")
    report = tmp_path / "report.json"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--config",
            str(config),
            "--output-report",
            str(report),
            "--gpurec-bin",
            str(fake),
            "--skip-artifact-validator",
            "--samples",
            "2",
            "--seed",
            "7",
            "--min-families",
            "2",
            "--min-species",
            "200",
            "--max-elapsed-s",
            "100",
            "--max-final-nll-bits-abs",
            "1000",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["schema"] == "gpurec.long_validation_report.v1"
    assert payload["observed"]["status"] == "converged"
    assert payload["observed"]["sampling_xml_files"] == 4
    assert payload["observed"]["sampling_expected_xml_files"] == 4
    assert len(payload["commands"]) >= 5


def test_long_validation_runner_enforces_thresholds(tmp_path: Path) -> None:
    fake = tmp_path / "gpurec"
    _write_fake_gpurec(fake)
    out_dir = tmp_path / "out"
    config = tmp_path / "run.json"
    config.write_text(json.dumps({"out_dir": str(out_dir)}), encoding="utf-8")
    report = tmp_path / "report.json"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--config",
            str(config),
            "--output-report",
            str(report),
            "--gpurec-bin",
            str(fake),
            "--skip-artifact-validator",
            "--max-elapsed-s",
            "1",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "exceeds threshold" in result.stderr or "exceeds threshold" in result.stdout
