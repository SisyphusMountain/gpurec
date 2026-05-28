#!/usr/bin/env python3
"""Run and verify the gpurec long validation workflow on a reproducible dataset."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object at {path}, got {type(payload).__name__}")
    return payload


def _resolve_path(base_dir: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _run_command(command: list[str], *, cwd: Path) -> dict[str, Any]:
    start = time.perf_counter()
    result = subprocess.run(
        command,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
    )
    elapsed_s = time.perf_counter() - start
    return {
        "command": command,
        "returncode": result.returncode,
        "elapsed_s": elapsed_s,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def _require_finite_number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{name} must be a finite number, got {value!r}")
    out = float(value)
    if not math.isfinite(out):
        raise RuntimeError(f"{name} must be finite, got {value!r}")
    return out


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="RunConfig JSON used for validate/optimize in the long-validation workflow.",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        required=True,
        help="Path to write the long-validation report JSON.",
    )
    parser.add_argument(
        "--gpurec-bin",
        default="gpurec",
        help="gpurec CLI executable path (default: gpurec).",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=2,
        help="Samples per family for reconciliation sampling.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Sampling RNG seed.",
    )
    parser.add_argument(
        "--checkpoint-choice",
        choices=("best", "latest"),
        default="best",
        help="Checkpoint to sample from after optimization.",
    )
    parser.add_argument(
        "--max-elapsed-s",
        type=float,
        default=None,
        help="Fail if summary elapsed_s exceeds this value.",
    )
    parser.add_argument(
        "--max-final-nll-bits-abs",
        type=float,
        default=None,
        help="Fail if abs(summary final_nll_bits) exceeds this value.",
    )
    parser.add_argument(
        "--min-families",
        type=int,
        default=None,
        help="Fail if summary families is below this value.",
    )
    parser.add_argument(
        "--min-species",
        type=int,
        default=None,
        help="Fail if summary species is below this value.",
    )
    parser.add_argument(
        "--skip-artifact-validator",
        action="store_true",
        help="Skip scripts/validate_output_artifacts.py (for dry-run or mocked environments).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config_path = args.config.expanduser().resolve()
    config = _load_json(config_path)
    config_dir = config_path.parent
    out_dir_raw = config.get("out_dir")
    if not isinstance(out_dir_raw, str):
        raise RuntimeError("config must define string out_dir")
    out_dir = _resolve_path(config_dir, out_dir_raw)
    summary_path = out_dir / "summary.json"
    history_path = out_dir / "history.jsonl"
    run_manifest_path = out_dir / "run_manifest.json"
    checkpoint_path = out_dir / "checkpoints" / f"{args.checkpoint_choice}.pt"
    rates_path = out_dir / "rates_final.tsv"
    per_family_path = out_dir / "per_fam_likelihoods.tsv"
    sampling_summary_path = out_dir / "reconciliations" / "summary.json"

    commands: list[dict[str, Any]] = []

    def _record(command: list[str]) -> None:
        outcome = _run_command(command, cwd=config_dir)
        commands.append(outcome)
        if outcome["returncode"] != 0:
            raise RuntimeError(
                "command failed: "
                + " ".join(command)
                + f"\nstdout:\n{outcome['stdout']}\nstderr:\n{outcome['stderr']}"
            )

    _record([args.gpurec_bin, "doctor", "--json"])
    _record(
        [
            args.gpurec_bin,
            "validate-config",
            "--config",
            str(config_path),
            "--check-preprocess",
            "--require-mode-default-optimizer",
            "--require-production-default-route",
            "--require-cuda-backward-ready",
        ]
    )
    _record(
        [
            args.gpurec_bin,
            "optimize",
            "--config",
            str(config_path),
            "--require-mode-default-optimizer",
            "--require-production-default-route",
            "--require-final-check-ok",
        ]
    )
    _record(
        [
            args.gpurec_bin,
            "summary-info",
            "--summary",
            str(summary_path),
            "--require-converged",
            "--require-final-check-ok",
            "--require-mode-default-optimizer",
            "--require-production-default-route",
        ]
    )
    _record(
        [
            args.gpurec_bin,
            "sample",
            "--checkpoint",
            str(checkpoint_path),
            "--samples",
            str(args.samples),
            "--seed",
            str(args.seed),
            "--sample-out-dir",
            str(out_dir),
        ]
    )

    if not args.skip_artifact_validator:
        validator_script = Path(__file__).resolve().parent / "validate_output_artifacts.py"
        _record(
            [
                sys.executable,
                str(validator_script),
                "--summary",
                str(summary_path),
                "--history",
                str(history_path),
                "--checkpoint",
                str(checkpoint_path),
                "--run-manifest",
                str(run_manifest_path),
                "--tsv",
                str(rates_path),
                "--tsv",
                str(per_family_path),
            ]
        )

    summary = _load_json(summary_path)
    sampling_summary = _load_json(sampling_summary_path)
    run_manifest = _load_json(run_manifest_path)

    final_nll_bits = _require_finite_number(summary.get("final_nll_bits"), name="summary.final_nll_bits")
    elapsed_s = _require_finite_number(summary.get("elapsed_s"), name="summary.elapsed_s")
    families = summary.get("families")
    species = summary.get("species")
    if not isinstance(families, int) or families < 0:
        raise RuntimeError(f"summary.families must be a non-negative integer, got {families!r}")
    if not isinstance(species, int) or species < 0:
        raise RuntimeError(f"summary.species must be a non-negative integer, got {species!r}")
    if summary.get("status") != "converged":
        raise RuntimeError(f"summary.status must be converged, got {summary.get('status')!r}")

    if args.max_elapsed_s is not None and elapsed_s > args.max_elapsed_s:
        raise RuntimeError(
            f"summary.elapsed_s {elapsed_s} exceeds threshold {args.max_elapsed_s}"
        )
    if (
        args.max_final_nll_bits_abs is not None
        and abs(final_nll_bits) > args.max_final_nll_bits_abs
    ):
        raise RuntimeError(
            "abs(summary.final_nll_bits) "
            f"{abs(final_nll_bits)} exceeds threshold {args.max_final_nll_bits_abs}"
        )
    if args.min_families is not None and families < args.min_families:
        raise RuntimeError(
            f"summary.families {families} is below minimum {args.min_families}"
        )
    if args.min_species is not None and species < args.min_species:
        raise RuntimeError(
            f"summary.species {species} is below minimum {args.min_species}"
        )

    xml_files = sampling_summary.get("xml_files")
    expected_xml = int(sampling_summary.get("families_sampled", 0)) * int(
        sampling_summary.get("samples_per_family", 0)
    )
    if not isinstance(xml_files, int) or xml_files < 0:
        raise RuntimeError(f"sampling xml_files must be a non-negative integer, got {xml_files!r}")
    if xml_files != expected_xml:
        raise RuntimeError(
            f"sampling xml_files {xml_files} does not match expected {expected_xml}"
        )

    report = {
        "schema": "gpurec.long_validation_report.v1",
        "generated_at_unix_s": time.time(),
        "environment": {
            "python_executable": sys.executable,
            "cwd": str(Path.cwd()),
            "gpurec_bin": args.gpurec_bin,
            "checkpoint_choice": args.checkpoint_choice,
            "artifact_validator_enabled": not args.skip_artifact_validator,
        },
        "inputs": {
            "config": str(config_path),
            "out_dir": str(out_dir),
            "samples": args.samples,
            "seed": args.seed,
            "max_elapsed_s": args.max_elapsed_s,
            "max_final_nll_bits_abs": args.max_final_nll_bits_abs,
            "min_families": args.min_families,
            "min_species": args.min_species,
        },
        "commands": commands,
        "observed": {
            "families": families,
            "species": species,
            "final_nll_bits": final_nll_bits,
            "elapsed_s": elapsed_s,
            "sampling_xml_files": xml_files,
            "sampling_expected_xml_files": expected_xml,
            "status": summary.get("status"),
            "reason": summary.get("reason"),
            "run_manifest_route": run_manifest.get("route"),
        },
    }

    output_path = args.output_report.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(str(output_path), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
