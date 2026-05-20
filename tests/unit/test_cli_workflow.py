from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import gpurec.workflow.model_factory as workflow_model_factory
from gpurec.cli import _run_config_from_args, build_parser, main
from gpurec.workflow.config import RunConfig
from gpurec.workflow.model_factory import build_alerax_workflow_model
from tests.unit.alerax_helpers import write_tiny_alerax_inputs


def test_build_alerax_workflow_model_forwards_run_config(tmp_path: Path, monkeypatch):
    config = RunConfig(
        species_tree=tmp_path / "sp.nwk",
        families_file=tmp_path / "families.txt",
        out_dir=tmp_path / "out",
        mode="specieswise",
        device="cuda",
        dtype="float64",
        start=2,
        max_families=5,
        preprocess_cache=tmp_path / "cache",
        refresh_preprocess_cache=True,
        family_chunk_size=4,
        clade_budget=42,
        batch_packing="sequential",
        max_wave_size=64,
        fixed_iters_e=3,
        fixed_iters_pi=8,
        neumann_terms=6,
    )
    sentinel = object()
    call: dict[str, object] = {}

    def fake_from_alerax_families(*args: object, **kwargs: object) -> object:
        call["args"] = args
        call["kwargs"] = kwargs
        return sentinel

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        workflow_model_factory.GeneReconModel,
        "from_alerax_families",
        staticmethod(fake_from_alerax_families),
    )

    model = build_alerax_workflow_model(
        config,
        refresh_preprocess_cache=False,
        prefetch_batches=0,
    )

    assert model is sentinel
    assert call["args"] == (str(config.species_tree), config.families_file)
    kwargs = call["kwargs"]
    assert kwargs["mode"] == "specieswise"
    assert kwargs["start"] == 2
    assert kwargs["max_families"] == 5
    assert kwargs["device"] == "cuda"
    assert kwargs["dtype"] is torch.float64
    assert kwargs["theta_init_rates"] == config.theta_init_rates
    assert kwargs["preprocess_cache_dir"] == config.preprocess_cache
    assert kwargs["refresh_preprocess_cache"] is False
    assert kwargs["family_chunk_size"] == 4
    assert kwargs["clade_budget"] == 42
    assert kwargs["batch_packing"] == "sequential"
    assert kwargs["max_wave_size"] == 64
    assert kwargs["fixed_iters_E"] == 3
    assert kwargs["fixed_iters_Pi"] == 8
    assert kwargs["neumann_terms"] == 6
    assert kwargs["lazy_preprocess"] is True
    assert kwargs["prefetch_batches"] == 0


def test_cli_forwards_refresh_preprocess_cache(tmp_path: Path):
    write_tiny_alerax_inputs(tmp_path)
    args = build_parser().parse_args(
        [
            "optimize",
            "--species-tree",
            str(tmp_path / "sp.nwk"),
            "--families-file",
            str(tmp_path / "families.txt"),
            "--out-dir",
            str(tmp_path / "out"),
            "--device",
            "cpu",
            "--refresh-preprocess-cache",
        ]
    )

    config = _run_config_from_args(args)

    assert config.refresh_preprocess_cache is True


def test_cli_accepts_family_chunk_all_alias(tmp_path: Path):
    write_tiny_alerax_inputs(tmp_path)
    args = build_parser().parse_args(
        [
            "optimize",
            "--species-tree",
            str(tmp_path / "sp.nwk"),
            "--families-file",
            str(tmp_path / "families.txt"),
            "--out-dir",
            str(tmp_path / "out"),
            "--device",
            "cpu",
            "--family-chunk-size",
            "all",
        ]
    )

    config = _run_config_from_args(args)

    assert config.family_chunk_size == 0


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("fp32", "float32"),
        ("single", "float32"),
        ("torch.float64", "float64"),
        ("double", "float64"),
    ],
)
def test_cli_normalizes_dtype_aliases(
    tmp_path: Path,
    value: str,
    expected: str,
):
    args = build_parser().parse_args(
        _minimal_workflow_cli_args("optimize", tmp_path) + ["--dtype", value]
    )

    config = _run_config_from_args(args)

    assert config.dtype == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("contiguous", "sequential"),
        ("input-order", "sequential"),
        ("first-fit-decreasing", "clade_first_fit"),
        ("ffd", "clade_first_fit"),
        ("clade-ffd", "clade_first_fit"),
        ("depth-ffd", "depth_first_fit"),
        ("critical-path-first-fit", "depth_first_fit"),
        ("wave-first-fit", "depth_first_fit"),
    ],
)
def test_cli_normalizes_batch_packing_aliases(
    tmp_path: Path,
    value: str,
    expected: str,
):
    args = build_parser().parse_args(
        _minimal_workflow_cli_args("optimize", tmp_path)
        + ["--batch-packing", value, "--clade-budget", "123"]
    )

    config = _run_config_from_args(args)

    assert config.batch_packing == expected


def test_cli_config_paths_are_config_relative_before_flag_overrides(
    tmp_path: Path,
    monkeypatch,
):
    config_dir = tmp_path / "configs"
    input_dir = config_dir / "inputs"
    input_dir.mkdir(parents=True)
    (input_dir / "sp.nwk").write_text("(a,b);", encoding="utf-8")
    (input_dir / "families.txt").write_text("[FAMILIES]\n", encoding="utf-8")
    override_dir = tmp_path / "override"
    override_dir.mkdir()
    (override_dir / "sp.nwk").write_text("(x,y);", encoding="utf-8")
    config_path = config_dir / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "species_tree": "inputs/sp.nwk",
                "families_file": "inputs/families.txt",
                "out_dir": "runs/main",
                "device": "cpu",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    args = build_parser().parse_args(
        [
            "optimize",
            "--config",
            str(config_path),
            "--species-tree",
            "override/sp.nwk",
        ]
    )

    config = _run_config_from_args(args)

    assert config.species_tree == (tmp_path / "override" / "sp.nwk").resolve()
    assert config.families_file == (input_dir / "families.txt").resolve()
    assert config.out_dir == (config_dir / "runs" / "main").resolve()


def test_cli_config_path_expands_user_before_validation(
    tmp_path: Path,
    monkeypatch,
):
    home = tmp_path / "home"
    config_dir = home / "configs"
    input_dir = config_dir / "inputs"
    input_dir.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    (input_dir / "sp.nwk").write_text("(a,b);", encoding="utf-8")
    (input_dir / "families.txt").write_text("[FAMILIES]\n", encoding="utf-8")
    (config_dir / "run.json").write_text(
        json.dumps(
            {
                "species_tree": "inputs/sp.nwk",
                "families_file": "inputs/families.txt",
                "out_dir": "runs/main",
                "device": "cpu",
            }
        ),
        encoding="utf-8",
    )

    args = build_parser().parse_args(["optimize", "--config", "~/configs/run.json"])
    config = _run_config_from_args(args)

    assert config.species_tree == (input_dir / "sp.nwk").resolve()
    assert config.families_file == (input_dir / "families.txt").resolve()
    assert config.out_dir == (config_dir / "runs" / "main").resolve()


def _minimal_workflow_cli_args(command: str, tmp_path: Path) -> list[str]:
    write_tiny_alerax_inputs(tmp_path)
    return [
        command,
        "--species-tree",
        str(tmp_path / "sp.nwk"),
        "--families-file",
        str(tmp_path / "families.txt"),
        "--out-dir",
        str(tmp_path / "out"),
        "--device",
        "cuda",
    ]


def test_cli_rejects_auto_family_chunk_size_at_parse(capsys):
    with pytest.raises(SystemExit) as exc_info:
        build_parser().parse_args(["optimize", "--family-chunk-size", "auto"])

    captured = capsys.readouterr()
    assert exc_info.value.code == 2
    assert "family chunk size 'auto' is not supported" in captured.err
    assert "Traceback" not in captured.err


def test_cli_rejects_hydra_yaml_config_without_traceback(tmp_path: Path, capsys):
    path = tmp_path / "config.yaml"
    path.write_text("paths:\n  species_tree: sp.nwk\n", encoding="utf-8")

    with pytest.raises(SystemExit) as exc_info:
        main(["optimize", "--config", str(path)])

    captured = capsys.readouterr()
    assert exc_info.value.code == 2
    assert "flat JSON RunConfig" in captured.err
    assert "Traceback" not in captured.err


def test_cli_rejects_hydra_yaml_config_before_workflow_import(tmp_path: Path):
    path = tmp_path / "config.yaml"
    path.write_text("paths:\n  species_tree: sp.nwk\n", encoding="utf-8")
    code = f"""
import sys
from gpurec.cli import main

try:
    main(["optimize", "--config", {str(path)!r}])
except SystemExit as exc:
    print(f"exit={{exc.code}}")
else:
    print("exit=0")
print(
    "workflow_imported="
    + str(
        any(
            name == "gpurec.workflow" or name.startswith("gpurec.workflow.")
            for name in sys.modules
        )
    )
)
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[2],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "exit=2" in result.stdout
    assert "workflow_imported=False" in result.stdout
    assert "flat JSON RunConfig" in result.stderr
    assert "Traceback" not in result.stderr


def test_cli_reports_missing_json_config_without_traceback(tmp_path: Path, capsys):
    path = tmp_path / "missing.json"

    with pytest.raises(SystemExit) as exc_info:
        main(["optimize", "--config", str(path)])

    captured = capsys.readouterr()
    assert exc_info.value.code == 2
    assert "could not read config" in captured.err
    assert str(path) in captured.err
    assert "Traceback" not in captured.err


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_cli_rejects_nonstandard_json_constants_without_traceback(
    tmp_path: Path,
    capsys,
    constant: str,
):
    path = tmp_path / "config.json"
    path.write_text(
        "\n".join(
            [
                "{",
                f'  "species_tree": "{tmp_path / "sp.nwk"}",',
                f'  "families_file": "{tmp_path / "families.txt"}",',
                f'  "out_dir": "{tmp_path / "out"}",',
                f'  "tol_e": {constant}',
                "}",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as exc_info:
        main(["optimize", "--config", str(path)])

    captured = capsys.readouterr()
    assert exc_info.value.code == 2
    assert "invalid JSON config" in captured.err
    assert constant in captured.err
    assert "Traceback" not in captured.err


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("fixed_iters_pi", "64", "fixed_iters_pi"),
        ("lr", "0.01", "lr"),
        ("adaptive_iters", "false", "adaptive_iters"),
        ("species_tree", 42, "species_tree must be a path string"),
    ],
)
def test_cli_rejects_bad_typed_json_config_values_without_traceback(
    tmp_path: Path,
    capsys,
    field: str,
    value: object,
    message: str,
):
    path = tmp_path / "config.json"
    payload = {
        "species_tree": str(tmp_path / "sp.nwk"),
        "families_file": str(tmp_path / "families.txt"),
        "out_dir": str(tmp_path / "out"),
        "device": "cpu",
        field: value,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(SystemExit) as exc_info:
        main(["optimize", "--config", str(path)])

    captured = capsys.readouterr()
    assert exc_info.value.code == 2
    assert message in captured.err
    assert "Traceback" not in captured.err


def test_cli_rejects_null_required_json_path_without_traceback(
    tmp_path: Path,
    capsys,
):
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "species_tree": None,
                "families_file": str(tmp_path / "families.txt"),
                "out_dir": str(tmp_path / "out"),
                "device": "cpu",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as exc_info:
        main(["optimize", "--config", str(path)])

    captured = capsys.readouterr()
    assert exc_info.value.code == 2
    assert "species_tree" in captured.err
    assert "Traceback" not in captured.err


def test_cli_rejects_unknown_json_config_keys_without_traceback(
    tmp_path: Path,
    capsys,
):
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "species_tree": str(tmp_path / "sp.nwk"),
                "families_file": str(tmp_path / "families.txt"),
                "out_dir": str(tmp_path / "out"),
                "device": "cpu",
                "unexpected": True,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as exc_info:
        main(["optimize", "--config", str(path)])

    captured = capsys.readouterr()
    assert exc_info.value.code == 2
    assert "unknown RunConfig field" in captured.err
    assert "unexpected" in captured.err
    assert "Traceback" not in captured.err


def test_cli_reports_missing_required_options_without_traceback(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["optimize"])

    captured = capsys.readouterr()
    assert exc_info.value.code == 2
    assert "missing required optimize option" in captured.err
    assert "species_tree" in captured.err
    assert "Traceback" not in captured.err


@pytest.mark.parametrize("command", ["optimize", "run"])
@pytest.mark.parametrize(
    ("missing_file", "option"),
    [
        ("sp.nwk", "--species-tree"),
        ("families.txt", "--families-file"),
    ],
)
def test_cli_rejects_missing_input_paths_before_workflow(
    tmp_path: Path,
    capsys,
    monkeypatch,
    command: str,
    missing_file: str,
    option: str,
):
    args = _minimal_workflow_cli_args(command, tmp_path)
    missing_path = tmp_path / missing_file
    missing_path.unlink()

    def unexpected_optimize(config):
        raise AssertionError("optimize should not be called")

    monkeypatch.setattr("gpurec.cli.optimize", unexpected_optimize)

    with pytest.raises(SystemExit) as exc_info:
        main(args)

    captured = capsys.readouterr()
    assert exc_info.value.code == 2
    assert option in captured.err
    assert str(missing_path) in captured.err
    assert "CUDA" not in captured.err
    assert "Traceback" not in captured.err


@pytest.mark.parametrize("command", ["optimize", "run"])
def test_cli_rejects_missing_resume_checkpoint_before_workflow(
    tmp_path: Path,
    capsys,
    monkeypatch,
    command: str,
):
    missing_resume = tmp_path / "missing-resume.pt"

    def unexpected_optimize(config):
        raise AssertionError("optimize should not be called")

    monkeypatch.setattr("gpurec.cli.optimize", unexpected_optimize)

    with pytest.raises(SystemExit) as exc_info:
        main(
            _minimal_workflow_cli_args(command, tmp_path)
            + ["--resume-from", str(missing_resume)]
        )

    captured = capsys.readouterr()
    assert exc_info.value.code == 2
    assert "--resume-from" in captured.err
    assert str(missing_resume) in captured.err
    assert "CUDA" not in captured.err
    assert "Traceback" not in captured.err


def test_cli_sample_rejects_invalid_seed_without_traceback(tmp_path: Path, capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["sample", "--checkpoint", str(tmp_path / "best.pt"), "--seed", "-1"])

    captured = capsys.readouterr()
    assert exc_info.value.code == 2
    assert "seed must be non-negative" in captured.err
    assert "Traceback" not in captured.err


def test_cli_sample_reports_missing_checkpoint_without_traceback(tmp_path: Path, capsys):
    checkpoint = tmp_path / "missing.pt"

    with pytest.raises(SystemExit) as exc_info:
        main(["sample", "--checkpoint", str(checkpoint)])

    captured = capsys.readouterr()
    assert exc_info.value.code == 2
    assert str(checkpoint) in captured.err
    assert "Traceback" not in captured.err


def test_cli_sample_reports_empty_checkpoint_without_traceback(tmp_path: Path, capsys):
    checkpoint = tmp_path / "empty.pt"
    checkpoint.write_bytes(b"")

    with pytest.raises(SystemExit) as exc_info:
        main(["sample", "--checkpoint", str(checkpoint)])

    captured = capsys.readouterr()
    assert exc_info.value.code == 1
    assert "could not safely load checkpoint" in captured.err
    assert str(checkpoint) in captured.err
    assert "usage:" not in captured.err
    assert "Traceback" not in captured.err


def test_cli_sample_raw_theta_checkpoint_error_suggests_real_checkpoints(
    tmp_path: Path,
    capsys,
):
    checkpoint = tmp_path / "theta_final.pt"
    torch.save(torch.zeros(2, 3), checkpoint)

    with pytest.raises(SystemExit) as exc_info:
        main(["sample", "--checkpoint", str(checkpoint)])

    captured = capsys.readouterr()
    assert exc_info.value.code == 1
    assert "must contain a dictionary payload" in captured.err
    assert "checkpoints/best.pt" in captured.err
    assert "checkpoints/latest.pt" in captured.err
    assert "not theta_final.pt" in captured.err
    assert "usage:" not in captured.err
    assert "Traceback" not in captured.err


def test_cli_sample_reports_workflow_errors_without_usage(
    tmp_path: Path,
    capsys,
    monkeypatch,
):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"not used")

    def fail_sample(config):
        raise RuntimeError("sampling failed")

    monkeypatch.setattr("gpurec.cli.sample", fail_sample)

    with pytest.raises(SystemExit) as exc_info:
        main(["sample", "--checkpoint", str(checkpoint)])

    captured = capsys.readouterr()
    assert exc_info.value.code == 1
    assert "sampling failed" in captured.err
    assert "usage:" not in captured.err
    assert "Traceback" not in captured.err


def test_cli_optimize_reports_workflow_errors_without_traceback(
    tmp_path: Path,
    capsys,
    monkeypatch,
):
    def fail_optimize(config):
        raise RuntimeError("workflow failed")

    monkeypatch.setattr("gpurec.cli.optimize", fail_optimize)

    with pytest.raises(SystemExit) as exc_info:
        main(_minimal_workflow_cli_args("optimize", tmp_path))

    captured = capsys.readouterr()
    assert exc_info.value.code == 1
    assert "workflow failed" in captured.err
    assert "usage:" not in captured.err
    assert "Traceback" not in captured.err


def test_cli_optimize_failed_result_exits_nonzero_without_traceback(
    tmp_path: Path,
    capsys,
    monkeypatch,
):
    def failed_optimize(config):
        return SimpleNamespace(
            out_dir=config.out_dir,
            status="failed",
            reason="nonfinite_objective_or_gradient",
            final_nll_bits=math.inf,
        )

    monkeypatch.setattr("gpurec.cli.optimize", failed_optimize)

    with pytest.raises(SystemExit) as exc_info:
        main(_minimal_workflow_cli_args("optimize", tmp_path))

    captured = capsys.readouterr()
    assert exc_info.value.code == 1
    assert "status=failed" in captured.out
    assert "nonfinite_objective_or_gradient" in captured.out
    assert "Traceback" not in captured.err


def test_cli_run_reports_optimize_errors_without_traceback(
    tmp_path: Path,
    capsys,
    monkeypatch,
):
    def fail_optimize(config):
        raise RuntimeError("workflow failed")

    monkeypatch.setattr("gpurec.cli.optimize", fail_optimize)

    with pytest.raises(SystemExit) as exc_info:
        main(_minimal_workflow_cli_args("run", tmp_path))

    captured = capsys.readouterr()
    assert exc_info.value.code == 1
    assert "workflow failed" in captured.err
    assert "usage:" not in captured.err
    assert "Traceback" not in captured.err


def test_cli_run_reports_sampling_errors_without_usage(
    tmp_path: Path,
    capsys,
    monkeypatch,
):
    def successful_optimize(config):
        checkpoint_dir = config.out_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True)
        (checkpoint_dir / "best.pt").write_bytes(b"not used")
        return SimpleNamespace(
            out_dir=config.out_dir,
            status="success",
            reason="completed",
            final_nll_bits=12.0,
        )

    def fail_sample(config):
        raise RuntimeError("sampling failed")

    monkeypatch.setattr("gpurec.cli.optimize", successful_optimize)
    monkeypatch.setattr("gpurec.cli.sample", fail_sample)

    with pytest.raises(SystemExit) as exc_info:
        main(_minimal_workflow_cli_args("run", tmp_path))

    captured = capsys.readouterr()
    assert exc_info.value.code == 1
    assert "sampling failed" in captured.err
    assert "usage:" not in captured.err
    assert "Traceback" not in captured.err


def test_cli_run_samples_reported_checkpoint_instead_of_stale_best(
    tmp_path: Path,
    capsys,
    monkeypatch,
):
    checkpoint_dir = tmp_path / "out" / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    stale_best = checkpoint_dir / "best.pt"
    current_latest = checkpoint_dir / "latest.pt"
    stale_best.write_bytes(b"stale")
    current_latest.write_bytes(b"current")
    sampled: dict[str, Path] = {}

    def successful_optimize(config):
        return SimpleNamespace(
            out_dir=config.out_dir,
            sampling_checkpoint=current_latest,
            status="success",
            reason="completed",
            final_nll_bits=12.0,
        )

    def capture_sample(config):
        sampled["checkpoint"] = config.checkpoint
        return SimpleNamespace(
            families_sampled=1,
            samples_per_family=1,
            xml_files=1,
            out_dir=config.out_dir,
        )

    monkeypatch.setattr("gpurec.cli.optimize", successful_optimize)
    monkeypatch.setattr("gpurec.cli.sample", capture_sample)

    main(_minimal_workflow_cli_args("run", tmp_path))

    captured = capsys.readouterr()
    assert "sampled_families=1" in captured.out
    assert sampled["checkpoint"] == current_latest.resolve()


def test_cli_run_rejects_sampling_options_before_optimization(
    tmp_path: Path,
    capsys,
    monkeypatch,
):
    def unexpected_optimize(config):
        raise AssertionError("optimize should not be called")

    monkeypatch.setattr("gpurec.cli.optimize", unexpected_optimize)

    with pytest.raises(SystemExit) as exc_info:
        main(_minimal_workflow_cli_args("run", tmp_path) + ["--samples", "0"])

    captured = capsys.readouterr()
    assert exc_info.value.code == 2
    assert "samples must be positive" in captured.err
    assert "Traceback" not in captured.err


def test_cli_run_refuses_sampling_after_failed_optimization(
    tmp_path: Path,
    capsys,
    monkeypatch,
):
    checkpoint_dir = tmp_path / "out" / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "latest.pt").write_bytes(b"not used")

    def failed_optimize(config):
        return SimpleNamespace(
            out_dir=config.out_dir,
            status="failed",
            reason="nonfinite_objective_or_gradient",
            final_nll_bits=math.inf,
        )

    def unexpected_sample(config):
        raise AssertionError("sample should not be called")

    monkeypatch.setattr("gpurec.cli.optimize", failed_optimize)
    monkeypatch.setattr("gpurec.cli.sample", unexpected_sample)

    with pytest.raises(SystemExit) as exc_info:
        main(_minimal_workflow_cli_args("run", tmp_path))

    captured = capsys.readouterr()
    assert exc_info.value.code == 1
    assert "optimization failed" in captured.err
    assert "nonfinite_objective_or_gradient" in captured.err
    assert "Traceback" not in captured.err


def test_cli_run_rejects_checkpoint_argument_without_traceback(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["run", "--checkpoint", "existing.pt"])

    captured = capsys.readouterr()
    assert exc_info.value.code == 2
    assert "gpurec sample --checkpoint" in captured.err
    assert "--resume-from" in captured.err
    assert "Traceback" not in captured.err


def test_cli_run_help_omits_checkpoint_argument(capsys):
    with pytest.raises(SystemExit) as exc_info:
        build_parser().parse_args(["run", "--help"])

    captured = capsys.readouterr()
    assert exc_info.value.code == 0
    assert "--checkpoint CHECKPOINT" not in captured.out
    assert "--resume-from" in captured.out


def test_cli_optimize_help_describes_config_and_path_inputs(capsys):
    with pytest.raises(SystemExit) as exc_info:
        build_parser().parse_args(["optimize", "--help"])

    captured = capsys.readouterr()
    assert exc_info.value.code == 0
    assert "Flat JSON RunConfig" in captured.out
    assert "relative config paths" in captured.out
    assert "resolve from the config file" in captured.out
    assert "Required unless supplied by --config" in captured.out
    assert "Workflow" in captured.out
    assert "default: cuda" in captured.out
    assert "0/all/none" in captured.out


def test_cli_sample_help_describes_checkpoint_and_backtracking(capsys):
    with pytest.raises(SystemExit) as exc_info:
        build_parser().parse_args(["sample", "--help"])

    captured = capsys.readouterr()
    assert exc_info.value.code == 0
    assert "Optimization checkpoint to sample" in captured.out
    assert "checkpoints/best.pt" in captured.out
    assert "checkpoints/latest.pt" in captured.out
    assert "theta_final.pt" in captured.out
    assert "--backtrack-binary" in captured.out
    assert "GPUREC_BACKTRACK_BIN" in captured.out
    assert "Samples per selected family" in captured.out
