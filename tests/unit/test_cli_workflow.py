from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import gpurec.cli as gpurec_cli
import gpurec.workflow.model_factory as workflow_model_factory
from gpurec.cli import (
    _run_config_cli_override_fields,
    _run_config_from_args,
    build_parser,
    main,
)
from gpurec.workflow.config import RunConfig, SamplingConfig, effective_route_metadata
from gpurec.workflow.checkpoint import CHECKPOINT_VERSION, save_checkpoint
from gpurec.workflow.model_factory import build_alerax_workflow_model
from tests.unit.alerax_helpers import write_tiny_alerax_inputs

SUBPROCESS_TIMEOUT = 30
_PRODUCTION_ROUTE_CONTRACT: dict[str, object] = {
    "objective": "negative_log_likelihood_bits",
    "gradient_route": "implicit_first_order_adjoint",
    "rate_parameterization": "base2_log_dlt_rates",
    "production_default_basis": "hogenom_and_" + "test_trees_" + "1000",
}


def _parser_action_dests(command: str) -> set[str]:
    parser = build_parser()
    subparsers = next(
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    return {
        action.dest
        for action in subparsers.choices[command]._actions
        if action.dest not in (argparse.SUPPRESS, "help")
        and action.help is not argparse.SUPPRESS
    }


def _parser_action(command: str, dest: str) -> argparse.Action:
    parser = build_parser()
    subparsers = next(
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    for action in subparsers.choices[command]._actions:
        if action.dest == dest:
            return action
    raise AssertionError(f"{command} has no parser action {dest!r}")


def _checkpoint_with_route_metadata(
    tmp_path: Path,
    route_metadata: dict[str, object],
) -> Path:
    class FakeModel:
        theta = torch.nn.Parameter(torch.zeros(1, 3))
        family_names = ["fam0"]
        species_names = ["sp0", "sp1"]

    config = RunConfig(
        species_tree=tmp_path / "sp.nwk",
        families_file=tmp_path / "families.txt",
        out_dir=tmp_path / "out",
        mode="genewise",
        device="cpu",
    )
    checkpoint = tmp_path / "latest.pt"
    save_checkpoint(
        checkpoint,
        config=config,
        model=FakeModel(),
        optimizer=None,
        optimizer_phase="hessian-sgd",
        step=3,
        next_step=4,
        status={
            "status": "running",
            "reason": "checkpoint_interval",
            "best_step": 3,
            "best_nll_bits": 10.0,
        },
        row={
            "optimizer/phase": "hessian-sgd",
            "likelihood/data_nll_bits": 10.0,
            "grad/inf": 0.1,
        },
    )
    payload = torch.load(checkpoint, weights_only=True)
    payload["route_metadata"] = {**_PRODUCTION_ROUTE_CONTRACT, **route_metadata}
    torch.save(payload, checkpoint)
    return checkpoint


def test_run_config_cli_surface_matches_dataclass_fields():
    run_config_fields = {field.name for field in fields(RunConfig)}
    expected_parser_dests = run_config_fields | {"config"}

    assert set(_run_config_cli_override_fields()) == run_config_fields
    assert not hasattr(gpurec_cli, "_RUN_CONFIG_CLI_OVERRIDE_FIELDS")
    assert _parser_action_dests("optimize") == (
        expected_parser_dests
        | {
            "require_converged",
            "require_final_check_ok",
            "require_mode_default_optimizer",
            "require_production_default_route",
        }
    )
    assert _parser_action_dests("validate-config") == (
        expected_parser_dests
        | {
            "check_preprocess",
            "require_cuda_backward_ready",
            "require_mode_default_optimizer",
            "require_production_default_route",
        }
    )
    assert _parser_action_dests("run") == expected_parser_dests | {
        "sample_out_dir",
        "samples",
        "seed",
        "family_start",
        "sample_max_families",
        "max_events",
        "backtrack_binary",
        "require_converged",
        "require_final_check_ok",
        "require_mode_default_optimizer",
        "require_production_default_route",
    }
    assert _parser_action_dests("backtrack-check") == {"backtrack_binary"}
    assert _parser_action_dests("checkpoint-info") == {
        "checkpoint",
        "require_final_check_ok",
        "require_mode_default_optimizer",
        "require_production_default_route",
    }
    assert _parser_action_dests("summary-info") == {
        "summary",
        "require_converged",
        "require_final_check_ok",
        "require_mode_default_optimizer",
        "require_production_default_route",
    }
    assert _parser_action_dests("config-template") == {
        "mode",
        "species_tree",
        "families_file",
        "out_dir",
        "device",
        "output",
        "force",
    }


def test_sampling_config_cli_surface_matches_dataclass_fields():
    sampling_dest_to_field = {
        "checkpoint": "checkpoint",
        "sample_out_dir": "out_dir",
        "samples": "samples",
        "seed": "seed",
        "family_start": "family_start",
        "sample_max_families": "max_families",
        "max_events": "max_events",
        "backtrack_binary": "backtrack_binary",
    }
    sampling_config_fields = {field.name for field in fields(SamplingConfig)}

    assert set(sampling_dest_to_field.values()) == sampling_config_fields
    assert set(sampling_dest_to_field) <= _parser_action_dests("sample")
    assert "require_mode_default_optimizer" in _parser_action_dests("sample")
    assert "require_production_default_route" in _parser_action_dests("sample")
    assert set(sampling_dest_to_field) - {"checkpoint"} <= _parser_action_dests("run")


@pytest.mark.parametrize(
    "command",
    [
        "optimize",
        "validate-config",
        "sample",
        "run",
        "checkpoint-info",
        "summary-info",
    ],
)
def test_cli_production_route_help_names_final_check_evidence(command: str):
    action = _parser_action(command, "require_production_default_route")

    assert "likelihood/gradient route" in str(action.help)
    assert "final_check_iters_e evidence" in str(action.help)
    assert "HOGENOM/" + "test_trees_" + "1000 production route" in str(action.help)


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
        preprocess_cpu_cores=7,
        family_chunk_size=4,
        clade_budget=42,
        batch_packing="sequential",
        max_wave_size=64,
        small_family_max_leaves=8,
        fixed_iters_e=3,
        max_iters_e=17,
        tol_e=1e-7,
        fixed_iters_pi=8,
        neumann_terms=6,
        final_check_iters=12,
        adaptive_iters=False,
        adaptive_neumann_terms=False,
        convergence_check_interval=6,
        e_logsumexp_tol=2e-5,
        pi_max_diff_tol=3e-5,
        gradient_change_tol=4e-4,
        gradient_change_rtol=5e-4,
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

    model = build_alerax_workflow_model(config, prefetch_batches=0)

    assert model is sentinel
    assert call["args"] == (str(config.species_tree), config.families_file)
    kwargs = call["kwargs"]
    assert kwargs["mode"] == "specieswise"
    assert kwargs["start"] == 2
    assert kwargs["max_families"] == 5
    assert kwargs["device"] == "cuda"
    assert kwargs["dtype"] is torch.float64
    assert kwargs["theta_init_rates"] == config.theta_init_rates
    assert kwargs["preprocess_cpu_cores"] == 7
    assert "preprocess_cache_dir" not in kwargs
    assert "refresh_preprocess_cache" not in kwargs
    assert kwargs["family_chunk_size"] == 4
    assert kwargs["clade_budget"] == 42
    assert kwargs["batch_packing"] == "sequential"
    assert kwargs["max_wave_size"] == 64
    assert kwargs["small_family_max_leaves"] == 8
    assert kwargs["fixed_iters_E"] == 3
    assert kwargs["max_iters_E"] == 17
    assert kwargs["tol_E"] == pytest.approx(1e-7)
    assert kwargs["fixed_iters_Pi"] == 8
    assert kwargs["neumann_terms"] == 6
    assert kwargs["adaptive_iters"] is False
    assert kwargs["adaptive_neumann_terms"] is False
    assert kwargs["convergence_check_interval"] == 6
    assert kwargs["e_logsumexp_tol"] == pytest.approx(2e-5)
    assert kwargs["pi_max_diff_tol"] == pytest.approx(3e-5)
    assert kwargs["gradient_change_tol"] == pytest.approx(4e-4)
    assert kwargs["gradient_change_rtol"] == pytest.approx(5e-4)
    assert kwargs["lazy_preprocess"] is True
    assert kwargs["prefetch_batches"] == 0


def test_build_alerax_workflow_model_configures_pi_adjoint_warmstart(
    tmp_path: Path,
    monkeypatch,
):
    config = RunConfig(
        species_tree=tmp_path / "sp.nwk",
        families_file=tmp_path / "families.txt",
        out_dir=tmp_path / "out",
        mode="genewise",
        optimizer="hessian-sgd",
        device="cuda",
        hessian_sgd_pi_adjoint_warmstart=True,
        pi_fixed_point_relaxation=1.25,
    )

    model = object()
    call: dict[str, object] = {}

    def fake_from_alerax_families(*args: object, **kwargs: object) -> object:
        call["kwargs"] = kwargs
        return model

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        workflow_model_factory.GeneReconModel,
        "from_alerax_families",
        staticmethod(fake_from_alerax_families),
    )

    result = build_alerax_workflow_model(config, prefetch_batches=0)

    assert result is model
    assert call["kwargs"]["pi_adjoint_warmstart"] is True
    assert call["kwargs"]["pi_adjoint_cache_update_mode"] == "stage"
    assert call["kwargs"]["pi_fixed_point_relaxation"] == pytest.approx(1.25)


def test_cli_forwards_preprocess_cpu_cores(tmp_path: Path):
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
            "--preprocess-cpu-cores",
            "3",
        ]
    )

    config = _run_config_from_args(args)

    assert config.preprocess_cpu_cores == 3


def test_cli_forwards_hessian_sgd_solver_controls(tmp_path: Path):
    args = build_parser().parse_args(
        _minimal_workflow_cli_args("optimize", tmp_path)
        + [
            "--optimizer",
            "hessian-sgd",
            "--hessian-sgd-normal-fixed-iters-pi",
            "12",
            "--hessian-sgd-normal-neumann-terms",
            "10",
            "--hessian-sgd-pi-adjoint-warmstart",
            "--pi-fixed-point-relaxation",
            "1.25",
            "--hessian-sgd-validation-interval",
            "4",
            "--hessian-sgd-validation-fixed-iters-pi",
            "32",
            "--hessian-sgd-validation-neumann-terms",
            "48",
        ]
    )

    config = _run_config_from_args(args)

    assert config.hessian_sgd_normal_fixed_iters_pi == 12
    assert config.hessian_sgd_normal_neumann_terms == 10
    assert config.hessian_sgd_pi_adjoint_warmstart is True
    assert config.pi_fixed_point_relaxation == pytest.approx(1.25)
    assert config.hessian_sgd_validation_interval == 4
    assert config.hessian_sgd_validation_fixed_iters_pi == 32
    assert config.hessian_sgd_validation_neumann_terms == 48


def test_cli_help_describes_current_genewise_hessian_sgd_controls():
    warmup_patience = _parser_action("optimize", "solver_warmup_loss_patience")
    adam_warmup = _parser_action("optimize", "fd_adam_warmup_steps")
    hessian_refresh = _parser_action("optimize", "fd_hessian_refresh_steps")
    pi_adjoint_warmstart = _parser_action(
        "optimize",
        "hessian_sgd_pi_adjoint_warmstart",
    )
    pi_fixed_point_relaxation = _parser_action(
        "optimize",
        "pi_fixed_point_relaxation",
    )
    validation_interval = _parser_action(
        "optimize",
        "hessian_sgd_validation_interval",
    )

    assert "genewise active-batch optimizers" in str(warmup_patience.help)
    assert "Hessian-conditioned genewise updates" in str(adam_warmup.help)
    assert "Hessian-conditioned genewise steps" in str(hessian_refresh.help)
    assert "staged Pi-adjoint warm-start cache" in str(pi_adjoint_warmstart.help)
    assert "fixed-point relaxation factor" in str(pi_fixed_point_relaxation.help)
    assert "high-budget validation" in str(validation_interval.help)
    assert "batched-LBFGS" not in str(warmup_patience.help)


def test_cli_forwards_adagrad_restart_controls(tmp_path: Path):
    args = build_parser().parse_args(
        _minimal_workflow_cli_args("optimize", tmp_path)
        + [
            "--mode",
            "specieswise",
            "--optimizer",
            "adagrad-restarts",
            "--adagrad-restart-schedule",
            "4:1.0:2,8:0.5:3",
            "--adagrad-restart-final-check-iters",
            "16",
        ]
    )

    config = _run_config_from_args(args)

    assert config.optimizer == "adagrad-restarts"
    assert config.adagrad_restart_schedule == "4:1:2,8:0.5:3"
    assert config.adagrad_restart_final_check_iters == 16


def test_cli_normalizes_mode_and_optimizer_alias_flags(tmp_path: Path):
    args = build_parser().parse_args(
        _minimal_workflow_cli_args("optimize", tmp_path)
        + [
            "--mode",
            " SpeciesWise ",
            "--optimizer",
            "ADAGRAD_RESTARTS",
        ]
    )

    config = _run_config_from_args(args)

    assert config.mode == "specieswise"
    assert config.optimizer == "adagrad-restarts"


def test_cli_rejects_misplaced_hessian_sgd_normal_controls(tmp_path: Path):
    args = build_parser().parse_args(
        _minimal_workflow_cli_args("optimize", tmp_path)
        + [
            "--mode",
            "specieswise",
            "--hessian-sgd-normal-fixed-iters-pi",
            "12",
        ]
    )

    with pytest.raises(
        ValueError,
        match="hessian_sgd_normal solver controls require genewise hessian-sgd",
    ):
        _run_config_from_args(args)


def test_cli_rejects_misplaced_adagrad_restart_controls(tmp_path: Path):
    args = build_parser().parse_args(
        _minimal_workflow_cli_args("optimize", tmp_path)
        + [
            "--mode",
            "genewise",
            "--adagrad-restart-schedule",
            "4:1.0:2",
        ]
    )

    with pytest.raises(
        ValueError,
        match="adagrad_restart controls require specieswise adagrad-restarts",
    ):
        _run_config_from_args(args)


def test_cli_config_template_prints_genewise_hessian_sgd_auto_defaults(capsys):
    main(["config-template", "--mode", "genewise"])

    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert captured.err == ""
    assert data["species_tree"] == "S.tree"
    assert data["families_file"] == "families.txt"
    assert data["out_dir"] == "output_gpurec"
    assert data["mode"] == "genewise"
    assert data["device"] == "cuda"
    assert data["optimizer"] == "auto"
    assert data["solver_warmup_iters"] == 4
    assert data["fd_adam_warmup_steps"] == 3
    assert data["fd_hessian_refresh_steps"] == 16
    assert data["hessian_sgd_normal_fixed_iters_pi"] is None
    assert data["hessian_sgd_normal_neumann_terms"] is None
    assert data["hessian_sgd_pi_adjoint_warmstart"] is False
    assert data["pi_fixed_point_relaxation"] == pytest.approx(1.0)
    assert data["hessian_sgd_validation_interval"] == 0
    assert data["hessian_sgd_validation_fixed_iters_pi"] is None
    assert data["hessian_sgd_validation_neumann_terms"] is None
    assert data["final_check_iters"] == 32
    assert "adagrad_restart_schedule" not in data


def test_cli_config_template_prints_specieswise_adagrad_restart_defaults(capsys):
    main(["config-template", "--mode", " SpeciesWise "])

    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert captured.err == ""
    assert data["mode"] == "specieswise"
    assert data["optimizer"] == "auto"
    assert data["adagrad_restart_schedule"] == "8:1.0:60,16:0.5:35,32:0.5:30"
    assert data["adagrad_restart_final_check_iters"] == 128
    assert "hessian_sgd_normal_fixed_iters_pi" not in data
    assert "hessian_sgd_pi_adjoint_warmstart" not in data
    assert "pi_fixed_point_relaxation" not in data
    assert "hessian_sgd_validation_interval" not in data
    assert "fd_hessian_refresh_steps" not in data


def test_cli_config_template_writes_output_and_refuses_overwrite(
    tmp_path: Path,
    capsys,
):
    output = tmp_path / "nested dir" / "run config.json"

    main(
        [
            "config-template",
            "--mode",
            "specieswise",
            "--species-tree",
            "data/S.tree",
            "--families-file",
            "data/families.txt",
            "--out-dir",
            "runs/specieswise",
            "--output",
            str(output),
        ]
    )

    captured = capsys.readouterr()
    assert captured.err == ""
    assert (
        gpurec_cli._optional_text("config_template", output.resolve())
        in captured.out
    )
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["species_tree"] == "data/S.tree"
    assert data["families_file"] == "data/families.txt"
    assert data["out_dir"] == "runs/specieswise"
    assert data["mode"] == "specieswise"

    with pytest.raises(SystemExit) as exc_info:
        main(["config-template", "--output", str(output)])

    captured = capsys.readouterr()
    assert exc_info.value.code == 2
    assert "output config already exists" in captured.err
    assert "--force" in captured.err
    assert "Traceback" not in captured.err

    main(["config-template", "--mode", "global", "--output", str(output), "--force"])
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["mode"] == "global"
    assert data["optimizer"] == "auto"


def test_cli_accepts_legacy_gradient_tolerance_options_as_noops(tmp_path: Path):
    args = build_parser().parse_args(
        _minimal_workflow_cli_args("optimize", tmp_path)
        + [
            "--grad-inf-tol",
            "10",
            "--solver-warmup-grad-inf-tol",
            "0",
        ]
    )

    config = _run_config_from_args(args)

    assert not hasattr(config, "grad_inf_tol")
    assert not hasattr(config, "solver_warmup_grad_inf_tol")


def test_cli_rejects_adaptive_neumann_terms_mode(tmp_path: Path):
    args = build_parser().parse_args(
        _minimal_workflow_cli_args("optimize", tmp_path)
        + ["--adaptive-neumann-terms"]
    )

    with pytest.raises(
        ValueError,
        match=(
            "adaptive_neumann_terms mode is disabled.*"
            "not part of the supported production optimization route"
        ),
    ):
        _run_config_from_args(args)


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


def test_cli_validate_config_reports_selected_family_references(
    tmp_path: Path,
    capsys,
):
    config_path = tmp_path / "run.json"
    write_tiny_alerax_inputs(tmp_path)
    config_path.write_text(
        json.dumps(
            {
                "species_tree": "sp.nwk",
                "families_file": "families.txt",
                "out_dir": "run outputs",
                "mode": "genewise",
                "device": "cuda",
            }
        ),
        encoding="utf-8",
    )

    main(
        [
            "validate-config",
            "--config",
            str(config_path),
            "--require-mode-default-optimizer",
        ]
    )

    captured = capsys.readouterr()
    basis = "hogenom_and_" + "test_trees_" + "1000"
    assert "valid_config=true" in captured.out
    assert "mode=genewise" in captured.out
    assert "optimizer=hessian-sgd" in captured.out
    assert "objective=negative_log_likelihood_bits" in captured.out
    assert "gradient_route=implicit_first_order_adjoint" in captured.out
    assert "rate_parameterization=base2_log_dlt_rates" in captured.out
    assert f"production_default_basis={basis}" in captured.out
    assert "mode_default_optimizer=hessian-sgd" in captured.out
    assert "uses_mode_default_optimizer=true" in captured.out
    assert "uses_production_default_optimizer_settings=true" in captured.out
    assert "production_default_optimizer_setting_mismatches=none" in captured.out
    assert "uses_production_default_route=true" in captured.out
    assert "production_default_route_mismatches=none" in captured.out
    assert "families=1" in captured.out
    assert "gene_tree_files=1" in captured.out
    assert "mapped_families=1" in captured.out
    assert "batch_packing=depth_first_fit" in captured.out
    assert "family_chunk_size=0" in captured.out
    assert "clade_budget=500000" in captured.out
    assert "fixed_iters_e=adaptive" in captured.out
    assert "fixed_iters_pi=16" in captured.out
    assert "neumann_terms=16" in captured.out
    assert "final_check_iters=32" in captured.out
    assert "final_check_iters_e=null" in captured.out
    assert "configured_steps=5000" in captured.out
    assert "optimizer_step_cap=5000" in captured.out
    assert "optimizer_step_cap_reason=configured_steps" in captured.out
    assert "solver_warmup_iters=4" in captured.out
    assert "fd_adam_warmup_steps=3" in captured.out
    assert "fd_hessian_refresh_steps=16" in captured.out
    assert "hessian_sgd_normal_fixed_iters_pi=full" in captured.out
    assert "hessian_sgd_normal_neumann_terms=full" in captured.out
    assert "hessian_sgd_pi_adjoint_warmstart=false" in captured.out
    assert "pi_fixed_point_relaxation=1.000000" in captured.out
    assert "hessian_sgd_validation_interval=0" in captured.out
    assert "hessian_sgd_validation_fixed_iters_pi=configured" in captured.out
    assert "hessian_sgd_validation_neumann_terms=configured" in captured.out
    assert "preprocess_checked" not in captured.out
    assert gpurec_cli._optional_text(
        "out_dir",
        (tmp_path / "run outputs").resolve(),
    ) in captured.out
    assert captured.err == ""


def test_cli_validate_config_require_mode_default_optimizer_rejects_override(
    tmp_path: Path,
    capsys,
):
    write_tiny_alerax_inputs(tmp_path)

    with pytest.raises(SystemExit) as exc_info:
        main(
            _minimal_workflow_cli_args("validate-config", tmp_path)
            + [
                "--optimizer",
                "adam",
                "--require-mode-default-optimizer",
            ]
        )

    captured = capsys.readouterr()
    assert exc_info.value.code == 2
    assert "config optimizer is 'adam'" in captured.err
    assert "expected mode default 'hessian-sgd' for mode 'genewise'" in captured.err
    assert "use optimizer=auto or the mode default optimizer" in captured.err
    assert "valid_config=true" not in captured.out
    assert "Traceback" not in captured.err


def test_cli_validate_config_require_production_default_route_rejects_custom_settings(
    tmp_path: Path,
    capsys,
):
    write_tiny_alerax_inputs(tmp_path)

    with pytest.raises(SystemExit) as exc_info:
        main(
            _minimal_workflow_cli_args("validate-config", tmp_path)
            + [
                "--optimizer",
                "hessian-sgd",
                "--fd-hessian-refresh-steps",
                "8",
                "--require-production-default-route",
            ]
        )

    captured = capsys.readouterr()
    assert exc_info.value.code == 2
    assert (
        "config production default route fields differ for mode 'genewise': "
        "fd_hessian_refresh_steps"
    ) in captured.err
    assert "use optimizer=auto and the shipped optimizer defaults" in captured.err
    assert "valid_config=true" not in captured.out
    assert "Traceback" not in captured.err


def test_cli_validate_config_require_production_default_route_accepts_genewise_default(
    tmp_path: Path,
    capsys,
):
    main(
        _minimal_workflow_cli_args("validate-config", tmp_path)
        + ["--require-production-default-route"]
    )

    captured = capsys.readouterr()
    assert captured.err == ""
    assert "valid_config=true" in captured.out
    assert "mode=genewise" in captured.out
    assert "optimizer=hessian-sgd" in captured.out
    assert "mode_default_optimizer=hessian-sgd" in captured.out
    assert "uses_mode_default_optimizer=true" in captured.out
    assert "uses_production_default_optimizer_settings=true" in captured.out
    assert "production_default_optimizer_setting_mismatches=none" in captured.out
    assert "uses_production_default_route=true" in captured.out
    assert "production_default_route_mismatches=none" in captured.out


def test_cli_validate_config_reports_hessian_sgd_normal_solver_overrides(
    tmp_path: Path,
    capsys,
):
    write_tiny_alerax_inputs(tmp_path)

    main(
        [
            "validate-config",
            "--species-tree",
            str(tmp_path / "sp.nwk"),
            "--families-file",
            str(tmp_path / "families.txt"),
            "--out-dir",
            str(tmp_path / "out"),
            "--mode",
            "genewise",
            "--optimizer",
            "hessian-sgd",
            "--device",
            "cuda",
            "--hessian-sgd-normal-fixed-iters-pi",
            "12",
            "--hessian-sgd-normal-neumann-terms",
            "10",
            "--hessian-sgd-pi-adjoint-warmstart",
            "--pi-fixed-point-relaxation",
            "1.25",
            "--hessian-sgd-validation-interval",
            "4",
            "--hessian-sgd-validation-fixed-iters-pi",
            "32",
            "--hessian-sgd-validation-neumann-terms",
            "48",
        ]
    )

    captured = capsys.readouterr()
    assert "valid_config=true" in captured.out
    assert "optimizer=hessian-sgd" in captured.out
    assert "hessian_sgd_normal_fixed_iters_pi=12" in captured.out
    assert "hessian_sgd_normal_neumann_terms=10" in captured.out
    assert "hessian_sgd_pi_adjoint_warmstart=true" in captured.out
    assert "pi_fixed_point_relaxation=1.250000" in captured.out
    assert "hessian_sgd_validation_interval=4" in captured.out
    assert "hessian_sgd_validation_fixed_iters_pi=32" in captured.out
    assert "hessian_sgd_validation_neumann_terms=48" in captured.out
    assert "uses_production_default_optimizer_settings=false" in captured.out
    assert (
        "production_default_optimizer_setting_mismatches="
        "hessian_sgd_normal_fixed_iters_pi,hessian_sgd_normal_neumann_terms,"
        "hessian_sgd_pi_adjoint_warmstart,pi_fixed_point_relaxation,"
        "hessian_sgd_validation_interval,hessian_sgd_validation_fixed_iters_pi,"
        "hessian_sgd_validation_neumann_terms"
    ) in captured.out
    assert "uses_production_default_route=false" in captured.out
    assert (
        "production_default_route_mismatches="
        "hessian_sgd_normal_fixed_iters_pi,hessian_sgd_normal_neumann_terms,"
        "hessian_sgd_pi_adjoint_warmstart,pi_fixed_point_relaxation,"
        "hessian_sgd_validation_interval,hessian_sgd_validation_fixed_iters_pi,"
        "hessian_sgd_validation_neumann_terms"
    ) in captured.out
    assert captured.err == ""


def test_cli_validate_config_reports_specieswise_restart_route(
    tmp_path: Path,
    capsys,
):
    config_path = tmp_path / "run.json"
    write_tiny_alerax_inputs(tmp_path)
    config_path.write_text(
        json.dumps(
            {
                "species_tree": "sp.nwk",
                "families_file": "families.txt",
                "out_dir": "out",
                "mode": "specieswise",
                "device": "cuda",
            }
        ),
        encoding="utf-8",
    )

    main(["validate-config", "--config", str(config_path)])

    captured = capsys.readouterr()
    assert "valid_config=true" in captured.out
    assert "mode=specieswise" in captured.out
    assert "optimizer=adagrad-restarts" in captured.out
    assert "mode_default_optimizer=adagrad-restarts" in captured.out
    assert "uses_mode_default_optimizer=true" in captured.out
    assert "uses_production_default_optimizer_settings=true" in captured.out
    assert "production_default_optimizer_setting_mismatches=none" in captured.out
    assert "uses_production_default_route=true" in captured.out
    assert "production_default_route_mismatches=none" in captured.out
    assert (
        "adagrad_restart_schedule=8:1:60,16:0.5:35,32:0.5:30"
        in captured.out
    )
    assert "adagrad_restart_total_steps=125" in captured.out
    assert "configured_steps=5000" in captured.out
    assert "optimizer_step_cap=125" in captured.out
    assert "optimizer_step_cap_reason=adagrad_restart_schedule" in captured.out
    assert "final_check_iters=128" in captured.out
    assert "final_check_iters_e=128" in captured.out
    assert "adagrad_restart_final_check_iters=128" in captured.out
    assert captured.err == ""


def test_cli_validate_config_require_production_default_route_accepts_specieswise_default(
    tmp_path: Path,
    capsys,
):
    main(
        _minimal_workflow_cli_args("validate-config", tmp_path)
        + [
            "--mode",
            "specieswise",
            "--require-production-default-route",
        ]
    )

    captured = capsys.readouterr()
    assert captured.err == ""
    assert "valid_config=true" in captured.out
    assert "mode=specieswise" in captured.out
    assert "optimizer=adagrad-restarts" in captured.out
    assert "mode_default_optimizer=adagrad-restarts" in captured.out
    assert "uses_mode_default_optimizer=true" in captured.out
    assert "uses_production_default_optimizer_settings=true" in captured.out
    assert "production_default_optimizer_setting_mismatches=none" in captured.out
    assert "uses_production_default_route=true" in captured.out
    assert "production_default_route_mismatches=none" in captured.out
    assert "optimizer_step_cap=125" in captured.out
    assert "optimizer_step_cap_reason=adagrad_restart_schedule" in captured.out
    assert "final_check_iters_e=128" in captured.out


def test_cli_validate_config_reuses_checkpoint_route_formatter(tmp_path: Path):
    genewise = RunConfig(
        species_tree=tmp_path / "sp.nwk",
        families_file=tmp_path / "families.txt",
        out_dir=tmp_path / "out-genewise",
        mode="genewise",
        device="cpu",
        hessian_sgd_normal_fixed_iters_pi=12,
        hessian_sgd_normal_neumann_terms=10,
        hessian_sgd_pi_adjoint_warmstart=True,
        pi_fixed_point_relaxation=1.25,
        hessian_sgd_validation_interval=4,
        hessian_sgd_validation_fixed_iters_pi=32,
        hessian_sgd_validation_neumann_terms=48,
    )
    specieswise = RunConfig(
        species_tree=tmp_path / "sp.nwk",
        families_file=tmp_path / "families.txt",
        out_dir=tmp_path / "out-specieswise",
        mode="specieswise",
        device="cpu",
    )

    for config in (genewise, specieswise):
        assert gpurec_cli._validate_config_route_text(
            config
        ) == gpurec_cli._route_metadata_text(effective_route_metadata(config))


def test_cli_validate_config_can_check_cpu_preprocessing(
    tmp_path: Path,
    capsys,
):
    write_tiny_alerax_inputs(tmp_path)

    main(
        [
            "validate-config",
            "--species-tree",
            str(tmp_path / "sp.nwk"),
            "--families-file",
            str(tmp_path / "families.txt"),
            "--out-dir",
            str(tmp_path / "out"),
            "--device",
            "cuda",
            "--check-preprocess",
        ]
    )

    captured = capsys.readouterr()
    assert "valid_config=true" in captured.out
    assert "preprocess_checked=true" in captured.out
    assert "preprocessed_families=1" in captured.out
    assert "preprocessed_species_nodes=3" in captured.out
    assert "cuda_backward_ready=false" in captured.out
    assert "cuda_backward_ready_reason=requires_s_gt_256" in captured.out
    assert captured.err == ""


def test_cli_validate_config_can_require_cuda_backward_readiness(
    tmp_path: Path,
    capsys,
):
    write_tiny_alerax_inputs(tmp_path)

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "validate-config",
                "--species-tree",
                str(tmp_path / "sp.nwk"),
                "--families-file",
                str(tmp_path / "families.txt"),
                "--out-dir",
                str(tmp_path / "out"),
                "--device",
                "cuda",
                "--check-preprocess",
                "--require-cuda-backward-ready",
            ]
        )

    captured = capsys.readouterr()
    assert exc_info.value.code == 2
    assert "cuda_backward_ready=false" in captured.err
    assert "requires_s_gt_256" in captured.err
    assert "more than 256 postorder species nodes" in captured.err
    assert "Traceback" not in captured.err


def test_cli_validate_config_requires_preprocess_for_cuda_backward_gate(
    tmp_path: Path,
    capsys,
):
    write_tiny_alerax_inputs(tmp_path)

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "validate-config",
                "--species-tree",
                str(tmp_path / "sp.nwk"),
                "--families-file",
                str(tmp_path / "families.txt"),
                "--out-dir",
                str(tmp_path / "out"),
                "--device",
                "cuda",
                "--require-cuda-backward-ready",
            ]
        )

    captured = capsys.readouterr()
    assert exc_info.value.code == 2
    assert "--require-cuda-backward-ready requires --check-preprocess" in captured.err
    assert "Traceback" not in captured.err


def test_cli_validate_config_check_preprocess_rejects_bad_newick_before_cuda(
    tmp_path: Path,
    capsys,
):
    write_tiny_alerax_inputs(
        tmp_path,
        species_tree="(A:1,B:1,C:1)Root;\n",
    )

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "validate-config",
                "--species-tree",
                str(tmp_path / "sp.nwk"),
                "--families-file",
                str(tmp_path / "families.txt"),
                "--out-dir",
                str(tmp_path / "out"),
                "--device",
                "cuda",
                "--check-preprocess",
            ]
        )

    captured = capsys.readouterr()
    assert exc_info.value.code == 2
    assert "Species tree" in captured.err
    assert "CUDA" not in captured.err
    assert "Traceback" not in captured.err


def test_cli_validate_config_rejects_missing_gene_tree_before_cuda(
    tmp_path: Path,
    capsys,
):
    write_tiny_alerax_inputs(
        tmp_path,
        family_lines=("starting_gene_tree = missing_gene.nwk", "mapping = gene.map"),
    )

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "validate-config",
                "--species-tree",
                str(tmp_path / "sp.nwk"),
                "--families-file",
                str(tmp_path / "families.txt"),
                "--out-dir",
                str(tmp_path / "out"),
                "--device",
                "cuda",
            ]
        )

    captured = capsys.readouterr()
    assert exc_info.value.code == 2
    assert "missing gene-tree path" in captured.err
    assert "missing_gene.nwk" in captured.err
    assert "CUDA" not in captured.err
    assert "Traceback" not in captured.err


def test_cli_optimize_rejects_missing_gene_tree_before_workflow(
    tmp_path: Path,
    capsys,
    monkeypatch,
):
    write_tiny_alerax_inputs(
        tmp_path,
        family_lines=("starting_gene_tree = missing_gene.nwk", "mapping = gene.map"),
    )

    def unexpected_optimize(config):
        raise AssertionError("optimize should not be called")

    monkeypatch.setattr("gpurec.cli.optimize", unexpected_optimize)

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "optimize",
                "--species-tree",
                str(tmp_path / "sp.nwk"),
                "--families-file",
                str(tmp_path / "families.txt"),
                "--out-dir",
                str(tmp_path / "out"),
                "--device",
                "cuda",
            ]
        )

    captured = capsys.readouterr()
    assert exc_info.value.code == 2
    assert "missing gene-tree path" in captured.err
    assert "missing_gene.nwk" in captured.err
    assert "CUDA" not in captured.err
    assert "Traceback" not in captured.err


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


def _assert_subcommand_usage(stderr: str, command: str) -> None:
    assert f"usage: gpurec {command}" in stderr
    assert "usage: gpurec [-h]" not in stderr


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
    _assert_subcommand_usage(captured.err, "optimize")
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
        timeout=SUBPROCESS_TIMEOUT,
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
        ("device", 42, "device must be a device string"),
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
    _assert_subcommand_usage(captured.err, "optimize")
    assert "Traceback" not in captured.err


def test_cli_rejects_invalid_device_before_workflow(
    tmp_path: Path,
    capsys,
    monkeypatch,
):
    def unexpected_optimize(config):
        raise AssertionError("optimize should not be called")

    monkeypatch.setattr("gpurec.cli.optimize", unexpected_optimize)

    with pytest.raises(SystemExit) as exc_info:
        main(_minimal_workflow_cli_args("optimize", tmp_path) + ["--device", "cdua"])

    captured = capsys.readouterr()
    assert exc_info.value.code == 2
    assert "device must be a valid torch device string" in captured.err
    assert "cdua" in captured.err
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
    _assert_subcommand_usage(captured.err, "sample")
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


def test_cli_checkpoint_info_reports_route_status_and_last_row(
    tmp_path: Path,
    capsys,
):
    class FakeModel:
        theta = torch.nn.Parameter(torch.zeros(2, 3))
        family_names = ["fam0", "fam1"]
        species_names = ["sp0", "sp1", "sp2"]

    config = RunConfig(
        species_tree=tmp_path / "sp.nwk",
        families_file=tmp_path / "families.txt",
        out_dir=tmp_path / "out",
        mode="specieswise",
        device="cpu",
    )
    checkpoint = tmp_path / "best.pt"
    save_checkpoint(
        checkpoint,
        config=config,
        model=FakeModel(),
        optimizer=None,
        optimizer_phase="fixed32",
        step=7,
        next_step=8,
        status={
            "status": "running",
            "reason": "checkpoint_interval",
            "best_step": 6,
            "best_nll_bits": 22.5,
        },
        row={
            "optimizer/phase": "final_eval",
            "likelihood/data_nll_bits": 23.5,
            "likelihood/log_likelihood_bits": -23.5,
            "grad/inf": 0.25,
            "grad/projected_inf": 0.125,
            "optimizer/final_check_iters": 128,
            "optimizer/final_check_iters_E": 128,
            "optimizer/final_check_status": "ok",
            "optimizer/final_check_source": "fallback_clade_budget",
            "optimizer/final_check_reason": "smaller resident batch",
            "optimizer/final_check_fallback_clade_budget": 250_000.0,
            "optimizer/final_check_loss_abs_delta_bits": 0.125,
            "optimizer/final_check_grad_max_abs_delta": 0.5,
            "optimizer/final_check_grad_rel_inf_delta": 0.25,
            "solver/e_adjoint_failed_batches": 1.0,
            "solver/e_adjoint_success_batches": 0.0,
            "solver/e_adjoint_rel_res_max": 0.125,
        },
    )

    main(
        [
            "checkpoint-info",
            "--checkpoint",
            str(checkpoint),
            "--require-final-check-ok",
        ]
    )

    captured = capsys.readouterr()
    basis = "hogenom_and_" + "test_trees_" + "1000"
    assert captured.err == ""
    assert gpurec_cli._optional_text("checkpoint", checkpoint.resolve()) in captured.out
    for token in (
        "version=1",
        "step=7",
        "next_step=8",
        "status=running",
        "reason=checkpoint_interval",
        "mode=specieswise",
        "optimizer=adagrad-restarts",
        "objective=negative_log_likelihood_bits",
        "gradient_route=implicit_first_order_adjoint",
        "rate_parameterization=base2_log_dlt_rates",
        f"production_default_basis={basis}",
        "mode_default_optimizer=adagrad-restarts",
        "uses_mode_default_optimizer=true",
        "uses_production_default_optimizer_settings=true",
        "production_default_optimizer_setting_mismatches=none",
        "uses_production_default_route=true",
        "production_default_route_mismatches=none",
        "batch_packing=depth_first_fit",
        "family_chunk_size=0",
        "clade_budget=500000",
        "fixed_iters_e=adaptive",
        "fixed_iters_pi=16",
        "neumann_terms=16",
        "final_check_iters=128",
        "final_check_iters_e=128",
        "configured_steps=5000",
        "optimizer_step_cap=125",
        "optimizer_step_cap_reason=adagrad_restart_schedule",
        "adagrad_restart_schedule=8:1:60,16:0.5:35,32:0.5:30",
        "adagrad_restart_total_steps=125",
        "adagrad_restart_final_check_iters=128",
        "route_metadata_source=checkpoint",
        "optimizer_phase=fixed32",
        "last_phase=final_eval",
        "families=2",
        "species=3",
        "best_step=6",
        "best_nll_bits=22.500000",
        "last_nll_bits=23.500000",
        "last_log_likelihood_bits=-23.500000",
        "last_grad_inf=0.250000",
        "last_projected_grad_inf=0.125000",
        "last_final_check_iters=128",
        "last_final_check_iters_e=128",
        "last_final_check_status=ok",
        "last_final_check_source=fallback_clade_budget",
        'last_final_check_reason="smaller\\u0020resident\\u0020batch"',
        "last_final_check_fallback_clade_budget=250000.000000",
        "last_final_check_loss_abs_delta_bits=0.125000",
        "last_final_check_grad_max_abs_delta=0.500000",
        "last_final_check_grad_rel_inf_delta=0.250000",
        "last_solver_e_adjoint_failed_batches=1.000000",
        "last_solver_e_adjoint_success_batches=0.000000",
        "last_solver_e_adjoint_rel_res_max=0.125000",
    ):
        assert token in captured.out


def test_cli_checkpoint_info_require_final_check_ok_fails_after_printing_checkpoint(
    tmp_path: Path,
    capsys,
):
    class FakeModel:
        theta = torch.nn.Parameter(torch.zeros(1, 3))
        family_names = ["fam0"]
        species_names = ["sp0", "sp1"]

    config = RunConfig(
        species_tree=tmp_path / "sp.nwk",
        families_file=tmp_path / "families.txt",
        out_dir=tmp_path / "out",
        mode="genewise",
        device="cpu",
    )
    checkpoint = tmp_path / "latest.pt"
    save_checkpoint(
        checkpoint,
        config=config,
        model=FakeModel(),
        optimizer=None,
        optimizer_phase="final_eval",
        step=3,
        next_step=4,
        status={
            "status": "converged",
            "reason": "loss_change",
            "best_step": 3,
            "best_nll_bits": 10.0,
        },
        row={
            "optimizer/phase": "final_eval",
            "likelihood/data_nll_bits": 10.0,
            "grad/inf": 0.1,
            "optimizer/final_check_status": "skipped",
            "optimizer/final_check_source": "not_evaluated",
            "optimizer/final_check_reason": "solver_cache_unavailable",
        },
    )

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "checkpoint-info",
                "--checkpoint",
                str(checkpoint),
                "--require-final-check-ok",
            ]
        )

    captured = capsys.readouterr()
    assert exc_info.value.code == 1
    assert "status=converged" in captured.out
    assert "last_final_check_status=skipped" in captured.out
    assert "last_final_check_source=not_evaluated" in captured.out
    assert "checkpoint final check status is 'skipped'; expected 'ok'" in captured.err
    assert "usage:" not in captured.err
    assert "Traceback" not in captured.err


def test_cli_checkpoint_info_require_mode_default_optimizer_fails_after_printing(
    tmp_path: Path,
    capsys,
):
    class FakeModel:
        theta = torch.nn.Parameter(torch.zeros(1, 3))
        family_names = ["fam0"]
        species_names = ["sp0", "sp1"]

    config = RunConfig(
        species_tree=tmp_path / "sp.nwk",
        families_file=tmp_path / "families.txt",
        out_dir=tmp_path / "out",
        mode="genewise",
        optimizer="adam",
        device="cpu",
    )
    checkpoint = tmp_path / "latest.pt"
    save_checkpoint(
        checkpoint,
        config=config,
        model=FakeModel(),
        optimizer=None,
        optimizer_phase="adam",
        step=3,
        next_step=4,
        status={
            "status": "running",
            "reason": "checkpoint_interval",
            "best_step": 3,
            "best_nll_bits": 10.0,
        },
        row={
            "optimizer/phase": "adam",
            "likelihood/data_nll_bits": 10.0,
            "grad/inf": 0.1,
        },
    )

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "checkpoint-info",
                "--checkpoint",
                str(checkpoint),
                "--require-mode-default-optimizer",
            ]
        )

    captured = capsys.readouterr()
    assert exc_info.value.code == 1
    assert "optimizer=adam" in captured.out
    assert "mode_default_optimizer=hessian-sgd" in captured.out
    assert "uses_mode_default_optimizer=false" in captured.out
    assert "checkpoint optimizer is 'adam'" in captured.err
    assert "expected mode default 'hessian-sgd' for mode 'genewise'" in captured.err
    assert "usage:" not in captured.err
    assert "Traceback" not in captured.err


def test_cli_checkpoint_info_require_mode_default_optimizer_recomputes_stale_audit(
    tmp_path: Path,
    capsys,
):
    checkpoint = _checkpoint_with_route_metadata(
        tmp_path,
        {
            "mode": "genewise",
            "optimizer": "adam",
            "mode_default_optimizer": "adam",
            "uses_mode_default_optimizer": True,
        },
    )

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "checkpoint-info",
                "--checkpoint",
                str(checkpoint),
                "--require-mode-default-optimizer",
            ]
        )

    captured = capsys.readouterr()
    assert exc_info.value.code == 1
    assert "route_metadata_source=checkpoint" in captured.out
    assert "optimizer=adam" in captured.out
    assert "mode_default_optimizer=hessian-sgd" in captured.out
    assert "uses_mode_default_optimizer=false" in captured.out
    assert "checkpoint optimizer is 'adam'" in captured.err
    assert "expected mode default 'hessian-sgd' for mode 'genewise'" in captured.err
    assert "usage:" not in captured.err
    assert "Traceback" not in captured.err


def test_cli_checkpoint_info_require_mode_default_optimizer_reports_missing_evidence(
    tmp_path: Path,
    capsys,
):
    checkpoint = tmp_path / "legacy.pt"
    torch.save(
        {
            "version": CHECKPOINT_VERSION,
            "config": {
                "species_tree": str(tmp_path / "sp.nwk"),
                "families_file": str(tmp_path / "families.txt"),
                "mode": "genewise",
                "start": 0,
                "max_families": None,
            },
            "theta": torch.zeros(1, 3),
            "step": 3,
            "next_step": 4,
            "status": {
                "status": "running",
                "reason": "checkpoint_interval",
                "best_step": 3,
                "best_nll_bits": 10.0,
            },
            "last_row": {
                "optimizer/phase": "unknown",
                "likelihood/data_nll_bits": 10.0,
                "grad/inf": 0.1,
            },
            "family_names": ["fam0"],
            "species_names": ["sp0", "sp1"],
        },
        checkpoint,
    )

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "checkpoint-info",
                "--checkpoint",
                str(checkpoint),
                "--require-mode-default-optimizer",
            ]
        )

    captured = capsys.readouterr()
    assert exc_info.value.code == 1
    assert "mode=genewise" in captured.out
    assert "optimizer=null" in captured.out
    assert "mode_default_optimizer=hessian-sgd" in captured.out
    assert "uses_mode_default_optimizer=null" in captured.out
    assert "route_metadata_source=config" in captured.out
    assert "checkpoint mode default optimizer evidence is incomplete" in captured.err
    assert "missing optimizer" in captured.err
    assert "mode='genewise', optimizer=None" in captured.err
    assert "usage:" not in captured.err
    assert "Traceback" not in captured.err


def test_cli_checkpoint_info_require_production_default_route_accepts_current_route(
    tmp_path: Path,
    capsys,
):
    config = RunConfig(
        species_tree=tmp_path / "sp.nwk",
        families_file=tmp_path / "families.txt",
        out_dir=tmp_path / "out",
        mode="genewise",
        device="cpu",
    )
    checkpoint = _checkpoint_with_route_metadata(
        tmp_path,
        effective_route_metadata(config),
    )

    main(
        [
            "checkpoint-info",
            "--checkpoint",
            str(checkpoint),
            "--require-production-default-route",
        ]
    )

    captured = capsys.readouterr()
    assert captured.err == ""
    assert "route_metadata_source=checkpoint" in captured.out
    assert "mode=genewise" in captured.out
    assert "optimizer=hessian-sgd" in captured.out
    assert "uses_production_default_route=true" in captured.out
    assert "production_default_route_mismatches=none" in captured.out


def test_cli_checkpoint_info_require_production_default_route_recomputes_stale_audit(
    tmp_path: Path,
    capsys,
):
    checkpoint = _checkpoint_with_route_metadata(
        tmp_path,
        {
            "mode": "genewise",
            "optimizer": "hessian-sgd",
            "uses_production_default_optimizer_settings": True,
            "production_default_optimizer_setting_mismatches": [],
            "final_check_iters": 32,
            "final_check_iters_e": None,
            "solver_warmup_iters": 4,
            "fd_adam_warmup_steps": 3,
            "fd_hessian_refresh_steps": 8,
            "hessian_sgd_normal_fixed_iters_pi": None,
            "hessian_sgd_normal_neumann_terms": None,
            "hessian_sgd_pi_adjoint_warmstart": False,
            "pi_fixed_point_relaxation": 1.0,
            "hessian_sgd_validation_interval": 0,
            "hessian_sgd_validation_fixed_iters_pi": None,
            "hessian_sgd_validation_neumann_terms": None,
        },
    )

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "checkpoint-info",
                "--checkpoint",
                str(checkpoint),
                "--require-production-default-route",
            ]
        )

    captured = capsys.readouterr()
    assert exc_info.value.code == 1
    assert "route_metadata_source=checkpoint" in captured.out
    assert "uses_production_default_optimizer_settings=false" in captured.out
    assert "production_default_optimizer_setting_mismatches=fd_hessian_refresh_steps" in (
        captured.out
    )
    assert "uses_production_default_route=false" in captured.out
    assert "production_default_route_mismatches=fd_hessian_refresh_steps" in (
        captured.out
    )
    assert "checkpoint production default route fields differ" in captured.err
    assert "fd_hessian_refresh_steps" in captured.err
    assert "usage:" not in captured.err
    assert "Traceback" not in captured.err


def test_cli_checkpoint_info_require_production_default_route_requires_settings_evidence(
    tmp_path: Path,
    capsys,
):
    checkpoint = _checkpoint_with_route_metadata(
        tmp_path,
        {
            "mode": "genewise",
            "optimizer": "hessian-sgd",
            "uses_production_default_optimizer_settings": True,
            "production_default_optimizer_setting_mismatches": [],
        },
    )

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "checkpoint-info",
                "--checkpoint",
                str(checkpoint),
                "--require-production-default-route",
            ]
        )

    captured = capsys.readouterr()
    assert exc_info.value.code == 1
    assert "route_metadata_source=checkpoint" in captured.out
    assert "uses_production_default_optimizer_settings=null" in captured.out
    assert "production_default_optimizer_setting_mismatches=null" in captured.out
    assert "uses_production_default_route=null" in captured.out
    assert "production_default_route_mismatches=null" in captured.out
    assert "checkpoint production default route evidence is incomplete" in captured.err
    assert "missing final_check_iters" in captured.err
    assert "final_check_iters_e" in captured.err
    assert "fd_hessian_refresh_steps" in captured.err
    assert "usage:" not in captured.err
    assert "Traceback" not in captured.err


def test_cli_checkpoint_info_raw_theta_error_suggests_real_checkpoints(
    tmp_path: Path,
    capsys,
):
    checkpoint = tmp_path / "theta_final.pt"
    torch.save(torch.zeros(2, 3), checkpoint)

    with pytest.raises(SystemExit) as exc_info:
        main(["checkpoint-info", "--checkpoint", str(checkpoint)])

    captured = capsys.readouterr()
    assert exc_info.value.code == 1
    assert "must contain a dictionary payload" in captured.err
    assert "checkpoints/best.pt" in captured.err
    assert "checkpoints/latest.pt" in captured.err
    assert "not theta_final.pt" in captured.err
    assert "usage:" not in captured.err
    assert "Traceback" not in captured.err


def test_cli_summary_info_reports_status_route_and_final_check(
    tmp_path: Path,
    capsys,
):
    basis = "hogenom_and_" + "test_trees_" + "1000"
    summary = tmp_path / "summary with spaces.json"
    payload = {
        "status": "not_converged",
        "reason": "max_steps",
        "mode": "genewise",
        "optimizer": "hessian-sgd",
        "mode_default_optimizer": "hessian-sgd",
        "uses_mode_default_optimizer": True,
        "families": 3,
        "species": 4,
        "batches": 2,
        "batch_packing": "depth_first_fit",
        "family_chunk_size": 0,
        "clade_budget": 500_000,
        "fixed_iters_e": None,
        "fixed_iters_pi": 16,
        "neumann_terms": 16,
        "objective": "negative_log_likelihood_bits",
        "gradient_route": "implicit_first_order_adjoint",
        "rate_parameterization": "base2_log_dlt_rates",
        "production_default_basis": basis,
        "configured_steps": 5000,
        "optimizer_step_cap": 5000,
        "optimizer_step_cap_reason": "configured_steps",
        "final_check_iters": 32,
        "final_check_iters_e": None,
        "solver_warmup_iters": 4,
        "fd_adam_warmup_steps": 3,
        "fd_hessian_refresh_steps": 16,
        "hessian_sgd_normal_fixed_iters_pi": None,
        "hessian_sgd_normal_neumann_terms": None,
        "hessian_sgd_pi_adjoint_warmstart": False,
        "pi_fixed_point_relaxation": 1.0,
        "hessian_sgd_validation_interval": 0,
        "hessian_sgd_validation_fixed_iters_pi": None,
        "hessian_sgd_validation_neumann_terms": None,
        "steps_completed": 7,
        "elapsed_s": 3.25,
        "best_step": 5,
        "sampling_checkpoint": str(tmp_path / "checkpoints" / "best.pt"),
        "final_nll_bits": 12.5,
        "final_log_likelihood_bits": -12.5,
        "final_grad_inf": 0.75,
        "final_projected_grad_inf": 0.625,
        "best_nll_bits": 10.25,
        "best_log_likelihood_bits": -10.25,
        "final_check_status": "ok",
        "final_check_source": "configured_solver_budget",
        "final_check_loss_abs_delta_bits": 0.0,
        "final_check_grad_max_abs_delta": 0.0,
        "final_check_grad_rel_inf_delta": 0.0,
        "final_solver_e_adjoint_failed_batches": 1.0,
        "final_solver_e_adjoint_success_batches": 0.0,
        "final_solver_e_adjoint_rel_res_max": 0.125,
    }
    summary.write_text(json.dumps(payload), encoding="utf-8")

    main(["summary-info", "--summary", str(summary)])

    captured = capsys.readouterr()
    assert captured.err == ""
    assert gpurec_cli._optional_text("summary", summary.resolve()) in captured.out
    for token in (
        "status=not_converged",
        "reason=max_steps",
        "mode=genewise",
        "optimizer=hessian-sgd",
        "mode_default_optimizer=hessian-sgd",
        "uses_mode_default_optimizer=true",
        "uses_production_default_optimizer_settings=true",
        "production_default_optimizer_setting_mismatches=none",
        "uses_production_default_route=true",
        "production_default_route_mismatches=none",
        "families=3",
        "species=4",
        "batches=2",
        "batch_packing=depth_first_fit",
        "family_chunk_size=0",
        "clade_budget=500000",
        "fixed_iters_e=null",
        "fixed_iters_pi=16",
        "neumann_terms=16",
        "objective=negative_log_likelihood_bits",
        "gradient_route=implicit_first_order_adjoint",
        "rate_parameterization=base2_log_dlt_rates",
        f"production_default_basis={basis}",
        "configured_steps=5000",
        "optimizer_step_cap=5000",
        "optimizer_step_cap_reason=configured_steps",
        "final_check_iters=32",
        "final_check_iters_e=null",
        "solver_warmup_iters=4",
        "fd_adam_warmup_steps=3",
        "fd_hessian_refresh_steps=16",
        "hessian_sgd_normal_fixed_iters_pi=null",
        "hessian_sgd_normal_neumann_terms=null",
        "hessian_sgd_pi_adjoint_warmstart=false",
        "pi_fixed_point_relaxation=1.000000",
        "hessian_sgd_validation_interval=0",
        "hessian_sgd_validation_fixed_iters_pi=null",
        "hessian_sgd_validation_neumann_terms=null",
        "steps_completed=7",
        "elapsed_s=3.250000",
        "best_step=5",
        "final_nll_bits=12.500000",
        "final_log_likelihood_bits=-12.500000",
        "final_grad_inf=0.750000",
        "final_projected_grad_inf=0.625000",
        "best_nll_bits=10.250000",
        "best_log_likelihood_bits=-10.250000",
        "final_check_status=ok",
        "final_check_source=configured_solver_budget",
        "final_check_reason=null",
        "final_check_fallback_clade_budget=null",
        "final_check_loss_abs_delta_bits=0.000000",
        "final_check_grad_max_abs_delta=0.000000",
        "final_check_grad_rel_inf_delta=0.000000",
        "final_solver_e_adjoint_failed_batches=1.000000",
        "final_solver_e_adjoint_success_batches=0.000000",
        "final_solver_e_adjoint_rel_res_max=0.125000",
    ):
        assert token in captured.out


def test_cli_summary_info_reports_adagrad_restart_route_fields(
    tmp_path: Path,
    capsys,
):
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "status": "converged",
                "reason": "adagrad_restart_schedule_complete",
                "mode": "specieswise",
                "optimizer": "adagrad-restarts",
                "mode_default_optimizer": "adagrad-restarts",
                "uses_mode_default_optimizer": True,
                "final_check_iters": 128,
                "optimizer_step_cap": 125,
                "optimizer_step_cap_reason": "adagrad_restart_schedule",
                "adagrad_restart_schedule": "8:1:60,16:0.5:35,32:0.5:30",
                "adagrad_restart_total_steps": 125,
                "adagrad_restart_final_check_iters": 128,
                "steps_completed": 125,
                "final_check_iters_e": 128,
                "final_nll_bits": 12.0,
                "final_log_likelihood_bits": -12.0,
                "final_grad_inf": 0.5,
            }
        ),
        encoding="utf-8",
    )

    main(["summary-info", "--summary", str(summary)])

    captured = capsys.readouterr()
    assert captured.err == ""
    assert "optimizer=adagrad-restarts" in captured.out
    assert "mode_default_optimizer=adagrad-restarts" in captured.out
    assert "uses_mode_default_optimizer=true" in captured.out
    assert "uses_production_default_optimizer_settings=true" in captured.out
    assert "production_default_optimizer_setting_mismatches=none" in captured.out
    assert "uses_production_default_route=null" in captured.out
    assert "production_default_route_mismatches=null" in captured.out
    assert "reason=adagrad_restart_schedule_complete" in captured.out
    assert "final_check_iters=128" in captured.out
    assert "final_check_iters_e=128" in captured.out
    assert "adagrad_restart_schedule=8:1:60,16:0.5:35,32:0.5:30" in captured.out
    assert "adagrad_restart_total_steps=125" in captured.out
    assert "adagrad_restart_final_check_iters=128" in captured.out
    assert "solver_warmup_iters" not in captured.out


def test_cli_summary_info_infers_mode_default_optimizer_audit_fields(
    tmp_path: Path,
    capsys,
):
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "status": "converged",
                "reason": "loss_change_patience",
                "mode": "specieswise",
                "optimizer": "adagrad-restarts",
                "final_nll_bits": 12.0,
            }
        ),
        encoding="utf-8",
    )

    main(["summary-info", "--summary", str(summary)])

    captured = capsys.readouterr()
    assert captured.err == ""
    assert "mode=specieswise" in captured.out
    assert "optimizer=adagrad-restarts" in captured.out
    assert "mode_default_optimizer=adagrad-restarts" in captured.out
    assert "uses_mode_default_optimizer=true" in captured.out


def test_cli_summary_info_normalizes_route_mode_and_optimizer_aliases(
    tmp_path: Path,
    capsys,
):
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "status": "converged",
                "reason": "adagrad_restart_schedule_complete",
                **_PRODUCTION_ROUTE_CONTRACT,
                "mode": " SpeciesWise ",
                "optimizer": "ADAGRAD_RESTARTS",
                "final_check_iters": 128,
                "final_check_iters_e": 128,
                "optimizer_step_cap": 125,
                "optimizer_step_cap_reason": "adagrad_restart_schedule",
                "adagrad_restart_schedule": "8:1.0:60,16:0.5:35,32:0.5:30",
                "adagrad_restart_total_steps": 125,
                "adagrad_restart_final_check_iters": 128,
                "steps_completed": 125,
                "final_nll_bits": 12.0,
            }
        ),
        encoding="utf-8",
    )

    main(
        [
            "summary-info",
            "--summary",
            str(summary),
            "--require-mode-default-optimizer",
            "--require-production-default-route",
        ]
    )

    captured = capsys.readouterr()
    assert captured.err == ""
    assert "mode=specieswise" in captured.out
    assert "optimizer=adagrad-restarts" in captured.out
    assert "mode_default_optimizer=adagrad-restarts" in captured.out
    assert "uses_mode_default_optimizer=true" in captured.out
    assert "uses_production_default_optimizer_settings=true" in captured.out
    assert "production_default_optimizer_setting_mismatches=none" in captured.out
    assert "uses_production_default_route=true" in captured.out
    assert "production_default_route_mismatches=none" in captured.out


def test_cli_summary_info_require_mode_default_optimizer_fails_after_printing(
    tmp_path: Path,
    capsys,
):
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "status": "converged",
                "reason": "loss_change_patience",
                "mode": "genewise",
                "optimizer": "adam",
                "final_nll_bits": 12.0,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "summary-info",
                "--summary",
                str(summary),
                "--require-mode-default-optimizer",
            ]
        )

    captured = capsys.readouterr()
    assert exc_info.value.code == 1
    assert "optimizer=adam" in captured.out
    assert "mode_default_optimizer=hessian-sgd" in captured.out
    assert "uses_mode_default_optimizer=false" in captured.out
    assert "summary optimizer is 'adam'" in captured.err
    assert "expected mode default 'hessian-sgd' for mode 'genewise'" in captured.err
    assert "usage:" not in captured.err
    assert "Traceback" not in captured.err


def test_cli_summary_info_require_mode_default_optimizer_recomputes_stale_audit(
    tmp_path: Path,
    capsys,
):
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "status": "converged",
                "reason": "loss_change_patience",
                "mode": "genewise",
                "optimizer": "adam",
                "mode_default_optimizer": "adam",
                "uses_mode_default_optimizer": True,
                "final_nll_bits": 12.0,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "summary-info",
                "--summary",
                str(summary),
                "--require-mode-default-optimizer",
            ]
        )

    captured = capsys.readouterr()
    assert exc_info.value.code == 1
    assert "optimizer=adam" in captured.out
    assert "mode_default_optimizer=hessian-sgd" in captured.out
    assert "uses_mode_default_optimizer=false" in captured.out
    assert "summary optimizer is 'adam'" in captured.err
    assert "expected mode default 'hessian-sgd' for mode 'genewise'" in captured.err
    assert "usage:" not in captured.err
    assert "Traceback" not in captured.err


def test_cli_summary_info_require_mode_default_optimizer_reports_missing_evidence(
    tmp_path: Path,
    capsys,
):
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "status": "converged",
                "reason": "loss_change_patience",
                "mode": "genewise",
                "final_nll_bits": 12.0,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "summary-info",
                "--summary",
                str(summary),
                "--require-mode-default-optimizer",
            ]
        )

    captured = capsys.readouterr()
    assert exc_info.value.code == 1
    assert "mode=genewise" in captured.out
    assert "optimizer=null" in captured.out
    assert "mode_default_optimizer=hessian-sgd" in captured.out
    assert "uses_mode_default_optimizer=null" in captured.out
    assert "summary mode default optimizer evidence is incomplete" in captured.err
    assert "missing optimizer" in captured.err
    assert "mode='genewise', optimizer=None" in captured.err
    assert "usage:" not in captured.err
    assert "Traceback" not in captured.err


def test_cli_summary_info_require_production_default_route_accepts_current_route(
    tmp_path: Path,
    capsys,
):
    config = RunConfig(
        species_tree=tmp_path / "sp.nwk",
        families_file=tmp_path / "families.txt",
        out_dir=tmp_path / "out",
        mode="genewise",
        device="cpu",
    )
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "status": "converged",
                "reason": "loss_change_patience",
                **effective_route_metadata(config),
                "steps_completed": 4,
                "final_nll_bits": 12.0,
            }
        ),
        encoding="utf-8",
    )

    main(
        [
            "summary-info",
            "--summary",
            str(summary),
            "--require-production-default-route",
        ]
    )

    captured = capsys.readouterr()
    assert captured.err == ""
    assert "status=converged" in captured.out
    assert "mode=genewise" in captured.out
    assert "optimizer=hessian-sgd" in captured.out
    assert "uses_production_default_route=true" in captured.out
    assert "production_default_route_mismatches=none" in captured.out


def test_cli_summary_info_require_production_default_route_rejects_custom_settings(
    tmp_path: Path,
    capsys,
):
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "status": "converged",
                "reason": "loss_change_patience",
                **_PRODUCTION_ROUTE_CONTRACT,
                "mode": "genewise",
                "optimizer": "hessian-sgd",
                "final_check_iters": 32,
                "final_check_iters_e": None,
                "solver_warmup_iters": 4,
                "fd_adam_warmup_steps": 3,
                "fd_hessian_refresh_steps": 8,
                "hessian_sgd_normal_fixed_iters_pi": None,
                "hessian_sgd_normal_neumann_terms": None,
                "hessian_sgd_pi_adjoint_warmstart": False,
                "pi_fixed_point_relaxation": 1.0,
                "hessian_sgd_validation_interval": 0,
                "hessian_sgd_validation_fixed_iters_pi": None,
                "hessian_sgd_validation_neumann_terms": None,
                "steps_completed": 4,
                "final_nll_bits": 12.0,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "summary-info",
                "--summary",
                str(summary),
                "--require-production-default-route",
            ]
        )

    captured = capsys.readouterr()
    assert exc_info.value.code == 1
    assert "uses_production_default_optimizer_settings=false" in captured.out
    assert "production_default_optimizer_setting_mismatches=fd_hessian_refresh_steps" in (
        captured.out
    )
    assert "uses_production_default_route=false" in captured.out
    assert "production_default_route_mismatches=fd_hessian_refresh_steps" in (
        captured.out
    )
    assert (
        "summary production default route fields differ for mode 'genewise': "
        "fd_hessian_refresh_steps"
    ) in captured.err
    assert "expected the shipped likelihood/gradient and optimizer route" in (
        captured.err
    )
    assert "usage:" not in captured.err
    assert "Traceback" not in captured.err


def test_cli_summary_info_require_production_default_route_recomputes_stale_audit(
    tmp_path: Path,
    capsys,
):
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "status": "converged",
                "reason": "loss_change_patience",
                **_PRODUCTION_ROUTE_CONTRACT,
                "mode": "genewise",
                "optimizer": "hessian-sgd",
                "uses_production_default_optimizer_settings": True,
                "production_default_optimizer_setting_mismatches": [],
                "final_check_iters": 32,
                "final_check_iters_e": None,
                "solver_warmup_iters": 4,
                "fd_adam_warmup_steps": 3,
                "fd_hessian_refresh_steps": 8,
                "hessian_sgd_normal_fixed_iters_pi": None,
                "hessian_sgd_normal_neumann_terms": None,
                "hessian_sgd_pi_adjoint_warmstart": False,
                "pi_fixed_point_relaxation": 1.0,
                "hessian_sgd_validation_interval": 0,
                "hessian_sgd_validation_fixed_iters_pi": None,
                "hessian_sgd_validation_neumann_terms": None,
                "steps_completed": 4,
                "final_nll_bits": 12.0,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "summary-info",
                "--summary",
                str(summary),
                "--require-production-default-route",
            ]
        )

    captured = capsys.readouterr()
    assert exc_info.value.code == 1
    assert "uses_production_default_optimizer_settings=false" in captured.out
    assert "production_default_optimizer_setting_mismatches=fd_hessian_refresh_steps" in (
        captured.out
    )
    assert "uses_production_default_route=false" in captured.out
    assert "production_default_route_mismatches=fd_hessian_refresh_steps" in (
        captured.out
    )
    assert "fd_hessian_refresh_steps" in captured.err
    assert "Traceback" not in captured.err


def test_cli_summary_info_require_production_default_route_rejects_stale_gradient_route(
    tmp_path: Path,
    capsys,
):
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                **_PRODUCTION_ROUTE_CONTRACT,
                "gradient_route": "legacy_autograd",
                "status": "converged",
                "reason": "loss_change_patience",
                "mode": "genewise",
                "optimizer": "hessian-sgd",
                "final_check_iters": 32,
                "final_check_iters_e": None,
                "solver_warmup_iters": 4,
                "fd_adam_warmup_steps": 3,
                "fd_hessian_refresh_steps": 16,
                "hessian_sgd_normal_fixed_iters_pi": None,
                "hessian_sgd_normal_neumann_terms": None,
                "hessian_sgd_pi_adjoint_warmstart": False,
                "pi_fixed_point_relaxation": 1.0,
                "hessian_sgd_validation_interval": 0,
                "hessian_sgd_validation_fixed_iters_pi": None,
                "hessian_sgd_validation_neumann_terms": None,
                "steps_completed": 4,
                "final_nll_bits": 12.0,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "summary-info",
                "--summary",
                str(summary),
                "--require-production-default-route",
            ]
        )

    captured = capsys.readouterr()
    assert exc_info.value.code == 1
    assert "gradient_route=legacy_autograd" in captured.out
    assert "uses_production_default_optimizer_settings=true" in captured.out
    assert "production_default_optimizer_setting_mismatches=none" in captured.out
    assert "uses_production_default_route=false" in captured.out
    assert "production_default_route_mismatches=gradient_route" in captured.out
    assert (
        "summary production default route fields differ for mode 'genewise': "
        "gradient_route"
    ) in captured.err
    assert "Traceback" not in captured.err


def test_cli_summary_info_require_production_default_route_requires_settings_evidence(
    tmp_path: Path,
    capsys,
):
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "status": "converged",
                "reason": "loss_change_patience",
                **_PRODUCTION_ROUTE_CONTRACT,
                "mode": "genewise",
                "optimizer": "hessian-sgd",
                "uses_production_default_optimizer_settings": True,
                "production_default_optimizer_setting_mismatches": [],
                "steps_completed": 4,
                "final_nll_bits": 12.0,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "summary-info",
                "--summary",
                str(summary),
                "--require-production-default-route",
            ]
        )

    captured = capsys.readouterr()
    assert exc_info.value.code == 1
    assert "uses_production_default_optimizer_settings=null" in captured.out
    assert "production_default_optimizer_setting_mismatches=null" in captured.out
    assert "uses_production_default_route=null" in captured.out
    assert "production_default_route_mismatches=null" in captured.out
    assert "summary production default route evidence is incomplete" in captured.err
    assert "missing final_check_iters" in captured.err
    assert "final_check_iters_e" in captured.err
    assert "fd_hessian_refresh_steps" in captured.err
    assert "Traceback" not in captured.err


def test_cli_summary_info_require_converged_accepts_converged_summary(
    tmp_path: Path,
    capsys,
):
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "status": "converged",
                "reason": "adagrad_restart_schedule_complete",
                "mode": "specieswise",
                "optimizer": "adagrad-restarts",
                "final_nll_bits": 12.0,
            }
        ),
        encoding="utf-8",
    )

    main(["summary-info", "--summary", str(summary), "--require-converged"])

    captured = capsys.readouterr()
    assert captured.err == ""
    assert "status=converged" in captured.out
    assert "reason=adagrad_restart_schedule_complete" in captured.out


def test_cli_summary_info_require_converged_fails_after_printing_summary(
    tmp_path: Path,
    capsys,
):
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "status": "not_converged",
                "reason": "max_steps",
                "mode": "genewise",
                "optimizer": "hessian-sgd",
                "final_nll_bits": 12.0,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as exc_info:
        main(["summary-info", "--summary", str(summary), "--require-converged"])

    captured = capsys.readouterr()
    assert exc_info.value.code == 1
    assert "status=not_converged" in captured.out
    assert "reason=max_steps" in captured.out
    assert "summary status is 'not_converged'; expected 'converged'" in captured.err
    assert "usage:" not in captured.err
    assert "Traceback" not in captured.err


def test_cli_summary_info_require_final_check_ok_accepts_ok_summary(
    tmp_path: Path,
    capsys,
):
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "status": "not_converged",
                "reason": "max_steps",
                "mode": "genewise",
                "optimizer": "hessian-sgd",
                "final_check_status": "ok",
                "final_check_source": "configured_solver_budget",
                "final_nll_bits": 12.0,
            }
        ),
        encoding="utf-8",
    )

    main(["summary-info", "--summary", str(summary), "--require-final-check-ok"])

    captured = capsys.readouterr()
    assert captured.err == ""
    assert "status=not_converged" in captured.out
    assert "final_check_status=ok" in captured.out
    assert "final_check_source=configured_solver_budget" in captured.out


def test_cli_summary_info_require_final_check_ok_fails_after_printing_summary(
    tmp_path: Path,
    capsys,
):
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "status": "converged",
                "reason": "best_likelihood_patience",
                "mode": "genewise",
                "optimizer": "hessian-sgd",
                "final_check_status": "failed",
                "final_check_source": "configured_solver_budget",
                "final_check_reason": "nonfinite_check_gradient",
                "final_nll_bits": 12.0,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as exc_info:
        main(["summary-info", "--summary", str(summary), "--require-final-check-ok"])

    captured = capsys.readouterr()
    assert exc_info.value.code == 1
    assert "status=converged" in captured.out
    assert "final_check_status=failed" in captured.out
    assert "final_check_reason=nonfinite_check_gradient" in captured.out
    assert "summary final check status is 'failed'; expected 'ok'" in captured.err
    assert "usage:" not in captured.err
    assert "Traceback" not in captured.err


def test_cli_summary_info_reports_missing_path_without_traceback(
    tmp_path: Path,
    capsys,
):
    summary = tmp_path / "missing.json"

    with pytest.raises(SystemExit) as exc_info:
        main(["summary-info", "--summary", str(summary)])

    captured = capsys.readouterr()
    assert exc_info.value.code == 2
    assert "--summary path does not exist or is not a file" in captured.err
    assert str(summary) in captured.err
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


@pytest.mark.parametrize("out_flag", ["--sample-out-dir", "--sampling-out-dir"])
def test_cli_sample_forwards_sampling_options(
    tmp_path: Path,
    capsys,
    monkeypatch,
    out_flag: str,
):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"not used")
    sample_out_dir = tmp_path / "sample outputs"
    backtrack_binary = tmp_path / "gpurec-backtrack"
    captured_config = {}

    def capture_sample(config):
        captured_config["config"] = config
        return SimpleNamespace(
            families_sampled=4,
            samples_per_family=config.samples,
            xml_files=12,
            out_dir=config.out_dir,
        )

    monkeypatch.setattr("gpurec.cli.sample", capture_sample)

    main(
        [
            "sample",
            "--checkpoint",
            str(checkpoint),
            out_flag,
            str(sample_out_dir),
            "--samples",
            "3",
            "--seed",
            "17",
            "--family-start",
            "2",
            "--sample-max-families",
            "4",
            "--max-events",
            "1000",
            "--backtrack-binary",
            str(backtrack_binary),
        ]
    )

    output = capsys.readouterr()
    config = captured_config["config"]
    assert config.checkpoint == checkpoint.resolve()
    assert config.out_dir == sample_out_dir.resolve()
    assert config.samples == 3
    assert config.seed == 17
    assert config.family_start == 2
    assert config.max_families == 4
    assert config.max_events == 1000
    assert config.backtrack_binary == backtrack_binary.resolve()
    assert "sampled_families=4 samples=3 xml=12" in output.out
    assert "sampled families=" not in output.out
    assert (
        gpurec_cli._optional_text("out_dir", sample_out_dir.resolve())
        in output.out
    )
    assert "Traceback" not in output.err


def test_cli_sample_require_mode_default_optimizer_accepts_default_route_checkpoint(
    tmp_path: Path,
    capsys,
    monkeypatch,
):
    class FakeModel:
        theta = torch.nn.Parameter(torch.zeros(1, 3))
        family_names = ["fam0"]
        species_names = ["sp0", "sp1"]

    config = RunConfig(
        species_tree=tmp_path / "sp.nwk",
        families_file=tmp_path / "families.txt",
        out_dir=tmp_path / "out",
        mode="genewise",
        device="cpu",
    )
    checkpoint = tmp_path / "best.pt"
    save_checkpoint(
        checkpoint,
        config=config,
        model=FakeModel(),
        optimizer=None,
        optimizer_phase="hessian-sgd",
        step=3,
        next_step=4,
        status={
            "status": "converged",
            "reason": "loss_change",
            "best_step": 3,
            "best_nll_bits": 10.0,
        },
        row={
            "optimizer/phase": "hessian-sgd",
            "likelihood/data_nll_bits": 10.0,
            "grad/inf": 0.1,
        },
    )

    captured_config = {}

    def capture_sample(sample_config):
        captured_config["config"] = sample_config
        return SimpleNamespace(
            families_sampled=1,
            samples_per_family=sample_config.samples,
            xml_files=2,
            out_dir=sample_config.out_dir,
        )

    monkeypatch.setattr("gpurec.cli.sample", capture_sample)

    main(
        [
            "sample",
            "--checkpoint",
            str(checkpoint),
            "--samples",
            "2",
            "--require-mode-default-optimizer",
        ]
    )

    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured_config["config"].checkpoint == checkpoint.resolve()
    assert "sampled_families=1 samples=2 xml=2 out_dir=null" in captured.out


def test_cli_sample_require_mode_default_optimizer_rejects_override_before_sampling(
    tmp_path: Path,
    capsys,
    monkeypatch,
):
    class FakeModel:
        theta = torch.nn.Parameter(torch.zeros(1, 3))
        family_names = ["fam0"]
        species_names = ["sp0", "sp1"]

    config = RunConfig(
        species_tree=tmp_path / "sp.nwk",
        families_file=tmp_path / "families.txt",
        out_dir=tmp_path / "out",
        mode="genewise",
        optimizer="adam",
        device="cpu",
    )
    checkpoint = tmp_path / "best.pt"
    save_checkpoint(
        checkpoint,
        config=config,
        model=FakeModel(),
        optimizer=None,
        optimizer_phase="adam",
        step=3,
        next_step=4,
        status={
            "status": "running",
            "reason": "checkpoint_interval",
            "best_step": 3,
            "best_nll_bits": 10.0,
        },
        row={
            "optimizer/phase": "adam",
            "likelihood/data_nll_bits": 10.0,
            "grad/inf": 0.1,
        },
    )

    def unexpected_sample(sample_config):
        raise AssertionError("sample should not be called")

    monkeypatch.setattr("gpurec.cli.sample", unexpected_sample)

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "sample",
                "--checkpoint",
                str(checkpoint),
                "--require-mode-default-optimizer",
            ]
        )

    captured = capsys.readouterr()
    assert exc_info.value.code == 1
    assert captured.out == ""
    assert "checkpoint optimizer is 'adam'" in captured.err
    assert "expected mode default 'hessian-sgd' for mode 'genewise'" in captured.err
    assert "sampled_families" not in captured.out
    assert "usage:" not in captured.err
    assert "Traceback" not in captured.err


def test_cli_sample_require_production_default_route_accepts_default_checkpoint(
    tmp_path: Path,
    capsys,
    monkeypatch,
):
    config = RunConfig(
        species_tree=tmp_path / "sp.nwk",
        families_file=tmp_path / "families.txt",
        out_dir=tmp_path / "out",
        mode="genewise",
        device="cpu",
    )
    checkpoint = _checkpoint_with_route_metadata(
        tmp_path,
        effective_route_metadata(config),
    )
    captured_config = {}

    def capture_sample(sample_config):
        captured_config["config"] = sample_config
        return SimpleNamespace(
            families_sampled=1,
            samples_per_family=sample_config.samples,
            xml_files=2,
            out_dir=sample_config.out_dir,
        )

    monkeypatch.setattr("gpurec.cli.sample", capture_sample)

    main(
        [
            "sample",
            "--checkpoint",
            str(checkpoint),
            "--samples",
            "2",
            "--require-production-default-route",
        ]
    )

    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured_config["config"].checkpoint == checkpoint.resolve()
    assert "sampled_families=1 samples=2 xml=2 out_dir=null" in captured.out


def test_cli_sample_combined_route_gates_load_checkpoint_once(
    tmp_path: Path,
    capsys,
    monkeypatch,
):
    from gpurec.workflow import checkpoint as checkpoint_module

    config = RunConfig(
        species_tree=tmp_path / "sp.nwk",
        families_file=tmp_path / "families.txt",
        out_dir=tmp_path / "out",
        mode="genewise",
        device="cpu",
    )
    checkpoint = _checkpoint_with_route_metadata(
        tmp_path,
        effective_route_metadata(config),
    )
    payload = checkpoint_module.load_checkpoint(checkpoint)
    load_calls: list[Path] = []
    captured_config = {}

    def capture_load(path):
        load_calls.append(Path(path))
        return payload

    def capture_sample(sample_config):
        captured_config["config"] = sample_config
        return SimpleNamespace(
            families_sampled=1,
            samples_per_family=sample_config.samples,
            xml_files=2,
            out_dir=sample_config.out_dir,
        )

    monkeypatch.setattr(checkpoint_module, "load_checkpoint", capture_load)
    monkeypatch.setattr("gpurec.cli.sample", capture_sample)

    main(
        [
            "sample",
            "--checkpoint",
            str(checkpoint),
            "--samples",
            "2",
            "--require-mode-default-optimizer",
            "--require-production-default-route",
        ]
    )

    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured_config["config"].checkpoint == checkpoint.resolve()
    assert load_calls == [checkpoint.resolve()]
    assert "sampled_families=1 samples=2 xml=2 out_dir=null" in captured.out


def test_cli_sample_require_production_default_route_rejects_stale_route(
    tmp_path: Path,
    capsys,
    monkeypatch,
):
    config = RunConfig(
        species_tree=tmp_path / "sp.nwk",
        families_file=tmp_path / "families.txt",
        out_dir=tmp_path / "out",
        mode="genewise",
        device="cpu",
    )
    route = effective_route_metadata(config)
    route["gradient_route"] = "legacy_autograd"
    checkpoint = _checkpoint_with_route_metadata(tmp_path, route)

    def unexpected_sample(sample_config):
        raise AssertionError("sample should not be called")

    monkeypatch.setattr("gpurec.cli.sample", unexpected_sample)

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "sample",
                "--checkpoint",
                str(checkpoint),
                "--require-production-default-route",
            ]
        )

    captured = capsys.readouterr()
    assert exc_info.value.code == 1
    assert captured.out == ""
    assert (
        "checkpoint production default route fields differ for mode 'genewise': "
        "gradient_route"
    ) in captured.err
    assert "expected the shipped likelihood/gradient and optimizer route" in (
        captured.err
    )
    assert "sampled_families" not in captured.out
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
            mode=config.mode,
            optimizer=config.optimizer,
            final_nll_bits=math.inf,
        )

    monkeypatch.setattr("gpurec.cli.optimize", failed_optimize)

    with pytest.raises(SystemExit) as exc_info:
        main(_minimal_workflow_cli_args("optimize", tmp_path))

    captured = capsys.readouterr()
    assert exc_info.value.code == 1
    assert "status=failed" in captured.out
    assert "nonfinite_objective_or_gradient" in captured.out
    assert "mode=genewise" in captured.out
    assert "optimizer=hessian-sgd" in captured.out
    assert "sampling_checkpoint=null" in captured.out
    assert "final_nll_bits=inf" in captured.out
    assert "final_log_likelihood_bits=null" in captured.out
    assert "final_grad_inf=null" in captured.out
    assert "final_projected_grad_inf=null" in captured.out
    assert "best_nll_bits=null" in captured.out
    assert "best_log_likelihood_bits=null" in captured.out
    assert "final_check_status=null" in captured.out
    assert "final_check_source=null" in captured.out
    assert "final_check_reason=null" in captured.out
    assert "final_check_fallback_clade_budget=null" in captured.out
    assert "final_check_loss_abs_delta_bits=null" in captured.out
    assert "final_check_grad_max_abs_delta=null" in captured.out
    assert "final_check_grad_rel_inf_delta=null" in captured.out
    assert "Traceback" not in captured.err


def test_cli_optimize_reports_final_and_best_objective_summary(
    tmp_path: Path,
    capsys,
    monkeypatch,
):
    basis = "hogenom_and_" + "test_trees_" + "1000"

    def successful_optimize(config):
        return SimpleNamespace(
            out_dir=config.out_dir,
            status="not_converged",
            reason="max_steps",
            mode=config.mode,
            optimizer=config.optimizer,
            families=3,
            species=4,
            batches=2,
            batch_packing=config.batch_packing,
            family_chunk_size=config.family_chunk_size,
            clade_budget=config.clade_budget,
            fixed_iters_e=config.fixed_iters_e,
            fixed_iters_pi=config.fixed_iters_pi,
            neumann_terms=config.neumann_terms,
            objective="negative_log_likelihood_bits",
            gradient_route="implicit_first_order_adjoint",
            rate_parameterization="base2_log_dlt_rates",
            production_default_basis=basis,
            configured_steps=config.steps,
            optimizer_step_cap=config.steps,
            optimizer_step_cap_reason="configured_steps",
            final_check_iters=config.final_check_iters,
            final_check_iters_e=None,
            solver_warmup_iters=config.solver_warmup_iters,
            fd_adam_warmup_steps=config.fd_adam_warmup_steps,
            fd_hessian_refresh_steps=config.fd_hessian_refresh_steps,
            hessian_sgd_normal_fixed_iters_pi=(
                config.hessian_sgd_normal_fixed_iters_pi
            ),
            hessian_sgd_normal_neumann_terms=(
                config.hessian_sgd_normal_neumann_terms
            ),
            hessian_sgd_pi_adjoint_warmstart=(
                config.hessian_sgd_pi_adjoint_warmstart
            ),
            pi_fixed_point_relaxation=config.pi_fixed_point_relaxation,
            hessian_sgd_validation_interval=config.hessian_sgd_validation_interval,
            hessian_sgd_validation_fixed_iters_pi=(
                config.hessian_sgd_validation_fixed_iters_pi
            ),
            hessian_sgd_validation_neumann_terms=(
                config.hessian_sgd_validation_neumann_terms
            ),
            steps_completed=7,
            elapsed_s=3.25,
            best_step=5,
            sampling_checkpoint=config.out_dir / "checkpoints" / "best.pt",
            final_nll_bits=12.5,
            final_grad_inf=0.75,
            final_projected_grad_inf=0.625,
            final_log_likelihood_bits=-12.5,
            best_nll_bits=10.25,
            best_log_likelihood_bits=-10.25,
            final_check_status="ok",
            final_check_source="fallback_clade_budget",
            final_check_reason="RuntimeError: scratch budget exceeded",
            final_check_fallback_clade_budget=250_000.0,
            final_check_loss_abs_delta_bits=0.125,
            final_check_grad_max_abs_delta=0.5,
            final_check_grad_rel_inf_delta=0.25,
        )

    monkeypatch.setattr("gpurec.cli.optimize", successful_optimize)

    args = _minimal_workflow_cli_args("optimize", tmp_path)
    out_dir = tmp_path / "optimized outputs"
    args[args.index("--out-dir") + 1] = str(out_dir)

    main(args)

    captured = capsys.readouterr()
    assert "status=not_converged" in captured.out
    assert "reason=max_steps" in captured.out
    assert "mode=genewise" in captured.out
    assert "optimizer=hessian-sgd" in captured.out
    assert "families=3" in captured.out
    assert "species=4" in captured.out
    assert "batches=2" in captured.out
    assert "batch_packing=depth_first_fit" in captured.out
    assert "family_chunk_size=0" in captured.out
    assert "clade_budget=500000" in captured.out
    assert "fixed_iters_e=null" in captured.out
    assert "fixed_iters_pi=16" in captured.out
    assert "neumann_terms=16" in captured.out
    assert "objective=negative_log_likelihood_bits" in captured.out
    assert "gradient_route=implicit_first_order_adjoint" in captured.out
    assert "rate_parameterization=base2_log_dlt_rates" in captured.out
    assert f"production_default_basis={basis}" in captured.out
    assert "configured_steps=5000" in captured.out
    assert "optimizer_step_cap=5000" in captured.out
    assert "optimizer_step_cap_reason=configured_steps" in captured.out
    assert "final_check_iters=32" in captured.out
    assert "final_check_iters_e=null" in captured.out
    assert "solver_warmup_iters=4" in captured.out
    assert "fd_adam_warmup_steps=3" in captured.out
    assert "fd_hessian_refresh_steps=16" in captured.out
    assert "hessian_sgd_normal_fixed_iters_pi=null" in captured.out
    assert "hessian_sgd_normal_neumann_terms=null" in captured.out
    assert "hessian_sgd_pi_adjoint_warmstart=false" in captured.out
    assert "pi_fixed_point_relaxation=1.000000" in captured.out
    assert "hessian_sgd_validation_interval=0" in captured.out
    assert "hessian_sgd_validation_fixed_iters_pi=null" in captured.out
    assert "hessian_sgd_validation_neumann_terms=null" in captured.out
    assert "steps_completed=7" in captured.out
    assert "elapsed_s=3.250000" in captured.out
    assert "best_step=5" in captured.out
    assert (
        gpurec_cli._optional_text(
            "sampling_checkpoint",
            out_dir.resolve() / "checkpoints" / "best.pt",
        )
        in captured.out
    )
    raw_checkpoint = f"sampling_checkpoint={out_dir.resolve()}/checkpoints/best.pt"
    assert raw_checkpoint not in captured.out
    assert "final_nll_bits=12.500000" in captured.out
    assert "final_log_likelihood_bits=-12.500000" in captured.out
    assert "final_grad_inf=0.750000" in captured.out
    assert "final_projected_grad_inf=0.625000" in captured.out
    assert "best_nll_bits=10.250000" in captured.out
    assert "best_log_likelihood_bits=-10.250000" in captured.out
    assert "final_check_status=ok" in captured.out
    assert "final_check_source=fallback_clade_budget" in captured.out
    assert (
        'final_check_reason="RuntimeError:\\u0020scratch\\u0020budget\\u0020exceeded"'
        in captured.out
    )
    assert "final_check_reason=RuntimeError: scratch budget exceeded" not in captured.out
    assert "final_check_fallback_clade_budget=250000.000000" in captured.out
    assert "final_check_loss_abs_delta_bits=0.125000" in captured.out
    assert "final_check_grad_max_abs_delta=0.500000" in captured.out
    assert "final_check_grad_rel_inf_delta=0.250000" in captured.out
    assert gpurec_cli._optional_text("out_dir", out_dir.resolve()) in captured.out
    assert f"out_dir={out_dir.resolve()}" not in captured.out
    assert "Traceback" not in captured.err


def test_cli_optimize_require_converged_accepts_converged_result(
    tmp_path: Path,
    capsys,
    monkeypatch,
):
    def converged_optimize(config):
        return SimpleNamespace(
            out_dir=config.out_dir,
            status="converged",
            reason="best_likelihood_patience",
            mode=config.mode,
            optimizer=config.optimizer,
            final_nll_bits=10.0,
            best_nll_bits=9.5,
        )

    monkeypatch.setattr("gpurec.cli.optimize", converged_optimize)

    main(_minimal_workflow_cli_args("optimize", tmp_path) + ["--require-converged"])

    captured = capsys.readouterr()
    assert "status=converged" in captured.out
    assert "reason=best_likelihood_patience" in captured.out
    assert "final_nll_bits=10.000000" in captured.out
    assert "best_nll_bits=9.500000" in captured.out
    assert captured.err == ""


def test_cli_optimize_require_mode_default_optimizer_rejects_override_before_run(
    tmp_path: Path,
    capsys,
    monkeypatch,
):
    def unexpected_optimize(config):
        raise AssertionError("optimize should not be called")

    monkeypatch.setattr("gpurec.cli.optimize", unexpected_optimize)

    with pytest.raises(SystemExit) as exc_info:
        main(
            _minimal_workflow_cli_args("optimize", tmp_path)
            + [
                "--optimizer",
                "adam",
                "--require-mode-default-optimizer",
            ]
        )

    captured = capsys.readouterr()
    assert exc_info.value.code == 2
    assert "config optimizer is 'adam'" in captured.err
    assert "expected mode default 'hessian-sgd' for mode 'genewise'" in captured.err
    assert "use optimizer=auto or the mode default optimizer" in captured.err
    assert "Traceback" not in captured.err


def test_cli_optimize_require_production_default_route_accepts_default_config(
    tmp_path: Path,
    capsys,
    monkeypatch,
):
    captured_config: dict[str, RunConfig] = {}

    def successful_optimize(config):
        captured_config["config"] = config
        fields = {
            **effective_route_metadata(config),
            "out_dir": config.out_dir,
            "status": "not_converged",
            "reason": "max_steps",
            "families": 1,
            "species": 2,
            "batches": 1,
            "steps_completed": 1,
            "elapsed_s": 0.5,
            "best_step": 1,
            "sampling_checkpoint": config.out_dir / "checkpoints" / "best.pt",
            "final_nll_bits": 12.0,
            "final_grad_inf": 0.25,
            "final_projected_grad_inf": 0.125,
            "best_nll_bits": 11.0,
        }
        return SimpleNamespace(**fields)

    monkeypatch.setattr("gpurec.cli.optimize", successful_optimize)

    main(
        _minimal_workflow_cli_args("optimize", tmp_path)
        + ["--require-production-default-route"]
    )

    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured_config["config"].optimizer == "hessian-sgd"
    assert "status=not_converged" in captured.out
    assert "mode=genewise" in captured.out
    assert "optimizer=hessian-sgd" in captured.out
    assert "uses_production_default_route=true" in captured.out
    assert "production_default_route_mismatches=none" in captured.out


def test_cli_optimize_require_production_default_route_accepts_specieswise_default(
    tmp_path: Path,
    capsys,
    monkeypatch,
):
    captured_config: dict[str, RunConfig] = {}

    def successful_optimize(config):
        captured_config["config"] = config
        fields = {
            **effective_route_metadata(config),
            "out_dir": config.out_dir,
            "status": "not_converged",
            "reason": "max_steps",
            "families": 1,
            "species": 2,
            "batches": 1,
            "steps_completed": 1,
            "elapsed_s": 0.5,
            "best_step": 1,
            "sampling_checkpoint": config.out_dir / "checkpoints" / "best.pt",
            "final_nll_bits": 12.0,
            "final_grad_inf": 0.25,
            "final_projected_grad_inf": 0.125,
            "best_nll_bits": 11.0,
        }
        return SimpleNamespace(**fields)

    monkeypatch.setattr("gpurec.cli.optimize", successful_optimize)

    main(
        _minimal_workflow_cli_args("optimize", tmp_path)
        + [
            "--mode",
            "specieswise",
            "--require-production-default-route",
        ]
    )

    captured = capsys.readouterr()
    assert captured.err == ""
    config = captured_config["config"]
    assert config.mode == "specieswise"
    assert config.optimizer == "adagrad-restarts"
    assert "status=not_converged" in captured.out
    assert "mode=specieswise" in captured.out
    assert "optimizer=adagrad-restarts" in captured.out
    assert "uses_production_default_route=true" in captured.out
    assert "production_default_route_mismatches=none" in captured.out
    assert "optimizer_step_cap=125" in captured.out
    assert "optimizer_step_cap_reason=adagrad_restart_schedule" in captured.out


def test_cli_optimize_require_production_default_route_rejects_custom_settings_before_run(
    tmp_path: Path,
    capsys,
    monkeypatch,
):
    def unexpected_optimize(config):
        raise AssertionError("optimize should not be called")

    monkeypatch.setattr("gpurec.cli.optimize", unexpected_optimize)

    with pytest.raises(SystemExit) as exc_info:
        main(
            _minimal_workflow_cli_args("optimize", tmp_path)
            + [
                "--optimizer",
                "hessian-sgd",
                "--fd-hessian-refresh-steps",
                "8",
                "--require-production-default-route",
            ]
        )

    captured = capsys.readouterr()
    assert exc_info.value.code == 2
    assert (
        "config production default route fields differ for mode 'genewise': "
        "fd_hessian_refresh_steps"
    ) in captured.err
    assert "use optimizer=auto and the shipped optimizer defaults" in captured.err
    assert captured.out == ""
    assert "usage:" in captured.err
    assert "Traceback" not in captured.err


def test_cli_optimize_require_converged_fails_after_printing_status(
    tmp_path: Path,
    capsys,
    monkeypatch,
):
    def not_converged_optimize(config):
        return SimpleNamespace(
            out_dir=config.out_dir,
            status="not_converged",
            reason="max_steps",
            mode=config.mode,
            optimizer=config.optimizer,
            final_nll_bits=12.0,
            best_nll_bits=11.0,
        )

    monkeypatch.setattr("gpurec.cli.optimize", not_converged_optimize)

    with pytest.raises(SystemExit) as exc_info:
        main(_minimal_workflow_cli_args("optimize", tmp_path) + ["--require-converged"])

    captured = capsys.readouterr()
    assert exc_info.value.code == 1
    assert "status=not_converged" in captured.out
    assert "reason=max_steps" in captured.out
    assert "mode=genewise" in captured.out
    assert "optimizer=hessian-sgd" in captured.out
    assert "final_nll_bits=12.000000" in captured.out
    assert "best_nll_bits=11.000000" in captured.out
    assert "optimization status is 'not_converged'; expected 'converged'" in captured.err
    assert "usage:" not in captured.err
    assert "Traceback" not in captured.err


def test_cli_optimize_require_final_check_ok_accepts_ok_result(
    tmp_path: Path,
    capsys,
    monkeypatch,
):
    def checked_optimize(config):
        return SimpleNamespace(
            out_dir=config.out_dir,
            status="not_converged",
            reason="max_steps",
            mode=config.mode,
            optimizer=config.optimizer,
            final_check_status="ok",
            final_check_source="configured_solver_budget",
            final_nll_bits=12.0,
            best_nll_bits=11.0,
        )

    monkeypatch.setattr("gpurec.cli.optimize", checked_optimize)

    main(
        _minimal_workflow_cli_args("optimize", tmp_path)
        + ["--require-final-check-ok"]
    )

    captured = capsys.readouterr()
    assert captured.err == ""
    assert "status=not_converged" in captured.out
    assert "final_check_status=ok" in captured.out
    assert "final_check_source=configured_solver_budget" in captured.out


def test_cli_optimize_require_final_check_ok_fails_after_printing_status(
    tmp_path: Path,
    capsys,
    monkeypatch,
):
    def unchecked_optimize(config):
        return SimpleNamespace(
            out_dir=config.out_dir,
            status="converged",
            reason="best_likelihood_patience",
            mode=config.mode,
            optimizer=config.optimizer,
            final_check_status="skipped",
            final_check_source="not_evaluated",
            final_nll_bits=12.0,
            best_nll_bits=11.0,
        )

    monkeypatch.setattr("gpurec.cli.optimize", unchecked_optimize)

    with pytest.raises(SystemExit) as exc_info:
        main(
            _minimal_workflow_cli_args("optimize", tmp_path)
            + ["--require-final-check-ok"]
        )

    captured = capsys.readouterr()
    assert exc_info.value.code == 1
    assert "status=converged" in captured.out
    assert "final_check_status=skipped" in captured.out
    assert "final_check_source=not_evaluated" in captured.out
    assert "optimization final check status is 'skipped'; expected 'ok'" in captured.err
    assert "usage:" not in captured.err
    assert "Traceback" not in captured.err


def test_optimization_result_text_reports_adagrad_restart_route_fields():
    text = gpurec_cli._optimization_result_text(
        SimpleNamespace(
            status="converged",
            reason="adagrad_restart_schedule_complete",
            mode="specieswise",
            optimizer="adagrad-restarts",
            final_check_iters=128,
            final_check_iters_e=128,
            adagrad_restart_schedule="8:1:60,16:0.5:35,32:0.5:30",
            adagrad_restart_total_steps=125,
            adagrad_restart_final_check_iters=128,
            steps_completed=125,
            final_nll_bits=12.0,
            final_grad_inf=0.5,
        )
    )

    assert "optimizer=adagrad-restarts" in text
    assert "final_check_iters=128" in text
    assert "final_check_iters_e=128" in text
    assert "adagrad_restart_schedule=8:1:60,16:0.5:35,32:0.5:30" in text
    assert "adagrad_restart_total_steps=125" in text
    assert "adagrad_restart_final_check_iters=128" in text
    assert "solver_warmup_iters" not in text


def test_cli_run_reports_optimize_errors_without_traceback(
    tmp_path: Path,
    capsys,
    monkeypatch,
):
    def fail_optimize(config):
        raise RuntimeError("workflow failed")

    monkeypatch.setattr("gpurec.cli.optimize", fail_optimize)
    monkeypatch.setattr("gpurec.cli._ensure_backtracking_available", lambda _: None)

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
    monkeypatch.setattr("gpurec.cli._ensure_backtracking_available", lambda _: None)

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
    basis = "hogenom_and_" + "test_trees_" + "1000"
    checkpoint_dir = tmp_path / "out" / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    sample_out_dir = tmp_path / "run sample outputs"
    stale_best = checkpoint_dir / "best.pt"
    current_latest = checkpoint_dir / "latest.pt"
    stale_best.write_bytes(b"stale")
    current_latest.write_bytes(b"current")
    sampled: dict[str, object] = {}

    def successful_optimize(config):
        return SimpleNamespace(
            out_dir=config.out_dir,
            sampling_checkpoint=current_latest,
            status="success",
            reason="completed",
            mode=config.mode,
            optimizer=config.optimizer,
            families=3,
            species=4,
            batches=2,
            batch_packing=config.batch_packing,
            family_chunk_size=config.family_chunk_size,
            clade_budget=config.clade_budget,
            fixed_iters_e=config.fixed_iters_e,
            fixed_iters_pi=config.fixed_iters_pi,
            neumann_terms=config.neumann_terms,
            objective="negative_log_likelihood_bits",
            gradient_route="implicit_first_order_adjoint",
            rate_parameterization="base2_log_dlt_rates",
            production_default_basis=basis,
            configured_steps=config.steps,
            optimizer_step_cap=config.steps,
            optimizer_step_cap_reason="configured_steps",
            final_check_iters=config.final_check_iters,
            solver_warmup_iters=config.solver_warmup_iters,
            fd_adam_warmup_steps=config.fd_adam_warmup_steps,
            fd_hessian_refresh_steps=config.fd_hessian_refresh_steps,
            hessian_sgd_normal_fixed_iters_pi=(
                config.hessian_sgd_normal_fixed_iters_pi
            ),
            hessian_sgd_normal_neumann_terms=(
                config.hessian_sgd_normal_neumann_terms
            ),
            hessian_sgd_pi_adjoint_warmstart=(
                config.hessian_sgd_pi_adjoint_warmstart
            ),
            pi_fixed_point_relaxation=config.pi_fixed_point_relaxation,
            hessian_sgd_validation_interval=config.hessian_sgd_validation_interval,
            hessian_sgd_validation_fixed_iters_pi=(
                config.hessian_sgd_validation_fixed_iters_pi
            ),
            hessian_sgd_validation_neumann_terms=(
                config.hessian_sgd_validation_neumann_terms
            ),
            steps_completed=3,
            elapsed_s=4.5,
            best_step=2,
            final_nll_bits=12.0,
            final_grad_inf=0.75,
            final_projected_grad_inf=0.5,
            best_nll_bits=11.0,
            final_check_status="ok",
            final_check_source="configured_solver_budget",
            final_check_loss_abs_delta_bits=0.0,
            final_check_grad_max_abs_delta=0.0,
            final_check_grad_rel_inf_delta=0.0,
        )

    def capture_sample(config):
        sampled["checkpoint"] = config.checkpoint
        sampled["out_dir"] = config.out_dir
        return SimpleNamespace(
            families_sampled=1,
            samples_per_family=config.samples,
            xml_files=2,
            out_dir=config.out_dir,
        )

    monkeypatch.setattr("gpurec.cli.optimize", successful_optimize)
    monkeypatch.setattr("gpurec.cli.sample", capture_sample)
    monkeypatch.setattr("gpurec.cli._ensure_backtracking_available", lambda _: None)

    main(
        _minimal_workflow_cli_args("run", tmp_path)
        + ["--sample-out-dir", str(sample_out_dir), "--samples", "3"]
    )

    captured = capsys.readouterr()
    assert "mode=genewise" in captured.out
    assert "optimizer=hessian-sgd" in captured.out
    assert "families=3" in captured.out
    assert "species=4" in captured.out
    assert "batches=2" in captured.out
    assert "batch_packing=depth_first_fit" in captured.out
    assert "family_chunk_size=0" in captured.out
    assert "clade_budget=500000" in captured.out
    assert "fixed_iters_e=null" in captured.out
    assert "fixed_iters_pi=16" in captured.out
    assert "neumann_terms=16" in captured.out
    assert "objective=negative_log_likelihood_bits" in captured.out
    assert "gradient_route=implicit_first_order_adjoint" in captured.out
    assert "rate_parameterization=base2_log_dlt_rates" in captured.out
    assert f"production_default_basis={basis}" in captured.out
    assert "configured_steps=5000" in captured.out
    assert "optimizer_step_cap=5000" in captured.out
    assert "optimizer_step_cap_reason=configured_steps" in captured.out
    assert "final_check_iters=32" in captured.out
    assert "solver_warmup_iters=4" in captured.out
    assert "fd_adam_warmup_steps=3" in captured.out
    assert "fd_hessian_refresh_steps=16" in captured.out
    assert "hessian_sgd_normal_fixed_iters_pi=null" in captured.out
    assert "hessian_sgd_normal_neumann_terms=null" in captured.out
    assert "hessian_sgd_pi_adjoint_warmstart=false" in captured.out
    assert "pi_fixed_point_relaxation=1.000000" in captured.out
    assert "hessian_sgd_validation_interval=0" in captured.out
    assert "hessian_sgd_validation_fixed_iters_pi=null" in captured.out
    assert "hessian_sgd_validation_neumann_terms=null" in captured.out
    assert "steps_completed=3" in captured.out
    assert "elapsed_s=4.500000" in captured.out
    assert "best_step=2" in captured.out
    assert gpurec_cli._optional_text(
        "sampling_checkpoint",
        current_latest.resolve(),
    ) in captured.out
    assert "final_nll_bits=12.000000" in captured.out
    assert "final_log_likelihood_bits=-12.000000" in captured.out
    assert "final_grad_inf=0.750000" in captured.out
    assert "final_projected_grad_inf=0.500000" in captured.out
    assert "best_nll_bits=11.000000" in captured.out
    assert "best_log_likelihood_bits=-11.000000" in captured.out
    assert "final_check_status=ok" in captured.out
    assert "final_check_source=configured_solver_budget" in captured.out
    assert "final_check_reason=null" in captured.out
    assert "final_check_fallback_clade_budget=null" in captured.out
    assert "final_check_loss_abs_delta_bits=0.000000" in captured.out
    assert "final_check_grad_max_abs_delta=0.000000" in captured.out
    assert "final_check_grad_rel_inf_delta=0.000000" in captured.out
    assert "sampled_families=1" in captured.out
    assert "samples=3" in captured.out
    assert "xml=2" in captured.out
    assert (
        gpurec_cli._optional_text("out_dir", (tmp_path / "out").resolve())
        in captured.out
    )
    assert (
        gpurec_cli._optional_text("sample_out_dir", sample_out_dir.resolve())
        in captured.out
    )
    assert sampled["checkpoint"] == current_latest.resolve()
    assert sampled["out_dir"] == sample_out_dir.resolve()


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


def test_cli_run_preflights_backtracking_before_optimization(
    tmp_path: Path,
    capsys,
    monkeypatch,
):
    calls: list[Path | None] = []

    def fail_preflight(backtrack_binary: Path | None) -> None:
        calls.append(backtrack_binary)
        raise RuntimeError("set GPUREC_BACKTRACK_BIN or pass --backtrack-binary")

    def unexpected_optimize(config):
        raise AssertionError("optimize should not be called")

    monkeypatch.setattr("gpurec.cli._ensure_backtracking_available", fail_preflight)
    monkeypatch.setattr("gpurec.cli.optimize", unexpected_optimize)

    with pytest.raises(SystemExit) as exc_info:
        main(_minimal_workflow_cli_args("run", tmp_path))

    captured = capsys.readouterr()
    assert exc_info.value.code == 1
    assert "GPUREC_BACKTRACK_BIN" in captured.err
    assert "--backtrack-binary" in captured.err
    assert "usage:" not in captured.err
    assert "Traceback" not in captured.err
    assert calls == [None]


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
            mode=config.mode,
            optimizer=config.optimizer,
            final_nll_bits=math.inf,
        )

    def unexpected_sample(config):
        raise AssertionError("sample should not be called")

    monkeypatch.setattr("gpurec.cli.optimize", failed_optimize)
    monkeypatch.setattr("gpurec.cli.sample", unexpected_sample)
    monkeypatch.setattr("gpurec.cli._ensure_backtracking_available", lambda _: None)

    with pytest.raises(SystemExit) as exc_info:
        main(_minimal_workflow_cli_args("run", tmp_path))

    captured = capsys.readouterr()
    assert exc_info.value.code == 1
    assert "status=failed" in captured.out
    assert "reason=nonfinite_objective_or_gradient" in captured.out
    assert "mode=genewise" in captured.out
    assert "optimizer=hessian-sgd" in captured.out
    assert "sampling_checkpoint=null" in captured.out
    assert "final_nll_bits=inf" in captured.out
    assert "final_log_likelihood_bits=null" in captured.out
    assert "final_grad_inf=null" in captured.out
    assert "final_projected_grad_inf=null" in captured.out
    assert "best_nll_bits=null" in captured.out
    assert "best_log_likelihood_bits=null" in captured.out
    assert gpurec_cli._optional_text(
        "out_dir",
        (tmp_path / "out").resolve(),
    ) in captured.out
    assert "sampled_families" not in captured.out
    assert "sample_out_dir" not in captured.out
    assert "optimization failed" in captured.err
    assert "nonfinite_objective_or_gradient" in captured.err
    assert "Traceback" not in captured.err


def test_cli_run_require_converged_refuses_not_converged_sampling(
    tmp_path: Path,
    capsys,
    monkeypatch,
):
    checkpoint_dir = tmp_path / "out" / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "latest.pt").write_bytes(b"not sampled")

    def not_converged_optimize(config):
        return SimpleNamespace(
            out_dir=config.out_dir,
            status="not_converged",
            reason="max_steps",
            mode=config.mode,
            optimizer=config.optimizer,
            sampling_checkpoint=checkpoint_dir / "latest.pt",
            final_nll_bits=12.0,
            best_nll_bits=11.0,
        )

    def unexpected_sample(config):
        raise AssertionError("sample should not be called")

    monkeypatch.setattr("gpurec.cli.optimize", not_converged_optimize)
    monkeypatch.setattr("gpurec.cli.sample", unexpected_sample)
    monkeypatch.setattr("gpurec.cli._ensure_backtracking_available", lambda _: None)

    with pytest.raises(SystemExit) as exc_info:
        main(_minimal_workflow_cli_args("run", tmp_path) + ["--require-converged"])

    captured = capsys.readouterr()
    assert exc_info.value.code == 1
    assert "status=not_converged" in captured.out
    assert "reason=max_steps" in captured.out
    assert "mode=genewise" in captured.out
    assert "optimizer=hessian-sgd" in captured.out
    assert "final_nll_bits=12.000000" in captured.out
    assert "best_nll_bits=11.000000" in captured.out
    assert "sampled_families" not in captured.out
    assert "sample_out_dir" not in captured.out
    assert (
        "optimization status is 'not_converged'; expected 'converged'; "
        "refusing to sample"
    ) in captured.err
    assert "usage:" not in captured.err
    assert "Traceback" not in captured.err


def test_cli_run_require_final_check_ok_refuses_unchecked_sampling(
    tmp_path: Path,
    capsys,
    monkeypatch,
):
    checkpoint_dir = tmp_path / "out" / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "latest.pt").write_bytes(b"not sampled")

    def unchecked_optimize(config):
        return SimpleNamespace(
            out_dir=config.out_dir,
            status="converged",
            reason="best_likelihood_patience",
            mode=config.mode,
            optimizer=config.optimizer,
            sampling_checkpoint=checkpoint_dir / "latest.pt",
            final_check_status="disabled",
            final_check_source="not_evaluated",
            final_nll_bits=12.0,
            best_nll_bits=11.0,
        )

    def unexpected_sample(config):
        raise AssertionError("sample should not be called")

    monkeypatch.setattr("gpurec.cli.optimize", unchecked_optimize)
    monkeypatch.setattr("gpurec.cli.sample", unexpected_sample)
    monkeypatch.setattr("gpurec.cli._ensure_backtracking_available", lambda _: None)

    with pytest.raises(SystemExit) as exc_info:
        main(
            _minimal_workflow_cli_args("run", tmp_path)
            + ["--require-final-check-ok"]
        )

    captured = capsys.readouterr()
    assert exc_info.value.code == 1
    assert "status=converged" in captured.out
    assert "final_check_status=disabled" in captured.out
    assert "final_check_source=not_evaluated" in captured.out
    assert "sampled_families" not in captured.out
    assert "sample_out_dir" not in captured.out
    assert (
        "optimization final check status is 'disabled'; expected 'ok'; "
        "refusing to sample"
    ) in captured.err
    assert "usage:" not in captured.err
    assert "Traceback" not in captured.err


def test_cli_run_require_mode_default_optimizer_rejects_override_before_run(
    tmp_path: Path,
    capsys,
    monkeypatch,
):
    def unexpected_optimize(config):
        raise AssertionError("optimize should not be called")

    def unexpected_sample(config):
        raise AssertionError("sample should not be called")

    monkeypatch.setattr("gpurec.cli.optimize", unexpected_optimize)
    monkeypatch.setattr("gpurec.cli.sample", unexpected_sample)
    monkeypatch.setattr("gpurec.cli._ensure_backtracking_available", lambda _: None)

    with pytest.raises(SystemExit) as exc_info:
        main(
            _minimal_workflow_cli_args("run", tmp_path)
            + [
                "--optimizer",
                "adam",
                "--require-mode-default-optimizer",
            ]
        )

    captured = capsys.readouterr()
    assert exc_info.value.code == 2
    assert "config optimizer is 'adam'" in captured.err
    assert "expected mode default 'hessian-sgd' for mode 'genewise'" in captured.err
    assert "sampled_families" not in captured.out
    assert "Traceback" not in captured.err


def test_cli_run_require_production_default_route_accepts_default_config(
    tmp_path: Path,
    capsys,
    monkeypatch,
):
    captured_objects: dict[str, object] = {}

    def successful_optimize(config):
        checkpoint = config.out_dir / "checkpoints" / "latest.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"checkpoint")
        captured_objects["optimize_config"] = config
        fields = {
            **effective_route_metadata(config),
            "out_dir": config.out_dir,
            "sampling_checkpoint": checkpoint,
            "status": "success",
            "reason": "completed",
            "families": 1,
            "species": 2,
            "batches": 1,
            "steps_completed": 1,
            "elapsed_s": 0.5,
            "best_step": 1,
            "final_nll_bits": 12.0,
            "final_grad_inf": 0.25,
            "final_projected_grad_inf": 0.125,
            "best_nll_bits": 11.0,
        }
        return SimpleNamespace(**fields)

    def capture_sample(config):
        captured_objects["sample_config"] = config
        return SimpleNamespace(
            families_sampled=1,
            samples_per_family=config.samples,
            xml_files=2,
            out_dir=config.out_dir,
        )

    monkeypatch.setattr("gpurec.cli.optimize", successful_optimize)
    monkeypatch.setattr("gpurec.cli.sample", capture_sample)
    monkeypatch.setattr("gpurec.cli._ensure_backtracking_available", lambda _: None)

    main(
        _minimal_workflow_cli_args("run", tmp_path)
        + [
            "--samples",
            "2",
            "--require-production-default-route",
        ]
    )

    captured = capsys.readouterr()
    optimize_config = captured_objects["optimize_config"]
    sample_config = captured_objects["sample_config"]
    assert captured.err == ""
    assert isinstance(optimize_config, RunConfig)
    assert optimize_config.optimizer == "hessian-sgd"
    assert isinstance(sample_config, SamplingConfig)
    assert sample_config.checkpoint == (
        tmp_path / "out" / "checkpoints" / "latest.pt"
    ).resolve()
    assert "uses_production_default_route=true" in captured.out
    assert "production_default_route_mismatches=none" in captured.out
    assert "sampled_families=1 samples=2 xml=2" in captured.out


def test_cli_run_require_production_default_route_accepts_specieswise_default(
    tmp_path: Path,
    capsys,
    monkeypatch,
):
    captured_objects: dict[str, object] = {}

    def successful_optimize(config):
        checkpoint = config.out_dir / "checkpoints" / "latest.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"checkpoint")
        captured_objects["optimize_config"] = config
        fields = {
            **effective_route_metadata(config),
            "out_dir": config.out_dir,
            "sampling_checkpoint": checkpoint,
            "status": "success",
            "reason": "completed",
            "families": 1,
            "species": 2,
            "batches": 1,
            "steps_completed": 1,
            "elapsed_s": 0.5,
            "best_step": 1,
            "final_nll_bits": 12.0,
            "final_grad_inf": 0.25,
            "final_projected_grad_inf": 0.125,
            "best_nll_bits": 11.0,
        }
        return SimpleNamespace(**fields)

    def capture_sample(config):
        captured_objects["sample_config"] = config
        return SimpleNamespace(
            families_sampled=1,
            samples_per_family=config.samples,
            xml_files=2,
            out_dir=config.out_dir,
        )

    monkeypatch.setattr("gpurec.cli.optimize", successful_optimize)
    monkeypatch.setattr("gpurec.cli.sample", capture_sample)
    monkeypatch.setattr("gpurec.cli._ensure_backtracking_available", lambda _: None)

    main(
        _minimal_workflow_cli_args("run", tmp_path)
        + [
            "--mode",
            "specieswise",
            "--samples",
            "2",
            "--require-production-default-route",
        ]
    )

    captured = capsys.readouterr()
    optimize_config = captured_objects["optimize_config"]
    sample_config = captured_objects["sample_config"]
    assert captured.err == ""
    assert isinstance(optimize_config, RunConfig)
    assert optimize_config.mode == "specieswise"
    assert optimize_config.optimizer == "adagrad-restarts"
    assert isinstance(sample_config, SamplingConfig)
    assert sample_config.checkpoint == (
        tmp_path / "out" / "checkpoints" / "latest.pt"
    ).resolve()
    assert "mode=specieswise" in captured.out
    assert "optimizer=adagrad-restarts" in captured.out
    assert "uses_production_default_route=true" in captured.out
    assert "production_default_route_mismatches=none" in captured.out
    assert "optimizer_step_cap=125" in captured.out
    assert "optimizer_step_cap_reason=adagrad_restart_schedule" in captured.out
    assert "sampled_families=1 samples=2 xml=2" in captured.out


def test_cli_run_require_production_default_route_rejects_custom_settings_before_run(
    tmp_path: Path,
    capsys,
    monkeypatch,
):
    def unexpected_backtracking_check(backtrack_binary):
        raise AssertionError("backtracking should not be checked")

    def unexpected_optimize(config):
        raise AssertionError("optimize should not be called")

    def unexpected_sample(config):
        raise AssertionError("sample should not be called")

    monkeypatch.setattr(
        "gpurec.cli._ensure_backtracking_available",
        unexpected_backtracking_check,
    )
    monkeypatch.setattr("gpurec.cli.optimize", unexpected_optimize)
    monkeypatch.setattr("gpurec.cli.sample", unexpected_sample)

    with pytest.raises(SystemExit) as exc_info:
        main(
            _minimal_workflow_cli_args("run", tmp_path)
            + [
                "--optimizer",
                "hessian-sgd",
                "--fd-hessian-refresh-steps",
                "8",
                "--require-production-default-route",
            ]
        )

    captured = capsys.readouterr()
    assert exc_info.value.code == 2
    assert (
        "config production default route fields differ for mode 'genewise': "
        "fd_hessian_refresh_steps"
    ) in captured.err
    assert "use optimizer=auto and the shipped optimizer defaults" in captured.err
    assert captured.out == ""
    assert "sampled_families" not in captured.out
    assert "usage:" in captured.err
    assert "Traceback" not in captured.err


def test_cli_run_rejects_checkpoint_argument_without_traceback(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["run", "--checkpoint", "existing.pt"])

    captured = capsys.readouterr()
    assert exc_info.value.code == 2
    assert "gpurec sample --checkpoint" in captured.err
    assert "--resume-from" in captured.err
    _assert_subcommand_usage(captured.err, "run")
    assert "Traceback" not in captured.err


def test_cli_run_help_omits_checkpoint_argument(capsys):
    with pytest.raises(SystemExit) as exc_info:
        build_parser().parse_args(["run", "--help"])

    captured = capsys.readouterr()
    assert exc_info.value.code == 0
    assert "Run optimization, then sample" in captured.out
    assert "Samples per selected family" in captured.out
    assert "--sample-out-dir" in captured.out
    assert "--checkpoint CHECKPOINT" not in captured.out
    assert "--resume-from" in captured.out
    assert "--require-converged" in captured.out
    assert "--require-final-check-ok" in captured.out
    assert "--require-mode-default-optimizer" in captured.out
    assert "--require-production-default-route" in captured.out
    assert "--backtrack-binary" in captured.out
    assert "GPUREC_BACKTRACK_BIN" in captured.out


def test_cli_backtrack_check_delegates_binary_preflight(
    tmp_path: Path,
    capsys,
    monkeypatch,
):
    calls: list[Path | None] = []
    binary = tmp_path / "gpurec-backtrack"

    monkeypatch.setattr(
        "gpurec.cli._ensure_backtracking_available",
        lambda backtrack_binary: calls.append(backtrack_binary),
    )

    main(["backtrack-check", "--backtrack-binary", str(binary)])

    captured = capsys.readouterr()
    assert captured.out == "backtracking_available=true\n"
    assert captured.err == ""
    assert calls == [binary]


def test_cli_backtrack_check_reports_missing_binary_without_traceback(
    capsys,
    monkeypatch,
):
    def fail_preflight(backtrack_binary: Path | None) -> None:
        assert backtrack_binary is None
        raise RuntimeError(
            "set GPUREC_BACKTRACK_BIN or pass --backtrack-binary"
        )

    monkeypatch.setattr("gpurec.cli._ensure_backtracking_available", fail_preflight)

    with pytest.raises(SystemExit) as exc_info:
        main(["backtrack-check"])

    captured = capsys.readouterr()
    assert exc_info.value.code == 1
    assert "GPUREC_BACKTRACK_BIN" in captured.err
    assert "--backtrack-binary" in captured.err
    assert "Traceback" not in captured.err


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
    assert "fp32/single" in captured.out
    assert "fp64/double" in captured.out
    assert "0/all/none/null" in captured.out
    assert "contiguous/input_order" in captured.out
    assert "ffd/clade_ffd" in captured.out
    assert "depth_ffd/wave_first_fit" in captured.out
    assert "Disabled compatibility flag" in captured.out
    assert "adaptive Neumann path is not part of the" in captured.out
    assert "supported production optimization route" in captured.out
    assert "--require-converged" in captured.out
    assert "--require-final-check-ok" in captured.out
    assert "--require-mode-default-optimizer" in captured.out
    assert "--require-production-default-route" in captured.out


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
    assert "--require-mode-default-optimizer" in captured.out
    assert "--require-production-default-route" in captured.out
