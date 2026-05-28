from __future__ import annotations

import subprocess
from pathlib import Path

import gpurec
import gpurec.workflow as workflow
import gpurec.cli as cli
from gpurec.core.model import parse_alerax_family_file
from gpurec.workflow.config import (
    DEFAULT_ADAGRAD_RESTART_SCHEDULE,
    RunConfig,
    adagrad_restart_schedule_specs,
    production_default_optimizer_config_overrides,
)


ROOT = Path(__file__).resolve().parents[2]
SUBPROCESS_TIMEOUT = 30


def test_top_level_workflow_shortcuts_match_readme_api():
    for name in (
        "RunConfig",
        "SamplingConfig",
        "OptimizationResult",
        "OptimizationRunner",
        "SamplingResult",
        "SamplingRunner",
        "optimize",
        "sample",
    ):
        assert getattr(gpurec, name) is getattr(workflow, name)


def test_minimal_run_config_example_loads_and_points_to_tiny_fixture():
    config = RunConfig.from_json(ROOT / "examples" / "minimal-run-config.json")

    assert config.species_tree == (
        ROOT / "examples" / "tiny" / "species.nwk"
    ).resolve()
    assert config.families_file == (
        ROOT / "examples" / "tiny" / "families.txt"
    ).resolve()
    assert config.mode == "genewise"
    assert config.device == "cuda"
    assert config.optimizer == "hessian-sgd"
    assert config.max_families == 1
    assert config.steps == 10
    for name, expected in production_default_optimizer_config_overrides(
        "genewise"
    ).items():
        assert getattr(config, name) == expected
    assert config.solver_warmup_iters == 4
    assert config.fd_adam_warmup_steps == 3
    assert config.fd_hessian_refresh_steps == 16
    assert config.hessian_sgd_normal_fixed_iters_pi is None
    assert config.hessian_sgd_normal_neumann_terms is None
    assert config.hessian_sgd_pi_adjoint_warmstart is False
    assert config.pi_fixed_point_relaxation == 1.0
    assert config.hessian_sgd_validation_interval == 0
    assert config.hessian_sgd_validation_fixed_iters_pi is None
    assert config.hessian_sgd_validation_neumann_terms is None
    assert config.final_check_iters == 32

    names, tree_paths, leaf_maps = parse_alerax_family_file(config.families_file)
    assert names == ["tiny_family"]
    assert tree_paths == [
        [str((ROOT / "examples" / "tiny" / "gene.nwk").resolve())]
    ]
    assert leaf_maps == [{"a": "A", "b": "B"}]


def test_specieswise_adagrad_restart_example_loads_and_uses_auto_default():
    config = RunConfig.from_json(
        ROOT / "examples" / "specieswise-adagrad-restarts-config.json"
    )

    assert config.species_tree == (
        ROOT / "examples" / "tiny" / "species.nwk"
    ).resolve()
    assert config.families_file == (
        ROOT / "examples" / "tiny" / "families.txt"
    ).resolve()
    assert config.out_dir == (
        ROOT / "examples" / "output" / "specieswise-adagrad-restarts"
    ).resolve()
    assert config.mode == "specieswise"
    assert config.device == "cuda"
    assert config.optimizer == "adagrad-restarts"
    for name, expected in production_default_optimizer_config_overrides(
        "specieswise"
    ).items():
        actual = getattr(config, name)
        if name == "adagrad_restart_schedule":
            assert adagrad_restart_schedule_specs(
                actual
            ) == adagrad_restart_schedule_specs(expected)
        else:
            assert actual == expected
    assert adagrad_restart_schedule_specs(
        config.adagrad_restart_schedule
    ) == adagrad_restart_schedule_specs(DEFAULT_ADAGRAD_RESTART_SCHEDULE)
    assert config.adagrad_restart_final_check_iters == 128
    assert config.max_families == 1


def test_examples_readme_documents_mode_specific_default_configs():
    readme = (ROOT / "examples" / "README.md").read_text(encoding="utf-8")
    normalized = " ".join(readme.split())

    for token in (
        "source-checkout and source-archive fixtures",
        "without constructing the CUDA likelihood model",
        "not end-to-end optimizer smokes",
        "gpurec validate-config --config examples/minimal-run-config.json",
        "gpurec validate-config --config examples/specieswise-adagrad-restarts-config.json",
        "gpurec validate-config --config examples/minimal-run-config.json --require-mode-default-optimizer --require-production-default-route",
        "gpurec validate-config --config examples/specieswise-adagrad-restarts-config.json --require-mode-default-optimizer --require-production-default-route",
        "gpurec validate-config --config examples/minimal-run-config.json --require-mode-default-optimizer --require-production-default-route --check-preprocess",
        "gpurec validate-config --config examples/specieswise-adagrad-restarts-config.json --require-mode-default-optimizer --require-production-default-route --check-preprocess",
        "`preprocess_checked=true`",
        "`preprocessed_species_nodes=3`",
        "`cuda_backward_ready=false`",
        "`cuda_backward_ready_reason=requires_s_gt_256`",
        "expected to exit with status 2 and empty stdout",
        "real launch configs",
        "`optimizer=auto` resolves to `hessian-sgd`",
        "Hessian-SGD warmup",
        "normal-stage solver overrides",
        "Pi-adjoint warmstart and relaxation defaults",
        "periodic validation defaults",
        "`optimizer=auto` resolves to `adagrad-restarts`",
        "`8:1.0:60,16:0.5:35,32:0.5:30`",
        "fixed128 final validation",
        "`final_check_iters_e=128`",
        "Installed wheels intentionally do not install this directory",
        "gpurec config-template --mode genewise --output run.json",
        "gpurec config-template --mode specieswise --output specieswise-run.json",
    ):
        assert token in normalized


def test_example_configs_validate_through_cli_production_route_gates(capsys):
    production_basis = "hogenom_and_" + "test_trees_" + "1000"
    cases = (
        (
            "minimal-run-config.json",
            (
                "mode=genewise",
                "optimizer=hessian-sgd",
                "mode_default_optimizer=hessian-sgd",
                "uses_mode_default_optimizer=true",
                "uses_production_default_optimizer_settings=true",
                "production_default_optimizer_setting_mismatches=none",
                "configured_steps=10",
                "optimizer_step_cap=10",
                "optimizer_step_cap_reason=configured_steps",
                "hessian_sgd_normal_fixed_iters_pi=full",
                "hessian_sgd_normal_neumann_terms=full",
                "hessian_sgd_pi_adjoint_warmstart=false",
                "pi_fixed_point_relaxation=1.000000",
                "hessian_sgd_validation_interval=0",
                "final_check_iters=32",
            ),
        ),
        (
            "specieswise-adagrad-restarts-config.json",
            (
                "mode=specieswise",
                "optimizer=adagrad-restarts",
                "mode_default_optimizer=adagrad-restarts",
                "uses_mode_default_optimizer=true",
                "uses_production_default_optimizer_settings=true",
                "production_default_optimizer_setting_mismatches=none",
                "adagrad_restart_schedule=8:1:60,16:0.5:35,32:0.5:30",
                "adagrad_restart_total_steps=125",
                "optimizer_step_cap=125",
                "optimizer_step_cap_reason=adagrad_restart_schedule",
                "adagrad_restart_final_check_iters=128",
                "final_check_iters=128",
            ),
        ),
    )

    for config_name, expected_tokens in cases:
        cli.main(
            [
                "validate-config",
                "--config",
                str(ROOT / "examples" / config_name),
                "--require-mode-default-optimizer",
                "--require-production-default-route",
            ]
        )
        captured = capsys.readouterr()

        assert captured.err == ""
        assert "valid_config=true" in captured.out
        assert f"production_default_basis={production_basis}" in captured.out
        assert "preprocess_checked" not in captured.out
        for token in expected_tokens:
            assert token in captured.out


def test_minimal_run_config_command_smoke_uses_documented_cli(monkeypatch):
    captured: dict[str, object] = {}

    def fake_optimize(config: RunConfig):
        captured["config"] = config
        return workflow.OptimizationResult(
            out_dir=config.out_dir,
            status="completed",
            reason="fixture",
            final_nll_bits=0.0,
            final_grad_inf=0.0,
            best_nll_bits=0.0,
            best_step=0,
            steps_completed=0,
            sampling_checkpoint=config.out_dir / "checkpoints" / "best.pt",
        )

    monkeypatch.setattr(cli, "optimize", fake_optimize)

    cli.main(["optimize", "--config", str(ROOT / "examples" / "minimal-run-config.json")])

    config = captured["config"]
    assert isinstance(config, RunConfig)
    assert config.species_tree == (ROOT / "examples" / "tiny" / "species.nwk").resolve()
    assert config.families_file == (
        ROOT / "examples" / "tiny" / "families.txt"
    ).resolve()
    assert config.out_dir == (ROOT / "examples" / "output" / "minimal-run").resolve()


def test_minimal_run_config_outputs_are_gitignored():
    config = RunConfig.from_json(ROOT / "examples" / "minimal-run-config.json")
    output_paths = [config.out_dir]

    for output_path in output_paths:
        assert output_path is not None
        relative = output_path.relative_to(ROOT).as_posix()
        result = subprocess.run(
            ["git", "check-ignore", "--quiet", relative],
            cwd=ROOT,
            check=False,
            timeout=SUBPROCESS_TIMEOUT,
        )
        assert result.returncode == 0, relative


def test_workflow_examples_include_post_optimization_quality_gates():
    snakemake = (
        ROOT / "docs" / "workflow-examples" / "snakemake" / "Snakefile"
    ).read_text(encoding="utf-8")
    nextflow = (
        ROOT / "docs" / "workflow-examples" / "nextflow" / "main.nf"
    ).read_text(encoding="utf-8")
    slurm = (
        ROOT / "docs" / "workflow-examples" / "slurm" / "run-gpurec.sbatch"
    ).read_text(encoding="utf-8")

    assert "gpurec optimize --config {RUN_CONFIG} \\" in snakemake
    assert "--require-converged \\" in snakemake
    assert "gpurec summary-info --summary {input.summary} \\" in snakemake
    assert "gpurec checkpoint-info --checkpoint {input.checkpoint} \\" in snakemake
    assert "--require-final-check-ok" in snakemake

    assert "process inspect {" in nextflow
    assert "gpurec optimize \\" in nextflow
    assert "--require-converged \\" in nextflow
    assert "gpurec summary-info --summary ${summary} \\" in nextflow
    assert "gpurec checkpoint-info --checkpoint ${checkpoint} \\" in nextflow

    assert "gpurec optimize \\" in slurm
    assert "--require-converged \\" in slurm
    assert "gpurec summary-info \\" in slurm
    assert "--summary output_gpurec/summary.json \\" in slurm
    assert "gpurec checkpoint-info \\" in slurm
    assert "--checkpoint output_gpurec/checkpoints/best.pt \\" in slurm
