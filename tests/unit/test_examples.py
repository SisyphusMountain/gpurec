from __future__ import annotations

import subprocess
from pathlib import Path

import gpurec
import gpurec.workflow as workflow
import gpurec.cli as cli
from gpurec.core.model import parse_alerax_family_file
from gpurec.workflow.config import RunConfig


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
    assert config.max_families == 1
    assert config.steps == 10

    names, tree_paths, leaf_maps = parse_alerax_family_file(config.families_file)
    assert names == ["tiny_family"]
    assert tree_paths == [
        [str((ROOT / "examples" / "tiny" / "gene.nwk").resolve())]
    ]
    assert leaf_maps == [{"a": "A", "b": "B"}]


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
