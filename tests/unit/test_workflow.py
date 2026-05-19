from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import gpurec
import gpurec.backtracking as backtracking
import gpurec.workflow.model_factory as workflow_model_factory
import gpurec.workflow.sampling as sampling_workflow
from gpurec.backtracking import (
    EVENT_KEYS,
    _activate_family_batch,
    _backtrack_command,
    export_backtracking_input,
    recphyloxml_event_counts,
    sample_recphyloxml,
    sample_recphyloxmls,
)
from gpurec.cli import _run_config_from_args, build_parser, main
from gpurec.core.batch_planning import normalize_family_chunk_size
from gpurec.api import (
    ActiveFamilyBatch,
    BatchMetadata,
    FamilyInput,
    ReconciliationState,
)
from gpurec.api.model import GeneReconModel
from gpurec.api.uniform_chunked import UniformChunkedReconModel
from gpurec.workflow.checkpoint import load_checkpoint, restore_model_theta, save_checkpoint
from gpurec.workflow.config import RunConfig, SamplingConfig
from gpurec.workflow.diagnostics import parameter_stats
from gpurec.workflow.model_factory import build_alerax_workflow_model
from gpurec.workflow.optimize import OptimizationRunner, _write_rate_table
from gpurec.workflow.sampling import SamplingRunner, _xml_species_and_transfer_counts


def test_run_config_json_roundtrip(tmp_path: Path):
    config = RunConfig(
        species_tree=tmp_path / "sp.nwk",
        families_file=tmp_path / "families.txt",
        out_dir=tmp_path / "out",
        mode="genewise",
        steps=3,
        family_chunk_size=2,
        clade_budget=None,
        batch_packing="sequential",
        device="cpu",
    )

    path = tmp_path / "config.json"
    config.write_json(path)
    loaded = RunConfig.from_json(path)

    assert loaded.to_dict() == config.to_dict()
    assert loaded.species_tree.is_absolute()
    assert loaded.families_file.is_absolute()
    assert loaded.out_dir.is_absolute()


def test_top_level_exports_api_metadata_types():
    assert gpurec.ActiveFamilyBatch is ActiveFamilyBatch
    assert gpurec.BatchMetadata is BatchMetadata
    assert gpurec.FamilyInput is FamilyInput
    assert gpurec.ReconciliationState is ReconciliationState
    for name in (
        "ActiveFamilyBatch",
        "BatchMetadata",
        "FamilyInput",
        "ReconciliationState",
    ):
        assert name in gpurec.__all__


def test_run_config_normalizes_batch_controls(tmp_path: Path):
    config = RunConfig(
        species_tree=tmp_path / "sp.nwk",
        families_file=tmp_path / "families.txt",
        out_dir=tmp_path / "out",
        family_chunk_size="all",
        batch_packing="depth-first-fit",
        clade_budget="12",
        max_wave_size="32",
        device="cpu",
    )

    assert config.family_chunk_size == 0
    assert config.batch_packing == "depth_first_fit"
    assert config.clade_budget == 12
    assert config.max_wave_size == 32


def test_run_config_defaults_to_cuda_for_production_workflow(tmp_path: Path):
    config = RunConfig(
        species_tree=tmp_path / "sp.nwk",
        families_file=tmp_path / "families.txt",
        out_dir=tmp_path / "out",
    )

    assert config.device == "cuda"


def test_run_config_rejects_unsupported_auto_chunking(tmp_path: Path):
    with pytest.raises(ValueError, match="auto"):
        RunConfig(
            species_tree=tmp_path / "sp.nwk",
            families_file=tmp_path / "families.txt",
            out_dir=tmp_path / "out",
            family_chunk_size="auto",
            device="cpu",
        )


@pytest.mark.parametrize(
    "field",
    [
        "tol_e",
        "e_logsumexp_tol",
        "pi_max_diff_tol",
        "gradient_change_tol",
        "gradient_change_rtol",
        "grad_inf_tol",
        "loss_change_tol",
        "best_likelihood_min_delta",
    ],
)
def test_run_config_rejects_negative_tolerances(tmp_path: Path, field: str):
    with pytest.raises(ValueError, match=field):
        RunConfig(
            species_tree=tmp_path / "sp.nwk",
            families_file=tmp_path / "families.txt",
            out_dir=tmp_path / "out",
            device="cpu",
            **{field: -1.0},
        )


def test_family_chunk_size_normalization_is_shared():
    for value in (None, "", "0", "all", "none", "null", 0):
        assert normalize_family_chunk_size(value) == 0
    assert normalize_family_chunk_size("12") == 12
    assert normalize_family_chunk_size("auto", allow_auto=True) == "auto"
    with pytest.raises(ValueError, match="auto"):
        normalize_family_chunk_size("auto")


def test_run_config_requires_budget_for_nonsequential_packing(tmp_path: Path):
    with pytest.raises(ValueError, match="requires clade_budget"):
        RunConfig(
            species_tree=tmp_path / "sp.nwk",
            families_file=tmp_path / "families.txt",
            out_dir=tmp_path / "out",
            batch_packing="depth_first_fit",
            clade_budget=None,
            device="cpu",
        )


@pytest.mark.parametrize(
    "rates",
    [
        {"theta_init_d": 0.0},
        {"theta_init_l": -0.1},
        {"theta_init_t": 0.0},
    ],
)
def test_run_config_rejects_nonpositive_theta_init_rates(
    tmp_path: Path,
    rates: dict[str, float],
):
    with pytest.raises(ValueError, match="theta_init"):
        RunConfig(
            species_tree=tmp_path / "sp.nwk",
            families_file=tmp_path / "families.txt",
            out_dir=tmp_path / "out",
            device="cpu",
            **rates,
        )


def test_gene_recon_constructors_reject_bad_theta_init_before_io(tmp_path: Path):
    with pytest.raises(ValueError, match="theta_init_rates"):
        GeneReconModel.from_trees(
            tmp_path / "missing_species.nwk",
            [tmp_path / "missing_gene.nwk"],
            device="cpu",
            theta_init_rates=(0.0, 0.1, 0.1),
        )
    with pytest.raises(ValueError, match="theta_init_rates"):
        GeneReconModel.from_alerax_families(
            tmp_path / "missing_species.nwk",
            tmp_path / "missing_families.txt",
            device="cpu",
            theta_init_rates=(0.1, -0.1, 0.1),
        )


def test_gene_recon_constructors_reject_cpu_device_before_io(tmp_path: Path):
    with pytest.raises(ValueError, match="requires a CUDA device"):
        GeneReconModel.from_trees(
            tmp_path / "missing_species.nwk",
            [tmp_path / "missing_gene.nwk"],
            device="cpu",
        )
    with pytest.raises(ValueError, match="requires a CUDA device"):
        GeneReconModel.from_alerax_families(
            tmp_path / "missing_species.nwk",
            tmp_path / "missing_families.txt",
            device="cpu",
        )


def test_uniform_chunked_alerax_constructor_validates_mode_before_io(tmp_path: Path):
    with pytest.raises(ValueError, match="from_alerax_families"):
        UniformChunkedReconModel.from_alerax_families(
            tmp_path / "missing_species.nwk",
            tmp_path / "missing_families.txt",
            mode="genewise",
        )


def test_uniform_chunked_constructors_reject_bad_theta_init_before_io(tmp_path: Path):
    with pytest.raises(ValueError, match="theta_init_rates"):
        UniformChunkedReconModel.from_trees(
            tmp_path / "missing_species.nwk",
            [tmp_path / "missing_gene.nwk"],
            device="cpu",
            theta_init_rates=(0.0, 0.1, 0.1),
        )
    with pytest.raises(ValueError, match="theta_init_rates"):
        UniformChunkedReconModel.from_alerax_families(
            tmp_path / "missing_species.nwk",
            tmp_path / "missing_families.txt",
            device="cpu",
            theta_init_rates=(0.1, -0.1, 0.1),
        )


def test_uniform_chunked_constructors_reject_unavailable_cuda_before_io(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA was requested"):
        UniformChunkedReconModel.from_trees(
            tmp_path / "missing_species.nwk",
            [tmp_path / "missing_gene.nwk"],
            device="cuda",
        )
    with pytest.raises(RuntimeError, match="CUDA was requested"):
        UniformChunkedReconModel.from_alerax_families(
            tmp_path / "missing_species.nwk",
            tmp_path / "missing_families.txt",
            device="cuda",
        )


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

    monkeypatch.setattr(workflow_model_factory.torch.cuda, "is_available", lambda: True)
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


def _minimal_workflow_cli_args(command: str, tmp_path: Path) -> list[str]:
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


def test_cli_reports_missing_json_config_without_traceback(tmp_path: Path, capsys):
    path = tmp_path / "missing.json"

    with pytest.raises(SystemExit) as exc_info:
        main(["optimize", "--config", str(path)])

    captured = capsys.readouterr()
    assert exc_info.value.code == 2
    assert "could not read config" in captured.err
    assert str(path) in captured.err
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
    assert exc_info.value.code == 2
    assert "workflow failed" in captured.err
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
    assert exc_info.value.code == 2
    assert "workflow failed" in captured.err
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
    assert "GPUREC_BACKTRACK_BIN" in captured.out
    assert "Samples per selected family" in captured.out


def test_workflow_rate_outputs_use_normalized_survival_probability(tmp_path: Path):
    theta = torch.log2(torch.tensor([[2.0, 3.0, 5.0]], dtype=torch.float64))
    expected_ps = 1.0 / (1.0 + 2.0 + 3.0 + 5.0)

    stats = parameter_stats(theta)
    assert stats["pS/mean"] == pytest.approx(expected_ps)
    assert stats["pS/min"] > 0.0

    model = SimpleNamespace(theta=torch.nn.Parameter(theta.clone()))
    path = tmp_path / "rates.tsv"
    _write_rate_table(path, model, "global")

    header, row = path.read_text(encoding="utf-8").strip().splitlines()
    columns = header.split("\t")
    values = dict(zip(columns, row.split("\t")))

    assert float(values["D"]) == pytest.approx(2.0)
    assert float(values["L"]) == pytest.approx(3.0)
    assert float(values["T"]) == pytest.approx(5.0)
    assert float(values["pS"]) == pytest.approx(expected_ps)


def test_sampling_config_validates_selection(tmp_path: Path):
    config = SamplingConfig(
        checkpoint=tmp_path / "checkpoints" / "best.pt",
        out_dir=tmp_path / "out",
        samples=2,
        family_start=1,
        max_families=3,
    )

    assert config.samples == 2
    assert config.family_start == 1
    assert config.max_families == 3


def test_sampling_config_rejects_invalid_seed_and_event_limits(tmp_path: Path):
    checkpoint = tmp_path / "checkpoints" / "best.pt"
    with pytest.raises(ValueError, match="seed"):
        SamplingConfig(checkpoint=checkpoint, seed=-1)
    with pytest.raises(ValueError, match="max_events"):
        SamplingConfig(checkpoint=checkpoint, max_events=0)


def test_public_backtracking_rejects_invalid_seed_and_event_limits():
    model = object()
    with pytest.raises(ValueError, match="seed"):
        export_backtracking_input(model, seed=-1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="max_events"):
        sample_recphyloxml(model, max_events=0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="seed"):
        sample_recphyloxmls(model, num_samples=1, seed=-1)  # type: ignore[arg-type]


def test_backtracking_sampler_helpers_share_subprocess_io(tmp_path: Path, monkeypatch):
    fake_backtracker = tmp_path / "fake_backtracker.py"
    fake_backtracker.write_text(
        """import json
import pathlib
import sys

args = sys.argv[1:]
if args[0] == "--samples":
    num_samples = int(args[1])
    seed = args[3]
    output_dir = pathlib.Path(args[5])
    input_path = pathlib.Path(args[6])
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    output_dir.mkdir()
    for idx in range(num_samples):
        (output_dir / f"sample_{idx}.xml").write_text(
            f"<sample family='{payload['family_index']}' seed='{seed}' index='{idx}'/>",
            encoding="utf-8",
        )
else:
    input_path = pathlib.Path(args[0])
    output_path = pathlib.Path(args[1])
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    output_path.write_text(
        f"<single family='{payload['family_index']}' "
        f"seed='{payload['seed']}' max='{payload['max_events']}'/>",
        encoding="utf-8",
    )
""",
        encoding="utf-8",
    )

    exported: list[dict[str, int | None]] = []

    def fake_export(
        model: object,
        *,
        family_index: int = 0,
        seed: int | None = None,
        max_events: int | None = None,
    ) -> dict[str, int | None]:
        payload = {
            "family_index": family_index,
            "seed": seed,
            "max_events": max_events,
        }
        exported.append(payload)
        return payload

    monkeypatch.setattr(backtracking, "export_backtracking_input", fake_export)
    monkeypatch.setattr(
        backtracking,
        "_backtrack_command",
        lambda **_: [sys.executable, str(fake_backtracker)],
    )

    single_xml = sample_recphyloxml(
        object(),
        family_index=3,
        seed=7,
        max_events=9,
        backtrack_binary=fake_backtracker,
    )
    batch_xmls = sample_recphyloxmls(
        object(),
        family_index=4,
        num_samples=2,
        seed=11,
        max_events=13,
        backtrack_binary=fake_backtracker,
    )

    assert single_xml == "<single family='3' seed='7' max='9'/>"
    assert batch_xmls == [
        "<sample family='4' seed='11' index='0'/>",
        "<sample family='4' seed='11' index='1'/>",
    ]
    assert exported == [
        {"family_index": 3, "seed": 7, "max_events": 9},
        {"family_index": 4, "seed": 11, "max_events": 13},
    ]


def test_recphyloxml_event_counts_uses_shared_event_schema():
    xml = """
    <recPhylo>
      <recGeneTree>
        <phylogeny>
          <clade>
            <eventsRec><speciation speciesLocation="A"/></eventsRec>
            <clade><eventsRec><loss speciesLocation="B"/></eventsRec></clade>
            <clade>
              <eventsRec><duplication speciesLocation="A"/></eventsRec>
              <clade><eventsRec><loss speciesLocation="A"/></eventsRec></clade>
              <clade>
                <eventsRec><branchingOut speciesLocation="A"/></eventsRec>
                <clade><eventsRec><loss speciesLocation="C"/></eventsRec></clade>
                <clade><eventsRec><leaf speciesLocation="D"/></eventsRec></clade>
              </clade>
            </clade>
          </clade>
        </phylogeny>
      </recGeneTree>
    </recPhylo>
    """

    counts = recphyloxml_event_counts(xml)
    raw_counts = recphyloxml_event_counts(xml, alerax_style=False)

    assert tuple(counts) == EVENT_KEYS
    assert counts == {
        "S": 0,
        "SL": 1,
        "D": 0,
        "DL": 1,
        "T": 0,
        "TL": 1,
        "L": 0,
        "Leaf": 1,
    }
    assert raw_counts == {
        "S": 1,
        "SL": 0,
        "D": 1,
        "DL": 0,
        "T": 1,
        "TL": 0,
        "L": 3,
        "Leaf": 1,
    }


def test_sampling_runner_writes_outputs_and_aggregates(tmp_path: Path, monkeypatch):
    xml = """
    <recPhylo>
      <recGeneTree>
        <phylogeny>
          <clade>
            <eventsRec><speciation speciesLocation="A"/></eventsRec>
            <clade>
              <eventsRec>
                <branchingOut speciesLocation="A"/>
                <transferBack destinationSpecies="B"/>
              </eventsRec>
              <clade><eventsRec><leaf speciesLocation="B"/></eventsRec></clade>
            </clade>
            <clade><eventsRec><loss speciesLocation="C"/></eventsRec></clade>
          </clade>
        </phylogeny>
      </recGeneTree>
    </recPhylo>
    """

    class FakeModel:
        family_names = ["fam0", "../fam/a", "fam2"]

        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    model = FakeModel()
    run_config = RunConfig(
        species_tree=tmp_path / "sp.nwk",
        families_file=tmp_path / "families.txt",
        out_dir=tmp_path / "run_out",
        device="cuda",
    )
    config = SamplingConfig(
        checkpoint=tmp_path / "checkpoints" / "best.pt",
        out_dir=tmp_path / "sample_out",
        samples=2,
        seed=5,
        family_start=1,
        max_families=2,
        max_events=123,
        backtrack_binary=tmp_path / "fake-backtrack",
    )
    calls: list[dict[str, object]] = []

    def fake_sample_recphyloxmls(
        model_arg: object,
        *,
        family_index: int,
        num_samples: int,
        seed: int,
        max_events: int | None,
        backtrack_binary: Path | None,
    ) -> list[str]:
        assert model_arg is model
        calls.append(
            {
                "family_index": family_index,
                "num_samples": num_samples,
                "seed": seed,
                "max_events": max_events,
                "backtrack_binary": backtrack_binary,
            }
        )
        return [xml for _ in range(num_samples)]

    runner = SamplingRunner(config)
    monkeypatch.setattr(runner, "_load_model", lambda: (run_config, model))
    monkeypatch.setattr(
        sampling_workflow,
        "sample_recphyloxmls",
        fake_sample_recphyloxmls,
    )

    result = runner.run()

    assert result.out_dir == config.out_dir
    assert result.families_sampled == 2
    assert result.samples_per_family == 2
    assert result.xml_files == 4
    assert model.closed
    assert calls == [
        {
            "family_index": 1,
            "num_samples": 2,
            "seed": 7,
            "max_events": 123,
            "backtrack_binary": config.backtrack_binary,
        },
        {
            "family_index": 2,
            "num_samples": 2,
            "seed": 9,
            "max_events": 123,
            "backtrack_binary": config.backtrack_binary,
        },
    ]

    recon_dir = config.out_dir / "reconciliations"
    all_dir = recon_dir / "all"
    assert sorted(path.name for path in all_dir.glob("*.xml")) == [
        "000001_fam_a_sample_0.xml",
        "000001_fam_a_sample_1.xml",
        "000002_fam2_sample_0.xml",
        "000002_fam2_sample_1.xml",
    ]
    for path in all_dir.iterdir():
        assert path.resolve().is_relative_to(all_dir.resolve())
    assert (
        all_dir / "000001_fam_a_eventCounts_0.txt"
    ).read_text(encoding="utf-8") == "S:0\nSL:1\nD:0\nDL:0\nT:1\nTL:0\nL:0\nLeaf:1\n"

    event_rows = (recon_dir / "event_counts.tsv").read_text(encoding="utf-8").splitlines()
    assert event_rows[0] == "family\tsample\t" + "\t".join(EVENT_KEYS)
    assert event_rows[1:] == [
        "../fam/a\t0\t0\t1\t0\t0\t1\t0\t0\t1",
        "../fam/a\t1\t0\t1\t0\t0\t1\t0\t0\t1",
        "fam2\t0\t0\t1\t0\t0\t1\t0\t0\t1",
        "fam2\t1\t0\t1\t0\t0\t1\t0\t0\t1",
    ]

    species_lines = (
        recon_dir / "totalSpeciesEventCounts.txt"
    ).read_text(encoding="utf-8").splitlines()
    species_header = [column.strip() for column in species_lines[0].split(",")]
    species_rows = {}
    for line in species_lines[1:]:
        parts = [part.strip() for part in line.split(",")]
        species_rows[parts[0]] = dict(zip(species_header[1:], map(float, parts[1:])))

    assert species_rows["A"]["speciations"] == pytest.approx(2.0)
    assert species_rows["A"]["transfers"] == pytest.approx(2.0)
    assert species_rows["A"]["origination"] == pytest.approx(2.0)
    assert species_rows["B"]["transfers_to"] == pytest.approx(2.0)
    assert species_rows["B"]["copies"] == pytest.approx(2.0)
    assert species_rows["B"]["singletons"] == pytest.approx(2.0)
    assert species_rows["C"]["losses"] == pytest.approx(2.0)

    assert (recon_dir / "totalTransfers.txt").read_text(encoding="utf-8") == "A B 2\n"
    assert json.loads((recon_dir / "summary.json").read_text(encoding="utf-8")) == {
        "checkpoint": str(config.checkpoint),
        "families_sampled": 2,
        "out_dir": str(config.out_dir),
        "samples_per_family": 2,
        "xml_files": 4,
    }


class _DummyModel:
    def __init__(self):
        self.theta = torch.nn.Parameter(torch.zeros(2, 3))
        self.family_names = ["a", "b"]
        self.cleared = False

    def clear(self):
        self.cleared = True


def test_checkpoint_roundtrip_restores_theta_and_status(tmp_path: Path):
    config = RunConfig(
        species_tree=tmp_path / "sp.nwk",
        families_file=tmp_path / "families.txt",
        out_dir=tmp_path / "out",
        device="cpu",
    )
    model = _DummyModel()
    with torch.no_grad():
        model.theta.fill_(2.0)

    path = tmp_path / "latest.pt"
    save_checkpoint(
        path,
        config=config,
        model=model,
        optimizer=None,
        step=4,
        status={"status": "running", "best_nll_bits": 12.0},
    )
    payload = load_checkpoint(path)

    with torch.no_grad():
        model.theta.zero_()
    restore_model_theta(model, payload)

    assert int(payload["step"]) == 4
    assert payload["status"]["best_nll_bits"] == 12.0
    assert torch.equal(model.theta, torch.full_like(model.theta, 2.0))
    assert model.cleared


def test_optimization_runner_reports_discarded_resume_optimizer_state(tmp_path: Path):
    config = RunConfig(
        species_tree=tmp_path / "sp.nwk",
        families_file=tmp_path / "families.txt",
        out_dir=tmp_path / "out",
        device="cpu",
    )
    runner = OptimizationRunner(config)

    class FakeOptimizer:
        def __init__(self, *, fail: bool = False):
            self.fail = fail
            self.loaded = None

        def load_state_dict(self, state):
            if self.fail:
                raise ValueError("incompatible optimizer state")
            self.loaded = state

    missing = runner._restore_optimizer_state(FakeOptimizer(), None)
    assert missing == {"resume_optimizer_state": "missing"}

    restored_optimizer = FakeOptimizer()
    restored = runner._restore_optimizer_state(restored_optimizer, {"state": []})
    assert restored == {"resume_optimizer_state": "restored"}
    assert restored_optimizer.loaded == {"state": []}

    discarded = runner._restore_optimizer_state(
        FakeOptimizer(fail=True),
        {"state": ["bad"]},
    )
    assert discarded["resume_optimizer_state"] == "discarded"
    assert "incompatible optimizer state" in discarded["resume_optimizer_error"]


def test_activate_family_batch_returns_batch_local_offset():
    model = SimpleNamespace()
    model._batched_resident = True
    model._dataset = SimpleNamespace(
        families=[{"C": 2}, {"C": 3}, {"C": 5}, {"C": 7}]
    )
    model.n_families = 4
    model.batch_metadata = [
        SimpleNamespace(family_indices=[0]),
        SimpleNamespace(family_indices=[1, 2, 3]),
    ]
    model._current_batch_index = 0
    ensured: list[int] = []

    def select_batch(idx: int):
        ensured.append(idx)
        model._current_batch_index = idx
        return model.batch_metadata[idx]

    model.select_batch = select_batch
    model.activate_family = lambda family_index: GeneReconModel.activate_family(
        model,
        family_index,
    )

    assert _activate_family_batch(model, 1) == (0, 0)
    active = model.activate_family(3)
    assert (active.clade_offset, active.local_family_index) == (8, 2)
    assert _activate_family_batch(model, 3) == (8, 2)
    assert model._current_batch_index == 1
    assert ensured == [1, 1, 1]


def test_backtracking_command_reports_missing_source_manifest(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("GPUREC_BACKTRACK_BIN", raising=False)

    with pytest.raises(RuntimeError, match="GPUREC_BACKTRACK_BIN"):
        _backtrack_command(
            cargo_manifest=tmp_path / "missing" / "Cargo.toml",
            backtrack_binary=None,
        )


def test_workflow_and_backtracking_use_public_model_surface():
    root = Path(__file__).resolve().parents[2]
    paths = [root / "gpurec" / "backtracking.py"]
    paths.extend(sorted((root / "gpurec" / "workflow").glob("*.py")))
    forbidden = (
        "model._dataset",
        "model._active_static",
        "model._active_theta",
        "model._current_batch_index",
        "model._ensure_batch_static",
        "model._batch_statics",
        "model._static",
        'getattr(model, "_',
        "from gpurec.api.autograd import _extract_parameters",
        "Pi_wave_forward",
        "E_fixed_point",
    )

    offenders: list[str] = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            if token in text:
                offenders.append(f"{path.relative_to(root)} contains {token}")

    assert offenders == []


def test_hogenom_scripts_use_public_model_surface():
    root = Path(__file__).resolve().parents[2]
    paths = [
        root / "scripts" / "export_hogenom_rates_from_checkpoint.py",
        root / "scripts" / "fast_optimize_hogenom_ccp.py",
        root / "scripts" / "hogenom_ccp_wandb_opt.py",
        root / "scripts" / "hogenom_opt_helpers.py",
        root / "scripts" / "make_hogenom_branchscale_penalty_report.py",
        root / "scripts" / "optimize_hogenom_ccp_global_uniform.py",
        root / "scripts" / "optimize_hogenom_ccp_hydra.py",
        root / "scripts" / "optimize_hogenom_ccp_specieswise_uniform.py",
        root / "scripts" / "optimize_hogenom_ccp_wandb.py",
        root / "scripts" / "optimize_hogenom_penalty316_kkt.py",
        root / "scripts" / "profile_hogenom_ccp_pass.py",
    ]
    forbidden = (
        "model._",
        'getattr(model, "_',
        "from gpurec.api.model import _",
        "from gpurec.api.model import _GeneReconFullLossFunction",
        "_stream_full_batches",
        "_ensure_batch_static",
        "_current_batch_index",
        "_schedule_prefetch",
        "_theta_for_batch_index",
        "_evaluate_static_state",
        "from gpurec.core.preprocess_cpp import _load_extension",
    )

    offenders: list[str] = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            if token in text:
                offenders.append(f"{path.relative_to(root)} contains {token}")

    assert offenders == []


def test_xml_species_and_transfer_counts():
    xml = """
    <recPhylo>
      <recGeneTree>
        <phylogeny rooted="true">
          <clade>
            <eventsRec><speciation speciesLocation="Root"/></eventsRec>
            <clade>
              <eventsRec><branchingOut speciesLocation="Donor"/></eventsRec>
              <clade>
                <eventsRec>
                  <transferBack destinationSpecies="Recipient"/>
                  <leaf speciesLocation="Recipient"/>
                </eventsRec>
              </clade>
            </clade>
            <clade><eventsRec><loss speciesLocation="Lost"/></eventsRec></clade>
          </clade>
        </phylogeny>
      </recGeneTree>
    </recPhylo>
    """

    species, transfers = _xml_species_and_transfer_counts(xml)

    assert species["Root"]["speciations"] == 1.0
    assert species["Root"]["origination"] == 1.0
    assert species["Donor"]["transfers"] == 1.0
    assert species["Recipient"]["transfers_to"] == 1.0
    assert species["Recipient"]["copies"] == 1.0
    assert species["Lost"]["losses"] == 1.0
    assert transfers[("Donor", "Recipient")] == 1.0
