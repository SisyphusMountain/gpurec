from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from gpurec.backtracking import _activate_family_batch, _backtrack_command
from gpurec.cli import _run_config_from_args, build_parser, main
from gpurec.api.model import GeneReconModel
from gpurec.workflow.checkpoint import load_checkpoint, restore_model_theta, save_checkpoint
from gpurec.workflow.config import RunConfig, SamplingConfig
from gpurec.workflow.diagnostics import parameter_stats
from gpurec.workflow.optimize import _write_rate_table
from gpurec.workflow.sampling import _xml_species_and_transfer_counts


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


def test_run_config_rejects_unsupported_auto_chunking(tmp_path: Path):
    with pytest.raises(ValueError, match="auto"):
        RunConfig(
            species_tree=tmp_path / "sp.nwk",
            families_file=tmp_path / "families.txt",
            out_dir=tmp_path / "out",
            family_chunk_size="auto",
            device="cpu",
        )


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


def test_cli_rejects_hydra_yaml_config_without_traceback(tmp_path: Path, capsys):
    path = tmp_path / "config.yaml"
    path.write_text("paths:\n  species_tree: sp.nwk\n", encoding="utf-8")

    with pytest.raises(SystemExit) as exc_info:
        main(["optimize", "--config", str(path)])

    captured = capsys.readouterr()
    assert exc_info.value.code == 2
    assert "flat JSON RunConfig" in captured.err


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
