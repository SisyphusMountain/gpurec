from __future__ import annotations

from pathlib import Path

import pytest
import torch

from scripts import export_hogenom_rates_from_checkpoint as export_rates
from scripts import hogenom_ccp_wandb_opt as hogenom_opt
from scripts.compare_backtracking_alerax_events import load_rates
from scripts.export_hogenom_rates_from_checkpoint import (
    parse_newick,
    species_order_labels,
)


def _rate_output_dir(tmp_path: Path) -> Path:
    output_dir = tmp_path / "output"
    (output_dir / "model_parameters").mkdir(parents=True)
    return output_dir


def test_compare_backtracking_load_rates_reads_global_model_parameters(
    tmp_path: Path,
):
    output_dir = _rate_output_dir(tmp_path)
    (output_dir / "model_parameters" / "model_parameters.txt").write_text(
        "node D L T\n16 0.1 0.2 0.3\n",
        encoding="utf-8",
    )

    assert load_rates(output_dir) == (0.1, 0.2, 0.3)


def test_compare_backtracking_load_rates_prefers_family_specific_rates(
    tmp_path: Path,
):
    output_dir = _rate_output_dir(tmp_path)
    (output_dir / "model_parameters" / "model_parameters.txt").write_text(
        "node D L T\n16 0.1 0.2 0.3\n",
        encoding="utf-8",
    )
    (output_dir / "model_parameters" / "family_0001_rates.txt").write_text(
        "D L T\n0.4 0.5 0.6\n",
        encoding="utf-8",
    )

    assert load_rates(output_dir, "family_0001") == (0.4, 0.5, 0.6)


def test_compare_backtracking_load_rates_reports_missing_global_file(
    tmp_path: Path,
):
    output_dir = _rate_output_dir(tmp_path)

    with pytest.raises(FileNotFoundError, match="model_parameters.txt"):
        load_rates(output_dir)


@pytest.mark.parametrize(
    ("filename", "text", "family_name", "message"),
    [
        (
            "model_parameters.txt",
            "node D L T\n",
            None,
            "header row and at least one rate row",
        ),
        (
            "model_parameters.txt",
            "node D L T\n16 0.1 0.2\n",
            None,
            "node D L T",
        ),
        (
            "model_parameters.txt",
            "node D L T\n16 bad 0.2 0.3\n",
            None,
            "could not parse D/L/T rates",
        ),
        (
            "family_0001_rates.txt",
            "D L T\n0.1 0.2\n",
            "family_0001",
            "D L T",
        ),
        (
            "family_0001_rates.txt",
            "D L T\n0.1 bad 0.3\n",
            "family_0001",
            "could not parse D/L/T rates",
        ),
    ],
)
def test_compare_backtracking_load_rates_reports_malformed_files(
    tmp_path: Path,
    filename: str,
    text: str,
    family_name: str | None,
    message: str,
):
    output_dir = _rate_output_dir(tmp_path)
    path = output_dir / "model_parameters" / filename
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match=message) as exc_info:
        load_rates(output_dir, family_name)

    assert str(path) in str(exc_info.value)


def test_export_rates_parse_newick_rejects_empty_file(tmp_path: Path):
    tree_path = tmp_path / "empty.nwk"
    tree_path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="empty Newick file") as exc_info:
        parse_newick(tree_path)

    assert str(tree_path) in str(exc_info.value)


def test_export_rates_parse_newick_keeps_valid_single_leaf(tmp_path: Path):
    tree_path = tmp_path / "one.nwk"
    tree_path.write_text("SpeciesA;\n", encoding="utf-8")

    assert species_order_labels(parse_newick(tree_path)) == ["SpeciesA"]


def test_export_rates_checkpoint_loader_uses_weights_only(
    tmp_path: Path,
    monkeypatch,
):
    calls: list[tuple[Path, str, bool]] = []

    def fake_load(path, *, map_location, weights_only):
        calls.append((path, map_location, weights_only))
        return {"theta": torch.zeros(1, 3)}

    monkeypatch.setattr(export_rates.torch, "load", fake_load)

    checkpoint = export_rates.load_checkpoint(tmp_path / "checkpoint.pt")

    assert torch.equal(checkpoint["theta"], torch.zeros(1, 3))
    assert calls == [(tmp_path / "checkpoint.pt", "cpu", True)]


def test_export_rates_checkpoint_loader_rejects_non_dict_payload(
    tmp_path: Path,
    monkeypatch,
):
    def fake_load(path, *, map_location, weights_only):
        return []

    monkeypatch.setattr(export_rates.torch, "load", fake_load)

    with pytest.raises(RuntimeError, match="dictionary payload"):
        export_rates.load_checkpoint(tmp_path / "checkpoint.pt")


def test_hogenom_wandb_checkpoint_loader_uses_weights_only(
    tmp_path: Path,
    monkeypatch,
):
    calls: list[tuple[Path, torch.device, bool]] = []
    device = torch.device("cpu")

    def fake_load(path, *, map_location, weights_only):
        calls.append((path, map_location, weights_only))
        raise RuntimeError("blocked unsafe legacy checkpoint")

    monkeypatch.setattr(hogenom_opt.torch, "load", fake_load)

    with pytest.raises(RuntimeError, match="could not safely load checkpoint"):
        hogenom_opt.load_checkpoint(
            tmp_path / "checkpoint.pt",
            model=None,  # type: ignore[arg-type]
            branch_params=None,
            optimizers={},
            config=None,  # type: ignore[arg-type]
            device=device,
        )

    assert calls == [(tmp_path / "checkpoint.pt", device, True)]
