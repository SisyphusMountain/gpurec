from __future__ import annotations

from pathlib import Path

import pytest
import torch

from profiling import evaluate_hogenom_alerax_rates as evaluate_rates
from scripts import compare_backtracking_alerax_events as compare_events
from scripts import export_hogenom_rates_from_checkpoint as export_rates
from scripts import hogenom_ccp_wandb_opt as hogenom_opt
from scripts import visualize_hogenom_loss_landscape as landscape
from scripts.compare_backtracking_alerax_events import load_rates
from scripts.export_hogenom_rates_from_checkpoint import (
    parse_newick,
    species_order_labels,
)


def _rate_output_dir(tmp_path: Path) -> Path:
    output_dir = tmp_path / "output"
    (output_dir / "model_parameters").mkdir(parents=True)
    return output_dir


class _FakeNllTensor:
    def __init__(self, values: list[float]) -> None:
        self._values = values

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self) -> list[float]:
        return self._values


class _FakeNllModel:
    def __init__(
        self,
        values: list[float] | None = None,
        error: Exception | None = None,
    ) -> None:
        self._values = values if values is not None else [1.0]
        self._error = error
        self.close_calls = 0

    def nll_per_family(self):
        if self._error is not None:
            raise self._error
        return _FakeNllTensor(self._values)

    def close(self) -> None:
        self.close_calls += 1


class _FakeBacktrackingModel:
    def __init__(self) -> None:
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


def _write_compare_family_inputs(tmp_path: Path) -> tuple[Path, Path]:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "sp.nwk").write_text("(A,B);\n", encoding="utf-8")
    (dataset / "g_0001.nwk").write_text("(a,b);\n", encoding="utf-8")
    output_dir = _rate_output_dir(tmp_path)
    (output_dir / "model_parameters" / "model_parameters.txt").write_text(
        "node D L T\n16 0.1 0.2 0.3\n",
        encoding="utf-8",
    )
    alerax_dir = output_dir / "reconciliations" / "all"
    alerax_dir.mkdir(parents=True)
    (alerax_dir / "family_0001_eventCounts_0.txt").write_text(
        "S: 1\n",
        encoding="utf-8",
    )
    return dataset, output_dir


def test_evaluate_rates_closes_model_after_successful_family_nll():
    model = _FakeNllModel(values=[1.25, 2.5])

    assert evaluate_rates._nll_per_family_with_cleanup(model) == [1.25, 2.5]
    assert model.close_calls == 1


def test_evaluate_rates_closes_model_after_family_nll_failure():
    model = _FakeNllModel(error=RuntimeError("nll failed"))

    with pytest.raises(RuntimeError, match="nll failed"):
        evaluate_rates._nll_per_family_with_cleanup(model)

    assert model.close_calls == 1


def test_landscape_representative_family_indices_cover_leaf_count_quantiles():
    leaf_counts = [50, 10, 30, 20, 40]

    assert landscape.representative_family_indices(
        leaf_counts,
        target_count=3,
    ) == [1, 2, 0]


def test_landscape_resolve_family_selection_by_name_and_leaf_count():
    selections = landscape.resolve_family_selection(
        names=["a", "b", "c"],
        tree_paths=[["a.trees"], ["b.trees"], ["c.trees"]],
        leaf_maps=[
            {"a1": "A", "a2": "B"},
            {"b1": "A"},
            {"c1": "A", "c2": "B", "c3": "C"},
        ],
        family_indices=None,
        family_names=["c", "a"],
        representative_count=2,
    )

    assert [(item.index, item.name, item.leaf_count) for item in selections] == [
        (2, "c", 3),
        (0, "a", 2),
    ]


def test_landscape_resolve_family_selection_rejects_mixed_selectors():
    with pytest.raises(ValueError, match="either family indices or family names"):
        landscape.resolve_family_selection(
            names=["a"],
            tree_paths=[["a.trees"]],
            leaf_maps=[{}],
            family_indices=[0],
            family_names=["a"],
            representative_count=1,
        )


def test_landscape_grid_points_must_be_odd():
    assert landscape.validate_grid_points(7) == 7
    with pytest.raises(Exception, match="odd integer"):
        landscape.validate_grid_points(8)


def test_landscape_load_anchor_theta_accepts_tensor_and_checkpoint(tmp_path: Path):
    tensor_path = tmp_path / "theta.pt"
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(torch.tensor([1.0, 2.0, 3.0]), tensor_path)
    torch.save({"theta": torch.arange(6, dtype=torch.float32).reshape(2, 3)}, checkpoint_path)

    torch.testing.assert_close(
        landscape.load_anchor_theta(tensor_path),
        torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64),
    )
    torch.testing.assert_close(
        landscape.load_anchor_theta(checkpoint_path),
        torch.arange(6, dtype=torch.float64).reshape(2, 3),
    )


def test_landscape_anchor_for_family_validates_row_count():
    anchor = torch.arange(6, dtype=torch.float64).reshape(2, 3)

    torch.testing.assert_close(
        landscape.anchor_for_family(anchor, family_index=1, family_count=2),
        torch.tensor([3.0, 4.0, 5.0], dtype=torch.float64),
    )
    with pytest.raises(ValueError, match="expected 1 or 3 family rows"):
        landscape.anchor_for_family(anchor, family_index=1, family_count=3)


def test_landscape_finite_difference_hessian_matches_quadratic():
    matrix = torch.tensor(
        [
            [2.0, 0.5, -0.25],
            [0.5, 4.0, 0.75],
            [-0.25, 0.75, 3.0],
        ],
        dtype=torch.float64,
    )

    def evaluate(theta: torch.Tensor) -> float:
        return float(0.5 * theta @ matrix @ theta)

    actual = landscape.finite_difference_hessian(
        evaluate,
        torch.tensor([0.7, -1.2, 0.25], dtype=torch.float64),
        step_log2=1e-3,
    )

    torch.testing.assert_close(actual, matrix, rtol=1e-5, atol=1e-5)


def test_landscape_evaluate_thetas_in_blocks_concatenates_block_outputs():
    seen_shapes: list[tuple[int, int]] = []

    def evaluate_block(block: torch.Tensor) -> torch.Tensor:
        seen_shapes.append(tuple(block.shape))
        return block.sum(dim=1)

    rows = torch.arange(15, dtype=torch.float64).reshape(5, 3)
    actual = landscape.evaluate_thetas_in_blocks(
        rows,
        batch_size=2,
        evaluate_block=evaluate_block,
    )

    assert seen_shapes == [(2, 3), (2, 3), (1, 3)]
    torch.testing.assert_close(actual, rows.sum(dim=1))


def test_landscape_evaluate_thetas_in_blocks_validates_result_count():
    with pytest.raises(RuntimeError, match="returned 1 value"):
        landscape.evaluate_thetas_in_blocks(
            torch.zeros((2, 3), dtype=torch.float64),
            batch_size=2,
            evaluate_block=lambda _block: torch.zeros(1),
        )


def test_landscape_batched_hessian_matches_quadratic():
    matrix = torch.tensor(
        [
            [3.0, -0.25, 0.5],
            [-0.25, 1.5, 0.75],
            [0.5, 0.75, 2.0],
        ],
        dtype=torch.float64,
    )

    def evaluate_many(theta_rows: torch.Tensor) -> torch.Tensor:
        return torch.tensor(
            [float(0.5 * theta @ matrix @ theta) for theta in theta_rows],
            dtype=torch.float64,
        )

    actual = landscape.finite_difference_hessian_batched(
        evaluate_many,
        torch.tensor([-0.2, 0.4, 1.1], dtype=torch.float64),
        step_log2=1e-3,
    )

    torch.testing.assert_close(actual, matrix, rtol=1e-5, atol=1e-5)


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


def test_compare_backtracking_closes_model_after_successful_sampling(
    tmp_path: Path,
    monkeypatch,
):
    dataset, output_dir = _write_compare_family_inputs(tmp_path)
    model = _FakeBacktrackingModel()

    class FakeFactory:
        @staticmethod
        def from_trees(*args, **kwargs):
            assert kwargs["theta_init_rates"] == (0.1, 0.2, 0.3)
            return model

    monkeypatch.setattr(compare_events, "GeneReconModel", FakeFactory)
    monkeypatch.setattr(
        compare_events,
        "sample_recphyloxmls",
        lambda *args, **kwargs: ["<xml />"],
    )
    monkeypatch.setattr(
        compare_events,
        "recphyloxml_event_counts",
        lambda xml: {"S": 2},
    )

    rows = compare_events.compare_family(
        dataset=dataset,
        output_dir=output_dir,
        family_index=1,
        samples=3,
        seed=5,
        fixed_iters_pi=6,
        max_iters_e=10,
        tol_e=1e-6,
        backtrack_binary=None,
        families_file=None,
        species_tree=None,
    )

    assert model.close_calls == 1
    assert ("family_0001", "S", 1, 1, 1, 2, 2, 2, 1) in rows


def test_compare_backtracking_closes_model_after_sampling_failure(
    tmp_path: Path,
    monkeypatch,
):
    dataset, output_dir = _write_compare_family_inputs(tmp_path)
    model = _FakeBacktrackingModel()

    class FakeFactory:
        @staticmethod
        def from_trees(*args, **kwargs):
            return model

    def fail_sampling(*args, **kwargs):
        raise RuntimeError("sampling failed")

    monkeypatch.setattr(compare_events, "GeneReconModel", FakeFactory)
    monkeypatch.setattr(compare_events, "sample_recphyloxmls", fail_sampling)

    with pytest.raises(RuntimeError, match="sampling failed"):
        compare_events.compare_family(
            dataset=dataset,
            output_dir=output_dir,
            family_index=1,
            samples=3,
            seed=5,
            fixed_iters_pi=6,
            max_iters_e=10,
            tol_e=1e-6,
                backtrack_binary=None,
                families_file=None,
                species_tree=None,
            )

    assert model.close_calls == 1


@pytest.mark.parametrize(
    ("parser_factory", "args", "message"),
    [
        (
            evaluate_rates.build_parser,
            ["--chunk-size", "0"],
            "chunk-size must be positive",
        ),
        (
            evaluate_rates.build_parser,
            ["--max-families", "-1"],
            "max-families must be positive",
        ),
        (
            evaluate_rates.build_parser,
            ["--fixed-iters-e", "0"],
            "fixed-iters-e must be positive",
        ),
        (
            evaluate_rates.build_parser,
            ["--fixed-iters-pi", "5"],
            "fixed-iters-pi must be a positive even integer",
        ),
        (
            evaluate_rates.build_parser,
            ["--max-wave-size", "0"],
            "max-wave-size must be positive",
        ),
        (
            compare_events.build_parser,
            ["--families", "0"],
            "families must be positive",
        ),
        (
            compare_events.build_parser,
            ["--start", "-1"],
            "start must be non-negative",
        ),
        (
            compare_events.build_parser,
            ["--samples", "0"],
            "samples must be positive",
        ),
        (
            compare_events.build_parser,
            ["--seed", "-1"],
            "seed must be non-negative",
        ),
        (
            compare_events.build_parser,
            ["--fixed-iters-pi", "5"],
            "fixed-iters-pi must be a positive even integer",
        ),
        (
            compare_events.build_parser,
            ["--max-iters-e", "0"],
            "max-iters-e must be positive",
        ),
        (
            compare_events.build_parser,
            ["--tol-e", "-0.1"],
            "tol-e must be non-negative",
        ),
    ],
)
def test_local_script_parsers_reject_invalid_count_controls(
    parser_factory,
    args: list[str],
    message: str,
    capsys,
):
    with pytest.raises(SystemExit) as exc_info:
        parser_factory().parse_args(args)

    assert exc_info.value.code == 2
    assert message in capsys.readouterr().err


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


@pytest.mark.parametrize(
    ("checkpoint", "message"),
    [
        ({}, "theta"),
        ({"theta": [1.0, 2.0, 3.0]}, "must be a tensor"),
        ({"theta": torch.zeros(2)}, "D/L/T triples"),
        ({"branchscaled": []}, "branchscaled payload"),
        (
            {
                "branchscaled": {
                    "shared_theta": torch.zeros(2),
                    "branch_log_l": torch.zeros(1),
                }
            },
            "shared_theta",
        ),
        (
            {
                "branchscaled": {
                    "shared_theta": torch.zeros(3),
                    "branch_log_l": "bad",
                }
            },
            "branch_log_l",
        ),
        (
            {
                "branchscaled": {
                    "shared_theta": torch.zeros(3),
                    "branch_log_l": torch.zeros(0),
                }
            },
            "must not be empty",
        ),
    ],
)
def test_export_rates_load_effective_theta_reports_malformed_checkpoint(
    checkpoint: dict[str, object],
    message: str,
):
    with pytest.raises(ValueError, match=message):
        export_rates.load_effective_theta(checkpoint)


def test_export_rates_write_rates_validates_branch_rows(tmp_path: Path):
    path = tmp_path / "rates.tsv"
    branch = {
        "shared_theta": torch.zeros(3),
        "branch_log_l": torch.zeros(1),
    }

    with pytest.raises(ValueError, match="branchscaled branch rows"):
        export_rates.write_rates(path, ["A", "B"], torch.zeros(1, 3), branch)


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
