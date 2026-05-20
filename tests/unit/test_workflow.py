from __future__ import annotations

import csv
import importlib
import json
import math
import os
import sys
from pathlib import Path
from threading import Lock
from types import SimpleNamespace

import pytest
import torch

import gpurec
import gpurec.backtracking as backtracking
import gpurec.api.model as api_model
import gpurec.workflow as workflow
import gpurec.workflow.sampling as sampling_workflow
from gpurec.backtracking import (
    EVENT_KEYS,
    _activate_family_batch,
    _backtrack_command,
    ensure_backtracking_available,
    export_backtracking_input,
    recphyloxml_event_counts,
    sample_recphyloxml,
    sample_recphyloxmls,
)
from gpurec.cli import _sampling_config_from_args
from gpurec.core.batch_planning import normalize_clade_budget, normalize_family_chunk_size
from gpurec.core.model import GeneDataset
from gpurec.api import (
    ActiveFamilyBatch,
    BatchMetadata,
    FamilyInput,
    ReconciliationState,
    UniformChunkMetadata,
)
from gpurec.api.model import GeneReconModel
from gpurec.api.uniform_chunked import (
    UniformBuiltChunk,
    UniformChunkSpec,
    UniformChunkedReconModel,
    _as_auto_int,
    _auto_positive_int,
    _selected_chunks,
)
from gpurec.workflow.checkpoint import (
    CHECKPOINT_VERSION,
    load_checkpoint,
    restore_model_theta,
    save_checkpoint,
)
from gpurec.workflow.config import RunConfig, SamplingConfig
from gpurec.workflow.diagnostics import parameter_stats
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


def test_run_config_from_json_resolves_relative_paths_from_config_file(
    tmp_path: Path,
):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    path = config_dir / "config.json"
    path.write_text(
        json.dumps(
            {
                "species_tree": "inputs/sp.nwk",
                "families_file": "inputs/families.txt",
                "out_dir": "runs/main",
                "preprocess_cache": "cache/preprocess",
                "resume_from": "checkpoints/latest.pt",
                "device": "cpu",
            }
        ),
        encoding="utf-8",
    )

    config = RunConfig.from_json(path)

    assert config.species_tree == (config_dir / "inputs" / "sp.nwk").resolve()
    assert config.families_file == (
        config_dir / "inputs" / "families.txt"
    ).resolve()
    assert config.out_dir == (config_dir / "runs" / "main").resolve()
    assert config.preprocess_cache == (
        config_dir / "cache" / "preprocess"
    ).resolve()
    assert config.resume_from == (
        config_dir / "checkpoints" / "latest.pt"
    ).resolve()


def test_run_config_from_json_expands_user_config_path(
    tmp_path: Path,
    monkeypatch,
):
    home = tmp_path / "home"
    config_dir = home / "configs"
    config_dir.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
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

    config = RunConfig.from_json("~/configs/run.json")

    assert config.species_tree == (config_dir / "inputs" / "sp.nwk").resolve()
    assert config.families_file == (
        config_dir / "inputs" / "families.txt"
    ).resolve()
    assert config.out_dir == (config_dir / "runs" / "main").resolve()


def test_run_config_write_json_creates_parent_directories_and_expands_user(
    tmp_path: Path,
    monkeypatch,
):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
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

    config.write_json("~/configs/nested/run.json")

    output_path = home / "configs" / "nested" / "run.json"
    assert output_path.is_file()
    assert RunConfig.from_json(output_path).to_dict() == config.to_dict()


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("torch.float64", "float64"),
        ("double", "float64"),
        ("fp32", "float32"),
        ("single", "float32"),
    ],
)
def test_run_config_normalizes_dtype_aliases(
    tmp_path: Path,
    value: str,
    expected: str,
):
    config = RunConfig(
        species_tree=tmp_path / "sp.nwk",
        families_file=tmp_path / "families.txt",
        out_dir=tmp_path / "out",
        device="cpu",
        dtype=value,
    )

    assert config.dtype == expected


def test_run_config_write_json_persists_canonical_dtype(tmp_path: Path):
    config = RunConfig(
        species_tree=tmp_path / "sp.nwk",
        families_file=tmp_path / "families.txt",
        out_dir=tmp_path / "out",
        device="cpu",
        dtype="torch.float64",
    )
    path = tmp_path / "config.json"

    config.write_json(path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["dtype"] == "float64"
    assert RunConfig.from_json(path).dtype == "float64"


def test_run_config_from_json_rejects_nonstandard_numeric_constants(tmp_path: Path):
    path = tmp_path / "config.json"
    path.write_text(
        "\n".join(
            [
                "{",
                f'  "species_tree": "{tmp_path / "sp.nwk"}",',
                f'  "families_file": "{tmp_path / "families.txt"}",',
                f'  "out_dir": "{tmp_path / "out"}",',
                '  "tol_e": NaN',
                "}",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid JSON numeric constant NaN"):
        RunConfig.from_json(path)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (None, "species_tree"),
        (42, "species_tree must be a path string"),
    ],
)
def test_run_config_from_json_rejects_bad_required_path(
    tmp_path: Path,
    value: object,
    message: str,
):
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "species_tree": value,
                "families_file": str(tmp_path / "families.txt"),
                "out_dir": str(tmp_path / "out"),
                "device": "cpu",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        RunConfig.from_json(path)


def test_sampling_config_from_cli_args_maps_shared_fields(tmp_path: Path):
    args = SimpleNamespace(
        sample_out_dir=tmp_path / "samples",
        samples=3,
        seed=17,
        family_start=2,
        sample_max_families=4,
        max_events=1000,
        backtrack_binary=tmp_path / "gpurec-backtrack",
    )

    config = _sampling_config_from_args(args, tmp_path / "best.pt")

    assert config.checkpoint == (tmp_path / "best.pt").resolve()
    assert config.out_dir == (tmp_path / "samples").resolve()
    assert config.samples == 3
    assert config.seed == 17
    assert config.family_start == 2
    assert config.max_families == 4
    assert config.max_events == 1000
    assert config.backtrack_binary == (tmp_path / "gpurec-backtrack").resolve()


def test_top_level_exports_api_metadata_types():
    assert set(gpurec.__all__) == set(gpurec._LAZY_EXPORTS)
    assert gpurec.ActiveFamilyBatch is ActiveFamilyBatch
    assert gpurec.BatchMetadata is BatchMetadata
    assert gpurec.FamilyInput is FamilyInput
    assert gpurec.ReconciliationState is ReconciliationState
    assert gpurec.UniformChunkMetadata is UniformChunkMetadata
    for name in (
        "ActiveFamilyBatch",
        "BatchMetadata",
        "FamilyInput",
        "ReconciliationState",
        "UniformChunkMetadata",
    ):
        assert name in gpurec.__all__


def test_top_level_exports_backtracking_surface():
    for name in (
        "EVENT_KEYS",
        "ensure_backtracking_available",
        "export_backtracking_input",
        "recphyloxml_event_counts",
        "sample_recphyloxml",
        "sample_recphyloxmls",
    ):
        assert name in gpurec.__all__
        assert gpurec._LAZY_EXPORTS[name] == "gpurec.backtracking"
        assert getattr(gpurec, name) is getattr(backtracking, name)


def test_top_level_exports_workflow_surface():
    for name in workflow.__all__:
        assert name in gpurec.__all__
        assert gpurec._LAZY_EXPORTS[name] == "gpurec.workflow"
        assert getattr(gpurec, name) is getattr(workflow, name)


def test_uniform_chunked_public_chunk_metadata_accessors():
    chunks = [
        UniformBuiltChunk(
            spec=UniformChunkSpec(indices=[0, 2], clades=7, splits=11),
            wave_layout={},
            waves=3,
            max_wave=5,
            split_rows=13,
            max_wave_split_rows=8,
        ),
        UniformBuiltChunk(
            spec=UniformChunkSpec(indices=[1], clades=4, splits=6),
            wave_layout={},
            waves=2,
            max_wave=4,
            split_rows=9,
            max_wave_split_rows=7,
        ),
    ]
    model = UniformChunkedReconModel.__new__(UniformChunkedReconModel)
    model._state = SimpleNamespace(
        built_chunks=chunks,
        dataset=SimpleNamespace(families=[object(), object(), object()]),
        fixed_iters_Pi=6,
        fixed_iters_E=4,
    )
    model.family_names = ["family_0", "family_1", "family_2"]
    model.gene_trees = [["g0.nwk"], ["g1a.nwk", "g1b.nwk"], ["g2.nwk"]]

    assert model.n_families == 3
    assert model.family_count == 3
    assert model.chunk_count == 2
    assert model.fixed_iters_Pi == 6
    assert model.fixed_iters_E == 4
    assert model.chunk_metadata == (
        UniformChunkMetadata(
            chunk_index=0,
            family_indices=(0, 2),
            family_names=("family_0", "family_2"),
            gene_tree_paths=(("g0.nwk",), ("g2.nwk",)),
            family_count=2,
            clade_count=7,
            split_count=11,
            wave_count=3,
            max_wave_size=5,
            split_rows=13,
            max_wave_split_rows=8,
        ),
        UniformChunkMetadata(
            chunk_index=1,
            family_indices=(1,),
            family_names=("family_1",),
            gene_tree_paths=(("g1a.nwk", "g1b.nwk"),),
            family_count=1,
            clade_count=4,
            split_count=6,
            wave_count=2,
            max_wave_size=4,
            split_rows=9,
            max_wave_split_rows=7,
        ),
    )


def test_close_prevents_later_prefetch_restart(monkeypatch):
    class FakeExecutor:
        instances: list["FakeExecutor"] = []

        def __init__(self, *, max_workers: int, thread_name_prefix: str):
            self.max_workers = max_workers
            self.thread_name_prefix = thread_name_prefix
            self.submitted: list[int] = []
            self.shutdown_kwargs: dict[str, bool] | None = None
            FakeExecutor.instances.append(self)

        def submit(self, func, batch_idx: int):
            self.submitted.append(batch_idx)
            return object()

        def shutdown(self, *, wait: bool, cancel_futures: bool) -> None:
            self.shutdown_kwargs = {
                "wait": wait,
                "cancel_futures": cancel_futures,
            }

    monkeypatch.setattr(api_model, "ThreadPoolExecutor", FakeExecutor)

    model = GeneReconModel.__new__(GeneReconModel)
    model._batched_resident = True
    model.prefetch_batches = "all"
    model._current_batch_index = 0
    model._batch_specs = [object(), object(), object()]
    model._batch_statics = [object(), None, None]
    model._batch_futures = {}
    model._prefetch_executor = None
    model._prefetch_closed = False
    model._batch_lock = Lock()
    model._build_batch_static = lambda batch_idx: object()

    model._schedule_prefetch()
    assert len(FakeExecutor.instances) == 1
    assert FakeExecutor.instances[0].submitted == [1, 2]

    model.close()
    assert FakeExecutor.instances[0].shutdown_kwargs == {
        "wait": False,
        "cancel_futures": True,
    }
    assert model._prefetch_executor is None
    assert model._batch_futures == {}

    model._schedule_prefetch()

    assert len(FakeExecutor.instances) == 1
    assert model._prefetch_executor is None
    assert model._batch_futures == {}


def test_family_input_returns_defensive_copies():
    ccp_helpers = {
        "split_counts": torch.tensor([0, 1], dtype=torch.long),
        "nested": {
            "weights": torch.tensor([0.25, 0.75], dtype=torch.float32),
            "labels": ["left"],
        },
    }
    leaf_row_index = torch.tensor([0, 1], dtype=torch.long)
    leaf_col_index = torch.tensor([1, 0], dtype=torch.long)
    family_record = {
        "C": 2,
        "N_splits": 1,
        "root_clade_id": 0,
        "ccp_helpers": ccp_helpers,
        "leaf_row_index": leaf_row_index,
        "leaf_col_index": leaf_col_index,
        "clade_leaf_labels": ["gene_a", ""],
    }
    dataset = SimpleNamespace(
        families=[family_record],
        family_names=["fam0"],
        gene_tree_paths=[["g0.nwk"]],
        leaf_species_maps=[{"gene_a": "SpeciesA"}],
    )
    model = object.__new__(GeneReconModel)
    object.__setattr__(model, "_dataset", dataset)

    public_family = GeneReconModel.family_input(model, 0)
    public_family.ccp_helpers["split_counts"][0] = 99
    public_family.ccp_helpers["nested"]["weights"][0] = 42.0
    public_family.ccp_helpers["nested"]["labels"].append("mutated")
    public_family.leaf_row_index[0] = 99
    public_family.leaf_col_index[0] = 99
    public_family.gene_tree_paths.append("mutated.nwk")
    public_family.leaf_species_map["gene_a"] = "Mutated"
    public_family.clade_leaf_labels.append("mutated")

    torch.testing.assert_close(ccp_helpers["split_counts"], torch.tensor([0, 1]))
    torch.testing.assert_close(
        ccp_helpers["nested"]["weights"],
        torch.tensor([0.25, 0.75]),
    )
    assert ccp_helpers["nested"]["labels"] == ["left"]
    torch.testing.assert_close(leaf_row_index, torch.tensor([0, 1]))
    torch.testing.assert_close(leaf_col_index, torch.tensor([1, 0]))
    assert dataset.gene_tree_paths == [["g0.nwk"]]
    assert dataset.leaf_species_maps == [{"gene_a": "SpeciesA"}]
    assert family_record["clade_leaf_labels"] == ["gene_a", ""]


def _public_selector_model() -> GeneReconModel:
    model = object.__new__(GeneReconModel)
    object.__setattr__(
        model,
        "_dataset",
        SimpleNamespace(
            families=[
                {
                    "C": 1,
                    "N_splits": 0,
                    "root_clade_id": 0,
                    "ccp_helpers": {},
                    "leaf_row_index": torch.empty(0, dtype=torch.long),
                    "leaf_col_index": torch.empty(0, dtype=torch.long),
                }
            ],
            family_names=["fam0"],
            gene_tree_paths=[["g0.nwk"]],
            leaf_species_maps=[{}],
        ),
    )
    model.batch_metadata = [SimpleNamespace(family_indices=[0])]
    model._current_batch_index = 0
    return model


@pytest.mark.parametrize(
    ("method", "field"),
    [
        ("family_input", "family_index"),
        ("activate_family", "family_index"),
        ("select_batch", "batch_index"),
    ],
)
@pytest.mark.parametrize("value", [True, 1.5, math.inf, math.nan])
def test_public_selectors_reject_nonintegral_indices(
    method: str,
    field: str,
    value: object,
):
    model = _public_selector_model()

    with pytest.raises(ValueError, match=field):
        getattr(GeneReconModel, method)(model, value)  # type: ignore[arg-type]


@pytest.mark.parametrize("method", ["family_input", "activate_family"])
@pytest.mark.parametrize("value", [-1, 1])
def test_family_selectors_reject_out_of_range_indices(method: str, value: int):
    model = _public_selector_model()

    with pytest.raises(IndexError, match="family_index"):
        getattr(GeneReconModel, method)(model, value)


@pytest.mark.parametrize("value", [-1, 1])
def test_select_batch_rejects_out_of_range_indices(value: int):
    model = _public_selector_model()

    with pytest.raises(IndexError, match="batch index"):
        GeneReconModel.select_batch(model, value)


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


def test_run_config_normalizes_direct_float_controls(tmp_path: Path):
    config = RunConfig(
        species_tree=tmp_path / "sp.nwk",
        families_file=tmp_path / "families.txt",
        out_dir=tmp_path / "out",
        device="cpu",
        theta_init_d=1,
        theta_init_l="0.2",
        theta_init_t="3e-1",
        min_rate="1e-8",
        max_rate=10,
        lr="0.25",
    )

    assert config.theta_init_rates == (1.0, 0.2, 0.3)
    assert config.min_rate == 1e-8
    assert config.max_rate == 10.0
    assert config.lr == 0.25
    assert all(
        isinstance(value, float)
        for value in (
            config.theta_init_d,
            config.theta_init_l,
            config.theta_init_t,
            config.min_rate,
            config.max_rate,
            config.lr,
        )
    )


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


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("tol_e", math.nan),
        ("tol_e", math.inf),
        ("best_likelihood_min_delta", math.nan),
        ("min_rate", math.nan),
        ("max_rate", math.inf),
        ("theta_init_d", math.nan),
        ("theta_init_l", math.inf),
        ("lr", math.nan),
        ("lbfgs_lr", math.inf),
    ],
)
def test_run_config_rejects_nonfinite_float_controls(
    tmp_path: Path,
    field: str,
    value: float,
):
    with pytest.raises(ValueError, match=field):
        RunConfig(
            species_tree=tmp_path / "sp.nwk",
            families_file=tmp_path / "families.txt",
            out_dir=tmp_path / "out",
            device="cpu",
            **{field: value},
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("tol_e", True),
        ("theta_init_d", False),
        ("lr", True),
        ("min_rate", True),
    ],
)
def test_run_config_rejects_boolean_float_controls(
    tmp_path: Path,
    field: str,
    value: bool,
):
    with pytest.raises(ValueError, match=field):
        RunConfig(
            species_tree=tmp_path / "sp.nwk",
            families_file=tmp_path / "families.txt",
            out_dir=tmp_path / "out",
            device="cpu",
            **{field: value},
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("refresh_preprocess_cache", "false"),
        ("refresh_preprocess_cache", 1),
        ("adaptive_iters", "false"),
        ("adaptive_iters", 0),
    ],
)
def test_run_config_rejects_nonbool_boolean_controls(
    tmp_path: Path,
    field: str,
    value: object,
):
    with pytest.raises(ValueError, match=f"{field} must be true or false"):
        RunConfig(
            species_tree=tmp_path / "sp.nwk",
            families_file=tmp_path / "families.txt",
            out_dir=tmp_path / "out",
            device="cpu",
            **{field: value},
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("start", 0.5),
        ("max_families", 1.5),
        ("fixed_iters_e", 1.5),
        ("max_iters_e", 2000.5),
        ("fixed_iters_pi", 64.5),
        ("neumann_terms", True),
        ("convergence_check_interval", 4.5),
        ("steps", 1.5),
        ("adam_warmup_steps", 0.5),
        ("lbfgs_max_iter", 1.5),
        ("checkpoint_every", 0.5),
        ("log_every", 1.5),
    ],
)
def test_run_config_rejects_nonintegral_integer_controls(
    tmp_path: Path,
    field: str,
    value: object,
):
    with pytest.raises(ValueError, match=field):
        RunConfig(
            species_tree=tmp_path / "sp.nwk",
            families_file=tmp_path / "families.txt",
            out_dir=tmp_path / "out",
            device="cpu",
            **{field: value},
        )


def test_run_config_rejects_odd_fixed_pi_iterations(tmp_path: Path):
    with pytest.raises(ValueError, match="fixed_iters_pi"):
        RunConfig(
            species_tree=tmp_path / "sp.nwk",
            families_file=tmp_path / "families.txt",
            out_dir=tmp_path / "out",
            device="cpu",
            fixed_iters_pi=3,
        )


def test_family_chunk_size_normalization_is_shared():
    for value in (None, "", "0", "all", "none", "null", 0):
        assert normalize_family_chunk_size(value) == 0
    assert normalize_family_chunk_size("12") == 12
    assert normalize_family_chunk_size(12.0) == 12
    assert normalize_family_chunk_size("auto", allow_auto=True) == "auto"
    for value in ("auto", True, 1.5, math.inf):
        with pytest.raises(ValueError, match="family"):
            normalize_family_chunk_size(value)


@pytest.mark.parametrize("value", [True, False, 1.5])
def test_uniform_auto_int_rejects_bool_and_nonintegral_float(value: object):
    with pytest.raises(ValueError, match="family_chunk_size"):
        _as_auto_int("family_chunk_size", value)


@pytest.mark.parametrize("value", [None, "0", "none", "null"])
def test_uniform_auto_positive_int_preserves_unbounded_aliases(value: object):
    assert _auto_positive_int("max_wave_size", value) is None


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"family_chunk_size": True}, "family_chunk_size"),
        ({"family_chunk_size": -1}, "family_chunk_size"),
        ({"max_wave_size": 0}, "max_wave_size"),
        ({"max_wave_size": -1}, "max_wave_size"),
        ({"max_wave_size": 2.5}, "max_wave_size"),
        ({"max_root_wave_size": 0}, "max_root_wave_size"),
        ({"max_root_wave_size": 1.5}, "max_root_wave_size"),
        ({"max_root_wave_size": True}, "max_root_wave_size"),
        ({"clade_budget": True}, "clade_budget"),
        ({"clade_budget": 0}, "clade_budget"),
        ({"batch_packing": "unknown"}, "batch_packing"),
        ({"family_chunk_candidates": None}, "family_chunk_candidates"),
        ({"family_chunk_candidates": [-1]}, "family_chunk_candidates"),
        ({"family_chunk_candidates": [1.5]}, "family_chunk_candidates"),
        ({"max_wave_candidates": None}, "max_wave_candidates"),
        ({"max_wave_candidates": [0]}, "max_wave_candidates"),
        ({"max_wave_candidates": [1.5]}, "max_wave_candidates"),
    ],
)
def test_uniform_chunked_rejects_bad_chunk_controls_before_device_or_io(
    tmp_path: Path,
    kwargs: dict[str, object],
    message: str,
):
    with pytest.raises(ValueError, match=message) as exc_info:
        UniformChunkedReconModel.from_trees(
            tmp_path / "missing_species.nwk",
            [tmp_path / "missing_gene.nwk"],
            device="cpu",
            **kwargs,
        )

    assert "CUDA" not in str(exc_info.value)
    assert "missing" not in str(exc_info.value)


def _uniform_chunk_state(count: int = 2) -> SimpleNamespace:
    return SimpleNamespace(
        built_chunks=[
            SimpleNamespace(spec=SimpleNamespace(indices=[idx]))
            for idx in range(count)
        ],
        dataset=SimpleNamespace(families=[{} for _ in range(count)]),
        origination_probs=None,
    )


@pytest.mark.parametrize(
    "chunk_indices",
    [
        [True],
        [0.5],
        [math.inf],
        [math.nan],
        torch.tensor([True]),
        torch.tensor([0.5]),
    ],
)
def test_uniform_chunk_selector_rejects_nonintegral_indices(chunk_indices: object):
    with pytest.raises(ValueError, match="chunk_indices"):
        _selected_chunks(_uniform_chunk_state(), chunk_indices)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("chunk_indices", "message"),
    [
        ([], "must not be empty"),
        ([0, 0], "duplicate chunk index"),
        ([-1], "out of range"),
        ([2], "out of range"),
    ],
)
def test_uniform_chunk_selector_rejects_invalid_index_sets(
    chunk_indices: object,
    message: str,
):
    with pytest.raises((IndexError, ValueError), match=message):
        _selected_chunks(_uniform_chunk_state(), chunk_indices)  # type: ignore[arg-type]


def test_uniform_chunk_selector_accepts_integral_float_indices():
    selected = _selected_chunks(_uniform_chunk_state(), [1.0])

    assert [idx for idx, _chunk in selected] == [1]


@pytest.mark.parametrize("value", ["12", 12, 12.0])
def test_clade_budget_normalization_accepts_integral_values(value: object):
    assert normalize_clade_budget(value) == 12


@pytest.mark.parametrize("value", [True, 1.5, math.inf, 0, -1])
def test_clade_budget_normalization_rejects_bad_values(value: object):
    with pytest.raises(ValueError, match="clade_budget"):
        normalize_clade_budget(value)


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


@pytest.mark.parametrize(
    "rates",
    [
        (math.nan, 0.1, 0.1),
        (math.inf, 0.1, 0.1),
    ],
)
def test_public_model_constructors_reject_nonfinite_theta_init_before_io(
    tmp_path: Path,
    rates: tuple[float, float, float],
):
    with pytest.raises(ValueError, match="theta_init_rates must be finite"):
        GeneReconModel.from_trees(
            tmp_path / "missing_species.nwk",
            [tmp_path / "missing_gene.nwk"],
            device="cpu",
            theta_init_rates=rates,
        )
    with pytest.raises(ValueError, match="theta_init_rates must be finite"):
        UniformChunkedReconModel.from_trees(
            tmp_path / "missing_species.nwk",
            [tmp_path / "missing_gene.nwk"],
            device="cpu",
            theta_init_rates=rates,
        )


@pytest.mark.parametrize("case", ["string", "path", "empty"])
@pytest.mark.parametrize(
    "factory",
    ["gene_from_trees", "uniform_from_trees", "uniform_direct"],
)
def test_public_tree_constructors_reject_invalid_gene_trees_before_device(
    tmp_path: Path,
    monkeypatch,
    factory: str,
    case: str,
):
    if case == "string":
        gene_trees: object = "g.nwk"
    elif case == "path":
        gene_trees = tmp_path / "missing_gene.nwk"
    else:
        gene_trees = []

    monkeypatch.delenv("GPUREC_SELF_LOOP_2D_BLOCK_W", raising=False)

    with pytest.raises(ValueError, match="gene_trees") as exc_info:
        if factory == "gene_from_trees":
            GeneReconModel.from_trees(
                tmp_path / "missing_species.nwk",
                gene_trees,  # type: ignore[arg-type]
                device="cpu",
            )
        elif factory == "uniform_from_trees":
            UniformChunkedReconModel.from_trees(
                tmp_path / "missing_species.nwk",
                gene_trees,  # type: ignore[arg-type]
                device="cpu",
            )
        else:
            UniformChunkedReconModel(
                species_tree=tmp_path / "missing_species.nwk",
                gene_trees=gene_trees,  # type: ignore[arg-type]
                device="cpu",
            )

    assert "CUDA" not in str(exc_info.value)
    assert "missing" not in str(exc_info.value)
    assert "GPUREC_SELF_LOOP_2D_BLOCK_W" not in os.environ


def test_gene_dataset_rejects_single_gene_tree_path_before_extension(
    tmp_path: Path,
    monkeypatch,
):
    calls: list[bool] = []

    def fake_load_extension():
        calls.append(True)
        raise AssertionError("_load_species_gene_ext should not run")

    monkeypatch.setattr("gpurec.core.model._load_species_gene_ext", fake_load_extension)

    with pytest.raises(ValueError, match="gene_trees"):
        GeneDataset(
            species_tree_path=tmp_path / "missing_species.nwk",
            gene_tree_paths="g.nwk",
            genewise=False,
            specieswise=False,
            device=torch.device("cpu"),
        )

    assert calls == []


@pytest.mark.parametrize("dtype", [torch.int64, torch.float16, "float32"])
def test_gene_recon_init_rejects_invalid_dtype_before_device(dtype: object):
    dataset = SimpleNamespace(
        genewise=False,
        specieswise=False,
        device=torch.device("cpu"),
        dtype=dtype,
    )

    with pytest.raises(ValueError, match="dtype"):
        GeneReconModel(dataset=dataset, mode="global")


@pytest.mark.parametrize("dtype", [torch.int64, torch.float16, "float32"])
@pytest.mark.parametrize("factory", ["from_trees", "from_alerax_families"])
def test_gene_recon_factories_reject_invalid_dtype_before_device_or_io(
    tmp_path: Path,
    factory: str,
    dtype: object,
):
    with pytest.raises(ValueError, match="dtype") as exc_info:
        if factory == "from_trees":
            GeneReconModel.from_trees(
                tmp_path / "missing_species.nwk",
                [tmp_path / "missing_gene.nwk"],
                device="cpu",
                dtype=dtype,
                theta_init_rates=(0.1, 0.1, 0.1),
            )
        else:
            GeneReconModel.from_alerax_families(
                tmp_path / "missing_species.nwk",
                tmp_path / "missing_families.txt",
                device="cpu",
                dtype=dtype,
                theta_init_rates=(0.1, 0.1, 0.1),
            )

    assert "CUDA" not in str(exc_info.value)


@pytest.mark.parametrize("dtype", [torch.int64, torch.float16, "float32"])
@pytest.mark.parametrize("factory", ["from_trees", "from_folder", "from_alerax_families"])
def test_uniform_chunked_factories_reject_invalid_dtype_before_device_or_io(
    tmp_path: Path,
    factory: str,
    dtype: object,
):
    with pytest.raises(ValueError, match="dtype") as exc_info:
        if factory == "from_trees":
            UniformChunkedReconModel.from_trees(
                tmp_path / "missing_species.nwk",
                [tmp_path / "missing_gene.nwk"],
                device="cpu",
                dtype=dtype,
                theta_init_rates=(0.1, 0.1, 0.1),
            )
        elif factory == "from_folder":
            UniformChunkedReconModel.from_folder(
                tmp_path / "missing_folder",
                device="cpu",
                dtype=dtype,
                theta_init_rates=(0.1, 0.1, 0.1),
            )
        else:
            UniformChunkedReconModel.from_alerax_families(
                tmp_path / "missing_species.nwk",
                tmp_path / "missing_families.txt",
                device="cpu",
                dtype=dtype,
                theta_init_rates=(0.1, 0.1, 0.1),
            )

    message = str(exc_info.value)
    assert "CUDA" not in message
    assert "missing" not in message


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("fixed_iters_E", 0),
        ("fixed_iters_E", 1.5),
        ("fixed_iters_E", math.nan),
        ("fixed_iters_Pi", 3),
        ("fixed_iters_Pi", 4.5),
        ("neumann_terms", 0),
        ("neumann_terms", True),
        ("neumann_terms", math.inf),
        ("convergence_check_interval", 0),
        ("convergence_check_interval", 2.5),
        ("max_iters_E", 0),
        ("max_iters_E", 10.5),
        ("max_iters_E", math.inf),
        ("tol_E", math.nan),
        ("e_logsumexp_tol", math.inf),
        ("pi_max_diff_tol", math.nan),
        ("gradient_change_tol", math.inf),
        ("gradient_change_rtol", math.nan),
        ("pruning_threshold", math.inf),
        ("adaptive_iters", "false"),
        ("use_pruning", "false"),
        ("family_chunk_size", True),
        ("family_chunk_size", 1.5),
        ("max_wave_size", 0),
        ("max_wave_size", -1),
        ("max_wave_size", 1.5),
        ("max_wave_size", True),
        ("max_root_wave_size", 0),
        ("max_root_wave_size", 1.5),
        ("max_root_wave_size", True),
        ("max_dts_partial_rows", 0),
        ("max_dts_partial_rows", 1.5),
        ("max_dts_partial_rows", True),
        ("clade_budget", True),
        ("clade_budget", 0),
        ("lazy_preprocess", "false"),
        ("prefetch_batches", True),
        ("prefetch_batches", 1.5),
        ("prefetch_batches", "many"),
    ],
)
def test_gene_recon_init_rejects_invalid_solver_controls_before_device(
    field: str,
    value: object,
):
    dataset = SimpleNamespace(
        genewise=False,
        specieswise=False,
        device=torch.device("cpu"),
    )

    with pytest.raises(ValueError, match=field):
        GeneReconModel(dataset=dataset, mode="global", **{field: value})


@pytest.mark.parametrize(
    ("factory", "kwargs", "message"),
    [
        ("from_trees", {"tol_E": math.nan}, "tol_E"),
        ("from_trees", {"fixed_iters_E": 1.5}, "fixed_iters_E"),
        ("from_trees", {"max_iters_E": 0}, "max_iters_E"),
        ("from_trees", {"max_iters_E": 20.5}, "max_iters_E"),
        ("from_trees", {"fixed_iters_Pi": math.inf}, "fixed_iters_Pi"),
        ("from_trees", {"fixed_iters_Pi": 4.5}, "fixed_iters_Pi"),
        ("from_trees", {"adaptive_iters": "false"}, "adaptive_iters"),
        ("from_trees", {"use_pruning": "false"}, "use_pruning"),
        (
            "from_trees",
            {"refresh_preprocess_cache": "false"},
            "refresh_preprocess_cache",
        ),
        ("from_trees", {"family_chunk_size": True}, "family_chunk_size"),
        ("from_trees", {"max_wave_size": 0}, "max_wave_size"),
        ("from_trees", {"max_wave_size": 1.5}, "max_wave_size"),
        ("from_trees", {"max_root_wave_size": 0}, "max_root_wave_size"),
        ("from_trees", {"max_dts_partial_rows": 0}, "max_dts_partial_rows"),
        ("from_trees", {"clade_budget": True}, "clade_budget"),
        ("from_trees", {"lazy_preprocess": "false"}, "lazy_preprocess"),
        (
            "from_alerax_families",
            {"gradient_change_rtol": math.inf},
            "gradient_change_rtol",
        ),
        (
            "from_alerax_families",
            {"neumann_terms": True},
            "neumann_terms",
        ),
        (
            "from_alerax_families",
            {"max_wave_size": -1},
            "max_wave_size",
        ),
        (
            "from_alerax_families",
            {"max_root_wave_size": True},
            "max_root_wave_size",
        ),
        (
            "from_alerax_families",
            {"max_dts_partial_rows": 1.5},
            "max_dts_partial_rows",
        ),
        (
            "from_alerax_families",
            {"adaptive_iters": True, "convergence_check_interval": 3},
            "adaptive_iters",
        ),
        (
            "from_alerax_families",
            {"refresh_preprocess_cache": "false"},
            "refresh_preprocess_cache",
        ),
        (
            "from_alerax_families",
            {"prefetch_batches": True},
            "prefetch_batches",
        ),
        (
            "from_alerax_families",
            {"prefetch_batches": "many"},
            "prefetch_batches",
        ),
    ],
)
def test_gene_recon_factories_reject_invalid_solver_controls_before_device_or_io(
    tmp_path: Path,
    factory: str,
    kwargs: dict[str, object],
    message: str,
):
    with pytest.raises(ValueError, match=message) as exc_info:
        if factory == "from_trees":
            GeneReconModel.from_trees(
                tmp_path / "missing_species.nwk",
                [tmp_path / "missing_gene.nwk"],
                device="cpu",
                **kwargs,
            )
        else:
            GeneReconModel.from_alerax_families(
                tmp_path / "missing_species.nwk",
                tmp_path / "missing_families.txt",
                device="cpu",
                **kwargs,
            )

    assert "CUDA" not in str(exc_info.value)
    assert "missing" not in str(exc_info.value)


@pytest.mark.parametrize(
    ("factory", "kwargs", "message"),
    [
        ("from_trees", {"tol_E": math.nan}, "tol_E"),
        ("from_trees", {"fixed_iters_E": 1.5}, "fixed_iters_E"),
        ("from_trees", {"max_iters_E": 0}, "max_iters_E"),
        ("from_trees", {"max_iters_E": 20.5}, "max_iters_E"),
        ("from_trees", {"fixed_iters_Pi": math.nan}, "fixed_iters_Pi"),
        ("from_trees", {"fixed_iters_Pi": 4.5}, "fixed_iters_Pi"),
        ("from_trees", {"family_chunk_size": -1}, "family_chunk_size"),
        ("from_trees", {"max_wave_size": 0}, "max_wave_size"),
        ("from_trees", {"max_root_wave_size": True}, "max_root_wave_size"),
        ("from_trees", {"clade_budget": 0}, "clade_budget"),
        ("from_trees", {"batch_packing": "unknown"}, "batch_packing"),
        (
            "from_trees",
            {"family_chunk_candidates": [1.5]},
            "family_chunk_candidates",
        ),
        ("from_trees", {"max_wave_candidates": [0]}, "max_wave_candidates"),
        ("from_folder", {"max_wave_size": -1}, "max_wave_size"),
        (
            "from_alerax_families",
            {"pruning_threshold": math.inf},
            "pruning_threshold",
        ),
        ("from_alerax_families", {"neumann_terms": 0}, "neumann_terms"),
        ("from_alerax_families", {"neumann_terms": True}, "neumann_terms"),
        (
            "from_alerax_families",
            {"family_chunk_candidates": [-1]},
            "family_chunk_candidates",
        ),
        (
            "from_alerax_families",
            {"max_wave_candidates": [1.5]},
            "max_wave_candidates",
        ),
    ],
)
def test_uniform_chunked_factories_reject_invalid_solver_controls_before_device_or_io(
    tmp_path: Path,
    factory: str,
    kwargs: dict[str, object],
    message: str,
):
    with pytest.raises(ValueError, match=message) as exc_info:
        if factory == "from_trees":
            UniformChunkedReconModel.from_trees(
                tmp_path / "missing_species.nwk",
                [tmp_path / "missing_gene.nwk"],
                device="cpu",
                **kwargs,
            )
        elif factory == "from_folder":
            UniformChunkedReconModel.from_folder(
                tmp_path / "missing_folder",
                device="cpu",
                **kwargs,
            )
        else:
            UniformChunkedReconModel.from_alerax_families(
                tmp_path / "missing_species.nwk",
                tmp_path / "missing_families.txt",
                device="cpu",
                **kwargs,
            )

    assert "CUDA" not in str(exc_info.value)
    assert "missing" not in str(exc_info.value)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"refresh_preprocess_cache": "false"}, "refresh_preprocess_cache"),
        ({"use_pruning": "false"}, "use_pruning"),
        ({"warm_start_E": "false"}, "warm_start_E"),
        ({"profile": "false"}, "profile"),
        ({"set_optimized_env": "false"}, "set_optimized_env"),
    ],
)
def test_uniform_chunked_init_rejects_nonbool_controls_before_side_effects(
    tmp_path: Path,
    monkeypatch,
    kwargs: dict[str, object],
    message: str,
):
    monkeypatch.delenv("GPUREC_SELF_LOOP_2D_BLOCK_W", raising=False)

    with pytest.raises(ValueError, match=message) as exc_info:
        UniformChunkedReconModel(
            species_tree=tmp_path / "missing_species.nwk",
            gene_trees=[tmp_path / "missing_gene.nwk"],
            device="cpu",
            **kwargs,
        )

    assert "CUDA" not in str(exc_info.value)
    assert "GPUREC_SELF_LOOP_2D_BLOCK_W" not in os.environ


@pytest.mark.parametrize(
    ("factory", "args"),
    [
        (
            "from_trees",
            lambda tmp_path: (
                tmp_path / "missing_species.nwk",
                [tmp_path / "missing_gene.nwk"],
            ),
        ),
        (
            "from_folder",
            lambda tmp_path: (tmp_path / "missing_folder",),
        ),
        (
            "from_alerax_families",
            lambda tmp_path: (
                tmp_path / "missing_species.nwk",
                tmp_path / "missing_families.txt",
            ),
        ),
    ],
)
@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"refresh_preprocess_cache": "false"}, "refresh_preprocess_cache"),
        ({"use_pruning": "false"}, "use_pruning"),
        ({"warm_start_E": "false"}, "warm_start_E"),
        ({"profile": "false"}, "profile"),
        ({"set_optimized_env": "false"}, "set_optimized_env"),
    ],
)
def test_uniform_chunked_factories_reject_nonbool_controls_before_device_or_io(
    tmp_path: Path,
    factory: str,
    args,
    kwargs: dict[str, object],
    message: str,
):
    make_model = getattr(UniformChunkedReconModel, factory)

    with pytest.raises(ValueError, match=message) as exc_info:
        make_model(*args(tmp_path), device="cpu", **kwargs)

    assert "CUDA" not in str(exc_info.value)
    assert "missing" not in str(exc_info.value)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"fixed_iters_Pi": math.inf}, "fixed_iters_Pi"),
        ({"fixed_iters_Pi": 4.5}, "fixed_iters_Pi"),
        ({"neumann_terms": math.nan}, "neumann_terms"),
        ({"neumann_terms": True}, "neumann_terms"),
        ({"pi_max_diff_tol": math.nan}, "pi_max_diff_tol"),
        ({"gradient_change_tol": math.inf}, "gradient_change_tol"),
    ],
)
def test_gene_recon_configure_solver_iterations_rejects_nonfinite_tolerances(
    kwargs: dict[str, object],
    message: str,
):
    with pytest.raises(ValueError, match=message):
        GeneReconModel.configure_solver_iterations(SimpleNamespace(), **kwargs)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"min_rate": math.nan}, "min_rate"),
        ({"max_rate": math.inf}, "max_rate"),
        ({"min_rate": 0.0}, "min_rate"),
    ],
)
def test_gene_recon_clamp_theta_rejects_invalid_rates(
    kwargs: dict[str, float],
    message: str,
):
    model = SimpleNamespace(theta=torch.nn.Parameter(torch.zeros(3)))

    with pytest.raises(ValueError, match=message):
        GeneReconModel.clamp_theta_(model, **kwargs)


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


@pytest.mark.parametrize(
    "factory",
    [
        pytest.param(GeneReconModel.from_alerax_families, id="resident"),
        pytest.param(UniformChunkedReconModel.from_alerax_families, id="uniform"),
    ],
)
@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"start": -1}, "start"),
        ({"start": 0.5}, "start"),
        ({"start": True}, "start"),
        ({"max_families": 0}, "max_families"),
        ({"max_families": 1.5}, "max_families"),
        ({"max_families": True}, "max_families"),
    ],
)
def test_alerax_constructors_validate_selection_before_device_or_io(
    tmp_path: Path,
    factory,
    kwargs: dict[str, object],
    message: str,
):
    with pytest.raises(ValueError, match=message) as exc_info:
        factory(
            tmp_path / "missing_species.nwk",
            tmp_path / "missing_families.txt",
            device="cpu",
            **kwargs,
        )

    assert "CUDA" not in str(exc_info.value)


def test_models_reject_alerax_compat_env_before_device_or_io(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setenv("GPUREC_ALERAX_COMPAT", "1")
    dataset = SimpleNamespace(
        genewise=False,
        specieswise=False,
        device=torch.device("cpu"),
    )
    calls = [
        lambda: GeneReconModel(dataset=dataset, mode="global"),
        lambda: GeneReconModel.from_trees(
            tmp_path / "missing_species.nwk",
            [tmp_path / "missing_gene.nwk"],
            device="cpu",
        ),
        lambda: GeneReconModel.from_alerax_families(
            tmp_path / "missing_species.nwk",
            tmp_path / "missing_families.txt",
            device="cpu",
        ),
        lambda: UniformChunkedReconModel.from_trees(
            tmp_path / "missing_species.nwk",
            [tmp_path / "missing_gene.nwk"],
            device="cpu",
        ),
        lambda: UniformChunkedReconModel.from_folder(
            tmp_path / "missing_folder",
            device="cpu",
        ),
        lambda: UniformChunkedReconModel.from_alerax_families(
            tmp_path / "missing_species.nwk",
            tmp_path / "missing_families.txt",
            device="cpu",
        ),
    ]

    for make_model in calls:
        with pytest.raises(RuntimeError, match="GPUREC_ALERAX_COMPAT") as exc_info:
            make_model()
        message = str(exc_info.value)
        assert "unset GPUREC_ALERAX_COMPAT" in message
        assert "CUDA" not in message


def test_uniform_chunked_alerax_constructor_validates_mode_before_io(tmp_path: Path):
    with pytest.raises(ValueError, match="from_alerax_families"):
        UniformChunkedReconModel.from_alerax_families(
            tmp_path / "missing_species.nwk",
            tmp_path / "missing_families.txt",
            mode="genewise",
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"start": -1}, "start"),
        ({"max_families": 0}, "max_families"),
    ],
)
def test_uniform_chunked_from_folder_validates_selection_before_io(
    tmp_path: Path,
    kwargs: dict[str, int],
    message: str,
):
    with pytest.raises(ValueError, match=message):
        UniformChunkedReconModel.from_folder(tmp_path / "missing_folder", **kwargs)


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
    with pytest.raises(ValueError, match="seed"):
        SamplingConfig(checkpoint=checkpoint, seed=1 << 64)
    with pytest.raises(ValueError, match="max_events"):
        SamplingConfig(checkpoint=checkpoint, max_events=0)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("samples", True),
        ("seed", True),
        ("family_start", 1.5),
        ("max_families", 2.2),
        ("max_events", True),
    ],
)
def test_sampling_config_rejects_nonintegral_limits(
    tmp_path: Path,
    field: str,
    value: object,
):
    with pytest.raises(ValueError, match=field):
        SamplingConfig(
            checkpoint=tmp_path / "checkpoints" / "best.pt",
            **{field: value},
        )


def test_public_backtracking_rejects_invalid_seed_and_event_limits():
    model = object()
    with pytest.raises(ValueError, match="family_index"):
        export_backtracking_input(model, family_index=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="family_index"):
        sample_recphyloxml(model, family_index=1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="family_index"):
        sample_recphyloxmls(model, family_index=math.inf, num_samples=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="seed"):
        export_backtracking_input(model, seed=-1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="seed"):
        export_backtracking_input(model, seed=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="seed"):
        export_backtracking_input(model, seed=1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="max_events"):
        sample_recphyloxml(model, max_events=0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="max_events"):
        sample_recphyloxml(model, max_events="10")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="seed"):
        sample_recphyloxmls(model, num_samples=1, seed=-1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="seed"):
        sample_recphyloxmls(model, num_samples=1, seed=1 << 64)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="seed range exceeds u64"):
        sample_recphyloxmls(model, num_samples=2, seed=(1 << 64) - 1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="num_samples"):
        sample_recphyloxmls(model, num_samples=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="num_samples"):
        sample_recphyloxmls(model, num_samples=1.5)  # type: ignore[arg-type]


def test_export_backtracking_input_rejects_nonfinite_payload_tensors(
    tmp_path: Path,
    monkeypatch,
):
    species_tree = tmp_path / "sp.nwk"
    species_tree.write_text("(s0,s1);", encoding="utf-8")
    family = FamilyInput(
        index=0,
        name="fam0",
        gene_tree_paths=["g0.nwk"],
        leaf_species_map={},
        clade_count=2,
        split_count=0,
        root_clade_id=0,
        ccp_helpers={
            "N_splits": 0,
            "split_parents_sorted": torch.empty(0, dtype=torch.long),
            "split_leftrights_sorted": torch.empty(0, dtype=torch.long),
            "log_split_probs_sorted": torch.empty(0, dtype=torch.float64),
        },
        leaf_row_index=torch.tensor([0, 1], dtype=torch.long),
        leaf_col_index=torch.tensor([0, 1], dtype=torch.long),
        clade_leaf_labels=["a", "b"],
    )
    model = SimpleNamespace(
        n_species=2,
        species_names=["s0", "s1"],
        species_tree_path=species_tree,
    )
    model.family_input = lambda family_index: family
    model.activate_family = lambda family_index: SimpleNamespace(
        clade_offset=0,
        local_family_index=0,
    )
    state = ReconciliationState(
        e=torch.zeros(2, dtype=torch.float64),
        pi=torch.tensor([[math.nan, 0.0], [0.0, 0.0]], dtype=torch.float64),
        log_p_s=torch.zeros(2, dtype=torch.float64),
        log_p_d=torch.zeros(2, dtype=torch.float64),
        log_p_l=torch.zeros(2, dtype=torch.float64),
        max_transfer=torch.zeros(2, dtype=torch.float64),
        origination_probs=None,
    )

    monkeypatch.setattr(backtracking, "_evaluate_backtracking_state", lambda _: state)
    with pytest.raises(ValueError, match=r"backtracking payload\.pi\.data\[0\]"):
        export_backtracking_input(model, family_index=0)  # type: ignore[arg-type]


def test_export_backtracking_input_uses_genewise_parameter_row_when_families_equal_species(
    tmp_path: Path,
    monkeypatch,
):
    species_tree = tmp_path / "sp.nwk"
    species_tree.write_text("(s0,s1);", encoding="utf-8")
    family = FamilyInput(
        index=1,
        name="fam1",
        gene_tree_paths=["g1.nwk"],
        leaf_species_map={},
        clade_count=2,
        split_count=0,
        root_clade_id=0,
        ccp_helpers={
            "N_splits": 0,
            "split_parents_sorted": torch.empty(0, dtype=torch.long),
            "split_leftrights_sorted": torch.empty(0, dtype=torch.long),
            "log_split_probs_sorted": torch.empty(0, dtype=torch.float64),
        },
        leaf_row_index=torch.tensor([0, 1], dtype=torch.long),
        leaf_col_index=torch.tensor([0, 1], dtype=torch.long),
        clade_leaf_labels=["a", "b"],
    )
    model = SimpleNamespace(
        mode="genewise",
        n_species=2,
        species_names=["s0", "s1"],
        species_tree_path=species_tree,
    )
    model.family_input = lambda family_index: family
    model.activate_family = lambda family_index: SimpleNamespace(
        clade_offset=0,
        local_family_index=1,
    )
    state = ReconciliationState(
        e=torch.tensor([[-1.0, -1.1], [-2.0, -2.1]], dtype=torch.float64),
        pi=torch.zeros((2, 2), dtype=torch.float64),
        log_p_s=torch.tensor([-10.0, -20.0], dtype=torch.float64),
        log_p_d=torch.tensor([-30.0, -40.0], dtype=torch.float64),
        log_p_l=torch.tensor([-50.0, -60.0], dtype=torch.float64),
        max_transfer=torch.tensor([-70.0, -80.0], dtype=torch.float64),
        origination_probs=None,
    )

    monkeypatch.setattr(backtracking, "_evaluate_backtracking_state", lambda _: state)

    payload = export_backtracking_input(model, family_index=1)  # type: ignore[arg-type]

    assert payload["e"] == [-2.0, -2.1]
    assert payload["log_p_s"] == [-20.0, -20.0]
    assert payload["log_p_d"] == [-40.0, -40.0]
    assert payload["max_transfer"] == [-80.0, -80.0]


def test_backtracking_payload_writer_rejects_nonfinite_json_before_subprocess(
    tmp_path: Path,
    monkeypatch,
):
    calls: list[object] = []

    def fake_run(*args: object, **kwargs: object) -> None:
        calls.append((args, kwargs))

    monkeypatch.setattr(backtracking.subprocess, "run", fake_run)

    with pytest.raises(ValueError, match=r"backtracking payload\.value"):
        backtracking._run_backtracking_payload(
            {"value": math.inf},
            cargo_manifest=tmp_path / "missing" / "Cargo.toml",
            backtrack_binary=tmp_path / "fake-backtracker",
            build_args=lambda *_: [],
            read_output=lambda _: None,
        )

    assert calls == []


def test_backtracking_env_binary_relative_path_uses_caller_cwd(
    tmp_path: Path,
    monkeypatch,
):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_backtracker = bin_dir / "fake-backtrack"
    fake_backtracker.write_text(
        """#!/usr/bin/env python3
import os
import pathlib
import sys

pathlib.Path(sys.argv[2]).write_text(os.getcwd(), encoding="utf-8")
""",
        encoding="utf-8",
    )
    fake_backtracker.chmod(0o755)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GPUREC_BACKTRACK_BIN", "bin/fake-backtrack")

    result = backtracking._run_backtracking_payload(
        {"value": 1},
        cargo_manifest=tmp_path / "missing" / "Cargo.toml",
        backtrack_binary=None,
        build_args=lambda input_path, output_dir: [
            str(input_path),
            str(output_dir / "done.txt"),
        ],
        read_output=lambda output_dir: (output_dir / "done.txt").read_text(
            encoding="utf-8"
        ),
    )

    assert result == str(tmp_path)


def test_backtracking_cargo_fallback_runs_from_source_root(
    tmp_path: Path,
    monkeypatch,
):
    captured: dict[str, object] = {}
    cargo_command = [
        "cargo",
        "run",
        "--locked",
        "--quiet",
        "--manifest-path",
        str(tmp_path / "Cargo.toml"),
        "--",
    ]

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        captured["command"] = command
        captured["cwd"] = kwargs["cwd"]
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(backtracking, "_backtrack_command", lambda **_: cargo_command)
    monkeypatch.setattr(backtracking.subprocess, "run", fake_run)

    result = backtracking._run_backtracking_payload(
        {"value": 1},
        cargo_manifest=tmp_path / "Cargo.toml",
        backtrack_binary=None,
        build_args=lambda *_: [],
        read_output=lambda _: "ok",
    )

    assert result == "ok"
    assert captured["command"] == cargo_command
    assert captured["cwd"] == str(backtracking._REPO_ROOT)


def test_backtracking_command_rejects_invalid_binary_before_subprocess(
    tmp_path: Path,
    monkeypatch,
):
    calls: list[object] = []

    def fake_run(*args: object, **kwargs: object) -> None:
        calls.append((args, kwargs))

    monkeypatch.setattr(backtracking.subprocess, "run", fake_run)

    missing = tmp_path / "missing-backtracker"
    with pytest.raises(
        RuntimeError,
        match="does not exist or is not a file",
    ) as missing_exc:
        backtracking._run_backtracking_payload(
            {"value": 1},
            cargo_manifest=tmp_path / "missing" / "Cargo.toml",
            backtrack_binary=missing,
            build_args=lambda *_: [],
            read_output=lambda _: None,
        )
    assert "backtrack_binary" in str(missing_exc.value)
    assert "--backtrack-binary" in str(missing_exc.value)
    assert str(missing) in str(missing_exc.value)

    not_executable = tmp_path / "not-executable"
    not_executable.write_text("#!/bin/sh\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="is not executable") as executable_exc:
        _backtrack_command(
            cargo_manifest=tmp_path / "missing" / "Cargo.toml",
            backtrack_binary=not_executable,
        )
    assert "backtrack_binary" in str(executable_exc.value)
    assert "--backtrack-binary" in str(executable_exc.value)
    assert str(not_executable) in str(executable_exc.value)

    assert calls == []


def test_backtracking_command_rejects_missing_env_command(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("GPUREC_BACKTRACK_BIN", "gpurec-backtrack-definitely-missing")

    with pytest.raises(RuntimeError, match="was not found on PATH") as exc_info:
        _backtrack_command(
            cargo_manifest=tmp_path / "missing" / "Cargo.toml",
            backtrack_binary=None,
        )

    message = str(exc_info.value)
    assert "GPUREC_BACKTRACK_BIN" in message
    assert "gpurec-backtrack-definitely-missing" in message


def test_ensure_backtracking_available_delegates_command_resolution(
    tmp_path: Path,
    monkeypatch,
):
    calls: list[tuple[Path, Path]] = []
    manifest = tmp_path / "Cargo.toml"
    binary = tmp_path / "gpurec-backtrack"

    def fake_backtrack_command(
        *,
        cargo_manifest: str | Path,
        backtrack_binary: str | Path | None,
    ) -> list[str]:
        assert backtrack_binary is not None
        calls.append((Path(cargo_manifest), Path(backtrack_binary)))
        return [str(binary)]

    monkeypatch.setattr(backtracking, "_backtrack_command", fake_backtrack_command)

    backtracking.ensure_backtracking_available(binary, cargo_manifest=manifest)

    assert calls == [(manifest, binary)]


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


def test_backtracking_runner_reports_subprocess_failure_with_stderr(
    tmp_path: Path,
    monkeypatch,
):
    fake_backtracker = tmp_path / "fail_backtracker.py"
    fake_backtracker.write_text(
        """import sys
sys.stdout.write("rust stdout\\n")
sys.stderr.write("rust stderr\\n")
sys.exit(7)
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        backtracking,
        "_backtrack_command",
        lambda **_: [sys.executable, str(fake_backtracker)],
    )

    with pytest.raises(RuntimeError, match="exit code 7") as exc_info:
        backtracking._run_backtracking_payload(
            {"value": 1},
            cargo_manifest=tmp_path / "missing" / "Cargo.toml",
            backtrack_binary=None,
            build_args=lambda *_: [],
            read_output=lambda _: None,
        )

    message = str(exc_info.value)
    assert "gpurec backtracking command failed" in message
    assert "rust stderr" in message
    assert "rust stdout" in message
    assert "Traceback" not in message


def test_backtracking_runner_reports_missing_expected_outputs(
    tmp_path: Path,
    monkeypatch,
):
    fake_backtracker = tmp_path / "partial_backtracker.py"
    fake_backtracker.write_text(
        """import pathlib
import sys

args = sys.argv[1:]
if args and args[0] == "--samples":
    output_dir = pathlib.Path(args[5])
    output_dir.mkdir()
    (output_dir / "sample_0.xml").write_text("<sample index='0'/>", encoding="utf-8")
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        backtracking,
        "_backtrack_command",
        lambda **_: [sys.executable, str(fake_backtracker)],
    )
    monkeypatch.setattr(
        backtracking,
        "export_backtracking_input",
        lambda *_, **__: {"family_index": 0},
    )

    with pytest.raises(RuntimeError, match="single-sample RecPhyloXML output") as single:
        sample_recphyloxml(object())
    assert "sample.xml" in str(single.value)

    with pytest.raises(RuntimeError, match="1 of 2 expected RecPhyloXML outputs") as multi:
        sample_recphyloxmls(object(), num_samples=2)
    assert "sample_1.xml" in str(multi.value)


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


def test_event_counts_table_quotes_tabbed_family_names(tmp_path: Path):
    path = tmp_path / "event_counts.tsv"
    row = {"family": "fam\tbad", "sample": 0, **{key: 0 for key in EVENT_KEYS}}
    sampling_workflow._write_event_counts_table(path, [row])

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle, delimiter="\t"))

    assert rows[0] == ["family", "sample", *EVENT_KEYS]
    assert rows[1] == ["fam\tbad", "0", *["0" for _ in EVENT_KEYS]]
    assert len(rows[1]) == 2 + len(EVENT_KEYS)


def test_sampling_runner_preflights_backtracking_before_loading_model(
    tmp_path: Path,
    monkeypatch,
):
    config = SamplingConfig(
        checkpoint=tmp_path / "checkpoints" / "best.pt",
        backtrack_binary=tmp_path / "missing-backtrack",
    )
    runner = SamplingRunner(config)
    load_calls: list[str] = []

    def fail_preflight(backtrack_binary: Path | None) -> None:
        assert backtrack_binary == config.backtrack_binary
        raise RuntimeError("missing backtracking binary")

    def unexpected_load_model() -> tuple[RunConfig, object]:
        load_calls.append("load_model")
        raise AssertionError("model loading should not run before preflight")

    monkeypatch.setattr(
        sampling_workflow,
        "ensure_backtracking_available",
        fail_preflight,
    )
    monkeypatch.setattr(runner, "_load_model", unexpected_load_model)

    with pytest.raises(RuntimeError, match="missing backtracking binary"):
        runner.run()

    assert load_calls == []


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
    preflight_calls: list[Path | None] = []
    monkeypatch.setattr(
        sampling_workflow,
        "ensure_backtracking_available",
        lambda backtrack_binary: preflight_calls.append(backtrack_binary),
    )
    monkeypatch.setattr(runner, "_load_model", lambda: (run_config, model))
    monkeypatch.setattr(
        sampling_workflow,
        "sample_recphyloxmls",
        fake_sample_recphyloxmls,
    )
    recon_dir = config.out_dir / "reconciliations"
    all_dir = recon_dir / "all"
    all_dir.mkdir(parents=True)
    (all_dir / "000001_old_sample_99.xml").write_text("stale", encoding="utf-8")
    (all_dir / "000001_old_eventCounts_99.txt").write_text(
        "stale",
        encoding="utf-8",
    )
    (all_dir / "manual.keep").write_text("keep", encoding="utf-8")
    for stale_name in (
        "event_counts.tsv",
        "summary.json",
        "totalSpeciesEventCounts.txt",
        "totalTransfers.txt",
    ):
        (recon_dir / stale_name).write_text("stale", encoding="utf-8")

    result = runner.run()

    assert result.out_dir == config.out_dir
    assert result.families_sampled == 2
    assert result.samples_per_family == 2
    assert result.xml_files == 4
    assert model.closed
    assert preflight_calls == [config.backtrack_binary]
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

    assert sorted(path.name for path in all_dir.glob("*.xml")) == [
        "000001_fam_a_sample_0.xml",
        "000001_fam_a_sample_1.xml",
        "000002_fam2_sample_0.xml",
        "000002_fam2_sample_1.xml",
    ]
    assert sorted(path.name for path in all_dir.glob("*_eventCounts_*.txt")) == [
        "000001_fam_a_eventCounts_0.txt",
        "000001_fam_a_eventCounts_1.txt",
        "000002_fam2_eventCounts_0.txt",
        "000002_fam2_eventCounts_1.txt",
    ]
    assert (all_dir / "manual.keep").read_text(encoding="utf-8") == "keep"
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


def test_sampling_runner_rejects_checkpoint_family_order_mismatch(
    tmp_path: Path,
    monkeypatch,
):
    class FakeModel:
        def __init__(self):
            self.theta = torch.nn.Parameter(torch.zeros(2, 3, dtype=torch.float32))
            self.family_names = ["fam_b", "fam_a"]
            self.species_names = ["sp0", "sp1"]
            self.closed = False

        def clear(self):
            raise AssertionError("theta should not be restored before identity check")

        def close(self):
            self.closed = True

    model = FakeModel()
    run_config = RunConfig(
        species_tree=tmp_path / "sp.nwk",
        families_file=tmp_path / "families.txt",
        out_dir=tmp_path / "run_out",
        mode="genewise",
        device="cpu",
    )
    config = SamplingConfig(checkpoint=tmp_path / "checkpoints" / "best.pt")

    def fake_load_checkpoint(path, *, map_location):
        return {
            "theta": torch.zeros(2, 3, dtype=torch.float32),
            "config": run_config.to_dict(),
            "family_names": ["fam_a", "fam_b"],
            "species_names": ["sp0", "sp1"],
        }

    monkeypatch.setattr(sampling_workflow, "load_checkpoint", fake_load_checkpoint)
    monkeypatch.setattr(
        sampling_workflow,
        "build_alerax_workflow_model",
        lambda *_args, **_kwargs: model,
    )

    runner = SamplingRunner(config)
    with pytest.raises(RuntimeError, match="family_names differ"):
        runner._load_model()

    assert model.closed


def test_sampling_runner_preserves_load_error_when_close_fails(
    tmp_path: Path,
    monkeypatch,
):
    class FakeModel:
        def __init__(self):
            self.close_calls = 0

        def close(self):
            self.close_calls += 1
            raise RuntimeError("close failed")

    model = FakeModel()
    run_config = RunConfig(
        species_tree=tmp_path / "sp.nwk",
        families_file=tmp_path / "families.txt",
        out_dir=tmp_path / "run_out",
        device="cpu",
    )
    config = SamplingConfig(checkpoint=tmp_path / "checkpoints" / "best.pt")

    monkeypatch.setattr(
        sampling_workflow,
        "load_checkpoint",
        lambda path, *, map_location: {"config": run_config.to_dict()},
    )
    monkeypatch.setattr(
        sampling_workflow,
        "build_alerax_workflow_model",
        lambda *_args, **_kwargs: model,
    )

    def fail_validation(**_kwargs):
        raise RuntimeError("family_names differ")

    monkeypatch.setattr(
        sampling_workflow,
        "validate_checkpoint_model_compatibility",
        fail_validation,
    )
    monkeypatch.setattr(
        sampling_workflow,
        "restore_model_theta",
        lambda *_args, **_kwargs: pytest.fail("restore should not run"),
    )

    runner = SamplingRunner(config)
    with pytest.raises(RuntimeError, match="family_names differ") as excinfo:
        runner._load_model()

    assert model.close_calls == 1
    assert isinstance(excinfo.value.__cause__, RuntimeError)
    assert str(excinfo.value.__cause__) == "close failed"


def test_sampling_runner_preserves_sampling_error_when_close_fails(
    tmp_path: Path,
    monkeypatch,
):
    class FakeModel:
        family_names = ["fam0"]

        def __init__(self):
            self.close_calls = 0

        def close(self):
            self.close_calls += 1
            raise RuntimeError("close failed")

    model = FakeModel()
    run_config = RunConfig(
        species_tree=tmp_path / "sp.nwk",
        families_file=tmp_path / "families.txt",
        out_dir=tmp_path / "run_out",
        device="cpu",
    )
    config = SamplingConfig(
        checkpoint=tmp_path / "checkpoints" / "best.pt",
        out_dir=tmp_path / "sample_out",
    )

    runner = SamplingRunner(config)
    monkeypatch.setattr(
        sampling_workflow,
        "ensure_backtracking_available",
        lambda backtrack_binary: None,
    )
    monkeypatch.setattr(runner, "_load_model", lambda: (run_config, model))

    def fail_sample_recphyloxmls(*_args: object, **_kwargs: object) -> list[str]:
        raise RuntimeError("backtrack failed")

    monkeypatch.setattr(
        sampling_workflow,
        "sample_recphyloxmls",
        fail_sample_recphyloxmls,
    )

    with pytest.raises(RuntimeError, match="backtrack failed") as excinfo:
        runner.run()

    assert model.close_calls == 1
    assert isinstance(excinfo.value.__cause__, RuntimeError)
    assert str(excinfo.value.__cause__) == "close failed"


def test_sampling_runner_closes_model_on_empty_family_selection(
    tmp_path: Path,
    monkeypatch,
):
    class FakeModel:
        family_names = ["fam0"]

        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    model = FakeModel()
    run_config = RunConfig(
        species_tree=tmp_path / "sp.nwk",
        families_file=tmp_path / "families.txt",
        out_dir=tmp_path / "run_out",
        device="cpu",
    )
    config = SamplingConfig(
        checkpoint=tmp_path / "checkpoints" / "best.pt",
        out_dir=tmp_path / "sample_out",
        family_start=1,
    )
    recon_dir = config.out_dir / "reconciliations"
    all_dir = recon_dir / "all"
    all_dir.mkdir(parents=True)
    sample_path = all_dir / "000000_previous_sample_0.xml"
    event_path = all_dir / "000000_previous_eventCounts_0.txt"
    aggregate_path = recon_dir / "summary.json"
    sample_path.write_text("previous sample", encoding="utf-8")
    event_path.write_text("previous counts", encoding="utf-8")
    aggregate_path.write_text("previous summary", encoding="utf-8")

    runner = SamplingRunner(config)
    monkeypatch.setattr(
        sampling_workflow,
        "ensure_backtracking_available",
        lambda backtrack_binary: None,
    )
    monkeypatch.setattr(runner, "_load_model", lambda: (run_config, model))

    with pytest.raises(ValueError, match="empty sampling family selection"):
        runner.run()

    assert model.closed
    assert sample_path.read_text(encoding="utf-8") == "previous sample"
    assert event_path.read_text(encoding="utf-8") == "previous counts"
    assert aggregate_path.read_text(encoding="utf-8") == "previous summary"


def test_sampling_runner_rejects_seed_range_overflow_before_outputs(
    tmp_path: Path,
    monkeypatch,
):
    class FakeModel:
        family_names = ["fam0", "fam1"]

        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    model = FakeModel()
    run_config = RunConfig(
        species_tree=tmp_path / "sp.nwk",
        families_file=tmp_path / "families.txt",
        out_dir=tmp_path / "run_out",
        device="cpu",
    )
    config = SamplingConfig(
        checkpoint=tmp_path / "checkpoints" / "best.pt",
        out_dir=tmp_path / "sample_out",
        samples=2,
        seed=(1 << 64) - 2,
        family_start=1,
    )
    recon_dir = config.out_dir / "reconciliations"
    all_dir = recon_dir / "all"
    all_dir.mkdir(parents=True)
    sample_path = all_dir / "000001_previous_sample_0.xml"
    aggregate_path = recon_dir / "summary.json"
    sample_path.write_text("previous sample", encoding="utf-8")
    aggregate_path.write_text("previous summary", encoding="utf-8")

    runner = SamplingRunner(config)
    monkeypatch.setattr(
        sampling_workflow,
        "ensure_backtracking_available",
        lambda backtrack_binary: None,
    )
    monkeypatch.setattr(runner, "_load_model", lambda: (run_config, model))

    def unexpected_sample_recphyloxmls(*_args: object, **_kwargs: object) -> list[str]:
        raise AssertionError("sampling should not run")

    monkeypatch.setattr(
        sampling_workflow,
        "sample_recphyloxmls",
        unexpected_sample_recphyloxmls,
    )

    with pytest.raises(ValueError, match="sampling seed range exceeds u64"):
        runner.run()

    assert model.closed
    assert sample_path.read_text(encoding="utf-8") == "previous sample"
    assert aggregate_path.read_text(encoding="utf-8") == "previous summary"


class _DummyModel:
    def __init__(self):
        self.theta = torch.nn.Parameter(torch.zeros(2, 3))
        self.family_names = ["a", "b"]
        self.species_names = ["s0", "s1"]
        self.cleared = False

    def clear(self):
        self.cleared = True


def test_checkpoint_roundtrip_restores_theta_and_status(tmp_path: Path):
    config = RunConfig(
        species_tree=tmp_path / "sp.nwk",
        families_file=tmp_path / "families.txt",
        out_dir=tmp_path / "out",
        device="cpu",
        dtype="torch.float64",
    )
    model = _DummyModel()
    with torch.no_grad():
        model.theta.fill_(2.0)
    optimizer = torch.optim.Adam([model.theta], lr=0.1)
    loss = model.theta.square().sum()
    loss.backward()
    optimizer.step()
    expected_theta = model.theta.detach().clone()

    path = tmp_path / "latest.pt"
    save_checkpoint(
        path,
        config=config,
        model=model,
        optimizer=optimizer,
        optimizer_phase="adam",
        step=4,
        status={"status": "running", "best_nll_bits": 12.0},
    )
    payload = load_checkpoint(path)

    with torch.no_grad():
        model.theta.zero_()
    restore_model_theta(model, payload)

    assert int(payload["step"]) == 4
    assert int(payload["next_step"]) == 5
    assert payload["config"]["dtype"] == "float64"
    assert RunConfig.from_dict(payload["config"]).dtype == "float64"
    assert payload["optimizer_phase"] == "adam"
    assert payload["status"]["best_nll_bits"] == 12.0
    assert isinstance(payload["optimizer_state"], dict)
    assert payload["optimizer_state"]["state"]
    assert payload["family_names"] == ["a", "b"]
    assert payload["species_names"] == ["s0", "s1"]
    assert torch.equal(model.theta, expected_theta)
    assert model.cleared

    final_path = tmp_path / "final.pt"
    save_checkpoint(
        final_path,
        config=config,
        model=model,
        optimizer=optimizer,
        optimizer_phase="adam",
        step=4,
        next_step=4,
        status={"status": "not_converged"},
    )
    assert int(load_checkpoint(final_path)["next_step"]) == 4


def test_checkpoint_load_uses_weights_only(tmp_path: Path, monkeypatch):
    calls: list[dict[str, object]] = []

    def fake_load(path, *, map_location, weights_only):
        calls.append(
            {
                "path": path,
                "map_location": map_location,
                "weights_only": weights_only,
            }
        )
        return {
            "version": CHECKPOINT_VERSION,
            "config": {
                "species_tree": str(tmp_path / "sp.nwk"),
                "families_file": str(tmp_path / "families.txt"),
                "out_dir": str(tmp_path / "out"),
            },
            "theta": torch.zeros(3),
            "optimizer_state": None,
            "status": {},
            "family_names": [],
        }

    monkeypatch.setattr("gpurec.workflow.checkpoint.torch.load", fake_load)

    payload = load_checkpoint(tmp_path / "checkpoint.pt", map_location="cpu")

    assert payload["version"] == CHECKPOINT_VERSION
    assert calls == [
        {
            "path": tmp_path / "checkpoint.pt",
            "map_location": "cpu",
            "weights_only": True,
        }
    ]


@pytest.mark.parametrize("version", [CHECKPOINT_VERSION + 1, "next"])
def test_checkpoint_load_rejects_unsupported_version(tmp_path: Path, version):
    path = tmp_path / "future.pt"
    torch.save(
        {
            "version": version,
            "config": {
                "species_tree": str(tmp_path / "sp.nwk"),
                "families_file": str(tmp_path / "families.txt"),
                "out_dir": str(tmp_path / "out"),
            },
            "theta": torch.zeros(3),
        },
        path,
    )

    with pytest.raises(RuntimeError, match="unsupported version"):
        load_checkpoint(path)


@pytest.mark.parametrize(
    ("theta", "message"),
    [
        (torch.tensor([float("nan")]), "nonfinite theta"),
        (torch.tensor([float("inf")]), "nonfinite theta"),
        (torch.tensor([1], dtype=torch.int64), "theta tensor dtype"),
    ],
)
def test_checkpoint_load_rejects_invalid_theta_values(
    tmp_path: Path,
    theta: torch.Tensor,
    message: str,
):
    path = tmp_path / "invalid_theta.pt"
    torch.save(
        {
            "version": CHECKPOINT_VERSION,
            "config": {
                "species_tree": str(tmp_path / "sp.nwk"),
                "families_file": str(tmp_path / "families.txt"),
                "out_dir": str(tmp_path / "out"),
            },
            "theta": theta,
        },
        path,
    )

    with pytest.raises(RuntimeError, match=message):
        load_checkpoint(path)


def test_checkpoint_load_rejects_raw_theta_export(tmp_path: Path):
    path = tmp_path / "theta_final.pt"
    torch.save(torch.zeros(2, 3), path)

    with pytest.raises(RuntimeError, match="must contain a dictionary payload"):
        load_checkpoint(path)


def test_optimization_runner_run_writes_outputs_with_fake_model(tmp_path: Path):
    class FakeOptimizationModel:
        def __init__(self):
            self.theta = torch.nn.Parameter(
                torch.tensor(
                    [
                        [0.25, -0.15, 0.05],
                        [0.10, 0.20, -0.05],
                    ],
                    dtype=torch.float32,
                )
            )
            self.family_names = ["fam0", "fam1"]
            self.species_names = ["sp0", "sp1", "sp2"]
            self.n_families = 2
            self.n_species = 3
            self.batch_metadata = [SimpleNamespace(batch_index=0)]
            self.clears = 0
            self.closed = False

        def full_loss(self):
            return self.theta.square().sum() + 5.0

        def full_nll_per_family(self):
            return self.theta.detach().square().sum(dim=1) + 2.0

        def clamp_theta_(self, min_rate, max_rate):
            with torch.no_grad():
                self.theta.clamp_(min=-4.0, max=4.0)

        def solver_stat_records(self):
            return [
                {
                    "E_iterations": 2,
                    "Pi_max_iterations": 4,
                    "Pi_wave_iterations": [1, 2],
                    "Pi_wave_count": 2,
                    "Pi_converged_waves": 2,
                    "Neumann_terms": 3,
                    "Gradient_converged": True,
                }
            ]

        def clear(self):
            self.clears += 1

        def close(self):
            self.closed = True

    class FakeRunner(OptimizationRunner):
        def __init__(self, config):
            super().__init__(config)
            self.saved_checkpoint_losses = []

        def build_model(self):
            self.fake_model = FakeOptimizationModel()
            return self.fake_model

        def _save_status(
            self,
            path,
            *,
            model,
            optimizer,
            step,
            status,
            row,
            next_step=None,
            optimizer_phase=None,
        ):
            super()._save_status(
                path,
                model=model,
                optimizer=optimizer,
                step=step,
                next_step=next_step,
                status=status,
                row=row,
                optimizer_phase=optimizer_phase,
            )
            if row is not None and "likelihood/data_nll_bits" in row:
                expected_loss = float((model.theta.detach().square().sum() + 5.0).cpu())
                self.saved_checkpoint_losses.append(
                    (
                        Path(path).name,
                        expected_loss,
                        float(row["likelihood/data_nll_bits"]),
                    )
                )

    config = RunConfig(
        species_tree=tmp_path / "sp.nwk",
        families_file=tmp_path / "families.txt",
        out_dir=tmp_path / "out",
        mode="genewise",
        device="cpu",
        optimizer="adam",
        steps=1,
        lr=0.05,
        checkpoint_every=1,
        log_every=10,
        grad_inf_tol=0.0,
        loss_patience=0,
        best_likelihood_patience=0,
    )
    runner = FakeRunner(config)

    result = runner.run()

    assert result.status == "not_converged"
    assert result.reason == "max_steps"
    assert result.steps_completed == 1
    assert result.best_step == 0
    assert result.out_dir == config.out_dir
    assert result.sampling_checkpoint == config.out_dir / "checkpoints" / "best.pt"
    assert runner.fake_model.closed
    assert runner.fake_model.clears >= 1
    assert runner.saved_checkpoint_losses
    for checkpoint_name, expected_loss, row_loss in runner.saved_checkpoint_losses:
        assert row_loss == pytest.approx(expected_loss), checkpoint_name

    history_rows = [
        json.loads(line)
        for line in (config.out_dir / "history.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["optimizer/phase"] for row in history_rows] == ["adam", "final_eval"]
    assert history_rows[-1]["step"] == 1
    assert history_rows[-1]["best_step"] == 0

    summary = json.loads((config.out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "not_converged"
    assert summary["families"] == 2
    assert summary["species"] == 3
    assert summary["batches"] == 1
    assert summary["final_nll_bits"] == pytest.approx(result.final_nll_bits)

    latest = load_checkpoint(config.out_dir / "checkpoints" / "latest.pt")
    best = load_checkpoint(config.out_dir / "checkpoints" / "best.pt")
    assert latest["status"]["status"] == "not_converged"
    assert best["status"]["best_step"] == 0
    assert latest["last_row"]["optimizer/phase"] == "final_eval"
    assert latest["family_names"] == ["fam0", "fam1"]

    assert (config.out_dir / "optimization_history.csv").exists()
    assert (config.out_dir / "theta_final.pt").exists()
    assert "fam0" in (config.out_dir / "rates_final.tsv").read_text(encoding="utf-8")
    per_family = (config.out_dir / "per_fam_likelihoods.tsv").read_text(
        encoding="utf-8"
    )
    assert "fam0" in per_family
    assert "fam1" in per_family


def test_optimization_runner_final_latest_resumes_at_next_optimizer_step(tmp_path: Path):
    class FakeResumeModel:
        def __init__(self):
            self.theta = torch.nn.Parameter(
                torch.tensor([0.25, -0.15, 0.05], dtype=torch.float32)
            )
            self.family_names = ["fam0"]
            self.n_families = 1
            self.n_species = 2
            self.batch_metadata = [SimpleNamespace(batch_index=0)]
            self.closed = False

        def full_loss(self):
            return self.theta.square().sum() + 3.0

        def clamp_theta_(self, min_rate, max_rate):
            with torch.no_grad():
                self.theta.clamp_(min=-4.0, max=4.0)

        def solver_stat_records(self):
            return []

        def clear(self):
            return None

        def close(self):
            self.closed = True

    class FakeResumeRunner(OptimizationRunner):
        def build_model(self):
            self.fake_model = FakeResumeModel()
            return self.fake_model

    species_tree = tmp_path / "sp.nwk"
    families_file = tmp_path / "families.txt"
    first_config = RunConfig(
        species_tree=species_tree,
        families_file=families_file,
        out_dir=tmp_path / "first",
        mode="global",
        device="cpu",
        optimizer="adam",
        steps=1,
        lr=0.05,
        checkpoint_every=0,
        log_every=10,
        grad_inf_tol=0.0,
    )

    first_runner = FakeResumeRunner(first_config)
    first_runner.run()
    first_latest_path = first_config.out_dir / "checkpoints" / "latest.pt"
    first_latest = load_checkpoint(first_latest_path)

    assert first_latest["last_row"]["optimizer/phase"] == "final_eval"
    assert int(first_latest["step"]) == 1
    assert int(first_latest["next_step"]) == 1

    resumed_config = RunConfig(
        species_tree=species_tree,
        families_file=families_file,
        out_dir=tmp_path / "resumed",
        mode="global",
        device="cpu",
        optimizer="adam",
        steps=2,
        lr=0.05,
        resume_from=first_latest_path,
        checkpoint_every=0,
        log_every=10,
        grad_inf_tol=0.0,
    )
    resumed_runner = FakeResumeRunner(resumed_config)

    resumed_result = resumed_runner.run()

    assert resumed_result.steps_completed == 2
    history_rows = [
        json.loads(line)
        for line in (resumed_config.out_dir / "history.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [(row["optimizer/phase"], row["step"]) for row in history_rows] == [
        ("adam", 1),
        ("final_eval", 2),
    ]
    resumed_latest = load_checkpoint(
        resumed_config.out_dir / "checkpoints" / "latest.pt"
    )
    assert int(resumed_latest["next_step"]) == 2


def test_optimization_runner_reports_latest_when_no_best_written_this_run(
    tmp_path: Path,
):
    class FakeResumeModel:
        def __init__(self):
            self.theta = torch.nn.Parameter(
                torch.tensor([0.25, -0.15, 0.05], dtype=torch.float32)
            )
            self.family_names = ["fam0"]
            self.species_names = ["sp0", "sp1"]
            self.n_families = 1
            self.n_species = 2
            self.batch_metadata = [SimpleNamespace(batch_index=0)]
            self.closed = False

        def full_loss(self):
            return self.theta.square().sum() + 3.0

        def clamp_theta_(self, min_rate, max_rate):
            with torch.no_grad():
                self.theta.clamp_(min=-4.0, max=4.0)

        def solver_stat_records(self):
            return []

        def clear(self):
            return None

        def close(self):
            self.closed = True

    class FakeResumeRunner(OptimizationRunner):
        def build_model(self):
            self.fake_model = FakeResumeModel()
            return self.fake_model

    species_tree = tmp_path / "sp.nwk"
    families_file = tmp_path / "families.txt"
    checkpoint_config = RunConfig(
        species_tree=species_tree,
        families_file=families_file,
        out_dir=tmp_path / "checkpoint-source",
        mode="global",
        device="cpu",
        optimizer="adam",
        steps=1,
        checkpoint_every=0,
    )
    resume_checkpoint = tmp_path / "resume.pt"
    save_checkpoint(
        resume_checkpoint,
        config=checkpoint_config,
        model=FakeResumeModel(),
        optimizer=None,
        step=0,
        next_step=1,
        status={
            "best_nll_bits": 0.0,
            "best_step": 0,
            "previous_objective": 0.0,
            "stable_loss_steps": 0,
        },
    )

    out_dir = tmp_path / "resumed"
    stale_best = out_dir / "checkpoints" / "best.pt"
    stale_best.parent.mkdir(parents=True)
    stale_best.write_bytes(b"stale best from previous invocation")
    config = RunConfig(
        species_tree=species_tree,
        families_file=families_file,
        out_dir=out_dir,
        mode="global",
        device="cpu",
        optimizer="adam",
        steps=1,
        resume_from=resume_checkpoint,
        checkpoint_every=0,
        log_every=10,
        grad_inf_tol=0.0,
    )
    runner = FakeResumeRunner(config)

    result = runner.run()

    latest = out_dir / "checkpoints" / "latest.pt"
    assert result.sampling_checkpoint == latest
    assert latest.is_file()
    assert stale_best.read_bytes() == b"stale best from previous invocation"


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

    mismatch_optimizer = FakeOptimizer()
    mismatch = runner._restore_optimizer_state(
        mismatch_optimizer,
        {"state": ["adam"]},
        current_phase="lbfgs",
        checkpoint_phase="adam",
    )
    assert mismatch == {
        "resume_optimizer_state": "discarded",
        "resume_optimizer_reason": "phase_mismatch",
        "resume_optimizer_checkpoint_phase": "adam",
        "resume_optimizer_current_phase": "lbfgs",
    }
    assert mismatch_optimizer.loaded is None

    discarded = runner._restore_optimizer_state(
        FakeOptimizer(fail=True),
        {"state": ["bad"]},
    )
    assert discarded["resume_optimizer_state"] == "discarded"
    assert "incompatible optimizer state" in discarded["resume_optimizer_error"]


def test_optimization_runner_resume_loads_checkpoint_once(tmp_path: Path, monkeypatch):
    class FakeResumeModel:
        def __init__(self):
            self.theta = torch.nn.Parameter(torch.zeros(3, dtype=torch.float32))
            self.family_names: list[str] = []
            self.n_families = 0
            self.n_species = 2
            self.batch_metadata = []
            self.closed = False

        def full_loss(self):
            return self.theta.square().sum() + 1.0

        def clamp_theta_(self, min_rate, max_rate):
            with torch.no_grad():
                self.theta.clamp_(min=-4.0, max=4.0)

        def solver_stat_records(self):
            return []

        def clear(self):
            return None

        def close(self):
            self.closed = True

    class FakeResumeRunner(OptimizationRunner):
        def build_model(self):
            self.fake_model = FakeResumeModel()
            return self.fake_model

    resume_path = tmp_path / "resume.pt"
    load_calls: list[tuple[Path, str]] = []

    def fake_load_checkpoint(path, *, map_location):
        load_calls.append((Path(path), str(map_location)))
        return {
            "theta": torch.tensor([0.25, -0.125, 0.0625], dtype=torch.float32),
            "optimizer_state": None,
            "next_step": 1,
            "status": {
                "previous_objective": 1.5,
                "stable_loss_steps": 0,
            },
        }

    workflow_optimize_module = importlib.import_module("gpurec.workflow.optimize")
    monkeypatch.setattr(workflow_optimize_module, "load_checkpoint", fake_load_checkpoint)
    config = RunConfig(
        species_tree=tmp_path / "sp.nwk",
        families_file=tmp_path / "families.txt",
        out_dir=tmp_path / "out",
        mode="global",
        device="cpu",
        optimizer="adam",
        steps=1,
        resume_from=resume_path,
        checkpoint_every=0,
        log_every=10,
    )
    runner = FakeResumeRunner(config)

    result = runner.run()

    assert load_calls == [(resume_path.resolve(), "cpu")]
    assert result.status == "not_converged"
    assert runner.fake_model.closed
    assert load_checkpoint(config.out_dir / "checkpoints" / "latest.pt")[
        "last_row"
    ]["resume_optimizer_state"] == "missing"


def test_optimization_runner_discards_resume_optimizer_state_on_phase_mismatch(
    tmp_path: Path,
    monkeypatch,
):
    class FakeResumeModel:
        def __init__(self):
            self.theta = torch.nn.Parameter(torch.zeros(3, dtype=torch.float32))
            self.family_names: list[str] = []
            self.n_families = 0
            self.n_species = 2
            self.batch_metadata = []
            self.closed = False

        def full_loss(self):
            return self.theta.square().sum() + 1.0

        def clamp_theta_(self, min_rate, max_rate):
            with torch.no_grad():
                self.theta.clamp_(min=-4.0, max=4.0)

        def solver_stat_records(self):
            return []

        def clear(self):
            return None

        def close(self):
            self.closed = True

    class FakeResumeRunner(OptimizationRunner):
        def build_model(self):
            self.fake_model = FakeResumeModel()
            return self.fake_model

    theta = torch.nn.Parameter(torch.zeros(3, dtype=torch.float32))
    adam = torch.optim.Adam([theta], lr=0.01)
    theta.square().sum().backward()
    adam.step()
    adam_state = adam.state_dict()
    resume_path = tmp_path / "resume.pt"

    def fake_load_checkpoint(path, *, map_location):
        return {
            "theta": torch.tensor([0.25, -0.125, 0.0625], dtype=torch.float32),
            "optimizer_state": adam_state,
            "optimizer_phase": "adam",
            "next_step": 1,
            "status": {
                "previous_objective": 1.5,
                "stable_loss_steps": 0,
            },
        }

    workflow_optimize_module = importlib.import_module("gpurec.workflow.optimize")
    monkeypatch.setattr(workflow_optimize_module, "load_checkpoint", fake_load_checkpoint)
    config = RunConfig(
        species_tree=tmp_path / "sp.nwk",
        families_file=tmp_path / "families.txt",
        out_dir=tmp_path / "out",
        mode="global",
        device="cpu",
        optimizer="adam-lbfgs",
        adam_warmup_steps=1,
        steps=1,
        lbfgs_lr=0.5,
        resume_from=resume_path,
        checkpoint_every=0,
        log_every=10,
    )
    runner = FakeResumeRunner(config)

    runner.run()

    row = load_checkpoint(config.out_dir / "checkpoints" / "latest.pt")["last_row"]
    assert row["resume_optimizer_state"] == "discarded"
    assert row["resume_optimizer_reason"] == "phase_mismatch"
    assert row["resume_optimizer_checkpoint_phase"] == "adam"
    assert row["resume_optimizer_current_phase"] == "lbfgs"


@pytest.mark.parametrize(
    ("payload_update", "message"),
    [
        ({"family_names": ["fam_a", "fam_b"]}, "family_names differ"),
        ({"species_names": ["sp_a", "sp_b"]}, "species_names differ"),
        ({"config": {"mode": "genewise"}}, r"config\.mode differs"),
    ],
)
def test_optimization_runner_resume_rejects_incompatible_checkpoint_identity(
    tmp_path: Path,
    monkeypatch,
    payload_update: dict[str, object],
    message: str,
):
    class FakeResumeModel:
        def __init__(self):
            self.theta = torch.nn.Parameter(torch.zeros(2, 3, dtype=torch.float32))
            self.family_names = ["fam_b", "fam_a"]
            self.species_names = ["sp_b", "sp_a"]
            self.closed = False

        def clear(self):
            raise AssertionError("theta should not be restored before identity check")

        def close(self):
            self.closed = True

    class FakeResumeRunner(OptimizationRunner):
        def build_model(self):
            self.fake_model = FakeResumeModel()
            return self.fake_model

    config = RunConfig(
        species_tree=tmp_path / "sp.nwk",
        families_file=tmp_path / "families.txt",
        out_dir=tmp_path / "out",
        mode="global",
        device="cpu",
        optimizer="adam",
        steps=1,
        resume_from=tmp_path / "resume.pt",
        checkpoint_every=0,
        log_every=10,
    )

    def fake_load_checkpoint(path, *, map_location):
        payload = {
            "theta": torch.zeros(2, 3, dtype=torch.float32),
            "optimizer_state": None,
            "next_step": 1,
            "config": config.to_dict(),
            "family_names": ["fam_b", "fam_a"],
            "species_names": ["sp_b", "sp_a"],
            "status": {
                "previous_objective": 1.5,
                "stable_loss_steps": 0,
            },
        }
        if "config" in payload_update:
            config_update = payload_update["config"]
            assert isinstance(config_update, dict)
            payload["config"] = {**payload["config"], **config_update}
        if "family_names" in payload_update:
            payload["family_names"] = payload_update["family_names"]
        if "species_names" in payload_update:
            payload["species_names"] = payload_update["species_names"]
        return payload

    workflow_optimize_module = importlib.import_module("gpurec.workflow.optimize")
    monkeypatch.setattr(workflow_optimize_module, "load_checkpoint", fake_load_checkpoint)
    runner = FakeResumeRunner(config)

    with pytest.raises(RuntimeError, match=message):
        runner.run()

    assert runner.fake_model.closed


def test_optimization_runner_resume_rejects_checkpoint_beyond_configured_steps(
    tmp_path: Path,
    monkeypatch,
):
    class FakeResumeModel:
        def __init__(self):
            self.theta = torch.nn.Parameter(torch.zeros(3, dtype=torch.float32))
            self.family_names = ["fam0"]
            self.species_names = ["sp0", "sp1"]
            self.closed = False

        def clear(self):
            raise AssertionError("theta should not be restored from invalid resume")

        def close(self):
            self.closed = True

    class FakeResumeRunner(OptimizationRunner):
        def build_model(self):
            self.fake_model = FakeResumeModel()
            return self.fake_model

    config = RunConfig(
        species_tree=tmp_path / "sp.nwk",
        families_file=tmp_path / "families.txt",
        out_dir=tmp_path / "out",
        mode="global",
        device="cpu",
        optimizer="adam",
        steps=2,
        resume_from=tmp_path / "resume.pt",
        checkpoint_every=0,
        log_every=10,
    )

    def fake_load_checkpoint(path, *, map_location):
        return {
            "theta": torch.zeros(3, dtype=torch.float32),
            "optimizer_state": None,
            "next_step": 5,
            "config": config.to_dict(),
            "family_names": ["fam0"],
            "species_names": ["sp0", "sp1"],
            "status": {
                "previous_objective": 1.5,
                "stable_loss_steps": 0,
            },
        }

    workflow_optimize_module = importlib.import_module("gpurec.workflow.optimize")
    monkeypatch.setattr(workflow_optimize_module, "load_checkpoint", fake_load_checkpoint)
    runner = FakeResumeRunner(config)

    with pytest.raises(RuntimeError, match=r"next_step 5.*configured steps 2"):
        runner.run()

    assert runner.fake_model.closed


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

    with pytest.raises(RuntimeError, match="GPUREC_BACKTRACK_BIN") as exc_info:
        _backtrack_command(
            cargo_manifest=tmp_path / "missing" / "Cargo.toml",
            backtrack_binary=None,
        )

    assert "--backtrack-binary" in str(exc_info.value)


def test_backtracking_command_uses_locked_cargo_fallback(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("GPUREC_BACKTRACK_BIN", raising=False)
    monkeypatch.setattr(backtracking.shutil, "which", lambda command: "/usr/bin/cargo")
    manifest = tmp_path / "Cargo.toml"
    manifest.write_text("[package]\nname = \"fixture\"\n", encoding="utf-8")

    assert _backtrack_command(cargo_manifest=manifest, backtrack_binary=None) == [
        "cargo",
        "run",
        "--locked",
        "--quiet",
        "--manifest-path",
        str(manifest),
        "--",
    ]


def test_backtracking_command_rejects_missing_cargo_fallback(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.delenv("GPUREC_BACKTRACK_BIN", raising=False)
    monkeypatch.setattr(backtracking.shutil, "which", lambda command: None)
    manifest = tmp_path / "Cargo.toml"
    manifest.write_text("[package]\nname = \"fixture\"\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="requires cargo on PATH") as exc_info:
        _backtrack_command(cargo_manifest=manifest, backtrack_binary=None)

    message = str(exc_info.value)
    assert "GPUREC_BACKTRACK_BIN" in message
    assert "backtrack_binary" in message
    assert "--backtrack-binary" in message


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


def test_xml_species_counts_singletons_only_for_one_copy_species():
    xml = """
    <recPhylo>
      <recGeneTree>
        <phylogeny>
          <clade>
            <eventsRec><speciation speciesLocation="Root"/></eventsRec>
            <clade><eventsRec><leaf speciesLocation="A"/></eventsRec></clade>
            <clade><eventsRec><leaf speciesLocation="A"/></eventsRec></clade>
            <clade><eventsRec><leaf speciesLocation="B"/></eventsRec></clade>
          </clade>
        </phylogeny>
      </recGeneTree>
    </recPhylo>
    """

    species, _ = _xml_species_and_transfer_counts(xml)

    assert species["A"]["copies"] == 2.0
    assert species["A"]["singletons"] == 0.0
    assert species["B"]["copies"] == 1.0
    assert species["B"]["singletons"] == 1.0
