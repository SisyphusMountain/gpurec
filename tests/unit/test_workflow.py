from __future__ import annotations

import csv
import importlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from threading import Lock
from types import SimpleNamespace

import pytest
import torch

import gpurec
import gpurec.backtracking as backtracking
import gpurec.api.model as api_model
import gpurec.api.uniform_chunked as uniform_chunked_api
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
    UniformChunkedReconModel,
    _UniformBuiltChunk,
    _UniformChunkSpec,
    _as_auto_int,
    _auto_positive_int,
    _selected_chunks,
)
from gpurec.workflow.checkpoint import (
    CHECKPOINT_VERSION,
    load_checkpoint,
    restore_model_theta,
    save_checkpoint,
    validate_checkpoint_model_compatibility,
)
from gpurec.workflow._metadata import (
    checkpoint_progress,
    checkpoint_status_dict,
    model_family_names,
    model_species_names,
)
from gpurec.workflow.config import (
    RunConfig,
    SamplingConfig,
    adagrad_restart_schedule_specs,
    adagrad_restart_schedule_total_steps,
    dtype_from_name,
    effective_route_metadata,
)
from gpurec.workflow.diagnostics import (
    append_jsonl,
    parameter_stats,
    solver_stats,
    tensor_stats,
    write_json_strict,
)
from gpurec.workflow.optimize import OptimizationRunner, _write_rate_table
from gpurec.workflow.sampling import SamplingRunner, _xml_species_and_transfer_counts

optimize_workflow = importlib.import_module("gpurec.workflow.optimize")
SUBPROCESS_TIMEOUT = 30


def _wildcard_export_names(import_statement: str) -> set[str]:
    namespace: dict[str, object] = {}
    exec(import_statement, namespace)
    return {name for name in namespace if not name.startswith("__")}


def _run_python_snippet(code: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[2],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        timeout=SUBPROCESS_TIMEOUT,
    )


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


def test_workflow_config_import_does_not_load_public_api_package():
    result = _run_python_snippet(
        "import sys; import gpurec.workflow.config; "
        "print('gpurec.api' in sys.modules)"
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"


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


def test_run_config_from_json_rejects_bad_device_type(tmp_path: Path):
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "species_tree": str(tmp_path / "sp.nwk"),
                "families_file": str(tmp_path / "families.txt"),
                "out_dir": str(tmp_path / "out"),
                "device": 42,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="device must be a device string"):
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
    assert gpurec.__all__ == list(gpurec._LAZY_EXPORTS)
    assert set(gpurec.__all__) <= set(dir(gpurec))
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


def test_batch_metadata_freezes_public_container_fields():
    family_indices = [0, 1]
    family_names = ["family_0", "family_1"]
    gene_tree_paths = [["g0.nwk"], ["g1a.nwk", "g1b.nwk"]]
    root_clade_rows = [3, 8]
    parameter_mapping = {
        "mode": "genewise",
        "theta_shape": [2, 3],
        "nested": {"batch_theta_rows": [0, 1]},
    }

    metadata = BatchMetadata(
        batch_index=0,
        family_indices=family_indices,
        family_names=family_names,
        gene_tree_paths=gene_tree_paths,
        family_count=2,
        clade_count=9,
        split_count=12,
        wave_count=4,
        max_wave_size=5,
        root_clade_rows=root_clade_rows,
        parameter_mapping=parameter_mapping,
    )
    family_indices.append(2)
    family_names[0] = "mutated"
    gene_tree_paths[0].append("mutated.nwk")
    root_clade_rows.append(13)
    parameter_mapping["theta_shape"].append(4)
    parameter_mapping["nested"]["batch_theta_rows"].append(2)

    assert metadata.family_indices == (0, 1)
    assert metadata.family_names == ("family_0", "family_1")
    assert metadata.gene_tree_paths == (("g0.nwk",), ("g1a.nwk", "g1b.nwk"))
    assert metadata.root_clade_rows == (3, 8)
    assert metadata.parameter_mapping["theta_shape"] == (2, 3)
    assert metadata.parameter_mapping["nested"]["batch_theta_rows"] == (0, 1)

    with pytest.raises(AttributeError):
        metadata.family_indices.append(3)  # type: ignore[attr-defined]
    with pytest.raises(TypeError):
        metadata.parameter_mapping["mode"] = "global"  # type: ignore[index]
    with pytest.raises(AttributeError):
        metadata.parameter_mapping["theta_shape"].append(4)  # type: ignore[attr-defined]


def test_top_level_exports_backtracking_surface():
    public_names = {
        "EVENT_KEYS",
        "ensure_backtracking_available",
        "export_backtracking_input",
        "recphyloxml_event_counts",
        "sample_backtracking_summaries",
        "sample_recphyloxml",
        "sample_recphyloxmls",
        "sample_recphyloxmls_to_dir",
    }

    assert set(backtracking.__all__) == public_names
    for name in backtracking.__all__:
        assert name in gpurec.__all__
        assert gpurec._LAZY_EXPORTS[name] == "gpurec.backtracking"
        assert getattr(gpurec, name) is getattr(backtracking, name)


def test_top_level_exports_workflow_surface():
    assert "load_checkpoint_config" not in workflow.__all__
    assert "load_checkpoint_config" not in gpurec.__all__
    assert not hasattr(workflow, "load_checkpoint_config")
    assert not hasattr(gpurec, "load_checkpoint_config")

    assert set(workflow.__all__) <= set(dir(workflow))
    for name in workflow.__all__:
        assert name in gpurec.__all__
        assert gpurec._LAZY_EXPORTS[name] == "gpurec.workflow"
        assert getattr(gpurec, name) is getattr(workflow, name)


def test_top_level_wildcard_import_matches_public_all():
    assert _wildcard_export_names("from gpurec import *") == set(gpurec.__all__)


def test_backtracking_wildcard_import_matches_public_all():
    assert _wildcard_export_names("from gpurec.backtracking import *") == set(
        backtracking.__all__
    )


def test_workflow_wildcard_import_matches_public_all():
    assert _wildcard_export_names("from gpurec.workflow import *") == set(
        workflow.__all__
    )


def test_uniform_chunked_wildcard_import_exposes_public_surface_only():
    assert uniform_chunked_api.__all__ == [
        "UniformChunkMetadata",
        "UniformChunkedReconModel",
    ]
    assert _wildcard_export_names("from gpurec.api.uniform_chunked import *") == set(
        uniform_chunked_api.__all__
    )


def test_import_gpurec_does_not_eagerly_import_workflow_or_backtracking():
    code = "\n".join(
        (
            "import sys",
            "import gpurec",
            "assert 'gpurec.workflow' not in sys.modules",
            "assert 'gpurec.backtracking' not in sys.modules",
        )
    )
    result = _run_python_snippet(code)

    assert result.returncode == 0
    assert result.stdout == ""
    assert result.stderr == ""


def test_import_workflow_does_not_eagerly_import_heavy_workflow_modules():
    code = "\n".join(
        (
            "import sys",
            "import gpurec.workflow",
            "assert 'gpurec.workflow.config' not in sys.modules",
            "assert 'gpurec.workflow.optimize' not in sys.modules",
            "assert 'gpurec.workflow.sampling' not in sys.modules",
            "assert 'gpurec.backtracking' not in sys.modules",
            "assert 'gpurec.api' not in sys.modules",
            "assert 'torch' not in sys.modules",
        )
    )
    result = _run_python_snippet(code)

    assert result.returncode == 0
    assert result.stdout == ""
    assert result.stderr == ""


def test_run_config_workflow_export_does_not_import_optimizer_or_sampling():
    code = "\n".join(
        (
            "import sys",
            "from gpurec.workflow import RunConfig",
            "assert RunConfig.__name__ == 'RunConfig'",
            "assert 'gpurec.workflow.config' in sys.modules",
            "assert 'gpurec.workflow.optimize' not in sys.modules",
            "assert 'gpurec.workflow.sampling' not in sys.modules",
            "assert 'gpurec.backtracking' not in sys.modules",
            "assert 'gpurec.api' not in sys.modules",
        )
    )
    result = _run_python_snippet(code)

    assert result.returncode == 0
    assert result.stdout == ""
    assert result.stderr == ""


def test_workflow_config_submodule_import_does_not_import_optimizer_or_sampling():
    code = "\n".join(
        (
            "import sys",
            "from gpurec.workflow.config import RunConfig",
            "assert RunConfig.__name__ == 'RunConfig'",
            "assert 'gpurec.workflow.config' in sys.modules",
            "assert 'gpurec.workflow.optimize' not in sys.modules",
            "assert 'gpurec.workflow.sampling' not in sys.modules",
            "assert 'gpurec.backtracking' not in sys.modules",
            "assert 'gpurec.api' not in sys.modules",
        )
    )
    result = _run_python_snippet(code)

    assert result.returncode == 0
    assert result.stdout == ""
    assert result.stderr == ""


def test_workflow_metadata_helper_import_does_not_import_heavy_modules():
    code = "\n".join(
        (
            "import sys",
            "import gpurec.workflow._metadata",
            "assert 'gpurec.api' not in sys.modules",
            "assert 'gpurec.backtracking' not in sys.modules",
            "assert 'gpurec.workflow.optimize' not in sys.modules",
            "assert 'gpurec.workflow.sampling' not in sys.modules",
            "assert 'torch' not in sys.modules",
        )
    )
    result = _run_python_snippet(code)

    assert result.returncode == 0
    assert result.stdout == ""
    assert result.stderr == ""


def test_workflow_optimize_export_survives_child_module_import_order():
    code = "\n".join(
        (
            "import importlib",
            "import gpurec.workflow as workflow",
            "optimize_module = importlib.import_module('gpurec.workflow.optimize')",
            "from gpurec.workflow import optimize",
            "assert optimize_module.optimize is optimize",
            "assert workflow.optimize is optimize",
            "assert callable(optimize)",
        )
    )
    result = _run_python_snippet(code)

    assert result.returncode == 0
    assert result.stdout == ""
    assert result.stderr == ""


def test_top_level_workflow_export_survives_child_module_import_order():
    code = "\n".join(
        (
            "import importlib",
            "import gpurec",
            "optimize_module = importlib.import_module('gpurec.workflow.optimize')",
            "from gpurec import optimize",
            "assert optimize_module.optimize is optimize",
            "assert callable(optimize)",
        )
    )
    result = _run_python_snippet(code)

    assert result.returncode == 0
    assert result.stdout == ""
    assert result.stderr == ""


def test_uniform_chunked_public_chunk_metadata_accessors():
    chunks = [
        _UniformBuiltChunk(
            spec=_UniformChunkSpec(indices=[0, 2], clades=7, splits=11),
            wave_layout={},
            waves=3,
            max_wave=5,
            split_rows=13,
            max_wave_split_rows=8,
        ),
        _UniformBuiltChunk(
            spec=_UniformChunkSpec(indices=[1], clades=4, splits=6),
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


def test_close_tolerates_partially_initialized_model():
    model = GeneReconModel.__new__(GeneReconModel)

    model.close()

    assert model._prefetch_closed is True
    assert model._prefetch_executor is None


def test_clear_batched_resident_does_not_materialize_missing_active_batch():
    model = GeneReconModel.__new__(GeneReconModel)
    model._batched_resident = True
    model._current_batch_index = 0
    model._batch_statics = [None]
    model._batch_lock = Lock()
    model._build_batch_static = lambda batch_idx: pytest.fail(
        "clear() should not materialize a missing resident batch"
    )

    model.clear()

    assert model._batch_statics == [None]


def test_clear_batched_resident_clears_existing_active_warm_state():
    static = SimpleNamespace(warm_E=object())
    model = GeneReconModel.__new__(GeneReconModel)
    model._batched_resident = True
    model._current_batch_index = 0
    model._batch_statics = [static]
    model._batch_lock = Lock()

    model.clear()

    assert static.warm_E is None


def test_close_shuts_down_executor_without_batch_lock():
    class FakeExecutor:
        def __init__(self):
            self.shutdown_kwargs: dict[str, bool] | None = None

        def shutdown(self, *, wait: bool, cancel_futures: bool) -> None:
            self.shutdown_kwargs = {
                "wait": wait,
                "cancel_futures": cancel_futures,
            }

    executor = FakeExecutor()
    model = GeneReconModel.__new__(GeneReconModel)
    model._prefetch_executor = executor
    model._batch_futures = {1: object()}

    model.close()

    assert executor.shutdown_kwargs == {"wait": False, "cancel_futures": True}
    assert model._prefetch_closed is True
    assert model._prefetch_executor is None
    assert model._batch_futures == {}


def test_family_input_returns_read_only_public_containers_and_tensor_copies():
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
    public_family.leaf_row_index[0] = 99
    public_family.leaf_col_index[0] = 99

    assert public_family.gene_tree_paths == ("g0.nwk",)
    assert dict(public_family.leaf_species_map) == {"gene_a": "SpeciesA"}
    assert public_family.ccp_helpers["nested"]["labels"] == ("left",)
    assert public_family.clade_leaf_labels == ("gene_a", "")

    with pytest.raises(AttributeError):
        public_family.gene_tree_paths.append("mutated.nwk")  # type: ignore[attr-defined]
    with pytest.raises(TypeError):
        public_family.leaf_species_map["gene_a"] = "Mutated"  # type: ignore[index]
    with pytest.raises(AttributeError):
        public_family.ccp_helpers["nested"]["labels"].append("mutated")  # type: ignore[attr-defined]
    with pytest.raises(TypeError):
        public_family.ccp_helpers["nested"] = {}  # type: ignore[index]
    with pytest.raises(AttributeError):
        public_family.clade_leaf_labels.append("mutated")  # type: ignore[attr-defined]

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


def test_family_input_materializes_compact_retained_dataset_on_demand():
    species = {"S": 1, "unnorm_row_max": torch.tensor([0.0], dtype=torch.float64)}
    compact_raw = {
        "species": species,
        "families": {"fam0": {"C": 2, "N_splits": 1, "root_clade_id": 0}},
    }
    full_raw = {
        "species": species,
        "families": {
            "fam0": {
                "ccp": {
                    "split_counts": torch.tensor([0, 1], dtype=torch.long),
                    "split_parents_sorted": torch.tensor([0], dtype=torch.long),
                    "split_leftrights_sorted": torch.tensor([[0, 1]], dtype=torch.long),
                    "log_split_probs_sorted": torch.tensor([0.0], dtype=torch.float64),
                    "C": 2,
                    "N_splits": 1,
                    "root_clade_id": 0,
                    "clade_leaf_labels": ["gene_a", ""],
                },
                "root_clade_id": 0,
                "leaf_row_index": torch.tensor([0], dtype=torch.long),
                "leaf_col_index": torch.tensor([0], dtype=torch.long),
            }
        },
    }

    class FakeRustPreprocessed:
        def to_torch(self):
            return full_raw

    dataset = GeneDataset._from_preprocessed_raw(
        raw=compact_raw,
        species_tree_path="sp.nwk",
        gene_tree_paths=[["g0.nwk"]],
        genewise=False,
        specieswise=True,
        dtype=torch.float32,
        device=torch.device("cpu"),
        family_names=["fam0"],
        leaf_species_maps=[{"gene_a": "SpeciesA"}],
        rust_preprocessed=FakeRustPreprocessed(),
        compact_families=True,
    )
    model = object.__new__(GeneReconModel)
    object.__setattr__(model, "_dataset", dataset)

    assert dataset._compact_families is True
    public_family = GeneReconModel.family_input(model, 0)

    assert dataset._compact_families is False
    assert public_family.clade_count == 2
    assert public_family.split_count == 1
    assert public_family.clade_leaf_labels == ("gene_a", "")


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


def test_materialize_batches_builds_each_resident_batch_and_returns_metadata_copy():
    model = object.__new__(GeneReconModel)
    metadata = [SimpleNamespace(batch_index=0), SimpleNamespace(batch_index=1)]
    calls: list[int] = []

    object.__setattr__(model, "_batched_resident", True)
    object.__setattr__(model, "_batch_specs", [object(), object()])
    object.__setattr__(model, "batch_metadata", metadata)

    def fake_ensure_batch_static(batch_index: int) -> object:
        calls.append(batch_index)
        return object()

    object.__setattr__(model, "_ensure_batch_static", fake_ensure_batch_static)

    result = GeneReconModel.materialize_batches(model)

    assert calls == [0, 1]
    assert result == metadata
    assert result is not metadata


def test_materialize_batches_rejects_unbuilt_nonbatched_state():
    model = object.__new__(GeneReconModel)
    object.__setattr__(model, "_batched_resident", False)
    object.__setattr__(model, "_static", None)

    with pytest.raises(RuntimeError, match="resident static state"):
        GeneReconModel.materialize_batches(model)


def test_full_loss_for_theta_uses_streaming_contract_for_explicit_theta():
    model = object.__new__(GeneReconModel)
    object.__setattr__(model, "_mode", "global")
    object.__setattr__(
        model,
        "_dataset",
        SimpleNamespace(S=2, families=[object(), object()]),
    )
    calls: list[dict[str, object]] = []

    def fake_stream_full_batches(
        theta: torch.Tensor,
        *,
        need_grad: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        calls.append({"theta": theta, "need_grad": need_grad})
        loss = torch.tensor(7.0, dtype=torch.float32)
        grad = (
            torch.tensor([0.5, -1.5, 0.25], dtype=torch.float32)
            if need_grad
            else None
        )
        return loss, grad

    object.__setattr__(model, "_stream_full_batches", fake_stream_full_batches)

    theta = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True)
    loss = GeneReconModel.full_loss_for_theta(model, theta)
    loss.backward()

    assert calls[0]["theta"] is theta
    assert calls[0]["need_grad"] is True
    torch.testing.assert_close(loss, torch.tensor(7.0, dtype=torch.float64))
    torch.testing.assert_close(
        theta.grad,
        torch.tensor([0.5, -1.5, 0.25], dtype=torch.float64),
    )

    probe = theta.detach().clone()
    with torch.no_grad():
        no_grad_loss = GeneReconModel.full_loss_for_theta(model, probe)

    assert calls[1]["theta"] is probe
    assert calls[1]["need_grad"] is False
    torch.testing.assert_close(
        no_grad_loss,
        torch.tensor(7.0, dtype=torch.float64),
    )


def test_run_config_normalizes_batch_controls(tmp_path: Path):
    config = RunConfig(
        species_tree=tmp_path / "sp.nwk",
        families_file=tmp_path / "families.txt",
        out_dir=tmp_path / "out",
        family_chunk_size="all",
        batch_packing="depth-first-fit",
        clade_budget="12",
        max_wave_size="32",
        small_family_max_leaves="8",
        device="cpu",
    )

    assert config.family_chunk_size == 0
    assert config.batch_packing == "depth_first_fit"
    assert config.clade_budget == 12
    assert config.max_wave_size == 32
    assert config.small_family_max_leaves == 8


def test_run_config_from_dict_preserves_batch_packing_default(tmp_path: Path):
    config = RunConfig.from_dict(
        {
            "species_tree": tmp_path / "sp.nwk",
            "families_file": tmp_path / "families.txt",
            "out_dir": tmp_path / "out",
            "device": "cpu",
        }
    )

    assert config.batch_packing == "depth_first_fit"
    assert config.small_family_max_leaves == 0


def test_run_config_ignores_legacy_optimizer_controls(tmp_path: Path):
    config = RunConfig.from_dict(
        {
            "species_tree": tmp_path / "sp.nwk",
            "families_file": tmp_path / "families.txt",
            "out_dir": tmp_path / "out",
            "mode": "genewise",
            "optimizer": "adam-fd-newton",
            "fd_newton_max_step": 0.0,
            "grad_inf_tol": 10.0,
            "hessian_sgd_polish_max_steps": 12,
            "hessian_sgd_polish_refresh_steps": 4,
            "hessian_sgd_polish_max_ls": 2,
            "solver_warmup_grad_inf_tol": 0.0,
            "device": "cpu",
        }
    )

    assert config.optimizer == "adam-fd-newton"
    data = config.to_dict()
    for field in (
        "fd_newton_max_step",
        "grad_inf_tol",
        "hessian_sgd_polish_max_steps",
        "hessian_sgd_polish_refresh_steps",
        "hessian_sgd_polish_max_ls",
        "solver_warmup_grad_inf_tol",
    ):
        assert not hasattr(config, field)
        assert field not in data


def test_run_config_rejects_null_batch_packing(tmp_path: Path):
    with pytest.raises(ValueError, match="batch_packing"):
        RunConfig.from_dict(
            {
                "species_tree": tmp_path / "sp.nwk",
                "families_file": tmp_path / "families.txt",
                "out_dir": tmp_path / "out",
                "device": "cpu",
                "batch_packing": None,
            }
        )


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


def test_run_config_defaults_to_natural_rate_bounds(tmp_path: Path):
    config = RunConfig(
        species_tree=tmp_path / "sp.nwk",
        families_file=tmp_path / "families.txt",
        out_dir=tmp_path / "out",
        device="cpu",
    )

    assert config.min_rate == pytest.approx(2.0**-30)
    assert config.max_rate == 2.0


def test_run_config_defaults_to_hessian_sgd_for_genewise_mode(tmp_path: Path):
    config = RunConfig(
        species_tree=tmp_path / "sp.nwk",
        families_file=tmp_path / "families.txt",
        out_dir=tmp_path / "out",
        mode="genewise",
        device="cpu",
    )

    assert config.optimizer == "hessian-sgd"


def test_effective_route_metadata_reports_production_likelihood_contract(
    tmp_path: Path,
):
    config = RunConfig(
        species_tree=tmp_path / "sp.nwk",
        families_file=tmp_path / "families.txt",
        out_dir=tmp_path / "out",
        mode="genewise",
        device="cpu",
    )

    route = effective_route_metadata(config)
    basis = "hogenom_and_" + "test_trees_" + "1000"

    assert route["objective"] == "negative_log_likelihood_bits"
    assert route["gradient_route"] == "implicit_first_order_adjoint"
    assert route["rate_parameterization"] == "base2_log_dlt_rates"
    assert route["production_default_basis"] == basis
    assert route["optimizer"] == "hessian-sgd"
    assert route["configured_steps"] == 5000
    assert route["optimizer_step_cap"] == 5000
    assert route["optimizer_step_cap_reason"] == "configured_steps"
    assert route["hessian_sgd_normal_fixed_iters_pi"] is None
    assert route["hessian_sgd_normal_neumann_terms"] is None


def test_effective_route_metadata_reports_hessian_sgd_normal_solver_overrides(
    tmp_path: Path,
):
    config = RunConfig(
        species_tree=tmp_path / "sp.nwk",
        families_file=tmp_path / "families.txt",
        out_dir=tmp_path / "out",
        mode="genewise",
        hessian_sgd_normal_fixed_iters_pi=12,
        hessian_sgd_normal_neumann_terms=12,
        device="cpu",
    )

    route = effective_route_metadata(config)

    assert route["optimizer"] == "hessian-sgd"
    assert route["hessian_sgd_normal_fixed_iters_pi"] == 12
    assert route["hessian_sgd_normal_neumann_terms"] == 12


def test_run_config_accepts_adam_fd_newton_for_genewise_mode(tmp_path: Path):
    config = RunConfig(
        species_tree=tmp_path / "sp.nwk",
        families_file=tmp_path / "families.txt",
        out_dir=tmp_path / "out",
        mode="genewise",
        optimizer="adam-fd-newton",
        device="cpu",
    )

    assert config.optimizer == "adam-fd-newton"
    assert config.fd_adam_warmup_steps == 3
    assert config.fd_hessian_refresh_steps == 16
    assert config.fd_hessian_epsilon == pytest.approx(1e-3)


def test_run_config_accepts_hessian_sgd_for_genewise_mode(tmp_path: Path):
    config = RunConfig(
        species_tree=tmp_path / "sp.nwk",
        families_file=tmp_path / "families.txt",
        out_dir=tmp_path / "out",
        mode="genewise",
        optimizer="hessian-sgd",
        device="cpu",
    )

    assert config.optimizer == "hessian-sgd"
    assert config.fd_hessian_refresh_steps == 16
    assert config.fd_hessian_epsilon == pytest.approx(1e-3)
    assert config.solver_warmup_iters == 4
    assert config.loss_change_tol == pytest.approx(3e-3)
    assert config.hessian_sgd_normal_fixed_iters_pi is None
    assert config.hessian_sgd_normal_neumann_terms is None


def test_run_config_auto_optimizer_uses_adam_for_shared_theta_modes(tmp_path: Path):
    config = RunConfig(
        species_tree=tmp_path / "sp.nwk",
        families_file=tmp_path / "families.txt",
        out_dir=tmp_path / "out",
        mode="global",
        device="cpu",
    )

    assert config.optimizer == "adam"


def test_run_config_auto_optimizer_uses_adagrad_restarts_for_specieswise_mode(
    tmp_path: Path,
):
    config = RunConfig(
        species_tree=tmp_path / "sp.nwk",
        families_file=tmp_path / "families.txt",
        out_dir=tmp_path / "out",
        mode="specieswise",
        device="cpu",
    )

    assert config.optimizer == "adagrad-restarts"
    assert config.adagrad_restart_schedule == "8:1:60,16:0.5:35,32:0.5:30"
    assert config.adagrad_restart_final_check_iters == 128
    assert adagrad_restart_schedule_total_steps(config.adagrad_restart_schedule) == 125
    route = effective_route_metadata(config)
    assert route["adagrad_restart_total_steps"] == 125
    assert route["configured_steps"] == 5000
    assert route["optimizer_step_cap"] == 125
    assert route["optimizer_step_cap_reason"] == "adagrad_restart_schedule"


def test_run_config_specieswise_adagrad_restarts_step_cap_honors_shorter_steps(
    tmp_path: Path,
):
    config = RunConfig(
        species_tree=tmp_path / "sp.nwk",
        families_file=tmp_path / "families.txt",
        out_dir=tmp_path / "out",
        mode="specieswise",
        steps=12,
        device="cpu",
    )

    route = effective_route_metadata(config)

    assert route["adagrad_restart_total_steps"] == 125
    assert route["configured_steps"] == 12
    assert route["optimizer_step_cap"] == 12
    assert route["optimizer_step_cap_reason"] == "configured_steps"


def test_run_config_accepts_specieswise_adagrad_restart_schedule(tmp_path: Path):
    config = RunConfig(
        species_tree=tmp_path / "sp.nwk",
        families_file=tmp_path / "families.txt",
        out_dir=tmp_path / "out",
        mode="specieswise",
        optimizer="adagrad-restarts",
        adagrad_restart_schedule="4:1.0:2, 8:0.25:3",
        adagrad_restart_final_check_iters=16,
        device="cpu",
    )

    assert config.optimizer == "adagrad-restarts"
    assert config.adagrad_restart_schedule == "4:1:2,8:0.25:3"
    assert config.adagrad_restart_final_check_iters == 16


def test_run_config_accepts_split_specieswise_adagrad_restart_schedule(
    tmp_path: Path,
):
    config = RunConfig(
        species_tree=tmp_path / "sp.nwk",
        families_file=tmp_path / "families.txt",
        out_dir=tmp_path / "out",
        mode="specieswise",
        optimizer="adagrad-restarts",
        adagrad_restart_schedule="8/4:1.0:2,16/8/6:0.25:3",
        adagrad_restart_final_check_iters=16,
        device="cpu",
    )

    assert config.adagrad_restart_schedule == "8/4:1:2,16/8/6:0.25:3"
    phases = adagrad_restart_schedule_specs(config.adagrad_restart_schedule)
    assert [
        (
            phase.fixed_iters_e,
            phase.fixed_iters_pi,
            phase.neumann_terms,
            phase.budget,
        )
        for phase in phases
    ] == [
        (8, 4, 4, 4),
        (16, 8, 6, 8),
    ]


def test_run_config_rejects_adagrad_restarts_outside_specieswise(tmp_path: Path):
    with pytest.raises(
        ValueError,
        match="adagrad-restarts optimizer requires specieswise mode",
    ):
        RunConfig(
            species_tree=tmp_path / "sp.nwk",
            families_file=tmp_path / "families.txt",
            out_dir=tmp_path / "out",
            mode="genewise",
            optimizer="adagrad-restarts",
            device="cpu",
        )


@pytest.mark.parametrize(
    ("fixed_iters_e", "expected"),
    [
        (None, 32),
        (6, 32),
        (64, 64),
    ],
)
def test_run_config_specieswise_high_pi_budget_raises_fixed_e_budget(
    tmp_path: Path,
    fixed_iters_e: int | None,
    expected: int,
):
    config = RunConfig(
        species_tree=tmp_path / "sp.nwk",
        families_file=tmp_path / "families.txt",
        out_dir=tmp_path / "out",
        mode="specieswise",
        device="cpu",
        fixed_iters_e=fixed_iters_e,
        fixed_iters_pi=32,
    )

    assert config.fixed_iters_e == expected


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("species_tree", "species_tree must be a path string"),
        ("families_file", "families_file must be a path string"),
        ("out_dir", "out_dir must be a path string"),
        ("resume_from", "resume_from must be a path string"),
    ],
)
def test_run_config_rejects_non_path_constructor_fields(
    tmp_path: Path,
    field: str,
    message: str,
):
    kwargs = {
        "species_tree": tmp_path / "sp.nwk",
        "families_file": tmp_path / "families.txt",
        "out_dir": tmp_path / "out",
        "device": "cpu",
        field: 42,
    }

    with pytest.raises(ValueError, match=message):
        RunConfig(**kwargs)


@pytest.mark.parametrize(
    ("device", "message"),
    [
        (42, "device must be a device string"),
        ("cdua", "device must be a valid torch device string"),
    ],
)
def test_run_config_rejects_invalid_device_values(
    tmp_path: Path,
    device: object,
    message: str,
):
    with pytest.raises(ValueError, match=message):
        RunConfig(
            species_tree=tmp_path / "sp.nwk",
            families_file=tmp_path / "families.txt",
            out_dir=tmp_path / "out",
            device=device,
        )


def test_run_config_rejects_unsupported_auto_chunking(tmp_path: Path):
    with pytest.raises(ValueError, match="auto"):
        RunConfig(
            species_tree=tmp_path / "sp.nwk",
            families_file=tmp_path / "families.txt",
            out_dir=tmp_path / "out",
            family_chunk_size="auto",
            device="cpu",
        )


def test_run_config_rejects_adam_fd_newton_outside_genewise(tmp_path: Path):
    with pytest.raises(ValueError, match="adam-fd-newton optimizer requires genewise"):
        RunConfig(
            species_tree=tmp_path / "sp.nwk",
            families_file=tmp_path / "families.txt",
            out_dir=tmp_path / "out",
            mode="global",
            optimizer="adam-fd-newton",
            device="cpu",
        )


def test_run_config_rejects_hessian_sgd_outside_genewise(tmp_path: Path):
    with pytest.raises(ValueError, match="hessian-sgd optimizer requires genewise"):
        RunConfig(
            species_tree=tmp_path / "sp.nwk",
            families_file=tmp_path / "families.txt",
            out_dir=tmp_path / "out",
            mode="global",
            optimizer="hessian-sgd",
            device="cpu",
        )


def test_run_config_rejects_batched_lbfgs_outside_genewise(tmp_path: Path):
    with pytest.raises(ValueError, match="batched-lbfgs.*genewise"):
        RunConfig(
            species_tree=tmp_path / "sp.nwk",
            families_file=tmp_path / "families.txt",
            out_dir=tmp_path / "out",
            mode="global",
            optimizer="batched-lbfgs",
            device="cpu",
        )


def test_run_config_accepts_strong_wolfe_for_batched_lbfgs(tmp_path: Path):
    config = RunConfig(
        species_tree=tmp_path / "sp.nwk",
        families_file=tmp_path / "families.txt",
        out_dir=tmp_path / "out",
        mode="genewise",
        optimizer="batched-lbfgs",
        lbfgs_line_search="strong_wolfe",
        device="cpu",
    )

    assert config.lbfgs_line_search == "strong_wolfe"


@pytest.mark.parametrize("optimizer", ["batched-lbfgs", "adam-fd-newton", "hessian-sgd"])
def test_run_config_accepts_adaptive_rebatch_for_genewise_batch_optimizers(
    tmp_path: Path,
    optimizer: str,
):
    config = RunConfig(
        species_tree=tmp_path / "sp.nwk",
        families_file=tmp_path / "families.txt",
        out_dir=tmp_path / "out",
        mode="genewise",
        optimizer=optimizer,
        adaptive_rebatch=True,
        adaptive_rebatch_fraction="0.75",
        adaptive_rebatch_check_interval="2",
        adaptive_rebatch_min_remaining_families="3",
        device="cpu",
    )

    assert config.adaptive_rebatch is True
    assert config.adaptive_rebatch_fraction == pytest.approx(0.75)
    assert config.adaptive_rebatch_check_interval == 2
    assert config.adaptive_rebatch_min_remaining_families == 3


def test_run_config_rejects_adaptive_rebatch_outside_batch_optimizers(tmp_path: Path):
    with pytest.raises(ValueError, match="adaptive_rebatch requires"):
        RunConfig(
            species_tree=tmp_path / "sp.nwk",
            families_file=tmp_path / "families.txt",
            out_dir=tmp_path / "out",
            mode="genewise",
            optimizer="adam",
            adaptive_rebatch=True,
            device="cpu",
        )


@pytest.mark.parametrize("value", [0.0, -0.1, 1.1])
def test_run_config_rejects_invalid_adaptive_rebatch_fraction(
    tmp_path: Path,
    value: float,
):
    with pytest.raises(ValueError, match="adaptive_rebatch_fraction"):
        RunConfig(
            species_tree=tmp_path / "sp.nwk",
            families_file=tmp_path / "families.txt",
            out_dir=tmp_path / "out",
            mode="genewise",
            optimizer="batched-lbfgs",
            adaptive_rebatch=True,
            adaptive_rebatch_fraction=value,
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
        ("fd_hessian_epsilon", math.nan),
        ("fd_newton_damping", math.inf),
        ("adaptive_rebatch_fraction", math.inf),
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
        ("adaptive_rebatch_fraction", False),
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
        ("adaptive_iters", "false"),
        ("adaptive_iters", 0),
        ("adaptive_neumann_terms", "false"),
        ("adaptive_neumann_terms", 0),
        ("adaptive_rebatch", "false"),
        ("adaptive_rebatch", 0),
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


def test_run_config_rejects_adaptive_neumann_terms_mode(tmp_path: Path):
    with pytest.raises(
        ValueError,
        match="current behaviour is absolutely terrible.*MUST be fixed",
    ):
        RunConfig(
            species_tree=tmp_path / "sp.nwk",
            families_file=tmp_path / "families.txt",
            out_dir=tmp_path / "out",
            device="cpu",
            adaptive_neumann_terms=True,
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
        ("final_check_iters", 12.5),
        ("solver_warmup_iters", 1.5),
        ("solver_warmup_loss_patience", 1.5),
        ("convergence_check_interval", 4.5),
        ("steps", 1.5),
        ("adam_warmup_steps", 0.5),
        ("fd_adam_warmup_steps", 0.5),
        ("fd_hessian_refresh_steps", 0.5),
        ("hessian_sgd_normal_fixed_iters_pi", 12.5),
        ("hessian_sgd_normal_neumann_terms", 12.5),
        ("adaptive_rebatch_check_interval", 0.5),
        ("adaptive_rebatch_min_remaining_families", 1.5),
        ("small_family_max_leaves", 1.5),
        ("lbfgs_max_iter", 1.5),
        ("lbfgs_max_ls", 1.5),
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


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("hessian_sgd_normal_fixed_iters_pi", 0),
        ("hessian_sgd_normal_neumann_terms", 0),
    ],
)
def test_run_config_rejects_invalid_hessian_sgd_controls(
    tmp_path: Path,
    field: str,
    value: object,
):
    with pytest.raises(ValueError, match=field):
        RunConfig(
            species_tree=tmp_path / "sp.nwk",
            families_file=tmp_path / "families.txt",
            out_dir=tmp_path / "out",
            optimizer="hessian-sgd",
            mode="genewise",
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
    with pytest.raises(ValueError, match="hessian_sgd_normal_fixed_iters_pi"):
        RunConfig(
            species_tree=tmp_path / "sp.nwk",
            families_file=tmp_path / "families.txt",
            out_dir=tmp_path / "out",
            optimizer="hessian-sgd",
            mode="genewise",
            device="cpu",
            hessian_sgd_normal_fixed_iters_pi=3,
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
        ({"family_chunk_candidates": "25"}, "family_chunk_candidates"),
        ({"family_chunk_candidates": [-1]}, "family_chunk_candidates"),
        ({"family_chunk_candidates": [1.5]}, "family_chunk_candidates"),
        ({"max_wave_candidates": None}, "max_wave_candidates"),
        ({"max_wave_candidates": "8192"}, "max_wave_candidates"),
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


def test_gene_dataset_rejects_single_gene_tree_path_before_extension(
    tmp_path: Path,
    monkeypatch,
):
    calls: list[bool] = []

    def fake_load_extension():
        calls.append(True)
        raise AssertionError("_load_preprocess_extension should not run")

    monkeypatch.setattr("gpurec.core.model._load_preprocess_extension", fake_load_extension)

    with pytest.raises(ValueError, match="gene_trees"):
        GeneDataset(
            species_tree_path=tmp_path / "missing_species.nwk",
            gene_tree_paths="g.nwk",
            genewise=False,
            specieswise=False,
            device=torch.device("cpu"),
        )

    assert calls == []


def test_gene_dataset_rejects_duplicate_family_names_before_extension(
    tmp_path: Path,
    monkeypatch,
):
    calls: list[bool] = []

    def fake_load_extension():
        calls.append(True)
        raise AssertionError("_load_preprocess_extension should not run")

    monkeypatch.setattr("gpurec.core.model._load_preprocess_extension", fake_load_extension)

    with pytest.raises(ValueError, match="duplicate family name 'fam0'"):
        GeneDataset(
            species_tree_path=tmp_path / "missing_species.nwk",
            gene_tree_paths=[tmp_path / "a.nwk", tmp_path / "b.nwk"],
            genewise=False,
            specieswise=False,
            device=torch.device("cpu"),
            family_names=["fam0", "fam0"],
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


@pytest.mark.parametrize(
    ("mode", "expected_flags", "expected_theta_shape"),
    [
        (" Global ", (False, False), (3,)),
        (" SpeciesWise ", (False, True), (2, 3)),
        (" GeneWise ", (True, False), (2, 3)),
    ],
)
def test_gene_recon_from_trees_normalizes_mode_like_uniform_api(
    tmp_path: Path,
    monkeypatch,
    mode: str,
    expected_flags: tuple[bool, bool],
    expected_theta_shape: tuple[int, ...],
):
    calls: dict[str, object] = {}

    class FakeDataset:
        S = 2

        @classmethod
        def from_retained_preprocess(cls, **kwargs: object) -> "FakeDataset":
            return cls(**kwargs)

        def __init__(
            self,
            *,
            genewise: bool,
            specieswise: bool,
            dtype: torch.dtype,
            device: torch.device,
            **_kwargs: object,
        ) -> None:
            calls["dataset_flags"] = (genewise, specieswise)
            self.genewise = genewise
            self.specieswise = specieswise
            self.dtype = dtype
            self.device = device
            self.families = [object(), object()]

    def fake_init(
        self: GeneReconModel,
        *,
        dataset: FakeDataset,
        mode: str,
        theta_init: torch.Tensor | None = None,
        **_kwargs: object,
    ) -> None:
        calls["init_mode"] = mode
        calls["theta_shape"] = None if theta_init is None else tuple(theta_init.shape)
        calls["theta_dtype"] = None if theta_init is None else theta_init.dtype

    monkeypatch.setattr(api_model, "GeneDataset", FakeDataset)
    monkeypatch.setattr(
        api_model,
        "require_cuda_device",
        lambda device, *, owner: torch.device("cpu"),
    )
    monkeypatch.setattr(api_model.GeneReconModel, "__init__", fake_init)

    GeneReconModel.from_trees(
        tmp_path / "sp.nwk",
        [tmp_path / "g0.nwk", tmp_path / "g1.nwk"],
        mode=mode,
        device="cpu",
        dtype=torch.float64,
        theta_init_rates=(0.1, 0.2, 0.3),
    )

    assert calls["dataset_flags"] == expected_flags
    assert calls["init_mode"] == mode.strip().lower()
    assert calls["theta_shape"] == expected_theta_shape
    assert calls["theta_dtype"] is torch.float64


@pytest.mark.parametrize(
    "factory",
    (
        GeneReconModel.from_trees,
        UniformChunkedReconModel.from_trees,
    ),
)
def test_public_constructors_reject_removed_preprocessing_cache_kwargs(
    tmp_path: Path,
    factory,
):
    with pytest.raises(TypeError, match="preprocess caching has been removed"):
        factory(
            tmp_path / "sp.nwk",
            [tmp_path / "g.nwk"],
            preprocess_cache_dir=tmp_path / "cache",
        )


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


@pytest.mark.parametrize("name", ["bf16", "bfloat16", "torch.bfloat16"])
def test_bfloat16_is_direct_uniform_api_only(name: str):
    with pytest.raises(ValueError, match="float32 or float64"):
        dtype_from_name(name)
    with pytest.raises(ValueError, match="dtype"):
        api_model._validate_gene_dtype(torch.bfloat16)

    assert uniform_chunked_api._validate_uniform_dtype(torch.bfloat16) is torch.bfloat16


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
        ("small_family_max_leaves", -1),
        ("small_family_max_leaves", 1.5),
        ("small_family_max_leaves", True),
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
        ("from_trees", {"preprocess_cpu_cores": 0}, "preprocess_cpu_cores"),
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
            {"preprocess_cpu_cores": True},
            "preprocess_cpu_cores",
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
        ("from_trees", {"preprocess_cpu_cores": 0}, "preprocess_cpu_cores"),
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
            {"preprocess_cpu_cores": True},
            "preprocess_cpu_cores",
        ),
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
        ({"use_pruning": "false"}, "use_pruning"),
        ({"warm_start_E": "false"}, "warm_start_E"),
        ({"profile": "false"}, "profile"),
    ],
)
def test_uniform_chunked_init_rejects_nonbool_controls_before_side_effects(
    tmp_path: Path,
    monkeypatch,
    kwargs: dict[str, object],
    message: str,
):
    with pytest.raises(ValueError, match=message) as exc_info:
        UniformChunkedReconModel(
            species_tree=tmp_path / "missing_species.nwk",
            gene_trees=[tmp_path / "missing_gene.nwk"],
            device="cpu",
            **kwargs,
        )

    assert "CUDA" not in str(exc_info.value)


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
        ({"use_pruning": "false"}, "use_pruning"),
        ({"warm_start_E": "false"}, "warm_start_E"),
        ({"profile": "false"}, "profile"),
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
    ("factory", "args"),
    [
        (
            UniformChunkedReconModel.from_trees,
            lambda tmp_path: (
                tmp_path / "missing_species.nwk",
                [tmp_path / "missing_gene.nwk"],
            ),
        ),
        (
            UniformChunkedReconModel.from_alerax_families,
            lambda tmp_path: (
                tmp_path / "missing_species.nwk",
                tmp_path / "missing_families.txt",
            ),
        ),
    ],
)
def test_uniform_chunked_factories_reject_removed_optimized_env_toggle(
    tmp_path: Path,
    factory,
    args,
):
    with pytest.raises(TypeError, match="optimized environment toggles"):
        factory(*args(tmp_path), set_optimized_env=False)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"fixed_iters_E": math.inf}, "fixed_iters_E"),
        ({"fixed_iters_E": 0}, "fixed_iters_E"),
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


def test_gene_recon_configure_solver_iterations_rejects_adaptive_neumann_terms_mode():
    model = SimpleNamespace()

    with pytest.raises(
        ValueError,
        match="current behaviour is absolutely terrible.*MUST be fixed",
    ):
        GeneReconModel.configure_solver_iterations(
            model,
            adaptive_neumann_terms=True,
        )

    assert not hasattr(model, "_adaptive_neumann_terms")


def test_gene_recon_configure_solver_iterations_can_restore_adaptive_e():
    static = SimpleNamespace(
        fixed_iters_E=6,
        fixed_iters_Pi=6,
        neumann_terms=6,
        pi_max_diff_tol=1e-5,
        gradient_change_tol=1e-4,
    )
    model = SimpleNamespace(
        _fixed_iters_E=6,
        _fixed_iters_Pi=6,
        _neumann_terms=6,
        cached_static_states=[static],
    )

    GeneReconModel.configure_solver_iterations(
        model,
        fixed_iters_E=None,
        fixed_iters_Pi=64,
        neumann_terms=64,
    )

    assert model._fixed_iters_E is None
    assert model._fixed_iters_Pi == 64
    assert model._neumann_terms == 64
    assert static.fixed_iters_E is None
    assert static.fixed_iters_Pi == 64
    assert static.neumann_terms == 64


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"min_rate": math.nan}, "min_rate"),
        ({"max_rate": math.inf}, "max_rate"),
        ({"min_rate": 0.0}, "min_rate"),
        ({"min_rate": -1.0}, "min_rate"),
    ],
)
@pytest.mark.parametrize("model_type", [GeneReconModel, UniformChunkedReconModel])
def test_recon_model_clamp_theta_rejects_invalid_rates(
    model_type: type[GeneReconModel] | type[UniformChunkedReconModel],
    kwargs: dict[str, float],
    message: str,
):
    model = SimpleNamespace(theta=torch.nn.Parameter(torch.zeros(3)))

    with pytest.raises(ValueError, match=message):
        model_type.clamp_theta_(model, **kwargs)


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


def test_gene_recon_constructors_reject_adaptive_neumann_terms_mode_before_io(
    tmp_path: Path,
):
    with pytest.raises(
        ValueError,
        match="current behaviour is absolutely terrible.*MUST be fixed",
    ):
        GeneReconModel.from_alerax_families(
            tmp_path / "missing_species.nwk",
            tmp_path / "missing_families.txt",
            device="cpu",
            adaptive_neumann_terms=True,
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


def test_full_nll_per_family_delegates_to_genewise_streaming_helper():
    model = object.__new__(GeneReconModel)
    model._mode = "genewise"
    expected = torch.tensor([1.0, 2.0], dtype=torch.float64)
    calls: list[bool] = []

    def full_genewise_nll_and_grad(*, need_grad: bool):
        calls.append(need_grad)
        return expected, None

    model.full_genewise_nll_and_grad = full_genewise_nll_and_grad  # type: ignore[method-assign]

    actual = model.full_nll_per_family()

    assert calls == [False]
    assert actual is expected


@pytest.mark.parametrize("mode", ["global", "specieswise"])
def test_full_nll_per_family_rejects_shared_theta_modes(mode: str):
    model = object.__new__(GeneReconModel)
    model._mode = mode

    with pytest.raises(ValueError, match="full_nll_per_family.*genewise mode"):
        model.full_nll_per_family()


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


def test_workflow_jsonl_diagnostics_sanitize_nonfinite_values(tmp_path: Path):
    path = tmp_path / "history.jsonl"
    row = {
        "finite": 1.25,
        "infinite": math.inf,
        "nested": {"nan": math.nan},
        "stats": tensor_stats("x", torch.tensor([float("inf")])),
    }

    append_jsonl(path, row)

    def reject_json_constant(constant: str) -> None:
        raise AssertionError(f"non-standard JSON constant {constant}")

    payload = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=reject_json_constant,
    )
    assert payload["finite"] == pytest.approx(1.25)
    assert payload["infinite"] is None
    assert payload["nested"]["nan"] is None
    assert payload["stats"]["x/max"] is None


def test_workflow_json_diagnostics_write_strict_file(tmp_path: Path):
    path = tmp_path / "nested" / "summary.json"

    write_json_strict(path, {"z": math.inf, "a": [1.0, math.nan]})

    text = path.read_text(encoding="utf-8")
    assert text.endswith("\n")
    assert text.index('"a"') < text.index('"z"')
    assert json.loads(text) == {"a": [1.0, None], "z": None}


def test_workflow_solver_stats_surface_e_adjoint_failure_telemetry():
    model = SimpleNamespace(
        solver_stat_records=lambda: [
            {
                "E_iterations": 2,
                "Pi_max_iterations": 6,
                "Pi_wave_iterations": [1, 2],
                "Pi_wave_count": 2,
                "Pi_converged_waves": 2,
                "Neumann_terms": 4,
                "Gradient_converged": True,
                "E_adjoint_iterations": 3,
                "E_adjoint_rel_res": 0.25,
                "E_adjoint_success": False,
            },
            {
                "E_iterations": 3,
                "Pi_max_iterations": 6,
                "Pi_wave_iterations": [2],
                "Pi_wave_count": 1,
                "Pi_converged_waves": 1,
                "Neumann_terms": 4,
                "Gradient_converged": True,
                "E_adjoint_iterations": 1,
                "E_adjoint_rel_res": 0.05,
                "E_adjoint_success": True,
            },
        ]
    )

    stats = solver_stats(model)

    assert stats["solver/e_adjoint_iterations_max"] == 3.0
    assert stats["solver/e_adjoint_iterations_mean"] == pytest.approx(2.0)
    assert stats["solver/e_adjoint_rel_res_max"] == pytest.approx(0.25)
    assert stats["solver/e_adjoint_rel_res_mean"] == pytest.approx(0.15)
    assert stats["solver/e_adjoint_success_batches"] == 1.0
    assert stats["solver/e_adjoint_failed_batches"] == 1.0
    assert stats["solver/gradient_converged_batches"] == 2.0


def test_workflow_metadata_model_name_helpers_return_copies_and_fallbacks():
    assert model_family_names(SimpleNamespace()) == []
    assert model_species_names(SimpleNamespace()) == []

    model = SimpleNamespace(family_names=["family_a"], species_names=["species_a"])
    family_names = model_family_names(model)
    species_names = model_species_names(model)
    family_names.append("mutated_family")
    species_names.append("mutated_species")

    assert model.family_names == ["family_a"]
    assert model.species_names == ["species_a"]


def test_workflow_metadata_checkpoint_progress_normalizes_and_validates(
    tmp_path: Path,
):
    path = tmp_path / "checkpoint.pt"

    assert checkpoint_progress(path, {"step": 2.0, "next_step": 3}) == (2, 3)

    with pytest.raises(RuntimeError, match="inconsistent progress metadata"):
        checkpoint_progress(path, {"step": 2, "next_step": 4})


def test_workflow_metadata_checkpoint_status_defaults_and_validates(
    tmp_path: Path,
):
    path = tmp_path / "checkpoint.pt"

    assert checkpoint_status_dict(path, {}) == {}
    assert checkpoint_status_dict(path, {"status": None}) == {}
    assert checkpoint_status_dict(path, {"status": {"best_step": 2}}) == {
        "best_step": 2
    }

    with pytest.raises(RuntimeError, match="invalid status metadata"):
        checkpoint_status_dict(path, {"status": "not-a-dict"})


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
    ("field", "message"),
    [
        ("checkpoint", "checkpoint must be a path string"),
        ("out_dir", "out_dir must be a path string"),
        ("backtrack_binary", "backtrack_binary must be a path string"),
    ],
)
def test_sampling_config_rejects_non_path_constructor_fields(
    tmp_path: Path,
    field: str,
    message: str,
):
    kwargs = {"checkpoint": tmp_path / "checkpoints" / "best.pt", field: 42}

    with pytest.raises(ValueError, match=message):
        SamplingConfig(**kwargs)


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
    with pytest.raises(ValueError, match="compression"):
        backtracking.sample_recphyloxmls_to_dir(
            model,
            num_samples=1,
            output_dir="out",
            compression="zip",
        )  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="parallel"):
        backtracking.sample_recphyloxmls_to_dir(
            model,
            num_samples=1,
            output_dir="out",
            parallel="false",
        )  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="parallel"):
        backtracking.sample_backtracking_summaries(
            model,
            num_samples=1,
            parallel="false",
        )  # type: ignore[arg-type]


def test_public_backtracking_accepts_integral_real_limits(
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
    calls: list[tuple[str, int]] = []
    model = SimpleNamespace(
        mode="genewise",
        n_species=2,
        species_names=["s0", "s1"],
        species_tree_path=species_tree,
    )

    def family_input(family_index: int) -> FamilyInput:
        calls.append(("family_input", family_index))
        return family

    def activate_family(family_index: int) -> SimpleNamespace:
        calls.append(("activate_family", family_index))
        return SimpleNamespace(clade_offset=0, local_family_index=1)

    model.family_input = family_input
    model.activate_family = activate_family
    state = ReconciliationState(
        e=torch.zeros((2, 2), dtype=torch.float64),
        pi=torch.zeros((2, 2), dtype=torch.float64),
        pibar=torch.zeros((2, 2), dtype=torch.float64),
        ebar=torch.zeros((2, 2), dtype=torch.float64),
        log_p_s=torch.zeros(2, dtype=torch.float64),
        log_p_d=torch.zeros(2, dtype=torch.float64),
        log_p_l=torch.zeros(2, dtype=torch.float64),
        max_transfer=torch.zeros(2, dtype=torch.float64),
        origination_probs=None,
    )

    monkeypatch.setattr(backtracking, "_evaluate_backtracking_state", lambda _: state)

    payload = export_backtracking_input(
        model,
        family_index=1.0,  # type: ignore[arg-type]
        seed=7.0,  # type: ignore[arg-type]
        max_events=9.0,  # type: ignore[arg-type]
    )

    assert calls == [("family_input", 1), ("activate_family", 1)]
    assert payload["seed"] == 7
    assert payload["max_events"] == 9


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
        pibar=torch.zeros((2, 2), dtype=torch.float64),
        ebar=torch.zeros(2, dtype=torch.float64),
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
        pibar=torch.zeros((2, 2), dtype=torch.float64),
        ebar=torch.tensor([[-3.0, -3.1], [-4.0, -4.1]], dtype=torch.float64),
        log_p_s=torch.tensor([-10.0, -20.0], dtype=torch.float64),
        log_p_d=torch.tensor([-30.0, -40.0], dtype=torch.float64),
        log_p_l=torch.tensor([-50.0, -60.0], dtype=torch.float64),
        max_transfer=torch.tensor([-70.0, -80.0], dtype=torch.float64),
        origination_probs=None,
    )

    monkeypatch.setattr(backtracking, "_evaluate_backtracking_state", lambda _: state)

    payload = export_backtracking_input(model, family_index=1)  # type: ignore[arg-type]

    assert payload["e"] == [-2.0, -2.1]
    assert payload["ebar"] == [-4.0, -4.1]
    assert payload["log_p_s"] == [-20.0, -20.0]
    assert payload["log_p_d"] == [-40.0, -40.0]
    assert payload["max_transfer"] == [-80.0, -80.0]


def _fake_backtracking_model(tmp_path: Path, monkeypatch):
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
        mode="global",
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
        e=torch.tensor([1.0, 1.1], dtype=torch.float64),
        pi=torch.tensor([[2.0, 2.1], [2.2, 2.3]], dtype=torch.float64),
        pibar=torch.tensor([[3.0, 3.1], [3.2, 3.3]], dtype=torch.float64),
        ebar=torch.tensor([4.0, 4.1], dtype=torch.float64),
        log_p_s=torch.tensor([5.0, 5.1], dtype=torch.float64),
        log_p_d=torch.tensor([6.0, 6.1], dtype=torch.float64),
        log_p_l=torch.tensor([7.0, 7.1], dtype=torch.float64),
        max_transfer=torch.tensor([8.0, 8.1], dtype=torch.float64),
        origination_probs=None,
    )
    monkeypatch.setattr(backtracking, "_evaluate_backtracking_state", lambda _: state)
    return model


def test_sample_recphyloxmls_to_dir_native_passes_pibar_ebar_and_options(
    tmp_path: Path,
    monkeypatch,
):
    model = _fake_backtracking_model(tmp_path, monkeypatch)
    calls: list[tuple[object, ...]] = []

    class NativeModule:
        def sample_recphyloxmls_to_dir_torch(self, *args: object):
            calls.append(args)
            return [
                {
                    "seed": args[17],
                    "event_counts": {key: 0 for key in EVENT_KEYS},
                    "log_probability": 0.0,
                }
            ]

    monkeypatch.setattr(backtracking, "_load_native_module", lambda _manifest: NativeModule())
    output_dir = tmp_path / "samples"

    result = backtracking.sample_recphyloxmls_to_dir(
        model,  # type: ignore[arg-type]
        family_index=0,
        num_samples=2,
        output_dir=output_dir,
        seed=3,
        max_events=99,
        compression="gzip",
        parallel=False,
    )

    assert result[0]["seed"] == 3
    args = calls[0]
    assert args[9].tolist() == [[2.0, 2.1], [2.2, 2.3]]
    assert args[10].tolist() == [[3.0, 3.1], [3.2, 3.3]]
    assert args[11].tolist() == [1.0, 1.1]
    assert args[12].tolist() == [4.0, 4.1]
    assert args[16] == str(output_dir)
    assert args[17] == 3
    assert args[18] == 2
    assert args[19] == 99
    assert args[20] is None
    assert args[21] is False
    assert args[22] == "gzip"


def test_sample_backtracking_summaries_native_passes_expected_arguments(
    tmp_path: Path,
    monkeypatch,
):
    model = _fake_backtracking_model(tmp_path, monkeypatch)
    calls: list[tuple[object, ...]] = []

    class NativeModule:
        def sample_summaries_torch(self, *args: object):
            calls.append(args)
            return [
                {
                    "seed": args[16],
                    "event_counts": {key: 0 for key in EVENT_KEYS},
                    "log_probability": -1.5,
                }
            ]

    monkeypatch.setattr(backtracking, "_load_native_module", lambda _manifest: NativeModule())

    result = backtracking.sample_backtracking_summaries(
        model,  # type: ignore[arg-type]
        family_index=0,
        num_samples=2,
        seed=5,
        max_events=77,
        parallel=True,
    )

    assert result[0]["log_probability"] == -1.5
    args = calls[0]
    assert args[9].tolist() == [[2.0, 2.1], [2.2, 2.3]]
    assert args[10].tolist() == [[3.0, 3.1], [3.2, 3.3]]
    assert args[16] == 5
    assert args[17] == 2
    assert args[18] == 77
    assert args[19] is None
    assert args[20] is True


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


def test_ensure_backtracking_available_validates_help(
    tmp_path: Path,
    monkeypatch,
):
    calls: list[tuple[list[str], dict[str, object]]] = []
    manifest = tmp_path / "Cargo.toml"
    binary = tmp_path / "gpurec-backtrack"
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    binary.chmod(0o755)

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append((command, kwargs))
        return SimpleNamespace(
            returncode=0,
            stdout=(
                "usage: gpurec-backtrack "
                "[--samples N --output-dir DIR --seed SEED --max-events N] "
                "[input.json] [output.xml]\n"
            ),
            stderr="",
        )

    monkeypatch.setattr(backtracking.subprocess, "run", fake_run)

    backtracking.ensure_backtracking_available(binary, cargo_manifest=manifest)

    assert len(calls) == 1
    assert calls[0][0] == [str(binary.resolve()), "--help"]
    assert calls[0][1]["capture_output"] is True
    assert calls[0][1]["check"] is False
    assert calls[0][1]["cwd"] is None
    assert calls[0][1]["text"] is True
    assert calls[0][1]["timeout"] == backtracking._BACKTRACK_HELP_TIMEOUT_SECONDS


def test_ensure_backtracking_available_rejects_stale_help_missing_wrapper_flags(
    tmp_path: Path,
    monkeypatch,
):
    binary = tmp_path / "gpurec-backtrack"
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    binary.chmod(0o755)

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        assert command == [str(binary.resolve()), "--help"]
        return SimpleNamespace(
            returncode=0,
            stdout="usage: gpurec-backtrack [--samples N] [input.json]\n",
            stderr="",
        )

    monkeypatch.setattr(backtracking.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="missing: .*--seed.*--max-events") as exc_info:
        backtracking.ensure_backtracking_available(binary)

    assert "--output-dir" in str(exc_info.value)


def test_ensure_backtracking_available_rejects_unrelated_executable(
    tmp_path: Path,
    monkeypatch,
):
    binary = tmp_path / "not-backtracking"
    binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    binary.chmod(0o755)

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        assert command == [str(binary.resolve()), "--help"]
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(backtracking.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="expected gpurec-backtrack help"):
        backtracking.ensure_backtracking_available(binary)


def test_ensure_backtracking_available_reports_help_failure(
    tmp_path: Path,
    monkeypatch,
):
    binary = tmp_path / "gpurec-backtrack"
    binary.write_text("#!/bin/sh\nexit 2\n", encoding="utf-8")
    binary.chmod(0o755)

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        assert command == [str(binary.resolve()), "--help"]
        return SimpleNamespace(returncode=2, stdout="", stderr="bad flag\n")

    monkeypatch.setattr(backtracking.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="validating --help") as exc_info:
        backtracking.ensure_backtracking_available(binary)

    assert "bad flag" in str(exc_info.value)


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
        num_samples=2.0,  # type: ignore[arg-type]
        seed=11.0,  # type: ignore[arg-type]
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


def test_backtracking_runner_reports_subprocess_timeout(
    tmp_path: Path,
    monkeypatch,
):
    calls: list[tuple[list[str], dict[str, object]]] = []
    command = ["gpurec-backtrack"]

    def fake_run(run_command: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append((run_command, kwargs))
        raise subprocess.TimeoutExpired(run_command, timeout=kwargs.get("timeout"))

    monkeypatch.setattr(backtracking, "_backtrack_command", lambda **_: command)
    monkeypatch.setattr(backtracking.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="timed out") as exc_info:
        backtracking._run_backtracking_payload(
            {"value": 1},
            cargo_manifest=tmp_path / "missing" / "Cargo.toml",
            backtrack_binary=None,
            build_args=lambda *_: ["input.json", "sample.xml"],
            read_output=lambda _: None,
        )

    message = str(exc_info.value)
    assert "gpurec backtracking command timed out" in message
    assert "gpurec-backtrack input.json sample.xml" in message
    assert "Traceback" not in message
    assert calls[0][1]["timeout"] == backtracking._BACKTRACK_RUN_TIMEOUT_SECONDS


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


def _write_fake_recphyloxml_dir(output_dir: Path, num_samples: int, xml: str) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    for sample_index in range(num_samples):
        (output_dir / f"sample_{sample_index}.xml").write_text(xml, encoding="utf-8")
    return []


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

    def fake_sample_recphyloxmls_to_dir(
        model_arg: object,
        *,
        family_index: int,
        num_samples: int,
        output_dir: Path,
        seed: int,
        max_events: int | None,
        backtrack_binary: Path | None,
    ) -> list[dict]:
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
        return _write_fake_recphyloxml_dir(output_dir, num_samples, xml)

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
        "sample_recphyloxmls_to_dir",
        fake_sample_recphyloxmls_to_dir,
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
        "family_start": 1,
        "family_stop": 3,
        "out_dir": str(config.out_dir),
        "samples_per_family": 2,
        "xml_files": 4,
    }


def test_sampling_runner_windowed_success_replaces_generated_output_set(
    tmp_path: Path,
    monkeypatch,
):
    xml = """
    <recPhylo>
      <recGeneTree>
        <phylogeny>
          <clade>
            <eventsRec><speciation speciesLocation="A"/></eventsRec>
            <clade><eventsRec><leaf speciesLocation="A"/></eventsRec></clade>
          </clade>
        </phylogeny>
      </recGeneTree>
    </recPhylo>
    """

    class FakeModel:
        family_names = ["fam0", "fam1", "fam2"]

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
        samples=1,
        family_start=1,
        max_families=1,
    )
    recon_dir = config.out_dir / "reconciliations"
    all_dir = recon_dir / "all"
    all_dir.mkdir(parents=True)
    previous_sample = all_dir / "000000_previous_sample_0.xml"
    previous_counts = all_dir / "000000_previous_eventCounts_0.txt"
    manual_file = all_dir / "manual.keep"
    previous_sample.write_text("previous sample", encoding="utf-8")
    previous_counts.write_text("previous counts", encoding="utf-8")
    manual_file.write_text("manual", encoding="utf-8")
    (recon_dir / "summary.json").write_text("previous summary", encoding="utf-8")

    sampled_families: list[int] = []

    def fake_sample_recphyloxmls_to_dir(
        model_arg: object,
        *,
        family_index: int,
        num_samples: int,
        output_dir: Path,
        seed: int,
        max_events: int | None,
        backtrack_binary: Path | None,
    ) -> list[dict]:
        assert model_arg is model
        sampled_families.append(family_index)
        return _write_fake_recphyloxml_dir(output_dir, num_samples, xml)

    runner = SamplingRunner(config)
    monkeypatch.setattr(
        sampling_workflow,
        "ensure_backtracking_available",
        lambda backtrack_binary: None,
    )
    monkeypatch.setattr(runner, "_load_model", lambda: (run_config, model))
    monkeypatch.setattr(
        sampling_workflow,
        "sample_recphyloxmls_to_dir",
        fake_sample_recphyloxmls_to_dir,
    )

    runner.run()

    assert sampled_families == [1]
    assert model.closed
    assert not previous_sample.exists()
    assert not previous_counts.exists()
    assert manual_file.read_text(encoding="utf-8") == "manual"
    assert sorted(path.name for path in all_dir.glob("*_sample_*.xml")) == [
        "000001_fam1_sample_0.xml"
    ]
    assert sorted(path.name for path in all_dir.glob("*_eventCounts_*.txt")) == [
        "000001_fam1_eventCounts_0.txt"
    ]
    event_rows = (recon_dir / "event_counts.tsv").read_text(encoding="utf-8").splitlines()
    assert event_rows[1].startswith("fam1\t0\t")
    summary = json.loads((recon_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["families_sampled"] == 1
    assert summary["family_start"] == 1
    assert summary["family_stop"] == 2


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


def test_sampling_runner_rejects_checkpoint_without_family_identity(
    tmp_path: Path,
    monkeypatch,
):
    class FakeModel:
        def __init__(self):
            self.theta = torch.nn.Parameter(torch.zeros(2, 3, dtype=torch.float32))
            self.family_names = ["fam_a", "fam_b"]
            self.species_names = ["sp0", "sp1"]
            self.closed = False

        def clear(self):
            raise AssertionError("theta should not be restored from legacy metadata")

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
        }

    monkeypatch.setattr(sampling_workflow, "load_checkpoint", fake_load_checkpoint)
    monkeypatch.setattr(
        sampling_workflow,
        "build_alerax_workflow_model",
        lambda *_args, **_kwargs: model,
    )

    runner = SamplingRunner(config)
    with pytest.raises(RuntimeError, match="family_names"):
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

    def fail_sample_recphyloxmls_to_dir(*_args: object, **_kwargs: object) -> list[dict]:
        raise RuntimeError("backtrack failed")

    monkeypatch.setattr(
        sampling_workflow,
        "sample_recphyloxmls_to_dir",
        fail_sample_recphyloxmls_to_dir,
    )

    with pytest.raises(RuntimeError, match="backtrack failed") as excinfo:
        runner.run()

    assert model.close_calls == 1
    assert isinstance(excinfo.value.__cause__, RuntimeError)
    assert str(excinfo.value.__cause__) == "close failed"


def test_sampling_runner_preserves_previous_outputs_after_sampling_error(
    tmp_path: Path,
    monkeypatch,
):
    xml = """
    <recPhylo>
      <recGeneTree>
        <phylogeny>
          <clade>
            <eventsRec><speciation speciesLocation="A"/></eventsRec>
            <clade><eventsRec><leaf speciesLocation="A"/></eventsRec></clade>
          </clade>
        </phylogeny>
      </recGeneTree>
    </recPhylo>
    """

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
        samples=1,
    )
    recon_dir = config.out_dir / "reconciliations"
    all_dir = recon_dir / "all"
    all_dir.mkdir(parents=True)
    manual_file = all_dir / "manual.keep"
    stale_sample = all_dir / "000000_stale_sample_0.xml"
    stale_event_counts = all_dir / "000000_stale_eventCounts_0.txt"
    stale_aggregates = [
        recon_dir / "event_counts.tsv",
        recon_dir / "summary.json",
        recon_dir / "totalSpeciesEventCounts.txt",
        recon_dir / "totalTransfers.txt",
    ]
    manual_file.write_text("keep", encoding="utf-8")
    stale_sample.write_text("stale", encoding="utf-8")
    stale_event_counts.write_text("stale counts", encoding="utf-8")
    for stale_aggregate in stale_aggregates:
        stale_aggregate.write_text("stale aggregate", encoding="utf-8")

    calls: list[int] = []

    def fake_sample_recphyloxmls_to_dir(
        model_arg: object,
        *,
        family_index: int,
        num_samples: int,
        output_dir: Path,
        seed: int,
        max_events: int | None,
        backtrack_binary: Path | None,
    ) -> list[dict]:
        assert model_arg is model
        calls.append(family_index)
        if family_index == 0:
            return _write_fake_recphyloxml_dir(output_dir, num_samples, xml)
        raise RuntimeError("backtrack failed on fam1")

    runner = SamplingRunner(config)
    monkeypatch.setattr(
        sampling_workflow,
        "ensure_backtracking_available",
        lambda backtrack_binary: None,
    )
    monkeypatch.setattr(runner, "_load_model", lambda: (run_config, model))
    monkeypatch.setattr(
        sampling_workflow,
        "sample_recphyloxmls_to_dir",
        fake_sample_recphyloxmls_to_dir,
    )

    with pytest.raises(RuntimeError, match="backtrack failed on fam1"):
        runner.run()

    assert calls == [0, 1]
    assert model.closed
    assert manual_file.read_text(encoding="utf-8") == "keep"
    assert stale_sample.read_text(encoding="utf-8") == "stale"
    assert stale_event_counts.read_text(encoding="utf-8") == "stale counts"
    for stale_aggregate in stale_aggregates:
        assert stale_aggregate.read_text(encoding="utf-8") == "stale aggregate"
    assert not (all_dir / "000000_fam0_sample_0.xml").exists()
    assert not (all_dir / "000000_fam0_eventCounts_0.txt").exists()
    assert list(recon_dir.glob(".gpurec-sampling-*")) == []


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

    def unexpected_sample_recphyloxmls_to_dir(*_args: object, **_kwargs: object) -> list[dict]:
        raise AssertionError("sampling should not run")

    monkeypatch.setattr(
        sampling_workflow,
        "sample_recphyloxmls_to_dir",
        unexpected_sample_recphyloxmls_to_dir,
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
    assert payload["route_metadata"] == effective_route_metadata(config)
    assert payload["route_metadata"]["objective"] == "negative_log_likelihood_bits"
    assert payload["route_metadata"]["gradient_route"] == "implicit_first_order_adjoint"
    assert payload["route_metadata"]["optimizer"] == "hessian-sgd"
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


def test_checkpoint_compatibility_rejects_route_metadata_mismatch(tmp_path: Path):
    config = RunConfig(
        species_tree=tmp_path / "sp.nwk",
        families_file=tmp_path / "families.txt",
        out_dir=tmp_path / "out",
        mode="genewise",
        device="cpu",
        optimizer="hessian-sgd",
    )
    checkpoint_config = RunConfig(
        species_tree=config.species_tree,
        families_file=config.families_file,
        out_dir=tmp_path / "checkpoint-out",
        mode="genewise",
        device="cpu",
        optimizer="adam",
    )
    payload = {
        "config": checkpoint_config.to_dict(),
        "route_metadata": effective_route_metadata(checkpoint_config),
        "family_names": ["a", "b"],
        "species_names": ["s0", "s1"],
    }

    with pytest.raises(RuntimeError, match=r"route_metadata\.optimizer differs"):
        validate_checkpoint_model_compatibility(
            path=tmp_path / "latest.pt",
            config=config,
            model=_DummyModel(),
            payload=payload,
        )


def test_checkpoint_compatibility_allows_changed_step_cap_for_resume(
    tmp_path: Path,
):
    checkpoint_config = RunConfig(
        species_tree=tmp_path / "sp.nwk",
        families_file=tmp_path / "families.txt",
        out_dir=tmp_path / "checkpoint-out",
        mode="genewise",
        device="cpu",
        optimizer="hessian-sgd",
        steps=1,
    )
    resumed_config = RunConfig(
        species_tree=checkpoint_config.species_tree,
        families_file=checkpoint_config.families_file,
        out_dir=tmp_path / "resumed-out",
        mode="genewise",
        device="cpu",
        optimizer="hessian-sgd",
        steps=5,
    )
    payload = {
        "config": checkpoint_config.to_dict(),
        "route_metadata": effective_route_metadata(checkpoint_config),
        "family_names": ["a", "b"],
        "species_names": ["s0", "s1"],
    }

    validate_checkpoint_model_compatibility(
        path=tmp_path / "latest.pt",
        config=resumed_config,
        model=_DummyModel(),
        payload=payload,
    )


def test_checkpoint_compatibility_allows_legacy_metadata_without_route(
    tmp_path: Path,
):
    config = RunConfig(
        species_tree=tmp_path / "sp.nwk",
        families_file=tmp_path / "families.txt",
        out_dir=tmp_path / "out",
        mode="genewise",
        device="cpu",
        optimizer="hessian-sgd",
    )
    checkpoint_config = RunConfig(
        species_tree=config.species_tree,
        families_file=config.families_file,
        out_dir=tmp_path / "checkpoint-out",
        mode="genewise",
        device="cpu",
        optimizer="adam",
    )
    payload = {
        "config": checkpoint_config.to_dict(),
        "family_names": ["a", "b"],
        "species_names": ["s0", "s1"],
    }

    validate_checkpoint_model_compatibility(
        path=tmp_path / "legacy.pt",
        config=config,
        model=_DummyModel(),
        payload=payload,
    )


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
            "step": 0,
            "next_step": 1,
            "config": {
                "species_tree": str(tmp_path / "sp.nwk"),
                "families_file": str(tmp_path / "families.txt"),
                "out_dir": str(tmp_path / "out"),
                "mode": "genewise",
                "start": 0,
                "max_families": None,
            },
            "theta": torch.zeros(3),
            "optimizer_state": None,
            "status": {},
            "family_names": [],
            "species_names": [],
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


def test_checkpoint_load_wraps_os_errors_with_context(tmp_path: Path):
    checkpoint = tmp_path / "missing.pt"

    with pytest.raises(RuntimeError) as exc_info:
        load_checkpoint(checkpoint)

    assert "could not safely load checkpoint" in str(exc_info.value)
    assert str(checkpoint) in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, FileNotFoundError)


def test_checkpoint_load_rejects_invalid_route_metadata(tmp_path: Path):
    path = tmp_path / "invalid_route.pt"
    torch.save(
        {
            "version": CHECKPOINT_VERSION,
            "step": 0,
            "next_step": 1,
            "config": {
                "species_tree": str(tmp_path / "sp.nwk"),
                "families_file": str(tmp_path / "families.txt"),
                "out_dir": str(tmp_path / "out"),
                "mode": "genewise",
                "start": 0,
                "max_families": None,
            },
            "route_metadata": "not-a-dict",
            "theta": torch.zeros(3),
            "family_names": [],
            "species_names": [],
        },
        path,
    )

    with pytest.raises(RuntimeError, match="invalid route_metadata"):
        load_checkpoint(path)


@pytest.mark.parametrize(
    "version",
    [CHECKPOINT_VERSION + 1, "next", "1", True, 1.0, 1.5],
)
def test_checkpoint_load_rejects_unsupported_version(tmp_path: Path, version):
    path = tmp_path / "future.pt"
    torch.save(
        {
            "version": version,
            "step": 0,
            "next_step": 1,
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
            "step": 0,
            "next_step": 1,
            "config": {
                "species_tree": str(tmp_path / "sp.nwk"),
                "families_file": str(tmp_path / "families.txt"),
                "out_dir": str(tmp_path / "out"),
                "mode": "genewise",
                "start": 0,
                "max_families": None,
            },
            "theta": theta,
            "family_names": [],
            "species_names": [],
        },
        path,
    )

    with pytest.raises(RuntimeError, match=message):
        load_checkpoint(path)


@pytest.mark.parametrize(
    ("progress", "message"),
    [
        ({"next_step": 1}, "missing key"),
        ({"step": 0}, "missing key"),
        ({"step": 5, "next_step": 0}, "inconsistent progress metadata"),
        ({"step": 0, "next_step": 2}, "inconsistent progress metadata"),
        ({"step": True, "next_step": 1}, "invalid step"),
        ({"step": -1, "next_step": 0}, "invalid step"),
        ({"step": 1.5, "next_step": 2}, "invalid step"),
        ({"step": math.nan, "next_step": 1}, "invalid step"),
        ({"step": math.inf, "next_step": 1}, "invalid step"),
        ({"step": 0, "next_step": True}, "invalid next_step"),
        ({"step": 0, "next_step": 1.5}, "invalid next_step"),
        ({"step": 0, "next_step": math.nan}, "invalid next_step"),
        ({"step": 0, "next_step": math.inf}, "invalid next_step"),
    ],
)
def test_checkpoint_load_rejects_invalid_progress_metadata(
    tmp_path: Path,
    progress: dict[str, object],
    message: str,
):
    path = tmp_path / "invalid_progress.pt"
    torch.save(
        {
            "version": CHECKPOINT_VERSION,
            "config": {
                "species_tree": str(tmp_path / "sp.nwk"),
                "families_file": str(tmp_path / "families.txt"),
                "out_dir": str(tmp_path / "out"),
                "mode": "genewise",
                "start": 0,
                "max_families": None,
            },
            "theta": torch.zeros(3),
            "family_names": [],
            "species_names": [],
            **progress,
        },
        path,
    )

    with pytest.raises(RuntimeError, match=message):
        load_checkpoint(path)


@pytest.mark.parametrize("missing_key", ["family_names", "species_names"])
def test_checkpoint_load_rejects_missing_identity_metadata(
    tmp_path: Path,
    missing_key: str,
):
    payload = {
        "version": CHECKPOINT_VERSION,
        "step": 0,
        "next_step": 1,
        "config": {
            "species_tree": str(tmp_path / "sp.nwk"),
            "families_file": str(tmp_path / "families.txt"),
            "out_dir": str(tmp_path / "out"),
            "mode": "genewise",
            "start": 0,
            "max_families": None,
        },
        "theta": torch.zeros(3),
        "family_names": [],
        "species_names": [],
    }
    payload.pop(missing_key)
    path = tmp_path / "missing_identity.pt"
    torch.save(payload, path)

    with pytest.raises(RuntimeError, match=missing_key):
        load_checkpoint(path)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("family_names", ["fam0", 1]),
        ("family_names", "fam0"),
        ("species_names", ["sp0", None]),
        ("species_names", ("sp0",)),
    ],
)
def test_checkpoint_load_rejects_invalid_identity_metadata(
    tmp_path: Path,
    key: str,
    value: object,
):
    payload = {
        "version": CHECKPOINT_VERSION,
        "step": 0,
        "next_step": 1,
        "config": {
            "species_tree": str(tmp_path / "sp.nwk"),
            "families_file": str(tmp_path / "families.txt"),
            "out_dir": str(tmp_path / "out"),
            "mode": "genewise",
            "start": 0,
            "max_families": None,
        },
        "theta": torch.zeros(3),
        "family_names": [],
        "species_names": [],
        key: value,
    }
    path = tmp_path / "invalid_identity.pt"
    torch.save(payload, path)

    with pytest.raises(RuntimeError, match=key):
        load_checkpoint(path)


def test_checkpoint_load_rejects_missing_config_identity_metadata(tmp_path: Path):
    path = tmp_path / "missing_config_identity.pt"
    torch.save(
        {
            "version": CHECKPOINT_VERSION,
            "step": 0,
            "next_step": 1,
            "config": {
                "species_tree": str(tmp_path / "sp.nwk"),
                "families_file": str(tmp_path / "families.txt"),
                "out_dir": str(tmp_path / "out"),
            },
            "theta": torch.zeros(3),
            "family_names": [],
            "species_names": [],
        },
        path,
    )

    with pytest.raises(RuntimeError, match="config.*identity"):
        load_checkpoint(path)


def test_checkpoint_load_rejects_raw_theta_export(tmp_path: Path):
    path = tmp_path / "theta_final.pt"
    torch.save(torch.zeros(2, 3), path)

    with pytest.raises(RuntimeError, match="must contain a dictionary payload"):
        load_checkpoint(path)


class _WorkflowOptimizerModeModel:
    def __init__(self):
        self.theta = torch.nn.Parameter(
            torch.tensor([0.50, -0.25, 0.125], dtype=torch.float32)
        )
        self.initial_theta = self.theta.detach().clone()
        self.family_names: list[str] = []
        self.species_names = ["sp0", "sp1"]
        self.n_families = 0
        self.n_species = 2
        self.batch_metadata: list[SimpleNamespace] = []
        self.clears = 0
        self.closed = False

    def full_loss(self):
        return self.theta.square().sum() + 1.0

    def full_nll_per_family(self):
        return torch.empty(0, dtype=self.theta.dtype)

    def clamp_theta_(self, min_rate, max_rate):
        with torch.no_grad():
            self.theta.clamp_(min=-4.0, max=4.0)

    def solver_stat_records(self):
        return []

    def clear(self):
        self.clears += 1

    def close(self):
        self.closed = True


class _WorkflowSpecieswiseOptimizerModeModel(_WorkflowOptimizerModeModel):
    def __init__(self):
        super().__init__()
        self.theta = torch.nn.Parameter(
            torch.tensor(
                [
                    [0.50, -0.25, 0.125],
                    [0.10, 0.20, -0.05],
                ],
                dtype=torch.float32,
            )
        )
        self.initial_theta = self.theta.detach().clone()
        self.species_names = ["sp0", "sp1"]
        self.n_species = 2
        self.solver_configs: list[dict[str, object]] = []

    def configure_solver_iterations(self, **kwargs):
        self.solver_configs.append(dict(kwargs))


class _WorkflowRejectingProjectedLBFGSModel(_WorkflowSpecieswiseOptimizerModeModel):
    def full_loss_for_theta(self, theta):
        return theta.new_tensor(1.0e9)


class _WorkflowBatchedLBFGSModeModel:
    def __init__(self):
        self.theta = torch.nn.Parameter(
            torch.tensor(
                [
                    [0.50, -0.25, 0.125],
                    [0.10, 0.20, -0.05],
                ],
                dtype=torch.float32,
            )
        )
        self.initial_theta = self.theta.detach().clone()
        self.family_names = ["fam0", "fam1"]
        self.species_names = ["sp0", "sp1"]
        self.n_families = 2
        self.n_species = 2
        self.batch_metadata: list[SimpleNamespace] = [
            SimpleNamespace(batch_index=0, family_indices=(0,)),
            SimpleNamespace(batch_index=1, family_indices=(1,)),
        ]
        self._current_batch_index = 0
        self.solver_configs: list[dict[str, object]] = []
        self.static_state = SimpleNamespace(
            warm_E=object(),
            last_solver_stats={"Pi_wave_iterations": [2]},
        )
        self.clears = 0
        self.drop_cached_static_states_calls = 0
        self.closed = False

    @property
    def current_batch_index(self):
        return self._current_batch_index

    @property
    def current_batch_metadata(self):
        return self.batch_metadata[self._current_batch_index]

    @property
    def cached_static_states(self):
        return [self.static_state]

    def select_batch(self, batch_index):
        self._current_batch_index = int(batch_index)
        return self.current_batch_metadata

    def nll_per_family(self):
        idx = torch.as_tensor(
            self.current_batch_metadata.family_indices,
            dtype=torch.long,
            device=self.theta.device,
        )
        theta = self.theta.index_select(0, idx)
        return theta.square().sum(dim=1) + 1.0

    def configure_solver_iterations(self, **kwargs):
        self.solver_configs.append(dict(kwargs))

    def full_loss(self):
        return self.theta.square().sum() + 2.0

    def full_genewise_nll_and_grad(self, *, need_grad: bool):
        values = self.theta.detach().square().sum(dim=1) + 1.0
        grad = 2.0 * self.theta.detach() if need_grad else None
        return values, grad

    def full_nll_per_family(self):
        values, _grad = self.full_genewise_nll_and_grad(need_grad=False)
        return values

    def clamp_theta_(self, min_rate, max_rate):
        with torch.no_grad():
            self.theta.clamp_(min=-4.0, max=4.0)

    def solver_stat_records(self):
        return []

    def clear(self):
        self.clears += 1

    def drop_cached_static_states(self):
        self.drop_cached_static_states_calls += 1

    def close(self):
        self.closed = True


class _WorkflowBoundedFDNewtonModel(_WorkflowBatchedLBFGSModeModel):
    def __init__(self):
        super().__init__()
        self.theta = torch.nn.Parameter(
            torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [0.10, 0.20, -0.05],
                ],
                dtype=torch.float32,
            )
        )
        self.initial_theta = self.theta.detach().clone()

    def nll_per_family(self):
        idx = torch.as_tensor(
            self.current_batch_metadata.family_indices,
            dtype=torch.long,
            device=self.theta.device,
        )
        theta = self.theta.index_select(0, idx)
        target = torch.full_like(theta, 5.0)
        return (theta - target).square().sum(dim=1) + 1.0

    def full_loss(self):
        target = torch.full_like(self.theta, 5.0)
        return (self.theta - target).square().sum() + 2.0

    def full_genewise_nll_and_grad(self, *, need_grad: bool):
        target = torch.full_like(self.theta.detach(), 5.0)
        values = (self.theta.detach() - target).square().sum(dim=1) + 1.0
        grad = 2.0 * (self.theta.detach() - target) if need_grad else None
        return values, grad

    def clamp_theta_(self, min_rate, max_rate):
        with torch.no_grad():
            self.theta.clamp_(min=math.log2(min_rate), max=math.log2(max_rate))


class _WorkflowAdaptiveRebatchModel(_WorkflowBatchedLBFGSModeModel):
    def __init__(self):
        super().__init__()
        self.batch_metadata = [
            SimpleNamespace(batch_index=0, family_indices=(0, 1)),
        ]
        self.replanned_indices: list[tuple[int, ...]] = []

    def replan_resident_batches(self, family_indices):
        indices = tuple(int(index) for index in family_indices)
        self.replanned_indices.append(indices)
        self.batch_metadata = [
            SimpleNamespace(batch_index=batch_index, family_indices=(index,))
            for batch_index, index in enumerate(indices)
        ]
        self._current_batch_index = 0
        return list(self.batch_metadata)


class _WorkflowAdaptiveRebatchLikelihoodPlateauModel(_WorkflowAdaptiveRebatchModel):
    def __init__(self):
        super().__init__()
        with torch.no_grad():
            self.theta[1].zero_()
        self.initial_theta = self.theta.detach().clone()


class _WorkflowLargeCladeAdaptiveRebatchModel(_WorkflowAdaptiveRebatchModel):
    def __init__(self):
        super().__init__()
        self.batch_metadata = [
            SimpleNamespace(
                batch_index=0,
                family_indices=(0, 1),
                clade_count=500_000,
            ),
        ]


class _WorkflowOptimizerModeRunner(OptimizationRunner):
    def build_model(self):
        self.fake_model = _WorkflowOptimizerModeModel()
        return self.fake_model


class _WorkflowSpecieswiseOptimizerModeRunner(OptimizationRunner):
    def build_model(self):
        self.fake_model = _WorkflowSpecieswiseOptimizerModeModel()
        return self.fake_model


class _WorkflowSpecieswiseAdagradRestartModel(_WorkflowSpecieswiseOptimizerModeModel):
    def __init__(self):
        super().__init__()


class _WorkflowSpecieswiseAdagradRestartRunner(OptimizationRunner):
    def build_model(self):
        self.fake_model = _WorkflowSpecieswiseAdagradRestartModel()
        return self.fake_model


class _WorkflowRejectingProjectedLBFGSRunner(OptimizationRunner):
    def build_model(self):
        self.fake_model = _WorkflowRejectingProjectedLBFGSModel()
        return self.fake_model


class _WorkflowBatchedLBFGSModeRunner(OptimizationRunner):
    def build_model(self):
        self.fake_model = _WorkflowBatchedLBFGSModeModel()
        return self.fake_model


class _WorkflowAdaptiveRebatchRunner(OptimizationRunner):
    def build_model(self):
        self.fake_model = _WorkflowAdaptiveRebatchModel()
        return self.fake_model


class _WorkflowAdaptiveRebatchLikelihoodPlateauRunner(OptimizationRunner):
    def build_model(self):
        self.fake_model = _WorkflowAdaptiveRebatchLikelihoodPlateauModel()
        return self.fake_model


class _WorkflowLargeCladeAdaptiveRebatchRunner(OptimizationRunner):
    def build_model(self):
        self.fake_model = _WorkflowLargeCladeAdaptiveRebatchModel()
        return self.fake_model


def _optimizer_mode_config(
    tmp_path: Path,
    *,
    optimizer: str,
    **overrides: object,
) -> RunConfig:
    values = {
        "species_tree": tmp_path / "sp.nwk",
        "families_file": tmp_path / "families.txt",
        "out_dir": tmp_path / f"out-{optimizer}",
        "mode": "global",
        "device": "cpu",
        "optimizer": optimizer,
        "steps": 1,
        "lr": 0.2,
        "lbfgs_lr": 0.25,
        "lbfgs_max_iter": 1,
        "checkpoint_every": 1,
        "log_every": 10,
        "loss_patience": 0,
        "best_likelihood_patience": 0,
    }
    values.update(overrides)
    return RunConfig(**values)


def _optimizer_mode_history_rows(out_dir: Path):
    lines = (out_dir / "history.jsonl").read_text(encoding="utf-8").splitlines()
    return [
        json.loads(line)
        for line in lines
    ]


def test_optimization_runner_adagrad_mode_records_public_phase(tmp_path: Path):
    config = _optimizer_mode_config(tmp_path, optimizer="adagrad")
    runner = _WorkflowOptimizerModeRunner(config)

    result = runner.run()

    history_rows = _optimizer_mode_history_rows(config.out_dir)
    assert [row["optimizer/phase"] for row in history_rows] == [
        "adagrad",
        "final_eval",
    ]
    assert history_rows[0]["closure_evals"] == 1
    assert history_rows[0]["optimizer/eval_position"] == "pre_step"
    assert history_rows[0]["optimizer/step_applied"] is True
    assert history_rows[0]["theta_step_inf"] > 0.0
    assert torch.linalg.vector_norm(runner.fake_model.theta.detach()) < torch.linalg.vector_norm(
        runner.fake_model.initial_theta
    )
    latest = load_checkpoint(config.out_dir / "checkpoints" / "latest.pt")
    assert latest["optimizer_phase"] == "adagrad"
    assert latest["last_row"]["optimizer/phase"] == "final_eval"
    assert result.status == "not_converged"
    assert runner.fake_model.closed


def test_optimization_runner_adagrad_restarts_specieswise_uses_schedule(
    tmp_path: Path,
):
    config = _optimizer_mode_config(
        tmp_path,
        optimizer="adagrad-restarts",
        mode="specieswise",
        steps=10,
        adagrad_restart_schedule="4:1.0:2,6:0.5:2",
        adagrad_restart_final_check_iters=8,
        final_check_iters=8,
        loss_patience=0,
        best_likelihood_patience=0,
    )
    runner = _WorkflowSpecieswiseAdagradRestartRunner(config)

    result = runner.run()

    history_rows = _optimizer_mode_history_rows(config.out_dir)
    step_rows = history_rows[:-1]
    assert [row["optimizer/phase"] for row in step_rows] == [
        "adagrad-restarts:fixed4_phase1",
        "adagrad-restarts:fixed4_phase1",
        "adagrad-restarts:fixed6_phase2",
        "adagrad-restarts:fixed6_phase2",
    ]
    assert [row["optimizer/adagrad_restart_budget"] for row in step_rows] == [
        4.0,
        4.0,
        6.0,
        6.0,
    ]
    assert [row["optimizer/adagrad_restart_lr"] for row in step_rows] == [
        1.0,
        1.0,
        0.5,
        0.5,
    ]
    assert [row["optimizer/adagrad_restart_phase_step"] for row in step_rows] == [
        0.0,
        1.0,
        0.0,
        1.0,
    ]
    assert [row["optimizer/adagrad_restart_restarted"] for row in step_rows] == [
        True,
        False,
        True,
        False,
    ]
    assert history_rows[-1]["optimizer/phase"] == "final_eval"
    assert result.status == "converged"
    assert result.reason == "adagrad_restart_schedule_complete"
    assert runner.fake_model.solver_configs[:3] == [
        {"fixed_iters_E": 4, "fixed_iters_Pi": 4, "neumann_terms": 4},
        {"fixed_iters_E": 6, "fixed_iters_Pi": 6, "neumann_terms": 6},
        {"fixed_iters_E": 8, "fixed_iters_Pi": 8, "neumann_terms": 8},
    ]
    latest = load_checkpoint(config.out_dir / "checkpoints" / "latest.pt")
    assert latest["optimizer_phase"] == "adagrad-restarts:fixed6_phase2"
    assert latest["last_row"]["optimizer/phase"] == "final_eval"
    assert runner.fake_model.closed


def test_optimization_runner_adagrad_restarts_accepts_split_solver_budgets(
    tmp_path: Path,
):
    config = _optimizer_mode_config(
        tmp_path,
        optimizer="adagrad-restarts",
        mode="specieswise",
        steps=10,
        adagrad_restart_schedule="8/4:1.0:2,16/8/6:0.5:2",
        adagrad_restart_final_check_iters=32,
        final_check_iters=8,
        loss_patience=0,
        best_likelihood_patience=0,
    )
    runner = _WorkflowSpecieswiseAdagradRestartRunner(config)

    result = runner.run()

    history_rows = _optimizer_mode_history_rows(config.out_dir)
    step_rows = history_rows[:-1]
    assert [row["optimizer/phase"] for row in step_rows] == [
        "adagrad-restarts:E8_Pi4_phase1",
        "adagrad-restarts:E8_Pi4_phase1",
        "adagrad-restarts:E16_Pi8_N6_phase2",
        "adagrad-restarts:E16_Pi8_N6_phase2",
    ]
    assert [row["optimizer/adagrad_restart_budget"] for row in step_rows] == [
        4.0,
        4.0,
        8.0,
        8.0,
    ]
    assert [
        (
            row["optimizer/adagrad_restart_fixed_iters_E"],
            row["optimizer/adagrad_restart_fixed_iters_Pi"],
            row["optimizer/adagrad_restart_neumann_terms"],
        )
        for row in step_rows
    ] == [
        (8.0, 4.0, 4.0),
        (8.0, 4.0, 4.0),
        (16.0, 8.0, 6.0),
        (16.0, 8.0, 6.0),
    ]
    assert result.status == "converged"
    assert result.reason == "adagrad_restart_schedule_complete"
    summary = json.loads((config.out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["mode"] == "specieswise"
    assert summary["optimizer"] == "adagrad-restarts"
    assert summary["adagrad_restart_schedule"] == "8/4:1:2,16/8/6:0.5:2"
    assert summary["adagrad_restart_total_steps"] == 4
    assert summary["configured_steps"] == 10
    assert summary["optimizer_step_cap"] == 4
    assert summary["optimizer_step_cap_reason"] == "adagrad_restart_schedule"
    assert summary["adagrad_restart_final_check_iters"] == 32
    assert summary["final_check_iters"] == 32
    assert summary["fixed_iters_pi"] == 16
    assert summary["neumann_terms"] == 16
    assert runner.fake_model.solver_configs[:3] == [
        {"fixed_iters_E": 8, "fixed_iters_Pi": 4, "neumann_terms": 4},
        {"fixed_iters_E": 16, "fixed_iters_Pi": 8, "neumann_terms": 6},
        {"fixed_iters_E": 32, "fixed_iters_Pi": 32, "neumann_terms": 32},
    ]
    assert runner.fake_model.closed


def test_optimization_runner_projected_sgd_specieswise_records_projected_gradient(
    tmp_path: Path,
):
    config = _optimizer_mode_config(
        tmp_path,
        optimizer="projected-sgd",
        mode="specieswise",
    )
    runner = _WorkflowSpecieswiseOptimizerModeRunner(config)

    result = runner.run()

    history_rows = _optimizer_mode_history_rows(config.out_dir)
    assert [row["optimizer/phase"] for row in history_rows] == [
        "projected-sgd",
        "final_eval",
    ]
    assert history_rows[0]["closure_evals"] == 1
    assert history_rows[0]["optimizer/eval_position"] == "pre_step"
    assert history_rows[0]["optimizer/step_applied"] is True
    assert history_rows[0]["grad/projected_inf"] > 0.0
    summary = json.loads((config.out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["final_projected_grad_inf"] == pytest.approx(
        history_rows[-1]["grad/projected_inf"]
    )
    assert history_rows[0]["theta_step_inf"] > 0.0
    assert torch.linalg.vector_norm(runner.fake_model.theta.detach()) < torch.linalg.vector_norm(
        runner.fake_model.initial_theta
    )
    latest = load_checkpoint(config.out_dir / "checkpoints" / "latest.pt")
    assert latest["optimizer_phase"] == "projected-sgd"
    assert latest["last_row"]["optimizer/phase"] == "final_eval"
    assert result.status == "not_converged"
    assert runner.fake_model.closed


def test_projected_gradient_uses_projection_mapping_near_rate_bound(tmp_path: Path):
    config = _optimizer_mode_config(
        tmp_path,
        optimizer="projected-sgd",
        mode="specieswise",
    )
    runner = OptimizationRunner(config)
    upper_bound = math.log2(config.max_rate)
    theta_value = torch.nextafter(
        torch.tensor(upper_bound, dtype=torch.float32),
        torch.tensor(-math.inf, dtype=torch.float32),
    )
    model = SimpleNamespace(theta=torch.nn.Parameter(theta_value.reshape(1)))
    model.theta.grad = torch.tensor([-500.0], dtype=torch.float32)

    projected, projected_inf = runner._projected_grad_inf(
        model,
        lower_bound=math.log2(config.min_rate),
        upper_bound=upper_bound,
    )

    assert projected_inf == pytest.approx(float(projected.abs().amax().cpu()))
    assert projected_inf < 1e-5


def test_optimization_runner_lbfgs_mode_records_public_phase(tmp_path: Path):
    config = _optimizer_mode_config(tmp_path, optimizer="lbfgs")
    runner = _WorkflowOptimizerModeRunner(config)

    result = runner.run()

    history_rows = _optimizer_mode_history_rows(config.out_dir)
    assert [row["optimizer/phase"] for row in history_rows] == [
        "lbfgs",
        "final_eval",
    ]
    assert history_rows[0]["closure_evals"] == 2
    assert history_rows[0]["optimizer/eval_position"] == "post_step"
    assert history_rows[0]["optimizer/step_applied"] is True
    assert history_rows[0]["theta_step_inf"] > 0.0
    assert torch.linalg.vector_norm(runner.fake_model.theta.detach()) < torch.linalg.vector_norm(
        runner.fake_model.initial_theta
    )
    latest = load_checkpoint(config.out_dir / "checkpoints" / "latest.pt")
    assert latest["optimizer_phase"] == "lbfgs"
    assert latest["last_row"]["optimizer/phase"] == "final_eval"
    assert result.status == "not_converged"
    assert runner.fake_model.closed


def test_optimization_runner_projected_lbfgs_specieswise_uses_loss_only_probes(
    tmp_path: Path,
):
    config = _optimizer_mode_config(
        tmp_path,
        optimizer="projected-lbfgs",
        mode="specieswise",
        lbfgs_lr=0.25,
        lbfgs_max_iter=1,
        lbfgs_max_ls=4,
    )
    runner = _WorkflowSpecieswiseOptimizerModeRunner(config)

    result = runner.run()

    history_rows = _optimizer_mode_history_rows(config.out_dir)
    assert [row["optimizer/phase"] for row in history_rows] == [
        "projected-lbfgs",
        "final_eval",
    ]
    assert history_rows[0]["closure_evals"] == 3
    assert history_rows[0]["optimizer/eval_position"] == "post_step"
    assert history_rows[0]["optimizer/step_applied"] is True
    assert history_rows[0]["optimizer/projected_lbfgs_grad_evals"] == 2.0
    assert history_rows[0]["optimizer/projected_lbfgs_loss_evals"] == 1.0
    assert history_rows[0]["optimizer/projected_lbfgs_accepted"] is True
    assert history_rows[0]["grad/projected_inf"] > 0.0
    assert history_rows[0]["theta_step_inf"] > 0.0
    assert torch.linalg.vector_norm(runner.fake_model.theta.detach()) < torch.linalg.vector_norm(
        runner.fake_model.initial_theta
    )
    latest = load_checkpoint(config.out_dir / "checkpoints" / "latest.pt")
    assert latest["optimizer_phase"] == "projected-lbfgs"
    assert latest["last_row"]["optimizer/phase"] == "final_eval"
    assert result.status == "not_converged"
    assert runner.fake_model.closed


def test_optimization_runner_lbfgsb_specieswise_records_kkt_metrics(tmp_path: Path):
    config = _optimizer_mode_config(
        tmp_path,
        optimizer="lbfgsb",
        mode="specieswise",
        lbfgs_lr=1.0,
        lbfgs_max_iter=1,
        lbfgs_max_ls=8,
    )
    runner = _WorkflowSpecieswiseOptimizerModeRunner(config)

    result = runner.run()

    history_rows = _optimizer_mode_history_rows(config.out_dir)
    assert [row["optimizer/phase"] for row in history_rows] == [
        "lbfgsb",
        "final_eval",
    ]
    assert history_rows[0]["optimizer/eval_position"] == "post_step"
    assert history_rows[0]["optimizer/step_applied"] is True
    assert history_rows[0]["optimizer/lbfgsb_grad_evals"] >= 1.0
    assert history_rows[0]["optimizer/lbfgsb_loss_evals"] >= 1.0
    assert history_rows[0]["optimizer/lbfgsb_accepted"] is True
    assert history_rows[0]["optimizer/lbfgsb_direction_kind"] in {
        "cauchy",
        "subspace",
        "projected_gradient",
    }
    assert history_rows[0]["grad/projected_inf"] >= 0.0
    latest = load_checkpoint(config.out_dir / "checkpoints" / "latest.pt")
    assert latest["optimizer_phase"] == "lbfgsb"
    assert latest["last_row"]["optimizer/phase"] == "final_eval"
    assert result.status == "not_converged"
    assert runner.fake_model.closed


def test_optimization_runner_projected_lbfgs_reduces_lr_instead_of_stopping_on_large_projected_gradient(
    tmp_path: Path,
):
    config = _optimizer_mode_config(
        tmp_path,
        optimizer="projected-lbfgs",
        mode="specieswise",
        steps=2,
        lbfgs_lr=0.25,
        lbfgs_max_iter=1,
        lbfgs_max_ls=2,
        loss_change_tol=0.0,
        loss_patience=1,
        best_likelihood_patience=1,
        projected_grad_tol=1e-6,
    )
    runner = _WorkflowRejectingProjectedLBFGSRunner(config)

    result = runner.run()

    history_rows = _optimizer_mode_history_rows(config.out_dir)
    optimizer_rows = [
        row for row in history_rows if row["optimizer/phase"] == "projected-lbfgs"
    ]
    assert len(optimizer_rows) == 2
    assert optimizer_rows[0]["optimizer/projected_lbfgs_accepted"] is False
    assert optimizer_rows[0]["optimizer/projected_lbfgs_lr_reduced"] is True
    assert optimizer_rows[0]["optimizer/projected_lbfgs_lr_before"] == pytest.approx(0.25)
    assert optimizer_rows[0]["optimizer/projected_lbfgs_lr_after"] == pytest.approx(0.125)
    assert optimizer_rows[0]["stable_loss_steps"] == 0
    assert optimizer_rows[1]["delta_likelihood_bits"] == pytest.approx(0.0)
    assert optimizer_rows[1]["optimizer/projected_lbfgs_lr_reduced"] is True
    assert optimizer_rows[1]["optimizer/projected_lbfgs_lr_before"] == pytest.approx(0.125)
    assert optimizer_rows[1]["optimizer/projected_lbfgs_lr_after"] == pytest.approx(0.0625)
    assert optimizer_rows[1]["stable_loss_steps"] == 0
    assert result.status == "not_converged"
    assert result.reason == "max_steps"
    assert runner.fake_model.closed


def test_optimization_runner_projected_lbfgs_reports_min_lr_with_large_projected_gradient(
    tmp_path: Path,
):
    config = _optimizer_mode_config(
        tmp_path,
        optimizer="projected-lbfgs",
        mode="specieswise",
        steps=5,
        lbfgs_lr=0.25,
        lbfgs_max_iter=1,
        lbfgs_max_ls=1,
        loss_change_tol=0.0,
        loss_patience=1,
        best_likelihood_patience=1,
        projected_grad_tol=1e-6,
        projected_lbfgs_min_lr=0.125,
    )
    runner = _WorkflowRejectingProjectedLBFGSRunner(config)

    result = runner.run()

    history_rows = _optimizer_mode_history_rows(config.out_dir)
    optimizer_rows = [
        row for row in history_rows if row["optimizer/phase"] == "projected-lbfgs"
    ]
    assert len(optimizer_rows) == 2
    assert optimizer_rows[0]["optimizer/projected_lbfgs_lr_reduced"] is True
    assert optimizer_rows[0]["optimizer/projected_lbfgs_lr_after"] == pytest.approx(
        0.125
    )
    assert optimizer_rows[1]["optimizer/projected_lbfgs_lr_reduced"] is False
    assert optimizer_rows[1]["optimizer/projected_lbfgs_min_lr_reached"] is True
    assert optimizer_rows[1]["stable_loss_steps"] == 0
    assert result.status == "not_converged"
    assert result.reason == "projected_lbfgs_min_lr_reached"
    assert runner.fake_model.closed


def test_optimization_runner_batched_lbfgs_mode_records_public_phase(tmp_path: Path):
    config = _optimizer_mode_config(
        tmp_path,
        optimizer="batched-lbfgs",
        mode="genewise",
        lbfgs_max_iter=1,
    )
    runner = _WorkflowBatchedLBFGSModeRunner(config)

    result = runner.run()

    history_rows = _optimizer_mode_history_rows(config.out_dir)
    assert [row["optimizer/phase"] for row in history_rows] == [
        "batched-lbfgs",
        "final_eval",
    ]
    assert history_rows[0]["closure_evals"] >= 2
    assert history_rows[0]["optimizer/eval_position"] == "post_step"
    assert history_rows[0]["optimizer/step_applied"] is True
    assert history_rows[0]["optimizer/batched_lbfgs_grad_evals"] >= 1
    assert history_rows[0]["optimizer/batched_lbfgs_loss_evals"] >= 1
    assert history_rows[0]["optimizer/batched_lbfgs_accepted_rows"] > 0
    assert history_rows[0]["theta_step_inf"] > 0.0
    assert torch.linalg.vector_norm(runner.fake_model.theta.detach()) < torch.linalg.vector_norm(
        runner.fake_model.initial_theta
    )
    latest = load_checkpoint(config.out_dir / "checkpoints" / "latest.pt")
    assert latest["optimizer_phase"] == "batched-lbfgs"
    assert latest["last_row"]["optimizer/phase"] == "final_eval"
    assert "optimizer/final_eval_source" not in history_rows[-1]
    assert result.status == "not_converged"
    assert runner.fake_model.closed


def test_optimization_runner_hessian_sgd_mode_records_public_phase(tmp_path: Path):
    config = _optimizer_mode_config(
        tmp_path,
        optimizer="hessian-sgd",
        mode="genewise",
        solver_warmup_iters=0,
        fd_hessian_epsilon=1e-3,
        fd_newton_damping=1e-6,
    )
    runner = _WorkflowBatchedLBFGSModeRunner(config)

    result = runner.run()

    history_rows = _optimizer_mode_history_rows(config.out_dir)
    assert [row["optimizer/phase"] for row in history_rows] == [
        "hessian-sgd",
        "final_eval",
    ]
    assert history_rows[0]["closure_evals"] == 8
    assert history_rows[0]["optimizer/eval_position"] == "post_step"
    assert history_rows[0]["optimizer/step_applied"] is True
    assert history_rows[0]["optimizer/fd_newton_subphase"] == "hessian_sgd"
    assert history_rows[0]["optimizer/fd_newton_hessian_update"] == "bfgs"
    assert history_rows[0]["optimizer/fd_newton_hessian_source"] == "finite_difference"
    assert history_rows[0]["optimizer/fd_newton_line_search"] is False
    assert history_rows[0]["optimizer/fd_newton_post_step_loss_filter"] is True
    assert history_rows[0]["optimizer/fd_newton_loss_evals"] == 0.0
    assert history_rows[0]["optimizer/fd_newton_loss_rejected_rows"] == 0.0
    assert history_rows[0]["optimizer/fd_newton_max_ls"] == 0.0
    assert history_rows[0]["optimizer/fd_newton_bfgs_updated_rows"] == 1.0
    assert history_rows[0]["optimizer/fd_newton_hessian_refresh_steps"] == 16.0
    assert history_rows[0]["optimizer/fd_newton_step_scale"] == pytest.approx(
        config.lr
    )
    assert history_rows[0]["theta_step_inf"] > 0.0
    assert torch.linalg.vector_norm(runner.fake_model.theta.detach()) < torch.linalg.vector_norm(
        runner.fake_model.initial_theta
    )
    latest = load_checkpoint(config.out_dir / "checkpoints" / "latest.pt")
    assert latest["optimizer_phase"] == "hessian-sgd"
    assert latest["last_row"]["optimizer/phase"] == "final_eval"
    assert result.status == "not_converged"
    assert runner.fake_model.closed


def test_hessian_sgd_solver_warmup_keeps_full_e_budget(tmp_path: Path):
    config = _optimizer_mode_config(
        tmp_path,
        optimizer="hessian-sgd",
        mode="genewise",
        fixed_iters_e=16,
        solver_warmup_iters=6,
    )
    runner = OptimizationRunner(config)
    model = _WorkflowBatchedLBFGSModeModel()

    runner._configure_solver_stage(model, "warmup")

    assert model.solver_configs == [
        {"fixed_iters_E": 16, "fixed_iters_Pi": 6, "neumann_terms": 6}
    ]


def test_hessian_sgd_large_batch_warmup_uses_short_pi_neumann_schedule(
    tmp_path: Path,
):
    config = _optimizer_mode_config(
        tmp_path,
        optimizer="hessian-sgd",
        mode="genewise",
        fixed_iters_e=16,
        solver_warmup_iters=4,
    )
    runner = OptimizationRunner(config)
    model = _WorkflowBatchedLBFGSModeModel()
    model.batch_metadata[0].clade_count = 500_000

    runner._configure_solver_stage(model, "warmup")

    assert model.solver_configs == [
        {"fixed_iters_E": 16, "fixed_iters_Pi": 2, "neumann_terms": 2}
    ]


def test_specieswise_full_solver_stage_raises_e_budget_with_high_pi(
    tmp_path: Path,
):
    config = _optimizer_mode_config(
        tmp_path,
        optimizer="projected-lbfgs",
        mode="specieswise",
        fixed_iters_e=6,
        fixed_iters_pi=32,
        neumann_terms=32,
    )
    runner = OptimizationRunner(config)
    model = _WorkflowSpecieswiseOptimizerModeModel()
    solver_configs: list[dict[str, object]] = []

    def configure_solver_iterations(**kwargs):
        solver_configs.append(dict(kwargs))

    model.configure_solver_iterations = configure_solver_iterations

    runner._configure_solver_stage(model, "full")

    assert solver_configs == [
        {"fixed_iters_E": 32, "fixed_iters_Pi": 32, "neumann_terms": 32}
    ]


def test_hessian_sgd_advances_batch_after_full_stage_plateau(tmp_path: Path):
    config = _optimizer_mode_config(
        tmp_path,
        optimizer="hessian-sgd",
        mode="genewise",
        steps=3,
        solver_warmup_iters=0,
        fd_hessian_refresh_steps=16,
        fd_hessian_epsilon=1e-3,
        fd_newton_damping=1e-6,
        loss_change_tol=1e9,
        loss_patience=1,
        best_likelihood_patience=0,
        checkpoint_every=1,
    )
    runner = _WorkflowBatchedLBFGSModeRunner(config)

    result = runner.run()

    history_rows = _optimizer_mode_history_rows(config.out_dir)
    hessian_rows = [
        row for row in history_rows if row["optimizer/phase"] == "hessian-sgd"
    ]
    assert [row["optimizer/fd_newton_subphase"] for row in hessian_rows] == [
        "hessian_sgd",
        "hessian_sgd",
        "hessian_sgd",
    ]
    assert [row["optimizer/batch_index"] for row in hessian_rows] == [0, 0, 1]
    assert all("optimizer/hessian_sgd_polish_active" not in row for row in hessian_rows)
    assert hessian_rows[2]["optimizer/fd_newton_line_search"] is False
    assert hessian_rows[2]["optimizer/fd_newton_post_step_loss_filter"] is True
    assert hessian_rows[0]["optimizer/fd_newton_hessian_refresh_steps"] == 16.0
    assert hessian_rows[1]["optimizer/fd_newton_hessian_refresh_steps"] == 16.0
    assert hessian_rows[2]["optimizer/fd_newton_hessian_refresh_steps"] == 16.0
    assert hessian_rows[2]["optimizer/fd_newton_step_scale"] == pytest.approx(config.lr)
    assert result.status == "not_converged"
    assert runner.fake_model.closed


def test_hessian_sgd_normal_solver_controls_drive_full_stage(
    tmp_path: Path,
):
    config = _optimizer_mode_config(
        tmp_path,
        optimizer="hessian-sgd",
        mode="genewise",
        steps=3,
        solver_warmup_iters=0,
        hessian_sgd_normal_fixed_iters_pi=12,
        hessian_sgd_normal_neumann_terms=12,
        fd_hessian_refresh_steps=16,
        fd_hessian_epsilon=1e-3,
        fd_newton_damping=1e-6,
        loss_change_tol=1e9,
        loss_patience=1,
        best_likelihood_patience=0,
    )
    runner = _WorkflowBatchedLBFGSModeRunner(config)

    result = runner.run()

    history_rows = _optimizer_mode_history_rows(config.out_dir)
    hessian_rows = [
        row for row in history_rows if row["optimizer/phase"] == "hessian-sgd"
    ]
    assert [row["optimizer/fd_newton_subphase"] for row in hessian_rows] == [
        "hessian_sgd",
        "hessian_sgd",
        "hessian_sgd",
    ]
    assert hessian_rows[0]["optimizer/fd_newton_hessian_refresh_steps"] == 16.0
    assert hessian_rows[2]["optimizer/fd_newton_hessian_refresh_steps"] == 16.0
    assert runner.fake_model.solver_configs[:2] == [
        {"fixed_iters_E": None, "fixed_iters_Pi": 12, "neumann_terms": 12},
        {"fixed_iters_E": None, "fixed_iters_Pi": 12, "neumann_terms": 12},
    ]
    assert result.status == "not_converged"
    assert runner.fake_model.closed


def test_final_genewise_eval_falls_back_to_smaller_clade_budget(
    tmp_path: Path,
    monkeypatch,
):
    config = _optimizer_mode_config(
        tmp_path,
        optimizer="hessian-sgd",
        mode="genewise",
        clade_budget=500_000,
    )
    runner = OptimizationRunner(config)
    model = _WorkflowBatchedLBFGSModeModel()

    def fail_full_genewise_nll_and_grad(*, need_grad: bool):
        raise RuntimeError(
            "2D self-loop fast path estimated scratch 0.00 GiB above memory budget"
        )

    model.full_genewise_nll_and_grad = fail_full_genewise_nll_and_grad
    fallback_model = _WorkflowBatchedLBFGSModeModel()
    budgets: list[int | None] = []

    def fake_build_alerax_workflow_model(fallback_config, *, prefetch_batches):
        budgets.append(fallback_config.clade_budget)
        assert prefetch_batches == 1
        return fallback_model

    monkeypatch.setattr(
        optimize_workflow,
        "build_alerax_workflow_model",
        fake_build_alerax_workflow_model,
    )

    loss_vec, metrics = runner._evaluate_genewise_vector_and_grad_with_memory_fallback(
        model
    )

    assert budgets == [250_000]
    assert metrics["optimizer/final_eval_source"] == "fallback_clade_budget"
    assert metrics["optimizer/final_eval_fallback_clade_budget"] == 250_000.0
    assert "2D self-loop fast path" in metrics["optimizer/final_eval_fallback_reason"]
    torch.testing.assert_close(
        loss_vec,
        model.theta.detach().square().sum(dim=1) + 1.0,
    )
    torch.testing.assert_close(model.theta.grad, 2.0 * model.theta.detach())
    assert fallback_model.closed


def test_memory_retryable_error_includes_scratch_budget_guard():
    exc = RuntimeError(
        "Pi_wave_backward fused path requires 2D self-loop scratch "
        "(0.61 GiB requested, 0.20 GiB budget)"
    )

    assert optimize_workflow._is_memory_retryable_runtime_error(exc)


def test_final_genewise_eval_does_not_fallback_for_non_memory_error(
    tmp_path: Path,
    monkeypatch,
):
    config = _optimizer_mode_config(
        tmp_path,
        optimizer="hessian-sgd",
        mode="genewise",
        clade_budget=500_000,
    )
    runner = OptimizationRunner(config)
    model = _WorkflowBatchedLBFGSModeModel()

    def fail_full_genewise_nll_and_grad(*, need_grad: bool):
        raise RuntimeError("logic bug")

    def fake_build_alerax_workflow_model(fallback_config, *, prefetch_batches):
        raise AssertionError("fallback should not be built")

    model.full_genewise_nll_and_grad = fail_full_genewise_nll_and_grad
    monkeypatch.setattr(
        optimize_workflow,
        "build_alerax_workflow_model",
        fake_build_alerax_workflow_model,
    )

    with pytest.raises(RuntimeError, match="logic bug"):
        runner._evaluate_genewise_vector_and_grad_with_memory_fallback(model)


def test_final_iteration_check_falls_back_to_smaller_clade_budget(
    tmp_path: Path,
    monkeypatch,
):
    config = _optimizer_mode_config(
        tmp_path,
        optimizer="hessian-sgd",
        mode="genewise",
        clade_budget=500_000,
        final_check_iters=32,
    )
    runner = OptimizationRunner(config)
    model = _WorkflowBatchedLBFGSModeModel()

    def fail_full_genewise_nll_and_grad(*, need_grad: bool):
        raise RuntimeError(
            "2D self-loop fast path estimated scratch 0.00 GiB above memory budget"
        )

    model.full_genewise_nll_and_grad = fail_full_genewise_nll_and_grad
    fallback_model = _WorkflowBatchedLBFGSModeModel()
    baseline_loss_vec, baseline_grad = fallback_model.full_genewise_nll_and_grad(
        need_grad=True
    )
    assert baseline_grad is not None
    budgets: list[int | None] = []

    def fake_build_alerax_workflow_model(fallback_config, *, prefetch_batches):
        budgets.append(fallback_config.clade_budget)
        assert prefetch_batches == 1
        return fallback_model

    monkeypatch.setattr(
        optimize_workflow,
        "build_alerax_workflow_model",
        fake_build_alerax_workflow_model,
    )

    metrics = runner._evaluate_final_iteration_check(
        model,
        baseline_loss=baseline_loss_vec.sum(),
        baseline_grad=baseline_grad,
    )

    assert budgets == [250_000]
    assert metrics["optimizer/final_check_status"] == "ok"
    assert metrics["optimizer/final_check_source"] == "fallback_clade_budget"
    assert metrics["optimizer/final_check_fallback_clade_budget"] == 250_000.0
    assert "2D self-loop fast path" in metrics["optimizer/final_check_reason"]
    assert "2D self-loop fast path" in metrics["optimizer/final_check_fallback_reason"]
    summary_metrics = optimize_workflow._final_check_summary_metrics(metrics)
    assert summary_metrics["final_check_source"] == "fallback_clade_budget"
    assert summary_metrics["final_check_fallback_clade_budget"] == 250_000.0
    assert "2D self-loop fast path" in summary_metrics["final_check_reason"]
    assert metrics["optimizer/final_check_loss_abs_delta_bits"] == pytest.approx(0.0)
    assert metrics["optimizer/final_check_grad_max_abs_delta"] == pytest.approx(0.0)
    assert fallback_model.closed


def test_hessian_sgd_advances_batch_after_best_likelihood_stall(
    tmp_path: Path,
):
    config = _optimizer_mode_config(
        tmp_path,
        optimizer="hessian-sgd",
        mode="genewise",
        steps=3,
        solver_warmup_iters=0,
        fd_hessian_refresh_steps=16,
        fd_hessian_epsilon=1e-3,
        fd_newton_damping=1e-6,
        loss_patience=0,
        best_likelihood_patience=1,
        best_likelihood_min_delta=1e9,
        checkpoint_every=1,
    )
    runner = _WorkflowBatchedLBFGSModeRunner(config)

    result = runner.run()

    history_rows = _optimizer_mode_history_rows(config.out_dir)
    hessian_rows = [
        row for row in history_rows if row["optimizer/phase"] == "hessian-sgd"
    ]
    assert [row["optimizer/fd_newton_subphase"] for row in hessian_rows] == [
        "hessian_sgd",
        "hessian_sgd",
        "hessian_sgd",
    ]
    assert [row["optimizer/batch_index"] for row in hessian_rows] == [0, 0, 1]
    assert "optimizer/hessian_sgd_polish_active" not in hessian_rows[2]
    assert hessian_rows[2]["optimizer/fd_newton_line_search"] is False
    assert hessian_rows[0]["optimizer/fd_newton_hessian_refresh_steps"] == 16.0
    assert hessian_rows[1]["optimizer/fd_newton_hessian_refresh_steps"] == 16.0
    assert hessian_rows[2]["optimizer/fd_newton_hessian_refresh_steps"] == 16.0
    assert result.status == "not_converged"
    assert runner.fake_model.closed


def test_hessian_sgd_likelihood_plateau_converges_with_nonzero_gradient(
    tmp_path: Path,
):
    config = _optimizer_mode_config(
        tmp_path,
        optimizer="hessian-sgd",
        mode="genewise",
        steps=5,
        solver_warmup_iters=0,
        fd_hessian_refresh_steps=16,
        fd_hessian_epsilon=1e-3,
        fd_newton_damping=1e-6,
        loss_patience=0,
        best_likelihood_patience=1,
        best_likelihood_min_delta=1e9,
    )
    runner = _WorkflowAdaptiveRebatchRunner(config)

    result = runner.run()

    history_rows = _optimizer_mode_history_rows(config.out_dir)
    final_step = history_rows[-2]
    assert final_step["optimizer/fd_newton_subphase"] == "hessian_sgd"
    assert final_step["grad/projected_inf"] > 0.0
    assert result.status == "converged"
    assert result.reason == "best_likelihood_patience"
    assert runner.fake_model.closed


def test_hessian_sgd_plateau_converges_without_refreshing_for_gradient(
    tmp_path: Path,
):
    config = _optimizer_mode_config(
        tmp_path,
        optimizer="hessian-sgd",
        mode="genewise",
        steps=5,
        solver_warmup_iters=0,
        fd_hessian_epsilon=1e-3,
        fd_newton_damping=1e-6,
        loss_change_tol=0.0,
        loss_patience=1,
        best_likelihood_patience=1,
    )
    runner = _WorkflowAdaptiveRebatchRunner(config)
    state_seen_as_none: list[bool] = []

    def fake_step(model, *, solver_stage, hessian_state=None, **_kwargs):
        state_seen_as_none.append(hessian_state is None)
        model.theta.grad = torch.ones_like(model.theta)
        loss_vec = torch.full(
            (int(model.n_families),),
            5.0,
            device=model.theta.device,
            dtype=model.theta.dtype,
        )
        metrics = runner._active_batch_metrics(
            model,
            loss_vec=loss_vec,
            solver_stage=solver_stage,
        )
        refreshed = hessian_state is None
        metrics["optimizer/fd_newton_accepted_rows"] = 0.0
        metrics["optimizer/fd_newton_hessian_refreshed"] = refreshed
        metrics["optimizer/fd_newton_hessian_source"] = (
            "finite_difference" if refreshed else "fixed_hessian"
        )
        return loss_vec, metrics, 1, object()

    runner._active_fd_newton_step = fake_step  # type: ignore[method-assign]

    result = runner.run()

    history_rows = _optimizer_mode_history_rows(config.out_dir)
    hessian_rows = [
        row for row in history_rows if row["optimizer/phase"] == "hessian-sgd"
    ]
    assert state_seen_as_none == [True, False]
    assert [row["optimizer/fd_newton_subphase"] for row in hessian_rows] == [
        "hessian_sgd",
        "hessian_sgd",
    ]
    assert "optimizer/fd_newton_force_refresh_after_plateau" not in hessian_rows[1]
    assert hessian_rows[1]["optimizer/fd_newton_hessian_source"] == "fixed_hessian"
    assert hessian_rows[1]["stable_loss_steps"] == 1
    assert result.status == "converged"
    assert result.reason == "loss_change_patience"
    assert runner.fake_model.closed


def test_hessian_sgd_likelihood_plateau_does_not_refine_on_projected_gradient(
    tmp_path: Path,
):
    config = _optimizer_mode_config(
        tmp_path,
        optimizer="hessian-sgd",
        mode="genewise",
        steps=5,
        solver_warmup_iters=0,
        fd_hessian_epsilon=1e-3,
        fd_newton_damping=1e-6,
        loss_change_tol=0.0,
        loss_patience=1,
        best_likelihood_patience=0,
    )
    runner = _WorkflowAdaptiveRebatchRunner(config)

    def fake_step(model, *, solver_stage, hessian_state=None, **_kwargs):
        model.theta.grad = torch.full_like(model.theta, 0.1)
        loss_vec = torch.full(
            (int(model.n_families),),
            5.0,
            device=model.theta.device,
            dtype=model.theta.dtype,
        )
        metrics = runner._active_batch_metrics(
            model,
            loss_vec=loss_vec,
            solver_stage=solver_stage,
        )
        return loss_vec, metrics, 1, object()

    runner._active_fd_newton_step = fake_step  # type: ignore[method-assign]

    result = runner.run()

    history_rows = _optimizer_mode_history_rows(config.out_dir)
    hessian_rows = [
        row for row in history_rows if row["optimizer/phase"] == "hessian-sgd"
    ]
    assert [row["optimizer/fd_newton_subphase"] for row in hessian_rows] == [
        "hessian_sgd",
        "hessian_sgd",
    ]
    assert all("optimizer/hessian_sgd_polish_active" not in row for row in hessian_rows)
    assert result.status == "converged"
    assert result.reason == "loss_change_patience"
    assert runner.fake_model.closed


def test_hessian_sgd_likelihood_plateau_skips_low_acceptance_line_search(
    tmp_path: Path,
):
    config = _optimizer_mode_config(
        tmp_path,
        optimizer="hessian-sgd",
        mode="genewise",
        steps=5,
        solver_warmup_iters=0,
        fd_hessian_epsilon=1e-3,
        fd_newton_damping=1e-6,
        loss_change_tol=0.0,
        loss_patience=1,
        best_likelihood_patience=0,
    )
    runner = _WorkflowAdaptiveRebatchRunner(config)
    calls: list[tuple[bool, bool, int | None, bool]] = []

    def fake_step(
        model,
        *,
        solver_stage,
        hessian_state=None,
        use_line_search,
        reject_loss_increases_after_step,
        line_search_max_steps,
        **_kwargs,
    ):
        calls.append(
            (
                bool(use_line_search),
                bool(reject_loss_increases_after_step),
                line_search_max_steps,
                hessian_state is None,
            )
        )
        model.theta.grad = torch.ones_like(model.theta)
        loss_vec = torch.full(
            (int(model.n_families),),
            5.0,
            device=model.theta.device,
            dtype=model.theta.dtype,
        )
        metrics = runner._active_batch_metrics(
            model,
            loss_vec=loss_vec,
            solver_stage=solver_stage,
        )
        metrics["optimizer/fd_newton_accepted_fraction"] = 0.0
        metrics["optimizer/fd_newton_loss_rejected_rows"] = float(model.n_families)
        return loss_vec, metrics, 1, object()

    runner._active_fd_newton_step = fake_step  # type: ignore[method-assign]

    result = runner.run()

    assert calls == [
        (False, True, None, True),
        (False, True, None, False),
    ]
    assert result.status == "converged"
    assert result.reason == "loss_change_patience"
    assert runner.fake_model.closed


def test_hessian_sgd_low_acceptance_uses_line_search_before_plateau(
    tmp_path: Path,
):
    config = _optimizer_mode_config(
        tmp_path,
        optimizer="hessian-sgd",
        mode="genewise",
        steps=4,
        solver_warmup_iters=0,
        fd_hessian_epsilon=1e-3,
        fd_newton_damping=1e-6,
        loss_change_tol=0.0,
        loss_patience=1,
        best_likelihood_patience=0,
    )
    runner = _WorkflowAdaptiveRebatchRunner(config)
    calls: list[tuple[bool, bool, int | None, bool]] = []

    def fake_step(
        model,
        *,
        solver_stage,
        hessian_state=None,
        use_line_search,
        reject_loss_increases_after_step,
        line_search_max_steps,
        **_kwargs,
    ):
        call_index = len(calls)
        calls.append(
            (
                bool(use_line_search),
                bool(reject_loss_increases_after_step),
                line_search_max_steps,
                hessian_state is None,
            )
        )
        model.theta.grad = torch.ones_like(model.theta)
        loss_value = 3.0 if use_line_search else 5.0 - call_index
        loss_vec = torch.full(
            (int(model.n_families),),
            loss_value,
            device=model.theta.device,
            dtype=model.theta.dtype,
        )
        metrics = runner._active_batch_metrics(
            model,
            loss_vec=loss_vec,
            solver_stage=solver_stage,
        )
        metrics["optimizer/fd_newton_accepted_fraction"] = (
            1.0 if use_line_search else 0.0
        )
        metrics["optimizer/fd_newton_loss_rejected_rows"] = (
            0.0 if use_line_search else float(model.n_families)
        )
        return loss_vec, metrics, 1, object()

    runner._active_fd_newton_step = fake_step  # type: ignore[method-assign]

    result = runner.run()

    assert calls == [
        (False, True, None, True),
        (False, True, None, False),
        (True, False, 8, True),
        (True, False, 8, False),
    ]
    assert result.status == "converged"
    assert result.reason == "loss_change_patience"
    assert runner.fake_model.closed


def test_hessian_sgd_large_batch_uses_long_refresh_until_line_search(
    tmp_path: Path,
):
    config = _optimizer_mode_config(
        tmp_path,
        optimizer="hessian-sgd",
        mode="genewise",
        steps=4,
        solver_warmup_iters=0,
        fd_hessian_refresh_steps=16,
        fd_hessian_epsilon=1e-3,
        fd_newton_damping=1e-6,
        loss_change_tol=0.0,
        loss_patience=1,
        best_likelihood_patience=0,
    )
    runner = _WorkflowLargeCladeAdaptiveRebatchRunner(config)
    calls: list[tuple[bool, int | None]] = []

    def fake_step(
        model,
        *,
        solver_stage,
        use_line_search,
        hessian_refresh_steps,
        **_kwargs,
    ):
        call_index = len(calls)
        calls.append((bool(use_line_search), hessian_refresh_steps))
        model.theta.grad = torch.ones_like(model.theta)
        loss_value = 6.0 if call_index == 0 else 5.0
        loss_vec = torch.full(
            (int(model.n_families),),
            loss_value,
            device=model.theta.device,
            dtype=model.theta.dtype,
        )
        metrics = runner._active_batch_metrics(
            model,
            loss_vec=loss_vec,
            solver_stage=solver_stage,
        )
        metrics["optimizer/fd_newton_accepted_fraction"] = (
            1.0 if use_line_search else 0.0
        )
        metrics["optimizer/fd_newton_loss_rejected_rows"] = (
            0.0 if use_line_search else float(model.n_families)
        )
        return loss_vec, metrics, 1, object()

    runner._active_fd_newton_step = fake_step  # type: ignore[method-assign]

    result = runner.run()

    assert calls == [
        (False, 64),
        (False, 64),
        (True, 16),
        (True, 16),
    ]
    assert result.status == "converged"
    assert result.reason == "loss_change_patience"
    assert runner.fake_model.closed


def test_hessian_sgd_large_batch_plateau_stops_before_line_search(
    tmp_path: Path,
):
    config = _optimizer_mode_config(
        tmp_path,
        optimizer="hessian-sgd",
        mode="genewise",
        steps=4,
        solver_warmup_iters=0,
        fd_hessian_refresh_steps=16,
        fd_hessian_epsilon=1e-3,
        fd_newton_damping=1e-6,
        loss_change_tol=0.0,
        loss_patience=1,
        best_likelihood_patience=0,
    )
    runner = _WorkflowLargeCladeAdaptiveRebatchRunner(config)
    calls: list[tuple[bool, int | None]] = []

    def fake_step(
        model,
        *,
        solver_stage,
        use_line_search,
        hessian_refresh_steps,
        **_kwargs,
    ):
        calls.append((bool(use_line_search), hessian_refresh_steps))
        model.theta.grad = torch.ones_like(model.theta)
        loss_vec = torch.full(
            (int(model.n_families),),
            5.0,
            device=model.theta.device,
            dtype=model.theta.dtype,
        )
        metrics = runner._active_batch_metrics(
            model,
            loss_vec=loss_vec,
            solver_stage=solver_stage,
        )
        metrics["optimizer/fd_newton_accepted_fraction"] = (
            1.0 if use_line_search else 0.0
        )
        metrics["optimizer/fd_newton_loss_rejected_rows"] = (
            0.0 if use_line_search else float(model.n_families)
        )
        return loss_vec, metrics, 1, object()

    runner._active_fd_newton_step = fake_step  # type: ignore[method-assign]

    result = runner.run()

    assert calls == [
        (False, 64),
        (False, 64),
    ]
    assert result.status == "converged"
    assert result.reason == "loss_change_patience"
    assert runner.fake_model.closed


def test_hessian_sgd_legacy_gradient_tolerance_is_ignored(
    tmp_path: Path,
):
    config = _optimizer_mode_config(
        tmp_path,
        optimizer="hessian-sgd",
        mode="genewise",
        steps=3,
        solver_warmup_iters=0,
        fd_hessian_epsilon=1e-3,
        fd_newton_damping=1e-6,
        loss_patience=0,
        best_likelihood_patience=0,
    )
    legacy_data = config.to_dict()
    legacy_data["grad_inf_tol"] = 10.0
    config = RunConfig.from_dict(legacy_data)
    runner = _WorkflowBatchedLBFGSModeRunner(config)

    result = runner.run()

    history_rows = _optimizer_mode_history_rows(config.out_dir)
    hessian_rows = [
        row for row in history_rows if row["optimizer/phase"] == "hessian-sgd"
    ]
    assert hessian_rows
    assert {row["optimizer/fd_newton_subphase"] for row in hessian_rows} == {
        "hessian_sgd"
    }
    assert all("optimizer/hessian_sgd_polish_active" not in row for row in hessian_rows)
    assert result.status == "not_converged"
    assert result.reason == "max_steps"
    assert runner.fake_model.closed


def test_hessian_sgd_warmup_plateau_promotes_to_full_solver(tmp_path: Path):
    config = _optimizer_mode_config(
        tmp_path,
        optimizer="hessian-sgd",
        mode="genewise",
        steps=3,
        solver_warmup_iters=6,
        solver_warmup_loss_patience=99,
        loss_change_tol=1e9,
        loss_patience=1,
        best_likelihood_patience=0,
        fd_hessian_epsilon=1e-3,
        fd_newton_damping=1e-6,
        lr=1e-9,
    )
    runner = _WorkflowBatchedLBFGSModeRunner(config)

    result = runner.run()

    history_rows = _optimizer_mode_history_rows(config.out_dir)
    hessian_rows = [
        row for row in history_rows if row["optimizer/phase"] == "hessian-sgd"
    ]
    assert [row["optimizer/solver_stage"] for row in hessian_rows] == [
        "warmup",
        "warmup",
        "full",
    ]
    assert result.status == "not_converged"
    assert runner.fake_model.closed


def test_hessian_sgd_large_batch_warmup_plateau_skips_full_solver(
    tmp_path: Path,
):
    config = _optimizer_mode_config(
        tmp_path,
        optimizer="hessian-sgd",
        mode="genewise",
        steps=3,
        solver_warmup_iters=6,
        solver_warmup_loss_patience=99,
        loss_change_tol=1e9,
        loss_patience=1,
        best_likelihood_patience=0,
        fd_hessian_epsilon=1e-3,
        fd_newton_damping=1e-6,
        lr=1e-9,
    )
    runner = _WorkflowLargeCladeAdaptiveRebatchRunner(config)

    result = runner.run()

    history_rows = _optimizer_mode_history_rows(config.out_dir)
    hessian_rows = [
        row for row in history_rows if row["optimizer/phase"] == "hessian-sgd"
    ]
    final_rows = [
        row for row in history_rows if row["optimizer/phase"] == "final_eval"
    ]
    assert [row["optimizer/solver_stage"] for row in hessian_rows] == [
        "warmup",
        "warmup",
    ]
    assert final_rows[-1]["optimizer/final_eval_source"] == "cached_active_batches"
    assert result.status == "converged"
    assert result.reason == "loss_change_patience"
    assert runner.fake_model.closed


def test_hessian_sgd_large_batch_warmup_skip_does_not_cache_noncanonical_full(
    tmp_path: Path,
):
    config = _optimizer_mode_config(
        tmp_path,
        optimizer="hessian-sgd",
        mode="genewise",
        steps=3,
        fixed_iters_pi=16,
        neumann_terms=16,
        hessian_sgd_normal_fixed_iters_pi=8,
        hessian_sgd_normal_neumann_terms=8,
        solver_warmup_iters=6,
        solver_warmup_loss_patience=99,
        loss_change_tol=1e9,
        loss_patience=1,
        best_likelihood_patience=0,
        fd_hessian_epsilon=1e-3,
        fd_newton_damping=1e-6,
        lr=1e-9,
    )
    runner = _WorkflowLargeCladeAdaptiveRebatchRunner(config)

    result = runner.run()

    history_rows = _optimizer_mode_history_rows(config.out_dir)
    hessian_rows = [
        row for row in history_rows if row["optimizer/phase"] == "hessian-sgd"
    ]
    final_rows = [
        row for row in history_rows if row["optimizer/phase"] == "final_eval"
    ]
    assert [row["optimizer/solver_stage"] for row in hessian_rows] == [
        "warmup",
        "warmup",
    ]
    assert "optimizer/final_eval_source" not in final_rows[-1]
    assert result.status == "converged"
    assert result.reason == "loss_change_patience"
    assert runner.fake_model.closed


def test_active_batch_plateau_tolerances_scale_by_family_count(tmp_path: Path):
    config = _optimizer_mode_config(
        tmp_path,
        optimizer="hessian-sgd",
        mode="genewise",
        steps=1,
        solver_warmup_iters=0,
        loss_change_tol=0.25,
        best_likelihood_min_delta=0.5,
        fd_hessian_epsilon=1e-3,
        fd_newton_damping=1e-6,
        lr=1e-9,
    )
    runner = _WorkflowAdaptiveRebatchRunner(config)

    result = runner.run()

    history_rows = _optimizer_mode_history_rows(config.out_dir)
    assert history_rows[0]["optimizer/batch_family_count"] == 2
    assert history_rows[0]["loss_change_tol_bits"] == pytest.approx(0.5)
    assert history_rows[0]["best_likelihood_min_delta_bits"] == pytest.approx(1.0)
    assert result.status == "not_converged"
    assert runner.fake_model.closed


def test_optimization_runner_adaptive_rebatch_replans_unconverged_families(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(optimize_workflow, "_ADAPTIVE_REBATCH_MIN_ACTIVE_FAMILIES", 1)
    config = _optimizer_mode_config(
        tmp_path,
        optimizer="batched-lbfgs",
        mode="genewise",
        steps=2,
        solver_warmup_iters=0,
        adaptive_rebatch=True,
        adaptive_rebatch_fraction=0.5,
        adaptive_rebatch_min_remaining_families=1,
        best_likelihood_patience=1,
        lbfgs_max_iter=1,
    )
    runner = _WorkflowAdaptiveRebatchLikelihoodPlateauRunner(config)

    result = runner.run()

    history_rows = _optimizer_mode_history_rows(config.out_dir)
    assert history_rows[0]["optimizer/rebatch_checked"] is True
    assert history_rows[0]["optimizer/rebatch_triggered"] is False
    assert history_rows[0]["optimizer/rebatch_active_converged_families"] == 0.0
    assert history_rows[1]["optimizer/rebatch_checked"] is True
    assert history_rows[1]["optimizer/rebatch_triggered"] is True
    assert (
        history_rows[1]["optimizer/rebatch_convergence_criterion"]
        == "best_likelihood_patience"
    )
    assert history_rows[1]["optimizer/rebatch_active_converged_families"] == 1.0
    assert history_rows[1]["optimizer/rebatch_remaining_families"] == 1.0
    assert runner.fake_model.replanned_indices == [(0,)]
    latest = load_checkpoint(config.out_dir / "checkpoints" / "latest.pt")
    assert latest["status"]["converged_family_indices"] == [1]
    assert latest["status"]["batch_plan_generation"] == 1
    assert result.status == "not_converged"
    assert runner.fake_model.closed


def test_optimization_runner_fd_newton_adaptive_rebatch_replans_unconverged_families(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(optimize_workflow, "_ADAPTIVE_REBATCH_MIN_ACTIVE_FAMILIES", 1)
    config = _optimizer_mode_config(
        tmp_path,
        optimizer="adam-fd-newton",
        mode="genewise",
        steps=2,
        solver_warmup_iters=0,
        adaptive_rebatch=True,
        adaptive_rebatch_fraction=0.5,
        adaptive_rebatch_min_remaining_families=1,
        fd_adam_warmup_steps=1,
        best_likelihood_patience=1,
        lr=0.2,
    )
    runner = _WorkflowAdaptiveRebatchLikelihoodPlateauRunner(config)

    result = runner.run()

    history_rows = _optimizer_mode_history_rows(config.out_dir)
    assert history_rows[0]["optimizer/phase"] == "adam-fd-newton"
    assert history_rows[0]["optimizer/fd_newton_subphase"] == "adam_warmup"
    assert history_rows[0]["optimizer/rebatch_checked"] is True
    assert history_rows[0]["optimizer/rebatch_triggered"] is False
    assert history_rows[0]["optimizer/rebatch_active_converged_families"] == 0.0
    assert history_rows[1]["optimizer/rebatch_checked"] is True
    assert history_rows[1]["optimizer/rebatch_triggered"] is True
    assert (
        history_rows[1]["optimizer/rebatch_convergence_criterion"]
        == "best_likelihood_patience"
    )
    assert history_rows[1]["optimizer/rebatch_active_converged_families"] == 1.0
    assert history_rows[1]["optimizer/rebatch_remaining_families"] == 1.0
    assert runner.fake_model.replanned_indices == [(0,)]
    latest = load_checkpoint(config.out_dir / "checkpoints" / "latest.pt")
    assert latest["status"]["converged_family_indices"] == [1]
    assert latest["status"]["batch_plan_generation"] == 1
    assert result.status == "not_converged"
    assert runner.fake_model.closed


def test_optimization_runner_hessian_sgd_adaptive_rebatch_replans_unconverged_families(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(optimize_workflow, "_ADAPTIVE_REBATCH_MIN_ACTIVE_FAMILIES", 1)
    config = _optimizer_mode_config(
        tmp_path,
        optimizer="hessian-sgd",
        mode="genewise",
        steps=2,
        solver_warmup_iters=0,
        adaptive_rebatch=True,
        adaptive_rebatch_fraction=0.5,
        adaptive_rebatch_min_remaining_families=1,
        fd_hessian_epsilon=1e-3,
        fd_newton_damping=1e-6,
        best_likelihood_patience=1,
        lr=0.5,
    )
    runner = _WorkflowAdaptiveRebatchLikelihoodPlateauRunner(config)

    result = runner.run()

    history_rows = _optimizer_mode_history_rows(config.out_dir)
    assert history_rows[0]["optimizer/phase"] == "hessian-sgd"
    assert history_rows[0]["optimizer/fd_newton_subphase"] == "hessian_sgd"
    assert history_rows[0]["optimizer/rebatch_checked"] is True
    assert history_rows[0]["optimizer/rebatch_triggered"] is False
    assert history_rows[0]["optimizer/rebatch_active_converged_families"] == 0.0
    assert history_rows[1]["optimizer/rebatch_checked"] is True
    assert history_rows[1]["optimizer/rebatch_triggered"] is True
    assert (
        history_rows[1]["optimizer/rebatch_convergence_criterion"]
        == "best_likelihood_patience"
    )
    assert history_rows[1]["optimizer/rebatch_active_converged_families"] == 1.0
    assert history_rows[1]["optimizer/rebatch_remaining_families"] == 1.0
    assert runner.fake_model.replanned_indices == [(0,)]
    latest = load_checkpoint(config.out_dir / "checkpoints" / "latest.pt")
    assert latest["status"]["converged_family_indices"] == [1]
    assert latest["status"]["batch_plan_generation"] == 1
    assert result.status == "not_converged"
    assert runner.fake_model.closed


def test_optimization_runner_adaptive_rebatch_skips_tiny_active_batches(
    tmp_path: Path,
):
    config = _optimizer_mode_config(
        tmp_path,
        optimizer="adam-fd-newton",
        mode="genewise",
        steps=1,
        solver_warmup_iters=0,
        adaptive_rebatch=True,
        adaptive_rebatch_fraction=0.5,
        adaptive_rebatch_min_remaining_families=1,
        fd_adam_warmup_steps=1,
        lr=1e-9,
    )
    runner = _WorkflowAdaptiveRebatchRunner(config)

    result = runner.run()

    history_rows = _optimizer_mode_history_rows(config.out_dir)
    assert history_rows[0]["optimizer/rebatch_checked"] is False
    assert history_rows[0]["optimizer/rebatch_triggered"] is False
    assert history_rows[0]["optimizer/rebatch_reason"] == "small_active_batch"
    assert runner.fake_model.replanned_indices == []
    assert result.status == "not_converged"
    assert runner.fake_model.closed


def test_batched_lbfgs_active_batch_closure_zeros_inactive_rows(tmp_path: Path):
    config = _optimizer_mode_config(
        tmp_path,
        optimizer="batched-lbfgs",
        mode="genewise",
    )
    runner = OptimizationRunner(config)
    model = _WorkflowBatchedLBFGSModeModel()

    model.select_batch(0)
    loss_vec, metrics = runner._evaluate_active_genewise_vector_and_grad(
        model,
        solver_stage="warmup",
    )

    assert loss_vec.shape == (2,)
    assert loss_vec[0] > 0.0
    assert loss_vec[1] == 0.0
    assert torch.count_nonzero(model.theta.grad[0]).item() == 3
    assert torch.count_nonzero(model.theta.grad[1]).item() == 0
    assert metrics["optimizer/objective_scope"] == "active_batch"
    assert metrics["optimizer/batch_index"] == 0
    assert metrics["optimizer/solver_stage"] == "warmup"


def test_adam_fd_newton_active_batch_step_uses_finite_difference_hessian(
    tmp_path: Path,
):
    config = _optimizer_mode_config(
        tmp_path,
        optimizer="adam-fd-newton",
        mode="genewise",
        fd_adam_warmup_steps=0,
        fd_hessian_epsilon=1e-3,
        fd_newton_damping=1e-6,
    )
    runner = OptimizationRunner(config)
    model = _WorkflowBatchedLBFGSModeModel()
    before = model.theta.detach().clone()
    model.select_batch(0)

    loss_vec, metrics, evals, state = runner._active_fd_newton_step(
        model,
        solver_stage="full",
    )

    assert evals == 9
    assert metrics["optimizer/fd_newton_accepted_rows"] == 1.0
    assert metrics["optimizer/fd_newton_fallback_rows"] == 0.0
    assert metrics["optimizer/fd_newton_hessian_source"] == "finite_difference"
    assert state.updates_since_refresh == 1
    assert loss_vec[0] < 1.000001
    torch.testing.assert_close(model.theta.detach()[0], torch.zeros(3), atol=1e-3, rtol=0)
    torch.testing.assert_close(model.theta.detach()[1], before[1])


def test_fd_newton_line_search_falls_back_to_projected_gradient(
    tmp_path: Path,
):
    config = _optimizer_mode_config(
        tmp_path,
        optimizer="hessian-sgd",
        mode="genewise",
        fd_hessian_epsilon=1e-3,
        fd_newton_damping=1e-6,
    )
    runner = OptimizationRunner(config)
    model = _WorkflowBatchedLBFGSModeModel()
    model.select_batch(0)
    with torch.no_grad():
        model.theta[0].copy_(torch.tensor([1.0, 0.0, 0.0]))
    before = model.theta.detach().clone()
    active_theta = model.theta.detach().index_select(
        0,
        torch.tensor([0], dtype=torch.long),
    )
    hessian = torch.eye(3, dtype=model.theta.dtype).unsqueeze(0) * 1e-3
    state = optimize_workflow._FDNewtonHessianState(
        batch_index=0,
        solver_stage="full",
        family_indices=(0,),
        hessian=hessian,
        active_theta=active_theta,
        active_grad=torch.tensor([[2.0, 0.0, 0.0]], dtype=model.theta.dtype),
        active_loss=torch.tensor([2.0], dtype=model.theta.dtype),
        updates_since_refresh=0,
    )

    loss_vec, metrics, evals, _state = runner._active_fd_newton_step(
        model,
        solver_stage="full",
        hessian_state=state,
        update_hessian_with_bfgs=True,
        step_scale=1.0,
        use_line_search=True,
        line_search_max_steps=2,
    )

    assert metrics["optimizer/fd_newton_hessian_source"] == "bfgs_update"
    assert metrics["optimizer/fd_newton_fallback_rows"] == 0.0
    assert metrics["optimizer/fd_newton_line_search_fallback_attempted_rows"] == 1.0
    assert metrics["optimizer/fd_newton_line_search_fallback_rows"] == 1.0
    assert metrics["optimizer/fd_newton_accepted_rows"] == 1.0
    assert evals == 5
    assert loss_vec[0] < 2.0
    torch.testing.assert_close(model.theta.detach()[0], torch.zeros(3), atol=1e-6, rtol=0)
    torch.testing.assert_close(model.theta.detach()[1], before[1])


def test_fd_newton_fallback_line_search_uses_short_cap(tmp_path: Path):
    config = _optimizer_mode_config(
        tmp_path,
        optimizer="hessian-sgd",
        mode="genewise",
        fd_hessian_epsilon=1e-3,
        fd_newton_damping=1e-6,
    )
    runner = OptimizationRunner(config)
    model = _WorkflowBatchedLBFGSModeModel()
    model.select_batch(0)
    with torch.no_grad():
        model.theta[0].copy_(torch.tensor([1.0, 0.0, 0.0]))
    active_theta = model.theta.detach().index_select(
        0,
        torch.tensor([0], dtype=torch.long),
    )
    hessian = torch.eye(3, dtype=model.theta.dtype).unsqueeze(0) * 1e-3
    state = optimize_workflow._FDNewtonHessianState(
        batch_index=0,
        solver_stage="full",
        family_indices=(0,),
        hessian=hessian,
        active_theta=active_theta,
        active_grad=torch.tensor([[2.0, 0.0, 0.0]], dtype=model.theta.dtype),
        active_loss=torch.tensor([2.0], dtype=model.theta.dtype),
        updates_since_refresh=0,
    )

    loss_vec, metrics, evals, _state = runner._active_fd_newton_step(
        model,
        solver_stage="full",
        hessian_state=state,
        update_hessian_with_bfgs=True,
        step_scale=1.0,
        use_line_search=True,
        line_search_max_steps=8,
    )

    assert metrics["optimizer/fd_newton_max_ls"] == 8.0
    assert metrics["optimizer/fd_newton_fallback_max_ls"] == 2.0
    assert metrics["optimizer/fd_newton_line_search_fallback_attempted_rows"] == 1.0
    assert metrics["optimizer/fd_newton_line_search_fallback_rows"] == 1.0
    assert metrics["optimizer/fd_newton_loss_evals"] == 10.0
    assert evals == 11
    assert loss_vec[0] < 2.0


def test_adam_fd_newton_step_ignores_legacy_cap_and_projects_to_rate_bounds(
    tmp_path: Path,
):
    config = RunConfig.from_dict(
        {
            "species_tree": tmp_path / "sp.nwk",
            "families_file": tmp_path / "families.txt",
            "out_dir": tmp_path / "out-adam-fd-newton",
            "mode": "genewise",
            "optimizer": "adam-fd-newton",
            "fd_adam_warmup_steps": 0,
            "fd_hessian_epsilon": 1e-3,
            "fd_newton_damping": 1e-6,
            "fd_newton_max_step": 1e-6,
            "device": "cpu",
        }
    )
    runner = OptimizationRunner(config)
    model = _WorkflowBoundedFDNewtonModel()
    before = model.theta.detach().clone()
    model.select_batch(0)

    _loss_vec, metrics, _evals, _state = runner._active_fd_newton_step(
        model,
        solver_stage="full",
    )

    upper_bound = math.log2(config.max_rate)
    lower_bound = math.log2(config.min_rate)
    assert metrics["optimizer/fd_newton_raw_step_inf"] > 1e-6
    assert metrics["optimizer/fd_newton_bound_projected_step_inf"] == pytest.approx(
        metrics["optimizer/fd_newton_raw_step_inf"],
        abs=1e-5,
    )
    assert model.theta.detach()[0].min() >= lower_bound
    assert model.theta.detach()[0].max() <= upper_bound
    assert not torch.equal(model.theta.detach()[0], before[0])
    torch.testing.assert_close(model.theta.detach()[1], before[1])


def test_adam_fd_newton_reuses_bfgs_updated_hessian_between_refreshes(
    tmp_path: Path,
):
    config = _optimizer_mode_config(
        tmp_path,
        optimizer="adam-fd-newton",
        mode="genewise",
        fd_adam_warmup_steps=0,
        fd_hessian_refresh_steps=5,
        fd_hessian_epsilon=1e-3,
        fd_newton_damping=1.0,
    )
    runner = OptimizationRunner(config)
    model = _WorkflowBatchedLBFGSModeModel()
    model.select_batch(0)

    _loss_vec, first_metrics, first_evals, state = runner._active_fd_newton_step(
        model,
        solver_stage="full",
    )
    _loss_vec, second_metrics, second_evals, state = runner._active_fd_newton_step(
        model,
        solver_stage="full",
        hessian_state=state,
    )

    assert first_metrics["optimizer/fd_newton_hessian_source"] == "finite_difference"
    assert second_metrics["optimizer/fd_newton_hessian_source"] == "bfgs_update"
    assert first_evals == 9
    assert second_evals < first_evals
    assert second_metrics["optimizer/fd_newton_grad_evals"] == 1.0
    assert second_metrics["optimizer/fd_newton_bfgs_updated_rows"] == 1.0
    assert state.updates_since_refresh == 2


def test_hessian_sgd_reuses_fixed_hessian_between_refreshes(tmp_path: Path):
    config = _optimizer_mode_config(
        tmp_path,
        optimizer="hessian-sgd",
        mode="genewise",
        fd_hessian_refresh_steps=5,
        fd_hessian_epsilon=1e-3,
        fd_newton_damping=1.0,
    )
    runner = OptimizationRunner(config)
    model = _WorkflowBatchedLBFGSModeModel()
    model.select_batch(0)

    _loss_vec, first_metrics, first_evals, state = runner._active_fd_newton_step(
        model,
        solver_stage="full",
        update_hessian_with_bfgs=False,
        step_scale=0.5,
        use_line_search=False,
    )
    fixed_hessian = state.hessian.detach().clone()
    _loss_vec, second_metrics, second_evals, state = runner._active_fd_newton_step(
        model,
        solver_stage="full",
        hessian_state=state,
        update_hessian_with_bfgs=False,
        step_scale=0.5,
        use_line_search=False,
    )

    assert first_metrics["optimizer/fd_newton_hessian_source"] == "finite_difference"
    assert first_metrics["optimizer/fd_newton_hessian_update"] == "fixed"
    assert second_metrics["optimizer/fd_newton_hessian_source"] == "fixed_hessian"
    assert second_metrics["optimizer/fd_newton_hessian_update"] == "fixed"
    assert first_metrics["optimizer/fd_newton_line_search"] is False
    assert first_metrics["optimizer/fd_newton_loss_evals"] == 0.0
    assert second_metrics["optimizer/fd_newton_line_search"] is False
    assert second_metrics["optimizer/fd_newton_loss_evals"] == 0.0
    assert first_evals == 8
    assert second_evals == 1
    assert second_metrics["optimizer/fd_newton_grad_evals"] == 1.0
    assert second_metrics["optimizer/fd_newton_bfgs_updated_rows"] == 0.0
    torch.testing.assert_close(state.hessian, fixed_hessian)
    assert state.updates_since_refresh == 2


def test_hessian_sgd_refreshes_fixed_hessian_after_configured_steps(
    tmp_path: Path,
):
    config = _optimizer_mode_config(
        tmp_path,
        optimizer="hessian-sgd",
        mode="genewise",
        fd_hessian_refresh_steps=1,
        fd_hessian_epsilon=1e-3,
        fd_newton_damping=1.0,
    )
    runner = OptimizationRunner(config)
    model = _WorkflowBatchedLBFGSModeModel()
    model.select_batch(0)

    _loss_vec, _first_metrics, _first_evals, state = runner._active_fd_newton_step(
        model,
        solver_stage="full",
        update_hessian_with_bfgs=False,
        step_scale=0.5,
        use_line_search=False,
    )
    _loss_vec, second_metrics, second_evals, _state = runner._active_fd_newton_step(
        model,
        solver_stage="full",
        hessian_state=state,
        update_hessian_with_bfgs=False,
        step_scale=0.5,
        use_line_search=False,
    )

    assert second_metrics["optimizer/fd_newton_hessian_source"] == "finite_difference"
    assert second_metrics["optimizer/fd_newton_hessian_update"] == "fixed"
    assert second_metrics["optimizer/fd_newton_line_search"] is False
    assert second_metrics["optimizer/fd_newton_loss_evals"] == 0.0
    assert second_evals == 7
    assert second_metrics["optimizer/fd_newton_grad_evals"] == 7.0


def test_hessian_sgd_refresh_override_forces_fixed_hessian_refresh(
    tmp_path: Path,
):
    config = _optimizer_mode_config(
        tmp_path,
        optimizer="hessian-sgd",
        mode="genewise",
        fd_hessian_refresh_steps=5,
        fd_hessian_epsilon=1e-3,
        fd_newton_damping=1.0,
    )
    runner = OptimizationRunner(config)
    model = _WorkflowBatchedLBFGSModeModel()
    model.select_batch(0)

    _loss_vec, _first_metrics, _first_evals, state = runner._active_fd_newton_step(
        model,
        solver_stage="full",
        update_hessian_with_bfgs=False,
        step_scale=0.5,
        use_line_search=False,
    )
    _loss_vec, second_metrics, second_evals, _state = runner._active_fd_newton_step(
        model,
        solver_stage="full",
        hessian_state=state,
        update_hessian_with_bfgs=False,
        step_scale=1.0,
        use_line_search=True,
        hessian_refresh_steps=1,
    )

    assert second_metrics["optimizer/fd_newton_hessian_source"] == "finite_difference"
    assert second_metrics["optimizer/fd_newton_hessian_refresh_steps"] == 1.0
    assert second_metrics["optimizer/fd_newton_line_search"] is True
    assert second_evals > 1


def test_adam_fd_newton_refreshes_hessian_after_configured_steps(
    tmp_path: Path,
):
    config = _optimizer_mode_config(
        tmp_path,
        optimizer="adam-fd-newton",
        mode="genewise",
        fd_adam_warmup_steps=0,
        fd_hessian_refresh_steps=1,
        fd_hessian_epsilon=1e-3,
        fd_newton_damping=1.0,
    )
    runner = OptimizationRunner(config)
    model = _WorkflowBatchedLBFGSModeModel()
    model.select_batch(0)

    _loss_vec, _first_metrics, _first_evals, state = runner._active_fd_newton_step(
        model,
        solver_stage="full",
    )
    _loss_vec, second_metrics, second_evals, _state = runner._active_fd_newton_step(
        model,
        solver_stage="full",
        hessian_state=state,
    )

    assert second_metrics["optimizer/fd_newton_hessian_source"] == "finite_difference"
    assert second_evals == 8
    assert second_metrics["optimizer/fd_newton_grad_evals"] == 7.0


def test_optimization_runner_batched_lbfgs_advances_resident_batches(tmp_path: Path):
    config = _optimizer_mode_config(
        tmp_path,
        optimizer="batched-lbfgs",
        mode="genewise",
        steps=5,
        loss_change_tol=1e9,
        loss_patience=1,
        solver_warmup_loss_patience=1,
        lbfgs_max_iter=1,
        solver_warmup_iters=6,
    )
    runner = _WorkflowBatchedLBFGSModeRunner(config)

    result = runner.run()

    history_rows = _optimizer_mode_history_rows(config.out_dir)
    batched_rows = [
        row for row in history_rows if row["optimizer/phase"] == "batched-lbfgs"
    ]
    assert [row["optimizer/batch_index"] for row in batched_rows] == [0, 0, 0, 0, 1]
    assert [row["optimizer/solver_stage"] for row in batched_rows] == [
        "warmup",
        "warmup",
        "full",
        "full",
        "warmup",
    ]
    assert {row["optimizer/objective_scope"] for row in batched_rows} == {
        "active_batch",
    }
    assert runner.fake_model.solver_configs == [
        {"fixed_iters_E": 6, "fixed_iters_Pi": 6, "neumann_terms": 6},
        {"fixed_iters_E": None, "fixed_iters_Pi": 16, "neumann_terms": 16},
        {"fixed_iters_E": 6, "fixed_iters_Pi": 6, "neumann_terms": 6},
        {"fixed_iters_E": None, "fixed_iters_Pi": 16, "neumann_terms": 16},
        {"fixed_iters_E": None, "fixed_iters_Pi": 32, "neumann_terms": 32},
        {"fixed_iters_E": None, "fixed_iters_Pi": 16, "neumann_terms": 16},
    ]
    assert result.status == "not_converged"
    assert result.reason == "max_steps"
    assert runner.fake_model.drop_cached_static_states_calls == 0
    assert runner.fake_model.static_state.warm_E is None
    assert runner.fake_model.static_state.last_solver_stats is None
    assert runner.fake_model.closed


def test_final_iteration_check_keeps_static_layout_and_clears_runtime_state(
    tmp_path: Path,
    monkeypatch,
):
    config = _optimizer_mode_config(
        tmp_path,
        optimizer="batched-lbfgs",
        mode="genewise",
        final_check_iters=32,
    )
    runner = OptimizationRunner(config)
    model = _WorkflowBatchedLBFGSModeModel()
    baseline_loss_vec, baseline_grad = model.full_genewise_nll_and_grad(
        need_grad=True,
    )
    calls: list[int] = []

    def record_cache_clear(model_arg):
        assert model_arg is model
        calls.append(len(model.solver_configs))

    monkeypatch.setattr(
        optimize_workflow,
        "_clear_cuda_allocator_cache_if_needed",
        record_cache_clear,
    )

    metrics = runner._evaluate_final_iteration_check(
        model,
        baseline_loss=baseline_loss_vec.sum(),
        baseline_grad=baseline_grad,
    )

    assert calls == [0]
    assert model.drop_cached_static_states_calls == 0
    assert model.static_state.warm_E is None
    assert model.static_state.last_solver_stats is None
    assert metrics["optimizer/final_check_status"] == "ok"
    assert metrics["optimizer/final_check_source"] == "configured_solver_budget"
    assert model.solver_configs == [
        {"fixed_iters_E": None, "fixed_iters_Pi": 32, "neumann_terms": 32},
        {"fixed_iters_E": None, "fixed_iters_Pi": 16, "neumann_terms": 16},
    ]


def test_final_iteration_check_runs_for_specieswise_mode(tmp_path: Path):
    config = _optimizer_mode_config(
        tmp_path,
        optimizer="lbfgsb",
        mode="specieswise",
        fixed_iters_e=6,
        fixed_iters_pi=16,
        neumann_terms=16,
        final_check_iters=32,
    )
    runner = OptimizationRunner(config)
    model = _WorkflowSpecieswiseOptimizerModeModel()
    solver_configs: list[dict[str, object]] = []

    def configure_solver_iterations(**kwargs):
        solver_configs.append(dict(kwargs))

    model.configure_solver_iterations = configure_solver_iterations
    model.theta.grad = None
    baseline_loss = model.full_loss()
    baseline_loss.backward()
    baseline_grad = model.theta.grad.detach().clone()

    metrics = runner._evaluate_final_iteration_check(
        model,
        baseline_loss=baseline_loss,
        baseline_grad=baseline_grad,
    )

    assert metrics["optimizer/final_check_status"] == "ok"
    assert metrics["optimizer/final_check_source"] == "configured_solver_budget"
    assert metrics["optimizer/final_check_iters"] == 32
    assert metrics["optimizer/final_check_iters_E"] == 32
    assert metrics["optimizer/final_check_loss_abs_delta_bits"] == pytest.approx(0.0)
    assert metrics["optimizer/final_check_grad_max_abs_delta"] == pytest.approx(0.0)
    assert solver_configs == [
        {"fixed_iters_E": 32, "fixed_iters_Pi": 32, "neumann_terms": 32},
        {"fixed_iters_E": 6, "fixed_iters_Pi": 16, "neumann_terms": 16},
    ]


def test_final_iteration_check_skipped_or_disabled_reports_reason(tmp_path: Path):
    config = _optimizer_mode_config(
        tmp_path,
        optimizer="adam",
        mode="genewise",
        final_check_iters=32,
    )
    runner = OptimizationRunner(config)

    metrics = runner._evaluate_final_iteration_check(
        object(),
        baseline_loss=torch.tensor(0.0),
        baseline_grad=torch.zeros(1),
    )

    assert metrics["optimizer/final_check_status"] == "skipped"
    assert metrics["optimizer/final_check_source"] == "not_evaluated"
    assert (
        metrics["optimizer/final_check_reason"]
        == "model_has_no_solver_iteration_controls"
    )

    disabled_config = _optimizer_mode_config(
        tmp_path,
        optimizer="adam",
        mode="genewise",
        final_check_iters=0,
    )
    disabled_runner = OptimizationRunner(disabled_config)
    disabled_model = SimpleNamespace(configure_solver_iterations=lambda **_: None)

    disabled_metrics = disabled_runner._evaluate_final_iteration_check(
        disabled_model,
        baseline_loss=torch.tensor(0.0),
        baseline_grad=torch.zeros(1),
    )

    assert disabled_metrics["optimizer/final_check_status"] == "disabled"
    assert disabled_metrics["optimizer/final_check_source"] == "not_evaluated"
    assert (
        disabled_metrics["optimizer/final_check_reason"]
        == "final_check_iters_disabled"
    )


def test_optimization_runner_batched_lbfgs_resume_restores_state(tmp_path: Path):
    first_config = _optimizer_mode_config(
        tmp_path,
        optimizer="batched-lbfgs",
        mode="genewise",
        steps=1,
        lbfgs_max_iter=1,
        solver_warmup_iters=0,
    )
    first_runner = _WorkflowBatchedLBFGSModeRunner(first_config)
    first_runner.run()
    first_latest = load_checkpoint(first_config.out_dir / "checkpoints" / "latest.pt")
    assert first_latest["optimizer_phase"] == "batched-lbfgs"

    resumed_config = _optimizer_mode_config(
        tmp_path,
        optimizer="batched-lbfgs",
        mode="genewise",
        steps=2,
        lbfgs_max_iter=1,
        solver_warmup_iters=0,
        out_dir=tmp_path / "out-batched-lbfgs-resumed",
        resume_from=first_config.out_dir / "checkpoints" / "latest.pt",
    )
    resumed_runner = _WorkflowBatchedLBFGSModeRunner(resumed_config)

    result = resumed_runner.run()

    history_rows = _optimizer_mode_history_rows(resumed_config.out_dir)
    assert [(row["optimizer/phase"], row["step"]) for row in history_rows] == [
        ("batched-lbfgs", 1),
        ("final_eval", 2),
    ]
    assert history_rows[0]["resume_optimizer_state"] == "restored"
    resumed_latest = load_checkpoint(resumed_config.out_dir / "checkpoints" / "latest.pt")
    assert resumed_latest["optimizer_phase"] == "batched-lbfgs"
    assert result.steps_completed == 2
    assert result.status == "not_converged"
    assert resumed_runner.fake_model.closed


def test_optimization_runner_adam_lbfgs_schedule_runs_active_phases(tmp_path: Path):
    config = _optimizer_mode_config(
        tmp_path,
        optimizer="adam-lbfgs",
        steps=3,
        adam_warmup_steps=1,
    )
    runner = _WorkflowOptimizerModeRunner(config)

    result = runner.run()

    history_rows = _optimizer_mode_history_rows(config.out_dir)
    assert [row["optimizer/phase"] for row in history_rows] == [
        "adam",
        "lbfgs",
        "lbfgs",
        "final_eval",
    ]
    assert history_rows[0]["closure_evals"] == 1
    assert history_rows[0]["optimizer/eval_position"] == "pre_step"
    assert history_rows[0]["optimizer/step_applied"] is True
    assert all(row["closure_evals"] >= 2 for row in history_rows[1:3])
    assert all(row["optimizer/eval_position"] == "post_step" for row in history_rows[1:3])
    assert all(row["theta_step_inf"] > 0.0 for row in history_rows[:3])
    latest = load_checkpoint(config.out_dir / "checkpoints" / "latest.pt")
    assert latest["optimizer_phase"] == "lbfgs"
    assert result.steps_completed == 3
    assert result.status == "not_converged"
    assert runner.fake_model.closed


def test_optimization_runner_lbfgs_runtime_error_is_failed_status(
    tmp_path: Path,
    monkeypatch,
):
    def raise_lbfgs_runtime_error(self, closure=None):
        raise RuntimeError("lbfgs failed")

    monkeypatch.setattr(torch.optim.LBFGS, "step", raise_lbfgs_runtime_error)
    config = _optimizer_mode_config(tmp_path, optimizer="lbfgs")
    runner = _WorkflowOptimizerModeRunner(config)

    result = runner.run()

    assert result.status == "failed"
    assert result.reason == "lbfgs_runtime_error"
    history_rows = _optimizer_mode_history_rows(config.out_dir)
    assert [row["optimizer/phase"] for row in history_rows] == ["final_eval"]
    latest = load_checkpoint(config.out_dir / "checkpoints" / "latest.pt")
    assert latest["status"]["status"] == "failed"
    assert latest["status"]["reason"] == "lbfgs_runtime_error"
    assert latest["optimizer_phase"] == "lbfgs"
    assert runner.fake_model.closed


def test_optimization_runner_lbfgs_rejects_nonfinite_post_step_evaluation(
    tmp_path: Path,
    monkeypatch,
):
    class FakePostStepNonfiniteModel(_WorkflowOptimizerModeModel):
        def __init__(self):
            super().__init__()
            self.loss_calls = 0

        def full_loss(self):
            self.loss_calls += 1
            if self.loss_calls == 2:
                return self.theta.sum() * torch.tensor(float("nan"))
            return super().full_loss()

    class FakePostStepNonfiniteRunner(OptimizationRunner):
        def build_model(self):
            self.fake_model = FakePostStepNonfiniteModel()
            return self.fake_model

    def finite_closure_then_move_theta(self, closure=None):
        loss = closure()
        with torch.no_grad():
            for group in self.param_groups:
                for parameter in group["params"]:
                    parameter.add_(0.125)
        return loss

    monkeypatch.setattr(torch.optim.LBFGS, "step", finite_closure_then_move_theta)
    config = _optimizer_mode_config(tmp_path, optimizer="lbfgs")
    runner = FakePostStepNonfiniteRunner(config)

    result = runner.run()

    assert result.status == "failed"
    assert result.reason == "nonfinite_objective_or_gradient"
    assert runner.fake_model.loss_calls == 3
    history_rows = _optimizer_mode_history_rows(config.out_dir)
    assert [row["optimizer/phase"] for row in history_rows] == ["final_eval"]
    latest = load_checkpoint(config.out_dir / "checkpoints" / "latest.pt")
    assert latest["status"]["status"] == "failed"
    assert latest["status"]["reason"] == "nonfinite_objective_or_gradient"
    assert latest["last_row"]["optimizer/phase"] == "final_eval"
    assert runner.fake_model.closed


def test_optimization_runner_marks_nonfinite_final_evaluation_failed(tmp_path: Path):
    class FakeFinalEvalNonfiniteModel(_WorkflowOptimizerModeModel):
        def __init__(self):
            super().__init__()
            self.loss_calls = 0

        def full_loss(self):
            self.loss_calls += 1
            if self.loss_calls == 2:
                return self.theta.sum() * torch.tensor(float("nan"))
            return super().full_loss()

    class FakeFinalEvalNonfiniteRunner(OptimizationRunner):
        def build_model(self):
            self.fake_model = FakeFinalEvalNonfiniteModel()
            return self.fake_model

    config = _optimizer_mode_config(tmp_path, optimizer="adam")
    runner = FakeFinalEvalNonfiniteRunner(config)

    result = runner.run()

    assert result.status == "failed"
    assert result.reason == "nonfinite_objective_or_gradient"
    assert math.isfinite(result.final_nll_bits)
    assert result.final_log_likelihood_bits is None
    assert result.best_log_likelihood_bits == pytest.approx(-result.best_nll_bits)
    assert runner.fake_model.loss_calls == 2
    history_rows = _optimizer_mode_history_rows(config.out_dir)
    assert [row["optimizer/phase"] for row in history_rows] == ["adam", "final_eval"]
    assert history_rows[-1]["optimizer/final_eval_status"] == "failed"
    assert (
        history_rows[-1]["optimizer/final_eval_reason"]
        == "nonfinite_objective_or_gradient"
    )
    assert "likelihood/data_nll_bits" not in history_rows[-1]
    latest = load_checkpoint(config.out_dir / "checkpoints" / "latest.pt")
    assert latest["status"]["status"] == "failed"
    assert latest["status"]["reason"] == "nonfinite_objective_or_gradient"
    assert latest["last_row"]["optimizer/final_eval_status"] == "failed"
    summary = json.loads((config.out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "failed"
    assert summary["reason"] == "nonfinite_objective_or_gradient"
    assert summary["elapsed_s"] >= 0.0
    assert summary["final_nll_bits"] == pytest.approx(result.final_nll_bits)
    assert summary["final_log_likelihood_bits"] is None
    assert math.isinf(result.final_grad_inf)
    assert summary["final_grad_inf"] is None
    assert summary["final_projected_grad_inf"] is None
    assert result.final_projected_grad_inf is None
    assert summary["best_log_likelihood_bits"] == pytest.approx(
        -summary["best_nll_bits"]
    )
    assert result.elapsed_s == pytest.approx(summary["elapsed_s"])
    assert runner.fake_model.closed


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
            self.solver_configs = []

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
                    "E_adjoint_iterations": 5,
                    "E_adjoint_rel_res": 0.125,
                    "E_adjoint_success": False,
                }
            ]

        def configure_solver_iterations(self, **kwargs):
            self.solver_configs.append(dict(kwargs))

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
                        row.get("optimizer/eval_position"),
                        row.get("optimizer/step_applied"),
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
        loss_patience=0,
        best_likelihood_patience=0,
    )
    runner = FakeRunner(config)

    result = runner.run()

    assert result.status == "not_converged"
    assert result.reason == "max_steps"
    assert result.steps_completed == 1
    assert result.best_step == 1
    assert result.final_log_likelihood_bits == pytest.approx(-result.final_nll_bits)
    assert result.best_log_likelihood_bits == pytest.approx(-result.best_nll_bits)
    assert result.final_check_status == "ok"
    assert result.final_check_source == "configured_solver_budget"
    assert result.final_check_reason is None
    assert result.final_check_fallback_clade_budget is None
    assert result.final_check_loss_abs_delta_bits == pytest.approx(0.0)
    assert result.final_check_grad_max_abs_delta == pytest.approx(0.0)
    assert result.final_check_grad_rel_inf_delta == pytest.approx(0.0)
    assert result.out_dir == config.out_dir
    assert result.sampling_checkpoint == config.out_dir / "checkpoints" / "best.pt"
    assert runner.fake_model.closed
    assert runner.fake_model.clears >= 1
    assert runner.saved_checkpoint_losses
    for (
        checkpoint_name,
        eval_position,
        step_applied,
        expected_loss,
        row_loss,
    ) in runner.saved_checkpoint_losses:
        if eval_position == "pre_step" and step_applied is True:
            continue
        assert row_loss == pytest.approx(expected_loss), checkpoint_name

    history_rows = [
        json.loads(line)
        for line in (config.out_dir / "history.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["optimizer/phase"] for row in history_rows] == ["adam", "final_eval"]
    assert history_rows[-1]["step"] == 1
    assert history_rows[-1]["best_step"] == 1
    assert all(
        row["solver/e_adjoint_failed_batches"] == 1.0
        for row in history_rows
    )
    assert all(
        row["solver/e_adjoint_success_batches"] == 0.0
        for row in history_rows
    )
    assert all(
        row["solver/e_adjoint_rel_res_max"] == pytest.approx(0.125)
        for row in history_rows
    )

    summary = json.loads((config.out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "not_converged"
    assert summary["elapsed_s"] >= 0.0
    assert summary["families"] == 2
    assert summary["species"] == 3
    assert summary["batches"] == 1
    assert summary["mode"] == "genewise"
    assert summary["optimizer"] == "adam"
    assert result.mode == summary["mode"]
    assert result.optimizer == summary["optimizer"]
    assert summary["batch_packing"] == "depth_first_fit"
    assert summary["family_chunk_size"] == 0
    assert summary["clade_budget"] == 500_000
    assert summary["fixed_iters_e"] is None
    assert summary["fixed_iters_pi"] == 16
    assert summary["neumann_terms"] == 16
    assert summary["final_check_iters"] == 32
    assert summary["final_nll_bits"] == pytest.approx(result.final_nll_bits)
    assert summary["final_log_likelihood_bits"] == pytest.approx(
        -result.final_nll_bits
    )
    assert summary["final_grad_inf"] == pytest.approx(result.final_grad_inf)
    assert summary["final_projected_grad_inf"] == pytest.approx(
        result.final_projected_grad_inf
    )
    assert summary["best_log_likelihood_bits"] == pytest.approx(
        -summary["best_nll_bits"]
    )
    assert summary["final_check_status"] == "ok"
    assert summary["final_check_source"] == "configured_solver_budget"
    assert "final_check_reason" not in summary
    assert "final_check_fallback_clade_budget" not in summary
    assert summary["final_check_loss_abs_delta_bits"] == pytest.approx(0.0)
    assert summary["final_check_grad_max_abs_delta"] == pytest.approx(0.0)
    assert summary["final_check_grad_rel_inf_delta"] == pytest.approx(0.0)
    assert result.elapsed_s == pytest.approx(summary["elapsed_s"])

    latest = load_checkpoint(config.out_dir / "checkpoints" / "latest.pt")
    best = load_checkpoint(config.out_dir / "checkpoints" / "best.pt")
    assert latest["status"]["status"] == "not_converged"
    assert latest["last_row"]["solver/e_adjoint_failed_batches"] == 1.0
    assert best["status"]["best_step"] == 1
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


def test_optimization_runner_reuses_final_genewise_vector_for_artifacts(
    tmp_path: Path,
):
    class FakeGenewiseVectorModel:
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
            self.species_names = ["sp0", "sp1"]
            self.n_families = 2
            self.n_species = 2
            self.batch_metadata = [SimpleNamespace(batch_index=0)]
            self.full_vector_calls = 0
            self.closed = False

        def _values(self):
            return self.theta.detach().square().sum(dim=1) + 1.0

        def full_loss(self):
            return self.theta.square().sum() + 2.0

        def full_genewise_nll_and_grad(self, *, need_grad: bool):
            self.full_vector_calls += 1
            values = self._values()
            grad = 2.0 * self.theta.detach() if need_grad else None
            return values, grad

        def full_nll_per_family(self):
            raise AssertionError("final artifact writer should reuse final vector")

        def clamp_theta_(self, min_rate, max_rate):
            with torch.no_grad():
                self.theta.clamp_(min=-4.0, max=4.0)

        def solver_stat_records(self):
            return []

        def clear(self):
            pass

        def close(self):
            self.closed = True

    class FakeRunner(OptimizationRunner):
        def build_model(self):
            self.fake_model = FakeGenewiseVectorModel()
            return self.fake_model

    config = RunConfig(
        species_tree=tmp_path / "sp.nwk",
        families_file=tmp_path / "families.txt",
        out_dir=tmp_path / "out",
        mode="genewise",
        device="cpu",
        optimizer="adam",
        steps=1,
        lr=0.05,
        checkpoint_every=0,
        log_every=10,
        loss_patience=0,
        best_likelihood_patience=0,
    )
    runner = FakeRunner(config)

    result = runner.run()

    assert result.status == "not_converged"
    assert runner.fake_model.full_vector_calls == 1
    per_family = (config.out_dir / "per_fam_likelihoods.tsv").read_text(
        encoding="utf-8"
    )
    assert "fam0" in per_family
    assert "fam1" in per_family


def test_optimization_runner_preserves_final_artifacts_when_staging_fails(
    tmp_path: Path,
    monkeypatch,
):
    class FakeFinalArtifactModel:
        def __init__(self):
            self.theta = torch.nn.Parameter(
                torch.tensor([[0.25, -0.15, 0.05]], dtype=torch.float32)
            )
            self.family_names = ["fam0"]
            self.species_names = ["sp0", "sp1"]
            self.n_families = 1
            self.n_species = 2
            self.batch_metadata = [SimpleNamespace(batch_index=0)]
            self.closed = False

        def full_loss(self):
            return self.theta.square().sum() + 3.0

        def full_nll_per_family(self):
            return self.theta.detach().square().sum().reshape(1) + 2.0

        def clamp_theta_(self, min_rate, max_rate):
            with torch.no_grad():
                self.theta.clamp_(min=-4.0, max=4.0)

        def solver_stat_records(self):
            return []

        def clear(self):
            return None

        def close(self):
            self.closed = True

    class FakeFinalArtifactRunner(OptimizationRunner):
        def build_model(self):
            self.fake_model = FakeFinalArtifactModel()
            return self.fake_model

    config = RunConfig(
        species_tree=tmp_path / "sp.nwk",
        families_file=tmp_path / "families.txt",
        out_dir=tmp_path / "out",
        mode="genewise",
        device="cpu",
        optimizer="adam",
        steps=1,
        lr=0.05,
        checkpoint_every=0,
        log_every=10,
        loss_patience=0,
        best_likelihood_patience=0,
    )
    config.out_dir.mkdir(parents=True)
    stale_text_artifacts = {
        "rates_final.tsv": "stale rates",
        "per_fam_likelihoods.tsv": "stale likelihoods",
        "optimization_history.csv": "stale csv",
        "summary.json": "stale summary",
    }
    for name, contents in stale_text_artifacts.items():
        (config.out_dir / name).write_text(contents, encoding="utf-8")
    theta_path = config.out_dir / "theta_final.pt"
    theta_path.write_bytes(b"stale theta")

    def fail_write_rate_table(path: Path, *_args: object, **_kwargs: object) -> None:
        assert path.parent.name.startswith(".gpurec-optimization-stage-")
        raise OSError("rate table full")

    monkeypatch.setattr(optimize_workflow, "_write_rate_table", fail_write_rate_table)
    runner = FakeFinalArtifactRunner(config)

    with pytest.raises(OSError, match="rate table full"):
        runner.run()

    assert runner.fake_model.closed
    for name, contents in stale_text_artifacts.items():
        assert (config.out_dir / name).read_text(encoding="utf-8") == contents
    assert theta_path.read_bytes() == b"stale theta"
    history_rows = [
        json.loads(line)
        for line in (config.out_dir / "history.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["optimizer/phase"] for row in history_rows] == ["adam"]
    assert list(config.out_dir.glob(".gpurec-optimization-*")) == []


def test_optimization_runner_preserves_primary_error_when_close_fails(
    tmp_path: Path,
):
    class FakeFailingCloseModel:
        def __init__(self):
            self.theta = torch.nn.Parameter(
                torch.tensor([0.25, -0.15, 0.05], dtype=torch.float32)
            )
            self.close_calls = 0

        def full_loss(self):
            raise RuntimeError("primary optimization failure")

        def clamp_theta_(self, min_rate, max_rate):
            return None

        def close(self):
            self.close_calls += 1
            raise RuntimeError("close failure")

    class FakeFailingCloseRunner(OptimizationRunner):
        def build_model(self):
            self.fake_model = FakeFailingCloseModel()
            return self.fake_model

    config = RunConfig(
        species_tree=tmp_path / "sp.nwk",
        families_file=tmp_path / "families.txt",
        out_dir=tmp_path / "out",
        mode="global",
        device="cpu",
        optimizer="adam",
        steps=1,
        lr=0.05,
        checkpoint_every=0,
        log_every=10,
    )
    runner = FakeFailingCloseRunner(config)

    with pytest.raises(RuntimeError, match="primary optimization failure") as excinfo:
        runner.run()

    assert runner.fake_model.close_calls == 1
    assert isinstance(excinfo.value.__cause__, RuntimeError)
    assert str(excinfo.value.__cause__) == "close failure"


def test_optimization_runner_periodic_latest_uses_completed_step_cadence(
    tmp_path: Path,
):
    class FakeCadenceModel:
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

        def full_nll_per_family(self):
            return self.theta.detach().square().sum().reshape(1) + 2.0

        def clamp_theta_(self, min_rate, max_rate):
            with torch.no_grad():
                self.theta.clamp_(min=-4.0, max=4.0)

        def solver_stat_records(self):
            return []

        def clear(self):
            return None

        def close(self):
            self.closed = True

    class FakeCadenceRunner(OptimizationRunner):
        def __init__(self, config):
            super().__init__(config)
            self.saves: list[tuple[str, int, str | None]] = []

        def build_model(self):
            self.fake_model = FakeCadenceModel()
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
            self.saves.append(
                (
                    Path(path).name,
                    int(step),
                    None if row is None else str(row.get("optimizer/phase")),
                )
            )
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

    config = RunConfig(
        species_tree=tmp_path / "sp.nwk",
        families_file=tmp_path / "families.txt",
        out_dir=tmp_path / "out",
        mode="global",
        device="cpu",
        optimizer="adam",
        steps=3,
        lr=0.05,
        checkpoint_every=2,
        log_every=10,
        loss_patience=0,
        best_likelihood_patience=0,
    )
    runner = FakeCadenceRunner(config)

    runner.run()

    periodic_latest_saves = [
        (name, step, phase)
        for name, step, phase in runner.saves
        if name == "latest.pt" and phase != "final_eval"
    ]
    assert periodic_latest_saves == [("latest.pt", 1, "adam")]
    assert runner.fake_model.closed


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


def test_optimization_runner_completed_resume_only_refreshes_final_artifacts(
    tmp_path: Path,
):
    species_tree = tmp_path / "sp.nwk"
    families_file = tmp_path / "families.txt"
    first_config = _optimizer_mode_config(
        tmp_path,
        optimizer="adam",
        species_tree=species_tree,
        families_file=families_file,
        out_dir=tmp_path / "completed-source",
        steps=1,
    )
    first_runner = _WorkflowOptimizerModeRunner(first_config)
    first_runner.run()
    first_latest_path = first_config.out_dir / "checkpoints" / "latest.pt"
    first_latest = load_checkpoint(first_latest_path)

    assert int(first_latest["next_step"]) == 1
    assert first_latest["last_row"]["optimizer/phase"] == "final_eval"

    resumed_config = _optimizer_mode_config(
        tmp_path,
        optimizer="adam",
        species_tree=species_tree,
        families_file=families_file,
        out_dir=tmp_path / "completed-resume",
        steps=1,
        resume_from=first_latest_path,
    )
    resumed_runner = _WorkflowOptimizerModeRunner(resumed_config)

    result = resumed_runner.run()

    assert result.status == "not_converged"
    assert result.reason == "max_steps"
    assert result.steps_completed == 1
    assert resumed_runner.fake_model.closed
    history_rows = _optimizer_mode_history_rows(resumed_config.out_dir)
    assert [(row["optimizer/phase"], row["step"]) for row in history_rows] == [
        ("final_eval", 1),
    ]
    latest = load_checkpoint(resumed_config.out_dir / "checkpoints" / "latest.pt")
    assert latest["status"]["status"] == "not_converged"
    assert latest["status"]["reason"] == "max_steps"
    assert latest["last_row"]["optimizer/phase"] == "final_eval"
    assert int(latest["next_step"]) == 1


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
        def __init__(self, *, fail_with: type[Exception] | None = None):
            self.fail_with = fail_with
            self.loaded = None

        def load_state_dict(self, state):
            if self.fail_with is not None:
                raise self.fail_with("incompatible optimizer state")
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
        FakeOptimizer(fail_with=ValueError),
        {"state": ["bad"]},
    )
    assert discarded["resume_optimizer_state"] == "discarded"
    assert "incompatible optimizer state" in discarded["resume_optimizer_error"]

    for error_type in (RuntimeError, TypeError):
        discarded = runner._restore_optimizer_state(
            FakeOptimizer(fail_with=error_type),
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
            "step": 0,
            "next_step": 1,
            "config": config.to_dict(),
            "family_names": [],
            "species_names": [],
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
        device="cuda",
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
            "step": 0,
            "next_step": 1,
            "config": config.to_dict(),
            "family_names": [],
            "species_names": [],
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
    ("identity_case", "message"),
    [
        ("family_names", "family_names differ"),
        ("species_names", "species_names differ"),
        ("species_tree", r"config\.species_tree differs"),
        ("families_file", r"config\.families_file differs"),
        ("mode", r"config\.mode differs"),
        ("start", r"config\.start differs"),
        ("max_families", r"config\.max_families differs"),
    ],
)
def test_optimization_runner_resume_rejects_incompatible_checkpoint_identity(
    tmp_path: Path,
    monkeypatch,
    identity_case: str,
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
            "step": 0,
            "next_step": 1,
            "config": config.to_dict(),
            "family_names": ["fam_b", "fam_a"],
            "species_names": ["sp_b", "sp_a"],
            "status": {
                "previous_objective": 1.5,
                "stable_loss_steps": 0,
            },
        }
        if identity_case == "family_names":
            payload["family_names"] = ["fam_a", "fam_b"]
        elif identity_case == "species_names":
            payload["species_names"] = ["sp_a", "sp_b"]
        else:
            config_updates = {
                "species_tree": tmp_path / "other_sp.nwk",
                "families_file": tmp_path / "other_families.txt",
                "mode": "genewise",
                "start": 1,
                "max_families": 1,
            }
            payload["config"] = {
                **payload["config"],
                identity_case: config_updates[identity_case],
            }
        return payload

    workflow_optimize_module = importlib.import_module("gpurec.workflow.optimize")
    monkeypatch.setattr(workflow_optimize_module, "load_checkpoint", fake_load_checkpoint)
    runner = FakeResumeRunner(config)

    with pytest.raises(RuntimeError, match=message):
        runner.run()

    assert runner.fake_model.closed


def test_optimization_runner_resume_rejects_legacy_checkpoint_without_identity(
    tmp_path: Path,
    monkeypatch,
):
    class FakeResumeModel:
        def __init__(self):
            self.theta = torch.nn.Parameter(torch.zeros(2, 3, dtype=torch.float32))
            self.family_names = ["fam0", "fam1"]
            self.species_names = ["sp0", "sp1"]
            self.closed = False

        def clear(self):
            raise AssertionError("theta should not be restored from legacy metadata")

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
        resume_from=tmp_path / "legacy.pt",
        checkpoint_every=0,
        log_every=10,
    )

    def fake_load_checkpoint(path, *, map_location):
        return {
            "theta": torch.zeros(2, 3, dtype=torch.float32),
            "optimizer_state": None,
            "step": 0,
            "next_step": 1,
            "config": {},
            "status": {},
        }

    workflow_optimize_module = importlib.import_module("gpurec.workflow.optimize")
    monkeypatch.setattr(workflow_optimize_module, "load_checkpoint", fake_load_checkpoint)
    runner = FakeResumeRunner(config)

    with pytest.raises(RuntimeError, match="config.*identity"):
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
            "step": 4,
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


def test_optimization_runner_resume_rejects_inconsistent_progress_before_restore(
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
        steps=10,
        resume_from=tmp_path / "resume.pt",
        checkpoint_every=0,
        log_every=10,
    )

    def fake_load_checkpoint(path, *, map_location):
        return {
            "theta": torch.ones(3, dtype=torch.float32),
            "optimizer_state": None,
            "step": 5,
            "next_step": 0,
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

    with pytest.raises(RuntimeError, match="inconsistent progress metadata"):
        runner.run()

    assert runner.fake_model.closed
    assert torch.equal(runner.fake_model.theta.detach(), torch.zeros(3))


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
