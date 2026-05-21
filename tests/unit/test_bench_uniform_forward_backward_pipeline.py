from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
BENCH_PATH = ROOT / "profiling" / "bench_uniform_forward_backward_pipeline.py"


def _load_bench_module():
    spec = importlib.util.spec_from_file_location(
        "bench_uniform_forward_backward_pipeline_under_test",
        BENCH_PATH,
    )
    if spec is None or spec.loader is None:
        raise AssertionError("unable to load benchmark module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_progress_jsonl_emits_parseable_flushed_record(capsys: pytest.CaptureFixture[str]):
    bench = _load_bench_module()
    args = argparse.Namespace(progress_jsonl=True)

    bench._emit_progress(args, "unit_event", families=1000, max_wave=8192)

    line = capsys.readouterr().out.strip()
    payload = json.loads(line)
    assert payload["record"] == "bench_uniform_forward_backward_pipeline"
    assert payload["event"] == "unit_event"
    assert payload["families"] == 1000
    assert payload["max_wave"] == 8192
    assert isinstance(payload["time_s"], float)


def test_progress_jsonl_is_quiet_when_not_requested(capsys: pytest.CaptureFixture[str]):
    bench = _load_bench_module()
    args = argparse.Namespace(progress_jsonl=False)

    bench._emit_progress(args, "unit_event", families=1000)

    assert capsys.readouterr().out == ""


def test_dataset_progress_hook_prefixes_benchmark_events(
    capsys: pytest.CaptureFixture[str],
):
    bench = _load_bench_module()
    args = argparse.Namespace(progress_jsonl=True)

    hook = bench._make_dataset_progress_hook(args)
    assert hook is not None
    hook("family_cache_miss", idx=3, family="fam3")

    payload = json.loads(capsys.readouterr().out)
    assert payload["event"] == "dataset_preprocess_family_cache_miss"
    assert payload["idx"] == 3
    assert payload["family"] == "fam3"


def test_dataset_progress_hook_is_disabled_without_progress_jsonl():
    bench = _load_bench_module()
    args = argparse.Namespace(progress_jsonl=False)

    assert bench._make_dataset_progress_hook(args) is None


def test_make_static_inputs_progress_reports_setup_sizes(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
):
    bench = _load_bench_module()

    class FakeDevice:
        type = "cuda"

        def __str__(self) -> str:
            return "cuda"

    class FakeTensor:
        def to(self, **_kwargs):
            return self

    class FakeDataset:
        S = 300
        families = [
            {"C": 11, "N_splits": 17},
            {"C": 13, "N_splits": 19},
        ]
        unnorm_row_max = FakeTensor()

        def __init__(self, **_kwargs):
            pass

        def _species_helpers_for_mode(self, **_kwargs):
            return {}, None

    spec = SimpleNamespace(indices=[0, 1], clades=24, splits=36)
    built = SimpleNamespace(
        spec=spec,
        waves=4,
        max_wave=11,
        split_rows=36,
        max_wave_split_rows=19,
        wave_layout={"root_clade_ids": [0, 1]},
    )
    policy = SimpleNamespace(
        family_chunk_size=2,
        max_wave_size=128,
        reason="unit",
        estimated_payload_bytes=1024,
    )

    monkeypatch.setattr(bench.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(bench.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(bench.torch, "device", lambda _name: FakeDevice())
    monkeypatch.setattr(bench.torch, "tensor", lambda *_args, **_kwargs: FakeTensor())
    monkeypatch.setattr(bench.torch, "log2", lambda value: value)
    monkeypatch.setattr(bench, "GeneDataset", FakeDataset)
    monkeypatch.setattr(bench, "_selected_gene_paths", lambda *_args, **_kwargs: ["g0", "g1"])
    monkeypatch.setattr(bench, "choose_uniform_pipeline_policy", lambda *_args, **_kwargs: policy)
    monkeypatch.setattr(bench, "_make_chunks", lambda *_args, **_kwargs: [spec])
    monkeypatch.setattr(bench, "_build_chunk", lambda *_args, **_kwargs: built)

    args = argparse.Namespace(
        progress_jsonl=True,
        dataset="fake_dataset",
        start=0,
        fams=2,
        dtype=bench.torch.float32,
        cache_dir=str(tmp_path / "cache"),
        theta_rate=0.05,
        family_chunk_size="auto",
        max_wave_size="auto",
        fixed_iters=6,
    )

    static = bench._make_static_inputs(args)

    events = [
        json.loads(line)
        for line in capsys.readouterr().out.splitlines()
    ]
    assert [event["event"] for event in events] == [
        "static_inputs_start",
        "gene_selection_done",
        "dataset_loaded",
        "chunk_policy_selected",
        "chunk_build_start",
        "chunk_built",
        "static_inputs_done",
    ]
    assert events[2]["families"] == 2
    assert events[2]["S"] == 300
    assert events[2]["total_clades"] == 24
    assert events[3]["chunks"] == 1
    assert events[3]["family_chunk_size"] == 2
    assert events[5]["max_wave_split_rows"] == 19
    assert events[6]["total_splits"] == 36
    assert static.built_chunks == [built]


def test_setup_only_alias_maps_to_preflight_flag(monkeypatch: pytest.MonkeyPatch):
    bench = _load_bench_module()
    monkeypatch.setattr(
        bench.sys,
        "argv",
        ["bench_uniform_forward_backward_pipeline.py", "--setup-only"],
    )

    args = bench._parse_args()

    assert args.preflight_only is True
