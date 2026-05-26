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


def _set_bench_argv(monkeypatch: pytest.MonkeyPatch, bench, *args: str) -> None:
    monkeypatch.setattr(
        bench.sys,
        "argv",
        ["bench_uniform_forward_backward_pipeline.py", *args],
    )


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
    assert "rss_mib" in payload
    assert "disk_free_gib" in payload
    assert "cuda_allocated_gib" in payload


def test_progress_jsonl_is_quiet_when_not_requested(capsys: pytest.CaptureFixture[str]):
    bench = _load_bench_module()
    args = argparse.Namespace(progress_jsonl=False)

    bench._emit_progress(args, "unit_event", families=1000)

    assert capsys.readouterr().out == ""


def test_make_static_inputs_progress_reports_setup_sizes(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
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
        seen_kwargs = {}

        def __init__(self, **kwargs):
            if kwargs:
                self.seen_kwargs = dict(kwargs)
                FakeDataset.seen_kwargs = self.seen_kwargs

        def _species_helpers_for_mode(self, **_kwargs):
            return {}, None

        @classmethod
        def _from_preprocessed_raw(cls, **kwargs):
            cls.seen_kwargs = dict(kwargs)
            return cls()

    class FakeRustPreprocessed:
        build_requests: list[dict[str, object]] = []

        def to_torch(self):
            return {}

        def family_basic_counts(self):
            return {
                "clade_counts": [11, 13],
                "split_counts": [17, 19],
            }

        def build_chunked_layouts(self, **kwargs):
            self.build_requests.append(dict(kwargs))
            return [{"unit": True}]

    fake_rust_preprocessed = FakeRustPreprocessed()

    class FakeRustExtension:
        def preprocess_dataset(self, *_args, **_kwargs):
            return fake_rust_preprocessed

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
    monkeypatch.setattr(bench, "RustPreprocessExtension", FakeRustExtension)
    monkeypatch.setattr(bench, "_selected_gene_paths", lambda *_args, **_kwargs: ["g0", "g1"])
    monkeypatch.setattr(bench, "choose_uniform_pipeline_policy", lambda *_args, **_kwargs: policy)
    monkeypatch.setattr(bench, "_built_chunks_from_rust", lambda *_args, **_kwargs: [built])

    args = argparse.Namespace(
        progress_jsonl=True,
        dataset="fake_dataset",
        start=0,
        fams=2,
        dtype=bench.torch.float32,
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
        "chunk_built",
        "static_inputs_done",
    ]
    assert events[2]["families"] == 2
    assert events[2]["S"] == 300
    assert events[2]["total_clades"] == 24
    assert events[3]["chunks"] == 1
    assert events[3]["family_chunk_size"] == 2
    assert events[4]["max_wave_split_rows"] == 19
    assert events[5]["total_splits"] == 36
    assert static.built_chunks == [built]
    assert FakeDataset.seen_kwargs["family_names"] == ["g0", "g1"]
    assert fake_rust_preprocessed.build_requests == [
        {
            "family_chunk_size": 2,
            "clade_budget": None,
            "batch_packing": "sequential",
            "max_wave_size": 128,
            "max_root_wave_size": None,
            "dtype": "float32",
        }
    ]


def test_setup_only_alias_maps_to_preflight_flag(monkeypatch: pytest.MonkeyPatch):
    bench = _load_bench_module()
    monkeypatch.setattr(
        bench.sys,
        "argv",
        ["bench_uniform_forward_backward_pipeline.py", "--setup-only"],
    )

    args = bench._parse_args()

    assert args.preflight_only is True


def test_preflight_window_size_arg(monkeypatch: pytest.MonkeyPatch):
    bench = _load_bench_module()
    monkeypatch.setattr(
        bench.sys,
        "argv",
        [
            "bench_uniform_forward_backward_pipeline.py",
            "--preflight-window-size",
            "128",
        ],
    )

    args = bench._parse_args()

    assert args.preflight_window_size == 128


def test_preflight_window_size_rejects_negative(monkeypatch: pytest.MonkeyPatch):
    bench = _load_bench_module()
    _set_bench_argv(monkeypatch, bench, "--preflight-window-size", "-1")

    with pytest.raises(SystemExit) as excinfo:
        bench._parse_args()

    assert excinfo.value.code == 2


def test_compute_e_and_params_passes_explicit_e_shape(monkeypatch: pytest.MonkeyPatch):
    bench = _load_bench_module()
    calls: list[dict[str, object]] = []

    def fake_e_fixed_point(**kwargs):
        calls.append(kwargs)
        return {"E": bench.torch.full((4,), -1.0), "iterations": 1}

    monkeypatch.setattr(bench, "E_fixed_point", fake_e_fixed_point)
    static = SimpleNamespace(
        theta=bench.torch.zeros(3, dtype=bench.torch.float64),
        unnorm_row_max=bench.torch.zeros(4, dtype=bench.torch.float64),
        species_helpers={"S": 4},
        dataset=SimpleNamespace(S=4),
        dtype=bench.torch.float64,
        device=bench.torch.device("cpu"),
        ancestors_T=bench.torch.eye(4, dtype=bench.torch.float64),
    )
    args = argparse.Namespace(max_iters_E=7, tol_E=1e-8)

    e_out, params = bench._compute_e_and_params(static, args)

    assert e_out["E"].shape == (4,)
    assert len(params) == 4
    assert calls[0]["e_shape"] == (4,)


@pytest.mark.parametrize(
    ("flag", "value", "message"),
    [
        ("--start", "-1", "start must be non-negative"),
        ("--fams", "0", "fams must be positive"),
        ("--family-chunk-size", "0", "family-chunk-size must be positive"),
        ("--family-chunk-size", "-1", "family-chunk-size must be positive"),
        ("--max-wave-size", "-1", "max-wave-size must be positive"),
        ("--fixed-iters", "-1", "fixed-iters must be positive"),
        ("--reps", "0", "reps must be positive"),
        ("--warmups", "-1", "warmups must be non-negative"),
        ("--max-iters-E", "0", "max-iters-E must be positive"),
        ("--neumann-terms", "0", "neumann-terms must be positive"),
        (
            "--preflight-window-size",
            "-1",
            "preflight-window-size must be non-negative",
        ),
        (
            "--compare-unchunked-max-fams",
            "-1",
            "compare-unchunked-max-fams must be non-negative",
        ),
    ],
)
def test_parse_args_rejects_invalid_count_controls(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    flag: str,
    value: str,
    message: str,
):
    bench = _load_bench_module()
    _set_bench_argv(monkeypatch, bench, flag, value)

    with pytest.raises(SystemExit) as excinfo:
        bench._parse_args()

    assert excinfo.value.code == 2
    assert message in capsys.readouterr().err


def test_windowed_preflight_runs_sequential_setup_windows_and_reports_progress(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    bench = _load_bench_module()
    calls = []

    def fake_run_static_preflight(window_args):
        calls.append((window_args.start, window_args.fams))
        window_args.family_chunk_size = 99
        window_args.max_wave_size = 88
        return object()

    monkeypatch.setattr(
        bench,
        "_selected_gene_paths",
        lambda *_args, **_kwargs: [f"g_{idx}.nwk" for idx in range(5)],
    )
    monkeypatch.setattr(bench, "_run_static_preflight", fake_run_static_preflight)
    monkeypatch.setattr(
        bench,
        "_static_progress_summary",
        lambda _static, window_args: {
            "family_start": window_args.start,
            "families": window_args.fams,
        },
    )
    monkeypatch.setattr(bench.torch.cuda, "empty_cache", lambda: None)

    args = argparse.Namespace(
        dataset="fake_dataset",
        start=10,
        fams=5,
        preflight_window_size=2,
        progress_jsonl=True,
        family_chunk_size="auto",
        max_wave_size="auto",
    )

    bench._run_windowed_preflight(args)

    assert calls == [(10, 2), (12, 2), (14, 1)]
    assert args.family_chunk_size == "auto"
    assert args.max_wave_size == "auto"
    output_lines = capsys.readouterr().out.splitlines()
    assert any(
        "windowed_preflight" in line and "performance_evidence 0" in line
        for line in output_lines
    )
    events = [
        json.loads(line)["event"]
        for line in output_lines
        if line.startswith("{")
    ]
    assert events == [
        "windowed_preflight_start",
        "preflight_window_start",
        "preflight_window_done",
        "preflight_window_start",
        "preflight_window_done",
        "preflight_window_start",
        "preflight_window_done",
        "windowed_preflight_done",
    ]
