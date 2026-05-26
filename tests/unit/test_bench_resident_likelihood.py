from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
BENCH_PATH = ROOT / "profiling" / "bench_resident_likelihood.py"


def _load_bench_module():
    spec = importlib.util.spec_from_file_location(
        "bench_resident_likelihood_under_test",
        BENCH_PATH,
    )
    if spec is None or spec.loader is None:
        raise AssertionError("unable to load benchmark module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_resident_likelihood_parser_accepts_adaptive_iteration_controls():
    bench = _load_bench_module()

    args = bench._build_parser().parse_args(
        [
            "--adaptive-iters",
            "--convergence-check-interval",
            "8",
            "--e-logsumexp-tol",
            "0",
            "--pi-max-diff-tol",
            "1e-4",
        ]
    )

    assert args.adaptive_iters is True
    assert args.convergence_check_interval == 8
    assert args.e_logsumexp_tol == pytest.approx(0.0)
    assert args.pi_max_diff_tol == pytest.approx(1e-4)


def test_resident_likelihood_validation_rejects_odd_adaptive_check_interval():
    bench = _load_bench_module()
    args = bench._build_parser().parse_args(
        [
            "--adaptive-iters",
            "--convergence-check-interval",
            "3",
        ]
    )

    with pytest.raises(ValueError, match="even --convergence-check-interval"):
        bench._validate_args(args)


def test_resident_likelihood_main_forwards_adaptive_iteration_controls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    bench = _load_bench_module()
    dataset = tmp_path / "trees"
    dataset.mkdir()
    (dataset / "sp.nwk").write_text("(A:1,B:1)Root:0;\n", encoding="utf-8")
    (dataset / "g_0.nwk").write_text("(A_a:1,B_b:1)g:0;\n", encoding="utf-8")
    seen_kwargs: dict[str, object] = {}

    class FakeTheta:
        grad = None

    class FakeModel:
        n_families = 1
        n_species = 2
        theta = FakeTheta()
        cached_static_states: list[object] = []
        batch_metadata = [
            SimpleNamespace(
                clade_count=3,
                wave_count=2,
                max_wave_size=2,
            )
        ]

        def __init__(self):
            self.budgets: list[dict[str, object]] = []
            self.closed = False

        def materialize_batches(self):
            return list(self.batch_metadata)

        def configure_solver_iterations(self, **kwargs):
            self.budgets.append(dict(kwargs))

        def close(self):
            self.closed = True

    fake_model = FakeModel()

    def fake_from_trees(**kwargs):
        seen_kwargs.update(kwargs)
        return fake_model

    monkeypatch.setattr(bench.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(bench.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(bench.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(
        bench.GeneReconModel,
        "from_trees",
        staticmethod(fake_from_trees),
    )
    monkeypatch.setattr(bench, "_time_loss_only", lambda _model: (0.25, 123.0))

    assert (
        bench.main(
            [
                "--dataset",
                str(dataset),
                "--fams",
                "1",
                "--fixed-iters",
                "4",
                "--warmups",
                "0",
                "--reps",
                "1",
                "--adaptive-iters",
                "--convergence-check-interval",
                "8",
                "--e-logsumexp-tol",
                "0",
                "--pi-max-diff-tol",
                "0.0001",
            ]
        )
        == 0
    )

    assert seen_kwargs["adaptive_iters"] is True
    assert seen_kwargs["convergence_check_interval"] == 8
    assert seen_kwargs["e_logsumexp_tol"] == pytest.approx(0.0)
    assert seen_kwargs["pi_max_diff_tol"] == pytest.approx(0.0001)
    assert fake_model.budgets == [
        {
            "fixed_iters_E": 4,
            "fixed_iters_Pi": 4,
            "neumann_terms": 4,
            "adaptive_neumann_terms": False,
        }
    ]
    records = [
        json.loads(line)
        for line in capsys.readouterr().out.splitlines()
    ]
    model_record = next(record for record in records if record["event"] == "model")
    assert model_record["adaptive_iters"] is True
    assert model_record["convergence_check_interval"] == 8
    assert model_record["e_logsumexp_tol"] == pytest.approx(0.0)
    assert model_record["pi_max_diff_tol"] == pytest.approx(0.0001)
