from pathlib import Path
import importlib.util
from types import SimpleNamespace

import pytest


_SCRIPT = Path(__file__).resolve().parents[1] / "benchmarks/large_dataset_capacity/run_gpurec_benchmark.py"
_SPEC = importlib.util.spec_from_file_location("run_gpurec_benchmark", _SCRIPT)
assert _SPEC is not None
assert _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
parse_args = _MODULE.parse_args
solver_options_from_args = _MODULE.solver_options_from_args
SelfLoopBackwardRecorder = _MODULE.SelfLoopBackwardRecorder


def _base_args(tmp_path: Path) -> list[str]:
    return [
        "--dataset-name",
        "smoke",
        "--species-tree",
        str(tmp_path / "species.nwk"),
        "--gene-tree-dir",
        str(tmp_path),
        "--output",
        str(tmp_path / "out.json"),
    ]


def test_run_gpurec_benchmark_wires_gmres_solver_options(tmp_path: Path):
    args = parse_args(
        _base_args(tmp_path)
        + [
            "--self-loop-solver",
            "gmres",
            "--neumann-terms",
            "32",
            "--gmres-max-iter",
            "10",
            "--gmres-tol",
            "1e-8",
            "--gmres-check-interval",
            "3",
            "--gmres-reuse-check-schedule",
            "--gmres-trust-check-schedule",
            "--gmres-reuse-solution",
            "--gmres-solution-cache-min-iterations",
            "4",
            "--gmres-preconditioner",
            "diagonal",
            "--gmres-diagonal-preconditioner-floor",
            "1e-5",
        ]
    )

    options = solver_options_from_args(args)

    assert options.self_loop_solver == "gmres"
    assert options.neumann_terms == 10
    assert options.gmres_tol == 1e-8
    assert options.gmres_check_interval == 3
    assert options.gmres_reuse_check_schedule is True
    assert options.gmres_trust_check_schedule is True
    assert options.gmres_reuse_solution is True
    assert options.gmres_solution_cache_min_iterations == 4
    assert options.gmres_preconditioner == "diagonal"
    assert options.gmres_diagonal_preconditioner_floor == 1e-5


def test_run_gpurec_benchmark_gmres_defaults_to_neumann_terms(tmp_path: Path):
    args = parse_args(
        _base_args(tmp_path)
        + [
            "--self-loop-solver",
            "gmres_fixed",
            "--neumann-terms",
            "12",
        ]
    )

    options = solver_options_from_args(args)

    assert options.self_loop_solver == "gmres_fixed"
    assert options.neumann_terms == 12


def test_run_gpurec_benchmark_rejects_gmres_max_iter_for_neumann(tmp_path: Path):
    with pytest.raises(SystemExit):
        parse_args(_base_args(tmp_path) + ["--gmres-max-iter", "10"])


def test_self_loop_backward_recorder_summarizes_neumann_iterations(tmp_path: Path):
    args = parse_args(_base_args(tmp_path) + ["--neumann-terms", "12"])
    model = SimpleNamespace(
        solver_options=solver_options_from_args(args),
        batch_statics=[
            SimpleNamespace(wave_layout={"wave_metas": [object(), object()]}),
            SimpleNamespace(wave_layout={"wave_metas": [object()]}),
        ],
    )
    recorder = SelfLoopBackwardRecorder(model)
    recorder.backward_pass_count = 2

    summary = recorder.summary()

    assert summary["self_loop_solver"] == "neumann"
    assert summary["self_loop_backward_pass_count"] == 2
    assert summary["self_loop_waves_per_backward"] == 3
    assert summary["self_loop_wave_solves"] == 6
    assert summary["self_loop_backward_iterations"] == 72
    assert summary["self_loop_mean_iterations_per_wave"] == 12.0
    assert summary["gmres_total_checks"] is None


def test_self_loop_backward_recorder_summarizes_gmres_iterations(tmp_path: Path):
    args = parse_args(
        _base_args(tmp_path)
        + [
            "--self-loop-solver",
            "gmres",
            "--gmres-max-iter",
            "10",
        ]
    )
    model = SimpleNamespace(
        solver_options=solver_options_from_args(args),
        batch_statics=[SimpleNamespace(wave_layout={"wave_metas": [object(), object()]})],
    )
    recorder = SelfLoopBackwardRecorder(model)
    recorder.backward_pass_count = 1
    recorder._gmres_stats = [
        {"iterations": 3, "check_count": 2, "rel_res": 1e-4, "arnoldi_backend": "triton_split"},
        {
            "iterations": 0,
            "a_applications": 1,
            "check_count": 1,
            "rel_res": 2e-5,
            "arnoldi_backend": "warm_start",
            "warm_start_used": True,
            "warm_start_accepted": True,
            "residual_probe_a_applications": 1,
        },
    ]

    summary = recorder.summary()

    assert summary["self_loop_solver"] == "gmres"
    assert summary["self_loop_wave_solves"] == 2
    assert summary["self_loop_backward_iterations"] == 4
    assert summary["self_loop_mean_iterations_per_wave"] == 2.0
    assert summary["self_loop_max_iterations_per_wave"] == 3
    assert summary["gmres_krylov_iterations"] == 3
    assert summary["gmres_warm_start_used"] == 1
    assert summary["gmres_warm_start_accepted"] == 1
    assert summary["gmres_trusted_check_used"] == 0
    assert summary["gmres_residual_probe_a_applications"] == 1
    assert summary["gmres_total_checks"] == 3
    assert summary["gmres_residual_cpu_readbacks"] == 3
    assert summary["gmres_max_rel_res"] == 1e-4
    assert summary["gmres_arnoldi_backend_counts"] == {"triton_split": 1, "warm_start": 1}


def test_self_loop_backward_recorder_ignores_trusted_check_residuals(tmp_path: Path):
    args = parse_args(
        _base_args(tmp_path)
        + [
            "--self-loop-solver",
            "gmres",
            "--gmres-max-iter",
            "10",
        ]
    )
    model = SimpleNamespace(
        solver_options=solver_options_from_args(args),
        batch_statics=[SimpleNamespace(wave_layout={"wave_metas": [object()]})],
    )
    recorder = SelfLoopBackwardRecorder(model)
    recorder.backward_pass_count = 1
    recorder._gmres_stats = [
        {
            "iterations": 2,
            "check_count": 1,
            "rel_res": 1.0,
            "trusted_check_used": True,
            "arnoldi_backend": "triton_large",
        },
    ]

    summary = recorder.summary()

    assert summary["gmres_trusted_check_used"] == 1
    assert summary["gmres_residual_cpu_readbacks"] == 0
    assert summary["gmres_max_rel_res"] is None
