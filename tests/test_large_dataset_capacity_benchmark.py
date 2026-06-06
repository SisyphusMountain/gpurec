from pathlib import Path
import importlib.util

import pytest


_SCRIPT = Path(__file__).resolve().parents[1] / "benchmarks/large_dataset_capacity/run_gpurec_benchmark.py"
_SPEC = importlib.util.spec_from_file_location("run_gpurec_benchmark", _SCRIPT)
assert _SPEC is not None
assert _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
parse_args = _MODULE.parse_args
solver_options_from_args = _MODULE.solver_options_from_args


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
        ]
    )

    options = solver_options_from_args(args)

    assert options.self_loop_solver == "gmres"
    assert options.neumann_terms == 10
    assert options.gmres_tol == 1e-8
    assert options.gmres_check_interval == 3


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
