from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from gpurec.optimization.lbfgsb_schilling import run_schilling_case


DATA_DIR = Path(__file__).parents[1] / "data" / "lbfgsb_schilling"

TOLERANCES = {
    "projgr": (0.0, 0.0),
    "active": (0.0, 0.0),
    "bmv": (1e-14, 1e-14),
    "cauchy": (1e-13, 1e-13),
    "subsm": (1e-13, 1e-13),
}


def _subroutine_name(path: Path) -> str:
    return path.stem.split("_case_", maxsplit=1)[0]


def _float_matches(actual: float, expected: float, abs_tol: float, rel_tol: float) -> bool:
    if abs_tol == 0.0 and rel_tol == 0.0:
        return np.float64(actual).tobytes() == np.float64(expected).tobytes()
    return abs(actual - expected) <= abs_tol + rel_tol * max(
        abs(actual),
        abs(expected),
        1.0,
    )


def _assert_matches(
    actual: Any,
    expected: Any,
    *,
    abs_tol: float,
    rel_tol: float,
    path: str,
) -> None:
    if isinstance(expected, list):
        assert isinstance(actual, list), path
        assert len(actual) == len(expected), path
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            _assert_matches(
                actual_item,
                expected_item,
                abs_tol=abs_tol,
                rel_tol=rel_tol,
                path=f"{path}[{index}]",
            )
        return
    if isinstance(expected, bool):
        assert bool(actual) is expected, path
        return
    if isinstance(expected, int):
        assert int(actual) == expected, path
        return
    if isinstance(expected, float):
        assert _float_matches(
            float(actual),
            expected,
            abs_tol,
            rel_tol,
        ), f"{path}: {actual!r} != {expected!r}"
        return
    assert actual == expected, path


@pytest.mark.parametrize(
    "case_path",
    sorted(DATA_DIR.glob("*_case_*.json")),
    ids=lambda path: path.stem,
)
def test_lbfgsb_schilling_spec_vectors(case_path: Path):
    case = json.loads(case_path.read_text(encoding="utf-8"))
    subroutine = _subroutine_name(case_path)
    abs_tol, rel_tol = TOLERANCES[subroutine]

    actual = run_schilling_case(subroutine, case["inputs"])

    for key, expected in case["expected"].items():
        assert key in actual
        _assert_matches(
            actual[key],
            expected,
            abs_tol=abs_tol,
            rel_tol=rel_tol,
            path=f"{case_path.name}:{key}",
        )
