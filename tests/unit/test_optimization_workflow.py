from __future__ import annotations

from pathlib import Path

import pytest

from gpurec.workflow.config import RunConfig
from gpurec.workflow.optimize import _step_stopping_status


def _run_config(tmp_path: Path, **overrides: object) -> RunConfig:
    values = {
        "species_tree": tmp_path / "sp.nwk",
        "families_file": tmp_path / "families.txt",
        "out_dir": tmp_path / "out",
        "device": "cuda",
        "grad_inf_tol": 0.01,
        "loss_patience": 0,
        "best_likelihood_patience": 0,
    }
    values.update(overrides)
    return RunConfig(**values)


@pytest.mark.parametrize(
    ("overrides", "grad_inf", "stable_loss_steps", "best_step", "step", "expected"),
    [
        (
            {"grad_inf_tol": 0.1, "loss_patience": 1, "best_likelihood_patience": 1},
            0.1,
            1,
            0,
            1,
            {"status": "converged", "reason": "gradient_tolerance"},
        ),
        (
            {"loss_patience": 2, "best_likelihood_patience": 1},
            1.0,
            2,
            0,
            2,
            {"status": "stalled", "reason": "loss_change_patience"},
        ),
        (
            {"best_likelihood_patience": 3},
            1.0,
            0,
            2,
            5,
            {"status": "stalled", "reason": "best_likelihood_patience"},
        ),
        (
            {"loss_patience": 2, "best_likelihood_patience": 3},
            1.0,
            1,
            4,
            5,
            None,
        ),
    ],
)
def test_step_stopping_status_matches_optimizer_loop_order(
    tmp_path: Path,
    overrides: dict[str, object],
    grad_inf: float,
    stable_loss_steps: int,
    best_step: int | None,
    step: int,
    expected: dict[str, str] | None,
):
    status = _step_stopping_status(
        _run_config(tmp_path, **overrides),
        step=step,
        grad_inf=grad_inf,
        stable_loss_steps=stable_loss_steps,
        best_step=best_step,
    )

    assert status == expected
