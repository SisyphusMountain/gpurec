from __future__ import annotations

import math
from pathlib import Path

import pytest

from gpurec.workflow.config import RunConfig
from gpurec.workflow.optimize import (
    _ResumeState,
    _resume_state_from_payload,
    _step_stopping_status,
)


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


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({}, r"invalid step"),
        ({"step": 0}, r"invalid next_step"),
        ({"step": 5, "next_step": 0}, r"inconsistent progress metadata"),
    ],
)
def test_resume_state_from_payload_requires_progress_metadata(
    tmp_path: Path,
    payload: dict[str, object],
    message: str,
):
    with pytest.raises(RuntimeError, match=message):
        _resume_state_from_payload(tmp_path / "resume.pt", payload)


def test_resume_state_from_payload_normalizes_checkpoint_metadata(tmp_path: Path):
    state = _resume_state_from_payload(
        tmp_path / "resume.pt",
        {
            "step": 2,
            "next_step": 3.0,
            "status": {
                "best_nll_bits": 12,
                "best_step": 2.0,
                "previous_objective": 13.5,
                "stable_loss_steps": 4.0,
            },
        },
    )

    assert state == _ResumeState(
        start_step=3,
        best_nll=12.0,
        best_step=2,
        previous_objective=13.5,
        stable_loss_steps=4,
    )


@pytest.mark.parametrize("status", [None, {}])
def test_resume_state_from_payload_defaults_optional_status_metadata(
    tmp_path: Path,
    status: dict[str, object] | None,
):
    payload: dict[str, object] = {"step": 0, "next_step": 1}
    if status is not None:
        payload["status"] = status

    state = _resume_state_from_payload(tmp_path / "resume.pt", payload)

    assert state == _ResumeState(start_step=1)


@pytest.mark.parametrize(
    ("payload_update", "message"),
    [
        ({"step": True}, r"invalid step"),
        ({"step": -1}, r"invalid step"),
        ({"step": 1.5}, r"invalid step"),
        ({"step": math.nan}, r"invalid step"),
        ({"step": math.inf}, r"invalid step"),
        ({"next_step": True}, r"invalid next_step"),
        ({"next_step": -1}, r"invalid next_step"),
        ({"next_step": math.nan}, r"invalid next_step"),
        ({"status": "not-a-dict"}, r"invalid status metadata"),
        (
            {
                "status": {
                    "best_nll_bits": True,
                    "stable_loss_steps": 0,
                },
            },
            r"invalid status\.best_nll_bits",
        ),
        (
            {
                "status": {
                    "best_nll_bits": math.nan,
                    "stable_loss_steps": 0,
                },
            },
            r"invalid status\.best_nll_bits",
        ),
        (
            {
                "status": {
                    "best_nll_bits": "not-a-number",
                    "stable_loss_steps": 0,
                },
            },
            r"invalid status\.best_nll_bits",
        ),
        (
            {
                "status": {
                    "previous_objective": math.inf,
                    "stable_loss_steps": 0,
                },
            },
            r"invalid status\.previous_objective",
        ),
        (
            {
                "status": {
                    "previous_objective": "not-a-number",
                    "stable_loss_steps": 0,
                },
            },
            r"invalid status\.previous_objective",
        ),
        (
            {
                "status": {
                    "best_step": True,
                    "stable_loss_steps": 0,
                },
            },
            r"invalid status\.best_step",
        ),
        (
            {
                "status": {
                    "best_step": 1.5,
                    "stable_loss_steps": 0,
                },
            },
            r"invalid status\.best_step",
        ),
        (
            {
                "status": {
                    "stable_loss_steps": -1,
                },
            },
            r"invalid status\.stable_loss_steps",
        ),
        (
            {
                "status": {
                    "stable_loss_steps": math.inf,
                },
            },
            r"invalid status\.stable_loss_steps",
        ),
    ],
)
def test_resume_state_from_payload_rejects_invalid_metadata(
    tmp_path: Path,
    payload_update: dict[str, object],
    message: str,
):
    payload = {
        "step": 0,
        "next_step": 1,
        "status": {
            "previous_objective": 1.5,
            "stable_loss_steps": 0,
        },
    }
    payload.update(payload_update)

    with pytest.raises(RuntimeError, match=message):
        _resume_state_from_payload(tmp_path / "resume.pt", payload)
