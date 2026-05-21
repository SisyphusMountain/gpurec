from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import gpurec.api.uniform_chunked as uniform_chunked_module
from gpurec import UniformChunkedReconModel
from gpurec.workflow.config import RunConfig
from gpurec.workflow.optimize import (
    _ResumeState,
    _resume_state_from_payload,
    _step_stopping_status,
)


def test_uniform_chunked_e_adjoint_stats_fields_are_public_stats_shape():
    stats = uniform_chunked_module._e_adjoint_stats_fields(
        SimpleNamespace(
            method="BiCGSTAB",
            iters=7,
            rel_res=0.125,
            success=False,
        )
    )

    assert stats == {
        "e_adjoint_method": "BiCGSTAB",
        "e_adjoint_iterations": 7,
        "e_adjoint_rel_res": 0.125,
        "e_adjoint_success": False,
    }


def test_uniform_chunked_read_only_helper_delegates_to_result_core(monkeypatch):
    state = object()
    theta = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    chunk_indices = [1]
    expected_loss = torch.tensor(9.0, dtype=torch.float64)
    expected_per_family = torch.tensor([4.0, 5.0], dtype=torch.float64)
    expected_stats = {"selected_families": 2}
    calls: list[dict[str, object]] = []

    def fake_evaluate_chunked_uniform_result(
        state_arg,
        theta_arg,
        *,
        need_grad,
        collect_per_family=False,
        chunk_indices=None,
    ):
        calls.append(
            {
                "state": state_arg,
                "theta": theta_arg,
                "need_grad": need_grad,
                "collect_per_family": collect_per_family,
                "chunk_indices": chunk_indices,
                "grad_enabled": torch.is_grad_enabled(),
            }
        )
        return uniform_chunked_module._UniformChunkedEvaluation(
            loss=expected_loss,
            grad_theta=None,
            stats=expected_stats,
            per_family_nll=expected_per_family,
        )

    monkeypatch.setattr(
        uniform_chunked_module,
        "_evaluate_chunked_uniform_result",
        fake_evaluate_chunked_uniform_result,
    )

    result = uniform_chunked_module._evaluate_chunked_uniform_read_only(
        state,
        theta,
        collect_per_family=True,
        chunk_indices=chunk_indices,
    )

    assert result.loss is expected_loss
    assert result.per_family_nll is expected_per_family
    assert result.stats is expected_stats
    assert calls == [
        {
            "state": state,
            "theta": theta,
            "need_grad": False,
            "collect_per_family": True,
            "chunk_indices": chunk_indices,
            "grad_enabled": False,
        }
    ]


def test_uniform_chunked_gradient_result_rejects_bf16_before_chunk_work():
    state = SimpleNamespace(dtype=torch.bfloat16)
    theta = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)

    with pytest.raises(
        RuntimeError,
        match=r"gradient evaluation requires float32 or float64.*Pi_wave_backward",
    ):
        uniform_chunked_module._evaluate_chunked_uniform_result(
            state,
            theta,
            need_grad=True,
        )


def test_uniform_chunked_read_only_result_allows_bf16_boundary(monkeypatch):
    state = SimpleNamespace(dtype=torch.bfloat16)
    theta = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    selected = object()
    calls: list[dict[str, object]] = []

    def fake_selected_chunks(state_arg, chunk_indices):
        calls.append(
            {
                "state": state_arg,
                "chunk_indices": chunk_indices,
            }
        )
        raise RuntimeError("sentinel after dtype boundary")

    monkeypatch.setattr(
        uniform_chunked_module,
        "_selected_chunks",
        fake_selected_chunks,
    )

    with pytest.raises(RuntimeError, match="sentinel after dtype boundary"):
        uniform_chunked_module._evaluate_chunked_uniform_result(
            state,
            theta,
            need_grad=False,
            chunk_indices=selected,
        )

    assert calls == [{"state": state, "chunk_indices": selected}]


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


def test_uniform_chunked_full_sum_estimate_scales_loss_and_grad(monkeypatch):
    model = UniformChunkedReconModel.__new__(UniformChunkedReconModel)
    torch.nn.Module.__init__(model)
    model.theta = torch.nn.Parameter(
        torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    )
    model._state = object()
    calls: list[dict[str, object]] = []

    def fake_evaluate_chunked_uniform(
        state,
        theta,
        *,
        need_grad,
        chunk_indices=None,
        **kwargs,
    ):
        calls.append(
            {
                "state": state,
                "theta": theta,
                "need_grad": need_grad,
                "chunk_indices": chunk_indices,
                "kwargs": kwargs,
            }
        )
        return (
            torch.tensor(10.0, dtype=torch.float64),
            torch.tensor([1.0, -2.0, 3.0], dtype=torch.float64),
            {
                "selected_families": 2,
                "total_families": 8,
                "selected_chunks": [1],
                "e_adjoint_method": "BiCGSTAB",
                "e_adjoint_iterations": 5,
                "e_adjoint_rel_res": 0.25,
                "e_adjoint_success": False,
            },
        )

    monkeypatch.setattr(
        uniform_chunked_module,
        "_evaluate_chunked_uniform",
        fake_evaluate_chunked_uniform,
    )

    loss, grad, stats = model.loss_and_grad(
        chunk_indices=[1],
        reduction="full_sum_estimate",
    )

    assert calls == [
        {
            "state": model._state,
            "theta": model.theta,
            "need_grad": True,
            "chunk_indices": [1],
            "kwargs": {},
        }
    ]
    torch.testing.assert_close(loss, torch.tensor(40.0, dtype=torch.float64))
    torch.testing.assert_close(
        grad,
        torch.tensor([4.0, -8.0, 12.0], dtype=torch.float64),
    )
    assert stats["reduction"] == "full_sum_estimate"
    assert stats["scale"] == 4.0
    assert stats["reduced_loss"] == 40.0
    assert stats["reduced_grad_norm"] == pytest.approx(math.sqrt(224.0))
    assert stats["e_adjoint_method"] == "BiCGSTAB"
    assert stats["e_adjoint_iterations"] == 5
    assert stats["e_adjoint_rel_res"] == pytest.approx(0.25)
    assert stats["e_adjoint_success"] is False


def test_uniform_chunked_nll_uses_read_only_chunked_result(monkeypatch):
    model = UniformChunkedReconModel.__new__(UniformChunkedReconModel)
    torch.nn.Module.__init__(model)
    model.theta = torch.nn.Parameter(
        torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    )
    model._state = object()
    chunk_indices = [1]
    expected = torch.tensor(9.0, dtype=torch.float64)
    calls: list[dict[str, object]] = []

    def fake_evaluate_chunked_uniform_read_only(
        state,
        theta,
        *,
        collect_per_family=False,
        chunk_indices=None,
    ):
        calls.append(
            {
                "state": state,
                "theta": theta,
                "collect_per_family": collect_per_family,
                "chunk_indices": chunk_indices,
                "grad_enabled": torch.is_grad_enabled(),
            }
        )
        return uniform_chunked_module._UniformChunkedReadOnlyEvaluation(
            loss=expected,
            stats={"selected_families": 2},
        )

    monkeypatch.setattr(
        uniform_chunked_module,
        "_evaluate_chunked_uniform_read_only",
        fake_evaluate_chunked_uniform_read_only,
    )

    actual = model.nll(chunk_indices=chunk_indices)

    assert actual is expected
    assert calls == [
        {
            "state": model._state,
            "theta": model.theta,
            "collect_per_family": False,
            "chunk_indices": chunk_indices,
            "grad_enabled": False,
        }
    ]


def test_uniform_chunked_nll_per_family_uses_no_grad_chunked_diagnostic(
    monkeypatch,
):
    model = UniformChunkedReconModel.__new__(UniformChunkedReconModel)
    torch.nn.Module.__init__(model)
    model.theta = torch.nn.Parameter(
        torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    )
    model._state = object()
    chunk_indices = [1]
    expected = torch.tensor([4.0, 5.0], dtype=torch.float64)
    calls: list[dict[str, object]] = []

    def fake_evaluate_chunked_uniform_read_only(
        state,
        theta,
        *,
        collect_per_family=False,
        chunk_indices=None,
    ):
        calls.append(
            {
                "state": state,
                "theta": theta,
                "collect_per_family": collect_per_family,
                "chunk_indices": chunk_indices,
                "grad_enabled": torch.is_grad_enabled(),
            }
        )
        return uniform_chunked_module._UniformChunkedReadOnlyEvaluation(
            loss=torch.tensor(9.0, dtype=torch.float64),
            stats={"selected_families": 2},
            per_family_nll=expected,
        )

    monkeypatch.setattr(
        uniform_chunked_module,
        "_evaluate_chunked_uniform_read_only",
        fake_evaluate_chunked_uniform_read_only,
    )

    actual = model.nll_per_family(chunk_indices=chunk_indices)

    assert actual is expected
    assert calls == [
        {
            "state": model._state,
            "theta": model.theta,
            "collect_per_family": True,
            "chunk_indices": chunk_indices,
            "grad_enabled": False,
        }
    ]


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
