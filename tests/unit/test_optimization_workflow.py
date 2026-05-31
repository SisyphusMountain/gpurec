from __future__ import annotations

import ast
import importlib
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import gpurec.api._uniform_chunked_eval as uniform_chunked_eval_module
import gpurec.api.uniform_chunked as uniform_chunked_module
import gpurec.workflow._adaptive_rebatch as adaptive_rebatch_module
import gpurec.workflow._evaluation as evaluation_module
import gpurec.workflow._step_execution as step_execution_module
import gpurec.workflow._step_runtime as step_runtime_module
import gpurec.workflow._stopping_policy as stopping_policy_module
import gpurec.workflow._transitions as transitions_module
from gpurec import UniformChunkedReconModel
from gpurec.workflow._batch_final_cache import BatchFinalCache
from gpurec.workflow.config import RunConfig
from gpurec.workflow._run_state import (
    BatchRunState,
    LBFGSBRunState,
    ObjectiveState,
    RestartRunState,
    _OptimizationRunState,
)
from gpurec.workflow._runtime_state import (
    _ResumeState,
    _apply_resume_checkpoint_state,
    _resume_state_from_payload,
)
from gpurec.workflow._stopping_policy import _active_batch_patience
from gpurec.workflow._step_plan import _StepPlanningContext, _StepPlanningState
from gpurec.workflow._transitions import (
    IterationTransition,
    IterationTransitionContext,
    IterationTransitionInputs,
    IterationTransitionOps,
    apply_iteration_transition,
    build_batch_transition_checkpoint_status,
    execute_iteration_transition,
)
from gpurec.workflow.optimize import (
    OptimizationRunner,
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


def test_uniform_chunked_chunk_stats_row_has_public_stats_shape():
    built = uniform_chunked_module._UniformBuiltChunk(
        spec=uniform_chunked_module._UniformChunkSpec(
            indices=[3, 4, 5],
            clades=17,
            splits=19,
        ),
        wave_layout={},
        waves=7,
        max_wave=11,
        split_rows=23,
        max_wave_split_rows=13,
    )

    row = uniform_chunked_module._chunk_stats_row(
        chunk_idx=2,
        built=built,
        forward_ms=1.25,
        pi_backward_ms=2.5,
    )

    assert row == {
        "idx": 2,
        "family_start": 3,
        "family_stop": 6,
        "families": 3,
        "clades": 17,
        "splits": 19,
        "waves": 7,
        "max_wave": 11,
        "split_rows": 23,
        "max_wave_split_rows": 13,
        "forward_ms": 1.25,
        "pi_backward_ms": 2.5,
    }


def test_uniform_chunked_built_chunks_from_rust_normalizes_payload_tensors():
    payloads = [
        {
            "indices": [1, "2"],
            "clades": "5",
            "splits": 7,
            "wave_layout": {
                "float_rows": torch.tensor([1.0, 2.0], dtype=torch.float32),
                "int_rows": [torch.tensor([3, 4], dtype=torch.int64)],
            },
            "waves": 3,
            "max_wave": "4",
            "split_rows": 11,
            "max_wave_split_rows": "6",
        }
    ]

    (built,) = uniform_chunked_module._built_chunks_from_rust(
        payloads,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )

    assert built.spec == uniform_chunked_module._UniformChunkSpec(
        indices=[1, 2],
        clades=5,
        splits=7,
    )
    assert built.waves == 3
    assert built.max_wave == 4
    assert built.split_rows == 11
    assert built.max_wave_split_rows == 6
    assert built.wave_layout["float_rows"].dtype is torch.float64
    assert built.wave_layout["float_rows"].device.type == "cpu"
    assert built.wave_layout["int_rows"][0].dtype is torch.int64
    assert built.wave_layout["int_rows"][0].device.type == "cpu"


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
        return uniform_chunked_eval_module._UniformChunkedEvaluation(
            loss=expected_loss,
            grad_theta=None,
            stats=expected_stats,
            per_family_nll=expected_per_family,
        )

    monkeypatch.setattr(
        uniform_chunked_eval_module,
        "_evaluate_chunked_uniform_result",
        fake_evaluate_chunked_uniform_result,
    )

    result = uniform_chunked_eval_module._evaluate_chunked_uniform_read_only(
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
        uniform_chunked_eval_module,
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


def _pi_backward_contribution(
    value: float,
    *,
    shape: tuple[int, ...] = (3,),
    dtype: torch.dtype = torch.float32,
) -> dict[str, object]:
    contribution: dict[str, object] = {
        key: torch.full(shape, value, dtype=dtype)
        for key in uniform_chunked_module._PI_BACKWARD_TENSOR_KEYS
    }
    contribution.update(
        {
            "n_waves_total": 3,
            "n_waves_skipped": 1,
            "n_waves_processed": 2,
            "n_clades_total": 5,
            "n_clades_skipped": 2,
            "n_clades_active": 3,
        }
    )
    return contribution


def test_uniform_chunked_pi_backward_accumulator_has_explicit_schema() -> None:
    accumulator = uniform_chunked_module._new_pi_backward_accumulator()
    accumulator.add(_pi_backward_contribution(1.0))
    accumulator.add(_pi_backward_contribution(2.0))

    result = accumulator.result()

    assert set(result) == set(uniform_chunked_module._PI_BACKWARD_TENSOR_KEYS) | set(
        uniform_chunked_module._PI_BACKWARD_COUNTER_KEYS
    )
    for key in uniform_chunked_module._PI_BACKWARD_TENSOR_KEYS:
        torch.testing.assert_close(
            result[key],
            torch.full((3,), 3.0, dtype=torch.float32),
        )
    assert result["n_waves_total"] == 6
    assert result["n_waves_skipped"] == 2
    assert result["n_waves_processed"] == 4
    assert result["n_clades_total"] == 10
    assert result["n_clades_skipped"] == 4
    assert result["n_clades_active"] == 6


def test_uniform_chunked_pi_backward_accumulator_rejects_shape_drift() -> None:
    accumulator = uniform_chunked_module._new_pi_backward_accumulator()
    accumulator.add(_pi_backward_contribution(1.0))

    with pytest.raises(ValueError, match=r"field 'grad_E' shape"):
        accumulator.add(_pi_backward_contribution(2.0, shape=(2, 3)))


def _run_config(tmp_path: Path, **overrides: object) -> RunConfig:
    values = {
        "species_tree": tmp_path / "sp.nwk",
        "families_file": tmp_path / "families.txt",
        "out_dir": tmp_path / "out",
        "device": "cuda",
        "loss_patience": 0,
        "best_likelihood_patience": 0,
    }
    values.update(overrides)
    return RunConfig(**values)


def test_specieswise_solver_warmup_starts_below_full_pi_budget(tmp_path: Path) -> None:
    config = _run_config(
        tmp_path,
        mode="specieswise",
        optimizer="projected-lbfgs",
        fixed_iters_e=None,
        fixed_iters_pi=8,
        neumann_terms=8,
        solver_warmup_iters=4,
    )
    runner = OptimizationRunner(config)
    calls: list[dict[str, object]] = []
    model = SimpleNamespace(
        configure_solver_iterations=lambda **kwargs: calls.append(kwargs),
        current_batch_metadata=SimpleNamespace(clade_count=0),
    )

    assert runner._uses_solver_warmup()

    runner._configure_active_solver_stage(model, "warmup")
    runner._configure_active_solver_stage(model, "full")

    assert calls == [
        {
            "fixed_iters_E": 4,
            "fixed_iters_Pi": 4,
            "neumann_terms": 4,
        },
        {
            "fixed_iters_E": None,
            "fixed_iters_Pi": 8,
            "neumann_terms": 8,
        },
    ]


def test_specieswise_solver_warmup_is_skipped_when_not_lower_budget(
    tmp_path: Path,
) -> None:
    config = _run_config(
        tmp_path,
        mode="specieswise",
        optimizer="projected-lbfgs",
        fixed_iters_pi=4,
        solver_warmup_iters=4,
    )

    assert not OptimizationRunner(config)._uses_solver_warmup()


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
    ("overrides", "stable_loss_steps", "best_step", "step", "expected"),
    [
        (
            {"loss_patience": 1, "best_likelihood_patience": 1},
            1,
            0,
            1,
            {"status": "converged", "reason": "loss_change_patience"},
        ),
        (
            {"loss_patience": 2, "best_likelihood_patience": 1},
            2,
            0,
            2,
            {"status": "converged", "reason": "loss_change_patience"},
        ),
        (
            {"best_likelihood_patience": 3},
            0,
            2,
            5,
            {"status": "converged", "reason": "best_likelihood_patience"},
        ),
        (
            {"loss_patience": 2, "best_likelihood_patience": 3},
            1,
            4,
            5,
            None,
        ),
        (
            {"loss_patience": 0, "best_likelihood_patience": 0},
            0,
            None,
            0,
            None,
        ),
    ],
)
def test_step_stopping_status_matches_optimizer_loop_order(
    tmp_path: Path,
    overrides: dict[str, object],
    stable_loss_steps: int,
    best_step: int | None,
    step: int,
    expected: dict[str, str] | None,
):
    status = _step_stopping_status(
        _run_config(tmp_path, **overrides),
        step=step,
        stable_loss_steps=stable_loss_steps,
        best_step=best_step,
    )

    assert status == expected


@pytest.mark.parametrize(
    ("configured_patience", "expected"),
    [(-1, -1), (0, 0), (1, 1), (4, 3)],
)
def test_active_batch_patience_caps_positive_values(
    configured_patience: int,
    expected: int,
):
    assert _active_batch_patience(configured_patience) == expected


def test_step_stopping_status_allows_explicit_patience_overrides(tmp_path: Path):
    config = _run_config(
        tmp_path,
        loss_patience=10,
        best_likelihood_patience=10,
    )

    assert _step_stopping_status(
        config,
        step=2,
        stable_loss_steps=1,
        best_step=0,
        loss_patience=1,
        best_likelihood_patience=1,
    ) == {"status": "converged", "reason": "loss_change_patience"}
    assert _step_stopping_status(
        config,
        step=2,
        stable_loss_steps=0,
        best_step=0,
        loss_patience=0,
        best_likelihood_patience=1,
    ) == {"status": "converged", "reason": "best_likelihood_patience"}
    assert _step_stopping_status(
        config,
        step=20,
        stable_loss_steps=20,
        best_step=0,
        loss_patience=0,
        best_likelihood_patience=0,
    ) is None


def test_step_stopping_status_requires_best_step_for_best_likelihood_stop(
    tmp_path: Path,
):
    status = _step_stopping_status(
        _run_config(tmp_path, loss_patience=0, best_likelihood_patience=1),
        step=10,
        stable_loss_steps=0,
        best_step=None,
    )

    assert status is None


def test_optimize_reexports_stopping_policy_helpers():
    import importlib

    optimize_module = importlib.import_module("gpurec.workflow.optimize")

    assert optimize_module._step_stopping_status is (
        stopping_policy_module._step_stopping_status
    )
    assert optimize_module._active_batch_patience is (
        stopping_policy_module._active_batch_patience
    )


def test_step_execution_reexports_step_runtime_helpers():
    for name in (
        "_NonfiniteParameterUpdate",
        "_set_model_theta",
        "_restore_theta_if_nonfinite_update",
        "_active_adam_step",
        "_clear_solver_runtime_state_preserving_pi_cache",
    ):
        assert getattr(step_execution_module, name) is getattr(
            step_runtime_module,
            name,
        )


def test_optimize_reexports_runtime_restore_and_evaluation_clear_helpers():
    import importlib

    optimize_module = importlib.import_module("gpurec.workflow.optimize")

    assert optimize_module._restore_theta_if_nonfinite_update is (
        step_execution_module._restore_theta_if_nonfinite_update
    )
    assert optimize_module._clear_solver_runtime_state_preserving_pi_cache is (
        evaluation_module._clear_solver_runtime_state_preserving_pi_cache
    )
    assert optimize_module._clear_solver_runtime_state_preserving_pi_cache is not (
        step_execution_module._clear_solver_runtime_state_preserving_pi_cache
    )


def test_restore_theta_if_nonfinite_update_restores_theta_and_clears_grad():
    theta_before = torch.tensor([0.25, -0.5], dtype=torch.float32)
    model = SimpleNamespace(
        theta=torch.nn.Parameter(
            torch.tensor([float("nan"), 1.0], dtype=torch.float32),
        )
    )
    model.theta.grad = torch.ones_like(model.theta)

    restored = step_execution_module._restore_theta_if_nonfinite_update(
        model,
        theta_before,
    )

    assert restored is True
    torch.testing.assert_close(model.theta.detach(), theta_before)
    assert model.theta.grad is None


def test_active_adam_step_commits_and_counts_staged_pi_adjoint_cache():
    pending_values: list[torch.Tensor] = []
    static = SimpleNamespace(
        pi_adjoint_warmstart=True,
        pi_adjoint_cache_update_mode="stage",
        pi_adjoint_cache=torch.tensor([-1.0], dtype=torch.float32),
        pi_adjoint_pending_cache=torch.tensor([-2.0], dtype=torch.float32),
    )

    class FakeModel:
        def __init__(self) -> None:
            self.theta = torch.nn.Parameter(
                torch.tensor(
                    [
                        [0.50, -0.25, 0.125],
                        [0.10, 0.20, -0.05],
                    ],
                    dtype=torch.float32,
                )
            )
            self.cached_static_states = [static]

        def clamp_theta_(self, min_rate: float, max_rate: float) -> None:
            with torch.no_grad():
                self.theta.clamp_(min=math.log2(min_rate), max=math.log2(max_rate))

    class FakeEvaluation:
        def evaluate_active_genewise_vector_grad_at_current_theta(
            self,
            model: FakeModel,
            *,
            solver_stage: str,
        ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float | str]]:
            grad = 2.0 * model.theta.detach()
            model.theta.grad = grad.clone()
            loss_vec = model.theta.detach().square().sum(dim=1)
            pending = torch.tensor(
                [float(len(pending_values) + 1)],
                dtype=model.theta.dtype,
            )
            pending_values.append(pending.detach().cpu())
            static.pi_adjoint_pending_cache = pending
            return loss_vec, grad, {"optimizer/solver_stage": solver_stage}

        def projected_grad_inf(
            self,
            model: FakeModel,
            *,
            lower_bound: float,
            upper_bound: float,
        ) -> tuple[torch.Tensor, float]:
            assert model.theta.grad is not None
            return model.theta.grad, float(model.theta.grad.abs().amax().cpu())

    model = FakeModel()
    optimizer = torch.optim.Adam([model.theta], lr=0.1)

    loss_vec, metrics, evals = step_execution_module._active_adam_step(
        model,
        optimizer,
        evaluation=FakeEvaluation(),
        config=SimpleNamespace(min_rate=0.01, max_rate=100.0),
        solver_stage="warmup",
        theta_before=model.theta.detach().clone(),
    )

    assert evals == 2
    assert loss_vec.shape == (2,)
    assert metrics["optimizer/solver_stage"] == "warmup"
    assert metrics["solver/pi_adjoint_pending_cache_discards"] == 1.0
    assert metrics["solver/pi_adjoint_pending_cache_commits"] == 1.0
    torch.testing.assert_close(static.pi_adjoint_cache.cpu(), pending_values[-1])
    assert static.pi_adjoint_pending_cache is None


def test_execute_optimization_step_uses_step_runtime_clear_behavior():
    static = SimpleNamespace(
        warm_E=object(),
        pi_adjoint_warmstart=True,
        pi_adjoint_cache_update_mode="stage",
        pi_adjoint_cache=torch.tensor([1.0], dtype=torch.float32),
        pi_adjoint_pending_cache=torch.tensor([2.0], dtype=torch.float32),
        last_solver_stats={"Pi_wave_iterations": [3]},
    )

    class FakeModel:
        def __init__(self) -> None:
            self.theta = torch.nn.Parameter(
                torch.tensor([[0.5, -0.25]], dtype=torch.float32),
            )
            self.cached_static_states = [static]
            self.current_batch_metadata = SimpleNamespace(clade_count=0)
            self.clears = 0

        def clear(self) -> None:
            self.clears += 1

    class FakeSolver:
        def is_hessian_sgd_validation_step(self, **kwargs) -> bool:
            return False

    model = FakeModel()

    def active_fd_newton_step(model_arg: FakeModel, **kwargs):
        model_arg.theta.grad = torch.ones_like(model_arg.theta)
        return torch.ones(1), {}, 1, None

    context = step_execution_module._StepExecutionContext(
        config=SimpleNamespace(
            fd_hessian_refresh_steps=1,
            hessian_sgd_validation_interval=0,
            lr=0.1,
            max_rate=100.0,
            min_rate=0.01,
        ),
        evaluation=SimpleNamespace(),
        solver=FakeSolver(),
        batchwise_active_optimizer=False,
        fd_adam_warmup_steps=0,
        hessian_sgd_no_line_refresh_min_clades=10,
        hessian_sgd_no_line_refresh_steps=5,
        hessian_sgd_line_search_max_steps=2,
        active_fd_newton_step=active_fd_newton_step,
    )

    result = step_execution_module.execute_optimization_step(
        context,
        step_execution_module._StepExecutionState(active_solver_stage="full"),
        model,
        torch.optim.SGD([model.theta], lr=0.1),
        phase="hessian-sgd",
        step=0,
        adagrad_restart_active_phase=None,
    )

    assert result.status is None
    assert static.warm_E is None
    assert static.pi_adjoint_cache is None
    assert static.pi_adjoint_pending_cache is None
    assert static.last_solver_stats is None
    assert model.clears == 0


def test_adaptive_rebatch_uses_shared_active_batch_patience():
    assert adaptive_rebatch_module._active_batch_patience is (
        stopping_policy_module._active_batch_patience
    )


def test_optimize_reexports_workflow_run_state_classes():
    import importlib

    from gpurec.workflow import _run_state

    optimize_module = importlib.import_module("gpurec.workflow.optimize")

    for name in (
        "ObjectiveState",
        "BatchRunState",
        "RestartRunState",
        "LBFGSBRunState",
        "_OptimizationRunState",
    ):
        assert getattr(optimize_module, name) is getattr(_run_state, name)


def test_transitions_reexports_workflow_transition_type_classes():
    import importlib
    from dataclasses import is_dataclass

    transition_types = importlib.import_module("gpurec.workflow._transition_types")
    transitions = importlib.import_module("gpurec.workflow._transitions")

    for name in (
        "IterationTransition",
        "IterationTransitionExecution",
        "IterationTransitionOps",
        "IterationStatusTransitionExecution",
        "IterationTransitionContext",
        "IterationTransitionInputs",
    ):
        transition_type = getattr(transition_types, name)
        assert getattr(transitions, name) is transition_type
        assert is_dataclass(transition_type)

    assert transition_types.IterationTransitionOps.__dataclass_params__.frozen


def test_transitions_reexports_workflow_transition_classifier():
    import importlib

    transition_policy = importlib.import_module("gpurec.workflow._transition_policy")
    transitions = importlib.import_module("gpurec.workflow._transitions")

    assert (
        transitions._classify_iteration_transition
        is transition_policy._classify_iteration_transition
    )


def test_transitions_reexports_workflow_transition_execution_helpers():
    import importlib

    transitions = importlib.import_module("gpurec.workflow._transitions")

    assert transitions.build_batch_transition_checkpoint_status is (
        build_batch_transition_checkpoint_status
    )
    assert transitions.execute_iteration_transition is execute_iteration_transition

    status = build_batch_transition_checkpoint_status(
        {"status": "running", "reason": "running"},
        SimpleNamespace(active_index=2, solver_stage="warmup", local_step=3),
        previous_objective=12.5,
        stable_loss_steps=4,
        best_nll_bits=9.5,
        best_step=8,
    )
    assert status == {
        "status": "running",
        "reason": "running",
        "active_batch_index": 2,
        "active_solver_stage": "warmup",
        "active_batch_local_step": 3,
        "previous_objective": 12.5,
        "stable_loss_steps": 4,
        "best_nll_bits": 9.5,
        "best_step": 8,
    }


def _transition_test_ops(
    save_calls: list[dict[str, object]],
    *,
    progress_calls: list[dict[str, object]] | None = None,
    clear_static_calls: list[object] | None = None,
) -> IterationTransitionOps:
    def save_status(path, **kwargs):
        save_calls.append({"path": path, **kwargs})

    def clear_cached_static_states_if_needed(model):
        if clear_static_calls is not None:
            clear_static_calls.append(model)

    def print_progress_row(**kwargs):
        if progress_calls is not None:
            progress_calls.append(kwargs)

    return IterationTransitionOps(
        active_batch_indices=lambda model: torch.arange(1),
        clear_cached_static_states_if_needed=clear_cached_static_states_if_needed,
        clear_cached_solver_runtime_state=lambda model: setattr(
            model,
            "cleared_runtime",
            getattr(model, "cleared_runtime", 0) + 1,
        ),
        load_checkpoint=lambda path: {},
        validate_checkpoint_model_compatibility=lambda **kwargs: None,
        restore_model_theta=lambda model, payload: None,
        make_optimizer=lambda config, model, phase: torch.optim.SGD(
            [torch.nn.Parameter(torch.zeros(1))],
            lr=0.1,
        ),
        restore_optimizer_state=lambda *args, **kwargs: {},
        resume_state_from_payload=lambda path, payload: SimpleNamespace(),
        save_status=save_status,
        adaptive_checkpoint_status=lambda status: {"wrapped": status},
        print_progress_row=print_progress_row,
        fd_adam_warmup_steps=2,
    )


def _transition_test_planning_state() -> _StepPlanningState:
    return _StepPlanningState(
        restart_dynamic_phase_index=0,
        restart_dynamic_phase_start_step=0,
        current_phase="adam-fd-newton",
        active_batch_index=0,
        active_optimizer_batch_index=0,
        active_adagrad_restart_phase_index=None,
        previous_objective=99.0,
        stable_loss_steps=7,
        lbfgsb_fallback_used_count=3,
    )


def _transition_test_inputs(
    *,
    step: int,
    phase: str,
    step_status: dict[str, str],
    active_batch_count: int,
) -> IterationTransitionInputs:
    return IterationTransitionInputs(
        status={"status": "running", "reason": "running"},
        step=step,
        phase=phase,
        row={"row": step},
        checkpoint_status={"status": "running", "reason": "running"},
        step_status=step_status,
        objective=1.0,
        row_best_nll=None,
        row_best_step=None,
        active_objective_scope=True,
        active_batch_count=active_batch_count,
        can_lbfgsb_retry=False,
        lbfgsb_high_kkt_status=None,
        hessian_sgd_activate_line_search=False,
        projected_lbfgs_min_lr_reached=False,
        adaptive_rebatch_stop=False,
        rejected_nonfinite_parameter_update=False,
        adaptive_rebatch_pending_indices=None,
        adagrad_restart_terminal_status=None,
        adagrad_restart_phase_next_index=None,
        adagrad_restart_phase_next_start_step=None,
        lbfgsb_loss_schedule_next_index=None,
    )


def _import_transition_post_step_module():
    module_name = "gpurec.workflow._transition_post_step"
    if importlib.util.find_spec(module_name) is None:
        pytest.skip(f"{module_name} has not been extracted yet")
    return importlib.import_module(module_name)


def test_post_step_transition_exports_are_identity_reexports():
    post_step_module = _import_transition_post_step_module()

    assert transitions_module.execute_step_status_transition is (
        post_step_module.execute_step_status_transition
    )
    assert transitions_module.execute_iteration_post_step_transition is (
        post_step_module.execute_iteration_post_step_transition
    )


def test_transition_post_step_does_not_import_transitions_module():
    post_step_module = _import_transition_post_step_module()
    module_file = getattr(post_step_module, "__file__", None)
    assert module_file is not None
    tree = ast.parse(Path(module_file).read_text())

    forbidden_imports: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            forbidden_imports.extend(
                (node.lineno, alias.name)
                for alias in node.names
                if alias.name == "gpurec.workflow._transitions"
            )
        if isinstance(node, ast.ImportFrom):
            imports_transitions_submodule = (
                node.module == "gpurec.workflow._transitions"
                or (node.module == "gpurec.workflow")
                and any(alias.name == "_transitions" for alias in node.names)
                or node.level > 0
                and (
                    node.module == "_transitions"
                    or node.module is None
                    and any(alias.name == "_transitions" for alias in node.names)
                )
            )
            if imports_transitions_submodule:
                module = "." * node.level + (node.module or "")
                forbidden_imports.append((node.lineno, module))

    assert forbidden_imports == []


def _execute_transition_for_test(
    tmp_path: Path,
    *,
    transition: IterationTransition,
    status: dict[str, str] | None = None,
    model: object | None = None,
    objective_state: ObjectiveState | None = None,
    batch_state: BatchRunState | None = None,
    restart_state: RestartRunState | None = None,
    lbfgsb_state: LBFGSBRunState | None = None,
    adaptive_state: object | None = None,
    planning_state: _StepPlanningState | None = None,
    optimizer: object | None = None,
    fd_newton_hessian_state: object | None = None,
    hessian_sgd_line_search_active: bool = True,
    hessian_sgd_low_accept_steps: int = 3,
    resume_info: dict[str, object] | None = None,
    checkpoint_every: int | None = 1,
    log_every: int = 1,
    batch_final_cache: BatchFinalCache | None = None,
):
    save_calls: list[dict[str, object]] = []
    progress_calls: list[dict[str, object]] = []
    clear_static_calls: list[object] = []
    solver_calls: list[tuple[object, str]] = []

    if model is None:
        model = SimpleNamespace(replanned_indices=[])

        def replan_resident_batches(indices):
            model.replanned_indices.append(tuple(indices))

        model.replan_resident_batches = replan_resident_batches
    if objective_state is None:
        objective_state = ObjectiveState(previous_objective=10.0, stable_loss_steps=4)
    if batch_state is None:
        batch_state = BatchRunState(
            active_index=1,
            local_step=5,
            solver_stage="warmup",
            best_nll=3.0,
            best_step=8,
            optimizer_batch_index=1,
        )
    if restart_state is None:
        restart_state = RestartRunState(
            dynamic_enabled=True,
            phase_index=0,
            phase_start_step=0,
            active_phase_index=1,
        )
    if lbfgsb_state is None:
        lbfgsb_state = LBFGSBRunState(fallback_used_count=5)
    if adaptive_state is None:
        adaptive_state = SimpleNamespace(
            batch_plan_generation=2,
            last_checked_converged_count=7,
        )
    if planning_state is None:
        planning_state = _transition_test_planning_state()
    if optimizer is None:
        optimizer = object()
    if fd_newton_hessian_state is None:
        fd_newton_hessian_state = object()
    if resume_info is None:
        resume_info = {"resume": "kept"}

    result = execute_iteration_transition(
        transition=transition,
        status=status or {"status": "running", "reason": "running"},
        model=model,
        objective_state=objective_state,
        batch_state=batch_state,
        restart_state=restart_state,
        lbfgsb_state=lbfgsb_state,
        adaptive_state=adaptive_state,
        planning_state=planning_state,
        optimizer=optimizer,
        fd_newton_hessian_state=fd_newton_hessian_state,
        hessian_sgd_line_search_active=hessian_sgd_line_search_active,
        hessian_sgd_low_accept_steps=hessian_sgd_low_accept_steps,
        resume_info=resume_info,
        step=8,
        phase="adagrad-restarts:phase0",
        objective=1.25,
        row_best_nll=0.75,
        row={"delta_likelihood_bits": 0.5},
        checkpoint_status={"status": "running", "reason": "running", "base": "kept"},
        solver=SimpleNamespace(
            configure_active_stage=lambda model_arg, stage: solver_calls.append(
                (model_arg, stage)
            )
        ),
        batch_final_cache=batch_final_cache,
        latest_checkpoint=tmp_path / "latest.pt",
        log_every=log_every,
        checkpoint_every=checkpoint_every,
        ops=_transition_test_ops(
            save_calls,
            progress_calls=progress_calls,
            clear_static_calls=clear_static_calls,
        ),
    )

    return SimpleNamespace(
        result=result,
        save_calls=save_calls,
        progress_calls=progress_calls,
        clear_static_calls=clear_static_calls,
        solver_calls=solver_calls,
        model=model,
        objective_state=objective_state,
        batch_state=batch_state,
        restart_state=restart_state,
        lbfgsb_state=lbfgsb_state,
        adaptive_state=adaptive_state,
        planning_state=planning_state,
        optimizer=optimizer,
        fd_newton_hessian_state=fd_newton_hessian_state,
    )


def test_execute_iteration_transition_noop_preserves_status_and_state(
    tmp_path: Path,
):
    transition = IterationTransition()

    case = _execute_transition_for_test(
        tmp_path,
        transition=transition,
        checkpoint_every=0,
    )

    assert case.result.status == {"status": "running", "reason": "running"}
    assert case.result.continue_loop is False
    assert case.result.break_loop is False
    assert case.result.optimizer is case.optimizer
    assert case.result.fd_newton_hessian_state is case.fd_newton_hessian_state
    assert case.result.hessian_sgd_line_search_active is True
    assert case.result.hessian_sgd_low_accept_steps == 3
    assert case.result.resume_info == {"resume": "kept"}
    assert case.result.planning_state is case.planning_state
    assert case.save_calls == []
    assert case.progress_calls == []


def test_execute_iteration_transition_adagrad_terminal_saves_terminal_status(
    tmp_path: Path,
):
    terminal_status = {
        "status": "converged",
        "reason": "adagrad_restart_schedule_complete",
    }
    transition = IterationTransition(
        status=terminal_status,
        break_loop=True,
        action="adagrad_restart_terminal",
    )

    case = _execute_transition_for_test(tmp_path, transition=transition)

    assert case.result.status == terminal_status
    assert case.result.continue_loop is False
    assert case.result.break_loop is True
    assert case.result.optimizer is case.optimizer
    assert case.result.resume_info == {"resume": "kept"}
    assert len(case.progress_calls) == 1
    assert case.progress_calls[0]["step"] == 8
    assert case.progress_calls[0]["phase"] == "adagrad-restarts:phase0"
    assert case.save_calls[0]["path"] == tmp_path / "latest.pt"
    assert case.save_calls[0]["optimizer"] is case.optimizer
    assert case.save_calls[0]["next_step"] == 9
    assert case.save_calls[0]["status"] == {
        "wrapped": {
            "status": "converged",
            "reason": "adagrad_restart_schedule_complete",
            "base": "kept",
        }
    }


def test_execute_iteration_transition_adagrad_advance_resets_dynamic_phase(
    tmp_path: Path,
):
    transition = IterationTransition(
        continue_loop=True,
        action="adagrad_restart_advance",
        next_adagrad_phase=(2, 9),
    )

    case = _execute_transition_for_test(tmp_path, transition=transition)

    assert case.result.status == {"status": "running", "reason": "running"}
    assert case.result.continue_loop is True
    assert case.result.break_loop is False
    assert case.result.optimizer is None
    assert case.result.resume_info == {}
    assert case.restart_state.phase_index == 2
    assert case.restart_state.phase_start_step == 9
    assert case.restart_state.active_phase_index is None
    assert case.batch_state.optimizer_batch_index is None
    assert case.objective_state.previous_objective is None
    assert case.objective_state.stable_loss_steps == 0
    assert case.result.planning_state.restart_dynamic_phase_index == 2
    assert case.result.planning_state.restart_dynamic_phase_start_step == 9
    assert case.result.planning_state.active_adagrad_restart_phase_index is None
    assert case.result.planning_state.active_optimizer_batch_index is None
    assert case.result.planning_state.previous_objective is None
    assert case.result.planning_state.stable_loss_steps == 0
    assert case.result.planning_state.lbfgsb_fallback_used_count == 5
    assert len(case.progress_calls) == 1
    assert case.save_calls[0]["optimizer"] is None
    assert case.save_calls[0]["status"] == {
        "wrapped": {
            "status": "running",
            "reason": "running",
            "base": "kept",
        }
    }


def test_execute_iteration_transition_adaptive_rebatch_resets_active_state(
    tmp_path: Path,
):
    cache = BatchFinalCache(
        loss=torch.zeros(3),
        grad=torch.zeros(3, 1),
        ready=torch.ones(3, dtype=torch.bool),
    )
    transition = IterationTransition(
        continue_loop=True,
        action="adaptive_rebatch",
        adaptive_rebatch_indices=[0, 2],
    )

    case = _execute_transition_for_test(
        tmp_path,
        transition=transition,
        batch_final_cache=cache,
    )

    assert case.result.status == {"status": "running", "reason": "running"}
    assert case.result.continue_loop is True
    assert case.result.break_loop is False
    assert case.result.optimizer is None
    assert case.result.fd_newton_hessian_state is None
    assert case.result.hessian_sgd_line_search_active is False
    assert case.result.hessian_sgd_low_accept_steps == 0
    assert case.result.resume_info == {}
    assert case.adaptive_state.batch_plan_generation == 3
    assert case.adaptive_state.last_checked_converged_count == 0
    assert case.model.replanned_indices == [(0, 2)]
    assert case.clear_static_calls == [case.model]
    assert case.batch_state.active_index == 0
    assert case.batch_state.local_step == 2
    assert case.batch_state.solver_stage == "full"
    assert case.batch_state.best_nll is None
    assert case.batch_state.best_step is None
    assert case.batch_state.optimizer_batch_index is None
    assert case.objective_state.previous_objective is None
    assert case.objective_state.stable_loss_steps == 0
    assert cache.ready.tolist() == [False, True, False]
    assert case.solver_calls == [(case.model, "full")]
    assert case.result.planning_state.active_batch_index == 0
    assert case.result.planning_state.active_optimizer_batch_index is None
    assert case.result.planning_state.previous_objective is None
    assert case.result.planning_state.stable_loss_steps == 0
    assert case.result.planning_state.lbfgsb_fallback_used_count == 5
    assert case.save_calls[0]["optimizer"] is None
    assert case.save_calls[0]["status"] == {
        "wrapped": {
            "status": "running",
            "reason": "running",
            "base": "kept",
            "active_batch_index": 0,
            "active_solver_stage": "full",
            "active_batch_local_step": 2,
            "previous_objective": None,
            "stable_loss_steps": 0,
            "best_nll_bits": None,
            "best_step": None,
        }
    }


def test_execute_iteration_transition_lbfgsb_loss_schedule_updates_retry_state(
    tmp_path: Path,
):
    objective_state = ObjectiveState(previous_objective=11.0, stable_loss_steps=6)
    lbfgsb_state = LBFGSBRunState(fallback_used_count=8, loss_schedule_index=0)
    transition = IterationTransition(
        continue_loop=True,
        action="lbfgsb_loss_schedule",
        lbfgsb_loss_schedule_next_index=1,
    )

    case = _execute_transition_for_test(
        tmp_path,
        transition=transition,
        objective_state=objective_state,
        lbfgsb_state=lbfgsb_state,
        checkpoint_every=0,
    )

    assert case.result.status == {"status": "running", "reason": "running"}
    assert case.result.continue_loop is True
    assert case.result.break_loop is False
    assert case.result.optimizer is case.optimizer
    assert case.result.fd_newton_hessian_state is case.fd_newton_hessian_state
    assert case.result.hessian_sgd_line_search_active is True
    assert case.result.hessian_sgd_low_accept_steps == 3
    assert case.result.resume_info == {}
    assert lbfgsb_state.loss_schedule_index == 1
    assert objective_state.previous_objective == 11.0
    assert objective_state.stable_loss_steps == 0
    assert case.result.planning_state.previous_objective == 11.0
    assert case.result.planning_state.stable_loss_steps == 0
    assert case.result.planning_state.lbfgsb_fallback_used_count == 8
    assert case.save_calls == []


@pytest.mark.parametrize(
    ("checkpoint_every", "expect_continue", "expect_break"),
    [(1, True, False), (0, False, True)],
)
def test_next_batch_transition_resets_active_batch_and_checkpoint_status(
    tmp_path: Path,
    checkpoint_every: int,
    expect_continue: bool,
    expect_break: bool,
):
    save_calls: list[dict[str, object]] = []
    model = SimpleNamespace(
        selected_batches=[],
        select_batch=lambda index: model.selected_batches.append(index),
    )
    solver_calls: list[tuple[object, str]] = []
    solver = SimpleNamespace(
        configure_active_stage=lambda model_arg, stage: solver_calls.append(
            (model_arg, stage)
        ),
        uses_warmup=lambda: True,
    )
    config = RunConfig(
        species_tree=tmp_path / "sp.nwk",
        families_file=tmp_path / "families.txt",
        out_dir=tmp_path / "out",
        device="cpu",
    )
    objective_state = ObjectiveState(previous_objective=10.0, stable_loss_steps=4)
    batch_state = BatchRunState(
        active_index=0,
        local_step=5,
        solver_stage="full",
        best_nll=3.0,
        best_step=8,
        optimizer_batch_index=0,
    )
    adaptive_state = SimpleNamespace(last_checked_converged_count=4)
    lbfgsb_state = LBFGSBRunState(fallback_used_count=3)

    result = apply_iteration_transition(
        context=IterationTransitionContext(
            config=config,
            model=model,
            evaluation=SimpleNamespace(),
            solver=solver,
            objective_state=objective_state,
            batch_state=batch_state,
            restart_state=RestartRunState(),
            lbfgsb_state=lbfgsb_state,
            adaptive_state=adaptive_state,
            planning_state=_transition_test_planning_state(),
            optimizer=object(),
            fd_newton_hessian_state=object(),
            hessian_sgd_line_search_active=True,
            hessian_sgd_low_accept_steps=2,
            resume_info={"resume": "kept"},
            batch_final_cache=None,
            solver_stage_scope=False,
            batchwise_hessian_sgd=False,
            global_solver_warmup=False,
            lbfgsb_loss_schedule=(),
            current_phase="adam-fd-newton",
            best_checkpoint=tmp_path / "best.pt",
            latest_checkpoint=tmp_path / "latest.pt",
            checkpoint_every=checkpoint_every,
            log_every=10,
            ops=_transition_test_ops(save_calls),
        ),
        inputs=_transition_test_inputs(
            step=11,
            phase="adam-fd-newton",
            step_status={"status": "converged", "reason": "loss_patience"},
            active_batch_count=2,
        ),
    )

    assert result.continue_loop is expect_continue
    assert result.break_loop is expect_break
    assert result.optimizer is None
    assert result.fd_newton_hessian_state is None
    assert result.hessian_sgd_line_search_active is False
    assert result.hessian_sgd_low_accept_steps == 0
    assert result.resume_info == {}
    assert batch_state.active_index == 1
    assert batch_state.local_step == 0
    assert batch_state.solver_stage == "warmup"
    assert objective_state.previous_objective is None
    assert objective_state.stable_loss_steps == 0
    assert adaptive_state.last_checked_converged_count == 0
    assert model.selected_batches == ([1] if checkpoint_every else [])
    assert getattr(model, "cleared_runtime", 0) == (1 if checkpoint_every else 0)
    assert solver_calls == ([(model, "warmup")] if checkpoint_every else [])
    assert result.planning_state.active_batch_index == 1
    assert result.planning_state.active_optimizer_batch_index is None
    assert result.planning_state.previous_objective is None
    assert result.planning_state.stable_loss_steps == 0
    if checkpoint_every:
        assert save_calls[0]["path"] == tmp_path / "latest.pt"
        assert save_calls[0]["optimizer"] is None
        assert save_calls[0]["next_step"] == 12
        assert save_calls[0]["status"] == {
            "wrapped": {
                "status": "running",
                "reason": "running",
                "active_batch_index": 1,
                "active_solver_stage": "warmup",
                "active_batch_local_step": 0,
                "previous_objective": None,
                "stable_loss_steps": 0,
                "best_nll_bits": None,
                "best_step": None,
            }
        }
    else:
        assert save_calls == []


def test_step_stopping_transition_saves_active_checkpoint_fields(tmp_path: Path):
    save_calls: list[dict[str, object]] = []
    optimizer = object()
    fd_state = object()
    objective_state = ObjectiveState(previous_objective=12.5, stable_loss_steps=2)
    batch_state = BatchRunState(active_index=1, local_step=3, solver_stage="full")
    config = RunConfig(
        species_tree=tmp_path / "sp.nwk",
        families_file=tmp_path / "families.txt",
        out_dir=tmp_path / "out",
        device="cpu",
    )

    result = apply_iteration_transition(
        context=IterationTransitionContext(
            config=config,
            model=SimpleNamespace(),
            evaluation=SimpleNamespace(),
            solver=SimpleNamespace(uses_warmup=lambda: False),
            objective_state=objective_state,
            batch_state=batch_state,
            restart_state=RestartRunState(),
            lbfgsb_state=LBFGSBRunState(fallback_used_count=5),
            adaptive_state=SimpleNamespace(last_checked_converged_count=0),
            planning_state=_transition_test_planning_state(),
            optimizer=optimizer,
            fd_newton_hessian_state=fd_state,
            hessian_sgd_line_search_active=True,
            hessian_sgd_low_accept_steps=4,
            resume_info={"resume": "kept"},
            batch_final_cache=None,
            solver_stage_scope=False,
            batchwise_hessian_sgd=False,
            global_solver_warmup=False,
            lbfgsb_loss_schedule=(),
            current_phase="hessian-sgd",
            best_checkpoint=tmp_path / "best.pt",
            latest_checkpoint=tmp_path / "latest.pt",
            checkpoint_every=1,
            log_every=10,
            ops=_transition_test_ops(save_calls),
        ),
        inputs=_transition_test_inputs(
            step=13,
            phase="hessian-sgd",
            step_status={"status": "converged", "reason": "loss_patience"},
            active_batch_count=2,
        ),
    )

    assert result.status == {"status": "converged", "reason": "loss_patience"}
    assert result.continue_loop is False
    assert result.break_loop is True
    assert result.optimizer is optimizer
    assert result.fd_newton_hessian_state is fd_state
    assert result.hessian_sgd_line_search_active is True
    assert result.hessian_sgd_low_accept_steps == 4
    assert result.resume_info == {"resume": "kept"}
    assert result.planning_state.previous_objective == 12.5
    assert result.planning_state.stable_loss_steps == 2
    assert result.planning_state.lbfgsb_fallback_used_count == 5
    assert save_calls[0]["optimizer"] is optimizer
    assert save_calls[0]["next_step"] == 14
    assert save_calls[0]["optimizer_phase"] == "hessian-sgd"
    assert save_calls[0]["status"] == {
        "wrapped": {
            "status": "running",
            "reason": "running",
            "active_batch_index": 1,
            "active_solver_stage": "full",
            "active_batch_local_step": 3,
        }
    }


def _hessian_sgd_line_search_case(tmp_path: Path, *, direct_post_step: bool):
    save_calls: list[dict[str, object]] = []
    objective_state = ObjectiveState(previous_objective=12.5, stable_loss_steps=2)
    batch_state = BatchRunState(
        active_index=0,
        local_step=3,
        solver_stage="warmup",
        best_nll=1.5,
        best_step=4,
        optimizer_batch_index=0,
    )
    lbfgsb_state = LBFGSBRunState(fallback_used_count=5)
    config = RunConfig(
        species_tree=tmp_path / "sp.nwk",
        families_file=tmp_path / "families.txt",
        out_dir=tmp_path / "out",
        device="cpu",
    )
    inputs = _transition_test_inputs(
        step=13,
        phase="hessian-sgd",
        step_status={"status": "converged", "reason": "loss_patience"},
        active_batch_count=1,
    )
    inputs.hessian_sgd_activate_line_search = True
    model = SimpleNamespace()
    evaluation = SimpleNamespace()
    solver = SimpleNamespace(uses_warmup=lambda: False)
    restart_state = RestartRunState(active_phase_index=2)
    adaptive_state = SimpleNamespace(last_checked_converged_count=0)
    planning_state = _transition_test_planning_state()
    optimizer = object()
    fd_newton_hessian_state = object()
    ops = _transition_test_ops(save_calls)

    transition_kwargs = dict(
        config=config,
        model=model,
        evaluation=evaluation,
        solver=solver,
        objective_state=objective_state,
        batch_state=batch_state,
        restart_state=restart_state,
        lbfgsb_state=lbfgsb_state,
        adaptive_state=adaptive_state,
        planning_state=planning_state,
        optimizer=optimizer,
        fd_newton_hessian_state=fd_newton_hessian_state,
        hessian_sgd_line_search_active=False,
        hessian_sgd_low_accept_steps=2,
        resume_info={"resume": "discard"},
        batch_final_cache=None,
        solver_stage_scope=True,
        batchwise_hessian_sgd=True,
        global_solver_warmup=False,
        lbfgsb_loss_schedule=(),
        current_phase="hessian-sgd",
        best_checkpoint=tmp_path / "best.pt",
        latest_checkpoint=tmp_path / "latest.pt",
        checkpoint_every=1,
        ops=ops,
    )
    if direct_post_step:
        post_step_module = _import_transition_post_step_module()
        result = post_step_module.execute_iteration_post_step_transition(
            **transition_kwargs,
            status=inputs.status,
            step=inputs.step,
            phase=inputs.phase,
            row=inputs.row,
            checkpoint_status=inputs.checkpoint_status,
            hessian_sgd_activate_line_search=(
                inputs.hessian_sgd_activate_line_search
            ),
            step_status=inputs.step_status,
            can_lbfgsb_retry=inputs.can_lbfgsb_retry,
            active_objective_scope=inputs.active_objective_scope,
            active_batch_count=inputs.active_batch_count,
            lbfgsb_high_kkt_status=inputs.lbfgsb_high_kkt_status,
        )
    else:
        result = apply_iteration_transition(
            context=IterationTransitionContext(
                **transition_kwargs,
                log_every=10,
            ),
            inputs=inputs,
        )

    return SimpleNamespace(
        result=result,
        save_calls=save_calls,
        objective_state=objective_state,
        batch_state=batch_state,
    )


def _assert_hessian_sgd_line_search_transition_result(case):
    result = case.result

    assert result.status == {"status": "running", "reason": "running"}
    assert result.continue_loop is True
    assert result.break_loop is False
    assert result.optimizer is None
    assert result.fd_newton_hessian_state is None
    assert result.hessian_sgd_line_search_active is True
    assert result.hessian_sgd_low_accept_steps == 0
    assert result.resume_info == {}
    assert case.objective_state.previous_objective is None
    assert case.objective_state.stable_loss_steps == 0
    assert case.batch_state.local_step == 0
    assert case.batch_state.solver_stage == "full"
    assert case.batch_state.best_nll is None
    assert case.batch_state.best_step is None
    assert case.batch_state.optimizer_batch_index is None
    assert result.planning_state.current_phase == "hessian-sgd"
    assert result.planning_state.active_optimizer_batch_index is None
    assert result.planning_state.active_adagrad_restart_phase_index == 2
    assert result.planning_state.previous_objective is None
    assert result.planning_state.stable_loss_steps == 0
    assert result.planning_state.lbfgsb_fallback_used_count == 5
    assert case.save_calls == []


def test_hessian_sgd_line_search_transition_resets_active_state(tmp_path: Path):
    case = _hessian_sgd_line_search_case(tmp_path, direct_post_step=False)

    _assert_hessian_sgd_line_search_transition_result(case)


def test_direct_post_step_hessian_sgd_line_search_matches_wrapper_shape(
    tmp_path: Path,
):
    case = _hessian_sgd_line_search_case(tmp_path, direct_post_step=True)

    _assert_hessian_sgd_line_search_transition_result(case)


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


def test_apply_resume_checkpoint_state_updates_run_state_and_planning_context(
    tmp_path: Path,
):
    events: list[str] = []

    class Model:
        def __init__(self):
            self.replanned_indices = None

        def replan_resident_batches(self, indices):
            events.append("replan")
            self.replanned_indices = tuple(indices)

    class AdaptiveState:
        def __init__(self):
            self.restore_call = None

        def restore_from_resume(
            self,
            *,
            model,
            resume_state,
            active_batch_index,
            checkpoint_path,
        ):
            events.append("adaptive")
            self.restore_call = {
                "model": model,
                "resume_state": resume_state,
                "active_batch_index": active_batch_index,
                "checkpoint_path": checkpoint_path,
            }
            return [5, 8]

    config = SimpleNamespace(resume_from=tmp_path / "resume.pt", steps=9)
    payload = {
        "step": 2,
        "next_step": 3,
        "status": {
            "status": "running",
            "best_nll_bits": 12.5,
            "best_step": 2,
            "previous_objective": 14.0,
            "stable_loss_steps": 6,
            "active_batch_index": 3,
            "active_solver_stage": "warmup",
            "active_batch_local_step": 4,
            "adagrad_restart_dynamic_phase_index": 2,
            "adagrad_restart_dynamic_phase_start_step": 11,
            "lbfgsb_fallback_used_count": 7,
            "lbfgsb_loss_schedule_index": 99,
            "lbfgsb_best_retry_count": 2,
        },
    }
    planning_state = _StepPlanningState(
        restart_dynamic_phase_index=0,
        restart_dynamic_phase_start_step=0,
        current_phase="",
        active_batch_index=0,
        active_optimizer_batch_index=None,
        active_adagrad_restart_phase_index=None,
        previous_objective=None,
        stable_loss_steps=0,
        lbfgsb_fallback_used_count=0,
        optimizer=None,
    )
    run_state = _OptimizationRunState(
        objective_state=ObjectiveState(),
        batch_state=BatchRunState(),
        restart_state=RestartRunState(dynamic_enabled=True),
        lbfgsb_state=LBFGSBRunState(),
        planning_state=planning_state,
        current_phase="",
        batch_final_cache=None,
    )
    planning_context = _StepPlanningContext(
        solver=object(),
        config=config,
        adagrad_restart_specs=(),
        adagrad_restart_step_limit=None,
        adagrad_restart_dynamic_enabled=True,
        adagrad_restart_dynamic_state_loaded=False,
        batchwise_active_optimizer=True,
        batchwise_active_optimizer_phases=frozenset({"batched-lbfgs"}),
        batchwise_batched_lbfgs=True,
        batchwise_fd_newton=False,
        batchwise_hessian_sgd=False,
        clear_cached_solver_runtime_state=lambda model: None,
        make_optimizer=lambda model, phase: None,
    )
    model = Model()
    adaptive_state = AdaptiveState()

    def load_checkpoint(path):
        events.append("load")
        assert path == config.resume_from
        return payload

    def validate_checkpoint_model_compatibility(**kwargs):
        events.append("validate")
        assert kwargs == {
            "path": config.resume_from,
            "config": config,
            "model": model,
            "payload": payload,
        }

    def restore_model_theta(model_arg, payload_arg):
        events.append("restore")
        assert model_arg is model
        assert payload_arg is payload

    result = _apply_resume_checkpoint_state(
        config=config,
        model=model,
        run_state=run_state,
        planning_context=planning_context,
        lbfgsb_loss_schedule=(object(), object()),
        solver_warmup_enabled=True,
        batchwise_active_optimizer=True,
        adagrad_restart_dynamic_enabled=True,
        adaptive_rebatch_enabled=True,
        adaptive_state=adaptive_state,
        load_checkpoint=load_checkpoint,
        validate_checkpoint_model_compatibility=(
            validate_checkpoint_model_compatibility
        ),
        restore_model_theta=restore_model_theta,
    )

    assert events == ["load", "validate", "restore", "adaptive", "replan"]
    assert run_state.resume_payload is payload
    assert run_state.start_step == 3
    assert run_state.objective_state.best_nll is None
    assert run_state.objective_state.best_step is None
    assert run_state.objective_state.previous_objective == 14.0
    assert run_state.objective_state.stable_loss_steps == 6
    assert run_state.batch_state.active_index == 3
    assert run_state.batch_state.solver_stage == "warmup"
    assert run_state.batch_state.local_step == 4
    assert run_state.batch_state.best_nll == 12.5
    assert run_state.batch_state.best_step == 2
    assert run_state.restart_state.phase_index == 2
    assert run_state.restart_state.phase_start_step == 11
    assert run_state.lbfgsb_state.fallback_used_count == 7
    assert run_state.lbfgsb_state.loss_schedule_index == 1
    assert run_state.lbfgsb_state.best_retry_count == 2
    assert result.planning_context.adagrad_restart_dynamic_state_loaded
    assert result.planning_state.restart_dynamic_phase_index == 2
    assert result.planning_state.restart_dynamic_phase_start_step == 11
    assert result.planning_state.current_phase == "warmup"
    assert result.planning_state.active_batch_index == 3
    assert result.planning_state.previous_objective == 14.0
    assert result.planning_state.stable_loss_steps == 6
    assert result.planning_state.lbfgsb_fallback_used_count == 7
    assert adaptive_state.restore_call is not None
    assert adaptive_state.restore_call["active_batch_index"] == 3
    assert adaptive_state.restore_call["checkpoint_path"] == str(config.resume_from)
    assert model.replanned_indices == (5, 8)
