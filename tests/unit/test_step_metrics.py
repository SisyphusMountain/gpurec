from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from gpurec.workflow._step_metrics import (
    _adagrad_restart_step_metrics,
    _batched_lbfgs_step_metrics,
    _cached_genewise_loss_metrics,
    _cached_scalar_loss_metrics,
    _hessian_sgd_budget_metrics,
    _projected_grad_inf_from_optimizer_state,
    _projected_optimizer_step_metrics,
)


class _Model:
    def __init__(self) -> None:
        self.theta = torch.nn.Parameter(
            torch.tensor([[0.0, 1.0, -1.0], [0.5, -0.5, 0.25]], dtype=torch.float64)
        )
        self.theta.grad = torch.tensor(
            [[0.1, -0.2, 0.3], [0.4, -0.5, 0.6]],
            dtype=torch.float64,
        )

    def solver_stat_records(self) -> list[dict[str, object]]:
        return []


class _Evaluation:
    def __init__(self) -> None:
        self.calls = 0

    def projected_grad_inf(
        self,
        model: object,
        *,
        lower_bound: float,
        upper_bound: float,
    ) -> tuple[torch.Tensor, float]:
        self.calls += 1
        assert lower_bound == -3.0
        assert upper_bound == 2.0
        return torch.empty(0), 9.25


@pytest.mark.parametrize("metric_prefix", ["projected_lbfgs", "lbfgsb"])
def test_projected_optimizer_step_metrics_preserve_optional_schema(
    metric_prefix: str,
) -> None:
    opt_state = {
        "last_grad_evals": 2,
        "last_loss_evals": 5,
        "last_accepted": True,
        "last_alpha": 0.125,
        "last_step_inf": 0.75,
        "last_direction_kind": "lbfgs",
        "last_line_search_decrease": 1.5,
        "last_armijo_required_decrease": 0.25,
        "last_fallback_attempted": True,
        "last_fallback_used": False,
        "last_fallback_alpha": 0.03125,
        "last_fallback_loss_evals": 4,
        "last_fallback_max_loss_evals": 7,
        "last_fallback_budget_exhausted": True,
        "last_fallback_reason": "budget",
        "last_high_kkt_stall_count": 3,
        "last_history_cleared_for_fallback": True,
    }

    metrics = _projected_optimizer_step_metrics(
        metric_prefix=metric_prefix,
        opt_state=opt_state,
        projected_loss_evals=11,
        theta_step=8.0,
        projected_grad_inf=6.5,
    )

    assert metrics == {
        "grad/projected_inf": 6.5,
        f"optimizer/{metric_prefix}_grad_evals": 2.0,
        f"optimizer/{metric_prefix}_loss_evals": 5.0,
        f"optimizer/{metric_prefix}_accepted": True,
        f"optimizer/{metric_prefix}_alpha": 0.125,
        f"optimizer/{metric_prefix}_step_inf": 0.75,
        f"optimizer/{metric_prefix}_line_search_decrease": 1.5,
        f"optimizer/{metric_prefix}_armijo_required_decrease": 0.25,
        f"optimizer/{metric_prefix}_fallback_attempted": True,
        f"optimizer/{metric_prefix}_fallback_used": False,
        f"optimizer/{metric_prefix}_fallback_alpha": 0.03125,
        f"optimizer/{metric_prefix}_fallback_loss_evals": 4.0,
        f"optimizer/{metric_prefix}_fallback_budget_exhausted": True,
        f"optimizer/{metric_prefix}_fallback_reason": "budget",
        f"optimizer/{metric_prefix}_high_kkt_stall_count": 3.0,
        f"optimizer/{metric_prefix}_history_cleared_for_fallback": True,
        f"optimizer/{metric_prefix}_direction_kind": "lbfgs",
        f"optimizer/{metric_prefix}_fallback_max_loss_evals": 7.0,
    }


def test_projected_optimizer_step_metrics_omit_absent_optional_keys() -> None:
    metrics = _projected_optimizer_step_metrics(
        metric_prefix="lbfgsb",
        opt_state={},
        projected_loss_evals=4,
        theta_step=0.5,
        projected_grad_inf=1.25,
    )

    assert metrics["optimizer/lbfgsb_loss_evals"] == 4.0
    assert metrics["optimizer/lbfgsb_step_inf"] == 0.5
    assert "optimizer/lbfgsb_direction_kind" not in metrics
    assert "optimizer/lbfgsb_fallback_max_loss_evals" not in metrics


def test_projected_grad_inf_prefers_optimizer_cached_projected_gradient() -> None:
    evaluation = _Evaluation()

    projected_inf = _projected_grad_inf_from_optimizer_state(
        opt_state={"last_projected_grad": torch.tensor([-1.0, 4.5, 2.0])},
        evaluation=evaluation,
        model=object(),
        lower_bound=-3.0,
        upper_bound=2.0,
    )

    assert projected_inf == 4.5
    assert evaluation.calls == 0


def test_projected_grad_inf_falls_back_to_evaluation() -> None:
    evaluation = _Evaluation()

    projected_inf = _projected_grad_inf_from_optimizer_state(
        opt_state={},
        evaluation=evaluation,
        model=object(),
        lower_bound=-3.0,
        upper_bound=2.0,
    )

    assert projected_inf == 9.25
    assert evaluation.calls == 1


def test_cached_loss_metrics_preserve_base_metrics_and_likelihood_totals() -> None:
    model = _Model()

    scalar_metrics = _cached_scalar_loss_metrics(
        {"preexisting": 1.0},
        model,
        torch.tensor(3.25),
    )
    genewise_metrics = _cached_genewise_loss_metrics(
        {"preexisting": 2.0},
        model,
        torch.tensor([1.0, 2.5]),
    )

    assert scalar_metrics["preexisting"] == 1.0
    assert scalar_metrics["likelihood/data_nll_bits"] == 3.25
    assert scalar_metrics["likelihood/log_likelihood_bits"] == -3.25
    assert scalar_metrics["grad/inf"] == pytest.approx(0.6)
    assert genewise_metrics["preexisting"] == 2.0
    assert genewise_metrics["likelihood/data_nll_bits"] == 3.5
    assert genewise_metrics["likelihood/log_likelihood_bits"] == -3.5
    assert genewise_metrics["theta/D/max"] == pytest.approx(0.5)


def test_batched_lbfgs_step_metrics_summarize_optimizer_state_tensors() -> None:
    metrics = _batched_lbfgs_step_metrics(
        opt_state={
            "last_n_iter": 6,
            "last_accepted": torch.tensor([True, False, True]),
            "last_alpha": torch.tensor([0.25, 0.5, 1.0]),
        },
        batched_grad_evals=3,
        batched_loss_evals=4,
        reused_optimizer_gradient=True,
    )

    assert metrics == {
        "optimizer/batched_lbfgs_grad_evals": 3.0,
        "optimizer/batched_lbfgs_loss_evals": 4.0,
        "optimizer/batched_lbfgs_reused_gradient": True,
        "optimizer/batched_lbfgs_inner_iters": 6.0,
        "optimizer/batched_lbfgs_accepted_rows": 2.0,
        "optimizer/batched_lbfgs_accepted_fraction": pytest.approx(2.0 / 3.0),
        "optimizer/batched_lbfgs_alpha_mean": pytest.approx(7.0 / 12.0),
        "optimizer/batched_lbfgs_alpha_max": 1.0,
    }


def test_hessian_sgd_budget_metrics_select_validation_warmup_and_normal_budget() -> None:
    config = SimpleNamespace(
        hessian_sgd_validation_fixed_iters_pi=21,
        hessian_sgd_validation_neumann_terms=13,
        hessian_sgd_normal_fixed_iters_pi=None,
        hessian_sgd_normal_neumann_terms=31,
        fixed_iters_pi=8,
        neumann_terms=9,
    )
    solver = SimpleNamespace(hessian_sgd_warmup_iters=lambda model: 17)

    validation = _hessian_sgd_budget_metrics(
        config=config,
        solver=solver,
        model=object(),
        active_solver_stage="full",
        hessian_sgd_validation_step=True,
    )
    warmup = _hessian_sgd_budget_metrics(
        config=config,
        solver=solver,
        model=object(),
        active_solver_stage="warmup",
        hessian_sgd_validation_step=False,
    )
    normal = _hessian_sgd_budget_metrics(
        config=config,
        solver=solver,
        model=object(),
        active_solver_stage="full",
        hessian_sgd_validation_step=False,
    )

    assert validation == {
        "optimizer/hessian_sgd_validation_step": True,
        "optimizer/hessian_sgd_solver_budget": "validation",
        "optimizer/hessian_sgd_active_fixed_iters_pi": 21.0,
        "optimizer/hessian_sgd_active_neumann_terms": 13.0,
    }
    assert warmup == {
        "optimizer/hessian_sgd_validation_step": False,
        "optimizer/hessian_sgd_solver_budget": "warmup",
        "optimizer/hessian_sgd_active_fixed_iters_pi": 17.0,
        "optimizer/hessian_sgd_active_neumann_terms": 17.0,
    }
    assert normal == {
        "optimizer/hessian_sgd_validation_step": False,
        "optimizer/hessian_sgd_solver_budget": "normal",
        "optimizer/hessian_sgd_active_fixed_iters_pi": 8.0,
        "optimizer/hessian_sgd_active_neumann_terms": 31.0,
    }


def test_adagrad_restart_step_metrics_preserve_phase_field_types() -> None:
    active_phase = SimpleNamespace(
        name="refine",
        index=2,
        start_step=10,
        phase=SimpleNamespace(
            steps=5,
            budget=123,
            fixed_iters_e=7,
            fixed_iters_pi=11,
            neumann_terms=13,
            lr=0.0125,
        ),
    )

    metrics = _adagrad_restart_step_metrics(active_phase=active_phase, step=10)

    assert metrics == {
        "optimizer/adagrad_restart_phase": "refine",
        "optimizer/adagrad_restart_phase_index": 2,
        "optimizer/adagrad_restart_phase_step": 0,
        "optimizer/adagrad_restart_phase_steps": 5,
        "optimizer/adagrad_restart_budget": 123,
        "optimizer/adagrad_restart_fixed_iters_E": 7,
        "optimizer/adagrad_restart_fixed_iters_Pi": 11,
        "optimizer/adagrad_restart_neumann_terms": 13,
        "optimizer/adagrad_restart_lr": 0.0125,
        "optimizer/adagrad_restart_restarted": True,
    }
