from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import gpurec.api.autograd as api_autograd
import gpurec.api.model as api_model


def _implicit_gradient_static(
    *,
    pi_adjoint_warmstart: bool,
    pi_adjoint_cache=None,
    pi_adjoint_cache_update_mode: str = "immediate",
    pi_fixed_point_relaxation: float = 1.0,
):
    return SimpleNamespace(
        wave_layout={
            "root_clade_ids": torch.tensor([0, 1], dtype=torch.long),
            "family_idx": torch.tensor([0, 1], dtype=torch.long),
        },
        species_helpers={},
        unnorm_row_max=torch.zeros(2, dtype=torch.float64),
        specieswise=False,
        genewise=True,
        device=torch.device("cpu"),
        dtype=torch.float64,
        neumann_terms=4,
        use_pruning=True,
        pruning_threshold=1e-6,
        ancestors_T=None,
        origination_prior=None,
        origination_probs=None,
        adaptive_neumann_terms=False,
        gradient_change_tol=1e-4,
        gradient_change_rtol=1e-4,
        convergence_check_interval=4,
        pi_adjoint_warmstart=pi_adjoint_warmstart,
        pi_adjoint_cache=pi_adjoint_cache,
        pi_adjoint_pending_cache=None,
        pi_adjoint_cache_update_mode=pi_adjoint_cache_update_mode,
        pi_fixed_point_relaxation=pi_fixed_point_relaxation,
        last_solver_stats=None,
    )


def _model_with_active_state(
    *,
    mode: str,
    theta_requires_grad: bool = True,
):
    model = api_model.GeneReconModel.__new__(api_model.GeneReconModel)
    torch.nn.Module.__init__(model)
    static = object()
    theta = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    theta.requires_grad_(theta_requires_grad)
    model._mode = mode
    model._active_static = lambda: static
    model._active_theta = lambda: theta
    return model, static, theta


def test_forward_with_non_grad_theta_uses_resident_no_grad_evaluator(monkeypatch):
    model, static, theta = _model_with_active_state(
        mode="global",
        theta_requires_grad=False,
    )
    expected = torch.tensor(3.5, dtype=torch.float32)
    calls: list[dict[str, object]] = []

    def fake_evaluate_resident_no_grad(
        static_arg,
        theta_arg,
        *,
        per_family=False,
    ):
        calls.append(
            {
                "static": static_arg,
                "theta": theta_arg,
                "per_family": per_family,
                "grad_enabled": torch.is_grad_enabled(),
            }
        )
        return expected

    monkeypatch.setattr(
        api_model,
        "evaluate_resident_no_grad",
        fake_evaluate_resident_no_grad,
    )

    actual = model.forward()

    torch.testing.assert_close(actual, expected.to(dtype=theta.dtype))
    assert len(calls) == 1
    assert calls[0]["static"] is static
    assert calls[0]["theta"] is theta
    assert calls[0]["per_family"] is False
    assert calls[0]["grad_enabled"] is False


def test_shared_per_family_forward_uses_same_no_grad_evaluator(monkeypatch):
    model, static, theta = _model_with_active_state(mode="global")
    expected = torch.tensor([1.25, 2.5], dtype=torch.float32)
    calls: list[dict[str, object]] = []

    def fake_evaluate_resident_no_grad(
        static_arg,
        theta_arg,
        *,
        per_family=False,
    ):
        calls.append(
            {
                "static": static_arg,
                "theta": theta_arg,
                "per_family": per_family,
                "grad_enabled": torch.is_grad_enabled(),
            }
        )
        return expected

    monkeypatch.setattr(
        api_model,
        "evaluate_resident_no_grad",
        fake_evaluate_resident_no_grad,
    )

    with torch.no_grad():
        actual = model.forward(reduce="per_family")

    torch.testing.assert_close(actual, expected.to(dtype=theta.dtype))
    assert len(calls) == 1
    assert calls[0]["static"] is static
    assert calls[0]["theta"] is theta
    assert calls[0]["per_family"] is True
    assert calls[0]["grad_enabled"] is False


def test_genewise_nll_per_family_no_grad_uses_resident_evaluator(monkeypatch):
    model, static, theta = _model_with_active_state(mode="genewise")
    expected = torch.tensor([0.5, 0.75], dtype=torch.float32)
    calls: list[dict[str, object]] = []

    def fake_evaluate_resident_no_grad(
        static_arg,
        theta_arg,
        *,
        per_family=False,
    ):
        calls.append(
            {
                "static": static_arg,
                "theta": theta_arg,
                "per_family": per_family,
                "grad_enabled": torch.is_grad_enabled(),
            }
        )
        return expected

    monkeypatch.setattr(
        api_model,
        "evaluate_resident_no_grad",
        fake_evaluate_resident_no_grad,
    )

    with torch.no_grad():
        actual = model.nll_per_family()

    torch.testing.assert_close(actual, expected.to(dtype=theta.dtype))
    assert len(calls) == 1
    assert calls[0]["static"] is static
    assert calls[0]["theta"] is theta
    assert calls[0]["per_family"] is True
    assert calls[0]["grad_enabled"] is False


def test_evaluate_static_state_no_grad_delegates_to_resident_evaluator(monkeypatch):
    static = object()
    theta = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    expected = torch.tensor([4.0, 5.0], dtype=torch.float64)
    calls: list[dict[str, object]] = []

    def fake_evaluate_resident_no_grad(
        static_arg,
        theta_arg,
        *,
        per_family=False,
    ):
        calls.append(
            {
                "static": static_arg,
                "theta": theta_arg,
                "per_family": per_family,
            }
        )
        return expected

    monkeypatch.setattr(
        api_model,
        "evaluate_resident_no_grad",
        fake_evaluate_resident_no_grad,
    )

    loss, grad = api_model._evaluate_static_state(
        static,
        theta,
        need_grad=False,
        per_family=True,
    )

    assert loss is expected
    assert grad is None
    assert len(calls) == 1
    assert calls[0]["static"] is static
    assert calls[0]["theta"] is theta
    assert calls[0]["per_family"] is True


@pytest.mark.parametrize(
    ("genewise", "specieswise", "theta", "expected_e_shape"),
    [
        (False, False, torch.zeros(3, dtype=torch.float64), (3,)),
        (False, True, torch.zeros(3, 3, dtype=torch.float64), (3,)),
        (True, False, torch.zeros(2, 3, dtype=torch.float64), (2, 3)),
    ],
)
def test_solve_resident_e_passes_layout_e_shape(
    monkeypatch,
    genewise: bool,
    specieswise: bool,
    theta: torch.Tensor,
    expected_e_shape: tuple[int, ...],
):
    static = SimpleNamespace(
        wave_layout={"root_clade_ids": torch.tensor([4, 9], dtype=torch.long)},
        species_helpers={"S": 3},
        unnorm_row_max=torch.zeros(3, dtype=torch.float64),
        genewise=genewise,
        specieswise=specieswise,
        device=torch.device("cpu"),
        dtype=torch.float64,
        fixed_iters_E=1,
        max_iters_E=4,
        tol_E=1e-8,
        adaptive_iters=False,
        e_logsumexp_tol=1e-5,
        convergence_check_interval=2,
        ancestors_T=torch.eye(3, dtype=torch.float64),
    )
    calls: list[dict[str, object]] = []

    def fake_e_fixed_point(**kwargs):
        calls.append(kwargs)
        return {"E": torch.full(expected_e_shape, -1.0), "iterations": 1}

    monkeypatch.setattr(api_autograd, "E_fixed_point", fake_e_fixed_point)

    result = api_autograd.solve_resident_e(static, theta)

    assert result.e_out["E"].shape == expected_e_shape
    assert calls[0]["e_shape"] == expected_e_shape


def test_shared_no_grad_full_loss_reuses_pi_scratch(monkeypatch):
    model = api_model.GeneReconModel.__new__(api_model.GeneReconModel)
    torch.nn.Module.__init__(model)
    theta = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    statics = [object(), object()]
    model._batched_resident = True
    model._mode = "specieswise"
    model._batch_specs = [object(), object()]
    model.batch_metadata = [
        SimpleNamespace(clade_count=3),
        SimpleNamespace(clade_count=5),
    ]
    model._dataset = SimpleNamespace(
        device=torch.device("cpu"),
        dtype=torch.float32,
        S=7,
    )
    model._origination_prior = SimpleNamespace(is_shared=True, probs=None)
    model._ensure_batch_static = lambda batch_idx: statics[batch_idx]
    e_solve = SimpleNamespace(e_out={"E": torch.full((7,), -3.0)})
    calls: list[dict[str, object]] = []

    def fake_solve_resident_e(static_arg, theta_arg):
        assert static_arg is statics[0]
        assert theta_arg is theta
        return e_solve

    def fake_evaluate_resident_no_grad_with_solved_e(
        static_arg,
        e_solve_arg,
        *,
        scratch_tensors=None,
        origination_denominator=None,
    ):
        calls.append(
            {
                "static": static_arg,
                "e_solve": e_solve_arg,
                "scratch_tensors": scratch_tensors,
                "origination_denominator": origination_denominator,
            }
        )
        return torch.tensor(float(len(calls)), dtype=torch.float32)

    monkeypatch.setattr(api_model, "solve_resident_e", fake_solve_resident_e)
    monkeypatch.setattr(
        api_model,
        "evaluate_resident_no_grad_with_solved_e",
        fake_evaluate_resident_no_grad_with_solved_e,
    )

    loss, grad = model._stream_full_batches(theta, need_grad=False)

    assert grad is None
    torch.testing.assert_close(loss, torch.tensor(3.0))
    assert [call["static"] for call in calls] == statics
    assert all(call["e_solve"] is e_solve for call in calls)
    first_scratch = calls[0]["scratch_tensors"]
    assert isinstance(first_scratch, tuple)
    assert tuple(first_scratch[0].shape) == (5, 7)
    assert tuple(first_scratch[1].shape) == (5, 7)
    assert calls[1]["scratch_tensors"] is first_scratch
    assert calls[0]["origination_denominator"] is calls[1]["origination_denominator"]
    assert torch.is_tensor(calls[0]["origination_denominator"])


def test_evaluate_static_state_grad_uses_resident_gradient_boundary(monkeypatch):
    theta = torch.tensor([[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]], dtype=torch.float64)
    static = SimpleNamespace(
        genewise=True,
    )
    e = torch.ones((2, 2), dtype=theta.dtype)
    e_out = {
        "E": e,
        "E_s1": e + 1.0,
        "E_s2": e + 2.0,
        "E_bar": e + 3.0,
    }
    pi_out = {
        "Pi_wave_ordered": torch.ones((2, 2), dtype=theta.dtype),
        "Pibar_wave_ordered": torch.ones((2, 2), dtype=theta.dtype) * 2.0,
        "uniform_pibar_row_max": torch.arange(2, dtype=theta.dtype),
    }
    theta_eval = theta.detach().clone()
    solve = api_autograd.ResidentSolveResult(
        theta=theta_eval,
        e_out=e_out,
        pi_out=pi_out,
        log_p_s=torch.full((2,), 0.1, dtype=theta.dtype),
        log_p_d=torch.full((2,), 0.2, dtype=theta.dtype),
        log_p_l=torch.full((2,), 0.3, dtype=theta.dtype),
        max_transfer=torch.full((2,), 0.4, dtype=theta.dtype),
    )
    loss_vec = torch.tensor([1.25, 2.5], dtype=theta.dtype)
    expected_grad = torch.arange(theta.numel(), dtype=theta.dtype).reshape_as(theta)
    forward_calls: list[dict[str, object]] = []
    grad_calls: list[dict[str, object]] = []

    def fake_evaluate_resident_gradient_forward(
        static_arg,
        theta_arg,
        *,
        warm_start_E=None,
    ):
        forward_calls.append(
            {
                "static": static_arg,
                "theta": theta_arg,
                "warm_start_E": warm_start_E,
            }
        )
        return api_autograd.ResidentGradientForwardResult(
            solve=solve,
            loss_vec=loss_vec,
        )

    def fake_compute_resident_implicit_gradient(
        static_arg,
        *,
        theta,
        pi_wave_ordered,
        pibar_wave_ordered,
        e,
        ebar,
        e_s1,
        e_s2,
        log_p_s,
        log_p_d,
        log_p_l,
        max_transfer,
        uniform_pibar_row_max,
    ):
        grad_calls.append(
            {
                "static": static_arg,
                "theta": theta,
                "pi_wave_ordered": pi_wave_ordered,
                "pibar_wave_ordered": pibar_wave_ordered,
                "e": e,
                "ebar": ebar,
                "e_s1": e_s1,
                "e_s2": e_s2,
                "log_p_s": log_p_s,
                "log_p_d": log_p_d,
                "log_p_l": log_p_l,
                "max_transfer": max_transfer,
                "uniform_pibar_row_max": uniform_pibar_row_max,
            }
        )
        return expected_grad

    monkeypatch.setattr(
        api_model,
        "evaluate_resident_gradient_forward",
        fake_evaluate_resident_gradient_forward,
    )
    monkeypatch.setattr(
        api_model,
        "compute_resident_implicit_gradient",
        fake_compute_resident_implicit_gradient,
    )

    loss, grad = api_model._evaluate_static_state(
        static,
        theta,
        need_grad=True,
        per_family=True,
    )

    torch.testing.assert_close(loss, loss_vec)
    assert loss.requires_grad is False
    torch.testing.assert_close(grad, expected_grad)
    assert grad.requires_grad is False
    assert forward_calls == [
        {"static": static, "theta": theta, "warm_start_E": None}
    ]
    assert len(grad_calls) == 1
    assert grad_calls[0]["static"] is static
    assert grad_calls[0]["theta"] is theta_eval
    assert grad_calls[0]["pi_wave_ordered"] is pi_out["Pi_wave_ordered"]
    assert grad_calls[0]["pibar_wave_ordered"] is pi_out["Pibar_wave_ordered"]
    assert grad_calls[0]["e"] is e_out["E"]
    assert grad_calls[0]["ebar"] is e_out["E_bar"]
    assert grad_calls[0]["e_s1"] is e_out["E_s1"]
    assert grad_calls[0]["e_s2"] is e_out["E_s2"]
    assert grad_calls[0]["log_p_s"] is solve.log_p_s
    assert grad_calls[0]["log_p_d"] is solve.log_p_d
    assert grad_calls[0]["log_p_l"] is solve.log_p_l
    assert grad_calls[0]["max_transfer"] is solve.max_transfer
    assert grad_calls[0]["uniform_pibar_row_max"] is pi_out["uniform_pibar_row_max"]


def test_evaluate_static_state_keeps_staged_pi_adjoint_for_workflow_commit(
    monkeypatch,
):
    theta = torch.zeros((2, 3), dtype=torch.float64)
    accepted = torch.full((2, 2), 0.25, dtype=theta.dtype)
    pending = torch.full((2, 2), 0.75, dtype=theta.dtype)
    static = SimpleNamespace(
        genewise=True,
        warm_E=object(),
        clear_runtime_after_backward=True,
        pi_adjoint_warmstart=True,
        pi_adjoint_cache=accepted,
        pi_adjoint_pending_cache=None,
        pi_adjoint_cache_update_mode="stage",
    )
    e = torch.ones((2, 2), dtype=theta.dtype)
    solve = api_autograd.ResidentSolveResult(
        theta=theta.detach().clone(),
        e_out={"E": e, "E_s1": e, "E_s2": e, "E_bar": e},
        pi_out={
            "Pi_wave_ordered": e,
            "Pibar_wave_ordered": e,
            "uniform_pibar_row_max": None,
        },
        log_p_s=torch.zeros(2, dtype=theta.dtype),
        log_p_d=torch.zeros(2, dtype=theta.dtype),
        log_p_l=torch.zeros(2, dtype=theta.dtype),
        max_transfer=torch.zeros(2, dtype=theta.dtype),
    )

    monkeypatch.setattr(
        api_model,
        "evaluate_resident_gradient_forward",
        lambda *_args, **_kwargs: api_autograd.ResidentGradientForwardResult(
            solve=solve,
            loss_vec=torch.ones(2, dtype=theta.dtype),
        ),
    )

    def fake_compute(static_arg, **_kwargs):
        static_arg.pi_adjoint_pending_cache = pending
        return torch.ones_like(theta)

    monkeypatch.setattr(api_model, "compute_resident_implicit_gradient", fake_compute)

    _loss, grad = api_model._evaluate_static_state(
        static,
        theta,
        need_grad=True,
        per_family=True,
    )

    torch.testing.assert_close(grad, torch.ones_like(theta))
    assert static.warm_E is None
    torch.testing.assert_close(static.pi_adjoint_cache, accepted)
    torch.testing.assert_close(static.pi_adjoint_pending_cache, pending)


def test_evaluate_static_state_clears_immediate_pi_adjoint_runtime_cache(
    monkeypatch,
):
    theta = torch.zeros((2, 3), dtype=torch.float64)
    static = SimpleNamespace(
        genewise=True,
        warm_E=object(),
        clear_runtime_after_backward=True,
        pi_adjoint_warmstart=True,
        pi_adjoint_cache=torch.full((2, 2), 0.25, dtype=theta.dtype),
        pi_adjoint_pending_cache=torch.full((2, 2), 0.75, dtype=theta.dtype),
        pi_adjoint_cache_update_mode="immediate",
    )
    e = torch.ones((2, 2), dtype=theta.dtype)
    solve = api_autograd.ResidentSolveResult(
        theta=theta.detach().clone(),
        e_out={"E": e, "E_s1": e, "E_s2": e, "E_bar": e},
        pi_out={
            "Pi_wave_ordered": e,
            "Pibar_wave_ordered": e,
            "uniform_pibar_row_max": None,
        },
        log_p_s=torch.zeros(2, dtype=theta.dtype),
        log_p_d=torch.zeros(2, dtype=theta.dtype),
        log_p_l=torch.zeros(2, dtype=theta.dtype),
        max_transfer=torch.zeros(2, dtype=theta.dtype),
    )

    monkeypatch.setattr(
        api_model,
        "evaluate_resident_gradient_forward",
        lambda *_args, **_kwargs: api_autograd.ResidentGradientForwardResult(
            solve=solve,
            loss_vec=torch.ones(2, dtype=theta.dtype),
        ),
    )
    monkeypatch.setattr(
        api_model,
        "compute_resident_implicit_gradient",
        lambda *_args, **_kwargs: torch.ones_like(theta),
    )

    api_model._evaluate_static_state(
        static,
        theta,
        need_grad=True,
        per_family=True,
    )

    assert static.warm_E is None
    assert static.pi_adjoint_cache is None
    assert static.pi_adjoint_pending_cache is None


def test_compute_resident_implicit_gradient_uses_pi_adjoint_cache_when_enabled(
    monkeypatch,
):
    theta = torch.zeros((2, 3), dtype=torch.float64)
    pi = torch.ones((2, 2), dtype=torch.float64)
    cached = torch.full_like(pi, 0.25)
    solved = torch.full_like(pi, 0.75)
    expected_grad = torch.arange(theta.numel(), dtype=theta.dtype).reshape_as(theta)
    static = _implicit_gradient_static(
        pi_adjoint_warmstart=True,
        pi_adjoint_cache=cached,
    )
    calls: list[dict[str, object]] = []

    def fake_implicit_grad(*args, **kwargs):
        calls.append(kwargs)
        assert kwargs["return_aux"] is True
        assert kwargs["record_pi_adjoint_residual"] is True
        assert kwargs["pi_fixed_point_relaxation"] == pytest.approx(1.0)
        torch.testing.assert_close(kwargs["pi_adjoint_initial_guess"], cached)
        return (
            expected_grad,
            SimpleNamespace(
                iters=3,
                rel_res=0.125,
                success=True,
                neumann_terms=4,
                pi_adjoint_residual_absmax=0.0625,
                pi_adjoint_residual_relmax=0.03125,
                pi_adjoint_residual_wave_count=2,
            ),
            {
                "pi_adjoint": solved,
                "used_pi_initial_guess": kwargs["pi_adjoint_initial_guess"] is not None,
            },
        )

    monkeypatch.setattr(
        api_autograd,
        "implicit_grad_loglik_vjp_wave",
        fake_implicit_grad,
    )

    grad = api_autograd.compute_resident_implicit_gradient(
        static,
        theta=theta,
        pi_wave_ordered=pi,
        pibar_wave_ordered=pi + 1.0,
        e=torch.zeros((2, 2), dtype=theta.dtype),
        ebar=torch.zeros((2, 2), dtype=theta.dtype),
        e_s1=torch.zeros((2, 2), dtype=theta.dtype),
        e_s2=torch.zeros((2, 2), dtype=theta.dtype),
        log_p_s=torch.zeros(2, dtype=theta.dtype),
        log_p_d=torch.zeros(2, dtype=theta.dtype),
        log_p_l=torch.zeros(2, dtype=theta.dtype),
        max_transfer=torch.zeros(2, dtype=theta.dtype),
        uniform_pibar_row_max=None,
    )

    torch.testing.assert_close(grad, expected_grad)
    assert len(calls) == 1
    torch.testing.assert_close(static.pi_adjoint_cache, solved)
    assert static.last_solver_stats == {
        "Neumann_terms": 4,
        "E_adjoint_iterations": 3,
        "E_adjoint_rel_res": 0.125,
        "E_adjoint_success": True,
        "Pi_adjoint_residual_absmax": 0.0625,
        "Pi_adjoint_residual_relmax": 0.03125,
        "Pi_adjoint_residual_wave_count": 2,
        "Pi_adjoint_warmstart_enabled": True,
        "Pi_adjoint_warmstart_used": True,
        "Pi_adjoint_cache_update_mode": "immediate",
        "Pi_adjoint_cache_pending": False,
        "Pi_adjoint_cache_rows": 2,
        "Pi_adjoint_cache_species": 2,
    }


def test_compute_resident_implicit_gradient_stages_pi_adjoint_cache_for_commit(
    monkeypatch,
):
    theta = torch.zeros((2, 3), dtype=torch.float64)
    pi = torch.ones((2, 2), dtype=torch.float64)
    cached = torch.full_like(pi, 0.25)
    solved = torch.full_like(pi, 0.75)
    static = _implicit_gradient_static(
        pi_adjoint_warmstart=True,
        pi_adjoint_cache=cached,
        pi_adjoint_cache_update_mode="stage",
    )

    def fake_implicit_grad(*args, **kwargs):
        assert kwargs["return_aux"] is True
        torch.testing.assert_close(kwargs["pi_adjoint_initial_guess"], cached)
        return (
            torch.ones_like(theta),
            SimpleNamespace(iters=3, rel_res=0.125, success=True, neumann_terms=4),
            {"pi_adjoint": solved, "used_pi_initial_guess": True},
        )

    monkeypatch.setattr(
        api_autograd,
        "implicit_grad_loglik_vjp_wave",
        fake_implicit_grad,
    )

    api_autograd.compute_resident_implicit_gradient(
        static,
        theta=theta,
        pi_wave_ordered=pi,
        pibar_wave_ordered=pi + 1.0,
        e=torch.zeros((2, 2), dtype=theta.dtype),
        ebar=torch.zeros((2, 2), dtype=theta.dtype),
        e_s1=torch.zeros((2, 2), dtype=theta.dtype),
        e_s2=torch.zeros((2, 2), dtype=theta.dtype),
        log_p_s=torch.zeros(2, dtype=theta.dtype),
        log_p_d=torch.zeros(2, dtype=theta.dtype),
        log_p_l=torch.zeros(2, dtype=theta.dtype),
        max_transfer=torch.zeros(2, dtype=theta.dtype),
        uniform_pibar_row_max=None,
    )

    torch.testing.assert_close(static.pi_adjoint_cache, cached)
    torch.testing.assert_close(static.pi_adjoint_pending_cache, solved)
    assert static.last_solver_stats["Pi_adjoint_cache_update_mode"] == "stage"
    assert static.last_solver_stats["Pi_adjoint_cache_pending"] is True

    assert api_autograd._commit_pi_adjoint_pending_cache(static) is True
    torch.testing.assert_close(static.pi_adjoint_cache, solved)
    assert static.pi_adjoint_pending_cache is None


def test_compute_resident_implicit_gradient_forwards_pi_fixed_point_relaxation(
    monkeypatch,
):
    theta = torch.zeros((2, 3), dtype=torch.float64)
    pi = torch.ones((2, 2), dtype=torch.float64)
    solved = torch.full_like(pi, 0.75)
    static = _implicit_gradient_static(
        pi_adjoint_warmstart=True,
        pi_adjoint_cache=torch.full_like(pi, 0.25),
        pi_fixed_point_relaxation=1.25,
    )

    def fake_implicit_grad(*args, **kwargs):
        assert kwargs["pi_fixed_point_relaxation"] == pytest.approx(1.25)
        return (
            torch.ones_like(theta),
            SimpleNamespace(iters=3, rel_res=0.125, success=True, neumann_terms=4),
            {"pi_adjoint": solved, "used_pi_initial_guess": True},
        )

    monkeypatch.setattr(
        api_autograd,
        "implicit_grad_loglik_vjp_wave",
        fake_implicit_grad,
    )

    api_autograd.compute_resident_implicit_gradient(
        static,
        theta=theta,
        pi_wave_ordered=pi,
        pibar_wave_ordered=pi + 1.0,
        e=torch.zeros((2, 2), dtype=theta.dtype),
        ebar=torch.zeros((2, 2), dtype=theta.dtype),
        e_s1=torch.zeros((2, 2), dtype=theta.dtype),
        e_s2=torch.zeros((2, 2), dtype=theta.dtype),
        log_p_s=torch.zeros(2, dtype=theta.dtype),
        log_p_d=torch.zeros(2, dtype=theta.dtype),
        log_p_l=torch.zeros(2, dtype=theta.dtype),
        max_transfer=torch.zeros(2, dtype=theta.dtype),
        uniform_pibar_row_max=None,
    )


def test_discard_pi_adjoint_pending_cache_keeps_accepted_cache():
    cached = torch.full((2, 2), 0.25, dtype=torch.float64)
    pending = torch.full((2, 2), 0.75, dtype=torch.float64)
    static = _implicit_gradient_static(
        pi_adjoint_warmstart=True,
        pi_adjoint_cache=cached,
        pi_adjoint_cache_update_mode="stage",
    )
    static.pi_adjoint_pending_cache = pending

    api_autograd._discard_pi_adjoint_pending_cache(static)

    torch.testing.assert_close(static.pi_adjoint_cache, cached)
    assert static.pi_adjoint_pending_cache is None


def test_compute_resident_implicit_gradient_drops_stale_pi_adjoint_cache(
    monkeypatch,
):
    theta = torch.zeros((2, 3), dtype=torch.float64)
    pi = torch.ones((2, 2), dtype=torch.float64)
    stale_cache = torch.ones((1, 2), dtype=torch.float64)
    solved = torch.full_like(pi, 2.0)
    static = _implicit_gradient_static(
        pi_adjoint_warmstart=True,
        pi_adjoint_cache=stale_cache,
    )

    def fake_implicit_grad(*args, **kwargs):
        assert kwargs["return_aux"] is True
        assert kwargs["pi_adjoint_initial_guess"] is None
        return (
            torch.ones_like(theta),
            SimpleNamespace(iters=0, rel_res=0.0, success=True, neumann_terms=4),
            {"pi_adjoint": solved, "used_pi_initial_guess": False},
        )

    monkeypatch.setattr(
        api_autograd,
        "implicit_grad_loglik_vjp_wave",
        fake_implicit_grad,
    )

    api_autograd.compute_resident_implicit_gradient(
        static,
        theta=theta,
        pi_wave_ordered=pi,
        pibar_wave_ordered=pi + 1.0,
        e=torch.zeros((2, 2), dtype=theta.dtype),
        ebar=torch.zeros((2, 2), dtype=theta.dtype),
        e_s1=torch.zeros((2, 2), dtype=theta.dtype),
        e_s2=torch.zeros((2, 2), dtype=theta.dtype),
        log_p_s=torch.zeros(2, dtype=theta.dtype),
        log_p_d=torch.zeros(2, dtype=theta.dtype),
        log_p_l=torch.zeros(2, dtype=theta.dtype),
        max_transfer=torch.zeros(2, dtype=theta.dtype),
        uniform_pibar_row_max=None,
    )

    torch.testing.assert_close(static.pi_adjoint_cache, solved)
    assert static.last_solver_stats["Pi_adjoint_warmstart_used"] is False


def test_autograd_forward_uses_resident_solve_boundary(monkeypatch):
    theta = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True)
    warm_E = torch.full((2, 2), 0.25, dtype=torch.float64)
    static = SimpleNamespace(
        device=theta.device,
        dtype=theta.dtype,
        fixed_iters_E=None,
        clear_runtime_after_backward=False,
        genewise=False,
        warm_E=warm_E,
        wave_layout={"root_clade_ids": torch.tensor([0, 1])},
        origination_probs=None,
        fixed_iters_Pi=2,
        last_solver_stats=None,
    )
    e = torch.ones((2, 2), dtype=theta.dtype)
    e_out = {
        "E": e,
        "E_s1": e + 1.0,
        "E_s2": e + 2.0,
        "E_bar": e + 3.0,
        "iterations": 3,
        "E_convergence_delta": 0.0,
    }
    pi_out = {
        "Pi_wave_ordered": torch.ones((2, 2), dtype=theta.dtype),
        "Pibar_wave_ordered": torch.ones((2, 2), dtype=theta.dtype) * 2.0,
        "uniform_pibar_row_max": torch.arange(2, dtype=theta.dtype),
        "Pi_max_iterations": 2,
        "Pi_wave_iterations": [2],
        "Pi_wave_converged": [True],
    }
    solve_calls: list[dict[str, object]] = []
    likelihood_calls: list[dict[str, object]] = []

    def fake_solve_resident_e_pi(
        static_arg,
        theta_arg,
        *,
        pi_request,
        warm_start_E=None,
    ):
        solve_calls.append(
            {
                "static": static_arg,
                "theta": theta_arg,
                "pi_request": pi_request,
                "warm_start_E": warm_start_E,
                "grad_enabled": torch.is_grad_enabled(),
            }
        )
        return api_autograd.ResidentSolveResult(
            theta=theta_arg.detach(),
            e_out=e_out,
            pi_out=pi_out,
            log_p_s=torch.zeros(2, dtype=theta.dtype),
            log_p_d=torch.zeros(2, dtype=theta.dtype),
            log_p_l=torch.zeros(2, dtype=theta.dtype),
            max_transfer=torch.zeros(2, dtype=theta.dtype),
        )

    def fake_compute_nll(
        pi,
        e_arg,
        root_clade_ids,
        origination_probs,
        *,
        origination_probs_prepared,
    ):
        likelihood_calls.append(
            {
                "pi": pi,
                "e": e_arg,
                "root_clade_ids": root_clade_ids,
                "origination_probs": origination_probs,
                "origination_probs_prepared": origination_probs_prepared,
            }
        )
        return torch.tensor([1.25, 2.5], dtype=theta.dtype)

    monkeypatch.setattr(
        api_autograd,
        "solve_resident_e_pi",
        fake_solve_resident_e_pi,
    )
    monkeypatch.setattr(api_autograd, "compute_nll", fake_compute_nll)

    actual = api_autograd._GeneReconFunction.apply(theta, static, "sum")

    torch.testing.assert_close(actual.detach(), torch.tensor(3.75, dtype=theta.dtype))
    assert len(solve_calls) == 1
    assert solve_calls[0]["static"] is static
    assert solve_calls[0]["theta"] is theta
    assert (
        solve_calls[0]["pi_request"].intent.name
        == "training_or_wave_export_state"
    )
    assert solve_calls[0]["warm_start_E"] is warm_E
    assert solve_calls[0]["grad_enabled"] is False
    assert len(likelihood_calls) == 1
    assert likelihood_calls[0]["pi"] is pi_out["Pi_wave_ordered"]
    assert likelihood_calls[0]["e"] is e
    assert likelihood_calls[0]["root_clade_ids"] is static.wave_layout["root_clade_ids"]
    assert likelihood_calls[0]["origination_probs"] is None
    assert likelihood_calls[0]["origination_probs_prepared"] is True
    assert static.warm_E is not e
    torch.testing.assert_close(static.warm_E, e)
    assert static.last_solver_stats == {
        "E_iterations": 3,
        "E_convergence_delta": 0.0,
        "Pi_max_iterations": 2,
        "Pi_wave_iterations": [2],
        "Pi_converged_waves": 1,
        "Pi_wave_count": 1,
    }
