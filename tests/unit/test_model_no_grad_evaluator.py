from __future__ import annotations

from types import SimpleNamespace

import torch

import gpurec.api.autograd as api_autograd
import gpurec.api.model as api_model


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
    model._ensure_batch_static = lambda batch_idx: statics[batch_idx]
    e_solve = object()
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
    ):
        calls.append(
            {
                "static": static_arg,
                "e_solve": e_solve_arg,
                "scratch_tensors": scratch_tensors,
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
