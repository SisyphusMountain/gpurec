from __future__ import annotations

import torch

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
