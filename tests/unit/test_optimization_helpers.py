from __future__ import annotations

import math

import pytest
import torch
from torch.optim import lbfgs as torch_lbfgs

from gpurec.optimization._armijo import armijo_accepts, armijo_required_decrease
from gpurec.optimization._bounds import (
    bound_for_flat,
    bounds_for_flat,
    project_flat,
)
from gpurec.optimization._closures import (
    evaluate_scalar_loss,
    flat_grad,
    loss_vector_tensor,
    scalar_loss_tensor,
)
from gpurec.optimization._line_search_interpolation import _cubic_interpolate


def _quadratic_line_samples(
    x1: torch.Tensor,
    x2: torch.Tensor,
    center: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    f1 = (x1 - center).square()
    g1 = 2 * (x1 - center)
    f2 = (x2 - center).square()
    g2 = 2 * (x2 - center)
    return f1, g1, f2, g2


def test_cubic_interpolate_matches_torch_scalar_rows():
    x1 = torch.tensor([0.0, 2.0, -1.0], dtype=torch.float64)
    x2 = torch.tensor([1.0, -1.0, 2.0], dtype=torch.float64)
    center = torch.tensor([0.25, 0.5, 0.75], dtype=torch.float64)
    f1, g1, f2, g2 = _quadratic_line_samples(x1, x2, center)

    actual = _cubic_interpolate(x1, f1, g1, x2, f2, g2)
    expected = torch.stack(
        [
            torch_lbfgs._cubic_interpolate(
                x1[idx],
                f1[idx],
                g1[idx],
                x2[idx],
                f2[idx],
                g2[idx],
            )
            for idx in range(x1.numel())
        ]
    )

    torch.testing.assert_close(actual, expected)


def test_cubic_interpolate_clamps_tensor_bounds():
    x1 = torch.tensor([0.0, 0.0], dtype=torch.float64)
    x2 = torch.tensor([1.0, 1.0], dtype=torch.float64)
    center = torch.tensor([2.0, -1.0], dtype=torch.float64)
    f1, g1, f2, g2 = _quadratic_line_samples(x1, x2, center)
    lower = torch.tensor([0.2, 0.2], dtype=torch.float64)
    upper = torch.tensor([0.6, 0.6], dtype=torch.float64)

    actual = _cubic_interpolate(x1, f1, g1, x2, f2, g2, bounds=(lower, upper))

    torch.testing.assert_close(actual, torch.tensor([0.6, 0.2], dtype=torch.float64))


def test_cubic_interpolate_degenerate_or_nonfinite_falls_back_to_midpoint():
    x1 = torch.tensor([1.0, 0.0], dtype=torch.float64)
    x2 = torch.tensor([1.0, 2.0], dtype=torch.float64)
    f1 = torch.tensor([0.0, float("nan")], dtype=torch.float64)
    g1 = torch.tensor([1.0, 1.0], dtype=torch.float64)
    f2 = torch.tensor([2.0, 3.0], dtype=torch.float64)
    g2 = torch.tensor([1.0, 1.0], dtype=torch.float64)
    lower = torch.tensor([0.0, 1.0], dtype=torch.float64)
    upper = torch.tensor([2.0, 3.0], dtype=torch.float64)

    actual = _cubic_interpolate(x1, f1, g1, x2, f2, g2, bounds=(lower, upper))

    torch.testing.assert_close(actual, torch.tensor([1.0, 2.0], dtype=torch.float64))
    assert bool(torch.all(actual >= lower))
    assert bool(torch.all(actual <= upper))


def test_project_flat_handles_scalar_and_tensor_bounds():
    flat = torch.tensor([-2.0, 0.5, 4.0])
    upper = torch.tensor([1.0, 2.0, 3.0])

    projected = project_flat(flat, 0.0, upper, flat.shape)

    torch.testing.assert_close(projected, torch.tensor([0.0, 0.5, 3.0]))


def test_bound_for_flat_accepts_original_shape_bounds():
    flat = torch.zeros(6, dtype=torch.float64)
    bound = torch.arange(6, dtype=torch.float64).reshape(2, 3)

    actual = bound_for_flat(bound, flat, (2, 3))

    torch.testing.assert_close(actual, bound.reshape(6))


def test_bound_for_flat_accepts_flat_shape_bounds():
    flat = torch.zeros(6, dtype=torch.float64)
    bound = torch.arange(6, dtype=torch.float64)

    actual = bound_for_flat(bound, flat, (2, 3))

    torch.testing.assert_close(actual, bound)


def test_bound_for_flat_can_broadcast_to_batched_flat_shape():
    flat = torch.zeros(2, 6, dtype=torch.float64)
    bound = torch.arange(6, dtype=torch.float64).reshape(1, 6)

    actual = bound_for_flat(
        bound,
        flat,
        (2, 2, 3),
        broadcast_to_flat=True,
    )

    torch.testing.assert_close(actual, bound.expand_as(flat))


def test_bound_for_flat_param_only_mode_rejects_flat_only_broadcast():
    flat = torch.zeros(2, 6, dtype=torch.float64)
    bound = torch.arange(6, dtype=torch.float64).reshape(1, 6)

    with pytest.raises(RuntimeError):
        bound_for_flat(bound, flat, (2, 2, 3), broadcast_to_flat=False)


def test_bounds_for_flat_rejects_lower_greater_than_upper():
    flat = torch.zeros(2)

    with pytest.raises(ValueError, match="lower_bound must be <= upper_bound"):
        bounds_for_flat(
            flat,
            torch.tensor([0.0, 3.0]),
            torch.tensor([1.0, 2.0]),
            flat.shape,
        )


def test_scalar_loss_tensor_validates_shape_and_owner():
    assert scalar_loss_tensor(torch.tensor([3.0]), "ProjectedLBFGS").shape == ()

    with pytest.raises(
        ValueError,
        match="ProjectedLBFGS closure must return a scalar Tensor",
    ):
        scalar_loss_tensor(torch.ones(2), "ProjectedLBFGS")


def test_evaluate_scalar_loss_uses_loss_closure_message():
    def closure() -> torch.Tensor:
        return torch.tensor(1.0)

    def loss_closure() -> torch.Tensor:
        return torch.ones(2)

    with pytest.raises(
        ValueError,
        match="LBFGSB loss closure must return a scalar Tensor",
    ):
        evaluate_scalar_loss(closure, loss_closure, "LBFGSB")


def test_evaluate_scalar_loss_runs_loss_closure_without_grad():
    grad_enabled = None

    def closure() -> torch.Tensor:
        raise AssertionError("loss_closure should be used")

    def loss_closure() -> torch.Tensor:
        nonlocal grad_enabled
        grad_enabled = torch.is_grad_enabled()
        return torch.tensor(1.0)

    loss = evaluate_scalar_loss(closure, loss_closure, "ProjectedLBFGS")

    assert loss.shape == ()
    assert grad_enabled is False


def test_loss_vector_tensor_validates_shape_and_owner():
    loss = torch.arange(3.0).reshape(3, 1)

    torch.testing.assert_close(
        loss_vector_tensor(loss, 3, "BatchedLBFGS"),
        torch.arange(3.0),
    )

    with pytest.raises(ValueError, match="one loss per parameter row"):
        loss_vector_tensor(torch.ones(2), 3, "BatchedLBFGS")


def test_flat_grad_densifies_sparse_gradient():
    param = torch.nn.Parameter(torch.zeros(3))
    param.grad = torch.tensor([1.0, 0.0, 3.0]).to_sparse()

    grad = flat_grad(param, torch.zeros(3), "LBFGSB")

    assert not grad.is_sparse
    torch.testing.assert_close(grad, torch.tensor([1.0, 0.0, 3.0]))


def test_flat_grad_returns_batched_zeros_when_gradient_is_missing():
    param = torch.nn.Parameter(torch.zeros(2, 3))

    grad = flat_grad(
        param,
        torch.empty(2, 3),
        "BatchedLBFGS",
        row_batch_size=2,
    )

    torch.testing.assert_close(grad, torch.zeros(2, 3))


def test_flat_grad_rejects_complex_gradient():
    param = torch.nn.Parameter(torch.zeros(2, dtype=torch.complex64))
    param.grad = torch.tensor([1.0 + 2.0j, 3.0 + 0.0j])

    with pytest.raises(
        TypeError,
        match="ProjectedLBFGS only supports real-valued gradients",
    ):
        flat_grad(param, torch.zeros(2, dtype=torch.complex64), "ProjectedLBFGS")


def test_armijo_required_decrease_and_acceptance():
    loss = torch.tensor(1.0, dtype=torch.float64)
    trial_gtd = torch.tensor(-2.0, dtype=torch.float64)

    assert armijo_required_decrease(loss, trial_gtd, 0.1) == pytest.approx(0.2)
    assert armijo_accepts(torch.tensor(0.8, dtype=torch.float64), loss, trial_gtd, 0.1)
    assert not armijo_accepts(
        torch.tensor(math.nextafter(0.8, math.inf), dtype=torch.float64),
        loss,
        trial_gtd,
        0.1,
    )
    assert not armijo_accepts(
        torch.tensor(float("nan"), dtype=torch.float64),
        loss,
        trial_gtd,
        0.1,
    )
