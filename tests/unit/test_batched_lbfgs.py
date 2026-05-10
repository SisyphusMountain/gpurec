"""Tests for the row-wise batched L-BFGS optimizer."""
from __future__ import annotations

import math

import pytest
import torch

from gpurec.optimization import BatchedLBFGS


def test_batched_lbfgs_converges_scaled_independent_quadratics():
    dtype = torch.float64
    theta = torch.nn.Parameter(
        torch.tensor(
            [[4.0, -3.0], [-2.0, 5.0], [0.5, -4.0]],
            dtype=dtype,
        )
    )
    target = torch.tensor(
        [[1.0, 2.0], [3.0, -1.0], [-2.0, 1.0]],
        dtype=dtype,
    )
    scale = torch.tensor([1e-3, 1.0, 1e3], dtype=dtype)

    opt = BatchedLBFGS(
        [theta],
        lr=1.0,
        max_iter=3,
        history_size=5,
        tolerance_grad=1e-10,
        tolerance_change=1e-12,
    )

    def loss_vec() -> torch.Tensor:
        return scale * ((theta - target) ** 2).sum(dim=1)

    def closure() -> torch.Tensor:
        opt.zero_grad(set_to_none=True)
        loss = loss_vec()
        loss.sum().backward()
        return loss

    start = loss_vec().detach().clone()
    for _ in range(4):
        final = opt.step(closure, loss_closure=loss_vec)

    torch.testing.assert_close(theta.detach(), target, rtol=0.0, atol=1e-8)
    assert torch.all(final < start * 1e-12)


def test_batched_lbfgs_uses_rowwise_line_search_not_global_loss():
    dtype = torch.float64
    theta = torch.nn.Parameter(
        torch.tensor([[2.0], [2.0]], dtype=dtype)
    )
    target = torch.tensor([[1.0], [-1.0]], dtype=dtype)
    scale = torch.tensor([1e-3, 1e6], dtype=dtype)

    opt = BatchedLBFGS(
        [theta],
        lr=1.0,
        max_iter=2,
        history_size=3,
        tolerance_grad=1e-12,
        tolerance_change=1e-14,
    )

    def loss_vec() -> torch.Tensor:
        return scale * ((theta - target) ** 2).sum(dim=1)

    def closure() -> torch.Tensor:
        opt.zero_grad(set_to_none=True)
        loss = loss_vec()
        loss.sum().backward()
        return loss

    start = loss_vec().detach().clone()
    final = opt.step(closure, loss_closure=loss_vec)

    assert torch.all(final < start)
    assert abs(float(theta.detach()[0, 0]) - 1.0) < 1e-6
    assert abs(float(theta.detach()[1, 0]) + 1.0) < 1e-6


def test_batched_lbfgs_strong_wolfe_converges_scaled_independent_quadratics():
    dtype = torch.float64
    theta = torch.nn.Parameter(
        torch.tensor(
            [[4.0, -3.0], [-2.0, 5.0], [0.5, -4.0]],
            dtype=dtype,
        )
    )
    target = torch.tensor(
        [[1.0, 2.0], [3.0, -1.0], [-2.0, 1.0]],
        dtype=dtype,
    )
    scale = torch.tensor([1e-3, 1.0, 1e3], dtype=dtype)

    opt = BatchedLBFGS(
        [theta],
        lr=1.0,
        max_iter=3,
        history_size=5,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-10,
        tolerance_change=1e-12,
    )

    def loss_vec() -> torch.Tensor:
        return scale * ((theta - target) ** 2).sum(dim=1)

    def closure() -> torch.Tensor:
        opt.zero_grad(set_to_none=True)
        loss = loss_vec()
        loss.sum().backward()
        return loss

    start = loss_vec().detach().clone()
    for _ in range(4):
        final = opt.step(closure)

    torch.testing.assert_close(theta.detach(), target, rtol=0.0, atol=1e-8)
    assert torch.all(final < start * 1e-12)


def test_batched_lbfgs_strong_wolfe_satisfies_rowwise_conditions():
    dtype = torch.float64
    theta0 = torch.tensor([[3.0], [-2.0], [0.25]], dtype=dtype)
    target = torch.tensor([[1.0], [2.0], [-1.5]], dtype=dtype)
    scale = torch.tensor([0.25, 10.0, 100.0], dtype=dtype)
    theta = torch.nn.Parameter(theta0.clone())
    c1 = 1e-4
    c2 = 0.9

    opt = BatchedLBFGS(
        [theta],
        lr=1.0,
        max_iter=1,
        history_size=3,
        line_search_fn="strong_wolfe",
        c1=c1,
        c2=c2,
    )

    def loss_vec() -> torch.Tensor:
        return scale * ((theta - target) ** 2).sum(dim=1)

    def closure() -> torch.Tensor:
        opt.zero_grad(set_to_none=True)
        loss = loss_vec()
        loss.sum().backward()
        return loss

    start_loss = scale * ((theta0 - target) ** 2).sum(dim=1)
    start_grad = 2.0 * scale[:, None] * (theta0 - target)
    direction = -start_grad
    gtd0 = (start_grad * direction).sum(dim=1)

    final = opt.step(closure)
    state = opt.state[theta]
    alpha = state["last_alpha"]
    accepted = state["last_accepted"]
    assert torch.all(accepted)
    assert torch.all(state["last_wolfe_satisfied"])

    final_grad = 2.0 * scale[:, None] * (theta.detach() - target)
    final_gtd = (final_grad * direction).sum(dim=1)
    armijo_rhs = start_loss + c1 * alpha * gtd0

    assert torch.all(final <= armijo_rhs + 1e-12)
    assert torch.all(final_gtd.abs() <= -c2 * gtd0 + 1e-12)


def test_batched_lbfgs_strong_wolfe_does_not_use_loss_only_closure():
    theta = torch.nn.Parameter(torch.tensor([[2.0], [-2.0]], dtype=torch.float64))
    target = torch.tensor([[1.0], [3.0]], dtype=torch.float64)
    opt = BatchedLBFGS([theta], line_search_fn="strong_wolfe", max_iter=2)

    def loss_vec() -> torch.Tensor:
        return ((theta - target) ** 2).sum(dim=1)

    def closure() -> torch.Tensor:
        opt.zero_grad(set_to_none=True)
        loss = loss_vec()
        loss.sum().backward()
        return loss

    def loss_only() -> torch.Tensor:
        raise AssertionError("strong Wolfe requires gradients at trial points")

    opt.step(closure, loss_closure=loss_only)
    assert torch.all(loss_vec().detach() < torch.tensor([1.0, 25.0], dtype=torch.float64))


def test_batched_lbfgs_projects_to_lower_bound():
    theta = torch.nn.Parameter(torch.tensor([[0.0], [0.0]], dtype=torch.float64))
    target = torch.tensor([[-10.0], [2.0]], dtype=torch.float64)
    lower_bound = -1.0
    opt = BatchedLBFGS(
        [theta],
        lr=1.0,
        max_iter=3,
        history_size=3,
        lower_bound=lower_bound,
    )

    def loss_vec() -> torch.Tensor:
        return ((theta - target) ** 2).sum(dim=1)

    def closure() -> torch.Tensor:
        opt.zero_grad(set_to_none=True)
        loss = loss_vec()
        loss.sum().backward()
        return loss

    for _ in range(3):
        opt.step(closure, loss_closure=loss_vec)

    assert torch.all(theta.detach() >= lower_bound)
    assert math.isclose(float(theta.detach()[0, 0]), lower_bound, abs_tol=1e-12)
    assert math.isclose(float(theta.detach()[1, 0]), 2.0, abs_tol=1e-8)


def test_batched_lbfgs_projects_to_upper_bound():
    theta = torch.nn.Parameter(torch.tensor([[0.0], [0.0]], dtype=torch.float64))
    target = torch.tensor([[10.0], [-2.0]], dtype=torch.float64)
    upper_bound = 1.0
    opt = BatchedLBFGS(
        [theta],
        lr=1.0,
        max_iter=3,
        history_size=3,
        upper_bound=upper_bound,
    )

    def loss_vec() -> torch.Tensor:
        return ((theta - target) ** 2).sum(dim=1)

    def closure() -> torch.Tensor:
        opt.zero_grad(set_to_none=True)
        loss = loss_vec()
        loss.sum().backward()
        return loss

    for _ in range(3):
        opt.step(closure, loss_closure=loss_vec)

    assert torch.all(theta.detach() <= upper_bound)
    assert math.isclose(float(theta.detach()[0, 0]), upper_bound, abs_tol=1e-12)
    assert math.isclose(float(theta.detach()[1, 0]), -2.0, abs_tol=1e-8)


def test_batched_lbfgs_rejects_scalar_loss():
    theta = torch.nn.Parameter(torch.zeros(2, 1))
    opt = BatchedLBFGS([theta])

    def closure() -> torch.Tensor:
        opt.zero_grad(set_to_none=True)
        loss = (theta ** 2).sum()
        loss.backward()
        return loss

    with pytest.raises(ValueError, match="one loss per parameter row"):
        opt.step(closure)
