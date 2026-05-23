"""Tests for the row-wise batched L-BFGS optimizer."""
from __future__ import annotations

import math

import pytest
import torch
from torch.optim.lbfgs import _strong_wolfe as torch_strong_wolfe

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


def test_batched_lbfgs_strong_wolfe_matches_pytorch_per_row_search():
    dtype = torch.float64
    theta0 = torch.tensor(
        [[2.5, -1.0], [-2.0, 3.0], [0.5, -3.5]],
        dtype=dtype,
    )
    target = torch.tensor(
        [[-0.25, 0.75], [1.5, -0.5], [-1.0, 2.0]],
        dtype=dtype,
    )
    scale = torch.tensor([0.5, 3.0, 20.0], dtype=dtype)
    theta = torch.nn.Parameter(theta0.clone())
    opt = BatchedLBFGS(
        [theta],
        lr=1.0,
        max_iter=1,
        history_size=3,
        tolerance_grad=1e-12,
        tolerance_change=1e-12,
        max_ls=25,
        line_search_fn="strong_wolfe",
    )

    def row_loss_grad(row: int, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        diff = value - target[row]
        loss = scale[row] * (diff.pow(4).sum() + 0.25 * diff.square().sum())
        grad = scale[row] * (4.0 * diff.pow(3) + 0.5 * diff)
        return loss, grad

    def loss_vec() -> torch.Tensor:
        losses = [row_loss_grad(row, theta[row])[0] for row in range(theta.shape[0])]
        return torch.stack(losses)

    def closure() -> torch.Tensor:
        opt.zero_grad(set_to_none=True)
        loss = loss_vec()
        loss.sum().backward()
        return loss

    start_loss = loss_vec().detach()
    start_grad = torch.stack(
        [row_loss_grad(row, theta0[row])[1] for row in range(theta0.shape[0])]
    )
    direction = -start_grad
    alpha0 = torch.minimum(
        torch.ones(theta0.shape[0], dtype=dtype),
        1.0 / start_grad.abs().sum(dim=1).clamp_min(1e-30),
    )
    expected_alpha = []
    expected_loss = []
    expected_grad = []
    for row in range(theta0.shape[0]):
        x = theta0[row].clone()
        d = direction[row].clone()

        def obj_func(x_init, step, direction_i):
            return row_loss_grad(row, x_init + step * direction_i)

        loss_i, grad_i, alpha_i, _evals = torch_strong_wolfe(
            obj_func,
            x,
            alpha0[row],
            d,
            start_loss[row],
            start_grad[row],
            start_grad[row].dot(d),
            tolerance_change=1e-12,
            max_ls=25,
        )
        expected_alpha.append(alpha_i.reshape(()))
        expected_loss.append(loss_i.reshape(()))
        expected_grad.append(grad_i)

    final = opt.step(closure)
    state = opt.state[theta]
    expected_alpha_t = torch.stack(expected_alpha)
    expected_loss_t = torch.stack(expected_loss)
    expected_grad_t = torch.stack(expected_grad)
    expected_theta = theta0 + expected_alpha_t[:, None] * direction

    torch.testing.assert_close(state["last_alpha"], expected_alpha_t)
    torch.testing.assert_close(final, expected_loss_t)
    torch.testing.assert_close(state["last_grad"], expected_grad_t)
    torch.testing.assert_close(theta.detach(), expected_theta)


def test_batched_lbfgs_respects_max_eval_after_line_search():
    theta = torch.nn.Parameter(torch.tensor([[2.0]], dtype=torch.float64))
    opt = BatchedLBFGS(
        [theta],
        lr=1.0,
        max_iter=5,
        max_eval=2,
        history_size=3,
        tolerance_grad=1e-12,
        tolerance_change=1e-14,
    )
    calls = {"grad": 0, "loss": 0}

    def loss_vec() -> torch.Tensor:
        return (theta ** 2).sum(dim=1)

    def closure() -> torch.Tensor:
        calls["grad"] += 1
        opt.zero_grad(set_to_none=True)
        loss = loss_vec()
        loss.sum().backward()
        return loss

    def loss_closure() -> torch.Tensor:
        calls["loss"] += 1
        return loss_vec()

    final = opt.step(closure, loss_closure=loss_closure)
    state = opt.state[theta]

    assert state["func_evals"] == 2
    assert calls == {"grad": 1, "loss": 1}
    torch.testing.assert_close(final, torch.tensor([1.0], dtype=torch.float64))
    assert torch.equal(theta.detach(), torch.tensor([[1.0]], dtype=torch.float64))
    assert state["old_dirs"] == []
    assert state["old_stps"] == []


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


def test_batched_lbfgs_rejects_unknown_line_search():
    theta = torch.nn.Parameter(torch.zeros(2, 1))
    with pytest.raises(ValueError, match="line_search_fn"):
        BatchedLBFGS([theta], line_search_fn="zoom")
