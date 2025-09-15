#!/usr/bin/env python3
import math
import torch
import pytest

from src.optimization.fixed_point_autograd import FixedPointProblem, fixed_point_nll


@pytest.mark.parametrize("case", [
    "tests/data/test_trees_1",
    "tests/data/test_trees_2",
    "tests/data/test_trees_3",
])
def test_backward_runs_and_grad_is_finite(case):
    device = torch.device("cpu")
    dtype = torch.float64

    sp = f"{case}/sp.nwk"
    g = f"{case}/g.nwk"

    problem = FixedPointProblem(sp, g, device=device, dtype=dtype)
    theta0 = torch.tensor([-3.0, -3.0, -3.0], dtype=dtype)
    problem.init_from_theta(theta0)

    theta = theta0.clone().detach().requires_grad_(True)
    nlll = fixed_point_nll(theta, problem, e_max_iters=200, pi_max_iters=200, e_tol=1e-12, pi_tol=1e-12)
    nlll.backward()

    assert theta.grad is not None
    assert torch.isfinite(theta.grad).all()


def test_implicit_grad_matches_finite_difference():
    device = torch.device("cpu")
    dtype = torch.float64

    case = "tests/data/test_trees_1"
    sp = f"{case}/sp.nwk"
    g = f"{case}/g.nwk"

    problem = FixedPointProblem(sp, g, device=device, dtype=dtype)
    theta0 = torch.tensor([-3.0, -3.0, -3.0], dtype=dtype)
    problem.init_from_theta(theta0)

    # Autograd implicit gradient
    theta = theta0.clone().detach().requires_grad_(True)
    nlll = fixed_point_nll(theta, problem, e_max_iters=300, pi_max_iters=300, e_tol=1e-12, pi_tol=1e-12)
    nlll.backward()
    grad_impl = theta.grad.detach().clone()

    # Finite differences (forward) on negative log-likelihood
    def eval_nlll(th):
        pr = FixedPointProblem(sp, g, device=device, dtype=dtype)
        pr.init_from_theta(th)
        return float(pr.nlll())

    eps = 1e-5
    base = theta0
    f0 = eval_nlll(base)
    grad_fd = []
    for i in range(3):
        th = base.clone()
        th[i] += eps
        fi = eval_nlll(th)
        grad_fd.append((fi - f0) / eps)
    grad_fd = torch.tensor(grad_fd, dtype=dtype)

    diff = torch.abs(grad_impl - grad_fd)
    assert torch.all(diff < 5e-4), f"Implicit gradient {grad_impl} vs FD {grad_fd}, diff {diff}"


def test_nll_decreases_with_small_gradient_step():
    device = torch.device("cpu")
    dtype = torch.float64

    case = "tests/data/test_trees_1"
    sp = f"{case}/sp.nwk"
    g = f"{case}/g.nwk"

    problem = FixedPointProblem(sp, g, device=device, dtype=dtype)
    theta0 = torch.tensor([-3.0, -3.0, -3.0], dtype=dtype)
    problem.init_from_theta(theta0)

    # Compute gradient
    theta = theta0.clone().detach().requires_grad_(True)
    nlll = fixed_point_nll(theta, problem, e_max_iters=200, pi_max_iters=200, e_tol=1e-12, pi_tol=1e-12)
    nlll.backward()
    grad = theta.grad.detach().clone()

    # Take a small descent step on NLL
    lr = 0.05
    theta1 = theta0 - lr * grad

    # Evaluate NLL at both
    pr0 = FixedPointProblem(sp, g, device=device, dtype=dtype)
    pr0.init_from_theta(theta0)
    nlll0 = pr0.nlll().item()

    pr1 = FixedPointProblem(sp, g, device=device, dtype=dtype)
    pr1.init_from_theta(theta1)
    nlll1 = pr1.nlll().item()

    assert nlll1 <= nlll0 + 1e-6, f"NLL did not decrease: {nlll0} -> {nlll1}"

