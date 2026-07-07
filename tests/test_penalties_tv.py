# tests/test_penalties_tv.py
import torch
from gpurec.solver.penalties import tv_prior_and_grad


def _tiny_tree():
    # 3 species: node 2 is root, 0 and 1 are its children.
    sp_parent = torch.tensor([2, 2, -1], dtype=torch.int32)
    return sp_parent


def test_tv_zero_lambda_is_noop():
    theta = torch.randn(3, 3, dtype=torch.float64)
    pen, grad = tv_prior_and_grad(theta, _tiny_tree(), lam=0.0)
    assert float(pen) == 0.0
    assert torch.count_nonzero(grad) == 0


def test_tv_gradient_matches_finite_difference():
    torch.manual_seed(0)
    theta = torch.randn(3, 3, dtype=torch.float64)
    sp_parent = _tiny_tree()
    lam, eps = 0.7, 1e-3

    def value(t):
        pen, _ = tv_prior_and_grad(t, sp_parent, lam=lam, eps=eps)
        return pen

    _, grad = tv_prior_and_grad(theta, sp_parent, lam=lam, eps=eps)
    fd = torch.zeros_like(theta)
    h = 1e-6
    for i in range(3):
        for j in range(3):
            tp = theta.clone(); tp[i, j] += h
            tm = theta.clone(); tm[i, j] -= h
            fd[i, j] = (value(tp) - value(tm)) / (2 * h)
    assert torch.allclose(grad, fd, atol=1e-5, rtol=1e-4)


def test_tv_penalty_positive_and_gradient_conserved():
    theta = torch.zeros(3, 3, dtype=torch.float64)
    theta[0] = 5.0  # child 0 differs sharply from root
    pen, grad = tv_prior_and_grad(theta, _tiny_tree(), lam=1.0, eps=1e-3)
    assert float(pen) > 0.0
    # Each edge contributes +g to the child row and -g to the parent row, so the
    # per-column gradient sums to zero (conservation). The root is excluded only as
    # a *difference term* (it has no parent edge of its own) but still RECEIVES the
    # negative of its children's edge gradients, so its row is NOT zero.
    assert torch.allclose(grad.sum(dim=0), torch.zeros(3, dtype=torch.float64), atol=1e-12)
    assert not torch.allclose(grad[2], torch.zeros(3, dtype=torch.float64))
