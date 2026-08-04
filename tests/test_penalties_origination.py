# tests/test_penalties_origination.py
import math
import torch
from gpurec.solver.penalties import (
    OriginationPenalty, origination_penalty_and_grad, origination_penalty_hvp, origination_log_pO,
)

S = 6


def _omega():
    torch.manual_seed(1)
    w = torch.randn(S, dtype=torch.float64)
    return w - w.mean()  # gauge slice (mean 0), as first_order maintains


def _fd_grad(omega, cfg):
    h = 1e-6
    g = torch.zeros_like(omega)
    for i in range(omega.numel()):
        wp = omega.clone(); wp[i] += h
        wm = omega.clone(); wm[i] -= h
        pp, _ = origination_penalty_and_grad(wp, cfg)
        pm, _ = origination_penalty_and_grad(wm, cfg)
        g[i] = (pp - pm) / (2 * h)
    return g


def _fd_hvp(omega, cfg, v):
    """FD directional derivative of the (already FD-verified) gradient along v: (grad(w+h v) -
    grad(w-h v)) / 2h -- exact to O(h^2) for a Hessian-vector product."""
    h = 1e-6
    _, gp = origination_penalty_and_grad(omega + h * v, cfg)
    _, gm = origination_penalty_and_grad(omega - h * v, cfg)
    return (gp - gm) / (2 * h)


def test_all_disabled_is_noop():
    w = _omega()
    pen, grad = origination_penalty_and_grad(w, OriginationPenalty())
    assert float(pen) == 0.0
    assert torch.count_nonzero(grad) == 0


def test_l2_matches_formula_and_fd():
    w = _omega()
    cfg = OriginationPenalty(l2=0.3)
    pen, grad = origination_penalty_and_grad(w, cfg)
    assert math.isclose(float(pen), 0.3 * float((w * w).sum()), rel_tol=1e-9)
    assert torch.allclose(grad, _fd_grad(w, cfg), atol=1e-5)


def test_depth_prior_gradient_matches_fd():
    w = _omega()
    depth = torch.arange(S, dtype=torch.float64)
    cfg = OriginationPenalty(depth_lambda=0.5, depth=depth)
    _, grad = origination_penalty_and_grad(w, cfg)
    assert torch.allclose(grad, _fd_grad(w, cfg), atol=1e-5)


def test_root_prior_gradient_matches_fd():
    w = _omega()
    cfg = OriginationPenalty(root_lambda=0.4, root_index=2)
    _, grad = origination_penalty_and_grad(w, cfg)
    assert torch.allclose(grad, _fd_grad(w, cfg), atol=1e-5)


def test_dirichlet_barriers_gradient_match_fd():
    w = _omega()
    for kind in ("meanlog", "simpson", "renyi2"):
        pi = torch.full((S,), 1.0 / S, dtype=torch.float64)
        cfg = OriginationPenalty(dirichlet_c=0.6, barrier_kind=kind, dirichlet_pi=pi)
        _, grad = origination_penalty_and_grad(w, cfg)
        assert torch.allclose(grad, _fd_grad(w, cfg), atol=1e-5), kind


def test_floor_penalty_gradient_matches_fd():
    w = _omega()
    cfg = OriginationPenalty(dirichlet_c=0.6, barrier_kind="simpson",
                             dirichlet_pi=torch.full((S,), 1.0 / S, dtype=torch.float64), floor=0.1)
    _, grad = origination_penalty_and_grad(w, cfg)
    assert torch.allclose(grad, _fd_grad(w, cfg), atol=1e-5)


def test_log_pO_is_base2_softmax():
    w = _omega()
    lp = origination_log_pO(w)
    expect = w - torch.logsumexp(w * math.log(2), dim=0) / math.log(2)
    assert torch.allclose(lp, expect, atol=1e-9)


def test_hvp_all_disabled_is_zero():
    w = _omega()
    v = torch.randn(S, dtype=torch.float64)
    hv = origination_penalty_hvp(w, OriginationPenalty(), v)
    assert torch.count_nonzero(hv) == 0


def test_hvp_l2_matches_formula_and_fd():
    w = _omega()
    v = torch.randn(S, dtype=torch.float64)
    cfg = OriginationPenalty(l2=0.3)
    hv = origination_penalty_hvp(w, cfg, v)
    assert torch.allclose(hv, 2 * 0.3 * v, atol=1e-9)  # d^2/dw^2 (l2*sum(w^2)) = 2*l2*I
    assert torch.allclose(hv, _fd_hvp(w, cfg, v), atol=1e-4)


def test_hvp_depth_prior_matches_fd():
    w = _omega()
    v = torch.randn(S, dtype=torch.float64)
    depth = torch.arange(S, dtype=torch.float64)
    cfg = OriginationPenalty(depth_lambda=0.5, depth=depth)
    hv = origination_penalty_hvp(w, cfg, v)
    assert torch.allclose(hv, _fd_hvp(w, cfg, v), atol=1e-4)


def test_hvp_root_prior_matches_fd():
    w = _omega()
    v = torch.randn(S, dtype=torch.float64)
    cfg = OriginationPenalty(root_lambda=0.4, root_index=2)
    hv = origination_penalty_hvp(w, cfg, v)
    assert torch.allclose(hv, _fd_hvp(w, cfg, v), atol=1e-4)


def test_hvp_dirichlet_barriers_match_fd():
    w = _omega()
    v = torch.randn(S, dtype=torch.float64)
    for kind in ("meanlog", "simpson", "renyi2"):
        pi = torch.full((S,), 1.0 / S, dtype=torch.float64)
        cfg = OriginationPenalty(dirichlet_c=0.6, barrier_kind=kind, dirichlet_pi=pi)
        hv = origination_penalty_hvp(w, cfg, v)
        assert torch.allclose(hv, _fd_hvp(w, cfg, v), atol=1e-4), kind


def test_hvp_floor_penalty_matches_fd():
    w = _omega()
    v = torch.randn(S, dtype=torch.float64)
    cfg = OriginationPenalty(dirichlet_c=0.6, barrier_kind="simpson",
                             dirichlet_pi=torch.full((S,), 1.0 / S, dtype=torch.float64), floor=0.1)
    hv = origination_penalty_hvp(w, cfg, v)
    assert torch.allclose(hv, _fd_hvp(w, cfg, v), atol=1e-4)


def test_hvp_is_symmetric():
    """H must be symmetric: v1^T (H v2) == v2^T (H v1), for a generic active config."""
    w = _omega()
    torch.manual_seed(2)
    v1 = torch.randn(S, dtype=torch.float64)
    v2 = torch.randn(S, dtype=torch.float64)
    cfg = OriginationPenalty(l2=0.2, depth_lambda=0.3, depth=torch.arange(S, dtype=torch.float64),
                             root_lambda=0.1, root_index=0, dirichlet_c=0.4, barrier_kind="renyi2")
    lhs = float(torch.dot(v1, origination_penalty_hvp(w, cfg, v2)))
    rhs = float(torch.dot(v2, origination_penalty_hvp(w, cfg, v1)))
    assert math.isclose(lhs, rhs, rel_tol=1e-9, abs_tol=1e-9)
