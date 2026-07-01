# tests/test_penalties_origination.py
import math
import torch
from gpurec.optim.penalties import (
    OriginationPenalty, origination_penalty_and_grad, origination_log_pO,
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
