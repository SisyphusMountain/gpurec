# tests/test_group_index.py
import torch
from gpurec.optim.penalties import group_expand, group_reduce


def test_none_is_identity():
    t = torch.randn(4, 3, dtype=torch.float64)
    assert torch.equal(group_expand(t, None), t)
    g = torch.randn(4, 3, dtype=torch.float64)
    assert torch.equal(group_reduce(g, None, None), g)


def test_expand_then_reduce_roundtrip_gradient():
    gidx = torch.tensor([0, 0, 1, 1])                  # two groups over 4 species
    theta_g = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
    expanded = group_expand(theta_g, gidx)             # [4,3]
    assert expanded.shape == (4, 3)
    assert torch.equal(expanded[0], expanded[1])
    loss = (expanded ** 2).sum()
    loss.backward()
    per_species = 2 * expanded.detach()
    reduced = group_reduce(per_species, gidx, n_groups=2)
    assert torch.allclose(reduced, theta_g.grad, atol=1e-9)


def test_reduce_member_counts():
    gidx = torch.tensor([0, 0, 1])
    grad_s = torch.ones(3, 3, dtype=torch.float64)
    grad_g = group_reduce(grad_s, gidx, 2)
    assert grad_g.shape == (2, 3)
    assert torch.allclose(grad_g[0], torch.full((3,), 2.0, dtype=torch.float64))
    assert torch.allclose(grad_g[1], torch.full((3,), 1.0, dtype=torch.float64))
