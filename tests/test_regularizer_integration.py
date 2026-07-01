"""Integration tests for the TV prior / origination penalty / grouped-theta wiring
in ``gpurec.optim.value_and_grad.make_value_and_grad``.

This file holds the CPU-checkable portion (no Triton/CUDA forward solve required).
The end-to-end no-op-equals-baseline GPU validation is added in Task 7.
"""
import torch
import pytest


def test_group_index_theta_shape_contract():
    from gpurec.optim.penalties import group_expand, group_reduce
    gidx = torch.tensor([0, 0, 1])           # 3 species, 2 categories
    theta_g = torch.randn(2, 3, dtype=torch.float64)
    theta_s = group_expand(theta_g, gidx)
    assert theta_s.shape == (3, 3)
    grad_s = torch.ones(3, 3, dtype=torch.float64)
    grad_g = group_reduce(grad_s, gidx, 2)
    assert grad_g.shape == (2, 3)
    assert torch.allclose(grad_g[0], torch.full((3,), 2.0, dtype=torch.float64))
    assert torch.allclose(grad_g[1], torch.full((3,), 1.0, dtype=torch.float64))
