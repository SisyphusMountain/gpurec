"""Parity tests for the fused uniform-Pibar cross-DTS VJP kernel."""

import pytest
import torch

from gpurec.core.kernels.wave_backward import (
    uniform_cross_pibar_vjp_fused,
    uniform_cross_pibar_vjp_tree_fused,
    uniform_cross_pibar_vjp_tree_from_ud_fused,
)
from gpurec.core.species_euler_layout import reference_subtree_sums


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _ancestor_cols_for_binary_tree(S, device, dtype=torch.float32):
    parent = [-1] + [(i - 1) // 2 for i in range(1, S)]
    child1 = torch.full((S,), S, dtype=torch.long)
    child2 = torch.full((S,), S, dtype=torch.long)
    for child, p in enumerate(parent):
        if p < 0:
            continue
        if child1[p] == S:
            child1[p] = child
        else:
            child2[p] = child

    ancestor_lists = []
    max_depth = 0
    for s in range(S):
        ancestors = []
        cur = s
        while cur >= 0:
            ancestors.append(cur)
            cur = parent[cur]
        ancestor_lists.append(ancestors)
        max_depth = max(max_depth, len(ancestors))

    cols_cpu = torch.full((S, max_depth), -1, dtype=torch.long)
    dense_cpu = torch.zeros((S, S), dtype=dtype)
    for s, ancestors in enumerate(ancestor_lists):
        cols_cpu[s, :len(ancestors)] = torch.tensor(ancestors, dtype=torch.long)
        dense_cpu[s, ancestors] = 1.0

    levels = [-1] * S

    def node_level(s):
        if levels[s] >= 0:
            return levels[s]
        level = 0
        c1 = int(child1[s])
        c2 = int(child2[s])
        if c1 < S:
            level = max(level, node_level(c1) + 1)
        if c2 < S:
            level = max(level, node_level(c2) + 1)
        levels[s] = level
        return level

    for s in range(S):
        node_level(s)
    max_level = max(levels)
    level_lists = []
    max_width = 1
    for level in range(1, max_level + 1):
        parents = [
            s for s, node_level in enumerate(levels)
            if node_level == level and (child1[s] < S or child2[s] < S)
        ]
        if parents:
            level_lists.append(parents)
            max_width = max(max_width, len(parents))
    level_parents = torch.full((max(len(level_lists), 1), max_width), -1, dtype=torch.long)
    for level, parents in enumerate(level_lists):
        level_parents[level, :len(parents)] = torch.tensor(parents, dtype=torch.long)

    return (
        cols_cpu.T.contiguous().to(device),
        dense_cpu.to(device),
        child1.to(device),
        child2.to(device),
        level_parents.to(device),
    )


@pytest.mark.parametrize("dtype,rtol,atol", [
    (torch.float32, 2e-5, 2e-4),
    (torch.float64, 1e-9, 1e-10),
])
def test_uniform_cross_pibar_vjp_matches_sparse_reference(dtype, rtol, atol):
    torch.manual_seed(0)
    device = torch.device("cuda")
    S = 257
    C = 8
    n_ws = 5

    Pi = (torch.randn(C, S, device=device, dtype=dtype) * 2.0).contiguous()
    grad_l = torch.randn(n_ws, S, device=device, dtype=dtype)
    grad_r = torch.randn(n_ws, S, device=device, dtype=dtype)

    # Deliberately include duplicate child clades to exercise atomic adds into
    # accumulated_rhs, matching index_add_ semantics from the PyTorch path.
    sl = torch.tensor([0, 1, 1, 3, 6], device=device, dtype=torch.long)
    sr = torch.tensor([4, 4, 5, 7, 2], device=device, dtype=torch.long)

    ancestor_cols, ancestors_dense, child1, child2, level_parents = _ancestor_cols_for_binary_tree(
        S, device, dtype=dtype
    )
    ancestors_T = ancestors_dense.T.to_sparse_coo()

    base = torch.randn(C, S, device=device, dtype=dtype)
    actual = base.clone()
    uniform_cross_pibar_vjp_fused(Pi, grad_l, grad_r, sl, sr, ancestor_cols, actual, S)
    actual_tree = base.clone()
    uniform_cross_pibar_vjp_tree_fused(
        Pi, grad_l, grad_r, sl, sr, ancestor_cols, child1, child2, level_parents, actual_tree, S
    )

    children = torch.cat([sl, sr])
    u = torch.cat([grad_l, grad_r])
    Pi_ch = Pi[children]
    Pi_max = Pi_ch.max(dim=1, keepdim=True).values
    p_prime = torch.exp2(Pi_ch - Pi_max)
    anc_sum = p_prime @ ancestors_T
    denom = p_prime.sum(dim=1, keepdim=True) - anc_sum
    denom_safe = torch.where(denom > 0, denom, torch.ones_like(denom))
    u_d = torch.where(denom > 0, u / denom_safe, torch.zeros_like(u))
    correction = (ancestors_T @ u_d.T).T
    expected_update = p_prime * (u_d.sum(dim=1, keepdim=True) - correction)
    expected = base.clone()
    expected.index_add_(0, children, expected_update)

    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    torch.testing.assert_close(actual_tree, expected, rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype,rtol,atol", [
    (torch.float32, 2e-5, 2e-4),
    (torch.float64, 1e-9, 1e-10),
])
def test_uniform_cross_pibar_from_ud_euler_prefix_matches_tree_reference(
    monkeypatch,
    dtype,
    rtol,
    atol,
):
    torch.manual_seed(1)
    monkeypatch.setenv("GPUREC_DTS_PIBAR_UD_EULER_PREFIX", "1")
    device = torch.device("cuda")
    S = 7
    C = 6
    n_ws = 4

    # Current preprocessing uses species postorder, so subtree intervals are
    # current-order ranges: leaves [s, s + 1), internal nodes [start, s + 1).
    child1 = torch.tensor([S, S, 0, S, S, 3, 2], device=device, dtype=torch.long)
    child2 = torch.tensor([S, S, 1, S, S, 4, 5], device=device, dtype=torch.long)
    subtree_start = torch.tensor([0, 1, 0, 3, 4, 3, 0], device=device, dtype=torch.int32)
    subtree_end = torch.tensor([1, 2, 3, 4, 5, 6, 7], device=device, dtype=torch.int32)

    Pi = (torch.randn(C, S, device=device, dtype=dtype) * 1.5).contiguous()
    pibar_ud = torch.randn(2 * n_ws, S, device=device, dtype=dtype)
    pibar_A = pibar_ud.sum(dim=1).contiguous()
    pibar_row_max = Pi.max(dim=1).values.contiguous()
    sl = torch.tensor([0, 1, 1, 3], device=device, dtype=torch.long)
    sr = torch.tensor([4, 4, 5, 2], device=device, dtype=torch.long)

    base = torch.randn(C, S, device=device, dtype=dtype)
    actual = base.clone()
    uniform_cross_pibar_vjp_tree_from_ud_fused(
        Pi,
        pibar_ud.clone(),
        pibar_A,
        sl,
        sr,
        child1,
        child2,
        torch.empty((1, 1), device=device, dtype=torch.long),
        actual,
        S,
        pibar_row_max=pibar_row_max,
        subtree_interval_start=subtree_start,
        subtree_interval_end=subtree_end,
    )

    children = torch.cat([sl, sr])
    p_prime = torch.exp2(Pi[children] - pibar_row_max[children].unsqueeze(1))
    subtree = reference_subtree_sums(pibar_ud, child1, child2)
    expected_update = p_prime * (pibar_A.unsqueeze(1) - subtree)
    expected = base.clone()
    expected.index_add_(0, children, expected_update)

    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
