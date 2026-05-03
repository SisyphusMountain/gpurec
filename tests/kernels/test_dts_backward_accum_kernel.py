"""Parity tests for direct DTS backward accumulation variants."""

import pytest
import torch

from gpurec.core.kernels.wave_backward import (
    dts_cross_backward_accum_fused,
    dts_cross_backward_fused,
    uniform_cross_pibar_vjp_tree_fused,
    uniform_cross_pibar_vjp_tree_from_ud_fused,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _scalar_param(value, *, device, dtype, shape):
    param = torch.tensor(value, device=device, dtype=dtype)
    return param.reshape(1) if shape == "1d" else param


def _binary_tree_helpers(S, device, dtype):
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
    ancestors_dense = torch.zeros((S, S), dtype=dtype)
    for s in range(S):
        ancestors = []
        cur = s
        while cur >= 0:
            ancestors.append(cur)
            cur = parent[cur]
        ancestor_lists.append(ancestors)
        max_depth = max(max_depth, len(ancestors))
        ancestors_dense[s, ancestors] = 1.0

    ancestor_cols = torch.full((S, max_depth), -1, dtype=torch.long)
    for s, ancestors in enumerate(ancestor_lists):
        ancestor_cols[s, :len(ancestors)] = torch.tensor(ancestors, dtype=torch.long)

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
            s for s, lvl in enumerate(levels)
            if lvl == level and (child1[s] < S or child2[s] < S)
        ]
        if parents:
            level_lists.append(parents)
            max_width = max(max_width, len(parents))
    level_parents = torch.full((max(len(level_lists), 1), max_width), -1, dtype=torch.long)
    for level, parents in enumerate(level_lists):
        level_parents[level, :len(parents)] = torch.tensor(parents, dtype=torch.long)

    return (
        ancestor_cols.T.contiguous().to(device),
        ancestors_dense.to(device),
        child1.to(device),
        child2.to(device),
        level_parents.to(device),
    )


def _uniform_pibar_from_pi(Pi, mt, ancestors_dense):
    row_max = Pi.max(dim=1).values
    p = torch.exp2(Pi - row_max[:, None])
    anc_sum = p @ ancestors_dense.T
    denom = p.sum(dim=1, keepdim=True) - anc_sum
    pibar = torch.where(
        denom > 0,
        torch.log2(denom) + row_max[:, None] + mt[None, :],
        torch.full_like(Pi, float("-inf")),
    )
    return pibar.contiguous(), row_max.contiguous()


def _run_accum_variant(*, merge_s_term, active_mask=None, dtype=torch.float32, scalar_shape="0d"):
    torch.manual_seed(17)
    device = torch.device("cuda")
    C, S, W, N = 9, 11, 3, 6
    ws = 6

    Pi = (torch.randn(C, S, device=device, dtype=dtype) * 0.2 - 2.0).contiguous()
    Pibar = (torch.randn(C, S, device=device, dtype=dtype) * 0.2 - 2.0).contiguous()
    v_k = (torch.randn(W, S, device=device, dtype=dtype) * 0.1).contiguous()

    # Duplicated child rows exercise the atomic accumulation path.
    sl = torch.tensor([0, 1, 2, 3, 1, 4], device=device, dtype=torch.long)
    sr = torch.tensor([1, 2, 3, 4, 0, 2], device=device, dtype=torch.long)
    reduce_idx = torch.tensor([0, 1, 1, 2, 2, 0], device=device, dtype=torch.long)
    wlsp = (torch.randn(N, device=device, dtype=dtype) * 0.1 - 1.0).contiguous()

    sp_child1 = torch.tensor(
        [1, 3, S, 5, S, 7, S, 9, S, S, S],
        device=device,
        dtype=torch.long,
    )
    sp_child2 = torch.tensor(
        [2, 4, S, 6, S, 8, S, 10, S, S, S],
        device=device,
        dtype=torch.long,
    )
    accumulated_rhs = torch.zeros(C, S, device=device, dtype=dtype)
    grad_pibar_l, grad_pibar_r, param_pD, param_pS = dts_cross_backward_accum_fused(
        Pi,
        Pibar,
        v_k,
        ws,
        sl,
        sr,
        reduce_idx,
        wlsp,
        _scalar_param(-4.0, device=device, dtype=dtype, shape=scalar_shape),
        _scalar_param(-5.0, device=device, dtype=dtype, shape=scalar_shape),
        sp_child1,
        sp_child2,
        accumulated_rhs,
        S,
        active_mask=active_mask,
        merge_s_term=merge_s_term,
    )
    torch.cuda.synchronize()
    return accumulated_rhs, grad_pibar_l, grad_pibar_r, param_pD, param_pS


def _run_accum_reduction_variant(
    *,
    merge_s_term,
    active_mask=None,
    dtype=torch.float32,
    accum_param_reductions=False,
    accum_mt_reduction=False,
):
    torch.manual_seed(17)
    device = torch.device("cuda")
    C, S, W, N = 9, 11, 3, 6
    ws = 6

    Pi = (torch.randn(C, S, device=device, dtype=dtype) * 0.2 - 2.0).contiguous()
    Pibar = (torch.randn(C, S, device=device, dtype=dtype) * 0.2 - 2.0).contiguous()
    v_k = (torch.randn(W, S, device=device, dtype=dtype) * 0.1).contiguous()
    sl = torch.tensor([0, 1, 2, 3, 1, 4], device=device, dtype=torch.long)
    sr = torch.tensor([1, 2, 3, 4, 0, 2], device=device, dtype=torch.long)
    reduce_idx = torch.tensor([0, 1, 1, 2, 2, 0], device=device, dtype=torch.long)
    wlsp = (torch.randn(N, device=device, dtype=dtype) * 0.1 - 1.0).contiguous()
    sp_child1 = torch.tensor(
        [1, 3, S, 5, S, 7, S, 9, S, S, S],
        device=device,
        dtype=torch.long,
    )
    sp_child2 = torch.tensor(
        [2, 4, S, 6, S, 8, S, 10, S, S, S],
        device=device,
        dtype=torch.long,
    )
    accumulated_rhs = torch.zeros(C, S, device=device, dtype=dtype)
    grad_log_pD = torch.zeros(1, device=device, dtype=dtype)
    grad_log_pS = torch.zeros(1, device=device, dtype=dtype)
    grad_mt = torch.zeros(S, device=device, dtype=dtype)

    grad_pibar_l, grad_pibar_r, param_pD, param_pS = dts_cross_backward_accum_fused(
        Pi,
        Pibar,
        v_k,
        ws,
        sl,
        sr,
        reduce_idx,
        wlsp,
        _scalar_param(-4.0, device=device, dtype=dtype, shape="1d"),
        _scalar_param(-5.0, device=device, dtype=dtype, shape="1d"),
        sp_child1,
        sp_child2,
        accumulated_rhs,
        S,
        active_mask=active_mask,
        merge_s_term=merge_s_term,
        grad_log_pD=grad_log_pD,
        grad_log_pS=grad_log_pS,
        grad_mt=grad_mt,
        accum_param_reductions=accum_param_reductions,
        accum_mt_reduction=accum_mt_reduction,
    )
    torch.cuda.synchronize()
    return (
        accumulated_rhs,
        grad_pibar_l,
        grad_pibar_r,
        param_pD,
        param_pS,
        grad_log_pD,
        grad_log_pS,
        grad_mt,
    )


def _run_nonaccum_variant(*, active_mask=None, dtype=torch.float32, scalar_shape="0d"):
    torch.manual_seed(23)
    device = torch.device("cuda")
    C, S, W, N = 9, 11, 3, 6
    ws = 6

    Pi = (torch.randn(C, S, device=device, dtype=dtype) * 0.2 - 2.0).contiguous()
    Pibar = (torch.randn(C, S, device=device, dtype=dtype) * 0.2 - 2.0).contiguous()
    v_k = (torch.randn(W, S, device=device, dtype=dtype) * 0.1).contiguous()
    sl = torch.tensor([0, 1, 2, 3, 1, 4], device=device, dtype=torch.long)
    sr = torch.tensor([1, 2, 3, 4, 0, 2], device=device, dtype=torch.long)
    reduce_idx = torch.tensor([0, 1, 1, 2, 2, 0], device=device, dtype=torch.long)
    wlsp = (torch.randn(N, device=device, dtype=dtype) * 0.1 - 1.0).contiguous()
    sp_child1 = torch.tensor(
        [1, 3, S, 5, S, 7, S, 9, S, S, S],
        device=device,
        dtype=torch.long,
    )
    sp_child2 = torch.tensor(
        [2, 4, S, 6, S, 8, S, 10, S, S, S],
        device=device,
        dtype=torch.long,
    )

    outputs = dts_cross_backward_fused(
        Pi,
        Pibar,
        v_k,
        ws,
        sl,
        sr,
        reduce_idx,
        wlsp,
        _scalar_param(-4.0, device=device, dtype=dtype, shape=scalar_shape),
        _scalar_param(-5.0, device=device, dtype=dtype, shape=scalar_shape),
        sp_child1,
        sp_child2,
        S,
        active_mask=active_mask,
    )
    torch.cuda.synchronize()
    return outputs


@pytest.mark.parametrize("dtype,atol,rtol", [(torch.float32, 2e-5, 2e-5), (torch.float64, 1e-10, 1e-10)])
def test_dts_backward_accum_merged_matches_two_loop(dtype, atol, rtol):
    old = _run_accum_variant(merge_s_term=False, dtype=dtype)
    merged = _run_accum_variant(merge_s_term=True, dtype=dtype)

    for actual, expected in zip(merged, old):
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


def test_dts_backward_accum_merged_matches_two_loop_with_active_mask():
    device = torch.device("cuda")
    active_mask = torch.tensor([True, False, True], device=device)

    old = _run_accum_variant(merge_s_term=False, active_mask=active_mask)
    merged = _run_accum_variant(merge_s_term=True, active_mask=active_mask)

    for actual, expected in zip(merged, old):
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("dtype,atol,rtol", [(torch.float32, 2e-5, 2e-5), (torch.float64, 1e-10, 1e-10)])
def test_dts_backward_accum_device_scalar_matches_python_scalar_fallback(monkeypatch, dtype, atol, rtol):
    monkeypatch.setenv("GPUREC_DTS_BACKWARD_DEVICE_SCALARS", "0")
    fallback = _run_accum_variant(merge_s_term=True, dtype=dtype, scalar_shape="1d")

    monkeypatch.setenv("GPUREC_DTS_BACKWARD_DEVICE_SCALARS", "1")
    device_scalar = _run_accum_variant(merge_s_term=True, dtype=dtype, scalar_shape="1d")

    for actual, expected in zip(device_scalar, fallback):
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype,atol,rtol", [(torch.float32, 2e-5, 2e-5), (torch.float64, 1e-10, 1e-10)])
def test_dts_backward_nonaccum_device_scalar_matches_python_scalar_fallback(monkeypatch, dtype, atol, rtol):
    device = torch.device("cuda")
    active_mask = torch.tensor([True, False, True], device=device)

    monkeypatch.setenv("GPUREC_DTS_BACKWARD_DEVICE_SCALARS", "0")
    fallback = _run_nonaccum_variant(active_mask=active_mask, dtype=dtype, scalar_shape="1d")

    monkeypatch.setenv("GPUREC_DTS_BACKWARD_DEVICE_SCALARS", "1")
    device_scalar = _run_nonaccum_variant(active_mask=active_mask, dtype=dtype, scalar_shape="1d")

    for actual, expected in zip(device_scalar, fallback):
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype,atol,rtol", [(torch.float32, 3e-5, 3e-5), (torch.float64, 1e-10, 1e-10)])
@pytest.mark.parametrize("active", [False, True])
@pytest.mark.parametrize("accum_mt_reduction", [False, True])
def test_dts_backward_accum_reduction_targets_match_materialized_reductions(
    dtype, atol, rtol, active, accum_mt_reduction
):
    device = torch.device("cuda")
    active_mask = torch.tensor([True, False, True], device=device) if active else None

    base = _run_accum_variant(
        merge_s_term=True,
        active_mask=active_mask,
        dtype=dtype,
        scalar_shape="1d",
    )
    reduced = _run_accum_reduction_variant(
        merge_s_term=True,
        active_mask=active_mask,
        dtype=dtype,
        accum_param_reductions=True,
        accum_mt_reduction=accum_mt_reduction,
    )

    torch.testing.assert_close(reduced[0], base[0], atol=atol, rtol=rtol)
    torch.testing.assert_close(reduced[1], base[1], atol=atol, rtol=rtol)
    torch.testing.assert_close(reduced[2], base[2], atol=atol, rtol=rtol)
    assert reduced[3] is None
    assert reduced[4] is None
    torch.testing.assert_close(reduced[5], base[3].sum().reshape(1), atol=atol, rtol=rtol)
    torch.testing.assert_close(reduced[6], base[4].sum().reshape(1), atol=atol, rtol=rtol)

    if accum_mt_reduction:
        expected_mt = base[1].sum(dim=0) + base[2].sum(dim=0)
        torch.testing.assert_close(reduced[7], expected_mt, atol=atol, rtol=rtol)
    else:
        torch.testing.assert_close(reduced[7], torch.zeros_like(reduced[7]), atol=0, rtol=0)


@pytest.mark.parametrize("dtype,atol,rtol", [(torch.float32, 4e-5, 4e-5), (torch.float64, 1e-9, 1e-10)])
@pytest.mark.parametrize("active", [False, True])
def test_dts_staged_pibar_ud_matches_materialized_pibar_vjp(dtype, atol, rtol, active):
    torch.manual_seed(29)
    device = torch.device("cuda")
    C, S, W, N = 10, 31, 3, 7
    ws = 7

    ancestor_cols, ancestors_dense, sp_child1, sp_child2, level_parents = _binary_tree_helpers(
        S, device, dtype
    )
    Pi = (torch.randn(C, S, device=device, dtype=dtype) * 0.2 - 2.0).contiguous()
    mt = (torch.randn(S, device=device, dtype=dtype) * 0.01 - 4.0).contiguous()
    Pibar, row_max = _uniform_pibar_from_pi(Pi, mt, ancestors_dense)
    v_k = (torch.randn(W, S, device=device, dtype=dtype) * 0.1).contiguous()

    sl = torch.tensor([0, 1, 2, 3, 1, 4, 5], device=device, dtype=torch.long)
    sr = torch.tensor([1, 2, 3, 4, 0, 2, 6], device=device, dtype=torch.long)
    reduce_idx = torch.tensor([0, 1, 1, 2, 2, 0, 1], device=device, dtype=torch.long)
    wlsp = (torch.randn(N, device=device, dtype=dtype) * 0.1 - 1.0).contiguous()
    active_mask = torch.tensor([True, False, True], device=device) if active else None

    base_rhs = torch.zeros(C, S, device=device, dtype=dtype)
    grad_pibar_l, grad_pibar_r, param_pD, param_pS = dts_cross_backward_accum_fused(
        Pi,
        Pibar,
        v_k,
        ws,
        sl,
        sr,
        reduce_idx,
        wlsp,
        _scalar_param(-4.0, device=device, dtype=dtype, shape="1d"),
        _scalar_param(-5.0, device=device, dtype=dtype, shape="1d"),
        sp_child1,
        sp_child2,
        base_rhs,
        S,
        active_mask=active_mask,
        merge_s_term=True,
    )
    uniform_cross_pibar_vjp_tree_fused(
        Pi,
        grad_pibar_l,
        grad_pibar_r,
        sl,
        sr,
        ancestor_cols,
        sp_child1,
        sp_child2,
        level_parents,
        base_rhs,
        S,
        active_mask=active_mask,
        reduce_idx=reduce_idx,
        Pibar_star=Pibar,
        mt_squeezed=mt,
        pibar_row_max=row_max,
    )

    staged_rhs = torch.zeros(C, S, device=device, dtype=dtype)
    grad_mt = torch.zeros(S, device=device, dtype=dtype)
    pibar_ud, pibar_A, staged_pD, staged_pS = dts_cross_backward_accum_fused(
        Pi,
        Pibar,
        v_k,
        ws,
        sl,
        sr,
        reduce_idx,
        wlsp,
        _scalar_param(-4.0, device=device, dtype=dtype, shape="1d"),
        _scalar_param(-5.0, device=device, dtype=dtype, shape="1d"),
        sp_child1,
        sp_child2,
        staged_rhs,
        S,
        active_mask=active_mask,
        merge_s_term=True,
        grad_mt=grad_mt,
        accum_mt_reduction=True,
        output_pibar_ud=True,
        mt_squeezed=mt,
        pibar_row_max=row_max,
    )
    uniform_cross_pibar_vjp_tree_from_ud_fused(
        Pi,
        pibar_ud,
        pibar_A,
        sl,
        sr,
        sp_child1,
        sp_child2,
        level_parents,
        staged_rhs,
        S,
        active_mask=active_mask,
        reduce_idx=reduce_idx,
        pibar_row_max=row_max,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(staged_rhs, base_rhs, atol=atol, rtol=rtol)
    torch.testing.assert_close(staged_pD, param_pD, atol=atol, rtol=rtol)
    torch.testing.assert_close(staged_pS, param_pS, atol=atol, rtol=rtol)
    torch.testing.assert_close(grad_mt, grad_pibar_l.sum(dim=0) + grad_pibar_r.sum(dim=0), atol=atol, rtol=rtol)
