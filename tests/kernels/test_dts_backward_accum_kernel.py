"""Parity tests for direct DTS backward accumulation variants."""

import pytest
import torch

from gpurec.core.kernels.wave_backward import (
    dts_cross_backward_accum_fused,
    dts_cross_backward_accum_parent_tiled_fused,
    dts_cross_backward_accum_parent_ragged_fused,
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


def _compact_level_helpers(level_parents, sp_child1, sp_child2):
    level_parents_cpu = level_parents.detach().cpu()
    child1_cpu = sp_child1.detach().cpu()
    child2_cpu = sp_child2.detach().cpu()
    ptr = [0]
    nodes = []
    child1 = []
    child2 = []
    for level in range(level_parents_cpu.shape[0]):
        valid = level_parents_cpu[level][level_parents_cpu[level] >= 0].tolist()
        nodes.extend(valid)
        child1.extend(int(child1_cpu[parent]) for parent in valid)
        child2.extend(int(child2_cpu[parent]) for parent in valid)
        ptr.append(len(nodes))
    compact_ptr = torch.tensor(ptr, device=level_parents.device, dtype=torch.long)
    compact_nodes = torch.tensor(nodes, device=level_parents.device, dtype=torch.int32)
    compact_child1 = torch.tensor(child1, device=level_parents.device, dtype=torch.int32)
    compact_child2 = torch.tensor(child2, device=level_parents.device, dtype=torch.int32)
    return compact_ptr, compact_nodes, compact_child1, compact_child2


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


def _run_accum_variant(
    *,
    merge_s_term,
    active_mask=None,
    dtype=torch.float32,
    scalar_shape="0d",
    index_dtype=torch.long,
):
    torch.manual_seed(17)
    device = torch.device("cuda")
    C, S, W, N = 9, 11, 3, 6
    ws = 6

    Pi = (torch.randn(C, S, device=device, dtype=dtype) * 0.2 - 2.0).contiguous()
    Pibar = (torch.randn(C, S, device=device, dtype=dtype) * 0.2 - 2.0).contiguous()
    v_k = (torch.randn(W, S, device=device, dtype=dtype) * 0.1).contiguous()

    # Duplicated child rows exercise the atomic accumulation path.
    sl = torch.tensor([0, 1, 2, 3, 1, 4], device=device, dtype=index_dtype)
    sr = torch.tensor([1, 2, 3, 4, 0, 2], device=device, dtype=index_dtype)
    reduce_idx = torch.tensor([0, 1, 1, 2, 2, 0], device=device, dtype=index_dtype)
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


def test_dts_backward_accum_accepts_int32_split_metadata():
    base = _run_accum_variant(merge_s_term=True, index_dtype=torch.long)
    actual = _run_accum_variant(merge_s_term=True, index_dtype=torch.int32)
    for lhs, rhs in zip(actual, base):
        torch.testing.assert_close(lhs, rhs, atol=2e-5, rtol=2e-5)


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


def _dts_layout_reference(
    *,
    Pi,
    Pibar,
    v_k,
    ws,
    sl,
    sr,
    reduce_idx,
    wlsp,
    log_pD,
    log_pS,
    sp_child1,
    sp_child2,
    family_idx,
    active_mask,
    mt_squeezed=None,
    row_max=None,
):
    device = Pi.device
    dtype = Pi.dtype
    C, S = Pi.shape
    n_ws = int(sl.numel())
    rhs = torch.zeros_like(Pi)
    grad_pibar_l = torch.zeros(n_ws, S, device=device, dtype=dtype)
    grad_pibar_r = torch.zeros_like(grad_pibar_l)
    pibar_ud = torch.zeros(2 * n_ws, S, device=device, dtype=dtype)
    pibar_A = torch.zeros(2 * n_ws, device=device, dtype=dtype)
    grad_log_pD = torch.zeros_like(log_pD)
    grad_log_pS = torch.zeros_like(log_pS)
    grad_mt = torch.zeros_like(mt_squeezed) if mt_squeezed is not None else None
    family_count = (
        int(family_idx.max().item()) + 1
        if family_idx is not None and int(family_idx.numel()) > 0
        else None
    )

    def _param_value(param, family, species):
        if family_count is not None and param.ndim == 1 and int(param.shape[0]) == family_count:
            return param[family]
        if param.ndim == 1 and int(param.shape[0]) == S:
            return param[species]
        if param.ndim == 1:
            return param[family]
        return param[family, species]

    def _add_param(grad, family, species, value):
        if family_count is not None and grad.ndim == 1 and int(grad.shape[0]) == family_count:
            grad[family] += value
            return
        if grad.ndim == 1 and int(grad.shape[0]) == S:
            grad[species] += value
        elif grad.ndim == 1:
            grad[family] += value
        else:
            grad[family, species] += value

    def _mt_value(mt, family, species):
        return mt[species] if mt.ndim == 1 else mt[family, species]

    def _add_mt(grad, family, species, value):
        if grad.ndim == 1:
            grad[species] += value
        else:
            grad[family, species] += value

    for i in range(n_ws):
        parent_w = int(reduce_idx[i])
        if active_mask is not None and not bool(active_mask[parent_w]):
            continue
        left = int(sl[i])
        right = int(sr[i])
        parent = ws + parent_w
        parent_family = int(family_idx[parent]) if family_idx is not None else 0
        left_family = int(family_idx[left]) if family_idx is not None else 0
        right_family = int(family_idx[right]) if family_idx is not None else 0
        for s in range(S):
            c1 = int(sp_child1[s])
            c2 = int(sp_child2[s])
            log_pD_s = _param_value(log_pD, parent_family, s)
            log_pS_s = _param_value(log_pS, parent_family, s)
            pi_parent = Pi[parent, s]
            if pi_parent <= -1e29:
                continue

            def _pi(row, col):
                return Pi[row, col] if col < S else torch.tensor(-1e30, device=device, dtype=dtype)

            d0 = log_pD_s + Pi[left, s] + Pi[right, s]
            d1 = Pi[left, s] + Pibar[right, s]
            d2 = Pi[right, s] + Pibar[left, s]
            d3 = log_pS_s + _pi(left, c1) + _pi(right, c2)
            d4 = log_pS_s + _pi(right, c1) + _pi(left, c2)
            scale = v_k[parent_w, s]
            vd0 = scale * torch.exp2(wlsp[i] + d0 - pi_parent)
            vd1 = scale * torch.exp2(wlsp[i] + d1 - pi_parent)
            vd2 = scale * torch.exp2(wlsp[i] + d2 - pi_parent)
            vd3 = scale * torch.exp2(wlsp[i] + d3 - pi_parent)
            vd4 = scale * torch.exp2(wlsp[i] + d4 - pi_parent)

            rhs[left, s] += vd0 + vd1
            rhs[right, s] += vd0 + vd2
            if c1 < S:
                rhs[left, c1] += vd3
                rhs[right, c1] += vd4
            if c2 < S:
                rhs[right, c2] += vd3
                rhs[left, c2] += vd4

            grad_pibar_l[i, s] = vd2
            grad_pibar_r[i, s] = vd1
            _add_param(grad_log_pD, parent_family, s, vd0)
            _add_param(grad_log_pS, parent_family, s, vd3 + vd4)
            if grad_mt is not None:
                _add_mt(grad_mt, left_family, s, vd2)
                _add_mt(grad_mt, right_family, s, vd1)
            if row_max is not None and mt_squeezed is not None:
                mt_l = _mt_value(mt_squeezed, left_family, s)
                mt_r = _mt_value(mt_squeezed, right_family, s)
                ud_l = vd2 * torch.exp2(row_max[left] + mt_l - Pibar[left, s])
                ud_r = vd1 * torch.exp2(row_max[right] + mt_r - Pibar[right, s])
                pibar_ud[i, s] = ud_l
                pibar_ud[n_ws + i, s] = ud_r
                pibar_A[i] += ud_l
                pibar_A[n_ws + i] += ud_r

    return rhs, grad_pibar_l, grad_pibar_r, grad_log_pD, grad_log_pS, grad_mt, pibar_ud, pibar_A


@pytest.mark.parametrize("layout", ["species", "family_scalar", "family_species"])
@pytest.mark.parametrize("output_pibar_ud", [False, True])
@pytest.mark.parametrize("active", [False, True])
@pytest.mark.parametrize("ambiguous_g_equals_s", [False, True])
def test_dts_backward_accum_fused_parameter_layouts(
    layout, output_pibar_ud, active, ambiguous_g_equals_s
):
    if ambiguous_g_equals_s and layout != "family_scalar":
        pytest.skip("G == S ambiguity only applies to 1D family-scalar layout")
    torch.manual_seed(41)
    device = torch.device("cuda")
    dtype = torch.float32
    C, S, W, N, G = (18, 13, 4, 7, 13) if ambiguous_g_equals_s else (10, 13, 4, 7, 3)
    ws = 6
    Pi = (torch.randn(C, S, device=device, dtype=dtype) * 0.2 - 2.0).contiguous()
    family_idx = (
        (torch.arange(C, device=device) % G).to(torch.long)
        if ambiguous_g_equals_s
        else torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0], device=device)
    )
    mt = (torch.randn((G, S), device=device, dtype=dtype) * 0.01 - 4.0).contiguous()
    if layout == "species":
        log_pD = (torch.randn(S, device=device, dtype=dtype) * 0.01 - 4.0).contiguous()
        log_pS = (torch.randn(S, device=device, dtype=dtype) * 0.01 - 5.0).contiguous()
        kernel_family_idx = None
        kernel_mt = mt[0].contiguous()
        ref_family_idx = None
    elif layout == "family_scalar":
        log_pD = (torch.randn(G, device=device, dtype=dtype) * 0.01 - 4.0).contiguous()
        log_pS = (torch.randn(G, device=device, dtype=dtype) * 0.01 - 5.0).contiguous()
        kernel_family_idx = family_idx
        kernel_mt = mt
        ref_family_idx = family_idx
    else:
        log_pD = (torch.randn(G, S, device=device, dtype=dtype) * 0.01 - 4.0).contiguous()
        log_pS = (torch.randn(G, S, device=device, dtype=dtype) * 0.01 - 5.0).contiguous()
        kernel_family_idx = family_idx
        kernel_mt = mt
        ref_family_idx = family_idx

    ancestor_cols, ancestors_dense, sp_child1, sp_child2, _ = _binary_tree_helpers(S, device, dtype)
    del ancestor_cols
    Pibar, row_max = _uniform_pibar_from_pi(Pi, kernel_mt[0] if kernel_mt.ndim == 2 else kernel_mt, ancestors_dense)
    v_k = (torch.randn(W, S, device=device, dtype=dtype) * 0.1).contiguous()
    sl = torch.tensor([0, 1, 2, 3, 1, 4, 5], device=device, dtype=torch.long)
    sr = torch.tensor([1, 2, 3, 4, 0, 2, 1], device=device, dtype=torch.long)
    reduce_idx = torch.tensor([0, 1, 1, 2, 2, 0, 3], device=device, dtype=torch.long)
    wlsp = (torch.randn(N, device=device, dtype=dtype) * 0.1 - 1.0).contiguous()
    active_mask = torch.tensor([True, False, True, True], device=device) if active else None

    ref = _dts_layout_reference(
        Pi=Pi,
        Pibar=Pibar,
        v_k=v_k,
        ws=ws,
        sl=sl,
        sr=sr,
        reduce_idx=reduce_idx,
        wlsp=wlsp,
        log_pD=log_pD,
        log_pS=log_pS,
        sp_child1=sp_child1,
        sp_child2=sp_child2,
        family_idx=ref_family_idx,
        active_mask=active_mask,
        mt_squeezed=kernel_mt,
        row_max=row_max if output_pibar_ud else None,
    )

    rhs = torch.zeros_like(Pi)
    grad_log_pD = torch.zeros_like(log_pD)
    grad_log_pS = torch.zeros_like(log_pS)
    grad_mt = torch.zeros_like(kernel_mt)
    result = dts_cross_backward_accum_fused(
        Pi,
        Pibar,
        v_k,
        ws,
        sl,
        sr,
        reduce_idx,
        wlsp,
        log_pD,
        log_pS,
        sp_child1,
        sp_child2,
        rhs,
        S,
        active_mask=active_mask,
        merge_s_term=True,
        grad_log_pD=grad_log_pD,
        grad_log_pS=grad_log_pS,
        grad_mt=grad_mt,
        accum_param_reductions=True,
        accum_mt_reduction=True,
        output_pibar_ud=output_pibar_ud,
        mt_squeezed=kernel_mt if output_pibar_ud else None,
        pibar_row_max=row_max if output_pibar_ud else None,
        family_idx=kernel_family_idx,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(rhs, ref[0], atol=5e-5, rtol=5e-5)
    torch.testing.assert_close(grad_log_pD, ref[3], atol=5e-5, rtol=5e-5)
    torch.testing.assert_close(grad_log_pS, ref[4], atol=5e-5, rtol=5e-5)
    torch.testing.assert_close(grad_mt, ref[5], atol=5e-5, rtol=5e-5)
    if output_pibar_ud:
        pibar_ud, pibar_A, param_pD, param_pS = result
        torch.testing.assert_close(pibar_ud, ref[6], atol=5e-5, rtol=5e-5)
        torch.testing.assert_close(pibar_A, ref[7], atol=5e-5, rtol=5e-5)
    else:
        grad_pibar_l, grad_pibar_r, param_pD, param_pS = result
        torch.testing.assert_close(grad_pibar_l, ref[1], atol=5e-5, rtol=5e-5)
        torch.testing.assert_close(grad_pibar_r, ref[2], atol=5e-5, rtol=5e-5)
    assert param_pD is None and param_pS is None


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


def _parent_tiled_fixture(dtype, active):
    torch.manual_seed(31)
    device = torch.device("cuda")
    C, S, W, N = 12, 31, 4, 10
    ws = 8

    ancestor_cols, ancestors_dense, sp_child1, sp_child2, level_parents = _binary_tree_helpers(
        S, device, dtype
    )
    del ancestor_cols, level_parents

    Pi = (torch.randn(C, S, device=device, dtype=dtype) * 0.2 - 2.0).contiguous()
    mt = (torch.randn(S, device=device, dtype=dtype) * 0.01 - 4.0).contiguous()
    Pibar, row_max = _uniform_pibar_from_pi(Pi, mt, ancestors_dense)
    v_k = (torch.randn(W, S, device=device, dtype=dtype) * 0.1).contiguous()

    sl = torch.tensor([7, 0, 1, 2, 3, 1, 4, 5, 2, 6], device=device, dtype=torch.long)
    sr = torch.tensor([0, 1, 2, 3, 4, 0, 2, 6, 5, 7], device=device, dtype=torch.long)
    reduce_idx = torch.tensor([2, 0, 0, 0, 0, 1, 1, 1, 3, 3], device=device, dtype=torch.long)
    wlsp = (torch.randn(N, device=device, dtype=dtype) * 0.1 - 1.0).contiguous()
    active_mask = torch.tensor([True, False, True, True], device=device) if active else None

    ge2_ptr = torch.tensor([0, 4, 7, 9], device=device, dtype=torch.long)
    ge2_parent_ids = torch.tensor([0, 1, 3], device=device, dtype=torch.long)

    return {
        "Pi": Pi,
        "Pibar": Pibar,
        "v_k": v_k,
        "ws": ws,
        "sl": sl,
        "sr": sr,
        "reduce_idx": reduce_idx,
        "wlsp": wlsp,
        "log_pD": _scalar_param(-4.0, device=device, dtype=dtype, shape="1d"),
        "log_pS": _scalar_param(-5.0, device=device, dtype=dtype, shape="1d"),
        "sp_child1": sp_child1,
        "sp_child2": sp_child2,
        "S": S,
        "active_mask": active_mask,
        "n_eq1": 1,
        "ge2_ptr": ge2_ptr,
        "ge2_parent_ids": ge2_parent_ids,
        "ge2_max_fanout": 4,
        "mt": mt,
        "row_max": row_max,
    }


_PARENT_TILE_VARIANTS = [
    pytest.param(dts_cross_backward_accum_parent_tiled_fused, id="rectangular"),
    pytest.param(dts_cross_backward_accum_parent_ragged_fused, id="ragged"),
]


@pytest.mark.parametrize("dtype,atol,rtol", [(torch.float32, 4e-5, 4e-5), (torch.float64, 1e-10, 1e-10)])
@pytest.mark.parametrize("active", [False, True])
@pytest.mark.parametrize("parent_impl", _PARENT_TILE_VARIANTS)
def test_dts_backward_parent_tile_variants_matches_direct_accum(
    dtype, atol, rtol, active, parent_impl
):
    args = _parent_tiled_fixture(dtype, active)
    base_rhs = torch.zeros_like(args["Pi"])
    tiled_rhs = torch.zeros_like(args["Pi"])
    base_pD = torch.zeros(1, device=args["Pi"].device, dtype=dtype)
    base_pS = torch.zeros(1, device=args["Pi"].device, dtype=dtype)
    tiled_pD = torch.zeros(1, device=args["Pi"].device, dtype=dtype)
    tiled_pS = torch.zeros(1, device=args["Pi"].device, dtype=dtype)

    base = dts_cross_backward_accum_fused(
        args["Pi"],
        args["Pibar"],
        args["v_k"],
        args["ws"],
        args["sl"],
        args["sr"],
        args["reduce_idx"],
        args["wlsp"],
        args["log_pD"],
        args["log_pS"],
        args["sp_child1"],
        args["sp_child2"],
        base_rhs,
        args["S"],
        active_mask=args["active_mask"],
        merge_s_term=True,
        grad_log_pD=base_pD,
        grad_log_pS=base_pS,
        accum_param_reductions=True,
        output_pibar_ud=True,
        mt_squeezed=args["mt"],
        pibar_row_max=args["row_max"],
    )
    tiled = parent_impl(
        args["Pi"],
        args["Pibar"],
        args["v_k"],
        args["ws"],
        args["sl"],
        args["sr"],
        args["reduce_idx"],
        args["wlsp"],
        args["n_eq1"],
        args["ge2_ptr"],
        args["ge2_parent_ids"],
        args["log_pD"],
        args["log_pS"],
        args["sp_child1"],
        args["sp_child2"],
        tiled_rhs,
        args["S"],
        active_mask=args["active_mask"],
        grad_log_pD=tiled_pD,
        grad_log_pS=tiled_pS,
        mt_squeezed=args["mt"],
        pibar_row_max=args["row_max"],
        tile_splits=2,
        ge2_max_fanout=args["ge2_max_fanout"],
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(tiled_rhs, base_rhs, atol=atol, rtol=rtol)
    torch.testing.assert_close(tiled[0], base[0], atol=atol, rtol=rtol)
    torch.testing.assert_close(tiled[1], base[1], atol=atol, rtol=rtol)
    torch.testing.assert_close(tiled_pD, base_pD, atol=atol, rtol=rtol)
    torch.testing.assert_close(tiled_pS, base_pS, atol=atol, rtol=rtol)
    assert tiled[2] is None and tiled[3] is None


@pytest.mark.parametrize("dtype,atol,rtol", [(torch.float32, 5e-5, 5e-5), (torch.float64, 1e-9, 1e-10)])
@pytest.mark.parametrize("active", [False, True])
@pytest.mark.parametrize("parent_impl", _PARENT_TILE_VARIANTS)
def test_dts_backward_parent_tile_variants_staged_pibar_ud_matches_direct(
    dtype, atol, rtol, active, parent_impl
):
    args = _parent_tiled_fixture(dtype, active)
    base_rhs = torch.zeros_like(args["Pi"])
    tiled_rhs = torch.zeros_like(args["Pi"])
    base_mt = torch.zeros(args["S"], device=args["Pi"].device, dtype=dtype)
    tiled_mt = torch.zeros(args["S"], device=args["Pi"].device, dtype=dtype)
    base_pD = torch.zeros(1, device=args["Pi"].device, dtype=dtype)
    base_pS = torch.zeros(1, device=args["Pi"].device, dtype=dtype)
    tiled_pD = torch.zeros(1, device=args["Pi"].device, dtype=dtype)
    tiled_pS = torch.zeros(1, device=args["Pi"].device, dtype=dtype)

    base = dts_cross_backward_accum_fused(
        args["Pi"],
        args["Pibar"],
        args["v_k"],
        args["ws"],
        args["sl"],
        args["sr"],
        args["reduce_idx"],
        args["wlsp"],
        args["log_pD"],
        args["log_pS"],
        args["sp_child1"],
        args["sp_child2"],
        base_rhs,
        args["S"],
        active_mask=args["active_mask"],
        merge_s_term=True,
        grad_log_pD=base_pD,
        grad_log_pS=base_pS,
        accum_param_reductions=True,
        grad_mt=base_mt,
        accum_mt_reduction=True,
        output_pibar_ud=True,
        mt_squeezed=args["mt"],
        pibar_row_max=args["row_max"],
    )
    tiled = parent_impl(
        args["Pi"],
        args["Pibar"],
        args["v_k"],
        args["ws"],
        args["sl"],
        args["sr"],
        args["reduce_idx"],
        args["wlsp"],
        args["n_eq1"],
        args["ge2_ptr"],
        args["ge2_parent_ids"],
        args["log_pD"],
        args["log_pS"],
        args["sp_child1"],
        args["sp_child2"],
        tiled_rhs,
        args["S"],
        active_mask=args["active_mask"],
        grad_log_pD=tiled_pD,
        grad_log_pS=tiled_pS,
        tile_splits=2,
        grad_mt=tiled_mt,
        accum_mt_reduction=True,
        mt_squeezed=args["mt"],
        pibar_row_max=args["row_max"],
        ge2_max_fanout=args["ge2_max_fanout"],
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(tiled_rhs, base_rhs, atol=atol, rtol=rtol)
    torch.testing.assert_close(tiled_mt, base_mt, atol=atol, rtol=rtol)
    torch.testing.assert_close(tiled[0], base[0], atol=atol, rtol=rtol)
    torch.testing.assert_close(tiled[1], base[1], atol=atol, rtol=rtol)
    torch.testing.assert_close(tiled_pD, base_pD, atol=atol, rtol=rtol)
    torch.testing.assert_close(tiled_pS, base_pS, atol=atol, rtol=rtol)
    assert tiled[2] is None and tiled[3] is None


@pytest.mark.parametrize("dtype,atol,rtol", [(torch.float32, 3e-5, 3e-5), (torch.float64, 1e-10, 1e-10)])
@pytest.mark.parametrize("active", [False, True])
def test_uniform_cross_pibar_from_ud_skip_zero_sides_matches_baseline(dtype, atol, rtol, active):
    torch.manual_seed(37)
    device = torch.device("cuda")
    C, S, W, N = 9, 31, 3, 6
    ws = 6

    _, _, sp_child1, sp_child2, level_parents = _binary_tree_helpers(S, device, dtype)
    Pi = (torch.randn(C, S, device=device, dtype=dtype) * 0.2 - 2.0).contiguous()
    pibar_row_max = Pi.max(dim=1).values.contiguous()
    sl = torch.tensor([0, 1, 2, 3, 1, 4], device=device, dtype=torch.long)
    sr = torch.tensor([1, 2, 3, 4, 0, 2], device=device, dtype=torch.long)
    reduce_idx = torch.tensor([0, 1, 1, 2, 2, 0], device=device, dtype=torch.long)
    active_mask = torch.tensor([True, False, True], device=device) if active else None

    pibar_ud = (torch.randn(2 * N, S, device=device, dtype=dtype) * 0.01).contiguous()
    zero_side_rows = torch.tensor([0, N + 4], device=device, dtype=torch.long)
    pibar_ud[zero_side_rows] = 0
    pibar_A = pibar_ud.sum(dim=1).contiguous()
    side_active = (pibar_ud.abs().amax(dim=1) != 0).contiguous()

    base_ud = pibar_ud.clone()
    skip_ud = pibar_ud.clone()
    compact_ud = pibar_ud.clone()
    base_rhs = torch.zeros(C, S, device=device, dtype=dtype)
    skip_rhs = torch.zeros(C, S, device=device, dtype=dtype)
    compact_rhs = torch.zeros(C, S, device=device, dtype=dtype)
    ptr = [0]
    flat_parent = []
    flat_child1 = []
    flat_child2 = []
    for level in level_parents.cpu().tolist():
        parents = [int(parent) for parent in level if int(parent) >= 0]
        flat_parent.extend(parents)
        flat_child1.extend(int(sp_child1[parent].item()) for parent in parents)
        flat_child2.extend(int(sp_child2[parent].item()) for parent in parents)
        ptr.append(len(flat_parent))
    compact_ptr = torch.tensor(ptr, device=device, dtype=torch.long)
    compact_parent = torch.tensor(flat_parent, device=device, dtype=torch.int32)
    compact_child1 = torch.tensor(flat_child1, device=device, dtype=torch.int32)
    compact_child2 = torch.tensor(flat_child2, device=device, dtype=torch.int32)

    uniform_cross_pibar_vjp_tree_from_ud_fused(
        Pi,
        base_ud,
        pibar_A,
        sl,
        sr,
        sp_child1,
        sp_child2,
        level_parents,
        base_rhs,
        S,
        active_mask=active_mask,
        reduce_idx=reduce_idx,
        pibar_row_max=pibar_row_max,
        skip_zero_sides=False,
    )
    uniform_cross_pibar_vjp_tree_from_ud_fused(
        Pi,
        skip_ud,
        pibar_A,
        sl,
        sr,
        sp_child1,
        sp_child2,
        level_parents,
        skip_rhs,
        S,
        active_mask=active_mask,
        reduce_idx=reduce_idx,
        pibar_row_max=pibar_row_max,
        skip_zero_sides=True,
        side_active=side_active,
    )
    uniform_cross_pibar_vjp_tree_from_ud_fused(
        Pi,
        compact_ud,
        pibar_A,
        sl,
        sr,
        sp_child1,
        sp_child2,
        level_parents,
        compact_rhs,
        S,
        active_mask=active_mask,
        reduce_idx=reduce_idx,
        pibar_row_max=pibar_row_max,
        skip_zero_sides=True,
        side_active=side_active,
        compact_level_ptr=compact_ptr,
        compact_level_parents=compact_parent,
        compact_level_child1=compact_child1,
        compact_level_child2=compact_child2,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(skip_rhs, base_rhs, atol=atol, rtol=rtol)
    torch.testing.assert_close(compact_rhs, base_rhs, atol=atol, rtol=rtol)
    torch.testing.assert_close(skip_ud, base_ud, atol=atol, rtol=rtol)
    torch.testing.assert_close(compact_ud, base_ud, atol=atol, rtol=rtol)


def test_uniform_cross_pibar_from_ud_default_side_active_is_exact_nonzero():
    device = torch.device("cuda")
    dtype = torch.float32
    C, S, W, N = 6, 15, 3, 4

    _, _, sp_child1, sp_child2, level_parents = _binary_tree_helpers(S, device, dtype)
    Pi = (torch.randn(C, S, device=device, dtype=dtype) * 0.2 - 2.0).contiguous()
    pibar_row_max = Pi.max(dim=1).values.contiguous()
    sl = torch.tensor([0, 1, 2, 3], device=device, dtype=torch.long)
    sr = torch.tensor([1, 2, 3, 4], device=device, dtype=torch.long)
    reduce_idx = torch.tensor([0, 1, 1, 2], device=device, dtype=torch.long)

    pibar_ud = torch.zeros((2 * N, S), device=device, dtype=dtype)
    pibar_ud[1, 0] = 1e-12
    pibar_ud[N + 2, 3] = -1e-12
    pibar_A = pibar_ud.sum(dim=1).contiguous()
    expected = pibar_ud.abs().amax(dim=1) != 0
    rhs = torch.zeros(C, S, device=device, dtype=dtype)

    side_active = uniform_cross_pibar_vjp_tree_from_ud_fused(
        Pi,
        pibar_ud,
        pibar_A,
        sl,
        sr,
        sp_child1,
        sp_child2,
        level_parents,
        rhs,
        S,
        reduce_idx=reduce_idx,
        pibar_row_max=pibar_row_max,
        skip_zero_sides=True,
    )
    torch.cuda.synchronize()

    assert side_active is not None
    assert torch.equal(side_active.cpu(), expected.cpu())


def test_uniform_cross_pibar_from_ud_threshold_masks_small_nonzero_sides():
    device = torch.device("cuda")
    dtype = torch.float32
    C, S, W, N = 6, 15, 3, 4
    side_threshold = 1e-5

    _, _, sp_child1, sp_child2, level_parents = _binary_tree_helpers(S, device, dtype)
    Pi = (torch.randn(C, S, device=device, dtype=dtype) * 0.2 - 2.0).contiguous()
    pibar_row_max = Pi.max(dim=1).values.contiguous()
    sl = torch.tensor([0, 1, 2, 3], device=device, dtype=torch.long)
    sr = torch.tensor([1, 2, 3, 4], device=device, dtype=torch.long)
    reduce_idx = torch.tensor([0, 1, 1, 2], device=device, dtype=torch.long)

    pibar_ud = torch.zeros((2 * N, S), device=device, dtype=dtype)
    pibar_ud[0, 0] = 1e-7
    pibar_ud[1, 0] = 8e-6
    pibar_ud[1, 1] = -4e-6
    pibar_ud[N + 2, 4] = 1e-3
    pibar_A = pibar_ud.sum(dim=1).contiguous()
    expected = pibar_ud.abs().sum(dim=1) > side_threshold
    rhs = torch.zeros(C, S, device=device, dtype=dtype)

    side_active = uniform_cross_pibar_vjp_tree_from_ud_fused(
        Pi,
        pibar_ud,
        pibar_A,
        sl,
        sr,
        sp_child1,
        sp_child2,
        level_parents,
        rhs,
        S,
        reduce_idx=reduce_idx,
        pibar_row_max=pibar_row_max,
        skip_zero_sides=True,
        side_active_threshold=side_threshold,
    )
    torch.cuda.synchronize()

    assert side_active is not None
    assert not bool(side_active[0].item())
    assert bool(side_active[1].item())
    assert bool(side_active[N + 2].item())
    assert torch.equal(side_active.cpu(), expected.cpu())


@pytest.mark.parametrize("use_compact_levels", [False, True])
def test_uniform_cross_pibar_from_ud_cuda_shared_preserves_pibar_ud(monkeypatch, use_compact_levels):
    torch.manual_seed(43)
    device = torch.device("cuda")
    dtype = torch.float32
    C, S, W, N = 9, 31, 3, 6

    _, _, sp_child1, sp_child2, level_parents = _binary_tree_helpers(S, device, dtype)
    compact_ptr, compact_nodes, compact_child1, compact_child2 = _compact_level_helpers(
        level_parents, sp_child1, sp_child2
    )
    Pi = (torch.randn(C, S, device=device, dtype=dtype) * 0.2 - 2.0).contiguous()
    pibar_row_max = Pi.max(dim=1).values.contiguous()
    sl = torch.tensor([0, 1, 2, 3, 1, 4], device=device, dtype=torch.long)
    sr = torch.tensor([1, 2, 3, 4, 0, 2], device=device, dtype=torch.long)
    reduce_idx = torch.tensor([0, 1, 1, 2, 2, 0], device=device, dtype=torch.long)
    active_mask = torch.tensor([True, False, True], device=device)

    pibar_ud = (torch.randn(2 * N, S, device=device, dtype=dtype) * 0.01).contiguous()
    pibar_ud[torch.tensor([0, N + 4], device=device, dtype=torch.long)] = 0
    pibar_A = pibar_ud.sum(dim=1).contiguous()
    side_active = (pibar_ud.abs().amax(dim=1) != 0).contiguous()

    original_ud = pibar_ud.clone()
    baseline_ud = pibar_ud.clone()
    actual_ud = pibar_ud.clone()
    baseline_rhs = torch.zeros(C, S, device=device, dtype=dtype)
    actual_rhs = torch.zeros(C, S, device=device, dtype=dtype)

    monkeypatch.setenv("GPUREC_CUDA_PIBAR_FROM_UD", "0")
    compact_kwargs = (
        {
            "compact_level_ptr": compact_ptr,
            "compact_level_parents": compact_nodes,
            "compact_level_child1": compact_child1,
            "compact_level_child2": compact_child2,
        }
        if use_compact_levels
        else {}
    )
    uniform_cross_pibar_vjp_tree_from_ud_fused(
        Pi,
        baseline_ud,
        pibar_A,
        sl,
        sr,
        sp_child1,
        sp_child2,
        level_parents,
        baseline_rhs,
        S,
        active_mask=active_mask,
        reduce_idx=reduce_idx,
        pibar_row_max=pibar_row_max,
        skip_zero_sides=True,
        side_active=side_active,
        **compact_kwargs,
    )

    monkeypatch.setenv("GPUREC_CUDA_PIBAR_FROM_UD", "1")
    monkeypatch.setenv("GPUREC_CUDA_PIBAR_FROM_UD_STRICT", "1")
    uniform_cross_pibar_vjp_tree_from_ud_fused(
        Pi,
        actual_ud,
        pibar_A,
        sl,
        sr,
        sp_child1,
        sp_child2,
        level_parents,
        actual_rhs,
        S,
        active_mask=active_mask,
        reduce_idx=reduce_idx,
        pibar_row_max=pibar_row_max,
        skip_zero_sides=True,
        side_active=side_active,
        **compact_kwargs,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(actual_rhs, baseline_rhs, atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(actual_ud, original_ud, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("dtype,atol,rtol", [(torch.float32, 3e-5, 3e-5), (torch.float64, 1e-10, 1e-10)])
@pytest.mark.parametrize("active", [False, True])
@pytest.mark.parametrize("skip_zero_sides", [False, True])
def test_uniform_cross_pibar_from_ud_compact_levels_matches_padded(
    dtype, atol, rtol, active, skip_zero_sides
):
    torch.manual_seed(41)
    device = torch.device("cuda")
    C, S, W, N = 9, 31, 3, 6
    ws = 6

    _, _, sp_child1, sp_child2, level_parents = _binary_tree_helpers(S, device, dtype)
    compact_ptr, compact_nodes, compact_child1, compact_child2 = _compact_level_helpers(
        level_parents, sp_child1, sp_child2
    )
    Pi = (torch.randn(C, S, device=device, dtype=dtype) * 0.2 - 2.0).contiguous()
    pibar_row_max = Pi.max(dim=1).values.contiguous()
    sl = torch.tensor([0, 1, 2, 3, 1, 4], device=device, dtype=torch.long)
    sr = torch.tensor([1, 2, 3, 4, 0, 2], device=device, dtype=torch.long)
    reduce_idx = torch.tensor([0, 1, 1, 2, 2, 0], device=device, dtype=torch.long)
    active_mask = torch.tensor([True, False, True], device=device) if active else None

    pibar_ud = (torch.randn(2 * N, S, device=device, dtype=dtype) * 0.01).contiguous()
    pibar_ud[torch.tensor([0, N + 4], device=device, dtype=torch.long)] = 0
    pibar_A = pibar_ud.sum(dim=1).contiguous()
    side_active = (pibar_ud.abs().amax(dim=1) != 0).contiguous()

    padded_ud = pibar_ud.clone()
    compact_ud = pibar_ud.clone()
    padded_rhs = torch.zeros(C, S, device=device, dtype=dtype)
    compact_rhs = torch.zeros(C, S, device=device, dtype=dtype)

    uniform_cross_pibar_vjp_tree_from_ud_fused(
        Pi,
        padded_ud,
        pibar_A,
        sl,
        sr,
        sp_child1,
        sp_child2,
        level_parents,
        padded_rhs,
        S,
        active_mask=active_mask,
        reduce_idx=reduce_idx,
        pibar_row_max=pibar_row_max,
        skip_zero_sides=skip_zero_sides,
        side_active=side_active if skip_zero_sides else None,
    )
    uniform_cross_pibar_vjp_tree_from_ud_fused(
        Pi,
        compact_ud,
        pibar_A,
        sl,
        sr,
        sp_child1,
        sp_child2,
        level_parents,
        compact_rhs,
        S,
        active_mask=active_mask,
        reduce_idx=reduce_idx,
        pibar_row_max=pibar_row_max,
        skip_zero_sides=skip_zero_sides,
        side_active=side_active if skip_zero_sides else None,
        compact_level_ptr=compact_ptr,
        compact_level_parents=compact_nodes,
        compact_level_child1=compact_child1,
        compact_level_child2=compact_child2,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(compact_rhs, padded_rhs, atol=atol, rtol=rtol)
    torch.testing.assert_close(compact_ud, padded_ud, atol=atol, rtol=rtol)


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


@pytest.mark.parametrize("dtype,atol,rtol", [(torch.float32, 5e-5, 5e-5), (torch.float64, 1e-9, 1e-10)])
@pytest.mark.parametrize("active", [False, True])
def test_dts_staged_grad_mt_two_stage_matches_vector_atomics(dtype, atol, rtol, active):
    torch.manual_seed(43)
    device = torch.device("cuda")
    C, S, W, N = 10, 31, 3, 8
    ws = 7

    _, ancestors_dense, sp_child1, sp_child2, _ = _binary_tree_helpers(S, device, dtype)
    Pi = (torch.randn(C, S, device=device, dtype=dtype) * 0.2 - 2.0).contiguous()
    mt = (torch.randn(S, device=device, dtype=dtype) * 0.01 - 4.0).contiguous()
    Pibar, row_max = _uniform_pibar_from_pi(Pi, mt, ancestors_dense)
    v_k = (torch.randn(W, S, device=device, dtype=dtype) * 0.1).contiguous()

    sl = torch.tensor([0, 1, 2, 3, 1, 4, 5, 2], device=device, dtype=torch.long)
    sr = torch.tensor([1, 2, 3, 4, 0, 2, 6, 5], device=device, dtype=torch.long)
    reduce_idx = torch.tensor([0, 1, 1, 2, 2, 0, 1, 2], device=device, dtype=torch.long)
    wlsp = (torch.randn(N, device=device, dtype=dtype) * 0.1 - 1.0).contiguous()
    active_mask = torch.tensor([True, False, True], device=device) if active else None

    atomic_rhs = torch.zeros(C, S, device=device, dtype=dtype)
    two_stage_rhs = torch.zeros(C, S, device=device, dtype=dtype)
    atomic_mt = torch.zeros(S, device=device, dtype=dtype)
    two_stage_mt = torch.zeros(S, device=device, dtype=dtype)
    atomic_pD = torch.zeros(1, device=device, dtype=dtype)
    atomic_pS = torch.zeros(1, device=device, dtype=dtype)
    two_stage_pD = torch.zeros(1, device=device, dtype=dtype)
    two_stage_pS = torch.zeros(1, device=device, dtype=dtype)
    two_stage_scratch = {
        "pibar_ud": torch.empty((2 * N + 4, S), device=device, dtype=dtype),
        "pibar_A": torch.empty((2 * N + 4,), device=device, dtype=dtype),
        "pibar_side_active": torch.empty((2 * N + 4,), device=device, dtype=torch.bool),
        "grad_mt_partial": torch.empty(((N + 2) // 3 + 2, S), device=device, dtype=dtype),
    }

    atomic = dts_cross_backward_accum_fused(
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
        atomic_rhs,
        S,
        active_mask=active_mask,
        merge_s_term=True,
        grad_log_pD=atomic_pD,
        grad_log_pS=atomic_pS,
        accum_param_reductions=True,
        grad_mt=atomic_mt,
        accum_mt_reduction=True,
        output_pibar_ud=True,
        output_pibar_side_active=True,
        mt_squeezed=mt,
        pibar_row_max=row_max,
    )
    two_stage = dts_cross_backward_accum_fused(
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
        two_stage_rhs,
        S,
        active_mask=active_mask,
        merge_s_term=True,
        grad_log_pD=two_stage_pD,
        grad_log_pS=two_stage_pS,
        accum_param_reductions=True,
        grad_mt=two_stage_mt,
        accum_mt_reduction=True,
        output_pibar_ud=True,
        output_pibar_side_active=True,
        mt_squeezed=mt,
        pibar_row_max=row_max,
        grad_mt_two_stage=True,
        grad_mt_two_stage_tile_splits=3,
        scratch=two_stage_scratch,
    )
    torch.cuda.synchronize()

    atomic_ud, atomic_A, atomic_side, atomic_param_pD, atomic_param_pS = atomic
    two_stage_ud, two_stage_A, two_stage_side, two_stage_param_pD, two_stage_param_pS = two_stage
    assert atomic_param_pD is None and atomic_param_pS is None
    assert two_stage_param_pD is None and two_stage_param_pS is None
    torch.testing.assert_close(two_stage_rhs, atomic_rhs, atol=atol, rtol=rtol)
    torch.testing.assert_close(two_stage_ud, atomic_ud, atol=atol, rtol=rtol)
    torch.testing.assert_close(two_stage_A, atomic_A, atol=atol, rtol=rtol)
    torch.testing.assert_close(two_stage_mt, atomic_mt, atol=atol, rtol=rtol)
    torch.testing.assert_close(two_stage_pD, atomic_pD, atol=atol, rtol=rtol)
    torch.testing.assert_close(two_stage_pS, atomic_pS, atol=atol, rtol=rtol)
    assert torch.equal(two_stage_side.cpu(), atomic_side.cpu())


def test_dts_staged_skip_inactive_pibar_output_zero_keeps_stale_rows_masked():
    torch.manual_seed(44)
    device = torch.device("cuda")
    dtype = torch.float32
    C, S, W, N = 10, 31, 3, 8
    ws = 7

    _, ancestors_dense, sp_child1, sp_child2, level_parents = _binary_tree_helpers(
        S, device, dtype
    )
    Pi = (torch.randn(C, S, device=device, dtype=dtype) * 0.2 - 2.0).contiguous()
    mt = (torch.randn(S, device=device, dtype=dtype) * 0.01 - 4.0).contiguous()
    Pibar, row_max = _uniform_pibar_from_pi(Pi, mt, ancestors_dense)
    v_k = (torch.randn(W, S, device=device, dtype=dtype) * 0.1).contiguous()

    sl = torch.tensor([0, 1, 2, 3, 1, 4, 5, 2], device=device, dtype=torch.long)
    sr = torch.tensor([1, 2, 3, 4, 0, 2, 6, 5], device=device, dtype=torch.long)
    reduce_idx = torch.tensor([0, 1, 1, 2, 2, 0, 1, 2], device=device, dtype=torch.long)
    wlsp = (torch.randn(N, device=device, dtype=dtype) * 0.1 - 1.0).contiguous()
    active_mask = torch.tensor([True, False, True], device=device)

    base_rhs = torch.zeros(C, S, device=device, dtype=dtype)
    base_mt = torch.zeros(S, device=device, dtype=dtype)
    base_pD = torch.zeros(1, device=device, dtype=dtype)
    base_pS = torch.zeros(1, device=device, dtype=dtype)
    base_ud, base_A, base_side, _, _ = dts_cross_backward_accum_fused(
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
        grad_log_pD=base_pD,
        grad_log_pS=base_pS,
        accum_param_reductions=True,
        grad_mt=base_mt,
        accum_mt_reduction=True,
        output_pibar_ud=True,
        output_pibar_side_active=True,
        mt_squeezed=mt,
        pibar_row_max=row_max,
    )

    sentinel = 77.0
    skip_rhs = torch.zeros(C, S, device=device, dtype=dtype)
    skip_mt = torch.zeros(S, device=device, dtype=dtype)
    skip_pD = torch.zeros(1, device=device, dtype=dtype)
    skip_pS = torch.zeros(1, device=device, dtype=dtype)
    scratch = {
        "pibar_ud": torch.full((2 * N, S), sentinel, device=device, dtype=dtype),
        "pibar_A": torch.full((2 * N,), sentinel, device=device, dtype=dtype),
        "pibar_side_active": torch.ones((2 * N,), device=device, dtype=torch.bool),
    }
    skip_ud, skip_A, skip_side, _, _ = dts_cross_backward_accum_fused(
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
        skip_rhs,
        S,
        active_mask=active_mask,
        merge_s_term=True,
        grad_log_pD=skip_pD,
        grad_log_pS=skip_pS,
        accum_param_reductions=True,
        grad_mt=skip_mt,
        accum_mt_reduction=True,
        output_pibar_ud=True,
        output_pibar_side_active=True,
        mt_squeezed=mt,
        pibar_row_max=row_max,
        skip_inactive_pibar_output_zero=True,
        scratch=scratch,
    )
    torch.cuda.synchronize()

    side_active_by_parent = torch.cat(
        [active_mask[reduce_idx], active_mask[reduce_idx]]
    )
    torch.testing.assert_close(skip_rhs, base_rhs, atol=5e-5, rtol=5e-5)
    torch.testing.assert_close(skip_mt, base_mt, atol=5e-5, rtol=5e-5)
    torch.testing.assert_close(skip_pD, base_pD, atol=5e-5, rtol=5e-5)
    torch.testing.assert_close(skip_pS, base_pS, atol=5e-5, rtol=5e-5)
    torch.testing.assert_close(skip_ud[side_active_by_parent], base_ud[side_active_by_parent])
    torch.testing.assert_close(skip_A[side_active_by_parent], base_A[side_active_by_parent])
    torch.testing.assert_close(
        skip_ud[~side_active_by_parent],
        torch.full_like(skip_ud[~side_active_by_parent], sentinel),
    )
    torch.testing.assert_close(
        skip_A[~side_active_by_parent],
        torch.full_like(skip_A[~side_active_by_parent], sentinel),
    )
    assert torch.equal(skip_side.cpu(), base_side.cpu())

    base_total_rhs = base_rhs.clone()
    skip_total_rhs = skip_rhs.clone()
    uniform_cross_pibar_vjp_tree_from_ud_fused(
        Pi,
        base_ud.clone(),
        base_A,
        sl,
        sr,
        sp_child1,
        sp_child2,
        level_parents,
        base_total_rhs,
        S,
        active_mask=active_mask,
        reduce_idx=reduce_idx,
        pibar_row_max=row_max,
        side_active=base_side,
    )
    uniform_cross_pibar_vjp_tree_from_ud_fused(
        Pi,
        skip_ud.clone(),
        skip_A,
        sl,
        sr,
        sp_child1,
        sp_child2,
        level_parents,
        skip_total_rhs,
        S,
        active_mask=active_mask,
        reduce_idx=reduce_idx,
        pibar_row_max=row_max,
        side_active=skip_side,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(skip_total_rhs, base_total_rhs, atol=5e-5, rtol=5e-5)


@pytest.mark.parametrize("dtype,atol,rtol", [(torch.float32, 4e-5, 4e-5), (torch.float64, 1e-9, 1e-10)])
@pytest.mark.parametrize("active", [False, True])
def test_dts_staged_pibar_ud_skip_zero_sides_matches_unpruned(dtype, atol, rtol, active):
    torch.manual_seed(37)
    device = torch.device("cuda")
    C, S, W, N = 10, 31, 3, 7
    ws = 7

    _, ancestors_dense, sp_child1, sp_child2, level_parents = _binary_tree_helpers(
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

    direct_rhs = torch.zeros(C, S, device=device, dtype=dtype)
    pibar_ud, pibar_A, dts_side_active, _, _ = dts_cross_backward_accum_fused(
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
        direct_rhs,
        S,
        active_mask=active_mask,
        merge_s_term=True,
        grad_mt=torch.zeros(S, device=device, dtype=dtype),
        accum_mt_reduction=True,
        output_pibar_ud=True,
        output_pibar_side_active=True,
        mt_squeezed=mt,
        pibar_row_max=row_max,
    )
    torch.cuda.synchronize()
    expected_side_active = pibar_ud.abs().amax(dim=1) != 0
    assert torch.equal(dts_side_active.cpu(), expected_side_active.cpu())

    pibar_ud_unpruned = pibar_ud.clone()
    pibar_A_unpruned = pibar_A.clone()
    pibar_ud_pruned = pibar_ud.clone()
    pibar_A_pruned = pibar_A.clone()
    zero_rows = torch.tensor([0, N + 2], device=device, dtype=torch.long)
    pibar_ud_unpruned[zero_rows] = 0
    pibar_ud_pruned[zero_rows] = 0
    pibar_A_unpruned[zero_rows] = 0
    pibar_A_pruned[zero_rows] = 0

    unpruned_rhs = direct_rhs.clone()
    pruned_rhs = direct_rhs.clone()
    uniform_cross_pibar_vjp_tree_from_ud_fused(
        Pi,
        pibar_ud_unpruned,
        pibar_A_unpruned,
        sl,
        sr,
        sp_child1,
        sp_child2,
        level_parents,
        unpruned_rhs,
        S,
        active_mask=active_mask,
        reduce_idx=reduce_idx,
        pibar_row_max=row_max,
    )
    side_active = uniform_cross_pibar_vjp_tree_from_ud_fused(
        Pi,
        pibar_ud_pruned,
        pibar_A_pruned,
        sl,
        sr,
        sp_child1,
        sp_child2,
        level_parents,
        pruned_rhs,
        S,
        active_mask=active_mask,
        reduce_idx=reduce_idx,
        pibar_row_max=row_max,
        skip_zero_sides=True,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(pruned_rhs, unpruned_rhs, atol=atol, rtol=rtol)
    assert side_active is not None
    assert not bool(side_active[0].item())
    assert not bool(side_active[N + 2].item())


@pytest.mark.parametrize("dtype,atol,rtol", [(torch.float32, 4e-5, 4e-5), (torch.float64, 1e-9, 1e-10)])
@pytest.mark.parametrize("active", [False, True])
def test_dts_staged_pibar_ud_side_threshold_uses_error_budget_bound(dtype, atol, rtol, active):
    torch.manual_seed(43)
    device = torch.device("cuda")
    C, S, W, N = 10, 31, 3, 7
    ws = 7

    _, ancestors_dense, sp_child1, sp_child2, level_parents = _binary_tree_helpers(
        S, device, dtype
    )
    Pi = (torch.randn(C, S, device=device, dtype=dtype) * 0.2 - 2.0).contiguous()
    mt = (torch.randn(S, device=device, dtype=dtype) * 0.01 - 4.0).contiguous()
    Pibar, row_max = _uniform_pibar_from_pi(Pi, mt, ancestors_dense)
    v_k = (torch.randn(W, S, device=device, dtype=dtype) * 0.025).contiguous()

    sl = torch.tensor([0, 1, 2, 3, 1, 4, 5], device=device, dtype=torch.long)
    sr = torch.tensor([1, 2, 3, 4, 0, 2, 6], device=device, dtype=torch.long)
    reduce_idx = torch.tensor([0, 1, 1, 2, 2, 0, 1], device=device, dtype=torch.long)
    wlsp = (torch.randn(N, device=device, dtype=dtype) * 0.1 - 1.0).contiguous()
    active_mask = torch.tensor([True, False, True], device=device) if active else None

    direct_rhs = torch.zeros(C, S, device=device, dtype=dtype)
    pibar_ud, pibar_A, exact_side_active, _, _ = dts_cross_backward_accum_fused(
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
        direct_rhs,
        S,
        active_mask=active_mask,
        merge_s_term=True,
        grad_mt=torch.zeros(S, device=device, dtype=dtype),
        accum_mt_reduction=True,
        output_pibar_ud=True,
        output_pibar_side_active=True,
        mt_squeezed=mt,
        pibar_row_max=row_max,
    )
    torch.cuda.synchronize()

    side_bound = pibar_ud.abs().sum(dim=1)
    positive_bounds = side_bound[side_bound > 0]
    assert positive_bounds.numel() > 1
    sorted_bounds = torch.sort(positive_bounds.detach().cpu()).values
    split = int(sorted_bounds.numel() // 2)
    threshold = float((sorted_bounds[split - 1] + sorted_bounds[split]) * 0.5)

    threshold_rhs = torch.zeros(C, S, device=device, dtype=dtype)
    threshold_ud, threshold_A, threshold_side_active, _, _ = dts_cross_backward_accum_fused(
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
        threshold_rhs,
        S,
        active_mask=active_mask,
        merge_s_term=True,
        grad_mt=torch.zeros(S, device=device, dtype=dtype),
        accum_mt_reduction=True,
        output_pibar_ud=True,
        output_pibar_side_active=True,
        pibar_side_threshold=threshold,
        mt_squeezed=mt,
        pibar_row_max=row_max,
    )
    torch.cuda.synchronize()

    expected_exact_active = pibar_ud.abs().amax(dim=1) != 0
    expected_threshold_active = side_bound > threshold
    assert torch.equal(exact_side_active.cpu(), expected_exact_active.cpu())
    assert torch.equal(threshold_side_active.cpu(), expected_threshold_active.cpu())
    assert bool((exact_side_active & ~threshold_side_active).any().item())
    torch.testing.assert_close(threshold_ud, pibar_ud, atol=atol, rtol=rtol)
    torch.testing.assert_close(threshold_A, pibar_A, atol=atol, rtol=rtol)
    torch.testing.assert_close(threshold_rhs, direct_rhs, atol=atol, rtol=rtol)

    full_rhs = direct_rhs.clone()
    pruned_rhs = threshold_rhs.clone()
    uniform_cross_pibar_vjp_tree_from_ud_fused(
        Pi,
        pibar_ud.clone(),
        pibar_A,
        sl,
        sr,
        sp_child1,
        sp_child2,
        level_parents,
        full_rhs,
        S,
        active_mask=active_mask,
        reduce_idx=reduce_idx,
        pibar_row_max=row_max,
    )
    uniform_cross_pibar_vjp_tree_from_ud_fused(
        Pi,
        threshold_ud.clone(),
        threshold_A,
        sl,
        sr,
        sp_child1,
        sp_child2,
        level_parents,
        pruned_rhs,
        S,
        active_mask=active_mask,
        reduce_idx=reduce_idx,
        pibar_row_max=row_max,
        side_active=threshold_side_active,
    )
    torch.cuda.synchronize()

    skipped_bound = float(side_bound[~threshold_side_active].sum().detach().cpu())
    max_rhs_diff = float((full_rhs - pruned_rhs).abs().max().detach().cpu())
    assert max_rhs_diff <= skipped_bound * (1.0 + 1e-5) + 1e-6
