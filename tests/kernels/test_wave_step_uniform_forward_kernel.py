import pytest
import torch

from gpurec.core.forward import _compute_Pibar_inline
from gpurec.core.kernels.wave_step import (
    build_uniform_linear_operator,
    wave_pibar_uniform_fused,
    wave_step_fused,
    wave_step_uniform_ancestor_fused,
    wave_step_uniform_csr_fused,
    wave_step_uniform_fused,
    wave_step_uniform_two_kernel_fused,
    wave_step_uniform_linear_fused,
)


def _balanced_species_tree(device):
    parent = torch.tensor([-1, 0, 0, 1, 1, 2, 2], dtype=torch.long)
    S = int(parent.numel())
    sp_child1 = torch.full((S,), S, dtype=torch.long)
    sp_child2 = torch.full((S,), S, dtype=torch.long)
    for child, par in enumerate(parent.tolist()):
        if par < 0:
            continue
        target = sp_child1 if int(sp_child1[par]) == S else sp_child2
        target[par] = child

    ancestors = torch.zeros((S, S), dtype=torch.float32)
    ancestor_lists = []
    max_depth = 0
    for desc in range(S):
        depth = 0
        cur = desc
        anc = []
        while cur >= 0:
            ancestors[desc, cur] = 1.0
            anc.append(cur)
            depth += 1
            cur = int(parent[cur])
        ancestor_lists.append(anc)
        max_depth = max(max_depth, depth)
    ancestor_cols = torch.full((S, max_depth), -1, dtype=torch.long)
    for desc, anc in enumerate(ancestor_lists):
        ancestor_cols[desc, :len(anc)] = torch.tensor(anc, dtype=torch.long)
    csr_indptr = [0]
    csr_indices = []
    for anc in ancestor_lists:
        csr_indices.extend(anc)
        csr_indptr.append(len(csr_indices))

    return (
        parent.to(device),
        sp_child1.to(device),
        sp_child2.to(device),
        ancestors.to(device),
        ancestor_cols.T.contiguous().to(device),
        torch.tensor(csr_indptr, dtype=torch.int32, device=device),
        torch.tensor(csr_indices, dtype=torch.int32, device=device),
        max_depth,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("per_clade_constants", [False, True])
def test_wave_step_uniform_fused_matches_sparse_ancestor_reference(per_clade_constants):
    torch.manual_seed(7)
    device = torch.device("cuda")
    dtype = torch.float32
    (
        sp_parent,
        sp_child1,
        sp_child2,
        ancestors,
        ancestor_cols,
        ancestor_csr_indptr,
        ancestor_csr_indices,
        max_depth,
    ) = _balanced_species_tree(device)

    W = 3
    S = int(sp_parent.numel())
    ws = 1
    C = W + 2
    Pi = torch.randn((C, S), device=device, dtype=dtype) * 1.5 - 3.0

    if per_clade_constants:
        mt = torch.randn((W, S), device=device, dtype=dtype) * 0.1
        DL = torch.randn((W, S), device=device, dtype=dtype) * 0.2 - 2.0
        Ebar = torch.randn((W, S), device=device, dtype=dtype) * 0.2 - 1.5
        E = torch.randn((W, S), device=device, dtype=dtype) * 0.2 - 2.5
        SL1 = torch.randn((W, S), device=device, dtype=dtype) * 0.2 - 2.0
        SL2 = torch.randn((W, S), device=device, dtype=dtype) * 0.2 - 2.0
    else:
        mt = torch.randn((S,), device=device, dtype=dtype) * 0.1
        DL = torch.randn((S,), device=device, dtype=dtype) * 0.2 - 2.0
        Ebar = torch.randn((S,), device=device, dtype=dtype) * 0.2 - 1.5
        E = torch.randn((S,), device=device, dtype=dtype) * 0.2 - 2.5
        SL1 = torch.randn((S,), device=device, dtype=dtype) * 0.2 - 2.0
        SL2 = torch.randn((S,), device=device, dtype=dtype) * 0.2 - 2.0

    leaf_term = torch.randn((W, S), device=device, dtype=dtype) * 0.2 - 4.0
    Pi_W = Pi[ws:ws + W]
    Pibar_ref = _compute_Pibar_inline(
        Pi_W,
        None,
        mt,
        "uniform",
        ancestors_T=ancestors.T.to_sparse_coo(),
    )
    Pi_ref = wave_step_fused(
        Pi_W,
        Pibar_ref,
        DL,
        Ebar,
        E,
        SL1,
        SL2,
        sp_child1,
        sp_child2,
        leaf_term,
    )

    Pibar = torch.full_like(Pi, float("-inf"))
    Pi_fused, max_diff = wave_step_uniform_fused(
        Pi,
        Pibar,
        ws,
        W,
        S,
        mt,
        DL,
        Ebar,
        E,
        SL1,
        SL2,
        sp_child1,
        sp_child2,
        sp_parent,
        max_depth,
        leaf_term,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(Pibar[ws:ws + W], Pibar_ref, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(Pi_fused, Pi_ref, rtol=2e-5, atol=2e-5)

    significant = Pi_ref > -100.0
    expected_diff = torch.abs(Pi_ref - Pi_W)[significant].max()
    torch.testing.assert_close(max_diff, expected_diff, rtol=2e-5, atol=2e-5)

    Pibar_two = torch.full_like(Pi, float("-inf"))
    Pi_two, max_diff_two = wave_step_uniform_two_kernel_fused(
        Pi,
        Pibar_two,
        ws,
        W,
        S,
        mt,
        DL,
        Ebar,
        E,
        SL1,
        SL2,
        sp_child1,
        sp_child2,
        sp_parent,
        max_depth,
        leaf_term,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(Pibar_two[ws:ws + W], Pibar_ref, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(Pi_two, Pi_ref, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(max_diff_two, expected_diff, rtol=2e-5, atol=2e-5)

    Pibar_ancestor = torch.full_like(Pi, float("-inf"))
    Pi_ancestor, max_diff_ancestor = wave_step_uniform_ancestor_fused(
        Pi,
        Pibar_ancestor,
        ws,
        W,
        S,
        mt,
        DL,
        Ebar,
        E,
        SL1,
        SL2,
        sp_child1,
        sp_child2,
        ancestor_cols,
        leaf_term,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(Pibar_ancestor[ws:ws + W], Pibar_ref, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(Pi_ancestor, Pi_ref, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(max_diff_ancestor, expected_diff, rtol=2e-5, atol=2e-5)

    Pibar_csr = torch.full_like(Pi, float("-inf"))
    Pi_csr, max_diff_csr = wave_step_uniform_csr_fused(
        Pi,
        Pibar_csr,
        ws,
        W,
        S,
        mt,
        DL,
        Ebar,
        E,
        SL1,
        SL2,
        sp_child1,
        sp_child2,
        ancestor_csr_indptr,
        ancestor_csr_indices,
        max_depth,
        leaf_term,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(Pibar_csr[ws:ws + W], Pibar_ref, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(Pi_csr, Pi_ref, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(max_diff_csr, expected_diff, rtol=2e-5, atol=2e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_wave_step_uniform_linear_fused_matches_sparse_ancestor_reference():
    torch.manual_seed(11)
    device = torch.device("cuda")
    dtype = torch.float32
    sp_parent, sp_child1, sp_child2, ancestors, _ancestor_cols, _csr_indptr, _csr_indices, _max_depth = _balanced_species_tree(device)

    W = 4
    S = int(sp_parent.numel())
    ws = 2
    C = W + 4
    Pi = torch.randn((C, S), device=device, dtype=dtype) * 1.2 - 3.0

    mt = torch.randn((S,), device=device, dtype=dtype) * 0.1 - 0.5
    DL = torch.randn((S,), device=device, dtype=dtype) * 0.2 - 2.0
    Ebar = torch.randn((S,), device=device, dtype=dtype) * 0.2 - 1.5
    E = torch.randn((S,), device=device, dtype=dtype) * 0.2 - 2.5
    SL1 = torch.randn((S,), device=device, dtype=dtype) * 0.2 - 2.0
    SL2 = torch.randn((S,), device=device, dtype=dtype) * 0.2 - 2.0
    leaf_term = torch.randn((W, S), device=device, dtype=dtype) * 0.2 - 4.0
    dts_r = torch.randn((W, S), device=device, dtype=dtype) * 0.3 - 5.0

    Pi_W = Pi[ws:ws + W]
    Pibar_ref = _compute_Pibar_inline(
        Pi_W,
        None,
        mt,
        "uniform",
        ancestors_T=ancestors.T.to_sparse_coo(),
    )
    Pi_ref = wave_step_fused(
        Pi_W,
        Pibar_ref,
        DL,
        Ebar,
        E,
        SL1,
        SL2,
        sp_child1,
        sp_child2,
        leaf_term,
        dts_r,
    )

    op = build_uniform_linear_operator(
        DL,
        Ebar,
        E,
        SL1,
        SL2,
        mt,
        sp_parent.cpu(),
        sp_child1.cpu(),
        sp_child2.cpu(),
        device=device,
        dtype=dtype,
    )
    Pi_fused, max_diff = wave_step_uniform_linear_fused(
        Pi,
        ws,
        W,
        S,
        op["op_cols"],
        op["op_vals"],
        op["v_scaled"],
        op["row_scale"],
        leaf_term,
        dts_r,
        debug_guard=True,
    )

    Pibar = torch.full_like(Pi, float("-inf"))
    wave_pibar_uniform_fused(Pi, Pibar, ws, W, S, mt, op["ancestor_cols"])
    torch.cuda.synchronize()

    torch.testing.assert_close(Pibar[ws:ws + W], Pibar_ref, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(Pi_fused, Pi_ref, rtol=2e-5, atol=2e-5)

    significant = Pi_ref > -100.0
    expected_diff = torch.abs(Pi_ref - Pi_W)[significant].max()
    torch.testing.assert_close(max_diff, expected_diff, rtol=2e-5, atol=2e-5)
