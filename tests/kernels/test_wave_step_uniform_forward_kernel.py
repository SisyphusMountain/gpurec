import pytest
import torch

from gpurec.core.kernels.dts_fused import dts_fused_parent_reduced
from gpurec.core.kernels.wave_step import (
    wave_pibar_uniform_parent_fused,
    wave_step_uniform_fused_into,
)


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


def _logsumexp2_reference(terms: torch.Tensor) -> torch.Tensor:
    vmax = terms.max(dim=0).values
    safe = torch.where(torch.isfinite(vmax), vmax, torch.zeros_like(vmax))
    return torch.log2(torch.exp2(terms - safe).sum(dim=0)) + vmax


def _uniform_pibar_reference(Pi_W, mt, ancestors):
    pi_max = Pi_W.max(dim=1, keepdim=True).values
    pi_exp = torch.exp2(Pi_W - pi_max)
    row_sum = pi_exp.sum(dim=1, keepdim=True)
    ancestor_sum = pi_exp @ ancestors.T
    denom = row_sum - ancestor_sum
    return torch.where(denom > 0, torch.log2(denom) + pi_max + mt, float("-inf"))


def _wave_step_reference(Pi_W, Pibar_W, DL, Ebar, E, SL1, SL2, sp_child1, sp_child2, leaf_term):
    W, S = Pi_W.shape
    pad = torch.full((W, 1), float("-inf"), device=Pi_W.device, dtype=Pi_W.dtype)
    Pi_pad = torch.cat([Pi_W, pad], dim=1)
    terms = torch.stack(
        [
            DL + Pi_W,
            Pi_W + Ebar,
            Pibar_W + E,
            SL1 + Pi_pad[:, sp_child1],
            SL2 + Pi_pad[:, sp_child2],
            leaf_term,
        ],
        dim=0,
    )
    return _logsumexp2_reference(terms)


def _dts_eq1_reference(
    Pi,
    Pibar,
    lefts,
    rights,
    sp_child1,
    sp_child2,
    log_pD,
    log_pS,
    split_logp,
    parent_ids,
    W,
):
    S = Pi.shape[1]
    pad = torch.full((Pi.shape[0], 1), float("-inf"), device=Pi.device, dtype=Pi.dtype)
    Pi_pad = torch.cat([Pi, pad], dim=1)
    out = torch.full((W, S), float("-inf"), device=Pi.device, dtype=Pi.dtype)
    for n in range(int(lefts.numel())):
        left = int(lefts[n])
        right = int(rights[n])
        parent = int(parent_ids[n])
        terms = torch.stack(
            [
                log_pD + Pi[left] + Pi[right],
                Pi[left] + Pibar[right],
                Pi[right] + Pibar[left],
                log_pS + Pi_pad[left, sp_child1] + Pi_pad[right, sp_child2],
                log_pS + Pi_pad[right, sp_child1] + Pi_pad[left, sp_child2],
            ],
            dim=0,
        )
        out[parent] = _logsumexp2_reference(terms) + split_logp[n]
    return out


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


@pytest.mark.parametrize("per_clade_constants", [False, True])
def test_wave_step_uniform_fused_matches_sparse_ancestor_reference(per_clade_constants):
    torch.manual_seed(7)
    device = torch.device("cuda")
    dtype = torch.float32
    sp_parent, sp_child1, sp_child2, ancestors, *_rest = _balanced_species_tree(device)
    max_depth = _rest[-1]

    W = 3
    S = int(sp_parent.numel())
    ws = 1
    C = W + 2
    Pi = torch.randn((C, S), device=device, dtype=dtype) * 1.5 - 3.0
    family_idx = None
    family_indexed_consts = False

    if per_clade_constants:
        family_idx = torch.zeros((C,), device=device, dtype=torch.long)
        family_idx[ws:ws + W] = torch.arange(W, device=device, dtype=torch.long)
        family_indexed_consts = True
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
    Pibar_ref = _uniform_pibar_reference(Pi_W, mt, ancestors)
    Pi_ref = _wave_step_reference(
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
    wave_pibar_uniform_parent_fused(
        Pi,
        Pibar,
        ws,
        W,
        S,
        mt,
        sp_parent,
        max_depth,
        family_idx=family_idx,
        family_indexed_consts=family_indexed_consts,
    )
    Pi_out = Pi.clone()
    wave_step_uniform_fused_into(
        Pi,
        Pi_out,
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
        family_idx=family_idx,
        family_indexed_consts=family_indexed_consts,
    )
    Pi_fused = Pi_out[ws:ws + W]
    torch.cuda.synchronize()

    torch.testing.assert_close(Pibar[ws:ws + W], Pibar_ref, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(Pi_fused, Pi_ref, rtol=2e-5, atol=2e-5)

    significant = Pi_ref > -100.0
    max_diff = torch.abs(Pi_fused - Pi_W)[significant].max()
    expected_diff = torch.abs(Pi_ref - Pi_W)[significant].max()
    torch.testing.assert_close(max_diff, expected_diff, rtol=2e-5, atol=2e-5)


def test_wave_step_uniform_local_input_rows_match_global_offset():
    torch.manual_seed(17)
    device = torch.device("cuda")
    dtype = torch.float32
    sp_parent, sp_child1, sp_child2, _ancestors, *_rest = _balanced_species_tree(device)
    max_depth = _rest[-1]

    W = 4
    S = int(sp_parent.numel())
    ws = 3
    C = W + 6
    Pi = torch.randn((C, S), device=device, dtype=dtype) * 1.1 - 3.0
    mt = torch.randn((S,), device=device, dtype=dtype) * 0.1
    DL = torch.randn((S,), device=device, dtype=dtype) * 0.2 - 2.0
    Ebar = torch.randn((S,), device=device, dtype=dtype) * 0.2 - 1.5
    E = torch.randn((S,), device=device, dtype=dtype) * 0.2 - 2.5
    SL1 = torch.randn((S,), device=device, dtype=dtype) * 0.2 - 2.0
    SL2 = torch.randn((S,), device=device, dtype=dtype) * 0.2 - 2.0
    leaf_term = torch.randn((W, S), device=device, dtype=dtype) * 0.2 - 4.0
    dts = torch.randn((W, S), device=device, dtype=dtype) * 0.2 - 5.0

    Pi_out_global = Pi.clone()
    Pibar_global = torch.full_like(Pi, float("-inf"))
    row_max_global = torch.full((C,), float("-inf"), device=device, dtype=dtype)
    wave_step_uniform_fused_into(
        Pi,
        Pi_out_global,
        Pibar_global,
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
        dts,
        store_final_pibar=True,
        final_pibar_row_max=row_max_global,
    )

    Pi_out_local = Pi.clone()
    Pibar_local = torch.full_like(Pi, float("-inf"))
    row_max_local = torch.full((C,), float("-inf"), device=device, dtype=dtype)
    wave_step_uniform_fused_into(
        Pi[ws:ws + W].contiguous(),
        Pi_out_local,
        Pibar_local,
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
        dts,
        store_final_pibar=True,
        final_pibar_row_max=row_max_local,
        input_ws=0,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(
        Pi_out_local[ws:ws + W],
        Pi_out_global[ws:ws + W],
        rtol=2e-5,
        atol=2e-5,
    )
    torch.testing.assert_close(
        Pibar_local[ws:ws + W],
        Pibar_global[ws:ws + W],
        rtol=2e-5,
        atol=2e-5,
    )
    torch.testing.assert_close(
        row_max_local[ws:ws + W],
        row_max_global[ws:ws + W],
        rtol=2e-5,
        atol=2e-5,
    )


@pytest.mark.parametrize("leaf_logp_mode", ["shared", "genewise_scalar", "genewise_specieswise"])
def test_wave_step_uniform_leaf_index_logp_modes_match_dense_leaf_term(leaf_logp_mode):
    torch.manual_seed(13)
    device = torch.device("cuda")
    dtype = torch.float32
    (
        sp_parent,
        sp_child1,
        sp_child2,
        _ancestors,
        _ancestor_cols,
        _ancestor_csr_indptr,
        _ancestor_csr_indices,
        max_depth,
    ) = _balanced_species_tree(device)

    W = 4
    S = int(sp_parent.numel())
    G = 3
    ws = 2
    C = W + 4
    Pi = torch.randn((C, S), device=device, dtype=dtype) * 1.2 - 3.0
    mt = torch.randn((G, S), device=device, dtype=dtype) * 0.1
    DL = torch.randn((G, S), device=device, dtype=dtype) * 0.2 - 2.0
    Ebar = torch.randn((G, S), device=device, dtype=dtype) * 0.2 - 1.5
    E = torch.randn((G, S), device=device, dtype=dtype) * 0.2 - 2.5
    SL1 = torch.randn((G, S), device=device, dtype=dtype) * 0.2 - 2.0
    SL2 = torch.randn((G, S), device=device, dtype=dtype) * 0.2 - 2.0

    leaf_species_idx = torch.full((C,), -1, device=device, dtype=torch.long)
    leaf_species_idx[ws:ws + W] = torch.tensor([2, -1, 5, 1], device=device)
    family_idx = torch.tensor([0, 0, 0, 1, 2, 1, 2, 0], device=device, dtype=torch.long)

    if leaf_logp_mode == "shared":
        leaf_logp = torch.randn((S,), device=device, dtype=dtype) * 0.2 - 4.0
    elif leaf_logp_mode == "genewise_scalar":
        leaf_logp = torch.randn((G,), device=device, dtype=dtype) * 0.2 - 4.0
    else:
        leaf_logp = torch.randn((G, S), device=device, dtype=dtype) * 0.2 - 4.0

    leaf_term = torch.full((W, S), -1e30, device=device, dtype=dtype)
    for w in range(W):
        species = int(leaf_species_idx[ws + w])
        if species < 0:
            continue
        if leaf_logp_mode == "shared":
            leaf_term[w, species] = leaf_logp[species]
        elif leaf_logp_mode == "genewise_scalar":
            leaf_term[w, species] = leaf_logp[family_idx[ws + w]]
        else:
            leaf_term[w, species] = leaf_logp[family_idx[ws + w], species]

    Pibar_dense = torch.full_like(Pi, float("-inf"))
    Pi_dense_out = Pi.clone()
    wave_step_uniform_fused_into(
        Pi,
        Pi_dense_out,
        Pibar_dense,
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
        family_idx=family_idx,
        family_indexed_consts=True,
    )
    Pi_dense = Pi_dense_out[ws:ws + W]
    significant_dense = Pi_dense > -100.0
    max_diff_dense = torch.abs(Pi_dense - Pi[ws:ws + W])[significant_dense].max()

    Pibar_indexed = torch.full_like(Pi, float("-inf"))
    Pi_indexed_out = Pi.clone()
    wave_step_uniform_fused_into(
        Pi,
        Pi_indexed_out,
        Pibar_indexed,
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
        leaf_logp,
        leaf_species_idx=leaf_species_idx,
        leaf_logp=leaf_logp,
        family_idx=family_idx,
        family_indexed_consts=True,
    )
    Pi_indexed = Pi_indexed_out[ws:ws + W]
    significant_indexed = Pi_indexed > -100.0
    max_diff_indexed = torch.abs(Pi_indexed - Pi[ws:ws + W])[significant_indexed].max()
    torch.cuda.synchronize()

    torch.testing.assert_close(Pi_indexed, Pi_dense, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(max_diff_indexed, max_diff_dense, rtol=2e-5, atol=2e-5)


def test_dts_parent_reduced_skips_output_initialization_when_all_rows_covered():
    device = torch.device("cuda")
    dtype = torch.float32
    torch.manual_seed(17)
    (
        _sp_parent,
        sp_child1,
        sp_child2,
        _ancestors,
        _ancestor_cols,
        _ancestor_csr_indptr,
        _ancestor_csr_indices,
        _max_depth,
    ) = _balanced_species_tree(device)
    S = int(sp_child1.numel())
    C = 4
    W = 2
    Pi = torch.randn((C, S), device=device, dtype=dtype) * 0.2 - 2.0
    Pibar = torch.randn((C, S), device=device, dtype=dtype) * 0.2 - 2.5
    lefts = torch.tensor([2, 0], device=device, dtype=torch.long)
    rights = torch.tensor([3, 1], device=device, dtype=torch.long)
    parent_ids = torch.tensor([0, 1], device=device, dtype=torch.long)
    ge2_ptr = torch.tensor([0], device=device, dtype=torch.long)
    ge2_parent_ids = torch.empty((0,), device=device, dtype=torch.long)
    log_pD = torch.randn((S,), device=device, dtype=dtype) * 0.1 - 3.0
    log_pS = torch.randn((S,), device=device, dtype=dtype) * 0.1 - 2.0
    split_logp = torch.tensor([-0.25, -0.75], device=device, dtype=dtype)

    initialized = dts_fused_parent_reduced(
        Pi,
        Pibar,
        lefts,
        rights,
        sp_child1,
        sp_child2,
        log_pD,
        log_pS,
        split_logp,
        W,
        int(parent_ids.numel()),
        parent_ids,
        ge2_ptr,
        ge2_parent_ids,
        initialize_out=True,
    )
    uninitialized = dts_fused_parent_reduced(
        Pi,
        Pibar,
        lefts,
        rights,
        sp_child1,
        sp_child2,
        log_pD,
        log_pS,
        split_logp,
        W,
        int(parent_ids.numel()),
        parent_ids,
        ge2_ptr,
        ge2_parent_ids,
        initialize_out=False,
    )
    expected = _dts_eq1_reference(
        Pi,
        Pibar,
        lefts,
        rights,
        sp_child1,
        sp_child2,
        log_pD,
        log_pS,
        split_logp,
        parent_ids,
        W,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(initialized, expected, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(uninitialized, expected, rtol=1e-5, atol=1e-5)
