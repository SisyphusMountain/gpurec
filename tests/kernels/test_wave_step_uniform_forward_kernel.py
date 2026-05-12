import pytest
import torch

from gpurec.core.kernels.wave_step import (
    wave_pibar_uniform_parent_fused,
    wave_step_uniform_fused_into,
)


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
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
