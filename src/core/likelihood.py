"""
Likelihood matrix computation for phylogenetic reconciliation.

This module implements the Pi matrix update for computing reconciliation
likelihoods in both linear and log space.

"""
import logging
import torch
from contextlib import contextmanager
from typing import Dict, Optional
from .core_fn.baselines.scatter_lse import ScatterLogSumExp

# # Import Triton kernels if available
# try:
#     from .triton.lse import lse4, lse5, lse7
#     from .triton.lse import lse5_triton_pair as lse5
#     from .triton.lse import lse7_triton_pair as lse
#     # 2D segmented logsumexp: y[g, s] = logsumexp(x[ptr[g]:ptr[g+1], s])
#     from triton.scatter_lse import seg_lse
#     TRITON_AVAILABLE = True
# except Exception:
#     TRITON_AVAILABLE = False
#     #logging.warning("Triton not available; falling back to PyTorch implementationssdfdffsd.")

NEG_INF = float('-inf')

@contextmanager
def _prof_range(name: str):
    """Context manager that adds both PyTorch record_function and NVTX ranges.

    - record_function labels show up in PyTorch profiler tables/trace
    - NVTX ranges are visible in Nsight Systems when using nsys with NVTX capture
    """
    nvtx_pushed = False
    if torch.cuda.is_available():
        try:
            torch.cuda.nvtx.range_push(name)
            nvtx_pushed = True
        except Exception:
            nvtx_pushed = False
    with torch.autograd.profiler.record_function(name):
        try:
            yield
        finally:
            if nvtx_pushed:
                try:
                    torch.cuda.nvtx.range_pop()
                except Exception:
                    pass
# TODO: try torch.compile
def compute_D(Pi_left, Pi_right, log_split_probs, log_pD):
    """
    input shapes: 
        Pi_left: [N_splits, S]
        Pi_right: [N_splits, S]
        log_split_probs: [N_splits]
        log_pD: [1]
    Formula:
        p_D * p(gamma', gamma''|gamma) * Pi_{gamma',e} * Pi_{gamma'',e}
    """
    return (log_split_probs + Pi_left + Pi_right + log_pD)

def compute_DL(Pi, E, log_pD, log_2):
    return ((log_2 + log_pD + E).unsqueeze(0) + Pi).contiguous()

def compute_T(log_split_probs_sorted, log_pT, Pi_left, Pi_right, Pibar_left, Pibar_right):
    pTsplitprobs = log_split_probs_sorted + log_pT
    T_term_1 = pTsplitprobs + Pi_left + Pibar_right
    T_term_2 = pTsplitprobs + Pi_right + Pibar_left
    return T_term_1, T_term_2

def compute_DTS(log_pD, log_pT, log_pS, Pi_s12, Pi_left, Pi_right, Pibar_right, Pibar_left, log_split_probs, split_leftrights_sorted, N_splits, S):
    DTS = torch.empty((5, N_splits, S))
    DTS[0] = log_pD + Pi_left + Pi_right
    DTS[1] = log_pT + Pi_left + Pibar_right
    DTS[2] = log_pT + Pi_right + Pibar_left

    Pi_s12_leftright = torch.index_select(Pi_s12, 0, split_leftrights_sorted) # [2 N_splits, 2 S]
    Pi_s1_left = Pi_s12_leftright[:N_splits, :S]
    Pi_s2_left = Pi_s12_leftright[:N_splits, S:]
    Pi_s1_right = Pi_s12_leftright[N_splits:, :S]
    Pi_s2_right = Pi_s12_leftright[N_splits:, S:]

    DTS[3] = log_pS + Pi_s1_left + Pi_s2_right
    DTS[4] = log_pS + Pi_s1_right + Pi_s2_left
    # DTS = DTS.contiguous()
    # DTS_term = log_split_probs + lse5(DTS[0], DTS[1], DTS[2], DTS[3], DTS[4])
    DTS_term = log_split_probs + torch.logsumexp(DTS, dim=0)
    return DTS_term

def compute_DTS_smallmem(log_pD, log_pT, log_pS, Pi_s12, Pi_left, Pi_right, Pibar_right, Pibar_left, log_split_probs, split_leftrights_sorted, N_splits, S):
    DTS = log_pD + Pi_left + Pi_right
    DTS = torch.logaddexp(DTS, log_pT + Pi_left + Pibar_right)
    DTS = torch.logaddexp(DTS, log_pT + Pi_right + Pibar_left)
    Pi_s12_leftright = torch.index_select(Pi_s12, 0, split_leftrights_sorted) # [2 N_splits, 2 S]
    Pi_s1_left = Pi_s12_leftright[:N_splits, :S]
    Pi_s2_left = Pi_s12_leftright[:N_splits, S:]
    Pi_s1_right = Pi_s12_leftright[N_splits:, :S]
    Pi_s2_right = Pi_s12_leftright[N_splits:, S:]
    DTS = torch.logaddexp(DTS, log_pS + Pi_s1_left + Pi_s2_right)
    DTS = torch.logaddexp(DTS, log_pS + Pi_s1_right + Pi_s2_left)
    DTS_term = log_split_probs + DTS
    return DTS_term

def compute_DTS_L(log_pD, log_pT, log_pS, Pi, Pibar, Pi_s1, Pi_s2, E, Ebar, E_s1, E_s2, clade_species_map, log_2):
    DTS_L = torch.empty((6, Pi.shape[0], Pi.shape[1]), dtype=log_pD.dtype, device=log_pD.device)
    # DL
    DTS_L[0] = ((log_2 + log_pD + E).unsqueeze(0) + Pi)
    # TL
    DTS_L[1] = (log_pT + Pi + Ebar.unsqueeze(0))
    DTS_L[2] = (log_pT + Pibar + E.unsqueeze(0))
    # SL
    DTS_L[3] = ((log_pS + E_s2).unsqueeze(0) + Pi_s1)
    DTS_L[4] = ((log_pS + E_s1).unsqueeze(0) + Pi_s2)
    # leaf
    DTS_L[5] = (log_pS + clade_species_map)
    # DTS_L = DTS_L.contiguous()
    # DTS_L_term = lse6(DTS_L[0], DTS_L[1], DTS_L[2], DTS_L[3], DTS_L[4], DTS_L[5])
    DTS_L_term = torch.logsumexp(DTS_L, dim=0)
    return DTS_L_term

def compute_DTS_L_smallmem(log_pD, log_pT, log_pS, Pi, Pibar, Pi_s1, Pi_s2, E, Ebar, E_s1, E_s2, clade_species_map, log_2):
    DTS_L = ((log_2 + log_pD + E).unsqueeze(0) + Pi)
    DTS_L = torch.logaddexp(DTS_L, (log_pT + Pi + Ebar.unsqueeze(0)))
    DTS_L = torch.logaddexp(DTS_L, (log_pT + Pibar + E.unsqueeze(0)))
    DTS_L = torch.logaddexp(DTS_L, ((log_pS + E_s2).unsqueeze(0) + Pi_s1))
    DTS_L = torch.logaddexp(DTS_L, ((log_pS + E_s1).unsqueeze(0) + Pi_s2))
    DTS_L = torch.logaddexp(DTS_L, (log_pS + clade_species_map))
    return DTS_L

# TODO: try torch.compile
def compute_S(Pi, sp_P_idx, sp_c12_idx, split_leftrights_sorted, log_split_probs_sorted, log_pS, E_s1, E_s2, clade_species_map):
    # region speciation
    with _prof_range("Pi_step:speciation_children"):
        Pi_s12 = gather_Pi_children(Pi, sp_P_idx, sp_c12_idx)  # [C, 2S]
        Pi_s1, Pi_s2 = torch.chunk(Pi_s12, 2, dim=1)  # Each [C, S]
        # each should be contiguous.
    # TODO: use only 1 gather and 1 index_select and 4 chunks. If there are contiguity issues with the addition
    # that comes after, we will try differently.
    # torch.chunk returns views, so it does not allocate new memory.
    # Extract for splits
    with _prof_range("Pi_step:speciation_splits"):
        Pi_s12_leftright = torch.index_select(Pi_s12, 0, split_leftrights_sorted) # [2 N_splits, 2 S]
        N_splits = Pi_s12_leftright.size(0) // 2
        Pi_s1_left = Pi_s12_leftright[:N_splits, :S]
        Pi_s2_left = Pi_s12_leftright[:N_splits, S:]
        Pi_s1_right = Pi_s12_leftright[N_splits:, :S]
        Pi_s2_right = Pi_s12_leftright[N_splits:, S:]
        # Pi_s1 = torch.index_select(Pi_s12, 0, split_lefts_sorted)   # [N_splits, S]
        # Pi_s2 = torch.index_select(Pi_s12, 0, split_rights_sorted)  # [N_splits, S]

    # both copies survive
    with _prof_range("Pi_step:speciation_both_survive"):
        S_term_1 = log_split_probs_sorted + log_pS + Pi_s1_left + Pi_s2_right  # [N_splits, S]
        S_term_2 = log_split_probs_sorted + log_pS + Pi_s1_right + Pi_s2_left  # [N_splits, S]

    
    # one copy goes extinct
    with _prof_range("Pi_step:speciation_one_extinct"):
        SL_term_1 = log_pS + Pi_s1 + E_s2.unsqueeze(0)  # [C, S]
        SL_term_2 = log_pS + Pi_s2 + E_s1.unsqueeze(0)  # [C, S]
        # For leaf speciation events, one copy doesn't really go extinct, but S event on leaves still gives only one copy, not two.
        # Therefore, the tensor clade_species_map has the same shape as Pi. [C, S]
        # TODO: maybe there's a more efficient way of obtaining the leaf contribution...
        log_leaf_contrib = log_pS + clade_species_map
    # endregion speciation
    return S_term_1, S_term_2, SL_term_1, SL_term_2, log_leaf_contrib


def gather_E_children(E, sp_P_idx, child_index):
    """Gather extinction probabilities for internal nodes"""
    with _prof_range("gather_E_children"):
        E_child = torch.full((2 * E.shape[0],), float("-inf"), device=E.device, dtype=E.dtype)
        values = torch.index_select(E, 0, child_index)  # [N_internal_nodes]
        E_child.index_copy_(0, sp_P_idx, values)
        return E_child
# TODO: try torch.compile
def gather_Pi_children(Pi, sp_P_idx, child_index):
    with _prof_range("gather_Pi_children"):
        C, S = Pi.shape
        Pi_children = torch.full((C, 2*S), float("-inf"), device=Pi.device, dtype=Pi.dtype)  # [2C, S]
        values = torch.index_select(Pi, 1, child_index) # [C, N_internal_nodes]
        Pi_children.index_copy_(1, sp_P_idx, values)
        return Pi_children
# TODO: try torch.compile
#@torch.compile
def get_log_params(theta):
    param_tensor = torch.zeros(4, device=theta.device, dtype=theta.dtype)
    param_tensor[1:] = theta
    # exp_params = torch.exp(param_tensor)
    # return torch.log(exp_params / exp_params.sum())
    return torch.log_softmax(param_tensor, dim=0)

def E_step(E, sp_P_idx, sp_child12_idx, Recipients_mat, theta,
           return_components=False, use_triton=True, compare_triton: bool = False):
    # theta is a tensor with the elements log_delta, log_tau, log_lambda
    # Ensure theta is on the same device and dtype as E
    # TODO: try get_log_params
    with _prof_range("E_step:event_probs"):
        log_pS, log_pD, log_pT, log_pL = get_log_params(theta)
    # TODO: gather both children by concatenating indices, then chunking the result
    # This would be more efficient on GPU, as it would reduce the number of gather operations
    # and allow for better memory coalescing.
    with _prof_range("E_step:gather_children"):
        E_s12 = gather_E_children(E, sp_P_idx, sp_child12_idx)
        E_s1, E_s2 = torch.chunk(E_s12, 2, dim=0)  # Each [S]
        # E_s1 = gather_E_children(E, sp_P_idx, sp_child1_idx, E_s1)
        # E_s2 = gather_E_children(E, sp_P_idx, sp_child2_idx, E_s2)

    with _prof_range("E_step:compose_terms"):
        speciation = log_pS + E_s1 + E_s2
        duplication = log_pD + 2 * E
    # TODO: write a Triton kernel for this
    with _prof_range("E_step:Ebar"):
        max_E = torch.max(E)
        Ebar = torch.log(torch.mv(Recipients_mat, torch.exp(E - max_E))) + max_E
        transfer = log_pT + E + Ebar
    # Use Triton kernel if available, enabled, and on CUDA
    with _prof_range("E_step:reduce"):
        log_pL_expanded = log_pL * torch.ones_like(E)
        if TRITON_AVAILABLE and use_triton and speciation.is_cuda:
            new_tr = lse4(speciation, duplication, transfer, log_pL_expanded)
            if compare_triton:
                ref = torch.logsumexp(torch.stack([speciation, duplication, transfer, log_pL_expanded], dim=0), dim=0)
                rtol = 1e-12 if new_tr.dtype == torch.float64 else 1e-6
                atol = rtol
                if not torch.allclose(new_tr, ref, rtol=rtol, atol=atol):
                    diff = (new_tr - ref).abs().max().item()
                    print(f"[E_step:reduce] Triton vs Torch mismatch: max_abs={diff:.3e}")
            new_E = new_tr
        else:
            ref = torch.logsumexp(
                torch.stack([speciation, duplication, transfer, log_pL_expanded], dim=0), dim=0
            )
            if compare_triton and TRITON_AVAILABLE and speciation.is_cuda:
                new_tr = lse4(speciation, duplication, transfer, log_pL_expanded)
                rtol = 1e-12 if ref.dtype == torch.float64 else 1e-6
                atol = rtol
                if not torch.allclose(new_tr, ref, rtol=rtol, atol=atol):
                    diff = (new_tr - ref).abs().max().item()
                    print(f"[E_step:reduce] Torch vs Triton mismatch: max_abs={diff:.3e}")
            new_E = ref
    

    if return_components:
        return new_E, E_s1, E_s2, Ebar
    else:
        return new_E

def Pi_step(Pi, ccp_helpers, species_helpers, clade_species_map,
            E, Ebar, E_s1, E_s2, theta, log_2, use_triton=True, compare_triton: bool = False):
    """
    Log-space version of Pi_update_ccp_parallel to handle numerical instability.
    
    Args:
        Pi: Log probabilities matrix [C, S] in log space
        ccp_helpers: Dictionary with split information
        species_helpers: Species tree information  
        clade_species_map: Mapping matrix [C, S]
        E, Ebar: Extinction probabilities
        log_pS, log_pD, log_pT: Event probabilities in log space
        debug: If True, log tensor statistics for debugging
        
    Returns:
        new_Pi: Updated log probabilities [C, S]
    """
    with _prof_range("Pi_step:event_probs"):
        log_pS, log_pD, log_pT, log_pL = get_log_params(theta)
    # region helpers
    # Extract helpers (precomputed sorted splits and CSR pointers)
    split_parents_sorted = ccp_helpers['split_parents_sorted']
    # split_lefts_sorted = ccp_helpers['split_lefts_sorted']
    # split_rights_sorted = ccp_helpers['split_rights_sorted']
    split_leftrights_sorted = ccp_helpers['split_leftrights_sorted']
    log_split_probs_sorted = ccp_helpers['log_split_probs_sorted'].unsqueeze(1).contiguous()  # [N_splits, 1]
    seg_ptr = ccp_helpers['ptr']
    seg_parent_ids = ccp_helpers['seg_parent_ids']
    # Segment partition helpers: contiguous blocks [len>=2][len==1][len==0]
    num_segs_ge2 = int(ccp_helpers.get('num_segs_ge2', 0))
    num_segs_eq1 = int(ccp_helpers.get('num_segs_eq1', 0))
    # num_segs_eq0 = int(ccp_helpers.get('num_segs_eq0', 0))
    end_rows_ge2 = int(ccp_helpers.get('end_rows_ge2', 0))
    ptr_ge2 = ccp_helpers.get('ptr_ge2', seg_ptr[:1])
    N_splits = ccp_helpers["N_splits"]

    sp_P_idx = species_helpers['s_P_indexes'] # index of parent for each internal node
    # sp_c1_idx = species_helpers['s_C1_indexes'] # index of first child for each internal node
    # sp_c2_idx = species_helpers['s_C2_indexes'] # index of second child for each internal node
    sp_c12_idx = species_helpers["s_C12_indexes"]
    Recipients_mat = species_helpers['Recipients_mat']
    C, S = Pi.shape
    # endregion helpers

    # TODO: gather both children by concatenating indices, then chunking the result
    # Get log Pi values for left and right children
    with _prof_range("Pi_step:prepare_splits"):
        Pi_leftright = torch.index_select(Pi, 0, split_leftrights_sorted)
        Pi_left, Pi_right = torch.chunk(Pi_leftright, 2, dim=0)  # Each [N_splits, S]
        # Pi_left = torch.index_select(Pi, 0, split_lefts_sorted)    # [N_splits, S]
        # Pi_right = torch.index_select(Pi, 0, split_rights_sorted)  # [N_splits, S]
    # region duplication
    # both copies survive
    with _prof_range("Pi_step:duplication"):
        D_term, DL_term = compute_D(Pi, E, Pi_left, Pi_right, log_split_probs_sorted, log_pD, log_2)
        

    # endregion duplication
    
    S_term_1, S_term_2, SL_term_1, SL_term_2, log_leaf_contrib = compute_S(Pi,
                                                                           sp_P_idx,
                                                                           sp_c12_idx,
                                                                           split_leftrights_sorted,
                                                                           log_split_probs_sorted,
                                                                           log_pS,
                                                                           E_s1,
                                                                           E_s2,
                                                                           clade_species_map,
                                                                        )

    # region transfer
    # TODO: create a Triton kernel for this
    # both copies survive
    with _prof_range("Pi_step:Pibar"):
        Pi_max = torch.max(Pi, dim=1, keepdim=True).values
        Pi_linear = torch.exp(Pi - Pi_max)  # [C, S]
        Pibar_linear = Pi_linear.mm(Recipients_mat.T)  # [C, S]
        Pibar = torch.log(Pibar_linear) + Pi_max  # [C, S]
    
    # Extract transfer terms for splits
    Pibar_leftrights = torch.index_select(Pibar, 0, split_leftrights_sorted) # [2 N_splits, S]
    Pibar_left, Pibar_right = torch.chunk(Pibar_leftrights, 2, dim=0)  # Each [N_splits, S]
    # Pibar_left = torch.index_select(Pibar, 0, split_lefts_sorted)   # [N_splits, S]
    # Pibar_right = torch.index_select(Pibar, 0, split_rights_sorted) # [N_splits, S]
    
    # Transfer: log(p_T * split_probs * (Pi_left * Pibar_right + Pi_right * Pibar_left))
    with _prof_range("Pi_step:transfer_both_survive"):
        T_term_1 = log_split_probs_sorted + log_pT + Pi_left + Pibar_right  # [N_splits, S]
        T_term_2 = log_split_probs_sorted + log_pT + Pi_right + Pibar_left  # [N_splits, S]

    # only one copy survives
    with _prof_range("Pi_step:transfer_one_extinct"):
        TL_term_1 = log_pT + Pi + Ebar.unsqueeze(0)  # [C, S]
        TL_term_2 = log_pT + Pibar + E.unsqueeze(0)  # [C, S]
    # endregion transfer



        
    # === COMBINE ALL CONTRIBUTIONS WITHOUT LOSSES ===
    # Stack all contributions and use logsumexp to add them
    # TODO: avoid doing logsumexp if we need the exp directly after in scatterlogsumexp
    with _prof_range("Pi_step:combine_split_terms"):
        survive_terms = lse5_triton_pair(D_term,
                                               S_term_1,
                                               S_term_2,
                                               T_term_1,
                                               T_term_2) # [N_splits, S]
        
    # Aggregation across splits per parent clade
    with _prof_range("Pi_step:scatter_splits"):
        if TRITON_AVAILABLE and use_triton and survive_terms.is_cuda:
            # Use precomputed CSR pointers over the already-sorted splits
            x_sorted = survive_terms

            # Preallocate output mapped to original parents; default to -inf (covers empty segments and leaves)
            contribs_1_tr = torch.full((C, S), NEG_INF, dtype=x_sorted.dtype, device=x_sorted.device)

            # Segments with len >= 2: run Triton once on the contiguous block, then index_copy to parent ids
            if num_segs_ge2 > 0 and end_rows_ge2 > 0:
                y_ge2 = seg_logsumexp(x_sorted[:end_rows_ge2], ptr_ge2)
                contribs_1_tr.index_copy_(0, seg_parent_ids[:num_segs_ge2], y_ge2)

            # Singleton segments (len == 1): copy the corresponding rows directly
            if num_segs_eq1 > 0:
                start = end_rows_ge2
                stop = end_rows_ge2 + num_segs_eq1
                contribs_1_tr.index_copy_(0, seg_parent_ids[num_segs_ge2:num_segs_ge2 + num_segs_eq1], x_sorted[start:stop])

            # No need to write len==0 segments: they remain -inf in contribs_1_tr

            if compare_triton:
                leaves_mask = torch.isfinite(clade_species_map).any(dim=1)
                contribs_1_ref = ScatterLogSumExp.apply(survive_terms, split_parents_sorted, C, leaves_mask)
                rtol = 1e-12 if contribs_1_tr.dtype == torch.float64 else 1e-6
                atol = rtol
                if not torch.allclose(contribs_1_tr, contribs_1_ref, rtol=rtol, atol=atol):
                    diff = (contribs_1_tr - contribs_1_ref).abs().max().item()
                    print(f"[Pi_step:scatter] Triton vs Torch mismatch: max_abs={diff:.3e}")
            D_T_S_term = contribs_1_tr
            # [N_splits, S] -> [C, S] by summing over splits for each parent clade
        else:
            # Proven-stable autograd path with sorted parents
            leaves_mask = torch.isfinite(clade_species_map).any(dim=1)
            contribs_1_ref = ScatterLogSumExp.apply(survive_terms, split_parents_sorted, C, leaves_mask)
            if compare_triton and TRITON_AVAILABLE and survive_terms.is_cuda:
                # Run Triton path for comparison only
                x_sorted = survive_terms
                contribs_1_tr = torch.full((C, S), NEG_INF, dtype=x_sorted.dtype, device=x_sorted.device)
                if num_segs_ge2 > 0 and end_rows_ge2 > 0:
                    y_ge2 = seg_logsumexp(x_sorted[:end_rows_ge2], ptr_ge2)
                    contribs_1_tr.index_copy_(0, seg_parent_ids[:num_segs_ge2], y_ge2)
                if num_segs_eq1 > 0:
                    start = end_rows_ge2
                    stop = end_rows_ge2 + num_segs_eq1
                    contribs_1_tr.index_copy_(0, seg_parent_ids[num_segs_ge2:num_segs_ge2 + num_segs_eq1], x_sorted[start:stop])
                rtol = 1e-12 if contribs_1_ref.dtype == torch.float64 else 1e-6
                atol = rtol
                if not torch.allclose(contribs_1_tr, contribs_1_ref, rtol=rtol, atol=atol):
                    diff = (contribs_1_tr - contribs_1_ref).abs().max().item()
                    print(f"[Pi_step:scatter] Torch vs Triton mismatch: max_abs={diff:.3e}")
            D_T_S_term = contribs_1_ref
    
    # === COMBINE ALL CONTRIBUTIONS INCLUDING LOSSES ===
    with _prof_range("Pi_step:final_reduce"):
        # Use Triton kernel if available, enabled, and on CUDA
        if TRITON_AVAILABLE and use_triton and D_T_S_term.is_cuda:
            new_tr = lse7_triton_pair(D_T_S_term, DL_term, SL_term_1, SL_term_2, 
                                      log_leaf_contrib, TL_term_1, TL_term_2)
            if compare_triton:
                all_terms = [D_T_S_term, DL_term, SL_term_1, SL_term_2, log_leaf_contrib, TL_term_1, TL_term_2]
                ref = torch.logsumexp(torch.stack(all_terms, dim=0), dim=0)
                rtol = 1e-12 if new_tr.dtype == torch.float64 else 1e-6
                atol = rtol
                if not torch.allclose(new_tr, ref, rtol=rtol, atol=atol):
                    diff = (new_tr - ref).abs().max().item()
                    print(f"[Pi_step:final_reduce] Triton vs Torch mismatch: max_abs={diff:.3e}")
            new_Pi = new_tr
        else:
            all_terms = [D_T_S_term, DL_term, SL_term_1, SL_term_2, log_leaf_contrib, TL_term_1, TL_term_2]
            ref = torch.logsumexp(torch.stack(all_terms, dim=0), dim=0)  # [C, S]
            if compare_triton and TRITON_AVAILABLE and D_T_S_term.is_cuda:
                new_tr = lse7_triton_pair(D_T_S_term,
                                          DL_term,
                                          SL_term_1,
                                          SL_term_2, 
                                          TL_term_1,
                                          TL_term_2,
                                          log_leaf_contrib)
                rtol = 1e-12 if ref.dtype == torch.float64 else 1e-6
                atol = rtol
                if not torch.allclose(new_tr, ref, rtol=rtol, atol=atol):
                    diff = (new_tr - ref).abs().max().item()
                    print(f"[Pi_step:final_reduce] Torch vs Triton mismatch: max_abs={diff:.3e}")
            new_Pi = ref


    
    return new_Pi

def E_fixed_point(species_helpers: Dict,
                          theta: torch.Tensor,
                          max_iters: int = 100,
                          tolerance: float = 1e-10,
                          return_components: bool = False,
                          warm_start_E=None,
                          use_triton: bool = True,
                          compare_triton: bool = False) -> Dict:
    """
    Compute extinction probabilities via fixed-point iteration in log space.
    
    Args:
        species_helpers: Dictionary with species tree information
        theta: 3-element tensor [log_delta, log_tau, log_lambda] (unconstrained)
        max_iters: Maximum iterations for convergence
        tolerance: Convergence tolerance
        return_components: If True, return intermediate components
        
    Returns:
        Dictionary with:
            - E: Converged log extinction probabilities [S]
            - E_s1, E_s2, E_bar: Component values (if requested)
            - iterations: Number of iterations to convergence
    """
    S = species_helpers['S']
    device = species_helpers['s_C1'].device
    dtype = species_helpers['s_C1'].dtype
    
    # Initialize with log(0.5), or use a warm-start value if available
    if warm_start_E is not None:
        E = warm_start_E
    else:
        E = torch.full((S,), -0.69, dtype=dtype, device=device) # use log(0.5) as initial log-probs (within the unit $\ell^\infty$-ball where the map is contracting)
    E_s1 = torch.full_like(E, NEG_INF)
    E_s2 = torch.full_like(E, NEG_INF)

    converged_iter = max_iters
    for iteration in range(max_iters):
        result = E_step(
            E,
            species_helpers['s_P_indexes'],
            species_helpers['s_C12_indexes'],
            species_helpers['Recipients_mat'],
            theta,
            return_components=True,
            use_triton=use_triton,
            compare_triton=compare_triton
        )
        
        E_new = result[0]
        
        # Check convergence
        if torch.abs(E_new - E).max() < tolerance:
            converged_iter = iteration + 1
            E = E_new
            break
        
        E = E_new
    
    output = {
        'E': E,
        'iterations': converged_iter
    }
    
    if return_components:
        _, E_s1, E_s2, E_bar = result
        output.update({
            'E_s1': E_s1,
            'E_s2': E_s2,
            'E_bar': E_bar
        })
    
    return output

def Pi_fixed_point(ccp_helpers: Dict,
                  species_helpers: Dict, 
                  clade_species_map: torch.Tensor,
                  E: torch.Tensor,
                  Ebar: torch.Tensor,
                  E_s1: torch.Tensor,
                  E_s2: torch.Tensor,
                  theta: torch.Tensor,
                  max_iters: int = 100,
                  tolerance: float = 1e-10,
                  warm_start_Pi: Optional[torch.Tensor] = None,
                  use_triton: bool = True,
                  compare_triton: bool = False) -> Dict:
    """
    Compute Pi matrix via fixed-point iteration in log space.
    
    Args:
        ccp_helpers: Dictionary with CCP split information
        species_helpers: Dictionary with species tree information  
        clade_species_map: Mapping matrix [C, S]
        E, Ebar, E_s1, E_s2: Extinction probabilities (already computed)
        theta: 3-element tensor [log_delta, log_tau, log_lambda] (unconstrained)
        max_iters: Maximum iterations for convergence
        tolerance: Convergence tolerance
        warm_start_Pi: Optional starting Pi matrix for warm start
        
    Returns:
        Dictionary with:
            - Pi: Converged log Pi matrix [C, S]
            - iterations: Number of iterations to convergence
    """
    C = ccp_helpers['C']
    S = clade_species_map.shape[1]
    device = clade_species_map.device
    dtype = clade_species_map.dtype
    
    # Initialize Pi matrix or use warm start
    if warm_start_Pi is not None:
        Pi = warm_start_Pi
    else:
        # Initialize like the working test: -log(2) for all entries, then set leaf probabilities
        import math
        Pi = torch.full((C, S), -math.log(100), dtype=dtype, device=device)
        
        # Set leaf probabilities based on clade-species mapping (convert from log space)
        for c in range(C):
            # Find mapped species (where clade_species_map is not -inf)
            finite_mask = torch.isfinite(clade_species_map[c])
            if finite_mask.any():
                mapped_species = torch.nonzero(finite_mask, as_tuple=False).flatten()
                # Uniform distribution among mapped species
                log_prob = torch.clamp(-torch.log(torch.tensor(len(mapped_species), dtype=dtype, device=device)), min=-0.001)
                Pi[c, mapped_species] = log_prob

    converged_iter = max_iters
    log_2 = torch.log(torch.tensor(2.0, dtype=dtype, device=device))
    for iteration in range(max_iters):
        Pi_new = Pi_step(
            Pi, ccp_helpers, species_helpers, clade_species_map,
            E, Ebar, E_s1, E_s2, theta, log_2, use_triton=use_triton, compare_triton=compare_triton
        )
        
        # Check convergence
        if torch.abs(Pi_new - Pi).max() < tolerance:
            converged_iter = iteration + 1
            Pi = Pi_new
            break
        
        Pi = Pi_new
    
    output = {
        'Pi': Pi,
        'iterations': converged_iter
    }
    
    return output

def compute_log_likelihood(Pi, root_clade_idx):
    root_probs = Pi[root_clade_idx, :]
    return torch.logsumexp(root_probs, dim=0)

def debug_tensor(tensor, name, enabled=False):
    """Log tensor statistics if debugging is enabled"""
    if enabled:
        from ..utils.debug import tensor_stats
        import logging
        logger = logging.getLogger('pi_update_debug')
        logger.debug(tensor_stats(tensor, name))
