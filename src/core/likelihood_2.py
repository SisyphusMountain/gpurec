import torch
from .core_fn.triton.scatter_lse import seg_logsumexp
from typing import Dict, Optional
import math


NEG_INF = float("-inf")

def gather_Pi_children(Pi, sp_P_idx, child_index):
    C, S = Pi.shape
    Pi_children = torch.full((C, 2*S), float("-inf"), device=Pi.device, dtype=Pi.dtype)  # [2C, S]
    values = torch.index_select(Pi, 1, child_index) # [C, N_internal_nodes]
    Pi_children.index_copy_(1, sp_P_idx, values)
    return Pi_children

def compute_DTS(log_pD, log_pT, log_pS, Pi_s12, Pi, Pibar, log_split_probs, split_leftrights_sorted, N_splits, S):
    Pi_leftright = torch.index_select(Pi, 0, split_leftrights_sorted) # [2 N_splits, S]
    Pi_left, Pi_right = torch.chunk(Pi_leftright, 2, dim=0)  # Each [N_splits, S]
    Pibar_leftrights = torch.index_select(Pibar, 0, split_leftrights_sorted) # [2 N_splits, S]
    Pibar_left, Pibar_right = torch.chunk(Pibar_leftrights, 2, dim=0)  # Each [N_splits, S]

    DTS = torch.empty((5, N_splits, S), device=Pi.device, dtype=Pi.dtype)
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

def compute_DTS_L(log_pD, log_pT, log_pS, Pi, Pibar, Pi_s12, E, Ebar, E_s1, E_s2, clade_species_map, log_2):
    DTS_L = torch.empty((6, Pi.shape[0], Pi.shape[1]), dtype=log_pD.dtype, device=log_pD.device)
    # DL
    DTS_L[0] = ((log_2 + log_pD + E).unsqueeze(0) + Pi)
    # TL
    DTS_L[1] = (log_pT + Pi + Ebar.unsqueeze(0))
    DTS_L[2] = (log_pT + Pibar + E.unsqueeze(0))
    # SL
    Pi_s1, Pi_s2 = torch.chunk(Pi_s12, 2, dim=1)  # Each [C, S]
    DTS_L[3] = ((log_pS + E_s2).unsqueeze(0) + Pi_s1)
    DTS_L[4] = ((log_pS + E_s1).unsqueeze(0) + Pi_s2)
    # leaf
    DTS_L[5] = (log_pS + clade_species_map)
    # DTS_L = DTS_L.contiguous()
    # DTS_L_term = lse6(DTS_L[0], DTS_L[1], DTS_L[2], DTS_L[3], DTS_L[4], DTS_L[5])
    DTS_L_term = torch.logsumexp(DTS_L, dim=0)
    return DTS_L_term

def get_log_params(theta):
    param_tensor = torch.zeros(4, device=theta.device, dtype=theta.dtype)
    param_tensor[1:] = theta
    return torch.log_softmax(param_tensor, dim=0)

def Pi_step(Pi, ccp_helpers, species_helpers, clade_species_map,
            E, Ebar, E_s1, E_s2, theta, log_2):
    # region helpers
    # Extract helpers (precomputed sorted splits and CSR pointers)
    split_leftrights_sorted = ccp_helpers['split_leftrights_sorted']
    log_split_probs = ccp_helpers['log_split_probs_sorted'].unsqueeze(1).contiguous()  # [N_splits, 1]
    seg_ptr = ccp_helpers['ptr']
    seg_parent_ids = ccp_helpers['seg_parent_ids']
    # Segment partition helpers: contiguous blocks [len>=2][len==1][len==0]
    num_segs_ge2 = ccp_helpers['num_segs_ge2']
    num_segs_eq1 = ccp_helpers['num_segs_eq1']
    # num_segs_eq0 = int(ccp_helpers.get('num_segs_eq0', 0))
    end_rows_ge2 = ccp_helpers['end_rows_ge2']
    ptr_ge2 = ccp_helpers['ptr_ge2']
    N_splits = ccp_helpers["N_splits"]
    sp_P_idx = species_helpers['s_P_indexes'] # index of parent for each internal node
    sp_c12_idx = species_helpers["s_C12_indexes"]
    Recipients_mat = species_helpers['Recipients_mat']
    C, S = Pi.shape
    # endregion helpers
    # Computing log event probabilities
    log_pS, log_pD, log_pT, log_pL = get_log_params(theta)

    Pi_s12 = gather_Pi_children(Pi, sp_P_idx, sp_c12_idx)  # [C, 2S]
    # Computing Pibar
    Pi_max = torch.max(Pi, dim=1, keepdim=True).values
    Pi_linear = torch.exp(Pi - Pi_max)  # [C, S]
    Pibar_linear = Pi_linear.mm(Recipients_mat.T)  # [C, S]
    Pibar = torch.log(Pibar_linear) + Pi_max  # [C, S]


    DTS_term = compute_DTS(log_pD, log_pT, log_pS, Pi_s12, Pi, Pibar, log_split_probs, split_leftrights_sorted, N_splits, S)

    DTS_L_term = compute_DTS_L(log_pD, log_pT, log_pS, Pi, Pibar, Pi_s12, E, Ebar, E_s1, E_s2, clade_species_map, log_2)

    DTS_reduced = torch.full((C, S), NEG_INF, device=Pi.device, dtype=Pi.dtype)
    # clades with >=2 splits
    y_ge2 = seg_logsumexp(DTS_term[:end_rows_ge2], ptr_ge2)
    DTS_reduced.index_copy_(0, seg_parent_ids[:num_segs_ge2], y_ge2)
    # clades with exactly 1 split
    if num_segs_eq1 > 0:
        DTS_reduced.index_copy_(0,
                                seg_parent_ids[num_segs_ge2:num_segs_ge2+num_segs_eq1],
                                DTS_term[end_rows_ge2:end_rows_ge2+num_segs_eq1])

    return torch.logaddexp(DTS_reduced, DTS_L_term)

def E_step(E, sp_P_idx, sp_child12_idx, Recipients_mat, theta, return_components=False):
    log_pS, log_pD, log_pT, log_pL = get_log_params(theta)
    E_stack = torch.empty((4, E.shape[0]), dtype=E.dtype, device=E.device)
    # S
    E_s12 = gather_E_children(E, sp_P_idx, sp_child12_idx)
    E_s1, E_s2 = torch.chunk(E_s12, 2, dim=0)  # Each [S]
    E_stack[0] = log_pS + E_s1 + E_s2
    # D
    E_stack[1] = log_pD + 2 * E
    # T
    max_E = torch.max(E)
    Ebar = torch.log(torch.mv(Recipients_mat, torch.exp(E - max_E))) + max_E
    E_stack[2] = log_pT + E + Ebar
    # L
    E_stack[3] = log_pL

    E_new = torch.logsumexp(E_stack, dim=0)
    if return_components:
        return E_new, E_s1, E_s2, Ebar
    else:
        return E_new
    

def gather_E_children(E, sp_P_idx, child_index):
    E_child = torch.full((2 * E.shape[0],), NEG_INF, device=E.device, dtype=E.dtype)
    values = torch.index_select(E, 0, child_index)  # [N_internal_nodes]
    E_child.index_copy_(0, sp_P_idx, values)
    return E_child

def E_fixed_point(species_helpers,
                          theta,
                          max_iters=100,
                          tolerance=1e-10,
                          return_components=False,
                          warm_start_E=None,
                          dtype=torch.float32,
                          device="cuda"):

    S = species_helpers['S']
    
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

def Pi_fixed_point(ccp_helpers,
                  species_helpers, 
                  clade_species_map,
                  E,
                  Ebar,
                  E_s1,
                  E_s2,
                  theta,
                  max_iters=100,
                  tolerance=1e-10,
                  warm_start_Pi=None):
    C = ccp_helpers['C']
    S = clade_species_map.shape[1]
    device = clade_species_map.device
    dtype = clade_species_map.dtype
    
    # Initialize Pi matrix or use warm start
    if warm_start_Pi is not None:
        Pi = warm_start_Pi
    else:
        # Initialize like the working test: -log(2) for all entries, then set leaf probabilities
        Pi = torch.full((C, S), -math.log(100), dtype=dtype, device=device)
        
        # Set leaf probabilities based on clade-species mapping (convert from log space)
        # BUG: change this weird part
        for c in range(C):
            # Find mapped species (where clade_species_map is not -inf)
            finite_mask = torch.isfinite(clade_species_map[c])
            if finite_mask.any():
                mapped_species = torch.nonzero(finite_mask, as_tuple=False).flatten()
                # Uniform distribution among mapped species
                log_prob = torch.clamp(-torch.log(torch.tensor(len(mapped_species), dtype=dtype, device=device)), min=-0.001)
                Pi[c, mapped_species] = log_prob

    converged_iter = max_iters
    log_2 = torch.tensor([math.log(2)], dtype=dtype, device=device)

    for iteration in range(max_iters):
        Pi_new = Pi_step(
            Pi, ccp_helpers, species_helpers, clade_species_map,
            E, Ebar, E_s1, E_s2, theta, log_2
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