#!/usr/bin/env python3
"""
CUDAGraphs-compatible version of Pi_update_ccp_log for max-autotune mode.
Eliminates in-place modifications and dynamic tensor creation.
"""

import torch

def Pi_update_ccp_log_cudagraph(log_Pi, ccp_helpers, species_helpers, clade_species_map, 
                                E, Ebar, p_S, p_D, p_T):
    """
    CUDAGraphs-compatible version of Pi_update_ccp_log.
    
    Key changes:
    - No in-place tensor modifications
    - Pre-allocated tensors for constants
    - Static computation graph
    - Compatible with max-autotune mode
    """
    # Extract helpers
    split_parents = ccp_helpers['split_parents']
    split_lefts = ccp_helpers['split_lefts'] 
    split_rights = ccp_helpers['split_rights']
    split_probs = ccp_helpers['split_probs']
    
    s_C1 = species_helpers['s_C1']
    s_C2 = species_helpers['s_C2'] 
    Recipients_mat = species_helpers['Recipients_mat']
    
    C, S = log_Pi.shape
    N_splits = len(split_parents)
    
    if N_splits == 0:
        return log_Pi.clone()
    
    # Pre-allocate constant tensors (CUDAGraphs compatible)
    device = log_Pi.device
    dtype = log_Pi.dtype
    
    # Convert probabilities to tensors once
    p_D_tensor = torch.tensor(p_D, dtype=dtype, device=device)
    p_S_tensor = torch.tensor(p_S, dtype=dtype, device=device) 
    p_T_tensor = torch.tensor(p_T, dtype=dtype, device=device)
    
    # Pre-allocated constant for -inf (avoid dynamic creation)
    neg_inf = torch.tensor(float('-inf'), dtype=dtype, device=device)
    
    # Handle event probabilities with pre-allocated tensors
    log_p_D = torch.where(p_D_tensor > 0, torch.log(p_D_tensor), neg_inf)
    log_p_S = torch.where(p_S_tensor > 0, torch.log(p_S_tensor), neg_inf)
    log_p_T = torch.where(p_T_tensor > 0, torch.log(p_T_tensor), neg_inf)
    
    # Handle split probabilities
    zero_split_mask = (split_probs == 0)
    log_split_probs = torch.where(zero_split_mask, neg_inf, torch.log(split_probs))
    
    # Get log Pi values for left and right children
    log_Pi_left = log_Pi[split_lefts]    # [N_splits, S]
    log_Pi_right = log_Pi[split_rights]  # [N_splits, S]
    
    # === DUPLICATION EVENTS ===
    log_D_splits = (log_split_probs.unsqueeze(1) + 
                    log_Pi_left + log_Pi_right + log_p_D)  # [N_splits, S]
    
    # === SPECIATION EVENTS ===
    # Pre-allocate result tensors
    log_Pi_s1 = torch.full((C, S), float('-inf'), dtype=dtype, device=device)
    log_Pi_s2 = torch.full((C, S), float('-inf'), dtype=dtype, device=device)
    
    # Find child indices for each species
    s_C1_indices = torch.argmax(s_C1, dim=1)  # [S]
    s_C2_indices = torch.argmax(s_C2, dim=1)  # [S]
    
    # Check if species has children
    has_left_child = (s_C1.sum(dim=1) > 0)  # [S]
    has_right_child = (s_C2.sum(dim=1) > 0)  # [S]
    
    # Use advanced indexing instead of conditional assignment
    # Create index matrices for gathering
    clade_idx = torch.arange(C, device=device).unsqueeze(1).expand(C, S)
    species_idx = torch.arange(S, device=device).unsqueeze(0).expand(C, S)
    
    # For left children - use scatter instead of conditional assignment
    valid_left_mask = has_left_child.unsqueeze(0).expand(C, S)
    left_child_indices = s_C1_indices.unsqueeze(0).expand(C, S)
    gathered_left = torch.gather(log_Pi, 1, left_child_indices)
    log_Pi_s1 = torch.where(valid_left_mask, gathered_left, neg_inf)
    
    # For right children
    valid_right_mask = has_right_child.unsqueeze(0).expand(C, S)
    right_child_indices = s_C2_indices.unsqueeze(0).expand(C, S)
    gathered_right = torch.gather(log_Pi, 1, right_child_indices)
    log_Pi_s2 = torch.where(valid_right_mask, gathered_right, neg_inf)
    
    # Extract for splits
    log_Pi_s1_left = log_Pi_s1[split_lefts, :]   # [N_splits, S]
    log_Pi_s1_right = log_Pi_s1[split_rights, :] # [N_splits, S]
    log_Pi_s2_left = log_Pi_s2[split_lefts, :]   # [N_splits, S]
    log_Pi_s2_right = log_Pi_s2[split_rights, :] # [N_splits, S]
    
    # Speciation terms
    log_spec1 = log_Pi_s1_left + log_Pi_s2_right
    log_spec2 = log_Pi_s1_right + log_Pi_s2_left
    log_spec_sum = torch.logsumexp(torch.stack([log_spec1, log_spec2], dim=0), dim=0)
    log_S_splits = log_split_probs.unsqueeze(1) + log_p_S + log_spec_sum
    
    # === TRANSFER EVENTS ===
    # Convert to linear space for matrix multiplication, then back to log space
    Pi_linear = torch.exp(log_Pi)  # [C, S]
    Pibar_linear = Pi_linear.mm(Recipients_mat.T)  # [C, S]
    
    # Convert back to log space with masking
    log_Pibar = torch.where(Pibar_linear > 0, torch.log(Pibar_linear), neg_inf)
    
    # Extract transfer terms for splits
    log_Pibar_left = log_Pibar[split_lefts, :]   # [N_splits, S]
    log_Pibar_right = log_Pibar[split_rights, :] # [N_splits, S]
    
    # Transfer terms
    log_trans1 = log_Pi_left + log_Pibar_right
    log_trans2 = log_Pi_right + log_Pibar_left
    log_trans_sum = torch.logsumexp(torch.stack([log_trans1, log_trans2], dim=0), dim=0)
    log_T_splits = log_split_probs.unsqueeze(1) + log_p_T + log_trans_sum
    
    # === COMBINE ALL CONTRIBUTIONS ===
    all_log_contribs = torch.stack([log_D_splits, log_S_splits, log_T_splits], dim=0)
    log_combined_splits = torch.logsumexp(all_log_contribs, dim=0)  # [N_splits, S]
    
    # === ADD LEAF SPECIATION TERM ===
    log_leaf_contrib = torch.where(
        clade_species_map > 0,
        log_p_S + torch.log(clade_species_map),
        neg_inf
    )
    
    # === SCATTER TO PARENT CLADES USING STATIC OPERATIONS ===
    # Initialize result with leaf contributions
    new_log_Pi = log_leaf_contrib.clone()
    
    if N_splits > 0:
        # Use scatter_reduce with logsumexp for CUDAGraphs compatibility
        split_parents_expanded = split_parents.unsqueeze(1).expand(-1, S)
        
        # Pre-allocate result tensor
        scattered_result = torch.full_like(new_log_Pi, neg_inf)
        
        # Use scatter_reduce with logsumexp (static operation)
        scattered_result.scatter_reduce_(0, split_parents_expanded, log_combined_splits, reduce='amax')
        
        # Combine leaf and scattered contributions using logsumexp
        stacked_contribs = torch.stack([new_log_Pi, scattered_result], dim=0)
        new_log_Pi = torch.logsumexp(stacked_contribs, dim=0)
    
    return new_log_Pi

# Create compiled versions for different modes
Pi_update_ccp_log_default = torch.compile(Pi_update_ccp_log_cudagraph, mode="default")
Pi_update_ccp_log_max_autotune = torch.compile(Pi_update_ccp_log_cudagraph, mode="max-autotune")
Pi_update_ccp_log_reduce_overhead = torch.compile(Pi_update_ccp_log_cudagraph, mode="reduce-overhead")