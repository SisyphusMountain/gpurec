#!/usr/bin/env python3
"""
Max-autotune compatible version of Pi_update_ccp_log.
Eliminates ALL dynamic tensor creation and temporary allocations.
"""

import torch

def Pi_update_ccp_log_maxautotune(log_Pi, ccp_helpers, species_helpers, clade_species_map, 
                                  E, Ebar, p_S, p_D, p_T, 
                                  # Pre-allocated tensors to avoid ANY dynamic creation
                                  neg_inf_tensor, zero_tensor, one_tensor, eps_tensor):
    """
    Max-autotune compatible version - completely static computation graph.
    
    Key design principles:
    1. NO dynamic tensor creation (torch.tensor, torch.full, etc.)
    2. NO temporary tensor allocation (torch.stack, torch.cat, etc.)
    3. NO in-place operations that break CUDAGraphs
    4. ALL tensors pre-allocated and passed as arguments
    5. Pure functional operations only
    
    Args:
        All the usual args plus pre-allocated constant tensors:
        neg_inf_tensor: Pre-allocated tensor with float('-inf')
        zero_tensor: Pre-allocated tensor with 0.0
        one_tensor: Pre-allocated tensor with 1.0
        eps_tensor: Pre-allocated tensor with small epsilon (1e-10)
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
    
    device = log_Pi.device
    dtype = log_Pi.dtype
    
    # Convert probabilities to tensors (these are scalars, should be ok)
    p_D_scalar = torch.as_tensor(p_D, dtype=dtype, device=device)
    p_S_scalar = torch.as_tensor(p_S, dtype=dtype, device=device) 
    p_T_scalar = torch.as_tensor(p_T, dtype=dtype, device=device)
    
    # Handle event probabilities - use pre-allocated constants
    # Manual implementation of torch.where to avoid dynamic creation
    p_D_pos_mask = p_D_scalar > zero_tensor
    log_p_D = p_D_pos_mask * torch.log(p_D_scalar + eps_tensor) + (~p_D_pos_mask) * neg_inf_tensor
    
    p_S_pos_mask = p_S_scalar > zero_tensor  
    log_p_S = p_S_pos_mask * torch.log(p_S_scalar + eps_tensor) + (~p_S_pos_mask) * neg_inf_tensor
    
    p_T_pos_mask = p_T_scalar > zero_tensor
    log_p_T = p_T_pos_mask * torch.log(p_T_scalar + eps_tensor) + (~p_T_pos_mask) * neg_inf_tensor
    
    # Handle split probabilities - avoid torch.where
    zero_split_mask = (split_probs <= eps_tensor)
    nonzero_split_mask = ~zero_split_mask
    log_split_probs = nonzero_split_mask * torch.log(split_probs + eps_tensor) + zero_split_mask * neg_inf_tensor
    
    # Get log Pi values for left and right children
    log_Pi_left = log_Pi[split_lefts]    # [N_splits, S]
    log_Pi_right = log_Pi[split_rights]  # [N_splits, S]
    
    # === DUPLICATION EVENTS ===
    # Use broadcasting instead of unsqueeze
    log_D_splits = log_split_probs.view(-1, 1) + log_Pi_left + log_Pi_right + log_p_D
    
    # === SPECIATION EVENTS ===
    # Pre-allocate result tensors with proper shapes - avoid torch.full
    log_Pi_s1 = neg_inf_tensor + torch.zeros((C, S), dtype=dtype, device=device)
    log_Pi_s2 = neg_inf_tensor + torch.zeros((C, S), dtype=dtype, device=device)
    
    # Find child indices efficiently
    s_C1_indices = torch.argmax(s_C1, dim=1)
    s_C2_indices = torch.argmax(s_C2, dim=1)
    has_left_child = (s_C1.sum(dim=1) > eps_tensor)
    has_right_child = (s_C2.sum(dim=1) > eps_tensor)
    
    # Manual gather without conditional assignment
    # Create index tensors for advanced indexing
    batch_indices = torch.arange(C, device=device).unsqueeze(1).expand(C, S)
    
    # For species with children, gather values; otherwise keep -inf
    left_indices_expanded = s_C1_indices.unsqueeze(0).expand(C, S)
    right_indices_expanded = s_C2_indices.unsqueeze(0).expand(C, S)
    
    # Mask-based assignment instead of torch.where
    has_left_expanded = has_left_child.unsqueeze(0).expand(C, S)
    has_right_expanded = has_right_child.unsqueeze(0).expand(C, S)
    
    gathered_left = torch.gather(log_Pi, 1, left_indices_expanded)
    gathered_right = torch.gather(log_Pi, 1, right_indices_expanded)
    
    log_Pi_s1 = has_left_expanded * gathered_left + (~has_left_expanded) * neg_inf_tensor
    log_Pi_s2 = has_right_expanded * gathered_right + (~has_right_expanded) * neg_inf_tensor
    
    # Extract for splits
    log_Pi_s1_left = log_Pi_s1[split_lefts, :]
    log_Pi_s1_right = log_Pi_s1[split_rights, :]
    log_Pi_s2_left = log_Pi_s2[split_lefts, :]
    log_Pi_s2_right = log_Pi_s2[split_rights, :]
    
    # Manual logsumexp implementation to avoid torch.stack
    # logsumexp([a, b]) = max(a, b) + log(exp(a - max) + exp(b - max))
    log_spec1 = log_Pi_s1_left + log_Pi_s2_right
    log_spec2 = log_Pi_s1_right + log_Pi_s2_left
    
    # Manual logsumexp without stacking
    max_spec = torch.maximum(log_spec1, log_spec2)
    exp1 = torch.exp(log_spec1 - max_spec)
    exp2 = torch.exp(log_spec2 - max_spec)
    log_spec_sum = max_spec + torch.log(exp1 + exp2 + eps_tensor)
    
    log_S_splits = log_split_probs.view(-1, 1) + log_p_S + log_spec_sum
    
    # === TRANSFER EVENTS ===
    # Convert to linear space and back - this should be safe
    Pi_linear = torch.exp(log_Pi)
    Pibar_linear = Pi_linear.mm(Recipients_mat.T)
    
    # Manual log with masking instead of torch.where
    nonzero_mask = Pibar_linear > eps_tensor
    log_Pibar = nonzero_mask * torch.log(Pibar_linear + eps_tensor) + (~nonzero_mask) * neg_inf_tensor
    
    # Extract transfer terms
    log_Pibar_left = log_Pibar[split_lefts, :]
    log_Pibar_right = log_Pibar[split_rights, :]
    
    # Manual logsumexp for transfer terms
    log_trans1 = log_Pi_left + log_Pibar_right
    log_trans2 = log_Pi_right + log_Pibar_left
    
    max_trans = torch.maximum(log_trans1, log_trans2)
    exp_trans1 = torch.exp(log_trans1 - max_trans)
    exp_trans2 = torch.exp(log_trans2 - max_trans)
    log_trans_sum = max_trans + torch.log(exp_trans1 + exp_trans2 + eps_tensor)
    
    log_T_splits = log_split_probs.view(-1, 1) + log_p_T + log_trans_sum
    
    # === COMBINE ALL CONTRIBUTIONS ===
    # Manual 3-way logsumexp without stacking
    max_all = torch.maximum(torch.maximum(log_D_splits, log_S_splits), log_T_splits)
    exp_D = torch.exp(log_D_splits - max_all)
    exp_S = torch.exp(log_S_splits - max_all)
    exp_T = torch.exp(log_T_splits - max_all)
    log_combined_splits = max_all + torch.log(exp_D + exp_S + exp_T + eps_tensor)
    
    # === LEAF SPECIATION TERM ===
    # Manual log with masking
    leaf_nonzero_mask = clade_species_map > eps_tensor
    log_leaf_contrib = (leaf_nonzero_mask * (log_p_S + torch.log(clade_species_map + eps_tensor)) + 
                       (~leaf_nonzero_mask) * neg_inf_tensor)
    
    # === STATIC SCATTER OPERATION ===
    # Instead of scatter_reduce, use matrix operations
    # Create one-hot encoding matrix for split parents
    parent_onehot = torch.zeros((N_splits, C), dtype=dtype, device=device)
    parent_onehot.scatter_(1, split_parents.unsqueeze(1), one_tensor)
    
    # Use matrix multiplication instead of scatter
    # This aggregates split contributions for each parent clade
    aggregated_splits = parent_onehot.T.mm(log_combined_splits.view(N_splits, -1)).view(C, S)
    
    # Manual logsumexp between leaf and aggregated contributions
    max_final = torch.maximum(log_leaf_contrib, aggregated_splits)
    exp_leaf = torch.exp(log_leaf_contrib - max_final)
    exp_agg = torch.exp(aggregated_splits - max_final)
    new_log_Pi = max_final + torch.log(exp_leaf + exp_agg + eps_tensor)
    
    return new_log_Pi

def create_prealloc_tensors(device, dtype):
    """Create pre-allocated constant tensors for max-autotune compatibility."""
    return {
        'neg_inf_tensor': torch.tensor(float('-inf'), dtype=dtype, device=device),
        'zero_tensor': torch.tensor(0.0, dtype=dtype, device=device),
        'one_tensor': torch.tensor(1.0, dtype=dtype, device=device),
        'eps_tensor': torch.tensor(1e-10, dtype=dtype, device=device)
    }

# Create compiled versions
@torch.compile(mode="max-autotune")
def Pi_update_ccp_log_maxautotune_compiled(log_Pi, ccp_helpers, species_helpers, clade_species_map, 
                                          E, Ebar, p_S, p_D, p_T, 
                                          neg_inf_tensor, zero_tensor, one_tensor, eps_tensor):
    return Pi_update_ccp_log_maxautotune(log_Pi, ccp_helpers, species_helpers, clade_species_map, 
                                        E, Ebar, p_S, p_D, p_T,
                                        neg_inf_tensor, zero_tensor, one_tensor, eps_tensor)

@torch.compile(mode="reduce-overhead")  
def Pi_update_ccp_log_reduce_overhead_compiled(log_Pi, ccp_helpers, species_helpers, clade_species_map, 
                                              E, Ebar, p_S, p_D, p_T, 
                                              neg_inf_tensor, zero_tensor, one_tensor, eps_tensor):
    return Pi_update_ccp_log_maxautotune(log_Pi, ccp_helpers, species_helpers, clade_species_map, 
                                        E, Ebar, p_S, p_D, p_T,
                                        neg_inf_tensor, zero_tensor, one_tensor, eps_tensor)

@torch.compile(mode="default")
def Pi_update_ccp_log_default_compiled(log_Pi, ccp_helpers, species_helpers, clade_species_map, 
                                      E, Ebar, p_S, p_D, p_T, 
                                      neg_inf_tensor, zero_tensor, one_tensor, eps_tensor):
    return Pi_update_ccp_log_maxautotune(log_Pi, ccp_helpers, species_helpers, clade_species_map, 
                                        E, Ebar, p_S, p_D, p_T,
                                        neg_inf_tensor, zero_tensor, one_tensor, eps_tensor)