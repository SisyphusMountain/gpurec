#!/usr/bin/env python3
"""
Non-compiled version of Pi_update_ccp_log for gradient computation compatibility.
"""

import torch

def Pi_update_ccp_log_no_compile(log_Pi, ccp_helpers, species_helpers, clade_species_map, 
                                  E, Ebar, p_S, p_D, p_T):
    """
    Log-space version of Pi_update_ccp_parallel to handle numerical instability.
    This version removes @torch.compile() for compatibility with autograd.
    
    Args:
        log_Pi: Log probabilities matrix [C, S] in log space
        ccp_helpers: Dictionary with split information
        species_helpers: Species tree information  
        clade_species_map: Mapping matrix [C, S]
        E, Ebar: Extinction probabilities
        p_S, p_D, p_T: Event probabilities
        
    Returns:
        new_log_Pi: Updated log probabilities [C, S]
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
    
    # Convert probabilities to log space, handling zeros carefully with masking
    device = log_Pi.device
    dtype = log_Pi.dtype
    
    # Handle event probabilities - if any are exactly 0, set to -inf in log space
    # Ensure probabilities are tensors while preserving gradients
    if not isinstance(p_D, torch.Tensor):
        p_D_tensor = torch.tensor(p_D, dtype=dtype, device=device)
    else:
        p_D_tensor = p_D.to(dtype=dtype, device=device)
    
    if not isinstance(p_S, torch.Tensor):
        p_S_tensor = torch.tensor(p_S, dtype=dtype, device=device)
    else:
        p_S_tensor = p_S.to(dtype=dtype, device=device)
    
    if not isinstance(p_T, torch.Tensor):
        p_T_tensor = torch.tensor(p_T, dtype=dtype, device=device)
    else:
        p_T_tensor = p_T.to(dtype=dtype, device=device)
    
    log_p_D = torch.where(p_D_tensor > 0, torch.log(p_D_tensor), torch.tensor(float('-inf'), dtype=dtype, device=device))
    log_p_S = torch.where(p_S_tensor > 0, torch.log(p_S_tensor), torch.tensor(float('-inf'), dtype=dtype, device=device))
    log_p_T = torch.where(p_T_tensor > 0, torch.log(p_T_tensor), torch.tensor(float('-inf'), dtype=dtype, device=device))
    
    # Handle split probabilities - these might be exactly 0
    # If split_probs[i] = 0, then contribution should be 0 (= -inf in log space)
    zero_split_mask = (split_probs == 0)
    log_split_probs = torch.where(
        zero_split_mask,
        split_probs.new_full(split_probs.shape, float('-inf')),
        torch.log(split_probs)
    )
    
    # Get log Pi values for left and right children
    log_Pi_left = log_Pi[split_lefts]    # [N_splits, S]
    log_Pi_right = log_Pi[split_rights]  # [N_splits, S]
    
    # === DUPLICATION EVENTS ===
    # log(split_probs * Pi_left * Pi_right * p_D)
    # = log(split_probs) + log(Pi_left) + log(Pi_right) + log(p_D)
    log_D_splits = (log_split_probs.unsqueeze(1) + 
                    log_Pi_left + log_Pi_right + log_p_D)  # [N_splits, S]
    
    # === SPECIATION EVENTS ===
    # Compute log(Pi.mm(s_C1.T)) and log(Pi.mm(s_C2.T)) efficiently
    # Since s_C1, s_C2 are sparse binary matrices, use efficient operations
    
    # For binary matrices, Pi.mm(s_C1.T) is just gathering Pi values
    # s_C1[i,j] = 1 means j is left child of i
    # So Pi.mm(s_C1.T)[c,i] = Pi[c, left_child(i)]
    
    # Find child indices for each species (more memory efficient)
    s_C1_indices = torch.argmax(s_C1, dim=1)  # [S] - left child index for each species
    s_C2_indices = torch.argmax(s_C2, dim=1)  # [S] - right child index for each species
    
    # Check if species has children (binary matrix means max will be 1 if child exists, 0 otherwise)
    has_left_child = (s_C1.sum(dim=1) > 0)  # [S]
    has_right_child = (s_C2.sum(dim=1) > 0)  # [S]
    
    # Gather Pi values for children efficiently
    log_Pi_s1 = torch.full((C, S), float('-inf'), dtype=dtype, device=device)
    log_Pi_s2 = torch.full((C, S), float('-inf'), dtype=dtype, device=device)
    
    # Only compute for species that have children
    if has_left_child.any():
        valid_species = torch.nonzero(has_left_child, as_tuple=False).flatten()
        child_indices = s_C1_indices[valid_species]
        log_Pi_s1[:, valid_species] = log_Pi[:, child_indices]
        
    if has_right_child.any():
        valid_species = torch.nonzero(has_right_child, as_tuple=False).flatten()
        child_indices = s_C2_indices[valid_species]
        log_Pi_s2[:, valid_species] = log_Pi[:, child_indices]
    
    # Extract for splits
    log_Pi_s1_left = log_Pi_s1[split_lefts, :]   # [N_splits, S]
    log_Pi_s1_right = log_Pi_s1[split_rights, :] # [N_splits, S]
    log_Pi_s2_left = log_Pi_s2[split_lefts, :]   # [N_splits, S]
    log_Pi_s2_right = log_Pi_s2[split_rights, :] # [N_splits, S]
    
    # Speciation: log(p_S * split_probs * (Pi_s1_left * Pi_s2_right + Pi_s1_right * Pi_s2_left))
    # = log(p_S) + log(split_probs) + log(Pi_s1_left * Pi_s2_right + Pi_s1_right * Pi_s2_left)
    
    # Use logsumexp for the addition: log(a + b) = logsumexp([log(a), log(b)])
    log_spec1 = log_Pi_s1_left + log_Pi_s2_right  # [N_splits, S]
    log_spec2 = log_Pi_s1_right + log_Pi_s2_left  # [N_splits, S]
    log_spec_sum = torch.logsumexp(torch.stack([log_spec1, log_spec2], dim=0), dim=0)  # [N_splits, S]
    
    log_S_splits = (log_split_probs.unsqueeze(1) + log_p_S + log_spec_sum)  # [N_splits, S]
    
    # === TRANSFER EVENTS ===
    # log(split_probs * Pi_left * (Recipients_mat @ Pi_right) * p_T)
    
    # Use efficient computation similar to original implementation
    # First compute Pibar = Pi.mm(Recipients_mat.T) in log space more efficiently
    
    # Convert to linear space for the matrix multiplication, then back to log space
    Pi_linear = torch.exp(log_Pi)  # [C, S]
    Pibar_linear = Pi_linear.mm(Recipients_mat.T)  # [C, S]
    
    # Convert back to log space with proper masking for zeros
    log_Pibar = torch.where(
        Pibar_linear > 0, 
        torch.log(Pibar_linear), 
        Pibar_linear.new_full(Pibar_linear.shape, float('-inf'))
    )  # [C, S]
    
    # Extract transfer terms for splits
    log_Pibar_left = log_Pibar[split_lefts, :]   # [N_splits, S]
    log_Pibar_right = log_Pibar[split_rights, :] # [N_splits, S]
    
    # Transfer: log(p_T * split_probs * (Pi_left * Pibar_right + Pi_right * Pibar_left))
    log_trans1 = log_Pi_left + log_Pibar_right  # [N_splits, S]
    log_trans2 = log_Pi_right + log_Pibar_left  # [N_splits, S]
    log_trans_sum = torch.logsumexp(torch.stack([log_trans1, log_trans2], dim=0), dim=0)  # [N_splits, S]
    
    log_T_splits = (log_split_probs.unsqueeze(1) + log_p_T + log_trans_sum)  # [N_splits, S]
    
    # === COMBINE ALL CONTRIBUTIONS ===
    # Stack all contributions and use logsumexp to add them
    all_log_contribs = torch.stack([log_D_splits, log_S_splits, log_T_splits], dim=0)  # [3, N_splits, S]
    log_combined_splits = torch.logsumexp(all_log_contribs, dim=0)  # [N_splits, S]
    
    # === ADD LEAF SPECIATION TERM ===
    # In the original: new_Pi += p_S * clade_species_map
    # In log space: log(new_Pi + p_S * clade_species_map)
    
    # Convert clade_species_map to log space
    log_leaf_contrib = torch.where(
        clade_species_map > 0,
        log_p_S + torch.log(clade_species_map),
        clade_species_map.new_full(clade_species_map.shape, float('-inf'))
    )
    
    # === SCATTER TO PARENT CLADES USING LOGSUMEXP ===
    # Initialize result with leaf contributions
    new_log_Pi = log_leaf_contrib.clone()
    
    if N_splits > 0:
        # Use the log-sum-exp trick for scatter operations
        split_parents_expanded = split_parents.unsqueeze(1).expand(-1, S)  # [N_splits, S]
        
        # Find maximum for each parent clade (including leaf contributions for stability)
        max_vals = log_leaf_contrib.clone()
        max_vals.scatter_reduce_(0, split_parents_expanded, log_combined_splits, reduce='amax')
        
        # Normalize leaf contributions
        normalized_leaf = torch.exp(log_leaf_contrib - max_vals)
        
        # Normalize split contributions
        normalized_splits = torch.exp(log_combined_splits - 
                                     torch.gather(max_vals, 0, split_parents_expanded))
        
        # Sum all contributions
        sum_contribs = normalized_leaf.clone()
        sum_contribs.scatter_add_(0, split_parents_expanded, normalized_splits)
        
        # Take log and add back the maximum
        new_log_Pi = torch.where(
            sum_contribs > 0,
            torch.log(sum_contribs) + max_vals,
            clade_species_map.new_full(clade_species_map.shape, float('-inf'))
        )
    
    # Additional check for NaN values
    if torch.isnan(new_log_Pi).any():
        print(f"WARNING: NaN detected in Pi update!")
        print(f"  max_vals range: {max_vals.min():.6f} to {max_vals.max():.6f}")
        print(f"  sum_contribs range: {sum_contribs.min():.6f} to {sum_contribs.max():.6f}")
        print(f"  NaN in max_vals: {torch.isnan(max_vals).sum()}")
        print(f"  NaN in sum_contribs: {torch.isnan(sum_contribs).sum()}")
    
    return new_log_Pi