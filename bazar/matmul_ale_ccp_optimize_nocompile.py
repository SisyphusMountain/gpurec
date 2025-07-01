#!/usr/bin/env python3
"""
Gradient descent optimization WITHOUT torch.compile to test gradient computation.
"""

import torch
import torch.nn.functional as F
from torch.autograd.functional import vjp
import time
import numpy as np
from typing import Dict, Tuple, Optional, Any

# Import existing CCP functions but avoid compiled version
from matmul_ale_ccp import (
    build_ccp_from_single_tree, build_species_helpers,
    build_clade_species_mapping, build_ccp_helpers,
    get_root_clade_id, E_step
)

def Pi_update_ccp_log_nocompile(log_Pi, ccp_helpers, species_helpers, clade_species_map, 
                               E, Ebar, p_S, p_D, p_T):
    """
    Log-space version of Pi_update_ccp_parallel WITHOUT torch.compile.
    """
    # Copy the implementation from matmul_ale_ccp_log.py but without @torch.compile decorator
    
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
    p_D_tensor = torch.tensor(p_D, dtype=dtype, device=device)
    p_S_tensor = torch.tensor(p_S, dtype=dtype, device=device) 
    p_T_tensor = torch.tensor(p_T, dtype=dtype, device=device)
    
    log_p_D = torch.where(p_D_tensor > 0, torch.log(p_D_tensor), torch.tensor(float('-inf'), dtype=dtype, device=device))
    log_p_S = torch.where(p_S_tensor > 0, torch.log(p_S_tensor), torch.tensor(float('-inf'), dtype=dtype, device=device))
    log_p_T = torch.where(p_T_tensor > 0, torch.log(p_T_tensor), torch.tensor(float('-inf'), dtype=dtype, device=device))
    
    # Handle split probabilities - these might be exactly 0
    zero_split_mask = (split_probs == 0)
    log_split_probs = torch.where(
        zero_split_mask,
        torch.tensor(float('-inf'), dtype=split_probs.dtype, device=split_probs.device),
        torch.log(split_probs)
    )
    
    # Get log Pi values for left and right children
    log_Pi_left = log_Pi[split_lefts]    # [N_splits, S]
    log_Pi_right = log_Pi[split_rights]  # [N_splits, S]
    
    # === DUPLICATION EVENTS ===
    log_D_splits = (log_split_probs.unsqueeze(1) + 
                    log_Pi_left + log_Pi_right + log_p_D)  # [N_splits, S]
    
    # === SPECIATION EVENTS ===
    # Find child indices for each species 
    s_C1_indices = torch.argmax(s_C1, dim=1)  # [S] - left child index for each species
    s_C2_indices = torch.argmax(s_C2, dim=1)  # [S] - right child index for each species
    
    # Check if species has children
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
    
    # Speciation: use logsumexp for the addition
    log_spec1 = log_Pi_s1_left + log_Pi_s2_right  # [N_splits, S]
    log_spec2 = log_Pi_s1_right + log_Pi_s2_left  # [N_splits, S]
    log_spec_sum = torch.logsumexp(torch.stack([log_spec1, log_spec2], dim=0), dim=0)  # [N_splits, S]
    
    log_S_splits = (log_split_probs.unsqueeze(1) + log_p_S + log_spec_sum)  # [N_splits, S]
    
    # === TRANSFER EVENTS ===
    # Convert to linear space for the matrix multiplication, then back to log space
    Pi_linear = torch.exp(log_Pi)  # [C, S]
    Pibar_linear = Pi_linear.mm(Recipients_mat.T)  # [C, S]
    
    # Convert back to log space with proper masking for zeros
    log_Pibar = torch.where(
        Pibar_linear > 0, 
        torch.log(Pibar_linear), 
        torch.tensor(float('-inf'), dtype=dtype, device=device)
    )  # [C, S]
    
    # Extract transfer terms for splits
    log_Pibar_left = log_Pibar[split_lefts, :]   # [N_splits, S]
    log_Pibar_right = log_Pibar[split_rights, :] # [N_splits, S]
    
    # Transfer: use logsumexp for addition
    log_trans1 = log_Pi_left + log_Pibar_right  # [N_splits, S]
    log_trans2 = log_Pi_right + log_Pibar_left  # [N_splits, S]
    log_trans_sum = torch.logsumexp(torch.stack([log_trans1, log_trans2], dim=0), dim=0)  # [N_splits, S]
    
    log_T_splits = (log_split_probs.unsqueeze(1) + log_p_T + log_trans_sum)  # [N_splits, S]
    
    # === COMBINE ALL CONTRIBUTIONS ===
    all_log_contribs = torch.stack([log_D_splits, log_S_splits, log_T_splits], dim=0)  # [3, N_splits, S]
    log_combined_splits = torch.logsumexp(all_log_contribs, dim=0)  # [N_splits, S]
    
    # === ADD LEAF SPECIATION TERM ===
    log_leaf_contrib = torch.where(
        clade_species_map > 0,
        log_p_S + torch.log(clade_species_map),
        torch.tensor(float('-inf'), dtype=log_Pi.dtype, device=log_Pi.device)
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
            torch.tensor(float('-inf'), dtype=log_Pi.dtype, device=log_Pi.device)
        )
    
    return new_log_Pi

def test_gradient_without_compile():
    """Test gradient computation without torch.compile."""
    print("🧪 Testing gradient computation WITHOUT torch.compile...")
    
    from matmul_ale_ccp_optimize import CCPOptimizer, compute_event_probabilities
    
    # Create optimizer but we'll use the non-compiled version
    optimizer = CCPOptimizer(
        "test_trees_1/sp.nwk", "test_trees_1/g.nwk",
        initial_params=(0.1, 0.1, 0.1),
        device=torch.device("cpu"),
        dtype=torch.float64
    )
    
    # Test simple parameter gradient
    params = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float64, requires_grad=True)
    p_S, p_D, p_T, p_L = compute_event_probabilities(params)
    
    print(f"Event probabilities: S={p_S:.4f}, D={p_D:.4f}, T={p_T:.4f}, L={p_L:.4f}")
    
    # Compute E (fixed for testing)
    device = torch.device("cpu")
    dtype = torch.float64
    S = optimizer.species_helpers["S"]
    E = torch.zeros(S, dtype=dtype, device=device)
    
    for _ in range(10):
        E_next, E_s1, E_s2, Ebar = E_step(E, optimizer.species_helpers["s_C1"], 
                                          optimizer.species_helpers["s_C2"], 
                                          optimizer.species_helpers["Recipients_mat"], 
                                          float(p_S), float(p_D), float(p_T), float(p_L))
        E = E_next
    
    # Test Pi update with parameter gradients
    log_Pi = optimizer.log_Pi_init.clone()
    
    def pi_update_with_params(params_input):
        p_S_i, p_D_i, p_T_i, p_L_i = compute_event_probabilities(params_input)
        return Pi_update_ccp_log_nocompile(log_Pi, optimizer.ccp_helpers, optimizer.species_helpers, 
                                          optimizer.clade_species_map, E, Ebar, 
                                          float(p_S_i), float(p_D_i), float(p_T_i))
    
    # Forward pass
    log_Pi_new = pi_update_with_params(params)
    print(f"Pi update successful: shape {log_Pi_new.shape}")
    print(f"Finite values: {torch.isfinite(log_Pi_new).sum()} / {log_Pi_new.numel()}")
    
    # Test gradient
    objective = log_Pi_new.sum()
    print(f"Objective: {objective}")
    
    objective.backward()
    print(f"Gradient through Pi update: {params.grad}")
    print(f"Gradient norm: {params.grad.norm():.6f}")
    
    if params.grad.norm() > 1e-8:
        print("✅ Gradient computation successful!")
        return True
    else:
        print("❌ Gradient is still zero")
        return False

if __name__ == "__main__":
    print("🔧 Testing Gradient Without torch.compile")
    print("=" * 50)
    
    success = test_gradient_without_compile()
    
    if success:
        print("\n🎯 Solution found: torch.compile was breaking gradients!")
        print("Next steps: Update optimization code to use non-compiled version")
    else:
        print("\n🤔 torch.compile was not the issue, need deeper debugging")