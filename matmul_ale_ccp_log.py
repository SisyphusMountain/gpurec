#!/usr/bin/env python3
"""
Log-space version of the GPU-parallelized CCP reconciliation algorithm.
This version computes log(Pi) to mitigate numerical instability and underflow.
"""

import sys
import time
import torch
import argparse
from tabulate import tabulate

# Import the original CCP functions
from matmul_ale_ccp import (
    build_ccp_from_single_tree, build_species_helpers, 
    build_clade_species_mapping, build_ccp_helpers,
    get_root_clade_id, E_step
)

#@torch.compile()
def Pi_update_ccp_log(log_Pi, ccp_helpers, species_helpers, clade_species_map, 
                      E, Ebar, p_S, p_D, p_T):
    """
    Log-space version of Pi_update_ccp_parallel to handle numerical instability.
    
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
    
    # === ADD LOSS TERMS ===
    # These are critical and were missing in the original implementation!
    # Need to add: D_loss + S_loss + T_loss to new_log_Pi
    
    # First, we need p_L for loss probability
    p_L = 1.0 - p_S_tensor - p_D_tensor - p_T_tensor
    log_p_L = torch.where(p_L > 0, torch.log(p_L), torch.tensor(float('-inf'), dtype=dtype, device=device))
    
    # Convert E and Ebar to log space if they aren't already tensors
    if not isinstance(E, torch.Tensor):
        E_tensor = torch.tensor(E, dtype=dtype, device=device)
    else:
        E_tensor = E.to(dtype=dtype, device=device)
        
    if not isinstance(Ebar, torch.Tensor):
        Ebar_tensor = torch.tensor(Ebar, dtype=dtype, device=device)
    else:
        Ebar_tensor = Ebar.to(dtype=dtype, device=device)
    
    # Handle E and Ebar in log space
    log_E = torch.where(E_tensor > 0, torch.log(E_tensor), torch.tensor(float('-inf'), dtype=dtype, device=device))
    log_Ebar = torch.where(Ebar_tensor > 0, torch.log(Ebar_tensor), torch.tensor(float('-inf'), dtype=dtype, device=device))
    
    # Compute E_s1 and E_s2 (needed for speciation loss)
    E_s1 = torch.mv(s_C1, E_tensor)  # [S]
    E_s2 = torch.mv(s_C2, E_tensor)  # [S]
    log_E_s1 = torch.where(E_s1 > 0, torch.log(E_s1), torch.tensor(float('-inf'), dtype=dtype, device=device))
    log_E_s2 = torch.where(E_s2 > 0, torch.log(E_s2), torch.tensor(float('-inf'), dtype=dtype, device=device))
    
    # Get species leaf mask
    sp_leaves_mask = species_helpers.get('sp_leaves_mask', torch.zeros(S, dtype=torch.bool, device=device))
    
    # === DUPLICATION LOSS ===
    # D_loss = 2 * p_D * Pi * E
    # In log space: log(2) + log(p_D) + log(Pi) + log(E)
    log_2 = torch.log(torch.tensor(2.0, dtype=dtype, device=device))
    log_D_loss = log_2 + log_p_D + log_Pi + log_E.unsqueeze(0)  # [C, S]
    
    # === SPECIATION LOSS ===
    # S_loss = p_S * (Pi_s1 * E_s2 + Pi_s2 * E_s1) * (~sp_leaves_mask)
    # First compute the two terms
    log_S_term1 = log_Pi_s1 + log_E_s2.unsqueeze(0)  # [C, S]
    log_S_term2 = log_Pi_s2 + log_E_s1.unsqueeze(0)  # [C, S]
    
    # Combine using logsumexp
    log_S_sum = torch.logsumexp(torch.stack([log_S_term1, log_S_term2], dim=0), dim=0)  # [C, S]
    
    # Apply species internal mask and p_S
    internal_mask = ~sp_leaves_mask
    log_S_loss = torch.where(
        internal_mask.unsqueeze(0),
        log_p_S + log_S_sum,
        torch.tensor(float('-inf'), dtype=dtype, device=device)
    )  # [C, S]
    
    # === TRANSFER LOSS ===
    # T_loss = p_T * (Pi * Ebar + Pibar * E)
    log_T_term1 = log_Pi + log_Ebar.unsqueeze(0)  # [C, S]
    log_T_term2 = log_Pibar + log_E.unsqueeze(0)  # [C, S]
    
    # Combine using logsumexp
    log_T_sum = torch.logsumexp(torch.stack([log_T_term1, log_T_term2], dim=0), dim=0)  # [C, S]
    log_T_loss = log_p_T + log_T_sum  # [C, S]
    
    # === COMBINE ALL CONTRIBUTIONS INCLUDING LOSSES ===
    # We need to add new_log_Pi + D_loss + S_loss + T_loss
    # In log space, this is logsumexp of all terms
    all_terms = [new_log_Pi, log_D_loss, log_S_loss, log_T_loss]
    
    # Filter out any all-inf slices to avoid numerical issues
    valid_terms = []
    for term in all_terms:
        if torch.isfinite(term).any():
            valid_terms.append(term)
    
    if valid_terms:
        new_log_Pi = torch.logsumexp(torch.stack(valid_terms, dim=0), dim=0)  # [C, S]
    
    # Additional check for NaN values
    if torch.isnan(new_log_Pi).any():
        print(f"WARNING: NaN detected in Pi update!")
        if 'max_vals' in locals():
            print(f"  max_vals range: {max_vals.min():.6f} to {max_vals.max():.6f}")
        if 'sum_contribs' in locals():
            print(f"  sum_contribs range: {sum_contribs.min():.6f} to {sum_contribs.max():.6f}")
        print(f"  NaN count: {torch.isnan(new_log_Pi).sum()}")
        print(f"  -inf count: {torch.isinf(new_log_Pi).sum()}")
    
    return new_log_Pi

def reconcile_ccp_log(species_tree_path, gene_tree_path, delta=1e-10, tau=0.05, lambda_param=1e-10, 
                      iters=100, device=None, dtype=torch.float64):
    """
    Log-space CCP reconciliation to handle numerical instability.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"🧮 Log-Space CCP Reconciliation")
    print(f"Device: {device}")
    print(f"Parameters: δ={delta}, τ={tau}, λ={lambda_param}")
    print()
    
    # Build CCP from gene tree
    print("📊 Building CCP...")
    ccp = build_ccp_from_single_tree(gene_tree_path)
    print(f"   └─ {len(ccp.clades)} clades, {len(ccp.splits)} split groups")
    
    # Build species helpers
    print("🌳 Building species helpers...")
    species_helpers = build_species_helpers(species_tree_path, device, dtype)
    print(f"   └─ {species_helpers['S']} species nodes")
    
    # Build clade-species mapping
    print("🗺️  Building clade-species mapping...")
    clade_species_map = build_clade_species_mapping(ccp, species_helpers, device, dtype)
    
    # Build CCP helpers for GPU parallelization
    print("⚡ Building CCP helpers...")
    ccp_helpers = build_ccp_helpers(ccp, device, dtype)
    total_splits = sum(len(splits) for splits in ccp.splits.values())
    print(f"   └─ {len(ccp_helpers['split_parents'])} vectorized splits")
    
    # Compute event probabilities
    rates_sum = 1.0 + delta + tau + lambda_param
    p_S = 1.0 / rates_sum
    p_D = delta / rates_sum
    p_T = tau / rates_sum
    p_L = lambda_param / rates_sum
    
    print(f"Event probabilities: S={p_S:.6f}, D={p_D:.6f}, T={p_T:.6f}, L={p_L:.6f}")
    
    # Initialize matrices
    S = species_helpers["S"]
    C = len(ccp.clades)
    E = torch.zeros(S, dtype=dtype, device=device)
    
    # Initialize log_Pi in log space (start with very small probabilities)
    log_Pi = torch.full((C, S), float('-inf'), dtype=dtype, device=device)
    
    # Set leaf probabilities based on clade-species mapping
    # If a leaf clade maps to a species, set log probability to 0 (probability = 1)
    leaf_count = 0
    for c in range(C):
        clade = ccp.id_to_clade[c]
        if clade.is_leaf():
            leaf_count += 1
            # Find which species this leaf belongs to
            mapped_species = torch.nonzero(clade_species_map[c] > 0, as_tuple=False).flatten()
            if len(mapped_species) > 0:
                # Distribute probability uniformly among mapped species
                log_prob = -torch.log(torch.tensor(len(mapped_species), dtype=dtype))
                log_Pi[c, mapped_species] = log_prob
                if leaf_count <= 3:  # Debug first few leaves
                    print(f"   Leaf {c} ({clade}): mapped to species {mapped_species.tolist()}, log_prob={log_prob:.6f}")
            else:
                if leaf_count <= 3:
                    print(f"   Leaf {c} ({clade}): NO MAPPING FOUND!")
    
    print(f"   Initialized {leaf_count} leaf clades")
    print(f"   Non-infinite log_Pi values after init: {torch.isfinite(log_Pi).sum()}")
    
    print(f"Matrix dimensions: Pi = {C} × {S} = {C*S:,} elements")
    print()
    
    # === EXTINCTION PROBABILITY COMPUTATION ===
    print("💀 Computing extinction probabilities...")
    for iter_e in range(iters):
        E_next, E_s1, E_s2, Ebar = E_step(E, species_helpers["s_C1"], species_helpers["s_C2"], 
                                          species_helpers["Recipients_mat"], p_S, p_D, p_T, p_L)
        E = E_next
    print(f"   └─ Converged after {iters} iterations")
    
    # === LIKELIHOOD COMPUTATION ===
    print("🧮 Computing likelihood matrix...")
    for iter_pi in range(iters):
        log_Pi_new = Pi_update_ccp_log(log_Pi, ccp_helpers, species_helpers, clade_species_map, 
                                       E, Ebar, p_S, p_D, p_T)
        
        # Check convergence in log space
        if iter_pi > 0:
            diff = torch.abs(log_Pi_new - log_Pi).max()
            if diff < 1e-10:
                print(f"   └─ Converged after {iter_pi+1} iterations")
                break
        
        log_Pi = log_Pi_new
    
    # === COMPUTE LOG-LIKELIHOOD ===
    root_clade_id = get_root_clade_id(ccp)
    root_clade = ccp.id_to_clade[root_clade_id]
    
    print(f"🌲 Root clade: {root_clade}")
    
    # Compute log-likelihood: log(sum(Pi[root, :]))
    log_likelihood = torch.logsumexp(log_Pi[root_clade_id, :], dim=0)
    
    print(f"📊 Log-likelihood: {log_likelihood:.6f}")
    
    # Check for numerical issues
    print(f"   Root log Pi values: min={log_Pi[root_clade_id, :].min():.6f}, max={log_Pi[root_clade_id, :].max():.6f}")
    print(f"   Non-finite values in log_Pi: {torch.logical_not(torch.isfinite(log_Pi)).sum()}")
    print(f"   NaN values in log_Pi: {torch.isnan(log_Pi).sum()}")
    print(f"   -inf values in log_Pi: {torch.isinf(log_Pi).sum()}")
    
    if torch.isnan(log_likelihood) or torch.isinf(log_likelihood):
        print("⚠️  WARNING: Numerical instability detected!")
        
        # Debug: check leaf initialization
        leaf_indices = [i for i in range(len(ccp.id_to_clade)) if ccp.id_to_clade[i].is_leaf()]
        print(f"   Leaf clades: {len(leaf_indices)}")
        for i in leaf_indices[:3]:  # Show first 3 leaves
            print(f"     Leaf {i}: {ccp.id_to_clade[i]} -> log_Pi = {log_Pi[i, :].max():.6f}")
        
    else:
        print("✅ No numerical instability detected")
    
    return {
        'log_likelihood': float(log_likelihood),
        'log_Pi': log_Pi,
        'ccp': ccp,
        'E': E
    }

def main():
    parser = argparse.ArgumentParser(description='Log-space CCP reconciliation for numerical stability')
    parser.add_argument('--species', required=True, help='Species tree file (.nwk)')
    parser.add_argument('--gene', required=True, help='Gene tree file (.nwk)')
    parser.add_argument('--delta', type=float, default=1e-10, help='Duplication rate (default: 1e-10)')
    parser.add_argument('--tau', type=float, default=0.05, help='Transfer rate (default: 0.05)')
    parser.add_argument('--lambda', dest='lambda_param', type=float, default=1e-10, help='Loss rate (default: 1e-10)')
    parser.add_argument('--iters', type=int, default=100, help='Maximum iterations (default: 100)')
    parser.add_argument('--device', choices=['cpu', 'cuda'], help='Device to use (default: auto)')
    
    args = parser.parse_args()
    
    device = None
    if args.device:
        device = torch.device(args.device)
    
    try:
        result = reconcile_ccp_log(
            args.species, args.gene,
            delta=args.delta, tau=args.tau, lambda_param=args.lambda_param,
            iters=args.iters, device=device
        )
        
        print(f"\n🎯 Final Results:")
        print(f"   Log-likelihood: {result['log_likelihood']:.6f}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
    