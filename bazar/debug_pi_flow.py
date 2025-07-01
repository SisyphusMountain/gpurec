#!/usr/bin/env python3
"""
Debug Pi matrix probability flow to identify why root clade gets zero probability.
"""

import sys
sys.path.append('.')

from matmul_ale_ccp import *
import torch

def debug_pi_flow():
    """Debug the Pi matrix updates to trace probability flow."""
    print("=== Pi Matrix Flow Debug ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    
    # Setup same as main_ccp
    ccp = build_ccp_from_single_tree("test_trees_1/g.nwk")
    species_helpers = build_species_helpers("test_trees_1/sp.nwk", device, dtype)
    clade_species_map = build_clade_species_mapping(ccp, species_helpers, device, dtype)
    
    # Use small rates for debugging
    delta, tau, lambda_param = 1e-10, 1e-10, 1e-10
    rates_sum = 1.0 + delta + tau + lambda_param
    p_S = 1.0 / rates_sum
    p_D = delta / rates_sum
    p_T = tau / rates_sum
    p_L = lambda_param / rates_sum
    
    print(f"Event probabilities: p_S={p_S:.6f}, p_D={p_D:.6f}, p_T={p_T:.6f}, p_L={p_L:.6f}")
    
    # Initialize E vector (extinction probabilities)
    S = species_helpers["S"]
    E = torch.zeros(S, dtype=dtype, device=device)
    
    # Compute extinction probabilities (just a few iterations)
    for iter_e in range(5):
        E_next, E_s1, E_s2, Ebar = E_step(E, species_helpers["s_C1"], species_helpers["s_C2"], 
                                          species_helpers["Recipients_mat"], p_S, p_D, p_T, p_L)
        E = E_next
    
    print(f"E vector (first 5): {E[:5]}")
    
    # Initialize Pi matrix
    C = len(ccp.clades)
    Pi = torch.zeros((C, S), dtype=dtype, device=device)
    root_clade_id = get_root_clade_id(ccp)
    ccp_helpers = build_ccp_helpers(ccp, device, dtype)
    
    print(f"\nInitial Pi matrix: shape={Pi.shape}, sum={Pi.sum():.6f}")
    
    # Examine leaf clade mappings first
    print(f"\nLeaf clade mappings (should be non-zero):")
    leaf_clades = []
    for clade_id, clade in ccp.id_to_clade.items():
        if clade.is_leaf():
            leaf_clades.append(clade_id)
            species_indices = torch.nonzero(clade_species_map[clade_id, :]).squeeze()
            if species_indices.numel() > 0:
                if species_indices.numel() == 1:
                    s_idx = species_indices.item()
                    value = clade_species_map[clade_id, s_idx].item()
                    print(f"  Clade {clade_id} ({clade}): species {s_idx}, value={value}")
                else:
                    for s_idx in species_indices:
                        s_idx = s_idx.item() if hasattr(s_idx, 'item') else s_idx
                        value = clade_species_map[clade_id, s_idx].item()
                        print(f"  Clade {clade_id} ({clade}): species {s_idx}, value={value}")
    
    # Run a few Pi updates with detailed tracing
    for iter_pi in range(3):
        print(f"\n--- Pi Update Iteration {iter_pi} ---")
        
        # Before update: check current Pi state
        pi_sum_before = Pi.sum().item()
        pi_max_before = Pi.max().item()
        root_pi_before = Pi[root_clade_id, :].sum().item()
        
        print(f"Before update: Pi sum={pi_sum_before:.6e}, max={pi_max_before:.6e}, root={root_pi_before:.6e}")
        
        # Check leaf clade Pi values
        leaf_pi_nonzero = 0
        for clade_id in leaf_clades:
            leaf_pi_sum = Pi[clade_id, :].sum().item()
            if leaf_pi_sum > 0:
                leaf_pi_nonzero += 1
            print(f"  Leaf clade {clade_id}: Pi sum = {leaf_pi_sum:.6e}")
        
        print(f"  Leaf clades with non-zero Pi: {leaf_pi_nonzero}/{len(leaf_clades)}")
        
        # Perform update
        Pi_new = Pi_update_ccp(Pi, ccp_helpers, species_helpers, clade_species_map, 
                              E, Ebar, E_s1, E_s2, p_S, p_D, p_T)
        
        # After update: check what changed
        pi_sum_after = Pi_new.sum().item()
        pi_max_after = Pi_new.max().item()
        root_pi_after = Pi_new[root_clade_id, :].sum().item()
        
        print(f"After update: Pi sum={pi_sum_after:.6e}, max={pi_max_after:.6e}, root={root_pi_after:.6e}")
        
        # Check which clades gained probability
        gained_prob = 0
        for clade_id in range(C):
            old_sum = Pi[clade_id, :].sum().item()
            new_sum = Pi_new[clade_id, :].sum().item()
            if new_sum > old_sum + 1e-15:  # Significant increase
                gained_prob += 1
                clade = ccp.id_to_clade[clade_id]  
                print(f"  Clade {clade_id} gained: {old_sum:.6e} -> {new_sum:.6e} ({clade.size} leaves)")
        
        print(f"  Clades that gained probability: {gained_prob}/{C}")
        
        Pi = Pi_new
        
        # Stop if we see no progress
        if pi_sum_after < 1e-12:
            print(f"  No significant probability accumulation, stopping")
            break
    
    return Pi, ccp, root_clade_id

def debug_single_update_terms():
    """Debug individual terms in the Pi update to see what contributes."""
    print("\n=== Single Pi Update Terms Debug ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    
    # Setup
    ccp = build_ccp_from_single_tree("test_trees_1/g.nwk")
    species_helpers = build_species_helpers("test_trees_1/sp.nwk", device, dtype)
    clade_species_map = build_clade_species_mapping(ccp, species_helpers, device, dtype)
    
    # Rates
    delta, tau, lambda_param = 1e-10, 1e-10, 1e-10
    rates_sum = 1.0 + delta + tau + lambda_param
    p_S = 1.0 / rates_sum
    p_D = delta / rates_sum
    p_T = tau / rates_sum
    p_L = lambda_param / rates_sum
    
    # E vector (converged)
    S = species_helpers["S"]
    E = torch.full((S,), 1e-10, dtype=dtype, device=device)  # Approximately converged value
    E_s1 = torch.mv(species_helpers["s_C1"], E)
    E_s2 = torch.mv(species_helpers["s_C2"], E)
    Ebar = torch.mv(species_helpers["Recipients_mat"], E)
    
    # Pi matrix (start from leaf mappings)
    C = len(ccp.clades)
    Pi = clade_species_map.clone()  # Start with leaf mappings as Pi_0
    
    ccp_helpers = build_ccp_helpers(ccp, device, dtype)
    
    print(f"Starting Pi sum: {Pi.sum():.6f} (from leaf mappings)")
    
    # Manually compute each term
    ccp_C1 = ccp_helpers["ccp_C1"]
    ccp_C2 = ccp_helpers["ccp_C2"]
    s_C1 = species_helpers["s_C1"]
    s_C2 = species_helpers["s_C2"]
    Recipients_mat = species_helpers["Recipients_mat"]
    
    # Helper terms
    Pi_c1, Pi_c2, Pi_s1, Pi_s2, \
    Pi_c1_s1, Pi_c1_s2, Pi_c2_s1, Pi_c2_s2, \
    Pibar, Pibar_c1, Pibar_c2 = Pi_update_ccp_helper(Pi, ccp_C1, ccp_C2, s_C1, s_C2, Recipients_mat)
    
    # Individual terms
    D = p_D * Pi_c1 * Pi_c2
    S = p_S * (Pi_c1_s1 * Pi_c2_s2 + Pi_c1_s2 * Pi_c2_s1)
    T = p_T * (Pi_c1 * Pibar_c2 + Pi_c2 * Pibar_c1)
    
    D_loss = 2 * p_D * torch.einsum("ij, j -> ij", Pi, E)
    S_loss = p_S * (Pi_s1 * E_s2 + Pi_s2 * E_s1)
    T_loss = p_T * (torch.einsum("ij, j -> ij", Pi, Ebar) + torch.einsum("ij, j -> ij", Pibar, E))
    
    S_leaf_branches = p_S * clade_species_map
    
    print(f"\nTerm contributions:")
    print(f"  Duplication (D): sum={D.sum():.6e}, max={D.max():.6e}")
    print(f"  Speciation (S): sum={S.sum():.6e}, max={S.max():.6e}")
    print(f"  Transfer (T): sum={T.sum():.6e}, max={T.max():.6e}")
    print(f"  D_loss: sum={D_loss.sum():.6e}, max={D_loss.max():.6e}")
    print(f"  S_loss: sum={S_loss.sum():.6e}, max={S_loss.max():.6e}")
    print(f"  T_loss: sum={T_loss.sum():.6e}, max={T_loss.max():.6e}")
    print(f"  S_leaf_branches: sum={S_leaf_branches.sum():.6e}, max={S_leaf_branches.max():.6e}")
    
    # Total
    Pi_new = D + T + S + D_loss + T_loss + S_loss + S_leaf_branches
    print(f"\nTotal Pi_new: sum={Pi_new.sum():.6e}, max={Pi_new.max():.6e}")
    
    # Check which terms contribute most to root clade
    root_clade_id = get_root_clade_id(ccp)
    print(f"\nContributions to root clade {root_clade_id}:")
    print(f"  D[{root_clade_id}, :].sum() = {D[root_clade_id, :].sum():.6e}")
    print(f"  S[{root_clade_id}, :].sum() = {S[root_clade_id, :].sum():.6e}")
    print(f"  T[{root_clade_id}, :].sum() = {T[root_clade_id, :].sum():.6e}")

if __name__ == "__main__":
    debug_pi_flow()
    debug_single_update_terms()