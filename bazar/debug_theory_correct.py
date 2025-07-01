#!/usr/bin/env python3
"""
Debug the theory-correct CCP implementation to identify issues.
"""

import sys
sys.path.append('.')

from matmul_ale_ccp import *
import torch

def debug_pi_update():
    """Debug the new Pi update implementation."""
    print("=== Debug Theory-Correct Pi Update ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    
    # Setup
    ccp = build_ccp_from_single_tree("test_trees_1/g.nwk")
    species_helpers = build_species_helpers("test_trees_1/sp.nwk", device, dtype)
    clade_species_map = build_clade_species_mapping(ccp, species_helpers, device, dtype)
    
    # Parameters
    delta, tau, lambda_param = 1e-10, 1e-10, 1e-10
    rates_sum = 1.0 + delta + tau + lambda_param
    p_S = 1.0 / rates_sum
    p_D = delta / rates_sum
    p_T = tau / rates_sum
    p_L = lambda_param / rates_sum
    
    print(f"Event probabilities: p_S={p_S:.6f}, p_D={p_D:.6f}, p_T={p_T:.6f}, p_L={p_L:.6f}")
    
    # E vector
    S = species_helpers["S"]
    E = torch.full((S,), 1e-10, dtype=dtype, device=device)
    E_s1 = torch.mv(species_helpers["s_C1"], E)
    E_s2 = torch.mv(species_helpers["s_C2"], E)
    Ebar = torch.mv(species_helpers["Recipients_mat"], E)
    
    # Initial Pi matrix
    C = len(ccp.clades)
    Pi = torch.zeros((C, S), dtype=dtype, device=device)
    
    # Set up CCP helpers
    ccp_helpers = build_ccp_helpers(ccp, device, dtype)
    
    print(f"\nInitial Pi: shape={Pi.shape}, sum={Pi.sum():.6f}")
    print(f"Clade-species map sum: {clade_species_map.sum():.6f}")
    
    # Examine species tree structure
    print(f"\nSpecies tree structure:")
    print(f"  s_C1 shape: {species_helpers['s_C1'].shape}")
    print(f"  s_C2 shape: {species_helpers['s_C2'].shape}")
    print(f"  sp_leaves_mask: {species_helpers['sp_leaves_mask']}")
    
    # Check which species branches are internal vs leaves
    sp_leaves_mask = species_helpers['sp_leaves_mask']
    internal_branches = []
    leaf_branches = []
    
    for e in range(S):
        if sp_leaves_mask[e]:
            leaf_branches.append(e)
        else:
            # Check if this branch has children (using correct indexing)
            left_child_idx = torch.nonzero(species_helpers['s_C1'][e, :] > 0).squeeze()
            right_child_idx = torch.nonzero(species_helpers['s_C2'][e, :] > 0).squeeze()
            
            if left_child_idx.numel() == 1 and right_child_idx.numel() == 1:
                f, g = left_child_idx.item(), right_child_idx.item()
                internal_branches.append(e)
                print(f"  Internal branch {e}: left child {f}, right child {g}")
            else:
                print(f"  Branch {e}: left_children={left_child_idx.numel()}, right_children={right_child_idx.numel()}")
    
    print(f"  Internal branches: {internal_branches}")
    print(f"  Leaf branches: {leaf_branches}")
    
    # Check CCP splits for root clade
    root_clade_id = get_root_clade_id(ccp)
    root_clade = ccp.id_to_clade[root_clade_id]
    print(f"\nRoot clade splits:")
    if root_clade in ccp.splits:
        for i, split in enumerate(ccp.splits[root_clade]):
            left_id = ccp.clade_to_id[split.left]
            right_id = ccp.clade_to_id[split.right]
            print(f"  Split {i}: prob={split.probability:.6f}, left={left_id} ({split.left.size} leaves), right={right_id} ({split.right.size} leaves)")
    else:
        print("  No splits found for root clade!")
    
    # Perform one Pi update and examine results
    print(f"\n--- Performing Pi Update ---")
    Pi_new = Pi_update_ccp(Pi, ccp_helpers, species_helpers, clade_species_map, 
                          E, Ebar, E_s1, E_s2, p_S, p_D, p_T)
    
    print(f"After update:")
    print(f"  Pi sum: {Pi_new.sum():.6f}")
    print(f"  Root clade Pi sum: {Pi_new[root_clade_id, :].sum():.6e}")
    
    # Check which clades gained probability
    for clade_id in range(C):
        old_sum = Pi[clade_id, :].sum().item()
        new_sum = Pi_new[clade_id, :].sum().item()
        if new_sum > old_sum + 1e-15:
            clade = ccp.id_to_clade[clade_id]
            print(f"  Clade {clade_id} gained: {old_sum:.6e} -> {new_sum:.6e} ({clade.size} leaves)")

if __name__ == "__main__":
    debug_pi_update()