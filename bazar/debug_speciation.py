#!/usr/bin/env python3
"""
Debug the speciation term calculation for the root clade.
"""

import sys
sys.path.append('.')

from matmul_ale_ccp import *
import torch

def debug_speciation_terms():
    """Debug the speciation terms for the root clade."""
    print("=== Debug Speciation Terms ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    
    # Setup
    ccp = build_ccp_from_single_tree("test_trees_1/g.nwk")
    species_helpers = build_species_helpers("test_trees_1/sp.nwk", device, dtype)
    clade_species_map = build_clade_species_mapping(ccp, species_helpers, device, dtype)
    ccp_helpers = build_ccp_helpers(ccp, device, dtype)
    
    # Parameters
    delta, tau, lambda_param = 1e-10, 1e-10, 1e-10
    rates_sum = 1.0 + delta + tau + lambda_param
    p_S = 1.0 / rates_sum
    
    print(f"p_S = {p_S:.6f}")
    
    # Set up initial Pi with leaf values
    C, S = clade_species_map.shape
    Pi = clade_species_map.clone()  # Start with leaf mappings
    
    print(f"Initial Pi sum: {Pi.sum():.6f}")
    
    # Get root clade and its splits  
    root_clade_id = get_root_clade_id(ccp)
    root_clade = ccp.id_to_clade[root_clade_id]
    
    print(f"\nRoot clade has {len(ccp.splits[root_clade])} splits")
    
    # Debug speciation for one internal branch
    s_C1 = species_helpers["s_C1"]
    s_C2 = species_helpers["s_C2"]
    sp_leaves_mask = species_helpers["sp_leaves_mask"]
    
    # Find a good internal branch to debug
    test_branch = 14  # From our debug output, this should be internal
    left_child_idx = torch.nonzero(s_C1[test_branch, :] > 0).squeeze()
    right_child_idx = torch.nonzero(s_C2[test_branch, :] > 0).squeeze()
    
    if left_child_idx.numel() == 1 and right_child_idx.numel() == 1:
        f, g = left_child_idx.item(), right_child_idx.item()
        print(f"\nDebugging speciation on internal branch {test_branch}")
        print(f"  Left child branch: {f}")
        print(f"  Right child branch: {g}")
        
        # Calculate speciation term for root clade on this branch
        speciation_term = 0.0
        
        print(f"\n  Examining splits of root clade:")
        for i, split in enumerate(ccp.splits[root_clade]):
            gamma_prime_id = ccp.clade_to_id[split.left]
            gamma_prime2_id = ccp.clade_to_id[split.right]
            p_split = split.probability
            
            # Pi values for child clades
            pi_left_f = Pi[gamma_prime_id, f].item()
            pi_left_g = Pi[gamma_prime_id, g].item()
            pi_right_f = Pi[gamma_prime2_id, f].item()
            pi_right_g = Pi[gamma_prime2_id, g].item()
            
            # Speciation terms
            term1 = pi_left_f * pi_right_g
            term2 = pi_left_g * pi_right_f
            split_contribution = p_split * (term1 + term2)
            speciation_term += split_contribution
            
            if i < 5:  # Only show first few splits
                print(f"    Split {i}: prob={p_split:.6f}")
                print(f"      Left clade {gamma_prime_id} ({split.left.size} leaves): Pi[{f}]={pi_left_f:.6f}, Pi[{g}]={pi_left_g:.6f}")
                print(f"      Right clade {gamma_prime2_id} ({split.right.size} leaves): Pi[{f}]={pi_right_f:.6f}, Pi[{g}]={pi_right_g:.6f}")
                print(f"      Terms: {pi_left_f:.6f}*{pi_right_g:.6f} + {pi_left_g:.6f}*{pi_right_f:.6f} = {term1:.6f} + {term2:.6f} = {term1+term2:.6f}")
                print(f"      Split contribution: {p_split:.6f} * {term1+term2:.6f} = {split_contribution:.6f}")
        
        final_speciation = p_S * speciation_term
        print(f"\n  Total speciation term before p_S: {speciation_term:.6f}")
        print(f"  Final speciation term: p_S * {speciation_term:.6f} = {final_speciation:.6f}")
        print(f"  Current Pi[root, {test_branch}]: {Pi[root_clade_id, test_branch].item():.6f}")
        
        if abs(final_speciation) < 1e-15:
            print(f"  ⚠️  Speciation term is essentially zero!")
            print(f"  This suggests child clades don't have probability on the right branches.")
        
    else:
        print(f"Branch {test_branch} is not properly internal!")

if __name__ == "__main__":
    debug_speciation_terms()