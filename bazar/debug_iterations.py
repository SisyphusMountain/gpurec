#!/usr/bin/env python3
"""
Debug what happens during Pi iterations that leads to 1/7 probability.
"""

import sys
sys.path.append('.')

from matmul_ale_ccp import *
import torch

def debug_pi_iterations():
    """Debug Pi evolution over iterations."""
    print("=== Debug Pi Iterations ===")
    
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
    p_D = delta / rates_sum
    p_T = tau / rates_sum
    p_L = lambda_param / rates_sum
    
    # E vector
    S = species_helpers["S"]
    E = torch.full((S,), 1e-10, dtype=dtype, device=device)
    E_s1 = torch.mv(species_helpers["s_C1"], E)
    E_s2 = torch.mv(species_helpers["s_C2"], E)
    Ebar = torch.mv(species_helpers["Recipients_mat"], E)
    
    # Initial Pi matrix
    C = len(ccp.clades)
    Pi = torch.zeros((C, S), dtype=dtype, device=device)
    root_clade_id = get_root_clade_id(ccp)
    
    print(f"Root clade ID: {root_clade_id}")
    print(f"Tracking Pi evolution for root clade on branch 14:")
    
    # Manual iteration loop with detailed tracking
    for iter_pi in range(10):
        # Before update
        root_pi_14_before = Pi[root_clade_id, 14].item()
        root_pi_sum_before = Pi[root_clade_id, :].sum().item()
        
        # Perform update
        Pi_new = Pi_update_ccp(Pi, ccp_helpers, species_helpers, clade_species_map, 
                              E, Ebar, E_s1, E_s2, p_S, p_D, p_T)
        
        # After update
        root_pi_14_after = Pi_new[root_clade_id, 14].item()
        root_pi_sum_after = Pi_new[root_clade_id, :].sum().item()
        
        print(f"  Iter {iter_pi}: Pi[root,14]: {root_pi_14_before:.8f} -> {root_pi_14_after:.8f}, "
              f"total: {root_pi_sum_before:.8f} -> {root_pi_sum_after:.8f}")
        
        # Check convergence
        if iter_pi > 0:
            diff = torch.abs(Pi_new - Pi).max().item()
            print(f"    Max diff: {diff:.2e}")
            if diff < 1e-10:
                print(f"    Converged at iteration {iter_pi}")
                break
        
        Pi = Pi_new
        
        # If this is the iteration where root gets probability, debug it
        if root_pi_14_before == 0 and root_pi_14_after > 0:
            print(f"    *** Root clade gained probability on branch 14 in this iteration! ***")
            debug_speciation_on_branch_14(ccp, Pi, species_helpers, p_S, root_clade_id)
    
    print(f"\nFinal root Pi values:")
    for e in range(S):
        pi_val = Pi[root_clade_id, e].item()
        if pi_val > 1e-10:
            print(f"  Branch {e}: {pi_val:.8f}")

def debug_speciation_on_branch_14(ccp, Pi, species_helpers, p_S, root_clade_id):
    """Debug what creates probability on branch 14 for root clade."""
    print(f"      Debugging speciation calculation for branch 14:")
    
    # Branch 14 has children 8 and 13
    f, g = 8, 13
    
    root_clade = ccp.id_to_clade[root_clade_id]
    speciation_term = 0.0
    
    for i, split in enumerate(ccp.splits[root_clade]):
        gamma_prime_id = ccp.clade_to_id[split.left]
        gamma_prime2_id = ccp.clade_to_id[split.right]
        p_split = split.probability
        
        pi_left_f = Pi[gamma_prime_id, f].item()
        pi_left_g = Pi[gamma_prime_id, g].item()
        pi_right_f = Pi[gamma_prime2_id, f].item()
        pi_right_g = Pi[gamma_prime2_id, g].item()
        
        term1 = pi_left_f * pi_right_g
        term2 = pi_left_g * pi_right_f
        split_contribution = p_split * (term1 + term2)
        speciation_term += split_contribution
        
        if split_contribution > 1e-10:  # Only show contributing splits
            print(f"        Split {i}: p={p_split:.6f}, left_clade={gamma_prime_id}, right_clade={gamma_prime2_id}")
            print(f"          Pi[{gamma_prime_id},{f}]={pi_left_f:.6f}, Pi[{gamma_prime_id},{g}]={pi_left_g:.6f}")
            print(f"          Pi[{gamma_prime2_id},{f}]={pi_right_f:.6f}, Pi[{gamma_prime2_id},{g}]={pi_right_g:.6f}")
            print(f"          Contribution: {p_split:.6f} * ({term1:.6f} + {term2:.6f}) = {split_contribution:.6f}")
    
    final_speciation = p_S * speciation_term
    print(f"      Total speciation: p_S * {speciation_term:.6f} = {final_speciation:.6f}")

if __name__ == "__main__":
    debug_pi_iterations()