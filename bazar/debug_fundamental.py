#!/usr/bin/env python3
"""
Fundamental analysis: Why only 1 split contributes vs 13 expected.
"""

import sys
sys.path.append('.')

from matmul_ale_ccp import *
import torch

def fundamental_analysis():
    """The REAL issue: Why only 1/13 splits contribute to reconciliation."""
    print("=== FUNDAMENTAL ISSUE ANALYSIS ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    
    # Setup
    ccp = build_ccp_from_single_tree("test_trees_1/g.nwk")
    species_helpers = build_species_helpers("test_trees_1/sp.nwk", device, dtype)
    clade_species_map = build_clade_species_mapping(ccp, species_helpers, device, dtype)
    
    # Run full algorithm to get final Pi
    result = main_ccp("test_trees_1/sp.nwk", "test_trees_1/g.nwk", 
                     1e-10, 1e-10, 1e-10, iters=50)
    Pi = result['Pi']
    root_clade_id = result['root_clade_id']
    
    # Key insight: Root clade Pi sum = 1/7, not expected value
    actual_pi_sum = Pi[root_clade_id, :].sum().item()
    print(f"Actual root Pi sum: {actual_pi_sum:.10f} = 1/{1/actual_pi_sum:.1f}")
    
    # What SHOULD the Pi sum be if all 13 splits contributed?
    root_clade = ccp.id_to_clade[root_clade_id]
    split_0_prob = 1/7  # We know Split 0 has probability 1/7
    other_split_prob = 1/14  # Others have probability 1/14
    
    print(f"\nCCP split analysis:")
    print(f"Split 0 probability: {split_0_prob:.6f}")
    print(f"Other 12 splits probability: {other_split_prob:.6f} each")
    
    # From previous analysis: only Split 0 contributes
    # Split 0 contributes exactly its probability (1/7) to root Pi
    # Other splits contribute 0
    
    expected_contribution_if_all_equal = 13 * (1/13)  # If all splits had equal prob 1/13
    expected_contribution_current = 1 * (1/7) + 12 * 0  # Current: only Split 0 contributes
    
    print(f"\nContribution analysis:")
    print(f"If all 13 splits contributed equally (1/13 each): Pi sum = {expected_contribution_if_all_equal:.6f}")
    print(f"Current (only Split 0 contributes): Pi sum = {expected_contribution_current:.6f} = 1/7")
    print(f"Ratio: {expected_contribution_if_all_equal / expected_contribution_current:.1f}")
    
    # The REAL question: Why don't the other 12 splits contribute?
    print(f"\n=== WHY OTHER SPLITS DON'T CONTRIBUTE ===")
    
    # Check each split's potential contribution to species branch 14
    s_C1 = species_helpers["s_C1"]
    s_C2 = species_helpers["s_C2"]
    
    # Branch 14 has children 8 and 13
    test_branch = 14
    left_child_idx = torch.nonzero(s_C1[test_branch, :] > 0).squeeze()
    right_child_idx = torch.nonzero(s_C2[test_branch, :] > 0).squeeze()
    f, g = left_child_idx.item(), right_child_idx.item()
    
    print(f"Testing speciation on branch {test_branch} (children: {f}, {g})")
    
    for i, split in enumerate(ccp.splits[root_clade]):
        gamma_prime_id = ccp.clade_to_id[split.left]
        gamma_prime2_id = ccp.clade_to_id[split.right]
        p_split = split.probability
        
        # Check if child clades have probability on the right branches
        pi_left_f = Pi[gamma_prime_id, f].item()
        pi_left_g = Pi[gamma_prime_id, g].item()
        pi_right_f = Pi[gamma_prime2_id, f].item()
        pi_right_g = Pi[gamma_prime2_id, g].item()
        
        speciation_contribution = p_split * (pi_left_f * pi_right_g + pi_left_g * pi_right_f)
        
        if i < 5:  # Show first 5 splits
            print(f"\nSplit {i}: prob={p_split:.6f}")
            print(f"  Child clades: {gamma_prime_id} ({split.left.size} leaves), {gamma_prime2_id} ({split.right.size} leaves)")
            print(f"  Pi[{gamma_prime_id}, {f}]={pi_left_f:.6f}, Pi[{gamma_prime_id}, {g}]={pi_left_g:.6f}")
            print(f"  Pi[{gamma_prime2_id}, {f}]={pi_right_f:.6f}, Pi[{gamma_prime2_id}, {g}]={pi_right_g:.6f}")
            print(f"  Speciation: {p_split:.6f} * ({pi_left_f:.6f}*{pi_right_g:.6f} + {pi_left_g:.6f}*{pi_right_f:.6f}) = {speciation_contribution:.6f}")
            
            if speciation_contribution == 0:
                print(f"  ❌ ZERO contribution - child clades not on compatible branches!")
            else:
                print(f"  ✅ Non-zero contribution")
    
    # Count how many splits actually contribute
    contributing_splits = 0
    total_contribution = 0
    for split in ccp.splits[root_clade]:
        gamma_prime_id = ccp.clade_to_id[split.left]
        gamma_prime2_id = ccp.clade_to_id[split.right]
        p_split = split.probability
        
        pi_left_f = Pi[gamma_prime_id, f].item()
        pi_left_g = Pi[gamma_prime_id, g].item()  
        pi_right_f = Pi[gamma_prime2_id, f].item()
        pi_right_g = Pi[gamma_prime2_id, g].item()
        
        contribution = p_split * (pi_left_f * pi_right_g + pi_left_g * pi_right_f)
        
        if contribution > 1e-15:
            contributing_splits += 1
            total_contribution += contribution
    
    print(f"\n=== SUMMARY ===")
    print(f"Total splits: {len(ccp.splits[root_clade])}")
    print(f"Contributing splits: {contributing_splits}")
    print(f"Non-contributing splits: {len(ccp.splits[root_clade]) - contributing_splits}")
    print(f"Total speciation contribution: {total_contribution:.6f}")
    print(f"Root Pi[{test_branch}]: {Pi[root_clade_id, test_branch].item():.6f}")
    
    # THE KEY INSIGHT
    print(f"\n🔑 KEY INSIGHT:")
    print(f"Root Pi sum = 1/7 because:")
    print(f"1. Split 0 has inflated probability 1/7 (instead of 1/13) due to CCP construction bug")
    print(f"2. Only Split 0 contributes to reconciliation - other 12 splits contribute ZERO")
    print(f"3. This gives Pi sum = 1*(1/7) + 12*0 = 1/7")
    print(f"")
    print(f"If both issues were fixed:")
    print(f"- All splits had equal probability 1/13")  
    print(f"- All splits contributed to reconciliation")
    print(f"Then Pi sum would be ≈ 13*(1/13) = 1")
    print(f"And log-likelihood would improve by ln(7) ≈ 1.95, getting close to AleRax!")

if __name__ == "__main__":
    fundamental_analysis()