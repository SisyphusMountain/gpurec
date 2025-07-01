#!/usr/bin/env python3
"""
Debug the suspicious 1/7 normalization in root clade Pi sum.
"""

import sys
sys.path.append('.')

from matmul_ale_ccp import *
import torch

def debug_normalization():
    """Debug why root clade Pi sum = 1/7."""
    print("=== Debug Normalization Issue ===")
    
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
    
    # Run the full algorithm
    result = main_ccp("test_trees_1/sp.nwk", "test_trees_1/g.nwk", 
                     delta, tau, lambda_param, iters=50)
    
    Pi = result['Pi']
    root_clade_id = result['root_clade_id']
    
    print(f"Root clade Pi sum: {Pi[root_clade_id, :].sum().item():.10f}")
    print(f"Root clade Pi values:")
    
    # Check Pi values per species branch
    sp_leaves_mask = species_helpers['sp_leaves_mask']
    internal_branches = []
    leaf_branches = []
    
    for e in range(Pi.shape[1]):
        pi_val = Pi[root_clade_id, e].item()
        if sp_leaves_mask[e]:
            leaf_branches.append((e, pi_val))
        else:
            # Check if internal
            s_C1 = species_helpers['s_C1']
            s_C2 = species_helpers['s_C2']
            left_child_idx = torch.nonzero(s_C1[e, :] > 0).squeeze()
            right_child_idx = torch.nonzero(s_C2[e, :] > 0).squeeze()
            
            if left_child_idx.numel() == 1 and right_child_idx.numel() == 1:
                internal_branches.append((e, pi_val))
                print(f"  Internal branch {e}: Pi = {pi_val:.10f}")
            else:
                print(f"  Other branch {e}: Pi = {pi_val:.10f}")
    
    print(f"\nLeaf branches:")
    for e, pi_val in leaf_branches:
        print(f"  Leaf branch {e}: Pi = {pi_val:.10f}")
    
    print(f"\nSummary:")
    print(f"  Number of internal branches: {len(internal_branches)}")
    print(f"  Sum of Pi on internal branches: {sum(pi for _, pi in internal_branches):.10f}")
    print(f"  Sum of Pi on leaf branches: {sum(pi for _, pi in leaf_branches):.10f}")
    print(f"  Total Pi sum: {Pi[root_clade_id, :].sum().item():.10f}")
    
    # Check if all internal branches have equal Pi
    if len(internal_branches) > 1:
        pi_values = [pi for _, pi in internal_branches]
        all_equal = all(abs(pi - pi_values[0]) < 1e-10 for pi in pi_values)
        print(f"  All internal branches have equal Pi: {all_equal}")
        if all_equal and pi_values[0] > 0:
            print(f"  Each internal branch Pi: {pi_values[0]:.10f}")
            print(f"  Expected total if uniform: {len(internal_branches) * pi_values[0]:.10f}")

if __name__ == "__main__":
    debug_normalization()