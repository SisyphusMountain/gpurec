#!/usr/bin/env python3
"""
Debug the parallel CCP implementation to find the source of discrepancy.
"""

import sys
sys.path.append('.')

from matmul_ale_ccp import *
import torch

def debug_matrices():
    """Debug the CCP split matrices and Pi updates."""
    
    # Build components
    species_helpers = build_species_helpers("test_trees_1/sp.nwk", torch.device("cuda"), torch.float64)
    ccp = build_ccp_from_single_tree("test_trees_1/g.nwk")
    ccp_helpers = build_ccp_helpers(ccp, torch.device("cuda"), torch.float64)
    clade_species_map = build_clade_species_mapping(ccp, species_helpers, torch.device("cuda"), torch.float64)
    
    print("=== CCP Matrix Analysis ===")
    print(f"CCP C1 shape: {ccp_helpers['ccp_C1'].shape}")
    print(f"CCP C2 shape: {ccp_helpers['ccp_C2'].shape}")
    print(f"CCP C1 sum: {ccp_helpers['ccp_C1'].sum()}")
    print(f"CCP C2 sum: {ccp_helpers['ccp_C2'].sum()}")
    
    # Check if matrices are properly normalized
    print(f"CCP C1 row sums: {ccp_helpers['ccp_C1'].sum(dim=1)}")
    print(f"CCP C2 row sums: {ccp_helpers['ccp_C2'].sum(dim=1)}")
    
    # Look at root clade splits
    root_clade_id = get_root_clade_id(ccp)
    print(f"Root clade C1 row: {ccp_helpers['ccp_C1'][root_clade_id, :]}")
    print(f"Root clade C2 row: {ccp_helpers['ccp_C2'][root_clade_id, :]}")
    
    # Check if the sum of C1 + C2 for root equals number of splits
    root_total = ccp_helpers['ccp_C1'][root_clade_id, :].sum() + ccp_helpers['ccp_C2'][root_clade_id, :].sum()
    expected_splits = len(ccp.splits[ccp.id_to_clade[root_clade_id]])
    print(f"Root clade total split probability: {root_total} (expected: {expected_splits})")
    
    return ccp_helpers, species_helpers, clade_species_map

if __name__ == "__main__":
    debug_matrices()