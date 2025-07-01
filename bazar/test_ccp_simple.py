#!/usr/bin/env python3
"""
Simple test for CCP implementation.
"""

import torch
from matmul_ale_ccp_corrected import main_ccp
import sys


def test_ccp_implementation():
    """Test CCP implementation on test_trees_1."""
    
    print("Testing CCP implementation on test_trees_1...")
    
    # Test parameters
    species_tree = "test_trees_1/sp.nwk"
    gene_tree = "test_trees_1/g.nwk"
    delta = 0.1
    tau = 0.1
    lambda_param = 0.1
    
    try:
        result = main_ccp(species_tree, gene_tree, delta, tau, lambda_param, iters=50)
        
        print("\n" + "="*60)
        print("TEST RESULTS:")
        print("="*60)
        print(f"Log-likelihood: {result['log_likelihood']:.6f}")
        print(f"Converged successfully: {'Yes' if result['log_likelihood'] != float('-inf') else 'No'}")
        
        # Check some basic properties
        Pi = result['Pi']
        E = result['E']
        ccp = result['ccp']
        
        print(f"\nNumber of clades: {len(ccp.clades)}")
        print(f"Number of leaf clades: {len([c for c in ccp.clades if c.is_leaf()])}")
        print(f"Root clade size: {max(c.size for c in ccp.clades)}")
        
        # Check Pi properties
        print(f"\nPi matrix shape: {Pi.shape}")
        print(f"Pi matrix sum: {Pi.sum():.6f}")
        print(f"Non-zero Pi entries: {(Pi > 0).sum().item()}/{Pi.numel()}")
        
        # Check E properties
        print(f"\nE vector shape: {E.shape}")
        print(f"E mean (internal): {E[result['species_helpers']['sp_internal_mask']].mean():.6f}")
        print(f"E range: [{E.min():.6f}, {E.max():.6f}]")
        
        # Detailed clade information
        print("\nDetailed clade information:")
        clade_info = []
        for clade_id, clade in ccp.id_to_clade.items():
            pi_sum = Pi[clade_id].sum().item()
            num_splits = len(ccp.splits[clade])
            clade_info.append((clade.size, sorted(clade.leaves), pi_sum, num_splits))
        
        clade_info.sort(key=lambda x: x[0], reverse=True)
        for size, leaves, pi_sum, num_splits in clade_info[:10]:  # Show top 10
            print(f"  Size {size}: {leaves} - Pi_sum={pi_sum:.6e}, splits={num_splits}")
        
        return result['log_likelihood'] != float('-inf')
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_ccp_implementation()
    sys.exit(0 if success else 1)