#!/usr/bin/env python3
"""
Debug the fixed CCP implementation to check split probabilities.
"""

import sys
sys.path.append('.')

from matmul_ale_ccp import *
import torch

def debug_fixed_ccp():
    """Debug the fixed CCP construction."""
    print("=== Debug Fixed CCP Construction ===")
    
    # Build CCP with new implementation
    ccp = build_ccp_from_single_tree("test_trees_1/g.nwk")
    
    # Get root clade splits
    root_clade_id = get_root_clade_id(ccp)
    root_clade = ccp.id_to_clade[root_clade_id]
    
    print(f"Root clade: {root_clade}")
    print(f"Number of splits: {len(ccp.splits[root_clade])}")
    
    # Compute probabilities
    ccp.compute_probabilities()
    
    # Check split probabilities
    print(f"\nSplit probabilities:")
    for i, split in enumerate(ccp.splits[root_clade]):
        print(f"Split {i}: probability = {split.probability:.10f} = 1/{1/split.probability:.1f}")
    
    # Check if all probabilities are equal
    probs = [split.probability for split in ccp.splits[root_clade]]
    all_equal = all(abs(p - probs[0]) < 1e-10 for p in probs)
    expected_prob = 1.0 / len(probs)
    
    print(f"\nAnalysis:")
    print(f"All probabilities equal: {all_equal}")
    print(f"Expected probability: {expected_prob:.10f} = 1/{len(probs)}")
    print(f"Actual first probability: {probs[0]:.10f}")
    print(f"Match expected: {abs(probs[0] - expected_prob) < 1e-10}")
    
    # Check total probability
    total_prob = sum(probs)
    print(f"Total probability: {total_prob:.10f} (should be 1.0)")

if __name__ == "__main__":
    debug_fixed_ccp()