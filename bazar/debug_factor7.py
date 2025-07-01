#!/usr/bin/env python3
"""
Deep analysis of the factor of 7 issue.
"""

import sys
sys.path.append('.')

from matmul_ale_ccp import *
import torch

def analyze_factor_7():
    """Analyze exactly where the factor of 7 comes from."""
    print("=== DEEP ANALYSIS: Factor of 7 ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    
    # Setup
    ccp = build_ccp_from_single_tree("test_trees_1/g.nwk")
    species_helpers = build_species_helpers("test_trees_1/sp.nwk", device, dtype)
    
    # Count internal species branches
    sp_leaves_mask = species_helpers['sp_leaves_mask']
    s_C1 = species_helpers['s_C1']
    s_C2 = species_helpers['s_C2']
    
    internal_branches = []
    for e in range(len(sp_leaves_mask)):
        if not sp_leaves_mask[e]:
            left_child_idx = torch.nonzero(s_C1[e, :] > 0).squeeze()
            right_child_idx = torch.nonzero(s_C2[e, :] > 0).squeeze()
            if left_child_idx.numel() == 1 and right_child_idx.numel() == 1:
                internal_branches.append(e)
    
    print(f"Number of internal species branches: {len(internal_branches)}")
    print(f"Internal branches: {internal_branches}")
    
    # Analyze root clade splits
    root_clade_id = get_root_clade_id(ccp)
    root_clade = ccp.id_to_clade[root_clade_id]
    
    print(f"\nRoot clade splits analysis:")
    print(f"Total splits: {len(ccp.splits[root_clade])}")
    
    ccp.compute_probabilities()
    
    # Check split probabilities
    split_probs = []
    for i, split in enumerate(ccp.splits[root_clade]):
        split_probs.append(split.probability)
        print(f"Split {i}: probability = {split.probability:.10f} = 1/{1/split.probability:.1f}")
    
    # Statistical analysis
    import numpy as np
    split_probs = np.array(split_probs)
    unique_probs = np.unique(split_probs)
    
    print(f"\nUnique probabilities:")
    for prob in unique_probs:
        count = np.sum(split_probs == prob)
        print(f"  {prob:.10f} (1/{1/prob:.1f}): appears {count} times")
    
    # Check the mathematical relationship
    print(f"\nMathematical verification:")
    prob_high = 1/7  # 0.142857...
    prob_low = 1/14  # 0.071429...
    
    n_high = np.sum(np.abs(split_probs - prob_high) < 1e-10)
    n_low = np.sum(np.abs(split_probs - prob_low) < 1e-10)
    
    print(f"Splits with probability ≈ 1/7: {n_high}")
    print(f"Splits with probability ≈ 1/14: {n_low}")
    print(f"Total check: {n_high + n_low} = {len(split_probs)}")
    
    # Probability sum check
    total_prob = n_high * prob_high + n_low * prob_low
    print(f"Probability sum: {n_high}*(1/7) + {n_low}*(1/14) = {total_prob:.10f}")
    
    # The key insight: why 1/7 and 1/14?
    print(f"\nKey insight analysis:")
    print(f"If frequencies were: 1 split gets frequency 2f, 12 splits get frequency f")
    print(f"Total frequency: 2f + 12f = 14f")
    print(f"Normalized probabilities: 2f/14f = 2/14 = 1/7, f/14f = 1/14")
    print(f"This matches our observations!")
    
    # Check if this relates to internal branches
    print(f"\nConnection to internal branches:")
    print(f"Number of internal species branches: {len(internal_branches)}")
    print(f"1/7 = 1/{len(internal_branches)}")
    print(f"The root clade gets exactly 1/{len(internal_branches)} probability!")
    
    return internal_branches, split_probs

def analyze_ccp_construction():
    """Analyze the CCP construction to find the double-counting."""
    print("\n=== CCP CONSTRUCTION ANALYSIS ===")
    
    from ete3 import Tree
    tree = Tree("test_trees_1/g.nwk", format=1)
    all_leaves = {leaf.name for leaf in tree.get_leaves()}
    
    print(f"Gene tree has {len(all_leaves)} leaves")
    print(f"Expected splits from edge-based method: {2*len(all_leaves) - 3} = {2*8-3}")
    
    # Trace the CCP construction step by step
    ccp_container = CCPContainer()
    ccp_container.add_clade(Clade(all_leaves))
    
    # Method 1: Edge-based splits (lines 148-155 in original code)
    unif_frequency = 1.0 / (2*len(all_leaves) - 3)
    print(f"\nMethod 1 (edge-based): unif_frequency = {unif_frequency:.6f} = 1/{1/unif_frequency:.0f}")
    
    edge_splits = []
    for node in tree.traverse():
        if not node.is_root():
            below_leaves = {leaf.name for leaf in node.get_leaves()}
            above_leaves = all_leaves - below_leaves
            below_clade = Clade(below_leaves)
            above_clade = Clade(above_leaves)
            edge_splits.append((Clade(all_leaves), below_clade, above_clade, unif_frequency))
            print(f"  Edge split: {len(below_leaves)} vs {len(above_leaves)} leaves, freq={unif_frequency:.6f}")
    
    print(f"Total edge splits: {len(edge_splits)}")
    
    # Method 2: Internal node splits (lines 160-177 in original code)  
    print(f"\nMethod 2 (internal node): frequency = 1.0")
    
    internal_splits = []
    for node in tree.traverse():
        if not node.is_root() and not node.is_leaf():
            left, right = node.get_children()
            left_leaves = {leaf.name for leaf in left.get_leaves()}
            right_leaves = {leaf.name for leaf in right.get_leaves()}
            above_leaves = all_leaves - left_leaves - right_leaves
            
            left_clade = Clade(left_leaves)
            right_clade = Clade(right_leaves)
            above_clade = Clade(above_leaves)
            
            # Three splits per internal node
            splits = [
                (left_clade + right_clade, left_clade, right_clade, 1.0),
                (left_clade + above_clade, left_clade, above_clade, 1.0),
                (right_clade + above_clade, right_clade, above_clade, 1.0)
            ]
            
            for parent, child1, child2, freq in splits:
                internal_splits.append((parent, child1, child2, freq))
                print(f"  Internal split: {len(child1.leaves)} vs {len(child2.leaves)} leaves (parent: {len(parent.leaves)}), freq={freq}")
    
    print(f"Total internal splits: {len(internal_splits)}")
    
    # Find overlaps
    print(f"\nChecking for overlaps between methods:")
    overlaps = 0
    for edge_parent, edge_child1, edge_child2, edge_freq in edge_splits:
        for int_parent, int_child1, int_child2, int_freq in internal_splits:
            if (edge_parent == int_parent and 
                ((edge_child1 == int_child1 and edge_child2 == int_child2) or
                 (edge_child1 == int_child2 and edge_child2 == int_child1))):
                overlaps += 1
                print(f"  OVERLAP: {len(edge_child1.leaves)} vs {len(edge_child2.leaves)} leaves")
                print(f"    Edge method: freq={edge_freq:.6f}")
                print(f"    Internal method: freq={int_freq}")
                print(f"    Combined frequency: {edge_freq + int_freq:.6f}")
                print(f"    After normalization: ???")
    
    print(f"Total overlaps found: {overlaps}")

if __name__ == "__main__":
    internal_branches, split_probs = analyze_factor_7()
    analyze_ccp_construction()