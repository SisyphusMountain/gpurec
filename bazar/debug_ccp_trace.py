#!/usr/bin/env python3
"""
Trace the CCP construction step by step to find the double-counting.
"""

import sys
sys.path.append('.')

from matmul_ale_ccp import *
from ete3 import Tree

def trace_ccp_construction():
    """Trace CCP construction step by step."""
    print("=== Trace CCP Construction ===")
    
    tree = Tree("test_trees_1/g.nwk", format=1)
    all_leaves = {leaf.name for leaf in tree.get_leaves()}
    
    print(f"Gene tree leaves: {sorted(all_leaves)}")
    print(f"Number of leaves: {len(all_leaves)}")
    print(f"Expected number of rootings: {2*len(all_leaves) - 3}")
    
    # Manually trace the construction
    ccp_container = CCPContainer()
    root_clade = Clade(all_leaves)
    ccp_container.add_clade(root_clade)
    
    print(f"\nTracing edge-based splits:")
    edge_count = 0
    for node in tree.traverse():
        if node.is_root():
            continue
            
        edge_count += 1
        below_leaves = {leaf.name for leaf in node.get_leaves()}
        above_leaves = all_leaves - below_leaves
        
        below_clade = Clade(below_leaves)
        above_clade = Clade(above_leaves)
        
        print(f"Edge {edge_count}: {len(below_leaves)} vs {len(above_leaves)} leaves")
        print(f"  Below: {sorted(below_leaves)}")
        print(f"  Above: {sorted(above_leaves)}")
        
        # Check if this split already exists
        existing_split = None
        for split in ccp_container.splits[root_clade]:
            if ((split.left == below_clade and split.right == above_clade) or
                (split.left == above_clade and split.right == below_clade)):
                existing_split = split
                break
        
        if existing_split:
            print(f"  ⚠️  Split already exists with frequency {existing_split.frequency}")
        else:
            print(f"  ✅ New split")
        
        ccp_container.add_clade(below_clade)
        ccp_container.add_clade(above_clade)
        ccp_container.add_split(root_clade, below_clade, above_clade, frequency=1.0)
    
    print(f"\nTotal edges processed: {edge_count}")
    print(f"Total root splits: {len(ccp_container.splits[root_clade])}")
    
    # Check frequencies before normalization
    print(f"\nRoot split frequencies:")
    for i, split in enumerate(ccp_container.splits[root_clade]):
        print(f"  Split {i}: frequency = {split.frequency}")
    
    # Check for duplicates
    print(f"\nChecking for duplicate splits:")
    splits_seen = set()
    duplicates = 0
    
    for i, split in enumerate(ccp_container.splits[root_clade]):
        # Create a canonical representation
        left_leaves = tuple(sorted(split.left.leaves))
        right_leaves = tuple(sorted(split.right.leaves))
        canonical = tuple(sorted([left_leaves, right_leaves]))
        
        if canonical in splits_seen:
            duplicates += 1
            print(f"  Split {i} is duplicate: {len(split.left.leaves)} vs {len(split.right.leaves)} leaves")
        else:
            splits_seen.add(canonical)
    
    print(f"Total duplicates: {duplicates}")
    print(f"Unique splits: {len(splits_seen)}")

if __name__ == "__main__":
    trace_ccp_construction()