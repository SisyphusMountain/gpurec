#!/usr/bin/env python3
"""
Debug script to examine CCP construction and matrix operations.
"""

import sys
sys.path.append('.')

from matmul_ale_ccp import *
import torch

def debug_ccp_construction():
    """Debug the CCP construction from the test gene tree."""
    print("=== CCP Construction Debug ===")
    
    # Build CCP from test tree
    ccp = build_ccp_from_single_tree("test_trees_1/g.nwk")
    
    print(f"Total clades: {len(ccp.clades)}")
    print(f"Total splits: {sum(len(splits) for splits in ccp.splits.values())}")
    
    # Print all clades by size
    clades_by_size = defaultdict(list)
    for clade_id, clade in ccp.id_to_clade.items():
        clades_by_size[clade.size].append((clade_id, clade))
    
    for size in sorted(clades_by_size.keys()):
        print(f"\nClades of size {size}:")
        for clade_id, clade in sorted(clades_by_size[size]):
            print(f"  {clade_id}: {clade}")
    
    # Find root clade
    all_leaves = set()
    for clade in ccp.clades:
        all_leaves.update(clade.leaves)
    
    root_clade = None
    for clade in ccp.clades:
        if clade.leaves == all_leaves:
            root_clade = clade
            break
    
    root_id = ccp.clade_to_id[root_clade]
    print(f"\nRoot clade ID: {root_id}")
    print(f"Root clade: {root_clade}")
    
    # Examine root clade splits
    print(f"\nRoot clade splits ({len(ccp.splits[root_clade])}):")
    for i, split in enumerate(ccp.splits[root_clade]):
        left_id = ccp.clade_to_id[split.left]
        right_id = ccp.clade_to_id[split.right] 
        print(f"  Split {i}: {root_id} -> {left_id} + {right_id}")
        print(f"    Left: {split.left}")
        print(f"    Right: {split.right}")
        print(f"    Frequency: {split.frequency}")
    
    return ccp, root_id

def debug_ccp_matrices(ccp, root_id):
    """Debug the CCP matrix construction."""
    print("\n=== CCP Matrix Debug ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    
    ccp_helpers = build_ccp_helpers(ccp, device, dtype)
    ccp_C1 = ccp_helpers['ccp_C1']
    ccp_C2 = ccp_helpers['ccp_C2']
    
    print(f"ccp_C1 shape: {ccp_C1.shape}")
    print(f"ccp_C2 shape: {ccp_C2.shape}")
    
    # Check root clade row in matrices
    print(f"\nRoot clade ({root_id}) in ccp_C1:")
    root_c1_row = ccp_C1[root_id, :]
    nonzero_c1 = torch.nonzero(root_c1_row).squeeze()
    if nonzero_c1.numel() > 0:
        for idx in nonzero_c1:
            if idx.numel() == 0:  # Handle single element case
                idx = nonzero_c1
            clade_id = int(idx.item()) if hasattr(idx, 'item') else int(idx)
            clade = ccp.id_to_clade[clade_id]
            value = root_c1_row[clade_id].item()
            print(f"  C1[{root_id}, {clade_id}] = {value:.6f} -> {clade}")
    else:
        print("  No non-zero entries in ccp_C1")
    
    print(f"\nRoot clade ({root_id}) in ccp_C2:")
    root_c2_row = ccp_C2[root_id, :]
    nonzero_c2 = torch.nonzero(root_c2_row).squeeze()
    if nonzero_c2.numel() > 0:
        for idx in nonzero_c2:
            if idx.numel() == 0:  # Handle single element case
                idx = nonzero_c2
            clade_id = int(idx.item()) if hasattr(idx, 'item') else int(idx)
            clade = ccp.id_to_clade[clade_id]
            value = root_c2_row[clade_id].item()
            print(f"  C2[{root_id}, {clade_id}] = {value:.6f} -> {clade}")
    else:
        print("  No non-zero entries in ccp_C2")

def debug_leaf_clades(ccp):
    """Debug leaf clade assignments."""
    print("\n=== Leaf Clade Debug ===")
    
    leaf_clades = []
    for clade_id, clade in ccp.id_to_clade.items():
        if clade.is_leaf():
            leaf_clades.append((clade_id, clade))
    
    print(f"Leaf clades ({len(leaf_clades)}):")
    for clade_id, clade in sorted(leaf_clades):
        print(f"  {clade_id}: {clade}")

def debug_clade_species_mapping(ccp):
    """Debug the clade-species mapping."""
    print("\n=== Clade-Species Mapping Debug ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    
    # Build species helpers
    species_helpers = build_species_helpers("test_trees_1/sp.nwk", device, dtype)
    
    # Build clade-species mapping
    clade_species_map = build_clade_species_mapping(ccp, species_helpers, device, dtype)
    
    print(f"Clade-species mapping shape: {clade_species_map.shape}")
    
    # Check which clades have species mappings
    has_mapping = torch.any(clade_species_map > 0, dim=1)
    mapped_clades = torch.nonzero(has_mapping).squeeze()
    
    print(f"Clades with species mappings: {len(mapped_clades) if mapped_clades.numel() > 1 else 1}")
    
    if mapped_clades.numel() > 0:
        if mapped_clades.numel() == 1:
            mapped_clades = [mapped_clades]
        
        for clade_id in mapped_clades:
            if hasattr(clade_id, 'item'):
                clade_id = clade_id.item()
            clade = ccp.id_to_clade[clade_id]
            species_indices = torch.nonzero(clade_species_map[clade_id, :]).squeeze()
            
            if species_indices.numel() > 0:
                if species_indices.numel() == 1:
                    species_indices = [species_indices]
                print(f"  Clade {clade_id} ({clade}): species {[int(s.item()) if hasattr(s, 'item') else int(s) for s in species_indices]}")

if __name__ == "__main__":
    ccp, root_id = debug_ccp_construction()
    debug_ccp_matrices(ccp, root_id)
    debug_leaf_clades(ccp)
    debug_clade_species_mapping(ccp)