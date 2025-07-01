#!/usr/bin/env python3
"""
Debug why only branch 14 gets probability for the root clade.
"""

import sys
sys.path.append('.')

from matmul_ale_ccp import *
import torch

def debug_branch_14():
    """Debug why branch 14 is special."""
    print("=== Debug Branch 14 Specialness ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    
    # Setup
    ccp = build_ccp_from_single_tree("test_trees_1/g.nwk")
    species_helpers = build_species_helpers("test_trees_1/sp.nwk", device, dtype)
    
    # Check species tree structure around branch 14
    s_C1 = species_helpers['s_C1']
    s_C2 = species_helpers['s_C2']
    sp_names_by_idx = species_helpers['sp_names_by_idx']
    
    print("Species tree structure:")
    for e in range(s_C1.shape[0]):
        left_child_idx = torch.nonzero(s_C1[e, :] > 0).squeeze()
        right_child_idx = torch.nonzero(s_C2[e, :] > 0).squeeze()
        
        if left_child_idx.numel() == 1 and right_child_idx.numel() == 1:
            f, g = left_child_idx.item(), right_child_idx.item()
            name_e = sp_names_by_idx.get(e, f"branch_{e}")
            name_f = sp_names_by_idx.get(f, f"branch_{f}")
            name_g = sp_names_by_idx.get(g, f"branch_{g}")
            print(f"  Internal branch {e} ({name_e}): children {f} ({name_f}), {g} ({name_g})")
            
            if e == 14:
                print(f"    *** This is the special branch 14! ***")
                # Let's see what clades have probability on its children
                print(f"    Checking which clades have probability on children {f} and {g}:")
                
                # Run one iteration to see initial state
                result = main_ccp("test_trees_1/sp.nwk", "test_trees_1/g.nwk", 
                                 1e-10, 1e-10, 1e-10, iters=1)
                Pi = result['Pi']
                
                print(f"    Clades with probability on child branch {f}:")
                for clade_id in range(Pi.shape[0]):
                    if Pi[clade_id, f].item() > 1e-10:
                        clade = ccp.id_to_clade[clade_id]
                        print(f"      Clade {clade_id} ({clade.size} leaves): Pi[{f}] = {Pi[clade_id, f].item():.6f}")
                
                print(f"    Clades with probability on child branch {g}:")
                for clade_id in range(Pi.shape[0]):
                    if Pi[clade_id, g].item() > 1e-10:
                        clade = ccp.id_to_clade[clade_id]
                        print(f"      Clade {clade_id} ({clade.size} leaves): Pi[{g}] = {Pi[clade_id, g].item():.6f}")

def debug_species_tree_structure():
    """Debug the species tree to understand the structure better."""
    print("\n=== Species Tree Structure ===")
    
    from ete3 import Tree
    sp_tree = Tree("test_trees_1/sp.nwk", format=1)
    
    print("Species tree:")
    print(sp_tree.get_ascii(show_internal=True))
    
    print("\nTraversal order:")
    for i, node in enumerate(sp_tree.traverse("postorder")):
        print(f"  {i}: {node.name} (children: {[child.name for child in node.get_children()]})")

if __name__ == "__main__":
    debug_species_tree_structure()
    debug_branch_14()