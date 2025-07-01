#!/usr/bin/env python3
"""
Analyze the source of exact zeros in the Pi matrix for phylogenetic reconciliation.
This script examines the mathematical constraints that create impossible clade-species combinations.
"""

import torch
import numpy as np
from ete3 import Tree
from matmul_ale_ccp import (
    build_ccp_from_single_tree, 
    build_species_helpers, 
    build_clade_species_mapping,
    build_ccp_helpers,
    E_step,
    Pi_update_ccp_parallel,
    get_root_clade_id
)

def analyze_recipients_matrix(species_tree_path: str):
    """Analyze the Recipients matrix to understand transfer constraints."""
    print("=== RECIPIENTS MATRIX ANALYSIS ===")
    
    # Load species tree and build helpers
    device = torch.device("cpu")
    dtype = torch.float64
    species_helpers = build_species_helpers(species_tree_path, device, dtype)
    
    S = species_helpers["S"]
    ancestors_dense = species_helpers["ancestors_dense"]
    Recipients_mat = species_helpers["Recipients_mat"]
    sp_names_by_idx = species_helpers["sp_names_by_idx"]
    
    print(f"Number of species branches: {S}")
    print(f"\nSpecies names by index:")
    for i in range(S):
        print(f"  {i}: {sp_names_by_idx[i]}")
    
    print(f"\nAncestors matrix (1 if column is ancestor of row):")
    print("Rows = species branches, Cols = potential ancestors")
    for i in range(S):
        ancestors_str = []
        for j in range(S):
            if ancestors_dense[i, j] > 0:
                ancestors_str.append(f"{j}({sp_names_by_idx[j]})")
        print(f"  Branch {i}({sp_names_by_idx[i]}): ancestors = {', '.join(ancestors_str)}")
    
    print(f"\nRecipients matrix (transfer target probabilities):")
    print("Rows = source branches, Cols = potential recipient branches")
    for i in range(S):
        recipients_str = []
        for j in range(S):
            if Recipients_mat[i, j] > 0:
                recipients_str.append(f"{j}({sp_names_by_idx[j]}):{Recipients_mat[i,j]:.3f}")
        print(f"  From branch {i}({sp_names_by_idx[i]}): can transfer to = {', '.join(recipients_str)}")
    
    # Identify transfer constraints
    print(f"\nTransfer constraints analysis:")
    print("A gene can transfer FROM branch i TO branch j if j is NOT an ancestor of i")
    
    for i in range(S):
        forbidden = []
        allowed = []
        for j in range(S):
            if ancestors_dense[i, j] > 0:
                forbidden.append(f"{j}({sp_names_by_idx[j]})")
            else:
                allowed.append(f"{j}({sp_names_by_idx[j]})")
        print(f"  Branch {i}({sp_names_by_idx[i]}):")
        print(f"    FORBIDDEN transfers to: {', '.join(forbidden)}")  
        print(f"    ALLOWED transfers to: {', '.join(allowed)}")
    
    return species_helpers

def analyze_clade_constraints(gene_tree_path: str, species_helpers: dict):
    """Analyze clade-species mapping constraints."""
    print("\n=== CLADE-SPECIES MAPPING CONSTRAINTS ===")
    
    device = torch.device("cpu")
    dtype = torch.float64
    
    # Build CCP container
    ccp = build_ccp_from_single_tree(gene_tree_path)
    clade_species_map = build_clade_species_mapping(ccp, species_helpers, device, dtype)
    
    C = len(ccp.clades)
    S = species_helpers["S"]
    sp_names_by_idx = species_helpers["sp_names_by_idx"]
    
    print(f"Number of clades: {C}")
    print(f"Number of species: {S}")
    
    print(f"\nClade-species mapping matrix (1 if leaf clade maps to species):")
    for clade_id in range(C):
        clade = ccp.id_to_clade[clade_id]
        if clade.is_leaf():
            mapped_species = []
            for s in range(S):
                if clade_species_map[clade_id, s] > 0:
                    mapped_species.append(f"{s}({sp_names_by_idx[s]})")
            print(f"  Leaf clade {clade_id} ({clade.get_leaf_name()}): maps to {', '.join(mapped_species)}")
    
    # Show non-leaf clades
    print(f"\nNon-leaf clades:")
    for clade_id in range(C):
        clade = ccp.id_to_clade[clade_id]
        if not clade.is_leaf():
            print(f"  Internal clade {clade_id}: contains leaves {sorted(clade.leaves)}")
    
    return ccp, clade_species_map

def run_reconciliation_and_analyze_zeros(species_tree_path: str, gene_tree_path: str):
    """Run reconciliation and analyze the exact zeros in the Pi matrix."""
    print("\n=== PI MATRIX ZERO ANALYSIS ===")
    
    device = torch.device("cpu")
    dtype = torch.float64
    
    # Build all helpers
    species_helpers = build_species_helpers(species_tree_path, device, dtype)
    ccp = build_ccp_from_single_tree(gene_tree_path)
    clade_species_map = build_clade_species_mapping(ccp, species_helpers, device, dtype)
    ccp_helpers = build_ccp_helpers(ccp, device, dtype)
    
    # Set parameters
    delta, tau, lambda_param = 0.1, 0.1, 0.1
    rates_sum = 1.0 + delta + tau + lambda_param
    p_S = 1.0 / rates_sum
    p_D = delta / rates_sum
    p_T = tau / rates_sum
    p_L = lambda_param / rates_sum
    
    print(f"Event probabilities: p_S={p_S:.4f}, p_D={p_D:.4f}, p_T={p_T:.4f}, p_L={p_L:.4f}")
    
    # Compute extinction probabilities
    S = species_helpers["S"]
    E = torch.zeros(S, dtype=dtype, device=device)
    for _ in range(50):
        E_next, E_s1, E_s2, Ebar = E_step(E, species_helpers["s_C1"], species_helpers["s_C2"], 
                                          species_helpers["Recipients_mat"], p_S, p_D, p_T, p_L)
        E = E_next
    
    print(f"Extinction probabilities: {E}")
    
    # Initialize and compute Pi matrix
    C = len(ccp.clades)
    Pi = torch.zeros((C, S), dtype=dtype, device=device)
    
    # Run several iterations to see where zeros persist
    for iter_num in range(10):
        Pi_new = Pi_update_ccp_parallel(Pi, ccp_helpers, species_helpers, clade_species_map, 
                                       E, Ebar, p_S, p_D, p_T)
        Pi = Pi_new
    
    print(f"\nPi matrix after {iter_num+1} iterations:")
    print(f"Shape: {Pi.shape}")
    print(f"Non-zero entries: {torch.count_nonzero(Pi)} / {Pi.numel()}")
    print(f"Exact zeros: {Pi.numel() - torch.count_nonzero(Pi)}")
    
    # Analyze the pattern of zeros
    sp_names_by_idx = species_helpers["sp_names_by_idx"]
    
    print(f"\nDetailed Pi matrix analysis:")
    print(f"Rows = clades, Columns = species branches")
    
    exact_zeros = []
    
    for c in range(C):
        clade = ccp.id_to_clade[c]
        clade_desc = f"Clade_{c}"
        if clade.is_leaf():
            clade_desc += f"(leaf:{clade.get_leaf_name()})"
        else:
            clade_desc += f"(internal:{sorted(clade.leaves)})"
        
        zero_species = []
        nonzero_species = []
        
        for s in range(S):
            if Pi[c, s] == 0:
                zero_species.append(f"{s}({sp_names_by_idx[s]})")
                exact_zeros.append((c, s, clade_desc, sp_names_by_idx[s]))
            else:
                nonzero_species.append(f"{s}({sp_names_by_idx[s]}):{Pi[c,s]:.2e}")
        
        if len(zero_species) > 0:
            print(f"\n  {clade_desc}:")
            print(f"    ZERO on species: {', '.join(zero_species)}")
            print(f"    NON-ZERO on species: {', '.join(nonzero_species)}")
    
    # Analyze the constraints that cause zeros
    print(f"\n=== CONSTRAINT ANALYSIS ===")
    print(f"Total exact zeros found: {len(exact_zeros)}")
    
    # Check if zeros are only on leaf clades that don't map to species
    leaf_mapping_zeros = []
    internal_clade_zeros = []
    
    for c, s, clade_desc, species_name in exact_zeros:
        clade = ccp.id_to_clade[c]
        if clade.is_leaf():
            # Check if this is due to leaf mapping constraint
            if clade_species_map[c, s] == 0:
                leaf_mapping_zeros.append((c, s, clade_desc, species_name))
            else:
                print(f"UNEXPECTED: Leaf clade {clade_desc} maps to species {species_name} but Pi=0")
        else:
            internal_clade_zeros.append((c, s, clade_desc, species_name))
    
    print(f"\nZeros due to leaf mapping constraints: {len(leaf_mapping_zeros)}")
    print("(Leaf clades can only appear on species branches they map to)")
    
    print(f"\nZeros on internal clades: {len(internal_clade_zeros)}")
    if len(internal_clade_zeros) > 0:
        print("These are the mathematically interesting constraints!")
        for c, s, clade_desc, species_name in internal_clade_zeros[:5]:  # Show first 5
            print(f"  {clade_desc} cannot appear on species {species_name}")
            
            # Try to understand why this internal clade cannot appear on this species
            clade = ccp.id_to_clade[c] 
            clade_leaves = sorted(clade.leaves)
            
            # Check what species the clade's leaves map to
            leaf_species = set()
            for leaf_name in clade_leaves:
                for leaf_c in range(C):
                    leaf_clade = ccp.id_to_clade[leaf_c]
                    if leaf_clade.is_leaf() and leaf_clade.get_leaf_name() == leaf_name:
                        for leaf_s in range(S):
                            if clade_species_map[leaf_c, leaf_s] > 0:
                                leaf_species.add(leaf_s)
                        break
            
            print(f"    Clade leaves {clade_leaves} map to species branches: {sorted(leaf_species)}")
            print(f"    But clade cannot appear on species branch {s}")
            
            # Check ancestral relationships
            ancestors_dense = species_helpers["ancestors_dense"]
            clade_cannot_reach = []
            for ls in leaf_species:
                if ancestors_dense[ls, s] == 0 and ancestors_dense[s, ls] == 0 and ls != s:
                    # s is neither ancestor nor descendant of any leaf species
                    clade_cannot_reach.append(f"species_{s}_not_connected_to_leaf_species_{ls}")
            
            if len(clade_cannot_reach) > 0:
                print(f"    Topological constraint: {', '.join(clade_cannot_reach)}")
    
    return Pi, exact_zeros

def main():
    species_tree_path = "test_trees_1/sp.nwk"
    gene_tree_path = "test_trees_1/g.nwk"
    
    print("Analyzing the mathematical source of exact zeros in Pi matrix...")
    print(f"Species tree: {species_tree_path}")
    print(f"Gene tree: {gene_tree_path}")
    
    # Analyze transfer constraints
    species_helpers = analyze_recipients_matrix(species_tree_path)
    
    # Analyze clade constraints  
    ccp, clade_species_map = analyze_clade_constraints(gene_tree_path, species_helpers)
    
    # Run reconciliation and analyze zeros
    Pi, exact_zeros = run_reconciliation_and_analyze_zeros(species_tree_path, gene_tree_path)
    
    print(f"\n=== SUMMARY ===")
    print(f"The exact zeros in the Pi matrix come from:")
    print(f"1. LEAF MAPPING CONSTRAINTS: Leaf clades can only appear on species they map to")
    print(f"2. TRANSFER CONSTRAINTS: Transfers only allowed to non-ancestral branches")  
    print(f"3. TOPOLOGICAL IMPOSSIBILITIES: Some clade-species combinations are impossible")
    print(f"   even with transfers due to the tree structure")
    print(f"\nWith duplication, transfer, and loss events, most clade-species combinations")
    print(f"should theoretically be possible (though potentially very unlikely).")
    print(f"However, the model has strict topological constraints that create true impossibilities.")

if __name__ == "__main__":
    main()