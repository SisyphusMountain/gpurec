#!/usr/bin/env python3
"""
Analyze the exact topological constraints that create impossible clade-species combinations.
Focus on the pure speciation case to understand the fundamental constraints.
"""

import torch
import numpy as np
from ete3 import Tree
from matmul_ale import build_helpers, E_step as E_step_original, Pi_update

def analyze_pure_speciation_constraints(species_tree_path: str, gene_tree_path: str):
    """Analyze constraints in pure speciation scenario (δ=τ=λ=0)."""
    print("=== PURE SPECIATION CONSTRAINT ANALYSIS ===")
    print("In pure speciation mode (δ=τ=λ=0), only speciation and leaf mapping events are allowed.")
    print("This reveals the fundamental topological constraints of the reconciliation model.")
    
    device = torch.device("cpu")
    dtype = torch.float64
    
    # Build helpers
    helpers = build_helpers(species_tree_path, gene_tree_path, device, dtype, "postorder")
    
    # Pure speciation parameters
    delta, tau, lambda_param = 0.0, 0.0, 0.0
    rates_sum = 1.0 + delta + tau + lambda_param
    p_S = 1.0 / rates_sum  # = 1.0
    p_D = delta / rates_sum  # = 0.0
    p_T = tau / rates_sum    # = 0.0
    p_L = lambda_param / rates_sum  # = 0.0
    
    print(f"Event probabilities: p_S={p_S:.4f}, p_D={p_D:.4f}, p_T={p_T:.4f}, p_L={p_L:.4f}")
    
    # Compute extinction probabilities (should be zero in pure speciation)
    S = helpers["S"]
    E = torch.zeros(S, dtype=dtype, device=device)
    for _ in range(10):  # Should converge immediately
        E_next, E_s1, E_s2, Ebar = E_step_original(E, helpers["s_C1"], helpers["s_C2"], 
                                                  helpers["Recipients_mat"], p_S, p_D, p_T, p_L)
        if torch.allclose(E, E_next):
            break
        E = E_next
    
    print(f"Extinction probabilities: {E}")
    print(f"All extinction probabilities are zero: {torch.allclose(E, torch.zeros_like(E))}")
    
    # Initialize Pi matrix and run iterations
    G = helpers["G"]
    Pi = torch.zeros((G, S), dtype=dtype, device=device)
    
    print(f"\nGene tree structure:")
    for g in range(G):
        gene_name = helpers["g_names_by_idx"].get(g, f"gene_{g}")
        is_leaf = helpers["g_leaves_mask"][g].item()
        print(f"  Gene {g}: {gene_name} ({'leaf' if is_leaf else 'internal'})")
    
    print(f"\nSpecies tree structure:")
    for s in range(S):
        species_name = helpers["sp_names_by_idx"][s]
        is_leaf = helpers["sp_leaves_mask"][s].item()
        print(f"  Species {s}: {species_name} ({'leaf' if is_leaf else 'internal'})")
    
    # Run Pi iterations
    for iter_num in range(20):
        new_Pi = torch.zeros_like(Pi)
        Pi_new = Pi_update(Pi, new_Pi, helpers["g_C1"], helpers["g_C2"], 
                          helpers["s_C1"], helpers["s_C2"], helpers["Recipients_mat"],
                          helpers["leaves_map"], E, Ebar, E_s1, E_s2, p_S, p_D, p_T)
        
        if iter_num % 5 == 0:
            n_zeros = torch.count_nonzero(Pi_new == 0).item()
            print(f"  Iteration {iter_num}: {n_zeros}/{Pi_new.numel()} exact zeros")
        
        if torch.allclose(Pi, Pi_new, atol=1e-15):
            print(f"  Converged at iteration {iter_num}")
            break
        Pi = Pi_new
    
    # Analyze the final Pi matrix
    print(f"\nFinal Pi matrix analysis:")
    print(f"Shape: {Pi.shape}")
    n_zeros = torch.count_nonzero(Pi == 0).item()
    print(f"Exact zeros: {n_zeros}/{Pi.numel()}")
    
    # Find and categorize the zeros
    leaf_zeros = []
    internal_zeros = []
    
    for g in range(G):
        for s in range(S):
            if Pi[g, s] == 0:
                gene_name = helpers["g_names_by_idx"].get(g, f"gene_{g}")
                species_name = helpers["sp_names_by_idx"][s]
                is_gene_leaf = helpers["g_leaves_mask"][g].item()
                is_species_leaf = helpers["sp_leaves_mask"][s].item()
                
                if is_gene_leaf:
                    leaf_zeros.append((g, s, gene_name, species_name, is_species_leaf))
                else:
                    internal_zeros.append((g, s, gene_name, species_name, is_species_leaf))
    
    print(f"\nLeaf gene zeros: {len(leaf_zeros)}")
    print("(These should only be zeros where leaf gene doesn't map to leaf species)")
    
    # Check leaf mapping matrix
    leaves_map = helpers["leaves_map"]
    print(f"\nLeaf mapping matrix (gene leaves -> species leaves):")
    for g in range(G):
        if helpers["g_leaves_mask"][g]:
            gene_name = helpers["g_names_by_idx"].get(g, f"gene_{g}")
            mapped_species = []
            for s in range(S):
                if leaves_map[g, s] > 0:
                    species_name = helpers["sp_names_by_idx"][s]
                    mapped_species.append(f"{s}({species_name})")
            print(f"  Gene leaf {g} ({gene_name}): maps to {', '.join(mapped_species)}")
    
    # Verify that all leaf gene zeros are due to mapping constraints
    unexpected_leaf_zeros = []
    for g, s, gene_name, species_name, is_species_leaf in leaf_zeros:
        if leaves_map[g, s] > 0:
            unexpected_leaf_zeros.append((g, s, gene_name, species_name))
    
    if len(unexpected_leaf_zeros) == 0:
        print("✅ All leaf gene zeros are correctly due to leaf mapping constraints")
    else:
        print(f"❌ {len(unexpected_leaf_zeros)} unexpected leaf zeros found!")
    
    print(f"\nInternal gene zeros: {len(internal_zeros)}")
    print("These reveal the topological constraints for internal gene nodes!")
    
    if len(internal_zeros) > 0:
        print("\nAnalyzing internal gene constraints...")
        
        # Group by gene node to understand the pattern
        zeros_by_gene = {}
        for g, s, gene_name, species_name, is_species_leaf in internal_zeros:
            if g not in zeros_by_gene:
                zeros_by_gene[g] = []
            zeros_by_gene[g].append((s, species_name, is_species_leaf))
        
        for g, species_list in zeros_by_gene.items():
            gene_name = helpers["g_names_by_idx"].get(g, f"gene_{g}")
            print(f"\n  Internal gene {g} ({gene_name}) cannot appear on:")
            
            # Get gene children to understand the constraint
            if g in helpers["g_children_idx"]:
                left_child, right_child = helpers["g_children_idx"][g]
                left_name = helpers["g_names_by_idx"].get(left_child, f"gene_{left_child}")
                right_name = helpers["g_names_by_idx"].get(right_child, f"gene_{right_child}")
                print(f"    Gene children: {left_child} ({left_name}), {right_child} ({right_name})")
            
            forbidden_leaf_species = []
            forbidden_internal_species = []
            
            for s, species_name, is_species_leaf in species_list:
                if is_species_leaf:
                    forbidden_leaf_species.append(f"{s}({species_name})")
                else:
                    forbidden_internal_species.append(f"{s}({species_name})")
            
            if forbidden_leaf_species:
                print(f"    Forbidden leaf species: {', '.join(forbidden_leaf_species)}")
            if forbidden_internal_species:
                print(f"    Forbidden internal species: {', '.join(forbidden_internal_species)}")
        
        print(f"\n=== KEY INSIGHT ===")
        print(f"In pure speciation mode, internal gene nodes can only appear on species branches")
        print(f"that allow their children to speciate into appropriate descendant species.")
        print(f"This creates strict topological constraints based on the species tree structure.")
        
        # Let's verify this hypothesis by checking the species tree structure
        print(f"\nSpecies tree children relationships:")
        for s in range(S):
            if not helpers["sp_leaves_mask"][s]:  # Internal species
                if s in helpers["s_children_idx"]:
                    left_child, right_child = helpers["s_children_idx"][s]
                    left_name = helpers["sp_names_by_idx"][left_child]
                    right_name = helpers["sp_names_by_idx"][right_child]
                    species_name = helpers["sp_names_by_idx"][s]
                    print(f"  Species {s} ({species_name}) -> children: {left_child} ({left_name}), {right_child} ({right_name})")

def main():
    species_tree_path = "test_trees_1/sp.nwk"
    gene_tree_path = "test_trees_1/g.nwk"
    
    analyze_pure_speciation_constraints(species_tree_path, gene_tree_path)

if __name__ == "__main__":
    main()