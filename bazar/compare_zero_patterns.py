#!/usr/bin/env python3
"""
Compare zero patterns between original single-tree reconciliation and CCP-based reconciliation.
This will help us understand the source of exact zeros.
"""

import torch
import numpy as np
from ete3 import Tree
from matmul_ale import build_helpers, E_step as E_step_original, Pi_update
from matmul_ale_ccp import (
    build_ccp_from_single_tree, 
    build_species_helpers, 
    build_clade_species_mapping,
    build_ccp_helpers,
    E_step,
    Pi_update_ccp_parallel,
    get_root_clade_id
)

def run_original_reconciliation(species_tree_path: str, gene_tree_path: str):
    """Run the original single-tree reconciliation."""
    print("=== ORIGINAL SINGLE-TREE RECONCILIATION ===")
    
    device = torch.device("cpu")
    dtype = torch.float64
    
    # Build original helpers
    helpers = build_helpers(species_tree_path, gene_tree_path, device, dtype, "postorder")
    
    # Set parameters
    delta, tau, lambda_param = 0.1, 0.1, 0.1
    rates_sum = 1.0 + delta + tau + lambda_param
    p_S = 1.0 / rates_sum
    p_D = delta / rates_sum
    p_T = tau / rates_sum
    p_L = lambda_param / rates_sum
    
    # Compute extinction probabilities
    S = helpers["S"]
    E = torch.zeros(S, dtype=dtype, device=device)
    for _ in range(50):
        E_next, E_s1, E_s2, Ebar = E_step_original(E, helpers["s_C1"], helpers["s_C2"], 
                                                  helpers["Recipients_mat"], p_S, p_D, p_T, p_L)
        E = E_next
    
    # Initialize Pi matrix
    G = helpers["G"]
    Pi = torch.zeros((G, S), dtype=dtype, device=device)
    
    # Run Pi iterations
    for iter_num in range(50):
        new_Pi = torch.zeros_like(Pi)
        Pi_new = Pi_update(Pi, new_Pi, helpers["g_C1"], helpers["g_C2"], 
                          helpers["s_C1"], helpers["s_C2"], helpers["Recipients_mat"],
                          helpers["leaves_map"], E, Ebar, E_s1, E_s2, p_S, p_D, p_T)
        
        # Check convergence
        if iter_num > 0:
            diff = torch.abs(Pi_new - Pi).max()
            if diff < 1e-12:
                print(f"  Original converged at iteration {iter_num} (diff = {diff:.2e})")
                break
        Pi = Pi_new
    
    print(f"Original Pi matrix: shape = {Pi.shape}")
    print(f"Non-zero entries: {torch.count_nonzero(Pi)} / {Pi.numel()}")
    print(f"Exact zeros: {Pi.numel() - torch.count_nonzero(Pi)}")
    
    # Analyze zero pattern
    exact_zeros_original = []
    for g in range(G):
        for s in range(S):
            if Pi[g, s] == 0:
                gene_name = helpers["g_names_by_idx"][g] if g in helpers["g_names_by_idx"] else f"gene_{g}"
                species_name = helpers["sp_names_by_idx"][s]
                exact_zeros_original.append((g, s, gene_name, species_name))
    
    print(f"Found {len(exact_zeros_original)} exact zeros in original reconciliation")
    
    # Show some examples
    if len(exact_zeros_original) > 0:
        print("Examples of exact zeros:")
        for i, (g, s, gene_name, species_name) in enumerate(exact_zeros_original[:10]):
            print(f"  Gene {g} ({gene_name}) on species {s} ({species_name}): Pi = 0")
    
    return Pi, exact_zeros_original, helpers

def run_ccp_reconciliation(species_tree_path: str, gene_tree_path: str):
    """Run CCP-based reconciliation."""
    print("\n=== CCP-BASED RECONCILIATION ===")
    
    device = torch.device("cpu")
    dtype = torch.float64
    
    # Build CCP helpers
    species_helpers = build_species_helpers(species_tree_path, device, dtype)
    ccp = build_ccp_from_single_tree(gene_tree_path)
    clade_species_map = build_clade_species_mapping(ccp, species_helpers, device, dtype)
    ccp_helpers = build_ccp_helpers(ccp, device, dtype)
    
    # Set parameters (same as original)
    delta, tau, lambda_param = 0.1, 0.1, 0.1
    rates_sum = 1.0 + delta + tau + lambda_param
    p_S = 1.0 / rates_sum
    p_D = delta / rates_sum
    p_T = tau / rates_sum
    p_L = lambda_param / rates_sum
    
    # Compute extinction probabilities
    S = species_helpers["S"]
    E = torch.zeros(S, dtype=dtype, device=device)
    for _ in range(50):
        E_next, E_s1, E_s2, Ebar = E_step(E, species_helpers["s_C1"], species_helpers["s_C2"], 
                                          species_helpers["Recipients_mat"], p_S, p_D, p_T, p_L)
        E = E_next
    
    # Initialize Pi matrix
    C = len(ccp.clades)
    Pi = torch.zeros((C, S), dtype=dtype, device=device)
    
    # Run Pi iterations
    for iter_num in range(50):
        Pi_new = Pi_update_ccp_parallel(Pi, ccp_helpers, species_helpers, clade_species_map, 
                                       E, Ebar, p_S, p_D, p_T)
        
        # Check convergence
        if iter_num > 0:
            diff = torch.abs(Pi_new - Pi).max()
            if diff < 1e-12:
                print(f"  CCP converged at iteration {iter_num} (diff = {diff:.2e})")
                break
        Pi = Pi_new
    
    print(f"CCP Pi matrix: shape = {Pi.shape}")
    print(f"Non-zero entries: {torch.count_nonzero(Pi)} / {Pi.numel()}")
    print(f"Exact zeros: {Pi.numel() - torch.count_nonzero(Pi)}")
    
    # Analyze zero pattern
    exact_zeros_ccp = []
    for c in range(C):
        for s in range(S):
            if Pi[c, s] == 0:
                clade = ccp.id_to_clade[c]
                species_name = species_helpers["sp_names_by_idx"][s]
                exact_zeros_ccp.append((c, s, str(clade), species_name))
    
    print(f"Found {len(exact_zeros_ccp)} exact zeros in CCP reconciliation")
    
    return Pi, exact_zeros_ccp, ccp, species_helpers

def analyze_differences(original_zeros, ccp_zeros):
    """Analyze the differences in zero patterns between methods."""
    print("\n=== COMPARISON OF ZERO PATTERNS ===")
    
    print(f"Original method: {len(original_zeros)} exact zeros")
    print(f"CCP method: {len(ccp_zeros)} exact zeros")
    
    if len(original_zeros) > 0 and len(ccp_zeros) == 0:
        print("\nKEY INSIGHT: The CCP method eliminates exact zeros that exist in the original method!")
        print("This suggests that the probabilistic treatment of gene tree topology")
        print("allows gene clades to appear on species branches that would be impossible")
        print("with a fixed gene tree topology.")
        
        print("\nLet's understand WHY the original method has zeros...")
        return
    
    if len(original_zeros) == 0 and len(ccp_zeros) > 0:
        print("\nUnexpected: CCP method introduces zeros not present in original method")
        
    if len(original_zeros) > 0 and len(ccp_zeros) > 0:
        print("\nBoth methods have exact zeros - analyzing patterns...")

def run_detailed_zero_analysis():
    """Run a more detailed analysis of what creates zeros in different scenarios."""
    print("\n=== DETAILED ZERO ANALYSIS ===")
    
    # Test with different parameter settings to see if zeros appear
    test_cases = [
        {"delta": 0.1, "tau": 0.1, "lambda": 0.1, "name": "balanced"},
        {"delta": 0.0, "tau": 0.0, "lambda": 0.0, "name": "pure_speciation"},
        {"delta": 10.0, "tau": 0.0, "lambda": 0.0, "name": "high_duplication"},
        {"delta": 0.0, "tau": 10.0, "lambda": 0.0, "name": "high_transfer"},
        {"delta": 0.0, "tau": 0.0, "lambda": 10.0, "name": "high_loss"},
    ]
    
    device = torch.device("cpu")
    dtype = torch.float64
    species_tree_path = "test_trees_1/sp.nwk"
    gene_tree_path = "test_trees_1/g.nwk"
    
    # Build helpers once
    helpers = build_helpers(species_tree_path, gene_tree_path, device, dtype, "postorder")
    species_helpers = build_species_helpers(species_tree_path, device, dtype)
    ccp = build_ccp_from_single_tree(gene_tree_path)
    clade_species_map = build_clade_species_mapping(ccp, species_helpers, device, dtype)
    ccp_helpers = build_ccp_helpers(ccp, device, dtype)
    
    for test_case in test_cases:
        delta = test_case["delta"]
        tau = test_case["tau"] 
        lambda_param = test_case["lambda"]
        name = test_case["name"]
        
        print(f"\n--- Testing {name}: δ={delta}, τ={tau}, λ={lambda_param} ---")
        
        # Compute event probabilities
        rates_sum = 1.0 + delta + tau + lambda_param
        p_S = 1.0 / rates_sum
        p_D = delta / rates_sum
        p_T = tau / rates_sum
        p_L = lambda_param / rates_sum
        
        # Compute extinction probabilities
        S = helpers["S"]
        E = torch.zeros(S, dtype=dtype, device=device)
        for _ in range(50):
            E_next, E_s1, E_s2, Ebar = E_step_original(E, helpers["s_C1"], helpers["s_C2"], 
                                                      helpers["Recipients_mat"], p_S, p_D, p_T, p_L)
            E = E_next
        
        # Test original method
        G = helpers["G"]
        Pi_orig = torch.zeros((G, S), dtype=dtype, device=device)
        for _ in range(30):
            new_Pi = torch.zeros_like(Pi_orig)
            Pi_orig = Pi_update(Pi_orig, new_Pi, helpers["g_C1"], helpers["g_C2"], 
                               helpers["s_C1"], helpers["s_C2"], helpers["Recipients_mat"],
                               helpers["leaves_map"], E, Ebar, E_s1, E_s2, p_S, p_D, p_T)
        
        orig_zeros = torch.count_nonzero(Pi_orig == 0).item()
        
        # Test CCP method  
        C = len(ccp.clades)
        Pi_ccp = torch.zeros((C, S), dtype=dtype, device=device)
        for _ in range(30):
            Pi_ccp = Pi_update_ccp_parallel(Pi_ccp, ccp_helpers, species_helpers, clade_species_map, 
                                           E, Ebar, p_S, p_D, p_T)
        
        ccp_zeros = torch.count_nonzero(Pi_ccp == 0).item()
        
        print(f"  Original zeros: {orig_zeros}/{Pi_orig.numel()}")
        print(f"  CCP zeros: {ccp_zeros}/{Pi_ccp.numel()}")
        
        if name == "pure_speciation" and orig_zeros > 0:
            print("  INTERESTING: Pure speciation still creates zeros in original method")
            print("  This suggests topological constraints beyond transfer restrictions")

def main():
    species_tree_path = "test_trees_1/sp.nwk"
    gene_tree_path = "test_trees_1/g.nwk"
    
    print("Comparing zero patterns between original and CCP reconciliation methods...")
    
    # Run original reconciliation
    Pi_orig, orig_zeros, orig_helpers = run_original_reconciliation(species_tree_path, gene_tree_path)
    
    # Run CCP reconciliation
    Pi_ccp, ccp_zeros, ccp, species_helpers = run_ccp_reconciliation(species_tree_path, gene_tree_path)
    
    # Analyze differences
    analyze_differences(orig_zeros, ccp_zeros)
    
    # Run detailed parameter analysis
    run_detailed_zero_analysis()

if __name__ == "__main__":
    main()