#!/usr/bin/env python3
"""
Test to check for zero values in the linear Pi matrix from matmul_ale_ccp.py
"""

import torch
import numpy as np
from matmul_ale_ccp import (
    build_ccp_from_single_tree,
    build_species_helpers,
    build_clade_species_mapping,
    build_ccp_helpers,
    get_root_clade_id,
    E_step,
    Pi_update_ccp
)

def test_zero_values_in_pi():
    """
    Test the linear implementation to see if it produces exact zeros in Pi matrix.
    """
    print("🔍 Testing for Zero Values in Linear Pi Matrix")
    print("=" * 50)
    
    # Test parameters
    species_path = "test_trees_1/sp.nwk"
    gene_path = "test_trees_1/g.nwk"
    
    # Parameters
    delta = 0.1
    tau = 0.05
    lambda_param = 0.1
    
    # Event probabilities
    rates_sum = 1.0 + delta + tau + lambda_param
    p_S = 1.0 / rates_sum
    p_D = delta / rates_sum
    p_T = tau / rates_sum
    p_L = lambda_param / rates_sum
    
    device = torch.device("cpu")
    dtype = torch.float64
    
    print(f"Parameters: δ={delta}, τ={tau}, λ={lambda_param}")
    print(f"Event probabilities: p_S={p_S:.4f}, p_D={p_D:.4f}, p_T={p_T:.4f}, p_L={p_L:.4f}")
    
    # Build structures
    print("\nBuilding CCP structures...")
    ccp = build_ccp_from_single_tree(gene_path)
    species_helpers = build_species_helpers(species_path, device, dtype)
    clade_species_map = build_clade_species_mapping(ccp, species_helpers, device, dtype)
    ccp_helpers = build_ccp_helpers(ccp, device, dtype)
    
    C = len(ccp.clades)
    S = species_helpers["S"]
    
    print(f"Matrix dimensions: {C} clades × {S} species = {C*S} elements")
    
    # Test different initialization strategies
    print(f"\n{'='*60}")
    print("TESTING DIFFERENT INITIALIZATION STRATEGIES")
    print(f"{'='*60}")
    
    strategies = [
        ("Small epsilon (1e-10)", 1e-10),
        ("Larger epsilon (1e-6)", 1e-6),
        ("Even larger epsilon (1e-3)", 1e-3),
        ("Zero initialization", 0.0)
    ]
    
    for strategy_name, epsilon in strategies:
        print(f"\n--- {strategy_name} ---")
        
        # Initialize Pi matrix
        if epsilon == 0.0:
            Pi = torch.zeros((C, S), dtype=dtype, device=device)
        else:
            Pi = torch.full((C, S), epsilon, dtype=dtype, device=device)
        
        # Set leaf probabilities
        for c in range(C):
            clade = ccp.id_to_clade[c]
            if clade.is_leaf():
                mapped_species = torch.nonzero(clade_species_map[c] > 0, as_tuple=False).flatten()
                if len(mapped_species) > 0:
                    if epsilon == 0.0:
                        Pi[c, :] = 0.0  # Keep zeros for non-mapped
                        Pi[c, mapped_species] = 1.0 / len(mapped_species)
                    else:
                        Pi[c, :] = epsilon  # Small value for non-mapped
                        Pi[c, mapped_species] = 1.0 / len(mapped_species)
        
        print(f"After initialization:")
        zeros_init = (Pi == 0.0).sum()
        exact_epsilon = (Pi == epsilon).sum() if epsilon > 0 else 0
        print(f"  Exact zeros: {zeros_init}")
        print(f"  Values equal to epsilon: {exact_epsilon}")
        print(f"  Minimum value: {Pi.min():.2e}")
        print(f"  Maximum value: {Pi.max():.6f}")
        
        # Run Pi updates
        n_iterations = 10
        for iteration in range(n_iterations):
            # Need extinction probabilities
            E = torch.zeros(S, dtype=dtype, device=device)
            E_s1 = torch.zeros(S, dtype=dtype, device=device)
            E_s2 = torch.zeros(S, dtype=dtype, device=device)
            Ebar = torch.zeros(S, dtype=dtype, device=device)
            
            Pi_new = Pi_update_ccp(
                Pi, ccp_helpers, species_helpers, clade_species_map,
                E, Ebar, E_s1, E_s2, p_S, p_D, p_T
            )
            
            # Check for zeros and very small values
            exact_zeros = (Pi_new == 0.0).sum()
            tiny_values = (Pi_new < 1e-15).sum()
            small_values = (Pi_new < 1e-10).sum()
            
            if iteration < 3 or iteration == n_iterations - 1:  # Show first few and last
                print(f"  Iteration {iteration + 1:2d}:")
                print(f"    Exact zeros: {exact_zeros:4d}")
                print(f"    Values < 1e-15: {tiny_values:4d}")
                print(f"    Values < 1e-10: {small_values:4d}")
                print(f"    Min value: {Pi_new.min():.2e}")
                print(f"    Max value: {Pi_new.max():.6f}")
                
                # Show distribution of values
                if exact_zeros > 0 or tiny_values > 0:
                    nonzero_values = Pi_new[Pi_new > 0]
                    if len(nonzero_values) > 0:
                        print(f"    Non-zero min: {nonzero_values.min():.2e}")
                        print(f"    Non-zero max: {nonzero_values.max():.6f}")
                        
                        # Show some example tiny values
                        if tiny_values > exact_zeros:
                            tiny_nonzero = Pi_new[(Pi_new > 0) & (Pi_new < 1e-15)]
                            if len(tiny_nonzero) > 0:
                                print(f"    Example tiny values: {tiny_nonzero[:5].tolist()}")
            
            Pi = Pi_new
            
            # Check for any NaN or inf values
            if torch.isnan(Pi).any():
                print(f"    ⚠️  NaN values detected at iteration {iteration + 1}")
                break
            if torch.isinf(Pi).any():
                print(f"    ⚠️  Infinite values detected at iteration {iteration + 1}")
                break
        
        print(f"\nFinal statistics for {strategy_name}:")
        exact_zeros_final = (Pi == 0.0).sum()
        print(f"  Exact zeros: {exact_zeros_final} / {C*S} ({100*exact_zeros_final/(C*S):.1f}%)")
        
        if exact_zeros_final > 0:
            print(f"  ✅ Found exact zeros! Log-space -inf is justified.")
        else:
            print(f"  ❌ No exact zeros found.")

def test_convergence_behavior():
    """
    Test how zeros behave during convergence.
    """
    print(f"\n{'='*60}")
    print("TESTING CONVERGENCE BEHAVIOR WITH ZEROS")
    print(f"{'='*60}")
    
    species_path = "test_trees_1/sp.nwk"
    gene_path = "test_trees_1/g.nwk"
    
    delta, tau, lambda_param = 0.01, 0.01, 0.01  # Very small rates
    rates_sum = 1.0 + delta + tau + lambda_param
    p_S = 1.0 / rates_sum
    p_D = delta / rates_sum
    p_T = tau / rates_sum
    p_L = lambda_param / rates_sum
    
    device = torch.device("cpu")
    dtype = torch.float64
    
    print(f"Using very small rates: δ={delta}, τ={tau}, λ={lambda_param}")
    print(f"Event probabilities: p_S={p_S:.6f}, p_D={p_D:.6f}, p_T={p_T:.6f}")
    
    # Build structures
    ccp = build_ccp_from_single_tree(gene_path)
    species_helpers = build_species_helpers(species_path, device, dtype)
    clade_species_map = build_clade_species_mapping(ccp, species_helpers, device, dtype)
    ccp_helpers = build_ccp_helpers(ccp, device, dtype)
    
    C = len(ccp.clades)
    S = species_helpers["S"]
    
    # Start with zero initialization
    Pi = torch.zeros((C, S), dtype=dtype, device=device)
    
    # Set only leaf probabilities
    for c in range(C):
        clade = ccp.id_to_clade[c]
        if clade.is_leaf():
            mapped_species = torch.nonzero(clade_species_map[c] > 0, as_tuple=False).flatten()
            if len(mapped_species) > 0:
                Pi[c, mapped_species] = 1.0 / len(mapped_species)
    
    print(f"\nInitial state:")
    print(f"  Exact zeros: {(Pi == 0.0).sum()} / {C*S}")
    print(f"  Non-zero values: {(Pi > 0.0).sum()}")
    
    # Run convergence
    print(f"\nRunning convergence iterations...")
    for iteration in range(20):
        E = torch.zeros(S, dtype=dtype, device=device)
        E_s1 = torch.zeros(S, dtype=dtype, device=device)
        E_s2 = torch.zeros(S, dtype=dtype, device=device)
        Ebar = torch.zeros(S, dtype=dtype, device=device)
        
        Pi_new = Pi_update_ccp(
            Pi, ccp_helpers, species_helpers, clade_species_map,
            E, Ebar, E_s1, E_s2, p_S, p_D, p_T
        )
        
        # Track changes
        exact_zeros = (Pi_new == 0.0).sum()
        changed_zeros = ((Pi == 0.0) & (Pi_new > 0.0)).sum()  # Zeros that became non-zero
        new_zeros = ((Pi > 0.0) & (Pi_new == 0.0)).sum()  # Non-zeros that became zero
        
        print(f"  Iteration {iteration + 1:2d}: "
              f"zeros={exact_zeros:3d}, "
              f"0→+={changed_zeros:3d}, "
              f"+→0={new_zeros:3d}, "
              f"min={Pi_new.min():.2e}, "
              f"max={Pi_new.max():.2e}")
        
        # Check for convergence
        if iteration > 0:
            diff = torch.abs(Pi_new - Pi).max()
            if diff < 1e-15:
                print(f"    Converged with diff={diff:.2e}")
                break
        
        Pi = Pi_new
    
    print(f"\nFinal convergence results:")
    final_zeros = (Pi == 0.0).sum()
    print(f"  Final exact zeros: {final_zeros} / {C*S} ({100*final_zeros/(C*S):.1f}%)")
    
    if final_zeros > 0:
        print(f"  ✅ Convergence preserves exact zeros!")
        
        # Show some example zero positions
        zero_positions = torch.nonzero(Pi == 0.0, as_tuple=False)
        print(f"  Example zero positions (clade, species):")
        for i in range(min(10, len(zero_positions))):
            c, s = zero_positions[i]
            clade_name = str(ccp.id_to_clade[c])[:20]  # Truncate long names
            print(f"    [{c:2d},{s:2d}] {clade_name}")
    else:
        print(f"  ❌ No exact zeros in final result.")

if __name__ == "__main__":
    test_zero_values_in_pi()
    test_convergence_behavior()