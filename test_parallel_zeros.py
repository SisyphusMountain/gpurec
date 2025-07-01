#!/usr/bin/env python3
"""
Test to check for zero values in the Pi matrix from Pi_update_ccp_parallel.
"""

import torch
from matmul_ale_ccp import (
    build_ccp_from_single_tree,
    build_species_helpers,
    build_clade_species_mapping,
    build_ccp_helpers,
    E_step,
    Pi_update_ccp_parallel
)

def test_parallel_zeros():
    """
    Test if Pi_update_ccp_parallel produces exact zeros.
    """
    print("🔍 Testing for Zero Values in Pi_update_ccp_parallel")
    print("=" * 60)
    
    # Test parameters
    species_path = "test_trees_1/sp.nwk"
    gene_path = "test_trees_1/g.nwk"
    
    # Parameters - same as in our consistency test
    delta = 1.0
    tau = 1.0
    lambda_param = 1.0
    
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
    strategies = [
        ("Zero initialization", 0.0),
        ("Small epsilon (1e-10)", 1e-10),
        ("Large epsilon (0.5)", 0.5),
    ]
    
    for strategy_name, init_value in strategies:
        print(f"\n{'='*60}")
        print(f"TESTING: {strategy_name}")
        print(f"{'='*60}")
        
        # Initialize Pi matrix
        if init_value == 0.0:
            Pi = torch.zeros((C, S), dtype=dtype, device=device)
        else:
            Pi = torch.full((C, S), init_value, dtype=dtype, device=device)
        
        # Set leaf probabilities
        for c in range(C):
            clade = ccp.id_to_clade[c]
            if clade.is_leaf():
                mapped_species = torch.nonzero(clade_species_map[c] > 0, as_tuple=False).flatten()
                if len(mapped_species) > 0:
                    if init_value == 0.0:
                        Pi[c, :] = 0.0  # Keep zeros for non-mapped
                        Pi[c, mapped_species] = 1.0 / len(mapped_species)
                    else:
                        Pi[c, :] = init_value  # Small value for non-mapped
                        Pi[c, mapped_species] = 1.0 / len(mapped_species)
        
        print(f"After initialization:")
        zeros_init = (Pi == 0.0).sum()
        print(f"  Exact zeros: {zeros_init}")
        print(f"  Minimum value: {Pi.min():.2e}")
        print(f"  Maximum value: {Pi.max():.6f}")
        
        # Initialize extinction probabilities properly
        E = torch.zeros(S, dtype=dtype, device=device)
        
        # Run Pi updates
        n_iterations = 100
        print(f"\nRunning {n_iterations} iterations...")
        
        for iteration in range(n_iterations):
            # Compute extinction probabilities properly
            E_new, E_s1, E_s2, Ebar = E_step(
                E, species_helpers["s_C1"], species_helpers["s_C2"],
                species_helpers["Recipients_mat"], p_S, p_D, p_T, p_L
            )
            E = E_new
            
            # Update Pi using parallel implementation
            Pi_new = Pi_update_ccp_parallel(
                Pi, ccp_helpers, species_helpers, clade_species_map,
                E, Ebar, p_S, p_D, p_T
            )
            
            # Check for zeros and very small values
            exact_zeros = (Pi_new == 0.0).sum()
            tiny_values = (Pi_new < 1e-15).sum()
            small_values = (Pi_new < 1e-10).sum()
            
            # Show progress at key iterations
            if iteration < 5 or iteration % 20 == 19 or iteration == n_iterations - 1:
                print(f"  Iteration {iteration + 1:3d}:")
                print(f"    Exact zeros:     {exact_zeros:4d} / {C*S}")
                print(f"    Values < 1e-15: {tiny_values:4d} / {C*S}")
                print(f"    Values < 1e-10: {small_values:4d} / {C*S}")
                print(f"    Min value: {Pi_new.min():.2e}")
                print(f"    Max value: {Pi_new.max():.6f}")
                
                # Show distribution of values
                if exact_zeros > 0:
                    nonzero_values = Pi_new[Pi_new > 0]
                    if len(nonzero_values) > 0:
                        print(f"    Non-zero min: {nonzero_values.min():.2e}")
                        print(f"    Non-zero max: {nonzero_values.max():.6f}")
            
            # Check convergence
            if iteration > 0:
                diff = torch.abs(Pi_new - Pi).max()
                if diff < 1e-12:
                    print(f"\n  Converged at iteration {iteration + 1} (diff = {diff:.2e})")
                    Pi = Pi_new
                    break
            
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
            print(f"  ⚠️  Found exact zeros in parallel implementation!")
            # Show some example zero positions
            zero_positions = torch.nonzero(Pi == 0.0, as_tuple=False)
            print(f"  Example zero positions (clade, species):")
            for i in range(min(10, len(zero_positions))):
                c, s = zero_positions[i]
                clade_name = str(ccp.id_to_clade[c])[:30]  # Truncate long names
                print(f"    [{c:2d},{s:2d}] {clade_name}")
        else:
            print(f"  ✅ No exact zeros found - biologically correct!")

def test_parameter_sensitivity():
    """
    Test how different parameter values affect zero production.
    """
    print(f"\n{'='*60}")
    print("TESTING PARAMETER SENSITIVITY")
    print(f"{'='*60}")
    
    species_path = "test_trees_1/sp.nwk"
    gene_path = "test_trees_1/g.nwk"
    
    device = torch.device("cpu")
    dtype = torch.float64
    
    # Build structures once
    ccp = build_ccp_from_single_tree(gene_path)
    species_helpers = build_species_helpers(species_path, device, dtype)
    clade_species_map = build_clade_species_mapping(ccp, species_helpers, device, dtype)
    ccp_helpers = build_ccp_helpers(ccp, device, dtype)
    
    C = len(ccp.clades)
    S = species_helpers["S"]
    
    # Test different parameter regimes
    parameter_sets = [
        ("Very small rates", 0.001, 0.001, 0.001),
        ("Small rates", 0.01, 0.01, 0.01),
        ("Moderate rates", 0.1, 0.1, 0.1),
        ("High rates", 1.0, 1.0, 1.0),
        ("Very high rates", 10.0, 10.0, 10.0),
    ]
    
    for param_name, delta, tau, lambda_param in parameter_sets:
        print(f"\n{param_name}: δ={delta}, τ={tau}, λ={lambda_param}")
        
        rates_sum = 1.0 + delta + tau + lambda_param
        p_S = 1.0 / rates_sum
        p_D = delta / rates_sum
        p_T = tau / rates_sum
        p_L = lambda_param / rates_sum
        
        # Initialize with zeros
        Pi = torch.zeros((C, S), dtype=dtype, device=device)
        
        # Set leaf probabilities
        for c in range(C):
            clade = ccp.id_to_clade[c]
            if clade.is_leaf():
                mapped_species = torch.nonzero(clade_species_map[c] > 0, as_tuple=False).flatten()
                if len(mapped_species) > 0:
                    Pi[c, mapped_species] = 1.0 / len(mapped_species)
        
        # Initialize E
        E = torch.zeros(S, dtype=dtype, device=device)
        
        # Run 20 iterations
        for iteration in range(20):
            E_new, E_s1, E_s2, Ebar = E_step(
                E, species_helpers["s_C1"], species_helpers["s_C2"],
                species_helpers["Recipients_mat"], p_S, p_D, p_T, p_L
            )
            E = E_new
            
            Pi = Pi_update_ccp_parallel(
                Pi, ccp_helpers, species_helpers, clade_species_map,
                E, Ebar, p_S, p_D, p_T
            )
        
        # Report final zero count
        exact_zeros = (Pi == 0.0).sum()
        min_nonzero = Pi[Pi > 0].min() if (Pi > 0).any() else 0.0
        print(f"  Final exact zeros: {exact_zeros} / {C*S} ({100*exact_zeros/(C*S):.1f}%)")
        print(f"  Min non-zero value: {min_nonzero:.2e}")

if __name__ == "__main__":
    test_parallel_zeros()
    test_parameter_sensitivity()