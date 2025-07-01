#!/usr/bin/env python3
"""
Test to verify that log-space -inf positions match linear-space zero positions.
"""

import torch
from matmul_ale_ccp import (
    build_ccp_from_single_tree as build_ccp_linear,
    build_species_helpers as build_species_helpers_linear,
    build_clade_species_mapping as build_clade_species_mapping_linear,
    build_ccp_helpers as build_ccp_helpers_linear,
    Pi_update_ccp as Pi_update_ccp_linear,
    E_step as E_step_linear
)
from matmul_ale_ccp_log import (
    build_ccp_from_single_tree as build_ccp_log,
    build_species_helpers as build_species_helpers_log,
    build_clade_species_mapping as build_clade_species_mapping_log,
    build_ccp_helpers as build_ccp_helpers_log,
    Pi_update_ccp_log as Pi_update_ccp_log
)

def test_zero_positions_consistency():
    """
    Test that -inf positions in log space match zero positions in linear space.
    """
    print("🔍 Testing Zero Position Consistency: Linear vs Log Space")
    print("=" * 60)
    
    # Test parameters
    species_path = "test_trees_1/sp.nwk"
    gene_path = "test_trees_1/g.nwk"
    
    # Parameters for both implementations
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
    print(f"Event probabilities: p_S={p_S:.4f}, p_D={p_D:.4f}, p_T={p_T:.4f}")
    
    # Build CCP structures for both implementations
    print("\nBuilding CCP structures...")
    
    # Linear space
    ccp_linear = build_ccp_linear(gene_path)
    species_helpers_linear = build_species_helpers_linear(species_path, device, dtype)
    clade_species_map_linear = build_clade_species_mapping_linear(ccp_linear, species_helpers_linear, device, dtype)
    ccp_helpers_linear = build_ccp_helpers_linear(ccp_linear, device, dtype)
    
    # Log space
    ccp_log = build_ccp_log(gene_path)
    species_helpers_log = build_species_helpers_log(species_path, device, dtype)
    clade_species_map_log = build_clade_species_mapping_log(ccp_log, species_helpers_log, device, dtype)
    ccp_helpers_log = build_ccp_helpers_log(ccp_log, device, dtype)
    
    C = len(ccp_linear.clades)
    S = species_helpers_linear["S"]
    
    print(f"Matrix dimensions: {C} clades × {S} species = {C*S} elements")
    
    # Test different initialization strategies
    strategies = [
        ("Zero initialization", "zero"),
        ("Large epsilon (0.5)", 0.5),
    ]
    
    for strategy_name, init_strategy in strategies:
        print(f"\n{'='*60}")
        print(f"TESTING: {strategy_name}")
        print(f"{'='*60}")
        
        # Initialize linear space Pi matrix
        if init_strategy == "zero":
            Pi_linear = torch.zeros((C, S), dtype=dtype, device=device)
        else:
            Pi_linear = torch.full((C, S), init_strategy, dtype=dtype, device=device)
        
        # Initialize log space Pi matrix
        log_Pi_log = torch.full((C, S), float('-inf'), dtype=dtype, device=device)
        
        # Set leaf probabilities for both
        for c in range(C):
            clade_linear = ccp_linear.id_to_clade[c]
            clade_log = ccp_log.id_to_clade[c]
            
            if clade_linear.is_leaf():
                # Linear space
                mapped_species_linear = torch.nonzero(clade_species_map_linear[c] > 0, as_tuple=False).flatten()
                if len(mapped_species_linear) > 0:
                    if init_strategy == "zero":
                        Pi_linear[c, :] = 0.0  # Keep zeros for non-mapped
                        Pi_linear[c, mapped_species_linear] = 1.0 / len(mapped_species_linear)
                    else:
                        Pi_linear[c, :] = init_strategy  # Small value for non-mapped
                        Pi_linear[c, mapped_species_linear] = 1.0 / len(mapped_species_linear)
                
                # Log space
                mapped_species_log = torch.nonzero(clade_species_map_log[c] > 0, as_tuple=False).flatten()
                if len(mapped_species_log) > 0:
                    log_prob_log = -torch.log(torch.tensor(len(mapped_species_log), dtype=dtype))
                    log_Pi_log[c, mapped_species_log] = log_prob_log
        
        # Run convergence for both implementations
        n_iterations = 100
        print(f"\nRunning {n_iterations} iterations to convergence...")
        
        # Initialize extinction probabilities properly
        E = torch.zeros(S, dtype=dtype, device=device)
        
        for iteration in range(n_iterations):
            # Linear space update - compute E properly using E_step
            E_new, E_s1, E_s2, Ebar = E_step_linear(
                E, species_helpers_linear["s_C1"], species_helpers_linear["s_C2"],
                species_helpers_linear["Recipients_mat"], p_S, p_D, p_T, p_L
            )
            E = E_new
            
            Pi_linear_new = Pi_update_ccp_linear(
                Pi_linear, ccp_helpers_linear, species_helpers_linear,
                clade_species_map_linear, E, Ebar, E_s1, E_s2, p_S, p_D, p_T
            )
            
            # Log space update - should also use proper E values
            log_Pi_log_new = Pi_update_ccp_log(
                log_Pi_log, ccp_helpers_log, species_helpers_log,
                clade_species_map_log, E, Ebar, p_S, p_D, p_T
            )
            
            # Check convergence
            linear_diff = torch.abs(Pi_linear_new - Pi_linear).max()
            log_diff = torch.abs(log_Pi_log_new - log_Pi_log).max()
            
            # Compare zero/inf patterns
            linear_zeros = (Pi_linear_new == 0.0)
            log_infs = torch.isinf(log_Pi_log_new) & (log_Pi_log_new < 0)  # -inf values
            
            # Check for exact 1.0 values
            linear_ones = (Pi_linear_new == 1.0)
            log_zeros = (log_Pi_log_new == 0.0)  # log(1) = 0
            
            zeros_match = torch.sum(linear_zeros == log_infs)
            ones_match = torch.sum(linear_ones == log_zeros)
            total_elements = C * S
            match_percentage = 100 * zeros_match / total_elements
            
            if iteration < 5 or iteration % 20 == 19 or iteration == n_iterations - 1 or (linear_diff < 1e-12 and log_diff < 1e-12):
                print(f"  Iteration {iteration + 1:2d}:")
                print(f"    Linear zeros:     {linear_zeros.sum():3d} / {total_elements}")
                print(f"    Log -inf values:  {log_infs.sum():3d} / {total_elements}")
                print(f"    Linear ones:      {linear_ones.sum():3d} / {total_elements}")
                print(f"    Log zeros (log1): {log_zeros.sum():3d} / {total_elements}")
                print(f"    Zero matches:     {zeros_match:3d} / {total_elements} ({match_percentage:.1f}%)")
                print(f"    One matches:      {ones_match:3d} / {total_elements}")
                print(f"    Linear conv diff: {linear_diff:.2e}")
                print(f"    Log conv diff:    {log_diff:.2e}")
                
                # Show value range statistics
                if not torch.all(linear_zeros):
                    nonzero_linear = Pi_linear_new[~linear_zeros]
                    print(f"    Linear range:     [{nonzero_linear.min():.2e}, {nonzero_linear.max():.2e}]")
                if not torch.all(log_infs):
                    finite_log = log_Pi_log_new[torch.isfinite(log_Pi_log_new)]
                    if len(finite_log) > 0:
                        print(f"    Log range:        [{finite_log.min():.2e}, {finite_log.max():.2e}]")
                
                if match_percentage < 100:
                    # Find mismatches
                    linear_only = linear_zeros & ~log_infs
                    log_only = log_infs & ~linear_zeros
                    print(f"    Linear-only zeros: {linear_only.sum()}")
                    print(f"    Log-only -inf:     {log_only.sum()}")
                    
                    # Show examples of mismatches
                    if linear_only.sum() > 0:
                        examples = torch.nonzero(linear_only, as_tuple=False)[:3]
                        print(f"    Linear-only examples: {examples.tolist()}")
                        for pos in examples:
                            c, s = pos[0], pos[1]
                            print(f"      [{c:2d},{s:2d}] linear={Pi_linear_new[c,s]:.2e}, log={log_Pi_log_new[c,s]:.2f}")
                    
                    if log_only.sum() > 0:
                        examples = torch.nonzero(log_only, as_tuple=False)[:3]
                        print(f"    Log-only examples: {examples.tolist()}")
                        for pos in examples:
                            c, s = pos[0], pos[1]
                            print(f"      [{c:2d},{s:2d}] linear={Pi_linear_new[c,s]:.2e}, log={log_Pi_log_new[c,s]:.2f}")
            
            Pi_linear = Pi_linear_new
            log_Pi_log = log_Pi_log_new
        
        # Final comparison
        print(f"\nFinal Results for {strategy_name}:")
        final_linear_zeros = (Pi_linear == 0.0)
        final_log_infs = torch.isinf(log_Pi_log) & (log_Pi_log < 0)
        
        final_matches = torch.sum(final_linear_zeros == final_log_infs)
        final_match_percentage = 100 * final_matches / total_elements
        
        final_linear_ones = (Pi_linear == 1.0)
        final_log_zeros = (log_Pi_log == 0.0)
        final_ones_matches = torch.sum(final_linear_ones == final_log_zeros)
        
        print(f"  Linear exact zeros:     {final_linear_zeros.sum()} / {total_elements}")
        print(f"  Log -inf values:        {final_log_infs.sum()} / {total_elements}")
        print(f"  Linear exact ones:      {final_linear_ones.sum()} / {total_elements}")
        print(f"  Log exact zeros(log1):  {final_log_zeros.sum()} / {total_elements}")
        print(f"  Zero position consistency: {final_matches} / {total_elements} ({final_match_percentage:.1f}%)")
        print(f"  One position consistency:  {final_ones_matches} / {total_elements}")
        
        # Show final value ranges
        if final_linear_zeros.sum() < total_elements:
            nonzero_final = Pi_linear[~final_linear_zeros]
            print(f"  Final linear range:     [{nonzero_final.min():.2e}, {nonzero_final.max():.2e}]")
        if final_log_infs.sum() < total_elements:
            finite_log_final = log_Pi_log[torch.isfinite(log_Pi_log)]
            if len(finite_log_final) > 0:
                print(f"  Final log range:        [{finite_log_final.min():.2e}, {finite_log_final.max():.2e}]")
        
        if final_match_percentage == 100:
            print(f"  ✅ PERFECT CONSISTENCY: Zero positions match exactly!")
        elif final_match_percentage > 95:
            print(f"  ⚠️  MOSTLY CONSISTENT: Small discrepancies detected")
        else:
            print(f"  ❌ INCONSISTENT: Significant position mismatches")
            
            # Detailed analysis of mismatches
            linear_only_final = final_linear_zeros & ~final_log_infs
            log_only_final = final_log_infs & ~final_linear_zeros
            
            print(f"\n  Mismatch Analysis:")
            print(f"    Zeros in linear but not log: {linear_only_final.sum()}")
            print(f"    -inf in log but not linear:  {log_only_final.sum()}")
            
            if linear_only_final.sum() > 0:
                print(f"    Linear-only zeros (first 5):")
                examples = torch.nonzero(linear_only_final, as_tuple=False)[:5]
                for pos in examples:
                    c, s = pos[0], pos[1]
                    print(f"      [{c:2d},{s:2d}] linear={Pi_linear[c,s]:.2e}, log={log_Pi_log[c,s]:.2f}")
            
            if log_only_final.sum() > 0:
                print(f"    Log-only -inf (first 5):")
                examples = torch.nonzero(log_only_final, as_tuple=False)[:5]
                for pos in examples:
                    c, s = pos[0], pos[1]
                    print(f"      [{c:2d},{s:2d}] linear={Pi_linear[c,s]:.2e}, log={log_Pi_log[c,s]:.2f}")

def test_small_parameter_regime():
    """
    Test consistency in the small parameter regime where more zeros are expected.
    """
    print(f"\n{'='*60}")
    print("TESTING SMALL PARAMETER REGIME")
    print(f"{'='*60}")
    
    species_path = "test_trees_1/sp.nwk"
    gene_path = "test_trees_1/g.nwk"
    
    # Very small parameters to maximize zeros
    delta, tau, lambda_param = 0.001, 0.001, 0.001
    rates_sum = 1.0 + delta + tau + lambda_param
    p_S = 1.0 / rates_sum
    p_D = delta / rates_sum
    p_T = tau / rates_sum
    p_L = lambda_param / rates_sum
    
    device = torch.device("cpu")
    dtype = torch.float64
    
    print(f"Small parameters: δ={delta}, τ={tau}, λ={lambda_param}")
    print(f"Event probabilities: p_S={p_S:.6f}, p_D={p_D:.6f}, p_T={p_T:.6f}")
    
    # Build structures
    ccp_linear = build_ccp_linear(gene_path)
    species_helpers_linear = build_species_helpers_linear(species_path, device, dtype)
    clade_species_map_linear = build_clade_species_mapping_linear(ccp_linear, species_helpers_linear, device, dtype)
    ccp_helpers_linear = build_ccp_helpers_linear(ccp_linear, device, dtype)
    
    ccp_log = build_ccp_log(gene_path)
    species_helpers_log = build_species_helpers_log(species_path, device, dtype)
    clade_species_map_log = build_clade_species_mapping_log(ccp_log, species_helpers_log, device, dtype)
    ccp_helpers_log = build_ccp_helpers_log(ccp_log, device, dtype)
    
    C = len(ccp_linear.clades)
    S = species_helpers_linear["S"]
    
    # Zero initialization for maximum zero preservation
    Pi_linear = torch.zeros((C, S), dtype=dtype, device=device)
    log_Pi_log = torch.full((C, S), float('-inf'), dtype=dtype, device=device)
    
    # Set leaf probabilities
    for c in range(C):
        clade = ccp_linear.id_to_clade[c]
        if clade.is_leaf():
            mapped_species = torch.nonzero(clade_species_map_linear[c] > 0, as_tuple=False).flatten()
            if len(mapped_species) > 0:
                # Linear
                Pi_linear[c, mapped_species] = 1.0 / len(mapped_species)
                # Log
                log_prob = -torch.log(torch.tensor(len(mapped_species), dtype=dtype))
                log_Pi_log[c, mapped_species] = log_prob
    
    print(f"\nInitial state:")
    init_linear_zeros = (Pi_linear == 0.0).sum()
    init_log_infs = (torch.isinf(log_Pi_log) & (log_Pi_log < 0)).sum()
    print(f"  Linear zeros: {init_linear_zeros} / {C*S}")
    print(f"  Log -inf:     {init_log_infs} / {C*S}")
    
    # Run to convergence
    print(f"\nConverging...")
    E = torch.zeros(S, dtype=dtype, device=device)
    
    for iteration in range(20):
        # Compute extinction probabilities properly
        E_new, E_s1, E_s2, Ebar = E_step_linear(
            E, species_helpers_linear["s_C1"], species_helpers_linear["s_C2"],
            species_helpers_linear["Recipients_mat"], p_S, p_D, p_T, p_L
        )
        E = E_new
        
        Pi_linear_new = Pi_update_ccp_linear(
            Pi_linear, ccp_helpers_linear, species_helpers_linear,
            clade_species_map_linear, E, Ebar, E_s1, E_s2, p_S, p_D, p_T
        )
        
        log_Pi_log_new = Pi_update_ccp_log(
            log_Pi_log, ccp_helpers_log, species_helpers_log,
            clade_species_map_log, E, Ebar, p_S, p_D, p_T
        )
        
        # Check convergence
        linear_diff = torch.abs(Pi_linear_new - Pi_linear).max()
        log_diff = torch.abs(log_Pi_log_new - log_Pi_log).max()
        
        # Continue without breaking for convergence
        Pi_linear = Pi_linear_new
        log_Pi_log = log_Pi_log_new
    
    # Final comparison
    final_linear_zeros = (Pi_linear == 0.0)
    final_log_infs = torch.isinf(log_Pi_log) & (log_Pi_log < 0)
    
    matches = torch.sum(final_linear_zeros == final_log_infs)
    match_percentage = 100 * matches / (C * S)
    
    print(f"\nFinal Results (Small Parameter Regime):")
    print(f"  Linear exact zeros: {final_linear_zeros.sum()} / {C*S} ({100*final_linear_zeros.sum()/(C*S):.1f}%)")
    print(f"  Log -inf values:    {final_log_infs.sum()} / {C*S} ({100*final_log_infs.sum()/(C*S):.1f}%)")
    print(f"  Position matches:   {matches} / {C*S} ({match_percentage:.1f}%)")
    
    if match_percentage == 100:
        print(f"  ✅ PERFECT: Zero positions match exactly in small parameter regime!")
    else:
        print(f"  ❌ MISMATCH: Zero positions don't match in small parameter regime")

if __name__ == "__main__":
    test_zero_positions_consistency()
    test_small_parameter_regime()