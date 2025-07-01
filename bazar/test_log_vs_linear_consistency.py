#!/usr/bin/env python3
"""
Test to verify that matmul_ale_ccp_log.py produces log(Pi) where Pi comes from matmul_ale_ccp.py
"""

import torch
import numpy as np
from matmul_ale_ccp import (
    build_ccp_from_single_tree as build_ccp_linear,
    build_species_helpers as build_species_helpers_linear,
    build_clade_species_mapping as build_clade_species_mapping_linear,
    build_ccp_helpers as build_ccp_helpers_linear,
    Pi_update_ccp as Pi_update_ccp_linear
)
from matmul_ale_ccp_log import (
    build_ccp_from_single_tree as build_ccp_log,
    build_species_helpers as build_species_helpers_log,
    build_clade_species_mapping as build_clade_species_mapping_log,
    build_ccp_helpers as build_ccp_helpers_log,
    Pi_update_ccp_log as Pi_update_ccp_log
)

def test_log_vs_linear_consistency():
    """
    Test that log-space implementation produces log of linear-space implementation.
    """
    print("🧪 Testing Log vs Linear Space Consistency")
    print("=" * 50)
    
    # Test parameters
    species_path = "test_trees_1/sp.nwk"
    gene_path = "test_trees_1/g.nwk"
    
    # Parameters for both implementations
    delta = 0.1
    tau = 0.05
    lambda_param = 0.1
    
    # Event probabilities
    rates_sum = 1.0 + delta + tau + lambda_param
    p_S = 1.0 / rates_sum
    p_D = delta / rates_sum
    p_T = tau / rates_sum
    
    device = torch.device("cpu")
    dtype = torch.float64
    
    print(f"Using parameters: δ={delta}, τ={tau}, λ={lambda_param}")
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
    
    # Verify CCP structures are equivalent
    print("Verifying CCP structure consistency...")
    assert len(ccp_linear.clades) == len(ccp_log.clades), "Different number of clades"
    assert species_helpers_linear["S"] == species_helpers_log["S"], "Different number of species"
    
    C = len(ccp_linear.clades)
    S = species_helpers_linear["S"]
    
    print(f"Number of clades: {C}")
    print(f"Number of species: {S}")
    
    # Initialize Pi matrices
    print("\nInitializing Pi matrices...")
    
    # Linear space: initialize with small positive values to avoid log(0)
    Pi_linear = torch.full((C, S), 1e-10, dtype=dtype, device=device)
    
    # Log space: initialize with -inf (log(0))
    log_Pi_log = torch.full((C, S), float('-inf'), dtype=dtype, device=device)
    
    # Set leaf probabilities for both
    for c in range(C):
        clade_linear = ccp_linear.id_to_clade[c]
        clade_log = ccp_log.id_to_clade[c]
        
        # Verify clades are the same
        assert clade_linear.is_leaf() == clade_log.is_leaf(), f"Clade {c} leaf status mismatch"
        
        if clade_linear.is_leaf():
            # Linear space
            mapped_species_linear = torch.nonzero(clade_species_map_linear[c] > 0, as_tuple=False).flatten()
            if len(mapped_species_linear) > 0:
                prob_linear = 1.0 / len(mapped_species_linear)
                Pi_linear[c, :] = 1e-10  # Small value for non-mapped
                Pi_linear[c, mapped_species_linear] = prob_linear
            
            # Log space
            mapped_species_log = torch.nonzero(clade_species_map_log[c] > 0, as_tuple=False).flatten()
            if len(mapped_species_log) > 0:
                log_prob_log = -torch.log(torch.tensor(len(mapped_species_log), dtype=dtype))
                log_Pi_log[c, mapped_species_log] = log_prob_log
    
    print("Initial leaf probabilities set")
    
    # Run one Pi update iteration for both
    print("\nRunning Pi update iterations...")
    
    # Linear space update
    # Need E, Ebar, E_s1, E_s2 for linear version
    S = species_helpers_linear["S"]
    E = torch.zeros(S, dtype=dtype, device=device)
    E_s1 = torch.zeros(S, dtype=dtype, device=device)
    E_s2 = torch.zeros(S, dtype=dtype, device=device)
    Ebar = torch.zeros(S, dtype=dtype, device=device)
    
    Pi_linear_updated = Pi_update_ccp_linear(
        Pi_linear, ccp_helpers_linear, species_helpers_linear, 
        clade_species_map_linear, E, Ebar, E_s1, E_s2, p_S, p_D, p_T
    )
    
    # Log space update  
    log_Pi_log_updated = Pi_update_ccp_log(
        log_Pi_log, ccp_helpers_log, species_helpers_log,
        clade_species_map_log, torch.zeros(S, dtype=dtype, device=device),  # E=0 for simplicity
        torch.zeros(S, dtype=dtype, device=device),  # Ebar=0
        p_S, p_D, p_T
    )
    
    print("Pi updates completed")
    
    # Convert linear to log for comparison
    print("\nComparing results...")
    
    # Avoid log(0) by using a small minimum value
    Pi_linear_safe = torch.clamp(Pi_linear_updated, min=1e-50)
    log_Pi_from_linear = torch.log(Pi_linear_safe)
    
    # Compare where both are finite
    finite_mask_linear = Pi_linear_updated > 1e-50
    finite_mask_log = torch.isfinite(log_Pi_log_updated)
    
    print(f"Finite values in linear space: {finite_mask_linear.sum()}/{C*S}")
    print(f"Finite values in log space: {finite_mask_log.sum()}/{C*S}")
    
    # Find positions where both are finite for comparison
    both_finite = finite_mask_linear & finite_mask_log
    print(f"Both finite: {both_finite.sum()}/{C*S}")
    
    if both_finite.sum() > 0:
        # Extract values where both are finite
        log_from_linear_finite = log_Pi_from_linear[both_finite]
        log_direct_finite = log_Pi_log_updated[both_finite]
        
        # Compute differences
        abs_diff = torch.abs(log_from_linear_finite - log_direct_finite)
        rel_diff = abs_diff / (torch.abs(log_direct_finite) + 1e-15)
        
        print(f"\nNumerical comparison on {both_finite.sum()} finite values:")
        print(f"Max absolute difference: {abs_diff.max():.2e}")
        print(f"Mean absolute difference: {abs_diff.mean():.2e}")
        print(f"Max relative difference: {rel_diff.max():.2e}")
        print(f"Mean relative difference: {rel_diff.mean():.2e}")
        
        # Check if differences are within tolerance
        abs_tol = 1e-12
        rel_tol = 1e-10
        
        abs_ok = abs_diff.max() < abs_tol
        rel_ok = rel_diff.max() < rel_tol
        
        print(f"\nTolerance check:")
        print(f"Absolute tolerance ({abs_tol:.0e}): {'✅ PASS' if abs_ok else '❌ FAIL'}")
        print(f"Relative tolerance ({rel_tol:.0e}): {'✅ PASS' if rel_ok else '❌ FAIL'}")
        
        if not (abs_ok and rel_ok):
            print("\n⚠️  Large differences detected!")
            print("Sample comparisons:")
            for i in range(min(10, len(log_from_linear_finite))):
                idx = torch.nonzero(both_finite, as_tuple=False)[i]
                c, s = idx[0], idx[1]
                linear_val = Pi_linear_updated[c, s]
                log_val = log_Pi_log_updated[c, s]
                log_from_lin = log_Pi_from_linear[c, s]
                print(f"  [{c:2d},{s:2d}] linear={linear_val:.6e} → log={log_from_lin:.6f}, direct_log={log_val:.6f}, diff={abs_diff[i]:.2e}")
    
    # Check positions where one is finite but the other isn't
    linear_only = finite_mask_linear & ~finite_mask_log
    log_only = ~finite_mask_linear & finite_mask_log
    
    print(f"\nMismatch analysis:")
    print(f"Finite in linear but not log: {linear_only.sum()}")
    print(f"Finite in log but not linear: {log_only.sum()}")
    
    if linear_only.sum() > 0:
        print("Sample linear-only finite values:")
        linear_only_indices = torch.nonzero(linear_only, as_tuple=False)[:5]
        for idx in linear_only_indices:
            c, s = idx[0], idx[1]
            print(f"  [{c:2d},{s:2d}] linear={Pi_linear_updated[c,s]:.6e}, log={log_Pi_log_updated[c,s]}")
    
    if log_only.sum() > 0:
        print("Sample log-only finite values:")
        log_only_indices = torch.nonzero(log_only, as_tuple=False)[:5]
        for idx in log_only_indices:
            c, s = idx[0], idx[1]
            print(f"  [{c:2d},{s:2d}] linear={Pi_linear_updated[c,s]:.6e}, log={log_Pi_log_updated[c,s]:.6f}")
    
    # Overall assessment
    print(f"\n{'='*50}")
    if both_finite.sum() > 0 and abs_diff.max() < 1e-10 and rel_diff.max() < 1e-8:
        print("✅ CONSISTENCY TEST PASSED")
        print("Log-space implementation produces log of linear-space implementation")
    else:
        print("❌ CONSISTENCY TEST FAILED")
        print("Significant differences detected between implementations")
    
    return {
        'both_finite': both_finite.sum().item(),
        'max_abs_diff': abs_diff.max().item() if both_finite.sum() > 0 else float('inf'),
        'max_rel_diff': rel_diff.max().item() if both_finite.sum() > 0 else float('inf'),
        'linear_only': linear_only.sum().item(),
        'log_only': log_only.sum().item()
    }

def test_multiple_iterations():
    """
    Test consistency over multiple iterations.
    """
    print("\n" + "="*50)
    print("🧪 Testing Multiple Iterations")
    print("="*50)
    
    species_path = "test_trees_1/sp.nwk"
    gene_path = "test_trees_1/g.nwk"
    
    delta, tau, lambda_param = 0.05, 0.03, 0.08
    rates_sum = 1.0 + delta + tau + lambda_param
    p_S = 1.0 / rates_sum
    p_D = delta / rates_sum
    p_T = tau / rates_sum
    
    device = torch.device("cpu")
    dtype = torch.float64
    
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
    
    # Initialize
    Pi_linear = torch.full((C, S), 1e-10, dtype=dtype, device=device)
    log_Pi_log = torch.full((C, S), float('-inf'), dtype=dtype, device=device)
    
    # Set leaves
    for c in range(C):
        clade = ccp_linear.id_to_clade[c]
        if clade.is_leaf():
            mapped_species = torch.nonzero(clade_species_map_linear[c] > 0, as_tuple=False).flatten()
            if len(mapped_species) > 0:
                prob = 1.0 / len(mapped_species)
                Pi_linear[c, :] = 1e-10
                Pi_linear[c, mapped_species] = prob
                
                log_prob = -torch.log(torch.tensor(len(mapped_species), dtype=dtype))
                log_Pi_log[c, mapped_species] = log_prob
    
    # Run multiple iterations
    n_iters = 5
    results = []
    
    for iteration in range(n_iters):
        print(f"\nIteration {iteration + 1}/{n_iters}")
        
        # Update both
        E = torch.zeros(S, dtype=dtype, device=device)
        E_s1 = torch.zeros(S, dtype=dtype, device=device)
        E_s2 = torch.zeros(S, dtype=dtype, device=device)
        Ebar = torch.zeros(S, dtype=dtype, device=device)
        
        Pi_linear = Pi_update_ccp_linear(
            Pi_linear, ccp_helpers_linear, species_helpers_linear,
            clade_species_map_linear, E, Ebar, E_s1, E_s2, p_S, p_D, p_T
        )
        
        log_Pi_log = Pi_update_ccp_log(
            log_Pi_log, ccp_helpers_log, species_helpers_log,
            clade_species_map_log, torch.zeros(S, dtype=dtype, device=device),
            torch.zeros(S, dtype=dtype, device=device), p_S, p_D, p_T
        )
        
        # Compare
        Pi_safe = torch.clamp(Pi_linear, min=1e-50)
        log_from_linear = torch.log(Pi_safe)
        
        finite_linear = Pi_linear > 1e-50
        finite_log = torch.isfinite(log_Pi_log)
        both_finite = finite_linear & finite_log
        
        if both_finite.sum() > 0:
            abs_diff = torch.abs(log_from_linear[both_finite] - log_Pi_log[both_finite])
            max_diff = abs_diff.max().item()
            results.append(max_diff)
            print(f"  Max absolute difference: {max_diff:.2e}")
        else:
            print("  No overlapping finite values")
            results.append(float('inf'))
    
    print(f"\nIteration summary:")
    for i, diff in enumerate(results):
        status = "✅" if diff < 1e-10 else "⚠️" if diff < 1e-8 else "❌"
        print(f"  Iteration {i+1}: {diff:.2e} {status}")
    
    return results

if __name__ == "__main__":
    # Run basic consistency test
    try:
        results = test_log_vs_linear_consistency()
        
        # Run multiple iteration test
        iter_results = test_multiple_iterations()
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()