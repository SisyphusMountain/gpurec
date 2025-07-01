#!/usr/bin/env python3
"""
Diagnose why log_Pi still has -inf values after convergence.
"""

import torch
import numpy as np

from matmul_ale_ccp import (
    build_ccp_from_single_tree,
    build_species_helpers,
    build_clade_species_mapping,
    build_ccp_helpers,
    get_root_clade_id,
    E_step
)
from matmul_ale_ccp_log import Pi_update_ccp_log


def diagnose_convergence():
    """Check convergence and -inf values in log_Pi."""
    
    print("=" * 80)
    print("Diagnosing log_Pi convergence on test_trees_1")
    print("=" * 80)
    
    # Setup - use command line argument or default to test_trees_1
    import sys
    test_dir = sys.argv[1] if len(sys.argv) > 1 else "test_trees_1"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    species_path = f"{test_dir}/sp.nwk"
    gene_path = f"{test_dir}/g.nwk"
    
    # Build structures
    print("\nBuilding structures...")
    ccp = build_ccp_from_single_tree(gene_path)
    species_helpers = build_species_helpers(species_path, device, torch.float64)
    clade_species_map = build_clade_species_mapping(ccp, species_helpers, device, torch.float64)
    ccp_helpers = build_ccp_helpers(ccp, device, torch.float64)
    root_clade_id = get_root_clade_id(ccp)
    
    print(f"  {len(ccp.clades)} clades, {species_helpers['S']} species")
    
    # Set parameters
    delta = 0.1
    tau = 0.1
    lambda_param = 0.1
    
    # Compute probabilities
    rates_sum = 1.0 + delta + tau + lambda_param
    p_S = 1.0 / rates_sum
    p_D = delta / rates_sum
    p_T = tau / rates_sum
    p_L = lambda_param / rates_sum
    
    print(f"\nRates: δ={delta}, τ={tau}, λ={lambda_param}")
    print(f"Probabilities: p_S={p_S:.6f}, p_D={p_D:.6f}, p_T={p_T:.6f}, p_L={p_L:.6f}")
    
    # Initialize and converge E
    print("\nConverging E...")
    E = torch.zeros(species_helpers["S"], dtype=torch.float64, device=device)
    
    for i in range(100):
        E_old = E.clone()
        E_new, _, _, Ebar = E_step(
            E, species_helpers["s_C1"], species_helpers["s_C2"],
            species_helpers["Recipients_mat"], p_S, p_D, p_T, p_L
        )
        E = E_new
        
        diff = torch.abs(E - E_old).max()
        if diff < 1e-10:
            print(f"  E converged after {i+1} iterations")
            break
    
    print(f"  E values: {E}")
    print(f"  E range: [{E.min():.6f}, {E.max():.6f}]")
    
    # Initialize log_Pi
    log_Pi = torch.full((ccp_helpers["C"], species_helpers["S"]), 
                        float('-inf'), dtype=torch.float64, device=device)
    
    # Initialize leaves
    print("\nInitializing leaf clades...")
    leaf_count = 0
    for c in range(ccp_helpers["C"]):
        clade = ccp_helpers['ccp'].id_to_clade[c]
        if clade.is_leaf():
            leaf_count += 1
            mapped_species = torch.nonzero(clade_species_map[c] > 0, as_tuple=False).flatten()
            if len(mapped_species) > 0:
                log_Pi[c, mapped_species] = -np.log(len(mapped_species))
                print(f"  Leaf clade {c}: {clade} -> species {mapped_species.tolist()}")
    
    print(f"  Initialized {leaf_count} leaf clades")
    print(f"  Initial -inf count: {torch.isinf(log_Pi).sum()}/{log_Pi.numel()}")
    
    # Converge log_Pi
    print("\nConverging log_Pi...")
    for i in range(100):
        log_Pi_old = log_Pi.clone()
        log_Pi_new = Pi_update_ccp_log(
            log_Pi, ccp_helpers, species_helpers,
            clade_species_map, E, Ebar, p_S, p_D, p_T
        )
        log_Pi = log_Pi_new
        
        # Check convergence on finite values
        finite_mask = torch.isfinite(log_Pi) & torch.isfinite(log_Pi_old)
        if finite_mask.any():
            max_diff = torch.abs(log_Pi[finite_mask] - log_Pi_old[finite_mask]).max()
            inf_count = torch.isinf(log_Pi).sum()
            
            if i % 10 == 0 or max_diff < 1e-10:
                print(f"  Iter {i+1}: max_diff={max_diff:.2e}, -inf count={inf_count}/{log_Pi.numel()}")
            
            if max_diff < 1e-10:
                print(f"  log_Pi converged after {i+1} iterations")
                break
    
    # Analyze final state
    print("\nFinal analysis:")
    inf_mask = torch.isinf(log_Pi)
    print(f"  Total -inf values: {inf_mask.sum()}/{log_Pi.numel()}")
    print(f"  Finite values: {torch.isfinite(log_Pi).sum()}/{log_Pi.numel()}")
    
    # Check which clades have all -inf
    print("\n  Clades with all -inf values:")
    for c in range(ccp_helpers["C"]):
        if torch.all(inf_mask[c, :]):
            clade = ccp_helpers['ccp'].id_to_clade[c]
            print(f"    Clade {c}: {clade}")
    
    # Check log_Pi values for each clade
    print("\n  log_Pi ranges for each clade:")
    for c in range(min(10, ccp_helpers["C"])):  # Show first 10
        clade = ccp_helpers['ccp'].id_to_clade[c]
        finite_vals = log_Pi[c, torch.isfinite(log_Pi[c, :])]
        if len(finite_vals) > 0:
            print(f"    Clade {c}: min={finite_vals.min():.3f}, max={finite_vals.max():.3f}, " + 
                  f"-inf count={inf_mask[c, :].sum()}/{species_helpers['S']}")
        else:
            print(f"    Clade {c}: all -inf")
    
    # Test if this is expected behavior
    print("\nIs this expected?")
    print("  If clades represent impossible configurations (e.g., wrong leaf mappings),")
    print("  then -inf values might be correct. But this would still cause NaN gradients.")
    
    return log_Pi, E, ccp_helpers, species_helpers


if __name__ == '__main__':
    diagnose_convergence()