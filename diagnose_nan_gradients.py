#!/usr/bin/env python3
"""
Diagnostic script to identify where NaN gradients appear in Pi_update_ccp_log.
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


def test_gradient_flow():
    """Test gradient flow through Pi_update_ccp_log with anomaly detection."""
    
    print("=" * 80)
    print("Diagnosing NaN Gradients in Pi_update_ccp_log")
    print("=" * 80)
    
    # Enable anomaly detection
    torch.autograd.set_detect_anomaly(True)
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    species_path = "test_trees_3/sp.nwk"
    gene_path = "test_trees_3/g.nwk"
    
    # Build structures
    print("\nBuilding structures...")
    ccp = build_ccp_from_single_tree(gene_path)
    species_helpers = build_species_helpers(species_path, device, torch.float64)
    clade_species_map = build_clade_species_mapping(ccp, species_helpers, device, torch.float64)
    ccp_helpers = build_ccp_helpers(ccp, device, torch.float64)
    root_clade_id = get_root_clade_id(ccp)
    
    print(f"  {len(ccp.clades)} clades, {species_helpers['S']} species")
    
    # Create parameters that require gradients
    log_delta = torch.tensor(-2.87, requires_grad=True, dtype=torch.float64, device=device)
    log_tau = torch.tensor(-6.9, requires_grad=True, dtype=torch.float64, device=device)
    log_lambda = torch.tensor(-6.9, requires_grad=True, dtype=torch.float64, device=device)
    
    # Compute rates using softplus
    delta = torch.nn.functional.softplus(log_delta)
    tau = torch.nn.functional.softplus(log_tau)
    lambda_param = torch.nn.functional.softplus(log_lambda)
    
    print(f"\nRates: δ={delta.item():.6f}, τ={tau.item():.6f}, λ={lambda_param.item():.6f}")
    
    # Compute probabilities
    rates_sum = 1.0 + delta + tau + lambda_param
    p_S = 1.0 / rates_sum
    p_D = delta / rates_sum
    p_T = tau / rates_sum
    p_L = lambda_param / rates_sum
    
    print(f"Probabilities: p_S={p_S.item():.6f}, p_D={p_D.item():.6f}, p_T={p_T.item():.6f}, p_L={p_L.item():.6f}")
    
    # Initialize E (converged)
    print("\nConverging E...")
    E = torch.zeros(species_helpers["S"], dtype=torch.float64, device=device)
    with torch.no_grad():
        for i in range(20):
            E_new, _, _, Ebar = E_step(
                E, species_helpers["s_C1"], species_helpers["s_C2"],
                species_helpers["Recipients_mat"], 
                p_S.detach(), p_D.detach(), p_T.detach(), p_L.detach()
            )
            E = E_new
    print(f"E converged: min={E.min():.6f}, max={E.max():.6f}")
    
    # Initialize log_Pi
    log_Pi = torch.full((ccp_helpers["C"], species_helpers["S"]), 
                        float('-inf'), dtype=torch.float64, device=device)
    
    # Initialize leaves
    for c in range(ccp_helpers["C"]):
        clade = ccp_helpers['ccp'].id_to_clade[c]
        if clade.is_leaf():
            mapped_species = torch.nonzero(clade_species_map[c] > 0, as_tuple=False).flatten()
            if len(mapped_species) > 0:
                log_Pi[c, mapped_species] = -np.log(len(mapped_species))
    
    # Converge log_Pi
    print("\nConverging log_Pi...")
    with torch.no_grad():
        for i in range(20):
            log_Pi_new = Pi_update_ccp_log(
                log_Pi, ccp_helpers, species_helpers,
                clade_species_map, E, Ebar, 
                p_S.detach(), p_D.detach(), p_T.detach()
            )
            log_Pi = log_Pi_new
    
    print(f"log_Pi converged: finite values = {torch.isfinite(log_Pi).sum()}/{log_Pi.numel()}")
    
    # Now test gradient flow
    print("\n" + "=" * 80)
    print("Testing gradient flow with anomaly detection...")
    print("=" * 80)
    
    try:
        # Single update with gradients
        E_new, _, _, Ebar = E_step(
            E.detach(), species_helpers["s_C1"], species_helpers["s_C2"],
            species_helpers["Recipients_mat"], p_S, p_D, p_T, p_L
        )
        
        print("E_step completed without NaN")
        
        # The critical test - Pi update
        log_Pi_new = Pi_update_ccp_log(
            log_Pi.detach(), ccp_helpers, species_helpers,
            clade_species_map, E_new, Ebar, p_S, p_D, p_T
        )
        
        print("Pi_update_ccp_log completed")
        
        # Compute log-likelihood
        root_log_pi = log_Pi_new[root_clade_id, :]
        log_likelihood = torch.logsumexp(root_log_pi, dim=0)
        
        print(f"Log-likelihood: {log_likelihood.item():.6f}")
        
        # Compute gradients
        print("\nComputing gradients...")
        log_likelihood.backward()
        
        print("Gradients computed successfully!")
        print(f"  log_delta.grad: {log_delta.grad}")
        print(f"  log_tau.grad: {log_tau.grad}")
        print(f"  log_lambda.grad: {log_lambda.grad}")
        
    except RuntimeError as e:
        print(f"\n!!! RuntimeError: {e}")
        print("\nThis should show us exactly where the NaN first appears.")
    
    finally:
        torch.autograd.set_detect_anomaly(False)


if __name__ == '__main__':
    test_gradient_flow()