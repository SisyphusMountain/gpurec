#!/usr/bin/env python3
"""
Test gradient computation using finite differences.
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


def compute_log_likelihood(delta, tau, lambda_param, E, log_Pi, 
                          species_helpers, ccp_helpers, clade_species_map, root_clade_id):
    """Compute log-likelihood for given parameters."""
    # Event probabilities
    rates_sum = 1.0 + delta + tau + lambda_param
    p_S = 1.0 / rates_sum
    p_D = delta / rates_sum
    p_T = tau / rates_sum
    p_L = lambda_param / rates_sum
    
    # One E update
    E_new, _, _, Ebar = E_step(
        E, species_helpers["s_C1"], species_helpers["s_C2"],
        species_helpers["Recipients_mat"], p_S, p_D, p_T, p_L
    )
    
    # One Pi update
    log_Pi_new = Pi_update_ccp_log(
        log_Pi, ccp_helpers, species_helpers,
        clade_species_map, E_new, Ebar, p_S, p_D, p_T
    )
    
    # Log-likelihood
    root_log_pi = log_Pi_new[root_clade_id, :]
    return torch.logsumexp(root_log_pi, dim=0).item()


def test_finite_differences():
    """Test gradients using finite differences."""
    print("Testing gradients with finite differences...")
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    species_path = "test_trees_3/sp.nwk"
    gene_path = "test_trees_3/g.nwk"
    
    # Build structures
    ccp = build_ccp_from_single_tree(gene_path)
    species_helpers = build_species_helpers(species_path, device, torch.float64)
    clade_species_map = build_clade_species_mapping(ccp, species_helpers, device, torch.float64)
    ccp_helpers = build_ccp_helpers(ccp, device, torch.float64)
    root_clade_id = get_root_clade_id(ccp)
    
    # Initialize parameters
    delta = 0.055
    tau = 0.001
    lambda_param = 0.001
    
    # Event probabilities
    rates_sum = 1.0 + delta + tau + lambda_param
    p_S = 1.0 / rates_sum
    p_D = delta / rates_sum
    p_T = tau / rates_sum
    p_L = lambda_param / rates_sum
    
    # Initialize and converge E
    E = torch.zeros(species_helpers["S"], dtype=torch.float64, device=device)
    for _ in range(20):
        E, _, _, Ebar = E_step(
            E, species_helpers["s_C1"], species_helpers["s_C2"],
            species_helpers["Recipients_mat"], p_S, p_D, p_T, p_L
        )
    
    # Initialize and converge log_Pi
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
    for _ in range(20):
        log_Pi = Pi_update_ccp_log(
            log_Pi, ccp_helpers, species_helpers,
            clade_species_map, E, Ebar, p_S, p_D, p_T
        )
    
    # Compute base log-likelihood
    ll_base = compute_log_likelihood(delta, tau, lambda_param, E.clone(), log_Pi.clone(),
                                    species_helpers, ccp_helpers, clade_species_map, root_clade_id)
    print(f"\nBase log-likelihood: {ll_base:.6f}")
    
    # Finite differences
    eps = 1e-6
    
    # Delta gradient
    ll_plus = compute_log_likelihood(delta + eps, tau, lambda_param, E.clone(), log_Pi.clone(),
                                    species_helpers, ccp_helpers, clade_species_map, root_clade_id)
    ll_minus = compute_log_likelihood(delta - eps, tau, lambda_param, E.clone(), log_Pi.clone(),
                                     species_helpers, ccp_helpers, clade_species_map, root_clade_id)
    grad_delta = (ll_plus - ll_minus) / (2 * eps)
    print(f"\nFinite difference gradients:")
    print(f"  d(log L)/d(delta) = {grad_delta:.6f}")
    
    # Tau gradient
    ll_plus = compute_log_likelihood(delta, tau + eps, lambda_param, E.clone(), log_Pi.clone(),
                                    species_helpers, ccp_helpers, clade_species_map, root_clade_id)
    ll_minus = compute_log_likelihood(delta, tau - eps, lambda_param, E.clone(), log_Pi.clone(),
                                     species_helpers, ccp_helpers, clade_species_map, root_clade_id)
    grad_tau = (ll_plus - ll_minus) / (2 * eps)
    print(f"  d(log L)/d(tau) = {grad_tau:.6f}")
    
    # Lambda gradient
    ll_plus = compute_log_likelihood(delta, tau, lambda_param + eps, E.clone(), log_Pi.clone(),
                                    species_helpers, ccp_helpers, clade_species_map, root_clade_id)
    ll_minus = compute_log_likelihood(delta, tau, lambda_param - eps, E.clone(), log_Pi.clone(),
                                     species_helpers, ccp_helpers, clade_species_map, root_clade_id)
    grad_lambda = (ll_plus - ll_minus) / (2 * eps)
    print(f"  d(log L)/d(lambda) = {grad_lambda:.6f}")
    
    # Check gradient direction
    print(f"\nGradient analysis:")
    print(f"  Delta gradient {'positive' if grad_delta > 0 else 'negative'} (current δ={delta})")
    print(f"  Tau gradient {'positive' if grad_tau > 0 else 'negative'} (current τ={tau})")
    print(f"  Lambda gradient {'positive' if grad_lambda > 0 else 'negative'} (current λ={lambda_param})")


if __name__ == '__main__':
    test_finite_differences()