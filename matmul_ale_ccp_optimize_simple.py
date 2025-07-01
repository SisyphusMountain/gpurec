#!/usr/bin/env python3
"""
Simplified single-iteration gradient optimization for CCP reconciliation.
This version focuses on numerical stability and proper gradient flow.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import json
import numpy as np
from time import time

# Import CCP functions
from matmul_ale_ccp import (
    build_ccp_from_single_tree,
    build_species_helpers,
    build_clade_species_mapping,
    build_ccp_helpers,
    get_root_clade_id,
    E_step
)


def compute_probabilities(delta, tau, lambda_param):
    """Compute event probabilities from rates."""
    rates_sum = 1.0 + delta + tau + lambda_param
    p_S = 1.0 / rates_sum
    p_D = delta / rates_sum
    p_T = tau / rates_sum
    p_L = lambda_param / rates_sum
    return p_S, p_D, p_T, p_L


def optimize_simple(species_tree_path, gene_tree_path, 
                   init_delta=0.1, init_tau=0.001, init_lambda=0.001,
                   lr=0.001, epochs=500, device=None):
    """
    Simple optimization with better numerical stability.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 80)
    print("Simple Single-Iteration Gradient Optimization")
    print("=" * 80)
    
    # Build structures
    ccp = build_ccp_from_single_tree(gene_tree_path)
    species_helpers = build_species_helpers(species_tree_path, device, torch.float64)
    clade_species_map = build_clade_species_mapping(ccp, species_helpers, device, torch.float64)
    ccp_helpers = build_ccp_helpers(ccp, device, torch.float64)
    root_clade_id = get_root_clade_id(ccp)
    
    C = ccp_helpers["C"]
    S = species_helpers["S"]
    
    print(f"  {C} clades, {S} species")
    
    # Initialize parameters with softplus transformation
    # softplus(x) = log(1 + exp(x)) ensures positive values
    log_delta = nn.Parameter(torch.tensor(np.log(np.exp(init_delta) - 1), 
                                         dtype=torch.float64, device=device))
    log_tau = nn.Parameter(torch.tensor(np.log(np.exp(init_tau) - 1), 
                                       dtype=torch.float64, device=device))
    log_lambda = nn.Parameter(torch.tensor(np.log(np.exp(init_lambda) - 1), 
                                          dtype=torch.float64, device=device))
    
    # Initialize E and Pi
    E = torch.zeros(S, dtype=torch.float64, device=device)
    log_Pi = torch.full((C, S), -10.0, dtype=torch.float64, device=device)  # Start with small uniform values
    
    # Initialize leaf probabilities
    for c in range(C):
        clade = ccp.id_to_clade[c]
        if clade.is_leaf():
            mapped_species = torch.nonzero(clade_species_map[c] > 0, as_tuple=False).flatten()
            if len(mapped_species) > 0:
                log_Pi[c, :] = -1000.0  # Very small for non-mapped
                log_Pi[c, mapped_species] = -np.log(len(mapped_species))  # Uniform among mapped
    
    # Warm start
    print("\nWarm start...")
    with torch.no_grad():
        delta = torch.nn.functional.softplus(log_delta)
        tau = torch.nn.functional.softplus(log_tau)
        lambda_param = torch.nn.functional.softplus(log_lambda)
        p_S, p_D, p_T, p_L = compute_probabilities(delta, tau, lambda_param)
        
        # Warm start E
        for _ in range(20):
            E_new, _, _, Ebar = E_step(E, species_helpers["s_C1"], species_helpers["s_C2"],
                                       species_helpers["Recipients_mat"], p_S, p_D, p_T, p_L)
            E = E_new.clamp(min=0, max=1)
        
        # Warm start Pi
        for _ in range(20):
            log_Pi = Pi_update_ccp_log_stable(log_Pi, ccp_helpers, species_helpers,
                                             clade_species_map, E, Ebar, p_S, p_D, p_T)
    
    # Compute initial log-likelihood
    initial_ll = torch.logsumexp(log_Pi[root_clade_id, :], dim=0).item()
    print(f"Initial log-likelihood: {initial_ll:.6f}")
    
    # Create optimizer
    optimizer = optim.Adam([log_delta, log_tau, log_lambda], lr=lr)
    
    # Training loop
    print("\nOptimization...")
    print(f"{'Epoch':>6} {'δ':>10} {'τ':>10} {'λ':>10} {'Log-L':>12}")
    print("-" * 60)
    
    best_ll = initial_ll
    best_params = None
    
    for epoch in range(epochs):
        # Get rates with softplus
        delta = torch.nn.functional.softplus(log_delta)
        tau = torch.nn.functional.softplus(log_tau)
        lambda_param = torch.nn.functional.softplus(log_lambda)
        
        # Clamp to reasonable bounds
        delta = delta.clamp(min=1e-10, max=10.0)
        tau = tau.clamp(min=1e-10, max=10.0)
        lambda_param = lambda_param.clamp(min=1e-10, max=10.0)
        
        # Compute probabilities
        p_S, p_D, p_T, p_L = compute_probabilities(delta, tau, lambda_param)
        
        # Single E update
        E_new, _, _, Ebar = E_step(E.detach(), species_helpers["s_C1"], species_helpers["s_C2"],
                                   species_helpers["Recipients_mat"], p_S, p_D, p_T, p_L)
        E = E_new.clamp(min=0, max=1)
        
        # Single Pi update
        log_Pi_new = Pi_update_ccp_log_stable(log_Pi.detach(), ccp_helpers, species_helpers,
                                              clade_species_map, E, Ebar, p_S, p_D, p_T)
        log_Pi = log_Pi_new
        
        # Compute log-likelihood
        log_likelihood = torch.logsumexp(log_Pi[root_clade_id, :], dim=0)
        
        # Check for numerical issues
        if torch.isnan(log_likelihood) or torch.isinf(log_likelihood):
            print(f"Numerical issue at epoch {epoch}")
            break
        
        # Compute loss (negative log-likelihood)
        loss = -log_likelihood
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_([log_delta, log_tau, log_lambda], max_norm=1.0)
        
        # Update
        optimizer.step()
        
        # Track best
        ll_value = log_likelihood.item()
        if ll_value > best_ll:
            best_ll = ll_value
            best_params = {
                'delta': delta.item(),
                'tau': tau.item(),
                'lambda': lambda_param.item(),
                'log_likelihood': ll_value
            }
        
        # Print progress
        if epoch % 50 == 0:
            print(f"{epoch:6d} {delta.item():10.6f} {tau.item():10.6f} "
                  f"{lambda_param.item():10.6f} {ll_value:12.6f}")
    
    print("-" * 60)
    
    if best_params:
        print("\nBest parameters:")
        print(f"  δ = {best_params['delta']:.6f}")
        print(f"  τ = {best_params['tau']:.6f}")
        print(f"  λ = {best_params['lambda']:.6f}")
        print(f"  Log-likelihood = {best_params['log_likelihood']:.6f}")
        
        # Compare with expected values for test_trees_3
        print("\nExpected parameters (from test_trees_3):")
        print(f"  δ = 0.055554")
        print(f"  τ = 0.000000")
        print(f"  λ = 0.000000")
    
    return best_params


def main():
    parser = argparse.ArgumentParser(description='Simple single-iteration optimization')
    parser.add_argument('--species', required=True, help='Species tree file')
    parser.add_argument('--gene', required=True, help='Gene tree file')
    parser.add_argument('--init-delta', type=float, default=0.1)
    parser.add_argument('--init-tau', type=float, default=0.001)
    parser.add_argument('--init-lambda', type=float, default=0.001)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=500)
    
    args = parser.parse_args()
    
    optimize_simple(
        args.species, args.gene,
        init_delta=args.init_delta,
        init_tau=args.init_tau,
        init_lambda=args.init_lambda,
        lr=args.lr,
        epochs=args.epochs
    )


if __name__ == '__main__':
    main()