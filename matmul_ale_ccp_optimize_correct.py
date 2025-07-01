#!/usr/bin/env python3
"""
CORRECT single-iteration gradient optimization for CCP reconciliation.
This version properly uses Pi_update_ccp_log without converting to linear space.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import json
import numpy as np

# Import CCP functions
from matmul_ale_ccp import (
    build_ccp_from_single_tree,
    build_species_helpers,
    build_clade_species_mapping,
    build_ccp_helpers,
    get_root_clade_id,
    E_step
)

# Import the ACTUAL log-space Pi update - DO NOT CONVERT TO LINEAR SPACE!
from matmul_ale_ccp_log import Pi_update_ccp_log


def compute_probabilities(delta, tau, lambda_param):
    """Compute event probabilities from rates."""
    rates_sum = 1.0 + delta + tau + lambda_param
    p_S = 1.0 / rates_sum
    p_D = delta / rates_sum
    p_T = tau / rates_sum
    p_L = lambda_param / rates_sum
    return p_S, p_D, p_T, p_L


class SingleIterCCPOptimizer(nn.Module):
    """
    Single-iteration optimizer that properly maintains log-space computation.
    """
    
    def __init__(self, species_helpers, ccp_helpers, clade_species_map, root_clade_id):
        super().__init__()
        
        self.species_helpers = species_helpers
        self.ccp_helpers = ccp_helpers
        self.clade_species_map = clade_species_map
        self.root_clade_id = root_clade_id
        
        # Get dimensions and device
        self.S = species_helpers["S"]
        self.C = ccp_helpers["C"]
        device = species_helpers["s_C1"].device
        
        # Initialize E (linear space) and log_Pi (log space)
        self.E = torch.zeros(self.S, dtype=torch.float64, device=device)
        self.log_Pi = torch.full((self.C, self.S), float('-inf'), dtype=torch.float64, device=device)
        
        # Initialize leaf probabilities in LOG SPACE
        self._initialize_leaf_log_probabilities()
        
        # Store helpers
        self.s_C1 = species_helpers["s_C1"]
        self.s_C2 = species_helpers["s_C2"]
        self.Recipients_mat = species_helpers["Recipients_mat"]
        
        # Parameters with softplus transformation
        self.log_delta = nn.Parameter(torch.tensor(0.0, dtype=torch.float64, device=device))
        self.log_tau = nn.Parameter(torch.tensor(0.0, dtype=torch.float64, device=device))
        self.log_lambda = nn.Parameter(torch.tensor(0.0, dtype=torch.float64, device=device))
    
    def _initialize_leaf_log_probabilities(self):
        """Initialize log probabilities for leaf clades."""
        ccp = self.ccp_helpers['ccp']
        
        for c in range(self.C):
            clade = ccp.id_to_clade[c]
            if clade.is_leaf():
                mapped_species = torch.nonzero(self.clade_species_map[c] > 0, as_tuple=False).flatten()
                if len(mapped_species) > 0:
                    # Uniform log probability among mapped species
                    log_prob = -torch.log(torch.tensor(len(mapped_species), dtype=torch.float64))
                    self.log_Pi[c, mapped_species] = log_prob
    
    def get_rates(self):
        """Get positive rates using softplus transformation."""
        delta = torch.nn.functional.softplus(self.log_delta)
        tau = torch.nn.functional.softplus(self.log_tau)
        lambda_param = torch.nn.functional.softplus(self.log_lambda)
        return delta, tau, lambda_param
    
    def forward(self):
        """
        Single iteration forward pass maintaining proper log-space computation.
        """
        # Get rates and probabilities
        delta, tau, lambda_param = self.get_rates()
        p_S, p_D, p_T, p_L = compute_probabilities(delta, tau, lambda_param)
        
        # Single E update - detach input but keep gradients through update
        E_old = self.E.detach()
        E_new, E_s1, E_s2, Ebar = E_step(
            E_old, self.s_C1, self.s_C2, self.Recipients_mat,
            p_S, p_D, p_T, p_L
        )
        self.E = E_new
        
        # Single Pi update in LOG SPACE - DO NOT CONVERT!
        log_Pi_old = self.log_Pi.detach()
        log_Pi_new = Pi_update_ccp_log(
            log_Pi_old, self.ccp_helpers, self.species_helpers,
            self.clade_species_map, E_new, Ebar, p_S, p_D, p_T
        )
        self.log_Pi = log_Pi_new
        
        # Compute log-likelihood in log space
        root_log_pi = log_Pi_new[self.root_clade_id, :]
        log_likelihood = torch.logsumexp(root_log_pi, dim=0)
        
        return log_likelihood


def optimize_correct(species_tree_path, gene_tree_path,
                    init_delta=0.1, init_tau=0.001, init_lambda=0.001,
                    lr=0.001, epochs=1000, device=None):
    """
    Optimize using correct log-space computation throughout.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 80)
    print("CORRECT Single-Iteration Gradient Optimization")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Initial rates: δ={init_delta}, τ={init_tau}, λ={init_lambda}")
    print()
    
    # Build CCP structures
    print("Building CCP structures...")
    ccp = build_ccp_from_single_tree(gene_tree_path)
    species_helpers = build_species_helpers(species_tree_path, device, torch.float64)
    clade_species_map = build_clade_species_mapping(ccp, species_helpers, device, torch.float64)
    ccp_helpers = build_ccp_helpers(ccp, device, torch.float64)
    root_clade_id = get_root_clade_id(ccp)
    
    print(f"  {ccp_helpers['C']} clades, {species_helpers['S']} species")
    
    # Create optimizer module
    model = SingleIterCCPOptimizer(
        species_helpers, ccp_helpers, clade_species_map, root_clade_id
    ).to(device)
    
    # Initialize parameters
    with torch.no_grad():
        # Inverse softplus: x = log(exp(y) - 1)
        model.log_delta.data = torch.log(torch.expm1(torch.tensor(init_delta, device=device)))
        model.log_tau.data = torch.log(torch.expm1(torch.tensor(init_tau, device=device)))
        model.log_lambda.data = torch.log(torch.expm1(torch.tensor(init_lambda, device=device)))
    
    # CRITICAL: Converge E and Pi before optimization
    print("\nWarm start - converging E and Pi...")
    with torch.no_grad():
        delta, tau, lambda_param = model.get_rates()
        p_S, p_D, p_T, p_L = compute_probabilities(delta, tau, lambda_param)
        
        # Converge E
        print("  Converging E...")
        for i in range(100):
            E_old = model.E.clone()
            E_new, _, _, Ebar = E_step(
                model.E, model.s_C1, model.s_C2, model.Recipients_mat,
                p_S, p_D, p_T, p_L
            )
            model.E = E_new
            if torch.abs(E_new - E_old).max() < 1e-10:
                print(f"    E converged after {i+1} iterations")
                break
        
        # Converge log_Pi
        print("  Converging log_Pi...")
        for i in range(100):
            log_Pi_old = model.log_Pi.clone()
            model.log_Pi = Pi_update_ccp_log(
                model.log_Pi, model.ccp_helpers, model.species_helpers,
                model.clade_species_map, model.E, Ebar, p_S, p_D, p_T
            )
            # Check convergence on finite values
            finite_mask = torch.isfinite(model.log_Pi) & torch.isfinite(log_Pi_old)
            if finite_mask.any():
                max_diff = torch.abs(model.log_Pi[finite_mask] - log_Pi_old[finite_mask]).max()
                if max_diff < 1e-10:
                    print(f"    log_Pi converged after {i+1} iterations")
                    break
    
    # Get initial log-likelihood
    with torch.no_grad():
        initial_ll = model().item()
    print(f"\nInitial log-likelihood: {initial_ll:.6f}")
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training history
    history = {
        'epoch': [],
        'delta': [],
        'tau': [],
        'lambda': [],
        'log_likelihood': []
    }
    
    best_ll = initial_ll
    best_params = None
    
    print("\nStarting optimization...")
    print(f"{'Epoch':>6} {'δ':>10} {'τ':>10} {'λ':>10} {'Log-L':>12}")
    print("-" * 60)
    
    for epoch in range(epochs):
        # Forward pass
        log_likelihood = model()
        
        # Loss is negative log-likelihood
        loss = -log_likelihood
        
        # Backward pass
        optimizer.zero_grad()
        
        # Add debugging for first few epochs
        if epoch < 3:
            print(f"\nEpoch {epoch} debug:")
            print(f"  Loss: {loss.item():.6f}")
            print(f"  Rates before backward: δ={delta.item():.6f}, τ={tau.item():.6f}, λ={lambda_param.item():.6f}")
            
        loss.backward()
        
        if epoch < 3:
            for name, param in model.named_parameters():
                if param.grad is not None:
                    print(f"  {name}: grad={param.grad.item():.6e}, has_nan={torch.isnan(param.grad).any()}")
                else:
                    print(f"  {name}: grad=None")
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update
        optimizer.step()
        
        # Get current values
        with torch.no_grad():
            delta, tau, lambda_param = model.get_rates()
            ll_value = log_likelihood.item()
        
        # Check for numerical issues
        if torch.isnan(log_likelihood) or torch.isinf(log_likelihood):
            print(f"\nNumerical issue at epoch {epoch}")
            break
        
        # Store history
        history['epoch'].append(epoch)
        history['delta'].append(delta.item())
        history['tau'].append(tau.item())
        history['lambda'].append(lambda_param.item())
        history['log_likelihood'].append(ll_value)
        
        # Track best
        if ll_value > best_ll:
            best_ll = ll_value
            best_params = {
                'delta': delta.item(),
                'tau': tau.item(),
                'lambda': lambda_param.item(),
                'log_likelihood': ll_value
            }
        
        # Print progress
        if epoch % 50 == 0 or epoch == epochs - 1:
            print(f"{epoch:6d} {delta.item():10.6f} {tau.item():10.6f} "
                  f"{lambda_param.item():10.6f} {ll_value:12.6f}")
    
    print("-" * 60)
    
    if best_params:
        print("\nBest parameters found:")
        print(f"  δ = {best_params['delta']:.6f}")
        print(f"  τ = {best_params['tau']:.6f}")
        print(f"  λ = {best_params['lambda']:.6f}")
        print(f"  Log-likelihood = {best_params['log_likelihood']:.6f}")
    
    # Compare with expected values
    print("\nExpected parameters (test_trees_3):")
    print(f"  δ = 0.055554")
    print(f"  τ = 0.000000")
    print(f"  λ = 0.000000")
    
    return history, best_params


def main():
    parser = argparse.ArgumentParser(
        description='CORRECT single-iteration gradient optimization'
    )
    parser.add_argument('--species', required=True, help='Species tree file')
    parser.add_argument('--gene', required=True, help='Gene tree file')
    parser.add_argument('--init-delta', type=float, default=0.1)
    parser.add_argument('--init-tau', type=float, default=0.001)
    parser.add_argument('--init-lambda', type=float, default=0.001)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=1000)
    
    args = parser.parse_args()
    
    optimize_correct(
        args.species, args.gene,
        init_delta=args.init_delta,
        init_tau=args.init_tau,
        init_lambda=args.init_lambda,
        lr=args.lr,
        epochs=args.epochs
    )


if __name__ == '__main__':
    main()