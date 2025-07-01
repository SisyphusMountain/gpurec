#!/usr/bin/env python3
"""
Proper log-space single-iteration gradient optimization for CCP reconciliation.
This version correctly maintains log-space computation throughout.
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

# Import the ACTUAL log-space Pi update
from matmul_ale_ccp_log import Pi_update_ccp_log


def compute_probabilities(delta, tau, lambda_param):
    """Compute event probabilities from rates."""
    rates_sum = 1.0 + delta + tau + lambda_param
    p_S = 1.0 / rates_sum
    p_D = delta / rates_sum
    p_T = tau / rates_sum
    p_L = lambda_param / rates_sum
    return p_S, p_D, p_T, p_L


class LogSpaceCCPOptimizer(nn.Module):
    """
    Proper log-space optimizer that maintains log probabilities throughout.
    """
    
    def __init__(self, species_helpers, ccp_helpers, clade_species_map, root_clade_id):
        super().__init__()
        
        self.species_helpers = species_helpers
        self.ccp_helpers = ccp_helpers
        self.clade_species_map = clade_species_map
        self.root_clade_id = root_clade_id
        
        # Extract dimensions
        self.S = species_helpers["S"]
        self.C = ccp_helpers["C"]
        
        # Get device from species helpers
        device = species_helpers["s_C1"].device
        
        # Initialize E (in linear space as it's used that way)
        self.E = torch.zeros(self.S, dtype=torch.float64, device=device)
        
        # Initialize log_Pi properly in log space
        self.log_Pi = torch.full((self.C, self.S), float('-inf'), dtype=torch.float64, device=device)
        
        # Initialize leaf probabilities in log space
        self._initialize_leaf_log_probabilities()
        
        # Store helpers
        self.s_C1 = species_helpers["s_C1"]
        self.s_C2 = species_helpers["s_C2"]
        self.Recipients_mat = species_helpers["Recipients_mat"]
        
        # Use softplus parameterization for rates
        # softplus(x) = log(1 + exp(x)) ensures positive values
        self.raw_delta = nn.Parameter(torch.tensor(0.0, dtype=torch.float64, device=device))
        self.raw_tau = nn.Parameter(torch.tensor(0.0, dtype=torch.float64, device=device))
        self.raw_lambda = nn.Parameter(torch.tensor(0.0, dtype=torch.float64, device=device))
    
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
        delta = torch.nn.functional.softplus(self.raw_delta)
        tau = torch.nn.functional.softplus(self.raw_tau)
        lambda_param = torch.nn.functional.softplus(self.raw_lambda)
        return delta, tau, lambda_param
    
    def forward(self):
        """
        Perform single iteration updates and compute log-likelihood.
        Maintains log-space computation throughout.
        """
        # Get current rates
        delta, tau, lambda_param = self.get_rates()
        
        # Compute event probabilities
        p_S, p_D, p_T, p_L = compute_probabilities(delta, tau, lambda_param)
        
        # Ensure probabilities are not exactly zero to avoid log(0) issues
        eps = 1e-10
        p_S = torch.clamp(p_S, min=eps)
        p_D = torch.clamp(p_D, min=eps)
        p_T = torch.clamp(p_T, min=eps)
        p_L = torch.clamp(p_L, min=eps)
        
        # Single E update - use detached E as input but keep gradients for the update
        E_old = self.E.detach()
        E_new, E_s1, E_s2, Ebar = E_step(
            E_old, self.s_C1, self.s_C2, self.Recipients_mat,
            p_S, p_D, p_T, p_L
        )
        self.E = E_new
        
        # Single Pi update IN LOG SPACE - use detached log_Pi as input but keep gradients
        log_Pi_old = self.log_Pi.detach()
        log_Pi_new = Pi_update_ccp_log(
            log_Pi_old, self.ccp_helpers, self.species_helpers,
            self.clade_species_map, E_new, Ebar, p_S, p_D, p_T
        )
        self.log_Pi = log_Pi_new
        
        # Compute log-likelihood directly in log space
        root_log_pi = log_Pi_new[self.root_clade_id, :]
        log_likelihood = torch.logsumexp(root_log_pi, dim=0)
        
        return log_likelihood


def optimize_log_space(species_tree_path, gene_tree_path, 
                      init_delta=0.1, init_tau=0.001, init_lambda=0.001,
                      lr=0.001, epochs=1000, 
                      warm_start_e_iters=20, warm_start_pi_iters=20,
                      device=None):
    """
    Optimize CCP reconciliation parameters using proper log-space computation.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 80)
    print("Log-Space Single-Iteration Gradient Optimization")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Initial rates: δ={init_delta}, τ={init_tau}, λ={init_lambda}")
    print(f"Learning rate: {lr}, Epochs: {epochs}")
    print()
    
    # Build structures
    ccp = build_ccp_from_single_tree(gene_tree_path)
    species_helpers = build_species_helpers(species_tree_path, device, torch.float64)
    clade_species_map = build_clade_species_mapping(ccp, species_helpers, device, torch.float64)
    ccp_helpers = build_ccp_helpers(ccp, device, torch.float64)
    root_clade_id = get_root_clade_id(ccp)
    
    print(f"  {ccp_helpers['C']} clades, {species_helpers['S']} species")
    
    # Create optimizer module
    model = LogSpaceCCPOptimizer(
        species_helpers, ccp_helpers, clade_species_map, root_clade_id
    ).to(device)
    
    # Initialize parameters using inverse softplus
    # If we want rate = init_value, then raw = log(exp(init_value) - 1)
    # Handle small values carefully to avoid numerical issues
    with torch.no_grad():
        # For small x, softplus(x) ≈ log(2) + x/2
        # So inverse_softplus(y) ≈ 2*(y - log(2)) for small y
        # Use exact inverse softplus: if softplus(x) = y, then x = log(exp(y) - 1)
        def inverse_softplus(y):
            # For numerical stability with small y
            if y < 1e-5:
                # For very small y, softplus(x) ≈ exp(x), so x ≈ log(y)
                return torch.log(torch.tensor(y, dtype=torch.float64, device=device))
            else:
                return torch.log(torch.expm1(torch.tensor(y, dtype=torch.float64, device=device)))
        
        model.raw_delta.data = inverse_softplus(init_delta)
        model.raw_tau.data = inverse_softplus(init_tau) 
        model.raw_lambda.data = inverse_softplus(init_lambda)
    
    # Warm start - run to convergence
    print("\nWarm start phase - converging E and Pi...")
    with torch.no_grad():
        delta, tau, lambda_param = model.get_rates()
        p_S, p_D, p_T, p_L = compute_probabilities(delta, tau, lambda_param)
        
        # Converge E
        print("  Converging E...")
        for i in range(100):  # More iterations
            E_old = model.E.clone()
            E_new, _, _, Ebar = E_step(
                model.E, model.s_C1, model.s_C2, model.Recipients_mat,
                p_S, p_D, p_T, p_L
            )
            model.E = E_new
            
            # Check convergence
            if i > 0 and torch.abs(E_new - E_old).max() < 1e-10:
                print(f"    E converged after {i+1} iterations")
                break
        
        # Converge log_Pi
        print("  Converging Pi...")
        for i in range(100):  # More iterations
            log_Pi_old = model.log_Pi.clone()
            model.log_Pi = Pi_update_ccp_log(
                model.log_Pi, model.ccp_helpers, model.species_helpers,
                model.clade_species_map, model.E, Ebar, p_S, p_D, p_T
            )
            
            # Check convergence in log space
            # Only check finite values
            finite_mask = torch.isfinite(model.log_Pi) & torch.isfinite(log_Pi_old)
            if finite_mask.any():
                max_diff = torch.abs(model.log_Pi[finite_mask] - log_Pi_old[finite_mask]).max()
                if i > 0 and max_diff < 1e-10:
                    print(f"    Pi converged after {i+1} iterations")
                    break
    
    # Get initial log-likelihood
    with torch.no_grad():
        initial_ll = model().item()
        # Check parameter values
        delta, tau, lambda_param = model.get_rates()
        print(f"After warm start - Rates: δ={delta:.6f}, τ={tau:.6f}, λ={lambda_param:.6f}")
    print(f"Initial log-likelihood: {initial_ll:.6f}")
    
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
    best_params = {
        'delta': init_delta,
        'tau': init_tau,
        'lambda': init_lambda,
        'log_likelihood': initial_ll
    }
    
    print("\nStarting optimization...")
    print(f"{'Epoch':>6} {'δ':>10} {'τ':>10} {'λ':>10} {'Log-L':>12}")
    print("-" * 60)
    
    for epoch in range(epochs):
        # Forward pass
        log_likelihood = model()
        
        # Compute loss (negative log-likelihood for minimization)
        loss = -log_likelihood
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Debug gradient computation
        if epoch == 0:
            print(f"\nDEBUG gradient computation:")
            print(f"  Loss: {loss.item():.6f}")
            print(f"  Log-likelihood: {log_likelihood.item():.6f}")
            for name, param in model.named_parameters():
                if param.grad is not None:
                    print(f"  {name}: value={param.item():.6f}, grad={param.grad.item():.6f}")
                else:
                    print(f"  {name}: value={param.item():.6f}, grad=None")
            
            # Check for inf/nan in intermediate values
            print(f"  E has NaN: {torch.isnan(model.E).any()}")
            print(f"  log_Pi has NaN: {torch.isnan(model.log_Pi).any()}")
            print(f"  log_Pi has -inf: {torch.isinf(model.log_Pi).sum()}/{model.log_Pi.numel()}")
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update parameters
        optimizer.step()
        
        # Get current values
        with torch.no_grad():
            delta, tau, lambda_param = model.get_rates()
            ll_value = log_likelihood.item()
            
            # Debug NaN issues
            if epoch < 5 and (torch.isnan(log_likelihood) or torch.isinf(log_likelihood) or torch.isnan(delta)):
                print(f"\nDEBUG at epoch {epoch}:")
                print(f"  Raw params: δ={model.raw_delta.item():.6f}, τ={model.raw_tau.item():.6f}, λ={model.raw_lambda.item():.6f}")
                print(f"  Rates: δ={delta:.6f}, τ={tau:.6f}, λ={lambda_param:.6f}")
                print(f"  E range: [{model.E.min():.6f}, {model.E.max():.6f}]")
                print(f"  log_Pi finite values: {torch.isfinite(model.log_Pi).sum()}/{model.log_Pi.numel()}")
                root_log_pi = model.log_Pi[model.root_clade_id, :]
                finite_mask = torch.isfinite(root_log_pi)
                if finite_mask.any():
                    print(f"  Root log_pi range: [{root_log_pi[finite_mask].min():.6f}, {root_log_pi[finite_mask].max():.6f}]")
                else:
                    print(f"  Root log_pi: all non-finite")
                if model.raw_delta.grad is not None:
                    print(f"  Gradients: delta={model.raw_delta.grad.item():.6f}, tau={model.raw_tau.grad.item():.6f}, lambda={model.raw_lambda.grad.item():.6f}")
                else:
                    print(f"  No gradients computed yet")
        
        # Store history
        history['epoch'].append(epoch)
        history['delta'].append(delta.item() if not torch.isnan(delta) else float('nan'))
        history['tau'].append(tau.item() if not torch.isnan(tau) else float('nan'))
        history['lambda'].append(lambda_param.item() if not torch.isnan(lambda_param) else float('nan'))
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
    print("\nOptimization completed!")
    print(f"\nBest parameters found:")
    print(f"  δ = {best_params['delta']:.6f}")
    print(f"  τ = {best_params['tau']:.6f}")
    print(f"  λ = {best_params['lambda']:.6f}")
    print(f"  Log-likelihood = {best_params['log_likelihood']:.6f}")
    
    # Save results
    results = {
        'best_params': best_params,
        'history': history,
        'config': {
            'species_tree': species_tree_path,
            'gene_tree': gene_tree_path,
            'init_delta': init_delta,
            'init_tau': init_tau,
            'init_lambda': init_lambda,
            'lr': lr,
            'epochs': epochs
        }
    }
    
    with open('log_space_optimization_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to log_space_optimization_results.json")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Proper log-space single-iteration gradient optimization'
    )
    parser.add_argument('--species', required=True, help='Species tree file')
    parser.add_argument('--gene', required=True, help='Gene tree file')
    parser.add_argument('--init-delta', type=float, default=0.1, help='Initial duplication rate')
    parser.add_argument('--init-tau', type=float, default=0.001, help='Initial transfer rate')
    parser.add_argument('--init-lambda', type=float, default=0.001, help='Initial loss rate')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs')
    
    args = parser.parse_args()
    
    optimize_log_space(
        args.species, args.gene,
        init_delta=args.init_delta,
        init_tau=args.init_tau,
        init_lambda=args.init_lambda,
        lr=args.lr,
        epochs=args.epochs
    )


if __name__ == '__main__':
    main()