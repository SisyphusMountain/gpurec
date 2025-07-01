#!/usr/bin/env python3
"""
Single-iteration optimization using finite difference gradients.
Since automatic differentiation through Pi_update_ccp_log has issues,
we'll use finite differences to compute gradients.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import argparse
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


def compute_probabilities(delta, tau, lambda_param):
    """Compute event probabilities from rates."""
    rates_sum = 1.0 + delta + tau + lambda_param
    p_S = 1.0 / rates_sum
    p_D = delta / rates_sum
    p_T = tau / rates_sum
    p_L = lambda_param / rates_sum
    return p_S, p_D, p_T, p_L


class FiniteDiffCCPOptimizer:
    """Optimizer using finite differences for gradients."""
    
    def __init__(self, species_helpers, ccp_helpers, clade_species_map, root_clade_id):
        self.species_helpers = species_helpers
        self.ccp_helpers = ccp_helpers
        self.clade_species_map = clade_species_map
        self.root_clade_id = root_clade_id
        
        # Get device
        self.device = species_helpers["s_C1"].device
        
        # Initialize E and log_Pi
        self.E = torch.zeros(species_helpers["S"], dtype=torch.float64, device=self.device)
        self.log_Pi = torch.full((ccp_helpers["C"], species_helpers["S"]), 
                                float('-inf'), dtype=torch.float64, device=self.device)
        
        # Initialize leaf probabilities
        self._initialize_leaf_log_probabilities()
        
        # Store helpers
        self.s_C1 = species_helpers["s_C1"]
        self.s_C2 = species_helpers["s_C2"]
        self.Recipients_mat = species_helpers["Recipients_mat"]
    
    def _initialize_leaf_log_probabilities(self):
        """Initialize log probabilities for leaf clades."""
        ccp = self.ccp_helpers['ccp']
        
        for c in range(self.ccp_helpers["C"]):
            clade = ccp.id_to_clade[c]
            if clade.is_leaf():
                mapped_species = torch.nonzero(self.clade_species_map[c] > 0, as_tuple=False).flatten()
                if len(mapped_species) > 0:
                    log_prob = -np.log(len(mapped_species))
                    self.log_Pi[c, mapped_species] = log_prob
    
    def compute_log_likelihood(self, delta, tau, lambda_param):
        """Compute log-likelihood after single iteration."""
        # Compute probabilities
        p_S, p_D, p_T, p_L = compute_probabilities(delta, tau, lambda_param)
        
        # Single E update
        E_new, _, _, Ebar = E_step(
            self.E.detach(), self.s_C1, self.s_C2, self.Recipients_mat,
            p_S, p_D, p_T, p_L
        )
        self.E = E_new
        
        # Single Pi update
        log_Pi_new = Pi_update_ccp_log(
            self.log_Pi.detach(), self.ccp_helpers, self.species_helpers,
            self.clade_species_map, E_new, Ebar, p_S, p_D, p_T
        )
        self.log_Pi = log_Pi_new
        
        # Compute log-likelihood
        root_log_pi = log_Pi_new[self.root_clade_id, :]
        return torch.logsumexp(root_log_pi, dim=0).item()
    
    def compute_finite_diff_gradients(self, delta, tau, lambda_param, eps=1e-6):
        """Compute gradients using centered finite differences."""
        # Save current state
        E_saved = self.E.clone()
        log_Pi_saved = self.log_Pi.clone()
        
        # Delta gradient (centered difference)
        self.E = E_saved.clone()
        self.log_Pi = log_Pi_saved.clone()
        ll_plus = self.compute_log_likelihood(delta + eps, tau, lambda_param)
        
        self.E = E_saved.clone()
        self.log_Pi = log_Pi_saved.clone()
        ll_minus = self.compute_log_likelihood(delta - eps, tau, lambda_param)
        grad_delta = (ll_plus - ll_minus) / (2 * eps)
        
        # Tau gradient
        self.E = E_saved.clone()
        self.log_Pi = log_Pi_saved.clone()
        ll_plus = self.compute_log_likelihood(delta, tau + eps, lambda_param)
        
        self.E = E_saved.clone()
        self.log_Pi = log_Pi_saved.clone()
        ll_minus = self.compute_log_likelihood(delta, tau - eps, lambda_param)
        grad_tau = (ll_plus - ll_minus) / (2 * eps)
        
        # Lambda gradient
        self.E = E_saved.clone()
        self.log_Pi = log_Pi_saved.clone()
        ll_plus = self.compute_log_likelihood(delta, tau, lambda_param + eps)
        
        self.E = E_saved.clone()
        self.log_Pi = log_Pi_saved.clone()
        ll_minus = self.compute_log_likelihood(delta, tau, lambda_param - eps)
        grad_lambda = (ll_plus - ll_minus) / (2 * eps)
        
        # Compute current likelihood (don't reset state - maintain warm start)
        ll_current = self.compute_log_likelihood(delta, tau, lambda_param)
        
        return grad_delta, grad_tau, grad_lambda, ll_current


def optimize_finite_diff(species_tree_path, gene_tree_path,
                        init_delta=0.1, init_tau=0.001, init_lambda=0.001,
                        lr=0.01, epochs=200, device=None):
    """Optimize using finite difference gradients."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 80)
    print("Finite Difference Single-Iteration Optimization")
    print("=" * 80)
    
    # Build structures
    ccp = build_ccp_from_single_tree(gene_tree_path)
    species_helpers = build_species_helpers(species_tree_path, device, torch.float64)
    clade_species_map = build_clade_species_mapping(ccp, species_helpers, device, torch.float64)
    ccp_helpers = build_ccp_helpers(ccp, device, torch.float64)
    root_clade_id = get_root_clade_id(ccp)
    
    print(f"  {ccp_helpers['C']} clades, {species_helpers['S']} species")
    
    # Create optimizer
    optimizer = FiniteDiffCCPOptimizer(species_helpers, ccp_helpers, clade_species_map, root_clade_id)
    
    # Initialize parameters
    delta = init_delta
    tau = init_tau
    lambda_param = init_lambda
    
    # Warm start
    print("\nWarm start - converging E and Pi...")
    p_S, p_D, p_T, p_L = compute_probabilities(delta, tau, lambda_param)
    
    # Converge E
    for i in range(50):
        E_old = optimizer.E.clone()
        E_new, _, _, Ebar = E_step(
            optimizer.E, optimizer.s_C1, optimizer.s_C2, optimizer.Recipients_mat,
            p_S, p_D, p_T, p_L
        )
        optimizer.E = E_new
        if torch.abs(E_new - E_old).max() < 1e-10:
            print(f"  E converged after {i+1} iterations")
            break
    
    # Converge log_Pi
    for i in range(50):
        log_Pi_old = optimizer.log_Pi.clone()
        optimizer.log_Pi = Pi_update_ccp_log(
            optimizer.log_Pi, optimizer.ccp_helpers, optimizer.species_helpers,
            optimizer.clade_species_map, optimizer.E, Ebar, p_S, p_D, p_T
        )
        finite_mask = torch.isfinite(optimizer.log_Pi) & torch.isfinite(log_Pi_old)
        if finite_mask.any():
            max_diff = torch.abs(optimizer.log_Pi[finite_mask] - log_Pi_old[finite_mask]).max()
            if max_diff < 1e-10:
                print(f"  log_Pi converged after {i+1} iterations")
                break
    
    # Optimization loop
    print("\nStarting optimization...")
    print(f"{'Epoch':>6} {'δ':>10} {'τ':>10} {'λ':>10} {'Log-L':>12}")
    print("-" * 60)
    
    best_ll = float('-inf')
    best_params = None
    
    for epoch in range(epochs):
        # Compute gradients
        grad_delta, grad_tau, grad_lambda, ll = optimizer.compute_finite_diff_gradients(
            delta, tau, lambda_param
        )
        
        # Compute gradient norm for adaptive step size
        grad_norm = np.sqrt(grad_delta**2 + grad_tau**2 + grad_lambda**2)
        
        # Adaptive learning rate
        adaptive_lr = lr / max(1.0, grad_norm)
        
        # Update parameters using gradient ascent (maximize log-likelihood)
        delta_new = delta + adaptive_lr * grad_delta
        tau_new = tau + adaptive_lr * grad_tau
        lambda_new = lambda_param + adaptive_lr * grad_lambda
        
        # Ensure positive rates and reasonable bounds
        delta = np.clip(delta_new, 1e-10, 10.0)
        tau = np.clip(tau_new, 1e-10, 10.0)
        lambda_param = np.clip(lambda_new, 1e-10, 10.0)
        
        # Track best
        if ll > best_ll:
            best_ll = ll
            best_params = {
                'delta': delta,
                'tau': tau,
                'lambda': lambda_param,
                'log_likelihood': ll
            }
        
        # Print progress
        if epoch % 20 == 0 or epoch == epochs - 1:
            print(f"{epoch:6d} {delta:10.6f} {tau:10.6f} {lambda_param:10.6f} {ll:12.6f}")
    
    print("-" * 60)
    
    if best_params:
        print("\nBest parameters found:")
        print(f"  δ = {best_params['delta']:.6f}")
        print(f"  τ = {best_params['tau']:.6f}")
        print(f"  λ = {best_params['lambda']:.6f}")
        print(f"  Log-likelihood = {best_params['log_likelihood']:.6f}")
        
        print("\nExpected (test_trees_3):")
        print(f"  δ = 0.055554")
        print(f"  τ = 0.000000")
        print(f"  λ = 0.000000")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--species', required=True)
    parser.add_argument('--gene', required=True)
    parser.add_argument('--init-delta', type=float, default=0.055)
    parser.add_argument('--init-tau', type=float, default=0.001)
    parser.add_argument('--init-lambda', type=float, default=0.001)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=200)
    
    args = parser.parse_args()
    
    optimize_finite_diff(
        args.species, args.gene,
        init_delta=args.init_delta,
        init_tau=args.init_tau,
        init_lambda=args.init_lambda,
        lr=args.lr,
        epochs=args.epochs
    )


if __name__ == '__main__':
    main()