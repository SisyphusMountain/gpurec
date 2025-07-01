#!/usr/bin/env python3
"""
Single-iteration gradient optimization for CCP reconciliation.

This implements the simplified gradient approach discussed in implicit_differentiation.md,
where we compute gradients through single fixed-point iterations rather than the full
implicit differentiation approach.

Key features:
- Warm start: Maintain E and Pi across optimization steps
- Single iteration updates for computational efficiency
- Gradient computation through automatic differentiation
- Should converge to MLE under reasonable conditions
"""

import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import json
from time import time
import matplotlib.pyplot as plt
import numpy as np

# Import CCP functions
from matmul_ale_ccp import (
    build_ccp_from_single_tree,
    build_species_helpers,
    build_clade_species_mapping,
    build_ccp_helpers,
    get_root_clade_id
)

# Import log-space Pi update
from matmul_ale_ccp_log import Pi_update_ccp_log


class SingleIterationCCPOptimizer(nn.Module):
    """
    Optimizer for CCP reconciliation using single-iteration gradients.
    
    This class maintains E and Pi across iterations (warm start) and computes
    gradients through single fixed-point iterations.
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
        
        # Initialize E and log_Pi (will be maintained across iterations)
        self.E = torch.zeros(self.S, dtype=torch.float64, device=species_helpers["s_C1"].device)
        self.log_Pi = torch.full((self.C, self.S), float('-inf'), dtype=torch.float64, device=species_helpers["s_C1"].device)
        
        # Initialize log_Pi for leaf clades
        self._initialize_leaf_probabilities()
        
        # Store extinction probability helpers
        self.s_C1 = species_helpers["s_C1"]
        self.s_C2 = species_helpers["s_C2"]
        self.Recipients_mat = species_helpers["Recipients_mat"]
        
        # Use log parameterization for rates to ensure positivity
        # Initialize with reasonable values
        device = species_helpers["s_C1"].device
        self.log_delta = nn.Parameter(torch.tensor(-2.0, dtype=torch.float64, device=device))  # exp(-2) ≈ 0.135
        self.log_tau = nn.Parameter(torch.tensor(-2.0, dtype=torch.float64, device=device))
        self.log_lambda = nn.Parameter(torch.tensor(-2.0, dtype=torch.float64, device=device))
    
    def _initialize_leaf_probabilities(self):
        """Initialize log probabilities for leaf clades."""
        ccp = self.ccp_helpers['ccp']
        
        for c in range(self.C):
            clade = ccp.id_to_clade[c]
            if clade.is_leaf():
                mapped_species = torch.nonzero(self.clade_species_map[c] > 0, as_tuple=False).flatten()
                if len(mapped_species) > 0:
                    # Uniform probability among mapped species
                    log_prob = -torch.log(torch.tensor(len(mapped_species), dtype=torch.float64))
                    self.log_Pi[c, mapped_species] = log_prob
    
    def E_step_single(self, E, p_S, p_D, p_T, p_L):
        """Single E update step with gradient tracking."""
        E_s1 = torch.mv(self.s_C1, E)
        E_s2 = torch.mv(self.s_C2, E)
        speciation = p_S * E_s1 * E_s2
        
        duplication = p_D * E * E
        
        Ebar = torch.mv(self.Recipients_mat, E)
        transfer = p_T * E * Ebar
        
        E_new = speciation + duplication + transfer + p_L
        return E_new, Ebar
    
    def get_rates(self):
        """Get the actual rates from log parameterization."""
        delta = torch.exp(self.log_delta)
        tau = torch.exp(self.log_tau)
        lambda_param = torch.exp(self.log_lambda)
        return delta, tau, lambda_param
    
    def compute_probabilities(self, delta, tau, lambda_param):
        """Compute event probabilities from rates."""
        rates_sum = 1.0 + delta + tau + lambda_param
        p_S = 1.0 / rates_sum
        p_D = delta / rates_sum
        p_T = tau / rates_sum
        p_L = lambda_param / rates_sum
        return p_S, p_D, p_T, p_L
    
    def forward(self):
        """
        Perform single iteration updates and compute log-likelihood.
        
        Returns:
            log_likelihood: The log-likelihood for the current parameters
        """
        # Get current rates
        delta, tau, lambda_param = self.get_rates()
        
        # Clamp rates to reasonable bounds to avoid numerical issues
        delta = torch.clamp(delta, min=1e-10, max=10.0)
        tau = torch.clamp(tau, min=1e-10, max=10.0)
        lambda_param = torch.clamp(lambda_param, min=1e-10, max=10.0)
        
        # Compute event probabilities
        p_S, p_D, p_T, p_L = self.compute_probabilities(delta, tau, lambda_param)
        
        # Single E update (detach to avoid backprop through previous iterations)
        E_detached = self.E.detach()
        E_new, Ebar = self.E_step_single(E_detached, p_S, p_D, p_T, p_L)
        # Clamp E values to avoid numerical issues
        E_new = torch.clamp(E_new, min=0.0, max=1.0)
        self.E = E_new  # Update stored E
        
        # Single Pi update in log space (detach previous log_Pi)
        log_Pi_detached = self.log_Pi.detach()
        log_Pi_new = Pi_update_ccp_log(
            log_Pi_detached, self.ccp_helpers, self.species_helpers,
            self.clade_species_map, E_new, Ebar, p_S, p_D, p_T
        )
        self.log_Pi = log_Pi_new  # Update stored log_Pi
        
        # Compute log-likelihood
        root_log_pi = log_Pi_new[self.root_clade_id, :]
        # Check for numerical issues
        if torch.all(torch.isinf(root_log_pi)):
            # All values are -inf, return a large negative value instead
            return torch.tensor(-1000.0, dtype=torch.float64, device=root_log_pi.device, requires_grad=True)
        
        log_likelihood = torch.logsumexp(root_log_pi, dim=0)
        
        # Check for NaN
        if torch.isnan(log_likelihood):
            print(f"WARNING: NaN detected in log-likelihood computation")
            print(f"  Rates: delta={delta:.6f}, tau={tau:.6f}, lambda={lambda_param:.6f}")
            print(f"  Probabilities: p_S={p_S:.6f}, p_D={p_D:.6f}, p_T={p_T:.6f}, p_L={p_L:.6f}")
            print(f"  E range: [{E_new.min():.6f}, {E_new.max():.6f}]")
            print(f"  Root log_pi range: [{root_log_pi[torch.isfinite(root_log_pi)].min() if torch.any(torch.isfinite(root_log_pi)) else float('nan'):.6f}, "
                  f"{root_log_pi[torch.isfinite(root_log_pi)].max() if torch.any(torch.isfinite(root_log_pi)) else float('nan'):.6f}]")
            return torch.tensor(-1000.0, dtype=torch.float64, device=root_log_pi.device, requires_grad=True)
        
        return log_likelihood
    
    def get_current_values(self):
        """Get current parameter values and likelihood."""
        with torch.no_grad():
            delta, tau, lambda_param = self.get_rates()
            log_likelihood = self.forward()
            
        return {
            'delta': float(delta),
            'tau': float(tau),
            'lambda': float(lambda_param),
            'log_likelihood': float(log_likelihood)
        }


def optimize_single_iteration(species_tree_path, gene_tree_path, 
                            init_delta=0.1, init_tau=0.1, init_lambda=0.1,
                            lr=0.01, epochs=200, 
                            e_warm_start_iters=10, pi_warm_start_iters=10,
                            device=None, dtype=torch.float64):
    """
    Optimize CCP reconciliation parameters using single-iteration gradients.
    
    Args:
        species_tree_path: Path to species tree
        gene_tree_path: Path to gene tree
        init_delta, init_tau, init_lambda: Initial parameter values
        lr: Learning rate
        epochs: Number of optimization epochs
        e_warm_start_iters: Initial E iterations for warm start
        pi_warm_start_iters: Initial Pi iterations for warm start
        device: PyTorch device
        dtype: PyTorch dtype
    
    Returns:
        Dictionary with optimization results
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 80)
    print("Single-Iteration Gradient Optimization for CCP Reconciliation")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Initial parameters: δ={init_delta}, τ={init_tau}, λ={init_lambda}")
    print(f"Learning rate: {lr}")
    print(f"Epochs: {epochs}")
    print()
    
    # Build CCP structures
    print("Building CCP structures...")
    ccp = build_ccp_from_single_tree(gene_tree_path)
    species_helpers = build_species_helpers(species_tree_path, device, dtype)
    clade_species_map = build_clade_species_mapping(ccp, species_helpers, device, dtype)
    ccp_helpers = build_ccp_helpers(ccp, device, dtype)
    root_clade_id = get_root_clade_id(ccp)
    
    print(f"  {len(ccp.clades)} clades, {species_helpers['S']} species")
    print()
    
    # Create optimizer module
    model = SingleIterationCCPOptimizer(
        species_helpers, ccp_helpers, clade_species_map, root_clade_id
    ).to(device)
    
    # Initialize parameters
    with torch.no_grad():
        model.log_delta.data = torch.log(torch.tensor(init_delta, dtype=dtype))
        model.log_tau.data = torch.log(torch.tensor(init_tau, dtype=dtype))
        model.log_lambda.data = torch.log(torch.tensor(init_lambda, dtype=dtype))
    
    # Warm start: Run several iterations to get reasonable E and Pi
    print("Warm start phase...")
    with torch.no_grad():
        delta, tau, lambda_param = model.get_rates()
        p_S, p_D, p_T, p_L = model.compute_probabilities(delta, tau, lambda_param)
        
        # Warm start E
        for i in range(e_warm_start_iters):
            E_new, _ = model.E_step_single(model.E, p_S, p_D, p_T, p_L)
            model.E = E_new
        
        # Warm start Pi
        for i in range(pi_warm_start_iters):
            _, Ebar = model.E_step_single(model.E, p_S, p_D, p_T, p_L)
            log_Pi_new = Pi_update_ccp_log(
                model.log_Pi, model.ccp_helpers, model.species_helpers,
                model.clade_species_map, model.E, Ebar, p_S, p_D, p_T
            )
            model.log_Pi = log_Pi_new
    
    initial_values = model.get_current_values()
    print(f"After warm start:")
    print(f"  δ={initial_values['delta']:.6f}, τ={initial_values['tau']:.6f}, λ={initial_values['lambda']:.6f}")
    print(f"  Log-likelihood: {initial_values['log_likelihood']:.6f}")
    print()
    
    # Create optimizer with lower learning rate for stability
    optimizer = optim.Adam(model.parameters(), lr=lr, eps=1e-8)
    
    # Training history
    history = {
        'epoch': [],
        'delta': [],
        'tau': [],
        'lambda': [],
        'log_likelihood': [],
        'grad_norm': []
    }
    
    # Best values tracking
    best_log_likelihood = float('-inf')
    best_params = None
    
    print("Starting optimization...")
    print("-" * 80)
    print(f"{'Epoch':>6} {'δ':>10} {'τ':>10} {'λ':>10} {'Log-L':>12} {'Grad Norm':>10}")
    print("-" * 80)
    
    for epoch in range(epochs):
        # Forward pass
        log_likelihood = model()
        
        # Compute loss (negative log-likelihood for minimization)
        loss = -log_likelihood
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Compute gradient norm
        grad_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                grad_norm += param.grad.norm().item() ** 2
        grad_norm = grad_norm ** 0.5
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update parameters
        optimizer.step()
        
        # Get current values
        current_values = model.get_current_values()
        
        # Store history
        history['epoch'].append(epoch)
        history['delta'].append(current_values['delta'])
        history['tau'].append(current_values['tau'])
        history['lambda'].append(current_values['lambda'])
        history['log_likelihood'].append(current_values['log_likelihood'])
        history['grad_norm'].append(grad_norm)
        
        # Track best parameters
        if current_values['log_likelihood'] > best_log_likelihood:
            best_log_likelihood = current_values['log_likelihood']
            best_params = current_values.copy()
        
        # Print progress
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"{epoch:6d} {current_values['delta']:10.6f} {current_values['tau']:10.6f} "
                  f"{current_values['lambda']:10.6f} {current_values['log_likelihood']:12.6f} {grad_norm:10.6f}")
    
    print("-" * 80)
    print()
    print("Optimization completed!")
    print()
    if best_params is not None:
        print("Best parameters found:")
        print(f"  δ = {best_params['delta']:.6f}")
        print(f"  τ = {best_params['tau']:.6f}")
        print(f"  λ = {best_params['lambda']:.6f}")
        print(f"  Log-likelihood = {best_params['log_likelihood']:.6f}")
    else:
        print("No improvement found during optimization.")
        print("Final parameters:")
        print(f"  δ = {current_values['delta']:.6f}")
        print(f"  τ = {current_values['tau']:.6f}")
        print(f"  λ = {current_values['lambda']:.6f}")
        print(f"  Log-likelihood = {current_values['log_likelihood']:.6f}")
    
    # Save results
    results = {
        'best_params': best_params,
        'final_params': current_values,
        'history': history,
        'config': {
            'species_tree': species_tree_path,
            'gene_tree': gene_tree_path,
            'init_delta': init_delta,
            'init_tau': init_tau,
            'init_lambda': init_lambda,
            'lr': lr,
            'epochs': epochs,
            'e_warm_start_iters': e_warm_start_iters,
            'pi_warm_start_iters': pi_warm_start_iters
        }
    }
    
    output_file = 'single_iter_optimization_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")
    
    return results


def plot_optimization_history(history, output_file='optimization_history.png'):
    """Plot the optimization history."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    epochs = history['epoch']
    
    # Plot parameters
    axes[0, 0].plot(epochs, history['delta'], 'b-', label='δ (duplication)')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Rate')
    axes[0, 0].set_title('Duplication Rate')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(epochs, history['tau'], 'r-', label='τ (transfer)')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Rate')
    axes[0, 1].set_title('Transfer Rate')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(epochs, history['lambda'], 'g-', label='λ (loss)')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Rate')
    axes[1, 0].set_title('Loss Rate')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot log-likelihood
    axes[1, 1].plot(epochs, history['log_likelihood'], 'k-')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Log-likelihood')
    axes[1, 1].set_title('Log-likelihood')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"Plot saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Single-iteration gradient optimization for CCP reconciliation'
    )
    parser.add_argument('--species', required=True, help='Species tree file')
    parser.add_argument('--gene', required=True, help='Gene tree file')
    parser.add_argument('--init-delta', type=float, default=0.1, help='Initial duplication rate')
    parser.add_argument('--init-tau', type=float, default=0.1, help='Initial transfer rate')
    parser.add_argument('--init-lambda', type=float, default=0.1, help='Initial loss rate')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--e-warm-start', type=int, default=10, help='E warm start iterations')
    parser.add_argument('--pi-warm-start', type=int, default=10, help='Pi warm start iterations')
    parser.add_argument('--plot', action='store_true', help='Plot optimization history')
    
    args = parser.parse_args()
    
    # Run optimization
    results = optimize_single_iteration(
        args.species, args.gene,
        init_delta=args.init_delta,
        init_tau=args.init_tau,
        init_lambda=args.init_lambda,
        lr=args.lr,
        epochs=args.epochs,
        e_warm_start_iters=args.e_warm_start,
        pi_warm_start_iters=args.pi_warm_start
    )
    
    # Plot if requested
    if args.plot:
        plot_optimization_history(results['history'])
    
    return 0


if __name__ == '__main__':
    main()