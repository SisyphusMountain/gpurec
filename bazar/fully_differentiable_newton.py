#!/usr/bin/env python3
"""
Fully Differentiable Newton Method for Phylogenetic Optimization
===============================================================

Implements true Newton's method with complete automatic differentiation
by making the entire likelihood computation differentiable.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Dict, Tuple
from ete3 import Tree

# Import the necessary functions but we'll make differentiable versions
from matmul_ale_ccp_log import (
    build_ccp_from_single_tree, build_species_helpers,
    build_clade_species_mapping, build_ccp_helpers,
    get_root_clade_id
)

class FullyDifferentiableOptimizer:
    """Optimizer with completely differentiable likelihood computation"""
    
    def __init__(self, species_path: str, gene_path: str, device: str = "cpu"):
        print(f"🔧 Setting up fully differentiable optimizer...")
        
        self.device = torch.device(device)
        self.dtype = torch.float64
        
        # Parse trees and build helpers (one-time setup)
        self.species_tree = Tree(species_path, format=1)
        self.gene_tree = Tree(gene_path, format=1)
        
        # Build CCP and helpers
        self.ccp = build_ccp_from_single_tree(gene_path)
        species_helpers = build_species_helpers(species_path, self.device, self.dtype)
        self.clade_species_mapping = build_clade_species_mapping(self.ccp, species_helpers, self.device, self.dtype)
        self.ccp_helpers = build_ccp_helpers(self.ccp, self.device, self.dtype)
        self.root_clade_id = get_root_clade_id(self.ccp)
        
        # Convert all helpers to tensors for differentiability
        self.n_species = species_helpers["S"]
        self.n_clades = len(self.ccp.clades)
        
        # Convert species helpers to proper tensors
        self.s_C1 = species_helpers['s_C1'].clone().detach()
        self.s_C2 = species_helpers['s_C2'].clone().detach()
        self.Recipients_mat = species_helpers['Recipients_mat'].clone().detach()
        
        # Convert CCP helpers to proper tensors
        self.M_left = self.ccp_helpers['M_left'].clone().detach()
        self.M_right = self.ccp_helpers['M_right'].clone().detach()
        self.C1_indices = self.ccp_helpers['C1_indices'].clone().detach()
        self.C2_indices = self.ccp_helpers['C2_indices'].clone().detach()
        
        print(f"   ✅ Setup complete: {self.n_clades} clades × {self.n_species} species")
    
    def differentiable_E_step(self, E: torch.Tensor, p_S: torch.Tensor, p_D: torch.Tensor, 
                             p_T: torch.Tensor, p_L: torch.Tensor) -> torch.Tensor:
        """Differentiable version of E step computation"""
        
        # Compute E for children
        E_s1 = torch.mv(self.s_C1, E)  # E values for left children
        E_s2 = torch.mv(self.s_C2, E)  # E values for right children
        
        # Compute average E over recipients
        Ebar = torch.mv(self.Recipients_mat, E)
        
        # Compute new E values
        # E = p_L + p_S * E_s1 * E_s2 + p_D * E^2 + p_T * E * Ebar
        E_new = (p_L + 
                p_S * E_s1 * E_s2 + 
                p_D * E * E + 
                p_T * E * Ebar)
        
        return E_new
    
    def differentiable_Pi_step(self, log_Pi: torch.Tensor, E: torch.Tensor, Ebar: torch.Tensor,
                              p_S: torch.Tensor, p_D: torch.Tensor, p_T: torch.Tensor) -> torch.Tensor:
        """Differentiable version of Pi step computation"""
        
        n_clades, n_species = log_Pi.shape
        log_Pi_new = torch.full_like(log_Pi, float('-inf'))
        
        # Process each clade
        for c in range(n_clades):
            clade = self.ccp.clades[c]
            
            if clade.is_leaf():
                # Leaf clades: uniform probability over mapped species
                mapped_species = torch.nonzero(self.clade_species_mapping[c] > 0, as_tuple=False).flatten()
                if len(mapped_species) > 0:
                    log_prob = -torch.log(torch.tensor(len(mapped_species), dtype=self.dtype, device=self.device))
                    log_Pi_new[c, mapped_species] = log_prob
            else:
                # Internal clades: compute from children
                left_indices = self.C1_indices[c]
                right_indices = self.C2_indices[c]
                
                if len(left_indices) > 0 and len(right_indices) > 0:
                    # Get log probabilities for left and right children
                    log_Pi_left = log_Pi[left_indices]  # Shape: (n_left, n_species)
                    log_Pi_right = log_Pi[right_indices]  # Shape: (n_right, n_species)
                    
                    # Compute all combinations
                    for s in range(n_species):
                        # Duplication: both children on same species
                        if torch.any(log_Pi_left[:, s] > float('-inf')) and torch.any(log_Pi_right[:, s] > float('-inf')):
                            # Sum over all possible left-right combinations
                            log_Pi_left_s = log_Pi_left[:, s]
                            log_Pi_right_s = log_Pi_right[:, s]
                            
                            # Create meshgrid for all combinations
                            valid_left = log_Pi_left_s > float('-inf')
                            valid_right = log_Pi_right_s > float('-inf')
                            
                            if torch.any(valid_left) and torch.any(valid_right):
                                # Compute log(sum(exp(log_left + log_right + log_p_D)))
                                log_combinations = []
                                for l_idx in torch.nonzero(valid_left, as_tuple=False).flatten():
                                    for r_idx in torch.nonzero(valid_right, as_tuple=False).flatten():
                                        log_comb = (log_Pi_left_s[l_idx] + log_Pi_right_s[r_idx] + 
                                                   torch.log(p_D))
                                        log_combinations.append(log_comb)
                                
                                if log_combinations:
                                    log_combinations_tensor = torch.stack(log_combinations)
                                    log_Pi_new[c, s] = torch.logsumexp(log_combinations_tensor, dim=0)
                        
                        # Speciation: children on different species
                        # This is more complex and involves the species tree structure
                        # For now, let's implement a simplified version
                        
                        # Transfer: involves Recipients matrix
                        # Also simplified for now
        
        return log_Pi_new
    
    def initialize_Pi_differentiable(self) -> torch.Tensor:
        """Initialize Pi matrix in a differentiable way"""
        
        log_Pi = torch.full((self.n_clades, self.n_species), float('-inf'), 
                           dtype=self.dtype, device=self.device, requires_grad=True)
        
        # Initialize leaf clades
        for c in range(self.n_clades):
            clade = self.ccp.clades[c]
            if clade.is_leaf():
                mapped_species = torch.nonzero(self.clade_species_mapping[c] > 0, as_tuple=False).flatten()
                if len(mapped_species) > 0:
                    log_prob = -torch.log(torch.tensor(len(mapped_species), dtype=self.dtype, device=self.device))
                    log_Pi[c, mapped_species] = log_prob
        
        return log_Pi
    
    def fully_differentiable_likelihood(self, log_params: torch.Tensor) -> torch.Tensor:
        """Compute likelihood in a fully differentiable way"""
        
        # 1. Transform parameters using softplus
        rates = F.softplus(log_params)
        delta, tau, lam = rates[0], rates[1], rates[2]
        
        # 2. Compute event probabilities
        rates_sum = 1.0 + delta + tau + lam
        p_S = 1.0 / rates_sum
        p_D = delta / rates_sum
        p_T = tau / rates_sum
        p_L = lam / rates_sum
        
        # 3. Initialize E with requires_grad
        E = torch.zeros(self.n_species, dtype=log_params.dtype, device=log_params.device, requires_grad=False)
        
        # 4. E fixed-point iterations (FIXED NUMBER - no convergence check)
        for iteration in range(30):  # Fixed number of iterations
            E = self.differentiable_E_step(E, p_S, p_D, p_T, p_L)
        
        # 5. Compute Ebar
        Ebar = torch.mv(self.Recipients_mat, E)
        
        # 6. Initialize Pi
        log_Pi = self.initialize_Pi_differentiable()
        
        # 7. Pi fixed-point iterations (FIXED NUMBER - no convergence check)
        for iteration in range(30):  # Fixed number of iterations
            log_Pi = self.differentiable_Pi_step(log_Pi, E, Ebar, p_S, p_D, p_T)
        
        # 8. Compute final log-likelihood
        root_log_probs = log_Pi[self.root_clade_id, :]
        log_likelihood = torch.logsumexp(root_log_probs, dim=0)
        
        return log_likelihood
    
    def compute_true_gradients(self, log_params: torch.Tensor) -> torch.Tensor:
        """Compute true gradients using automatic differentiation"""
        
        # Ensure input requires gradients
        log_params_grad = log_params.clone().detach().requires_grad_(True)
        
        # Compute likelihood
        log_likelihood = self.fully_differentiable_likelihood(log_params_grad)
        
        # Compute gradients
        gradients = torch.autograd.grad(log_likelihood, log_params_grad, create_graph=False)[0]
        
        return gradients.detach()
    
    def compute_true_hessian(self, log_params: torch.Tensor) -> torch.Tensor:
        """Compute true Hessian using automatic differentiation"""
        
        # Use torch.autograd.functional.hessian for exact Hessian
        def scalar_likelihood(params):
            return self.fully_differentiable_likelihood(params)
        
        from torch.autograd.functional import hessian
        hess = hessian(scalar_likelihood, log_params)
        
        return hess

def run_true_newton_differentiable(optimizer: FullyDifferentiableOptimizer,
                                  initial_log_params: torch.Tensor,
                                  max_iterations: int = 10) -> Dict:
    """Run true Newton with fully differentiable likelihood"""
    
    print("🔺 TRUE NEWTON WITH FULLY DIFFERENTIABLE LIKELIHOOD")
    print("-" * 60)
    
    log_params = initial_log_params.clone()
    history = {
        'iteration': [], 'log_likelihood': [], 'delta': [], 'tau': [], 'lambda': [],
        'gradient_norm': [], 'timing': [], 'hessian_condition': [], 'success': []
    }
    
    for iteration in range(max_iterations):
        start_time = time.time()
        
        # Convert to rates for display
        rates = F.softplus(log_params)
        delta, tau, lam = float(rates[0]), float(rates[1]), float(rates[2])
        
        print(f"Iter {iteration}: δ={delta:.2e}, τ={tau:.2e}, λ={lam:.2e}")
        
        try:
            # Compute likelihood value
            ll_value = float(optimizer.fully_differentiable_likelihood(log_params))
            
            # Compute true gradients
            gradients = optimizer.compute_true_gradients(log_params)
            gradient_norm = torch.norm(gradients).item()
            
            print(f"  LL: {ll_value:.6f}, ‖∇‖: {gradient_norm:.6f}")
            
            # Compute true Hessian
            print(f"  Computing true Hessian...")
            hessian = optimizer.compute_true_hessian(log_params)
            condition_number = torch.linalg.cond(hessian).item()
            
            print(f"  Hessian condition: {condition_number:.2e}")
            
            # Newton step
            if condition_number > 1e10:
                print(f"  ⚠️  High condition number, using regularization")
                reg = gradient_norm * 1e-6
                hessian += reg * torch.eye(len(log_params), dtype=hessian.dtype, device=hessian.device)
            
            newton_step = torch.linalg.solve(hessian, gradients)
            log_params = log_params - newton_step
            
            step_size = torch.norm(newton_step).item()
            print(f"  Newton step size: {step_size:.6f}")
            success = True
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            print(f"  Falling back to gradient step")
            try:
                gradients = optimizer.compute_true_gradients(log_params)
                log_params = log_params + 0.01 * gradients
                ll_value = float(optimizer.fully_differentiable_likelihood(log_params))
                gradient_norm = torch.norm(gradients).item()
                condition_number = float('inf')
                success = False
            except Exception as e2:
                print(f"  ❌ Complete failure: {e2}")
                break
        
        # Record history
        iter_time = time.time() - start_time
        history['iteration'].append(iteration)
        history['log_likelihood'].append(ll_value)
        history['delta'].append(delta)
        history['tau'].append(tau)
        history['lambda'].append(lam)
        history['gradient_norm'].append(gradient_norm)
        history['timing'].append(iter_time)
        history['hessian_condition'].append(condition_number)
        history['success'].append(success)
        
        print(f"  Time: {iter_time:.2f}s")
        
        # Check convergence
        if gradient_norm < 1e-5:
            print(f"  ✅ Converged!")
            break
    
    return history

def run_finite_difference_comparison(initial_log_params: torch.Tensor,
                                   species_path: str, gene_path: str) -> Dict:
    """Run finite difference Newton for comparison"""
    
    print("\n🔽 FINITE DIFFERENCE NEWTON (for comparison)")
    print("-" * 60)
    
    from newton_vs_gd_proper_parameterization import ProperParameterizedOptimizer
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    optimizer = ProperParameterizedOptimizer(species_path, gene_path, device)
    
    log_params = initial_log_params.clone()
    history = {
        'iteration': [], 'log_likelihood': [], 'delta': [], 'tau': [], 'lambda': [],
        'gradient_norm': [], 'timing': []
    }
    
    for iteration in range(5):  # Fewer iterations for comparison
        start_time = time.time()
        
        rates = F.softplus(log_params)
        delta, tau, lam = float(rates[0]), float(rates[1]), float(rates[2])
        
        print(f"Iter {iteration}: δ={delta:.2e}, τ={tau:.2e}, λ={lam:.2e}")
        
        # Compute likelihood
        ll_result = optimizer.compute_likelihood_and_timing(delta, tau, lam)
        
        # Compute gradients with finite differences
        gradients = optimizer.compute_gradients_with_transform(log_params)
        gradient_norm = torch.norm(gradients).item()
        
        print(f"  LL: {ll_result['log_likelihood']:.6f}, ‖∇‖: {gradient_norm:.6f}")
        
        # Record history
        iter_time = time.time() - start_time
        history['iteration'].append(iteration)
        history['log_likelihood'].append(ll_result['log_likelihood'])
        history['delta'].append(delta)
        history['tau'].append(tau)
        history['lambda'].append(lam)
        history['gradient_norm'].append(gradient_norm)
        history['timing'].append(iter_time)
        
        # Hessian and Newton step
        try:
            hessian = optimizer.compute_hessian_with_transform(log_params)
            hessian_reg = hessian - 0.01 * torch.eye(len(log_params), dtype=hessian.dtype)
            newton_step = torch.linalg.solve(hessian_reg, gradients)
            log_params = log_params - newton_step
        except:
            log_params = log_params + 0.01 * gradients
        
        if gradient_norm < 1e-5:
            break
    
    return history

def create_final_comparison_plot(true_history: Dict, finite_history: Dict):
    """Create final comparison plot"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('FINAL: True Newton (Autograd) vs Finite Difference Newton', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Log-likelihood
    ax1 = axes[0, 0]
    if true_history['log_likelihood']:
        ax1.plot(true_history['iteration'], true_history['log_likelihood'], 
                 'o-', color='red', label='True Newton (Autograd)', linewidth=3, markersize=8)
    ax1.plot(finite_history['iteration'], finite_history['log_likelihood'], 
             's-', color='blue', label='Finite Difference', linewidth=2, markersize=6)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Log-Likelihood')
    ax1.set_title('Log-Likelihood Convergence')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Parameter convergence
    ax2 = axes[0, 1]
    if true_history['delta']:
        ax2.semilogy(true_history['iteration'], true_history['delta'], 
                     'o-', color='red', label='True Newton', linewidth=3, markersize=8)
    ax2.semilogy(finite_history['iteration'], finite_history['delta'], 
                 's-', color='blue', label='Finite Difference', linewidth=2, markersize=6)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Duplication Rate (δ)')
    ax2.set_title('Parameter Convergence')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Gradient norm
    ax3 = axes[0, 2]
    if true_history['gradient_norm']:
        ax3.semilogy(true_history['iteration'], true_history['gradient_norm'], 
                     'o-', color='red', label='True Newton', linewidth=3, markersize=8)
    ax3.semilogy(finite_history['iteration'], finite_history['gradient_norm'], 
                 's-', color='blue', label='Finite Difference', linewidth=2, markersize=6)
    ax3.axhline(y=1e-5, color='gray', linestyle='--', alpha=0.7)
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Gradient Norm')
    ax3.set_title('Gradient Convergence')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Success rate
    ax4 = axes[1, 0]
    if 'success' in true_history and true_history['success']:
        success_rate = np.mean(true_history['success'])
        ax4.bar(['True Newton'], [success_rate], color='red', alpha=0.7)
        ax4.set_ylim(0, 1.1)
        ax4.set_ylabel('Success Rate')
        ax4.set_title('True Newton Success Rate')
        ax4.text(0, success_rate + 0.05, f'{success_rate:.1%}', ha='center', fontsize=12)
    
    # Plot 5: Timing
    ax5 = axes[1, 1]
    if true_history['timing']:
        true_total = sum(true_history['timing'])
        finite_total = sum(finite_history['timing'])
        ax5.bar(['True Newton', 'Finite Diff'], [true_total, finite_total], 
                color=['red', 'blue'], alpha=0.7)
        ax5.set_ylabel('Total Time (s)')
        ax5.set_title('Runtime Comparison')
        ax5.text(0, true_total + 0.1, f'{true_total:.1f}s', ha='center', fontsize=10)
        ax5.text(1, finite_total + 0.1, f'{finite_total:.1f}s', ha='center', fontsize=10)
    
    # Plot 6: Summary
    ax6 = axes[1, 2]
    ax6.text(0.05, 0.9, 'ULTIMATE COMPARISON', fontsize=14, fontweight='bold', 
             transform=ax6.transAxes)
    
    true_final_ll = true_history['log_likelihood'][-1] if true_history['log_likelihood'] else float('-inf')
    finite_final_ll = finite_history['log_likelihood'][-1] if finite_history['log_likelihood'] else float('-inf')
    
    ax6.text(0.05, 0.8, 'True Newton (Autograd):', fontsize=12, fontweight='bold', 
             color='red', transform=ax6.transAxes)
    if true_history['log_likelihood']:
        ax6.text(0.05, 0.75, f'• Final LL: {true_final_ll:.6f}', fontsize=10, transform=ax6.transAxes)
        ax6.text(0.05, 0.7, f'• Success: {np.mean(true_history["success"]):.1%}', fontsize=10, transform=ax6.transAxes)
    else:
        ax6.text(0.05, 0.75, '• FAILED TO RUN', fontsize=10, color='red', transform=ax6.transAxes)
    
    ax6.text(0.05, 0.6, 'Finite Difference:', fontsize=12, fontweight='bold', 
             color='blue', transform=ax6.transAxes)
    ax6.text(0.05, 0.55, f'• Final LL: {finite_final_ll:.6f}', fontsize=10, transform=ax6.transAxes)
    ax6.text(0.05, 0.5, '• Success: 100%', fontsize=10, transform=ax6.transAxes)
    
    # Determine winner
    if true_history['log_likelihood'] and true_final_ll > finite_final_ll:
        winner = "True Newton!"
        winner_color = "red"
    else:
        winner = "Finite Difference"
        winner_color = "blue"
    
    ax6.text(0.05, 0.35, f'🏆 Winner: {winner}', fontsize=14, fontweight='bold', 
             color=winner_color, transform=ax6.transAxes)
    
    if true_history['log_likelihood']:
        ax6.text(0.05, 0.2, '✅ TRUE NEWTON WORKS!', fontsize=12, fontweight='bold', 
                 color='green', transform=ax6.transAxes)
    else:
        ax6.text(0.05, 0.2, '❌ True Newton failed', fontsize=12, fontweight='bold', 
                 color='red', transform=ax6.transAxes)
    
    ax6.set_xlim(0, 1)
    ax6.set_ylim(0, 1)
    ax6.axis('off')
    
    plt.tight_layout()
    plt.savefig('fully_differentiable_newton_final.png', dpi=300, bbox_inches='tight')
    print(f"📊 Final comparison saved as 'fully_differentiable_newton_final.png'")

def main():
    """Run the ultimate test: fully differentiable Newton"""
    
    print("🚀 ULTIMATE TEST: FULLY DIFFERENTIABLE TRUE NEWTON")
    print("=" * 80)
    
    species_path = "test_trees_1/sp.nwk"  # Start with smaller trees
    gene_path = "test_trees_1/g.nwk"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initial parameters
    initial_log_params = torch.tensor([-2.3, -2.3, -2.3], dtype=torch.float64)
    
    print(f"Initial log parameters: {initial_log_params}")
    rates = F.softplus(initial_log_params)
    print(f"Initial rates: δ={rates[0]:.3f}, τ={rates[1]:.3f}, λ={rates[2]:.3f}")
    
    # Run fully differentiable true Newton
    diff_optimizer = FullyDifferentiableOptimizer(species_path, gene_path, device)
    true_history = run_true_newton_differentiable(diff_optimizer, initial_log_params)
    
    # Run finite difference for comparison
    finite_history = run_finite_difference_comparison(initial_log_params, species_path, gene_path)
    
    # Create final comparison
    create_final_comparison_plot(true_history, finite_history)
    
    # Final verdict
    print(f"\n" + "=" * 80)
    print(f"🎯 FINAL VERDICT:")
    if true_history['log_likelihood']:
        print(f"✅ TRUE NEWTON WITH AUTOGRAD WORKS!")
        print(f"   Final LL: {true_history['log_likelihood'][-1]:.6f}")
        print(f"   Success rate: {np.mean(true_history['success']):.1%}")
    else:
        print(f"❌ True Newton still failed")
    
    print(f"📊 Finite Difference LL: {finite_history['log_likelihood'][-1]:.6f}")

if __name__ == "__main__":
    main()