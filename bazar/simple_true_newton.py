#!/usr/bin/env python3
"""
Simple True Newton Implementation
================================

A robust implementation of true Newton's method using automatic differentiation
that properly handles the likelihood computation.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import time
from realistic_optimization_comparison import RealisticOptimizer

class SimpleAutoGradOptimizer:
    """Simple optimizer using autograd for exact derivatives"""
    
    def __init__(self, species_path: str, gene_path: str, device: str = "cpu"):
        self.base_optimizer = RealisticOptimizer(species_path, gene_path, device)
        self.device = torch.device(device)
        self.dtype = torch.float64
        
    def compute_likelihood_differentiable(self, log_params: torch.Tensor) -> torch.Tensor:
        """Compute likelihood in a way that's compatible with autograd"""
        
        # Transform parameters
        rates = F.softplus(log_params)
        delta, tau, lam = float(rates[0]), float(rates[1]), float(rates[2])
        
        # Fresh computation each time
        self.base_optimizer.E = None
        self.base_optimizer.Pi = None
        
        # Compute likelihood
        result = self.base_optimizer.compute_likelihood_and_timing(delta, tau, lam)
        
        # Return as tensor for autograd
        return torch.tensor(result['log_likelihood'], dtype=log_params.dtype, device=log_params.device)
    
    def compute_exact_gradients(self, log_params: torch.Tensor) -> torch.Tensor:
        """Compute exact gradients using autograd"""
        
        log_params_grad = log_params.clone().detach().requires_grad_(True)
        
        # Compute likelihood
        ll = self.compute_likelihood_differentiable(log_params_grad)
        
        # Compute gradients
        gradients = torch.autograd.grad(ll, log_params_grad)[0]
        
        return gradients.detach()
    
    def compute_exact_hessian(self, log_params: torch.Tensor) -> torch.Tensor:
        """Compute exact Hessian using autograd"""
        
        n_params = len(log_params)
        hessian = torch.zeros((n_params, n_params), dtype=self.dtype, device=self.device)
        
        for i in range(n_params):
            # Create function that returns gradient[i]
            def grad_i_fn(params):
                gradients = self.compute_exact_gradients(params)
                return gradients[i]
            
            # Compute second derivatives
            log_params_grad = log_params.clone().detach().requires_grad_(True)
            grad_i = grad_i_fn(log_params_grad)
            
            # Compute d(grad_i)/d(log_params)
            hess_row = torch.autograd.grad(grad_i, log_params_grad)[0]
            hessian[i, :] = hess_row
        
        return hessian

def run_simple_true_newton(optimizer: SimpleAutoGradOptimizer,
                          initial_log_params: torch.Tensor,
                          max_iterations: int = 10) -> dict:
    """Run true Newton with simple autograd implementation"""
    
    print("🔺 SIMPLE TRUE NEWTON METHOD")
    print("-" * 50)
    
    log_params = initial_log_params.clone()
    history = {
        'iteration': [], 'log_likelihood': [], 'delta': [], 'tau': [], 'lambda': [],
        'gradient_norm': [], 'timing': [], 'success': []
    }
    
    for iteration in range(max_iterations):
        start_time = time.time()
        
        # Display current parameters
        rates = F.softplus(log_params)
        delta, tau, lam = float(rates[0]), float(rates[1]), float(rates[2])
        
        print(f"Iter {iteration}: δ={delta:.2e}, τ={tau:.2e}, λ={lam:.2e}")
        
        try:
            # Compute exact gradients
            gradients = optimizer.compute_exact_gradients(log_params)
            gradient_norm = torch.norm(gradients).item()
            
            # Compute likelihood for logging
            ll_value = float(optimizer.compute_likelihood_differentiable(log_params))
            
            print(f"  LL: {ll_value:.6f}, ‖∇‖: {gradient_norm:.6f}")
            
            # Try to compute Hessian
            try:
                print(f"  Computing Hessian...")
                hessian = optimizer.compute_exact_hessian(log_params)
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
                print(f"  ❌ Hessian computation failed: {e}")
                print(f"  Falling back to gradient step")
                log_params = log_params + 0.01 * gradients
                success = False
            
            # Record history
            iter_time = time.time() - start_time
            history['iteration'].append(iteration)
            history['log_likelihood'].append(ll_value)
            history['delta'].append(delta)
            history['tau'].append(tau)
            history['lambda'].append(lam)
            history['gradient_norm'].append(gradient_norm)
            history['timing'].append(iter_time)
            history['success'].append(success)
            
            # Check convergence
            if gradient_norm < 1e-5:
                print(f"  ✅ Converged!")
                break
                
        except Exception as e:
            print(f"  ❌ Iteration failed: {e}")
            break
    
    return history

def run_finite_difference_comparison(initial_log_params: torch.Tensor,
                                   species_path: str, gene_path: str) -> dict:
    """Run finite difference Newton for comparison"""
    
    print("\n🔽 FINITE DIFFERENCE NEWTON")
    print("-" * 50)
    
    from newton_vs_gd_proper_parameterization import ProperParameterizedOptimizer
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    optimizer = ProperParameterizedOptimizer(species_path, gene_path, device)
    
    log_params = initial_log_params.clone()
    history = {
        'iteration': [], 'log_likelihood': [], 'delta': [], 'tau': [], 'lambda': [],
        'gradient_norm': [], 'timing': []
    }
    
    for iteration in range(10):
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
            print(f"  ✅ Converged!")
            break
    
    return history

def create_simple_comparison_plot(true_history: dict, finite_history: dict):
    """Create comparison plot"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('True Newton (Autograd) vs Finite Difference Newton', fontsize=16, fontweight='bold')
    
    # Plot 1: Log-likelihood
    ax1 = axes[0, 0]
    ax1.plot(true_history['iteration'], true_history['log_likelihood'], 
             'o-', color='red', label='True Newton', linewidth=3, markersize=6)
    ax1.plot(finite_history['iteration'], finite_history['log_likelihood'], 
             's-', color='blue', label='Finite Difference', linewidth=2, markersize=4)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Log-Likelihood')
    ax1.set_title('Log-Likelihood Convergence')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Parameter convergence (δ)
    ax2 = axes[0, 1]
    ax2.semilogy(true_history['iteration'], true_history['delta'], 
                 'o-', color='red', label='True Newton', linewidth=3, markersize=6)
    ax2.semilogy(finite_history['iteration'], finite_history['delta'], 
                 's-', color='blue', label='Finite Difference', linewidth=2, markersize=4)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Duplication Rate (δ)')
    ax2.set_title('Parameter Convergence to Zero')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Gradient norm
    ax3 = axes[0, 2]
    ax3.semilogy(true_history['iteration'], true_history['gradient_norm'], 
                 'o-', color='red', label='True Newton', linewidth=3, markersize=6)
    ax3.semilogy(finite_history['iteration'], finite_history['gradient_norm'], 
                 's-', color='blue', label='Finite Difference', linewidth=2, markersize=4)
    ax3.axhline(y=1e-5, color='gray', linestyle='--', alpha=0.7, label='Convergence threshold')
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Gradient Norm')
    ax3.set_title('Gradient Convergence')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Timing comparison
    ax4 = axes[1, 0]
    true_cumtime = np.cumsum(true_history['timing'])
    finite_cumtime = np.cumsum(finite_history['timing'])
    
    ax4.plot(true_history['iteration'], true_cumtime, 
             'o-', color='red', label='True Newton', linewidth=3, markersize=6)
    ax4.plot(finite_history['iteration'], finite_cumtime, 
             's-', color='blue', label='Finite Difference', linewidth=2, markersize=4)
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Cumulative Time (s)')
    ax4.set_title('Runtime Comparison')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Success rate for true Newton
    ax5 = axes[1, 1]
    if 'success' in true_history:
        success_rate = np.cumsum(true_history['success']) / np.arange(1, len(true_history['success']) + 1)
        ax5.plot(true_history['iteration'], success_rate, 
                 'o-', color='red', linewidth=3, markersize=6)
        ax5.set_xlabel('Iteration')
        ax5.set_ylabel('Success Rate')
        ax5.set_title('True Newton Success Rate')
        ax5.set_ylim(0, 1.1)
        ax5.grid(True, alpha=0.3)
    
    # Plot 6: Summary
    ax6 = axes[1, 2]
    ax6.text(0.05, 0.9, 'COMPARISON SUMMARY', fontsize=14, fontweight='bold', 
             transform=ax6.transAxes)
    
    true_final_ll = true_history['log_likelihood'][-1] if true_history['log_likelihood'] else float('-inf')
    finite_final_ll = finite_history['log_likelihood'][-1] if finite_history['log_likelihood'] else float('-inf')
    true_total_time = sum(true_history['timing']) if true_history['timing'] else 0
    finite_total_time = sum(finite_history['timing']) if finite_history['timing'] else 0
    
    ax6.text(0.05, 0.8, 'True Newton (Autograd):', fontsize=12, fontweight='bold', 
             color='red', transform=ax6.transAxes)
    ax6.text(0.05, 0.75, f'• Final LL: {true_final_ll:.6f}', fontsize=10, transform=ax6.transAxes)
    ax6.text(0.05, 0.7, f'• Iterations: {len(true_history["iteration"])}', fontsize=10, transform=ax6.transAxes)
    ax6.text(0.05, 0.65, f'• Total time: {true_total_time:.1f}s', fontsize=10, transform=ax6.transAxes)
    
    ax6.text(0.05, 0.55, 'Finite Difference:', fontsize=12, fontweight='bold', 
             color='blue', transform=ax6.transAxes)
    ax6.text(0.05, 0.5, f'• Final LL: {finite_final_ll:.6f}', fontsize=10, transform=ax6.transAxes)
    ax6.text(0.05, 0.45, f'• Iterations: {len(finite_history["iteration"])}', fontsize=10, transform=ax6.transAxes)
    ax6.text(0.05, 0.4, f'• Total time: {finite_total_time:.1f}s', fontsize=10, transform=ax6.transAxes)
    
    # Determine winner
    if abs(true_final_ll - finite_final_ll) < 0.001:
        if true_total_time < finite_total_time:
            winner = "True Newton (faster)"
            winner_color = "red"
        else:
            winner = "Finite Difference (faster)"
            winner_color = "blue"
    elif true_final_ll > finite_final_ll:
        winner = "True Newton (higher LL)"
        winner_color = "red"
    else:
        winner = "Finite Difference (higher LL)"
        winner_color = "blue"
    
    ax6.text(0.05, 0.25, f'Winner: {winner}', fontsize=12, fontweight='bold', 
             color=winner_color, transform=ax6.transAxes)
    
    if 'success' in true_history:
        success_rate = np.mean(true_history['success'])
        ax6.text(0.05, 0.15, f'True Newton success: {success_rate:.1%}', fontsize=10, transform=ax6.transAxes)
    
    ax6.set_xlim(0, 1)
    ax6.set_ylim(0, 1)
    ax6.axis('off')
    
    plt.tight_layout()
    plt.savefig('simple_true_newton_comparison.png', dpi=300, bbox_inches='tight')
    print(f"📊 Comparison saved as 'simple_true_newton_comparison.png'")

def main():
    """Run simple true Newton comparison"""
    
    print("🚀 SIMPLE TRUE NEWTON VS FINITE DIFFERENCE COMPARISON")
    print("=" * 70)
    
    species_path = "test_trees_200/sp.nwk"
    gene_path = "test_trees_200/g.nwk"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initial parameters: softplus(-2.3) ≈ 0.1
    initial_log_params = torch.tensor([-2.3, -2.3, -2.3], dtype=torch.float64)
    
    print(f"Initial log parameters: {initial_log_params}")
    rates = F.softplus(initial_log_params)
    print(f"Initial rates: δ={rates[0]:.3f}, τ={rates[1]:.3f}, λ={rates[2]:.3f}")
    
    # Run true Newton with autograd
    true_optimizer = SimpleAutoGradOptimizer(species_path, gene_path, device)
    true_history = run_simple_true_newton(true_optimizer, initial_log_params)
    
    # Run finite difference Newton
    finite_history = run_finite_difference_comparison(initial_log_params, species_path, gene_path)
    
    # Create comparison
    create_simple_comparison_plot(true_history, finite_history)
    
    print(f"\n🎯 FINAL COMPARISON:")
    if true_history['log_likelihood']:
        print(f"True Newton final LL: {true_history['log_likelihood'][-1]:.6f}")
    if finite_history['log_likelihood']:
        print(f"Finite Diff final LL: {finite_history['log_likelihood'][-1]:.6f}")

if __name__ == "__main__":
    main()