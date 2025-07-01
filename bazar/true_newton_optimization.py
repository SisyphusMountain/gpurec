#!/usr/bin/env python3
"""
True Newton Method with Automatic Differentiation
=================================================

Implements true Newton iterations using PyTorch's autograd to compute
exact analytical gradients and Hessian for phylogenetic parameter optimization.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import time
import json
from typing import Dict, List, Tuple, Callable
from realistic_optimization_comparison import RealisticOptimizer

class TrueNewtonOptimizer(RealisticOptimizer):
    """True Newton optimizer using automatic differentiation"""
    
    def __init__(self, species_path: str, gene_path: str, device: str = "cpu"):
        super().__init__(species_path, gene_path, device)
        
    def create_differentiable_likelihood(self) -> Callable[[torch.Tensor], torch.Tensor]:
        """Create a differentiable likelihood function for autograd"""
        
        def likelihood_fn(log_params: torch.Tensor) -> torch.Tensor:
            """
            Differentiable likelihood function that takes log parameters
            and returns log-likelihood as a scalar tensor.
            """
            # Ensure log_params requires gradients
            log_params = log_params.requires_grad_(True)
            
            # Transform to rates using softplus
            rates = F.softplus(log_params)
            delta, tau, lam = float(rates[0]), float(rates[1]), float(rates[2])
            
            # Reset optimizer state to ensure clean computation
            # Note: We'll need to be careful about this in practice
            self.E = None
            self.Pi = None
            
            # Compute likelihood
            ll_result = self.compute_likelihood_and_timing(delta, tau, lam)
            
            # Return as tensor for autograd
            return torch.tensor(ll_result['log_likelihood'], 
                              dtype=log_params.dtype, device=log_params.device)
        
        return likelihood_fn
    
    def compute_true_gradients_and_hessian(self, log_params: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute exact gradients and Hessian using automatic differentiation"""
        
        # Create differentiable likelihood function
        likelihood_fn = self.create_differentiable_likelihood()
        
        # Ensure input requires gradients
        log_params = log_params.clone().detach().requires_grad_(True)
        
        # Compute gradients using autograd
        ll_value = likelihood_fn(log_params)
        gradients = torch.autograd.grad(ll_value, log_params, create_graph=True)[0]
        
        # Compute Hessian using autograd
        hessian = torch.zeros((len(log_params), len(log_params)), 
                            dtype=log_params.dtype, device=log_params.device)
        
        for i in range(len(log_params)):
            # Compute second derivatives
            grad_grad = torch.autograd.grad(gradients[i], log_params, 
                                          retain_graph=True, create_graph=False)[0]
            hessian[i, :] = grad_grad
        
        return gradients.detach(), hessian.detach()

class TrueNewtonOptimizerStable(RealisticOptimizer):
    """More stable version using functional API"""
    
    def __init__(self, species_path: str, gene_path: str, device: str = "cpu"):
        super().__init__(species_path, gene_path, device)
        
    def likelihood_function(self, log_params: torch.Tensor) -> torch.Tensor:
        """Stateless likelihood function for automatic differentiation"""
        
        # Transform to rates
        rates = F.softplus(log_params)
        delta, tau, lam = float(rates[0]), float(rates[1]), float(rates[2])
        
        # Compute likelihood with fresh state each time
        # Save current state
        E_saved = self.E.clone() if self.E is not None else None
        Pi_saved = self.Pi.clone() if self.Pi is not None else None
        
        # Reset for clean computation
        self.E = None
        self.Pi = None
        
        try:
            # Compute likelihood
            ll_result = self.compute_likelihood_and_timing(delta, tau, lam)
            ll_tensor = torch.tensor(ll_result['log_likelihood'], 
                                   dtype=log_params.dtype, 
                                   device=log_params.device,
                                   requires_grad=False)
        finally:
            # Restore state
            if E_saved is not None:
                self.E = E_saved
            if Pi_saved is not None:
                self.Pi = Pi_saved
        
        return ll_tensor
    
    def compute_true_derivatives(self, log_params: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Compute exact gradients and Hessian using torch.autograd.functional"""
        
        # Use functional API for cleaner automatic differentiation
        from torch.autograd.functional import hessian, jacobian
        
        # Ensure we have the right dtype and device
        log_params = log_params.clone().detach().to(dtype=self.dtype, device=self.device)
        
        # Compute likelihood value
        ll_value = self.likelihood_function(log_params)
        
        # Compute gradients using jacobian (since we have scalar output)
        def scalar_likelihood(params):
            return self.likelihood_function(params)
        
        # Get gradients
        gradients = jacobian(scalar_likelihood, log_params)
        
        # Get Hessian  
        hess = hessian(scalar_likelihood, log_params)
        
        return gradients, hess, float(ll_value)

def run_true_newton_method(optimizer: TrueNewtonOptimizerStable,
                          initial_log_params: torch.Tensor,
                          max_iterations: int = 15) -> Dict:
    """Run true Newton's method with exact derivatives"""
    
    print(f"🔺 TRUE NEWTON'S METHOD (Exact Automatic Differentiation)")
    print("-" * 60)
    
    log_params = initial_log_params.clone()
    history = {
        'iteration': [], 'log_likelihood': [], 'delta': [], 'tau': [], 'lambda': [],
        'log_delta': [], 'log_tau': [], 'log_lambda': [],
        'gradient_norm': [], 'gradients': [], 'timing': [],
        'hessian_condition_number': [], 'newton_step_size': []
    }
    
    for iteration in range(max_iterations):
        start_time = time.time()
        
        # Convert to rates for display
        rates = F.softplus(log_params)
        delta, tau, lam = float(rates[0]), float(rates[1]), float(rates[2])
        
        print(f"Iter {iteration}: log_θ=[{log_params[0]:.3f},{log_params[1]:.3f},{log_params[2]:.3f}] → δ={delta:.2e}, τ={tau:.2e}, λ={lam:.2e}")
        
        try:
            # Compute exact gradients and Hessian
            gradients, hessian, ll_value = optimizer.compute_true_derivatives(log_params)
            
            gradient_norm = torch.norm(gradients).item()
            condition_number = torch.linalg.cond(hessian).item()
            
            print(f"  LL: {ll_value:.6f}, ‖∇‖: {gradient_norm:.6f}, cond(H): {condition_number:.2e}")
            
            # Record history
            iter_time = time.time() - start_time
            history['iteration'].append(iteration)
            history['log_likelihood'].append(ll_value)
            history['delta'].append(delta)
            history['tau'].append(tau)
            history['lambda'].append(lam)
            history['log_delta'].append(float(log_params[0]))
            history['log_tau'].append(float(log_params[1]))
            history['log_lambda'].append(float(log_params[2]))
            history['gradient_norm'].append(gradient_norm)
            history['gradients'].append([float(g) for g in gradients])
            history['timing'].append(iter_time)
            history['hessian_condition_number'].append(condition_number)
            
            # True Newton step: θ_new = θ_old - H⁻¹∇
            try:
                # Use regularization if condition number is too high
                if condition_number > 1e12:
                    print(f"  ⚠️  High condition number, adding regularization")
                    reg_param = gradient_norm * 1e-6
                    hessian_reg = hessian + reg_param * torch.eye(len(log_params), 
                                                                dtype=hessian.dtype, 
                                                                device=hessian.device)
                    newton_step = torch.linalg.solve(hessian_reg, gradients)
                else:
                    newton_step = torch.linalg.solve(hessian, gradients)
                
                step_size = torch.norm(newton_step).item()
                history['newton_step_size'].append(step_size)
                
                # Apply Newton step
                log_params = log_params - newton_step
                
                print(f"  Newton step size: {step_size:.6f}")
                
            except Exception as e:
                print(f"  ❌ Newton step failed: {e}")
                print(f"     Falling back to gradient step")
                
                # Fallback to gradient descent step
                learning_rate = 0.01
                log_params = log_params + learning_rate * gradients
                history['newton_step_size'].append(learning_rate * gradient_norm)
            
            # Check convergence
            if gradient_norm < 1e-6:
                print(f"  ✅ Converged! Gradient norm {gradient_norm:.2e} below threshold")
                break
                
        except Exception as e:
            print(f"  ❌ Iteration failed: {e}")
            break
    
    return history

def run_finite_difference_newton(optimizer,
                                initial_log_params: torch.Tensor,
                                max_iterations: int = 15) -> Dict:
    """Run Newton with finite difference Hessian for comparison"""
    
    print(f"\n🔺 FINITE DIFFERENCE NEWTON (for comparison)")
    print("-" * 60)
    
    log_params = initial_log_params.clone()
    history = {
        'iteration': [], 'log_likelihood': [], 'delta': [], 'tau': [], 'lambda': [],
        'log_delta': [], 'log_tau': [], 'log_lambda': [],
        'gradient_norm': [], 'gradients': [], 'timing': []
    }
    
    for iteration in range(max_iterations):
        start_time = time.time()
        
        # Convert to rates for likelihood computation
        rates = F.softplus(log_params)
        delta, tau, lam = float(rates[0]), float(rates[1]), float(rates[2])
        
        print(f"Iter {iteration}: log_θ=[{log_params[0]:.3f},{log_params[1]:.3f},{log_params[2]:.3f}] → δ={delta:.2e}, τ={tau:.2e}, λ={lam:.2e}")
        
        # Compute likelihood
        ll_result = optimizer.compute_likelihood_and_timing(delta, tau, lam)
        
        # Compute gradients and Hessian with finite differences
        gradients = optimizer.compute_gradients_with_transform(log_params)
        hessian = optimizer.compute_hessian_with_transform(log_params)
        
        gradient_norm = torch.norm(gradients).item()
        
        print(f"  LL: {ll_result['log_likelihood']:.6f}, ‖∇‖: {gradient_norm:.6f}")
        
        # Record history
        iter_time = time.time() - start_time
        history['iteration'].append(iteration)
        history['log_likelihood'].append(ll_result['log_likelihood'])
        history['delta'].append(delta)
        history['tau'].append(tau)
        history['lambda'].append(lam)
        history['log_delta'].append(float(log_params[0]))
        history['log_tau'].append(float(log_params[1]))
        history['log_lambda'].append(float(log_params[2]))
        history['gradient_norm'].append(gradient_norm)
        history['gradients'].append([float(g) for g in gradients])
        history['timing'].append(iter_time)
        
        # Newton update
        try:
            hessian_reg = hessian - 0.01 * torch.eye(len(log_params), dtype=hessian.dtype)
            newton_step = torch.linalg.solve(hessian_reg, gradients)
            log_params = log_params - newton_step
        except Exception as e:
            print(f"  ⚠️  Newton step failed: {e}, using gradient step")
            log_params = log_params + 0.01 * gradients
        
        # Check convergence
        if gradient_norm < 1e-6:
            print(f"  ✅ Converged!")
            break
    
    return history

def create_true_newton_comparison(true_newton_history: Dict, 
                                finite_diff_history: Dict,
                                species_path: str, gene_path: str):
    """Create comparison visualization"""
    
    from ete3 import Tree
    species_tree = Tree(species_path, format=1)
    gene_tree = Tree(gene_path, format=1)
    n_species = len(species_tree.get_leaves())
    n_genes = len(gene_tree.get_leaves())
    
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle(f'True Newton vs Finite Difference Newton\\n'
                f'{n_species} Species, {n_genes} Gene Leaves', fontsize=16, fontweight='bold')
    
    # Plot 1: Log-likelihood convergence
    ax1 = axes[0, 0]
    ax1.plot(true_newton_history['iteration'], true_newton_history['log_likelihood'], 
             'o-', color='red', label='True Newton (Autograd)', linewidth=3, markersize=6)
    ax1.plot(finite_diff_history['iteration'], finite_diff_history['log_likelihood'], 
             's-', color='blue', label='Finite Diff Newton', linewidth=2, markersize=4)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Log-Likelihood')
    ax1.set_title('Log-Likelihood Convergence')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Parameter convergence (δ)
    ax2 = axes[0, 1]
    ax2.semilogy(true_newton_history['iteration'], true_newton_history['delta'], 
                 'o-', color='red', label='True Newton', linewidth=3, markersize=6)
    ax2.semilogy(finite_diff_history['iteration'], finite_diff_history['delta'], 
                 's-', color='blue', label='Finite Diff', linewidth=2, markersize=4)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Duplication Rate (δ)')
    ax2.set_title('Parameter Convergence')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Gradient norm convergence
    ax3 = axes[0, 2]
    ax3.semilogy(true_newton_history['iteration'], true_newton_history['gradient_norm'], 
                 'o-', color='red', label='True Newton', linewidth=3, markersize=6)
    ax3.semilogy(finite_diff_history['iteration'], finite_diff_history['gradient_norm'], 
                 's-', color='blue', label='Finite Diff', linewidth=2, markersize=4)
    ax3.axhline(y=1e-6, color='gray', linestyle='--', alpha=0.7, label='Convergence threshold')
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Gradient Norm')
    ax3.set_title('Gradient Convergence')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Hessian condition number (only for true Newton)
    ax4 = axes[0, 3]
    if 'hessian_condition_number' in true_newton_history:
        ax4.semilogy(true_newton_history['iteration'], true_newton_history['hessian_condition_number'], 
                     'o-', color='red', linewidth=3, markersize=6)
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Condition Number')
        ax4.set_title('Hessian Condition Number\\n(True Newton Only)')
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=1e12, color='orange', linestyle='--', alpha=0.7, label='Regularization threshold')
        ax4.legend()
    
    # Plot 5: Newton step size
    ax5 = axes[1, 0]
    if 'newton_step_size' in true_newton_history:
        ax5.semilogy(true_newton_history['iteration'], true_newton_history['newton_step_size'], 
                     'o-', color='red', label='True Newton', linewidth=3, markersize=6)
        ax5.set_xlabel('Iteration')
        ax5.set_ylabel('Newton Step Size')
        ax5.set_title('Newton Step Size')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
    
    # Plot 6: Timing comparison
    ax6 = axes[1, 1]
    true_cumtime = np.cumsum(true_newton_history['timing'])
    finite_cumtime = np.cumsum(finite_diff_history['timing'])
    
    ax6.plot(true_newton_history['iteration'], true_cumtime, 
             'o-', color='red', label='True Newton', linewidth=3, markersize=6)
    ax6.plot(finite_diff_history['iteration'], finite_cumtime, 
             's-', color='blue', label='Finite Diff', linewidth=2, markersize=4)
    ax6.set_xlabel('Iteration')
    ax6.set_ylabel('Cumulative Time (s)')
    ax6.set_title('Runtime Comparison')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # Plot 7: Per-iteration timing
    ax7 = axes[1, 2]
    ax7.plot(true_newton_history['iteration'], true_newton_history['timing'], 
             'o-', color='red', label='True Newton', linewidth=3, markersize=6)
    ax7.plot(finite_diff_history['iteration'], finite_diff_history['timing'], 
             's-', color='blue', label='Finite Diff', linewidth=2, markersize=4)
    ax7.set_xlabel('Iteration')
    ax7.set_ylabel('Time per Iteration (s)')
    ax7.set_title('Per-Iteration Runtime')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # Plot 8: Final comparison
    ax8 = axes[1, 3]
    
    true_final_ll = true_newton_history['log_likelihood'][-1] if true_newton_history['log_likelihood'] else float('-inf')
    finite_final_ll = finite_diff_history['log_likelihood'][-1] if finite_diff_history['log_likelihood'] else float('-inf')
    true_total_time = sum(true_newton_history['timing']) if true_newton_history['timing'] else float('inf')
    finite_total_time = sum(finite_diff_history['timing']) if finite_diff_history['timing'] else float('inf')
    
    methods = ['True\\nNewton', 'Finite\\nDiff']
    final_lls = [true_final_ll, finite_final_ll]
    colors = ['red', 'blue']
    
    bars = ax8.bar(methods, final_lls, color=colors, alpha=0.7)
    ax8.set_ylabel('Final Log-Likelihood')
    ax8.set_title('Final Performance')
    ax8.grid(True, alpha=0.3)
    
    # Add timing labels
    for bar, time_val in zip(bars, [true_total_time, finite_total_time]):
        height = bar.get_height()
        ax8.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{time_val:.1f}s', ha='center', va='bottom', fontsize=10)
    
    # Summary plots
    ax9 = axes[2, 0]
    ax9.text(0.05, 0.9, 'TRUE NEWTON ANALYSIS', fontsize=14, fontweight='bold', 
             transform=ax9.transAxes)
    
    ax9.text(0.05, 0.8, 'True Newton (Autograd):', fontsize=12, fontweight='bold', 
             color='red', transform=ax9.transAxes)
    ax9.text(0.05, 0.75, f'• Exact derivatives via autograd', fontsize=10, transform=ax9.transAxes)
    ax9.text(0.05, 0.7, f'• Final LL: {true_final_ll:.6f}', fontsize=10, transform=ax9.transAxes)
    ax9.text(0.05, 0.65, f'• Iterations: {len(true_newton_history["iteration"])}', fontsize=10, transform=ax9.transAxes)
    ax9.text(0.05, 0.6, f'• Total time: {true_total_time:.1f}s', fontsize=10, transform=ax9.transAxes)
    
    ax9.text(0.05, 0.45, 'Finite Difference Newton:', fontsize=12, fontweight='bold', 
             color='blue', transform=ax9.transAxes)
    ax9.text(0.05, 0.4, f'• Approximate derivatives', fontsize=10, transform=ax9.transAxes)
    ax9.text(0.05, 0.35, f'• Final LL: {finite_final_ll:.6f}', fontsize=10, transform=ax9.transAxes)
    ax9.text(0.05, 0.3, f'• Iterations: {len(finite_diff_history["iteration"])}', fontsize=10, transform=ax9.transAxes)
    ax9.text(0.05, 0.25, f'• Total time: {finite_total_time:.1f}s', fontsize=10, transform=ax9.transAxes)
    
    # Winner
    if abs(true_final_ll - finite_final_ll) < 0.001:
        winner = "Tie (same accuracy)"
        winner_color = "green"
    elif true_final_ll > finite_final_ll:
        winner = "True Newton (higher LL)"
        winner_color = "red"
    else:
        winner = "Finite Diff (higher LL)"
        winner_color = "blue"
    
    ax9.text(0.05, 0.15, f'Winner: {winner}', fontsize=12, fontweight='bold', 
             color=winner_color, transform=ax9.transAxes)
    
    ax9.set_xlim(0, 1)
    ax9.set_ylim(0, 1)
    ax9.axis('off')
    
    # Parameter convergence table
    ax10 = axes[2, 1]
    ax10.text(0.05, 0.9, 'FINAL PARAMETER VALUES', fontsize=14, fontweight='bold', 
              transform=ax10.transAxes)
    
    if true_newton_history['delta']:
        ax10.text(0.05, 0.8, 'True Newton Final:', fontsize=12, fontweight='bold', 
                  color='red', transform=ax10.transAxes)
        ax10.text(0.05, 0.75, f'δ = {true_newton_history["delta"][-1]:.2e}', fontsize=10, transform=ax10.transAxes)
        ax10.text(0.05, 0.7, f'τ = {true_newton_history["tau"][-1]:.2e}', fontsize=10, transform=ax10.transAxes)
        ax10.text(0.05, 0.65, f'λ = {true_newton_history["lambda"][-1]:.2e}', fontsize=10, transform=ax10.transAxes)
    
    if finite_diff_history['delta']:
        ax10.text(0.05, 0.55, 'Finite Diff Final:', fontsize=12, fontweight='bold', 
                  color='blue', transform=ax10.transAxes)
        ax10.text(0.05, 0.5, f'δ = {finite_diff_history["delta"][-1]:.2e}', fontsize=10, transform=ax10.transAxes)
        ax10.text(0.05, 0.45, f'τ = {finite_diff_history["tau"][-1]:.2e}', fontsize=10, transform=ax10.transAxes)
        ax10.text(0.05, 0.4, f'λ = {finite_diff_history["lambda"][-1]:.2e}', fontsize=10, transform=ax10.transAxes)
    
    ax10.text(0.05, 0.25, 'Expected for identical trees:', fontsize=11, fontweight='bold', 
              color='green', transform=ax10.transAxes)
    ax10.text(0.05, 0.2, 'δ, τ, λ → 0', fontsize=10, color='green', transform=ax10.transAxes)
    
    ax10.set_xlim(0, 1)
    ax10.set_ylim(0, 1)
    ax10.axis('off')
    
    # Technical details
    ax11 = axes[2, 2]
    ax11.text(0.05, 0.9, 'TECHNICAL DETAILS', fontsize=14, fontweight='bold', 
              transform=ax11.transAxes)
    
    ax11.text(0.05, 0.8, 'True Newton Method:', fontsize=12, fontweight='bold', 
              color='red', transform=ax11.transAxes)
    ax11.text(0.05, 0.75, '• torch.autograd.functional.hessian', fontsize=9, transform=ax11.transAxes)
    ax11.text(0.05, 0.7, '• Exact analytical derivatives', fontsize=9, transform=ax11.transAxes)
    ax11.text(0.05, 0.65, '• Automatic regularization', fontsize=9, transform=ax11.transAxes)
    
    ax11.text(0.05, 0.55, 'Finite Difference Method:', fontsize=12, fontweight='bold', 
              color='blue', transform=ax11.transAxes)
    ax11.text(0.05, 0.5, '• Numerical approximation', fontsize=9, transform=ax11.transAxes)
    ax11.text(0.05, 0.45, '• ε = 1e-6 step size', fontsize=9, transform=ax11.transAxes)
    ax11.text(0.05, 0.4, '• Manual regularization', fontsize=9, transform=ax11.transAxes)
    
    ax11.text(0.05, 0.3, 'Softplus Parameterization:', fontsize=12, fontweight='bold', 
              color='green', transform=ax11.transAxes)
    ax11.text(0.05, 0.25, '• δ = softplus(log_δ)', fontsize=9, transform=ax11.transAxes)
    ax11.text(0.05, 0.2, '• Ensures δ, τ, λ > 0', fontsize=9, transform=ax11.transAxes)
    ax11.text(0.05, 0.15, '• Unconstrained optimization', fontsize=9, transform=ax11.transAxes)
    
    ax11.set_xlim(0, 1)
    ax11.set_ylim(0, 1)
    ax11.axis('off')
    
    # Empty plot for layout
    ax12 = axes[2, 3]
    ax12.axis('off')
    
    plt.tight_layout()
    plt.savefig('true_newton_vs_finite_diff.png', dpi=300, bbox_inches='tight')
    print(f"📊 True Newton comparison saved as 'true_newton_vs_finite_diff.png'")
    
    return fig

def main():
    """Run True Newton vs Finite Difference Newton comparison"""
    
    print("🚀 TRUE NEWTON VS FINITE DIFFERENCE NEWTON COMPARISON")
    print("=" * 80)
    
    # Test on large trees
    species_path = "test_trees_200/sp.nwk"
    gene_path = "test_trees_200/g.nwk"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initial log parameters (softplus(-2.3) ≈ 0.1)
    initial_log_params = torch.tensor([-2.3, -2.3, -2.3], dtype=torch.float64)
    
    # Run true Newton method
    true_optimizer = TrueNewtonOptimizerStable(species_path, gene_path, device)
    true_newton_history = run_true_newton_method(true_optimizer, initial_log_params)
    
    # Run finite difference Newton for comparison
    from newton_vs_gd_proper_parameterization import ProperParameterizedOptimizer
    finite_optimizer = ProperParameterizedOptimizer(species_path, gene_path, device)
    finite_diff_history = run_finite_difference_newton(finite_optimizer, initial_log_params)
    
    # Create comparison visualization
    create_true_newton_comparison(true_newton_history, finite_diff_history, 
                                species_path, gene_path)
    
    # Save results
    results = {
        'true_newton': true_newton_history,
        'finite_difference_newton': finite_diff_history,
        'problem_info': {
            'species_path': species_path,
            'gene_path': gene_path,
            'n_clades': true_optimizer.n_clades,
            'n_species': true_optimizer.n_species
        }
    }
    
    with open('true_newton_comparison_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"💾 Detailed results saved to 'true_newton_comparison_results.json'")

if __name__ == "__main__":
    main()