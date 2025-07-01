#!/usr/bin/env python3
"""
Newton vs Gradient Descent with Proper Softplus Parameterization
===============================================================

Compares Newton's method vs Gradient Descent for phylogenetic parameter optimization
using proper softplus parameterization: δ = softplus(log_δ), etc.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import time
import json
from typing import Dict, List, Tuple
from realistic_optimization_comparison import RealisticOptimizer

class ProperParameterizedOptimizer(RealisticOptimizer):
    """Optimizer with proper softplus parameterization"""
    
    def __init__(self, species_path: str, gene_path: str, device: str = "cpu"):
        super().__init__(species_path, gene_path, device)
        
    def params_to_rates(self, log_params: torch.Tensor) -> Tuple[float, float, float]:
        """Convert log parameters to rates using softplus"""
        rates = F.softplus(log_params)
        return float(rates[0]), float(rates[1]), float(rates[2])
    
    def compute_gradients_with_transform(self, log_params: torch.Tensor, 
                                       epsilon: float = 1e-6) -> torch.Tensor:
        """Compute gradients with respect to log parameters"""
        
        # Convert to rates
        delta, tau, lam = self.params_to_rates(log_params)
        
        # Compute raw gradients with respect to rates
        gradients_raw, _ = self.compute_gradients_realistic(delta, tau, lam, epsilon)
        
        # Transform gradients: ∂LL/∂log_θ = ∂LL/∂θ × ∂θ/∂log_θ = ∂LL/∂θ × sigmoid(log_θ)
        sigmoid_vals = torch.sigmoid(log_params)
        gradients_log = gradients_raw * sigmoid_vals
        
        return gradients_log
    
    def compute_hessian_with_transform(self, log_params: torch.Tensor,
                                     epsilon: float = 1e-6) -> torch.Tensor:
        """Compute Hessian matrix with respect to log parameters"""
        
        n_params = len(log_params)
        hessian = torch.zeros((n_params, n_params), dtype=self.dtype)
        
        # Save current state
        E_saved = self.E.clone() if self.E is not None else None
        Pi_saved = self.Pi.clone() if self.Pi is not None else None
        
        # Compute base gradients
        base_grad = self.compute_gradients_with_transform(log_params, epsilon)
        
        for i in range(n_params):
            # Positive perturbation
            log_params_pos = log_params.clone()
            log_params_pos[i] += epsilon
            
            # Restore state
            if E_saved is not None:
                self.E = E_saved.clone()
            if Pi_saved is not None:
                self.Pi = Pi_saved.clone()
            
            grad_pos = self.compute_gradients_with_transform(log_params_pos, epsilon)
            
            # Negative perturbation
            log_params_neg = log_params.clone()
            log_params_neg[i] -= epsilon
            
            # Restore state
            if E_saved is not None:
                self.E = E_saved.clone()
            if Pi_saved is not None:
                self.Pi = Pi_saved.clone()
            
            grad_neg = self.compute_gradients_with_transform(log_params_neg, epsilon)
            
            # Central difference for Hessian column
            hessian[:, i] = (grad_pos - grad_neg) / (2 * epsilon)
        
        # Restore original state
        if E_saved is not None:
            self.E = E_saved
        if Pi_saved is not None:
            self.Pi = Pi_saved
        
        return hessian

def run_gradient_descent(optimizer: ProperParameterizedOptimizer, 
                        initial_log_params: torch.Tensor,
                        learning_rate: float = 0.01, 
                        max_iterations: int = 20) -> Dict:
    """Run gradient descent with proper parameterization"""
    
    print(f"🔽 GRADIENT DESCENT (lr={learning_rate})")
    print("-" * 50)
    
    log_params = initial_log_params.clone()
    history = {
        'iteration': [], 'log_likelihood': [], 'delta': [], 'tau': [], 'lambda': [],
        'log_delta': [], 'log_tau': [], 'log_lambda': [],
        'gradient_norm': [], 'gradients': [], 'timing': []
    }
    
    for iteration in range(max_iterations):
        start_time = time.time()
        
        # Convert to rates for likelihood computation
        delta, tau, lam = optimizer.params_to_rates(log_params)
        
        print(f"Iter {iteration}: log_θ=[{log_params[0]:.3f},{log_params[1]:.3f},{log_params[2]:.3f}] → δ={delta:.2e}, τ={tau:.2e}, λ={lam:.2e}")
        
        # Compute likelihood
        ll_result = optimizer.compute_likelihood_and_timing(delta, tau, lam)
        
        # Compute gradients with respect to log parameters
        gradients = optimizer.compute_gradients_with_transform(log_params)
        gradient_norm = torch.norm(gradients).item()
        
        print(f"  LL: {ll_result['log_likelihood']:.6f}, ‖∇‖: {gradient_norm:.3f}")
        
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
        
        # Update log parameters
        log_params = log_params + learning_rate * gradients
        
        # Check convergence
        if gradient_norm < 1e-4:
            print(f"  ✅ Converged!")
            break
    
    return history

def run_newton_method(optimizer: ProperParameterizedOptimizer,
                     initial_log_params: torch.Tensor,
                     max_iterations: int = 20) -> Dict:
    """Run Newton's method with proper parameterization"""
    
    print(f"\n🔺 NEWTON'S METHOD")
    print("-" * 50)
    
    log_params = initial_log_params.clone()
    history = {
        'iteration': [], 'log_likelihood': [], 'delta': [], 'tau': [], 'lambda': [],
        'log_delta': [], 'log_tau': [], 'log_lambda': [],
        'gradient_norm': [], 'gradients': [], 'timing': []
    }
    
    for iteration in range(max_iterations):
        start_time = time.time()
        
        # Convert to rates for likelihood computation
        delta, tau, lam = optimizer.params_to_rates(log_params)
        
        print(f"Iter {iteration}: log_θ=[{log_params[0]:.3f},{log_params[1]:.3f},{log_params[2]:.3f}] → δ={delta:.2e}, τ={tau:.2e}, λ={lam:.2e}")
        
        # Compute likelihood
        ll_result = optimizer.compute_likelihood_and_timing(delta, tau, lam)
        
        # Compute gradients and Hessian with respect to log parameters
        gradients = optimizer.compute_gradients_with_transform(log_params)
        hessian = optimizer.compute_hessian_with_transform(log_params)
        
        gradient_norm = torch.norm(gradients).item()
        
        print(f"  LL: {ll_result['log_likelihood']:.6f}, ‖∇‖: {gradient_norm:.3f}")
        
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
        
        # Newton update: θ ← θ - H⁻¹∇
        try:
            # Add regularization for numerical stability
            hessian_reg = hessian - 0.01 * torch.eye(len(log_params), dtype=hessian.dtype)
            newton_step = torch.linalg.solve(hessian_reg, gradients)
            log_params = log_params - newton_step
            print(f"  Newton step: {[float(s) for s in newton_step]}")
        except Exception as e:
            print(f"  ⚠️  Newton step failed: {e}, using gradient step")
            log_params = log_params + 0.01 * gradients
        
        # Check convergence
        if gradient_norm < 1e-4:
            print(f"  ✅ Converged!")
            break
    
    return history

def create_comparison_visualization(gd_history: Dict, newton_history: Dict,
                                  species_path: str, gene_path: str):
    """Create comprehensive comparison visualization"""
    
    from ete3 import Tree
    species_tree = Tree(species_path, format=1)
    gene_tree = Tree(gene_path, format=1)
    n_species = len(species_tree.get_leaves())
    n_genes = len(gene_tree.get_leaves())
    
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle(f'Newton vs Gradient Descent with Proper Softplus Parameterization\\n'
                f'{n_species} Species, {n_genes} Gene Leaves', fontsize=16, fontweight='bold')
    
    # Plot 1: Log-likelihood convergence
    ax1 = axes[0, 0]
    ax1.plot(gd_history['iteration'], gd_history['log_likelihood'], 'o-', 
             color='blue', label='Gradient Descent', linewidth=2, markersize=4)
    ax1.plot(newton_history['iteration'], newton_history['log_likelihood'], 's-', 
             color='red', label='Newton Method', linewidth=2, markersize=4)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Log-Likelihood')
    ax1.set_title('Log-Likelihood Convergence')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Parameter convergence (δ)
    ax2 = axes[0, 1]
    ax2.semilogy(gd_history['iteration'], gd_history['delta'], 'o-', 
                 color='blue', label='GD: δ', linewidth=2, markersize=4)
    ax2.semilogy(newton_history['iteration'], newton_history['delta'], 's-', 
                 color='red', label='Newton: δ', linewidth=2, markersize=4)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Duplication Rate (δ)')
    ax2.set_title('Duplication Parameter Convergence')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Parameter convergence (τ)
    ax3 = axes[0, 2]
    ax3.semilogy(gd_history['iteration'], gd_history['tau'], 'o-', 
                 color='blue', label='GD: τ', linewidth=2, markersize=4)
    ax3.semilogy(newton_history['iteration'], newton_history['tau'], 's-', 
                 color='red', label='Newton: τ', linewidth=2, markersize=4)
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Transfer Rate (τ)')
    ax3.set_title('Transfer Parameter Convergence')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Parameter convergence (λ)
    ax4 = axes[0, 3]
    ax4.semilogy(gd_history['iteration'], gd_history['lambda'], 'o-', 
                 color='blue', label='GD: λ', linewidth=2, markersize=4)
    ax4.semilogy(newton_history['iteration'], newton_history['lambda'], 's-', 
                 color='red', label='Newton: λ', linewidth=2, markersize=4)
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Loss Rate (λ)')
    ax4.set_title('Loss Parameter Convergence')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Log parameter convergence
    ax5 = axes[1, 0]
    ax5.plot(gd_history['iteration'], gd_history['log_delta'], 'o-', 
             color='lightblue', label='GD: log δ', linewidth=2, markersize=4)
    ax5.plot(gd_history['iteration'], gd_history['log_tau'], 's-', 
             color='lightgreen', label='GD: log τ', linewidth=2, markersize=4)
    ax5.plot(gd_history['iteration'], gd_history['log_lambda'], '^-', 
             color='lightcoral', label='GD: log λ', linewidth=2, markersize=4)
    ax5.plot(newton_history['iteration'], newton_history['log_delta'], 'o-', 
             color='darkblue', label='Newton: log δ', linewidth=2, markersize=4)
    ax5.plot(newton_history['iteration'], newton_history['log_tau'], 's-', 
             color='darkgreen', label='Newton: log τ', linewidth=2, markersize=4)
    ax5.plot(newton_history['iteration'], newton_history['log_lambda'], '^-', 
             color='darkred', label='Newton: log λ', linewidth=2, markersize=4)
    ax5.set_xlabel('Iteration')
    ax5.set_ylabel('Log Parameter Value')
    ax5.set_title('Log Parameter Convergence')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Gradient norm convergence
    ax6 = axes[1, 1]
    ax6.semilogy(gd_history['iteration'], gd_history['gradient_norm'], 'o-', 
                 color='blue', label='Gradient Descent', linewidth=2, markersize=4)
    ax6.semilogy(newton_history['iteration'], newton_history['gradient_norm'], 's-', 
                 color='red', label='Newton Method', linewidth=2, markersize=4)
    ax6.axhline(y=1e-4, color='gray', linestyle='--', alpha=0.7, label='Convergence threshold')
    ax6.set_xlabel('Iteration')
    ax6.set_ylabel('Gradient Norm')
    ax6.set_title('Gradient Norm Convergence')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # Plot 7: Timing comparison
    ax7 = axes[1, 2]
    gd_cumtime = np.cumsum(gd_history['timing'])
    newton_cumtime = np.cumsum(newton_history['timing'])
    
    ax7.plot(gd_history['iteration'], gd_cumtime, 'o-', 
             color='blue', label='Gradient Descent', linewidth=2, markersize=4)
    ax7.plot(newton_history['iteration'], newton_cumtime, 's-', 
             color='red', label='Newton Method', linewidth=2, markersize=4)
    ax7.set_xlabel('Iteration')
    ax7.set_ylabel('Cumulative Time (s)')
    ax7.set_title('Cumulative Runtime Comparison')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # Plot 8: Per-iteration timing
    ax8 = axes[1, 3]
    ax8.plot(gd_history['iteration'], gd_history['timing'], 'o-', 
             color='blue', label='Gradient Descent', linewidth=2, markersize=4)
    ax8.plot(newton_history['iteration'], newton_history['timing'], 's-', 
             color='red', label='Newton Method', linewidth=2, markersize=4)
    ax8.set_xlabel('Iteration')
    ax8.set_ylabel('Time per Iteration (s)')
    ax8.set_title('Per-Iteration Runtime')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # Plot 9: Parameter space trajectory (δ vs τ)
    ax9 = axes[2, 0]
    ax9.loglog(gd_history['delta'], gd_history['tau'], 'o-', 
               color='blue', label='Gradient Descent', linewidth=2, markersize=4, alpha=0.7)
    ax9.loglog(newton_history['delta'], newton_history['tau'], 's-', 
               color='red', label='Newton Method', linewidth=2, markersize=4, alpha=0.7)
    ax9.scatter(gd_history['delta'][0], gd_history['tau'][0], color='green', s=100, 
                marker='*', label='Start', zorder=5)
    ax9.set_xlabel('Duplication Rate (δ)')
    ax9.set_ylabel('Transfer Rate (τ)')
    ax9.set_title('Parameter Space Trajectory')
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    
    # Plot 10: Efficiency comparison
    ax10 = axes[2, 1]
    
    # Final results comparison
    gd_final_ll = gd_history['log_likelihood'][-1] if gd_history['log_likelihood'] else float('-inf')
    newton_final_ll = newton_history['log_likelihood'][-1] if newton_history['log_likelihood'] else float('-inf')
    gd_total_time = sum(gd_history['timing']) if gd_history['timing'] else float('inf')
    newton_total_time = sum(newton_history['timing']) if newton_history['timing'] else float('inf')
    
    methods = ['Gradient\\nDescent', 'Newton\\nMethod']
    final_lls = [gd_final_ll, newton_final_ll]
    total_times = [gd_total_time, newton_total_time]
    colors = ['blue', 'red']
    
    bars = ax10.bar(methods, final_lls, color=colors, alpha=0.7)
    ax10.set_ylabel('Final Log-Likelihood')
    ax10.set_title('Final Performance Comparison')
    ax10.grid(True, alpha=0.3)
    
    # Add timing labels on bars
    for bar, time_val in zip(bars, total_times):
        height = bar.get_height()
        ax10.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{time_val:.2f}s', ha='center', va='bottom', fontsize=10)
    
    # Plot 11-12: Summary
    ax11 = axes[2, 2]
    ax11.text(0.05, 0.9, 'OPTIMIZATION COMPARISON SUMMARY', fontsize=14, fontweight='bold', 
              transform=ax11.transAxes)
    
    # Performance summary
    ax11.text(0.05, 0.8, 'Gradient Descent:', fontsize=12, fontweight='bold', 
              color='blue', transform=ax11.transAxes)
    ax11.text(0.05, 0.75, f'• Iterations: {len(gd_history["iteration"])}', fontsize=10, transform=ax11.transAxes)
    ax11.text(0.05, 0.7, f'• Final LL: {gd_final_ll:.6f}', fontsize=10, transform=ax11.transAxes)
    ax11.text(0.05, 0.65, f'• Total time: {gd_total_time:.2f}s', fontsize=10, transform=ax11.transAxes)
    
    ax11.text(0.05, 0.55, 'Newton Method:', fontsize=12, fontweight='bold', 
              color='red', transform=ax11.transAxes)
    ax11.text(0.05, 0.5, f'• Iterations: {len(newton_history["iteration"])}', fontsize=10, transform=ax11.transAxes)
    ax11.text(0.05, 0.45, f'• Final LL: {newton_final_ll:.6f}', fontsize=10, transform=ax11.transAxes)
    ax11.text(0.05, 0.4, f'• Total time: {newton_total_time:.2f}s', fontsize=10, transform=ax11.transAxes)
    
    # Winner determination
    if abs(gd_final_ll - newton_final_ll) < 0.001:
        if gd_total_time < newton_total_time:
            winner = "Gradient Descent (faster)"
            winner_color = "blue"
        else:
            winner = "Newton Method (faster)"
            winner_color = "red"
    elif gd_final_ll > newton_final_ll:
        winner = "Gradient Descent (higher LL)"
        winner_color = "blue"
    else:
        winner = "Newton Method (higher LL)"
        winner_color = "red"
    
    ax11.text(0.05, 0.25, f'Winner: {winner}', fontsize=12, fontweight='bold', 
              color=winner_color, transform=ax11.transAxes)
    
    ax11.set_xlim(0, 1)
    ax11.set_ylim(0, 1)
    ax11.axis('off')
    
    # Final convergence values
    ax12 = axes[2, 3]
    ax12.text(0.05, 0.9, 'FINAL CONVERGENCE VALUES', fontsize=14, fontweight='bold', 
              transform=ax12.transAxes)
    
    if gd_history['delta']:
        ax12.text(0.05, 0.8, 'Gradient Descent Final:', fontsize=12, fontweight='bold', 
                  color='blue', transform=ax12.transAxes)
        ax12.text(0.05, 0.75, f'δ = {gd_history["delta"][-1]:.2e}', fontsize=10, transform=ax12.transAxes)
        ax12.text(0.05, 0.7, f'τ = {gd_history["tau"][-1]:.2e}', fontsize=10, transform=ax12.transAxes)
        ax12.text(0.05, 0.65, f'λ = {gd_history["lambda"][-1]:.2e}', fontsize=10, transform=ax12.transAxes)
    
    if newton_history['delta']:
        ax12.text(0.05, 0.55, 'Newton Method Final:', fontsize=12, fontweight='bold', 
                  color='red', transform=ax12.transAxes)
        ax12.text(0.05, 0.5, f'δ = {newton_history["delta"][-1]:.2e}', fontsize=10, transform=ax12.transAxes)
        ax12.text(0.05, 0.45, f'τ = {newton_history["tau"][-1]:.2e}', fontsize=10, transform=ax12.transAxes)
        ax12.text(0.05, 0.4, f'λ = {newton_history["lambda"][-1]:.2e}', fontsize=10, transform=ax12.transAxes)
    
    # Theoretical optimal for identical trees
    ax12.text(0.05, 0.25, 'Expected for identical trees:', fontsize=11, fontweight='bold', 
              color='green', transform=ax12.transAxes)
    ax12.text(0.05, 0.2, 'δ, τ, λ → 0 (pure speciation)', fontsize=10, 
              color='green', transform=ax12.transAxes)
    
    ax12.set_xlim(0, 1)
    ax12.set_ylim(0, 1)
    ax12.axis('off')
    
    plt.tight_layout()
    plt.savefig('newton_vs_gd_proper_parameterization.png', dpi=300, bbox_inches='tight')
    print(f"📊 Comparison saved as 'newton_vs_gd_proper_parameterization.png'")
    
    return fig

def main():
    """Run Newton vs Gradient Descent comparison with proper parameterization"""
    
    print("🚀 NEWTON VS GRADIENT DESCENT WITH PROPER SOFTPLUS PARAMETERIZATION")
    print("=" * 80)
    
    # Test on large trees
    species_path = "test_trees_200/sp.nwk"
    gene_path = "test_trees_200/g.nwk"
    
    # Initialize optimizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    optimizer = ProperParameterizedOptimizer(species_path, gene_path, device)
    
    # Initial log parameters (will be transformed via softplus)
    # log(0.1) ≈ -2.3, so softplus(-2.3) ≈ 0.1
    initial_log_params = torch.tensor([-2.3, -2.3, -2.3], dtype=torch.float64)
    
    print(f"Initial log parameters: {initial_log_params}")
    delta, tau, lam = optimizer.params_to_rates(initial_log_params)
    print(f"Initial rates: δ={delta:.3f}, τ={tau:.3f}, λ={lam:.3f}")
    
    # Run both methods
    gd_history = run_gradient_descent(optimizer, initial_log_params, learning_rate=0.1)
    
    # Reset optimizer state for fair comparison
    optimizer.E = None
    optimizer.Pi = None
    
    newton_history = run_newton_method(optimizer, initial_log_params)
    
    # Create comparison visualization
    create_comparison_visualization(gd_history, newton_history, species_path, gene_path)
    
    # Save results
    results = {
        'gradient_descent': gd_history,
        'newton_method': newton_history,
        'problem_info': {
            'species_path': species_path,
            'gene_path': gene_path,
            'n_clades': optimizer.n_clades,
            'n_species': optimizer.n_species,
            'matrix_elements': optimizer.n_clades * optimizer.n_species
        }
    }
    
    with open('newton_vs_gd_proper_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"💾 Detailed results saved to 'newton_vs_gd_proper_results.json'")

if __name__ == "__main__":
    main()