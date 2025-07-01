#!/usr/bin/env python3
"""
Newton's Method vs Gradient Descent Comparison for test_trees_200
================================================================

Creates a comprehensive comparison visualization between Newton's method and 
gradient descent optimization for large phylogenetic reconciliation problems.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import json
import time
from typing import Dict, List, Tuple, Optional
from matmul_ale_ccp_optimize_finite_diff import compute_log_likelihood
from ete3 import Tree
# Set style for professional plots
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.grid'] = True

class FiniteDiffOptimizer:
    """Finite difference gradient optimizer for CCP reconciliation"""
    
    def __init__(self, species_path: str, gene_path: str, device: str = "cpu"):
        self.species_path = species_path
        self.gene_path = gene_path
        self.device = device
        self.species_tree = Tree(species_path, format=1)
        self.gene_tree = Tree(gene_path, format=1)
        
        # Initialize parameters
        self.params = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float64, requires_grad=False)
        
        print(f"✅ Finite Diff Optimizer initialized:")
        print(f"   Device: {device}")
        print(f"   Initial params: δ={self.params[0]:.3f}, τ={self.params[1]:.3f}, λ={self.params[2]:.3f}")
    
    def compute_likelihood(self, params: torch.Tensor) -> float:
        """Compute log-likelihood for given parameters"""
        delta, tau, lam = params.tolist()
        
        # Run CCP reconciliation using finite diff module
        ll = compute_log_likelihood(
            self.species_path, self.gene_path,
            delta, tau, lam,
            torch.device(self.device), torch.float64
        )
        
        return float(ll)
    
    def compute_gradients(self, epsilon: float = 1e-6) -> torch.Tensor:
        """Compute gradients using finite differences"""
        base_ll = self.compute_likelihood(self.params)
        gradients = torch.zeros_like(self.params)
        
        for i in range(len(self.params)):
            # Positive perturbation
            params_pos = self.params.clone()
            params_pos[i] += epsilon
            ll_pos = self.compute_likelihood(params_pos)
            
            # Negative perturbation  
            params_neg = self.params.clone()
            params_neg[i] -= epsilon
            ll_neg = self.compute_likelihood(params_neg)
            
            # Central difference
            gradients[i] = (ll_pos - ll_neg) / (2 * epsilon)
        
        return gradients
    
    def compute_hessian_diagonal(self, epsilon: float = 1e-6) -> torch.Tensor:
        """Compute diagonal Hessian approximation using finite differences"""
        base_ll = self.compute_likelihood(self.params)
        hessian_diag = torch.zeros_like(self.params)
        
        for i in range(len(self.params)):
            # Forward difference for second derivative
            params_pos = self.params.clone()
            params_pos[i] += epsilon
            ll_pos = self.compute_likelihood(params_pos)
            
            params_neg = self.params.clone()
            params_neg[i] -= epsilon  
            ll_neg = self.compute_likelihood(params_neg)
            
            # Second derivative approximation
            hessian_diag[i] = (ll_pos - 2*base_ll + ll_neg) / (epsilon**2)
        
        return hessian_diag

def run_gradient_descent_optimization(species_path: str, gene_path: str, 
                                    max_iterations: int = 20,
                                    learning_rate: float = 0.01) -> Dict:
    """Run gradient descent optimization"""
    
    print(f"🚀 GRADIENT DESCENT OPTIMIZATION")
    print(f"=" * 60)
    
    optimizer = FiniteDiffOptimizer(species_path, gene_path)
    
    # Track optimization history
    history = {
        'iteration': [],
        'log_likelihood': [],
        'delta': [],
        'tau': [],
        'lambda': [],
        'gradient_norm': [],
        'iteration_time': [],
        'cumulative_time': []
    }
    
    cumulative_time = 0.0
    
    for iteration in range(max_iterations):
        iter_start = time.time()
        
        # Compute current likelihood
        current_ll = optimizer.compute_likelihood(optimizer.params)
        
        # Compute gradients
        gradients = optimizer.compute_gradients()
        gradient_norm = torch.norm(gradients).item()
        
        # Gradient descent update
        optimizer.params = optimizer.params + learning_rate * gradients
        
        # Ensure parameters stay positive
        optimizer.params = torch.clamp(optimizer.params, min=1e-6)
        
        iter_time = time.time() - iter_start
        cumulative_time += iter_time
        
        # Record history
        history['iteration'].append(iteration)
        history['log_likelihood'].append(current_ll)
        history['delta'].append(optimizer.params[0].item())
        history['tau'].append(optimizer.params[1].item())
        history['lambda'].append(optimizer.params[2].item())
        history['gradient_norm'].append(gradient_norm)
        history['iteration_time'].append(iter_time)
        history['cumulative_time'].append(cumulative_time)
        
        print(f"Iter {iteration:2d}: LL={current_ll:8.4f}, "
              f"δ={optimizer.params[0]:.4f}, τ={optimizer.params[1]:.4f}, λ={optimizer.params[2]:.4f}, "
              f"‖∇‖={gradient_norm:.2e}, time={iter_time:.1f}s")
        
        # Check convergence
        if gradient_norm < 1e-6:
            print(f"✅ Converged! Gradient norm {gradient_norm:.2e} below threshold")
            break
    
    return history

def run_newton_optimization(species_path: str, gene_path: str,
                          max_iterations: int = 20,
                          learning_rate: float = 1.0) -> Dict:
    """Run Newton's method optimization"""
    
    print(f"🚀 NEWTON'S METHOD OPTIMIZATION")
    print(f"=" * 60)
    
    optimizer = FiniteDiffOptimizer(species_path, gene_path)
    
    # Track optimization history
    history = {
        'iteration': [],
        'log_likelihood': [],
        'delta': [],
        'tau': [], 
        'lambda': [],
        'gradient_norm': [],
        'iteration_time': [],
        'cumulative_time': [],
        'll_improvement': []
    }
    
    cumulative_time = 0.0
    previous_ll = None
    
    for iteration in range(max_iterations):
        iter_start = time.time()
        
        # Compute current likelihood
        current_ll = optimizer.compute_likelihood(optimizer.params)
        
        # Compute gradients and Hessian diagonal
        gradients = optimizer.compute_gradients()
        hessian_diag = optimizer.compute_hessian_diagonal()
        
        gradient_norm = torch.norm(gradients).item()
        
        # Newton update with diagonal Hessian approximation
        # Regularize Hessian to avoid division by zero
        hessian_diag_reg = torch.where(torch.abs(hessian_diag) < 1e-8, 
                                     torch.sign(hessian_diag) * 1e-8, 
                                     hessian_diag)
        
        newton_step = gradients / hessian_diag_reg
        optimizer.params = optimizer.params + learning_rate * newton_step
        
        # Ensure parameters stay positive
        optimizer.params = torch.clamp(optimizer.params, min=1e-6)
        
        iter_time = time.time() - iter_start
        cumulative_time += iter_time
        
        # Compute log-likelihood improvement
        ll_improvement = current_ll - previous_ll if previous_ll is not None else 0.0
        
        # Record history
        history['iteration'].append(iteration)
        history['log_likelihood'].append(current_ll)
        history['delta'].append(optimizer.params[0].item())
        history['tau'].append(optimizer.params[1].item())
        history['lambda'].append(optimizer.params[2].item())
        history['gradient_norm'].append(gradient_norm)
        history['iteration_time'].append(iter_time)
        history['cumulative_time'].append(cumulative_time)
        history['ll_improvement'].append(ll_improvement)
        
        print(f"Iter {iteration:2d}: LL={current_ll:8.4f}, "
              f"δ={optimizer.params[0]:.4f}, τ={optimizer.params[1]:.4f}, λ={optimizer.params[2]:.4f}, "
              f"‖∇‖={gradient_norm:.2e}, ΔLL={ll_improvement:.2e}, time={iter_time:.1f}s")
        
        # Check convergence (log-likelihood improvement)
        if previous_ll is not None and abs(ll_improvement) < 1e-8:
            print(f"✅ Converged! LL improvement {abs(ll_improvement):.2e} below threshold")
            break
            
        # Also check gradient norm convergence
        if gradient_norm < 1e-6:
            print(f"✅ Converged! Gradient norm {gradient_norm:.2e} below threshold")
            break
        
        previous_ll = current_ll
    
    return history

def create_comparison_plot(gd_history: Dict, newton_history: Dict, 
                         species_path: str, gene_path: str):
    """Create comprehensive comparison plot"""
    
    # Get tree information for title
    species_tree = Tree(species_path, format=1)
    gene_tree = Tree(gene_path, format=1)
    n_species = len(species_tree.get_leaves())
    n_genes = len(gene_tree.get_leaves())
    
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle(f'Newton vs Gradient Descent: Large Trees (Species: {n_species} leaves, Gene: {n_genes} leaves)', 
                 fontsize=16, fontweight='bold')
    
    # Colors for consistency
    gd_color = '#2E86AB'
    newton_color = '#A23B72'
    
    # Plot 1: Log-likelihood convergence
    ax1 = axes[0, 0]
    ax1.plot(gd_history['iteration'], gd_history['log_likelihood'], 
             'o-', color=gd_color, label='Gradient Descent', linewidth=2, markersize=4)
    ax1.plot(newton_history['iteration'], newton_history['log_likelihood'], 
             's-', color=newton_color, label="Newton's Method", linewidth=2, markersize=4)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Log-Likelihood')
    ax1.set_title('Convergence Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Parameter evolution - Delta
    ax2 = axes[0, 1]
    ax2.plot(gd_history['iteration'], gd_history['delta'], 
             'o-', color=gd_color, label='GD δ', linewidth=2, markersize=4)
    ax2.plot(newton_history['iteration'], newton_history['delta'], 
             's-', color=newton_color, label='Newton δ', linewidth=2, markersize=4)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Duplication Rate (δ)')
    ax2.set_title('Parameter Evolution: δ')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Parameter evolution - Tau
    ax3 = axes[0, 2]
    ax3.plot(gd_history['iteration'], gd_history['tau'], 
             'o-', color=gd_color, label='GD τ', linewidth=2, markersize=4)
    ax3.plot(newton_history['iteration'], newton_history['tau'], 
             's-', color=newton_color, label='Newton τ', linewidth=2, markersize=4)
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Transfer Rate (τ)')
    ax3.set_title('Parameter Evolution: τ')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Parameter evolution - Lambda
    ax4 = axes[0, 3]
    ax4.plot(gd_history['iteration'], gd_history['lambda'], 
             'o-', color=gd_color, label='GD λ', linewidth=2, markersize=4)
    ax4.plot(newton_history['iteration'], newton_history['lambda'], 
             's-', color=newton_color, label='Newton λ', linewidth=2, markersize=4)
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Loss Rate (λ)')
    ax4.set_title('Parameter Evolution: λ')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Gradient norm evolution
    ax5 = axes[1, 0]
    ax5.semilogy(gd_history['iteration'], gd_history['gradient_norm'], 
                 'o-', color=gd_color, label='Gradient Descent', linewidth=2, markersize=4)
    ax5.semilogy(newton_history['iteration'], newton_history['gradient_norm'], 
                 's-', color=newton_color, label="Newton's Method", linewidth=2, markersize=4)
    ax5.set_xlabel('Iteration')
    ax5.set_ylabel('Gradient Norm (log scale)')
    ax5.set_title('Gradient Norm Convergence')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Iteration timing
    ax6 = axes[1, 1]
    ax6.bar(range(len(gd_history['iteration_time'])), gd_history['iteration_time'], 
            alpha=0.7, color=gd_color, label='Gradient Descent', width=0.4)
    newton_x = [x + 0.4 for x in range(len(newton_history['iteration_time']))]
    ax6.bar(newton_x, newton_history['iteration_time'], 
            alpha=0.7, color=newton_color, label="Newton's Method", width=0.4)
    ax6.set_xlabel('Iteration')
    ax6.set_ylabel('Time per Iteration (s)')
    ax6.set_title('Per-Iteration Timing')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # Plot 7: Cumulative time
    ax7 = axes[1, 2]
    ax7.plot(gd_history['iteration'], gd_history['cumulative_time'], 
             'o-', color=gd_color, label='Gradient Descent', linewidth=2, markersize=4)
    ax7.plot(newton_history['iteration'], newton_history['cumulative_time'], 
             's-', color=newton_color, label="Newton's Method", linewidth=2, markersize=4)
    ax7.set_xlabel('Iteration')
    ax7.set_ylabel('Cumulative Time (s)')
    ax7.set_title('Total Time Comparison')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # Plot 8: Log-likelihood improvement (Newton only)
    ax8 = axes[1, 3]
    if len(newton_history['ll_improvement']) > 1:
        ax8.semilogy(newton_history['iteration'][1:], 
                     [abs(x) for x in newton_history['ll_improvement'][1:]], 
                     's-', color=newton_color, label="Newton LL Improvement", linewidth=2, markersize=4)
        ax8.axhline(y=1e-8, color='red', linestyle='--', alpha=0.7, label='Convergence Threshold')
    ax8.set_xlabel('Iteration')
    ax8.set_ylabel('|LL Improvement| (log scale)')
    ax8.set_title('Newton Convergence Criterion')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # Plot 9: Efficiency comparison
    ax9 = axes[2, 0]
    methods = ['Gradient Descent', 'Newton Method']
    total_times = [gd_history['cumulative_time'][-1], newton_history['cumulative_time'][-1]]
    final_lls = [gd_history['log_likelihood'][-1], newton_history['log_likelihood'][-1]]
    
    bars = ax9.bar(methods, total_times, color=[gd_color, newton_color], alpha=0.7)
    ax9.set_ylabel('Total Time to Convergence (s)')
    ax9.set_title('Efficiency Comparison')
    ax9.grid(True, alpha=0.3)
    
    # Add final LL values as text on bars
    for bar, ll in zip(bars, final_lls):
        height = bar.get_height()
        ax9.text(bar.get_x() + bar.get_width()/2., height + 0.01*height,
                f'Final LL: {ll:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Plot 10: Parameter space trajectory
    ax10 = axes[2, 1]
    ax10.plot(gd_history['delta'], gd_history['tau'], 
              'o-', color=gd_color, label='GD Trajectory', linewidth=2, markersize=4, alpha=0.7)
    ax10.plot(newton_history['delta'], newton_history['tau'], 
              's-', color=newton_color, label='Newton Trajectory', linewidth=2, markersize=4, alpha=0.7)
    ax10.scatter(gd_history['delta'][0], gd_history['tau'][0], 
                color='green', s=100, marker='*', label='Start', zorder=5)
    ax10.set_xlabel('Duplication Rate (δ)')
    ax10.set_ylabel('Transfer Rate (τ)')
    ax10.set_title('Parameter Space Trajectory')
    ax10.legend()
    ax10.grid(True, alpha=0.3)
    
    # Plot 11: Performance metrics
    ax11 = axes[2, 2]
    metrics = ['Iterations', 'Avg Time/Iter', 'Final LL']
    gd_metrics = [len(gd_history['iteration']), 
                  np.mean(gd_history['iteration_time']), 
                  gd_history['log_likelihood'][-1]]
    newton_metrics = [len(newton_history['iteration']), 
                     np.mean(newton_history['iteration_time']), 
                     newton_history['log_likelihood'][-1]]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax11.bar(x - width/2, [gd_metrics[0], gd_metrics[1], 0], width, 
                     color=gd_color, alpha=0.7, label='Gradient Descent')
    bars2 = ax11.bar(x + width/2, [newton_metrics[0], newton_metrics[1], 0], width, 
                     color=newton_color, alpha=0.7, label='Newton Method')
    
    ax11.set_xlabel('Metrics')
    ax11.set_ylabel('Values')
    ax11.set_title('Performance Summary')
    ax11.set_xticks(x)
    ax11.set_xticklabels(['Iterations', 'Avg Time/Iter (s)', 'Final LL'])
    ax11.legend()
    ax11.grid(True, alpha=0.3)
    
    # Plot 12: Scaling analysis summary
    ax12 = axes[2, 3]
    ax12.text(0.1, 0.9, 'SCALING ANALYSIS', fontsize=14, fontweight='bold', transform=ax12.transAxes)
    ax12.text(0.1, 0.8, f'Problem Size: {n_species} species, {n_genes} genes', fontsize=12, transform=ax12.transAxes)
    ax12.text(0.1, 0.7, f'Matrix Elements: ~{n_species * n_genes * 27:,}', fontsize=12, transform=ax12.transAxes)
    
    ax12.text(0.1, 0.6, 'GRADIENT DESCENT:', fontsize=12, fontweight='bold', color=gd_color, transform=ax12.transAxes)
    ax12.text(0.1, 0.55, f'• Iterations: {len(gd_history["iteration"])}', fontsize=11, transform=ax12.transAxes)
    ax12.text(0.1, 0.5, f'• Avg time/iter: {np.mean(gd_history["iteration_time"]):.1f}s', fontsize=11, transform=ax12.transAxes)
    ax12.text(0.1, 0.45, f'• Total time: {gd_history["cumulative_time"][-1]:.1f}s', fontsize=11, transform=ax12.transAxes)
    
    ax12.text(0.1, 0.35, "NEWTON'S METHOD:", fontsize=12, fontweight='bold', color=newton_color, transform=ax12.transAxes)
    ax12.text(0.1, 0.3, f'• Iterations: {len(newton_history["iteration"])}', fontsize=11, transform=ax12.transAxes)
    ax12.text(0.1, 0.25, f'• Avg time/iter: {np.mean(newton_history["iteration_time"]):.1f}s', fontsize=11, transform=ax12.transAxes)
    ax12.text(0.1, 0.2, f'• Total time: {newton_history["cumulative_time"][-1]:.1f}s', fontsize=11, transform=ax12.transAxes)
    
    speedup = gd_history["cumulative_time"][-1] / newton_history["cumulative_time"][-1]
    ax12.text(0.1, 0.1, f'SPEEDUP: {speedup:.1f}x faster', fontsize=12, fontweight='bold', 
              color='green', transform=ax12.transAxes)
    
    ax12.set_xlim(0, 1)
    ax12.set_ylim(0, 1)
    ax12.axis('off')
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('newton_vs_gradient_descent_200.png', dpi=300, bbox_inches='tight')
    print(f"📊 Comprehensive comparison plot saved as 'newton_vs_gradient_descent_200.png'")
    
    return fig

def main():
    """Main function to run the complete comparison"""
    
    print("🚀 COMPREHENSIVE TIMING ANALYSIS: Newton vs Gradient Descent (Large Trees)")
    print("=" * 80)
    
    # Test on large trees
    species_path = "test_trees_200/sp.nwk"
    gene_path = "test_trees_200/g.nwk"
    
    print(f"🧪 Testing Large Trees ({species_path})")
    print("-" * 50)
    
    # Run gradient descent optimization
    gd_history = run_gradient_descent_optimization(species_path, gene_path, 
                                                 max_iterations=15, learning_rate=0.01)
    
    print(f"\n")
    
    # Run Newton's method optimization  
    newton_history = run_newton_optimization(species_path, gene_path,
                                           max_iterations=15, learning_rate=1.0)
    
    print(f"\n")
    
    # Create comparison visualization
    create_comparison_plot(gd_history, newton_history, species_path, gene_path)
    
    # Print summary
    print("📈 OPTIMIZATION SUMMARY")
    print("=" * 50)
    print(f"Gradient Descent:")
    print(f"  • Iterations: {len(gd_history['iteration'])}")
    print(f"  • Final LL: {gd_history['log_likelihood'][-1]:.6f}")
    print(f"  • Total time: {gd_history['cumulative_time'][-1]:.1f}s")
    print(f"  • Avg time/iter: {np.mean(gd_history['iteration_time']):.1f}s")
    
    print(f"\nNewton's Method:")
    print(f"  • Iterations: {len(newton_history['iteration'])}")
    print(f"  • Final LL: {newton_history['log_likelihood'][-1]:.6f}")
    print(f"  • Total time: {newton_history['cumulative_time'][-1]:.1f}s")
    print(f"  • Avg time/iter: {np.mean(newton_history['iteration_time']):.1f}s")
    
    speedup = gd_history['cumulative_time'][-1] / newton_history['cumulative_time'][-1]
    print(f"\n🏆 Newton's method is {speedup:.1f}x faster for large phylogenetic trees!")
    
    # Save results
    results = {
        'gradient_descent': gd_history,
        'newton_method': newton_history,
        'summary': {
            'speedup': speedup,
            'gd_total_time': gd_history['cumulative_time'][-1],
            'newton_total_time': newton_history['cumulative_time'][-1],
            'gd_iterations': len(gd_history['iteration']),
            'newton_iterations': len(newton_history['iteration'])
        }
    }
    
    with open('newton_vs_gradient_descent_200_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"💾 Results saved to 'newton_vs_gradient_descent_200_results.json'")

if __name__ == "__main__":
    main()