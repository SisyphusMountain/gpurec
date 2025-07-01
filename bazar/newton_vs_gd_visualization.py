#!/usr/bin/env python3
"""
Comprehensive visualization comparing Newton's method vs Gradient Descent.
Creates detailed plots showing parameter evolution, convergence rates, and efficiency.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import time
from pathlib import Path

from matmul_ale_ccp_optimize_finite_diff import FiniteDiffCCPOptimizer, compute_log_likelihood, softplus_transform, inverse_softplus_transform

def silent_newton_step(species_path, gene_path, log_params, epsilon=1e-6):
    """Newton step computation without verbose output."""
    
    def evaluate_at(params):
        transformed = softplus_transform(params)
        return compute_log_likelihood(species_path, gene_path, 
                                    float(transformed[0]), float(transformed[1]), float(transformed[2]),
                                    torch.device("cpu"), torch.float64)
    
    # Compute gradient
    f0 = evaluate_at(log_params)
    gradient = torch.zeros_like(log_params)
    
    for i in range(len(log_params)):
        params_plus = log_params.clone()
        params_minus = log_params.clone()
        params_plus[i] += epsilon
        params_minus[i] -= epsilon
        
        f_plus = evaluate_at(params_plus)
        f_minus = evaluate_at(params_minus)
        gradient[i] = (f_plus - f_minus) / (2 * epsilon)
    
    # Compute diagonal Hessian
    hessian_diag = torch.zeros_like(log_params)
    
    for i in range(len(log_params)):
        params_plus = log_params.clone()
        params_minus = log_params.clone()
        params_plus[i] += epsilon
        params_minus[i] -= epsilon
        
        f_plus = evaluate_at(params_plus)
        f_minus = evaluate_at(params_minus)
        hessian_diag[i] = (f_plus - 2*f0 + f_minus) / (epsilon**2)
    
    # Regularize diagonal
    hessian_diag = torch.clamp(hessian_diag, min=-1e6, max=-1e-6)
    
    # Newton step
    step = -gradient / hessian_diag
    
    return step, gradient, hessian_diag

def run_newton_optimization(max_iterations=10):
    """Run Newton's method optimization."""
    print("🚀 Running Newton's Method...")
    
    species_path = "test_trees_1/sp.nwk"
    gene_path = "test_trees_1/g.nwk"
    
    # Initialize
    initial_params = (0.1, 0.1, 0.1)
    log_params = inverse_softplus_transform(torch.tensor(initial_params, dtype=torch.float64))
    
    history = {
        'iterations': [],
        'deltas': [],
        'taus': [],
        'lambdas': [],
        'log_likelihoods': [],
        'gradient_norms': [],
        'times': []
    }
    
    start_time = time.time()
    convergence_threshold = 1e-6
    
    for iteration in range(max_iterations):
        iter_start = time.time()
        
        # Evaluate current point
        current_params = softplus_transform(log_params)
        current_ll = compute_log_likelihood(species_path, gene_path,
                                          float(current_params[0]), float(current_params[1]), float(current_params[2]),
                                          torch.device("cpu"), torch.float64)
        
        # Compute Newton step
        step, gradient, hessian_diag = silent_newton_step(species_path, gene_path, log_params)
        gradient_norm = torch.norm(gradient)
        
        # Store current state
        history['iterations'].append(iteration)
        history['deltas'].append(float(current_params[0]))
        history['taus'].append(float(current_params[1]))
        history['lambdas'].append(float(current_params[2]))
        history['log_likelihoods'].append(current_ll)
        history['gradient_norms'].append(float(gradient_norm))
        history['times'].append(time.time() - start_time)
        
        print(f"Newton iter {iteration}: LL={current_ll:.6f}, δ={current_params[0]:.6f}, |∇|={gradient_norm:.2e}")
        
        # Check convergence
        if gradient_norm < convergence_threshold:
            print(f"Newton converged after {iteration} iterations")
            break
        
        # Line search
        alpha = 1.0
        best_ll = current_ll
        best_log_params = log_params
        
        for _ in range(10):
            new_log_params = log_params + alpha * step
            new_params = softplus_transform(new_log_params)
            new_ll = compute_log_likelihood(species_path, gene_path,
                                          float(new_params[0]), float(new_params[1]), float(new_params[2]),
                                          torch.device("cpu"), torch.float64)
            
            if new_ll > best_ll:
                best_ll = new_ll
                best_log_params = new_log_params
                break
            alpha *= 0.5
        
        # Update
        log_params = best_log_params
        
        # Early stopping for very small parameters
        final_params = softplus_transform(log_params)
        if all(p < 1e-4 for p in [float(final_params[0]), float(final_params[1]), float(final_params[2])]):
            print(f"Newton reached near-zero parameters")
            break
    
    return history

def run_gradient_descent_optimization(max_epochs=100):
    """Run gradient descent optimization."""
    print("📉 Running Gradient Descent...")
    
    optimizer = FiniteDiffCCPOptimizer(
        "test_trees_1/sp.nwk", "test_trees_1/g.nwk",
        initial_params=(0.1, 0.1, 0.1),
        device=torch.device("cpu"),
        dtype=torch.float64
    )
    
    result = optimizer.optimize(lr=0.002, epochs=max_epochs, epsilon=1e-7)
    
    # Convert to same format as Newton
    history = {
        'iterations': result['history']['epochs'],
        'deltas': result['history']['deltas'],
        'taus': result['history']['taus'],
        'lambdas': result['history']['lambdas'],
        'log_likelihoods': result['history']['log_likelihoods'],
        'times': np.cumsum(result['history']['times'])
    }
    
    return history

def create_comprehensive_comparison_plot(newton_history, gd_history):
    """Create comprehensive comparison visualization."""
    
    plt.style.use('default')
    fig = plt.figure(figsize=(16, 12))
    
    # Color scheme
    newton_color = '#2E86C1'  # Blue
    gd_color = '#E74C3C'      # Red
    
    # Plot 1: Parameter Evolution (Delta)
    plt.subplot(3, 4, 1)
    plt.plot(newton_history['iterations'], newton_history['deltas'], 'o-', 
             color=newton_color, linewidth=2, markersize=6, label='Newton')
    plt.plot(gd_history['iterations'], gd_history['deltas'], 's-', 
             color=gd_color, linewidth=1.5, markersize=4, alpha=0.8, label='Gradient Descent')
    plt.xlabel('Iteration')
    plt.ylabel('δ (Delta)')
    plt.title('Delta Parameter Evolution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Plot 2: Parameter Evolution (Tau)
    plt.subplot(3, 4, 2)
    plt.plot(newton_history['iterations'], newton_history['taus'], 'o-', 
             color=newton_color, linewidth=2, markersize=6, label='Newton')
    plt.plot(gd_history['iterations'], gd_history['taus'], 's-', 
             color=gd_color, linewidth=1.5, markersize=4, alpha=0.8, label='Gradient Descent')
    plt.xlabel('Iteration')
    plt.ylabel('τ (Tau)')
    plt.title('Tau Parameter Evolution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Plot 3: Parameter Evolution (Lambda)
    plt.subplot(3, 4, 3)
    plt.plot(newton_history['iterations'], newton_history['lambdas'], 'o-', 
             color=newton_color, linewidth=2, markersize=6, label='Newton')
    plt.plot(gd_history['iterations'], gd_history['lambdas'], 's-', 
             color=gd_color, linewidth=1.5, markersize=4, alpha=0.8, label='Gradient Descent')
    plt.xlabel('Iteration')
    plt.ylabel('λ (Lambda)')
    plt.title('Lambda Parameter Evolution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Plot 4: Log-Likelihood Evolution
    plt.subplot(3, 4, 4)
    plt.plot(newton_history['iterations'], newton_history['log_likelihoods'], 'o-', 
             color=newton_color, linewidth=2, markersize=6, label='Newton')
    plt.plot(gd_history['iterations'], gd_history['log_likelihoods'], 's-', 
             color=gd_color, linewidth=1.5, markersize=4, alpha=0.8, label='Gradient Descent')
    plt.xlabel('Iteration')
    plt.ylabel('Log-Likelihood')
    plt.title('Log-Likelihood Evolution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Distance from Origin
    plt.subplot(3, 4, 5)
    newton_distances = [np.sqrt(d**2 + t**2 + l**2) for d, t, l in 
                       zip(newton_history['deltas'], newton_history['taus'], newton_history['lambdas'])]
    gd_distances = [np.sqrt(d**2 + t**2 + l**2) for d, t, l in 
                   zip(gd_history['deltas'], gd_history['taus'], gd_history['lambdas'])]
    
    plt.plot(newton_history['iterations'], newton_distances, 'o-', 
             color=newton_color, linewidth=2, markersize=6, label='Newton')
    plt.plot(gd_history['iterations'], gd_distances, 's-', 
             color=gd_color, linewidth=1.5, markersize=4, alpha=0.8, label='Gradient Descent')
    plt.xlabel('Iteration')
    plt.ylabel('Distance from Origin')
    plt.title('Convergence to Zero')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Plot 6: Time Evolution
    plt.subplot(3, 4, 6)
    plt.plot(newton_history['times'], newton_history['log_likelihoods'], 'o-', 
             color=newton_color, linewidth=2, markersize=6, label='Newton')
    plt.plot(gd_history['times'], gd_history['log_likelihoods'], 's-', 
             color=gd_color, linewidth=1.5, markersize=4, alpha=0.8, label='Gradient Descent')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Log-Likelihood')
    plt.title('Convergence vs Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 7: Final Parameters Comparison
    plt.subplot(3, 4, 7)
    final_newton = [newton_history['deltas'][-1], newton_history['taus'][-1], newton_history['lambdas'][-1]]
    final_gd = [gd_history['deltas'][-1], gd_history['taus'][-1], gd_history['lambdas'][-1]]
    
    x = np.arange(3)
    width = 0.35
    
    plt.bar(x - width/2, final_newton, width, label='Newton', color=newton_color, alpha=0.7)
    plt.bar(x + width/2, final_gd, width, label='Gradient Descent', color=gd_color, alpha=0.7)
    
    plt.xlabel('Parameter')
    plt.ylabel('Final Value')
    plt.title('Final Parameter Values')
    plt.xticks(x, ['δ', 'τ', 'λ'])
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Plot 8: Convergence Rate Comparison
    plt.subplot(3, 4, 8)
    
    # Calculate improvement per iteration
    newton_improvements = np.diff(newton_history['log_likelihoods'])
    gd_improvements = np.diff(gd_history['log_likelihoods'])
    
    if len(newton_improvements) > 0:
        plt.bar(['Newton\n(1st iter)', 'Gradient Descent\n(avg)'], 
                [newton_improvements[0], np.mean(gd_improvements[:20])],
                color=[newton_color, gd_color], alpha=0.7)
    
    plt.ylabel('LL Improvement per Iteration')
    plt.title('Convergence Rate')
    plt.grid(True, alpha=0.3)
    
    # Plot 9: 3D Parameter Space Trajectory
    ax = plt.subplot(3, 4, 9, projection='3d')
    
    # Sample trajectories for visualization
    newton_sample = min(len(newton_history['deltas']), 10)
    gd_sample = min(len(gd_history['deltas']), 50)
    
    ax.plot(newton_history['deltas'][:newton_sample], 
            newton_history['taus'][:newton_sample], 
            newton_history['lambdas'][:newton_sample], 
            'o-', color=newton_color, linewidth=2, markersize=6, label='Newton')
    
    ax.plot(gd_history['deltas'][::gd_sample//10], 
            gd_history['taus'][::gd_sample//10], 
            gd_history['lambdas'][::gd_sample//10], 
            's-', color=gd_color, linewidth=1.5, markersize=4, alpha=0.8, label='GD')
    
    ax.set_xlabel('δ')
    ax.set_ylabel('τ')
    ax.set_zlabel('λ')
    ax.set_title('3D Parameter Trajectory')
    ax.legend()
    
    # Plot 10: Efficiency Metrics
    plt.subplot(3, 4, 10)
    
    # Calculate efficiency metrics
    newton_final_ll = newton_history['log_likelihoods'][-1]
    gd_final_ll = gd_history['log_likelihoods'][-1]
    newton_time = newton_history['times'][-1]
    gd_time = gd_history['times'][-1]
    newton_iters = len(newton_history['iterations'])
    gd_iters = len(gd_history['iterations'])
    
    metrics = ['Final LL', 'Time (s)', 'Iterations']
    newton_vals = [newton_final_ll, newton_time, newton_iters]
    gd_vals = [gd_final_ll, gd_time, gd_iters]
    
    # Normalize for comparison
    newton_norm = [v/gd_vals[i] for i, v in enumerate(newton_vals)]
    
    x = np.arange(len(metrics))
    plt.bar(x, newton_norm, color=newton_color, alpha=0.7)
    plt.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    plt.xlabel('Metric')
    plt.ylabel('Newton / Gradient Descent')
    plt.title('Efficiency Comparison')
    plt.xticks(x, metrics, rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Plot 11: Learning Curve Comparison
    plt.subplot(3, 4, 11)
    
    # Show first few iterations in detail
    max_show = min(20, len(gd_history['log_likelihoods']))
    
    plt.plot(range(len(newton_history['log_likelihoods'])), newton_history['log_likelihoods'], 
             'o-', color=newton_color, linewidth=3, markersize=8, label='Newton')
    plt.plot(range(max_show), gd_history['log_likelihoods'][:max_show], 
             's-', color=gd_color, linewidth=2, markersize=5, alpha=0.8, label='Gradient Descent')
    
    plt.xlabel('Iteration')
    plt.ylabel('Log-Likelihood')
    plt.title('Learning Curves (First 20 iters)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 12: Summary Statistics
    plt.subplot(3, 4, 12)
    plt.axis('off')
    
    # Calculate summary stats
    newton_improvement = newton_final_ll - newton_history['log_likelihoods'][0]
    gd_improvement = gd_final_ll - gd_history['log_likelihoods'][0]
    time_speedup = gd_time / newton_time if newton_time > 0 else float('inf')
    iter_speedup = gd_iters / newton_iters
    
    summary_text = f"""
PERFORMANCE SUMMARY

Newton's Method:
• Iterations: {newton_iters}
• Time: {newton_time:.1f}s
• Final LL: {newton_final_ll:.6f}
• Improvement: +{newton_improvement:.6f}

Gradient Descent:
• Iterations: {gd_iters}
• Time: {gd_time:.1f}s
• Final LL: {gd_final_ll:.6f}
• Improvement: +{gd_improvement:.6f}

Speedup Factors:
• Time: {time_speedup:.1f}x faster
• Iterations: {iter_speedup:.1f}x fewer
• Quality: {newton_improvement/gd_improvement:.2f}x better
    """
    
    plt.text(0.05, 0.95, summary_text, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('newton_vs_gradient_descent_comparison.png', dpi=300, bbox_inches='tight')
    print("📊 Comprehensive comparison plot saved as 'newton_vs_gradient_descent_comparison.png'")
    
    return fig

def main():
    """Run complete comparison and generate visualization."""
    print("🔬 NEWTON'S METHOD vs GRADIENT DESCENT COMPREHENSIVE COMPARISON")
    print("=" * 70)
    
    # Run both optimizations
    newton_history = run_newton_optimization(max_iterations=10)
    gd_history = run_gradient_descent_optimization(max_epochs=100)
    
    # Create visualization
    fig = create_comprehensive_comparison_plot(newton_history, gd_history)
    
    # Print final summary
    print(f"\n🎯 FINAL RESULTS:")
    print(f"Newton: {len(newton_history['iterations'])} iterations, {newton_history['times'][-1]:.1f}s")
    print(f"Gradient Descent: {len(gd_history['iterations'])} iterations, {gd_history['times'][-1]:.1f}s")
    print(f"Speedup: {len(gd_history['iterations'])/len(newton_history['iterations']):.1f}x fewer iterations")
    
    return newton_history, gd_history, fig

if __name__ == "__main__":
    newton_history, gd_history, fig = main()