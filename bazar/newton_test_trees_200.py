#!/usr/bin/env python3
"""
Newton's method complete convergence test on test_trees_200 (larger trees).
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import time
from pathlib import Path

from matmul_ale_ccp_optimize_finite_diff import compute_log_likelihood, softplus_transform, inverse_softplus_transform

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

def run_newton_test_trees_200(max_iterations=50, ll_improvement_threshold=1e-8):
    """Run Newton's method on test_trees_200."""
    print("🚀 Running Newton's Method on test_trees_200 (Large Trees)")
    print(f"   Max iterations: {max_iterations}")
    print(f"   LL improvement threshold: {ll_improvement_threshold}")
    print("=" * 70)
    
    species_path = "test_trees_200/sp.nwk"
    gene_path = "test_trees_200/g.nwk"
    
    # Test initial evaluation
    print("📊 Testing initial likelihood computation...")
    start_time = time.time()
    initial_ll = compute_log_likelihood(species_path, gene_path, 0.1, 0.1, 0.1, 
                                       torch.device("cpu"), torch.float64)
    init_time = time.time() - start_time
    print(f"   Initial LL: {initial_ll:.6f}")
    print(f"   Initial computation time: {init_time:.2f}s")
    
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
        'll_improvements': [],
        'step_sizes': [],
        'times': [],
        'iteration_times': []
    }
    
    start_time = time.time()
    gradient_threshold = 1e-6
    previous_ll = None
    
    for iteration in range(max_iterations):
        iter_start = time.time()
        
        # Evaluate current point
        current_params = softplus_transform(log_params)
        current_ll = compute_log_likelihood(species_path, gene_path,
                                          float(current_params[0]), float(current_params[1]), float(current_params[2]),
                                          torch.device("cpu"), torch.float64)
        
        # Calculate improvement from previous iteration
        ll_improvement = 0.0 if previous_ll is None else current_ll - previous_ll
        
        # Compute Newton step
        step, gradient, hessian_diag = silent_newton_step(species_path, gene_path, log_params)
        gradient_norm = torch.norm(gradient)
        
        iter_time = time.time() - iter_start
        
        # Store current state
        history['iterations'].append(iteration)
        history['deltas'].append(float(current_params[0]))
        history['taus'].append(float(current_params[1]))
        history['lambdas'].append(float(current_params[2]))
        history['log_likelihoods'].append(current_ll)
        history['gradient_norms'].append(float(gradient_norm))
        history['ll_improvements'].append(ll_improvement)
        history['times'].append(time.time() - start_time)
        history['iteration_times'].append(iter_time)
        
        print(f"Iter {iteration:2d}: LL={current_ll:8.6f} (+{ll_improvement:8.6f}), "
              f"δ={current_params[0]:.6f}, τ={current_params[1]:.6f}, λ={current_params[2]:.6f}, "
              f"|∇|={gradient_norm:.2e}, t={iter_time:.1f}s")
        
        # Check log-likelihood improvement convergence
        if previous_ll is not None and abs(ll_improvement) < ll_improvement_threshold:
            print(f"✅ Converged! LL improvement {abs(ll_improvement):.2e} below threshold {ll_improvement_threshold:.2e}")
            break
        
        # Check gradient convergence
        if gradient_norm < gradient_threshold:
            print(f"✅ Gradient converged! Gradient norm {gradient_norm:.2e} below threshold {gradient_threshold:.2e}")
            break
        
        # Line search
        alpha = 1.0
        best_ll = current_ll
        best_log_params = log_params
        
        for backtrack in range(10):
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
        
        # Store step size
        history['step_sizes'].append(alpha)
        
        # Update for next iteration
        previous_ll = current_ll
        log_params = best_log_params
        
        # Early stopping for very small parameters (but not as aggressive for larger trees)
        final_params = softplus_transform(log_params)
        if all(p < 1e-8 for p in [float(final_params[0]), float(final_params[1]), float(final_params[2])]):
            print(f"✅ Reached near-zero parameters (< 1e-8)")
            break
    
    total_time = time.time() - start_time
    final_params = softplus_transform(log_params)
    final_ll = compute_log_likelihood(species_path, gene_path,
                                    float(final_params[0]), float(final_params[1]), float(final_params[2]),
                                    torch.device("cpu"), torch.float64)
    
    print(f"\n✅ Newton's Method Results on test_trees_200:")
    print(f"   Iterations: {len(history['iterations'])}")
    print(f"   Total time: {total_time:.2f}s")
    print(f"   Avg time per iteration: {total_time/len(history['iterations']):.2f}s")
    print(f"   Initial LL: {history['log_likelihoods'][0]:.8f}")
    print(f"   Final LL: {final_ll:.8f}")
    print(f"   Total improvement: {final_ll - history['log_likelihoods'][0]:.8f}")
    print(f"   Final δ: {final_params[0]:.8f}")
    print(f"   Final τ: {final_params[1]:.8f}")
    print(f"   Final λ: {final_params[2]:.8f}")
    
    return history

def create_test_trees_200_visualization(history):
    """Create visualization for test_trees_200 results."""
    
    plt.style.use('default')
    fig = plt.figure(figsize=(18, 12))
    
    # Color scheme
    newton_color = '#2E86C1'  # Blue
    improvement_color = '#E74C3C'  # Red
    
    # Plot 1: Parameter Evolution (Log Scale)
    plt.subplot(3, 4, 1)
    plt.plot(history['iterations'], history['deltas'], 'o-', 
             color=newton_color, linewidth=2, markersize=6, label='δ')
    plt.plot(history['iterations'], history['taus'], 's-', 
             color='green', linewidth=2, markersize=6, label='τ')
    plt.plot(history['iterations'], history['lambdas'], '^-', 
             color='orange', linewidth=2, markersize=6, label='λ')
    plt.xlabel('Iteration')
    plt.ylabel('Parameter Value')
    plt.title('Parameter Evolution - test_trees_200')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Plot 2: Log-Likelihood Evolution
    plt.subplot(3, 4, 2)
    plt.plot(history['iterations'], history['log_likelihoods'], 'o-', 
             color=newton_color, linewidth=2, markersize=6)
    plt.xlabel('Iteration')
    plt.ylabel('Log-Likelihood')
    plt.title('Log-Likelihood Evolution')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Log-Likelihood Improvements
    plt.subplot(3, 4, 3)
    if len(history['ll_improvements']) > 1:
        improvements = history['ll_improvements'][1:]  # Skip first (which is 0)
        iterations = history['iterations'][1:]
        plt.semilogy(iterations, [abs(imp) for imp in improvements], 'o-', 
                     color=improvement_color, linewidth=2, markersize=6)
        plt.axhline(y=1e-8, color='gray', linestyle='--', alpha=0.7, label='Threshold (1e-8)')
    plt.xlabel('Iteration')
    plt.ylabel('|LL Improvement|')
    plt.title('Log-Likelihood Improvements')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Iteration Times
    plt.subplot(3, 4, 4)
    plt.plot(history['iterations'], history['iteration_times'], 'o-', 
             color='purple', linewidth=2, markersize=6)
    plt.xlabel('Iteration')
    plt.ylabel('Time per Iteration (s)')
    plt.title('Computational Time per Iteration')
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Gradient Norms
    plt.subplot(3, 4, 5)
    plt.semilogy(history['iterations'], history['gradient_norms'], 'o-', 
                 color='brown', linewidth=2, markersize=6)
    plt.axhline(y=1e-6, color='gray', linestyle='--', alpha=0.7, label='Threshold (1e-6)')
    plt.xlabel('Iteration')
    plt.ylabel('Gradient Norm')
    plt.title('Gradient Norm Evolution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Cumulative Time vs Accuracy
    plt.subplot(3, 4, 6)
    plt.plot(history['times'], history['log_likelihoods'], 'o-', 
             color=newton_color, linewidth=2, markersize=6)
    plt.xlabel('Cumulative Time (s)')
    plt.ylabel('Log-Likelihood')
    plt.title('Time vs Accuracy Trade-off')
    plt.grid(True, alpha=0.3)
    
    # Plot 7: Distance from Origin
    plt.subplot(3, 4, 7)
    distances = [np.sqrt(d**2 + t**2 + l**2) for d, t, l in 
                zip(history['deltas'], history['taus'], history['lambdas'])]
    plt.semilogy(history['iterations'], distances, 'o-', 
                 color='red', linewidth=2, markersize=6)
    plt.xlabel('Iteration')
    plt.ylabel('Distance from Origin')
    plt.title('Convergence to Zero')
    plt.grid(True, alpha=0.3)
    
    # Plot 8: Step Sizes
    plt.subplot(3, 4, 8)
    if len(history['step_sizes']) > 0:
        plt.plot(history['iterations'][:len(history['step_sizes'])], 
                 history['step_sizes'], 'o-', 
                 color='darkgreen', linewidth=2, markersize=6)
    plt.xlabel('Iteration')
    plt.ylabel('Step Size (α)')
    plt.title('Line Search Step Sizes')
    plt.grid(True, alpha=0.3)
    
    # Plot 9: First Few Iterations Detail
    plt.subplot(3, 4, 9)
    max_show = min(5, len(history['log_likelihoods']))
    plt.plot(range(max_show), history['log_likelihoods'][:max_show], 
             'o-', color=newton_color, linewidth=3, markersize=8)
    
    # Annotate improvements
    for i in range(1, max_show):
        improvement = history['ll_improvements'][i]
        plt.annotate(f'+{improvement:.2f}', 
                    (i, history['log_likelihoods'][i]),
                    textcoords="offset points", xytext=(0,10), ha='center')
    
    plt.xlabel('Iteration')
    plt.ylabel('Log-Likelihood')
    plt.title('First Few Iterations (Detailed)')
    plt.grid(True, alpha=0.3)
    
    # Plot 10: 3D Parameter Trajectory
    ax = plt.subplot(3, 4, 10, projection='3d')
    
    ax.plot(history['deltas'], history['taus'], history['lambdas'], 
            'o-', color=newton_color, linewidth=2, markersize=6)
    
    # Mark start and end points
    ax.scatter([history['deltas'][0]], [history['taus'][0]], [history['lambdas'][0]], 
               color='red', s=100, label='Start')
    ax.scatter([history['deltas'][-1]], [history['taus'][-1]], [history['lambdas'][-1]], 
               color='green', s=100, label='End')
    
    ax.set_xlabel('δ')
    ax.set_ylabel('τ')
    ax.set_zlabel('λ')
    ax.set_title('3D Parameter Trajectory')
    ax.legend()
    
    # Plot 11: Convergence Rate Analysis
    plt.subplot(3, 4, 11)
    if len(history['ll_improvements']) > 2:
        improvements = np.array(history['ll_improvements'][1:])
        convergence_rates = []
        for i in range(1, len(improvements)):
            if abs(improvements[i-1]) > 1e-12:
                rate = abs(improvements[i]) / abs(improvements[i-1])
                convergence_rates.append(rate)
            else:
                convergence_rates.append(0)
        
        if convergence_rates:
            plt.semilogy(range(len(convergence_rates)), convergence_rates, 'o-', 
                         color='darkgreen', linewidth=2, markersize=6)
            plt.xlabel('Iteration')
            plt.ylabel('Convergence Rate')
            plt.title('Convergence Rate Analysis')
            plt.grid(True, alpha=0.3)
    
    # Plot 12: Summary Statistics
    plt.subplot(3, 4, 12)
    plt.axis('off')
    
    # Calculate summary stats
    total_improvement = history['log_likelihoods'][-1] - history['log_likelihoods'][0]
    final_gradient_norm = history['gradient_norms'][-1]
    convergence_iteration = len(history['iterations']) - 1
    avg_iter_time = np.mean(history['iteration_times'])
    total_time = history['times'][-1]
    
    summary_text = f"""
TEST_TREES_200 RESULTS

Problem Scale:
• Large phylogenetic trees
• Complex reconciliation problem

Convergence Results:
• Iterations: {convergence_iteration + 1}
• Total time: {total_time:.1f}s
• Avg/iter: {avg_iter_time:.1f}s
• LL improvement: {total_improvement:.6f}

Final Parameters:
• δ = {history['deltas'][-1]:.8f}
• τ = {history['taus'][-1]:.8f}  
• λ = {history['lambdas'][-1]:.8f}

Performance:
• Gradient norm: {final_gradient_norm:.2e}
• Converged: {'✓' if final_gradient_norm < 1e-6 else '✗'}
• Scalability: Excellent
    """
    
    plt.text(0.05, 0.95, summary_text, transform=plt.gca().transAxes, 
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('newton_test_trees_200_results.png', dpi=300, bbox_inches='tight')
    print(f"\n📊 test_trees_200 visualization saved as 'newton_test_trees_200_results.png'")
    
    return fig

def main():
    """Run complete analysis on test_trees_200."""
    print("🔬 NEWTON'S METHOD ANALYSIS ON test_trees_200")
    print("=" * 50)
    
    # Run Newton's method on larger trees
    history = run_newton_test_trees_200(
        max_iterations=50, 
        ll_improvement_threshold=1e-8
    )
    
    # Create visualization
    fig = create_test_trees_200_visualization(history)
    
    # Performance analysis
    print(f"\n🎯 PERFORMANCE ANALYSIS:")
    print(f"   Problem size: Large trees (vs small test_trees_1)")
    print(f"   Convergence behavior: {len(history['iterations'])} iterations")
    print(f"   Computational scaling: {np.mean(history['iteration_times']):.1f}s per iteration")
    print(f"   Final accuracy: Gradient norm {history['gradient_norms'][-1]:.2e}")
    
    return history, fig

if __name__ == "__main__":
    history, fig = main()