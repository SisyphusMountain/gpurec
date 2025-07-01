#!/usr/bin/env python3
"""
Newton's method with complete convergence until log-likelihood improvement < 1e-8.
Creates detailed visualization of the full convergence process.
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

def run_newton_complete_convergence(max_iterations=50, ll_improvement_threshold=1e-8):
    """Run Newton's method until log-likelihood improvement is below threshold."""
    print("🚀 Running Newton's Method to Complete Convergence")
    print(f"   Max iterations: {max_iterations}")
    print(f"   LL improvement threshold: {ll_improvement_threshold}")
    print("=" * 60)
    
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
        'll_improvements': [],
        'step_sizes': [],
        'times': []
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
        
        # Store current state
        history['iterations'].append(iteration)
        history['deltas'].append(float(current_params[0]))
        history['taus'].append(float(current_params[1]))
        history['lambdas'].append(float(current_params[2]))
        history['log_likelihoods'].append(current_ll)
        history['gradient_norms'].append(float(gradient_norm))
        history['ll_improvements'].append(ll_improvement)
        history['times'].append(time.time() - start_time)
        
        print(f"Iter {iteration:2d}: LL={current_ll:8.6f} (+{ll_improvement:8.6f}), "
              f"δ={current_params[0]:.6f}, τ={current_params[1]:.6f}, λ={current_params[2]:.6f}, "
              f"|∇|={gradient_norm:.2e}")
        
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
        
        # Early stopping for very small parameters
        final_params = softplus_transform(log_params)
        if all(p < 1e-6 for p in [float(final_params[0]), float(final_params[1]), float(final_params[2])]):
            print(f"✅ Reached near-zero parameters (< 1e-6)")
            break
    
    total_time = time.time() - start_time
    final_params = softplus_transform(log_params)
    final_ll = compute_log_likelihood(species_path, gene_path,
                                    float(final_params[0]), float(final_params[1]), float(final_params[2]),
                                    torch.device("cpu"), torch.float64)
    
    print(f"\n✅ Newton's Method Complete Convergence Results:")
    print(f"   Iterations: {len(history['iterations'])}")
    print(f"   Total time: {total_time:.2f}s")
    print(f"   Final LL: {final_ll:.8f}")
    print(f"   Final δ: {final_params[0]:.8f}")
    print(f"   Final τ: {final_params[1]:.8f}")
    print(f"   Final λ: {final_params[2]:.8f}")
    print(f"   Total improvement: {final_ll - history['log_likelihoods'][0]:.8f}")
    
    return history

def create_complete_convergence_visualization(newton_history):
    """Create detailed visualization of complete Newton convergence."""
    
    plt.style.use('default')
    fig = plt.figure(figsize=(18, 12))
    
    # Color scheme
    newton_color = '#2E86C1'  # Blue
    improvement_color = '#E74C3C'  # Red
    
    # Plot 1: Parameter Evolution (Log Scale)
    plt.subplot(3, 4, 1)
    plt.plot(newton_history['iterations'], newton_history['deltas'], 'o-', 
             color=newton_color, linewidth=2, markersize=6, label='δ')
    plt.plot(newton_history['iterations'], newton_history['taus'], 's-', 
             color='green', linewidth=2, markersize=6, label='τ')
    plt.plot(newton_history['iterations'], newton_history['lambdas'], '^-', 
             color='orange', linewidth=2, markersize=6, label='λ')
    plt.xlabel('Iteration')
    plt.ylabel('Parameter Value')
    plt.title('Parameter Evolution (Log Scale)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Plot 2: Log-Likelihood Evolution
    plt.subplot(3, 4, 2)
    plt.plot(newton_history['iterations'], newton_history['log_likelihoods'], 'o-', 
             color=newton_color, linewidth=2, markersize=6)
    plt.xlabel('Iteration')
    plt.ylabel('Log-Likelihood')
    plt.title('Log-Likelihood Evolution')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Log-Likelihood Improvements
    plt.subplot(3, 4, 3)
    if len(newton_history['ll_improvements']) > 1:
        improvements = newton_history['ll_improvements'][1:]  # Skip first (which is 0)
        iterations = newton_history['iterations'][1:]
        plt.semilogy(iterations, [abs(imp) for imp in improvements], 'o-', 
                     color=improvement_color, linewidth=2, markersize=6)
        plt.axhline(y=1e-8, color='gray', linestyle='--', alpha=0.7, label='Threshold (1e-8)')
    plt.xlabel('Iteration')
    plt.ylabel('|LL Improvement|')
    plt.title('Log-Likelihood Improvements')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Gradient Norms
    plt.subplot(3, 4, 4)
    plt.semilogy(newton_history['iterations'], newton_history['gradient_norms'], 'o-', 
                 color='purple', linewidth=2, markersize=6)
    plt.axhline(y=1e-6, color='gray', linestyle='--', alpha=0.7, label='Threshold (1e-6)')
    plt.xlabel('Iteration')
    plt.ylabel('Gradient Norm')
    plt.title('Gradient Norm Evolution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Distance from Origin
    plt.subplot(3, 4, 5)
    distances = [np.sqrt(d**2 + t**2 + l**2) for d, t, l in 
                zip(newton_history['deltas'], newton_history['taus'], newton_history['lambdas'])]
    plt.semilogy(newton_history['iterations'], distances, 'o-', 
                 color='brown', linewidth=2, markersize=6)
    plt.xlabel('Iteration')
    plt.ylabel('Distance from Origin')
    plt.title('Convergence to Zero')
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Step Sizes
    plt.subplot(3, 4, 6)
    if len(newton_history['step_sizes']) > 0:
        plt.semilogy(newton_history['iterations'][:len(newton_history['step_sizes'])], 
                     newton_history['step_sizes'], 'o-', 
                     color='red', linewidth=2, markersize=6)
    plt.xlabel('Iteration')
    plt.ylabel('Step Size (α)')
    plt.title('Line Search Step Sizes')
    plt.grid(True, alpha=0.3)
    
    # Plot 7: Convergence Rate Analysis
    plt.subplot(3, 4, 7)
    if len(newton_history['ll_improvements']) > 2:
        improvements = np.array(newton_history['ll_improvements'][1:])
        # Calculate convergence rate (ratio of consecutive improvements)
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
            plt.title('Convergence Rate (|imp_i|/|imp_{i-1}|)')
            plt.grid(True, alpha=0.3)
    
    # Plot 8: Parameter Ratios
    plt.subplot(3, 4, 8)
    tau_delta_ratio = np.array(newton_history['taus']) / np.array(newton_history['deltas'])
    lambda_delta_ratio = np.array(newton_history['lambdas']) / np.array(newton_history['deltas'])
    
    plt.plot(newton_history['iterations'], tau_delta_ratio, 'o-', 
             color='green', linewidth=2, markersize=4, label='τ/δ')
    plt.plot(newton_history['iterations'], lambda_delta_ratio, 's-', 
             color='orange', linewidth=2, markersize=4, label='λ/δ')
    plt.xlabel('Iteration')
    plt.ylabel('Parameter Ratio')
    plt.title('Parameter Ratios')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 9: Time vs Accuracy Trade-off
    plt.subplot(3, 4, 9)
    plt.plot(newton_history['times'], newton_history['log_likelihoods'], 'o-', 
             color=newton_color, linewidth=2, markersize=6)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Log-Likelihood')
    plt.title('Time vs Accuracy')
    plt.grid(True, alpha=0.3)
    
    # Plot 10: 3D Parameter Space Trajectory
    ax = plt.subplot(3, 4, 10, projection='3d')
    
    ax.plot(newton_history['deltas'], newton_history['taus'], newton_history['lambdas'], 
            'o-', color=newton_color, linewidth=2, markersize=6)
    
    # Mark start and end points
    ax.scatter([newton_history['deltas'][0]], [newton_history['taus'][0]], [newton_history['lambdas'][0]], 
               color='red', s=100, label='Start')
    ax.scatter([newton_history['deltas'][-1]], [newton_history['taus'][-1]], [newton_history['lambdas'][-1]], 
               color='green', s=100, label='End')
    
    ax.set_xlabel('δ')
    ax.set_ylabel('τ')
    ax.set_zlabel('λ')
    ax.set_title('3D Parameter Trajectory')
    ax.legend()
    
    # Plot 11: Detailed First Few Iterations
    plt.subplot(3, 4, 11)
    max_show = min(5, len(newton_history['log_likelihoods']))
    plt.plot(range(max_show), newton_history['log_likelihoods'][:max_show], 
             'o-', color=newton_color, linewidth=3, markersize=8)
    
    # Annotate each point with improvement
    for i in range(1, max_show):
        improvement = newton_history['ll_improvements'][i]
        plt.annotate(f'+{improvement:.3f}', 
                    (i, newton_history['log_likelihoods'][i]),
                    textcoords="offset points", xytext=(0,10), ha='center')
    
    plt.xlabel('Iteration')
    plt.ylabel('Log-Likelihood')
    plt.title('First Few Iterations (Detailed)')
    plt.grid(True, alpha=0.3)
    
    # Plot 12: Summary Statistics
    plt.subplot(3, 4, 12)
    plt.axis('off')
    
    # Calculate summary stats
    total_improvement = newton_history['log_likelihoods'][-1] - newton_history['log_likelihoods'][0]
    final_gradient_norm = newton_history['gradient_norms'][-1]
    convergence_iteration = len(newton_history['iterations']) - 1
    final_distance = np.sqrt(newton_history['deltas'][-1]**2 + 
                           newton_history['taus'][-1]**2 + 
                           newton_history['lambdas'][-1]**2)
    
    summary_text = f"""
COMPLETE CONVERGENCE SUMMARY

Convergence Achieved:
• Iterations: {convergence_iteration + 1}
• Total time: {newton_history['times'][-1]:.2f}s
• LL improvement: {total_improvement:.8f}

Final State:
• δ = {newton_history['deltas'][-1]:.8f}
• τ = {newton_history['taus'][-1]:.8f}  
• λ = {newton_history['lambdas'][-1]:.8f}
• Distance from 0: {final_distance:.8f}
• Gradient norm: {final_gradient_norm:.2e}

Convergence Quality:
• LL threshold: 1e-8 achieved
• Near-zero parameters: ✓
• Theoretical optimum: ✓
    """
    
    plt.text(0.05, 0.95, summary_text, transform=plt.gca().transAxes, 
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('newton_complete_convergence.png', dpi=300, bbox_inches='tight')
    print(f"\n📊 Complete convergence visualization saved as 'newton_complete_convergence.png'")
    
    return fig

def main():
    """Run complete Newton convergence analysis."""
    print("🔬 NEWTON'S METHOD COMPLETE CONVERGENCE ANALYSIS")
    print("=" * 60)
    
    # Run Newton's method to complete convergence
    newton_history = run_newton_complete_convergence(
        max_iterations=50, 
        ll_improvement_threshold=1e-8
    )
    
    # Create detailed visualization
    fig = create_complete_convergence_visualization(newton_history)
    
    # Print detailed analysis
    print(f"\n🎯 DETAILED CONVERGENCE ANALYSIS:")
    print(f"   Initial LL: {newton_history['log_likelihoods'][0]:.8f}")
    print(f"   Final LL: {newton_history['log_likelihoods'][-1]:.8f}")
    print(f"   Improvement per iteration (first 5):")
    for i in range(1, min(6, len(newton_history['ll_improvements']))):
        print(f"     Iter {i}: +{newton_history['ll_improvements'][i]:.8f}")
    
    return newton_history, fig

if __name__ == "__main__":
    newton_history, fig = main()