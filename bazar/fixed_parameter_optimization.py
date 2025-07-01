#!/usr/bin/env python3
"""
Fixed Parameter Optimization with Proper Learning Rate and Gradient Handling
==========================================================================

This version addresses the gradient clamping issues by:
1. Using much smaller learning rates
2. Better parameter initialization 
3. Gradient clipping to prevent overshooting
4. Log-parameterization to avoid boundary issues
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import time
from realistic_optimization_comparison import RealisticOptimizer

def log_parameterization(log_params):
    """Convert log parameters to actual parameters using softplus"""
    return F.softplus(log_params)

def inverse_log_parameterization(params):
    """Convert parameters to log space"""
    # Inverse softplus: log(exp(x) - 1) ≈ x for large x
    return torch.where(params > 20, params, torch.log(torch.expm1(params)))

def run_fixed_gradient_descent(species_path: str, gene_path: str, 
                              max_iterations: int = 15) -> dict:
    """Run gradient descent with proper learning rate and parameterization"""
    
    print(f"🚀 FIXED GRADIENT DESCENT OPTIMIZATION")
    print(f"=" * 60)
    
    optimizer = RealisticOptimizer(species_path, gene_path)
    
    # Use log parameterization to avoid clamping issues
    # Start with log(0.1) for all parameters
    log_params = torch.log(torch.tensor([0.1, 0.1, 0.1], dtype=torch.float64))
    
    # Much smaller learning rate to prevent overshooting
    learning_rate = 0.001  # 100x smaller than before
    
    print(f"Using log parameterization with learning_rate={learning_rate}")
    
    history = {
        'iteration': [], 'log_likelihood': [],
        'delta': [], 'tau': [], 'lambda': [],
        'log_delta': [], 'log_tau': [], 'log_lambda': [],
        'gradient_norm': [], 'raw_gradients': [],
        'e_time': [], 'pi_time': [], 'total_time': [],
        'e_iterations': [], 'pi_iterations': []
    }
    
    for iteration in range(max_iterations):
        iter_start = time.time()
        
        # Convert log parameters to actual parameters
        params = log_parameterization(log_params)
        
        print(f"\nIteration {iteration}:")
        print(f"  Log params: δ_log={log_params[0]:.4f}, τ_log={log_params[1]:.4f}, λ_log={log_params[2]:.4f}")
        print(f"  Actual params: δ={params[0]:.6f}, τ={params[1]:.6f}, λ={params[2]:.6f}")
        
        # Compute likelihood
        ll_result = optimizer.compute_likelihood_and_timing(
            float(params[0]), float(params[1]), float(params[2])
        )
        
        # Compute gradients w.r.t. actual parameters
        gradients_actual, grad_timing = optimizer.compute_gradients_realistic(
            float(params[0]), float(params[1]), float(params[2])
        )
        
        # Transform gradients to log parameter space using chain rule
        # d(LL)/d(log_param) = d(LL)/d(param) * d(param)/d(log_param)
        # For softplus: d(softplus(x))/dx = sigmoid(x)
        sigmoid_vals = torch.sigmoid(log_params)
        gradients_log = gradients_actual * sigmoid_vals
        
        gradient_norm = torch.norm(gradients_log).item()
        
        print(f"  LL: {ll_result['log_likelihood']:.6f}")
        print(f"  Gradients (actual): δ={gradients_actual[0]:.3f}, τ={gradients_actual[1]:.3f}, λ={gradients_actual[2]:.3f}")
        print(f"  Gradients (log): δ={gradients_log[0]:.3f}, τ={gradients_log[1]:.3f}, λ={gradients_log[2]:.3f}")
        print(f"  Gradient norm: {gradient_norm:.3f}")
        
        # Clip gradients to prevent overshooting
        max_gradient_norm = 10.0
        if gradient_norm > max_gradient_norm:
            gradients_log = gradients_log * (max_gradient_norm / gradient_norm)
            print(f"  ⚡ Gradients clipped to norm {max_gradient_norm}")
        
        # Update log parameters
        log_params = log_params + learning_rate * gradients_log
        
        iter_time = time.time() - iter_start
        
        # Record history
        history['iteration'].append(iteration)
        history['log_likelihood'].append(ll_result['log_likelihood'])
        history['delta'].append(float(params[0]))
        history['tau'].append(float(params[1]))
        history['lambda'].append(float(params[2]))
        history['log_delta'].append(float(log_params[0]))
        history['log_tau'].append(float(log_params[1]))
        history['log_lambda'].append(float(log_params[2]))
        history['gradient_norm'].append(gradient_norm)
        history['raw_gradients'].append([float(g) for g in gradients_actual])
        history['e_time'].append(ll_result['e_time'])
        history['pi_time'].append(ll_result['pi_time'])
        history['total_time'].append(iter_time)
        history['e_iterations'].append(ll_result['e_iterations'])
        history['pi_iterations'].append(ll_result['pi_iterations'])
        
        print(f"  Time: {iter_time:.1f}s")
        
        # Check convergence
        if gradient_norm < 1e-4:  # Relaxed threshold
            print(f"✅ Converged! Gradient norm {gradient_norm:.2e} below threshold")
            break
    
    return history

def run_fixed_newton_method(species_path: str, gene_path: str,
                           max_iterations: int = 15) -> dict:
    """Run Newton's method with proper learning rate and parameterization"""
    
    print(f"🚀 FIXED NEWTON'S METHOD OPTIMIZATION")
    print(f"=" * 60)
    
    optimizer = RealisticOptimizer(species_path, gene_path)
    
    # Use log parameterization
    log_params = torch.log(torch.tensor([0.1, 0.1, 0.1], dtype=torch.float64))
    
    # Conservative learning rate for Newton's method
    learning_rate = 0.1
    
    print(f"Using log parameterization with learning_rate={learning_rate}")
    
    history = {
        'iteration': [], 'log_likelihood': [], 'll_improvement': [],
        'delta': [], 'tau': [], 'lambda': [],
        'log_delta': [], 'log_tau': [], 'log_lambda': [],
        'gradient_norm': [], 'raw_gradients': [],
        'e_time': [], 'pi_time': [], 'total_time': [],
        'e_iterations': [], 'pi_iterations': []
    }
    
    previous_ll = None
    
    for iteration in range(max_iterations):
        iter_start = time.time()
        
        # Convert log parameters to actual parameters
        params = log_parameterization(log_params)
        
        print(f"\nIteration {iteration}:")
        print(f"  Log params: δ_log={log_params[0]:.4f}, τ_log={log_params[1]:.4f}, λ_log={log_params[2]:.4f}")
        print(f"  Actual params: δ={params[0]:.6f}, τ={params[1]:.6f}, λ={params[2]:.6f}")
        
        # Compute likelihood
        ll_result = optimizer.compute_likelihood_and_timing(
            float(params[0]), float(params[1]), float(params[2])
        )
        
        # Compute gradients w.r.t. actual parameters
        gradients_actual, grad_timing = optimizer.compute_gradients_realistic(
            float(params[0]), float(params[1]), float(params[2])
        )
        
        # Transform gradients to log parameter space
        sigmoid_vals = torch.sigmoid(log_params)
        gradients_log = gradients_actual * sigmoid_vals
        
        # Compute approximate Hessian in log space (diagonal approximation)
        epsilon = 1e-5
        hessian_diag_log = torch.zeros_like(log_params)
        
        base_ll = ll_result['log_likelihood']
        for i in range(3):
            # Save current state
            E_saved = optimizer.E.clone() if optimizer.E is not None else None
            Pi_saved = optimizer.Pi.clone() if optimizer.Pi is not None else None
            
            # Positive perturbation in log space
            log_params_pos = log_params.clone()
            log_params_pos[i] += epsilon
            params_pos = log_parameterization(log_params_pos)
            
            # Restore state
            if E_saved is not None:
                optimizer.E = E_saved.clone()
            if Pi_saved is not None:
                optimizer.Pi = Pi_saved.clone()
            ll_pos = optimizer.compute_likelihood_and_timing(
                float(params_pos[0]), float(params_pos[1]), float(params_pos[2])
            )['log_likelihood']
            
            # Negative perturbation in log space
            log_params_neg = log_params.clone()
            log_params_neg[i] -= epsilon
            params_neg = log_parameterization(log_params_neg)
            
            # Restore state
            if E_saved is not None:
                optimizer.E = E_saved.clone()
            if Pi_saved is not None:
                optimizer.Pi = Pi_saved.clone()
            ll_neg = optimizer.compute_likelihood_and_timing(
                float(params_neg[0]), float(params_neg[1]), float(params_neg[2])
            )['log_likelihood']
            
            # Second derivative in log space
            hessian_diag_log[i] = (ll_pos - 2*base_ll + ll_neg) / (epsilon**2)
            
            # Restore original state
            if E_saved is not None:
                optimizer.E = E_saved
            if Pi_saved is not None:
                optimizer.Pi = Pi_saved
        
        gradient_norm = torch.norm(gradients_log).item()
        
        print(f"  LL: {ll_result['log_likelihood']:.6f}")
        print(f"  Gradients (log): δ={gradients_log[0]:.3f}, τ={gradients_log[1]:.3f}, λ={gradients_log[2]:.3f}")
        print(f"  Hessian diag: δ={hessian_diag_log[0]:.3f}, τ={hessian_diag_log[1]:.3f}, λ={hessian_diag_log[2]:.3f}")
        print(f"  Gradient norm: {gradient_norm:.3f}")
        
        # Newton update with regularization
        hessian_reg = torch.where(torch.abs(hessian_diag_log) < 1e-6,
                                torch.sign(hessian_diag_log) * 1e-6,
                                hessian_diag_log)
        newton_step = gradients_log / hessian_reg
        
        # Clip Newton step
        step_norm = torch.norm(newton_step)
        max_step_norm = 1.0
        if step_norm > max_step_norm:
            newton_step = newton_step * (max_step_norm / step_norm)
            print(f"  ⚡ Newton step clipped to norm {max_step_norm}")
        
        # Update log parameters
        log_params = log_params + learning_rate * newton_step
        
        iter_time = time.time() - iter_start
        ll_improvement = ll_result['log_likelihood'] - previous_ll if previous_ll is not None else 0.0
        
        # Record history
        history['iteration'].append(iteration)
        history['log_likelihood'].append(ll_result['log_likelihood'])
        history['ll_improvement'].append(ll_improvement)
        history['delta'].append(float(params[0]))
        history['tau'].append(float(params[1]))
        history['lambda'].append(float(params[2]))
        history['log_delta'].append(float(log_params[0]))
        history['log_tau'].append(float(log_params[1]))
        history['log_lambda'].append(float(log_params[2]))
        history['gradient_norm'].append(gradient_norm)
        history['raw_gradients'].append([float(g) for g in gradients_actual])
        history['e_time'].append(ll_result['e_time'])
        history['pi_time'].append(ll_result['pi_time'])
        history['total_time'].append(iter_time)
        history['e_iterations'].append(ll_result['e_iterations'])
        history['pi_iterations'].append(ll_result['pi_iterations'])
        
        print(f"  ΔLL: {ll_improvement:.2e}, Time: {iter_time:.1f}s")
        
        # Check convergence
        if previous_ll is not None and abs(ll_improvement) < 1e-6:
            print(f"✅ Converged! LL improvement {abs(ll_improvement):.2e} below threshold")
            break
            
        if gradient_norm < 1e-4:
            print(f"✅ Converged! Gradient norm {gradient_norm:.2e} below threshold") 
            break
        
        previous_ll = ll_result['log_likelihood']
    
    return history

def create_fixed_comparison_plot(gd_history: dict, newton_history: dict):
    """Create comparison plot showing proper parameter evolution"""
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle('Fixed Optimization: Parameter Evolution Comparison', fontsize=16, fontweight='bold')
    
    # Colors
    gd_color = '#2E86AB'
    newton_color = '#A23B72'
    
    # Plot 1: Log-likelihood evolution
    ax1 = axes[0, 0]
    ax1.plot(gd_history['iteration'], gd_history['log_likelihood'], 
             'o-', color=gd_color, label='Gradient Descent', linewidth=2, markersize=4)
    ax1.plot(newton_history['iteration'], newton_history['log_likelihood'], 
             's-', color=newton_color, label="Newton's Method", linewidth=2, markersize=4)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Log-Likelihood')
    ax1.set_title('Log-Likelihood Evolution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Delta parameter evolution
    ax2 = axes[0, 1]
    ax2.plot(gd_history['iteration'], gd_history['delta'], 
             'o-', color=gd_color, label='GD δ', linewidth=2, markersize=4)
    ax2.plot(newton_history['iteration'], newton_history['delta'], 
             's-', color=newton_color, label='Newton δ', linewidth=2, markersize=4)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Duplication Rate (δ)')
    ax2.set_title('Duplication Parameter Evolution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Tau parameter evolution
    ax3 = axes[0, 2]
    ax3.plot(gd_history['iteration'], gd_history['tau'], 
             'o-', color=gd_color, label='GD τ', linewidth=2, markersize=4)
    ax3.plot(newton_history['iteration'], newton_history['tau'], 
             's-', color=newton_color, label='Newton τ', linewidth=2, markersize=4)
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Transfer Rate (τ)')
    ax3.set_title('Transfer Parameter Evolution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Lambda parameter evolution
    ax4 = axes[1, 0]
    ax4.plot(gd_history['iteration'], gd_history['lambda'], 
             'o-', color=gd_color, label='GD λ', linewidth=2, markersize=4)
    ax4.plot(newton_history['iteration'], newton_history['lambda'], 
             's-', color=newton_color, label='Newton λ', linewidth=2, markersize=4)
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Loss Rate (λ)')
    ax4.set_title('Loss Parameter Evolution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: All parameters together (log scale)
    ax5 = axes[1, 1]
    ax5.plot(gd_history['iteration'], gd_history['delta'], 
             'o-', color='red', label='δ (duplication)', linewidth=2, markersize=3)
    ax5.plot(gd_history['iteration'], gd_history['tau'], 
             's-', color='blue', label='τ (transfer)', linewidth=2, markersize=3)
    ax5.plot(gd_history['iteration'], gd_history['lambda'], 
             '^-', color='green', label='λ (loss)', linewidth=2, markersize=3)
    ax5.set_xlabel('Iteration')
    ax5.set_ylabel('Parameter Value')
    ax5.set_title('GD: All Parameters')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_yscale('log')
    
    # Plot 6: Newton all parameters
    ax6 = axes[1, 2]
    ax6.plot(newton_history['iteration'], newton_history['delta'], 
             'o-', color='red', label='δ (duplication)', linewidth=2, markersize=3)
    ax6.plot(newton_history['iteration'], newton_history['tau'], 
             's-', color='blue', label='τ (transfer)', linewidth=2, markersize=3)
    ax6.plot(newton_history['iteration'], newton_history['lambda'], 
             '^-', color='green', label='λ (loss)', linewidth=2, markersize=3)
    ax6.set_xlabel('Iteration')
    ax6.set_ylabel('Parameter Value')
    ax6.set_title('Newton: All Parameters')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.set_yscale('log')
    
    # Plot 7: Gradient norm evolution
    ax7 = axes[2, 0]
    ax7.semilogy(gd_history['iteration'], gd_history['gradient_norm'], 
                 'o-', color=gd_color, label='Gradient Descent', linewidth=2, markersize=4)
    ax7.semilogy(newton_history['iteration'], newton_history['gradient_norm'], 
                 's-', color=newton_color, label="Newton's Method", linewidth=2, markersize=4)
    ax7.set_xlabel('Iteration')
    ax7.set_ylabel('Gradient Norm (log scale)')
    ax7.set_title('Gradient Norm Convergence')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # Plot 8: Parameter trajectory in 3D parameter space (2D projection)
    ax8 = axes[2, 1]
    ax8.plot(gd_history['delta'], gd_history['tau'], 
             'o-', color=gd_color, label='GD Trajectory', linewidth=2, markersize=4, alpha=0.7)
    ax8.plot(newton_history['delta'], newton_history['tau'], 
             's-', color=newton_color, label='Newton Trajectory', linewidth=2, markersize=4, alpha=0.7)
    ax8.scatter(gd_history['delta'][0], gd_history['tau'][0], 
                color='green', s=100, marker='*', label='Start', zorder=5)
    ax8.set_xlabel('Duplication Rate (δ)')
    ax8.set_ylabel('Transfer Rate (τ)')
    ax8.set_title('Parameter Space Trajectory')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    ax8.set_xscale('log')
    ax8.set_yscale('log')
    
    # Plot 9: Summary statistics
    ax9 = axes[2, 2]
    ax9.text(0.1, 0.9, 'FIXED OPTIMIZATION SUMMARY', fontsize=12, fontweight='bold', transform=ax9.transAxes)
    
    # Parameter ranges
    gd_delta_range = f"{min(gd_history['delta']):.2e} - {max(gd_history['delta']):.2e}"
    gd_tau_range = f"{min(gd_history['tau']):.2e} - {max(gd_history['tau']):.2e}"
    gd_lambda_range = f"{min(gd_history['lambda']):.2e} - {max(gd_history['lambda']):.2e}"
    
    newton_delta_range = f"{min(newton_history['delta']):.2e} - {max(newton_history['delta']):.2e}"
    newton_tau_range = f"{min(newton_history['tau']):.2e} - {max(newton_history['tau']):.2e}"
    newton_lambda_range = f"{min(newton_history['lambda']):.2e} - {max(newton_history['lambda']):.2e}"
    
    ax9.text(0.1, 0.8, 'GRADIENT DESCENT:', fontsize=10, fontweight='bold', color=gd_color, transform=ax9.transAxes)
    ax9.text(0.1, 0.75, f'δ range: {gd_delta_range}', fontsize=9, transform=ax9.transAxes)
    ax9.text(0.1, 0.7, f'τ range: {gd_tau_range}', fontsize=9, transform=ax9.transAxes)
    ax9.text(0.1, 0.65, f'λ range: {gd_lambda_range}', fontsize=9, transform=ax9.transAxes)
    ax9.text(0.1, 0.6, f'Final LL: {gd_history["log_likelihood"][-1]:.4f}', fontsize=9, transform=ax9.transAxes)
    ax9.text(0.1, 0.55, f'Iterations: {len(gd_history["iteration"])}', fontsize=9, transform=ax9.transAxes)
    
    ax9.text(0.1, 0.45, "NEWTON'S METHOD:", fontsize=10, fontweight='bold', color=newton_color, transform=ax9.transAxes)
    ax9.text(0.1, 0.4, f'δ range: {newton_delta_range}', fontsize=9, transform=ax9.transAxes)
    ax9.text(0.1, 0.35, f'τ range: {newton_tau_range}', fontsize=9, transform=ax9.transAxes)
    ax9.text(0.1, 0.3, f'λ range: {newton_lambda_range}', fontsize=9, transform=ax9.transAxes)
    ax9.text(0.1, 0.25, f'Final LL: {newton_history["log_likelihood"][-1]:.4f}', fontsize=9, transform=ax9.transAxes)
    ax9.text(0.1, 0.2, f'Iterations: {len(newton_history["iteration"])}', fontsize=9, transform=ax9.transAxes)
    
    # Check if parameters evolved
    gd_evolved = (max(gd_history['delta']) / min(gd_history['delta']) > 2.0)
    newton_evolved = (max(newton_history['delta']) / min(newton_history['delta']) > 2.0)
    
    if gd_evolved or newton_evolved:
        ax9.text(0.1, 0.1, '✅ Parameters are evolving!', fontsize=10, fontweight='bold', 
                color='green', transform=ax9.transAxes)
    else:
        ax9.text(0.1, 0.1, '⚠️  Parameters still not evolving much', fontsize=10, fontweight='bold', 
                color='orange', transform=ax9.transAxes)
    
    ax9.set_xlim(0, 1)
    ax9.set_ylim(0, 1)
    ax9.axis('off')
    
    plt.tight_layout()
    plt.savefig('fixed_parameter_evolution_comparison.png', dpi=300, bbox_inches='tight')
    print(f"📊 Fixed parameter evolution comparison saved as 'fixed_parameter_evolution_comparison.png'")

def main():
    """Main function to test fixed optimization"""
    
    species_path = "test_trees_200/sp.nwk"
    gene_path = "test_trees_200/g.nwk"
    
    print("🚀 TESTING FIXED PARAMETER OPTIMIZATION")
    print("=" * 80)
    
    # Run fixed gradient descent
    gd_history = run_fixed_gradient_descent(species_path, gene_path, max_iterations=10)
    
    print(f"\n")
    
    # Run fixed Newton's method
    newton_history = run_fixed_newton_method(species_path, gene_path, max_iterations=10)
    
    print(f"\n")
    
    # Create comparison plot
    create_fixed_comparison_plot(gd_history, newton_history)
    
    # Print summary
    print("📈 FIXED OPTIMIZATION SUMMARY")
    print("=" * 50)
    
    print("Parameter Evolution Analysis:")
    print(f"  GD δ: {gd_history['delta'][0]:.2e} → {gd_history['delta'][-1]:.2e} (ratio: {gd_history['delta'][-1]/gd_history['delta'][0]:.2f})")
    print(f"  Newton δ: {newton_history['delta'][0]:.2e} → {newton_history['delta'][-1]:.2e} (ratio: {newton_history['delta'][-1]/newton_history['delta'][0]:.2f})")
    
    print(f"\nFinal Log-Likelihoods:")
    print(f"  GD: {gd_history['log_likelihood'][-1]:.6f}")
    print(f"  Newton: {newton_history['log_likelihood'][-1]:.6f}")

if __name__ == "__main__":
    main()