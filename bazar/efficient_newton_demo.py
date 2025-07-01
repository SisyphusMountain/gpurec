#!/usr/bin/env python3
"""
Efficient Newton's method demo with simplified Hessian approximation.
"""

import torch
import time
from matmul_ale_ccp_optimize_finite_diff import FiniteDiffCCPOptimizer, compute_log_likelihood, softplus_transform, inverse_softplus_transform

def efficient_newton_step(species_path, gene_path, log_params, epsilon=1e-6):
    """Compute Newton step with diagonal Hessian approximation for speed."""
    
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
    
    # Compute diagonal Hessian (approximation for speed)
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
    hessian_diag = torch.clamp(hessian_diag, min=-1e6, max=-1e-6)  # Ensure negative for maximization
    
    # Newton step (diagonal approximation)
    step = -gradient / hessian_diag
    
    return step, gradient, hessian_diag

def efficient_newton_optimization():
    """Run efficient Newton optimization with diagonal Hessian."""
    print("🚀 Efficient Newton's Method (Diagonal Hessian Approximation)")
    print("=" * 65)
    
    species_path = "test_trees_1/sp.nwk"
    gene_path = "test_trees_1/g.nwk"
    
    # Initialize
    initial_params = (0.1, 0.1, 0.1)
    log_params = inverse_softplus_transform(torch.tensor(initial_params, dtype=torch.float64))
    
    print(f"Starting from: δ={initial_params[0]:.6f}, τ={initial_params[1]:.6f}, λ={initial_params[2]:.6f}")
    
    history = []
    start_time = time.time()
    
    convergence_threshold = 1e-6
    max_iterations = 50
    
    for iteration in range(max_iterations):
        iter_start = time.time()
        
        # Evaluate current point
        current_params = softplus_transform(log_params)
        current_ll = compute_log_likelihood(species_path, gene_path,
                                          float(current_params[0]), float(current_params[1]), float(current_params[2]),
                                          torch.device("cpu"), torch.float64)
        
        print(f"\n📍 Newton Iteration {iteration + 1}")
        print(f"   Current: δ={current_params[0]:.6f}, τ={current_params[1]:.6f}, λ={current_params[2]:.6f}")
        print(f"   Log-likelihood: {current_ll:.6f}")
        
        # Compute Newton step
        step, gradient, hessian_diag = efficient_newton_step(species_path, gene_path, log_params)
        
        gradient_norm = torch.norm(gradient)
        print(f"   Gradient norm: {gradient_norm:.2e}")
        print(f"   Hessian diagonal: [{hessian_diag[0]:.2e}, {hessian_diag[1]:.2e}, {hessian_diag[2]:.2e}]")
        
        # Line search (simple backtracking)
        alpha = 1.0
        for _ in range(10):  # More backtracking steps
            new_log_params = log_params + alpha * step
            new_params = softplus_transform(new_log_params)
            new_ll = compute_log_likelihood(species_path, gene_path,
                                          float(new_params[0]), float(new_params[1]), float(new_params[2]),
                                          torch.device("cpu"), torch.float64)
            
            if new_ll > current_ll:  # Improvement found
                break
            alpha *= 0.5
        
        # Update
        log_params = log_params + alpha * step
        iter_time = time.time() - iter_start
        
        print(f"   Step size: {alpha:.2e}")
        print(f"   Iteration time: {iter_time:.2f}s")
        
        # Store history
        final_params = softplus_transform(log_params)
        history.append({
            'iteration': iteration + 1,
            'params': (float(final_params[0]), float(final_params[1]), float(final_params[2])),
            'log_likelihood': new_ll,
            'gradient_norm': float(gradient_norm),
            'alpha': alpha,
            'time': iter_time
        })
        
        # Check convergence
        if gradient_norm < convergence_threshold:
            print(f"✅ Converged! Gradient norm {gradient_norm:.2e} below threshold {convergence_threshold:.2e}")
            break
        
        # Check for very small parameters (near theoretical optimum)
        if all(p < 1e-4 for p in [float(final_params[0]), float(final_params[1]), float(final_params[2])]):
            print(f"✅ Reached near-zero parameters (theoretical optimum)")
            break
    
    total_time = time.time() - start_time
    final_params = softplus_transform(log_params)
    
    print(f"\n✅ Efficient Newton Complete!")
    print(f"Final: δ={final_params[0]:.6f}, τ={final_params[1]:.6f}, λ={final_params[2]:.6f}")
    print(f"Final log-likelihood: {new_ll:.6f}")
    print(f"Total time: {total_time:.2f}s, Iterations: {len(history)}")
    
    return history

def compare_convergence_rates():
    """Compare Newton vs Gradient Descent convergence rates."""
    print(f"\n🔬 CONVERGENCE RATE COMPARISON")
    print("=" * 50)
    
    # Newton's method
    print(f"Running efficient Newton's method...")
    newton_history = efficient_newton_optimization()
    
    # Gradient descent (few iterations for comparison)
    print(f"\n📉 Running gradient descent for comparison...")
    gd_optimizer = FiniteDiffCCPOptimizer(
        "test_trees_1/sp.nwk", "test_trees_1/g.nwk",
        initial_params=(0.1, 0.1, 0.1),
        device=torch.device("cpu"),
        dtype=torch.float64
    )
    
    gd_result = gd_optimizer.optimize(lr=0.002, epochs=100, epsilon=1e-7)
    
    # Comparison table
    print(f"\n📊 CONVERGENCE COMPARISON:")
    print(f"{'Iteration':<10} {'Newton LL':<12} {'Newton δ':<10} {'GD LL':<12} {'GD δ':<10} {'Improvement'}")
    print("-" * 70)
    
    initial_ll = -6.449834
    
    for i in range(min(len(newton_history), len(gd_result['history']['log_likelihoods']))):
        newton_ll = newton_history[i]['log_likelihood']
        newton_delta = newton_history[i]['params'][0]
        
        gd_ll = gd_result['history']['log_likelihoods'][i]
        gd_delta = gd_result['history']['deltas'][i]
        
        newton_improvement = newton_ll - initial_ll
        gd_improvement = gd_ll - initial_ll
        
        advantage = "Newton" if newton_improvement > gd_improvement else "GD"
        
        print(f"{i+1:<10} {newton_ll:<12.6f} {newton_delta:<10.6f} {gd_ll:<12.6f} {gd_delta:<10.6f} {advantage}")
    
    # Summary
    newton_final_improvement = newton_history[-1]['log_likelihood'] - initial_ll
    gd_final_improvement = gd_result['best_log_likelihood'] - initial_ll
    
    print(f"\n🎯 FINAL COMPARISON:")
    print(f"Newton: {newton_final_improvement:.6f} improvement in {len(newton_history)} iterations")
    print(f"Gradient Descent: {gd_final_improvement:.6f} improvement in {len(gd_result['history']['log_likelihoods'])} iterations")
    
    if newton_final_improvement > gd_final_improvement:
        advantage = newton_final_improvement / gd_final_improvement
        print(f"🚀 Newton achieved {advantage:.2f}x better final result!")
    
    return newton_history, gd_result

if __name__ == "__main__":
    newton_history, gd_result = compare_convergence_rates()