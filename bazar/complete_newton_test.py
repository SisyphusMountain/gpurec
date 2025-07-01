#!/usr/bin/env python3
"""
Complete Newton's method test until convergence without verbose output.
"""

import torch
import time
from matmul_ale_ccp_optimize_finite_diff import FiniteDiffCCPOptimizer, compute_log_likelihood, softplus_transform, inverse_softplus_transform

def silent_newton_step(species_path, gene_path, log_params, epsilon=1e-6):
    """Compute Newton step with diagonal Hessian approximation (silent version)."""
    
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

def complete_newton_optimization():
    """Run complete Newton optimization to convergence."""
    print("🚀 Complete Newton's Method - Silent Optimization")
    print("=" * 55)
    
    species_path = "test_trees_1/sp.nwk"
    gene_path = "test_trees_1/g.nwk"
    
    # Initialize
    initial_params = (0.1, 0.1, 0.1)
    log_params = inverse_softplus_transform(torch.tensor(initial_params, dtype=torch.float64))
    
    print(f"Starting from: δ={initial_params[0]:.6f}, τ={initial_params[1]:.6f}, λ={initial_params[2]:.6f}")
    
    history = []
    start_time = time.time()
    
    convergence_threshold = 1e-6
    max_iterations = 100
    
    for iteration in range(max_iterations):
        # Evaluate current point
        current_params = softplus_transform(log_params)
        current_ll = compute_log_likelihood(species_path, gene_path,
                                          float(current_params[0]), float(current_params[1]), float(current_params[2]),
                                          torch.device("cpu"), torch.float64)
        
        # Compute Newton step
        step, gradient, hessian_diag = silent_newton_step(species_path, gene_path, log_params)
        
        gradient_norm = torch.norm(gradient)
        
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
        final_params = softplus_transform(log_params)
        
        # Store history
        history.append({
            'iteration': iteration + 1,
            'params': (float(final_params[0]), float(final_params[1]), float(final_params[2])),
            'log_likelihood': best_ll,
            'gradient_norm': float(gradient_norm),
            'alpha': alpha
        })
        
        # Progress reporting (every 10 iterations or important ones)
        if iteration % 10 == 0 or iteration < 5 or gradient_norm < convergence_threshold:
            print(f"Iter {iteration+1:2d}: LL={best_ll:8.6f}, δ={final_params[0]:.6f}, τ={final_params[1]:.6f}, λ={final_params[2]:.6f}, |∇|={gradient_norm:.2e}")
        
        # Check convergence
        if gradient_norm < convergence_threshold:
            print(f"✅ Converged! Gradient norm {gradient_norm:.2e} below threshold {convergence_threshold:.2e}")
            break
        
        # Check for very small parameters
        if all(p < 1e-5 for p in [float(final_params[0]), float(final_params[1]), float(final_params[2])]):
            print(f"✅ Reached near-zero parameters (theoretical optimum)")
            break
    
    total_time = time.time() - start_time
    final_params = softplus_transform(log_params)
    
    print(f"\n✅ Newton's Method Complete!")
    print(f"Final: δ={final_params[0]:.8f}, τ={final_params[1]:.8f}, λ={final_params[2]:.8f}")
    print(f"Final log-likelihood: {best_ll:.8f}")
    print(f"Total time: {total_time:.2f}s, Iterations: {len(history)}")
    print(f"Improvement: {best_ll - (-6.449834):.6f}")
    
    return history

def compare_with_gradient_descent():
    """Compare Newton's method with gradient descent convergence."""
    print(f"\n🔬 COMPARISON WITH GRADIENT DESCENT")
    print("=" * 50)
    
    # Newton's method
    print(f"Running Newton's method...")
    newton_history = complete_newton_optimization()
    
    # Gradient descent
    print(f"\n📉 Running gradient descent...")
    gd_optimizer = FiniteDiffCCPOptimizer(
        "test_trees_1/sp.nwk", "test_trees_1/g.nwk",
        initial_params=(0.1, 0.1, 0.1),
        device=torch.device("cpu"),
        dtype=torch.float64
    )
    
    gd_result = gd_optimizer.optimize(lr=0.002, epochs=200, epsilon=1e-7)
    
    # Final comparison
    newton_final = newton_history[-1]
    newton_improvement = newton_final['log_likelihood'] - (-6.449834)
    gd_improvement = gd_result['best_log_likelihood'] - (-6.449834)
    
    print(f"\n📊 FINAL COMPARISON:")
    print(f"Newton:          {newton_improvement:.6f} improvement in {len(newton_history)} iterations")
    print(f"Gradient Descent: {gd_improvement:.6f} improvement in {gd_result['epochs_run']} iterations")
    
    if len(newton_history) < gd_result['epochs_run']:
        speedup = gd_result['epochs_run'] / len(newton_history)
        print(f"🚀 Newton's method converged {speedup:.1f}x faster!")
    
    if newton_improvement > gd_improvement:
        quality_ratio = newton_improvement / gd_improvement
        print(f"⚡ Newton achieved {quality_ratio:.2f}x better final result!")
    
    return newton_history, gd_result

if __name__ == "__main__":
    newton_history, gd_result = compare_with_gradient_descent()