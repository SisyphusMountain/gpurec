#!/usr/bin/env python3
"""
Newton's method optimization for phylogenetic reconciliation parameters.
Uses finite differences for both gradient and Hessian computation.
"""

import torch
import numpy as np
import time
from typing import Dict, Tuple, Optional, Any

from matmul_ale_ccp_optimize_finite_diff import (
    FiniteDiffCCPOptimizer, compute_log_likelihood,
    softplus_transform, inverse_softplus_transform
)

class NewtonCCPOptimizer:
    """Newton's method optimizer for CCP reconciliation parameters."""
    
    def __init__(self, species_tree_path: str, gene_tree_path: str,
                 initial_params: Optional[Tuple[float, float, float]] = None,
                 device: Optional[torch.device] = None, dtype: torch.dtype = torch.float64):
        """Initialize the Newton optimizer."""
        self.species_tree_path = species_tree_path
        self.gene_tree_path = gene_tree_path
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        
        # Initialize parameters
        if initial_params is None:
            initial_params = (0.1, 0.1, 0.1)  # (delta, tau, lambda)
        
        params_tensor = torch.tensor(initial_params, dtype=dtype, device=self.device)
        self.log_params = torch.nn.Parameter(inverse_softplus_transform(params_tensor))
        
        print(f"✅ Newton Optimizer initialized:")
        print(f"   Device: {self.device}")
        print(f"   Initial params: δ={initial_params[0]:.3f}, τ={initial_params[1]:.3f}, λ={initial_params[2]:.3f}")
    
    def get_current_params(self) -> Tuple[float, float, float]:
        """Get current parameter values (delta, tau, lambda)."""
        with torch.no_grad():
            params = softplus_transform(self.log_params)
            return float(params[0]), float(params[1]), float(params[2])
    
    def evaluate_likelihood(self, log_params: Optional[torch.Tensor] = None) -> float:
        """Evaluate likelihood at given or current parameters."""
        if log_params is None:
            log_params = self.log_params.data
            
        with torch.no_grad():
            params = softplus_transform(log_params)
            delta, tau, lambda_param = float(params[0]), float(params[1]), float(params[2])
            return compute_log_likelihood(self.species_tree_path, self.gene_tree_path,
                                        delta, tau, lambda_param, self.device, self.dtype)
    
    def compute_gradient(self, log_params: torch.Tensor, epsilon: float = 1e-7) -> torch.Tensor:
        """Compute gradient using finite differences."""
        gradient = torch.zeros_like(log_params)
        
        # Central difference for each parameter
        f0 = self.evaluate_likelihood(log_params)
        
        for i in range(len(log_params)):
            log_params_plus = log_params.clone()
            log_params_minus = log_params.clone()
            
            log_params_plus[i] += epsilon
            log_params_minus[i] -= epsilon
            
            f_plus = self.evaluate_likelihood(log_params_plus)
            f_minus = self.evaluate_likelihood(log_params_minus)
            
            gradient[i] = (f_plus - f_minus) / (2 * epsilon)
        
        # Handle NaN/inf values
        gradient = torch.where(torch.isfinite(gradient), gradient, torch.zeros_like(gradient))
        
        return gradient
    
    def compute_hessian(self, log_params: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
        """Compute Hessian matrix using finite differences."""
        n = len(log_params)
        hessian = torch.zeros((n, n), dtype=self.dtype, device=self.device)
        
        print(f"   Computing Hessian matrix ({n}×{n})...")
        
        # Method 1: Symmetric finite differences for off-diagonal terms
        # H[i,j] = (f(x+εᵢ+εⱼ) - f(x+εᵢ-εⱼ) - f(x-εᵢ+εⱼ) + f(x-εᵢ-εⱼ)) / (4ε²)
        
        for i in range(n):
            for j in range(i, n):  # Only compute upper triangle due to symmetry
                if i == j:
                    # Diagonal terms: second derivative
                    log_params_plus = log_params.clone()
                    log_params_minus = log_params.clone()
                    log_params_plus[i] += epsilon
                    log_params_minus[i] -= epsilon
                    
                    f_center = self.evaluate_likelihood(log_params)
                    f_plus = self.evaluate_likelihood(log_params_plus)
                    f_minus = self.evaluate_likelihood(log_params_minus)
                    
                    hessian[i, i] = (f_plus - 2*f_center + f_minus) / (epsilon**2)
                else:
                    # Off-diagonal terms: mixed partial derivatives
                    eps_i = torch.zeros_like(log_params)
                    eps_j = torch.zeros_like(log_params)
                    eps_i[i] = epsilon
                    eps_j[j] = epsilon
                    
                    f_pp = self.evaluate_likelihood(log_params + eps_i + eps_j)
                    f_pm = self.evaluate_likelihood(log_params + eps_i - eps_j)
                    f_mp = self.evaluate_likelihood(log_params - eps_i + eps_j)
                    f_mm = self.evaluate_likelihood(log_params - eps_i - eps_j)
                    
                    hessian[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * epsilon**2)
                    hessian[j, i] = hessian[i, j]  # Symmetry
        
        # Handle NaN/inf values
        hessian = torch.where(torch.isfinite(hessian), hessian, torch.zeros_like(hessian))
        
        return hessian
    
    def regularize_hessian(self, hessian: torch.Tensor, regularization: float = 1e-6) -> torch.Tensor:
        """Regularize Hessian to ensure positive definiteness."""
        # Add regularization to diagonal
        regularized = hessian + regularization * torch.eye(hessian.shape[0], 
                                                           dtype=self.dtype, device=self.device)
        
        # Check condition number
        try:
            eigenvals = torch.linalg.eigvals(regularized)
            min_eig = torch.min(eigenvals.real)
            max_eig = torch.max(eigenvals.real)
            condition_number = max_eig / (min_eig + 1e-12)
            
            print(f"   Hessian eigenvalues: min={min_eig:.2e}, max={max_eig:.2e}, cond={condition_number:.2e}")
            
            # If poorly conditioned, add more regularization
            if condition_number > 1e12 or min_eig <= 0:
                print(f"   Adding extra regularization due to poor conditioning")
                extra_reg = max(1e-3, -min_eig + 1e-6)
                regularized += extra_reg * torch.eye(hessian.shape[0], 
                                                   dtype=self.dtype, device=self.device)
        except Exception as e:
            print(f"   Warning: Could not compute eigenvalues: {e}")
            # Fallback: use larger regularization
            regularized += 1e-3 * torch.eye(hessian.shape[0], 
                                          dtype=self.dtype, device=self.device)
        
        return regularized
    
    def line_search(self, log_params: torch.Tensor, direction: torch.Tensor, 
                   current_ll: float, gradient: torch.Tensor,
                   alpha_init: float = 1.0, rho: float = 0.5, c1: float = 1e-4) -> float:
        """Backtracking line search to find appropriate step size."""
        alpha = alpha_init
        
        # Armijo condition: f(x + α*p) ≤ f(x) + c1*α*∇f^T*p
        grad_dot_dir = torch.dot(gradient, direction)
        
        for _ in range(20):  # Maximum 20 backtracking steps
            new_log_params = log_params + alpha * direction
            new_ll = self.evaluate_likelihood(new_log_params)
            
            # Check Armijo condition
            if new_ll >= current_ll + c1 * alpha * grad_dot_dir:
                return alpha
            
            alpha *= rho
        
        return alpha  # Return final alpha even if condition not met
    
    def newton_step(self, log_params: torch.Tensor, gradient_epsilon: float = 1e-7,
                   hessian_epsilon: float = 1e-6, regularization: float = 1e-6) -> Tuple[torch.Tensor, Dict]:
        """Compute a single Newton step."""
        # Compute gradient
        print(f"   Computing gradient...")
        gradient = self.compute_gradient(log_params, gradient_epsilon)
        grad_norm = torch.norm(gradient)
        
        # Compute Hessian
        hessian = self.compute_hessian(log_params, hessian_epsilon)
        
        # Regularize Hessian
        hessian_reg = self.regularize_hessian(hessian, regularization)
        
        # Solve Newton system: H * step = -gradient
        try:
            step = -torch.linalg.solve(hessian_reg, gradient)
            print(f"   Newton step computed successfully")
        except Exception as e:
            print(f"   Newton step failed: {e}, using gradient descent")
            step = -gradient / grad_norm * 0.01  # Fallback to gradient descent
        
        # Line search for step size
        current_ll = self.evaluate_likelihood(log_params)
        alpha = self.line_search(log_params, step, current_ll, gradient)
        
        # Apply step
        new_log_params = log_params + alpha * step
        
        step_info = {
            'gradient_norm': float(grad_norm),
            'step_norm': float(torch.norm(step)),
            'alpha': alpha,
            'hessian_condition': None  # Could add this if needed
        }
        
        return new_log_params, step_info
    
    def optimize(self, max_iterations: int = 20, gradient_tolerance: float = 1e-6,
                parameter_tolerance: float = 1e-8, gradient_epsilon: float = 1e-7,
                hessian_epsilon: float = 1e-6) -> Dict[str, Any]:
        """Run Newton's method optimization."""
        print(f"\n🚀 Starting Newton's Method Optimization")
        print(f"Max iterations: {max_iterations}")
        print(f"Gradient tolerance: {gradient_tolerance}")
        print(f"Parameter tolerance: {parameter_tolerance}")
        
        # Track optimization history
        history = {
            "iterations": [],
            "log_likelihoods": [],
            "deltas": [],
            "taus": [],
            "lambdas": [],
            "gradient_norms": [],
            "step_norms": [],
            "alphas": [],
            "times": []
        }
        
        log_params = self.log_params.data.clone()
        start_time = time.time()
        
        for iteration in range(max_iterations):
            iteration_start = time.time()
            
            print(f"\n📍 Newton Iteration {iteration + 1}/{max_iterations}")
            
            # Compute Newton step
            new_log_params, step_info = self.newton_step(log_params, gradient_epsilon, hessian_epsilon)
            
            # Evaluate new point
            new_likelihood = self.evaluate_likelihood(new_log_params)
            new_params = softplus_transform(new_log_params)
            delta, tau, lambda_param = float(new_params[0]), float(new_params[1]), float(new_params[2])
            
            iteration_time = time.time() - iteration_start
            
            # Update history
            history["iterations"].append(iteration)
            history["log_likelihoods"].append(new_likelihood)
            history["deltas"].append(delta)
            history["taus"].append(tau)
            history["lambdas"].append(lambda_param)
            history["gradient_norms"].append(step_info['gradient_norm'])
            history["step_norms"].append(step_info['step_norm'])
            history["alphas"].append(step_info['alpha'])
            history["times"].append(iteration_time)
            
            print(f"   Result: LL={new_likelihood:.6f}, δ={delta:.6f}, τ={tau:.6f}, λ={lambda_param:.6f}")
            print(f"   Gradient norm: {step_info['gradient_norm']:.2e}, Step size: {step_info['alpha']:.2e}")
            print(f"   Iteration time: {iteration_time:.2f}s")
            
            # Check convergence criteria
            gradient_converged = step_info['gradient_norm'] < gradient_tolerance
            
            if iteration > 0:
                param_change = torch.norm(new_log_params - log_params)
                param_converged = param_change < parameter_tolerance
                print(f"   Parameter change: {param_change:.2e}")
            else:
                param_converged = False
            
            # Update for next iteration
            log_params = new_log_params
            self.log_params.data = log_params
            
            # Check convergence
            if gradient_converged and param_converged:
                print(f"✅ Converged! Gradient norm and parameter change below tolerance")
                break
            elif gradient_converged:
                print(f"✅ Gradient converged (norm < {gradient_tolerance})")
                break
        
        total_time = time.time() - start_time
        final_params = self.get_current_params()
        final_likelihood = self.evaluate_likelihood()
        
        print(f"\n✅ Newton's Method Complete!")
        print(f"Final parameters: δ={final_params[0]:.6f}, τ={final_params[1]:.6f}, λ={final_params[2]:.6f}")
        print(f"Final log-likelihood: {final_likelihood:.6f}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Iterations: {iteration + 1}")
        
        return {
            "final_params": final_params,
            "final_log_likelihood": final_likelihood,
            "total_time": total_time,
            "iterations_run": iteration + 1,
            "converged": gradient_converged,
            "history": history
        }

def test_newton_vs_gradient_descent():
    """Compare Newton's method with gradient descent."""
    print("🆚 Newton's Method vs Gradient Descent Comparison")
    print("=" * 60)
    
    # Test Newton's method
    print(f"\n🔺 Testing Newton's Method:")
    newton_optimizer = NewtonCCPOptimizer(
        "test_trees_1/sp.nwk", "test_trees_1/g.nwk",
        initial_params=(0.1, 0.1, 0.1),
        device=torch.device("cpu"),
        dtype=torch.float64
    )
    
    newton_result = newton_optimizer.optimize(
        max_iterations=10,
        gradient_tolerance=1e-6,
        parameter_tolerance=1e-8
    )
    
    # Test gradient descent for comparison  
    print(f"\n📉 Testing Gradient Descent:")
    gd_optimizer = FiniteDiffCCPOptimizer(
        "test_trees_1/sp.nwk", "test_trees_1/g.nwk",
        initial_params=(0.1, 0.1, 0.1),
        device=torch.device("cpu"),
        dtype=torch.float64
    )
    
    gd_result = gd_optimizer.optimize(
        lr=0.002,
        epochs=50,
        epsilon=1e-7,
        early_stopping_patience=15
    )
    
    # Comparison
    print(f"\n📊 COMPARISON RESULTS:")
    print(f"{'Method':<20} {'Iterations':<12} {'Time':<10} {'Final LL':<12} {'Final δ':<10} {'Final τ':<10} {'Final λ':<10}")
    print("-" * 80)
    print(f"{'Newton':<20} {newton_result['iterations_run']:<12} {newton_result['total_time']:<10.2f} "
          f"{newton_result['final_log_likelihood']:<12.6f} {newton_result['final_params'][0]:<10.6f} "
          f"{newton_result['final_params'][1]:<10.6f} {newton_result['final_params'][2]:<10.6f}")
    print(f"{'Gradient Descent':<20} {gd_result['epochs_run']:<12} {gd_result['total_time']:<10.2f} "
          f"{gd_result['best_log_likelihood']:<12.6f} {gd_result['final_params'][0]:<10.6f} "
          f"{gd_result['final_params'][1]:<10.6f} {gd_result['final_params'][2]:<10.6f}")
    
    # Convergence analysis
    newton_improvement = newton_result['final_log_likelihood'] - (-6.45)
    gd_improvement = gd_result['best_log_likelihood'] - (-6.45)
    
    print(f"\n🎯 CONVERGENCE ANALYSIS:")
    print(f"Newton improvement: {newton_improvement:.6f} in {newton_result['iterations_run']} iterations")
    print(f"Gradient descent improvement: {gd_improvement:.6f} in {gd_result['epochs_run']} iterations")
    print(f"Newton efficiency: {newton_improvement/newton_result['iterations_run']:.6f} per iteration")
    print(f"GD efficiency: {gd_improvement/gd_result['epochs_run']:.6f} per iteration")
    
    if newton_result['iterations_run'] < gd_result['epochs_run']:
        speedup = gd_result['epochs_run'] / newton_result['iterations_run']
        print(f"🚀 Newton's method converged {speedup:.1f}x faster!")
    
    return newton_result, gd_result

if __name__ == "__main__":
    newton_result, gd_result = test_newton_vs_gradient_descent()