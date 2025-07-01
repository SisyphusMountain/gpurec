#!/usr/bin/env python3
"""
Finite difference version of gradient descent optimization.
This is a robust fallback that can handle -inf values in log-space computation.
"""

import torch
import torch.nn.functional as F
import time
import numpy as np
from typing import Dict, Tuple, Optional, Any

# Import existing CCP functions
from matmul_ale_ccp_log import (
    build_ccp_from_single_tree, build_species_helpers,
    build_clade_species_mapping, build_ccp_helpers,
    get_root_clade_id, reconcile_ccp_log
)

def softplus_transform(log_params: torch.Tensor) -> torch.Tensor:
    """Transform unconstrained parameters to positive parameters using softplus."""
    return F.softplus(log_params)

def inverse_softplus_transform(params: torch.Tensor) -> torch.Tensor:
    """Inverse softplus for numerical stability."""
    return torch.where(
        params > 20.0,  
        torch.log(params) + torch.log1p(-torch.exp(-params)),
        torch.log(torch.expm1(params))
    )

def compute_log_likelihood(species_tree_path: str, gene_tree_path: str, 
                          delta: float, tau: float, lambda_param: float,
                          device: torch.device, dtype: torch.dtype) -> float:
    """Compute log-likelihood for given parameters."""
    try:
        result = reconcile_ccp_log(
            species_tree_path, gene_tree_path,
            delta=delta, tau=tau, lambda_param=lambda_param,
            iters=50, device=device, dtype=dtype
        )
        return result['log_likelihood']
    except Exception:
        # Return very low likelihood if computation fails
        return -1e10

def finite_difference_gradients(species_tree_path: str, gene_tree_path: str,
                               log_params: torch.Tensor, epsilon: float = 1e-6,
                               device: torch.device = None, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Compute gradients using finite differences."""
    if device is None:
        device = torch.device("cpu")
    
    # Transform to positive parameters
    params = softplus_transform(log_params)
    delta, tau, lambda_param = float(params[0]), float(params[1]), float(params[2])
    
    # Compute baseline likelihood
    f0 = compute_log_likelihood(species_tree_path, gene_tree_path, 
                               delta, tau, lambda_param, device, dtype)
    
    # Compute gradients w.r.t. each parameter
    gradients = torch.zeros_like(log_params)
    
    for i in range(len(log_params)):
        # Forward difference
        log_params_plus = log_params.clone()
        log_params_plus[i] += epsilon
        
        params_plus = softplus_transform(log_params_plus)
        delta_plus = float(params_plus[0])
        tau_plus = float(params_plus[1])
        lambda_plus = float(params_plus[2])
        
        f_plus = compute_log_likelihood(species_tree_path, gene_tree_path,
                                       delta_plus, tau_plus, lambda_plus, device, dtype)
        
        # Finite difference gradient
        gradients[i] = (f_plus - f0) / epsilon
    
    # Handle NaN/inf values
    gradients = torch.where(torch.isfinite(gradients), gradients, torch.zeros_like(gradients))
    
    return gradients

class FiniteDiffCCPOptimizer:
    """CCP optimizer using finite difference gradients."""
    
    def __init__(self, species_tree_path: str, gene_tree_path: str,
                 initial_params: Optional[Tuple[float, float, float]] = None,
                 device: Optional[torch.device] = None, dtype: torch.dtype = torch.float64):
        """Initialize the finite difference optimizer."""
        self.species_tree_path = species_tree_path
        self.gene_tree_path = gene_tree_path
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        
        # Initialize parameters
        if initial_params is None:
            initial_params = (0.1, 0.1, 0.1)  # (delta, tau, lambda)
        
        params_tensor = torch.tensor(initial_params, dtype=dtype, device=self.device)
        self.log_params = torch.nn.Parameter(inverse_softplus_transform(params_tensor))
        
        print(f"✅ Finite Diff Optimizer initialized:")
        print(f"   Device: {self.device}")
        print(f"   Initial params: δ={initial_params[0]:.3f}, τ={initial_params[1]:.3f}, λ={initial_params[2]:.3f}")
    
    def get_current_params(self) -> Tuple[float, float, float]:
        """Get current parameter values (delta, tau, lambda)."""
        with torch.no_grad():
            params = softplus_transform(self.log_params)
            return float(params[0]), float(params[1]), float(params[2])
    
    def evaluate_likelihood(self) -> float:
        """Evaluate likelihood at current parameters."""
        with torch.no_grad():
            params = softplus_transform(self.log_params)
            delta, tau, lambda_param = float(params[0]), float(params[1]), float(params[2])
            return compute_log_likelihood(self.species_tree_path, self.gene_tree_path,
                                        delta, tau, lambda_param, self.device, self.dtype)
    
    def compute_gradients(self, epsilon: float = 1e-6) -> torch.Tensor:
        """Compute finite difference gradients."""
        with torch.no_grad():
            return finite_difference_gradients(
                self.species_tree_path, self.gene_tree_path,
                self.log_params.data, epsilon, self.device, self.dtype
            )
    
    def optimize(self, lr: float = 0.01, epochs: int = 100, 
                early_stopping_patience: int = 10, min_improvement: float = 1e-6,
                epsilon: float = 1e-6) -> Dict[str, Any]:
        """Run optimization using finite difference gradients."""
        print(f"\n🚀 Starting finite difference optimization")
        print(f"Learning rate: {lr}, Max epochs: {epochs}, Epsilon: {epsilon}")
        
        # Track optimization history
        history = {
            "epochs": [],
            "log_likelihoods": [],
            "deltas": [],
            "taus": [],
            "lambdas": [],
            "times": []
        }
        
        best_log_likelihood = float('-inf')
        patience_counter = 0
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Compute current likelihood and gradients
            current_likelihood = self.evaluate_likelihood()
            gradients = self.compute_gradients(epsilon=epsilon)
            
            # Check if gradients are meaningful
            grad_norm = torch.norm(gradients)
            if grad_norm < 1e-15:
                print(f"Epoch {epoch}: Zero gradients, stopping optimization")
                break
            
            # Gradient ascent step (we want to maximize likelihood)
            with torch.no_grad():
                self.log_params.data += lr * gradients
            
            current_params = self.get_current_params()
            epoch_time = time.time() - epoch_start
            
            # Update history
            history["epochs"].append(epoch)
            history["log_likelihoods"].append(current_likelihood)
            history["deltas"].append(current_params[0])
            history["taus"].append(current_params[1])
            history["lambdas"].append(current_params[2])
            history["times"].append(epoch_time)
            
            # Check for improvement
            if current_likelihood > best_log_likelihood + min_improvement:
                best_log_likelihood = current_likelihood
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Progress reporting
            if epoch % 5 == 0 or epoch < 10:
                print(f"Epoch {epoch:3d}: LL={current_likelihood:8.3f}, "
                      f"δ={current_params[0]:.4f}, τ={current_params[1]:.4f}, λ={current_params[2]:.4f}, "
                      f"grad_norm={grad_norm:.2e}, t={epoch_time:.2f}s")
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping: no improvement for {early_stopping_patience} epochs")
                break
        
        total_time = time.time() - start_time
        final_params = self.get_current_params()
        
        print(f"\n✅ Optimization complete!")
        print(f"Best log-likelihood: {best_log_likelihood:.6f}")
        print(f"Final parameters: δ={final_params[0]:.6f}, τ={final_params[1]:.6f}, λ={final_params[2]:.6f}")
        print(f"Total time: {total_time:.2f}s")
        
        return {
            "best_log_likelihood": best_log_likelihood,
            "final_params": final_params,
            "total_time": total_time,
            "epochs_run": epoch + 1,
            "history": history
        }

def test_finite_diff_optimization():
    """Test the finite difference optimization."""
    print("🧪 Testing Finite Difference Optimization")
    print("=" * 50)
    
    # Test on simple case
    optimizer = FiniteDiffCCPOptimizer(
        "test_trees_1/sp.nwk", "test_trees_1/g.nwk",
        initial_params=(0.1, 0.1, 0.1),
        device=torch.device("cpu"),
        dtype=torch.float64
    )
    
    # Evaluate initial likelihood
    initial_likelihood = optimizer.evaluate_likelihood()
    print(f"Initial likelihood: {initial_likelihood:.6f}")
    
    # Test gradient computation
    print("\n🔍 Testing gradient computation...")
    gradients = optimizer.compute_gradients(epsilon=1e-6)
    print(f"Gradients: {gradients}")
    print(f"Gradient norm: {torch.norm(gradients):.6f}")
    
    # Run optimization
    result = optimizer.optimize(lr=0.001, epochs=50, epsilon=1e-6, early_stopping_patience=10)
    
    return result

if __name__ == "__main__":
    result = test_finite_diff_optimization()
    print(f"\n🎯 Final Results:")
    print(f"   Improvement: {result['best_log_likelihood'] - (-6.45):.6f}")
    print(f"   Final params: δ={result['final_params'][0]:.6f}, τ={result['final_params'][1]:.6f}, λ={result['final_params'][2]:.6f}")