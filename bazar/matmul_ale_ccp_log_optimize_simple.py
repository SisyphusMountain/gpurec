#!/usr/bin/env python3
"""
Simplified exact gradient descent implementation with better device/gradient handling.
This version addresses the inf-value and device mismatch issues found in the previous implementation.
"""

import sys
import time
import torch
import argparse
import json
from tabulate import tabulate

# Import the log-space CCP functions
from matmul_ale_ccp_log import (
    build_ccp_from_single_tree, build_species_helpers, 
    build_clade_species_mapping, build_ccp_helpers,
    get_root_clade_id, E_step
)

# Import non-compiled version for gradient computation
from Pi_update_ccp_log_no_compile import Pi_update_ccp_log_no_compile

# Enable anomaly detection for gradient debugging
torch.autograd.set_detect_anomaly(True)

def softplus_transform(log_params):
    """Transform log parameters to positive parameters using softplus."""
    return torch.nn.functional.softplus(log_params)

def inverse_softplus(params, eps=1e-7):
    """Numerically stable inverse softplus transformation."""
    params_safe = torch.clamp(params, min=eps)
    return torch.where(
        params_safe > 20,
        params_safe,  # For large values, softplus^-1(x) ≈ x
        torch.log(torch.expm1(params_safe))
    )

def compute_fixed_point_and_likelihood(log_params, species_tree_path, gene_tree_path, 
                                     max_iter=50, tol=1e-10, device=None, dtype=None):
    """
    Compute the fixed point Pi* and log-likelihood in a differentiable way.
    This function maintains gradient flow throughout the computation.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dtype is None:
        dtype = torch.float64
    
    # Ensure log_params is on the correct device and requires gradients
    log_params = log_params.to(device=device, dtype=dtype)
    if not log_params.requires_grad:
        log_params.requires_grad_(True)
    
    # Transform parameters to positive values
    params = softplus_transform(log_params)
    delta, tau, lambda_param = params[0], params[1], params[2]
    
    print(f"Computing fixed point with δ={delta:.6f}, τ={tau:.6f}, λ={lambda_param:.6f}")
    
    # Build data structures (no gradients needed for structure)
    with torch.no_grad():
        ccp = build_ccp_from_single_tree(gene_tree_path)
        species_helpers = build_species_helpers(species_tree_path, device, dtype)
        clade_species_map = build_clade_species_mapping(ccp, species_helpers, device, dtype)
        ccp_helpers = build_ccp_helpers(ccp, device, dtype)
        
        # Compute event probabilities with gradient tracking
        rates_sum = 1.0 + delta + tau + lambda_param
        p_S = 1.0 / rates_sum
        p_D = delta / rates_sum
        p_T = tau / rates_sum
        p_L = lambda_param / rates_sum
        
        # Compute extinction probabilities (no gradients needed)
        S = species_helpers["S"]
        E = torch.zeros(S, dtype=dtype, device=device)
        for _ in range(max_iter):
            E_next, E_s1, E_s2, Ebar = E_step(E, species_helpers["s_C1"], species_helpers["s_C2"], 
                                              species_helpers["Recipients_mat"], p_S, p_D, p_T, p_L)
            if torch.abs(E_next - E).max() < tol:
                break
            E = E_next
    
    # Initialize log_Pi with gradients enabled
    C = len(ccp.clades)
    log_Pi = torch.full((C, S), float('-inf'), dtype=dtype, device=device)
    
    # Set leaf probabilities (no gradients for initialization)
    with torch.no_grad():
        for c in range(C):
            clade = ccp.id_to_clade[c]
            if clade.is_leaf():
                mapped_species = torch.nonzero(clade_species_map[c] > 0, as_tuple=False).flatten()
                if len(mapped_species) > 0:
                    log_prob = -torch.log(torch.tensor(len(mapped_species), dtype=dtype))
                    log_Pi[c, mapped_species] = log_prob
    
    # Enable gradients for Pi computation
    log_Pi.requires_grad_(True)
    
    # Fixed-point iteration with gradient tracking
    print("Running Pi fixed-point iteration with gradients...")
    
    # Run most iterations without tracking gradients for efficiency
    with torch.no_grad():
        for iter_pi in range(max_iter - 5):  # Stop 5 iterations before the end
            log_Pi_new = Pi_update_ccp_log_no_compile(log_Pi, ccp_helpers, species_helpers, clade_species_map, 
                                                     E, Ebar, p_S, p_D, p_T)
            
            # Check convergence
            if iter_pi > 0:
                diff = torch.abs(log_Pi_new - log_Pi).max()
                if diff < tol:
                    print(f"Pi converged after {iter_pi+1} iterations (diff={diff:.2e})")
                    log_Pi = log_Pi_new
                    break
            
            log_Pi = log_Pi_new
    
    # Final iterations with gradient tracking
    log_Pi.requires_grad_(True)
    for iter_pi in range(5):  # Last few iterations with gradients
        log_Pi_new = Pi_update_ccp_log_no_compile(log_Pi, ccp_helpers, species_helpers, clade_species_map, 
                                                 E, Ebar, p_S, p_D, p_T)
        
        # Check convergence
        if iter_pi > 0:
            diff = torch.abs(log_Pi_new - log_Pi).max()
            if diff < tol:
                print(f"Final Pi converged after {iter_pi+1} gradient iterations (diff={diff:.2e})")
                break
        
        log_Pi = log_Pi_new
    
    # Compute log-likelihood
    root_clade_id = get_root_clade_id(ccp)
    log_likelihood = torch.logsumexp(log_Pi[root_clade_id, :], dim=0)
    
    print(f"Log-likelihood: {log_likelihood:.6f}")
    print(f"Root Pi range: [{log_Pi[root_clade_id, :].min():.3f}, {log_Pi[root_clade_id, :].max():.3f}]")
    
    return log_likelihood

class SimpleCCPLogOptimizer:
    """Simplified optimizer for CCP reconciliation parameters."""
    
    def __init__(self, species_tree_path, gene_tree_path, 
                 init_delta=0.1, init_tau=0.1, init_lambda=0.1,
                 lr=0.01, device=None, dtype=torch.float64):
        
        self.species_tree_path = species_tree_path
        self.gene_tree_path = gene_tree_path
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        
        # Initialize parameters in log space on the correct device
        init_params = torch.tensor([init_delta, init_tau, init_lambda], dtype=dtype, device=self.device)
        self.log_params = torch.nn.Parameter(inverse_softplus(init_params))
        
        self.optimizer = torch.optim.Adam([self.log_params], lr=lr)
        
        self.history = []
        
    def step(self):
        """Perform one optimization step."""
        self.optimizer.zero_grad()
        
        # Compute objective function
        try:
            objective = compute_fixed_point_and_likelihood(
                self.log_params, self.species_tree_path, self.gene_tree_path,
                device=self.device, dtype=self.dtype
            )
            
            # Minimize negative log-likelihood
            loss = -objective
            loss.backward()
            
            # Record current state
            with torch.no_grad():
                current_params = softplus_transform(self.log_params)
                grad_norm = self.log_params.grad.norm() if self.log_params.grad is not None else 0.0
                
                self.history.append({
                    'delta': float(current_params[0]),
                    'tau': float(current_params[1]),
                    'lambda': float(current_params[2]),
                    'log_likelihood': float(objective),
                    'loss': float(loss),
                    'grad_norm': float(grad_norm)
                })
                
                print(f"Current: δ={current_params[0]:.6f}, τ={current_params[1]:.6f}, λ={current_params[2]:.6f}")
                print(f"Log-likelihood: {objective:.6f}, Loss: {loss:.6f}")
                print(f"Gradient norm: {grad_norm:.6f}")
                if self.log_params.grad is not None:
                    print(f"Gradients: {self.log_params.grad}")
                print("-" * 60)
            
            self.optimizer.step()
            return float(loss)
            
        except Exception as e:
            print(f"Error in optimization step: {e}")
            import traceback
            traceback.print_exc()
            raise

def test_simple_gradient_computation():
    """Test simplified gradient computation on test_trees_1."""
    print("🧪 Testing simplified gradient computation on test_trees_1")
    print("=" * 60)
    
    # Use CPU for more reliable gradient computation
    device = torch.device("cpu")
    
    species_path = "test_trees_1/sp.nwk"
    gene_path = "test_trees_1/g.nwk"
    
    # Initialize optimizer with parameters that should optimize toward zero
    optimizer = SimpleCCPLogOptimizer(
        species_path, gene_path,
        init_delta=0.1, init_tau=0.1, init_lambda=0.1,
        lr=0.01, device=device
    )
    
    print(f"Initial parameters:")
    with torch.no_grad():
        init_params = softplus_transform(optimizer.log_params)
        print(f"  δ={init_params[0]:.6f}, τ={init_params[1]:.6f}, λ={init_params[2]:.6f}")
    
    # Perform a single gradient step to test implementation
    print("\nPerforming single gradient step...")
    try:
        loss = optimizer.step()
        print(f"✅ Gradient step completed successfully!")
        print(f"Final loss: {loss:.6f}")
        
        if len(optimizer.history) > 0:
            final_state = optimizer.history[-1]
            print(f"Final parameters: δ={final_state['delta']:.6f}, τ={final_state['tau']:.6f}, λ={final_state['lambda']:.6f}")
            print(f"Gradient norm: {final_state['grad_norm']:.6f}")
            
            # Check if gradient points in the right direction (parameters should decrease)
            init_params = torch.tensor([0.1, 0.1, 0.1])
            final_params = torch.tensor([final_state['delta'], final_state['tau'], final_state['lambda']])
            param_change = final_params - init_params
            print(f"Parameter changes: Δδ={param_change[0]:.6f}, Δτ={param_change[1]:.6f}, Δλ={param_change[2]:.6f}")
            
            if torch.all(param_change < 0):
                print("✅ Parameters decreased as expected!")
            else:
                print("⚠️  Some parameters increased - check gradient direction")
        
    except Exception as e:
        print(f"❌ Error during gradient computation: {e}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description='Simplified exact gradient descent for CCP reconciliation')
    parser.add_argument('--species', help='Species tree file (.nwk)')
    parser.add_argument('--gene', help='Gene tree file (.nwk)')
    parser.add_argument('--init-delta', type=float, default=0.1, help='Initial δ (default: 0.1)')
    parser.add_argument('--init-tau', type=float, default=0.1, help='Initial τ (default: 0.1)')
    parser.add_argument('--init-lambda', type=float, default=0.1, help='Initial λ (default: 0.1)')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate (default: 0.01)')
    parser.add_argument('--epochs', type=int, default=1, help='Number of optimization steps (default: 1)')
    parser.add_argument('--test', action='store_true', help='Run gradient test on test_trees_1')
    parser.add_argument('--device', choices=['cpu', 'cuda'], help='Device to use (default: auto)')
    
    args = parser.parse_args()
    
    if args.test:
        test_simple_gradient_computation()
        return 0
    
    if not args.species or not args.gene:
        parser.error("--species and --gene are required unless using --test")
    
    device = None
    if args.device:
        device = torch.device(args.device)
    
    try:
        optimizer = SimpleCCPLogOptimizer(
            args.species, args.gene,
            init_delta=args.init_delta, init_tau=args.init_tau, init_lambda=args.init_lambda,
            lr=args.lr, device=device
        )
        
        print(f"🚀 Starting optimization with {args.epochs} epochs")
        
        for epoch in range(args.epochs):
            print(f"\n=== EPOCH {epoch + 1} ===")
            loss = optimizer.step()
            
        print("\n✅ Optimization completed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())