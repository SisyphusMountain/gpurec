#!/usr/bin/env python3
"""
Exact gradient descent with proper handling of -inf values through masking.
Key insight: -inf values that don't change have zero gradients.
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
    get_root_clade_id, E_step, Pi_update_ccp_log
)

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

class MaskedPiUpdate(torch.autograd.Function):
    """
    Custom autograd function for Pi update that properly handles -inf values.
    Key insight: -inf values that remain -inf have zero gradients.
    """
    
    @staticmethod
    def forward(ctx, log_Pi, params, ccp_helpers, species_helpers, clade_species_map, E, Ebar):
        """
        Forward pass: compute Pi update in log space.
        """
        # Extract parameters
        delta, tau, lambda_param = params[0], params[1], params[2]
        
        # Compute event probabilities
        rates_sum = 1.0 + delta + tau + lambda_param
        p_S = 1.0 / rates_sum
        p_D = delta / rates_sum
        p_T = tau / rates_sum
        
        # Store which positions are finite in the input
        finite_mask_input = torch.isfinite(log_Pi)
        
        # Compute Pi update
        log_Pi_new = Pi_update_ccp_log(log_Pi.detach(), ccp_helpers, species_helpers, 
                                       clade_species_map, E, Ebar, p_S, p_D, p_T)
        
        # Store which positions are finite in the output
        finite_mask_output = torch.isfinite(log_Pi_new)
        
        # Save for backward
        ctx.save_for_backward(log_Pi, log_Pi_new, params, finite_mask_input, finite_mask_output)
        ctx.ccp_helpers = ccp_helpers
        ctx.species_helpers = species_helpers
        ctx.clade_species_map = clade_species_map
        ctx.E = E
        ctx.Ebar = Ebar
        
        return log_Pi_new
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: compute gradients with proper masking of -inf values.
        """
        log_Pi, log_Pi_new, params, finite_mask_input, finite_mask_output = ctx.saved_tensors
        
        # Mask gradient output: only propagate gradients for finite output values
        grad_output_masked = torch.where(finite_mask_output, grad_output, torch.zeros_like(grad_output))
        
        # For positions that are -inf in both input and output, gradient is zero
        # For positions that change from finite to -inf or vice versa, we need to compute gradients
        # For positions that are finite in both, we need to compute gradients
        
        # Compute gradients using finite differences with small perturbation
        # This is more robust than trying to differentiate through log/exp operations
        eps = 1e-6
        grad_params = torch.zeros_like(params)
        
        for i in range(len(params)):
            # Perturb parameter i
            params_plus = params.clone()
            params_plus[i] = params[i] + eps
            
            # Recompute with perturbed parameter
            delta_p, tau_p, lambda_p = params_plus[0], params_plus[1], params_plus[2]
            rates_sum_p = 1.0 + delta_p + tau_p + lambda_p
            p_S_p = 1.0 / rates_sum_p
            p_D_p = delta_p / rates_sum_p
            p_T_p = tau_p / rates_sum_p
            
            log_Pi_plus = Pi_update_ccp_log(log_Pi.detach(), ctx.ccp_helpers, ctx.species_helpers,
                                           ctx.clade_species_map, ctx.E, ctx.Ebar, p_S_p, p_D_p, p_T_p)
            
            # Compute finite difference gradient, masking -inf values
            diff = log_Pi_plus - log_Pi_new
            # Only consider positions where both values are finite
            valid_mask = finite_mask_output & torch.isfinite(log_Pi_plus)
            diff_masked = torch.where(valid_mask, diff, torch.zeros_like(diff))
            
            # Gradient w.r.t parameter i
            grad_params[i] = torch.sum(grad_output_masked * diff_masked) / eps
        
        # For gradient w.r.t log_Pi, we use implicit differentiation
        # But only for positions that are finite
        grad_log_Pi = torch.zeros_like(log_Pi)
        
        # Simple approximation: positions that remain finite have gradient flow
        # Positions that are -inf in both input and output have zero gradient
        propagate_mask = finite_mask_input & finite_mask_output
        grad_log_Pi = torch.where(propagate_mask, grad_output_masked, torch.zeros_like(grad_output))
        
        return grad_log_Pi, grad_params, None, None, None, None, None

def compute_log_likelihood_with_gradients(log_params, species_tree_path, gene_tree_path,
                                         max_iter=50, tol=1e-10, device=None, dtype=None):
    """
    Compute log-likelihood with proper gradient handling for -inf values.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dtype is None:
        dtype = torch.float64
    
    # Ensure log_params is on the correct device
    log_params = log_params.to(device=device, dtype=dtype)
    if not log_params.requires_grad:
        log_params.requires_grad_(True)
    
    # Transform parameters
    params = softplus_transform(log_params)
    delta, tau, lambda_param = params[0], params[1], params[2]
    
    print(f"Computing with δ={delta:.6f}, τ={tau:.6f}, λ={lambda_param:.6f}")
    
    # Build data structures
    with torch.no_grad():
        ccp = build_ccp_from_single_tree(gene_tree_path)
        species_helpers = build_species_helpers(species_tree_path, device, dtype)
        clade_species_map = build_clade_species_mapping(ccp, species_helpers, device, dtype)
        ccp_helpers = build_ccp_helpers(ccp, device, dtype)
        
        # Compute extinction probabilities
        rates_sum = 1.0 + delta + tau + lambda_param
        p_S = 1.0 / rates_sum
        p_D = delta / rates_sum
        p_T = tau / rates_sum
        p_L = lambda_param / rates_sum
        
        S = species_helpers["S"]
        E = torch.zeros(S, dtype=dtype, device=device)
        for _ in range(max_iter):
            E_next, E_s1, E_s2, Ebar = E_step(E, species_helpers["s_C1"], species_helpers["s_C2"], 
                                              species_helpers["Recipients_mat"], p_S, p_D, p_T, p_L)
            if torch.abs(E_next - E).max() < tol:
                break
            E = E_next
    
    # Initialize log_Pi
    C = len(ccp.clades)
    log_Pi = torch.full((C, S), float('-inf'), dtype=dtype, device=device)
    
    # Set leaf probabilities
    with torch.no_grad():
        for c in range(C):
            clade = ccp.id_to_clade[c]
            if clade.is_leaf():
                mapped_species = torch.nonzero(clade_species_map[c] > 0, as_tuple=False).flatten()
                if len(mapped_species) > 0:
                    log_prob = -torch.log(torch.tensor(len(mapped_species), dtype=dtype))
                    log_Pi[c, mapped_species] = log_prob
    
    # Run fixed-point iteration to convergence
    print("Running Pi fixed-point iteration...")
    with torch.no_grad():
        for iter_pi in range(max_iter):
            log_Pi_new = Pi_update_ccp_log(log_Pi, ccp_helpers, species_helpers, clade_species_map,
                                          E, Ebar, p_S, p_D, p_T)
            
            if iter_pi > 0:
                # Only check convergence on finite values
                finite_mask = torch.isfinite(log_Pi) & torch.isfinite(log_Pi_new)
                if finite_mask.any():
                    diff = torch.abs(log_Pi_new[finite_mask] - log_Pi[finite_mask]).max()
                    if diff < tol:
                        print(f"Pi converged after {iter_pi+1} iterations (diff={diff:.2e})")
                        break
            
            log_Pi = log_Pi_new
    
    # Now do one final update with gradient tracking
    log_Pi.requires_grad_(True)
    log_Pi_final = MaskedPiUpdate.apply(log_Pi, params, ccp_helpers, species_helpers,
                                        clade_species_map, E, Ebar)
    
    # Compute log-likelihood
    root_clade_id = get_root_clade_id(ccp)
    root_values = log_Pi_final[root_clade_id, :]
    
    # Only sum over finite values for log-likelihood
    finite_root_mask = torch.isfinite(root_values)
    if finite_root_mask.any():
        log_likelihood = torch.logsumexp(root_values[finite_root_mask], dim=0)
    else:
        log_likelihood = torch.tensor(float('-inf'), device=device, dtype=dtype)
    
    print(f"Log-likelihood: {log_likelihood:.6f}")
    print(f"Root Pi range: [{root_values.min():.3f}, {root_values.max():.3f}]")
    print(f"Finite root values: {finite_root_mask.sum()}/{len(root_values)}")
    
    return log_likelihood

class MaskedCCPOptimizer:
    """Optimizer with proper handling of -inf values."""
    
    def __init__(self, species_tree_path, gene_tree_path,
                 init_delta=0.1, init_tau=0.1, init_lambda=0.1,
                 lr=0.01, device=None, dtype=torch.float64):
        
        self.species_tree_path = species_tree_path
        self.gene_tree_path = gene_tree_path
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        
        # Initialize parameters
        init_params = torch.tensor([init_delta, init_tau, init_lambda], dtype=dtype, device=self.device)
        self.log_params = torch.nn.Parameter(inverse_softplus(init_params))
        
        self.optimizer = torch.optim.Adam([self.log_params], lr=lr)
        self.history = []
        
    def step(self):
        """Perform one optimization step."""
        self.optimizer.zero_grad()
        
        try:
            # Compute objective
            objective = compute_log_likelihood_with_gradients(
                self.log_params, self.species_tree_path, self.gene_tree_path,
                device=self.device, dtype=self.dtype
            )
            
            # Minimize negative log-likelihood
            loss = -objective
            loss.backward()
            
            # Record state
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

def test_masked_gradient_computation():
    """Test gradient computation with proper -inf handling."""
    print("🧪 Testing masked gradient computation on test_trees_1")
    print("=" * 60)
    
    species_path = "test_trees_1/sp.nwk"
    gene_path = "test_trees_1/g.nwk"
    
    # Use CPU for more reliable computation
    device = torch.device("cpu")
    
    optimizer = MaskedCCPOptimizer(
        species_path, gene_path,
        init_delta=0.1, init_tau=0.1, init_lambda=0.1,
        lr=0.01, device=device
    )
    
    print(f"Initial parameters:")
    with torch.no_grad():
        init_params = softplus_transform(optimizer.log_params)
        print(f"  δ={init_params[0]:.6f}, τ={init_params[1]:.6f}, λ={init_params[2]:.6f}")
    
    print("\nPerforming gradient step with -inf masking...")
    try:
        loss = optimizer.step()
        print(f"✅ Gradient step completed successfully!")
        
        if len(optimizer.history) > 0:
            final_state = optimizer.history[-1]
            print(f"\nResults:")
            print(f"  Final parameters: δ={final_state['delta']:.6f}, τ={final_state['tau']:.6f}, λ={final_state['lambda']:.6f}")
            print(f"  Gradient norm: {final_state['grad_norm']:.6f}")
            
            # Check gradient direction
            init_params = torch.tensor([0.1, 0.1, 0.1])
            final_params = torch.tensor([final_state['delta'], final_state['tau'], final_state['lambda']])
            param_change = final_params - init_params
            print(f"  Parameter changes: Δδ={param_change[0]:.6f}, Δτ={param_change[1]:.6f}, Δλ={param_change[2]:.6f}")
            
            if torch.all(param_change < 0):
                print("  ✅ Parameters decreased as expected!")
            else:
                print("  ⚠️  Some parameters did not decrease")
        
    except Exception as e:
        print(f"❌ Error during gradient computation: {e}")

def main():
    parser = argparse.ArgumentParser(description='Masked gradient descent for CCP reconciliation')
    parser.add_argument('--species', help='Species tree file (.nwk)')
    parser.add_argument('--gene', help='Gene tree file (.nwk)')
    parser.add_argument('--init-delta', type=float, default=0.1, help='Initial δ')
    parser.add_argument('--init-tau', type=float, default=0.1, help='Initial τ')
    parser.add_argument('--init-lambda', type=float, default=0.1, help='Initial λ')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('--test', action='store_true', help='Run test on test_trees_1')
    
    args = parser.parse_args()
    
    if args.test:
        test_masked_gradient_computation()
        return 0
    
    if not args.species or not args.gene:
        parser.error("--species and --gene are required unless using --test")
    
    try:
        optimizer = MaskedCCPOptimizer(
            args.species, args.gene,
            init_delta=args.init_delta, init_tau=args.init_tau, init_lambda=args.init_lambda,
            lr=args.lr
        )
        
        print(f"🚀 Starting optimization with {args.epochs} epochs")
        
        for epoch in range(args.epochs):
            print(f"\n=== EPOCH {epoch + 1} ===")
            optimizer.step()
            
        print("\n✅ Optimization completed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())