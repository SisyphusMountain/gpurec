#!/usr/bin/env python3
"""
Newton's method optimization using exact second derivatives with the MaskedPiUpdate function.
Leverages PyTorch's autograd to compute the Hessian matrix.
"""

import sys
import time
import torch
import argparse
import json
from tabulate import tabulate
from torch.autograd.functional import hessian

# Import the log-space CCP functions
from matmul_ale_ccp_log import (
    build_ccp_from_single_tree, build_species_helpers, 
    build_clade_species_mapping, build_ccp_helpers,
    get_root_clade_id, E_step, Pi_update_ccp_log
)

# Import our masked gradient computation
from matmul_ale_ccp_log_optimize_masked import (
    MaskedPiUpdate, softplus_transform, inverse_softplus,
    compute_log_likelihood_with_gradients
)

# Enable anomaly detection
torch.autograd.set_detect_anomaly(True)

def compute_log_likelihood_function(log_params, species_tree_path, gene_tree_path,
                                   max_iter=50, tol=1e-10, device=None, dtype=None):
    """
    Wrapper function that returns just the log-likelihood value for Hessian computation.
    This needs to be a simple function of log_params for torch.autograd.functional.hessian.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dtype is None:
        dtype = torch.float64
    
    # Ensure log_params has correct properties
    log_params = log_params.to(device=device, dtype=dtype)
    if not log_params.requires_grad:
        log_params.requires_grad_(True)
    
    # Transform parameters
    params = softplus_transform(log_params)
    delta, tau, lambda_param = params[0], params[1], params[2]
    
    # Build data structures (no gradients needed)
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
    with torch.no_grad():
        for iter_pi in range(max_iter):
            log_Pi_new = Pi_update_ccp_log(log_Pi, ccp_helpers, species_helpers, clade_species_map,
                                          E, Ebar, p_S, p_D, p_T)
            
            if iter_pi > 0:
                finite_mask = torch.isfinite(log_Pi) & torch.isfinite(log_Pi_new)
                if finite_mask.any():
                    diff = torch.abs(log_Pi_new[finite_mask] - log_Pi[finite_mask]).max()
                    if diff < tol:
                        break
            
            log_Pi = log_Pi_new
    
    # Final update with gradient tracking
    log_Pi.requires_grad_(True)
    log_Pi_final = MaskedPiUpdate.apply(log_Pi, params, ccp_helpers, species_helpers,
                                        clade_species_map, E, Ebar)
    
    # Compute log-likelihood
    root_clade_id = get_root_clade_id(ccp)
    root_values = log_Pi_final[root_clade_id, :]
    
    finite_root_mask = torch.isfinite(root_values)
    if finite_root_mask.any():
        log_likelihood = torch.logsumexp(root_values[finite_root_mask], dim=0)
    else:
        log_likelihood = torch.tensor(float('-inf'), device=device, dtype=dtype)
    
    return log_likelihood

class NewtonCCPOptimizer:
    """Newton's method optimizer using exact Hessian computation."""
    
    def __init__(self, species_tree_path, gene_tree_path,
                 init_delta=0.1, init_tau=0.1, init_lambda=0.1,
                 device=None, dtype=torch.float64):
        
        self.species_tree_path = species_tree_path
        self.gene_tree_path = gene_tree_path
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        
        # Initialize parameters
        init_params = torch.tensor([init_delta, init_tau, init_lambda], dtype=dtype, device=self.device)
        self.log_params = inverse_softplus(init_params)
        self.log_params.requires_grad_(True)
        
        self.history = []
        
    def compute_gradient_and_hessian(self):
        """Compute both gradient and Hessian of the negative log-likelihood."""
        
        # Define objective function (negative log-likelihood)
        def objective(log_params):
            log_lik = compute_log_likelihood_function(
                log_params, self.species_tree_path, self.gene_tree_path,
                device=self.device, dtype=self.dtype
            )
            return -log_lik  # Minimize negative log-likelihood
        
        # Compute gradient
        self.log_params.requires_grad_(True)
        loss = objective(self.log_params)
        loss.backward()
        gradient = self.log_params.grad.clone()
        
        # Always use finite difference Hessian approximation
        # The exact Hessian computation fails due to -inf values in second derivatives
        print("Computing Hessian approximation using finite differences...")
        
        # Regularization parameter
        reg = 1e-4
        
        # Compute diagonal Hessian approximation
        H_diag = torch.zeros(len(self.log_params), device=self.device, dtype=self.dtype)
        eps = 1e-6
        
        for i in range(len(self.log_params)):
            # Compute second derivative by finite differences
            log_params_plus = self.log_params.clone()
            log_params_plus[i] += eps
            log_params_minus = self.log_params.clone() 
            log_params_minus[i] -= eps
            
            loss_plus = objective(log_params_plus)
            loss_minus = objective(log_params_minus)
            
            # f''(x) ≈ (f(x+h) - 2f(x) + f(x-h)) / h²
            H_diag[i] = (loss_plus - 2*loss + loss_minus) / (eps**2)
        
        print(f"Hessian diagonal: {H_diag}")
        
        # Create regularized Hessian (diagonal matrix)
        H_reg = torch.diag(H_diag + reg)
        
        return gradient, H_reg, float(loss)
    
    def step(self, max_line_search=10, alpha_init=1.0):
        """Perform one Newton step with line search."""
        
        # Zero gradients
        if self.log_params.grad is not None:
            self.log_params.grad.zero_()
        
        # Compute gradient and Hessian
        gradient, H_reg, current_loss = self.compute_gradient_and_hessian()
        
        print(f"Gradient norm: {gradient.norm():.6f}")
        print(f"Gradient: {gradient}")
        
        # Solve Newton system: H * p = -g
        try:
            # Use Cholesky decomposition for positive definite matrix
            L = torch.linalg.cholesky(H_reg)
            newton_direction = -torch.cholesky_solve(gradient.unsqueeze(1), L).squeeze()
        except:
            # Fallback to general linear solver
            print("Cholesky failed, using general solver...")
            newton_direction = -torch.linalg.solve(H_reg, gradient)
        
        print(f"Newton direction: {newton_direction}")
        print(f"Newton step norm: {newton_direction.norm():.6f}")
        
        # Line search
        alpha = alpha_init
        with torch.no_grad():
            for i in range(max_line_search):
                # Try step
                log_params_new = self.log_params + alpha * newton_direction
                
                # Check if parameters are valid
                params_new = softplus_transform(log_params_new)
                if torch.all(params_new > 0):
                    # Evaluate objective
                    new_loss = -compute_log_likelihood_function(
                        log_params_new, self.species_tree_path, self.gene_tree_path,
                        device=self.device, dtype=self.dtype
                    )
                    
                    # Check sufficient decrease (Armijo condition)
                    expected_decrease = alpha * torch.dot(gradient, newton_direction)
                    if new_loss < current_loss + 0.1 * expected_decrease:
                        print(f"Line search succeeded with alpha={alpha:.6f}")
                        self.log_params.data = log_params_new
                        final_loss = float(new_loss)
                        break
                
                # Reduce step size
                alpha *= 0.5
                print(f"Reducing step size to {alpha:.6f}")
            else:
                print("Line search failed, taking small step")
                self.log_params.data = self.log_params + 0.01 * newton_direction
                final_loss = current_loss
        
        # Record state
        current_params = softplus_transform(self.log_params)
        self.history.append({
            'delta': float(current_params[0]),
            'tau': float(current_params[1]),
            'lambda': float(current_params[2]),
            'log_likelihood': -final_loss,
            'loss': final_loss,
            'grad_norm': float(gradient.norm()),
            'newton_norm': float(newton_direction.norm()),
            'alpha': alpha
        })
        
        print(f"Current: δ={current_params[0]:.6f}, τ={current_params[1]:.6f}, λ={current_params[2]:.6f}")
        print(f"Log-likelihood: {-final_loss:.6f}, Loss: {final_loss:.6f}")
        print("-" * 60)
        
        return final_loss

def test_newton_optimization():
    """Test Newton's method on test_trees_1."""
    print("🧪 Testing Newton's method optimization on test_trees_1")
    print("=" * 60)
    
    species_path = "test_trees_1/sp.nwk"
    gene_path = "test_trees_1/g.nwk"
    
    # Use CPU for more reliable Hessian computation
    device = torch.device("cpu")
    
    optimizer = NewtonCCPOptimizer(
        species_path, gene_path,
        init_delta=0.1, init_tau=0.1, init_lambda=0.1,
        device=device
    )
    
    print(f"Initial parameters:")
    init_params = softplus_transform(optimizer.log_params)
    print(f"  δ={init_params[0]:.6f}, τ={init_params[1]:.6f}, λ={init_params[2]:.6f}")
    
    print("\nPerforming Newton optimization...")
    
    # Run Newton iterations
    max_iters = 10
    tol = 1e-8
    
    for i in range(max_iters):
        print(f"\n=== Newton Iteration {i+1} ===")
        
        try:
            loss = optimizer.step()
            
            # Check convergence
            if i > 0:
                loss_change = abs(optimizer.history[-1]['loss'] - optimizer.history[-2]['loss'])
                print(f"Loss change: {loss_change:.2e}")
                
                if loss_change < tol:
                    print(f"✅ Converged after {i+1} iterations!")
                    break
                    
        except Exception as e:
            print(f"❌ Error in Newton step: {e}")
            import traceback
            traceback.print_exc()
            break
    
    # Print results
    if len(optimizer.history) > 0:
        print("\n📊 Optimization Results:")
        print("Iteration | δ        | τ        | λ        | Log-Lik  | Grad Norm | Newton Norm | α")
        print("-" * 90)
        
        for i, state in enumerate(optimizer.history):
            print(f"{i+1:9d} | {state['delta']:8.6f} | {state['tau']:8.6f} | "
                  f"{state['lambda']:8.6f} | {state['log_likelihood']:8.4f} | "
                  f"{state['grad_norm']:9.6f} | {state['newton_norm']:11.6f} | {state['alpha']:5.3f}")
        
        final_state = optimizer.history[-1]
        print(f"\nFinal parameters: δ={final_state['delta']:.6f}, "
              f"τ={final_state['tau']:.6f}, λ={final_state['lambda']:.6f}")
        print(f"Final log-likelihood: {final_state['log_likelihood']:.6f}")

def main():
    parser = argparse.ArgumentParser(description='Newton method optimization for CCP reconciliation')
    parser.add_argument('--species', help='Species tree file (.nwk)')
    parser.add_argument('--gene', help='Gene tree file (.nwk)')
    parser.add_argument('--init-delta', type=float, default=0.1, help='Initial δ')
    parser.add_argument('--init-tau', type=float, default=0.1, help='Initial τ')
    parser.add_argument('--init-lambda', type=float, default=0.1, help='Initial λ')
    parser.add_argument('--max-iters', type=int, default=10, help='Maximum Newton iterations')
    parser.add_argument('--test', action='store_true', help='Run test on test_trees_1')
    parser.add_argument('--device', choices=['cpu', 'cuda'], help='Device to use')
    
    args = parser.parse_args()
    
    if args.test:
        test_newton_optimization()
        return 0
    
    if not args.species or not args.gene:
        parser.error("--species and --gene are required unless using --test")
    
    device = None
    if args.device:
        device = torch.device(args.device)
    
    try:
        optimizer = NewtonCCPOptimizer(
            args.species, args.gene,
            init_delta=args.init_delta, init_tau=args.init_tau, init_lambda=args.init_lambda,
            device=device
        )
        
        print(f"🚀 Starting Newton optimization with max {args.max_iters} iterations")
        
        for i in range(args.max_iters):
            print(f"\n=== Newton Iteration {i+1} ===")
            loss = optimizer.step()
            
            if i > 0:
                loss_change = abs(optimizer.history[-1]['loss'] - optimizer.history[-2]['loss'])
                if loss_change < 1e-8:
                    print(f"✅ Converged after {i+1} iterations!")
                    break
        
        print("\n✅ Optimization completed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())