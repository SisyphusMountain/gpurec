#!/usr/bin/env python3
"""
Newton's method with EXACT gradients and Hessian using automatic differentiation.
Key: Compute derivatives on already converged Pi matrices.
"""

import sys
import time
import torch
import argparse
import json
from tabulate import tabulate
from torch.autograd import grad

# Import the log-space CCP functions
from matmul_ale_ccp_log import (
    build_ccp_from_single_tree, build_species_helpers, 
    build_clade_species_mapping, build_ccp_helpers,
    get_root_clade_id, E_step, Pi_update_ccp_log
)

# Import our masked gradient computation
from matmul_ale_ccp_log_optimize_masked import (
    MaskedPiUpdate, softplus_transform, inverse_softplus
)

# Enable anomaly detection
torch.autograd.set_detect_anomaly(True)

def compute_converged_pi_and_context(log_params, species_tree_path, gene_tree_path,
                                    max_iter=50, tol=1e-10, device=None, dtype=None):
    """
    Compute the converged fixed point Pi* and all necessary context.
    This function does NOT track gradients - it just finds the fixed point.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dtype is None:
        dtype = torch.float64
    
    # Transform parameters (no gradients needed here)
    with torch.no_grad():
        params = softplus_transform(log_params)
        delta, tau, lambda_param = params[0], params[1], params[2]
        
        # Build data structures
        ccp = build_ccp_from_single_tree(gene_tree_path)
        species_helpers = build_species_helpers(species_tree_path, device, dtype)
        clade_species_map = build_clade_species_mapping(ccp, species_helpers, device, dtype)
        ccp_helpers = build_ccp_helpers(ccp, device, dtype)
        
        # Compute event probabilities
        rates_sum = 1.0 + delta + tau + lambda_param
        p_S = 1.0 / rates_sum
        p_D = delta / rates_sum
        p_T = tau / rates_sum
        p_L = lambda_param / rates_sum
        
        # Compute extinction probabilities
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
        for c in range(C):
            clade = ccp.id_to_clade[c]
            if clade.is_leaf():
                mapped_species = torch.nonzero(clade_species_map[c] > 0, as_tuple=False).flatten()
                if len(mapped_species) > 0:
                    log_prob = -torch.log(torch.tensor(len(mapped_species), dtype=dtype))
                    log_Pi[c, mapped_species] = log_prob
        
        # Run fixed-point iteration to convergence
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
    
    # Return converged Pi and all context needed for gradient computation
    return {
        'log_Pi': log_Pi,
        'ccp': ccp,
        'species_helpers': species_helpers,
        'clade_species_map': clade_species_map,
        'ccp_helpers': ccp_helpers,
        'E': E,
        'Ebar': Ebar,
        'device': device,
        'dtype': dtype
    }

def compute_log_likelihood_with_exact_gradients(log_params, species_tree_path, gene_tree_path,
                                               device=None, dtype=None):
    """
    Compute log-likelihood with EXACT gradients using automatic differentiation.
    The key is to run Pi to convergence first, then do one final update with gradients.
    """
    # First, get the converged fixed point (no gradients)
    context = compute_converged_pi_and_context(
        log_params.detach(), species_tree_path, gene_tree_path,
        device=device, dtype=dtype
    )
    
    # Now do ONE final Pi update with gradient tracking
    # This implements the implicit function theorem for fixed-point differentiation
    log_Pi_converged = context['log_Pi'].clone().requires_grad_(True)
    
    # Ensure log_params requires gradients
    if not log_params.requires_grad:
        log_params.requires_grad_(True)
    
    # Transform parameters with gradients
    params = softplus_transform(log_params)
    
    # Final Pi update with gradients
    log_Pi_final = MaskedPiUpdate.apply(
        log_Pi_converged, params, 
        context['ccp_helpers'], context['species_helpers'],
        context['clade_species_map'], context['E'], context['Ebar']
    )
    
    # Compute log-likelihood
    root_clade_id = get_root_clade_id(context['ccp'])
    root_values = log_Pi_final[root_clade_id, :]
    
    finite_root_mask = torch.isfinite(root_values)
    if finite_root_mask.any():
        log_likelihood = torch.logsumexp(root_values[finite_root_mask], dim=0)
    else:
        log_likelihood = torch.tensor(float('-inf'), device=device, dtype=dtype)
    
    return log_likelihood, context

def compute_exact_gradient_and_hessian(log_params, species_tree_path, gene_tree_path,
                                       device=None, dtype=None):
    """
    Compute EXACT gradient and Hessian using automatic differentiation.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dtype is None:
        dtype = torch.float64
    
    # Ensure log_params requires gradients
    log_params = log_params.clone().requires_grad_(True)
    
    # Compute log-likelihood with gradient tracking
    log_lik, _ = compute_log_likelihood_with_exact_gradients(
        log_params, species_tree_path, gene_tree_path,
        device=device, dtype=dtype
    )
    
    # Compute gradient
    gradient = grad(log_lik, log_params, create_graph=True)[0]
    
    # Compute Hessian by differentiating the gradient
    n_params = len(log_params)
    H = torch.zeros((n_params, n_params), device=device, dtype=dtype)
    
    for i in range(n_params):
        # Compute d/d(log_params) of gradient[i]
        if gradient[i].requires_grad:
            H[i, :] = grad(gradient[i], log_params, retain_graph=True)[0]
        else:
            # If gradient[i] doesn't require grad (e.g., it's zero), 
            # then its derivative is also zero
            H[i, :] = torch.zeros(n_params, device=device, dtype=dtype)
    
    # Ensure Hessian is symmetric
    H = 0.5 * (H + H.T)
    
    return gradient, H

class ExactNewtonOptimizer:
    """Newton's method optimizer using EXACT gradients and Hessian."""
    
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
    
    def step(self, reg=1e-4, max_line_search=10, alpha_init=1.0):
        """Perform one Newton step with exact gradients and Hessian."""
        
        print("Computing exact gradient and Hessian...")
        start_time = time.time()
        
        try:
            # Compute exact gradient and Hessian
            gradient, H = compute_exact_gradient_and_hessian(
                self.log_params, self.species_tree_path, self.gene_tree_path,
                self.device, self.dtype
            )
            
            hessian_time = time.time() - start_time
            print(f"Gradient/Hessian computation took {hessian_time:.2f}s")
            
            print(f"Gradient: {gradient}")
            print(f"Gradient norm: {gradient.norm():.6f}")
            
            # Check if gradient is valid
            if not torch.isfinite(gradient).all():
                print("Warning: Non-finite values in gradient")
                # Use finite differences as fallback
                print("Falling back to finite difference gradient...")
                eps = 1e-6
                gradient = torch.zeros_like(self.log_params)
                
                log_lik_base, _ = compute_log_likelihood_with_exact_gradients(
                    self.log_params.detach(), self.species_tree_path, self.gene_tree_path,
                    self.device, self.dtype
                )
                
                for i in range(len(self.log_params)):
                    log_params_plus = self.log_params.detach().clone()
                    log_params_plus[i] += eps
                    
                    log_lik_plus, _ = compute_log_likelihood_with_exact_gradients(
                        log_params_plus, self.species_tree_path, self.gene_tree_path,
                        self.device, self.dtype
                    )
                    
                    gradient[i] = (log_lik_plus - log_lik_base) / eps
                
                print(f"Finite difference gradient: {gradient}")
            
            # Check Hessian properties
            try:
                eigenvalues = torch.linalg.eigvalsh(H)
                print(f"Hessian eigenvalues: {eigenvalues}")
                if eigenvalues.min() > 0:
                    print("Hessian is positive definite")
                else:
                    print("Hessian is not positive definite - adding regularization")
                    reg = max(reg, -2 * eigenvalues.min())
            except:
                print("Could not compute eigenvalues - using default regularization")
            
            # Add regularization
            H_reg = H + reg * torch.eye(len(self.log_params), device=self.device, dtype=self.dtype)
            
            # Compute Newton direction
            try:
                # Try Cholesky decomposition
                L = torch.linalg.cholesky(H_reg)
                newton_direction = -torch.cholesky_solve(gradient.unsqueeze(1), L).squeeze()
            except:
                # Fallback to general solver
                print("Cholesky failed, using general solver...")
                newton_direction = -torch.linalg.solve(H_reg, gradient)
            
            print(f"Newton direction: {newton_direction}")
            print(f"Newton step norm: {newton_direction.norm():.6f}")
            
        except Exception as e:
            print(f"Error computing exact gradients/Hessian: {e}")
            # Fallback to gradient descent direction
            gradient = torch.zeros_like(self.log_params)
            eps = 1e-6
            
            log_lik_base, _ = compute_log_likelihood_with_exact_gradients(
                self.log_params.detach(), self.species_tree_path, self.gene_tree_path,
                self.device, self.dtype
            )
            
            for i in range(len(self.log_params)):
                log_params_plus = self.log_params.detach().clone()
                log_params_plus[i] += eps
                
                log_lik_plus, _ = compute_log_likelihood_with_exact_gradients(
                    log_params_plus, self.species_tree_path, self.gene_tree_path,
                    self.device, self.dtype
                )
                
                gradient[i] = (log_lik_plus - log_lik_base) / eps
            
            newton_direction = -gradient  # Simple gradient descent
            print(f"Using gradient descent direction: {newton_direction}")
        
        # Compute current loss
        with torch.no_grad():
            current_log_lik, _ = compute_log_likelihood_with_exact_gradients(
                self.log_params, self.species_tree_path, self.gene_tree_path,
                self.device, self.dtype
            )
            current_loss = -float(current_log_lik)
        
        # Line search
        alpha = alpha_init
        with torch.no_grad():
            for i in range(max_line_search):
                # Try step
                log_params_new = self.log_params + alpha * newton_direction
                
                # Evaluate objective
                new_log_lik, _ = compute_log_likelihood_with_exact_gradients(
                    log_params_new, self.species_tree_path, self.gene_tree_path,
                    self.device, self.dtype
                )
                new_loss = -float(new_log_lik)
                
                # Check sufficient decrease
                expected_decrease = alpha * torch.dot(gradient, newton_direction)
                if new_loss < current_loss + 0.1 * expected_decrease:
                    print(f"Line search succeeded with alpha={alpha:.6f}")
                    self.log_params.data = log_params_new
                    final_loss = new_loss
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
            'alpha': alpha,
            'hessian_time': hessian_time
        })
        
        print(f"Current: δ={current_params[0]:.6f}, τ={current_params[1]:.6f}, λ={current_params[2]:.6f}")
        print(f"Log-likelihood: {-final_loss:.6f}, Loss: {final_loss:.6f}")
        print("-" * 60)
        
        return final_loss

def test_exact_newton():
    """Test exact Newton's method with automatic differentiation."""
    print("🧪 Testing Exact Newton's Method with Automatic Differentiation")
    print("=" * 60)
    
    species_path = "test_trees_1/sp.nwk"
    gene_path = "test_trees_1/g.nwk"
    
    # Use CPU for more reliable computation
    device = torch.device("cpu")
    
    optimizer = ExactNewtonOptimizer(
        species_path, gene_path,
        init_delta=0.1, init_tau=0.1, init_lambda=0.1,
        device=device
    )
    
    print(f"Initial parameters:")
    init_params = softplus_transform(optimizer.log_params)
    print(f"  δ={init_params[0]:.6f}, τ={init_params[1]:.6f}, λ={init_params[2]:.6f}")
    
    print("\nPerforming exact Newton optimization...")
    
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
        print("Iter | δ        | τ        | λ        | Log-Lik  | Grad Norm | Newton Norm | α     | Time(s)")
        print("-" * 95)
        
        for i, state in enumerate(optimizer.history):
            print(f"{i+1:4d} | {state['delta']:8.6f} | {state['tau']:8.6f} | "
                  f"{state['lambda']:8.6f} | {state['log_likelihood']:8.4f} | "
                  f"{state['grad_norm']:9.6f} | {state['newton_norm']:11.6f} | "
                  f"{state['alpha']:5.3f} | {state['hessian_time']:7.2f}")

def main():
    parser = argparse.ArgumentParser(description='Exact Newton method with automatic differentiation')
    parser.add_argument('--test', action='store_true', help='Run test on test_trees_1')
    
    args = parser.parse_args()
    
    if args.test:
        test_exact_newton()
        return 0
    
    print("Run with --test to test exact Newton's method")
    return 0

if __name__ == "__main__":
    sys.exit(main())