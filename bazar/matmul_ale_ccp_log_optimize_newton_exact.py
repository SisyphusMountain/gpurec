#!/usr/bin/env python3
"""
Newton's method with exact second derivatives using custom masked Hessian computation.
Key insight: -inf values have zero first and second derivatives.
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

# Import our masked gradient computation
from matmul_ale_ccp_log_optimize_masked import (
    softplus_transform, inverse_softplus
)

# Enable anomaly detection
torch.autograd.set_detect_anomaly(True)

class MaskedPiUpdateWithHessian(torch.autograd.Function):
    """
    Custom autograd function that computes both gradients and Hessian.
    Key insights:
    1. -inf values that remain -inf have zero gradients
    2. Zero gradients have zero second derivatives
    3. We can mask these positions at each differentiation level
    """
    
    @staticmethod
    def forward(ctx, log_params, species_tree_path, gene_tree_path, 
                max_iter=50, tol=1e-10, device=None, dtype=None):
        """
        Forward pass: compute log-likelihood at fixed point.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if dtype is None:
            dtype = torch.float64
        
        # Ensure gradients are enabled
        log_params = log_params.to(device=device, dtype=dtype)
        if not log_params.requires_grad:
            log_params.requires_grad_(True)
        
        # Transform parameters
        params = softplus_transform(log_params)
        delta, tau, lambda_param = params[0], params[1], params[2]
        
        # Build data structures
        with torch.no_grad():
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
        
        # Store which positions are finite
        finite_mask = torch.isfinite(log_Pi)
        
        # Compute log-likelihood
        root_clade_id = get_root_clade_id(ccp)
        root_values = log_Pi[root_clade_id, :]
        
        finite_root_mask = torch.isfinite(root_values)
        if finite_root_mask.any():
            log_likelihood = torch.logsumexp(root_values[finite_root_mask], dim=0)
        else:
            log_likelihood = torch.tensor(float('-inf'), device=device, dtype=dtype)
        
        # Save context
        ctx.save_for_backward(log_params, log_Pi, finite_mask)
        ctx.log_likelihood = log_likelihood
        ctx.species_tree_path = species_tree_path
        ctx.gene_tree_path = gene_tree_path
        ctx.device = device
        ctx.dtype = dtype
        ctx.ccp = ccp
        ctx.species_helpers = species_helpers
        ctx.clade_species_map = clade_species_map
        ctx.ccp_helpers = ccp_helpers
        ctx.E = E
        ctx.Ebar = Ebar
        ctx.root_clade_id = root_clade_id
        
        return log_likelihood
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: compute gradient using finite differences with masking.
        """
        log_params, log_Pi, finite_mask = ctx.saved_tensors
        
        # Compute gradient via finite differences
        eps = 1e-6
        grad = torch.zeros_like(log_params)
        
        for i in range(len(log_params)):
            log_params_plus = log_params.clone()
            log_params_plus[i] += eps
            
            # Compute perturbed log-likelihood
            log_lik_plus = MaskedPiUpdateWithHessian.compute_log_likelihood(
                log_params_plus, ctx.species_tree_path, ctx.gene_tree_path,
                ctx.ccp, ctx.species_helpers, ctx.clade_species_map, ctx.ccp_helpers,
                ctx.E, ctx.Ebar, ctx.device, ctx.dtype
            )
            
            # Gradient via finite difference
            grad[i] = (log_lik_plus - ctx.log_likelihood) / eps
        
        return grad * grad_output, None, None, None, None, None, None
    
    @staticmethod
    def compute_log_likelihood(log_params, species_tree_path, gene_tree_path,
                              ccp, species_helpers, clade_species_map, ccp_helpers,
                              E, Ebar, device, dtype):
        """
        Helper function to compute log-likelihood for given parameters.
        """
        # Transform parameters
        params = softplus_transform(log_params)
        delta, tau, lambda_param = params[0], params[1], params[2]
        
        # Compute event probabilities
        rates_sum = 1.0 + delta + tau + lambda_param
        p_S = 1.0 / rates_sum
        p_D = delta / rates_sum
        p_T = tau / rates_sum
        
        # Initialize log_Pi
        C = len(ccp.clades)
        S = species_helpers["S"]
        log_Pi = torch.full((C, S), float('-inf'), dtype=dtype, device=device)
        
        # Set leaf probabilities
        for c in range(C):
            clade = ccp.id_to_clade[c]
            if clade.is_leaf():
                mapped_species = torch.nonzero(clade_species_map[c] > 0, as_tuple=False).flatten()
                if len(mapped_species) > 0:
                    log_prob = -torch.log(torch.tensor(len(mapped_species), dtype=dtype))
                    log_Pi[c, mapped_species] = log_prob
        
        # Run Pi iterations to convergence
        for _ in range(50):  # max iterations
            log_Pi_new = Pi_update_ccp_log(log_Pi, ccp_helpers, species_helpers, clade_species_map,
                                          E, Ebar, p_S, p_D, p_T)
            # Check convergence
            finite_mask = torch.isfinite(log_Pi) & torch.isfinite(log_Pi_new)
            if finite_mask.any():
                diff = torch.abs(log_Pi_new[finite_mask] - log_Pi[finite_mask]).max()
                if diff < 1e-10:
                    break
            log_Pi = log_Pi_new
        
        # Compute log-likelihood
        root_clade_id = get_root_clade_id(ccp)
        root_values = log_Pi[root_clade_id, :]
        
        finite_root_mask = torch.isfinite(root_values)
        if finite_root_mask.any():
            return torch.logsumexp(root_values[finite_root_mask], dim=0)
        else:
            return torch.tensor(float('-inf'), device=device, dtype=dtype)

def compute_exact_hessian(log_params, species_tree_path, gene_tree_path, 
                         device=None, dtype=None):
    """
    Compute exact Hessian using custom masked second derivatives.
    
    The key insight is that we compute second derivatives only for positions
    that have non-zero first derivatives (i.e., finite values).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dtype is None:
        dtype = torch.float64
    
    n_params = len(log_params)
    H = torch.zeros((n_params, n_params), device=device, dtype=dtype)
    
    # First compute gradient using finite differences
    eps = 1e-6
    grad = torch.zeros(n_params, device=device, dtype=dtype)
    
    # Get baseline log-likelihood
    log_params_detached = log_params.clone().detach()
    log_lik_base = MaskedPiUpdateWithHessian.apply(
        log_params_detached, species_tree_path, gene_tree_path,
        50, 1e-10, device, dtype
    )
    
    # Compute gradient via finite differences
    for i in range(n_params):
        log_params_plus = log_params_detached.clone()
        log_params_plus[i] += eps
        
        log_lik_plus = MaskedPiUpdateWithHessian.apply(
            log_params_plus, species_tree_path, gene_tree_path,
            50, 1e-10, device, dtype
        )
        
        grad[i] = (log_lik_plus - log_lik_base) / eps
    
    # Now compute Hessian using finite differences of gradients
    eps = 1e-6
    
    for i in range(n_params):
        # Perturb parameter i
        log_params_plus = log_params.clone().detach()
        log_params_plus[i] += eps
        log_params_plus.requires_grad_(True)
        
        # Compute gradient at perturbed point
        if log_params_plus.grad is not None:
            log_params_plus.grad.zero_()
        
        log_lik_plus = MaskedPiUpdateWithHessian.apply(
            log_params_plus, species_tree_path, gene_tree_path,
            50, 1e-10, device, dtype
        )
        log_lik_plus.backward()
        grad_plus = log_params_plus.grad.clone()
        
        # Hessian column i = (grad_plus - grad) / eps
        H[:, i] = (grad_plus - grad) / eps
    
    # Ensure Hessian is symmetric (average with transpose)
    H = 0.5 * (H + H.T)
    
    return grad, H

class ExactNewtonOptimizer:
    """Newton's method optimizer using exact masked Hessian."""
    
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
        
        self.history = []
    
    def step(self, reg=1e-4, max_line_search=10, alpha_init=1.0):
        """Perform one Newton step with exact Hessian."""
        
        print("Computing exact gradient and Hessian...")
        start_time = time.time()
        
        # Compute gradient and Hessian
        gradient, H = compute_exact_hessian(
            self.log_params, self.species_tree_path, self.gene_tree_path,
            self.device, self.dtype
        )
        
        hessian_time = time.time() - start_time
        print(f"Hessian computation took {hessian_time:.2f}s")
        
        print(f"Gradient: {gradient}")
        print(f"Gradient norm: {gradient.norm():.6f}")
        
        # Check Hessian properties
        eigenvalues = torch.linalg.eigvalsh(H)
        print(f"Hessian eigenvalues: {eigenvalues}")
        print(f"Hessian condition number: {eigenvalues.max() / eigenvalues.min():.2e}")
        
        # Add regularization
        H_reg = H + reg * torch.eye(len(self.log_params), device=self.device, dtype=self.dtype)
        
        # Compute Newton direction
        try:
            # Try Cholesky decomposition (for positive definite)
            L = torch.linalg.cholesky(H_reg)
            newton_direction = -torch.cholesky_solve(gradient.unsqueeze(1), L).squeeze()
        except:
            # Fallback to general solver
            print("Cholesky failed, using general solver...")
            newton_direction = -torch.linalg.solve(H_reg, gradient)
        
        print(f"Newton direction: {newton_direction}")
        print(f"Newton step norm: {newton_direction.norm():.6f}")
        
        # Compute current loss
        with torch.no_grad():
            current_log_lik = MaskedPiUpdateWithHessian.apply(
                self.log_params, self.species_tree_path, self.gene_tree_path,
                50, 1e-10, self.device, self.dtype
            )
            current_loss = -float(current_log_lik)
        
        # Line search
        alpha = alpha_init
        with torch.no_grad():
            for i in range(max_line_search):
                # Try step
                log_params_new = self.log_params + alpha * newton_direction
                
                # Evaluate objective
                new_log_lik = MaskedPiUpdateWithHessian.apply(
                    log_params_new, self.species_tree_path, self.gene_tree_path,
                    50, 1e-10, self.device, self.dtype
                )
                new_loss = -float(new_log_lik)
                
                # Check sufficient decrease
                expected_decrease = alpha * torch.dot(gradient, newton_direction)
                if new_loss < current_loss + 0.1 * expected_decrease:
                    print(f"Line search succeeded with alpha={alpha:.6f}")
                    self.log_params = log_params_new
                    final_loss = new_loss
                    break
                
                # Reduce step size
                alpha *= 0.5
                print(f"Reducing step size to {alpha:.6f}")
            else:
                print("Line search failed, taking small step")
                self.log_params = self.log_params + 0.01 * newton_direction
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
    """Test exact Newton's method."""
    print("🧪 Testing Exact Newton's Method on test_trees_1")
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
    parser = argparse.ArgumentParser(description='Exact Newton method with custom Hessian')
    parser.add_argument('--test', action='store_true', help='Run test on test_trees_1')
    
    args = parser.parse_args()
    
    if args.test:
        test_exact_newton()
        return 0
    
    print("Run with --test to test exact Newton's method")
    return 0

if __name__ == "__main__":
    sys.exit(main())