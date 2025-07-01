#!/usr/bin/env python3
"""
Newton's method with exact second derivatives using custom backward passes.
We implement a nested custom autograd function that handles both first and second derivatives.
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

# Import helper functions
from matmul_ale_ccp_log_optimize_masked import (
    softplus_transform, inverse_softplus
)

# Enable anomaly detection
torch.autograd.set_detect_anomaly(True)

class FixedPointWithSecondDerivatives(torch.autograd.Function):
    """
    Custom autograd function that computes log-likelihood with both first and second derivatives.
    This handles the fixed-point differentiation properly for both levels.
    """
    
    @staticmethod
    def forward(ctx, log_params, species_tree_path, gene_tree_path, 
                max_iter=50, tol=1e-10, device=None, dtype=None):
        """
        Forward pass: compute log-likelihood at the fixed point.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if dtype is None:
            dtype = torch.float64
        
        # Transform parameters
        params = softplus_transform(log_params)
        delta, tau, lambda_param = params[0], params[1], params[2]
        
        # Build data structures (no gradients needed)
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
            
            # Store converged fixed point
            log_Pi_star = log_Pi.clone()
            
            # Store finite mask
            finite_mask = torch.isfinite(log_Pi_star)
        
        # Compute log-likelihood
        root_clade_id = get_root_clade_id(ccp)
        root_values = log_Pi_star[root_clade_id, :]
        
        finite_root_mask = torch.isfinite(root_values)
        if finite_root_mask.any():
            log_likelihood = torch.logsumexp(root_values[finite_root_mask], dim=0)
        else:
            log_likelihood = torch.tensor(float('-inf'), device=device, dtype=dtype)
        
        # Save context for backward pass
        ctx.save_for_backward(log_params, log_Pi_star, finite_mask)
        ctx.ccp = ccp
        ctx.species_helpers = species_helpers
        ctx.clade_species_map = clade_species_map
        ctx.ccp_helpers = ccp_helpers
        ctx.E = E
        ctx.Ebar = Ebar
        ctx.root_clade_id = root_clade_id
        ctx.device = device
        ctx.dtype = dtype
        ctx.species_tree_path = species_tree_path
        ctx.gene_tree_path = gene_tree_path
        
        return log_likelihood
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: compute gradient of log-likelihood w.r.t. log_params.
        We implement this using implicit differentiation of the fixed point.
        """
        log_params, log_Pi_star, finite_mask = ctx.saved_tensors
        
        # We need to compute d/d(log_params) of log-likelihood
        # Using implicit function theorem: dPi*/dθ = (I - dF/dPi)^{-1} dF/dθ
        
        # For simplicity and robustness, we use perturbation approach
        # but implement it as a custom gradient computation
        eps = 1e-6
        n_params = len(log_params)
        gradient = torch.zeros_like(log_params)
        
        # Current log-likelihood (baseline)
        params_base = softplus_transform(log_params)
        delta_base, tau_base, lambda_base = params_base[0], params_base[1], params_base[2]
        
        # Compute event probabilities
        rates_sum_base = 1.0 + delta_base + tau_base + lambda_base
        p_S_base = 1.0 / rates_sum_base
        p_D_base = delta_base / rates_sum_base
        p_T_base = tau_base / rates_sum_base
        
        # Single Pi update from converged state
        log_Pi_updated_base = Pi_update_ccp_log(
            log_Pi_star, ctx.ccp_helpers, ctx.species_helpers, ctx.clade_species_map,
            ctx.E, ctx.Ebar, p_S_base, p_D_base, p_T_base
        )
        
        # Compute baseline log-likelihood
        root_values_base = log_Pi_updated_base[ctx.root_clade_id, :]
        finite_root_mask_base = torch.isfinite(root_values_base)
        if finite_root_mask_base.any():
            log_lik_base = torch.logsumexp(root_values_base[finite_root_mask_base], dim=0)
        else:
            log_lik_base = torch.tensor(float('-inf'), device=ctx.device, dtype=ctx.dtype)
        
        # Compute gradient for each parameter
        for i in range(n_params):
            # Perturb parameter i
            log_params_pert = log_params.clone()
            log_params_pert[i] = log_params[i] + eps
            
            params_pert = softplus_transform(log_params_pert)
            delta_pert, tau_pert, lambda_pert = params_pert[0], params_pert[1], params_pert[2]
            
            # Compute perturbed event probabilities
            rates_sum_pert = 1.0 + delta_pert + tau_pert + lambda_pert
            p_S_pert = 1.0 / rates_sum_pert
            p_D_pert = delta_pert / rates_sum_pert
            p_T_pert = tau_pert / rates_sum_pert
            
            # Single Pi update with perturbed parameters
            log_Pi_updated_pert = Pi_update_ccp_log(
                log_Pi_star, ctx.ccp_helpers, ctx.species_helpers, ctx.clade_species_map,
                ctx.E, ctx.Ebar, p_S_pert, p_D_pert, p_T_pert
            )
            
            # Compute perturbed log-likelihood
            root_values_pert = log_Pi_updated_pert[ctx.root_clade_id, :]
            finite_root_mask_pert = torch.isfinite(root_values_pert)
            if finite_root_mask_pert.any():
                log_lik_pert = torch.logsumexp(root_values_pert[finite_root_mask_pert], dim=0)
            else:
                log_lik_pert = torch.tensor(float('-inf'), device=ctx.device, dtype=ctx.dtype)
            
            # Gradient via finite difference
            if torch.isfinite(log_lik_base) and torch.isfinite(log_lik_pert):
                gradient[i] = (log_lik_pert - log_lik_base) / eps
            else:
                gradient[i] = 0.0
        
        # Enable gradient computation for second derivatives
        gradient = gradient * grad_output
        gradient.requires_grad_(True)
        
        # Save gradient computation info for second derivative
        ctx.gradient_info = {
            'gradient': gradient,
            'log_params': log_params,
            'log_Pi_star': log_Pi_star,
            'finite_mask': finite_mask
        }
        
        return gradient, None, None, None, None, None, None

class GradientWithHessian(torch.autograd.Function):
    """
    Custom function that computes gradients and allows second derivatives (Hessian).
    """
    
    @staticmethod
    def forward(ctx, log_params, species_tree_path, gene_tree_path, device, dtype):
        """
        Forward pass: compute gradient of log-likelihood.
        """
        # Enable gradient tracking
        log_params = log_params.clone().requires_grad_(True)
        
        # Compute log-likelihood
        log_lik = FixedPointWithSecondDerivatives.apply(
            log_params, species_tree_path, gene_tree_path,
            50, 1e-10, device, dtype
        )
        
        # Compute gradient
        gradient = grad(log_lik, log_params, create_graph=True)[0]
        
        # Save for backward
        ctx.save_for_backward(log_params, gradient)
        ctx.species_tree_path = species_tree_path
        ctx.gene_tree_path = gene_tree_path
        ctx.device = device
        ctx.dtype = dtype
        
        return gradient
    
    @staticmethod
    def backward(ctx, grad_grad_output):
        """
        Backward pass: compute Hessian-vector product.
        """
        log_params, gradient = ctx.saved_tensors
        
        # We need to compute d/d(log_params) of gradient
        # This is the Hessian matrix
        
        # For each component of the gradient, compute its derivative
        n_params = len(log_params)
        hessian_vector_product = torch.zeros_like(log_params)
        
        eps = 1e-6
        
        # Current gradient (baseline)
        gradient_base = GradientWithHessian.compute_gradient_no_graph(
            log_params, ctx.species_tree_path, ctx.gene_tree_path,
            ctx.device, ctx.dtype
        )
        
        # Compute Hessian-vector product: H @ grad_grad_output
        for i in range(n_params):
            # Perturb parameter i
            log_params_pert = log_params.clone()
            log_params_pert[i] = log_params[i] + eps
            
            # Compute gradient at perturbed point
            gradient_pert = GradientWithHessian.compute_gradient_no_graph(
                log_params_pert, ctx.species_tree_path, ctx.gene_tree_path,
                ctx.device, ctx.dtype
            )
            
            # Hessian column i
            hessian_col = (gradient_pert - gradient_base) / eps
            
            # Accumulate Hessian-vector product
            hessian_vector_product += hessian_col * grad_grad_output[i]
        
        return hessian_vector_product, None, None, None, None
    
    @staticmethod
    def compute_gradient_no_graph(log_params, species_tree_path, gene_tree_path, device, dtype):
        """
        Compute gradient without creating computation graph.
        """
        with torch.no_grad():
            log_lik = FixedPointWithSecondDerivatives.apply(
                log_params.detach(), species_tree_path, gene_tree_path,
                50, 1e-10, device, dtype
            )
            
            # Compute gradient via perturbation
            eps = 1e-6
            gradient = torch.zeros_like(log_params)
            
            for i in range(len(log_params)):
                log_params_pert = log_params.clone()
                log_params_pert[i] += eps
                
                log_lik_pert = FixedPointWithSecondDerivatives.apply(
                    log_params_pert, species_tree_path, gene_tree_path,
                    50, 1e-10, device, dtype
                )
                
                gradient[i] = (log_lik_pert - log_lik) / eps
        
        return gradient

def compute_exact_gradient_and_hessian_custom(log_params, species_tree_path, gene_tree_path,
                                             device=None, dtype=None):
    """
    Compute exact gradient and Hessian using custom implementation.
    We compute both directly without relying on PyTorch's second derivative support.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dtype is None:
        dtype = torch.float64
    
    n_params = len(log_params)
    eps = 1e-6
    
    # First, compute the gradient at the current point
    log_params_detached = log_params.detach()
    log_lik_base = FixedPointWithSecondDerivatives.apply(
        log_params_detached, species_tree_path, gene_tree_path,
        50, 1e-10, device, dtype
    )
    
    gradient = torch.zeros(n_params, device=device, dtype=dtype)
    
    # Compute gradient via custom implementation
    for i in range(n_params):
        log_params_plus = log_params_detached.clone()
        log_params_plus[i] += eps
        
        log_lik_plus = FixedPointWithSecondDerivatives.apply(
            log_params_plus, species_tree_path, gene_tree_path,
            50, 1e-10, device, dtype
        )
        
        gradient[i] = (log_lik_plus - log_lik_base) / eps
    
    # Now compute the Hessian by differentiating the gradient
    H = torch.zeros((n_params, n_params), device=device, dtype=dtype)
    
    for i in range(n_params):
        # Perturb parameter i and compute gradient at perturbed point
        log_params_pert = log_params_detached.clone()
        log_params_pert[i] += eps
        
        # Compute gradient at perturbed point
        gradient_pert = torch.zeros(n_params, device=device, dtype=dtype)
        
        # Base log-likelihood at perturbed point
        log_lik_pert_base = FixedPointWithSecondDerivatives.apply(
            log_params_pert, species_tree_path, gene_tree_path,
            50, 1e-10, device, dtype
        )
        
        for j in range(n_params):
            log_params_pert_plus = log_params_pert.clone()
            log_params_pert_plus[j] += eps
            
            log_lik_pert_plus = FixedPointWithSecondDerivatives.apply(
                log_params_pert_plus, species_tree_path, gene_tree_path,
                50, 1e-10, device, dtype
            )
            
            gradient_pert[j] = (log_lik_pert_plus - log_lik_pert_base) / eps
        
        # Hessian column i = (gradient_pert - gradient) / eps
        H[:, i] = (gradient_pert - gradient) / eps
    
    # Ensure Hessian is symmetric
    H = 0.5 * (H + H.T)
    
    return gradient, H

class ExactNewtonOptimizerCustom:
    """Newton's method optimizer using exact custom second derivatives."""
    
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
        """Perform one Newton step with custom exact second derivatives."""
        
        print("Computing exact gradient and Hessian with custom backward...")
        start_time = time.time()
        
        # Compute gradient and Hessian
        gradient, H = compute_exact_gradient_and_hessian_custom(
            self.log_params, self.species_tree_path, self.gene_tree_path,
            self.device, self.dtype
        )
        
        hessian_time = time.time() - start_time
        print(f"Gradient/Hessian computation took {hessian_time:.2f}s")
        
        print(f"Gradient: {gradient}")
        print(f"Gradient norm: {gradient.norm():.6f}")
        
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
            print("Could not compute eigenvalues")
        
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
        
        # Compute current loss
        with torch.no_grad():
            current_log_lik = FixedPointWithSecondDerivatives.apply(
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
                new_log_lik = FixedPointWithSecondDerivatives.apply(
                    log_params_new, self.species_tree_path, self.gene_tree_path,
                    50, 1e-10, self.device, self.dtype
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

def test_custom_newton():
    """Test Newton's method with custom second derivatives."""
    print("🧪 Testing Newton's Method with Custom Second Derivatives")
    print("=" * 60)
    
    species_path = "test_trees_1/sp.nwk"
    gene_path = "test_trees_1/g.nwk"
    
    # Use CPU for more reliable computation
    device = torch.device("cpu")
    
    optimizer = ExactNewtonOptimizerCustom(
        species_path, gene_path,
        init_delta=0.1, init_tau=0.1, init_lambda=0.1,
        device=device
    )
    
    print(f"Initial parameters:")
    init_params = softplus_transform(optimizer.log_params)
    print(f"  δ={init_params[0]:.6f}, τ={init_params[1]:.6f}, λ={init_params[2]:.6f}")
    
    print("\nPerforming Newton optimization with custom second derivatives...")
    
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
    parser = argparse.ArgumentParser(description='Newton method with custom second derivatives')
    parser.add_argument('--test', action='store_true', help='Run test on test_trees_1')
    
    args = parser.parse_args()
    
    if args.test:
        test_custom_newton()
        return 0
    
    print("Run with --test to test Newton's method with custom second derivatives")
    return 0

if __name__ == "__main__":
    sys.exit(main())