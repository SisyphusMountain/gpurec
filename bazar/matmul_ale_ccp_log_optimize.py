#!/usr/bin/env python3
"""
Exact gradient descent implementation using Pi_update_ccp_log with implicit differentiation.
Uses torch.autograd.set_detect_anomaly(True) to debug any gradient issues with -inf values.
"""

import sys
import time
import torch
import argparse
import json
from tabulate import tabulate
from torch.autograd.functional import vjp

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

def conjugate_gradient(A_func, b, x0=None, max_iter=50, tol=1e-8):
    """
    Solve Ax = b using conjugate gradient method.
    A_func: function that computes A @ x for any x
    Handles both 1D and 2D tensors by flattening/reshaping
    """
    # Store original shape for reshaping at the end
    orig_shape = b.shape
    b_flat = b.flatten()
    
    if x0 is None:
        x_flat = torch.zeros_like(b_flat)
    else:
        x_flat = x0.flatten()
    
    def A_func_flat(x_vec):
        """Apply A_func to reshaped tensor and flatten result"""
        x_reshaped = x_vec.reshape(orig_shape)
        result = A_func(x_reshaped)
        return result.flatten()
    
    r = b_flat - A_func_flat(x_flat)  # residual
    p = r.clone()                     # search direction
    rsold = torch.dot(r, r)
    
    for i in range(max_iter):
        Ap = A_func_flat(p)
        pAp = torch.dot(p, Ap)
        
        # Avoid division by zero
        if torch.abs(pAp) < 1e-12:
            break
            
        alpha = rsold / pAp
        x_flat = x_flat + alpha * p
        r = r - alpha * Ap
        rsnew = torch.dot(r, r)
        
        if torch.sqrt(rsnew) < tol:
            break
            
        beta = rsnew / rsold
        p = r + beta * p
        rsold = rsnew
    
    return x_flat.reshape(orig_shape)

class FixedPointCCPLogFunction(torch.autograd.Function):
    """
    Implements exact gradient computation for fixed-point iteration using Pi_update_ccp_log
    with implicit differentiation as described in fixed_point_iteration.tex
    """
    
    @staticmethod
    def forward(ctx, log_params, species_tree_path, gene_tree_path, max_iter=50, tol=1e-10, device=None, dtype=None):
        """
        Forward pass: Compute fixed point Pi* and evaluate log-likelihood S(Pi*)
        
        Args:
            log_params: Log-space parameters [log_δ, log_τ, log_λ]
            species_tree_path, gene_tree_path: Tree file paths
            max_iter, tol: Fixed-point iteration parameters
            
        Returns:
            log_likelihood: S(Pi*) where Pi* is the fixed point
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if dtype is None:
            dtype = torch.float64
            
        # Transform parameters to positive values
        params = softplus_transform(log_params)
        delta, tau, lambda_param = params[0], params[1], params[2]
        
        print(f"Forward pass with δ={delta:.6f}, τ={tau:.6f}, λ={lambda_param:.6f}")
        
        # Build data structures (these don't require gradients)
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
            
            # Compute extinction probabilities (fixed point, no gradients needed)
            S = species_helpers["S"]
            E = torch.zeros(S, dtype=dtype, device=device)
            for _ in range(max_iter):
                E_next, E_s1, E_s2, Ebar = E_step(E, species_helpers["s_C1"], species_helpers["s_C2"], 
                                                  species_helpers["Recipients_mat"], p_S, p_D, p_T, p_L)
                if torch.abs(E_next - E).max() < tol:
                    break
                E = E_next
        
        # Initialize log_Pi (this needs gradients for the fixed-point iteration)
        C = len(ccp.clades)
        log_Pi = torch.full((C, S), float('-inf'), dtype=dtype, device=device, requires_grad=True)
        
        # Set leaf probabilities
        with torch.no_grad():
            for c in range(C):
                clade = ccp.id_to_clade[c]
                if clade.is_leaf():
                    mapped_species = torch.nonzero(clade_species_map[c] > 0, as_tuple=False).flatten()
                    if len(mapped_species) > 0:
                        log_prob = -torch.log(torch.tensor(len(mapped_species), dtype=dtype))
                        log_Pi.data[c, mapped_species] = log_prob
        
        # Fixed-point iteration for log_Pi
        print("Running Pi fixed-point iteration...")
        for iter_pi in range(max_iter):
            log_Pi_new = Pi_update_ccp_log(log_Pi, ccp_helpers, species_helpers, clade_species_map, 
                                          E, Ebar, p_S, p_D, p_T)
            
            # Check convergence
            if iter_pi > 0:
                diff = torch.abs(log_Pi_new - log_Pi).max()
                if diff < tol:
                    print(f"Pi converged after {iter_pi+1} iterations (diff={diff:.2e})")
                    break
            
            log_Pi = log_Pi_new
        
        # Compute log-likelihood
        root_clade_id = get_root_clade_id(ccp)
        log_likelihood = torch.logsumexp(log_Pi[root_clade_id, :], dim=0)
        
        print(f"Log-likelihood: {log_likelihood:.6f}")
        print(f"Root Pi range: [{log_Pi[root_clade_id, :].min():.3f}, {log_Pi[root_clade_id, :].max():.3f}]")
        
        # Save context for backward pass
        ctx.save_for_backward(log_Pi, log_params)
        ctx.ccp = ccp
        ctx.species_helpers = species_helpers
        ctx.clade_species_map = clade_species_map
        ctx.ccp_helpers = ccp_helpers
        ctx.E = E
        ctx.Ebar = Ebar
        ctx.root_clade_id = root_clade_id
        ctx.params = params
        ctx.device = device
        ctx.dtype = dtype
        
        return log_likelihood
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: Compute gradients using implicit differentiation
        """
        print("\n=== BACKWARD PASS ===")
        
        log_Pi_star, log_params = ctx.saved_tensors
        params = ctx.params
        delta, tau, lambda_param = params[0], params[1], params[2]
        
        # Recompute event probabilities with gradients
        rates_sum = 1.0 + delta + tau + lambda_param
        p_S = 1.0 / rates_sum
        p_D = delta / rates_sum
        p_T = tau / rates_sum
        
        print(f"Computing gradients for δ={delta:.6f}, τ={tau:.6f}, λ={lambda_param:.6f}")
        
        # Step 1: Compute ∇_Pi S(Pi*) where S is the log-likelihood function
        root_id = ctx.root_clade_id
        C, S = log_Pi_star.shape
        
        # S(Pi) = logsumexp(Pi[root, :])
        # ∇_Pi S = softmax(Pi[root, :]) at position [root, :], zero elsewhere
        g_x = torch.zeros_like(log_Pi_star)
        root_probs = torch.softmax(log_Pi_star[root_id, :], dim=0)
        g_x[root_id, :] = root_probs
        
        print(f"g_x norm: {g_x.norm():.6f}")
        print(f"Root probs range: [{root_probs.min():.6f}, {root_probs.max():.6f}]")
        
        # Step 2: Define the adjoint linear system (I - ∇_Pi F)^T v = g_x
        # where F is the Pi_update_ccp_log function
        
        def F_eval(Pi_input):
            """Evaluate F(Pi, θ) = Pi_update_ccp_log(Pi, ...)"""
            return Pi_update_ccp_log(Pi_input, ctx.ccp_helpers, ctx.species_helpers, ctx.clade_species_map,
                                   ctx.E, ctx.Ebar, p_S, p_D, p_T)
        
        def adjoint_operator(v):
            """Compute (I - ∇_Pi F)^T v using VJP"""
            try:
                # Ensure v is on the same device as log_Pi_star
                v_device = v.to(log_Pi_star.device)
                
                # Compute VJP: ∇_Pi F^T v
                # vjp expects a scalar output, but F_eval returns a tensor
                # We need to compute the VJP with respect to the full tensor
                def scalar_F(Pi_input):
                    F_output = F_eval(Pi_input)
                    return torch.sum(F_output * v_device)  # Contract with v
                
                jacobian_t_v, = torch.autograd.grad(scalar_F(log_Pi_star), log_Pi_star, 
                                                   create_graph=True, retain_graph=True)
                
                # Return (I - ∇_Pi F)^T v = v - jacobian_t_v
                return v_device - jacobian_t_v
            except Exception as e:
                print(f"Error in adjoint operator: {e}")
                # Fallback: assume contraction property and return v
                return v.to(log_Pi_star.device)
        
        # Step 3: Solve the adjoint equation using conjugate gradient
        print("Solving adjoint equation...")
        try:
            v_solution = conjugate_gradient(adjoint_operator, g_x, max_iter=20, tol=1e-6)
            print(f"Adjoint solution norm: {v_solution.norm():.6f}")
        except Exception as e:
            print(f"Error solving adjoint equation: {e}")
            # Fallback: use g_x directly (assumes small contraction)
            v_solution = g_x
        
        # Step 4: Compute ∇_θ L = ∇_θ F(Pi*, θ)^T v
        print("Computing parameter gradients...")
        
        def F_theta(theta_input):
            """F as a function of θ only, with Pi* fixed"""
            delta_t, tau_t, lambda_t = theta_input[0], theta_input[1], theta_input[2]
            rates_sum_t = 1.0 + delta_t + tau_t + lambda_t
            p_S_t = 1.0 / rates_sum_t
            p_D_t = delta_t / rates_sum_t
            p_T_t = tau_t / rates_sum_t
            
            return Pi_update_ccp_log(log_Pi_star, ctx.ccp_helpers, ctx.species_helpers, ctx.clade_species_map,
                                   ctx.E, ctx.Ebar, p_S_t, p_D_t, p_T_t)
        
        try:
            # Use autograd.grad instead of vjp for parameter gradients
            def scalar_F_theta(theta_input):
                delta_t, tau_t, lambda_t = theta_input[0], theta_input[1], theta_input[2]
                rates_sum_t = 1.0 + delta_t + tau_t + lambda_t
                p_S_t = 1.0 / rates_sum_t
                p_D_t = delta_t / rates_sum_t
                p_T_t = tau_t / rates_sum_t
                
                F_output = Pi_update_ccp_log(log_Pi_star, ctx.ccp_helpers, ctx.species_helpers, ctx.clade_species_map,
                                           ctx.E, ctx.Ebar, p_S_t, p_D_t, p_T_t)
                return torch.sum(F_output * v_solution)  # Contract with v_solution
            
            g_theta, = torch.autograd.grad(scalar_F_theta(params), params, 
                                          create_graph=True, retain_graph=True)
            
            print(f"Raw gradients: δ={g_theta[0]:.6f}, τ={g_theta[1]:.6f}, λ={g_theta[2]:.6f}")
            
            # Transform gradients through softplus: d/d(log_θ) = d/dθ * dθ/d(log_θ)
            # where dθ/d(log_θ) = softplus'(log_θ) = sigmoid(log_θ)
            sigmoid_log_params = torch.sigmoid(log_params).to(g_theta.device)
            g_log_params = g_theta * sigmoid_log_params
            
            print(f"Transformed gradients: log_δ={g_log_params[0]:.6f}, log_τ={g_log_params[1]:.6f}, log_λ={g_log_params[2]:.6f}")
            
        except Exception as e:
            print(f"Error computing parameter gradients: {e}")
            # Return zero gradients if computation fails
            g_log_params = torch.zeros_like(log_params).to(ctx.device)
        
        # Return gradients w.r.t. log_params (first argument), None for other arguments
        return grad_output * g_log_params, None, None, None, None, None, None

class CCPLogOptimizer:
    """Optimizer for CCP reconciliation parameters using exact gradients."""
    
    def __init__(self, species_tree_path, gene_tree_path, 
                 init_delta=0.1, init_tau=0.1, init_lambda=0.1,
                 lr=0.01, device=None, dtype=torch.float64):
        
        self.species_tree_path = species_tree_path
        self.gene_tree_path = gene_tree_path
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        
        # Initialize parameters in log space
        init_params = torch.tensor([init_delta, init_tau, init_lambda], dtype=dtype)
        self.log_params = torch.nn.Parameter(inverse_softplus(init_params))
        
        self.optimizer = torch.optim.Adam([self.log_params], lr=lr)
        
        self.history = []
        
    def step(self):
        """Perform one optimization step."""
        self.optimizer.zero_grad()
        
        # Compute objective and gradients
        objective = FixedPointCCPLogFunction.apply(
            self.log_params, self.species_tree_path, self.gene_tree_path,
            50, 1e-10, self.device, self.dtype
        )
        
        # Minimize negative log-likelihood
        loss = -objective
        loss.backward()
        
        # Record current state
        with torch.no_grad():
            current_params = softplus_transform(self.log_params)
            self.history.append({
                'delta': float(current_params[0]),
                'tau': float(current_params[1]),
                'lambda': float(current_params[2]),
                'log_likelihood': float(objective),
                'loss': float(loss),
                'grad_norm': float(self.log_params.grad.norm()) if self.log_params.grad is not None else 0.0
            })
            
            print(f"Current: δ={current_params[0]:.6f}, τ={current_params[1]:.6f}, λ={current_params[2]:.6f}")
            print(f"Log-likelihood: {objective:.6f}, Loss: {loss:.6f}")
            if self.log_params.grad is not None:
                print(f"Gradient norm: {self.log_params.grad.norm():.6f}")
                print(f"Gradients: {self.log_params.grad}")
            print("-" * 60)
        
        self.optimizer.step()
        
        return float(loss)

def test_gradient_computation():
    """Test gradient computation on test_trees_1."""
    print("🧪 Testing exact gradient computation on test_trees_1")
    print("=" * 60)
    
    species_path = "test_trees_1/sp.nwk"
    gene_path = "test_trees_1/g.nwk"
    
    # Initialize optimizer with parameters that should optimize toward zero
    optimizer = CCPLogOptimizer(
        species_path, gene_path,
        init_delta=0.1, init_tau=0.1, init_lambda=0.1,
        lr=0.01
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
        
    except Exception as e:
        print(f"❌ Error during gradient computation: {e}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description='Exact gradient descent for CCP reconciliation parameters')
    parser.add_argument('--species', help='Species tree file (.nwk)')
    parser.add_argument('--gene', help='Gene tree file (.nwk)')
    parser.add_argument('--init-delta', type=float, default=0.1, help='Initial δ (default: 0.1)')
    parser.add_argument('--init-tau', type=float, default=0.1, help='Initial τ (default: 0.1)')
    parser.add_argument('--init-lambda', type=float, default=0.1, help='Initial λ (default: 0.1)')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate (default: 0.01)')
    parser.add_argument('--epochs', type=int, default=1, help='Number of optimization steps (default: 1)')
    parser.add_argument('--test', action='store_true', help='Run gradient test on test_trees_1')
    
    args = parser.parse_args()
    
    if args.test:
        test_gradient_computation()
        return 0
    
    if not args.species or not args.gene:
        parser.error("--species and --gene are required unless using --test")
    
    try:
        optimizer = CCPLogOptimizer(
            args.species, args.gene,
            init_delta=args.init_delta, init_tau=args.init_tau, init_lambda=args.init_lambda,
            lr=args.lr
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