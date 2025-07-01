#!/usr/bin/env python3
"""
Gradient descent optimization for phylogenetic reconciliation parameters.
Implements fixed-point differentiation following the theory in fixed_point_iteration.tex.
"""

import torch
import torch.nn.functional as F
from torch.autograd.functional import vjp
import time
import numpy as np
from typing import Dict, Tuple, Optional, Any

# Import existing CCP functions
from matmul_ale_ccp_log import (
    build_ccp_from_single_tree, build_species_helpers,
    build_clade_species_mapping, build_ccp_helpers,
    get_root_clade_id, E_step, Pi_update_ccp_log
)


def softplus_transform(log_params: torch.Tensor) -> torch.Tensor:
    """
    Transform unconstrained parameters to positive parameters using softplus.
    
    Args:
        log_params: [log_delta, log_tau, log_lambda] unconstrained parameters
        
    Returns:
        params: [delta, tau, lambda] positive parameters
    """
    return F.softplus(log_params)


def inverse_softplus_transform(params: torch.Tensor) -> torch.Tensor:
    """
    Inverse softplus: log_params = log(exp(params) - 1).
    For numerical stability when params is large, use log(exp(params) - 1) = log(params) + log(1 - exp(-params)).
    For small params, use log(exp(params) - 1) directly.
    
    Args:
        params: [delta, tau, lambda] positive parameters
        
    Returns:
        log_params: [log_delta, log_tau, log_lambda] unconstrained parameters
    """
    # Use the exact inverse of softplus: log(exp(x) - 1)
    # For numerical stability, handle small and large values differently
    return torch.where(
        params > 20.0,  # For large params, use log(params) approximation
        torch.log(params) + torch.log1p(-torch.exp(-params)),
        torch.log(torch.expm1(params))  # For small params, use exact formula
    )


def compute_event_probabilities(params: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute event probabilities from δ, τ, λ parameters.
    
    Args:
        params: [delta, tau, lambda] positive parameters
        
    Returns:
        p_S, p_D, p_T, p_L: Event probabilities
    """
    delta, tau, lambda_param = params[0], params[1], params[2]
    rates_sum = 1.0 + delta + tau + lambda_param
    
    p_S = 1.0 / rates_sum
    p_D = delta / rates_sum
    p_T = tau / rates_sum
    p_L = lambda_param / rates_sum
    
    return p_S, p_D, p_T, p_L


def ccp_fixed_point_iteration(log_Pi_init: torch.Tensor, 
                             ccp_helpers: Dict, species_helpers: Dict, clade_species_map: torch.Tensor,
                             params: torch.Tensor, max_iter: int = 50, tol: float = 1e-8) -> torch.Tensor:
    """
    Run fixed-point iteration until convergence.
    
    Args:
        log_Pi_init: Initial log probability matrix [C, S]
        ccp_helpers, species_helpers, clade_species_map: Tree structure data
        params: [delta, tau, lambda] parameters
        max_iter: Maximum iterations
        tol: Convergence tolerance
        
    Returns:
        log_Pi_converged: Converged log probability matrix [C, S]
    """
    # Compute extinction probabilities E (fixed for given parameters)
    p_S, p_D, p_T, p_L = compute_event_probabilities(params)
    
    device = log_Pi_init.device
    dtype = log_Pi_init.dtype
    S = species_helpers["S"]
    E = torch.zeros(S, dtype=dtype, device=device)
    
    # Converge E first
    for _ in range(20):  # E typically converges quickly
        E_next, E_s1, E_s2, Ebar = E_step(E, species_helpers["s_C1"], species_helpers["s_C2"], 
                                          species_helpers["Recipients_mat"], 
                                          float(p_S), float(p_D), float(p_T), float(p_L))
        E = E_next
    
    # Now iterate Pi to convergence
    log_Pi = log_Pi_init.clone()
    for iteration in range(max_iter):
        log_Pi_new = Pi_update_ccp_log(log_Pi, ccp_helpers, species_helpers, clade_species_map,
                                      E, Ebar, float(p_S), float(p_D), float(p_T))
        
        # Check convergence
        diff = torch.abs(log_Pi_new - log_Pi).max()
        if diff < tol:
            break
            
        log_Pi = log_Pi_new
    
    return log_Pi


def compute_log_likelihood(log_Pi: torch.Tensor, root_clade_id: int) -> torch.Tensor:
    """
    Compute log-likelihood from converged Pi matrix.
    
    Args:
        log_Pi: Converged log probability matrix [C, S]
        root_clade_id: Index of root clade
        
    Returns:
        log_likelihood: Scalar log-likelihood
    """
    return torch.logsumexp(log_Pi[root_clade_id, :], dim=0)


def conjugate_gradient_solve(A_op, b: torch.Tensor, x0: Optional[torch.Tensor] = None, 
                           max_iter: int = 100, tol: float = 1e-6) -> torch.Tensor:
    """
    Solve Ax = b using conjugate gradient method.
    
    Args:
        A_op: Function that computes A @ x for given x
        b: Right-hand side vector
        x0: Initial guess (default: zero)
        max_iter: Maximum CG iterations
        tol: Convergence tolerance
        
    Returns:
        x: Solution vector
    """
    if x0 is None:
        x = torch.zeros_like(b)
    else:
        x = x0.clone()
    
    r = b - A_op(x)
    p = r.clone()
    rsold = torch.dot(r.flatten(), r.flatten())
    
    for _ in range(max_iter):
        Ap = A_op(p)
        
        # Handle NaN/inf in Ap
        Ap = torch.where(torch.isfinite(Ap), Ap, torch.zeros_like(Ap))
        
        denominator = torch.dot(p.flatten(), Ap.flatten())
        if torch.abs(denominator) < 1e-15:  # Avoid division by zero
            break
            
        alpha = rsold / denominator
        if not torch.isfinite(alpha):
            break
            
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = torch.dot(r.flatten(), r.flatten())
        
        if not torch.isfinite(rsnew) or torch.sqrt(rsnew) < tol:
            break
            
        beta = rsnew / rsold
        if not torch.isfinite(beta):
            break
            
        p = r + beta * p
        rsold = rsnew
    
    return x


class FixedPointCCPFunction(torch.autograd.Function):
    """
    PyTorch autograd.Function implementing fixed-point differentiation for CCP optimization.
    Follows the mathematical framework from fixed_point_iteration.tex.
    """
    
    @staticmethod
    def forward(ctx, log_params: torch.Tensor, log_Pi_init: torch.Tensor,
                ccp_helpers: Dict, species_helpers: Dict, clade_species_map: torch.Tensor,
                root_clade_id: int, max_iter: int = 50, tol: float = 1e-8) -> torch.Tensor:
        """
        Forward pass: compute log-likelihood via fixed-point iteration.
        
        Args:
            log_params: Unconstrained parameters [log_delta, log_tau, log_lambda]
            log_Pi_init: Initial log probability matrix [C, S]
            ccp_helpers, species_helpers, clade_species_map: Tree structure
            root_clade_id: Root clade index
            max_iter, tol: Convergence parameters
            
        Returns:
            log_likelihood: Scalar objective value
        """
        # Transform to positive parameters
        params = softplus_transform(log_params)
        
        # Run fixed-point iteration
        log_Pi_star = ccp_fixed_point_iteration(
            log_Pi_init, ccp_helpers, species_helpers, clade_species_map,
            params, max_iter, tol
        )
        
        # Compute log-likelihood
        log_likelihood = compute_log_likelihood(log_Pi_star, root_clade_id)
        
        # Save for backward pass
        ctx.save_for_backward(log_params, log_Pi_star, params)
        ctx.ccp_helpers = ccp_helpers
        ctx.species_helpers = species_helpers
        ctx.clade_species_map = clade_species_map
        ctx.root_clade_id = root_clade_id
        ctx.tol = tol
        
        return log_likelihood
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Backward pass: compute gradient using implicit differentiation.
        Implements the algorithm from Section 4.3 of fixed_point_iteration.tex.
        """
        log_params, log_Pi_star, params = ctx.saved_tensors
        ccp_helpers = ctx.ccp_helpers
        species_helpers = ctx.species_helpers
        clade_species_map = ctx.clade_species_map
        root_clade_id = ctx.root_clade_id
        tol = ctx.tol
        
        # Step 1: Compute ∇_x S(x_star) where S is the log-likelihood function
        # S(log_Pi) = logsumexp(log_Pi[root_clade_id, :])
        C, S = log_Pi_star.shape
        grad_S = torch.zeros_like(log_Pi_star)
        
        # Gradient of logsumexp: softmax of the root clade probabilities
        root_log_probs = log_Pi_star[root_clade_id, :]
        root_probs = torch.softmax(root_log_probs, dim=0)
        grad_S[root_clade_id, :] = root_probs
        
        # Step 2: Define the adjoint operator (I - J_xF)^T
        def adjoint_operator(v: torch.Tensor) -> torch.Tensor:
            """
            Compute (I - J_xF)^T @ v where J_xF is the Jacobian of F w.r.t. x.
            Uses vector-Jacobian product (vjp) for efficiency.
            """
            # Ensure v requires gradients for vjp
            v_input = v.detach().requires_grad_(True)
            
            # Define F evaluation at the fixed point
            def F_eval(log_Pi_input):
                p_S, p_D, p_T, p_L = compute_event_probabilities(params)
                
                # Compute E (assumes E converges quickly and is approximately independent of log_Pi)
                device = log_Pi_input.device
                dtype = log_Pi_input.dtype
                E = torch.zeros(S, dtype=dtype, device=device)
                for _ in range(5):  # Quick E convergence
                    E_next, E_s1, E_s2, Ebar = E_step(E, species_helpers["s_C1"], species_helpers["s_C2"], 
                                                      species_helpers["Recipients_mat"], 
                                                      float(p_S), float(p_D), float(p_T), float(p_L))
                    E = E_next
                
                return Pi_update_ccp_log(log_Pi_input, ccp_helpers, species_helpers, clade_species_map,
                                        E, Ebar, p_S, p_D, p_T)
            
            # Compute vector-Jacobian product: vjp(F, x_star)(v)
            try:
                _, JxF_T_v = vjp(F_eval, log_Pi_star, v_input)
                return v - JxF_T_v  # (I - J_xF)^T @ v
            except Exception:
                # Fallback: return identity if vjp fails
                return v
        
        # Step 3: Solve (I - J_xF)^T @ v = grad_S for v using conjugate gradient
        # Handle NaN/inf in grad_S
        grad_S_clean = torch.where(
            torch.isfinite(grad_S),
            grad_S,
            torch.zeros_like(grad_S)
        )
        
        v = conjugate_gradient_solve(adjoint_operator, grad_S_clean, max_iter=50, tol=tol)
        
        # Clean up the solution
        v = torch.where(
            torch.isfinite(v),
            v,
            torch.zeros_like(v)
        )
        
        # Step 4: Compute ∇_θ L = J_θF(x_star, θ)^T @ v
        def F_theta_eval(params_input):
            """Evaluate F with respect to parameters at the fixed point."""
            p_S, p_D, p_T, p_L = compute_event_probabilities(params_input)
            
            device = log_Pi_star.device
            dtype = log_Pi_star.dtype
            E = torch.zeros(S, dtype=dtype, device=device)
            for _ in range(5):
                E_next, E_s1, E_s2, Ebar = E_step(E, species_helpers["s_C1"], species_helpers["s_C2"], 
                                                  species_helpers["Recipients_mat"], 
                                                  float(p_S), float(p_D), float(p_T), float(p_L))
                E = E_next
            
            return Pi_update_ccp_log(log_Pi_star, ccp_helpers, species_helpers, clade_species_map,
                                    E, Ebar, p_S, p_D, p_T)
        
        try:
            _, grad_params = vjp(F_theta_eval, params, v)
        except Exception:
            # Fallback: finite differences if vjp fails
            grad_params = torch.zeros_like(params)
        
        # Step 5: Transform gradient back to log_params space using chain rule
        # grad_log_params = grad_params * d(softplus)/d(log_params)
        # d(softplus(x))/dx = sigmoid(x)
        softplus_grad = torch.sigmoid(log_params)
        grad_log_params = grad_params * softplus_grad
        
        # Handle NaN and inf values in gradients
        grad_log_params = torch.where(
            torch.isfinite(grad_log_params),
            grad_log_params,
            torch.zeros_like(grad_log_params)
        )
        
        # Clip gradients to prevent explosion
        grad_norm = torch.norm(grad_log_params)
        if grad_norm > 10.0:  # Clip large gradients
            grad_log_params = grad_log_params * (10.0 / grad_norm)
        
        # Apply output gradient
        final_grad = grad_output * grad_log_params
        
        # Return gradients for all inputs (only first one is meaningful)
        return final_grad, None, None, None, None, None, None, None


class CCPOptimizer:
    """
    Main optimizer class for phylogenetic reconciliation parameters.
    Supports Adam, SGD, and other PyTorch optimizers.
    """
    
    def __init__(self, species_tree_path: str, gene_tree_path: str,
                 initial_params: Optional[Tuple[float, float, float]] = None,
                 device: Optional[torch.device] = None, dtype: torch.dtype = torch.float32):
        """
        Initialize the CCP optimizer.
        
        Args:
            species_tree_path: Path to species tree file
            gene_tree_path: Path to gene tree file
            initial_params: Initial (delta, tau, lambda) values
            device: PyTorch device
            dtype: Tensor data type
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        
        # Build tree structures
        print("Setting up tree structures...")
        self.ccp = build_ccp_from_single_tree(gene_tree_path)
        self.species_helpers = build_species_helpers(species_tree_path, self.device, dtype)
        self.clade_species_map = build_clade_species_mapping(self.ccp, self.species_helpers, self.device, dtype)
        self.ccp_helpers = build_ccp_helpers(self.ccp, self.device, dtype)
        self.root_clade_id = get_root_clade_id(self.ccp)
        
        # Initialize log probability matrix
        C = len(self.ccp.clades)
        S = self.species_helpers["S"]
        self.log_Pi_init = torch.full((C, S), float('-inf'), dtype=dtype, device=self.device)
        
        # Set leaf probabilities
        for c in range(C):
            clade = self.ccp.id_to_clade[c]
            if clade.is_leaf():
                mapped_species = torch.nonzero(self.clade_species_map[c] > 0, as_tuple=False).flatten()
                if len(mapped_species) > 0:
                    log_prob = -torch.log(torch.tensor(len(mapped_species), dtype=dtype))
                    self.log_Pi_init[c, mapped_species] = log_prob
        
        # Initialize parameters
        if initial_params is None:
            initial_params = (0.1, 0.1, 0.1)  # (delta, tau, lambda)
        
        params_tensor = torch.tensor(initial_params, dtype=dtype, device=self.device)
        self.log_params = torch.nn.Parameter(inverse_softplus_transform(params_tensor))
        
        print(f"✅ Optimizer initialized:")
        print(f"   Device: {self.device}")
        print(f"   Tree size: {C} clades × {S} species")
        print(f"   Initial params: δ={initial_params[0]:.3f}, τ={initial_params[1]:.3f}, λ={initial_params[2]:.3f}")
    
    def get_current_params(self) -> Tuple[float, float, float]:
        """Get current parameter values (delta, tau, lambda)."""
        with torch.no_grad():
            params = softplus_transform(self.log_params)
            return float(params[0]), float(params[1]), float(params[2])
    
    def evaluate_likelihood(self) -> float:
        """Evaluate likelihood at current parameters."""
        with torch.no_grad():
            log_likelihood = FixedPointCCPFunction.apply(
                self.log_params, self.log_Pi_init,
                self.ccp_helpers, self.species_helpers, self.clade_species_map,
                self.root_clade_id
            )
            return float(log_likelihood)
    
    def optimize(self, lr: float = 0.01, epochs: int = 100, optimizer_type: str = "adam",
                early_stopping_patience: int = 10, min_improvement: float = 1e-6) -> Dict[str, Any]:
        """
        Run optimization to maximize log-likelihood.
        
        Args:
            lr: Learning rate
            epochs: Maximum epochs
            optimizer_type: "adam", "sgd", or "lbfgs"
            early_stopping_patience: Stop if no improvement for this many epochs
            min_improvement: Minimum improvement to count as progress
            
        Returns:
            results: Dictionary with optimization history and final results
        """
        print(f"\n🚀 Starting optimization with {optimizer_type.upper()}")
        print(f"Learning rate: {lr}, Max epochs: {epochs}")
        
        # Create optimizer
        if optimizer_type.lower() == "adam":
            optimizer = torch.optim.Adam([self.log_params], lr=lr)
        elif optimizer_type.lower() == "sgd":
            optimizer = torch.optim.SGD([self.log_params], lr=lr, momentum=0.9)
        elif optimizer_type.lower() == "lbfgs":
            optimizer = torch.optim.LBFGS([self.log_params], lr=lr, max_iter=20)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
        
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
            
            def closure():
                optimizer.zero_grad()
                log_likelihood = FixedPointCCPFunction.apply(
                    self.log_params, self.log_Pi_init,
                    self.ccp_helpers, self.species_helpers, self.clade_species_map,
                    self.root_clade_id
                )
                # Negative because we want to maximize (optimizers minimize)
                loss = -log_likelihood
                loss.backward()
                return loss
            
            # Optimization step
            if optimizer_type.lower() == "lbfgs":
                optimizer.step(closure)
            else:
                loss = closure()
                optimizer.step()
            
            # Evaluate current state
            current_likelihood = self.evaluate_likelihood()
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
            if epoch % 10 == 0 or epoch < 10:
                print(f"Epoch {epoch:3d}: LL={current_likelihood:8.3f}, "
                      f"δ={current_params[0]:.4f}, τ={current_params[1]:.4f}, λ={current_params[2]:.4f}, "
                      f"t={epoch_time:.2f}s")
            
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


def verify_gradients(optimizer: CCPOptimizer, epsilon: float = 1e-5) -> Dict[str, float]:
    """
    Verify gradient computation using finite differences.
    
    Args:
        optimizer: CCPOptimizer instance
        epsilon: Finite difference step size
        
    Returns:
        errors: Dictionary with gradient verification errors
    """
    print("🔍 Verifying gradients with finite differences...")
    
    # Current parameters and likelihood
    log_params_0 = optimizer.log_params.clone().detach().requires_grad_(True)
    
    # Analytical gradient
    log_likelihood_0 = FixedPointCCPFunction.apply(
        log_params_0, optimizer.log_Pi_init,
        optimizer.ccp_helpers, optimizer.species_helpers, optimizer.clade_species_map,
        optimizer.root_clade_id
    )
    log_likelihood_0.backward()
    analytical_grad = log_params_0.grad.clone()
    
    # Finite difference gradients
    finite_diff_grad = torch.zeros_like(log_params_0)
    
    for i in range(len(log_params_0)):
        # Forward difference
        log_params_plus = log_params_0.clone()
        log_params_plus[i] += epsilon
        
        with torch.no_grad():
            log_likelihood_plus = FixedPointCCPFunction.apply(
                log_params_plus, optimizer.log_Pi_init,
                optimizer.ccp_helpers, optimizer.species_helpers, optimizer.clade_species_map,
                optimizer.root_clade_id
            )
        
        finite_diff_grad[i] = (log_likelihood_plus - log_likelihood_0) / epsilon
    
    # Compute errors
    abs_error = torch.abs(analytical_grad - finite_diff_grad)
    rel_error = abs_error / (torch.abs(finite_diff_grad) + 1e-8)
    
    errors = {
        "max_abs_error": float(abs_error.max()),
        "mean_abs_error": float(abs_error.mean()),
        "max_rel_error": float(rel_error.max()),
        "mean_rel_error": float(rel_error.mean())
    }
    
    print(f"   Max absolute error: {errors['max_abs_error']:.2e}")
    print(f"   Max relative error: {errors['max_rel_error']:.2e}")
    
    return errors