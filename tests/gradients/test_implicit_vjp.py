#!/usr/bin/env python3
"""
Test implicit VJP differentiation for fixed-point reconciliation.

This module tests the correctness of implicit gradient computation
by comparing with finite differences and unrolled gradients.
"""

import torch
import numpy as np
import pytest
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.reconciliation.implicit_vjp import (
    solve_adjoint_fixedpoint,
    implicit_grad_L_vjp,
    implicit_param_grads
)
from src.reconciliation.reconcile import fixed_points
from src.reconciliation.likelihood import E_step, Pi_step
from src.core.ccp import (
    build_ccp_from_single_tree, 
    get_root_clade_id, 
    build_ccp_helpers,
    build_clade_species_mapping
)
from src.core.tree_helpers import build_species_helpers


class TestImplicitVJP:
    """Test suite for implicit VJP differentiation."""
    
    @pytest.fixture(scope="class")
    def test_data(self):
        """Prepare test data for all tests."""
        species_tree = "test_trees_1/sp.nwk"
        gene_tree = "test_trees_1/g.nwk"
        
        # Parameters
        tau = 0.1
        delta = 0.15
        lambda_param = 0.05
        
        device = torch.device("cpu")  # Use CPU for testing
        dtype = torch.float64  # Use float64 for accuracy
        
        # Build structures
        ccp = build_ccp_from_single_tree(gene_tree)
        species_helpers = build_species_helpers(species_tree, device, dtype)
        ccp_helpers = build_ccp_helpers(ccp, device, dtype)
        
        # Fix log_split_probs
        split_probs = ccp_helpers['split_probs']
        log_split_probs = torch.where(
            split_probs == 0,
            torch.full_like(split_probs, float('-inf')),
            torch.log(split_probs)
        )
        ccp_helpers['log_split_probs'] = log_split_probs
        
        # Build mappings
        clade_species_map = build_clade_species_mapping(ccp, species_helpers, device, dtype)
        log_clade_species_map = torch.log(clade_species_map + 1e-45)
        log_clade_species_map[clade_species_map == 0] = float('-inf')
        
        return {
            'species_tree': species_tree,
            'gene_tree': gene_tree,
            'tau': tau,
            'delta': delta,
            'lambda': lambda_param,
            'device': device,
            'dtype': dtype,
            'ccp': ccp,
            'species_helpers': species_helpers,
            'ccp_helpers': ccp_helpers,
            'log_clade_species_map': log_clade_species_map
        }
    
    def test_solve_adjoint_fixedpoint(self, test_data):
        """Test the adjoint fixed-point solver."""
        device = test_data['device']
        dtype = test_data['dtype']
        
        # Create a simple contractive linear operator
        # J = 0.5 * I (spectral radius = 0.5)
        n = 10
        J = 0.5 * torch.eye(n, device=device, dtype=dtype)
        
        def vjp_apply(x):
            return J.T @ x
        
        rhs = torch.randn(n, device=device, dtype=dtype)
        
        # Solve (I - J^T) x = rhs
        x = solve_adjoint_fixedpoint(vjp_apply, rhs, max_iter=100, tol=1e-10)
        
        # Check solution: x should satisfy (I - J^T) x = rhs
        residual = x - J.T @ x - rhs
        assert torch.allclose(residual, torch.zeros_like(residual), atol=1e-9)
    
    def test_finite_difference_gradients(self, test_data):
        """Test implicit gradients against finite differences."""
        # Extract test data
        species_tree = test_data['species_tree']
        gene_tree = test_data['gene_tree']
        tau = test_data['tau']
        delta = test_data['delta']
        lambda_param = test_data['lambda']
        device = test_data['device']
        dtype = test_data['dtype']
        
        # Compute implicit gradients
        grad_tau, grad_delta, grad_lambda = implicit_param_grads(
            species_tree, gene_tree,
            tau, delta, lambda_param,
            e_iters=50, pi_iters=50,
            device=device, dtype=dtype,
            verbose=False
        )
        
        # Compute finite difference gradients
        epsilon = 1e-6
        
        # Helper function to compute likelihood
        def compute_likelihood(t, d, l):
            result = fixed_points(
                species_tree, gene_tree,
                delta=d, tau=t, lambda_param=l,
                iters=50,
                device=device, dtype=dtype,
                debug=False
            )
            return result['log_likelihood']
        
        # Base likelihood
        L0 = compute_likelihood(tau, delta, lambda_param)
        
        # Finite difference for tau
        L_tau_plus = compute_likelihood(tau + epsilon, delta, lambda_param)
        fd_grad_tau = (L_tau_plus - L0) / epsilon
        
        # Finite difference for delta
        L_delta_plus = compute_likelihood(tau, delta + epsilon, lambda_param)
        fd_grad_delta = (L_delta_plus - L0) / epsilon
        
        # Finite difference for lambda
        L_lambda_plus = compute_likelihood(tau, delta, lambda_param + epsilon)
        fd_grad_lambda = (L_lambda_plus - L0) / epsilon
        
        # Compare gradients (relaxed tolerance due to finite difference approximation)
        print(f"\nGradient comparison:")
        print(f"tau:    implicit={grad_tau:.6e}, fd={fd_grad_tau:.6e}, diff={abs(grad_tau - fd_grad_tau):.6e}")
        print(f"delta:  implicit={grad_delta:.6e}, fd={fd_grad_delta:.6e}, diff={abs(grad_delta - fd_grad_delta):.6e}")
        print(f"lambda: implicit={grad_lambda:.6e}, fd={fd_grad_lambda:.6e}, diff={abs(grad_lambda - fd_grad_lambda):.6e}")
        
        # Assert close (with reasonable tolerance for finite differences)
        assert abs(grad_tau - fd_grad_tau) < 1e-4, f"tau gradient mismatch: {grad_tau} vs {fd_grad_tau}"
        assert abs(grad_delta - fd_grad_delta) < 1e-4, f"delta gradient mismatch: {grad_delta} vs {fd_grad_delta}"
        assert abs(grad_lambda - fd_grad_lambda) < 1e-4, f"lambda gradient mismatch: {grad_lambda} vs {fd_grad_lambda}"
    
    def test_gradient_stability(self, test_data):
        """Test gradient computation stability across parameter ranges."""
        species_tree = test_data['species_tree']
        gene_tree = test_data['gene_tree']
        device = test_data['device']
        dtype = test_data['dtype']
        
        # Test different parameter values
        test_params = [
            (0.01, 0.01, 0.01),  # Small values
            (0.1, 0.1, 0.1),     # Moderate values
            (0.5, 0.5, 0.5),     # Large values
        ]
        
        for tau, delta, lambda_param in test_params:
            print(f"\nTesting parameters: tau={tau}, delta={delta}, lambda={lambda_param}")
            
            try:
                grad_tau, grad_delta, grad_lambda = implicit_param_grads(
                    species_tree, gene_tree,
                    tau, delta, lambda_param,
                    e_iters=30, pi_iters=30,
                    device=device, dtype=dtype,
                    verbose=False
                )
                
                # Check that gradients are finite
                assert torch.isfinite(grad_tau), f"tau gradient is not finite: {grad_tau}"
                assert torch.isfinite(grad_delta), f"delta gradient is not finite: {grad_delta}"
                assert torch.isfinite(grad_lambda), f"lambda gradient is not finite: {grad_lambda}"
                
                print(f"  Gradients: tau={grad_tau:.6e}, delta={grad_delta:.6e}, lambda={grad_lambda:.6e}")
                
            except Exception as e:
                pytest.fail(f"Gradient computation failed for params {(tau, delta, lambda_param)}: {e}")
    
    @pytest.mark.slow
    def test_compare_with_unrolled(self, test_data):
        """Compare implicit gradients with unrolled gradients (small problem)."""
        # This test creates a differentiable version with unrolled iterations
        # and compares the gradients
        
        species_tree = test_data['species_tree']
        gene_tree = test_data['gene_tree']
        device = test_data['device']
        dtype = test_data['dtype']
        
        # Use small number of iterations for unrolling
        n_iters = 10
        
        # Parameters as tensors requiring gradients
        tau = torch.tensor(0.1, device=device, dtype=dtype, requires_grad=True)
        delta = torch.tensor(0.15, device=device, dtype=dtype, requires_grad=True)
        lambda_param = torch.tensor(0.05, device=device, dtype=dtype, requires_grad=True)
        
        # Build structures
        ccp = test_data['ccp']
        species_helpers = test_data['species_helpers']
        ccp_helpers = test_data['ccp_helpers']
        log_clade_species_map = test_data['log_clade_species_map']
        
        # Compute event probabilities
        rates_sum = 1.0 + delta + tau + lambda_param
        log_pS = torch.log(1.0 / rates_sum)
        log_pD = torch.log(delta / rates_sum)
        log_pT = torch.log(tau / rates_sum)
        log_pL = torch.log(lambda_param / rates_sum)
        
        # Initialize
        import math
        S = species_helpers["S"]
        C = len(ccp.clades)
        log_E = torch.full((S,), -math.log(2), dtype=dtype, device=device)
        log_Pi = torch.full((C, S), -math.log(2), dtype=dtype, device=device)
        
        # Set leaf probabilities
        from src.reconciliation.reconcile import fixed_points
        clade_species_map = build_clade_species_mapping(ccp, species_helpers, device, dtype)
        for c in range(C):
            clade = ccp.id_to_clade[c]
            if clade.is_leaf():
                mapped_species = torch.nonzero(clade_species_map[c] > 0, as_tuple=False).flatten()
                if len(mapped_species) > 0:
                    log_prob = -torch.log(torch.tensor(len(mapped_species), dtype=dtype))
                    log_Pi[c, mapped_species] = log_prob
        
        # Unrolled iterations (differentiable)
        for _ in range(n_iters):
            log_E, log_E_s1, log_E_s2, log_Ebar = E_step(
                log_E, species_helpers["s_C1_indexes"], species_helpers["s_C2_indexes"],
                species_helpers["sp_internal_mask"], species_helpers["Recipients_mat"],
                log_pS, log_pD, log_pT, log_pL
            )
        
        for _ in range(n_iters):
            log_Pi = Pi_step(
                log_Pi, ccp_helpers, species_helpers, log_clade_species_map,
                log_E, log_Ebar, log_E_s1, log_E_s2, log_pS, log_pD, log_pT
            )
        
        # Compute likelihood
        root_clade_id = get_root_clade_id(ccp)
        log_likelihood = torch.logsumexp(log_Pi[root_clade_id, :], dim=0)
        
        # Compute unrolled gradients
        unrolled_grads = torch.autograd.grad(
            log_likelihood, [tau, delta, lambda_param],
            retain_graph=False
        )
        
        # Compute implicit gradients
        implicit_grads = implicit_param_grads(
            species_tree, gene_tree,
            tau.item(), delta.item(), lambda_param.item(),
            e_iters=n_iters, pi_iters=n_iters,
            device=device, dtype=dtype,
            verbose=False
        )
        
        # Compare
        print(f"\nUnrolled vs Implicit gradients ({n_iters} iterations):")
        print(f"tau:    unrolled={unrolled_grads[0]:.6e}, implicit={implicit_grads[0]:.6e}")
        print(f"delta:  unrolled={unrolled_grads[1]:.6e}, implicit={implicit_grads[1]:.6e}")
        print(f"lambda: unrolled={unrolled_grads[2]:.6e}, implicit={implicit_grads[2]:.6e}")
        
        # They should be similar but not exact (unrolled is not at fixed point)
        # Just check they have the same sign and order of magnitude
        for unrolled, implicit in zip(unrolled_grads, implicit_grads):
            assert torch.sign(unrolled) == torch.sign(implicit), "Gradients have different signs"
            ratio = unrolled / implicit
            assert 0.1 < ratio < 10, f"Gradients differ by more than an order of magnitude: {ratio}"


if __name__ == "__main__":
    # Run tests
    test = TestImplicitVJP()
    
    # Prepare test data
    class TestDataHolder:
        pass
    
    holder = TestDataHolder()
    test_data = test.test_data()
    
    # Run individual tests
    print("Testing adjoint fixed-point solver...")
    test.test_solve_adjoint_fixedpoint(test_data)
    print("✓ Adjoint solver test passed\n")
    
    print("Testing finite difference gradients...")
    test.test_finite_difference_gradients(test_data)
    print("✓ Finite difference test passed\n")
    
    print("Testing gradient stability...")
    test.test_gradient_stability(test_data)
    print("✓ Stability test passed\n")
    
    print("Testing comparison with unrolled gradients...")
    test.test_compare_with_unrolled(test_data)
    print("✓ Unrolled comparison test passed\n")
    
    print("\n✅ All tests passed!")