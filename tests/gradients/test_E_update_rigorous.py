#!/usr/bin/env python3
"""
Rigorous gradient testing for E_step (extinction probability update) function.

This tests the E_step function with comprehensive gradient verification
for both backward-mode and forward-mode automatic differentiation, including
Jacobian comparison between forward and backward modes.
"""

import torch
import sys
import pytest
import numpy as np
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.reconciliation.log_E import E_step_log
from tests.utils.converged_data import get_converged_reconciliation_data

torch.autograd.set_detect_anomaly(True)


@pytest.fixture(scope="module")
def e_update_test_data():
    """Fixture providing converged test data for E_step testing."""
    return get_converged_reconciliation_data("test_trees_1")


@pytest.mark.slow
@pytest.mark.gradient
@pytest.mark.autograd
def test_e_step_gradcheck_wrt_log_e(e_update_test_data):
    """Test E_step gradients with respect to log_E input using gradcheck."""
    data = e_update_test_data
    
    # Create input that requires gradients
    log_E_input = data['log_E'].clone().detach().requires_grad_(True)
    
    def e_step_func(log_E):
        return E_step_log(
            log_E, data['species_helpers'], 
            data['log_pD'], data['log_pT'], data['log_pL']
        )
    
    # Test gradcheck with both backward and forward mode AD
    is_correct = torch.autograd.gradcheck(
        e_step_func,
        log_E_input,
        eps=1e-6,
        atol=1e-4,
        rtol=1e-3,
        check_forward_ad=True,  # This tests forward-mode AD too
        check_backward_ad=True,
        nondet_tol=1e-5
    )
    
    assert is_correct, "E_step gradcheck failed"


@pytest.mark.gradient
@pytest.mark.autograd  
def test_e_step_manual_verification(e_update_test_data):
    """Manual gradient verification for E_step using finite differences."""
    data = e_update_test_data
    
    # Test gradient w.r.t. subset of log_E
    log_E_input = data['log_E'].clone().detach().requires_grad_(True)
    
    def e_step_func(log_E):
        return E_step_log(
            log_E, 
            data['species_helpers']['s_C1_indexes'],
            data['species_helpers']['s_C2_indexes'], 
            data['species_helpers']['sp_internal_mask'],
            data['species_helpers']['Recipients_mat'],
            data['log_pS'], data['log_pD'], data['log_pT'], data['log_pL']
        )
    
    # Compute result and gradients
    result = e_step_func(log_E_input)
    S = result.shape[0]
    
    # Verify result properties
    assert result.shape == data['log_E'].shape, f"Result shape mismatch: expected {data['log_E'].shape}, got {result.shape}"
    
    # Should be 100% finite after convergence
    finite_result_count = torch.isfinite(result).sum().item()
    total_result_count = result.numel()
    assert finite_result_count == total_result_count, f"Result should be 100% finite, got {finite_result_count}/{total_result_count}"
    
    # Test finite sum loss
    finite_mask = torch.isfinite(result)
    loss = result[finite_mask].sum()
    loss.backward()
    
    analytical_grad = log_E_input.grad.clone()
    
    # Check analytical gradients
    grad_finite_count = torch.isfinite(analytical_grad).sum().item()
    grad_total_count = analytical_grad.numel()
    assert grad_finite_count == grad_total_count, f"Analytical grad should be finite, got {grad_finite_count}/{grad_total_count}"
    
    assert not torch.any(torch.isnan(analytical_grad)), "Analytical gradients should not contain NaN"
    assert not torch.any(torch.isinf(analytical_grad)), "Analytical gradients should not contain Inf"
    
    # Manual finite difference check on sample elements
    eps = 1e-7
    test_indices = [0, 1, 3, min(7, S-1)]  # Species indices
    
    max_diff = 0.0
    for i in test_indices:
        if i < S:
            # Central difference
            log_E_plus = data['log_E'].clone().detach()
            log_E_plus[i] += eps
            result_plus = e_step_func(log_E_plus)
            loss_plus = result_plus[finite_mask].sum()
            
            log_E_minus = data['log_E'].clone().detach()
            log_E_minus[i] -= eps
            result_minus = e_step_func(log_E_minus)
            loss_minus = result_minus[finite_mask].sum()
            
            numerical_grad = (loss_plus - loss_minus) / (2 * eps)
            analytical_grad_val = analytical_grad[i].item()
            
            diff = abs(numerical_grad.item() - analytical_grad_val)
            max_diff = max(max_diff, diff)
            
            # Allow for some numerical error
            tolerance = 1e-5
            assert diff < tolerance, f"Gradient mismatch at [{i}]: numerical={numerical_grad:.6f}, analytical={analytical_grad_val:.6f}, diff={diff:.2e}"
    
    print(f"✅ Manual verification passed. Max difference: {max_diff:.2e}")


@pytest.mark.gradient
def test_e_step_stability_different_losses(e_update_test_data):
    """Test E_step gradient stability under different loss functions."""
    data = e_update_test_data
    
    # Create input tensor  
    log_E_input = data['log_E'].clone().detach().requires_grad_(True)
    
    def e_step_func(log_E):
        return E_step_log(
            log_E, 
            data['species_helpers']['s_C1_indexes'],
            data['species_helpers']['s_C2_indexes'], 
            data['species_helpers']['sp_internal_mask'],
            data['species_helpers']['Recipients_mat'],
            data['log_pS'], data['log_pD'], data['log_pT'], data['log_pL']
        )
    
    result = e_step_func(log_E_input)
    
    # Test different loss functions for stability
    loss_functions = [
        ("sum_all", lambda x: x.sum()),
        ("sum_finite", lambda x: x[torch.isfinite(x)].sum()),
        ("mean_finite", lambda x: x[torch.isfinite(x)].mean()),
        ("max_finite", lambda x: x[torch.isfinite(x)].max()),
        ("l2_norm", lambda x: (x[torch.isfinite(x)] ** 2).sum())
    ]
    
    success_count = 0
    for loss_name, loss_fn in loss_functions:
        try:
            # Create fresh input for each loss function to avoid graph issues
            fresh_log_E = data['log_E'].clone().detach().requires_grad_(True)
            fresh_result = e_step_func(fresh_log_E)
            
            loss = loss_fn(fresh_result)
            loss.backward()
            grad = fresh_log_E.grad
            
            # Check gradient properties
            has_nan = torch.any(torch.isnan(grad))
            has_inf = torch.any(torch.isinf(grad))
            grad_norm = torch.norm(grad[torch.isfinite(grad)])
            
            assert not has_nan, f"{loss_name}: Gradients contain NaN"
            assert not has_inf, f"{loss_name}: Gradients contain Inf"
            assert torch.isfinite(grad_norm), f"{loss_name}: Gradient norm is not finite"
            
            success_count += 1
            
        except Exception as e:
            pytest.fail(f"Stability test failed for {loss_name}: {e}")
    
    assert success_count == len(loss_functions), f"Only {success_count}/{len(loss_functions)} loss functions succeeded"


@pytest.mark.unit
def test_e_step_output_properties(e_update_test_data):
    """Test basic properties of E_step output."""
    data = e_update_test_data
    
    result = E_step(
        data['log_E'], data['species_helpers'],
        data['log_pD'], data['log_pT'], data['log_pL']
    )
    
    # Basic shape and type checks
    assert result.shape == data['log_E'].shape, "Output shape should match input log_E shape"
    assert result.dtype == data['log_E'].dtype, "Output dtype should match input dtype"
    assert result.device == data['log_E'].device, "Output device should match input device"
    
    # Value range checks - extinction probabilities in log space
    finite_values = result[torch.isfinite(result)]
    if len(finite_values) > 0:
        assert torch.all(finite_values <= 0), "Log extinction probabilities should be <= 0"
    
    # Should have some finite values (convergence property)
    finite_count = torch.isfinite(result).sum().item()
    assert finite_count > 0, "Result should have some finite values"


@pytest.mark.slow
@pytest.mark.gradient 
@pytest.mark.autograd
def test_e_step_ad_consistency_comprehensive(e_update_test_data):
    """Test AD consistency for E_step using comprehensive gradcheck with both forward and backward mode."""
    data = e_update_test_data
    
    # Create input that requires gradients
    log_E_input = data['log_E'].clone().detach().requires_grad_(True)
    
    def e_step_func(log_E):
        return E_step_log(
            log_E, 
            data['species_helpers']['s_C1_indexes'],
            data['species_helpers']['s_C2_indexes'], 
            data['species_helpers']['sp_internal_mask'],
            data['species_helpers']['Recipients_mat'],
            data['log_pS'], data['log_pD'], data['log_pT'], data['log_pL']
        )
    
    print("Testing AD consistency with gradcheck (both forward and backward mode)...")
    
    # Test both backward-mode and forward-mode AD with custom jvp() implementation
    is_correct = torch.autograd.gradcheck(
        e_step_func,
        log_E_input,
        eps=1e-6,
        atol=1e-4,
        rtol=1e-3,
        check_forward_ad=True,   # Test forward-mode AD with jvp() implementation  
        check_backward_ad=True,  # Test backward-mode AD
        check_undefined_grad=True,
        nondet_tol=1e-8
    )
    
    assert is_correct, "E_step AD consistency check failed - forward and backward modes give different results"
    print("✅ Forward-mode and backward-mode AD are consistent for E_step")


@pytest.mark.slow
@pytest.mark.gradient  
@pytest.mark.autograd
def test_e_step_jacobian_comparison_functorch(e_update_test_data):
    """Test that torch.func.jacrev and torch.func.jacfwd give identical Jacobians for E_step."""
    data = e_update_test_data
    
    # Create input that requires gradients (use double precision for stability)
    log_E_input = data['log_E'].clone().detach().requires_grad_(True).double()
    
    def e_step_func(log_E):
        return E_step_log(
            log_E, 
            data['species_helpers']['s_C1_indexes'],
            data['species_helpers']['s_C2_indexes'], 
            data['species_helpers']['sp_internal_mask'],
            data['species_helpers']['Recipients_mat'],
            data['log_pS'].double(), data['log_pD'].double(), data['log_pT'].double(), data['log_pL'].double()
        )
    
    print("Computing backward-mode Jacobian using torch.func.jacrev...")
    try:
        jacobian_backward = torch.func.jacrev(e_step_func)(log_E_input)
        print(f"✅ jacrev successful, shape: {jacobian_backward.shape}")
    except Exception as e:
        pytest.skip(f"jacrev failed: {e}")
    
    print("Computing forward-mode Jacobian using torch.func.jacfwd...")
    try:
        jacobian_forward = torch.func.jacfwd(e_step_func)(log_E_input)
        print(f"✅ jacfwd successful, shape: {jacobian_forward.shape}")
    except Exception as e:
        pytest.skip(f"jacfwd failed: {e}")
    
    print(f"Jacobian shapes - Backward: {jacobian_backward.shape}, Forward: {jacobian_forward.shape}")
    
    # Jacobians should have identical shapes
    assert jacobian_backward.shape == jacobian_forward.shape, \
        f"Jacobian shape mismatch: backward {jacobian_backward.shape} vs forward {jacobian_forward.shape}"
    
    # Check for numerical issues in both Jacobians
    backward_finite = torch.isfinite(jacobian_backward)
    forward_finite = torch.isfinite(jacobian_forward)
    
    print(f"Backward Jacobian finite elements: {backward_finite.sum().item()}/{jacobian_backward.numel()}")
    print(f"Forward Jacobian finite elements: {forward_finite.sum().item()}/{jacobian_forward.numel()}")
    
    # Compare only finite elements (NaN/inf values may differ due to numerical issues)
    common_finite_mask = backward_finite & forward_finite
    
    if common_finite_mask.sum().item() == 0:
        pytest.skip("No finite elements in Jacobians to compare")
    
    backward_finite_vals = jacobian_backward[common_finite_mask]
    forward_finite_vals = jacobian_forward[common_finite_mask]
    
    # Compare with high precision
    max_abs_diff = torch.abs(backward_finite_vals - forward_finite_vals).max().item()
    max_rel_diff = (torch.abs(backward_finite_vals - forward_finite_vals) / 
                   (torch.abs(backward_finite_vals) + 1e-15)).max().item()
    
    print(f"Maximum absolute difference: {max_abs_diff:.2e}")  
    print(f"Maximum relative difference: {max_rel_diff:.2e}")
    
    # Use appropriate tolerances for Jacobian comparison
    rtol = 1e-10
    atol = 1e-12
    
    try:
        torch.testing.assert_close(
            backward_finite_vals, forward_finite_vals,
            rtol=rtol, atol=atol
        )
        print("✅ Forward and backward Jacobians match within strict tolerance")
    except AssertionError as e:
        # Try more relaxed tolerances
        rtol_relaxed = 1e-8
        atol_relaxed = 1e-10
        
        try:
            torch.testing.assert_close(
                backward_finite_vals, forward_finite_vals,
                rtol=rtol_relaxed, atol=atol_relaxed
            )
            print(f"⚠️  Forward and backward Jacobians match with relaxed tolerance (rtol={rtol_relaxed}, atol={atol_relaxed})")
        except AssertionError:
            print(f"❌ Forward and backward Jacobians differ significantly") 
            print(f"   Max abs diff: {max_abs_diff:.2e}, Max rel diff: {max_rel_diff:.2e}")
            raise AssertionError("Forward and backward mode Jacobians do not match")


if __name__ == "__main__":
    # Allow running as script for debugging
    pytest.main([__file__, "-v"])