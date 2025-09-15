#!/usr/bin/env python3
"""
Rigorous gradient testing for ScatterLogSumExp autograd function.

Tests both backward-mode (VJP) and forward-mode (JVP) automatic differentiation
using converged reconciliation data to ensure numerical accuracy.
"""

import torch
import sys
import pytest
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.reconciliation.autograd_functions import ScatterLogSumExp
from tests.utils.converged_data import get_converged_reconciliation_data, get_scatter_test_data


@pytest.fixture(scope="module")
def converged_test_data():
    """Fixture providing converged reconciliation data for testing."""
    return get_converged_reconciliation_data("test_trees_1")


@pytest.fixture(scope="module") 
def scatter_test_data(converged_test_data):
    """Fixture providing ScatterLogSumExp-specific test data."""
    return get_scatter_test_data(converged_test_data)


@pytest.mark.slow
@pytest.mark.gradient
@pytest.mark.autograd
def test_scatter_gradcheck_with_converged_data(scatter_test_data):
    """Test ScatterLogSumExp gradcheck using converged computation patterns."""
    # Extract test data
    split_parents = scatter_test_data['split_parents']
    ccp_leaves_mask = scatter_test_data['ccp_leaves_mask']
    C = scatter_test_data['C']
    S = scatter_test_data['S']
    N_splits = len(split_parents)
    
    # Use realistic log values similar to converged computation
    log_combined_splits = torch.randn(N_splits, S, dtype=torch.float64) * 20 - 50
    log_combined_splits.requires_grad_(True)
    
    def scatter_func(x):
        return ScatterLogSumExp.apply(x, split_parents, C, ccp_leaves_mask)
    
    # Test forward pass
    result = scatter_func(log_combined_splits)
    assert result.shape == (C, S), f"Expected output shape ({C}, {S}), got {result.shape}"
    
    # Test backward pass
    finite_result_count = torch.isfinite(result).sum().item()
    total_result_count = result.numel()
    
    # Should have some finite values (non-leaf clades)
    assert finite_result_count > 0, "Result should have some finite values"
    
    # Test gradients on finite outputs only to avoid gradcheck -inf issues
    def scatter_func_finite_only(x):
        full_result = ScatterLogSumExp.apply(x, split_parents, C, ccp_leaves_mask)
        finite_mask = torch.isfinite(full_result)
        return full_result[finite_mask]
    
    log_combined_splits_fresh = torch.randn(N_splits, S, dtype=torch.float64) * 10 - 30
    log_combined_splits_fresh.requires_grad_(True)
    
    # Test gradcheck with both backward and forward mode AD
    is_correct = torch.autograd.gradcheck(
        scatter_func_finite_only,
        log_combined_splits_fresh,
        eps=1e-6,
        atol=1e-5,
        rtol=1e-3,
        check_undefined_grad=True,
        nondet_tol=1e-8,
        check_forward_ad=True
    )
    
    assert is_correct, "ScatterLogSumExp gradcheck failed"


@pytest.mark.gradient
def test_scatter_manual_verification():
    """Manual gradient verification against finite differences."""
    # Simple test case - no leaves to avoid -inf
    N_splits, S, C = 4, 3, 3
    split_parents = torch.tensor([0, 0, 1, 2], dtype=torch.long)
    leaves_mask = torch.tensor([False, False, False], dtype=torch.bool)  # NO LEAVES
    
    log_values = torch.tensor([
        [1.0, 2.0, 0.5],
        [0.8, 1.8, 1.2],
        [1.5, 1.0, 2.0],
        [2.0, 0.5, 1.8]
    ], dtype=torch.float64, requires_grad=True)
    
    # Forward pass
    def scatter_func(x):
        return ScatterLogSumExp.apply(x, split_parents, C, leaves_mask)
    
    result = scatter_func(log_values)
    
    # Should be no -inf since no leaves
    assert torch.all(torch.isfinite(result)), "Result should be 100% finite with no leaves"
    
    # Get analytical gradients
    loss = result.sum()
    loss.backward()
    analytical_grad = log_values.grad.clone()
    
    # Compute numerical gradients manually
    eps = 1e-7
    numerical_grad = torch.zeros_like(log_values)
    
    for i in range(N_splits):
        for j in range(S):
            # Central difference
            log_plus = log_values.clone().detach()
            log_plus[i, j] += eps
            result_plus = scatter_func(log_plus).sum()
            
            log_minus = log_values.clone().detach()  
            log_minus[i, j] -= eps
            result_minus = scatter_func(log_minus).sum()
            
            numerical_grad[i, j] = (result_plus - result_minus) / (2 * eps)
    
    # Compare gradients
    diff = torch.abs(analytical_grad - numerical_grad)
    max_diff = torch.max(diff)
    
    tolerance = 1e-6
    assert max_diff < tolerance, f"Gradient difference {max_diff:.2e} exceeds tolerance {tolerance:.1e}"


@pytest.mark.gradient
def test_scatter_backward_pass_stability(scatter_test_data):
    """Test that backward pass produces stable gradients."""
    split_parents = scatter_test_data['split_parents']
    ccp_leaves_mask = scatter_test_data['ccp_leaves_mask'] 
    C = scatter_test_data['C']
    S = scatter_test_data['S']
    N_splits = len(split_parents)
    
    log_combined_splits = torch.randn(N_splits, S, dtype=torch.float64) * 15 - 40
    log_combined_splits.requires_grad_(True)
    
    def scatter_func(x):
        return ScatterLogSumExp.apply(x, split_parents, C, ccp_leaves_mask)
    
    result = scatter_func(log_combined_splits)
    
    # Test backward pass with finite values only
    finite_mask = torch.isfinite(result)
    if torch.any(finite_mask):
        loss = result[finite_mask].sum()
        loss.backward()
        
        # Check gradient properties
        grad = log_combined_splits.grad
        assert not torch.any(torch.isnan(grad)), "Gradients should not contain NaN"
        assert not torch.any(torch.isinf(grad)), "Gradients should not contain Inf"
        assert grad.dtype == torch.float64, "Gradient dtype should match input"
        assert grad.shape == log_combined_splits.shape, "Gradient shape should match input"


@pytest.mark.slow
@pytest.mark.autograd
def test_scatter_forward_mode_ad():
    """Test forward-mode automatic differentiation (JVP)."""
    # Simple test case
    N_splits, S, C = 4, 3, 3
    split_parents = torch.tensor([0, 0, 1, 2], dtype=torch.long)
    leaves_mask = torch.tensor([False, False, False], dtype=torch.bool)
    
    log_values = torch.tensor([
        [1.0, 2.0, 0.5],
        [0.8, 1.8, 1.2],
        [1.5, 1.0, 2.0],
        [2.0, 0.5, 1.8]
    ], dtype=torch.float64, requires_grad=True)
    
    def scatter_func(x):
        return ScatterLogSumExp.apply(x, split_parents, C, leaves_mask)
    
    # Test gradcheck specifically for forward-mode AD
    is_correct = torch.autograd.gradcheck(
        scatter_func,
        log_values,
        eps=1e-6,
        atol=1e-5,
        rtol=1e-3,
        check_forward_ad=True
    )
    
    assert is_correct, "Forward-mode AD (JVP) gradcheck failed"


if __name__ == "__main__":
    # Allow running as script for debugging
    pytest.main([__file__, "-v"])