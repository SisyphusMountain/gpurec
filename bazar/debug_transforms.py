#!/usr/bin/env python3
"""
Debug parameter transformation issues specifically.
"""

import torch
import torch.nn.functional as F

def debug_softplus_gradient():
    """Debug the softplus gradient computation."""
    print("🔧 Debugging softplus gradient...")
    
    x = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0], requires_grad=True)
    
    # Test regular softplus
    y1 = F.softplus(x)
    print(f"Input: {x.data}")
    print(f"Softplus: {y1.data}")
    
    # Test gradient of softplus
    loss1 = y1.sum()
    loss1.backward()
    print(f"Softplus gradient: {x.grad}")
    print(f"Analytical (sigmoid): {torch.sigmoid(x.data)}")
    
    # Reset gradient
    x.grad.zero_()
    
    # Test log(softplus(x))
    y2 = torch.log(F.softplus(x))
    print(f"Log(softplus): {y2.data}")
    
    loss2 = y2.sum()
    loss2.backward()
    print(f"Log(softplus) gradient: {x.grad}")
    
    # Analytical gradient of log(softplus(x)) = x - log(1 + exp(-x))
    # d/dx = 1 - (-exp(-x))/(1 + exp(-x)) = 1 - sigmoid(-x) = sigmoid(x)
    analytical = torch.sigmoid(x.data)
    print(f"Analytical log(softplus) gradient: {analytical}")
    print(f"Error: {torch.abs(x.grad - analytical).max()}")

def debug_inverse_softplus():
    """Debug the inverse softplus function and its gradient."""
    print("\n🔧 Debugging inverse softplus...")
    
    from matmul_ale_ccp_optimize import softplus_transform, inverse_softplus_transform
    
    # Test values
    params = torch.tensor([0.01, 0.1, 1.0, 10.0], requires_grad=True)
    print(f"Original params: {params.data}")
    
    # Forward transform
    log_params = inverse_softplus_transform(params)
    print(f"Log params: {log_params}")
    
    # Check round trip
    recovered = softplus_transform(log_params)
    print(f"Recovered: {recovered}")
    print(f"Round-trip error: {torch.abs(params - recovered).max()}")
    
    # Test gradient through inverse transform
    loss = log_params.sum()
    loss.backward()
    print(f"Gradient of inverse_softplus: {params.grad}")
    
    # The gradient of inverse_softplus(x) should be d/dx log(exp(x) - 1)
    # For log(exp(x) - 1), the derivative is exp(x)/(exp(x) - 1) = 1/(1 - exp(-x))
    # But we need to be careful about numerical stability
    
    analytical_grad = 1.0 / (1.0 - torch.exp(-params.data))
    print(f"Analytical gradient: {analytical_grad}")
    print(f"Gradient error: {torch.abs(params.grad - analytical_grad).max()}")

def debug_chain_rule():
    """Debug the full chain rule through parameter transforms."""
    print("\n🔧 Debugging chain rule through transforms...")
    
    from matmul_ale_ccp_optimize import softplus_transform, inverse_softplus_transform
    
    # Start with log parameters
    log_params = torch.tensor([-3.0, -1.0, 0.0], requires_grad=True)
    print(f"Log params: {log_params.data}")
    
    # Transform to positive parameters
    params = softplus_transform(log_params)
    print(f"Params: {params.data}")
    
    # Simple objective function: sum of squares
    objective = (params ** 2).sum()
    print(f"Objective: {objective.data}")
    
    # Backward pass
    objective.backward()
    print(f"Gradient w.r.t. log_params: {log_params.grad}")
    
    # Manual chain rule check:
    # d(objective)/d(log_params) = d(objective)/d(params) * d(params)/d(log_params)
    # d(objective)/d(params) = 2 * params
    # d(params)/d(log_params) = d(softplus)/d(log_params) = sigmoid(log_params)
    
    manual_grad = 2 * params.data * torch.sigmoid(log_params.data)
    print(f"Manual chain rule: {manual_grad}")
    print(f"Chain rule error: {torch.abs(log_params.grad - manual_grad).max()}")

def test_ccp_function_gradient():
    """Test gradient computation through the CCP function."""
    print("\n🔧 Testing CCP function gradient...")
    
    from matmul_ale_ccp_optimize import FixedPointCCPFunction, CCPOptimizer
    
    # Create a small optimizer
    optimizer = CCPOptimizer(
        "test_trees_1/sp.nwk", "test_trees_1/g.nwk",
        initial_params=(0.1, 0.1, 0.1),
        device=torch.device("cpu"),
        dtype=torch.float64
    )
    
    # Test gradient computation
    log_params = optimizer.log_params.clone().detach().requires_grad_(True)
    print(f"Log params: {log_params.data}")
    
    # Forward pass
    log_likelihood = FixedPointCCPFunction.apply(
        log_params, optimizer.log_Pi_init,
        optimizer.ccp_helpers, optimizer.species_helpers, optimizer.clade_species_map,
        optimizer.root_clade_id
    )
    print(f"Log likelihood: {log_likelihood}")
    
    # Backward pass
    try:
        log_likelihood.backward()
        print(f"Gradient computed: {log_params.grad}")
        print(f"Gradient norm: {log_params.grad.norm()}")
        
        # Check if gradient is reasonable (not all zeros or all ones)
        if log_params.grad.norm() < 1e-10:
            print("⚠️ Gradient is too small (likely zero)")
        elif log_params.grad.norm() > 1e10:
            print("⚠️ Gradient is too large (likely numerical instability)")
        else:
            print("✅ Gradient looks reasonable")
            
    except Exception as e:
        print(f"❌ Error computing gradient: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🐛 Parameter Transformation Debugging")
    print("=" * 50)
    
    debug_softplus_gradient()
    debug_inverse_softplus()
    debug_chain_rule()
    test_ccp_function_gradient()