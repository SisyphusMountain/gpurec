#!/usr/bin/env python3
"""
Debug script to identify gradient computation issues.
"""

import torch
import torch.nn.functional as F

def test_simple_gradient():
    """Test gradient computation with a simple function."""
    print("🔧 Testing simple gradient computation...")
    
    # Simple test: gradient of log(softplus(x))
    x = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0], requires_grad=True)
    y = torch.log(F.softplus(x)).sum()
    y.backward()
    
    print(f"Input: {x.data}")
    print(f"Output: {y.data}")
    print(f"Gradient: {x.grad}")
    
    # Analytical gradient should be sigmoid(x)
    analytical = torch.sigmoid(x.data)
    print(f"Analytical: {analytical}")
    print(f"Error: {torch.abs(x.grad - analytical).max()}")

def test_parameter_transforms():
    """Test parameter transformation gradients."""
    print("\n🔧 Testing parameter transformation gradients...")
    
    from matmul_ale_ccp_optimize import softplus_transform, inverse_softplus_transform
    
    # Test parameters
    params = torch.tensor([0.1, 0.5, 1.0], requires_grad=True)
    
    # Transform to log space and back
    log_params = inverse_softplus_transform(params)
    recovered_params = softplus_transform(log_params)
    
    # Test gradient through transformation
    loss = (recovered_params - params).pow(2).sum()
    loss.backward()
    
    print(f"Original params: {params.data}")
    print(f"Log params: {log_params.data}")
    print(f"Recovered params: {recovered_params.data}")
    print(f"Reconstruction error: {torch.abs(recovered_params - params).max()}")
    print(f"Gradient through transform: {params.grad}")

def test_basic_fixed_point():
    """Test a very simple fixed point function."""
    print("\n🔧 Testing basic fixed point differentiation...")
    
    def simple_fixed_point_func(x, theta):
        """Simple fixed point: x = 0.5 * x + theta"""
        return 0.5 * x + theta
    
    # Analytical solution: x* = 2 * theta
    # Gradient: d(x*)/d(theta) = 2
    
    theta = torch.tensor([1.0], requires_grad=True)
    
    # Manual fixed point iteration
    x = torch.zeros(1)
    for _ in range(20):
        x = simple_fixed_point_func(x, theta)
    
    print(f"Theta: {theta.data}")
    print(f"Fixed point: {x.data}")
    print(f"Analytical: {2 * theta.data}")
    
    # Try to compute gradient manually
    x.backward()
    print(f"Gradient: {theta.grad}")

def test_log_pi_forward():
    """Test just the forward pass of log Pi computation."""
    print("\n🔧 Testing log Pi forward pass...")
    
    from matmul_ale_ccp_optimize import CCPOptimizer
    
    # Create optimizer
    optimizer = CCPOptimizer(
        "test_trees_1/sp.nwk", "test_trees_1/g.nwk",
        initial_params=(0.1, 0.1, 0.1),
        device=torch.device("cpu"),  # Use CPU for debugging
        dtype=torch.float64  # Use double precision
    )
    
    print(f"Optimizer created successfully")
    print(f"Initial likelihood: {optimizer.evaluate_likelihood()}")
    
    # Test parameter sensitivity by manually changing parameters
    original_params = optimizer.get_current_params()
    print(f"Original params: {original_params}")
    
    # Manually perturb parameters
    new_log_params = optimizer.log_params.data.clone()
    new_log_params[0] += 0.1  # Change delta
    optimizer.log_params.data = new_log_params
    
    new_likelihood = optimizer.evaluate_likelihood()
    new_params = optimizer.get_current_params()
    
    print(f"New params: {new_params}")
    print(f"New likelihood: {new_likelihood}")
    print(f"Likelihood change: {new_likelihood - optimizer.evaluate_likelihood()}")

if __name__ == "__main__":
    print("🐛 Gradient Debugging Session")
    print("=" * 50)
    
    try:
        test_simple_gradient()
        test_parameter_transforms()
        test_basic_fixed_point()
        test_log_pi_forward()
        
        print("\n✅ All debug tests completed")
        
    except Exception as e:
        print(f"❌ Error during debugging: {e}")
        import traceback
        traceback.print_exc()