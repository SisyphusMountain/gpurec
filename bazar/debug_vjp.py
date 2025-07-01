#!/usr/bin/env python3
"""
Debug the vector-Jacobian product computation in fixed-point differentiation.
"""

import torch
from torch.autograd.functional import vjp

def test_simple_vjp():
    """Test VJP with simple functions."""
    print("🔧 Testing simple VJP...")
    
    def f(x):
        return (x ** 2).sum()  # Return scalar
    
    x = torch.tensor([1.0, 2.0, 3.0])
    
    # Compute VJP
    y, vjp_func = vjp(f, x)
    grad = vjp_func(torch.tensor(1.0))[0]  # v=1 for scalar output
    
    print(f"Input: {x}")
    print(f"Output: {y}")
    print(f"VJP: {grad}")
    print(f"Expected (2x): {2 * x}")
    
    # Test with vector output
    def f_vec(x):
        return x ** 2
    
    v = torch.tensor([1.0, 1.0, 1.0])
    y_vec, vjp_func_vec = vjp(f_vec, x)
    grad_vec = vjp_func_vec(v)[0]
    
    print(f"Vector output: {y_vec}")
    print(f"VJP with v={v}: {grad_vec}")
    print(f"Expected (2x): {2 * x}")

def test_fixed_point_vjp():
    """Test VJP with a fixed point iteration."""
    print("\n🔧 Testing fixed point VJP...")
    
    def F(x, theta):
        """Simple fixed point function: x = 0.9 * x + theta"""
        return 0.9 * x + theta
    
    def fixed_point_solve(theta, max_iter=50):
        """Solve fixed point equation x = F(x, theta)"""
        x = torch.zeros_like(theta)
        for _ in range(max_iter):
            x_new = F(x, theta)
            if torch.abs(x_new - x).max() < 1e-8:
                break
            x = x_new
        return x
    
    theta = torch.tensor([1.0])
    x_star = fixed_point_solve(theta)
    print(f"Theta: {theta}")
    print(f"Fixed point: {x_star}")
    print(f"Analytical fixed point: {theta / 0.1}")  # x* = theta / (1 - 0.9)
    
    # Test VJP of F with respect to x
    v = torch.tensor([1.0])
    
    def F_x(x):
        return F(x, theta)
    
    def F_theta(th):
        return F(x_star, th)
    
    # VJP with respect to x
    _, vjp_x_func = vjp(F_x, x_star)
    grad_x = vjp_x_func(v)[0]
    print(f"VJP F w.r.t. x: {grad_x}")
    print(f"Expected (0.9): {torch.tensor([0.9])}")
    
    # VJP with respect to theta
    _, vjp_theta_func = vjp(F_theta, theta)
    grad_theta = vjp_theta_func(v)[0]
    print(f"VJP F w.r.t. theta: {grad_theta}")
    print(f"Expected (1.0): {torch.tensor([1.0])}")

def test_ccp_pi_update_vjp():
    """Test VJP with the actual Pi_update_ccp_log function."""
    print("\n🔧 Testing CCP Pi update VJP...")
    
    from matmul_ale_ccp_optimize import CCPOptimizer, compute_event_probabilities
    from matmul_ale_ccp_log import Pi_update_ccp_log, E_step
    
    # Create small test case
    optimizer = CCPOptimizer(
        "test_trees_1/sp.nwk", "test_trees_1/g.nwk",
        initial_params=(0.1, 0.1, 0.1),
        device=torch.device("cpu"),
        dtype=torch.float64
    )
    
    # Get current parameters
    params = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float64)
    p_S, p_D, p_T, p_L = compute_event_probabilities(params)
    
    # Compute E
    device = torch.device("cpu")
    dtype = torch.float64
    S = optimizer.species_helpers["S"]
    E = torch.zeros(S, dtype=dtype, device=device)
    
    for _ in range(10):  # Quick E convergence
        E_next, E_s1, E_s2, Ebar = E_step(E, optimizer.species_helpers["s_C1"], 
                                          optimizer.species_helpers["s_C2"], 
                                          optimizer.species_helpers["Recipients_mat"], 
                                          float(p_S), float(p_D), float(p_T), float(p_L))
        E = E_next
    
    print(f"E computed: shape {E.shape}, range [{E.min():.4f}, {E.max():.4f}]")
    
    # Test VJP of Pi_update_ccp_log
    log_Pi = optimizer.log_Pi_init.clone()
    print(f"Initial log_Pi: shape {log_Pi.shape}, finite values: {torch.isfinite(log_Pi).sum()}")
    
    def pi_update_wrapper(log_Pi_input):
        return Pi_update_ccp_log(log_Pi_input, optimizer.ccp_helpers, optimizer.species_helpers, 
                                optimizer.clade_species_map, E, Ebar, float(p_S), float(p_D), float(p_T))
    
    # Compute one Pi update
    log_Pi_new = pi_update_wrapper(log_Pi)
    print(f"Updated log_Pi: shape {log_Pi_new.shape}, finite values: {torch.isfinite(log_Pi_new).sum()}")
    
    # Test VJP
    v = torch.ones_like(log_Pi_new)  # Simple test vector
    
    try:
        _, vjp_func = vjp(pi_update_wrapper, log_Pi)
        grad = vjp_func(v)[0]
        print(f"VJP successful: gradient shape {grad.shape}, norm {grad.norm():.6f}")
        print(f"Gradient range: [{grad.min():.6f}, {grad.max():.6f}]")
        print(f"Non-zero gradient elements: {(torch.abs(grad) > 1e-10).sum()}")
        
    except Exception as e:
        print(f"❌ VJP failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🐛 VJP Debugging Session")
    print("=" * 50)
    
    test_simple_vjp()
    test_fixed_point_vjp()
    test_ccp_pi_update_vjp()