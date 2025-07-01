#!/usr/bin/env python3
"""
Step-by-step debugging of the gradient computation.
"""

import torch
from torch.autograd.functional import vjp

def test_fixed_point_step_by_step():
    """Debug fixed point gradient computation step by step."""
    print("🔧 Step-by-step fixed point gradient debugging...")
    
    from matmul_ale_ccp_optimize import CCPOptimizer, FixedPointCCPFunction, ccp_fixed_point_iteration, compute_log_likelihood
    
    # Create optimizer with double precision for accuracy
    optimizer = CCPOptimizer(
        "test_trees_1/sp.nwk", "test_trees_1/g.nwk",
        initial_params=(0.1, 0.1, 0.1),
        device=torch.device("cpu"),
        dtype=torch.float64
    )
    
    log_params = optimizer.log_params.clone().detach().requires_grad_(True)
    print(f"Log params: {log_params}")
    
    # Step 1: Forward pass manually
    print("\n📍 Step 1: Forward pass")
    from matmul_ale_ccp_optimize import softplus_transform
    params = softplus_transform(log_params)
    print(f"Transformed params: {params}")
    
    # Step 2: Fixed point iteration
    print("\n📍 Step 2: Fixed point iteration")
    log_Pi_star = ccp_fixed_point_iteration(
        optimizer.log_Pi_init, optimizer.ccp_helpers, optimizer.species_helpers, 
        optimizer.clade_species_map, params
    )
    print(f"Fixed point converged: {torch.isfinite(log_Pi_star).all()}")
    print(f"Fixed point range: [{log_Pi_star.min():.4f}, {log_Pi_star.max():.4f}]")
    
    # Step 3: Compute log-likelihood
    print("\n📍 Step 3: Log-likelihood")
    log_likelihood = compute_log_likelihood(log_Pi_star, optimizer.root_clade_id)
    print(f"Log-likelihood: {log_likelihood}")
    
    # Step 4: Test gradient of log-likelihood w.r.t. log_Pi
    print("\n📍 Step 4: Gradient w.r.t. log_Pi")
    log_Pi_test = log_Pi_star.clone().requires_grad_(True)
    ll_test = compute_log_likelihood(log_Pi_test, optimizer.root_clade_id)
    ll_test.backward()
    grad_ll_wrt_logPi = log_Pi_test.grad
    print(f"Gradient shape: {grad_ll_wrt_logPi.shape}")
    print(f"Gradient norm: {grad_ll_wrt_logPi.norm():.6f}")
    print(f"Non-zero elements: {(torch.abs(grad_ll_wrt_logPi) > 1e-10).sum()}")
    
    # Step 5: Test if fixed point iteration is differentiable w.r.t. params
    print("\n📍 Step 5: Fixed point w.r.t. params")
    
    def fixed_point_wrapper(params_input):
        return ccp_fixed_point_iteration(
            optimizer.log_Pi_init, optimizer.ccp_helpers, optimizer.species_helpers, 
            optimizer.clade_species_map, params_input
        )
    
    # Test VJP of fixed point iteration
    v_test = torch.ones_like(log_Pi_star)
    
    try:
        print("   Testing VJP of fixed point iteration...")
        _, grad_fp = vjp(fixed_point_wrapper, params, v_test)
        print(f"   VJP successful: gradient shape {grad_fp.shape}, norm {grad_fp.norm():.6f}")
        
    except Exception as e:
        print(f"   ❌ VJP failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Step 6: Test the full FixedPointCCPFunction
    print("\n📍 Step 6: Full FixedPointCCPFunction")
    
    try:
        log_likelihood_auto = FixedPointCCPFunction.apply(
            log_params, optimizer.log_Pi_init,
            optimizer.ccp_helpers, optimizer.species_helpers, optimizer.clade_species_map,
            optimizer.root_clade_id
        )
        print(f"   Forward pass successful: {log_likelihood_auto}")
        
        log_likelihood_auto.backward()
        print(f"   Backward pass successful: gradient = {log_params.grad}")
        
    except Exception as e:
        print(f"   ❌ FixedPointCCPFunction failed: {e}")
        import traceback
        traceback.print_exc()

def test_simple_pi_update_gradient():
    """Test gradient through a single Pi update."""
    print("\n🔧 Testing single Pi update gradient...")
    
    from matmul_ale_ccp_optimize import CCPOptimizer, compute_event_probabilities
    from matmul_ale_ccp_log import Pi_update_ccp_log, E_step
    
    optimizer = CCPOptimizer(
        "test_trees_1/sp.nwk", "test_trees_1/g.nwk",
        initial_params=(0.1, 0.1, 0.1),
        device=torch.device("cpu"),
        dtype=torch.float64
    )
    
    # Set up parameters
    params = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float64, requires_grad=True)
    p_S, p_D, p_T, p_L = compute_event_probabilities(params)
    
    # Compute E
    device = torch.device("cpu")
    dtype = torch.float64
    S = optimizer.species_helpers["S"]
    E = torch.zeros(S, dtype=dtype, device=device)
    
    for _ in range(10):
        E_next, E_s1, E_s2, Ebar = E_step(E, optimizer.species_helpers["s_C1"], 
                                          optimizer.species_helpers["s_C2"], 
                                          optimizer.species_helpers["Recipients_mat"], 
                                          float(p_S), float(p_D), float(p_T), float(p_L))
        E = E_next
    
    # Test Pi update with gradient
    log_Pi = optimizer.log_Pi_init.clone()
    
    def pi_update_with_params(params_input):
        p_S_i, p_D_i, p_T_i, p_L_i = compute_event_probabilities(params_input)
        return Pi_update_ccp_log(log_Pi, optimizer.ccp_helpers, optimizer.species_helpers, 
                               optimizer.clade_species_map, E, Ebar, 
                               float(p_S_i), float(p_D_i), float(p_T_i))
    
    # Compute Pi update
    log_Pi_new = pi_update_with_params(params)
    print(f"Pi update successful: shape {log_Pi_new.shape}")
    
    # Test gradient
    objective = log_Pi_new.sum()
    objective.backward()
    print(f"Gradient through Pi update: {params.grad}")
    print(f"Gradient norm: {params.grad.norm():.6f}")

if __name__ == "__main__":
    print("🐛 Step-by-Step Gradient Debugging")
    print("=" * 50)
    
    test_fixed_point_step_by_step()
    test_simple_pi_update_gradient()