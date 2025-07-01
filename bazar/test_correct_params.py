#!/usr/bin/env python3
"""
Test both implementations with correctly converted AleRax parameters
"""

import torch
import sys
sys.path.append('/home/enzo/Documents/git/WP2/gpurec')

from matmul_ale import main_fn as ale_main
from matmul_ale_ccp import main_ccp

# AleRax found rates: D=1e-10, L=1e-10, T=1e-10
alerax_delta = 1e-10
alerax_tau = 1e-10
alerax_lambda = 1e-10

print("Testing with AleRax equivalent parameters...")
print(f"AleRax rates: δ={alerax_delta}, τ={alerax_tau}, λ={alerax_lambda}")

# For matmul_ale.py: need log(rate) since rates = softplus(input)
import math
d_rate = math.log(alerax_delta) 
t_rate = math.log(alerax_tau)
l_rate = math.log(alerax_lambda)

print(f"matmul_ale.py inputs: d_rate={d_rate:.2f}, t_rate={t_rate:.2f}, l_rate={l_rate:.2f}")

# Original implementation  
print("\n=== Original matmul_ale.py ===")
try:
    # Modified to return likelihood
    import matmul_ale
    
    # Calculate probabilities manually to show them
    rates_raw = torch.tensor([d_rate, t_rate, l_rate], dtype=torch.float64)
    rates_pos = torch.nn.functional.softplus(rates_raw)
    denom = 1.0 + rates_pos.sum()
    p_D = rates_pos[0] / denom
    p_T = rates_pos[1] / denom
    p_L = rates_pos[2] / denom
    p_S = 1.0 / denom
    
    print(f"Computed probabilities: p_S={p_S:.6f}, p_D={p_D:.2e}, p_T={p_T:.2e}, p_L={p_L:.2e}")
    
    result_original = ale_main(
        sp_tree_path="test_trees_1/sp.nwk",
        g_tree_path="test_trees_1/g.nwk", 
        d_rate=d_rate,
        t_rate=t_rate,
        l_rate=l_rate,
        iters=50,
        n_reconciliations=0
    )
    
except Exception as e:
    print(f"Error: {e}")

# CCP implementation - use rates directly
print("\n=== CCP matmul_ale_ccp.py ===")
result_ccp = main_ccp(
    species_tree_path="test_trees_1/sp.nwk",
    gene_tree_path="test_trees_1/g.nwk",
    delta=alerax_delta,
    tau=alerax_tau, 
    lambda_param=alerax_lambda,
    iters=50
)

print(f"\nCCP log-likelihood: {result_ccp['log_likelihood']}")