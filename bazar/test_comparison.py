#!/usr/bin/env python3
"""
Quick test to compare matmul_ale.py and matmul_ale_ccp.py results
"""

import torch
import sys
sys.path.append('/home/enzo/Documents/git/WP2/gpurec')

from matmul_ale import main_fn as ale_main
from matmul_ale_ccp import main_ccp

# Test with very low rates (essentially pure speciation)
print("Testing with near-zero DTL rates (pure speciation)...")

# Original implementation
print("\n=== Original matmul_ale.py ===")
result_original = ale_main(
    sp_tree_path="test_trees_1/sp.nwk",
    g_tree_path="test_trees_1/g.nwk", 
    d_rate=-23,  # Very low duplication
    t_rate=-23,  # Very low transfer
    l_rate=-23,  # Very low loss
    iters=50,
    n_reconciliations=0  # Skip sampling to see likelihood only
)

# CCP implementation
print("\n=== CCP matmul_ale_ccp.py ===")
result_ccp = main_ccp(
    species_tree_path="test_trees_1/sp.nwk",
    gene_tree_path="test_trees_1/g.nwk",
    delta=1e-10,
    tau=1e-10, 
    lambda_param=1e-10,
    iters=50
)

print(f"\nOriginal log-likelihood: {result_original}")
print(f"CCP log-likelihood: {result_ccp['log_likelihood']}")