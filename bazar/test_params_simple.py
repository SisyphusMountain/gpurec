#!/usr/bin/env python3
"""Simple parameter test."""

from matmul_ale_ccp_corrected import main_ccp
import torch

# Suppress most output
import sys
import io

test_params = [
    (1e-10, 1e-10, 1e-10, "Pure speciation"),
    (0.1, 1e-10, 1e-10, "Duplication"),
    (1e-10, 0.1, 1e-10, "Transfer"),
    (1e-10, 1e-10, 0.1, "Loss"),
    (0.1, 0.1, 0.1, "All equal"),
]

species_tree = "test_trees_1/sp.nwk"
gene_tree = "test_trees_1/g.nwk"

print("Parameter comparison:")
print("-" * 40)

for delta, tau, lam, name in test_params:
    # Capture output
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    
    try:
        result = main_ccp(species_tree, gene_tree, delta, tau, lam, iters=50)
        log_lik = result['log_likelihood']
    except Exception as e:
        log_lik = float('-inf')
    finally:
        sys.stdout = old_stdout
    
    print(f"{name:20} : {log_lik:10.6f}")