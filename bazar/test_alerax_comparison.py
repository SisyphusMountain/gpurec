#!/usr/bin/env python3
"""Test our CCP implementation against AleRax values."""

import subprocess
import os
from matmul_ale_ccp import main_ccp

def run_alerax(species_tree, gene_tree, delta, tau, lam):
    """Run AleRax and extract log-likelihood."""
    # First observe the gene tree
    cmd = f"ALEobserve {gene_tree}"
    subprocess.run(cmd, shell=True, check=True)
    
    # Observe species tree  
    cmd = f"ALEobserve {species_tree}"
    subprocess.run(cmd, shell=True, check=True)
    
    # Run AleRax with specified parameters
    cmd = f"ALEml_undated {species_tree} {gene_tree}.ale tau={tau} delta={delta} lambda={lam}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    # Extract log-likelihood from output
    for line in result.stdout.split('\n'):
        if "LL=" in line:
            # Extract the log-likelihood value
            ll_str = line.split("LL=")[1].split()[0]
            return float(ll_str)
    
    return None

# Test cases
test_cases = [
    {"delta": 1e-10, "tau": 1e-10, "lambda": 1e-10, "name": "Pure speciation"},
    {"delta": 0.1, "tau": 1e-10, "lambda": 1e-10, "name": "Duplication"},
    {"delta": 1e-10, "tau": 0.1, "lambda": 1e-10, "name": "Transfer"},
    {"delta": 1e-10, "tau": 1e-10, "lambda": 0.1, "name": "Loss"},
]

species_tree = "test_trees_1/sp.nwk"
gene_tree = "test_trees_1/g.nwk"

print("Comparing our implementation with AleRax:")
print("=" * 60)

for test in test_cases:
    print(f"\n{test['name']}:")
    print(f"  Parameters: delta={test['delta']}, tau={test['tau']}, lambda={test['lambda']}")
    
    # Run our implementation
    our_result = main_ccp(species_tree, gene_tree, 
                         delta=test['delta'], 
                         tau=test['tau'], 
                         lambda_param=test['lambda'],
                         iters=50)
    our_ll = our_result['log_likelihood']
    
    print(f"  Our log-likelihood: {our_ll:.6f}")
    
    # Try to run AleRax if available
    try:
        alerax_ll = run_alerax(species_tree, gene_tree, 
                              test['delta'], test['tau'], test['lambda'])
        if alerax_ll is not None:
            print(f"  AleRax log-likelihood: {alerax_ll:.6f}")
            print(f"  Difference: {abs(our_ll - alerax_ll):.6f}")
    except:
        print("  AleRax not available for comparison")