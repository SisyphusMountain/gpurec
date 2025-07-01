#!/usr/bin/env python3
"""
Test relative likelihoods with different parameters.
Even if absolute values don't match AleRax exactly, 
relative values should show the same trends.
"""

from matmul_ale_ccp_corrected import main_ccp
import torch

# Test cases with different parameters
test_params = [
    {"delta": 1e-10, "tau": 1e-10, "lambda": 1e-10, "name": "Pure speciation"},
    {"delta": 0.1, "tau": 1e-10, "lambda": 1e-10, "name": "Duplication only"},
    {"delta": 1e-10, "tau": 0.1, "lambda": 1e-10, "name": "Transfer only"},
    {"delta": 1e-10, "tau": 1e-10, "lambda": 0.1, "name": "Loss only"},
    {"delta": 0.1, "tau": 0.1, "lambda": 0.1, "name": "All equal"},
]

species_tree = "test_trees_1/sp.nwk"
gene_tree = "test_trees_1/g.nwk"

print("Testing relative likelihoods with different parameters")
print("="*60)

results = []

for params in test_params:
    print(f"\n{params['name']}:")
    print(f"  delta={params['delta']}, tau={params['tau']}, lambda={params['lambda']}")
    
    result = main_ccp(
        species_tree, gene_tree,
        delta=params['delta'], 
        tau=params['tau'], 
        lambda_param=params['lambda'],
        iters=100
    )
    
    log_lik = result['log_likelihood']
    results.append((params['name'], log_lik))
    print(f"  Log-likelihood: {log_lik:.6f}")

# Compare results
print("\n" + "="*60)
print("Summary:")
print("="*60)

# Sort by likelihood
results.sort(key=lambda x: x[1], reverse=True)

print("\nRanked by likelihood (best to worst):")
for i, (name, ll) in enumerate(results):
    print(f"{i+1}. {name}: {ll:.6f}")

# Compute likelihood ratios relative to pure speciation
pure_spec_ll = next(ll for name, ll in results if "Pure speciation" in name)
print(f"\nLikelihood ratios relative to pure speciation:")
for name, ll in results:
    if "Pure speciation" not in name:
        ratio = torch.exp(torch.tensor(ll - pure_spec_ll)).item()
        print(f"  {name}: {ratio:.6e}x")