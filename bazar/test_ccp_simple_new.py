#!/usr/bin/env python3
"""Test CCP implementation with simple case."""

from matmul_ale_ccp import main_ccp
import torch

# Test with pure speciation (delta=tau=lambda=1e-10)
result = main_ccp("test_trees_1/sp.nwk", "test_trees_1/g.nwk", 
                  delta=1e-10, tau=1e-10, lambda_param=1e-10, iters=50)

print("\n=== DEBUG INFO ===")
print(f"Log-likelihood: {result['log_likelihood']}")

# Check Pi matrix
Pi = result['Pi']
ccp = result['ccp']
root_id = max(ccp.clade_to_id.values())  # Try getting the last clade ID
print(f"\nRoot clade ID (max): {root_id}")
print(f"Root clade: {ccp.id_to_clade[root_id]}")
print(f"Root clade Pi sum: {Pi[root_id].sum():.6e}")

# Check a few other clades
print("\nPi sums for all clades:")
for clade_id in sorted(ccp.id_to_clade.keys()):
    clade = ccp.id_to_clade[clade_id]
    pi_sum = Pi[clade_id].sum()
    if pi_sum > 0:
        print(f"  Clade {clade_id} (size {clade.size}): Pi_sum={pi_sum:.6e}")

# Check leaf mapping
print("\nLeaf clade mappings:")
for clade_id, clade in ccp.id_to_clade.items():
    if clade.is_leaf():
        print(f"  Clade {clade_id}: {clade.get_leaf_name()}")

# Check splits for root
root_clade = ccp.id_to_clade[0]  # Based on output, root is ID 0
print(f"\nRoot clade splits: {len(ccp.splits[root_clade])}")
for i, split in enumerate(ccp.splits[root_clade][:3]):
    print(f"  Split {i}: {split.left.size} + {split.right.size}, prob={split.probability:.6f}")