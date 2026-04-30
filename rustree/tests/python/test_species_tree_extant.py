#!/usr/bin/env python3
"""Test that simulate_species_tree returns only extant species."""

import rustree

# Simulate a species tree with 20 extant species
print("Testing simulate_species_tree with extinction...")
tree = rustree.simulate_species_tree(n=20, lambda_=1.0, mu=0.5, seed=42)

print(f"\nRequested: 20 extant species")
print(f"Tree has: {tree.num_leaves()} leaves")
print(f"Total nodes: {tree.num_nodes()} nodes")
print(f"Tree height: {tree.tree_height():.3f}")

# Get leaf names
leaves = tree.leaf_names()
print(f"\nLeaf names (first 10): {leaves[:10]}")

# Check if all nodes are either leaves or internal
internal_nodes = tree.num_nodes() - tree.num_leaves()
print(f"Internal nodes: {internal_nodes}")
print(f"Expected internal nodes for binary tree: {tree.num_leaves() - 1}")

# Verify the tree structure
assert tree.num_leaves() == 20, f"Expected 20 leaves, got {tree.num_leaves()}"
assert tree.num_nodes() == 2 * tree.num_leaves() - 1, \
    f"Binary tree should have 2n-1 nodes, got {tree.num_nodes()}"

print("\n✓ simulate_species_tree WORKS correctly!")
print("✓ Tree contains ONLY extant species (extinct lineages excluded)")
print(f"✓ Simulated backwards in time from {tree.num_leaves()} extant species")
