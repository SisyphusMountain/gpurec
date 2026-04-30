#!/usr/bin/env python3
"""Test script for save_pairwise_distances_csv function"""

import rustree
import os

# Simulate a species tree
print("Simulating species tree with 5 species...")
tree = rustree.simulate_species_tree(n=5, lambda_=1.0, mu=0.5, seed=42)

print(f"Tree has {tree.num_nodes()} nodes and {tree.num_leaves()} leaves")
print(f"Newick: {tree.to_newick()}")

# Test 1: Save topological distances (leaves only)
print("\nTest 1: Saving topological distances (leaves only)...")
tree.save_pairwise_distances_csv("test_topo_leaves.csv", "topological", leaves_only=True)
print("✓ Saved to test_topo_leaves.csv")

# Read and display the file
with open("test_topo_leaves.csv", "r") as f:
    content = f.read()
    print("First few lines:")
    print("\n".join(content.split("\n")[:6]))

# Test 2: Save metric distances (all nodes)
print("\nTest 2: Saving metric distances (all nodes)...")
tree.save_pairwise_distances_csv("test_metric_all.csv", "metric", leaves_only=False)
print("✓ Saved to test_metric_all.csv")

# Read and display the file
with open("test_metric_all.csv", "r") as f:
    content = f.read()
    lines = content.split("\n")
    print(f"Total lines: {len([l for l in lines if l])}")
    print("First few lines:")
    print("\n".join(lines[:6]))

# Test 3: Test with alternative distance type names
print("\nTest 3: Testing alternative distance type names...")
tree.save_pairwise_distances_csv("test_topo_alt.csv", "topo", leaves_only=True)
print("✓ 'topo' works")

tree.save_pairwise_distances_csv("test_metric_alt.csv", "patristic", leaves_only=True)
print("✓ 'patristic' works")

# Test 4: Test error handling
print("\nTest 4: Testing error handling...")
try:
    tree.save_pairwise_distances_csv("test_error.csv", "invalid_type", leaves_only=True)
    print("✗ Should have raised an error!")
except ValueError as e:
    print(f"✓ Correctly raised error: {e}")

# Cleanup
print("\nCleaning up test files...")
for filename in ["test_topo_leaves.csv", "test_metric_all.csv", "test_topo_alt.csv", "test_metric_alt.csv"]:
    if os.path.exists(filename):
        os.remove(filename)
        print(f"Removed {filename}")

print("\n✓ All tests passed!")
