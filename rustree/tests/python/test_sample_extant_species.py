#!/usr/bin/env python3
"""Test PySpeciesTree.sample_extant()"""

import rustree

# Test 1: Basic functionality
print("Test 1: Basic functionality")
tree = rustree.simulate_species_tree(n=20, lambda_=1.0, mu=0.6, seed=42)
original_leaves = tree.num_leaves()
print(f"  Original: {original_leaves} leaves")

extant_only = tree.sample_extant()
extant_leaves = extant_only.num_leaves()
print(f"  Extant: {extant_leaves} leaves")

assert extant_leaves == 20, f"Expected 20 extant species, got {extant_leaves}"
assert extant_leaves <= original_leaves, "Extant count should be <= original"
print("  ✓ Test 1 passed")

# Test 2: No extinction (mu=0)
print("\nTest 2: No extinction case")
tree_no_extinct = rustree.simulate_species_tree(n=15, lambda_=1.0, mu=0.0, seed=123)
extant_no_extinct = tree_no_extinct.sample_extant()
assert tree_no_extinct.num_leaves() == 15
assert extant_no_extinct.num_leaves() == 15
print("  ✓ Test 2 passed")

# Test 3: Extant-only tree should work with other methods
print("\nTest 3: Extant tree works with other methods")
extant_only.save_newick("/tmp/extant.nwk")
newick_str = extant_only.to_newick()
assert len(newick_str) > 0
print("  ✓ Test 3 passed")

# Test 4: Can simulate gene trees on extant-only tree
print("\nTest 4: Gene trees on extant-only species tree")
gene_tree = extant_only.simulate_dtl(0.5, 0.2, 0.3, seed=999)
assert gene_tree.num_nodes() > 0
print("  ✓ Test 4 passed")

# Test 5: Verify BD events
print("\nTest 5: Verify BD events")
events = tree.get_bd_events()
n_extant = sum(1 for et in events['event_type'] if et == 'Leaf')
n_extinct = sum(1 for et in events['event_type'] if et == 'Extinction')
print(f"  Original tree: {n_extant} extant, {n_extinct} extinct")
assert n_extant == 20
assert extant_only.num_leaves() == n_extant
print("  ✓ Test 5 passed")

print("\n✓✓✓ All tests passed!")
