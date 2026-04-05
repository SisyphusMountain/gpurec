#!/usr/bin/env python3
"""Test script for induced transfers functionality."""

import rustree

# Simulate a complete species tree with 50 species
print("Simulating complete species tree with 50 species...")
species_tree = rustree.simulate_species_tree(n=50, lambda_=1.0, mu=0.5, seed=42)
print(f"Species tree: {species_tree.num_leaves()} leaves, {species_tree.num_nodes()} nodes")

# Simulate a gene tree on the complete species tree
print("\nSimulating gene tree with DTL events...")
gene_tree = species_tree.simulate_dtl(
    lambda_d=0.5,
    lambda_t=0.2,
    lambda_l=0.3,
    seed=123
)
print(f"Gene tree: {gene_tree.num_extant()} extant genes")

# Count events
spec, dup, trans, loss, leaves = gene_tree.count_events()
print(f"Events: {spec} speciations, {dup} duplications, {trans} transfers, {loss} losses, {leaves} leaves")

# Sample a subset of species (first 20 leaves)
all_leaf_names = species_tree.leaf_names()
sampled_leaf_names = all_leaf_names[:20]
print(f"\nSampling {len(sampled_leaf_names)} species from complete tree...")
sampled_species_tree = species_tree.extract_induced_subtree_by_names(sampled_leaf_names)
print(f"Sampled species tree: {sampled_species_tree.num_leaves()} leaves, {sampled_species_tree.num_nodes()} nodes")

# Compute induced transfers
print("\nComputing induced transfers...")
try:
    induced = gene_tree.compute_induced_transfers(sampled_species_tree, sampled_leaf_names)
    print(f"Found {len(induced)} induced transfers")

    if len(induced) > 0:
        print("\nFirst 5 induced transfers:")
        for i, transfer in enumerate(induced[:5]):
            print(f"{i+1}. {transfer}")

        # Count projections
        successful = sum(1 for t in induced if t.from_species_sampled is not None and t.to_species_sampled is not None)
        print(f"\n{successful}/{len(induced)} transfers successfully projected to sampled tree")
except Exception as e:
    print(f"Error computing induced transfers: {e}")
    import traceback
    traceback.print_exc()

# Compute ghost lengths
print("\nComputing ghost branch lengths...")
try:
    ghost_lengths = gene_tree.compute_ghost_lengths(sampled_species_tree, sampled_leaf_names)
    print(f"Ghost lengths computed for {len(ghost_lengths)} nodes")

    # Show statistics
    total_ghost = sum(ghost_lengths)
    max_ghost = max(ghost_lengths)
    avg_ghost = total_ghost / len(ghost_lengths) if ghost_lengths else 0

    print(f"Total ghost length: {total_ghost:.2f}")
    print(f"Maximum ghost length: {max_ghost:.2f}")
    print(f"Average ghost length: {avg_ghost:.2f}")

    # Show nodes with significant ghost length
    significant = [(i, gl) for i, gl in enumerate(ghost_lengths) if gl > 1.0]
    if significant:
        print(f"\nNodes with ghost length > 1.0:")
        for node_idx, gl in significant[:10]:
            print(f"  Node {node_idx}: {gl:.2f}")
except Exception as e:
    print(f"Error computing ghost lengths: {e}")
    import traceback
    traceback.print_exc()

print("\n✓ Test completed successfully!")
