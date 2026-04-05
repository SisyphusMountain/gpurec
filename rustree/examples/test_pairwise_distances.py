"""
Example script demonstrating the pairwise_distances method in rustree.

This script shows how to:
1. Simulate or parse a species tree
2. Compute pairwise distances between nodes
3. Use both topological and metric distance types
4. Filter to leaves only or include all nodes
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "target", "release"))
import rustree


def example_simulated_tree():
    """Example using a simulated species tree."""
    print("=" * 60)
    print("Example 1: Simulated Species Tree")
    print("=" * 60)

    # Simulate a species tree with 5 extant species
    tree = rustree.simulate_species_tree(5, 1.0, 0.5, seed=42)

    print(f"Simulated tree with {tree.num_leaves()} leaves")
    print(f"Newick: {tree.to_newick()}")
    print()

    # Compute metric distances between leaves only
    print("Metric distances (branch lengths) between leaves:")
    df_metric = tree.pairwise_distances("metric", leaves_only=True)
    print(df_metric)
    print()

    # Compute topological distances between leaves only
    print("Topological distances (edge counts) between leaves:")
    df_topo = tree.pairwise_distances("topological", leaves_only=True)
    print(df_topo)
    print()


def example_parsed_tree():
    """Example using a parsed Newick tree."""
    print("=" * 60)
    print("Example 2: Parsed Newick Tree")
    print("=" * 60)

    # Parse a simple Newick tree
    newick = "((A:1.0,B:1.0):1.0,C:2.0):0.0;"
    tree = rustree.parse_species_tree(newick)

    print(f"Parsed tree: {newick}")
    print(f"Leaves: {tree.leaf_names()}")
    print()

    # Compute distances
    print("Metric distances between leaves:")
    df = tree.pairwise_distances("metric", leaves_only=True)
    print(df)
    print()

    # Verify expected distances:
    # - A to B: 1.0 + 1.0 = 2.0 (through their parent)
    # - A to C: 1.0 + 1.0 + 2.0 = 4.0 (through root)
    # - B to C: 1.0 + 1.0 + 2.0 = 4.0 (through root)
    print("Expected distances:")
    print("  A-B: 2.0, A-C: 4.0, B-C: 4.0")
    print()


def example_all_nodes():
    """Example computing distances for all nodes, not just leaves."""
    print("=" * 60)
    print("Example 3: Distances Between All Nodes")
    print("=" * 60)

    newick = "((A:1.0,B:1.0):1.0,C:2.0):0.0;"
    tree = rustree.parse_species_tree(newick)

    print(f"Tree: {newick}")
    print(f"Total nodes: {tree.num_nodes()}")
    print(f"Leaf nodes: {tree.num_leaves()}")
    print()

    # Compute distances for all nodes
    df = tree.pairwise_distances("metric", leaves_only=False)
    print(f"Pairwise distances for all {tree.num_nodes()} nodes:")
    print(f"Total pairs: {len(df)} ({tree.num_nodes()}^2)")
    print(df.head(15))
    print()


def example_distance_type_aliases():
    """Example showing distance type aliases."""
    print("=" * 60)
    print("Example 4: Distance Type Aliases")
    print("=" * 60)

    tree = rustree.simulate_species_tree(4, 1.0, 0.3, seed=123)

    print("Topological distance aliases ('topological', 'topo'):")
    df1 = tree.pairwise_distances("topological")
    df2 = tree.pairwise_distances("topo")
    print(f"Results identical: {df1.equals(df2)}")
    print()

    print("Metric distance aliases ('metric', 'patristic', 'branch'):")
    df3 = tree.pairwise_distances("metric")
    df4 = tree.pairwise_distances("patristic")
    df5 = tree.pairwise_distances("branch")
    print(f"Results identical: {df3.equals(df4) and df3.equals(df5)}")
    print()


def example_csv_export():
    """Example saving distances to CSV file."""
    print("=" * 60)
    print("Example 5: Exporting to CSV")
    print("=" * 60)

    tree = rustree.simulate_species_tree(5, 1.0, 0.5, seed=42)

    # Save to CSV using the save_pairwise_distances_csv method
    output_file = "/tmp/species_tree_distances.csv"
    tree.save_pairwise_distances_csv(output_file, "metric", leaves_only=True)

    print(f"Saved pairwise distances to: {output_file}")

    # Read and display first few lines
    with open(output_file, "r") as f:
        lines = f.readlines()[:6]
        print("First few lines of CSV:")
        for line in lines:
            print(f"  {line.rstrip()}")
    print()


if __name__ == "__main__":
    try:
        example_simulated_tree()
        example_parsed_tree()
        example_all_nodes()
        example_distance_type_aliases()
        example_csv_export()

        print("=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
