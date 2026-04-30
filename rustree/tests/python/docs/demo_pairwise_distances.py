#!/usr/bin/env python3
"""
Demonstration of pairwise distance functionality in rustree.

This script shows example usage of the pairwise_distances() and
save_pairwise_distances_csv() methods once they are added to the Python bindings.

Note: This script will only run after the Python bindings are updated.
See README_PAIRWISE_DISTANCES.md for implementation details.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "target", "release"))

try:
    import rustree
    import pandas as pd
except ImportError as e:
    print(f"Error importing required modules: {e}")
    sys.exit(1)


def demo_basic_usage():
    """Demonstrate basic pairwise distance computation."""
    print("=" * 70)
    print("DEMO 1: Basic Pairwise Distances")
    print("=" * 70)

    # Create a simple tree
    newick = "((A:1.0,B:1.0):1.0,C:2.0):0.0;"
    tree = rustree.parse_species_tree(newick)

    print(f"\nTree: {newick}")
    print(f"Leaves: {tree.num_leaves()}")
    print(f"Total nodes: {tree.num_nodes()}")

    # Compute metric distances (leaf nodes only)
    print("\n--- Metric Distances (leaves only) ---")
    try:
        distances = tree.pairwise_distances("metric", leaves_only=True)
        print(distances)
    except AttributeError:
        print("ERROR: pairwise_distances() method not yet available.")
        print("See README_PAIRWISE_DISTANCES.md for implementation details.")


def demo_distance_types():
    """Compare topological and metric distance types."""
    print("\n" + "=" * 70)
    print("DEMO 2: Topological vs Metric Distances")
    print("=" * 70)

    # Create a tree with varying branch lengths
    newick = "((A:0.5,B:1.5):0.5,C:2.5):0.0;"
    tree = rustree.parse_species_tree(newick)

    print(f"\nTree: {newick}")
    print("Note: Branch lengths vary (0.5, 1.5, 0.5, 2.5)")

    try:
        # Topological distances (count edges)
        print("\n--- Topological Distances (edge count) ---")
        topo = tree.pairwise_distances("topological", leaves_only=True)
        print(topo)

        # Metric distances (sum branch lengths)
        print("\n--- Metric Distances (sum of branch lengths) ---")
        metric = tree.pairwise_distances("metric", leaves_only=True)
        print(metric)

        # Compare
        print("\n--- Comparison ---")
        print("For pair (A, B):")
        ab_topo = topo[(topo["node1"] == "A") & (topo["node2"] == "B")]["distance"].values[0]
        ab_metric = metric[(metric["node1"] == "A") & (metric["node2"] == "B")]["distance"].values[0]
        print(f"  Topological: {ab_topo} edges")
        print(f"  Metric: {ab_metric} (= 0.5 + 0.5 + 1.5)")

    except AttributeError:
        print("ERROR: pairwise_distances() method not yet available.")


def demo_leaves_only_parameter():
    """Compare leaves_only=True vs leaves_only=False."""
    print("\n" + "=" * 70)
    print("DEMO 3: Leaves Only vs All Nodes")
    print("=" * 70)

    # Create a tree
    tree = rustree.simulate_species_tree(5, 1.0, 0.3, seed=42)

    print(f"\nSimulated tree with {tree.num_leaves()} leaves")
    print(f"Total nodes (including internal): {tree.num_nodes()}")

    try:
        # Only leaves
        print("\n--- Distances between leaf nodes only ---")
        leaf_distances = tree.pairwise_distances("metric", leaves_only=True)
        print(f"Number of distance pairs: {len(leaf_distances)}")
        print(f"  = {tree.num_leaves()} leaves × {tree.num_leaves()} leaves")
        print(f"  = {tree.num_leaves() ** 2}")

        # All nodes
        print("\n--- Distances between all nodes (including internal) ---")
        all_distances = tree.pairwise_distances("metric", leaves_only=False)
        print(f"Number of distance pairs: {len(all_distances)}")
        print(f"  = {tree.num_nodes()} nodes × {tree.num_nodes()} nodes")
        print(f"  = {tree.num_nodes() ** 2}")

        print(f"\nDifference: {len(all_distances) - len(leaf_distances)} additional pairs")

    except AttributeError:
        print("ERROR: pairwise_distances() method not yet available.")


def demo_csv_export():
    """Demonstrate CSV export functionality."""
    print("\n" + "=" * 70)
    print("DEMO 4: CSV Export")
    print("=" * 70)

    tree = rustree.simulate_species_tree(8, 1.0, 0.3, seed=123)

    print(f"\nSimulated tree with {tree.num_leaves()} leaves")

    try:
        import tempfile
        import os

        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            filepath = f.name

        print(f"\nSaving pairwise distances to: {filepath}")
        tree.save_pairwise_distances_csv(filepath, "metric", leaves_only=True)

        # Read and display first few rows
        print("\n--- First 10 rows of saved CSV ---")
        saved_df = pd.read_csv(filepath)
        print(saved_df.head(10))

        # Statistics
        print("\n--- Statistics ---")
        print(f"Total entries: {len(saved_df)}")
        print(f"Mean distance: {saved_df['distance'].mean():.4f}")
        print(f"Max distance: {saved_df['distance'].max():.4f}")
        print(f"Min distance: {saved_df['distance'].min():.4f}")

        # Cleanup
        os.remove(filepath)
        print(f"\nCleaned up temporary file: {filepath}")

    except AttributeError:
        print("ERROR: save_pairwise_distances_csv() method not yet available.")


def demo_practical_example():
    """Practical example: analyzing distance distribution."""
    print("\n" + "=" * 70)
    print("DEMO 5: Practical Analysis - Distance Distribution")
    print("=" * 70)

    tree = rustree.simulate_species_tree(15, 1.0, 0.3, seed=456)

    print(f"\nAnalyzing pairwise distances in a tree with {tree.num_leaves()} species")

    try:
        distances = tree.pairwise_distances("metric", leaves_only=True)

        # Filter out self-distances
        non_self = distances[distances["node1"] != distances["node2"]]

        print("\n--- Distance Statistics (excluding self-distances) ---")
        print(f"Number of species pairs: {len(non_self) // 2}")  # Divide by 2 for symmetry
        print(f"Mean pairwise distance: {non_self['distance'].mean():.4f}")
        print(f"Std dev: {non_self['distance'].std():.4f}")
        print(f"Min distance: {non_self['distance'].min():.4f}")
        print(f"Max distance: {non_self['distance'].max():.4f}")

        # Find closest and farthest pairs
        closest = non_self.loc[non_self['distance'].idxmin()]
        farthest = non_self.loc[non_self['distance'].idxmax()]

        print(f"\nClosest pair: {closest['node1']} - {closest['node2']}")
        print(f"  Distance: {closest['distance']:.4f}")

        print(f"\nFarthest pair: {farthest['node1']} - {farthest['node2']}")
        print(f"  Distance: {farthest['distance']:.4f}")

        # Histogram (text-based)
        print("\n--- Distance Distribution (histogram) ---")
        hist, bins = pd.cut(non_self['distance'], bins=5, retbins=True)
        value_counts = hist.value_counts().sort_index()

        for interval, count in value_counts.items():
            bar = '█' * int(count / len(non_self) * 50)
            print(f"{interval}: {bar} ({count})")

    except AttributeError:
        print("ERROR: pairwise_distances() method not yet available.")


def main():
    """Run all demonstrations."""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "  Rustree Pairwise Distance Functionality - Demonstration".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "═" * 68 + "╝")

    demos = [
        demo_basic_usage,
        demo_distance_types,
        demo_leaves_only_parameter,
        demo_csv_export,
        demo_practical_example,
    ]

    for demo in demos:
        try:
            demo()
        except Exception as e:
            print(f"\nError in {demo.__name__}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    print("\nNote: If you see AttributeError messages above, it means the")
    print("pairwise distance methods are not yet available in the Python bindings.")
    print("See README_PAIRWISE_DISTANCES.md for implementation instructions.")
    print()


if __name__ == "__main__":
    main()
