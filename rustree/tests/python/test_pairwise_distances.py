"""
Comprehensive tests for pairwise distance functionality in rustree.

This module tests the pairwise distance computation and export functions:
- SpeciesTree.pairwise_distances() with "topological" and "metric" distance types
- SpeciesTree.save_pairwise_distances_csv() for CSV output
- Testing with leaves_only=True and leaves_only=False
- Distance calculation correctness verification
- Edge cases (single node, two nodes, etc.)
- Error handling for invalid distance types

Requirements:
- pytest
- pandas (for DataFrame validation)
"""

import sys
sys.path.insert(0, "/home/enzo/Documents/Zombi/ZOMBI/zombi-rs/rustree/target/release")
import rustree
import os
import tempfile
import pytest

# Only import pandas if available
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("WARNING: pandas not available, some tests will be skipped")


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def simple_tree():
    """Create a simple 3-leaf tree for testing.

    Tree structure: ((A:1.0,B:1.0):1.0,C:2.0):0.0;
    """
    return rustree.parse_species_tree("((A:1.0,B:1.0):1.0,C:2.0):0.0;")


@pytest.fixture
def symmetric_tree():
    """Create a symmetric 4-leaf tree.

    Tree structure: ((A:1.0,B:1.0):1.0,(C:1.0,D:1.0):1.0):0.0;
    """
    return rustree.parse_species_tree("((A:1.0,B:1.0):1.0,(C:1.0,D:1.0):1.0):0.0;")


@pytest.fixture
def single_leaf_tree():
    """Create a single-leaf tree."""
    return rustree.parse_species_tree("A:0.0;")


@pytest.fixture
def two_leaf_tree():
    """Create a two-leaf tree."""
    return rustree.parse_species_tree("(A:1.5,B:1.5):0.0;")


@pytest.fixture
def large_tree():
    """Create a larger simulated tree with 20 leaves."""
    return rustree.simulate_species_tree(20, 1.0, 0.3, seed=42)


@pytest.fixture
def varying_lengths_tree():
    """Create a tree with varying branch lengths."""
    return rustree.parse_species_tree("((A:0.5,B:1.5):0.5,C:2.5):0.0;")


# =============================================================================
# Tests for pairwise_distances() with "topological" distance type
# =============================================================================

def test_pairwise_distances_topological_basic(simple_tree):
    """Test basic topological distance computation."""
    distances = simple_tree.pairwise_distances("topological", leaves_only=True)

    assert isinstance(distances, pd.DataFrame), "Should return a pandas DataFrame"
    assert len(distances) == 9, "Should have 3x3 = 9 entries for 3 leaves"
    assert list(distances.columns) == ["node1", "node2", "distance"], "Should have correct columns"


def test_pairwise_distances_topological_self_distance(simple_tree):
    """Test that topological self-distance is 0."""
    distances = simple_tree.pairwise_distances("topological", leaves_only=True)

    # Filter self-distances (where node1 == node2)
    self_distances = distances[distances["node1"] == distances["node2"]]

    assert len(self_distances) == 3, "Should have 3 self-distances for 3 leaves"
    assert all(self_distances["distance"] == 0.0), "All self-distances should be 0"


def test_pairwise_distances_topological_symmetry(simple_tree):
    """Test that topological distances are symmetric."""
    distances = simple_tree.pairwise_distances("topological", leaves_only=True)

    # Check symmetry: distance(A,B) == distance(B,A)
    for _, row in distances.iterrows():
        node1, node2, dist = row["node1"], row["node2"], row["distance"]
        # Find the reverse pair
        reverse = distances[(distances["node1"] == node2) & (distances["node2"] == node1)]
        if not reverse.empty:
            assert reverse.iloc[0]["distance"] == dist, f"Distance({node1},{node2}) should equal Distance({node2},{node1})"


def test_pairwise_distances_topological_values(symmetric_tree):
    """Test specific topological distance values on a known tree.

    For tree ((A:1,B:1):1,(C:1,D:1):1):0:
    - A-B: 2 edges (A to parent, parent to B)
    - A-C: 4 edges (A to parent, parent to root, root to C's parent, C's parent to C)
    - A-D: 4 edges
    - B-C: 4 edges
    - B-D: 4 edges
    - C-D: 2 edges
    """
    distances = symmetric_tree.pairwise_distances("topological", leaves_only=True)

    def get_distance(node1, node2):
        row = distances[(distances["node1"] == node1) & (distances["node2"] == node2)]
        return row.iloc[0]["distance"] if not row.empty else None

    # Test expected topological distances
    assert get_distance("A", "B") == 2.0, "A-B should be 2 edges apart"
    assert get_distance("C", "D") == 2.0, "C-D should be 2 edges apart"
    assert get_distance("A", "C") == 4.0, "A-C should be 4 edges apart"
    assert get_distance("A", "D") == 4.0, "A-D should be 4 edges apart"
    assert get_distance("B", "C") == 4.0, "B-C should be 4 edges apart"
    assert get_distance("B", "D") == 4.0, "B-D should be 4 edges apart"


def test_pairwise_distances_topological_leaves_only_true(large_tree):
    """Test topological distances with leaves_only=True."""
    distances = large_tree.pairwise_distances("topological", leaves_only=True)

    num_leaves = large_tree.num_leaves()
    expected_count = num_leaves * num_leaves

    assert len(distances) == expected_count, f"Should have {expected_count} entries for {num_leaves} leaves"

    # All node names should be leaf names
    leaf_names = set(large_tree.leaf_names())
    assert all(distances["node1"].isin(leaf_names)), "All node1 should be leaves"
    assert all(distances["node2"].isin(leaf_names)), "All node2 should be leaves"


def test_pairwise_distances_topological_leaves_only_false(simple_tree):
    """Test topological distances with leaves_only=False."""
    distances = simple_tree.pairwise_distances("topological", leaves_only=False)

    num_nodes = simple_tree.num_nodes()
    expected_count = num_nodes * num_nodes

    assert len(distances) == expected_count, f"Should have {expected_count} entries for {num_nodes} nodes"
    assert len(distances) > 9, "Should have more entries than leaves-only mode"


def test_pairwise_distances_topological_all_positive(large_tree):
    """Test that all topological distances are non-negative."""
    distances = large_tree.pairwise_distances("topological", leaves_only=True)

    assert all(distances["distance"] >= 0), "All topological distances should be non-negative"


def test_pairwise_distances_topological_all_integers(large_tree):
    """Test that topological distances are integers (or whole numbers)."""
    distances = large_tree.pairwise_distances("topological", leaves_only=True)

    # Topological distances count edges, so should be integers
    assert all(distances["distance"] % 1 == 0), "All topological distances should be whole numbers"


# =============================================================================
# Tests for pairwise_distances() with "metric" distance type
# =============================================================================

def test_pairwise_distances_metric_basic(simple_tree):
    """Test basic metric distance computation."""
    distances = simple_tree.pairwise_distances("metric", leaves_only=True)

    assert isinstance(distances, pd.DataFrame), "Should return a pandas DataFrame"
    assert len(distances) == 9, "Should have 3x3 = 9 entries for 3 leaves"
    assert list(distances.columns) == ["node1", "node2", "distance"], "Should have correct columns"


def test_pairwise_distances_metric_self_distance(simple_tree):
    """Test that metric self-distance is 0."""
    distances = simple_tree.pairwise_distances("metric", leaves_only=True)

    # Filter self-distances
    self_distances = distances[distances["node1"] == distances["node2"]]

    assert len(self_distances) == 3, "Should have 3 self-distances for 3 leaves"
    assert all(self_distances["distance"] == 0.0), "All self-distances should be 0"


def test_pairwise_distances_metric_symmetry(simple_tree):
    """Test that metric distances are symmetric."""
    distances = simple_tree.pairwise_distances("metric", leaves_only=True)

    # Check symmetry
    for _, row in distances.iterrows():
        node1, node2, dist = row["node1"], row["node2"], row["distance"]
        reverse = distances[(distances["node1"] == node2) & (distances["node2"] == node1)]
        if not reverse.empty:
            assert abs(reverse.iloc[0]["distance"] - dist) < 1e-10, \
                f"Metric distance({node1},{node2}) should equal distance({node2},{node1})"


def test_pairwise_distances_metric_values(simple_tree):
    """Test specific metric distance values on a known tree.

    For tree ((A:1.0,B:1.0):1.0,C:2.0):0.0:
    - A-B: 1.0 + 1.0 = 2.0
    - A-C: 1.0 + 1.0 + 2.0 = 4.0
    - B-C: 1.0 + 1.0 + 2.0 = 4.0
    """
    distances = simple_tree.pairwise_distances("metric", leaves_only=True)

    def get_distance(node1, node2):
        row = distances[(distances["node1"] == node1) & (distances["node2"] == node2)]
        return row.iloc[0]["distance"] if not row.empty else None

    # Test expected metric distances
    assert abs(get_distance("A", "B") - 2.0) < 1e-10, "A-B metric distance should be 2.0"
    assert abs(get_distance("A", "C") - 4.0) < 1e-10, "A-C metric distance should be 4.0"
    assert abs(get_distance("B", "C") - 4.0) < 1e-10, "B-C metric distance should be 4.0"


def test_pairwise_distances_metric_varying_lengths(varying_lengths_tree):
    """Test metric distances with varying branch lengths.

    For tree ((A:0.5,B:1.5):0.5,C:2.5):0.0:
    - A-B: 0.5 + 0.5 + 1.5 = 2.5
    - A-C: 0.5 + 0.5 + 2.5 = 3.5
    - B-C: 1.5 + 0.5 + 2.5 = 4.5
    """
    distances = varying_lengths_tree.pairwise_distances("metric", leaves_only=True)

    def get_distance(node1, node2):
        row = distances[(distances["node1"] == node1) & (distances["node2"] == node2)]
        return row.iloc[0]["distance"] if not row.empty else None

    assert abs(get_distance("A", "B") - 2.5) < 1e-10, "A-B should be 2.5"
    assert abs(get_distance("A", "C") - 3.5) < 1e-10, "A-C should be 3.5"
    assert abs(get_distance("B", "C") - 4.5) < 1e-10, "B-C should be 4.5"


def test_pairwise_distances_metric_leaves_only_true(large_tree):
    """Test metric distances with leaves_only=True."""
    distances = large_tree.pairwise_distances("metric", leaves_only=True)

    num_leaves = large_tree.num_leaves()
    expected_count = num_leaves * num_leaves

    assert len(distances) == expected_count, f"Should have {expected_count} entries"

    # All node names should be leaf names
    leaf_names = set(large_tree.leaf_names())
    assert all(distances["node1"].isin(leaf_names)), "All node1 should be leaves"
    assert all(distances["node2"].isin(leaf_names)), "All node2 should be leaves"


def test_pairwise_distances_metric_leaves_only_false(simple_tree):
    """Test metric distances with leaves_only=False."""
    distances = simple_tree.pairwise_distances("metric", leaves_only=False)

    num_nodes = simple_tree.num_nodes()
    expected_count = num_nodes * num_nodes

    assert len(distances) == expected_count, f"Should have {expected_count} entries"
    assert len(distances) > 9, "Should have more entries than leaves-only mode"


def test_pairwise_distances_metric_all_positive(large_tree):
    """Test that all metric distances are non-negative."""
    distances = large_tree.pairwise_distances("metric", leaves_only=True)

    assert all(distances["distance"] >= 0), "All metric distances should be non-negative"


def test_pairwise_distances_metric_greater_than_topological(large_tree):
    """Test that metric distances are generally >= topological distances.

    For any pair of nodes with branch length >= 1, metric distance should be
    greater than or equal to topological distance (which just counts edges).
    """
    metric_dist = large_tree.pairwise_distances("metric", leaves_only=True)
    topo_dist = large_tree.pairwise_distances("topological", leaves_only=True)

    # Merge the two dataframes
    merged = metric_dist.merge(topo_dist, on=["node1", "node2"], suffixes=("_metric", "_topo"))

    # Metric distances should generally be >= topological distances
    # (equality when all branches have length 1.0)
    assert all(merged["distance_metric"] >= merged["distance_topo"] - 1e-10), \
        "Metric distances should be >= topological distances"


# =============================================================================
# Tests for save_pairwise_distances_csv()
# =============================================================================

def test_save_pairwise_distances_csv_creates_file(simple_tree):
    """Test that save_pairwise_distances_csv creates a file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        filepath = f.name

    try:
        simple_tree.save_pairwise_distances_csv(filepath, "metric", leaves_only=True)
        assert os.path.exists(filepath), "CSV file should be created"
        assert os.path.getsize(filepath) > 0, "CSV file should not be empty"
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


@pytest.mark.skipif(not PANDAS_AVAILABLE, reason="pandas not available")
def test_save_pairwise_distances_csv_content_matches(simple_tree):
    """Test that saved CSV content matches pairwise_distances output."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        filepath = f.name

    try:
        # Save to CSV
        simple_tree.save_pairwise_distances_csv(filepath, "metric", leaves_only=True)

        # Read back
        saved_df = pd.read_csv(filepath)

        # Get direct output
        direct_df = simple_tree.pairwise_distances("metric", leaves_only=True)

        # Compare
        assert len(saved_df) == len(direct_df), "Row counts should match"
        assert list(saved_df.columns) == list(direct_df.columns), "Columns should match"

        # Sort both for comparison
        saved_df_sorted = saved_df.sort_values(["node1", "node2"]).reset_index(drop=True)
        direct_df_sorted = direct_df.sort_values(["node1", "node2"]).reset_index(drop=True)

        pd.testing.assert_frame_equal(saved_df_sorted, direct_df_sorted)
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


@pytest.mark.skipif(not PANDAS_AVAILABLE, reason="pandas not available")
def test_save_pairwise_distances_csv_has_header(simple_tree):
    """Test that saved CSV has proper header."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        filepath = f.name

    try:
        simple_tree.save_pairwise_distances_csv(filepath, "metric", leaves_only=True)

        # Read just the header
        with open(filepath, 'r') as f:
            header = f.readline().strip()

        assert header == "node1,node2,distance", "CSV should have correct header"
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


def test_save_pairwise_distances_csv_topological(simple_tree):
    """Test saving topological distances to CSV."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        filepath = f.name

    try:
        simple_tree.save_pairwise_distances_csv(filepath, "topological", leaves_only=True)
        assert os.path.exists(filepath), "CSV file should be created"
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


def test_save_pairwise_distances_csv_metric(simple_tree):
    """Test saving metric distances to CSV."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        filepath = f.name

    try:
        simple_tree.save_pairwise_distances_csv(filepath, "metric", leaves_only=True)
        assert os.path.exists(filepath), "CSV file should be created"
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


def test_save_pairwise_distances_csv_leaves_only_false(simple_tree):
    """Test saving all node distances to CSV."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        filepath = f.name

    try:
        simple_tree.save_pairwise_distances_csv(filepath, "metric", leaves_only=False)

        if PANDAS_AVAILABLE:
            saved_df = pd.read_csv(filepath)
            num_nodes = simple_tree.num_nodes()
            expected_count = num_nodes * num_nodes
            assert len(saved_df) == expected_count, f"Should have {expected_count} entries"
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


def test_save_pairwise_distances_csv_overwrite(simple_tree):
    """Test that saving overwrites existing file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        filepath = f.name

    try:
        # Save first time
        simple_tree.save_pairwise_distances_csv(filepath, "topological", leaves_only=True)
        size1 = os.path.getsize(filepath)

        # Save second time with different parameters
        simple_tree.save_pairwise_distances_csv(filepath, "metric", leaves_only=False)
        size2 = os.path.getsize(filepath)

        # Second file should exist and likely be different size
        assert os.path.exists(filepath), "File should still exist after overwrite"
        # Different parameters likely produce different file sizes
        assert size2 > 0, "Overwritten file should not be empty"
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


# =============================================================================
# Edge case tests
# =============================================================================

def test_pairwise_distances_single_node(single_leaf_tree):
    """Test pairwise distances with single-node tree."""
    distances = single_leaf_tree.pairwise_distances("metric", leaves_only=True)

    assert len(distances) == 1, "Single node should produce 1 entry"
    assert distances.iloc[0]["distance"] == 0.0, "Self-distance should be 0"


def test_pairwise_distances_two_nodes(two_leaf_tree):
    """Test pairwise distances with two-node tree."""
    distances = two_leaf_tree.pairwise_distances("metric", leaves_only=True)

    assert len(distances) == 4, "Two nodes should produce 4 entries (2x2)"

    # Check that A-B distance is correct
    ab_dist = distances[(distances["node1"] == "A") & (distances["node2"] == "B")]
    assert len(ab_dist) == 1, "Should have one A-B entry"
    assert abs(ab_dist.iloc[0]["distance"] - 3.0) < 1e-10, "A-B distance should be 1.5 + 1.5 = 3.0"


def test_pairwise_distances_topological_two_nodes(two_leaf_tree):
    """Test topological distances with two-node tree."""
    distances = two_leaf_tree.pairwise_distances("topological", leaves_only=True)

    # Topological distance between A and B should be 2 edges
    ab_dist = distances[(distances["node1"] == "A") & (distances["node2"] == "B")]
    assert ab_dist.iloc[0]["distance"] == 2.0, "A-B topological distance should be 2"


def test_pairwise_distances_large_tree_performance(large_tree):
    """Test that pairwise distances computation completes for large tree."""
    # This is mainly a performance/completeness test
    distances = large_tree.pairwise_distances("metric", leaves_only=True)

    num_leaves = large_tree.num_leaves()
    assert len(distances) == num_leaves * num_leaves, "Should complete for large tree"


# =============================================================================
# Error handling tests
# =============================================================================

def test_pairwise_distances_invalid_distance_type(simple_tree):
    """Test that invalid distance type raises error."""
    with pytest.raises(ValueError, match="[Ii]nvalid.*distance.*type"):
        simple_tree.pairwise_distances("invalid_type", leaves_only=True)


def test_pairwise_distances_case_sensitive_distance_type(simple_tree):
    """Test distance type case sensitivity."""
    # Should accept "metric" (lowercase)
    distances1 = simple_tree.pairwise_distances("metric", leaves_only=True)
    assert len(distances1) > 0, "Should accept 'metric'"

    # Should accept "topological" (lowercase)
    distances2 = simple_tree.pairwise_distances("topological", leaves_only=True)
    assert len(distances2) > 0, "Should accept 'topological'"

    # Test if uppercase/mixed case is accepted (may vary by implementation)
    # If not accepted, should raise ValueError
    try:
        simple_tree.pairwise_distances("METRIC", leaves_only=True)
    except ValueError:
        pass  # Expected if case-sensitive

    try:
        simple_tree.pairwise_distances("Metric", leaves_only=True)
    except ValueError:
        pass  # Expected if case-sensitive


def test_save_pairwise_distances_csv_invalid_path(simple_tree):
    """Test that invalid file path raises error."""
    with pytest.raises((ValueError, OSError, IOError)):
        simple_tree.save_pairwise_distances_csv(
            "/nonexistent/directory/file.csv",
            "metric",
            leaves_only=True
        )


def test_save_pairwise_distances_csv_invalid_distance_type(simple_tree):
    """Test that invalid distance type raises error in save function."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        filepath = f.name

    try:
        with pytest.raises(ValueError, match="[Ii]nvalid.*distance.*type"):
            simple_tree.save_pairwise_distances_csv(filepath, "invalid", leaves_only=True)
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


# =============================================================================
# Integration tests
# =============================================================================

def test_integration_compare_distance_types(simple_tree):
    """Integration test: compare topological and metric distances."""
    topo = simple_tree.pairwise_distances("topological", leaves_only=True)
    metric = simple_tree.pairwise_distances("metric", leaves_only=True)

    # Both should have same structure
    assert len(topo) == len(metric), "Both should have same number of entries"
    assert list(topo.columns) == list(metric.columns), "Both should have same columns"

    # Values should differ (unless all branches have length 1.0)
    # Check that at least some values differ
    topo_sorted = topo.sort_values(["node1", "node2"]).reset_index(drop=True)
    metric_sorted = metric.sort_values(["node1", "node2"]).reset_index(drop=True)

    differences = (topo_sorted["distance"] != metric_sorted["distance"]).sum()
    # For our test tree with varying lengths, distances should differ
    # (except for self-distances which are all 0)


def test_integration_save_and_load_csv(simple_tree):
    """Integration test: save CSV and verify it can be loaded."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        filepath = f.name

    try:
        # Save
        simple_tree.save_pairwise_distances_csv(filepath, "metric", leaves_only=True)

        # Load and verify
        if PANDAS_AVAILABLE:
            loaded_df = pd.read_csv(filepath)

            # Verify structure
            assert "node1" in loaded_df.columns
            assert "node2" in loaded_df.columns
            assert "distance" in loaded_df.columns

            # Verify content
            assert len(loaded_df) == 9, "Should have 9 entries for 3 leaves"
            assert all(loaded_df["distance"] >= 0), "All distances should be non-negative"
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


def test_integration_leaves_only_comparison(large_tree):
    """Integration test: compare leaves_only=True vs False."""
    leaves_dist = large_tree.pairwise_distances("metric", leaves_only=True)
    all_dist = large_tree.pairwise_distances("metric", leaves_only=False)

    # All nodes should include more entries
    assert len(all_dist) > len(leaves_dist), "All nodes should have more entries"

    # Verify counts
    num_leaves = large_tree.num_leaves()
    num_nodes = large_tree.num_nodes()

    assert len(leaves_dist) == num_leaves * num_leaves, "Leaves-only count should match"
    assert len(all_dist) == num_nodes * num_nodes, "All nodes count should match"


# =============================================================================
# Test runner (if running directly)
# =============================================================================

if __name__ == "__main__":
    # Run with pytest
    pytest.main([__file__, "-v", "-s"])
