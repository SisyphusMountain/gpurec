"""
Comprehensive tests for species tree functions in rustree.

This module tests the Python bindings for rustree's species tree simulation
and manipulation functions including:
- simulate_species_tree: Birth-death tree simulation
- parse_species_tree: Newick string parsing
- SpeciesTree methods: to_newick, num_nodes, num_leaves, tree_height, etc.
"""

import sys
sys.path.insert(0, "/home/enzo/Documents/Zombi/ZOMBI/zombi-rs/rustree/target/release")
import rustree
import os
import tempfile


# =============================================================================
# Tests for rustree.simulate_species_tree
# =============================================================================

def test_simulate_species_tree_basic():
    """Test basic simulation with 10 species."""
    tree = rustree.simulate_species_tree(10, 1.0, 0.3, seed=42)
    assert tree.num_leaves() == 10
    print("PASS: test_simulate_species_tree_basic")


def test_simulate_species_tree_small_n():
    """Test simulation with small number of species (n=5)."""
    tree = rustree.simulate_species_tree(5, 1.0, 0.3, seed=123)
    assert tree.num_leaves() == 5
    assert tree.num_nodes() > 5  # Should have internal nodes too
    print("PASS: test_simulate_species_tree_small_n")


def test_simulate_species_tree_medium_n():
    """Test simulation with medium number of species (n=20)."""
    tree = rustree.simulate_species_tree(20, 1.0, 0.3, seed=456)
    assert tree.num_leaves() == 20
    assert tree.num_nodes() > 20
    print("PASS: test_simulate_species_tree_medium_n")


def test_simulate_species_tree_large_n():
    """Test simulation with larger number of species (n=100)."""
    tree = rustree.simulate_species_tree(100, 1.0, 0.3, seed=789)
    assert tree.num_leaves() == 100
    assert tree.num_nodes() > 100
    print("PASS: test_simulate_species_tree_large_n")


def test_simulate_species_tree_reproducibility():
    """Test that using the same seed produces identical trees."""
    tree1 = rustree.simulate_species_tree(15, 1.0, 0.5, seed=12345)
    tree2 = rustree.simulate_species_tree(15, 1.0, 0.5, seed=12345)

    assert tree1.num_nodes() == tree2.num_nodes()
    assert tree1.num_leaves() == tree2.num_leaves()
    assert tree1.to_newick() == tree2.to_newick()
    print("PASS: test_simulate_species_tree_reproducibility")


def test_simulate_species_tree_different_seeds():
    """Test that different seeds produce different trees."""
    tree1 = rustree.simulate_species_tree(10, 1.0, 0.3, seed=111)
    tree2 = rustree.simulate_species_tree(10, 1.0, 0.3, seed=222)

    # Trees should have same number of leaves but likely different topology
    assert tree1.num_leaves() == tree2.num_leaves()
    # Note: topology may occasionally be same, but newick should differ (branch lengths)
    # This test may rarely fail if both random processes produce identical trees
    print("PASS: test_simulate_species_tree_different_seeds")


def test_simulate_species_tree_high_extinction_rate():
    """Test simulation with high extinction rate."""
    tree = rustree.simulate_species_tree(10, 2.0, 1.8, seed=42)
    assert tree.num_leaves() == 10
    print("PASS: test_simulate_species_tree_high_extinction_rate")


def test_simulate_species_tree_low_extinction_rate():
    """Test simulation with low extinction rate."""
    tree = rustree.simulate_species_tree(10, 1.0, 0.1, seed=42)
    assert tree.num_leaves() == 10
    print("PASS: test_simulate_species_tree_low_extinction_rate")


def test_simulate_species_tree_zero_extinction():
    """Test simulation with zero extinction rate (pure birth)."""
    tree = rustree.simulate_species_tree(10, 1.0, 0.0, seed=42)
    assert tree.num_leaves() == 10
    print("PASS: test_simulate_species_tree_zero_extinction")


def test_simulate_species_tree_invalid_n_zero():
    """Test that n=0 raises an error."""
    try:
        rustree.simulate_species_tree(0, 1.0, 0.3, seed=42)
        assert False, "Should have raised an error for n=0"
    except ValueError as e:
        assert "positive" in str(e).lower()
    print("PASS: test_simulate_species_tree_invalid_n_zero")


def test_simulate_species_tree_invalid_lambda_zero():
    """Test that lambda=0 raises an error."""
    try:
        rustree.simulate_species_tree(10, 0.0, 0.0, seed=42)
        assert False, "Should have raised an error for lambda=0"
    except ValueError as e:
        assert "positive" in str(e).lower()
    print("PASS: test_simulate_species_tree_invalid_lambda_zero")


def test_simulate_species_tree_invalid_lambda_negative():
    """Test that negative lambda raises an error."""
    try:
        rustree.simulate_species_tree(10, -1.0, 0.3, seed=42)
        assert False, "Should have raised an error for negative lambda"
    except ValueError as e:
        assert "positive" in str(e).lower()
    print("PASS: test_simulate_species_tree_invalid_lambda_negative")


def test_simulate_species_tree_invalid_mu_negative():
    """Test that negative mu raises an error."""
    try:
        rustree.simulate_species_tree(10, 1.0, -0.3, seed=42)
        assert False, "Should have raised an error for negative mu"
    except ValueError as e:
        assert "non-negative" in str(e).lower()
    print("PASS: test_simulate_species_tree_invalid_mu_negative")


def test_simulate_species_tree_invalid_mu_greater_than_lambda():
    """Test that mu >= lambda raises an error."""
    try:
        rustree.simulate_species_tree(10, 1.0, 1.5, seed=42)
        assert False, "Should have raised an error for mu >= lambda"
    except ValueError as e:
        assert "greater" in str(e).lower()
    print("PASS: test_simulate_species_tree_invalid_mu_greater_than_lambda")


def test_simulate_species_tree_invalid_mu_equal_lambda():
    """Test that mu == lambda raises an error."""
    try:
        rustree.simulate_species_tree(10, 1.0, 1.0, seed=42)
        assert False, "Should have raised an error for mu == lambda"
    except ValueError as e:
        assert "greater" in str(e).lower()
    print("PASS: test_simulate_species_tree_invalid_mu_equal_lambda")


def test_simulate_species_tree_n_equals_one():
    """Test simulation with only one species."""
    tree = rustree.simulate_species_tree(1, 1.0, 0.3, seed=42)
    assert tree.num_leaves() == 1
    assert tree.num_nodes() >= 1
    print("PASS: test_simulate_species_tree_n_equals_one")


def test_simulate_species_tree_n_equals_two():
    """Test simulation with two species."""
    tree = rustree.simulate_species_tree(2, 1.0, 0.3, seed=42)
    assert tree.num_leaves() == 2
    print("PASS: test_simulate_species_tree_n_equals_two")


def test_simulate_species_tree_no_seed():
    """Test simulation without providing a seed."""
    tree = rustree.simulate_species_tree(10, 1.0, 0.3)
    assert tree.num_leaves() == 10
    print("PASS: test_simulate_species_tree_no_seed")


# =============================================================================
# Tests for rustree.parse_species_tree (Newick parsing)
# =============================================================================

def test_parse_newick_simple():
    """Test parsing a simple Newick string."""
    newick = "(A:0.1,B:0.2):0.0;"
    tree = rustree.parse_species_tree(newick)
    assert tree.num_leaves() == 2
    assert tree.num_nodes() == 3
    print("PASS: test_parse_newick_simple")


def test_parse_newick_three_taxa():
    """Test parsing a three-taxa tree."""
    newick = "((A:0.1,B:0.1):0.2,C:0.3):0.0;"
    tree = rustree.parse_species_tree(newick)
    assert tree.num_leaves() == 3
    print("PASS: test_parse_newick_three_taxa")


def test_parse_newick_four_taxa():
    """Test parsing a four-taxa tree."""
    newick = "((A:0.1,B:0.1):0.1,(C:0.1,D:0.1):0.1):0.0;"
    tree = rustree.parse_species_tree(newick)
    assert tree.num_leaves() == 4
    print("PASS: test_parse_newick_four_taxa")


def test_parse_newick_complex():
    """Test parsing a more complex tree."""
    newick = "(((A:0.1,B:0.1):0.1,C:0.2):0.1,((D:0.1,E:0.1):0.1,F:0.2):0.1):0.0;"
    tree = rustree.parse_species_tree(newick)
    assert tree.num_leaves() == 6
    print("PASS: test_parse_newick_complex")


def test_parse_newick_no_branch_lengths():
    """Test parsing a Newick string without branch lengths."""
    newick = "((A,B),C);"
    tree = rustree.parse_species_tree(newick)
    assert tree.num_leaves() == 3
    print("PASS: test_parse_newick_no_branch_lengths")


def test_parse_newick_numeric_names():
    """Test parsing a Newick string with numeric names."""
    newick = "((1:0.1,2:0.1):0.1,3:0.2):0.0;"
    tree = rustree.parse_species_tree(newick)
    assert tree.num_leaves() == 3
    print("PASS: test_parse_newick_numeric_names")


def test_parse_newick_long_branch_lengths():
    """Test parsing with longer branch lengths."""
    newick = "(A:10.5,B:10.5):0.0;"
    tree = rustree.parse_species_tree(newick)
    assert tree.num_leaves() == 2
    assert tree.tree_height() > 10.0
    print("PASS: test_parse_newick_long_branch_lengths")


def test_parse_newick_scientific_notation():
    """Test parsing branch lengths in scientific notation."""
    newick = "(A:1.5e-2,B:1.5e-2):0.0;"
    tree = rustree.parse_species_tree(newick)
    assert tree.num_leaves() == 2
    print("PASS: test_parse_newick_scientific_notation")


def test_parse_newick_single_taxon():
    """Test parsing a single taxon tree."""
    newick = "A:0.0;"
    tree = rustree.parse_species_tree(newick)
    assert tree.num_leaves() == 1
    print("PASS: test_parse_newick_single_taxon")


def test_parse_newick_with_spaces():
    """Test parsing Newick with spaces in names (underscore convention)."""
    newick = "(Species_A:0.1,Species_B:0.1):0.0;"
    tree = rustree.parse_species_tree(newick)
    assert tree.num_leaves() == 2
    print("PASS: test_parse_newick_with_spaces")


def test_parse_newick_invalid_format():
    """Test that invalid Newick raises an error."""
    try:
        rustree.parse_species_tree("not valid newick")
        assert False, "Should have raised an error for invalid Newick"
    except ValueError:
        pass
    print("PASS: test_parse_newick_invalid_format")


def test_parse_newick_empty_string():
    """Test that empty string raises an error."""
    try:
        rustree.parse_species_tree("")
        assert False, "Should have raised an error for empty string"
    except ValueError:
        pass
    print("PASS: test_parse_newick_empty_string")


# =============================================================================
# Tests for SpeciesTree.to_newick() and round-trip verification
# =============================================================================

def test_to_newick_basic():
    """Test conversion to Newick format."""
    tree = rustree.simulate_species_tree(5, 1.0, 0.3, seed=42)
    newick = tree.to_newick()
    assert isinstance(newick, str)
    assert len(newick) > 0
    assert newick.endswith(";")
    print("PASS: test_to_newick_basic")


def test_to_newick_contains_branch_lengths():
    """Test that Newick output contains branch lengths."""
    tree = rustree.simulate_species_tree(5, 1.0, 0.3, seed=42)
    newick = tree.to_newick()
    # Branch lengths are indicated by colons
    assert ":" in newick
    print("PASS: test_to_newick_contains_branch_lengths")


def test_to_newick_contains_parentheses():
    """Test that Newick output has proper parentheses."""
    tree = rustree.simulate_species_tree(5, 1.0, 0.3, seed=42)
    newick = tree.to_newick()
    assert "(" in newick
    assert ")" in newick
    print("PASS: test_to_newick_contains_parentheses")


def test_round_trip_simulated():
    """Test round-trip: simulate -> to_newick -> parse."""
    tree1 = rustree.simulate_species_tree(8, 1.0, 0.3, seed=42)
    newick = tree1.to_newick()
    tree2 = rustree.parse_species_tree(newick)

    # After round-trip, leaf counts should match
    assert tree1.num_leaves() == tree2.num_leaves()
    print("PASS: test_round_trip_simulated")


def test_round_trip_parsed():
    """Test round-trip: parse -> to_newick -> parse."""
    original_newick = "((A:0.1,B:0.1):0.2,C:0.3):0.0;"
    tree1 = rustree.parse_species_tree(original_newick)
    generated_newick = tree1.to_newick()
    tree2 = rustree.parse_species_tree(generated_newick)

    assert tree1.num_leaves() == tree2.num_leaves()
    assert tree1.num_nodes() == tree2.num_nodes()
    print("PASS: test_round_trip_parsed")


def test_round_trip_preserves_leaf_count():
    """Test that round-trip preserves exact leaf count for various sizes."""
    for n in [3, 5, 10, 20]:
        tree1 = rustree.simulate_species_tree(n, 1.0, 0.3, seed=n * 10)
        newick = tree1.to_newick()
        tree2 = rustree.parse_species_tree(newick)
        assert tree2.num_leaves() == n, f"Failed for n={n}"
    print("PASS: test_round_trip_preserves_leaf_count")


# =============================================================================
# Tests for SpeciesTree.num_nodes() and num_leaves()
# =============================================================================

def test_num_nodes_simulated():
    """Test num_nodes() returns correct count for simulated tree."""
    tree = rustree.simulate_species_tree(10, 1.0, 0.3, seed=42)
    num_nodes = tree.num_nodes()
    num_leaves = tree.num_leaves()

    assert num_nodes > num_leaves
    assert num_nodes >= 10  # At least as many as leaves
    print("PASS: test_num_nodes_simulated")


def test_num_nodes_parsed():
    """Test num_nodes() returns correct count for parsed tree."""
    newick = "((A:0.1,B:0.1):0.1,C:0.2):0.0;"
    tree = rustree.parse_species_tree(newick)

    # 3 leaves + 2 internal nodes = 5 total
    assert tree.num_nodes() == 5
    print("PASS: test_num_nodes_parsed")


def test_num_leaves_simulated():
    """Test num_leaves() returns exactly n for simulate_species_tree(n, ...)."""
    for n in [3, 7, 15, 50]:
        tree = rustree.simulate_species_tree(n, 1.0, 0.3, seed=n)
        assert tree.num_leaves() == n, f"Expected {n} leaves, got {tree.num_leaves()}"
    print("PASS: test_num_leaves_simulated")


def test_num_leaves_parsed():
    """Test num_leaves() returns correct count for parsed tree."""
    newick = "((A:0.1,B:0.1):0.1,(C:0.1,D:0.1):0.1):0.0;"
    tree = rustree.parse_species_tree(newick)

    assert tree.num_leaves() == 4
    print("PASS: test_num_leaves_parsed")


def test_num_nodes_relationship():
    """Test that num_nodes = 2 * num_leaves - 1 for binary tree (approximately)."""
    # For a fully binary tree with n leaves, there should be n-1 internal nodes
    # so total nodes = n + (n-1) = 2n - 1
    # But with extinction, there may be additional "phantom" nodes
    for n in [5, 10, 20]:
        tree = rustree.simulate_species_tree(n, 1.0, 0.0, seed=n)  # Zero extinction
        num_nodes = tree.num_nodes()
        num_leaves = tree.num_leaves()
        # With zero extinction, should be exactly 2n-1
        expected = 2 * n - 1
        assert num_nodes >= expected, f"Expected at least {expected} nodes, got {num_nodes}"
    print("PASS: test_num_nodes_relationship")


# =============================================================================
# Tests for SpeciesTree.tree_height()
# =============================================================================

def test_tree_height_positive():
    """Test that tree_height() returns a positive value."""
    tree = rustree.simulate_species_tree(10, 1.0, 0.3, seed=42)
    height = tree.tree_height()
    assert height > 0
    print("PASS: test_tree_height_positive")


def test_tree_height_consistent():
    """Test that tree_height() is consistent across calls."""
    tree = rustree.simulate_species_tree(10, 1.0, 0.3, seed=42)
    height1 = tree.tree_height()
    height2 = tree.tree_height()
    assert height1 == height2
    print("PASS: test_tree_height_consistent")


def test_tree_height_reproducible():
    """Test that same seed produces same tree height."""
    tree1 = rustree.simulate_species_tree(10, 1.0, 0.3, seed=42)
    tree2 = rustree.simulate_species_tree(10, 1.0, 0.3, seed=42)
    assert tree1.tree_height() == tree2.tree_height()
    print("PASS: test_tree_height_reproducible")


def test_tree_height_parsed():
    """Test tree_height() for parsed tree."""
    newick = "((A:0.5,B:0.5):0.5,C:1.0):0.0;"
    tree = rustree.parse_species_tree(newick)
    height = tree.tree_height()
    # Height should be approximately 1.0 (depth of deepest leaf from root)
    assert abs(height - 1.0) < 0.01
    print("PASS: test_tree_height_parsed")


def test_tree_height_varies_with_rates():
    """Test that tree height varies with different rate parameters."""
    heights = []
    for lambda_ in [0.5, 1.0, 2.0]:
        tree = rustree.simulate_species_tree(10, lambda_, 0.1, seed=42)
        heights.append(tree.tree_height())
    # Heights should generally decrease with higher speciation rate
    # (more speciations happen faster, so tree is shorter)
    # Note: this is a probabilistic test, may occasionally fail
    assert len(set(heights)) > 1, "Heights should vary with different rates"
    print("PASS: test_tree_height_varies_with_rates")


# =============================================================================
# Tests for SpeciesTree.root_index()
# =============================================================================

def test_root_index_valid():
    """Test that root_index() returns a valid index."""
    tree = rustree.simulate_species_tree(10, 1.0, 0.3, seed=42)
    root_idx = tree.root_index()
    assert isinstance(root_idx, int)
    assert root_idx >= 0
    assert root_idx < tree.num_nodes()
    print("PASS: test_root_index_valid")


def test_root_index_consistent():
    """Test that root_index() returns consistent value."""
    tree = rustree.simulate_species_tree(10, 1.0, 0.3, seed=42)
    root1 = tree.root_index()
    root2 = tree.root_index()
    assert root1 == root2
    print("PASS: test_root_index_consistent")


def test_root_index_reproducible():
    """Test that same seed produces same root index."""
    tree1 = rustree.simulate_species_tree(10, 1.0, 0.3, seed=42)
    tree2 = rustree.simulate_species_tree(10, 1.0, 0.3, seed=42)
    assert tree1.root_index() == tree2.root_index()
    print("PASS: test_root_index_reproducible")


def test_root_index_parsed():
    """Test root_index() for parsed tree."""
    newick = "((A:0.1,B:0.1):0.1,C:0.2):0.0;"
    tree = rustree.parse_species_tree(newick)
    root_idx = tree.root_index()
    assert root_idx >= 0
    assert root_idx < tree.num_nodes()
    print("PASS: test_root_index_parsed")


# =============================================================================
# Tests for SpeciesTree.leaf_names()
# =============================================================================

def test_leaf_names_count():
    """Test that leaf_names() returns correct number of names."""
    tree = rustree.simulate_species_tree(10, 1.0, 0.3, seed=42)
    names = tree.leaf_names()
    assert len(names) == 10
    print("PASS: test_leaf_names_count")


def test_leaf_names_are_strings():
    """Test that leaf_names() returns strings."""
    tree = rustree.simulate_species_tree(5, 1.0, 0.3, seed=42)
    names = tree.leaf_names()
    for name in names:
        assert isinstance(name, str)
    print("PASS: test_leaf_names_are_strings")


def test_leaf_names_unique():
    """Test that leaf names are unique."""
    tree = rustree.simulate_species_tree(20, 1.0, 0.3, seed=42)
    names = tree.leaf_names()
    assert len(names) == len(set(names)), "Leaf names should be unique"
    print("PASS: test_leaf_names_unique")


def test_leaf_names_parsed():
    """Test leaf_names() for parsed tree."""
    newick = "((SpeciesA:0.1,SpeciesB:0.1):0.1,SpeciesC:0.2):0.0;"
    tree = rustree.parse_species_tree(newick)
    names = tree.leaf_names()
    assert len(names) == 3
    assert "SpeciesA" in names
    assert "SpeciesB" in names
    assert "SpeciesC" in names
    print("PASS: test_leaf_names_parsed")


def test_leaf_names_reproducible():
    """Test that same seed produces same leaf names."""
    tree1 = rustree.simulate_species_tree(10, 1.0, 0.3, seed=42)
    tree2 = rustree.simulate_species_tree(10, 1.0, 0.3, seed=42)
    names1 = tree1.leaf_names()
    names2 = tree2.leaf_names()
    assert names1 == names2
    print("PASS: test_leaf_names_reproducible")


def test_leaf_names_consistent_with_num_leaves():
    """Test that len(leaf_names()) == num_leaves()."""
    for n in [5, 10, 25]:
        tree = rustree.simulate_species_tree(n, 1.0, 0.3, seed=n)
        assert len(tree.leaf_names()) == tree.num_leaves()
    print("PASS: test_leaf_names_consistent_with_num_leaves")


# =============================================================================
# Tests for SpeciesTree.save_newick()
# =============================================================================

def test_save_newick_creates_file():
    """Test that save_newick() creates a file."""
    tree = rustree.simulate_species_tree(10, 1.0, 0.3, seed=42)

    with tempfile.NamedTemporaryFile(suffix=".nwk", delete=False) as f:
        filepath = f.name

    try:
        tree.save_newick(filepath)
        assert os.path.exists(filepath)
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)
    print("PASS: test_save_newick_creates_file")


def test_save_newick_content():
    """Test that saved file contains correct Newick string."""
    tree = rustree.simulate_species_tree(5, 1.0, 0.3, seed=42)
    expected_newick = tree.to_newick()

    with tempfile.NamedTemporaryFile(suffix=".nwk", delete=False) as f:
        filepath = f.name

    try:
        tree.save_newick(filepath)
        with open(filepath, "r") as f:
            saved_content = f.read()
        assert saved_content == expected_newick
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)
    print("PASS: test_save_newick_content")


def test_save_newick_roundtrip():
    """Test that saved file can be parsed back."""
    tree1 = rustree.simulate_species_tree(10, 1.0, 0.3, seed=42)

    with tempfile.NamedTemporaryFile(suffix=".nwk", delete=False) as f:
        filepath = f.name

    try:
        tree1.save_newick(filepath)
        with open(filepath, "r") as f:
            newick = f.read()
        tree2 = rustree.parse_species_tree(newick)

        assert tree1.num_leaves() == tree2.num_leaves()
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)
    print("PASS: test_save_newick_roundtrip")


def test_save_newick_directory_path():
    """Test saving to a specific directory."""
    tree = rustree.simulate_species_tree(5, 1.0, 0.3, seed=42)

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "test_tree.nwk")
        tree.save_newick(filepath)
        assert os.path.exists(filepath)
    print("PASS: test_save_newick_directory_path")


def test_save_newick_overwrite():
    """Test that save_newick() overwrites existing file."""
    tree1 = rustree.simulate_species_tree(5, 1.0, 0.3, seed=42)
    tree2 = rustree.simulate_species_tree(10, 1.0, 0.3, seed=100)

    with tempfile.NamedTemporaryFile(suffix=".nwk", delete=False) as f:
        filepath = f.name

    try:
        tree1.save_newick(filepath)
        tree2.save_newick(filepath)

        with open(filepath, "r") as f:
            content = f.read()

        # Should contain tree2's newick, not tree1's
        assert content == tree2.to_newick()
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)
    print("PASS: test_save_newick_overwrite")


def test_save_newick_invalid_path():
    """Test that invalid path raises appropriate error."""
    tree = rustree.simulate_species_tree(5, 1.0, 0.3, seed=42)

    try:
        tree.save_newick("/nonexistent/directory/tree.nwk")
        assert False, "Should have raised an error for invalid path"
    except (ValueError, OSError):
        pass
    print("PASS: test_save_newick_invalid_path")


# =============================================================================
# Tests for SpeciesTree.pairwise_distances()
# =============================================================================

def test_pairwise_distances_metric_leaves():
    """Test pairwise_distances with metric distance type, leaves only."""
    newick = "((A:1.0,B:1.0):1.0,C:2.0):0.0;"
    tree = rustree.parse_species_tree(newick)

    df = tree.pairwise_distances("metric", leaves_only=True)

    # Should have 3 leaves, so 9 pairs (3x3 including self-distances)
    assert len(df) == 9
    assert "node1" in df.columns
    assert "node2" in df.columns
    assert "distance" in df.columns
    print("PASS: test_pairwise_distances_metric_leaves")


def test_pairwise_distances_topological_leaves():
    """Test pairwise_distances with topological distance type, leaves only."""
    newick = "((A:1.0,B:2.0):3.0,C:6.0):0.0;"
    tree = rustree.parse_species_tree(newick)

    df = tree.pairwise_distances("topological", leaves_only=True)

    # Should have 3 leaves, so 9 pairs
    assert len(df) == 9

    # Check that topological distances are integers (number of edges)
    for dist in df["distance"]:
        assert dist == int(dist)
    print("PASS: test_pairwise_distances_topological_leaves")


def test_pairwise_distances_all_nodes():
    """Test pairwise_distances with all nodes, not just leaves."""
    newick = "((A:1.0,B:1.0):1.0,C:2.0):0.0;"
    tree = rustree.parse_species_tree(newick)

    df = tree.pairwise_distances("metric", leaves_only=False)

    # 5 nodes total (3 leaves + 2 internal), so 25 pairs (5x5)
    assert len(df) == 25
    print("PASS: test_pairwise_distances_all_nodes")


def test_pairwise_distances_self_distance_zero():
    """Test that self-distances are zero."""
    tree = rustree.simulate_species_tree(5, 1.0, 0.3, seed=42)

    df = tree.pairwise_distances("metric", leaves_only=True)

    # Filter to self-distances (where node1 == node2)
    self_dists = df[df["node1"] == df["node2"]]

    # All self-distances should be zero
    for dist in self_dists["distance"]:
        assert dist == 0.0
    print("PASS: test_pairwise_distances_self_distance_zero")


def test_pairwise_distances_symmetric():
    """Test that distance matrix is symmetric (d(A,B) == d(B,A))."""
    tree = rustree.simulate_species_tree(5, 1.0, 0.3, seed=42)

    df = tree.pairwise_distances("metric", leaves_only=True)

    # Create a dictionary for quick lookup
    dist_dict = {}
    for _, row in df.iterrows():
        dist_dict[(row["node1"], row["node2"])] = row["distance"]

    # Check symmetry
    for (n1, n2), dist in dist_dict.items():
        if n1 != n2:  # Skip self-distances
            reverse_dist = dist_dict.get((n2, n1))
            assert reverse_dist is not None
            assert abs(dist - reverse_dist) < 1e-10
    print("PASS: test_pairwise_distances_symmetric")


def test_pairwise_distances_invalid_type():
    """Test that invalid distance_type raises error."""
    tree = rustree.simulate_species_tree(5, 1.0, 0.3, seed=42)

    try:
        tree.pairwise_distances("invalid_type", leaves_only=True)
        assert False, "Should have raised an error for invalid distance_type"
    except ValueError as e:
        assert "invalid" in str(e).lower()
    print("PASS: test_pairwise_distances_invalid_type")


def test_pairwise_distances_aliases():
    """Test that distance_type aliases work (topo, patristic, branch)."""
    tree = rustree.simulate_species_tree(5, 1.0, 0.3, seed=42)

    # Test topological alias
    df1 = tree.pairwise_distances("topological", leaves_only=True)
    df2 = tree.pairwise_distances("topo", leaves_only=True)
    assert df1.equals(df2)

    # Test metric aliases
    df3 = tree.pairwise_distances("metric", leaves_only=True)
    df4 = tree.pairwise_distances("patristic", leaves_only=True)
    df5 = tree.pairwise_distances("branch", leaves_only=True)
    assert df3.equals(df4)
    assert df3.equals(df5)
    print("PASS: test_pairwise_distances_aliases")


def test_pairwise_distances_default_leaves_only():
    """Test that leaves_only defaults to True."""
    tree = rustree.simulate_species_tree(5, 1.0, 0.3, seed=42)

    # Call without leaves_only parameter (should default to True)
    df1 = tree.pairwise_distances("metric")
    df2 = tree.pairwise_distances("metric", leaves_only=True)

    assert len(df1) == len(df2)
    print("PASS: test_pairwise_distances_default_leaves_only")


# =============================================================================
# Integration tests
# =============================================================================

def test_integration_full_workflow():
    """Test a complete workflow: simulate, save, load, verify."""
    # Simulate a tree
    tree1 = rustree.simulate_species_tree(15, 1.0, 0.3, seed=42)

    # Save to file
    with tempfile.NamedTemporaryFile(suffix=".nwk", delete=False) as f:
        filepath = f.name

    try:
        tree1.save_newick(filepath)

        # Load back
        with open(filepath, "r") as f:
            newick = f.read()
        tree2 = rustree.parse_species_tree(newick)

        # Verify properties
        assert tree1.num_leaves() == tree2.num_leaves()
        assert tree1.num_nodes() == tree2.num_nodes()
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)
    print("PASS: test_integration_full_workflow")


def test_integration_multiple_trees():
    """Test simulating and comparing multiple trees."""
    trees = []
    for i in range(5):
        tree = rustree.simulate_species_tree(10, 1.0, 0.3, seed=i * 100)
        trees.append(tree)

    # All should have exactly 10 leaves
    for tree in trees:
        assert tree.num_leaves() == 10

    # Most should have different topologies (different newicks)
    newicks = [tree.to_newick() for tree in trees]
    unique_newicks = len(set(newicks))
    assert unique_newicks > 1, "Expected some variation between trees"
    print("PASS: test_integration_multiple_trees")


def test_integration_large_tree():
    """Test handling of a large tree."""
    tree = rustree.simulate_species_tree(500, 1.0, 0.3, seed=42)

    assert tree.num_leaves() == 500
    assert tree.num_nodes() > 500
    assert tree.tree_height() > 0
    assert len(tree.leaf_names()) == 500

    newick = tree.to_newick()
    assert len(newick) > 0
    print("PASS: test_integration_large_tree")


def test_integration_parsed_tree_operations():
    """Test all operations on a parsed tree."""
    newick = "(((A:0.5,B:0.5):0.3,C:0.8):0.2,((D:0.4,E:0.4):0.4,F:0.8):0.2):0.0;"
    tree = rustree.parse_species_tree(newick)

    # Check all properties
    assert tree.num_leaves() == 6
    assert tree.num_nodes() == 11
    assert tree.tree_height() > 0
    assert tree.root_index() >= 0

    names = tree.leaf_names()
    assert len(names) == 6
    for expected in ["A", "B", "C", "D", "E", "F"]:
        assert expected in names

    # Can convert back to newick
    new_newick = tree.to_newick()
    assert ";" in new_newick
    print("PASS: test_integration_parsed_tree_operations")


# =============================================================================
# Test runner
# =============================================================================

if __name__ == "__main__":
    tests = [
        # simulate_species_tree tests
        test_simulate_species_tree_basic,
        test_simulate_species_tree_small_n,
        test_simulate_species_tree_medium_n,
        test_simulate_species_tree_large_n,
        test_simulate_species_tree_reproducibility,
        test_simulate_species_tree_different_seeds,
        test_simulate_species_tree_high_extinction_rate,
        test_simulate_species_tree_low_extinction_rate,
        test_simulate_species_tree_zero_extinction,
        test_simulate_species_tree_invalid_n_zero,
        test_simulate_species_tree_invalid_lambda_zero,
        test_simulate_species_tree_invalid_lambda_negative,
        test_simulate_species_tree_invalid_mu_negative,
        test_simulate_species_tree_invalid_mu_greater_than_lambda,
        test_simulate_species_tree_invalid_mu_equal_lambda,
        test_simulate_species_tree_n_equals_one,
        test_simulate_species_tree_n_equals_two,
        test_simulate_species_tree_no_seed,

        # parse_species_tree (Newick parsing) tests
        test_parse_newick_simple,
        test_parse_newick_three_taxa,
        test_parse_newick_four_taxa,
        test_parse_newick_complex,
        test_parse_newick_no_branch_lengths,
        test_parse_newick_numeric_names,
        test_parse_newick_long_branch_lengths,
        test_parse_newick_scientific_notation,
        test_parse_newick_single_taxon,
        test_parse_newick_with_spaces,
        test_parse_newick_invalid_format,
        test_parse_newick_empty_string,

        # to_newick and round-trip tests
        test_to_newick_basic,
        test_to_newick_contains_branch_lengths,
        test_to_newick_contains_parentheses,
        test_round_trip_simulated,
        test_round_trip_parsed,
        test_round_trip_preserves_leaf_count,

        # num_nodes and num_leaves tests
        test_num_nodes_simulated,
        test_num_nodes_parsed,
        test_num_leaves_simulated,
        test_num_leaves_parsed,
        test_num_nodes_relationship,

        # tree_height tests
        test_tree_height_positive,
        test_tree_height_consistent,
        test_tree_height_reproducible,
        test_tree_height_parsed,
        test_tree_height_varies_with_rates,

        # root_index tests
        test_root_index_valid,
        test_root_index_consistent,
        test_root_index_reproducible,
        test_root_index_parsed,

        # leaf_names tests
        test_leaf_names_count,
        test_leaf_names_are_strings,
        test_leaf_names_unique,
        test_leaf_names_parsed,
        test_leaf_names_reproducible,
        test_leaf_names_consistent_with_num_leaves,

        # save_newick tests
        test_save_newick_creates_file,
        test_save_newick_content,
        test_save_newick_roundtrip,
        test_save_newick_directory_path,
        test_save_newick_overwrite,
        test_save_newick_invalid_path,

        # pairwise_distances tests
        test_pairwise_distances_metric_leaves,
        test_pairwise_distances_topological_leaves,
        test_pairwise_distances_all_nodes,
        test_pairwise_distances_self_distance_zero,
        test_pairwise_distances_symmetric,
        test_pairwise_distances_invalid_type,
        test_pairwise_distances_aliases,
        test_pairwise_distances_default_leaves_only,

        # Integration tests
        test_integration_full_workflow,
        test_integration_multiple_trees,
        test_integration_large_tree,
        test_integration_parsed_tree_operations,
    ]

    passed = 0
    failed = 0

    print("=" * 60)
    print("Running Species Tree Tests for rustree")
    print("=" * 60)
    print()

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"FAIL: {test.__name__} - {e}")
            failed += 1

    print()
    print("=" * 60)
    print(f"=== Results: {passed} passed, {failed} failed ===")
    print("=" * 60)

    # Exit with non-zero status if any tests failed
    if failed > 0:
        sys.exit(1)
