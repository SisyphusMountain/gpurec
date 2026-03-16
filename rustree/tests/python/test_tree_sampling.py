"""
Comprehensive tests for tree sampling functionality in rustree.

This module tests the induced subtree extraction functionality including:
- extract_induced_subtree_by_names() (exposed as sample_by_names())
- Verification of resulting tree structure
- Error handling for edge cases
- Topology preservation after sampling
"""

import sys
sys.path.insert(0, "/home/enzo/Documents/Zombi/ZOMBI/zombi-rs/rustree/target/release")
import rustree
import pytest


# =============================================================================
# Fixtures - Shared test data
# =============================================================================

@pytest.fixture
def simple_species_tree():
    """Create a simple species tree for testing."""
    return rustree.simulate_species_tree(5, 1.0, 0.3, seed=42)


@pytest.fixture
def medium_species_tree():
    """Create a medium-sized species tree for testing."""
    return rustree.simulate_species_tree(20, 1.0, 0.3, seed=100)


@pytest.fixture
def simple_gene_tree(simple_species_tree):
    """Create a gene tree with enough extant genes for testing."""
    # Use parameters that typically produce multiple extant genes
    return simple_species_tree.simulate_dtl(0.5, 0.2, 0.1, require_extant=True, seed=42)


@pytest.fixture
def large_gene_tree(medium_species_tree):
    """Create a larger gene tree for comprehensive testing."""
    # Use high duplication to ensure many genes
    return medium_species_tree.simulate_dtl(2.0, 0.1, 0.1, require_extant=True, seed=200)


# =============================================================================
# Tests for extract_induced_subtree_by_names() - Basic Functionality
# =============================================================================

def test_sample_by_names_valid_subset(simple_gene_tree):
    """Test extract_induced_subtree with valid leaf names."""
    all_names = simple_gene_tree.extant_gene_names()

    # Skip if we don't have enough genes
    if len(all_names) < 2:
        pytest.skip("Not enough extant genes for this test")

    # Sample subset of genes
    subset = all_names[:2]
    sampled = simple_gene_tree.sample_by_names(subset)

    assert sampled is not None
    assert sampled.num_nodes() > 0
    assert sampled.num_extant() == len(subset)


def test_sample_by_names_returns_gene_tree(simple_gene_tree):
    """Test that sample_by_names returns a valid GeneTree object."""
    names = simple_gene_tree.extant_gene_names()

    if len(names) < 1:
        pytest.skip("No extant genes")

    sampled = simple_gene_tree.sample_by_names([names[0]])

    # Verify it has gene tree methods
    assert hasattr(sampled, 'num_extant')
    assert hasattr(sampled, 'to_newick')
    assert hasattr(sampled, 'count_events')


def test_sample_by_names_single_leaf(simple_gene_tree):
    """Test sampling with a single leaf name."""
    names = simple_gene_tree.extant_gene_names()

    if len(names) < 1:
        pytest.skip("No extant genes")

    sampled = simple_gene_tree.sample_by_names([names[0]])

    assert sampled.num_extant() == 1
    assert sampled.num_nodes() == 1
    sampled_names = sampled.extant_gene_names()
    assert len(sampled_names) == 1
    assert sampled_names[0] == names[0]


def test_sample_by_names_two_leaves(simple_gene_tree):
    """Test sampling with exactly two leaves."""
    names = simple_gene_tree.extant_gene_names()

    if len(names) < 2:
        pytest.skip("Not enough extant genes")

    subset = names[:2]
    sampled = simple_gene_tree.sample_by_names(subset)

    assert sampled.num_extant() == 2
    assert sampled.num_nodes() >= 2  # At least the 2 leaves, possibly MRCA
    sampled_names = set(sampled.extant_gene_names())
    assert sampled_names == set(subset)


def test_sample_by_names_all_leaves(simple_gene_tree):
    """Test sampling with all extant leaves."""
    all_names = simple_gene_tree.extant_gene_names()

    if len(all_names) == 0:
        pytest.skip("No extant genes")

    sampled = simple_gene_tree.sample_by_names(all_names)

    # Should preserve all extant genes
    assert sampled.num_extant() == len(all_names)
    assert set(sampled.extant_gene_names()) == set(all_names)


def test_sample_by_names_subset_half(large_gene_tree):
    """Test sampling approximately half the leaves."""
    all_names = large_gene_tree.extant_gene_names()

    if len(all_names) < 4:
        pytest.skip("Not enough extant genes")

    # Take every other gene
    subset = all_names[::2]
    sampled = large_gene_tree.sample_by_names(subset)

    assert sampled.num_extant() == len(subset)
    assert set(sampled.extant_gene_names()) == set(subset)


# =============================================================================
# Tests for Verifying Tree Structure After Sampling
# =============================================================================

def test_sampled_tree_has_only_specified_leaves(simple_gene_tree):
    """Verify the sampled tree contains only the specified leaves."""
    all_names = simple_gene_tree.extant_gene_names()

    if len(all_names) < 3:
        pytest.skip("Not enough extant genes")

    subset = all_names[:3]
    sampled = simple_gene_tree.sample_by_names(subset)

    sampled_names = sampled.extant_gene_names()

    # Should have exactly the specified leaves
    assert len(sampled_names) == len(subset)
    assert set(sampled_names) == set(subset)

    # No extra leaves
    for name in sampled_names:
        assert name in subset


def test_sampled_tree_no_loss_events(simple_gene_tree):
    """Verify sampled tree has no loss events (only extant genes)."""
    all_names = simple_gene_tree.extant_gene_names()

    if len(all_names) < 2:
        pytest.skip("Not enough extant genes")

    subset = all_names[:2]
    sampled = simple_gene_tree.sample_by_names(subset)

    events = sampled.count_events()

    # Sampled tree should have no loss events
    assert events["losses"] == 0, "Sampled tree should not contain loss events"
    assert events["leaves"] == len(subset), "Should have exactly as many leaves as sampled"


def test_sampled_tree_valid_structure(simple_gene_tree):
    """Verify sampled tree has valid binary tree structure."""
    all_names = simple_gene_tree.extant_gene_names()

    if len(all_names) < 2:
        pytest.skip("Not enough extant genes")

    subset = all_names[:2]
    sampled = simple_gene_tree.sample_by_names(subset)

    # Should have reasonable node count (at most 2n-1 for n leaves)
    n_leaves = len(subset)
    assert sampled.num_nodes() <= 2 * n_leaves - 1
    assert sampled.num_nodes() >= n_leaves


def test_sampled_tree_valid_newick(simple_gene_tree):
    """Verify sampled tree produces valid Newick string."""
    all_names = simple_gene_tree.extant_gene_names()

    if len(all_names) < 2:
        pytest.skip("Not enough extant genes")

    subset = all_names[:2]
    sampled = simple_gene_tree.sample_by_names(subset)

    newick = sampled.to_newick()

    # Basic Newick validation
    assert isinstance(newick, str)
    assert len(newick) > 0
    assert newick.endswith(';')
    assert newick.count('(') == newick.count(')')


# =============================================================================
# Tests with Different Subsets of Leaves
# =============================================================================

def test_sample_different_subsets(large_gene_tree):
    """Test sampling with various different subsets."""
    all_names = large_gene_tree.extant_gene_names()

    if len(all_names) < 10:
        pytest.skip("Not enough extant genes")

    # Test different subset sizes
    test_sizes = [2, 5, min(10, len(all_names))]

    for size in test_sizes:
        subset = all_names[:size]
        sampled = large_gene_tree.sample_by_names(subset)

        assert sampled.num_extant() == size
        assert set(sampled.extant_gene_names()) == set(subset)


def test_sample_first_vs_last_leaves(large_gene_tree):
    """Test sampling from beginning vs end of leaf list."""
    all_names = large_gene_tree.extant_gene_names()

    if len(all_names) < 6:
        pytest.skip("Not enough extant genes")

    # Sample first 3
    first_subset = all_names[:3]
    sampled_first = large_gene_tree.sample_by_names(first_subset)

    # Sample last 3
    last_subset = all_names[-3:]
    sampled_last = large_gene_tree.sample_by_names(last_subset)

    # Both should work correctly
    assert sampled_first.num_extant() == 3
    assert sampled_last.num_extant() == 3
    assert set(sampled_first.extant_gene_names()) == set(first_subset)
    assert set(sampled_last.extant_gene_names()) == set(last_subset)


def test_sample_random_scattered_leaves(large_gene_tree):
    """Test sampling scattered (non-consecutive) leaves."""
    all_names = large_gene_tree.extant_gene_names()

    if len(all_names) < 10:
        pytest.skip("Not enough extant genes")

    # Take every third gene
    scattered = all_names[::3]
    sampled = large_gene_tree.sample_by_names(scattered)

    assert sampled.num_extant() == len(scattered)
    assert set(sampled.extant_gene_names()) == set(scattered)


def test_sample_overlapping_subsets(simple_gene_tree):
    """Test that sampling overlapping subsets produces consistent results."""
    all_names = simple_gene_tree.extant_gene_names()

    if len(all_names) < 3:
        pytest.skip("Not enough extant genes")

    # Create two overlapping subsets
    subset1 = all_names[:2]
    subset2 = all_names[1:3]

    sampled1 = simple_gene_tree.sample_by_names(subset1)
    sampled2 = simple_gene_tree.sample_by_names(subset2)

    # Each should contain exactly its specified leaves
    assert set(sampled1.extant_gene_names()) == set(subset1)
    assert set(sampled2.extant_gene_names()) == set(subset2)


# =============================================================================
# Tests for Error Handling
# =============================================================================

def test_sample_empty_list_raises_error(simple_gene_tree):
    """Test that sampling with empty name list raises ValueError."""
    with pytest.raises(ValueError):
        simple_gene_tree.sample_by_names([])


def test_sample_nonexistent_name_raises_error(simple_gene_tree):
    """Test that sampling with non-existent name raises ValueError."""
    with pytest.raises(ValueError):
        simple_gene_tree.sample_by_names(["nonexistent_gene_12345"])


def test_sample_multiple_nonexistent_names_raises_error(simple_gene_tree):
    """Test that sampling with multiple non-existent names raises ValueError."""
    with pytest.raises(ValueError):
        simple_gene_tree.sample_by_names(["fake1", "fake2", "fake3"])


def test_sample_mixed_valid_invalid_names(simple_gene_tree):
    """Test that sampling with mix of valid and invalid names raises ValueError."""
    all_names = simple_gene_tree.extant_gene_names()

    if len(all_names) < 1:
        pytest.skip("No extant genes")

    # Mix valid and invalid names
    mixed_names = [all_names[0], "nonexistent_gene"]

    with pytest.raises(ValueError):
        simple_gene_tree.sample_by_names(mixed_names)


def test_sample_duplicate_names_in_list(simple_gene_tree):
    """Test sampling with duplicate names in the list."""
    all_names = simple_gene_tree.extant_gene_names()

    if len(all_names) < 1:
        pytest.skip("No extant genes")

    # Include same name twice
    duplicate_list = [all_names[0], all_names[0]]

    # Should either work (treating as single gene) or raise error
    try:
        sampled = simple_gene_tree.sample_by_names(duplicate_list)
        # If it works, should only have 1 gene
        assert sampled.num_extant() == 1
    except ValueError:
        # Also acceptable to reject duplicates
        pass


def test_sample_none_as_parameter(simple_gene_tree):
    """Test that passing None raises appropriate error."""
    with pytest.raises((TypeError, ValueError)):
        simple_gene_tree.sample_by_names(None)


def test_sample_non_list_parameter(simple_gene_tree):
    """Test that passing non-list raises appropriate error."""
    with pytest.raises((TypeError, ValueError)):
        simple_gene_tree.sample_by_names("not_a_list")


# =============================================================================
# Tests for Edge Cases
# =============================================================================

def test_sample_single_leaf_from_large_tree(large_gene_tree):
    """Test sampling just one leaf from a large tree."""
    all_names = large_gene_tree.extant_gene_names()

    if len(all_names) < 10:
        pytest.skip("Not enough extant genes")

    # Sample just one gene from many
    single = [all_names[5]]
    sampled = large_gene_tree.sample_by_names(single)

    assert sampled.num_extant() == 1
    assert sampled.num_nodes() == 1
    assert sampled.extant_gene_names()[0] == single[0]


def test_sample_all_but_one(simple_gene_tree):
    """Test sampling all leaves except one."""
    all_names = simple_gene_tree.extant_gene_names()

    if len(all_names) < 3:
        pytest.skip("Not enough extant genes")

    # Take all but the last one
    subset = all_names[:-1]
    sampled = simple_gene_tree.sample_by_names(subset)

    assert sampled.num_extant() == len(subset)
    assert set(sampled.extant_gene_names()) == set(subset)


def test_sample_tree_with_single_extant_gene():
    """Test sampling from tree that has only one extant gene."""
    # Create tree that might have few genes
    sp_tree = rustree.simulate_species_tree(5, 1.0, 0.3, seed=42)

    # Try to get a tree with few extant genes
    for i in range(50):
        gene_tree = sp_tree.simulate_dtl(0.1, 0.0, 0.5, require_extant=True, seed=1000 + i)
        names = gene_tree.extant_gene_names()

        if len(names) == 1:
            # Found one with single extant gene
            sampled = gene_tree.sample_by_names(names)
            assert sampled.num_extant() == 1
            return

    # If we didn't find such a tree, that's okay
    pytest.skip("Could not generate tree with single extant gene")


def test_sample_preserves_gene_order(simple_gene_tree):
    """Test that sampling with names in different order works correctly."""
    all_names = simple_gene_tree.extant_gene_names()

    if len(all_names) < 3:
        pytest.skip("Not enough extant genes")

    # Sample with different orderings
    subset = all_names[:3]
    reversed_subset = list(reversed(subset))

    sampled1 = simple_gene_tree.sample_by_names(subset)
    sampled2 = simple_gene_tree.sample_by_names(reversed_subset)

    # Both should have same genes (order of input shouldn't matter)
    assert set(sampled1.extant_gene_names()) == set(subset)
    assert set(sampled2.extant_gene_names()) == set(subset)
    assert sampled1.num_extant() == sampled2.num_extant()


# =============================================================================
# Tests for Topology Preservation
# =============================================================================

def test_topology_single_leaf_has_no_structure():
    """Test that single-leaf sampled tree has minimal structure."""
    sp_tree = rustree.simulate_species_tree(5, 1.0, 0.3, seed=42)
    gene_tree = sp_tree.simulate_dtl(0.5, 0.2, 0.1, require_extant=True, seed=42)

    names = gene_tree.extant_gene_names()
    if len(names) < 1:
        pytest.skip("No extant genes")

    sampled = gene_tree.sample_by_names([names[0]])

    # Single leaf should have no internal nodes
    assert sampled.num_nodes() == 1
    events = sampled.count_events()
    assert events["leaves"] == 1
    assert events["speciations"] == 0
    assert events["duplications"] == 0
    assert events["transfers"] == 0
    assert events["losses"] == 0


def test_topology_two_leaves_has_parent():
    """Test that two-leaf sampled tree has correct structure."""
    sp_tree = rustree.simulate_species_tree(5, 1.0, 0.3, seed=42)
    gene_tree = sp_tree.simulate_dtl(0.5, 0.2, 0.1, require_extant=True, seed=42)

    names = gene_tree.extant_gene_names()
    if len(names) < 2:
        pytest.skip("Not enough extant genes")

    sampled = gene_tree.sample_by_names(names[:2])

    # Two leaves should have at least 2 nodes (the leaves), possibly 3 with MRCA
    assert sampled.num_nodes() >= 2
    assert sampled.num_nodes() <= 3
    assert sampled.num_extant() == 2


def test_topology_complete_tree_unchanged():
    """Test that sampling all leaves preserves the extant gene topology."""
    sp_tree = rustree.simulate_species_tree(5, 1.0, 0.3, seed=42)
    gene_tree = sp_tree.simulate_dtl(0.5, 0.2, 0.1, require_extant=True, seed=42)

    all_names = gene_tree.extant_gene_names()
    if len(all_names) == 0:
        pytest.skip("No extant genes")

    # Sample all extant genes
    sampled = gene_tree.sample_by_names(all_names)

    # Should be equivalent to sample_extant()
    fully_sampled = gene_tree.sample_extant()

    assert sampled.num_extant() == fully_sampled.num_extant()
    assert set(sampled.extant_gene_names()) == set(fully_sampled.extant_gene_names())


def test_topology_binary_tree_structure(simple_gene_tree):
    """Test that sampled tree maintains binary structure."""
    all_names = simple_gene_tree.extant_gene_names()

    if len(all_names) < 4:
        pytest.skip("Not enough extant genes")

    subset = all_names[:4]
    sampled = simple_gene_tree.sample_by_names(subset)

    # For 4 leaves, binary tree should have at most 7 nodes (2*4-1)
    assert sampled.num_nodes() <= 7
    assert sampled.num_nodes() >= 4


def test_topology_newick_parseable(simple_gene_tree):
    """Test that sampled tree's Newick can be parsed by external tools."""
    all_names = simple_gene_tree.extant_gene_names()

    if len(all_names) < 2:
        pytest.skip("Not enough extant genes")

    subset = all_names[:2]
    sampled = simple_gene_tree.sample_by_names(subset)

    newick = sampled.to_newick()

    # Basic structural checks
    assert newick.count('(') == newick.count(')')
    assert ';' in newick

    # Should contain the gene names
    for name in subset:
        assert name in newick


# =============================================================================
# Tests for Reproducibility and Consistency
# =============================================================================

def test_sample_reproducible(simple_gene_tree):
    """Test that sampling the same names twice gives same result."""
    all_names = simple_gene_tree.extant_gene_names()

    if len(all_names) < 2:
        pytest.skip("Not enough extant genes")

    subset = all_names[:2]

    # Sample twice with same parameters
    sampled1 = simple_gene_tree.sample_by_names(subset)
    sampled2 = simple_gene_tree.sample_by_names(subset)

    # Should produce identical results
    assert sampled1.num_nodes() == sampled2.num_nodes()
    assert sampled1.num_extant() == sampled2.num_extant()
    assert sampled1.to_newick() == sampled2.to_newick()
    assert set(sampled1.extant_gene_names()) == set(sampled2.extant_gene_names())


def test_sample_consistent_event_counts(simple_gene_tree):
    """Test that event counts are consistent across multiple samplings."""
    all_names = simple_gene_tree.extant_gene_names()

    if len(all_names) < 3:
        pytest.skip("Not enough extant genes")

    subset = all_names[:3]

    # Sample multiple times
    counts_list = []
    for _ in range(3):
        sampled = simple_gene_tree.sample_by_names(subset)
        counts_list.append(sampled.count_events())

    # All counts should be identical
    assert all(counts == counts_list[0] for counts in counts_list)


# =============================================================================
# Integration Tests
# =============================================================================

def test_sample_then_resample(large_gene_tree):
    """Test sampling from an already-sampled tree."""
    all_names = large_gene_tree.extant_gene_names()

    if len(all_names) < 6:
        pytest.skip("Not enough extant genes")

    # First sampling
    subset1 = all_names[:6]
    sampled1 = large_gene_tree.sample_by_names(subset1)

    # Second sampling (subset of first)
    sampled1_names = sampled1.extant_gene_names()
    subset2 = sampled1_names[:3]
    sampled2 = sampled1.sample_by_names(subset2)

    # Should work correctly
    assert sampled2.num_extant() == 3
    assert set(sampled2.extant_gene_names()) == set(subset2)


def test_sample_multiple_different_subsets_from_same_tree(large_gene_tree):
    """Test creating multiple different samples from the same tree."""
    all_names = large_gene_tree.extant_gene_names()

    if len(all_names) < 9:
        pytest.skip("Not enough extant genes")

    # Create three non-overlapping samples
    subset1 = all_names[0:3]
    subset2 = all_names[3:6]
    subset3 = all_names[6:9]

    sampled1 = large_gene_tree.sample_by_names(subset1)
    sampled2 = large_gene_tree.sample_by_names(subset2)
    sampled3 = large_gene_tree.sample_by_names(subset3)

    # All should be valid and independent
    assert sampled1.num_extant() == 3
    assert sampled2.num_extant() == 3
    assert sampled3.num_extant() == 3

    assert set(sampled1.extant_gene_names()) == set(subset1)
    assert set(sampled2.extant_gene_names()) == set(subset2)
    assert set(sampled3.extant_gene_names()) == set(subset3)


def test_sample_workflow_save_and_export(simple_gene_tree, tmp_path):
    """Test complete workflow: sample, save Newick, save XML."""
    all_names = simple_gene_tree.extant_gene_names()

    if len(all_names) < 2:
        pytest.skip("Not enough extant genes")

    subset = all_names[:2]
    sampled = simple_gene_tree.sample_by_names(subset)

    # Save Newick
    newick_path = tmp_path / "sampled.nwk"
    sampled.save_newick(str(newick_path))
    assert newick_path.exists()
    assert newick_path.stat().st_size > 0

    # Save XML
    xml_path = tmp_path / "sampled.xml"
    sampled.save_xml(str(xml_path))
    assert xml_path.exists()
    assert xml_path.stat().st_size > 0


# =============================================================================
# Performance and Stress Tests
# =============================================================================

def test_sample_large_subset_from_very_large_tree():
    """Test sampling a large subset from a very large gene tree."""
    sp_tree = rustree.simulate_species_tree(50, 1.0, 0.3, seed=500)
    gene_tree = sp_tree.simulate_dtl(2.0, 0.1, 0.1, require_extant=True, seed=600)

    all_names = gene_tree.extant_gene_names()

    if len(all_names) < 50:
        pytest.skip("Not enough extant genes for stress test")

    # Sample 50 genes
    subset = all_names[:50]
    sampled = gene_tree.sample_by_names(subset)

    assert sampled.num_extant() == 50
    assert len(sampled.extant_gene_names()) == 50


def test_sample_many_small_subsets():
    """Test sampling many different small subsets sequentially."""
    sp_tree = rustree.simulate_species_tree(20, 1.0, 0.3, seed=700)
    gene_tree = sp_tree.simulate_dtl(1.0, 0.2, 0.1, require_extant=True, seed=800)

    all_names = gene_tree.extant_gene_names()

    if len(all_names) < 10:
        pytest.skip("Not enough extant genes")

    # Sample 10 different pairs
    for i in range(min(10, len(all_names) - 1)):
        subset = [all_names[i], all_names[i + 1]]
        sampled = gene_tree.sample_by_names(subset)
        assert sampled.num_extant() == 2


if __name__ == "__main__":
    # Run with pytest
    pytest.main([__file__, "-v"])
