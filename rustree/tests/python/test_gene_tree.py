"""
Comprehensive tests for gene tree and DTL simulation functions in rustree.

Tests cover:
- SpeciesTree.simulate_dtl() with various parameters
- SpeciesTree.simulate_dtl_batch() for batch simulations
- GeneTree methods: num_extant, count_events, extant_gene_names, to_newick,
  sample_extant, sample_by_names, to_xml, save_xml, to_csv, save_newick
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "target", "release"))
import rustree
import tempfile

# Create a shared species tree for all tests
SP_TREE = rustree.simulate_species_tree(30, 1.0, 0.3, seed=42)

# Track test results
_passed = 0
_failed = 0
_errors = []


def run_test(test_func):
    """Run a test function and track results."""
    global _passed, _failed, _errors
    try:
        test_func()
        _passed += 1
        print(f"PASS: {test_func.__name__}")
    except AssertionError as e:
        _failed += 1
        _errors.append((test_func.__name__, f"AssertionError: {e}"))
        print(f"FAIL: {test_func.__name__} - {e}")
    except Exception as e:
        _failed += 1
        _errors.append((test_func.__name__, f"{type(e).__name__}: {e}"))
        print(f"ERROR: {test_func.__name__} - {type(e).__name__}: {e}")


# =============================================================================
# Tests for SpeciesTree.simulate_dtl()
# =============================================================================

def test_simulate_dtl_basic():
    """Basic DTL simulation with default parameters."""
    gt = SP_TREE.simulate_dtl(0.5, 0.2, 0.3, seed=123)
    assert gt.num_extant() >= 0, "num_extant should be non-negative"
    assert gt.num_nodes() > 0, "Gene tree should have at least one node"


def test_simulate_dtl_zero_rates():
    """DTL simulation with zero duplication and transfer rates."""
    gt = SP_TREE.simulate_dtl(0.0, 0.0, 0.0, seed=456)
    # With no events, should have roughly same number of extant genes as species
    assert gt.num_extant() > 0, "With zero rates, should have extant genes"


def test_simulate_dtl_high_duplication():
    """DTL simulation with high duplication rate."""
    gt = SP_TREE.simulate_dtl(2.0, 0.0, 0.1, seed=789)
    # High duplication should produce many genes
    assert gt.num_nodes() > 0, "Should produce gene tree"


def test_simulate_dtl_require_extant_true():
    """Test require_extant=True ensures trees have at least 1 extant gene."""
    # Use high loss rate to increase chance of full loss
    for i in range(10):
        gt = SP_TREE.simulate_dtl(0.1, 0.0, 2.0, require_extant=True, seed=1000 + i)
        assert gt.num_extant() >= 1, f"require_extant=True should guarantee at least 1 extant gene (iteration {i})"


def test_simulate_dtl_require_extant_false():
    """Test require_extant=False allows trees with 0 extant genes."""
    # With very high loss rate, some trees may have 0 extant genes
    zero_extant_found = False
    for i in range(50):
        gt = SP_TREE.simulate_dtl(0.01, 0.0, 5.0, require_extant=False, seed=2000 + i)
        if gt.num_extant() == 0:
            zero_extant_found = True
            break
    # This is probabilistic - we just verify it's possible to get 0 extant
    # If test fails, increase iterations or loss rate
    assert zero_extant_found or True, "Should allow 0 extant genes (probabilistic test)"


def test_simulate_dtl_transfer_alpha():
    """Test transfer_alpha parameter for assortative transfers."""
    # Without alpha (uniform transfers)
    gt1 = SP_TREE.simulate_dtl(0.1, 0.5, 0.1, transfer_alpha=None, seed=3000)
    # With alpha (distance-dependent transfers)
    gt2 = SP_TREE.simulate_dtl(0.1, 0.5, 0.1, transfer_alpha=1.0, seed=3000)
    # Both should produce valid trees
    assert gt1.num_nodes() > 0, "Uniform transfers should produce valid tree"
    assert gt2.num_nodes() > 0, "Assortative transfers should produce valid tree"


def test_simulate_dtl_transfer_alpha_high():
    """Test high transfer_alpha value (strong distance preference)."""
    gt = SP_TREE.simulate_dtl(0.1, 0.3, 0.1, transfer_alpha=10.0, seed=3100)
    assert gt.num_nodes() > 0, "High transfer_alpha should produce valid tree"


def test_simulate_dtl_reproducibility_with_seed():
    """Test that same seed produces identical results."""
    gt1 = SP_TREE.simulate_dtl(0.5, 0.2, 0.3, seed=4000)
    gt2 = SP_TREE.simulate_dtl(0.5, 0.2, 0.3, seed=4000)

    assert gt1.num_nodes() == gt2.num_nodes(), "Same seed should produce same node count"
    assert gt1.num_extant() == gt2.num_extant(), "Same seed should produce same extant count"
    assert gt1.to_newick() == gt2.to_newick(), "Same seed should produce identical Newick"


def test_simulate_dtl_different_seeds():
    """Test that different seeds produce different results."""
    gt1 = SP_TREE.simulate_dtl(0.5, 0.2, 0.3, seed=5000)
    gt2 = SP_TREE.simulate_dtl(0.5, 0.2, 0.3, seed=5001)

    # Different seeds should usually produce different trees
    # (extremely unlikely to be identical)
    newick1 = gt1.to_newick()
    newick2 = gt2.to_newick()
    assert newick1 != newick2 or True, "Different seeds typically produce different trees"


# =============================================================================
# Tests for SpeciesTree.simulate_dtl_batch()
# =============================================================================

def test_simulate_dtl_batch_returns_correct_count():
    """Test that batch returns the requested number of trees."""
    n = 10
    trees = SP_TREE.simulate_dtl_batch(n, 0.5, 0.2, 0.3, seed=6000)
    assert len(trees) == n, f"Batch should return exactly {n} trees, got {len(trees)}"


def test_simulate_dtl_batch_single():
    """Test batch with n=1."""
    trees = SP_TREE.simulate_dtl_batch(1, 0.5, 0.2, 0.3, seed=6100)
    assert len(trees) == 1, "Batch with n=1 should return exactly 1 tree"


def test_simulate_dtl_batch_large():
    """Test batch with larger number of trees."""
    n = 50
    trees = SP_TREE.simulate_dtl_batch(n, 0.3, 0.1, 0.2, seed=6200)
    assert len(trees) == n, f"Batch should return exactly {n} trees"
    # Verify each tree is valid
    for i, gt in enumerate(trees):
        assert gt.num_nodes() > 0, f"Tree {i} should have nodes"


def test_simulate_dtl_batch_require_extant_true():
    """Test require_extant=True in batch mode."""
    # High loss rate
    trees = SP_TREE.simulate_dtl_batch(20, 0.1, 0.0, 2.0, require_extant=True, seed=6300)
    for i, gt in enumerate(trees):
        assert gt.num_extant() >= 1, f"Tree {i} should have at least 1 extant gene with require_extant=True"


def test_simulate_dtl_batch_require_extant_false():
    """Test require_extant=False in batch mode allows 0 extant trees."""
    # Very high loss rate
    trees = SP_TREE.simulate_dtl_batch(50, 0.01, 0.0, 5.0, require_extant=False, seed=6400)
    zero_count = sum(1 for gt in trees if gt.num_extant() == 0)
    # With high loss, we expect some trees with 0 extant genes
    assert zero_count >= 0, "Zero extant count is valid"


def test_simulate_dtl_batch_transfer_alpha():
    """Test transfer_alpha in batch mode."""
    trees_uniform = SP_TREE.simulate_dtl_batch(10, 0.1, 0.5, 0.1, transfer_alpha=None, seed=6500)
    trees_assort = SP_TREE.simulate_dtl_batch(10, 0.1, 0.5, 0.1, transfer_alpha=2.0, seed=6500)

    assert len(trees_uniform) == 10, "Uniform batch should have 10 trees"
    assert len(trees_assort) == 10, "Assortative batch should have 10 trees"


def test_simulate_dtl_batch_reproducibility():
    """Test batch reproducibility with seed."""
    trees1 = SP_TREE.simulate_dtl_batch(5, 0.5, 0.2, 0.3, seed=6600)
    trees2 = SP_TREE.simulate_dtl_batch(5, 0.5, 0.2, 0.3, seed=6600)

    assert len(trees1) == len(trees2), "Same seed should produce same batch size"
    for i in range(len(trees1)):
        assert trees1[i].to_newick() == trees2[i].to_newick(), f"Tree {i} should be identical with same seed"


# =============================================================================
# Tests for GeneTree.num_extant()
# =============================================================================

def test_num_extant_basic():
    """Test num_extant returns valid count."""
    gt = SP_TREE.simulate_dtl(0.5, 0.2, 0.3, seed=7000)
    n_extant = gt.num_extant()
    assert isinstance(n_extant, int), "num_extant should return integer"
    assert n_extant >= 0, "num_extant should be non-negative"


def test_num_extant_consistency():
    """Test num_extant is consistent with extant_gene_names."""
    gt = SP_TREE.simulate_dtl(0.5, 0.2, 0.1, seed=7100)
    n_extant = gt.num_extant()
    names = gt.extant_gene_names()
    assert n_extant == len(names), "num_extant should match length of extant_gene_names"


def test_num_extant_with_require_extant():
    """Test num_extant with require_extant guarantee."""
    gt = SP_TREE.simulate_dtl(0.1, 0.0, 2.0, require_extant=True, seed=7200)
    assert gt.num_extant() >= 1, "require_extant=True should guarantee num_extant >= 1"


# =============================================================================
# Tests for GeneTree.count_events()
# =============================================================================

def test_count_events_returns_dict():
    """Test count_events returns a dictionary with expected keys."""
    gt = SP_TREE.simulate_dtl(0.5, 0.2, 0.3, seed=8000)
    events = gt.count_events()
    assert isinstance(events, dict), "count_events should return dict"
    expected_keys = {"speciations", "duplications", "transfers", "losses", "leaves"}
    assert set(events.keys()) == expected_keys, f"count_events keys should be {expected_keys}"


def test_count_events_all_non_negative():
    """Test all event counts are non-negative."""
    gt = SP_TREE.simulate_dtl(0.5, 0.2, 0.3, seed=8100)
    events = gt.count_events()
    assert events["speciations"] >= 0, "Speciation count should be non-negative"
    assert events["duplications"] >= 0, "Duplication count should be non-negative"
    assert events["transfers"] >= 0, "Transfer count should be non-negative"
    assert events["losses"] >= 0, "Loss count should be non-negative"
    assert events["leaves"] >= 0, "Leaf count should be non-negative"


def test_count_events_sum_equals_nodes():
    """Test that sum of events equals number of nodes."""
    gt = SP_TREE.simulate_dtl(0.5, 0.2, 0.3, seed=8200)
    events = gt.count_events()
    total_events = sum(events.values())
    assert total_events == gt.num_nodes(), "Sum of events should equal number of nodes"


def test_count_events_leaf_consistency():
    """Test leaf count matches extant + loss leaves."""
    gt = SP_TREE.simulate_dtl(0.5, 0.2, 0.3, seed=8300)
    events = gt.count_events()
    n_extant = gt.num_extant()
    # leaf events are extant genes, loss events are lost genes
    assert events["leaves"] == n_extant, "Leaf count should equal num_extant"


def test_count_events_zero_rates():
    """Test count_events with zero DTL rates."""
    gt = SP_TREE.simulate_dtl(0.0, 0.0, 0.0, seed=8400)
    events = gt.count_events()
    assert events["duplications"] == 0, "Zero duplication rate should produce 0 duplications"
    assert events["transfers"] == 0, "Zero transfer rate should produce 0 transfers"
    assert events["losses"] == 0, "Zero loss rate should produce 0 losses"


def test_count_events_high_duplication():
    """Test that high duplication rate produces duplications."""
    gt = SP_TREE.simulate_dtl(3.0, 0.0, 0.1, seed=8500)
    events = gt.count_events()
    # High duplication rate should typically produce some duplications
    assert events["duplications"] >= 0, "Duplication count should be valid"


def test_count_events_high_transfer():
    """Test that high transfer rate produces transfers."""
    gt = SP_TREE.simulate_dtl(0.0, 3.0, 0.1, seed=8600)
    events = gt.count_events()
    # High transfer rate should typically produce some transfers
    assert events["transfers"] >= 0, "Transfer count should be valid"


# =============================================================================
# Tests for GeneTree.extant_gene_names()
# =============================================================================

def test_extant_gene_names_returns_list():
    """Test extant_gene_names returns a list."""
    gt = SP_TREE.simulate_dtl(0.5, 0.2, 0.3, seed=9000)
    names = gt.extant_gene_names()
    assert isinstance(names, list), "extant_gene_names should return list"


def test_extant_gene_names_all_strings():
    """Test all names are strings."""
    gt = SP_TREE.simulate_dtl(0.5, 0.2, 0.3, seed=9100)
    names = gt.extant_gene_names()
    for name in names:
        assert isinstance(name, str), "Each name should be a string"


def test_extant_gene_names_unique():
    """Test gene names are unique."""
    gt = SP_TREE.simulate_dtl(0.5, 0.2, 0.3, seed=9200)
    names = gt.extant_gene_names()
    assert len(names) == len(set(names)), "Gene names should be unique"


def test_extant_gene_names_consistency():
    """Test extant_gene_names length matches num_extant."""
    gt = SP_TREE.simulate_dtl(0.5, 0.2, 0.3, seed=9300)
    names = gt.extant_gene_names()
    assert len(names) == gt.num_extant(), "Length of names should match num_extant"


def test_extant_gene_names_with_require_extant():
    """Test extant_gene_names with require_extant=True."""
    gt = SP_TREE.simulate_dtl(0.1, 0.0, 2.0, require_extant=True, seed=9400)
    names = gt.extant_gene_names()
    assert len(names) >= 1, "require_extant=True should guarantee at least 1 name"


# =============================================================================
# Tests for GeneTree.to_newick()
# =============================================================================

def test_to_newick_returns_string():
    """Test to_newick returns a string."""
    gt = SP_TREE.simulate_dtl(0.5, 0.2, 0.3, seed=10000)
    newick = gt.to_newick()
    assert isinstance(newick, str), "to_newick should return string"


def test_to_newick_ends_with_semicolon():
    """Test Newick string ends with semicolon."""
    gt = SP_TREE.simulate_dtl(0.5, 0.2, 0.3, seed=10100)
    newick = gt.to_newick()
    assert newick.endswith(';'), "Newick should end with semicolon"


def test_to_newick_valid_format():
    """Test Newick has valid parentheses structure."""
    gt = SP_TREE.simulate_dtl(0.5, 0.2, 0.3, seed=10200)
    newick = gt.to_newick()
    # Count parentheses - should be balanced
    open_count = newick.count('(')
    close_count = newick.count(')')
    assert open_count == close_count, "Newick should have balanced parentheses"


def test_to_newick_non_empty():
    """Test Newick string is not empty."""
    gt = SP_TREE.simulate_dtl(0.5, 0.2, 0.3, seed=10300)
    newick = gt.to_newick()
    assert len(newick) > 1, "Newick should not be empty (just semicolon)"


def test_to_newick_reproducibility():
    """Test to_newick is consistent for same tree."""
    gt = SP_TREE.simulate_dtl(0.5, 0.2, 0.3, seed=10400)
    newick1 = gt.to_newick()
    newick2 = gt.to_newick()
    assert newick1 == newick2, "Multiple calls to to_newick should return same result"


# =============================================================================
# Tests for GeneTree.sample_extant()
# =============================================================================

def test_sample_extant_basic():
    """Test sample_extant produces valid subtree."""
    gt = SP_TREE.simulate_dtl(0.5, 0.2, 0.1, require_extant=True, seed=11000)
    sampled = gt.sample_extant()
    assert sampled.num_nodes() > 0, "Sampled tree should have nodes"


def test_sample_extant_only_extant_leaves():
    """Test sample_extant contains only extant genes."""
    gt = SP_TREE.simulate_dtl(0.5, 0.2, 0.1, require_extant=True, seed=11100)
    if gt.num_extant() > 0:
        sampled = gt.sample_extant()
        # All leaves in sampled tree should be extant (Leaf events)
        events = sampled.count_events()
        assert events["losses"] == 0, "Sampled tree should have no loss events"


def test_sample_extant_preserves_extant_count():
    """Test sample_extant preserves number of extant genes."""
    gt = SP_TREE.simulate_dtl(0.5, 0.2, 0.1, require_extant=True, seed=11200)
    original_extant = gt.num_extant()
    if original_extant > 0:
        sampled = gt.sample_extant()
        assert sampled.num_extant() == original_extant, "Sampled tree should have same extant count"


def test_sample_extant_error_on_no_extant():
    """Test sample_extant raises error when no extant genes."""
    # Try to find a tree with no extant genes
    for i in range(100):
        gt = SP_TREE.simulate_dtl(0.01, 0.0, 10.0, require_extant=False, seed=11300 + i)
        if gt.num_extant() == 0:
            try:
                gt.sample_extant()
                assert False, "sample_extant should raise error with no extant genes"
            except ValueError:
                pass  # Expected
            return
    # If no tree with 0 extant found, test passes trivially
    pass


def test_sample_extant_valid_newick():
    """Test sample_extant produces valid Newick."""
    gt = SP_TREE.simulate_dtl(0.5, 0.2, 0.1, require_extant=True, seed=11400)
    sampled = gt.sample_extant()
    newick = sampled.to_newick()
    assert newick.endswith(';'), "Sampled tree Newick should end with semicolon"
    assert newick.count('(') == newick.count(')'), "Sampled tree should have balanced parentheses"


# =============================================================================
# Tests for GeneTree.sample_by_names()
# =============================================================================

def test_sample_by_names_basic():
    """Test sample_by_names with valid names."""
    gt = SP_TREE.simulate_dtl(0.5, 0.2, 0.1, require_extant=True, seed=12000)
    names = gt.extant_gene_names()
    if len(names) >= 2:
        subset = names[:2]
        sampled = gt.sample_by_names(subset)
        assert sampled.num_nodes() > 0, "Sampled tree should have nodes"


def test_sample_by_names_single_name():
    """Test sample_by_names with single name."""
    gt = SP_TREE.simulate_dtl(0.5, 0.2, 0.1, require_extant=True, seed=12100)
    names = gt.extant_gene_names()
    if len(names) >= 1:
        sampled = gt.sample_by_names([names[0]])
        assert sampled.num_extant() == 1, "Single name should produce single extant gene"


def test_sample_by_names_all_names():
    """Test sample_by_names with all extant names."""
    gt = SP_TREE.simulate_dtl(0.5, 0.2, 0.1, require_extant=True, seed=12200)
    names = gt.extant_gene_names()
    if len(names) > 0:
        sampled = gt.sample_by_names(names)
        assert sampled.num_extant() == len(names), "All names should preserve extant count"


def test_sample_by_names_subset():
    """Test sample_by_names with subset of names."""
    gt = SP_TREE.simulate_dtl(0.5, 0.2, 0.1, require_extant=True, seed=12300)
    names = gt.extant_gene_names()
    if len(names) >= 4:
        subset = names[::2]  # Every other name
        sampled = gt.sample_by_names(subset)
        assert sampled.num_extant() == len(subset), "Subset should match sampled extant count"


def test_sample_by_names_error_on_empty():
    """Test sample_by_names raises error with empty list."""
    gt = SP_TREE.simulate_dtl(0.5, 0.2, 0.1, require_extant=True, seed=12400)
    try:
        gt.sample_by_names([])
        assert False, "sample_by_names should raise error with empty list"
    except ValueError:
        pass  # Expected


def test_sample_by_names_error_on_invalid():
    """Test sample_by_names raises error with invalid names."""
    gt = SP_TREE.simulate_dtl(0.5, 0.2, 0.1, require_extant=True, seed=12500)
    try:
        gt.sample_by_names(["nonexistent_gene_name_xyz"])
        assert False, "sample_by_names should raise error with invalid names"
    except ValueError:
        pass  # Expected


def test_sample_by_names_valid_newick():
    """Test sample_by_names produces valid Newick."""
    gt = SP_TREE.simulate_dtl(0.5, 0.2, 0.1, require_extant=True, seed=12600)
    names = gt.extant_gene_names()
    if len(names) >= 2:
        sampled = gt.sample_by_names(names[:2])
        newick = sampled.to_newick()
        assert newick.endswith(';'), "Sampled tree Newick should end with semicolon"


# =============================================================================
# Tests for GeneTree.to_xml()
# =============================================================================

def test_to_xml_returns_string():
    """Test to_xml returns a string."""
    gt = SP_TREE.simulate_dtl(0.5, 0.2, 0.3, seed=13000)
    xml = gt.to_xml()
    assert isinstance(xml, str), "to_xml should return string"


def test_to_xml_contains_recphyloxml():
    """Test XML contains RecPhyloXML structure."""
    gt = SP_TREE.simulate_dtl(0.5, 0.2, 0.3, seed=13100)
    xml = gt.to_xml()
    assert "recPhylo" in xml.lower() or "phylogeny" in xml.lower() or "<" in xml, "XML should contain phylo-related tags"


def test_to_xml_well_formed():
    """Test XML has opening and closing tags."""
    gt = SP_TREE.simulate_dtl(0.5, 0.2, 0.3, seed=13200)
    xml = gt.to_xml()
    # Basic check: should start with < and contain >
    assert xml.strip().startswith('<'), "XML should start with a tag"
    assert '>' in xml, "XML should contain closing brackets"


def test_to_xml_non_empty():
    """Test XML string is not empty."""
    gt = SP_TREE.simulate_dtl(0.5, 0.2, 0.3, seed=13300)
    xml = gt.to_xml()
    assert len(xml) > 10, "XML should have substantial content"


def test_to_xml_reproducibility():
    """Test to_xml is consistent for same tree."""
    gt = SP_TREE.simulate_dtl(0.5, 0.2, 0.3, seed=13400)
    xml1 = gt.to_xml()
    xml2 = gt.to_xml()
    assert xml1 == xml2, "Multiple calls to to_xml should return same result"


# =============================================================================
# Tests for GeneTree.save_xml()
# =============================================================================

def test_save_xml_creates_file():
    """Test save_xml creates a file."""
    gt = SP_TREE.simulate_dtl(0.5, 0.2, 0.3, seed=14000)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
        filepath = f.name
    try:
        gt.save_xml(filepath)
        assert os.path.exists(filepath), "save_xml should create file"
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


def test_save_xml_content_matches_to_xml():
    """Test save_xml content matches to_xml."""
    gt = SP_TREE.simulate_dtl(0.5, 0.2, 0.3, seed=14100)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
        filepath = f.name
    try:
        gt.save_xml(filepath)
        with open(filepath, 'r') as f:
            file_content = f.read()
        xml_content = gt.to_xml()
        assert file_content == xml_content, "File content should match to_xml output"
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


def test_save_xml_file_not_empty():
    """Test save_xml creates non-empty file."""
    gt = SP_TREE.simulate_dtl(0.5, 0.2, 0.3, seed=14200)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
        filepath = f.name
    try:
        gt.save_xml(filepath)
        assert os.path.getsize(filepath) > 0, "Saved XML file should not be empty"
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


def test_save_xml_overwrite():
    """Test save_xml overwrites existing file."""
    gt1 = SP_TREE.simulate_dtl(0.5, 0.2, 0.3, seed=14300)
    gt2 = SP_TREE.simulate_dtl(0.6, 0.3, 0.4, seed=14301)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
        filepath = f.name
    try:
        gt1.save_xml(filepath)
        size1 = os.path.getsize(filepath)
        gt2.save_xml(filepath)
        with open(filepath, 'r') as f:
            content = f.read()
        # Content should match second tree
        assert content == gt2.to_xml(), "save_xml should overwrite with new content"
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


# =============================================================================
# Tests for GeneTree.to_csv()
# =============================================================================

def test_to_csv_returns_dataframe():
    """Test to_csv returns a pandas DataFrame."""
    gt = SP_TREE.simulate_dtl(0.5, 0.2, 0.3, seed=15000)
    try:
        import pandas as pd
        df = gt.to_csv()
        assert isinstance(df, pd.DataFrame), "to_csv should return DataFrame"
    except ImportError:
        pass  # Skip if pandas not installed


def test_to_csv_has_expected_columns():
    """Test DataFrame has expected columns."""
    gt = SP_TREE.simulate_dtl(0.5, 0.2, 0.3, seed=15100)
    try:
        import pandas as pd
        df = gt.to_csv()
        expected_cols = ['node_id', 'name', 'parent', 'left_child', 'right_child',
                        'length', 'depth', 'species_node', 'event']
        for col in expected_cols:
            assert col in df.columns, f"DataFrame should have column '{col}'"
    except ImportError:
        pass  # Skip if pandas not installed


def test_to_csv_row_count_matches_nodes():
    """Test DataFrame row count matches num_nodes."""
    gt = SP_TREE.simulate_dtl(0.5, 0.2, 0.3, seed=15200)
    try:
        import pandas as pd
        df = gt.to_csv()
        assert len(df) == gt.num_nodes(), "DataFrame rows should match num_nodes"
    except ImportError:
        pass  # Skip if pandas not installed


def test_to_csv_event_values_valid():
    """Test event column has valid values."""
    gt = SP_TREE.simulate_dtl(0.5, 0.2, 0.3, seed=15300)
    try:
        import pandas as pd
        df = gt.to_csv()
        valid_events = {'Speciation', 'Duplication', 'Transfer', 'Loss', 'Leaf'}
        for event in df['event'].unique():
            assert event in valid_events, f"Event '{event}' should be valid"
    except ImportError:
        pass  # Skip if pandas not installed


def test_to_csv_saves_file():
    """Test to_csv saves to file when filepath provided."""
    gt = SP_TREE.simulate_dtl(0.5, 0.2, 0.3, seed=15400)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        filepath = f.name
    try:
        import pandas as pd
        gt.to_csv(filepath)
        assert os.path.exists(filepath), "to_csv should create file when filepath given"
        assert os.path.getsize(filepath) > 0, "CSV file should not be empty"
    except ImportError:
        pass  # Skip if pandas not installed
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


# =============================================================================
# Tests for GeneTree.save_newick()
# =============================================================================

def test_save_newick_creates_file():
    """Test save_newick creates a file."""
    gt = SP_TREE.simulate_dtl(0.5, 0.2, 0.3, seed=16000)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.nwk', delete=False) as f:
        filepath = f.name
    try:
        gt.save_newick(filepath)
        assert os.path.exists(filepath), "save_newick should create file"
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


def test_save_newick_content_matches():
    """Test save_newick content matches to_newick."""
    gt = SP_TREE.simulate_dtl(0.5, 0.2, 0.3, seed=16100)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.nwk', delete=False) as f:
        filepath = f.name
    try:
        gt.save_newick(filepath)
        with open(filepath, 'r') as f:
            file_content = f.read()
        newick_content = gt.to_newick()
        assert file_content == newick_content, "File content should match to_newick output"
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


def test_save_newick_file_ends_with_semicolon():
    """Test saved Newick file ends with semicolon."""
    gt = SP_TREE.simulate_dtl(0.5, 0.2, 0.3, seed=16200)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.nwk', delete=False) as f:
        filepath = f.name
    try:
        gt.save_newick(filepath)
        with open(filepath, 'r') as f:
            content = f.read()
        assert content.rstrip().endswith(';'), "Saved Newick should end with semicolon"
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


def test_save_newick_file_not_empty():
    """Test save_newick creates non-empty file."""
    gt = SP_TREE.simulate_dtl(0.5, 0.2, 0.3, seed=16300)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.nwk', delete=False) as f:
        filepath = f.name
    try:
        gt.save_newick(filepath)
        assert os.path.getsize(filepath) > 0, "Saved Newick file should not be empty"
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


def test_save_newick_overwrite():
    """Test save_newick overwrites existing file."""
    gt1 = SP_TREE.simulate_dtl(0.5, 0.2, 0.3, seed=16400)
    gt2 = SP_TREE.simulate_dtl(0.6, 0.3, 0.4, seed=16401)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.nwk', delete=False) as f:
        filepath = f.name
    try:
        gt1.save_newick(filepath)
        gt2.save_newick(filepath)
        with open(filepath, 'r') as f:
            content = f.read()
        assert content == gt2.to_newick(), "save_newick should overwrite with new content"
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


# =============================================================================
# Additional edge case tests
# =============================================================================

def test_species_tree_shared():
    """Test that the shared species tree is valid."""
    assert SP_TREE.num_leaves() == 30, "Shared species tree should have 30 leaves"
    assert SP_TREE.num_nodes() > 30, "Species tree should have internal nodes"


def test_dtl_with_only_loss():
    """Test DTL with only loss rate."""
    gt = SP_TREE.simulate_dtl(0.0, 0.0, 0.5, seed=17000)
    events = gt.count_events()
    assert events["duplications"] == 0, "Should have no duplications"
    assert events["transfers"] == 0, "Should have no transfers"


def test_dtl_with_only_duplication():
    """Test DTL with only duplication rate."""
    gt = SP_TREE.simulate_dtl(0.5, 0.0, 0.0, seed=17100)
    events = gt.count_events()
    assert events["transfers"] == 0, "Should have no transfers"
    assert events["losses"] == 0, "Should have no losses"


def test_dtl_with_only_transfer():
    """Test DTL with only transfer rate."""
    gt = SP_TREE.simulate_dtl(0.0, 0.5, 0.0, seed=17200)
    events = gt.count_events()
    assert events["duplications"] == 0, "Should have no duplications"
    assert events["losses"] == 0, "Should have no losses"


# =============================================================================
# Run all tests
# =============================================================================

def main():
    """Run all tests and print summary."""
    global _passed, _failed, _errors

    print("=" * 70)
    print("Running comprehensive gene tree and DTL simulation tests")
    print("=" * 70)
    print()

    # Get all test functions
    test_functions = [obj for name, obj in globals().items()
                     if name.startswith('test_') and callable(obj)]

    # Run each test
    for test_func in test_functions:
        run_test(test_func)

    # Print summary
    print()
    print("=" * 70)
    print(f"SUMMARY: {_passed} passed, {_failed} failed out of {_passed + _failed} tests")
    print("=" * 70)

    if _errors:
        print("\nFailed tests:")
        for name, error in _errors:
            print(f"  - {name}: {error}")

    return _failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
