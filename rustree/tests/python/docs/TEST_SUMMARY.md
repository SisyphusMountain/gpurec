# Tree Sampling Test Suite - Summary

## Overview

Created comprehensive pytest tests for the tree sampling functionality, specifically testing `extract_induced_subtree_by_names()` which is exposed as `sample_by_names()` in the Python API.

## Files Created

1. **test_tree_sampling.py** (697 lines)
   - Comprehensive test suite with 38 test functions
   - Uses pytest fixtures for efficient test setup
   - Covers all requirements specified in the task

2. **README_TREE_SAMPLING_TESTS.md**
   - Documentation for running and understanding the tests
   - Lists all test categories and individual tests
   - Provides examples and usage instructions

## Test Statistics

- **Total test functions**: 38
- **Total lines of code**: 697
- **Test categories**: 9

## Test Coverage by Category

### 1. Basic Functionality (6 tests)
- Valid subset sampling
- Return type verification
- Single leaf handling
- Two leaves handling
- All leaves sampling
- Half subset sampling

### 2. Tree Structure Verification (4 tests)
- Verify only specified leaves present
- Confirm no loss events
- Validate binary tree structure
- Check Newick string validity

### 3. Different Leaf Subsets (4 tests)
- Various subset sizes
- First vs. last leaves
- Scattered non-consecutive leaves
- Overlapping subsets

### 4. Error Handling (7 tests)
- Empty list → ValueError
- Non-existent name → ValueError
- Multiple invalid names → ValueError
- Mixed valid/invalid → ValueError
- Duplicate names handling
- None parameter → TypeError/ValueError
- Non-list parameter → TypeError/ValueError

### 5. Edge Cases (4 tests)
- Single leaf from large tree
- All but one leaf
- Tree with single extant gene
- Order independence verification

### 6. Topology Preservation (5 tests)
- Single leaf minimal structure
- Two-leaf MRCA presence
- Complete tree preservation
- Binary structure maintenance
- Newick parseability

### 7. Reproducibility & Consistency (2 tests)
- Deterministic sampling results
- Consistent event counts

### 8. Integration Tests (3 tests)
- Nested sampling (sample from sampled tree)
- Multiple independent samples
- Complete workflow (sample + save + export)

### 9. Performance & Stress Tests (2 tests)
- Large subset from very large tree
- Many sequential small samples

## Key Features

### Pytest Fixtures
```python
@pytest.fixture
def simple_species_tree():
    return rustree.simulate_species_tree(5, 1.0, 0.3, seed=42)

@pytest.fixture
def simple_gene_tree(simple_species_tree):
    return simple_species_tree.simulate_dtl(0.5, 0.2, 0.1, require_extant=True, seed=42)
```

### Graceful Skip Handling
Tests automatically skip when preconditions aren't met:
```python
if len(all_names) < 2:
    pytest.skip("Not enough extant genes")
```

### Comprehensive Assertions
Each test includes descriptive assertion messages:
```python
assert sampled.num_extant() == len(subset), "Sampled tree should have exact number of specified genes"
```

## Test Examples

### Basic Usage Test
```python
def test_sample_by_names_valid_subset(simple_gene_tree):
    """Test extract_induced_subtree with valid leaf names."""
    all_names = simple_gene_tree.extant_gene_names()
    subset = all_names[:2]
    sampled = simple_gene_tree.sample_by_names(subset)

    assert sampled.num_extant() == len(subset)
    assert set(sampled.extant_gene_names()) == set(subset)
```

### Error Handling Test
```python
def test_sample_empty_list_raises_error(simple_gene_tree):
    """Test that sampling with empty name list raises ValueError."""
    with pytest.raises(ValueError):
        simple_gene_tree.sample_by_names([])
```

### Topology Verification Test
```python
def test_sampled_tree_no_loss_events(simple_gene_tree):
    """Verify sampled tree has no loss events (only extant genes)."""
    all_names = simple_gene_tree.extant_gene_names()
    subset = all_names[:2]
    sampled = simple_gene_tree.sample_by_names(subset)

    spec, dup, transfer, loss, leaf = sampled.count_events()
    assert loss == 0, "Sampled tree should not contain loss events"
```

## Running the Tests

### Run all tests:
```bash
pytest tests/python/test_tree_sampling.py -v
```

### Run specific category:
```bash
# Error handling tests
pytest tests/python/test_tree_sampling.py -k "error" -v

# Topology tests
pytest tests/python/test_tree_sampling.py -k "topology" -v

# Edge case tests
pytest tests/python/test_tree_sampling.py -k "edge" -v
```

### Run with coverage:
```bash
pytest tests/python/test_tree_sampling.py --cov=rustree --cov-report=html
```

## Alignment with Existing Tests

The test suite follows the same patterns as existing test files:

1. **test_species_tree.py** (863 lines, 77 tests)
   - Species tree simulation and parsing
   - Newick round-trip testing
   - Property verification

2. **test_gene_tree.py** (820 lines, 70+ tests)
   - DTL simulation
   - Gene tree properties
   - XML/CSV export

3. **test_tree_sampling.py** (697 lines, 38 tests) ← NEW
   - Tree sampling/induced subtrees
   - Topology preservation
   - Comprehensive error handling

## Test Quality Metrics

✓ **Comprehensive**: Covers all specified requirements
✓ **Well-documented**: Docstrings for every test
✓ **Robust**: Graceful handling of edge cases via pytest.skip
✓ **Maintainable**: Clear structure and organization
✓ **Reusable**: Fixtures for common setup
✓ **Follows patterns**: Consistent with existing test files

## Requirements Coverage

All requirements from the task are covered:

1. ✅ extract_induced_subtree_by_names() with valid leaf names
2. ✅ Verify resulting tree has only specified leaves
3. ✅ Test with different subsets of leaves
4. ✅ Test error handling (empty names, non-existent names)
5. ✅ Verify tree topology is correct after sampling
6. ✅ Test edge cases (single leaf, all leaves, etc.)
7. ✅ Use pytest
8. ✅ Follow existing test patterns

## Additional Value

Beyond the requirements, the test suite also includes:

- Reproducibility tests
- Integration tests (nested sampling)
- Performance/stress tests
- Workflow tests (save/export)
- Binary tree structure validation
- Newick format verification
- Event count consistency checks
