# Tree Sampling Tests

This directory contains comprehensive tests for the tree sampling functionality in rustree.

## Test File: `test_tree_sampling.py`

Comprehensive tests for the `extract_induced_subtree_by_names()` function (exposed as `sample_by_names()` in Python).

### Test Coverage

#### 1. Basic Functionality Tests
- `test_sample_by_names_valid_subset`: Test with valid leaf names
- `test_sample_by_names_returns_gene_tree`: Verify return type
- `test_sample_by_names_single_leaf`: Edge case with single leaf
- `test_sample_by_names_two_leaves`: Test with exactly two leaves
- `test_sample_by_names_all_leaves`: Sample all extant leaves
- `test_sample_by_names_subset_half`: Sample approximately half the leaves

#### 2. Tree Structure Verification
- `test_sampled_tree_has_only_specified_leaves`: Verify exact leaf set
- `test_sampled_tree_no_loss_events`: Confirm no loss events in sampled tree
- `test_sampled_tree_valid_structure`: Validate binary tree structure
- `test_sampled_tree_valid_newick`: Check Newick string validity

#### 3. Different Leaf Subsets
- `test_sample_different_subsets`: Various subset sizes
- `test_sample_first_vs_last_leaves`: Different positions in leaf list
- `test_sample_random_scattered_leaves`: Non-consecutive leaves
- `test_sample_overlapping_subsets`: Overlapping subset handling

#### 4. Error Handling
- `test_sample_empty_list_raises_error`: Empty name list
- `test_sample_nonexistent_name_raises_error`: Invalid gene name
- `test_sample_multiple_nonexistent_names_raises_error`: Multiple invalid names
- `test_sample_mixed_valid_invalid_names`: Mix of valid/invalid names
- `test_sample_duplicate_names_in_list`: Duplicate gene names
- `test_sample_none_as_parameter`: None parameter handling
- `test_sample_non_list_parameter`: Type checking

#### 5. Edge Cases
- `test_sample_single_leaf_from_large_tree`: Single leaf from many
- `test_sample_all_but_one`: All leaves except one
- `test_sample_tree_with_single_extant_gene`: Minimal tree
- `test_sample_preserves_gene_order`: Order independence

#### 6. Topology Preservation
- `test_topology_single_leaf_has_no_structure`: Single leaf structure
- `test_topology_two_leaves_has_parent`: Two-leaf MRCA
- `test_topology_complete_tree_unchanged`: Full sampling preservation
- `test_topology_binary_tree_structure`: Binary structure maintenance
- `test_topology_newick_parseable`: Newick string parseability

#### 7. Reproducibility and Consistency
- `test_sample_reproducible`: Deterministic results
- `test_sample_consistent_event_counts`: Event count stability

#### 8. Integration Tests
- `test_sample_then_resample`: Nested sampling
- `test_sample_multiple_different_subsets_from_same_tree`: Multiple independent samples
- `test_sample_workflow_save_and_export`: Complete workflow

#### 9. Performance Tests
- `test_sample_large_subset_from_very_large_tree`: Large-scale sampling
- `test_sample_many_small_subsets`: Sequential small samples

## Running the Tests

### Run all tree sampling tests:
```bash
cd /path/to/rustree
python -m pytest tests/python/test_tree_sampling.py -v
```

### Run a specific test:
```bash
python -m pytest tests/python/test_tree_sampling.py::test_sample_by_names_valid_subset -v
```

### Run with coverage:
```bash
python -m pytest tests/python/test_tree_sampling.py --cov=rustree --cov-report=html
```

### Run only error handling tests:
```bash
python -m pytest tests/python/test_tree_sampling.py -k "error" -v
```

### Run only topology tests:
```bash
python -m pytest tests/python/test_tree_sampling.py -k "topology" -v
```

## Test Requirements

- Python 3.7+
- pytest
- rustree compiled in release mode (`target/release/`)

## Test Patterns

The tests follow these patterns:

1. **Fixtures**: Reusable test data created via pytest fixtures
2. **Skip conditions**: Tests gracefully skip when conditions aren't met (e.g., not enough genes)
3. **Clear assertions**: Each test has descriptive assertion messages
4. **Edge case handling**: Comprehensive coverage of boundary conditions
5. **Error validation**: Proper exception type and message checking

## Example Test Output

```
test_tree_sampling.py::test_sample_by_names_valid_subset PASSED
test_tree_sampling.py::test_sample_by_names_single_leaf PASSED
test_tree_sampling.py::test_sample_empty_list_raises_error PASSED
test_tree_sampling.py::test_topology_binary_tree_structure PASSED
...
```

## Related Files

- `test_species_tree.py`: Tests for species tree simulation
- `test_gene_tree.py`: Tests for gene tree simulation and DTL
- `../src/sampling.rs`: Rust implementation of sampling functions
- `../src/python.rs`: Python bindings for sampling functions
