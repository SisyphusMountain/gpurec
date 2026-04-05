# Quick Start Guide - Tree Sampling Tests

## Prerequisites

Ensure rustree is compiled:
```bash
cd /path/to/rustree
cargo build --release
```

## Running Tests

### 1. Run All Tree Sampling Tests
```bash
cd /path/to/rustree
python -m pytest tests/python/test_tree_sampling.py -v
```

Expected output:
```
tests/python/test_tree_sampling.py::test_sample_by_names_valid_subset PASSED
tests/python/test_tree_sampling.py::test_sample_by_names_returns_gene_tree PASSED
tests/python/test_tree_sampling.py::test_sample_by_names_single_leaf PASSED
...
```

### 2. Run Specific Test Categories

#### Basic Functionality Tests
```bash
pytest tests/python/test_tree_sampling.py::test_sample_by_names_valid_subset -v
pytest tests/python/test_tree_sampling.py::test_sample_by_names_single_leaf -v
pytest tests/python/test_tree_sampling.py::test_sample_by_names_all_leaves -v
```

#### Error Handling Tests
```bash
pytest tests/python/test_tree_sampling.py -k "error" -v
```

This runs:
- test_sample_empty_list_raises_error
- test_sample_nonexistent_name_raises_error
- test_sample_multiple_nonexistent_names_raises_error
- test_sample_mixed_valid_invalid_names
- etc.

#### Topology Tests
```bash
pytest tests/python/test_tree_sampling.py -k "topology" -v
```

This runs:
- test_topology_single_leaf_has_no_structure
- test_topology_two_leaves_has_parent
- test_topology_complete_tree_unchanged
- test_topology_binary_tree_structure
- test_topology_newick_parseable

#### Edge Case Tests
```bash
pytest tests/python/test_tree_sampling.py -k "edge" -v
```

### 3. Run With Detailed Output

Show print statements and full error messages:
```bash
pytest tests/python/test_tree_sampling.py -v -s
```

### 4. Run Specific Test by Name
```bash
pytest tests/python/test_tree_sampling.py::test_sample_by_names_valid_subset -v
```

### 5. Run Multiple Specific Tests
```bash
pytest tests/python/test_tree_sampling.py::test_sample_by_names_single_leaf \
       tests/python/test_tree_sampling.py::test_sample_empty_list_raises_error -v
```

## Example Test Run

```bash
$ pytest tests/python/test_tree_sampling.py::test_sample_by_names_valid_subset -v

========================= test session starts ==========================
platform linux -- Python 3.10.12, pytest-7.4.0
collected 1 item

tests/python/test_tree_sampling.py::test_sample_by_names_valid_subset PASSED [100%]

========================== 1 passed in 0.15s ===========================
```

## Interactive Testing

You can also test the functionality interactively:

```python
import sys
sys.path.insert(0, "target/release")
import rustree

# Create a species tree
sp_tree = rustree.simulate_species_tree(10, 1.0, 0.3, seed=42)

# Simulate a gene tree
gene_tree = sp_tree.simulate_dtl(0.5, 0.2, 0.1, require_extant=True, seed=42)

# Get extant gene names
all_names = gene_tree.extant_gene_names()
print(f"Extant genes: {all_names}")

# Sample a subset
subset = all_names[:3]
sampled = gene_tree.sample_by_names(subset)

# Verify the sampling
print(f"Original: {len(all_names)} genes")
print(f"Sampled: {sampled.num_extant()} genes")
print(f"Sampled genes: {sampled.extant_gene_names()}")
print(f"Newick: {sampled.to_newick()}")
```

## Troubleshooting

### Import Error
If you get `ModuleNotFoundError: No module named 'rustree'`:
```bash
# Make sure rustree is compiled
cd /path/to/rustree
cargo build --release

# Check the compiled library exists
ls target/release/rustree*.so
```

### pytest Not Found
```bash
pip install pytest
```

### Tests Skip Due to Not Enough Genes
Some tests may skip if the randomly generated gene tree doesn't have enough extant genes. This is expected behavior. The tests use `pytest.skip()` to handle this gracefully. Try running the full suite - most tests should pass.

## Understanding Test Results

### PASSED
✅ Test passed successfully

### SKIPPED
⏭️ Test was skipped (usually because preconditions weren't met, e.g., not enough extant genes)

### FAILED
❌ Test failed - check the error message

### Example Output with Skips
```
test_tree_sampling.py::test_sample_by_names_valid_subset PASSED
test_tree_sampling.py::test_sample_subset_half SKIPPED (not enough genes)
test_tree_sampling.py::test_sample_empty_list_raises_error PASSED
```

## Running All Test Files Together

To run all Python tests (species tree, gene tree, and sampling):
```bash
pytest tests/python/ -v
```

This will run:
- test_species_tree.py (77 tests)
- test_gene_tree.py (70+ tests)
- test_tree_sampling.py (38 tests)

Total: ~185 tests

## Next Steps

1. Review the test output
2. Check coverage with: `pytest tests/python/test_tree_sampling.py --cov=rustree`
3. Add custom tests for your specific use cases
4. Integrate into CI/CD pipeline

## Common Test Patterns

### Test Valid Input
```python
def test_sample_by_names_valid_subset(simple_gene_tree):
    all_names = simple_gene_tree.extant_gene_names()
    if len(all_names) < 2:
        pytest.skip("Not enough extant genes")

    subset = all_names[:2]
    sampled = simple_gene_tree.sample_by_names(subset)
    assert sampled.num_extant() == len(subset)
```

### Test Error Cases
```python
def test_sample_empty_list_raises_error(simple_gene_tree):
    with pytest.raises(ValueError):
        simple_gene_tree.sample_by_names([])
```

### Test Topology
```python
def test_sampled_tree_no_loss_events(simple_gene_tree):
    all_names = simple_gene_tree.extant_gene_names()
    if len(all_names) < 2:
        pytest.skip("Not enough extant genes")

    sampled = simple_gene_tree.sample_by_names(all_names[:2])
    spec, dup, transfer, loss, leaf = sampled.count_events()
    assert loss == 0
```
