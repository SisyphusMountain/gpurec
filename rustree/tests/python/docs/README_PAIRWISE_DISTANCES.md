# Pairwise Distance Tests

This directory contains comprehensive tests for pairwise distance functionality in rustree.

## Current Status

The pairwise distance functions (`pairwise_distances` and `save_pairwise_distances_csv`) are:
- ✅ **Implemented in Rust** (`src/metric_functions.rs`)
- ✅ **Exposed in R bindings** (`src/r.rs`)
- ✅ **Exposed in Python bindings** (`src/python.rs`)

Both `pairwise_distances(distance_type, leaves_only)` and `save_pairwise_distances_csv(filepath, distance_type, leaves_only)` are fully implemented and available in the Python API.

## Test Coverage

The test file `test_pairwise_distances.py` provides comprehensive coverage including:

### Basic Functionality Tests
- ✅ DataFrame structure and column validation
- ✅ Self-distance verification (should be 0)
- ✅ Symmetry checks (distance(A,B) == distance(B,A))
- ✅ Specific distance value verification on known trees

### Distance Type Tests
- ✅ Topological distances (edge count)
  - Correct edge counting between nodes
  - Integer values
  - All non-negative
- ✅ Metric distances (branch length sum)
  - Correct summation of branch lengths
  - Handles varying branch lengths
  - All non-negative
  - Generally >= topological distances

### Parameter Tests
- ✅ `leaves_only=True` (only leaf nodes)
- ✅ `leaves_only=False` (all nodes including internal)
- ✅ Comparison between the two modes

### CSV Export Tests
- ✅ File creation
- ✅ Content matches direct method output
- ✅ Proper CSV header
- ✅ File overwrite behavior
- ✅ Both distance types (topological and metric)

### Edge Cases
- ✅ Single node tree
- ✅ Two node tree
- ✅ Large trees (20+ leaves)
- ✅ Trees with varying branch lengths

### Error Handling
- ✅ Invalid distance type
- ✅ Invalid file path
- ✅ Case sensitivity checks

## Running the Tests

```bash
# Run all tests
pytest rustree/tests/python/test_pairwise_distances.py -v

# Run specific test
pytest rustree/tests/python/test_pairwise_distances.py::test_pairwise_distances_metric_basic -v

# Run with coverage
pytest rustree/tests/python/test_pairwise_distances.py --cov=rustree --cov-report=html
```

## Example Usage

```python
import rustree

# Create or load a tree
tree = rustree.simulate_species_tree(10, 1.0, 0.3, seed=42)

# Compute metric distances between leaves
distances = tree.pairwise_distances("metric", leaves_only=True)
print(distances.head())

# Output:
#   node1 node2  distance
# 0    S0    S0      0.00
# 1    S0    S1      2.45
# 2    S0    S2      3.21
# ...

# Compute topological distances (edge count)
topo_distances = tree.pairwise_distances("topological", leaves_only=True)

# Include internal nodes
all_distances = tree.pairwise_distances("metric", leaves_only=False)

# Save to CSV
tree.save_pairwise_distances_csv("distances.csv", "metric", leaves_only=True)
```

## Integration with Existing Code

The pairwise distance functionality integrates with existing rustree features:

- Works with both simulated trees (`simulate_species_tree`) and parsed trees (`parse_species_tree`)
- Complements other tree metrics like `tree_height()`, `num_leaves()`, etc.
- CSV export follows similar patterns to gene tree CSV export
- Uses the same `FlatTree` structure as other operations

## Performance Notes

- Complexity: O(n²) for n nodes (unavoidable for pairwise distances)
- For large trees with many nodes, use `leaves_only=True` to reduce computation
- Distance computation uses efficient LCA-based algorithms
- CSV writing is buffered for efficiency
