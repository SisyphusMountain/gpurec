# Pairwise Distance Tests - Validation Checklist

Use this checklist to verify that the pairwise distance implementation is complete and correct.

## Pre-Implementation Checklist

All pre-implementation requirements have been met:

- [x] `src/python.rs` has the required imports
- [x] `PySpeciesTree` has the `pairwise_distances()` method
- [x] `PySpeciesTree` has the `save_pairwise_distances_csv()` method
- [x] Code compiles without errors: `cargo build --release`
- [x] Python module is rebuilt and accessible

## Basic Functionality Tests

### pairwise_distances() Method

- [ ] Method exists and is callable
  ```python
  tree = rustree.simulate_species_tree(5, 1.0, 0.3, seed=42)
  df = tree.pairwise_distances("metric", leaves_only=True)
  ```

- [ ] Returns a pandas DataFrame
  ```python
  import pandas as pd
  assert isinstance(df, pd.DataFrame)
  ```

- [ ] DataFrame has correct columns
  ```python
  assert list(df.columns) == ["node1", "node2", "distance"]
  ```

- [ ] Works with "metric" distance type
  ```python
  df = tree.pairwise_distances("metric", leaves_only=True)
  assert len(df) > 0
  ```

- [ ] Works with "topological" distance type
  ```python
  df = tree.pairwise_distances("topological", leaves_only=True)
  assert len(df) > 0
  ```

- [ ] Works with leaves_only=True
  ```python
  df = tree.pairwise_distances("metric", leaves_only=True)
  assert len(df) == tree.num_leaves() ** 2
  ```

- [ ] Works with leaves_only=False
  ```python
  df = tree.pairwise_distances("metric", leaves_only=False)
  assert len(df) == tree.num_nodes() ** 2
  ```

### save_pairwise_distances_csv() Method

- [ ] Method exists and is callable
  ```python
  tree.save_pairwise_distances_csv("test.csv", "metric", leaves_only=True)
  ```

- [ ] Creates a file
  ```python
  import os
  assert os.path.exists("test.csv")
  ```

- [ ] File is not empty
  ```python
  assert os.path.getsize("test.csv") > 0
  ```

- [ ] File has correct header
  ```python
  with open("test.csv", 'r') as f:
      header = f.readline().strip()
  assert header == "node1,node2,distance"
  ```

- [ ] File content matches pairwise_distances() output
  ```python
  df_direct = tree.pairwise_distances("metric", leaves_only=True)
  df_from_csv = pd.read_csv("test.csv")
  # Should be identical (after sorting)
  ```

## Correctness Tests

### Distance Values

- [ ] Self-distances are 0
  ```python
  df = tree.pairwise_distances("metric", leaves_only=True)
  self_dist = df[df["node1"] == df["node2"]]
  assert all(self_dist["distance"] == 0.0)
  ```

- [ ] Distances are symmetric
  ```python
  # distance(A,B) == distance(B,A)
  for _, row in df.iterrows():
      node1, node2, dist = row["node1"], row["node2"], row["distance"]
      reverse = df[(df["node1"] == node2) & (df["node2"] == node1)]
      if not reverse.empty:
          assert abs(reverse.iloc[0]["distance"] - dist) < 1e-10
  ```

- [ ] All distances are non-negative
  ```python
  assert all(df["distance"] >= 0)
  ```

- [ ] Topological distances are integers
  ```python
  df_topo = tree.pairwise_distances("topological", leaves_only=True)
  assert all(df_topo["distance"] % 1 == 0)
  ```

- [ ] Metric distances >= topological distances
  ```python
  df_metric = tree.pairwise_distances("metric", leaves_only=True)
  df_topo = tree.pairwise_distances("topological", leaves_only=True)
  # Compare same pairs
  ```

### Specific Known Values

For tree: `((A:1.0,B:1.0):1.0,C:2.0):0.0`

- [ ] Metric distance A-B is 2.0
  ```python
  tree = rustree.parse_species_tree("((A:1.0,B:1.0):1.0,C:2.0):0.0;")
  df = tree.pairwise_distances("metric", leaves_only=True)
  ab = df[(df["node1"] == "A") & (df["node2"] == "B")]["distance"].values[0]
  assert abs(ab - 2.0) < 1e-10
  ```

- [ ] Metric distance A-C is 4.0
  ```python
  ac = df[(df["node1"] == "A") & (df["node2"] == "C")]["distance"].values[0]
  assert abs(ac - 4.0) < 1e-10
  ```

- [ ] Topological distance A-B is 2
  ```python
  df_topo = tree.pairwise_distances("topological", leaves_only=True)
  ab_topo = df_topo[(df_topo["node1"] == "A") & (df_topo["node2"] == "B")]["distance"].values[0]
  assert ab_topo == 2.0
  ```

## Edge Cases

- [ ] Single-node tree
  ```python
  tree = rustree.parse_species_tree("A:0.0;")
  df = tree.pairwise_distances("metric", leaves_only=True)
  assert len(df) == 1
  assert df.iloc[0]["distance"] == 0.0
  ```

- [ ] Two-node tree
  ```python
  tree = rustree.parse_species_tree("(A:1.0,B:1.0):0.0;")
  df = tree.pairwise_distances("metric", leaves_only=True)
  assert len(df) == 4  # 2x2
  ```

- [ ] Large tree (100+ leaves)
  ```python
  tree = rustree.simulate_species_tree(100, 1.0, 0.3, seed=42)
  df = tree.pairwise_distances("metric", leaves_only=True)
  assert len(df) == 100 * 100
  ```

## Error Handling

- [ ] Invalid distance type raises ValueError
  ```python
  import pytest
  with pytest.raises(ValueError):
      tree.pairwise_distances("invalid", leaves_only=True)
  ```

- [ ] Invalid file path raises error
  ```python
  with pytest.raises((ValueError, OSError, IOError)):
      tree.save_pairwise_distances_csv("/invalid/path.csv", "metric", leaves_only=True)
  ```

- [ ] Case sensitivity (depends on implementation)
  ```python
  # Test if "METRIC" or "Metric" work
  # Or verify they raise ValueError if case-sensitive
  ```

## Performance Tests

- [ ] Large tree completes in reasonable time
  ```python
  import time
  tree = rustree.simulate_species_tree(50, 1.0, 0.3, seed=42)
  start = time.time()
  df = tree.pairwise_distances("metric", leaves_only=True)
  elapsed = time.time() - start
  assert elapsed < 5.0  # Should complete in under 5 seconds
  ```

- [ ] CSV export completes
  ```python
  tree.save_pairwise_distances_csv("large_test.csv", "metric", leaves_only=True)
  assert os.path.exists("large_test.csv")
  ```

## Integration Tests

- [ ] Works with simulated trees
  ```python
  tree = rustree.simulate_species_tree(10, 1.0, 0.3, seed=42)
  df = tree.pairwise_distances("metric", leaves_only=True)
  assert len(df) == 100
  ```

- [ ] Works with parsed trees
  ```python
  tree = rustree.parse_species_tree("((A:1,B:1):1,C:2):0;")
  df = tree.pairwise_distances("metric", leaves_only=True)
  assert len(df) == 9
  ```

- [ ] Round-trip: save and load CSV
  ```python
  tree.save_pairwise_distances_csv("test.csv", "metric", leaves_only=True)
  df_saved = pd.read_csv("test.csv")
  df_direct = tree.pairwise_distances("metric", leaves_only=True)
  # Compare after sorting
  ```

## Full Test Suite

- [ ] Run pytest suite
  ```bash
  pytest tests/python/test_pairwise_distances.py -v
  ```

- [ ] All tests pass (35 tests)
  ```
  ======================= 35 passed in X.XXs =======================
  ```

- [ ] Run demo script
  ```bash
  python tests/python/demo_pairwise_distances.py
  ```

- [ ] Demo completes without AttributeError

## Documentation

- [ ] Python docstrings are present
- [ ] Method signatures are correct
- [ ] Examples in docstrings work
- [ ] README is updated (if applicable)

## Final Verification

- [ ] Code is properly formatted
- [ ] No compiler warnings
- [ ] No runtime warnings
- [ ] Memory usage is reasonable
- [ ] Works on example use cases
- [ ] Integrated with existing rustree API

## Sign-off

Implementation completed by: _______________

Date: _______________

All tests passing: [ ] Yes [ ] No

Ready for merge/release: [ ] Yes [ ] No

Notes:
_________________________________________________________________
_________________________________________________________________
_________________________________________________________________
