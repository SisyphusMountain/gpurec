# Python Bindings Gap Analysis

## Executive Summary

The rustree Python bindings provide **excellent coverage** for simulation and reconciliation workflows (~100% DTL, ~80% BD simulation), but **limited access** to advanced tree analysis, manipulation, and low-level operations (~0-33% coverage in several modules).

**Overall API Coverage**: Approximately **40% of Rust functions** are exposed to Python.

---

## What IS Available in Python

### Core Classes (PyO3 Bindings)

1. **PySpeciesTree** - Birth-death species tree with full simulation/export capabilities
2. **PyGeneTree** - DTL-reconciled gene tree with event information
3. **PyGeneForest** - Collection of gene trees sharing a species tree
4. **PyDtlSimIter** - Lazy iterator for streaming DTL simulations
5. **PyEventCounts** - Event count statistics
6. **PyReconciliationStatistics** - Reconciliation analysis with DataFrames
7. **PyAleRaxResult** - Single-family ALERax reconciliation result
8. **PyAleRaxForestResult** - Multi-family ALERax reconciliation with aggregates

### Well-Covered Functionality

#### ✓ DTL Simulation (100% coverage)
```python
# All simulation modes available
gt = species_tree.simulate_dtl(lambda_d=0.5, lambda_t=0.2, lambda_l=0.3, seed=42)
gts = species_tree.simulate_dtl_batch(n=100, lambda_d=0.5, ...)
gt = species_tree.simulate_dtl_per_species(lambda_d=0.5, ...)
gts = species_tree.simulate_dtl_per_species_batch(n=100, ...)

# Streaming iterators
iter = species_tree.simulate_dtl_iter(lambda_d=0.5, ..., max_trees=1000)
gt = next(iter.single())
gts = iter.collect_all()
```

#### ✓ Birth-Death Simulation (80% coverage)
```python
# Tree generation
tree = rustree.simulate_species_tree(n=100, lambda_=1.0, mu=0.5, seed=42)

# Event access
events = tree.get_bd_events()  # Returns dict
tree.save_bd_events_csv("events.csv")

# LTT plots
ltt = tree.get_ltt_data()
tree.plot_ltt(filepath="ltt.png")
```

#### ✓ Newick/RecPhyloXML I/O (70% coverage)
```python
# Parsing
tree = rustree.parse_species_tree("((A:1,B:1):1,C:2):0;")
gt = rustree.parse_recphyloxml("alerax_g.xml")

# Export
newick_str = tree.to_newick()
xml_str = gt.to_xml()
tree.save_newick("tree.nwk")
gt.save_xml("reconciled.xml")
gt.to_csv("events.csv")  # Returns DataFrame
```

#### ✓ ALERax Reconciliation (high-level complete)
```python
# Single-family reconciliation
result = rustree.reconcile_with_alerax(
    species_newick="species.nwk",
    gene_newick="gene.nwk",
    num_samples=100,
    model="PER-FAMILY"
)

# Multi-family reconciliation
forest = rustree.PyGeneForest(species_tree, gene_trees)
forest_result = forest.reconcile_with_alerax(num_samples=100, model="GLOBAL")

# Access results
df = result.statistics.events_df()
transfers_df = result.statistics.transfers_df()
```

#### ✓ Gene Tree Sampling (60% coverage)
```python
# Sample extant genes
sampled_gt = gene_tree.sample_extant(seed=42)

# Sample by names
sampled_gt = gene_tree.sample_by_names(["SpeciesA", "SpeciesB"])

# Extract induced subtree
subtree = species_tree.extract_induced_subtree_by_names(["A", "B", "C"])
```

#### ✓ Basic Tree Properties (50% coverage)
```python
# Leaf information
names = tree.leaf_names()
n_leaves = tree.num_leaves()
n_nodes = tree.num_nodes()

# Tree metrics
height = tree.tree_height()

# Pairwise distances (returns pandas DataFrame only)
df = tree.pairwise_distances(distance_type="metric", leaves_only=True)
tree.save_pairwise_distances_csv("distances.csv", distance_type="topological")
```

#### ✓ Induced Transfers (NEW - 100% coverage)
```python
# Compute induced transfers when analyzing sampled trees
induced = gene_tree.compute_induced_transfers(sampled_species_tree, sampled_leaf_names)

# Compute ghost branch lengths (hidden evolutionary time)
ghost_lengths = gene_tree.compute_ghost_lengths(sampled_species_tree, sampled_leaf_names)
```

---

## What is NOT Available in Python

### Critical Gaps (0% coverage - completely absent)

#### ✗ Tree Surgery / Modification Operations

**Missing from** `src/surgery.rs`:
```rust
// No Python equivalents exist for:
is_ancestor(tree, ancestor_idx, node_idx) -> bool
spr_topology(tree, prune_idx, regraft_idx) -> FlatTree
detach(node_idx) -> SubTree
attach(subtree, target_idx) -> FlatTree
```

**Impact**: Cannot perform Subtree Pruning and Regrafting (SPR) moves, cannot edit tree topology programmatically.

**Workaround**: Export to Newick, edit externally, re-import.

---

#### ✗ Tree Comparison

**Missing from** `src/comparison.rs`:
```rust
// No Python equivalents:
compare_nodes(tree1, tree2, use_lengths, tolerance) -> bool
compare_nodes_topology(tree1, tree2) -> bool
```

**Impact**: Cannot compare two trees for topological or metric equivalence.

**Workaround**: Export both trees to Newick, use external libraries like dendropy or ete3.

---

#### ✗ Tree Iteration with Custom Orders

**Missing from** `src/node/iter.rs`:
```rust
// No Python bindings for:
tree.iter(TraversalOrder::PreOrder)
tree.iter(TraversalOrder::InOrder)
tree.iter(TraversalOrder::PostOrder)
tree.iter_indices(order)
tree.postorder_indices()
```

**Impact**: Cannot traverse tree in specific orders, cannot access nodes by index.

**Workaround**: Export to Newick, use dendropy/ete3 traversal.

---

#### ✗ Node Index-Based Operations

**Limitation**: All Python methods use **names**, not **indices**.

**Missing capabilities**:
```rust
// Rust has:
find_lca(idx_a, idx_b) -> usize  // Returns index
distance_between(idx_a, idx_b, type) -> f64
find_leaf_indices_by_names(names) -> Vec<usize>
get_descendant_leaf_names(idx) -> Vec<String>

// Python only has:
tree.leaf_names()  // Returns list of names, not indices
```

**Impact**:
- Cannot work with node indices for efficient bulk operations
- Cannot precompute LCA tables for repeated queries
- Cannot map between trees using indices

**Workaround**: Work with leaf names only (slower for large operations).

---

### Partial Gaps (limited functionality)

#### △ Distance & LCA Analysis

**What Python has**:
```python
# Only DataFrame output, no direct matrix access
df = tree.pairwise_distances(distance_type="metric", leaves_only=True)
```

**What Rust has additionally**:
```rust
// Missing from Python:
LcaTable::new(tree) -> LcaTable  // O(n log n) precomputation
lca_table.lca(idx_a, idx_b) -> usize  // O(1) query
pairwise_distance_matrix(tree, type) -> Vec<Vec<f64>>  // Raw matrix
find_lca(a, b) -> Result<usize, String>
```

**Impact**:
- Cannot precompute LCA table for repeated O(1) queries
- Cannot access raw distance matrices (only DataFrames)
- Cannot perform bulk LCA queries efficiently

**Workaround**: Call `pairwise_distances()` which computes all pairs at once.

---

#### △ Tree Depth & Time Analysis

**What Rust has** (from `src/node/mod.rs` and `src/metric_functions.rs`):
```rust
// Not exposed to Python:
tree.assign_depths()
tree.make_subdivision() -> Vec<f64>
tree.make_intervals() -> Vec<f64>
find_contemporaneity(depths) -> HashMap<f64, Vec<usize>>
number_of_species(contemporaneity) -> Vec<usize>
```

**Impact**: Cannot analyze tree time slicing, cannot find contemporaneous species at specific depths.

**Workaround**: Use LTT plots (`tree.get_ltt_data()`) for time-based lineage analysis.

---

#### △ Newick & RecPhyloXML Parsing

**What Python has**:
```python
tree = rustree.parse_species_tree("((A:1,B:1):1,C:2):0;")  # File or string
gt = rustree.parse_recphyloxml("file.xml")  # File path only
```

**What Rust has additionally**:
```rust
// Not exposed:
parse_newick(str) -> Result<Node, String>  // Recursive Node structure
parse_gene_tree_only(xml_str) -> Result<RecTree, String>
parse_gene_tree_only_file(path) -> Result<RecTree, String>
```

**Impact**:
- Cannot parse RecPhyloXML from string (only file paths)
- Cannot access recursive Node representation
- Cannot parse gene-tree-only XML

**Workaround**: Write strings to temp files before parsing.

---

#### △ Event Structure Access

**What Python has**:
```python
events = tree.get_bd_events()  # Returns dict: {"speciations": [...], "extinctions": [...]}
tree.save_bd_events_csv("events.csv")
counts = gt.count_events()  # Returns (spec, dup, trans, loss, leaf) tuple
```

**What Rust has additionally**:
```rust
// Direct struct access not exposed:
pub struct TreeEvent { time: f64, event_type: EventType, ... }
pub struct BDEvent { ... }
pub struct DTLEvent { ... }

// Can access:
rec_tree.event_mapping[idx]  // Vec<Event>
```

**Impact**: Cannot access raw event vectors, cannot manipulate event structures directly.

**Workaround**: Use high-level methods (`count_events()`, `to_csv()`).

---

#### △ Sampling Operations

**What Python has**:
```python
sampled = gene_tree.sample_extant(seed=42)
sampled = gene_tree.sample_by_names(["A", "B", "C"])
subtree = species_tree.extract_induced_subtree_by_names(names)
```

**What Rust has additionally**:
```rust
// Index-based sampling (not exposed):
find_leaf_indices_by_names(tree, names) -> Vec<usize>
find_all_leaf_indices(tree) -> Vec<usize>
extract_induced_subtree(tree, leaf_indices) -> FlatTree
compute_lca(a, b) -> usize
build_leaf_pair_lca_map(tree) -> HashMap<(usize, usize), usize>
build_sampled_to_original_mapping(...) -> HashMap<usize, usize>
```

**Impact**: Cannot use index-based sampling for efficiency, cannot precompute LCA maps.

**Workaround**: Use name-based methods (marginally slower for large trees).

---

#### △ ALERax Configuration

**What Python has**:
```python
result = rustree.reconcile_with_alerax(
    species_newick="sp.nwk",
    gene_newick="g.nwk",
    num_samples=100,
    model="PER-FAMILY",  # String, not enum
    seed=42,
    alerax_path="/path/to/alerax"
)
```

**What Rust has additionally**:
```rust
// Low-level configuration not exposed:
pub struct AleRaxConfig { ... }
pub struct GeneFamily { name: String, alignment_file: String }
pub enum ModelType { PerFamily, Global }

validate_inputs(config) -> Result<(), String>
run_alerax(config) -> Result<AleRaxFamilyResult, String>
reconcile_forest(forest, ...) -> Result<AleRaxForestResult, String>
```

**Impact**:
- Cannot access raw ALERax configuration
- Cannot validate inputs separately
- Cannot run low-level ALERax commands directly

**Workaround**: Use high-level wrappers (sufficient for most use cases).

---

#### △ RecTree Column Access

**What Python has**:
```python
df = gene_tree.to_csv()  # Returns pandas DataFrame
gene_tree.save_xml("file.xml")
```

**What Rust has additionally**:
```rust
// RecTreeColumns struct not exposed:
rec_tree.to_columns() -> RecTreeColumns
columns.to_csv_string() -> String
columns.save_csv(path) -> Result<(), String>
```

**Impact**: Cannot access structured column data directly, only DataFrames.

**Workaround**: Use `to_csv()` which internally converts to DataFrame.

---

### Advanced Feature Gaps

#### △ Tree Conversion & Mapping

**Missing from Python** (`src/node/conversion.rs`):
```rust
// Used internally (ALERax auto-renaming) but not exposed:
map_by_topology(source_tree, target_tree) -> HashMap<usize, usize>
rename_gene_tree(ref_tree, rec_tree) -> RecTree
```

**Impact**: Cannot create topology-based node mappings between trees manually.

**Workaround**: None - this is automatic in ALERax reconciliation.

---

#### △ FlatTree Utilities

**Missing methods** (from `src/node/mod.rs`):
```rust
// Not exposed to Python:
tree.to_node() -> Node  // Convert to recursive structure
tree.len() -> usize
tree.is_empty() -> bool
tree.total_length() -> f64
tree.zero_root_length() -> FlatTree
tree.find_node_index(name) -> Option<usize>
```

**Impact**:
- Cannot convert to recursive Node structure
- Cannot find node indices by name
- Cannot query total tree length
- Cannot zero out root branch length

**Workaround**:
- `tree.num_nodes()` replaces `len()`
- `tree.tree_height()` provides tree depth

---

#### △ Debug Utilities

**Missing from Python** (`src/debug.rs`):
```rust
diffed_flat_tree_table(tree1, tree2) -> String
```

**Impact**: Cannot generate debug tree difference tables.

**Workaround**: Use tree comparison in external tools.

---

## Coverage Summary Table

| Rust Module | Total Functions | Python Bindings | Coverage | Status |
|-------------|----------------|-----------------|----------|--------|
| **DTL Simulation** | 8+ | 8 | ~100% | ✓ Excellent |
| **BD Simulation** | 5+ | 4 | ~80% | ✓ Good |
| **RecTree I/O** | 5 | 3 | ~60% | △ Adequate |
| **Newick I/O** | 4 | 2 | ~50% | △ Adequate |
| **RecPhyloXML I/O** | 5 | 1 | ~20% | △ Limited |
| **FlatTree Basic** | 10+ | 5 | ~50% | △ Adequate |
| **FlatTree Advanced** | 20+ | 3 | ~15% | ✗ Poor |
| **Sampling** | 8 | 2 | ~25% | △ Limited |
| **Metric Functions** | 12+ | 2 | ~17% | ✗ Poor |
| **ALERax (high-level)** | 8 | 2 | ~25% | ✓ Good* |
| **ALERax (low-level)** | 8 | 0 | 0% | ✗ None |
| **Tree Surgery** | 4 | 0 | **0%** | ✗ None |
| **Tree Comparison** | 2 | 0 | **0%** | ✗ None |
| **Induced Transfers** | 2 | 2 | **100%** | ✓ Excellent |
| **Tree Iteration** | 6 | 0 | **0%** | ✗ None |
| **Conversion/Mapping** | 2 | 0 | **0%** | ✗ None |
| **Debug** | 1 | 0 | **0%** | ✗ None |

\* High-level ALERax interface is complete; low-level configuration is not exposed.

---

## Recommendations for Python Users

### When to Use Python Bindings

The Python bindings are **excellent** for:

1. **Simulation workflows** - Full DTL/BD simulation capabilities
2. **Reconciliation** - Complete ALERax integration
3. **Gene tree sampling** - Extant and named sampling
4. **Basic tree analysis** - Leaf names, heights, distances (DataFrame format)
5. **I/O operations** - Newick, XML, CSV export
6. **LTT analysis** - Birth-death event visualization

### When to Use Rust Directly

Use the Rust library when you need:

1. **Tree surgery** - SPR moves, topology editing
2. **Tree comparison** - Topological/metric equivalence testing
3. **Index-based operations** - Efficient bulk node operations
4. **Custom tree traversal** - Pre-order, in-order, post-order iteration
5. **LCA precomputation** - O(1) repeated LCA queries
7. **Raw distance matrices** - Direct access without DataFrame conversion
8. **Low-level ALERax** - Custom configuration beyond high-level wrappers

### Migration Path

If you need advanced features:

```python
# Option 1: Export/import workflow
tree.save_newick("tree.nwk")
# Process in Rust
# Load results back
tree = rustree.parse_species_tree("modified.nwk")

# Option 2: Use external Python libraries
import dendropy
import ete3
# Convert via Newick for advanced tree operations

# Option 3: Write custom Rust extension
# Implement your specific need in Rust + PyO3
```

---

## Potential Future Bindings

**High Priority** (most requested features):
1. Tree comparison (`compare_topology()`, `compare_metric()`)
2. Node index access (`find_node_index()`, `get_node_by_index()`)
3. LCA precomputation (`build_lca_table()`, `query_lca()`)
4. Tree iteration (`iter_preorder()`, `iter_postorder()`)

**Medium Priority**:
6. Tree surgery (`spr_move()`, `prune_subtree()`, `regraft()`)
7. Raw distance matrices (`distance_matrix()` returning numpy array)
8. Depth-based analysis (`make_subdivision()`, `find_contemporaneity()`)
9. Index-based sampling (`sample_by_indices()`)

**Low Priority** (niche use cases):
10. Recursive Node conversion (`to_node()`)
11. Debug utilities (`diff_trees()`)
12. Low-level ALERax configuration

---

## API Design Principles (Current Bindings)

The Python bindings follow these principles:

1. **High-level by default** - Hide internal complexity
2. **DataFrame-first** - Return pandas DataFrames for tabular data
3. **Name-based operations** - Use leaf/node names, not indices
4. **File I/O integrated** - Save/load methods on objects
5. **Streaming support** - Iterators for memory-efficient processing
6. **Automatic conversions** - Arc cloning, RecTree wrapping handled internally

These principles favor **ease of use** over **maximum flexibility**, which explains the gaps in low-level operations.

---

## Conclusion

The rustree Python bindings provide **production-ready coverage** for phylogenetic simulation and reconciliation workflows, with approximately **40% of Rust functionality** exposed.

**Strengths**:
- Complete DTL/BD simulation
- Full ALERax reconciliation
- Comprehensive I/O (Newick, RecPhyloXML, CSV)
- LTT plotting and basic tree analysis

**Gaps**:
- No tree surgery or modification
- No tree comparison utilities
- Limited low-level tree operations
- No index-based node access
- No induced transfer analysis

For most users performing simulation, reconciliation, and export workflows, the Python bindings are **sufficient**. Advanced tree manipulation and analysis require **direct Rust usage** or external Python libraries (dendropy, ete3).

---

**Document Version**: 1.0
**Last Updated**: 2026-02-14
**Rustree Version**: 0.1.0
