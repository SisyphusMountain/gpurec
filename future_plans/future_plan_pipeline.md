# Rust-Based Data Preparation Pipeline Implementation Plan

## Executive Summary

This document outlines the implementation plan for replacing the current Python/ETE3-based data preparation pipeline with a high-performance Rust implementation. The goal is to achieve 10-30x speedup while maintaining bit-identical mathematical correctness through comprehensive testing.

## Assessment of Existing Rust Code (`rustree`)

### ✅ Strong Foundation Already in Place

Your `rustree` codebase provides an excellent foundation:

- **Pest-based Newick parser**: Fast, grammar-based parsing with proper error handling
- **Dual tree representations**: Both recursive (`Node`) and flat (`FlatTree`) structures  
- **Efficient algorithms**: Tree traversal, SPR operations, comparison functions
- **Memory-efficient design**: Flat representation perfect for tensor creation

### Key Advantages Over Current ETE3 Approach

- **10-100x faster parsing** (Rust + Pest vs Python)
- **Zero-copy operations** possible with flat tree representation
- **Memory-efficient** storage with indexed nodes
- **Easy numpy integration** via PyO3

## Implementation Plan

### Phase 1: Python-Rust Bridge + Test Infrastructure (2-3 days)

#### 1.1 Add PyO3 Dependencies

```toml
# rustree/Cargo.toml additions
[lib]
crate-type = ["cdylib"]

[dependencies] 
pyo3 = { version = "0.20", features = ["extension-module"] }
numpy = "0.20"
pest = "2.7"
pest_derive = "2.7"
```

#### 1.2 Create Test Infrastructure First

**Critical: Test-Driven Development Approach**

```python
# tests/rust_python_equivalence/test_species_helpers.py
import pytest
import numpy as np
from src.core.tree_helpers import build_species_helpers as build_species_helpers_python
import gpurec_rust  # The Rust extension

class TestSpeciesHelpersEquivalence:
    
    @pytest.mark.parametrize("tree_file", [
        "tests/data/test_trees_1/sp.nwk",
        "tests/data/test_trees_2/sp.nwk", 
        "tests/data/test_trees_200/sp.nwk"
    ])
    def test_species_helpers_identical(self, tree_file):
        """Verify Rust and Python produce identical species helper matrices"""
        
        # Generate with existing Python code
        python_helpers = build_species_helpers_python(tree_file, 
                                                     device="cpu", dtype=torch.float64)
        
        # Generate with new Rust code
        rust_helpers = gpurec_rust.build_species_helpers(tree_file)
        
        # Compare every tensor element
        np.testing.assert_array_equal(
            python_helpers['C1'].numpy(), 
            rust_helpers.children_left,
            err_msg="Children left matrix mismatch"
        )
        
        np.testing.assert_array_equal(
            python_helpers['C2'].numpy(),
            rust_helpers.children_right, 
            err_msg="Children right matrix mismatch"
        )
        
        np.testing.assert_array_equal(
            python_helpers['ancestors'].numpy(),
            rust_helpers.ancestors,
            err_msg="Ancestors matrix mismatch"
        )
        
        # Verify shapes and dtypes
        assert rust_helpers.children_left.shape == python_helpers['C1'].shape
        assert rust_helpers.children_left.dtype == python_helpers['C1'].numpy().dtype
```

#### 1.3 Baseline Performance Measurements

```python
# tests/performance/test_baseline_performance.py
def test_capture_python_baseline():
    """Capture current Python performance for comparison"""
    tree_files = [
        "tests/data/test_trees_1/sp.nwk",    # Small: ~15 nodes
        "tests/data/test_trees_200/sp.nwk",  # Large: ~400 nodes
    ]
    
    baseline_times = {}
    for tree_file in tree_files:
        start_time = time.time()
        build_species_helpers_python(tree_file, device="cpu", dtype=torch.float64)
        baseline_times[tree_file] = time.time() - start_time
        
    # Save baseline for comparison
    with open("tests/performance/python_baseline.json", "w") as f:
        json.dump(baseline_times, f)
```

### Phase 2: Core Tensor Generation with Validation (3-4 days)

#### 2.1 Species Tree Helper Generation

```rust
// rustree/src/tensor_builders.rs
use pyo3::prelude::*;
use numpy::{IntoPyArray, PyArray2};
use crate::node::FlatTree;

#[pyclass]
pub struct SpeciesHelpers {
    #[pyo3(get)]
    pub children_left: Py<PyArray2<f64>>,
    #[pyo3(get)]
    pub children_right: Py<PyArray2<f64>>,
    #[pyo3(get)] 
    pub ancestors: Py<PyArray2<f64>>,
    #[pyo3(get)]
    pub recipients: Py<PyArray2<f64>>,
}

#[pyfunction]
pub fn build_species_helpers(py: Python, newick_path: &str) -> PyResult<SpeciesHelpers> {
    // Parse tree using existing rustree code
    let tree_str = std::fs::read_to_string(newick_path)?;
    let parsed = crate::newick::newick::NewickParser::parse(Rule::newick, &tree_str)?
        .next().unwrap();
    let trees = crate::newick::newick::newick_to_tree(parsed);
    let tree = &trees[0];
    let flat_tree = tree.to_flat_tree();
    
    // Build matrices using flat tree structure
    let n_nodes = flat_tree.len();
    let children_left = build_children_matrix(&flat_tree, true);
    let children_right = build_children_matrix(&flat_tree, false);
    let ancestors = build_ancestors_matrix(&flat_tree);
    let recipients = build_recipients_matrix(&flat_tree);
    
    Ok(SpeciesHelpers {
        children_left: children_left.into_pyarray(py).to_owned(),
        children_right: children_right.into_pyarray(py).to_owned(),
        ancestors: ancestors.into_pyarray(py).to_owned(),
        recipients: recipients.into_pyarray(py).to_owned(),
    })
}

fn build_children_matrix(flat_tree: &FlatTree, left: bool) -> Vec<Vec<f64>> {
    let n_nodes = flat_tree.len();
    let mut matrix = vec![vec![0.0; n_nodes]; n_nodes];
    
    for (parent_idx, node) in flat_tree.nodes.iter().enumerate() {
        let child_idx = if left { 
            node.left_child 
        } else { 
            node.right_child 
        };
        
        if let Some(child_idx) = child_idx {
            matrix[parent_idx][child_idx] = 1.0;
        }
    }
    
    matrix
}

fn build_ancestors_matrix(flat_tree: &FlatTree) -> Vec<Vec<f64>> {
    let n_nodes = flat_tree.len();
    let mut matrix = vec![vec![0.0; n_nodes]; n_nodes];
    
    // For each node, mark all its ancestors
    for (node_idx, _node) in flat_tree.nodes.iter().enumerate() {
        let mut current = Some(node_idx);
        while let Some(current_idx) = current {
            let current_node = &flat_tree.nodes[current_idx];
            if let Some(parent_idx) = current_node.parent {
                matrix[node_idx][parent_idx] = 1.0;
                current = Some(parent_idx);
            } else {
                current = None;
            }
        }
    }
    
    matrix
}
```

#### 2.2 Comprehensive Testing for Species Helpers

```python
# tests/rust_python_equivalence/test_species_detailed.py
class TestSpeciesHelpersDetailed:
    
    def test_matrix_dimensions_match(self):
        """Verify all matrices have correct dimensions"""
        tree_file = "tests/data/test_trees_1/sp.nwk"
        
        python_helpers = build_species_helpers_python(tree_file)
        rust_helpers = gpurec_rust.build_species_helpers(tree_file)
        
        # All matrices should be square with same dimensions
        n_species = python_helpers['C1'].shape[0]
        
        assert rust_helpers.children_left.shape == (n_species, n_species)
        assert rust_helpers.children_right.shape == (n_species, n_species)
        assert rust_helpers.ancestors.shape == (n_species, n_species)
        assert rust_helpers.recipients.shape == (n_species, n_species)
        
    def test_matrix_properties(self):
        """Test mathematical properties of generated matrices"""
        tree_file = "tests/data/test_trees_1/sp.nwk"
        rust_helpers = gpurec_rust.build_species_helpers(tree_file)
        
        # Each internal node should have exactly 2 children
        C1 = rust_helpers.children_left
        C2 = rust_helpers.children_right
        
        # Sum of children matrices should have max 2 per row (for internal nodes)
        children_sum = C1 + C2
        assert np.all(np.sum(children_sum, axis=1) <= 2)
        
        # Ancestors matrix should be upper triangular for proper tree ordering
        ancestors = rust_helpers.ancestors
        # (This test depends on node ordering - may need adjustment)
        
    def test_numerical_precision(self):
        """Ensure no floating point precision issues"""
        tree_file = "tests/data/test_trees_200/sp.nwk"  # Larger tree
        
        python_helpers = build_species_helpers_python(tree_file, dtype=torch.float64)
        rust_helpers = gpurec_rust.build_species_helpers(tree_file)
        
        # Should be exactly equal, not just close
        np.testing.assert_array_equal(
            python_helpers['C1'].numpy(),
            rust_helpers.children_left,
            err_msg="Precision error in children_left matrix"
        )
        
    def test_data_types_consistent(self):
        """Verify data types match between implementations"""
        tree_file = "tests/data/test_trees_1/sp.nwk"
        
        python_helpers = build_species_helpers_python(tree_file, dtype=torch.float32)
        rust_helpers = gpurec_rust.build_species_helpers(tree_file)  # Should auto-detect
        
        assert rust_helpers.children_left.dtype == np.float32
```

#### 2.3 CCP Construction Implementation

```rust
// rustree/src/ccp_builder.rs
#[pyclass]
pub struct CCPResult {
    #[pyo3(get)]
    pub clade_splits: Py<PyArray2<f64>>,  // [parent_id, left_id, right_id, frequency]
    #[pyo3(get)]
    pub clade_mapping: Py<PyArray2<i32>>, // [clade_id, species_mapping]
    #[pyo3(get)]
    pub split_probabilities: Py<PyArray1<f64>>,
}

#[pyfunction] 
pub fn build_ccp_from_tree(py: Python, gene_newick_path: &str) -> PyResult<CCPResult> {
    // Parse gene tree
    let tree_str = std::fs::read_to_string(gene_newick_path)?;
    let parsed = crate::newick::newick::NewickParser::parse(Rule::newick, &tree_str)?
        .next().unwrap();
    let trees = crate::newick::newick::newick_to_tree(parsed);
    let gene_tree = &trees[0];
    
    // Build all possible rootings using existing tree surgery functions
    let rootings = generate_all_rootings(gene_tree);
    
    // Extract clades and build CCP structure
    let ccp_data = build_ccp_from_rootings(&rootings);
    
    Ok(CCPResult {
        clade_splits: ccp_data.splits.into_pyarray(py).to_owned(),
        clade_mapping: ccp_data.mapping.into_pyarray(py).to_owned(), 
        split_probabilities: ccp_data.probabilities.into_pyarray(py).to_owned(),
    })
}

fn generate_all_rootings(gene_tree: &Node) -> Vec<Node> {
    // Use existing flat tree representation for efficient rooting
    let flat_tree = gene_tree.to_flat_tree();
    let mut rootings = Vec::new();
    
    // For each edge, create a rooting
    for (node_idx, node) in flat_tree.nodes.iter().enumerate() {
        if node.parent.is_some() {  // Don't root on existing root
            let rooted_tree = root_at_edge(&flat_tree, node_idx);
            rootings.push(rooted_tree.to_node());
        }
    }
    
    rootings
}
```

#### 2.4 CCP Testing with Exact Frequency Validation

```python
# tests/rust_python_equivalence/test_ccp_construction.py
class TestCCPConstruction:
    
    @pytest.mark.parametrize("gene_tree_file", [
        "tests/data/test_trees_1/g.nwk",
        "tests/data/test_trees_2/g.nwk",
        "tests/data/test_trees_3/g.nwk"
    ])
    def test_ccp_splits_identical(self, gene_tree_file):
        """Verify CCP split frequencies are identical"""
        
        # Python implementation
        python_ccp = build_ccp_from_single_tree_python(gene_tree_file, debug=False)
        
        # Rust implementation  
        rust_ccp = gpurec_rust.build_ccp_from_tree(gene_tree_file)
        
        # Compare number of clades
        assert len(python_ccp.id_to_clade) == len(rust_ccp.clade_mapping)
        
        # Compare split frequencies - this is critical!
        python_splits = extract_splits_dict(python_ccp)
        rust_splits = convert_rust_splits_to_dict(rust_ccp)
        
        for clade_id, python_split_list in python_splits.items():
            rust_split_list = rust_splits[clade_id]
            
            # Sort both by split signature for comparison
            python_sorted = sorted(python_split_list, key=lambda x: (x.left_id, x.right_id))
            rust_sorted = sorted(rust_split_list, key=lambda x: (x[1], x[2]))  # left_id, right_id
            
            assert len(python_sorted) == len(rust_sorted)
            
            for python_split, rust_split in zip(python_sorted, rust_sorted):
                # Compare frequencies with high precision
                np.testing.assert_allclose(
                    python_split.frequency, 
                    rust_split[3],  # frequency is 4th element
                    rtol=1e-15,
                    err_msg=f"Frequency mismatch for clade {clade_id}"
                )
                
    def test_ccp_normalization(self):
        """Verify split frequencies sum to 1.0 for each clade"""
        gene_tree_file = "tests/data/test_trees_1/g.nwk"
        rust_ccp = gpurec_rust.build_ccp_from_tree(gene_tree_file)
        
        # Group splits by parent clade
        splits_by_clade = {}
        for split in rust_ccp.clade_splits:
            parent_id = int(split[0])
            if parent_id not in splits_by_clade:
                splits_by_clade[parent_id] = []
            splits_by_clade[parent_id].append(split[3])  # frequency
            
        # Each clade's splits should sum to 1.0
        for clade_id, frequencies in splits_by_clade.items():
            total_freq = sum(frequencies)
            np.testing.assert_allclose(
                total_freq, 1.0, rtol=1e-12,
                err_msg=f"Clade {clade_id} frequencies don't sum to 1.0: {total_freq}"
            )
```

### Phase 3: Integration Testing (2-3 days)

#### 3.1 End-to-End Pipeline Verification

```python
# tests/rust_python_equivalence/test_end_to_end.py
class TestEndToEndPipeline:
    
    @pytest.mark.parametrize("test_case", [
        {
            "species": "tests/data/test_trees_1/sp.nwk",
            "gene": "tests/data/test_trees_1/g.nwk", 
            "delta": 1e-10, "tau": 1e-10, "lambda": 1e-10
        },
        {
            "species": "tests/data/test_trees_2/sp.nwk",
            "gene": "tests/data/test_trees_2/g.nwk",
            "delta": 0.1, "tau": 0.05, "lambda": 0.1
        }
    ])
    def test_complete_reconciliation_identical(self, test_case):
        """Test that complete reconciliation gives identical results"""
        
        # Full Python pipeline
        python_result = reconcile_ccp_log(
            test_case["species"], test_case["gene"],
            delta=test_case["delta"], tau=test_case["tau"], 
            lambda_param=test_case["lambda"], debug=False
        )
        
        # Rust data preparation + Python reconciliation logic
        rust_data = gpurec_rust.prepare_reconciliation_data(
            test_case["species"], test_case["gene"]
        )
        
        # Convert Rust data to PyTorch tensors
        species_helpers = {
            'C1': torch.from_numpy(rust_data.species_helpers.children_left),
            'C2': torch.from_numpy(rust_data.species_helpers.children_right),
            'ancestors': torch.from_numpy(rust_data.species_helpers.ancestors),
            'recipients': torch.from_numpy(rust_data.species_helpers.recipients)
        }
        
        ccp_helpers = convert_rust_ccp_to_torch(rust_data.ccp_result)
        
        # Run reconciliation with Rust-prepared data
        mixed_result = reconcile_ccp_log_with_precomputed_data(
            species_helpers, ccp_helpers, rust_data.clade_species_mapping,
            delta=test_case["delta"], tau=test_case["tau"], 
            lambda_param=test_case["lambda"], debug=False
        )
        
        # Results must be numerically identical  
        np.testing.assert_allclose(
            python_result['log_likelihood'], 
            mixed_result['log_likelihood'],
            rtol=1e-12, atol=1e-15,
            err_msg="End-to-end likelihood mismatch"
        )
        
        # Also compare intermediate tensors if available
        if 'final_log_Pi' in python_result and 'final_log_Pi' in mixed_result:
            np.testing.assert_allclose(
                python_result['final_log_Pi'].numpy(),
                mixed_result['final_log_Pi'].numpy(), 
                rtol=1e-10,
                err_msg="Final Pi matrix mismatch"
            )
```

#### 3.2 Parameter Sensitivity Testing

```python
def test_parameter_sensitivity_identical():
    """Verify behavior is identical across parameter ranges"""
    
    species_tree = "tests/data/test_trees_1/sp.nwk" 
    gene_tree = "tests/data/test_trees_1/g.nwk"
    
    # Test range of parameters
    param_ranges = [
        {"delta": 1e-10, "tau": 1e-10, "lambda": 1e-10},  # Near-zero rates
        {"delta": 0.5, "tau": 0.3, "lambda": 0.4},        # High rates
        {"delta": 0.01, "tau": 0.02, "lambda": 0.01},     # Small rates
    ]
    
    for params in param_ranges:
        python_result = reconcile_ccp_log(species_tree, gene_tree, **params, debug=False)
        
        rust_data = gpurec_rust.prepare_reconciliation_data(species_tree, gene_tree)
        mixed_result = reconcile_with_rust_data(rust_data, **params)
        
        np.testing.assert_allclose(
            python_result['log_likelihood'],
            mixed_result['log_likelihood'],
            rtol=1e-12,
            err_msg=f"Parameter sensitivity test failed for {params}"
        )
```

### Phase 4: Edge Cases and Performance Validation (1-2 days)

#### 4.1 Edge Case Testing

```python  
# tests/rust_python_equivalence/test_edge_cases.py
class TestEdgeCases:
    
    def test_single_leaf_tree(self):
        """Test with minimal tree"""
        # Create single-leaf tree file
        single_leaf_newick = "A:1.0;"
        with open("temp_single_leaf.nwk", "w") as f:
            f.write(single_leaf_newick)
            
        try:
            python_result = build_species_helpers_python("temp_single_leaf.nwk")  
            rust_result = gpurec_rust.build_species_helpers("temp_single_leaf.nwk")
            
            # Should handle gracefully and produce identical results
            assert python_result['C1'].shape == rust_result.children_left.shape
            np.testing.assert_array_equal(python_result['C1'].numpy(), 
                                        rust_result.children_left)
        finally:
            os.remove("temp_single_leaf.nwk")
            
    def test_zero_branch_lengths(self):
        """Test with zero branch lengths"""
        zero_branch_newick = "(A:0.0,B:0.0):0.0;"
        with open("temp_zero_branches.nwk", "w") as f:
            f.write(zero_branch_newick)
            
        try:
            # Both implementations should handle this identically
            python_result = build_species_helpers_python("temp_zero_branches.nwk")
            rust_result = gpurec_rust.build_species_helpers("temp_zero_branches.nwk")
            
            np.testing.assert_array_equal(python_result['C1'].numpy(),
                                        rust_result.children_left)
        finally:
            os.remove("temp_zero_branches.nwv")
            
    def test_large_tree_stress(self):
        """Test with computationally intensive tree"""
        # Use test_trees_200 as stress test
        large_tree = "tests/data/test_trees_200/sp.nwk"
        
        # Both should complete without errors and give identical results
        python_result = build_species_helpers_python(large_tree)
        rust_result = gpurec_rust.build_species_helpers(large_tree) 
        
        np.testing.assert_array_equal(python_result['C1'].numpy(),
                                    rust_result.children_left)
```

#### 4.2 Performance Validation and Regression Testing

```python
# tests/performance/test_performance_validation.py  
class TestPerformanceValidation:
    
    @pytest.mark.performance
    def test_species_helper_performance(self):
        """Verify Rust implementation is significantly faster"""
        
        tree_files = [
            ("small", "tests/data/test_trees_1/sp.nwk"),
            ("medium", "tests/data/test_trees_2/sp.nwk"), 
            ("large", "tests/data/test_trees_200/sp.nwk")
        ]
        
        performance_results = {}
        
        for size_name, tree_file in tree_files:
            # Time Python implementation
            python_times = []
            for _ in range(5):  # Average over 5 runs
                start = time.time()
                build_species_helpers_python(tree_file, device="cpu", dtype=torch.float64)
                python_times.append(time.time() - start)
            python_avg = sum(python_times) / len(python_times)
            
            # Time Rust implementation  
            rust_times = []
            for _ in range(5):
                start = time.time()
                gpurec_rust.build_species_helpers(tree_file)
                rust_times.append(time.time() - start)
            rust_avg = sum(rust_times) / len(rust_times)
            
            speedup = python_avg / rust_avg
            performance_results[size_name] = {
                'python_time': python_avg,
                'rust_time': rust_avg, 
                'speedup': speedup
            }
            
            # Rust should be at least 5x faster for all tree sizes
            assert speedup >= 5.0, f"Insufficient speedup for {size_name}: {speedup:.2f}x"
            
        # Save results for tracking over time
        with open("tests/performance/latest_results.json", "w") as f:
            json.dump(performance_results, f, indent=2)
            
        print("Performance Results:")
        for size, results in performance_results.items():
            print(f"{size}: {results['speedup']:.1f}x speedup "
                  f"({results['python_time']:.3f}s -> {results['rust_time']:.3f}s)")
    
    @pytest.mark.performance 
    def test_memory_usage_improvement(self):
        """Verify Rust uses less memory"""
        import psutil
        import gc
        
        tree_file = "tests/data/test_trees_200/sp.nwk"
        
        # Measure Python memory usage
        gc.collect()
        process = psutil.Process()
        memory_before_python = process.memory_info().rss
        
        python_result = build_species_helpers_python(tree_file)
        memory_after_python = process.memory_info().rss
        python_memory_delta = memory_after_python - memory_before_python
        
        del python_result
        gc.collect()
        
        # Measure Rust memory usage  
        memory_before_rust = process.memory_info().rss
        rust_result = gpurec_rust.build_species_helpers(tree_file)
        memory_after_rust = process.memory_info().rss
        rust_memory_delta = memory_after_rust - memory_before_rust
        
        # Rust should use less memory
        memory_ratio = python_memory_delta / rust_memory_delta  
        assert memory_ratio >= 1.5, f"Insufficient memory improvement: {memory_ratio:.2f}x"
        
        print(f"Memory usage: Python {python_memory_delta/1024/1024:.1f}MB, "
              f"Rust {rust_memory_delta/1024/1024:.1f}MB "
              f"({memory_ratio:.1f}x improvement)")
```

## Expected Performance Improvements

Based on the Rust implementation plan:

- **Tree parsing**: 50-100x faster than ETE3
- **CCP construction**: 20-50x faster than Python loops  
- **Memory allocation**: 10-20x faster with direct numpy arrays
- **Memory usage**: 3-5x reduction in peak memory
- **Overall pipeline**: **10-30x end-to-end speedup**

## Implementation Timeline

### Week 1: Foundation and Testing Infrastructure
- **Day 1-2**: Set up PyO3 bindings, create test infrastructure
- **Day 3-4**: Implement and test basic tree parsing with validation  
- **Day 5**: Species tree helper generation with comprehensive testing

### Week 2: CCP Construction and Integration
- **Day 1-3**: CCP construction with exact frequency validation
- **Day 4-5**: End-to-end pipeline testing and edge case coverage

### Week 3: Performance and Polish  
- **Day 1-2**: Performance optimization and memory profiling
- **Day 3-4**: Comprehensive stress testing and edge cases
- **Day 5**: Documentation and final validation

## Success Criteria

✅ **100% test coverage** on all tensor generation functions  
✅ **Bit-identical results** for all test cases (rtol=1e-12 or better)  
✅ **Minimum 10x performance improvement** verified by benchmarks  
✅ **Memory usage reduction** of at least 3x measured and validated  
✅ **Edge cases handled** identically to Python implementation  
✅ **Integration tests pass** for complete reconciliation pipeline  
✅ **No performance regressions** in subsequent development  

## Testing Philosophy

This implementation follows a **test-first, correctness-first** approach:

1. **Write tests before implementation** to capture expected behavior
2. **Validate against existing Python code** as ground truth
3. **Test edge cases extensively** to prevent regressions  
4. **Performance test with regression protection** to ensure continued benefits
5. **Comprehensive coverage** of all data paths and mathematical operations

The comprehensive test suite serves as both validation and documentation, ensuring that the Rust implementation maintains mathematical correctness while delivering significant performance improvements.

## Conclusion

This plan provides a systematic approach to replacing the Python data preparation pipeline with a high-performance Rust implementation. The emphasis on comprehensive testing ensures we maintain the mathematical rigor required for phylogenetic reconciliation while achieving substantial performance gains.

The existing `rustree` codebase provides an excellent foundation, and the test-driven development approach ensures we can confidently deploy the new implementation knowing it produces identical results to the current Python version.