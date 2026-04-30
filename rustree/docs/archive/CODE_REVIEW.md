# Rustree Codebase Review

**Date:** 2026-02-13
**Scope:** Full codebase review of the `rustree` crate
**Methodology:** 3 specialized review agents (core data structures & architecture, simulation engine & algorithms, I/O & analysis & bindings), synthesized into this report

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [What's Already Good](#2-whats-already-good)
3. [Critical Issues](#3-critical-issues)
4. [Medium Issues](#4-medium-issues)
5. [Lower Priority / Polish](#5-lower-priority--polish)
6. [Per-Module Detailed Findings](#6-per-module-detailed-findings)
7. [Code Duplication (DRY Violations)](#7-code-duplication-dry-violations)
8. [Dead / Questionable Code](#8-dead--questionable-code)
9. [Test Quality](#9-test-quality)
10. [Summary Table](#10-summary-table)
11. [Recommended Action Plan](#11-recommended-action-plan)

---

## 1. Executive Summary

The `rustree` crate (~12,000 lines) is a solid phylogenetics library with correct algorithms, efficient data structures, and good module separation. The simulation engine (Birth-Death + DTL Gillespie) is algorithmically sound. Key design wins include `Arc<FlatTree>` sharing, optimal O(1) LCA queries via sparse table, and the new streaming `DtlSimIter`.

**Primary concerns:**
- **String-based errors everywhere** — the single biggest code quality gap
- **Panics in library code** (`assert!`, `expect!`) instead of returning errors
- **Infinite retry loop** in the DTL iterator when `require_extant=true`
- **Memory leak** via `Box::leak()` in XML generation
- **No XML/CSV escaping** — produces invalid output on special characters

**Overall quality rating: 7.5/10** — Production-quality algorithms with gaps in error handling discipline and defensive programming.

---

## 2. What's Already Good

### Architecture
- Clean module separation (`simulation/bd`, `simulation/dtl`, `io/`, `node/`, `external/`)
- Re-exports in `lib.rs` provide a sensible public API
- Feature-gated Python/R bindings are well-isolated
- `pub(crate)` visibility used appropriately for internal APIs

### Algorithms
- Gillespie DTL engine is correctly implemented (verified line-by-line)
- LCA uses optimal O(n log n) sparse table preprocessing with O(1) queries (Euler tour + RMQ)
- Birth-Death backward simulation is sound
- Streaming `DtlSimIter` is an excellent design: lazy, memory-efficient, chainable
- `find_time_index()` uses binary search with correct ceiling behavior — well-tested with 12 unit tests

### Data Structures
- Index-based `FlatTree` is cache-friendly and efficient for serialization/traversal
- `Arc<FlatTree>` sharing for species trees across `RecTree` and `GeneForest` — zero-cost sharing
- Parallel vectors for node/event mappings in `RecTree` are well-aligned
- `swap_remove()` in gene-per-species tracking is O(1)

### Testing
- 72+ library tests with good coverage
- Excellent LCA parity tests (8 topologies: ultrametric, non-ultrametric, caterpillar, balanced, etc.)
- Comprehensive RecPhyloXML round-trip testing
- DTL tests cover: pure speciation, duplication, transfer, loss, replacement, assortative, backward compatibility
- Benchmarks properly marked `#[ignore]`

### Other Strengths
- NaN-safe numeric operations using `f64::total_cmp()` throughout
- Production-grade simulation with unified Gillespie loop handling both per-gene and per-species modes
- Sophisticated ALERax integration with streaming output, auto-renaming via topology mapping
- Rich Python integration with pandas DataFrames, matplotlib plotting, IPython display

---

## 3. Critical Issues

### 3.1 String-based errors everywhere

Nearly every function returns `Result<T, String>`. This is the single biggest code quality gap.

```rust
// Current pattern (throughout the codebase):
pub fn simulate_dtl(...) -> Result<(RecTree, Vec<DTLEvent>), String>
pub fn map_by_topology(...) -> Result<HashMap<usize, usize>, String>
pub fn extract_induced_subtree(...) -> Option<(FlatTree, ...)>  // even worse: Option with no error info
```

**Files affected:** `per_gene.rs`, `per_species.rs`, `stream.rs`, `gillespie.rs`, `utils.rs`, `state.rs`, `conversion.rs`, `sampling.rs`, `induced_transfers.rs`, `alerax.rs`, and more.

**Problems:**
1. No pattern matching on error kinds
2. No `?` composition with `std::error::Error` trait
3. No backtrace or error chain
4. Inconsistent: some functions use `Result<T, String>`, others use `Option<T>` with no error info

**The one exception** is `io/recphyloxml.rs` which has a proper `ParseError` enum with `impl Error` — this should be the model for the rest.

**Recommendation:** Create a unified `RustreeError` enum:
```rust
#[derive(Debug)]
pub enum RustreeError {
    Parsing(String),
    TopologyMismatch { expected: usize, got: usize },
    SimulationError(String),
    IoError(std::io::Error),
    InvalidRate { name: &'static str, value: f64 },
    MaxRetriesExceeded(usize),
    // ...
}
impl std::fmt::Display for RustreeError { ... }
impl std::error::Error for RustreeError { ... }
```
Consider using `thiserror` crate for ergonomics.

### 3.2 `assert!` / `expect!` in library code

| File | Line(s) | Issue |
|------|---------|-------|
| `src/node/rectree.rs` | 53-62 | `assert_eq!` in `RecTree::new()` panics on mismatched mapping lengths |
| `src/simulation/dtl/utils.rs` | 30 | `precompute_lca()` uses `.expect()` which panics on malformed trees |
| `src/metric_functions.rs` | 266, 299 | `.expect()` on depth values that might not be assigned |

These should all return `Result` instead of panicking. Library code should never panic on user input.

### 3.3 Infinite retry loop in `DtlSimIter::next()`

**File:** `src/simulation/dtl/stream.rs:159-187`

```rust
fn next(&mut self) -> Option<Self::Item> {
    if self.completed >= self.n_simulations { return None; }
    loop {  // <-- UNBOUNDED: loops forever if all trees have zero extant genes
        let result = simulate_dtl_gillespie(...);
        match result {
            Ok((rec_tree, events)) => {
                if !self.require_extant || count_extant_genes(&rec_tree) > 0 {
                    self.completed += 1;
                    return Some(Ok((rec_tree, events)));
                }
                // Retry: tree had no extant genes — NO BOUND ON RETRIES
            }
            Err(e) => return Some(Err(e)),
        }
    }
}
```

When `require_extant=true` and rates are highly unfavorable (e.g., `lambda_l=10.0, lambda_d=0.01`), this loops forever. Same issue in `per_gene.rs`, `per_species.rs` wrappers, and `PyDtlSimIter::next_simulation()`.

**Recommendation:** Add a max retry count (e.g., 10,000) and return an error if exceeded.

### 3.4 Memory leak via `Box::leak()`

**File:** `src/io/rectree_xml.rs:195-198`

```rust
Box::leak(Box::new("\t".repeat(indent)))
```

For trees with depth >= 10, indent strings are leaked on every call. This is acknowledged in a comment but remains a genuine memory leak for deep trees.

**Recommendation:** Use a larger static array (e.g., 32 levels) or return `Cow<'static, str>`.

---

## 4. Medium Issues

### 4.1 Missing derives on core types

`FlatNode`, `FlatTree`, `Node` only derive `Clone, Debug`. Missing `PartialEq` forces tests to use manual comparison workarounds. Adding `PartialEq, Eq` would simplify tests and enable `assert_eq!` on trees directly.

Other missing derives:

| Type | File | Missing |
|------|------|---------|
| `TraversalOrder` | `src/node/mod.rs:68` | `PartialEq, Eq, Copy` |
| `AleRaxConfig` | `src/external/alerax.rs:16` | `Clone, Debug` |
| `AleRaxFamilyResult` | `src/external/alerax.rs:76` | `Clone, Debug` |
| `AleRaxForestResult` | `src/external/alerax.rs:115` | `Clone, Debug` |
| `RecTreeColumns` | `src/io/rectree_csv.rs` | `Clone, Debug` |

### 4.2 `bd_event` field wastes memory

**File:** `src/node/mod.rs`

`FlatNode::bd_event: Option<BDEvent>` is only set during BD simulation but occupies 8-16 bytes on every node in every tree.

**Recommendation:** Move to a separate `Vec<Option<BDEvent>>` alongside the tree, or a wrapper type like `BDAnnotatedTree`.

### 4.3 `depth: Option<f64>` — latent precondition

Depths are optional but **required** by most operations (metrics, DTL simulation, contemporaneity). Several places use `.expect()` or `.unwrap()` on it (see issue 3.2). This creates a runtime-enforceable invariant that should be a compile-time guarantee.

**Recommendation:** Either always assign depths on construction, or create a newtype wrapper `DepthAssignedTree(FlatTree)` to enforce statically.

### 4.4 No XML/CSV escaping in output

**File:** `src/io/rectree_xml.rs`

Node names with `<`, `>`, `&`, or quotes produce invalid XML. Same issue in CSV output — names containing commas, newlines, or formula-triggering characters (`=`, `+`, `@`) break the format or create CSV injection risk.

**Recommendation:** Implement XML escaping (`&amp;`, `&lt;`, etc.) and CSV quoting (RFC 4180).

### 4.5 Undocumented `2.0` multiplier in assortative transfer

**File:** `src/simulation/dtl/utils.rs:115`

```rust
let distance = 2.0 * (event_time - lca_depth);
```

The factor of 2 is unexplained. Is it coalescent scaling? Patristic distance (accounting for both branches from LCA to tips)? This is a scientific parameter that affects simulation results and needs clear documentation.

### 4.6 `random_gene_copy()` is O(n) per call

**File:** `src/simulation/dtl/state.rs:175-187`

Iterates all species/genes on every event selection in PerGene mode. For simulations with thousands of species and high duplication rates, this becomes a bottleneck.

**Recommendation:** Consider alias method or segment tree for O(1) selection, though current approach is acceptable for moderate problem sizes.

### 4.7 `eprintln!` in library code

**File:** `src/surgery.rs:25,33`

Library functions print to stderr instead of returning errors.

**Recommendation:** Return `Result<_, String>` (or `RustreeError`) instead.

### 4.8 Unnecessary deep clone in Newick parser

**File:** `src/newick/newick.rs:112-113`

```rust
subtrees.get(0).cloned()  // forces deep clone of entire Node tree
```

**Fix:** Use `subtrees.into_iter().next()` for zero-cost move.

### 4.9 Unnecessary event clone in Gillespie simulation

**File:** `src/simulation/dtl/gillespie.rs:132`

```rust
let sp_event = species_events[species_event_idx].clone();  // clone where reference suffices
```

**Fix:** Use `let sp_event = &species_events[species_event_idx]`.

---

## 5. Lower Priority / Polish

### 5.1 Comparison module is skeletal

**File:** `src/comparison.rs` (47 lines)

Only has recursive `Node` comparison. No `FlatTree` support, no Robinson-Foulds distance, no tree distance metrics. Seems incomplete.

### 5.2 Newick parser doesn't support quoted names

Standard Newick allows `'Species 1':1.0` — the parser rejects this silently. Should document the limitation or add support.

### 5.3 `induced_transfers.rs` has redundant API

**File:** `src/induced_transfers.rs:152`

Requires both `sampled_tree` AND `sampled_leaf_names` when the latter could be derived from the former. There's a TODO comment acknowledging this.

### 5.4 Python bindings lack docstrings

Most `#[pymethods]` (now in `src/python/species_tree.rs`, `gene_tree.rs`, etc.) have no Python-visible documentation. A `.pyi` stub file exists (`rustree.pyi`) but may be incomplete. R bindings similarly lack complete roxygen2 documentation.

### 5.5 No `serde` support

Despite handling serialization to multiple formats (Newick, XML, CSV), there's no optional `serde` derive on core types. Would be useful for JSON/TOML config, caching, and Python interop.

### 5.6 No property-based tests

No `proptest` or `quickcheck` usage. Randomized testing would catch edge cases in tree traversal, index bounds, and simulation invariants that seed-specific tests miss.

### 5.7 No module-level documentation

Missing `//!` doc comments on: `src/simulation/mod.rs`, `src/comparison.rs`, `src/node/iter.rs`, `src/node/traits.rs`.

### 5.8 Hard-coded absolute paths in tests

**Files:** `tests/test_alerax_real_file.rs:6`, `tests/test_real_separate_files.rs`

Use hardcoded absolute paths. Should use environment variables or relative paths for portability.

### 5.9 `genes_per_species: Option<HashMap>` appears always `Some`

**File:** `src/simulation/dtl/state.rs:30`

The `Option` wrapper adds unnecessary complexity. If it's always `Some`, just use the `HashMap` directly.

### 5.10 Python ylabel typo

**File:** `src/python/species_tree.rs` (was `src/python.rs:475`)

`"Number of lineages)"` — missing opening parenthesis.

---

## 6. Per-Module Detailed Findings

### `src/node/` (Core Data Structures)

| File | Issues | Top Concern |
|------|--------|-------------|
| `mod.rs` | 3 | Cross-module re-exports mixing concerns (IO types through node module) |
| `rectree.rs` | 4 | Panic-prone constructor and accessors; `species_node_for()` has no bounds checking |
| `conversion.rs` | 2 | Good topology mapping algorithm; unused `pos` variable from enumerate |
| `traits.rs` | 2 | Redundant `HasName` impls for `&Node` / `&FlatNode` (deref handles this) |
| `iter.rs` | 2 | Undocumented unreachable patterns; no lazy traversal option |
| `gene_forest.rs` | 1 | Well-designed; near-identical error messages could consolidate |

### `src/simulation/` (BD & DTL Simulation)

| File | Issues | Top Concern |
|------|--------|-------------|
| `bd/simulation.rs` | 1 | Minor: stray character in comment; pre-allocation heuristic is sound |
| `dtl/gillespie.rs` | 3 | Unnecessary event clone; incomplete comments; induced transfer handling unclear |
| `dtl/state.rs` | 3 | O(n) random gene selection; `Option` wrapper on always-Some field; unresolved TODO |
| `dtl/stream.rs` | 1 | **Infinite retry loop** when `require_extant=true` with unfavorable rates |
| `dtl/utils.rs` | 2 | `expect()` instead of Result; undocumented `2.0` multiplier |
| `dtl/event.rs` | 1 | Redundant `from_species` field (acknowledged in comment) |
| `dtl/per_gene.rs` + `per_species.rs` | 1 | Duplicated setup logic (~20 lines each) |

**Overall:** Algorithmically sound. Gillespie loop, waiting times, event selection, transfer recipient selection are all correct.

### `src/io/` (I/O Module)

| File | Issues | Top Concern |
|------|--------|-------------|
| `recphyloxml.rs` | 4 | Code duplication between species/gene tree parsing (~200 lines); string allocations in hot loop |
| `rectree_xml.rs` | 3 | **Memory leak** via `Box::leak()`; no XML escaping |
| `rectree_csv.rs` | 3 | No CSV escaping/quoting; missing derives |
| `csv.rs` | 1 | Inconsistent error handling between BD and DTL event export |

**Bright spot:** `recphyloxml.rs` has a proper `ParseError` enum with `impl Error` — the only module with structured errors.

### `src/metric_functions.rs`

**Excellent** — O(n log n) LCA preprocessing, O(1) queries, well-tested. Minor issues: 2 unnecessary explicit returns, undocumented time complexity.

### `src/sampling.rs`

**Well-designed** induced subtree algorithm. Issues: linear search where HashMap would help; leaf collection doesn't short-circuit after 2 leaves.

### `src/external/alerax.rs`

**Comprehensive** ALERax integration. Issues: missing derives on public types; full tree clone for renaming; no version checking; threading could lose data on panic.

### `src/python/` (split from monolithic `python.rs`)

**Comprehensive** Python API, now organized into submodules: `mod.rs` (thin wiring + module registration), `species_tree.rs`, `gene_tree.rs`, `sim_iter.rs`, `types.rs`, plus existing `reconciliation.rs`, `alerax.rs`, `forest.rs`, `training.rs`. Shared validation logic delegated to `bindings_common`. Remaining issues: incomplete docstrings; ylabel typo; `PyDtlSimIter` inherits the infinite retry bug.

### `src/r/` (split from monolithic `r.rs`)

**Good** R API, now organized as `mod.rs` (all `#[extendr]` functions + `extendr_module!` macro) + `conversions.rs` (R↔Rust type conversion helpers). Rate validation and distance type parsing delegated to `bindings_common`. Remaining issues: no unit tests for R bindings; incomplete parameter validation.

---

## 7. Code Duplication (DRY Violations)

Sorted by estimated lines saved:

| # | Files | Duplication | Est. Lines Saved |
|---|-------|-------------|------------------|
| 1 | `io/recphyloxml.rs` | `parse_species_tree()` / `parse_gene_tree()` share ~70% logic; `species_node_to_flat_tree()` / `gene_node_to_flat_tree()` ~90% identical | ~200 |
| 2 | `python.rs` | Thirdkind SVG generation duplicated between `PySpeciesTree::to_svg()` and `PyGeneTree::to_svg()` | ~100 |
| 3 | `python.rs` | Sampling index-remapping code in `sample_extant()` and `sample_by_names()` | ~83 |
| 4 | ~~`python.rs`~~ **FIXED** | ~~Same distance type match block appears 4 times~~ Extracted to `bindings_common::parse_distance_type()` | ~~60~~ 0 |
| 5 | `io/rectree_csv.rs` | CSV row formatting duplicated between `to_csv_string()` and `save_csv()` | ~30 |
| 6 | `dtl/per_gene.rs` + `per_species.rs` | Identical setup logic (Arc, events, depths, contemporaneity, LCA) | ~20 |

---

## 8. Dead / Questionable Code

| # | File | Line(s) | Issue |
|---|------|---------|-------|
| 1 | `src/sampling.rs` | 330 | Unused parameter `_sampled_lca_map` in `build_sampled_to_original_mapping` |
| 2 | `src/io/recphyloxml.rs` | 20 | Unused `ParseError::InvalidEvent` variant (never constructed) |
| 3 | `src/simulation/dtl/event.rs` | 29 | Redundant `from_species` field (comment says "redundant with species_id") |
| 4 | `src/simulation/bd/simulation.rs` | 159 | Stray character `a` at end of comment |
| 5 | `src/simulation/dtl/state.rs` | 23 | Unresolved TODO: "make the function names more coherent and clear" |
| 6 | `src/simulation/dtl/state.rs` | 30 | `genes_per_species: Option<HashMap>` — the `Option` wrapper is always `Some` |
| 7 | ~~`src/main.rs`~~ | ~~2~~ | ~~Comment says "just testing code here"~~ File deleted |
| 8 | `src/node/traits.rs` | 11-32 | Redundant `HasName` impls for `&Node` / `&FlatNode` |
| 9 | `src/node/conversion.rs` | 140 | Unused `pos` variable from `enumerate()` |

---

## 9. Test Quality

### Strengths
- Excellent LCA parity tests (8 topologies tested)
- Comprehensive RecPhyloXML round-trip testing
- RecTree sampling tests cover edge cases (single leaf, all leaves, with duplication, error cases)
- DTL tests cover pure speciation, high D/T/L rates, replacement, assortative, backward compatibility
- Benchmarks properly marked `#[ignore]` with clear run instructions

### Gaps
- **No property-based tests** (proptest/quickcheck) — would catch edge cases in tree operations
- **No stress tests** with extreme DTL parameters — would reveal the infinite retry loop
- **No test for empty/degenerate species trees** in DTL simulation
- **No floating-point precision tests** with very small branch lengths or rates
- **No R binding tests** visible in the Rust test suite
- **Hard-coded absolute paths** in 2 test files (now fixed)
- **Loose assertions** in BD tests (`tree.nodes.len() > 0` instead of exact count)
- Only 1 test in `sampling_tests.rs` despite plural file name

---

## 10. Summary Table

| Priority | Issue | Impact |
|----------|-------|--------|
| **Critical** | String-based errors everywhere | Unusable for error composition, no error discrimination |
| **Critical** | `assert!`/`expect!` in library code | Panics on bad input instead of returning errors |
| **Critical** | Infinite retry loop in DTL iterator | Hangs forever with unfavorable rates |
| **Critical** | `Box::leak()` in XML generation | Memory leak for deep trees |
| **Medium** | Missing `PartialEq` derives on core types | Test ergonomics; can't compare trees with `==` |
| **Medium** | `bd_event` wasting space on all nodes | 8-16 bytes overhead per node |
| **Medium** | `depth: Option<f64>` precondition | Runtime panics if `assign_depths()` forgotten |
| **Medium** | No XML/CSV escaping | Invalid output on special characters |
| **Medium** | Undocumented assortative transfer math | Scientific correctness unclear to readers |
| **Medium** | O(n) random gene selection | Performance bottleneck at scale |
| **Medium** | `eprintln!` in library code | Side effects instead of error returns |
| **Medium** | Unnecessary clones (Newick parser, Gillespie) | Avoidable allocations |
| **Low** | Comparison module incomplete | Missing RF distance, FlatTree support |
| **Low** | No serde support | Limits interoperability |
| **Low** | No property-based tests | Edge cases not systematically covered |
| **Low** | ~500 lines of code duplication (reduced by ~60 via `bindings_common`) | Maintenance burden |
| **Low** | Missing module-level documentation | Onboarding difficulty |
| **Low** | Python bindings lack docstrings | Poor discoverability for Python users |
| **Low** | Newick parser doesn't support quoted names | Limitation not documented |

---

## 11. Recommended Action Plan

### Phase 1: Critical Fixes
1. **Add max retry count** to `DtlSimIter::next()` and all retry loops (stream.rs, per_gene.rs, per_species.rs, PyDtlSimIter)
2. **Fix `Box::leak()` memory leak** in `rectree_xml.rs` — use static array or `Cow<'static, str>`
3. **Replace `assert!`/`expect!`** in `RecTree::new()`, `precompute_lca()`, metric functions — return `Result` instead

### Phase 2: Error Handling Overhaul
4. **Create `RustreeError` enum** with variants for parsing, simulation, topology, I/O, rate validation
5. **Replace all `Result<T, String>`** with `Result<T, RustreeError>` across the codebase
6. **Replace `eprintln!`** in `surgery.rs` with error returns
7. Consider `thiserror` crate for ergonomic error derive

### Phase 3: Correctness & Safety
8. **Add XML escaping** to `rectree_xml.rs` output
9. **Add CSV quoting** (RFC 4180) to `rectree_csv.rs`
10. **Add bounds checking** to `RecTree::species_node_for()` and `event_for()`
11. **Document the `2.0` multiplier** in assortative transfer distance formula

### Phase 4: Code Quality
12. **Add `PartialEq, Eq`** derives to `FlatNode`, `FlatTree`, `Node`
13. **Add missing derives** to ALERax types, `TraversalOrder`, `RecTreeColumns`
14. **Extract shared DTL setup logic** into helper function
15. **Remove dead code** (unused parameters, unreachable variants, redundant fields)
16. **Eliminate major code duplication** (RecPhyloXML parsing, Python SVG/sampling helpers — distance parsing already extracted to `bindings_common`)

### Phase 5: Performance
17. **Replace `String::from_utf8_lossy()`** with byte-slice comparisons in RecPhyloXML parser
18. **Replace `.get(0).cloned()`** with `.into_iter().next()` in Newick parser
19. **Use reference** instead of clone in `gillespie.rs:132`
20. **Pre-build HashMap** in `build_sampled_to_original_mapping` instead of linear `.position()`

### Phase 6: Future Improvements
21. Add optional `serde` feature for core types
22. Add property-based tests (`proptest`)
23. Add stress tests with extreme DTL parameters
24. Complete the comparison module (RF distance, FlatTree support)
25. Add Python `.pyi` stub and R roxygen2 documentation
26. Support quoted names in Newick parser

---

*Report generated by 3 automated review agents examining all source files in the rustree codebase.*
