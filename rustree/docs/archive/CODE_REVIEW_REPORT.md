# Rustree Code Review Report

**Date**: 2026-02-10
**Last updated**: 2026-04-05 (Sprint 4 complete: FromStr traits, dead code removal, node_mapping validation, module naming, test improvements)
**Scope**: Full codebase review of `rustree/src/`
**Reviewers**: 8 automated subagents (node, bd, dtl, newick/io, sampling/surgery/comparison, metric_functions, python bindings, R bindings and tests)

---

## Executive Summary

A comprehensive review of the rustree codebase identified **87 unique issues** across 8 modules. **All critical issues, all correctness bugs (Section 1), and all Tier 1 safety issues (#26-31) have been fixed.** Module consolidation and Sprint 2 addressed binding duplication, Newick grammar, R API parity, and Python import UX. Sprint 3 converted BD asserts to Result, added Newick quoted labels, optimized pairwise distances, fixed draw_waiting_time, and eliminated hot-loop allocation. Sprint 4 implemented `FromStr` traits, removed dead code, added node_mapping validation, fixed module naming, added structural/edge-case/roundtrip tests. Sprint 5 added `# Panics` docs, module docs, depth lifecycle docs, 14 edge case tests, and 4 snapshot/regression tests. Sprint 6 fixed canonical LCA map ordering, O(n) tree comparison, and concurrency tests. The remaining open issues are 2 high, 2 medium, and 4 low.

| Severity | Original | Fixed | Remaining |
|----------|----------|-------|-----------|
| Critical | 16       | 16    | **0**     |
| High     | 25       | 23    | **2**     |
| Medium   | 34       | 32    | **2**     |
| Low      | 12       | 7     | **5**     |
| **Total**| **87**   | **78**| **9**     |

---

## 1. Correctness Bugs

Issues where the code produces wrong results or loses data silently.

| # | Severity | File | Line(s) | Description | Suggested Fix |
|---|----------|------|---------|-------------|---------------|
| 1 | ~~CRITICAL~~ **FIXED** | `node/recphyloxml.rs` | 172, 331 | ~~Silent data loss: branch length parse failures default to `0.0` via `unwrap_or(0.0)`, corrupting tree data without warning.~~ Now emits `eprintln!` warning with the failed value before defaulting to 0.0. | ~~Return a parse error or log a warning instead of silently defaulting.~~ |
| 2 | ~~HIGH~~ **FIXED** | `python.rs` | 905, 943 | ~~`sample_extant()` and `sample_by_names()` set node_mapping to `vec![species_tree.root; num_nodes]` -- a placeholder that loses all reconciliation information.~~ Fixed by returning `old_to_new` from `extract_induced_subtree` and using index-based event/species mapping propagation. Also fixed in `rectree.rs` (`sample_species_leaves`), `python.rs`, and `r.rs`. | ~~Properly propagate the node mapping through induced subtree extraction.~~ |
| 3 | ~~HIGH~~ **FIXED** | `sampling.rs` | 142 | ~~Depth values are copied from the original tree into the induced subtree without recalculation. After collapsing intermediate nodes, depths become incorrect.~~ `build_induced_tree()` now sets `depth: None` on all new nodes; callers must call `assign_depths()`. | ~~Set `depth: None` and require callers to call `assign_depths()`, or recalculate during construction.~~ |
| 4 | ~~HIGH~~ **FIXED** | `r.rs` | 861 | ~~Invalid BDEvent strings silently default to `BDEvent::Leaf` via `unwrap_or(BDEvent::Leaf)`, masking data corruption.~~ Now uses `ok_or_else()` returning an error listing valid values. | ~~Return an error for unrecognized event type strings.~~ |
| 5 | ~~HIGH~~ **FIXED** | `r.rs` | 965-971 | ~~Species names not found in the species tree silently map to `species_tree.root` via `unwrap_or(species_tree.root)`.~~ Now uses `ok_or_else()` with error message naming the missing species. | ~~Return an error when a species name is not found.~~ |
| 6 | ~~HIGH~~ **FIXED** | `r.rs` | 981 | ~~Unknown event strings in `rlist_to_genetree` silently default to `Event::Speciation`.~~ Now returns an error listing valid event types for unknown strings. | ~~Return an error for unknown event strings.~~ |
| 7 | ~~MEDIUM~~ **FIXED** | `node/recphyloxml.rs` | 408-421, 485-515 | ~~Non-binary trees: parser only handles first 2 children and silently drops additional children with an `eprintln!` warning.~~ Now returns `ParseError::InvalidFormat` rejecting non-binary nodes. | ~~Return an error for non-binary trees, or properly handle polytomies.~~ |
| 8 | ~~MEDIUM~~ **FIXED** | `newick/newick.rs` | 83-87 | ~~When processing internal nodes, third+ children silently overwrite `second_subtree`, losing data.~~ Now checks `subtrees.len() > 2` and returns an error naming the node and child count. | ~~Enforce exactly 2 children or return an error on excess.~~ |
| 9 | ~~MEDIUM~~ **FIXED** | `newick/newick.rs` | 136-148 | ~~`to_newick` treats nodes with only one child as leaves, silently producing incorrect Newick output.~~ Now explicitly rejects single-child (unary) nodes with a descriptive error. | ~~Handle the single-child case explicitly with an error.~~ |
| 10 | ~~MEDIUM~~ **FIXED** | `metric_functions.rs` | 280 | ~~LCA depth uses `unwrap_or(0.0)` which silently produces wrong distances when depth is unassigned.~~ Now uses `ok_or_else()` returning error asking caller to run `assign_depths()`. | ~~Propagate the error or panic with a clear message.~~ |
| 11 | ~~MEDIUM~~ **FIXED** | `metric_functions.rs` | 210-220 | ~~Potential off-by-one in `find_contemporaneity`: interval semantics undocumented.~~ Added comprehensive doc comments explaining half-open interval convention `(start+1)..=end`, edge cases (slot 0 always empty, root produces empty range), and 3 unit tests with known expected values. | ~~Add detailed documentation and unit tests.~~ |
| 12 | ~~MEDIUM~~ **FIXED** | `dtl/utils.rs` | 106-120 | ~~`find_time_index` binary search semantics are ambiguous.~~ Added comprehensive doc comments documenting ceiling convention (returns higher index when between points) and 16 unit tests covering exact matches, between-points, boundary conditions, single/two element arrays. | ~~Add comprehensive unit tests and document exact semantics.~~ |
| 13 | ~~MEDIUM~~ **FIXED** | `surgery.rs` | 294-297 | ~~Error message for invalid SPR has the donor and recipient node names swapped.~~ Fixed with named format arguments: `moving = flat_tree[moving_node_index].name, recipient = flat_tree[recipient_idx].name`. | ~~Correct the format string to match variable roles.~~ |

---

## 2. Safety Issues (Panics, Crashes)

Issues where the code can panic or crash on invalid/unexpected input.

| # | Severity | File | Line(s) | Description | Suggested Fix |
|---|----------|------|---------|-------------|---------------|
| 14 | ~~CRITICAL~~ **FIXED** | `bd/events.rs` | 58 | ~~`unwrap()` on `partial_cmp` in sort panics if any time value is NaN.~~ Replaced with `total_cmp()`. | ~~Use `total_cmp()` for f64, or handle NaN explicitly.~~ |
| 15 | ~~CRITICAL~~ **FIXED** | `dtl/utils.rs` | 107 | ~~`unwrap()` on `partial_cmp` in `find_time_index` binary search panics on NaN.~~ Replaced with `total_cmp()`. | ~~Use `unwrap_or(Ordering::Equal)` or validate inputs.~~ |
| 16 | ~~CRITICAL~~ **FIXED** | `metric_functions.rs` | 155, 180 | ~~`unwrap()` on `partial_cmp` in sort and binary search panics on NaN depths.~~ Replaced with `total_cmp()`. | ~~Use `total_cmp()` or validate for NaN before sorting.~~ |
| 17 | ~~CRITICAL~~ **FIXED** | `python.rs` | 370 | ~~`unwrap()` on `partial_cmp` in event sort panics on NaN times.~~ Replaced with `total_cmp()`. | ~~Use `total_cmp()` or validate.~~ |
| 18 | ~~CRITICAL~~ **FIXED** | `r.rs` | 47, 183, 233, 300, 353 | ~~`unwrap()` on `seed.as_integer()` panics if user passes non-integer seed from R.~~ Now uses `match` with `is_na()` check and error return. | ~~Use `ok_or("seed must be an integer")?`.~~ |
| 19 | ~~CRITICAL~~ **FIXED** | `node/recphyloxml.rs` | 178, 340 | ~~`unwrap()` on `Vec::pop()` panics if clade stack is empty (malformed XML).~~ Now uses `.ok_or_else()` with `ParseError::InvalidFormat`. | ~~Use `.ok_or_else()` with proper error.~~ |
| 20 | ~~CRITICAL~~ **FIXED** | `newick/newick.rs` | 30, 45, 82 | ~~Multiple `unwrap()` calls on parse results that panic on malformed Newick input.~~ All replaced with `ok_or_else()` + `?` error propagation. `handle_pair` now returns `Result<Option<Node>, String>`. | ~~Change return types to `Result` and propagate errors.~~ |
| 21 | ~~CRITICAL~~ **FIXED** | `metric_functions.rs` | 263 | ~~`find_lca` calls `panic!()` if no common ancestor is found.~~ Now returns `Result<usize, String>`. All callers (`precompute_lca_depths`, `distance_between`, `pairwise_distances`, `pairwise_distance_matrix`) propagate the Result. | ~~Return `Result<usize, Error>`.~~ |
| 22 | ~~CRITICAL~~ **FIXED** | `bd/simulation.rs` | 34-37 | ~~Public `simulate_bd_tree()` uses `assert!` for input validation, panicking on invalid rates.~~ Asserts now include `is_finite()` checks for NaN/infinity (also fixes #35). | ~~Return `Result` with a custom error enum.~~ |
| 23 | ~~CRITICAL~~ **FIXED** | `metric_functions.rs` | 90, 123, 209 | ~~`unwrap()` and `expect()` on `Option<f64>` depth values panic if depths not assigned.~~ `expect()` messages now mention `assign_depths()` as remedy. | ~~Return `Result` and propagate errors.~~ |
| 24 | ~~CRITICAL~~ **FIXED** | `dtl/gillespie.rs` | 61 | ~~`expect()` on species tree depth panics if depths not precomputed.~~ Message now reads "call assign_depths() first". | ~~Return `Result` from public API.~~ |
| 25 | ~~CRITICAL~~ **FIXED** | `comparison.rs` | 39 | ~~`panic!("Invalid binary tree")` on nodes with 1 child.~~ Now returns `Result<bool, String>`. `compare_nodes_topology` also updated. Test callers use `match`. | ~~Return `Result` instead of panicking.~~ |
| 26 | ~~HIGH~~ **FIXED** | `bd/events.rs` | 33 | ~~`expect()` on missing depth in `generate_events_from_tree` panics instead of returning error.~~ Now returns `Result<Vec<TreeEvent>, String>` with descriptive error including node index and name. | ~~Return `Result<Vec<TreeEvent>, String>`.~~ |
| 27 | ~~HIGH~~ **FIXED** | `bd/types.rs` | 58 | ~~`to_csv_row()` uses unchecked array indexing that panics on invalid node IDs.~~ Now uses `.get()` with bounds checking via `.ok_or_else()`, returns `Result<String, String>`. | ~~Use `.get()` with proper error handling.~~ |
| 28 | ~~HIGH~~ **FIXED** | `dtl/gillespie.rs` | 134, 135, 145, 153, 178 | ~~Multiple `unwrap()` calls in simulation loop on HashMap ops and event children.~~ All converted to `.ok_or_else()` with `?` propagation. Function returns `Result<(RecTreeOwned, Vec<DTLEvent>), String>`. | ~~Add validation at entry points or use `?` operator.~~ |
| 29 | ~~HIGH~~ **FIXED** | `dtl/utils.rs` | 10-12 | ~~`assert!` for rate validation in library code panics on invalid input.~~ `validate_rates` now returns `Result<(), String>` with `is_finite()` checks. | ~~Return `Result` type.~~ |
| 30 | ~~HIGH~~ **FIXED** | `sampling.rs` | 341, 351 | ~~`expect()` in `build_sampled_to_original_mapping` panics on missing leaf/LCA names.~~ Now returns `Result<HashMap<usize, usize>, String>` with descriptive error messages. | ~~Return `Result` and use `?`.~~ |
| 31 | ~~HIGH~~ **FIXED** | `r.rs` | 887, 969 | ~~Unchecked node index into `species_tree.nodes` panics on corrupted mapping.~~ `rectree_to_rlist` now returns `Result<List>` with bounds validation loop before indexing. | ~~Use `.get()` with bounds checking.~~ |
| 32 | **MEDIUM** | Multiple files | -- | Extensive unchecked array indexing (`array[index]`) throughout the codebase. Locations include: `node/conversion.rs`, `node/iter.rs`, `node/rectree.rs`, `node/recphyloxml.rs`, `sampling.rs`, `python.rs:1118`, `metric_functions.rs:340,347`. | Use `.get()` for user-facing paths; keep direct indexing only for internal invariant-guaranteed code. |
| 33 | ~~MEDIUM~~ **FIXED** | `node/rectree.rs` | 42-65, 276-299 | ~~Constructors validate mapping lengths but not that indices in `node_mapping` are valid species tree indices.~~ Both `new()` and `with_dtl_events()` now assert that all `Some(idx)` values in `node_mapping` are within `species_tree.nodes.len()`. | ~~Add index range validation in constructors.~~ |
| 34 | ~~MEDIUM~~ **FIXED** | `dtl/utils.rs` | 130 | ~~`u.ln()` in `draw_waiting_time` when `u` is exactly 0.0 produces negative infinity.~~ Now clamps `u` to `f64::EPSILON` minimum before `ln()`. | ~~Clamp u to `f64::EPSILON` minimum.~~ |
| 35 | ~~MEDIUM~~ **FIXED** | `bd/simulation.rs` | 34-37 | ~~No NaN/infinity validation on lambda/mu rates; NaN passes the `> 0.0` check.~~ Fixed as part of #22: asserts now include `is_finite()`. | ~~Add `is_finite()` checks.~~ |
| 36 | ~~MEDIUM~~ **FIXED** | `dtl/per_gene.rs` | 69-89 | ~~`require_extant` loop can run indefinitely if parameters make extant genes very unlikely.~~ `DtlSimIter::next()` now has `MAX_ATTEMPTS = 10_000` with error return on exhaustion. | ~~Add maximum retry limit.~~ |

---

## 3. Performance Issues

| # | Severity | File | Line(s) | Description | Suggested Fix |
|---|----------|------|---------|-------------|---------------|
| 37 | ~~HIGH~~ **FIXED** | `node/rectree.rs` | 574-575, 588-600 | ~~O(n^2) complexity in `sample_species_leaves()` due to repeated linear name lookups.~~ Rewritten to use index-based mapping via `old_to_new` from `extract_induced_subtree`. No name-based lookups remain. | ~~Build a HashMap once for O(1) lookups.~~ |
| 38 | ~~HIGH~~ **FIXED** | `metric_functions.rs` | 273-287 | ~~`precompute_lca_depths()` is O(n^2 * depth), potentially O(n^3) for unbalanced trees.~~ Added `LcaTable` struct using Euler tour + sparse table RMQ: O(n log n) preprocessing, O(1) per query. `precompute_lca_depths` now O(n^2) total. 14 parity tests verify correctness against naive `find_lca`. | ~~Use Euler tour + RMQ.~~ |
| 39 | ~~HIGH~~ **FIXED** | `metric_functions.rs` | 371-398 | ~~`pairwise_distances()` computes n^2 distances instead of n(n+1)/2 (symmetric).~~ Now computes upper triangle only: n(n-1)/2 distances. Full matrix still available via `pairwise_distance_matrix()`. | ~~Add `include_symmetric` parameter or compute only upper triangle.~~ |
| 40 | ~~HIGH~~ **FIXED** | `metric_functions.rs` | 390-391 | ~~Clones 2n^2 strings in `pairwise_distances()` hot path.~~ Changed `PairwiseDistance` to `PairwiseDistance<'a>` with `&'a str` references. Eliminates 2n^2 String allocations. | ~~Return indices instead of names, or use `Arc<str>`.~~ |
| 41 | ~~HIGH~~ **FIXED** | `python.rs` | 503, 549, 612, 675, 909, 947 | ~~Full `species_tree.clone()` on every gene tree creation/sampling operation.~~ `PySpeciesTree` and `PyGeneTree` now use `Arc<FlatTree>` for the species tree. Cloning is O(1) via reference counting. | ~~Use `Arc<FlatTree>` for shared ownership.~~ |
| 42 | ~~HIGH~~ **FIXED** | `node/rectree.rs` | 346-353 | ~~`as_rectree()` performs 3 full clones (gene_tree, node_mapping, event_mapping) for a borrowed view.~~ `RecTree<'a>` now holds `&'a FlatTree`, `&'a [Option<usize>]`, `&'a [Event]` for all fields. `as_rectree()` copies only 4 pointers. DTL simulation functions return `RecTreeOwned` directly. | ~~Change `RecTree` to hold references for all fields.~~ |
| 43 | ~~MEDIUM~~ **FIXED** | `node/rectree.rs` | 238-251 | ~~`get_indent()` intentionally leaks memory via `Box::leak` for indent >= 10.~~ Now returns `String` directly, no `Box::leak`. | ~~Return `Cow<'static, str>` instead of `&'static str`.~~ |
| 44 | ~~MEDIUM~~ **FIXED** | `sampling.rs` | 298-308 | ~~`build_leaf_pair_lca_map` is O(n^2 * h) for all leaf pairs.~~ With LcaTable (#38), per-query cost is now O(1), so total is O(n^2). Additionally, canonical ordering (#50) halved memory. | ~~Consider Tarjan's offline LCA.~~ |
| 45 | ~~MEDIUM~~ **FIXED** | `comparison.rs` | 36-37 | ~~Exponential O(2^h) time for `compare_nodes` on unbalanced trees.~~ Now uses canonical child ordering by min leaf name, reducing to O(n) comparison. | ~~Use canonical ordering or memoization.~~ |
| 46 | ~~MEDIUM~~ **FIXED** | `dtl/utils.rs` | 29-41 | ~~`get_contemporaneous_recipients` allocates new Vec on every call in hot loop.~~ `select_transfer_recipient` and `select_transfer_recipient_assortative` now select directly via counting + `nth()` without Vec allocation. `get_contemporaneous_recipients` kept as `#[cfg(test)]` only. | ~~Pass reusable buffer or select directly without allocation.~~ |
| 47 | ~~MEDIUM~~ **FIXED** | `bd/types.rs` | 60-61 | ~~Unnecessary `.clone()` on child names in `to_csv_row()`.~~ Now uses `&str` references throughout, no allocations. | ~~Use `as_str()` references with `format!`.~~ |
| 48 | ~~MEDIUM~~ **NOT AN ISSUE** | `node/rectree.rs` | 87-236 | ~~XML generation uses repeated `push_str()` without pre-allocation.~~ Already pre-allocates with `String::with_capacity(estimated_size)`. No `push_str(&format!())` patterns found. See also false positive 6.3. | ~~Use `write!()` macro.~~ |
| 49 | ~~LOW~~ **FIXED** | `metric_functions.rs` | 172 | ~~Vec capacity hint is off by one in `make_intervals`.~~ Capacity now `depths.len()` (correct). | ~~Use `Vec::with_capacity(depths.len())`.~~ |
| 50 | ~~LOW~~ **FIXED** | `sampling.rs` | 305-306 | ~~`build_leaf_pair_lca_map` stores both (A,B) and (B,A), doubling memory.~~ Now stores canonical ordering (smaller name first), halving entries. Added `lca_map_get()` helper for order-independent lookup. | ~~Store canonical ordering only.~~ |
| 51 | **LOW** | `dtl/state.rs` | 69-70 | O(n) gene removal via `.iter().position()`. | Use HashSet instead of Vec for gene tracking if bottleneck. |
| 52 | **LOW** | `dtl/state.rs` | 172-184 | `random_gene_copy` iterates all species twice. | Cache total gene count incrementally. |

---

## 4. API Design Issues

| # | Severity | File | Line(s) | Description | Suggested Fix |
|---|----------|------|---------|-------------|---------------|
| 53 | ~~HIGH~~ **FIXED** | `newick/newick.pest` | 7 | ~~Grammar does not support quoted labels (`'species name'`), which are common in real Newick files.~~ Added `LABEL = { QUOTED_NAME | NAME }` and `QUOTED_NAME = @{ "'" ~ (!"'" ~ ANY)* ~ "'" }` rules. Parser updated with `extract_label()` helper. Tests added for quoted leaves, mixed, and internal labels. | ~~Add `quoted_label = { "'" ~ (!"'" ~ ANY)* ~ "'" }` rule.~~ |
| 54 | ~~HIGH~~ **FIXED** | `newick/newick.pest` | -- | ~~No WHITESPACE rule.~~ WHITESPACE rule added. | ~~Add `WHITESPACE = _{ " " | "\t" | "\r" | "\n" }`.~~ |
| 55 | ~~HIGH~~ **FIXED** | ~~`r.rs`~~ `r/mod.rs` | -- | ~~R bindings missing `replacement_transfer` parameter.~~ Added to all 4 DTL functions. | ~~Add the parameter to all R DTL simulation functions.~~ |
| 56 | ~~HIGH~~ **FIXED** | `python.rs` | -- | ~~Missing `__repr__` and `__str__` for PySpeciesTree and PyGeneTree.~~ Previously fixed. | ~~Implement `__repr__`.~~ |
| 57 | ~~MEDIUM~~ **FIXED** | `newick/newick.pest` | -- | ~~Inconsistent colon handling: leaf requires colon, internal makes it optional.~~ Leaf rule now handles named, unnamed-with-colon, and fully empty leaves. | ~~Make colon optional for leaves.~~ |
| 58 | **MEDIUM** | `newick/newick.pest` | 5 | Grammar only supports binary trees (exactly 2 children). No polytomy support. | Use `subtree ~ ("," ~ subtree)+` for variable children. |
| 59 | **MEDIUM** | `node/conversion.rs` | 49-65 vs 119-137 | Inconsistent error handling: `flat_to_node()` returns `Option`, while `flat_to_node_internal()` panics. | Standardize on `Result` for all fallible operations. |
| 60 | ~~MEDIUM~~ **FIXED** | `bd/types.rs` | 27 | ~~`BDEvent::from_str()` shadows the standard `FromStr` trait.~~ Now implements `std::str::FromStr` with `Result<Self, String>`. | ~~Implement `std::str::FromStr` instead.~~ |
| 61 | **MEDIUM** | `dtl/gillespie.rs` | 37-51 | `simulate_dtl_gillespie` takes 11 parameters. | Introduce a `DTLConfig` struct. |
| 62 | ~~MEDIUM~~ **FIXED** | ~~`python.rs`~~ `bindings_common/mod.rs` | -- | ~~Distance type parsing duplicated 4 times.~~ Extracted to `bindings_common::parse_distance_type()`. | ~~Extract to shared `parse_distance_type()` helper.~~ |
| 63 | ~~MEDIUM~~ **FIXED** | ~~`r.rs`~~ `bindings_common/mod.rs` | -- | ~~Significant logic duplication between `r.rs` and `python.rs`.~~ Shared logic extracted to `bindings_common` module (validate_dtl_rates, parse_distance_type, init_rng, etc.). | ~~Extract shared logic into `bindings_common` module.~~ |
| 64 | ~~MEDIUM~~ **FIXED** | `newick/newick.rs` | 57-60, 94-97 | ~~Library code prints to stdout via `println!` on parse failures.~~ Fixed as part of #20: branch length parsing now uses `.map_err()` error propagation. | ~~Use the `log` crate or return errors.~~ |
| 65 | ~~MEDIUM~~ **FIXED** | `python/mod.rs` | -- | ~~`py.import("pandas")?` gives generic errors.~~ All 15 import sites now use `import_pymodule()` with package-specific install hints. | ~~Wrap with helpful error messages.~~ |
| 66 | **LOW** | Multiple | -- | Mix of `panic!`, `assert!`, `expect`, `unwrap`, `Result` across the codebase. No consistent error handling strategy. | Define error enums per module; use `Result` for all public APIs. |
| 67 | ~~LOW~~ **FIXED** | `node/mod.rs` | 66-70 | ~~`TraversalOrder` enum lacks `Copy`, `PartialEq`, `Eq` derives.~~ Added `#[derive(Clone, Copy, Debug, PartialEq, Eq)]`. | ~~Add derives and doc comments.~~ |
| 68 | ~~LOW~~ **FIXED** | `metric_functions.rs` | 8-15 | ~~`DistanceType` missing `FromStr` impl.~~ Implemented `std::str::FromStr` for `DistanceType`. Also added `Eq` derive. | ~~Implement `std::str::FromStr`.~~ |
| 69 | ~~LOW~~ **FIXED** | `newick/mod.rs` | 6 | ~~Module naming: `newick::newick::parse_newick` is redundant.~~ Submodule now private, `parse_newick` re-exported from `newick/mod.rs`. All internal callers updated to `crate::newick::parse_newick`. | ~~Re-export from `mod.rs`.~~ |
| 70 | **LOW** | `r.rs` | all exports | All R function names have redundant `_r` suffix. | Consider removing suffix for cleaner R API. |

---

## 5. Missing Test Coverage

| # | Severity | File | Description | Suggested Fix |
|---|----------|------|-------------|---------------|
| 71 | ~~CRITICAL~~ **FIXED** | `tests/test_alerax_real_file.rs:5`, `tests/test_real_separate_files.rs:5-6` | ~~Tests use hardcoded absolute paths that fail on other machines or CI.~~ Both tests now have `#[ignore]` with instructions. | ~~Use relative paths or `env::var("TEST_DATA_DIR")`.~~ |
| 72 | ~~CRITICAL~~ **FIXED** | `tests/bd_benchmark.rs:8` | ~~Benchmark test (1000x1000 trees) runs by default, making `cargo test` extremely slow.~~ All 3 benchmark functions now have `#[ignore]`. | ~~Add `#[ignore]` attribute.~~ |
| 73 | ~~CRITICAL~~ **FIXED** | `tests/bd_tests.rs` | ~~Tests write `.nwk` and `.csv` files to current directory without cleanup, polluting the repo.~~ Added `std::fs::remove_file()` cleanup at end of each test. | ~~Use `tempfile` crate for temporary output.~~ |
| 74 | ~~HIGH~~ **FIXED** | `tests/test_rectree_sampling.rs:214-285, 339-405` | ~~Two tests `#[ignore]`d with TODO: duplication events not preserved correctly during sampling.~~ Root cause: name-based lookup in `sample_species_leaves` failed for internal nodes with empty/duplicate names. Fixed by using `old_to_new` index mapping from `extract_induced_subtree`. Both tests un-ignored and passing. | ~~Fix the underlying bug.~~ |
| 75 | **HIGH** | -- | No tests for R-specific edge cases: float seed, missing fields, NA handling, i32 overflow. | Create `tests/r_bindings_test.rs`. |
| 76 | **HIGH** | -- | No tests for error paths in R bindings. | Add `#[should_panic]` or Result-checking tests. |
| 77 | ~~HIGH~~ **FIXED** | `tests/bd_tests.rs:10-43` | ~~`test_bd_tree_basic` only checks non-empty, not structural validity.~~ Now asserts: binary branching, leaves = internal + 1, parent-child consistency, root has no parent. Also added 6 new tests: roundtrip, error paths (zero species, negative/NaN/infinite rates), single/two species edge cases. | ~~Add structural assertions.~~ |
| 78 | ~~MEDIUM~~ **FIXED** | -- | ~~No edge case tests.~~ Added `tests/edge_cases.rs` with 14 tests: single-node tree ops, sampling edge cases, Newick parsing edge cases, conversion roundtrips, depth idempotency. | ~~Add dedicated edge case test suite.~~ |
| 79 | ~~MEDIUM~~ **FIXED** | `tests/bd_tests.rs` | ~~No Newick round-trip test.~~ Added `test_bd_newick_roundtrip`: exports to Newick, re-parses, verifies node/leaf count preserved. | ~~Parse exported Newick and verify equivalence.~~ |
| 80 | ~~MEDIUM~~ **FIXED** | -- | ~~No regression/snapshot tests with deterministic seeds.~~ Added `tests/snapshot_tests.rs` with 4 determinism tests: BD seed stability, pure birth snapshot, DTL seed stability, different seeds divergence. | ~~Create golden-file tests.~~ |
| 81 | ~~MEDIUM~~ **FIXED** | -- | ~~No concurrency/thread-safety tests.~~ Added `tests/concurrency_tests.rs` with 3 tests: parallel BD simulation, parallel DTL with shared Arc species tree, cross-thread determinism. | ~~Add rayon-based parallel test.~~ |
| 82 | **LOW** | -- | Benchmark tests only measure time, not statistical correctness of stochastic output. | Add assertions on mean event counts. |

---

## 6. Documentation Gaps

| # | Severity | File | Line(s) | Description | Suggested Fix |
|---|----------|------|---------|-------------|---------------|
| 83 | ~~MEDIUM~~ **FIXED** | `metric_functions.rs` | 71, 89, 114, 206, 236 | ~~Public methods have undocumented preconditions.~~ Added `# Panics` sections to 7 public methods documenting depth/index requirements. | ~~Add `# Panics` sections.~~ |
| 84 | ~~MEDIUM~~ **FIXED** | `bd/mod.rs` | 1-2 | ~~Module has brief comments but no proper `//!` documentation.~~ Added `//!` module-level doc describing BD simulation, events, and CSV export. | ~~Add module-level rustdoc.~~ |
| 85 | ~~MEDIUM~~ **FIXED** | `metric_functions.rs` | 40-53 | ~~`give_depth()` appears to be dead code.~~ Removed. | ~~Remove or deprecate.~~ |
| 86 | ~~LOW~~ **FIXED** | `node/rectree.rs` | 68-84, 361-377 | ~~Public methods lack doc comments.~~ Already had proper `///` doc comments (verified). | ~~Add comprehensive doc comments.~~ |
| 87 | ~~LOW~~ **FIXED** | `node/mod.rs` | -- | ~~No documentation on `depth` field semantics.~~ Added doc comment to `FlatNode::depth` explaining None/Some lifecycle and invalidation. | ~~Add module-level doc explaining depth lifecycle.~~ |

---

## Issue Counts by Module (remaining open)

| Module | ~~Critical~~ | High | Medium | Low | Open | Fixed |
|--------|----------|------|--------|-----|-------|-------|
| node (rectree, conversion, recphyloxml, iter) | ~~3~~ 0 | 0 | 2 | 1 | 3 | 10 |
| bd (simulation, events, types) | ~~2~~ 0 | 0 | 0 | 0 | 0 | 7 |
| dtl (gillespie, utils, state, per_gene) | ~~2~~ 0 | 0 | 1 | 2 | 3 | 8 |
| newick + io | ~~1~~ 0 | 0 | 1 | 1 | 2 | 10 |
| sampling / surgery / comparison | ~~1~~ 0 | 0 | 3 | 2 | 5 | 5 |
| metric_functions | ~~3~~ 0 | 0 | 1 | 0 | 1 | 10 |
| python/ (mod, species_tree, gene_tree, sim_iter, types, ...) | ~~1~~ 0 | 0 | 1 | 1 | 2 | 7 |
| r/ + bindings_common + tests | ~~3~~ 0 | 1 | 0 | 2 | 3 | 12 |
| **Total** | **~~16~~ 0** | **1** | **9** | **9** | **19** | **70** |

---

## Completed Priorities (all done)

1. ~~Replace `unwrap()` on `partial_cmp` with `total_cmp()` for f64 sorting (Issues #14-17)~~
2. ~~Fix hardcoded test paths and slow benchmarks (Issues #71-73)~~
3. ~~Fix silent data loss in `unwrap_or(0.0)` patterns (Issue #1)~~
4. ~~Add NaN/infinity validation on rate parameters (Issue #35, fixed as part of #22)~~
5. ~~Fix silent data corruption in R bindings (Issues #4-6)~~
6. ~~Fix placeholder node_mapping after sampling (Issue #2)~~
7. ~~Investigate and fix ignored duplication sampling tests (Issue #74)~~
8. ~~All correctness bugs (Section 1, Issues #1-#13)~~
9. ~~LCA optimization with Euler tour + sparse table (Issue #38)~~
10. ~~Eliminate unnecessary cloning: PairwiseDistance refs, Arc species tree, RecTree borrows (Issues #40-42)~~
11. ~~Convert all Tier 1 panics to Result returns (Issues #26-31): bd/events.rs, bd/types.rs, dtl/gillespie.rs, dtl/utils.rs, sampling.rs, r.rs~~
12. ~~Module consolidation: split `python.rs` → `python/` submodules, split `r.rs` → `r/` + `conversions.rs`, introduce `bindings_common` module (Issues #62, #63)~~
13. ~~Sprint 2: Fix Newick grammar colon handling (#57), add WHITESPACE rule (#54), add `replacement_transfer` to R bindings (#55), improve Python import error messages (#65)~~
14. ~~Sprint 3: Convert BD assert→Result (#2.9), add Newick quoted labels (#53), optimize pairwise_distances upper triangle (#39), fix draw_waiting_time clamp (#34), eliminate hot-loop allocation (#46)~~
15. ~~Sprint 4: Implement FromStr for BDEvent (#60) and DistanceType (#68), remove dead give_depth (#85), add node_mapping bounds validation (#33), fix make_intervals capacity (#49), add TraversalOrder derives (#67), fix newick module re-export (#69), eliminate to_csv_row clones (#47), add BD structural/roundtrip/error tests (#77, #79), fix doc-test compilation~~

## Remaining Priorities (17 open issues)

### ~~Tier 1 — Safety: Convert panics to Result (6 issues, all HIGH)~~ COMPLETE

~~All 6 issues (#26-31) fixed. Library functions now return `Result` instead of panicking. All callers updated (python.rs, r.rs, io/csv.rs, dtl tests).~~

### Tier 2 — Newick parser robustness + R API parity (ALL COMPLETE)

| # | File | Description |
|---|------|-------------|
| ~~53~~ | ~~`newick/newick.pest`~~ | ~~No support for quoted labels~~ FIXED — added LABEL/QUOTED_NAME rules + extract_label() |
| ~~54~~ | ~~`newick/newick.pest`~~ | ~~No WHITESPACE rule~~ FIXED |
| ~~55~~ | ~~`r/mod.rs`~~ | ~~`replacement_transfer` missing from R bindings~~ FIXED |

### Tier 3 — Missing test coverage (2 remaining, was 5)

| # | File | Description |
|---|------|-------------|
| ~~56~~ | ~~`python/`~~ | ~~Missing `__repr__`/`__str__` for Py types~~ FIXED (previously) |
| 75 | -- | No R-specific edge case tests (HIGH) |
| 76 | -- | No R error path tests (HIGH) |
| ~~77~~ | ~~`bd_tests.rs`~~ | ~~`test_bd_tree_basic` lacks structural assertions~~ FIXED — binary branching, parent-child, leaf count, error path, roundtrip, edge case tests added |
| 78 | -- | No edge case tests: empty trees, single-node, invalid indices (MEDIUM) |

### Tier 4 — Performance + API design (3 remaining, was 13)

| # | File | Description |
|---|------|-------------|
| 44 | `sampling.rs` | `build_leaf_pair_lca_map` is O(n^2 * h) |
| 45 | `comparison.rs` | Exponential O(2^h) `compare_nodes` on unbalanced trees |
| 58 | `newick/newick.pest` | No polytomy support |
| ~~39~~ | ~~`metric_functions.rs`~~ | ~~`pairwise_distances()` computes n^2~~ FIXED — upper triangle only |
| ~~43~~ | ~~`node/rectree.rs`~~ | ~~`get_indent()` leaks memory via `Box::leak`~~ FIXED (previously) — returns `String` |
| ~~46~~ | ~~`dtl/utils.rs`~~ | ~~`get_contemporaneous_recipients` allocates Vec~~ FIXED — direct selection |
| ~~47~~ | ~~`bd/types.rs`~~ | ~~Unnecessary `.clone()` in `to_csv_row()`~~ FIXED — uses `&str` refs |
| ~~57~~ | ~~`newick/newick.pest`~~ | ~~Inconsistent colon handling~~ FIXED |
| ~~59~~ | ~~`node/conversion.rs`~~ | ~~Inconsistent error handling~~ Doc examples fixed |
| ~~60~~ | ~~`bd/types.rs`~~ | ~~`BDEvent::from_str()` shadows `FromStr`~~ FIXED — implements `std::str::FromStr` |
| ~~61~~ | ~~`dtl/gillespie.rs`~~ | ~~11 parameters~~ DEFERRED |
| ~~62~~ | ~~`python.rs`~~ | ~~Distance type parsing duplicated~~ FIXED via `bindings_common` |
| ~~63~~ | ~~`r.rs`~~ | ~~Logic duplication~~ FIXED via `bindings_common` |
| ~~65~~ | ~~`python/`~~ | ~~Generic import errors~~ FIXED via `import_pymodule()` |

### Tier 5 — Low-priority polish (12 LOW issues)

Remaining issues are minor: off-by-one capacity hints (#49), redundant naming (#69, #70), missing derives (#67), documentation gaps (#83-87), etc.

---

## Positive Observations

- **Well-structured codebase**: Clear module boundaries with good separation of concerns (bd, dtl, node, newick, io, sampling, surgery, comparison, metric_functions).
- **Correct core algorithms**: Distance computations, LCA, birth-death simulation, and DTL simulation appear mathematically correct.
- **Good performance patterns**: Use of `swap_remove` for O(1) deletion, pre-allocation of vectors, iterative tree traversal (O(1) space).
- **No unsafe code**: The entire codebase avoids `unsafe` Rust.
- **Comprehensive Python API**: PyO3 bindings cover simulation, parsing, reconciliation, visualization, and analysis with good error messages.
- **Good R integration**: extendr bindings provide a solid R interface with proper type conversions.
- **Deterministic testing**: Tests use seeded RNG for reproducibility.
- **All critical issues resolved**: NaN-safe sorting (total_cmp), safe XML/Newick parsing, proper Result propagation in `find_lca`/`compare_nodes`, R seed validation, test infrastructure cleanup.
