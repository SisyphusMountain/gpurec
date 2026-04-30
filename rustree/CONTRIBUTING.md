# Contributing to rustree

## Getting Started

```bash
# Build the library
cargo build

# Run tests
cargo test

# Run clippy
cargo clippy -- -D warnings

# Build with Python bindings
cargo build --features python

# Build with R bindings
cargo build --features r
```

## Module Structure

```
src/
  lib.rs              # Crate root, re-exports
  error.rs            # RustreeError unified error type
  node/               # Tree data structures, traversal, RecTree
  newick/             # Newick parser (pest grammar)
  io/                 # RecPhyloXML parsing, CSV export
  simulation/
    bd/               # Birth-death tree simulation
    dtl/              # DTL gene tree simulation (Gillespie)
  sampling.rs         # Induced subtree extraction
  comparison.rs       # Tree topology/reconciliation comparison
  metric_functions.rs # Pairwise distances, depth computations
  robinson_foulds.rs  # Robinson-Foulds distance
  surgery.rs          # SPR moves
  induced_transfers.rs# Ghost length / induced transfer analysis
  bindings_common/    # Shared validation for Python + R
  python/             # PyO3 bindings
    training/         # ML tensor pipeline (extraction, tensors, collation)
  r/                  # extendr bindings
  external/           # External tool integration (ALERax)
```

## Module Size Limits

Keep individual source files under **1500 lines**. When a module grows beyond this:

1. Split into a directory module (`mod.rs` + submodules)
2. Use `pub(crate)` or `pub(super)` for internal contracts between submodules
3. Re-export public items from `mod.rs` to preserve the external API

Current modules near the limit should be split before adding significant new code:
- `python/gene_tree.rs` (~800 lines) -- monitor
- `python/species_tree.rs` (~870 lines) -- monitor

## Ownership Boundaries

| Area | Description |
|------|-------------|
| `node/` | Core data structures. Changes here affect everything -- test thoroughly. |
| `simulation/` | BD and DTL algorithms. Must maintain deterministic output for given seeds. |
| `python/`, `r/` | Language bindings. Must maintain API parity (see `tests/api_parity.rs`). |
| `bindings_common/` | Shared binding logic. Changes must work for both Python and R. |
| `io/` | File format parsing/writing. Preserve round-trip fidelity. |

## Coding Conventions

### Error Handling

- **Public functions** return `Result<_, RustreeError>` (or `Result<_, String>` for unmigrated modules).
- **Never panic** in library code. Use `Result` instead of `unwrap()`, `expect()`, or `assert!()`.
- Input validation uses `RustreeError::Validation`.
- Tree structure errors use `RustreeError::Tree`.
- Parse failures use `RustreeError::Parse`.
- See `src/error.rs` for the full variant list.

### Python / R Binding Parity

Both bindings should expose the same core features. When adding a new feature:

1. Implement the core logic in a non-binding module
2. Add a shared validator/helper to `bindings_common/` if needed
3. Expose in both `python/` and `r/`
4. Add a test case to `tests/api_parity.rs`

### Performance

- Run benchmarks before and after changes to comparison/sampling hotspots:
  ```bash
  cargo bench --bench comparison_sampling_benchmarks
  ```
- Use `criterion` for benchmarks (see `benches/`).
- Avoid O(n^2) or worse algorithms for operations that may run on trees with 1000+ nodes.

### Testing

- All public functions need tests.
- Use deterministic seeds for simulation tests to ensure reproducibility.
- Edge cases (single-node trees, empty inputs) are covered in `tests/edge_cases.rs`.
- API parity between Python and R is verified in `tests/api_parity.rs`.

## Pull Request Checklist

- [ ] `cargo test` passes
- [ ] `cargo clippy -- -D warnings` passes
- [ ] `cargo check --features python` passes
- [ ] `cargo check --features r` passes
- [ ] New public functions have doc comments
- [ ] No files exceed 1500 lines
- [ ] If adding a binding feature, both Python and R are updated
