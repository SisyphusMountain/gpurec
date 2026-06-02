# Rust Crates

This directory contains Rust native extensions used by the Python GPU
reconciliation workflow. Both crates build `cdylib` Python extension modules
through PyO3.

## Entries

- `gpurec-preprocess/`: preprocessing extension for parsing species and gene
  trees, extracting clade conditional probability data, planning family batches,
  scheduling clade waves, and emitting GPU-friendly layout metadata.
- `gpurec-backtrack/`: backtracking extension for sampling reconciliations from
  dynamic-programming tensors and species topology arrays.

Each crate keeps its own `Cargo.toml`, `Cargo.lock`, and `src/` tree so the
native modules can be built or iterated independently.

Source installs of the Python package build both crates through
`setuptools-rust`:

```bash
python -m pip install .
```

Direct Cargo builds remain useful for local development and are used as a
source-checkout fallback by the Python loaders:

```bash
cargo build --release --manifest-path crates/gpurec-preprocess/Cargo.toml
cargo build --release --manifest-path crates/gpurec-backtrack/Cargo.toml
```
