# Rust Preprocessing Port Progress

Goal: replace the C++ preprocessing implementation with Rust while preserving
output parity and matching or improving preprocessing runtime.

Current Rust slice:

- `crates/gpurec-preprocess` implements a Rust preprocessing library and JSON
  CLI.
- Species and gene Newick parsing use the retained-subset parser so the Rust
  path matches the current C++ support for optional branch-length spellings,
  multifurcations, and deterministic right-binarization.
- The implemented path emits the compact production arrays used by
  `GeneDataset`: species postorder helpers, leaf row/column mapping,
  `split_counts`, `split_parents_sorted`, `split_leftrights_sorted`,
  natural-log `log_split_probs_sorted`, segment counts, root id, and leaf
  labels.
- The Rust JSON request accepts `num_threads`; positive values run family
  preprocessing inside a Rayon thread pool, while `0` uses Rayon's default
  worker configuration.
- `gpurec.core.preprocess_rust.RustPreprocessExtension` exposes the native
  PyO3 Rust extension behind the same raw Python contract as the C++ pybind
  extension. It returns Torch tensors directly through NumPy-backed PyO3
  arrays. Setting `GPUREC_PREPROCESS_BACKEND=rust` makes `GeneDataset` use
  that adapter.
- Positive `num_threads` values use cached Rayon thread pools keyed by worker
  count, so repeated preprocessing calls keep honoring the requested CPU core
  count without rebuilding the pool every call.
- `profiling/bench_preprocess_rust_vs_cpp.py` compares the current C++ pybind
  path with Rust preprocessing, Rust binary output, and the Python adapter.
- `tests/integration/test_rust_preprocess_parity.py` compares the Rust output
  with the current C++ pybind output for a binary multi-record family and a
  multifurcating gene-tree family with species matrices enabled and a threaded
  Rust request. It also covers a multi-family request with branch lengths and
  the Python adapter's C++-shaped raw output.

Latest local timing from `profiling/bench_preprocess_rust_vs_cpp.py` on
`tests/data/hogenom_bench` (1,055 families, 8 threads, 9 repeats, species
matrices disabled):

- C++ pybind median: `0.040163 s`.
- Rust preprocessing with output discarded: `0.032559 s`.
- Rust CLI sparse-binary output median: `0.086972 s` for `20,626,857` bytes.
- Rust native Python adapter median: `0.037159 s`.
- Rust subprocess adapter median: `0.168274 s`.

Latest local timing with species matrices enabled on the same fixture
(7 repeats):

- C++ pybind median: `0.048253 s`.
- Rust preprocessing with output discarded: `0.040924 s`.
- Rust CLI sparse-binary output median: `0.160685 s` for `48,716,873` bytes.
- Rust native Python adapter median: `0.040590 s`.
- Rust subprocess adapter median: `0.249493 s`.

Interpretation: the native Rust adapter now matches the C++-shaped Python
output contract and is slightly faster than the C++ pybind path on the local
HOGENOM benchmark. The subprocess/binary-output adapter remains useful for CLI
experiments but is not the production-performance path.

Known remaining work before replacement:

- Add parity coverage for mapping error behavior, malformed inputs, broader
  branch-length edge cases, and HOGENOM-style fixtures.
- Add packaging/install integration for the Rust extension before making Rust
  the default backend.
- Decide when to retire or demote the C++ extension after the Rust extension is
  available in normal installs.
