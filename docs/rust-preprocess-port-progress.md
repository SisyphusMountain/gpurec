# Rust Preprocessing Port Progress

Goal: keep preprocessing on the Rust implementation while preserving the raw
Python contract and matching or improving preprocessing runtime.

Current Rust slice:

- `crates/gpurec-preprocess` implements a Rust preprocessing library and JSON
  CLI.
- Species and gene Newick parsing use the retained-subset Rust parser with
  optional branch-length spellings, multifurcations, and deterministic
  right-binarization.
- The implemented path emits the compact production arrays used by
  `GeneDataset`: species postorder helpers, leaf row/column mapping,
  `split_counts`, `split_parents_sorted`, `split_leftrights_sorted`,
  natural-log `log_split_probs_sorted`, segment counts, root id, and leaf
  labels.
- The Rust JSON request accepts `num_threads`; positive values run family
  preprocessing inside a Rayon thread pool, while `0` uses Rayon's default
  worker configuration.
- `gpurec.core.preprocess_rust.RustPreprocessExtension` exposes the native
  PyO3 Rust extension behind the raw Python contract used by `GeneDataset`. It
  returns Torch tensors directly through NumPy-backed PyO3 arrays.
- Positive `num_threads` values use cached Rayon thread pools keyed by worker
  count, so repeated preprocessing calls keep honoring the requested CPU core
  count without rebuilding the pool every call.
- `profiling/bench_preprocess_rust.py` compares Rust preprocessing, Rust binary
  output, the native adapter, and the subprocess adapter.
- `tests/integration/test_rust_preprocess_parity.py` compares Rust CLI JSON
  output with the native adapter for a binary multi-record family, a
  multifurcating gene-tree family with species matrices enabled, and a
  multi-family request with signed, decimal, and exponent branch-length
  spellings. It also covers native-vs-subprocess adapter raw output parity plus
  mapping-error and malformed-Newick error behavior. The tracked
  `tests/fixtures/alerax_hogenom_style` fixture exercises HOGENOM-style AleRax
  family records with relative tree/mapping paths, explicit mappings, multiple
  families, multiple tree records, and species matrices without requiring the
  private HOGENOM dataset.
- `gpurec preprocess-check` validates the native PyO3 preprocessing extension or
  source-tree Cargo build fallback without reading dataset files, and the release
  smoke checks both the installed-wheel missing-extension diagnostic and the
  success path with `GPUREC_PREPROCESS_NATIVE_LIB`.

Latest local timing from the preprocessing benchmark on `tests/data/hogenom_bench`
(1,055 families, 8 threads, 9 repeats, species matrices disabled):

- Rust preprocessing with output discarded: `0.032559 s`.
- Rust CLI sparse-binary output median: `0.086972 s` for `20,626,857` bytes.
- Rust native Python adapter median: `0.037159 s`.
- Rust subprocess adapter median: `0.168274 s`.

Latest local timing with species matrices enabled on the same fixture
(7 repeats):

- Rust preprocessing with output discarded: `0.040924 s`.
- Rust CLI sparse-binary output median: `0.160685 s` for `48,716,873` bytes.
- Rust native Python adapter median: `0.040590 s`.
- Rust subprocess adapter median: `0.249493 s`.

Interpretation: the native Rust adapter matches the raw Python output contract
and is the production path. The subprocess/binary-output adapter remains useful
for CLI experiments but is not the production-performance path.

Known remaining work before replacement:

- None for the raw preprocessing replacement path. Remaining redistribution work
  is release-policy work such as adding a project license and choosing whether
  future platform wheels should bundle native artifacts.
