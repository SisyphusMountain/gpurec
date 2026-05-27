# Known Limitations

This page tracks constraints that should be visible before launch.

## Current Production Constraint Matrix

- `gpurec` production likelihood/gradient runs are CUDA-first; there is no
  fully featured CPU fallback.
- Backward on the retained Pi route requires `S > 256` postorder species nodes.
  Tiny species-tree fixtures can still be used for parser/config checks, but not
  as end-to-end CUDA optimization smokes.
- The retained parser accepts a deliberately narrow Newick subset:
  unquoted labels, optional branch lengths, ordinary species-tree topologies,
  and no nested comments, NHX/BEAST metadata, unary species nodes, or non-binary
  species trees.
- Native preprocessing and native backtracking are required at runtime for full
  workflow commands. Wheels that do not bundle them require compatible external
  artifacts (`GPUREC_PREPROCESS_NATIVE_LIB`, `GPUREC_BACKTRACK_BIN`) or source
  builds.
- bf16 is intentionally limited to selected direct-model experiments; it is not a
  supported workflow dtype and is not supported by the retained Pi backward
  path.
- `mode=global` is supported for shared-rate diagnostics, but it is outside the
  strict `--require-production-default-route` gate.

## Why these limitations exist

The production defaults are tuned for benchmarked CUDA behavior and reproducible
evidence. If a limitation affects your dataset, use one of the CLI gates to stop
early (`--require-cuda-backward-ready`, `--require-mode-default-optimizer`,
`--require-production-default-route`, `--require-final-check-ok`) instead of
running long optimizations that are destined to fail.
