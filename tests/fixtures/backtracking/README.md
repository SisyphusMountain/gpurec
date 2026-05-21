# Backtracking JSON Fixture Contract

`speciation.json` is a hand-authored CPU-only smoke fixture for the Rust
backtracking CLI.  It validates binary discovery, JSON decoding, deterministic
sampling, and RecPhyloXML writing without constructing a CUDA model.

Fixture shape:

- Species tree: `(A:1,B:1)Root:0;`
- Species order: postorder `A`, `B`, `Root`, with indexes `0`, `1`, `2`.
- Clades: `0` is leaf `a`, `1` is leaf `b`, and `2` is the root clade.
- Split set: one deterministic split, parent `2` into children `0` and `1`,
  with base-2 log probability `0.0`.
- `pi`: row-major `3 x 3` base-2 log matrix with only `(0,A)`, `(1,B)`, and
  `(2,Root)` possible; impossible states use the `-1.0e300` sentinel.
- `e`, `log_p_s`, `log_p_d`, and `max_transfer`: base-2 log vectors aligned to
  the postorder species indexes.
- `origination_probs`: nonnegative natural-space weights with all mass on the
  root species.
- `seed`: fixed at `7` for deterministic CLI smoke.
- `max_events`: fixed at `32`, well above the expected two-leaf speciation
  reconciliation.

Pinned output contract:

- The CLI must write RecPhyloXML whose root tag ends with `recPhylo`.
- The sampled tree must contain a visible speciation at `Root`.
- The sampled leaves must map to species `A` and `B`.
