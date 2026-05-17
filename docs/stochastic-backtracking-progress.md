# Stochastic Backtracking Progress

## Implemented

- Added `crates/gpurec-backtrack`, a Rust sampler that depends on the local
  `rustree` checkout and writes `RecTree::to_xml()` RecPhyloXML.
- Added `gpurec.backtracking.export_backtracking_input()` to export one
  family from a `GeneReconModel` as base-2 log probabilities, CCP splits,
  species topology, leaf mappings, and root origination probabilities.
- Added `gpurec.backtracking.sample_recphyloxml()` as a Python bridge that
  runs the Rust sampler and returns one XML document.
- Added `sample_recphyloxmls()` plus Rust CLI `--samples/--output-dir` support
  so bulk comparisons reuse one exported state and one Rust process.
- Added `GPUREC_BACKTRACK_BIN` / `backtrack_binary` support so Python callers
  can invoke a prebuilt Rust binary instead of paying `cargo run` startup.
- Added `gpurec.backtracking.recphyloxml_event_counts()` to report
  AleRax-style `S`, `SL`, `D`, `DL`, `T`, `TL`, `L`, and `Leaf` counts from
  gpurec RecPhyloXML.
- Added `scripts/compare_backtracking_alerax_events.py` for fixture-level
  event-count comparisons against AleRax saved samples.
- Added a CUDA integration smoke test on `tests/data/test_trees_3`.
- Added a skipped-when-missing AleRax event-range integration test for
  `tests/data/test_trees_100/output_global/family_0000`.

## Verification

Commands run:

```bash
cargo test --manifest-path crates/gpurec-backtrack/Cargo.toml
pytest -q tests/integration/test_stochastic_backtracking.py
GPUREC_BACKTRACK_BIN=crates/gpurec-backtrack/target/release/gpurec-backtrack \
  pytest -q tests/integration/test_stochastic_backtracking.py
```

Both pass locally.

## AleRax Comparison

AleRax represents visible speciation-loss and transfer-loss histories in
RecPhyloXML, but its saved `test_trees_100` samples do not emit same-species
duplication-loss self-loops. The Rust sampler now contracts `HiddenDupLoss`
while retaining visible `SL`/`TL`, which removes the earlier `DL` residual.

First comparison target: `tests/data/test_trees_100`, `output_global`,
`family_0000`, using AleRax global rates from
`output_global/model_parameters/model_parameters.txt`:

```text
D=0.0191209 L=0.0199312 T=0.0208267
```

AleRax saved 100 samples:

```text
S    min=103 mean=103.00 max=103
SL   min=2   mean=2.00   max=2
D    min=1   mean=1.99   max=2
T    min=4   mean=4.01   max=5
L    min=0   mean=0.00   max=0
Leaf min=110 mean=110.0  max=110
```

gpurec Rust backtracking, 5 samples after AleRax-style XML classification:

```text
S    min=103 mean=103.0 max=103
SL   min=2   mean=2.0   max=2
D    min=2   mean=2.0   max=2
DL   min=0   mean=0.0   max=0
T    min=4   mean=4.0   max=4
TL   min=0   mean=0.0   max=0
L    min=0   mean=0.0   max=0
Leaf min=110 mean=110.0 max=110
```

This matches the AleRax event taxonomy for the first small-family check. Raw
XML tag counts differ because RecPhyloXML represents `SL` as a `<speciation>`
node with an explicit direct `<loss>` child.

Current broader check:

```bash
python scripts/compare_backtracking_alerax_events.py --families 10 --samples 20
python scripts/compare_backtracking_alerax_events.py --families 10 --samples 20 \
  --backtrack-binary crates/gpurec-backtrack/target/release/gpurec-backtrack
```

For families `0000` through `0009`, `DL`, `L`, and `Leaf` match exactly. The
largest absolute mean deltas in this 20-sample run were small: `D` +0.33
(`family_0006`), `T` -0.28 (`family_0006`), `SL` +0.22 (`family_0006`), and
`TL` +0.15 (`family_0006`/`0008`).

## Next Checks

- Compare more families from `test_trees_100`, then move to the available
  HOGENOM fixtures.
- Add larger-run summary artifacts once the broader fixture comparisons are
  complete.
