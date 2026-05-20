# Stochastic Backtracking Progress

## Implemented

- Added `crates/gpurec-backtrack`, a Rust sampler that depends on the pinned
  `rustree` git dependency declared in `Cargo.toml` and writes
  `RecTree::to_xml()` RecPhyloXML.
- Added `gpurec.backtracking.export_backtracking_input()` to export one
  family from a `GeneReconModel` as base-2 log probabilities, CCP splits,
  species topology, leaf mappings, and root origination probabilities.
- Added `gpurec.backtracking.sample_recphyloxml()` as a Python bridge that
  runs the Rust sampler and returns one XML document.
- Added `sample_recphyloxmls()` plus Rust CLI `--samples`, `--output-dir`,
  `--seed`, and `--max-events` support so bulk comparisons reuse one exported
  state and one Rust process.
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
cargo test --locked --manifest-path crates/gpurec-backtrack/Cargo.toml
cargo build --locked --release --manifest-path crates/gpurec-backtrack/Cargo.toml
cargo run --locked --quiet --manifest-path crates/gpurec-backtrack/Cargo.toml -- --help
pytest -q tests/integration/test_rust_backtracking_fixture.py
```

These pass locally.

## AleRax Comparison

AleRax represents visible speciation-loss and transfer-loss histories in
RecPhyloXML, but its saved `test_trees_100` samples do not emit same-species
duplication-loss self-loops or transfers whose recipient is immediately lost.
The Rust sampler now contracts those hidden self-loops while retaining visible
`SL` and donor-loss `TL`, which removes the earlier `DL` and HOGENOM `TL`
residuals.

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

Current broader checks:

```bash
python scripts/compare_backtracking_alerax_events.py --families 100 --samples 20 \
  --backtrack-binary crates/gpurec-backtrack/target/release/gpurec-backtrack
python scripts/compare_backtracking_alerax_events.py \
  --dataset tests/data/hogenom_bench \
  --output-name output_alerax_corrected \
  --families 20 --samples 20 \
  --backtrack-binary crates/gpurec-backtrack/target/release/gpurec-backtrack
python scripts/compare_backtracking_alerax_events.py \
  --dataset tests/data/HOGENOM/hogenom \
  --output-name output_alerax_corrected \
  --families-file tests/data/HOGENOM/hogenom/hogenom_families.local.txt \
  --species-tree tests/data/HOGENOM/hogenom/hogenom_S.tree \
  --families 5 --samples 10 \
  --backtrack-binary crates/gpurec-backtrack/target/release/gpurec-backtrack \
  --preprocess-cache-dir /tmp/gpurec_backtrack_hogenom_native_cache
```

For all `test_trees_100` families, `DL`, `L`, and `Leaf` match exactly. With
20 gpurec samples per family, the largest absolute mean deltas were `D` +0.33
(`family_0006`), `T` -0.32 (`family_0098`), `SL` +0.30 (`family_0040`), and
`TL` -0.26 (`family_0065`).

For the first 20 HOGENOM bench families, `DL`, `L`, and `Leaf` also match
exactly. With 20 gpurec samples per family, the largest absolute mean deltas
were `SL` -2.42 (`family_0011`), `T` +1.27 (`family_0012`), `TL` +1.43
(`family_0004`), and `S` +0.99 (`family_0013`); these families have much wider
AleRax sample ranges than `test_trees_100`.

Higher-sample checks support that variance explanation: with 100 gpurec samples,
`family_0011` has `SL` delta -0.22 and `TL` delta -0.98, while `family_0004`
has `SL` delta -1.01 and `TL` delta +0.07.

The comparison script also supports AleRax `[FAMILIES]` files and per-family
rate files. On the first five native `tests/data/HOGENOM/hogenom` families,
`DL`, `L`, and `Leaf` match exactly. With 10 gpurec samples per family, the
largest absolute mean deltas were `SL` -2.94 (`CLU_003896_4_2_C`), `T` -1.10
(`CLU_000604_8_1_C`), and `TL` +1.08 (`CLU_000604_8_1_C`) across wide AleRax
sample ranges.

## Next Checks

- Run larger HOGENOM batches when runtime budget allows.
- Add committed comparison tables if we need reproducible audit artifacts.
