# Fresh AleRax Backtracking Validation

## Target

- Species tree: `tests/data/test_trees_100/sp.nwk`
- Gene tree: `tests/data/test_trees_100/g_0003.nwk`
- Family: `family_0003`
- Rates: `D=0.0191209`, `L=0.0199312`, `T=0.0208267`
- Reason for choosing this tree: the saved fixture previously showed a
  nontrivial mixture of `S`, `SL`, `D`, `T`, and `TL` events.

## Fresh AleRax Run

Output directory:

```text
output_alerax_backtracking_validation/family_0003_1000/alerax
```

Command:

```bash
AleRax_oliver/build/bin/alerax \
  -s tests/data/test_trees_100/sp.nwk \
  -f output_alerax_backtracking_validation/family_0003_1000/families_family_0003.txt \
  -p output_alerax_backtracking_validation/family_0003_1000/alerax \
  -g 1000 \
  --seed 20260517 \
  --model-parametrization GLOBAL \
  --fix-rates \
  --d 0.0191209 \
  --l 0.0199312 \
  --t 0.0208267 \
  --species-tree-search SKIP \
  --rec-model UndatedDTL
```

AleRax wrote 1000 `family_0003_eventCounts_*.txt` files.

## Event Comparison

The Rust sampler was run for 1000 samples with the same tree, species tree,
rates, and seed base through
`crates/gpurec-backtrack/target/release/gpurec-backtrack`.

```text
event  alerax_min  alerax_mean  alerax_max  gpurec_min  gpurec_mean  gpurec_max  mean_delta
S      116         117.343000   118         116         117.329000   118         -0.014000
SL     6           6.737000     10          6           6.745000     9           0.008000
D      4           4.996000     6           4           4.994000     6           -0.002000
DL     0           0.000000     0           0           0.000000     0           0.000000
T      2           3.661000     5           2           3.677000     6           0.016000
TL     0           0.349000     2           0           0.342000     2           -0.007000
L      0           0.000000     0           0           0.000000     0           0.000000
Leaf   127         127.000000   127         127         127.000000   127         0.000000
```

The largest absolute mean difference over 1000 samples is `0.016` events.
