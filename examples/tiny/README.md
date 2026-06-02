# Tiny Example

This folder contains a minimal two-species, two-gene fixture for smoke testing
the preprocessing and model-construction path.

## Files

- `families.txt`: one-family manifest. The family is named `tiny_family` and
  points to `gene.nwk` plus `gene.map`.
- `species.nwk`: species tree `(A:1,B:1)Root;`.
- `gene.nwk`: two-leaf gene tree for genes `a` and `b`.
- `gene.map`: tabular gene-to-species mapping, assigning `a` to species `A` and
  `b` to species `B`.

This dataset is intentionally too small to benchmark GPU throughput.  Use it
for parser, preprocessing, and API sanity checks.
