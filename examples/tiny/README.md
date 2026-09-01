# Tiny Example

This folder contains a minimal two-species, two-gene fixture for smoke testing
the preprocessing and model-construction path.

## Files

- `families.txt`: one-family AleRax-style manifest. The family is named
  `tiny_family` and points to `gene.nwk` plus `gene.map`.
- `species.nwk`: species tree `(A:1,B:1)Root;`.
- `gene.nwk`: two-leaf gene tree for genes `A_a` and `B_b`. gpurec derives the
  species from the prefix before the first underscore.
- `gene.map`: matching AleRax mapping metadata. gpurec currently derives the
  mapping from the leaf names and does not consume this file.

This dataset is intentionally too small to benchmark GPU throughput.  Use it
for parser, preprocessing, and API sanity checks.
