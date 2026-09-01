# Examples

This directory contains small input datasets for exercising the GPUREC
preprocessing and reconciliation workflow.

## Files

- `tiny/families.txt`: tiny dataset manifest with one family named
  `tiny_family`; points to the starting gene tree and gene-to-species mapping.
- `tiny/species.nwk`: two-species Newick species tree with root `Root` and
  leaves `A` and `B`.
- `tiny/gene.nwk`: two-leaf Newick gene tree with leaves `A_a` and `B_b`.
- `tiny/gene.map`: corresponding AleRax mapping metadata. gpurec itself derives
  species names from gene-leaf prefixes rather than reading this file.

See [`docs/input-contract.md`](../docs/input-contract.md) for the complete
naming and manifest-path rules.
