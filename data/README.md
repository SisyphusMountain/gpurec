# gpurec data

`data/external/` is the canonical local data root for gpurec benchmarks and
coupled research projects. Its
contents are intentionally ignored by Git because the empirical datasets and
retained source archives are tens of gigabytes.

The store was consolidated from `../gpurec-data` on 2026-09-01. Dataset
provenance, checksums, and retrieval instructions will be promoted from the
store into tracked manifests as each maintained benchmark is migrated.

Set `GPUREC_DATA_ROOT` to the absolute path of `data/external` when running
experiments that require the local datasets.

Current top-level groups are:

- `benchmarks/large_dataset_capacity/datasets/`: empirical HOGENOM, archaea,
  and related benchmark inputs;
- `kernel-bench/`: large saved tensors from the retired kernel benchmark;
- `ghost-lineages/`: GeneRax/cyanobacteria inputs for the separately versioned
  ghost-lineage work, including a local link to its archived reference workspace.

Individual scripts may define `GPUREC_DATA_ROOT` as the large-dataset `datasets/`
subdirectory; their READMEs state that narrower contract where applicable.
