# Szollosi-Linked Dataset Tests

Date: 2026-06-04

## Dataset Search Summary

Subagents found several usable datasets from recent Gergely J. Szollosi
coauthored tool papers:

- AleRax HOGENOM-Core data:
  `https://cme.h-its.org/exelixis/material/alerax_data.tar.gz`
- GeneRax simulation and empirical data:
  `https://cme.h-its.org/exelixis/material/generax_data.tar.gz`
- SpeciesRax data:
  `https://cme.h-its.org/exelixis/material/speciesrax_data.tar.gz`

The AleRax and GeneRax archives were used for tests. SpeciesRax was not
downloaded because its `Content-Length` is 40,380,322,123 bytes and the local
filesystem had about 32 GiB free.

Local literature records are in
`benchmarks/large_dataset_capacity/papers/literature_inventory.md`.

## AleRax HOGENOM-Core

Paper context:

- Morel et al. 2024, AleRax, Bioinformatics, DOI
  `https://doi.org/10.1093/bioinformatics/btae162`
- Reported benchmark target: HOGENOM-Core, 666 species, 12,408 families.

Local data:

- Archive: `benchmarks/large_dataset_capacity/archives/alerax_data.tar.gz`
- Species tree:
  `benchmarks/large_dataset_capacity/datasets/alerax_hogenom_core/hogenom/runs/MFP/true_start_ufboot1000/run_--gene-tree-samples_100_--per-family-rates_1/alegenerax/species_trees/starting_species_tree.newick`
- Gene trees:
  `benchmarks/large_dataset_capacity/datasets/alerax_hogenom_core/hogenom/families/*/gene_trees/ufboot1000.MFP.geneTree.newick`
- Extracted target files: 12,408 gene-tree distribution files.

Run configuration:

- gpurec mode: `genewise`
- Device: CUDA, NVIDIA GeForce RTX 4090
- `family_chunk_size=25`
- `clade_budget=315000`
- `e_max_iter=500`
- `e_tol=1e-8`
- `pi_iters=6`
- `neumann_terms=3`
- Training runs used `lr=1.0`, convergence window 15,
  relative-loss tolerance `1e-5`, projected-gradient relative tolerance `1e-5`.
  Subset runs used `steps=120`; the all-family run used `steps=180`.

Results:

| Run | Families | Species leaves | Total clades | Total splits | Result | Wall time | Peak CUDA reserved |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| Sorted-100 eval | 100 | 666 | 45,908 | 118,484 | finite eval | 6.28 s | 0.47 GB |
| Largest-100 eval | 100 | 666 | 369,706 | 886,386 | finite eval | 8.54 s | 1.95 GB |
| Sorted-100 train | 100 | 666 | 45,908 | 118,484 | converged in 80 steps | 28.78 s | 1.03 GB |
| All-family eval | 12,408 | 666 | 3,931,344 | 10,496,142 | finite eval | 35.20 s | 2.16 GB |
| Sorted-1000 train | 1,000 | 666 | 395,610 | 1,038,199 | converged in 82 steps | 121.58 s | 2.26 GB |
| All-family train | 12,408 | 666 | 3,931,344 | 10,496,142 | converged in 87 steps | 932.56 s | 5.40 GB |

All-family convergence evidence:

- First loss: 9,015,291.0
- Final loss: 2,837,984.5
- Relative loss change over final 15-step window: `7.576e-6`
- Projected gradient relative to loss: `8.777e-8`
- Both convergence criteria are below the configured `1e-5` threshold.

Runtime comparison:

- AleRax paper reported 4.5 h for HOGENOM-Core on 20 CPU cores.
- gpurec all-family genewise optimization finished in 932.56 s, or 15.54 min.
- This is about 17.4x faster by wall time than the reported AleRax run.
- The comparison is against the same 666-species, 12,408-family benchmark data,
  but not the same implementation or hardware: gpurec used one RTX 4090 and
  genewise optimization; AleRax used its published CPU setup and model workflow.

Primary logs:

- `benchmarks/large_dataset_capacity/logs/alerax_hogenom_core_smoke100_eval.json`
- `benchmarks/large_dataset_capacity/logs/alerax_hogenom_core_largest100_eval.json`
- `benchmarks/large_dataset_capacity/logs/alerax_hogenom_core_smoke100_train.json`
- `benchmarks/large_dataset_capacity/logs/alerax_hogenom_core_all_eval.json`
- `benchmarks/large_dataset_capacity/logs/alerax_hogenom_core_1000_train.json`
- `benchmarks/large_dataset_capacity/logs/alerax_hogenom_core_all_train.json`

## GeneRax DTL Simulation Fixture

Paper context:

- GeneRax data archive:
  `https://cme.h-its.org/exelixis/material/generax_data.tar.gz`

Local data:

- Archive: `benchmarks/large_dataset_capacity/archives/generax_data.tar.gz`
- Extracted fixture:
  `benchmarks/large_dataset_capacity/datasets/generax_data/jsimdtl_s19_f100_sites250_dna4_bl0.5_d0.1_l0.2_t0.1_p0.0/`
- Species tree:
  `species_trees/trueSpeciesTree.newick`
- Original gene trees:
  `families/*_pruned/gene_trees/true.true.geneTree.newick`

The GeneRax gene leaves are gene IDs, while gpurec currently maps a gene leaf
to a species by taking the prefix before the first underscore. I converted the
96 available `true.true.geneTree.newick` files using each family's
`mappings/mapping.link`, producing labels like `H11_G1UUU...` so the species
prefix is preserved while gene-copy labels remain unique.

Converted gene-tree directory:

- `benchmarks/large_dataset_capacity/datasets/generax_jsimdtl_s19_f100_true_mapped/gene_trees/`

Results:

| Run | Families | Species leaves | Total clades | Total splits | Result | Wall time | Peak CUDA reserved |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| Mapped true-tree eval | 96 | 19 | 6,868 | 8,321 | finite eval | 2.05 s | 0.002 GB |
| Mapped true-tree train | 96 | 19 | 6,868 | 8,321 | converged in 66 steps | 9.98 s | 0.025 GB |

Primary logs:

- `benchmarks/large_dataset_capacity/logs/generax_jsimdtl_s19_f100_true_mapped_eval.json`
- `benchmarks/large_dataset_capacity/logs/generax_jsimdtl_s19_f100_true_mapped_train.json`

## Notes

- HOGENOM-Core is the most relevant direct comparison target because it matches
  the AleRax paper's 666-species, 12,408-family benchmark.
- The all-family HOGENOM-Core genewise optimization converged in 87 steps and
  beat the AleRax paper's reported 4.5 h wall time by about 17.4x.
- GeneRax data is directly useful after leaf-label conversion. The conversion
  is data preparation only; no gpurec source code was changed.
