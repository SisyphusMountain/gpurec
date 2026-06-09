# gpurec Large-Dataset Capacity Interim Report

Date: 2026-06-04

## Current Baselines

Local HOGENOM restoration:

- Species tree: `tests/data/hogenom_S.tree`
- `gpurec-preprocess` species nodes: 1,325
- Binary-tree terminal leaves: 663
- Gene-tree files: 1,055

AleRax paper HOGENOM-CORE benchmark:

- Source: Morel et al. 2024, Bioinformatics, DOI
  https://doi.org/10.1093/bioinformatics/btae162
- Species: 666 core species
- Families: 12,408
- Reported runtime: AleRax 4.5 h vs ALE 44 h, both using 20 CPU cores
- Data archive: `https://cme.h-its.org/exelixis/material/alerax_data.tar.gz`

## Direct AleRax HOGENOM-Core Benchmark

This is now the strongest direct result because it uses the same HOGENOM-Core
benchmark scale reported in the AleRax paper.

Dataset:

- Archive:
  `benchmarks/large_dataset_capacity/archives/alerax_data.tar.gz`
- Species tree:
  `benchmarks/large_dataset_capacity/datasets/alerax_hogenom_core/hogenom/runs/MFP/true_start_ufboot1000/run_--gene-tree-samples_100_--per-family-rates_1/alegenerax/species_trees/starting_species_tree.newick`
- Gene trees:
  `benchmarks/large_dataset_capacity/datasets/alerax_hogenom_core/hogenom/families/*/gene_trees/ufboot1000.MFP.geneTree.newick`
- Species-tree leaves: 666, which is larger than local HOGENOM's 663 leaves
- Families run: 12,408

Input complexity measured by gpurec:

- Species nodes: 1,331
- Total clades: 3,931,344
- Total splits: 10,496,142
- Total gene leaves: 381,022
- Max clades in one family: 82,743
- Max gene leaves in one family: 883
- Batch count: 497 batches of up to 25 families

Measured results:

| Run | Families | Species leaves | Result | Wall time | Peak CUDA reserved |
| --- | ---: | ---: | --- | ---: | ---: |
| Eval only | 12,408 | 666 | finite likelihood | 35.20 s | 2.16 GB |
| Genewise optimization | 12,408 | 666 | converged in 87 steps | 932.56 s | 5.40 GB |

Full-run convergence evidence:

- First loss: 9,015,291.0
- Final loss: 2,837,984.5
- Relative loss change over final 15-step window: `7.576e-6`
- Projected gradient relative to loss: `8.777e-8`
- Both are below the configured `1e-5` convergence threshold.

Runtime comparison:

- AleRax reported HOGENOM-Core runtime: 4.5 h, i.e. 16,200 s, on 20 CPU cores.
- gpurec HOGENOM-Core genewise runtime here: 932.56 s, i.e. 15.54 min, on one
  NVIDIA GeForce RTX 4090.
- Wall-time speedup vs reported AleRax runtime: about 17.4x.
- AleRax also reported ALE at 44 h; gpurec is about 169.9x faster than that
  reported ALE runtime on this benchmark.

Primary logs:

- `benchmarks/large_dataset_capacity/logs/alerax_hogenom_core_all_eval.json`
- `benchmarks/large_dataset_capacity/logs/alerax_hogenom_core_all_train.json`

## Completed 1,007-Species Bacterial Benchmark

Dataset:

- Local archive: `/home/enzo/Téléchargements/23899299.zip`
- Extracted subset:
  `benchmarks/large_dataset_capacity/datasets/bacteria_1007_randomsample/`
- Species tree: `Section01.SpeciesTree/ReferenceTree.nwk`
- Gene trees:
  `Section01.SpeciesTree/EDCluster_randomsample_step1_iqtree_lg_g4.tar.gz`
- Species-tree leaves: 1,007, which is larger than local HOGENOM's 663 leaves
- Families run: 335 IQ-TREE ML gene trees

Input complexity measured by gpurec:

- Species nodes: 2,013
- Total clades: 500,453
- Total splits: 624,645
- Total gene leaves: 125,532
- Max clades in one family: 9,847
- Max gene leaves in one family: 2,463
- Batch count: 14 batches of up to 25 families

Run configuration:

- Mode: `genewise`
- Device: CUDA
- Optimizer: Adam over genewise log2 DTL rates
- Receiver weights fixed at uniform
- `family_chunk_size=25`
- `clade_budget=315000`
- `e_max_iter=500`
- `e_tol=1e-8`
- `pi_iters=6`
- `neumann_terms=3`
- Convergence window: 15 steps
- Loss convergence criterion: relative loss change <= `1e-5`
- Gradient convergence criterion: projected gradient relative to loss <= `1e-5`

Measured results:

| Run | Families | Species leaves | Result | Wall time | Peak CUDA reserved |
| --- | ---: | ---: | --- | ---: | ---: |
| Eval only | 335 | 1,007 | finite likelihood | 10.29 s | 1.93 GB |
| Genewise optimization | 335 | 1,007 | converged in 64 steps | 91.33 s | 4.50 GB |
| Largest-family stress slice | 25 | 1,007 | converged in 58 steps | 21.49 s | 5.64 GB |

Full-run convergence evidence:

- First loss: 4,202,253.0
- Final loss: 1,554,717.625
- Relative loss change over final 15-step window: `7.879e-6`
- Projected gradient relative to loss: `1.538e-6`
- Both are below the configured `1e-5` convergence threshold.

Primary logs:

- `benchmarks/large_dataset_capacity/logs/bacteria_1007_all_eval.json`
- `benchmarks/large_dataset_capacity/logs/bacteria_1007_all_train.json`
- `benchmarks/large_dataset_capacity/logs/bacteria_1007_25_largest_train.json`

## Comparison Status

gpurec now has two completed larger-than-local-HOGENOM genewise capacity
results:

- Direct AleRax HOGENOM-Core: 12,408 families, 666 species leaves, converged in
  932.56 s. This beats the AleRax paper's reported 4.5 h wall time by about
  17.4x, with the caveat that gpurec used one RTX 4090 while AleRax used 20 CPU
  cores and a different implementation/model workflow.
- Bacterial 1,007-species dataset: 335 families, 1,007 species leaves, converged
  in 91.33 s. This is not an AleRax apples-to-apples comparison, but it extends
  the species-tree scale beyond both local HOGENOM and HOGENOM-Core.

## Candidate Data Inventory

See `benchmarks/large_dataset_capacity/metadata/dataset_candidates.md` for the
literature search table and download links. The strongest next targets are:

- AleRax HOGENOM-CORE data archive, because it provides the direct reported
  4.5 h comparison target.
- CASTLES-Pro 1,000-species simulations, because they provide controlled
  larger-than-HOGENOM Newick gene-tree workloads.
- Web of Life / WoL 10,575-genome trees, because it is an extreme empirical
  species-tree scale test.
