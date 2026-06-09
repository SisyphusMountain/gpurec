# AleRax Dataset Inventory

Generated on 2026-06-04 from local downloads and archive file lists.

## Scope

The AleRax supplement describes four dataset buckets:

- `DLSIM` and `DTLSIM`: SimPhy simulated datasets reused from the SpeciesRax benchmark setup.
- `ALESIM`: the original ALE benchmark dataset from Szollosi et al. 2013, published on Dryad.
- `HOGENOM`: 12,408 empirical HOGENOM gene families with IQ-TREE2 ML trees and 1000 ultrafast bootstrap trees.
- `Archaea`: 60 archaeal genomes from Davin et al. 2018, using existing gene-tree distributions.

The official AleRax archive at `https://cme.h-its.org/exelixis/material/alerax_data.tar.gz` contains only the HOGENOM dataset. The simulation data is in the SpeciesRax archive, the ALESIM files are on Dryad, and the Archaea files are in the Davin 2017 PBIL dataset.

## Local Download Status

| Dataset bucket | Source | Local path | Status |
| --- | --- | --- | --- |
| HOGENOM | `https://cme.h-its.org/exelixis/material/alerax_data.tar.gz` | `benchmarks/large_dataset_capacity/archives/alerax_data.tar.gz` | Downloaded, 15,781,432,926 bytes; listed in `metadata/alerax_data_filelist.txt`. |
| DLSIM/DTLSIM | `https://cme.h-its.org/exelixis/material/speciesrax_data.tar.gz` | `benchmarks/large_dataset_capacity/archives/speciesrax_data.tar.gz` | Downloaded, 40,380,322,123 bytes; listed in `metadata/speciesrax_data_filelist.txt`. |
| Archaea | `ftp://pbil.univ-lyon1.fr/pub/datasets/davin2017/` | `benchmarks/large_dataset_capacity/archives/alerax_datasets/davin2017_archaea/` | Downloaded the Archaea files; local directory is 222M. |
| ALESIM | `https://doi.org/10.5061/dryad.pv6df` | Not downloaded | Dryad metadata is accessible, but anonymous file downloads returned 401/403 for both API and file-stream routes. |

## HOGENOM

The AleRax supplement says HOGENOM contains 12,408 gene families from 666 core species. Each family has an ML tree and 1000 IQ-TREE2 ultrafast bootstrap trees under `LG+G`; AleRax and ALE used the bootstrap trees as gene-tree distributions.

Measured contents of `alerax_data.tar.gz`:

- Total archive entries: 5,373,878.
- Top-level directories: `hogenom` only.
- Family directories: 12,408.
- Alignment FASTA files under `hogenom/alignments`: 12,408.
- Original bootstrap tree sample files: 12,408 at `hogenom/families/*/gene_trees/ufboot1000.MFP.geneTree.newick`.
- Derived bootstrap-tree variants: 37,199 additional files with the same suffix after method prefixes.
- Total files ending in `ufboot1000.MFP.geneTree.newick`: 49,607.
- Consensus trees: 12,408 `consensus_50.newick` files.
- Precomputed AleRax CCP files: 12,408 at `hogenom/runs/MFP/true_start_ufboot1000/run_--gene-tree-samples_100_--per-family-rates_0/alegenerax/ccps/*.ccp`.

The gpurec-ready subset already extracted at `benchmarks/large_dataset_capacity/datasets/alerax_hogenom_core` contains:

- 12,408 original `ufboot1000.MFP.geneTree.newick` tree-sample files.
- 1 `starting_species_tree.newick`.
- Local size: 7.2G.

For `gpurec`, HOGENOM is the easiest empirical benchmark because it has raw multi-Newick gene-tree samples. It also has precomputed CCP files, but using those directly requires a CCP/AleRax `.ccp` reader rather than the current tree-sample path.

## DLSIM and DTLSIM

The AleRax supplement says DLSIM and DTLSIM were generated with SimPhy, using 50 replicates per parameter setting. DLSIM excludes HGT; DTLSIM includes distance-independent HGT. The tested parameters include species count, family count, sequence length, D/L/T rates, branch-length multiplier, and population size.

The SpeciesRax archive contains both empirical and simulated data:

- Total archive entries: 16,005,583.
- `speciesrax_data/simulated`: 12,325,073 entries.
- `speciesrax_data/empirical`: 3,680,508 entries.

Measured simulated contents:

- Simulated dataset directories: 4,100.
- DLSIM dataset directories: 1,900.
- DTLSIM dataset directories: 2,200.
- True species trees: 4,100 `species_trees/speciesTree.newick` files.
- Total species-tree `.newick` files: 53,304.
- `simphy_config.txt` files: 4,100.
- `indelible_config.txt` files: 4,100.

Dataset-directory tokens encode the varied parameters. Counts by nominal family count:

| Family-count token | Dataset directories |
| --- | ---: |
| `f50` | 200 |
| `f100` | 3,300 |
| `f200` | 200 |
| `f500` | 200 |
| `f1000` | 200 |

Counts by species-count token:

| Species-count token | Dataset directories |
| --- | ---: |
| `s15` | 200 |
| `s25` | 3,300 |
| `s35` | 200 |
| `s50` | 200 |
| `s75` | 200 |

Family and gene-tree contents:

- Materialized family directories: 656,967.
- DLSIM family directories: 314,076.
- DTLSIM family directories: 342,891.
- `alignment.msa`: 656,967.
- `gene_trees/true.true.geneTree.newick`: 656,967.
- `gene_trees/raxml-ng.GTR+G.geneTree.newick`: 656,967.
- `gene_trees/raxmlMultiple.GTR+G.geneTree.newick`: 656,967.
- `gene_trees/generax-last.GTR+G.geneTree.newick`: 97.
- `1/g_trees*.trees` files: 680,000.
- `1/s_tree.trees` and `1/l_trees.trees` files: 8,200 total.

The nominal family count from dataset names sums to 680,000, matching the `g_trees*.trees` count. The postprocessed `families/*` directories are fewer, so not every nominal simulated family has a full materialized family directory in the archive.

For `gpurec`, these simulations are useful for controlled correctness and scaling benchmarks. The best direct inputs are either the materialized true/RAxML Newick gene trees, or the `g_trees*.trees` distribution files if the parser accepts that format. They also include true species trees, so they are better than HOGENOM for error/accuracy benchmarking.

The SpeciesRax archive also contains empirical datasets not directly part of the AleRax simulation bucket:

| Empirical dataset | Family directories | Gene-tree files |
| --- | ---: | ---: |
| `life92` | 41,222 | 123,666 |
| `vertebrates188` | 31,612 | 31,612 |
| `plants23` | 21,469 | 128,814 |
| `vertebrates21` | 18,848 | 18,848 |
| `primates13` | 16,670 | 117,470 |
| `plants83` | 9,237 | 9,237 |
| `fungi16` | 7,180 | 7,180 |
| `fungi60` | 5,659 | 39,613 |
| `cyanobacteria36` | 1,099 | 23,441 |
| `life92_mincov0.5` | 703 | 703 |
| `archaea364` | 150 | 600 |

Do not confuse `speciesrax_data/empirical/archaea364` with the 60-genome Archaea analysis used in the AleRax paper.

## ALESIM

The AleRax supplement says ALESIM reuses the original ALE benchmark: 1,099 Cyanobacteria gene families, with simulated sequences generated along ALE-inferred trees while retaining sequence lengths and branch lengths.

Dryad API metadata for DOI `10.5061/dryad.pv6df` exposes these files:

| File | Dryad file id | Size | MD5 | Description |
| --- | ---: | ---: | --- | --- |
| `real_ale.tgz` | 84729 | 20,432,109 | `9907724d0454a046e295ebcb94befbfd` | `.ale` files from PhyloBayes samples for real alignments. |
| `real_alignments.tgz` | 84727 | 4,006,633 | `2102b090babb8496a0d734aeac15734b` | Alignments for 1,099 Cyanobacteria HOGENOM v5 families. |
| `real_mlrec.tgz` | 84730 | 795,623 | `cdeaf56af531a9f1a557cd37c95a84f2` | ALEml output for real ALE files. |
| `real_trees.tgz` | 84728 | 592,601 | `553c2edf915e7a4a61deb50cd283d3b9` | CCP-maximizing trees sampled from ALEsample outputs. |
| `sequence_mlrec.tgz` | 84731 | 863,082 | `227327558ca08be6f0038493c9394c6a` | ALEml output from PhyML trees based on real alignments. |
| `simulated_ale.tgz` | 84733 | 14,985,790 | `2321efe2097ddd67eaef8920c4413990` | `.ale` files from PhyloBayes SIMPLE samples for simulated alignments. |
| `simulated_alignments.tgz` | 84732 | 3,178,590 | `5a86821d9be27dee3cff24e3f4b931d8` | Simulated alignments based on real trees. |
| `simulated_mlrec.tgz` | 84734 | 805,471 | `7915e7dc80c553a7bb145ad31eca6c74` | ALEml output for simulated ALE files. |

Total listed file payload size: 45,659,899 bytes.

Download attempts that failed:

- `https://datadryad.org/api/v2/files/<id>/download`: HTTP 401.
- `https://datadryad.org/downloads/file_stream/<id>`: HTTP 403.
- `https://datadryad.org/stash/downloads/file_stream/<id>` redirects to the same file-stream URL and returns HTTP 403.

For `gpurec`, ALESIM would be useful if obtained because it has compact `.ale` CCP summaries and small alignment/tree archives. It is not currently available locally.

## Archaea

The AleRax supplement says the Archaea experiment used 60 archaeal genomes from Davin et al. 2018, considering 5,379 gene families with at least 4 sequences. AleRax reused the original gene-tree distributions, then excluded the 20 largest families by CCP size with `--trim-ratio 0.0038`.

Downloaded source files:

| File | Size | Measured contents |
| --- | ---: | --- |
| `Archaea.phy` | 837,789 | PHYLIP header `60 10738`: 60 taxa, 10,738 alignment columns. |
| `ArchaeaTree` | 866 | Species tree file. |
| `archaea_ales.tgz` | 106,869,822 | 31,236 archive entries: 5,446 under `treedists`, 25,790 under `small_fams`. |
| `archaea_recs.tar.gz` | 59,580,276 | 35,849 reconciliation entries. |
| `Archaea_Transfers.tsv` | 65,045,671 | 1,779,989 data lines; no header observed. |
| `Archaea_constraints_maxtic` | 18,603 | 1,477 lines. |
| `Archaea_constraints_analysis.tsv` | 54,449 | Header plus 1,889 data rows. |

For `gpurec`, the Archaea archive is useful as a direct ALE/CCP benchmark. It does not provide the same easy raw multi-Newick bootstrap-tree samples as HOGENOM. The closest paper-matching input is the `treedists` subset of `archaea_ales.tgz`, then filtering to the 5,379 families with at least 4 sequences and applying the paper's 20-family CCP-size trim.

The decompressed benchmark folder has been renamed for readability:

```text
benchmarks/large_dataset_capacity/datasets/alerax_archaea_davin2017/
├── ale_gene_tree_distributions/
│   ├── main_families_ge4seq/   # 5,446 .ale files from treedists/
│   └── small_families/         # 25,790 .ale files from small_fams/
├── ale_reconciliations_uml/    # 35,849 .uml_rec files
├── species_reference/
└── transfers_and_constraints/
```

Since gpurec can handle ALE files, the recommended direct Archaea benchmark input is `ale_gene_tree_distributions/main_families_ge4seq/*.ale` with `species_reference/reference_species_tree.newick`.

## Practical Benchmark Choices for gpurec

Use HOGENOM first when the benchmark needs raw gene-tree samples:

- It is already extracted in `datasets/alerax_hogenom_core`.
- It has one 1000-tree Newick sample file per family.
- It covers 12,408 empirical families and one rooted species tree.

Use DLSIM/DTLSIM for controlled accuracy/scaling:

- It has true species trees and true gene trees.
- It has 4,100 parameterized datasets and 656,967 materialized family directories.
- The archive is large, so extract targeted parameter slices instead of the full tarball.

Use Archaea when benchmarking CCP/ALE input handling:

- The paper-relevant dataset is based on `.ale` gene-tree distributions, not raw treefiles.
- Use `datasets/alerax_archaea_davin2017/ale_gene_tree_distributions/main_families_ge4seq/*.ale` directly.

ALESIM is potentially useful but currently unavailable locally due to Dryad download authorization failures.
