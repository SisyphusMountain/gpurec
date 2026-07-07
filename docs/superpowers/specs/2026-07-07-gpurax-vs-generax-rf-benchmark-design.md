# gpurax vs GeneRax — RF-to-true-tree benchmark (design)

**Date:** 2026-07-07
**Status:** approved (smoke-test-first)

## Goal

Measure whether `gpurax` (this repo's GeneRax reimplementation: joint
sequence×reconciliation SPR gene-tree correction on GPU) recovers the true gene
tree as well as GeneRax, across a difficulty sweep. Single metric: **normalized
Robinson–Foulds distance to the true gene tree** (ete3, unrooted, internal
labels stripped — `eff_lib.norm_rf`).

Success is *not* likelihood equality (gpurec's UndatedDTL differs from GeneRax's
by a topology-independent survival normalization; see
`docs/gpurax/reconciliation_convention.md`). Success is **tree quality (RF)**.

## Data

- True gene trees: `experiments/ghost_lineages/results/sim_allgenes/genes/*.nwk`
  (2000-family rustree DTL simulation; 60-species BD tree at
  `.../results/sim_allgenes/species_tree.newick`). Leaf names are
  `<species>_<gene>` (e.g. `44_77`); family names `g0, g1, …`.
- Family list source: `.../results/sim_inferred_L1000/families.txt`.
- **Smoke test: 1 family** (first in the list) × 3 lengths. Then scale to
  **30 families** (fixed-seed sample) × 3 lengths.

## Sweep

Protein lengths **L ∈ {55, 120, 300}**, pyvolve **LG**. Short = hard (noisy
start tree, room to correct); long = easy (start already near-truth). Traces
where correction helps and where the tools diverge.

## Pipeline (per family, per L) — fully paired

1. true tree → pyvolve `LG(L)` → amino-acid alignment (deterministic seed).
2. alignment → FastTree `-lg` → sequence-only ML **start tree** (shared start +
   the "uncorrected floor").
3. Build inputs shared by both tools: the alignment, the FastTree start tree,
   the full species tree, `subst_model = LG`, `rec-model = UndatedDTL`.
4. **GeneRax** (UndatedDTL, `--per-family-rates`, `--geneSearchStrategy SPR`,
   `--max-spr-radius 5`, `--unrooted-gene-tree`, `--seed 1`, via `mpirun`) →
   `<gx_prefix>/results/<name>/geneTree.newick`.
5. **gpurax** (`python -m gpurax -r UndatedDTL --max-spr-radius 5`) from the
   **same** start tree → `<gpx_prefix>/<name>.reconciled.nwk`.
6. `norm_rf(true, ·)` for start / GeneRax / gpurax.

Both tools see the identical alignment, start tree, species tree, and rec model
— the comparison isolates the reconciliation engine + search.

### Mapping-format divergence (important)

The two tools require **different** mapping-file formats, so each gets its own
mapping file (and its own families file); everything else is shared.

- GeneRax `.link`: `species:gene1;gene2;…` (colon/semicolon) — as produced by
  `driver_generax.py::build_mapping`, proven against this GeneRax binary.
- gpurax: `gene species` per line (Treerecs format) — required by
  `gpurax.io.families.parse_mapping` (only parses 2-token lines).

Use **separate output prefixes** per tool (GeneRax refuses to overwrite a
prefix; gpurax writes flat into its prefix dir).

## Environment

`.venv` has everything: torch, pyvolve, ete3, dendropy, gpurec, gpurax. The
whole orchestrator runs in `.venv`. External binaries only:

- FastTree: `/home/enzo/miniforge3/envs/phylo/bin/FastTree`
- GeneRax: `experiments/ghost_lineages/tools/GeneRax/build/bin/generax`
  (already compiled; via `mpirun`).

## Implementation

New standalone script `experiments/gpurax_vs_generax/run_benchmark.py`:

- Reuses `driver_generax.py`'s helpers where clean (pyvolve alignment +
  FastTree; GeneRax invocation; `eff_lib.norm_rf`); **drops** all ghost-lineage
  efficiency/recall/precision machinery (out of scope).
- `--n-families` (default **1** = smoke) and `--lengths 55,120,300`.
- Resumable per length (skip families whose tool output already exists).
- Adds the gpurax arm alongside FastTree(floor) + GeneRax.

Constants (FastTree / GeneRax / species tree / true-tree paths / leaf→species
prefix rule) lifted from `driver_generax.py`.

## Output

- `experiments/gpurax_vs_generax/results/results.json` — per (family, L):
  `{rf_start, rf_generax, rf_gpurax}`.
- Printed + saved summary: mean normalized RF per method per L; **paired
  win/tie/loss (gpurax vs GeneRax)**; mean improvement over the FastTree floor;
  RF(gpurax, GeneRax) agreement.
- Optional line plot (mean RF vs L, three curves) — deferred until after the
  30-family run.

## Out of scope (YAGNI)

Ghost-lineage efficiency/recall/precision; species-tree search / SpeciesRax;
per-species rates; real (non-simulated) data; GeneRax rebuild (already built);
branch-length optimization tuning inside gpurax SPR (Phase-1 default).

## Smoke-test acceptance

Run 1 family × {55,120,300}. Pass = all three arms produce a tree and finite
normalized RF for every L, with no crashes and correct paired bookkeeping. Only
after that, scale to 30 families.
