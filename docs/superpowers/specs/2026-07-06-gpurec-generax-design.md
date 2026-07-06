# Design: gpurec-backed GeneRax (batch-native joint reconciliation)

**Date:** 2026-07-06
**Status:** Approved (design), pending implementation plan
**Author:** brainstormed with Claude

## 1. Goal

Reproduce the functionality of **GeneRax** (Morel et al. 2020, *MBE* 37(9):2763)
— maximum-likelihood, species-tree-aware inference of reconciled gene family
trees under the UndatedDTL model — but with **gpurec** as the engine that
computes (and differentiates) the reconciliation likelihood.

GeneRax maximizes a per-family **joint likelihood**

```
L(G, S, N | A)  ∝  ∏_i  L(S, N | G_i) · L(G_i | A_i)
                        └── reconciliation ──┘ └ sequence ┘
                             (UndatedDTL)        (Felsenstein
                             = gpurec            pruning, libpll)
```

by alternating (a) DTL-rate optimization at fixed gene trees and (b) per-family
SPR search over gene-tree topologies scored by the joint likelihood.

gpurec is fast at evaluating reconciliation likelihoods for **many gene families
at once** (its native batched, ragged workload). The design goal is therefore a
GeneRax-equivalent that exploits that batch strength to produce accurate
reconciled gene trees for many families **accurately and rapidly**.

The one engine GeneRax has that gpurec lacks is the **sequence / phylogenetic
likelihood** `L(G_i | A_i)`. Everything below is organized around obtaining that
term without sacrificing gpurec's batch advantage on the reconciliation term.

## 2. Key decisions (from brainstorming)

| Decision | Choice | Rationale |
|---|---|---|
| Ambition / phasing | **Phased batch-native** | Correct, parity-checkable tool soonest; defer the big GPU effort until profiling justifies it. |
| Sequence-likelihood term (Phase 1) | **libpll / coraxlib** (CPU, thread-pooled) | Exact GeneRax substitution-model parity; battle-tested; already in the tree as a submodule. |
| Sequence-likelihood term (Phase 2) | GPU Felsenstein pruning **only if profiling shows the seq term is the bottleneck** | Respects "profile before optimizing / simplest correct first". |
| Sequence integration mechanism | **pybind11 wrapper reusing GeneRax `LibpllEvaluation` + coraxlib `pll_utree` SPR** | Least new numerical code; inherits GeneRax's validated model parsing, incremental CLV, branch-length optimization, and SPR-neighbor enumeration. |
| Search ownership | **Python owns the search loop**; C++ exposes per-family scoring/enumeration primitives | Lets gpurec batch reconciliation across families and neighbors. |
| Initial gene trees | Use provided `starting_gene_tree`; **also build one when absent** (parsimony / raxml-ng = GeneRax Step 0) | Matches GeneRax usage and convenience. |
| DTL rate granularity | **Global first**; per-species / per-family reachable for free | gpurec already supports all three (`optimize` / `fit_genewise`). |
| Success metric | **Joint-loglk + RF parity vs GeneRax**, plus fixed-rate reconciliation-term parity | Strongest, most objective "reproduced GeneRax" check. |
| Outputs | **Full GeneRax-compatible export**: Newick + RecPhyloXML + Notung + `scores.tsv` | Drop-in comparison in downstream tools. |

## 3. Pipeline (mirrors GeneRax `GeneRaxCore::geneTreeJointSearch`)

Re-hosted in Python so gpurec drives the reconciliation term in batch. **Two
distinct pre-main-search phases exist in GeneRax and go opposite directions** —
both are preserved and named explicitly:

| Phase | When | Sequence term (libpll) | Reconciliation term (recLL) | Purpose |
|---|---|---|---|---|
| **Step 0** (paper, Fig. 3) | only when a starting tree is absent / random | **ON** | OFF | build a decent initial GFT from the MSA (phylogenetic-ML only) |
| **`--rec-radius` warm-up** (`GeneRaxCore.cpp:238`) | always, if `recRadius > 0` | OFF | **ON** | cheaply pull the tree toward the species tree before joint search |
| **Main joint search** (`GeneRaxCore.cpp:243`) | always | **ON** | **ON** | joint-likelihood SPR search, radius 1 → maxSPRRadius |

Source evidence: in `optimizeRatesAndGeneTrees`, `enableRecLL` is a constant
`true` (`GeneRaxCore.cpp:425`); the warm-up loop passes `enableLibpll=false`
(`:239`), the main loop passes `enableLibpll=true` (`:244`). Step 0 is the
separate initial-tree-building step (paper p.2767), sequence-only.

```
load families (MSA + gene→species mapping + optional starting tree) + species tree
Step 0:   build initial trees where missing            (parsimony / raxml-ng; seq-only)
warm-up:  for r = 1 .. recRadius:                       (reconciliation-only SPR)
              SPR search at radius r, gpurec recLL only, libpll OFF
main:     for r = 1 .. maxSPRRadius:
              Step 1  optimize DTL rates at fixed trees → gpurec.optim (batched, ALL families)
              Step 2  batch-native joint SPR search at radius r (seq + recon), rates fixed
              (per-species rates enabled only in the last two rounds, as in GeneRax)
final:    reconcile (gpurec backtracking) + export (nwk / RecPhyloXML / Notung / scores.tsv)
```

The warm-up phase is a pure gpurec workload (no libpll in the loop) and is
therefore the cheapest, most batch-friendly part of the whole procedure.

## 4. The batch-native SPR search (core idea)

**Claim: batching requires no change to GeneRax's search semantics.** At each
step GeneRax already enumerates the *entire neighbor set* of a family's current
tree and applies the single best joint-improving move. Two batching axes follow
without altering the trajectory:

1. **Across families** — families are independent (GeneRax parallelizes them over MPI ranks).
2. **Across the neighbor set** of each family's current tree.

Each synchronized step:

```
for each still-improving family f:
    enumerate SPR neighbors of f's current tree           (C++ / coraxlib)      [axis 2]
compute seq loglk for every neighbor of every family      (CPU thread pool)     [axis 1]
ship ALL neighbor topologies (all families) to gpurec  →  batched recon loglk   (ONE call)
joint = seq + recon ;  per family pick best neighbor ;  apply if it improves
families with no improving move drop out  →  batch shrinks (ragged; gpurec's specialty)
```

This reproduces GeneRax's greedy "apply single best move, re-enumerate, repeat;
radius 1 → max" trajectory move-for-move. The *only* difference is **when** work
synchronizes: all families advance one accepted move at a time so their
evaluations co-batch. Ragged, shrinking family batches are exactly what gpurec
is built for.

Notes:
- During the warm-up phase the seq step is skipped entirely (recon-only).
- The reconciliation term for a fixed gene tree is a **degenerate CCP** (one
  split per clade, `log_split_prob = 0`) — gpurec's preprocessor already
  produces this from a single Newick tree.

## 5. Components & the C++/Python boundary

| Module | Language | Responsibility |
|---|---|---|
| `_seqlik` (pybind11 extension) | C++ | Wrap GeneRax `LibpllEvaluation` + coraxlib `pll_utree`. API: `load_family(alignment, model, tree)`, `seq_loglk(handle, opt_bl)`, `enumerate_spr(handle, radius)`, `apply_spr(handle, move)`, `get_newick(handle)`, `set/get_branch_lengths`. Inherits validated model parsing, incremental CLV, branch-length optimization. |
| I/O | Python | Parse GeneRax families file, species tree, gene→species mapping (reuse identical formats for validation parity). |
| recon adapter | Python | (family × candidate topology) → gpurec batched input → recon-loglk vector. **Incremental**: only re-preprocess families whose current tree changed this step. |
| SPR search | Python | The ragged multi-family greedy loop of §4 (warm-up and main). |
| rate adapter | Python | Wrap `gpurec.optim.optimize` (global / specieswise) and `fit_genewise` (per-family). |
| reconcile + export | Python | gpurec backtracking → event samples; write Newick / RecPhyloXML / Notung / `scores.tsv`. |
| CLI driver | Python | Mirror GeneRax args: `--families --species-tree --rec-model --max-spr-radius --rec-radius --per-family-rates --per-species-rates --rec-weight --prefix`. |

**Boundary contract.** C++ owns per-family tree objects, libpll partitions,
branch lengths, and SPR mechanics (topologies are cheap to enumerate/apply
there). Python owns orchestration, the reconciliation batch, and rate
optimization. Topologies cross the boundary as Newick (or a compact
split-encoding if profiling shows Newick round-tripping is hot).

## 6. Data flow (one main-search step)

```
Python: ask each active family (C++) for its SPR neighbor topologies at radius r
   │
   ├─► C++ thread pool: seq_loglk(neighbor, opt_bl)  for every neighbor      → seq[f][k]
   │
   └─► recon adapter: pack {all neighbors, all families} → gpurec batch
          gpurec (GPU): batched UndatedDTL recon loglk                        → rec[f][k]
   │
Python: joint[f][k] = seq[f][k] + recWeight · rec[f][k]
        best_k[f]   = argmax_k joint[f][k]
        if joint[f][best_k] > current[f]: C++ apply_spr(f, best_k); mark f active
        else: retire f for this radius
repeat until no family improves; then radius += 1
```

## 7. Risks (prototype/measure early)

- **Cost of one candidate topology → gpurec input.** Re-running the Rust
  preprocessor per neighbor could dominate. Needs a cheap/incremental
  single-topology path. **First thing to profile-prototype.**
- **Rooting-convention parity** (gpurec single-Newick vs GeneRax rooted-GFT
  handling). The fixed-rate recon-parity test (below) will catch mismatches.
- **CPU sequence term becoming the bottleneck** — instrument seq-vs-recon wall
  time from day one; this measurement is precisely the Phase-2 trigger.
- **Batch memory.** Neighbor count ≈ O(taxa × radius) per family × F families;
  chunk the recon batch if it exceeds GPU memory (gpurec already chunks families).
- **Build system.** Compiling coraxlib + the needed GeneRaxCore pieces into the
  pybind11 extension via CMake is real setup work; treat as its own plan step.

## 8. Validation (chosen success metric)

- **Unit — sequence term:** `_seqlik.seq_loglk` == GeneRax `LibpllEvaluation` on
  identical tree + model (should be near bit-identical; same library).
- **Unit — reconciliation term:** gpurec recon-loglk == GeneRax `UndatedDTLModel`
  at fixed tree + fixed rates, per family, within tolerance.
- **Integration:** full pipeline on a shared dataset → per-family joint-loglk
  within tolerance **and** low RF vs GeneRax's output trees.
- **Regression:** reproduce a slice of the paper's cyanobacteria / primates
  comparison (joint log-likelihoods, Fig. 9).

## 9. Explicitly out of scope for Phase 1

- GPU Felsenstein pruning engine (Phase 2, gated on profiling).
- Species-tree search / SpeciesRax (`--si-strategy`).
- MPI multi-node scheduling (single-node GPU + CPU thread pool only).
- Non-UndatedDTL reconciliation models beyond what gpurec exposes (UndatedDTL
  first; UndatedDL if trivially available).

## 10. Reference paths

- GeneRax orchestration: `generax_gpurec/generax/src/GeneRax/GeneRaxCore.cpp`
  (`geneTreeJointSearch` :231, warm-up :238, main :243, `optimizeRatesAndGeneTrees` :385, `enableRecLL` :425).
- Joint coupling: `ext/GeneRaxCore/src/trees/JointTree.cpp` (:190 libpll, :197 recon, :204 joint).
- Sequence engine (keep): `ext/GeneRaxCore/src/likelihoods/LibpllEvaluation.{cpp,hpp}`, `ext/coraxlib`.
- Reconciliation engine (replace): `ext/GeneRaxCore/src/likelihoods/ReconciliationEvaluation.{cpp,hpp}`, `reconciliation_models/UndatedDTLModel.hpp`.
- SPR search: `ext/GeneRaxCore/src/search/SPRSearch.cpp`.
- Rate optimization: `ext/GeneRaxCore/src/optimizers/DTLOptimizer.cpp`, `PerFamilyDTLOptimizer.cpp`.
- gpurec API: `gpurec/api/model.py` (`GeneReconModel`), `gpurec/optim/optimize.py`, `gpurec/optim/genewise_fit.py`.
- gpurec reconciliation sampling: `gpurec/core/backtracking/input.py` (`sample_reconciliations`).
- gpurec preprocessing (single Newick → degenerate CCP): `crates/gpurec-preprocess/src/lib.rs`.
