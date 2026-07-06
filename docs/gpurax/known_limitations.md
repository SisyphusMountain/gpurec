# gpurax — Known Limitations & Follow-Ups (Phase 1)

Phase 1 delivers a working GeneRax-equivalent: a joint sequence×reconciliation
ML gene-tree search using gpurec (batched GPU reconciliation) + libpll/coraxlib
(sequence term). On the test fixtures it recovers the true gene tree and matches
GeneRax's reconciled tree exactly (RF = 0). The items below are deliberately
deferred; none is a correctness bug on supported input.

## Resolved before merge
- **Gene→species mapping now applied** (`gpurax/io/prepare.py`): gene-tree leaves
  and alignment headers are relabeled to `<species>_<gene>` via the family's
  mapping file, so real GeneRax datasets (arbitrary gene leaf names) work — not
  only species-prefixed inputs. No-op on already-prefixed leaves.
- **Test-suite portability**: driver/CLI/parity tests generate their families
  file at runtime; no machine-specific absolute paths are consumed.

## Deferred follow-ups (prioritized)

### High priority for real many-family runs
1. **GPU-memory chunking of the cross-family candidate batch** (spec §7).
   `search/spr.py::spr_search` materializes every neighbor of every active
   family into one list and `recon/adapter.py::ReconBatch.score` builds a single
   genewise `GeneReconModel` over all of them. Neighbor count ≈ O(taxa·radius)·F;
   the "many families" workload can OOM. Chunk the recon batch (gpurec chunks
   *families* internally, but here all candidates go through one genewise model).
2. **Per-family error isolation in the driver** (`driver.py`). One malformed
   family currently aborts the whole run. Wrap per-family stages in try/except +
   skip-list so a 1000-family batch survives a few bad families. Compounds with
   the mapping validation (a bad mapping should skip its family, not the run).
3. **Model-construction overhead per `score()` call** (Task C3, Task I3 finding).
   Each `ReconBatch.score` rebuilds a `GeneReconModel` (~0.6 s on the toy
   fixture, dominated by construction, not compute). On real search trajectories
   this dominates; use the incremental preprocess path (`build_family_ccp` /
   `replan_batches`) to reuse the model across steps. Measure first
   (`JointScorer.timings`) — do not assume a speedup.

### Medium / quality
4. **Branch-length optimization during SPR** (`opt_bl=False`). Candidates are
   ranked with unoptimized branch lengths; GeneRax optimizes BLs during move
   evaluation. RF parity holds on the fixtures, but on harder real data this can
   misrank moves. Consider `opt_bl=True` (or `optimize_all`) at materialization,
   handling the apply/opt/rollback interaction carefully.
5. **RecPhyloXML transfer events** emit only `<transferBack>` on the transfer
   node; GeneRax pairs `<branchingOut>` (donor) + `<transferBack>` (recipient).
   Well-formed and destination-correct, but third-party viewers (Thirdkind) may
   not render transfers. Fix if full visualizer compatibility is needed.
6. **RecPhyloXML gene-clade names are `NULL`** — gpurec's backtracking node list
   carries species indices, not gene-leaf names (matches GeneRax's own fallback).
   Recover original gene-leaf labels via a tree↔node-list alignment if needed.
7. **Per-family / per-species DTL rates** not exposed (spec §5). Phase 1 is
   global rates only (spec §2 "global first"). gpurec supports both
   (`fit_genewise`, specieswise) — add `--per-family-rates` / `--per-species-rates`.
8. **Species names must be `_`-free.** gpurec derives species from the leaf-name
   prefix before the first `_`; the relabeler validates and fails loudly if a
   species name contains `_`. Supporting `_`-bearing species names would need a
   different separator convention through gpurec's preprocessor.

### Minor
- `test_spr_multi.py` uses identical families/rates (cross-family attribution was
  verified separately with differentiated rates during review); harden with a
  differentiated-rates case.
- Register the `benchmark` pytest marker (or drop it) — currently a harmless
  `PytestUnknownMarkWarning` in `test_recon_cost.py`.
- Driver re-runs radii `1..r` on each outer iteration (O(max²) sweeps); correct
  and GeneRax-like, but a `min_radius` tweak or comment would clarify intent.
- Package `gpurax` and declare its deps (`dendropy`, etc.) when it graduates from
  the in-repo subsystem to a distributed package (`pyproject.toml` currently
  packages only `gpurec*`).

## Out of scope for Phase 1 (planned Phase 2+)
- GPU Felsenstein pruning engine (port the sequence term to GPU) — gated on the
  seq-vs-recon timing signal (`JointScorer.timings`) at realistic scale.
- Species-tree search / SpeciesRax; MPI multi-node scheduling.
- Non-UndatedDTL reconciliation models.

## Reconciliation-parity note
gpurec's UndatedDTL is validated == AleRax (2.6e-4 nats) but is NOT bit-identical
to GeneRax's UndatedDTL — they differ by a topology-independent survival
normalization (harmless to the search argmax) plus a small topology-dependent
per-rooting term. Success is measured by **tree quality (RF)**, and the tool
matches GeneRax's reconciled tree (RF = 0) on the fixture. See
`docs/gpurax/reconciliation_convention.md`.
