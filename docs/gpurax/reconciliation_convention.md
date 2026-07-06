# gpurec vs GeneRax reconciliation convention

This note documents a **known, diagnosed, and accepted** difference between gpurec's
`UndatedDTL` reconciliation likelihood and GeneRax's reported `reconciliation:` value.
It is not a bug to fix — it is a modelling-convention difference that has been root-caused
and is left as-is (no gpurec core edits). The full investigation trail lives in
`scratchpad/gpurax/{rooting-investigation,offset-constancy,residual-diagnosis}.md`
(not committed; kept for reference by whoever revisits this).

## Engine validation

gpurec's engine is the same `UndatedDTL` model used throughout gpurec/gpurax. It is
validated against AleRax to **2.6e-4** nats (`tests/test_fidelity_alerax.py`). The
gpurec-vs-GeneRax difference documented below is a separate, later finding — it does not
call the AleRax fidelity result into question.

## The difference has two components

Comparing gpurec's `GeneReconModel` reconciliation log-likelihood against GeneRax's
`reconciliation:` output on a 4-taxon fixture (species tree `((A,B),(C,D));`,
D=0.2, L=0.3, T=0.1), the gap decomposes into:

1. **Survival (observability) normalization — topology-independent.**
   gpurec divides the per-family likelihood by the survival factor
   `Σ_s (1 − 2^{E_s})` (`gpurec/core/inference/solver.py`, `nll_vector_from_root_rows` /
   `log2_survival`). GeneRax's *reported* `reconciliation:` value applies no such factor.
   This was confirmed by GeneRax's clean-tree (rates → 0) log-likelihood staying ≈0
   independent of species-tree size (4/8/16-taxon: −0.020/−0.045/−0.093), whereas gpurec's
   clean-tree value tracks `−log(#species nodes)` exactly at every depth tested. Because
   this factor depends only on the species tree and rates — not on the gene-tree topology —
   it is the **same additive term for every SPR candidate of a given family**, and is
   therefore harmless to the SPR-search argmax.

2. **Per-rooting DTL event-recursion term — topology-dependent.**
   After accounting for (1) and for the rooting-count convention (gpurec's CCP preprocessor
   assigns a uniform `1/(2n−3)` prior across the candidate rootings of an unrooted gene
   tree, matching the ALE/AleRax convention; GeneRax sums the same candidate rootings
   unweighted), a residual remains in the DTL event-probability recursion that combines
   child-clade values near the root (`UndatedDTLModel::computeProbability`, GeneRax C++).
   This term is **~0 at the pure-speciation limit** (any tree depth, any rates → 0) and
   grows with event rate and tree size — roughly **0.07 nat per species node** at baseline
   rates (D=0.2, L=0.3, T=0.1). It is larger for gene trees that require duplication/transfer
   events to reconcile against the species tree than for gene trees reconcilable by pure
   speciation. This is a genuine implementation difference between the two tools'
   recursions; it has been localized (root-adjacent DTL combination terms) but not pinned
   to an exact line-for-line cause, and is **left as-is** — no gpurec core edits are made to
   chase bit-for-bit parity.

## Ruled out

Two plausible bugs were investigated and both ruled out with direct numeric evidence:

- **A `×ln(2)` base-conversion bug.** The gap does not move as an integer multiple of
  `ln(2)` or `(1 − ln(2))`, and it scales smoothly and continuously with the DTL rates —
  inconsistent with a fixed base-conversion constant. `extract_parameters.py`'s
  `log_softmax(logits·ln2)/ln2` was audited and confirmed to return the correct
  `log2(P_S/P_D/P_L/P_T)` values, bit-identical to GeneRax's `setRates` normalization.
- **A DLT-vs-DTL rate-order swap.** Both tools use the same rate order. GeneRax's
  `GeneRaxInstance::getUserParameters` builds `Parameters(dup, loss, transfer)` and
  `UndatedDTLModel::setRates` reads `rates[0]=dup, rates[1]=loss, rates[2]=transfer` — i.e.
  GeneRax is DLT, identical to gpurec's `theta = [log2 D, log2 L, log2 T]`. An asymmetric
  all-distinct-rate experiment (D=0.2, L=0.5, T=0.05) confirmed gpurec matches GeneRax's
  correctly-ordered run and is grossly off against a deliberately L/T-swapped run, and a
  symmetric-rate control (D=L=T, where any permutation would be invisible) still shows the
  residual — so the residual cannot be an order/permutation artifact.

## Consequence for gpurax

Because of component (2), **joint reconciliation log-likelihoods will not bit-match
GeneRax**, and this is expected, not a regression to chase. Success for gpurax's tree
search is measured by:

- **RF distance / tree-quality** against GeneRax's output trees (Task I1), not by matching
  scalar log-likelihood values, and
- **Ranking consistency**: gpurec must rank candidate gene-tree topologies in the same
  order GeneRax would, since the SPR search only ever needs the *argmax* over topologies,
  not the absolute likelihood value. Component (1) above is topology-independent so it
  cannot change any ranking; component (2) is topology-dependent, so ranking agreement is
  verified empirically rather than assumed.

## Ranking-consistency test

`tests/gpurax/test_recon_ranking.py` checks this directly on the 3 distinct 4-taxon
gene-tree topologies over species tree `((A,B),(C,D));`:

| topology | gene tree (unrooted) | gpurec logL (nats) |
|---|---|---|
| T1 | `((A_a,B_b),(C_c,D_d));` (matches species tree — pure speciation) | −5.910620 |
| T2 | `((A_a,C_c),(B_b,D_d));` | −9.572941 |
| T3 | `((A_a,D_d),(B_b,C_c));` | −9.572941 |

gpurec ranks **T1 as most likely**, with **T2 == T3** exactly (as expected from the
C↔D automorphism of the species tree). GeneRax ranks the same three topologies identically
(T1 best, T2 == T3), cross-checked in the underlying investigation
(`scratchpad/gpurax/offset-constancy.md`). This is the property that matters for the SPR
search, and it holds despite the two tools' scalar log-likelihoods not being bit-identical.
