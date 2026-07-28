# gpurec vs AleRax: origination-distribution gap on duplication+transfer families — RESOLVED

## Summary

`tests/test_backtracking_matches_alerax.py` compares gpurec's backtracking
sampler against AleRax v1.4.0's own sampler (`-g/--gene-tree-samples`) on the
species where a gene family's reconciliation history is inferred to have
started ("origination"). On a fixture where duplication AND transfer
genuinely compete as alternative explanations, the two initially disagreed by
a real, reproducible margin, even though the existing
`tests/test_fraction_missing_alerax.py` already externally confirms gpurec's
total per-family likelihood matches AleRax's to ~1e-5 nats on the base DTL
model (no fraction-missing). **Two independent implementations agreeing on a
sum while disagreeing on how it's split up** was the situation this
investigation started from.

**Root cause: a test-methodology mismatch, not a model bug.** gpurec's CCP
preprocessing always marginalizes uniformly over every possible rooting of
the input gene tree (a 4-leaf tree has 5 possible root edges; gpurec's
`root_clade_id`'s clade had exactly 5 alternative splits, each at probability
`1/5` — see "Resolution" below). AleRax does the same **by default**, but
this investigation's test harness passed `--gene-tree-rooting ROOTED` to
force it to condition on the *one* given rooting instead — reasoning, at the
time, that this was needed to match gpurec (which has no equivalent
"respect-the-given-rooting" option to turn off). That reasoning was
backwards: it made AleRax compute a genuinely different quantity than gpurec,
for no good reason. Removing that flag (i.e. leaving AleRax at its default,
which already matches gpurec's actual behavior) resolves the discrepancy
completely — see the numbers below.

Three other hypotheses were tested and refuted before finding this (kept
below since they document real, verified facts about both codebases that
remain useful context — the transfer-admissibility rule, the uniform-receiver
normalization, and the fixed-point convergence are all confirmed to match
between gpurec and AleRax).

## Exact reproduction

- Species tree: `((A:1,B:1)AB:1,C:1)Root;` (5 species branches: A, B, C, AB, Root)
- Gene tree: `(((A_1:1,B_1:1)x:1,C_1:1)y:1,A_2:1)GeneRoot;` — this is the
  `_GENE_DUP` fixture from `tests/test_fraction_missing_hvp.py`. It forces a
  real duplication (there are two "A" copies: `A_1` under the `x`/`y`
  subtree, `A_2` as `GeneRoot`'s other child) — as opposed to the
  topology-matching gene tree used by the existing NLL parity test, which is
  rate-degenerate (only one reconciliation is possible at all, so it can't
  exercise this).
- Rates: D = L = T = 0.3 (global mode), fraction_missing = 0 (not involved —
  see the module docstring in `test_backtracking_matches_alerax.py` for how
  fraction-missing was isolated cleanly on a transfer-free fixture).
- gpurec: `GeneReconModel(..., mode="global", ...)`, `theta = log2([0.3, 0.3, 0.3])`,
  default (uniform) `receiver_weights`, sampled via
  `gpurec.core.backtracking.input.sample_reconciliations`, 2000 draws
  (seeds 0..1999).
- AleRax: `alerax -f families.txt -s sp.nwk -g 2000 --seed 0
  --species-tree-search SKIP --rec-model UndatedDTL --model-parametrization GLOBAL
  --fraction-missing-file fm0.txt --min-covered-species 3
  --gene-tree-rooting ROOTED --fix-rates --d 0.3 --l 0.3 --t 0.3`,
  origination read from `family_0_speciesEventCounts_<k>.txt`'s `origination`
  column (exactly one species per sample has `origination=1`).

## Observed discrepancy

Origination species over 2000 samples each (counts, both sum to 2000):

| species | gpurec | AleRax (default `--transfer-constraint`) |
|---|---|---|
| Root | 1103 | 1080 |
| AB   | 549  | 441  |
| C    | 144  | 143  |
| A    | 137  | 283  |
| B    | 67   | 53   |

Root, C, and B are close (within sampling noise). **A is off by ~2x, AB by
~25%.** This is a large, reproducible gap, not noise (chi-square against this
table rejects overwhelmingly).

The same divergence shows up more granularly in the raw sampled transfer
events (`family_0_transfers_<k>.txt` on the AleRax side,
`sample_family_counts` on the gpurec side, same fixture, same rates): every
`(donor, recipient)` pair AleRax reports also appears in gpurec's output in
roughly the same ballpark, **except `Root→A`**, which is AleRax's single
largest transfer category (741/3480 = 21.3% of all sampled transfers) but one
of gpurec's smallest (150/3205 = 4.7%).

## Hypotheses tested and refuted

### 1. Transfer-admissibility rule mismatch — refuted

AleRax's CLI defaults to `--transfer-constraint PARENTS`, and the log prints
"transfers to parents are forbidden" — which reads as if only the *immediate*
parent is excluded, unlike gpurec's rule (exclude *all* ancestors of the
donor; `crates/gpurec-backtrack/src/lib.rs`'s `is_ancestor` check). Forcing
`--transfer-constraint NONE` to match what the name suggested gpurec does
made the gap **worse** (Root's origination share dropped from 54% to 40% and A's
rose to 36%, moving further from gpurec's 6.9%), which contradicted the
premise, so this needed checking properly rather than accepting the first
plausible story.

Reading the actual source
(`AleRax/src/ale/UndatedDTLMultiModel.hpp:259-264`,
`AleRax/ext/GeneRaxCore/src/trees/PLLRootedTree.cpp:505-510`) shows
`PARENTS` calls `isAncestorOf`, which is backed by a full LCA/ancestor cache
(`_lcaCache->ancestors[...]`), not a direct-parent-only check — despite the
name, AleRax's **default** setting excludes *all* ancestors of the donor,
identical to gpurec's rule. `NONE` is the mismatched setting, not `PARENTS`;
using `NONE` explains why that experiment made things worse. All comparisons
in this document use AleRax's default (`PARENTS`, i.e. no explicit
`--transfer-constraint` flag), which is confirmed to match gpurec's rule.

### 2. Uniform-receiver double-normalization — refuted

`crates/gpurec-preprocess/src/lib.rs:436-449` computes `unnorm_row_max[s] =
-log2(S - depth(s))` — a per-donor normalizer by the count of admissible
recipients, matching AleRax's `getTransferWeightNorm(e)` (a plain admissible-
recipient count divisor, `UndatedDTLMultiModel.hpp:89-92`). Separately,
`gpurec/core/inference/solver.py:64` sets a **global** constant
`receiver_log_probs = -log2(S)` for the uniform case. Comparing
`extract_parameters_uniform`'s `max_transfer` against
`extract_parameters_weighted_receivers`'s (fed near-uniform but *not exactly
equal* receiver weights, to force the general code path) showed the two
differ by exactly `log2(S)` for every species — which looked exactly like a
missing `+log2(S)` correction in `unnorm_row_max` (the weighted path's
`receiver_valid_log_normalizer`, `extract_parameters.py:116-146`, computes
`log2(S) - log2(N_admissible)`, not just `-log2(N_admissible)`).

This looked like a real bug, but sampling both paths (default uniform vs.
forced-weighted-near-uniform) on the exact discrepancy fixture gave
**bit-identical results, seed for seed** — no difference at all. Tracing why:
`gpurec/core/kernels/pi_forward.py`'s `_compute_total_receiver_mass` only adds
`receiver_log_probs` into the sum `if USE_RECEIVER_WEIGHTS:` — the uniform
(fast) path sets that flag `False` and skips the addition entirely
(`solver.py`'s `use_receiver_weights = not receiver_weights_are_uniform(...)`).
So the "missing `+log2(S)`" in `unnorm_row_max` is not missing at all once you
account for the fact that the `-log2(S)` it would need to cancel is *also*
never added in that code path. The two formulations are mathematically
equivalent by construction, confirmed both by direct value comparison
(`max_transfer` differs by exactly `log2(S)`, as expected once you know one
path adds a compensating `-log2(S)` elsewhere and the other doesn't) and by
the sampled output being identical.

### 3. AleRax's fixed 4-iteration CLV fixed point vs gpurec's full convergence — refuted

The existing NLL parity test already documents that AleRax's *extinction*
fixed point runs a hard-coded 4 iterations
(`UndatedDTLMultiModel.hpp::recomputeSpeciesProbabilities`, `maxIt = 4`) and
never fully converges, while gpurec iterates to convergence — accounting for
a small (~2.5e-4 nat) but real residual at fraction_missing=0.3. The clade
CLV computation (`updateCLV`, the Pi/Pibar equivalent) uses the exact same
`maxIt = this->_info.noTL ? 1 : 4` pattern
(`UndatedDTLMultiModel.hpp:212`), which looked like a strong candidate for
the *same* mechanism now affecting the reconciliation numerator, not just the
extinction denominator.

Sweeping gpurec's `pi_iters` down from 256 (fully converged) to 2 (`model.
configure_solver(pi_iters=N)`, same fixture) produced **no visible change at
all from pi_iters=4 onward, and only a 1-2 count wobble at pi_iters=2** (out
of 2000 samples) — the fixed point for this tiny fixture converges almost
immediately, well within AleRax's 4 iterations too. Under-convergence cannot
be the explanation here: both implementations are effectively at their
converged value by iteration 4.

## What's confirmed to match (so the search can stay narrow)

- The base rate normalization (softmax of `[0, dup_logit, loss_logit,
  transfer_logit]` in log2 units) — externally validated exactly by
  `test_fraction_missing_alerax.py::test_base_model_matches_alerax_fm0`.
- The transfer-admissibility set (exclude all ancestors of the donor,
  including itself).
- The uniform-receiver aggregate-transfer formula (`sum over admissible
  recipients, divided by their count`) — same in both, confirmed by direct
  source comparison and a bit-identical empirical cross-check.
- The clade-splitting D/T event composition formulas
  (`computeProbability`, `UndatedDTLMultiModel.hpp:513-673`, vs
  `sample_term`'s `SplitDup`/`SplitTransfer` candidates,
  `crates/gpurec-backtrack/src/lib.rs`) — read line by line, structurally
  identical (`Pi(left)*Pi(right)*P_D*freq` for duplication;
  `Pi(left)*Pibar(right)*P_T*freq` [+ symmetric swap] for transfer, matching
  AleRax's `uq(cidLeft)*uq(cidRight)*(PD*freq)` and
  `uq(cidLeft)*tq(cidRight)*(PT*freq)`).
- Convergence: the fixed point for this fixture is essentially converged by
  iteration 4 in gpurec; AleRax's fixed 4 iterations is not under-converged
  here.

## Resolution

Rebuilt AleRax locally (`/home/enzo/Documents/alerax_1_4/AleRax/build/bin/alerax`,
never touching the system-installed `/usr/local/bin/alerax`) with one debug
print added to `UndatedDTLMultiModel.hpp::updateCLV`, gated on
`getenv("GPUREC_DUMP_CLV")`, dumping `_dtlclvs[cid]._uq`/`._tq` for every
`(cid, species)` pair (reverted after use — see the git diff of that repo for
the exact one-line change if it needs redoing). This produced AleRax's raw
CLV table: **7 clades** (one per gene-tree node: `A_1`, `B_1`, `C_1`, `A_2`,
`x`, `y`, `GeneRoot` — matching the input newick exactly, as expected under
`--gene-tree-rooting ROOTED`).

Printing gpurec's own `family["split_parents_sorted"]` /
`split_leftrights_sorted"]` / `log_split_probs_sorted"]` for the identical
gene tree (see `_family_arrays` in `tests/test_backtracking_matches_alerax.py`
for how to extract these) showed **11 clades**, with `root_clade_id`'s clade
having **five** alternative splits — `(1,7)`, `(2,8)`, `(3,9)`, `(4,10)`,
`(5,6)` — each at `log_split_probs = -log2(5)`, i.e. exactly uniform
probability `1/5`. A 4-leaf tree has exactly 5 edges to root on if treated as
unrooted; gpurec is marginalizing uniformly over all of them. There is no
`unroot`/`rooting` option anywhere in `gpurec/` or `crates/` to turn this
off — it's simply how gpurec's CCP preprocessing always behaves for a single
input gene tree.

AleRax's own startup log states its default matches this exactly: *"Gene
tree rooting: all gene tree root positions are considered with the same
probability"* — `--gene-tree-rooting ROOTED` is the **non-default** override,
and it was the wrong one to reach for. Re-running the origination-distribution
comparison without that flag (AleRax at its default, gpurec unchanged):

| species | gpurec (fm=0) | AleRax, default rooting (fm=0) | gpurec (fm(B,C)=0.8) | AleRax, default rooting (fm(B,C)=0.8) |
|---|---|---|---|---|
| Root | 1103 | 1122 | 1205 | 1218 |
| AB   | 549  | 510  | 444  | 448  |
| C    | 144  | 166  | 188  | 157  |
| A    | 137  | 136  | 81   | 88   |
| B    | 67   | 66   | 82   | 89   |

Two-sample chi-square test of homogeneity: **p=0.53** at fm=0, **p=0.49** at
fm(B,C)=0.8 (2000 samples each side, same fixture and rates as throughout
this document). Both comfortably non-significant — this is a clean,
unqualified match, on the *full* duplication+transfer+fraction-missing model
this investigation was trying to validate, not the transfer-avoiding
workaround fixture from the original test.

## Follow-up: fix `tests/test_backtracking_matches_alerax.py`

The committed test avoided this entirely by using a transfer-free (T≈0)
fixture specifically *because* the real cause wasn't understood yet, and (less
critically) it also passed `--gene-tree-rooting ROOTED` on that fixture, which
this finding says was never the right call. Both should be corrected: drop
`--gene-tree-rooting ROOTED` everywhere, and prefer testing the full
duplication+transfer+fraction-missing model now that it's known to check out,
rather than the narrower workaround.
