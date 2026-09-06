# Coleman: count-informed EM warm-up campaign

Status: initial EM campaign and the user-requested post-EM hierarchical screening
are complete (see [hybrid results](hybrid/RESULTS.md)). All cluster jobs have finished. Two GPT-5.6 Sol workers only;
the coordinator chose experiments, controlled GPU leases, reviewed integration,
and ran the full H100 comparisons.

## Current result

Two complete-history EM warm-up steps, followed by the existing BFGS/Newton
optimizer, completed all 5,124 families in **396.305 seconds** on one H100 NVL.
All families passed the original freeze-time projected-gradient certificate.
Reported NLL was **9,049,360.7436 bits**, versus the historical best of
9,049,362.363 bits at 520.5 seconds. This is 23.9% less time than that historical
record. The contemporaneous warmed Adam run took **512.718 seconds** on the same
H100: EM2 saves **22.7% time and 21.3% gradient/clade work** against that control.
Fresh quality audits confirm the total NLL improvement, with the numerical and
per-family qualifications below. The EM2 repeat took 403.470 seconds: its
two-run mean is **399.888 seconds**, 22.0% below warmed Adam. EM3's two-run mean
is **397.345 seconds**, 22.5% below Adam. Their 0.6% mean-time separation is too
small, with only two runs each, to establish a decisive winner. Both are about
400 seconds under the unchanged benchmark contract.

| Full 5,124-family run | Fit seconds | NLL, bits | Certified | Peak allocated GiB |
| --- | ---: | ---: | ---: | ---: |
| Historical best | 520.5 | 9,049,362.363 | 5,124 | — |
| Current Adam, cold startup | 634.016 | 9,049,362.3797 | 5,124 | 39.73 |
| Current Adam, warmed comparison | 512.718 | 9,049,362.3770 | 5,124 | 39.74 |
| EM2 A | 396.305 | 9,049,360.7436 | 5,124 | 23.59 |
| EM2 B | 403.470 | 9,049,360.7546 | 5,124 | 23.57 |
| EM3 A | 398.290 | 9,049,360.6102 | 5,124 | 23.42 |
| EM3 B | 396.401 | 9,049,360.5977 | 5,124 | 23.02 |

The cold baseline is **not** used as the sole denominator for a speedup claim:
its Adam phase cost 147.55 seconds rather than the historical roughly 70 seconds.
Its Newton gradient phase, 383.55 seconds, was close to the historical 379.
The warmed Adam comparison reproduced 70.59 seconds for Adam and 378.49 for
Newton gradients; three exact Hessian refreshes cost 31.09 seconds.
EM2 cost 48.97 seconds for warm-up, 306.78 for Newton gradients, 11.42 for one
exact Hessian refresh, and about 11 for replanning. Its 43 gradient calls cost
13.403 full-dataset clade equivalents, charging settled-but-resident families.
Hessian work is reported separately, not disguised as gradient equivalents.

Three EM steps passed the 500-family gate, but their full-scale speed gain did
not carry over: EM3 A took 398.290 seconds and 13.466 gradient/clade equivalents.
Its slightly lower NLL and two-second timing difference did not settle the
choice from one run. Both variants were therefore repeated. EM3's two-run mean
is 397.345 seconds; its two fits differ by only 0.00121 bits in the common-model
fresh audit. EM2 consistently uses slightly less gradient/clade work (13.400
versus EM3's 13.463 equivalents, two-run means) and one fewer mandatory full
warm-up pass. EM3 has a long but cheap tiny-family tail. Both options are retained;
EM2 is the simpler opt-in, while EM3 has the slightly better measured mean time
and aggregate NLL. Neither has demonstrated per-family likelihood dominance.

## Fresh quality audit: important qualifications

These are independent, same-model evaluations at the original pruning threshold
1e-6, with the original 16/16 fallback budgets and FP32/FP64 precision policy.
Audit time is not included in either fit's benchmark time, consistently with the
historical cached-certificate recipe.

| Fresh comparison | Total NLL change, bits | Families worse/better by >0.01 bit | Fresh Pg <1e-3: Adam / candidate |
| --- | ---: | ---: | ---: |
| EM2 A versus Adam | -1.63451 | 13 / 12 | 5,058 / 5,053 |
| EM2 B versus Adam | -1.63089 | 13 / 12 | 5,056 / 5,051 |
| EM3 B versus Adam | -1.76365 | 14 / 12 | 5,052 / 5,052 |

The largest family regression is +1.4897 bits; the largest improvement is
-2.3865 bits. EM improves the total objective, **not every family's optimum**.
No fitted baseline rates were used to choose the candidate's per-family results.

The original "5,124 certified" claim means the stored freeze-time certificate.
It does not survive fresh evaluation of every family at a strict 1e-3 threshold
for Adam either. In the EM3 audit, evaluating **identical Adam parameters twice**
changed the maximum per-family projected-gradient measurement by 0.001292 and
changed the passing count from 5,052 to 5,050. The NLL vectors repeated exactly.
Only 36 of the first evaluation's 72 failing families also failed the repeat.
This demonstrates sensitivity of that threshold to gradient arithmetic; it is
not evidence of a new EM-specific failure or proof of stricter stationarity.

EM3's two independent fits have no per-family fresh NLL difference above
0.001742 bits. The total likelihood gain is much larger than that fit-to-fit
variation. Complete per-family vectors and changes are in the `results/audit_*.pt`
and `.json` files; see also [the independent quality review](em/QUALITY_REVIEW.md).

The final EM2 repeat audit reached the same conclusion: its two fits differ by
0.003619 bits total and at most 0.001310 bits per family, with no family above
0.01 bit. The identical-Adam repeat's maximum projected-gradient change was
0.001515. Across all fresh passes, Adam certified 5,049–5,058 families, EM2
5,048–5,053, and EM3 5,052–5,055. These are numerically sensitive fresh counts,
not new optimizer stopping criteria. Separate audit wall times were 215.94 s
(EM3 plus repeats), 163.04 s (EM2 A comparison), and 221.69 s (EM2 plus repeats),
including each audit's cold model construction.

## What changed, and why

Keep the production log2 rate coordinates and the original box [1e-6, 2].
Improve initialization and its curvature instead of replacing the entire
optimizer with EM or changing the likelihood or stopping tolerance.

1. The existing implicit reverse pass optionally returns positive complete-history
   counts in S,D,L,T order. These include the extinct "ghost" histories needed
   for survival conditioning. Count extraction adds intermediate inputs to the
   same final VJP; there is no additional extinction solve or global hook.
2. For frozen counts N, maximize `Q = sum_k N_k log p_k` in the original box.
   The unconstrained rates are `D=N_D/N_S`, `L=N_L/N_S`, `T=N_T/N_S`.
   Bounds couple these ratios: enumerate all 27 lower/free/upper active sets
   and check KKT conditions. Irreversible clipping is incorrect.
3. After two or three M-steps, build the fixed-count complete-information Hessian
   at the endpoint: `I_c = ln(2) N_total (diag(p)-p p^T)`, where p contains D,L,T.
   Scale it by `(s^T y)/(s^T I_c s)` when that ratio is finite and positive,
   then apply the existing safeguarded BFGS update using the latest EM secant.
   Both gradients were already paid for by the EM steps. Only coordinates
   interior at both secant endpoints contribute to the BFGS update.
4. Resume the unchanged trust-region/Newton loop, exact-refresh policy,
   convergence threshold, and clade-based replanning.

This is a surrogate-curvature initialization, not the observed Hessian.
Complete-data curvature alone overestimates useful observed curvature because
latent-history uncertainty removes information. The measured secant supplies
an inexpensive correction. See [the independent mathematical review](geometry/EM_MATH_REVIEW.md)
for ghost semantics, the gradient identity, and the finite-precision caveats.

## Experiment selection

Claude's recovered EM notes were newer than the original mathematical report.
They already showed that plain EM/SQUAREM had poor tails, and that the first
EM steps were useful basin entry. We corrected the bounded M-step before testing.
Saved fitted optima were used only for evaluation, never as optimizer inputs.

| Local RTX4090, 500 families | Prototype wall seconds | Actual gradient/clade equivalents |
| --- | ---: | ---: |
| Adam baseline | 108.330 | 17.273 |
| EM2 | 91.506 | 13.660 |
| EM3 | 87.206 | 13.299 |

All three certified all 500 families. EM3's fresh forward NLL was 0.0142 bits
better than Adam; EM2 was 0.0276 bits worse. Four families changed basins by more
than 0.01 bits relative to Adam. EM3 versus EM2 had no family difference above
0.01 bits. The integrated EM3 driver took 81.422 seconds and 13.343 equivalents;
integration removes the prototype's redundant parse/model construction.

The second worker tested hierarchical logits
`u=log2((D+T)/(1+L)), v=log2(T/D), w=log2(L)`.
They diagonalize complete-history curvature and substantially improve local
Hessian definiteness, but the tested transformed BFGS variants did not beat EM
and left a difficult uncertified tail without exact refresh. That route was
closed; the worker independently implemented and tested the count-output API.

Scope correction: those hierarchical experiments started **after Adam**, not
after EM. They do not rule out the EM-plus-hierarchical hybrid. The user identified
that missing comparison; it has now been tested from a shared EM endpoint with
coordinate-consistent count curvature and scheduled exact-Hessian refreshes.

The corrected 200-family test certified every family in all arms, with no
zero-step stalls. In two order-balanced repetitions, native log rates averaged
25.459 s of continuation and 13.5399 gradient/clade equivalents including EM;
hierarchical coordinates with the native trust metric averaged 26.777 s and
14.0009 equivalents. Fresh likelihoods were effectively equivalent. The single
coordinate-metric sensitivity run also used more work. This implementation was
not promoted to a full H100 run; the roughly 400-second production result above
is unchanged. This is limited evidence about the tested bound solver and
globalization, not a claim that every post-EM reparameterization is ineffective.

The currently validated reparameterization is confined to the analytic warm-up:
use event probabilities/count ratios for the surrogate solution, then return to
native log rates. This preserves the axis-aligned rate box and the tested Newton
machinery instead of introducing curved box constraints in hierarchical logits.

## Reproduction and scope

The default recipe remains Adam. For the simpler EM2 option, use:

```python
fit_dtl(species_tree, gene_trees, "genewise", device="cuda",
        genewise_warmup_method="em", genewise_em_steps=2)
```

The benchmark adds `--warmup-method em --em-steps 2` (or 3) to the existing
`benchmark/cc/run_genewise.py` command. Use every family, `--limit 0`,
`--init-rate none`, and `--clade-budget 0` for the recorded H100 recipe.
The Slurm scripts in this directory provide complete commands.

EM2 recorded the best single time (396.305 s) and slightly less gradient work;
EM3 recorded the better two-run mean (397.345 versus 399.888 s) and slightly
better total NLL. Use `genewise_em_steps=3` if preferring that measured mean/NLL
tradeoff. The practical result is a roughly 400-second recipe, not evidence
that one additional EM step is universally better. More repetitions would be
needed to resolve a sub-percent timing difference confidently.

Preserved: all 5,124 families including COG3676_X, rate bounds, initialization,
model/accumulator precision, pruning threshold, solver tolerances, original
gradient threshold 1e-3, and certification policy. The certificate is a cached
freeze-time **pruned FP32** measurement, not proof of an unpruned stationary
point or a global maximum. Fresh matched audits are recorded outside fit time.

Source snapshots and Slurm job IDs are in [PLAN.md](PLAN.md). The initial source
for changed existing files is archived in `source_baseline/`, making the campaign
diff separable from pre-existing user/Claude changes. No changes were reverted.

Validation: 91 focused tests passed, covering counts, bounded M-step,
configuration/default preservation, public option wiring, and existing solver
paths. Manual integrated checks passed at 5, 200, 500 and 5,124 families.
