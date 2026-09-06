# Independent quality review of the full EM2/EM3 audits

## Bottom line

The speed comparison is fair under the requested, existing convergence definition. Adam and both
EM variants use the same likelihood, rate box, float32/float64 precision policy, adjoint pruning threshold,
projected-gradient tolerance (`1e-3`), freeze-time certificate, and final fit code. EM changes only
the warm-up trajectory and curvature seed. They do not gain time by relaxing the model or stopping
test.

The phrase "all 5,124 certified" must nevertheless remain qualified as **certified by the existing
freeze-time mechanism**. A fresh common-model gradient pass does not reproduce 5,124/5,124 for
either method: depending on the pass, it finds 5,049--5,058 below `1e-3` for the same Adam theta,
5,048--5,053 for the two EM2 fits, and 5,052--5,055 for the two EM3 fits. This is the
already-known float32/pruned certificate instability, not an EM-specific regression.

## Cold-gradient evidence

| theta set / evaluation | below `1e-3` | cold `|Pg|max` |
|---|---:|---:|
| paired Adam | 5,052 | 0.002487 |
| same Adam theta, repeated | 5,050 | 0.001744 |
| paired Adam, separate EM2 audit | 5,058 | 0.002414 |
| same Adam theta, final-audit first / repeat | 5,056 / 5,049 | 0.003031 / 0.002688 |
| EM2A, two separate passes | 5,053 / 5,048 | 0.003294 / 0.003097 |
| EM2B | 5,051 | 0.003818 |
| EM3B | 5,052 | 0.002356 |
| EM3A | 5,055 | 0.002101 |

At the identical Adam theta, repeating the gradient changes the maximum per-family projected
gradient by 0.001292 in the EM3 audit and 0.001515 in the final EM2 audit--both larger than the
certificate threshold itself. Of the 72 first-pass failures and 74 repeated-pass failures in the
EM3 audit, only 36 are shared. Thus individual classifications around the `1e-3` cliff are not
reproducible under the current float32 atomics/pruned gradient. Separate passes over that same
Adam theta extend its observed range to nine families. EM3's
first cold count is exactly its paired baseline's, while EM2's differs by five; those differences
are smaller than the nine-family spread already observed for the identical baseline theta. The
EM2 worst residual is larger, but one unstable cold pass does not establish a systematic
certificate regression. The defensible statement is that neither audit separates EM from the
existing cold-gradient variability.

This distinction matters for reporting:

- Fair: "all 5,124 pass the unchanged production freeze-time certificate."
- Not supported: "all 5,124 pass a fresh common-model gradient check."
- A stricter reproducible certificate would be a separate product decision and would change the
  convergence settings/cost for both baseline and candidate.

## Likelihood and basin changes

The common-model forward audit is stable at fixed theta: the repeated Adam NLL is bit-identical.
EM2A is 1.634510 bits lower and EM3B is 1.763648 bits lower in aggregate than Adam. The two EM3
full fits differ by only 0.001206 bits in aggregate, with maximum per-family difference 0.001742
bits and no family above 0.01 bits. The two EM2 fits differ by 0.003619 bits in aggregate under
the common forward pass, with maximum per-family difference 0.001310 bits and no family above
0.01 bits. Therefore the aggregate EM improvements and selected material basins are repeatable at
the audit's resolution, rather than forward-evaluation noise.

The improvement is not uniform. For EM2, 25 families change by more than 0.01 bits: 13 are worse
(sum `+8.4396` bits, largest `+1.4897`) and 12 are better (sum `-10.3293` bits, largest improvement
`-2.3865`); the other 5,099 families sum to `+0.2552` bits. For EM3, 26 families are material: 14
are worse (sum `+8.6895`), 12 are better (sum `-10.3304`), and the other 5,098 sum to `-0.1227`
bits. The same 25 material families occur in both audits; EM3 additionally moves COG2352_1 by
`+0.25155` bits. Directly comparing the audited candidates, EM3 is `0.12914` bits lower overall,
but COG2352_1 is `0.25155` bits worse and all other individual differences are below 0.01 bits.
This is consistent with different certified local/flat basins. The candid quality statement is
therefore:

> EM preserves the unchanged fit certificate and improves the audited aggregate objective by
> 1.635 bits for EM2 and 1.764 bits for EM3, but it trades small numbers of per-family basin
> improvements and regressions; neither variant is a per-family dominance result.

## Speed-claim interpretation

The speed claim should use paired warmed runs and the production ledger, not the cold first process
or rounded log reconstruction. The ledger counts all resident-model clades, including frozen rows
until an actual re-plan, and separately reports EM, Newton-gradient, Hessian, and rebuild time.
Hessians are not mislabelled as three gradient passes. Because both methods use the same original
freeze/certificate mechanism, earlier EM-triggered freezing is a legitimate algorithmic saving.

| method | H100 wall A / B | mean wall | clade equivalents A / B | mean equivalents | audited NLL gain vs Adam |
|---|---:|---:|---:|---:|---:|
| warmed Adam | 512.718 s | 512.718 s | 17.0406 | 17.0406 | reference |
| EM2 | 396.305 / 403.470 s | 399.888 s | 13.4031 / 13.3977 | 13.4004 | 1.631--1.635 bits |
| EM3 | 398.290 / 396.401 s | 397.345 s | 13.4659 / 13.4601 | 13.4630 | 1.764--1.765 bits |

Both EM variants decisively beat the warmed Adam control: EM2 reduces mean wall time by 22.01%
and clade-weighted work by 21.36%; EM3 reduces them by 22.50% and 20.99%. Their difference is not
a clear speed result. EM3's mean is only 2.54 s (0.64%) faster, which is smaller than EM2's 7.17 s
paired spread, while EM3 performs 0.47% more clade-weighted work and requires a third mandatory
full-population count pass. The fastest individual run is EM2A at 396.305 s, effectively tied with
EM3B at 396.401 s.

Keep both options exposed, but use **EM2 as the default**: it is simpler, does less measured work,
and its speed is indistinguishable from EM3 at `n=2`. EM3 is a reasonable explicit option when
the modest additional aggregate NLL improvement (about 0.13 bits) is valued, with the caveat that
it additionally sends COG2352_1 to a basin 0.25155 bits worse than EM2. No claim of an EM3 speed
advantage is supported by this campaign.
