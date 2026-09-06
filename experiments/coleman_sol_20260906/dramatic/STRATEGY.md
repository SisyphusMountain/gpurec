# Hierarchical geometry and the single-GPU speed target

## Answer

The hierarchical coordinates can be useful as a **preconditioner and curvature
initialization**, but do not unlock BFGS or line search: both work in native log
rates, and both coordinate implementations already carry a full 3x3 BFGS matrix.
The measured opportunity is fewer costly likelihood/gradient evaluations, not
cheaper small-matrix algebra. A diagonal matrix can still be ill-conditioned if
its diagonal entries have very different scales.

For fixed expected event counts, the binary-tree logits

    u = log2((D+T)/(1+L)), v = log2(T/D), w = log2(L)

make the complete-history surrogate a sum of three one-variable terms. Its
Hessian is exactly diagonal. The observed likelihood is not that surrogate:
with the positive ghost-family augmentation and current posterior expectations,

    H_observed = I_complete - I_missing.

The second matrix is a posterior score covariance and need not be diagonal.
Whitening the complete information gives I - M, not necessarily the identity.
Consequently, retaining full BFGS corrections is preferable to assuming three
independent observed-likelihood optimizations. The cheap seed uses already-paid
EM counts, not newly computed posterior counts at every trial point.

For an invertible fixed linear coordinate map, consistently transformed full
BFGS seeds and secants produce equivalent steps with an equivalent line search.
Here the map is nonlinear, so finite secants, path curvature, bounds, and step
control can genuinely change the trajectory. Any benefit must be demonstrated
with those ingredients implemented consistently; diagonal complete information
alone is not a guarantee.

## New measured result

The same two GPT-5.6 Sol workers completed a matched four-arm 200-family
experiment and an independent current-kernel profile. Every run uses one GPU;
no multi-GPU speedup is proposed. See [optimizer results](optimizer/RESULTS.md)
and [kernel profile](kernels/PROFILE_REPORT.md).

| Coordinates | Original seed, gradient-clade equivalents | Fresh-pair seed |
| --- | ---: | ---: |
| Native log rates | 13.5925 | 12.8101 |
| Hierarchical | 14.1598 | 12.7439 |

These work totals include the same two EM passes; Hessian work is separately
reported, with one refresh in each arm. Fresh initialization calibrates the
complete-information seed with the now-available EM1-to-EM2 gradient pair and
then performs a safeguarded dense BFGS update. It needs no additional gradient.
All four runs passed the existing 200-family freeze-time certificate and the
common forward audit found no family NLL change exceeding 0.01 bits.

The improvement is 5.8% in native coordinates and 10.0% in hierarchy versus
their own old seeds. The improved methods differ by only 0.52% in work, too
small to establish a coordinate advantage. These are subset results, not a new
full-Coleman H100 runtime. No production optimizer was edited in this campaign.

## What a line-search experiment would need

A line search controls distance along a chosen direction. It does not remove
coupling or repair a poor direction. A Wolfe line search also controls the trial
directional derivative and, on a smooth straight descent ray, supplies the
positive secant curvature used by BFGS. Armijo-only backtracking checks decrease
but does not supply that curvature guarantee; safeguarded/damped BFGS remains
necessary. See the [BFGS lecture](https://pages.cs.wisc.edu/~yudongchen/cs726_sp25/Lecture_22_BFGS_SR1.pdf)
and [line-search lecture](https://pages.cs.wisc.edu/~yudongchen/cs726_sp25/Lecture_20_line_search_modified_Newton.pdf).

A meaningful GPURec comparison would:

1. Start both native and hierarchical methods at the identical EM endpoint,
   with the corrected fresh-pair seed and unchanged refresh/stopping rules.
2. Compare current trust control against a batched, per-family feasible line
   search. Maintain the original native rate box, not a rectangular uvw box.
3. Retain accepted forward states for their reverse pass, and avoid repeating
   accepted families when another family backtracks. Otherwise likelihood-only
   screening duplicates forward work inside the existing gradient API.
4. Count forward work, reverse work, Hessian work, and certification—not just
   accepted iterations. Preserve the current near-convergence noise safeguards.

The current profile puts forward at about 31% of a gradient. With the existing
API, unconditional screening therefore adds substantial work; a split,
state-reusing forward/reverse API is needed to avoid that duplication. Full
Wolfe searches additionally need directional derivatives. Also, the current
solver uses rejected-point gradients in BFGS before rollback: eliminating them
changes curvature learning, not merely accounting.

There is little first-step rejection left after reseeding: the rejected native
and hierarchical families hold only 0.323% and 0.113% of clades respectively.
Later fresh-seed rejections were not traced, so this does not rule out a benefit
elsewhere. It does make unconditional first-step screening a poor next bet.

## What approximately 2x would require

The measured full-Coleman single-H100 baseline is about 396–403 seconds after
EM, down from the earlier approximately 520 seconds. For the 396.3-second run,
roughly 355.8 seconds are EM-count/Newton-gradient evaluations. Holding the other
40.6 seconds fixed, a 198.2-second target requires removing about 56% of that
evaluation cost, or accelerating it about 2.26x. A 6–10% initialization gain
does not approach this by itself.

The current 200-family RTX 4090 profile is about 31% forward and 69% reverse.
It is not evidence that the H100 has identical phase ratios. Nevertheless, it
identifies the structural work to benchmark on H100:

- **Fused value plus three forward sensitivities:** share topology, operator
  coefficients, child loads, and nonlinear operations, eliminating the reverse
  for Newton gradients. Three independent extra forward passes would be slower;
  strong amortization is essential. The three rate derivatives do not determine
  all four ghost counts, so this version leaves the two EM reverse passes intact.
- **Count-specific reverse dataflow:** retain adjoint pruning but avoid repeated
  dense/expanded-split traversals and intermediate arrays, rather than merely
  joining launches or tuning warps.

If only Newton gradients change, the approximately 89.5 seconds of EM and other
work remain: Newton-gradient cost must improve about 2.82x to halve the fit with
unchanged evaluation count. On the local phase ratios, a fused value-plus-three-
derivative pass would need to cost only about 1.16 times today's forward. Even
with a hypothetical 20% Newton-work reduction—not demonstrated here—the gate is
about 1.45 times forward. These are feasibility requirements, not predictions.

The next bounded structural test is an experiment-only three-RHS exact tree
solve followed, if promising, by a fused derivative split recurrence. Compare
against the existing serial tangent kernels, check numerical agreement, and
measure registers, spills, memory traffic, and representative wave timings
before committing to a complete derivative rewrite. No such prototype has yet
been benchmarked. The current evidence supports these as research directions,
not a claim that a 2x improvement has been found.
