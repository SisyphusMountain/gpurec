# Current exact-gradient profile: where a dramatic single-GPU gain must come from

## Scope and method

This profiles the **current exact/exact implementation**, not the historical iterative/Neumann
implementation.  The inputs are the same 200 Coleman families and the production genewise recipe
(`float32` state, `float64` accumulation, pruning threshold `1e-6`, clade budget 200,740):

- the paid, common EM2 endpoint; and
- the final native point from the paired diagnostic fit, used only as a representative late rate
  point.  It is not fed back to an optimizer.

There was one ordinary warmup at each point followed by exactly one profiled gradient at each
point.  Scoped wrappers put CUDA events around the unmodified production forward and reverse
functions and retain their already-computed pruning masks.  The wrappers are restored in
`finally`.  The Chrome analysis counts only raw CUDA kernel/memcpy/memset events; it does not
double-count PyTorch CPU operators and their CUDA children.

Artifacts are `current_gradient_profiles.json`, `current_gradient_analysis.json`, and the two
Chrome traces beside this report.  `profile_current_gradients.py` and `analyze_profiles.py`
reproduce the measurement and analysis.  Source hashes for the exact kernels are embedded in the
profile artifact.

## Result

| 200 families, RTX 4090 | EM2 endpoint | late native point |
|---|---:|---:|
| whole gradient, CUDA event | 2173.3 ms | 2219.2 ms |
| forward (E + exact Pi) | 667.0 ms (30.7%) | 672.6 ms (30.3%) |
| reverse | 1499.3 ms (69.0%) | 1540.2 ms (69.4%) |
| raw CUDA busy time | 2058.1 ms | 2101.3 ms |
| idle in first-to-last CUDA event span | 114.8 ms (5.28%) | 117.8 ms (5.31%) |
| active adjoint rows | 709,691 / 1,491,100 (47.6%) | 724,674 / 1,491,100 (48.6%) |
| rows pruned from the reverse | 781,409 | 766,426 |
| exact-forward range fallbacks | 0 | 0 |
| peak allocated / reserved | 7.31 / 15.02 GiB | 7.30 / 16.52 GiB |

The rate point barely changes the profile.  In particular, the late point does not turn this into
a cheap sparse reverse: about half the clade rows remain active at both points.

The dominant raw kernels, averaged across the two points and expressed as a fraction of the whole
CUDA-event gradient, are:

| operation | mean ms | whole-gradient share | launches/profile |
|---|---:|---:|---:|
| exact transposed self-loop solve | 411.1 | 18.7% | 1,045 |
| gene-split event VJP | 349.5 | 15.9% | 1,030 |
| transfer-subtree VJP | 337.1 | 15.3% | 1,030 |
| exact forward self-loop solve | 303.6 | 13.8% | 1,045 |
| multiple-split forward staging | 181.3 | 8.3% | 1,022 |
| reconciliation-event VJP | 107.5 | 4.9% | 1,045 |
| log-space split-input forward update | 81.0 | 3.7% | 1,030 |

The four named reverse mathematical kernels (transpose, split-event, transfer-subtree, and local
reconciliation-event VJP) take 1.19--1.22 s, 54.6--55.2% of the entire gradient.  PyTorch
`index_add` kernels used by per-wave parameter accumulation add about 108--109 ms.  This is not a
single bad kernel and it is not mostly host bookkeeping.

## Compute, layout, and memory inventory

- The problem is dense across `S=2013` species.  There are 1,491,100 clade rows: approximately
  3.00 billion clade/species cells in one whole-population table.  `Pi` and `Pibar` are row-major
  `[C,S]` float32 arrays, each 11.18 GiB across all families, streamed here as eight batches.  One
  maximum-size batch table is 1.51 GiB.
- The schedule has 1,045 waves, median width 922, mean 1,427, maximum 8,192.  Only 2.3% of waves
  have at most 32 rows and they consume 0.3% of exact-forward time.  Repacking tiny waves cannot
  produce a multiplicative gain.
- The forward exact solve walks all rows.  Its four-slot `[max wave,S,4]` scratch is about 252 MiB
  at `W=8192`.  The transpose instead keeps one padded 2,048-lane row in registers and walks the
  33-level species tree twice.  On active rows it costs about 0.57 microseconds/row versus 0.20
  microseconds/row for the forward exact solve.  The current source documents the reason: signed
  adjoints need a more involved affine tree elimination and the register-resident kernel is
  latency/register constrained.
- The gradient retains dense primal `Pi/Pibar`, a dense reverse RHS, and (when memory permits, as
  it does here) the forward gene-split rows.  Those arrays explain most of the 7.3 GiB allocated
  peak.  The split reverse then works on expanded split sides, not merely the 1.49 M parent clade
  rows.
- Species padding is only 2,048/2,013 = 1.017x, so padding waste is not material.  The fundamental
  multiplier is the `C x S` recurrence and, for split terms, its side expansion.

These observations agree with, but do not turn into lower-bound claims from, the prior campaign:
TF32 transfer GEMM had split-side rather than clade-row work and failed the precision gate; fp32
GEMM was slower; transposed and subtree-per-lane tree walks were slower; and sparse-background
approximations changed gradients by 20--40%.  None is being proposed again.  Likewise, the old
empirical "dense floor" is not treated as a proof that another algorithm cannot win.

## Amdahl gates

At the two points, respectively:

- making all observed GPU idle free is only a 1.056x gain;
- making the entire reverse free caps speedup at 3.22--3.27x;
- with the forward unchanged, a 2x gradient needs the 1.50--1.54 s reverse reduced to about
  0.41--0.43 s: a **3.58--3.63x reverse speedup**;
- even making the four dominant reverse kernels free caps speedup at 2.20--2.23x.  Optimizing only
  those four would require an implausible 10.7--11.9x gain to reach 2x, because the untouched
  forward and reverse remainder already nearly consume the target budget.

Consequently, launch tuning, atomics cleanup, or a better bound solver can still be useful, but
cannot explain or deliver a dramatic end-to-end gain.  A 2x target requires changing how the
three rate derivatives are computed or changing the dense state representation.

## Structural route 1: a four-channel expectation/derivative semiring

There are only three fitted variables per family.  A forward recurrence that carries the scalar
likelihood plus its three `(D,L,T)` derivatives could return the rate gradient at the roots and
delete the whole 69% reverse sweep.  This is equivalent to three fused JVPs, not three independent
forwards:

1. Differentiate the extinction fixed point for the three directions together.
2. In each split recurrence, load the children once and update the value plus three partials.
3. In the exact self-loop, factor the same family/rate-dependent tree operator once and solve its
   three differentiated right-hand sides together.
4. Reduce the three root derivatives directly to each family's gradient.

The measured per-gradient gate is sharp.  Let `F` be the present 667--673 ms forward.  The complete
value-plus-three-derivative pass must cost at most **1.62--1.64 F** to halve per-gradient time.  A
naive primal plus three independent JVPs is about `4 F` and loses.  With 20% fewer gradient-work
passes, a **2.03--2.05 F** derivative pass would still halve the Newton-gradient portion.  Neither
number by itself means the whole fit is 2x: EM and other fixed work remain.

The latest full-run accounting makes the end-to-end gate tighter: about 306.8 s of a 396.3 s fit is
Newton-gradient cost and about 89.5 s is fixed with respect to this backend (including roughly 49 s
of EM).  Reaching 198.15 s with unchanged optimizer work therefore requires a **2.82x** reduction
of Newton-gradient cost, corresponding to a semiring near **1.15--1.16 F**.  If a separately
validated optimizer change removes 20% of clade-weighted Newton-gradient work, the backend still
needs about **2.26x**, or a semiring near **1.44--1.45 F**.  These are campaign-level projections,
not timings of an implementation.

Why it might amortize: all four channels share the expensive species topology, primal event
masses, gauges, child-row reads, and the exact self-loop operator.  Why it might fail: it loses the
current reverse's 51--52% row pruning, adds three output channels, and raises register pressure in
an already register-heavy exact solve.  Materializing derivatives of both `Pi` and `Pibar` plus
split rows may also approach the 24 GiB card limit; a real design likely needs a smaller clade
budget or liveness-based row recycling.

This is the most credible unmeasured single-GPU path to 2x, but the profile supplies a performance
threshold, not evidence that the threshold has been met.

It also does **not** replace the two EM count passes as stated.  Three rate derivatives do not
identify the four positive ghost-augmented counts `(S,D,L,T)`; in particular `Ntot` is
undetermined.  The conservative design keeps the current reverse count API for EM and switches
only Newton gradients.  A value plus four independently perturbed free-event weights (five
channels total) might recover all counts, but its survival-conditioning augmentation and count
identity require a separate derivation and validation before making either a correctness or speed
claim.

### Bounded feasibility test before any rewrite

Build one experiment-only multi-RHS version of the existing exact tangent self-loop.  Feed it the
same primal row/operator and three synthetic right-hand sides; compare it against three serial
calls to the current exact one-RHS tangent on representative `W≈922` and `W=8192` waves.  It need
not implement the E step, DTS derivatives, or a full gradient.  Record correctness against the
three serial outputs, spills/registers, time, and bytes.  If the self-loop cannot substantially
amortize three RHS while preserving current numerical error, the full semiring cannot meet the
1.64 F standalone gate.  If it can, the next gate is a fused three-channel DTS recurrence, whose
current primal analogue already costs about 294 ms including split staging/finalization/update.

## Structural route 2: count-specific persistent reverse dataflow

The alternative is to preserve adjoint pruning but redesign the reverse as a count-producing
pipeline rather than a sequence of materialized generic VJPs.  Within a reverse wave it would:

- consume `Pi/Pibar` and the RHS once;
- solve the transpose and immediately contract local event counts while `v` is resident;
- fuse gene-split event and transfer-complement propagation over a compact list of active split
  sides; and
- perform one segmented `[families,4]` count reduction instead of seven repeated per-wave
  `index_add` reductions.

Simple launch fusion is not enough: idle is 5.3%, and eliminating the local-event pass beside the
transpose is capped near 1.3x overall.  This route must reduce the **whole reverse** by about 3.6x,
which requires avoiding multiple dense/expanded-side traversals and their intermediate traffic,
not just joining kernel launches.  Its advantage over the semiring is retaining the measured
51--52% pruning; its disadvantage is a harder global dependency structure between split waves.
The four-channel forward feasibility test is therefore the cheaper discriminator to run first.

## Conclusion

The present ~400 s fit is not held back primarily by bounds, line-search bookkeeping, host idle,
or one poorly tuned warp schedule.  On current code, 69% of a gradient is a multi-stage reverse
over dense species rows, and the remaining forward already consumes about 31%.  Roughly 2x on one
GPU needs either a fused value-plus-three-derivative forward near 1.16x today's forward with
unchanged Newton work (about 1.45x if paired with a real 20% work reduction), or a comparably
aggressive count-specific reverse/backend redesign.  The looser 1.64x and 3.6x gates halve one
gradient, not the entire fit.  Both are representation/dataflow changes; neither is established by
this profile alone.
