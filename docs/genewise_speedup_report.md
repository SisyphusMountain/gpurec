# The genewise fit, 7x faster: what was done and why it is safe

Written 2026-09-03. Companion to the chronological measurement log
`docs/genewise_h100_runtime.md` (every number below is taken from a run recorded there or in
`benchmark/cc/results/*.json`).

## The task and the result

The task was to make `gpurec fit --mode genewise` fast on a large real dataset without changing the
fitted likelihood: the Coleman et al. bacterial dataset (species tree `ReferenceTree.nwk`, 1007
leaves, so 2013 species-tree nodes; 5124 gene families as ALE files; the single 400,918-clade family
`COG3676_X` excluded, leaving 5123 families and 22.9 million clades). The first goal was "as fast as
possible", the second "under 800 seconds on one GPU". All measurements are on one NVIDIA H100 NVL
(94 GB) of the CC-IN2P3 cluster; the user's local RTX 4090 took over an hour before this work.

| full fit, 5123 families, one H100 | wall time | total NLL (bits) | families certified converged |
|---|---|---|---|
| original code | 5353 s (89 min) | 9048956.57 | 5073 of 5123 |
| final code, quiet node, two runs | **777 s and 786 s** | 9048938.28 and 9048938.29 | 5121 and 5119 |

The total negative log-likelihood (NLL, the quantity the fit minimises, in bits) ended 18 bits
*lower* than before, and more families reach the convergence certificate. The same code on a node
whose CPUs were busy with other users' jobs took 890 to 1244 s, so the wall time depends on the
node; the likelihood does not.

The 500-family subset went from 1017 s to 136 s, the 40-family smoke test from 354 s to 12 s.

How to run the fastest configuration on the cluster (see `benchmark/cc/env.sh` for the paths):

```
python benchmark/cc/run_genewise.py --species ReferenceTree.nwk --families families_no_largest.txt \
  --limit 0 --out-dir RESULTS --tag NAME --forward-self-loop exact --adjoint-self-loop exact --init-rate none
```

`--forward-self-loop exact --adjoint-self-loop exact` select the new exact solvers described below
(the driver makes every choice explicit); `--init-rate none` uses the new default starting rates.
Through the library nothing needs to be passed: the exact solves are now the `SolverOptions`
defaults (`forward_self_loop = "exact"`, `adjoint_self_loop = "exact"`), so `gpurec fit`, `fit_dtl`
and `GeneReconModel` use them unless told otherwise. Before that switch, a float64 test with
optimized, non-uniform transfer receiver weights exposed one missing term in the exact tangent at
internal species nodes (Hessian asymmetry 2.8e-3); it is fixed (asymmetry 6e-14, equal to the
iterated solves), bitwise-neutral for uniform weights, and guarded by a new test that compares the
exact and iterated Hessians with random receiver weights. With the exact defaults the full test
suite shows exactly the pre-existing failures of the original code and none added.

## How a genewise fit works (the words used below)

A **family** is one gene family; a **clade** is one node of that family's clade distribution
(a family has a few thousand clades). Families are grouped into **batches** (about 315,000 clades
each; 79 batches for this dataset). Inside a batch the clades are processed in **waves**: all
clades whose children are already computed form one wave (150 to 200 waves per batch). For each
clade the model keeps one number per species-tree node, the likelihood of that clade being at that
species; that row of 2013 numbers is a **clade row**, and one number of it a **species lane**.

Within a wave, each clade row obeys a fixed-point equation: the row depends on itself through the
events "transfer out then the donor lineage is lost" and "duplicate then one copy is lost". Solving
that per-row equation is the **self-loop**. The original code iterated it (16 sweeps per wave in the
cheap tier, 64 in the accurate tier).

The fit is Newton's method run independently for every family on its three rates (duplication,
loss, transfer, stored as log2 rates). Each Newton iteration needs, for every live family, the
gradient of the NLL: a **forward** pass (the self-loops above, wave by wave) and a **backward**
pass (the **adjoint**, i.e. the same equations transposed, solved wave by wave in reverse). Every
few iterations it needs the 3x3 curvature (the **Hessian**) from three **Hessian-vector products**,
each of which needs a **tangent** pass (the forward equations with a different right-hand side).
A family counts as converged when its **projected gradient** (the gradient with components pointing
into a rate bound zeroed) is below 1e-3; converged families are re-checked at the accurate solver
tier ("**verification**"), frozen, and eventually the model is re-planned over the survivors
("**re-plan**"). At the end a **certificate** recomputes the total NLL and the per-family projected
gradients over all families at the accurate tier.

## Where the time went originally

Measured on the H100 with the code as it was (first day):

| stage | measured | consequence |
|---|---|---|
| building the model over 5123 families | 775 s | done again at every re-plan, tier change and for the certificate: 4 to 16 times per fit |
| first gradients of a run | over 300 s each | Triton was compiling kernels, not computing |
| one gradient in steady state | 58 s (79 batches) | GPU 96 % busy: 45 % of GPU time in the forward self-loop kernel, 21 % in the backward one |
| one Hessian (3 probes) | about 7 gradients | the forward and the adjoint cache were rebuilt once per probe |
| certificate | 22 min | included a full 3-probe Hessian whose only use was a positive-definiteness count |
| recipe | | verification gradient computed over the whole active set, not the candidates; first drop only at iteration 12; exact Hessian every 5 iterations |

Three independent problems explained the build time. (1) The Rust parsing extension shipped in
the repository (`gpurec/gpurec_preprocess.abi3.so`) was a **debug build**; a release build of the
same source gives byte-identical output 11x faster. (2) ALE files grow as clades x leaves (the
`#set-id` section lists every leaf of every clade), so parsing looked quadratic in the clade count;
the parser materialised every leaf list. (3) Rust handed Python **one JSON string** (5.3 GB for
5123 families): parsing it took 98 s, turning its lists into tensors 54 s, and the process held
57 GB of Python objects.

## What changed

### A. Building the model (775 s to about 20 s, then a few seconds per re-plan)

- The extension is built in release mode (both crates; the backtracking extension had the same
  problem). `setup.py` already builds release; the shipped files had been built by hand.
- `.ale` set-id lists are read in place (byte spans, no per-clade allocation), the schedule depth
  is computed in one topological pass, wave layouts are built in parallel. Output byte-identical to
  the old parser on all sample files and all batch-plan settings; parse time is now proportional
  to file size (about 390 MB/s). Files: `crates/gpurec-preprocess/src/lib.rs`.
- Families are **parsed once per fit** and kept in Rust memory (`ParsedFamilies` in
  `crates/gpurec-preprocess/src/pybridge.rs`). Every rebuild only re-plans the batches over the
  requested subset of families and receives numpy arrays (zero-copy from Rust) that become tensors
  with `torch.from_numpy`. All batch tensors are `torch.equal` to the legacy JSON path
  (`tests/test_parsed_families_equivalence.py`). Python side: `parse_families`,
  `preprocess_from_parsed` in `gpurec/core/scheduling/batching.py`; `GeneReconModel` accepts
  `parsed_families` + `family_indices`; the legacy path is unchanged and still tested.
- Measured at full scale: first build 18 to 20 s (parse 44 s once, included in the first build in
  the timings above), re-plan over ~4500 families about 7 s (Rust 63 %, Python batch statics 22 %).

### B. Triton compilation (first gradient over 300 s to about 35 s)

Triton compiles one kernel variant per distinct combination of compile-time constants and
argument specialisations. Three kernel arguments that change with every wave (`n_ws`, the number
of splits in the wave; `MAX_TILES`; `n_tiles`) were declared `tl.constexpr`, and integer/pointer
arguments were left to Triton's automatic specialisation (value == 1, divisibility by 16, 16-byte
alignment of sliced views). The shared kernel cache had accumulated 19,615 variants. The per-wave
arguments are now runtime integers (they are only used in index arithmetic, so the arithmetic is
unchanged) and the varying arguments carry `do_not_specialize`; a forward + gradient + Hessian over
100 families now compiles 35 variants. Files: `gpurec/core/kernels/*.py`.

### C. The fit recipe (`gpurec/fit/genewise_fit.py`, `gpurec/fit/dtl_fit.py`)

Each change keeps the optimum the same and removes work:

- **Verify only the candidates.** The accurate-tier gradient that re-checks converged families is
  computed on a small temporary model over those families only, not over the whole active set.
- **Freeze without rebuilding.** Verified families are frozen in place (masked out of the Newton
  step and of the statistics) and the model is re-planned only once frozen families own 25 % of its
  clades (`rebuild_frac`), because a re-plan costs about 7 s while carrying a frozen family costs its
  share of one gradient. Checks happen every 2 iterations and a drop round starts at 32 families or
  5 % of the active set (`min_drop`, `drop_frac`), instead of every 4 iterations and 30 %.
- **Cheaper curvature.** The exact 3-probe Hessian is computed at the first Newton iteration of a
  tier and every 15 iterations (`hessian_refresh`); in between each family's 3x3 matrix is updated by
  the BFGS formula from the (step, gradient change) pairs the iterations already produce; the initial
  curvature comes from BFGS updates over the Adam warm-up's own pairs. Measured: no extra Newton
  iterations.
- **Certificate reuses verification.** A family's accurate-tier projected gradient was already
  measured, at its final rates, when it was frozen; the certificate now reuses it and only runs the
  forward pass for the total NLL plus a gradient over the few never-frozen families. The Hessian in
  the certificate (a positive-definiteness count) is skipped unless `certify_curvature=True`.
- **Sensible start.** Every family starts at duplication 0.01, loss 0.1, transfer 0.01 (relative to
  speciation) instead of all rates equal to 1.0 (about 8 % of the fit; `init_log2_rates` is now a
  required keyword of `fit_genewise`, derived from `init_rate` in `fit_dtl`).
- **Single tier with the exact forward.** With the exact solver (D below) the second, more
  accurate tier recomputed an identical forward, so the recipe runs one tier
  (Newton steps 220 to 110 at full scale).
- **Tail safety.** A family that never certifies within the iteration cap ends at its best evaluated
  iterate rather than its last one (one run had lost 127 bits in a single knife-edge family). A
  "stall" rule that settles such families earlier (`stall_patience`) exists but is off in
  `fit_dtl` (patience = the iteration cap): it saved about 28 s but cost 3 to 4 certified families.
- The recipe also fixes the warm-adjoint memory gate: with the Hessian warm start on, each probe
  direction keeps its own cache the size of the gradient's, so the gate now counts 1 + 3 caches
  (`warm_adjoint_fits(..., resident_caches=...)`). The old gate admitted a 500-family run at 32 GB
  that then ran out of memory at 90 GB inside the Hessian.

Effect of the recipe alone at full scale: 5353 s to 3166 s (old kernels, old start).

### D. Kernels

**D1. The fused backward series (`_reconciliation_self_loop_transpose_series_kernel`).** The wave
adjoint was solved by a Neumann series: 16 separate kernel launches per wave (64 in the accurate
tier), each adding one term, whether or not the terms still mattered. It now runs in one launch per
wave and stops a row block once its largest remaining term is below `neumann_term_tol` (1e-7)
times the block's largest adjoint value. Bitwise identical to the old launches at tolerance 0;
mean 2.3 terms taken instead of 16 (4.1 on rows the adjoint pruner keeps). Gradient at 500
families 9.9 s to 8.3 s. This path is still available (`adjoint_self_loop = "series"`).

**D2. The forward self-loop in linear space (`_fused_linear_pi_self_loop_kernel`).** This
followed the suggestion to replace log-sum-exp arithmetic by multiplications with tracked
scale factors. Each clade row is converted once to `p[s] = 2**(Pi[s] - scale)` with the row
maximum as scale (the code already stores rows with a per-row gauge, so this is natural), every
log-space per-species constant is converted once to a multiplier, and the update

```
A[s]     = recv[s] * p[s]                      transfer mass offered by species s
T        = sum over s of A[s]                  total transfer mass of the row
X[s]     = sum of A over s and its ancestors   mass a donor at s may not receive back
pbar[s]  = mt[s] * (T - X[s])                  transfer complement
p_new[s] = leaf[s] + dts[s]                    fixed source terms (leaf observation, gene splits)
         + dl[s] * p[s] + ebar[s] * p[s]       duplicate-and-lose, stay-with-transferred-copy-lost
         + e[s] * pbar[s]                      transfer out, donor lost
         + sl1[s] * p[child1(s)] + sl2[s] * p[child2(s)]   speciation with one child lost
```

is iterated as multiply-adds with a per-lane early exit at relative change 1e-6 (`pi_linear_tol`),
re-gauging the row every iteration. This is the same seven-term sum the log kernel evaluates with
six `exp2` and one `log2` per lane per iteration. Mean 6 iterations instead of 15 launches. The
first version crashed on the full dataset: at transfer rates near the cap, `T - X[s]` cancels
catastrophically in float32 (a float64 replay showed the *log* path was the inaccurate one there,
by 129 log2 units). The mass is now built from **two running sums of non-negative terms** over the
depth-first species order (species whose subtree has not opened yet, plus species whose subtree has
already closed), with no subtraction and no 34-step ancestor walk. That made the kernel faster too
(1.52x over the log path). Known limit: one scale per row means a lane more than about 126 binary
orders below its row maximum is zero in float32; the log path (`forward_self_loop = "log"`) keeps
it and remains the reference implementation.

*This mode has since been removed.* D3's exact solve replaced it: it is faster, it has the same
one-scale-per-row range limit (and handles it automatically, by handing wide rows back to the log
sweeps), and no default selected the linear path any more. `forward_self_loop` is now `"log"` or
`"exact"`, and `pi_linear_tol` is gone. The additive prefix-scan construction of the transfer mass
described above survives, in the backward self-loop kernel.

**D3. Exact tree solves (the largest kernel gain).** In the update above the `max(T - X, 0)` of the
original code never clips, because every value is a likelihood and hence non-negative. So the
fixed point is a **linear system on the species tree**: species s is coupled to its two children
through `p[child]`, to its ancestors through one scalar (the mass its ancestors have taken), and
to everything through the scalar `T`. Such a system is solved exactly by elimination in a fixed
number of passes over the tree (`_exact_tree_pi_self_loop_kernel`):

1. Leaves to root: write each node's answer as an affine function of what comes from above,
   `p[s] = alpha[s] + gamma[s] * u[s]`, where `u[s]` is the transfer mass still available once s's
   ancestors have taken theirs. A leaf gives `alpha = src / diag`, `gamma = q / diag` with
   `q = e * mt` and `diag = 1 - dl - ebar + q * recv`; an internal node substitutes its children's
   affine forms and divides by `pivot = diag + (children's feedback) * recv`, which is bounded below
   by `diag`, so nothing is subtracted.
2. Root to leaves for the two basis vectors, giving one scalar equation for `T`.
3. Two more passes rebuild `u` from additions only (subtree mass bottom-up, off-path mass top-down),
   because `T` minus what the ancestors took cancels for deep species in float32.

Four passes of O(S) per row, no iteration, and the answer is what the old iteration converges to
as the sweep count grows. The **adjoint** is the transposed system for the same row and is solved
by the analogous elimination (`_solve_reconciliation_self_loop_transpose_row_kernel`); the **tangent** used by the
Hessian probes is the forward system with a different right-hand side and reuses the forward
elimination (`gpurec/core/kernels/wave_tangent.py`). Evidence, at 100 families and fitted rates,
against the old path run to convergence (256 sweeps or 256 terms): total NLL within 2.6e-3 bits,
gradients and per-wave adjoints inside the run-to-run noise of the reference itself (the backward
accumulates with atomics, so two identical runs differ at the 1e-3 level), per-wave adjoints within
about 8 float32 ulps, zero divisions by a non-positive pivot over millions of row solves. Timings
at 500 families: one gradient 9.6 s (log) / 8.9 s (linear) / **6.0 s** (exact forward) / **5.2 s**
(exact forward + adjoint); one 3-probe Hessian 61 s to 43 s. Modes: `forward_self_loop = "exact"`,
`adjoint_self_loop = "exact"` (the exact tangent follows the adjoint setting).

**D4. The backward "prepare" kernel built additively.** After D3 the largest remaining kernel in a
gradient was `_prepare_reconciliation_self_loop_vjp_kernel`, which recomputed each donor's valid
receiver mass as "row total minus ancestor sum" with a 34-step ancestor walk and 46 `exp2` per
element. It now uses the same two running sums as D2 (shared helper `gpurec/core/valid_receivers.py`).
Kernel 174 to 61 microseconds, gradient 13 to 17 % faster; in float64 both formulations agree to
4e-12; in float32 the gradient's run-to-run noise drops (1.1e-3 to 6.8e-4 at fitted rates) and
reordering a reduction no longer moves the gradient by O(1), which made a warp-count change on the
transfer-subtree kernel safe (+5 %).

**D5. Float64 kept exact.** The early-exit tolerances of the day (`neumann_term_tol`, and
`pi_linear_tol` before the linear forward was removed) are written in units of float32 precision
and are rescaled by the ratio of machine epsilons for the dtype in use
(`dtype_scaled_self_loop_tol`), so float64 runs exit only below one fp64 ulp; float32 behaviour is
bit-identical. Without this, the fp64 curvature-symmetry test lost eight digits.

**D6. Things measured and left alone.** Python's garbage collector (no effect), larger batches
(9 % at most; the work is GPU-bound), warm versus cold adjoint (same speed), and a full `num_warps`
sweep of the five hottest kernels on the H100 (nothing beyond 2 %; the defaults tuned on an RTX 4090
stand). The four per-wave backward kernels are latency-bound at low occupancy rather than
compute- or bandwidth-bound; every tiling change tried was either slower or rejected on
correctness before D4 removed the cancellation.

### E. A rounding floor in every transfer sum (third round)

**What was found.** Two corners of the rate box were left where the exact path disagreed with the
log path even in float64. Chasing the worse one (duplication and transfer rates at their floor,
loss at its cap: log2 rates D = -19.9, L = 1, T = -19.9) on the 29,014-clade family COG0009_0 led
not to a solver bug but to one arithmetic step every kernel shared. A species lane's available
transfer mass (the receiver-weighted mass of every species that is neither the lane nor one of its
ancestors) was formed as **the row's total minus the mass on the lane's own ancestor chain**. For
every species hanging under the lane that holds the row's mass, the true remainder is below the
rounding unit of the total (2^-24 in float32, 2^-53 in float64), so what came out was rounding
noise, sometimes turned into "no transfer possible" when the difference went negative. Those lanes
are 50 to 300 binary orders below their own row maximum and looked harmless, but under high loss a
lane far under its own row maximum is the dominant factor of a product in the parent clade's
gene-split source, so the noise climbs the gene tree: a whole-row comparison of the two paths
(`benchmark/cc/corner_row_probe.py`) showed the first disagreement at exactly 54 binary orders
below the row maximum in wave 1, growing to 0.08 log2 by wave 25, 11 by wave 32 and 100 by wave
100; the family's likelihood was off by 8.7 bits. Both paths were fixed points of the log kernel to
1e-8 (`corner_residual_probe.py`: one more sweep moved neither), which is what gave it away: the
equation being iterated was itself floor-limited, so the converged log path was not an oracle.

The same construction sat in every derivative kernel, and in the adjoint it is amplified: each
receiver lane's adjoint coefficient divides by that lane's own valid receiver mass, so for lanes
under the dominant species it is about 2^depth, and "row total minus own subtree" of those terms,
for the dominant lane, is a difference of two astronomically large, nearly equal numbers. At the
corner the float64 gradient was 1e8 times larger than central finite differences of the corrected
likelihood, for both adjoint solvers (`corner_fd_grad.py`); in a mild regime (D = -6, L = -3,
T = -6) analytic and finite differences agreed to 8e-5, the step's own truncation error.

**The fix: additions only, everywhere.** A lane's available mass is the mass hanging off its
ancestor chain plus its children's subtree masses, both sums of non-negative terms built by two
walks over the species tree (subtree sums from the leaves up, off-chain sums from the root down
through parent and sibling). In the exact forward solve this is carried as an affine function of
the mass entering each subtree (two coefficients per species), which replaced three walks by one
and removed the noisy first pass (commit bacea6b1; a host-side lane-by-lane residual of the
fixed-point equation, `exact_row_host_check.py`, went from 1.9e-8 to 1.4e-13). The log forward
sweep now uses two running sums over the depth-first species order instead of a 34-deep ancestor
walk, and got 8 to 18 % faster (89aab151). The three tangent kernels share one additive helper
(edc92533, 7f95d53e). The series adjoint term and the exact adjoint solve eliminate on the
off-subtree adjoint passed down by addition instead of the row total (60a8b993, 8ad9bf49); the two
transfer VJP kernels followed (e24fd7f4). Second-order (Hessian) kernels: see the log.

**Result.** The float64 log and exact paths now agree to 1e-11 bits on the corner family at every
depth; the analytic gradient there is -71.944 / +2748.83 / -407.69 (D, L, T) against finite
differences -71.951 / +2749.09 / -407.73, the truncation floor, identically for both adjoint
solvers. At fitted rates nothing changed: the full fit with the forward fix alone gave
9048938.300 bits (previous runs 9048938.28 to 9048938.31), 5119 of 5123 certified; the full fit
with every kernel converted (747 s, 9048938.292 bits) certifies all 5123 families for the first time,
in 68 Newton steps instead of about 110. The float32 rate-box sweep against a converged float64 oracle
now passes at all 27 corners (3 failed before), with the likelihood within 8e-3 bits everywhere. The point of the change is
that the rate box is now trustworthy where the fit may wander, for the likelihood and for the
gradient, in both precisions.

## Is the likelihood preserved?

Yes in total and for almost every family; a few dozen families with two competing optima changed
basin, mostly for the better. Scoring the fitted rates of the original code and of the final code
under one common solver (`benchmark/cc/compare_fit_thetas.py`): 5060 of 5123 families agree within
0.01 bits; 63 differ by more (29 worse, 34 better); 22 are worse by more than 0.1 bits (largest 2.4)
and 30 better by more than 0.1 bits (largest 5.3); worsenings sum to 18.3 bits, improvements to
36.1 bits, net 17.8 bits better. The families that move have flat or bimodal likelihood surfaces
(for example a duplication rate of 2^-2.8 versus 2^-16.6 with nearly equal likelihood), so their
fitted **rates** can differ by more than a factor of two while their likelihoods differ by a few
bits: 412 of 5123 families differ by more than one log2 unit in some rate. Anyone comparing
per-family rates rather than likelihoods should know this.

"Certified converged" means the accurate-tier projected gradient at the family's final rates is
below 1e-3; 5119 to 5121 families certify now versus 5073 before. The remaining 2 to 4 are
knife-edge families that no path converges within the 120-iteration cap; their rates are the best
iterate seen.

Reproducibility: the gradient is not bitwise reproducible (atomic accumulation), so per-family
convergence flags in the tail can differ by a few families between any two runs, old or new; the
total NLL varies by about 0.02 bits between runs.

Precision: production runs are float32 with float64 accumulators (unchanged). The exact path holds
one scale per row, so a row spanning more than `exact_range_log2` binary orders is handed to the log
sweeps (see D2); float64 runs are supported by the same kernels and were used as the oracle whenever
two float32 answers disagreed. Since the third round (section E) the float64 log and exact paths
agree at every depth of every row, so either is an oracle.

## The progression of full-dataset runs

| configuration | wall | NLL bits | certified |
|---|---|---|---|
| original | 5353 s | 9048956.57 | 5073 |
| + recipe (C), old kernels | 3166 s | 9048964.87 | 5052 |
| + new start, log forward, fused series | 1948 s | 9048939.01 | 5120 |
| + robust linear forward (D2) | 1570 s | 9048959.32 | 5119 |
| + exact forward (D3), single tier | 1053 to 1101 s | 9048938.4 | 5119 to 5120 |
| + exact adjoint (D3) | 782 to 794 s | 9048938.3 | 5120 |
| + exact tangent, additive prepare kernel (D4), fp64 fix (final) | 777 to 786 s | 9048938.28 | 5119 to 5121 |
| + every transfer sum additive (E), all first- and second-order kernels | 747 s (shared node) | 9048938.29 | 5123 |

The 747 s run (job final_fulli, interactive node shared with other users: its warm-up took 148 s
against 125 s on a quiet node) needed 68 Newton steps instead of about 110, because the two to four
families that used to iterate to the cap now converge: warm-up 148 s, Newton gradients 449 s, curvature
53 s, verification 47 s, re-plans 13 s, certificate 18 s; peak memory 26.7 GiB (2 GiB more than
before: the additive backward kernels park each node's own term in a scratch buffer).

Time split of the 777 s run: warm-up (5 Adam gradients) 125 s, Newton gradients 397 s, curvature
38 s, verification 44 s, re-plans 13 s, certificate 17 s, first build 20 s, and a tail of a few
families iterating to the cap for the rest. One full-dataset gradient now costs about 30 s
(79 batches x 0.4 s) at fitted rates, spread over the per-wave backward kernels, the two exact
solves (about 15 % each) and the split reductions; all are latency-bound at low occupancy.

## What is left

- Kernel occupancy work on the per-wave backward kernels (each launches one block per clade row or
  per split with 40 to 187 registers per thread; the profile in `docs/genewise_h100_runtime.md`
  lists them with their stall reasons).
- The warm-up's five full gradients (125 s) and the tail of a few slow families (30 to 120 s,
  run-dependent).
- Independent families shard perfectly across GPUs: `benchmark/cc/run_genewise_sharded.py` splits
  the family list by squared clade count and merges the results (2 GPUs: 48 min in the first round).
- The exact tangent in float32 is a few times the reference noise at a flat starting point (signed
  accumulation through 33 tree levels) and indistinguishable from 16 sweeps at fitted rates; carrying
  the two affine coefficients in float64 inside the walk would remove even that.
- Reproducibility of the Hessian-vector product: a bit-identity test had become flaky after the
  fused-series merge. The cause was not a race but uninitialised memory: the wave backward writes
  only the clade rows the adjoint pruner keeps, and its returned buffers were `torch.empty`, so
  pruned rows carried allocator leftovers that the HVP (unlike the gradient) does not mask. Pruned
  rows are now zeroed (commit 0d319ad9). What remains is 1 to 3 float32-ulp scatter in the
  Hessian-vector product for every forward path alike, traced to the relaxed float atomics that
  accumulate parameter cotangents in the second-order kernels (the forward's own outputs are
  bitwise stable across repeats, and a poisoned-allocator run shows no uninitialised read is left).
  A bit-identity assertion on the HVP is therefore inherently flaky at the ulp level; making it hold
  would need a fixed-order reduction in those kernels. Measurement tools:
  `benchmark/cc/test_hvp_reproducibility.py`, `test_hvp_bit_identity_flake.py`.

## Reproducing

Cluster (CC-IN2P3, `ssh cc`; when the Kerberos servers are unreachable, `ssh -o
HostKeyAlias=cca.in2p3.fr emarsot@cca019.in2p3.fr` works with the cached ticket): repo copy
`/sps/biometr/emarsot/gpurec`, Python environment `/sps/biometr/emarsot/envs/gpurec-h100`
(torch 2.13.0+cu130, triton 3.7.1), data `/sps/biometr/emarsot/gpurec-data/coleman`, results
`/sps/biometr/emarsot/gpurec-runs`. `source benchmark/cc/env.sh; bash benchmark/cc/sbatch_h100.sh
NAME HH:MM:SS 'cmd'` submits a one-GPU job (`srun` does not work from the login node; GPU nodes are
shared, pin a quiet one with `--nodelist` for timings). Drivers: `run_genewise.py` (timed fit with
JSON + fitted rates), `run_genewise_sharded.py` (N GPUs), `compare_fit_thetas.py` (per-family
likelihood comparison of two fits), `profile_phases.py`, `nsys_theta.py` / `run_nsys_theta.sh`
(Nsight Systems), `run_ncu_bwd.sh` (Nsight Compute), `test_exact_forward.py`,
`test_exact_adjoint.py`, `test_exact_tangent.py`, `test_neumann_exit.py`
(kernel equivalence tests against converged references).

Locally: build both Rust crates in release mode before anything else (`cargo build --release
--manifest-path crates/gpurec-preprocess/Cargo.toml`, same for `crates/gpurec-backtrack`, then copy
`target/release/lib<name>.so` to `gpurec/<name>.abi3.so`, or `pip install -e .` which does this);
the `.so` files are gitignored and the ones previously in the tree were debug builds.

## Knobs added

| where | knob | meaning |
|---|---|---|
| `SolverOptions` | `forward_self_loop` = "log" / "exact" (default) | which forward self-loop kernel runs (a third value, "linear", existed for a while and was removed) |
| `SolverOptions` | `adjoint_self_loop` = "series" / "exact" (default) | which adjoint solve runs (the exact tangent follows "exact") |
| `SolverOptions` | `neumann_term_tol` (1e-7) | early-exit tolerance of the fused adjoint series, in float32 units |
| `fit_genewise` (required keywords, set by `fit_dtl`) | `min_drop` 32, `rebuild_frac` 0.25, `hessian_refresh` 15, `certify_curvature` False, `init_log2_rates` (log2 0.01, log2 0.1, log2 0.01), `stall_patience` 120 | recipe controls described in section C |
| `GeneReconModel` | `parsed_families`, `family_indices` | build from an already parsed dataset |
| `run_genewise.py` | `--forward-self-loop`, `--adjoint-self-loop`, `--init-rate`, `--debug-dump-dir` | driver flags |
