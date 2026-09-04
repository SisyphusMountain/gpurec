# Genewise fit runtime on the Coleman 1007-leaf dataset (H100, 2026-09-02)

Goal: make `gpurec fit --mode genewise` (the `fit_dtl` genewise recipe, certificate on) faster on
the Coleman et al. bacterial dataset without changing the fitted likelihood.

Dataset: species tree `ReferenceTree.nwk` (1007 leaves, 2013 species nodes) and 5124 ALE
gene-family files (23.3M clades in total). The single 400,918-clade family `COG3676_X` is
excluded, leaving 5123 families and 22.9M clades. With the default `clade_budget=315_000` the
model is 79 batches of roughly 150-200 waves each.

Hardware: CC-IN2P3 cluster, one NVIDIA H100 NVL (94 GiB), 12 CPU cores per job. Cluster
scripts live in `benchmark/cc/` (see `env.sh`, `sbatch_h100.sh`, `run_genewise.py`,
`run_genewise_sharded.py`).

## Where the time went (before)

Measured with the code at commit `817007e6` (the state of `main` plus the job scripts):

| stage | measured | note |
|---|---|---|
| model build, 5123 families | 775 s | repeated at every rebatch, tier change and for the certificate (4-15 times per fit) |
| first gradients at full scale | > 300 s each | Triton JIT compilation, not GPU work |
| one gradient, steady state, 500 families / 13 batches | 10.8 s | GPU 96 % busy; 45 % of kernel time in `_update_reconciliation_likelihood_kernel`, 21 % in `_apply_reconciliation_self_loop_transpose_kernel` |
| one Hessian (3 HVP probes), 500 families | 173 s | forward solve + adjoint cache rebuilt once per probe |
| 40-family end-to-end fit | 354 s | 4 model builds |

Three root causes were found:

1. **The shipped Rust extension was a debug build.** `gpurec/gpurec_preprocess.abi3.so`
   contained the overflow-check strings only a debug build emits, and a release build of the
   same source produced byte-identical output 11x faster. On top of that, ALE files grow as
   clades x leaves (the `#set-id` section lists every leaf of every clade), so parsing looked
   quadratic in the clade count; the parser now reads those lists in place and computes the
   schedule depth in one topological pass (output byte-identical, another 1.5x on large
   families). Per-family parse time is now proportional to file size (~390 MB/s).
2. **Rust to Python interchange was one JSON string** (5.3 GB for 5123 families): `json.loads`
   took 98 s, the list-to-tensor conversion 54 s, and the process held 57 GB of Python objects.
   The families are now parsed once per fit (`ParsedFamilies` in
   `crates/gpurec-preprocess/src/pybridge.rs`); every rebuild re-plans the active subset and
   receives numpy arrays (zero-copy from Rust) that become tensors with `torch.from_numpy`.
   All batch tensors are `torch.equal` to the legacy path (`tests/test_parsed_families_equivalence.py`).
3. **Three Triton kernel arguments that change with every wave were `tl.constexpr`**
   (`n_ws` in the transfer-subtree VJP kernel and its second-order twin, `MAX_TILES` in the
   multi-split DTS reduction, `n_tiles` in the max-transfer VJP reduce). Triton therefore
   compiled one kernel variant per distinct wave shape: the shared cache held 19,615 variants,
   and a first full-dataset gradient spent more than 10 minutes compiling while a steady-state
   gradient takes 58 s. They are now runtime integers (only used for index arithmetic, so the
   floating-point work is unchanged).

Two smaller items:

- `_analytic_hessian` in `gpurec/fit/genewise_fit.py` now streams the batches in the outer
  loop and applies the 3 probes to one forward solve + adjoint cache per batch (the library's
  multi-batch HVP rebuilt both per probe). 2.6x on the Hessian at 500 families (173 s to 66 s);
  the difference to the old result (0.10) is below the run-to-run noise of the atomics (0.15-0.18).
- The warm-adjoint memory gate only counted the gradient cache; the HVP warm start keeps one
  more `[total_clades, S]` cache per probe (3x). On the H100 a 500-family run passed the gate at
  32 GiB and ran out of memory at 90 GiB inside the Hessian. `warm_adjoint_fits` now takes
  `resident_caches` and the model passes 1 + 3.

Things checked and ruled out: Python's cyclic garbage collector (disabling it changes nothing),
a larger `clade_budget` (13 to 3 batches gives 9 % on the gradient; the work is GPU-bound, not
launch-bound), warm vs cold adjoint (same speed), and Triton `num_warps` for the five hottest
kernels (`benchmark/cc/sweep_num_warps.py`: every alternative is within 2 % of the current
default or slower; 2 warps on the tangent self-loop kernel is 6.7x slower).

## After

| stage | before | after |
|---|---|---|
| model build, 5123 families (first) | 775 s | 18 s |
| rebuild over a subset | 775 s | a few seconds (re-plan only) |
| first full-dataset gradient (includes JIT) | > 300 s | 64 s |
| steady-state full-dataset gradient | 58 s (hidden behind JIT) | 58 s |
| 40-family end-to-end fit | 354 s | 68 s, same NLL (115604.90 bits), same 25 steps / 4 builds |
| 500-family end-to-end fit | 2290 s (old code, warm cache forced off so it fits the H100), NLL 1618463.746 bits, 230 steps, 12 builds, 485/500 certified | 1017 s, NLL 1618463.769 bits, 231 steps, 11 builds, 490/500 certified |
| 5123-family end-to-end fit, 1 GPU | crashed after 2 h 34 min (CUDA out of memory inside the old streaming Hessian after the first rebatch; it had reached iteration 20 with 2981 active families at 143 min) | **5353 s = 89 min**, NLL 9048956.572 bits, 226 steps, 16 builds, 5073/5123 certified converged, peak 70 GiB (reached iteration 20 with 2983 active at 52 min; first tier done at 62 min; pi=64 tier done at 67 min; certificate 22 min) |
| 5123-family fit, 2 GPUs sharded (`run_genewise_sharded.py --n-shards 2`) | - | **2855 s = 48 min** (1.87x the single-GPU run), total NLL 9048957.290 bits (0.7 bits from the single-GPU value, tail noise: 43 vs 50 unconverged), 2561+2562 families, peak 69 GiB per GPU |
| 5123-family fit, 4 GPUs sharded | - | not run: no node with 4 free H100s was available during the session (`GPUS=4 ... run_genewise_sharded.py --n-shards 4` is ready) |

Fitted per-family log2 rates (old vs new code, 500 families): median difference 1.5e-6, 99.6 % of
families within 1e-3, worst 0.063 on one family whose duplication rate is tiny (a flat direction of
its likelihood); on 40 families the largest difference is 4.6e-4. The certified total NLL agrees between old and new code to 0.02 bits on 500 families (1.6M bits
total, 1.4e-8 relative) and to 0.001 bits on 40 families; the gradient is not bitwise
reproducible (atomic accumulation), so the per-family convergence flags in the tail differ by a
few families between any two runs, old or new. The full-dataset fixed run followed the same
rebatch trajectory as the crashed baseline (19 %, 43 % converged at iterations 8 and 12; 2983 vs
2981 families active after the first drop).

## Second round: towards 800 s on one GPU

Goal set after the first round: full 5123-family fit under 800 s on one H100 (from 5353 s).

**Recipe changes (commit 5c62ba4a, `gpurec/fit/genewise_fit.py`).** Converged candidates are
re-verified at the accurate tier on a temporary model over the candidates only (before: over the
whole active set); the convergence check runs every 2 iterations and a drop happens as soon as 32
families or 5 % of the active set pass (before: every 4, and only above 30 %); the exact 3-probe
Hessian is computed at the first Newton iteration of a tier and every 15 iterations, with per-family
BFGS updates in between (before: exact every 5); the certificate skips the Hessian (its only use was
the positive-definite count) unless `certify_curvature=True`. Measured: 40 families 68 s to 37.5 s,
500 families 1017 s to 552 s (211 vs 231 Newton steps: the BFGS curvature did not cost iterations).
On 500 families, 498 families reach the same optimum (total within 0.013 bits); two bimodal families
(`COG0210_2`, `COG0099_1`) land on the other, better branch (-1.9 bits in total), both stationary
under the 1e-3 tolerance in both runs.

**Kernel changes (commits c16b6201, 4ecf9540).** The wave adjoint's Neumann series now runs in one launch
per wave with a per-row-block early exit (relative test `max|term| <= 1e-7 * max|v|`; bitwise identical
at tolerance 0; mean 2.3 of 16 terms taken, 4.1 on rows the pruner keeps): gradient 9.9 s to 8.3 s at
500 families, Hessian 66 s to 60 s. The forward self-loop runs in one launch per wave in scaled linear
space (row converted once to `p = 2**(Pi - scale)`, seven-term log-sum-exp replaced by multiply-adds,
per-lane relative early exit at 1e-6; mean 6.1 iterations instead of 15 launches): gradient 9.85 s to
7.34 s at 500 families; the log-vs-linear difference is float32 rounding of the row frame (an fp64
control agrees to 1e-11). Both keep the old path selectable (`SolverOptions.forward_self_loop = "log"`,
`neumann_term_tol = 0`). 40-family fit unchanged in NLL, steps and builds for each.

**Recipe follow-up (commit fe642caf).** Verified families are frozen in place (masked out of the Newton
step) and the survivor model is re-planned only when frozen families own 25 % of its clades. 500
families: 552 s to 493 s, peak memory 57 GiB to 28 GiB; a full-scale re-plan costs ~7 s (Rust 63 %,
Python batch statics 22 %), verification of candidates is the larger per-round cost.

| run (500 families) | wall | NLL bits |
|---|---|---|
| original code | 2290 s (warm cache forced off) / 1017 s (first round) | 1618463.75 / .77 |
| recipe only | 493 s | 1618461.83 |
| recipe + both kernel tracks | 450 s | 1618461.88 |

**Full dataset, recipe only (first version 5c62ba4a, old kernels): 3166 s** (from 5353 s), NLL
9048964.87 bits vs 9048956.57 for the first-round code: 8.3 bits higher, with 71 unconverged families
(50 before) and a worst projected gradient of 12.3 (0.51 before) — the prompt-drop/BFGS recipe left a
few stiff families less converged in the second tier. This is being addressed (exact curvature for the
small second tier is cheap).

**Full dataset, everything merged (first attempt):** the run crashed at 1490 s (iteration 14-16 of the
first tier, after a re-plan to 2339 live families) with an infinite residual in the E-adjoint Neumann
series, i.e. a non-finite value produced upstream at a mid-fit theta; the same code passes the 40- and
500-family fits and an isolated full-scale gradient at the final fitted theta. Under investigation
(suspect: overflow in the linear-space forward when a source term is far above the row scale).

**Gradient cost depends on the rates.** At full scale the merged kernels give 33.7 s per gradient at
a flat theta (all rates 2^-6) but 54 s at the fitted theta (95 % of families have a rate above 0.25):
the early-exit loops and E fixed points take more iterations at high rates. Being profiled.

**Recipe round 3 (commit c77ebb4e).** Convergence is certified at freeze time: the accurate-tier
projected gradient computed when a family is verified and frozen is reused by the certificate, which
now only computes the total NLL (forward pass) plus the accurate-tier gradient of the few never-frozen
families. The initial 3x3 curvature comes from BFGS updates over the Adam warm-up's own gradient pairs
instead of an exact 3-probe Hessian (A/B at 500 families: 392 s vs 430 s, same NLL); exact refreshes
every 15 iterations remain. 500 families: **305 s** (Adam 48 s, curvature 25 s, Newton gradients 178 s,
verification 19 s, re-plans 3 s, certificate 7 s), NLL 1618461.86 bits, 499/500 converged.

**Triton re-specialization (commit 52bc2bec).** `do_not_specialize` on the per-launch-varying integer
and pointer arguments of the wave kernels: a forward+gradient+Hessian over 100 families now compiles
35 kernel variants instead of hundreds; gradient and Hessian timings unchanged.

**Why gradients cost more at fitted rates.** Profile at 500 families: 8.9 s at the fitted theta vs 5.7 s
at a flat theta of -6. At fitted rates 88 % of clade rows use the full 14 forward self-loop iterations
(mean 13.9 vs 6.1 flat), the backward series takes 6.7 terms (vs 2.2), and the E fixed point hits its
128-iteration cap for 12 of 13 batches (vs 6 iterations flat; the E loops themselves cost < 0.1 s).
So the extra time is genuine iteration work at high rates, not synchronisation overhead.

**Starting rates (commit ae204fc1).** `fit_dtl` now starts every family at D=0.01, L=0.1, T=0.01
(relative to speciation) instead of all rates equal to 1.0; `fit_genewise` takes the start as a
required keyword. On 500 families with the log-space forward this gave 365 s, NLL 1618463.31 bits
(the two bimodal families land on their original branch), 499/500 converged, Adam 46 s; the controls
on the same path and node gave 397 s (all-zeros start) and 406 s (explicit 1.0 start), so the new start
saves about 8 %.

**Full dataset, all merged tracks except the linear forward (log-space forward, job full_v7log,
interactive H100 node): 2305 s** (from 5353 s), NLL 9048959.87 bits (+3.3 bits vs the first-round
value; 5120 of 5123 certified converged vs 5073), 220 Newton steps. Split: warm-up 379 s (5 full
gradients at ~76 s each on that node), Newton gradients 1059 s, candidate verification 209 s (55
rounds at the pi=64 tier), curvature 152 s (15 exact refreshes), certificate 131 s, first build 84 s,
re-plans 17 s. To reach 800 s the per-gradient cost at fitted rates must fall about 3x more: the
forward self-loop (still ~40 % of a gradient, 14 iterations per row at fitted rates) and the
pi=64 verification/certificate passes (64 iterations per row) are the levers, i.e. a converging
per-row solve (tree-ordered sweeps or a direct solve) instead of truncated Jacobi iteration.

**Linear forward made robust (commit d09445bd).** The crash was float32 cancellation in the transfer
complement (`total mass - ancestor mass`, nearly equal at transfer rates near the cap): it is now built
from two additive prefix scans over species orders (no subtraction), which also removes the 34-deep
ancestor walk (linear path now 1.52x the log path); a read-write race got a barrier; the row is
re-gauged every iteration. A float64 replay of the failing batch shows the linear path was the
accurate one (6.6e-3 log2 from the fp64 oracle vs 129 log2 for the fp32 log path). Full fit with it:
**1570 s**, NLL 9048959.32 bits, 5119/5123 converged.

**Full fit, log forward with the new start (job full_v8log, main partition): 1948 s**, NLL 9048939.01 bits
(17.6 bits below the first-round value), 5120/5123 converged; split: warm-up 265 s, Newton gradients
1008 s, verification 213 s, curvature 153 s, certificate 126 s.

**Exact tree-elimination forward (commit 9bf4143f, `forward_self_loop = "exact"`).** Each clade row's
fixed point is a linear system on the species tree (the transfer-complement `max` never clips for
non-negative likelihoods); it is solved exactly per row in four O(S) walks (leaves-to-root affine
elimination `p = alpha + gamma * u`, one scalar equation for the row's total transfer mass, root-to-leaves
back-substitution, with the remaining mass rebuilt from additions only). No iteration, no `pi_iters`
dependence, zero pivot guard trips over 1.2M row solves. Against the converged log reference
(pi_iters=256, 100 families, fitted rates): total NLL within 2.6e-3 bits (the 16-sweep linear path:
1.8e-2), gradient inside the atomics noise. Timing, 500 families, one gradient: log 9.58 s, linear
8.94 s, exact **5.96 s** at fitted rates (log 8.35 / linear 5.72 / exact 4.71 at a flat theta).
500-family fit: **231 s** (linear 305 s, log 365 s), 499/500 converged. 40 families: 29 s.

*Later: the linear forward was removed.* The exact solve beat it on every timing above, shares its
one-scale-per-row range limit (and falls back to the log sweeps by itself for rows that do not fit),
and no default selected it. `SolverOptions.forward_self_loop` now takes `"log"` or `"exact"` only,
and `pi_linear_tol` no longer exists. Entries above that mention `"linear"` are the record of the
period when it did.

**Full dataset with the exact forward (job full_v9exact, main partition): 1101 s**, NLL 9048938.42 bits
(the lowest so far, 18 bits below the first-round value), 5119/5123 converged (4 unconverged, worst
projected gradient 3.6), 220 Newton steps. Split: warm-up 162 s, Newton gradients 570 s, curvature
118 s (15 exact refreshes), verification 66 s (58 rounds), certificate 19 s, re-plans 19 s, first build
19 s; the two tiers' tails (1-4 live families iterating to the 120-iteration cap) take most of the
rest. Full-scale gradient: 36 s (was 58 s at the start of the second round, 54 s at fitted rates).
Per-gradient profile at fitted rates (500 families, 6.05 s): adjoint Neumann series 17 %, exact forward
16 %, prepare-VJP kernel 16 %, transfer-subtree VJP 12 %, gene-split VJP 7 %, DTS forward 7 %,
index_add 6 %, log-space prologue 6 %, receiver VJP 5 %.

**Single tier in exact mode (commit 5946dc89).** With the exact forward, the second (pi=64) tier
recomputed an identical forward for the deferred families; the recipe now runs one tier when
`forward_self_loop == "exact"`. 500 families: **184 s** (231 s two-tier; 1017 s originally), 111 Newton
steps instead of 228, NLL 1618463.78 bits (the original code's value to 0.01 bits), 499/500 converged.

**Exact adjoint solve (in progress, same agent).** The transposed tree system is solved exactly per row
in place of the Neumann series: per-wave adjoints within ~8 float32 ulps of the series run to 256
terms; one gradient at 500 families 5.99 s to 5.15 s at fitted rates; the pi=64/neumann=64
verification pass costs the same as a normal gradient (series at 256 terms: 12.5 s); 40-family fit
18 s with both exact solves.

**Unconverged tail safety (commit 69a93ced).** In one 500-family run with the exact adjoint, a single
knife-edge family that never certifies (projected gradient 2 vs 256 at the end in the two runs) ended
127 bits worse because a 0.4 %-level change in its gradient sent its trajectory elsewhere. The recipe
now tracks each family's best NLL over every evaluated iterate and returns that theta for the families
that run out of iterations; certified families are unaffected.

**Full dataset, exact forward, single tier (job full_v10exact): 1053 s**, NLL 9048938.38 bits, 5120/5123
converged, 110 Newton steps (220 with two tiers), peak 26 GiB; split: warm-up 223 s (node under
contention: 162 s in the previous run), Newton gradients 559 s, curvature 71 s (7 refreshes),
verification 65 s, certificate 19 s, re-plans 15 s, first build 69 s (19 s uncontended).

**Full dataset, exact forward + exact adjoint, single tier (job full_v11exact2): 793.6 s — under the
800 s target.** NLL 9048938.39 bits (18 bits below the first-round value 9048956.57), 5120/5123
certified converged (3 unconverged), 109 Newton steps, peak 24.4 GiB. Split: warm-up 145 s, Newton
gradients 463 s, curvature 58 s (7 exact refreshes), verification 49 s (32 rounds), re-plans 17 s,
certificate 18 s, first build ~20 s. Fitted rates vs the original code's optimum: median difference
1e-5 log2 units, but 43 % of families differ by more than 1e-3 and 412 families by more than one
log2 unit — the per-family likelihood surfaces are flat or multimodal in those directions. Scoring both fitted theta sets under one
common exact solver: 5060 of 5123 families agree within 0.01 bits; 63 differ by more (29 worse, 34
better); 22 are worse by more than 0.1 bits (largest 2.4 bits) and 30 better by more than 0.1 bits
(largest 5.3 bits); worsenings sum to 18.3 bits, improvements to 36.1 bits, net 17.8 bits better. The
families that move are ones with two competing basins (for example a duplication rate of 2^-2.8
versus 2^-16.6 with similar likelihoods).

**Repeat of the full fit (job full_v12exact2): 781.6 s**, NLL 9048938.28 bits, 5120/5123 converged,
109 Newton steps; split: warm-up 145 s, Newton gradients 462 s, curvature 55 s, verification 48 s,
re-plans 14 s, certificate 18 s. The two runs (793.6 s, 781.6 s) bracket the run-to-run variation on
a shared node.

## Summary of the second round

| full 5123-family fit, one H100 | wall | NLL bits | certified converged |
|---|---|---|---|
| original code (first-round reference) | 5353 s | 9048956.57 | 5073 |
| + recipe (candidate verification, prompt freezing, BFGS curvature, freeze-time certificate) | 3166 s / 1948 s (new start) | 9048964.87 / 9048939.01 | 5052 / 5120 |
| + fused early-exit backward series, robust linear forward | 1570 s | 9048959.32 | 5119 |
| + exact tree-elimination forward, single tier | 1053-1101 s | 9048938.4 | 5119-5120 |
| + exact adjoint solve | 782-794 s | 9048938.3-9048938.4 | 5120 |
| + exact tangent solve, additive backward prepare kernel, fp64 fix (final) | **777-786 s** (quiet nodes; 890-1244 s on nodes whose CPUs were shared with other jobs) | 9048938.28-9048938.29 | 5119-5121 |

**Exact tangent solve for the Hessian probes (commit c4bd87ba, gated on `adjoint_self_loop = "exact"`).**
The tangent system of each row is the forward's linear system with a different right-hand side and is
solved by the same elimination. In float64 it agrees with 16 sweeps to five significant figures and with
the converged reference to the reference's own noise; in float32 at a flat theta it is a few times the
reference noise (signed accumulation through 33 tree levels, a conditioning difference of elimination vs
iteration; fp64 accumulation of the two affine coefficients would close it if ever needed), and at fitted
rates it is indistinguishable from 16 sweeps. One 3-probe Hessian at 500 families: 61.0 s to 42.7 s
(the converged 256-sweep reference costs 376.6 s). End-to-end at 500 families: **139 s** (from 196 s with
the exact adjoint alone), 113 Newton steps instead of 229, 7 Hessian refreshes instead of 15, NLL
1618462.61 bits; the exact curvature also steers the one knife-edge family back into the reference band.
40 families: 14.6 s.

**Per-wave backward kernels (Nsight Compute, commit 66bb18f0: measurements only).** All four VJP kernels
are latency-bound, none compute- or bandwidth-bound: the prepare kernel re-reads its 8 KB Pi row 34
times (ancestor walk) at 12.5 % occupancy (187 registers), the transfer-subtree kernel is bound by the 33
barriers of its level walk, the receiver kernel launches too few warps. Every tiling / warp-count change
tried was rejected because `total_receiver_mass - ancestor_sum` in the prepare kernel cancels
catastrophically, so any reordering of the 2013-term sum moves the gradient by O(1). The remedy is the
forward's additive construction of that mass (in progress).

**Full fit with all three exact solves (job full_v13exact3): 891.7 s on a node shared with two other
jobs of mine** (first build 69 s and warm-up 206 s vs 20 s and 145 s in the quiet runs; Newton gradients
463 s unchanged; curvature 55 s to 46 s; verification 54 s; certificate 18 s), NLL 9048938.27 bits,
5120/5123 converged, 108 Newton steps. Node contention moves the total by about ±100 s, larger than the
individual improvements still being made, so final timings should be taken with nothing else running on
the node.

**float64 regression fixed (commit 8a82e3c7).** The fp64 curvature-symmetry test failed since the fused
Neumann-series merge because both self-loop early-exit tolerances (`neumann_term_tol` 1e-7,
`pi_linear_tol` 1e-6) are float32-scale numbers and were applied unchanged in fp64, stopping iterations
about nine orders of magnitude before fp64 resolution. They are now documented as written in units of
float32 precision and scaled by the ratio of machine epsilons (`dtype_scaled_self_loop_tol`): fp32 is
bit-identical (factor exactly 1), fp64 exits just under one ulp. CPU suite back to 252 passed with the six
pre-existing failures (`bicgstab_*` / `e_adjoint_solver` options, benchmark `--help`).

**Timing sample before the additive backward kernel (job full_v14exact3): 781.2 s**, NLL 9048938.28,
5119/5123 converged.

**Final configuration (job full_v15final, all merged tracks, fp64 fix): 776.8 s**, NLL 9048938.28 bits,
5121/5123 converged (2 unconverged), 109 Newton steps, peak 24 GiB. Split: warm-up 125 s, Newton
gradients 397 s, curvature 38 s, verification 44 s, re-plans 13 s, certificate 17 s, build 20 s; the
remaining ~120 s is the tail, where the last 2-17 live families iterate to the 120-iteration cap at
1-2 s per iteration (in the 781 s run the tail was 33 s; it depends on which families stall).

**Stall rule (commits 4476dd8c, refined next).** A live family whose best NLL has not improved for 24
Newton steps is settled at its best iterate as unconverged. Full fit: **748.8 s**, NLL 9048938.31 bits,
but 5117/5123 certified (5121 before): near a float32-flat optimum the NLL stops registering
improvement while |Pg| still shrinks, so the rule now also requires |Pg| not to have improved by 10 %
over the same window (re-measured below). The 500-family control of this run landed on a node whose
CPUs were shared with other jobs and took 270 s for 35 steps (7.7 s per step against 1.2 s on a quiet
node): timings from shared nodes are not comparable.

**Refined stall rule (commit b6dd38bb) measured on a heavily shared node: 1244 s wall (not comparable;
re-plans took 41 s instead of 13 s), NLL 9048938.30 bits, 5118/5123 certified.** Both stall variants
cost 3-4 certified families relative to no stall rule (5121) for a ~28 s gain, so `fit_dtl` now passes
`stall_patience = 120` (= max_iter, i.e. off); the mechanism stays available as a knob.

**Final timing run (job full_v18final, node with two other users' GPU jobs): 786.2 s**, NLL 9048938.29
bits, 5119/5123 certified, 109 Newton steps; split: warm-up 125 s, Newton gradients 400 s, curvature
31 s, verification 53 s, re-plans 19 s, certificate 24 s, build 68 s (20 s on a quiet node).

## Third round: a rounding floor in every transfer sum (2026-09-04)

**What was wrong.** The rate-box sweep left two corners where the exact path disagreed with the log
path even in float64. Chasing the worse one (log2 rates D = -19.9, L = 1, T = -19.9: duplication and
transfer at the floor, loss at the cap) on the 29,014-clade family COG0009_0 led to a defect that is
not in either solver's algebra but in one arithmetic step shared by every kernel: **a lane's
available transfer mass was formed as "row total minus the mass on that lane's ancestor chain"**
(forward log kernel: `total_receiver_mass - excluded_ancestor_mass`; exact forward: `u[c] = u[s] -
recv[s] p[s]`; tangent kernels: the same two; adjoint: "row total minus own subtree" of the receiver
adjoints). For every species hanging under the lane that holds the row's mass the true remainder is
below the unit roundoff of the total (2^-24 in float32, 2^-53 in float64), so the difference is
rounding noise. Those lanes sit 50-300 binary orders below their own row maximum and looked
harmless, but under high loss a lane far under its own row maximum is the dominant factor of a
product in the parent clade's gene-split source, so the noise climbs wave by wave: a whole-row
comparison of the two paths (`benchmark/cc/corner_row_probe.py`) showed the first disagreement at
exactly 54 binary orders below the row maximum in wave 1 (2.5e-9 log2), growing to 0.08 by wave 25,
11 by wave 32 and 100 by wave 100; the family's NLL was off by 8.7 bits. Both paths were fixed points
of the log kernel to 1e-8 (`corner_residual_probe.py`: one more sweep moved neither), which is what
gave it away: the equation being iterated was itself floor-limited, so the converged log path was
not an oracle either.

**In the adjoint the same floor is amplified.** Each receiver lane's adjoint coefficient divides by
its own valid receiver mass, so for lanes under the dominant species it is about 2^depth; "row total
minus own subtree" of those terms, for the dominant lane, is a difference of two astronomically
large nearly equal numbers. At the corner the float64 gradient was 1e8 times larger than central
finite differences of the (corrected) likelihood, with both adjoint solvers (`corner_fd_grad.py`);
in a mild regime (D = -6, L = -3, T = -6) analytic and finite differences agreed to 8e-5, the
step's own truncation error.

**The fix, everywhere: additions only.** A lane's available mass is the mass hanging off its
ancestor chain plus its children's subtree masses; both are sums of non-negative terms built by two
tree walks (subtree sums bottom-up, off-chain sums top-down through parent and sibling). In the
exact forward solve this is carried as an affine function of the mass entering each subtree
(`M = mA + mG*u`, gains `mG` in [0, 1)), which replaced three walks by one and removed the
first-pass rebuild (commit bacea6b1); the exact tangent got the same walk with a signed constant
part (edc92533); the series adjoint term (60a8b993) and the exact adjoint solve (8ad9bf49) now
eliminate on the off-subtree adjoint passed down by addition instead of the row total, the exact
one with a 2x2 sibling coupling per node and no scalar closure; the two sweep tangent kernels share
one helper (`_valid_receiver_sum`). The log forward kernel and the two remaining backward VJP
kernels were converted by agents from the same template (see their commits). A host-side
lane-by-lane residual of the fixed-point equation (`benchmark/cc/exact_row_host_check.py`) went
from 1.9e-8 to 1.4e-13 on the first bad row. After all first-order kernels were converted, the float64 log
and exact paths agree to 1e-11 bits on the corner family (whole-row disagreement 1.8e-11 log2 at every depth), and
the analytic gradient there is -71.944 / +2748.83 / -407.69 (D, L, T) against central finite differences
-71.951 / +2749.09 / -407.73, the step's truncation floor, identically for both adjoint solvers; in a mild regime
nothing moved (8e-5). As a bonus the log sweep got 8-18 % faster: two prefix scans replace a 34-deep ancestor walk.

**Effect at fitted rates: same optimum, cleaner fit.** Full fit with the forward fix alone (job 57791446):
856 s on a shared node, NLL 9048938.300 bits, 5119/5123 certified. With every kernel converted (job
final_fulli, interactive node shared with other users, warm-up 148 s against 125 s quiet): **747.2 s**,
NLL 9048938.292 bits, **5123/5123 certified** in 68 Newton steps instead of about 110, peak 26.7 GiB
(+2 GiB of scratch in the additive backward kernels). The families that used to iterate to the cap were
the ones whose gradient was rounding noise in a high-loss region.

**Rate-box sweep after the fix (job corners_add, 20 families, float32 exact and float32 log against a converged
float64 log oracle at 2048 sweeps / 512 terms, `benchmark/cc/test_exact_range_corners.py`).** 27 of 27 corners
pass (3 failed before): the float32 likelihood is within 8e-3 bits of the oracle at every corner for both paths,
and the exact path is never worse than the log path. At the three corners with loss at its cap and transfer at
its floor, 34-54 % of the rows exceed the 100-order range and take the log sweeps; both float32 paths there show
the same large lane-level disagreement with the oracle (identical for log and exact), which is float32 itself
(extinction probabilities within 2^-24 of one round to one), not the transfer sums, and the likelihood is still
within 3e-3 bits. `exact_range_log2 = 100` keeps a 26-order margin from float32 underflow (2^-126) and stays.

**Per-kernel profile after the third round (RTX 4090, 200 Coleman families, 15 batches, float32, exact
solves; `benchmark/cc/profile_gradient_kernels.py`, `results/profile_kernels_rtx4090_200fam.txt`).**
Forward only: 756 ms of GPU time, of which the exact forward solve 61 %, the gene-split (DTS) reduction
21 %, the log-space prologue 10 %. Forward plus gradient: 2354 ms, so the backward is about twice the
forward: transfer-subtree VJP 12.9 %, gene-split reduction (recomputed in the backward, not stored)
11.3 %, gene-split VJP 10.5 %, exact transposed solve 9.0 %, receiver-weight VJP 6.4 % (computed
even though the receiver weights are frozen in a genewise fit: `_implicit_grad.py` always allocates
its output), prepare kernel 5.0 %, `index_add` scatters 4.5 %, zero-fills 3.6 % (18,620 launches),
event VJP 3.5 %, series adjoint kernel 2.7 % (launched for spilled rows even when no row spilled).
Cheap wins visible here: skip the receiver-weight VJP when the weights need no gradient (6.4 %), skip
the series launch when the spill mask is empty (2.7 %), stop zero-filling per wave (3.6 %), and keep
the forward's gene-split rows for the backward (11.3 %). All four are done -- see the next section.

## Round four: four pieces of backward work that produced nothing (RTX 4090, 2026-09-04)

The profile above named four avoidable items in the gradient. All four are now gone. Everything
below is the same 200-family Coleman batch (RTX 4090, 15 batches of ~100,000 clades, 2013 species,
float32, exact forward and exact adjoint, 1977 waves). Another process shared the card for most of
the session, so each comparison ALTERNATES before and after one measurement at a time -- a single
ordered pass would compare them at two different loads -- and the headline wall clock was taken in
a window when the card was genuinely idle.

**1. The receiver-weight gradient nobody asked for (6.4 %).** A genewise rate fit freezes the
receiver weights, but `_implicit_grad.py` always allocated their cotangent vector, and every kernel
that can add into it reads "the vector exists" as "accumulate into me". `need_receiver_grad` now
travels from the entry points down, with no default anywhere -- getting it wrong would be silent,
because the receiver gradient would still come back with the right shape and only the wrong value.
When it is not asked for, the gradient is None and
`_accumulate_transfer_receiver_log_probability_vjp_kernel` (1977 launches, 160 ms) is never
launched; the receiver branch inside the transfer-subtree VJP also switches off.

**2. The Neumann series with no rows to solve (2.7 %).** The exact transposed solve hands a clade
row to the series only when its elimination is badly conditioned, which almost never happens -- yet
the series ran 1977 times per gradient on an empty mask, and "empty" was not cheap: with every load
and store masked off it still walked the whole species tree once per program, 66.4 ms. The kernel
now returns on its first load when its row is not in the mask: 3.9 ms for the same 1977 launches,
2 microseconds each. Skipping the launch outright was measured too (the exact kernel counts its
spills with an atomic, `ADJOINT_SERIES_SPILL_DECISIONS` switches between reading that count back
per wave and launching regardless, `benchmark/cc/test_adjoint_series_cost.py`): reading it back was
**1.15 s per gradient SLOWER** (median of the per-rep difference, 5 of 7 reps slower), because 1977
stream drains stop the host running ahead. "always" stays the default.

**3. The zero-fills (3.6 %).** Six of the seven `[wave clades x species]` buffers the wave solve
returns were zero-filled on the host so pruned rows would come back clean -- 11,862 memsets per
gradient. `_accumulate_reconciliation_event_vjp_kernel`, the last kernel to write all six, already
stores `where(row active, value, 0)` over every valid row, so the zero was written twice, and the
kernels in between only read those rows behind the same active mask. `FillFunctor<float>` launches
drop from 18,620 (86.7 ms) to 4,735 (26.3 ms). `v_k` keeps its memset: the prepare kernel seeds it
for active rows only and nothing rewrites the pruned ones.

**4. The gene-split (DTS) reduction, run twice (11.3 %).** The forward reduces each split wave's
gene-split row block and throws it away; the backward reduced the same numbers again from the same
Pi/Pibar rows. They are reusable because a child clade's rows are written by the child's own,
earlier wave and never touched again. Checked directly on an 8-family model: over all 132 split
waves, kept minus recomputed is **0 in every element** of both the rows and their row offsets.
`pi_wave_forward` now takes a dict to fill (or None to keep freeing), `implicit_grad_loglik_vjp_wave`
a dict to read (or None to recompute), and the two gradient entry points install one and drop it
straight after. The three reduction kernels go from 3894/3878/3894 launches to 1947/1939/1947.
`memory_policy.forward_gene_split_cache_fits` gates it at build time against the largest batch: it
is one more `[batch clades x species]` tensor, and peak allocated goes **3315 MiB -> 4037 MiB**
(+722 MiB) with peak reserved 6210 MiB -> 6240 MiB. A card that cannot afford it says so on stdout
and keeps recomputing.

**What it bought.** On an idle card, `benchmark/cc/time_exact_vs_iterated.py --limit 200 --reps 5`
(median of 5, min in brackets): one forward is unchanged at **892.9 ms (890.7) before, 891.1 ms
(890.7) after** -- the control that says the forward was not disturbed -- and one
forward-plus-gradient goes **2720.7 ms (2696.2) -> 2326.2 ms (2295.3), -14.5 %**. The backward half
alone is 1828 ms -> 1435 ms, **-21.5 %**. Per-kernel GPU time agrees, measured alternating on the
shared card: forward 776/777 ms -> 767/776 ms, forward+gradient 2855/2865 ms -> 2234/2269 ms, -21 %.

**The gradient did not move.** `benchmark/cc/save_gradient_snapshot.py` saves the float64 copy of
the 200-family theta gradient and compares two snapshots. Before against after: max absolute
difference divided by max absolute gradient is 3.2e-7 and 4.7e-7 on two measurements of the final
code (3.6e-7 and 7.5e-7 at the intermediate stages, and the per-family NLL vector itself is
identical to the last bit throughout). Two runs of the SAME code differ by 2.8e-7 on the same
measure, because the parameter
gradients are accumulated with float32 atomics and are therefore not bit-reproducible run to run --
so the change sits inside the noise the code already had. The receiver-weight gradient, on a model
that asks for it (`receiver_weights.requires_grad_(True)`), moves by 8.3e-7 against an 8.4e-7
run-to-run noise floor. `pytest -q tests/` is 441 passed, 14 skipped, unchanged.

The full after-profile is in `results/profile_kernels_rtx4090_200fam_round4.txt`; it was taken on
the shared card, so its per-kernel milliseconds are inflated by about a quarter while its launch
counts are exact.

## Round five: the two host-side stalls that were left, and what turned out not to matter (RTX 4090, 2026-09-04)

Round four removed backward work the card was doing for nothing. What was left in the gap between
"the card is busy" and "the wall clock" was the HOST: places where python stops and waits for the
driver, so the queue drains and the card has nothing to run. Same 200-family Coleman batch
throughout (RTX 4090, 15 batches of ~100,000 clades, 2013 species, float32, exact forward and exact
adjoint, 1977 waves, flat theta -6/-3/-6), timings taken on a genuinely idle card.

**How "GPU idle" is measured here.** `torch.profiler` with CPU **and** CUDA activities, exported as
a chrome trace; the kernel/memcpy intervals are unioned on the GPU timeline and compared against the
span, so "idle" means the card had nothing queued. The profiler adds host overhead, which inflates
idle, so the wall clock and the CUDA-only per-kernel total are quoted alongside it.

**1. The free-memory reading taken once per wave (commit dbfa8839).** Before allocating a wave's
self-loop scratch the backward asked "does this fit?", and on the cold path -- no resident
warm-adjoint cache, which is the production path whenever that cache does not fit, as it does not
here (44.7 GiB of cache wanted against an 18.5 GiB budget) -- answering meant reading free memory
from the driver on the spot: one blocking `cudaMemGetInfo` plus two `torch.cuda.memory_stats()`
nested-dict builds, 1977 times per gradient. The answer cannot move between waves, because each
wave's scratch is allocated and freed inside that wave. `memory_policy.wave_scratch_budget_bytes`
now reads it once at the top of the reverse sweep and hands the number down as the reservation the
warm path already used, so the gate makes the identical comparison (`scratch <= budget`; these
callers pass no `already_live_bytes`) and a wave that genuinely does not fit is still rejected. Both
reverse sweeps got it: the gradient (`gpurec/api/_implicit_grad.py`) and the exact Hessian's own
loop (`gpurec/solver/hvp/exact.py`).

**2. The extinction fixed point's convergence test (commit 768eab6c).** `e_fixed_point_triton` read
its residual back from the card at the end of every iteration -- 195 of the 240 device-to-host reads
in one forward, each stopping the host until that iteration's kernel had finished. It now reads the
residual one iteration LATE: iteration k's residual is copied into pinned host memory right after
iteration k's kernel (stream-ordered, so it captures that kernel's value before the next iteration
overwrites the buffer) and is only looked at once iteration k+1 has been queued. The iterate handed
back is the same one bit for bit -- the double buffer already holds it, so when the late test says
iterate k converged we swap back and return exactly what the immediate test returned. The cost is
one extra e-step launch per solve; the 195 max-reduction launches it replaces make the forward's
launch count go DOWN, 15,751 -> 15,571.

| 200 families, idle card | before | after |
|---|---|---|
| `cudaMemGetInfo` per gradient | 1977 | 15 |
| blocking device-to-host reads, one forward | 240 | 45 |
| blocking device-to-host reads, one gradient | 375 | 180 |
| GPU idle, one forward (profiled span) | 35.7 ms of 794.3 (4.5 %) | 35.2 ms of 792.6 (4.4 %) |
| GPU idle, one gradient (profiled span) | 565.1 ms of 2493.4 (22.7 %) | 238.7 ms of 2163.0 (11.0 %) |
| wall, one forward (median of 5) | 782.1 ms (min 781.1) | 780.2 ms (min 777.6) |
| wall, one forward+gradient (median of 5) | 2175.3 ms (min 2157.5) | 2030.8 ms (min 2023.4) |

The forward's wall clock does not move, which is the control that says nothing was disturbed: the
forward is GPU-bound on this card (756 ms of CUDA time inside 780 ms of wall), so its 35 ms of idle
was never the E-step's stalls -- those were absorbed by work already queued. The stalls and the
launches are gone all the same, which is what a card roughly twice as fast would have been starved
by. The gradient is the one that pays: **-144 ms, -6.6 %.** A second before/after pair taken while
another process shared the card gave 2407.8 -> 2150.8 ms, same direction.

**The gradient did not move.** `benchmark/cc/save_gradient_snapshot.py`, before against after: the
per-family NLL vector is identical to the last bit, and the theta gradient's max absolute difference
divided by max absolute gradient is 5.9e-7 against a 7.9e-7 run-to-run noise floor measured from two
runs of the same code (float32 atomics are not bit-reproducible run to run). `pytest -q tests/` is
441 passed, 14 skipped, unchanged.

**3. What is left of the per-wave zero-fills, and why none of it should go.** One gradient does 6322
CUDA zero-fills writing 23,009 MiB. Two buffers are essentially all of those bytes and both must
start at zero: `accumulated_rhs` (`_implicit_grad.py`, one `[batch clades x species]` per batch, 15
launches, ~11.5 GiB -- the reverse sweep accumulates each clade's children into it, and a clade that
is neither a root nor anybody's child must read zero) and `v_k` (`wave_backward.py`, 1977 launches,
~11.5 GiB -- already argued in round four: the prepare kernel seeds active rows only and
`_scatter_accum` sums every row). Together they are the 4735 `FillFunctor<float>` launches costing
25.7 ms, 1.3 % of the 1921 ms the card is busy, and at 23 GiB in 25.7 ms that is ~900 GB/s, i.e.
write bandwidth rather than overhead. Everything else is small: `series_rows` (1977 int8 fills,
2.0 ms), float64 fills (2247, 2.4 ms), int64 fills (525, 0.5 ms) -- 5 ms, 0.26 % of GPU time, and
each one is read by a kernel that expects zeros where the pruner skipped a row. Nothing here is
provably redundant, so nothing was removed. The `torch.empty` calls are a different animal: 35,857
per gradient, no kernel at all, ~21 ms of pure host time -- see the next point for why that does not
show up either.

**4. The per-launch validation helpers cost host time and buy no wall clock here.**
`_validate_residual_tensors`, `_validate_offset_tensor` and `_prepare_wave_launch`
(`gpurec/core/kernels/pi_forward.py`, re-exported into the backward and tangent kernel modules) run
5916 + 23,604 + 3954 times in one forward and 11,817 + 33,444 + 3954 times in one gradient.
Microbenchmarked at the real argument shapes they are 1.79 / 0.36 / 0.82 microseconds a call, so
**22.4 ms of host time per forward and 36.6 ms per gradient**. Replacing all three with do-nothing
stubs and timing forward and gradient alternately against the real ones (5 pairs,
`stub_validators`-style A/B) changed the wall clock by **-0.4 ms on the forward and -4.9 ms on the
gradient -- both inside the run-to-run noise, and the gradient's is negative**. They were therefore
left alone: the host time they cost overlaps GPU work that is already queued, and the
behaviour-preserving version of "hoist them" is not free either (each wave validates different
tensors -- its own row slices and its own offsets -- so checking only the first wave would stop
checking the rest, which is a behaviour change, not a hoist). If a faster card makes the host the
critical path, the cheap fix is to make the checks cheaper, not fewer.

**Where the headroom stands after this round.** CUDA-only per-kernel totals
(`benchmark/cc/profile_gradient_kernels.py`, idle card): forward **756 ms** of GPU time inside
780 ms of wall (96.9 % GPU-bound, 24 ms of host headroom left); forward+gradient **1921 ms** inside
2031 ms of wall (94.6 % GPU-bound, 110 ms left). Before this round the same 1921 ms of GPU work sat
inside 2175 ms of wall, i.e. 254 ms of headroom, of which 144 ms is now gone. The remaining 110 ms is
spread over ~46,000 sub-microsecond gaps between 67,680 kernel launches (python and Triton dispatch
between launches), not over a few big stalls, so it will not come back from removing any one call
site -- it needs fewer launches or bigger ones.

## Round four, recipe side: why Newton needed 8 to 14 steps, and what was changed (2026-09-04)

Per-family traces of every gradient evaluation on 200 Coleman families (`benchmark/cc/recipe_trace.py`):
median 19 evaluations per family, 5 of them the Adam warm-up; only 6 % of the evaluations happen after a
family's NLL has stopped improving by more than 1e-4 bits, so the iterations genuinely move the fit, just
slowly. Steps are short: 4 % hit the 2.0 log2 cap, the median step is 0.29 log2 units while the median
family has to travel 6.5 units, and paths are 1.75 times longer than the straight line. Two mechanisms
were visible in the individual trajectories. The Adam warm-up (learning rate 1.0) moves every rate about
one log2 unit per step whatever the gradient says and overshoots: typical families end the warm-up with a
worse NLL than two steps earlier. And the curvature floor `mu = 1e-2` throttled flat directions: a rate
heading towards zero has a gradient and a curvature both proportional to the rate, so its Newton step was
gradient / 1e-2, 0.1 to 0.2 log2 units per iteration, for 20 iterations while the family's NLL moved by
0.01 bits.

Changes (`gpurec/fit/genewise_fit.py`): each eigen-direction of the Newton step is now bounded by the
family's trust radius through `lam = max(e, mu, |g_v| / radius)` instead of flooring tiny curvature at
`mu`, with `mu = 1e-4` as the sign guard; a per-family adaptive trust radius with the standard ratio test
(gated on the float32 noise floor of a family's NLL, 0.05 bits, and consuming each pending test once) is
in place but measured neutral once gated (ungated it oscillated: 115 steps). Sweep on 200 families, cost in
full-dataset gradient equivalents with Hessians priced at their live share:

| recipe | Newton steps | cost | NLL vs baseline |
|---|---:|---:|---:|
| baseline (adam_bfgs start, mu 1e-2, fixed cap 2) | 27 | 23.5 | 0 |
| cap 4 / cap 8 | 26 / 111 | 23.6 / - | 0 / +0.24 |
| per-direction cap, mu 1e-4 | 22 | 21.6 | -0.07 |
| + exact starting Hessian | 26 | 19.1 | -0.74 |
| exact Hessian every iteration (adam 5 / adam 2 / no Adam) | 11 / 10 / 19 | 20.2 / 19.8 / 22.9 | -0.72 / -0.07 / -0.70 |
| exact Hessian every 2 iterations, adam 2 | 13 | 18.2 | -0.07 |

Exact curvature does what it should (10 to 11 Newton steps instead of 27) but one 3-probe Hessian costs
7.9 gradients on the RTX 4090 with the probes' cache cold, and the same regime holds at full scale on the
H100 (the cache wants 688 GiB). The full-dataset run with the exact starting Hessian (job r4a_fulli,
shared node, so the wall time is not comparable): Newton gradients 317 s against 449 s before, 55 steps,
NLL 9048930.68 bits (7.6 bits better than every earlier run), all 5123 certified, but 376 s in four
Hessians. Production therefore keeps the warm-up curvature as the start with the per-direction cap and
`mu = 1e-4`; making the Hessian cheap (sharing the direction-independent work across the three probes,
then batching the three directions) is the open item that would unlock exact Newton.

## Round six: the adjoint self-loop in one register-resident kernel (RTX 4090, 2026-09-04)

After round four the two kernels that solve the transposed self-loop were 16.9 % of one gradient's
GPU time: `_prepare_reconciliation_self_loop_vjp_kernel` 115.1 ms and
`_exact_tree_self_loop_transpose_kernel` 210.3 ms of 1926 ms, on the same 200-family Coleman batch
(15 batches of ~100,000 clades, 2013 species, float32, exact forward and exact adjoint, 1977
waves). Between them they moved **eight** `[wave clades x species]` arrays through global memory --
each one 12 GB per gradient at this batch size:

* five coefficients the prepare kernel wrote and the elimination read back (the self-loop diagonal,
  the donor coefficient, the receiver mass, the two speciation edge probabilities);
* two running sums for the valid receiver mass, written and read inside the prepare kernel;
* three working arrays the elimination's own two species-tree walks wrote and re-read at every
  level.

`_solve_reconciliation_self_loop_transpose_row_kernel` does all of it in one launch, one program
per clade row: the species row sits in registers as `BLOCK_S` lanes across several warps, the
coefficients are built from the primal row on the spot, and the two elimination walks reach
children, parent and sibling with `tl.gather`. This is the shape the forward tangent
(`_solve_reconciliation_self_loop_jvp_exact_kernel`) has used since the exact tangent landed. The
prepare kernel and the Neumann series stay for the `"series"` adjoint mode and for the rows the
elimination refuses on conditioning grounds; on the exact path prepare is now launched with only
those rows and returns on its first load when there are none.

**The arithmetic is unchanged on purpose**, term for term and in the same order, so the gradient
does not move. In particular the valid receiver mass is still the depth-first "not yet open" plus
"already closed" pair of running sums summed by `tl.cumsum` over the same `BLOCK_S` lanes -- only
the two gathers around it now read registers instead of a scratch array. A 2013-lane test says the
1-D scan is bit-identical to the `[BLOCK_S, 1]` one at 4, 8 and 16 warps. It is deliberately NOT
the additive tree walk in `species_tree_sums.py`, which is the same number to a different rounding.
The one thing that could not be carried over literally is the scatter: the prepare kernel wrote
"speciate at parent(t), follow the edge into t" into the child's slot, and a register-resident
program cannot write another lane's register, so each lane gathers the number its parent computed
for it instead. Same number.

**What it bought, and what it did not.** Per-kernel GPU time at matched totals:

| | before | after |
|---|---|---|
| `_prepare_reconciliation_self_loop_vjp_kernel` | 115.1 ms | 6.2 ms |
| `_exact_tree_self_loop_transpose_kernel` | 210.3 ms | (deleted) |
| `_solve_reconciliation_self_loop_transpose_row_kernel` | -- | 312.3 ms |
| the three together | 325.4 ms (16.9 %) | 318.5 ms (16.6 %) |
| one gradient, all kernels | 1926 ms | 1912 ms |
| peak CUDA memory | 4041 MiB | 4041 MiB |

So the round trip really is gone -- Nsight Compute on one wave puts DRAM throughput at **36 % of
peak before and 1.5 % after** -- and the gradient is about **2 % faster overall**, not the 12 to
19 % the array traffic suggested. The reason is that neither kernel was bandwidth-bound to begin
with; both were latency-bound at 16 % occupancy, and `tl.gather` pays back in the shared-memory
pipe what the arrays cost in DRAM. After the change the kernel's top stall is `mio_throttle` at
2.43 warps stalled per issue (the memory-input/output queue that `tl.gather` stages through), L1
throughput 53 %, SM throughput only 7.6 %; registers 112 per thread at 16 warps and occupancy 33 %,
up from 187 registers and 16 %.

An isolated cost model separates the two halves -- the pipeline cannot, because its row pruner
reads the adjoint this kernel produces, so shortening the walks changes how many rows the next wave
even looks at. Capturing one real 254-row wave's arguments and re-launching the kernel on them
(`num_warps=16`): the whole solve is 58.3 us, of which the coefficient setup is **18.4 us (32 %)**
and the two species-tree walks are **40 us (68 %), 1.21 us per level**. The saving is the setup
half; the walks cost what the block-tiled walks through global memory cost.

Warp sweep on the whole gradient, this kernel only: 4 warps 342.8 ms, 8 warps 328.6 ms, **16 warps
312.3 ms**, 32 warps 323.9 ms. 16 is the default.

**The gradient did not move.** `benchmark/cc/save_gradient_snapshot.py`, 200 families,
`--receiver-grad` on so the receiver-weight path is exercised too: the per-family NLL vector is
identical to the last bit (max difference exactly 0), `grad_theta` moves 4.272461e-04 out of
7.717233e+02 (**5.5e-7 relative**) and `grad_receiver` 2.29e-05 out of 5.77e+01 (4.0e-7) -- against
two runs of the SAME code, which differ by 4.272461e-04 (5.5e-7) and 1.91e-05 (3.3e-7). The change
sits exactly at the run-to-run noise of the float32 cross-wave atomics. In float64, where there are
no atomics to reorder, the COG0009 corner gradients (`benchmark/cc/corner_fd_grad.py --step 1e-3
--adjoint exact`) are bit-identical to before the change at both the loss-rate cap
(D -7.194388992e+01, L +2.748826774e+03, T -4.076941077e+02) and the mild point
(D +1.388556863e+01, L -3.773244493e+02, T -6.182668980e+02). `pytest -q tests/` is 441 passed,
14 skipped, unchanged.

Wall clock could not be resolved. Another process shared the card for most of the session, so
every measurement alternated the two builds one at a time (`git checkout 2620fa3d -- gpurec/`
between arms) and only the runs whose FORWARD came out at its quiet value of ~783 ms count -- the
forward is identical in both arms, so it is the control that says the card was free. Those runs
give `forward+gradient` **2133.3 ms and 2348.3 ms before, 2202.2 ms and 2326.2 ms after**: two
overlapping ranges about 10 % wide, around a change worth 14 ms of GPU time. The per-kernel numbers
above, taken at matched gradient totals (1926 ms before, 1912 and 1917 ms after), are the
load-bearing measurement.

The full after-listing is in `results/profile_kernels_rtx4090_200fam_round5.txt`; the card was busy
when it was captured, so its milliseconds are about double while its launch counts and percentages
are exact.

**Where the remaining cost is, for whoever picks this up.** The walks are 68 % of the kernel and
are throttled on bank-conflicted `tl.gather`s: the child, parent and sibling indices are arbitrary
species numbers, so each gather is a many-way shared-memory conflict. The bottom-up walk also runs
one whole-row pass per species-tree LEVEL, and this tree is badly balanced -- of 33 levels, the
deepest 14 hold 23 of the 1006 internal nodes, so 42 % of the passes do 2 % of the work. Cutting
either would pay; the per-node algebra must not change, because the elimination triples, the 2x2
sibling coupling and the additive `Off` are what keep a catastrophic cancellation out of the
adjoint (commits 60a8b993 and 8ad9bf49).

## Round seven: what the three gene-split (DTS) kernels actually move (RTX 4090, 2026-09-04)

The three biggest gene-split kernels of a gradient were measured byte for byte against the minimum
their arithmetic needs. Same 200-family Coleman batch as every round above (15 batches of ~100,000
clades, 2013 species, float32, exact forward and exact adjoint, 1977 waves, flat theta -6/-3/-6).
Nsight Compute numbers are one launch of the LARGEST wave of a 40-family gradient at the same
theta: 14,507 splits sharing one parent clade, so 29,014 split sides, S = 2013, one species row =
8052 bytes. That wave is representative by weight -- half of this kernel's time is in launches of
4645 splits or more -- and its profiled milliseconds are inflated by Nsight's serialization, so
they are used only against each other.

**Per split, what each kernel moves, against what it needs.**

| | reads | writes | the minimum one pass needs |
|---|---|---|---|
| forward `_stage_multiple_gene_split_event_reduction_kernel` | 32,593 B | 250 B | 32,208 B = the four child rows (Pi and Pibar of both children) |
| backward `_accumulate_gene_split_event_vjp_kernel` | 48,524 B | 30,240 B | 48,312 B read = those four rows + the two child rows of `accumulated_rhs` it adds into; 32,208 B written = those two rows + the two staged donor rows |
| backward `_accumulate_transfer_subtree_vjp_kernel` (per SIDE) | 24,193 B | 19,105 B | 24,156 B read = the side's donor row, the child's Pi row, the `accumulated_rhs` row; 8052 B written = that row |

So **the forward reduction moves 1.2 % more than the minimum and runs at 90.4 % of DRAM peak** --
it is a streaming sum over splits, each split's four child rows are read exactly once, and there is
nothing in it to remove. **The gene-split VJP's reads and writes are at the minimum too**: it reads
212 B per split more than the six rows it must touch -- `Pi[parent]` and `v_k[parent]`, which 3.17
splits share on average and all 14,507 share in this wave, cost essentially nothing because L2
absorbs them -- and it writes 6 % LESS than four rows, because the rows the adjoint pruner marked
inactive are skipped. Only the transfer-subtree VJP moved more than it needed, and only on the
write side: 2.37 rows written per split side where one would do.

**The reuse the profile suggested is not there.** Splits of one parent are contiguous in
`sl`/`sr`/`reduce_idx` (`build_wave_layout` sorts them that way) and 3.17 splits share a parent,
but the parent rows are already free -- see above. Child clades are the ones that would matter, and
they are barely shared: over all 1962 split waves, 8,964,908 child slots point at 6,815,175
distinct child rows, **1.32 uses per row**, and in the wave above every one of the 14,507 left
children and 14,507 right children is distinct. Tiling by child row cannot pay at 1.32.
The species-child gathers are not a problem either: `Pi[child1(s)]` over consecutive species
touches 363 of the 32-byte lines of an 8052-byte row against 252 for a contiguous pass, and
`Pi[child2(s)]` 283 -- the species numbering is already local.

**What was scattered: the transfer-complement walk's stores.** That kernel gives every species the
sum of a split side's donor adjoints lying OFF its subtree, built by addition only (round three's
cancellation fix). Its top-down pass wrote one number per species back into the species row, at
whatever index the tree structure chose. Indexing a species row by tree structure is scattered: a
whole pass over the 1006 internal nodes' parent slots touches 879 lines, over their first children
883 and their second children 883, where a contiguous pass touches 252. Nsight put the kernel at
**92.7 % of L2 throughput** with 66.5 M write sectors against 33.4 M read sectors and said only 5.7
of every 32 stored bytes were used.

The pass now stores one number per INTERNAL NODE, in the compact-level order the walk already
iterates in, so the store is contiguous (126 lines): `pair[j] = off-subtree(node j's species) +
own(node j's species)`, which is exactly the quantity both children of node j add their sibling's
subtree sum to. The species row keeps its subtree sums, so the kernel's last pass finishes any
species -- leaf or internal -- with one add, `pair[slot of my parent] + subtree[my sibling]`.
`pair[j]` overwrites `own[j]` in the scratch that already existed (the pass reads a node's own term
and writes its pair in the same statement, each slot belongs to exactly one level, and no other
node reads slot j's own term); a second array of the same shape was tried and cost its own 117 MB
of DRAM writes on this wave, +0.12 ms. Four int32 tables built once per species tree
(`_off_subtree_walk_tables` in `gpurec/core/scheduling/batching.py`) say where the two operands
live, and the root-zeroing pass over the species row is gone, because "the root has no parent slot"
is now what says its off-subtree sum is zero.

| one launch, 29,014 split sides | before | after |
|---|---|---|
| L2 throughput | 92.7 % | 65.8 % |
| L2 write sectors | 66.5 M | 35.6 M |
| L1 store sectors | 81.8 M | 35.7 M |
| DRAM throughput | 65.0 % | 73.8 % |
| duration | 1.97 ms | 1.73 ms |

Over the whole 200-family gradient, taken at a matched load -- the untouched
`_accumulate_gene_split_event_vjp_kernel` next to it reads 278.0 ms before and 278.9 ms after, 0.3 %
apart, which is what says the card was in the same state -- `_accumulate_transfer_subtree_vjp_kernel`
goes **302.0 ms -> 270.7 ms, -10.4 %**. Wall clock, alternating the two builds with the forward as
the control (unchanged at 780 ms in all six runs, its quiet value): forward+gradient medians 2016.9
and 2038.3 ms before against 1996.5 and 2010.2 ms after, minima 2008.2 / 2007.7 before against
1988.6 / 1999.2 after -- **about -20 ms on 2020 ms, -1.1 %**, which is what a 31 ms kernel saving
should look like. Peak CUDA memory is unchanged at 4041 MiB.

**Nothing moved.** The float64 COG0009 corner gradients are bit-identical to 17 digits at both
corners (loss-rate cap D -71.943889924633581, L 2748.8267744464943, T -407.69410771831559; mild
point D 13.885568633078321, L -377.32444930100792, T -618.26689801293253). The 200-family
per-family NLL vector is identical to the last bit. `grad_theta` moves 8.3e-7 and 5.5e-7 relative
on two before/after pairs against 5.9e-7 and 4.7e-7 between two runs of the SAME code -- the float32
cross-wave atomics, whose ORDER is all this changes. `pytest -q tests/` is 441 passed, 14 skipped.

**Three things that were measured and NOT kept.**

1. *Chunking the two backward kernels so the staged donor rows stay in L2.* The gene-split VJP
   writes a `[2 x splits, species]` array (233 MB on this wave) and the transfer-subtree VJP reads
   it back; splitting the wave into 1024-split chunks and running the pair per chunk makes that
   array 16 MB, inside the 4090's 72 MB L2, and PyTorch's allocator hands back the same block every
   chunk. It works as advertised on the writes -- DRAM writes per side 19,105 B -> 5,620 B -- and
   not at all on the reads (24,193 B -> 24,248 B). Both kernels came out ~15 % SLOWER against the
   forward self-loop as control, because a 2048-program grid does not fill the card. At 4096 splits
   the grid is fine but the L2 saving is gone (DRAM writes back to 20,591 B per side). Reverted.
2. *Giving each program both sides of a split,* to halve the CTA barriers, which are the kernel's
   top stall after the change above (6.8 of the 19.4 cycles between two issued instructions; 13 of
   the tree's 33 levels hold 2 nodes or fewer and 1006 internal nodes are spread over 34 passes of
   256 lanes, so most warps reach each barrier with nothing to do). It does what it says to the
   instruction side -- L1 load requests 58.7 M -> 37.1 M, SM throughput 62.3 % -> 48.2 % -- but
   doubling a program's live working set from 16 KB to 32 KB
   makes the caches worse: L2 read sectors +38 %, DRAM writes +14 %. Net 1.73 -> 1.69 ms on the big
   wave and nothing measurable over the gradient. Reverted -- and it is the direct evidence against
   the larger version of the same idea, fusing the two backward kernels into one launch: that would
   enlarge the working set further still.
3. *A wider species block for the gene-split VJP.* That kernel is not bandwidth-bound but
   memory-LATENCY-bound: 84 % of its cycles have no eligible warp, 0.30 eligible warps per
   scheduler out of 9.85 active, top stall a memory scoreboard dependency at 20.6 of 61.6 cycles,
   and its access pattern is already near-perfect (27.0 of 32 bytes used per loaded sector, 26.2
   per stored one). `BLOCK_S` 256 -> 512 gives each thread two independent loads in flight and is
   **1.83 ms -> 1.67 ms, -8.7 %** on the big wave, worth about 1.2 % of a gradient (1024 is the
   same, 1.67 ms; `num_warps` 4 instead of 8 is 1.65 ms, inside the noise). It was NOT taken,
   because `BLOCK_S` also blocks the per-program sum over species that forms the DTS rate gradient,
   so the float64 corner gradients move by 1 to 3 units in the last place instead of staying
   bit-identical (d/dT -407.69410771831559 -> ...565, d/dD 13.885568633078321 -> ...33). It is one
   number in `wave_backward.py:1451` if a later round decides that certificate can be relaxed.

**Where the gene-split work stands.** The forward reduction is at the DRAM roofline with nothing
to remove. The gene-split VJP is at its minimum traffic and latency-bound, so the lever there is
instruction-level parallelism (point 3), not bytes. The transfer-subtree VJP is now balanced --
DRAM 73.8 %, L2 65.8 %, L1 62.8 %, SM 62.4 % -- with no single unit above 80 %, so its ceiling from
here is 1.36x and would take a rewrite: its remaining excess is the bottom-up pass, which still
writes its subtree sums scattered across the species row (879 lines against 126) and so rewrites
that whole row through DRAM. Moving those sums to a node-indexed array costs a wider staged row
(one extra slot per internal node, +117 MB on this wave, and a stride through the gene-split
kernel) and buys about 9 % of DRAM traffic -- roughly 1.2 % of a gradient, which is why it was
priced and left.

## What is left

With both self-loops and the tangent solved exactly, one full-dataset gradient at fitted rates costs
~30 s (79 batches x 0.4 s) and is spread over the per-wave backward kernels (prepare-VJP, transfer-subtree
VJP, gene-split VJP), the exact forward and adjoint solves (~15 % each), the DTS forward reduction and the
`index_add` scatter; all are latency-bound at low occupancy rather than compute- or bandwidth-bound, so
further gains need occupancy/tiling work per kernel. Round four removed the backward work that was
producing nothing at all (see above); what is left in the backward is genuine arithmetic. Round five
removed the host-side stalls that drained the queue around it, and the 200-family gradient is now
94.6 % GPU-bound, so the next gain has to come from the kernels themselves rather than from the
python driving them. The warm-up (5 Adam gradients, ~125 s) and the tail

~30 s (79 batches x 0.4 s) and is spread over the per-wave backward kernels (the transposed self-loop
solve, transfer-subtree VJP, gene-split VJP), the exact forward solve (~24 %), the DTS forward reduction
and the `index_add` scatter; all are latency-bound at low occupancy rather than compute- or
bandwidth-bound, so further gains need occupancy/tiling work per kernel. Round four removed the backward
work that was producing nothing at all and round five removed the adjoint self-loop's global-memory round
trip (both above); what is left in the backward is genuine arithmetic, and round five is the direct
evidence that removing memory traffic from a LATENCY-bound kernel buys little -- the next gains have to
come from the walks themselves. The warm-up (5 Adam gradients, ~125 s) and the tail
of a few families iterating to `max_iter` (30-120 s, run-dependent) are the remaining recipe-level costs;
the stall rule (`stall_patience`) trades the tail against 3-4 certified families and is off by default.
Timings on the shared GPU nodes vary by ±100 s with other users' CPU load; pin a quiet node
(`sbatch --nodelist=...`) for measurements. Independent families still shard perfectly across GPUs
(`run_genewise_sharded.py`).

Round seven corrects the "all latency-bound at low occupancy" line above for the three gene-split
(DTS) kernels, which are together 35 % of a gradient: measured against the bytes their arithmetic
needs, the forward reduction is at **90 % of DRAM peak** moving 1.2 % more than the minimum, the
gene-split VJP is at its minimum traffic and memory-LATENCY-bound at 82 % occupancy (not low), and
the transfer-subtree VJP is now balanced with no unit above 74 %. None of them has redundant
traffic left to remove; see that section for the per-split accounting and for the three
experiments -- L2 chunking, two split sides per program, a wider species block -- that were
measured and not kept.
