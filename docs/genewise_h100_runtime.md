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

## What is left

The gradient itself is GPU-bound (96 % busy) with two kernels taking two thirds of the time,
so further gains need kernel work (`ncu` on `_update_reconciliation_likelihood_kernel` and the
backward self-loop kernel). The Hessian costs about 7 gradients (and 4x more at the
`pi_iters=64` certificate tier, where the tangent self-loop kernel runs 64 fixed iterations);
the recipe evaluates it every 5 iterations and once for the certificate. Independent families
shard perfectly across GPUs (`run_genewise_sharded.py`).
