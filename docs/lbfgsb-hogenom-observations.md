# L-BFGS-B HOGENOM Observations

This note summarizes the current L-BFGS-B experiment on the HOGENOM
specieswise objective and the follow-up work I would prioritize.

## What Was Implemented

The new optimizer is a PyTorch L-BFGS-B-style optimizer for one dense bounded
parameter tensor. It follows the structure of the Schilling L-BFGS-B reference
implementation at <https://jonathanschilling.github.io/L-BFGS-B/index.html>:

- projected-gradient KKT residuals for box constraints
- raw BFGS curvature pairs, `s = x_new - x_old` and `y = g_new - g_old`
- generalized Cauchy point computation
- free-subspace step after the Cauchy point
- bounded line search and rate projection

It is not a line-for-line port of the Fortran/Python reference routines. It is
adapted to this codebase's single `model.theta` tensor, PyTorch autograd, and
expensive HOGENOM loss evaluations. In particular, it uses loss-only probes in
line search and refreshes gradients only after an accepted step.

## Test Coverage So Far

The implementation was tested for parity with SciPy's `method="L-BFGS-B"` on:

- a boxed quadratic whose solution lies on the boundary
- a bounded Rosenbrock problem

Both tests converged to the same solution and objective within tight tolerances.
The workflow integration also has focused tests for:

- the `lbfgsb` optimizer mode
- projected-gradient metric recording
- loss-only probe accounting
- KKT-aware stopping behavior for bounded optimizers

These tests validate basic algorithm behavior, but they are not yet conformance
tests against the Schilling reference traces for every internal routine.

## HOGENOM Result

Starting from the previous projected-LBFGS stuck checkpoint:

- previous best NLL: `528695.6875` bits
- previous projected gradient: about `17.6`
- new `lbfgsb` best/final NLL: `525239.5` bits
- elapsed time: about `360.5s`
- improvement: about `3456` bits
- final raw gradient infinity norm: about `112.9`
- final projected gradient infinity norm: about `36.6`
- final status: `not_converged`, reason `max_steps`

So the new L-BFGS-B optimizer clearly improved the optimization and escaped the
previous bad point. It did not achieve KKT convergence.

A direct projected-gradient probe at the final point showed that descent still
exists, but only at very small step sizes. Large projected-gradient steps made
the loss much worse. Around `alpha = 6.1e-05`, the loss improved by only
`0.0625` bits. This suggests the final region is very stiff and the objective is
being probed near float32 loss resolution.

After changing Armijo acceptance to compare scalar Python floats instead of
doing the acceptance arithmetic in the tensor dtype, a short continuation from
the best checkpoint improved one more notch from `525239.5` to `525239.4375`.
That confirms some no-op or near-no-op steps were previously being accepted
because the required Armijo decrease rounded away at this objective scale.

## Why Finishing Is Hard

The hard part is not reducing the likelihood. The hard part is making the
projected gradient approach zero under box constraints.

The main issues appear to be:

1. Bad conditioning. The projected gradient can be large while useful step
   sizes are tiny. That is typical of steep, narrow regions.

2. Float32 objective resolution. At an NLL around `525000` bits, small changes
   can be quantized. Near the end, useful improvements may be only `0.0625`
   bits or less.

3. Approximate objective and gradients. The HOGENOM workflow is using fixed
   solver iteration budgets, such as low Pi and Neumann counts. BFGS curvature
   pairs rely on consistent gradients, so solver approximation error can make
   the Hessian model noisy.

4. Box constraints. Bound projection changes the active free subspace. Curvature
   gathered before an active-set change may be a poor model for the current
   free subspace.

5. Limited memory. With thousands of theta coordinates and a history size around
   20, L-BFGS only models a low-rank slice of curvature. If the remaining hard
   directions are not represented in recent history, it can still propose steps
   that are too aggressive or poorly oriented.

## Boundary Correctness Concerns

The current implementation uses the projected-gradient mapping

```text
pg = x - project(x - g)
```

which is the right KKT residual for bound constraints. It also zeros directions
that would immediately leave the feasible box at active bounds.

The part that needs more scrutiny is not the definition of projected gradient.
It is the full active-set behavior of the Cauchy point and free-subspace step.
The implementation follows the L-BFGS-B structure, but it has not yet been
validated routine-by-routine against reference `cauchy`, `subsm`, and `mainlb`
outputs. If there is a subtle boundary bug, it is most likely there, not in the
basic KKT residual.

## Recommended Next Steps

1. Add conformance tests against the reference implementation.

   Port or call the Schilling reference spec cases for `cauchy`, `subsm`, and
   the top-level iteration loop. Compare Cauchy points, active/free sets,
   accepted steps, and projected gradients. This is the fastest way to answer
   whether the boundary behavior is truly L-BFGS-B-correct.

2. Treat no-op Armijo acceptance as a hard failure.

   Keep scalar float comparisons for Armijo. Also log:

   - predicted decrease
   - required Armijo decrease
   - realized loss decrease
   - whether the realized decrease is distinguishable at the current loss scale

   If the loss does not strictly improve while projected gradient is large, the
   optimizer should reject the step, shrink the trust radius or learning rate,
   and keep searching.

3. Add a high-KKT stall fallback.

   When `grad/projected_inf` is above tolerance and L-BFGS-B repeatedly accepts
   tiny or no-improvement subspace steps:

   - clear BFGS history
   - try a projected-gradient step with adaptive backtracking
   - or switch to a projected trust-region step

   The direct probe showed that projected-gradient descent can still improve the
   objective if the step size is small enough.

4. Separate line-search radius from BFGS history.

   A bad line-search scale should not necessarily invalidate curvature history,
   and bad curvature should not necessarily force the global base learning rate
   to collapse forever. Track these separately:

   - curvature history quality
   - active-set changes
   - accepted alpha distribution
   - consecutive tiny-step count

5. Try higher-accuracy refinement after the objective reaches the current basin.

   Once the optimizer reaches about `525239` bits, rerun the refinement phase
   with more accurate inner solves, for example higher Pi and Neumann budgets.
   This can reduce gradient inconsistency and make BFGS curvature pairs more
   meaningful.

6. Consider higher-precision scalar loss evaluation for line search.

   Full float64 kernels may be too expensive or unsupported, but line-search
   acceptance should avoid fp32-scale artifacts where possible. If true float64
   objective probes are not practical, at least use scalar float comparisons and
   conservative acceptance thresholds.

7. Add restart criteria based on active-set churn.

   If many coordinates change active/free status, old curvature pairs may be
   misleading. Restarting the L-BFGS history when active-set churn is high may
   make the free-subspace model more reliable.

8. Record enough diagnostics to distinguish the failure mode.

   For every L-BFGS-B step, log:

   - direction type: Cauchy, subspace, or projected-gradient fallback
   - active coordinate count
   - free coordinate count
   - accepted alpha
   - step infinity norm
   - directional derivative
   - predicted versus realized decrease
   - curvature update accepted/skipped
   - `s'y`, `y'y`, and the BFGS scaling `theta`

   These will show whether the blocker is bad curvature, bad line search,
   active-set churn, solver noise, or loss precision.

## Current Bottom Line

The new L-BFGS-B implementation improved the HOGENOM objective substantially,
but it did not finish optimization in the KKT sense. The most likely blocker is
a combination of bad conditioning, noisy approximate gradients, active-set
changes, and float32-scale line-search resolution. The immediate priority should
be reference conformance tests plus a robust high-KKT stall fallback, not simply
running more of the same L-BFGS-B loop.

## Continuation Findings

Follow-up experiments changed the recommended HOGENOM schedule.

The fixed `6/6/6` and `16/16/16` solver budgets are useful for finding basins,
but they are not reliable objective references. A fixed `16/16/16` checkpoint
reached `520724.0625` bits, but the same theta evaluated at fixed `32/32/32`
was `526303.3125` bits. That is too large to treat the lower-budget value as
near the true objective.

Two high-accuracy branches behaved differently:

- A `32/32/32` polish from the fixed-16 theta reached `526204.5` bits at
  fixed 32, but fixed `64/64/64` evaluation of the same theta was
  `529628.0625` bits. This is another solver-budget artifact and should not be
  used as the main path.
- A clean `32/32/32` run from initialization is slower, but stable across
  higher budgets. Its current best checkpoint is:

```text
/tmp/gpurec_hogenom_specieswise_lbfgsb_high32_from_init_continue3/checkpoints/best.pt
step=360
fixed32 NLL=526885.875
fixed32 projected_grad_inf=2.9347386360168457
fixed64 NLL=526887.6875
fixed64 projected_grad_inf=3.440159320831299
fixed128 NLL=526887.6875
fixed128 projected_grad_inf=3.4550485610961914
```

For the current code and data, the fastest credible route is therefore:

1. Optimize with fixed `32/32/32` from initialization or from a checkpoint that
   has already been shown to be stable under fixed `64/64/64`.
2. Periodically run final-only fixed `64/64/64` and fixed `128/128/128`
   evaluations by resuming the checkpoint with `--steps` equal to checkpoint
   `next_step`.
3. Keep a checkpoint only if fixed 64 and fixed 128 agree within roughly one
   bit. At this objective scale, smaller differences are close to float32
   resolution.
4. `--final-check-iters` now works for specieswise models that expose
   `configure_solver_iterations`, so it is useful for validating the final row
   of a run. For checkpoint-by-checkpoint validation, separate final-only
   resume runs are still clearer because they validate the exact saved theta.

When changing solver budgets for validation, clear optimizer state and stale
status fields (`best_nll_bits`, `best_step`, `previous_objective`,
`stable_loss_steps`) in a copy of the checkpoint. Otherwise the workflow can
carry low-budget best metadata into a high-budget run.

## Latest Validated Branch

The `32/32/32` from-init route was continued through a stable fixed-64 polish.
The best validated point is now:

```text
/tmp/gpurec_hogenom_specieswise_lbfgsb_high64_clean_lr0125_h5_from_step470/checkpoints/best.pt
step=471
fixed64 NLL=526823.9375
fixed64 projected_grad_inf=5.367127895355225
fixed128 NLL=526824.0
fixed128 projected_grad_inf=5.643856048583984
```

This supersedes the older `526887.6875` fixed128 checkpoint and the later
`526824.0625` step-469 checkpoint. The true objective improvement is small
(`0.0625` bits from step 469 to step 471), but it validated at fixed 128.

Two nearby points are useful for interpreting the branch:

- Step 469 from
  `/tmp/gpurec_hogenom_specieswise_high64_polish_continue/checkpoints/latest.pt`
  validated at fixed128 `526824.0625` with projected gradient about `7.73`.
- Step 470 from the competitive-fallback run validated at the same fixed128
  objective `526824.0625`, but reduced projected gradient to about `5.64`.
  The next smaller-radius step reached the validated `526824.0` point above.

The current point is still not KKT converged. Attempts to take another single
step from step 471 with fixed128 `lr=0.0625` and fixed64 `lr=0.0625` both
timed out before producing a history row, so this basin is now dominated by
very slow line-search probes.

When validating these checkpoints, preserve the original rate bounds
(`min_rate=1e-10`, `max_rate=100`). Some early manual final-only checks used
the workflow defaults (`max_rate=2`), which left the NLL unchanged but made the
projected-gradient metric wrong because many valid HOGENOM rates exceed 2.

A short fixed64 projected-SGD probe from step 471 with `lr=1e-4` was cheap and,
with the correct bounds, reduced the fixed128 projected gradient from about
`5.37` to about `0.77` at fixed128 `526824.0`. Continuing projected-SGD with
optimizer state reset between learning-rate changes produced further validated
objective and KKT progress, but uniform projected-SGD alone is slow.

The L-BFGS-B fallback was updated after observing the step-469 behavior:
accepted projected-gradient fallbacks that only make loss-resolution-scale
progress now compete against sign, top-k sign, and coordinate-sign fallbacks
before the step is committed. The alternate fallback probes are capped so the
competition remains bounded on HOGENOM.

## Top-K Pulse Findings

The fastest credible route found after the fixed64 polish is no longer plain
L-BFGS-B. It is:

1. Use projected-SGD repair steps at fixed `64/64/64` to enter the stable
   basin, validating with `final_check_iters=128`.
2. Periodically evaluate a one-shot top-k projected-gradient pulse on the
   largest projected-gradient coordinates.
3. Keep only pulses that validate at fixed128.
4. Repair with small projected-SGD steps only when the repair preserves or
   improves the validated objective. Later in the run, uniform projected-SGD
   can reduce projected gradient slowly while giving back a pulse's objective
   gain.

The best validated point currently found by this route is:

```text
/tmp/gpurec_hogenom_specieswise_coord3141_micro_from_coord3147/candidate_coord3141.pt
fixed128 check: /tmp/gpurec_hogenom_specieswise_truecheck_coord3141_micro_e128
fixed128 NLL=526821.875
fixed128 projected_grad_inf=0.4123495817184448
fixed128 grad_inf=2.125490427017212
```

The most useful recent sequence was:

```text
/tmp/gpurec_hogenom_specieswise_projectedsgd_high64_lr1e4_repair_from_objective779/checkpoints/latest.pt
fixed128 NLL=526822.3125
fixed128 projected_grad_inf=0.556588351726532

/tmp/gpurec_hogenom_specieswise_topk_probe_from_step829/candidate_objective.pt
topk=400
alpha=0.05
fixed128 NLL=526822.125
fixed128 projected_grad_inf=1.7062797546386719

/tmp/gpurec_hogenom_specieswise_projectedsgd_high64_lr1e4_repair_from_objective829/checkpoints/latest.pt
fixed128 NLL=526822.0
fixed128 projected_grad_inf=0.5363166332244873

/tmp/gpurec_hogenom_specieswise_topk_probe_from_step879/candidate_tradeoff.pt
topk=800
alpha=0.005
fixed128 NLL=526821.9375
fixed128 projected_grad_inf=0.5256260633468628

/tmp/gpurec_hogenom_specieswise_kkt_probe_from_topk879/candidate_kkt.pt
topk=100
alpha=0.001
fixed128 NLL=526821.9375
fixed128 projected_grad_inf=0.5234647393226624

/tmp/gpurec_hogenom_specieswise_frontier_grad_probe_from_kkt879/candidate_frontier.pt
topk=2
alpha=0.02
fixed128 NLL=526821.9375
fixed128 projected_grad_inf=0.48702603578567505

/tmp/gpurec_hogenom_specieswise_frontier_grad_probe2_from_frontier879/candidate_frontier.pt
topk=1
alpha=0.02
fixed128 NLL=526821.9375
fixed128 projected_grad_inf=0.4537384510040283

/tmp/gpurec_hogenom_specieswise_greedy_frontier_from_frontier2/candidate_greedy_frontier.pt
fixed128 NLL=526821.9375
fixed128 projected_grad_inf=0.4458876848220825

/tmp/gpurec_hogenom_specieswise_frontier2_objective_candidate/candidate_objective.pt
topk=20
alpha=0.03
fixed128 NLL=526821.875
fixed128 projected_grad_inf=0.7129073143005371

/tmp/gpurec_hogenom_specieswise_greedy_frontier_from_objective875/candidate_greedy_objective.pt
fixed128 NLL=526821.875
fixed128 projected_grad_inf=0.41447359323501587

/tmp/gpurec_hogenom_specieswise_greedy_frontier2_from_objective875/candidate_greedy_objective.pt
fixed128 NLL=526821.875
fixed128 projected_grad_inf=0.41294723749160767

/tmp/gpurec_hogenom_specieswise_coord3147_micro_from_objective875_cycle2/candidate_coord3147.pt
coord=3147
theta_abs_step=0.0002
fixed128 NLL=526821.875
fixed128 projected_grad_inf=0.41251140832901

/tmp/gpurec_hogenom_specieswise_coord3141_micro_from_coord3147/candidate_coord3141.pt
coord=3141
theta_abs_step=0.00015
fixed128 NLL=526821.875
fixed128 projected_grad_inf=0.4123495817184448
```

The step-829 balanced pulse is still useful as a KKT reference:

```text
/tmp/gpurec_hogenom_specieswise_topk_probe_from_step829/candidate_tradeoff.pt
topk=50
alpha=0.02
fixed128 NLL=526822.25
fixed128 projected_grad_inf=0.5116384625434875
```

It had the lowest validated projected gradient in this batch, but the objective
was worse than the objective branch. Its projected-SGD repair ended at
`526822.25` with projected gradient about `0.524`.

Repairing the step-879 pulse with projected-SGD at `lr=1e-4` lowered projected
gradient from about `0.526` to about `0.522`, but the final fixed128 objective
returned to `526822.0`. The saved best checkpoint in that repair is the initial
pre-step pulse itself, so for the best objective use `candidate_tradeoff.pt`,
not the repair's `latest.pt`.

A tiny KKT-only pulse from the step-879 top-k point improved the compromise
without giving back the objective:

```text
/tmp/gpurec_hogenom_specieswise_kkt_probe_from_topk879/candidate_kkt.pt
topk=100
alpha=0.001
fixed128 NLL=526821.9375
fixed128 projected_grad_inf=0.5234647393226624
```

Other tiny KKT probes either gave back one `0.0625`-bit loss quantum or did not
beat this residual. The best one-quantum trade validated at fixed128
`526822.0` with projected gradient `0.5198068618774414`; none reached the
`0.50` projected-gradient threshold.

Follow-up frontier pulses made that finding obsolete. The useful move is to
refresh the gradient and greedily apply very small top-k projected-gradient
pulses, usually with `topk` between `1` and `3`, accepting only candidates that
preserve the current fixed64 objective quantum. This reduced projected gradient
from about `0.523` to `0.446` at fixed128 without changing `526821.9375`.

An objective pulse from the first frontier point (`topk=20`, `alpha=0.03`)
validated at fixed128 `526821.875`, but raised projected gradient to about
`0.713`. Greedy frontier polishing from that lower-objective point repaired it
to fixed128 projected gradient `0.41294723749160767` while preserving
`526821.875`. A fine top-k sweep from that point found no same-objective
top-k-gradient improvement in the tested neighborhood, but single-coordinate
micro probes did continue to improve the residual at the same objective
quantum. The best validated sequence so far is coordinate `3147` with
`theta_abs_step=0.0002`, then coordinate `3141` with
`theta_abs_step=0.00015`, reaching fixed128 projected gradient
`0.4123495817184448`.

A row-coupled probe from the same point did not beat the coordinate-micro
route. The same-row ALCBS D/L pair and D/L/T triplet worsened projected
gradient at the tested radii; POLNS and LACPJ row pairs were nearly neutral.
The only accepted row-probe candidate was a single Actinomycetota-7 L move, and
its fixed64 projected gradient was still higher than the later coordinate 3147
and 3141 micro-steps.

A follow-up coordinate scan from `candidate_coord3141.pt` over the top five
projected-gradient coordinates `[3147, 3148, 2739, 1174, 3141]` and absolute
theta steps `[5e-05, 1e-04, 1.5e-04, 2e-04]` found no same-objective projected
gradient improvement. The largest residuals are now locally coupled: direct
single-coordinate descent on coordinates 3147 and 3148 immediately raises the
projected-gradient infinity norm, and the best near miss in the scan was still
above the base fixed64 projected gradient.

More aggressive pulses can lower NLL but worsen KKT substantially. For example,
`topk=200`, `alpha=0.1` from step 679 validated at fixed128 `526822.625`, but
projected gradient jumped to about `3.80`. It was repairable, but a balanced
pulse is a better speed/conditioning tradeoff. Repeating a fixed
`topk=20`, `alpha=0.01` step without a fresh line search was unstable after the
first step, so top-k pulses should be treated as one-shot, line-searched moves,
not as a fixed-step optimizer.

The current point is still not converged; projected gradient is about `0.41235`,
well above the `1e-3` tolerance and above the immediate target of `0.1`. Fresh
top-k/frontier pulses remain the fastest way found to improve the objective and
KKT residual together. Uniform projected-SGD is useful as a KKT polish, but at
this stage it is too slow to be the main route and can erase objective gains
from a good pulse.

## Reproducible Pulse Benchmark Harness

The ad hoc pulse probes are now covered by
`scripts/benchmark_hogenom_specieswise_pulses.py`. The harness loads a
specieswise checkpoint, evaluates bounded projected-gradient pulse candidates
at a probe solver budget, validates selected candidates at a higher fixed
budget, streams progress to `pulse_benchmark.live.jsonl`, and writes
`pulse_benchmark.csv`, `pulse_benchmark.jsonl`, `summary.json`, and a candidate
checkpoint only when the candidate improves the baseline objective or
same-objective projected-gradient residual.

The current harness makes pulse direction explicit. Top-k and pair probes
default to scaled projected-gradient steps (`theta -= alpha * projected_grad`)
because that matches the historical HOGENOM top-k probes. Single-coordinate
probes default to absolute sign steps because the successful coordinate
micro-polish was recorded as `theta_abs_step`.

Fresh current-code baseline check:

```text
python -m gpurec.cli optimize ... --fixed-iters-e 128 --fixed-iters-pi 128 \
  --neumann-terms 128 --no-adaptive-iters --steps 879 \
  --resume-from /tmp/gpurec_hogenom_specieswise_coord3141_micro_from_coord3147/candidate_coord3141.pt

/tmp/gpurec_hogenom_codex_baseline_coord3141_e128_fixed
fixed128 NLL=526821.875
```

The same command without `--no-adaptive-iters` evaluates the checkpoint at
`526980.0625` bits because the workflow default enables adaptive solver
stopping. HOGENOM validation runs that claim fixed solver budgets must pass
`--no-adaptive-iters` or use a config with `adaptive_iters=false`.

Two follow-up scans from `candidate_coord3141.pt` did not improve the current
validated point:

```text
/tmp/gpurec_hogenom_codex_pair_only_pulse_from_coord3141
pair-only top-5 residual coordinate scan
best fixed128 candidate: pair 391:L + 1047:D, alpha=1e-4
fixed128 NLL=526821.875
fixed128 projected_grad_inf=0.41236478090286255
baseline fixed128 projected_grad_inf=0.41234683990478516
saved_checkpoint=null

/tmp/gpurec_hogenom_codex_topk_objective_from_coord3141
top-k objective scan, topk in {10,20,50,100,200,400,800}
alpha in {0.001,0.002,0.005,0.01,0.02,0.03,0.05}
best fixed128 candidate: top10 alpha=0.001
fixed128 NLL=526821.875
fixed128 projected_grad_inf=0.4241284430027008
saved_checkpoint=null
```

These results leave the best validated point unchanged at fixed128
`526821.875` bits. The useful next search should avoid broad top-k pulses from
this checkpoint and focus on either a more structured coupled-coordinate model
or a different basin-entry schedule before the final coordinate micro-polish.

## End-to-End Route Benchmark

The accepted route is now summarized by
`scripts/benchmark_hogenom_specieswise_e2e.py`. It reads the local run
directories and candidate checkpoints, writes per-stage CSV/JSON evidence, and
separates raw low-budget objectives from fixed-budget validated objectives.

Full-run exact-tail summary:

```text
python scripts/benchmark_hogenom_specieswise_e2e.py \
  --replace-stage topk_pulse_step679=/tmp/gpurec_hogenom_codex_delta_replay_step679 \
  --replace-stage topk_balanced_step729=/tmp/gpurec_hogenom_codex_delta_replay_step729 \
  --replace-stage topk_objective_step779=/tmp/gpurec_hogenom_codex_delta_replay_step779 \
  --replace-stage topk_objective_step829=/tmp/gpurec_hogenom_codex_delta_replay_step829 \
  --tail-replay-dir /tmp/gpurec_hogenom_codex_tail_delta_replay_from_step879 \
  --out-dir /tmp/gpurec_hogenom_codex_e2e_route_summary_v8_exact_tail

target_nll_bits=526821.875
target_validation_iters=128
target_stage=tail_replay_step879
validated_best_nll_bits=526821.875
validated_best_projected_grad_inf=0.41235461831092834
known_elapsed_s=6401.800589857332
known_elapsed_h=1.7782779416270367
unknown_elapsed_stage_count=0
```

This full-run total charges every historical run directory by its recorded
summary elapsed, including intermediate fixed128 final-check rows and
post-best steps that were later abandoned. For a rerun of the accepted route,
the tighter optimization-only accounting is:

```text
python scripts/benchmark_hogenom_specieswise_e2e.py \
  --replace-stage topk_pulse_step679=/tmp/gpurec_hogenom_codex_delta_replay_step679 \
  --replace-stage topk_balanced_step729=/tmp/gpurec_hogenom_codex_delta_replay_step729 \
  --replace-stage topk_objective_step779=/tmp/gpurec_hogenom_codex_delta_replay_step779 \
  --replace-stage topk_objective_step829=/tmp/gpurec_hogenom_codex_delta_replay_step829 \
  --tail-replay-dir /tmp/gpurec_hogenom_codex_tail_delta_replay_from_step879 \
  --effective-resume-elapsed \
  --out-dir /tmp/gpurec_hogenom_codex_e2e_route_summary_v9_effective_exact_tail

target_nll_bits=526821.875
target_validation_iters=128
target_stage=tail_replay_step879
validated_best_nll_bits=526821.875
validated_best_projected_grad_inf=0.41235461831092834
known_elapsed_s=6157.094623648678
known_elapsed_h=1.7103040621246328
unknown_elapsed_stage_count=0
```

The `--effective-resume-elapsed` total charges each run directory only through
the checkpoint consumed by the next stage. This removes `244.705966208654s`
from the full-run table, mostly from intermediate final-check rows and from the
bad excursions after the step594 and step649 best checkpoints. The final
fixed128 target validation is still charged in the tail replay.

If the acceptance target is relaxed to within one bit of the best validated
NLL, the target threshold is `526822.875`. Under that criterion the route can
stop immediately after the accepted step679 top-k delta and a fixed128
validation:

```text
python scripts/benchmark_hogenom_specieswise_e2e.py \
  --replace-stage topk_pulse_step679=/tmp/gpurec_hogenom_codex_delta_replay_step679 \
  --truncate-after-stage topk_pulse_step679 \
  --append-stage fixed128_validation_step679=/tmp/gpurec_hogenom_codex_validate_delta_step679_e128 \
  --effective-resume-elapsed \
  --target-nll-bits 526822.875 \
  --out-dir /tmp/gpurec_hogenom_codex_e2e_route_summary_v12_relaxed_step679

target_nll_bits=526822.875
target_validation_iters=128
target_stage=fixed128_validation_step679
validated_best_nll_bits=526822.625
validated_best_projected_grad_inf=3.7994980812072754
known_elapsed_s=4998.935334998765
known_elapsed_h=1.388593148610768
unknown_elapsed_stage_count=0
```

The immediately preceding fixed128 final check, before the step679 top-k
delta, is `526823.125`, so it is outside the relaxed threshold. The step679
route saves `1158.159288649913s` (`19.302654810831882min`) relative to the
effective exact-best route while staying inside the accepted NLL window.

The route's raw low-budget best is `526819.375` at the fixed32 stage
`lbfgsb32_continue_361_433`, but that is not a validated target because the
same basin later evaluates near `526836` at fixed128. The route benchmark
therefore treats fixed128 validation as the target evidence.

This zero-unknown accounting uses exact-delta replays for manual pulse
checkpoints. It measures the cost to apply and evaluate the accepted deltas, not
the original manual search cost that found them. The plain provenance route,
without replacement replay directories, still reports the step-679, step-729,
step-779, step-829, and post-step879 candidate checkpoints as unknown elapsed
stages.

Exact-delta replay evidence:

```text
/tmp/gpurec_hogenom_codex_delta_replay_step679
elapsed_s=13.224594317027368
fixed64 NLL=526822.625
fixed64 projected_grad_inf=3.8000988960266113

/tmp/gpurec_hogenom_codex_delta_replay_step729
elapsed_s=13.270848230051342
fixed64 NLL=526822.5
fixed64 projected_grad_inf=0.5797186493873596

/tmp/gpurec_hogenom_codex_delta_replay_step779
elapsed_s=13.282168215024285
fixed64 NLL=526822.375
fixed64 projected_grad_inf=0.6182775497436523

/tmp/gpurec_hogenom_codex_delta_replay_step829
elapsed_s=13.321792050963268
fixed64 NLL=526822.125
fixed64 projected_grad_inf=1.7043495178222656

/tmp/gpurec_hogenom_codex_tail_delta_replay_from_step879
elapsed_s=60.668528918002266
fixed64 NLL=526821.875
fixed64 projected_grad_inf=0.41235682368278503
fixed128 NLL=526821.875
fixed128 projected_grad_inf=0.41235461831092834
```

The first dynamic replay used unit sign steps for top-k pulses and did not
reproduce the historical tail:

```text
/tmp/gpurec_hogenom_codex_tail_replay_from_step879
elapsed_s=70.95032295602141
fixed128 NLL=526822.625
fixed128 projected_grad_inf=6.810783386230469
```

That mismatch was a replay bug: historical top-k pulses used scaled projected
gradients, not unit signs. With projected-gradient top-k steps and sign
coordinate micro-steps, the dynamic schedule reaches the same best likelihood
quantum without exact checkpoint deltas:

```text
/tmp/gpurec_hogenom_codex_tail_dynamic_pg_coordsign_replay_from_step879
elapsed_s=70.85208377701929
fixed64 NLL=526821.875
fixed64 projected_grad_inf=0.41387057304382324
fixed128 NLL=526821.875
fixed128 projected_grad_inf=0.4138827919960022
```

Replacing only the post-step879 tail with this dynamic schedule gives a
full-run measured route at `6411.984144716349s` (`1.7811067068656525h`) to the
same fixed128 target. With effective resume accounting, the dynamic-tail route
is `6167.278178507695s` (`1.7131328273632487h`). The exact-delta tail remains
the fastest measured replay and has the slightly better residual, but the
dynamic tail is the reproducible optimizer schedule for that final pulse block.

Largest known stage costs:

```text
lbfgsb32_continue_100_219   895.27s
lbfgsb32_init_000_099       746.86s
lbfgsb32_continue_284_360   579.20s
lbfgsb32_continue_361_433   543.57s
lbfgsb32_continue_220_283   474.48s
lbfgsb64_polish_450_469     318.73s
projected_sgd64 repair blocks, 50 steps each, about 282-284s
```

The main route cost buckets in the effective exact-tail timing table are:

```text
fixed32 L-BFGS-B basin entry          3230.68s
fixed64 L-BFGS-B polish                623.11s
projected-SGD repair / polish blocks  2189.54s
exact pulse delta replays              113.77s
```

The most important speed target is still the basin-entry phase, not final
checkpoint validation or the pulse application itself. A faster route needs to
reduce or replace the first `3231s` of fixed32 L-BFGS-B work before the high64
polish, or replace the repeated 50-step projected-SGD repair blocks with a
stronger validated repair operator. Exact pulse application is now measured and
is a small part of the end-to-end wall time.

### Repair Speed Trials

Several attempted replacements for the final step829 repair block did not beat
the accepted route:

```text
/tmp/gpurec_hogenom_codex_repair_step829_fixed32_50
50 projected-SGD steps at fixed32 instead of fixed64
elapsed_s=169.97366390703246
configured fixed32 NLL=526823.4375
fixed128 final-check NLL=526825.8125
fixed32 projected_grad_inf=0.5362688302993774

/tmp/gpurec_hogenom_codex_pulse_from_step845_best
focused pulse scan from the step845 best checkpoint
best fixed128 candidate: top800 alpha=0.001 sign step
fixed128 NLL=526822.0
fixed128 projected_grad_inf=0.5379502773284912

/tmp/gpurec_hogenom_codex_tail_dynamic_from_step845_shortcut
dynamic tail from the step845 shortcut candidate
fixed128 NLL=527187.5
fixed128 projected_grad_inf=36.19819259643555

/tmp/gpurec_hogenom_codex_tail_dynamic_from_step845_best
dynamic tail directly from the step845 best checkpoint
fixed128 NLL=526834.0625
fixed128 projected_grad_inf=37.8035774230957

/tmp/gpurec_hogenom_codex_repair_step829_fixed64_to860
shortened fixed64 repair through step860
elapsed_s=177.76188709499547
fixed64 NLL=526822.0625
fixed64 projected_grad_inf=0.6477079391479492

/tmp/gpurec_hogenom_codex_tail_dynamic_from_step860_repair
dynamic tail after the step860 cut
fixed128 NLL=526836.6875
fixed128 projected_grad_inf=37.838687896728516

/tmp/gpurec_hogenom_codex_repair_step829_fixed64_to870
shortened fixed64 repair through step870
elapsed_s=231.8711816170253
fixed64 NLL=526822.0625
fixed64 projected_grad_inf=0.5378491878509521

/tmp/gpurec_hogenom_codex_tail_dynamic_from_step870_repair
dynamic tail after the step870 cut
fixed128 NLL=526821.9375
fixed128 projected_grad_inf=0.4140489101409912
```

The step870 cut is the closest miss, but it is still one `0.0625`-bit objective
quantum worse than the best validated likelihood. For the current schedule, the
late fixed64 repair cannot be shortened before step879 without losing the best
known NLL.

### Basin Switch Trials

The largest single route cost is the fixed32 L-BFGS-B basin-entry phase. Two
early-switch trials tested whether fixed64 polishing could replace the end of
that phase:

```text
/tmp/gpurec_hogenom_codex_high64_from_step360
resume: /tmp/gpurec_hogenom_specieswise_lbfgsb_high32_from_init_continue3/checkpoints/best_reset_for_checks.pt
20 fixed64 L-BFGS-B steps, matching the accepted high64 polish settings
elapsed_s=293.7307326599839
fixed64 NLL=526881.125
fixed128 final-check NLL=526881.125
fixed64 projected_grad_inf=8.164962768554688

/tmp/gpurec_hogenom_codex_high64_from_step419
resume: /tmp/gpurec_hogenom_codex_step419_reset_for_high64.pt
16 fixed64 L-BFGS-B steps, matching the accepted high64 polish settings
elapsed_s=247.16399874101626
fixed64 NLL=526838.375
fixed128 final-check NLL=526838.5
fixed64 projected_grad_inf=3.8232479095458984
```

Both are much worse than the accepted path that switches after the step433
fixed32 checkpoint and reaches `526830.5` after the first fixed64 polish block.
The step361-433 fixed32 segment is therefore still part of the fastest
validated route despite its high wall time.

### Short Basin-Entry Check

A five-step fixed32 L-BFGS-B smoke from initialization compared fixed versus
adaptive solver stopping:

```text
/tmp/gpurec_hogenom_codex_basin_fixed32_5step
elapsed_s=43.0461
fixed32 NLL=563782.0
fixed128 check NLL=563782.0
fixed128 projected_grad_inf=33.82101058959961

/tmp/gpurec_hogenom_codex_basin_adaptive32_5step
elapsed_s=35.7036
adaptive32 NLL=563987.3125
fixed128 check NLL=563965.6875
fixed128 projected_grad_inf=33.151756286621094
```

Adaptive stopping saved about 17% wall time over five early L-BFGS-B steps, but
was about 184 validated bits worse at fixed128. That is not enough evidence to
replace the fixed32 basin-entry route. A larger adaptive early-stage trial
would need periodic fixed128 checks and a catch-up criterion before it can be
treated as a faster route.

## Sub-5-Minute Trials

The relaxed acceptance threshold is now within one bit of the best fixed128
checkpoint:

```text
best fixed128 NLL=526821.875
accepted threshold=526822.875
```

The fastest fully measured route that reaches this relaxed threshold is still
the step679 relaxed route:

```text
/tmp/gpurec_hogenom_codex_e2e_route_summary_v12_relaxed_step679/summary.json
target stage=fixed128_validation_step679
final fixed128 NLL=526822.625
elapsed_s=4998.935334998765
```

Several sub-5-minute candidates are not valid end-to-end routes because they
start from historical checkpoints. The fastest fixed128 checks around the target
are validation-only runs, for example:

```text
/tmp/gpurec_hogenom_specieswise_truecheck_topk_candidate_e128
resume_from=/tmp/gpurec_hogenom_specieswise_topk_probe_from_step679/candidate.pt
fixed128 NLL=526822.625
elapsed_s=12.6518
```

A scan of local HOGENOM workflow artifacts found no full specieswise
from-scratch run under 300 seconds near the relaxed threshold. The best
from-scratch run under 300 seconds was only a five-step basin smoke:

```text
/tmp/gpurec_hogenom_codex_basin_fixed32_5step
elapsed_s=43.0461
fixed128 NLL=563782.0
```

A counts-derived specieswise initializer from AleRax
`totalSpeciesEventCounts.txt` was tested as a clean-start shortcut. The best
formula in the fixed6 probe used:

```text
D = 2.0 * (duplications + 0.1) / (copies + 0.1)
L = 2.0 * (losses + 0.1) / (copies + 0.1)
T = 0.5 * (transfers + 0.1) / (copies + 0.1)
floor=1e-5
```

It made fixed6 L-BFGS-B much faster, but the cheap objective was not stable:

```text
/tmp/gpurec_hogenom_codex_counts_init_lbfgsb6_lr1_100
external wall including model build and counts init=295.01739156700205
fixed6 NLL=525281.75

/tmp/gpurec_hogenom_codex_validate_counts_init_lr1_step99_e128
fixed128 NLL=533618.5625
fixed128 projected_grad_inf=39.863136291503906
```

The same initializer with higher solver budgets followed the true objective but
did not get close enough under five minutes:

```text
/tmp/gpurec_hogenom_codex_counts_init_lbfgsb32_lr1_35
external wall including model build and counts init=286.38200562499696
fixed32 NLL=531178.5625

/tmp/gpurec_hogenom_codex_counts_init_lbfgsb16_lr1_60
stopped after step25 during the trial
fixed16 NLL at step25=531455.875
```

A high32 correction from the fast fixed6 checkpoint was also not promising:

```text
/tmp/gpurec_hogenom_codex_counts_init_lr1_step99_high32_correct10
resume_from=/tmp/gpurec_hogenom_codex_counts_init_lbfgsb6_lr1_100/checkpoints/best.pt
fixed32 NLL after three correction steps=530276.75
```

After the counts initializer was combined with staged Adagrad accumulator
resets, a clean route did satisfy the relaxed threshold under five minutes:

```text
/tmp/gpurec_hogenom_counts_adagrad_route_clean/route_summary.json
total wall=266.0934499010327s
target fixed128 NLL=526822.875
final fixed128 NLL=526791.3125
```

The route is:

```text
stage1: counts-derived specieswise init, fixed16 Adagrad lr=1.0 to step40
stage2: reset Adagrad accumulator, fixed16 Adagrad lr=0.5 to step100
stage3: reset Adagrad accumulator, fixed32 Adagrad lr=0.2 to step110
final check: fixed128 validation at step110
```

The benchmark reproducer is
`scripts/benchmark_hogenom_counts_adagrad_route.py`. The final fixed32 loss
was `526791.25`; the fixed128 final check was effectively identical at
`526791.3125`, unlike the earlier fixed6 and late fixed16 surrogate tails.

### Non-AleRax-Counts Schilling L-BFGS-B Screen

The counts-derived route above uses AleRax reconciliation summaries and is
therefore not a clean gpurec-only optimizer route. After adding the Schilling
L-BFGS-B conformance kernels in `gpurec/optimization/lbfgsb_schilling.py`, the
active `lbfgsb` optimizer was screened again from the default uniform
specieswise initialization, without event-count initialization.

Short fixed16 screens from the default initialization:

```text
/tmp/gpurec_hogenom_schilling_lbfgsb16_default_lr0p1_10_screen
fixed16 NLL=561376.1875
elapsed_s=49.1071

/tmp/gpurec_hogenom_schilling_lbfgsb16_default_lr0p5_10_screen
fixed16 NLL=547606.625
elapsed_s=50.6077

/tmp/gpurec_hogenom_schilling_lbfgsb16_default_lr1p0_10_screen
fixed16 NLL=547519.6875
elapsed_s=51.7160

/tmp/gpurec_hogenom_schilling_lbfgsb16_default_lr2p0_10_screen
fixed16 NLL=564556.1875
elapsed_s=52.1035
```

Extending the best short screen (`lbfgs-lr=1.0`) to step60:

```text
/tmp/gpurec_hogenom_schilling_lbfgsb16_default_lr1p0_to60_screen
resume_from=/tmp/gpurec_hogenom_schilling_lbfgsb16_default_lr1p0_10_screen/checkpoints/latest.pt
fixed16 NLL=529928.625
elapsed_s=235.3017

/tmp/gpurec_hogenom_validate_schilling_lbfgsb16_default_lr1p0_step60_e128
fixed128 NLL=529953.5
fixed128 projected_grad_inf=36.43765640258789
```

The combined fixed16 screen plus validation is already about five minutes and
still more than 3000 bits above the relaxed fixed128 target (`526822.875`), so
the Schilling-correct L-BFGS-B path is not enough by itself to make a clean
sub-300-second route from the default initialization.
