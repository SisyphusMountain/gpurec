# Coleman optimization campaign — 6 September 2026

Coordinator: root. Experiment workers: at most two GPT-5.6 Sol agents,
`em_acceleration` and `hierarchical_geometry`. User authorized experimentation
and implementation to improve on the approximately 520-second Coleman fit.

## Reference and acceptance criteria

- All 5,124 Coleman families, including COG3676_X; one H100 NVL.
- Historical best: 520.5 s, NLL 9,049,362.363 bits, 5,124 freeze-time
  certificates, 47 Newton iterations. Later current-HVP run: 526.6 s.
- Current source snapshot includes the user's uncommitted changes at commit
  `6ca3e3b3bd763bd94082fedaa6e3007e75b8aab7`.
- Preserve original data, likelihood, rate bounds [1e-6, 2], projected
  log-rate gradient tolerance 1e-3, solver precision, and pruning settings.
- The historical certificate uses a pruned float32 gradient and cached
  freeze-time measurements. Compare under that same definition; separately
  audit candidate and baseline at matched parameters/settings if needed.
  Do not present this as an unpruned mathematical stationarity certificate.
- Count every initialization, gradient, count-extraction, rejected-trial,
  Hessian, rebuild, and certification cost. Report actual time and
  clade-weighted work. Fit quality includes total and per-family NLL.
- No target-family fitted rates/Hessians used as initialization or inputs to
  candidate algorithms; saved optima are evaluation references only.

## Evidence recovered

Read `docs/genewise_h100_runtime.md`, especially rounds five and six. The
mathematical report's claim that EM remained unmeasured is stale: the recovered
Claude scratchpad contains count validation, missing-information fractions,
and a 20-pass EM/SQUAREM experiment. The count fold matches production gradients
within numerical error; counts are positive. Near optimum the missing fraction
is median 0.837 / p90 0.915. Plain EM and the existing SQUAREM experiment have
poor tails. One EM update from the common start is promising: median log-rate
distance 1.90, versus 2.48 after three Adam updates.

Recovered source directory:
`/tmp/claude-1000/-home-enzo-Documents-git-gpurec-gpurec/8392fbe3-5570-4d60-9656-16e4db97a7a9/scratchpad/`.

## Work allocation

1. EM worker (`em/`): validate/correct bounded M-step and cheap count extraction;
   test one to three EM warm-up updates, followed by production BFGS with
   count-informed or secant curvature. Accelerated EM only if evidence improves.
2. Geometry worker (`geometry/`): CPU screening of hierarchical logits
   u=log2((D+T)/(1+L)), v=log2(T/D), w=log2(L), with their diagonal complete-data
   curvature; then GPU comparison of affordable quasi-Newton variants.
3. Coordinator: baseline snapshot and H100 reproduction, mathematical review,
   scheduling, promotion/rejection decisions, integration, full-dataset checks.

Workers own only their respective experiment directories initially. Preserve
all existing workspace changes. No worker submits cluster jobs or starts a
second agent. Coordinator handles any production edits.

## Experiment sequence and resources

- First gate: algebra/count/KKT checks and matched 200-family comparisons.
- Second gate: 500-family validation, checking clade-weighted cost and difficult
  families, not merely median distance or frozen-family count.
- Third gate: all 5,124 families on H100, baseline and candidate on the same
  allocation when possible. Repeat promising results to quantify timing spread.
- Integrate only justified improvements; preserve negative experiment records.
- Initial local RTX4090 lease: EM worker. Geometry worker begins CPU-only.
  Local GPU use is serialized by coordinator messages. H100 comparisons run in
  an isolated source copy, avoiding unrelated changes to the main cluster copy.

## Status

- Both Sol workers launched.
- Cluster login works; no existing jobs owned by the user at discovery.
- H100 baseline staged at
  `/sps/biometr/emarsot/gpurec-sol-20260906-Ulx6CJ`; Slurm job `58003570`
  requests one H100, 32 CPU threads, and runs two full baselines. Queued for
  resources at submission (all H100 GPUs allocated).
- Bounded EM update corrected and independently checked by the worker: three
  saved opt-point families changed; KKT residual below 2.3e-16, including 10,000
  randomized boundary cases. Common-start EM update is unchanged.
- Local RTX4090 lease granted to EM worker for smoke and first 200-family runs.
- User pointed out available interactive H100s. `ccwgislurm0200` had three
  unallocated H100s and CPU load 0; full baseline job `58003625` started there
  at 21:00:40 CEST with 32 CPU threads. Canceled only our superseded pending
  batch job `58003570`. Candidate comparisons will use the same interactive
  node, with its contemporaneous baseline as the timing reference.
- Local 200-family baseline: 44.02 s including the prototype's extra parse;
  16.977 actual resident-model gradient/clade equivalents. EM2 prototype:
  35.82 s, 13.553 equivalents, 200 certificates. Integrated driver: 34.54 s,
  200 certificates, NLL 613261.7982 bits. Per-family basin changes were tracked.
- Local 500-family gate: baseline 108.33 s / 17.273 equivalents; EM2 prototype
  91.51 s / 13.660 equivalents. Both certify 500 families. Fresh forward NLL
  difference +0.02757 bits; four families differ by more than 0.01 bits.
- Promoted two bounded EM steps with endpoint complete-information curvature,
  scaled by the first EM secant and updated by safeguarded BFGS. Counts share
  the existing adjoint pass. The integrated version reuses the initial model;
  default Adam remains unchanged. New API and M-step tests pass.
- Hierarchical-coordinate-only variants rejected: weaker likelihood/cost
  tradeoff than EM and a stalled tail without exact Hessian refresh.
- First full interactive baseline completed: 634.016 s, NLL 9049362.3797,
  5124 certificates, 48 Newton steps. Startup/Adam cost was unusually high
  (147.55 s Adam versus about 70 s historically); Newton gradients 383.55 s
  remain close to the historical 379 s. This cold result is not a fair sole
  denominator for a warm speedup claim.
- Paired job `58006123` started 21:19:31 CEST on the same H100 UUID
  `GPU-8a9add16-cb62-9282-f039-9213619d52fa`. Isolated integrated snapshot:
  `/sps/biometr/emarsot/gpurec-sol-em2-20260906-NVUB8w`. Run order: EM2 A,
  Adam B, EM2 B, fresh matched likelihood/gradient audit. Every optimizer
  gradient records resident family/clade count; Hessians retain separate timing.
- During the paired run, EM worker tests exactly one additional EM3 variant
  on 500 local families; geometry worker independently reviews the math.
- Full EM2 A completed in 396.305 s, NLL 9049360.7436, all 5124 certified,
  13.403 actual gradient/clade equivalents and 23.590 GiB peak allocation.
- EM3 improved the 500-family prototype to 87.206 s / 13.299 equivalents.
  Integrated EM3 passed at 81.422 s / 13.343 equivalents, all 500 certified.
  Promoted EM3 to two full trials in isolated snapshot
  `/sps/biometr/emarsot/gpurec-sol-em3-20260906-mBtGwJ`, job `58006172`.
  It depends on completion/cancellation of `58006123`. A coordinator watcher
  will cancel only our old job after the completed Adam B tensor is saved,
  replacing the redundant EM2 repeat with EM3 A/B. The new job audits both
  EM variants and measures same-theta numerical noise plus EM3 repeatability.
- Integrated options: `fit_dtl(..., genewise_warmup_method="em",
  genewise_em_steps=2)`; two or three steps supported. Adam remains default.
  Final focused suite currently passes 91 tests (CPU and CUDA), including
  real EM2/EM3 model-reuse tests and fixed/trainable receiver gradient checks.
- Warmed Adam B completed: 512.718 s, NLL 9049362.3770, all 5124 certified,
  17.041 gradient/clade equivalents. EM2 saves 22.7% wall time and 21.3% gradient
  work against this same-card control. The watcher canceled only our superseded
  job `58006123` after its checkpoint; EM3 job `58006172` started at 21:35 CEST
  on the same GPU UUID. Both completed EM2/Adam artifacts are saved locally.
- EM3 A completed at 398.290 s, NLL 9049360.6102, 5124 certificates,
  13.466 gradient/clade equivalents. Its extra warm-up offsets the saved Newton
  work at full scale; 500-family speed advantage did not carry. EM3 B continues.
  Because the 2-second EM2/EM3 gap is small, final job `58006381` repeats EM2
  and audits both EM2 fits, dependent on success of `58006172`. No further
  optimizer variants planned: finish with two full repeats per EM variant,
  the warmed Adam control, and matched quality/noise measurements.
- EM3 B finished 396.401 s (two-run mean 397.345), all 5124 freeze-time
  certificates. Fresh audit: NLL -1.76365 bits versus Adam; 14 families worse
  and 12 better by >0.01 bit. Fresh Pg counts 5052/5052, versus 5050 on an
  identical-Adam-theta repeat. Maximum repeated Pg change 0.001292 exceeds
  the stopping threshold; do not call cached certification strict cold
  stationarity. EM3 A/B fresh NLL differ only 0.00121 bits total.
- EM2 A fresh audit: NLL -1.63451 bits; 13 worse/12 better families above
  0.01 bit; fresh Pg counts Adam5058/EM25053. Final EM2 B job `58006381`
  started 21:55 CEST, followed by its own repeated-fit/noise audit.
- EM2 B finished 403.470 s, all 5124 certified, NLL 9049360.7546,
  13.398 gradient/clade equivalents. EM2 mean 399.888 s, EM3 mean 397.345 s:
  both about 400 s, with only 0.6% mean separation and two samples per variant.
  Final EM2 repeated-fit/common-gradient audit is running; no fitting jobs remain.
- Campaign complete: job `58006381` finished successfully; no user-owned jobs
  remain in the queue. EM2 repeat audit confirms -1.63089 bits versus Adam and
  only 0.003619-bit total / 0.001310-bit maximum-family difference between the
  two EM2 fits. Same-Adam-theta repeated Pg changes reach 0.001515. All source,
  scripts, test results, fitted tensors, job logs, and quality vectors are saved.
  Both EM step counts remain opt-in; defaults and the original stopping rule
  were not changed. See REPORT.md for the final evidence and qualifications.
- User correction reopens a distinct follow-up: the earlier hierarchical
  experiment started after Adam and does not rule out an EM-plus-hierarchical
  hybrid. Same two Sol workers are now preparing a shared-EM-endpoint native
  versus hierarchical comparison, including coordinate-consistent curvature,
  coupled bounds, and the production exact-Hessian schedule. The original EM
  campaign results stand; only the overbroad closure is withdrawn. See
  `hybrid/PLAN.md` for the new protocol and validation gates.
