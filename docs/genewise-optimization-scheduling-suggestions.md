# Genewise Optimization Scheduling Suggestions

Date: 2026-05-24

This note summarizes proposed systematic approaches for making complete
HOGENOM and `test_trees_1000` genewise optimization faster without accepting
wrong convergence.

## Current Accepted Baselines

The current best accepted full-dataset runs are:

- HOGENOM, 1055 families:
  - run: `/tmp/gpurec_hogenom_loss003_current`
  - elapsed: `74.03s`
  - final NLL: `607282.875`
  - final 32-pass check: ok, absolute loss delta `1.125`
  - shape: one resident batch, 47 warmup rows, 6 full rows

- `test_trees_1000`, 1000 families:
  - run: `/tmp/gpurec_tt1000_loss003_current`
  - elapsed: `245.62s`
  - final NLL: `1755785.375`
  - final 32-pass check: ok, absolute loss delta `0.0`
  - shape: 13 resident batches, all optimized in warmup stage, final check
    costs about `21s`

Several faster HOGENOM attempts were rejected because they reached worse final
likelihoods. For example, late warmup line-search caps produced `62.70s` to
`69.94s` runs, but final NLLs were `607677.75` and `607440.0`, both worse than
the accepted `607282.875`.

## Main Lesson

The expensive work is not obviously disposable. HOGENOM warmup line-search
probes are costly, but fixed caps or disabling warmup line search changes the
optimization path enough to converge to worse likelihoods. Future scheduling
should therefore be adaptive and evidence-based, not based on a fixed shortcut
such as "only 4 probes" or "always cheap Pi/Neumann for N steps".

## Proposed Strategy

### 1. Collect Per-Family Line-Search Telemetry

Add telemetry for each family row during Hessian-SGD line search:

- accepted or rejected at each step
- final accepted `alpha`
- number of line-search probes needed
- whether the projected Newton step was accepted
- whether fallback projected-gradient line search was needed
- whether post-step loss filtering rejected the row
- repeated hard-row counters across optimizer iterations

This should be collected per family, not just per batch. It is the cleanest
signal for which families are slowing a batch down.

### 2. Rebatch By Observed Optimization Difficulty

Use the line-search telemetry to split families into groups such as:

- fast/easy: accepts full or large Newton steps consistently
- medium: needs a few probes but keeps improving
- hard: repeatedly requires many probes or is loss-rejected
- stuck/boundary: projected gradient is small or all free coordinates are
  clamped

Then replan resident batches so easy families are not repeatedly evaluated
together with hard families. This is different from the current rejected
adaptive rebatch attempt, which used likelihood patience rather than observed
line-search difficulty.

Important implementation caveat: calling `replan_resident_batches()` with a
sorted list is probably not enough. The Rust `depth_first_fit` planner sorts by
schedule depth and clade count internally, so difficulty order can be lost.
Proper support likely needs difficulty bins around the existing depth-first
packing: plan easy, medium, hard, and stuck groups separately, while keeping the
current clade/depth packing inside each group.

### 3. Store Per-Family Line-Search State

Keep a small state vector per family:

- last accepted `alpha`
- recent median accepted `alpha`
- recent probe count
- recent rejection count
- recent cheap/full disagreement, if available

Use this state to initialize the next line search. For example, a family that
has repeatedly accepted only `alpha=0.125` should not start every future search
at `alpha=1.0`. Easy families can still start at `1.0`.

This is more precise than globally reducing the maximum number of probes. It
avoids wasting probes on known hard rows while preserving aggressive steps for
families that can use them.

### 4. Stop Probing When Marginal Acceptance Is Low

During batched line search, track the fraction of additional rows accepted at
each probe. Stop the probe loop early when extra probes accept too few new
families, for example:

- probe 1 accepts many rows: continue
- probe 2 accepts more rows: continue
- probe 5 accepts less than 1% new rows: stop and leave remaining rows
  unchanged

This is safer than a fixed cap because it adapts to the current batch. If a
probe is still accepting many rows, keep going. If the remaining rows are hard,
do not force the whole batch through low-value probes.

### 5. Adaptive Pi/Neumann Fidelity By Agreement

Use cheap Pi/Neumann settings only while they agree with the canonical solver on
the decisions that matter.

Possible schedule:

1. Propose steps with cheap settings, such as Pi/Neumann `4/4` or `6/6`.
2. Periodically evaluate the same theta with canonical `16/16`.
3. Compare:
   - loss ordering
   - accepted/rejected rows
   - gradient direction or cosine similarity
   - projected-gradient infinity norm
4. Escalate a batch/family if cheap and canonical decisions disagree.
5. Use the final `32/32` check only as validation, not as the main optimizer
   signal.

The key rule is: cheap fidelity is allowed only while it predicts the same
optimization choices as full fidelity.

### 6. Bias-Corrected Cheap Objective

Track the bias between cheap and full objectives:

```text
bias = full_loss_16_16 - cheap_loss_6_6
```

If the bias is stable for a batch or family, the cheap objective can guide
line-search and plateau checks after adding the estimated bias. If the bias
changes rapidly, escalate fidelity.

Useful telemetry:

- per-family cheap loss
- per-family full loss
- bias mean and variance
- bias drift over optimizer rows
- whether cheap and full objective deltas have the same sign

### 7. Per-Family Pi Difficulty, Not Batch-Level Pi Stats

The current solver stats are not safe for individual family rebatching. In a
multi-family resident batch, Pi convergence is tracked per wave and uses a max
over the wave, so one hard family can make other families in that wave appear
hard.

A safe version would collect per-family Pi difficulty during the normal
multi-family pass:

- per-family wave iteration sum
- per-family wave iteration max
- per-family hit-cap count
- per-family convergence delta summaries

This should be telemetry-only at first. Use it to form difficulty bins, not to
lower solver caps or skip correctness checks.

### 8. TT1000 Final-Check Policy

For `test_trees_1000`, the final 32-pass check costs about `21s`, and the
accepted run had exactly `0.0` loss delta. For production speed, consider an
option to avoid always paying this cost:

- keep the full final check for benchmark/correctness runs
- add a configurable sampled or cached final check for production runs
- allow skipping full final check only when the run used canonical solver
  settings or when repeated prior checks on the same config had zero delta

This should be opt-in because the final check is the strongest correctness
evidence.

### 9. Recommended Experiment Order

1. Add per-family line-search telemetry without changing optimization behavior.
2. Run HOGENOM and `test_trees_1000` baselines with telemetry enabled and verify
   final NLLs match accepted runs.
3. Analyze which families repeatedly require many probes or rejections.
4. Implement difficulty-bin rebatching based on observed line-search hardness.
5. Benchmark on a small subset first, then full HOGENOM and full
   `test_trees_1000`.
6. Add per-family initial alpha reuse and compare against rebatching alone.
7. Only then test adaptive Pi/Neumann fidelity with full-vs-cheap agreement
   checks.

## Acceptance Criteria

A proposed optimization should only be accepted if it satisfies all of these:

- lower end-to-end time on the full dataset
- final NLL no worse than the accepted run within normal run-to-run noise
- final check remains ok
- final-check loss delta remains in the accepted range
- no increase in nonfinite failures
- no hidden dependence on a narrow subset benchmark

For HOGENOM, use `/tmp/gpurec_hogenom_loss003_current` as the current reference:

- elapsed `74.03s`
- final NLL `607282.875`
- final check ok, absolute loss delta `1.125`

For `test_trees_1000`, use `/tmp/gpurec_tt1000_loss003_current` as the current
reference:

- elapsed `245.62s`
- final NLL `1755785.375`
- final check ok, absolute loss delta `0.0`

## Rejected Shortcuts

Based on current evidence, avoid these as default strategies:

- disabling warmup line search
- globally capping warmup line search too early
- globally lowering warmup Pi/Neumann to `2/2`
- assuming current batch-level Pi stats identify individual hard families
- accepting faster HOGENOM runs that stop hundreds of bits above the accepted
  likelihood
