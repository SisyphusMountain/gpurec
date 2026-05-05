# Global bf16-start Optimization Report

Date: 2026-05-05.

Scope: document the new global/uniform DTL-rate optimization pass that uses a
real bf16 initial phase for both forward and backward, then hands off to fp32
when the parameter updates or objective improvements become small.

Status: true resident-bf16 forward/backward is implemented, the strengthened
integration test passes, and the first threshold sweep is complete.

## Executive Summary

The prior bf16-start experiment was not a true bf16 optimization phase. It saved
some forward tensors in bf16, but the autograd bridge cast the saved forward
state and static floating tensors back to fp32 before running the implicit
gradient. That meant backward still paid fp32 memory and conversion costs, while
the optimizer also had to preserve resident fp32 static state to avoid rounding
the final fp32 objective.

The new pass changes the intended contract:

- bf16 start means bf16 forward and bf16 backward over a temporary bf16 static
  state;
- fp32 remains reserved for scalar optimizer state and selected numerically
  sensitive reductions/accumulators;
- handoff to fp32 can be driven by convergence signals, but the measured robust
  recommendation is a fixed one-evaluation bf16 bootstrap;
- the original fp32 static state is restored before the fp32 L-BFGS polish.

Measured recommendation for the current global/uniform optimizer:

```python
optimize_global_rates_lbfgs(
    model,
    min_rate=1e-10,
    steps=12,
    dtype=torch.float32,
    bf16_start_steps=1,
    bf16_switch_rate_rtol=None,
    bf16_switch_nll_abs_tol=None,
)
```

This is not because "small" was found to be a reliable bf16 convergence signal.
The useful setting was: do exactly one bf16 forward/backward/update, then hand
off to fp32. NLL-improvement thresholds are unsafe in bf16 because the scalar
NLL is quantized; rate-step thresholds in the tested `1e-2..1e-3` range did not
fire early enough to reduce time.

## What Changed Technically

### Prior path: bf16 forward, fp32 cast-back backward

The previous implementation kept the bf16-start phase safe, but not cheap
enough. It used a temporary bf16 static state for forward, then converted the
saved forward tensors and floating static state to fp32 inside
`_GeneReconFunction.backward` before calling the existing implicit-gradient
path.

Pseudo-code for the old path:

```python
original_static = model.static
model.static = cast_static(original_static, torch.bfloat16)

for step in range(fixed_bf16_steps):
    theta = theta.float()          # optimizer parameter stayed fp32
    loss = forward(theta, static_bf16)
    saved = ctx.saved_tensors      # Pi, Pibar, E, Ebar stored from bf16 forward

    # Old backward behavior.
    saved_fp32 = [x.float() for x in saved]
    static_fp32 = cast_static(static_bf16, torch.float32)
    grad = implicit_backward(saved_fp32, static_fp32)
    theta = adam_update(theta, grad.float())

model.static = original_static
theta = lbfgs_fp32(theta, model.static)
```

That design preserved correctness for experimentation but created three costs:

- saved bf16 forward tensors were materialized as fp32 for backward;
- temporary bf16 static state lived alongside the original fp32 static state;
- fp32 L-BFGS had to restore the exact original static tensors to avoid
  optimizing a rounded objective.

### New path: bf16 forward and bf16 backward start

The new pass removes the explicit autograd cast-back and allows the CUDA uniform
backward paths to run when `dtype == torch.bfloat16`. The bf16 phase still uses a
temporary static state, but the saved forward tensors remain bf16 through
backward unless an internal kernel or reduction deliberately accumulates in
fp32.

Pseudo-code for the new path:

```python
original_static = model.static
model.theta = model.theta.bfloat16()
model.static = cast_static(original_static, torch.bfloat16)

previous = None
for step in range(max_bf16_steps):
    loss = forward(theta_bf16, static_bf16)
    grad = backward_bf16_saved_tensors(loss, static_bf16)
    theta_bf16 = adam_update_fp32_moments(theta_bf16, grad).bfloat16()

    rates = exp2(theta_bf16.float())
    rel_step = None
    nll_gain = None
    if previous is not None:
        rel_step = max_abs(rates - previous.rates) / clamp_abs(previous.rates)
        nll_gain = previous.nll - loss

    if previous is not None and handoff(rel_step, nll_gain):
        break
    previous = Snapshot(rates=rates, nll=loss)

model.static = original_static
model.static.warm_E = None
theta = lbfgs_fp32(theta_bf16.float(), model.static)
```

The key behavioral difference is that bf16 is no longer just a forward-storage
experiment. Backward is expected to accept bf16 saved tensors directly through
the uniform CUDA path.

### Code-level touch points

- `gpurec/api/autograd.py`: removed the bf16 backward cast-back block that
  converted saved tensors and static tensors to fp32 before implicit gradient
  computation.
- `gpurec/core/backward.py`: expanded the supported CUDA backward dtype set to
  include `torch.bfloat16` for fused uniform backward paths.
- `gpurec/core/likelihood.py`: keeps the CUDA bf16 uniform ancestor sum in fp32
  accumulation, then casts `Ebar` back to the working dtype.
- `gpurec/optimization/global_optimizer.py`: keeps the resident evaluated
  `theta` tensor in bf16 during bf16 start, keeps only the three-parameter Adam
  moment vectors in fp32, swaps in a temporary bf16 static state, and restores
  the original static state before fp32 L-BFGS.

## What Remains Internally fp32

The new pass should not be "all operations bf16". The intended split is storage
and bandwidth in bf16, accumulation and optimizer control in fp32 where bf16
rounding would be too coarse.

| Component | Working dtype in bf16 start | Reason |
|---|---|---|
| `theta` / log-rates | bf16 resident, fp32 temporary update arithmetic | The user-requested start phase evaluates bf16 parameters; the actual Adam arithmetic uses fp32 temporaries before storing the rounded bf16 update. |
| Adam first/second moments | fp32 | Moment accumulation is sensitive to repeated small updates. |
| gradient consumed by optimizer | cast to fp32 | The update rule should see smooth gradients even if backward produced bf16 tensors internally. |
| uniform ancestor sum in `E_step` | fp32 accumulation for CUDA bf16 input | `row_sum - ancestor_sum` has cancellation risk before `safe_log2`. |
| selected reductions/logsumexp-style accumulations | fp32 where kernels require it | Reductions over many species/clades amplify bf16 rounding. |
| final fp32 polish objective | fp32 original static state | The final answer should be comparable to the existing fp32 baseline. |
| stored forward dynamic state | bf16 target | This is where memory pressure should fall if kernels avoid cast-back copies. |

The strengthened integration test monkeypatches the autograd bridge and checks
that the bf16 phase passes bf16 saved forward tensors and bf16 backward inputs.
The remaining fp32 islands above are internal accumulation/update choices, not a
whole-state cast-back of saved Pi/Pibar/E tensors.

## Handoff Criteria

The handoff should be based on accepted bf16-start evaluations, not raw line
search probes. The minimum useful policy records both relative rate movement and
NLL improvement:

```python
rel_rate_step = max(abs(rates_t - rates_prev) / clamp(abs(rates_prev), 1e-30))
nll_improvement = nll_prev - nll_t

small_step = rel_rate_step <= rate_step_threshold
small_gain = nll_improvement <= nll_improvement_threshold
handoff = min_bf16_steps_done and (small_step or small_gain)
```

Threshold sweep results:

| Policy ID | `rate_step_threshold` | `nll_improvement_threshold` | Min bf16 evals | Max bf16 evals | Result |
|---|---:|---:|---:|---:|---|
| fixed-1 | disabled | disabled | 1 | 1 | Best measured speed/quality tradeoff. |
| fixed-2 | disabled | disabled | 2 | 2 | Better large-slice NLL than fixed-1, but slower than fp32. |
| fixed-4 | disabled | disabled | 4 | 4 | Slower than fp32. |
| rate-1e-2 | `1e-2` | disabled | 2 | 8 | Did not fire before max on the large slice; slow. |
| rate-5e-3 | `5e-3` | disabled | 2 | 8 | Did not fire before max on the large slice; slow. |
| rate-1e-3 | `1e-3` | disabled | 2 | 8 | Did not fire before max on the large slice; slow. |
| gain-1e-3 | disabled/rate combined | `1e-3` bits | 2 | 8 | Fired after 2 evals on the large slice, but final NLL was much worse. |

Interpretation:

- bf16 NLL-improvement thresholds are not reliable because the bf16 scalar NLL
  can stay unchanged even while the fp32 objective is still far from the fp32
  solution. On the first 100 families of `test_trees_1000`, `abs(dNLL) <= 1e-3`
  switched after two bf16 evals and finished at `175528.015625`, about
  `+186.36` bits worse than fp32.
- rate-step thresholds from `1e-2` down to `1e-3` were too conservative on the
  larger workload: they never fired before the max of eight bf16 evals, making
  the run much slower.
- The measured useful setting is therefore not a positive "small" threshold; it
  is a fixed one-evaluation bf16 bootstrap followed by fp32.

## Benchmark Matrix

The final comparison should include at least:

- fp32 baseline: current production helper, no bf16 start;
- old bf16-start: fixed bf16 forward with fp32 cast-back backward;
- new pure-bf16 start: threshold handoff variants;
- selected final recommendation.

### Prior Recorded Results

These numbers are retained as the old-path baseline from the previous report.

#### `test_trees_100`

Command:

```bash
/usr/bin/time -f 'process_wall_s %e' \
  python profiling/bench_global_parameter_optimization.py \
    --dataset tests/data/test_trees_100 \
    --cache-dir /tmp/gpurec_paramopt_bf16_final_cache \
    --strategies recommended-fp32,bf16-start-fp32-polish \
    --init-rate 0.05 \
    --bf16-start-steps 4 \
    --fp32-polish-steps 8 \
    --no-print-evals
```

| Strategy | Evals | Hit eval | Optimizer time | Avg eval | Best NLL gap | Rate rel err | Peak alloc |
|---|---:|---:|---:|---:|---:|---:|---:|
| `recommended-fp32` | 11 | 5 | `2.219 s` | `0.202 s` | `0.00114 bits` | `6.79e-5` | `780 MB` |
| old `bf16-start-fp32-polish` | 14 | 9 | `4.160 s` | `0.297 s` | `0.00114 bits` | `1.17e-4` | `817 MB` |

Old-path read: same NLL neighborhood, but `1.87x` slower.

#### First 100 families of `test_trees_1000`

This benchmark used `g_0000.nwk` through `g_0099.nwk`.

| Strategy | Evals | Optimizer time | Avg eval | NLL | Rates `(D,L,T)` | Peak alloc | Peak reserved |
|---|---:|---:|---:|---:|---|---:|---:|
| fp32, `steps=12` | 13 | `12.070 s` | `0.886 s` | `175341.65625` | `(1.88e-5, 1.85e-5, 3.241e-2)` | `18.19 GB` | `20.70 GB` |
| old bf16 start 4 + fp32 steps 8 | 13 | `16.072 s` | `1.218 s` | `175490.60938` | `(2.12e-4, 2.04e-4, 3.240e-2)` | `23.54 GB` | `24.29 GB` |

Old-path read: slower, worse final NLL under the same evaluation budget, and
higher peak allocation because fp32 state and cast-back tensors were both live.

### New Resident-bf16 Results

| Workload | Strategy | Handoff threshold | bf16 evals | fp32 evals | Total evals | Final NLL | Rates `(D,L,T)` | Peak alloc | Peak reserved | Total time | Recommendation |
|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|---|
| `test_trees_100` | fp32 baseline | n/a | 0 | 12 | 12 | `14047.62890625` | `(1.912e-2, 1.993e-2, 2.083e-2)` | `779.59 MB` | `1044 MB` | `2.352 s` | reference |
| `test_trees_100` | pure bf16 start | fixed 1 | 1 | 7 | 8 | `14047.62890625` | `(1.912e-2, 1.993e-2, 2.083e-2)` | `780.60 MB` | `1044 MB` | `1.923 s` | use if bf16 start is enabled |
| `test_trees_100` | pure bf16 start | fixed 4 | 4 | 10 | 14 | `14047.62890625` | `(1.912e-2, 1.993e-2, 2.083e-2)` | `780.60 MB` | `1044 MB` | `4.806 s` | reject |
| `test_trees_100` | pure bf16 start | best threshold tested | 7 | 6 | 13 | `14047.63281250` | `(1.912e-2, 1.993e-2, 2.083e-2)` | `780.60 MB` | `1044 MB` | `2.635 s` | reject |
| first 100 of `test_trees_1000` | fp32 baseline | n/a | 0 | 13 | 13 | `175341.65625` | `(1.884e-5, 1.848e-5, 3.241e-2)` | `17349.46 MB` | `19740 MB` | `17.749 s` | reference |
| first 100 of `test_trees_1000` | pure bf16 start | fixed 1 | 1 | 13 | 14 | `175341.859375` | `(1.927e-5, 1.873e-5, 3.239e-2)` | `17349.21 MB` | `19638 MB` | `15.790 s` | best measured time/quality |
| first 100 of `test_trees_1000` | pure bf16 start | fixed 2 | 2 | 13 | 15 | `175339.53125` | `(1.597e-5, 1.512e-5, 3.241e-2)` | `17349.21 MB` | `19638 MB` | `19.175 s` | slower, better NLL |
| first 100 of `test_trees_1000` | pure bf16 start | fixed 4 | 4 | 10 | 14 | `175558.421875` | `(2.939e-4, 2.851e-4, 3.262e-2)` | `17349.21 MB` | `19638 MB` | `36.115 s` | reject |
| first 100 of `test_trees_1000` | pure bf16 start | NLL threshold, 2 evals | 2 | 9 | 11 | `175528.015625` | `(2.563e-4, 2.501e-4, 3.248e-2)` | `17349.21 MB` | `19638 MB` | `15.68 s` | reject: bad NLL |
| first 100 of `test_trees_1000` | pure bf16 start | rate-only max 8 | 8 | 13 | 21 | `175338.40625` | `(1.375e-5, 1.394e-5, 3.240e-2)` | `17349.21 MB` | `19638 MB` | `39.08 s` | reject: too slow |

The larger slice shows a modest speed win for fixed-one bf16 start:

```text
17.749 s fp32 baseline -> 15.790 s fixed-one bf16 start
speedup = 1.12x
NLL delta = +0.203125 bits
```

Memory did not improve materially. Peak allocated memory changed from
`17349.46 MB` to `17349.21 MB` on the larger slice, so the current benefit is
optimizer trajectory, not resident memory reduction.

## Recommendation Rule

Use the following promotion rule when worker results arrive:

1. Reject any threshold that produces non-finite values or requires a looser
   final NLL/rate tolerance than the fp32 baseline.
2. Reject any threshold whose final NLL is worse than fp32 by more than the
   established benchmark tolerance, even if it is faster.
3. Prefer the earliest handoff threshold that preserves final NLL/rates and
   improves total time on the larger workload.
4. If several thresholds tie on quality and time, choose the more conservative
   earlier handoff to reduce bf16 numerical exposure.

The measured recommendation is:

```text
bf16 start evaluations: 1
rate-step threshold: disabled
NLL-improvement threshold: disabled
fp32 polish: 12 LBFGS steps / max_eval as before
```

Equivalent helper call:

```python
optimize_global_rates_lbfgs(
    model,
    min_rate=1e-10,
    steps=12,
    max_eval=60,
    dtype=torch.float32,
    bf16_start_steps=1,
    bf16_switch_rate_rtol=None,
    bf16_switch_nll_abs_tol=None,
)
```

For the benchmark CLI this is:

```bash
python profiling/bench_global_parameter_optimization.py \
  --strategies bf16-start-fp32-polish \
  --bf16-start-steps 1 \
  --bf16-threshold-min-steps 1 \
  --bf16-threshold-max-steps 1 \
  --bf16-switch-rate-rtol 0 \
  --bf16-switch-nll-abs-tol 0 \
  --fp32-polish-steps 12
```

## Remaining Caveats

- The bf16 backward phase is slower per evaluation than fp32. On the larger
  slice, one bf16 eval spent `3.64 s` in backward, while fp32 LBFGS evals
  averaged about `1.07 s` of backward time. This is why more bf16 evaluations
  are quickly dominated.
- The current win comes from changing the starting point for fp32 L-BFGS, not
  from faster bf16 kernels.
- If future kernels remove the fp32 sparse fallbacks and reduce bf16 backward
  time, the rate-step thresholds should be revisited. Until then, positive
  "small" thresholds are not the right default.
