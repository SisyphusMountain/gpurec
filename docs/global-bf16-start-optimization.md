# Global bf16-start Optimization Report

Date: 2026-05-05.

Scope: document the new global/uniform DTL-rate optimization pass that uses a
real bf16 initial phase for both forward and backward, then hands off to fp32
when the parameter updates or objective improvements become small.

Status: true resident-bf16 forward/backward is implemented and the strengthened
integration test passes. Full threshold-sweep results are being collected.

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
- handoff to fp32 is driven by convergence signals rather than a fixed warmup
  length alone;
- the original fp32 static state is restored before the fp32 L-BFGS polish.

Provisional default recommendation: keep the production default at pure fp32
until the threshold sweep is complete:

```python
optimize_global_rates_lbfgs(
    model,
    min_rate=1e-10,
    steps=12,
    max_eval=60,
    dtype=torch.float32,
    bf16_start_steps=0,
)
```

The final default bf16-to-fp32 handoff threshold should not be promoted until it
matches fp32 final NLL/rates within the accepted tolerance and improves total
time or resident memory on the larger workload.

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

    rates = exp2(theta)
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

### Expected code-level touch points

The current worker-side implementation shape, inferred from the active diff, is:

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

This section should be revised after worker code lands if any API names or
handoff knobs differ.

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

The document should call out any additional fp32 islands found by profiling.
Those are acceptable when they are true accumulators, but not if they silently
materialize whole saved bf16 forward tensors as fp32.

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

The worker sweep should fill in this table.

| Policy ID | `rate_step_threshold` | `nll_improvement_threshold` | Min bf16 evals | Max bf16 evals | Result |
|---|---:|---:|---:|---:|---|
| fixed-old | n/a | n/a | 4 | 4 | Prior fixed-step comparison retained below. |
| rate-1e-2 | `1e-2` | disabled | TBD | TBD | Pending worker result. |
| rate-5e-3 | `5e-3` | disabled | TBD | TBD | Pending worker result. |
| rate-1e-3 | `1e-3` | disabled | TBD | TBD | Pending worker result. |
| gain-1e-1 | disabled | `1e-1` bits | TBD | TBD | Pending worker result. |
| gain-1e-2 | disabled | `1e-2` bits | TBD | TBD | Pending worker result. |
| combined | TBD | TBD | TBD | TBD | Pending worker result. |

Required result fields for each threshold:

- number of bf16 evaluations before handoff;
- fp32 polish evaluations;
- final NLL and NLL delta versus fp32 baseline;
- final rates `(D, L, T)` and max relative rate error versus fp32 baseline;
- total optimizer time;
- peak allocated and peak reserved GPU memory;
- any non-finite, instability, or convergence anomaly.

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

### Worker Results Intake

Fill this table from the new pure-bf16 start benchmark results.

| Workload | Strategy | Handoff threshold | bf16 evals | fp32 evals | Total evals | Final NLL | Rates `(D,L,T)` | Peak alloc | Peak reserved | Total time | Recommendation |
|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|---|
| `test_trees_100` | fp32 baseline | n/a | 0 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | reference |
| `test_trees_100` | old bf16-start | fixed 4 | 4 | TBD | 14 | see prior table | see prior table | `817 MB` | TBD | `4.160 s` | reject |
| `test_trees_100` | new pure-bf16 start | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | pending |
| first 100 of `test_trees_1000` | fp32 baseline | n/a | 0 | TBD | 13 | `175341.65625` | `(1.88e-5, 1.85e-5, 3.241e-2)` | `18.19 GB` | `20.70 GB` | `12.070 s` | reference |
| first 100 of `test_trees_1000` | old bf16-start | fixed 4 | 4 | TBD | 13 | `175490.60938` | `(2.12e-4, 2.04e-4, 3.240e-2)` | `23.54 GB` | `24.29 GB` | `16.072 s` | reject |
| first 100 of `test_trees_1000` | new pure-bf16 start | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | pending |

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

Pending data, the clear default recommendation is:

```text
default bf16 handoff threshold: disabled / no bf16 start
```

If the worker sweep shows parity and a material speed or memory win, update this
line to the chosen threshold, for example:

```text
default bf16 handoff threshold: rel_rate_step <= <value>
minimum bf16 evaluations: <value>
maximum bf16 evaluations: <value>
fallback: force fp32 after <value> bf16 evaluations
```

## Open Items For Workers

- Confirm that no whole saved forward tensors are cast to fp32 in the bf16
  backward path.
- Report any kernels that still allocate large fp32 temporaries during bf16
  backward.
- Fill the threshold sweep table.
- Fill the benchmark matrix for `test_trees_100` and the first 100 families of
  `test_trees_1000`.
- Provide the final default threshold recommendation with the exact command and
  environment used to produce it.
