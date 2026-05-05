# Global bf16-start Optimization Report

Date: 2026-05-05.

Scope: testing whether global/uniform DTL-rate optimization benefits from an
initial bf16 phase before fp32 L-BFGS.

## Recommendation

Do not use bf16-start as the default for the current global optimizer.

The implemented bf16-start path is correct enough for experimentation, but it
is slower than the fp32 default on both tested workloads.  On the 100-family
slice of `test_trees_1000`, where memory pressure is high enough that bf16 had
the best chance to help, it increased optimizer time from `12.07 s` to
`16.07 s` and peak allocation from `18.19 GB` to `23.54 GB`.

The practical default remains:

```python
optimize_global_rates_lbfgs(
    model,
    min_rate=1e-10,
    steps=12,
    max_eval=60,
    dtype=torch.float32,
)
```

## Implemented Semantics

`optimize_global_rates_lbfgs` now accepts:

```python
bf16_start_steps: int = 0
bf16_start_lr: float = 0.05
```

When `bf16_start_steps > 0`:

1. The helper builds a temporary bf16 static state from the resident fp32
   static state.
2. `theta` remains fp32.
3. The bf16-start updates use Adam-style fp32 accumulators.
4. Backward casts the bf16 saved forward tensors to fp32 before calling the
   existing implicit-gradient path.
5. The original fp32 static state is restored before fp32 L-BFGS.

The last point is important.  An intermediate implementation cast the resident
static tensors to bf16 and then back to fp32.  That permanently rounded
topology-independent floating values and caused the fp32 polish to optimize a
different objective.  The final implementation avoids this by swapping in a
temporary bf16 static state and restoring the original fp32 state.

## Code Changes

- `gpurec/core/likelihood.py`: the uniform E-step ancestor sparse matmul now
  accumulates in fp32 for CUDA bf16 inputs, with autocast disabled around the
  sparse operation.
- `gpurec/api/autograd.py`: CUDA bf16 backward casts saved forward tensors and
  static floating tensors to fp32 before the implicit-gradient computation.
- `gpurec/optimization/global_optimizer.py`: added bf16-start support, per-eval
  forward/backward timing fields, fp32 Adam accumulators for bf16 start, and
  fp32 static-state restoration before L-BFGS.
- `profiling/bench_global_parameter_optimization.py`: added
  `bf16-start-fp32-polish`, `--bf16-start-steps`, `--fp32-polish-steps`, and
  memory/timing summary fields.
- `tests/integration/test_global_parameter_optimization.py`: added a short
  CUDA regression checking that bf16-start runs, returns finite fp32 results,
  and stays close to a short fp32 run.

## Correctness

Focused integration tests:

```bash
pytest -q -rs tests/integration/test_global_parameter_optimization.py
```

Result:

```text
3 passed in 4.38 s
```

The bf16-start test verifies:

- bf16 and fp32 phases are both present;
- final `theta` and returned rates are fp32;
- NLL and rates remain finite;
- the short bf16-start plus fp32-polish result remains close to the short fp32
  result on a small CUDA subset.

## Benchmarks

### `test_trees_100`

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
| `bf16-start-fp32-polish` | 14 | 9 | `4.160 s` | `0.297 s` | `0.00114 bits` | `1.17e-4` | `817 MB` |

The bf16-start path reached the same NLL neighborhood, but took `1.87x` longer.

### First 100 families of `test_trees_1000`

This benchmark used `g_0000.nwk` through `g_0099.nwk`.

| Strategy | Evals | Optimizer time | Avg eval | NLL | Rates `(D,L,T)` | Peak alloc | Peak reserved |
|---|---:|---:|---:|---:|---|---:|---:|
| fp32, `steps=12` | 13 | `12.070 s` | `0.886 s` | `175341.65625` | `(1.88e-5, 1.85e-5, 3.241e-2)` | `18.19 GB` | `20.70 GB` |
| bf16 start 4 + fp32 steps 8 | 13 | `16.072 s` | `1.218 s` | `175490.60938` | `(2.12e-4, 2.04e-4, 3.240e-2)` | `23.54 GB` | `24.29 GB` |

The bf16-start path is worse on the larger-slice workload:

- `1.33x` slower optimizer time.
- Higher NLL after the same evaluation budget.
- Higher peak allocation because the implementation keeps the original fp32
  static state alive while using a temporary bf16 static state, and backward
  casts saved bf16 forward tensors to fp32.

## Why It Did Not Help

The global optimizer is not limited by the three-parameter `theta` storage.
Most memory and time are in dynamic-program state and custom kernels.  The
current bf16-start path saves some forward storage, but then pays for:

- temporary bf16 static state alongside original fp32 static state;
- fp32 copies of saved bf16 forward tensors during backward;
- generic or less optimized dtype paths in parts of the uniform pipeline;
- an Adam-style warmup phase that is less efficient than L-BFGS in this
  three-parameter problem.

The experiment therefore confirms that this is not a useful optimization line
unless we implement true bf16 kernels with fp32 accumulation and avoid holding
both fp32 and bf16 static states simultaneously.

## Future Work

Only revisit bf16 if the goal is to fit a larger resident batch that otherwise
OOMs.  The next design should be kernel-level, not optimizer-level:

```text
stored Pi/Pibar/E: bf16
logsumexp and reductions: fp32
implicit-gradient accumulators: fp32
theta, loss, optimizer state: fp32
fp32 static topology/probability state: preserved exactly
```

That would require targeted Triton/CUDA kernel support and parity tests for
each affected forward/backward primitive.
