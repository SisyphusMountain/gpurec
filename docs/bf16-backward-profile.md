# bf16 Uniform Backward Profile

Date: 2026-05-05.

Scope: first 100 families from `tests/data/test_trees_1000`, global/uniform
mode, `fixed_iters_Pi=6`, `neumann_terms=3`, `max_wave_size=32768`,
`use_pruning=True`, CUDA device CC 8.9. Timings below are warmed unless the
table explicitly says otherwise.

## Summary

The bf16 backward slowdown was caused by bf16 global atomics in the fused
self-loop backward kernel, not by ordinary memory bandwidth.

The slow instruction pattern was:

```python
for s in species:
    u_d = term[s] * pibar_weight[s]
    for ancestor in ancestors(s):
        atomic_add(pibar_corr[ancestor], u_d)
```

Before the fix, `pibar_corr` used the same dtype as the saved forward state.
For bf16 this made `_wave_backward_uniform_kernel` spend almost all of the
backward pass in bf16 global `atomic_add`. Switching only that internal
correction scratch buffer to fp32 removes the pathological atomic path. The
kernel still consumes bf16 saved `Pi`, `Pibar`, constants, and RHS, and still
returns bf16 wave adjoints.

## Direct Backward Timing

Command shape:

```bash
python - <<'PY'
# Build GeneReconModel.from_trees(... dtype={fp32,bf16}, mode="global",
# pibar_mode="uniform"), run 2 warmups, then average 3 measured forward/backward
# passes with torch.cuda.synchronize() around each interval.
PY
```

| dtype | forward avg | backward avg | peak allocation | loss | theta gradient |
|---|---:|---:|---:|---:|---|
| fp32 | `0.229 s` | `0.633 s` | `18.18 GB` | `214831.53125` | `[12690.25, 11516.66, 3490.34]` |
| bf16 before fix | `0.187 s` | `2.97 s` | `9.16 GB` | `214016.0` | `[1515520, 950272, -1044480]` |
| bf16 after fix | `0.187 s` | `0.672 s` | `9.30 GB` | `214016.0` | `[1515520, 950272, -1028096]` |

The speed regression was therefore almost entirely in backward: `2.97 s ->
0.67 s`, a `4.4x` improvement, bringing bf16 to near fp32 speed while keeping
about half the peak allocation for the direct model evaluation.

The bf16 gradient is still a low-precision bootstrap gradient, not a parity
gradient. This fix addresses the performance bug; it does not make bf16 a
drop-in replacement for the final fp32/fp64 optimization phase.

## Component Isolation

I monkeypatched `implicit_grad_loglik_vjp_wave` to time its main regions. The
bf16 slowdown was isolated to `Pi_wave_backward`.

| dtype / version | total backward | `Pi_wave_backward` | CG / implicit solve | theta VJP |
|---|---:|---:|---:|---:|
| fp32 | `~0.63 s` | `~0.626 s` | `~0.006 s` | `~0.001 s` |
| bf16 before fix | `~3.01 s` | `~2.999 s` | `~0.008 s` | `~0.001 s` |

Parameter-gradient atomics were tested as a false lead:

```bash
GPUREC_FUSED_WAVE_PARAM_ACCUM=0
GPUREC_SELF_LOOP_PARAM_TWO_STAGE=1
```

These variants stayed around `3.0 s` before the fix. The slow atomics were not
the final parameter-gradient accumulation atomics; they were the per-wave
ancestor-correction atomics inside the Neumann self-loop.

## Nsight Systems Kernel Breakdown

Captured with:

```bash
nsys profile --trace=cuda,nvtx,osrt \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  python profiling/profile_bf16_backward_once.py \
    --dtype bf16 --max-families 100 --warmup 2
```

Top GPU kernel buckets:

| kernel bucket | fp32 total | bf16 before fix | bf16 after fix |
|---|---:|---:|---:|
| `_wave_backward_uniform_kernel` | `480.5 ms` / 39 launches | `2913.2 ms` / 43 launches | `494.9 ms` / 43 launches |
| `_wave_step_uniform_kernel` | `151.9 ms` | `134.5 ms` | `134.5 ms` |
| `_dts_cross_backward_accum_kernel` | `63.3 ms` | `74.3 ms` | `74.9 ms` |
| `_uniform_cross_pibar_vjp_tree_from_ud_compact_kernel` | `31.5 ms` | `39.4 ms` | `39.6 ms` |

The forward kernels were already faster in bf16. Only the self-loop backward
bucket was abnormal.

Representative large `_wave_backward_uniform_kernel` launches from the Nsys
SQLite export:

| launch shape | fp32 | bf16 before fix | bf16 after fix |
|---|---:|---:|---:|
| `grid=32768, block=128` | `~48.3-48.5 ms` | `~283-288 ms` | `~49.1-49.8 ms` |
| `grid=32139, block=128` | `47.8 ms` | `282.4 ms` | `48.9 ms` |
| `grid=27896, block=128` | `41.1 ms` | `240.9 ms` | `41.9 ms` |
| `grid=20565, block=128` | `30.5 ms` | `182.2 ms` | `31.4 ms` |

This is the main evidence for the fix: the same large self-loop waves moved
from roughly `6x` slower than fp32 to fp32 parity.

## Nsight Compute Large-Wave Counters

Captured with:

```bash
ncu --target-processes all --profile-from-start off --set detailed \
  --kernel-id ::regex:_wave_backward_uniform_kernel:37 --launch-count 1 \
  python profiling/profile_bf16_backward_once.py \
    --dtype bf16 --max-families 100 --warmup 2
```

The `--kernel-id` selector is important. `--launch-skip-before-match` selected
a tiny two-row specialization and produced misleading low-utilization counters.

| metric | fixed bf16 large wave | fp32 large wave |
|---|---:|---:|
| grid / block | `20565 x 128` | `32768 x 128` |
| NCU duration | `36.25 ms` | `56.80 ms` |
| registers / thread | `48` | `40` |
| achieved occupancy | `82.31%` | `99.06%` |
| achieved active warps / SM | `39.51` | `47.55` |
| L2 throughput | `70.75%` | `75.64%` |
| DRAM throughput | `4.91%` | `21.79%` |
| compute throughput | `18.36%` | `18.69%` |
| SM busy | `9.80%` | `8.76%` |
| L1 hit rate | `25.14%` | `23.99%` |
| L2 hit rate | `98.71%` | `92.46%` |
| local spilling requests | `329040` | `1048576` |

After the atomic fix, bf16 looks like the expected memory/L2-limited kernel:
L2 is heavily used, compute is underused, and occupancy is high enough to hide
some latency. The remaining resource pressure is the same algorithmic pressure
we already saw in fp32: the ancestor walk causes uncoalesced global accesses.
NCU reports about `44%` excessive global sectors for both fixed bf16 and fp32.

## Code Change

Only `gpurec/core/kernels/wave_backward.py` changed in the production path:

```python
pibar_corr_dtype = torch.float32 if dtype == torch.bfloat16 else dtype

if pibar_corr_dtype == dtype:
    pibar_corr_buf = scratch["pibar_corr"]
else:
    pibar_corr_buf = torch.empty((W, S), dtype=pibar_corr_dtype, device=device)
```

This is intentionally not a broad fp32 cast-back. It is a one-buffer exception
for a buffer whose only purpose is receiving many global atomics during the
ancestor correction:

```python
atomic_add(pibar_corr[ancestor], u_d)
correction = load(pibar_corr[s]).to(working_dtype)
```

Giving up fp32 accumulation everywhere is acceptable for some bf16 paths, but
not for this one. Using bf16 for this buffer is exactly what produced the
slowdown.

## Correctness Checks

Focused tests:

```bash
python -m py_compile gpurec/core/kernels/wave_backward.py \
  profiling/profile_bf16_backward_once.py

pytest -q \
  tests/kernels/test_wave_backward_kernel.py \
  tests/integration/test_global_parameter_optimization.py::test_global_lbfgs_bf16_start_uses_bf16_forward_backward_then_fp32_polish
```

Result: `14 passed` after adding a direct bf16 synthetic fused-kernel test.

The new test compares the bf16 fused self-loop output with an fp32 analytical
reference at bf16 tolerances and checks that all bf16 outputs remain finite.

## Remaining Notes

The first bf16 optimizer evaluation in a fresh process can still include Triton
JIT compile time. In the parameter-optimization CLI, a fixed-one bf16 bootstrap
measured:

| strategy | optimizer time | bf16 phase backward | final NLL |
|---|---:|---:|---:|
| fp32 baseline | `13.34 s` | n/a | `175341.65625` |
| bf16 fixed-one + fp32 polish | `14.13 s` | `2.30 s` | `175341.84375` |

That number is not a warmed-kernel throughput measurement. It is useful for
end-to-end cold-start optimization accounting, while the direct backward and
Nsight measurements above isolate the actual kernel performance.
