# Uniform Backward Optimization Proposals, Pass 3

This note is a third pass over the uniform backward pass after the two
optimization waves documented in:

- `docs/uniform-backward-fp32-fused-profile.md`
- `docs/uniform-backward-50tree-wave2-profile.md`
- `docs/uniform-backward-pruning-granularity.tex`
- `docs/uniform-backward-optimizations-summary.tex`

The goal here is not to repeat previous suggestions. The previous passes
already removed the largest obvious scratch streams, made most scalar
reductions device-side, staged the Pibar VJP input in the DTS kernel, and
changed the self-loop Neumann speciation update from scatter-scratch to
parent-gather. The next optimizations need to attack the remaining split-side
schedule and the places where pruning is measured but not actually used.

## Current reference point

The most recent documented 50-family default is:

```text
CUDA-event backward median: about 121.016 ms
Nsight captured backward event: 135.759 ms
Nsight summed kernel time: 109.177 ms
Kernel launches: 2970
```

The largest post-proposal-6 Nsight buckets are:

| Component | 50-family Nsys bucket | Status |
|---|---:|---|
| `_dts_cross_backward_accum_kernel` | `32.606 ms` | now largest bucket |
| `_uniform_cross_pibar_vjp_tree_from_ud_kernel` | `25.618 ms` | staged Pibar tree correction |
| `_wave_backward_uniform_kernel` | `25.412 ms` | much improved, still memory-latency limited |
| `_dts_fused_kernel` | `12.207 ms` | backward-side DTS forward recompute |
| `_active_mask_from_rhs_absmax_kernel` | `2.919 ms` | simple streaming row max |

Representative resource counters from the final/near-final profiles:

| Kernel | Representative launch | Main resource signal |
|---|---:|---|
| `_dts_cross_backward_accum_kernel`, staged path | `4.855 ms`, `2.027 GB` read, `1.319 GB` write, `96` regs/thread, `41.5%` occupancy | register pressure plus memory/RED traffic |
| `_uniform_cross_pibar_vjp_tree_from_ud_kernel` | `4.220 ms`, `2.023 GB` read, `1.318 GB` write, `40` regs/thread, `98.4%` occupancy | memory-bound tree-buffer traffic |
| `_wave_backward_uniform_kernel`, final gather path | `5.102 ms`, `1.008 GB` read, `1.895 GB` write, `40` regs/thread, `99.4%` occupancy | load-latency and memory throughput |
| `_dts_fused_kernel` | `2.120 ms`, `1.540 GB` read, `0.350 GB` write | pure split-term streaming/recompute |

So the next useful work is mostly not "make occupancy higher". The self-loop
kernel already has high occupancy, and the staged Pibar tree also has high
occupancy. The remaining cost is repeated memory traffic, split-side work that
is scheduled too broadly, and some host decisions that still serialize the wave
loop.

## What has already been tried

These ideas should not be repeated without a substantially different dataflow.

| Idea | Result | Why it failed or plateaued |
|---|---:|---|
| Disable pruning | `157.329 ms -> 241.924 ms` in a later 50-family profile | whole-wave pruning skips real work |
| Fixed-schedule device pruning | about `157.329 ms -> 161.084-161.736 ms` | removed syncs but launched waves the host skipped |
| Child-grouped uniform Pibar VJP | tree kernel `37.784 -> 32.119 ms`, but group reduction `16.102 ms` | duplicate child factor was only about `1.111x` |
| Forward Pibar stat reuse alone | 10-family small win, 50-family median regression to `166.956 ms` | replaced cache-friendly ancestor work with extra DRAM reads |
| Backward row-stat precompute | tree kernel saved `0.848 ms`, precompute cost `2.690 ms` | full extra pass over rows did not pay back |
| Prefix-denominator Pibar VJP | 50-family median `162.704 -> 180.791 ms` | level barriers and L2/DRAM traffic dominated |
| Child-grouped DTS accumulation | 50-family median `164.352 -> 175.860 ms` | high-fanout waves were not child-duplicate-heavy |
| No-atomic DTS accumulation | 50-family median `164.352 -> 163.053 ms`, Nsys DTS bucket unchanged | atomics were visible but not the first-order cost |
| Full `grad_mt` accumulation inside DTS | slower than scalar-only reduction accumulation | many atomics into only `S` species lanes |
| Monolithic DTS plus Pibar fusion | rejected after staged DTS reached `96` regs/thread | more fusion would likely reduce occupancy further |
| Bigger wave block/warp sweeps | `BLOCK_S=512` and 16 warps slower | the winning self-loop point is already `BLOCK_S=256`, 8 warps |
| No-split leaf specializations | slower than compact all-wave scratch | compiler already specializes major branches; extra scratch reads hurt |
| Tensor cores | not used | hot kernels are log/exp, tree gathers, reductions, atomics, and memory streams |

The important lesson is that separate reduction/materialization passes usually
lose. Successful changes either removed a large memory stream, removed a host
sync without adding GPU work, or fused a reduction into values already live in
the hot kernel.

## Proposal 0: remove unused fused-path pruning work

This is the lowest-risk cleanup. In the default fused uniform path, the loop
still computes:

```python
n_active = int(active_mask.sum().item())
if n_active < W:
    active_idx = active_mask.nonzero(as_tuple=True)[0]
    rhs_active = rhs_k[active_idx]
```

For the fused path, `active_idx` and `rhs_active` are not used by
`wave_backward_uniform_fused`. They are useful in the generic PyTorch fallback,
but the default CUDA path only uses the mask for whole-wave skipping unless
the experimental device-pruning flags are enabled.

The cleanup should split pruning into two modes:

```python
wave_active = active_mask.any()      # still needed for host whole-wave skip
if not wave_active:
    continue

if use_fused:
    # no active_idx, no rhs_active, no nonzero
    # optionally skip active_mask.sum().item() outside debug/stats mode
    pass
else:
    n_active = int(active_mask.sum().item())
    active_idx = active_mask.nonzero(as_tuple=True)[0]
```

Expected gain:

- small but almost free;
- likely around `0.5-2 ms` at 50 families if it removes CUB `nonzero` work and
  one scalar sync for many partially active waves;
- also fixes misleading statistics: "rows below threshold" should not be
  reported as "rows whose fused work was skipped".

Correctness risk is low because this does not change computed adjoints.

Profiling gate:

- Nsight Systems should show fewer CUB select/compact kernels, fewer PyTorch
  index kernels, and possibly fewer D2H scalar syncs.
- Numerical output should be bitwise identical because the fused kernels
  receive the same inputs as before.

### Proposal 0 implementation result

Implemented in `gpurec/core/backward.py`.  The production fused uniform path now
keeps only the whole-wave pruning decision:

```python
active_mask = _compute_active_mask(rhs_k)
if not active_mask.any():
    n_waves_skipped += 1
    n_clades_skipped += W
    continue

if use_fused:
    if GPUREC_BACKWARD_PRUNING_ROW_STATS:
        n_active = int(active_mask.sum().item())
        n_clades_skipped += W - n_active
    else:
        n_active = W
else:
    n_active = int(active_mask.sum().item())
    n_clades_skipped += W - n_active
```

Then compaction is also restricted to the generic fallback:

```python
if not kernel_pruning_wave and not use_fused:
    active_idx = active_mask.nonzero(as_tuple=True)[0]
    rhs_active = rhs_k[active_idx]
else:
    active_idx = None
    rhs_active = rhs_k
```

This keeps fallback behavior unchanged, while the default fused Triton path no
longer pays for compacted rows it never consumes.  `n_clades_skipped` in the
default fused path now means clades skipped by whole-wave pruning.  The old
partial-row count remains available only as the diagnostic
`GPUREC_BACKWARD_PRUNING_ROW_STATS=1`.

Correctness checks:

| Check | Result |
|---|---:|
| `python -m py_compile gpurec/core/backward.py` | pass |
| `pytest -q tests/kernels/test_active_mask_kernel.py tests/kernels/test_uniform_cross_pibar_vjp_kernel.py tests/kernels/test_dts_backward_accum_kernel.py` | `24 passed` |
| `pytest -q tests/gradients/test_autograd_bridge.py` | `15 passed` |
| `GPUREC_FUSED_UNIFORM_BACKWARD=0 pytest -q tests/gradients/test_autograd_bridge.py` | `15 passed` |
| `/tmp/gpurec_profile/check_b2_pruning_parity.py`, `FAMS=10` | loss diff `0`, theta max-rel `3.11e-6` for no-CPU/device pruning diagnostics |
| clean `HEAD` old path vs proposal-0 path, `FAMS=10` | loss diff `0`, theta max-rel `0` |

One broader standalone kernel file was also tried:
`pytest tests/kernels/test_active_mask_kernel.py tests/kernels/test_wave_backward_kernel.py`.
It still has pre-existing direct-kernel tolerance failures unrelated to this
host-pruning branch, so it was not used as the acceptance gate for proposal 0.

Event benchmark commands:

```bash
FAMS=10 REPS=15 WARMUPS=5 python /tmp/gpurec_profile/bench_uniform_backward.py
FAMS=50 REPS=15 WARMUPS=5 python /tmp/gpurec_profile/bench_uniform_backward.py
```

For old-code measurements I used a clean detached worktree at `HEAD` and forced
imports with `PYTHONPATH=/tmp/gpurec_prop0_old`, because the benchmark script is
executed by absolute path and otherwise imports the editable checkout.

| Workload | Old median | Proposal 0 median | Delta |
|---|---:|---:|---:|
| 10 families | `37.137 ms` | `36.419 ms` | `-0.718 ms` / `-1.9%` |
| 50 families | `120.771 ms` | `119.707 ms` | `-1.064 ms` / `-0.9%` |

Nsight Systems 50-family single-backward capture:

| Metric | Old clean `HEAD` | Proposal 0 | Delta |
|---|---:|---:|---:|
| CUDA event in capture | `134.394 ms` | `132.435 ms` | `-1.959 ms` |
| summed GPU kernel time | `109.003 ms` | `108.682 ms` | `-0.321 ms` |
| kernel launches | `2970` | `2846` | `-124` |
| `cudaStreamSynchronize` calls | `254` | `205` | `-49` |
| `cudaMemcpyAsync` calls | `576` | `527` | `-49` |
| D2H memcpy events | `194` | `145` | `-49` |
| `cudaLaunchKernel` API time | `4.952 ms` | `4.603 ms` | `-0.349 ms` |

The removed work breaks down as:

| Removed bucket | Old launches | Proposal 0 launches | Old GPU time |
|---|---:|---:|---:|
| active row `sum().item()` long reductions | `36` | `0` | `0.104 ms` |
| `nonzero` count/reduce kernels | `26` | `0` | `0.043 ms` |
| CUB device select kernels | `13` | `0` | `0.020 ms` |
| compacted row index-gather kernels | `13` | `0` | `0.048 ms` |

A diagnostic run with `GPUREC_BACKWARD_PRUNING_ROW_STATS=1` kept the row-count
reduction but still disabled fused-path compaction.  It had `2918` kernel
launches, `241` `cudaStreamSynchronize` calls, and a 50-family event median of
`119.896 ms`, essentially inside timing noise from the production proposal-0
path.  This shows that the row-count sync is measurable in Nsys, but the runtime
impact is small; the useful cleanup is mainly semantic and removes avoidable
host decisions.

The hot fused kernels were unchanged within noise:

| Kernel | Old Nsys bucket | Proposal 0 bucket |
|---|---:|---:|
| `_dts_cross_backward_accum_kernel` | `32.561 ms` | `32.580 ms` |
| `_uniform_cross_pibar_vjp_tree_from_ud_kernel` | `25.527 ms` | `25.467 ms` |
| `_wave_backward_uniform_kernel` | `25.379 ms` | `25.369 ms` |
| `_dts_fused_kernel` | `12.204 ms` | `12.214 ms` |
| `_active_mask_from_rhs_absmax_kernel` | `2.919 ms` | `2.919 ms` |

Conclusion: accept proposal 0 as a small cleanup.  It removes 49 host-visible
scalar decisions and 124 launches in the 50-family backward pass, but it does
not move the major compute buckets because only 13 processed waves were
partially active enough to trigger the old `nonzero` path, and the removed
kernels were tiny compared with DTS/Pibar/self-loop work.  The remaining
pruning bottleneck is still the host whole-wave `active_mask.any()` decision;
changing that belongs to proposal 1, not this cleanup.

## Proposal 1: hybrid host-wave skip plus fused row masks

The failed device-pruning experiment launched a fixed schedule for all waves.
That was the wrong comparison: it removed CPU syncs but lost whole-wave
skipping. The more useful hybrid is:

```python
mask = active_rows(rhs)
if not mask.any():
    skip the entire wave pipeline
else:
    pass mask into the fused wave, DTS forward, DTS backward, and Pibar kernels
```

This keeps the strongest property of host pruning, namely skipping whole waves,
while making row-level pruning real inside active waves. It is different from
the rejected `GPUREC_DEVICE_PRUNING=1` path because inactive waves remain
unscheduled.

Policies to test:

| Policy | Purpose |
|---|---|
| always pass mask for active waves | maximum row pruning, measures branch/mask overhead |
| pass mask only when `n_active < W` | avoids overhead on fully active waves |
| pass mask only when inactive fraction is at least `10%` or `25%` | avoids branch overhead for almost-full waves |
| pass mask only to split-side kernels | targets DTS/Pibar buckets first |
| pass mask only to self-loop kernel | isolates self-loop row-pruning value |

Important implementation detail: policies that require `n_active` should not
use `sum().item()` in production. First implement the simple "always pass mask
after whole-wave skip" variant, then add a debug/profiling variant that
collects active fractions. If active-fraction thresholds are needed in
production, compute them with a device-side counter or preallocated worklist.

Expected gain:

- upper bound is the work spent on inactive rows inside waves that are not
  wholly inactive;
- the pruning document suggests the current default counts many inactive rows
  that are not actually skipped, but we still need exact post-proposal-6
  inactive-row fractions;
- likely range is `0-8 ms` at 50 families, with the high end only if many
  processed split waves are partially inactive.

Main risk:

- every masked kernel has an extra branch and may write zeros for inactive
  rows unless carefully structured;
- if most active waves are almost full, the branch overhead can exceed savings.

Profiling gate:

- compare Nsys buckets for `_dts_fused_kernel`,
  `_dts_cross_backward_accum_kernel`, `_uniform_cross_pibar_vjp_tree_from_ud`,
  and `_wave_backward_uniform_kernel`;
- collect actual skipped row and skipped split-side counts separately from
  threshold-inactive diagnostics;
- reject if total kernel time does not fall even when row counts look good.

### Proposal 1 implementation result

Implemented in `gpurec/core/backward.py` and accepted as the default fused
uniform backward behavior.  `GPUREC_BACKWARD_HYBRID_ROW_PRUNING=0` disables it
for A/B profiling.  The older aliases `GPUREC_BACKWARD_HYBRID_ROW_MASK` and
`GPUREC_FUSED_ROW_ACTIVE_MASK` are still accepted when the primary variable is
unset.

The subagent split was:

| Role | Result |
|---|---|
| Agent 1, code-path audit | confirmed that proposal 0 made the fused path use the active mask only for whole-wave host skip, so partial active rows were still computed inside active waves |
| Agent 2, prototype and parity | prototyped mask forwarding into fused kernels; found the useful policy is `all`, with smaller wins for `self` and `splits` only |
| Agent 3, profiling plan | isolated the necessary Nsys buckets and NCU launches: largest launches plus representative partially inactive launches |
| Supervisor | integrated the targeted masks, ran correctness/FD tests, ran event/Nsys/NCU profiles, and documented the result here |

The implementation keeps the existing host whole-wave pruning decision and adds
three separate masks:

```python
active_mask = active_rows(rhs_k)
if not active_mask.any():
    continue                  # preserve whole-wave skip

active_mask_for_dts_forward = None
active_mask_for_wave_kernel = None
active_mask_for_split_kernels = None

if hybrid_row_pruning_enabled:
    active_mask = active_mask.contiguous()
    if target in {"all", "self", "wave"}:
        active_mask_for_dts_forward = active_mask
        active_mask_for_wave_kernel = active_mask
    if target in {"all", "split", "splits", "dts", "dts_pibar", "pibar"}:
        active_mask_for_split_kernels = active_mask
```

This matters because the fused backward wave has two different kinds of work.
The self-loop/DTS-forward part consumes wave rows, while the DTS accumulation
and Pibar VJP consume split-side rows through `reduce_idx`.  A single
`active_mask_for_kernels` switch made experiments ambiguous; separate masks let
us test where the win comes from.

Additional profiling-only policy knobs were added:

| Variable | Meaning |
|---|---|
| `GPUREC_BACKWARD_HYBRID_ROW_PRUNING_TARGETS=self` | mask only the DTS forward recompute and self-loop wave kernel |
| `GPUREC_BACKWARD_HYBRID_ROW_PRUNING_TARGETS=splits` | mask only DTS backward accumulation and split-side Pibar VJP |
| `GPUREC_BACKWARD_HYBRID_ROW_PRUNING_REQUIRE_PARTIAL=1` | compute `active_mask.sum().item()` and pass the mask only when at least one row is inactive |
| `GPUREC_BACKWARD_HYBRID_ROW_PRUNING_MIN_INACTIVE_FRAC=0.10` | compute the inactive fraction on the host and pass the mask only above the threshold |

The last two knobs deliberately reintroduce a scalar synchronization, so they
are not default production policy.  They are there to test whether the branch
overhead on nearly full waves is large enough to justify a future device-side
counter/worklist.

Correctness checks:

| Check | Result |
|---|---:|
| `python -m py_compile gpurec/core/backward.py` | pass |
| `pytest -q tests/gradients/test_autograd_bridge.py tests/kernels/test_active_mask_kernel.py tests/kernels/test_dts_fused_kernel.py tests/kernels/test_dts_backward_accum_kernel.py tests/kernels/test_uniform_cross_pibar_vjp_kernel.py` | `49 passed` |
| `pytest -q tests/gradients/test_fd_all_modes.py::test_analytic_matches_fd` | `6 passed` |
| exact-zero pruning disabled, 10 families, hybrid all on vs off | loss diff `0`, theta max-abs `2.441e-4`, theta max-rel `1.828e-7` |
| threshold pruning enabled, 10 families, hybrid `all` vs off | loss diff `0`, theta max-abs `0.004028`, theta max-rel `3.016e-6` |
| threshold pruning enabled, 10 families, hybrid `self` vs off | loss diff `0`, theta max-abs `0.004272`, theta max-rel `3.199e-6` |
| threshold pruning enabled, 10 families, hybrid `splits` vs off | loss diff `0`, theta max-abs `0.004150`, theta max-rel `3.108e-6` |

The small threshold-mode gradient differences are expected from doing real
row-level pruning at the same numerical cutoff that was previously used only
for whole-wave skipping.  The finite-difference test still passes.

CUDA-event benchmarks, 50 families, 15 measured repetitions:

| Policy | Env | Median | Delta vs off |
|---|---|---:|---:|
| row masks disabled | `GPUREC_BACKWARD_HYBRID_ROW_PRUNING=0` | `119.411 ms` | reference |
| all masks, default | none | `116.055 ms` | `-3.356 ms` / `-2.8%` |
| self-loop side only | `GPUREC_BACKWARD_HYBRID_ROW_PRUNING_TARGETS=self` | `118.280 ms` | `-1.131 ms` / `-0.9%` |
| split side only | `GPUREC_BACKWARD_HYBRID_ROW_PRUNING_TARGETS=splits` | `118.081 ms` | `-1.330 ms` / `-1.1%` |
| all masks, only partial waves | `GPUREC_BACKWARD_HYBRID_ROW_PRUNING_REQUIRE_PARTIAL=1` | `116.860 ms` | `-2.551 ms` / `-2.1%` |
| all masks, inactive fraction at least `10%` | `GPUREC_BACKWARD_HYBRID_ROW_PRUNING_MIN_INACTIVE_FRAC=0.10` | `117.224 ms` | `-2.187 ms` / `-1.8%` |
| all masks, inactive fraction at least `25%` | `GPUREC_BACKWARD_HYBRID_ROW_PRUNING_MIN_INACTIVE_FRAC=0.25` | `117.060 ms` | `-2.351 ms` / `-2.0%` |

A separate clean A/B run after making `all` the default measured `115.980 ms`
default versus `119.616 ms` disabled, a `3.636 ms` improvement.  The 10-family
workload did not benefit consistently: default and disabled both measured about
`35.75 ms`, which matches the idea that the benefit needs enough partially
inactive waves to amortize the branch and mask load.

Negative controls:

| Policy | Median |
|---|---:|
| `GPUREC_DEVICE_PRUNING=1` | `123.624 ms` |
| `GPUREC_BACKWARD_NO_CPU_PRUNING=1` | `122.712 ms` |

These remain slower because they launch fixed-schedule work for waves the host
would have skipped entirely.  Proposal 1 is specifically a hybrid: keep the
host whole-wave skip, then use the same mask inside the kernels for active
waves.

Nsight Systems, 50 families, one captured backward after warmup:

| Metric | Masks disabled | Hybrid `all` | Delta |
|---|---:|---:|---:|
| CUDA event in capture | `132.992 ms` | `130.488 ms` | `-2.504 ms` |
| summed GPU kernel time | `108.640 ms` | `105.521 ms` | `-3.119 ms` |
| kernel launches | `2846` | `2846` | `0` |
| `cudaStreamSynchronize` calls | `205` | `205` | `0` |
| `cudaLaunchKernel` API time | `4.695 ms` | `4.595 ms` | `-0.100 ms` |
| D2D memcpy time | `0.390 ms` | `0.391 ms` | `0.000 ms` |
| D2H memcpy time | `0.129 ms` | `0.127 ms` | `-0.002 ms` |

The improvement is not from fewer launches or fewer host syncs.  It is actual
GPU work removed inside already-launched kernels.

Nsys hot buckets:

| Kernel bucket | Masks disabled | Hybrid `all` | Delta |
|---|---:|---:|---:|
| `_dts_cross_backward_accum_kernel` | `32.571 ms` | `31.448 ms` | `-1.123 ms` |
| `_wave_backward_uniform_kernel` | `25.363 ms` | `24.089 ms` | `-1.274 ms` |
| `_uniform_cross_pibar_vjp_tree_from_ud_kernel` | `25.443 ms` | `24.923 ms` | `-0.520 ms` |
| `_dts_fused_kernel` | `12.210 ms` | `12.000 ms` | `-0.210 ms` |
| `_active_mask_from_rhs_absmax_kernel` | `2.918 ms` | `2.957 ms` | `+0.039 ms` |
| `_seg_lse_hdim_kernel` | `1.518 ms` | `1.515 ms` | `-0.003 ms` |

The per-launch deltas show why the total win is only a few milliseconds: the
largest launches are mostly active and do not change, but several mid-sized
launches become much cheaper.

| Kernel and launch-skip | Masks disabled | Hybrid `all` | Delta |
|---|---:|---:|---:|
| `_wave_backward_uniform_kernel`, skip 7 | `0.597 ms` | `0.052 ms` | `-0.544 ms` |
| `_wave_backward_uniform_kernel`, skip 8 | `0.363 ms` | `0.046 ms` | `-0.318 ms` |
| `_wave_backward_uniform_kernel`, skip 6 | `0.201 ms` | `0.036 ms` | `-0.164 ms` |
| `_dts_cross_backward_accum_kernel`, skip 7 | `0.483 ms` | `0.072 ms` | `-0.411 ms` |
| `_dts_cross_backward_accum_kernel`, skip 8 | `0.288 ms` | `0.044 ms` | `-0.244 ms` |
| `_uniform_cross_pibar_vjp_tree_from_ud_kernel`, skip 7 | `0.386 ms` | `0.020 ms` | `-0.367 ms` |
| `_uniform_cross_pibar_vjp_tree_from_ud_kernel`, skip 8 | `0.219 ms` | `0.019 ms` | `-0.200 ms` |
| `_dts_fused_kernel`, skip 7 | `0.113 ms` | `0.032 ms` | `-0.081 ms` |
| `_dts_fused_kernel`, skip 8 | `0.081 ms` | `0.020 ms` | `-0.062 ms` |

Nsight Compute confirmed the same shape.  For the largest representative
launches, the mask branch is almost free but also has almost nothing to skip:

| Launch | Metric | Masks disabled | Hybrid `all` |
|---|---|---:|---:|
| largest `_wave_backward_uniform_kernel`, skip 34 | duration | `5.175 ms` | `5.171 ms` |
| same | DRAM throughput | `55.86%` | `56.05%` |
| same | compute throughput | `39.61%` | `39.62%` |
| same | achieved occupancy | `99.42%` | `99.41%` |
| largest `_dts_cross_backward_accum_kernel`, skip 4 | duration | `5.330 ms` | `5.303 ms` |
| same | DRAM throughput | `62.32%` | `62.35%` |
| same | compute throughput | `24.43%` | `24.44%` |
| same | achieved occupancy | `41.54%` | `41.54%` |
| same | registers/thread | `96` | `96` |

For a representative partially inactive launch, the early-return branch removes
most of the work:

| Launch | Metric | Masks disabled | Hybrid `all` |
|---|---|---:|---:|
| `_wave_backward_uniform_kernel`, skip 7 | duration | `655.424 us` | `53.696 us` |
| same | memory throughput | `520.8 GB/s` | `37.1 GB/s` |
| same | DRAM throughput | `51.71%` | `3.69%` |
| same | compute throughput | `37.09%` | `5.56%` |
| same | achieved occupancy | `96.59%` | `76.30%` |
| `_dts_cross_backward_accum_kernel`, skip 7 | duration | `466.528 us` | `64.320 us` |
| same | memory throughput | `589.7 GB/s` | `399.0 GB/s` |
| same | DRAM throughput | `58.56%` | `39.64%` |
| same | compute throughput | `25.10%` | `7.51%` |
| same | L2 hit rate | `66.43%` | `98.60%` |
| same | achieved occupancy | `41.32%` | `34.88%` |

The low throughput percentages in the masked partially inactive launches are
not a regression.  They mean the kernel returns early for most rows, so there
is too little remaining work to saturate the device.

Decision: accept `all` as the default.  The target-only policies prove the win
is split across both parts of the backward pass.  The host-threshold policies
were slower than unconditional mask forwarding because they reintroduced
`active_mask.sum().item()` synchronization and did not save enough branch work.
If we revisit thresholds, the count needs to be produced device-side together
with a compact worklist; the current production path should keep passing the
mask unconditionally after the host whole-wave skip.

## Proposal 2: parent-reduced DTS forward recompute

The backward pass recomputes cross-DTS forward terms before the self-loop VJP:

```text
dts_fused(Pi, Pibar, splits) -> dts_term[n_splits, S]
seg_logsumexp(dts_term, reduce_idx) -> dts_r[W, S]
wave_backward_uniform_fused(..., dts_r) -> v_k
```

For high-fanout root-like waves this materializes a very large
`[n_splits, S]` tensor only to reduce it by parent. The largest documented
50-family split wave has:

```text
n_splits = 42155
S = 1999
dts_term size ~= 42155 * 1999 * 4 bytes = 337 MB
```

The current buckets are:

```text
_dts_fused_kernel:       12.207 ms
_seg_lse_hdim_kernel:     about 1.5 ms in earlier profiles
```

A parent-reduced DTS kernel should compute `dts_r` directly for multi-split
parents, avoiding the full split-by-species materialization.

The wave metadata already sorts multi-split parents contiguously:

```text
eq1 splits first, then ge2 splits sorted by reduce_idx
meta["ge2_ptr"] gives CSR parent ranges
meta["ge2_parent_ids"] maps CSR groups to wave-local parents
```

A practical implementation is a two-stage tiled reduction over split ranges:

```text
stage 1 grid: (parent_group, split_tile, species_block)
    for split in split_tile:
        compute the 5 DTS terms for species_block
        update local max and shifted sum
    write partial_max[parent_group, split_tile, species_block]
    write partial_sum[parent_group, split_tile, species_block]

stage 2 grid: (parent_group, species_block)
    reduce partial max/sum over split_tile
    write dts_r[parent, species_block]

eq1 path:
    compute dts_r directly without staging split terms
```

Why this is different from rejected child grouping:

- child grouping reduced duplicate child adjoints after DTS;
- this proposal reduces the split dimension by parent before materializing
  `dts_r`;
- the high-fanout structure is parent-heavy, not child-duplicate-heavy, so this
  matches the observed shape.

Expected gain:

- hard upper bound is roughly `_dts_fused_kernel + seg_lse`, about `13-14 ms`
  in the 50-family profile;
- realistic first implementation target is `4-8 ms`, because the new kernel
  still reads all child `Pi`/`Pibar` rows and does the same log/exp work;
- memory peak should also fall because the full `dts_term[n_splits, S]` buffer
  is avoided or replaced by much smaller partials.

Risks:

- if split tiles are too large, one program serializes too much work and loses
  parallelism;
- if partial buffers are too large or badly laid out, the second stage becomes
  another memory-bound reduction;
- the eq1 path may not benefit, so the implementation should switch only for
  high fanout or large `n_splits`.

Profiling gate:

- first instrument per-wave `n_splits`, number of ge2 parents, max/mean fanout,
  and bytes allocated for `dts_term`;
- enable the parent-reduced path only when `n_splits / n_ge2_parents` is high;
- NCU should show lower DRAM writes than `_dts_fused_kernel + seg_lse`;
- Nsys should show the combined DTS-forward bucket falling, not just moving
  time into a new partial-reduction kernel.

### Proposal 2 implementation result

Implemented and accepted as the default uniform backward DTS forward-recompute
path.  It can be disabled with `GPUREC_BACKWARD_PARENT_REDUCED_DTS=0`.  The
default implementation is `tiled` with `64` splits per partial tile and keeps
the conservative `8192` split threshold:

```text
GPUREC_BACKWARD_PARENT_REDUCED_DTS=tiled
GPUREC_BACKWARD_PARENT_REDUCED_DTS_TILE_SPLITS=64
GPUREC_BACKWARD_PARENT_REDUCED_DTS_MIN_SPLITS=8192
```

The subagent split was:

| Role | Result |
|---|---|
| Agent 1, code-path audit | verified that the target is the backward `_compute_dts_cross -> dts_fused -> seg_logsumexp` chain and that `meta["ge2_ptr"]` / `meta["ge2_parent_ids"]` already encode the CSR parent grouping |
| Agent 2, prototype | built the first direct parent-loop prototype and showed it was correct but too serial for production |
| Agent 3, profiling plan | measured wave fanout/materialization sizes and identified the high-ge2 waves and the all-eq1 wave that should be treated separately |
| Supervisor | added the tiled two-stage ge2 reduction, added a direct eq1-to-`dts_r` kernel, ran parity/FD/event/Nsys/NCU checks, promoted the winning path to default, and documented the result here |

The final dataflow is:

```text
old:
    dts_fused(Pi, Pibar, splits) -> dts_term[n_splits, S]
    dts_r[eq1_parent] = dts_term[eq1]
    dts_r[ge2_parent] = seg_logsumexp(dts_term[ge2], ge2_ptr)

new:
    eq1 kernel:
        compute one split and write directly to dts_r[parent, species_block]

    ge2 stage 1:
        grid = (parent_group, split_tile, species_block)
        compute logsumexp over 5 * TILE_SPLITS terms
        write partial_max and partial_sum

    ge2 stage 2:
        grid = (parent_group, species_block)
        merge partial_max / partial_sum tiles
        write dts_r[parent, species_block]
```

The ge2 stage computes the mathematically equivalent flat reduction

```text
dts_r[p,s] = logsumexp2_{split i under p, term t in 0..4}
             (log_split_prob[i] + DTS5[t,i,s])
```

instead of first forming `log_split_prob[i] + logsumexp2_t(DTS5[t,i,s])` and
then reducing across splits.  This removes one `log2` per split/species in the
ge2 path and avoids the full `[ge2_splits, S]` materialization.

Implementation details:

| File | Change |
|---|---|
| `gpurec/core/kernels/dts_fused.py` | added `_dts_eq1_to_rows_kernel`, `_dts_parent_reduced_ge2_stage1_kernel`, `_dts_parent_reduced_ge2_stage2_kernel`, and `dts_fused_parent_reduced` |
| `gpurec/core/forward.py` | extended `_compute_dts_cross` with explicit `parent_reduced` arguments; the shared forward helper remains default-off unless backward opts in |
| `gpurec/core/backward.py` | enables the parent-reduced path by default for uniform CUDA backward DTS recompute and passes the implementation/tile/threshold knobs |
| `gpurec/core/batching.py` | stores `ge2_max_fanout` and `ge2_mean_fanout` while `ge2_counts` is already available, avoiding a backward-loop sync for tile sizing |
| `tests/kernels/test_dts_fused_kernel.py` | adds parity coverage for direct/tiled parent-reduced DTS with fp32/fp64, active-mask on/off, and shared/per-split parameters |

Correctness checks:

| Check | Result |
|---|---:|
| `python -m py_compile gpurec/core/backward.py gpurec/core/forward.py gpurec/core/batching.py gpurec/core/kernels/dts_fused.py` | pass |
| `pytest -q tests/kernels/test_dts_fused_kernel.py tests/gradients/test_autograd_bridge.py` | `41 passed` |
| `pytest -q tests/kernels/test_dts_fused_kernel.py tests/gradients/test_autograd_bridge.py tests/gradients/test_fd_all_modes.py::test_analytic_matches_fd` after making it default | `47 passed` |
| `GPUREC_BACKWARD_PARENT_REDUCED_DTS=tiled GPUREC_BACKWARD_PARENT_REDUCED_DTS_TILE_SPLITS=64 pytest -q tests/gradients/test_fd_all_modes.py::test_analytic_matches_fd` | `6 passed` |

Per-wave instrumentation from the 50-family workload:

| Metric | 10 families | 50 families |
|---|---:|---:|
| total splits across backward waves | `83,135` | `402,275` |
| old DTS materialization volume per backward | `633.953 MiB` | `2.996 GiB` |
| ge2 splits | `33,260` | `160,940` |
| ge2 parent groups | `10` | `50` |
| mean ge2 fanout | `3326.00` | `3218.80` |

Largest 50-family DTS waves:

| Wave | Splits | Eq1 | Ge2 groups | Ge2 splits | Mean ge2 fanout | Max ge2 fanout | Old `dts_term` bytes |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 44 | `42155` | `234` | `13` | `41921` | `3224.69` | `3787` | `321.456 MiB` |
| 43 | `36200` | `572` | `12` | `35628` | `2969.00` | `3419` | `276.046 MiB` |
| 3 | `27023` | `27023` | `0` | `0` | `0.00` | `0` | `206.066 MiB` |
| 46 | `24229` | `32` | `7` | `24197` | `3456.71` | `4009` | `184.760 MiB` |
| 45 | `23345` | `86` | `7` | `23259` | `3322.71` | `3837` | `178.019 MiB` |
| 42 | `19642` | `1230` | `6` | `18412` | `3068.67` | `3391` | `149.782 MiB` |

This is why the final implementation includes both kernels.  The ge2 tiled
path handles waves 42-48; the eq1 direct path handles the large all-eq1 wave 3.

CUDA-event benchmark results:

| Workload | Policy | Median | Delta |
|---|---|---:|---:|
| 50 families, 15 reps | parent-reduced default | `112.976 ms` | `-3.710 ms` / `-3.2%` |
| 50 families, 15 reps | disabled, `GPUREC_BACKWARD_PARENT_REDUCED_DTS=0` | `116.686 ms` | reference |
| 10 families, 15 reps | parent-reduced default | `35.867 ms` | `-0.494 ms` / `-1.4%` |
| 10 families, 15 reps | disabled, `GPUREC_BACKWARD_PARENT_REDUCED_DTS=0` | `36.361 ms` | reference |

Variant sweep on 50 families before promotion:

| Variant | Median | Interpretation |
|---|---:|---|
| disabled | `116.238 ms` | reference in that sweep |
| direct parent loop | `117.152 ms` | correct but serializes thousands of splits per parent |
| tiled, 32 splits/tile | `114.932 ms` | better, more partial tiles |
| tiled, 64 splits/tile | `112.580 ms` | best measured point |
| tiled, 128 splits/tile | `113.897 ms` | less parallelism than 64 |
| tiled, 256 splits/tile | `113.665 ms` | still good, not best |

Nsight Systems 50-family capture:

| Metric | Disabled | Parent-reduced default | Delta |
|---|---:|---:|---:|
| CUDA event in capture | `130.306 ms` | `126.178 ms` | `-4.128 ms` |
| summed GPU kernel time | `105.375 ms` | `102.132 ms` | `-3.243 ms` |
| kernel launches | `2846` | `2852` | `+6` |
| `cudaStreamSynchronize` calls | `205` | `205` | `0` |
| `cudaLaunchKernel` API time | `4.631 ms` | `4.665 ms` | `+0.034 ms` |
| D2D memcpy time | `0.391 ms` | `0.391 ms` | `0.000 ms` |
| D2H memcpy time | `0.128 ms` | `0.128 ms` | `0.000 ms` |

The path adds six net launches, but avoids enough memory traffic that total
kernel time still falls.  Host synchronization does not increase.

Nsys DTS-forward bucket replacement:

| Bucket | Disabled | Parent-reduced default |
|---|---:|---:|
| `_dts_fused_kernel` | `12.002 ms`, `33` launches | `1.566 ms`, `24` launches |
| `_seg_lse_hdim_kernel` | `1.514 ms`, `7` launches | `0.066 ms`, `1` launch |
| `_dts_eq1_to_rows_kernel` | none | `2.696 ms`, `9` launches |
| `_dts_parent_reduced_ge2_stage1_kernel` | none | `7.037 ms`, `6` launches |
| `_dts_parent_reduced_ge2_stage2_kernel` | none | `0.119 ms`, `6` launches |
| combined DTS forward recompute | `13.516 ms` | `11.484 ms` |

The bucket falls by `2.032 ms`.  The full backward event falls more because the
direct `dts_r` production also removes PyTorch `index_put` work on the large
eq1/ge2 waves: the `index_put` bucket drops from `1.382 ms` to `0.392 ms`.

Nsys per-launch view:

| Old launch | Duration | New replacement | Duration |
|---|---:|---|---:|
| `_dts_fused_kernel`, wave 44, grid `(42155,16)` | `2.154 ms` | stage1 `(13,60,16)` + stage2 `(13,16)` + eq1 `(234,16)` | `1.912 + 0.022 + 0.003 ms` |
| `_seg_lse_hdim_kernel`, wave 44, grid `(13,16)` | `0.382 ms` | included above | included above |
| `_dts_fused_kernel`, all-eq1 wave 3, grid `(27023,16)` | about `1.31 ms` in NCU / large Nsys launch | `_dts_eq1_to_rows_kernel`, grid `(27023,16)` | `1.396 ms` |

For high-ge2 waves the new path is faster because it removes the full split
matrix write and later segment read.  For the all-eq1 wave, the direct kernel is
similar to old `dts_fused` but avoids the subsequent large `dts_term -> dts_r`
index copy.

Nsight Compute, largest high-ge2 wave 44:

| Kernel | Duration | Memory throughput | DRAM | SM | L2 hit | Registers/thread | Achieved occupancy |
|---|---:|---:|---:|---:|---:|---:|---:|
| old `_dts_fused_kernel` | `2.13 ms` | `888.20 GB/s` | `88.19%` | `17.73%` | `23.57%` | `31` | `94.45%` |
| old `_seg_lse_hdim_kernel` | `375.65 us` | `900.69 GB/s` | `89.44%` | `9.11%` | `8.35%` | `155` | `13.07%` |
| new `_dts_parent_reduced_ge2_stage1_kernel` | `1.879 ms` | `790.35 GB/s` | `78.47%` | `15.80%` | `11.05%` | `39` | `94.53%` |
| new `_dts_parent_reduced_ge2_stage2_kernel` | `23.104 us` | `457.77 GB/s` | `45.56%` | `4.65%` | `7.94%` | `25` | `12.69%` |

Stage 1 is still DRAM-bound and dominated by long-scoreboard stalls, but it
writes/reads a much smaller partial buffer than the old split matrix.  Stage 2
is tiny.  The old `seg_lse` was also DRAM-bound and low occupancy because of
its high register count; the new stage 2 avoids making that a substantial
bucket.

Nsight Compute, large all-eq1 wave:

| Kernel | Duration | Memory throughput | DRAM | SM | L2 hit | Registers/thread | Achieved occupancy |
|---|---:|---:|---:|---:|---:|---:|---:|
| new `_dts_eq1_to_rows_kernel`, grid `(27023,16)` | `1.355 ms` | `884.55 GB/s` | `87.83%` | `18.95%` | `25.32%` | `38` | `94.85%` |

The eq1 kernel is not fundamentally faster than `_dts_fused` for the same
single-split math; its value is eliminating the extra dense copy from
`dts_term` into `dts_r`.

Decision: accept the tiled implementation with `TILE_SPLITS=64` as the default.
The direct parent-loop prototype is retained as an experiment but should not be
used by default.  Remaining opportunities are mostly memory-layout and reuse
work inside stage 1: it is still DRAM-bound, has low L2 reuse, and spends most
stall cycles waiting on global loads.  Smaller algorithmic tweaks to stage 2
will not matter unless stage 1 is reduced further.

## Proposal 3: parent-tiled DTS backward accumulation

The current DTS backward accumulation is split-major:

```text
grid = (n_splits,)
program i:
    parent = reduce_idx[i]
    load parent Pi and v_k for every species block
    load left/right child Pi and Pibar
    compute vd0..vd4
    atomic_add direct Pi adjoints into child rows
    write staged Pibar u_d and A
```

This is now the largest bucket:

```text
_dts_cross_backward_accum_kernel: 32.606 ms at 50 families
representative staged launch: 4.855 ms
registers/thread: 96
achieved occupancy: 41.5%
global RED ops: 18.72 M
```

The high-fanout waves have many splits per parent:

```text
42155 splits over 247 parents: 170.7 splits/parent
24229 splits over 39 parents: 621.3 splits/parent
```

Split-major scheduling reloads the same parent `Pi_parent[s]` and `v_k[parent,s]`
for every split. A parent-tiled layout would instead make parent and species
block explicit:

```text
grid = (parent_group, split_tile, species_block)
load Pi_parent[species_block] once
load v_k[parent, species_block] once
for split in this parent's split_tile:
    load child rows
    compute vd0..vd4
    write or accumulate child adjoints and staged u_d
```

This is not the same as the rejected grouped DTS path. The rejected path grouped
by child after materializing local gradients, but the measured child duplicate
factor was about `1.00`. This proposal groups by parent, where the measured
fanout is large.

There are two variants to test.

### 3A. Parent-tiled split kernel with the same atomic outputs

Keep the same outputs as the current staged DTS kernel:

- direct atomic adds into `accumulated_rhs`;
- staged `pibar_ud[2 * n_splits, S]`;
- `pibar_A[2 * n_splits]`;
- scalar parameter reductions as today.

This isolates the benefit of parent reuse without changing the downstream
Pibar path.

Expected gain:

- parent `Pi_parent` and `v_k` loads become roughly one per split tile instead
  of one per split;
- if parent loads are a meaningful part of the current `2.027 GB` DRAM read,
  this can save several ms;
- realistic target is `3-6 ms` at 50 families.

Risk:

- looping over split tiles can increase live ranges and worsen the already high
  `96` register/thread pressure;
- direct child atomics and staged `u_d` writes remain, so the total gain is
  capped.

### 3B. Parent-tiled backward plus local split-side pruning stats

While parent tiling computes `vd1/vd2`, also compute cheap split-side magnitude
statistics:

```text
side_abs_l[i] = max_s abs(vd2 * inv_denom_l)
side_abs_r[i] = max_s abs(vd1 * inv_denom_r)
```

This does not change results yet. It tells us whether the staged Pibar tree is
processing many split sides whose `u_d` is exactly zero or below a threshold.
If many sides are inactive, Proposal 4 becomes much more attractive.

Profiling gate:

- compare representative NCU register count, occupancy, DRAM bytes, and RED
  operations against the current staged kernel;
- reject parent tiling if register pressure rises enough to erase memory-load
  savings;
- keep the first implementation restricted to root-like waves with high
  `n_splits / W`.

### Proposal 3 results

Subagent split:

| Worker | Role | Result |
|---|---|---|
| Agent 1 | read-only audit | confirmed the hot path is `_dts_cross_backward_accum_kernel`, that the repeated parent loads are `Pi_star[ws + reduce_idx[i], s]` and `v_k[reduce_idx[i], s]`, and that `n_eq1`/`ge2_ptr`/`ge2_parent_ids` already provide the needed parent CSR metadata |
| Agent 2 | implementation | added an opt-in CSR parent-tiled staged DTS backward kernel and parity tests for fp32/fp64, active masks, scalar reductions, staged `pibar_ud`, and `grad_mt` |
| Agent 3 | profiling plan | produced the baseline Nsys/NCU launch targets and acceptance gates; the key target launch was the 42,155-split wave at NCU skip 4 |
| Supervisor | integration and profiling | verified correctness, benchmarked default versus parent-tiled, profiled the regression with Nsys and NCU, and documented the decision |

Implementation:

```text
GPUREC_DTS_BACKWARD_ACCUM_IMPL=parent_tiled[_all]
GPUREC_PARENT_TILED_DTS_BACKWARD_TILE_SPLITS=16   # tested completing point
```

The prototype is intentionally opt-in.  It only covers the staged uniform hot
path: merged S-term, direct atomic Pi accumulation, scalar parameter reductions,
optional `grad_mt` accumulation, and `pibar_ud/pibar_A` output.  The direct path
remains the default.

Correctness:

| Command | Result |
|---|---:|
| `pytest -q tests/kernels/test_dts_backward_accum_kernel.py` | `27 passed` |
| `GPUREC_DTS_BACKWARD_ACCUM_IMPL=parent_tiled pytest -q tests/gradients/test_fd_all_modes.py::test_analytic_matches_fd` | `6 passed` |
| worker run: `GPUREC_DTS_BACKWARD_ACCUM_IMPL=parent_tiled_all GPUREC_PARENT_TILED_DTS_BACKWARD_TILE_SPLITS=2 pytest -q tests/gradients/test_autograd_bridge.py` | `15 passed` |

Event timings:

| Workload | Variant | Median backward time | Change |
|---|---|---:|---:|
| 10 families, 15 reps | default direct | `36.168 ms` | reference |
| 10 families, 15 reps | parent-tiled, tile 64 | `36.098 ms` | noise-level `-0.070 ms` |
| 50 families, 15 reps | default direct | `113.892 ms` | reference |
| 50 families, worker run | parent-tiled, tile 8 | `129.276 ms` | `+15.709 ms` / `+13.8%` |
| 50 families, worker run | parent-tiled, tile 16 | `127.043 ms` | `+13.476 ms` / `+11.9%` |

`TILE_SPLITS=64` was not usable for the 50-family run: the kernel loop is
statically unrolled by Triton, and the benchmark did not complete after several
minutes.  This is a compile/code-size failure mode, not a production candidate.

Nsys, one profiled 50-family backward:

| Bucket | Direct | Parent-tiled tile 16 | Interpretation |
|---|---:|---:|---|
| direct `_dts_cross_backward_accum_kernel` | `31.447 ms`, 33 launches | `11.932 ms`, 27 launches | small and non-ge2 waves still use the direct path |
| `_dts_cross_backward_accum_parent_tiled_ge2_kernel` | none | `33.832 ms`, 6 launches | replaces only the high-fanout ge2 launches |
| total DTS backward accumulation bucket | `31.447 ms` | `45.764 ms` | regression `+14.317 ms` |
| `_uniform_cross_pibar_vjp_tree_from_ud_kernel` | `24.781 ms` | `24.886 ms` | downstream staged Pibar VJP is unchanged |
| `_wave_backward_uniform_kernel` | `24.006 ms` | `24.028 ms` | self-loop backward is unchanged |

The largest direct launch:

```text
_dts_cross_backward_accum_kernel
grid=(42155, 1, 1)
duration=5.223 ms
registers/thread=96
achieved occupancy=41.53%
```

The corresponding parent-tiled launch:

```text
_dts_cross_backward_accum_parent_tiled_ge2_kernel
grid=(247, 237, 8)
duration=9.559 ms
registers/thread=96
achieved occupancy=41.09%
```

The parent-tiled launch has far more CTAs because the grid is rectangular in
`(n_eq1 + n_ge2_groups, max_tiles, species_blocks)`.  Eq1 rows only need tile
zero, but they are still launched for every ge2 `max_tiles` value and return
immediately.  For the high-split waves at tile 16:

| Wave | Splits | `n_eq1` | ge2 groups | active CTAs | rectangular CTAs | Overlaunch |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 19,642 | 1,230 | 6 | 19,072 | 2,096,256 | `109.91x` |
| 43 | 36,200 | 572 | 12 | 22,440 | 999,808 | `44.55x` |
| 44 | 42,155 | 234 | 13 | 22,888 | 468,312 | `20.46x` |
| 45 | 23,345 | 86 | 7 | 12,352 | 178,560 | `14.46x` |
| 46 | 24,229 | 32 | 7 | 12,384 | 78,312 | `6.32x` |
| 47 | 13,546 | 8 | 4 | 6,848 | 22,176 | `3.24x` |
| 48 | 3,985 | 0 | 1 | 2,000 | 2,000 | `1.00x` |

NCU, representative 42,155-split wave:

| Metric | Direct split-major | Parent-tiled tile 16 |
|---|---:|---:|
| Duration | `5.224 ms` | `9.559 ms` |
| Grid size | `42,155` CTAs | `468,312` CTAs |
| Threads launched | `5.396 M` | `59.944 M` |
| Registers/thread | `96` | `96` |
| Achieved occupancy | `41.53%` | `41.09%` |
| DRAM throughput | `63.30%` | `37.88%` |
| Memory throughput | `637.5 GB/s` | `381.5 GB/s` |
| Compute throughput | `24.83%` | `12.84%` |
| L1/TEX hit rate | `61.64%` | `48.76%` |
| L2 hit rate | `66.08%` | `61.14%` |
| Warp cycles / issued instruction | `42.42` | `71.98` |
| L1TEX scoreboard stall share | `33.7%` | `37.3%` |
| Global load instructions | `36.392 M` | `29.560 M` |
| Global store instructions | `5.396 M` | `5.312 M` |
| Global RED instructions | `18.613 M` | `19.871 M` |
| Global load sectors | `334.314 M` | `198.742 M` |
| Global store sectors | `25.799 M` | `25.715 M` |
| Global RED sectors | `103.275 M` | `104.533 M` |
| DRAM read bytes | `2.015 GB` | `2.217 GB` |
| DRAM write bytes | `1.316 GB` | `1.430 GB` |

The proposed reuse does happen: global load instructions and L1 sectors fall.
But the kernel still loses because:

1. The rectangular grid launches many empty CTAs, especially for eq1 groups.
2. Splitting species into a grid axis turns per-split scalar reductions
   (`pibar_A`, `grad_log_pD`, `grad_log_pS`) into per-split-per-species-block
   atomics.
3. The direct kernel aggregates all species blocks for one split inside one CTA,
   so it performs one scalar update per split side; the parent-tiled kernel
   performs up to eight partial scalar updates per split side for `S=1999`.
4. Cache behavior worsens enough that DRAM read/write bytes increase despite
   fewer global load instructions.
5. The new kernel has lower achieved memory and compute throughput, so it is
   latency/overhead limited rather than bandwidth limited.

Decision: reject this parent-tiled implementation as a default optimization.
Keep it opt-in only as a documented negative result.  A future viable parent
grouping would need a ragged tile list that launches only active parent tiles,
keeps eq1 on the old direct path, and avoids multiplying scalar reduction
atomics by the number of species blocks.

## Proposal 4: split-side worklists for staged Pibar VJP

The staged DTS path writes `u_d` and `A`, then the Pibar tree correction runs
one row per split side:

```text
_uniform_cross_pibar_vjp_tree_from_ud_kernel[(2 * n_splits,)]
```

The kernel no longer computes denominators, but it still reads and writes a
large tree buffer:

```text
representative launch: 4.220 ms
DRAM read: 2.023 GB
DRAM write: 1.318 GB
50-family Nsys bucket: 25.618 ms
```

A split side with zero or negligible `u_d` contributes nothing useful to the
tree correction:

```text
u_d = grad_Pibar * inv_denom
A = sum_s u_d[s]
contrib[s] = p_prime[s] * (A - subtree_sum[s])
```

If `u_d` is exactly zero, then `A` and every subtree sum are zero, so the whole
split-side tree program can be skipped exactly. For threshold pruning, the same
idea is approximate and must be guarded by the pruning threshold semantics.

Implementation sequence:

```text
1. In DTS staged kernel, compute side_abs_l/side_abs_r or exact side_nonzero.
2. For debugging only, report:
       active sides / total sides
       active sides by wave
       active sides inside already-active parent rows
3. If sparse enough, compact active split sides into a device worklist:
       side_child[side_id]
       side_ud_offset[side_id]
       side_A[side_id]
4. Launch Pibar tree only on active side_id rows.
```

The worklist should be generated on device. A host `nonzero().item()` version
would likely repeat the same mistake as fixed-schedule device pruning or
CPU-driven compaction.

Expected gain:

- upper bound is the `25.618 ms` staged Pibar tree bucket;
- realistic gain depends entirely on measured side sparsity;
- if `20%` of split sides are exactly inactive in processed split waves, the
  target is roughly `4-5 ms` before worklist overhead;
- if almost every side is active, this proposal should be dropped.

Risks:

- worklist construction is another reduction/compaction pass;
- if side sparsity is low, the worklist pass will cost more than it saves;
- approximate split-side thresholding can perturb gradients, so start with
  exact-zero skipping and only later evaluate thresholded pruning.

Profiling gate:

- add a stats-only mode first; do not implement compaction until side sparsity
  is known;
- measure worklist construction separately from the tree kernel reduction;
- verify old/new parity exactly for exact-zero mode and with finite-difference
  checks for thresholded mode.

### Proposal 4 experiment: exact-zero split-side skipping

Supervisor date: 2026-05-03.

Subagent split:

- audit agent: checked the staged `pibar_ud` layout and safety condition;
- implementation agent: added an opt-in exact-zero side mask and parity tests;
- profiling agent: generated Nsight Systems / Nsight Compute artifacts under
  `/tmp/gpurec_profile`; the supervisor parsed and re-ran the key benchmarks
  locally because the profiler agent did not return a final message before the
  timeout.

The implemented test is the low-risk prefix of the proposal, not a full
compacted worklist.  The current code marks rows of staged `u_d` that are not
exactly all zero:

```text
side_active[row] = max_s(abs(pibar_ud[row, s])) != 0
```

and passes that boolean mask to the from-`u_d` Pibar tree kernel:

```text
row = program_id(0)
if side_active[row] == 0:
    return

bottom_up_subtree_sum(pibar_ud[row, :])
atomic_add(accumulated_rhs[child, :], p_prime * (A - subtree_sum))
```

The path is off by default and enabled with either:

```text
GPUREC_DTS_PIBAR_UD_SKIP_ZERO_SIDES=1
GPUREC_DTS_PIBAR_UD_WORKLIST=1
```

The second spelling is accepted as an alias for this exact-zero experiment, but
the code does not yet compact a true active-side worklist.

#### Correctness

The safety condition is based on the full `u_d` row, not on `A=sum_s u_d[s]`.
Using `A == 0` would be unsafe because positive and negative `u_d` entries can
cancel.

Commands run:

```text
python -m py_compile gpurec/core/backward.py gpurec/core/kernels/wave_backward.py tests/kernels/test_dts_backward_accum_kernel.py
pytest -q tests/kernels/test_dts_backward_accum_kernel.py
GPUREC_DTS_PIBAR_UD_SKIP_ZERO_SIDES=1 pytest -q tests/gradients/test_fd_all_modes.py::test_analytic_matches_fd
GPUREC_DTS_PIBAR_UD_SKIP_ZERO_SIDES=1 pytest -q tests/gradients/test_autograd_bridge.py
```

Results:

```text
tests/kernels/test_dts_backward_accum_kernel.py: 35 passed
finite-difference analytic-vs-FD test: 6 passed
autograd bridge tests: 15 passed
```

Two focused tests were added:

- direct from-`u_d` Pibar tree parity with zero side rows;
- full staged DTS `u_d` output followed by from-`u_d` Pibar tree parity.

Both tests run for fp32/fp64 and with/without the active row mask.

#### Exact-zero side sparsity

Stats script:

```text
FAMS=50 REPS=1 WARMUPS=1 python /tmp/gpurec_profile/prop4_side_sparsity.py
```

For the 50-family benchmark:

```text
total_sides         503358
parent_active_sides 483372
exact nonzero u_d   344227  (68.39% of all sides)
exact nonzero u_d among parent-active sides: 71.21%
```

Threshold-only diagnostic, not used by the exact implementation:

```text
threshold > 1e-12: active u_d rows 32.47%
threshold > 1e-09: active u_d rows 32.24%
threshold > 1e-06: active u_d rows 14.15%
threshold > 1e-04: active u_d rows  1.06%
```

This means exact skipping removes about one third of side programs.  The much
larger apparent threshold sparsity would require approximate derivative pruning
and therefore belongs to a different proposal.

Largest split waves:

```text
wave  n_splits  sides  parent-active  exact-active  active>1e-6
44    42155     84310  83842          48436         118
43    36200     72400  71256          40388         92
3     27023     54046  54046          54046         6422
46    24229     48458  48394          26244         55
45    23345     46690  46518          27004         49
```

The exact-zero pattern is wave-dependent.  Early waves can be completely
nonzero; later high-fanout waves often have only about 54-57% exact-active side
rows.

#### Wall-time benchmark

The clean comparison uses one already-built 50-family model and alternates the
environment flag inside the same process:

```text
GPUREC_DTS_BACKWARD_ACCUM_IMPL=direct
FAMS=50
fixed_iters_Pi=6
max_wave_size=32768
dtype=fp32
```

Interleaved event-timed backward-only results:

```text
skip_zero_sides=0: mean 114.073 ms, median 113.889 ms, min 113.621 ms
samples: 114.312, 113.753, 113.990, 113.789, 114.973, 113.621

skip_zero_sides=1: mean 110.636 ms, median 110.598 ms, min 110.383 ms
samples: 110.748, 110.605, 110.591, 110.383, 111.039, 110.448
```

Net measured gain:

```text
113.889 ms -> 110.598 ms
gain: 3.291 ms, 2.9%
```

The profiling subagent also ran independent non-interleaved medians:

```text
10 families: default 36.045 ms, skip-zero 35.335 ms, gain 0.710 ms
50 families: default 115.312 ms, skip-zero 111.463 ms, gain 3.849 ms
```

An earlier non-interleaved run was noisier because later samples drifted upward,
but the first stable samples agreed with the interleaved result.

#### Nsight Systems breakdown

Nsight Systems reports used:

```text
/tmp/gpurec_profile/prop3_direct50.nsys-rep
/tmp/gpurec_profile/prop4_skipzero_fams50.nsys-rep
```

Kernel-sum comparison for the backward capture:

```text
bucket                                      baseline      skip-zero
_dts_cross_backward_accum_kernel            31.447 ms      31.433 ms
_wave_backward_uniform_kernel               24.006 ms      24.015 ms
_uniform_cross_pibar_vjp_tree_from_ud        24.781 ms      17.754 ms
_pibar_ud_side_active_kernel                  0.000 ms       4.433 ms
_dts_parent_reduced_ge2_stage1_kernel         7.072 ms       7.067 ms
_active_mask_from_rhs_absmax_kernel           2.955 ms       2.943 ms
_dts_eq1_to_rows_kernel                       2.695 ms       2.696 ms
```

The from-`u_d` Pibar tree bucket drops by `7.027 ms`, but the side-activity
scan costs `4.433 ms`, so the Nsight kernel-sum gain is about `2.6 ms`.  The
event-timed interleaved benchmark sees about `3.3 ms`; the difference is within
the noise we usually see between single captured passes and repeated event
timing.

No new host synchronization was introduced.  CUDA API summaries remain dominated
by the existing `cudaStreamSynchronize` calls around the benchmark/autograd
boundaries.  The proposal adds one Triton launch per split wave.

#### Nsight Compute resource use

Representative largest split wave, `84310` side rows, block size `128`.

From-`u_d` Pibar tree, baseline:

```text
duration                    4.21 ms
DRAM read                   about 2.01 GB
DRAM write                  about 1.31 GB
DRAM throughput             78.42%
L2 throughput               74.53%
compute throughput          28.60%
achieved occupancy          98.71%
registers/thread            40
L2 hit rate                 77.65%
```

From-`u_d` Pibar tree, exact-zero side mask:

```text
duration                    2.48 ms
DRAM read                   about 1.16 GB
DRAM write                  about 744.87 MB
DRAM throughput             76.40%
L2 throughput               73.06%
compute throughput          28.15%
achieved occupancy          98.51%
registers/thread            40
L2 hit rate                 77.62%
```

The kernel remains memory-bound and almost fully occupied.  The gain comes from
skipping the bottom-up subtree buffer traffic for inactive side rows; the active
rows have essentially the same resource profile as before.

Side-active scan kernel:

```text
duration                    693.41 us
DRAM read                   about 674.16 MB
DRAM write                  about 2.98 MB
DRAM throughput             96.96%
L2 throughput               42.32%
compute throughput          36.03%
achieved occupancy          98.54%
registers/thread            18
L1 hit rate                 17.72%
L2 hit rate                  0.93%
```

This scan streams `pibar_ud` once and is almost pure DRAM bandwidth.  That is
why the total gain is much smaller than the raw tree-kernel reduction.

Useful NCU launch skips for 50-family follow-up profiles:

```text
skip 4:  84310 side rows, largest hot sparse Pibar launch
skip 5:  72400 side rows, second hot sparse launch
skip 2:  48458 side rows, high-fanout sparse launch
skip 32: 54046 side rows, all-active negative control
```

The same launch skips apply to `_pibar_ud_side_active_kernel`.

#### Why not compact a true worklist yet?

A true device worklist needs both compaction and a launch count.  In Python /
Triton, reading the device-side active count back to size the next launch would
reintroduce a host synchronization.  Launching the compacted worklist over the
full original grid avoids the host sync, but then still launches one program per
original side and adds a compaction pass; it mainly replaces the cheap boolean
early return with an expensive prefix/compact stage.

The current NCU data shows that inactive programs are not the main residual
cost.  Once a side row is inactive, the program returns before the subtree walk.
The expensive part is identifying exact-zero rows by scanning `pibar_ud`.

The more promising next step is to produce `side_active` directly in the DTS
staging kernel while `u_d` is already being computed:

```text
any_l = false
any_r = false
for species tile:
    ud_l = ...
    ud_r = ...
    store pibar_ud_l, pibar_ud_r
    any_l |= any(abs(ud_l) != 0)
    any_r |= any(abs(ud_r) != 0)

store side_active_l = any_l
store side_active_r = any_r
```

That would remove the `4.433 ms` side scan bucket.  If the DTS kernel cost did
not grow much, the upper bound for this exact proposal would become close to the
full `7.0 ms` tree-kernel saving.  It needs separate implementation because
both the direct DTS staging kernel and the parent-tiled DTS staging kernel need
an optional side-active output.

If a true compact worklist is tested later, measure four components separately:

```text
1. side_active or side_absmax production
2. device compaction into side_ids
3. compact/worklist Pibar tree
4. full backward wall time
```

A production path should avoid copying the active count to host.  A diagnostic
upper bound may copy the count and launch exactly `active_count` rows, but that
timing should be labeled as a non-production bound because it includes a host
decision point.

Acceptance criteria for a future compact worklist should be stricter than for
the boolean early-return path:

```text
50-family median: at least 4 ms faster than default
incremental gain over exact-zero mask: at least 1.5 ms
construction cost: <= 1 ms if side activity is fused into DTS
all-active launch skip 32: no material regression
exact-zero parity: exact; thresholded mode: finite-difference-gated
```

Supervisor decision: keep this exact-zero skip path as an opt-in experiment for
now.  It is correct and measurably faster on 50 families, but enabling it by
default should wait until the side mask is produced in DTS rather than by a
separate streaming scan.

### Proposal 4 follow-up: side activity produced by direct DTS

After committing the scan-based exact-zero test, I tried the direct refinement
suggested by the profile: produce `side_active` inside the direct DTS staging
kernel while it already has `u_d` in registers.

Implementation:

```text
sum_ud_l = 0
sum_ud_r = 0
side_nonzero_l = 0
side_nonzero_r = 0

for species tile:
    ud_l = vd2 * inv_denom_l
    ud_r = vd1 * inv_denom_r

    store pibar_ud[left_side, tile] = ud_l
    store pibar_ud[right_side, tile] = ud_r

    sum_ud_l += sum(ud_l)
    sum_ud_r += sum(ud_r)
    side_nonzero_l += any(abs(ud_l) != 0)
    side_nonzero_r += any(abs(ud_r) != 0)

store pibar_A[left_side] = sum_ud_l
store pibar_A[right_side] = sum_ud_r
store side_active[left_side] = side_nonzero_l != 0
store side_active[right_side] = side_nonzero_r != 0
```

`uniform_cross_pibar_vjp_tree_from_ud_fused` now accepts a caller-supplied
`side_active` tensor.  The normal backward direct-DTS path passes this tensor
when `GPUREC_DTS_PIBAR_UD_SKIP_ZERO_SIDES=1`, so the separate
`_pibar_ud_side_active_kernel` scan is avoided.  If no side-active tensor is
supplied, the wrapper still has the scan fallback used by the first experiment.

Scope:

- implemented for the direct DTS staging kernel, which is the default backward
  accumulation path;
- the parent-tiled DTS experiment still falls back to the scan path if it is
  manually enabled together with side skipping.

Additional correctness checks:

```text
pytest -q tests/kernels/test_dts_backward_accum_kernel.py
GPUREC_DTS_PIBAR_UD_SKIP_ZERO_SIDES=1 pytest -q tests/gradients/test_fd_all_modes.py::test_analytic_matches_fd
GPUREC_DTS_PIBAR_UD_SKIP_ZERO_SIDES=1 pytest -q tests/gradients/test_autograd_bridge.py
```

Results:

```text
DTS backward accumulation tests: 35 passed
finite-difference analytic-vs-FD test: 6 passed
autograd bridge tests: 15 passed
```

The staged DTS test also checks that the in-kernel side-active output exactly
matches:

```text
pibar_ud.abs().amax(dim=1) != 0
```

#### Wall-time after fusing side activity

Interleaved backward-only timing, one already-built model, same direct path:

```text
10 families
skip_zero_sides=0: median 35.941 ms, mean 36.237 ms, min 35.689 ms
skip_zero_sides=1: median 34.354 ms, mean 34.407 ms, min 33.893 ms
gain: 1.587 ms, 4.4%

50 families
skip_zero_sides=0: median 113.688 ms, mean 113.699 ms, min 113.078 ms
skip_zero_sides=1: median 107.294 ms, mean 107.157 ms, min 106.224 ms
gain: 6.394 ms, 5.6%
```

Compared with the scan-based version, the 50-family median improved from about
`110.6 ms` to `107.3 ms`.  The remaining difference between the `7.0 ms`
Nsight tree-kernel saving and the `6.4 ms` event-timed wall gain is mostly
normal run-to-run noise plus a very small amount of extra scalar work inside
DTS.

#### Nsight Systems after fusing side activity

Report:

```text
/tmp/gpurec_profile/prop4_fusedmask_fams50.nsys-rep
```

Kernel-sum comparison:

```text
bucket                                      baseline      scan-mask     fused-mask
_dts_cross_backward_accum_kernel            31.447 ms      31.433 ms      31.427 ms
_wave_backward_uniform_kernel               24.006 ms      24.015 ms      24.011 ms
_uniform_cross_pibar_vjp_tree_from_ud        24.781 ms      17.754 ms      18.148 ms
_pibar_ud_side_active_kernel                  0.000 ms       4.433 ms       0.000 ms
_dts_parent_reduced_ge2_stage1_kernel         7.072 ms       7.067 ms       7.053 ms
_active_mask_from_rhs_absmax_kernel           2.955 ms       2.943 ms       2.959 ms
_dts_eq1_to_rows_kernel                       2.695 ms       2.696 ms       2.695 ms
```

The important point is that DTS did not get slower in the captured pass.  The
side-active bookkeeping reuses values already computed for `pibar_ud`, and the
extra final boolean stores are tiny relative to the existing staged `u_d`
traffic.  The scan kernel disappears entirely.

#### Nsight Compute: DTS impact

Representative direct DTS launch, skip `4`, grid `42155` split rows.

```text
metric                         baseline direct DTS   fused-mask direct DTS
duration                       5.224 ms              5.216 ms
DRAM throughput                63.30%                63.39%
compute throughput             24.83%                28.85%
registers/thread               96                    96
theoretical occupancy          41.67%                41.67%
achieved occupancy             41.53%                41.52%
achieved active warps/SM        19.93                 19.93
```

The side-active reduction slightly increases the compute-side instruction mix,
but it does not change register pressure, occupancy, memory throughput, or
duration for this hot launch.

Current decision: the direct fused-mask variant meets the acceptance criteria
for exact-zero skipping.  It remains behind
`GPUREC_DTS_PIBAR_UD_SKIP_ZERO_SIDES=1` for now so the default path is unchanged
until we decide whether to also cover the parent-tiled experiment and enable the
optimization by default.

## Proposal 5: reduce Pibar tree level-padding and species-topology traffic

The staged Pibar tree kernel is now mostly a tree-buffer correction:

```text
for level in range(N_LEVELS):
    for p_start in range(0, MAX_LEVEL_WIDTH, BLOCK_S):
        parent = level_parents[level, p_offs]
        parent_val += child1_val + child2_val
        store parent_val
```

This representation is simple, but it pads every level to `MAX_LEVEL_WIDTH`.
If the species tree is unbalanced, many lanes are masked off. Even in a
balanced tree, every split side reloads the same `level_parents`, `sp_child1`,
and `sp_child2` metadata. The current from-`u_d` tree kernel is memory-bound,
so topology and tree-buffer layout still matter.

The first step is instrumentation:

```text
level_widths[level]
padding_work = N_LEVELS * MAX_LEVEL_WIDTH - sum(level_widths)
padding_fraction = padding_work / (N_LEVELS * MAX_LEVEL_WIDTH)
```

If padding is significant, test a compact level layout:

```text
level_nodes_flat = concat(nodes_at_level_0, nodes_at_level_1, ...)
level_offsets[level], level_counts[level]
```

Triton cannot easily make the inner loop trip count dynamic without still
looping to a compile-time maximum, so there are two realistic implementations:

1. Generate a small family of specialized Triton kernels for the observed
   species tree shape, with per-level widths baked as constexprs.
2. Write this one kernel in CUDA/C++ if the level-width padding is confirmed to
   be a first-order cost.

This is more suitable for CUDA than most of the current kernels because the
algorithm is tree-topology-specific and not a regular row-wise map. The
maintenance cost is also bounded: one specialized Pibar tree-correction kernel,
not a wholesale rewrite.

Expected gain:

- if padding is low, no gain;
- if padding is high, target `2-5 ms` out of the `25.618 ms` Pibar tree bucket;
- this can also reduce instruction count and barrier pressure without changing
  the DTS staged dataflow.

Risks:

- prefix-denominator work already failed because extra passes and barriers
  dominated; this proposal must only change the bottom-up correction layout,
  not reintroduce denominator construction;
- generated Triton variants add complexity;
- CUDA/C++ may be justified only after padding is quantified.

Profiling gate:

- report padding fraction and actual active parent nodes per level;
- NCU should show lower executed instructions and lower DRAM/L2 traffic in the
  from-`u_d` tree kernel;
- Nsys must show the Pibar tree bucket falling without a new setup kernel
  erasing the win.

### Proposal 5 test report: compact per-level Pibar tree traversal

The 3-agent + supervisor pass tested the padding hypothesis directly.  The
species tree used by `test_trees_1000` has `S = 1999` species nodes and `999`
internal parent nodes.  The current padded layout has `22` bottom-up levels with
`MAX_LEVEL_WIDTH = 335`, so each split side scans `22 * 335 = 7370` level slots
even though only `999` are real parent nodes:

```text
level widths:
335, 203, 125, 90, 63, 46, 36, 26, 21, 14, 10, 9, 5, 4, 2, 2, 2, 2, 1, 1, 1, 1

active parent nodes: 999
padded slots per side: 7370
padding slots per side: 6371
padding fraction: 86.45%
```

Across split waves, this means the Pibar tree correction repeatedly executes a
large amount of masked topology work:

| case | split waves | splits | split sides | real parent visits | padded parent slots |
|---|---:|---:|---:|---:|---:|
| 10 families | 44 | 83,135 | 166,270 | 166,103,730 | 1,225,409,900 |
| 50 families | 46 | 402,275 | 804,550 | 803,745,450 | 5,929,533,500 |

The implementation is opt-in with
`GPUREC_DTS_PIBAR_UD_COMPACT_LEVELS=1`.  It precomputes and caches four device
arrays from the species topology:

```text
compact_level_ptr[level]      -> flat start offset
compact_level_parents[offset] -> parent node id
compact_level_child1[offset]  -> first child id
compact_level_child2[offset]  -> second child id
```

The compact kernel preserves the per-level barrier structure, but it scans only
the active parent range for each level:

```text
for level in levels:
    start = compact_level_ptr[level]
    end = compact_level_ptr[level + 1]
    for off in range(start, end, BLOCK_S):
        parent = compact_level_parents[off]
        c1 = compact_level_child1[off]
        c2 = compact_level_child2[off]
        pibar_ud[parent] += pibar_ud[c1] + pibar_ud[c2]
    barrier()
```

There were two implementation stages.  The first compact version flattened only
the parent nodes and still loaded `sp_child1[parent]` and `sp_child2[parent]`
inside the kernel.  That reduced masked parent metadata loads, but the hot
launch remained mostly dominated by the same child-topology loads.  The final
variant also flattens `child1` and `child2`, so each active parent slot now
uses contiguous compact metadata and the kernel no longer performs indexed
child lookups through the full species arrays.

Correctness checks:

```text
python -m py_compile gpurec/core/backward.py gpurec/core/kernels/wave_backward.py
pytest -q tests/kernels/test_dts_backward_accum_kernel.py
  -> 43 passed

GPUREC_DTS_PIBAR_UD_COMPACT_LEVELS=1 \
pytest -q tests/gradients/test_autograd_bridge.py
  -> 15 passed

GPUREC_DTS_PIBAR_UD_SKIP_ZERO_SIDES=1 \
GPUREC_DTS_PIBAR_UD_COMPACT_LEVELS=1 \
pytest -q tests/gradients/test_autograd_bridge.py \
          tests/gradients/test_fd_all_modes.py::test_analytic_matches_fd
  -> 21 passed
```

The kernel tests now compare padded and compact Pibar tree correction for
`fp32` and `fp64`, with and without the active parent mask, and with and without
exact-zero side skipping.  The gradient checks exercise the full autograd path
with the compact path enabled.

End-to-end benchmark, current best path with direct DTS and exact-zero side
skipping enabled:

| case | compact off median | compact on median | median gain |
|---|---:|---:|---:|
| 10 families | `35.142 ms` | `34.035 ms` | `1.107 ms` |
| 50 families | `106.102 ms` | `103.581 ms` | `2.521 ms` |

The result is workload dependent.  With exact-zero side skipping disabled, the
implementation worker measured `36.166 -> 36.455 ms` on 10 families and
`114.576 -> 112.177 ms` on 50 families.  Because the 10-family no-skip case was
a slight regression, the compact path remains behind the environment flag
instead of becoming the default.

Nsight Systems, 50-family backward:

| variant | Pibar from-`u_d` bucket | launches | hottest launch |
|---|---:|---:|---:|
| padded + side skip | `18.099 ms` | 33 | `2.788 ms` all-active negative-control wave |
| compact parents only + side skip | `17.337 ms` | 33 | about `2.41 ms` on the sampled hot wave |
| compact parents+children + side skip | `15.892 ms` | 33 | `2.454 ms` |

The 50-family Pibar bucket therefore drops by about `2.21 ms` versus the
current padded side-skip path.  The full backward pass gain was a little larger
in the benchmark (`2.52 ms` median), which is within normal run-to-run noise
for these captures.

NCU hot-launch comparison for split-wave index 4:

| metric | padded | compact parents only | compact parents+children |
|---|---:|---:|---:|
| kernel duration | `2.479 ms` | `2.369 ms` | `2.213 ms` |
| DRAM read bytes | `1.164625 GB` | `1.164518 GB` | `1.164479 GB` |
| DRAM write bytes | `744.901 MB` | `744.806 MB` | `744.772 MB` |
| global load instructions | `30,464,432` | `29,495,712` | `29,495,712` |
| global reduction instructions | `3,051,468` | `3,051,468` | `3,051,468` |
| global store instructions | `2,228,056` | `2,228,056` | `2,228,056` |
| registers/thread | `40` | `40` | `36` |
| achieved occupancy | `98.38%` | `98.16%` | `97.65%` |
| DRAM throughput | `76.47%` | `78.64%` | `86.15%` |
| compute throughput | `27.62%` | `23.88%` | `25.73%` |

The important observation is that DRAM bytes barely move.  The dominant bytes
are still `pibar_ud` row traffic and atomic additions into `accumulated_rhs`.
The compact path helps by cutting masked topology/control work, reducing global
load instruction count by about `3.2%`, and lowering register pressure from
`40` to `36` registers/thread after child metadata was flattened.  The kernel
is still primarily memory-latency limited: compact traversal makes each launch
shorter, but it does not change the fundamental `pibar_ud` and atomic traffic.

Artifacts:

```text
/tmp/gpurec_profile/prop5_skipzero_fams50.nsys-rep
/tmp/gpurec_profile/prop5_compact_fams50.nsys-rep
/tmp/gpurec_profile/prop5_compact_child_fams50.nsys-rep
/tmp/gpurec_profile/prop5_ncu_pibar_padded_skip4.csv
/tmp/gpurec_profile/prop5_ncu_pibar_compact_skip4.csv
/tmp/gpurec_profile/prop5_ncu_pibar_compact_child_skip4.csv
```

Decision: keep the compact traversal as a validated opt-in path.  It is a real
win for the 50-family target and for the current side-skip configuration, but
it should not be enabled unconditionally until we add a small runtime heuristic
or collect more evidence on small batches.  A reasonable heuristic would enable
it when exact-zero side skipping is active and the product
`n_ws * padding_fraction` is large enough to amortize the extra compact metadata
loads.

## Proposal 6: two-stage `grad_mt` reduction for the staged DTS path

The staged DTS path must still accumulate the vector `grad_mt` contribution:

```text
grad_mt[s] += sum_over_splits(vd1[i, s] + vd2[i, s])
```

Before DTS-to-Pibar staging, the `grad_mt` accumulation variant was tested and
rejected because it added many atomics into only `S=1999` species lanes. After
staging, however, the default path effectively has to do this work somewhere,
because `grad_Pibar_l/r` are no longer materialized for a later PyTorch
`sum(dim=0)`.

The current staged kernel does the vector accumulation directly with atomics.
The cost is visible in the proposal-5 NCU comparison:

```text
old DTS accum:     78 regs/thread, 49.7% occupancy, 16.02 M global RED ops
staged DTS accum:  96 regs/thread, 41.5% occupancy, 18.72 M global RED ops
```

The goal is to keep the staged Pibar win while taking vector `grad_mt` pressure
out of the already register-heavy DTS program.

A split-tiled two-stage reduction would look like this:

```text
stage 1, inside a parent-tiled or split-tiled DTS kernel:
    for split in split_tile:
        compute vd1, vd2
        local_mt[species_block] += vd1 + vd2
    store mt_partial[tile_id, species_block]

stage 2:
    reduce mt_partial[:, s] over tile_id
    add to grad_mt[s]
```

For the largest documented split wave, a tile size of `64` gives:

```text
ceil(42155 / 64) * 1999 * 4 bytes ~= 5.3 MB of partials
```

That is tiny compared with the `pibar_ud` staging buffer and much smaller than
the full `grad_Pibar_l/r` materialization that staging removed.

Why this is different from the rejected `grad_mt` atomics:

- the rejected path atomically added every split/species lane into `S` global
  addresses;
- this path reduces many splits locally first, then writes one partial per
  split tile and species lane;
- it is most naturally paired with Proposal 3's parent/split tiling, which
  already loops over multiple splits per program.

Expected gain:

- target is the extra register/RED pressure introduced by DTS-to-Pibar staging;
- plausible `1-4 ms` at 50 families if it lowers registers or MIO/RED stalls
  in `_dts_cross_backward_accum_kernel`;
- if implemented as an entirely separate pass that rereads large DTS values, it
  will likely lose.

Risks:

- if the partial-reduction kernel is memory-bound or launch-heavy, it may erase
  the savings;
- moving `grad_mt` summation changes fp32 reduction order, so exact bitwise
  parity should not be expected;
- this should be tested after or together with parent-tiled DTS, not as a
  blind extra pass on the current split-major kernel.

Profiling gate:

- NCU should show lower global RED operations and preferably lower registers in
  the DTS kernel;
- Nsys should show the DTS bucket plus the new reduction bucket lower than the
  current DTS bucket;
- gradient differences should match the expected reordered-fp32 tolerance and
  pass finite-difference checks.

## Proposal 7: scratch-buffer pooling for large wave temporaries

The hot wrappers allocate several large temporary tensors per wave:

- self-loop `v_k`, `aw*`, `spec_buf`, `term_buf`;
- DTS staged `pibar_ud[2 * n_splits, S]`;
- `pibar_A[2 * n_splits]`;
- `dts_r[W, S]`;
- sometimes full `dts_term[n_splits, S]`.

Most `torch.empty` calls are handled by the PyTorch caching allocator, so this
is not expected to be a large kernel-time win. However, the current backward
still has a nontrivial gap between summed kernel time and event time, and the
large buffers drive peak memory. A small scratch pool could help larger
batches or reduce allocator overhead variance.

Implementation:

```text
At backward setup:
    allocate max-sized scratch buffers for the largest wave and largest split wave.
Per wave:
    pass views/slices into Triton wrappers.
```

Expected gain:

- probably less than `1 ms` at 50 families;
- more valuable for 1000-tree scheduling because it bounds peak memory and
  avoids allocator churn;
- can be combined with Proposal 2, where avoiding `dts_term` may reduce the
  required scratch pool substantially.

Risks:

- buffer lifetime mistakes are easy because later kernels consume earlier
  scratch;
- overlapping views must not alias outputs that are simultaneously live;
- this should not be attempted before a liveness table is written for each
  wave phase.

## Proposal 8: family/chunk-aware pruning for very large batches

Whole-wave pruning weakens as more families are batched together. A global wave
is active if any family has an active row in that wave. With 1000 trees, most
global waves may remain active even if most families are locally inactive.

The current 50-family profile still benefits from host whole-wave pruning, but
larger batches likely need a second granularity:

```text
global wave
    -> chunks of rows, preferably family-contiguous or active-row-contiguous
    -> split chunks for high-fanout parents
```

This interacts with memory limits. The goal is not to return to tiny per-tree
waves. The goal is to keep chunks large enough for occupancy while letting
inactive families drop out.

Possible schedule:

```text
For each static wave:
    build active row mask
    if active rows are sparse by family:
        compact active row chunks on device
        process chunks with fixed max rows/splits
    else:
        process the full global wave
```

This should be considered after Proposal 1 gives real row masks inside active
waves. It is likely more important for 1000-family throughput than for the
50-family profile.

## Proposed implementation order

The order below maximizes learning per engineering hour.

1. **Remove unused fused-path active-index work and fix pruning statistics.**
   This should be exact and low risk.
2. **Instrument active rows, active splits, active split sides, DTS fanout, and
   Pibar level-padding.** Do this before building any more schedulers.
3. **Test hybrid host-wave skip plus fused row masks.** Preserve whole-wave
   skipping; do not repeat fixed-schedule device pruning.
4. **Implement parent-reduced DTS forward recompute for high-fanout ge2
   waves.** This attacks the `dts_fused + seg_lse` materialization.
5. **Prototype parent-tiled DTS backward accumulation on high-fanout waves.**
   Keep outputs unchanged first; only then add split-side stats.
6. **Move staged-path `grad_mt` into split-tile partial reductions if parent
   tiling exposes a cheap local reduction.** Do not add a blind full pass.
7. **If side sparsity is high, build staged-Pibar split-side worklists.**
   Start with exact-zero mode.
8. **If Pibar level padding is high, specialize the from-`u_d` tree correction
   layout.** Consider CUDA/C++ only for this one topology-specific kernel.
9. **Pool scratch buffers once the winning dataflow is clearer.**

## Measurement plan

Every proposal above should be measured with the same discipline as the earlier
passes:

```text
1. Correctness:
   - old-vs-new loss and theta-gradient parity on 3, 10, and 50 families;
   - focused Triton kernel parity tests;
   - finite-difference/autograd bridge tests;
   - fp32 and fp64 coverage when the kernel supports both.

2. Timing:
   - CUDA-event backward-only timings outside Nsight;
   - at least two run orders for small expected gains;
   - peak memory.

3. Nsight Systems:
   - total backward interval;
   - summed kernel buckets;
   - kernel launch count;
   - D2H sync/copy counts;
   - D2D copy volume;
   - whether time moved into a new helper kernel.

4. Nsight Compute:
   - representative largest launches;
   - DRAM read/write bytes;
   - register count and occupancy;
   - eligible warps per scheduler;
   - global RED operations;
   - excessive L2 sectors;
   - stall mix.
```

## Expected upper bound

The remaining 50-family median is about `121 ms`. A direct language rewrite to
CUDA is unlikely to cut this in half because the hot work is memory movement,
tree reductions, and split scheduling. But a third optimization pass still has
room:

| Direction | Plausible gain at 50 families | Notes |
|---|---:|---|
| Remove unused active-index/sync work | `0.5-2 ms` | exact cleanup |
| Hybrid row masks inside active waves | `0-8 ms` | depends on measured partial-wave inactivity |
| Parent-reduced DTS forward | `4-8 ms` | avoids large `[n_splits, S]` materialization for high-fanout waves |
| Parent-tiled DTS backward | `3-6 ms` | if parent reloads are significant and register pressure stays controlled |
| Two-stage `grad_mt` reduction | `1-4 ms` | only if it lowers staged DTS RED/register pressure |
| Split-side Pibar worklists | `0-5 ms` | only if side sparsity is real |
| Pibar tree layout specialization | `0-5 ms` | only if level padding/topology traffic is large |
| Scratch pooling | `<1 ms` at 50 families | more useful for memory and larger batches |

A realistic third-pass target is therefore about `10-20 ms` on the 50-family
backward if the parent-reduced DTS and parent-tiled DTS ideas work. A more
conservative target is `5-10 ms` if pruning and split-side sparsity are weak.
The hard upper bound is much lower than the full `109 ms` summed kernel time:
most of that work is still the real dense species-by-row math that must be
performed.
