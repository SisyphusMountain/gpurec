# Uniform Forward + Backward Full Pipeline Profile

Date: 2026-05-05.

Scope: optimized global/uniform mode on `tests/data/test_trees_1000`,
fp32, `S=1999`, fixed `Pi` iterations `6`, RTX 4090.  The goal of this
round was to build and measure a full training-step path, not independent
forward-only or backward-only microbenchmarks.

## Implemented Path

The new benchmark harness is:

```text
profiling/bench_uniform_forward_backward_pipeline.py
```

It uses the same optimized kernels as the current uniform forward and backward
paths, but streams resident family chunks so the full 1000-family training pass
fits in memory:

```python
E, params = E_fixed_point_once(theta)
loss = 0
pi_adjoint_acc = zero_like_E_and_params()

for chunk in family_chunks:
    Pi, Pibar, Pibar_row_max = Pi_wave_forward(
        chunk,
        E,
        params,
        pibar_mode="uniform",
        fixed_iters=6,
        need_pibar=True,
    )
    loss += root_log_likelihood(Pi, E, chunk.roots)

    dPi = Pi_wave_backward(
        chunk,
        Pi,
        Pibar,
        E,
        params,
        roots_as_cpu_ints,
        uniform_pibar_row_max=Pibar_row_max,
    )
    pi_adjoint_acc += dPi
    release_chunk_state()

grad_theta = E_adjoint_and_theta_vjp_once(pi_adjoint_acc, E, params)
```

The harness prints active optimized-path guards and fails in strict mode if it
falls back to generic self-loop code.  It also emits NVTX ranges for Nsight
Systems captures.

## Code Changes

1. Added the full-pipeline harness above.
2. Changed the harness default `--family-chunk-size` from `50` to `75`, based
   on the clean sweep below.
3. Added CPU-side root-ID precomputation in the harness.  This avoids passing a
   CUDA tensor into the root initialization loop in `Pi_wave_backward`.
4. Hardened `gpurec/core/backward.py`: if another caller still passes CUDA
   root IDs, `Pi_wave_backward` now materializes them once on the CPU instead of
   doing one scalar synchronization per root.

The root-ID change is intentionally small:

```python
if torch.is_tensor(root_clade_ids_perm):
    root_ids_iter = root_clade_ids_perm.detach()
    if root_ids_iter.device.type != "cpu":
        root_ids_iter = root_ids_iter.cpu()
    root_ids_iter = root_ids_iter.tolist()
else:
    root_ids_iter = root_clade_ids_perm

for r in root_ids_iter:
    r = int(r)
    accumulated_rhs[r] = root_adjoint(Pi_star_wave[r])
```

## Correctness

Local checks after the root-ID changes:

| Check | Result |
|---|---:|
| `python -m py_compile gpurec/core/backward.py profiling/bench_uniform_forward_backward_pipeline.py` | pass |
| 2-family chunked vs unchunked harness | pass |
| 2-family loss difference | `0.00000000e+00` |
| 2-family gradient max abs difference | `4.22668457e-03` |
| 2-family gradient relative difference | `1.80128216e-05` |
| `test_gradcheck_global_uniform_small` | pass |
| `test_global_uniform_fused_high_neumann_matches_tight_generic_gmres` | pass |

The 2-family smoke was run concurrently with the pytest process, so its timing
is not used as a performance number.  Its correctness verdict is valid.

## Active Optimized Path

The full 1000-family runs reported:

```text
optimized_path_verdict verdict optimized generic_pytorch_fallback 0
strict_optimized_verdict pass
saved_full_state 1
pibar_row_max_saved 1
root_rows_only 0
generic_self_loop_calls 0
```

Important active flags:

```text
GPUREC_FORWARD_LEAF_INDEX=1
GPUREC_UNIFORM_PINGPONG=1
GPUREC_FORWARD_PARENT_REDUCED_DTS=1
GPUREC_FORWARD_PARENT_REDUCED_DTS_IMPL=tiled
GPUREC_FORWARD_PARENT_REDUCED_DTS_GE2_ONLY=1
GPUREC_KERNELIZED_ACTIVE_MASK=1
GPUREC_KERNELIZED_BACKWARD_DTS=1
GPUREC_FUSED_DTS_BACKWARD_ACCUM=1
GPUREC_FUSED_CROSS_PIBAR_VJP=1
GPUREC_FUSED_CROSS_PIBAR_VJP_IMPL=tree
GPUREC_FUSED_UNIFORM_BACKWARD=1
GPUREC_BACKWARD_LEAF_INDEX=1
GPUREC_FUSED_WAVE_PARAM_ACCUM=1
GPUREC_DTS_PIBAR_UD_FUSION=1
GPUREC_DTS_PIBAR_UD_SKIP_ZERO_SIDES=1
GPUREC_DTS_PIBAR_UD_COMPACT_LEVELS=1
GPUREC_DTS_GRAD_MT_TWO_STAGE=1
GPUREC_BACKWARD_PARENT_REDUCED_DTS=tiled
```

## 1000-Family Benchmark

Times are CUDA-event medians.  Preprocessing and layout construction are
reported separately by the harness and are not included in `total_ms`.

| Chunk size | Chunks | Forward ms | Backward ms | Total ms | Peak GiB | Loss | Grad norm | Status |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 50 | 20 | `2392.825` | `6609.288` | `9002.112` | `10.620` | `2157097.0000` | `1.76743266e5` | safe but slower |
| 75 | 14 | `2348.796` | `6371.309` | `8720.106` | `15.595` | `2157097.2500` | `1.76742516e5` | pre root-ID fix |
| 75 | 14 | `2343.716` | `6341.228` | `8684.944` | `15.595` | `2157097.2500` | `1.76742516e5` | best current default |
| 100 | 10 | `2353.369` | `6614.622` | `8967.991` | `20.396` | `2157097.0000` | `1.76763984e5` | high memory, slower |
| 100 | 10 | `2356.362` | `6583.314` | `8938.743` | `20.396` | `2157097.0000` | `1.76763984e5` | after root-ID fix |
| 125 | 8 | OOM | OOM | OOM | OOM | n/a | n/a | not viable |

Interpretation:

- Chunk `75` is the best current default.  It is `317.168 ms` faster than the
  measured chunk-50 baseline, a `3.52%` total improvement, while still leaving
  much more memory headroom than chunk `100`.
- The root-ID precompute saved `35.162 ms` at chunk `75` and `29.248 ms` at
  chunk `100`.  That is small but real, and the production guard prevents other
  callers from reintroducing per-root scalar syncs.
- Chunk `100` has fewer launches and fewer waves, but it is slower because the
  backward kernels become larger and more memory/L2 pressured.  It also peaks
  at `20.396 GiB`, leaving too little margin.
- Chunk `125` fails while allocating full `Pibar`; it tried to allocate about
  `6 GiB` with the process already near the practical resident-memory limit.
- `GPUREC_BACKWARD_SCRATCH_POOL=1` at chunk `75` measured `8810.410 ms`.
  That did not reproduce a meaningful speedup, so the scratch pool remains
  opt-in.

The best 1000-family phase split is:

| Component | Median ms | Share of total |
|---|---:|---:|
| E fixed point | `4.093` | `0.05%` |
| Pi forward | `2339.623` | `26.94%` |
| Pi backward | `6329.040` | `72.87%` |
| E adjoint/theta VJP | `12.188` | `0.14%` |
| Total | `8684.944` | `100.00%` |

The combined pass is therefore overwhelmingly a `Pi` wave problem.  The shared
`E` solve and final theta VJP are already negligible at this scale.

## Nsight Systems Slice

Profile:

```text
/tmp/gpurec_full_pipeline_parent_profiles/nsys_uniform_fb_f50_c50_rootlist.nsys-rep
```

Workload: first 50 families, one chunk, same optimized path.  Harness event
timing:

| Component | CUDA-event ms |
|---|---:|
| E fixed point | `7.494` |
| Pi forward | `117.690` |
| Forward total | `125.184` |
| Pi backward | `329.051` |
| E adjoint/theta VJP | `16.996` |
| Backward total | `346.047` |
| Full pass | `471.231` |
| Peak memory | `9.823 GiB` |

Top GPU kernels in the captured interval:

| Kernel | GPU ms | Share | Launches | Notes |
|---|---:|---:|---:|---|
| `_wave_backward_uniform_kernel` | `256.368` | `58.1%` | `36` | dominant backward self-loop kernel |
| `_wave_step_uniform_kernel` | `78.223` | `17.7%` | `294` | forward and E fixed-point wave step |
| `_dts_cross_backward_accum_kernel` | `27.308` | `6.2%` | `33` | backward DTS accumulation |
| `_uniform_cross_pibar_vjp_tree_from_ud_compact_kernel` | `16.181` | `3.7%` | `33` | compact tree Pibar VJP |
| `_dts_parent_reduced_ge2_stage1_kernel` | `15.007` | `3.4%` | `13` | forward parent-reduced DTS |
| `_dts_fused_kernel` | `11.326` | `2.6%` | `63` | forward DTS |
| `_wave_pibar_uniform_parent_kernel` | `10.486` | `2.4%` | `49` | forward Pibar |

CUDA API time is dominated by harness synchronization:

| API | Time ms | Calls | Notes |
|---|---:|---:|---|
| `cudaStreamSynchronize` | `262.267` | `219` | mostly CUDA event timing boundaries |
| `cudaDeviceSynchronize` | `150.901` | `8` | profiler and phase synchronization |
| kernel launch APIs | `8.400` | `4356` | launch overhead is much smaller than kernel time |

These synchronization totals should not be interpreted as unavoidable
production overhead.  The harness intentionally synchronizes around phases so
we can measure them.  The useful signal is that, even with those syncs, kernel
time is dominated by a few custom Triton/CUDA kernels rather than by PyTorch
fallback work.

## Nsight Compute Resource Use

Representative NCU captures from the 50-family workload:

| Kernel | Duration | Memory throughput | DRAM throughput | L2 throughput | SM throughput | Occupancy | Registers/thread | Main bottleneck |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `_wave_backward_uniform_kernel` | `58.549 ms` | `73.64%` | `21.15%` | `73.64%` | `18.15%` | `99.10%` | `40` | L2/cache traffic and scoreboard stalls |
| `_wave_step_uniform_kernel` | `1.307 ms` | `83.00%` | `38.37%` | `57.49%` | `83.00%` | `98.11%` | `40` | already well utilized |
| `_dts_cross_backward_accum_kernel` | `0.457 ms` | `71.53%` | `71.53%` | `52.28%` | `34.45%` | `41.42%` | `96` | DRAM bound and register limited |

Additional NCU details:

- `_wave_backward_uniform_kernel` has L2 hit rate `92.48%`, but still drives
  `73.64%` L2 throughput.  It is not starved for occupancy; it is moving and
  revisiting a lot of state.
- `_wave_backward_uniform_kernel` spends about `40.9%` of cycles between
  issued instructions stalled on L1TEX scoreboard dependencies.  More blocks
  will not fix this; the kernel is already at `99.10%` achieved occupancy.
- `_wave_step_uniform_kernel` has no local spilling and is balanced at roughly
  `83%` compute and memory throughput.  This is not the first target for
  another large gain.
- `_dts_cross_backward_accum_kernel` uses `96` registers/thread, limiting
  theoretical occupancy to about `41.67%`.  It is also DRAM bound.  Reducing
  register pressure or splitting the heaviest path could help, but this kernel
  is only `6.2%` of the f50 profile.

## Current Bottlenecks

1. Backward self-loop kernel memory traffic.  This is the dominant kernel and
   is L2/cache-scoreboard limited, not launch-limited or occupancy-limited.
2. Forward wave step is large but already close to balanced hardware use.  A
   major speedup here probably requires less work or less data movement, not a
   simple launch-parameter tweak.
3. DTS backward accumulation is a secondary bottleneck.  Its high register use
   and DRAM pressure make it worth revisiting, but the maximum isolated upside
   is much smaller than the self-loop kernel.
4. Chunk scheduling trades launch count against memory pressure.  Larger chunks
   do not monotonically improve time.  Chunk `75` is currently the best balance
   on this GPU.

## Next Optimization Proposals

1. Reduce `_wave_backward_uniform_kernel` state traffic.
   - Audit the loads of `Pi`, `Pibar`, active masks, and saved row maxima.
   - Prefer recomputation for cheap scalar terms if it avoids extra L2 traffic.
   - Try a variant that keeps the hottest row-local values in registers/shared
     memory across the D/S/T contributions.

2. Split or specialize `_wave_backward_uniform_kernel` by wave shape.
   - The current fused kernel handles many cases.  A small set of specialized
     kernels for no-split, light-split, and heavy-split waves may reduce
     useless branches and memory reads.
   - Measure this carefully; extra launches can lose if the specialized kernels
     do not reduce memory traffic enough.

3. Retune `_dts_cross_backward_accum_kernel`.
   - Try lower-register variants and alternate block sizes.
   - The target is not just occupancy; NCU says it is DRAM bound, so the variant
     must reduce bytes moved or improve coalescing.

4. Add chunk auto-selection.
   - Use a dry-run layout pass to estimate peak resident memory from
     `max_chunk_clades * S * sizeof(dtype) * saved_state_count`.
   - Choose the largest chunk in a small candidate set that leaves a fixed
     memory margin, then prefer the empirically faster chunk if there is a tie.
   - On this GPU and dataset, that rule should pick `75`, not `100`.

5. Remove benchmark-only synchronization from production training.
   - The harness needs phase syncs for measurement.  A production API should
     only synchronize for final scalar extraction or external optimizer
     boundaries.
   - This will mostly reduce wall-time overhead and profiler API noise; it will
     not change the CUDA-event kernel budget.

