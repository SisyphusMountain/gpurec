# Uniform Backward Optimization Proposals, Pass 4

This is a fourth pass over the `pibar_mode="uniform"` backward pass.  It uses
the current committed default after the first three optimization waves, and it
tries to avoid re-proposing ideas that were already measured and rejected in:

- `docs/uniform-backward-fp32-fused-profile.md`
- `docs/uniform-backward-50tree-wave2-profile.md`
- `docs/uniform-backward-third-pass-proposals.md`
- `docs/uniform-backward-pruning-granularity.tex`
- `docs/uniform-backward-optimizations-summary.tex`

The main conclusion is that the remaining wins are no longer simple Triton
branch cleanups.  The hot kernels are moving a few GB per representative launch,
and the useful next experiments should change the memory layout or where
intermediate row vectors live.

## Current reference point

Workload:

```text
dataset: tests/data/test_trees_1000
mode: global
pibar_mode: uniform
dtype: fp32
fixed_iters_Pi: 6
neumann_terms: 3
max_wave_size: 32768
GPU: RTX 4090
```

Current default environment from the benchmark harness:

```text
GPUREC_KERNELIZED_BACKWARD_DTS=1
GPUREC_FUSED_DTS_BACKWARD_ACCUM=1
GPUREC_FUSED_CROSS_PIBAR_VJP=1
GPUREC_FUSED_CROSS_PIBAR_VJP_IMPL=tree
GPUREC_FUSED_UNIFORM_BACKWARD=1
GPUREC_UNIFORM_PINGPONG=1
GPUREC_BACKWARD_LEAF_INDEX=1
GPUREC_FUSED_WAVE_PARAM_ACCUM=1
GPUREC_DTS_PIBAR_UD_SKIP_ZERO_SIDES=1
GPUREC_DTS_PIBAR_UD_COMPACT_LEVELS=1
GPUREC_DTS_GRAD_MT_TWO_STAGE=1
```

Warm CUDA-event timings outside Nsight:

| Families | Clades | Waves | Split rows | Median backward | Peak allocation |
|---:|---:|---:|---:|---:|---:|
| 10 | 66,530 | 45 | 83,135 | `34.021 ms` | `2.696 GB` |
| 50 | 321,930 | 49 | 402,275 | `101.037 ms` | `10.308 GB` |
| 100 | 635,372 | 56 | 793,940 | `185.024 ms` | `17.950 GB` |

Commands:

```bash
source /home/enzo/Documents/git/gpurec/gpurec/.venv/bin/activate
FAMS=50 REPS=7 WARMUPS=4 MAX_WAVE_SIZE=32768 \
  python profiling/proposal8/bench_uniform_backward.py
FAMS=100 REPS=5 WARMUPS=3 MAX_WAVE_SIZE=32768 \
  python profiling/proposal8/bench_uniform_backward.py
```

The 50-family shape is still dominated by large species-by-row kernels:

```text
S=1999, C=321930, waves=49, maxW=32768
leaves=80545, split_rows=402275
largest high-fanout waves:
  wave 44: W=247,  splits=42155, fanout=170.7
  wave 43: W=584,  splits=36200, fanout=62.0
  wave 46: W=39,   splits=24229, fanout=621.3
  wave 45: W=93,   splits=23345, fanout=251.0
```

## Current Nsight Systems breakdown

One warmed 50-family backward was captured with:

```bash
PROFILE_CUDA_API=1 FAMS=50 REPS=1 WARMUPS=3 MAX_WAVE_SIZE=32768 \
  nsys profile --force-overwrite=true \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  -o /tmp/gpurec_profile/fourthpass_bwd50 \
  python profiling/proposal8/bench_uniform_backward.py
```

Nsight overhead makes the profiled event slower than normal timing:

| Metric | Value |
|---|---:|
| CUDA-event backward under Nsight | `112.956 ms` |
| Summed GPU kernel time | `88.757 ms` |
| Kernel launches in capture | about `2918` |
| `cudaLaunchKernel` calls | `2578` |
| `cudaStreamSynchronize` calls | `205` |
| `cudaMemcpyAsync` calls | `527` |
| D2D copy time | `0.389 ms` |
| D2H copy time | `0.128 ms` |

Top GPU kernel buckets:

| Component | Kernel time | Launches | Share of kernel time |
|---|---:|---:|---:|
| DTS backward accumulation | `26.853 ms` | 33 | `30.3%` |
| Self-loop wave backward | `23.789 ms` | 36 | `26.8%` |
| Staged uniform Pibar VJP | `15.937 ms` | 33 | `18.0%` |
| Parent-reduced DTS forward recompute, stage 1 | `7.028 ms` | 6 | `7.9%` |
| Active-row mask construction | `2.959 ms` | 49 | `3.3%` |
| Eq1 DTS forward-to-row copy | `2.700 ms` | 9 | `3.0%` |
| PyTorch fill kernels, many small | `2.194 ms` | 501 | `2.5%` |
| Standard DTS forward recompute | `1.570 ms` | 24 | `1.8%` |
| One large PyTorch fill | `1.313 ms` | 1 | `1.5%` |

The top three custom kernels account for `66.58 ms`, which is `75.0%` of
summed kernel time and about two thirds of the normal 50-family backward wall
time.  Most low-hanging PyTorch reduction work is already gone; the remaining
PyTorch kernels are small individually but numerous.

## Current Nsight Compute samples

Representative launches were chosen from the Nsight Systems GPU trace:

| Kernel | Matching launch | Nsys duration | Shape |
|---|---:|---:|---|
| `_wave_backward_uniform_kernel` | skip 34 | `4.557 ms` | `grid=32768`, `block=256`, `40 regs/thread` |
| `_dts_cross_backward_accum_kernel` | skip 4 | `4.467 ms` | `grid=42155`, `block=128`, `96 regs/thread` |
| `_uniform_cross_pibar_vjp_tree_from_ud_compact_kernel` | skip 32 | `2.461 ms` | `grid=54046`, `block=128`, `36 regs/thread` |
| `_dts_parent_reduced_ge2_stage1_kernel` | skip 3 | `1.915 ms` | `grid=13`, `block=128`, `39 regs/thread` |

NCU commands used `--set detailed --csv --page raw --profile-from-start off`.
The files are:

```text
/tmp/gpurec_profile/fourthpass_ncu_wave.csv
/tmp/gpurec_profile/fourthpass_ncu_dts_accum.csv
/tmp/gpurec_profile/fourthpass_ncu_pibar_compact.csv
/tmp/gpurec_profile/fourthpass_ncu_parent_stage1.csv
```

Key NCU counters:

| Kernel | NCU duration | DRAM read | DRAM write | Memory throughput | DRAM throughput | SM throughput | Occupancy | Main signal |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| self-loop wave | `4.941 ms` | `1.009 GB` | `1.905 GB` | `90.25%` | `58.56%` | `41.53%` | `99.32%` | memory traffic plus global RED |
| DTS accum | `4.532 ms` | `2.017 GB` | `1.319 GB` | `73.08%` | `73.08%` | `34.05%` | `41.51%` | bandwidth, RED, 96-register pressure |
| staged Pibar VJP | `2.457 ms` | `1.302 GB` | `0.836 GB` | `86.37%` | `86.37%` | `25.98%` | `98.66%` | bandwidth from tree scratch |
| parent-reduced DTS stage 1 | `1.865 ms` | `1.467 GB` | `0.013 GB` | `78.79%` | `78.79%` | `15.93%` | `94.55%` | read-only bandwidth |

More detailed signals:

| Kernel | Global RED instructions | Global load instructions | Global store instructions | Long-scoreboard samples | LG throttle samples | MIO throttle samples |
|---|---:|---:|---:|---:|---:|---:|
| self-loop wave | `10.846 M` | `134.775 M` | `22.708 M` | `449,752` | `19,594` | `17,332` |
| DTS accum | `18.571 M` | `36.392 M` | `5.480 M` | `220,882` | `14,192` | `74,958` |
| staged Pibar VJP | `3.405 M` | `32.752 M` | `2.486 M` | `161,333` | `60,251` | `16,044` |
| parent-reduced DTS stage 1 | `0` | `29.543 M` | `0.083 M` | `222,433` | `24,822` | `187` |

Interpretation:

- The self-loop kernel is not occupancy-starved anymore.  It reaches almost
  full achieved occupancy but still writes and rereads row-sized scratch.
- DTS accumulation is constrained by a mix of memory bandwidth, global
  reductions, and register pressure.  The 96-register launch has only about
  `41.5%` achieved occupancy.
- The staged Pibar VJP is the cleanest memory-bound target: it has high
  occupancy, low compute throughput, and spends most of its time transforming
  the staged `pibar_ud` rows through the species tree.
- Parent-reduced forward recompute is read-only and bandwidth-bound.  It is
  already much better than materializing `[n_splits, S]`, but it still rereads
  child rows that the later DTS backward rereads.

## Previously tested paths not worth repeating unchanged

| Idea | Current status |
|---|---|
| Disable pruning or always run a fixed schedule | Still slower.  Current 50-family medians: default `101.037 ms`, `GPUREC_DEVICE_PRUNING=1` `104.954 ms`, `GPUREC_BACKWARD_NO_CPU_PRUNING=1` `105.141 ms`. |
| Child-grouped Pibar VJP by duplicate child rows | Rejected.  The duplicate factor is only about `1.111x` on the 50-family layout, and the grouping reduction was more expensive than the saved tree work. |
| Existing parent-tiled DTS backward prototype | Rejected.  It reduced some parent loads but overlaunched rectangular CTAs and multiplied scalar-reduction atomics by species-block count. |
| No-split leaf specializations inside the current Triton wave kernel | Rejected.  Compiler constexpr specialization already removes the easy branches; low-risk variants increased traffic or were noise. |
| Blind scratch pooling | Rejected as default.  It helped memory accounting more than time.  A narrower zero-elision audit is still different and listed below. |
| More tensor-core work | Not promising for the hot path.  The hot kernels are exp/log, reductions, tree gathers, atomics, and memory streams rather than dense matrix multiply. |

The rest of this document focuses on dataflow changes that are different from
these rejected versions.

## Proposal 0: CUDA shared-memory row kernels for self-loop Neumann

The current self-loop Triton kernel runs one program per clade row.  It cannot
keep an entire `S=1999` row in registers, so it stores row-sized scratch in
global memory:

```text
aw0      = diagonal Neumann coefficient
aw1      = compact Pibar coefficient
aw4      = speciation child-1 coefficient
aw345    = speciation child-2 coefficient
term_buf = current Neumann vector
spec_buf = next Neumann vector
v_k      = accumulated Neumann sum
```

The accepted gather path removed the worst speciation scatter scratch, but the
representative current launch still moves:

```text
read:  1.009 GB
write: 1.905 GB
time:  4.941 ms
```

A CUDA block can hold one row's temporary vectors in shared memory instead:

```cuda
extern __shared__ float sh[];
float* term = sh;          // S floats
float* next = sh + S;      // S floats
float* coeff_diag = ...;   // optional, or recompute

load Pi/Pibar/constants in coalesced tiles
compute coefficients
for iter in 0..NEUMANN_TERMS-1:
    A = block_reduce_sum(term[s] * pibar_coeff[s])
    next[s] = term[s] * diag[s]
            + p_prime[s] * (A - term[s] * pibar_coeff[s])
            + speciation_gather(term, parent[s])
    v_k[s] += next[s]
    swap(term, next)
write v_k once
accumulate or emit parameter-gradient partials
```

For `S=1999`, two fp32 row buffers are only about `16 KB`; even four buffers are
about `32 KB`.  That fits on Ada shared memory.  The key is to make global
memory see each row vector once per logical phase rather than once per Neumann
sub-pass.

Expected gain:

- target `6-12 ms` at 50 families if it removes a meaningful fraction of the
  `23.789 ms` self-loop bucket;
- larger memory benefit at 100 families because wave scratch is part of the
  `17.950 GB` peak;
- likely not achievable in plain current Triton without using enormous
  `BLOCK_S=2048` vectors, which would explode register pressure.

Risks:

- CUDA implementation and testing cost is high;
- shared-memory bank conflicts need explicit layout/padding;
- final parameter reductions still need a plan, otherwise global RED pressure
  remains.

Recommended experiment:

1. Implement only the no-split leaf waves first, because they are the largest
   current self-loop launches and avoid `dts_r`.
2. Keep the old Triton kernel for split waves until parity and NCU are clean.
3. Compare DRAM bytes, global RED bytes, and wall time.  Promote only if the
   representative wave launch drops by at least `20%`.

### Proposal 0 test report: NVRTC CUDA no-split row kernel

Status: implemented as an opt-in prototype, not promoted as the default.

Files:

- `gpurec/core/kernels/wave_backward_cuda.py` contains
  `gpurec_wave_backward_nosplit_uniform_fp32`, compiled at runtime with NVRTC
  through CUDA Python bindings.  This avoids an `nvcc` build dependency.
- `gpurec/core/backward.py` routes only no-split uniform fp32 waves to this
  kernel when `GPUREC_CUDA_SELF_LOOP_NOSPLIT=1`; split waves stay on the
  current Triton path.
- `GPUREC_CUDA_SELF_LOOP_NOSPLIT_CORRECTION=self` is the default and matches
  the current Triton fused semantics.  `tree` is available explicitly and uses
  the full ancestor/subtree correction.

The implementation assigns one CUDA block to one clade row and keeps the row's
Neumann temporaries in shared memory:

```cuda
// one block per no-split row
term[s]  = rhs[s]
vacc[s]  = rhs[s]
diag[s]  = q_D[s] + q_Tlocal[s]
pcoef[s] = q_Pibar[s] * inv_denom[s]
sl1w[s]  = q_spec_child1[s]
sl2w[s]  = q_spec_child2[s]

for iter in 0..neumann_terms-1:
    u[s] = term[s] * pcoef[s]
    A = block_sum_s(u[s])

    // self mode: correction[s] = u[s]
    // tree mode: correction[s] = subtree_sum_s(u)
    next[s] = term[s] * diag[s]
            + p_prime[s] * (A - correction[s])
            + term[parent[s]] * speciation_weight(parent[s] -> s)
    vacc[s] += next[s]
    swap(term, next)

write vacc[s]
atomic_add parameter gradients
```

For `S=1999`, the prototype uses seven fp32 shared-memory row arrays:

```text
7 * 1999 * 4 B = 55,972 B dynamic shared memory per block
```

That removes most global row scratch traffic, but it also limits occupancy to
one CTA per SM on the RTX 4090.

#### Correctness checks

Commands run:

```bash
pytest -q tests/gradients/test_autograd_bridge.py

GPUREC_CUDA_SELF_LOOP_NOSPLIT=1 \
GPUREC_CUDA_SELF_LOOP_NOSPLIT_CORRECTION=self \
pytest -q tests/gradients/test_autograd_bridge.py

GPUREC_CUDA_SELF_LOOP_NOSPLIT=1 \
GPUREC_CUDA_SELF_LOOP_NOSPLIT_CORRECTION=tree \
pytest -q tests/gradients/test_autograd_bridge.py
```

All three runs passed: `15 passed`.

Large-workload parity was checked by running the same 10- and 50-family
`test_trees_1000` model with the current Triton path, CUDA `self`, and CUDA
`tree`, then comparing `model.theta.grad`.

| workload | mode | loss diff | max abs grad diff | max rel grad diff | mean abs grad diff |
|---:|---|---:|---:|---:|---:|
| 10 families | CUDA `self` vs Triton | `0` | `5.615e-3` | `1.459e-5` | `4.598e-3` |
| 10 families | CUDA `tree` vs Triton | `0` | `6.592e-3` | `1.451e-5` | `4.913e-3` |
| 50 families | CUDA `self` vs Triton | `0` | `4.639e-2` | `1.252e-5` | `3.076e-2` |
| 50 families | CUDA `tree` vs Triton | `0` | `4.492e-2` | `1.319e-5` | `3.280e-2` |

The subagent correctness audit also ran a direct synthetic no-split probe with
`S=511` and a nontrivial binary species tree:

- CUDA `self` vs current Triton fused: `v_k` relative error `4.6e-08`;
- CUDA `tree` vs the PyTorch ancestor-corrected reference:
  `v_k` relative error `9.2e-08`;
- CUDA `tree` vs current Triton fused: `v_k` relative error `3.1e-02`.

This confirms the semantic split:

- `self` is the implementation-parity mode for the current Triton fused
  no-split kernel, which uses `p_prime * (A - u[s])`;
- `tree` is the full ancestor-corrected VJP, corresponding to
  `p_prime * (A - subtree_sum[s])`, and is expected to differ from Triton on
  a nontrivial tree.

For that reason the opt-in default was set to `self`.  `tree` remains explicit
for experiments that want the exact ancestor correction instead of strict
current-behavior parity.

#### Event-timed benchmarks

All timings below use `tests/data/test_trees_1000`, `fixed_iters_Pi=6`,
`neumann_terms=3`, `MAX_WAVE_SIZE=32768`, and the same fused backward flags as
the previous pass.  Each row reports warmed CUDA-event backward timing.

| families | mode | median backward | delta vs baseline | peak allocation |
|---:|---|---:|---:|---:|
| 10 | baseline Triton | `33.949 ms` | - | `2.696 GB` |
| 10 | CUDA `self` | `34.561 ms` | `+0.612 ms` slower | `2.069 GB` |
| 10 | CUDA `tree` | `34.727 ms` | `+0.778 ms` slower | `2.069 GB` |
| 50 | baseline Triton | `102.318 ms` | - | `10.308 GB` |
| 50 | CUDA `self` | `101.055 ms` | `-1.263 ms` faster | `9.898 GB` |
| 50 | CUDA `tree` | `107.333 ms` | `+5.015 ms` slower | `9.898 GB` |
| 100 | baseline Triton | `190.158 ms` | - | `17.950 GB` |
| 100 | CUDA `self` | `183.866 ms` | `-6.292 ms` faster | `17.920 GB` |
| 100 | CUDA `tree` | `199.215 ms` | `+9.057 ms` slower | `17.920 GB` |

Interpretation:

- The kernel is too occupancy-limited to help at 10 families.
- It becomes useful when several full no-split waves exist.  The 100-family
  run has four `W=32768` no-split waves and one `W=27896` no-split wave, so the
  shared-memory scratch reduction is amortized better.
- Peak allocation savings are workload-dependent.  They are large when the
  no-split wave scratch contributes to the peak, but small when other split or
  DTS scratch dominates.

#### Nsight Systems, 50 families

The benchmark was profiled with:

```bash
PROFILE_CUDA_API=1 FAMS=50 REPS=1 WARMUPS=4 MAX_WAVE_SIZE=32768 \
nsys profile --force-overwrite=true \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --trace=cuda,nvtx,osrt \
  -o /tmp/gpurec_profile/prop0/nsys_baseline_50 \
  python profiling/proposal8/bench_uniform_backward.py

GPUREC_CUDA_SELF_LOOP_NOSPLIT=1 \
GPUREC_CUDA_SELF_LOOP_NOSPLIT_CORRECTION=self \
PROFILE_CUDA_API=1 FAMS=50 REPS=1 WARMUPS=4 MAX_WAVE_SIZE=32768 \
nsys profile --force-overwrite=true \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --trace=cuda,nvtx,osrt \
  -o /tmp/gpurec_profile/prop0/nsys_cuda_self_50 \
  python profiling/proposal8/bench_uniform_backward.py
```

Single captured backward interval:

| metric | baseline | CUDA `self` |
|---|---:|---:|
| benchmark-reported backward time under NSYS | `114.698 ms` | `111.935 ms` |
| total GPU kernel time | `89.996 ms` | `87.308 ms` |
| kernel launches | `2918` | `2918` |
| `_wave_backward_uniform_kernel` bucket | `36 launches`, `23.746 ms` | `33 launches`, `12.589 ms` |
| `gpurec_wave_backward_nosplit_uniform_fp32` bucket | - | `3 launches`, `8.493 ms` |
| DTS cross accumulation bucket | `27.447 ms` | `27.439 ms` |
| compact Pibar VJP bucket | `16.194 ms` | `16.221 ms` |
| parent-reduced DTS stage 1 bucket | `7.218 ms` | `7.203 ms` |

The launch order showed that the three no-split waves were the last three
self-loop launches:

| no-split wave launch | baseline Triton | CUDA `self` |
|---:|---:|---:|
| `W=15009` | `2.091 ms` | `1.591 ms` |
| `W=32768` | `4.536 ms` | `3.452 ms` |
| `W=32768` | `4.540 ms` | `3.451 ms` |
| total | `11.167 ms` | `8.493 ms` |

So the no-split wave replacement itself improves by `2.674 ms`, or `23.9%`,
which passes the original per-launch acceptance target.  The end-to-end
50-family speedup is much smaller because DTS accumulation and compact Pibar
VJP are unchanged and remain larger buckets.

CUDA API summaries were not the limiting difference.  `cudaLaunchKernel` time
was `4.521 ms` baseline and `4.695 ms` CUDA `self`; `cuLaunchKernel` rose only
from `0.162 ms` to `0.182 ms` for the three NVRTC launches.  The speedup is
from kernel duration, not launch count.

#### Nsight Compute, representative `W=32768` no-split wave

Representative commands:

```bash
PROFILE_CUDA_API=1 FAMS=50 REPS=1 WARMUPS=4 MAX_WAVE_SIZE=32768 \
ncu --target-processes all --profile-from-start off \
  --kernel-name _wave_backward_uniform_kernel \
  --launch-skip 34 --launch-count 1 \
  --set detailed --csv --page raw \
  --log-file /tmp/gpurec_profile/prop0/ncu_baseline_wave50.csv \
  python profiling/proposal8/bench_uniform_backward.py

GPUREC_CUDA_SELF_LOOP_NOSPLIT=1 \
GPUREC_CUDA_SELF_LOOP_NOSPLIT_CORRECTION=self \
PROFILE_CUDA_API=1 FAMS=50 REPS=1 WARMUPS=4 MAX_WAVE_SIZE=32768 \
ncu --target-processes all --profile-from-start off \
  --kernel-name gpurec_wave_backward_nosplit_uniform_fp32 \
  --launch-skip 1 --launch-count 1 \
  --set detailed --csv --page raw \
  --log-file /tmp/gpurec_profile/prop0/ncu_cuda_self_wave50.csv \
  python profiling/proposal8/bench_uniform_backward.py
```

| metric | baseline Triton | CUDA `self` | CUDA `tree` |
|---|---:|---:|---:|
| duration | `4.970 ms` | `3.808 ms` | `7.343 ms` |
| DRAM read | `1.016 GB` | `0.786 GB` | `0.787 GB` |
| DRAM write | `1.907 GB` | `0.246 GB` | `0.246 GB` |
| total DRAM bytes | `2.923 GB` | `1.033 GB` | `1.033 GB` |
| compute-memory throughput | `89.33%` | `38.68%` | `22.65%` |
| DRAM throughput | `59.82%` | `27.58%` | `14.31%` |
| SM throughput | `41.28%` | `34.55%` | `22.65%` |
| active warps | `99.48%` | `16.56%` | `16.62%` |
| registers/thread | `40` | `40` | `40` |
| dynamic shared memory/block | `32 B` | `55,972 B` | `55,972 B` |
| occupancy limit from shared memory | `14 blocks/SM` | `1 block/SM` | `1 block/SM` |
| global load instructions | `134.775 M` | `71.827 M` | `103.481 M` |
| global store instructions | `22.708 M` | `2.064 M` | `2.064 M` |
| global reduction instructions | `10.846 M` | `10.387 M` | `10.387 M` |
| shared load instructions | `8.126 M` | `57.967 M` | `71.533 M` |
| shared store instructions | `11.796 M` | `32.375 M` | `36.897 M` |
| local spilling requests | `0` | `0` | `0` |
| sampled long-scoreboard stalls | `458,593` | `247,187` | `335,730` |
| sampled barrier stalls | `88,616` | `41,814` | `310,318` |

The CUDA `self` kernel does exactly what the proposal wanted on memory:

```text
DRAM writes:       1.907 GB -> 0.246 GB
global stores:     22.708 M -> 2.064 M
global loads:     134.775 M -> 71.827 M
```

The problem is that it trades global scratch traffic for a very large shared
memory footprint.  At `55,972 B/block`, only one CTA can reside on an SM.  That
drops active warps from `99.48%` to `16.56%`, leaving too little latency hiding.
The kernel is no longer saturating DRAM (`27.58%` DRAM throughput), but it also
does not have enough resident warps to keep the SMs busy.

Global reduction atomics also remain.  The species-vector parameter gradients
still use one atomic per row/species contribution, so global RED instructions
only move from `10.846 M` to `10.387 M`.  The proposal removed scratch stores,
not parameter atomics.

`tree` mode is slower for a separate reason: it adds a bottom-up tree pass on
the shared row scratch every Neumann iteration.  That raises total instructions
from `846 M` to `1.254 B` and barrier stalls from `41,814` to `310,318`, while
active warps remain stuck near `16.6%`.  It is the exact ancestor-corrected
mode, but this implementation is not a performance win.

#### Decision

Keep the implementation as an opt-in prototype and do not promote it as the
default backward path.

What we learned:

- The global-scratch hypothesis was correct.  A no-split `W=32768` launch drops
  from `4.970 ms` to `3.808 ms`, and DRAM traffic drops by about `1.89 GB`.
- The one-row-per-block shared-memory design is too occupancy-limited to give a
  large end-to-end win at 50 families.
- The exact ancestor-corrected `tree` mode is mathematically useful for future
  correctness work, but too slow in this shape.

The next version should not store all seven row arrays in one CTA.  More
promising variants are:

1. split the row into smaller species tiles and use a second reduction kernel
   for the global `A` term, trading one extra launch for higher occupancy;
2. keep only `term`, `next`, and possibly `pcoef` in shared memory, recomputing
   cheap coefficients to reduce shared-memory footprint;
3. produce block-level partial parameter-gradient reductions and reduce them in
   a second kernel, so global atomics do not remain unchanged;
4. specialize only full `W=32768` no-split waves, because smaller waves do not
   amortize the occupancy loss.

## Proposal 1: shared-memory staged Pibar VJP instead of global `pibar_ud`

The current DTS/Pibar dataflow is:

```text
DTS backward accumulation:
    compute vd1/vd2
    compute u_d = v_Pibar * inv_denom
    write pibar_ud[2 * n_splits, S]
    write pibar_A[2 * n_splits]

Pibar VJP tree kernel:
    read pibar_ud row
    repeatedly read/write the same row while accumulating species-tree subtree sums
    atomic_add final Pi contribution into accumulated_rhs[child, :]
```

This was a major improvement over recomputing `u_d` later, but it leaves a
large global scratch stream.  The current representative Pibar launch alone
moves:

```text
read:  1.302 GB
write: 0.836 GB
time:  2.457 ms
```

The full 50-family Pibar bucket is `15.937 ms`.  Most of that is not arithmetic;
it is global memory used as a row-local tree scratchpad.

A CUDA shared-memory kernel could process one split side at a time:

```cuda
// one block handles one split side and one child clade
u[s] = vd_side[s] * inv_denom_child[s]       // computed or loaded
A = block_reduce_sum(u[s])

// tree accumulation in shared memory
for level in postorder_levels:
    u[parent] += u[child1] + u[child2]

contrib[s] = p_prime_child[s] * (A - u[s])
atomic_add(accumulated_rhs[child, s], contrib[s])
```

There are two variants:

1. **two-kernel variant:** keep current DTS accumulation, but replace only the
   staged Pibar tree kernel with a CUDA shared-memory row kernel that loads
   `pibar_ud` once and never writes tree intermediates back to DRAM;
2. **one-kernel variant:** combine the DTS side's `u_d` construction and the
   tree correction in the same CUDA block, avoiding the `pibar_ud` write
   completely.

The two-kernel variant is easier and directly measures whether row-local
shared tree scratch wins.  The one-kernel variant has the larger upside but may
inherit the 96-register pressure of DTS accumulation.

Expected gain:

- two-kernel variant: `4-8 ms` at 50 families if it mainly removes the Pibar
  tree writeback stream;
- one-kernel variant: `8-14 ms` if it removes both the `pibar_ud` write in DTS
  and the Pibar tree scratch traffic;
- likely also lowers peak memory by removing or shrinking the
  `[2 * n_splits, S]` staged buffer.

Risks:

- this is probably a CUDA kernel, not a small Triton patch;
- if the one-kernel variant pushes DTS accumulation beyond 96 registers/thread,
  occupancy may fall enough to lose;
- duplicate child atomics remain unless combined with a later grouping pass.

Recommended experiment:

Start with the two-kernel variant.  It isolates the Pibar tree memory problem
without touching DTS accumulation.  Only after that wins should the one-kernel
DTS/Pibar fusion be attempted.

### Proposal 1 test report: shared-memory Pibar-from-UD kernel

Status: implemented as an opt-in two-kernel prototype, not promoted as the
default.

Files:

- `gpurec/core/kernels/pibar_vjp_cuda.py` contains
  `gpurec_pibar_from_ud_shared_fp32`, compiled at runtime with NVRTC through
  CUDA Python bindings.
- `gpurec/core/kernels/wave_backward.py` routes
  `uniform_cross_pibar_vjp_tree_from_ud_fused` to this kernel when
  `GPUREC_CUDA_PIBAR_FROM_UD=1`.  `GPUREC_CUDA_PIBAR_FROM_UD_STRICT=1` makes
  the run fail instead of silently falling back.
- The hook is intentionally narrow: CUDA fp32, staged `pibar_ud`/`pibar_A`
  already produced by the existing DTS accumulation kernel.  It supports both
  compact level metadata and the older padded `level_parents` metadata; the
  production benchmark path uses compact levels.

The two-kernel dataflow keeps the current DTS producer unchanged:

```text
DTS backward accumulation:
    compute vd1/vd2
    compute u_d = v_Pibar * inv_denom
    write pibar_ud[2 * n_splits, S]
    write pibar_A[2 * n_splits]

CUDA Pibar-from-UD:
    load one staged side row into shared memory
    do bottom-up species-tree subtree accumulation in shared memory
    atomic_add p_prime * (A - subtree_sum) into accumulated_rhs[child, :]
```

The new kernel uses one fp32 shared-memory row:

```text
1999 * 4 B = 7,996 B dynamic shared memory per block
```

That is very different from the proposal-0 self-loop prototype.  It still keeps
high occupancy because it is limited to about `8 KB/block`, not `56 KB/block`.

Pseudo-code:

```cuda
// one block per split side
for s in species:
    work[s] = pibar_ud[row, s]

for level in postorder_levels:
    for node in level:
        work[parent(node)] += work[child1(node)] + work[child2(node)]
    __syncthreads()

for s in species:
    p_prime = exp2(Pi[child, s] - row_max[child])
    atomic_add(accumulated_rhs[child, s], p_prime * (A[row] - work[s]))
```

This preserves the current active-mask and side-skip semantics:

- `active_mask` is parent-row based and indexed with `reduce_idx[split_i]`;
- `side_active` is the exact nonzero side-row predicate produced by the DTS
  kernel, not `A != 0`;
- duplicate child clades still require atomics into `accumulated_rhs`.

#### Correctness checks

Commands run:

```bash
python -m py_compile \
  gpurec/core/kernels/wave_backward.py \
  gpurec/core/kernels/pibar_vjp_cuda.py

pytest -q tests/kernels/test_dts_backward_accum_kernel.py

pytest -q tests/gradients/test_autograd_bridge.py

GPUREC_CUDA_PIBAR_FROM_UD=1 \
GPUREC_CUDA_PIBAR_FROM_UD_STRICT=1 \
pytest -q tests/gradients/test_autograd_bridge.py
```

The direct DTS/Pibar kernel file passed: `49 passed`.  This includes the
worker-added direct test
`test_uniform_cross_pibar_from_ud_cuda_shared_preserves_pibar_ud`, which checks
both padded and compact metadata paths and verifies that the CUDA shared-memory
consumer does not clobber `pibar_ud`.

Both public autograd runs passed: `15 passed`.

Large-workload parity was checked by running the same `test_trees_1000` model
with the current Triton compact Pibar-from-UD path and with
`GPUREC_CUDA_PIBAR_FROM_UD=1 GPUREC_CUDA_PIBAR_FROM_UD_STRICT=1`.

| workload | loss diff | max abs grad diff | max rel grad diff | mean abs grad diff |
|---:|---:|---:|---:|---:|
| 10 families | `0` | `2.441e-4` | `6.345e-7` | `2.441e-4` |
| 50 families | `0` | `2.441e-3` | `1.332e-6` | `2.279e-3` |

The differences are fp32 atomic-order noise.  They are smaller than the
proposal-0 self-loop prototype because the new kernel changes only one
tree-scratch phase and preserves the same mathematical correction.

Subagent correctness audit notes:

- `pibar_ud` rows are laid out as left sides in `0:n_ws` and right sides in
  `n_ws:2*n_ws`;
- `pibar_A[row]` is the original `sum_s u_d[s]`, before tree accumulation;
- active masks are over parent wave rows, not split sides;
- exact zero-side skipping must use the staged row's nonzero predicate, because
  `A == 0` can happen by cancellation even when `u_d` is nonzero;
- the current Triton consumer clobbers `pibar_ud` in-place as global subtree
  scratch.  The CUDA prototype keeps the subtree scratch in shared memory and
  preserves `pibar_ud`; the direct test asserts that property.

#### Event-timed benchmarks

All timings below use `tests/data/test_trees_1000`, `fixed_iters_Pi=6`,
`neumann_terms=3`, `MAX_WAVE_SIZE=32768`, and the same fused backward defaults
as the previous pass.  Proposal 0 is off unless explicitly noted.

| families | mode | median backward | delta vs baseline | peak allocation |
|---:|---|---:|---:|---:|
| 10 | baseline Triton Pibar-from-UD | `33.784 ms` | - | `2.696 GB` |
| 10 | CUDA Pibar-from-UD | `34.832 ms` | `+1.048 ms` slower | `2.696 GB` |
| 50 | baseline Triton Pibar-from-UD | `103.788 ms` | - | `10.308 GB` |
| 50 | CUDA Pibar-from-UD | `101.558 ms` | `-2.230 ms` faster | `10.308 GB` |
| 100 | baseline Triton Pibar-from-UD | `192.292 ms` | - | `17.950 GB` |
| 100 | CUDA Pibar-from-UD | `186.765 ms` | `-5.527 ms` faster | `17.950 GB` |
| 50 | proposal 0 + proposal 1 | `98.119 ms` | `-5.669 ms` faster | `9.898 GB` |

After the worker's broader padded+compact implementation and direct tests were
integrated, a quick 5-repetition 50-family check gave `102.570 ms` baseline and
`99.736 ms` with `GPUREC_CUDA_PIBAR_FROM_UD=1`, a `2.834 ms` median win.  The
tables above keep the earlier 7-repetition comparison because it was collected
as a matched baseline/candidate pair before the final documentation pass; both
runs show the same conclusion.

Interpretation:

- The two-kernel Pibar replacement is too small to help at 10 families.
- It becomes useful as split-side work grows.  At 100 families it saves about
  `5.5 ms`.
- Peak allocation is unchanged.  This is expected: the two-kernel variant still
  requires the DTS kernel to materialize `pibar_ud`.  It only avoids using
  `pibar_ud` as global tree scratch in the consumer.
- It composes with proposal 0.  On the fresh 50-family baseline, proposal 1
  alone saves `2.230 ms`; proposal 0 + proposal 1 saves `5.669 ms` and keeps
  proposal 0's lower peak allocation.

#### Nsight Systems, 50 families

Commands:

```bash
PROFILE_CUDA_API=1 FAMS=50 REPS=1 WARMUPS=4 MAX_WAVE_SIZE=32768 \
nsys profile --force-overwrite=true \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --trace=cuda,nvtx,osrt \
  -o /tmp/gpurec_profile/prop1/nsys_baseline_50 \
  python profiling/proposal8/bench_uniform_backward.py

GPUREC_CUDA_PIBAR_FROM_UD=1 \
GPUREC_CUDA_PIBAR_FROM_UD_STRICT=1 \
PROFILE_CUDA_API=1 FAMS=50 REPS=1 WARMUPS=4 MAX_WAVE_SIZE=32768 \
nsys profile --force-overwrite=true \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --trace=cuda,nvtx,osrt \
  -o /tmp/gpurec_profile/prop1/nsys_cuda_50 \
  python profiling/proposal8/bench_uniform_backward.py
```

Single captured backward interval:

| metric | baseline | CUDA Pibar-from-UD |
|---|---:|---:|
| benchmark-reported backward time under NSYS | `116.390 ms` | `114.541 ms` |
| total GPU kernel time | `91.181 ms` | `87.107 ms` |
| kernel launches | `2918` | `2918` |
| DTS cross accumulation bucket | `27.449 ms` | `27.439 ms` |
| self-loop wave bucket | `24.943 ms` | `25.004 ms` |
| Pibar-from-UD bucket | `33 launches`, `16.212 ms` | `33 launches`, `12.206 ms` |
| parent-reduced DTS stage 1 bucket | `7.190 ms` | `7.133 ms` |
| active-mask bucket | `3.026 ms` | `3.008 ms` |

The proposal is isolated: DTS and self-loop buckets are unchanged within noise.
The Pibar-from-UD bucket drops by `4.006 ms`, or `24.7%`.

Per-launch timings show the same pattern:

```text
baseline compact Pibar-from-UD:
0.235, 0.711, 1.235, 1.292, 2.250, 1.899, 1.011, ...
..., 0.504, 0.705, 0.967, 1.536, 2.483 ms

CUDA shared Pibar-from-UD:
0.175, 0.531, 0.931, 0.958, 1.704, 1.424, 0.760, ...
..., 0.377, 0.525, 0.727, 1.164, 1.915 ms
```

CUDA API overhead does not explain the improvement.  `cudaLaunchKernel` is
essentially unchanged (`4.745 ms` baseline, `4.779 ms` candidate), while
`cuLaunchKernel` rises only from `0.172 ms` to `0.260 ms` for the NVRTC
launches.

#### Nsight Compute, representative Pibar launches

Two launch positions were profiled:

- skip `4`: an early sparse/hot Pibar-from-UD launch;
- skip `32`: the largest late Pibar-from-UD launch in the captured 50-family
  schedule.

Commands used the same shape as the NSYS run, changing only the kernel name and
launch skip:

```bash
ncu --target-processes all --profile-from-start off \
  --kernel-name _uniform_cross_pibar_vjp_tree_from_ud_compact_kernel \
  --launch-skip 32 --launch-count 1 \
  --set detailed --csv --page raw \
  --log-file /tmp/gpurec_profile/prop1/ncu_baseline_pibar_skip32.csv \
  python profiling/proposal8/bench_uniform_backward.py

GPUREC_CUDA_PIBAR_FROM_UD=1 \
GPUREC_CUDA_PIBAR_FROM_UD_STRICT=1 \
ncu --target-processes all --profile-from-start off \
  --kernel-name gpurec_pibar_from_ud_shared_fp32 \
  --launch-skip 32 --launch-count 1 \
  --set detailed --csv --page raw \
  --log-file /tmp/gpurec_profile/prop1/ncu_cuda_pibar_skip32.csv \
  python profiling/proposal8/bench_uniform_backward.py
```

Skip `4`:

| metric | baseline compact Triton | CUDA shared |
|---|---:|---:|
| duration | `2.229 ms` | `1.684 ms` |
| DRAM read | `1.164 GB` | `1.164 GB` |
| DRAM write | `0.747 GB` | `0.365 GB` |
| compute-memory throughput | `87.20%` | `92.37%` |
| DRAM throughput | `87.20%` | `92.37%` |
| SM throughput | `25.71%` | `17.77%` |
| active warps | `98.63%` | `96.15%` |
| registers/thread | `36` | `26` |
| dynamic shared memory/block | `0 B` | `7,996 B` |
| global load instructions | `29.496 M` | `24.311 M` |
| global store instructions | `2.228 M` | `0` |
| global reduction instructions | `3.051 M` | `3.051 M` |
| shared load/store instructions | `0 / 0` | `9.736 M / 5.280 M` |
| local spilling requests | `0` | `0` |

Skip `32`:

| metric | baseline compact Triton | CUDA shared |
|---|---:|---:|
| duration | `2.514 ms` | `1.896 ms` |
| DRAM read | `1.302 GB` | `1.301 GB` |
| DRAM write | `0.838 GB` | `0.412 GB` |
| compute-memory throughput | `86.57%` | `91.90%` |
| DRAM throughput | `86.57%` | `91.90%` |
| SM throughput | `25.39%` | `17.49%` |
| active warps | `98.23%` | `95.81%` |
| registers/thread | `36` | `26` |
| dynamic shared memory/block | `0 B` | `7,996 B` |
| global load instructions | `32.752 M` | `26.807 M` |
| global store instructions | `2.486 M` | `0` |
| global reduction instructions | `3.405 M` | `3.405 M` |
| shared load/store instructions | `0 / 0` | `10.863 M / 5.891 M` |
| local spilling requests | `0` | `0` |

The resource story is clean:

- The proposal removes the global stores used by the old tree-scratch phase.
  DRAM writes fall by about `0.38-0.43 GB` per representative launch.
- DRAM reads barely change because both kernels still read `Pi`, `pibar_ud`,
  tree metadata, and row maxima.  A one-kernel DTS/Pibar fusion would be needed
  to remove the `pibar_ud` read itself.
- Global reduction instructions are unchanged because duplicate child clades
  still require atomics into `accumulated_rhs`.
- Occupancy remains high.  The CUDA kernel uses only `7,996 B` dynamic shared
  memory per block, so active warps stay near `96%`.  This is why proposal 1
  succeeds where the proposal-0 shared-memory self-loop kernel was occupancy
  limited.
- SM throughput drops because the kernel is even more memory dominated after
  removing arithmetic/instruction overhead.  That is not a regression; the
  duration still falls by `24-25%`.

#### Decision

Keep the two-kernel shared-memory Pibar-from-UD path as an opt-in prototype.
It is a real improvement, but it does not meet the original `4 ms` event-median
acceptance gate at 50 families in this run:

```text
event median:      103.788 ms -> 101.558 ms  (2.230 ms faster)
Pibar NSYS bucket:  16.212 ms ->  12.206 ms  (4.006 ms faster)
```

The bucket-level improvement is exactly what proposal 1 targeted, but the full
backward pass still has larger unchanged buckets:

```text
DTS cross accumulation: ~27.4 ms
self-loop wave bucket:  ~25.0 ms
Pibar-from-UD after fix: ~12.2 ms
```

The next step for this line of work is the one-kernel variant: fuse the DTS
side's `u_d` construction with the shared-memory tree correction.  That would
remove both the `pibar_ud` write from DTS and the `pibar_ud` read from Pibar VJP,
and it is the only proposal-1 variant that can reduce peak memory.  The risk is
register pressure in the DTS kernel, so it should be attempted separately with
NCU checks on registers/thread, spills, and occupancy.

## Proposal 2: ragged parent-tile worklist for high-fanout DTS backward

The rejected parent-tiled DTS backward prototype still taught us something
useful.  Parent reuse is real: the parent-tiled NCU run reduced global load
instructions and sectors.  It lost because the launch geometry was rectangular:

```text
grid = (eq1 + ge2_groups, max_tiles, species_blocks)
```

This launched many CTAs that returned immediately and multiplied scalar
reduction atomics by the number of species blocks.  The worst wave had more
than `20x` rectangular overlaunch.

A different implementation should precompute a ragged descriptor list:

```text
descriptor[t] = {
    parent_w,
    split_start,
    split_count,      // <= TILE_SPLITS
}
```

Then launch only real high-fanout tiles:

```text
grid = (n_descriptors,)
program descriptor:
    load parent v_k/Pi once per species tile
    loop over only this parent's split tile
    keep scalar reductions inside the descriptor, not per species-block axis
```

Eq1 rows should stay on the current direct split-major kernel.  The goal is to
keep the direct kernel's good scalar-reduction behavior while reusing parent
rows for true high-fanout parents.

Expected gain:

- `3-8 ms` at 50 families if the ragged worklist removes the overlaunch seen in
  the rejected prototype;
- the ceiling is capped because child row loads/stores, `pibar_ud` output, and
  direct Pi atomics remain.

Risks:

- a Triton 2D tile over `(TILE_SPLITS, BLOCK_S)` may raise register pressure;
- a CUDA version may be cleaner if we want parent data in shared memory;
- the descriptor build must happen during preprocessing, not inside backward.

Acceptance gate:

For the 42,155-split launch, the new high-fanout path must beat the direct
`_dts_cross_backward_accum_kernel` launch on both NCU duration and total
DTS-accum Nsys bucket.  Lower global-load instructions alone are not enough;
the previous parent-tiled path already achieved that and still regressed.

### Proposal 2 results

Status: implemented as an opt-in prototype and rejected as a default.

Implementation commit:

```text
996148e Add ragged parent-tile DTS backward accumulation.
```

Changed files:

- `gpurec/core/kernels/wave_backward.py`
- `gpurec/core/backward.py`
- `tests/kernels/test_dts_backward_accum_kernel.py`

Runtime controls:

```text
GPUREC_DTS_BACKWARD_ACCUM_IMPL=parent_ragged[_all]
GPUREC_PARENT_RAGGED_DTS_BACKWARD_TILE_SPLITS=<tile splits>
```

Defaults are preserved.  The prototype caches a ge2 tile worklist on wave
metadata and routes selected high-fanout waves through the ragged kernel.
Non-selected waves stay on the current direct split-major kernel.  Within a
selected wave, eq1 rows are represented as one-split ragged descriptors rather
than running through a separate direct prefix kernel.

The implementation removes the rectangular empty CTAs from the earlier
`parent_tiled` prototype, but it is still only a partial version of the ideal
proposal.  The current Triton launch geometry is:

```text
grid = (n_work_tiles, ceil(S / BLOCK_S))
```

not the ideal:

```text
grid = (n_descriptors,)
```

Because species blocks are still a grid axis, scalar reductions for `pibar_A`,
`grad_log_pD`, `grad_log_pS`, and optional `grad_mt` are still issued once per
species block.  This keeps one of the rejected rectangular prototype's major
costs.

Subagent split:

| Worker | Role | Result |
|---|---|---|
| Agent 1 | implementation | added cached ragged ge2 tile metadata, opt-in `parent_ragged` routing, and kernel tests |
| Agent 2 | correctness | verified direct kernel parity, FD coverage, autograd bridge coverage, and a skewed synthetic high-fanout case |
| Agent 3 | profiling | ran paired event timing, Nsys buckets, NCU comparisons, and tile-split sweep |
| Supervisor | decision | confirmed the ragged path removes overlaunch but still fails the NCU and Nsys acceptance gate |

Correctness:

| Command | Result |
|---|---:|
| `pytest -q tests/kernels/test_dts_backward_accum_kernel.py` | `57 passed` |
| `GPUREC_DTS_BACKWARD_ACCUM_IMPL=parent_ragged_all GPUREC_PARENT_RAGGED_DTS_BACKWARD_TILE_SPLITS=16 pytest -q tests/gradients/test_fd_all_modes.py::test_analytic_matches_fd tests/gradients/test_autograd_bridge.py` | `21 passed` |

The worker's skewed synthetic high-fanout case used `S=512` and `n_ws=830`.
The rectangular geometry would have launched `6144` tiles; the ragged worklist
launched `223` tiles.  Parity matched with max relative error at or below
`5.5e-7`, and `pibar_ud`/`pibar_A` were exact.  The isolated synthetic timing
improved only slightly, from `0.099 ms` rectangular to `0.091 ms` ragged.

CUDA-event timings outside Nsight:

| Run | Variant | Median backward | Mean | Min | Change vs direct |
|---|---|---:|---:|---:|---:|
| local, `FAMS=50 REPS=7 WARMUPS=5 MAX_WAVE_SIZE=32768` | direct | `102.356 ms` | `102.447 ms` | `101.800 ms` | reference |
| local, same command | `parent_tiled`, tile 16 | `116.829 ms` | `119.016 ms` | `116.104 ms` | `+14.473 ms` |
| local, same command | `parent_ragged`, tile 16 | `116.186 ms` | `118.902 ms` | `115.691 ms` | `+13.830 ms` |
| performance agent paired run | direct | `103.362 ms` | - | - | reference |
| performance agent paired run | `parent_tiled` | `117.731 ms` | - | - | `+14.369 ms` |
| performance agent paired run | `parent_ragged` | `116.388 ms` | - | - | `+13.026 ms` |

The paired run reported unchanged loss and memory footprint:

```text
loss = 107804.2734375
peak allocation = 10.308 GB
```

Sequential tile sweep, `FAMS=50 REPS=3 WARMUPS=3`:

| `GPUREC_PARENT_RAGGED_DTS_BACKWARD_TILE_SPLITS` | Median backward |
|---:|---:|
| 4 | `123.549 ms` |
| 8 | `115.166 ms` |
| 16 | `114.935 ms` |
| 32 | `120.464 ms` |

The best tile setting in this short sweep was still much slower than the
direct default.

Nsys paired results from `/tmp/gpurec_profile/prop2_readonly`:

| Variant | Nsys backward | DTS accum subtotal | DTS detail | Other hot buckets |
|---|---:|---:|---|---|
| current direct | `116.741 ms` | `27.454 ms` | `_dts_cross_backward_accum_kernel`: `27.454 ms`, 33 launches | `_wave_backward_uniform_kernel`: `24.618 ms`, 36 launches; `_uniform_cross_pibar_vjp_tree_from_ud_compact_kernel`: `16.219 ms`, 33 launches |
| `parent_tiled` | `127.553 ms` | `40.646 ms` | `parent_tiled_ge2`: `27.343 ms`, 6 launches; remaining direct DTS: `10.372 ms`, 27 launches | - |
| `parent_ragged` | `129.080 ms` | `40.860 ms` | `parent_ragged`: `27.538 ms`, 6 launches; remaining direct DTS: `10.383 ms`, 27 launches | `_pibar_ud_side_active`: `2.939 ms`, 6 launches from ragged side-active handling |

The ragged worklist fixes the rectangular overlaunch but does not reduce the
total DTS accumulation bucket.  It is slightly slower than the rectangular
prototype in this Nsys capture because the remaining species-block scalar
atomics and side-active handling dominate the saved empty CTA work.

Wave 44 diagnostics:

```text
S=1999
W=247
splits=42155
n_eq1=234
ge2_groups=13
ge2_mean=3224.7
ge2_max=3787
```

| Variant | Grid | CTAs | Nsys launch duration |
|---|---|---:|---:|
| direct | `(42155, 1, 1)` | `42,155` | about `4.569 ms` |
| rectangular `parent_tiled` | `(247, 237, 8)` | `468,312` | about `7.314 ms` |
| ragged `parent_ragged` | `(2861, 8, 1)` | `22,888` | about `7.390 ms` |

The ragged grid cuts the rectangular CTA count by about `20.5x` and launches
fewer CTAs than the direct split-major kernel.  That reduction is real, but it
does not translate into a faster launch.

NCU representative wave 44 from `/tmp/gpurec_profile/prop2_ragged`:

| Metric | Direct split-major | Ragged parent tile | Rectangular parent tile |
|---|---:|---:|---:|
| Kernel | `_dts_cross_backward_accum_kernel` | `_dts_cross_backward_accum_parent_ragged_kernel` | `_dts_cross_backward_accum_parent_tiled_ge2_kernel` |
| Duration | `4.631712 ms` | `7.439616 ms` | `9.710176 ms` |
| Grid size | `42,155` CTAs | `22,888` CTAs | `468,312` CTAs |
| Threads launched | `5,395,840` | `2,929,664` | `59,943,936` |
| Registers/thread | `96` | `96` | `96` |
| Waves/SM | `65.87` | `35.76` | `731.74` |
| Achieved warps | `41.47%` | `41.41%` | `41.11%` |
| Memory throughput | `73.26%` | `49.88%` | `38.20%` |
| DRAM throughput | `73.26%` | `49.88%` | `38.20%` |
| SM throughput | `33.31%` | `16.47%` | `12.61%` |
| DRAM read | `2.017 GB` | `2.218 GB` | `2.217 GB` |
| DRAM write | `1.319 GB` | `1.430 GB` | `1.430 GB` |
| Global load instructions | `36.392 M` | `29.531 M` | `29.560 M` |
| Global store instructions | `5.480 M` | `5.312 M` | `5.312 M` |
| Global RED instructions | `18.571 M` | `19.871 M` | `19.871 M` |
| Instructions | `778.598 M` | `728.577 M` | `744.170 M` |
| Spills | none | none | none |

Stall samples:

| Stall reason | Direct | Ragged |
|---|---:|---:|
| Long scoreboard | `226,725 / 642,129 = 35.3%` | `465,447 / 1,034,894 = 45.0%` |
| Barrier | `25.3%` | `23.9%` |
| MIO | `12.0%` | `7.8%` |
| Wait | `5.5%` | `3.1%` |

The NCU result explains the regression.  Ragged parent tiling lowers global
load instructions and removes empty CTAs, but it also lowers achieved memory
throughput, moves more DRAM bytes, issues more global reductions, and spends a
larger share of samples stalled on long scoreboard.  The extra DRAM traffic is
nearly identical to the rectangular parent-tiled path, which shows that the
remaining species-block scalar atomics and cache behavior are the important
costs after overlaunch is removed.

Decision: keep `parent_ragged` opt-in only; do not promote it to the default.
It fails the acceptance gate.  On the representative `42,155`-split launch,
ragged takes `7.44 ms` in NCU versus `4.63 ms` for the direct kernel, and the
total DTS accumulation Nsys bucket is about `40.86 ms` versus `27.45 ms`
direct.  Lower load instruction count is not enough to offset lower memory
throughput, more DRAM bytes, more global reductions, and long-scoreboard
stalls.

This is still a useful negative result.  It proves that the rectangular
overlaunch was not the only problem with parent tiling in Triton.  A true
`grid=(n_descriptors,)` implementation that keeps scalar reductions inside a
descriptor, or a CUDA implementation that keeps parent rows in shared memory,
would be separate work.  The current Triton ragged implementation is not
default-worthy.

## Proposal 3: CUDA graph or graph segments for the fixed backward schedule

The current host still synchronizes per wave to preserve whole-wave pruning:

```python
active_mask = _compute_active_mask(rhs_k)
wave_active = bool(active_mask.any())   # host sync
if not wave_active:
    continue
```

Current fixed/device pruning is slower:

| Variant | 50-family median |
|---|---:|
| default host whole-wave skip | `101.037 ms` |
| `GPUREC_DEVICE_PRUNING=1` | `104.954 ms` |
| `GPUREC_BACKWARD_NO_CPU_PRUNING=1` | `105.141 ms` |

So simply removing the host decisions is still not enough; skipped waves save
real GPU work.  However, Nsight still reports `205` stream synchronizations and
about `2578` kernel launches in one backward capture.  A CUDA graph changes
the cost model because unconditional small launches become much cheaper.

There are three graph variants worth considering:

1. **full fixed graph:** capture the whole backward with device masks and no
   host wave skip.  This is the simplest and directly retests whether lower
   launch overhead offsets the extra skipped-wave kernels.
2. **large-wave host skip, small-wave graph:** keep host decisions for the few
   expensive waves, but graph a fixed suffix/prefix of small waves where launch
   overhead dominates.
3. **per-wave graph fragments:** capture the kernel sequence for one wave shape
   and replay it when the host decides that wave is active.

Expected gain:

- full graph: uncertain, probably `0-4 ms` because current fixed pruning is
  `~4 ms` slower before graph benefits;
- segmented graph: `2-6 ms` if it removes most launch/sync overhead while
  preserving skips of large inactive waves.

Risks:

- PyTorch CUDA graph capture requires static allocation.  The current backward
  still creates temporary tensors and may need a real scratch pool first.
- Graphing should not hide correctness issues in dynamic pruning.

This is not a first coding target, but it is the most direct way to attack the
non-kernel gap once the memory kernels stop dominating.

### Proposal 3 tested results

Implemented as an opt-in benchmark/prototype path, not a production default.
`profiling/proposal8/bench_uniform_backward.py` now supports:

```text
--cuda-graph
--cuda-graph-target=model|pi_backward
--graph-fixed-schedule-mode
--cuda-graph-profile-phase
```

The viable captured path is the direct `Pi_wave_backward` benchmark with
precomputed forward/Pi inputs, static kwargs, static root IDs, and species
topology hoisted out of the captured region.  `gpurec/core/backward.py` now
reuses the species child/topology tensors cached by the forward path, which
removes the CPU-to-CUDA topology copies that otherwise blocked direct
`Pi_wave_backward` capture.

Full model graph capture is still blocked before backward.  It fails in the
forward `E_fixed_point` path at:

```text
gpurec/core/likelihood.py
ancestor_sum = (expE_2d @ ancestors_T).contiguous()
```

with CUDA stream-capture unsupported.  This therefore is not a full
forward-plus-backward production graph.  Device-pruning graph mode also remains
not graph-safe because it still has host stats synchronizations such as
`.sum().item()`.  The successful capture mode is fixed schedule via
`GPUREC_BACKWARD_NO_CPU_PRUNING=1`.

Correctness and validation:

| Check | Result |
|---|---:|
| `python -m py_compile gpurec/core/backward.py profiling/proposal8/bench_uniform_backward.py` | passed |
| `pytest -q tests/gradients/test_autograd_bridge.py` | `15 passed in 2.73s` |
| `GPUREC_BACKWARD_NO_CPU_PRUNING=1 pytest -q tests/gradients/test_autograd_bridge.py` | `15 passed in 3.71s` |

Direct `Pi_wave_backward` graph check, 50 families:

```bash
FAMS=50 REPS=5 WARMUPS=5 MAX_WAVE_SIZE=32768 \
  python profiling/proposal8/bench_uniform_backward.py \
    --cuda-graph --cuda-graph-target=pi_backward --graph-fixed-schedule-mode=no_cpu
```

| Metric | Normal direct Pi backward | CUDA graph replay |
|---|---:|---:|
| Median event time | `93.874 ms` | `92.189 ms` |
| Peak allocation | `12.873 GB` | `15.489 GB` |

The maximum output difference was `1.22070312e-04` on `grad_log_pD`, with
relative error to that key's max of `1.79689305e-07`.  A second `REPS=1` check
measured normal `94.083 ms`, graph replay `92.149 ms`, and maximum difference
`6.10351562e-05` on `grad_log_pD` with relative error `8.98446281e-08`.

For context, the full default backward after these changes still measures:

| Workload | Median | Mean | Min | Peak allocation |
|---|---:|---:|---:|---:|
| `FAMS=50 REPS=3 WARMUPS=8 MAX_WAVE_SIZE=32768` | `101.971 ms` | `101.988 ms` | `101.362 ms` | `10.308 GB` |

#### Nsight Systems

Normal fixed-schedule direct Pi backward:

```text
/tmp/gpurec_profile/prop3_graph/pi_backward_normal50.nsys-rep
```

The captured event time under Nsight was `100.898 ms`; the paired replay was
not profiled in that capture range and measured `92.641 ms`.  Nsight stats put
the summed kernel total at about `96.7 ms`.

Top normal kernel buckets:

| Kernel bucket | Time | Launches | Share |
|---|---:|---:|---:|
| `_dts_cross_backward_accum_kernel` | `30.068548 ms` | 46 | `31.1%` |
| `_wave_backward_uniform_kernel` | `26.827647 ms` | 49 | `27.7%` |
| `_uniform_cross_pibar_vjp_tree_from_ud_compact_kernel` | `16.432025 ms` | 46 | `17.0%` |
| `_dts_parent_reduced_ge2_stage1_kernel` | `7.230238 ms` | 6 | `7.5%` |
| `_dts_eq1_to_rows_kernel` | `4.079860 ms` | 18 | `4.2%` |
| `_active_mask_from_rhs_absmax_kernel` | `3.462378 ms` | 49 | `3.6%` |

Normal CUDA API launch overhead:

| CUDA API | Time | Calls |
|---|---:|---:|
| `cudaLaunchKernel` | `2.512846 ms` | 1415 |
| `cuLaunchKernelEx` | `0.651806 ms` | 295 |
| `cuLaunchKernel` | `0.152036 ms` | 100 |
| `cudaMemcpyAsync` | `0.154900 ms` | 50 |

The explicit launch/API budget is only about `3.3 ms`.  The `cudaDeviceSynchronize`
entry at `78.587905 ms` is the event-timing synchronization and should not be
counted as avoidable launch overhead.

Default graph replay trace:

```text
/tmp/gpurec_profile/prop3_graph/pi_backward_graph_replay50.nsys-rep
```

This showed `cudaGraphLaunch_v10000` at `0.388115 ms` and one
`cudaDeviceSynchronize` at `94.443833 ms`, but default graph tracing collapsed
the graph nodes and did not emit useful kernel rows.

Node-level graph replay trace:

```text
/tmp/gpurec_profile/prop3_graph/pi_backward_graph_replay50_node.nsys-rep
```

This used `--cuda-graph-trace=node`.  The event timing was distorted by node
instrumentation, with normal `95.467 ms` and replay `96.720 ms`, so this capture
is useful only for composition.  The node kernel composition matched the normal
profile:

| Kernel bucket | Node-trace time |
|---|---:|
| `_dts_cross_backward_accum_kernel` | `30.017434 ms` |
| `_wave_backward_uniform_kernel` | `25.017150 ms` |
| `_uniform_cross_pibar_vjp_tree_from_ud_compact_kernel` | `16.684564 ms` |
| `_dts_parent_reduced_ge2_stage1_kernel` | `7.243548 ms` |
| `_dts_eq1_to_rows_kernel` | `4.074898 ms` |
| `_active_mask_from_rhs_absmax_kernel` | `3.449391 ms` |

CUDA API in the node trace had one `cudaGraphLaunch_v10000` at `1.715360 ms`
under instrumentation and one `cudaDeviceSynchronize` at `94.903605 ms`.

#### Decision

Do not promote CUDA graph replay as a production default now.  Proposal 3 is
correct as a direct `Pi_wave_backward` benchmark path, but the non-instrumented
replay saves only about `1.7 ms` versus `93.9 ms` normal direct Pi backward,
roughly `1.8%`, while increasing peak allocation by about `2.6 GB` because of
CUDA graph private pools.

The Nsight launch/API budget is only about `3.3 ms` out of roughly
`96-102 ms`, and the replay trace has essentially the same kernel composition
as the normal fixed-schedule path.  Graphing therefore cannot address the
dominant memory-bound kernels.  Keep the benchmark/prototype for future
experiments, but revisit production graphing only after the top memory kernels
shrink materially or after the forward path becomes graph-capturable.

## Proposal 4: tailored two-stage self-loop parameter reductions

The current self-loop wave kernel atomically accumulates global-mode parameter
gradients directly:

```text
grad_log_pD += sum(term0)
grad_log_pS += sum(term3 + term4 + leaf)
grad_E[s]      += term0[s] + term2[s]
grad_Ebar[s]   += term1[s]
grad_E_s1[s]   += term4[s]
grad_E_s2[s]   += term3[s]
grad_mt[s]     += term2[s]
```

This was much faster than materializing six full contribution tensors, but NCU
now shows the remaining cost clearly:

```text
self-loop representative launch:
  global RED instructions: 10.846 M
  DRAM write:              1.905 GB
  memory throughput:       90.25%
```

A custom two-stage reduction would not return to the old materialized path.
Instead, it would reduce row tiles into compact partials:

```text
stage 1, per wave:
    grid = (row_tile, species_block)
    recompute final parameter weights from Pi/Pibar/v_k
    reduce BLOCK_ROWS rows locally
    write partial[param, tile, species]
    write scalar partials for log_pD/log_pS

stage 2, once per backward or per group of waves:
    reduce partials into grad_E, grad_Ebar, grad_E_s1, grad_E_s2, grad_mt
```

The wave kernel would then stop doing species-vector global atomics.  It would
still write `v_k`, and the reduction kernel would read `v_k/Pi/Pibar` once.
This trades one extra structured pass for fewer contended RED operations in the
already memory-saturated self-loop kernel.

Expected gain:

- `2-6 ms` at 50 families if RED pressure and write traffic are a meaningful
  part of the `23.789 ms` wave bucket;
- larger gain possible for leaf waves because the two 32,768-row no-split
  launches perform many atomics into only `S=1999` species lanes.

Risks:

- if the extra pass simply rereads too much Pi/Pibar, it can lose like the old
  contribution-tensor path;
- it should be tested only after isolating no-split waves, where final weights
  are simplest and launch shapes are largest.

This proposal pairs naturally with Proposal 0.  If a CUDA shared-memory
self-loop kernel is built, it should emit row-tile partial reductions rather
than doing per-row global RED atomics.

### Proposal 4 tested results

Implemented in:

- `3804d09` Prototype two-stage self-loop param reductions
- `377e202` Tune self-loop two-stage reduction default

The implementation changed only `gpurec/core/kernels/wave_backward.py`.  The
path is opt-in with `GPUREC_SELF_LOOP_PARAM_TWO_STAGE=1`; the default remains
off.

#### Implementation shape

Safe scope:

- fused uniform self-loop only;
- no-split waves only;
- `accum_param_grads` enabled;
- scalar `grad_log_pD` and `grad_log_pS`;
- vector `grad_E`, `grad_Ebar`, `grad_E_s1`, `grad_E_s2`, and `grad_mt`
  tensors of length `S`;
- split waves and unsupported shapes fall back to the existing direct atomic
  accumulation path.

Mechanism:

1. For eligible no-split waves, `_wave_backward_uniform_kernel` computes `v_k`
   and exits before final parameter-VJP atomics.
2. `_wave_backward_uniform_param_stage1_kernel` recomputes no-split final term
   weights from `Pi/Pibar/v_k` over row tiles and species blocks, then writes
   compact `partial_vec[5, n_tiles, S]` and
   `partial_scalar[2, n_tiles, n_s_blocks]`.
3. `_wave_backward_uniform_param_stage2_kernel` reduces those partials into the
   existing gradient tensors.

Final default tunables for the opt-in path:

| Env flag | Final default / reported tuned value |
|---|---|
| `GPUREC_SELF_LOOP_PARAM_TWO_STAGE_TILE_ROWS` | `16` |
| `GPUREC_SELF_LOOP_PARAM_TWO_STAGE_BLOCK_S` | `min(256, next_power_of_2(S))` |
| `GPUREC_SELF_LOOP_PARAM_TWO_STAGE_REDUCE_TILES` | `16` |
| `GPUREC_SELF_LOOP_PARAM_TWO_STAGE_SCALAR_BLOCK` | `1024` |
| `GPUREC_SELF_LOOP_PARAM_TWO_STAGE_NUM_WARPS` | optional override, unset in reported tuned runs |
| `GPUREC_SELF_LOOP_PARAM_TWO_STAGE_REDUCE_WARPS` | `4` |

#### Correctness checks

Worker checks passed:

- `py_compile`;
- implementation diff check;
- synthetic no-split direct-vs-two-stage parity with `S=16`;
- synthetic leaf-index plus active-mask parity with `S=17`;
- large-`S` synthetic parity with `S=1999`, max gradient difference
  `5.96e-08`;
- real `Pi_wave_backward` parity on `test_trees_100`, with max differences
  around `1e-10`;
- `GPUREC_SELF_LOOP_PARAM_TWO_STAGE=1 pytest -q tests/kernels/test_wave_backward_kernel.py::test_wave_speciation_gather_matches_scatter`.

Main-agent checks:

| Command | Result |
|---|---:|
| `python -m py_compile gpurec/core/kernels/wave_backward.py` | passed |
| `GPUREC_SELF_LOOP_PARAM_TWO_STAGE=1 pytest -q tests/kernels/test_wave_backward_kernel.py::test_wave_speciation_gather_matches_scatter tests/gradients/test_autograd_bridge.py` | `16 passed in 2.76s` before tuning |
| same combined pytest after tuning | `16 passed in 3.11s` |
| `GPUREC_SELF_LOOP_PARAM_TWO_STAGE=1 GPUREC_BACKWARD_NO_CPU_PRUNING=1 pytest -q tests/gradients/test_autograd_bridge.py` | `15 passed in 2.70s` |

Direct Pi-backward graph/parity smoke:

```bash
GPUREC_SELF_LOOP_PARAM_TWO_STAGE=1 FAMS=50 REPS=1 WARMUPS=3 \
MAX_WAVE_SIZE=32768 \
python profiling/proposal8/bench_uniform_backward.py \
  --cuda-graph --cuda-graph-target=pi_backward \
  --graph-fixed-schedule-mode=no_cpu --cuda-graph-check
```

| Metric | Result |
|---|---:|
| normal Pi backward | `95.883 ms` |
| graph replay | `94.196 ms` |
| max output absolute difference | `3.125e-02` on `grad_log_pS` |
| relative to that key's max | `1.9925e-07` |
| normal peak allocation | `10.381 GB` |
| graph peak allocation | `15.489 GB` |

#### Event-timed benchmarks

All benchmark rows below use 50 families with `MAX_WAVE_SIZE=32768`.

| Run | Median | Mean | Min | Peak allocation | Log |
|---|---:|---:|---:|---:|---|
| profiling-worker baseline before implementation | `103.449 ms` | `103.472 ms` | `102.599 ms` | `10.308 GB` | `/tmp/gpurec_profile/prop4_selfloop/baseline_bench_50fams_default.log` |
| current default after implementation, flag off | `101.799 ms` | `101.914 ms` | `101.478 ms` | `10.308 GB` | `current_default_bench_50fams.log` |
| current default after tuned commit, flag off | `101.615 ms` | `101.653 ms` | `101.374 ms` | `10.308 GB` | `current_default_bench_50fams_after_tuned.log` |
| two-stage initial default, `TILE_ROWS=16`, `BLOCK_S=128` | `103.930 ms` | `104.106 ms` | `103.548 ms` | `10.390 GB` | - |
| final tuned default, `BLOCK_S=256`, `REPS=5`, `WARMUPS=5` | `102.722 ms` | `102.869 ms` | `102.450 ms` | `10.390 GB` | `/tmp/gpurec_profile/prop4_selfloop/twostage_bench_50fams_default_tuned.log` |

Tuning sweep medians:

| Sweep | Median results |
|---|---|
| `tile_rows=8/16/32/64`, `block_s=128`, `REPS=3` | `104.469`, `104.030`, `103.624`, `106.423 ms` |
| `tile_rows=32`, `block_s=64/128/256`, `REPS=3` | `104.182`, `104.278`, `102.959 ms` |
| `block_s=256`, `tile_rows=8/16/32`, `REPS=5` | `103.253`, `103.179`, `103.946 ms` |
| `tile_rows=16`, `block_s=192/256/512`, `REPS=3` | `102.930`, `103.268`, `103.881 ms` |
| `reduce_tiles=8/16/32/64`, `tile_rows=32`, `block_s=256`, `REPS=3` | `104.932`, `103.480`, `104.067`, `105.557 ms` |

Against the back-to-back flag-off median `101.615 ms`, the final tuned
two-stage default is `1.107 ms` slower, a `+1.1%` regression, and uses
`0.082 GB` more peak memory.

#### Nsight Systems, 50 families

Baseline profile:

```text
/tmp/gpurec_profile/prop4_selfloop/nsys_baseline_50fams_default_profileapi.nsys-rep
```

| Kernel bucket | Time | Launches | Share / notes |
|---|---:|---:|---|
| `_dts_cross_backward_accum_kernel` | `27.448 ms` | 33 | `30.2%` |
| `_wave_backward_uniform_kernel` | `24.477 ms` | 36 | `27.0%`, avg `679.9 us`, max `4.689 ms` |
| `_uniform_cross_pibar_vjp_tree_from_ud_compact_kernel` | `16.232 ms` | 33 | `17.9%` |
| `_dts_parent_reduced_ge2_stage1_kernel` | `7.201 ms` | 6 | `7.9%` |

Tuned two-stage profile:

```text
/tmp/gpurec_profile/prop4_selfloop/nsys_twostage_50fams_t16_bs256_profileapi.nsys-rep
```

| Kernel bucket | Time | Launches | Share |
|---|---:|---:|---:|
| `_dts_cross_backward_accum_kernel` | `27.445606 ms` | 33 | `30.0%` |
| `_wave_backward_uniform_kernel` | `21.576885 ms` | 36 | `23.6%` |
| `_uniform_cross_pibar_vjp_tree_from_ud_compact_kernel` | `16.220101 ms` | 33 | `17.7%` |
| `_dts_parent_reduced_ge2_stage1_kernel` | `7.207845 ms` | 6 | `7.9%` |
| `_wave_backward_uniform_param_stage1_kernel` | `3.142828 ms` | 3 | `3.4%` |
| `_active_mask_from_rhs_absmax_kernel` | `2.957896 ms` | 49 | `3.2%` |
| `_dts_eq1_to_rows_kernel` | `2.765286 ms` | 9 | `3.0%` |
| `_wave_backward_uniform_param_stage2_kernel` | `0.516296 ms` | 3 | `0.6%` |

The main wave kernel saves about `2.90 ms` in aggregate
(`24.477 -> 21.577 ms`), but the new stage-1 and stage-2 kernels add about
`3.66 ms`.  Net kernel time therefore regresses by about `0.76 ms` before
including launch, allocation, and extra memory effects.

#### Nsight Compute, representative no-split launches

Baseline max-wave launch:

```text
/tmp/gpurec_profile/prop4_selfloop/ncu_wave_backward_uniform_launch35_grid32768_sections.ncu-rep
plus the atom-metrics raw CSV for the same launch
```

| Metric | Baseline `_wave_backward_uniform_kernel` |
|---|---:|
| launch shape | grid `32768`, block `256` |
| registers/thread | `40` |
| duration under NCU replay | `5.10 ms` |
| memory throughput | `571.65 GB/s` |
| compute-memory throughput | `87.51%` peak |
| DRAM throughput | `58.09%` |
| L2 throughput | `87.51%` |
| DRAM bytes | `2.915 GB` total: `1.011 GB` read, `1.904 GB` write |
| global RED requests / sectors | `10,846,208 / 41,484,288` |
| global atom requests | `0` |
| L2 atomic input active | `11.33%` |
| theoretical occupancy | `100%` |
| achieved occupancy | `99.41%` |
| active warps/SM | `47.72` |

Two-stage main-wave max launch:

```text
/tmp/gpurec_profile/prop4_selfloop/ncu_twostage_wave_backward_uniform_launch35_t16_bs256.ncu-rep
/tmp/gpurec_profile/prop4_selfloop/ncu_twostage_wave_backward_uniform_launch35_atom_metrics.ncu-rep
```

| Metric | Two-stage `_wave_backward_uniform_kernel` |
|---|---:|
| launch shape | grid `32768`, block `256` |
| registers/thread | `40` |
| duration under NCU replay | `3.72-3.74 ms` |
| memory throughput | `739.74 GB/s` |
| memory throughput vs peak | `92.91%` |
| DRAM throughput | `75.24%` |
| L2 throughput | `92.91%` |
| DRAM bytes | `2.762 GB` total: `0.790 GB` read, `1.973 GB` write |
| global RED requests / sectors | `0 / 0` |
| L2 atomic input active | `0%` |
| theoretical occupancy | `100%` |
| achieved occupancy | `97.99%` |

This confirms the local objective: the eligible main wave no longer issues
global RED operations, and the large no-split launch is about `27%` faster
under NCU replay.

Two-stage stage 1:

```text
/tmp/gpurec_profile/prop4_selfloop/ncu_twostage_param_stage1_launch2_t16_bs256.ncu-rep
```

| Metric | `_wave_backward_uniform_param_stage1_kernel` |
|---|---:|
| launch shape | grid `(2048, 8)`, block `256` |
| registers/thread | `199` |
| dynamic shared memory | `16.384 KB` |
| duration | `1.25 ms` |
| memory throughput | `735.24 GB/s` |
| memory throughput vs peak | `74.78%` |
| DRAM throughput | `74.78%` |
| L2 throughput | `29.21%` |
| L2 hit rate | `21.59%` |
| executed instructions | `366,788,608` |
| theoretical occupancy | `16.67%` |
| achieved occupancy | `16.63%` |
| active warps/SM | `7.98` |
| local spilling | `0` |

Stage 1 is limited by register/shared-memory occupancy and rereads
`Pi/Pibar/constants/v_k` to recompute the final parameter weights.

Two-stage stage 2:

```text
/tmp/gpurec_profile/prop4_selfloop/ncu_twostage_param_stage2_launch2_t16_bs256.ncu-rep
```

| Metric | `_wave_backward_uniform_param_stage2_kernel` |
|---|---:|
| launch shape | grid `8`, block `128` |
| registers/thread | `168` |
| duration | `215.55 us` |
| memory throughput | `394.18 GB/s` |
| memory throughput vs peak | `40.10%` |
| achieved occupancy | `8.33%` |
| active warps/SM | `4.0` |

Stage 2 is small-grid underutilized, but it costs only about `0.17-0.21 ms`
per large no-split wave.

#### Decision

Keep the two-stage self-loop parameter-reduction path as an opt-in diagnostic
prototype.  Do not enable it by default for the current 50-family workload.

The proposal is technically valid: it eliminates global RED operations from
eligible no-split self-loop waves and speeds up the main wave kernel.  The
end-to-end timing regresses because stage 1 recomputes much of the final weight
calculation in a separate pass.  That pass has very high register pressure
(`199` registers/thread), low occupancy (`~16.6%`), extra DRAM traffic, and
partial writes.  Those costs exceed the atomic savings.

This line is worth revisiting only if stage 1 can be fused into a lower-register
CUDA/shared-memory kernel, if partial reductions can be accumulated inside the
existing wave kernel without recomputing terms, or if the target workload has
much larger no-split batches where scalar/vector atomics dominate more strongly.

## Proposal 5: zero-fill and allocation audit, not generic scratch pooling

The current Nsight trace still contains:

```text
501 small PyTorch FillFunctor launches: 2.194 ms
1 large PyTorch FillFunctor launch:     1.313 ms
378 D2D copies:                         0.389 ms
```

The previous scratch-pool proposal did not move wall time enough to become a
default.  A narrower audit should focus on zero work that kernels immediately
overwrite.  Examples to inspect:

- inactive-row zero stores in `_wave_backward_uniform_kernel`;
- `grad_mt_partial.zero_()` before every two-stage DTS reduction;
- temporary `dts_r` buffers where every active output row is overwritten;
- side-active and `pibar_A` buffers where inactive rows are already guarded by
  an active mask;
- the initial full `accumulated_rhs` zero fill.

The key distinction from scratch pooling is that this proposal removes fills;
it is not just reusing allocations.

Potential implementation patterns:

```text
1. make kernels fully overwrite inactive slices that downstream kernels read;
2. use generation counters or active-row masks instead of zeroing whole buffers;
3. make two-stage partial reducers write all partial slots, so no pre-zero is needed;
4. split the large accumulated_rhs initialization by wave if whole waves are
   known never to receive adjoints.
```

Expected gain:

- low-risk target `1-3 ms` at 50 families;
- memory peak may also improve if fewer scratch buffers need persistent zeros.

Risks:

- zero-elision bugs are easy to miss because stale values only affect inactive
  rows or rare branch combinations;
- every change needs parity tests with pruning on/off and with forced inactive
  rows.

### Proposal 5 tested results

Implemented in:

- `06385ea` Add opt-in inactive zero-store elision

Changed paths in that implementation:

- `gpurec/core/backward.py`
- `gpurec/core/kernels/wave_backward.py`
- `tests/kernels/test_dts_backward_accum_kernel.py`
- `tests/kernels/test_wave_backward_kernel.py`

#### Implementation shape

The implementation adds an opt-in flag:

| Env flag | Default status |
|---|---|
| `GPUREC_BACKWARD_SKIP_INACTIVE_ZERO_STORES=1` | off |

All behavior is unchanged unless the flag is set.

Self-loop path:

- `_wave_backward_uniform_kernel` can return immediately for inactive rows
  without zeroing `v_k` or scratch.
- This is allowed only when `gpurec/core/backward.py` proves
  `active_mask_for_wave_kernel` is present, `accum_param_grads` is enabled,
  and either the wave has no splits or `active_mask_for_split_kernels` is the
  same mask.
- The guard prevents stale `v_k` from being read by unmasked split or Pibar
  consumers.

DTS staged Pibar output path:

- `dts_cross_backward_accum_fused` accepts
  `skip_inactive_pibar_output_zero`.
- For inactive parent rows it can skip writing `pibar_ud` and `pibar_A` zeros
  when `output_pibar_ud` is active and the downstream from-UD Pibar VJP
  receives the same active mask.
- Under that same-active-mask invariant, stale inactive `pibar_ud` and
  `pibar_A` are masked by the consumer.
- `side_active` is still written to `false` for inactive sides when requested.

#### Audit findings

The audit found that most obvious fills are not redundant:

- `grad_mt_partial.zero_()` is needed because the kernel uses `atomic_add` into
  partial slots and the reducer reads all slots.
- Parent-ragged `pibar_A` zeros are needed because `pibar_A` receives atomics.
- `dts_r` `-inf` fill is dangerous to remove without a per-row has-DTS mask.
- `accumulated_rhs` zero is required except root rows because non-root rows are
  read and receive atomics.
- Global gradient zeros are required.

The two safe elisions are narrower:

- direct staged `pibar_A` and side-active inactive writes are redundant only
  under the same-active-mask invariant;
- inactive self-loop `v_k` zero is redundant only if all downstream consumers
  receive the same active mask, or if there are no splits.

#### Correctness checks

Worker checks:

| Command / check | Result |
|---|---:|
| `pytest -q tests/kernels/test_dts_backward_accum_kernel.py ... tests/gradients/test_autograd_bridge.py` | `76 passed` |
| `GPUREC_BACKWARD_SKIP_INACTIVE_ZERO_STORES=1 pytest -q tests/gradients/test_autograd_bridge.py` | `15 passed` |
| `py_compile` | passed |
| `git diff --check` | passed |

Main-agent checks:

| Command | Result |
|---|---:|
| `python -m py_compile gpurec/core/backward.py gpurec/core/kernels/wave_backward.py tests/kernels/test_dts_backward_accum_kernel.py tests/kernels/test_wave_backward_kernel.py` | passed |
| `GPUREC_BACKWARD_SKIP_INACTIVE_ZERO_STORES=1 pytest -q tests/kernels/test_wave_backward_kernel.py::test_wave_backward_skip_inactive_zero_stores_keeps_stale_rows_masked tests/kernels/test_dts_backward_accum_kernel.py::test_dts_staged_skip_inactive_pibar_output_zero_keeps_stale_rows_masked tests/gradients/test_autograd_bridge.py` | `17 passed in 2.73s` |

The added tests intentionally seed scratch buffers with sentinels.  They prove
that skipped inactive rows remain stale but are masked from all consumers, and
they compare active rows and outputs against the zero-writing baseline.

#### Event-timed benchmarks

All benchmark rows below use `FAMS=50`, `REPS=5`,
`MAX_WAVE_SIZE=32768`, except the explicit 10-family smoke row.

| Run | Mean | Median | Min | Peak allocation | Log |
|---|---:|---:|---:|---:|---|
| profiling-worker baseline before implementation, `WARMUPS=5` | `102.006 ms` | `101.909 ms` | `101.327 ms` | `10.308 GB` | - |
| main sequential post-commit, flag off, `WARMUPS=8` | `101.924 ms` | `101.916 ms` | `101.612 ms` | `10.308 GB` | `/tmp/gpurec_profile/prop5_zero_fill/main_default_after_impl_timing.txt` |
| main sequential post-commit, `GPUREC_BACKWARD_SKIP_INACTIVE_ZERO_STORES=1`, `WARMUPS=8` | `102.852 ms` | `102.743 ms` | `102.495 ms` | `10.308 GB` | `/tmp/gpurec_profile/prop5_zero_fill/main_skip_zero_after_impl_timing.txt` |

Worker smoke medians:

| Workload | Flag off | Opt-in flag on | Peak allocation |
|---|---:|---:|---:|
| 10 families | `34.485 ms` | `34.783 ms` | unchanged |
| 50 families | `103.092 ms` | `103.643 ms` | unchanged |

Against the back-to-back main sequential flag-off median `101.916 ms`, the
opt-in path is `0.827 ms` slower by median, about `+0.8%`, with no measured
peak-memory change.  A concurrent main timing pair that showed about `300 ms`
is intentionally excluded because the two benchmarks were run simultaneously.

#### Nsight Systems findings

Profiling-worker artifacts are under:

```text
/tmp/gpurec_profile/prop5_zero_fill
```

Baseline artifacts:

```text
baseline_nsys.{1..5}.nsys-rep
baseline_nsys.{1..5}.sqlite
/tmp/gpurec_profile/prop5_zero_fill/nsys_aggregate.txt
```

Opt-in artifacts:

```text
impl_skip_zero_nsys.{1..5}.nsys-rep
impl_skip_zero_nsys.{1..5}.sqlite
/tmp/gpurec_profile/prop5_zero_fill/impl_skip_zero_nsys_aggregate.txt
```

Profiled shape:

| Metric | Value |
|---|---:|
| `S` | `1999` |
| `G` | `50` |
| `C` | `321930` |
| waves | `49` |
| max wave size | `32768` |
| split rows | `402275` |
| leaves | `80545` |
| roots | `50` |

Baseline profiled timing and CUDA activity over 5 captured backward ranges:

| Metric | Total | Per rep / notes |
|---|---:|---|
| Nsight-profiled backward mean | - | `115.358 ms` with profiling overhead |
| GPU kernels | `14,590` | `2,918` per rep |
| aggregate GPU kernel time | `455.819 ms` | - |
| CUDA API launches | `14,590` | - |
| CUDA API copies/memsets | `2,605` | all `cudaMemcpyAsync`; no `cudaMemset` |
| CUDA API syncs | `1,025` | `995 cudaStreamSynchronize` + `30 cudaDeviceSynchronize` |
| malloc/free/alloc/memset runtime API records in capture | `0` | - |

Fill/zero-like baseline kernels over 5 captured backward ranges:

| Kernel bucket | Kernels | Time | Per-rep interpretation |
|---|---:|---:|---:|
| vectorized `FillFunctor<float>` | `2,505` | `11.194 ms` | - |
| unrolled `FillFunctor<float>` | `5` | `6.729 ms` | - |
| `FillFunctor<long>` | `145` | `0.144 ms` | - |
| total `FillFunctor` | `2,655` | `18.067 ms` | about `3.61 ms/rep` |

GPU copies over 5 captured backward ranges:

| Copy bucket | Copies | Bytes | Time |
|---|---:|---:|---:|
| D2D | `1,890` | `178.189 MB` | `1.946 ms` |
| D2H | `715` | `0.003 MB` | `0.626 ms` |

Top baseline kernels over 5 captured backward ranges:

| Kernel bucket | Launches | Time |
|---|---:|---:|
| `_dts_cross_backward_accum_kernel` | `165` | `137.240 ms` |
| `_wave_backward_uniform_kernel` | `180` | `124.323 ms` |
| `_uniform_cross_pibar_vjp_tree_from_ud_compact_kernel` | `165` | `81.256 ms` |
| `_dts_parent_reduced_ge2_stage1_kernel` | `30` | `36.038 ms` |
| `_active_mask_from_rhs_absmax_kernel` | `245` | `15.130 ms` |
| `_dts_eq1_to_rows_kernel` | `45` | `13.815 ms` |
| vectorized `FillFunctor<float>` | `2,505` | `11.194 ms` |

With `GPUREC_BACKWARD_SKIP_INACTIVE_ZERO_STORES=1`, the event-timed profiling
worker run reported:

| Metric | Result |
|---|---:|
| mean | `102.714 ms` |
| median | `102.669 ms` |
| min | `102.360 ms` |
| peak allocation | `10.308 GB` |

Nsight counts were unchanged under the opt-in flag:

| Metric | Baseline | Opt-in |
|---|---:|---:|
| kernels | `14,590` | `14,590` |
| copies | `2,605` | `2,605` |
| `FillFunctor` kernels | `2,655` | `2,655` |
| aggregate GPU kernel time | `455.819 ms` | `454.222 ms` |

The opt-in flag reduced aggregate GPU kernel time by only `1.597 ms` across
five profiled reps, or `0.35%`, while the event timing was noise/slower.

#### Interpretation

The implementation correctly removes some Triton inactive-row stores, but the
visible zero/fill overhead in this workload is dominated by PyTorch
`FillFunctor` kernels and copy/sync overhead, not by the inactive stores changed
by this patch.  That is why `FillFunctor`, copy, and launch counts are
unchanged under the flag.

The guarded elision is useful as a diagnostic prototype and proves the required
active-mask invariants.  It does not deliver a speedup on the measured
50-family workload.

The next useful Proposal 5 direction should target actual PyTorch
`FillFunctor` sources, for example:

- large `accumulated_rhs` or gradient zeros;
- `grad_mt_partial.zero_()` through a non-atomic partial writer;
- parent-ragged `pibar_A` zero by changing atomics to stores where ownership is
  unique;
- replacing PyTorch zero/full allocations with kernels that fully initialize
  only consumed regions.

Each of those needs deeper correctness work than this narrow inactive-store
elision.

#### Decision

Keep `GPUREC_BACKWARD_SKIP_INACTIVE_ZERO_STORES=1` as an opt-in diagnostic
prototype.  Do not enable it by default and do not promote this implementation
as a performance path.

## Proposal 6: species-dimension Euler layout for uniform Pibar

Uniform Pibar VJP ultimately needs subtree sums.  The current compact tree
kernel computes them by walking species-tree levels and using global memory as
the per-row buffer:

```text
for level in levels:
    u[parent] += u[child1] + u[child2]
contrib[s] = p_prime[s] * (A - u[s])
```

If species were laid out in a DFS/Euler order, each node's descendants would be
a contiguous interval.  Then the subtree sum can be expressed as an interval
sum:

```text
prefix[t + 1] = prefix[t] + u[dfs_species[t]]
subtree_sum[node] = prefix[end[node] + 1] - prefix[start[node]]
```

This does not reduce the asymptotic work below `O(S)` per row, but it changes
the access pattern:

- descendant queries become contiguous interval reads instead of level-wise
  parent/child gathers;
- species topology arrays shrink to `start/end/permutation`;
- child and parent rows are more likely to have useful locality in all uniform
  kernels;
- a CUDA shared-memory implementation can do row prefix sums without global
  tree scratch.

This is a deeper layout refactor than Proposal 1.  It may affect forward,
backward, leaf indexing, species parameter vectors, and user-visible gradient
ordering.  It should be treated as a separate branch with heavy parity tests.

Expected gain:

- `5-15 ms` at 50 families if it replaces the current Pibar tree scratch and
  improves species-topology locality in the self-loop kernel;
- additional memory simplification for ancestor/uniform operators.

Risks:

- broad codebase blast radius;
- prefix scans per row still need an efficient implementation;
- if the current species order is already close to topological, the locality
  gain may be smaller than expected.

First diagnostic:

Build only a CPU-side permutation report: measure whether current species ids
already make descendant sets contiguous, and estimate the average interval
length/fragment count under the current order.  If descendant sets are already
mostly contiguous, this proposal drops in priority.

### Proposal 6 tested results

No implementation branch was needed for the first diagnostic.  A CPU-side
species-order report was run against the fourth-pass benchmark species tree:

```text
dataset: tests/data/test_trees_1000
species tree: tests/data/test_trees_1000/sp.nwk
mode: global
pibar_mode: uniform
device: cpu
dtype: fp32
```

The report reconstructed parent/child species pointers from
`species_helpers["s_P_indexes"]` and `species_helpers["s_C12_indexes"]`, then
counted how many contiguous species-id intervals are needed to represent each
node's descendant set under the current order.

| Metric | Value |
|---|---:|
| species nodes | `1999` |
| root count | `1` |
| internal nodes | `999` |
| non-contiguous descendant sets, all nodes | `0 / 1999` |
| non-contiguous descendant sets, internal nodes | `0 / 999` |
| average fragment count, all nodes | `1.000` |
| average fragment count, internal nodes | `1.000` |
| max fragment count | `1` |
| average fill ratio | `1.000` |
| subtree-size-weighted average fragment count | `1.000` |

Largest subtree intervals were also contiguous:

| Node | Fragments | Descendants | Interval width | Fill ratio |
|---:|---:|---:|---:|---:|
| `1998` | `1` | `1999` | `1999` | `1.000000` |
| `1550` | `1` | `1551` | `1551` | `1.000000` |
| `1549` | `1` | `859` | `859` | `1.000000` |
| `1548` | `1` | `815` | `815` | `1.000000` |
| `690` | `1` | `691` | `691` | `1.000000` |

A DFS preorder pass preserving the current child order produced identical
subtree interval sizes; every `end[node] - start[node] + 1` matched the
descendant count exactly.

#### Interpretation

The current species ids already have the key Euler-layout property Proposal 6
was meant to introduce: every subtree is represented by one contiguous interval.
That means a broad species-dimension relayout is unlikely to provide the
expected locality win on this workload.  The potential remaining work is not a
global permutation, but a narrower kernel change that exploits the existing
interval property with explicit `start/end` arrays or per-row prefix scans.

#### Diagnostic decision

Drop Proposal 6 as a broad layout refactor for the current benchmark tree.  The
current species ids already behave like a postorder/Euler layout for subtree
queries, so a codebase-wide species permutation is not justified by this
workload.  The useful follow-up is narrower: exploit the existing contiguous
subtree intervals directly inside the uniform `Pibar` VJP.

#### Interval-prefix implementation

Implemented in:

- `b5c63c5` Add species Euler layout diagnostic
- `1b50fcf` Add opt-in Euler-prefix Pibar VJP

Changed paths in that implementation:

- `gpurec/core/species_euler_layout.py`
- `gpurec/core/backward.py`
- `gpurec/core/kernels/wave_backward.py`
- `tests/unit/test_species_euler_layout.py`
- `tests/kernels/test_uniform_cross_pibar_vjp_kernel.py`

The implementation adds an opt-in flag:

| Env flag | Default status |
|---|---|
| `GPUREC_DTS_PIBAR_UD_EULER_PREFIX=1` | off |

The new `_uniform_cross_pibar_vjp_tree_from_ud_euler_prefix_kernel` is wired
through `Pi_wave_backward` only when the flag is enabled.  It builds cached
int32 `subtree_interval_start/end` arrays in the current species order, and
only uses them when every subtree is contiguous.  The kernel overwrites
`pibar_ud` with per-row inclusive prefix sums and computes:

```text
subtree_sum = prefix[end - 1] - prefix[start - 1]
```

This changes the staged uniform `Pibar` VJP from a level-wise compact
parent/child walk to a per-row interval-prefix calculation.  Kernel tests cover
both fp32 and fp64.

#### Correctness

Commands and results:

```bash
python -m py_compile \
  gpurec/core/backward.py \
  gpurec/core/kernels/wave_backward.py \
  gpurec/core/species_euler_layout.py \
  tests/kernels/test_uniform_cross_pibar_vjp_kernel.py

pytest -q \
  tests/unit/test_species_euler_layout.py \
  tests/kernels/test_uniform_cross_pibar_vjp_kernel.py
# 7 passed in 1.34s

GPUREC_DTS_PIBAR_UD_EULER_PREFIX=1 pytest -q \
  tests/gradients/test_autograd_bridge.py \
  tests/kernels/test_dts_backward_accum_kernel.py
# 73 passed in 3.08s
```

The finite-difference command:

```bash
GPUREC_DTS_PIBAR_UD_EULER_PREFIX=1 pytest -q \
  tests/gradients/test_wave_gradient.py::TestUniformExactFullChainFD::test_full_chain_gradient_uniform \
  tests/gradients/test_wave_gradient.py::TestSpecieswiseUniformExactFD::test_specieswise_uniform_gradient_matches_fd \
  tests/gradients/test_wave_gradient.py::TestGenewiseGradient::test_genewise_gradient_matches_fd
```

reported two passing tests.  The genewise test failed before backward because
`E_fixed_point` received `ancestors_T=None`; the same genewise test also fails
without the new flag, so this is treated as pre-existing and unrelated.

Direct 50-family parity on the same model:

| Metric | Value |
|---|---:|
| default loss | `107804.2734375` |
| Euler-prefix loss | `107804.2734375` |
| loss diff | `0` |
| grad max abs diff | `4.8828125e-04` |
| grad max relative to default max | `7.573488972e-08` |

#### Event timings

Workload:

```text
FAMS=50
REPS=15
WARMUPS=8
MAX_WAVE_SIZE=32768
peak allocation: 10.308 GB in both modes
```

Alternating timing pairs:

| Run | Default median | Euler-prefix median | Median gain |
|---|---:|---:|---:|
| `15a` | `102.159 ms` | `101.288 ms` | `0.871 ms` |
| `15b` | `102.776 ms` | `100.726 ms` | `2.050 ms` |

Full timing summaries:

| Run | Mean | Median | Min |
|---|---:|---:|---:|
| default `15a` | `102.284 ms` | `102.159 ms` | `101.648 ms` |
| Euler-prefix `15a` | `101.225 ms` | `101.288 ms` | `100.752 ms` |
| default `15b` | `102.829 ms` | `102.776 ms` | `102.134 ms` |
| Euler-prefix `15b` | `100.771 ms` | `100.726 ms` | `100.225 ms` |

The longer alternating runs show about `0.87-2.05 ms` median end-to-end speedup.
An earlier 5-rep pair had default `101.832 ms` versus Euler-prefix
`102.206 ms`, but that was treated as noise because the longer alternating
runs and Nsight Systems kernel timing both favored Euler-prefix.

#### Nsight Systems

Aggregate across five captured reps:

| Metric | Default | Euler-prefix |
|---|---:|---:|
| total kernel time | `455.570 ms` | `452.236 ms` |
| kernel launches | `14,590` | `14,590` |
| `Pibar` VJP kernel time | `81.090 ms` | `73.757 ms` |
| `Pibar` VJP launches | `165` | `165` |
| `Pibar` VJP regs/thread | `36` | `40` |
| `Pibar` VJP block size | `128` | `256` |

The staged `Pibar` VJP bucket saves `7.333 ms` over five reps, or about
`1.47 ms/rep`.  Total kernel time saves `3.334 ms` over five reps, or about
`0.67 ms/rep`, because other kernels and profiling noise varied.

#### Nsight Compute

Representative launches from `/tmp/gpurec_profile/prop6_euler`:

| Kernel | Durations | DRAM throughput | L2 throughput | SM throughput | Regs/thread | Achieved occupancy |
|---|---:|---:|---:|---:|---:|---:|
| compact tree | `0.228, 0.695, 1.223, 1.263, 2.250, 1.893 ms` | `71.95-86.30%` | `63.56-65.01%` | `24.91-25.52%` | `36` | `94.01-98.83%` |
| Euler-prefix | `0.180, 0.610, 1.120, 1.150, 2.090, 1.730 ms` | `91.19-93.42%` | `29.54-36.04%` | `13.00-15.33%` | `40` | `93.43-97.32%` |

Euler-prefix removes scattered level-walk topology traffic and pushes the
kernel closer to pure streaming DRAM bandwidth.  It is faster despite lower SM
throughput because the memory access pattern is more coalesced.  The gains are
bounded by the extra registers and the 2048-element cumulative sum per row.

Artifacts:

```text
/tmp/gpurec_profile/prop6_euler/default_timing_15a.txt
/tmp/gpurec_profile/prop6_euler/euler_prefix_timing_15a.txt
/tmp/gpurec_profile/prop6_euler/default_timing_15b.txt
/tmp/gpurec_profile/prop6_euler/euler_prefix_timing_15b.txt
/tmp/gpurec_profile/prop6_euler/default_nsys.*.sqlite/.nsys-rep
/tmp/gpurec_profile/prop6_euler/euler_prefix_nsys.*.sqlite/.nsys-rep
/tmp/gpurec_profile/prop6_euler/ncu_compact_pibar_50.ncu-rep
/tmp/gpurec_profile/prop6_euler/ncu_euler_prefix_pibar_50.ncu-rep
```

#### Final decision

Keep `GPUREC_DTS_PIBAR_UD_EULER_PREFIX=1` as a validated opt-in performance
path.  It is beneficial on the 50-family benchmark, but the speedup is modest
and workload-sensitive.  Do not enable it by default until more datasets,
species counts, and species-tree orderings are checked.

## Proposal 7: thresholded split-side Pibar pruning with an error budget

Current split-side pruning is exact-zero only:

```text
side_active[row] = max_s(abs(u_d[row, s])) != 0
```

This is safe but weak.  In floating-point backward pruning we already accept a
row-level `pruning_threshold`.  The same policy could be extended to split
sides with a conservative bound:

```text
u_inf = max_s abs(u_d[s])
A_abs = sum_s abs(u_d[s])
bound_on_contrib_inf <= max_s(p_prime[s]) * (A_abs + subtree_abs_inf)
```

The exact bound needs care, but the purpose is simple: skip a Pibar side when
its maximum possible contribution to `accumulated_rhs[child, :]` is below the
same derivative tolerance used elsewhere.

Expected gain:

- workload-dependent; exact-zero side skipping already helped in earlier
  profiles, but many nonzero sides may still be numerically negligible;
- possible `0-6 ms` at 50 families if high-fanout root waves have many tiny
  `vd1/vd2` sides.

Risks:

- this is intentionally approximate and should not be enabled by default until
  we have a documented gradient-error bound;
- finite-difference comparisons need tolerances tied to the pruning threshold.

This is a good research branch, not an immediate production default.

### Proposal 7 tested results

Implemented as an opt-in approximate pruning path behind:

```text
GPUREC_DTS_PIBAR_UD_SIDE_THRESHOLD=<non-negative float>
GPUREC_DTS_PIBAR_UD_SIDE_BUDGET=<non-negative float>  # alias
```

Committed implementation: `c6efdf5 Add opt-in thresholded Pibar side pruning`.

The implementation extends the fused DTS backward accumulation path.  Exact-zero
side skipping remains unchanged and default behavior stays exact.  When a
positive threshold or side-budget alias is set, `Pi_wave_backward` also enables
staged Pibar side-active output even if exact-zero side skipping is unset.  The
threshold is passed to Triton as a one-element device tensor to avoid scalar
specialization/cache churn.  The DTS accumulation kernel computes `pibar_ud` and
`pibar_A` exactly as before, but changes the side-active predicate from exact
nonzero to an audited tight local bound:

```text
side_active = sum_s(abs(u_d[s])) > GPUREC_DTS_PIBAR_UD_SIDE_THRESHOLD
```

The downstream compact Pibar-from-UD kernel then skips sides whose bound is
under the threshold.  The default threshold remains `0`, so production behavior
is exact unless the user opts into the approximate path.

#### Correctness

Final checks:

| Check | Result |
|---|---:|
| `python -m py_compile gpurec/core/backward.py gpurec/core/kernels/wave_backward.py tests/kernels/test_dts_backward_accum_kernel.py` | passed |
| `pytest -q tests/kernels/test_dts_backward_accum_kernel.py tests/gradients/test_autograd_bridge.py` | `79 passed in 12.64s` |
| `GPUREC_DTS_PIBAR_UD_SIDE_THRESHOLD=1e-12 pytest -q tests/gradients/test_autograd_bridge.py tests/kernels/test_dts_backward_accum_kernel.py::test_dts_staged_pibar_ud_side_threshold_uses_error_budget_bound tests/kernels/test_dts_backward_accum_kernel.py::test_uniform_cross_pibar_from_ud_threshold_masks_small_nonzero_sides` | `20 passed in 3.04s` |

The new threshold test verifies that the thresholded side mask matches
`sum(abs(pibar_ud)) > threshold`, that `pibar_ud`, `pibar_A`, and direct DTS RHS
outputs are unchanged, and that the RHS difference after the downstream Pibar
VJP is bounded by the skipped-side absolute mass.  This is a local kernel bound,
not an end-to-end gradient-error guarantee.

Direct 50-tree gradient diffs against the exact-zero default on the same model:

| Threshold | Loss diff | Grad max abs diff | Grad relative diff |
|---:|---:|---:|---:|
| `1e-12` | `0` | `4.8828125e-04` | `7.573487251e-08` |
| `1e-10` | `0` | `4.8828125e-04` | `7.573487251e-08` |
| `1e-8` | `0` | `1.46484375e-03` | `2.272046175e-07` |
| `1e-6` | `0` | `5.810546875e-02` | `9.012449828e-06` |
| `1e-5` | `0` | `1.227050781` | `1.903217346e-04` |
| `1e-4` | `0` | `2.700683594` | `4.188895798e-04` |

On the 50-family benchmark shape:

```text
S=1999, C=321930, waves=49, split_rows=402275
waves_with_pibar_ud=33
total staged side rows=503358, parent-active side rows=483372
```

The side-count scan from
`/tmp/gpurec_profile/prop7_side_threshold/side_threshold_stats_50.txt` uses the
`sum(abs(u_d))` threshold and shows that the approximation is aggressive even
for tiny thresholds:

| Threshold | Active parent sides | Skipped parent sides | Skipped parent fraction |
|---:|---:|---:|---:|
| exact zero | `344227` | `139145` | `28.79%` |
| `1e-12` | `163475` | `319897` | `66.18%` |
| `1e-9` | `162312` | `321060` | `66.42%` |
| `1e-6` | `161678` | `321694` | `66.55%` |
| `1e-5` | `11036` | `472336` | `97.72%` |
| `1e-4` | `5367` | `478005` | `98.89%` |

The high-fanout late waves are where the threshold is most destructive: for the
largest `84310`-side launch, exact-zero side skipping leaves `48436` active
sides, `1e-12` leaves `579`, and `1e-6` leaves only `124`.

#### Event-timed benchmarks

Clean rows below use `FAMS=50`, `REPS=9`, `WARMUPS=10`,
`MAX_WAVE_SIZE=32768`.  Earlier contaminated 7-rep sweep results with
`1e-12` and `1e-5` around `300/270 ms` are discarded.

| Variant | Mean backward | Median backward | Min backward | Peak allocation | Source |
|---|---:|---:|---:|---:|---|
| default exact-zero side skipping | `104.021 ms` | `103.902 ms` | `103.425 ms` | `10.308 GB` | `/tmp/gpurec_profile/prop7_side_threshold/timing2_default.txt` |
| `GPUREC_DTS_PIBAR_UD_SIDE_THRESHOLD=1e-12` | `95.849 ms` | `95.738 ms` | `95.546 ms` | `10.308 GB` | `/tmp/gpurec_profile/prop7_side_threshold/timing2_1e-12.txt` |
| `GPUREC_DTS_PIBAR_UD_SIDE_THRESHOLD=1e-8` | `109.346 ms` | `99.943 ms` | `95.858 ms` | `10.308 GB` | noisy row |
| `GPUREC_DTS_PIBAR_UD_SIDE_THRESHOLD=1e-6` | `95.524 ms` | `95.747 ms` | `94.251 ms` | `10.308 GB` | `/tmp/gpurec_profile/prop7_side_threshold/timing2_1e-6.txt` |
| `GPUREC_DTS_PIBAR_UD_SIDE_THRESHOLD=1e-5` | `90.207 ms` | `89.369 ms` | `88.730 ms` | `10.308 GB` | `/tmp/gpurec_profile/prop7_side_threshold/timing2_1e-5.txt` |
| `GPUREC_DTS_PIBAR_UD_SIDE_THRESHOLD=1e-4` | `88.778 ms` | `88.763 ms` | `88.133 ms` | `10.308 GB` | `/tmp/gpurec_profile/prop7_side_threshold/timing2_1e-4.txt` |

The `1e-8` row has outliers (`125.314`, `154.997`, `110.298 ms`), so it is not
clean median evidence.  The `1e-12` row is the best conservative timing signal:
it saves about `8.164 ms` median against the exact-zero default while keeping
the measured 50-tree relative gradient diff below `1e-7`.  Higher thresholds
are faster but visibly increase gradient difference.

#### Profiling

Nsight Systems aggregate across five captured reps:

| Metric | Default exact-zero | Threshold `1e-12` | Threshold `1e-6` |
|---|---:|---:|---:|
| total GPU kernel time | `455.551 ms` | `413.736 ms` | `414.012 ms` |
| kernels | `14590` | `14590` | `14590` |
| Pibar compact bucket | `81.105 ms` | `39.208 ms` | `39.203 ms` |
| DTS accumulation bucket | `137.209 ms` | `137.720 ms` | `137.781 ms` |
| self-loop wave bucket | `124.282 ms` | `124.027 ms` | `124.148 ms` |

The Pibar bucket saves about `41.9 ms` over five reps, or about `8.38 ms` per
rep.  The extra DTS `sum(abs)` accounting costs only about `0.5 ms` over five
reps, or about `0.1 ms` per rep.  Launch count is unchanged.

Nsight Compute samples on the largest compact Pibar-from-UD launch show why side
skipping helps: the kernel remains memory-bound, but exact-zero side skipping
cuts the processed side traffic substantially.

| Metric | Side skipping off | Exact-zero side skipping |
|---|---:|---:|
| representative launch duration | `3.837 ms` | `2.264 ms` |
| grid / block | `84310 / 128` | `84310 / 128` |
| registers/thread | `36` | `36` |
| DRAM read | `2.014 GB` | `1.164 GB` |
| DRAM write | `1.311 GB` | `0.745 GB` |
| DRAM throughput | `88.13%` | `85.79%` |
| global load instructions | `50.477 M` | `29.496 M` |
| global store instructions | `3.857 M` | `2.228 M` |
| global RED instructions | `5.282 M` | `3.051 M` |

The thresholded `1e-12` NCU sample
(`/tmp/gpurec_profile/prop7_side_threshold/ncu_pibar_threshold_1e-12_50.ncu-rep`,
launch skip `33`, count `6`, basic set) sampled compact Pibar-from-UD durations
of `16.13 us`, `26.53 us`, `37.92 us`, `37.38 us`, `58.11 us`, and `50.78 us`
for grids `7970`, `27092`, `48458`, `46690`, `84310`, and `72400`.
Throughput is low in these thresholded samples, roughly `10-25%` DRAM and
`6-16%` SM, because most CTAs return immediately after `side_active == false`.

Artifacts:

```text
/tmp/gpurec_profile/prop7_side_threshold/timing2_default.txt
/tmp/gpurec_profile/prop7_side_threshold/timing2_1e-12.txt
/tmp/gpurec_profile/prop7_side_threshold/timing2_1e-6.txt
/tmp/gpurec_profile/prop7_side_threshold/timing2_1e-5.txt
/tmp/gpurec_profile/prop7_side_threshold/timing2_1e-4.txt
/tmp/gpurec_profile/prop7_side_threshold/nsys_default.*.sqlite/.nsys-rep
/tmp/gpurec_profile/prop7_side_threshold/nsys_1e-12.*.sqlite/.nsys-rep
/tmp/gpurec_profile/prop7_side_threshold/nsys_1e-6.*.sqlite/.nsys-rep
/tmp/gpurec_profile/prop7_side_threshold/ncu_pibar_threshold_1e-12_50.ncu-rep
```

#### Final decision

Keep Proposal 7 default-off as a research/diagnostic opt-in.  Exact-zero side
skipping is already valuable and remains part of the benchmark default flags.
The `1e-12` threshold gives most of the speedup, about `8 ms` median on the
50-tree case, with a tiny measured relative gradient difference there.  It still
skips about `66%` of parent-active Pibar sides and lacks a global accumulated
error budget, so it should not be promoted without an end-to-end error budget
tied to `pruning_threshold` and optimizer-level convergence tests.

## Proposal 8: split metadata and topology compression audit

### Tested report

This report supersedes the stale species-topology text that used to occupy this
section.  The final tested implementation is the two-commit split-metadata
change:

- `707bcbf Add int32 wave split metadata`
- `be4706f Add split metadata dtype fallback`

The implementation promotes Proposal 8's split-id storage change, not a new
wave/topology algorithm.  Wave layout now stores split metadata as int32 by
default, with `GPUREC_WAVE_SPLIT_METADATA_INT32=0` preserving the old `long`
layout.  This is a correctness-preserving memory cleanup with neutral latency at
50 families and a small possible benefit at 100 families; it is not a meaningful
latency optimization by itself.

### What changed technically

`build_wave_layout` now selects an `index_dtype` for per-wave split metadata.
The default is `torch.int32`; setting `GPUREC_WAVE_SPLIT_METADATA_INT32=0`
selects `torch.long`.  The default path guards against impossible int32 clade
ids by raising if `C > torch.iinfo(torch.int32).max`.

The implementation deliberately keeps compatibility boundaries explicit:

- PyTorch fallback and indexing sites call `.long()` before using split metadata
  as PyTorch indices.
- Triton kernels accept the int32 tensors but cast loaded ids to `tl.int64`
  before pointer arithmetic and row offset calculations.
- The optional CUDA Pibar-from-`u_d` prototype accepts int32 inputs at the Python
  boundary, then widens them to contiguous int64 tensors internally because the
  prototype CUDA signature still uses `long long*`.
- `ge2_ptr` remains `torch.long`; it is a CSR pointer vector passed to
  `seg_logsumexp`, which asserts int64 pointers.

This means the storage dtype changed, while the address/index arithmetic remains
64-bit where the current PyTorch, Triton, and CUDA call sites require it.

### Metadata dtypes after the audit

| Metadata | Final dtype/path | Reason |
|---|---|---|
| `sl`, `sr` | int32 by default; long with `GPUREC_WAVE_SPLIT_METADATA_INT32=0` | global child clade ids fit int32 in the tested layouts; storage is halved |
| `reduce_idx` | int32 by default; long with fallback flag | wave-local parent row ids fit int32 and are loaded by Triton/CUDA paths |
| `eq1_reduce_idx` | int32 by default; long with fallback flag | view/slice of the same parent-id storage used for single-split parents |
| `ge2_parent_ids` | int32 by default; long with fallback flag | wave-local parent ids for multi-split parent groups |
| `ge2_ptr` | kept long | CSR pointer vector for `seg_logsumexp`; that helper requires int64 ptrs |
| `perm`, `inv_perm`, `wave_starts` | kept long | global layout/permutation tensors used directly by PyTorch indexing |
| `leaf_row_index`, `leaf_col_index`, `leaf_species_index` | kept long | PyTorch indexing and sentinel-compatible leaf metadata |
| `root_clade_ids`, `original_root_clade_ids`, `family_idx` | kept long | PyTorch indexing and family/root bookkeeping |
| species topology and compact level topology | separate from this change | previous topology-int32 work remains controlled by its own paths; Proposal 8 here is split metadata |

### Correctness evidence

Final correctness checks on 2026-05-04:

| Command | Result |
|---|---:|
| `py_compile` for touched implementation/profiling files | passed |
| focused new dtype/fallback tests | `3 passed` |
| `pytest -q tests/kernels/test_dts_fused_kernel.py tests/kernels/test_dts_backward_accum_kernel.py tests/gradients/test_autograd_bridge.py` | `107 passed in 17.28s` |
| supervisor rerun combining py_compile, wave-layout fallback tests, DTS tests, and autograd bridge | `109 passed in 15.29s` |

The parity checks compare the final int32 default against the long-metadata
fallback:

| Case | Loss diff | Grad max abs | Grad relative |
|---|---:|---:|---:|
| 10 families | `0` | `1.220703125e-4` | `9.14e-8` |
| 50 families | `0` | `1.46484375e-3` | `2.272e-7` |

These differences are in the expected fp32 accumulation-noise range.  The long
fallback test is important because it verifies `GPUREC_WAVE_SPLIT_METADATA_INT32=0`
still constructs long `sl`/`sr`/`reduce_idx`/`eq1_reduce_idx`/`ge2_parent_ids`
metadata while leaving `ge2_ptr` long in both modes.

### Timing and profiling evidence

CUDA-event medians show the expected shape: memory improves, latency is mostly
neutral.

| Case and run order | Long median | Int32 median | Int32 minus long | Read |
|---|---:|---:|---:|---|
| 10 families | `34.500 ms` | `34.766 ms` | `+0.266 ms` | neutral |
| 50 families, long first | `103.323 ms` | `103.593 ms` | `+0.270 ms` | neutral/slightly worse, with outliers |
| 50 families, int32 first | `102.961 ms` | `104.728 ms` | `+1.767 ms` | order-sensitive, not a win |
| 100 families, long first | `188.837 ms` | `187.401 ms` | `-1.436 ms` | small possible win |
| 100 families, int32 first | `188.452 ms` | `187.293 ms` | `-1.159 ms` | small possible win |

Peak memory is the clearer benefit:

| Case | Peak-memory effect |
|---|---:|
| 50 families | about `6 MB` saved |
| 100 families | about `12 MB` saved |

Nsight Systems on the 50-family case confirms the hot kernel mix did not change
materially:

| Mode | Summed kernel time | Kernel launches | Read |
|---|---:|---:|---|
| long metadata fallback | `91.660 ms` | 2918 | baseline for the final code with `GPUREC_WAVE_SPLIT_METADATA_INT32=0` |
| default int32 metadata | `90.977 ms` | 2942 | same DTS/Pibar buckets within noise; small launch-count increase from compatibility work |

Nsight Compute for the DTS accumulation kernel is also essentially unchanged:
both modes use 96 registers/thread, `41.67%` theoretical occupancy, about
`41.5%` achieved occupancy, about `69-73%` DRAM throughput, and about `33-35%`
SM throughput.

The resource interpretation is therefore straightforward.  Split ids are scalar
metadata, while the dominant kernels still move GB-scale Pi/Pibar/DTS row data.
Halving the metadata storage reduces persistent wave-layout/cache footprint and
shows up as the measured peak-memory savings.  It does not remove the need for
64-bit address math: Triton widens ids before pointer arithmetic, PyTorch
fallbacks create `.long()` compatibility views for indexing, and the optional
CUDA Pibar prototype widens to match its current `long long*` signature.  Those
compatibility costs explain why the 50-family timing is neutral to slightly
worse even though memory use is cleaner.

### Decision

Keep the default int32 split metadata and keep
`GPUREC_WAVE_SPLIT_METADATA_INT32=0` as the conservative long fallback.

This is correct and low-risk, and it gives a small, scaling memory cleanup:
about `6 MB` at 50 families and about `12 MB` at 100 families in the tested
profile.  It should not be counted as a meaningful latency optimization.  The
50-family latency is neutral or slightly worse, and the 100-family timing shows
only about a `1 ms` possible benefit.  Do not spend more effort compressing this
metadata unless future source counters show scalar split-id loads as a real hot
path, or unless the PyTorch/CUDA fallback boundaries are removed so int32 ids can
stay int32 end to end.

Artifacts:

```text
/tmp/gpurec_profile/prop8_int32_metadata/
/tmp/gpurec_profile/prop8_int32_metadata/baseline/summary.txt
/tmp/gpurec_profile/prop8_int32_metadata/be4706f_int32/logs/
/tmp/gpurec_profile/prop8_int32_metadata/be4706f_long/
/tmp/gpurec_profile/prop8_int32_metadata/nsys_50_i32_0.nsys-rep
/tmp/gpurec_profile/prop8_int32_metadata/nsys_50_i32_0.sqlite
/tmp/gpurec_profile/prop8_int32_metadata/nsys_50_i32_1.nsys-rep
/tmp/gpurec_profile/prop8_int32_metadata/nsys_50_i32_1.sqlite
/tmp/gpurec_profile/prop8_int32_metadata/ncu_dts_accum_i32_0.ncu-rep
/tmp/gpurec_profile/prop8_int32_metadata/ncu_dts_accum_i32_1.ncu-rep
/tmp/gpurec_profile/prop8_int32_metadata/ncu_dts_accum_basic_i32_0.ncu-rep
/tmp/gpurec_profile/prop8_int32_metadata/ncu_dts_accum_basic_i32_1.ncu-rep
```

## Ranked next plan

The best next experiments by expected value are:

| Rank | Proposal | Why first | Expected 50-family gain |
|---:|---|---|---:|
| 1 | Proposal 1, shared-memory staged Pibar VJP two-kernel variant | Directly attacks a pure memory-bound `15.937 ms` bucket | `4-8 ms` |
| 2 | Proposal 0, CUDA shared-memory self-loop for no-split waves | Largest self-loop launches still move almost `3 GB` each | `3-6 ms` for leaf-only, `6-12 ms` if generalized |
| 3 | Proposal 2, ragged parent-tile DTS | Fixes the specific failure mode of the rejected parent-tiled prototype | `3-8 ms` |
| 4 | Proposal 4, tailored self-loop param reductions | Targets visible global RED pressure without returning to six tensors | `2-6 ms` |
| 5 | Proposal 5, zero-fill audit | Smaller but lower-risk cleanup | `1-3 ms` |
| 6 | Proposal 3, CUDA graph segments | More useful after memory kernels shrink | `0-6 ms` |
| 7 | Proposal 6, Euler species layout | High upside but broad refactor | `5-15 ms`, high risk |
| 8 | Proposal 7, thresholded side pruning | Potentially useful but approximate | `0-6 ms`, opt-in |
| 9 | Proposal 8 follow-up | Implemented as default int32 split metadata with long fallback; revisit only if scalar ids become source-counter hot | none now |

## Measurement gates for the fourth wave

Every proposal should use the same gates:

1. Correctness:
   - current implementation versus candidate loss and `theta.grad` parity for
     3, 10, and 50 families;
   - `pytest -q tests/gradients/test_autograd_bridge.py`;
   - finite-difference check on at least one small uniform/global case;
   - fp32 and fp64 parity if the touched kernel supports fp64.
2. Timing:
   - CUDA-event backward-only timings outside Nsight;
   - at least two run orders for gains below `3 ms`;
   - 10, 50, and 100 family timings;
   - peak memory.
3. Nsight Systems:
   - total backward event;
   - summed kernel buckets;
   - launch count;
   - sync/copy counts;
   - whether time moved into helper kernels.
4. Nsight Compute:
   - representative largest launches;
   - DRAM read/write bytes;
   - L1/L2 sectors and hit rates when relevant;
   - global RED instructions/sectors;
   - registers/thread and achieved occupancy;
   - stall mix, especially long scoreboard, LG throttle, MIO throttle, and
     barrier stalls.

The most important rule for this fourth pass is to judge by bytes moved and
kernel buckets, not by occupancy alone.  The current hot kernels already run
enough CTAs; the bottleneck is where the row vectors live and how often they
are reread.
