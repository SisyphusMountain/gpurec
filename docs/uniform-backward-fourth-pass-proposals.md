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

## Proposal 8: split metadata and topology compression audit

Species topology already has an int32 path.  Split metadata still flows through
many kernels as int64 `sl`, `sr`, and `reduce_idx`.  The scalar metadata bytes
are small compared with Pi/Pibar traffic, so this is not a major expected win,
but it is easy to audit:

```text
50-family C = 321930
100-family C = 635372
1000-family chunks are still well below int32 range
```

Potential changes:

- store `sl`, `sr`, `reduce_idx`, `eq1_reduce_idx`, and ge2 parent ids as
  int32 in the wave layout;
- keep a long fallback only for PyTorch indexing paths that require it;
- pass int32 metadata into Triton/CUDA kernels by default.

Expected gain:

- probably `<1 ms` at 50 families;
- may reduce register pressure and memory sectors slightly in DTS/Pibar
  kernels;
- low risk if all PyTorch indexing fallbacks keep long tensors.

This should not interrupt higher-upside work, but it is a useful cleanup before
writing new CUDA kernels.

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
| 9 | Proposal 8, int32 split metadata | Cleanup, likely small | `<1 ms` |

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
