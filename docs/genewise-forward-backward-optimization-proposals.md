# Genewise Forward and Backward Optimization Proposals

Date: 2026-05-04

This document is a proposal pass for `mode="genewise"` with
`pibar_mode="uniform"`.  The goal is to transfer as much as possible from the
optimized uniform/global kernels while keeping the genewise semantics:

- each gene family has its own diversification parameters;
- the species topology and uniform Pibar topology are still shared;
- the cross-family wave scheduler should still batch many trees together to
  keep large waves resident on the GPU.

The important distinction is that "uniform" describes the Pibar model, while
"genewise" describes the parameter layout.  The genewise path should be the
uniform path plus family-indexed constants, not a separate dense/PyTorch-heavy
algorithm.

## Current Reference Point

Workload and machine:

```text
dataset: tests/data/test_trees_1000
species: S=1999
GPU: NVIDIA RTX 4090, 24 GB
dtype: fp32 unless stated otherwise
Pi iterations: fixed 6
commit: 3b94b66
```

Current measured forward timings:

| Path | Chunk policy | Current time for 1000 trees | Notes |
|---|---:|---:|---|
| global/uniform forward | 7 chunks, `150x6 + 100`, `max_wave_size=32768` | `2.318 s` | optimized root-row likelihood path |
| genewise/uniform forward | 7 chunks, `150x6 + 100`, `max_wave_size=32768` | `4.468 s` | optimized leaf-index path, but still pre-expands per-family constants |

The current genewise forward is therefore about `1.93x` slower than the
global/uniform forward on the same 1000-tree dataset.

Current measured backward status:

| Path | Current status | Time |
|---|---|---:|
| global/uniform backward | optimized fused path, exact ancestor correction, 10 resident chunks of 100 families | `6.523 s` for 1000 trees |
| genewise/uniform backward | no accepted optimized 1000-tree entrypoint yet | not available |
| genewise generic autograd smoke | 10 families only, not the optimized target path | `548.620 ms` |

The 10-family generic genewise backward number should not be treated as a
production baseline.  It mainly proves that the current genewise backward is
not using the same fast fused uniform route as global mode.

## What Already Transfers

Several global/uniform optimizations already help genewise forward:

| Uniform technique | Genewise status | Effect |
|---|---|---|
| fixed 6 Pi iterations | already shared | removes convergence polling syncs |
| ping-pong Pi scratch | already shared when compact leaf path is enabled | removes large D2D copies |
| root-row output for likelihood-only forward | already shared | avoids returning full `[C,S]` output for inference |
| parent-reduced forward DTS | already shared | avoids materializing all high-fanout split rows when possible |
| int32 topology metadata | mostly shared | lowers topology bandwidth |
| compact uniform species preprocessing | shared by `pibar_mode="uniform"` | avoids dense species matrices |
| genewise leaf-index path | implemented specifically for genewise | avoids dense `[W,S]` leaf masks |

The genewise leaf-index change is the best example of the desired pattern.  It
did not invent a new algorithm.  It taught the existing uniform wave-step
kernel how to read family-specific leaf log-probabilities:

```text
mode 0: leaf_logp[s]                 # shared [S]
mode 1: leaf_logp[family]            # genewise scalar [G]
mode 2: leaf_logp[family, s]         # genewise specieswise [G,S]

t_leaf = leaf_logp_value if leaf_species[row] == s else -inf
```

That removed dense leaf tensors and allowed the genewise path to use the same
ping-pong fast path as global mode.

## Where Genewise Still Diverges

The forward setup still materializes per-wave constants:

```text
fi_w = family_idx[ws:we]

DL_w   = DL_const[fi_w]      # [W,S]
SL1_w  = SL1_const[fi_w]     # [W,S]
SL2_w  = SL2_const[fi_w]     # [W,S]
Ebar_w = Ebar[fi_w]          # [W,S]
E_w    = E[fi_w]             # [W,S]
mt_w   = mt[fi_w]            # [W,S]
```

The fused wave-step then sees `stride_c=S`, so every clade row behaves as if it
had distinct constants.  In global mode it sees `stride_c=0`, so the same
`[S]` constants are reused for all rows.

This is the main forward inefficiency.  It causes:

- PyTorch advanced indexing before many waves;
- large `[W,S]` temporary tensors;
- extra global-memory reads in `_wave_step_uniform_kernel`;
- weaker cache reuse for the same family constants across many rows.

The backward divergence is larger.  The current optimized fused uniform
backward is guarded by `G == 1`:

```text
can_use_fused_uniform_backward =
    fused enabled
    and G == 1
    and pibar_mode == "uniform"
    and cuda dtype in {fp32, fp64}
    and S > 256
```

When `G > 1`, the backward loop falls into generic PyTorch-heavy paths for
several pieces:

- dense leaf masks unless the fused path is active;
- generic self-loop VJP;
- materialized `DTS_5`;
- PyTorch `scatter_add_` into per-family parameter gradients;
- generic uniform Pibar VJP when the fused scalar-parameter route is not
  available.

This explains why genewise backward currently has no acceptable 1000-tree
optimized benchmark.

## Lessons From Uniform Work

The useful lessons are:

1. Keep row-wise log-space recurrences fused.  The wave-step and backward
   self-loop kernels are where `[rows,S]` traffic explodes.
2. Do not materialize `[rows,S]` intermediates just to immediately reduce or
   add them.  Parent-reduced DTS and compact Pibar scratch were wins for this
   reason.
3. Remove host syncs only after the GPU work is right.  Fixed-6 forward already
   removed the main syncs; backward pruning needs to remain exact unless an
   approximate mode is explicitly requested.
4. Do not reintroduce generic sparse matmul for uniform Pibar.  The tested
   SpMM/CSR routes lost because they added index streams and surrounding
   exp/log/copy work.  The good path keeps Pibar row max, row sum, ancestor
   correction, and DTS_L terms in the same row kernel.
5. Tensor cores are not a near-term target for the current kernels.  The hot
   operations are logsumexp, exp2/log2, tree reductions, gathers, and scatter
   accumulation, not dense matrix multiply.

## Proposal 0: Family-Indexed Constants Inside The Forward Wave Kernel

Replace per-wave `[W,S]` constant materialization with in-kernel family loads.

Current genewise forward:

```text
# Python/Torch before launch
fi_w = family_idx[ws:we]
DL_w = DL_const[fi_w]        # materializes [W,S]

# Triton
dl = load(DL_w + w*S + s)
```

Target genewise forward:

```text
# Python/Torch before launch
pass DL_const as [G,S]
pass family_idx as [C]

# Triton
family = load(family_idx + ws + w)
dl = load(DL_const + family*S + s)
```

The same change applies to:

- `DL_const`;
- `SL1_const`;
- `SL2_const`;
- `Ebar`;
- `E`;
- `mt`.

The wave-step kernel already has the right conceptual interface because it
accepts `family_idx` for leaf log-probabilities.  It needs a second constant
layout mode:

```text
CONST_MODE_SHARED:   base = 0
CONST_MODE_ROW:      base = w*S          # current genewise fallback
CONST_MODE_FAMILY:   base = family*S     # desired genewise path
```

Expected effect:

- remove most PyTorch gather/index work from the genewise forward wave loop;
- reduce temporary memory pressure by avoiding repeated `[W,S]` constants;
- improve L2 reuse because many adjacent rows in a family use the same
  `[S]` parameter vectors;
- make the genewise forward structurally match global/uniform.

This is the first proposal to test because the measured 1000-tree gap is
`4.468 s - 2.318 s = 2.150 s`, and a large part of that gap is in wave-step
time and PyTorch indexing, not DTS.

Acceptance gates:

```bash
pytest -q tests/unit/test_genewise_wave.py \
          tests/kernels/test_wave_step_uniform_forward_kernel.py \
          tests/gradients/test_autograd_bridge.py::test_model_nll_matches_compute_likelihood_batch
```

Performance gates:

- genewise forward chunks with 50, 150, and 1000 total families;
- Nsys bucket for `_wave_step_uniform_kernel` and PyTorch index/elementwise
  kernels;
- NCU representative large wave-step launch, checking DRAM bytes, L2 hit rate,
  issue slots, and occupancy.

### Proposal 0 follow-up: implemented and promoted

Proposal 0 is implemented as the default genewise uniform forward path.

Implementation:

- `_wave_step_uniform_kernel` now has `CONST_LAYOUT` modes:
  - `0`: shared `[S]` constants;
  - `1`: row-expanded `[W,S]` constants, the old genewise fallback;
  - `2`: family-indexed `[G,S]` constants addressed by
    `family_idx[ws + row]`.
- `wave_step_uniform_fused`, `wave_step_uniform_fused_into`,
  `wave_step_uniform_ancestor_fused`, and `wave_step_uniform_csr_fused` accept
  `family_indexed_consts`.
- `wave_pibar_uniform_parent_fused` also accepts the same mode, because the
  fixed-6 ping-pong path recomputes final Pibar rows and needs family-indexed
  `mt`.
- `Pi_wave_forward` uses `GPUREC_FORWARD_FAMILY_INDEXED_CONSTS=1` by default
  for batched uniform fused mode and returns the original `[G,S]` constants
  from `_wave_consts` instead of gathering `[W,S]`.
- `profiling/bench_uniform_forward_parent_dts.py` now exposes
  `--family-indexed-consts`, `--no-family-indexed-consts`, and
  `--compare-family-consts`.

The old path is still available with:

```bash
GPUREC_FORWARD_FAMILY_INDEXED_CONSTS=0
```

or with:

```bash
python profiling/bench_uniform_forward_parent_dts.py ... --no-family-indexed-consts
```

The kernel-level change is:

```text
old:
    const_base = row * S
    load DL_w[row, s]     # DL_w was materialized as [W,S]

new:
    family = family_idx[ws + row]
    const_base = family * S
    load DL_const[family, s]
```

This covers `DL_const`, `SL1_const`, `SL2_const`, `Ebar`, `E`, and `mt`.

#### Correctness

Commands:

```bash
python -m py_compile \
  gpurec/core/forward.py \
  gpurec/core/kernels/wave_step.py \
  profiling/bench_uniform_forward_parent_dts.py

pytest -q tests/unit/test_genewise_wave.py \
          tests/kernels/test_wave_step_uniform_forward_kernel.py \
          tests/gradients/test_autograd_bridge.py::test_model_nll_matches_compute_likelihood_batch
```

Result:

```text
22 passed
```

Focused row-expanded versus family-indexed parity:

```bash
PREPROCESS_CACHE_DIR=/tmp/gpurec_prop0_genewise_cache \
FAMS=3 REPS=1 WARMUPS=0 MAX_WAVE_SIZE=32768 \
python profiling/bench_uniform_forward_parent_dts.py \
  --dataset tests/data/test_trees_1000 \
  --mode genewise --dtype fp32 --variant new \
  --need-pibar --no-root-rows --leaf-index \
  --family-indexed-consts --compare-family-consts \
  --topology-int32 --overlap-mode off
```

Observed:

| Check | Result |
|---|---:|
| row-expanded NLL | `6421.17333984375` |
| family-indexed NLL | `6421.17333984375` |
| NLL absolute difference | `0.0` |
| full `Pi_wave_ordered` max abs difference | `0.0` |
| full `Pibar_wave_ordered` max abs difference | `0.0` |

The unit suite also contains a dedicated
`test_genewise_uniform_family_indexed_constants_match_row_expanded` check for
both genewise scalar `[G]` and genewise specieswise `[G,S]` parameters.

#### Timing

CUDA-event timings below exclude model construction and E fixed-point setup.
Both paths use `test_trees_1000`, fp32, fixed 6 Pi iterations,
`pibar_mode="uniform"`, root-row output, `need_pibar=False`, compact leaf index,
parent-reduced DTS, int32 topology, `max_wave_size=32768`, and DTS overlap off.

| Workload | Row-expanded constants | Family-indexed constants | Delta | Speedup | Peak old | Peak new | NLL |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 families | `224.332 ms` | `132.528 ms` | `-91.804 ms` | `1.69x` | `7.745 GiB` | `5.989 GiB` | `107804.265625` |
| 150 families | `664.575 ms` | `388.419 ms` | `-276.156 ms` | `1.71x` | `17.945 GiB` | `17.565 GiB` | `323018.6875` |

Full 1000-tree genewise forward was measured as seven resident chunks:
`150x6 + 100`.  The old pre-Proposal-0 sum was `4468.234 ms`.  The new chunk
medians were:

```text
389.251, 395.454, 397.684, 393.422, 389.957, 393.037, 259.246 ms
```

Summed new 1000-tree time:

```text
2618.051 ms
```

So Proposal 0 improves the optimized genewise 1000-tree forward by:

```text
4468.234 ms -> 2618.051 ms
delta = -1850.183 ms
speedup = 1.71x
time reduction = 41.4%
```

The remaining gap to the current global/uniform 1000-tree forward is now much
smaller:

```text
global/uniform forward:   2317.719 ms
genewise/uniform forward: 2618.051 ms
overhead:                 300.332 ms, about 13.0%
```

#### Nsight Systems

Paired 50-family captures:

```bash
PREPROCESS_CACHE_DIR=/tmp/gpurec_prop0_genewise_cache \
FAMS=50 REPS=1 WARMUPS=2 MAX_WAVE_SIZE=32768 \
nsys profile --force-overwrite=true \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  -o /tmp/gpurec_profile/prop0_genewise_forward/f50_row_consts \
  python profiling/bench_uniform_forward_parent_dts.py \
    --dataset tests/data/test_trees_1000 \
    --mode genewise --dtype fp32 --variant new \
    --no-need-pibar --root-rows --leaf-index \
    --no-family-indexed-consts --topology-int32 \
    --overlap-mode off --profile-cuda-api

PREPROCESS_CACHE_DIR=/tmp/gpurec_prop0_genewise_cache \
FAMS=50 REPS=1 WARMUPS=2 MAX_WAVE_SIZE=32768 \
nsys profile --force-overwrite=true \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  -o /tmp/gpurec_profile/prop0_genewise_forward/f50_family_consts \
  python profiling/bench_uniform_forward_parent_dts.py \
    --dataset tests/data/test_trees_1000 \
    --mode genewise --dtype fp32 --variant new \
    --no-need-pibar --root-rows --leaf-index \
    --family-indexed-consts --topology-int32 \
    --overlap-mode off --profile-cuda-api
```

| Metric / bucket | Row-expanded | Family-indexed | Delta | Interpretation |
|---|---:|---:|---:|---|
| CUDA-event interval under Nsys | `225.500 ms` | `131.818 ms` | `-93.682 ms` | matches direct timing trend |
| `_wave_step_uniform_kernel` | `154.101 ms / 294` | `77.297 ms / 294` | `-76.804 ms` | primary win |
| PyTorch per-wave constant index gather | `16.860 ms / 387` | `0.279 ms / 93` | `-16.581 ms` | `[W,S]` constant materialization removed |
| `_dts_fused_kernel` | `14.740 ms / 39` | `14.741 ms / 39` | `0.001 ms` | DTS unchanged |
| `_dts_parent_reduced_ge2_stage1_kernel` | `11.425 ms / 7` | `11.360 ms / 7` | `-0.065 ms` | DTS unchanged |
| `_wave_pibar_uniform_parent_kernel` | `10.010 ms / 48` | `9.819 ms / 48` | `-0.190 ms` | final Pibar roughly unchanged |
| PyTorch direct-copy elementwise | `6.005 ms / 93` | `6.158 ms / 93` | `0.153 ms` | unrelated residual work |
| PyTorch fill kernels | `3.962 ms / 50` | `3.968 ms / 50` | `0.006 ms` | unchanged |
| Kernel launch count | `1173` | `879` | `-294` | removed per-wave gather launches |
| GPU memcpy time | about `0.002 ms` | negligible | none | copies are not the issue |

Artifacts:

```text
/tmp/gpurec_profile/prop0_genewise_forward/f50_row_consts.nsys-rep
/tmp/gpurec_profile/prop0_genewise_forward/f50_family_consts.nsys-rep
```

#### Nsight Compute

Representative first `_wave_step_uniform_kernel` launch, 50 families:

```text
shape: W=32768 rows, S=1999 species
grid=(32768,1,1), block=(128,1,1)
```

Commands:

```bash
PREPROCESS_CACHE_DIR=/tmp/gpurec_prop0_genewise_cache \
FAMS=50 REPS=1 WARMUPS=1 MAX_WAVE_SIZE=32768 \
ncu --target-processes all --force-overwrite --set detailed \
  --kernel-name regex:_wave_step_uniform_kernel \
  --launch-skip 0 --launch-count 1 --profile-from-start off \
  -o /tmp/gpurec_profile/prop0_genewise_forward/ncu/f50_row_consts_wave0 \
  python profiling/bench_uniform_forward_parent_dts.py \
    --dataset tests/data/test_trees_1000 \
    --mode genewise --dtype fp32 --variant new \
    --no-need-pibar --root-rows --leaf-index \
    --no-family-indexed-consts --topology-int32 \
    --overlap-mode off --profile-cuda-api

PREPROCESS_CACHE_DIR=/tmp/gpurec_prop0_genewise_cache \
FAMS=50 REPS=1 WARMUPS=1 MAX_WAVE_SIZE=32768 \
ncu --target-processes all --force-overwrite --set detailed \
  --kernel-name regex:_wave_step_uniform_kernel \
  --launch-skip 0 --launch-count 1 --profile-from-start off \
  -o /tmp/gpurec_profile/prop0_genewise_forward/ncu/f50_family_consts_wave0 \
  python profiling/bench_uniform_forward_parent_dts.py \
    --dataset tests/data/test_trees_1000 \
    --mode genewise --dtype fp32 --variant new \
    --no-need-pibar --root-rows --leaf-index \
    --family-indexed-consts --topology-int32 \
    --overlap-mode off --profile-cuda-api
```

NCU replay timing is slower than normal execution, so use these numbers for
resource interpretation rather than wall time.

| Metric | Row-expanded | Family-indexed | Interpretation |
|---|---:|---:|---|
| Duration | `2.375 ms` | `1.323 ms` | `1.80x` faster for the representative launch |
| DRAM read bytes | `1.880 GB` | `263.582 MB` | `7.13x` less DRAM read traffic |
| DRAM write bytes | `255.750 MB` | `230.902 MB` | output traffic mostly unchanged |
| DRAM throughput | `91.45%` | `38.03%` | old path was DRAM-bound |
| L1/TEX hit rate | `83.65%` | `83.90%` | L1 behavior similar |
| L2 hit rate | `43.08%` | `92.02%` | family constants become cache-resident |
| SM throughput | `45.58%` | `83.69%` | kernel becomes much better utilized |
| Issue active | `41.60%` | `76.24%` | fewer memory stalls expose compute work |
| Achieved occupancy proxy | `99.28%` | `98.08%` | occupancy stayed high |
| Registers/thread | `40` | `40` | no register regression |
| Local spills | `0` | `0` | no spill regression |
| Executed instructions | `1.127B` | `1.148B` | instruction count is similar |
| Long scoreboard samples | `211573` | `23148` | memory-dependency stalls down about `9.1x` |

Artifacts:

```text
/tmp/gpurec_profile/prop0_genewise_forward/ncu/f50_row_consts_wave0.ncu-rep
/tmp/gpurec_profile/prop0_genewise_forward/ncu/f50_family_consts_wave0.ncu-rep
```

The key result is that Proposal 0 does not make the kernel faster by reducing
arithmetic.  It makes the same arithmetic much better fed: the old path reads
six per-row `[W,S]` constant matrices from global memory, while the new path
reuses the small per-family `[G,S]` constant vectors through L2.  This is why
instruction count is similar but DRAM reads, L2 hit rate, issue activity, and
long-scoreboard stalls change dramatically.

#### Decision

Promote Proposal 0 as the default genewise uniform forward path.

Rationale:

- Correctness is exact against the row-expanded path in the direct full-output
  comparison and covered by unit tests for `[G]` and `[G,S]`.
- 50-family and 150-family chunks both improve by about `1.7x`.
- The full 1000-tree optimized genewise forward improves from `4.468 s` to
  `2.618 s`.
- Nsys shows that DTS and final Pibar buckets are unchanged; the win is exactly
  the intended wave-step and PyTorch materialization removal.
- NCU shows the representative launch moves from a DRAM-bound profile to a
  cache-reuse profile with high issue activity and no register/spill cost.

Next step: Proposal 1, family-indexed forward DTS parameters.  Proposal 0
removed per-wave constant expansion for DTS_L, but `_wave_dts_params` still
expands scalar genewise `log_pD/log_pS` to `[n_splits,S]` for DTS cross-clade
waves.

## Proposal 1: Family-Indexed DTS Parameters In Forward Kernels

The forward DTS path has the same materialization problem for per-split
parameters:

```text
fi_splits = family_idx[ws + reduce_idx]
pD = log_pD[fi_splits]
pS = log_pS[fi_splits]

if pD is [N]:
    pD = pD[:, None].expand(N, S).contiguous()
```

That creates `[n_splits,S]` tensors for scalar genewise parameters.  The DTS
kernel should instead load per-family parameters directly:

```text
parent = reduce_idx[split]
family = family_idx[ws + parent]

if PARAM_MODE_G:
    pD = log_pD[family]
    pS = log_pS[family]

if PARAM_MODE_GS:
    pD = log_pD[family, s]
    pS = log_pS[family, s]
```

This should be added to:

- `_dts_fused_kernel`;
- parent-reduced DTS stage 1;
- parent-reduced DTS stage 2 if it consumes parameterized partials;
- any forward recompute variant used by backward.

The scalar genewise case `[G]` is especially attractive because the same
`pD/pS` value is reused across all species lanes for a split.  The kernel can
broadcast the loaded scalar rather than reading a full `[N,S]` expanded matrix.

Expected effect:

- lower PyTorch index/materialization time;
- lower peak memory in high-fanout root waves;
- improve 150-family genewise chunks more than 10-family chunks because split
  fanout becomes large near root waves.

This should be tested after Proposal 0, because otherwise the wave-step
constant materialization may hide the DTS win.

### Proposal 1 follow-up: implemented and promoted

Proposal 1 is implemented and enabled by default for genewise uniform forward.
The old path materialized per-split DTS parameters before launching Triton:

```text
fi_splits = family_idx[ws + reduce_idx]
pD = log_pD[fi_splits]                  # [N] or [N,S]
pS = log_pS[fi_splits]

if scalar genewise:
    pD = pD[:, None].expand(N, S).contiguous()
    pS = pS[:, None].expand(N, S).contiguous()
```

The promoted path keeps the original family parameter tensors resident:

```text
_wave_dts_params(meta):
    return log_pD, log_pS                # [G] or [G,S]

Triton split program:
    parent_w = reduce_idx[split]
    family = family_idx[wave_start + parent_w]

    mode 0: p = param[s]                 # shared [S]
    mode 1: p = param[split, s]          # per-split [N,S]
    mode 2: p = param[split]             # per-split scalar [N]
    mode 3: p = param[family]            # family scalar [G]
    mode 4: p = param[family, s]         # family specieswise [G,S]
```

Implementation details:

- `dts_fused._prepare_param(..., family_indexed=True)` now returns modes `3`
  and `4` for `[G]`/`[G,S]` parameters.
- `_dts_fused_kernel`, `_dts_eq1_to_rows_kernel`,
  `_dts_parent_reduced_ge2_kernel`, and
  `_dts_parent_reduced_ge2_stage1_kernel` receive `family_idx` plus
  `family_offset=wave_start`.
- `_dts_parent_reduced_ge2_stage2_kernel` stays parameter-layout agnostic; it
  only reduces partial max/sum buffers produced by stage 1.
- Scalar parameter modes return a broadcast vector inside Triton.  This matters
  because Triton rejects helper functions whose branches return a scalar in one
  mode and a `[BLOCK_S]` vector in another.
- `Pi_wave_forward` enables this path when
  `GPUREC_FORWARD_FAMILY_INDEXED_DTS_PARAMS` is true, the mode is batched
  uniform CUDA forward, and `log_pD/log_pS` have `[G]` or `[G,S]` layout.
- The row-expanded fallback remains available:

```bash
GPUREC_FORWARD_FAMILY_INDEXED_DTS_PARAMS=0
```

- `profiling/bench_uniform_forward_parent_dts.py` now exposes:

```bash
--family-indexed-dts-params
--no-family-indexed-dts-params
--compare-family-dts-params
```

Both Proposal 1 timing paths below keep Proposal 0 enabled
(`--family-indexed-consts`), so these measurements isolate DTS parameter
materialization.

#### Correctness

Required compile gate:

```bash
python -m py_compile \
  gpurec/core/forward.py \
  gpurec/core/kernels/dts_fused.py \
  profiling/bench_uniform_forward_parent_dts.py
```

Required unit and integration gates:

```bash
pytest -q tests/kernels/test_dts_fused_kernel.py \
          tests/unit/test_genewise_wave.py \
          tests/gradients/test_autograd_bridge.py::test_model_nll_matches_compute_likelihood_batch
```

Additional focused tests to add or verify before promotion:

- scalar genewise `[G]` `log_pD/log_pS` family-indexed DTS matches the
  row-expanded fallback;
- specieswise genewise `[G,S]` `log_pD/log_pS` family-indexed DTS matches the
  row-expanded fallback;
- parent-reduced DTS direct and tiled implementations match for both parameter
  layouts;
- active-mask handling still writes `-inf` for inactive parent rows;
- int32 split metadata still works for the family-indexed path.

Result:

```text
python -m py_compile ...: passed
pytest -q tests/kernels/test_dts_fused_kernel.py \
          tests/unit/test_genewise_wave.py \
          tests/gradients/test_autograd_bridge.py::test_model_nll_matches_compute_likelihood_batch

45 passed in 7.76s
```

The dedicated Proposal 1 unit test covers both scalar genewise `[G]` and
specieswise genewise `[G,S]` `log_pD/log_pS` layouts.  The full
`tests/unit/test_genewise_wave.py` file also passed (`13 passed`) after adding
the new parity checks.

Focused row-expanded versus family-indexed DTS parameter parity:

```bash
PREPROCESS_CACHE_DIR=/tmp/gpurec_prop1_genewise_cache \
FAMS=3 REPS=1 WARMUPS=0 MAX_WAVE_SIZE=32768 \
python profiling/bench_uniform_forward_parent_dts.py \
  --dataset tests/data/test_trees_1000 \
  --mode genewise --dtype fp32 --variant new \
  --need-pibar --no-root-rows --leaf-index \
  --family-indexed-consts \
  --family-indexed-dts-params --compare-family-dts-params \
  --topology-int32 --overlap-mode off
```

Observed:

| Check | Result |
|---|---:|
| row-expanded DTS-param NLL | `6421.17333984375` |
| family-indexed DTS-param NLL | `6421.17333984375` |
| NLL absolute difference | `0.0` |
| full `Pi_wave_ordered` max abs difference | `0.0` |
| full `Pibar_wave_ordered` max abs difference | `0.0` |

The specieswise smoke compare also passed exactly:

```text
mode=genewise_specieswise, FAMS=3
NLL diff: 0.0
Pi max abs: 0.0
Pibar max abs: 0.0
```

#### Timing

CUDA-event timings should exclude model construction and E fixed-point setup,
matching the Proposal 0 protocol.  Both paths should use `test_trees_1000`,
fp32, fixed 6 Pi iterations, `pibar_mode="uniform"`, root-row output,
`need_pibar=False`, compact leaf index, parent-reduced DTS, int32 topology,
`max_wave_size=32768`, DTS overlap off, and family-indexed constants enabled.

| Workload | Row-expanded DTS params | Family-indexed DTS params | Delta | Speedup | Peak old | Peak new | NLL |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 families | `131.285 ms` | `116.467 ms` | `-14.818 ms` | `1.13x` | `5.989 GiB` | `5.259 GiB` | `107804.265625` |
| 150 families | `387.446 ms` | `344.763 ms` | `-42.683 ms` | `1.12x` | `17.565 GiB` | `15.015 GiB` | `323018.6875` |

Full 1000-tree genewise forward should be reported as resident chunks using the
same chunking policy as Proposal 0:

```text
old row-expanded DTS-param chunk medians:
start=0:   387.096 ms
start=150: 393.559 ms
start=300: 398.907 ms
start=450: 391.349 ms
start=600: 388.104 ms
start=750: 391.566 ms
start=900: 259.697 ms

new family-indexed DTS-param chunk medians:
start=0:   348.095 ms
start=150: 350.568 ms
start=300: 355.890 ms
start=450: 348.351 ms
start=600: 345.558 ms
start=750: 348.924 ms
start=900: 230.582 ms

summed old 1000-tree time:
2610.278 ms

summed new 1000-tree time:
2327.967 ms

delta:
-282.311 ms

speedup:
1.12x
```

Relative to the pre-Proposal-0 genewise forward baseline (`4.468 s`), the
current Proposal-0-plus-Proposal-1 path is `1.92x` faster.  Relative to the
Proposal 0 result (`2.618 s`), Proposal 1 adds another `1.12x`.

#### Nsight Systems

Paired 50-family captures should compare only the DTS parameter layout while
keeping Proposal 0 enabled:

```bash
PREPROCESS_CACHE_DIR=/tmp/gpurec_prop1_genewise_cache \
FAMS=50 REPS=1 WARMUPS=2 MAX_WAVE_SIZE=32768 \
nsys profile --force-overwrite=true \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  -o /tmp/gpurec_profile/prop1_genewise_forward/f50_row_dts_params \
  python profiling/bench_uniform_forward_parent_dts.py \
    --dataset tests/data/test_trees_1000 \
    --mode genewise --dtype fp32 --variant new \
    --no-need-pibar --root-rows --leaf-index \
    --family-indexed-consts --no-family-indexed-dts-params \
    --topology-int32 --overlap-mode off --profile-cuda-api

PREPROCESS_CACHE_DIR=/tmp/gpurec_prop1_genewise_cache \
FAMS=50 REPS=1 WARMUPS=2 MAX_WAVE_SIZE=32768 \
nsys profile --force-overwrite=true \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  -o /tmp/gpurec_profile/prop1_genewise_forward/f50_family_dts_params \
  python profiling/bench_uniform_forward_parent_dts.py \
    --dataset tests/data/test_trees_1000 \
    --mode genewise --dtype fp32 --variant new \
    --no-need-pibar --root-rows --leaf-index \
    --family-indexed-consts --family-indexed-dts-params \
    --topology-int32 --overlap-mode off --profile-cuda-api
```

| Metric / bucket | Row-expanded DTS params | Family-indexed DTS params | Delta | Interpretation |
|---|---:|---:|---:|---|
| CUDA-event interval under Nsys | `132.698 ms` | `117.608 ms` | `-15.091 ms` | end-to-end trend under profiler |
| Visible PyTorch DTS expand/index/copy kernels | `~6.94 ms / 363 launches` | `~0.11 ms / 39 launches` | `~-6.83 ms` | per-wave scalar expansion disappears |
| `_dts_fused_kernel` | `14.756 ms / 39` | `9.728 ms / 39` | `-5.028 ms` | scalar `[G]` loads remove `[N,S]` parameter reads |
| `_dts_parent_reduced_ge2_stage1_kernel` | `11.394 ms / 7` | `7.530 ms / 7` | `-3.864 ms` | high-fanout parent-reduced path improves |
| `_dts_parent_reduced_ge2_stage2_kernel` | `0.130 ms / 7` | `0.131 ms / 7` | `+0.001 ms` | unchanged, as expected |
| `_dts_eq1_to_rows_kernel` | `0.160 ms / 6` | `0.057 ms / 6` | `-0.103 ms` | small bucket but same direction |
| `_wave_step_uniform_kernel` | `77.846 ms / 294` | `78.517 ms / 294` | `+0.671 ms` | effectively unchanged; Proposal 1 targets DTS |
| Kernel launch count | `879` | `558` | `-321` | PyTorch materialization launches disappear |
| GPU memcpy API time | `0.027 ms` | `0.029 ms` | `+0.002 ms` | not material |

Artifacts:

```text
/tmp/gpurec_profile/prop1_genewise_forward/f50_row_dts_params.nsys-rep
/tmp/gpurec_profile/prop1_genewise_forward/f50_family_dts_params.nsys-rep
```

The unchanged `~3.95 ms / 40` PyTorch index bucket in both captures is not the
DTS parameter materialization being removed here; it remained stable across the
A/B run and is therefore outside Proposal 1.

#### Nsight Compute

Profile a representative DTS launch with large split fanout.  For the default
parent-reduced tiled path, the most relevant kernel is expected to be
`_dts_parent_reduced_ge2_stage1_kernel`; `_dts_parent_reduced_ge2_stage2_kernel`
should not change because it reduces already computed partials.

```bash
PREPROCESS_CACHE_DIR=/tmp/gpurec_prop1_genewise_cache \
FAMS=50 REPS=1 WARMUPS=1 MAX_WAVE_SIZE=32768 \
ncu --target-processes all --force-overwrite --set detailed \
  --kernel-name regex:_dts_parent_reduced_ge2_stage1_kernel \
  --launch-skip 0 \
  --launch-count 1 --profile-from-start off \
  -o /tmp/gpurec_profile/prop1_genewise_forward/ncu/f50_row_dts_stage1_0 \
  python profiling/bench_uniform_forward_parent_dts.py \
    --dataset tests/data/test_trees_1000 \
    --mode genewise --dtype fp32 --variant new \
    --no-need-pibar --root-rows --leaf-index \
    --family-indexed-consts --no-family-indexed-dts-params \
    --topology-int32 --overlap-mode off --profile-cuda-api

PREPROCESS_CACHE_DIR=/tmp/gpurec_prop1_genewise_cache \
FAMS=50 REPS=1 WARMUPS=1 MAX_WAVE_SIZE=32768 \
ncu --target-processes all --force-overwrite --set detailed \
  --kernel-name regex:_dts_parent_reduced_ge2_stage1_kernel \
  --launch-skip 0 \
  --launch-count 1 --profile-from-start off \
  -o /tmp/gpurec_profile/prop1_genewise_forward/ncu/f50_family_dts_stage1_0 \
  python profiling/bench_uniform_forward_parent_dts.py \
    --dataset tests/data/test_trees_1000 \
    --mode genewise --dtype fp32 --variant new \
    --no-need-pibar --root-rows --leaf-index \
    --family-indexed-consts --family-indexed-dts-params \
    --topology-int32 --overlap-mode off --profile-cuda-api
```

Representative launch:

```text
kernel: _dts_parent_reduced_ge2_stage1_kernel
launch skip: 0
grid: (6, 53, 16)
block: (128, 1, 1)
```

NCU replay timing is slower than normal execution, so use these numbers for
resource interpretation rather than wall time.

| Metric | Row-expanded DTS params | Family-indexed DTS params | Interpretation |
|---|---:|---:|---|
| Duration | `1.209 ms` | `0.814 ms` | `-32.6%` representative launch time |
| DRAM read bytes | `911.510 MB` | `607.770 MB` | `-33.3%`, main hardware reason |
| DRAM write bytes | `6.463 MB` | `6.578 MB` | effectively unchanged |
| L1/TEX hit rate | `47.57%` | `56.18%` | better local reuse |
| L2 hit rate | `13.83%` | `15.71%` | slightly better; still low |
| DRAM throughput | `77.26%` | `76.75%` | still DRAM-bound |
| SM throughput / issue active | `9.85%` | `12.07%` | more issue progress after reducing memory pressure |
| Achieved occupancy proxy | `94.49%` | `94.08%` | unchanged |
| Registers/thread | `40` | `40` | no register regression |
| Local spills | `0` | `0` | no spilling |
| Long scoreboard samples | `135701` | `96353` | fewer global-memory wait samples |
| Executed instructions | `136.1M` | `112.4M` | scalar broadcasts remove repeated parameter-load work |

Artifacts:

```text
/tmp/gpurec_profile/prop1_genewise_forward/ncu/f50_row_dts_stage1_0.ncu-rep
/tmp/gpurec_profile/prop1_genewise_forward/ncu/f50_family_dts_stage1_0.ncu-rep
```

#### Interpretation

Proposal 1 is smaller than Proposal 0 but still worthwhile.  DTS kernels still
read large `Pi` and `Pibar` child rows, so eliminating `[N_splits,S]`
`log_pD/log_pS` materialization cannot make the DTS bucket disappear.  The win
appears in the expected three places:

- PyTorch gather, expand, and contiguous-copy buckets for DTS parameters shrink
  from about `6.94 ms` to about `0.11 ms` in the 50-family Nsys capture.
- Peak memory decreases from `17.565 GiB` to `15.015 GiB` in the 150-family
  chunk.
- NCU shows `-33.3%` DRAM reads for the representative stage-1 launch, with the
  same register count, no spills, and the same high occupancy.

The family-indexed path is also faster inside `_dts_fused_kernel` and
`_dts_parent_reduced_ge2_stage1_kernel`, not only in Python-side launch count.
That is because scalar `[G]` parameters stop being read from expanded
`[N_splits,S]` tensors.  In the representative stage-1 launch, the kernel still
saturates DRAM (`~77%`) while SM issue remains low (`12%`), so future DTS work
should focus on child-row memory reuse and parent/split locality, not on
micro-optimizing scalar family loads.

For specieswise `[G,S]` parameters, the kernel still loads one value per split
and species lane.  The full-output specieswise smoke test proves correctness,
but the scalar `[G]` case is where the largest bandwidth reduction is expected.
If future specieswise profiles show worse L2 behavior because split rows jump
across many families, revisit the family-local scheduling proposal before
promoting larger specieswise training chunks.

#### Decision

Decision: accept and promote Proposal 1 as the default genewise uniform forward
path.

Rationale:

- Parity is exact for scalar `[G]` and specieswise `[G,S]` checks.
- The full 1000-tree forward time improves from `2610.278 ms` to
  `2327.967 ms`.
- Peak memory decreases materially in the 150-family resident chunk.
- Nsys confirms the PyTorch materialization launch count and time drop.
- NCU confirms lower DRAM reads and no register, spill, or occupancy regression.

Keep the row-expanded path available for at least one more profiling cycle:

```bash
GPUREC_FORWARD_FAMILY_INDEXED_DTS_PARAMS=0
```

#### Follow-up

- Carry the same family-indexed `log_pD/log_pS` loading convention into
  Proposal 3 for genewise DTS backward accumulation.
- Run a second NCU pass on a larger 150-family high-fanout wave if future work
  targets the remaining DTS memory bandwidth.
- Inspect family locality if L2 hit rate regresses in larger specieswise runs,
  because that would make the family-local scheduling proposal a prerequisite
  rather than a later guardrail.

## Proposal 2: Port The Fused Uniform Backward Self-Loop To Genewise

The fastest global backward self-loop path should become family-aware rather
than `G == 1` only.

The current optimized global path computes:

```text
rhs -> Neumann VJP through fixed-point self-loop
    -> alpha weights for DTS_L terms
    -> direct accumulation of global parameter gradients
```

For genewise, the same row-local math applies.  Only the parameter loads and
gradient accumulation targets change:

```text
family = family_idx[ws + row]

load:
    DL[family,s], E[family,s], Ebar[family,s],
    SL1[family,s], SL2[family,s], mt[family,s]

accumulate:
    grad_log_pD[family]      or grad_log_pD[family,s]
    grad_log_pS[family]      or grad_log_pS[family,s]
    grad_E[family,s]
    grad_Ebar[family,s]
    grad_E_s1[family,s]
    grad_E_s2[family,s]
    grad_mt[family,s]
```

There are two implementation strategies:

| Strategy | Pros | Cons |
|---|---|---|
| direct atomic accumulation to `[G]` and `[G,S]` | closest to current fused global path; avoids writing full contribution tensors | atomics contend when many rows of one family are active |
| two-stage per-wave family reductions | less contention and more deterministic | needs scratch shaped by active families per wave |

The direct atomic version should be tried first for genewise scalar parameters
because it is the smallest semantic step from the current global fused kernel.
The wave layout already tends to keep family rows contiguous inside chunks, so
atomic locality may be acceptable.  If NCU shows RED pressure dominating, move
to two-stage reductions.

Acceptance gates:

- compare genewise batch gradients against running individual families through
  global/uniform mode;
- run fp64 high-iteration agreement checks to catch ancestor-correction or
  fixed-point mistakes;
- run gradcheck on small trees.

Expected effect:

- make genewise backward fit for larger chunks;
- remove large generic PyTorch materializations from the self-loop;
- close the most obvious gap between genewise and global backward.

## Proposal 3: Port Fused DTS Backward Accumulation To Genewise

After the self-loop, the next generic genewise cost is cross-clade DTS
backward.  In the global path, the fused DTS backward kernels:

- recompute DTS terms in Triton;
- accumulate direct Pi adjoints into `accumulated_rhs`;
- produce compact Pibar VJP inputs;
- optionally accumulate scalar parameter reductions and `grad_mt`.

The current scalar fast path is tied to shared parameters.  For genewise it
should load `pD/pS` using the split parent family exactly as Proposal 1 does:

```text
split_parent = ws + reduce_idx[split]
family = family_idx[split_parent]

pD = log_pD[family]       or log_pD[family,s]
pS = log_pS[family]       or log_pS[family,s]
```

Gradient accumulation should write to per-family outputs:

```text
grad_log_pD[family] += sum_s grad_D_term[split,s]
grad_log_pS[family] += sum_s grad_S_terms[split,s]
```

For specieswise genewise parameters, use `[G,S]` destinations.  For scalar
genewise parameters, use `[G]` scalar destinations and keep the species
reduction inside the kernel.

Expected effect:

- remove `DTS_5 = torch.stack([...], dim=0)`;
- remove large PyTorch `scatter_add_` calls for split parameters;
- keep the current direct accumulation of child Pi adjoints;
- enable the staged Pibar VJP path instead of falling back to generic PyTorch
  uniform Pibar VJP.

This proposal should be profiled separately for two wave shapes:

- large `W`, fanout near 1;
- small root-like `W`, very large split fanout.

Those shapes stressed different DTS kernels in the uniform backward docs.

## Proposal 4: Make Uniform Pibar VJP Family-Aware

Uniform Pibar itself is mostly independent of the gene family because the
species topology is shared.  The family dependence enters through `mt` and, for
some fused variants, through cached row statistics from the forward pass.

The genewise VJP should reuse the global tree kernels:

```text
grad_Pibar child side
    -> stable row coefficient
    -> tree ancestor correction VJP
    -> accumulate into accumulated_rhs[child,:]
```

Required changes:

- pass `family_idx` for child rows;
- load `mt[family,s]` rather than `mt_shared[s]`;
- accept forward `pibar_row_max[C]` from the genewise forward;
- keep `pibar_ud` compact scratch and exact-zero side skipping;
- avoid the generic fallback that computes `p_prime`, `anc_sum`, `denom`, and
  scatter updates through PyTorch.

The key lesson from uniform backward is that Pibar VJP is memory-bound.  The
genewise implementation should not add an extra `[rows,S]` staging tensor for
family constants.  It should load the family row directly in the tree VJP
kernel.

Expected effect:

- turn the genewise Pibar VJP from generic PyTorch work into the same staged
  custom-kernel path used by global mode;
- reduce memory enough for larger genewise backward chunks;
- preserve exact ancestor correction.

## Proposal 5: Genewise Backward Saved-Tensor Strategy

Likelihood-only genewise forward can use root-row output, but training still
needs saved tensors for backward.  A 1000-family genewise backward cannot keep
unbounded full batched state resident.

The target should be explicit chunked autograd:

```text
for family_chunk in schedule:
    forward chunk with full saved Pi/Pibar needed by backward
    compute chunk loss
    backward chunk loss
    accumulate theta gradients
    release chunk saved tensors
```

This differs from likelihood-only inference:

```text
for family_chunk in schedule:
    forward chunk with root rows only
    accumulate log likelihood
    release full Pi/Pibar immediately
```

The backward chunk size will probably be smaller than the forward chunk size.
Current global backward uses 100-family chunks in the exact ancestor-corrected
path, while genewise forward fits 150-family chunks.  A good first policy is:

| Pass | Initial chunk size | Reason |
|---|---:|---|
| genewise forward inference | 150 | already measured to fit |
| genewise backward training | 50 or 100 | leave room for adjoints and Pibar VJP scratch |

The scheduler should report:

- chunk family count;
- clade rows;
- waves;
- max wave rows;
- split rows;
- peak allocation;
- forward and backward CUDA-event time.

This proposal is mostly engineering support, but it is required to benchmark
the real optimized genewise backward over all 1000 trees without accidentally
falling back to old generic scripts.

## Proposal 6: Preserve Family Locality In Cross-Family Waves

Once constants are loaded by `family_idx` inside kernels, row order matters.
Rows from the same family should stay near each other inside a wave so the
small `[S]` family constant vectors remain cache-hot.

The wave scheduler should therefore prefer:

```text
for each wave depth:
    concatenate family wave rows by family
    split by max_wave_size without interleaving families unless needed
```

Avoid:

```text
interleave rows from many families at fine granularity
```

because that turns `DL[family,s]`, `E[family,s]`, and `mt[family,s]` loads into
many unrelated streams.

This proposal should be measured with NCU after Proposal 0:

- L2 hit rate for constant loads;
- DRAM bytes per `_wave_step_uniform_kernel`;
- sector utilization;
- achieved occupancy and issue slots.

If family locality is already preserved by the existing collate order, this
proposal becomes a documentation/guardrail item.  If not, it may be a real
forward and backward win.

## Proposal 7: Genewise-Specific Profiling Harnesses

The previous confusion between optimized and old paths shows that genewise
needs dedicated profiling entrypoints.

Add or standardize:

```text
profiling/bench_genewise_forward_chunking.py
profiling/bench_genewise_backward_chunking.py
```

These harnesses should print the active kernel flags and reject unsupported
paths loudly.  For example:

```text
mode=genewise
pibar_mode=uniform
fixed_iters_Pi=6
leaf_index=1
parent_reduced_dts=1
family_indexed_constants=1
family_indexed_dts_params=1
fused_genewise_backward=1
```

They should also print whether the run used:

- root-row inference output;
- full saved tensors for backward;
- fused uniform backward;
- generic PyTorch fallback.

If a requested benchmark falls back to generic code, the harness should say so
and mark the result as non-optimized.  This avoids comparing the new kernels to
the wrong path.

## Proposal 8: Correctness Matrix

Genewise correctness should be checked against individual global/uniform runs.
For a family chunk:

```text
batched genewise likelihood[g]
    == global/uniform likelihood for family g with theta[g]

batched genewise grad theta[g]
    == global/uniform grad theta for family g
```

Recommended checks:

| Check | Purpose |
|---|---|
| fp32 likelihood parity, 3 to 10 families | catches indexing and family-order bugs |
| fp64 likelihood parity with many fixed-point/Neumann terms | catches numerical and ancestor-correction bugs |
| gradient parity versus per-family global mode | verifies family-indexed backward kernels |
| `gradcheck` on tiny trees | verifies differentiability independent of the optimized large path |
| finite-difference bridge tests | keeps public autograd semantics honest |

Tolerance policy:

- Do not relax tests because an optimized path is new.
- For fp64 high-iteration tests, require near-perfect agreement when algorithms
  should be mathematically identical.
- For fp32 fused atomic reductions, allow small order-dependent gradient
  differences only after fp64 and finite-difference checks pass.

## Ranked Plan

| Rank | Proposal | Why first | Main metric |
|---:|---|---|---|
| 0 | Family-indexed constants inside forward wave-step | Accepted; removed `[W,S]` uniform constant materialization | 1000-tree forward: `4.468 s -> 2.618 s` |
| 1 | Family-indexed forward DTS parameters | Accepted; removes `[n_splits,S]` DTS parameter expansion in high-fanout waves | 1000-tree forward: `2.610 s -> 2.328 s` |
| 2 | Genewise profiling harnesses | Prevents benchmarking old generic paths | reproducible optimized forward/backward numbers |
| 3 | Fused genewise backward self-loop | Required to make genewise backward use the uniform fast path | 10/50-family backward time, peak memory, NCU self-loop |
| 4 | Fused genewise DTS backward accumulation | Removes generic `DTS_5` and split scatter work | DTS backward bucket and parameter scatter time |
| 5 | Family-aware uniform Pibar VJP | Reuses exact ancestor-corrected global Pibar VJP kernels | Pibar VJP bucket and memory |
| 6 | Saved-tensor and chunked autograd strategy | Required for 1000-family genewise training | full 1000-tree backward without OOM |
| 7 | Preserve family locality in wave scheduling | Helps after family-indexed kernel loads exist | L2 hit rate and DRAM bytes |
| 8 | Correctness matrix | Required for every production promotion | parity and gradcheck |

## Expected Performance Envelope

Forward:

- Original optimized genewise forward before Proposal 0 was `4.468 s` for
  1000 trees.
- After Proposal 0, genewise forward was `2.618 s`.
- After Proposal 1, genewise forward is `2.328 s`.
- Current global/uniform forward is `2.318 s` for the same 1000 trees.
- Genewise cannot be expected to be exactly free relative to global because it
  really does load family-specific constants and DTS parameters.  The two
  largest avoidable forward materializations are now removed.
- The current genewise forward is within about `0.4%` of the measured
  global/uniform forward for this likelihood-only 1000-tree benchmark.
  Better than that may require deeper wave-step algorithm changes, not just
  removing materialization.

Backward:

- The first goal is not a small percentage win.  The first goal is to make an
  optimized genewise backward path exist and fit.
- Once the fused self-loop, DTS backward, and Pibar VJP are family-aware,
  genewise backward should be compared to global/uniform backward per resident
  chunk, not to the current generic 10-family smoke number.
- The expected lower bound is the global/uniform fused path plus extra
  family-indexed parameter loads and per-family gradient accumulation.  The
  expected upper bound is much better than generic PyTorch autograd because the
  generic path materializes large `[rows,S]` and `[splits,S]` tensors that the
  uniform optimization work already showed how to avoid.

## Immediate Next Experiment

Start with Proposal 0.

Implementation sketch:

1. Add `CONST_MODE_FAMILY` to `_wave_step_uniform_kernel`.
2. Pass original `[G,S]` constants and `family_idx` from `Pi_wave_forward`.
3. Keep the old `CONST_MODE_ROW` as a fallback flag for comparison.
4. Verify genewise likelihood parity against the current implementation.
5. Profile 50-family and 150-family chunks with Nsys and NCU.

The decision criterion should be strict:

```text
accept if:
    NLL parity holds
    peak memory decreases
    PyTorch index/materialization bucket decreases
    150-family genewise forward time decreases

reject or revise if:
    wave-step NCU shows much worse cache behavior
    extra family_idx loads erase the removed materialization cost
```

Proposal 0 and Proposal 1 both succeeded.  The next forward-specific work should
be driven by profiler evidence, because the genewise forward is now close to the
current global/uniform timing envelope.  The larger remaining implementation
gap is genewise backward, starting with the self-loop and DTS backward
accumulation proposals.
