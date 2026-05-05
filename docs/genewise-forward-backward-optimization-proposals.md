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

### Proposal 2 follow-up: implemented family-indexed fused self-loop

Proposal 2 was implemented and accepted as the default genewise uniform
backward self-loop path when the shapes are supported.

The old gate was effectively:

```python
can_use_fused_uniform_backward = fused and G == 1 and pibar_mode == "uniform"
```

The new gate also accepts genewise batches when the self-loop constants are
available as family-indexed species rows:

```python
family_indexed_self_loop_supported =
    E, Ebar, E_s1, E_s2, mt are [G,S]
    and log_pD/log_pS are [G] or [G,S]
    and pibar_mode == "uniform"

can_use_fused_uniform_backward =
    fused and (_auto_wrapped or family_indexed_self_loop_supported)
```

The fused Triton kernel now has three constant-addressing layouts:

| Layout | Tensor shape | Addressing |
|---|---|---|
| shared | `[S]` | `const[s]` |
| row-expanded | `[W,S]` | `const[row,s]` |
| family-indexed | `[G,S]` | `const[family_idx[ws + row], s]` |

For the genewise self-loop we use the family-indexed layout, so the backward
wave does not first materialize `DL`, `E`, `Ebar`, `SL1`, `SL2`, or `mt` as
`[W,S]`.  Parameter-gradient accumulation is also family-aware:

```text
scalar genewise:
    atomic_add grad_log_pD[family] += sum_s awD[row,s]
    atomic_add grad_log_pS[family] += sum_s awS[row,s]

specieswise genewise:
    atomic_add grad_log_pD[family,s] += awD[row,s]
    atomic_add grad_log_pS[family,s] += awS[row,s]

all cases:
    atomic_add grad_E[family,s]     += ...
    atomic_add grad_Ebar[family,s]  += ...
    atomic_add grad_E_s1[family,s]  += ...
    atomic_add grad_E_s2[family,s]  += ...
    atomic_add grad_mt[family,s]    += ...
```

The CUDA no-split prototype remains global-only.  Genewise DTS backward and
cross-Pibar VJP are still generic after this proposal.

#### Correctness

Focused local validation:

```bash
python -m py_compile \
  gpurec/core/backward.py \
  gpurec/core/kernels/wave_backward.py \
  profiling/proposal2/bench_genewise_backward.py \
  tests/gradients/test_genewise_fused_backward.py

pytest -q \
  tests/kernels/test_wave_step_uniform_forward_kernel.py \
  tests/unit/test_specieswise_uniform.py \
  tests/gradients/test_genewise_fused_backward.py \
  tests/unit/test_genewise_wave.py::test_genewise_uniform_leaf_index_gradient_bridge_matches_dense_leaf_fallback \
  tests/gradients/test_autograd_bridge.py::test_genewise_uniform_backward_matches_individual_uniform_trees
```

Result:

```text
15 passed in 7.52 s
```

Broader backward-kernel and genewise-wave regression pass:

```bash
pytest -q tests/kernels/test_wave_backward_kernel.py tests/unit/test_genewise_wave.py
```

Result:

```text
25 passed in 12.43 s
```

The new dedicated genewise backward test file covers:

- small-tree `torch.autograd.gradcheck`;
- batched genewise weighted gradients against independent global/uniform
  per-family runs;
- high-`S` fp64 optimized genewise self-loop against a tight generic fallback,
  with a guard that fails if the optimized path reaches the old generic
  self-loop.

Direct kernel parity smokes from the implementation worker:

| Case | Output max diff | Gradient max diff |
|---|---:|---:|
| small `S=199`, scalar `[G]` | `0.000e+00` | `2.682e-07` |
| small `S=199`, specieswise `[G,S]` | `1.705e-13` | `1.192e-07` |
| large `S=1999`, scalar `[G]` | `7.451e-09` | `7.629e-06` |
| large `S=1999`, specieswise `[G,S]` | `3.725e-09` | `2.503e-06` |

Full `Pi_wave_backward` large smokes produced finite `v_Pi` and finite
parameter gradients for both scalar genewise and specieswise genewise layouts.

#### Benchmark

Clean local 10-family benchmark, `tests/data/test_trees_1000`, fp32,
`max_wave_size=4096`, pruning enabled:

```bash
PREPROCESS_CACHE_DIR=/tmp/gpurec_prop2_genewise_cache \
python profiling/proposal2/bench_genewise_backward.py \
  --dataset tests/data/test_trees_1000 \
  --fams 10 \
  --reps 5 \
  --warmups 2 \
  --max-wave-size 4096 \
  --backward-path optimized-genewise
```

The helper installs a guard that raises if `_self_loop_vjp_precompute` or
`_gmres_self_loop_solve` is reached, so the timing is definitely the optimized
family-indexed fused path.

| Path | Mean | Median | Min | Loss | Peak allocation |
|---|---:|---:|---:|---:|---:|
| generic self-loop fallback | `531.410 ms` | `531.569 ms` | `530.306 ms` | `22182.9140625` | `6.615 GB` |
| optimized genewise self-loop | `310.082 ms` | `310.018 ms` | `309.586 ms` | `22182.9140625` | `6.595 GB` |

This clean rerun gives a `1.71x` median speedup for the 10-family backward.
The absolute timings differ from the noisier Nsys profiling run, but the
direction is the same.

Profiler run from Pauli, same 10-family shape:

| Path | Mean | Median | Min | Peak allocation | Loss |
|---|---:|---:|---:|---:|---:|
| generic fallback | `669.738 ms` | `589.145 ms` | `556.977 ms` | `12.136 GB` | `22182.9140625` |
| optimized genewise self-loop | `483.833 ms` | `444.749 ms` | `309.859 ms` | `6.595 GB` | `22182.9140625` |

Nsys 10-family breakdown:

| Metric | Generic | Optimized |
|---|---:|---:|
| captured backward event time | `1053.412 ms` | `354.899 ms` |
| GPU active time | `508.093 ms` | `298.207 ms` |
| kernel launches | `25253` | `7394` |
| summed kernel time | `502.338 ms` | `297.140 ms` |
| `_wave_backward_uniform_kernel` | `0 / 0 ms` | `32 / 61.738 ms` |
| PyTorch elementwise/math/fill | `15944 / 266.712 ms` | `3457 / 98.848 ms` |
| cuSPARSE sparse matmul | `1039 / 94.721 ms` | `312 / 29.011 ms` |
| cat/materialization | `265 / 51.894 ms` | `201 / 47.639 ms` |
| scatter/add atomics | `891 / 22.667 ms` | `365 / 10.574 ms` |
| DTS forward recompute | `37 / 2.792 ms` | `37 / 2.744 ms` |

The optimized path removes most of the old generic self-loop launch pressure.
It does not change DTS forward recompute, DTS backward, or cross-Pibar VJP.

NCU on the largest optimized self-loop launch:

| Metric | Value |
|---|---:|
| duration | `29.44 ms` |
| grid / block | `(16645,1,1)` / `(128,1,1)` |
| registers/thread | `40` |
| achieved occupancy | `98.33%` |
| spills/local load-store | `0` |
| memory throughput | `204.28 GB/s` |
| DRAM read/write | `3.679 GB / 2.334 GB` |
| L1 / L2 hit rate | `24.52% / 93.13%` |
| global atom / red ops | `49.935M / 5.509M` |
| L2 atomic input cycles active | `36.09%` |
| issue slots busy | `8.64%` |

The kernel is not register- or occupancy-limited.  The new visible cost is
direct family-gradient atomic pressure, especially when many active rows in a
wave share the same family.

#### Remaining blocker

The 50-family genewise backward still OOMs even after Proposal 2.  The failure
is not in the fused self-loop.  It happens later in the remaining generic
DTS/Pibar path:

```text
gpurec/core/backward.py:2304 grad_DTS_5 = ...
tried to allocate 1.57 GiB
about 21.85 GiB already in use
```

The 1000-tree full backward was therefore not attempted.  The 1000-tree shape
is:

```text
S=1999 G=1000 C=6417248 waves=231 maxW=32768
split_rows=8018810 leaves=1605562 roots=1000
```

Decision: keep Proposal 2.  It is correct, removes the old generic self-loop
from optimized genewise backward, and gives a clear 10-family speedup.  The
next useful work is Proposal 3 and Proposal 4: make genewise DTS backward and
uniform Pibar VJP family-aware so the larger batches fit.

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

### Proposal 3 follow-up: implemented family-aware fused DTS backward

Proposal 3 was implemented and accepted for optimized genewise backward.

#### What changed in code

Before Proposal 3, the optimized genewise path still used the old PyTorch DTS
backward after the fused self-loop.  Each split wave materialized a
`[5,n_splits,S]` tensor and then scattered the five VJP components:

```python
# old genewise DTS backward, after Proposal 2 but before Proposal 3
fi = family_idx[ws + reduce_idx]        # parent family per split
pD = log_pD[fi]                         # [n_splits] or [n_splits,S]
pS = log_pS[fi]

DTS_5 = torch.stack([
    pD + Pi_l + Pi_r,                   # D
    Pi_l + Pibar_r,                     # T left
    Pi_r + Pibar_l,                     # T right
    pS + Pi_l_s1 + Pi_r_s2,             # S orientation 1
    pS + Pi_r_s1 + Pi_l_s2,             # S orientation 2
], dim=0)

grad_DTS_5 = v_parent * exp2(wlsp + DTS_5 - Pi_parent)

grad_log_pD.scatter_add_(family, grad_DTS_5[0].sum(dim=1))
grad_log_pS.scatter_add_(family, (grad_DTS_5[3] + grad_DTS_5[4]).sum(dim=1))
accumulated_rhs.index_add_(0, left_child,  grad_DTS_5[0] + grad_DTS_5[1] + S_terms)
accumulated_rhs.index_add_(0, right_child, grad_DTS_5[0] + grad_DTS_5[2] + S_terms)
grad_Pibar_l = grad_DTS_5[2]
grad_Pibar_r = grad_DTS_5[1]
```

The memory problem is the explicit `DTS_5` and `grad_DTS_5`: on high-fanout
root-like waves, their size is proportional to `5 * n_splits * S`, and the
following scatter/index operations add several large temporary tensors.

Proposal 3 moves that work into `_dts_cross_backward_accum_kernel`.  The new
path recomputes the five terms inside the Triton program, accumulates direct
Pi adjoints immediately, and writes parameter gradients directly to the right
layout:

```text
for split in split_wave:
    parent = ws + reduce_idx[split]
    family = family_idx[parent]              # only for genewise layouts

    for species tile s:
        pD_s = load_param(log_pD, family, s) # [G] or [G,S]
        pS_s = load_param(log_pS, family, s)

        vd0 = v_parent_s * exp2(wlsp + pD_s + Pi_l_s + Pi_r_s - Pi_parent_s)
        vd1 = v_parent_s * exp2(wlsp + Pi_l_s + Pibar_r_s - Pi_parent_s)
        vd2 = v_parent_s * exp2(wlsp + Pi_r_s + Pibar_l_s - Pi_parent_s)
        vd3 = v_parent_s * exp2(wlsp + pS_s + Pi_l_child1 + Pi_r_child2 - Pi_parent_s)
        vd4 = v_parent_s * exp2(wlsp + pS_s + Pi_r_child1 + Pi_l_child2 - Pi_parent_s)

        atomic_add accumulated_rhs[left,  s] += vd0 + vd1
        atomic_add accumulated_rhs[right, s] += vd0 + vd2
        atomic_add S-term child adjoints

        accumulate grad_log_pD / grad_log_pS in [G], [G,S], or [S]
        accumulate grad_mt
        optionally write compact Pibar-UD inputs for Proposal 4
```

This changes the algorithmic shape from "materialize five dense split/species
planes, then reduce/scatter them" to "stream through each split/species tile
once and atomically accumulate only the final VJP destinations".

The code baseline before the worker pass was commit `8346ad3`,
`Generalize fused backward DTS layouts`.  After that baseline, Worker 1 fixed
an edge-case classifier bug in `gpurec/core/kernels/wave_backward.py`: 1D
genewise scalar DTS parameters shaped `[G]` were previously classified as
shared specieswise `[S]` when `G == S`.  The layout classifier now prefers the
family-scalar layout when `family_idx` is present, and
`tests/kernels/test_dts_backward_accum_kernel.py` adds explicit `G == S`
family-scalar coverage.

The fused DTS backward accumulation path now supports four parameter layouts:

| Layout | Meaning | Tensor shape | Family index |
|---:|---|---|---|
| 0 | shared scalar | `[1]` | absent |
| 1 | shared species | `[S]` | absent |
| 2 | family scalar | `[G]` | parent family |
| 3 | family species | `[G,S]` | parent family |

For genewise scalar `[G]` parameters, the split parent row determines the
family:

```text
split_parent = ws + reduce_idx[split]
family = family_idx[split_parent]

atomic_add grad_log_pD[family] += sum_s vd0
atomic_add grad_log_pS[family] += sum_s (vd3 + vd4)
```

For genewise specieswise `[G,S]` parameters, the kernel accumulates directly to
`grad_log_pD[family,s]` and `grad_log_pS[family,s]` using atomics.  For shared
specieswise `[S]` parameters, `family_idx` is absent and the shared species
layout accumulates directly to `[s]`.

The fused kernel also keeps the direct Pi adjoint accumulation in
`accumulated_rhs`, merges the S-term when that path is enabled, and handles
`grad_mt` accumulation in-kernel.  When the fused flags are active, the
generic materialized `DTS_5 = torch.stack(...)` branch is avoided.

Worker 1 added guarded high-`S` checks that raise if the `DTS_5` fallback is
entered.  The guarded call counts were:

| Case | Fused DTS calls | Generic `DTS_5` fallback |
|---|---:|---:|
| shared specieswise `[S]` | 26 | 0 |
| genewise scalar `[G]` | 26 | 0 |
| genewise specieswise `[G,S]` | 26 | 0 |

#### Correctness

Local validation after the Worker 1 edge-case patch:

```bash
pytest -q \
  tests/kernels/test_dts_backward_accum_kernel.py \
  tests/gradients/test_genewise_fused_backward.py \
  tests/unit/test_specieswise_uniform.py::test_specieswise_uniform_backward_optimized_matches_reference
```

Result:

```text
85 passed, 8 skipped in 4.77s
```

Worker 1 focused validation:

| Check | Result |
|---|---:|
| full DTS kernel file | `81 passed, 8 skipped` |
| genewise fused backward | `3 passed` |
| specieswise optimized/reference backward | `1 passed` |
| finite difference `[True-True-uniform]` | passed |
| finite difference `[False-True-uniform]` | passed |

The tests cover shared specieswise, genewise scalar, and genewise specieswise
layouts, including the ambiguous `G == S` shape.  The fallback guards make the
optimized checks meaningful: they fail if the run silently routes through the
old materialized PyTorch DTS branch.

#### Benchmark

The main resident-chunk benchmark after Proposal 3 and Proposal 4 uses
`tests/data/test_trees_1000`, because `tests/data/test_trees_100` has
`S=199` and the optimized genewise guard intentionally does not activate when
`S <= 256`.

The controlled before/after comparison below keeps Proposal 2's optimized
family-indexed self-loop enabled and changes only the DTS/Pibar part of the
backward.  The 10-family runs use `max_wave_size=4096`; the 50-family runs use
`max_wave_size=32768`.

```bash
# before Proposal 3 and Proposal 4, but after Proposal 2 self-loop
GPUREC_FUSED_DTS_BACKWARD_ACCUM=0
GPUREC_DTS_PIBAR_UD_FUSION=0
GPUREC_FUSED_CROSS_PIBAR_VJP=0

# Proposal 3 only
GPUREC_FUSED_DTS_BACKWARD_ACCUM=1
GPUREC_DTS_PIBAR_UD_FUSION=0
GPUREC_FUSED_CROSS_PIBAR_VJP=0

# Proposal 3 + Proposal 4, current promoted path
GPUREC_FUSED_DTS_BACKWARD_ACCUM=1
GPUREC_DTS_PIBAR_UD_FUSION=1
GPUREC_FUSED_CROSS_PIBAR_VJP=1
GPUREC_FUSED_CROSS_PIBAR_VJP_IMPL=tree
```

Clean 10-family timing:

| Path | Median | Mean | Min | Peak allocation | Speedup vs old DTS/Pibar |
|---|---:|---:|---:|---:|---:|
| old generic DTS + old generic Pibar | `344.178 ms` | `344.377 ms` | `343.806 ms` | `6.594 GB` | baseline |
| Proposal 3 only: fused DTS, generic Pibar | `217.141 ms` | `218.669 ms` | `217.080 ms` | `2.847 GB` | `1.59x` |
| Proposal 3 + 4: fused DTS, staged Pibar | `95.779 ms` | `95.614 ms` | `95.004 ms` | `1.976 GB` | `3.59x` |

Proposal 3 alone saves `127.037 ms` median on this 10-family shape and removes
`3.747 GB` of peak allocation.  That is the cost of eliminating the dense
`DTS_5`/`grad_DTS_5` materialization and the PyTorch scatter/add work around
it.  Proposal 4 then saves another `121.362 ms` median by replacing the generic
Pibar VJP with the compact staged path.

The 50-family result is more important because this is the resident chunk size
that was failing before Proposals 3 and 4:

| Path | Result | Peak allocation | Interpretation |
|---|---:|---:|---|
| old generic DTS + old generic Pibar | OOM | `22.80 GiB` in use at failure | failed while allocating `grad_DTS_5` in `_safe_exp2_ratio` |
| Proposal 3 only: fused DTS, generic Pibar | `1271.534 ms` | `14.864 GB` | now fits; generic Pibar is still very slow |
| Proposal 3 + 4: fused DTS, staged Pibar | `337.239 ms` median | `10.550 GB` | `3.77x` faster than Proposal 3-only and `4.314 GB` lower peak |

So Proposal 3 is the change that turns the 50-family optimized-genewise
backward from "cannot complete" into a valid run.  Proposal 4 is the change
that makes the now-valid run close to the global/uniform timing envelope.

Current local 100-family genewise optimized run:

```bash
python profiling/proposal2/bench_genewise_backward.py \
  --dataset tests/data/test_trees_1000 \
  --fams 100 \
  --start 0 \
  --reps 2 \
  --warmups 1 \
  --max-wave-size 32768 \
  --backward-path optimized-genewise \
  --cache-dir /tmp/gpurec_current_bench_cache
```

| Path | Families | Median | Peak allocation |
|---|---:|---:|---:|
| optimized genewise | 100 | `651.534 ms` | `18.191 GB` |
| global reference | 100 | `716.431 ms` | `18.184 GB` |
| optimized specieswise | 100 | `963.523 ms` | `16.937 GiB` |

The optimized genewise path is now essentially global-speed on the same
100-family resident chunk, and slightly faster in these local runs.  The
specieswise path is slower because per-species gradients add atomic and
reduction pressure.

Additional worker and profiler runs:

| Run | Median / mean | Peak allocation |
|---|---:|---:|
| Worker 3, 50-family genewise optimized | `335.789 ms` median | `10.550 GB` |
| Worker 3, 100-family genewise optimized, expandable segments | `632.345 ms` median | `18.186 GB` |
| same local 100-family run under Nsys, reps 1/warmups 1 | `652.357 ms` backward | `18.191 GB` |
| Worker 2, 10-family high-`S` optimized guard | `95.658 ms` median/mean | `1.976 GB` |
| Worker 1, 2-family high-`S` fast benchmark | `35.439 ms` median | `0.506 GB` |

Nsys profile for the 100-family genewise script,
`/tmp/gpurec_prop34_genewise_bwd100.nsys-rep`:

| Kernel bucket | Time | Launches | Kernel time |
|---|---:|---:|---:|
| `_wave_backward_uniform_kernel` | `985.688 ms` | 78 | `57.2%` |
| `_wave_step_uniform_kernel` | `301.036 ms` | 672 | `17.5%` |
| `_dts_cross_backward_accum_kernel` | `122.079 ms` | 68 | `7.1%` |
| `_uniform_cross_pibar_vjp_tree_from_ud_compact_kernel` | `62.838 ms` | 68 | `3.6%` |
| `_dts_parent_reduced_ge2_stage1_kernel` | `61.487 ms` | 26 | `3.6%` |
| `_dts_fused_kernel` | `44.614 ms` | 132 | `2.6%` |
| `_wave_pibar_uniform_parent_kernel` | `39.130 ms` | 112 | `2.3%` |

CUDA memory operations were tiny relative to kernel time: D2H `2.674 ms`, H2D
`2.565 ms`, D2D `1.670 ms`, and memset `0.309 ms`.

NCU representative Proposal 3 launch,
`/tmp/gpurec_profile/prop34_genewise_backward/ncu/dts_accum_skip0.csv`:

| Metric | Value |
|---|---:|
| kernel | `_dts_cross_backward_accum_kernel` |
| grid / block | `(7330,1,1)` / `(128,1,1)` |
| duration | `1.020576 ms` |
| registers/thread | `96` |
| shared memory/block | `3072 B` |
| waves/SM | `11.45` |
| achieved active warps | `41.26%` |
| SM throughput | `27.77%` |
| memory throughput | `55.56%` |
| DRAM throughput | `~546 GB/s` |
| DRAM read/write | `352.35 MB / 205.13 MB` |
| L1 / L2 sector hit rate | `60.49% / 66.05%` |
| L2 atomic input cycles active | `25.91%` |
| global load/store/reduction requests | `6.919M / 0.953M / 3.709M` |
| local spilling requests | `0` |

Proposal 3 removed the PyTorch materializations it targeted.  The remaining
fused DTS accumulation cost is a memory plus global-reduction/atomic-pressure
kernel, not a compute-bound kernel.  It has no local spilling, so the next DTS
optimization target should be memory locality, reduction traffic, or staged
family reductions rather than register trimming.

#### Decision

Decision: accept and promote Proposal 3 for optimized genewise backward.

Keep the fallback gates.  Small-`S` fixtures intentionally use generic paths,
and the optimized path should continue to fail loudly in guarded benchmarks if
the family-aware fused DTS route is unavailable.

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

### Proposal 4 follow-up: implemented family-aware staged Pibar VJP

Proposal 4 was implemented and accepted for optimized genewise backward.

#### What changed in code

Before Proposal 4, even with Proposal 3's fused DTS accumulation, the
non-scalar genewise path still had to materialize Pibar side gradients and then
run the generic PyTorch Pibar VJP:

```python
# old Pibar VJP after Proposal 3 only
grad_Pibar_l, grad_Pibar_r = materialized_split_side_gradients
all_children = torch.cat([sl, sr])
all_pibar_grad = torch.cat([grad_Pibar_l, grad_Pibar_r])

nz = all_pibar_grad.abs().sum(dim=1) > 0
Pi_ch = Pi_star_wave[all_children[nz]]
p_prime = exp2(Pi_ch - Pi_ch.max(dim=1, keepdim=True))

anc_sum = p_prime @ ancestors_T
denom = p_prime.sum(dim=1, keepdim=True) - anc_sum
u_d = all_pibar_grad[nz] / denom
A = u_d.sum(dim=1, keepdim=True)
correction = (ancestors_T @ u_d.T).T
pi_from_pibar = p_prime * (A - correction)

accumulated_rhs.index_add_(0, all_children[nz], pi_from_pibar)
```

That code is mathematically correct, but it rebuilds dense `[active_sides,S]`
temporaries, performs sparse/dense ancestor products through PyTorch, and pays
extra filtering, concatenation, and `index_add_` overhead.  It also cannot use
the stable row-max values already computed during the forward pass unless those
values are saved for batched/genewise mode.

Proposal 4 makes the DTS kernel produce the normalized Pibar-UD inputs
directly, then passes them to the compact tree VJP kernel:

```text
# inside fused DTS backward, for each side of each split
family_l = family_idx[left_child]
family_r = family_idx[right_child]
mt_l_s = mt[family_l, s]
mt_r_s = mt[family_r, s]

ud_l[split,s] = grad_Pibar_l[split,s] * exp2(row_max[left]  + mt_l_s - Pibar[left,s])
ud_r[split,s] = grad_Pibar_r[split,s] * exp2(row_max[right] + mt_r_s - Pibar[right,s])
A_l[split] = sum_s ud_l[split,s]
A_r[split] = sum_s ud_r[split,s]

# then compact tree VJP
for active side:
    use pibar_ud[side,:] and A[side]
    walk compact species-tree levels
    atomic_add accumulated_rhs[child,:] += exact ancestor-corrected contribution
```

The forward-pass change is deliberately small but important:

```python
# old condition: batched/genewise runs did not save this row-max buffer
reuse_forward_pibar_stats = need_pibar and use_uniform_fused and not batched

# current condition: save one [C] row-max buffer whenever staged UD fusion needs it
reuse_forward_pibar_stats = (
    need_pibar
    and use_uniform_fused
    and GPUREC_DTS_PIBAR_UD_FUSION
)
```

This keeps the extra saved state to one `[C]` vector, not a row-expanded
`[C,S]` tensor.  Family dependence stays in-kernel through `mt[family,s]`
loads.

Batched/genewise forward now saves `uniform_pibar_row_max` when staged Pibar UD
fusion is enabled, including batched mode.  That saved row statistic lets the
backward use the same compact/tree Pibar VJP from the UD path instead of the
generic materialized PyTorch Pibar VJP.

The fused DTS backward can now output staged `pibar_ud` and `pibar_A` using
family-aware `mt` loads.  For family-indexed `mt`, the UD-producing path loads
`mt[family_l,s]` and `mt[family_r,s]` for the two child sides rather than
materializing row-expanded `mt` tensors.  Backward then calls
`_uniform_cross_pibar_vjp_tree_from_ud_compact_kernel`; the direct tree and
generic tree paths are not reached in the optimized genewise guard run.

Worker 2's 2-family high-`S` guard reported:

```text
forward_row_max_saved True (11734,) finite True
dts_calls=37
dts_ud_calls=37
from_ud_calls=37
direct_tree_calls=0
generic_tree_calls=0
gradients finite
```

This proves both halves of the Proposal 4 route: forward saves the row maxima
needed by the staged path, and backward consumes the compact UD data rather
than routing through generic materialized Pibar VJP.

#### Before/after timing

The Proposal 4 isolated comparison starts from Proposal 3-only: fused DTS is
enabled, but staged Pibar UD fusion and fused cross-Pibar VJP are disabled.
The promoted path then enables the staged UD route.

| Families | Proposal 3 only: generic Pibar | Proposal 3 + 4: staged Pibar | Median speedup | Peak allocation change |
|---:|---:|---:|---:|---:|
| 10 | `217.141 ms`, `2.847 GB` | `95.779 ms`, `1.976 GB` | `2.27x` | `-0.871 GB` |
| 50 | `1271.534 ms`, `14.864 GB` | `337.239 ms`, `10.550 GB` | `3.77x` | `-4.314 GB` |

The larger gain at 50 families is expected.  The old Pibar VJP path scales with
the number of split sides that survive pruning and with dense `[side,S]`
temporaries.  The staged route keeps one compact side worklist and performs the
ancestor correction in the same Triton tree kernel family already used by the
global path.

The combined effect from the old generic DTS/Pibar route to the promoted
Proposal 3+4 route on the 10-family shape is:

```text
344.178 ms -> 95.779 ms  (3.59x faster)
6.594 GB  -> 1.976 GB    (-4.618 GB peak allocation)
```

On the 50-family shape, the combined effect is stronger qualitatively:

```text
old generic DTS/Pibar: OOM in grad_DTS_5
promoted Proposal 3+4: 337.239 ms median, 10.550 GB peak
```

#### Correctness

Worker 2 focused Proposal 4 validation:

```text
29 passed, 8 skipped
```

Worker 2 broader regression:

```text
24 passed
```

The broader pass covered genewise wave, wave-step, autograd, and fused
regression tests.  Worker 3 also reran the public autograd bridge and gradient
parity checks:

| Check | Result |
|---|---:|
| autograd bridge gradchecks, parity, finite difference | `8 passed` |
| genewise fused backward tests | `3 passed` |
| specieswise checks | `2 passed` |

Extra genewise individual-uniform parity from Worker 3:

| Metric | Max diff |
|---|---:|
| per-family NLL | `0.0` |
| weighted gradient absolute/relative | `1.967215204623507e-07` |

The exercised tolerances were:

| Check | Tolerance |
|---|---|
| gradcheck | `eps=1e-4`, `atol=2e-3`, `rtol=2e-3` |
| genewise batched vs individual uniform NLL | `1e-10` |
| genewise batched vs individual uniform weighted gradient | `2e-6` |
| specieswise optimized vs reference loss | `atol=2e-3`, `rtol=1e-5` |
| specieswise optimized vs reference gradient | `atol=3e-2`, `rtol=2e-2` |

#### Profiling

NCU representative Proposal 4 launch,
`/tmp/gpurec_profile/prop34_genewise_backward/ncu/pibar_ud_compact_skip0.csv`:

| Metric | Value |
|---|---:|
| kernel | `_uniform_cross_pibar_vjp_tree_from_ud_compact_kernel` |
| grid / block | `(14660,1,1)` / `(128,1,1)` |
| duration | `0.378752 ms` |
| registers/thread | `36` |
| shared memory/block | `1024 B` |
| waves/SM | `9.54` |
| active warps | `94.32%` |
| SM throughput | `24.91%` |
| memory throughput | `78.04%` |
| DRAM throughput | `~767 GB/s` |
| DRAM read/write | `191.27 MB / 99.30 MB` |
| L1 / L2 sector hit rate | `73.09% / 66.51%` |
| L2 atomic input cycles active | `8.94%` |
| global load/store/reduction requests | `4.851M / 0.366M / 0.501M` |
| local spilling requests | `0` |

Proposal 4 successfully converts genewise uniform Pibar VJP to the compact
staged path.  The representative kernel is memory-bandwidth dominated, but it
is not spill- or occupancy-limited.  Atomic pressure is visible but far smaller
than in fused DTS accumulation.

#### Decision

Decision: accept and promote Proposal 4 for optimized genewise backward.

Keep fallback gates because the family-aware fused path requires the forward
row maxima and staged tree metadata.  Generic and direct-tree fallbacks remain
useful for unsupported shapes, small fixtures, and debugging, but optimized
genewise benchmarks should guard that they are not used accidentally.

With Proposal 3 and Proposal 4 promoted, the remaining 100-family bottleneck is
no longer generic DTS/Pibar materialization.  The Nsys profile is now dominated
by the self-loop backward, then forward wave-step/fixed-point kernels, then
fused DTS/Pibar.  Specieswise remains slower because per-species parameter
gradients introduce additional atomic and reduction work.

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

### Proposal 5 follow-up: implemented explicit chunked autograd harness

Proposal 5 was implemented and accepted as the benchmark and training-step
strategy for full genewise uniform backward passes.

#### What changed in code

Before Proposal 5, `profiling/proposal2/bench_genewise_backward.py` measured a
single resident `GeneReconModel`:

```text
build one model over all requested families
for rep:
    loss = model()
    loss.backward()
```

That is the right resident-chunk microbenchmark, but it cannot represent a
1000-family training step because autograd must keep the full forward state
resident until that model's backward pass consumes it.  The Proposal 5 harness
adds `--family-chunk-size` and changes the large benchmark shape to:

```text
for rep:
    total_loss = 0
    grad_chunks = []

    for family_start, family_stop in chunk_ranges:
        model = build GeneReconModel for this family chunk
        model.static.warm_E = None

        loss = model()                 # saves chunk-local Pi/Pibar/row stats
        loss.backward()                # consumes those saved tensors
        grad_chunks.append(theta.grad)
        total_loss += loss

        delete loss/model
        optionally empty CUDA cache

    grad_theta = concat(grad_chunks)
```

The important lifetime change is that each chunk's full `Pi`, `Pibar`,
`uniform_pibar_row_max`, adjoints, and backward scratch die immediately after
that chunk's `loss.backward()`.  Only the small per-family loss and
`theta.grad` slice survive across chunks.

Worker A added the following harness features to
`profiling/proposal2/bench_genewise_backward.py`:

- `--family-chunk-size`, with `0` preserving the old single-model benchmark;
- chunked autograd scheduling over the requested family range;
- per-chunk shape rows: families, `S`, clade rows `C`, wave count, max wave
  rows, split rows, leaves, and roots;
- per-chunk timing rows: forward CUDA-event time, backward CUDA-event time,
  forward peak allocation, backward peak allocation, loss, and build time;
- active path flag printing so the result records the optimized route actually
  used;
- `--strict-optimized-kernels`, enabled by default for optimized genewise
  runs;
- `--empty-cache-between-chunks`, enabled by default to keep the allocator from
  carrying a large previous resident chunk into the next one.

Worker A also added a core fallback guard in `gpurec/core/backward.py`:
`GPUREC_REQUIRE_OPTIMIZED_GENEWISE_BACKWARD=1`.  For genewise uniform backward,
that guard raises if the run reaches any of these generic routes:

- fused self-loop gate inactive;
- generic DTS forward recompute fallback;
- generic PyTorch DTS backward accumulation fallback;
- generic cross-Pibar VJP fallback.

The benchmark helper sets this guard automatically for
`--backward-path optimized-genewise --strict-optimized-kernels`.

Worker B added `_chunked_genewise_nll_and_grad` and chunk-size invariance tests
to `tests/gradients/test_genewise_fused_backward.py`.  The helper runs the same
semantic chunked autograd schedule as the benchmark, but compares per-family
NLLs and concatenated gradients against larger resident chunks.

#### Correctness

Focused correctness commands:

```bash
pytest -q \
  tests/gradients/test_genewise_fused_backward.py::test_genewise_uniform_backward_invariant_under_family_chunk_size \
  tests/gradients/test_genewise_fused_backward.py::test_genewise_uniform_optimized_high_s_fp64_chunk_size_parity_is_strict \
  tests/gradients/test_genewise_fused_backward.py::test_genewise_uniform_optimized_high_s_fp64_matches_generic_fallback
```

Result:

```text
3 passed
```

Public autograd bridge checks:

```bash
pytest -q \
  tests/gradients/test_autograd_bridge.py::test_genewise_uniform_backward_matches_individual_uniform_trees \
  tests/gradients/test_autograd_bridge.py::test_per_family_sums_to_total \
  tests/gradients/test_autograd_bridge.py::test_no_grad_genewise_uniform_uses_equivalent_inference_path
```

Result:

```text
3 passed
```

Full genewise fused backward file:

```bash
pytest -q tests/gradients/test_genewise_fused_backward.py
```

Result:

```text
5 passed
```

The measured parity errors were:

| Check | NLL max abs diff | Gradient max abs diff |
|---|---:|---:|
| small fp64 chunk size 1 vs resident 4 | `0.0` | `1.947263035262e-07` |
| small fp64 chunk size 2 vs resident 4 | `0.0` | `2.388680298004e-08` |
| high-`S` fp64 optimized chunk size 1 vs resident 2 | `0.0` | `8.394351880270e-10` |
| high-`S` fp64 optimized vs generic fallback | `0.0` | `9.208633855451e-12` |

These checks prove that splitting families changes only the autograd tensor
lifetime, not the per-family objective or gradient.

#### Commands

Worker A strict guard smoke:

```bash
python profiling/proposal2/bench_genewise_backward.py \
  --dataset tests/data/test_trees_1000 \
  --fams 4 \
  --start 0 \
  --family-chunk-size 2 \
  --reps 1 \
  --warmups 0 \
  --max-wave-size 4096 \
  --backward-path optimized-genewise \
  --cache-dir /tmp/gpurec_prop5_smoke_cache
```

Worker C full 1000-family chunk-size 50 run:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python profiling/proposal2/bench_genewise_backward.py \
  --dataset tests/data/test_trees_1000 \
  --start 0 \
  --fams 1000 \
  --family-chunk-size 50 \
  --warmups 1 \
  --reps 1 \
  --backward-path optimized-genewise
```

Worker C full 1000-family chunk-size 100 run:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python profiling/proposal2/bench_genewise_backward.py \
  --dataset tests/data/test_trees_1000 \
  --start 0 \
  --fams 1000 \
  --family-chunk-size 100 \
  --warmups 1 \
  --reps 1 \
  --backward-path optimized-genewise
```

Parent local strict guard smoke:

```bash
PREPROCESS_CACHE_DIR=/tmp/gpurec_prop5_cache \
python profiling/proposal2/bench_genewise_backward.py \
  --dataset tests/data/test_trees_1000 \
  --fams 2 \
  --family-chunk-size 1 \
  --reps 1 \
  --warmups 0 \
  --backward-path optimized-genewise \
  --max-wave-size 4096
```

All optimized benchmark runs printed the active path flags:

```text
mode genewise
pibar_mode uniform
fixed_iters_Pi 6
family_idx 1
leaf_index 1
forward_family_consts 1
forward_family_dts_params 1
fused_genewise_backward 1
fused_genewise_self_loop 1
fused_uniform_backward 1
kernelized_backward_dts 1
fused_dts_backward_accum 1
dts_pibar_ud_fusion 1
fused_cross_pibar_vjp 1
cross_pibar_impl tree
strict_optimized_kernels 1
require_optimized_guard 1
```

So the full 1000-family results are guarded optimized-kernel results, not
silent generic fallbacks.

#### Benchmark

Resident microbenchmarks from the Proposal 3/4 baseline and Worker C:

| Run | Families resident | Forward | Backward | Peak allocation | Shape |
|---|---:|---:|---:|---:|---|
| optimized resident smoke | 10 | not reported | `95.509 ms` | `2.812 GB` | `S=1999`, `C=66530`, `45` waves, maxW `16645`, split rows `83135` |
| optimized resident chunk | 100 | about `229-276 ms` locally | `651.534 ms` median in the Proposal 3/4 run; `683-736 ms` per 100-family chunk in Worker C | `18.191 GB` to `18.840 GB` | `S=1999`, about `625k-653k` clade rows |

A single resident 1000-family backward is not a valid training policy on the
24 GB RTX 4090 target.  Extrapolating the resident 100-family state already
exceeds memory by a wide margin, and Proposal 5's purpose is to avoid that
unbounded saved-tensor lifetime.  The useful comparison is therefore resident
per-chunk memory versus full-dataset chunked autograd time.

Full 1000-family chunked autograd:

| Chunk size | Chunks | Forward total | Backward total | Total event time | Max backward peak | Loss |
|---:|---:|---:|---:|---:|---:|---:|
| 50 | 20 | `2980.171 ms` | `7672.296 ms` | `10652.466 ms` | `10.787 GB` (`10.046 GiB`) | `2157097.03906250` |
| 100 | 10 | `2574.073 ms` | `6723.733 ms` | `9297.806 ms` | `18.840 GB` (`17.546 GiB`) | `2157097.01562500` |

Chunk shape ranges:

| Chunk size | `C` range | Waves | Max wave rows | Split rows | Leaves | Roots |
|---:|---:|---:|---:|---:|---:|---:|
| 50 | `312366-331290` | `49-54` | `32768` | `390320-413975` | `78154-82885` | `50` |
| 100 | `624908-652936` | `56-61` | `32768` | `780860-815895` | `156352-163359` | `100` |

Per-chunk backward times:

| Chunk size | Backward times by chunk |
|---:|---|
| 50 | `378.349`, `359.613`, `420.895`, `390.491`, `382.500`, `403.112`, `387.854`, `420.363`, `390.251`, `402.881`, `373.290`, `376.255`, `375.781`, `359.001`, `361.643`, `364.908`, `395.473`, `384.066`, `367.112`, `378.456` ms |
| 100 | `683.279`, `667.243`, `666.241`, `677.759`, `676.506`, `666.291`, `667.945`, `653.834`, `705.125`, `659.508` ms |

Small strict guard smokes:

| Run | Chunk size | Families | Forward | Backward | Total | Peak | Interpretation |
|---|---:|---:|---:|---:|---:|---:|---|
| Worker A smoke | 2 | 4 | `2203.145 ms` | `2877.023 ms` | `5080.168 ms` | `0.627 GB` max | verifies harness and strict optimized guard on two chunks |
| Parent local smoke | 1 | 2 | `4450.047 ms` | `5915.515 ms` | `10365.562 ms` | `0.258 GB` | deliberately slow chunk size 1 guard smoke |

The smoke runs use tiny chunks and, for the parent smoke, `max_wave_size=4096`.
They should not be used as performance guidance; they only prove that the
strict optimized route can be exercised under chunked autograd.

#### Interpretation

Chunk size 100 is the faster 1000-family policy in this benchmark:

```text
chunk 50:  10.652 s total, 10.787 GB peak
chunk 100:  9.298 s total, 18.840 GB peak
```

The speedup comes from halving the number of chunks.  Each 100-family chunk
also has larger waves and fewer repeated model/build and launch overheads.  The
cost is memory: `18.840 GB` allocated is close enough to a 24 GB card that
allocator fragmentation, additional optimizer state, larger parameter layouts,
or a slightly larger dataset could push it over the limit.

Chunk size 50 is slower by about `1.15x` on event time, but it leaves a much
larger memory margin.  Its `10.787 GB` peak leaves room for optimizer state,
diagnostics, other allocations, or less favorable family mixes.  It is the
safer default for general training and CI/profiling reproducibility.

The loss differs by only about `0.0234` between the two full 1000-family fp32
runs on a `~2.16e6` total loss.  That is consistent with fp32 reduction-order
differences across different chunk groupings; the correctness tests above use
fp64 and explicit chunk parity to verify the underlying gradients.

#### Decision

Decision: accept Proposal 5.

Use explicit chunked autograd as the production strategy for full genewise
training passes.  Recommend chunk size 50 as the conservative default on 24 GB
GPUs, with chunk size 100 as the faster high-memory option when the benchmark
prints the strict optimized flags and the observed peak leaves enough margin
for the intended optimizer.

Keep `GPUREC_REQUIRE_OPTIMIZED_GENEWISE_BACKWARD=1` in optimized profiling
runs.  A full 1000-family timing is only meaningful if generic self-loop,
generic DTS, and generic Pibar VJP fallbacks are forbidden.

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

### Proposal 6 follow-up: current scheduler already preserves family locality

Proposal 6 was tested and accepted as a scheduler guardrail item.  No
scheduler change is needed for the current genewise uniform path: the existing
scheduler already emits family-major row blocks at each wave depth, and the
`max_wave_size` split step slices those blocks contiguously rather than
interleaving families.

#### What was inspected

The production genewise builders all follow the same ordering pattern:

- `gpurec/api/model.py::_build_static_state`;
- `gpurec/core/model.py::GeneDataset.compute_likelihood_batch`;
- `gpurec/optimization/wave_optimizer.py` lazy family batches;
- `gpurec/optimization/genewise_optimizer.py` chunked optimization layouts.

They build per-family waves, merge them with `collate_wave(...)`, optionally
split oversized waves with `split_phase_waves(...)`, and then call
`build_wave_layout(...)`.

The actual current ordering is:

```text
for depth k in cross-family wave depths:
    wave = []

    for family in resident family order:
        if family has wave depth k:
            for local_clade in family_waves[family][k]:
                wave.append(local_clade + family_clade_offset[family])

    for start in range(0, len(wave), max_wave_size):
        emit wave[start : start + max_wave_size]

build_wave_layout emitted waves:
    inv_perm = concatenated emitted clades
    family_idx[new_row] = family owning inv_perm[new_row]
```

This is exactly the locality-preserving order proposed above: within each
emitted wave, rows for a family form one contiguous run; families appear in
resident family order; and splitting can cut a family run at a wave boundary
but does not interleave another family into the middle of that run.

No scheduler code was changed.  The tested implementation is the existing
`collate_wave` plus `split_phase_waves` behavior.

Worker code/test changes were limited to instrumentation and guardrails:

- `profiling/proposal2/bench_genewise_backward.py` now computes
  `_wave_family_locality(model)` from `wave_metas` and `family_idx`, then
  prints `family_locality_summary` plus per-chunk `family_runs`,
  `family_touches`, `extra_family_runs`, `interleaved_waves`,
  `locality_preserved`, `run_per_row`, and `runs_per_family_touch` fields.
- `tests/unit/test_genewise_wave.py` now has helpers for collated wave
  construction, topological validation, family-run validation, and split-wave
  parity.
- The added tests assert that `split_phase_waves` preserves topological order,
  keeps each family's rows in a single ordered block per emitted wave, and
  leaves genewise uniform likelihood/gradient unchanged after forced wave
  splitting.

#### Locality metrics

Locality was measured on `tests/data/test_trees_1000`, `S=1999`,
`max_wave_size=32768`, using the same 50-family and 100-family resident chunk
sizes used by the Proposal 5 backward benchmark.

| Resident families | `C` | Waves before split | Waves after split | Mixed waves | Total family runs | Max runs/wave | Family reappears inside a wave | Same-family adjacent rows | Family-boundary cuts |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | `321930` | `47` | `49` | `48` | `2165` | `50` | `0` | `99.342614%` | `2` |
| 100 | `635372` | `47` | `56` | `56` | `4333` | `100` | `0` | `99.326792%` | `9` |

Run length distribution:

| Resident families | Min run | Median run | P90 run | Max run |
|---:|---:|---:|---:|---:|
| 50 | `1` | `46` | `376` | `1994` |
| 100 | `1` | `46` | `372` | `2290` |

The short runs are mostly roots and small late-depth waves.  The hot large
waves still have long contiguous family blocks.  The `0` reappearance count is
the important guardrail: no emitted wave has a pattern like
`family 0, family 1, family 0`.

As a negative-control comparison, a synthetic round-robin scheduler was built
in memory for the same per-depth rows.  It preserved dependency correctness
but deliberately destroyed family locality:

| Resident families | Ordering | Same-family adjacent rows | Total family runs | Max runs/wave | Locality violations |
|---:|---|---:|---:|---:|---:|
| 50 | current family-major | `99.343%` | `2165` | `50` | `0` |
| 50 | synthetic round-robin | `0.115%` | `321561` | `32768` | `48` |
| 100 | current family-major | `99.327%` | `4333` | `100` | `0` |
| 100 | synthetic round-robin | `0.050%` | `635053` | `32768` | `55` |

This shows the metric has enough sensitivity to catch the bad ordering
Proposal 6 was concerned about.

#### Correctness checks

Focused checks:

```bash
pytest -q \
  tests/unit/test_genewise_wave.py::test_genewise_wave_order_is_topological_and_family_blocked_after_splitting \
  tests/unit/test_genewise_wave.py::test_genewise_uniform_split_wave_layout_preserves_likelihood_and_gradient \
  tests/unit/test_genewise_wave.py::test_genewise_uniform_family_indexed_constants_match_row_expanded \
  tests/unit/test_genewise_wave.py::test_genewise_uniform_family_indexed_dts_params_match_row_expanded \
  tests/gradients/test_genewise_fused_backward.py::test_genewise_uniform_backward_invariant_under_family_chunk_size
```

Result:

```text
7 passed in 2.81s
```

The synthetic round-robin negative-control forward also produced identical
root-row NLLs to the current family-major layout for the 50-family and
100-family chunks:

```text
50 families:  max abs NLL diff 0
100 families: max abs NLL diff 0
```

That confirms the ordering choice is a performance/locality concern, not a
semantic dependency.

#### Benchmark and profiling comparison

A controlled forward-only comparison was run on a 50-family chunk after both
layouts were compiled and warmed.  The benchmark alternated the current
family-major layout with the synthetic round-robin layout for 12 samples each,
using fixed 6 Pi iterations, root-row output, uniform Pibar, and
`max_wave_size=32768`.

| 50-family forward layout | Median | Mean | Min | Max |
|---|---:|---:|---:|---:|
| current family-major | `132.483 ms` | `130.991 ms` | `119.523 ms` | `133.070 ms` |
| synthetic round-robin | `132.777 ms` | `130.480 ms` | `116.671 ms` | `135.028 ms` |

Peak allocation was unchanged in the earlier 50-family spot pass
(`5.666 GB` for both layouts), because the same clade rows and tensors are
resident.  A 100-family spot pass also preserved correctness and allocation,
but its timing was not used for the decision because the alternating 50-family
run already showed no stable forward benefit from changing row locality.

The interpretation is narrow: family locality is already good, and making it
bad did not create a stable forward timing loss in this quick comparison.  That
means family constant vector locality is not the dominant current forward
limit for this shape, or it is hidden under the larger wave-step traffic.  It
does not justify changing the scheduler.

The backward locality/profile worker then ran the promoted optimized genewise
path with strict kernel guards enabled:

```text
GPUREC_REQUIRE_OPTIMIZED_GENEWISE_BACKWARD=1
strict_optimized_kernels=1
forward_family_consts=1
forward_family_dts_params=1
fused_genewise_backward=1
fused_genewise_self_loop=1
fused_uniform_backward=1
fused_dts_backward_accum=1
fused_cross_pibar_vjp=1
```

Resident warmed backward timing:

| Resident families | `max_wave_size` | Backward median | Peak alloc | Locality result |
|---:|---:|---:|---:|---|
| 50 | `32768` | `341.446 ms` | `10.550 GB` | `extra_family_runs=0`, `runs_per_family_touch=1.0` |
| 100 | `32768` | `637.644 ms` | `18.191 GB` | `extra_family_runs=0`, `runs_per_family_touch=1.0` |
| 50 | `4096` | `352.922 ms` | `9.022 GB` | `extra_family_runs=0`, `runs_per_family_touch=1.0` |
| 100 | `4096` | `673.838 ms` | `17.822 GB` | `extra_family_runs=0`, `runs_per_family_touch=1.0` |

The smaller `4096` split size preserved locality, but added waves and slowed
the backward pass by `3.4%` for 50 families and `5.7%` for 100 families.  That
is consistent with launch/scheduling overhead and less work per launch; it is
not evidence that extra splitting helps family-cache locality.

Full 1000-family cold chunked passes were used for coverage rather than final
timing:

| Family chunk size | Chunks | Total backward | Peak backward alloc | Locality result |
|---:|---:|---:|---:|---|
| 100 | 10 | `10298.976 ms` | `18.844 GB` | every chunk `extra_family_runs=0` |
| 50 | 20 | `16132.647 ms` | `10.789 GB` | every chunk `extra_family_runs=0` |

Representative Proposal 6 NCU captures:

| Kernel | Shape | L2 hit | DRAM bytes | Global load sectors/request | Achieved active warps | Issue active |
|---|---:|---:|---:|---:|---:|---:|
| `_wave_step_uniform_kernel` | 50 fams, `grid=(32768,1,1)` | `92.13%` | `494.6 MB` | `6.21` | `98.09%` | `76.53%` |
| `_wave_backward_uniform_kernel` | 50 fams, skipped to large launch, `grid=(609,1,1)` | `97.27%` | `5.57 MB` | `6.74` | `12.28%` | `6.17%` |

The forward wave-step launch is already highly occupied and mostly limited by
the large wave-step traffic.  The sampled backward launch has very high cache
hit rate but poor warp and issue utilization because the launch is small
(`609` programs).  Neither profile suggests that changing family row order is
the next bottleneck; future work should focus on the wave-step and self-loop
kernels themselves, chunk sizing, and launch parameter autotuning.

#### Decision

Decision: accept Proposal 6 as already satisfied by the current scheduler.

Keep the family-major wave order as an invariant for genewise chunks.  If a
future load-balancing scheduler is introduced, it should report the locality
metrics above and reject wave layouts where a family appears in multiple runs
inside the same emitted wave.  It is acceptable for `max_wave_size` to cut a
single family block across adjacent emitted waves; it should not interleave
another family into that block.

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

### Proposal 7 follow-up: dedicated harnesses accepted

Proposal 7 is accepted.  The implementation adds both dedicated genewise
chunking entrypoints, strengthens the backward harness with explicit active
path verdicts, and adds subprocess tests that exercise the actual CLI
contracts.  The purpose is measurement hygiene: a command-line benchmark now
prints enough state to tell whether it measured the optimized genewise kernels
or a fallback path.

#### What changed

Worker changes visible in the worktree:

- `profiling/bench_genewise_forward_chunking.py` is a dedicated genewise
  uniform forward inference harness.  It runs `GeneReconModel` in no-grad mode
  with root-row likelihood output, family-indexed constants,
  family-indexed DTS parameters, parent-reduced DTS, leaf-index leaves, and no
  backward saved tensors.
- `profiling/bench_genewise_backward_chunking.py` is a dedicated wrapper for
  genewise uniform backward chunking.  It delegates to
  `profiling/proposal2/bench_genewise_backward.py` so the benchmark logic stays
  in one implementation.
- The wrapper sets strict optimized defaults before delegation:

```text
BACKWARD_PATH=optimized-genewise
STRICT_OPTIMIZED_KERNELS=1
FAMILY_CHUNK_SIZE=50
GPUREC_BENCH_VALIDATE_OPTIMIZED_FLAGS=1
```

- `profiling/proposal2/bench_genewise_backward.py` now prints an
  `optimized_path_verdict` row after `active_path_flags`.
- The verdict checks whether the run is actually the intended optimized
  genewise backward path: full saved tensors for backward, no root-row-only
  output, `family_idx`, fused uniform backward, fused genewise backward,
  fused genewise self-loop, fused DTS backward accumulation, compact/tree
  Pibar VJP, CUDA dtype support, `S > 256`, strict optimized kernels, and the
  require-optimized guard.
- When `GPUREC_BENCH_VALIDATE_OPTIMIZED_FLAGS=1`, strict optimized runs raise
  if any required optimized feature is inactive.
- `tests/unit/test_genewise_chunking_harnesses.py` adds subprocess contract
  coverage for the proposed forward/backward harness CLIs.  It looks for the
  active path flags, strict optimized verdict, root-row/full-saved-tensor
  contract, fallback reporting, locality counters, and family-run counters.

The forward harness CLI defaults are:

| Option | Default / behavior |
|---|---|
| `--dataset` | `tests/data/test_trees_1000` |
| `--family-start` / `--fams` | `0` / `50` |
| `--family-chunk-size` | `150`, the usual 1000-family likelihood-only policy becomes `150x6 + 100` |
| `--max-wave-size` | `32768` |
| `--warmups` / `--reps` | `3` / `7` |
| `--dtype` / `--device` | `fp32` / `cuda` |
| `--strict-optimized-kernels` | enabled |

It sets these forward defaults before building models:

```text
GPUREC_FORWARD_LEAF_INDEX=1
GPUREC_FORWARD_PARENT_REDUCED_DTS=1
GPUREC_FORWARD_PARENT_REDUCED_DTS_IMPL=tiled
GPUREC_FORWARD_PARENT_REDUCED_DTS_GE2_ONLY=1
GPUREC_FORWARD_FAMILY_INDEXED_CONSTS=1
GPUREC_FORWARD_FAMILY_INDEXED_DTS_PARAMS=1
GPUREC_UNIFORM_PINGPONG=1
```

The backward wrapper inherits the Proposal 5 benchmark CLI.  Important
defaults from `--help` are:

| Option | Default / behavior |
|---|---|
| `--dataset` | `tests/data/test_trees_1000` |
| `--fams` | `50` |
| `--start` | `0` |
| `--reps` / `--warmups` | `9` / `5` |
| `--family-chunk-size` | wrapper default is `50`, giving the memory-safe Proposal 5 policy for 1000 families |
| `--max-wave-size` | `32768` |
| `--pruning-threshold` / `--use-pruning` | `1e-6` / enabled |
| `--backward-path` | wrapper default is `optimized-genewise` |
| `--strict-optimized-kernels` | wrapper default is enabled |
| `--empty-cache-between-chunks` | enabled |

Both wrappers now default to chunked policies when the requested family count
exceeds the chunk size.  Users can still pass `--family-chunk-size 0` for an
explicit resident single-model experiment, but the default entrypoint behavior
is OOM-resistant for the 1000-family profiling case.

#### Why this matters

Proposal 7 is mostly about measurement hygiene.  The earlier genewise work had
multiple paths that could silently measure the wrong thing: dense leaf terms,
row-expanded constants, row-expanded DTS parameters, generic self-loop, generic
PyTorch DTS accumulation, or generic cross-Pibar VJP.  A timing number is only
useful if the benchmark says which path was active.

The current backward harness now makes the key choices explicit:

```text
active_path_flags
  mode genewise
  pibar_mode uniform
  fixed_iters_Pi 6
  family_idx 1
  leaf_index 1
  forward_family_consts 1
  forward_family_dts_params 1
  fused_genewise_backward 1
  fused_genewise_self_loop 1
  fused_uniform_backward 1
  kernelized_backward_dts 1
  fused_dts_backward_accum 1
  dts_pibar_ud_fusion 1
  fused_cross_pibar_vjp 1
  cross_pibar_impl tree
  strict_optimized_kernels 1
  require_optimized_guard 1

optimized_path_verdict
  verdict optimized
  generic_pytorch_fallback 0
  root_row_output 0
  full_saved_tensors_for_backward 1
  compact_tree_pibar_vjp 1
```

That directly addresses the old mistake: a result cannot look like an
optimized backward number unless the harness also shows the fused backward,
compact Pibar, family-indexed forward constants, family-indexed DTS params,
full saved tensors, and strict guard state.

The forward harness prints the complementary inference contract:

```text
active_path_flags
  mode genewise
  pibar_mode uniform
  fixed_iters_Pi 6
  root_row_output 1
  need_pibar 0
  full_saved_tensors 0
  backward_saved_tensors 0
  forward_only_no_grad 1
  family_idx 1
  leaf_index 1
  parent_reduced_dts 1
  family_indexed_constants 1
  family_indexed_dts_params 1
  generic_pytorch_fallback 0
```

#### Commands and local evidence

Strict optimized 10-family forward timing smoke:

```bash
PREPROCESS_CACHE_DIR=/tmp/gpurec_prop7_cache \
python profiling/bench_genewise_forward_chunking.py \
  --dataset tests/data/test_trees_1000 \
  --fams 10 \
  --family-chunk-size 5 \
  --reps 1 \
  --warmups 0 \
  --max-wave-size 32768
```

Result:

| Families | Chunk size | Chunks | Forward | Peak | Loss |
|---:|---:|---:|---:|---:|---:|
| 10 | 5 | 2 | `370.227 ms` | `0.605 GB` | `22182.91503906` |

Per-chunk rows:

| Chunk | Families | `C` | Waves | MaxW | Split rows | Forward | Peak |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 5 | `32127` | `45` | `8038` | `40145` | `309.972 ms` | `0.565 GB` |
| 1 | 5 | `34403` | `45` | `8607` | `42990` | `60.255 ms` | `0.605 GB` |

Like the backward smoke below, this uses `--warmups 0`, so the first chunk
includes one-time setup and compilation effects.  The useful evidence is the
printed forward contract: root-row output is enabled, saved backward tensors
are disabled, family-indexed constants/DTS params are enabled, and the generic
fallback flag is zero.

Strict optimized 2-family stats-only wrapper smoke:

```bash
PREPROCESS_CACHE_DIR=/tmp/gpurec_prop7_cache \
python profiling/bench_genewise_backward_chunking.py \
  --dataset tests/data/test_trees_1000 \
  --fams 2 \
  --family-chunk-size 1 \
  --stats-only \
  --reps 1 \
  --warmups 0 \
  --max-wave-size 4096
```

Key output:

```text
chunked_autograd_policy ... families 2 chunks 2 family_chunk_size 1 backward_path optimized-genewise
active_path_flags ... family_idx 1 ... strict_optimized_kernels 1 require_optimized_guard 1
optimized_path_verdict verdict optimized ... root_row_output 0 full_saved_tensors_for_backward 1 ... compact_tree_pibar_vjp 1
chunk_shape 0 ... families 1 S 1999 C 6071 waves 42 maxW 1519 split_rows 7586 ... locality_preserved 1
chunk_shape 1 ... families 1 S 1999 C 5663 waves 42 maxW 1417 split_rows 7076 ... locality_preserved 1
```

This proves the wrapper default selects optimized genewise backward, enables
the strict guard, validates the active features, and prints chunk/locality
shape rows before running any long timing loop.

Strict optimized 10-family timing smoke:

```bash
PREPROCESS_CACHE_DIR=/tmp/gpurec_prop7_cache \
python profiling/bench_genewise_backward_chunking.py \
  --dataset tests/data/test_trees_1000 \
  --fams 10 \
  --family-chunk-size 5 \
  --reps 1 \
  --warmups 0 \
  --backward-path optimized-genewise \
  --strict-optimized-kernels \
  --max-wave-size 32768
```

Result:

| Families | Chunk size | Chunks | Forward | Backward | Total | Max backward peak | Loss |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 5 | 2 | `353.190 ms` | `639.171 ms` | `992.360 ms` | `1.460 GB` | `22182.91503906` |

Per-chunk rows:

| Chunk | Families | `C` | Waves | MaxW | Split rows | Forward | Backward | Backward peak |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 5 | `32127` | `45` | `8038` | `40145` | `297.819 ms` | `542.826 ms` | `1.359 GB` |
| 1 | 5 | `34403` | `45` | `8607` | `42990` | `55.371 ms` | `96.345 ms` | `1.460 GB` |

The first chunk includes more one-time setup and kernel warmup despite
`--warmups 0`; this is a smoke result, not a stable performance gate.  It
does prove that the chunked wrapper can run forward, save full tensors, run
backward, produce finite gradients, and report timing/memory by chunk.

Strict optimized 50-family timing sample:

```bash
PREPROCESS_CACHE_DIR=/tmp/gpurec_prop7_cache \
python profiling/bench_genewise_backward_chunking.py \
  --dataset tests/data/test_trees_1000 \
  --fams 50 \
  --family-chunk-size 50 \
  --reps 1 \
  --warmups 0 \
  --backward-path optimized-genewise \
  --strict-optimized-kernels \
  --max-wave-size 32768
```

Result:

| Families | Chunk size | `C` | Waves | MaxW | Split rows | Forward | Backward | Total | Max backward peak | Loss |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | 50 | `321930` | `49` | `32768` | `402275` | `385.250 ms` | `849.806 ms` | `1235.056 ms` | `10.542 GB` | `107804.26562500` |

This 50-family single-rep sample is slower than the promoted Proposal 5
multi-rep numbers, which is expected with `--warmups 0`.  The useful Proposal
7 evidence is not the absolute time; it is that the timing row carries the
optimized verdict, strict guard state, chunk shape, locality counters, and
memory peak in the same output.

Minimal Nsight Systems smoke on the backward wrapper:

```bash
PREPROCESS_CACHE_DIR=/tmp/gpurec_prop7_cache \
nsys profile --force-overwrite=true --stats=false \
  --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  -o /tmp/gpurec_profile/prop7_harness/backward_f2_chunk1 \
  python profiling/bench_genewise_backward_chunking.py \
    --dataset tests/data/test_trees_1000 \
    --fams 2 \
    --family-chunk-size 1 \
    --max-wave-size 32768 \
    --warmups 0 \
    --reps 1
```

The generated trace is:

```text
/tmp/gpurec_profile/prop7_harness/backward_f2_chunk1.nsys-rep
```

Top CUDA kernel buckets from `nsys stats --report cuda_gpu_kern_sum`:

| Kernel | Time | Instances | Share |
|---|---:|---:|---:|
| `_wave_backward_uniform_kernel` | `27.715 ms` | `51` | `44.0%` |
| `_wave_step_uniform_kernel` | `10.922 ms` | `504` | `17.4%` |
| `_uniform_cross_pibar_vjp_tree_kernel` | `3.685 ms` | `49` | `5.9%` |
| `_wave_pibar_uniform_parent_kernel` | `1.421 ms` | `84` | `2.3%` |
| `_dts_cross_backward_accum_kernel` | `1.225 ms` | `49` | `1.9%` |
| `_dts_fused_kernel` | `0.760 ms` | `129` | `1.2%` |

This is not a performance profile for decision-making because the shape is only
two families and uses cold single-rep timing.  It is a harness smoke: the new
entrypoint can be profiled directly with Nsight, and the trace contains the
intended optimized Triton kernels rather than only generic PyTorch fallback
work.

Fallback-mode smoke:

```bash
PREPROCESS_CACHE_DIR=/tmp/gpurec_prop7_cache \
python profiling/bench_genewise_backward_chunking.py \
  --dataset tests/data/test_trees_1000 \
  --fams 2 \
  --family-chunk-size 0 \
  --stats-only \
  --reps 0 \
  --warmups 0 \
  --backward-path generic-self-loop-fallback \
  --max-wave-size 4096
```

Key output:

```text
backward_path generic-self-loop-fallback
active_path_flags ... fused_genewise_self_loop 0 fused_uniform_backward 0 ... require_optimized_guard 0
optimized_path_verdict verdict non_optimized ... fused_uniform_backward 0 ... fused_genewise_self_loop 0 ... require_optimized_guard 0
```

This proves fallback runs are labeled non-optimized instead of silently
competing with optimized results.

#### Correctness and guard tests

Focused correctness tests that already cover the optimized backward/chunking
semantics:

```bash
pytest -q \
  tests/gradients/test_genewise_fused_backward.py::test_genewise_uniform_backward_invariant_under_family_chunk_size \
  tests/gradients/test_genewise_fused_backward.py::test_genewise_uniform_optimized_high_s_fp64_chunk_size_parity_is_strict \
  tests/gradients/test_genewise_fused_backward.py::test_genewise_uniform_optimized_high_s_fp64_matches_generic_fallback \
  tests/unit/test_genewise_wave.py::test_genewise_uniform_leaf_index_path_matches_dense_leaf_fallback \
  tests/unit/test_genewise_wave.py::test_genewise_uniform_leaf_index_gradient_bridge_matches_dense_leaf_fallback
```

Result:

```text
6 passed in 6.18s
```

New Proposal 7 subprocess contract tests:

```bash
pytest -q tests/unit/test_genewise_chunking_harnesses.py
```

Current result:

```text
3 passed in 7.62s
```

The tests run the forward and backward wrappers in `--stats-only` mode through
subprocesses, so they check the real command-line contract rather than helper
functions.  The backward negative case disables strict optimized guards and
requires either an explicit non-optimized/fallback marker or a guard-state
marker such as `require_optimized_guard 0`.

Strict preflight failure was also smoke-tested by forcing
`GPUREC_FUSED_CROSS_PIBAR_VJP_IMPL=torch` through the backward wrapper.  The
command exited nonzero before timing and reported
`unsupported optimized features are inactive: compact_tree_pibar_vjp`, with
`optimized_path_verdict verdict non_optimized generic_pytorch_fallback 1`.

#### Decision

Decision: accept Proposal 7.

Use `profiling/bench_genewise_forward_chunking.py` as the supported genewise
uniform likelihood-only forward profiling entrypoint, and use
`profiling/bench_genewise_backward_chunking.py` as the supported genewise
uniform backward chunking entrypoint.  The older Proposal 2 benchmark remains
the underlying implementation for backward, but the new wrapper supplies safe
optimized defaults and strict active-feature preflight checks.

The next worker pass can use these wrappers for Proposal 8 correctness-matrix
measurements and for the first small autotuning sweep, instead of calling
older profiling scripts directly.

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
| 2 | Genewise profiling harnesses | Accepted; dedicated forward/backward wrappers now print optimized/non-optimized contracts and default to chunked policies | subprocess CLI contract: `3 passed`; backward fallback smoke is labeled `non_optimized` |
| 3 | Fused genewise backward self-loop | Accepted; removes the generic self-loop for supported genewise uniform batches | 10-family backward: `531.569 ms -> 310.018 ms` locally |
| 4 | Fused genewise DTS backward accumulation | Accepted; removes generic `DTS_5` and split scatter work | 100-family DTS bucket: `122.079 ms`, no guarded fallback |
| 5 | Family-aware uniform Pibar VJP | Accepted; reuses exact ancestor-corrected compact UD tree VJP kernels | 100-family Pibar UD bucket: `62.838 ms`, no generic tree calls |
| 6 | Saved-tensor and chunked autograd strategy | Accepted; streams full genewise training through resident chunks with strict optimized guards | 1000-family chunked backward: chunk 100 `6.724 s`, `18.840 GB`; chunk 50 `7.672 s`, `10.787 GB` |
| 7 | Preserve family locality in wave scheduling | Accepted as current scheduler invariant; no code change needed | 50/100-family waves have `0` family reappearances and about `99.3%` same-family adjacency |
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

- Proposal 2 now makes the optimized genewise backward self-loop exist and fit
  for 10-family chunks.  In a clean local benchmark it improved the 10-family
  backward from `531.569 ms` median to `310.018 ms` median.
- Proposal 3 and Proposal 4 remove the previous generic DTS/Pibar
  materialization blocker from optimized genewise backward.
- The current 100-family optimized genewise backward on `test_trees_1000` is
  `651.534 ms` median with `18.191 GB` peak allocation.  The same local
  100-family global reference run is `716.431 ms` median with `18.184 GB`
  peak allocation, so the optimized genewise path is now essentially
  global-speed per resident chunk.
- Proposal 5 makes the full 1000-family optimized genewise training pass
  measurable by streaming resident chunks.  Chunk size 100 completed the full
  pass in `9.298 s` forward+backward event time with `18.840 GB` peak
  allocation.  Chunk size 50 completed in `10.652 s` with `10.787 GB` peak.
  The recommended default is 50 for memory margin, with 100 available as the
  faster high-memory setting.
- The 100-family optimized specieswise run is `963.523 ms` median with
  `16.937 GiB` peak allocation.  Specieswise remains slower because it
  accumulates per-species parameter gradients and pays more atomic/reduction
  cost.
- Nsys now shows the 100-family backward dominated by
  `_wave_backward_uniform_kernel`, followed by forward wave-step/fixed-point
  work, then fused DTS and compact Pibar.  Future work should optimize those
  remaining kernels or improve chunking, not chase the removed generic
  DTS/Pibar branches.
- Proposal 6 does not change the performance envelope.  It confirms that the
  promoted wave layout already preserves family locality: current 50-family
  and 100-family chunks have no family reappearance inside a wave and about
  `99.3%` same-family adjacent rows.  A synthetic round-robin negative control
  destroyed locality but did not produce a stable 50-family forward timing
  difference, so there is no scheduler change to promote.
- Proposal 7 does not change kernel performance.  It improves benchmark
  interpretability: the forward wrapper prints root-row/no-saved-tensor
  inference status, and the backward wrapper prints whether the run used full
  saved tensors, fused genewise/uniform backward, compact/tree Pibar VJP, and
  an `optimized` or `non_optimized` verdict.  A local single-rep 10-family
  forward wrapper sample reported `370.227 ms` and `0.605 GB` peak.  A local
  single-rep 50-family backward wrapper sample reported `385.250 ms` forward,
  `849.806 ms` backward, and `10.542 GB` max backward peak with the strict
  optimized verdict.  Treat these as harness evidence only; the promoted
  Proposal 5 multi-rep numbers remain the performance envelope.

## Cross-Mode Triton Autotuning Opportunity

The optimized global/uniform, genewise, and specieswise paths now share most
of the same Triton kernels.  That makes Triton meta-parameter tuning a useful
cross-mode follow-up rather than a mode-specific rewrite.  The relevant knobs
are compile-time constants or launch options, so they affect register pressure,
occupancy, instruction count, loop trip count, and memory coalescing without
changing the mathematical recurrence.

The current code already exposes a few knobs through environment variables,
but the settings were chosen by focused sweeps rather than by an explicit
autotuning cache:

| Kernel area | Current knobs | Why it may matter |
|---|---|---|
| Forward wave step | `GPUREC_FORWARD_WAVE_BLOCK_S`, `GPUREC_FORWARD_WAVE_NUM_WARPS` | Changes species tile width and CTA warp count for `_wave_step_uniform_kernel`; affects ancestor-walk loop count, register pressure, and occupancy. |
| Backward self-loop | `GPUREC_WAVE_BLOCK_S`, `GPUREC_WAVE_NUM_WARPS` | Same species-tile tradeoff for `_wave_backward_uniform_kernel`, currently the largest backward bucket in optimized genewise/global runs. |
| Self-loop parameter two-stage reduction | `GPUREC_SELF_LOOP_PARAM_TWO_STAGE_TILE_ROWS`, `GPUREC_SELF_LOOP_PARAM_TWO_STAGE_BLOCK_S`, `GPUREC_SELF_LOOP_PARAM_TWO_STAGE_REDUCE_TILES`, `GPUREC_SELF_LOOP_PARAM_TWO_STAGE_SCALAR_BLOCK`, stage warps | Controls how row tiles write compact partial reductions before the final parameter-gradient reduction. |
| Parent-reduced DTS forward/backward recompute | `GPUREC_FORWARD_PARENT_REDUCED_DTS_TILE_SPLITS`, `GPUREC_BACKWARD_PARENT_REDUCED_DTS_TILE_SPLITS`, fixed `BLOCK_S=128` today | Balances parallelism over high-fanout split groups against partial-buffer traffic and per-CTA serial work. |
| DTS backward accumulation | mostly fixed `BLOCK_S=min(256,next_power_of_2(S))`, fixed warps in several launches | Still a major kernel; tile width may trade fewer species blocks for higher register pressure. |
| Compact/tree Pibar VJP | mostly fixed `BLOCK_S=min(256,next_power_of_2(S))`, `num_warps=4` | The best point may differ between global, genewise, and specieswise because pruning density and active-side patterns differ. |
| Segmented logsumexp fallback | `BLOCK_H`, `BLOCK_S`, `num_warps`, `num_stages`; `@triton.autotune` scaffold exists but is disabled | Low priority unless this path reappears in a hot profile, but it is an obvious mechanical autotune target. |

The first autotune grid should stay small and profile-driven:

```text
forward wave:
    BLOCK_S in {128, 256, 512}
    num_warps in {4, 8}

backward self-loop:
    BLOCK_S in {128, 256, 512}
    num_warps in {4, 8, 16}

parent-reduced DTS:
    TILE_SPLITS in {32, 64, 128, 256}

DTS backward accumulation:
    BLOCK_S in {128, 256, 512}
    num_warps in {4, 8}

compact/tree Pibar VJP:
    BLOCK_S in {128, 256, 512}
    num_warps in {4, 8}
```

This should not be raw `@triton.autotune` on every production run.  The compile
and search cost is too high, and the correct key is not just `S`.  A practical
design is an offline benchmark/autotune script that writes a small cache:

```text
(kernel, mode, dtype, S, wave_size_bucket, fanout_bucket, pruning_bucket)
    -> {BLOCK_S, num_warps, num_stages, TILE_SPLITS, TILE_ROWS}
```

Production code would then read the cached choice and launch the already
compiled specialization directly.  The benchmark key should include mode
because specieswise/genewise/global now share kernels but not always the same
constant layout, active mask density, or parameter-gradient layout.

Correctness validation must account for reduction-order changes.  Changing
`BLOCK_S` or `TILE_SPLITS` can alter fp32 summation order, so `Pi` entries and
gradients may move by small amounts even when the algorithm is correct.  The
autotuned configurations should therefore be validated with the same policy as
other optimized kernels:

```text
1. fp64 high-iteration likelihood parity when implementations should match;
2. fp32 likelihood parity at optimized tolerances;
3. gradient parity versus the current optimized default;
4. gradcheck / finite-difference checks on small trees;
5. Nsys/NCU confirmation that the chosen config improves the intended kernel
   without increasing launch count or moving time into another bucket.
```

Previous focused sweeps already give useful priors.  On the forward side,
`BLOCK_S=256, num_warps=8` helped some 50-family runs but was not uniformly
best at larger sizes.  On the backward side, the self-loop kernel was already
near its best measured point at `BLOCK_S=256`, `num_warps=8`, while
`num_warps=16` increased pressure.  Parent-reduced DTS settled near
`TILE_SPLITS=64` in the documented global/uniform sweeps.  Autotuning is
therefore expected to produce incremental gains and better mode/dataset
robustness, not a new order-of-magnitude improvement.

## Immediate Next Experiment

Proposal 0 through Proposal 6 have succeeded and are promoted for the supported
optimized genewise uniform path.  Proposal 7 is partially promoted for backward
profiling only.

The immediate next experiment should finish Proposal 7 before starting a larger
autotuning pass: align the subprocess contract test with the new
`optimized_path_verdict` wording and decide whether the backward wrapper should
default to chunk size 50.  After the forward/backward harness contract is
green, use those entrypoints for the first small wave-step and self-loop Triton
launch-parameter sweep on 50-family and 100-family chunks.  Nsys/NCU should
confirm whether any gain comes from the intended kernel buckets rather than
from changing launch count or moving time into DTS or Pibar VJP.
