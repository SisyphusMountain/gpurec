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
| 3 | Fused genewise backward self-loop | Accepted; removes the generic self-loop for supported genewise uniform batches | 10-family backward: `531.569 ms -> 310.018 ms` locally |
| 4 | Fused genewise DTS backward accumulation | Accepted; removes generic `DTS_5` and split scatter work | 100-family DTS bucket: `122.079 ms`, no guarded fallback |
| 5 | Family-aware uniform Pibar VJP | Accepted; reuses exact ancestor-corrected compact UD tree VJP kernels | 100-family Pibar UD bucket: `62.838 ms`, no generic tree calls |
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
- The 100-family optimized specieswise run is `963.523 ms` median with
  `16.937 GiB` peak allocation.  Specieswise remains slower because it
  accumulates per-species parameter gradients and pays more atomic/reduction
  cost.
- Nsys now shows the 100-family backward dominated by
  `_wave_backward_uniform_kernel`, followed by forward wave-step/fixed-point
  work, then fused DTS and compact Pibar.  Future work should optimize those
  remaining kernels or improve chunking, not chase the removed generic
  DTS/Pibar branches.

## Immediate Next Experiment

Proposal 0 through Proposal 4 have succeeded and are promoted for the supported
optimized genewise uniform path.

The next experiment should be Proposal 5: run the optimized backward under an
explicit chunked-autograd scheduler and measure the full 1000-tree training
pass without accidentally retaining all saved tensors.  The scheduler should
start with 50-family and 100-family chunks on `tests/data/test_trees_1000`,
print the active fused path flags, and reject guarded runs that fall back to
generic DTS or generic Pibar VJP.
