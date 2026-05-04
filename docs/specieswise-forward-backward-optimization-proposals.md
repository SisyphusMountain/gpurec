# Specieswise Uniform Forward/Backward Optimization Notes

This note summarizes the 3-worker specieswise pass from May 4, 2026.  The
target was non-genewise specieswise mode with `pibar_mode="uniform"` and theta
shape `[S, 3]`.

## Main Finding

Specieswise uniform forward was already mostly using the global uniform fast
path.  Unlike genewise mode, specieswise mode does not need per-family indexing:

```text
log_pD, log_pS, mt, E, Ebar, E_s1, E_s2: [S]
```

Those tensors are shared species vectors.  The default fused wave-step and DTS
kernels can load them directly as `[S]` constants, so there is no `[W,S]`
constant expansion to remove and no `[split_rows,S]` family-parameter expansion
like the genewise Proposal 0/1 fixes.

The real missing corner was the optional two-kernel uniform path.  That path
computed `Pibar` in one kernel and the `DTS_L` update in a second kernel, but
the second kernel still expected a dense leaf-term buffer.  The implementation
now threads compact leaf addressing through that path:

```text
leaf_species_idx[row] -> species id or -1
leaf_logp[s]          -> log_pS for the hit species
```

The two-kernel `DTS_L` kernel now reconstructs the leaf term on the fly:

```text
if leaf_species_idx[ws + row] == s:
    t_leaf = leaf_logp[s]
else:
    t_leaf = -inf
```

This matches the default one-kernel uniform path and avoids materializing a
dense `[W,S]` leaf matrix when `GPUREC_UNIFORM_IMPL=two`.

## Code Changes

- `gpurec/core/forward.py`
  - Enables the compact leaf-index condition for the two-kernel uniform path.
  - Passes `leaf_species_idx` and `leaf_logp` into
    `wave_step_uniform_two_kernel_fused`.

- `gpurec/core/kernels/wave_step.py`
  - Adds `USE_LEAF_INDEX` and `LEAF_LOGP_MODE` to the
    `_wave_step_uniform_from_pibar_kernel`.
  - Supports shared specieswise `[S]`, genewise scalar `[G]`, and genewise
    specieswise `[G,S]` leaf log-probability layouts.

- `tests/kernels/test_wave_step_uniform_forward_kernel.py`
  - Adds a two-kernel parity test comparing compact leaf indexing with the old
    dense leaf-term buffer.

- `tests/unit/test_specieswise_uniform.py`
  - Adds specieswise optimized-vs-reference forward parity.
  - Adds specieswise optimized-vs-reference backward parity.
  - Adds the constant-specieswise semantic check: constant `[S,3]` rates must
    match global mode, and the specieswise gradient summed over species must
    match the global gradient.
  - Adds an AleRax specieswise fixture regression over all 100 families in
    `tests/data/test_trees_100`.  The test loads branch-specific D/L/T rates
    from `output_specieswise/model_parameters/model_parameters.txt`, runs the
    optimized specieswise uniform likelihood in fp64, and compares each family
    with `output_specieswise/per_fam_likelihoods.txt`.

- `profiling/bench_specieswise_uniform.py`
  - Adds a focused specieswise benchmark harness with forward, backward,
    reference comparison, semantic comparison, root-row likelihood timing, and
    family-chunked forward timing.

## Correctness

Focused local validation after merging the worker patches:

```bash
python -m py_compile \
  gpurec/core/forward.py \
  gpurec/core/backward.py \
  gpurec/core/kernels/wave_step.py \
  gpurec/core/kernels/wave_backward.py \
  profiling/bench_specieswise_uniform.py \
  tests/unit/test_specieswise_uniform.py

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

Broader regression pass after the staged changes:

```bash
pytest -q tests/kernels/test_wave_backward_kernel.py tests/unit/test_genewise_wave.py
```

Result:

```text
25 passed in 12.43 s
```

Additional worker checks:

| Check | Result |
|---|---:|
| full forward kernel test file | `7 passed` |
| genewise family-indexed constant/DTS regression checks | `4 passed` |
| two-kernel leaf-index smoke | root rows and `Pibar` max diff `0.0` |
| specieswise unit tests | `4 passed` |

The AleRax fixture comparison uses the per-family log-likelihoods as written in
`output_specieswise/per_fam_likelihoods.txt`; no extra `ln(S)` origin correction
is applied for this fixture.  With fp64, fixed-point tolerances of `1e-10`, and
4000 maximum E/Pi iterations, the local maximum absolute per-family difference
was below `5e-4` nats.

## Forward Timing

Likelihood-only forward, `tests/data/test_trees_1000`, fp32, root rows,
`need_pibar=False`, fixed 6 Pi iterations, RTX 4090:

| Families | Chunk size | Chunks | Median Pi/forward time | Peak allocation | Notes |
|---:|---:|---:|---:|---:|---|
| 50 | resident | 1 | `116.770 ms` | `10.254 GiB` | reference compare: NLL abs diff `0.015625` |
| 150 | resident | 1 | `343.152 ms` | `15.007 GiB` | timing-only |
| 1000 | 50 | 20 | `2803.312 ms` | `5.762 GiB` | slower because many chunks |
| 1000 | 150 | 7 | `2318.478 ms` | `15.681 GiB` | clean local rerun |

The 150-family chunk is the useful setting for likelihood-only inference on the
24 GB RTX 4090: it reaches essentially the same timing envelope as current
global/uniform and genewise forward, while still fitting in memory.

Small Pi-only worker smoke on `tests/data/test_trees_100`, `S=199`,
`C=45396`:

| Path | Global median | Specieswise median |
|---|---:|---:|
| default fused uniform | `5.61 ms` | `5.55 ms` |
| `GPUREC_UNIFORM_IMPL=two` | `7.81 ms` | `7.75 ms` |

The specieswise overhead is therefore not visible in the wave kernels.  The
main forward variable is chunk size.

## Backward Profiling

The specieswise backward story is different.  In the chunked training profile,
resident public autograd mode OOMed at only 50 trees on the 24 GB card:

```text
tried to allocate 1.57 GiB with the process already near 22.54 GiB
```

Chunking controls memory, but turns the workload into many small chunks and
therefore many launches.  Ampere profiled the current chunked path with
family-batch size 5:

| Trees | Chunks | C / waves / split rows | E ms | Pi ms | Grad ms | Layout ms | Full step ms | Wall ms |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 50 | 10 | `321930 / 49 / 402275` | `0.516` | `164.380` | `3479.998` | `277.271` | `3645.055` | `3790.046` |
| 150 | 30 | `954706 / 65 / 1192970` | `0.485` | `490.553` | `11735.916` | `457.444` | `12229.221` | `12697.988` |
| 1000 | 200 | `6417248 / 231 / 8018810` | `0.442` | `3168.378` | `59809.562` | `2532.529` | `62978.437` | `65567.667` |

The stable 1000-tree row is the second repetition; the first repetition was
inflated by first-pass effects and an `E_iters=11` solve.

Nsight Systems on the 50/150-tree chunked profile:

| Trees | Kernel launches | Kernel GPU ms | Memcpy count / bytes / ms | Top CPU runtime |
|---:|---:|---:|---|---|
| 50 | `251829` | `2455.405 ms` | `51650 / 3.913 GB / 54.739 ms` | `cudaStreamSynchronize`: `25008` calls, `1717.985 ms` |
| 150 | `747428` | `7272.164 ms` | `153137 / 11.606 GB / 164.364 ms` | `cudaStreamSynchronize`: `74061` calls, `5059.638 ms` |

Kernel buckets were stable:

| Bucket | Share |
|---|---:|
| PyTorch elementwise | `39.8%` |
| cuSPARSE | `17.8-17.9%` |
| indexing/scatter/gather | `9.3%` |
| cat materialization | `9.0-9.1%` |
| copy/fill | `8.4-8.5%` |
| gpurec custom Triton/CUDA | `6.0-6.1%` |

NCU resource snapshots:

| Kernel | Duration | SM | DRAM | L2 | Active warps | Registers | Block/grid |
|---|---:|---:|---:|---:|---:|---:|---|
| `_wave_step_uniform_kernel` | `60.544 us` | `60.72%` | `27.08%` | `23.63%` | `53.32%` | `40` | `128 / 1001` |
| `CatArrayBatchedCopy` | `3.424 us` | `3.88%` | `19.10%` | `11.09%` | `16.82%` | `30` | `128 / 306` |
| `csrmm_alg2_kernel` | `6.368 us` | `18.49%` | `4.25%` | `4.56%` | `14.69%` | `53` | `128 / 259` |

Interpretation: specieswise forward is already where we wanted it.  Specieswise
training/backward is not bottlenecked by the custom forward wave kernel; it is
dominated by generic backward ATen work, launch count, synchronizations,
materialization, indexing/scatter/gather, and cuSPARSE.

## Decisions

- Accept the compact leaf-index support for the two-kernel uniform path.
- Keep 150-family chunks as the default benchmark point for specieswise
  likelihood-only forward on 24 GB hardware.
- Do not spend more effort on specieswise-specific forward materialization
  removal unless a new profile shows a new bottleneck.  The current
  specieswise 1000-tree forward is `2318.478 ms`, which is effectively the
  same as current global/uniform and genewise forward timings.
- Treat specieswise backward as a separate optimization wave.  The next useful
  work is not in `_wave_step_uniform_kernel`; it is to make the backward path
  use the same fused uniform ideas that were successful for global and
  genewise self-loop/DTS/Pibar VJPs.
