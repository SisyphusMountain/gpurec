# Lean Fast Path

This branch is anchored to the measured `test_trees_1000` path:

- `pibar_mode="uniform"`
- full global/specieswise/genewise wave layout
- parent-reduced DTS forward
- fused uniform forward self-loop
- Proposal 0 uniform backward self-loop
- fused DTS backward accumulation
- tree-based cross-Pibar VJP
- benchmark memory policy: family chunk size `25`, max wave size `8192`

The benchmark memory policy above records the measured pruning baseline, not
the production workflow defaults.  `RunConfig` currently defaults to one
resident batch (`family_chunk_size=0`), `clade_budget=305000`, and
`max_wave_size=8192`.

Measured full-dataset result before this pruning:

```text
forward_median_ms  2445.187
backward_median_ms 3532.057
total_median_ms    5979.043
peak_gib           5.942
generic_self_loop_calls 0
strict_optimized_verdict pass
```

The retained benchmark command is:

```bash
python profiling/bench_uniform_forward_backward_pipeline.py \
  --dataset /path/to/test_trees_1000 \
  --fams 1000 \
  --family-chunk-size auto \
  --max-wave-size auto \
  --fixed-iters 6 \
  --neumann-terms 3 \
  --warmups 1 \
  --reps 3 \
  --strict-optimized-kernels
```

Current benchmark status after removing the debug inclusion-DAG payload from
preprocessing: a valid full 1000-family timed run exists.  Windowed preflight
remains a setup diagnostic only; it reports `performance_evidence 0` and must
not justify deleting self-loop backends, active-mask pruning modes, environment
flags, or scheduler policies without a completed timed benchmark.

## 2026-05-22 Benchmark Attempts

The full 1000-family benchmark was retried on an RTX 4090 workstation with
125 GiB system RAM and 17 GiB free on the root filesystem.  A larger cache
target was available at `/media/enzo/Stockage`.

No-cache full timed attempt:

```bash
timeout 3600s env PYTHONDONTWRITEBYTECODE=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /usr/bin/time -v python profiling/bench_uniform_forward_backward_pipeline.py \
  --dataset tests/data/test_trees_1000 \
  --fams 1000 \
  --family-chunk-size auto \
  --max-wave-size auto \
  --fixed-iters 6 \
  --neumann-terms 3 \
  --warmups 1 \
  --reps 3 \
  --strict-optimized-kernels \
  --compare-unchunked-max-fams 0 \
  --progress-jsonl
```

This run was stopped during preprocessing before timed reps.  It reached
224/1000 families and 43,117,352 KiB maximum resident set size in 1:45.85.
The memory trajectory matched the previous setup blocker, so this is not valid
performance evidence.

Full timed attempt:

```bash
timeout 7200s env PYTHONDONTWRITEBYTECODE=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /usr/bin/time -v python profiling/bench_uniform_forward_backward_pipeline.py \
  --dataset tests/data/test_trees_1000 \
  --fams 1000 \
  --family-chunk-size auto \
  --max-wave-size auto \
  --fixed-iters 6 \
  --neumann-terms 3 \
  --warmups 1 \
  --reps 3 \
  --strict-optimized-kernels \
  --compare-unchunked-max-fams 0 \
  --progress-jsonl
```

This run was killed by the OS before timed reps after reaching 640/1000
families.  Maximum resident set size was 127,440,120 KiB.  This is also not
valid performance evidence for the full 1000-family path.

Largest completed timed fallback in the same session:

```bash
timeout 3600s env PYTHONDONTWRITEBYTECODE=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /usr/bin/time -v python profiling/bench_uniform_forward_backward_pipeline.py \
  --dataset tests/data/test_trees_1000 \
  --fams 512 \
  --family-chunk-size auto \
  --max-wave-size auto \
  --fixed-iters 6 \
  --neumann-terms 3 \
  --warmups 1 \
  --reps 3 \
  --strict-optimized-kernels \
  --compare-unchunked-max-fams 0
```

This partial benchmark completed with strict optimized kernels active and
finite gradients:

```text
pipeline_policy families 512 chunks 21 family_chunk_size 25 max_wave_size 8192
strict_optimized_verdict pass
compare_unchunked skipped reason fams_above_threshold fams 512 threshold 0
pipeline_summary reps 3
forward_median_ms 1131.508
backward_median_ms 857.255
total_median_ms 1988.967
max_peak_gib 5.223
max_peak_reserved_gib 7.482
grad_finite 1
```

Host maximum resident set size for the 512-family timed fallback was
87,603,236 KiB.  Before removing the inclusion-DAG cache payloads, the
1000-family blocker was still a host-memory/setup blocker, not a timed runtime
regression result.

The host-memory blocker was traced to debug clade-inclusion payloads in
preprocess caches.  For the 640 cached families above, the cache occupied
101.842 GiB, of which 101.382 GiB was only `ccp.inclusion_parents` and
`ccp.inclusion_children`; all other tensors were 0.422 GiB.  Those inclusion
payloads are not required by the optimized forward/backward iterations.

After deleting the generated caches and removing inclusion-DAG materialization
from C++ preprocessing, the retained full benchmark completed:

```bash
timeout 7200s env PYTHONDONTWRITEBYTECODE=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /usr/bin/time -v python profiling/bench_uniform_forward_backward_pipeline.py \
  --dataset tests/data/test_trees_1000 \
  --fams 1000 \
  --family-chunk-size auto \
  --max-wave-size auto \
  --fixed-iters 6 \
  --neumann-terms 3 \
  --warmups 1 \
  --reps 3 \
  --strict-optimized-kernels \
  --compare-unchunked-max-fams 0
```

```text
pipeline_policy families 1000 chunks 40 family_chunk_size 25 max_wave_size 8192
strict_optimized_verdict pass
compare_unchunked skipped reason fams_above_threshold fams 1000 threshold 0
pipeline_summary reps 3
forward_median_ms 2179.338
backward_median_ms 1637.578
total_median_ms 3817.207
max_peak_gib 5.377
max_peak_reserved_gib 6.490
grad_finite 1
```

The full run used 13,562,496 KiB maximum resident set size and completed in
4:09.40 wall time including preprocessing and layout.  The rebuilt cache for
the full 1000-family dataset is about 734 MiB rather than the prior 100+ GiB
partial cache.

The branch intentionally removes alternatives that were slower, unvalidated, or
not part of the measured path. Ordinary PyTorch optimizers are the primary
optimization interface; `BatchedLBFGS` is retained for row-wise genewise polish.
