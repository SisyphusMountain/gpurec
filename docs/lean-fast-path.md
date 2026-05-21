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

Current blocker status after the benchmark instrumentation refresh: no valid
current full 1000-family timed run exists.  Windowed preflight is a setup
diagnostic only; it reports `performance_evidence 0` and must not justify
deleting self-loop backends, active-mask pruning modes, environment flags, or
scheduler policies.  `ENV-01`, `SCHED-01`, `BWD-01`, and `BWD-02` remain
blocked until the full benchmark produces timed performance evidence.

The branch intentionally removes alternatives that were slower, unvalidated, or
not part of the measured path. Ordinary PyTorch optimizers are the primary
optimization interface; `BatchedLBFGS` is retained for row-wise genewise polish.
