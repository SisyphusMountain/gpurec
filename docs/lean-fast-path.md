# Lean Fast Path

This branch is anchored to the measured `test_trees_1000` path:

- `pibar_mode="uniform"`
- full global/specieswise/genewise wave layout
- parent-reduced DTS forward
- fused uniform forward self-loop
- Proposal 0 uniform backward self-loop
- fused DTS backward accumulation
- tree-based cross-Pibar VJP
- memory policy default: family chunk size `25`, max wave size `8192`

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

The branch intentionally removes alternatives that were slower, unvalidated, or
not part of the measured path. Ordinary PyTorch optimizers are the primary
optimization interface; `BatchedLBFGS` is retained for row-wise genewise polish.
