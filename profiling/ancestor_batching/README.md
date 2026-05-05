# Uniform Backward Ancestor-Batching Profiling

Reusable profiling harness for `docs/uniform-backward-ancestor-batching-experiments.md`.

Default run:

```bash
python profiling/ancestor_batching/run_profiles.py
```

This creates a timestamped directory under `profiling/ancestor_batching/artifacts/`
with:

- `commands.sh`: exact commands and env overrides used for every run;
- `timing_summary.csv` and `parity_summary.csv`: warmed CUDA-event timings and
  baseline/prototype loss-gradient deltas;
- `nsys/*.nsys-rep`, `nsys/*.sqlite`, and `nsys/*_kernel_buckets.csv`: Nsight
  Systems captures and kernel bucket summaries;
- `ncu/*.ncu-rep`, `ncu/*.csv`, and `ncu_summary.csv`: Nsight Compute captures
  and selected representative-kernel counters.

The artifact directory is intentionally ignored by git. Commit the scripts, not
large `.nsys-rep` or `.ncu-rep` files.

Useful narrower runs:

```bash
python profiling/ancestor_batching/run_profiles.py --phases timing
python profiling/ancestor_batching/run_profiles.py --phases nsys,ncu --variants baseline,proposal5_cuda_nosplit_tree
python profiling/ancestor_batching/bench_uniform_backward.py --fams 50 --reps 9 --warmups 5
```

Proposal 0-4 variants are registered but skipped automatically until their
`GPUREC_*` flags appear in `gpurec/`. Proposal 5 currently maps to:

- `proposal5_cuda_nosplit_self`:
  `GPUREC_CUDA_SELF_LOOP_NOSPLIT=1`,
  `GPUREC_CUDA_SELF_LOOP_NOSPLIT_CORRECTION=self`
- `proposal5_cuda_nosplit_tree`:
  `GPUREC_CUDA_SELF_LOOP_NOSPLIT=1`,
  `GPUREC_CUDA_SELF_LOOP_NOSPLIT_CORRECTION=tree`

