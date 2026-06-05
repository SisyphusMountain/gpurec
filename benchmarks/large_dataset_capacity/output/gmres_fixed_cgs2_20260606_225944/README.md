# GMRES Fixed CGS2 Backward-Only Run

Date: `2026-06-05 22:59 CEST`

Implementation commit:

```text
01a0faa90856a9aa730b21957baac59e0084717c
```

Command:

```bash
PYTHONPATH="$PWD" python -u benchmarks/large_dataset_capacity/profile_hogenom_gmres_backward.py \
  --self-loop-solver gmres_fixed \
  --gmres-iters 10 \
  --warmup 1 \
  --output-json benchmarks/large_dataset_capacity/output/gmres_fixed_cgs2_20260606_225944/run_result.json
```

Run summary:

```text
family: CLU_000680_20_4_C
solver: gmres_fixed
fixed m: 10
wave_count: 68
total_backward_iterations: 680
elapsed_s: 0.16014486318454146
max_rel_res: 3.848858102743519e-06
gradient: [-4.928546349462685, -2.377755721941347, 0.85798053849603]
```

Compared with the previous fixed-m Apply-A result (`0.208197 s`), the batched
CGS2 Arnoldi path is about `23%` faster with the same VJP count and gradient.
