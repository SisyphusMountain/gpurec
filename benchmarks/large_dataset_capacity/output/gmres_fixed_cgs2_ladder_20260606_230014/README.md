# GMRES Fixed CGS2 Reference Ladder

Date: `2026-06-05 23:00 CEST`

Implementation commit:

```text
01a0faa90856a9aa730b21957baac59e0084717c
```

Command:

```bash
PYTHONPATH="$PWD" python -u benchmarks/large_dataset_capacity/hogenom_gmres_neumann_family_experiment.py \
  --gmres-solver gmres_fixed \
  --neumann-terms 32 \
  --gmres-iters 8,10,12 \
  --output-json benchmarks/large_dataset_capacity/output/gmres_fixed_cgs2_ladder_20260606_230014/run_result.json
```

Reference gradient:

```text
Neumann=512: [-4.92851490280253, -2.3777742210535737, 0.8579814490324713]
```

Results:

| Solver | Iterations | Total Backward Iterations | Relative L2 Error | Elapsed |
|---|---:|---:|---:|---:|
| Neumann | 32 | `2176` | `3.459027e-05` | `5.406 s` |
| GMRES fixed CGS2 | 8 | `544` | `2.285849e-03` | `5.469 s` |
| GMRES fixed CGS2 | 10 | `680` | `6.588901e-06` | `5.450 s` |
| GMRES fixed CGS2 | 12 | `816` | `1.074765e-07` | `5.450 s` |

The end-to-end `run_gradient` timing includes the forward fixed-point solve, so
it is not sensitive to this backward-only optimization. Use
`profile_hogenom_gmres_backward.py` for backward-only wall time.
