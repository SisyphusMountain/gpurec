# GMRES Apply-A Fixed-M Mask-Safe Ladder

Date: `2026-06-05 22:41 Europe/Paris`

Repository commit:

```text
5d63743f0be5e718d4fd515d57e7fb3d97f16bb1
```

The worktree had local GMRES implementation edits when this benchmark was
recorded.

Command pattern:

```bash
PYTHONPATH="$PWD" python -u benchmarks/large_dataset_capacity/profile_hogenom_gmres_backward.py \
  --self-loop-solver gmres_fixed \
  --gmres-iters M \
  --warmup 1 \
  --output-json benchmarks/large_dataset_capacity/output/gmres_applya_fixed_masksafe_ladder_20260605_224134/run_mM.json
```

Reference gradient:

```text
Neumann=512: [-4.9285149028026485, -2.377774221053495, 0.8579814490324738]
```

Results:

| Fixed m | Elapsed s | Total backward iterations | Relative L2 error | Relative inf error | Max residual |
|---:|---:|---:|---:|---:|---:|
| 8 | `0.175573` | `544` | `2.285849e-03` | `1.967334e-03` | `7.262364e-04` |
| 10 | `0.208197` | `680` | `6.588901e-06` | `6.380555e-06` | `3.848858e-06` |
| 12 | `0.247367` | `816` | `1.074764e-07` | `1.004131e-07` | `4.185615e-08` |
| 16 | `0.343459` | `1088` | `3.260914e-12` | `2.439352e-12` | `2.573028e-13` |

Conclusion:

Fixed `m=10` is the best current speed/accuracy point for this family. It is
faster than adaptive GMRES max-10 because it avoids per-iteration least-squares
solves and residual checks, even though it uses more self-loop matvecs.
