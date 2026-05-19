# AleRax ScaledValue Follow-Up Optimization Results

Date: 2026-05-11

## Setup

Benchmark command shape:

```sh
AleRax/build/bin/alerax \
  -s tests/data/test_trees_1000/sp.nwk \
  -f /tmp/gpurec_alerax_profile/family_0000.txt \
  -p /tmp/gpurec_alerax_profile/out_NAME \
  -r UndatedDTL --model-parametrization GLOBAL \
  --d 1 --l 1 --t 1 \
  --fix-rates --species-tree-search SKIP -g 0
```

The family manifest points to `tests/data/test_trees_1000/g_0000.nwk`.
All runs below produced the same final likelihood:

```text
ll=-4248.87
```

## Changes Applied

1. Gated the expensive `debugLogAbsDiff()` calls in
   `AleRax/src/ale/UndatedDTLMultiModel.hpp`.
   The log-difference convergence metrics are now computed only when
   `ALERAX_DEBUG_DUMP_DIR` is set.

2. Added bucketized `ScaledValue` root/category summation in
   `AleRax/src/ale/MultiModel.hpp`.
   The generic `double` path keeps a direct running sum.

3. Added `ALERAX_FORCE_HIGH_PRECISION=1` in
   `AleRax/src/ale/AleEvaluator.cpp`.
   This is an explicit fast path for families known to require
   `ScaledValue`; it skips the initial failed `double` evaluator.

The prior L2 bucketed transfer/extinction summation remains in
`AleRax/src/ale/UndatedDTLMultiModel.hpp`, with `tileSize = 512`.

## Timing Results

| Variant | Runs | Wall time |
|---|---:|---:|
| Original pre-fix `ScaledValue` | 2 | `11.56-11.59s` |
| Committed scalar arithmetic fix | smoke | `5.32s` |
| L2 bucketed summation before this pass | 5 | avg `5.19s` |
| Follow-up default behavior | 5 | avg `4.12s`, range `4.10-4.16s` |
| Follow-up with `ALERAX_FORCE_HIGH_PRECISION=1` | 5 | avg `3.01s`, range `2.91-3.34s`, median `2.94s` |

Relative to the previous L2 bucketed version, default behavior improved by
about `20.6%` (`5.19s -> 4.12s`).

When high precision is known up front and the initial `double` pass is skipped,
the improvement is about `42.0%` versus the previous L2 bucketed version
(`5.19s -> 3.01s`), or about `74.0%` versus the original pre-fix timing
(`11.58s -> 3.01s`).

## Profiling Effect

Fresh `perf record` after the follow-up default run:

| Hotspot | Approx. profile weight | Notes |
|---|---:|---|
| `UndatedDTLMultiModel<ScaledValue>::updateCLV` | `51.5%` | Still the main target. Remaining work is mostly recurrence logic and `ScaledValue` arithmetic. |
| `UndatedDTLMultiModel<double>::updateCLV` | `26.1%` | This is the initial low-precision attempt that underflows on this family. `ALERAX_FORCE_HIGH_PRECISION=1` removes it. |
| `ScaledValue::operator+=` | `13.1%` | Still visible after the scalar and bucketed summation improvements. |
| `log` / `__ieee754_log_fma` | no longer in the top default profile | Gating debug convergence metrics removed the prior unconditional log overhead. |

The large default improvement is primarily from not computing debug log-diff
metrics when debug dumping is disabled. The root/category bucketed summation is
kept because it matches the `ScaledValue` batch-sum strategy, but its standalone
effect was not isolated in this pass.

## Debug Check

I also ran one debug-dump smoke test with:

```sh
ALERAX_DEBUG_DUMP_DIR=/tmp/gpurec_alerax_profile/debug_check
ALERAX_DEBUG_DUMP_PREFIX=followup_debug
```

It completed with the same likelihood and wrote the expected dump files. That
debug run took about `15.7s`, which is expected because it intentionally emits
large diagnostic tables and computes convergence log differences.

## Remaining Opportunities

The profile is now dominated by `updateCLV`. The next meaningful optimization
would be more invasive: split `computeProbability()` into a no-sampling
likelihood-only path, so the hot likelihood recurrence does not carry the many
`recCell` sampling checks and event writes. That should be measured separately
because it duplicates a sensitive probability recurrence.
