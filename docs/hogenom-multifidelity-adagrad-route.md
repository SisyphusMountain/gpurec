# HOGENOM Specieswise Multifidelity Adagrad Route

Date: 2026-05-25

This is the first counts-free route found in this branch that starts from
uniform `0.05` D/L/T rates for every species branch and reaches the accepted
HOGENOM specieswise basin inside five minutes. It does not read AleRax event
count summaries or any checkpoint initialized from them.

## Schedule

The route uses exact gpurec full-objective gradients at progressively higher
fixed solver budgets:

| Phase | Solver budget | Optimizer | Steps | LR |
|---|---:|---|---:|---:|
| fixed8 warmup | `E=8, Pi=8, Neumann=8` | Adagrad | 60 | 1.0 |
| fixed16 bridge | `E=16, Pi=16, Neumann=16` | Adagrad, reset state | 35 | 0.5 |
| fixed32 repair | `E=32, Pi=32, Neumann=32` | Adagrad, reset state | 30 | 0.5 |
| final check | `E=128, Pi=128, Neumann=128` | loss-only validation | 1 | n/a |

Verified reproducer:

```bash
python scripts/benchmark_hogenom_specieswise_multifidelity_adagrad.py \
  --out-dir /tmp/gpurec_hogenom_multifidelity_adagrad_route_verify_20260525
```

Verified output:

| Metric | Value |
|---|---:|
| wall time | `259.769763608987s` |
| fixed128 NLL | `526785.875` bits |
| fixed128 validation time | `4.581551495008171s` |
| families | `1055` |
| species | `1325` |
| batches | `5` |

## Discovery Run

The debug run that established the schedule was:

`/tmp/gpurec_hogenom_adagrad8_60_16bridge_32_budget_route`

It included extra loss-only monitoring during the phases and still finished
below five minutes:

| Metric | Value |
|---|---:|
| fixed32 repair step 24 | `526801.125` bits at `272.7s` |
| fixed32 repair step 29 | `526787.938` bits at `289.3s` |
| fixed64 validation | `526788.125` bits at `291.6s` |
| fixed128 validation | `526788.125` bits at `296.2s` |

The verified scripted fixed128 value is below the previous relaxed target used
in local route benchmarks (`526822.875`) and within ten bits of the best
currently observed HOGENOM specieswise objective in this checkout. The scripted
reproducer avoids the debug run's repeated monitoring checks, which is why it
finishes about 36 seconds faster than the discovery run.

## Notes

- The fixed8-only endpoint is not acceptable: a 160-step fixed8 Adagrad run
  validated at fixed128 `528318.8125` despite a lower fixed16 loss.
- Resetting Adagrad state at each solver-budget increase was important. Keeping
  the fixed8 accumulator into fixed32 slowed the repair trajectory.
- A direct fixed32 Adagrad run from uniform 0.05 was too slow to enter the basin
  under the five-minute budget.

## Adaptive Schedule

The script also supports an adaptive mode that avoids hard-coding per-phase
step counts:

```bash
python scripts/benchmark_hogenom_specieswise_multifidelity_adagrad.py \
  --schedule-mode adaptive \
  --out-dir /tmp/gpurec_hogenom_adaptive_multifidelity_adagrad_default_20260525
```

Adaptive mode validates fixed8 and fixed16 phases at fixed32, promotes when the
validation improvement rate stalls, restores the best validated theta before
promotion, and runs the fixed32 phase until the wall-time guard leaves room for
final fixed128 validation.

Default adaptive HOGENOM result:

| Metric | Value |
|---|---:|
| wall time | `297.7560245550121s` |
| fixed128 NLL | `526777.3125` bits |
| fixed8 chosen steps | `70` |
| fixed16 chosen steps | `50` |
| fixed32 chosen steps | `22` |
| fixed8 stop reason | `validation_stall` |
| fixed16 stop reason | `validation_stall` |
| fixed32 stop reason | `wall_budget` |

The adaptive result is slower than the hand-scheduled replay but better in
likelihood, and it chose phase lengths from validation behavior rather than
from manually supplied phase step counts.
