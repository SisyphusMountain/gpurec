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
- A fixed replay can be forced to start with fixed4 by passing
  `--fixed-initial-budget 4`.  On 2026-05-26, the direct fixed4->8->16->32
  replay with `40/60/35/30` steps took `290.6722433550167s` and validated at
  fixed128 `526990.625` bits.  That is slower and worse than the fixed8-first
  fixed replay above, so fixed mode keeps the fixed8 start by default.
- Split Pi warm starts are also not a replacement for the fixed8-first
  replay on HOGENOM.  On 2026-05-26, the production
  `adagrad-restarts` split schedule `8/4:1.0:60,16:0.5:35,32:0.5:30`
  took `242.26219764497364s` and validated at fixed128
  `527272.875` bits.  The more aggressive `8/4:1.0:60,16/8:0.5:35,32/16:0.5:30`
  schedule took `191.90608406998217s`, but validated at fixed128
  `535334.9375` bits; its best checkpoint still validated at
  `534355.5` bits.  Starting Pi at `4` saves time, but the HOGENOM
  basin found by the original tied fixed8 warmup is materially better.
- After adding opt-in phase-loss promotion to the production workflow, a short
  smoke run with `8/4:1.0:2,16/8:0.5:2`,
  `adagrad_restart_phase_loss_patience=1`, and fixed16 final validation took
  `15.33s` process wall time (`8.957843975978903s` optimizer elapsed).  Solver
  stats verified the first phase used Pi/Neumann `4`, the second used
  Pi/Neumann `8`, and the final evaluation used fixed16.  This only validates
  the workflow scheduling path; it is not an optimum comparison.

## Adaptive Schedule

The script also supports an adaptive mode that avoids hard-coding per-phase
step counts:

```bash
python scripts/benchmark_hogenom_specieswise_multifidelity_adagrad.py \
  --schedule-mode adaptive \
  --out-dir /tmp/gpurec_hogenom_adaptive_multifidelity_adagrad_default_20260525
```

Adaptive mode now prepends a fixed4 phase before fixed8.  The fixed4 phase is
validated at fixed16, fixed8 and fixed16 are validated at fixed32, each
promotion restores the best validated theta, and the fixed32 phase runs until
the wall-time guard leaves room for final fixed128 validation.

For comparison, the previous fixed8-start adaptive result was:

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

That fixed8-start adaptive result was slower than the hand-scheduled replay but
better in likelihood, and it chose phase lengths from validation behavior rather
than from manually supplied phase step counts.

With the current defaults, the fixed4-start adaptive route still stops too early
under a strict five-minute wall: it chooses `40` fixed4 steps, `40` fixed8
steps, and `34` fixed16 steps, never reaching fixed32 repair.  That run took
`298.23166375397705s` and validated at fixed128 `527104.5` bits, about
`327.1875` bits worse than the previous fixed8 adaptive result.

With the wall guard relaxed so the fixed4-start schedule can reach fixed32
repair:

```bash
python scripts/benchmark_hogenom_specieswise_multifidelity_adagrad.py \
  --schedule-mode adaptive \
  --adaptive-max-wall-s 420 \
  --out-dir /tmp/gpurec_hogenom_adaptive_multifidelity_fixed4_420_20260525
```

Fixed4-start adaptive HOGENOM result:

| Metric | Value |
|---|---:|
| wall time | `421.1917021870031s` |
| fixed128 NLL | `526736.75` bits |
| fixed4 chosen steps | `40` |
| fixed8 chosen steps | `40` |
| fixed16 chosen steps | `50` |
| fixed32 chosen steps | `60` |
| fixed4 stop reason | `validation_stall` |
| fixed8 stop reason | `validation_stall` |
| fixed16 stop reason | `validation_stall` |
| fixed32 stop reason | `wall_budget` |

This is `40.5625` bits better than the earlier fixed8 adaptive result and
`49.125` bits better than the verified fixed-length replay, but it uses about
`162s` more wall time than the fixed-length replay.
