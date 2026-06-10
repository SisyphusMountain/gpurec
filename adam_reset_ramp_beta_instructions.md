# Instructions: Run The Missing Adam Reset/Ramp/Beta Experiments

The current `archaea_experiments.md` says Adam/RMSProp/Adafactor-style optimizers should be rerun with optimizer-state resets and LR ramping at Pi/Neumann/precision/GMRES transitions. Those experiments were **not actually run**. Run them now and update the document with real results.

## Goal

Measure whether Adam converges better when:

- optimizer state is reset at each solver-schedule phase boundary,
- LR is ramped back up after each transition,
- Adam beta memory is shortened for a 100-200 step deterministic optimization,
- the final phase is `float64` with backward self-loop `gmres`.

Do not report old default-Adam results as tuned Adam results.

## Baseline Dataset

Use the same 256-family genewise complete-CCP screen first:

```text
mode: genewise
families: 256 smallest eligible families
min leaves: 4
hierarchical EB: enabled
unbounded/unprojected: enabled
backtracking: disabled
final phase: float64 + gmres
```

Use this schedule shape:

```text
40:4:4:float32:neumann:LR32,
40:8:8:float32:neumann:LR32,
40:12:12:float32:neumann:LR32B,
120:16:16:float64:gmres:LR64
```

Use reset/ramp:

```text
--lr-ramp-steps 10
--lr-ramp-start-factor 0.2
```

Do **not** pass `--preserve-optimizer-state-across-phases`.

## Adam Sweep

Run at least these Adam configurations:

| Name | beta1 | beta2 | LR32 | LR32B | LR64 |
|---|---:|---:|---:|---:|---:|
| adam_b09_b099 | 0.9 | 0.99 | 0.03 | 0.03 | 0.003 |
| adam_b05_b09 | 0.5 | 0.90 | 0.01 | 0.005 | 0.001 |
| adam_b05_b095 | 0.5 | 0.95 | 0.01 | 0.005 | 0.001 |
| adam_b07_b095 | 0.7 | 0.95 | 0.02 | 0.01 | 0.002 |

For each run, save a unique output JSON under:

```text
output/alerax_archaea_genewise_adam/
```

Include `reset_ramp` and beta values in the filename.

## Required Command Pattern

Use this pattern, substituting beta/LR/output values:

```bash
python scripts/optimize_alerax_archaea_genewise_adam.py \
  --mode genewise \
  --optimizer adam \
  --adam-beta1 BETA1 \
  --adam-beta2 BETA2 \
  --hierarchical-eb \
  --unbounded-unprojected \
  --max-families 256 \
  --family-order smallest \
  --min-leaves 4 \
  --schedule 40:4:4:float32:neumann:LR32,40:8:8:float32:neumann:LR32,40:12:12:float32:neumann:LR32B,120:16:16:float64:gmres:LR64 \
  --lr-ramp-steps 10 \
  --lr-ramp-start-factor 0.2 \
  --tail-window 25 \
  --backtrack-families 0 \
  --allow-unconverged \
  --output-json output/alerax_archaea_genewise_adam/NAME.json
```

## Verification

For every JSON, extract and report:

- `convergence.converged`
- `convergence.final_loss_bits`
- `convergence.best_loss_bits`
- `convergence.tail_slope_bits_per_step`
- `convergence.final_joint_grad_norm` if present, otherwise final theta/projected grad
- final data loss
- final prior loss
- rate min/max/mean
- elapsed time
- whether every non-first phase has `optimizer_reset_at_phase_start: true`
- whether LR ramping is visible in the first 10 steps of each post-transition phase

## Document Update

Update `archaea_experiments.md` with a new section named:

```text
## Adam Reset/Ramp/Beta Sweep
```

The section must clearly distinguish:

- old default Adam result,
- new reset/ramp beta-tuned Adam results,
- whether any run actually beats or matches Rprop by gradient criterion,
- whether the result is only a 256-family screen or a whole-dataset result.

Remove or rewrite any language that implies the tuned Adam experiments were already done before these JSONs exist.

## If A Tuned Adam Run Works

If one Adam configuration converges cleanly on the 256-family screen, then use it as a candidate for the specieswise all-family continuation from the stabilized strong-prior specieswise checkpoint. Keep that as a separate experiment and do not mix it with the 256-family genewise sweep.
