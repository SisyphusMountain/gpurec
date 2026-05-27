# Production Likelihood Optimization Strategy

This note turns the HOGENOM and `test_trees_1000` optimization results into a
production strategy for likelihood optimization. It is intentionally a policy,
not a replay script: HOGENOM and `test_trees_1000` are calibration datasets, not
the full production distribution.

The goal is to reach a good negative log-likelihood from the standard uniform
`0.05` D/L/T initialization quickly, while keeping enough validation that the
optimizer is not just exploiting low-fidelity solver artifacts.

## Default Policy

Use `optimizer=auto` unless a dataset has already passed the validation gates
below.

| Mode | Production default | Why |
|---|---|---|
| `genewise` | `hessian-sgd` | Genewise rows are independent per family, so row-wise projected Hessian-conditioned steps use the model structure directly and preserve canonical final evaluation. |
| `specieswise` | `adagrad-restarts` | Specieswise parameters share one full objective, and the retained HOGENOM route reaches the accepted basin from uniform `0.05` with a conservative multifidelity Adagrad ladder. |
| `global` | `adam` | The surface is tiny and shared; this is the conservative baseline. |

The production strategy is asymmetric because the parameter sharing is
asymmetric. Genewise optimization is a collection of row-wise problems.
Specieswise optimization is one coupled high-dimensional bounded problem, where
the inner E/Pi solver fidelity can change the basin.

## Lessons From The Benchmarks

HOGENOM specieswise:

- The retained counts-free route is tied-budget `adagrad-restarts`:
  `8:1.0:60,16:0.5:35,32:0.5:30`, with Adagrad state reset at each fidelity
  increase and fixed128 validation.
- Starting with fixed4 or split Pi budgets saved time but reached worse basins
  unless much more high-fidelity repair was allowed.
- L-BFGS-B can improve some HOGENOM specieswise checkpoints, but the measured
  route did not become the retained default because the boundary/conditioning
  region is stiff and KKT convergence remains expensive.

`test_trees_1000` specieswise:

- The generated large-S shape benefits from a short split-fidelity prefix:
  E7/Pi4 warmup, a fixed8 bridge, one E16/Pi8 repair row, then ordinary
  L-BFGS-B.
- More Pi/E repair is not automatically better. The best TT1000 trajectories
  came from handing off to end-to-end L-BFGS-B early, not from optimizing the
  inner Pi/E iterations harder.
- Late L-BFGS-B fallback and line-search knobs are path-sensitive. They provide
  Pareto points, but they are not stable enough to become global defaults.

Genewise HOGENOM and TT1000:

- The accepted genewise route is `hessian-sgd`. Faster shortcuts that globally
  capped warmup line search or lowered Pi/Neumann too aggressively reached worse
  final likelihoods.
- For genewise, the right adaptation target is family difficulty, not a global
  number of optimizer or solver steps.

## Genewise Production Strategy

Use:

```bash
gpurec optimize \
  --mode genewise \
  --optimizer auto \
  --species-tree S.tree \
  --families-file families.txt \
  --out-dir output_gpurec \
  --device cuda
```

This resolves to `hessian-sgd`.

Operational rules:

- Keep the default row-wise `hessian-sgd` route for production.
- Treat `batched-lbfgs` and `adam-fd-newton` as comparison or diagnostic
  optimizers, not the default production path.
- Keep canonical final evaluation enabled. Genewise shortcuts are acceptable
  only if the final full-solver per-family likelihoods remain in the accepted
  range.
- Do not globally lower Pi/Neumann budgets or line-search probes just because a
  single dataset gets faster. HOGENOM showed that such shortcuts can change the
  final basin by hundreds of bits.
- Prefer adaptive row policies:
  - track per-family accepted step sizes and line-search probe counts;
  - reuse each family's recent accepted alpha as the next starting alpha;
  - split easy, medium, hard, and boundary families into separate resident
    batches when enough telemetry supports it;
  - escalate solver fidelity per family or per difficulty bin when cheap and
    canonical decisions disagree.

The production direction for genewise is therefore:

```text
row-wise Hessian-SGD default
  -> add per-family telemetry
  -> rebatch by observed optimization difficulty
  -> adapt line-search starts per family
  -> use cheap Pi/Neumann only while it agrees with canonical decisions
```

Do not replace this with a fixed "N warmup steps" or "K line-search probes"
rule across all datasets.

## Specieswise Production Strategy

Use the conservative default first:

```bash
gpurec optimize \
  --mode specieswise \
  --optimizer auto \
  --species-tree S.tree \
  --families-file families.txt \
  --out-dir output_gpurec \
  --device cuda
```

This resolves to `adagrad-restarts` with:

```text
8:1.0:60,16:0.5:35,32:0.5:30
```

Retain this as the general production default because it is the route validated
on HOGENOM from uniform `0.05`.

Use `adagrad-restarts-lbfgsb` only after a dataset-level validation run shows
the default ladder is over-spending or entering a weaker basin. The TT1000
result gives the shape of the candidate, not a universal preset:

```text
cheap/split Adagrad prefix
  -> one short higher-fidelity repair
  -> L-BFGS-B tail
  -> fixed high-fidelity validation
```

For a new specieswise dataset, the production-safe calibration procedure is:

1. Run the default `adagrad-restarts` route and keep fixed high-fidelity final
   validation as the reference.
2. Probe cheaper split E/Pi settings only in short calibration runs. Promote
   them only when their loss ordering and gradient direction agree with the
   canonical budget.
3. If the cheap prefix reaches a comparable basin, try a composite
   `adagrad-restarts-lbfgsb` route with phase caps and objective-stall
   promotion. Use objective improvement and validation, not manually tuned phase
   lengths, as the stopping signal.
4. Keep L-BFGS-B tail controls conservative:
   - use `loss_stop_projected_grad_gate=false` when the production target is
     wall-time-to-likelihood rather than KKT polishing;
   - use a staged loss-change schedule instead of manual checkpoint resumes;
   - disable or tightly budget coordinate fallback unless a validation run shows
     it consistently improves the final validated likelihood.
5. Promote the composite route only if it beats the default on both wall time
   and validated likelihood for that dataset family.

The TT1000 best-known route is useful evidence for when to try the composite
path. It is not evidence that all specieswise production runs should start at
E7/Pi4 or use a one-row E16/Pi8 repair. HOGENOM showed the opposite: tied
fixed8/16/32 was the safer basin-entry path.

## Stopping And Validation

Use likelihood validation as the production contract. Projected gradient is
still diagnostic, especially near bounds, but the recent specieswise routes show
that driving projected gradient to a small value can cost a lot of wall time
after objective improvements have become float32-resolution-scale.

Recommended stopping policy:

- Genewise: stop rows by projected-gradient and row convergence criteria, then
  run canonical final per-family evaluation.
- Specieswise Adagrad default: stop by the scheduled phases unless an adaptive
  phase-promotion policy has been validated; always keep high-fidelity final
  validation.
- Specieswise composite: use objective-stall schedules for the L-BFGS-B tail
  and validate the final theta at the configured final-check budget.

Never promote a timing-only improvement if:

- final high-fidelity NLL is materially worse;
- final-check loss deltas increase outside the accepted range;
- cheap and canonical solver decisions disagree during calibration;
- the change only works on HOGENOM or only works on `test_trees_1000`.

## What To Tune Per Dataset

Safe to tune after validation:

- specieswise Adagrad phase caps and phase-loss promotion patience;
- split E/Pi warmup budgets, if cheap and canonical decisions agree;
- L-BFGS-B loss-stall thresholds and final validation budget;
- genewise adaptive rebatching and per-family line-search starting alpha.

Risky as global defaults:

- event-count or previous-run warm starts, unless production explicitly accepts
  a different initialization contract;
- fixed low Pi/Neumann budgets for all rows or phases;
- globally shortened genewise line-search caps;
- specieswise coordinate fallback and schedule-forced fallback;
- a TT1000-specific one-row repair schedule applied to HOGENOM-like data.

## Promotion Checklist

Before changing production defaults, require:

- a clean run from uniform `0.05` unless the feature is explicitly a warm-start
  feature;
- final high-fidelity validation on the same theta;
- no material likelihood regression versus the retained default;
- wall-time improvement on the target dataset family;
- at least one cross-check on the other benchmark axis: HOGENOM for specieswise
  basin quality, and `test_trees_1000` for large resident-layout behavior;
- history and summary artifacts kept for the route that is being promoted.

In short: keep `hessian-sgd` as the genewise production default, keep
`adagrad-restarts` as the specieswise production default, and use the TT1000
composite Adagrad-plus-L-BFGS-B shape as an opt-in calibrated route when a new
dataset proves that early handoff beats the conservative HOGENOM ladder.
