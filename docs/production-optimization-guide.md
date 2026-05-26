# Production Optimization Guide

This guide describes the supported production route for AleRax-style
optimization. It ties together the likelihood objective, gradient surface,
solver fidelity controls, default optimizers, and artifacts that operators
should inspect when running `gpurec optimize` or `gpurec run`.

## Objective And Parameters

The workflow minimizes negative log-likelihood in bits. History rows report
`likelihood/data_nll_bits`; the corresponding log-likelihood is
`likelihood/log_likelihood_bits`. The final run summary repeats those views as
`final_nll_bits`/`final_log_likelihood_bits` and
`best_nll_bits`/`best_log_likelihood_bits`.
`summary.json`, current checkpoints, and `gpurec validate-config` also report
the stable route fields `objective=negative_log_likelihood_bits`,
`gradient_route=implicit_first_order_adjoint`,
`rate_parameterization=base2_log_dlt_rates`, and
`production_default_basis=hogenom_and_test_trees_1000` so exported artifacts
carry the likelihood, gradient, parameterization, and benchmark-basis contract.

`theta` stores base-2 log rates for duplication, loss, and transfer. The public
rate table writes columns in D/T/L order as probabilities/rates plus the raw
theta values. Workflow updates clamp rates to `min_rate` and `max_rate`, which
default to `2^-30` and `2`.

The sharing mode defines the optimization surface:

| Mode | Theta shape | Production use |
|---|---:|---|
| `global` | `[3]` | One D/L/T row for the full dataset. The `auto` optimizer resolves to `adam`. |
| `specieswise` | `[S, 3]` | One D/L/T row per species branch. The `auto` optimizer resolves to `adagrad-restarts`. |
| `genewise` | `[G, 3]` | One D/L/T row per family. The `auto` optimizer resolves to `hessian-sgd`. |

Genewise `nll_per_family()` and `full_nll_per_family()` are the row-wise
likelihood APIs used by row-wise optimizers. In global and specieswise modes,
per-family NLLs are diagnostics for a shared-theta model, not independent
gradient rows.

## Likelihood And Gradient Route

Production runs build a `GeneReconModel` from an AleRax `[FAMILIES]` file and
species tree. Resident batches keep static wave layouts on the GPU and stream
the full objective through `model.full_loss()`.

Each gradient evaluation:

1. Extracts D/L/T rates from the active `theta`.
2. Solves extinction probabilities `E`.
3. Runs the Pi fixed-point/self-loop dynamic program over resident waves.
4. Computes the root negative log-likelihood.
5. Computes the implicit gradient through Pi and E adjoints.
6. Reduces gradients back to the configured theta shape.

The route is an analytical first-order gradient path. It is not a second-order
autograd program through the fixed-point solvers; Hessian-aware optimizers use
finite-difference or BFGS-style curvature models around this first-order
surface.

Solver fidelity is controlled by `fixed_iters_e`, `fixed_iters_pi`, and
`neumann_terms`. The workflow can run warmup stages at cheaper budgets, then
promote to the configured full stage or to a final validation budget. History
rows surface aggregate `solver/*` telemetry, and E-adjoint nonconvergence is
diagnostic unless the objective or gradient becomes nonfinite.

## Default Optimizer Routes

`optimizer=auto` is deliberately mode-dependent:

| Mode | Default | Why this route is retained |
|---|---|---|
| `genewise` | `hessian-sgd` | Genewise rows are independent, so the runner can optimize active batches with projected, Hessian-conditioned row steps while caching canonical full-solver values for final artifacts. |
| `specieswise` | `adagrad-restarts` | Specieswise rates share a single full objective; the HOGENOM route reached the accepted basin fastest with multifidelity Adagrad and explicit state resets. |
| `global` | `adam` | The global surface is small and shared. Adam remains the conservative default. |

### Genewise `hessian-sgd`

`hessian-sgd` works on active genewise batches. It starts each active batch with
the solver warmup budget, refreshes 3x3 finite-difference row Hessians every
`fd_hessian_refresh_steps`, applies BFGS row updates between refreshes, and
uses projected steps inside the workflow rate bounds. The recorded
`grad/projected_inf` is the stopping metric that matters at the bounds.

Large active batches use a short warmup route and can skip redundant full-stage
optimizer rows after a plateau, while still evaluating and caching canonical
full-solver loss/gradient values for the final evaluation. Optional
`adaptive_rebatch` rebuilds resident batches from remaining unconverged
families when enough rows in the active batch have converged.

Use this route for production genewise runs unless a specific comparison needs
`batched-lbfgs` or `adam-fd-newton`.

### Specieswise `adagrad-restarts`

`adagrad-restarts` is specieswise-only. Its default schedule is:

```text
8:1.0:60,16:0.5:35,32:0.5:30
```

Each entry is `solver_budget:learning_rate:steps`. Split entries are also
accepted as `E/Pi[/Neumann]:learning_rate:steps`, for example
`8/4:1.0:60` to start with `fixed_iters_E=8`, `fixed_iters_Pi=4`, and
`neumann_terms=4`. The runner resets Adagrad state at each budget change and
records `optimizer/adagrad_restart_*` fields in history, including explicit
E/Pi/Neumann budgets. The final validation uses
`adagrad_restart_final_check_iters=128` by default.

The default is based on the retained counts-free HOGENOM route: uniform `0.05`
D/L/T initialization, no AleRax event-count checkpoint, fixed budgets of 8, 16,
then 32, and a fixed128 validation check. It is the specieswise `auto` route
because direct high-fidelity specieswise optimization was slower to enter the
same basin.

Override the ladder with `adagrad_restart_schedule` or
`--adagrad-restart-schedule` only when a dataset-specific validation run shows
that the default under- or over-spends early fidelity.

## Dataset-Grounded Defaults

Two local benchmark families anchor the current defaults:

| Dataset | What it informs | Production rule |
|---|---|---|
| HOGENOM | Specieswise end-to-end optimization from uniform rates. | Keep the `adagrad-restarts` ladder as the specieswise default unless a new route beats fixed128 NLL and wall time without using event-count initialization. |
| `tests/data/test_trees_1000` | Cold resident likelihood construction and first-pass timing on a generated large-S shape. | Keep retained Rust preprocessing, `clade_first_fit` for that shape, lazy resident prefetch, and low-fidelity first passes only when the caller can use the approximate likelihood before promotion. |

The two datasets stress different parts of the system. HOGENOM is the
specieswise optimizer gate. `test_trees_1000` is the resident likelihood and
construction gate. A production default should not move just because it improves
one of those axes while degrading likelihood/gradient parity or end-to-end
optimization on the other.

## Operating Checklist

For a normal genewise production run:

```bash
gpurec config-template --mode genewise --output run.json

gpurec validate-config \
  --species-tree S.tree \
  --families-file families.txt \
  --out-dir output_gpurec \
  --mode genewise \
  --device cuda

gpurec optimize \
  --species-tree S.tree \
  --families-file families.txt \
  --out-dir output_gpurec \
  --mode genewise \
  --device cuda
```

For specieswise, set `--mode specieswise` and let `auto` choose
`adagrad-restarts`. Installed users can start from
`gpurec config-template --mode specieswise --output specieswise-run.json`; that
template keeps `optimizer=auto` and writes the default
`adagrad_restart_schedule` and fixed128 final validation fields explicitly.
`validate-config` checks the flat JSON/CLI config, selected AleRax family
records, mapping files, and referenced gene-tree files without CUDA or
preprocessing. Its summary prints the resolved optimizer, batch planning,
solver budgets, effective `final_check_iters`, and optimizer-specific defaults
such as the specieswise restart schedule. It is a preflight for path and parser
issues, not a likelihood or gradient correctness check. Add `--check-preprocess`
for a heavier CPU
preprocessing pass that uses the retained Rust parser to validate selected
Newick trees and leaf/species mappings before optimization.

Inspect these outputs first:

- `history.jsonl`: every optimizer step, including phase, solver stage,
  objective, gradient norms, projected gradients, and solver telemetry.
- `summary.json`: final status/reason, elapsed seconds, final objective, and
  best objective metadata as both NLL and log-likelihood in bits.
- `checkpoints/best.pt` and `checkpoints/latest.pt`: resumable model,
  optimizer state, and effective route metadata.
- `rates_final.tsv`: final D/T/L rates and theta values.
- `per_fam_likelihoods.tsv`: genewise-only final per-family NLLs.

The complete optimization, checkpoint, rate-table, and sampling output contract
is maintained in `docs/output-artifacts.md`.

Resume uses checkpoint `next_step`. If `next_step` already equals configured
`steps`, the runner performs the final evaluation/artifact refresh only.

## Verification Gate

Before promoting a likelihood, gradient, solver, or optimizer change as
production behavior, verify the appropriate layer:

- Unit/workflow tests for config parsing, checkpoint/resume, optimizer phase
  transitions, and artifact behavior.
- CUDA likelihood/gradient parity tests for any change in kernels, batching, or
  solver semantics.
- HOGENOM specieswise end-to-end checks for `adagrad-restarts` changes.
- `test_trees_1000` resident likelihood timing checks for construction,
  batching, or low-fidelity first-pass changes.

Do not accept timing-only improvements that change likelihood or gradient
semantics outside the established fp32 tolerance for that route.
