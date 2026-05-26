# gpurec

`gpurec` provides GPU-accelerated PyTorch reconciliation models and workflow
tooling for AleRax-style family inputs, checkpointed optimization, and
stochastic RecPhyloXML sampling.

## Runtime Surface

- `GeneReconModel` for `mode="global"`, `mode="specieswise"`, and
  `mode="genewise"`.
- `UniformChunkedReconModel` for large global/uniform datasets, with public
  chunk metadata for inspecting resident chunk planning.
- `gpurec.workflow` production runners for AleRax-style family inputs,
  checkpointed optimization, convergence diagnostics, and stochastic
  backtracking.
- `gpurec` CLI entry point with `config-template`, `validate-config`,
  `optimize`, `summary-info`, `checkpoint-info`, `sample`, `run`, and
  `backtrack-check` commands.
- Standard PyTorch optimizers over `model.theta`, including `torch.optim.Adam`.
- `gpurec.optimization.BatchedLBFGS` for row-wise genewise polishing.
- The optimized uniform CUDA forward/backward kernels used by the 1000-tree
  benchmark.

Performance-pruning history and retained benchmark context live in
[`docs/lean-fast-path.md`](docs/lean-fast-path.md); the README focuses on the
supported runtime surface.

## Installation

Supported Python versions are Python 3.10-3.12, matching
`requires-python = ">=3.10,<3.13"` in the project metadata.

For a source checkout:

```bash
pip install .
```

For development:

```bash
pip install -e ".[dev]"
```

The CUDA kernels import Triton directly, so Triton is a core dependency rather
than an optional extra.  Install a PyTorch build that matches the local CUDA
runtime before installing `gpurec`.  CPU preprocessing is implemented by the
Rust `crates/gpurec-preprocess` extension; source-checkout runs need a Rust
toolchain unless `GPUREC_PREPROCESS_NATIVE_LIB` points at a prebuilt extension.

For the checkout-local HOGENOM experiment scripts:

```bash
pip install -e ".[hogenom,dev]"
```

## Basic Optimization

```python
import torch
from gpurec import GeneReconModel

model = GeneReconModel.from_trees(
    species_tree="sp.nwk",
    gene_trees=["g_0.nwk", "g_1.nwk"],
    mode="genewise",          # also: "global", "specieswise"
    device="cuda",
    dtype=torch.float32,
    fixed_iters_Pi=6,
    neumann_terms=6,
)

opt = torch.optim.Adam([model.theta], lr=0.03)
for _ in range(20):
    opt.zero_grad(set_to_none=True)
    loss = model()
    loss.backward()
    opt.step()
    model.clamp_theta_(min_rate=2.0**-30, max_rate=2.0)
```

For direct `from_trees` inputs, gene-tree leaf labels are mapped to species by
the legacy prefix fallback: `Species_gene` maps to species `Species`, and a leaf
without `_` maps to the full leaf label.  Use AleRax family files with `mapping`
entries, `UniformChunkedReconModel(..., leaf_species_maps=...)`, or
the narrow low-level `GeneDataset(..., leaf_species_maps=...)` exception when
gene labels do not follow that prefix convention.  `GeneDataset` is retained
for this explicit preprocessing/mapping use; the rest of `gpurec.core` should
be treated as unstable implementation surface unless separately documented.

The retained preprocessing parser supports a deliberately small Newick
subset: nested trees with unquoted labels, optional internal labels, optional
numeric branch lengths, and ordinary whitespace.  Branch lengths are ignored.
When a branch length is present, the numeric text must immediately follow `:`.
The final semicolon is optional for a single tree or final gene-tree record.
Labels cannot rely on quotes, escaping, comments, NHX or BEAST-style metadata,
or embedded `:`, `,`, `(`, `)`, or `;` delimiters.  A species-tree file must
contain exactly one rooted binary tree.  Each gene-tree file may contain one or
more semicolon-delimited records; all records supplied for one family are
amalgamated into that family's CCP.  Gene multifurcations are right-binarized,
while unary gene nodes and non-binary species nodes are rejected.

`GeneReconModel.nll_per_family()` and `GeneReconModel.full_nll_per_family()`
are genewise-only: they return one independent NLL per family and are the
public surface for row-wise optimizers.  In `global` or `specieswise` mode, use
`model(reduce="per_family")` under `torch.no_grad()` only as a diagnostic
shared-theta breakdown; independent per-family gradients are not defined there.

Parameter sharing uses unambiguous model theta shapes: `global` uses `[3]`,
`specieswise` uses `[S, 3]`, and `genewise` uses `[G, 3]`.  The model's
internal normalizers convert genewise scalar event vectors to `[G, 1]` before
the retained DTS kernels.  Direct callers should avoid bare `[G]` DTS parameter
vectors when `G == S`: use `[]` for a scalar, `[S]` for a shared species vector
only on direct forward calls without backward parity requirements, `[G, 1]` for
family scalar rows, and `[G, S]` for family/species rows.  The retained direct
DTS forward helper treats a one-dimensional length-`S` tensor as shared
species-indexed, while the retained backward helper with `family_idx` treats a
one-dimensional tensor as family-indexed.

`model.materialize_batches()` builds every resident batch static state and
returns a copy of the batch metadata list, which is useful before diagnostics or
solver reconfiguration that should touch every batch.  `model.full_loss_for_theta(theta)`
streams all resident batches with an explicit theta tensor that matches the
active sharing-mode shape, model device, and model dtype; differentiable probes
use the gradient-producing streaming path, while calls made under
`torch.no_grad()` use the loss-only streaming path.

For large global/uniform datasets, `UniformChunkedReconModel.loss_and_grad()`
returns `(loss, grad, stats)` for direct stochastic or sampled-chunk optimizers.
The stats dictionary includes selected chunk/family counts, timing fields,
`grad_norm`, reduction metadata, and E-adjoint solve telemetry:
`e_adjoint_method`, `e_adjoint_iterations`, `e_adjoint_rel_res`, and
`e_adjoint_success`.
`UniformChunkedReconModel.nll_per_family(chunk_indices=...)` is a no-grad
global/uniform diagnostic that returns one shared-theta NLL per selected family
after chunk filtering; it does not define independent per-family gradients.

With lazy preprocessing or resident-batch prefetching,
`model.configure_solver_iterations()` updates the model defaults and resident
batch static states that are already built.  It does not cancel or rewrite
pending background prefetch work.  Configure solver iteration controls before
scheduling lazy prefetch, or call `model.materialize_batches()` and configure
again when all resident batches should share the new controls.

For genewise row-wise polishing:

```python
from gpurec.optimization import BatchedLBFGS

opt = BatchedLBFGS([model.theta], lr=1.0)
```

## Python Workflow API

The CLI is the supported entry point for production runs, and the same workflow
objects are also available as top-level Python shortcuts:

```python
from gpurec import RunConfig, SamplingConfig, optimize, sample

run = RunConfig.from_json("run.json")
result = optimize(run)
if result.sampling_checkpoint is None:
    raise RuntimeError(f"optimization failed: {result.reason}")

sampling = SamplingConfig(checkpoint=result.sampling_checkpoint, samples=100)
sample_result = sample(sampling)
```

For direct imports from the workflow package, `gpurec.workflow` exports the same
`RunConfig`, `SamplingConfig`, `OptimizationRunner`, `SamplingRunner`,
`OptimizationResult`, `SamplingResult`, and `optimize`/`sample` functions.
`OptimizationResult` includes the family/species/batch counts, selected
sampling checkpoint for usable runs, objective, gradient route, rate
parameterization, batch/solver route, optimizer-specific route fields,
configured steps, effective optimizer step cap, and final-check solver budgets
reported in `summary.json`.

Top-level backtracking helpers are also available for lower-level sampling and
validation workflows:

```python
from gpurec import (
    ensure_backtracking_available,
    export_backtracking_input,
    recphyloxml_event_counts,
    sample_backtracking_summaries,
    sample_recphyloxml,
    sample_recphyloxmls,
    sample_recphyloxmls_to_dir,
)
```

The `sample_recphyloxml*` helpers accept `backend="auto"`, `"native"`, or
`"cli"`; `auto` uses the in-process native Rust extension unless
`backtrack_binary` or `GPUREC_BACKTRACK_BIN` selects the CLI/binary path.
`sample_backtracking_summaries()` is native-only.  The `gpurec sample` and
`gpurec run` commands use the binary configuration documented below.

Top-level entropy helpers are available for analytical reconciliation entropy:

```python
from gpurec import compute_reconciliation_entropy, reconciliation_entropy_from_payload

entropy = compute_reconciliation_entropy(model, family_index=0, mode="both")
```

`compute_reconciliation_entropy()` works from a solved `GeneReconModel` family.
`reconciliation_entropy_from_payload()` works from an `export_backtracking_input()`
payload plus matching species-tree topology arrays.  The `collapsed` result
matches the distribution sampled by the stochastic backtracker, while
`expanded` also includes hidden extinction histories represented by `E` and
`Ebar`.

## Production AleRax-Style Workflow

The production workflow accepts an AleRax `[FAMILIES]` file and a species tree.
It defaults to genewise D/T/L parameters, writes resumable checkpoints, logs
optimization diagnostics, and can sample RecPhyloXML reconciliation scenarios.
The optimized likelihood path currently requires CUDA.
History JSONL is recorded for every optimizer step; `log_every` and
`--log-every` only throttle console progress prints.
The `--config` option accepts a flat JSON `RunConfig`; Hydra-style YAML
configs should be converted to JSON or passed as explicit CLI flags.  Relative
paths in JSON configs are resolved from the config file's directory; relative
paths passed as explicit CLI flags are resolved from the current working
directory.  For the maintained field-by-field config and CLI option reference,
see [`docs/run-config-reference.md`](docs/run-config-reference.md).
Use `gpurec validate-config --config ...` to check JSON/CLI config values,
input paths, AleRax family records, mapping files, and referenced gene-tree
files without constructing the CUDA likelihood model.
The output includes the resolved optimizer, effective optimizer step cap,
batch planning, solver budgets, and
optimizer-specific defaults such as the specieswise restart schedule and
genewise Hessian-SGD normal-stage solver overrides.
Add `--check-preprocess` when you also want the retained Rust parser to run on
CPU and validate the selected Newick trees plus leaf/species mappings before a
full optimization run.  That heavier preflight also prints
`cuda_backward_ready`; this currently requires more than 256 postorder species
nodes (`S > 256`) for the retained CUDA likelihood/gradient path.  Add
`--require-cuda-backward-ready` to make that readiness check a hard preflight
failure.
For a source checkout or source archive, checked JSON configs and a tiny
AleRax-style fixture live under `examples/`.  They cover the genewise and
specieswise production defaults documented in `examples/README.md`, including
the mode-specific optimizer route knobs and final-check solver budgets that
`validate-config` reports.  The
CLI command shape is:

```bash
gpurec validate-config --config examples/minimal-run-config.json
gpurec validate-config --config examples/specieswise-adagrad-restarts-config.json
gpurec validate-config --config examples/minimal-run-config.json --check-preprocess
gpurec checkpoint-info --checkpoint output_gpurec/checkpoints/latest.pt
```

The checked config is a source-tree config/parser fixture and sets `"device": "cuda"`.
The tiny tree files are portable, but the current optimized
likelihood implementation is not a CPU fallback.  The retained Pi
backward/gradient path currently requires more than 256 postorder species nodes
(`S > 256`), so this tiny fixture is not an end-to-end optimizer smoke until a
small-species backward fallback is restored.

Installed wheels do not install the `examples/` directory as runtime package
data. Use the installed CLI to generate a flat JSON starting point, then edit
the tree paths and mode as needed:

```bash
gpurec config-template --mode genewise --output run.json
gpurec config-template --mode specieswise --output specieswise-run.json
```

The command keeps `"optimizer": "auto"`, so `mode=genewise` resolves to
`hessian-sgd` and `mode=specieswise` resolves to `adagrad-restarts`. You can
use the same stripped, case-normalized mode and optimizer spelling in flat JSON
configs or CLI flags; optimizer underscores are accepted as aliases for the
canonical hyphenated names. You can also copy or adapt a flat JSON config
alongside your own tree files:

```json
{
  "species_tree": "S.tree",
  "families_file": "families.txt",
  "out_dir": "output_gpurec",
  "mode": "genewise",
  "device": "cuda"
}
```

Workflow config aliases are shared between JSON configs and CLI flags.  `dtype`
accepts `float32`/`float64` plus aliases such as `fp32`, `single`, `fp64`,
`double`, `torch.float32`, and `torch.float64`.  `family_chunk_size` accepts a
non-negative integer, with `0`, `all`, `none`, or JSON `null` meaning one
resident batch.  `batch_packing` accepts `sequential`, `clade_first_fit`, and
`depth_first_fit`; aliases include `contiguous`/`input_order`,
`first_fit_decreasing`/`ffd`/`clade_ffd`, and
`depth_ffd`/`critical_path_first_fit`/`wave_first_fit`, with hyphenated forms
accepted by the CLI.  Non-sequential packing requires `clade_budget`.
`small_family_max_leaves` defaults to `0` in the workflow, which disables
leaf-count priority grouping.  Set it to a positive value to plan families with
at most that many leaves before larger families while still respecting the
normal clade budget.

The workflow CLI intentionally supports only float32 and float64.  The direct
`UniformChunkedReconModel` constructor also accepts `torch.bfloat16` as an
experimental CUDA-only path for memory-constrained forward/NLL probes, but
bf16 is not a supported workflow configuration dtype and is not supported by
the retained Pi backward/gradient path.  Do not use bf16 for release smokes,
optimizer checkpoints, or Hessian/second-order diagnostics.

Optimizer modes are selected with `optimizer` in JSON or `--optimizer` on the
CLI. If omitted, `auto` resolves to `hessian-sgd` for `mode=genewise`,
`adagrad-restarts` for `mode=specieswise`, and `adam` for `mode=global`.
Route metadata, summaries, and status lines also report
`mode_default_optimizer` and `uses_mode_default_optimizer` so operators can
tell whether a run is on the production default optimizer for its sharing mode,
plus `uses_production_default_optimizer_settings` and
`production_default_optimizer_setting_mismatches` to show whether the
optimizer-specific settings still match the shipped HOGENOM/`test_trees_1000`
optimizer profile. The full-route verdict fields
`uses_production_default_route` and `production_default_route_mismatches`
combine those optimizer-setting checks with the shipped objective, gradient
route, rate parameterization, and production default basis metadata enforced by
`--require-production-default-route`.
Workflow rate bounds default to `min_rate=2^-30` and `max_rate=2`:

The production optimization guide
[`docs/production-optimization-guide.md`](docs/production-optimization-guide.md)
explains how the likelihood objective, gradient route, solver budgets,
genewise `hessian-sgd` default, specieswise `adagrad-restarts` default, and
HOGENOM/`test_trees_1000` validation gates fit together.
For the complete `RunConfig` field and CLI flag reference, see
[`docs/run-config-reference.md`](docs/run-config-reference.md).
For source data layout, AleRax family syntax, mapping files, path resolution,
and preflight validation, see
[`docs/input-preparation.md`](docs/input-preparation.md).

| Mode | Behavior | Notes |
| --- | --- | --- |
| `auto` | Mode-dependent workflow default. | Uses `hessian-sgd` for `mode=genewise`, `adagrad-restarts` for `mode=specieswise`, and `adam` for `mode=global`. |
| `adam` | Adam optimizer for all configured steps. | Uses `lr` and ordinary PyTorch Adam state. |
| `adagrad` | Adagrad optimizer for all configured steps. | Uses `lr`; retained for long-running comparison runs. |
| `adagrad-restarts` | Specieswise multifidelity Adagrad with state resets. | Requires `mode=specieswise`; this is the specieswise `auto` default. The default schedule is `8:1.0:60,16:0.5:35,32:0.5:30`, meaning fixed `E/Pi/Neumann` budgets of 8, 16, then 32, with Adagrad state reset at each budget increase. The default ladder has `adagrad_restart_total_steps=125`; `validate-config`, checkpoints, and `summary.json` report `optimizer_step_cap=125` and `optimizer_step_cap_reason=adagrad_restart_schedule` when the schedule is the active cap. The run stops when the schedule is complete even if `steps` is larger. The final validation/gradient evaluation uses `adagrad_restart_final_check_iters=128` and reports `final_check_iters_e=128`. Override the phase ladder with `adagrad_restart_schedule` or `--adagrad-restart-schedule`. |
| `projected-sgd` | Projected SGD for all configured steps. | Uses `lr`, records projected gradients at rate bounds, and clamps D/L/T rates to `min_rate`/`max_rate` after every step. |
| `lbfgs` | PyTorch LBFGS for all configured steps. | Uses `lbfgs_lr`, `lbfgs_history_size`, `lbfgs_max_iter`, and `lbfgs_line_search`.  `lbfgs_line_search` is `none` or `strong_wolfe`; LBFGS runtime errors stop the run with a failed status. |
| `adam-lbfgs` | Adam warmup, then LBFGS polishing. | `adam_warmup_steps` controls the phase switch; incompatible resumed optimizer state is discarded when the checkpoint phase differs from the current phase. |
| `projected-lbfgs` | Single-objective projected L-BFGS-B-style polishing. | Uses projected gradients at rate bounds, `lbfgs_lr`, `lbfgs_history_size`, and `lbfgs_max_ls`; Armijo line-search probes use loss-only evaluations before one accepted gradient refresh. If loss stalls while `grad/projected_inf` exceeds `projected_grad_tol`, the workflow reduces the base L-BFGS step size instead of declaring convergence. HOGENOM specieswise convergence sweeps found `--lbfgs-lr 0.4` fastest by wall time and `--lbfgs-lr 0.5` the best tested time/objective tradeoff. |
| `lbfgsb` | Single-objective L-BFGS-B-style polishing. | Uses raw BFGS curvature pairs, a generalized Cauchy point, a free-subspace solve, projected gradients at rate bounds, `lbfgs_lr`, `lbfgs_history_size`, and `lbfgs_max_ls`; Armijo line-search probes use loss-only evaluations before one accepted gradient refresh. If loss stalls while `grad/projected_inf` exceeds `projected_grad_tol`, the workflow keeps the run `not_converged` instead of declaring convergence. |
| `batched-lbfgs` | Row-wise batched L-BFGS-B for genewise runs. | Requires `mode=genewise`; uses per-family NLL/gradient vectors, projected gradients at rate bounds, `lbfgs_lr`, `lbfgs_history_size`, `lbfgs_max_iter`, `lbfgs_max_ls`, and `lbfgs_line_search`. `none` uses internal row-wise Armijo probes; `strong_wolfe` uses a vectorized row-wise port of PyTorch's bracket/zoom line search. |
| `adam-fd-newton` | Short Adam warmup, then finite-difference/quasi-Newton updates for genewise batches. | Requires `mode=genewise`; `fd_adam_warmup_steps` controls per-batch Adam warmup, `fd_hessian_refresh_steps` controls how many rate-bounded Newton steps reuse BFGS-updated row-wise 3x3 Hessians between finite-difference refreshes, `fd_hessian_epsilon` controls refresh probes, and `fd_newton_damping` controls Hessian regularization. Newton trial rates are projected to `min_rate`/`max_rate`; there is no separate log-rate movement cap. |
| `hessian-sgd` | Projected Hessian-conditioned gradient steps for genewise batches. | Requires `mode=genewise`; finite-difference 3x3 row Hessians refresh every `fd_hessian_refresh_steps`, receive BFGS row updates between refreshes, and precondition fresh gradients with step scale `lr`. Warmup uses reduced Pi/Neumann iterations, with a shorter schedule for very large active batches; very large batches that plateau during warmup skip redundant full-stage optimizer rows while still caching canonical full-solver values for final evaluation. Normal full-stage steps can use `hessian_sgd_normal_fixed_iters_pi` and `hessian_sgd_normal_neumann_terms`. Opt-in `hessian_sgd_pi_adjoint_warmstart` can stage accepted Pi-adjoint caches; experimental `pi_fixed_point_relaxation` defaults to `1.0` and accepts non-default positive values only with that warmstart enabled. Steps are projected to rate bounds and skip Armijo loss probes before Adam warmup. |

For `mode=genewise` with `optimizer=batched-lbfgs`, `adam-fd-newton`, or
`hessian-sgd`,
`adaptive_rebatch` can rebuild resident waves for the remaining unconverged
families when the current batch crosses `adaptive_rebatch_fraction`.  The check
uses the post-step projected gradients already produced by the optimizer, and
`adaptive_rebatch_check_interval` controls how often that aggregate threshold
is tested.

Optimizer-specific controls are scoped to the optimizer that consumes them.
`hessian_sgd_normal_*`, validation, and Pi-adjoint warm-start controls require
genewise `hessian-sgd`; non-default `adagrad_restart_*` controls require
specieswise `adagrad-restarts`.

`adagrad_restart_schedule` is validated before model loading.  Each phase must
use `budget:lr:steps` or `E/Pi[/Neumann]:lr:steps`; tied budgets and `Pi`
budgets must be positive even integers, split `E` and Neumann budgets must be
positive integers, learning rates must be positive finite numbers, and step
counts must be positive integers.  Later phases must not decrease
`fixed_iters_E`, `fixed_iters_Pi`, or `neumann_terms`; same-budget LR restarts
are allowed.  `adagrad_restart_final_check_iters` must be zero to disable the
final specieswise validation pass or a positive even integer.

```bash
gpurec optimize \
  --species-tree S.tree \
  --families-file families.txt \
  --out-dir output_gpurec \
  --preprocess-cpu-cores 8 \
  --mode genewise \
  --device cuda
```

`--preprocess-cpu-cores` sets the worker thread count for CPU preprocessing.
When it is omitted, Rust preprocessing uses its runtime default.

Main outputs include:

- `checkpoints/latest.pt` and `checkpoints/best.pt`, metadata-bearing
  checkpoints for resume, sampling, restore workflows, and route inspection
- `optimization_history.csv` and `history.jsonl`
- `rates_final.tsv`, `theta_final.pt`, `summary.json` with status, the selected
  sampling checkpoint when available, final/best NLL and log-likelihood, and
  the effective optimizer/batch/solver route
- `per_fam_likelihoods.tsv` for genewise runs

The `gpurec optimize` status line and the optimization portion of
`gpurec run` also print the resolved `mode`, `optimizer`, mode default
optimizer, whether the optimizer matches that default, whether the
optimizer-specific settings match the shipped production route, any setting
mismatches, `uses_production_default_route`,
`production_default_route_mismatches`, family/species/batch counts, batch
packing, family chunk size, clade budget, solver iteration budgets, objective,
gradient route, rate parameterization, production default basis,
optimizer-specific route fields,
`configured_steps`, `optimizer_step_cap`, `optimizer_step_cap_reason`,
`final_check_iters`, `final_check_iters_e`, `steps_completed`, `elapsed_s`,
`best_step`,
`sampling_checkpoint`,
`final_nll_bits`, `final_log_likelihood_bits`, `best_nll_bits`, and
`best_log_likelihood_bits`, plus `final_grad_inf`,
`final_projected_grad_inf`, `final_check_status`, `final_check_source`,
`final_check_reason`,
`final_check_fallback_clade_budget`, `final_check_loss_abs_delta_bits`,
`final_check_grad_max_abs_delta`, `final_check_grad_rel_inf_delta`,
`final_solver_e_adjoint_failed_batches`,
`final_solver_e_adjoint_success_batches`, and
`final_solver_e_adjoint_rel_res_max` for quick terminal triage.
Add `--require-converged` to `gpurec optimize` when a
shell pipeline should print the same optimization status line and then exit
nonzero unless the run reached `status=converged`. Add
`--require-final-check-ok` when the same pipeline should also fail unless the
final high-fidelity likelihood/gradient validation reports
`final_check_status=ok`. Add `--require-mode-default-optimizer` to
`gpurec validate-config`, `gpurec optimize`, or `gpurec run` when production
automation should fail unless the resolved optimizer is the mode default
(`hessian-sgd` for genewise, `adagrad-restarts` for specieswise). Add
`--require-production-default-route` when automation should also fail on
stale likelihood/gradient route metadata or non-default optimizer-specific
settings such as a changed Hessian-SGD refresh budget or a truncated
specieswise restart ladder.

See [`docs/output-artifacts.md`](docs/output-artifacts.md) for the output
artifact contract, including the normalized config snapshot, history fields,
checkpoint contents, rate-table columns, genewise per-family likelihoods, and
sampling files.
Use `gpurec summary-info --summary output_gpurec/summary.json` to inspect the
final summary status, likelihood/gradient diagnostics, and route metadata
without opening JSON by hand. Add `--require-converged` when a shell pipeline
or workflow manager should fail unless the summary status is `converged`, and
add `--require-final-check-ok` when it should also require
`final_check_status=ok`. Add `--require-mode-default-optimizer` to fail unless
the summary proves the run used the production default optimizer for its mode;
add `--require-production-default-route` when it should also require the
shipped likelihood/gradient route metadata and optimizer-specific settings
reported by `uses_production_default_route`.
Use `gpurec checkpoint-info --checkpoint output_gpurec/checkpoints/latest.pt`
to inspect checkpoint progress, status, optimizer route, and last
likelihood/gradient diagnostics without constructing the CUDA likelihood model.
When the checkpoint's last row contains final validation metrics, the command
also prints `last_final_check_iters`, `last_final_check_iters_e`,
`last_final_check_status`, and the matching source/reason and loss/gradient
delta fields. Add `--require-final-check-ok` when automation
should fail unless that checkpoint row reports `optimizer/final_check_status=ok`;
add `--require-mode-default-optimizer` to require a checkpoint route whose
optimizer matches the production default for its mode, or
`--require-production-default-route` to require the full shipped production
route, including the likelihood/gradient contract fields.

History rows include aggregate `solver/*` telemetry when the model reports
solver statistics.  E-adjoint nonconvergence is diagnostic-only: the retained
BiCGSTAB solve returns its best iterate with `success=False`, optimization
continues unless the objective or gradient becomes nonfinite, and history rows
surface `solver/e_adjoint_failed_batches` plus relative-residual and iteration
summaries for monitoring. Opt-in Pi-adjoint warmstart runs also report
`solver/pi_adjoint_residual_absmax_max` and
`solver/pi_adjoint_residual_relmax_max`, computed from one extra fixed-point
self-loop application, so warm-budget experiments can monitor whether the
implicit-gradient solve is actually converging. These experiments can set
`pi_fixed_point_relaxation` for cached Pi-adjoint updates; the default `1.0`
keeps the standard fixed-point update.

`theta_final.pt` is a raw tensor export for inspection or custom analysis.  It
does not carry run configuration, family ordering, or species ordering metadata;
use `checkpoints/best.pt` or `checkpoints/latest.pt` whenever a workflow needs
to restore parameters into a model or sample reconciliation scenarios.
Python tooling that needs checkpoint configuration metadata should read
`load_checkpoint(path)["config"]` from `gpurec.workflow.checkpoint` and pass it
to `RunConfig.from_dict(...)`; tooling that only needs the resolved
likelihood/gradient/optimizer route can read
`load_checkpoint(path)["route_metadata"]` from checkpoints written by current
versions, and no separate public `load_checkpoint_config` helper is supported.
The lower-level `gpurec.workflow.checkpoint` submodule explicitly supports
`save_checkpoint`, `load_checkpoint`, `restore_model_theta`,
`validate_checkpoint_model_compatibility`, and `CHECKPOINT_VERSION` for
advanced tooling that inspects or restores workflow checkpoints directly.  These
helpers are not top-level `gpurec.workflow` shortcuts; prefer `optimize`,
`sample`, `RunConfig`, and `SamplingConfig` unless code specifically needs the
versioned checkpoint payload.

Version-1 workflow checkpoints carry identity metadata for safe restore:
`family_names`, `species_names`, and config identity fields `species_tree`,
`families_file`, `mode`, `start`, and `max_families`. `load_checkpoint()`
requires those fields to be present and validates the name-list metadata.
Current checkpoints also carry `route_metadata` with the resolved objective,
gradient route, parameterization, optimizer, mode default optimizer, and solver
route.
`validate_checkpoint_model_compatibility()` first validates the stored config
with `RunConfig.from_dict(...)`, then compares identity and route metadata with
the active `RunConfig` and rebuilt model before `restore_model_theta()` copies
parameters. Path identity fields are normalized during comparison. The
checkpoint loader remains a lower-level payload reader and does not reconstruct
a full `RunConfig`.

Resume starts from the checkpoint `next_step`.  If `next_step` already equals
the configured `steps`, `gpurec optimize --resume-from ...` performs only the
final evaluation/artifact refresh, writes a fresh `latest.pt`, and returns the
same `not_converged`/`max_steps` status used by ordinary max-step exhaustion.
Increase `steps` beyond the checkpoint `next_step` to run additional optimizer
steps.

To sample scenarios from the best checkpoint:

```bash
gpurec sample \
  --checkpoint output_gpurec/checkpoints/best.pt \
  --samples 100 \
  --sample-out-dir output_gpurec
```

Sampling writes per-sample RecPhyloXML files and event-count files under
`output_gpurec/reconciliations/all/`.  Aggregate summaries live under
`output_gpurec/reconciliations/`, including `event_counts.tsv`,
`totalSpeciesEventCounts.txt`, and `totalTransfers.txt`.
`event_counts.tsv` is tab-separated.  `totalSpeciesEventCounts.txt` is the
AleRax-compatible comma-space text format with one species label followed by
event totals.  `totalTransfers.txt` uses whitespace-separated source species,
destination species, and average transfer count.  Aggregate values are averaged
over the requested sample count for each retained family, not over all families
in the original checkpoint.
The sampled RecPhyloXML subset expected by gpurec contains `recGeneTree` blocks
with `clade` nodes, `eventsRec` event containers, and the event tags
`speciation`, `duplication`, `branchingOut`, `transferBack`, `loss`, and
`leaf`.  gpurec-generated sample XML files are expected to contain one
`recGeneTree` per file; the shared event-count traversal can still read
multiple `recGeneTree` blocks in compatibility inputs.  For aggregate species
counts, origination is a file-level event: the first speciation reached by the
shared traversal is counted as origination, matching gpurec's one-tree-per-file
sample output while keeping multi-tree compatibility inputs deterministic.
Use `--family-start` and `--sample-max-families` to sample a family window,
`--seed` for reproducible stochastic backtracking, and `--max-events` to cap
pathological samples.
Add `--require-mode-default-optimizer` to `gpurec sample` when standalone
sampling automation should fail unless the checkpoint route used the production
default optimizer for its mode; use `--require-production-default-route` when
the checkpoint must also prove `uses_production_default_route=true` for the
shipped likelihood/gradient route metadata and optimizer-specific settings.
Successful sampling reruns replace prior gpurec-generated reconciliation
artifacts in the target output directory, including generated files outside a
requested window; use a separate `--sample-out-dir` to keep multiple windows.

To optimize and sample in one supported CLI workflow, run `gpurec run` with the
same optimization config plus sampling options:

```bash
gpurec run \
  --config run.json \
  --samples 100 \
  --sample-out-dir output_gpurec
```

Sampling flags such as `--samples`, `--family-start`, `--sample-max-families`,
`--max-events`, and `--backtrack-binary` are accepted on `gpurec run`.
`gpurec run` does not accept `--checkpoint`; it samples from the checkpoint
reported by the optimizer, falling back to `checkpoints/best.pt` or
`checkpoints/latest.pt` when needed, and exits without sampling if optimization
fails.  Failed optimization still prints the optimization status line before
`gpurec run` exits. Add `--require-converged` when `gpurec run` should print
the optimization status and exit before sampling unless the run reached
`status=converged`; add `--require-final-check-ok` when it should also skip
sampling unless `final_check_status=ok`; add `--require-mode-default-optimizer`
when it should reject non-default optimizer routes before optimization or
sampling; add `--require-production-default-route` when changed optimizer
settings or stale likelihood/gradient route metadata should also stop the run
before optimization or sampling. When
sampling succeeds, the final status line also reports
`sampled_families`, `samples`, `xml`, and `sample_out_dir`. Use
`gpurec sample --checkpoint ...` to sample an existing run.

### Sampling Binary Setup

`gpurec sample` and the sampling phase of `gpurec run` use the Rust
backtracking binary.  Wheels intentionally do not ship that binary or the Rust
crate sources, so installed environments should provide a compiled binary
through `GPUREC_BACKTRACK_BIN` or `--backtrack-binary`.  In a wheel-only
install, point gpurec at a prebuilt binary produced by your deployment or build
process:

```bash
export GPUREC_BACKTRACK_BIN="/opt/gpurec/bin/gpurec-backtrack"
gpurec backtrack-check
gpurec sample --checkpoint output_gpurec/checkpoints/best.pt --samples 100
```

For a source checkout or unpacked source archive, build that binary with Cargo:

```bash
cargo build --locked --release --manifest-path crates/gpurec-backtrack/Cargo.toml
export GPUREC_BACKTRACK_BIN="$PWD/crates/gpurec-backtrack/target/release/gpurec-backtrack"
gpurec backtrack-check
gpurec sample --checkpoint output_gpurec/checkpoints/best.pt --samples 100
```

The same `GPUREC_BACKTRACK_BIN` environment variable or `--backtrack-binary`
flag applies to `gpurec run` when it samples after optimization.
Use `gpurec backtrack-check` to validate the binary or source-tree Cargo
fallback by running its `--help` path without loading a checkpoint.

The automatic `cargo run` fallback works from a source checkout or unpacked
source archive.  It requires a Rust toolchain and fetches the pinned `rustree`
git dependency declared by `crates/gpurec-backtrack/Cargo.toml`; otherwise use a
prebuilt binary.

Python callers that use `sample_recphyloxml*` with `backend="native"` or
`backend="auto"` can point `GPUREC_BACKTRACK_NATIVE_LIB` at a prebuilt PyO3
backtracking extension.  Without that variable, source checkouts build
`crates/gpurec-backtrack` with Cargo when the default release library is
missing.  The native extension is for Python helper calls; CLI sampling still
uses `GPUREC_BACKTRACK_BIN`, `--backtrack-binary`, or the source-tree Cargo
binary fallback.

### Runtime Environment Flags

Most production settings should be expressed through JSON config files or CLI
flags.  The `GPUREC_*` environment variables are retained for binary discovery,
compatibility guards, memory-policy margins, and Rust preprocessing discovery.
Kernel routing, scheduler selection, cache locations, and launch tuning are not
supported environment contracts; use explicit API, CLI, or profiling arguments
for those controls.  Backward CUDA/Triton prototype selectors have been removed
from the production environment surface; the retained backward path is
Triton-only.

| Variable | Scope |
| --- | --- |
| `GPUREC_PREPROCESS_NATIVE_LIB` | Optional path to a prebuilt native Rust preprocessing extension. |
| `GPUREC_PREPROCESS_BIN` | Optional path to the Rust preprocessing CLI used by the subprocess adapter and profiling helpers. |
| `GPUREC_BACKTRACK_BIN` | Path to the Rust backtracking binary used by `gpurec sample`, `gpurec run`, and `gpurec backtrack-check`. |
| `GPUREC_BACKTRACK_NATIVE_LIB` | Optional path to a prebuilt native Rust backtracking extension used by Python helper calls with `backend="native"` or native `auto` resolution. |
| `GPUREC_ALERAX_COMPAT` | Compatibility guard; differentiable model optimization supports only unset or `0`. |
| `GPUREC_MEMORY_POLICY_FRACTION`, `GPUREC_MEMORY_POLICY_RESERVE_GIB` | GPU memory-budget margins used by uniform chunk planning. |

## HOGENOM Workflows

The installed `gpurec` CLI is the supported general workflow.  The HOGENOM
scripts under `scripts/` are legacy checkout-local experiment launchers and
diagnostics for a local, untracked HOGENOM benchmark layout.  The HOGENOM data
under `tests/data/HOGENOM/...` is intentionally not distributed with the package.
The optimization launchers default to those local paths; pass explicit
`--species-tree`, `--families-file`, and `--out-dir` values to those launchers
for other datasets.  The one-pass profiler is tied to
that local HOGENOM layout and exposes profiling and batch-control flags instead
of dataset path overrides.

| Task | Command | Notes |
| --- | --- | --- |
| General installed workflow | `gpurec config-template`, `gpurec validate-config`, `gpurec optimize`, `gpurec summary-info`, `gpurec checkpoint-info`, `gpurec sample`, `gpurec run` | Uses flat JSON for `--config`; Hydra YAML is not accepted by the main CLI. `config-template` prints or writes installed JSON templates for mode-specific defaults. `validate-config` is a CPU-safe path/reference preflight; `summary-info` and `checkpoint-info` are CPU-safe artifact inspection commands. None of those inspection/preflight commands construct the CUDA likelihood model. |
| Sampling binary preflight | `gpurec backtrack-check` | Validates `GPUREC_BACKTRACK_BIN`, `--backtrack-binary`, or the source-tree Cargo fallback without loading a checkpoint. |
| Legacy HOGENOM W&B wrapper | `python scripts/optimize_hogenom_ccp_wandb.py` | Checkout-local compatibility wrapper with argparse flags, checkpoints, plots, and optional W&B logging. |
| Hydra HOGENOM run | `python scripts/optimize_hogenom_ccp_hydra.py` | Uses `configs/hogenom_ccp_wandb.yaml`, a checkout-local full experiment config, and Hydra override syntax; see `configs/README.md` for config ownership. |
| Checkpoint rate export | `python scripts/export_hogenom_rates_from_checkpoint.py` | Exports D/T/L rates from a HOGENOM optimization checkpoint. |
| One-pass profiling | `python scripts/profile_hogenom_ccp_pass.py` | Profiles full, streamed, active-batch, or largest-batch forward/backward passes on the local HOGENOM layout. |
| Specieswise multifidelity Adagrad | `python scripts/benchmark_hogenom_specieswise_multifidelity_adagrad.py` | Counts-free route from uniform 0.05 rates: fixed8 Adagrad warmup, fixed16 bridge, fixed32 repair, and fixed128 validation; `--schedule-mode adaptive` chooses phase lengths from higher-budget validation stalls. |
| Specieswise route benchmark | `python scripts/benchmark_hogenom_specieswise_e2e.py` | Summarizes the local accepted optimization route, time-to-target, and manual stages with unknown elapsed time. |
| Specieswise pulse benchmark | `python scripts/benchmark_hogenom_specieswise_pulses.py` | Checkout-local benchmark for short projected-gradient pulse probes from a HOGENOM specieswise checkpoint; defaults to fixed, non-adaptive solver iterations for validation. |
| Specieswise tail replay | `python scripts/replay_hogenom_specieswise_tail.py` | Replays and times the accepted post-SGD pulse tail, then validates the final theta at fixed128. |
| Historical notebooks | `notebooks/` | Checkout-local HOGENOM analyses; see `notebooks/README.md` before using or migrating them. |

Optional script dependencies are intentionally separate from the core package
and are grouped under the `hogenom` extra.  The R plotting helper also needs
its CRAN/Bioconductor plotting packages.

Minimal HOGENOM smoke when CUDA memory allows:

```bash
python scripts/optimize_hogenom_ccp_wandb.py \
  --max-families 1 \
  --steps 1 \
  --wandb-mode disabled \
  --no-timestamped-out-dir
```

## Source-Checkout Performance Check

The full-dataset benchmark harness lives under `profiling/` and is intended
for source checkouts, not installed wheels.  See `profiling/README.md` for the
supported profiling entrypoints, local-data assumptions, and artifact retention
policy:

```bash
python profiling/bench_uniform_forward_backward_pipeline.py \
  --dataset /path/to/test_trees_1000 \
  --fams 1000 \
  --family-chunk-size auto \
  --max-wave-size auto \
  --fixed-iters 6 \
  --neumann-terms 3 \
  --warmups 1 \
  --reps 3 \
  --strict-optimized-kernels
```

Expected on the measured RTX 4090 setup is roughly:

```text
forward_median_ms  ~2445
backward_median_ms ~3532
total_median_ms    ~5979
generic_self_loop_calls 0
strict_optimized_verdict pass
```

## Documentation

See `docs/README.md` for the current documentation map.  It separates current
operating notes from historical performance and research logs.
For source data layout and preflight validation, see
[`docs/input-preparation.md`](docs/input-preparation.md).
For run triage, see [`docs/troubleshooting.md`](docs/troubleshooting.md).
