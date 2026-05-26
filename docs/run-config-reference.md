# RunConfig And CLI Reference

`RunConfig` is the flat JSON and Python dataclass contract used by
`gpurec optimize`, `gpurec run`, and `gpurec validate-config`. JSON files use
snake_case field names. CLI flags use the same names with underscores changed
to hyphens. For example, `fixed_iters_pi` is `--fixed-iters-pi`.

`--config` accepts a JSON object, not Hydra YAML. Relative paths inside JSON
configs are resolved relative to the config file. Relative paths passed as CLI
flags are resolved relative to the current working directory. Explicit CLI
flags override fields loaded from `--config`. Unknown JSON fields are rejected
by `RunConfig.from_dict(...)` before model construction. Mode strings are
stripped and case-normalized before `optimizer=auto` resolves the production
default optimizer. Optimizer strings are stripped and case-normalized, and
underscore aliases such as `hessian_sgd` and `adagrad_restarts` are converted
to the canonical hyphenated names. The same normalization rules are applied to
flat JSON configs and explicit CLI flags.

`gpurec config-template --mode genewise` writes the genewise production starter
with `optimizer=auto`, which resolves to `hessian-sgd`. The specieswise
template command, `gpurec config-template --mode specieswise`, writes the
specieswise starter, where `optimizer=auto` resolves to `adagrad-restarts`.
`gpurec config-template --mode global` remains available for shared-rate
diagnostic configs, but it resolves to the mode-default `adam` optimizer and
will not pass `--require-production-default-route`.
The template command includes the fields normally edited for that mode; this
reference lists the complete `RunConfig` surface.

## Required Inputs

| Field | CLI flag | Contract |
|---|---|---|
| `species_tree` | `--species-tree` | Species tree Newick file. Required unless loaded from `--config`. |
| `families_file` | `--families-file` | AleRax `[FAMILIES]` file. Required unless loaded from `--config`. |
| `out_dir` | `--out-dir` | Output directory for checkpoints, history, summaries, rates, and sampling artifacts. Required unless loaded from `--config`. |
| `mode` | `--mode` | Parameter-sharing mode: `genewise`, `specieswise`, or `global`. Default is `genewise`. |
| `device` | `--device` | Torch device string. The production likelihood/gradient path currently expects CUDA for real optimization. |
| `dtype` | `--dtype` | Floating dtype, normalized to `float32` or `float64`; aliases include `fp32`, `single`, `fp64`, and `double`. |
| `start` | `--start` | Zero-based first family index to load. |
| `max_families` | `--max-families` | Optional positive cap on loaded families. |
| `preprocess_cpu_cores` | `--preprocess-cpu-cores` | Optional CPU worker count for Rust preprocessing. |

## Resident Batching

| Field | CLI flag | Contract |
|---|---|---|
| `family_chunk_size` | `--family-chunk-size` | Families per resident chunk; `0`, `all`, `none`, or `null` means one resident batch. |
| `clade_budget` | `--clade-budget` | Positive clade budget for non-sequential batch packing; use `null` only with `batch_packing=sequential`. |
| `batch_packing` | `--batch-packing` | Resident-batch packing policy. Supported normalized values are `sequential`, `clade_first_fit`, and `depth_first_fit`. |
| `max_wave_size` | `--max-wave-size` | Optional positive cap on clades scheduled into one resident wave. |
| `small_family_max_leaves` | `--small-family-max-leaves` | Families with at most this many leaves can be planned first; `0` disables the split. |
| `adaptive_rebatch` | `--adaptive-rebatch` / `--no-adaptive-rebatch` | Enables adaptive resident-batch rebuilding for supported genewise active-batch optimizers. |
| `adaptive_rebatch_fraction` | `--adaptive-rebatch-fraction` | Fraction threshold for rebuilding around remaining unconverged families. |
| `adaptive_rebatch_check_interval` | `--adaptive-rebatch-check-interval` | Positive optimizer-step interval for adaptive rebatch checks. |
| `adaptive_rebatch_min_remaining_families` | `--adaptive-rebatch-min-remaining-families` | Positive lower bound on remaining families before adaptive rebatching can run. |

## Solver Fidelity

| Field | CLI flag | Contract |
|---|---|---|
| `fixed_iters_e` | `--fixed-iters-e` | Optional fixed E iterations. `null` uses adaptive E up to `max_iters_e`. In specieswise runs with `fixed_iters_pi > 16`, E is raised to at least the Pi budget. |
| `max_iters_e` | `--max-iters-e` | Positive maximum adaptive E iterations. |
| `tol_e` | `--tol-e` | Non-negative adaptive E convergence tolerance. |
| `fixed_iters_pi` | `--fixed-iters-pi` | Positive even Pi iteration budget. |
| `neumann_terms` | `--neumann-terms` | Positive Neumann-series term count for the implicit gradient. |
| `solver_warmup_iters` | `--solver-warmup-iters` | Initial low-fidelity Pi/Neumann budget for supported genewise active-batch optimizers and specieswise runs with larger full budgets; `0` disables warmup. |
| `solver_warmup_loss_patience` | `--solver-warmup-loss-patience` | Non-negative flat-loss patience before genewise active-batch optimizers promote from warmup to full solver budgets. |
| `adaptive_iters` | `--adaptive-iters` / `--no-adaptive-iters` | Enables adaptive E/Pi iteration stopping. |
| `adaptive_neumann_terms` | `--adaptive-neumann-terms` / `--no-adaptive-neumann-terms` | Disabled compatibility flag. Enabling it is rejected because the adaptive Neumann path recomputes full gradients at each check and is not part of the supported production optimization route. |
| `final_check_iters` | `--final-check-iters` | Final high-fidelity validation budget for non-`adagrad-restarts` optimizers; `0` disables the final check, otherwise it must be positive and even. |
| `convergence_check_interval` | `--convergence-check-interval` | Positive iteration interval for adaptive solver checks; must be even when `adaptive_iters=true`. |
| `e_logsumexp_tol` | `--e-logsumexp-tol` | Non-negative E logsumexp convergence tolerance. |
| `pi_max_diff_tol` | `--pi-max-diff-tol` | Non-negative Pi max-difference convergence tolerance. |
| `gradient_change_tol` | `--gradient-change-tol` | Non-negative absolute gradient-change tolerance retained for compatibility with the disabled adaptive-Neumann surface. |
| `gradient_change_rtol` | `--gradient-change-rtol` | Non-negative relative gradient-change tolerance retained for compatibility with the disabled adaptive-Neumann surface. |

## Rate Parameterization

| Field | CLI flag | Contract |
|---|---|---|
| `theta_init_d` | `--theta-init-d` | Strictly positive initial duplication rate. |
| `theta_init_l` | `--theta-init-l` | Strictly positive initial loss rate. |
| `theta_init_t` | `--theta-init-t` | Strictly positive initial transfer rate. |
| `min_rate` | `--min-rate` | Strictly positive lower bound for D/L/T rates. |
| `max_rate` | `--max-rate` | Strict upper bound for D/L/T rates; must be greater than `min_rate`. |

Rates are optimized in base-2 log D/L/T space. `theta_final.pt` stores raw
theta values for inspection; `rates_final.tsv` stores final rates and theta
values with labels.

## Optimizer Selection

| Field | CLI flag | Contract |
|---|---|---|
| `optimizer` | `--optimizer` | Optimizer schedule. `auto` resolves to `hessian-sgd` for `mode=genewise`, `adagrad-restarts` for `mode=specieswise`, and `adam` for `mode=global`. |
| `steps` | `--steps` | Positive maximum optimizer-step count. For `adagrad-restarts`, the schedule can impose a smaller effective cap. |
| `lr` | `--lr` | Positive Adam/Adagrad learning rate or `hessian-sgd` preconditioned step scale. |
| `adam_warmup_steps` | `--adam-warmup-steps` | Non-negative Adam warmup length before `adam-lbfgs` polishing. |
| `fd_adam_warmup_steps` | `--fd-adam-warmup-steps` | Non-negative Adam warmup steps per resident batch before Hessian-conditioned genewise updates. |
| `fd_hessian_refresh_steps` | `--fd-hessian-refresh-steps` | Positive step interval between full finite-difference Hessian refreshes for Hessian-conditioned genewise optimizers. |
| `hessian_sgd_normal_fixed_iters_pi` | `--hessian-sgd-normal-fixed-iters-pi` | Optional positive even Pi budget for `hessian-sgd` full-stage steps. Requires genewise `hessian-sgd` when set. |
| `hessian_sgd_normal_neumann_terms` | `--hessian-sgd-normal-neumann-terms` | Optional positive Neumann budget for `hessian-sgd` full-stage steps. Requires genewise `hessian-sgd` when set. |
| `hessian_sgd_pi_adjoint_warmstart` | `--hessian-sgd-pi-adjoint-warmstart` / `--no-hessian-sgd-pi-adjoint-warmstart` | Enables the experimental staged Pi-adjoint warm-start cache. Requires genewise `hessian-sgd`. |
| `pi_fixed_point_relaxation` | `--pi-fixed-point-relaxation` | Positive Pi-adjoint relaxation factor. Non-default values require `hessian_sgd_pi_adjoint_warmstart=true`. |
| `hessian_sgd_validation_interval` | `--hessian-sgd-validation-interval` | Non-negative full-stage cadence for periodic high-budget `hessian-sgd` validation steps; `0` disables periodic validation. |
| `hessian_sgd_validation_fixed_iters_pi` | `--hessian-sgd-validation-fixed-iters-pi` | Optional positive even Pi budget for periodic `hessian-sgd` validation steps; requires a positive validation interval. |
| `hessian_sgd_validation_neumann_terms` | `--hessian-sgd-validation-neumann-terms` | Optional positive Neumann budget for periodic `hessian-sgd` validation steps; requires a positive validation interval. |
| `adagrad_restart_schedule` | `--adagrad-restart-schedule` | Specieswise `adagrad-restarts` phase ladder as `budget:lr:steps` or `E/Pi[/Neumann]:lr:steps`. Later phases must not decrease `fixed_iters_E`, `fixed_iters_Pi`, or `neumann_terms`; same-budget LR restarts are allowed. Non-default values require specieswise `adagrad-restarts`. |
| `adagrad_restart_final_check_iters` | `--adagrad-restart-final-check-iters` | Final specieswise validation budget for `adagrad-restarts`; `0` disables, otherwise positive even. Non-default values require specieswise `adagrad-restarts`. |
| `lbfgs_lr` | `--lbfgs-lr` | Positive base learning rate for L-BFGS style optimizers. |
| `lbfgs_history_size` | `--lbfgs-history-size` | Positive number of curvature pairs retained by L-BFGS style optimizers. |
| `lbfgs_max_iter` | `--lbfgs-max-iter` | Positive L-BFGS inner iteration count per optimizer step. |
| `lbfgs_max_ls` | `--lbfgs-max-ls` | Positive line-search probe cap for L-BFGS style optimizers. |
| `lbfgs_line_search` | `--lbfgs-line-search` | Batched L-BFGS line search mode: `none` or `strong_wolfe`. |
| `fd_hessian_epsilon` | `--fd-hessian-epsilon` | Positive finite-difference epsilon for Hessian-conditioned genewise probes. |
| `fd_newton_damping` | `--fd-newton-damping` | Positive diagonal damping added to finite-difference Hessians. |

Supported explicit optimizers are `adam`, `adagrad`, `projected-sgd`,
`lbfgs`, `adam-lbfgs`, `projected-lbfgs`, `lbfgsb`, `batched-lbfgs`,
`adam-fd-newton`, `hessian-sgd`, and `adagrad-restarts`. `batched-lbfgs`,
`adam-fd-newton`, and `hessian-sgd` require `mode=genewise`.
`adagrad-restarts` requires `mode=specieswise`.

## Stopping, Logging, And Resume

| Field | CLI flag | Contract |
|---|---|---|
| `loss_change_tol` | `--loss-change-tol` | Non-negative small-loss-change threshold. Genewise active-batch optimizers apply it per active family. |
| `loss_patience` | `--loss-patience` | Non-negative consecutive small-loss-change patience. |
| `best_likelihood_patience` | `--best-likelihood-patience` | Non-negative patience for steps without best-likelihood improvement. |
| `best_likelihood_min_delta` | `--best-likelihood-min-delta` | Non-negative improvement required to reset best-likelihood patience. |
| `projected_grad_tol` | `--projected-grad-tol` | Non-negative projected-gradient infinity-norm tolerance for projected optimizers. |
| `projected_lbfgs_min_lr` | `--projected-lbfgs-min-lr` | Positive lower bound for automatic projected-LBFGS base learning-rate backoff. |
| `checkpoint_every` | `--checkpoint-every` | Non-negative checkpoint interval in optimizer steps; `0` disables periodic checkpoints. |
| `log_every` | `--log-every` | Positive console progress interval; `history.jsonl` is still recorded every optimizer step. |
| `resume_from` | `--resume-from` | Optional checkpoint path for resuming optimization state. |

Resume starts at checkpoint `next_step`. The checkpoint config is validated
with `RunConfig.from_dict(...)`, then identity and route metadata are compared
against the active config and rebuilt model before theta is restored. If
`next_step` already equals configured `steps`, `gpurec optimize --resume-from`
performs only final evaluation and artifact refresh, writes a fresh
`latest.pt`, and returns the same `not_converged`/`max_steps` status used by
ordinary max-step exhaustion. Increase `steps` beyond `next_step` to continue
optimization.

Route metadata and status outputs include `mode_default_optimizer` and
`uses_mode_default_optimizer`, making explicit whether the resolved optimizer is
the mode default optimizer for the selected sharing mode. They also include
`uses_production_default_optimizer_settings` and
`production_default_optimizer_setting_mismatches`, which audit whether the
optimizer-specific settings still match the shipped HOGENOM/`test_trees_1000`
optimizer profile. They also include `uses_production_default_route` and
`production_default_route_mismatches`, which combine those optimizer-specific
checks with the shipped objective, gradient route, rate parameterization, and
production default basis metadata enforced by
`--require-production-default-route`.

For automation, `gpurec optimize`, `gpurec run`, and `gpurec summary-info`
support `--require-converged`. Add `--require-final-check-ok` when the command
should also fail unless final high-fidelity likelihood/gradient validation
reports `final_check_status=ok`. Add `--require-mode-default-optimizer` to
preflight, run, standalone sampling, or artifact-inspection commands when
production automation must reject optimizers that do not match the selected
mode default. Add `--require-production-default-route` when stale
likelihood/gradient route metadata or changed optimizer-specific settings should
also fail those gates and be reported in
`production_default_route_mismatches`. When both route gates are requested, the
config route is resolved once and reused for both gates; `validate-config`
prints that same route snapshot in its status line. The strict production-route
gate is limited to the retained genewise `hessian-sgd` and specieswise
`adagrad-restarts` HOGENOM/`test_trees_1000` profiles; `mode=global` remains a
mode-default `adam` route, but it is reported as a production-route `mode`
mismatch.
