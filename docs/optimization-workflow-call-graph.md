# gpurec Optimization Workflow Call Graph

Scope: this maps the current production workflow entry points after the CLI,
API, and workflow split. It covers `gpurec validate-config`, `gpurec optimize`,
`gpurec run`, and the Python call `gpurec.workflow.optimize.optimize(config)`.
The supported workflow modes are `genewise`, `specieswise`, and
`global`. In this document, "uniform" means the shared-theta global route
(`mode=global`) unless the direct `UniformChunkedReconModel` API is named
explicitly.

## Entry Points

```mermaid
flowchart TD
  Facade["gpurec.cli.main(argv)<br/>gpurec/cli.py:54"]
  Commands["gpurec._cli_commands.main(argv)<br/>gpurec/_cli_commands.py:959"]
  Parser["build_parser()<br/>gpurec/_cli_commands.py:601"]
  Args["_run_config_from_args(args)<br/>gpurec/_cli_helpers.py:1996"]
  Config["RunConfig.from_dict(...)<br/>gpurec/workflow/config.py:1125"]
  ValidatePaths["_validate_run_config_input_paths(config)<br/>gpurec/_cli_helpers.py:1245"]
  ValidateFamilies["_validate_run_config_family_references(config)<br/>gpurec/_cli_helpers.py:1284"]
  ValidatePreprocess["optional CPU preprocessing check<br/>_validate_run_config_preprocess()<br/>gpurec/_cli_helpers.py:1619"]
  OptimizeCmd["optimize command<br/>gpurec/_cli_commands.py:972"]
  RunCmd["run command<br/>optimize then sample<br/>gpurec/_cli_commands.py:1528"]
  HelperOptimize["_run_optimize_command(config, argv)<br/>gpurec/_cli_helpers.py:1153"]
  WorkflowOptimize["workflow.optimize.optimize(config)<br/>gpurec/workflow/optimize.py:1724"]
  Runner["OptimizationRunner(config).run()<br/>gpurec/workflow/optimize.py:656"]

  Facade --> Commands --> Parser --> Args --> Config
  Config --> ValidatePaths --> ValidateFamilies
  ValidateFamilies -. "--check-preprocess" .-> ValidatePreprocess
  ValidateFamilies --> OptimizeCmd --> HelperOptimize --> WorkflowOptimize --> Runner
  ValidateFamilies --> RunCmd --> HelperOptimize
```

`gpurec validate-config` stops at the preflight node after config/path/family
validation by default. With `--check-preprocess`, it runs CPU `GeneDataset`
preprocessing through `_build_cpu_preprocess_dataset()`
(`gpurec/_cli_helpers.py:1498`), loads the retained Rust parser, and reports
whether the species-node count passes the retained CUDA backward `S > 256`
gate. `--require-cuda-backward-ready` makes that report a hard CLI gate. This
path runs without constructing `GeneReconModel`, loading the Rust preprocessing
extension for CUDA model construction, or touching CUDA; it validates parser and
mapping readiness before the optimization path.
In older release-gate wording, it runs without constructing `GeneReconModel`,
loading the Rust preprocessing extension, or touching CUDA.

## Mode Routing

| Mode | Theta shape | `optimizer=auto` | Main workflow branch |
| --- | --- | --- | --- |
| `genewise` | `[families, 3]` | `hessian-sgd` | Active resident batches, per-family vector loss/gradient, finite-difference Hessian row updates. |
| `specieswise` | `[species, 3]` | `adagrad-restarts` | Shared full-model closure with specieswise Adagrad restart phases and final high-budget check. |
| `global` | `[3]` | `adam` | Shared-theta uniform route through the scalar full-loss closure. |

The source of truth for defaults is
`MODE_DEFAULT_OPTIMIZERS` (`gpurec/workflow/config.py:36`). Mode validation and
optimizer compatibility live in `RunConfig` (`gpurec/workflow/config.py:915`
and `gpurec/workflow/config.py:991`).

`optimizer=auto` resolves to `hessian-sgd` for `mode=genewise`,
`adagrad-restarts` for `mode=specieswise`, and `adam` for `mode=global`.
`hessian-sgd` is genewise-only. `adagrad-restarts` is specieswise-only.
`adagrad-restarts-lbfgsb` is specieswise-only.

## Model Build

```mermaid
flowchart TD
  RunnerBuild["OptimizationRunner.build_model()<br/>gpurec/workflow/optimize.py:528"]
  Prefetch["mode/optimizer prefetch choice<br/>genewise active optimizers use prefetch=1<br/>gpurec/workflow/optimize.py:541"]
  Factory["build_alerax_workflow_model(config)<br/>gpurec/workflow/model_factory.py:13"]
  FromFamilies["GeneReconModel.from_alerax_families(...)<br/>gpurec/api/model.py:579"]
  ParseFamilies["parse_alerax_family_file(...)<br/>gpurec/core/model.py:57"]
  Dataset["GeneDataset(..., genewise/specieswise flags)<br/>gpurec/core/model.py:284"]
  Preprocess["GeneDataset._preprocess_without_cache(...)<br/>gpurec/core/model.py:428"]
  Rust["RustPreprocessExtension.preprocess_multiple_families(...)<br/>gpurec/core/preprocess_rust.py:645"]
  BatchSpecs["_build_batch_specs(...)<br/>gpurec/api/_batch_specs.py:310"]
  FamilyInputs["family_wave_inputs(...)<br/>gpurec/api/_family_layout.py:85"]
  Schedule["schedule_family_waves(...)<br/>gpurec/api/_family_layout.py:126"]
  Static["_build_batch_static_state(...)<br/>gpurec/api/_static_builder.py:86"]
  Layout["build_family_wave_layout(...)<br/>gpurec/api/_family_layout.py:152"]
  WaveLayout["build_wave_layout(...)<br/>gpurec/core/batching.py:394"]

  RunnerBuild --> Prefetch --> Factory --> FromFamilies
  FromFamilies --> ParseFamilies
  FromFamilies --> Dataset --> Preprocess --> Rust
  FromFamilies --> BatchSpecs --> FamilyInputs --> Schedule
  FromFamilies --> Static --> Layout --> WaveLayout
```

`OptimizationRunner.build_model()` also applies the first
`adagrad-restarts` phase budgets before model construction
(`gpurec/workflow/optimize.py:531`) so specieswise restart runs start with the
phase solver settings already installed.

## Iteration Loop

```mermaid
flowchart TD
  Run["OptimizationRunner.run()<br/>gpurec/workflow/optimize.py:656"]
  Plan0["prepare_initial_optimization_plan(...)<br/>gpurec/workflow/_step_plan.py:83"]
  PlanStep["select_step_optimization_plan(...)<br/>gpurec/workflow/_step_plan.py:225"]
  Phase["_build_phase_for_step(config, step)<br/>gpurec/workflow/_phase_controller.py:10"]
  MakeOpt["_make_optimizer(config, model, phase)<br/>gpurec/workflow/_optimizer_factory.py:30"]
  Step["execute_optimization_step(...)<br/>gpurec/workflow/_step_execution.py:176"]
  Artifacts["build_iteration_artifacts(...)<br/>gpurec/workflow/_rows.py:55"]
  Transition["apply_iteration_transition(...)<br/>gpurec/workflow/_transitions.py:607"]
  Record["_record(row)<br/>gpurec/workflow/optimize.py:592"]
  Checkpoint["_save_status(... latest/best.pt)<br/>gpurec/workflow/optimize.py:632"]
  Finalize["finalize_optimization(...)<br/>gpurec/workflow/_finalization.py:68"]

  Run --> Plan0 --> MakeOpt
  Run --> PlanStep --> Phase
  PlanStep --> MakeOpt
  MakeOpt --> Step --> Artifacts --> Transition
  Artifacts --> Record
  Transition --> Checkpoint
  Transition --> PlanStep
  Transition --> Finalize
```

The loop body starts at `gpurec/workflow/optimize.py:979`. The active-batch
genewise branch is enabled when `mode=genewise` and the optimizer is one of
`batched-lbfgs`, `adam-fd-newton`, or `hessian-sgd`
(`gpurec/workflow/optimize.py:117` and `gpurec/workflow/optimize.py:685`).

## Per-Step Likelihood And Gradient

```mermaid
flowchart TD
  Step["execute_optimization_step(...)<br/>gpurec/workflow/_step_execution.py:176"]
  ScalarClosure["scalar closure()<br/>evaluate_and_backward()<br/>gpurec/workflow/_step_execution.py:220"]
  BatchedClosure["batched_closure()<br/>genewise vector gradient<br/>gpurec/workflow/_step_execution.py:232"]
  LossProbe["loss-only probes<br/>evaluate_loss_only_probe()<br/>gpurec/workflow/_step_execution.py:265"]
  EvalScalar["EvaluationOps.evaluate_and_backward()<br/>gpurec/workflow/_evaluation.py:86"]
  EvalVector["EvaluationOps.evaluate_genewise_vector_and_grad()<br/>gpurec/workflow/_evaluation.py:130"]
  EvalActive["EvaluationOps.evaluate_active_genewise_vector_and_grad()<br/>gpurec/workflow/_evaluation.py:293"]
  FullLoss["GeneReconModel.full_loss()<br/>gpurec/api/model.py:1143"]
  FullVector["GeneReconModel.full_genewise_nll_and_grad()<br/>gpurec/api/model.py:1186"]
  ActiveNLL["GeneReconModel.nll_per_family()<br/>gpurec/api/model.py:1175"]
  Stream["stream_full_batches(..., need_grad)<br/>gpurec/api/_streaming.py:215"]
  StaticEval["evaluate_resident_static_state(..., need_grad)<br/>gpurec/api/_uniform_evaluator.py:132"]
  Forward["evaluate_resident_gradient_forward()<br/>gpurec/api/_uniform_evaluator.py:104"]
  Solve["solve_resident_e_pi()<br/>gpurec/api/autograd.py:396"]
  RootNLL["compute_pi_output_root_nll()<br/>gpurec/core/likelihood.py:510"]
  Implicit["compute_resident_implicit_gradient()<br/>gpurec/api/autograd.py:414"]
  VJP["implicit_grad_loglik_vjp_wave()<br/>gpurec/optimization/implicit_grad.py:126"]
  PiBackward["Pi_wave_backward()<br/>gpurec/core/backward.py:78"]
  BwdKernels["compute_wave_adjoint() / accumulate_split_dts_vjp()<br/>gpurec/core/kernels/wave_backward.py:1124,1701"]

  Step --> ScalarClosure --> EvalScalar --> FullLoss --> Stream --> StaticEval
  Step --> BatchedClosure --> EvalVector --> FullVector --> StaticEval
  BatchedClosure --> EvalActive --> ActiveNLL --> StaticEval
  Step --> LossProbe
  StaticEval --> Forward --> Solve --> RootNLL
  StaticEval --> Implicit --> VJP --> PiBackward --> BwdKernels
```

The same resident evaluator is used for all workflow modes. The shape of theta
and the optimizer branch decide whether the scalar closure, the genewise vector
closure, or an active-batch vector closure owns the step.

## Mode-Specific Optimization Paths

### Genewise

Default route: `mode=genewise`, `optimizer=hessian-sgd`.

1. `OptimizationRunner.build_model()` sets `prefetch_batches=1` for active
   genewise optimizers (`gpurec/workflow/optimize.py:541`).
2. `OptimizationRunner.run()` initializes active-batch state, selects the
   resident batch, and configures warmup/full solver stages
   (`gpurec/workflow/optimize.py:891`).
3. Each step uses `select_step_optimization_plan()` and then
   `execute_optimization_step()`.
4. The `hessian-sgd` branch configures validation/full/warmup solver budgets
   (`gpurec/workflow/_step_execution.py:566`) and calls the shared finite
   difference Newton runtime (`gpurec/workflow/_step_execution.py:622`).
5. The active-batch evaluator calls `model.nll_per_family()` and zeros inactive
   rows (`gpurec/workflow/_evaluation.py:293`).
6. Finalization prefers cached active-batch final loss/gradient when every
   family row is ready, otherwise it recomputes with
   `evaluate_genewise_vector_and_grad_with_memory_fallback()`
   (`gpurec/workflow/_finalization.py:95` and
   `gpurec/workflow/_finalization.py:114`).

Other genewise optimizers (`batched-lbfgs`, `adam-fd-newton`) reuse the same
active-batch vector evaluation surface. `batched-lbfgs` runs its row-wise
optimizer branch at `gpurec/workflow/_step_execution.py:456`; `adam-fd-newton`
shares the Hessian-conditioned branch at
`gpurec/workflow/_step_execution.py:566`.

### Specieswise

Default route: `mode=specieswise`, `optimizer=adagrad-restarts`.

1. `build_model()` installs the first restart phase solver budget before
   model construction (`gpurec/workflow/optimize.py:531`).
2. `prepare_initial_optimization_plan()` resolves the active restart phase and
   calls `configure_specieswise_adagrad_restart_phase()`
   (`gpurec/workflow/_step_plan.py:109` and
   `gpurec/workflow/_solver_stage.py:79`).
3. `select_step_optimization_plan()` emits phase names such as
   `adagrad-restarts:fixed8_warmup`
   (`gpurec/workflow/_step_plan.py:254`).
4. `_make_optimizer()` creates a standard `torch.optim.Adagrad` for each
   restart phase (`gpurec/workflow/_optimizer_factory.py:38`).
5. `execute_optimization_step()` falls through to the scalar closure branch,
   records projected gradient metrics for restart phases, and applies the
   optimizer step (`gpurec/workflow/_step_execution.py:720`).
6. Finalization can run a specieswise final solver-budget check via
   `configure_specieswise_final_eval_solver_stage()`
   (`gpurec/workflow/_finalization.py:86` and
   `gpurec/workflow/_solver_stage.py:31`).

`adagrad-restarts-lbfgsb` follows the same prefix and then switches to the
`lbfgsb` phase when the restart prefix ends
(`gpurec/workflow/_step_plan.py:169`).

Restart evidence is intentionally explicit in history and summary artifacts:
`adagrad_restart_total_steps`, `optimizer_step_cap`,
`optimizer_step_cap_reason`, and `adagrad_restart_final_check_iters` record the
schedule-derived step cap and final validation budget. Iteration rows record
these values and the workflow records `optimizer/adagrad_restart_*` fields.
Phase indexes, phase steps, phase
lengths, and E/Pi/Neumann budget fields are typed JSON integers;
`optimizer/adagrad_restart_restarted` is boolean.

### Global / Uniform Shared Theta

Default route: `mode=global`, `optimizer=adam`.

1. `_default_theta_init()` creates one `[3]` theta row for global mode
   (`gpurec/api/_theta_init.py:29`).
2. `build_model()` uses `prefetch_batches="all"` because global mode is not an
   active-batch genewise route (`gpurec/workflow/optimize.py:541`).
3. `_build_phase_for_step()` returns `adam` for every step unless a non-default
   optimizer was configured (`gpurec/workflow/_phase_controller.py:10`).
4. `_make_optimizer()` creates `torch.optim.Adam` for the phase
   (`gpurec/workflow/_optimizer_factory.py:36`).
5. `execute_optimization_step()` uses the scalar closure branch
   (`gpurec/workflow/_step_execution.py:720`), which calls
   `EvaluationOps.evaluate_and_backward()`.
6. The shared resident evaluator uses `GeneReconModel.full_loss()` and
   `stream_full_batches()`; because theta is shared, every resident batch
   contributes to one scalar loss and one `[3]` gradient.

The direct uniform chunked API is separate from the workflow runner:
`UniformChunkedReconModel.forward()` (`gpurec/api/uniform_chunked.py:1196`),
`nll()` (`gpurec/api/uniform_chunked.py:1202`), and `loss_and_grad()`
(`gpurec/api/uniform_chunked.py:1235`) call
`_evaluate_chunked_uniform()` / `_evaluate_chunked_uniform_read_only()`
(`gpurec/api/uniform_chunked.py:724` and
`gpurec/api/uniform_chunked.py:746`).

## Outputs And Checkpoints

```mermaid
flowchart TD
  Row["iteration row"]
  Record["_record(row)<br/>gpurec/workflow/optimize.py:592"]
  History["append_jsonl(history.jsonl, row)<br/>gpurec/workflow/diagnostics.py:231"]
  Save["_save_status(... latest.pt / best.pt)<br/>gpurec/workflow/optimize.py:632"]
  Checkpoint["save_checkpoint(...)<br/>gpurec/workflow/checkpoint.py:186"]
  Finalize["finalize_optimization()<br/>gpurec/workflow/_finalization.py:68"]
  Manifest["_build_run_manifest()<br/>gpurec/workflow/_artifacts.py:228"]
  FinalArtifacts["_write_final_artifacts()<br/>gpurec/workflow/_artifacts.py:461"]
  Rates["_write_rate_table()<br/>gpurec/workflow/_artifacts.py:384"]
  PerFamily["_write_per_family_likelihoods()<br/>genewise only<br/>gpurec/workflow/_artifacts.py:448"]
  Publish["_publish_final_artifacts()<br/>gpurec/workflow/_artifacts.py:309"]
  Result["optimization_result_from_summary()<br/>gpurec/workflow/_result.py:198"]

  Row --> Record --> History
  Row --> Save --> Checkpoint
  Finalize --> Manifest --> FinalArtifacts
  FinalArtifacts --> Rates
  FinalArtifacts --> PerFamily
  FinalArtifacts --> Publish --> Result
```

Final artifacts are staged before publication. `rates_final.tsv`,
`theta_final.pt`, `optimization_history.csv`, `history.jsonl`, `summary.json`,
and `run_manifest.json` are common outputs. `per_fam_likelihoods.tsv` is
written only for successful genewise final evaluation.

## Optimization Hot Path Notes

- The dominant likelihood path is shared by all workflow modes:
  `solve_resident_e_pi()` -> `Pi_wave_forward()` / `E_fixed_point()` ->
  `compute_pi_output_root_nll()` -> `compute_resident_implicit_gradient()` ->
  `Pi_wave_backward()`.
- The biggest workflow control split is not the likelihood kernel path; it is
  scalar shared-theta optimization versus genewise vector/active-batch
  optimization in `execute_optimization_step()`.
- Bounded optimizers still duplicate common concepts across
  `LBFGSB`, `ProjectedLBFGS`, and `BatchedLBFGS`. The current safe extraction
  target is shared bounds/closure/Armijo plumbing, while leaving
  `BatchedLBFGS._strong_wolfe()` behavior intact.
- Genewise Hessian-conditioned routes are the finite-difference Hessians
  hotspot. Specieswise restart routes mainly stress projected gradients and
  loss-only probes.
- The workflow runner is now mostly orchestration. Future slimming should keep
  mode policy in `_step_plan.py`, step mechanics in `_step_execution.py`,
  solver-budget policy in `_solver_stage.py`, and artifact publication in
  `_artifacts.py`.
