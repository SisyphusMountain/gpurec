# gpurec Optimization Workflow Call Graph

Scope: this follows the supported production path used by `gpurec
validate-config`, `gpurec optimize`, `gpurec run`, and the equivalent Python
API call `gpurec.optimize(RunConfig)`. It covers CPU-safe config preflight,
AleRax-style dataset processing, likelihood optimization, and final artifact
publication. `gpurec run` calls the same optimization path first, then continues
into sampling.

## End-To-End Workflow

```mermaid
flowchart TD
  CLI["gpurec.cli.main(argv)<br/>gpurec/cli.py:492"]
  Parser["build_parser() / _run_config_from_args(args)<br/>gpurec/cli.py:446,117"]
  Config["RunConfig.from_dict(...)<br/>gpurec/workflow/config.py:492"]
  Preflight["CLI preflight<br/>input paths and AleRax family references"]
  ValidateConfig["gpurec validate-config<br/>print valid_config summary"]
  CheckPreprocess["optional --check-preprocess<br/>CPU GeneDataset preprocessing"]
  CLIOptimize["gpurec.cli.optimize(config)<br/>gpurec/cli.py:40"]
  PyAPI["gpurec.optimize(config)<br/>gpurec/__init__.py:31<br/>gpurec/workflow/__init__.py:10"]
  Optimize["workflow.optimize(config)<br/>gpurec/workflow/optimize.py:809"]
  Runner["OptimizationRunner(config).run()<br/>gpurec/workflow/optimize.py:417"]

  BuildModel["OptimizationRunner.build_model()<br/>gpurec/workflow/optimize.py:312"]
  ModelFactory["build_alerax_workflow_model(config)<br/>gpurec/workflow/model_factory.py:11"]
  FromFamilies["GeneReconModel.from_alerax_families(...)<br/>gpurec/api/model.py:1072"]
  ParseFamilies["parse_alerax_family_file(...)<br/>gpurec/core/model.py:78"]
  Dataset["GeneDataset.__init__(...)<br/>gpurec/core/model.py:257"]
  LoadExt["_load_preprocess_extension()<br/>gpurec/core/model.py:202"]
  Preprocess["GeneDataset._preprocess_without_cache(..., preprocess_cpu_cores)<br/>gpurec/core/model.py:339"]
  RustPreprocess["RustPreprocessExtension.preprocess_multiple_families(..., num_threads)<br/>gpurec/core/preprocess_rust.py:442"]
  FamilyRaw["GeneDataset._family_from_raw(raw)<br/>gpurec/core/model.py:237"]

  ModelInit["GeneReconModel.__init__(...)<br/>gpurec/api/model.py:754"]
  BatchSpecs["_build_batch_specs(...)<br/>gpurec/api/model.py:453"]
  FamilyInputs["family_wave_inputs(...)<br/>gpurec/api/_family_layout.py:86"]
  Schedule["schedule_family_waves(...)<br/>gpurec/api/_family_layout.py:127"]
  BuildStatic["_build_batch_static_state(...)<br/>gpurec/api/model.py:605"]
  BuildLayout["build_family_wave_layout(...)<br/>gpurec/api/_family_layout.py:144"]
  WaveLayout["build_wave_layout(...)<br/>gpurec/core/batching.py:1224"]

  OptimizerLoop["optimizer loop over config.steps<br/>gpurec/workflow/optimize.py:486"]
  FinalEval["final _evaluate_and_backward(model)<br/>gpurec/workflow/optimize.py:673"]
  FinalArtifacts["_write_final_artifacts(...)<br/>gpurec/workflow/optimize.py:223"]
  Result["OptimizationResult<br/>gpurec/workflow/optimize.py:47"]

  CLI --> Parser --> Config --> Preflight
  Preflight --> ValidateConfig
  ValidateConfig -. "--check-preprocess" .-> CheckPreprocess
  CheckPreprocess --> Dataset
  Preflight --> CLIOptimize --> Optimize
  PyAPI --> Optimize
  Optimize --> Runner
  Runner --> BuildModel --> ModelFactory --> FromFamilies
  FromFamilies --> ParseFamilies
  FromFamilies --> Dataset
  Dataset --> LoadExt --> Preprocess --> RustPreprocess --> FamilyRaw
  FromFamilies --> ModelInit
  ModelInit --> BatchSpecs --> FamilyInputs --> Schedule
  ModelInit --> BuildStatic --> BuildLayout --> WaveLayout
  Runner --> OptimizerLoop --> FinalEval --> FinalArtifacts --> Result
```

`gpurec validate-config` stops at the preflight node by default. It validates
the flat JSON/CLI `RunConfig`, required paths, selected AleRax family records,
mapping files, and referenced gene-tree files without constructing
`GeneReconModel`, loading the Rust preprocessing extension, or touching CUDA.
With `--check-preprocess`, `validate-config` additionally runs CPU
`GeneDataset` preprocessing for the selected families; that optional path loads
the retained Rust parser and catches Newick or mapping errors before
optimization, then reports whether the preprocessed species-node count passes
the retained CUDA backward `S > 256` gate. It still does not construct the CUDA
likelihood model. `gpurec optimize` and `gpurec run` use the default lightweight
preflight before
entering the CUDA likelihood workflow. The Python API starts at
`workflow.optimize(config)` and therefore skips only the CLI parser/preflight
layer.

`--require-cuda-backward-ready` makes the `--check-preprocess` readiness report
a hard CLI gate. It is intended for production launch checks and still runs
entirely on CPU.

## Per-Step Likelihood And Gradient

This is the work done by each optimizer closure. The workflow model is built
with `lazy_preprocess=True`, so `full_loss()` uses the full-batch streaming
autograd bridge.

```mermaid
flowchart TD
  Closure["closure()<br/>gpurec/workflow/optimize.py:2747"]
  ClampBefore["model.clamp_theta_(min_rate, max_rate)<br/>gpurec/api/model.py:2496"]
  EvalBackward["OptimizationRunner._evaluate_and_backward(model)<br/>gpurec/workflow/optimize.py:822"]
  FullLoss["model.full_loss()<br/>gpurec/api/model.py:2319"]
  FullLossFn["_GeneReconFullLossFunction.forward(...)<br/>gpurec/api/model.py:1201"]
  Stream["_stream_full_batches(theta, need_grad=True)<br/>gpurec/api/model.py:1881"]
  ModelEval["_evaluate_static_state(static, theta, need_grad=True)<br/>gpurec/api/model.py:1184"]
  EvalStatic["evaluate_resident_static_state(static, theta, need_grad=True)<br/>gpurec/api/_uniform_evaluator.py:95"]
  ForwardSolve["evaluate_resident_gradient_forward(static, theta)<br/>gpurec/api/_uniform_evaluator.py:67"]
  SolveEPi["solve_resident_e_pi(static, theta, pi_training_state_request())<br/>gpurec/api/autograd.py:394"]
  Extract["extract_parameters_uniform(...)<br/>gpurec/core/extract_parameters.py:44"]
  EFixed["E_fixed_point(...)<br/>gpurec/core/likelihood.py:184"]
  EStep["E_step(...)<br/>gpurec/core/likelihood.py:105"]
  PiRequest["pi_training_state_request().run(...)<br/>gpurec/core/forward.py:88"]
  PiForward["Pi_wave_forward(...)<br/>gpurec/core/forward.py:155"]
  DTS["_compute_dts_cross(...)<br/>gpurec/core/forward.py:116"]
  DTSKernel["dts_fused_parent_reduced(...)<br/>gpurec/core/kernels/dts_fused.py:351"]
  SelfLoop["_run_wave_self_loop(...)<br/>gpurec/core/forward.py:433"]
  WaveStep["wave_step_uniform_fused_into(...)<br/>gpurec/core/kernels/wave_step.py:501"]
  Pibar["wave_pibar_uniform_parent_fused(...)<br/>gpurec/core/kernels/wave_step.py:650"]
  GatherRoots["gather_root_rows(...)<br/>gpurec/core/likelihood.py:423"]
  NLL["compute_nll_root_rows(...)<br/>gpurec/core/likelihood.py:428"]

  Implicit["compute_resident_implicit_gradient(...)<br/>gpurec/api/autograd.py:412"]
  VJP["implicit_grad_loglik_vjp_wave(...)<br/>gpurec/optimization/implicit_grad.py:126"]
  PiBackward["Pi_wave_backward(...)<br/>gpurec/core/backward.py:80"]
  BackwardKernels["wave_backward_uniform_fused(...) / dts_cross_backward_accum_fused(...)<br/>gpurec/core/kernels/wave_backward.py:1172,1749"]
  EAdjoint["_e_adjoint_and_theta_vjp(...)<br/>gpurec/optimization/implicit_grad.py:262"]
  Bicgstab["_bicgstab(...)<br/>gpurec/optimization/implicit_grad.py:59"]
  LossBackward["loss.backward()<br/>gpurec/workflow/optimize.py:825"]
  FullLossFnBackward["_GeneReconFullLossFunction.backward(...)<br/>gpurec/api/model.py:1211"]
  OptimizerStep["optimizer step<br/>Adam, Adagrad, adagrad-restarts,<br/>hessian-sgd, LBFGS variants"]
  Clear["model.clear()<br/>gpurec/api/model.py:2240"]

  Closure --> ClampBefore --> EvalBackward --> FullLoss --> FullLossFn --> Stream --> ModelEval --> EvalStatic
  EvalStatic --> ForwardSolve --> SolveEPi
  SolveEPi --> Extract
  SolveEPi --> EFixed --> EStep
  SolveEPi --> PiRequest --> PiForward
  PiForward --> DTS --> DTSKernel
  PiForward --> SelfLoop --> WaveStep
  SelfLoop --> Pibar
  ForwardSolve --> GatherRoots --> NLL
  EvalStatic --> Implicit --> VJP
  VJP --> PiBackward --> BackwardKernels
  VJP --> EAdjoint
  EAdjoint --> Bicgstab
  EAdjoint --> EStep
  EAdjoint --> Extract
  EvalBackward --> LossBackward --> FullLossFnBackward
  Closure --> OptimizerStep --> Clear
```

## Outputs And Checkpoints

```mermaid
flowchart TD
  StepRow["per-step metrics row<br/>gpurec/workflow/optimize.py:570"]
  Record["_record(row)<br/>gpurec/workflow/optimize.py:354"]
  Append["append_jsonl(history.jsonl, row)<br/>gpurec/workflow/diagnostics.py:163"]
  SaveStatus["_save_status(... latest.pt / best.pt)<br/>gpurec/workflow/optimize.py:393"]
  SaveCheckpoint["save_checkpoint(...)<br/>gpurec/workflow/checkpoint.py:122"]

  FinalArtifacts["_write_final_artifacts(...)<br/>gpurec/workflow/optimize.py:223"]
  Stage["create_artifact_temp_dir(...)<br/>gpurec/workflow/_artifact_publish.py:13"]
  History["_write_history_jsonl_with_final_row(...)<br/>gpurec/workflow/optimize.py:182"]
  Rates["_write_rate_table(rates_final.tsv, ...)<br/>gpurec/workflow/optimize.py:140"]
  PerFamily["_write_per_family_likelihoods(per_fam_likelihoods.tsv, ...)<br/>gpurec/workflow/optimize.py:174"]
  PerFamilyNLL["model.full_nll_per_family()<br/>gpurec/api/model.py:1693"]
  Theta["torch.save(model.theta, theta_final.pt)<br/>gpurec/workflow/optimize.py:262"]
  CSV["write_csv(optimization_history.csv, history)<br/>gpurec/workflow/diagnostics.py:169"]
  Summary["write_json_strict(summary.json, summary)<br/>gpurec/workflow/diagnostics.py:39"]
  Publish["_publish_final_artifacts(...)<br/>gpurec/workflow/optimize.py:210"]
  AtomicPublish["publish_staged_artifacts(...)<br/>gpurec/workflow/_artifact_publish.py:68"]
  OutDir["output directory<br/>history.jsonl, optimization_history.csv,<br/>rates_final.tsv, theta_final.pt, summary.json,<br/>checkpoints/latest.pt, checkpoints/best.pt,<br/>per_fam_likelihoods.tsv in genewise mode"]

  StepRow --> Record --> Append
  StepRow --> SaveStatus --> SaveCheckpoint
  FinalArtifacts --> Stage
  FinalArtifacts --> History
  FinalArtifacts --> Rates
  FinalArtifacts --> PerFamily --> PerFamilyNLL
  FinalArtifacts --> Theta
  FinalArtifacts --> CSV
  FinalArtifacts --> Summary
  FinalArtifacts --> Publish --> AtomicPublish --> OutDir
  SaveCheckpoint --> OutDir
```

## Notes

- `optimizer=auto` resolves to `hessian-sgd` for `mode=genewise`,
  `adagrad-restarts` for `mode=specieswise`, and `adam` for `mode=global`.
- The workflow optimizer phases are selected in
  `OptimizationRunner._phase_for_step()` and built in `_make_optimizer()`.
- `adagrad-restarts` is specieswise-only. Each schedule phase reconfigures
  `fixed_iters_E`, `fixed_iters_Pi`, and `neumann_terms` to the phase budget,
  resets Adagrad state, records `optimizer/adagrad_restart_*` fields, and uses
  `adagrad_restart_total_steps` as the schedule-derived optimizer cap.
  Route metadata reports the resulting `optimizer_step_cap` and
  `optimizer_step_cap_reason` alongside `configured_steps`, and uses
  `adagrad_restart_final_check_iters` for the final high-fidelity evaluation.
- `hessian-sgd` is genewise-only. It works on active resident batches, uses
  warmup solver budgets before the full stage, refreshes row-wise 3x3
  finite-difference Hessians, applies BFGS row updates between refreshes, and
  records projected gradients for rate-bound-aware stopping.
- Adam and Adagrad evaluate once before the parameter update. LBFGS-style
  optimizers use loss-only probes where supported, then record one accepted
  current-theta gradient evaluation.
- The direct `UniformChunkedReconModel` path has its own public surface in `gpurec/api/uniform_chunked.py`, but the production workflow above currently constructs `GeneReconModel`.
