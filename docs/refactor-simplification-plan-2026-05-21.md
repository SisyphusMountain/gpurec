# Refactor Simplification Plan, 2026-05-21

This is a current-state, documentation-only plan for reducing branchy runtime
logic while preserving the supported computational modes:

- uniform/global rates: one D/T/L vector shared across all families;
- specieswise rates: one D/T/L vector per species;
- genewise rates: one D/T/L vector per family.

If the question is "what specific paths can be simplified or removed?", start
with `simplification-opportunity-index-2026-05-21.md`.  This file is the
implementation plan behind that index: it explains how to consolidate the
listed paths without dropping uniform/global, specieswise, or genewise support.

The plan was produced from the dirty worktree on 2026-05-21.  Code was not
edited for this pass.  Existing uncommitted code changes were present before
this documentation update, so implementation follow-up should rebase these
notes against the exact source state being edited.

## Evidence Snapshot

- Largest active complexity hotspots from an AST branch count:
  - `gpurec/api/model.py:832` `GeneReconModel`, about 1026 class lines.
  - `gpurec/core/backward.py:33` `Pi_wave_backward`, about 579 function lines.
  - `gpurec/core/forward.py:53` `Pi_wave_forward`, about 345 function lines.
  - `gpurec/core/batching.py:1054` `schedule_global_phased_waves`, plus four
    competing non-leaf scheduling helpers.
  - `gpurec/api/uniform_chunked.py:477` `_evaluate_chunked_uniform`, about 232
    function lines.
- Current package runtime reads about 31 distinct `GPUREC_*` environment
  variables under `gpurec/`.
- The public high-level package surface is already small:
  `GeneReconModel`, `UniformChunkedReconModel`, metadata dataclasses, workflow
  runners/configs, and backtracking helpers.
- Much of the remaining complexity is therefore internal: duplicated evaluation
  engines, shape-driven mode inference, optional kernel/prototype selectors,
  scheduler experiments, and compatibility aliases.

## Target Architecture

Use one internal uniform-transfer evaluation stack with explicit typed
contracts:

1. `RateMode` and `ParameterLayout` describe global, specieswise, and genewise
   theta addressing.  They replace scattered `genewise` / `specieswise` /
   `family_idx is None` branches and ad hoc tensor-shape precedence.
2. `UniformStaticState` holds species topology, ancestor matrix, wave layout,
   origination prior, solver controls, and memory policy.  It should be shared
   by resident and chunked callers.
3. `EvaluationRequest` states what is needed: loss only, loss plus gradient,
   per-family loss, or export state.
4. `UniformEvaluator.evaluate(request)` owns E solve, Pi solve, root NLL, and
   optional Pi/E-adjoint gradient.  Autograd functions become thin wrappers
   that cache the evaluator's gradient.
5. `BatchPlan` and `WavePlan` are explicit prepared objects.  Public APIs ask
   for batches/chunks through those abstractions instead of rebuilding or
   reinterpreting scheduling metadata in several places.

This keeps the three supported modes while removing most branches that exist
only because the current implementation lets each layer rediscover mode,
layout, and output intent independently.

## Highest-Impact Work Packages

### 1. Merge Resident And Chunked Evaluation Paths

Current duplication:

- `gpurec/api/autograd.py:115` `_GeneReconFunction.forward()` runs E, Pi,
  `compute_nll()`, saves tensors, and delegates backward.
- `gpurec/api/model.py:685` `_evaluate_static_state()` independently runs E,
  Pi, root-row loss-only evaluation, and optional gradient.
- `gpurec/api/model.py:1748` `reconciliation_state()` repeats the E/Pi solve
  for export state.
- `gpurec/api/uniform_chunked.py:477` `_evaluate_chunked_uniform()` repeats E,
  chunked Pi forward, optional Pi backward, E-adjoint, stats, and reductions.

Plan:

- Add a model-internal evaluator module, for example
  `gpurec/api/_uniform_evaluator.py`.
- Move common E/Pi/root-likelihood logic behind one request object.
- Make `_GeneReconFunction`, `_GeneReconFullLossFunction`, and
  `_UniformChunkedFunction` call the evaluator instead of open-coding the
  pipeline.
- Move chunk selection into a `BatchPlan` adapter so `GeneReconModel` can serve
  the sampled-chunk `loss_and_grad()` use case.
- Keep `UniformChunkedReconModel` initially as a public facade over the shared
  evaluator, then decide whether to deprecate it or keep it as a convenience
  constructor for large global/uniform datasets.

Expected simplification:

- Removes one full duplicate E/Pi solve from `reconciliation_state()`.
- Makes no-grad loss probes, differentiable autograd calls, full resident
  streaming, and chunked `loss_and_grad()` share one gradient/loss contract.
- Reduces failure modes where one path updates `warm_E`, stats, or origination
  probabilities differently from another.

Risk:

- High behavioral blast radius.  Start with parity tests over:
  global/uniform resident, specieswise resident, genewise resident,
  resident-batched genewise, and `UniformChunkedReconModel`.

### 2. Replace Shape-Driven Mode Branches With `ParameterLayout`

Current duplication and ambiguity:

- `gpurec/core/extract_parameters.py:6` `as_family_param()` and
  `as_family_species()` infer family/species layout from tensor ranks and sizes.
- `gpurec/core/extract_parameters.py:29` `extract_parameters_uniform()` has
  separate `genewise` and `specieswise` branches.
- `gpurec/core/forward.py:126` prepares separate batched and non-batched
  constants for the same kernels.
- `gpurec/core/backward.py:130` auto-wraps shared mode as `G=1`, then keeps
  separate shared and family-indexed paths.
- `gpurec/core/kernels/dts_fused.py:14` and
  `gpurec/core/kernels/wave_backward.py:46` have independent layout parsers;
  the audit already found a `G == S` precedence ambiguity for one-dimensional
  tensors.

Plan:

- Define a single `ParameterLayout` with:
  - mode: `global`, `specieswise`, or `genewise`;
  - `theta_shape`;
  - `family_count` and `species_count`;
  - `family_idx` policy for wave-ordered clades;
  - row stride and species stride for scalar, species, family, and
    family-species tensors.
- Convert theta once per evaluation into a `UniformRates` object:
  `log_pS`, `log_pD`, `log_pL`, `max_transfer`, plus layout metadata for
  gradient reduction.
- Pass layout metadata to forward/backward kernels instead of asking each
  kernel wrapper to infer semantics from shape.
- Make impossible shapes invalid at API construction time, not inside kernels.

Expected simplification:

- Removes `_auto_wrapped` logic from `Pi_wave_backward`.
- Removes most `family_indexed_consts` and `family_idx is not None` conditionals
  from forward wrappers.
- Gives one place to reason about global, specieswise, and genewise semantics.

Risk:

- Kernel argument compatibility and gradient reduction must be tested carefully,
  especially for specieswise `[S, 3]` and genewise `[G, 3]`.

### 3. Split `Pi_wave_forward` By Output Intent

Current branch surface:

- `return_original`, `return_root_rows`, `trace_root_logsumexp`,
  `convergence_tolerance`, `progress_callback`, final-Pibar storage, and root
  row skipping all live inside `gpurec/core/forward.py:53`.
- Training needs wave-ordered `Pi`, wave-ordered `Pibar`, and
  `uniform_pibar_row_max`.
- Loss-only inference needs only root rows.
- Export state needs full `Pi`, optionally original order.

Plan:

- Keep one low-level kernel loop, but expose explicit wrappers:
  - `pi_forward_for_gradient(...)`;
  - `pi_forward_root_rows(...)`;
  - `pi_forward_state(...)`.
- Return a dataclass instead of a loose dictionary.
- Move root-logsumexp tracing and progress callbacks out of the hot production
  function, or keep them in a debug wrapper.
- Hard-code the retained defaults for final-Pibar fusion and non-leaf leaf-term
  specialization after a benchmark gate, then remove the runtime env selectors
  from the wave loop.

Expected simplification:

- Easier evaluator call sites.
- Fewer optional `None` outputs and fewer defensive `get()` calls.
- Lower risk of calling backward without `uniform_pibar_row_max`.

Risk:

- Adaptive iteration and root trace behavior need focused parity tests.

### 4. Prune Backward Runtime Alternatives

Current branch surface:

- `Pi_wave_backward()` contains shared-mode auto-wrapping, family-indexed
  constants, active-mask CPU sync pruning, optional native CUDA self-loop
  prototypes, optional CUDA Pibar VJP prototype, and several env-controlled
  zero-fill/tuning branches.
- The retained production path is the Triton 2D self-loop plus fused DTS
  backward accumulation plus compact Pibar VJP.

Plan:

- Keep one production self-loop backend in `Pi_wave_backward`: the retained
  Triton 2D path.
- Move native CUDA self-loop and CUDA Pibar prototypes behind an explicit
  experimental module or benchmark script, or delete them if no required-mode
  CUDA smoke is going to own them.
- Make device active-mask pruning the only policy; avoid per-wave CPU sync
  decisions in production.
- Make compact species topology mandatory.  This is already effectively true.
- Replace ad hoc per-wave scatter accumulation with a `GradientAccumulator`
  object that receives the `ParameterLayout`.

Expected simplification:

- Converts a 579-line function into a shorter orchestration loop plus typed
  helpers.
- Removes fallback exception handling and env parsing from the gradient hot
  path.
- Reduces the test matrix for backward kernels.

Risk:

- High performance risk.  Benchmark on `test_trees_1000` before deleting
  alternative code paths.  Keep prototypes in git history or a clearly
  experimental file until parity and performance are proven.

### 5. Simplify E And Root-Likelihood Contracts

Current branch surface:

- `E_fixed_point()` infers row count from several parameter shapes.
- `compute_nll()` and `compute_nll_root_rows()` duplicate root likelihood logic.
- The former `compute_log_likelihood()` and
  `compute_log_likelihood_root_rows()` compatibility aliases returned NLL
  despite their names and have been removed.
- `prepare_origination_probs()` is called from several layers with an
  `assume_prepared` trust boundary.

Plan:

- Make `ParameterLayout` provide the E row shape explicitly.
- Normalize origination probabilities once into an `OriginationPrior` object.
- Standardize on root-row likelihood internally.  Full-Pi callers should gather
  root rows before calling the likelihood helper.
- Keep profiling/tests on `compute_nll*` and prevent reintroduction of the
  removed misleading likelihood aliases.
- Make E warm-start policy explicit in the evaluator: disabled, active-batch,
  or streaming global.  Avoid current differences where some no-grad paths pass
  `warm_start_E=None` while autograd/chunked paths can use `warm_E`.

Expected simplification:

- Removes shape inference from the fixed-point solver.
- Removes duplicated likelihood functions and removed-alias tests.
- Makes loss-only line-search probes easier to reason about.

Risk:

- Public low-level helpers may be used by notebooks.  Use a deprecation period
  for aliases, but keep internal code on the new names immediately.

### 6. Choose One Scheduler Policy

Current branch surface:

- `gpurec/core/batching.py` contains forward, reverse-compacted, deadline, and
  Coffman-Graham-style non-leaf schedulers.
- `schedule_global_phased_waves()` tries multiple candidates and chooses the
  one with fewer waves.
- `collate_wave()`, `split_phase_waves()`, and
  `gpurec/core/scheduling.py:13` `compute_clade_waves()` appear used only by
  tests/docs, not by production model construction.

Plan:

- Benchmark scheduler variants on representative small, medium, and HOGENOM
  batches.
- Pick one supported scheduler and one fallback-free implementation.
- Move diagnostic-only scheduler helpers to tests or delete them if they are
  not public API.
- Keep a single public batch/wave metadata surface through model metadata.

Expected simplification:

- Removes several hundred lines of scheduling alternatives.
- Makes wave layout deterministic and easier to debug.

Risk:

- The current multi-candidate scheduler may hide poor cases.  Need performance
  benchmarks before deletion.

### 7. Collapse Environment Options Into Typed Runtime Options

Current package env surface includes memory policy, kernel tuning, prototype
backend selectors, pruning toggles, and binary discovery.  Runtime code reads
many variables inside forward/backward functions.

Plan:

- Add a `RuntimeOptions` or `KernelOptions` object built once per model/static
  state.
- Keep only stable user-facing env vars:
  - `GPUREC_BACKTRACK_BIN`;
  - `GPUREC_BACKTRACK_NATIVE_LIB`;
  - `GPUREC_ALERAX_COMPAT`;
  - `GPUREC_MEMORY_POLICY_FRACTION`;
  - `GPUREC_MEMORY_POLICY_RESERVE_GIB`;
  - `GPUREC_PREPROCESS_BIN`;
  - `GPUREC_PREPROCESS_NATIVE_LIB`.
- Move kernel block-size tuning and prototype selectors to profiling scripts or
  explicit constructor/debug arguments.
- Remove env reads from per-wave hot loops.

Expected simplification:

- Smaller README env contract.
- Fewer hidden behavior changes between runs.
- Easier reproducibility for benchmarks and workflow checkpoints.

Risk:

- Some env options may be useful during performance investigations.  Preserve
  them in profiling scripts before deleting production reads.

## Medium-Impact Work Packages

### Public API Surface And Compatibility

- Make `GeneDataset` purely a preprocessing/cache object.  It is already close:
  runtime evaluation now lives in `gpurec/api/model.py` and
  `gpurec/api/uniform_chunked.py`.
- Update `gpurec/__init__.py` documentation to steer users to the high-level
  API, not low-level `gpurec.core` functions.
- Decide whether `UniformChunkedReconModel` remains a first-class class or
  becomes a compatibility facade over `GeneReconModel` plus the shared
  evaluator.
- Remove or deprecate core-level public imports that are only used by tests and
  profiling.

### C++ Preprocessing Extension Surface

- Decide whether the legacy pybind `preprocess()` wrapper remains.  Python
  runtime calls `preprocess_multiple_families(..., include_details=True)`.
- Decide whether `include_details=False` is public or dead compatibility.
- Remove unowned C++ diagnostic wave-stat pybind exports, or move them to a
  benchmark/diagnostic extension with explicit ownership.

### Workflow And Optimizer Surface

- Keep optimizer modes if they are truly user-facing: `adam`, `adagrad`,
  `lbfgs`, and `adam-lbfgs`.
- Move line-search loss-only probing into the shared evaluator when implementing
  the second-order/pseudo-second-order plan.
- Avoid adding optimizer-specific model internals.  Optimizers should consume
  `loss`, `grad`, stats, and batch metadata only.

### Scripts And Profiling

- Convert legacy fixed-HOGENOM scripts into config examples or delete them.
- Keep one maintained benchmark entry point per supported path:
  resident model, chunked/global uniform, scheduler, and backtracking.
- Move stale report scripts to an archive directory or document them as
  historical.

## Deletion Candidate Inventory

Deletion candidates should not be removed all at once.  They need the parity
and benchmark gates in this plan.

- Removed misleading compatibility aliases:
  `compute_log_likelihood()` and `compute_log_likelihood_root_rows()`.
- Production env selectors after defaults are chosen:
  `GPUREC_FUSE_FINAL_PIBAR`,
  `GPUREC_SPECIALIZE_NONLEAF_LEAF_TERM`,
  `GPUREC_BACKWARD_NO_CPU_PRUNING`,
  `GPUREC_DTS_SKIP_INACTIVE_PIBAR_ZERO`, and kernel block/warp tuning reads.
- Native CUDA prototype selectors from production paths:
  `GPUREC_CUDA_SELF_LOOP_NOSPLIT`,
  `GPUREC_CUDA_SELF_LOOP_SPLIT`,
  `GPUREC_CUDA_PIBAR_FROM_UD`, and associated tuning flags, unless they become
  owned benchmark-only paths.
- Scheduler helpers used only by tests/docs:
  `collate_wave()`, `split_phase_waves()`, and `compute_clade_waves()`.
- C++ pybind diagnostic exports if no supported workflow uses them:
  wave-stat and cross-family wave-stat helpers.
- Legacy C++ `preprocess()` wrapper and
  `preprocess_multiple_families(..., include_details=False)` if no public
  consumer is kept.
- `UniformChunkedReconModel` internals duplicated with `GeneReconModel`, after
  a shared evaluator can serve large global/uniform datasets.

## Suggested Implementation Order

1. Add characterization tests around current global, specieswise, genewise, and
   chunked outputs.  Include root-row loss, full-Pi state, per-family genewise
   loss, full resident streaming, and direct chunked `loss_and_grad()`.
2. Introduce `RateMode`, `ParameterLayout`, `UniformRates`, and
   `OriginationPrior` without changing kernels.
3. Refactor E/root likelihood to use explicit layout and root rows.
4. Add the shared evaluator and route `_evaluate_static_state()` through it.
5. Route autograd functions through the evaluator.
6. Route `UniformChunkedReconModel` through the evaluator.
7. Split `Pi_wave_forward` wrappers by output intent.
8. Refactor `Pi_wave_backward` around `ParameterLayout` and
   `GradientAccumulator`.
9. Benchmark and pick one scheduler policy.
10. Prune env/prototype/C++ diagnostic surfaces.
11. Clean tests, docs, and scripts after runtime behavior is stable.

## Verification Gates

Run lightweight gates after every structural step:

- `python -m py_compile` over touched modules.
- CPU unit tests for validation, model API, origination probability, scheduler,
  batch planning, and repository hygiene.
- `python -m pytest --collect-only -q`.

Run CUDA/data gates before deleting any alternate compute path:

- `tests/integration/test_gene_recon_model.py` on global, specieswise, and
  genewise paths.
- `tests/integration/test_uniform_chunked_model.py`.
- `tests/unit/test_specieswise_uniform.py` and backward/core GPU tests when
  `test_trees_1000` is available.
- `profiling/bench_uniform_forward_backward_pipeline.py` with
  `--strict-optimized-kernels`.

Acceptance criteria:

- Public modes remain `global`/uniform, `specieswise`, and `genewise`.
- Losses and gradients match current behavior within documented tolerances.
- Backtracking export state still has the same public fields.
- Large uniform benchmark does not regress beyond an agreed threshold.
- Runtime env surface is smaller and documented.
- Internal line count and branch count decrease in the hotspot modules.
