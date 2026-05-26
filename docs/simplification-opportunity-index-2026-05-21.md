# Simplification Opportunity Index, 2026-05-21

This is the direct answer to: which documents and code paths show concrete
opportunities to simplify the project?

The rows below are not a new architecture proposal by themselves.  Each row
names an existing alternative path, branch family, compatibility surface, or
experimental route that can be removed, merged, or made private while retaining
the supported behavior:

- uniform/global rates: one D/T/L vector shared by all families;
- specieswise rates: one D/T/L vector per species;
- genewise rates: one D/T/L vector per family;
- first-order gradients for the supported likelihood path;
- export/backtracking through the high-level API.

## How This Answers The Question

The previous refactor documents explain how to implement the cleanup.  This
file explains what the cleanup is.  Read it as a deletion and consolidation
index:

- "Current alternatives" names the code paths that currently compete or
  duplicate work.
- "Simplification" states what to remove, merge, or make explicit.
- "Keep" states the core behavior that must remain.
- "Gate" states the proof needed before deleting code.

## High-Priority Consolidations

### EVAL-01 - Collapse The Four Evaluation Pipelines

Current alternatives:

- `gpurec/api/autograd.py:115` `_GeneReconFunction.forward()` runs E, Pi,
  root likelihood, tensor saving, and backward delegation.
- `gpurec/api/model.py:685` `_evaluate_static_state()` independently runs E,
  Pi, root-row likelihood, optional gradient, and stats.
- `gpurec/api/model.py:1748` `reconciliation_state()` repeats E/Pi work for
  export state.
- `gpurec/api/uniform_chunked.py:477` `_evaluate_chunked_uniform()` repeats
  E/Pi/gradient/stat reductions for chunked uniform mode.

Simplification:

- Put E solve, Pi solve, root likelihood, optional Pi backward, and optional
  E-adjoint behind one internal evaluator.
- Make autograd, no-grad full loss, export state, and chunked uniform calls
  select outputs from that evaluator instead of owning separate pipelines.

Keep:

- `GeneReconModel` as the main public model.
- `UniformChunkedReconModel` as either a facade or a compatibility wrapper.
- Uniform/global, specieswise, and genewise evaluation.

Gate:

- Parity tests over resident uniform, specieswise, genewise, resident-batched
  genewise, no-grad full loss, autograd loss, and chunked uniform loss/grad.

### MODE-01 - Replace Shape-Driven Mode Inference With `ParameterLayout`

Current alternatives:

- `gpurec/core/extract_parameters.py:6` `as_family_param()` infers family
  layout from rank and size.
- `gpurec/core/extract_parameters.py:20` `as_family_species()` infers
  family/species layout from rank and size.
- `gpurec/core/extract_parameters.py:29` `extract_parameters_uniform()` has
  separate `genewise` and `specieswise` branches.
- `gpurec/core/kernels/dts_fused.py:14` and
  `gpurec/core/kernels/wave_backward.py:46` maintain separate layout parsers.

Simplification:

- Build a single `ParameterLayout` once when preparing a model or batch.
- Pass explicit layout information into forward, backward, and kernels.
- Stop using tensor shape precedence as the source of truth.

Keep:

- The three supported modes: uniform/global, specieswise, genewise.

Gate:

- Tests where `G == S`, `G != S`, scalar/shared tensors, specieswise tensors,
  and genewise tensors all resolve to the intended layout.

### MODE-02 - Remove Shared-Mode Auto-Wrapping From Backward

Current alternatives:

- `gpurec/core/backward.py:130` wraps shared tensors as `G=1`, then keeps
  branches for shared constants and family-indexed constants.
- Forward and backward each rediscover whether a parameter is shared or
  family-specific.

Simplification:

- Make backward consume the same explicit `ParameterLayout` as forward.
- Remove the local "if shared, unsqueeze or wrap" policy from backward.

Keep:

- Shared uniform/global gradients.
- Specieswise gradients.
- Genewise gradients.

Gate:

- Gradient parity for all three modes against the current implementation.
- Characterization coverage for the current backward helper semantics:
  `family_idx=None` creates a zero-indexed `G=1` shared row, while explicit
  `family_idx` preserves row intent when species and family counts match.

### PI-01 - Split Pi Outputs By Intent Instead Of Boolean Flags

Current alternatives:

- `gpurec/core/forward.py:53` `Pi_wave_forward()` supports training tensors,
  root-row loss-only tensors, optional original returns, convergence tracing,
  final-Pibar storage, callbacks, and root row skipping in one function.

Simplification:

- Make output intent explicit, for example training, loss-only root rows, or
  export state.
- Keep one wave execution core, but remove call-site-specific boolean
  combinations from the public function boundary.

Keep:

- Wave-ordered `Pi`/`Pibar` for gradient computation.
- Root rows for loss-only evaluation.
- Full `Pi` for reconciliation export.

Gate:

- Identical losses and export rows for representative datasets.

### LIK-01 - Standardize Internal Likelihood On Root Rows

Current state:

- `compute_nll()` is a full-`Pi` adapter that gathers root rows.
- `compute_nll_root_rows()` owns the likelihood math.
- Runtime API loss and gradient paths gather or request root rows before NLL
  evaluation.

Simplification:

- Use root rows as the internal likelihood contract.
- Convert full `Pi` to root rows only at the evaluator boundary when needed.

Keep:

- Public loss values and per-family loss behavior.

Gate:

- Parity tests comparing full-`Pi` and root-row losses across all modes.

### LIK-02 - Remove Or Deprecate Misleading Log-Likelihood Aliases

Historical alternatives:

- `compute_log_likelihood()` and `compute_log_likelihood_root_rows()` are
  compatibility aliases that return negative log-likelihood values.

Simplification:

- Keep `compute_nll*` internally.
- Remove the misleading aliases after public usage is checked.
- Keep a source guard that prevents tracked runtime, tests, scripts, and
  profiling code from reintroducing them.

Keep:

- High-level model loss APIs.

Gate:

- Search results showing no supported public caller depends on the aliases.
- Repository hygiene coverage proving tracked Python surfaces use `compute_nll*`.

### ORIG-01 - Normalize Origination Probability Handling Once

Current alternatives:

- Multiple model/evaluation paths call `prepare_origination_probs()`.
- Some call sites rely on `assume_prepared` rather than a typed prepared
  object.

Simplification:

- Create one prepared origination-prior object at model or static-state setup.
- Pass that object into evaluator and likelihood code.

Keep:

- Existing origination-probability semantics and validation.

Gate:

- Tests for scalar, vector, matrix, and already-prepared origination inputs.

## Backward And Kernel Pruning

### BWD-01 - Keep One Production Self-Loop Backend

Original alternatives:

- `gpurec/core/backward.py` previously could route through retained Triton
  paths and experimental native CUDA self-loop paths controlled by environment
  flags.
- Several removed CUDA self-loop flags selected split, no-split, correction,
  block size, and edge-weight variants.

Simplification:

- Completed: keep the retained Triton self-loop backend as the production path.
- Move remaining prototypes to an experimental module, or delete them if they
  are not actively used.

Keep:

- Correct Pi-adjoint gradients for the supported modes.

Gate:

- Current performance path is at least as fast on the retained benchmark and
  passes gradient parity.

### BWD-02 - Make Active-Mask Pruning A Single Policy

Current alternatives:

- Backward includes active-mask pruning with a CPU synchronization decision and
  an environment flag to disable CPU pruning.

Simplification:

- Pick one policy, preferably device-side or always-on pruning if benchmarks
  support it.
- Remove the runtime CPU-pruning branch and environment flag.

Keep:

- Correct handling of inactive nodes and zero adjoints.

Gate:

- Benchmarks show no regression on sparse and dense active masks.

### BWD-03 - Centralize Gradient Accumulation

Current alternatives:

- Different paths accumulate D/T/L gradients, root contributions, and chunked
  reductions with local scatter/reduction logic.

Simplification:

- Introduce one internal gradient accumulator that knows the prepared
  `ParameterLayout`.
- Route resident, batched, and chunked gradients through it.

Keep:

- Same gradient tensor shapes and dtype behavior at public boundaries.

Gate:

- Gradient parity for shared, specieswise, genewise, and chunked uniform paths.

### DTS-01 - Share The DTS Layout Parser

Current alternatives:

- DTS forward and backward kernel helpers parse parameter layout independently.
- Layout ambiguity can appear when one-dimensional tensor sizes overlap.

Simplification:

- Use the prepared `ParameterLayout` for both forward and backward DTS kernels.
- Remove independent shape parsers from kernel wrappers.

Keep:

- Current fused DTS kernels and their performance-critical launch shapes.

Gate:

- Kernel wrapper tests for ambiguous dimensions and all retained rate modes.

### ENV-01 - Collapse Kernel Experiment Flags Into Typed Options

Current alternatives:

- The runtime reads about 31 `GPUREC_*` environment variables under `gpurec/`,
  many of which select kernel experiments or tuning variants.

Simplification:

- Keep only stable, documented user-facing environment variables.
- Move tuning choices into typed internal options or benchmark scripts.
- Delete flags for removed prototypes.

Keep:

- Memory policy overrides if they are documented and useful.
- Any retained benchmark knob needed for reproducible performance testing.

Gate:

- Documentation lists every remaining environment variable and its owner.

## Scheduler And Preprocessing Cleanup

### SCHED-01 - Choose One Non-Leaf Scheduler

Current alternatives:

- `gpurec/core/batching.py` contains forward, reverse-compacted, deadline, and
  Coffman-Graham-style nonleaf scheduler helpers.
- `schedule_global_phased_waves()` tries multiple candidate plans.

Simplification:

- Select one scheduler as the production scheduler.
- Move alternatives into historical benchmark notes or delete them.

Keep:

- Correct phased wave execution for the retained forward/backward pipeline.

Gate:

- Characterization tests for wave dependencies and benchmark comparison against
  the current selected plan.

### SCHED-02 - Remove Test-Only Scheduler Helpers From Runtime Surface

Current alternatives:

- Helpers such as `collate_wave`, `split_phase_waves`, and
  `compute_clade_waves` appear to be retained mainly for tests or diagnostics.

Simplification:

- Move them to tests, mark them private, or delete them after coverage is
  adjusted.

Keep:

- Public workflow and model behavior.

Gate:

- No high-level API or documented workflow imports these helpers.

### CPP-01 - Remove Legacy Pybind Preprocess Entry Points

Current alternatives:

- Python runtime uses `preprocess_multiple_families(..., include_details=True,
  include_species_matrices=False)`.
- Older pybind `preprocess()` and `include_details=False` branches preserve
  alternate contracts.

Simplification:

- Keep the one preprocessing contract used by the Python runtime.
- Remove or make private legacy pybind entry points after checking callers.

Keep:

- Current preprocessing outputs needed by `GeneReconModel` and workflow APIs.

Gate:

- End-to-end preprocessing tests through high-level Python APIs.

### CPP-02 - Remove Or Rehome C++ Diagnostic Exports

Current alternatives:

- C++ wave-stat diagnostic exports are available alongside runtime
  preprocessing outputs.

Simplification:

- Move diagnostics behind a development-only tool or remove them if unused.

Keep:

- Runtime preprocessing data needed by the evaluator.

Gate:

- Search confirms diagnostics are not part of supported user workflows.

## API, Validation, Scripts, And Tests

### API-01 - Stop Treating `gpurec.core` As Public API

Current alternatives:

- High-level APIs live under `gpurec`, `gpurec.api`, and `gpurec.workflow`.
- Tests and some docs import `gpurec.core` helpers directly, which makes
  internals look supported.

Simplification:

- Document `gpurec.core` as an unstable implementation namespace.
- Keep tests free to import internals, but do not treat those imports as public
  API evidence.

Keep:

- High-level public imports and workflow helpers.

Gate:

- Documentation and repository hygiene tests agree on the supported surface.

### VALID-01 - Centralize Input Validation

Current alternatives:

- API, workflow, CLI, and chunked model paths each validate pieces of the same
  model inputs and runtime options.

Simplification:

- Move shared validation into one internal validation module or prepared-state
  constructor.

Keep:

- Current user-facing error behavior where it is tested or documented.

Gate:

- Invalid-input tests cover the centralized validator instead of duplicate
  per-path checks.

### CHUNK-01 - Make `UniformChunkedReconModel` A Client Of The Shared Evaluator

Current alternatives:

- `UniformChunkedReconModel` duplicates static setup, chunk planning,
  likelihood/gradient evaluation, stats, dtype policy, and metadata handling.

Simplification:

- Keep the class as a facade if the public API is useful.
- Remove its private duplicate evaluator after it delegates to the shared
  evaluator.

Keep:

- Large uniform/global chunked training behavior.

Gate:

- Parity between current chunked `loss_and_grad()` and shared-evaluator
  chunked evaluation.

### SCRIPT-01 - Delete Or Reclassify Fixed-Dataset Scripts

Current alternatives:

- Several tracked scripts and generated reports are tied to local HOGENOM or
  historical profiling workflows.

Simplification:

- Keep portable examples and maintained benchmark commands.
- Move local artifacts to ignored scratch space or historical notes.

Keep:

- Minimal examples and maintained performance reproduction docs.

Gate:

- Release-readiness checks distinguish portable examples from local artifacts.

### TEST-01 - Stop Preserving Dead Internals Through Tests

Current alternatives:

- Tests cover useful behavior, but some white-box tests preserve internal helper
  shapes or scheduler helpers that are candidates for deletion.

Simplification:

- Keep high-value behavioral tests.
- Convert tests for deleted internals into high-level characterization tests or
  remove them with the code.

Keep:

- Regression coverage for likelihood, gradients, preprocessing, and workflow
  behavior.

Gate:

- Test plan maps each deleted internal to either a replacement behavior test or
  a deliberate deletion note.

## Suggested Order

1. Add characterization tests around current loss, gradient, preprocessing,
   scheduler, and export behavior.
2. Introduce explicit `ParameterLayout` and prepared origination-prior objects.
3. Build the shared evaluator and route resident no-grad/autograd/export paths
   through it.
4. Route `UniformChunkedReconModel` through the shared evaluator.
5. Prune backward kernel alternatives and collapse environment flags.
6. Pick one scheduler and remove scheduler experiments from runtime code.
7. Shrink C++ exports, scripts, and white-box tests that only preserve deleted
   internals.

## Links To Detailed Plans

- `refactor-simplification-plan-2026-05-21.md` explains the target internal
  architecture and phased implementation.
- `gradient-likelihood-refactor-plan-2026-05-21.md` expands E/Pi/root
  likelihood and gradient consolidation.
- `runtime-surface-pruning-plan-2026-05-21.md` expands API, scheduler, C++,
  environment-variable, script, and test cleanup.
