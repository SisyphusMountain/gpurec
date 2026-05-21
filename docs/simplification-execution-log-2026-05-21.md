# Simplification Execution Log, 2026-05-21

This log tracks concrete work against
`simplification-opportunity-index-2026-05-21.md`.  It is not the proposal
inventory; it is the execution record used to keep commits, tests, and
benchmark gates tied to specific proposal IDs.

## Completed Commits

### `73f8752` - Document audit surface and hygiene guards

Proposal coverage:

- `API-01`: clarified high-level API and unstable `gpurec.core` boundary.
- `LIK-02`: added deprecation warnings for misleading likelihood aliases and
  moved ordinary test usage to `compute_nll*`.
- `CPP-01`: documented the legacy direct `preprocess` pybind as compatibility
  surface.
- `CPP-02`: documented direct C++ scheduler/stat exports as diagnostic surface.
- `SCHED-02`: guarded deleted Python scheduler helpers against returning to
  runtime source.
- `SCRIPT-01`: documented config and profiling ownership boundaries.
- `TEST-01`: added repository-hygiene guards for the new boundaries.
- `VALID-01`: documented internal validation helper ownership.

Verification:

- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_repository_hygiene.py tests/unit/test_origination_probs.py tests/unit/test_core_helpers.py tests/unit/test_validation.py tests/unit/test_family_layout.py -q`: 195 passed.
- `CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider -m "unit and not gpu"`: 947 passed, 1 skipped, 30 deselected after the follow-up CPU-gate README fix.
- `PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests/kernels/test_wave_step_uniform_forward_kernel.py tests/integration/test_gene_recon_model.py::test_gene_recon_model_forward_backward_modes tests/integration/test_uniform_chunked_model.py -m "not slow"`: 11 passed, 1 deselected.
- `python profiling/bench_uniform_forward_backward_pipeline.py --help`: passed.
- `python profiling/bench_uniform_forward_backward_pipeline.py --dataset tests/data/test_trees_1000 --fams 1 --family-chunk-size 1 --max-wave-size 8192 --fixed-iters 6 --warmups 0 --reps 1 --stats-only --strict-optimized-kernels --cache-dir /tmp/gpurec_perf_cache`: `strict_optimized_verdict pass`.
- `git diff --check`: passed before the follow-up commit.

Notes:

- This commit mostly classifies and guards surfaces.  It does not remove the
  legacy pybinds, remove the likelihood aliases, or refactor evaluator/runtime
  paths.

### `882a908` - Document extract-parameters CPU test gate

Proposal coverage:

- `TEST-01`: fixed the explicit CPU-unit test list so the documented command
  matches the marker-selected unit suite.

Verification:

- `CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider -m "unit and not gpu"`: 947 passed, 1 skipped, 30 deselected.

### `04d0aab` - Characterize evaluator paths and repair benchmark harness

Proposal coverage:

- `TEST-01`: added characterization coverage for resident evaluator/export
  paths, chunked selected-chunk behavior, scheduler phase/root-cap behavior,
  and multi-record preprocessing.
- `EVAL-01`: guarded duplicated resident paths by comparing `forward()`,
  `full_loss()`, `full_loss_for_theta()`, `pi_matrix()`, and
  `reconciliation_state()` losses for global, specieswise, and genewise modes.
- `CHUNK-01`: guarded chunked selected/full chunk behavior, per-family NLL
  ordering, reduction scaling, and loss/gradient stats.
- `SCHED-01`: guarded scheduler phase barriers, topological validity, invalid
  parent rejection, and root-cap behavior.
- `CPP-01`: guarded multi-record family preprocessing when the final Newick
  record omits a trailing semicolon.
- Benchmark guard repair: removed stale `ancestors_T` usage from the profiling
  harness `Pi_wave_backward()` call and extended the signature hygiene guard.

Verification:

- `python -m py_compile tests/integration/test_gene_recon_model.py tests/integration/test_uniform_chunked_model.py tests/unit/test_global_wave_scheduler.py tests/unit/test_alerax_family_input.py tests/unit/test_repository_hygiene.py profiling/bench_uniform_forward_backward_pipeline.py`: passed.
- `PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests/integration/test_gene_recon_model.py tests/integration/test_uniform_chunked_model.py`: 16 passed.
- `CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests/unit/test_global_wave_scheduler.py tests/unit/test_alerax_family_input.py tests/unit/test_repository_hygiene.py::test_pi_wave_backward_signature_omits_unused_ancestors_t`: 83 passed.
- `CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider -m "unit and not gpu"`: 950 passed, 1 skipped, 33 deselected.
- `python profiling/bench_uniform_forward_backward_pipeline.py --stats-only --strict-optimized-kernels --fams 1 --family-chunk-size 1 --max-wave-size 8192 --fixed-iters 2 --compare-unchunked-max-fams 0`: `strict_optimized_verdict pass`.
- `python profiling/bench_uniform_forward_backward_pipeline.py --fams 1 --family-chunk-size 1 --max-wave-size 8192 --fixed-iters 2 --reps 1 --warmups 0 --compare-unchunked-max-fams 0`: passed with finite gradients and `total_ms` around 870 ms on the local RTX 4090 run.
- `git diff --check`: passed.

### `587a3c8` - Add explicit parameter layout contracts

Proposal coverage:

- `MODE-01`: added inert `RateMode` and `ParameterLayout` contracts for
  global/uniform, specieswise, and genewise theta layouts.
- Guarded explicit shape intent, including `G == S` ambiguity when mode is
  omitted.
- Exposed family/species row metadata for later evaluator and kernel refactors
  without rerouting current hot paths.

Verification:

- `pytest -q tests/unit/test_parameter_layout.py`: 13 passed in the worker
  worktree.
- `pytest -q tests/unit/test_extract_parameters.py tests/unit/test_validation.py`:
  58 passed in the worker worktree.
- Main combined gates are listed under `5bb70f6`.

### `b80ee9c` - Add origination prior contract

Proposal coverage:

- `ORIG-01`: added inert `OriginationPrior` and `PreparedOriginationPrior`
  objects around the existing origination-probability semantics.
- Covered default uniform, shared vector, family-specific matrix, prepared
  trust-boundary, log-weight, device/dtype conversion, and family-subset
  selection behavior.
- Did not edit likelihood hot paths.

Verification:

- `pytest tests/unit/test_origination_prior.py`: 7 passed in the worker
  worktree.
- `pytest tests/unit/test_origination_probs.py tests/unit/test_family_layout.py`:
  19 passed in the worker worktree.
- Main combined gates are listed under `5bb70f6`.

### `bb7efc2` - Centralize uniform chunk validation helpers

Proposal coverage:

- `VALID-01`: moved duplicated auto-int and integer-sequence validation helpers
  into shared API validation support.
- Preserved the existing private helper names in `uniform_chunked.py` as aliases
  so public behavior and error timing stay stable.

Verification:

- Worker targeted command covering validation and workflow chunk controls:
  108 passed.
- Main combined gates are listed under `5bb70f6`.

### `c0f1cf4` - Test likelihood root-row origination parity

Proposal coverage:

- `LIK-01`: added root-row/full-Pi NLL parity tests for default/shared, raw
  vector, prepared vector, and family-specific origination probabilities.
- No production code changed.

Verification:

- `pytest -q tests/unit/test_origination_probs.py`: 13 passed in the worker
  worktree.
- Main combined gates are listed under `5bb70f6`.

### `f865c8b` - Document GPUREC environment variable ownership

Proposal coverage:

- `ENV-01`: added an owner manifest for all tracked package `GPUREC_*`
  environment variables.
- Added a hygiene guard that compares tracked package env references against
  the manifest before any flag removal occurs.

Verification:

- Targeted worker hygiene command: 3 passed.
- Main combined gates are listed under `5bb70f6`.

### `5bb70f6` - Document new contract CPU tests

Proposal coverage:

- `TEST-01`: updated the explicit CPU-unit command list for
  `test_origination_prior.py` and `test_parameter_layout.py`.

Verification:

- `python -m py_compile gpurec/core/parameter_layout.py gpurec/core/origination.py gpurec/api/_validation.py gpurec/api/uniform_chunked.py tests/unit/test_parameter_layout.py tests/unit/test_origination_prior.py tests/unit/test_origination_probs.py tests/unit/test_validation.py tests/unit/test_repository_hygiene.py`: passed.
- `CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests/unit/test_parameter_layout.py tests/unit/test_origination_prior.py tests/unit/test_origination_probs.py tests/unit/test_family_layout.py tests/unit/test_validation.py tests/unit/test_workflow.py::test_uniform_auto_int_rejects_bool_and_nonintegral_float tests/unit/test_workflow.py::test_uniform_auto_positive_int_preserves_unbounded_aliases tests/unit/test_workflow.py::test_uniform_chunked_rejects_bad_chunk_controls_before_device_or_io`: 151 passed.
- `PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests/integration/test_uniform_chunked_model.py -m "not slow"`: 3 passed, 1 deselected.
- `CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests/unit/test_repository_hygiene.py`: 83 passed.
- `CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider -m "unit and not gpu"`: 1007 passed, 1 skipped, 33 deselected.
- `python profiling/bench_uniform_forward_backward_pipeline.py --stats-only --strict-optimized-kernels --fams 1 --family-chunk-size 1 --max-wave-size 8192 --fixed-iters 2 --compare-unchunked-max-fams 0`: `strict_optimized_verdict pass`.
- `git diff --check`: passed.

### `827e792` - Consolidate resident no-grad evaluator path

Proposal coverage:

- `EVAL-01`: added the internal `_uniform_evaluator` resident no-grad boundary
  and routed `GeneReconModel.forward()` loss-only calls, shared-theta
  per-family diagnostics, genewise no-grad `nll_per_family()`, and
  `_evaluate_static_state(..., need_grad=False)` through it.
- Preserved the existing root-row loss-only path so no-grad evaluation does
  not retain full Pi/Pibar tensors.

Verification:

- Worker worktree: `python -m py_compile gpurec/api/model.py gpurec/api/_uniform_evaluator.py tests/unit/test_model_no_grad_evaluator.py`: passed.
- Worker worktree: `python -m pytest tests/unit/test_model_no_grad_evaluator.py -q`: 4 passed.
- Main combined gates are listed under `50280c5`.

### `8d9004d` - Consolidate export state solve path

Proposal coverage:

- `EVAL-01`: extended the internal resident evaluator boundary with a shared
  E/Pi solve helper and routed `reconciliation_state()` and `pi_matrix()`
  through the same solve surface used by resident no-grad evaluation.
- Preserved export order behavior via `original_order=True/False` and kept
  caller-owned side effects such as solver-stat recording and warm-cache
  clearing unchanged.

Verification:

- `python -m py_compile gpurec/api/model.py gpurec/api/_uniform_evaluator.py tests/integration/test_gene_recon_model.py`: passed during conflict resolution.
- Main combined gates are listed under `50280c5`.

### `7539687` - Add resident evaluator consolidation guards

Proposal coverage:

- `TEST-01`: strengthened resident evaluator characterization to cover
  scalar `torch.no_grad()` forward, wave-order export, `pi_matrix()` in both
  row orders, export tensor device/dtype, and post-export backward-gradient
  stability.

Verification:

- `PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests/integration/test_gene_recon_model.py::test_resident_evaluation_paths_remain_consistent tests/integration/test_gene_recon_model.py::test_batched_lbfgs_genewise_runs_one_polish_step tests/integration/test_uniform_chunked_model.py::test_chunked_uniform_chunk_subset_nll_and_gradient`: 5 passed.
- Main combined gates are listed under `50280c5`.

### `1e2f847` - Consolidate chunked read-only evaluation

Proposal coverage:

- `CHUNK-01`: split the chunked evaluator into a structured result core while
  preserving the existing tuple-returning `_evaluate_chunked_uniform()` API for
  autograd and `loss_and_grad()`.
- Routed `UniformChunkedReconModel.nll()` and `nll_per_family()` through a
  read-only wrapper that collects per-family output only when requested.

Verification:

- Worker worktree: `python -m py_compile gpurec/api/uniform_chunked.py tests/unit/test_optimization_workflow.py tests/integration/test_uniform_chunked_model.py`: passed.
- Worker worktree: `python -m pytest tests/unit/test_optimization_workflow.py -q`: 33 passed.
- Main combined gates are listed under `50280c5`.

### `50280c5` - Document resident evaluator CPU test

Proposal coverage:

- `TEST-01`: added `tests/unit/test_model_no_grad_evaluator.py` to the
  explicit CPU-unit manifest after introducing the resident evaluator helper.

Verification:

- `python -m py_compile gpurec/api/model.py gpurec/api/_uniform_evaluator.py gpurec/api/uniform_chunked.py tests/unit/test_model_no_grad_evaluator.py tests/unit/test_optimization_workflow.py tests/integration/test_gene_recon_model.py tests/integration/test_uniform_chunked_model.py`: passed.
- `PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests/unit/test_model_no_grad_evaluator.py tests/unit/test_optimization_workflow.py::test_uniform_chunked_read_only_helper_delegates_to_result_core tests/unit/test_optimization_workflow.py::test_uniform_chunked_nll_uses_read_only_chunked_result tests/unit/test_optimization_workflow.py::test_uniform_chunked_nll_per_family_uses_no_grad_chunked_diagnostic tests/unit/test_workflow.py::test_full_loss_for_theta_uses_streaming_contract_for_explicit_theta tests/unit/test_workflow.py::test_full_nll_per_family_delegates_to_genewise_streaming_helper tests/unit/test_workflow.py::test_full_nll_per_family_rejects_shared_theta_modes`: 11 passed.
- `CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests/unit/test_repository_hygiene.py`: 83 passed.
- `CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider -m "unit and not gpu"`: 1013 passed, 1 skipped, 33 deselected.
- `PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests/integration/test_gene_recon_model.py tests/integration/test_uniform_chunked_model.py`: 16 passed.
- `PYTHONDONTWRITEBYTECODE=1 python profiling/bench_uniform_forward_backward_pipeline.py --stats-only --strict-optimized-kernels --fams 1 --family-chunk-size 1 --max-wave-size 8192 --fixed-iters 2 --compare-unchunked-max-fams 0`: `strict_optimized_verdict pass`.
- `PYTHONDONTWRITEBYTECODE=1 python profiling/bench_uniform_forward_backward_pipeline.py --fams 4 --family-chunk-size 2 --max-wave-size 8192 --fixed-iters 2 --neumann-terms 2 --reps 1 --warmups 0 --compare-unchunked-max-fams 4 --fail-on-correctness-mismatch`: compare verdict pass, finite gradients, `total_median_ms 52.840`.
- `PYTHONDONTWRITEBYTECODE=1 python profiling/bench_uniform_forward_backward_pipeline.py --fams 1 --family-chunk-size 1 --max-wave-size 8192 --fixed-iters 2 --reps 1 --warmups 0 --compare-unchunked-max-fams 0`: finite gradients, `total_median_ms 798.486`.
- Full 1000-family benchmark attempt:
  `PYTHONDONTWRITEBYTECODE=1 python profiling/bench_uniform_forward_backward_pipeline.py --dataset tests/data/test_trees_1000 --fams 1000 --family-chunk-size auto --max-wave-size auto --fixed-iters 6 --neumann-terms 3 --warmups 1 --reps 3 --strict-optimized-kernels --compare-unchunked-max-fams 0` exited with code `-1` before emitting benchmark output, so it is not a valid pass/fail performance result for this log entry.
- `git diff --check`: passed.

### `a13e8fc` - Add Pi forward output intent helper

Proposal coverage:

- `PI-01`: added a private `_PiOutputIntent` contract so
  `Pi_wave_forward()` resolves legacy `return_original` / `return_root_rows`
  booleans into explicit output intent before deciding whether to emit root
  rows, original-order Pi, wave-ordered Pi, saved Pibar, and Pibar row maxima.
- Preserved the public function signature and returned dictionary keys,
  including the legacy `return_original=True, return_root_rows=True` case.

Verification:

- `python -m py_compile gpurec/core/forward.py tests/unit/test_forward_output_intent.py`: passed.
- `PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests/unit/test_forward_output_intent.py tests/unit/test_specieswise_uniform.py::test_specieswise_uniform_forward_root_rows_match_saved_state`: 5 passed.
- Main combined gates are listed under `86430d8`.

### `73cac0f` - Document Pi output intent CPU test

Proposal coverage:

- `TEST-01`: added `tests/unit/test_forward_output_intent.py` to the explicit
  CPU-unit manifest after introducing the Pi output intent helper.

Verification:

- `CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests/unit/test_repository_hygiene.py`: 83 passed.
- `git diff --check`: passed.
- Main combined gates are listed under `86430d8`.

### `d745f2e` - Add DTS layout contract helper

Proposal coverage:

- `DTS-01`: added a private CPU-testable DTS layout contract/parser for
  forward and backward kernel wrappers.
- Routed `_prepare_param()`, `_dts_layout_param_args()`, and
  `_dts_grad_layout()` through the shared contract while preserving existing
  Triton layout codes, strides, tensor normalization, and kernel launch
  semantics.
- Explicitly preserved and documented the current bare 1-D ambiguity:
  family-indexed forward treats length-`S` vectors as shared species, while
  retained backward treats 1-D tensors as family scalar rows when
  `family_idx` is present.

Verification:

- `python -m py_compile gpurec/core/kernels/_dts_layout_contract.py gpurec/core/kernels/dts_fused.py gpurec/core/kernels/wave_backward.py tests/unit/test_dts_layout_contract.py`: passed.
- `PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests/unit/test_dts_layout_contract.py tests/unit/test_repository_hygiene.py::test_dts_shape_precedence_is_documented_before_runtime_change tests/unit/test_parameter_layout.py tests/unit/test_extract_parameters.py`: 26 passed.
- Main combined gates are listed under `86430d8`.

### `3d6db8a` - Document DTS layout CPU test

Proposal coverage:

- `TEST-01`: added `tests/unit/test_dts_layout_contract.py` to the explicit
  CPU-unit manifest after introducing the DTS layout contract helper.

Verification:

- `git diff --check`: passed.
- Main combined gates are listed under `86430d8`.

### `6b59e0b` - Characterize backward auto-wrap layout

Proposal coverage:

- `MODE-02`: extracted current `Pi_wave_backward()` `family_idx=None`
  auto-wrap behavior into `_auto_wrap_backward_inputs()` without changing
  gradient math.
- Added CPU helper-level characterization for shared `G=1` wrapping, explicit
  family-index preservation, and `G == S` ambiguity before any removal of the
  local auto-wrap policy.
- Updated the opportunity index with the characterization gate.

Verification:

- `python -m py_compile gpurec/core/backward.py tests/unit/test_core_backward.py`: passed.
- `PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests/unit/test_core_backward.py tests/unit/test_extract_parameters.py tests/unit/test_parameter_layout.py tests/unit/test_dts_layout_contract.py`: 30 passed.
- Main combined gates are listed under `86430d8`.

### `e4a5dd6` - Add gradient accumulator helper

Proposal coverage:

- `BWD-03`: added a CPU-testable `GradientAccumulator` over the existing
  `ParameterLayout` contract for public theta-shaped gradient accumulation.
- Routed only `GeneReconModel._stream_full_batches()` through the helper,
  preserving hot CUDA backward scatter code and chunked Pi-backward local
  accumulation for later dedicated gates.

Verification:

- `python -m py_compile gpurec/core/gradient_accumulator.py gpurec/api/model.py tests/unit/test_gradient_accumulator.py`: passed.
- `PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests/unit/test_gradient_accumulator.py tests/unit/test_parameter_layout.py tests/unit/test_workflow.py::test_full_loss_for_theta_uses_streaming_contract_for_explicit_theta tests/integration/test_gene_recon_model.py::test_memory_safe_resident_batches_match_resident_and_slice`: 29 passed.
- Main combined gates are listed under `86430d8`.

### `86430d8` - Document gradient accumulator CPU test

Proposal coverage:

- `TEST-01`: added `tests/unit/test_gradient_accumulator.py` to the explicit
  CPU-unit manifest after introducing the gradient accumulator helper.

Verification:

- `CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider -m "unit and not gpu"`: 1036 passed, 1 skipped, 33 deselected.
- `PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests/kernels/test_wave_step_uniform_forward_kernel.py tests/integration/test_gene_recon_model.py tests/integration/test_uniform_chunked_model.py tests/unit/test_specieswise_uniform.py::test_specieswise_uniform_forward_root_rows_match_saved_state tests/unit/test_specieswise_uniform.py::test_constant_specieswise_matches_global_loss_and_gradient_semantics`: 23 passed.
- `PYTHONDONTWRITEBYTECODE=1 python profiling/bench_uniform_forward_backward_pipeline.py --stats-only --strict-optimized-kernels --fams 1 --family-chunk-size 1 --max-wave-size 8192 --fixed-iters 2 --compare-unchunked-max-fams 0`: `strict_optimized_verdict pass`.
- `PYTHONDONTWRITEBYTECODE=1 python profiling/bench_uniform_forward_backward_pipeline.py --dataset tests/data/test_trees_1000 --fams 8 --family-chunk-size 2 --max-wave-size 32768 --fixed-iters 6 --reps 3 --warmups 1 --compare-unchunked-max-fams 8 --fail-on-correctness-mismatch --strict-optimized-kernels`: compare verdict pass, finite gradients, `strict_optimized_verdict pass`, `total_median_ms 100.248`, `max_peak_gib 0.445`.
- Full 1000-family benchmark attempt:
  `PYTHONDONTWRITEBYTECODE=1 python profiling/bench_uniform_forward_backward_pipeline.py --dataset tests/data/test_trees_1000 --fams 1000 --family-chunk-size auto --max-wave-size auto --fixed-iters 6 --warmups 1 --reps 3 --strict-optimized-kernels --compare-unchunked-max-fams 0` exited with code `-1` before emitting benchmark output, so it is not a valid pass/fail performance result for this log entry.
- `git diff --check`: passed.

### `d8b5fb1` - Add uniform benchmark preflight progress

Proposal coverage:

- `BWD-01`, `BWD-02`, and `SCHED-01` all require benchmark evidence before
  runtime alternatives can be removed.  The 1000-family benchmark had been
  exiting with code `-1` before any output, so it could not answer those
  proposals.
- Added opt-in `--progress-jsonl` records and `--preflight-only` /
  `--setup-only` mode to the benchmark harness.  Normal benchmark output is
  unchanged unless progress is requested.
- Progress records now identify gene selection, dataset construction,
  chunk-policy selection, per-chunk layout construction, optimized-path status,
  warmups, and timed reps.

Verification:

- `python -m py_compile profiling/bench_uniform_forward_backward_pipeline.py tests/unit/test_bench_uniform_forward_backward_pipeline.py`: passed.
- `PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests/unit/test_bench_uniform_forward_backward_pipeline.py tests/unit/test_repository_hygiene.py::test_documented_uniform_pipeline_benchmark_help_imports_current_api`: 5 passed.
- `python profiling/bench_uniform_forward_backward_pipeline.py --help`: passed.

### `7ed00a2` - Strengthen env flag ownership hygiene

Proposal coverage:

- `ENV-01`: strengthened the runtime-surface plan guard so every package-read
  `GPUREC_*` variable has a manifest owner, and so user-facing flags stay
  limited to `GPUREC_BACKTRACK_BIN`, `GPUREC_ALERAX_COMPAT`,
  `GPUREC_MEMORY_POLICY_FRACTION`, and `GPUREC_MEMORY_POLICY_RESERVE_GIB`.
- No flags were removed or behavior changed; prototype, tuning, and internal
  production flags remain documented as non-user-facing surfaces until they
  have typed replacements or benchmark-owned CLI knobs.

Verification:

- `python -m py_compile tests/unit/test_repository_hygiene.py`: passed.
- `PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests/unit/test_repository_hygiene.py -k 'environment or cuda_prototype or retired_leaf_hit'`: 6 passed, 77 deselected.
- `PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests/unit/test_bench_uniform_forward_backward_pipeline.py tests/unit/test_repository_hygiene.py`: 87 passed after the benchmark test was fixed to use a pytest-managed temp path.

### `ee7a328` - Fix benchmark preflight test temp path

Proposal coverage:

- `TEST-01`: repaired the new benchmark preflight unit test so it uses
  `tmp_path` instead of a hard-coded `/tmp` cache path.

Verification:

- `python -m py_compile tests/unit/test_bench_uniform_forward_backward_pipeline.py tests/unit/test_repository_hygiene.py`: passed.
- `PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests/unit/test_bench_uniform_forward_backward_pipeline.py tests/unit/test_repository_hygiene.py`: 87 passed.

### `e441cbd` - Add backward self-loop policy characterization

Proposal coverage:

- `BWD-01`: extracted CPU-testable private helpers for current native CUDA
  self-loop option parsing, per-wave eligibility/routing, and optional-failure
  fallback behavior.
- Added executable guards for default `auto` modes, required-vs-optional
  modes, split vs no-split selection, eligibility gates, narrow optional
  exception handling, and disabling only the failed optional backend.
- No self-loop backend was deleted.  The retained Triton path and native CUDA
  prototype paths remain until gradient parity and retained-benchmark evidence
  justify a production choice.

Verification:

- `python -m py_compile gpurec/core/backward.py tests/unit/test_backward_self_loop_policy.py tests/unit/test_core_backward.py`: passed.
- `PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests/unit/test_backward_self_loop_policy.py tests/unit/test_core_backward.py`: 26 passed.

### `36bd0c0` - Characterize backward pruning policy

Proposal coverage:

- `BWD-02`: added `gpurec/core/backward_pruning_policy.py` to resolve current
  active-mask pruning flags in one CPU-testable policy object.
- Routed `Pi_wave_backward()` through that policy without changing the CUDA
  kernels.  The default still avoids host-side all-inactive wave readbacks
  while passing device active masks when pruning is enabled.
- Added guards for default no-CPU-pruning behavior, pruning-disabled exact-zero
  masks on the CPU-pruning branch, threshold-boundary behavior, inactive-wave
  accounting counters, and `GPUREC_DTS_SKIP_INACTIVE_PIBAR_ZERO` truth rules.
- The runtime CPU-pruning branch was not removed because sparse/dense
  active-mask benchmarks are still needed.

Verification:

- `python -m py_compile gpurec/core/backward.py gpurec/core/backward_pruning_policy.py tests/unit/test_backward_pruning_policy.py tests/unit/test_backward_self_loop_policy.py tests/unit/test_core_backward.py`: passed.
- `PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests/unit/test_backward_pruning_policy.py tests/unit/test_backward_self_loop_policy.py tests/unit/test_core_backward.py`: 32 passed.

### `d92f15d` - Characterize scheduler candidate selection

Proposal coverage:

- `SCHED-01`: extracted the current non-leaf scheduler candidate selection into
  `_select_nonleaf_schedule_candidate()` so tests can identify which policy wins
  before any deletion.
- Added representative guards proving `forward`, `deadline`, and
  `coffman_graham` can each be the selected policy.  This means the current
  documents do not justify deleting the alternative schedulers yet.
- No scheduler candidate was removed; choosing one production scheduler still
  requires benchmark evidence against the current multi-candidate selector.

Verification:

- `python -m py_compile gpurec/core/batching.py gpurec/api/_family_layout.py tests/unit/test_global_wave_scheduler.py`: passed.
- `CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests/unit/test_global_wave_scheduler.py tests/unit/test_family_layout.py`: 61 passed.

### `8bbd428` - Instrument benchmark dataset preprocessing progress

Proposal coverage:

- `BWD-01`, `BWD-02`, and `SCHED-01`: extended the benchmark progress stream
  into `GeneDataset` preprocessing so failed 1000-family setup runs identify
  whether the stop happens in species cache/hash handling, family hash/cache
  scanning, batch C++ preprocessing, cache validation, or later CUDA setup.
- Added a private no-op `_preprocess_progress` hook to `GeneDataset`; normal
  API behavior and default benchmark output are unchanged.
- The latest 1000-family preflight now reaches the missing-family C++
  preprocessing call with 992 cache-missing families.  It still does not
  produce a timed benchmark result.

Verification:

- `python -m py_compile profiling/bench_uniform_forward_backward_pipeline.py gpurec/core/model.py tests/unit/test_bench_uniform_forward_backward_pipeline.py tests/unit/test_alerax_family_input.py`: passed.
- `PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests/unit/test_bench_uniform_forward_backward_pipeline.py tests/unit/test_alerax_family_input.py`: 41 passed.

### `da2715a` - Consolidate resident autograd solve path

Proposal coverage:

- `EVAL-01`: moved the resident E/Pi solve result and solve helper into the
  autograd module and routed `_GeneReconFunction.forward()` through the shared
  solve boundary.
- Preserved the existing warm-start behavior and kept the no-grad/root-row
  evaluator boundary in `_uniform_evaluator.py`.
- Autograd backward and streaming gradient evaluation remain separate; this
  commit only removes the duplicated resident forward solve path.

Verification:

- `python -m py_compile gpurec/api/autograd.py gpurec/api/model.py gpurec/api/_uniform_evaluator.py tests/unit/test_model_no_grad_evaluator.py`: passed.
- `PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests/unit/test_model_no_grad_evaluator.py tests/unit/test_workflow.py::test_full_loss_for_theta_uses_streaming_contract_for_explicit_theta`: 6 passed.

### `e6fb2c9` - Guard chunked bf16 gradient path

Proposal coverage:

- `CHUNK-01`: added a shared dtype guard at the chunked result boundary so
  gradient-producing chunked evaluation rejects `torch.bfloat16` before any
  chunk work starts.
- Read-only chunked `bfloat16` evaluation remains allowed.  The guard records
  the current retained-backward limitation instead of letting a later CUDA path
  fail less clearly.

Verification:

- `python -m py_compile gpurec/api/uniform_chunked.py tests/unit/test_optimization_workflow.py`: passed.
- `PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests/unit/test_optimization_workflow.py tests/unit/test_workflow.py -k 'uniform_chunked and not constructors and not factories'`: 36 passed, 443 deselected.

### `fb23ab1` - Guard C++ preprocess pybind surface

Proposal coverage:

- `CPP-01` and `CPP-02`: added an exact pybind export manifest for
  `gpurec/core/cpp/preprocess.cpp`.
- Classified direct exports as production, compatibility, or diagnostic before
  deletion.  `preprocess_multiple_families` remains production-owned; legacy
  `preprocess` and direct stat exports still need deprecation or replacement
  evidence before removal.
- No C++ behavior changed.

Verification:

- `python -m py_compile tests/unit/test_repository_hygiene.py`: passed.
- `CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests/unit/test_repository_hygiene.py::test_preprocess_cpp_pybind_exports_match_classified_manifest tests/unit/test_repository_hygiene.py::test_runtime_surface_plan_documents_scheduler_and_pybind_ownership tests/unit/test_repository_hygiene.py::test_cpp_wave_stat_exports_validate_positive_max_wave_size`: 3 passed.
- `git diff --check HEAD^ HEAD`: passed.

### `119999e` - Guard script and test surface ownership

Proposal coverage:

- `SCRIPT-01`: added an executable guard requiring every tracked
  `scripts/*.py` and `scripts/*.R` entry to be listed in `scripts/README.md`
  with an allowed ownership status.
- `TEST-01`: documented and guarded legacy-script test ownership, including
  cleanup expectations and deletion notes for white-box internals.
- Marked historical AleRax validation docs as using untracked generated
  fixtures so path-reference hygiene can distinguish archived output from
  tracked test fixtures.

Verification:

- `python -m py_compile tests/unit/test_repository_hygiene.py tests/unit/test_legacy_scripts.py scripts/check_release_metadata.py scripts/compare_backtracking_alerax_events.py scripts/export_hogenom_rates_from_checkpoint.py profiling/evaluate_hogenom_alerax_rates.py profiling/bench_uniform_forward_backward_pipeline.py`: passed.
- `PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests/unit/test_legacy_scripts.py tests/unit/test_repository_hygiene.py`: 123 passed.
- `PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests/unit/test_release_metadata.py -k 'not tests_readme_explicit_cpu_unit_paths_match_marker_gate'`: 42 passed, 1 skipped, 1 deselected.

### `892229f` - Record second-wave simplification progress

Proposal coverage:

- `TEST-01`: recorded the second-wave commits, verification commands, and
  unresolved benchmark blocker in this execution log so proposal documents and
  executable work stay connected.
- No runtime behavior changed.

Verification:

- `PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests/unit/test_repository_hygiene.py tests/unit/test_release_metadata.py -k 'not tests_readme_explicit_cpu_unit_paths_match_marker_gate'`: 128 passed, 1 skipped, 1 deselected.
- `git diff --check`: passed.

### `9f1fb98` - Batch cached family preprocessing misses

Proposal coverage:

- `BWD-01`, `BWD-02`, and `SCHED-01`: split cache-missing family
  preprocessing into bounded private batches before the C++ call, with default
  batch size 64.
- Added progress events for each missing-family batch start, batch completion,
  and incremental cache-write completion.
- Family cache entries are now validated and saved after each batch, so a
  failed large preflight keeps completed batch work instead of losing the whole
  missing-family set.
- Public dataset behavior is unchanged; the batch-size override is private and
  exists for tests/diagnostics.

Verification:

- `python -m py_compile gpurec/core/model.py profiling/bench_uniform_forward_backward_pipeline.py tests/unit/test_alerax_family_input.py tests/unit/test_bench_uniform_forward_backward_pipeline.py`: passed.
- `PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests/unit/test_alerax_family_input.py tests/unit/test_bench_uniform_forward_backward_pipeline.py`: 43 passed.
- 96-family preflight:
  `PYTHONDONTWRITEBYTECODE=1 timeout 240s python profiling/bench_uniform_forward_backward_pipeline.py --dataset tests/data/test_trees_1000 --fams 96 --family-chunk-size auto --max-wave-size auto --fixed-iters 6 --preflight-only --progress-jsonl --strict-optimized-kernels --compare-unchunked-max-fams 0` exited 0.  It emitted batch progress for 88 cache misses across 2 batches, reached `dataset_loaded`, built 4 chunks, and emitted `preflight_done`.
- 1000-family preflight in the worker confirmed batching got past the original
  single-call blocker: batch 0 completed and cached, and batch 1 completed C++
  preprocessing.  The run then failed while saving `family_000102` with an
  iostream error followed by `no space left on device`; this is now a
  cache-storage capacity blocker on the local filesystem, not evidence about
  runtime kernels.

### `68778dd` - Make chunked Pi backward accumulation explicit

Proposal coverage:

- `CHUNK-01` and `BWD-03`: added `StructuredGradientAccumulator` for fixed
  schema tensor/counter gradient dictionaries.
- Routed chunked `Pi_wave_backward` result accumulation through that helper
  with explicit tensor and counter keys, replacing an ad hoc dict merge in the
  chunked evaluator.
- Hot CUDA backward scatter behavior is unchanged; this only clarifies the
  model-boundary accumulation contract for chunked gradient paths.

Verification:

- `python -m py_compile gpurec/core/gradient_accumulator.py gpurec/api/uniform_chunked.py tests/unit/test_gradient_accumulator.py tests/unit/test_optimization_workflow.py`: passed.
- `PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests/unit/test_gradient_accumulator.py tests/unit/test_optimization_workflow.py`: 52 passed.

### `d3813d8` - Consolidate resident gradient evaluator boundary

Proposal coverage:

- `EVAL-01`: added shared resident gradient-forward and implicit-gradient
  helpers in `autograd.py`.
- Routed both autograd backward and `_evaluate_static_state(...,
  need_grad=True)` through the shared helper boundary, removing duplicated
  implicit-gradient call construction.
- Resident no-grad, export-state, autograd forward, autograd backward, and
  static-state gradient paths now use explicit resident evaluator boundaries.

Verification:

- `python -m py_compile gpurec/api/autograd.py gpurec/api/model.py gpurec/api/_uniform_evaluator.py tests/unit/test_model_no_grad_evaluator.py tests/unit/test_workflow.py tests/integration/test_gene_recon_model.py`: passed.
- `PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests/unit/test_model_no_grad_evaluator.py tests/unit/test_workflow.py::test_full_loss_for_theta_uses_streaming_contract_for_explicit_theta`: 7 passed.
- Integration gradient checks in the worker skipped where `tests/data/test_trees_1000`
  was absent; the main checkout has the dataset and broader gates are tracked
  below.

### `1ca011a` - Record third-wave simplification progress

Proposal coverage:

- `TEST-01`: recorded the third-wave runtime and evaluator changes, validation
  commands, and updated 1000-family blocker status.
- No runtime behavior changed.

Verification:

- `git diff --check`: passed.

### `17f7d75` - Batch uncached family preprocessing

Proposal coverage:

- `BWD-01`, `BWD-02`, and `SCHED-01`: added an explicit benchmark setup mode
  that can disable preprocess cache reads/writes while still batching C++
  preprocessing.  This removes the local `/tmp` cache-capacity dependency from
  large preflight attempts.
- Added `--no-preprocess-cache` and `--cache-dir none` to the benchmark.  The
  default cached behavior is unchanged.
- Added uncached preprocessing progress events:
  `uncached_preprocess_start`, `uncached_preprocess_batch_start`,
  `uncached_preprocess_batch_done`, and `uncached_preprocess_done`.
- Materializes uncached family payloads per batch so completed raw batch output
  can be released before the next batch.

Verification:

- `python -m py_compile gpurec/core/model.py profiling/bench_uniform_forward_backward_pipeline.py tests/unit/test_alerax_family_input.py tests/unit/test_bench_uniform_forward_backward_pipeline.py`: passed.
- `PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests/unit/test_alerax_family_input.py tests/unit/test_bench_uniform_forward_backward_pipeline.py`: 46 passed.
- `git diff --check HEAD^ HEAD`: passed.
- 1000-family no-cache preflight attempts in the worker did not reach
  `dataset_loaded`.  With cache disabled, a 64-family uncached batch size
  completed 704 families before exit `-1`; the 16-family default completed 752
  families before exit `-1`, with no Python traceback and no `/tmp` cache
  capacity failure.  The next setup blocker is therefore not cache writes; it
  is native preprocessing or resident memory scale around the 750-family mark.

### `5a7b9de` - Guard simplification proposal execution coverage

Proposal coverage:

- `TEST-01`: added a repository hygiene guard that parses every proposal ID
  from the simplification opportunity index and fails if the execution log no
  longer mentions one.  This keeps the docs tied to the actual work queue
  instead of letting proposal IDs silently fall out of the log.
- No runtime behavior changed.

Verification:

- `python -m py_compile tests/unit/test_repository_hygiene.py`: passed.
- `PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests/unit/test_repository_hygiene.py::test_simplification_execution_log_mentions_every_index_proposal tests/unit/test_repository_hygiene.py::test_simplification_opportunity_index_is_mapped_and_gate_oriented`: 2 passed.

### `d9fa443` - Thread prepared origination priors through models

Proposal coverage:

- `ORIG-01`: routed `GeneReconModel` and `UniformChunkedReconModel`
  construction through `prepare_origination_prior` instead of calling the raw
  probability helper directly.
- Stored the prepared prior object on both model surfaces while preserving the
  existing `origination_probs` buffer used by resident likelihood/evaluator
  paths.
- Added constructor-level tests that stub CUDA/preprocessing work and assert
  that family-specific `OriginationPrior` inputs normalize once, stay attached
  to the model, and flow into the existing static/chunked state boundaries.

Verification:

- `python -m py_compile gpurec/api/model.py gpurec/api/uniform_chunked.py tests/unit/test_origination_prior.py`: passed.
- `PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests/unit/test_origination_prior.py tests/unit/test_origination_probs.py tests/unit/test_model_no_grad_evaluator.py`: 28 passed.
- `PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' python -m pytest -q -p no:cacheprovider tests/unit/test_optimization_workflow.py tests/integration/test_gene_recon_model.py tests/integration/test_uniform_chunked_model.py`: 37 passed, 16 skipped.
- `git diff --check -- gpurec/api/model.py gpurec/api/uniform_chunked.py tests/unit/test_origination_prior.py`: passed.

### `b2c4709` - Instrument 1000-family preflight progress

Proposal coverage:

- `BWD-01`, `BWD-02`, and `SCHED-01`: added benchmark progress telemetry for
  RSS, peak RSS, disk free space, CUDA allocator state, CUDA driver memory,
  and selected uncached preprocessing windows.  This makes benchmark setup
  failures diagnosable before any timed runtime comparison is trusted.
- Added `--uncached-preprocess-batch-size` so the benchmark driver can vary
  the uncached preprocessing batch size without changing code.
- The benchmarker ran no-cache preflight probes on the 1000-family fixture:
  256 families passed with max RSS about 44 GiB, 512 passed with max RSS about
  85 GiB, and selected windows around families 704, 736, and 752 passed.  The
  full-prefix failure is therefore most consistent with host RAM exhaustion
  from retaining/materializing all preprocessed families, not a single bad
  family near 752.
- Cached mode is still blocked locally by cache-write disk pressure; the old
  progress log ends in `torch.save(...): RuntimeError: basic_ios::clear:
  iostream error`.

Verification:

- `python -m py_compile profiling/bench_uniform_forward_backward_pipeline.py tests/unit/test_bench_uniform_forward_backward_pipeline.py`: passed.
- `PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests/unit/test_bench_uniform_forward_backward_pipeline.py tests/unit/test_alerax_family_input.py`: 48 passed.
- Benchmarker worktree validation also passed the same 48-test gate.
- No valid 1000-family timed benchmark exists yet.  The current blocker is
  setup memory/disk capacity, so benchmark-sensitive deletions remain blocked.

### Current combined gates after no-cache setup and diagnosis work

Verification:

- `PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests/unit/test_alerax_family_input.py tests/unit/test_bench_uniform_forward_backward_pipeline.py tests/unit/test_gradient_accumulator.py tests/unit/test_optimization_workflow.py tests/unit/test_model_no_grad_evaluator.py tests/unit/test_workflow.py::test_full_loss_for_theta_uses_streaming_contract_for_explicit_theta`: 102 passed.
- `PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests/unit/test_bench_uniform_forward_backward_pipeline.py tests/unit/test_repository_hygiene.py::test_tests_use_pytest_managed_temporary_paths`: 9 passed after replacing a literal `/tmp` path in the benchmark cache-disable test with `tmp_path`.
- `PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests/unit/test_repository_hygiene.py tests/unit/test_release_metadata.py -k 'not tests_readme_explicit_cpu_unit_paths_match_marker_gate'`: 128 passed, 1 skipped, 1 deselected.
- `CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider -m "unit and not gpu" -k 'not tests_readme_explicit_cpu_unit_paths_match_marker_gate'`: 1087 passed, 1 skipped, 41 deselected.  The excluded manifest test is blocked locally by untracked dated-model test files, which are not part of this simplification branch.
- `PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests/kernels/test_wave_step_uniform_forward_kernel.py tests/integration/test_gene_recon_model.py tests/integration/test_uniform_chunked_model.py tests/unit/test_specieswise_uniform.py::test_specieswise_uniform_forward_root_rows_match_saved_state tests/unit/test_specieswise_uniform.py::test_constant_specieswise_matches_global_loss_and_gradient_semantics`: 23 passed.
- `PYTHONDONTWRITEBYTECODE=1 python profiling/bench_uniform_forward_backward_pipeline.py --stats-only --strict-optimized-kernels --fams 1 --family-chunk-size 1 --max-wave-size 8192 --fixed-iters 2 --compare-unchunked-max-fams 0`: `strict_optimized_verdict pass`.
- `PYTHONDONTWRITEBYTECODE=1 python profiling/bench_uniform_forward_backward_pipeline.py --dataset tests/data/test_trees_1000 --fams 8 --family-chunk-size 2 --max-wave-size 32768 --fixed-iters 6 --reps 3 --warmups 1 --compare-unchunked-max-fams 8 --fail-on-correctness-mismatch --strict-optimized-kernels`: compare verdict pass, finite gradients, `strict_optimized_verdict pass`, `total_median_ms 102.815`, `max_peak_gib 0.445`.
- 1000-family preflight status after batching: no valid timed benchmark yet.
  The original single C++ call failure and `/tmp` cache-write capacity failure
  have both been bypassed by batching and explicit cache disabling.  The
  no-cache full-prefix failure is now most likely host RAM pressure:
  512-family setup already peaks around 85 GiB RSS, selected windows past 752
  pass, and extrapolating the prefix slope reaches the host memory limit before
  1000 families.  Do not use this as performance evidence.

## Active Work Queue

1. `EVAL-01` and `CHUNK-01`: continue consolidation for autograd and
   gradient-producing paths.  Resident no-grad, export-state, resident
   autograd forward solve, resident implicit-gradient calls, static-state
   gradient evaluation, and chunked read-only paths now share explicit
   evaluator boundaries.  `UniformChunkedReconModel` still owns separate
   chunk setup/stats and per-chunk forward/backward orchestration.
2. `ORIG-01`: the public model constructors now prepare and retain
   `PreparedOriginationPrior` objects while preserving the existing
   `origination_probs` tensor boundary.  The remaining step is to thread that
   prepared object deeper into resident evaluator/static-state boundaries
   without breaking the current prepared-tensor trust boundary.
3. `PI-01`, `MODE-02`, `BWD-03`, and `DTS-01`: continue from the explicit
   contracts now in place.  Pi output intent, DTS layout parsing,
   backward auto-wrap characterization, and model-boundary gradient
   accumulation now have first-step guards.  Removing backward auto-wrap and
   routing hot CUDA scatter paths still require full gradient/parity gates.
4. `BWD-01`, `BWD-02`, `ENV-01`, and `SCHED-01`: first characterization and
   ownership guards are in place.  Do not remove self-loop backends,
   CPU-pruning branches, env toggles, or scheduler alternatives while the
   1000-family benchmark still lacks a valid timed run.  Resolve the
   setup memory/disk blocker, likely by avoiding all-family resident
   materialization for preflight or by using a larger cache/storage target,
   before treating sparse/dense runtime benchmark results as authoritative.
5. `CPP-01`, `CPP-02`, `SCRIPT-01`, and `TEST-01`: continue pruning and
   splitting only after each surface has an owner, deprecation path, or
   replacement behavior test.  The first ownership guards are now executable;
   actual deletion remains separate work.

## Recent Subagent Assignments

- Benchmark preflight worker:
  `profiling/bench_uniform_forward_backward_pipeline.py` and
  `tests/unit/test_bench_uniform_forward_backward_pipeline.py`, integrated in
  `d8b5fb1` and fixed in `ee7a328`.
- Environment ownership worker:
  `docs/runtime-surface-pruning-plan-2026-05-21.md` and
  `tests/unit/test_repository_hygiene.py`, integrated in `7ed00a2`.
- Self-loop policy worker:
  `gpurec/core/backward.py` and
  `tests/unit/test_backward_self_loop_policy.py`, integrated in `e441cbd`.
- Pruning policy worker:
  `gpurec/core/backward_pruning_policy.py`, `gpurec/core/backward.py`, and
  `tests/unit/test_backward_pruning_policy.py`, integrated in `36bd0c0`.
- Scheduler policy worker:
  `gpurec/core/batching.py` and `tests/unit/test_global_wave_scheduler.py`,
  integrated in `d92f15d`.
- Read-only proposal auditor: confirmed the safe integration order and flagged
  backend deletion, CPU-pruning removal, env deletion, and scheduler
  simplification as unsafe without benchmark evidence.
- Dataset-preprocess progress worker:
  `gpurec/core/model.py`,
  `profiling/bench_uniform_forward_backward_pipeline.py`, and benchmark
  progress tests, integrated in `8bbd428`.
- Resident autograd solve worker:
  `gpurec/api/autograd.py`, `gpurec/api/_uniform_evaluator.py`, and resident
  evaluator tests, integrated in `da2715a`.
- Chunked gradient boundary worker:
  `gpurec/api/uniform_chunked.py` and optimization workflow tests, integrated
  in `e6fb2c9`.
- C++ pybind surface worker:
  `gpurec/core/cpp/preprocess.cpp`, runtime-surface docs, and repository
  hygiene tests, integrated in `fb23ab1`.
- Script/test surface worker:
  `tests/unit/test_repository_hygiene.py`, `tests/README.md`, archived AleRax
  docs, and runtime-surface docs, integrated in `119999e`.
- Preprocess batching worker:
  `gpurec/core/model.py` and cache/progress tests, integrated in `9f1fb98`.
- Chunked accumulator worker:
  `gpurec/core/gradient_accumulator.py`, `gpurec/api/uniform_chunked.py`, and
  gradient accumulator tests, integrated in `68778dd`.
- Resident gradient evaluator worker:
  `gpurec/api/autograd.py`, `gpurec/api/model.py`, and resident evaluator
  tests, integrated in `d3813d8`.
- No-cache preprocess batching worker:
  `gpurec/core/model.py`,
  `profiling/bench_uniform_forward_backward_pipeline.py`, and benchmark/cache
  tests, integrated in `17f7d75`.
- Proposal coverage guard:
  `tests/unit/test_repository_hygiene.py`, integrated in `5a7b9de`.
- Origination-prior threading:
  `gpurec/api/model.py`, `gpurec/api/uniform_chunked.py`, and origination
  prior tests, integrated in `d9fa443`.
- 1000-family benchmark diagnosis worker:
  `profiling/bench_uniform_forward_backward_pipeline.py` and benchmark
  progress tests, integrated in `b2c4709`.

Future parallel workers should continue to use separate git worktrees for
larger runtime changes.
