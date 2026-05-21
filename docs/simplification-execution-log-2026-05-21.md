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

## Active Work Queue

1. `EVAL-01` and `CHUNK-01`: consolidate resident, no-grad, export, autograd,
   and chunked evaluation paths behind one evaluator after the gates above.
2. `PI-01`, `MODE-02`, `BWD-03`, and `DTS-01`: refactor Pi/backward/DTS
   contracts only after the explicit layout contract exists.
3. `BWD-01`, `BWD-02`, `ENV-01`, and `SCHED-01`: remove runtime alternatives
   only after benchmark gates show the retained path is not regressed.
4. `CPP-01`, `CPP-02`, `SCRIPT-01`, and `TEST-01`: continue pruning and
   splitting only after each surface has an owner, deprecation path, or
   replacement behavior test.

## Active Subagent Assignments

- Resident characterization worker: `tests/integration/test_gene_recon_model.py`.
- Chunked characterization worker: `tests/integration/test_uniform_chunked_model.py`.
- Scheduler/preprocess characterization worker:
  `tests/unit/test_global_wave_scheduler.py` and
  `tests/unit/test_alerax_family_input.py`.
- Benchmark-harness worker: `profiling/bench_uniform_forward_backward_pipeline.py`.
- Proposal auditor: read-only status matrix and dependency audit.

These assignments completed in commit `04d0aab`; future parallel workers should
use separate git worktrees for larger runtime changes.
