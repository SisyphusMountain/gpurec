# Repo-Wide Audit, 2026-05-21

This note records a read-only audit pass over the tracked repository state on
branch `lean-fast-path`.  It is intentionally documentation-only: no runtime
logic was changed before recording the findings.

## Scope And Evidence

- Tracked scope: `git ls-files | wc -l` reported 134 files.
- Source-like size: `git ls-files '*.py' '*.cpp' '*.hpp' '*.rs' '*.R' | xargs wc -l`
  reported 43,009 lines.
- Test inventory: `pytest --collect-only -q` collected 823 tests.
- Coverage tooling: importing `coverage` failed with
  `ModuleNotFoundError: No module named 'coverage'`, so this pass used static
  evidence, test references, and subagent review instead of a line coverage
  report.
- Static docstring scan: 510 Python symbols were found under `gpurec/`; 138
  public classes/functions lacked docstrings.

Five read-only subagents inspected disjoint file groups:

- Core Python runtime: `gpurec/core/*.py` except kernels and C++ sources.
- Kernel and C++ runtime: `gpurec/core/kernels/*` and `gpurec/core/cpp/*`.
- Public API, workflow, CLI, optimization, and RecPhyloXML modules.
- Rust backtracking, configs, examples, notebooks, profiling, and scripts.
- Tests, documentation, package metadata, workflow YAML, examples metadata, and
  release hygiene files.

## Static Signals

Largest production symbols by source length:

- `gpurec/api/model.py:832` `GeneReconModel`, 992 lines.
- `gpurec/core/backward.py:27` `Pi_wave_backward`, 581 lines.
- `gpurec/api/uniform_chunked.py:721` `UniformChunkedReconModel`, 458 lines.
- `gpurec/workflow/optimize.py:306` `OptimizationRunner`, 439 lines.
- `gpurec/core/kernels/wave_backward.py:1033`
  `_dts_cross_backward_accum_kernel`, 416 lines.
- `gpurec/optimization/batched_lbfgs.py:25` `BatchedLBFGS`, 384 lines.
- `gpurec/core/forward.py:53` `Pi_wave_forward`, 345 lines.

Modules with weak direct test references in the static scan include
`gpurec/core/kernels/dts_fused.py`, `gpurec/core/kernels/pibar_vjp_cuda.py`,
`gpurec/core/kernels/wave_backward.py`,
`gpurec/core/kernels/wave_backward_cuda.py`,
`gpurec/core/extract_parameters.py`, and the profiling scripts.  Some are
covered indirectly through higher-level GPU tests, but direct branch-level
evidence is thin.

## Findings

### Runtime Contracts And Algorithm Edges

1. Small-species backward behavior is unclear.  `Pi_wave_backward` rejects
   non-CUDA, unsupported dtypes, and `S <= 256` at `gpurec/core/backward.py:100`
   and `gpurec/core/backward.py:104`.  GPU backward coverage uses
   `test_trees_1000`, while user docs and examples show tiny CUDA configs.
   Document this as an intentional limitation or add a tested fallback before
   changing backward logic.

2. `ancestors_T` is optional by signature but required in practice.
   `E_step(..., ancestors_T=None)` and `E_fixed_point(..., ancestors_T=None)`
   expose defaults at `gpurec/core/likelihood.py:32` and
   `gpurec/core/likelihood.py:105`, but `_uniform_ancestor_sum` immediately uses
   `expE_2d @ ancestors_T` at `gpurec/core/likelihood.py:22`.  Existing tests
   pass `static.ancestors_T`, so the default path is untested.

3. Direct duplicate `family_names` can collapse preprocessing data.  The direct
   `GeneDataset` constructor checks only length at `gpurec/core/model.py:545`,
   then builds dictionaries keyed by family name at `gpurec/core/model.py:569`
   and cache maps at `gpurec/core/model.py:669`.  AleRax file parsing rejects
   duplicate names, but the direct constructor path lacks equivalent coverage.

4. `clade_budget` is a packing target, not a hard cap.  In `clade_first_fit`
   and `depth_first_fit`, a single over-budget family still starts its own
   chunk at `gpurec/core/batch_planning.py:221` and
   `gpurec/core/batch_planning.py:287`.  Tests cover under-budget fixtures, but
   not oversized-family semantics.

5. Adaptive root trace behavior is under-specified.  `Pi_wave_forward`
   preallocates `root_logsumexp_trace` for all fixed iterations at
   `gpurec/core/forward.py:223`, can return early at
   `gpurec/core/forward.py:329`, and returns the full trace at
   `gpurec/core/forward.py:392`.  Current tests cover fixed-iteration trace
   behavior, not early convergence.

6. DTS parameter shape semantics are ambiguous when `G == S`.  The forward DTS
   helper treats a 1-D parameter with `numel() == S` as shared species-indexed
   at `gpurec/core/kernels/dts_fused.py:18`.  The backward helper prioritizes
   family layout when `family_idx` exists at
   `gpurec/core/kernels/wave_backward.py:57`.  Direct callers can therefore get
   different forward/backward interpretations for `[G]` parameters when the
   family count equals the species count.

7. Exposed C++ scheduler helpers do not validate `max_wave_size`.  The phased
   wave implementation advances by `max_wave_size` at
   `gpurec/core/cpp/preprocess.cpp:1086` and loops while
   `batch.size() < max_wave_size` at `gpurec/core/cpp/preprocess.cpp:1126`.
   Cross-family variants have the same pattern at
   `gpurec/core/cpp/preprocess.cpp:2401` and
   `gpurec/core/cpp/preprocess.cpp:2622`.  Python wrapper tests cover invalid
   values, but the pybind exports do not.  The direct pybind wave-stat entry
   points should reject `max_wave_size <= 0` before parsing input files, and a
   source-level hygiene guard should keep every exported `max_wave_size`
   scheduler wired to the shared validator.

8. The opt-in CUDA Pibar VJP path can mask failures in default `auto` mode.
   `gpurec/core/kernels/wave_backward.py:1910` enables the CUDA prototype for
   fp32 CUDA tensors, then catches broad `Exception` at
   `gpurec/core/kernels/wave_backward.py:1935`.  Warnings are limited to
   non-`auto` modes.  There are env-toggle tests, but no direct test for
   `uniform_cross_pibar_vjp_tree_from_ud_cuda`.

9. Backward and DTS kernel coverage is thin relative to risk.  Direct kernel
   tests import forward wave-step functions, but no direct tests were found for
   `dts_fused_parent_reduced`, `wave_backward_uniform_fused`,
   `dts_cross_backward_accum_fused`, `wave_backward_uniform_nosplit_cuda`, or
   the CUDA Pibar prototype.

10. `preprocess.cpp` relies on transitive includes.  The include block lacks
    `<set>` and `<chrono>` near `gpurec/core/cpp/preprocess.cpp:6`, but the file
    uses `std::set` at `gpurec/core/cpp/preprocess.cpp:2462` and `std::chrono`
    at `gpurec/core/cpp/preprocess.cpp:2720`.  The cleanup should add the
    direct standard-library includes and a source hygiene guard so future
    edits do not reintroduce transitive include reliance.

11. `GPUREC_LEAF_HIT_ONLY_LOGP` appears stale.  It is read at
    `gpurec/core/kernels/wave_backward.py:984` and passed into kernels, but the
    `LEAF_HIT_ONLY_LOGP` constexpr does not appear to be used inside those
    kernels.  This is a deletion candidate after a focused guard.

12. Several pybind debug or scheduler exports appear unowned by in-repo
    callers.  `compute_wave_stats`, `compute_packet_wave_stats`,
    `compute_phased_wave_stats`, cross-family stats, and `bench_parse` are
    exported around `gpurec/core/cpp/preprocess.cpp:2706`, but search found only
    definitions.  Either document them as public diagnostics or remove them
    with input-validation tests for retained exports.

13. `Pi_wave_backward` accepts `ancestors_T` at
    `gpurec/core/backward.py:41`, but the function does not use it.  This is a
    possible signature cleanup after call-site compatibility is documented.

### Public API, Workflow, And Optimization

14. `BatchedLBFGS.max_eval` can be exceeded.  The outer loop checks
    `func_evals < max_eval` at `gpurec/optimization/batched_lbfgs.py:295`, but
    `step()` performs an unconditional final gradient evaluation after the line
    search at `gpurec/optimization/batched_lbfgs.py:374`.  Existing LBFGS tests
    do not cover `max_eval`.  The next guard should assert both the optimizer's
    `state["func_evals"]` counter and the observed closure-call count stay
    within a tight `max_eval` budget.  The runtime fix should skip curvature
    history updates when the budget is exhausted before the post-line-search
    gradient refresh, because no valid `y_k` pair exists for the accepted step.

15. `GeneReconModel.configure_solver_iterations()` is unclear with active lazy
    prefetch.  Pending `_batch_futures` may already exist when solver fields
    are changed at `gpurec/api/model.py:1256`, `gpurec/api/model.py:1382`, and
    `gpurec/api/model.py:1398`.  Tests cover invalid inputs and close/restart
    behavior, not reconfiguration during pending prefetch.

16. Backtracking sampling can hang indefinitely if the external Rust binary
    stalls.  Help validation has a timeout, but actual sampling uses
    `subprocess.run()` without one at `gpurec/backtracking.py:283`,
    `gpurec/backtracking.py:365`, and `gpurec/backtracking.py:497`.  The
    shared sampling subprocess helper should pass a finite timeout and convert
    `subprocess.TimeoutExpired` into the same no-traceback `RuntimeError` style
    used for nonzero sampler exits, including the command text and timeout
    value.

17. Sampling aggregate output formats are underdocumented.  The workflow writes
    comma-space-separated `totalSpeciesEventCounts.txt`, whitespace-separated
    `totalTransfers.txt`, and values normalized by sample count rather than by
    family count at `gpurec/workflow/sampling.py:158`,
    `gpurec/workflow/sampling.py:177`, and `gpurec/workflow/sampling.py:361`.
    Tests pin current behavior, but user docs list filenames without defining
    format and normalization semantics.

18. `UniformChunkedReconModel.loss_and_grad(reduction="full_sum_estimate")` is
    a public stochastic-optimizer helper branch without direct coverage.  The
    existing integration test covers default, `sum`, and `mean` reductions, but
    not the `total_families / selected_families` scaling applied to both the
    returned loss and gradient.  This can be covered with a CPU-safe unit test
    by monkeypatching the internal chunk evaluator rather than constructing a
    CUDA model.

19. `gpurec.workflow.checkpoint.load_checkpoint_config()` is an unreferenced
    helper.  It is not exported by `gpurec.workflow`, not exported at the
    package top level, and `rg "load_checkpoint_config"` finds only the
    definition.  Removing it leaves the documented checkpoint surface
    (`save_checkpoint`, `load_checkpoint`, and `restore_model_theta`) intact and
    reduces an otherwise unsupported module-level API.

### Scripts, Rust, Profiling, And Examples

20. Legacy HOGENOM launchers have inconsistent path override support.
    `scripts/README.md` labels them legacy checkout-local scripts, while
    `scripts/optimize_hogenom_ccp_global_uniform.py:21` and
    `scripts/optimize_hogenom_ccp_specieswise_uniform.py:26` hard-code local
    data paths and expose mostly optimizer/regularization flags.  Document
    which launchers are fixed-dataset before shared optimizer changes.

21. `profiling/bench_uniform_forward_backward_pipeline.py` references missing
    `docs/forward-backward-full-pipeline-plan.md` at lines 4-5.  The benchmark
    contract is stale until the reference is restored or removed.

22. `scripts/make_hogenom_branchscale_penalty_report.py` appears stale relative
    to current run-directory naming.  It only loads `penalty_*` directories at
    lines 103-110, while newer launchers create timestamped names, and the
    report text hard-codes a date and "1325 branch multipliers".

23. `configs/hogenom_ccp_wandb.yaml` is not a portable smoke config.  It assumes
    local HOGENOM data paths, CUDA, per-step checkpointing, and online W&B.  It
    should be documented as a full local experiment config rather than a general
    example.

24. `examples/minimal-run-config.json` defaults to `"device": "cuda"` even
    though the tiny fixtures are otherwise portable.  This is a documentation
    and reproducibility footgun for CPU-only users.

25. Rust backtracking input validation is shape-focused but numeric contracts
    are not fully documented.  Matrix validation checks only
    `rows * cols == data.len()` in `crates/gpurec-backtrack/src/lib.rs:31`, and
    origination probabilities are log-converted only if positive around line
    274 while non-finite values are filtered later around line 750.  Add schema
    docs and tests before changing sampler behavior.

26. Some profiling/evaluation scripts encode brittle external file-format
    assumptions.  `profiling/evaluate_hogenom_alerax_rates.py:29` reads only
    the second line of each `*_rates.txt` and treats the first three columns as
    D/L/T; defaults around line 147 hard-code the HOGENOM root, CUDA device, and
    iteration count.  The next low-risk cleanup is documentation-only: make the
    script module/help text state that it is a checkout-local HOGENOM AleRax
    validation utility, not a general rate-file parser.

### Tests, Docs, And Packaging

27. Release metadata still has an expected blocker.  `pyproject.toml` lacks a
    license key and license classifier, while `docs/release-readiness.md`
    requires adding both and a top-level license file.  The test suite currently
    treats this as an expected release metadata blocker.

28. The docs index presented historical cleanup notes as current.  This audit
    moved `core-simplification-suggestions.md` out of "Current Operating Notes"
    because the file itself says it is a historical snapshot and includes
    already-implemented items such as removing `scatter_lse.py`.

29. Performance docs contain broken references.  Examples include missing docs
    and benchmark scripts in `docs/lean-performance-path-regression.md`, and
    `docs/second-order-optimization-opportunities.md` references
    `profiling/bench_global_parameter_optimization.py` plus a line number that
    no longer exists in `tests/integration/test_gene_recon_model.py`.

30. Some GPU/data-heavy tests are classified as unit tests.  `tests/conftest.py`
    auto-marks everything under `tests/unit` as `unit`, but
    `tests/unit/test_adaptive_iterations.py` requires CUDA and `test_trees_1000`
    and some `test_specieswise_uniform.py` CUDA checks lack local `slow`
    markers.  This conflicts with `tests/README.md` guidance.  The follow-up
    keeps those tests in `tests/unit` for ownership, but requires every unit
    test that directly or indirectly depends on the 1000-family CUDA fixture to
    carry a local `@pytest.mark.slow` marker.

31. `tests/unit/test_release_metadata.py` mirrors docs and GitHub Actions YAML
    with many exact substring assertions.  These guards catch release drift, but
    they are brittle during harmless wording or workflow layout changes.

32. `tests/unit/test_workflow.py` is an oversized mixed-surface test module at
    5,123 lines.  It covers exports, config, checkpointing, optimization,
    backtracking commands, sampling, and more.  Splitting it by behavior would
    improve ownership and reduce stale-test risk.

33. `pytest.ini` globally ignores all `DeprecationWarning` and
    `PendingDeprecationWarning`.  Scoping suppression to known external noise
    would make project-owned deprecations visible.  A CPU unit run with
    `-W default` did not surface known warning noise, so the low-risk cleanup is
    to remove the blanket ignores and add a repository hygiene guard that only
    permits targeted warning filters.

34. `tests/__init__.py` is stale or unnecessary.  It describes `gradients` and
    `performance` suites, while the current marker taxonomy is `unit`,
    `integration`, `kernel`, `gpu`, and `slow`.  The file still helps direct
    imports such as `tests.unit.alerax_helpers`, so the low-risk cleanup is to
    simplify the package docstring rather than delete it.

35. CLI help smoke tests are sensitive to stale installed console scripts.  In
    this checkout, `which gpurec` resolved to `/home/enzo/miniforge3/bin/gpurec`,
    whose entry point imports `gpurec.cli.reconcile`.  The repo-local
    `python -m gpurec.cli --help` command passed, but
    `tests/unit/test_release_metadata.py::test_cli_help_smokes_are_quiet_on_cpu`
    failed through the stale PATH executable.  This is an environment/setup
    fragility to document or guard in release checks.

## Adequately Covered Or Lower-Risk Areas

- `gpurec/core/_helpers.py`, `gpurec/core/log2_utils.py`,
  `gpurec/core/terms.py`, `gpurec/core/scheduling.py`,
  `gpurec/core/species.py`, and `gpurec/core/memory_policy.py` have focused
  unit coverage.
- `gpurec/core/batching.py` and `gpurec/core/batch_planning.py` have strong
  scheduler and layout coverage, with the oversized-family `clade_budget` edge
  left open.
- `gpurec/core/model.py` cache validation and AleRax parsing are well covered,
  with the direct duplicate-family-name constructor path left open.
- Workflow checkpointing, CLI parse failures, public export guards, and
  backtracking command failure paths have broad unit coverage.

## Deletion And Simplification Candidates

- Remove or document unused pybind debug exports in `preprocess.cpp`.
- Decide whether the legacy pybind `preprocess()` wrapper should remain.  The
  current Python runtime routes through `preprocess_multiple_families(...,
  include_details=True)`, while the legacy wrapper duplicates extraction logic
  and has no in-repo Python callers.
- Decide whether `preprocess_multiple_families(..., include_details=False)` is
  a public C++ extension mode or dead compatibility surface; production Python
  callers request details.
- Decide whether `compute_clade_waves`, `collate_wave`, and
  `split_phase_waves` remain public scheduler helpers or are test-only legacy
  surface.
- Remove stale `GPUREC_LEAF_HIT_ONLY_LOGP` plumbing if a focused guard proves it
  is inert.
- Remove unused `ancestors_T` from `Pi_wave_backward` after documenting call-site
  compatibility.
- Simplify or delete stale `tests/__init__.py`.
- Rework `tests/unit/test_workflow.py` into focused modules.
- Mark historical docs clearly, remove broken links, and either restore or
  delete stale benchmark plan references.
- Keep the core `GPUREC_*` environment toggles documented in the user-facing
  README.  They currently control binary discovery, memory policy, retained
  optimized CUDA/Triton paths, and diagnostic launch tuning.

## Documentation Cleanup Completed

The first follow-up pass stayed documentation-only and addressed the lowest-risk
staleness found above:

- `docs/README.md` now lists `core-simplification-suggestions.md` as a
  historical cleanup snapshot rather than a current operating note.
- `profiling/bench_uniform_forward_backward_pipeline.py` no longer points to the
  missing `docs/forward-backward-full-pipeline-plan.md`.
- `docs/lean-performance-path-regression.md` now says its missing reference
  documents and benchmark commands are historical provenance, not a current
  reproducible command set.
- `docs/second-order-optimization-opportunities.md` no longer names a missing
  tracked benchmark file as an existing related file and no longer relies on a
  stale integration-test line number.
- `scripts/README.md` now calls out fixed-dataset HOGENOM reproducers whose
  paths live in module constants rather than general path flags.
- `configs/hogenom_ccp_wandb.yaml` now states that it is a checkout-local full
  HOGENOM experiment config, not a portable smoke example.
- `README.md` now documents that the checked minimal JSON config is CUDA-only,
  defines the sampling aggregate file formats and normalization, and identifies
  the Hydra HOGENOM YAML as a checkout-local full experiment config.
- `tests/README.md` now makes the `tests/unit` plus `gpu` marker overlap
  explicit so CPU-only audit gates use `-m "unit and not gpu"`.
- `tests/unit/test_release_metadata.py` now skips only a known stale external
  `gpurec` console script that imports `gpurec.cli.reconcile`, while still
  exercising the repo-local `python -m gpurec.cli --help` path.
- `tests/unit/test_global_wave_scheduler.py` now pins the documented
  `clade_budget` behavior for both first-fit planners: the budget is a packing
  target, so an individual oversized family can occupy its own batch.
- `gpurec/core/likelihood.py` now documents and validates that `ancestors_T` is
  required for the retained uniform-transfer E solver, replacing an indirect
  matrix-multiply failure with a clear `ValueError`.
- `tests/unit/test_origination_probs.py` now covers both `E_step` and
  `E_fixed_point` missing-`ancestors_T` errors.
- `gpurec/core/model.py` now rejects duplicate direct `family_names` before
  loading the preprocessing extension, so the direct constructor matches the
  AleRax parser's no-duplicate-name contract.
- `tests/unit/test_workflow.py` now covers the duplicate direct `family_names`
  validation and proves it runs before extension loading.
- `gpurec/optimization/batched_lbfgs.py` now treats the post-line-search
  gradient refresh as a budgeted evaluation.  If the Armijo probes consume the
  final allowed `max_eval`, `step()` returns the accepted probed loss without
  issuing one more gradient closure and skips curvature-history updates for
  that step because no fresh gradient pair exists.
- `tests/unit/test_batched_lbfgs.py` now covers the tight-budget case by
  checking both `state["func_evals"]` and actual gradient/loss closure call
  counts.
- `gpurec/backtracking.py` now applies a finite one-hour timeout to the shared
  Rust sampler subprocess invocation and converts `subprocess.TimeoutExpired`
  into the same no-traceback `RuntimeError` style used for nonzero sampler
  exits, including the command text.
- `tests/unit/test_workflow.py` now covers the timeout path by asserting that
  the shared runner passes `_BACKTRACK_RUN_TIMEOUT_SECONDS` into
  `subprocess.run()`.
- `gpurec/core/cpp/preprocess.cpp` now validates `max_wave_size > 0` at the
  direct pybind scheduler diagnostics before parsing input files.
- `tests/unit/test_repository_hygiene.py` now pins that all exported C++
  scheduler diagnostics accepting `max_wave_size` call the shared validator.
- `gpurec/core/cpp/preprocess.cpp` now includes `<chrono>` and `<set>`
  directly instead of relying on transitive standard-library includes.
- `tests/__init__.py` now accurately describes the current test package layout
  and documents why the package namespace is retained for helper imports such
  as `tests.unit.alerax_helpers`.
- `tests/unit/test_repository_hygiene.py` now guards both the direct C++
  standard includes and the current `tests/__init__.py` package docstring.
- `pytest.ini` no longer globally ignores all `DeprecationWarning` and
  `PendingDeprecationWarning`; targeted filters can still be added later for
  specific dependency noise.
- `tests/unit/test_repository_hygiene.py` now rejects future blanket
  deprecation-warning ignores in `pytest.ini`.
- `tests/unit/test_adaptive_iterations.py` and
  `tests/unit/test_specieswise_uniform.py` now mark 1000-family CUDA unit checks
  as `slow`, while retaining their `unit` ownership.
- `tests/unit/test_repository_hygiene.py` now guards direct and
  fixture-mediated `data_dir_1000` usage in unit tests so expensive CUDA unit
  checks keep a local `@pytest.mark.slow` marker.
- `README.md` now documents the package-level `GPUREC_*` environment flags for
  binary discovery, compatibility guards, memory policy, and kernel diagnostics.
- `tests/unit/test_repository_hygiene.py` now guards that every `GPUREC_*`
  environment variable read by tracked package code appears in the README.
- `profiling/evaluate_hogenom_alerax_rates.py` now documents itself as a
  checkout-local HOGENOM AleRax validation utility rather than a general
  rate-file parser, and its help text states the per-family checkpoint file
  layout and second-line D/L/T rate assumption.
- `tests/unit/test_repository_hygiene.py` now guards that the HOGENOM AleRax
  rate-evaluation helper keeps that local file-format contract visible.
- `tests/unit/test_optimization_workflow.py` now covers
  `UniformChunkedReconModel.loss_and_grad(reduction="full_sum_estimate")` with a
  CPU-safe monkeypatched evaluator, asserting that both returned loss and
  gradient are scaled by `total_families / selected_families`.

## Verification Run This Round

- `pytest --collect-only -q`: 823 tests collected during the initial
  tracked-file audit inventory.
- Stale-reference grep over tracked docs/profiling/scripts/config files for
  `forward-backward-full-pipeline-plan`,
  `bench_global_parameter_optimization`, and `test_gene_recon_model.py:293`:
  no matches after the documentation cleanup.
- `python -m py_compile profiling/bench_uniform_forward_backward_pipeline.py`:
  passed.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_repository_hygiene.py tests/unit/test_examples.py -q`:
  23 passed.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_release_metadata.py -q`:
  39 passed, 1 skipped.  The skip is the stale external `gpurec` console script
  noted above; the repo-local module help smoke passed.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_global_wave_scheduler.py::test_plan_family_batches_treats_clade_budget_as_soft_packing_target -q`:
  2 passed.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_global_wave_scheduler.py -q`:
  40 passed.
- `python -m py_compile gpurec/core/likelihood.py tests/unit/test_origination_probs.py`:
  passed.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_origination_probs.py -q`:
  6 passed.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_specieswise_uniform.py -q -k 'origination or E_fixed_point or trace'`:
  1 skipped, 4 deselected in the CPU-only environment.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit -q -m "unit and not gpu"`:
  799 passed, 1 skipped, 6 deselected after the duplicate-family-name guard.
- `python -m py_compile gpurec/core/model.py tests/unit/test_workflow.py`:
  passed.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_workflow.py::test_gene_dataset_rejects_duplicate_family_names_before_extension -q`:
  1 passed.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_workflow.py::test_gene_dataset_rejects_single_gene_tree_path_before_extension tests/unit/test_workflow.py::test_gene_dataset_rejects_duplicate_family_names_before_extension -q`:
  2 passed.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_workflow.py -q -k 'gene_dataset_rejects_single_gene_tree_path_before_extension or gene_dataset_rejects_duplicate_family_names_before_extension'`:
  2 passed, 420 deselected.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_batched_lbfgs.py::test_batched_lbfgs_respects_max_eval_after_line_search -q`:
  failed before the optimizer change with `state["func_evals"] == 3` for
  `max_eval=2`, then passed after the fix.
- `python -m py_compile gpurec/optimization/batched_lbfgs.py tests/unit/test_batched_lbfgs.py`:
  passed.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_batched_lbfgs.py -q`:
  6 passed.
- `python -m pytest --collect-only -q`: 829 tests collected after the latest
  regression guards.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit -q -m "unit and not gpu"`:
  800 passed, 1 skipped, 6 deselected after the LBFGS `max_eval` guard.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_workflow.py::test_backtracking_runner_reports_subprocess_timeout -q`:
  failed before the timeout fix because `subprocess.TimeoutExpired` propagated
  directly and the sampler call had no `timeout`, then passed after the fix.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_workflow.py::test_backtracking_runner_reports_subprocess_timeout tests/unit/test_workflow.py::test_backtracking_runner_reports_subprocess_failure_with_stderr tests/unit/test_workflow.py::test_backtracking_sampler_helpers_share_subprocess_io tests/unit/test_workflow.py::test_backtracking_runner_reports_missing_expected_outputs -q`:
  4 passed.
- `python -m py_compile gpurec/backtracking.py tests/unit/test_workflow.py`:
  passed.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_workflow.py -q`:
  423 passed.
- `python -m pytest --collect-only -q`: 830 tests collected after the sampling
  timeout guard.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit -q -m "unit and not gpu"`:
  801 passed, 1 skipped, 6 deselected after the sampling timeout guard.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_repository_hygiene.py::test_cpp_wave_stat_exports_validate_positive_max_wave_size -q`:
  failed before the C++ guard because the shared validator did not exist, then
  passed after the guard.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_repository_hygiene.py -q`:
  20 passed.
- C++ extension build/load probe:
  `python - <<'PY' ... _load_extension() ... PY` printed `preprocess_cpp`.
- Direct pybind probe over `compute_phased_waves`, `compute_wave_stats`,
  `compute_packet_wave_stats`, `compute_phased_wave_stats`,
  `compute_phased_cross_family_wave_stats`, and `compute_cross_family_wave_stats`
  with `max_wave_size=0`: each raised `ValueError: max_wave_size must be
  positive` before parsing the intentionally missing input file.
- `python -m pytest --collect-only -q`: 831 tests collected after the direct C++
  `max_wave_size` guard.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit -q -m "unit and not gpu"`:
  802 passed, 1 skipped, 6 deselected after the direct C++ `max_wave_size`
  guard.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_repository_hygiene.py::test_preprocess_cpp_declares_direct_standard_includes tests/unit/test_repository_hygiene.py::test_tests_package_docstring_matches_current_layout -q`:
  failed before the hygiene cleanup because `<chrono>` was missing and
  `tests/__init__.py` still mentioned stale `gradients/` and `performance/`
  directories, then passed after the cleanup.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_repository_hygiene.py -q`:
  22 passed.
- C++ extension build/load probe after the include cleanup:
  `python - <<'PY' ... _load_extension() ... PY` printed `preprocess_cpp`.
- `python -m pytest --collect-only -q`: 833 tests collected after the C++
  include and test-package docstring hygiene guards.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit -q -m "unit and not gpu"`:
  804 passed, 1 skipped, 6 deselected after the C++ include and
  test-package docstring hygiene guards.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit -q -m "unit and not gpu" -W default`:
  804 passed, 1 skipped, 6 deselected and surfaced no warning noise before the
  warning-filter cleanup.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_repository_hygiene.py::test_pytest_warning_filters_are_not_blanket_ignores -q`:
  failed before the cleanup because `pytest.ini` still contained blanket
  deprecation-warning ignores, then passed after the cleanup.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_repository_hygiene.py -q`:
  23 passed.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit -q -m "unit and not gpu"`:
  805 passed, 1 skipped, 6 deselected under the actual config after removing
  blanket warning filters.
- `python -m pytest --collect-only -q`: 834 tests collected after the
  warning-filter hygiene guard.
- `python -m pytest --collect-only -q -m slow`: 6 of 835 tests selected after
  adding the missing 1000-family CUDA slow markers.
- `python -m pytest --collect-only -q -m "gpu and not slow"`: 21 of 835 tests
  selected after moving the expensive unit CUDA checks into the slow set.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_repository_hygiene.py -q`:
  25 passed after adding the slow-marker and README environment-flag guards.
- `python -m pytest --collect-only -q`: 836 tests collected after the README
  environment-flag guard.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit -q -m "unit and not gpu"`:
  807 passed, 1 skipped, 6 deselected after the README environment-flag guard.
- `python profiling/evaluate_hogenom_alerax_rates.py --help`: passed and shows
  the checkout-local HOGENOM checkpoint-rate contract.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_repository_hygiene.py -q`:
  26 passed after adding the HOGENOM AleRax rate-evaluator documentation guard.
- `python -m pytest --collect-only -q`: 837 tests collected after the
  HOGENOM AleRax rate-evaluator documentation guard.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit -q -m "unit and not gpu"`:
  808 passed, 1 skipped, 6 deselected after the HOGENOM AleRax rate-evaluator
  documentation guard.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_optimization_workflow.py::test_uniform_chunked_full_sum_estimate_scales_loss_and_grad -q`:
  1 passed after adding the CPU-safe `full_sum_estimate` scaling guard.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_optimization_workflow.py -q`:
  29 passed after the `full_sum_estimate` scaling guard.
- `python -m pytest --collect-only -q`: 838 tests collected after the
  `full_sum_estimate` scaling guard.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit -q -m "unit and not gpu"`:
  809 passed, 1 skipped, 6 deselected after the `full_sum_estimate` scaling
  guard.
- `python -m py_compile gpurec/workflow/checkpoint.py tests/unit/test_workflow.py`:
  passed after removing the unused checkpoint config helper.
- `rg -n "load_checkpoint_config" . -S`: now finds only audit documentation
  mentions, confirming the unsupported helper is no longer present in package
  code.
- Direct import probe: `hasattr(gpurec.workflow.checkpoint, "load_checkpoint_config")`
  returns `False`.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_workflow.py -q -k 'checkpoint or restore_model_theta or resume'`:
  50 passed, 373 deselected after removing the helper.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_workflow.py -q`:
  423 passed after removing the helper.
- `python -m pytest --collect-only -q`: 838 tests collected after removing the
  helper.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit -q -m "unit and not gpu"`:
  809 passed, 1 skipped, 6 deselected after removing the helper.
- `git diff --check`: passed.
- `python scripts/check_release_metadata.py`: failed with the known release
  blockers: missing top-level `LICENSE`, missing `pyproject.toml` license
  metadata, and missing license classifier.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_release_metadata.py tests/unit/test_repository_hygiene.py -q`:
  58 passed, 1 skipped.  The skip is the stale external `gpurec` console script
  noted above.
- `python -m gpurec.cli --help`: passed.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_repository_hygiene.py -q`:
  19 passed.

## Recommended Next Order

1. Continue turning documented findings into focused guards before runtime
   redesigns.  Contract coverage now exists for duplicate direct
   `family_names`, oversized `clade_budget`, `ancestors_T=None`, and LBFGS
   `max_eval` evaluation accounting, sampling subprocess timeout behavior, and
   direct C++ `max_wave_size` validation.
2. Fix remaining documentation-only staleness as it is found in touched areas.
3. Make low-risk hygiene changes with tests: slow markers and any future
   warning filters only if scoped to a specific dependency warning.
4. Only then consider behavior changes for backward small-`S`, DTS parameter
   shape semantics, CUDA Pibar fallback policy, and sampling aggregate formats.
