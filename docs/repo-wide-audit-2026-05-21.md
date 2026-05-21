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

10. `preprocess.cpp` had relied on transitive includes.  The direct `<set>`
    include is now present because the retained C++ source uses `std::set`.
    The former `<chrono>` need was isolated to the unowned `bench_parse`
    pybind benchmark export, so both the benchmark export and `<chrono>`
    include have now been removed.

11. `GPUREC_LEAF_HIT_ONLY_LOGP` was stale.  The flag had been read by the
    retained wave-backward wrapper and passed into Triton kernels, but the
    `LEAF_HIT_ONLY_LOGP` constexpr was not read inside those kernels.  The
    runtime plumbing and README row have now been removed, with a source-level
    hygiene guard keeping the retired flag out of runtime code and public env
    docs.

12. Several pybind debug or scheduler exports appear unowned by in-repo
    callers.  `compute_wave_stats`, `compute_packet_wave_stats`,
    `compute_phased_wave_stats`, cross-family stats, and `bench_parse` are
    exported around `gpurec/core/cpp/preprocess.cpp:2706`, but search found only
    definitions.  Retain the wave-stat exports as diagnostic extension helpers
    while they have explicit `max_wave_size` validation guards.  `bench_parse`
    has now been removed because it was an unowned timing-only benchmark lambda
    and was not used by supported Python workflows or tests.

13. `Pi_wave_backward` no longer accepts unused `ancestors_T`.  The two
    production call sites pass `ancestors_T` separately to E-adjoint and
    likelihood code that still needs the sparse ancestor matrix, so narrowing
    only the Pi-backward signature is call-site compatible.

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

15. `GeneReconModel.configure_solver_iterations()` now documents its active
    lazy-prefetch contract.  The method updates model defaults and resident
    batch static states that are already built; pending background prefetch work
    is not cancelled or rewritten.  The README and method docstring tell users
    to configure before scheduling lazy prefetch, or to materialize resident
    batches and configure again when all batches should share new controls.

16. Backtracking sampling can hang indefinitely if the external Rust binary
    stalls.  Help validation has a timeout, but actual sampling uses
    `subprocess.run()` without one at `gpurec/backtracking.py:283`,
    `gpurec/backtracking.py:365`, and `gpurec/backtracking.py:497`.  The
    shared sampling subprocess helper should pass a finite timeout and convert
    `subprocess.TimeoutExpired` into the same no-traceback `RuntimeError` style
    used for nonzero sampler exits, including the command text and timeout
    value.

17. Sampling aggregate output formats are now documented.  The README defines
    `event_counts.tsv` as tab-separated, `totalSpeciesEventCounts.txt` as the
    AleRax-compatible comma-space text format, `totalTransfers.txt` as
    whitespace-separated source species, destination species, and average
    transfer count, and states that aggregate values are averaged over the
    requested sample count for each retained family rather than all families in
    the checkpoint.  A repository hygiene guard keeps those format and
    normalization details present in user docs.

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

20. `GeneReconModel.materialize_batches()` and
    `GeneReconModel.full_loss_for_theta(theta)` are public API helpers whose
    contracts are only indirectly exercised by CUDA integration coverage.
    `materialize_batches()` should build every resident batch and return a copy
    of batch metadata; `full_loss_for_theta(theta)` should stream all resident
    batches with `need_grad=True` for differentiable explicit-theta probes and
    with `need_grad=False` for no-grad probes.  Both contracts can be pinned
    with CPU-safe monkeypatched model instances before any runtime behavior
    change.

### Scripts, Rust, Profiling, And Examples

21. Legacy HOGENOM launchers have inconsistent path override support.
    `scripts/README.md` labels them legacy checkout-local scripts, while
    `scripts/optimize_hogenom_ccp_global_uniform.py:21` and
    `scripts/optimize_hogenom_ccp_specieswise_uniform.py:26` hard-code local
    data paths and expose mostly optimizer/regularization flags.  Document
    which launchers are fixed-dataset before shared optimizer changes.

22. `profiling/bench_uniform_forward_backward_pipeline.py` references missing
    `docs/forward-backward-full-pipeline-plan.md` at lines 4-5.  The benchmark
    contract is stale until the reference is restored or removed.

23. `scripts/make_hogenom_branchscale_penalty_report.py` appears stale relative
    to current run-directory naming.  It only loads `penalty_*` directories at
    lines 103-110, while newer launchers create timestamped names, and the
    report text hard-codes a date and "1325 branch multipliers".

24. `configs/hogenom_ccp_wandb.yaml` is not a portable smoke config.  It assumes
    local HOGENOM data paths, CUDA, per-step checkpointing, and online W&B.  It
    should be documented as a full local experiment config rather than a general
    example.

25. `examples/minimal-run-config.json` defaults to `"device": "cuda"` even
    though the tiny fixtures are otherwise portable.  This is a documentation
    and reproducibility footgun for CPU-only users.

26. Rust backtracking input validation is shape-focused but numeric contracts
    are not fully documented.  Matrix validation currently computes
    `rows * cols` directly in `crates/gpurec-backtrack/src/lib.rs:32`, which can
    panic on overflow in debug builds instead of returning `InvalidInput`.
    Public Rust payload types also lack schema docs for row-major matrices,
    base-2 log units, the `-1e300` sentinel, postorder species indexing,
    leaf/split bounds, and origination probability semantics.  Rust validation
    is thinner than the Python bridge for leaf species indices and finite/range
    contracts, while origination probabilities are log-converted only if
    positive around line 274.  Add schema docs and targeted tests before
    changing sampler behavior.

27. Some profiling/evaluation scripts encode brittle external file-format
    assumptions.  `profiling/evaluate_hogenom_alerax_rates.py:29` reads only
    the second line of each `*_rates.txt` and treats the first three columns as
    D/L/T; defaults around line 147 hard-code the HOGENOM root, CUDA device, and
    iteration count.  The next low-risk cleanup is documentation-only: make the
    script module/help text state that it is a checkout-local HOGENOM AleRax
    validation utility, not a general rate-file parser.

28. The Rust backtracking CLI accepts an ignored positional output file when
    `--samples 1 --output-dir DIR input.json output.xml` is passed.  Directory
    mode is selected by `output_dir.is_some()` in
    `crates/gpurec-backtrack/src/main.rs:37`, but the parse-time rejection of a
    second positional output path only triggers when `samples > 1` around line
    132.  Document or reject that combination before users rely on it.

### Tests, Docs, And Packaging

29. Release metadata still has an expected blocker.  `pyproject.toml` lacks a
    license key and license classifier, while `docs/release-readiness.md`
    requires adding both and a top-level license file.  The test suite currently
    treats this as an expected release metadata blocker.

30. The docs index presented historical cleanup notes as current.  This audit
    moved `core-simplification-suggestions.md` out of "Current Operating Notes"
    because the file itself says it is a historical snapshot and includes
    already-implemented items such as removing `scatter_lse.py`.

31. Performance docs contain broken references.  Examples include missing docs
    and benchmark scripts in `docs/lean-performance-path-regression.md`, and
    `docs/second-order-optimization-opportunities.md` references
    `profiling/bench_global_parameter_optimization.py` plus a line number that
    no longer exists in `tests/integration/test_gene_recon_model.py`.

32. Some GPU/data-heavy tests are classified as unit tests.  `tests/conftest.py`
    auto-marks everything under `tests/unit` as `unit`, but
    `tests/unit/test_adaptive_iterations.py` requires CUDA and `test_trees_1000`
    and some `test_specieswise_uniform.py` CUDA checks lack local `slow`
    markers.  This conflicts with `tests/README.md` guidance.  The follow-up
    keeps those tests in `tests/unit` for ownership, but requires every unit
    test that directly or indirectly depends on the 1000-family CUDA fixture to
    carry a local `@pytest.mark.slow` marker.

33. `tests/unit/test_release_metadata.py` mirrors docs and GitHub Actions YAML
    with many exact substring assertions.  These guards catch release drift, but
    they are brittle during harmless wording or workflow layout changes.

34. `tests/unit/test_workflow.py` is an oversized mixed-surface test module at
    5,123 lines.  It covers exports, config, checkpointing, optimization,
    backtracking commands, sampling, and more.  Splitting it by behavior would
    improve ownership and reduce stale-test risk.

35. `pytest.ini` globally ignores all `DeprecationWarning` and
    `PendingDeprecationWarning`.  Scoping suppression to known external noise
    would make project-owned deprecations visible.  A CPU unit run with
    `-W default` did not surface known warning noise, so the low-risk cleanup is
    to remove the blanket ignores and add a repository hygiene guard that only
    permits targeted warning filters.

36. `tests/__init__.py` is stale or unnecessary.  It describes `gradients` and
    `performance` suites, while the current marker taxonomy is `unit`,
    `integration`, `kernel`, `gpu`, and `slow`.  The file still helps direct
    imports such as `tests.unit.alerax_helpers`, so the low-risk cleanup is to
    simplify the package docstring rather than delete it.

37. CLI help smoke tests are sensitive to stale installed console scripts.  In
    this checkout, `which gpurec` resolved to `/home/enzo/miniforge3/bin/gpurec`,
    whose entry point imports `gpurec.cli.reconcile`.  The repo-local
    `python -m gpurec.cli --help` command passed, but
    `tests/unit/test_release_metadata.py::test_cli_help_smokes_are_quiet_on_cpu`
    failed through the stale PATH executable.  This is an environment/setup
    fragility to document or guard in release checks.

### Subagent Refresh Findings

- Workflow optimizer modes are public but underdocumented and under-tested.
  `RunConfig` and the CLI expose `adagrad`, `lbfgs`, and `adam-lbfgs`, while
  README guidance still mostly describes Adam.  Before changing optimizer logic
  or deleting modes, add an optimizer-mode reference covering stopping and LBFGS
  failure semantics, then add fake-model tests for Adagrad, active LBFGS, and
  LBFGS runtime failure paths.
- E-adjoint solver failures can disappear into partial diagnostics.  The
  implicit-gradient solver can return `success=False`, while workflow
  diagnostics currently aggregate only iteration/convergence summaries.  Decide
  whether failed adjoint solves fail optimization, warn, or only appear in
  history, then surface `E_adjoint_success` and `E_adjoint_rel_res` before
  relying on the diagnostics for production monitoring.
- `gpurec.workflow.diagnostics.safe_float()` appears unused outside its direct
  unit test.  If it is not intended as public workflow API, delete the helper
  and test; if it is intended public surface, document and export it first.
- Completed-resume status is not documented or asserted.  Metadata validation
  permits `step == next_step`, and resuming at `config.steps` falls through the
  optimization loop status path.  Document the expected completed-checkpoint
  status before changing resume behavior.
- RunConfig and CLI reference docs lag the current option surface.  Add a
  maintained or generated option table before changing defaults or validation
  rules, so constraints such as even `fixed_iters_pi`, optimizer modes, and
  checkpoint cadence are not only captured in tests.
- The tests audit found one overbroad integration-test skip: the
  `test_uniform_chunked_model.py` module skips entirely when `test_trees_1000`
  is absent, even though its HOGENOM unrooted-Newick test only needs
  `hogenom_bench`.  Move the large-fixture skip to the tests that actually use
  `test_trees_1000`, or split the HOGENOM case into its own module.
- Several GPU tests are still smoke-heavy: Adam/LBFGS integration checks mostly
  assert that theta changed, the HOGENOM unrooted parsing check asserts
  metadata only, and the specieswise backward check asserts finite values.
  Prefer before/after NLL decrease or reference-close assertions when local
  data makes that practical.
- Adaptive iteration coverage only proves the forced-max path.  Add a slow GPU
  case with loose tolerances that asserts solver iterations stop before the
  configured maximum.
- Some useful tests intentionally construct private or partially initialized
  objects.  `tests/README.md` should document when that is acceptable as a
  guardrail, what a smoke test must prove, and how Rust/cargo checks fit into
  the test-authoring rules.
- Release/docs hygiene tests contain brittle wording snapshots and duplicated
  export/import checks.  Consolidate to one public import behavior test plus
  one README/public-doc coverage test where possible, and prefer parsing
  structured files over long substring lists.
- The docs/scripts audit confirmed the known release metadata blocker, and
  added that tracked notebooks are undocumented checkout-local HOGENOM analysis
  artifacts with hard-coded CUDA/data assumptions.  Add a `notebooks/README.md`
  or docs-index section before deciding whether to keep, archive, or delete
  them.
- The HOGENOM script/profiling surface needs an ownership matrix.  Document
  which legacy launchers are fixed-dataset, which one-off report tools are
  stale, and which benchmarks depend on private internals before deleting or
  migrating scripts.
- Historical docs still contain intentionally stale or broken references.  Add
  an explicit historical/not-reproducible convention plus a docs link-check
  allowlist so current docs can be checked without flagging archived notes.
- `configs/hogenom_ccp_wandb.yaml` is checkout-local and online-W&B-oriented.
  Add a smoke/offline override snippet before changing defaults or presenting it
  as a reusable config.
- The core/API audit reinforced that `Pi_wave_backward` has hard runtime
  contracts that are not direct public docs: CUDA-only execution,
  `float32`/`float64`, `S > 256`, required `leaf_species_index`, scratch budget,
  and `uniform_pibar_row_max`.  Document these before changing backward fallback
  behavior.
- Native CUDA prototype paths remain experimental and thinly tested.  The
  `GPUREC_CUDA_SELF_LOOP_*` and `GPUREC_CUDA_PIBAR_FROM_UD` routes should have a
  documented support/fallback policy before adding parity tests or removing
  broad auto-mode fallbacks.
- DTS parameter shape precedence is ambiguous when `G == S`: forward and
  backward paths can interpret a 1-D tensor differently.  Document the intended
  precedence in public parameter/API docs before changing either implementation.
- `GPUREC_LEAF_HIT_ONLY_LOGP` is not a supported environment contract and has
  now been removed from the README environment table and retained kernel
  plumbing.  Source inspection showed its `LEAF_HIT_ONLY_LOGP` constexpr was
  passed through but not read in the Triton kernels, so the CPU-safe audit guard
  checks runtime source and public docs rather than CUDA behavior.
- `GeneReconModel.full_nll_per_family()` delegates to genewise-only logic but is
  documented generically.  Document it as genewise-only first, then add explicit
  mode-error tests or implement shared/specieswise per-family values.
- Lazy resident prefetch plus `configure_solver_iterations()` now has a public
  contract: the method updates model defaults and already-built static states,
  but it does not cancel or rewrite pending background prefetch work.  Future
  runtime changes should update that documented contract first.
- Scheduler and diagnostic exports such as `collate_wave`, `split_phase_waves`,
  and C++ wave-stat exports look unowned or test-only.  Decide whether they are
  supported diagnostics; otherwise delete or guard them.  `bench_parse` is no
  longer retained as public surface.
- The legacy leaf-to-species fallback is now documented in public user-facing
  docs: direct `from_trees` inputs map `Species_gene` to species `Species` and
  labels without `_` to the full label, while AleRax `mapping` entries and
  explicit `leaf_species_maps` should be used when labels do not follow that
  convention.

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
- Keep the retired `GPUREC_LEAF_HIT_ONLY_LOGP` guard so dead diagnostic env
  plumbing does not return to runtime code or public README docs.
- Keep the `Pi_wave_backward` signature guard so unused `ancestors_T` does not
  return to the Pi-adjoint path; E-adjoint and likelihood paths retain their own
  `ancestors_T` arguments.
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
- `gpurec/core/cpp/preprocess.cpp` now includes `<set>` directly instead of
  relying on transitive standard-library includes.  The stale `<chrono>` include
  is gone with the unowned `bench_parse` benchmark helper.
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
- `tests/unit/test_workflow.py` now needs CPU-safe public API guards for
  `GeneReconModel.materialize_batches()` and
  `GeneReconModel.full_loss_for_theta(theta)` before any runtime change touches
  resident batch materialization or explicit-theta full-loss streaming.
- `tests/unit/test_workflow.py` now covers those public API contracts with
  CPU-safe monkeypatched model instances: resident batch materialization must
  build every batch and return a metadata-list copy, and explicit-theta
  full-loss streaming must use `need_grad=True` for differentiable probes and
  `need_grad=False` for no-grad probes.
- `crates/gpurec-backtrack/src/lib.rs` now documents the public Rust
  backtracking schema and rejects `Matrix` shape products that overflow `usize`
  with `InvalidInput` instead of a debug-build panic.
- `tests/README.md` now documents when private-helper guardrail tests are
  acceptable, what smoke tests should prove, and how Rust/cargo checks fit into
  CPU-safe test authoring.
- `tests/integration/test_uniform_chunked_model.py` no longer skips the HOGENOM
  unrooted-Newick test just because the unrelated `test_trees_1000` fixture is
  absent.
- `notebooks/README.md` now documents both tracked notebooks as checkout-local
  HOGENOM analysis artifacts rather than portable examples, and `docs/README.md`
  plus the main README point to that ownership note.
- `scripts/README.md` now has an ownership matrix for tracked scripts,
  including keep/migrate/delete guidance for legacy HOGENOM launchers,
  one-off report tools, profiling helpers, validation utilities, plotting, and
  release metadata checks.
- `tests/unit/test_repository_hygiene.py` now guards that tracked notebooks are
  documented as checkout-local artifacts and that every tracked script appears
  in the script ownership matrix.
- `README.md` now documents the public workflow optimizer modes (`adam`,
  `adagrad`, `lbfgs`, and `adam-lbfgs`), including LBFGS line-search/failure
  semantics and Adam-to-LBFGS resume-state phase behavior.
- Public workflow optimizer modes now have CPU-safe fake-model runner guards.
  `tests/unit/test_workflow.py` exercises Adagrad rows and checkpoint phase
  metadata, active LBFGS rows and current-theta re-evaluation accounting, the
  `adam-lbfgs` warmup-to-LBFGS schedule, and the no-escape
  `lbfgs_runtime_error` failed-result path.
- Completed-checkpoint resume behavior is now documented and asserted.  The
  README states that resuming when checkpoint `next_step` already equals
  configured `steps` performs only final evaluation/artifact refresh and returns
  the existing `not_converged`/`max_steps` status.  A workflow regression now
  verifies that no optimizer-step row is emitted in that no-op resume path.
- E-adjoint solver nonconvergence now has an explicit workflow contract and
  visible telemetry.  The README documents it as diagnostic-only unless the
  objective or gradient becomes nonfinite, and `solver_stats()` now carries
  aggregate E-adjoint iteration, relative-residual, success, and failed-batch
  fields into history rows.
- The genewise-only per-family NLL contract is now explicit.  The README and
  `GeneReconModel.full_nll_per_family()` docstring state that
  `nll_per_family()` and `full_nll_per_family()` are genewise-only public
  surfaces for independent per-family losses, while shared-theta modes should
  use `forward(reduce="per_family")` under `torch.no_grad()` only as a
  diagnostic breakdown.
- `Pi_wave_backward` call-site compatibility is now documented before removing
  the unused `ancestors_T` argument: the Pi-adjoint path never reads the sparse
  ancestor matrix, while `_e_adjoint_and_theta_vjp()` and likelihood code still
  receive `ancestors_T` through their own parameters.
- `Pi_wave_backward` now omits the unused `ancestors_T` keyword.  The two
  production call sites in `gpurec/optimization/implicit_grad.py` and
  `gpurec/api/uniform_chunked.py` no longer pass it to the Pi-adjoint function,
  while their E-adjoint and likelihood calls still pass `ancestors_T` where it
  is used.
- The public leaf-species mapping contract is now documented.  `README.md` and
  `GeneReconModel.from_trees()` describe the legacy `Species_gene` prefix
  fallback for direct Newick inputs and point nonconforming labels to AleRax
  family-file `mapping` entries or explicit `leaf_species_maps`.
- Lazy-prefetch solver reconfiguration is now documented.  `README.md` and
  `GeneReconModel.configure_solver_iterations()` state that reconfiguration
  updates model defaults and resident batch static states that are already
  built, but does not cancel or rewrite pending background prefetch work.
- The stale `GPUREC_LEAF_HIT_ONLY_LOGP` diagnostic flag is removed from the
  retained wave-backward kernel wrapper and public README environment table.
  `tests/unit/test_repository_hygiene.py` now guards that the retired flag stays
  out of runtime source and public environment documentation.
- A Rust subagent rechecked `crates/gpurec-backtrack`: direct `quick-xml` usage
  was not found in `src/main.rs` or `src/lib.rs`, and `cargo tree -i quick-xml`
  showed it remains available transitively through `rustree`.  The unused direct
  dev-dependency is now removed from `Cargo.toml` and the root package lockfile
  dependency list.  The same pass found `WorkItem.clade` is still semantically
  needed but always populated: the root and every `apply_term()` producer
  enqueue concrete clade IDs, and no `clade: None` producer exists.  The
  `Option<usize>` wrapper is now removed while retaining the clade value.
- The Rust backtracking CLI now rejects `--output-dir DIR input.json output.xml`
  for single-sample runs as well as multi-sample runs, so the extra positional
  output path is not silently ignored in directory mode.
- `gpurec.workflow.diagnostics.safe_float()` was confirmed unused outside its
  direct unit test and removed rather than promoted as public workflow API.

## Refreshed Subagent Audit

A current continuation launched five read-only subagents over the tracked file
inventory again: core Python runtime; public API/workflow/optimization; native
kernels/C++/Rust; tests/CI; and docs/scripts/notebooks/configuration.  They did
not edit files.  New or still-open findings from that refresh are:

- Rust sampler payload validation now rejects non-finite base-2 log inputs and
  out-of-range `leaf_species` entries before sampling.  The Rust preparation
  path validates matrix payloads, log-probability arrays, transfer arrays,
  split log probabilities, origination weights, and leaf species indexes before
  computing derived sampler weights.
- Cached preprocessing validation now rejects `leaf_col_index` values outside
  the species range before they can become forward-kernel species indices.  The
  regression loads a cached family with a species index equal to `S` and
  verifies that cache validation rejects it without rerunning preprocessing.
- Direct bfloat16 support is now documented as direct-API experimental only.
  The direct `UniformChunkedReconModel` constructor accepts `torch.bfloat16`
  for CUDA memory-constrained forward/NLL probes, but the README and class
  docstring state that workflow configuration and CLI runs intentionally expose
  only fp32/fp64, and that the retained Pi backward/gradient path does not
  support bf16.  bf16 is not for release smokes, optimizer checkpoints, or
  Hessian/second-order diagnostics.  Repository hygiene and workflow unit guards
  keep the boundary aligned across README, direct API docs, `GeneReconModel`,
  `UniformChunkedReconModel`, CLI, and workflow config.
- Prepared origination probabilities are now documented as an internal trust
  boundary.  `prepare_origination_probs()` states that `assume_prepared=True`
  is only for model/static-owned tensors already prepared during construction;
  it still checks shape after device/dtype conversion, but skips finite,
  nonnegative, positive-mass, and normalization checks.  A focused regression
  pins that default preparation rejects invalid inputs while the prepared path
  returns them unchanged.
- `BiCGSTAB` failure telemetry is not enforced in the implicit-gradient path:
  the adjoint vector is consumed even when the solver reports failure, and
  workflow diagnostics surface the failure only after the gradient has already
  been used.  Add a solver-failure policy and regression before behavior
  changes.
- The native CUDA prototype loaders are inconsistent.  `wave_backward_cuda.py`
  preloads wheel-provided NVRTC builtins before compilation, while
  `pibar_vjp_cuda.py` compiles without that preflight.  The CUDA Pibar path also
  remains silent in `auto` fallback mode, so document and then consolidate the
  loader/fallback policy before changing runtime behavior.
- The self-loop CUDA path computes dynamic shared memory and calls
  `cuFuncSetAttribute` without the explicit max-shared-memory preflight used by
  the Pibar CUDA prototype.  Add a source guard or CUDA smoke after documenting
  the expected failure mode.
- Kernel performance/control environment variables and production-vs-prototype
  status remain spread across runtime modules.  Publish the supported kernel
  environment surface and classify retained paths before changing logic.
- The C++ Newick parser accepts a narrow unquoted-label dialect and splits
  multi-tree gene input on semicolons.  That supported subset is now documented
  in the README, public model/dataset docstrings, and pybind docstrings before
  any parser replacement work.  CPU preprocessing regressions now pin accepted
  simple-Newick inputs, semicolon-optional final records, multi-record gene
  files, gene multifurcation binarization, and rejections for multiple species
  trees, non-binary species trees, unary gene nodes, and unsupported metadata
  after branch lengths.
- Scheduler surface area remains high.  Python helpers such as `collate_wave`,
  `split_phase_waves`, and `compute_clade_waves` appear used only by tests/docs,
  while the global scheduler combines several heuristics whose objective,
  ordering stability, and performance intent are not fully documented.  The
  pybind scheduler/stat exports likewise remain a broad diagnostic ABI surface.
  Decide which helpers are supported diagnostics before isolating or removing
  any of them.
- RecPhyloXML traversal assumptions are now documented before parser/counter
  changes.  The README defines the supported sampled XML subset as
  `recGeneTree` blocks with `clade` nodes, `eventsRec` containers, and
  `speciation`, `duplication`, `branchingOut`, `transferBack`, `loss`, and
  `leaf` events.  It also states that gpurec-generated sample XML files are
  expected to contain one `recGeneTree` per file while the shared event-count
  traversal can read multiple `recGeneTree` blocks in compatibility inputs.
  Behavior changes around first-clade selection or global origination counting
  still need dedicated regressions.
- `json_dumps_strict()` now documents its sanitizing contract.  The helper name
  refers to standards-compliant JSON output: non-finite floats are converted to
  JSON `null` before dumping, rather than being rejected or emitted as
  Python-only `NaN`/`Infinity` tokens.  A repository hygiene guard keeps that
  contract visible beside the helper.
- Release metadata is still blocked by the missing license decision.  The
  checker and tests intentionally require top-level license metadata; a human
  license choice is needed before adding `LICENSE`, `pyproject.toml` metadata,
  and classifiers.
- Legacy HOGENOM scripts and notebooks are documented as checkout-local, but
  several large launchers still duplicate workflow logic and optimizer
  schedules outside the supported CLI.  Record behavior worth preserving before
  migration or deletion; for the fixed local profiler, add a help/argument smoke
  or an explicit fixed-profiler contract.
- Test and CI coverage gaps remain visible.  CPU CI does not enforce CUDA or
  kernel tests, the Rust JSON integration smoke intentionally exercises only a
  trivial speciation path, and workflow tests are large and private-API-heavy.
  The stochastic backtracking and Rust JSON fixture contracts are now
  documented beside the checked fixtures and guarded by repository hygiene.
  Test subprocess calls also have explicit timeouts guarded by repository
  hygiene.  The release/CI hygiene checks now parse `cpu-unit.yml` with PyYAML
  and assert package, Rust, matrix, permissions, concurrency, and example-data
  contracts against named jobs and steps instead of whole-file wording
  snapshots.  Continue simplifying the remaining wording-heavy tests where a
  structured source of truth exists.

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
- `python -m py_compile tests/unit/test_workflow.py`: passed after adding the
  public API guards for batch materialization and explicit-theta full-loss
  streaming.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_workflow.py -q -k 'materialize_batches or full_loss_for_theta'`:
  3 passed, 423 deselected after adding those guards.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_workflow.py -q`:
  426 passed after adding those guards.
- `cargo test --locked --manifest-path crates/gpurec-backtrack/Cargo.toml rejects_matrix_shape_overflow_without_panicking`:
  1 passed after adding the Rust matrix-overflow validation regression.
- `cargo test --locked --manifest-path crates/gpurec-backtrack/Cargo.toml`:
  10 library tests and 4 CLI tests passed after documenting the Rust schema and
  making matrix shape validation overflow-safe.
- `python -m py_compile tests/unit/test_workflow.py tests/integration/test_uniform_chunked_model.py`:
  passed after adding the public API guards and narrowing the integration-test
  fixture skip.
- `python -m pytest --collect-only -q tests/integration/test_uniform_chunked_model.py`:
  4 tests collected after narrowing the `test_trees_1000` skip.
- `cargo fmt --manifest-path crates/gpurec-backtrack/Cargo.toml --check`:
  passed after the Rust schema and validation changes.
- `python -m pytest --collect-only -q`: 841 tests collected after this audit
  pass.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit -q -m "unit and not gpu"`:
  812 passed, 1 skipped, 6 deselected after this audit pass.
- `python -m py_compile tests/unit/test_repository_hygiene.py`: passed after
  adding notebook and script ownership guards.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_repository_hygiene.py -q -k 'notebook or ownership or hogenom_alerax_rate_evaluator'`:
  3 passed, 25 deselected after adding those guards.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_repository_hygiene.py -q`:
  28 passed after adding notebook and script ownership guards.
- `python -m pytest --collect-only -q`: 843 tests collected after adding the
  ownership guards.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit -q -m "unit and not gpu"`:
  814 passed, 1 skipped, 6 deselected after adding the ownership guards.
- `python -m py_compile tests/unit/test_repository_hygiene.py`: passed after
  adding the optimizer-mode README guard.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_repository_hygiene.py -q -k 'optimizer_modes or ownership or notebook'`:
  3 passed, 26 deselected after adding the optimizer-mode guard.
- `cargo test --locked --manifest-path crates/gpurec-backtrack/Cargo.toml rejects_output_file_when_output_dir_is_set`:
  1 passed after fixing Rust CLI directory-mode parsing.
- `cargo fmt --manifest-path crates/gpurec-backtrack/Cargo.toml --check`:
  passed after the Rust CLI parsing change.
- `cargo test --locked --manifest-path crates/gpurec-backtrack/Cargo.toml`:
  10 library tests and 5 CLI tests passed after the Rust CLI parsing change.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_repository_hygiene.py -q`:
  29 passed after adding the optimizer-mode README guard.
- `python -m pytest --collect-only -q`: 844 tests collected after the
  optimizer-mode and Rust CLI parsing guards.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit -q -m "unit and not gpu"`:
  815 passed, 1 skipped, 6 deselected after the optimizer-mode and Rust CLI
  parsing guards.
- `rg -n "\bsafe_float\b" gpurec tests docs README.md -S`: after removal, only
  audit documentation mentions remain.
- `python -m py_compile gpurec/workflow/diagnostics.py tests/unit/test_workflow.py`:
  passed after removing `safe_float()`.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_workflow.py -q -k 'diagnostics or metadata_model_name'`:
  3 passed, 422 deselected after removing `safe_float()`.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_workflow.py -q -k 'adagrad_mode or lbfgs_mode or adam_lbfgs_schedule or lbfgs_runtime_error'`:
  4 passed, 425 deselected after adding the public optimizer-mode behavior
  guards.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_workflow.py -q`:
  429 passed after adding the public optimizer-mode behavior guards.
- `python -m pytest --collect-only -q`: 847 tests collected after adding the
  public optimizer-mode behavior guards.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit -q -m "unit and not gpu"`:
  818 passed, 1 skipped, 6 deselected after adding the public optimizer-mode
  behavior guards.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_workflow.py::test_optimization_runner_completed_resume_only_refreshes_final_artifacts tests/unit/test_repository_hygiene.py::test_project_readme_documents_completed_resume_status -q`:
  2 passed after documenting and pinning completed-checkpoint resume behavior.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_workflow.py -q`:
  430 passed after adding the completed-resume regression.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_repository_hygiene.py -q`:
  30 passed after adding the completed-resume README guard.
- `python -m pytest --collect-only -q`: 849 tests collected after adding the
  completed-resume guard.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit -q -m "unit and not gpu"`:
  820 passed, 1 skipped, 6 deselected after adding the completed-resume guard.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_workflow.py::test_workflow_solver_stats_surface_e_adjoint_failure_telemetry tests/unit/test_repository_hygiene.py::test_project_readme_documents_e_adjoint_diagnostics -q`:
  failed before the diagnostics change because `solver/e_adjoint_iterations_max`
  was missing, then passed after surfacing the aggregate E-adjoint telemetry.
- `python -m py_compile gpurec/workflow/diagnostics.py tests/unit/test_workflow.py tests/unit/test_repository_hygiene.py`:
  passed after the E-adjoint diagnostics change.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_workflow.py -q`:
  431 passed after adding E-adjoint workflow diagnostics coverage.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_repository_hygiene.py -q`:
  31 passed after adding the E-adjoint README guard.
- `python -m pytest --collect-only -q`: 851 tests collected after adding the
  E-adjoint diagnostics guard.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit -q -m "unit and not gpu"`:
  822 passed, 1 skipped, 6 deselected after adding the E-adjoint diagnostics
  guard.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_workflow.py -q -k 'full_nll_per_family'`:
  3 passed, 432 deselected after documenting and pinning the genewise-only
  `full_nll_per_family()` contract.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_repository_hygiene.py::test_project_readme_documents_genewise_per_family_api_contract -q`:
  1 passed after adding the README guard for the genewise per-family API
  contract.
- `python -m py_compile gpurec/api/model.py tests/unit/test_workflow.py tests/unit/test_repository_hygiene.py`:
  passed after the genewise per-family API documentation and guard changes.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_workflow.py tests/unit/test_repository_hygiene.py -q`:
  466 passed after the genewise per-family API guard.
- `python -m pytest --collect-only -q`: 855 tests collected after adding the
  genewise per-family API guard.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit -q -m "unit and not gpu"`:
  826 passed, 1 skipped, 6 deselected after adding the genewise per-family API
  guard.
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
- `rg -n "GPUREC_LEAF_HIT_ONLY_LOGP|LEAF_HIT_ONLY_LOGP|leaf_hit_only_logp" gpurec README.md tests docs -S`:
  after removal, only this audit log mentions the retired kernel flag.
- `python -m py_compile gpurec/core/kernels/wave_backward.py tests/unit/test_repository_hygiene.py`:
  passed after removing the retired kernel flag plumbing.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_repository_hygiene.py::test_project_readme_documents_package_environment_flags tests/unit/test_repository_hygiene.py::test_retired_leaf_hit_env_flag_stays_out_of_runtime_surface -q`:
  2 passed.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_repository_hygiene.py -q`:
  33 passed after adding the retired-env-flag guard.
- `python -m pytest --collect-only -q`: 856 tests collected after adding the
  retired-env-flag guard.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit -q -m "unit and not gpu"`:
  827 passed, 1 skipped, 6 deselected after removing the retired kernel flag
  plumbing.
- Baseline `cd crates/gpurec-backtrack && cargo test`: 10 library tests, 5 CLI
  tests, and 0 doc tests passed before removing the direct `quick-xml`
  dev-dependency.
- `cd crates/gpurec-backtrack && cargo tree -i quick-xml`: before removal,
  `quick-xml` appeared both as a direct dev-dependency and transitively through
  `rustree`; after removal, only the transitive `rustree` path remains.
- Focused `WorkItem` simplification checks:
  `cargo test --lib hidden_`, `cargo test --lib samples_forced_speciation_xml`,
  and `cargo test --lib seeded_sampling_replays_transfer_xml` passed after
  replacing `Option<usize>` with `usize`.
- `cd crates/gpurec-backtrack && cargo fmt`: passed after the Rust edits.
- `cd crates/gpurec-backtrack && cargo check --all-targets`: passed after the
  Rust dependency and `WorkItem` simplifications.
- `cd crates/gpurec-backtrack && cargo test`: 10 library tests, 5 CLI tests, and
  0 doc tests passed after the Rust dependency and `WorkItem` simplifications.
- `python -m py_compile tests/unit/test_repository_hygiene.py gpurec/core/kernels/wave_backward.py`:
  passed after adding the Rust source hygiene guards.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_repository_hygiene.py::test_rust_backtracking_does_not_declare_quick_xml_directly tests/unit/test_repository_hygiene.py::test_rust_backtracking_work_items_use_concrete_clade_state -q`:
  2 passed.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_repository_hygiene.py -q`:
  35 passed after adding the Rust dependency and `WorkItem` guards.
- `python -m pytest --collect-only -q`: 858 tests collected after adding the
  Rust hygiene guards.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit -q -m "unit and not gpu"`:
  829 passed, 1 skipped, 6 deselected after the Rust cleanup guards.
- `rg -n "bench_parse|std::chrono|#include <chrono>" gpurec/core/cpp/preprocess.cpp README.md tests docs -S`:
  after removal, only audit documentation and the repository hygiene guard
  mention the retired C++ benchmark export or chrono include.
- C++ extension build/load probe after removing `bench_parse`: `_load_extension()`
  printed `preprocess_cpp`, all six retained wave-stat diagnostic exports were
  present, and `has_bench_parse=False`.
- `python -m py_compile tests/unit/test_repository_hygiene.py`: passed after
  adding the C++ export-surface guard.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_repository_hygiene.py::test_preprocess_cpp_declares_direct_standard_includes tests/unit/test_repository_hygiene.py::test_preprocess_cpp_does_not_export_unowned_bench_parse -q`:
  2 passed.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_repository_hygiene.py -q`:
  36 passed after removing `bench_parse`.
- `python -m pytest --collect-only -q`: 859 tests collected after removing
  `bench_parse`.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit -q -m "unit and not gpu"`:
  830 passed, 1 skipped, 6 deselected after the C++ benchmark export cleanup.
- `python -m py_compile gpurec/core/backward.py gpurec/optimization/implicit_grad.py gpurec/api/uniform_chunked.py tests/unit/test_repository_hygiene.py`:
  passed after removing the unused `Pi_wave_backward` `ancestors_T` argument.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_repository_hygiene.py::test_pi_wave_backward_signature_omits_unused_ancestors_t -q`:
  1 passed.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_core_backward.py tests/unit/test_implicit_grad_solver.py tests/unit/test_repository_hygiene.py -q`:
  40 passed after adding the Pi-backward signature guard.
- `python -m pytest --collect-only -q`: 860 tests collected after adding the
  Pi-backward signature guard.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit -q -m "unit and not gpu"`:
  831 passed, 1 skipped, 6 deselected after removing the unused
  `Pi_wave_backward` `ancestors_T` argument.
- `python -m py_compile gpurec/api/model.py tests/unit/test_repository_hygiene.py`:
  passed after documenting the leaf-species mapping contract.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_repository_hygiene.py::test_project_readme_documents_leaf_species_mapping_contract -q`:
  1 passed.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_repository_hygiene.py -q`:
  38 passed after adding the leaf-species mapping README guard.
- `python -m py_compile gpurec/api/model.py tests/unit/test_repository_hygiene.py`:
  passed after documenting the lazy-prefetch solver reconfiguration contract.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_repository_hygiene.py::test_solver_reconfiguration_docs_cover_lazy_prefetch_contract -q`:
  1 passed.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_repository_hygiene.py -q`:
  39 passed after adding the lazy-prefetch solver reconfiguration guard.
- `python -m pytest --collect-only -q`: 862 tests collected after the refreshed
  subagent audit documentation and solver reconfiguration docs.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit -q -m "unit and not gpu"`:
  833 passed, 1 skipped, 6 deselected.
- `python -m py_compile gpurec/core/model.py tests/unit/test_alerax_family_input.py`:
  passed after adding cached `leaf_col_index` upper-bound validation.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_alerax_family_input.py::test_cached_family_preprocess_rejects_leaf_species_indexes_outside_species_range -q`:
  1 passed.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_alerax_family_input.py -q`:
  29 passed after adding the cache validation regression.
- `python -m pytest --collect-only -q`: 863 tests collected after adding the
  cached `leaf_col_index` validation regression.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit -q -m "unit and not gpu"`:
  834 passed, 1 skipped, 6 deselected.
- Baseline `cd crates/gpurec-backtrack && cargo test --lib rejects_`: first
  failed the new non-finite log-payload and out-of-range `leaf_species`
  regressions before Rust validation was added.
- `cd crates/gpurec-backtrack && cargo test --lib rejects_`: 6 passed after
  adding Rust payload validation.
- `cd crates/gpurec-backtrack && cargo fmt`: passed after the Rust payload
  validation changes.
- `cd crates/gpurec-backtrack && cargo check --all-targets`: passed after the
  Rust payload validation changes.
- `cd crates/gpurec-backtrack && cargo test`: 12 library tests, 5 CLI tests,
  and 0 doc tests passed after the Rust payload validation changes.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_repository_hygiene.py tests/unit/test_alerax_family_input.py -q`:
  68 passed after the Rust and cache-validation changes.
- `python -m py_compile tests/unit/test_repository_hygiene.py tests/unit/test_release_metadata.py tests/unit/test_workflow.py tests/unit/test_global_wave_scheduler.py tests/unit/test_cli_workflow.py tests/unit/test_examples.py tests/integration/test_rust_backtracking_fixture.py`:
  passed after adding explicit subprocess timeouts to tests.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_repository_hygiene.py::test_tests_subprocess_calls_have_explicit_timeouts -q`:
  1 passed.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_examples.py tests/unit/test_global_wave_scheduler.py::test_collate_gene_families_validates_split_lengths_under_optimized_python tests/unit/test_cli_workflow.py::test_cli_rejects_hydra_yaml_config_before_workflow_import -q`:
  6 passed after adding subprocess timeouts.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_release_metadata.py -q`:
  39 passed, 1 skipped after adding subprocess timeouts.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_workflow.py -q -k 'wildcard or import_smoke or cli or export'`:
  14 passed, 420 deselected after adding subprocess timeouts.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/integration/test_rust_backtracking_fixture.py -q`:
  1 passed after adding the Cargo subprocess timeout.
- `python -m py_compile gpurec/api/model.py gpurec/api/uniform_chunked.py gpurec/core/model.py tests/unit/test_repository_hygiene.py`:
  passed after documenting the supported simple-Newick input subset.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_repository_hygiene.py::test_newick_input_subset_is_documented_on_public_surfaces -q`:
  1 passed after adding the public Newick-subset documentation guard.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_repository_hygiene.py -q`:
  41 passed after adding the Newick-subset documentation guard.
- `python -m pytest --collect-only -q`: 865 tests collected after adding the
  Newick-subset documentation guard.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_alerax_family_input.py::test_gene_dataset_accepts_documented_simple_newick_subset tests/unit/test_alerax_family_input.py::test_gene_dataset_rejects_unsupported_newick_dialect_cases -q`:
  first failed because the documented whitespace contract was too broad for
  branch lengths with whitespace after `:`, then 5 passed after the public
  contract and fixture were tightened to the actual parser behavior.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_alerax_family_input.py -q`:
  34 passed after adding Newick parser compatibility regressions.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_repository_hygiene.py -q`:
  41 passed after tightening the Newick documentation guard.
- `python -m pytest --collect-only -q`: 870 tests collected after adding the
  Newick parser compatibility regressions.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit -q -m "unit and not gpu"`:
  841 passed, 1 skipped, 6 deselected after adding the Newick parser
  compatibility regressions.
- `python -m py_compile tests/unit/test_repository_hygiene.py tests/integration/test_rust_backtracking_fixture.py tests/integration/test_stochastic_backtracking.py`:
  passed after adding fixture-contract documentation and guards.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_repository_hygiene.py::test_checked_fixture_contracts_are_documented -q`:
  1 passed after adding checked-fixture documentation guards.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/integration/test_rust_backtracking_fixture.py -q`:
  2 passed after adding the Rust JSON fixture contract assertion.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/integration/test_stochastic_backtracking.py --collect-only -q`:
  2 collected after naming the `test_trees_3` expected shape constants.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_repository_hygiene.py -q`:
  42 passed after adding fixture-contract documentation guards.
- `python -m pytest --collect-only -q`: 872 tests collected after adding the
  fixture-contract documentation guards.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit -q -m "unit and not gpu"`:
  842 passed, 1 skipped, 6 deselected after adding the fixture-contract
  documentation guards.
- `python -m py_compile tests/unit/test_release_metadata.py`: passed after
  converting the CPU CI workflow checks to structured YAML/job/step assertions.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_release_metadata.py::test_cpu_ci_builds_and_smokes_release_artifacts tests/unit/test_release_metadata.py::test_cpu_ci_workflow_uses_minimal_permissions_and_concurrency tests/unit/test_release_metadata.py::test_cpu_ci_matrix_covers_declared_python_versions tests/unit/test_release_metadata.py::test_cpu_ci_runs_rust_backtracking_gate tests/unit/test_release_metadata.py::test_readme_scopes_example_config_to_source_artifacts -q`:
  5 passed after the structured workflow assertion cleanup.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_release_metadata.py -q`:
  39 passed, 1 skipped after adding the PyYAML-backed workflow parser helper and
  step-scoped assertions.
- `python -m pytest --collect-only -q`: 872 tests collected after the
  structured workflow assertion cleanup.
- `python -m py_compile gpurec/workflow/diagnostics.py tests/unit/test_repository_hygiene.py`:
  passed after documenting the strict JSON serializer sanitization contract.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_repository_hygiene.py::test_strict_json_serializer_documents_sanitizing_contract tests/unit/test_workflow.py::test_workflow_jsonl_diagnostics_sanitize_nonfinite_values tests/unit/test_workflow.py::test_workflow_json_diagnostics_write_strict_file -q`:
  3 passed after adding the `json_dumps_strict()` documentation guard.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_repository_hygiene.py::test_project_readme_documents_sampling_output_layout tests/unit/test_repository_hygiene.py::test_strict_json_serializer_documents_sanitizing_contract -q`:
  2 passed after extending the README sampling-output guard to cover aggregate
  file formats, normalization semantics, and the supported sampled
  RecPhyloXML subset.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_repository_hygiene.py -q`:
  43 passed after adding the RecPhyloXML subset and strict JSON serializer
  documentation guards.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_release_metadata.py -q`:
  39 passed, 1 skipped after the README sampling-output documentation update.
- `python -m pytest --collect-only -q`: 873 tests collected after adding the
  RecPhyloXML subset and strict JSON serializer documentation guards.
- `python -m py_compile gpurec/api/uniform_chunked.py tests/unit/test_repository_hygiene.py tests/unit/test_workflow.py`:
  passed after documenting bf16 as a direct API-only experimental dtype.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_repository_hygiene.py::test_bfloat16_policy_is_documented_as_direct_api_only tests/unit/test_workflow.py::test_bfloat16_is_direct_uniform_api_only -q`:
  4 passed after adding the direct-API/workflow dtype boundary guard and direct
  dtype-gate coverage for workflow config, `GeneReconModel`, and
  `UniformChunkedReconModel`.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_repository_hygiene.py -q`:
  44 passed after adding the bf16 policy documentation guard.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_workflow.py -q -k 'dtype or bfloat16'`:
  27 passed, 410 deselected after adding the bf16 dtype-gate regression.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_release_metadata.py -q`:
  39 passed, 1 skipped after the bf16 README documentation update.
- `python -m pytest --collect-only -q`: 877 tests collected after adding the
  bf16 dtype-policy regression.
- `python -m py_compile gpurec/core/likelihood.py tests/unit/test_origination_probs.py tests/unit/test_repository_hygiene.py`:
  passed after documenting the prepared-origination-probability trust boundary.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_origination_probs.py::test_prepared_origination_probs_are_internal_trust_boundary tests/unit/test_repository_hygiene.py::test_prepared_origination_probs_trust_boundary_is_documented -q`:
  4 passed after adding behavior and docstring guards for
  `assume_prepared=True`.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_origination_probs.py -q`:
  9 passed after adding the prepared-origination-probability trust-boundary
  regression.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_repository_hygiene.py -q`:
  45 passed after adding the trust-boundary documentation guard.
- `python -m pytest --collect-only -q`: 881 tests collected after adding the
  prepared-origination-probability trust-boundary regression.

## Recommended Next Order

1. Continue turning documented findings into focused guards before runtime
   redesigns.  Contract coverage now exists for duplicate direct
   `family_names`, oversized `clade_budget`, `ancestors_T=None`, and LBFGS
   `max_eval` evaluation accounting, sampling subprocess timeout behavior, and
   direct C++ `max_wave_size` validation, plus public workflow optimizer modes
   and Newick parser compatibility, plus checked fixture contracts.
2. Fix remaining documentation-only staleness as it is found in touched areas.
3. Prefer validation/test fixes that do not need policy choices; the next small
   candidates are remaining wording-heavy release/docs checks with an available
   structured source of truth.
4. Make low-risk hygiene changes with tests: slow markers and any future
   warning filters only if scoped to a specific dependency warning.
5. Only then consider behavior changes for backward small-`S`, bf16 dtype
   implementation, DTS parameter shape semantics, CUDA Pibar fallback policy,
   RecPhyloXML assumptions, and sampling aggregate behavior.
