# Repo-Wide Audit, 2026-05-21

This note records a read-only audit pass over the tracked repository state on
branch `lean-fast-path`.  It is intentionally documentation-only: no runtime
logic was changed before recording the findings.

## Scope And Evidence

The bullets below are the initial read-only audit snapshot, not live repository
metrics.  Later verification entries record current collection counts as files,
docs, and guards are added during the audit.

- Initial tracked scope snapshot: `git ls-files | wc -l` reported 134 files.
- Initial source-like size snapshot: `git ls-files '*.py' '*.cpp' '*.hpp' '*.rs' '*.R' | xargs wc -l`
  reported 43,009 lines.
- Initial test inventory snapshot: `pytest --collect-only -q` collected 823 tests.
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
`gpurec/core/kernels/dts_fused.py`,
`gpurec/core/kernels/wave_backward.py`,
`gpurec/core/extract_parameters.py`, and the profiling scripts.  Some are
covered indirectly through higher-level GPU tests, but direct branch-level
evidence is thin.  The former native CUDA prototype modules named in earlier
audit slices, `pibar_vjp_cuda.py` and `wave_backward_cuda.py`, have since been
removed and are guarded against returning.

## Findings

### Runtime Contracts And Algorithm Edges

1. Small-species backward behavior is now documented as an intentional retained
   fused-path limitation.  `Pi_wave_backward` rejects non-CUDA, unsupported
   dtypes, and `S <= 256`.  README and the docs map state that the checked tiny
   CUDA config is a config/parser fixture, not an end-to-end optimizer smoke,
   because the retained Pi backward/gradient path currently requires `S > 256`
   until a small-species backward fallback is restored.

2. `ancestors_T` is optional by signature but required in practice.  The
   retained uniform-transfer E solver now rejects missing, non-tensor,
   non-square, wrong-shape, wrong-device, or wrong-dtype `ancestors_T` values
   before entering the fixed-point loop or ancestor matrix multiply.

3. Direct duplicate `family_names` are now rejected before preprocessing.  The
   direct `GeneDataset` constructor matches the AleRax parser's no-duplicate
   contract, and CPU-safe tests prove the validation runs before extension or
   IO work.

4. `clade_budget` is specified as a packing target, not a hard cap.  Both
   `clade_first_fit` and `depth_first_fit` allow one oversized family to occupy
   its own batch, and scheduler tests pin that behavior for future memory-policy
   changes.

5. Adaptive root trace behavior is now specified and guarded.  `Pi_wave_forward`
   keeps the existing fixed `[fixed_iters, n_roots]` trace shape and carries
   each early-converged root wave's last computed value through its unused tail
   rows, so consumers do not see preallocation sentinels after adaptive stops.
   Unit coverage checks the carry-forward helper on CPU and the CUDA trace
   parity test covers the early-stop path when CUDA fixtures are available.

6. DTS parameter shape semantics when `G == S` are now documented before any
   runtime unification.  Public model paths use unambiguous theta shapes and
   normalize genewise scalar event vectors to `[G, 1]` before the retained DTS
   kernels.  README and source docstrings state that direct callers should
   avoid bare `[G]` vectors when `G == S`: the forward DTS helper treats a 1-D
   parameter with `numel() == S` as shared species-indexed, while the retained
   backward helper with `family_idx` treats a one-dimensional tensor as
   family-indexed.  Direct callers needing parity should use `[G, 1]` or
   `[G, S]` until a runtime shape-policy change is made.

7. The former direct C++ scheduler `max_wave_size` gap is now guarded.  The
   retained diagnostic exports were covered by positive-`max_wave_size`
   validation before the C++ preprocessing source was retired from the tracked
   runtime tree; repository hygiene now also guards that the removed C++ source
   and `preprocess_cpp.py` wrapper do not return.

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

14. `BatchedLBFGS.max_eval` evaluation accounting is now guarded.  Tight-budget
    tests assert both the optimizer's `state["func_evals"]` counter and the
    observed closure-call count stay within `max_eval`; when Armijo probes use
    the final allowed evaluation, the accepted probed loss is returned without a
    budget-breaking gradient refresh or invalid curvature-history update.

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

18. `UniformChunkedReconModel.loss_and_grad(reduction="full_sum_estimate")`
    now has direct CPU-safe coverage.  The regression monkeypatches the internal
    chunk evaluator and verifies that `total_families / selected_families`
    scales both the returned loss and gradient.

19. `gpurec.workflow.checkpoint.load_checkpoint_config()` has been removed from
    the package code and is not exported by `gpurec.workflow` or the package top
    level.  Python tooling that needs checkpoint configuration metadata should
    use `load_checkpoint(path)["config"]` from `gpurec.workflow.checkpoint` and
    pass it to `RunConfig.from_dict(...)`; no separate public
    `load_checkpoint_config` helper is supported.

20. `GeneReconModel.materialize_batches()` and
    `GeneReconModel.full_loss_for_theta(theta)` now have explicit public
    contracts in README and method docstrings, plus CPU-safe unit guards.
    `materialize_batches()` builds every resident batch static state and returns
    a metadata-list copy; `full_loss_for_theta(theta)` streams all resident
    batches with the gradient-producing path for differentiable explicit-theta
    probes and with the loss-only path under `torch.no_grad()`.

### Scripts, Rust, Profiling, And Examples

21. Legacy HOGENOM launchers have inconsistent path override support.
    `scripts/README.md` labels them legacy checkout-local scripts, while
    `scripts/optimize_hogenom_ccp_global_uniform.py:21` and
    `scripts/optimize_hogenom_ccp_specieswise_uniform.py:26` hard-code local
    data paths and expose mostly optimizer/regularization flags.  Document
    which launchers are fixed-dataset before shared optimizer changes.

22. `profiling/bench_uniform_forward_backward_pipeline.py` previously
    referenced missing `docs/forward-backward-full-pipeline-plan.md`; the
    reference has been removed, and the retained profiler contract now lives in
    README/source-checkout benchmark docs.  A refreshed stale-import finding is
    also closed: the benchmark now imports the current underscored chunk
    dataclasses it already depends on, uses `compute_nll` instead of the
    deprecated `compute_log_likelihood` alias, and has a CPU-safe `--help`
    smoke guard.

23. `scripts/make_hogenom_branchscale_penalty_report.py` is now documented as
    a checkout-local one-off report builder for the original branchscale
    penalty sweep.  It intentionally discovers only `penalty_*` child
    directories, while newer launchers can create timestamped names, and its
    report text preserves the original hard-coded date and "1325 branch
    multipliers" caption until the script is either migrated to data-driven
    supported CLI reporting or archived/deleted.

24. `configs/hogenom_ccp_wandb.yaml` is documented as a checkout-local full
    HOGENOM Hydra/W&B experiment config rather than a portable smoke example.
    The docs map now points users to it separately from the flat JSON example
    config.

25. `examples/minimal-run-config.json` defaults to `"device": "cuda"` even
    though the tiny fixtures are otherwise portable.  README and the docs map
    now state that it is a source-checkout/source-archive CUDA smoke for the
    retained optimized path, not a CPU fallback.  `scripts/README.md` also
    makes the supported CLI example explicit about `--device cuda`.

26. Rust backtracking payload schema and validation boundaries are now
    documented in `crates/gpurec-backtrack/src/lib.rs`.  The source documents
    row-major matrix layout, base-2 log units, the `-1e300` negative-infinity
    sentinel, postorder species indexing, clade-indexed leaf/split bounds, and
    natural-space nonnegative origination weights.  Matrix shape products use
    checked multiplication and report `InvalidInput` on overflow, and the Rust
    preparation path rejects non-finite log/weight payloads, out-of-range leaf
    species, invalid split clade IDs, negative origination weights, and
    nonpositive `max_events` before sampling.

27. Profiling/evaluation scripts that encode brittle external file-format
    assumptions are now labeled as checkout-local utilities before broader
    script migration.  In particular, `profiling/evaluate_hogenom_alerax_rates.py`
    documents its HOGENOM AleRax rate-file assumptions instead of presenting
    itself as a general parser.

28. The Rust backtracking CLI now rejects an ignored positional output file
    when `--samples 1 --output-dir DIR input.json output.xml` is passed, matching
    the multi-sample directory-mode contract.

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

32. GPU/data-heavy tests under `tests/unit` are now explicitly documented as the
    `unit` plus `gpu` marker overlap.  The 1000-family CUDA fixture cases carry
    local `@pytest.mark.slow` markers, and `tests/README.md` documents the
    CPU-only audit gate as `-m "unit and not gpu"`.

33. `tests/unit/test_release_metadata.py` mirrors docs and GitHub Actions YAML
    with many exact substring assertions.  These guards catch release drift, but
    they are brittle during harmless wording or workflow layout changes.

34. `tests/unit/test_workflow.py` is an oversized mixed-surface test module at
    5,123 lines.  It covers exports, config, checkpointing, optimization,
    backtracking commands, sampling, and more.  Splitting it by behavior would
    improve ownership and reduce stale-test risk.

35. `pytest.ini` no longer globally ignores all `DeprecationWarning` and
    `PendingDeprecationWarning`.  Repository hygiene now rejects blanket warning
    ignores and leaves room only for targeted dependency filters.

36. `tests/__init__.py` now accurately documents the retained test-package
    namespace and current `unit`, `integration`, `kernel`, `gpu`, and `slow`
    ownership, while preserving imports such as `tests.unit.alerax_helpers`.

37. CLI help smoke tests are sensitive to stale installed console scripts.  In
    this checkout, `which gpurec` resolved to `/home/enzo/miniforge3/bin/gpurec`,
    whose entry point imports `gpurec.cli.reconcile`.  The repo-local
    `python -m gpurec.cli --help` command passed, but
    `tests/unit/test_release_metadata.py::test_cli_help_smokes_are_quiet_on_cpu`
    failed through the stale PATH executable.  This is an environment/setup
    fragility to document or guard in release checks.

38. `log_every` is now documented as console progress throttling, not history
    logging.  Source inspection showed the workflow records history rows every
    optimizer step through `self._record(row)`, while `config.log_every` only
    gates the stdout progress print.  README, CLI help, and `RunConfig` source
    comments now state that history JSONL is recorded every step and
    `--log-every` only controls console progress output.

### Subagent Refresh Findings

- Workflow optimizer modes are now documented and guarded.  README and
  `docs/run-config-reference.md` cover `adagrad`, `lbfgs`, `adam-lbfgs`, and
  the production defaults, while fake-model workflow tests cover Adagrad rows,
  active LBFGS accounting, the Adam-to-LBFGS phase transition, and LBFGS runtime
  failure status.
- E-adjoint solver failure telemetry is now explicit diagnostics.  Workflow
  history records iteration, relative-residual, success, and failed-batch
  aggregates; direct uniform-chunk `loss_and_grad()` stats expose the same
  nonconvergence fields.  The retained policy remains diagnostic-only rather
  than fail-fast.
- `gpurec.workflow.diagnostics.safe_float()` was confirmed unused outside its
  direct unit test and removed rather than promoted as public workflow API.
- Completed-resume status is now documented and asserted.  A checkpoint whose
  `next_step` already equals configured `steps` refreshes final artifacts and
  returns the ordinary `not_converged`/`max_steps` status instead of performing
  another optimizer step.
- RunConfig and CLI reference docs now track the current option surface.
  `docs/run-config-reference.md` is guarded against the `RunConfig` dataclass
  fields, and CLI parser tests compare parser destinations to `RunConfig`
  fields so constraints such as even `fixed_iters_pi`, optimizer modes, and
  checkpoint cadence are not only implicit in scattered tests.
- The former overbroad `test_uniform_chunked_model.py` fixture skip is closed:
  large `test_trees_1000` availability is checked only by tests that need it,
  while the HOGENOM unrooted-Newick case has its own `hogenom_bench` guard.
- Several GPU tests are still smoke-heavy: Adam/LBFGS integration checks now
  require post-step NLL non-increase, and the HOGENOM unrooted parsing check now
  asserts per-family likelihood plus direct/PyTorch gradient consistency.  The
  specieswise backward check now compares the retained backward gradient against
  a directional finite difference.  Prefer before/after NLL decrease or
  reference-close assertions when local data makes that practical.
- Adaptive iteration coverage now includes forced-max parity and a slow GPU
  loose-tolerance guard that asserts E/Pi solver iterations stop before the
  configured maximum.  Broaden it with more data-backed close-reference cases
  when local GPU time permits.
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
- Native CUDA prototype paths are no longer part of the production runtime.
  The backward source omits `GPUREC_CUDA_SELF_LOOP_*`,
  `GPUREC_CUDA_PIBAR_FROM_UD`, and related prototype selectors, while
  repository hygiene asserts the former `wave_backward_cuda.py` and
  `pibar_vjp_cuda.py` modules stay absent.  Historical performance logs may
  still mention those routes as archived provenance, not current fallback
  policy.
- DTS parameter shape precedence is now documented for direct callers before
  changing either implementation.  Public model paths normalize genewise scalar
  event vectors to `[G, 1]`; direct DTS callers should avoid bare `[G]` vectors
  when `G == S` because forward and backward retained helpers still interpret
  one-dimensional tensors differently.
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
  `compute_clade_waves`, and retired C++ wave-stat exports now have an
  ownership table in `docs/runtime-surface-pruning-plan-2026-05-21.md`.
  Production-owned Rust/PyO3 paths should be kept or hidden only behind
  replacements; test-only and retired diagnostic exports are guarded against
  returning.  `bench_parse` is no longer retained as public surface.
- The legacy leaf-to-species fallback is now documented in public user-facing
  docs: direct `from_trees` inputs map `Species_gene` to species `Species` and
  labels without `_` to the full label, while AleRax `mapping` entries and
  explicit `leaf_species_maps` should be used when labels do not follow that
  convention.
- The refreshed workflow audit found that checkpoint compatibility validates the
  v1 identity slice and now reconstructs the stored config with
  `RunConfig.from_dict(...)` before comparison.  The current documented boundary
  is: `load_checkpoint()`
  requires `family_names`, `species_names`, and config identity keys
  `species_tree`, `families_file`, `mode`, `start`, and `max_families`, while
  `validate_checkpoint_model_compatibility()` validates the stored config with
  `RunConfig.from_dict(...)`, then compares those values with the active
  `RunConfig` and rebuilt model before theta restore, normalizing only path
  fields.  The focused optimization-resume incompatibility regression now covers
  every documented config identity key, not only `mode`, and direct
  compatibility tests reject invalid stored RunConfig metadata before restore.
- The workflow checkpoint submodule support boundary is now explicit below the
  lazy `gpurec.workflow` exports.  Top-level workflow exports remain the stable
  shortcut surface; `gpurec.workflow.checkpoint` now declares
  `save_checkpoint`, `load_checkpoint`, `restore_model_theta`,
  `validate_checkpoint_model_compatibility`, and `CHECKPOINT_VERSION` as
  supported lower-level helpers for advanced tooling that needs the versioned
  checkpoint payload directly.  The helpers are intentionally not top-level
  `gpurec.workflow` or `gpurec` shortcuts.
- `profiling/bench_uniform_forward_backward_pipeline.py` has been refreshed
  after the stale benchmark finding: its `--help` path imports successfully,
  it aliases the current underscored chunk dataclasses used by the retained
  private benchmark helpers, and it calls `compute_nll` instead of the
  deprecated `compute_log_likelihood` compatibility alias.
- `scripts/optimize_hogenom_penalty316_kkt.py` now has the same explicit
  checkout-local contract as the branchscale report before any loader change or
  deletion.  Its source, help text, and scripts ownership matrix document the
  legacy HOGENOM W&B optimizer dependency, CUDA branchscaled mode with W&B
  disabled, default penalty 316.22776601683796, 100 Adam warmup steps,
  Strong-Wolfe LBFGS, L1 KKT residual checks, timestamped output directories,
  `latest_run.txt`, and the archive/delete-or-migrate criterion.
- `active_mask_from_rhs_absmax_fused()` now documents its bf16 boundary before
  any dtype cleanup.  The helper remains a private retained-kernel helper that
  accepts fp32/fp64/bf16 CUDA tensors for standalone row-mask experiments, while
  the public retained `Pi_wave_backward` path still supports only fp32/fp64 and
  rejects bf16 before this helper is reached.
- The follow-up core/API explorer added a guarded decision table to
  `docs/runtime-surface-pruning-plan-2026-05-21.md` before any behavior change.
  Unresolved surfaces include bool acceptance in direct API float validators,
  direct shape tests for `gpurec/core/extract_parameters.py`, the unused
  `_normalize_family_tree_paths()` compatibility alias, unclear export status for
  `normalize_family_chunk_size()`, internal ownership of `UniformChunkedState`,
  chunked `nll_per_family()` diagnostics, and classification of
  `implicit_grad_loglik_vjp_wave()` as internal bridge versus low-level API.
- The follow-up workflow/backtracking explorer added a guarded decision table to
  the same pruning plan.  Remaining unresolved surfaces include local scripts
  that should close per-family/per-chunk models, narrow optimizer-state discard
  errors, test-only dynamic CLI compatibility attributes, branch-level Rust
  sampler term coverage.
- The stale Rust sampler help-marker finding is now fixed.  The Python
  backtracking preflight requires help text for `--samples`, `--seed`,
  `--output-dir`, `--max-events`, and `input.json`, and the missing-marker error
  names the stale flags before a sampler is accepted.
- The LBFGS post-step nonfinite finding is now fixed.  After LBFGS line search,
  optimization resume/runner diagnostics still evaluate the saved current theta,
  but that evaluation now repeats the same finite loss/gradient guard used after
  Adam/Adagrad updates and fails with `nonfinite_objective_or_gradient` before an
  optimizer-step row or checkpoint can be recorded.
- The Python 3.10 TOML fallback finding is now guarded.  The private
  `_parse_minimal_pyproject()` compatibility parser has direct fixture tests for
  the release-metadata subset needed when `tomllib` is unavailable, including
  readme/license string-or-table values, multiline classifiers, project URLs,
  ignored unrelated tables, and the current project fields.
- The tests/Rust/docs follow-up explorer found a related Python 3.10 collection
  risk outside the release checker itself.  TOML-reading unit modules imported
  `tomllib` at module import time even though the advertised CPU matrix includes
  Python 3.10, where the stdlib module is unavailable unless a `tomli`
  compatibility dependency is installed.  That test-collection issue is now
  fixed: TOML-reading unit modules use a `tomllib`/`tomli` conditional import,
  the Python-version-scoped `tomli` dependency is in the dev extra, and release
  metadata tests guard the dependency while Python 3.10 remains supported.  The
  same pass also found an unguarded Rust CLI multi-sample `--output-dir`
  contract: the library covers multi-sample sequencing and the integration
  fixture covered only single-file output.  That gap is now guarded: the
  CPU-only Rust fixture integration test runs the real CLI with
  `--samples 2 --output-dir <tmpdir>`, checks `sample_0.xml` and
  `sample_1.xml`, and parses both files against the deterministic RecPhyloXML
  fixture contract.
- The workflow/scripts follow-up explorer found two additional unguarded
  workflow and local-utility surfaces.  The mandatory final optimization
  evaluation nonfinite gap is now fixed: final evaluation repeats the finite
  loss/gradient guard, failed final evaluations mark the run as
  `failed/nonfinite_objective_or_gradient`, and the `final_eval` row records
  explicit failed final-eval markers instead of copying nonfinite metrics.
  Resume optimizer-state restore now also discards `ValueError`, `RuntimeError`,
  and `TypeError` from `optimizer.load_state_dict`, so malformed or
  backend-incompatible optimizer state is reported in history instead of
  aborting resume.
  The maintained full-pipeline profiling benchmark now rejects invalid count
  controls before setup, including zero or negative family counts, chunk sizes,
  iteration counts, and wave sizes.  Older checkout-local profiling proposals
  should keep getting the same parser-level treatment before they are promoted.
- The core/API follow-up explorer found three current contracts to guard before
  scheduler or parameter-shape refactors.  The direct `build_wave_layout()`
  family-index gap is now fixed: family count/offset metadata must be provided
  together, have matching lengths, stay in bounds, avoid overlaps, and cover
  every clade before `family_idx` is materialized.  Explicit `theta_init` and
  `full_loss_for_theta(theta)` tensor validation is now fixed as well: shared
  raw-theta checks reject wrong shapes, non-floating tensors, and nonfinite
  values before parameter extraction, CUDA checks, or streaming work.  The stale
  `collate_gene_families()` docstring found in the same pass is also fixed and
  guarded so source docs point at preprocessed CCP payloads and
  `build_wave_layout()` instead of removed
  `preprocess_gene_with_species` / `likelihood_2.py` surfaces.

## Adequately Covered Or Lower-Risk Areas

- `gpurec/core/_helpers.py`, `gpurec/core/log2_utils.py`,
  `gpurec/core/terms.py`, `gpurec/core/scheduling.py`,
  `gpurec/core/species.py`, and `gpurec/core/memory_policy.py` have focused
  unit coverage.
- `gpurec/core/batching.py` and `gpurec/core/batch_planning.py` have strong
  scheduler and layout coverage, including the oversized-family
  `clade_budget` edge.
- `gpurec/core/model.py` cache validation, AleRax parsing, and direct
  duplicate-family-name constructor validation are well covered.
- Workflow checkpointing, CLI parse failures, public export guards, and
  backtracking command failure paths have broad unit coverage.

## Deletion And Simplification Candidates

- Keep retired pybind debug exports out of the current Rust/PyO3 preprocessing
  manifest unless a maintained diagnostic owner reintroduces them explicitly.
- The legacy pybind `preprocess()` wrapper is absent from the current native
  manifest.  The current Python runtime routes through
  `preprocess_multiple_families(..., include_details=True)`.
- Decide whether `preprocess_multiple_families(..., include_details=False)` is
  a public extension mode or dead compatibility surface; production Python
  callers request details, including species-only empty-family preprocessing.
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
  names and benchmark commands are historical provenance from an original
  performance workspace, not a current reproducible command set.
- `tests/unit/test_repository_hygiene.py` now scans README/docs inline path
  references and requires absent tracked-looking paths to be explicitly marked
  historical, optional, missing, or otherwise not reproducible from a clean
  checkout.
- `scripts/make_hogenom_branchscale_penalty_report.py` now documents its
  legacy `penalty_*` sweep layout, timestamped-output mismatch, and hard-coded
  original-report assumptions before any loader/report rewrite.  `scripts/README.md`
  records the same expected layout and delete-or-migrate criterion.
- `scripts/README.md` now shows `--device cuda` in its supported CLI example
  and states that CPU-only checkouts can run help/config/package/unit hygiene
  checks but not the optimized likelihood workflow.
- `docs/README.md` now distinguishes the source-checkout/source-archive CUDA
  smoke config from the checkout-local HOGENOM Hydra/W&B config, so users do
  not have to infer portability from README alone.
- `tests/unit/test_release_metadata.py` now guards the existing
  release-readiness license/no-publish wording without choosing a license.
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
  required for the retained uniform-transfer E solver and has the expected
  tensor type, `[S, S]` shape, device, and dtype, replacing indirect
  matrix-multiply failures with clear `ValueError`s.
- `tests/unit/test_origination_probs.py` now covers both `E_step` and
  `E_fixed_point` missing- and malformed-`ancestors_T` errors.
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
  fields into history rows.  The implicit-gradient solver docstrings now state
  that BiCGSTAB returns the current best iterate with
  `_SolveStats(success=False)` on nonconvergence, and that the retained
  E-adjoint gradient path consumes that iterate while forwarding the failure
  flag as workflow telemetry.
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
- `BiCGSTAB` failure telemetry is intentionally diagnostic-only in the retained
  implicit-gradient path: the current best E-adjoint iterate is consumed even
  when the solver reports failure, and workflow diagnostics surface the failure
  after the gradient step.  Treat any fail-fast or retry policy as a future
  logic change requiring a numerical criterion and regression coverage.
- Direct uniform-chunk evaluation now exposes E-adjoint solver stats in its
  public `loss_and_grad()` stats payload.  `_evaluate_chunked_uniform()` keeps
  loss, gradient, and reduction behavior unchanged, but maps `_SolveStats` from
  `_e_adjoint_and_theta_vjp()` into `e_adjoint_method`,
  `e_adjoint_iterations`, `e_adjoint_rel_res`, and `e_adjoint_success` so
  direct `UniformChunkedReconModel` users have the same nonconvergence
  visibility that workflow history already surfaces in aggregate form.
- The native CUDA prototype loader/fallback finding is closed for production
  runtime by deletion.  The former `wave_backward_cuda.py` and
  `pibar_vjp_cuda.py` modules are absent, the backward source omits their
  selectors and imports, and public runtime-environment docs exclude
  `GPUREC_CUDA_SELF_LOOP_*`/`GPUREC_CUDA_PIBAR_FROM_UD` as supported contracts.
  Historical performance logs can still describe those experiments as archived
  provenance.
- The former native CUDA launcher dynamic-shared-memory finding is likewise
  historical for the production branch.  Since the launchers are absent, the
  maintained guard is absence of the modules/selectors rather than parity
  between deleted launcher implementations.
- Historical benchmark provenance is now separated from current checkout
  contracts in `docs/lean-performance-path-regression.md`.  The missing
  genewise/uniform reference docs and benchmark harness names are labeled as
  untracked historical workspace inputs, and the command block is explicitly
  not a clean-checkout reproducibility recipe.  A repository hygiene guard now
  checks README/docs inline path references for the same historical/optional
  marker when the referenced tracked-looking path is absent.
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
  First-root-clade selection and file-level origination counting for multi-tree
  compatibility inputs now have dedicated RecPhyloXML regressions before parser
  or counter changes.
- `json_dumps_strict()` now documents its sanitizing contract.  The helper name
  refers to standards-compliant JSON output: non-finite floats are converted to
  JSON `null` before dumping, rather than being rejected or emitted as
  Python-only `NaN`/`Infinity` tokens.  A repository hygiene guard keeps that
  contract visible beside the helper.
- Release metadata is still blocked by the missing license decision.  The
  checker and tests intentionally require top-level license metadata; a human
  license choice is needed before adding `LICENSE`, `pyproject.toml` metadata,
  and classifiers.
- Legacy HOGENOM scripts and notebooks are documented as checkout-local.  The
  ignored local notebook/profiling workspace now has a keep/delete/migrate
  inventory, and the fixed global-uniform, specieswise-uniform, and HOGENOM
  profiler scripts now document their unique behavior and output contracts.
  Several launchers still duplicate workflow logic outside the supported CLI;
  migrate behavior worth keeping into `gpurec.workflow` before deletion.
- Ignored test-data roots are now inventoried before relying on local-only files
  during the repo-wide audit.  `tests/README.md` and the runtime-surface pruning
  plan identify generated `test_trees_*` datasets, external HOGENOM/Davin
  roots, `tests/data.tar.gz`, and `tests/data/**/output/`
  as non-distributed local state.  Required tests should use tracked fixtures or
  a documented generator instead of depending on those ignored paths.
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
- `python -m py_compile gpurec/optimization/implicit_grad.py gpurec/api/autograd.py tests/unit/test_implicit_grad_solver.py tests/unit/test_repository_hygiene.py tests/unit/test_workflow.py`:
  passed after documenting the BiCGSTAB/E-adjoint failure contract.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_implicit_grad_solver.py tests/unit/test_workflow.py::test_workflow_solver_stats_surface_e_adjoint_failure_telemetry tests/unit/test_workflow.py::test_optimization_runner_run_writes_outputs_with_fake_model tests/unit/test_repository_hygiene.py::test_project_readme_documents_e_adjoint_diagnostics tests/unit/test_repository_hygiene.py::test_implicit_gradient_documents_bicgstab_failure_policy -q`:
  6 passed after adding the solver failure-stat regression, implicit-gradient
  docstring guard, and finite workflow run guard for nonfatal E-adjoint failure
  telemetry.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_repository_hygiene.py -q`:
  46 passed after adding the BiCGSTAB/E-adjoint documentation guard.
- `python -m pytest --collect-only -q`: 883 tests collected after adding the
  BiCGSTAB nonconvergence and documentation guards.
- `python -m py_compile gpurec/core/_helpers.py gpurec/core/backward.py gpurec/core/kernels/wave_backward.py gpurec/core/kernels/wave_backward_cuda.py gpurec/core/kernels/pibar_vjp_cuda.py tests/unit/test_core_helpers.py tests/unit/test_repository_hygiene.py`:
  passed after documenting the native CUDA prototype loader/fallback policy.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_core_helpers.py -k 'cuda_pibar_from_ud or env_mode' -q`:
  22 passed, 21 deselected after adding the `enabled` best-effort mode guard.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_repository_hygiene.py::test_project_readme_documents_cuda_prototype_fallback_policy tests/unit/test_repository_hygiene.py::test_cuda_prototype_source_documents_loader_and_fallback_policy -q`:
  2 passed after adding README and source-docstring guards for CUDA prototype
  fallback behavior.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_core_helpers.py tests/unit/test_core_backward.py -q`:
  45 passed after documenting selected-path strictness and preserving the
  optional self-loop exception boundary.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_repository_hygiene.py -q`:
  48 passed after adding the CUDA prototype policy documentation guards.
- `python -m pytest --collect-only -q`: 886 tests collected after adding the
  CUDA prototype policy guards.
- `python -m py_compile gpurec/api/uniform_chunked.py tests/unit/test_optimization_workflow.py tests/unit/test_repository_hygiene.py`:
  passed after exposing direct uniform-chunk E-adjoint telemetry.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_optimization_workflow.py::test_uniform_chunked_e_adjoint_stats_fields_are_public_stats_shape tests/unit/test_optimization_workflow.py::test_uniform_chunked_full_sum_estimate_scales_loss_and_grad tests/unit/test_repository_hygiene.py::test_uniform_chunked_loss_and_grad_documents_e_adjoint_stats -q`:
  3 passed after documenting and guarding the public `loss_and_grad()` stats
  keys.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_workflow.py::test_workflow_solver_stats_surface_e_adjoint_failure_telemetry tests/unit/test_repository_hygiene.py::test_project_readme_documents_e_adjoint_diagnostics -q`:
  2 passed after confirming workflow aggregate telemetry remains separate and
  intact.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_optimization_workflow.py -q`:
  30 passed after adding direct uniform-chunk telemetry coverage.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_repository_hygiene.py -q`:
  49 passed after adding the direct uniform-chunk telemetry documentation guard.
- `python -m pytest --collect-only -q`: 888 tests collected after adding the
  direct uniform-chunk telemetry guards.
- `python -m py_compile gpurec/core/kernels/wave_backward_cuda.py gpurec/core/kernels/pibar_vjp_cuda.py tests/unit/test_repository_hygiene.py`:
  passed after mirroring the Pibar dynamic shared-memory preflight in the
  self-loop CUDA prototype.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_repository_hygiene.py::test_project_readme_documents_cuda_prototype_fallback_policy tests/unit/test_repository_hygiene.py::test_cuda_prototype_source_documents_loader_and_fallback_policy tests/unit/test_repository_hygiene.py::test_cuda_prototype_launchers_preflight_dynamic_shared_memory -q`:
  3 passed after adding the source-level launch-contract guard.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_repository_hygiene.py -q`:
  50 passed after adding the dynamic shared-memory launch guard.
- `python -m pytest --collect-only -q`: 889 tests collected after adding the
  dynamic shared-memory launch guard.
- `python -m py_compile profiling/bench_uniform_forward_backward_pipeline.py tests/unit/test_repository_hygiene.py`:
  passed after adding the historical inline-path reference guard.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_repository_hygiene.py::test_missing_inline_path_references_are_explicitly_historical_or_optional -q`:
  1 passed after guarding absent tracked-looking inline path references in
  README/docs.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_repository_hygiene.py -q`:
  51 passed after adding the historical inline-path reference guard.
- `CUDA_VISIBLE_DEVICES='' python -m pytest --collect-only -q`: 890 tests
  collected after adding the historical inline-path reference guard.
- `python -m py_compile scripts/make_hogenom_branchscale_penalty_report.py tests/unit/test_repository_hygiene.py`:
  passed after documenting the legacy branchscale report layout.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_repository_hygiene.py::test_scripts_readme_lists_tracked_scripts_in_ownership_matrix tests/unit/test_repository_hygiene.py::test_branchscale_penalty_report_documents_legacy_layout_and_staleness -q`:
  2 passed after guarding the branchscale report source/help and scripts
  ownership matrix.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_repository_hygiene.py -q`:
  52 passed after adding the branchscale report documentation guard.
- `CUDA_VISIBLE_DEVICES='' python -m pytest --collect-only -q`: 891 tests
  collected after adding the branchscale report documentation guard.
- `python -m py_compile tests/unit/test_repository_hygiene.py tests/unit/test_release_metadata.py`:
  passed after adding CUDA config-map and release license-blocker guards.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_repository_hygiene.py::test_hogenom_scripts_are_marked_as_legacy_experiment_surface tests/unit/test_repository_hygiene.py::test_docs_map_distinguishes_cuda_smoke_from_checkout_local_config tests/unit/test_release_metadata.py::test_release_readiness_preserves_license_no_publish_blocker tests/unit/test_release_metadata.py::test_readme_scopes_example_config_to_source_artifacts -q`:
  4 passed after guarding CUDA-only example/config docs and release no-publish
  wording.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_repository_hygiene.py -q`:
  53 passed after adding the docs-map CUDA/config guard.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_release_metadata.py -q`:
  40 passed, 1 skipped after adding the release-readiness license blocker guard.
- `CUDA_VISIBLE_DEVICES='' python -m pytest --collect-only -q`: 893 tests
  collected after adding the CUDA/config and release-readiness guards.
- `python -m py_compile tests/unit/test_repository_hygiene.py`: passed after
  adding the Rust backtracking source-schema documentation guard.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_repository_hygiene.py::test_rust_backtracking_source_documents_payload_schema_and_validation -q`:
  1 passed after guarding the Rust payload schema comments and validation
  boundaries in `crates/gpurec-backtrack/src/lib.rs`.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_repository_hygiene.py -q`:
  54 passed after adding the Rust backtracking source-schema documentation
  guard.
- `CUDA_VISIBLE_DEVICES='' python -m pytest --collect-only -q`: 894 tests
  collected after adding the Rust backtracking source-schema documentation
  guard.
- `python -m py_compile tests/unit/test_repository_hygiene.py tests/unit/test_workflow.py`:
  passed after documenting the checkpoint config metadata surface and guarding
  that `load_checkpoint_config` remains out of public exports.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_repository_hygiene.py::test_rust_backtracking_source_documents_payload_schema_and_validation tests/unit/test_repository_hygiene.py::test_project_readme_documents_checkpoint_config_metadata_surface tests/unit/test_workflow.py::test_top_level_exports_workflow_surface -q`:
  3 passed after expanding the Rust schema guard and adding checkpoint config
  surface guards.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_repository_hygiene.py -q`:
  55 passed after adding the checkpoint config metadata README guard.
- `CUDA_VISIBLE_DEVICES='' python -m pytest --collect-only -q`: 895 tests
  collected after adding the checkpoint config metadata and export guards.
- `python -m py_compile gpurec/api/model.py tests/unit/test_repository_hygiene.py tests/unit/test_workflow.py`:
  passed after documenting the `materialize_batches()` and
  `full_loss_for_theta(theta)` public helper contracts.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_workflow.py -q -k 'materialize_batches or full_loss_for_theta'`:
  3 passed, 434 deselected after confirming batch materialization and
  explicit-theta full-loss streaming behavior.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_repository_hygiene.py::test_project_readme_and_model_docstrings_document_full_batch_helpers -q`:
  1 passed after adding README/docstring guards for those public helpers.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_repository_hygiene.py -q`:
  56 passed after adding the public helper documentation guard.
- `CUDA_VISIBLE_DEVICES='' python -m pytest --collect-only -q`: 896 tests
  collected after adding the public helper documentation guard.
- `python -m py_compile gpurec/core/forward.py gpurec/core/kernels/dts_fused.py gpurec/core/kernels/wave_backward.py tests/unit/test_repository_hygiene.py`:
  passed after documenting direct DTS shape precedence.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_repository_hygiene.py::test_dts_shape_precedence_is_documented_before_runtime_change -q`:
  1 passed after guarding README and source docstrings for the direct DTS
  `G == S` shape ambiguity.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_repository_hygiene.py -q`:
  58 passed after adding the direct DTS shape-precedence documentation guard.
- `CUDA_VISIBLE_DEVICES='' python -m pytest --collect-only -q`: 898 tests
  collected after adding the direct DTS shape-precedence documentation guard.
- `python -m py_compile gpurec/cli.py gpurec/workflow/config.py tests/unit/test_repository_hygiene.py`:
  passed after correcting the `log_every` documentation boundary.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_repository_hygiene.py::test_log_every_docs_distinguish_stdout_from_history -q`:
  1 passed after guarding that history JSONL is recorded every optimizer step
  and `--log-every` only throttles console progress prints.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_repository_hygiene.py -q`:
  59 passed after adding the `log_every` documentation guard.
- `CUDA_VISIBLE_DEVICES='' python -m pytest --collect-only -q`: 899 tests
  collected after adding the `log_every` documentation guard.
- `python -m py_compile gpurec/workflow/checkpoint.py tests/unit/test_repository_hygiene.py`:
  passed after documenting the checkpoint submodule support boundary.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_repository_hygiene.py::test_project_readme_documents_checkpoint_config_metadata_surface -q`:
  1 passed after guarding `gpurec.workflow.checkpoint.__all__`, README, and
  module docstring support-boundary wording.
- `python -m py_compile profiling/bench_uniform_forward_backward_pipeline.py tests/unit/test_repository_hygiene.py`:
  passed after refreshing the documented uniform pipeline benchmark imports.
- `python profiling/bench_uniform_forward_backward_pipeline.py --help`: passed
  after the benchmark switched to the current chunk dataclass names.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_repository_hygiene.py::test_documented_uniform_pipeline_benchmark_help_imports_current_api -q`:
  1 passed after adding the CPU-safe benchmark help/import smoke.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_repository_hygiene.py -q`:
  60 passed after adding checkpoint-boundary and benchmark help guards.
- `CUDA_VISIBLE_DEVICES='' python -m pytest --collect-only -q`: 900 tests
  collected after adding the checkpoint-boundary and benchmark help guards.
- `python -m py_compile scripts/optimize_hogenom_penalty316_kkt.py tests/unit/test_repository_hygiene.py`:
  passed after documenting the penalty-316 KKT script contract.
- `python scripts/optimize_hogenom_penalty316_kkt.py --help`: passed and shows
  the checkout-local branchscaled/KKT contract.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_repository_hygiene.py::test_penalty316_kkt_script_documents_checkout_local_contract -q`:
  1 passed after guarding source, help, and scripts ownership wording for the
  penalty-316 KKT analysis script.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_repository_hygiene.py -q`:
  61 passed after adding the penalty-316 KKT script contract guard.
- `CUDA_VISIBLE_DEVICES='' python -m pytest --collect-only -q`: 901 tests
  collected after adding the penalty-316 KKT script contract guard.
- `python -m py_compile gpurec/core/kernels/wave_backward.py tests/unit/test_repository_hygiene.py`:
  passed after documenting the active-mask bf16 helper boundary.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_repository_hygiene.py::test_active_mask_bfloat16_boundary_is_documented_as_private_helper -q`:
  1 passed after guarding the private helper docstring and the retained
  fp32/fp64 Pi backward dtype gate.
- `python -m py_compile gpurec/workflow/checkpoint.py tests/unit/test_repository_hygiene.py`:
  passed after documenting the current v1 checkpoint identity boundary.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_repository_hygiene.py::test_checkpoint_identity_boundary_is_documented_before_stricter_validation -q`:
  1 passed after guarding README and module-docstring wording for the current
  checkpoint identity comparison boundary.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_repository_hygiene.py -q`:
  63 passed after adding the checkpoint identity boundary documentation guard.
- `CUDA_VISIBLE_DEVICES='' python -m pytest --collect-only -q`: 903 tests
  collected after adding the checkpoint identity boundary documentation guard.
- `python -m py_compile tests/unit/test_repository_hygiene.py`: passed after
  adding the runtime-surface ownership-table guard.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_repository_hygiene.py::test_runtime_surface_plan_documents_scheduler_and_pybind_ownership -q`:
  1 passed after guarding the scheduler/C++ pybind ownership table.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_repository_hygiene.py -q`:
  64 passed after adding the runtime-surface ownership-table guard.
- `CUDA_VISIBLE_DEVICES='' python -m pytest --collect-only -q`: 904 tests
  collected after adding the runtime-surface ownership-table guard.
- `python -m py_compile scripts/optimize_hogenom_ccp_global_uniform.py scripts/optimize_hogenom_ccp_specieswise_uniform.py scripts/profile_hogenom_ccp_pass.py tests/unit/test_repository_hygiene.py`:
  passed after documenting ignored workspace and fixed-dataset launcher
  contracts.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_repository_hygiene.py::test_ignored_local_workspace_inventory_documents_notebooks_and_profiles tests/unit/test_repository_hygiene.py::test_fixed_dataset_hogenom_launchers_document_unique_contracts -q`:
  2 passed after guarding ignored local notebook/profiling inventory and the
  global-uniform/specieswise-uniform/profiler help contracts.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_repository_hygiene.py -q`:
  66 passed after adding the ignored-workspace and fixed-launcher guards.
- `CUDA_VISIBLE_DEVICES='' python -m pytest --collect-only -q`: 906 tests
  collected after adding the ignored-workspace and fixed-launcher guards.
- `python -m py_compile tests/unit/test_workflow.py`: passed after extending
  optimization-resume checkpoint identity mismatch coverage across all documented
  config identity fields.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_workflow.py::test_optimization_runner_resume_rejects_incompatible_checkpoint_identity -q`:
  7 passed after covering `species_tree`, `families_file`, `mode`, `start`, and
  `max_families` mismatches plus family/species name mismatches before theta
  restore.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_repository_hygiene.py -q`:
  66 passed after the checkpoint identity coverage documentation update.
- `CUDA_VISIBLE_DEVICES='' python -m pytest --collect-only -q`: 910 tests
  collected after expanding the parametrized checkpoint identity regression.
- `python -m py_compile tests/unit/test_repository_hygiene.py`: passed after
  documenting ignored local test-data/cache ownership.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_repository_hygiene.py::test_ignored_local_test_data_inventory_is_documented -q`:
  1 passed after guarding `.gitignore`, `tests/README.md`, and the runtime
  pruning plan for ignored test data, caches, and generated output roots.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_repository_hygiene.py -q`:
  67 passed after adding the ignored local test-data inventory guard.
- `CUDA_VISIBLE_DEVICES='' python -m pytest --collect-only -q`: 911 tests
  collected after adding the ignored local test-data inventory guard.
- `python -m py_compile tests/unit/test_repository_hygiene.py`: passed after
  correcting documentation-only drift reported by the docs/metadata explorer.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_repository_hygiene.py::test_repo_audit_headline_metrics_are_marked_as_initial_snapshot tests/unit/test_repository_hygiene.py::test_release_readiness_gpu_smoke_matches_small_species_limitation tests/unit/test_repository_hygiene.py::test_second_order_docs_reference_current_public_loss_apis -q`:
  3 passed after marking headline audit metrics as an initial snapshot, aligning
  release GPU-smoke wording with the `S > 256` backward limitation, and removing
  stale exact line references from the second-order planning note.
- `python -m py_compile tests/unit/test_repository_hygiene.py`: passed after
  recording core/API and workflow/backtracking refresh findings.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_repository_hygiene.py::test_runtime_surface_plan_records_refresh_findings_before_behavior_changes -q`:
  1 passed after guarding the new runtime-surface decision tables for direct API
  validation, parameter-shape helpers, chunked diagnostics, sampler preflight,
  LBFGS nonfinite handling, local script closure, CLI compatibility, Rust term
  variants, and release TOML parsing fallback.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_repository_hygiene.py -q`:
  70 passed after recording the docs/metadata, core/API, and
  workflow/backtracking refresh findings.
- `CUDA_VISIBLE_DEVICES='' python -m pytest --collect-only -q`: 914 tests
  collected after adding the refresh-finding hygiene guards.
- `git diff --check`: passed after the refresh-finding documentation and guards.
- `python -m py_compile gpurec/backtracking.py tests/unit/test_workflow.py`:
  passed after tightening Rust sampler help preflight markers.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_workflow.py::test_ensure_backtracking_available_validates_help tests/unit/test_workflow.py::test_ensure_backtracking_available_rejects_stale_help_missing_wrapper_flags tests/unit/test_workflow.py::test_ensure_backtracking_available_rejects_unrelated_executable tests/unit/test_workflow.py::test_ensure_backtracking_available_reports_help_failure -q`:
  4 passed after requiring sampler help for `--seed`, `--output-dir`, and
  `--max-events` and adding a negative stale-help regression.
- `cargo run --locked --quiet --manifest-path crates/gpurec-backtrack/Cargo.toml -- --help`:
  passed and printed the current Rust help with `--samples`, `--output-dir`,
  `--seed`, and `--max-events`.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_workflow.py -q -k 'ensure_backtracking_available or backtracking_command'`:
  9 passed, 433 deselected after tightening the sampler preflight.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_repository_hygiene.py -q`:
  70 passed after marking the Rust sampler help-marker finding as fixed.
- `CUDA_VISIBLE_DEVICES='' python -m pytest --collect-only -q`: 915 tests
  collected after adding the stale-help regression.
- `git diff --check`: passed after the sampler preflight change and audit-doc
  updates.
- `python -m py_compile gpurec/workflow/optimize.py tests/unit/test_workflow.py`:
  passed after adding the LBFGS post-step nonfinite guard.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_workflow.py::test_optimization_runner_lbfgs_rejects_nonfinite_post_step_evaluation -q`:
  1 passed after guarding the finite-closure/nonfinite-current-theta LBFGS path.
- `python -m py_compile gpurec/core/batching.py tests/unit/test_release_metadata.py tests/unit/test_repository_hygiene.py scripts/check_release_metadata.py`:
  passed after adding Python 3.10 TOML test-import compatibility and updating
  the stale `collate_gene_families()` docstring.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_release_metadata.py::test_minimal_pyproject_parser_extracts_release_metadata_subset tests/unit/test_release_metadata.py::test_minimal_pyproject_parser_supports_current_project_release_fields tests/unit/test_release_metadata.py::test_dev_extra_installs_tomli_for_python310_toml_tests tests/unit/test_repository_hygiene.py::test_runtime_surface_plan_records_refresh_findings_before_behavior_changes tests/unit/test_repository_hygiene.py::test_collate_gene_families_docstring_uses_current_layout_owner -q`:
  5 passed after guarding the TOML fallback, Python 3.10 `tomli` dependency,
  refreshed subagent findings, and current batching docstring.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_release_metadata.py -q`:
  43 passed, 1 skipped after the Python 3.10 TOML compatibility guard.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_repository_hygiene.py -q`:
  71 passed after recording the fresh subagent findings and source docstring
  guard.
- `CUDA_VISIBLE_DEVICES='' python -m pytest --collect-only -q`: 920 tests
  collected after adding the release metadata and repository hygiene guards.
- `git diff --check`: passed after the fresh subagent findings, TOML
  compatibility, and batching docstring updates.
- `python -m py_compile tests/integration/test_rust_backtracking_fixture.py tests/unit/test_repository_hygiene.py`:
  passed after adding the Rust CLI multi-sample output-dir fixture guard.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/integration/test_rust_backtracking_fixture.py -q`:
  3 passed after running the real Rust CLI for both single-output and
  `--samples 2 --output-dir` fixture paths.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_repository_hygiene.py::test_checked_fixture_contracts_are_documented -q`:
  1 passed after documenting the `sample_0.xml` / `sample_1.xml` output-dir
  contract in the fixture README.
- `python -m py_compile gpurec/core/batching.py tests/unit/test_global_wave_scheduler.py`:
  passed after adding family-clade metadata validation to `build_wave_layout()`.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_global_wave_scheduler.py::test_build_wave_layout_rejects_invalid_family_clade_metadata tests/unit/test_family_layout.py::test_build_family_wave_layout_matches_inputs_and_wave_order -q`:
  8 passed after rejecting mismatched, missing, negative, out-of-bounds,
  overlapping, and incomplete family metadata while preserving the valid
  family-layout path.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_global_wave_scheduler.py tests/unit/test_family_layout.py -q`:
  57 passed after the direct layout metadata guard.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/integration/test_rust_backtracking_fixture.py tests/unit/test_release_metadata.py tests/unit/test_repository_hygiene.py -q`:
  117 passed, 1 skipped after the Rust fixture, release metadata, and hygiene
  guard updates.
- `CUDA_VISIBLE_DEVICES='' python -m pytest --collect-only -q`: 928 tests
  collected after adding the multi-sample Rust fixture and family metadata
  validation parameter cases.
- `git diff --check`: passed after the multi-sample Rust fixture and direct
  layout metadata guard.
- `python -m py_compile gpurec/workflow/optimize.py tests/unit/test_workflow.py`:
  passed after adding the final-evaluation nonfinite guard.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_workflow.py::test_optimization_runner_marks_nonfinite_final_evaluation_failed -q`:
  1 passed after proving a nonfinite mandatory final evaluation now fails the
  run, marks the final row, and leaves the summary on the last finite objective.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_workflow.py -q -k 'optimization_runner and (mode or lbfgs or final or artifacts or resume or checkpoint)'`:
  23 passed, 421 deselected after the final-evaluation failure handling change.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_workflow.py -q`:
  444 passed after the final-evaluation nonfinite guard.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_repository_hygiene.py tests/unit/test_release_metadata.py -q`:
  114 passed, 1 skipped after the refreshed audit docs and hygiene guards.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_global_wave_scheduler.py tests/unit/test_family_layout.py tests/integration/test_rust_backtracking_fixture.py -q`:
  60 passed after the direct layout metadata and Rust multi-sample fixture
  guards.
- `CUDA_VISIBLE_DEVICES='' python -m pytest --collect-only -q`: 929 tests
  collected after adding the final-evaluation nonfinite regression.
- `git diff --check`: passed after the final-evaluation nonfinite guard and
  audit-doc updates.
- `python -m py_compile gpurec/workflow/optimize.py tests/unit/test_workflow.py`:
  passed after broadening optimizer-state resume discard handling.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_workflow.py::test_optimization_runner_reports_discarded_resume_optimizer_state -q`:
  1 passed after verifying `ValueError`, `RuntimeError`, and `TypeError` from
  `optimizer.load_state_dict` are all reported as discarded resume state.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_workflow.py -q`:
  444 passed after broadening optimizer-state resume discard handling.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_repository_hygiene.py -q`:
  71 passed after marking the resume optimizer-state finding as fixed.
- The local HOGENOM script model-lifetime finding is now fixed.  The profiling
  helper closes each chunk-local `GeneReconModel` through
  `_nll_per_family_with_cleanup()`, while the AleRax backtracking comparison
  helper closes each family-local model through
  `_gpurec_event_counts_with_cleanup()`.  Both helpers close on success and
  after evaluation/sampling exceptions.
- `python -m py_compile profiling/evaluate_hogenom_alerax_rates.py scripts/compare_backtracking_alerax_events.py tests/unit/test_legacy_scripts.py`:
  passed after the local model cleanup helpers and fake-model regressions.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_legacy_scripts.py -q`:
  25 passed at the cleanup-only point after adding success and exception
  cleanup coverage for both local script paths.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_repository_hygiene.py -q`:
  71 passed after marking the local model-construction finding as fixed in the
  pruning plan.
- The local HOGENOM script count-control finding is now fixed.  The profiling
  and AleRax backtracking comparison helpers expose `build_parser()` and share
  `gpurec/_argparse_types.py` validators for positive, non-negative, and
  positive-even command-line values.
- `python -m py_compile gpurec/_argparse_types.py profiling/evaluate_hogenom_alerax_rates.py scripts/compare_backtracking_alerax_events.py tests/unit/test_legacy_scripts.py`:
  passed after adding the shared argparse helpers and parser builders.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_legacy_scripts.py::test_local_script_parsers_reject_invalid_count_controls -q`:
  12 passed after proving invalid chunk size, family count, sample count,
  start index, seed, iteration, tolerance, and wave-size controls now fail at
  the parser boundary.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_legacy_scripts.py -q`:
  37 passed after combining the local model cleanup and parser validation
  regressions.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_repository_hygiene.py -q`:
  71 passed after marking the local count-control finding as fixed in the
  pruning plan.
- The dynamic CLI compatibility attribute finding is now fixed.  The CLI
  surface test now calls `_run_config_cli_override_fields()` directly, continues
  to compare parser destinations to `RunConfig` fields, and asserts that the old
  dynamic `_RUN_CONFIG_CLI_OVERRIDE_FIELDS` attribute is absent.  The
  module-level `gpurec/cli.py` `__getattr__` hook was removed.
- `python -m py_compile gpurec/cli.py tests/unit/test_cli_workflow.py tests/unit/test_repository_hygiene.py`:
  passed after removing the dynamic CLI compatibility attribute.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_cli_workflow.py::test_run_config_cli_surface_matches_dataclass_fields tests/unit/test_repository_hygiene.py::test_runtime_surface_plan_records_refresh_findings_before_behavior_changes -q`:
  2 passed after moving the CLI surface test to the helper and pinning absence
  of the old dynamic attribute.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_cli_workflow.py -q`:
  62 passed after removing the CLI module `__getattr__` hook.
- The Rust sampler term-variant finding is now fixed.  Direct
  `Sampler::apply_term()` unit tests cover `HiddenTransferLossDonor`, both
  hidden speciation directions, both split-transfer directions, normal split
  speciation, and swapped split speciation, with assertions on emitted event
  shape, species mapping, and queued `WorkItem` clade/species state.
- `cargo test --locked --manifest-path crates/gpurec-backtrack/Cargo.toml --lib hidden -- --nocapture`:
  5 passed after adding the hidden-term branch tests.
- `cargo test --locked --manifest-path crates/gpurec-backtrack/Cargo.toml --lib split -- --nocapture`:
  4 passed after adding the split-transfer and split-speciation branch tests.
- `cargo fmt --manifest-path crates/gpurec-backtrack/Cargo.toml --check`:
  passed after formatting the Rust test additions.
- `cargo test --locked --manifest-path crates/gpurec-backtrack/Cargo.toml`:
  19 library tests, 5 CLI tests, and 0 doctests passed after the direct
  sampler branch coverage.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_repository_hygiene.py -q`:
  72 passed after adding the Rust source guard for direct sampler branch tests.
- The direct API float-bool validation finding is now fixed.  Shared
  `finite_float()`, `nonnegative_float()`, and `positive_float()` reject Python
  bools and bool tensors before numeric coercion, so `tol_E`,
  `pi_max_diff_tol`, and `min_rate` cannot treat `True` as `1.0`.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_validation.py::test_float_validators_reject_bool_values tests/unit/test_validation.py::test_gene_recon_model_rejects_bool_float_controls_before_device_check tests/unit/test_validation.py::test_gene_recon_model_clamp_rejects_bool_min_rate_before_mutation -q`:
  9 passed after first reproducing the bool-coercion gap and adding bool-tensor
  coverage.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_validation.py -q`:
  38 passed after the direct API float-bool validation change.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_workflow.py::test_run_config_rejects_boolean_float_controls tests/unit/test_workflow.py::test_uniform_chunked_init_rejects_nonbool_controls_before_side_effects tests/unit/test_workflow.py::test_uniform_chunked_factories_reject_nonbool_controls_before_device_or_io -q`:
  24 passed for existing workflow and chunked direct-API bool-control guards
  after the shared validator change.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_repository_hygiene.py -q`:
  72 passed after marking the direct API float-bool validation finding fixed in
  the pruning plan.
- The private `_normalize_family_tree_paths()` compatibility alias is now
  deleted.  A source hygiene guard first failed only on `gpurec/core/model.py`,
  then passed after removing the one-line alias while retaining the public
  `normalize_family_tree_paths()` helper.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_repository_hygiene.py::test_private_family_tree_path_alias_is_not_in_source_surface -q`:
  1 passed after proving the private alias is absent from tracked runtime,
  script, and profiling Python sources.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_repository_hygiene.py -q`:
  73 passed after adding the private family-tree path alias source guard.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_alerax_family_input.py tests/unit/test_validation.py -q`:
  72 passed after deleting the private alias and keeping the public family-tree
  path normalization helper unchanged.
- The `normalize_family_chunk_size()` export-intent finding is now fixed.  The
  helper is retained as a supported `gpurec.core.batch_planning` helper because
  API, workflow, CLI, and tests already share it for the same family-batch
  control semantics; it now appears in `gpurec.core.batch_planning.__all__`.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_core_helpers.py::test_batch_planning_exports_supported_family_chunk_size_helper -q`:
  first failed on the missing `__all__` entry, then passed after adding the
  helper to the explicit batch-planning export surface.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_core_helpers.py::test_batch_planning_exports_supported_family_chunk_size_helper tests/unit/test_workflow.py::test_family_chunk_size_normalization_is_shared tests/unit/test_repository_hygiene.py::test_runtime_surface_plan_records_refresh_findings_before_behavior_changes -q`:
  3 passed after the export-intent update.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_core_helpers.py -q`:
  44 passed after adding the batch-planning wildcard export guard.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_repository_hygiene.py -q`:
  73 passed after marking the family chunk-size export-intent finding fixed in
  the pruning plan.
- The `UniformChunkedState` ownership finding is now fixed.  The class was
  renamed to `_UniformChunkedState` after documenting that the container is
  owned by `gpurec/api/uniform_chunked.py` autograd/evaluator internals and is
  not exported from `gpurec.api.uniform_chunked.__all__`.
- `python -m py_compile gpurec/api/uniform_chunked.py tests/unit/test_repository_hygiene.py`:
  passed after the internal state rename and hygiene guard addition.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_repository_hygiene.py::test_uniform_chunked_state_container_is_internal -q`:
  1 passed after adding the AST guard that rejects the public-looking class/name
  reference and keeps the private state class out of `__all__`.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_workflow.py::test_uniform_chunked_wildcard_import_exposes_public_surface_only -q`:
  1 passed after confirming the uniform chunked wildcard surface still exposes
  only `UniformChunkMetadata` and `UniformChunkedReconModel`.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_repository_hygiene.py -q`:
  74 passed after marking the uniform chunked state ownership finding fixed in
  the pruning plan.
- The explicit theta tensor shape validation finding is now fixed.  Shared
  `validate_theta_shape()` enforces exact public raw-theta shapes for the active
  parameter-sharing mode: global `[3]`, specieswise `[S, 3]`, and genewise
  `[G, 3]`.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_validation.py::test_gene_recon_model_rejects_invalid_theta_init_shape_before_device_check tests/unit/test_validation.py::test_full_loss_for_theta_rejects_invalid_explicit_theta_shape_before_streaming -q`:
  first failed with 12 current-gap failures, then passed after validating
  `theta_init` before CUDA checks and explicit `theta` before full-batch
  streaming.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_validation.py -q`:
  50 passed after adding the raw-theta shape validation regressions.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_workflow.py::test_full_loss_for_theta_uses_streaming_contract_for_explicit_theta -q`:
  1 passed after updating the existing streaming-contract fixture to use a
  valid global `[3]` theta tensor.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_repository_hygiene.py -q`:
  74 passed after marking the explicit theta tensor shape validation finding
  fixed in the pruning plan.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_workflow.py -q`:
  444 passed after validating explicit theta shapes on the public
  `full_loss_for_theta()` path.
- The parameter extraction shape-contract finding is now guarded.  CPU table
  tests cover `as_family_param()`, `as_family_species()`, and
  `extract_parameters_uniform()` for global, specieswise, genewise,
  `family_rows`, and `G == S` ambiguity semantics, and the
  `as_family_species()` docstring now describes the broadcast contract.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_extract_parameters.py -q`:
  first exposed the missing `as_family_species()` contract docstring, then
  8 passed after adding the docstring and correcting the independent base-2
  log-softmax test oracle.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_extract_parameters.py tests/unit/test_repository_hygiene.py::test_runtime_surface_plan_records_refresh_findings_before_behavior_changes -q`:
  9 passed after marking the parameter extraction shape-contract finding
  guarded in the pruning plan.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_repository_hygiene.py -q`:
  74 passed after updating the pruning-plan and audit-log source guards for the
  extraction helper coverage.
- The `UniformChunkedReconModel.nll_per_family()` diagnostic-contract finding is
  now guarded.  The README now class-qualifies the genewise-only
  `GeneReconModel.nll_per_family()` / `full_nll_per_family()` APIs and
  separately documents `UniformChunkedReconModel.nll_per_family(chunk_indices=...)`
  as a no-grad global/uniform diagnostic for selected shared-theta family NLLs.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_optimization_workflow.py::test_uniform_chunked_nll_per_family_uses_no_grad_chunked_diagnostic tests/unit/test_repository_hygiene.py::test_uniform_chunked_nll_per_family_documents_diagnostic_contract tests/unit/test_repository_hygiene.py::test_project_readme_documents_genewise_per_family_api_contract -q`:
  3 passed after adding the direct monkeypatched evaluator guard and README/API
  docstring source guards.
- `python -m py_compile gpurec/api/_validation.py gpurec/api/model.py gpurec/api/uniform_chunked.py gpurec/core/extract_parameters.py tests/unit/test_extract_parameters.py tests/unit/test_optimization_workflow.py tests/unit/test_repository_hygiene.py tests/unit/test_validation.py tests/unit/test_workflow.py`:
  passed after the uniform chunked per-family diagnostic doc/test update.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_optimization_workflow.py -q`:
  31 passed after adding the direct
  `UniformChunkedReconModel.nll_per_family()` unit guard.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_repository_hygiene.py -q`:
  75 passed after adding the class-qualified README and uniform chunked
  diagnostic documentation guards.
- The `implicit_grad_loglik_vjp_wave()` ownership finding is now guarded.  The
  function is documented as an internal bridge between `gpurec.api.model`,
  `gpurec.api.autograd`, and retained optimization internals; it remains out of
  `gpurec.optimization.__all__`, and a source guard rejects additional tracked
  runtime references outside the two API callers and definition module.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_repository_hygiene.py::test_implicit_gradient_documents_bicgstab_failure_policy -q`:
  1 passed after extending the existing implicit-gradient hygiene guard with the
  internal bridge ownership/export/call-site checks.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_repository_hygiene.py::test_runtime_surface_plan_records_refresh_findings_before_behavior_changes tests/unit/test_repository_hygiene.py::test_implicit_gradient_documents_bicgstab_failure_policy -q`:
  2 passed after marking the implicit-gradient bridge ownership finding guarded
  in the pruning plan.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_repository_hygiene.py -q`:
  75 passed after the implicit-gradient ownership source/doc/export guard.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_validation.py tests/unit/test_workflow.py tests/unit/test_extract_parameters.py tests/unit/test_optimization_workflow.py tests/unit/test_repository_hygiene.py -q`:
  608 passed for the current core/API audit slice.
- The test-only scheduler helper deletion finding is now fixed.
  `collate_wave()` and `split_phase_waves()` were removed from
  `gpurec.core.batching` after documenting that they had no tracked production
  callers and only helper-level tests.  Repository hygiene now guards that those
  helper names stay out of tracked runtime Python source.
- `python -m py_compile gpurec/core/batching.py tests/unit/test_global_wave_scheduler.py tests/unit/test_repository_hygiene.py`:
  passed after removing the test-only scheduler helpers.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_global_wave_scheduler.py tests/unit/test_repository_hygiene.py::test_test_only_scheduler_helpers_stay_out_of_runtime_source tests/unit/test_repository_hygiene.py::test_runtime_surface_plan_documents_scheduler_and_pybind_ownership -q`:
  47 passed after removing the helper-level tests and adding the runtime-source
  hygiene guard.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_scheduling.py tests/unit/test_global_wave_scheduler.py tests/unit/test_family_layout.py tests/unit/test_repository_hygiene.py -q`:
  138 passed after deleting `collate_wave()` and `split_phase_waves()` while
  retaining the `compute_clade_waves()` adapter and production family-layout
  scheduler coverage.
- The Python `compute_clade_waves()` adapter deletion finding is now fixed.
  `gpurec/core/scheduling.py` and `tests/unit/test_scheduling.py` were removed
  after documenting that the Python adapter had no tracked production callers
  and only helper-level tests.  The same-name C++ preprocessing helper remains
  production-internal, and repository hygiene now guards that the Python adapter
  does not return to tracked runtime source.
- `python -m py_compile gpurec/core/batching.py tests/unit/test_global_wave_scheduler.py tests/unit/test_repository_hygiene.py`:
  passed after deleting the Python `compute_clade_waves()` adapter module and
  its helper-level unit tests.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_global_wave_scheduler.py tests/unit/test_family_layout.py tests/unit/test_repository_hygiene.py::test_test_only_scheduler_helpers_stay_out_of_runtime_source tests/unit/test_repository_hygiene.py::test_runtime_surface_plan_documents_scheduler_and_pybind_ownership tests/unit/test_repository_hygiene.py::test_runtime_surface_plan_records_refresh_findings_before_behavior_changes -q`:
  58 passed after updating the scheduler ownership table and source guard for
  the Python clade-wave adapter deletion.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_global_wave_scheduler.py tests/unit/test_family_layout.py tests/unit/test_repository_hygiene.py -q`:
  131 passed after deleting the Python clade-wave adapter and its helper-level
  test module.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_validation.py tests/unit/test_workflow.py tests/unit/test_extract_parameters.py tests/unit/test_optimization_workflow.py tests/unit/test_global_wave_scheduler.py tests/unit/test_family_layout.py tests/unit/test_repository_hygiene.py -q`:
  664 passed for the current touched-unit baseline after the scheduler adapter
  deletions.
- The low-level package-doc finding is now fixed.  `gpurec/__init__.py` no
  longer advertises direct imports from `gpurec.core.model`,
  `gpurec.core.likelihood`, or `gpurec.core.forward`; the package docstring
  points users to the high-level `gpurec.api` and `gpurec.workflow` surfaces,
  and `docs/README.md` records that `gpurec.core` is an unstable internal
  namespace except for explicitly documented supported helpers.  Repository
  hygiene now guards both the package docstring and the developer-doc note.
- `python -m py_compile gpurec/__init__.py tests/unit/test_repository_hygiene.py`:
  passed after updating the package-doc guard.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_repository_hygiene.py::test_package_docs_do_not_advertise_core_as_public_surface -q`:
  1 passed after adding the package-doc/developer-doc stability guard.
- The likelihood compatibility-alias deprecation pass is now documented before
  changing behavior.  The retained low-level aliases
  `compute_log_likelihood()` and `compute_log_likelihood_root_rows()` return
  NLL, so the cleanup should add `DeprecationWarning`s, move ordinary tests to
  `compute_nll()` / `compute_nll_root_rows()`, and leave only a direct
  compatibility test exercising the old names.
- That deprecation pass is now complete.  Both compatibility aliases warn with
  `DeprecationWarning`, `tests/unit/test_specieswise_uniform.py` uses the
  current `compute_nll*` names, and repository hygiene guards that the legacy
  names remain limited to their implementation and the direct compatibility
  test.
- `python -m py_compile gpurec/core/likelihood.py tests/unit/test_origination_probs.py tests/unit/test_specieswise_uniform.py tests/unit/test_repository_hygiene.py`:
  passed after adding the deprecation warnings and alias-owner guard.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_origination_probs.py tests/unit/test_repository_hygiene.py::test_legacy_likelihood_aliases_warn_and_have_single_test_owner -q`:
  10 passed after the compatibility alias warning update.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_specieswise_uniform.py --collect-only -q`:
  5 tests collected after moving the CUDA-marked helper module to the NLL names.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_repository_hygiene.py -q`:
  78 passed after the package-doc and alias-deprecation guards.
- LIK-02 advanced past the deprecation pass: the misleading low-level
  likelihood aliases were removed from `gpurec.core.likelihood`, ordinary tests
  no longer import them, and repository hygiene now guards that tracked runtime,
  test, script, and profiling Python surfaces stay on `compute_nll*`.
- Fresh native/C++/kernel subagent audit findings were recorded from a
  read-only pass.  The historical production-auto native CUDA prototype routing
  finding is now closed by deleting the modules and guarding their absence.  The
  former direct pybind scheduler/stat and legacy `preprocess` surfaces are now
  closed by the Rust/PyO3 export-manifest guard.  The remaining high-risk
  native/kernel findings from that pass have since been reduced by dedicated
  guards: DTS direct-kernel one-dimensional parameter ambiguity is governed by
  the shared DTS layout contract, retained backward direct wrappers have
  pre-launch metadata validation tests, and kernel modules are kept free of
  runtime environment tuning knobs.
- Fresh workflow/CLI/scripts/profiling subagent audit findings were recorded
  from a read-only pass.  The largest remaining script surface is the legacy
  HOGENOM optimizer family; other unresolved surfaces are duplicated validation
  and normalization, weaker cleanup windows in legacy/profiling paths, an
  underspecified profiling ownership boundary, source-checkout-only configs
  that can look like installed workflows, and historical branchscale/KKT report
  scripts that need migration or archival.
- The DTS direct-kernel one-dimensional parameter ambiguity finding is now
  closed for retained production code by the shared `_dts_layout_contract.py`
  classifier and `tests/unit/test_dts_layout_contract.py`.  Direct callers
  still must use `[G, 1]` or `[G, S]` when `G == S`; the contract documents and
  tests the current forward/backward precedence before any behavior change.
- The retained backward direct-wrapper characterization slice is now guarded
  by CPU-safe tests for pre-launch metadata validation in the uniform backward
  wrapper, DTS Pibar staging wrapper, and DTS-staged Pibar VJP wrapper.  These
  tests pin the direct caller contract without requiring CUDA or reaching
  Triton launch.
- The broad env-driven kernel launch tuning finding is now closed for package
  runtime.  Tracked kernel modules carry no `GPUREC_*` literals or direct
  environment reads, package code reads only the supported environment owner
  manifest, and repository hygiene guards both contracts.
- The family path/name/map normalization part of the duplicated-validation
  finding is now fixed for dataset construction and the uniform chunked API:
  they share `gpurec.core.model.normalize_family_inputs()` for default family
  names, duplicate-name rejection, and leaf-species-map length checks.
- The integer-control part of the duplicated-validation finding is now reduced.
  `gpurec._validation` owns `integer_value()`, `positive_int()`,
  `nonnegative_int()`, and `positive_even_int()`; the API validation module
  re-exports those support helpers, and `RunConfig` string adapters delegate to
  them for workflow integer controls.  Optional integer controls now follow the
  same support boundary through `optional_positive_int()`,
  `optional_nonnegative_int()`, and `optional_positive_even_int()`, including
  workflow string adapters, core AleRax family-selection validation, and shared
  batch-planning controls such as `clade_budget` and `family_chunk_size`.  The
  resident uniform memory-policy estimators also delegate their integer
  dimensions and candidate controls to the shared helpers.  The Rust scheduler
  bridge now keeps only its string adapter locally and delegates non-string
  integer semantics to the same shared validation helper.  Checkpoint resume
  metadata now uses the shared nonnegative-integer validator beneath its
  checkpoint-specific error messages.  The resident-model `prefetch_batches`
  adapter now keeps only its `all`/disabled string aliases locally before
  delegating count semantics to the same nonnegative-integer helper.  The
  stochastic backtracking bridge now keeps only seed/event range checks locally
  and delegates integral-number coercion to the shared integer helper.
- The float-range part of the duplicated-validation finding is now reduced as
  well.  `gpurec._validation` owns `finite_float()`, `positive_float()`, and
  `nonnegative_float()`; the API validation module re-exports those helpers for
  direct API callers while workflow float adapters and checkpoint resume
  metadata already delegate to the same shared finite-float validator.  The
  Pi-adjoint fixed-point relaxation control now delegates through a core solver
  helper to the shared positive-float validator while preserving the retained
  backward path's legacy error wording.  The shared validation module keeps
  torch-specific bool-tensor detection lazy so checkpoint metadata imports stay
  lightweight.
- The profiling ownership-boundary documentation gap is now fixed.
  `profiling/README.md` documents the two tracked profiling entrypoints, their
  source-checkout/CUDA/local-data assumptions, output-contract expectations,
  ignored artifact policy, and bytecode-only `profiling/proposal2/` /
  `profiling/proposal8/` scratch directories.  The main README and docs map now
  point to that note, and repository hygiene guards the entrypoint list and
  artifact policy.
- Fresh public-API/docs subagent findings were recorded from a read-only pass.
  The remaining unresolved release blocker is the metadata/license decision.
  The Rust sampling binary distribution model is now explicit release contract:
  wheels require an external compatible `gpurec-backtrack` binary, while source
  archives include the locked crate and Cargo fallback.  The pass also flagged
  the `GeneDataset(..., leaf_species_maps=...)` documentation inconsistency
  created by classifying `gpurec.core` as internal, broad
  `core.batch_planning.__all__` exports, retained deprecated likelihood aliases,
  internal-looking API helper modules without clear support notes,
  duplicated evaluator/gradient logic,
  high scheduler complexity, public helper/property docstring gaps, stale
  ignored notebook artifacts, and repository hygiene tests that intentionally
  preserve some private or deprecated surfaces.
- The `GeneDataset(..., leaf_species_maps=...)` documentation inconsistency is
  now fixed as documentation only.  The README and docs map now describe it as
  a narrow low-level preprocessing/mapping exception while leaving the rest of
  `gpurec.core` unstable unless explicitly documented; `from_trees()` points
  users to that narrow exception for labels that cannot use prefix fallback or
  AleRax `mapping` entries.
- The whole `gpurec.core.batch_planning.__all__` export set is now documented
  after the public-API/docs refresh finding.  `FamilyBatchPlan`,
  `normalize_batch_packing`, `normalize_clade_budget`,
  `normalize_family_chunk_size`, and `plan_family_batches` are retained as a
  narrow shared low-level planning boundary for in-repo API, workflow, CLI,
  memory-policy, and white-box test callers, not as a broad `gpurec.core`
  stability promise.  The direct wildcard export guard now locks the exact set.
- `python -m py_compile gpurec/core/batch_planning.py tests/unit/test_core_helpers.py tests/unit/test_repository_hygiene.py`:
  passed after documenting the shared batch-planning export set.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_core_helpers.py::test_batch_planning_exports_supported_shared_planning_helpers tests/unit/test_repository_hygiene.py::test_runtime_surface_plan_records_refresh_findings_before_behavior_changes -q`:
  2 passed after tightening the batch-planning export guard and runtime-plan
  guard.
- `python scripts/check_release_metadata.py`: still fails on the expected
  release policy blockers: missing top-level `LICENSE`, missing
  `[project].license`, and missing license classifier.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_core_helpers.py tests/unit/test_origination_probs.py tests/unit/test_repository_hygiene.py -q`:
  132 passed for the touched CPU units after the package-doc, alias,
  profiling, `GeneDataset`, and batch-planning documentation updates.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_specieswise_uniform.py --collect-only -q`:
  5 tests collected after the CUDA-marked module moved off the deprecated
  likelihood aliases.
- `git diff --check`: passed after the current audit slice.
- The simplification opportunity index is now guarded as the direct deletion
  and consolidation inventory requested by the audit.  The docs map points to
  `docs/simplification-opportunity-index-2026-05-21.md`, the refactor plan
  directs readers to it for the concrete removable/mergeable paths, and
  repository hygiene checks that the index keeps source-file evidence, retained
  behavior, and deletion gates for evaluation, mode layout, likelihood alias,
  C++ diagnostic export, fixed-dataset script, and dead-internal test cleanup
  candidates.
- `python -m py_compile tests/unit/test_repository_hygiene.py`: passed after
  adding the simplification-index guard.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_repository_hygiene.py::test_simplification_opportunity_index_is_mapped_and_gate_oriented -q`:
  1 passed after guarding the simplification index.
- The internal `gpurec.api` helper-module documentation gap is now guarded.
  `gpurec/api/_family_layout.py` documents itself as internal support shared by
  `GeneReconModel` and `UniformChunkedReconModel`, with docstrings on its
  public-looking dataclasses and helper functions.  `gpurec/api/_validation.py`
  documents itself as shared internal validation support rather than standalone
  public API, and `validate_theta_shape()` now has a mode-shape docstring.
- `python -m py_compile gpurec/api/_family_layout.py gpurec/api/_validation.py tests/unit/test_repository_hygiene.py`:
  passed after adding internal helper-module documentation.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_repository_hygiene.py::test_internal_api_helper_modules_document_support_boundary tests/unit/test_family_layout.py tests/unit/test_validation.py -q`:
  61 passed after guarding the internal API helper module docstrings.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_repository_hygiene.py -q`:
  81 passed after adding the simplification-index and internal API helper
  documentation guards.
- `git diff --check`: passed after the simplification-index and internal helper
  documentation updates.
- The config ownership documentation gap is now fixed.  `configs/README.md`
  distinguishes source-checkout config files from installed package templates,
  classifies `hogenom_ccp_wandb.yaml` as a checkout-local Hydra/W&B experiment
  input consumed by `scripts/optimize_hogenom_ccp_hydra.py`, and reiterates
  that `examples/minimal-run-config.json` is a flat JSON CUDA parser fixture
  rather than a CPU fallback or end-to-end optimizer smoke.  The README and docs
  map now point to the config ownership note.
- `python -m py_compile tests/unit/test_repository_hygiene.py`: passed after
  adding the config ownership guard.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_repository_hygiene.py::test_docs_map_distinguishes_cuda_smoke_from_checkout_local_config -q`:
  1 passed after extending the docs-map/config ownership guard.
- The public property and optimizer-knob documentation gap is now guarded.
  Public `GeneReconModel` dataset/batch/species properties, public
  `UniformChunkedReconModel` count/iteration/chunk metadata properties, and
  `BatchedLBFGS` constructor knobs such as `max_eval`, tolerances, Armijo
  probe count, sufficient-decrease constant, and shrink factor now have source
  docstrings checked by repository hygiene.
- `python -m py_compile gpurec/api/model.py gpurec/api/uniform_chunked.py gpurec/optimization/batched_lbfgs.py tests/unit/test_repository_hygiene.py`:
  passed after documenting the public properties and LBFGS knobs.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_repository_hygiene.py::test_public_properties_and_batched_lbfgs_knobs_are_documented tests/unit/test_batched_lbfgs.py tests/unit/test_workflow.py -q -k 'materialize_batches or full_loss_for_theta or BatchedLBFGS or batched_lbfgs or public_properties'`:
  10 passed, 441 deselected after adding the public property and LBFGS knob
  documentation guard.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_repository_hygiene.py -q`:
  82 passed after adding the config ownership and public property/LBFGS
  documentation guards.
- `git diff --check`: passed after the config ownership and public
  property/LBFGS documentation updates.
- The direct C++ pybind ownership wording gap is now guarded.  The
  `preprocess` binding docstring now labels it a legacy compatibility export
  retained for historical low-level callers while deprecation/removal is
  evaluated.  The direct `compute_phased_waves` and wave-stat pybind docstrings
  now label themselves diagnostic exports and state that maintained profiling
  or diagnostic ownership is required.
- `python -m py_compile tests/unit/test_repository_hygiene.py`: passed after
  adding the direct C++ pybind docstring guard.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_repository_hygiene.py::test_runtime_surface_plan_documents_scheduler_and_pybind_ownership -q`:
  1 passed after guarding the legacy/diagnostic pybind wording.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_repository_hygiene.py -q`:
  82 passed after the direct C++ pybind docstring guard.
- `git diff --check`: passed after the direct C++ pybind documentation update.
- `python -m py_compile gpurec/__init__.py gpurec/api/model.py gpurec/core/likelihood.py tests/unit/test_origination_probs.py tests/unit/test_specieswise_uniform.py tests/unit/test_repository_hygiene.py`:
  passed after the profiling ownership and narrow `GeneDataset` documentation
  updates.
- `CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit/test_repository_hygiene.py::test_package_docs_do_not_advertise_core_as_public_surface tests/unit/test_repository_hygiene.py::test_project_readme_documents_leaf_species_mapping_contract tests/unit/test_repository_hygiene.py::test_profiling_readme_documents_entrypoints_and_artifact_policy tests/unit/test_repository_hygiene.py::test_ignored_local_workspace_inventory_documents_notebooks_and_profiles tests/unit/test_repository_hygiene.py::test_legacy_likelihood_aliases_warn_and_have_single_test_owner -q`:
  5 passed after adding the profiling README guard and documenting the
  `GeneDataset` mapping exception.
- The Pi-adjoint warm-start cache is now an explicit API-bridge runtime
  boundary instead of only a core-kernel argument.  It remains opt-in, records
  whether an initial guess was used, supports staging a solved `v_Pi` separately
  from the accepted cache, drops stale layout-shaped caches, participates in
  the existing runtime-cache clear path, and the Hessian-conditioned genewise
  workflow now commits staged adjoints only after the accepted current-theta
  gradient.  Production optimizer defaults are unchanged until warmstarted
  gradient budgets are validated end to end.
- Pi-adjoint fixed-point relaxation validation now uses the shared
  positive-float semantics through `gpurec.core._solver_validation` in both the
  retained `Pi_wave_backward()` entrypoint and its fused self-loop wrapper,
  while preserving the old `"fixed_point_relaxation must be a positive finite
  number"` error text for direct callers.
- `python -m py_compile gpurec/core/_solver_validation.py gpurec/core/backward.py gpurec/core/kernels/wave_backward.py tests/unit/test_core_backward.py tests/unit/test_repository_hygiene.py`:
  passed after adding the shared Pi-adjoint relaxation helper and source guard.
- `CUDA_VISIBLE_DEVICES='' python -m pytest -q tests/unit/test_core_backward.py::test_fixed_point_relaxation_value_accepts_numeric_scalars tests/unit/test_core_backward.py::test_fixed_point_relaxation_value_rejects_invalid_controls tests/unit/test_repository_hygiene.py::test_workflow_numeric_validation_uses_shared_helpers`:
  14 passed after adding the helper regression and hygiene guard.
- `git diff --check`: passed after the Pi-adjoint relaxation validation
  cleanup.
- `CUDA_VISIBLE_DEVICES='' python -m pytest -q -m "unit and not gpu"`:
  1449 passed, 1 skipped, 51 deselected after the Pi-adjoint relaxation
  validation cleanup.
- Workflow `RunConfig` mode strings are now normalized before `optimizer=auto`
  resolution, matching the public model API behavior.  Mixed-case or
  whitespace-padded JSON config modes now still select the intended production
  defaults: genewise `hessian-sgd`, specieswise `adagrad-restarts`, and global
  `adam`.  Checkpoint route audits also normalize mode strings before testing
  production-default optimizer settings.
- `python -m py_compile gpurec/workflow/config.py tests/unit/test_workflow.py tests/unit/test_repository_hygiene.py`:
  passed after adding the workflow mode-normalization guard.
- `CUDA_VISIBLE_DEVICES='' python -m pytest -q tests/unit/test_workflow.py::test_run_config_normalizes_mode_before_auto_optimizer_resolution tests/unit/test_workflow.py::test_run_config_from_dict_normalizes_mode_before_auto_optimizer_resolution tests/unit/test_workflow.py::test_run_config_from_dict_rejects_non_string_mode tests/unit/test_workflow.py::test_route_audit_normalizes_checkpoint_mode_strings tests/unit/test_repository_hygiene.py::test_run_config_reference_covers_current_config_surface`:
  7 passed after guarding workflow mode normalization and documentation.
- `git diff --check`: passed before the full CPU gate for the workflow
  mode-normalization slice.
- `CUDA_VISIBLE_DEVICES='' python -m pytest -q -m "unit and not gpu"`:
  1455 passed, 1 skipped, 51 deselected after the workflow
  mode-normalization slice.
- Summary and checkpoint route audits now normalize optimizer evidence with the
  same workflow rules as `RunConfig`: underscore aliases and `auto` resolve
  through the canonical mode before mode-default and production-default gates
  are evaluated.  `gpurec summary-info` prints the canonical route evidence for
  legacy summaries before applying `--require-mode-default-optimizer` or
  `--require-production-default-route`.
- `python -m py_compile gpurec/workflow/config.py gpurec/cli.py tests/unit/test_workflow.py tests/unit/test_cli_workflow.py tests/unit/test_repository_hygiene.py`:
  passed after adding the artifact optimizer-route normalization helpers.
- `CUDA_VISIBLE_DEVICES='' python -m pytest -q tests/unit/test_workflow.py::test_route_audit_normalizes_optimizer_alias_strings tests/unit/test_cli_workflow.py::test_cli_summary_info_normalizes_route_mode_and_optimizer_aliases tests/unit/test_repository_hygiene.py::test_output_artifact_reference_is_linked_and_documents_contract`:
  4 passed after guarding canonical route evidence in summaries and docs.
- `git diff --check`: passed before the full CPU gate for the artifact
  optimizer-route normalization slice.
- `CUDA_VISIBLE_DEVICES='' python -m pytest -q -m "unit and not gpu"`:
  1458 passed, 1 skipped, 51 deselected after the artifact optimizer-route
  normalization slice.
- CLI mode and optimizer flags now use the same normalization rules as
  `RunConfig`: mode names are stripped and case-normalized, and optimizer
  names are stripped, case-normalized, and allowed to use underscore aliases
  before argparse choices are checked.  This keeps `gpurec optimize`,
  `validate-config`, `run`, and `config-template` consistent with flat JSON
  configs for the production default routes.
- `python -m py_compile gpurec/workflow/config.py gpurec/cli.py tests/unit/test_workflow.py tests/unit/test_cli_workflow.py tests/unit/test_repository_hygiene.py`:
  passed after sharing the mode/optimizer CLI normalization helpers.
- `CUDA_VISIBLE_DEVICES='' python -m pytest -q tests/unit/test_workflow.py::test_run_config_normalizes_explicit_optimizer_alias_strings tests/unit/test_cli_workflow.py::test_cli_normalizes_mode_and_optimizer_alias_flags tests/unit/test_cli_workflow.py::test_cli_config_template_prints_specieswise_adagrad_restart_defaults tests/unit/test_repository_hygiene.py::test_run_config_reference_covers_current_config_surface`:
  5 passed after guarding CLI and JSON normalization parity.
- `CUDA_VISIBLE_DEVICES='' python -m pytest -q -m "unit and not gpu"`:
  1461 passed, 1 skipped, 51 deselected after the CLI mode/optimizer
  normalization slice.
- Public explicit-theta likelihood probes now validate device and dtype before
  resident-batch streaming, in addition to the existing mode-specific theta
  shape and finite-value checks.  `GeneReconModel(..., theta_init=...)` and
  `GeneReconModel.full_loss_for_theta(theta)` now both use the shared
  `validate_theta_shape()` device/dtype contract, so direct API callers get
  field-specific `ValueError`s before CUDA/static-state failures.
- `python -m py_compile gpurec/api/_validation.py gpurec/api/model.py tests/unit/test_validation.py tests/unit/test_workflow.py tests/unit/test_repository_hygiene.py`:
  passed after adding explicit-theta device/dtype validation.
- `CUDA_VISIBLE_DEVICES='' python -m pytest -q tests/unit/test_validation.py::test_gene_recon_model_rejects_theta_init_device_and_dtype_before_device_check tests/unit/test_validation.py::test_validate_theta_shape_checks_device_before_finite_values tests/unit/test_validation.py::test_full_loss_for_theta_rejects_explicit_theta_device_or_dtype_before_streaming tests/unit/test_workflow.py::test_full_loss_for_theta_uses_streaming_contract_for_explicit_theta tests/unit/test_repository_hygiene.py::test_package_docs_do_not_advertise_core_as_public_surface tests/unit/test_repository_hygiene.py::test_project_readme_and_model_docstrings_document_full_batch_helpers`:
  7 passed after guarding explicit-theta public API validation and docs.
- `CUDA_VISIBLE_DEVICES='' python -m pytest -q -m "unit and not gpu"`:
  1465 passed, 1 skipped, 51 deselected after the explicit-theta device/dtype
  validation slice.
- `--require-production-default-route` now verifies the shipped likelihood
  route contract, not only optimizer-specific settings: artifacts and configs
  must prove `objective=negative_log_likelihood_bits`,
  `gradient_route=implicit_first_order_adjoint`,
  `rate_parameterization=base2_log_dlt_rates`, and
  `production_default_basis=hogenom_and_test_trees_1000` before the stricter
  gate accepts them.
- `python -m py_compile gpurec/workflow/config.py gpurec/cli.py tests/unit/test_cli_workflow.py tests/unit/test_workflow.py tests/unit/test_repository_hygiene.py`:
  passed after extending production-route gate evidence.
- `CUDA_VISIBLE_DEVICES='' python -m pytest -q tests/unit/test_workflow.py::test_production_route_audit_requires_likelihood_gradient_contract_fields tests/unit/test_workflow.py::test_route_audit_infers_production_default_settings_from_route_dict tests/unit/test_cli_workflow.py::test_cli_validate_config_require_production_default_route_rejects_custom_settings tests/unit/test_cli_workflow.py::test_cli_checkpoint_info_require_production_default_route_recomputes_stale_audit tests/unit/test_cli_workflow.py::test_cli_checkpoint_info_require_production_default_route_requires_settings_evidence tests/unit/test_cli_workflow.py::test_cli_summary_info_require_production_default_route_rejects_custom_settings tests/unit/test_cli_workflow.py::test_cli_summary_info_require_production_default_route_recomputes_stale_audit tests/unit/test_cli_workflow.py::test_cli_summary_info_require_production_default_route_rejects_stale_gradient_route tests/unit/test_cli_workflow.py::test_cli_summary_info_require_production_default_route_requires_settings_evidence tests/unit/test_repository_hygiene.py::test_run_config_reference_covers_current_config_surface tests/unit/test_repository_hygiene.py::test_production_optimization_guide_is_linked_and_documents_routes tests/unit/test_repository_hygiene.py::test_output_artifact_reference_is_linked_and_documents_contract tests/unit/test_repository_hygiene.py::test_troubleshooting_guide_documents_operator_failure_triage`:
  13 passed after adding likelihood/gradient route-contract evidence to the
  production default route gate.
- `CUDA_VISIBLE_DEVICES='' python -m pytest -q -m "unit and not gpu"`:
  1467 passed, 1 skipped, 51 deselected after the production-route
  likelihood/gradient contract gate.
- Production route metadata now exposes the full strict-route verdict in
  artifacts and status lines, not only the optimizer-specific settings verdict:
  `uses_production_default_route` and
  `production_default_route_mismatches` are derived from the shipped
  likelihood route contract, mode default optimizer, and optimizer-specific
  settings. Older checkpoints remain resume-compatible because these new
  derived audit fields are exempt from route metadata identity comparison.
- `python -m py_compile gpurec/workflow/config.py gpurec/cli.py gpurec/workflow/optimize.py gpurec/workflow/checkpoint.py tests/unit/test_cli_workflow.py tests/unit/test_workflow.py tests/unit/test_repository_hygiene.py`:
  passed after exposing production-route verdict metadata.
- `CUDA_VISIBLE_DEVICES='' python -m pytest -q tests/unit/test_workflow.py::test_effective_route_metadata_reports_production_likelihood_contract tests/unit/test_workflow.py::test_effective_route_metadata_marks_nondefault_optimizer tests/unit/test_workflow.py::test_effective_route_metadata_reports_hessian_sgd_normal_solver_overrides tests/unit/test_workflow.py::test_run_config_auto_optimizer_uses_adagrad_restarts_for_specieswise_mode tests/unit/test_workflow.py::test_run_config_specieswise_adagrad_restarts_step_cap_honors_shorter_steps tests/unit/test_cli_workflow.py::test_cli_validate_config_reports_selected_family_references tests/unit/test_cli_workflow.py::test_cli_checkpoint_info_reports_route_status_and_last_row tests/unit/test_cli_workflow.py::test_cli_checkpoint_info_require_production_default_route_recomputes_stale_audit tests/unit/test_cli_workflow.py::test_cli_checkpoint_info_require_production_default_route_requires_settings_evidence tests/unit/test_cli_workflow.py::test_cli_summary_info_reports_status_route_and_final_check tests/unit/test_cli_workflow.py::test_cli_summary_info_normalizes_route_mode_and_optimizer_aliases tests/unit/test_cli_workflow.py::test_cli_summary_info_require_production_default_route_rejects_custom_settings tests/unit/test_cli_workflow.py::test_cli_summary_info_require_production_default_route_recomputes_stale_audit tests/unit/test_cli_workflow.py::test_cli_summary_info_require_production_default_route_rejects_stale_gradient_route tests/unit/test_cli_workflow.py::test_cli_summary_info_require_production_default_route_requires_settings_evidence tests/unit/test_repository_hygiene.py::test_production_optimization_guide_is_linked_and_documents_routes tests/unit/test_repository_hygiene.py::test_output_artifact_reference_is_linked_and_documents_contract tests/unit/test_repository_hygiene.py::test_run_config_reference_covers_current_config_surface`:
  18 passed after adding focused full-route verdict coverage to config,
  checkpoint, summary, and docs paths.
- `CUDA_VISIBLE_DEVICES='' python -m pytest -q tests/unit/test_workflow.py::test_optimization_result_is_derived_from_summary_contract tests/unit/test_workflow.py::test_optimization_runner_adagrad_restarts_accepts_split_solver_budgets tests/unit/test_workflow.py::test_optimization_runner_run_writes_outputs_with_fake_model tests/unit/test_cli_workflow.py::test_cli_validate_config_reports_hessian_sgd_normal_solver_overrides tests/unit/test_cli_workflow.py::test_cli_validate_config_reports_specieswise_restart_route tests/unit/test_cli_workflow.py::test_cli_summary_info_reports_adagrad_restart_route_fields`:
  6 passed after confirming OptimizationResult and CLI route lines carry the
  new verdict fields.
- `git diff --check`: passed before the full CPU gate for the production-route
  verdict metadata slice.
- `CUDA_VISIBLE_DEVICES='' python -m pytest -q -m "unit and not gpu"`:
  1467 passed, 1 skipped, 51 deselected after the production-route verdict
  metadata slice.
- Operator-facing docs now explicitly name the full production-route verdict in
  the `gpurec optimize`/`gpurec run` status-line contract and document that
  `gpurec checkpoint-info --require-production-default-route` requires the
  checkpoint route to match the shipped likelihood/gradient contract and
  optimizer-specific route, not only the mode-default optimizer.
- `python -m py_compile tests/unit/test_repository_hygiene.py`: passed after
  tightening the route-verdict documentation guards.
- `CUDA_VISIBLE_DEVICES='' python -m pytest -q tests/unit/test_repository_hygiene.py::test_project_readme_documents_gpurec_run_end_to_end_workflow tests/unit/test_repository_hygiene.py::test_output_artifact_reference_is_linked_and_documents_contract`:
  2 passed after guarding the README status-line contract and checkpoint route
  gate docs.
- `git diff --check`: passed before the full CPU gate for the route-verdict docs
  guard slice.
- `CUDA_VISIBLE_DEVICES='' python -m pytest -q -m "unit and not gpu"`:
  1467 passed, 1 skipped, 51 deselected after the route-verdict docs guard
  slice.
- Standalone `gpurec sample --require-production-default-route` now has direct
  CLI coverage proving the command accepts current default-route checkpoints
  and rejects stale likelihood-route checkpoints before invoking the sampling
  workflow.
- `python -m py_compile tests/unit/test_cli_workflow.py`: passed after adding
  standalone sampling production-route gate coverage.
- `CUDA_VISIBLE_DEVICES='' python -m pytest -q tests/unit/test_cli_workflow.py::test_cli_sample_require_production_default_route_accepts_default_checkpoint tests/unit/test_cli_workflow.py::test_cli_sample_require_production_default_route_rejects_stale_route`:
  2 passed after guarding standalone sampling production-route gates.
- `git diff --check`: passed before the full CPU gate for the standalone
  sampling production-route gate slice.
- `CUDA_VISIBLE_DEVICES='' python -m pytest -q -m "unit and not gpu"`:
  1469 passed, 1 skipped, 51 deselected after the standalone sampling
  production-route gate slice.
- `gpurec run --require-production-default-route` now has direct CLI coverage
  proving the combined optimize-and-sample command accepts the default
  genewise route and rejects non-default Hessian-SGD route settings before
  checking backtracking availability or invoking optimization.
- `python -m py_compile tests/unit/test_cli_workflow.py`: passed after adding
  combined run production-route gate coverage.
- `CUDA_VISIBLE_DEVICES='' python -m pytest -q tests/unit/test_cli_workflow.py::test_cli_run_require_production_default_route_accepts_default_config tests/unit/test_cli_workflow.py::test_cli_run_require_production_default_route_rejects_custom_settings_before_run`:
  2 passed after guarding combined run production-route gates.
- `git diff --check`: passed before the full CPU gate for the combined run
  production-route gate slice.
- `CUDA_VISIBLE_DEVICES='' python -m pytest -q -m "unit and not gpu"`:
  1471 passed, 1 skipped, 51 deselected after the combined run
  production-route gate slice.
- Direct `gpurec optimize --require-production-default-route` now has CLI
  coverage proving default genewise optimization proceeds and non-default
  Hessian-SGD route settings are rejected before the optimizer workflow is
  invoked.
- `python -m py_compile tests/unit/test_cli_workflow.py`: passed after adding
  direct optimize production-route gate coverage.
- `CUDA_VISIBLE_DEVICES='' python -m pytest -q tests/unit/test_cli_workflow.py::test_cli_optimize_require_production_default_route_accepts_default_config tests/unit/test_cli_workflow.py::test_cli_optimize_require_production_default_route_rejects_custom_settings_before_run`:
  2 passed after guarding direct optimize production-route gates.
- `git diff --check`: passed before the full CPU gate for the direct optimize
  production-route gate slice.
- `CUDA_VISIBLE_DEVICES='' python -m pytest -q -m "unit and not gpu"`:
  1473 passed, 1 skipped, 51 deselected after the direct optimize
  production-route gate slice.
- `gpurec checkpoint-info --require-production-default-route` and
  `gpurec summary-info --require-production-default-route` now have direct
  success-path CLI coverage proving current production-route artifacts pass
  the CPU-safe inspection gates, complementing the stale/incomplete artifact
  rejection tests.
- `python -m py_compile tests/unit/test_cli_workflow.py`: passed after adding
  artifact inspection production-route success coverage.
- `CUDA_VISIBLE_DEVICES='' python -m pytest -q tests/unit/test_cli_workflow.py::test_cli_checkpoint_info_require_production_default_route_accepts_current_route tests/unit/test_cli_workflow.py::test_cli_summary_info_require_production_default_route_accepts_current_route`:
  2 passed after guarding artifact inspection production-route success gates.
- `git diff --check`: passed before the full CPU gate for the artifact
  inspection production-route success slice.
- `CUDA_VISIBLE_DEVICES='' python -m pytest -q -m "unit and not gpu"`:
  1475 passed, 1 skipped, 51 deselected after the artifact inspection
  production-route success slice.
- `gpurec validate-config --require-production-default-route` now has direct
  success-path CLI coverage for both shipped production default modes:
  genewise `hessian-sgd` and specieswise `adagrad-restarts`.
- `python -m py_compile tests/unit/test_cli_workflow.py`: passed after adding
  strict validate-config production-route success coverage.
- `CUDA_VISIBLE_DEVICES='' python -m pytest -q tests/unit/test_cli_workflow.py::test_cli_validate_config_require_production_default_route_accepts_genewise_default tests/unit/test_cli_workflow.py::test_cli_validate_config_require_production_default_route_accepts_specieswise_default`:
  2 passed after guarding the strict validate-config production-route success
  gates.
- `git diff --check`: passed before the full CPU gate for the strict
  validate-config production-route success slice.
- `CUDA_VISIBLE_DEVICES='' python -m pytest -q -m "unit and not gpu"`:
  1477 passed, 1 skipped, 51 deselected after the strict validate-config
  production-route success slice.
- `docs/input-preparation.md` now tells production users to run
  `gpurec validate-config --config run.json --require-production-default-route`
  when launch automation should reject changed HOGENOM/`test_trees_1000`
  optimizer settings or stale likelihood/gradient route metadata before
  spending CUDA time.
- `python -m py_compile tests/unit/test_repository_hygiene.py`: passed after
  adding the strict preflight wording guard.
- `CUDA_VISIBLE_DEVICES='' python -m pytest -q tests/unit/test_repository_hygiene.py::test_input_preparation_guide_documents_alerax_data_contract`:
  1 passed after guarding the input-preparation strict production-route
  preflight wording.
- `git diff --check`: passed before the full CPU gate for the input-preparation
  strict preflight documentation slice.
- `CUDA_VISIBLE_DEVICES='' python -m pytest -q -m "unit and not gpu"`:
  1477 passed, 1 skipped, 51 deselected after the input-preparation strict
  preflight documentation slice.
- `gpurec optimize --require-production-default-route` and
  `gpurec run --require-production-default-route` now have direct specieswise
  success-path CLI coverage proving execution commands accept the shipped
  `adagrad-restarts` production route, including the schedule-derived
  `optimizer_step_cap=125`, before workflow and sampling handoff.
- `python -m py_compile tests/unit/test_cli_workflow.py`: passed after adding
  specieswise execution production-route gate coverage.
- `CUDA_VISIBLE_DEVICES='' python -m pytest -q tests/unit/test_cli_workflow.py::test_cli_optimize_require_production_default_route_accepts_specieswise_default tests/unit/test_cli_workflow.py::test_cli_run_require_production_default_route_accepts_specieswise_default`:
  2 passed after guarding specieswise strict execution gates.
- `git diff --check`: passed before the full CPU gate for the specieswise
  execution production-route success slice.
- `CUDA_VISIBLE_DEVICES='' python -m pytest -q -m "unit and not gpu"`:
  1479 passed, 1 skipped, 51 deselected after the specieswise execution
  production-route success slice.
- `gpurec.workflow.config.effective_final_check_iters(config)` now centralizes
  the final high-fidelity likelihood/gradient validation budget.  Route metadata,
  production-default setting audits, and `OptimizationRunner` now share this
  helper, so specieswise `adagrad-restarts` fixed128 validation cannot drift
  from the `final_check_iters` value reported in artifacts.
- `python -m py_compile gpurec/workflow/config.py gpurec/workflow/optimize.py tests/unit/test_workflow.py`:
  passed after centralizing the effective final-check budget.
- `CUDA_VISIBLE_DEVICES='' python -m pytest -q tests/unit/test_workflow.py::test_effective_route_metadata_reports_production_likelihood_contract tests/unit/test_workflow.py::test_run_config_auto_optimizer_uses_adagrad_restarts_for_specieswise_mode tests/unit/test_workflow.py::test_effective_final_check_iters_uses_optimizer_specific_budget tests/unit/test_workflow.py::test_optimization_runner_adagrad_restarts_accepts_split_solver_budgets`:
  4 passed after guarding that route metadata and the runner use the same final
  likelihood/gradient validation budget source.
- `git diff --check`: passed before the full CPU gate for the effective
  final-check budget helper slice.
- `CUDA_VISIBLE_DEVICES='' python -m pytest -q -m "unit and not gpu"`:
  1480 passed, 1 skipped, 51 deselected after centralizing and guarding the
  effective final-check budget.
- Final likelihood/gradient validation summaries now carry `final_check_iters_e`,
  derived from the existing `optimizer/final_check_iters_E` history metric.
  This keeps `summary.json`, the Python `OptimizationResult`, and CLI status
  lines explicit about both the Pi/Neumann final-check budget and the paired
  E-solver budget, including adaptive `null` E budgets for genewise routes.
- `gpurec checkpoint-info` now also exposes `last_final_check_iters` and
  `last_final_check_iters_e` when those budgets exist in the checkpoint last
  row, matching the final likelihood/gradient validation budget visibility in
  `summary.json` and the optimization status line.
- `gpurec.workflow.config.effective_final_check_iters_e(config)` now
  centralizes the paired E-solver budget used for final likelihood/gradient
  validation. Route metadata and `validate-config` report the same
  `final_check_iters_e` value that the optimization runner uses at runtime.
- Production-default optimizer-setting audits now require explicit
  `final_check_iters_e` evidence. Strict artifact gates therefore reject stale or
  incomplete route metadata that omits the paired E-solver budget even when the
  optimizer name and Pi/Neumann final-check budget still match.
- `gpurec sample` now loads a checkpoint once when both
  `--require-mode-default-optimizer` and `--require-production-default-route`
  are enabled. The shared payload keeps standalone sampling gates from doing
  duplicate large-checkpoint IO before the same production route audit.
- The shared `--require-production-default-route` CLI help now names
  `final_check_iters_e` evidence explicitly, matching the stricter route audit
  shown in summaries, checkpoints, and preflight output.
- Strict artifact gate failures now say they expected the shipped
  likelihood/gradient and optimizer route, not only the optimizer route, so the
  remediation text matches route-contract fields such as `gradient_route`.
- Config preflight failures now use matching remediation text: operators should
  use `optimizer=auto` and omit route overrides so the shipped
  likelihood/gradient and optimizer defaults apply.
- Mode-default artifact gate failures now say they expected the mode default
  optimizer route, avoiding production-route wording on the narrower
  `--require-mode-default-optimizer` check.
- Current operator docs now use mode-default wording for the narrower
  `--require-mode-default-optimizer` gate and reserve production-route wording
  for the full likelihood/gradient plus optimizer route enforced by
  `--require-production-default-route`.
- Release-readiness smoke documentation now uses the same wording: installed
  help checks expose mode-default optimizer gates separately from the stricter
  production-route gates.
- The README and output artifact contract no longer use the ambiguous
  "shipped production route" shorthand; current docs now name either the
  shipped HOGENOM/`test_trees_1000` optimizer route or the full shipped
  likelihood/gradient and optimizer route.
- The CLI gate help and remediation strings now share route-specific constants.
  `--require-mode-default-optimizer` help says mode default optimizer for the
  selected mode, while the full production-route gate keeps the shipped
  likelihood/gradient and optimizer wording.
- The shared `--require-production-default-route` help now spells out the full
  contract: objective, likelihood/gradient route, rate parameterization,
  `final_check_iters_e` evidence, optimizer-specific settings, and the shipped
  HOGENOM/`test_trees_1000` likelihood/gradient and optimizer route.  The
  installed-wheel smoke now greps for those route-contract terms in
  `gpurec optimize --help`.
- `effective_route_metadata()` now emits the production route contract fields
  directly from `_PRODUCTION_DEFAULT_ROUTE_CONTRACT`, so artifact metadata and
  `--require-production-default-route` comparisons cannot drift between
  separate copies of the shipped likelihood/gradient route literals.
- `production_default_route_contract()` and
  `production_default_route_contract_fields()` now expose that contract inside
  the workflow config module. CLI route-gate fallback evidence and CLI test
  checkpoint fixtures reuse those helpers instead of carrying another copy of
  the shipped likelihood/gradient field set.
- `production_default_optimizer_config_overrides()` now exposes the editable
  RunConfig fields for the shipped optimizer profiles. `gpurec config-template`
  and the source example checks reuse that helper, keeping installed starter
  templates tied to the same genewise `hessian-sgd` and specieswise
  `adagrad-restarts` defaults audited by the route metadata.
- Generated `gpurec config-template --mode genewise` and `--mode specieswise`
  JSON now have direct unit coverage that round-trips through `RunConfig` and
  reports `uses_production_default_route=true`, guarding the installed starter
  configs before any CUDA likelihood model is built.
- Generated templates now also pass the public
  `validate-config --require-mode-default-optimizer
  --require-production-default-route` path when pointed at existing AleRax
  inputs, so the installed starter workflow is guarded end to end through CLI
  preflight without constructing the CUDA likelihood model.
- The package workflow's installed-wheel smoke now generates genewise and
  specieswise templates from the installed `gpurec config-template` command,
  points them at the checked tiny AleRax fixture paths, and validates both with
  `gpurec validate-config --require-mode-default-optimizer
  --require-production-default-route`. Release-readiness documentation and
  release metadata tests track the same operator path.
- The package workflow's source-archive example smoke now runs both checked
  example configs through
  `validate-config --require-mode-default-optimizer
  --require-production-default-route --check-preprocess`, so unpacked source
  artifacts must prove the shipped genewise and specieswise route gates while
  still exercising the retained CPU preprocessing path.

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
5. Add ownership tables before deleting unowned pybind scheduler diagnostics,
   workflow submodule helpers, profiling benchmarks, or fixed-dataset HOGENOM
   scripts.
6. Only then consider behavior changes for backward small-`S`, bf16 dtype
   implementation, DTS runtime shape unification, CUDA Pibar fallback policy,
   RecPhyloXML assumptions, and sampling aggregate behavior.
