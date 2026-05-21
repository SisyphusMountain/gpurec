# Runtime Surface Pruning Plan, 2026-05-21

This plan covers simplification outside the core likelihood formula: public
APIs, scheduler surfaces, C++ extension exports, environment variables,
scripts, profiling entry points, and tests.  It should be implemented only
after the gradient/likelihood refactor has characterization coverage.

## Keep As Product Surface

Keep these user-facing capabilities:

- `GeneReconModel` with modes `global`, `specieswise`, and `genewise`.
- `UniformChunkedReconModel` as either a first-class class or a compatibility
  facade for large global/uniform datasets.
- Workflow `RunConfig`, `OptimizationRunner`, CLI `optimize`, CLI `sample`,
  and CLI `run`.
- Backtracking/export helpers needed by sampling workflows.
- Batch/chunk metadata sufficient for users to inspect what was loaded and
  optimized.

Everything else should be classified as internal, diagnostic, benchmark-only,
or deprecated.

## Public API Pruning

### Low-Level Core Imports

Addressed after the public-API documentation follow-up audit.
`gpurec/__init__.py` no longer advertises lower-level imports such as
`GeneDataset`, `E_fixed_point`, `compute_nll`, and `Pi_wave_forward`.  The
top-level exported API stays high-level, and the package docstring now points
users to `gpurec.api` / `gpurec.workflow` while classifying `gpurec.core` as an
internal implementation namespace.  `docs/README.md` carries the matching
low-level API stability note for developer-facing documentation.

Plan:

- Keep `gpurec.core` internal except for explicitly supported helpers.
- Keep the "low-level API is unstable" section in developer docs.
- Keep tests importing internals as white-box tests, not as evidence of public
  support.

### Compatibility Aliases

Addressed after the public-API documentation follow-up audit.
`compute_log_likelihood()` and `compute_log_likelihood_root_rows()` return NLL.
They now emit `DeprecationWarning`s that point callers to `compute_nll()` and
`compute_nll_root_rows()`.  Ordinary tracked tests use the NLL names; the old
names are exercised only by the explicit compatibility test.

Plan:

- Keep internal/profiling usage on `compute_nll()` or the root-row helper.
- Warn on alias use for one release if external compatibility matters.
- Delete aliases after public docs and tests stop naming them.

### Uniform Chunked API

`UniformChunkedReconModel` duplicates construction, validation, chunk planning,
evaluation, loss/gradient scaling, metadata, and dtype policy already present
in or adjacent to `GeneReconModel`.

Plan:

- Move chunk selection and `loss_and_grad(chunk_indices=...)` into the shared
  evaluator.
- Keep the public class as a thin adapter while users migrate.
- Remove duplicated validation helpers from `uniform_chunked.py` once they are
  shared through `gpurec/api/_validation.py` or the evaluator.

### Core/API Refresh Findings

Current unresolved findings from the follow-up core/API audit:

| Surface | Current owner / risk | Guard before behavior change |
| --- | --- | --- |
| `finite_float()`, `positive_float()`, and `nonnegative_float()` in `gpurec/api/_validation.py` | Addressed after the core/API follow-up audit. Shared direct-API float validators now reject Python bools and bool tensors before numeric coercion, so controls such as `tol_E`, `pi_max_diff_tol`, and `min_rate` fail before CUDA checks or theta mutation. | Keep direct validation tests for bool values in the shared helpers, `GeneReconModel` constructor float controls, and `GeneReconModel.clamp_theta_()`. |
| `as_family_param()`, `as_family_species()`, and `extract_parameters_uniform()` in `gpurec/core/extract_parameters.py` | Addressed after the core/API follow-up audit. CPU table tests now cover global, specieswise, and genewise `extract_parameters_uniform()` output shapes/values, plus `as_family_param()` / `as_family_species()` `family_rows` precedence when `G == S`. The `as_family_species()` docstring now documents the broadcast contract and the bare length-`G` ambiguity. | Keep the direct extraction-helper table tests and the docstring guard before refactoring parameter-shape policy. |
| `_normalize_family_tree_paths()` in `gpurec/core/model.py` | Addressed after the core/API follow-up audit. The one-line private compatibility alias was deleted after a source hygiene guard proved no tracked `gpurec/`, `scripts/`, or `profiling/` callers remain; callers use public `normalize_family_tree_paths()`. | Keep the source hygiene guard proving the private alias is absent from tracked runtime/script/profiling Python sources. |
| `gpurec.core.batch_planning.__all__`: `FamilyBatchPlan`, `normalize_batch_packing`, `normalize_clade_budget`, `normalize_family_chunk_size`, and `plan_family_batches` | Addressed after the public-API/docs follow-up audit. The whole exported set is retained as a narrow shared low-level planning boundary because API, workflow, CLI, memory policy, and white-box tests share these helpers for the same family-batch semantics. The module docstring marks this as a support boundary, not a promise that the rest of `gpurec.core` is stable. | Keep the batch-planning wildcard export guard for the exact exported set before changing planning ownership. |
| `UniformChunkedState` in `gpurec/api/uniform_chunked.py` | Addressed after the core/API follow-up audit. The state container was renamed to `_UniformChunkedState` after documenting that it is owned by chunked autograd/evaluator internals and has no direct tracked runtime callers outside `uniform_chunked.py`. | Keep the source/export guard proving `UniformChunkedState` is absent as a class/name reference and that `_UniformChunkedState` stays out of `gpurec.api.uniform_chunked.__all__`. |
| `UniformChunkedReconModel.nll_per_family()` | Addressed after the core/API follow-up audit. README and API docstrings now distinguish `GeneReconModel.nll_per_family()` / `full_nll_per_family()` genewise-only independent losses from `UniformChunkedReconModel.nll_per_family(chunk_indices=...)`, a no-grad global/uniform diagnostic returning selected shared-theta family NLLs after chunk filtering. | Keep the direct CPU unit guard that monkeypatches the chunked evaluator and asserts `need_grad=False`, `per_family=True`, exact `chunk_indices`, and disabled grad mode. |
| `implicit_grad_loglik_vjp_wave()` in `gpurec/optimization/implicit_grad.py` | Addressed after the core/API follow-up audit. The function is documented as an internal bridge between `gpurec.api.model`, `gpurec.api.autograd`, and retained optimization internals, not a supported low-level public API. It remains out of `gpurec.optimization.__all__`, and a source guard limits tracked runtime references to the API/autograd bridge callers plus its definition module. | Keep the doc/export/call-site hygiene guard; do not add external callers without promoting and testing it as public API. |
| Direct `build_wave_layout()` family-index inputs in `gpurec/core/batching.py` | Addressed after the core/API follow-up audit. The helper now requires `family_clade_counts` and `family_clade_offsets` to be provided together, have matching lengths, contain nonnegative integer ranges, stay within `C`, avoid overlaps, and cover every clade before `family_idx` is materialized. | Keep CPU unit guards for mismatched, overlapping, out-of-bounds, and incomplete family metadata while direct callers can pass family metadata. |
| Explicit theta tensors in `gpurec/api/model.py` and `full_loss_for_theta()` | Addressed after the core/API follow-up audit. Shared `validate_theta_shape()` now enforces exact raw tensor shapes for the active parameter-sharing mode: global `[3]`, specieswise `[S, 3]`, or genewise `[G, 3]`. Short tensors, extra event columns, wrong ranks, and wrong row counts now fail before CUDA checks, theta cloning, or streaming parameter extraction. | Keep direct API tests for invalid `theta_init` shapes and invalid explicit `theta` shapes, plus the existing valid streaming-contract guard for `full_loss_for_theta()`. |
| `collate_gene_families()` docstring in `gpurec/core/batching.py` | Addressed after the core/API follow-up audit. The docstring now describes preprocessed gene-family CCP payloads and the current `build_wave_layout()` owner instead of removed `preprocess_gene_with_species` / `likelihood_2.py` surfaces. | Keep the source hygiene guard that rejects those removed surface names from the helper docstring while it stays in `gpurec.core`. |

## Scheduler Surface

Current scheduler surfaces:

- production construction uses `schedule_family_waves()` and
  `build_family_wave_layout()`;
- `schedule_global_phased_waves()` tries several internal algorithms;
- the Python `compute_clade_waves()` adapter was deleted from runtime source
  after confirming it was helper-level tests only;
- `collate_wave()` and `split_phase_waves()` were deleted from runtime source
  after confirming they were helper-level tests only;
- C++ extension exports multiple wave-stat diagnostic functions.

Ownership table from the current read-only audit:

| Surface | Current owner / callers | Tests / docs | Deletion risk |
| --- | --- | --- | --- |
| `preprocess_multiple_families` pybind | Production-owned. `GeneDataset` calls it for normal preprocessing, family cache misses, and species-only empty-family cache fill in `gpurec/core/model.py`. | Fake/cache tests in `tests/unit/test_alerax_family_input.py`, real parser coverage through `GeneDataset`, and integration construction in `tests/integration/test_gene_recon_model.py`. | High. Keep. Non-empty family preprocessing needs `include_details=True`; the empty-family species-only cache path currently uses the default `include_details=False`. |
| Legacy `preprocess` pybind | No in-repo production caller found; exported from `gpurec/core/cpp/preprocess.cpp`. | Existing pruning docs flag it as legacy/open surface, and the pybind docstring now calls it a legacy compatibility export retained for historical low-level callers while deprecation/removal is evaluated. | Medium external/API risk, low in-repo runtime risk. Document as legacy/deprecated before removal. |
| `compute_phased_waves` pybind | No direct production caller found, but the underlying implementation is production-used to populate `phased_waves`/`phased_phases` during preprocessing. | Source-level hygiene guards it with the other max-wave exports, and the pybind docstring now calls the direct binding a diagnostic export rather than supported workflow API. | Do not remove the implementation. Deprecate the direct export only after diagnostic ownership is documented. |
| Wave-stat pybinds: `compute_wave_stats`, `compute_packet_wave_stats`, `compute_phased_wave_stats`, `compute_phased_cross_family_wave_stats`, `compute_cross_family_wave_stats` | No production caller found. | Hygiene checks positive `max_wave_size`; audit docs describe them as broad diagnostic ABI; pybind docstrings now require maintained profiling or diagnostic ownership. | Low runtime risk, medium diagnostic/API risk. Keep only with a maintained profiling or diagnostic command. |
| `bench_parse` | Not currently exported. | Removal is guarded in repository hygiene and audit docs. | Already retired; keep the guard. |
| `compute_clade_waves` Python helper | Addressed after the scheduler follow-up audit. The Python adapter and its helper-level unit module were deleted after confirming no tracked production caller imports it and no high-level public export exposes it. The C++ implementation with the same name remains production-internal to preprocessing. | Keep the source guard proving `gpurec/core/scheduling.py` and the Python adapter name do not return to tracked runtime Python. | Low in-repo runtime risk; direct low-level Python imports should use `schedule_global_phased_waves()` or `build_wave_layout()`, while C++ preprocessing still owns phased-wave generation. |
| `collate_wave`, `split_phase_waves` | Addressed after the scheduler follow-up audit. These helper-level scheduler functions were deleted from `gpurec.core.batching` after confirming no tracked production caller imports them and their only tracked users were direct helper tests. | Keep the source hygiene guard proving these helper names are absent from tracked runtime Python sources. | Low in-repo runtime risk; direct low-level external imports should use `schedule_global_phased_waves()` or `build_wave_layout()`. |
| Runtime Python scheduler/layout path | Production-owned through `gpurec/api/_family_layout.py`, `schedule_global_phased_waves()`, and `build_wave_layout()`. | Covered by global scheduler and family-layout tests. | High. Can be hidden behind private wrappers, but not deleted without replacement and benchmarks. |
| `family_schedule_summary` | Production-owned for depth-first batch packing in `GeneReconModel` and `UniformChunkedReconModel`. | Indirectly covered through planning/layout tests. | High while depth-first packing remains supported. |

Plan:

1. Keep the ownership table current when scheduler or pybind exports move
   between product runtime, benchmark diagnostic, test-only helper, and delete
   buckets.
2. Benchmark the current multi-candidate scheduler against one candidate policy.
3. Keep the chosen runtime scheduler private to the layout builder.
4. Move test-only scheduler helpers into `tests/` fixtures or delete them.
5. Remove diagnostic C++ pybind exports unless they have a maintained profiling
   command.

Remaining candidate deletions after classification:

- direct pybind wave-stat exports in `gpurec/core/cpp/preprocess.cpp`.

## C++ Preprocess Extension Surface

Current runtime calls:

- `GeneDataset` uses `preprocess_multiple_families(..., include_details=True,
  include_species_matrices=False)`.
- The species-only empty-family cache path uses
  `preprocess_multiple_families(..., include_details=False)` to materialize
  species topology data without family details.
- Cache loading expects detailed CCP helpers and leaf mapping tensors.

Open surfaces:

- legacy pybind `preprocess()`;
- `preprocess_multiple_families(..., include_details=False)`;
- wave-stat diagnostic pybind exports.

Pybind export manifest:

| Export | Classification | Replacement / deletion gate |
| --- | --- | --- |
| `preprocess_multiple_families` | Production-owned preprocessing entry point. | Keep while `GeneDataset` uses C++ preprocessing. Non-empty families require `include_details=True`; the species-only empty-family cache fill still owns `include_details=False`. |
| `preprocess` | Legacy compatibility export. | Delete only after deprecation/replacement evidence confirms no maintained low-level caller depends on the single-family direct pybind. Replacement is `preprocess_multiple_families(..., include_details=True)` for detailed family payloads. |
| `compute_phased_waves` | Direct scheduler diagnostic export. | Do not delete the underlying implementation while preprocessing emits `phased_waves` / `phased_phases`; hide or delete only the pybind after diagnostic ownership is replaced by preprocessing output or a maintained profiling command. |
| `compute_wave_stats` | Direct wave-stat diagnostic export. | Keep only with a maintained profiling or diagnostic command. |
| `compute_packet_wave_stats` | Direct wave-stat diagnostic export. | Keep only with a maintained profiling or diagnostic command. |
| `compute_phased_wave_stats` | Direct wave-stat diagnostic export. | Keep only with a maintained profiling or diagnostic command. |
| `compute_phased_cross_family_wave_stats` | Direct wave-stat diagnostic export. | Keep only with a maintained profiling or diagnostic command. |
| `compute_cross_family_wave_stats` | Direct wave-stat diagnostic export. | Keep only with a maintained profiling or diagnostic command. |

Plan:

- Search installed/user docs for each export before deletion.
- If no public owner exists, remove legacy `preprocess()` and
  `include_details=False`.
- Keep one path from Newick inputs to detailed family/species helpers.
- Keep C++ validation guards around the retained path.

Verification:

- C++ extension imports cleanly.
- AleRax family input tests pass.
- Preprocess cache validation tests pass.
- Integration construction from `from_trees()` and `from_alerax_families()`
  still works.

## Environment Variable Surface

Current package runtime reads the following environment variables.  This table
is an ownership manifest, not a deletion list; pruning should move non-user
surfaces behind constructor/config objects or benchmark/profiling entry points
before any flag is removed.

### Environment Owner Manifest

| Variable(s) | Current ownership | Runtime owner / notes |
| --- | --- | --- |
| `GPUREC_BACKTRACK_BIN` | User-facing | Binary/distribution contract for `gpurec sample`, `gpurec run`, and `gpurec backtrack-check`. |
| `GPUREC_ALERAX_COMPAT` | User-facing compatibility | Compatibility guard read by API validation; supported differentiable optimization accepts only unset or `0`. |
| `GPUREC_MEMORY_POLICY_FRACTION`, `GPUREC_MEMORY_POLICY_RESERVE_GIB` | User-facing | Memory-budget margins for uniform chunk planning. |
| `GPUREC_FUSE_FINAL_PIBAR`, `GPUREC_SPECIALIZE_NONLEAF_LEAF_TERM` | Internal production fast path | Forward/backward retained-kernel selectors that should become fixed behavior after benchmark gates. |
| `GPUREC_BACKWARD_NO_CPU_PRUNING`, `GPUREC_DTS_SKIP_INACTIVE_PIBAR_ZERO` | Internal production/diagnostic | Backward pruning and inactive-output zero-fill controls retained for behavior comparison. |
| `GPUREC_WAVE_STEP_BLOCK_S`, `GPUREC_WAVE_STEP_NUM_WARPS` | Benchmark/internal tuning | Triton forward wave-step launch tuning. |
| `GPUREC_DTS_BLOCK_S`, `GPUREC_DTS_NUM_WARPS`, `GPUREC_DTS_GRAD_MT_TILE_SPLITS` | Benchmark/internal tuning | Triton cross-DTS backward launch tuning. |
| `GPUREC_DTS_PARENT_BLOCK_S`, `GPUREC_DTS_PARENT_NUM_WARPS`, `GPUREC_DTS_PARENT_TILE_SPLITS` | Benchmark/internal tuning | Triton parent-reduced DTS launch tuning. |
| `GPUREC_PIBAR_UD_BLOCK_S`, `GPUREC_PIBAR_UD_NUM_WARPS` | Benchmark/internal tuning | Triton Pibar-from-`u_d` launch tuning. |
| `GPUREC_SELF_LOOP_2D_BLOCK_W`, `GPUREC_SELF_LOOP_2D_BLOCK_NODES`, `GPUREC_SELF_LOOP_2D_NUM_WARPS`, `GPUREC_SELF_LOOP_2D_JT_NUM_WARPS`, `GPUREC_SELF_LOOP_2D_SKIP_INACTIVE_SCRATCH_ZERO` | Benchmark/internal tuning | Triton 2D backward self-loop launch tuning and scratch-zero behavior. |
| `GPUREC_CUDA_SELF_LOOP_NOSPLIT`, `GPUREC_CUDA_SELF_LOOP_SPLIT`, `GPUREC_CUDA_SELF_LOOP_NOSPLIT_CORRECTION` | Prototype/internal | Native CUDA self-loop prototype selectors and correction mode. |
| `GPUREC_CUDA_SELF_LOOP_BLOCK`, `GPUREC_CUDA_SELF_LOOP_CHILD_EDGE_WEIGHT` | Prototype/internal tuning | Native CUDA self-loop launch tuning. |
| `GPUREC_CUDA_PIBAR_FROM_UD`, `GPUREC_CUDA_PIBAR_FROM_UD_STRICT` | Prototype/internal | Native CUDA Pibar-from-`u_d` prototype selector and strict-failure mode. |
| `GPUREC_CUDA_PIBAR_FROM_UD_BLOCK`, `GPUREC_CUDA_PIBAR_FROM_UD_PAD_SHARED` | Prototype/internal tuning | Native CUDA Pibar-from-`u_d` launch tuning. |

User-facing environment flags are limited to `GPUREC_BACKTRACK_BIN`,
`GPUREC_ALERAX_COMPAT`, `GPUREC_MEMORY_POLICY_FRACTION`, and
`GPUREC_MEMORY_POLICY_RESERVE_GIB`.  All other package-read `GPUREC_*` flags
are internal production, benchmark/internal tuning, or prototype/internal
diagnostics and should not be promoted in README wording without updating this
manifest.

Plan:

- Keep binary/distribution and memory-policy env vars.
- Convert production toggles into fixed behavior after benchmark gates.
- Move tuning variables to benchmark CLI flags or a `KernelOptions` object
  printed in benchmark output.
- Move native CUDA prototype controls out of production model execution unless
  a required-mode CUDA smoke test owns them.
- Read env once during model/static construction, not in every wave/kernel
  wrapper.

Expected user-facing result:

- Smaller README env table.
- Reproducible model behavior from constructor/config state.
- Kernel experiments still possible from profiling scripts.

## Workflow And CLI Surface

Retain the workflow modes, but remove duplicate normalization:

- `RunConfig`;
- CLI overrides;
- `GeneReconModel.from_alerax_families()`;
- evaluator/batch planning.

Plan:

- Let shared validation helpers own dtype, device, fixed iteration, adaptive
  iteration, rate bounds, batch packing, and mode normalization.
- Keep CLI accepted values narrow.
- Move optimizer-specific behavior out of model internals.
- Make `RunConfig` map directly to model/evaluator request objects.

Possible simplifications:

- Remove duplicate int/float/bool normalizers from `workflow/config.py` once
  `gpurec/api/_validation.py` covers the same contracts.
- Keep `dtype_from_name()` workflow-specific only if CLI wording needs it.
- Keep optimizer modes only if behavior is tested by fake-model guards.

## Scripts And Profiling Surface

Current scripts include maintained workflow helpers, profiling utilities,
legacy HOGENOM launchers, report generators, and export tools.  The docs now
have an ownership matrix, but deletion has not happened.

Ignored/local workspace inventory:

| Surface | Purpose | Inputs | Outputs | Current reproducibility | Decision |
| --- | --- | --- | --- | --- | --- |
| `notebooks/evaluate_gpurec_at_alerax_params.ipynb` | Evaluate gpurec likelihoods at AleRax global-rate parameters. | Ignored `tests/data/hogenom_bench`, AleRax `output_global/model_rates.csv`, AleRax per-family likelihood text, CUDA. | `tests/data/hogenom_bench/gpurec/nll_at_alerax_params_*.csv`. | Checkout-local only; depends on ignored data and notebook state. | Archive/delete or migrate into a tested rate-evaluation script. |
| `notebooks/hogenom_adam_bfgs_schedule.ipynb` | Historical scheduled Adam/BFGS optimizer experiment. | Ignored `tests/data/hogenom_bench`, CUDA, historical `gpurec.optimization.optimize_scheduled` import. | `tests/data/hogenom_bench/gpurec/scheduled_adam_bfgs_notebook/` CSV outputs. | Not currently reproducible from tracked source because the optimizer helper is not part of the retained API. | Archive/delete or rewrite against supported workflow optimizers before keeping. |
| `notebooks/optimize_hogenom_ccp_specieswise_origination.ipynb` | One-off specieswise optimization with custom origination distribution. | Local HOGENOM data, CUDA, fixed notebook constants. | `output_gpurec_specieswise_origination_opt` CSV, JSON, and PNG artifacts. | Checkout-local and stale; the notebook contains captured runtime error output. | Migrate unique origination behavior into `gpurec.workflow`/CLI or archive/delete. |
| `notebooks/pi_iteration_bound_diagnostic.ipynb` | Pi fixed-point iteration bound diagnostic. | Ignored `tests/data/hogenom_bench`, CUDA/float64, local `pi_iteration_bound_diagnostic_impl` source. | `tests/data/hogenom_bench/diagnostics/pi_iteration_bound_diagnostic/` CSV and plot files. | Not reproducible from tracked source because the helper source is absent. | Restore helper source and tests before keeping, otherwise archive/delete. |
| `profiling/ancestor_batching/` | Historical ancestor batching timing, Nsight, and NCU artifact trees. | Local profiling harnesses, HOGENOM benchmark data, Nsight tools. | `artifacts/*` timing JSONL, CSV summaries, `.nsys-rep`, `.ncu-rep`, and SQLite files. | Historical/non-reproducible; some generated commands reference missing local harnesses. | Keep summarized findings in docs, then externalize or delete bulky artifact trees. |
| `profiling/bf16_backward_nsys/` and `profiling/bf16_handoff_prod/` | bf16 backward/handoff experiment reports and logs. | Local CUDA runs and HOGENOM/test-tree fixtures. | Nsight/NCU reports, SQLite files, CSV summaries, and `.log` files. | Checkout-local experiment artifacts; bf16 is now documented as direct-API-only and not a release-smoke dtype. | Keep only summarized conclusions; archive/delete raw artifacts. |
| `profiling/hogenom_ccp/` | Local HOGENOM CCP performance sweeps. | Local HOGENOM data, CUDA/Nsight, profiling scripts and environment toggles. | JSONL sweeps, `.nsys-rep`, `.ncu-rep`, SQLite, and CSV summaries. | Checkout-local and too broad for release verification. | Migrate one maintained benchmark path; archive/delete ad hoc raw sweeps after summaries are preserved. |
| `profiling/specieswise_worker3/` | Local specieswise worker profiling scratch space. | Local CUDA/HOGENOM profiling runs. | `artifacts/` and `artifacts_smoke/` reports. | No tracked owner beyond ignored workspace state. | Document any retained conclusion, then archive/delete. |
| `profiling/proposal2/` and `profiling/proposal8/` | Local prototype residue. | Historical local Python prototype runs. | Python bytecode cache only in the current checkout. | Not source, fixtures, or retained benchmark results. | Delete locally; restore real source plus tests/docs before treating either as a maintained prototype. |

Ignored test-data and cache inventory:

| Surface | Purpose | Inputs | Outputs | Current reproducibility | Decision |
| --- | --- | --- | --- | --- | --- |
| `tests/data/test_trees_20/`, `tests/data/test_trees_100/`, `tests/data/test_trees_1000/`, `tests/data/test_trees_10000/` | Generated tree-scale fixtures for optional large-family and CUDA/profiling checks. | Local generated Newick/family files; no tracked generator contract in this repo. | Local `families.txt`, `sp.nwk`, `g_*.nwk`, and generated output subtrees. | Optional and checkout-local; tests that need them must skip or use explicit paths. | Keep out of required CPU gates; add a small tracked fixture or documented generator before relying on any of them. |
| `tests/data/test_trees_dtl01/` | Local DTL experiment fixture with prior output. | Local species/gene trees and generated output. | `output/` likelihood artifacts. | Scratch/local fixture, not a distributed contract. | Migrate the useful DTL expectation into a tracked fixture before CI, otherwise treat as deletable scratch. |
| `tests/data/HOGENOM/`, `tests/data/hogenom_bench/`, `tests/data/davin/` | External biological datasets used by HOGENOM notebooks, scripts, profiling, and local validation. | Local HOGENOM/AleRax/Davin files outside package distribution. | Local rate tables, benchmark outputs, and profiler inputs. | Checkout-local only; package and release checks must not require these roots. | Archive/delete local copies or migrate unique behavior into tracked fixtures/workflows before promotion. |
| `tests/data.tar.gz` | Local archive/transfer bundle for generated data. | Previous local dataset snapshot. | Compressed generated fixture bundle. | Not a source of truth and intentionally ignored. | Replace with a documented source/generator before any required workflow depends on it. |
| `.preprocess_cache/` and `tests/data/**/output/` | Runtime-generated cache and local test outputs. | Prior preprocessing or local runs. | Torch cache files, per-family likelihoods, XML, CSV, and other generated artifacts. | Regenerable byproducts. | Delete/regenerate as needed; never promote as expected fixtures. |

Plan:

- Keep:
  - release metadata checker;
  - one maintained full-pipeline benchmark;
  - current AleRax/HOGENOM validation utilities with explicit local-data
    labels;
  - necessary export/backtracking comparison helpers.
- Migrate or delete:
  - fixed-dataset optimizer launchers that duplicate the CLI;
  - stale report generators tied to old run-directory names;
  - one-off profiling scripts superseded by the maintained benchmark.
- Update tests to assert only maintained scripts are in the product/benchmark
  matrix.

### Workflow/Backtracking Refresh Findings

Current findings from the follow-up workflow/backtracking audit:

| Surface | Current owner / risk | Guard before behavior change |
| --- | --- | --- |
| Rust sampler help preflight in `gpurec/backtracking.py` | Addressed after the refresh audit. `_BACKTRACK_HELP_MARKERS` now requires the wrapper-supported `--samples`, `--seed`, `--output-dir`, and `--max-events` flags instead of accepting stale short help. | Keep the negative stale-help regression so old Rust binaries fail preflight before sampling. |
| LBFGS post-step evaluation in `gpurec/workflow/optimize.py` | Addressed after the refresh audit. The LBFGS branch evaluates the current theta after `optimizer.step(closure)` and now repeats the finite loss/gradient guard used by Adam/Adagrad. | Keep the fake LBFGS regression that returns a finite closure followed by a nonfinite current-theta evaluation and expects failed `nonfinite_objective_or_gradient` status with no LBFGS row recorded. |
| Final optimization evaluation in `gpurec/workflow/optimize.py` | Addressed after the workflow/scripts follow-up audit. The mandatory `final_eval` now repeats the finite loss/gradient guard; if it fails, the run status becomes `failed/nonfinite_objective_or_gradient`, the final row carries explicit `optimizer/final_eval_status` and `optimizer/final_eval_reason` fields, and nonfinite final metrics are not copied into the final row. | Keep the fake-model regression where a finite optimizer step is followed by a nonfinite final evaluation, and verify failed status, failed final-row markers, finite previous objective in the summary, and a failed latest checkpoint. |
| Local model construction in `profiling/evaluate_hogenom_alerax_rates.py` and `scripts/compare_backtracking_alerax_events.py` | Addressed after the workflow/scripts follow-up audit. The profiling helper now evaluates chunk NLLs through `_nll_per_family_with_cleanup()`, and the AleRax comparison helper samples through `_gpurec_event_counts_with_cleanup()`; both close the local `GeneReconModel` on success and after evaluation/sampling exceptions. | Keep fake-model legacy-script tests covering successful cleanup and exception cleanup for both local script paths. |
| Local validation/profiling CLI count controls | Addressed after the workflow/scripts follow-up audit. `profiling/evaluate_hogenom_alerax_rates.py` and `scripts/compare_backtracking_alerax_events.py` now build parsers with shared `gpurec/_argparse_types.py` helpers, so count controls such as chunk size, family count, sample count, iteration count, start index, seed, and wave size fail with parser-level positive, non-negative, or positive-even errors. | Keep parser-level legacy-script tests for invalid count controls on both checkout-local CLIs. |
| Resume optimizer-state restore in `gpurec/workflow/optimize.py` | Addressed after the workflow/scripts follow-up audit. Discard behavior now catches `ValueError`, `RuntimeError`, and `TypeError` from `optimizer.load_state_dict`, so malformed or backend-incompatible optimizer state is reported as discarded resume state instead of aborting resume. | Keep the direct fake-optimizer regression that exercises all three exception types and verifies `resume_optimizer_state=discarded` plus the original error text. |
| Dynamic CLI compatibility attribute `_RUN_CONFIG_CLI_OVERRIDE_FIELDS` in `gpurec/cli.py` | Addressed after the workflow/scripts follow-up audit. The only observed in-repo consumer was a CLI surface test; it now calls `_run_config_cli_override_fields()` directly and asserts the dynamic `_RUN_CONFIG_CLI_OVERRIDE_FIELDS` attribute is absent, so the module-level `__getattr__` compatibility hook was removed. | Keep the CLI surface test on parser destinations and the private helper, and keep the absence assertion so the test-only compatibility name does not return. |
| Rust sampler term variants in `crates/gpurec-backtrack/src/lib.rs` | Addressed after the workflow/backtracking follow-up audit. Direct `Sampler::apply_term` unit tests now cover `HiddenTransferLossDonor`, both hidden speciation directions, both split-transfer directions, normal split speciation, and swapped split speciation; they assert emitted event shape, species node mapping, and queued `WorkItem` clade/species state. | Keep the Rust unit tests and repository hygiene source guard that checks these direct branch tests remain present. |
| Python 3.10 TOML fallback in `scripts/check_release_metadata.py` | Addressed after the refresh audit. `_parse_minimal_pyproject()` remains a Python 3.10 compatibility fallback for hosts without `tomllib`, and direct tests now exercise its release-metadata subset without depending on the host interpreter path. | Keep fixture coverage for string/table readme and license fields, multiline classifier arrays, project URLs, ignored unrelated tables, and the current project release fields before editing the fallback parser. |

## Test Surface Cleanup

The test suite is broad, but some complexity now lives in tests that preserve
historical internals.

Fresh findings from the tests/Rust/docs follow-up audit:

| Surface | Current owner / risk | Guard before behavior change |
| --- | --- | --- |
| Python 3.10 unit collection for TOML-reading tests | Addressed after the tests/Rust/docs follow-up audit. TOML-reading unit modules now use a `tomllib`/`tomli` conditional import, and the dev extra installs `tomli` only on Python versions older than 3.11. | Keep the release metadata guard that verifies the Python-version-scoped `tomli` dev dependency while Python 3.10 remains supported. |
| Rust backtracking CLI multi-sample output | Addressed after the tests/Rust/docs follow-up audit. The CPU-only Rust fixture integration test now runs the real CLI with `--samples 2 --output-dir <tmpdir>`, checks `sample_0.xml` and `sample_1.xml`, and parses both files against the deterministic RecPhyloXML contract. | Keep the fixture README and hygiene guard tied to the multi-sample output names while the Rust CLI keeps `sample_{idx}.xml` semantics. |

Plan:

- Split `tests/unit/test_workflow.py` into behavior-focused modules:
  config, model construction, checkpointing, optimization runner, sampling CLI,
  public exports.
- Keep white-box tests for internals only where they guard deletion-prone
  contracts.
- Replace tests for soon-to-be-deleted helpers with characterization tests at
  the public model/evaluator level.
- Avoid preserving dead functions just because tests import them.
- Keep repository hygiene tests focused on public contracts and source guards
  that matter after pruning.

## Suggested Pruning Order

1. Mark internal/diagnostic/deprecated surfaces in docs.
2. Move tests off misleading likelihood aliases and scheduler helpers.
3. Add shared validation and layout abstractions.
4. Migrate `UniformChunkedReconModel` internals to shared evaluator.
5. Remove production env toggles whose defaults are retained.
6. Move native CUDA prototypes to benchmark/experimental ownership or delete.
7. Delete test-only scheduler helpers and C++ diagnostic exports.
8. Delete legacy preprocess compatibility paths.
9. Prune scripts and profiling entry points.
10. Split large tests and remove white-box references to deleted helpers.

## Verification Matrix

For each deletion:

- `rg` proves no production source imports the symbol.
- Docs either stop mentioning it or label it historical.
- Tests assert the new public behavior, not the old helper.
- `python -m pytest --collect-only -q` succeeds.
- CPU unit suite succeeds.
- CUDA/integration parity succeeds for affected compute paths.
- If deletion affects kernels or scheduling, strict benchmark command succeeds.

## Non-Goals

- Do not add new reconciliation models beyond the retained uniform-transfer
  modes.
- Do not promise double backward or exact Hessians.
- Do not keep every profiling knob as production behavior.
- Do not keep compatibility aliases indefinitely when their names contradict
  behavior.
