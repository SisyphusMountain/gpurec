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

## Workflow Submodule Helpers

The supported workflow shortcut surface is the lazy `gpurec.workflow` export
set plus top-level `gpurec` re-exports.  Direct submodule imports are classified
below so internal helpers can be simplified without accidentally turning them
into public contracts.

| Surface | Current owner / risk | Guard before deletion or promotion |
| --- | --- | --- |
| `gpurec/workflow/__init__.py` | Public lazy export facade for `RunConfig`, `SamplingConfig`, workflow runners, result dataclasses, `optimize`, and `sample`. | Keep top-level workflow export guards and import-smoke tests. |
| `gpurec/workflow/config.py` | Public flat `RunConfig` / `SamplingConfig` contract, JSON path resolution, and compatibility wrappers for optimizer defaults and route metadata. | Keep CLI/dataclass surface parity, run-config reference coverage, and template/preflight smokes. |
| `gpurec/workflow/optimize.py` | Public runner implementation behind `OptimizationRunner` and `optimize`; owns end-to-end likelihood optimization and final artifacts. | Keep workflow optimizer mode tests, artifact publication tests, and CPU unit gate before changing optimizer behavior. |
| `gpurec/workflow/_run_state.py` | Internal optimizer run-state plumbing for `OptimizationRunner`; direct imports are compatibility-only through `gpurec.workflow.optimize`. | Keep optimize re-export identity and checkpoint/row parity tests before changing run-state ownership. |
| `gpurec/workflow/_transition_types.py` | Internal workflow transition DTOs shared by optimization orchestration helpers. | Keep `_transitions` compatibility aliases and transition/checkpoint parity tests before changing DTO ownership. |
| `gpurec/workflow/_route_defaults.py` | Internal production route/default policy helper behind `gpurec.workflow.config` wrappers. | Keep config facade imports, route-audit parity tests, and strict CLI route gates before changing route ownership. |
| `gpurec/workflow/sampling.py` | Public runner implementation behind `SamplingRunner` and `sample`; owns checkpoint-backed RecPhyloXML sampling outputs. | Keep sampling workflow tests and backtracking distribution checks. |
| `gpurec/workflow/checkpoint.py` | Supported lower-level checkpoint tooling for advanced restore/inspection, but not a top-level shortcut. | Keep checkpoint `__all__`, identity metadata, route metadata, and compatibility guards. |
| `gpurec/workflow/_artifact_publish.py` | Internal staged-artifact publication helper shared by optimization and sampling. | Keep rollback/backup tests before changing final artifact publishing. |
| `gpurec/workflow/_cleanup.py` | Internal cleanup and exception-chaining helper shared by optimization and sampling. | Keep cleanup failure-context tests before changing error handling. |
| `gpurec/workflow/_metadata.py` | Internal checkpoint payload validation and model identity helper shared by checkpoint/optimization paths. | Keep checkpoint metadata validation tests before moving or deleting it. |
| `gpurec/workflow/diagnostics.py` | Internal strict JSON/CSV and likelihood/gradient/solver summary helper. | Keep strict JSON and summary/artifact tests before changing diagnostics output. |
| `gpurec/workflow/model_factory.py` | Internal AleRax workflow model construction boundary for CUDA checks and workflow-only model options. | Keep CLI/model factory forwarding tests and production preflight docs before changing construction policy. |

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

### Removed Compatibility Aliases

Addressed after the public-API documentation follow-up audit.
The former low-level likelihood aliases `compute_log_likelihood()` and
`compute_log_likelihood_root_rows()` returned NLL despite their names.  Tracked
runtime, profiling, and ordinary test usage now uses `compute_nll()`,
`gather_root_rows()`, and `compute_nll_root_rows()`, and the aliases have been
removed from `gpurec.core.likelihood`.

Plan:

- Keep runtime/profiling usage on `gather_root_rows()` plus the root-row helper
  when callers already hold full wave-ordered `Pi`.
- Keep repository hygiene coverage that prevents tracked Python surfaces from
  reintroducing the removed aliases.
- If external compatibility matters again, add a new explicitly owned
  compatibility shim instead of restoring the misleading names silently.

### Uniform Chunked API

`UniformChunkedReconModel` duplicates construction, validation, chunk planning,
evaluation, loss/gradient scaling, metadata, and dtype policy already present
in or adjacent to `GeneReconModel`.

Addressed first API cleanup slice: chunked evaluation internals now live in the
private `gpurec/api/_uniform_chunked_eval.py` helper.  The public
`UniformChunkedReconModel` / `UniformChunkMetadata` facade, constructor flow,
chunk/state containers, and old private import aliases remain in
`gpurec/api/uniform_chunked.py`.

Plan:

- Move chunk selection and `loss_and_grad(chunk_indices=...)` into the shared
  evaluator.
- Keep the public class as a thin adapter while users migrate.
- Keep family path/name/map normalization shared through
  `gpurec.core.model.normalize_family_inputs()` so dataset constructors and the
  uniform chunked API reject mismatched or duplicate family metadata through one
  implementation.

### Core/API Refresh Findings

Current unresolved findings from the follow-up core/API audit:

| Surface | Current owner / risk | Guard before behavior change |
| --- | --- | --- |
| `finite_float()`, `positive_float()`, and `nonnegative_float()` shared through `gpurec._validation` | Addressed after the core/API follow-up audit. Shared direct-API and workflow float validators now reject Python bools and bool tensors before numeric coercion, so controls such as `tol_E`, `pi_max_diff_tol`, and `min_rate` fail before CUDA checks or theta mutation. The Pi-adjoint fixed-point relaxation control now uses the same positive-float semantics through a core solver helper while retaining its fixed legacy error message. | Keep direct validation tests for bool values in the shared helpers, `GeneReconModel` constructor float controls, `GeneReconModel.clamp_theta_()`, and the Pi-adjoint relaxation helper plus source guard. |
| `as_family_param()`, `as_family_species()`, and `extract_parameters_uniform()` in `gpurec/core/extract_parameters.py` | Addressed after the core/API follow-up audit. CPU table tests now cover global, specieswise, and genewise `extract_parameters_uniform()` output shapes/values, plus `as_family_param()` / `as_family_species()` `family_rows` precedence when `G == S`. The `as_family_species()` docstring now documents the broadcast contract and the bare length-`G` ambiguity. | Keep the direct extraction-helper table tests and the docstring guard before refactoring parameter-shape policy. |
| `_normalize_family_tree_paths()` in `gpurec/core/model.py` | Addressed after the core/API follow-up audit. The one-line private compatibility alias was deleted after a source hygiene guard proved no tracked `gpurec/`, `scripts/`, or `profiling/` callers remain; callers use public `normalize_family_tree_paths()`. | Keep the source hygiene guard proving the private alias is absent from tracked runtime/script/profiling Python sources. |
| `gpurec.core.batch_planning.__all__`: `FamilyBatchPlan`, `normalize_batch_packing`, `normalize_clade_budget`, `normalize_family_chunk_size`, and `plan_family_batches` | Addressed after the public-API/docs follow-up audit. The whole exported set is retained as a narrow shared low-level planning boundary because API, workflow, CLI, memory policy, and white-box tests share these helpers for the same family-batch semantics. The module docstring marks this as a support boundary, not a promise that the rest of `gpurec.core` is stable. | Keep the batch-planning wildcard export guard for the exact exported set before changing planning ownership. |
| `UniformChunkedState` in `gpurec/api/uniform_chunked.py` | Addressed after the core/API follow-up audit. The state container was renamed to `_UniformChunkedState` after documenting that it is owned by chunked autograd/evaluator internals and has no direct tracked runtime callers outside `uniform_chunked.py`. | Keep the source/export guard proving `UniformChunkedState` is absent as a class/name reference and that `_UniformChunkedState` stays out of `gpurec.api.uniform_chunked.__all__`. |
| `UniformChunkedReconModel.nll_per_family()` | Addressed after the core/API follow-up audit. README and API docstrings now distinguish `GeneReconModel.nll_per_family()` / `full_nll_per_family()` genewise-only independent losses from `UniformChunkedReconModel.nll_per_family(chunk_indices=...)`, a no-grad global/uniform diagnostic returning selected shared-theta family NLLs after chunk filtering. | Keep the direct CPU unit guard that monkeypatches the chunked evaluator and asserts `need_grad=False`, `per_family=True`, exact `chunk_indices`, and disabled grad mode. |
| `implicit_grad_loglik_vjp_wave()` in `gpurec/optimization/implicit_grad.py` | Addressed after the core/API follow-up audit. The function is documented as an internal bridge between `gpurec.api.model`, `gpurec.api.autograd`, and retained optimization internals, not a supported low-level public API. It remains out of `gpurec.optimization.__all__`, and a source guard limits tracked runtime references to the API/autograd bridge callers plus its definition module. | Keep the doc/export/call-site hygiene guard; do not add external callers without promoting and testing it as public API. |
| Direct `build_wave_layout()` family-index inputs in `gpurec/core/batching.py` | Addressed after the core/API follow-up audit. The helper now requires `family_clade_counts` and `family_clade_offsets` to be provided together, have matching lengths, contain nonnegative integer ranges, stay within `C`, avoid overlaps, and cover every clade before `family_idx` is materialized. | Keep CPU unit guards for mismatched, overlapping, out-of-bounds, and incomplete family metadata while direct callers can pass family metadata. |
| Explicit theta tensors in `gpurec/api/model.py` and `full_loss_for_theta()` | Addressed after the core/API follow-up audit. Shared `validate_theta_shape()` now enforces exact raw tensor shapes for the active parameter-sharing mode: global `[3]`, specieswise `[S, 3]`, or genewise `[G, 3]`, and rejects non-floating or nonfinite theta values before CUDA work. Short tensors, extra event columns, wrong ranks, and wrong row counts now fail before CUDA checks, theta cloning, or streaming parameter extraction; integer tensors, NaNs, and infinities are rejected at the same boundary. | Keep direct API tests for invalid `theta_init` shapes/values and invalid explicit `theta` shapes/values, plus the existing valid streaming-contract guard for `full_loss_for_theta()`. |
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
- former C++ wave-stat diagnostic exports are absent from the current
  Rust/PyO3 extension and guarded from returning.

Ownership table from the current read-only audit:

| Surface | Current owner / callers | Tests / docs | Deletion risk |
| --- | --- | --- | --- |
| Rust/PyO3 preprocessing exports: `preprocess_dataset`, `preprocess_request_binary`, `preprocess_request_numpy`, `preprocess_request_torch` | Production-owned. `GeneDataset`, `UniformChunkedReconModel`, and the preprocessing wrappers call these for resident preprocessing and request conversion. | Fake/cache tests in `tests/unit/test_alerax_family_input.py`, real parser coverage through `GeneDataset`, uniform chunked tests, and integration construction in `tests/integration/test_gene_recon_model.py`. | High. Keep. Package runtime passes `include_details=True` for all tracked `preprocess_multiple_families` wrapper calls, including the empty-family species-only cache fill; `include_details=False` is only a Python-wrapper compatibility knob pending public-ABI review. |
| Rust/PyO3 topology exports: `species_parent_from_indexes_torch`, `species_wave_topology_torch`, `uniform_ancestors_t_indices_torch` | Production-owned through `gpurec/core/preprocess_rust.py` helpers used by topology and ancestor-matrix construction. | Covered by topology/ancestor preprocessing tests and source-level export-manifest hygiene. | High. Keep while the Python wrappers call the native topology helpers. |
| Rust/PyO3 scheduler/layout JSON exports: `schedule_global_phased_waves_json`, `family_schedule_summary_json`, `plan_family_batches_json`, `build_wave_layout_plan_json` | Production-owned through `gpurec/core/schedule_rust.py` and layout/batch planning wrappers. | Covered by global scheduler, family-layout, batch-planning, Rust parity, and workflow tests. | High. Keep while Rust scheduler/layout planning remains the default native path. |
| Former legacy and diagnostic C++ pybind exports: `preprocess`, `compute_phased_waves`, wave-stat helpers, and `bench_parse` | Retired from the current Rust/PyO3 extension surface. | Repository hygiene guards the exact `wrap_pyfunction!` manifest and rejects the old `PYBIND11_MODULE`/`m.def` style from the native source. | Already retired; keep the absence guard. |
| `compute_clade_waves` Python helper | Addressed after the scheduler follow-up audit. The Python adapter and its helper-level unit module were deleted after confirming no tracked production caller imports it and no high-level public export exposes it. | Keep the source guard proving `gpurec/core/scheduling.py` and the Python adapter name do not return to tracked runtime Python. | Low in-repo runtime risk; direct low-level Python imports should use `schedule_global_phased_waves()` or `build_wave_layout()`, while Rust scheduling owns phased-wave generation. |
| `collate_wave`, `split_phase_waves` | Addressed after the scheduler follow-up audit. These helper-level scheduler functions were deleted from `gpurec.core.batching` after confirming no tracked production caller imports them and their only tracked users were direct helper tests. | Keep the source hygiene guard proving these helper names are absent from tracked runtime Python sources. | Low in-repo runtime risk; direct low-level external imports should use `schedule_global_phased_waves()` or `build_wave_layout()`. |
| Rust-default scheduler/layout path | Production-owned through `gpurec/api/_family_layout.py`, Rust scheduler/layout planning wrappers, and retained Python fallback implementations. | Covered by global scheduler, family-layout, Rust parity, and full workflow tests. | High. Keep Python fallback as a parity oracle until replacement packaging and benchmark policy are settled. |
| `family_schedule_summary` | Production-owned for depth-first batch packing in `GeneReconModel` and `UniformChunkedReconModel`. | Indirectly covered through planning/layout tests. | High while depth-first packing remains supported. |

Plan:

1. Keep the ownership table current when scheduler or pybind exports move
   between product runtime, benchmark diagnostic, test-only helper, and delete
   buckets.
2. Benchmark the current multi-candidate scheduler against one candidate policy.
3. Keep the chosen runtime scheduler private to the layout builder.
4. Move test-only scheduler helpers into `tests/` fixtures or delete them.
5. Keep retired C++ diagnostic pybind exports out of the current Rust/PyO3
   manifest unless a maintained profiling command reintroduces them explicitly.

Remaining candidate deletions after classification:

- no direct wave-stat pybind deletions remain in the current source; keep the
  removed-surface guards.

## Native Preprocess Extension Surface

Current runtime calls:

- `GeneDataset` uses `preprocess_multiple_families(..., include_details=True,
  include_species_matrices=False)`.
- The species-only empty-family cache path also requests
  `include_details=True`; no tracked package runtime caller depends on
  `include_details=False`.
- Cache loading expects detailed CCP helpers and leaf mapping tensors.

Open surfaces:

- low-level `include_details=False` compatibility on the Python
  `preprocess_multiple_families` wrapper;
- retained Rust/PyO3 preprocessing, topology, scheduler, and layout helper
  exports listed in the manifest above.

CPP-01/CPP-02 refresh: tracked package Python still has exactly one production
preprocessing route, `GeneDataset` calling
`preprocess_multiple_families`.  Both tracked package call sites request
`include_details=True`, including the species-only empty-family cache fill.  No
tracked package Python calls `include_details=False`, and the legacy
`preprocess()` plus direct wave-stat diagnostic exports are absent from the
current Rust/PyO3 manifest.  Deletion remains blocked only for the Python
wrapper's low-level `include_details=False` compatibility knob: remove that
after deprecation/replacement evidence for low-level callers.  Retired C++
diagnostic exports should stay absent unless a maintained profiling/diagnostic
owner explicitly reintroduces them.

Rust/PyO3 export manifest:

| Export | Classification | Replacement / deletion gate |
| --- | --- | --- |
| `preprocess_dataset`, `preprocess_request_binary`, `preprocess_request_numpy`, `preprocess_request_torch` | Production-owned preprocessing entry points. | Keep while `GeneDataset`, `UniformChunkedReconModel`, and preprocessing wrappers use native preprocessing. |
| `species_parent_from_indexes_torch`, `species_wave_topology_torch`, `uniform_ancestors_t_indices_torch` | Production-owned topology helper exports. | Keep while `gpurec/core/preprocess_rust.py` uses native topology helpers. |
| `schedule_global_phased_waves_json`, `family_schedule_summary_json`, `plan_family_batches_json`, `build_wave_layout_plan_json` | Production-owned scheduler/layout planning exports. | Keep while Rust scheduling and layout planning remain supported. |
| `preprocess`, `compute_phased_waves`, wave-stat helpers, `bench_parse` | Retired legacy/diagnostic exports. | Keep absent unless a maintained low-level API or profiling owner is added with tests and docs. |

Plan:

- Search installed/user docs for the low-level `include_details=False`
  compatibility knob before deletion.
- If no public owner exists, remove low-level `include_details=False`
  compatibility from the Python wrapper.
- Keep one path from Newick inputs to detailed family/species helpers.
- Keep native export-manifest guards around the retained path.

Verification:

- Rust/PyO3 extension imports cleanly.
- AleRax family input tests pass.
- Preprocess cache validation tests pass.
- Integration construction from `from_trees()` and `from_alerax_families()`
  still works.

## Environment Variable Surface

The supported runtime environment surface is intentionally small.  This table
is a support manifest for environment variables that users may rely on; package
code reads only the supported variables in this manifest, and repository
hygiene checks compare tracked package reads against it.

### Environment Owner Manifest

| Variable(s) | Current ownership | Runtime owner / notes |
| --- | --- | --- |
| `GPUREC_BACKTRACK_BIN` | User-facing | Binary/distribution contract for `gpurec sample`, `gpurec run`, `gpurec backtrack-check`, and Python backtracking helpers forced onto `backend="cli"`. |
| `GPUREC_BACKTRACK_NATIVE_LIB` | User-facing discovery | Optional native PyO3 backtracking library discovery for Python helper calls using `backend="native"` or native `auto` resolution. CLI sampling still uses `GPUREC_BACKTRACK_BIN`, `--backtrack-binary`, or the source-tree Cargo binary fallback. |
| `GPUREC_ALERAX_COMPAT` | User-facing compatibility | Compatibility guard read by API validation; supported differentiable optimization accepts only unset or `0`. |
| `GPUREC_MEMORY_POLICY_FRACTION`, `GPUREC_MEMORY_POLICY_RESERVE_GIB` | User-facing | Memory-budget margins for uniform chunk planning. |
| `GPUREC_PREPROCESS_BIN`, `GPUREC_PREPROCESS_NATIVE_LIB` | User-facing discovery | Optional Rust preprocessing CLI/native-library discovery while source builds and prebuilt artifacts coexist. |
| `GPUREC_TORCH_SEED` | User-facing | Optional deterministic seed for optimization/runtime startup; recorded in run-manifest reproducibility metadata when set. |

Supported environment flags are limited to `GPUREC_BACKTRACK_BIN`,
`GPUREC_BACKTRACK_NATIVE_LIB`, `GPUREC_ALERAX_COMPAT`,
`GPUREC_MEMORY_POLICY_FRACTION`, `GPUREC_MEMORY_POLICY_RESERVE_GIB`,
`GPUREC_PREPROCESS_BIN`, `GPUREC_PREPROCESS_NATIVE_LIB`, and `GPUREC_TORCH_SEED`.  Scheduler backend
selection, preprocess cache locations, backward CUDA/Triton selectors, and
kernel launch tuning are not supported environment contracts.

Plan:

- Keep binary/distribution, memory-policy, AleRax compatibility, and Rust
  preprocessing discovery env vars.
- Keep scheduler selection, preprocess cache paths, and profiling controls as
  explicit API/CLI/profiling arguments rather than environment aliases.
- Keep backward CUDA/Triton diagnostic selectors and kernel launch tuning out
  of the supported docs/test manifest.
- Keep new package environment reads out of the runtime unless they are added
  to this manifest with a user-facing owner and documentation.

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

- Keep `workflow/config.py` string/path adapters thin over shared validators;
  required and optional integer controls now delegate to `gpurec._validation`,
  including the shared batch-planning adapters and resident memory-policy
  estimators.  The Rust scheduler bridge keeps only its string adapter locally
  before using the shared integer validator, and checkpoint resume metadata
  keeps checkpoint-specific errors while using the shared nonnegative-integer
  and finite-float validators.  Resident-model `prefetch_batches` keeps only
  its `all`/disabled string aliases locally before using the same
  nonnegative-integer validator.  The backtracking bridge keeps seed/event
  range checks locally but uses the shared integer validator for coercion.
  Keep `gpurec._validation` torch-lazy so this does not make checkpoint
  metadata imports heavy.
- Keep `dtype_from_name()` workflow-specific only if CLI wording needs it.
- Keep optimizer modes only if behavior is tested by fake-model guards.

## Scripts And Profiling Surface

Current scripts include maintained workflow helpers, profiling utilities,
legacy HOGENOM launchers, report generators, and export tools.  The docs now
have an ownership matrix, and repository hygiene tests require every tracked
`scripts/*.py` / `scripts/*.R` entry point to appear in that matrix with an
allowed status and migration/deletion wording for retained legacy surfaces.
Deletion has not happened.

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
| `tests/data/**/output/` | Runtime-generated local test outputs. | Prior local runs. | Per-family likelihoods, XML, CSV, and other generated artifacts. | Regenerable byproducts. | Delete/regenerate as needed; never promote as expected fixtures. |

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
- Keep tests asserting only maintained scripts are in the product/benchmark
  matrix and every retained legacy script has an explicit owner or
  migration/deletion path.

### Workflow/Backtracking Refresh Findings

Current findings from the follow-up workflow/backtracking audit:

| Surface | Current owner / risk | Guard before behavior change |
| --- | --- | --- |
| Rust sampler help preflight in `gpurec/backtracking.py` | Addressed after the refresh audit. `_BACKTRACK_HELP_MARKERS` now requires the wrapper-supported `--samples`, `--seed`, `--output-dir`, and `--max-events` flags instead of accepting stale short help. | Keep the negative stale-help regression so old Rust binaries fail preflight before sampling. |
| LBFGS post-step evaluation in `gpurec/workflow/optimize.py` | Addressed after the refresh audit. The LBFGS branch evaluates the current theta after `optimizer.step(closure)` and now repeats the finite loss/gradient guard used by Adam/Adagrad. | Keep the fake LBFGS regression that returns a finite closure followed by a nonfinite current-theta evaluation and expects failed `nonfinite_objective_or_gradient` status with no LBFGS row recorded. |
| Final optimization evaluation in `gpurec/workflow/optimize.py` | Addressed after the workflow/scripts follow-up audit. The mandatory `final_eval` now repeats the finite loss/gradient guard; if it fails, the run status becomes `failed/nonfinite_objective_or_gradient`, the final row carries explicit `optimizer/final_eval_status` and `optimizer/final_eval_reason` fields, and nonfinite final metrics are not copied into the final row. | Keep the fake-model regression where a finite optimizer step is followed by a nonfinite final evaluation, and verify failed status, failed final-row markers, finite previous objective in the summary, and a failed latest checkpoint. |
| Local model construction in `profiling/evaluate_hogenom_alerax_rates.py` and `scripts/compare_backtracking_alerax_events.py` | Addressed after the workflow/scripts follow-up audit. The profiling helper now evaluates chunk NLLs through `_nll_per_family_with_cleanup()`, and the AleRax comparison helper samples through `_gpurec_event_counts_with_cleanup()`; both close the local `GeneReconModel` on success and after evaluation/sampling exceptions. | Keep fake-model legacy-script tests covering successful cleanup and exception cleanup for both local script paths. |
| Local validation/profiling CLI count controls | Addressed after the workflow/scripts follow-up audit and the full-pipeline benchmark refresh. `profiling/evaluate_hogenom_alerax_rates.py`, `scripts/compare_backtracking_alerax_events.py`, and `profiling/bench_uniform_forward_backward_pipeline.py` now build parsers or post-parse converters with shared `gpurec/_argparse_types.py` helpers, so count controls such as chunk size, family count, sample count, iteration count, start index, seed, and wave size fail with parser-level positive, non-negative, or positive-even errors. | Keep parser-level legacy-script tests for invalid count controls on checkout-local CLIs and the maintained full-pipeline benchmark. |
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
- Keep `tests/README.md` and repository hygiene guards aligned so legacy script
  tests own cleanup/parser/checkpoint contracts, while removed internals get a
  replacement behavior test or a deliberate deletion note.
- Keep repository hygiene tests focused on public contracts and source guards
  that matter after pruning.

## Suggested Pruning Order

1. Mark internal/diagnostic/deprecated surfaces in docs.
2. Keep tests off removed likelihood aliases and scheduler helpers.
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
