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

`gpurec/__init__.py` still documents lower-level imports such as
`GeneDataset`, `E_fixed_point`, `compute_nll`, and `Pi_wave_forward`.  The
top-level exported API is already high-level, but this docstring encourages
direct use of internals that are about to change.

Plan:

- Change package docs to call `gpurec.core` internal except for explicitly
  supported helpers.
- Add a "low-level API is unstable" section to developer docs.
- Keep tests importing internals as white-box tests, not as evidence of public
  support.

### Compatibility Aliases

`compute_log_likelihood()` and `compute_log_likelihood_root_rows()` return NLL.
They are currently used by tests and the benchmark script.

Plan:

- Move internal/profiling usage to `compute_nll()` or the new root-row helper.
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
| `finite_float()`, `positive_float()`, and `nonnegative_float()` in `gpurec/api/_validation.py` | Direct API float controls currently accept `True` as `1.0`, while integer controls and `theta_init_rates` reject bools. | Decide whether bools are invalid numeric controls; add direct-API tests for bool values in controls such as `tol_E`, `pi_max_diff_tol`, and `min_rate` before changing validation. |
| `as_family_param()`, `as_family_species()`, and `extract_parameters_uniform()` in `gpurec/core/extract_parameters.py` | Production-owned by forward/backward/autograd/implicit-gradient paths, but lightly documented and not directly table-tested. | Add CPU table tests for global, specieswise, genewise, `family_rows`, and `G == S` ambiguity before refactoring parameter-shape policy; add a contract docstring for `as_family_species()`. |
| `_normalize_family_tree_paths()` in `gpurec/core/model.py` | One-line private compatibility alias for `normalize_family_tree_paths()` with no in-repo callers found. | Add a source hygiene guard proving no tracked callers remain, then delete or explicitly document as a private compatibility shim if external use matters. |
| `normalize_family_chunk_size()` in `gpurec/core/batch_planning.py` | Shared by API, workflow, CLI, and tests but omitted from `__all__`, leaving export intent unclear. | Decide public versus internal status; if public, add to `__all__` and guard wildcard exports, otherwise migrate callers behind a supported validation API before renaming. |
| `UniformChunkedState` in `gpurec/api/uniform_chunked.py` | Public-looking state container used only by chunked autograd/evaluator internals, not exported or documented. | Rename to `_UniformChunkedState` with a source guard, or document it as an internal state container and keep it out of public exports. |
| `UniformChunkedReconModel.nll_per_family()` | Public method with integration coverage but no docstring/direct unit guard; README genewise-only wording can be misread as applying to chunked global models. | Add a direct unit test for `chunk_indices` and per-family semantics, and document it as a global/uniform chunked diagnostic before changing behavior. |
| `implicit_grad_loglik_vjp_wave()` in `gpurec/optimization/implicit_grad.py` | Public-looking low-level name used by API/autograd but not exported from `gpurec.optimization`; direct tests cover only `_bicgstab()`. | Classify as internal bridge or supported low-level API. If retained as public, add a tiny fake-system test for stats and convergence scheduling before changing the signature. |
| Direct `build_wave_layout()` family-index inputs in `gpurec/core/batching.py` | Addressed after the core/API follow-up audit. The helper now requires `family_clade_counts` and `family_clade_offsets` to be provided together, have matching lengths, contain nonnegative integer ranges, stay within `C`, avoid overlaps, and cover every clade before `family_idx` is materialized. | Keep CPU unit guards for mismatched, overlapping, out-of-bounds, and incomplete family metadata while direct callers can pass family metadata. |
| Explicit theta tensors in `gpurec/api/model.py` and `full_loss_for_theta()` | `theta_init` and explicit `theta` arguments are cloned or forwarded without a public shape check. Extra event columns can alter the softmax denominator, while short tensors fail later in parameter extraction. `theta_init_rates` has validation, but raw tensor inputs do not. | Add direct API tests for short, extra-column, wrong-rank, and mode-specific explicit theta shapes before centralizing shape validation. |
| `collate_gene_families()` docstring in `gpurec/core/batching.py` | Addressed after the core/API follow-up audit. The docstring now describes preprocessed gene-family CCP payloads and the current `build_wave_layout()` owner instead of removed `preprocess_gene_with_species` / `likelihood_2.py` surfaces. | Keep the source hygiene guard that rejects those removed surface names from the helper docstring while it stays in `gpurec.core`. |

## Scheduler Surface

Current scheduler surfaces:

- production construction uses `schedule_family_waves()` and
  `build_family_wave_layout()`;
- `schedule_global_phased_waves()` tries several internal algorithms;
- `collate_wave()`, `split_phase_waves()`, and `compute_clade_waves()` appear
  test/doc-only in current source;
- C++ extension exports multiple wave-stat diagnostic functions.

Ownership table from the current read-only audit:

| Surface | Current owner / callers | Tests / docs | Deletion risk |
| --- | --- | --- | --- |
| `preprocess_multiple_families` pybind | Production-owned. `GeneDataset` calls it for normal preprocessing, family cache misses, and species-only empty-family cache fill in `gpurec/core/model.py`. | Fake/cache tests in `tests/unit/test_alerax_family_input.py`, real parser coverage through `GeneDataset`, and integration construction in `tests/integration/test_gene_recon_model.py`. | High. Keep. Non-empty family preprocessing needs `include_details=True`; the empty-family species-only cache path currently uses the default `include_details=False`. |
| Legacy `preprocess` pybind | No in-repo production caller found; exported from `gpurec/core/cpp/preprocess.cpp`. | Existing pruning docs flag it as legacy/open surface. | Medium external/API risk, low in-repo runtime risk. Document as legacy/deprecated before removal. |
| `compute_phased_waves` pybind | No direct production caller found, but the underlying implementation is production-used to populate `phased_waves`/`phased_phases` during preprocessing. | Source-level hygiene guards it with the other max-wave exports. | Do not remove the implementation. Deprecate the direct export only after diagnostic ownership is documented. |
| Wave-stat pybinds: `compute_wave_stats`, `compute_packet_wave_stats`, `compute_phased_wave_stats`, `compute_phased_cross_family_wave_stats`, `compute_cross_family_wave_stats` | No production caller found. | Hygiene checks positive `max_wave_size`; audit docs describe them as broad diagnostic ABI. | Low runtime risk, medium diagnostic/API risk. Keep only with a maintained profiling or diagnostic command. |
| `bench_parse` | Not currently exported. | Removal is guarded in repository hygiene and audit docs. | Already retired; keep the guard. |
| `compute_clade_waves` Python helper | No production caller found; adapts required C++ `phased_waves`/`phased_phases`. | Unit-only in `tests/unit/test_scheduling.py`; current docs call it test/doc-only. | Low runtime risk. Move to a test fixture or keep as a documented compatibility adapter. |
| `collate_wave`, `split_phase_waves` | No production caller found. | Unit-only in `tests/unit/test_global_wave_scheduler.py`; current docs list them as deletion candidates. | Low runtime risk. Migrate tests or docs before deleting from `gpurec.core`. |
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

Candidate deletions after classification:

- `gpurec/core/batching.py:213` `collate_wave()`;
- `gpurec/core/batching.py:244` `split_phase_waves()`;
- `gpurec/core/scheduling.py:13` `compute_clade_waves()`;
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

Current package runtime reads these groups:

- binary/distribution:
  `GPUREC_BACKTRACK_BIN`, `GPUREC_ALERAX_COMPAT`;
- memory policy:
  `GPUREC_MEMORY_POLICY_FRACTION`, `GPUREC_MEMORY_POLICY_RESERVE_GIB`;
- forward/backward production toggles:
  `GPUREC_FUSE_FINAL_PIBAR`, `GPUREC_SPECIALIZE_NONLEAF_LEAF_TERM`,
  `GPUREC_BACKWARD_NO_CPU_PRUNING`,
  `GPUREC_DTS_SKIP_INACTIVE_PIBAR_ZERO`;
- Triton/CUDA tuning:
  `GPUREC_WAVE_STEP_BLOCK_S`, `GPUREC_WAVE_STEP_NUM_WARPS`,
  `GPUREC_DTS_PARENT_BLOCK_S`, `GPUREC_DTS_PARENT_TILE_SPLITS`,
  `GPUREC_DTS_PARENT_NUM_WARPS`, `GPUREC_SELF_LOOP_2D_*`,
  `GPUREC_DTS_BLOCK_S`, `GPUREC_DTS_NUM_WARPS`,
  `GPUREC_DTS_GRAD_MT_TILE_SPLITS`,
  `GPUREC_PIBAR_UD_BLOCK_S`, `GPUREC_PIBAR_UD_NUM_WARPS`;
- native CUDA prototype selectors/tuning:
  `GPUREC_CUDA_SELF_LOOP_NOSPLIT`, `GPUREC_CUDA_SELF_LOOP_SPLIT`,
  `GPUREC_CUDA_SELF_LOOP_NOSPLIT_CORRECTION`,
  `GPUREC_CUDA_SELF_LOOP_BLOCK`,
  `GPUREC_CUDA_SELF_LOOP_CHILD_EDGE_WEIGHT`,
  `GPUREC_CUDA_PIBAR_FROM_UD`, `GPUREC_CUDA_PIBAR_FROM_UD_STRICT`,
  `GPUREC_CUDA_PIBAR_FROM_UD_BLOCK`,
  `GPUREC_CUDA_PIBAR_FROM_UD_PAD_SHARED`.

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
| Local model construction in `profiling/evaluate_hogenom_alerax_rates.py` and `scripts/compare_backtracking_alerax_events.py` | Per-chunk/per-family `GeneReconModel` instances are not closed on success or exceptions. | Wrap model use in `try/finally: model.close()` and add fake-model tests covering success and exception paths. |
| Local validation/profiling CLI count controls | `profiling/evaluate_hogenom_alerax_rates.py` and `scripts/compare_backtracking_alerax_events.py` parse count controls such as chunk size, family count, sample count, iteration count, and wave size as raw integers. Zero or negative values can produce generic runtime errors or no-op output before script-level validation. | Add parser/unit tests for invalid count controls, then share positive/nonnegative integer validators or `argparse` type helpers before changing these checkout-local CLIs. |
| Resume optimizer-state restore in `gpurec/workflow/optimize.py` | Addressed after the workflow/scripts follow-up audit. Discard behavior now catches `ValueError`, `RuntimeError`, and `TypeError` from `optimizer.load_state_dict`, so malformed or backend-incompatible optimizer state is reported as discarded resume state instead of aborting resume. | Keep the direct fake-optimizer regression that exercises all three exception types and verifies `resume_optimizer_state=discarded` plus the original error text. |
| Dynamic CLI compatibility attribute `_RUN_CONFIG_CLI_OVERRIDE_FIELDS` in `gpurec/cli.py` | Exposed through module `__getattr__`; observed in-repo usage is test-only. | Move tests to `_run_config_cli_override_fields()` or parser destinations, then remove the dynamic attribute unless it is an intentional public compatibility promise. |
| Rust sampler term variants in `crates/gpurec-backtrack/src/lib.rs` | Branches such as `HiddenTransferLossDonor`, hidden speciation, split transfer, and swapped split speciation mutate event labels and queued work differently but lack direct branch tests. | Add `Sampler::apply_term` unit tests asserting emitted event shape and queued `WorkItem` species/clade for each variant. |
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
