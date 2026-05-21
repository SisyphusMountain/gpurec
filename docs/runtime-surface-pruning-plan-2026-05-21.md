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

## Scheduler Surface

Current scheduler surfaces:

- production construction uses `schedule_family_waves()` and
  `build_family_wave_layout()`;
- `schedule_global_phased_waves()` tries several internal algorithms;
- `collate_wave()`, `split_phase_waves()`, and `compute_clade_waves()` appear
  test/doc-only in current source;
- C++ extension exports multiple wave-stat diagnostic functions.

Plan:

1. Add a scheduler ownership table:
   - product runtime;
   - benchmark diagnostic;
   - test-only helper;
   - delete.
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

## Test Surface Cleanup

The test suite is broad, but some complexity now lives in tests that preserve
historical internals.

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
