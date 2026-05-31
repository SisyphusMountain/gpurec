# Core Simplification Suggestions

This note lists cleanup opportunities found in `gpurec/core` after the lean fast-path work. The goal is to keep uniform, genewise, and specieswise modes while removing slower fallbacks, duplicated scheduling paths, and debug-only code from the runtime surface.

Status: historical cleanup snapshot. Several items below have since been
implemented or superseded, including removal of the old segmented logsumexp
kernel, Python BFS wave scheduling fallback, `collate_wave_cross`, legacy
split-parent reconstruction, non-parent-reduced DTS path, non-compact Pibar VJP
path, and `transfer_mat` solver inputs. Revalidate each item against the
current source tree before treating it as active backlog.

## Highest Confidence Deletions

### Remove `scatter_lse.py`

`gpurec/core/kernels/scatter_lse.py` appears unused by the retained DTS path. It now mostly contains a standalone segmented logsumexp kernel, internal tests, and a benchmark CLI.

Related cleanup:

- Delete `_seg_logsumexp_host` from `gpurec/core/_helpers.py`.
- Remove comments in `batching.py` that describe ge2 reduction through `seg_logsumexp`.

### Require C++ Phased Waves

`gpurec/core/scheduling.py` still has a Python BFS fallback. In the lean branch, preprocessing should be required to emit `phased_waves` and `phased_phases`.

Suggested cleanup:

- Keep `compute_clade_waves` as a thin validator/adapter for C++ phased waves.
- Delete `_compute_clade_waves_bfs`.
- Delete `wave_stats` if it is not used by tests or profiling.
- Raise a clear error when phased wave data is missing.

### Delete `collate_wave_cross`

`gpurec/core/batching.py::collate_wave_cross` is another cross-family scheduler implementation. It is not part of the retained measured path.

Suggested cleanup:

- Delete `collate_wave_cross`.
- Keep only `collate_wave`, `split_phase_waves`, and `build_wave_layout`.

### Remove Legacy Split-Parent Reconstruction

`batching.py` reconstructs `split_parents_sorted` from legacy segment metadata. If C++ preprocessing always emits `split_parents_sorted`, this branch can be removed.

Suggested cleanup:

- Delete `_reconstruct_split_parents`.
- In `collate_gene_families` and `build_wave_layout`, require `split_parents_sorted`.
- Add a focused preprocessing test that asserts this field exists.

## Kernel Consolidation

### Always Use Parent-Reduced DTS

`forward.py::_compute_split_dts` still branches between:

- `compute_dts_forward` for waves with ge2 clades.
- `dts_fused` plus a materialized `[n_splits, S]` temporary for eq1-only waves.

The lean path should prefer the parent-reduced route everywhere if benchmarks confirm no eq1-only regression.

Suggested cleanup:

- Make `_compute_split_dts` always call `compute_dts_forward`.
- Delete `dts_fused` and `_dts_fused_kernel` from `gpurec/core/kernels/dts_fused.py`.
- Keep the eq1 direct-write kernel and ge2 two-stage reduced kernels.

### Replace Mode Branches With Strided Layouts

Several kernels branch on parameter modes such as shared scalar, shared species, family scalar, and family species. These can be represented with base pointers plus row/species strides.

Targets:

- `gpurec/core/kernels/dts_fused.py`
- `gpurec/core/kernels/wave_step.py`
- `gpurec/core/kernels/wave_backward.py`

Suggested representation:

- Shared scalar: row stride `0`, species stride `0`.
- Shared species: row stride `0`, species stride `1`.
- Family scalar: row stride `1`, species stride `0`.
- Family species: row stride `S`, species stride `1`.

This keeps uniform, genewise, and specieswise support without keeping separate logic branches in every kernel.

### Require Compact Pibar VJP

`wave_backward.py` still contains both compact-level and non-compact Pibar VJP paths. Since `backward.py` now builds and passes compact topology, the non-compact path should be removable.

Suggested cleanup:

- Require `compact_level_ptr`, `compact_level_parents`, `compact_level_child1`, and `compact_level_child2`.
- Delete the legacy non-compact Pibar VJP kernel.
- Keep `_pibar_vjp_kernel`.

## Runtime Policy Cleanup

### Remove Baseline Memory-Policy Fallbacks

`gpurec/core/memory_policy.py` still models both the retained 2D self-loop path and an older baseline scratch path.

Suggested cleanup:

- Delete `baseline_wave_scratch_bytes`.
- Remove `proposal0=False` policy candidates.
- Make `choose_uniform_pipeline_policy` either return a 2D self-loop policy or raise a clear memory error.
- Keep `proposal0_memory_gate` only if it is still needed as a preflight guard.

### Remove Slow-Path Env Selectors

The lean code should avoid env vars that select slower implementations.

Candidates to remove or make benchmark-only:

- `GPUREC_SELF_LOOP_2D_TRITON` as an enable/disable selector.
- `GPUREC_SELF_LOOP_2D_MEMORY_GATE` as a selector.
- `GPUREC_DTS_BACKWARD_DEVICE_SCALARS`.
- `GPUREC_WAVE_SPLIT_METADATA_INT32`.

Block size and warp env vars can remain temporarily only if profiling shows they are useful for tuning.

## Core API Cleanup

### Split Data Loading From Runtime Evaluation

`gpurec/core/model.py` still contains older likelihood evaluation methods. The higher-level runtime now mostly lives in `gpurec/api/model.py` and `gpurec/api/uniform_chunked.py`.

Suggested cleanup:

- Keep `GeneDataset` focused on preprocessing, cache loading, and family metadata.
- Move or delete `compute_likelihood` and `compute_likelihood_batch` from `core/model.py`.
- Remove `change_dtype`, `set_params`, and `_normalize_max_transfer` if no public API relies on them.

### Simplify E Solver Inputs

`gpurec/core/likelihood.py::E_step` and `E_fixed_point` still accept `transfer_mat`, but the retained path uses uniform transfer through `max_transfer_mat`.

Suggested cleanup:

- Remove `transfer_mat` from `E_step`, `E_fixed_point`, and call sites.
- Keep only `max_transfer_mat`.
- Keep the 1D and 2D broadcast support needed for uniform, specieswise, and genewise modes.

### Centralize Species Topology Caches

Forward, backward, and helper modules rebuild overlapping species-tree structures.

Suggested cleanup:

- Add one helper in `gpurec/core/species.py` that returns cached device tensors:
  - `sp_child1`
  - `sp_child2`
  - `sp_parent`
  - `max_ancestor_depth`
  - compact level arrays
  - `ancestors_T`
- Make `forward.py` and `backward.py` use that helper instead of duplicating topology setup.

## Forward API Simplification

`Pi_wave_forward` currently has optional output modes:

- full original-order `Pi`
- root rows only
- wave-ordered `Pi`
- saved `Pibar`
- optional `uniform_pibar_row_max`

Suggested cleanup:

- Split inference and training into two explicit functions, or return one stable object with required fields.
- Keep training path as the default performance path: wave-ordered `Pi`, wave-ordered `Pibar`, and `uniform_pibar_row_max`.
- Keep root-row inference as a separate lightweight path.
- Remove `clade_species_map`, which is currently always `None`.

## Validation Checklist

Before deleting each branch, run:

- Unit tests for uniform, specieswise, and genewise modes.
- A focused `test_trees_1000` forward plus backward benchmark.
- A parity check for `pi_matrix()` in genewise and specieswise modes.
- A strict kernel check that no deleted fallback path is called.

The riskiest changes are DTS consolidation and strided kernel layouts. Those should be benchmarked immediately after implementation because they touch the peak-performance path.
