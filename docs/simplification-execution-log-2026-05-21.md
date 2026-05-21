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

## Active Work Queue

1. `MODE-01`, `ORIG-01`, `VALID-01`: introduce explicit layout/origination
   contracts only after characterization tests are committed.
2. `LIK-01`: standardize internal likelihood on root rows after layout and
   characterization coverage are in place.
3. `EVAL-01` and `CHUNK-01`: consolidate resident, no-grad, export, autograd,
   and chunked evaluation paths behind one evaluator after the gates above.
4. `PI-01`, `MODE-02`, `BWD-03`, and `DTS-01`: refactor Pi/backward/DTS
   contracts only after the explicit layout contract exists.
5. `BWD-01`, `BWD-02`, `ENV-01`, and `SCHED-01`: remove runtime alternatives
   only after benchmark gates show the retained path is not regressed.

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
