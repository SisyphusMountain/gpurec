# Backlog extracted from the review of `main@358a5b80`

The original review is archived under
`archive/internal-reviews/2026-08-11-main-358a5b80/`. This list contains only
findings that were still applicable when checked against `dev` on 2026-09-01.

## User-facing correctness and documentation

- Resolve `starting_gene_tree =` paths relative to their manifest and add a
  regression test.
- Decide whether `mapping =` is supported. Either implement it or reject it
  clearly instead of leaving a plausible but unused field.
- Link `docs/cli.md`, `docs/input-contract.md`, papers, and benchmarks from the
  root README.
- Add CI for tests, Rust builds, documentation links, and maintained LaTeX.
- Add a class/constructor docstring to `gpurec.api.model.GeneReconModel`.

## Configuration and layering

- Remove or justify the lower-level `neumann_terms=3` default in
  `gpurec/core/kernels/wave_backward.py`; public configuration uses 64.
- Finish single-sourcing solver, rate-bound, memory, and penalty settings.
- Reconsider the intended dependency direction
  `config -> core -> solver -> api -> fit -> cli`. `SolverOptions` remains in
  `api`, `PenaltyOptions` remains in `solver`, and lower layers import private
  `api._*` modules.
- Reconcile the owner's “one configuration source/no silent clamping” rule with
  the public projection helpers and document the chosen contract.

## Robustness and maintainability

- Replace user-triggerable Rust `.unwrap()`/`.expect()` paths with contextual
  errors in the preprocessing and backtracking crates.
- Deduplicate repeated setup in `wave_backward_kernels.py` and repeated paths
  in `_execution.py`, guarded by numerical regression tests.
- Split oversized Rust parser/scheduler and Pi/wave kernel modules only after
  their numerical and integration test surfaces are explicit.
- Classify research/test-only solver modules instead of deleting them based on
  the archived review's obsolete “986 dead lines” count.

The archived findings about absent tests, absent documentation, and dangling
notebook/doc targets are not copied here because `dev` already restores them.
