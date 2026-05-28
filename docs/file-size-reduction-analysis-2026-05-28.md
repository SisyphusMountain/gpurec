# File Size Reduction Analysis

Date: 2026-05-28

This analysis was produced from local repository metrics plus four focused
`gpt-5.4-mini` explorer passes over independent slices:

- CLI and workflow: `gpurec/cli.py`, `gpurec/workflow/optimize.py`,
  `gpurec/workflow/config.py`.
- API and core runtime: `gpurec/api/*`, `gpurec/core/*`, retained kernel host
  wrappers.
- Optimization, native bridges, and entropy helpers.
- `scripts/` and `profiling/`.

The target is not shorter code by compression. The target is less code because
there are fewer duplicate paths, fewer local exception wrappers, clearer
ownership boundaries, and one retained production implementation per behavior.

## Baseline

Current Python footprint:

| Area | Python files | Python lines |
| --- | ---: | ---: |
| `gpurec/` | package files | 33,595 |
| `scripts/` | script files | 15,666 |
| `profiling/` | profiling files | 3,124 |
| `gpurec/` + `scripts/` + `profiling/` | 82 files | 52,385 |
| `tests/` | test files | 44,125 |

Other local signals:

- `try` / `except` occurrences in active source/tooling: 408.
- `os.environ`, `getenv`, or `GPUREC_` occurrences in active source/tooling:
  29.
- Largest structural item found by AST: `OptimizationRunner` in
  `gpurec/workflow/optimize.py`, about 4,287 class lines.
- Biggest single functions include `OptimizationRunner.run`, `main` in
  `gpurec/cli.py`, `Pi_wave_backward`, and `Pi_wave_forward`.

## Summary Of Reduction Potential

| Slice | Current hotspot footprint | Quality-preserving net reduction | Confidence |
| --- | ---: | ---: | --- |
| CLI and workflow | 11,151 lines across `cli.py`, `optimize.py`, `config.py` | 700-1,000 lines | high |
| Scripts and profiling | 12,089 lines across the top 10 files inspected | 1,400-2,100 lines | medium-high |
| Optimization/native bridge/entropy | 5,956 lines in focus files, plus `projected_lbfgs.py` overlap | 250-450 package lines | medium |
| API and core runtime | about 9,477 lines in inspected runtime files | 230-340 package lines | medium |
| First-pass realistic total | active source/tooling | 2,600-3,800 lines | medium |

The immediate win is outside the CUDA kernels. The safest path is:

1. Collapse CLI/workflow duplication.
2. Shrink or quarantine HOGENOM-only scripts.
3. Extract native bridge helpers shared by preprocessing and backtracking.
4. Touch API/core runtime only where the duplication is orchestration, not
   kernel math.

## File-Level Findings

### CLI And Workflow

| File | Lines | Main source of size | Reduction move | Estimate |
| --- | ---: | --- | --- | ---: |
| `gpurec/workflow/optimize.py` | 5,553 | large `OptimizationRunner`, manual result projection, manifest/final artifact assembly | split workflow orchestration, solver phase policy, checkpoint/final artifact writing; share summary/result schema | 300-450 |
| `gpurec/cli.py` | 4,003 | hand-written argparse schema, repeated subparser wiring, route-audit formatting, long result renderer | declarative parser spec; shared option groups; central route diagnostics; schema-driven result rendering | 300-450 |
| `gpurec/workflow/config.py` | 1,595 | long `RunConfig.__post_init__`, `validate`, route-contract helpers | table-drive normalization; keep special policy checks explicit; expose one route-audit source of truth | 100-180 |

Concrete hotspots:

- `gpurec/cli.py::_add_run_config_args` is mostly repeated
  `parser.add_argument(...)` scaffolding.
- `gpurec/cli.py::build_parser` repeats subparser patterns and common command
  options.
- `gpurec/cli.py` route-audit helpers duplicate policy already represented in
  workflow config and route metadata.
- `gpurec/workflow/optimize.py::OptimizationResult` and
  `_optimization_result_from_summary` manually mirror summary JSON fields.
- `gpurec/workflow/optimize.py::_build_run_manifest` and
  `_write_final_artifacts` are repetitive data assembly code.
- `gpurec/workflow/config.py::RunConfig.__post_init__` and `validate` mix
  mechanical normalization with important policy.

Quality-preserving approach:

- Keep CLI behavior, flag names, help text, and exit semantics stable.
- Replace repeated command wiring with a small declarative argument table.
- Move route evidence and route gate messages to one helper owned by workflow
  policy.
- Share one summary field schema between result parsing, text rendering, and
  manifest assembly.

Verification gates:

- Snapshot `gpurec --help` and subcommand help for every public command.
- Test parse/default behavior for required and optional run-config flags.
- Cover route gates for global, specieswise, and genewise modes.
- Compare text and JSON output for `optimize`, `run`, `summary-info`, and
  `checkpoint-info` on representative summaries, including missing fields and
  `NaN`/`inf` cases.

### Scripts And Profiling

| File | Lines | Ownership | Reduction move | Estimate |
| --- | ---: | --- | --- | ---: |
| `scripts/hogenom_ccp_wandb_opt.py` | 2,088 | legacy full HOGENOM launcher | move repeated optimizer/checkpoint/logging plumbing into shared HOGENOM helper or reduce to maintained wrapper | 300-500 |
| `scripts/check_release_metadata.py` | 1,691 | release hygiene gate | externalize long catalogs and phrase checks while preserving strict failures | 80-140 |
| `profiling/bench_uniform_forward_backward_pipeline.py` | 1,458 | retained profiling harness | share resource reporting, progress JSON, CUDA/NVTX helpers | 120-220 |
| `scripts/optimize_hogenom_ccp_global_uniform.py` | 1,230 | fixed-dataset legacy reproducer | reuse HOGENOM optimizer helpers for regularization, summaries, IO, objective setup | 220-380 |
| `scripts/fast_optimize_hogenom_ccp.py` | 1,210 | legacy fast reproducer | share optimizer/evaluation/reporting helpers with global launcher | 220-380 |
| `scripts/visualize_hogenom_loss_landscape.py` | 1,034 | analysis-only visualizer | factor generic CLI, CSV/JSON writing, block evaluation scaffolding | 70-140 |
| `scripts/benchmark_hogenom_specieswise_e2e.py` | 898 | route benchmark evidence | externalize route/stage tables; share checkpoint-history parsing and report assembly | 100-180 |
| `scripts/benchmark_hogenom_gradient_convergence.py` | 865 | gradient-convergence benchmark | share checkpoint/model setup, projected-gradient helpers, CSV/JSON writers | 120-200 |
| `scripts/benchmark_hogenom_specieswise_pulses.py` | 864 | pulse-search benchmark | share checkpoint/model setup and candidate serialization | 100-180 |
| `scripts/benchmark_hogenom_specieswise_multifidelity_adagrad.py` | 751 | multifidelity route benchmark | share solver phase/checkpoint/history writing boilerplate | 80-150 |

The script reduction should preserve reproducibility, not erase evidence. The
right move is to keep a small number of maintained entry points and move common
HOGENOM plumbing into `scripts/hogenom_opt_helpers.py` or another shared script
helper.

Quality-preserving approach:

- Classify each script as supported tool, release check, benchmark evidence,
  legacy reproducer, or deletion candidate.
- Preserve output schemas for CSV, JSONL, checkpoint payloads, `run_config.json`,
  W&B field names, and profiling event names.
- Delete only obsolete aliases or scripts whose results are already captured in
  docs and no longer reproduce a current gate.
- Prefer parameterized maintained scripts over several copied HOGENOM launchers.

Verification gates:

- Re-run one known HOGENOM checkpoint or route after helper extraction.
- Compare final loss, selected checkpoints, rate summaries, and saved artifacts.
- For profiling, compare event names and JSON-line schema, not just runtime.
- For release metadata checks, keep failure messages strict and stable.

### Optimization, Native Bridges, And Entropy

| File | Lines | Main source of size | Reduction move | Estimate |
| --- | ---: | --- | --- | ---: |
| `gpurec/optimization/lbfgsb.py` | 1,593 | repeated bounded-projection helpers plus large fallback/search layer | share bounded optimization primitives with `projected_lbfgs.py`; simplify fallback competition only after parity tests | 200-320 file lines |
| `gpurec/optimization/batched_lbfgs.py` | 993 | batched copy of projection/evaluation helpers plus large strong-Wolfe flow | share bound/projection/evaluation utilities; leave row-wise strong-Wolfe explicit until tested | 120-220 file lines |
| `gpurec/optimization/projected_lbfgs.py` | 434 | duplicate bounded helper scaffolding | reuse shared bounded optimization helpers | 60-100 file lines |
| `gpurec/backtracking.py` | 1,247 | Rust bridge skeleton and repeated `sample_*` dispatch/output validation | share native artifact bridge with preprocessing; table-drive sampling dispatch | 130-210 |
| `gpurec/core/preprocess_rust.py` | 710 | Rust bridge skeleton similar to backtracking | shared native bridge helper for cargo fallback, manifest version, library loading, command errors | 90-150 |
| `gpurec/entropy.py` | 725 | similar fixed-point loops and verbose payload validation | share convergence loop scaffolding; table-drive payload field validation where safe | 60-110 |
| `gpurec/optimization/lbfgsb_schilling.py` | 688 | literal conformance-style port | leave mostly alone | 0-20 |

The net package reduction here is smaller than the per-file cuts because shared
helpers still need to exist. A realistic net is 250-450 lines, with better
quality from less duplicated solver and native-bridge behavior.

Quality-preserving approach:

- Do not rewrite optimizer math for style. Extract only repeated bound,
  projection, evaluation, and Armijo helpers with clear tests.
- Treat `lbfgsb_schilling.py` as a conformance port and avoid clever
  compaction.
- Share Rust native-artifact bridge behavior across preprocessing and
  backtracking so diagnostics and env-var overrides do not drift.
- Keep entropy recurrence logic explicit unless golden outputs make a shared
  loop safe.

Verification gates:

- Optimizer parity on scalar and batched bound-touching cases.
- Armijo and strong-Wolfe acceptance behavior before/after helper extraction.
- Native preprocessing/backtracking checks with env-var overrides, missing
  artifacts, valid artifacts, and cargo fallback.
- Entropy golden outputs for representative solved models, including degenerate
  and missing-data cases.

### API And Core Runtime

| File | Lines | Main source of size | Reduction move | Estimate |
| --- | ---: | --- | --- | ---: |
| `gpurec/api/model.py` | 2,695 | constructor setup, resident-batch planning, duplicated constructors, mirrored full/batch/genewise paths | shared constructor normalization, theta init, static-state assembly, per-batch accumulation helpers | 90-140 |
| `gpurec/api/uniform_chunked.py` | 1,312 | heavy constructor doing preprocessing, policy, state build, repeated `from_*` entry points | split dataset/state preparation from model assembly; centralize `from_*` plumbing | 30-55 |
| `gpurec/api/_uniform_evaluator.py` | 228 | thin variants around solve/NLL boundary | merge resident no-grad and solved-E variants behind one helper | 15-25 |
| `gpurec/core/forward.py` | 710 | `Pi_wave_forward` setup for batched/shared params and output selection | extract setup helpers; keep loop and kernel launch contract explicit | 40-70 |
| `gpurec/core/backward.py` | 669 | wrapper-level layout normalization | share argument/layout normalization with forward-side layout code | 20-40 |
| `gpurec/core/kernels/wave_backward.py` | 2,173 | mostly intentional Triton specializations, plus wrapper layout glue | trim wrapper plumbing only; do not compact kernel bodies casually | 20-50 |
| `gpurec/core/kernels/wave_step.py` | 686 | repeated launch-arg normalization | share const-layout/block-size setup | 10-25 |
| `gpurec/core/batching.py` | 444 | verbose `collate_gene_families` accumulator plumbing | small collector helper for GE2/EQ1 parts and length checks | 20-35 |
| `gpurec/core/likelihood.py` | 560 | repeated shape validation and parameter alignment | extract param-alignment and shape-validation helpers | 20-35 |

This slice is not where the first large reduction should come from. The kernels
are dense because they encode specialized CUDA/Triton behavior. The code to
shrink first is orchestration around constructors, layout setup, and repeated
validation.

Quality-preserving approach:

- Keep kernel bodies explicit unless a benchmark and parity test proves the
  refactor is harmless.
- Consolidate constructor and state-building paths before changing evaluation
  semantics.
- Use `ParameterLayout` and typed evaluator requests to remove repeated mode
  inference gradually.
- Do not merge resident and chunked paths until public parity tests are easy to
  run.

Verification gates:

- Exact-output regression for global, specieswise, and genewise fixtures.
- Gradient parity for resident and chunked APIs.
- CUDA smoke for `Pi_wave_forward`, `Pi_wave_backward`, `loss_and_grad`,
  `nll_per_family`, and `trace_root_logsumexp`.
- Explicit branch coverage for `family_idx`, prepared shared constants,
  pruning, `initial_v_pi`, `chunk_indices`, and lazy prefetch behavior.

## Recommended Execution Order

1. Add a reproducible code-size report command.
   This makes every cleanup measurable and keeps the line budget visible.

2. Centralize CLI/workflow route diagnostics.
   This directly addresses exception-handling sprawl and reduces duplicated
   remediation text.

3. Convert CLI argument wiring to declarative specs.
   This is a low-risk way to shrink `gpurec/cli.py` while preserving public
   behavior.

4. Classify and shrink HOGENOM scripts.
   This gives the largest immediate active-source reduction without touching
   production runtime math.

5. Extract the shared Rust native bridge.
   This improves diagnostics and reduces duplicated artifact setup behavior.

6. Refactor optimizer helper duplication.
   Do this only with focused optimizer parity cases already in place.

7. Trim API/core orchestration around constructors and layout helpers.
   Avoid large kernel refactors until the surrounding workflow is smaller and
   easier to verify.

## Non-Goals

- Do not shrink by making dense one-liners or hiding behavior behind magic
  metaprogramming.
- Do not delete tests only to improve the line count.
- Do not merge CUDA/Triton kernel code just because it looks long.
- Do not weaken diagnostics; move them to shared boundary helpers instead.
- Do not turn HOGENOM evidence scripts into unsupported mystery wrappers unless
  the reproduced output schema is preserved somewhere.

## First Backlog Items

| Priority | Item | Expected result |
| --- | --- | --- |
| P0 | Add `scripts/report_code_size.py` or equivalent release-check mode | reproducible baseline and review signal |
| P0 | Build one route diagnostic helper used by CLI/workflow summaries | fewer local exception and remediation blocks |
| P0 | Classify `scripts/` files in `scripts/README.md` with keep/quarantine/delete status | visible script ownership |
| P1 | Declarative parser spec for repeated CLI run-config flags | smaller CLI with stable public help |
| P1 | Shared summary/result schema for optimization output | less manual field drift |
| P1 | Shared native artifact bridge for preprocessing/backtracking | fewer duplicated Rust setup failures |
| P2 | Shared bounded-optimization helper module | less solver duplication with parity tests |
| P2 | Runtime constructor/static-state helper extraction | smaller API surface without kernel risk |

## Bottom Line

The codebase is large, but the reduction path is not random. The first safe pass
should target roughly 2,600-3,800 active Python lines, mostly by deleting
duplication and script sprawl. The production package can shrink more slowly by
centralizing diagnostics, native bridge setup, and optimizer scaffolding while
leaving numerically sensitive kernel and solver logic protected by parity tests.
