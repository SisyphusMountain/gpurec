# Production-Grade Slimdown Plan

Date: 2026-05-28

This plan changes how we work on `gpurec`: production quality must come from a
smaller, clearer implementation, not from adding defensive code around every
edge case. The goal is to keep the production roadmap moving while actively
reducing the amount of Python that maintainers must understand.

## Current Size Baseline

There is no top-level `src/` directory in this repository. For size tracking,
the source baseline is the active package plus maintained source-checkout
tooling:

| Area | Current Python lines | Role |
| --- | ---: | --- |
| `gpurec/` | 33,595 | production package |
| `scripts/` | 15,666 | source-checkout tools, experiments, release checks |
| `profiling/` | 3,124 | benchmark and profiling entry points |
| `gpurec/` + `scripts/` + `profiling/` | 52,385 | active Python source/tooling footprint |
| `tests/` | 44,125 | validation footprint, tracked separately |

Largest current hotspots:

| File | Lines | Initial intent |
| --- | ---: | --- |
| `gpurec/workflow/optimize.py` | 5,553 | split workflow policy from solver mechanics |
| `gpurec/cli.py` | 4,003 | keep command wiring thin; centralize diagnostics |
| `gpurec/api/model.py` | 2,695 | move runtime internals behind typed evaluator objects |
| `gpurec/core/kernels/wave_backward.py` | 2,173 | keep one retained backward path; move/delete experiments |
| `scripts/hogenom_ccp_wandb_opt.py` | 2,088 | quarantine or replace with documented workflow |
| `scripts/check_release_metadata.py` | 1,691 | split release checks by concern |
| `gpurec/workflow/config.py` | 1,595 | separate schema, defaults, validation, and examples |
| `gpurec/optimization/lbfgsb.py` | 1,593 | remove duplicate solver orchestration paths |

## Target Budgets

Budgets are review triggers, not arbitrary style goals. They force us to justify
complexity before it becomes permanent.

| Scope | Near-term budget | Release-candidate budget |
| --- | ---: | ---: |
| `gpurec/` | <= 30,000 lines | <= 25,000 lines |
| `scripts/` | <= 10,000 lines | <= 5,000 lines |
| `profiling/` | <= 2,000 lines | <= 1,500 lines |
| active source/tooling total | <= 42,000 lines | <= 32,000 lines |
| new production module | <= 600 lines normally | hard review above 900 lines |
| new production function | <= 80 lines normally | hard review above 150 lines |

Existing files above the budgets are not automatically wrong, but every commit
touching one should move it toward a smaller shape unless there is a documented
reason not to.

## Working Rules

1. Delete, merge, or quarantine before adding.
   Every production-grade item starts with the question: which existing path can
   this replace? A feature that only adds a parallel path is incomplete until the
   old path is removed, deprecated with a removal date, or moved out of the
   product surface.

2. Keep one supported runtime path per behavior.
   Experimental CUDA kernels, benchmark-only selectors, HOGENOM research
   scripts, and local profiling helpers should not live in the production
   execution path. They belong in `profiling/`, `scripts/legacy/`, docs, or
   deletion.

3. Put exception handling at product boundaries.
   CLI and workflow entry points should translate failures into clear user
   diagnostics. Core and API code should validate contracts early, raise typed
   exceptions or compact standard exceptions, and avoid broad `try`/`except`
   blocks whose only purpose is to restate context.

4. Centralize remediation text.
   User-facing suggestions should come from shared helpers or error types. Do
   not duplicate long remediation strings across commands.

5. Prefer typed request/result objects over branchy option sets.
   If a function has several boolean flags, optional outputs, and mode-specific
   branches, split it by intent or pass a request object that makes the contract
   explicit.

6. Treat environment-variable runtime selectors as temporary.
   New `GPUREC_*` selectors need an owner, a reason, and an expiry condition.
   Once a path is retained, delete the selector and the losing implementation.

7. Track net code size in each production commit.
   Commits that add production code should state the line impact for `gpurec/`,
   `scripts/`, and `profiling/`, plus what was removed or consolidated.

## Exception-Handling Policy

The codebase should not grow by wrapping every call site in local recovery code.
Use this ownership model instead:

| Layer | Responsibility | Avoid |
| --- | --- | --- |
| CLI commands | catch expected user/setup failures and print concise remediation | repeated long `try`/`except` blocks per subcommand |
| workflow layer | validate run state, checkpoint paths, output directories, native artifacts | catching generic exceptions from deep internals and guessing cause |
| public Python API | raise stable typed errors for contract violations | printing, exiting, or formatting CLI prose |
| core kernels/runtime | assert internal invariants and fail loudly with compact context | broad recovery paths, hidden fallbacks, duplicate mode inference |
| scripts/profiling | can be rougher, but must not define production behavior | product-only diagnostics buried in research scripts |

When a new failure mode needs a better message, first add or reuse a shared
diagnostic helper. Only add local exception handling when the local frame has
unique information that cannot be carried by the exception itself.

## Workstreams

### 1. Freeze The Product Boundary

Decide which Python files are product, maintained tooling, benchmark-only,
legacy research, or deletion candidates.

Actions:

- Mark supported CLI, workflow, API, and native-check paths as product code.
- Mark HOGENOM-only scripts and exploratory profiling as checkout-local tools.
- Move obsolete research scripts to `scripts/legacy/` or delete them when a
  documented workflow replaces them.
- Keep `docs/README.md` aligned so users can distinguish stable workflows from
  internal notes.

Acceptance gates:

- Every file under `scripts/` has one of: supported tool, release check,
  benchmark, legacy, or deletion candidate.
- No user-facing documentation points at a legacy script as the recommended
  path.
- `scripts/` drops below 10,000 lines before the release-candidate push.

### 2. Stop New Growth

Add lightweight maintainability gates before larger refactors begin.

Actions:

- Add a code-size report command that prints the same folder totals used in this
  document.
- Add a soft CI/reporting gate for files above budget and net positive source
  growth.
- Add a short PR/commit checklist: public behavior, tests, docs, net line
  impact, deletion or consolidation performed.
- Refuse new production fallbacks unless they replace an existing path or have a
  documented removal condition.

Acceptance gates:

- Code-size numbers are reproducible with one command.
- New large files or functions are visible in review.
- Production commits stop increasing `gpurec/` without an explicit reason.

### 3. Collapse CLI And Workflow Diagnostics

The CLI should wire commands and format boundary errors. It should not own
parallel remediation logic for every command.

Actions:

- Continue extracting shared diagnostic helpers from `gpurec/cli.py`.
- Move command-specific validation into workflow/preflight functions that return
  structured results.
- Keep CLI handlers mostly as parse, call, render, exit.
- Consolidate checkpoint, config, native-artifact, and CUDA-readiness failures
  into a small set of error types.
- Split `gpurec/workflow/optimize.py` into orchestration, solver phase policy,
  checkpoint writing, and final artifact validation.

Acceptance gates:

- `gpurec/cli.py` is below 2,500 lines.
- `gpurec/workflow/optimize.py` is below 3,500 lines.
- User-facing failure output remains covered by focused tests.
- There is no duplicated multi-line remediation text in CLI commands.

### 4. Consolidate Evaluation Internals

This is the high-risk, high-payoff simplification. It should build on the
existing evaluator and simplification plans instead of inventing a new
architecture.

Actions:

- Define one `ParameterLayout` for global, specieswise, and genewise theta
  semantics.
- Convert theta into a `UniformRates` object once per evaluation.
- Route resident, chunked, loss-only, gradient, and export-state calls through
  one typed evaluator contract where practical.
- Split forward calls by output intent: gradient training, root-row inference,
  and export state.
- Retain one production Pi backward path and delete or quarantine old native,
  prototype, and env-selected alternatives after benchmark evidence.

Acceptance gates:

- Global, specieswise, and genewise modes have parity tests before and after the
  refactor.
- Runtime selectors that no longer select retained production behavior are
  deleted.
- `gpurec/api/model.py` and `gpurec/core/kernels/wave_backward.py` shrink or
  move complexity into smaller single-purpose modules without expanding total
  package lines.

### 5. Shrink Scripts And Profiling

The largest immediate reduction is outside the package. This should happen
before deep kernel refactors because it lowers repository noise quickly.

Actions:

- Keep only supported release checks and documented workflow helpers in
  top-level `scripts/`.
- Replace duplicated HOGENOM optimization scripts with one parameterized
  maintained entry point or documented config path.
- Delete scripts whose results are already captured in docs and are not needed
  to reproduce current validation.
- Keep profiling entry points only when they have a named retained performance
  gate.

Acceptance gates:

- `scripts/` below 10,000 lines in the first cleanup pass.
- `profiling/` below 2,000 lines in the first cleanup pass.
- Any deleted script has either no current caller or a documented replacement.

### 6. Keep Tests Useful, Then Make Them Smaller

Tests are allowed to be large when they protect dangerous behavior, but repeated
fixtures and copy-pasted CLI cases slow maintenance.

Actions:

- Keep coverage around public CLI behavior, config validation, native artifact
  diagnostics, checkpoint semantics, and CUDA correctness gates.
- Replace repeated fixture setup with builders and table-driven cases.
- Move long golden data into fixture files instead of inline Python literals
  when that improves readability.
- Avoid deleting tests only to hit a line budget; reduce duplication first.

Acceptance gates:

- Production-code reductions land with focused parity or regression tests.
- Test LOC can shrink, but coverage of public behavior does not.
- Large tests explain the behavior they protect.

## First Implementation Milestones

1. Add the code-size reporting command and record the baseline in CI or release
   checks.
2. Classify `scripts/` and delete or quarantine the easiest unsupported
   HOGENOM-only duplicates.
3. Finish centralizing CLI remediation helpers so exception-handling code stops
   spreading.
4. Split `gpurec/workflow/optimize.py` around clear responsibilities without
   changing behavior.
5. Start the `ParameterLayout` work only after the CLI/workflow surface is
   smaller and the parity tests are easy to run.

## Done Criteria

The slimdown is working when all of these are true:

- The active source/tooling footprint is trending down toward 32,000 lines.
- Production features no longer add new fallback stacks by default.
- User-facing errors are clearer because diagnostics are centralized, not
  because every call site catches and rewrites exceptions.
- Large modules have named responsibilities and measurable reduction targets.
- The public production roadmap can advance without making the repository
  harder to inspect, test, and change.
