# `gpurec/fit` cleanup plan

Status: proposed

Audit date: 2026-07-14

Audited tree: `dev` at `1605fc5d`
Scope read: every Python source file under `gpurec/fit` (10 files, 2,132 lines), plus its CLI,
configuration, test, gate, experiment, and documentation consumers.

## Executive recommendation

`gpurec/fit` currently contains two different products:

1. the supported rate-fitting path used by `gpurec fit` (`global` and `genewise`); and
2. a port of the kernel-bench optimization research environment (generic first-order drivers,
   several Newton variants, MAP experiments, CV smoke code, SciPy baselines, receiver/origination
   experiments, and benchmark-specific narratives).

They should not remain in the same importable package.

The recommended end state is:

- Keep one small, explicit production API for the two modes the CLI actually supports:
  `global` and `genewise`.
- Move specieswise MAP/CV work to `experiments/specieswise_fit/` until it has a stable product
  contract and a supported CLI. Do not advertise a direct library path from a CLI mode that always
  fails.
- Delete `map_fit.py` outright.
- Remove `baselines.py`, the generic `optimize.py` research pipeline, and the unused Newton variants
  from the production package after their test/gate consumers are either deleted or moved to test
  helpers.
- Keep finite-difference and HVP validation utilities in tests, not in the production fit API.
- Replace long experiment reports in module docstrings with short contracts. Preserve durable
  rationale in developer documentation and historical results in the experiment/archive area.
- Normalize options, results, validation, logging, dtype handling, and input resolution across the
  retained global/genewise recipes.

The first boundary cut removes or relocates 1,539 of the current 2,132 lines (72%) from the
production package. The four currently relevant files (`__init__.py`, `dtl_fit.py`,
`global_fit.py`, and `genewise_fit.py`) contain 593 lines before their own simplification.

This is not a recommendation to delete the underlying specieswise or second-order research without
preserving reproducibility. It is a recommendation to stop presenting that research as supported
application code merely because tests and one-off gates still import it.

## Current reachability

The actual product path is small:

```text
gpurec CLI
  -> gpurec.cli.fit.run_fit
     -> gpurec.fit.dtl_fit.fit_dtl
        -> mode=global   -> fit_global
        -> mode=genewise -> fit_genewise
        -> mode=specieswise -> NotImplementedError before model construction
```

The advanced specieswise path is separate:

```text
direct caller / regression mint / CV experiment
  -> fit_specieswise
     -> Schedule + make_value_and_grad
     -> newton_lanczos
     -> final_eval

map_cv
  -> build a new model for every train/test fold
  -> fit_specieswise once per fold/lambda
  -> heldout_nll
  -> all-family fit_specieswise refit
```

Everything else in `gpurec/fit` is outside the supported CLI flow. Repository-wide import tracing
found:

- `map_fit.py` has no caller.
- `baselines.py` is used by one legacy gate and by `map_fit.py`.
- `newton_cg.newton_tr` is used only by an experiment.
- `newton_cg.newton_cg` has no real caller.
- `newton_cg._fd_hessian_hvp` is used as a test/gate oracle, not by a production fit.
- `optimize.first_order` is used by experiments, gates, and tests.
- `optimize.newton_polish` is used by experiments and a regression test.
- `optimize.ridge_anneal` is only called by the orphaned `map_fit.py`.
- `optimize.optimize` is retained by tests and historical comparisons, not by `fit_dtl`.
- `optimize.final_eval` and `Schedule` are the only pieces of `optimize.py` used by the current
  specieswise fitter.

This independently agrees with the earlier static manifest committed on
`chore/prune-dead-optimization-code` (`cd42c8e1`), but this plan is based on the current `dev` tree
and current repository references rather than assuming that manifest is still authoritative.

## File-by-file disposition

| Current file | Current role | Recommendation | Reason |
|---|---|---|---|
| `__init__.py` | Re-exports only `fit_genewise` | Rewrite | It hides the canonical dispatcher and gives an arbitrary public surface. |
| `dtl_fit.py` | CLI dispatcher | Keep and simplify | This is the true product entry point, but it accepts ignored options and over-documents internal recipe history. |
| `global_fit.py` | Supported shared-rate recipe | Keep and refactor | Used by CLI; currently duplicates genewise optimizer mechanics and ignores `solver_options`. |
| `genewise_fit.py` | Supported per-family recipe | Keep and refactor | Used by CLI and exported at package root; currently combines path parsing, configuration resolution, model construction, optimization, rebatching, certification, environment mutation, and reporting. |
| `specieswise_fit.py` | Direct-only MAP fitter | Move to `experiments/specieswise_fit/` | The CLI explicitly rejects this mode. Its API is low-level (`batch_statics`) and fp32-specific, and its scientific contract is not stable enough to advertise as a supported fit. |
| `map_cv.py` | Direct-only specieswise CV harness | Move with specieswise experiments | It has a hard-coded smoke driver, incomplete validation, and no supported CLI. It is an experiment orchestration layer rather than a core fitting primitive. |
| `map_fit.py` | Older end-to-end MAP experiment | Delete | Zero callers; superseded by `fit_specieswise`; mutates an environment variable at import time; advertises a nonexistent CLI/module path. |
| `baselines.py` | Kernel-bench GD/SciPy baselines | Delete from package | No production caller. Move the one useful L-BFGS comparison into a benchmark/experiment if it still needs to run. |
| `newton_cg.py` | Four generations of Newton/HVP outer loops | Remove from `fit`; retain only an explicitly supported solver if specieswise is later promoted | Three symbols are dead in production; the remaining `newton_lanczos` serves only specieswise. Test FD helpers belong in tests. |
| `optimize.py` | Generic research optimizer and several polish variants | Remove from package after migration | No supported fit dispatch uses it. Its broad, polymorphic interface and stale compatibility modes create more apparent API than maintained functionality. |

## Detailed findings

### 1. The public surface contradicts itself

- `gpurec.fit.__init__` exports only `fit_genewise`, although `fit_dtl` is described as the one
  canonical mode-aware entry point.
- `gpurec.__init__` also exports `fit_genewise` directly, bypassing `fit_dtl` and its mode policy.
- `fit_dtl` accepts `specieswise` as a valid mode but always raises. The CLI advertises
  `--mode specieswise`, then exits with an error telling the user to use low-level Python APIs.
- `fit_dtl.max_steps` is never used by either reachable mode. The CLI exposes it as `--steps` and
  labels it “unused”. An unused option should not be part of a user interface.
- `fit_dtl.init_rate` is meaningful only for global fitting and deliberately ignored for genewise.
- `fit_dtl` says `solver_options` overrides the E/adjoint solver, but `fit_global` accepts and never
  reads that argument. A CLI user can supply solver flags that silently do not affect a global fit.
- Result dictionaries differ by mode. Genewise nests a second implementation-specific result dict;
  global and specieswise return different fields; devices/dtypes differ; `gnorm` does not have one
  consistent definition.

### 2. Production recipes duplicate mechanics

`fit_global` was adapted from `fit_genewise`, but the shared code was copied rather than extracted.
Both now contain variants of:

- Adam basin-entry steps;
- rate-bound projection;
- a three-column finite-difference Hessian;
- Hessian symmetrization and eigenvalue flooring;
- active-bound masking;
- a trust-radius cap;
- solver-tier construction;
- final high-tier evaluation.

They already drift:

- global exposes `hess_every`; genewise hard-codes `5`;
- global has a loss-plateau stop; genewise has per-family convergence/rebatching logic;
- solver-option precedence differs;
- global hard-codes the genewise rate bounds but does not consume `config.rates` consistently;
- final reporting and certification differ.

Extract only the genuinely shared 3x3 operations into a private module. Do not create another
general optimizer framework. The rebatching policy should remain genewise-specific and aggregate
loss/gradient handling should remain global-specific.

### 3. Configuration precedence is too subtle to be safe

The code currently uses “if the argument still equals its signature default, treat it as unset”.
That makes it impossible to distinguish:

- a caller who omitted a value; and
- a caller who explicitly selected the same numeric value as the preset.

The resulting docstrings need several paragraphs to explain precedence. Configuration should not
need prose this defensive.

Specific problems:

- `fit_genewise` sometimes uses `GpurecConfig.genewise_reference()`, sometimes explicit kwargs, and
  sometimes a partial `solver_options` dict.
- Passing a non-default `GpurecConfig` replaces tuned recipe defaults wholesale, even when the
  caller intended to change one field.
- `fit_global` receives `config`, constructs its own solver tiers, and ignores its
  `solver_options` parameter.
- `map_cv` documents explicit `solver_options` precedence, but constructs
  `SolverOptions(**(solver_options or so_base))`; a `SolverOptions` instance is not a mapping and is
  not handled like it is in `fit_genewise`.
- `GpurecConfig.optimize_reference()` exists only to say that it equals the default config.
- Reference dictionaries duplicate function signature defaults and require tests that inspect
  signatures rather than behavior.

Replace this with typed per-recipe option dataclasses and a single merge function. Use an explicit
unset sentinel or require callers to provide an options object; never infer “unset” from equality.

### 4. Input resolution is in the wrong module

`genewise_fit._resolve_gene_trees` is used by:

- genewise fitting;
- global fitting; and
- CLI common helpers.

It is not genewise functionality. It also has avoidable edge cases:

- a `pathlib.Path` is treated as a generic iterable instead of a path;
- file handles are opened without a context manager;
- a directory scan is shallow and format-specific;
- listfile parsing mixes AleRax parsing and one-path-per-line heuristics;
- library fitters accept globs/directories/listfiles even though those are CLI/input-layer concerns.

Move resolution to an input module such as `gpurec/io/gene_trees.py`. The Python fitting API should
prefer an explicit sequence of resolved paths; the CLI may provide convenience expansion.

### 5. Global environment state is used as an optimizer option

`fit_genewise` repeatedly sets, removes, and restores `GPUREC_WARM_ADJOINT` to control internal
caching. This makes concurrent fits interfere with each other and forces certification to mutate
process state temporarily.

`map_fit.py` is worse: importing it executes
`os.environ.setdefault("NEWTON_TANGENT_SELF_ITERS", "64")`.

Warm-adjoint and tangent iteration policy should be explicit solver/runtime state. If the lower
layer cannot be changed immediately, contain the temporary environment compatibility in one
well-tested context manager rather than open-coding nested save/pop/restore sequences in a fitter.

### 6. Dtype and device behavior is inconsistent

- `fit_genewise` starts from the configured model dtype.
- `fit_global` follows the constructed model dtype.
- `fit_specieswise` unconditionally converts theta and `theta_ref` to fp32.
- `first_order` unconditionally converts theta and optimized tail logits to fp32.
- `final_eval` claims to provide a “fair fp64” evaluation by passing a double theta into statics
  built elsewhere, but the contract does not state which intermediates or accumulators actually
  change dtype.
- Several functions call `torch.cuda.empty_cache()` directly despite accepting a generic `device`.

Every retained API should either preserve the model/parameter dtype or reject unsupported dtypes
explicitly. Precision guarantees should be tested and stated narrowly.

### 7. The generic optimizer interface has accumulated unrelated experiments

`optimize.first_order` supports:

- six torch optimizers;
- four learning-rate schedules;
- optional best-iterate return;
- optional early stopping;
- theta-only fitting;
- theta plus receiver logits;
- theta plus origination logits;
- TV penalties;
- origination penalties;
- group indexing; and
- polymorphic return shapes.

No supported fit path uses this generality. Tests and experiments keep it alive. It also has weak
failure behavior: unknown optimizer names raise a raw `KeyError`; an unknown schedule kind silently
acts like a constant schedule; an unrecognized `polish_mode` falls through to a Lanczos polish;
`return_best` does not evaluate the final post-step iterate.

This is a research harness, not a production fit primitive. Move any experiment that still needs it
out of `gpurec`, then remove it rather than polishing its broad API.

### 8. `newton_cg.py` contains multiple abandoned solver generations

The module title describes analytic GGN/Fisher Newton-CG, while its currently relevant
`newton_lanczos` path defaults to a finite-difference true Hessian. It also contains:

- a receiver-weight delegation branch not reached by any production fit;
- a trust-region Newton variant used by one experiment;
- an older LM Newton-CG variant with no caller;
- shape assumptions specialized to specieswise `[S,3]` theta;
- inconsistent multi-batch saved-state handling;
- line-search finite-value guards present in one solver but absent in others;
- approximate evaluation counters presented as if exact;
- extensive device-memory workarounds tied to particular historical fixtures.

If specieswise fitting remains experimental, move the one solver it needs with the experiment.
If specieswise becomes supported later, design one solver contract in `gpurec/solver`, validate it
for multi-batch execution, and delete the alternatives.

### 9. MAP/CV modules are experiments presented as APIs

`map_fit.py` is unambiguously obsolete:

- no caller imports it;
- it names old `newton/...` documentation and a nonexistent `python -m newton.specieswise_fit` CLI;
- it performs file output inside a library fitting function;
- it changes an environment variable during import;
- its generic `fit` name is ambiguous;
- it duplicates the newer specieswise workflow.

`map_cv.py` is more useful, but still experiment orchestration:

- its module has a `__main__` HOGENOM smoke test with a hard-coded absolute path;
- it permits a generic `mode` even though its theta construction and worker are specieswise;
- it does not validate `k`, `n`, empty folds, lambdas, or `init_rate`;
- it rebuilds full/train/test models repeatedly without an explicit lifecycle/memory policy;
- it averages fold totals rather than clearly reporting a per-family predictive score;
- it returns a GPU tensor in an otherwise mostly serializable dict;
- it uses printing rather than structured progress/reporting.

Retain the scientific workflow under `experiments/specieswise_fit/`, with commands and outputs
there. Promote it only when there is a tested CLI and a stable result schema.

### 10. Documentation is embedded at the wrong layer

Public docstrings currently include:

- kernel-bench provenance;
- task numbers and implementation-brief references;
- specific commits;
- benchmark sizes such as `666x80`;
- measured speedups and convergence claims;
- GPU memory observations tied to 24 GB cards;
- historical failed approaches;
- uppercase argumentative language (`ONLY`, `SAME`, `IMPORTANT`, `CERTIFIED`);
- instructions for editing defaults; and
- obsolete package paths.

This material is valuable during research but harmful in user-facing API documentation because it
ages independently of the function contract.

Use this documentation policy:

- Module docstring: one paragraph describing responsibility and supported status.
- Public function docstring: inputs, shapes, units, defaults, return type, errors, and important
  numerical guarantees that are enforced by tests.
- Inline comment: only a local invariant or non-obvious implementation constraint.
- Developer design document: algorithm choice, Hessian structure, solver-tier rationale, and memory
  strategy.
- Benchmark report: hardware, datasets, timings, convergence studies, and rejected alternatives.
- Historical plans/specs: archive; do not link from runtime APIs as normative documentation.

The existing `gpurec/docs/optim/README.md` also needs correction: it still describes a
`gpurec/optim/` package, references removed diagnostics, and gives the obsolete command
`python -m gpurec.optim.map_cv`.

## Proposed production structure

After the boundary cleanup, use a small package organized around the supported contract:

```text
gpurec/fit/
  __init__.py          # FitResult, fit_rates, fit_global, fit_genewise
  api.py               # mode dispatch and result normalization
  result.py            # typed FitResult and progress/history records
  options.py           # GlobalFitOptions, GenewiseFitOptions
  _three_rate.py       # private shared 3x3 FD-Hessian/active-bound operations
  global_fit.py        # aggregate-family policy
  genewise_fit.py      # per-family rebatching/tier-escalation policy
```

The exact filenames are less important than these boundaries:

- input expansion is outside `fit`;
- config resolution happens once at the API boundary;
- model construction is separate from optimizer math;
- shared numerical operations have one implementation;
- mode-specific convergence policies remain separate;
- experiments do not live in the importable production package.

### Proposed Python API

Prefer one explicit entry point:

```python
result = fit_rates(
    species_tree,
    gene_trees,                 # resolved Sequence[PathLike]
    mode="global",              # Literal["global", "genewise"]
    device="cuda",
    dtype=torch.float32,
    config=config,
    options=GlobalFitOptions(),
)
```

Return a typed result with a mode-independent core:

```text
FitResult
  mode
  theta_cpu
  rates_cpu                 # always D, L, T
  nll_bits
  nll_nats
  n_families
  elapsed_seconds
  convergence              # typed summary, not a mode-dependent meaning of gnorm
  history                  # optional structured records
```

Mode-specific diagnostics may be nested under a typed `details` field. Do not expose the entire
internal `fit_genewise` dictionary as `genewise_result`.

Compatibility options:

- If preserving external imports matters, keep `fit_dtl = fit_rates` and direct
  `fit_global`/`fit_genewise` aliases for one release with deprecation warnings.
- If the project is intentionally pre-release, make the clean break now and remove the aliases.
- In either case, `gpurec.__init__` and `gpurec.fit.__init__` must expose the same canonical fitting
  story.

## Phased implementation plan

### Phase 0: pin current supported behavior

Before deleting code, record a baseline from the current `dev` commit.

1. Capture global and genewise results on:
   - the smallest tracked CUDA fixture;
   - one multi-batch regression fixture; and
   - the existing 200-family simulated goldens.
2. Record theta/rates, NLL, projected-gradient statistics, family convergence counts, peak CUDA
   memory, and wall time.
3. Add a fast dispatch test proving exactly which modes the product supports.
4. Add tests that currently expose the two silent API bugs:
   - `fit_dtl(max_steps=...)` has no effect; and
   - global `solver_options` is ignored.
5. Decide the intended behavior, then change/remove the parameters rather than preserving the bugs.

Gate: no cleanup patch begins until current global/genewise outputs can be compared against a pinned
baseline.

### Phase 1: perform safe deletions

1. Delete `gpurec/fit/map_fit.py`.
2. Remove its mentions from `gpurec/docs/optim/README.md`, config docs, and historical “current
   package map” sections.
3. Delete `_smoke` and `if __name__ == "__main__"` from production `map_cv.py` even before the larger
   move; put any retained smoke command under experiments.
4. Remove committed/generated `__pycache__` artifacts if any are tracked and ensure ignore rules
   cover them.

Gate: import tests plus a repository search showing no live reference to `map_fit`,
`spectrum_min`, or `_deflate_step`.

### Phase 2: move specieswise research out of the supported package

1. Create `experiments/specieswise_fit/` with a README describing status, datasets, commands,
   expected outputs, and reproducibility limits.
2. Move the useful pieces of `specieswise_fit.py` and `map_cv.py` there.
3. Consolidate them with the existing `experiments/sanderson_cv/` scripts instead of creating two
   competing specieswise CV implementations.
4. Move specieswise recipe tests to experiment/integration coverage or convert them to lower-level
   solver tests.
5. Change `gpurec fit` choices to `global|genewise`. Do not accept a mode whose only outcome is an
   exception.
6. Update `docs/cli.md` to explain that specieswise likelihood evaluation remains possible if that
   is true, but specieswise fitting is experimental and not a supported `fit` mode.
7. Remove `GpurecConfig.map_cv_reference()` and CV lambda defaults from the core config unless another
   supported feature consumes them.

Gate: no production package or CLI imports from `experiments`, and the global/genewise CLI behavior
is unchanged.

If specieswise fitting is a near-term product requirement, use a stricter alternative to this
phase: keep it in `gpurec/fit/specieswise.py`, add a real `gpurec fit-specieswise` or explicit CLI
workflow, define typed options/results, and support it end to end. Do not retain the present middle
state.

### Phase 3: remove the research optimizer stack from `gpurec/fit`

Migrate consumers before deleting files:

1. Move `_fd_hessian_hvp` to `tests/helpers/finite_difference.py`; update HVP symmetry/parity tests
   and gates to import the test helper.
2. Replace `test_fit_global_matches_optimize` with a direct regression against the pinned global
   result or a converged reference. A dead optimizer is not a useful permanent oracle.
3. Rewrite origination/receiver tests to exercise `make_value_and_grad` or the corresponding solver
   primitive directly instead of going through `first_order`.
4. Move `_fit_kbench.py` and similar historical gates to an archive/benchmark area or delete them if
   their result is already covered by pinned regression tests.
5. Remove config-wiring and signature-default tests for `optimize`, `MAP_CV_REFERENCE`, and
   `OPTIMIZE_REFERENCE` together with those APIs.
6. Decide whether second-order/HVP regression tests are testing a production solver or research.
   Keep genuine kernel mathematics tests, but remove outer-loop recipe tests that no supported
   feature reaches.
7. Delete `baselines.py`, `optimize.py`, and `newton_cg.py` from `gpurec/fit` once no supported or
   mathematical correctness test imports them.

Gate: `rg "gpurec\.fit\.(optimize|newton_cg|baselines)" gpurec tests` returns no production import;
all retained HVP kernel tests pass with test-local reference helpers.

### Phase 4: normalize the retained API

1. Introduce `FitResult`, `GlobalFitOptions`, and `GenewiseFitOptions`.
2. Replace `fit_dtl`'s dict assembly with one normalized result path.
3. Remove `max_steps` from the common API/CLI or map it to a clearly defined per-mode option.
4. Make `init_rate` mode-specific; validate it as finite and positive.
5. Either honor `solver_options` consistently or remove it in favor of resolved fit options. Never
   accept and ignore it.
6. Define `gnorm` precisely or replace it with named fields such as
   `projected_grad_max_fit_tier` and `projected_grad_max_eval_tier`.
7. Return CPU tensors consistently and make the result directly serializable through an explicit
   method.
8. Validate all options up front: rate bounds, even Pi iterations, tier ordering, tolerances,
   iteration counts, trust radius, Hessian cadence, and drop fractions.
9. Replace direct `print` calls with `logging` or a progress callback; retain history as structured
   data.

Gate: API tests cover validation, result schema, dtype/device normalization, and CLI serialization.

### Phase 5: refactor global/genewise implementation

1. Move gene-tree expansion to the input layer and make fitters consume resolved paths.
2. Extract a private three-rate numerical helper containing:
   - finite-difference Hessian construction;
   - Hessian symmetrization/eigenvalue floor;
   - active-bound mask;
   - trust-capped solve; and
   - projected-gradient calculation.
3. Keep global aggregation and genewise rebatching in separate policy functions.
4. Replace environment mutation for warm adjoints with an explicit runtime option. As an interim
   step, use one context manager with guaranteed restoration.
5. Use one solver-tier builder and one rate-bound source.
6. Remove calls to private model methods such as `_theta_for_static` by adding the narrow public
   model/evaluation operation the fitter actually needs.
7. Use the central memory policy rather than scattered unconditional `torch.cuda.empty_cache()`
   calls.
8. Split genewise certification from fitting so callers do not pay for it implicitly and the
   result states which tier was evaluated.

Gate: global/genewise golden values remain within their established tolerances and neither mode
regresses materially in wall time or peak memory.

### Phase 6: clean documentation and configuration

1. Rewrite retained module/function docstrings to contract-only documentation.
2. Create one developer design note for the global/genewise recipes:
   - objective and rate order;
   - why global aggregates per-family 3x3 curvature;
   - why genewise permits independent 3x3 blocks and rebatching;
   - solver-tier and bound rationale;
   - convergence/certification definitions.
3. Move benchmark claims and fixture-specific tuning to a versioned benchmark report.
4. Correct or archive `gpurec/docs/optim/README.md` and related stale `gpurec/optim` paths.
5. Remove task-number, commit, external-repository, and machine-specific commentary from runtime
   modules.
6. Simplify `docs/config_convention.md` after replacing equality-based precedence.
7. Correct `pyproject.toml` dependency commentary. It currently says `map_cv.py` uses
   `scipy.optimize.minimize`, which is no longer true. Re-evaluate whether SciPy remains a core
   dependency; `gpurec/solver/cg.py` still uses `scipy.linalg.eigh_tridiagonal`, so removing fit
   baselines alone may not make it optional.

Gate: documentation links resolve, examples import only supported APIs, and `gpurec fit --help`
contains no knowingly unused or always-failing option.

### Phase 7: audit the downstream second-order solver tree

Moving specieswise fitting out of production may leave the exact-HVP/curvature tree without a
production caller. Do not delete it automatically as part of the fit cleanup. Run a separate
reachability and scientific-value review covering:

- `gpurec/solver/hvp_exact.py`;
- `gpurec/solver/forward_tangent.py`;
- `gpurec/solver/ggn.py`;
- `gpurec/solver/cg.py`;
- curvature modules; and
- tangent/second-order Triton kernels.

Classify each as production, mathematical test support, active research, or archival. This is a
separate deletion decision because those kernels may remain valuable even if no current fit recipe
uses them.

## Test migration matrix

| Current coverage | Change |
|---|---|
| `tests/regression/test_global_recipe.py` compares `fit_global` with generic `optimize` | Replace generic optimizer oracle with a pinned converged global reference and simulated-truth sanity checks. |
| `tests/regression/test_specieswise_recipe.py` | Move with the specieswise experiment or redesign as solver-level MAP/HVP tests. |
| `tests/regression/test_newton_multibatch.py` | Keep only if exact-HVP/Newton remains supported; otherwise move to research integration tests. |
| `tests/test_reference_defaults.py` | Delete signature-dict reflection tests; test typed option defaults and behavior instead. |
| `tests/test_config_wiring.py` optimizer/MAP-CV sections | Remove with those APIs; retain focused global/genewise option-resolution tests. |
| `tests/test_genewise_hvp.py`, `tests/test_hvp_multibatch.py` importing `_fd_hessian_hvp` | Use a test-local finite-difference oracle. |
| `tests/test_origination_weights.py` importing `first_order` | Test objective/gradient wiring directly, unless origination fitting becomes a supported separate API. |
| `tests/test_regularizer_integration.py` importing `optimize` | Retarget `make_value_and_grad`/penalty composition or remove if it only validates the deleted orchestration. |
| `gates/_fit_kbench.py` | Archive or convert to a standalone benchmark with local baseline code. |
| `gates/_verify_map.py` importing `_DEFAULT_SO` | Give the gate its own explicit fixture configuration; never import private production constants. |

Tests should not be counted as evidence that a feature is production-reachable when the test exists
only to preserve that feature. Preserve tests for supported behavior and mathematical invariants;
remove tests whose sole purpose is preventing deletion of an abandoned orchestration layer.

## Validation checklist

### Static and CPU checks

- All retained modules parse and import.
- No production module imports from `experiments`, `gates`, or tests.
- No fit module mutates `os.environ` at import time.
- No public parameter is ignored.
- No user-facing docstring references kernel-bench, task numbers, local absolute paths, or obsolete
  package names.
- The package export surface is explicit and tested.
- Result objects serialize without CUDA tensors or implementation-specific nested dictionaries.

### CUDA correctness checks

- Global and genewise loss/rates match the Phase 0 baselines.
- Small tracked fixtures remain finite in fp32 and fp64 where supported.
- Multi-batch global/genewise runs complete without stale saved-state or memory-gate failures.
- Genewise rebatching preserves family order and does not change the all-family final NLL.
- Bound-active families satisfy projected-gradient convergence rather than raw-gradient convergence.
- Final evaluation explicitly states its precision and solver tier.

### Performance checks

- Compare warm steady-state timings on the same GPU and commit.
- Measure end-to-end fit time, number of forward/gradient evaluations, model rebuild count, peak
  allocated/reserved CUDA memory, and final NLL.
- Require no meaningful regression in the supported recipes. A useful default threshold is 5% wall
  time after enough repetitions to resolve noise.
- Do not preserve dead generic infrastructure because it once produced a benchmark result; preserve
  the result and reproducible command in a benchmark report.

## Completion criteria

The cleanup is complete when:

- `gpurec/fit` contains only supported fitting code;
- `gpurec fit` exposes only working modes and meaningful options;
- one canonical Python entry point and one result schema cover global/genewise fitting;
- specieswise MAP/CV is either a fully supported, separately documented product workflow or clearly
  isolated under experiments;
- `map_fit.py`, dead optimizer variants, and benchmark-only baselines are gone from production;
- tests no longer keep abandoned fit APIs alive indirectly;
- configuration precedence is explicit and typed;
- no import changes process-wide environment state;
- docstrings describe contracts rather than the history of the research project; and
- global/genewise numerical results and performance remain at least as good as the pinned baseline.
