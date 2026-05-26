# Gradient And Likelihood Refactor Plan, 2026-05-21

This plan focuses on simplifying the likelihood and gradient computation
itself.  It complements `refactor-simplification-plan-2026-05-21.md` by giving
the detailed path from today's duplicated execution flow to one explicit
uniform-transfer evaluator.

## Current Computation Graph

For all retained modes, the mathematical pipeline is:

1. Convert theta to log probabilities:
   `extract_parameters_uniform(theta, unnorm_row_max, specieswise, genewise)`.
2. Solve the uniform-transfer E fixed point:
   `E_fixed_point(...)`.
3. Run wave-ordered Pi fixed iterations:
   `Pi_wave_forward(...)`.
4. Compute root negative log-likelihood:
   `compute_nll(...)` or `compute_nll_root_rows(...)`.
5. For gradients, run:
   - `Pi_wave_backward(...)`;
   - `_e_adjoint_and_theta_vjp(...)`;
   - VJP through `extract_parameters_uniform(...)`.

The graph is conceptually simple, but the code has several entry points that
rebuild it differently:

- Differentiable active-batch path:
  `gpurec/api/autograd.py:115` `_GeneReconFunction`.
- No-grad / explicit-theta / full-stream path:
  `gpurec/api/model.py:685` `_evaluate_static_state()` and
  `gpurec/api/model.py:1329` `_stream_full_batches()`.
- Export-state path:
  `gpurec/api/model.py:1748` `reconciliation_state()`.
- Large global/uniform chunk path:
  `gpurec/api/uniform_chunked.py:477` `_evaluate_chunked_uniform()`.
- Benchmark path:
  `profiling/bench_uniform_forward_backward_pipeline.py`.

The refactor goal is one implementation of the graph and many small adapters.

## Core Abstractions To Add

### `RateMode`

An enum-like value with exactly the retained modes:

- `global`
- `specieswise`
- `genewise`

It should replace repeated pairs of booleans where a mixed
`genewise=True, specieswise=True` state is theoretically possible in helper
code but not a supported public mode.

### `ParameterLayout`

Owns theta and parameter addressing:

- validates theta shape once;
- knows family count and species count;
- maps current resident batch rows to global theta rows;
- produces kernel addressing strides;
- owns gradient reduction back to theta shape.

This replaces shape inference in:

- `gpurec/core/extract_parameters.py`;
- `gpurec/core/forward.py`;
- `gpurec/core/backward.py`;
- `gpurec/core/kernels/dts_fused.py`;
- `gpurec/core/kernels/wave_backward.py`.

The important design choice is to make shared layouts explicit:

| Mode | Theta | Parameter Row Stride | Species Stride |
|---|---:|---:|---:|
| global scalar event probability | `[3]` | `0` | `0` or expanded `[S]` |
| specieswise event probability | `[S, 3]` | `0` | `1` |
| genewise scalar event probability | `[G, 3]` | `1` | `0` |
| future family-species internal tensor | `[G, S]` | `S` | `1` |

Only the first three are public modes; the fourth can exist internally as a
normalized representation for per-family species constants.

### `UniformRates`

Return type for theta extraction:

- `log_pS`
- `log_pD`
- `log_pL`
- `max_transfer`
- `layout`

It should be the only object passed from theta extraction to E/Pi/backward
logic.

### `OriginationPrior`

Owns validation, normalization, and root likelihood weights:

- shared `[S]`;
- family-specific `[G, S]`;
- selected-batch view;
- log-weight view for root likelihood.

This removes repeated `prepare_origination_probs(..., assume_prepared=True)`
calls from hot paths.

### `EvaluationRequest`

Fields:

- `need_gradient: bool`;
- `output: "sum" | "per_family" | "root_rows" | "state"`;
- `theta`;
- optional selected batch/chunk indices;
- warm-start policy;
- profiling/timing policy.

### `EvaluationResult`

Fields:

- `loss`;
- optional `grad_theta`;
- optional `per_family_loss`;
- optional `state` with E/Pi/rates;
- `solver_stats`;
- optional timing/chunk stats.

The autograd functions should save only the gradient from this result, not
reimplement the solve.

## Detailed Refactor Steps

### Step 1: Characterize The Current Graph

Add tests before refactoring:

- global `GeneReconModel.forward()` equals `_evaluate_static_state(...,
  need_grad=False)` loss.
- specieswise `GeneReconModel.forward()` gradients match current
  `Pi_wave_backward` path.
- genewise `model(reduce="per_family")` gradient scaling with `grad_output`
  still works.
- resident-batched `full_loss()` equals sum over explicit batch evaluation.
- `UniformChunkedReconModel.loss_and_grad(chunk_indices=...)` equals the sum of
  selected chunks and scales correctly for `mean` and `full_sum_estimate`.
- `reconciliation_state(original_order=True/False)` returns the same Pi as
  `pi_matrix()`.

These tests should use existing tiny CPU-safe monkeypatches where possible and
the existing CUDA fixtures for numerical parity.

### Step 2: Introduce Layout Objects Without Kernel Changes

Create layout objects and adapt them to the current tensor signatures:

- `ParameterLayout.from_model(model/static/batch)`;
- `layout.extract_rates(theta, unnorm_row_max)`;
- `layout.family_idx_for_wave(wave_layout)`;
- `layout.reduce_gradient(raw_grad)`.

At this stage, the wrappers can still return the current scalar, `[S]`, `[G]`,
or `[G, S]` tensors to avoid changing kernels immediately.

Gates:

- Existing `extract_parameters_uniform` tests.
- New tests for `G == S` shape precedence.
- Model construction rejects unsupported mixed layouts early.

### Step 3: Make E Solver Shape Explicit

`E_fixed_point()` now accepts an explicit `e_shape`. Resident/chunked model
callers pass the shape derived from `ParameterLayout`, and maintained global /
uniform CUDA warmup and full-pipeline benchmark callers pass explicit `[S]`
shape instead of relying on parameter-shape inference. The legacy inference
remains for direct low-level callers until every supported path supplies an
explicit layout or `UniformRates.e_rows`.

Internal behavior should become:

- global/specieswise: E shape `[S]`;
- genewise: E shape `[G, S]`.

`E_step()` math is unchanged.  Remove the remaining N-detection branches only
after tests prove all supported callers supply explicit layout.

Gates:

- `tests/unit/test_origination_probs.py`;
- `tests/unit/test_specieswise_uniform.py`;
- genewise integration tests.

### Step 4: Standardize Root Likelihood On Root Rows

Internal code should call a single helper:

```python
root_nll(root_rows, E, prior)
```

Then:

- gradient/training callers pass `Pi_wave_ordered[root_clade_ids]`;
- loss-only callers pass `Pi_root_rows`;
- export/state callers compute loss only if requested.

After internal migration:

- keep `compute_nll()` as a thin root-gather adapter if it is useful;
- keep the removed `compute_log_likelihood*` aliases out of runtime code.

Gates:

- Existing origination probability tests.
- Benchmark script uses `compute_nll` rather than the removed likelihood alias.

### Step 5: Build The Shared Evaluator

Move the current sequence from `_evaluate_static_state()` into a new evaluator:

1. extract rates;
2. solve E;
3. solve Pi according to output intent;
4. compute NLL;
5. if requested, run gradient.

Then route call sites incrementally:

- `GeneReconModel.forward()` no-grad path;
- `_GeneReconFunction.forward/backward`;
- `_GeneReconFullLossFunction`;
- `full_genewise_nll_and_grad()`;
- `reconciliation_state()`;
- `UniformChunkedReconModel.nll()` / `nll_per_family()` /
  `loss_and_grad()`;
- benchmark script.

The evaluator should own `last_solver_stats` updates so the same keys are
reported from resident and chunked calls.

### Step 6: Make Pi Forward Output Intent Explicit

Replace `Pi_wave_forward(..., return_original, return_root_rows, ...)` at call
sites with wrappers:

- `forward_pi_gradient_state()`;
- `forward_pi_root_rows()`;
- `forward_pi_export_state()`.

The low-level implementation can remain one loop during the first pass, but
callers should stop passing many booleans.  Once call sites are explicit,
delete unreachable combinations.

Examples of combinations to remove:

- `return_root_rows=True` with `return_original=True`;
- saved `Pibar` requested for root-row-only inference;
- root trace allocated for callers that do not inspect it.

### Step 7: Refactor Pi Backward Around Layout And Accumulators

Break `Pi_wave_backward()` into:

- `initialize_root_rhs(prior, root_rows)`;
- `compute_wave_dts(...)`;
- `run_self_loop_vjp(...)`;
- `accumulate_self_loop_grads(...)`;
- `run_dts_vjp(...)`;
- `finalize_gradient_dict(...)`.

Replace internal `_scatter_accum()` and `_auto_wrapped` with a
`GradientAccumulator(layout)` object.

Keep the retained Triton implementation first.  Move native CUDA prototypes
out of the function before deleting them so benchmarks can still be run during
the transition.

### Step 8: Rework Implicit Gradient API

`implicit_grad_loglik_vjp_wave()` currently does both adaptive Neumann-term
selection and E-adjoint/theta VJP.  Keep it, but make inputs typed:

- `UniformRates`;
- `RootLikelihoodAdjoint`;
- `BackwardState`;
- `ParameterLayout`.

The function should no longer know about raw `specieswise`, `genewise`, and
`family_idx` flags.

Also decide the E-adjoint failure policy in one place:

- current behavior: consume best BiCGSTAB iterate and report telemetry;
- future behavior, if wanted: retry/fail based on relative residual.

## Behavioral Invariants

The refactor must preserve:

- NLL sign convention: lower is better, returned loss is NLL in bits.
- Log base 2 throughout D/T/L rates, E, Pi, Pibar, and likelihood.
- Uniform transfer model: no dense transfer matrix in the retained path.
- Public modes: global/uniform, specieswise, genewise.
- Genewise independent per-family gradient semantics.
- Specieswise rates indexed by species, not family.
- Root origination prior semantics for shared and family-specific priors.
- Autograd first-order behavior, with no promise of double backward.

## Suggested Numerical Gates

Minimum parity set:

- Compare loss and gradient before/after refactor on one small CUDA family in
  global mode.
- Compare specieswise loss and gradient on the existing specieswise CUDA test.
- Compare genewise per-family losses and gradient rows.
- Compare chunked global loss/gradient for all chunks and a selected subset.
- Compare `pi_matrix(original_order=True/False)` before/after.

Performance gates:

- full uniform benchmark with strict optimized kernels;
- chunked global `loss_and_grad()` timing before/after evaluator migration;
- scheduler-sensitive batch with `max_wave_size=8192`.

Failure-mode gates:

- missing `ancestors_T` still fails clearly;
- unsupported CPU/dtype/small-S backward limitations stay documented or are
  replaced by tested implementations;
- E-adjoint nonconvergence telemetry remains visible.

## Open Design Decisions

- Whether `UniformChunkedReconModel` should remain first-class or become a
  facade over `GeneReconModel`.
- Whether root-row-only inference should keep using the same Pi loop or a
  separate memory-minimal loop.
- Whether adaptive iteration support is production behavior or diagnostic
  behavior.  If production, trace/stats need a stable typed output.
- Whether native CUDA prototypes deserve ownership.  If not, they should leave
  the production gradient path.
