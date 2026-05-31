# API Theta Constraints Refactor Plan - 2026-05-31

## Scope

- Extract duplicated theta natural-rate validation, log2 bound conversion,
  in-place clamping, and projected-gradient mapping into
  `gpurec/api/_theta_constraints.py`.
- Keep `GeneReconModel.clamp_theta_()` and
  `UniformChunkedReconModel.clamp_theta_()` as the public methods; the helper is
  private support code.
- Let workflow import the private helper only for shared theta math. Optimizer
  phase selection, stopping policy, line search policy, and FD-Newton control
  flow stay in `gpurec.workflow`.

## Invariants

- `min_rate` and `max_rate` keep the existing positive-float validation and
  error messages, including bool rejection through the shared validation layer.
- `max_rate=None` remains valid for public clamping; workflow still supplies a
  finite upper bound through `RunConfig`.
- Public clamp methods mutate `theta` in place under `torch.no_grad()` and do
  not change return values.
- Projected gradients keep the exact map
  `theta - clamp(theta - grad, lower_bound, upper_bound)` and the existing
  infinity-norm metric semantics.
- FD-Newton's free mask stays `projected.abs() > 0`.

## Implementation Steps

1. Add the private helper module with `theta_rate_bounds_log2()`,
   `finite_theta_rate_bounds_log2()`, `clamp_theta_rates_()`,
   `projected_theta_gradient_inf()`, and
   `projected_theta_gradient_and_free()`.
2. Rewire the two public model clamp methods to the helper and remove local
   `math.log2()` / validation duplication.
3. Rewire workflow optimizer bounds, projected-gradient metrics, finalization,
   and FD-Newton active-set logic to the same helper.
4. Add low-level CPU tests for rate-bound conversion, in-place clamping,
   projected-gradient maps, and FD-Newton free-mask parity.
5. Run focused validation/workflow tests, then the broad CPU unit marker.

## Verification Gates

- `python -m compileall -q gpurec/api/_theta_constraints.py gpurec/api/model.py gpurec/api/uniform_chunked.py gpurec/workflow/_optimizer_factory.py gpurec/workflow/_evaluation.py gpurec/workflow/_fd_newton.py gpurec/workflow/_step_execution.py gpurec/workflow/_finalization.py tests/unit/test_validation.py tests/unit/test_workflow.py`
- `ruff check gpurec/api/_theta_constraints.py gpurec/api/model.py gpurec/api/uniform_chunked.py gpurec/workflow/_optimizer_factory.py gpurec/workflow/_evaluation.py gpurec/workflow/_fd_newton.py gpurec/workflow/_step_execution.py gpurec/workflow/_finalization.py tests/unit/test_validation.py tests/unit/test_workflow.py`
- `python -m pytest -q tests/unit/test_validation.py -k "theta_rate_bounds or theta_rates or projected_theta_gradient or clamp_rejects_bool"`
- `python -m pytest -q tests/unit/test_workflow.py -k "projected_gradient_uses_projection_mapping_near_rate_bound or fd_newton_line_search_falls_back_to_projected_gradient or recon_model_clamp_theta_rejects_invalid_rates"`
- `python -m pytest -q tests/unit/test_repository_hygiene.py::test_internal_api_helper_modules_document_support_boundary`
- `python -m pytest -q -m "unit and not gpu"`
