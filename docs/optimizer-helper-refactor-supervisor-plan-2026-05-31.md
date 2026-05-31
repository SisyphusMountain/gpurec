# Optimizer Helper Refactor Supervisor Plan, 2026-05-31

Scope: documentation-only supervisor plan for the next optimizer-helper
extraction slice. Do not edit production code in this pass. The reviewed
production files are `gpurec/optimization/lbfgsb.py`,
`gpurec/optimization/projected_lbfgs.py`, and
`gpurec/optimization/batched_lbfgs.py`.

Concurrent-work note: during this supervisor pass, helper implementation files
and `ProjectedLBFGS` routing appeared in the worktree. Treat those as another
agent's work. Do not overwrite them; audit them against the APIs and invariants
below, then adapt the remaining sequence around whichever pieces already exist.

## Recommendation

Extract the low-level bounded-optimizer plumbing first:

- bound normalization, projection, projected-gradient maps, and feasible
  direction masking;
- closure result validation and flat-gradient gathering;
- scalar Armijo accept/required-decrease predicates shared by `LBFGSB` and
  `ProjectedLBFGS`.

Keep optimizer algorithm mechanics local for this slice:

- `LBFGSB` generalized Cauchy point, reduced-CG subspace step, fallback
  competition, high-KKT stall handling, adaptive fallback alpha, and
  `_LineSearchResult`;
- `ProjectedLBFGS` step loop and accepted/rejected probe handling;
- `BatchedLBFGS._strong_wolfe()`, `_cubic_interpolate()`, row-wise Armijo loop,
  `_append_history()`, and `_direction()`;
- class constructors, `_flat_param()`, `_set_flat_param()`, and state writers.

The current broader plan is directionally right, but the first slice should not
extract a generic line-search driver. `LBFGSB` computes probe derivatives with
raw gradients, `ProjectedLBFGS` uses projected gradients, and `BatchedLBFGS`
uses per-row masked acceptance plus a PyTorch-parity strong-Wolfe path. Unifying
those now is where counter and accepted-step regressions are most likely.

## Helper APIs

Add internal modules only; do not export them from `gpurec.optimization`.
Preserve the existing private optimizer methods as thin delegates where useful
so concurrent work and white-box tests do not need to move immediately.

### `gpurec/optimization/_bounds.py`

```python
from typing import Literal, NamedTuple

BoundBroadcastMode = Literal["param_only", "param_then_flat"]

class FlatBounds(NamedTuple):
    lower: Tensor | None
    upper: Tensor | None

def bound_for_flat(
    bound: float | Tensor | None,
    *,
    flat: Tensor,
    param_shape: torch.Size | tuple[int, ...],
    mode: BoundBroadcastMode,
) -> Tensor | None: ...

def bounds_for_flat(
    flat: Tensor,
    lower_bound: float | Tensor | None,
    upper_bound: float | Tensor | None,
    *,
    param_shape: torch.Size | tuple[int, ...],
    mode: BoundBroadcastMode,
) -> FlatBounds: ...

def project_flat(flat: Tensor, bounds: FlatBounds) -> Tensor: ...
def projected_gradient(flat: Tensor, grad: Tensor, bounds: FlatBounds) -> Tensor: ...
def feasible_direction(flat: Tensor, direction: Tensor, bounds: FlatBounds) -> Tensor: ...
```

Broadcast semantics must be explicit:

- All bound tensors are detached and moved to `flat.device`/`flat.dtype`.
- `None` remains `None`.
- Python numbers and 0-D tensors broadcast over every flat coordinate.
- Exact `flat.shape` is accepted before original parameter shape.
- Exact `param_shape` is accepted and reshaped to `flat.shape`.
- `mode="param_only"` then tries only broadcast-to-`param_shape` and reshape.
  Use this for `LBFGSB` to preserve its stricter invalid-shape behavior.
- `mode="param_then_flat"` first tries broadcast-to-`param_shape`; if that
  fails, it tries broadcast-to-`flat.shape`. Use this for `ProjectedLBFGS` and
  `BatchedLBFGS`.
- For `BatchedLBFGS`, `flat.shape == (B, N)`. This must accept per-row bounds
  shaped `(B, 1)`, per-coordinate bounds shaped `(N,)` when they broadcast to
  flat, full flat bounds shaped `(B, N)`, and original-parameter bounds shaped
  like `param.shape`.
- After broadcasting lower and upper, keep the exact error string
  `lower_bound must be <= upper_bound` when any lower coordinate exceeds upper.

### `gpurec/optimization/_closures.py`

```python
LossClosure = Callable[[], Tensor]

def gather_flat_grad(param: Tensor, flat_like: Tensor, *, owner: str) -> Tensor: ...

def scalar_loss_tensor(loss: Tensor, *, owner: str, loss_only: bool) -> Tensor: ...
def evaluate_scalar_with_grad(
    param: Tensor,
    flat_like: Tensor,
    closure: LossClosure,
    *,
    owner: str,
) -> tuple[Tensor, Tensor]: ...
def evaluate_scalar_loss(
    closure: LossClosure,
    loss_closure: LossClosure | None,
    *,
    owner: str,
) -> Tensor: ...

def vector_loss_tensor(loss: Tensor, *, batch_size: int, owner: str) -> Tensor: ...
def evaluate_vector_with_grad(
    param: Tensor,
    flat_like: Tensor,
    closure: LossClosure,
    *,
    batch_size: int,
    owner: str,
) -> tuple[Tensor, Tensor]: ...
def evaluate_vector_loss(
    closure: LossClosure,
    loss_closure: LossClosure | None,
    *,
    batch_size: int,
    owner: str,
) -> Tensor: ...
```

Closure semantics to preserve:

- `evaluate_*_with_grad()` runs under `torch.enable_grad()`.
- `evaluate_*_loss()` runs `loss_closure` under `torch.no_grad()` when supplied;
  otherwise it runs `closure` under `torch.enable_grad()`.
- Missing gradients return `torch.zeros_like(flat_like)`.
- Sparse gradients are densified; complex gradients raise
  `{owner} only supports real-valued gradients`.
- Scalar losses must be tensors with `numel() == 1` and reshape to `()`.
- Vector losses must be tensors with `numel() == batch_size` and reshape to
  `(batch_size,)`.
- Preserve exact tested messages, including:
  `LBFGSB closure must return a scalar Tensor`,
  `LBFGSB loss closure must return a scalar Tensor`,
  `ProjectedLBFGS closure must return a scalar Tensor`,
  `ProjectedLBFGS loss closure must return a scalar Tensor`,
  `BatchedLBFGS closure must return a Tensor`, and
  `BatchedLBFGS closure must return one loss per parameter row; got shape ...`.

### `gpurec/optimization/_armijo.py`

```python
def scalar_armijo_accepts(
    *,
    trial_loss: Tensor,
    loss: Tensor,
    trial_gtd: Tensor,
    c1: float,
) -> bool: ...

def scalar_armijo_required_decrease(
    *,
    loss: Tensor,
    trial_gtd: Tensor,
    c1: float,
) -> float: ...
```

This helper keeps the scalar strict-decrease rule:
`trial_loss <= min(loss + c1 * trial_gtd, nextafter(loss, -inf))`, with any
non-finite input rejected. Do not apply this helper to `BatchedLBFGS` Armijo in
this slice; its current row-wise condition is vectorized and intentionally not
the scalar `nextafter` rule.

## Invariants

Public surface:

- `from gpurec.optimization import LBFGSB, ProjectedLBFGS, BatchedLBFGS` must
  keep working.
- Do not change `gpurec/optimization/__init__.py` or public class names,
  constructor signatures, defaults, or `step(closure, *, loss_closure=None)`.
- New helper modules remain private and import only stdlib plus `torch`.

State keys:

- `LBFGSB`: preserve `old_dirs`, `old_stps`, `consecutive_high_kkt_stalls`,
  `last_projected_gradient_fallback_alpha`, `last_loss`, `last_grad`,
  `last_projected_grad`, `last_grad_evals`, `last_loss_evals`,
  `last_accepted`, `last_alpha`, `last_step_inf`,
  `last_directional_derivative`, `last_direction_kind`,
  `last_line_search_decrease`, `last_armijo_required_decrease`,
  `last_fallback_attempted`, `last_fallback_used`, `last_fallback_alpha`,
  `last_fallback_loss_evals`, `last_fallback_max_loss_evals`,
  `last_fallback_budget_exhausted`, `last_fallback_reason`,
  `last_high_kkt_stall_count`, `last_history_cleared_for_fallback`, and
  `n_iter`. Keep legacy `param_groups[0].get(...)` fallbacks for fallback keys.
- `ProjectedLBFGS`: preserve `old_dirs`, `old_stps`, `ro`, `last_loss`,
  `last_grad`, `last_projected_grad`, `last_grad_evals`, `last_loss_evals`,
  `last_accepted`, `last_alpha`, `last_step_inf`,
  `last_directional_derivative`, and `n_iter`.
- `BatchedLBFGS`: preserve `func_evals`, `n_iter`, `old_dirs`, `old_stps`,
  `ro`, `H_diag`, `last_loss`, `last_grad`, `last_projected_grad`,
  `last_accepted`, `last_alpha`, and `last_n_iter`.

Counters and accepted-step behavior:

- Scalar optimizers start each `step()` with one gradient closure evaluation;
  `last_grad_evals` starts at `1` and increments only after accepted-point
  gradient refreshes.
- Scalar `last_loss_evals` counts only evaluated loss-only probes, not skipped
  tiny-step or non-descent probes.
- Scalar `state["n_iter"]` still increments by configured `max_iter` at method
  exit, not by actual loop iterations.
- `BatchedLBFGS.state["func_evals"]` increments for the initial gradient
  closure, every loss probe, and any refreshed gradient closure. Preserve the
  `max_eval` case where an accepted probed loss can be returned without a
  refreshed gradient.
- All optimizers still project the initial parameter into bounds before the
  first closure.
- Rejected scalar probes must restore the original flat parameter. The
  `ProjectedLBFGS` all-probes-rejected case must leave the parameter unchanged.
- `LBFGSB` fallback reason and direction strings are stable:
  `line_search_failed`, `high_kkt_tiny_progress`,
  `projected_gradient_fallback`, `projected_gradient_sign_fallback`,
  `projected_gradient_top{k}_sign_fallback`, and
  `projected_gradient_coord{rank}_sign_fallback`.
- Keep `BatchedLBFGS._strong_wolfe()` byte-for-byte local unless a later slice
  is dedicated to it; tests compare it against PyTorch's per-row search.

## Test Matrix

Add focused helper tests only for behavior not already covered indirectly:

- `_bounds`: scalar, exact flat, exact parameter-shape, broadcast-to-parameter,
  and broadcast-to-flat fallback cases; explicit `LBFGSB` `param_only` invalid
  shape; batched `(B, 1)`, `(N,)`, `(B, N)`, and original-shape bounds;
  lower-greater-than-upper error string.
- `_closures`: scalar and vector loss shape validation; non-tensor vector loss
  type error; sparse-gradient densification; complex-gradient rejection;
  missing-gradient zero fill with scalar and batched flat shapes.
- `_armijo`: finite strict-decrease acceptance, equality-to-loss rejection via
  `nextafter`, and non-finite rejection.

Then run the existing optimizer parity suite:

```bash
python -m pytest -q \
  tests/unit/test_projected_lbfgs.py \
  tests/unit/test_lbfgsb.py \
  tests/unit/test_batched_lbfgs.py \
  tests/unit/test_lbfgsb_schilling_conformance.py
```

Run import/hygiene smoke gates:

```bash
python -m pytest -q \
  tests/unit/test_dependency_inventory.py \
  tests/unit/test_repository_hygiene.py

python - <<'PY'
from gpurec.optimization import BatchedLBFGS, LBFGSB, ProjectedLBFGS
print(BatchedLBFGS.__name__, LBFGSB.__name__, ProjectedLBFGS.__name__)
PY
```

Acceptance gates:

- No production files outside `gpurec/optimization/` change.
- No public optimizer import, signature, default, or state-key change.
- Existing optimizer tests pass before adding broader workflow tests.
- `BatchedLBFGS` strong-Wolfe PyTorch parity remains unchanged.
- Focused helper tests prove bound broadcasting before any optimizer is routed
  through helpers.

## Sequencing To Avoid Conflicts

1. Land helper modules plus helper unit tests first, with no optimizer routing.
   This gives concurrent implementers a stable internal target and isolates
   broadcast semantics.
2. If helper modules or `ProjectedLBFGS` routing already exist from concurrent
   work, review them against this plan before adding more routes. Prefer small
   compatibility edits over renaming APIs mid-slice.
3. Route `ProjectedLBFGS` through `_bounds`, `_closures`, and `_armijo` first.
   It is the smallest scalar optimizer and has the tight all-probes-rejected
   restore test.
4. Route `BatchedLBFGS` only through `_bounds` and `_closures`. Do not touch
   `_strong_wolfe()` or row-wise line-search structure in this slice.
5. Route `LBFGSB` through `_bounds`, `_closures`, and `_armijo` last. Its
   fallback/high-KKT state surface is the largest and most likely to conflict
   with active performance work.
6. Keep each routing step in a separate commit or PR section. If another agent
   is already editing one optimizer file, skip that file and still land the
   helper tests plus the non-conflicting optimizer routes.
7. Defer any generic scalar backtracking-line-search extraction to a later
   supervised slice after helper routing is green and there is a characterization
   test comparing `LBFGSB` and `ProjectedLBFGS` probe counters and restore
   behavior.
