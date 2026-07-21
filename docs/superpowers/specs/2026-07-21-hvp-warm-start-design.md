# HVP warm-starting design

## Context

`gpurec/solver/hvp_exact.py`'s analytic exact-Hessian HVP (`make_exact_hvp` /
`make_exact_hvp_single`) is built fresh on every call: `build_point_cache` runs a
full backward pass to seed the point cache, and each `hvp(u)` probe runs its own
tangent-adjoint sweep from a cold start. Neither of these benefits from the
warm-starting the *primal* gradient path already has (`GPUREC_WARM_ADJOINT` +
`static.warm_v`, wired through `_execution.py`), which caches each wave's adjoint
`v_k` as the Neumann self-loop's initial guess for the next call at a nearby theta.

A benchmark comparing `fit_genewise` (production FD-Hessian Newton) against an
analytic-HVP variant on real archaea60 data (5446 families) showed the analytic
path needing fewer outer Newton steps (53 vs 75) but ending up 10% *slower*
overall (375.4s vs 340.2s) — consistent with paying a cold-start cost on every
Hessian-construction call that the primal path does not pay.

Investigating why turned up two distinct places the self-loop Neumann solve runs
cold inside the HVP construction:

1. **`build_point_cache`'s own backward pass** — it calls `vjp_root_to_theta`,
   which is a thin wrapper over the same `implicit_grad_loglik_vjp_wave` the
   primal gradient uses, but `vjp_root_to_theta` doesn't accept or forward a
   `warm_v` argument at all. This is a missing pass-through, not new machinery:
   `implicit_grad_loglik_vjp_wave` already knows how to consume `warm_v`.
2. **The tangent-adjoint sweep inside each `hvp(u)` probe** — the second-order
   `v_k` this computes has no existing cache anywhere; this is genuinely new
   state, not a missing wire-up.

## Goals

- Warm-start both (1) and (2), controlled entirely through `SolverOptions` —
  no new `os.environ` reads. Backward's existing `GPUREC_WARM_ADJOINT` +
  `static.warm_v` mechanism stays exactly as it is (out of scope here); the
  new HVP gating is independent of that env var, gated instead by the new
  `use_hvp_warm_start` config field (default `True`) and the existing
  `static.warm_adjoint_ok` memory gate — which is itself computed purely from
  a memory-budget check (`warm_adjoint_fits` in `gpurec/api/model.py`,
  `gpurec/core/memory_policy.py`) at model-build time, with no dependency on
  the env var either. So HVP warm-starting is config-driven top to bottom:
  set `use_hvp_warm_start=False` in your `GpurecConfig`/TOML file and it's off,
  full stop.
- The *tangent-adjoint* warm-start (part 2) additionally requires each call to
  opt in with an explicit `probe_id` (see below) — so even with the config
  field at its default, zero behavior change for existing CG/Lanczos-based HVP
  callers (`newton_joint_genewise`, `origination_curvature.newton_joint`,
  `optimize.py`'s `newton_polish`/`newton_lanczos`), none of which pass one.
- Ship as a real library feature in `gpurec/solver/hvp_exact.py` (and
  `gpurec/solver/ggn.py` for the point-cache piece), not scoped to a benchmark
  script — but the intent for *this* piece of work is validation, not adoption:
  we test and compare against FD before any fit recipe is changed to use it.

## Non-goals

- Not wiring this into `fit_genewise` or any other production recipe yet. That's
  a separate, later decision once this is validated.
- Not adding a warm-start hook to the E-adjoint GMRES/Neumann solve
  (`_gmres`/`_neumann_e_adjoint` in `_implicit_grad.py`). Backward doesn't warm-
  start that solve either (only the per-wave self-loop), so this stays symmetric
  with the existing strategy rather than extending it further.
- Not adding eviction/LRU logic for the new per-probe cache. If memory becomes a
  real problem, the `use_hvp_warm_start=False` opt-out is the escape hatch.

## Design

### 1. Point-cache warm-start (reuses the existing `static.warm_v`)

- `vjp_root_to_theta` (`gpurec/solver/ggn.py`) gains `warm_v: dict | None = None`,
  forwarded to `implicit_grad_loglik_vjp_wave`'s existing `warm_v` parameter.
- `build_point_cache` (`gpurec/solver/hvp_exact.py`) gains the same
  `warm_v: dict | None = None`, forwarded to `vjp_root_to_theta`.
- `make_exact_hvp_single` gates and supplies it — config-only, no env var:

  ```python
  if getattr(static, "warm_adjoint_ok", True) and static.solver_options.use_hvp_warm_start:
      if static.warm_v is None:
          static.warm_v = {}
      _warm_v = static.warm_v
  else:
      _warm_v = None
  ```

  then passes `warm_v=_warm_v` into `build_point_cache(...)`.
- This is the *same* dict the primal gradient call populates when
  `GPUREC_WARM_ADJOINT` is on. The two mechanisms are decoupled but
  cooperative: whichever code path (backward or HVP) populates `static.warm_v`
  first, the other can read it back, as long as its own gate allows reading.
  If backward's env var is off, `static.warm_v` simply starts empty and the
  HVP path builds it up on its own (still a net win across repeated
  Hessian-construction calls at nearby theta, just without a backward head
  start). No new memory beyond a single `warm_v` dict's footprint.

### 2. Tangent-adjoint warm-start (new state, opt-in per call)

- `hvp(u_vec)`, the closure `make_exact_hvp`/`make_exact_hvp_single` return,
  gains an optional second parameter: `hvp(u_vec, probe_id=None)`. `probe_id`
  must be hashable (an `int` for genewise's 0/1/2 theta components; any
  hashable works for other callers) — it's used only as a dict key, never
  inspected or compared against `u_vec`.
- `probe_id` identifies "this call's `u` plays the same role as a previous
  call's `u`" — it is NOT derived from `u`'s contents; the caller supplies it.
  CG/Lanczos-style callers, whose search direction changes every call with no
  stable identity to key on, pass nothing (`probe_id=None`) and get today's
  cold-start behavior, unconditionally — no config flag changes their behavior.
  Callers with a stable probe across outer iterations (e.g. genewise's three
  broadcast unit-theta-component probes, rebuilt every few Newton steps) pass
  `probe_id=0/1/2` explicitly.
- New per-static cache: `static.warm_v_tangent: dict[Any, dict[int, Tensor]]`
  — outer key is `probe_id`, inner key is wave-start offset `ws` (mirrors
  `static.warm_v`'s existing `{ws: tensor}` shape, just one layer deeper).
- Gating: `static.warm_adjoint_ok` AND `static.solver_options.use_hvp_warm_start`
  AND `probe_id is not None` — no env var. All three must hold for a given
  probe's tangent-adjoint solve to read/write `static.warm_v_tangent[probe_id]`.
- Same NaN-safe sanitization as backward: pruned/inactive rows hold
  uninitialized scratch, so before caching, zero those rows via
  `torch.where(row_active, v_k, 0.0)` — never multiply (0 * NaN = NaN).

### 3. New config field

- `SolverOptions.use_hvp_warm_start: bool = True` (`gpurec/api/solver_options.py`),
  following the `use_adjoint_pruning` naming convention. Governs *both*
  mechanisms above, entirely independently of `GPUREC_WARM_ADJOINT`. Default
  `True` (no behavior change for anyone today, since nothing currently passes
  `probe_id=`) — but settable to `False` to opt out and save the
  `warm_v_tangent` memory (up to ~3x a single `warm_v`'s footprint for
  genewise's 3-probe case) when memory is tight.
- Added to `gpurec/config/defaults.toml`'s `[solver]` section, matching the
  dataclass default, same as every other `SolverOptions` field (the
  `test_defaults_toml_matches_dataclass_defaults` guard test enforces this).

### Memory

No new eviction logic. `use_hvp_warm_start=False` is the documented way to
avoid the extra memory. Flagged with a comment mirroring
`_warm_reserved_scratch_bytes`'s docstring, but not integrated into that
function's byte-accounting in this pass — a follow-up if it proves necessary.

## Testing plan

1. **Unit-level correctness**, extending `tests/test_genewise_hvp.py`:
   - No `probe_id` passed (default): byte-identical to current behavior —
     critical regression guard, since this must never change anything for
     existing callers, regardless of `use_hvp_warm_start`.
   - `use_hvp_warm_start=True` (default), `probe_id` passed across two calls
     at nearby theta: warm-started result still agrees with the FD/cold
     reference within the existing tolerance (`tests/test_genewise_hvp.py`'s
     `5e-4` gate).
   - Probe isolation: probe 0's warm state must not leak into probe 1's
     solve (build two probes with deliberately different cached `v_k` and
     confirm each reads back its own).
   - `use_hvp_warm_start=False` disables warm-starting even with `probe_id`
     passed — confirms the config field is a real, sufficient off-switch.
2. **End-to-end benchmark re-run**: repeat the archaea60 `fit_genewise` (FD)
   vs `fit_genewise_analytic` (analytic HVP) comparison from the earlier
   session, this time with both warm-start pieces wired into the analytic
   variant and `probe_id=0/1/2` passed for the three broadcast theta-component
   probes. Report actual measured wall-clock — no speculation on whether it
   closes the gap with FD (340.2s) until measured.
3. Confirm nothing else in the existing test suite regresses (full `pytest
   tests/ -m gpu` run), same verification bar as the GMRES-removal work.
