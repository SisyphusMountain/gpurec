# HVP Warm-Starting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Warm-start the analytic exact-Hessian HVP (`gpurec/solver/hvp_exact.py`) the same way the primal backward pass already warm-starts its per-wave self-loop adjoint solve, so repeated Hessian-construction calls at nearby theta points don't pay a full cold-start cost every time.

**Architecture:** Two independent mechanisms, both gated by a new `SolverOptions.use_hvp_warm_start` field (no `os.environ` reads): (1) thread the existing `static.warm_v` dict through `build_point_cache`'s own backward pass (a missing pass-through, not new state); (2) a new `static.warm_v_tangent` cache, keyed by an explicit `probe_id` the caller opts into per `hvp(u, probe_id=...)` call, for the tangent-adjoint sweep's own `v_k`.

**Tech Stack:** Python, PyTorch, CUDA (Triton kernels via existing `wave_backward.py` self-loop solve — untouched by this plan, only its existing `initial_v` parameter is used).

## Global Constraints

- No `os.environ` reads anywhere in the new code — gating is `static.warm_adjoint_ok` (existing, memory-computed) AND `static.solver_options.use_hvp_warm_start` (new field), plus `probe_id is not None` for part 2.
- Zero behavior change for any caller that never passes `probe_id=` to `hvp(...)` — this is the tangent-adjoint mechanism's regression contract (part 2). Part 1 (point-cache reuse) is allowed to change *numerical values slightly* (still within existing correctness tolerances) for repeated calls on the same `static`, since that's the intended "pure upside" reuse — Task 3 explicitly tests this stays within tolerance, and Task 6's full-suite run is the backstop for catching any golden-test drift.
- Not wiring this into any fit recipe (`fit_genewise`, etc.) in this plan — validation only, per the approved spec's non-goals.
- Reference spec: `docs/superpowers/specs/2026-07-21-hvp-warm-start-design.md`.

---

## File Map

- `gpurec/api/solver_options.py` — new `use_hvp_warm_start: bool = True` field.
- `gpurec/config/defaults.toml` — mirror the new field's default.
- `gpurec/api/_batch_state.py` — new `warm_v_tangent: dict | None = None` field on `_BatchStatic`.
- `gpurec/solver/ggn.py` — `vjp_root_to_theta` gains `warm_v=None`, forwarded to `implicit_grad_loglik_vjp_wave`.
- `gpurec/solver/hvp_exact.py` — `build_point_cache` gains `warm_v=None`, forwarded; `make_exact_hvp_single` gates+supplies `static.warm_v`; its `hvp(u_vec)` closure gains `probe_id=None` and warm-starts the tangent-adjoint sweep via `static.warm_v_tangent`; `_make_exact_hvp_streaming`'s `hvp(u_vec)` closure gains `probe_id=None`, forwarded to each batch's `hvp_b(...)`.
- `tests/test_genewise_hvp.py` — new tests for parts 1 and 2 (single-batch).
- `tests/test_hvp_multibatch.py` — new test for part 2 in the multi-batch streaming path.

---

### Task 1: Config field + new static cache field (no wiring yet)

**Files:**
- Modify: `gpurec/api/solver_options.py`
- Modify: `gpurec/config/defaults.toml`
- Modify: `gpurec/api/_batch_state.py`
- Test: `tests/test_reference_defaults.py`, `tests/test_genewise_hvp.py`

**Interfaces:**
- Produces: `SolverOptions.use_hvp_warm_start: bool` (default `True`); `_BatchStatic.warm_v_tangent: dict | None` (default `None`).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_reference_defaults.py` (anywhere after the existing imports/fixtures, alongside the other `SolverOptions` default checks):

```python
def test_solver_options_use_hvp_warm_start_defaults_true():
    from gpurec import SolverOptions
    so = SolverOptions()
    assert so.use_hvp_warm_start is True
```

Add to `tests/test_genewise_hvp.py` (after `build_genewise_model`, before the first `@pytest.mark.gpu` test):

```python
@pytest.mark.gpu
def test_batch_static_warm_v_tangent_defaults_to_none():
    m = build_genewise_model()
    assert m.batch_statics[0].warm_v_tangent is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_reference_defaults.py::test_solver_options_use_hvp_warm_start_defaults_true tests/test_genewise_hvp.py::test_batch_static_warm_v_tangent_defaults_to_none -v`

Expected: both FAIL — `AttributeError: 'SolverOptions' object has no attribute 'use_hvp_warm_start'` and `AttributeError: '_BatchStatic' object has no attribute 'warm_v_tangent'`.

- [ ] **Step 3: Add the fields**

In `gpurec/api/solver_options.py`, add the new field right after `use_adjoint_pruning` (so it sits with the other boolean toggles):

```python
    adjoint_pruning_threshold: float = 1e-6
    use_adjoint_pruning: bool = True
    # Warm-starts gpurec.solver.hvp_exact's analytic-HVP construction: (1) reuses the existing
    # static.warm_v dict for build_point_cache's own backward pass (no new memory), and (2) caches
    # the tangent-adjoint sweep's own v_k per probe_id in static.warm_v_tangent (new memory, ~probe
    # count x a single warm_v's footprint). Independent of GPUREC_WARM_ADJOINT -- config-only, no
    # env var. Default True: part (1) is pure upside when repeatedly calling make_exact_hvp on the
    # same static; part (2) only activates when a caller explicitly passes probe_id= to hvp(), so
    # existing CG/Lanczos-based callers (which never do) are unaffected either way. Set False to
    # save the warm_v_tangent memory.
    use_hvp_warm_start: bool = True
    pibar_side_threshold: float = 0.0
```

In `gpurec/config/defaults.toml`, add the matching line in the `[solver]` section, right after `use_adjoint_pruning = true`:

```toml
use_adjoint_pruning = true
use_hvp_warm_start = true
```

In `gpurec/api/_batch_state.py`, add the new field right after `warm_v`:

```python
    warm_v: dict | None = None   # per-wave backward Pi-adjoint warm-start cache (keyed by wave-start ws)
    warm_v_tangent: dict | None = None  # gpurec.solver.hvp_exact tangent-adjoint warm-start cache,
    # keyed by {probe_id: {ws: v_k}} -- see SolverOptions.use_hvp_warm_start.
    warm_adjoint_ok: bool = True  # memory gate: False -> ignore GPUREC_WARM_ADJOINT (cache won't fit), run cold
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_reference_defaults.py::test_solver_options_use_hvp_warm_start_defaults_true tests/test_genewise_hvp.py::test_batch_static_warm_v_tangent_defaults_to_none -v -m gpu`

Expected: both PASS.

- [ ] **Step 5: Confirm the TOML round-trip guard still passes**

Run: `.venv/bin/python -m pytest tests/test_config_toml.py -v`

Expected: all pass, including `test_defaults_toml_matches_dataclass_defaults` (would fail if `defaults.toml`'s value didn't match the dataclass default).

- [ ] **Step 6: Commit**

```bash
git add gpurec/api/solver_options.py gpurec/config/defaults.toml gpurec/api/_batch_state.py \
        tests/test_reference_defaults.py tests/test_genewise_hvp.py
git commit -m "Add use_hvp_warm_start config field and warm_v_tangent static cache"
```

---

### Task 2: Thread `warm_v` through `vjp_root_to_theta` and `build_point_cache`

**Files:**
- Modify: `gpurec/solver/ggn.py`
- Modify: `gpurec/solver/hvp_exact.py`
- Test: `tests/test_genewise_hvp.py`

**Interfaces:**
- Consumes: `implicit_grad_loglik_vjp_wave`'s existing `warm_v: dict | None` parameter (`gpurec/api/_implicit_grad.py`, already accepts and uses it — no change needed there).
- Produces: `vjp_root_to_theta(..., warm_v=None)` and `build_point_cache(..., warm_v=None)` — both forward-only, no gating logic yet (that's Task 3).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_genewise_hvp.py`:

```python
@pytest.mark.gpu
def test_build_point_cache_accepts_and_forwards_warm_v():
    from gpurec.solver.hvp_exact import build_point_cache

    m = build_genewise_model()
    static = m.batch_statics[0]
    F = len(m.families)
    S = int(m.species_helpers["S"])
    theta = torch.full((F, 3), math.log2(0.1), device="cuda", dtype=torch.float64)
    rw = torch.zeros(S, device="cuda", dtype=torch.float64)
    _l, sv = forward_solve([static], theta, rw)

    # warm_v=None (today's behavior) and warm_v={} (empty -- no cached entries yet) must agree,
    # since an empty dict has nothing to look up (init_v is None either way on a first call).
    g_theta_none, g_col_none, _cache_none = build_point_cache(static, theta, rw, sv, )
    g_theta_empty, g_col_empty, _cache_empty = build_point_cache(static, theta, rw, sv, warm_v={})
    torch.testing.assert_close(g_theta_none, g_theta_empty)
    torch.testing.assert_close(g_col_none, g_col_empty)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_genewise_hvp.py::test_build_point_cache_accepts_and_forwards_warm_v -v -m gpu`

Expected: FAIL with `TypeError: build_point_cache() got an unexpected keyword argument 'warm_v'`.

- [ ] **Step 3: Thread `warm_v` through `vjp_root_to_theta`**

In `gpurec/solver/ggn.py`, modify the `vjp_root_to_theta` signature and its forwarding call (currently at line ~31 and the `implicit_grad_loglik_vjp_wave(...)` call inside it):

```python
def vjp_root_to_theta(static, sv, seed_root, theta, receiver_weights, *, drop_norm=True,
                      neumann_terms=None, use_pruning=None, bicgstab_tol=None, cache=None,
                      origination_log_probs=None, origination_probs=None,
                      reserved_scratch_bytes=None, warm_v=None):
```

and in the body, add `warm_v=warm_v,` to the `implicit_grad_loglik_vjp_wave(...)` call's kwargs (alongside the existing `reserved_scratch_bytes=reserved_scratch_bytes,`).

- [ ] **Step 4: Thread `warm_v` through `build_point_cache`**

In `gpurec/solver/hvp_exact.py`, modify `build_point_cache`:

```python
@torch.no_grad()
def build_point_cache(static, theta, col_weights, sv, *, origination_log_probs=None,
                      origination_probs=None, warm_v=None):
    """Cache each wave adjoint, split likelihood, activity mask, and the E adjoint.

    Returns ``(grad_theta, grad_receiver_weights, cache)``.
    """
    static = _single_static(static)
    cache: dict = {}
    grad_theta, grad_col = vjp_root_to_theta(
        static, sv, None, theta, col_weights, drop_norm=False, cache=cache,
        origination_log_probs=origination_log_probs, origination_probs=origination_probs,
        reserved_scratch_bytes=_warm_reserved_scratch_bytes(static),
        warm_v=warm_v,
    )
    return grad_theta, grad_col, cache
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_genewise_hvp.py::test_build_point_cache_accepts_and_forwards_warm_v -v -m gpu`

Expected: PASS.

- [ ] **Step 6: Run the existing genewise HVP suite to confirm no regression**

Run: `.venv/bin/python -m pytest tests/test_genewise_hvp.py -v -m gpu`

Expected: all pass (the existing tests never pass `warm_v=`, so `build_point_cache`'s default `warm_v=None` keeps them unaffected).

- [ ] **Step 7: Commit**

```bash
git add gpurec/solver/ggn.py gpurec/solver/hvp_exact.py tests/test_genewise_hvp.py
git commit -m "Thread warm_v through vjp_root_to_theta and build_point_cache"
```

---

### Task 3: Wire config-gated point-cache warm-start into `make_exact_hvp_single`

**Files:**
- Modify: `gpurec/solver/hvp_exact.py`
- Test: `tests/test_genewise_hvp.py`

**Interfaces:**
- Consumes: `build_point_cache(..., warm_v=...)` from Task 2; `static.warm_v` (`gpurec/api/_batch_state.py`); `static.warm_adjoint_ok`; `static.solver_options.use_hvp_warm_start`.
- Produces: `make_exact_hvp`/`make_exact_hvp_single` now populate `static.warm_v` as a side effect when `use_hvp_warm_start` is `True` (the default) and the memory gate allows it — part 1 complete.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_genewise_hvp.py`:

```python
@pytest.mark.gpu
def test_genewise_point_cache_warm_start_matches_fd_across_repeated_calls():
    """Two make_exact_hvp calls at nearby theta on the same static: the second call's
    point-cache backward pass reuses static.warm_v (populated by the first call). The
    result must still match FD within the existing correctness tolerance."""
    m = build_genewise_model()
    static = m.batch_statics[0]
    F = len(m.families)
    S = int(m.species_helpers["S"])
    rw = torch.zeros(S, device="cuda", dtype=torch.float64)

    theta0 = torch.full((F, 3), math.log2(0.1), device="cuda", dtype=torch.float64)
    _l0, sv0 = forward_solve([static], theta0, rw)
    make_exact_hvp([static], theta0, rw, sv0, tangent_self_iters=128)  # populates static.warm_v
    assert static.warm_v is not None and len(static.warm_v) > 0

    theta1 = theta0 + 0.01
    _l1, sv1 = forward_solve([static], theta1, rw)
    hvp1 = make_exact_hvp([static], theta1, rw, sv1, tangent_self_iters=128)
    fd1 = _fd_hessian_hvp(make_value_and_grad([static], rw, theta_shape=(F, 3)),
                          theta1.reshape(-1).contiguous(), None, eps=1e-5)
    for j in range(3):
        u = torch.zeros(F, 3, device="cuda", dtype=torch.float64); u[:, j] = 1.0
        u = u.reshape(-1)
        Ha, Hf = hvp1(u).double(), fd1(u).double()
        rel = float((Ha - Hf).abs().max()) / max(float(Hf.abs().max()), 1e-30)
        assert torch.isfinite(Ha).all() and rel < 5e-4, f"broadcast e_{j}: rel={rel:.2e}"


@pytest.mark.gpu
def test_genewise_point_cache_warm_start_disabled_by_config():
    """use_hvp_warm_start=False -> static.warm_v is never touched."""
    so = SolverOptions(**_SO)
    so.use_hvp_warm_start = False
    so.validate()
    m = GeneReconModel(f"{_D}/sp.nwk", [f"{_D}/g.nwk"] * 2, mode="genewise",
                       device="cuda", dtype=torch.float64, solver_options=so)
    static = m.batch_statics[0]
    F = len(m.families)
    S = int(m.species_helpers["S"])
    rw = torch.zeros(S, device="cuda", dtype=torch.float64)
    theta = torch.full((F, 3), math.log2(0.1), device="cuda", dtype=torch.float64)
    _l, sv = forward_solve([static], theta, rw)
    make_exact_hvp([static], theta, rw, sv, tangent_self_iters=128)
    assert static.warm_v is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_genewise_hvp.py::test_genewise_point_cache_warm_start_matches_fd_across_repeated_calls tests/test_genewise_hvp.py::test_genewise_point_cache_warm_start_disabled_by_config -v -m gpu`

Expected: the first test fails the `assert static.warm_v is not None` line (still `None` — nothing populates it yet); the second passes vacuously today (also fine — it should keep passing after the change).

- [ ] **Step 3: Wire the gate into `make_exact_hvp_single`**

In `gpurec/solver/hvp_exact.py`, `make_exact_hvp_single` currently builds the cache like this (around line 273-276):

```python
    if cache is None:
        _, _, cache = build_point_cache(static, theta, col_weights, sv,
                                        origination_log_probs=origination_log_probs,
                                        origination_probs=origination_probs)
```

Replace with:

```python
    if cache is None:
        if getattr(static, "warm_adjoint_ok", True) and static.solver_options.use_hvp_warm_start:
            if static.warm_v is None:
                static.warm_v = {}
            _warm_v = static.warm_v
        else:
            _warm_v = None
        _, _, cache = build_point_cache(static, theta, col_weights, sv,
                                        origination_log_probs=origination_log_probs,
                                        origination_probs=origination_probs,
                                        warm_v=_warm_v)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_genewise_hvp.py::test_genewise_point_cache_warm_start_matches_fd_across_repeated_calls tests/test_genewise_hvp.py::test_genewise_point_cache_warm_start_disabled_by_config -v -m gpu`

Expected: both PASS.

- [ ] **Step 5: Run the full genewise HVP suite**

Run: `.venv/bin/python -m pytest tests/test_genewise_hvp.py -v -m gpu`

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add gpurec/solver/hvp_exact.py tests/test_genewise_hvp.py
git commit -m "Wire config-gated point-cache warm-start into make_exact_hvp_single"
```

---

### Task 4: Tangent-adjoint warm-start via `probe_id` (single-batch path)

**Files:**
- Modify: `gpurec/solver/hvp_exact.py`
- Test: `tests/test_genewise_hvp.py`

**Interfaces:**
- Consumes: `static.warm_v_tangent` (Task 1); `static.warm_adjoint_ok`; `static.solver_options.use_hvp_warm_start`; `wave["ws"]`, `wave["active_mask"]` (existing loop variables inside `make_exact_hvp_single`'s `hvp(u_vec)` closure).
- Produces: `hvp(u_vec, probe_id=None)` — the closure `make_exact_hvp_single` returns now accepts an optional `probe_id`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_genewise_hvp.py`:

```python
@pytest.mark.gpu
def test_genewise_tangent_warm_start_matches_fd():
    """probe_id passed on two nearby-theta calls: warm-started tangent-adjoint result
    still matches FD within the existing correctness tolerance."""
    m = build_genewise_model()
    static = m.batch_statics[0]
    F = len(m.families)
    S = int(m.species_helpers["S"])
    rw = torch.zeros(S, device="cuda", dtype=torch.float64)

    theta0 = torch.full((F, 3), math.log2(0.1), device="cuda", dtype=torch.float64)
    _l0, sv0 = forward_solve([static], theta0, rw)
    hvp0 = make_exact_hvp([static], theta0, rw, sv0, tangent_self_iters=128)
    for j in range(3):
        u = torch.zeros(F, 3, device="cuda", dtype=torch.float64); u[:, j] = 1.0
        hvp0(u.reshape(-1), probe_id=j)
    assert static.warm_v_tangent is not None
    assert set(static.warm_v_tangent.keys()) == {0, 1, 2}

    theta1 = theta0 + 0.01
    _l1, sv1 = forward_solve([static], theta1, rw)
    hvp1 = make_exact_hvp([static], theta1, rw, sv1, tangent_self_iters=128)
    fd1 = _fd_hessian_hvp(make_value_and_grad([static], rw, theta_shape=(F, 3)),
                          theta1.reshape(-1).contiguous(), None, eps=1e-5)
    for j in range(3):
        u = torch.zeros(F, 3, device="cuda", dtype=torch.float64); u[:, j] = 1.0
        u = u.reshape(-1)
        Ha, Hf = hvp1(u, probe_id=j).double(), fd1(u).double()
        rel = float((Ha - Hf).abs().max()) / max(float(Hf.abs().max()), 1e-30)
        assert torch.isfinite(Ha).all() and rel < 5e-4, f"warm probe_id={j}: rel={rel:.2e}"


@pytest.mark.gpu
def test_genewise_tangent_warm_start_probes_do_not_cross_contaminate():
    m = build_genewise_model()
    static = m.batch_statics[0]
    F = len(m.families)
    S = int(m.species_helpers["S"])
    rw = torch.zeros(S, device="cuda", dtype=torch.float64)
    theta = torch.full((F, 3), math.log2(0.1), device="cuda", dtype=torch.float64)
    _l, sv = forward_solve([static], theta, rw)
    hvp = make_exact_hvp([static], theta, rw, sv, tangent_self_iters=128)
    u0 = torch.zeros(F, 3, device="cuda", dtype=torch.float64); u0[:, 0] = 1.0
    u1 = torch.zeros(F, 3, device="cuda", dtype=torch.float64); u1[:, 1] = 1.0
    hvp(u0.reshape(-1), probe_id=0)
    hvp(u1.reshape(-1), probe_id=1)
    assert static.warm_v_tangent[0].keys() == static.warm_v_tangent[1].keys()
    for ws in static.warm_v_tangent[0]:
        v0 = static.warm_v_tangent[0][ws]
        v1 = static.warm_v_tangent[1][ws]
        assert not torch.allclose(v0, v1), "different probe directions must cache distinct v_k"


@pytest.mark.gpu
def test_genewise_tangent_warm_start_disabled_by_config():
    so = SolverOptions(**_SO)
    so.use_hvp_warm_start = False
    so.validate()
    m = GeneReconModel(f"{_D}/sp.nwk", [f"{_D}/g.nwk"] * 2, mode="genewise",
                       device="cuda", dtype=torch.float64, solver_options=so)
    static = m.batch_statics[0]
    F = len(m.families)
    S = int(m.species_helpers["S"])
    rw = torch.zeros(S, device="cuda", dtype=torch.float64)
    theta = torch.full((F, 3), math.log2(0.1), device="cuda", dtype=torch.float64)
    _l, sv = forward_solve([static], theta, rw)
    hvp = make_exact_hvp([static], theta, rw, sv, tangent_self_iters=128)
    u = torch.zeros(F, 3, device="cuda", dtype=torch.float64); u[:, 0] = 1.0
    hvp(u.reshape(-1), probe_id=0)
    assert static.warm_v_tangent is None or 0 not in static.warm_v_tangent
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_genewise_hvp.py::test_genewise_tangent_warm_start_matches_fd tests/test_genewise_hvp.py::test_genewise_tangent_warm_start_probes_do_not_cross_contaminate tests/test_genewise_hvp.py::test_genewise_tangent_warm_start_disabled_by_config -v -m gpu`

Expected: FAIL with `TypeError: hvp() got an unexpected keyword argument 'probe_id'`.

- [ ] **Step 3: Add `probe_id` to the `hvp(u_vec)` closure signature**

In `gpurec/solver/hvp_exact.py`, inside `make_exact_hvp_single`, change the closure definition (currently `def hvp(u_vec):` at line ~413) to:

```python
    def hvp(u_vec, probe_id=None):
```

- [ ] **Step 4: Warm-start the tangent-adjoint solve inside the wave loop**

Still inside `hvp`, the wave loop currently reads (around line 508-513):

```python
            for _wi, wave in enumerate(cache["waves"]):  # already reverse order
                if _wi % free_cache_every == 0:
                    free_cuda_cache_if_tight()
                ws, W = wave["ws"], wave["W"]
```

Right after `ws, W = wave["ws"], wave["W"]`, add the warm-start lookup:

```python
                ws, W = wave["ws"], wave["W"]
                _tangent_warm = (
                    probe_id is not None
                    and getattr(static, "warm_adjoint_ok", True)
                    and static.solver_options.use_hvp_warm_start
                )
                _probe_cache = None
                _init_v = None
                if _tangent_warm:
                    if static.warm_v_tangent is None:
                        static.warm_v_tangent = {}
                    _probe_cache = static.warm_v_tangent.setdefault(probe_id, {})
                    _init_v = _probe_cache.get(ws)
```

Then the `solve_reconciliation_wave_vjp(...)` call (currently around line 587-612) gains `initial_v=_init_v`:

```python
                ) = solve_reconciliation_wave_vjp(
                    sv["pi_wave"], sv["pibar_wave"], ws, W, S,
                    gene_split_log_likelihood, seed, wave_constants["max_transfer"],
                    wave_constants["duplication_loss_const"],
                    wave_constants["extinction_complement"], wave_constants["extinction"],
                    wave_constants["speciation_child1_const"],
                    wave_constants["speciation_child2_const"], receiver_log_probs,
                    species_child1, species_child2, None, neumann_terms=int(so.neumann_terms),
                    leaf_species_idx=leaf_species_idx,
                    leaf_logp=wave_constants["leaf_log_probability"],
                    has_leaf_term=has_leaf,
                    active_mask=wave["active_mask"], species_parent=species_parent,
                    max_ancestor_depth=max_ancestor_depth,
                    pibar_row_max=pibar_row_max, family_idx=family_idx,
                    family_indexed_consts=True,
                    compact_level_ptr=sh["compact_level_ptr"],
                    compact_level_parents=sh["compact_level_parents"],
                    compact_level_child1=sh["compact_level_child1"],
                    compact_level_child2=sh["compact_level_child2"],
                    grad_receiver_log_probs=d_grad_receiver_log_probs, use_receiver_weights=use_receiver_weights,
                    initial_v=_init_v,
                    return_last_increment=False,
                    reserved_scratch_bytes=reserved_scratch_bytes,
                    pi_offset=pi_offset,
                    pibar_offset=pibar_offset,
                    gene_split_offset=gene_split_offset,
                )
```

Immediately after that call (right after the `)` closing it, before `duplication_loss_event_vjp = (...)`), cache the freshly solved `dv` back for next time, NaN-safe against pruned/inactive rows (same sanitization backward uses):

```python
                if _tangent_warm:
                    _mask = wave.get("active_mask")
                    if _mask is not None:
                        _row_active = _mask.reshape(_mask.shape[0], -1).ne(0).any(dim=1)
                        _cached_v = torch.where(
                            _row_active.unsqueeze(-1), dv, torch.zeros((), dtype=dv.dtype, device=dv.device)
                        )
                    else:
                        _cached_v = dv
                    _probe_cache[ws] = _cached_v.detach()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_genewise_hvp.py::test_genewise_tangent_warm_start_matches_fd tests/test_genewise_hvp.py::test_genewise_tangent_warm_start_probes_do_not_cross_contaminate tests/test_genewise_hvp.py::test_genewise_tangent_warm_start_disabled_by_config -v -m gpu`

Expected: all PASS.

- [ ] **Step 6: Run the full genewise HVP suite (regression guard for the no-`probe_id` path)**

Run: `.venv/bin/python -m pytest tests/test_genewise_hvp.py -v -m gpu`

Expected: all pass, including every pre-existing test — none of them pass `probe_id=`, so this confirms the default path is unaffected.

- [ ] **Step 7: Commit**

```bash
git add gpurec/solver/hvp_exact.py tests/test_genewise_hvp.py
git commit -m "Add probe_id-keyed tangent-adjoint warm-start to make_exact_hvp_single"
```

---

### Task 5: Thread `probe_id` through the multi-batch streaming path

**Files:**
- Modify: `gpurec/solver/hvp_exact.py`
- Test: `tests/test_hvp_multibatch.py`

**Interfaces:**
- Consumes: `hvp(u_vec, probe_id=None)` from Task 4 (each batch's own `hvp_b` closure).
- Produces: `_make_exact_hvp_streaming`'s returned `hvp(u_vec, probe_id=None)` forwards `probe_id` to every batch.

**Note:** this file's module-level `rustree = pytest.importorskip("rustree")` (line 20) means every
test in it — including the new one below — SKIPS rather than runs in an environment without the
`rustree` package (confirmed absent in this session's `.venv`: `ModuleNotFoundError: No module named
'rustree'`, seen earlier making `tests/regression/test_memory_gate.py` skip the same way). Write and
commit the test regardless — it's correct and will run in CI/other environments that have `rustree`
— but don't be surprised when Step 3/5 below show `SKIPPED` instead of `PASSED`/`FAILED` here.

- [ ] **Step 1: Write the failing test**

The file's existing model-building helper (`tests/test_hvp_multibatch.py:38`) is:

```python
def _build(n_species=20, n_families=8, family_chunk_size=3, e_adjoint_solver="gmres", seed=3,
           mode="specieswise"):
```

Add to `tests/test_hvp_multibatch.py`, following the existing `test_streaming_fd_hessian_parity_genewise_theta` pattern (which already uses `_build(family_chunk_size=3, mode="genewise")`):

```python
@pytest.mark.gpu
def test_genewise_streaming_tangent_warm_start_probe_id_forwards_to_batches():
    """probe_id passed to the streaming hvp() must reach each batch's own static.warm_v_tangent."""
    m = _build(family_chunk_size=3, mode="genewise")  # n_families=8 default -> >=2 batches
    assert len(m.batch_statics) > 1
    F = len(m.families)
    S = int(m.species_helpers["S"])
    rw = torch.zeros(S, device="cuda", dtype=torch.float64)
    theta = torch.full((F, 3), math.log2(0.1), device="cuda", dtype=torch.float64)

    hvp = make_exact_hvp(m.batch_statics, theta, rw, None, tangent_self_iters=128)
    u = torch.zeros(F, 3, device="cuda", dtype=torch.float64); u[:, 0] = 1.0
    hvp(u.reshape(-1), probe_id=0)

    for static in m.batch_statics:
        assert static.warm_v_tangent is not None
        assert 0 in static.warm_v_tangent
        assert len(static.warm_v_tangent[0]) > 0
```

- [ ] **Step 2: Run test to verify it fails (or skips, per the note above)**

Run: `.venv/bin/python -m pytest tests/test_hvp_multibatch.py::test_genewise_streaming_tangent_warm_start_probe_id_forwards_to_batches -v -m gpu`

Expected: FAIL with `TypeError: hvp() got an unexpected keyword argument 'probe_id'` if `rustree` is installed; SKIPPED (`No module named 'rustree'`) otherwise — either is consistent with "not yet implemented," proceed either way.

- [ ] **Step 3: Thread `probe_id` through `_make_exact_hvp_streaming`**

In `gpurec/solver/hvp_exact.py`, `_make_exact_hvp_streaming`'s `hvp` closure (currently `def hvp(u_vec):` at line ~882) becomes:

```python
    def hvp(u_vec, probe_id=None):
```

In the non-genewise branch (the `if not genewise:` block), the call `contrib = hvp_b(u_vec)` becomes:

```python
                contrib = hvp_b(u_vec, probe_id=probe_id)
```

In the genewise branch, the call `o_b = hvp_b(torch.cat(parts) if len(parts) > 1 else u_theta_b).to(dtype=dtype)` becomes:

```python
            o_b = hvp_b(
                torch.cat(parts) if len(parts) > 1 else u_theta_b, probe_id=probe_id
            ).to(dtype=dtype)
```

- [ ] **Step 4: Run test to verify it passes (or skips)**

Run: `.venv/bin/python -m pytest tests/test_hvp_multibatch.py::test_genewise_streaming_tangent_warm_start_probe_id_forwards_to_batches -v -m gpu`

Expected: PASS if `rustree` is installed; SKIPPED otherwise (per the module-level `importorskip` noted above).

- [ ] **Step 5: Run the full multi-batch HVP suite**

Run: `.venv/bin/python -m pytest tests/test_hvp_multibatch.py -v -m gpu`

Expected: all pass or all skip together (whole-module `importorskip`), including pre-existing tests (none pass `probe_id=`, so unaffected either way).

- [ ] **Step 6: Commit**

```bash
git add gpurec/solver/hvp_exact.py tests/test_hvp_multibatch.py
git commit -m "Forward probe_id through the multi-batch streaming HVP wrapper"
```

---

### Task 6: Full regression sweep

**Files:** none (verification only)

- [ ] **Step 1: Run the complete GPU test suite**

Run: `GPUREC_PREPROCESS_PATH=$(pwd)/crates/gpurec-preprocess/target/debug/libgpurec_preprocess.so .venv/bin/python -m pytest tests/ -m gpu -q --ignore=tests/data`

Expected: same pass count as the pre-change baseline (347 passed, 10 skipped, per the last full run this session) plus the new tests from Tasks 1, 3, 4, 5 (5 new tests: `test_genewise_point_cache_warm_start_matches_fd_across_repeated_calls`, `test_genewise_point_cache_warm_start_disabled_by_config`, `test_genewise_tangent_warm_start_matches_fd`, `test_genewise_tangent_warm_start_probes_do_not_cross_contaminate`, `test_genewise_tangent_warm_start_disabled_by_config`, `test_genewise_streaming_tangent_warm_start_probe_id_forwards_to_batches`, `test_build_point_cache_accepts_and_forwards_warm_v`, `test_batch_static_warm_v_tangent_defaults_to_none`, `test_solver_options_use_hvp_warm_start_defaults_true` — 9 total). No regressions.

- [ ] **Step 2: If anything regresses, investigate before proceeding**

Pay particular attention to any test that calls `make_exact_hvp`/`build_point_cache`/`newton_joint_genewise`/`newton_polish` (`hvp_mode="exact"`) **more than once on the same `static`** and asserts an *exact* numerical value (not just "finite" or "converges") — Task 3's point-cache reuse (default-on) is the one part of this plan allowed to shift numerics within tolerance, so a golden/exact-value test hitting that repeated-call pattern is the most likely place for unexpected drift. If found, report the specific test and the magnitude of drift rather than guessing at a fix.

- [ ] **Step 3: Commit if any fixes were needed**

Only if Step 2 required changes:

```bash
git add -A
git commit -m "Fix regression found in full-suite HVP warm-start sweep"
```

---

### Task 7: Re-run the archaea60 benchmark with warm-starting enabled

**Files:**
- Modify: `/tmp/claude-1000/-home-enzo-Documents-git-gpurec-gpurec/619a320a-b59f-4ead-acca-6bf9d79bdbc8/scratchpad/fit_genewise_analytic.py` (the benchmark-only copy of `fit_genewise` built earlier this session, using the analytic HVP instead of FD)
- Run: `/tmp/claude-1000/-home-enzo-Documents-git-gpurec-gpurec/619a320a-b59f-4ead-acca-6bf9d79bdbc8/scratchpad/run_genewise_bench.py` (unchanged)

This is a measurement task, not a code-with-tests task — no assertions, just running and reporting real numbers, per the project's standing rule against speculating on performance.

- [ ] **Step 1: Add `probe_id` to the benchmark's analytic Hessian construction**

In `fit_genewise_analytic.py`'s `_analytic_hessian` helper, the three broadcast-probe calls currently look like (single-batch branch):

```python
        cols = []
        for j in range(3):
            u_b = torch.zeros(G, 3, device=dev, dtype=dtype); u_b[:, j] = 1.0
            out_b = hvp(u_b.reshape(-1))[: G * 3].reshape(G, 3)
```

Change the `hvp(...)` call to pass the probe identity:

```python
            out_b = hvp(u_b.reshape(-1), probe_id=j)[: G * 3].reshape(G, 3)
```

Do the same in the multi-batch branch (the `if len(m.batch_statics) > 1:` block), where `hvp(u.reshape(-1))` becomes `hvp(u.reshape(-1), probe_id=j)`.

- [ ] **Step 2: Re-run the 200-family sanity check (same as the earlier session's baseline)**

Run:
```bash
GPUREC_PREPROCESS_PATH=$(pwd)/crates/gpurec-preprocess/target/debug/libgpurec_preprocess.so \
GPUREC_ARCHAEA_ROOT=$(pwd)/../gpurec-data/benchmarks/large_dataset_capacity/datasets/alerax_archaea_davin2017 \
N_FAM=200 CERTIFY=1 \
.venv/bin/python /tmp/claude-1000/-home-enzo-Documents-git-gpurec-gpurec/619a320a-b59f-4ead-acca-6bf9d79bdbc8/scratchpad/run_genewise_bench.py
```

Expected: both paths still converge 200/200 with matching NLL (as before: 20968.031 vs 20968.032 bits). Compare the analytic path's new wall-clock against the earlier unwarmed baseline (32.5s) — report the actual number, do not predict it.

- [ ] **Step 3: Re-run the full 5446-family archaea60 comparison**

Run (background, as in the earlier session — this takes several minutes):
```bash
GPUREC_PREPROCESS_PATH=$(pwd)/crates/gpurec-preprocess/target/debug/libgpurec_preprocess.so \
GPUREC_ARCHAEA_ROOT=$(pwd)/../gpurec-data/benchmarks/large_dataset_capacity/datasets/alerax_archaea_davin2017 \
N_FAM=all CERTIFY=1 \
.venv/bin/python /tmp/claude-1000/-home-enzo-Documents-git-gpurec-gpurec/619a320a-b59f-4ead-acca-6bf9d79bdbc8/scratchpad/run_genewise_bench.py
```

Expected: report the actual measured wall-clock for both paths (`fd.wall`, `analytic.wall` from the printed JSON) and compare against the earlier session's baseline (FD 340.2s, analytic-unwarmed 375.4s). State plainly whether warm-starting closed the gap, and by how much — measured, not predicted.

- [ ] **Step 4: Report results**

Summarize: correctness (still matches FD within tolerance, same convergence profile), and the measured wall-clock delta from warm-starting at both scales. This is the deciding data for whether analytic HVP is worth wiring into `fit_genewise` for real — a decision explicitly out of scope for this plan (per the spec's non-goals).
