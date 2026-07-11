# Specieswise Fit Recipe Organization — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give specieswise a canonical single-prior MAP fit tool (`fit_specieswise`), route the MAP+CV experiment and the benchmark through it, and stop `fit_dtl` from returning an ill-posed raw-MLE specieswise result.

**Architecture:** Three layers. `fit_specieswise` (new) = one MAP fit at a GIVEN `lam` using ~10 Adam warm-up steps then the saddle-aware `newton_lanczos` (negative-curvature CG witness) with the ridge/MAP term. `map_cv` (existing) chooses `lam` and calls `fit_specieswise` per fold. `fit_dtl` raises for specieswise, pointing to those tools. Benchmark fits specieswise via `fit_specieswise` at a fixed committed `lam`.

**Tech Stack:** Python, PyTorch, gpurec (`newton_lanczos`, `make_value_and_grad` with `prior=`, `final_eval`, `Schedule`), pytest.

## Global Constraints

- Branch `dev`. Commit after each task; do not push.
- Run tests with `/home/enzo/Documents/git/gpurec/gpurec/.venv/bin/python -m pytest`. CUDA is available.
- No `clamp`/`clip`/`min`/`max` to MASK numerical values (Levenberg eigenvalue floors / trust caps inside `newton_lanczos` are pre-existing algorithm internals, not touched here).
- `theta` layout is `[log2 D, log2 L, log2 T]`; `rates = 2**theta`. Log-space is log2.
- `make_value_and_grad(...)` returns `f(theta_vec, *, warm_E=None, want_grad=True) -> (loss, g_vec, saved, warm_E_out)`. `prior=(lam, theta_ref)` adds `(lam/2)||theta - theta_ref||^2`; `theta_ref` is flattened internally, so any shape matching `theta` is fine.
- `newton_lanczos(static, theta0, receiver_weights, *, ..., gtol=1e-2, max_newton=40, lam=0.0, theta_ref=None, hvp_mode="fd", verbose=True) -> (theta_hat[theta0.shape], history)`; `history[-1]["gnorm"]` is the MAP projected-gradient norm. Use `hvp_mode="exact"` for specieswise `[S,3]` theta.
- Design spec: `docs/superpowers/specs/2026-07-11-specieswise-recipe-organization-design.md`.

---

### Task 1: `fit_specieswise` — the canonical single-prior MAP tool

**Files:**
- Create: `gpurec/fit/specieswise_fit.py`
- Test: `tests/regression/test_specieswise_recipe.py`

**Interfaces:**
- Consumes: `newton_lanczos` (gpurec/fit/newton_cg.py), `make_value_and_grad` (gpurec/solver/value_and_grad.py), `final_eval` + `Schedule` (gpurec/fit/optimize.py).
- Produces: `fit_specieswise(batch_statics, theta0, receiver_weights, *, lam, theta_ref=None, adam_steps=10, adam_lr=1.0, max_newton=8, gtol=1e-2, lanczos_m=10, sigma=0.01, verbose=False) -> dict` with keys `{mode, theta[cpu S,3], rates[cpu S,3], nll_bits, nll_nats, gnorm, lam, wall_s}`.

- [ ] **Step 1: Write the failing test**

Add to `tests/regression/test_specieswise_recipe.py`:

```python
import pytest

pytest.importorskip("rustree")
torch = pytest.importorskip("torch")


def test_fit_specieswise_requires_lam():
    from gpurec.fit.specieswise_fit import fit_specieswise
    with pytest.raises(ValueError, match="requires an explicit prior"):
        # lam is keyword-only with no default -> omitting it is a TypeError; passing None -> ValueError
        fit_specieswise("bs", "th", "rw", lam=None)


@pytest.mark.gpu
def test_fit_specieswise_fits_at_given_lam(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    from gpurec.bench.simulate import simulate_dataset
    from gpurec.api.model import GeneReconModel
    from gpurec.api.solver_options import SolverOptions
    from gpurec.fit.specieswise_fit import fit_specieswise
    import numpy as np

    sp, genes = simulate_dataset("specieswise", tmp_path, n_species=60, n_families=80,
                                 dtl=0.05, seed=5)
    m = GeneReconModel(sp, genes, mode="specieswise", device="cuda", dtype=torch.float32,
                       solver_options=SolverOptions(e_adjoint_solver="neumann"))
    S = m.theta.shape[0]
    res = fit_specieswise(m.batch_statics, m.theta.detach(), m.receiver_weights.detach(),
                          lam=10.0, verbose=False)
    rates = np.asarray(res["rates"])
    assert res["mode"] == "specieswise" and res["lam"] == 10.0
    assert rates.shape == (S, 3) and np.isfinite(res["nll_bits"]) and np.isfinite(rates).all()
    assert res["gnorm"] < 1.0, f"MAP not converged: |gF|={res['gnorm']}"

    # ridge sanity: a huge lam pins the optimum at theta_ref (a constant reference).
    theta_ref = torch.full((S, 3), float(np.log2(0.1)), device="cuda")
    res_big = fit_specieswise(m.batch_statics, m.theta.detach(), m.receiver_weights.detach(),
                              lam=1e6, theta_ref=theta_ref, verbose=False)
    drift = (res_big["theta"] - theta_ref.cpu()).abs().max().item()
    assert drift < 1e-1, f"large lam should hold theta near theta_ref, drift={drift}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/regression/test_specieswise_recipe.py::test_fit_specieswise_requires_lam -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'gpurec.fit.specieswise_fit'`.

- [ ] **Step 3: Write minimal implementation**

Create `gpurec/fit/specieswise_fit.py`:

```python
"""Canonical specieswise DTL fit: a SINGLE MAP fit at a GIVEN prior precision ``lam``.

Specieswise raw MLE over ``theta[S,3]`` is ill-posed -- non-identifiable, with ~half the per-state
rates saturating the 0/1 boundary (flat near-zero Hessian eigenvalues) and index-1 saddles
(``lam_min < 0``). See docs/superpowers/specs/2026-07-11-specieswise-recipe-organization-design.md and
kernel-bench/newton/_specieswise_basin_findings.md. The well-posed target is a MAP fit with a
Gaussian/ridge prior ``(lam/2)||theta - theta_ref||^2`` whose precision ``lam`` is chosen by
cross-validation (:func:`gpurec.fit.map_cv.map_cv`).

This module is the per-fit TOOL: it fits ONE prior. It does NOT choose ``lam`` and does NOT
cross-validate -- ``lam`` is a required argument, the explicit "you chose this" signal; ``lam=0.0`` is
the raw MLE and is intentionally never a default. Recipe: ~10 Adam warm-up steps (basin entry) on the
MAP objective, then the saddle-aware ``newton_lanczos`` (CG negative-curvature witness) with the ridge
term. At ``lam > |lam_min|`` the MAP Hessian ``H + lam*I`` is PD, restoring a quadratic endgame.
"""
from __future__ import annotations

import time

import torch

from gpurec.fit.newton_cg import newton_lanczos
from gpurec.fit.optimize import Schedule, final_eval
from gpurec.solver.value_and_grad import make_value_and_grad

_LN2 = 0.6931471805599453


def fit_specieswise(batch_statics, theta0, receiver_weights, *, lam, theta_ref=None,
                    adam_steps=10, adam_lr=1.0, max_newton=8, gtol=1e-2, lanczos_m=10,
                    sigma=0.01, verbose=False) -> dict:
    """Single MAP fit of specieswise theta[S,3] at prior precision ``lam``. See module docstring."""
    if lam is None:
        raise ValueError(
            "fit_specieswise requires an explicit prior precision `lam` -- the specieswise raw MLE "
            "is ill-posed. Choose lam by cross-validation (gpurec.fit.map_cv.map_cv) and pass it "
            "here; lam=0.0 is the raw MLE and is intentionally not a default."
        )
    theta_shape = tuple(theta0.shape)
    theta = theta0.detach().reshape(theta_shape).float().contiguous().clone()
    if theta_ref is None:
        theta_ref = theta.clone()
    theta_ref = theta_ref.detach().reshape(theta_shape).float().to(theta.device)
    t0 = time.perf_counter()

    # 1. Adam warm-up on the MAP objective (prior-enabled value-and-grad), mirroring map_cv.fit_map.
    f = make_value_and_grad(batch_statics, receiver_weights, theta_shape=theta_shape,
                            prior=(float(lam), theta_ref))
    if adam_steps > 0:
        leaf = theta.clone().requires_grad_(True)
        opt = torch.optim.Adam([leaf], lr=adam_lr)
        sched = Schedule("adaptive", adam_lr, t_max=adam_steps)
        warm = None
        for _ in range(int(adam_steps)):
            loss, g, _sv, warm = f(leaf.detach().reshape(-1), warm_E=warm)
            opt.param_groups[0]["lr"] = sched.update(loss, g)
            leaf.grad = g.reshape(theta_shape)
            opt.step()
        theta = leaf.detach().reshape(theta_shape).contiguous()

    # 2. saddle-aware Newton with the ridge/MAP term (exact HVP for specieswise theta[S,3]).
    theta_hat, hist = newton_lanczos(
        batch_statics, theta, receiver_weights, hvp_mode="exact", lam=float(lam),
        theta_ref=theta_ref, lanczos_m=lanczos_m, sigma=sigma, max_newton=max_newton,
        gtol=gtol, verbose=verbose)
    gnorm = float(hist[-1]["gnorm"]) if hist else float("nan")  # MAP projected-gradient norm

    # data NLL (excludes the ridge) via the fair fp64 eval -> comparable across modes and lam.
    nll_bits, _g = final_eval(batch_statics, theta_hat, receiver_weights)
    nll_bits = float(nll_bits)
    wall_s = time.perf_counter() - t0
    return {"mode": "specieswise", "theta": theta_hat.detach().cpu(),
            "rates": (2.0 ** theta_hat.detach().float().cpu()),
            "nll_bits": nll_bits, "nll_nats": nll_bits * _LN2, "gnorm": gnorm,
            "lam": float(lam), "wall_s": wall_s}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/regression/test_specieswise_recipe.py -v`
Expected: `test_fit_specieswise_requires_lam` PASS; `test_fit_specieswise_fits_at_given_lam` PASS (needs CUDA, ~30-60s).

- [ ] **Step 5: Commit**

```bash
git add gpurec/fit/specieswise_fit.py tests/regression/test_specieswise_recipe.py
git commit -m "feat(fit): fit_specieswise -- single-prior MAP fit via saddle-aware Newton (specieswise tool)"
```

---

### Task 2: `fit_dtl` raises for specieswise

**Files:**
- Modify: `gpurec/fit/dtl_fit.py`
- Test: `tests/regression/test_specieswise_recipe.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `fit_dtl(..., mode="specieswise")` raises `NotImplementedError` with a message naming `fit_specieswise` and `map_cv`, BEFORE resolving gene trees or building any model.

- [ ] **Step 1: Write the failing test**

Add to `tests/regression/test_specieswise_recipe.py`:

```python
def test_fit_dtl_raises_for_specieswise():
    from gpurec.fit.dtl_fit import fit_dtl
    with pytest.raises(NotImplementedError, match="fit_specieswise"):
        # must raise on mode alone -- no model build, no CUDA, dummy paths are never touched.
        fit_dtl("sp.nwk", ["g.nwk"], "specieswise")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/regression/test_specieswise_recipe.py::test_fit_dtl_raises_for_specieswise -v`
Expected: FAIL — currently `fit_dtl` tries to resolve `["g.nwk"]` / build a model (an unrelated error), not the `NotImplementedError`.

- [ ] **Step 3: Add the raise**

In `gpurec/fit/dtl_fit.py`, immediately after the `if mode == "global":` block and BEFORE the `# specieswise (coupled)` model build, insert:

```python
    if mode == "specieswise":
        raise NotImplementedError(
            "specieswise has no well-posed one-shot fit: the raw MLE over theta[S,3] is "
            "non-identifiable and boundary-saturated. Fit a single MAP prior with "
            "gpurec.fit.specieswise_fit.fit_specieswise(model.batch_statics, theta0, rw, lam=<chosen>), "
            "or cross-validate the prior with gpurec.fit.map_cv.map_cv(species, genes)."
        )
```

Then update the module docstring's specieswise bullet to state it is fit by MAP+CV (not `optimize`) and that `fit_dtl` raises for it. Remove the now-dead specieswise path (the `# specieswise (coupled)` block that builds a model + runs `optimize`) — global no longer falls through to it either (global returns above), so that trailing block only served specieswise. Keep the `optimize`/`final_eval`/`GeneReconModel`/`SolverOptions`/`_resolve_gene_trees` imports ONLY if still used; if the block's removal makes any import unused, delete that import.

- [ ] **Step 4: Verify unused imports are cleaned**

Run: `.venv/bin/python -c "import ast,sys; src=open('gpurec/fit/dtl_fit.py').read(); ast.parse(src); print('parses')"` then visually confirm no remaining reference to a removed symbol (`grep -n 'optimize(\|final_eval\|GeneReconModel\|_resolve_gene_trees\|SolverOptions' gpurec/fit/dtl_fit.py`). Any import with zero references must be removed.

- [ ] **Step 5: Run tests**

Run: `.venv/bin/python -m pytest tests/regression/test_specieswise_recipe.py -v -m "not gpu"`
Expected: `test_fit_dtl_raises_for_specieswise` and `test_fit_specieswise_requires_lam` PASS.

- [ ] **Step 6: Commit**

```bash
git add gpurec/fit/dtl_fit.py tests/regression/test_specieswise_recipe.py
git commit -m "feat(fit): fit_dtl raises for specieswise (no plug-and-play raw MLE); point to fit_specieswise/map_cv"
```

---

### Task 3: Rewire `map_cv` per-fold worker to `fit_specieswise`

**Files:**
- Modify: `gpurec/fit/map_cv.py`
- Test: `tests/regression/test_specieswise_recipe.py`

**Interfaces:**
- Consumes: `fit_specieswise` (Task 1).
- Produces: `map_cv(...)` unchanged public signature/return; internally each fold's fit is `fit_specieswise` (saddle-aware Newton) instead of `fit_map` (L-BFGS). The final all-families refit also uses `fit_specieswise`.

- [ ] **Step 1: Write the failing test**

Add to `tests/regression/test_specieswise_recipe.py`:

```python
@pytest.mark.gpu
def test_map_cv_smoke_uses_fit_specieswise(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    import inspect
    from gpurec.fit import map_cv as mod
    # the rewire: map_cv's body calls fit_specieswise (not the removed L-BFGS fit_map path).
    assert "fit_specieswise" in inspect.getsource(mod.map_cv)

    from gpurec.bench.simulate import simulate_dataset
    sp, genes = simulate_dataset("specieswise", tmp_path, n_species=40, n_families=60,
                                 dtl=0.05, seed=3)
    out = mod.map_cv(sp, genes, k=2, lambdas=(1.0, 100.0), adam_steps=5, max_newton=4)
    import math
    assert out["lam_star"] in (1.0, 100.0)
    assert all(math.isfinite(v) for v in out["cv"].values())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest "tests/regression/test_specieswise_recipe.py::test_map_cv_smoke_uses_fit_specieswise" -v`
Expected: FAIL on the `inspect.getsource` assertion (`map_cv` still calls `fit_map`), or on unknown kwargs `adam_steps`/`max_newton` if the signature isn't threaded yet.

- [ ] **Step 3: Rewire the per-fold and refit calls**

In `gpurec/fit/map_cv.py`:
1. Add `from gpurec.fit.specieswise_fit import fit_specieswise` at the top with the other imports.
2. Replace each `fit_map(batch_statics, theta0, receiver_weights, lam=..., theta_ref=...)` call (per fold, and the final all-families refit) with:
   ```python
   theta_hat = fit_specieswise(batch_statics, theta0, receiver_weights, lam=lam,
                               theta_ref=theta_ref, adam_steps=adam_steps,
                               max_newton=max_newton, verbose=False)["theta"].to(theta0.device)
   ```
   (`fit_specieswise` returns a dict; take `["theta"]` and move it back to the working device. The old `fit_map` returned a bare tensor.)
3. Thread `adam_steps` and `max_newton` through `map_cv`'s signature (defaults `adam_steps=10`, `max_newton=8`) and its `MAP_CV_REFERENCE` dict, replacing the L-BFGS-specific `lbfgs_iters`/`maxcor` knobs (remove them if now unused).
4. Delete the now-unused `fit_map` function and any import it alone used (`scipy.optimize.minimize`, `Schedule` if unused elsewhere). Keep `heldout_nll`, `kfold_indices`, `_build`, and the CV masking logic untouched.

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest "tests/regression/test_specieswise_recipe.py::test_map_cv_smoke_uses_fit_specieswise" -v`
Expected: PASS (a 2-fold, 2-lambda CV on a 40-species toy completes with a finite CV curve; ~1-3 min on CUDA).

- [ ] **Step 5: Commit**

```bash
git add gpurec/fit/map_cv.py tests/regression/test_specieswise_recipe.py
git commit -m "refactor(fit): map_cv per-fold worker -> fit_specieswise (saddle-aware Newton, drop inline L-BFGS fit_map)"
```

---

### Task 4: Benchmark fits specieswise via `fit_specieswise` at a fixed committed `lam`

**Files:**
- Modify: `tests/regression/mint_goldens.py`
- Modify: `gpurec/bench/simulate.py` (add the pinned constant next to `SIM_PARAMS`)

**Interfaces:**
- Consumes: `fit_specieswise` (Task 1); `SPECIESWISE_GOLDEN_LAM` (new constant).
- Produces: `mint_goldens.fit_mode("specieswise", sp, genes)` returns `(nll_bits, rates[S,3], wall_s)` from a `fit_specieswise` fit at `SPECIESWISE_GOLDEN_LAM`; global/genewise still route through `fit_dtl`.

- [ ] **Step 1: Add the pinned constant**

In `gpurec/bench/simulate.py`, immediately after the `SIM_PARAMS = {...}` dict, add:

```python
# Specieswise has no well-posed one-shot MLE, so its perf-golden fits fit_specieswise at a FIXED,
# committed prior precision (a deterministic recipe test at a pinned prior; a full map_cv is too
# expensive/noisy for a golden). See docs/.../2026-07-11-specieswise-recipe-organization-design.md.
SPECIESWISE_GOLDEN_LAM = 10.0
```

- [ ] **Step 2: Route specieswise in `fit_mode`**

In `tests/regression/mint_goldens.py`, replace the body of `fit_mode` so specieswise builds a model and calls `fit_specieswise`, while global/genewise keep going through `fit_dtl`:

```python
def fit_mode(mode, species_path, gene_paths, *, verbose=False):
    """Fit one mode via its production recipe; returns (nll_bits, rates[.,3] D,L,T, wall_s).

    global/genewise are plug-and-play through fit_dtl. specieswise has no one-shot MLE, so it is fit
    by fit_specieswise at the committed SPECIESWISE_GOLDEN_LAM prior (see the design spec)."""
    if mode == "specieswise":
        import torch
        from gpurec.api.model import GeneReconModel
        from gpurec.api.solver_options import SolverOptions
        from gpurec.fit.specieswise_fit import fit_specieswise
        from gpurec.bench.simulate import SPECIESWISE_GOLDEN_LAM
        model = GeneReconModel(species_path, gene_paths, mode="specieswise", device="cuda",
                               dtype=torch.float32,
                               solver_options=SolverOptions(e_adjoint_solver="neumann"))
        res = fit_specieswise(model.batch_statics, model.theta.detach(),
                              model.receiver_weights.detach(), lam=SPECIESWISE_GOLDEN_LAM,
                              verbose=verbose)
        rates = np.asarray(res["rates"])
        return float(res["nll_bits"]), rates, float(res["wall_s"])
    res = fit_dtl(species_path, gene_paths, mode, device="cuda", dtype=torch.float32, verbose=verbose)
    rates = np.asarray(res["rates"])
    return float(res["nll_bits"]), rates, float(res["wall_s"])
```

Ensure `mint_goldens.py` still imports `fit_dtl` (for global/genewise) and `np`. The specieswise-specific imports are function-local (above) to avoid importing torch at module import for the fast tests.

- [ ] **Step 3: Record the pinned lam in provenance**

In `mint_goldens.py`'s `mint(...)`, add `"specieswise_golden_lam"` to the golden `provenance` dict only for the specieswise mode (or unconditionally, reading it from `gpurec.bench.simulate.SPECIESWISE_GOLDEN_LAM`), so the golden self-documents the prior it was minted at:

```python
        "provenance": {
            ...,
            "specieswise_golden_lam": __import__("gpurec.bench.simulate",
                                                 fromlist=["SPECIESWISE_GOLDEN_LAM"]).SPECIESWISE_GOLDEN_LAM,
        },
```

- [ ] **Step 4: Verify the wiring compiles and specieswise no longer touches fit_dtl**

Run:
```bash
.venv/bin/python -c "import ast; ast.parse(open('tests/regression/mint_goldens.py').read()); ast.parse(open('gpurec/bench/simulate.py').read()); print('parse OK')"
.venv/bin/python -c "from gpurec.bench.simulate import SPECIESWISE_GOLDEN_LAM; print('lam', SPECIESWISE_GOLDEN_LAM)"
```
Expected: `parse OK` and `lam 10.0`.

- [ ] **Step 5: Manual GPU verification (small, not committed)**

Run a small specieswise fit through `fit_mode` to confirm the route returns finite results (write to the scratchpad, not the repo):
```bash
.venv/bin/python -c "
import tempfile, numpy as np, torch
from pathlib import Path
from gpurec.bench.simulate import simulate_dataset
from tests.regression.mint_goldens import fit_mode
d=tempfile.mkdtemp(); sp,g=simulate_dataset('specieswise',Path(d)/'s',n_species=60,n_families=80,dtl=0.05,seed=5)
nll,rates,wall=fit_mode('specieswise',sp,g); print('nll',nll,'shape',np.asarray(rates).shape,'wall',wall)
assert np.isfinite(nll) and np.isfinite(rates).all()
print('SPECIESWISE fit_mode OK')
"
```
Expected: prints finite `nll`, shape `(119, 3)`, `SPECIESWISE fit_mode OK`.

- [ ] **Step 6: Commit**

```bash
git add tests/regression/mint_goldens.py gpurec/bench/simulate.py
git commit -m "feat(bench): specieswise perf-golden fits fit_specieswise at fixed SPECIESWISE_GOLDEN_LAM"
```

---

### Task 5: Re-mint the specieswise golden (run step)

**Files:**
- Modify (generated): `tests/regression/goldens/specieswise.json`

- [ ] **Step 1: Re-mint specieswise only**

Run (≈ minutes at 500 leaves; requires the go-ahead per the no-long-benchmarks rule):
```bash
.venv/bin/python tests/regression/mint_goldens.py --mode specieswise --repeats 1 --verbose
```

- [ ] **Step 2: Confirm the golden updated**

Run: `.venv/bin/python -c "import json; g=json.load(open('tests/regression/goldens/specieswise.json')); print(g['nll'], g['provenance'].get('specieswise_golden_lam'), g['recorded_wall_s'])"`
Expected: a finite NLL, `specieswise_golden_lam` = 10.0, and a recorded wall time.

- [ ] **Step 3: Commit**

```bash
git add tests/regression/goldens/specieswise.json
git commit -m "test(bench): re-mint specieswise golden under fit_specieswise at lam=10"
```

---

## Self-Review notes

- **Spec coverage:** fit_specieswise tool (Task 1), map_cv rewire (Task 3), fit_dtl raise (Task 2), benchmark fixed-lam (Task 4), re-mint (Task 5), tests (`test_specieswise_recipe.py` across Tasks 1-3). `map_fit.py` untouched (non-goal). Return dict `nll_bits`=data NLL via `final_eval`, `gnorm`=MAP projected gradient — matches spec.
- **Type consistency:** `fit_specieswise` returns a dict everywhere; `map_cv` takes `["theta"]` from it (Task 3). `SPECIESWISE_GOLDEN_LAM` defined in Task 4 Step 1, consumed in the same task and Task 5.
- **Ordering:** Task 1 → (2, 3, 4 depend on 1) → 5 depends on all. Task 2 is independent of 1 but cheap; keep after 1 so the raise message can reference the real symbol.
