# gpurec Centralized Config (`GpurecConfig`) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Replace the ~120 scattered numeric literals across `gpurec` with a single, composable, TOML-loadable `GpurecConfig`, eliminating duplication and the latent inconsistencies it has already produced — without changing any numerical *result* except the handful of deliberate inconsistency-resolutions listed in Global Constraints.

**Architecture:** Evolve the existing `SolverOptions` dataclass pattern into a composed tree — `GpurecConfig{ solver, newton, rates, regularizer, memory }` — where every tunable number lives in exactly one dataclass field. Scattered literals are rewired to *read from* the config. A hand-written `defaults.toml` plus a `tomllib`-based loader lets all numbers be edited/version-controlled in one file; dataclass defaults remain the source of truth, guarded by a test asserting `GpurecConfig() == GpurecConfig.from_toml(defaults.toml)`. The three recipe dicts (`GENEWISE_REFERENCE`, …) become `GpurecConfig` factory classmethods.

**Tech Stack:** Python 3.12 (`tomllib` stdlib, read-only TOML), `dataclasses`, existing torch/triton runtime.

## Global Constraints

- **No result changes except the four deliberate resolutions below.** Every other literal move must be value-identical. Verify per phase against the golden gate (`tests/test_optim_golden.py`, grad rel-L2 ≤ 2e-3, forward cross-parity 1e-6) and the full suite (145 tests, 0 collection errors).
- **Deliberate inconsistency resolutions (the only intended behavior changes):**
  1. **E-solve tolerance split** — forward `e_tol=1e-8` (`core/kernels/e_step.py:453`) vs tangent `1e-9` (`e_step_tangent.py:188`). Resolution: make both explicit fields — `solver.e_tol=1e-8`, `solver.e_tangent_tol=1e-9` — no value change, just naming. (Both currently below fp32 eps; unchanged.)
  2. **Rate floor split** — `1e-10` (`optimization.py`, `model.py:75` init) vs `1e-6`/`max_rate=2.0` (`fit_genewise` signature only, absent from `GENEWISE_REFERENCE`). Resolution: `rates` defaults to the global `(min_rate=1e-10, max_rate=None, init_rate=1e-10)`; the **genewise preset** overrides to `(1e-6, 2.0)`. No value change — the genewise cap becomes an explicit preset field instead of a hidden signature default.
  3. **Divergent solver signature defaults** — `_bicgstab max_iter=500` (`_implicit_grad.py:62`) vs `SolverOptions.bicgstab_max_iter=128`; `implicit_grad_loglik_vjp_wave neumann_terms=3` (`:314`) vs `SolverOptions.neumann_terms=64`. Production threads `solver_options`, so these bite only direct/test calls. Resolution: signature defaults become `None` → fall back to `SolverOptions()` field. Confirm no golden/suite change.
  4. **Duplicated dtype-tol helper** — `_bicgstab_rel_tol_default`/`_floor` (`_implicit_grad.py:42,52`) re-hardcoded in `forward_tangent._default_tol` (`:99`). Resolution: one shared helper; both call it. Value-identical.
- **Preserve the `test_reference_defaults.py` invariant** (reference-dict entry == fn signature default) OR migrate it to assert the new preset factories match. Never silently drop it.
- **Backward compatibility:** keep `SolverOptions`, `OriginationPenalty`, and the three `*_REFERENCE` names importable from their current modules (re-export). Existing scripts that build a `SolverOptions` must keep working.
- **Non-goal (do NOT do):** parametrizing Triton kernel launch knobs (`BLOCK_S` 512/256, `num_warps` 8, tile factors). They are derived speed/accuracy caps; leave as-is. Also leave `-inf` sentinels, `log2` bases, array dims, indices.
- **Source of truth = dataclass defaults.** `defaults.toml` is checked against them by test, never the reverse.

---

## Config Inventory (authoritative site list for the wiring tasks)

Grouped from the three-subsystem survey. "→ field" = the `GpurecConfig` field it maps to.

### Group SOLVER (exists as `SolverOptions`; fix the scattered duplicates)
- `SolverOptions` fields (keep): `e_max_iter=128, e_tol=1e-8, pi_iters=64, neumann_terms=64, bicgstab_max_iter=128, bicgstab_tol=None, bicgstab_breakdown_tol=None, adjoint_pruning_threshold=1e-6, pibar_side_threshold=0.0`. **Add:** `e_tangent_tol=1e-9`.
- Scattered duplicates to rewire → `solver`:
  - `core/kernels/e_step.py:452-453` `max_iter=128, tol=1e-8` (signature fallback)
  - `core/kernels/e_step_tangent.py:188` `max_iter=128, tol=1e-9`
  - `core/inference/forward.py:22` `pi_iters=6` (fallback; real value from options)
  - `api/_implicit_grad.py:62` `_bicgstab max_iter=500` (→ divergent, see constraint 3)
  - `api/_implicit_grad.py:194` `_gmres max_iter=128`
  - `api/_implicit_grad.py:314,316,319,321` `neumann_terms=3, bicgstab_max_iter=128, adjoint_pruning_threshold=1e-6, pibar_side_threshold=0.0`
  - `api/_implicit_grad.py:624` `bicgstab_max_iter=128`
  - `solver/forward_tangent.py:99-101,151` `_default_tol` consts + `self_max_iter=200`
  - `solver/ggn.py:67` `self_max_iter=200`
  - Dtype-tol helper: `api/_implicit_grad.py:42,52` (1e-6/1e-12, 4·eps) — canonical home.

### Group NEWTON (NEW `NewtonOptions`; the 4×-copy-pasted set)
Identical defaults in `solver/curvature.py:74-77`, `solver/genewise_curvature.py:220-223`, `solver/origination_curvature.py:181-184`, `solver/receiver_curvature.py:265-268`, and echoed in `fit/optimize.py`/`fit/newton_cg.py`:
- `sigma=0.01, sigma_floor=1e-4, lanczos_m=10, nu=1.5, decrease=1.5, max_bumps=3, max_cg=40, c1=1e-4, ls_max=25, gtol=1e-2, max_newton=40, ftol=1e-9, seed=0`
- Inline (curvature.py): `lam_ceil=10.0·lam_max (:95)`, `lam_max floor 1e-12 (:92)`, `certify_min m=200 (:47)`, `certify Lanczos min(20,p) (:60)`, gauge shift `2.0 (:61)`, forcing `eta=0.1 (:119)`, accept bump `1.5 (:154)`, stall count `2 (:157)`, ls-fail bump `4.0 (:166)`, `min_free_gib=8.0 (:115)`
- CG primitives (`solver/cg.py`): beta-breakdown `1e-12 (:45)`, `lanczos_extremes m=40 (:52)`, `lanczos_min_eigpair m=120 (:68)`, `cg_solve` floors `1e-30/1e-12 (:189)`
- Curvature consumers: `certify_joint_min m=200`, genewise `cert_m=120`, `cg_tol=1e-7`, `cg_max=400` (receiver/origination_curvature)
- **Conflict to preserve as two fields:** `fd_eps=1e-2` (genewise 3×3 FD) vs `fd_eps=1e-5` (full-HVP FD) — different scale conventions; `NewtonOptions.fd_eps_blockwise=1e-2`, `fd_eps_hvp=1e-5`.

### Group RATES (NEW `RateBounds`)
- `optimization.py:9,34,59` `min_rate=1e-10`, `max_rate=None`; `model.py:75` init `log2(1e-10)`; `fit/genewise_fit.py:80-81` `min_rate=1e-6, max_rate=2.0`; bound-active eps `1e-6` (`genewise_fit.py:247,281`); reconcile CLI `1e-10` (`cli/reconcile.py:12-14`).

### Group REGULARIZER (extend around `OriginationPenalty`)
- `solver/penalties.py:27` `tv_eps=1e-3` (dup at `solver/value_and_grad.py:162,165`); `OriginationPenalty` fields (`penalties.py:96-105`, already a dataclass); ridge `lambdas`, `init_rate=0.1` (`fit/map_cv.py`); `lam_margin=1.3, lam_floor=1e-3` (`fit/map_fit.py:87`).

### Group MEMORY (NEW `MemoryOptions`)
- `core/memory_policy.py:44-45` `fraction=0.85, reserve_gib=1.0` (already env-overridable); `scratch_tensors=10 (:60)`; `solver/value_and_grad.py:40` `min_free_gib=4.0`; `solver/curvature.py:115` `min_free_gib=8.0`; `solver/hvp_exact.py:150` cache cadence `32`; `grad_avg_K=1 (value_and_grad.py:89)`.

### Group FIT-RECIPE (stay dual-surfaced; become preset factories in Phase 3)
- `GENEWISE_REFERENCE` (15 values), `OPTIMIZE_REFERENCE` (3), `MAP_CV_REFERENCE` (7) + `_BASE_SOLVER`/`_DEFAULT_SO`/`_CV_SO` solver dicts; ~60 signature-only Newton kwargs (now sourced from `NewtonOptions`); scattered LM/TR body constants stay inline unless they duplicate a `NewtonOptions` field.

### Group MODEL/CLI (batching + diagnostics — mostly already ctor kwargs)
- `api/model.py:40,41,46` `family_chunk_size=300, clade_budget=315_000, max_wave_size=8192` (ctor kwargs — leave, optionally reference config); `pi_iters_high=400`, classify `fwd_tol/bwd_tol=1e-3, overshoot_tol=1e3`, tier scaling floors `32/64`; CLI `_common.py:20-22` mirror `SolverOptions`.

---

## Package layout (created in Phase 1, Task 1)

```
gpurec/config/
  __init__.py         # re-exports GpurecConfig, SolverOptions, NewtonOptions, RateBounds,
                      #   MemoryOptions, PenaltyOptions, load_config
  newton.py           # NewtonOptions
  rates.py            # RateBounds
  memory.py           # MemoryOptions
  gpurec_config.py    # GpurecConfig (composes; from_dict/to_dict/from_toml + preset factories)
  defaults.toml       # shipped defaults (checked against dataclass defaults by test)
```
`SolverOptions` stays in `gpurec/api/solver_options.py` (re-exported); `PenaltyOptions` wraps the existing `OriginationPenalty` (`gpurec/solver/penalties.py`).

---

## Phase 1 — Foundation: config package + de-dup (no behavior change)

### Task 1: Config package skeleton + shared dtype-tol helper

**Files:**
- Create: `gpurec/config/__init__.py`, `gpurec/config/gpurec_config.py`
- Modify: `gpurec/api/solver_options.py` (add `e_tangent_tol`), `gpurec/solver/forward_tangent.py` (delegate `_default_tol`)
- Test: `tests/test_config_core.py`

**Interfaces:**
- Produces: `SolverOptions.e_tangent_tol: float = 1e-9`; `gpurec.config.dtype_rel_tol_default(dtype)`, `dtype_rel_tol_floor(dtype)` (moved from `_implicit_grad`, re-exported there for back-compat).

- [ ] **Step 1: Failing test** — `test_config_core.py`:
```python
import torch
from gpurec.config import dtype_rel_tol_default, dtype_rel_tol_floor
from gpurec.api.solver_options import SolverOptions

def test_dtype_tol_single_source():
    assert dtype_rel_tol_default(torch.float32) == 1e-6
    assert dtype_rel_tol_default(torch.float64) == 1e-12
    assert dtype_rel_tol_floor(torch.float32) == 4.0 * torch.finfo(torch.float32).eps

def test_forward_tangent_uses_shared_helper():
    from gpurec.solver.forward_tangent import _default_tol
    assert _default_tol(torch.float64) == dtype_rel_tol_default(torch.float64)

def test_solver_options_has_tangent_tol():
    assert SolverOptions().e_tangent_tol == 1e-9
```
- [ ] **Step 2: Run — expect ImportError/AttributeError.**
- [ ] **Step 3: Implement** — move the two helper functions into `gpurec/config/gpurec_config.py` (or a small `gpurec/config/_tol.py`), keep thin re-exports named `_bicgstab_rel_tol_default`/`_floor` in `_implicit_grad.py` that delegate. Add `e_tangent_tol: float = 1e-9` to `SolverOptions`. Rewrite `forward_tangent._default_tol` to `return dtype_rel_tol_default(dtype)`.
- [ ] **Step 4: Run — pass.**
- [ ] **Step 5: Run `tests/test_bicgstab_tolerance.py`, `tests/test_gmres_e_adjoint.py` — still pass (values unchanged).**
- [ ] **Step 6: Commit.**

### Task 2: Wire E-solve fallbacks + divergent signature defaults to `SolverOptions`

**Files:** Modify `gpurec/core/kernels/e_step.py`, `e_step_tangent.py`, `gpurec/api/_implicit_grad.py`; Test: extend `tests/test_config_core.py`.

- [ ] **Step 1: Failing test** — assert the previously-divergent signature defaults now agree with `SolverOptions()`:
```python
import inspect
from gpurec.api import _implicit_grad as ig
from gpurec.api.solver_options import SolverOptions

def test_no_divergent_signature_defaults():
    so = SolverOptions()
    sig = inspect.signature(ig.implicit_grad_loglik_vjp_wave).parameters
    # None-sentinel -> resolved to SolverOptions default at call time
    assert sig["neumann_terms"].default in (None, so.neumann_terms)
    assert sig["bicgstab_max_iter"].default in (None, so.bicgstab_max_iter)
    assert inspect.signature(ig._bicgstab).parameters["max_iter"].default in (None, so.bicgstab_max_iter)
```
- [ ] **Step 2: Run — expect fail (defaults are 3 / 500).**
- [ ] **Step 3: Implement** — change the divergent signature defaults to `None`; at the top of each function, `x = SolverOptions().<field> if x is None else x`. For `e_step.py`/`e_step_tangent.py` `max_iter`/`tol`, likewise default `None` → `SolverOptions().e_max_iter` / `e_tol` / `e_tangent_tol`. (Per Global Constraint 3.)
- [ ] **Step 4: Run — pass.**
- [ ] **Step 5: GATE — run `tests/test_optim_golden.py` + `tests/test_genewise_hvp.py`; grad rel-L2 ≤ 2e-3, all green (proves production paths unaffected).**
- [ ] **Step 6: Commit.**

### Task 3: Deduplicate `tv_eps`, self-loop cap (`200`), kernel-tile `128`

**Files:** Modify `gpurec/solver/value_and_grad.py`, `gpurec/solver/penalties.py`, `gpurec/solver/forward_tangent.py`, `gpurec/solver/ggn.py`; Test: `tests/test_config_core.py`.
- [ ] **Step 1: Failing test** — `from gpurec.solver.penalties import DEFAULT_TV_EPS` and assert `value_and_grad` uses it (import the same constant).
- [ ] **Step 2: Run — fail.**
- [ ] **Step 3: Implement** — define `DEFAULT_TV_EPS = 1e-3` once in `penalties.py`; import it in `value_and_grad.py`. Define `DEFAULT_SELF_MAX_ITER = 200` once (in `forward_tangent.py`), import in `ggn.py`.
- [ ] **Step 4-5: Run test + `tests/test_regularizer_integration.py` — pass.**
- [ ] **Step 6: Commit.**

**Phase 1 gate:** full suite green (145 tests); golden bit-parity intact.

---

## Phase 2 — New option dataclasses

### Task 4: `NewtonOptions` dataclass + wire the 4 curvature files

**Files:** Create `gpurec/config/newton.py`; Modify `gpurec/solver/curvature.py`, `genewise_curvature.py`, `origination_curvature.py`, `receiver_curvature.py`; Test: `tests/test_config_newton.py`.

**Interfaces — Produces:**
```python
# gpurec/config/newton.py
from dataclasses import dataclass

@dataclass
class NewtonOptions:
    """Newton / Levenberg-Marquardt / Lanczos / CG controls for the curvature solvers."""
    sigma: float = 0.01            # initial LM damping fraction (lam0 = sigma * lam_max)
    sigma_floor: float = 1e-4      # min damping fraction
    lanczos_m: int = 10            # Lanczos steps for lam_max estimate
    nu: float = 1.5                # neg-curvature damping-bump factor
    decrease: float = 1.5          # accepted-step damping-decrease factor
    max_bumps: int = 3             # max neg-curv re-solves per Newton step
    max_cg: int = 40               # CG iters per Newton system
    c1: float = 1e-4               # Armijo sufficient-decrease constant
    ls_max: int = 25               # max line-search backtracks
    gtol: float = 1e-2             # projected-gradient-norm stop
    max_newton: int = 40           # max Newton iterations
    ftol: float = 1e-9             # relative-improvement stall floor
    seed: int = 0                  # Lanczos start-vector RNG seed
    fd_eps_blockwise: float = 1e-2 # FD step, genewise 3x3 Hessian
    fd_eps_hvp: float = 1e-5       # FD step, full-HVP solvers
    lam_ceil_factor: float = 10.0  # damping ceiling = factor * lam_max
    forcing_eta: float = 0.1       # inexact-Newton forcing-term cap
    certify_m: int = 200           # Lanczos steps for PD certificate
    cg_tol: float = 1e-7           # Fisher-info CG residual tol
    cg_max: int = 400              # Fisher-info CG iteration cap

    def validate(self) -> None:
        for name in ("lanczos_m","max_cg","ls_max","max_newton","max_bumps","certify_m","cg_max"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be >= 1")
        for name in ("sigma","sigma_floor","c1","gtol","ftol","fd_eps_blockwise","fd_eps_hvp","cg_tol"):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be > 0")
```
- [ ] **Step 1: Failing test** — `tests/test_config_newton.py`: construct `NewtonOptions()`, assert each field equals the literal it replaces (pin the values); `NewtonOptions().validate()` passes; a bad value raises.
- [ ] **Step 2: Run — fail (no module).**
- [ ] **Step 3: Implement** `newton.py`; then replace the copy-pasted signature defaults in the 4 curvature `newton_*` functions with a single `newton: NewtonOptions | None = None` param (default `NewtonOptions()`), reading `opts.sigma` etc. Map each old kwarg → field. Keep old kwarg names accepted (deprecation shim) only if any caller passes them; otherwise remove. Inline constants (`lam_ceil`, `forcing_eta`, `certify_m`, `cg_tol`, `cg_max`) read from `opts`.
- [ ] **Step 4: Run — pass.**
- [ ] **Step 5: GATE — `tests/test_genewise_hvp.py`, `tests/test_optim_golden.py`, plus any curvature/certificate tests. Values identical → green.**
- [ ] **Step 6: Commit.**

### Task 5: `RateBounds` dataclass + resolve the rate-floor split

**Files:** Create `gpurec/config/rates.py`; Modify `gpurec/optimization.py`, `gpurec/api/model.py`, `gpurec/fit/genewise_fit.py`; Test: `tests/test_config_rates.py`.

**Interfaces — Produces:**
```python
# gpurec/config/rates.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class RateBounds:
    """Box bounds + init for log2 DTL event rates (relative to speciation=1)."""
    min_rate: float = 1e-10          # global floor (optimization.py / model init)
    max_rate: Optional[float] = None # no cap by default
    init_rate: float = 1e-10         # theta Parameter init
    bound_active_eps: float = 1e-6   # |theta - bound| < eps => 'active' (genewise cert)
```
- [ ] **Step 1: Failing test** — `RateBounds()` defaults match global values; a `RateBounds.genewise()` classmethod (or the genewise preset) yields `min_rate=1e-6, max_rate=2.0`.
- [ ] **Step 2: Run — fail.**
- [ ] **Step 3: Implement** — `optimization.py` functions take `bounds: RateBounds | None = None` (default `RateBounds()`), replacing `min_rate=1e-10`/`max_rate=None` kwargs (keep kwargs as overrides). `model.py:75` init uses `bounds.init_rate`. `fit_genewise` builds `RateBounds(min_rate=1e-6, max_rate=2.0)` in its preset (Global Constraint 2) instead of bare signature defaults; wire `bound_active_eps` at `genewise_fit.py:247,281`.
- [ ] **Step 4-5: Run test + `tests/test_reference_defaults.py` (adjust if genewise rate bounds move into the preset) + a genewise smoke test — pass.**
- [ ] **Step 6: Commit.**

### Task 6: `MemoryOptions` dataclass

**Files:** Create `gpurec/config/memory.py`; Modify `gpurec/core/memory_policy.py`, `gpurec/solver/value_and_grad.py`, `gpurec/solver/curvature.py`, `gpurec/solver/hvp_exact.py`; Test: `tests/test_config_memory.py`.

**Interfaces — Produces:**
```python
# gpurec/config/memory.py
from dataclasses import dataclass

@dataclass
class MemoryOptions:
    fraction: float = 0.85          # usable frac of total GPU mem (env GPUREC_MEMORY_POLICY_FRACTION)
    reserve_gib: float = 1.0        # headroom GiB (env GPUREC_MEMORY_POLICY_RESERVE_GIB)
    scratch_tensors: int = 10       # [W,S]-tensor multiplier in wave mem estimate
    min_free_gib_driver: float = 4.0  # value_and_grad cache-empty gate
    min_free_gib_hvp: float = 8.0     # curvature HVP-rebuild gate
    free_cache_every: int = 32        # empty CUDA cache every-K-waves (env NEWTON_FREE_CACHE_EVERY)
    grad_avg_k: int = 1               # backward passes averaged
```
- [ ] **Step 1: Failing test** — defaults match current literals; env overrides still honored (`GPUREC_MEMORY_POLICY_FRACTION`).
- [ ] **Step 2-4: implement, wiring each site to the field (env-var precedence preserved); run — pass.**
- [ ] **Step 5: Commit.** (No golden gate needed — memory gates don't affect numerics; run `tests/` collection to confirm no import breakage.)

### Task 7: `PenaltyOptions` facade

**Files:** Modify `gpurec/solver/penalties.py` (add `PenaltyOptions` grouping `OriginationPenalty` + `tv_eps` + ridge `lambdas`/`lam_margin`/`lam_floor`); Test: `tests/test_config_penalty.py`.
- [ ] Wrap existing `OriginationPenalty` (unchanged) + `tv_eps=DEFAULT_TV_EPS` + ridge defaults into `PenaltyOptions`; keep `OriginationPenalty` importable. Test defaults; run `tests/test_regularizer_integration.py`. Commit.

**Phase 2 gate:** full suite green; golden + HVP bit-parity intact.

---

## Phase 3 — `GpurecConfig` top-level + TOML

### Task 8: `GpurecConfig` composition + `from_dict`/`to_dict`

**Files:** `gpurec/config/gpurec_config.py`, `gpurec/config/__init__.py`; Test: `tests/test_gpurec_config.py`.

**Interfaces — Produces:**
```python
# gpurec/config/gpurec_config.py
from dataclasses import dataclass, field, asdict
from gpurec.api.solver_options import SolverOptions
from gpurec.config.newton import NewtonOptions
from gpurec.config.rates import RateBounds
from gpurec.config.memory import MemoryOptions
from gpurec.solver.penalties import PenaltyOptions

@dataclass
class GpurecConfig:
    solver: SolverOptions = field(default_factory=SolverOptions)
    newton: NewtonOptions = field(default_factory=NewtonOptions)
    rates: RateBounds = field(default_factory=RateBounds)
    regularizer: PenaltyOptions = field(default_factory=PenaltyOptions)
    memory: MemoryOptions = field(default_factory=MemoryOptions)

    def validate(self) -> None:
        self.solver.validate(); self.newton.validate()

    def to_dict(self) -> dict: ...      # nested asdict
    @classmethod
    def from_dict(cls, d: dict) -> "GpurecConfig": ...  # deep-merge onto defaults, unknown key -> error
```
- [ ] **Step 1: Failing test** — `GpurecConfig()` builds; `from_dict(to_dict())` round-trips equal; unknown key raises; partial dict deep-merges onto defaults.
- [ ] **Step 2-4: implement, run — pass. Commit.**

### Task 9: `defaults.toml` + `from_toml` loader + parity test

**Files:** `gpurec/config/defaults.toml`, `gpurec/config/gpurec_config.py` (add `from_toml`/`load_config`); Test: extend `tests/test_gpurec_config.py`.
- [ ] **Step 1: Failing test — the source-of-truth guard:**
```python
from pathlib import Path
from gpurec.config import GpurecConfig
def test_defaults_toml_matches_dataclass_defaults():
    p = Path(GpurecConfig.__module__ and "gpurec/config/defaults.toml")
    assert GpurecConfig.from_toml(p) == GpurecConfig()

def test_user_toml_overrides_merge():
    cfg = GpurecConfig.from_dict({"solver": {"pi_iters": 128}})
    assert cfg.solver.pi_iters == 128 and cfg.newton.max_cg == 40  # others default
```
- [ ] **Step 2: Run — fail.**
- [ ] **Step 3: Implement** — hand-write `defaults.toml` mirroring every dataclass default (sections `[solver] [newton] [rates] [regularizer] [regularizer.origination] [memory]`). `from_toml(path)` = `tomllib.load` → `from_dict` (deep-merged onto defaults, so a user TOML need only list overrides). `load_config(path=None)` returns `GpurecConfig()` when `path is None`.
- [ ] **Step 4: Run — pass.** (This test is the permanent guard that TOML never drifts from code.)
- [ ] **Step 5: Commit.**

### Task 10: Recipe presets as `GpurecConfig` factories

**Files:** Modify `gpurec/fit/genewise_fit.py`, `gpurec/fit/optimize.py`, `gpurec/fit/map_cv.py`; add factories to `gpurec/config/gpurec_config.py`; Test: update `tests/test_reference_defaults.py`.
- [ ] Add `GpurecConfig.genewise_reference()`, `.optimize_reference()`, `.map_cv_reference()` returning the tuned configs (composing `SolverOptions`, `NewtonOptions`, `RateBounds` overrides). Keep `GENEWISE_REFERENCE`/`OPTIMIZE_REFERENCE`/`MAP_CV_REFERENCE` dicts as thin views derived from the factories (or vice-versa) so `test_reference_defaults.py` still holds. Update that test to also assert `GpurecConfig.genewise_reference()` reproduces the dict. Run the reference-defaults + a fit smoke test. Commit.

### Task 11: Accept `GpurecConfig` at the driver/model boundary

**Files:** Modify `gpurec/api/model.py` (constructor accepts `config: GpurecConfig | None`, decomposing into `solver_options` etc.), `gpurec/cli/_common.py` (`--config path.toml` → `load_config`); Test: `tests/test_config_wiring.py`.
- [ ] `GeneReconModel(..., config=None)`: when given, use `config.solver` as `solver_options`, thread `config.newton`/`rates`/`memory` to the fit/curvature calls; when `None`, current behavior. Add `--config` CLI flag that loads a TOML and overrides the argparse solver defaults. Test that a TOML round-trips through the CLI to a `SolverOptions`. Run `tests/test_cli.py`. Commit.

**Phase 3 gate:** full suite green; golden + HVP bit-parity; `test_defaults_toml_matches_dataclass_defaults` green.

---

## Phase 4 — Docs + final verification

### Task 12: Update `docs/config_convention.md` + final gate
- [ ] Rewrite `docs/config_convention.md`: the new hierarchy (`GpurecConfig` + TOML), how to override (script dataclass OR `--config file.toml`), the source-of-truth rule, and the non-goal (kernel knobs stay derived). Note the four resolved inconsistencies.
- [ ] **Final GATE:** full suite (`pytest tests/ -q`, expect all green + the new config tests), `tests/test_optim_golden.py` (grad rel-L2 ≤ 2e-3, cross-parity 1e-6), `tests/test_genewise_hvp.py`. Confirm zero collection errors.
- [ ] Commit; then use superpowers:finishing-a-development-branch.

---

## Verification summary (per-phase gates)
- **Every phase:** `pytest tests/ -q` all green; `tests/test_optim_golden.py` grad rel-L2 ≤ 2e-3 + forward cross-parity 1e-6 (proves no numeric drift).
- **Phase 2/3:** `tests/test_genewise_hvp.py` (curvature path uses `NewtonOptions`).
- **Phase 3:** `test_defaults_toml_matches_dataclass_defaults` (TOML ≡ dataclass defaults, permanent guard).
- Any golden/HVP change that is NOT one of the four declared resolutions = a regression: stop and diagnose, do not re-baseline.
