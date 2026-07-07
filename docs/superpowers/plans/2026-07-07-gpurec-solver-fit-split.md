# gpurec solver/fit Split + Config Convention — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split the flat `gpurec/optim/` into `gpurec/solver/` (stable toolkit) + `gpurec/fit/` (drivers), surface the dataset-tuned defaults as importable reference constants, and document the per-dataset config convention — all behavior-preserving.

**Architecture:** A mechanical `git mv` + import sweep (Task 1), then additive reference constants + a convention doc (Task 2), then a gpurax re-vendor (Task 3). No numerics change; the gate is the full test suite staying green, plus a consistency test for the constants. Spec: `docs/superpowers/specs/2026-07-07-gpurec-solver-fit-split-design.md`.

**Tech Stack:** Python 3.12, PyTorch/Triton, pytest, git. Interpreter: `.venv/bin/python`. GPU box (RTX 4090) required for the suite.

## Global Constraints

- **Behavior-preserving:** no change to any function's numerics, signatures' *values*, or the `SolverOptions`/`OriginationPenalty` dataclasses. `pi_tiers`/`neu_opt`/`neu_cert` (genewise escalation) and per-driver `adam_steps` are NOT duplication — leave them.
- **No `FitRecipe`/dispatcher/presets framework.** Non-goals per the spec.
- **Git safety:** the working tree has untracked experiment dirs (`experiments/`, `ghost_experiments/`, `docs/gergely_comparison/`, …) — never `git add -A`; stage only the exact paths each task names. Never `checkout`/`reset` a file with uncommitted changes.
- **No back-compat `optim/` shim** — clean break (re-vendor accepted).
- **Gate:** full suite baseline is **175 passed / 3 skipped / 0 failures** + the pre-existing `test_cli` basename collection clash (run with `--continue-on-collection-errors`; the gpurec CLI tests run in isolation via `pytest tests/test_cli.py`). Every task must hold this.
- **Move mapping (canonical — used by every import rewrite):**
  - → `gpurec/solver/`: `value_and_grad`, `cg`, `curvature`, `hvp_exact`, `ggn`, `forward_tangent`, `receiver_curvature`, `origination_curvature`, `genewise_curvature`, `penalties`
  - → `gpurec/fit/`: `optimize`, `newton_cg`, `map_fit`, `map_cv`, `genewise_fit`, `baselines`

---

## Task 0: Baseline

**Files:** none.

- [ ] **Step 1: Branch**
```bash
cd /home/enzo/Documents/git/gpurec/consolidate-release
git status                       # only untracked experiment files
git checkout -b refactor/solver-fit-split
```

- [ ] **Step 2: Capture the green baseline (GPU box)**
```bash
.venv/bin/python -m pytest tests/ --continue-on-collection-errors -q -p no:cacheprovider 2>&1 | tail -3
.venv/bin/python -m pytest tests/test_cli.py -q 2>&1 | tail -2
```
Expected: `175 passed, 3 skipped, … 1 error` (the `test_cli` clash) and `10 passed` for the isolated CLI run.

---

## Task 1: Move `optim/` → `solver/` + `fit/` and sweep imports

**Files:**
- Move (16): `gpurec/optim/*.py` → `gpurec/solver/` and `gpurec/fit/` per the move mapping.
- Create: `gpurec/solver/__init__.py`, `gpurec/fit/__init__.py`.
- Delete: `gpurec/optim/__init__.py` (and the now-empty `gpurec/optim/`).
- Modify (import sweep): every `.py` under `gpurec/`, `tests/`, `gates/` that references `gpurec.optim.*`; plus `gpurec/__init__.py`.

**Interfaces:**
- Produces: `gpurec.solver.<mod>` and `gpurec.fit.<mod>` importable at the new paths; `gpurec.solver` re-exports the penalties + value_and_grad public names; `gpurec.fit` re-exports `fit_genewise`; `gpurec.fit_genewise` (top-level) still importable.

- [ ] **Step 1: Move the files with git (preserves history)**
```bash
cd /home/enzo/Documents/git/gpurec/consolidate-release
mkdir -p gpurec/solver gpurec/fit
git mv gpurec/optim/value_and_grad.py gpurec/optim/cg.py gpurec/optim/curvature.py \
       gpurec/optim/hvp_exact.py gpurec/optim/ggn.py gpurec/optim/forward_tangent.py \
       gpurec/optim/receiver_curvature.py gpurec/optim/origination_curvature.py \
       gpurec/optim/genewise_curvature.py gpurec/optim/penalties.py gpurec/solver/
git mv gpurec/optim/optimize.py gpurec/optim/newton_cg.py gpurec/optim/map_fit.py \
       gpurec/optim/map_cv.py gpurec/optim/genewise_fit.py gpurec/optim/baselines.py gpurec/fit/
git rm gpurec/optim/__init__.py
```

- [ ] **Step 2: Sweep every `gpurec.optim.<mod>` reference to its new home**
```bash
cd /home/enzo/Documents/git/gpurec/consolidate-release
FILES=$(grep -rlE "gpurec\.optim" gpurec tests gates --include="*.py")
sed -i -E \
  -e 's/gpurec\.optim\.(value_and_grad|cg|curvature|hvp_exact|ggn|forward_tangent|receiver_curvature|origination_curvature|genewise_curvature|penalties)/gpurec.solver.\1/g' \
  -e 's/gpurec\.optim\.(optimize|newton_cg|map_fit|map_cv|genewise_fit|baselines)/gpurec.fit.\1/g' \
  -e 's/from gpurec\.optim import curvature/from gpurec.solver import curvature/g' \
  $FILES
```

- [ ] **Step 3: Create the two package `__init__.py` (split of the old optim exports)**

`gpurec/solver/__init__.py`:
```python
"""Stable optimization toolkit: value/grad, CG/Lanczos, curvature + exact HVP, tangents, penalties."""
from gpurec.solver.penalties import (
    OriginationPenalty,
    group_expand,
    group_reduce,
    origination_penalty_and_grad,
    tv_prior_and_grad,
)
from gpurec.solver.value_and_grad import (
    FORWARD_SAVED_NAMES,
    forward_solve,
    free_cuda_cache_if_tight,
    make_value_and_grad,
)

__all__ = [
    "FORWARD_SAVED_NAMES", "forward_solve", "free_cuda_cache_if_tight", "make_value_and_grad",
    "OriginationPenalty", "origination_penalty_and_grad", "tv_prior_and_grad",
    "group_expand", "group_reduce",
]
```

`gpurec/fit/__init__.py`:
```python
"""Fitting drivers: first-order / Newton-CG / MAP / CV / genewise recipes."""
from gpurec.fit.genewise_fit import fit_genewise

__all__ = ["fit_genewise"]
```

- [ ] **Step 4: Fix the top-level re-export in `gpurec/__init__.py`**

The sweep already rewrote line 8 to `from gpurec.fit.genewise_fit import fit_genewise`. Confirm:
```bash
grep -nE "fit_genewise|gpurec\.(optim|fit|solver)" gpurec/__init__.py
```
Expected: `from gpurec.fit.genewise_fit import fit_genewise as fit_genewise`; no `gpurec.optim`.

- [ ] **Step 5: Verify no dangling refs + everything imports**
```bash
grep -rnE "gpurec\.optim" gpurec tests gates --include="*.py" && echo "!! DANGLING" || echo "  no gpurec.optim references remain ✓"
[ -d gpurec/optim ] && echo "!! optim/ still exists" || echo "  optim/ removed ✓"
.venv/bin/python -c "import gpurec; from gpurec import fit_genewise; import gpurec.solver, gpurec.fit; import gpurec.fit.optimize, gpurec.solver.curvature, gpurec.fit.newton_cg, gpurec.solver.ggn; print('imports OK')"
.venv/bin/python -m pytest tests/ --collect-only -q -p no:cacheprovider 2>&1 | tail -1
```
Expected: no dangling refs, `imports OK`, and the same collected count as baseline.

- [ ] **Step 6: Full suite (GPU box) + commit**
```bash
.venv/bin/python -m pytest tests/ --continue-on-collection-errors -q -p no:cacheprovider 2>&1 | tail -3
.venv/bin/python -m pytest tests/test_cli.py -q 2>&1 | tail -2
git add gpurec/ tests/ gates/
git diff --cached --name-only | grep -vE "^(gpurec|tests|gates)/" && echo "!! unexpected" || echo "  clean"
git commit -m "refactor: split optim/ into solver/ (toolkit) + fit/ (drivers)"
```
Expected: `175 passed / 3 skipped` + `10 passed`, unchanged.

---

## Task 2: Reference constants + config convention

**Files:**
- Modify: `gpurec/fit/genewise_fit.py`, `gpurec/fit/optimize.py`, `gpurec/fit/map_cv.py` — add a labeled reference constant per driver; leave signatures' literal defaults as-is.
- Create: `tests/test_reference_defaults.py` — consistency test.
- Create: `docs/config_convention.md` — the one-page convention.

**Interfaces:**
- Produces: `GENEWISE_REFERENCE` (in `gpurec.fit.genewise_fit`), `OPTIMIZE_REFERENCE` (`gpurec.fit.optimize`), `MAP_CV_REFERENCE` (`gpurec.fit.map_cv`) — each a `dict` of the driver's dataset-tuned defaults, importable so a script does `fit_genewise(sp, genes, **{**GENEWISE_REFERENCE, "tol": 5e-4})`.

- [ ] **Step 1: Write the failing consistency test**

`tests/test_reference_defaults.py`:
```python
import inspect
from gpurec.fit.genewise_fit import fit_genewise, GENEWISE_REFERENCE
from gpurec.fit.optimize import optimize, OPTIMIZE_REFERENCE
from gpurec.fit.map_cv import map_cv, MAP_CV_REFERENCE


def _check(fn, ref):
    params = inspect.signature(fn).parameters
    for k, v in ref.items():
        assert k in params, f"{fn.__name__} has no param {k}"
        assert params[k].default == v, f"{fn.__name__}.{k}: default {params[k].default!r} != ref {v!r}"


def test_genewise_reference_matches_signature():
    _check(fit_genewise, GENEWISE_REFERENCE)


def test_optimize_reference_matches_signature():
    _check(optimize, OPTIMIZE_REFERENCE)


def test_map_cv_reference_matches_signature():
    _check(map_cv, MAP_CV_REFERENCE)
```

- [ ] **Step 2: Run it — expect ImportError (constants don't exist yet)**
```bash
.venv/bin/python -m pytest tests/test_reference_defaults.py -q 2>&1 | tail -3
```
Expected: FAIL (ImportError: cannot import name `GENEWISE_REFERENCE`).

- [ ] **Step 3: Add the reference constants (values = the current literal defaults)**

In `gpurec/fit/genewise_fit.py`, above `fit_genewise`:
```python
# Reference recipe tuned for the standard genewise problem. Import and clone-override per dataset:
#   fit_genewise(sp, genes, **{**GENEWISE_REFERENCE, "tol": 5e-4})
# Per-dataset values belong in your experiment script, NOT edited here (see docs/config_convention.md).
GENEWISE_REFERENCE = dict(
    adam_steps=5, adam_lr=1.0, grad_clip=10.0, pi_tiers=(16, 64), neu_opt=16, neu_cert=64,
    clade_budget=None, tol=1e-3, max_iter=120, check_every=4, drop_frac=0.30, trust=2.0,
    fd_eps=1e-2, mu=1e-2, fwd_tol=1e-3, improve_frac=0.8, verify_drop=True, eager_defer=True,
    warm_adjoint=True, certify=False,
)
```

In `gpurec/fit/optimize.py`, above `optimize`:
```python
# Reference tuning from the 666x80 characterization; clone-override per dataset. See docs/config_convention.md.
OPTIMIZE_REFERENCE = dict(
    optimizer="adam", lr0=1.0, schedule="adaptive", max_steps=300, polish_mode="ridge", max_newton=8,
)
```

In `gpurec/fit/map_cv.py`, above `map_cv`:
```python
# Reference CV tuning; clone-override per dataset. See docs/config_convention.md.
MAP_CV_REFERENCE = dict(
    k=5, lambdas=(0.0, 1.0, 10.0, 100.0, 1000.0), mode="specieswise", init_rate=0.1, seed=0,
    adam_steps=60, lbfgs_iters=80, maxcor=50,
)
```

- [ ] **Step 4: Run the consistency test — expect PASS**
```bash
.venv/bin/python -m pytest tests/test_reference_defaults.py -q 2>&1 | tail -2
```
Expected: `3 passed`. (If a default differs from the constant, fix the constant to match the signature — the signature is ground truth.)

- [ ] **Step 5: Add the convention doc**

`docs/config_convention.md`:
```markdown
# gpurec config convention

**Rule:** per-dataset configuration lives in your experiment script, never edited into a library
function body. The library ships only reference/neutral defaults.

- **Solver settings** → construct a `SolverOptions` in your script; pass it in.
- **Regularizers / priors** → build `OriginationPenalty(...)`, `tv_penalty=...`, `ridge`/`lam` in
  your script; pass them to the driver.
- **Recipe hyperparameters** → start from the driver's reference constant and override:
  `fit_genewise(sp, genes, **{**GENEWISE_REFERENCE, "tol": 5e-4})`. The reference constants
  (`GENEWISE_REFERENCE`, `OPTIMIZE_REFERENCE`, `MAP_CV_REFERENCE`) are tuned for reference problems
  (the genewise recipe / the 666x80 characterization) — they are starting points, not universal.

If you catch yourself editing a default inside `gpurec/fit/` or `gpurec/solver/` to make a dataset
work, that value belongs in your script instead.
```

- [ ] **Step 6: Full suite + commit**
```bash
.venv/bin/python -m pytest tests/test_reference_defaults.py -q && \
.venv/bin/python -m pytest tests/ --continue-on-collection-errors -q -p no:cacheprovider 2>&1 | tail -3
git add gpurec/fit/genewise_fit.py gpurec/fit/optimize.py gpurec/fit/map_cv.py \
        tests/test_reference_defaults.py docs/config_convention.md
git commit -m "refactor(fit): surface dataset-tuned defaults as reference constants + config convention doc"
```

---

## Task 3: Re-vendor gpurec into gpurax

**Files:**
- Modify (in `~/Documents/git/gpurax`): the vendored `gpurec/` copy + any `gpurec.optim.*` imports in gpurax's own code/tests.

**Interfaces:** consumes the new `gpurec.solver`/`gpurec.fit` paths; gpurax uses `gpurec.fit.optimize.optimize`, `gpurec.api.model.GeneReconModel`, `gpurec.core.scheduling.batching.preprocess_dataset`, `gpurec.sample_reconciliations`.

- [ ] **Step 1: Re-vendor the refactored gpurec**
```bash
rsync -a --delete --exclude='__pycache__/' --exclude='*.pyc' --exclude='*.egg-info/' \
  /home/enzo/Documents/git/gpurec/consolidate-release/gpurec/ \
  /home/enzo/Documents/git/gpurax/gpurec/
diff -rq --exclude=__pycache__ --exclude='*.pyc' /home/enzo/Documents/git/gpurax/gpurec \
  /home/enzo/Documents/git/gpurec/consolidate-release/gpurec && echo "  IDENTICAL ✓"
```

- [ ] **Step 2: Rewrite gpurax's own `gpurec.optim.*` imports**
```bash
cd /home/enzo/Documents/git/gpurax
GX=$(grep -rlE "gpurec\.optim" gpurax tests --include="*.py")
[ -n "$GX" ] && sed -i -E \
  -e 's/gpurec\.optim\.(value_and_grad|cg|curvature|hvp_exact|ggn|forward_tangent|receiver_curvature|origination_curvature|genewise_curvature|penalties)/gpurec.solver.\1/g' \
  -e 's/gpurec\.optim\.(optimize|newton_cg|map_fit|map_cv|genewise_fit|baselines)/gpurec.fit.\1/g' \
  $GX
grep -rnE "gpurec\.optim" gpurax tests --include="*.py" && echo "!! DANGLING in gpurax" || echo "  gpurax clean ✓"
```

- [ ] **Step 3: gpurax suite + commit (in the gpurax repo)**
```bash
cd /home/enzo/Documents/git/gpurax
.venv/bin/python -m pytest tests/ -q -p no:cacheprovider 2>&1 | tail -3
git add gpurec/ gpurax/ tests/
git commit -m "vendor: sync gpurec (optim -> solver/fit split); update gpurax imports"
```
Expected: `53 passed`.

---

## Self-review

- **Spec coverage:** folder split (Task 1) ✓; reference constants `GENEWISE_REFERENCE`/`OPTIMIZE_REFERENCE`/`MAP_CV_REFERENCE` (Task 2) ✓; convention doc (Task 2) ✓; gpurax re-vendor (Task 3) ✓; non-goals (no framework) respected — only constants + a doc added.
- **Placeholders:** none — exact `git mv` lists, exact sed rules, exact constant values, exact commands + expected output.
- **Consistency:** the move mapping in Global Constraints, the `git mv` in Task 1 Step 1, and the sed rules in Task 1 Step 2 / Task 3 Step 2 use the identical 10-solver / 6-fit partition. The reference-constant names match between Task 2's test (Step 1) and definitions (Step 3).
- **Risk note:** the sed sweep is the one non-trivial step; Step 5's "no dangling `gpurec.optim`" grep + collection check + full suite catch any missed or mis-mapped reference before commit.
