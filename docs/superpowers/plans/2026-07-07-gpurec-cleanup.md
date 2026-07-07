# gpurec Package Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prune shipped test/research scaffolding and de-duplicate the `optim/` curvature and implicit-VJP code, reducing the `gpurec/` package by ~2,500–3,000 LOC (~15–17%) with **zero change to production behavior**.

**Architecture:** These are **behavior-preserving refactors**, not features. The safety net is the existing repo test suite (`tests/`, 48 files), NOT new tests. The discipline is: (1) record the baseline of which tests pass, (2) make the change, (3) confirm the *same* tests still pass. No production `.py` under `gpurec/` should change observable behavior; where a task must edit production code (P2, P3), the gate is bit-for-bit identical outputs on the relevant golden/FD tests.

**Tech Stack:** Python, PyTorch, Triton (GPU kernels), pytest. Native Rust `.abi3.so` extensions. Most tests require CUDA + Triton — **run verification on the RTX 4090 box**, not in a CPU-only environment.

## Global Constraints

- **Git safety (user rule):** NEVER `git checkout`/`restore`/`reset` a file with uncommitted changes. The working tree currently has **untracked** experiment files (`ccps.pdf`, `experiments/…`, `ghost_experiments/…`, etc.) — do NOT add, move, or delete any of them. Every task touches only the files it names.
- **Branch:** current branch is `feat/cli-and-fidelity`. Do this work on a dedicated branch `refactor/optim-cleanup` cut from it. Commit each task separately with the message shown.
- **No behavior change:** do not "improve" adjacent code, rename public API, or touch `core/kernels/*` math. Do exactly the scoped move/dedup and nothing more.
- **No clamp:** do not introduce `clamp`/`clamp_` anywhere (existing `clamp_log_rate_` is an intentional projection helper — leave it).
- **Verification command (baseline for all tasks):** `pytest tests/ -q` on the GPU box. Fast subsets are named per task for tighter loops.
- **Tasks are independent and individually optional.** The user will pick which to run. Dependency notes are called out; absent a note, a task stands alone.

---

## Task 0: Baseline safety net (do this once, before any P task)

**Files:** none modified.

- [ ] **Step 1: Create the working branch**

```bash
cd /home/enzo/Documents/git/gpurec/consolidate-release
git status                      # confirm only untracked files, no tracked modifications
git checkout -b refactor/optim-cleanup
```

- [ ] **Step 2: Record the current LOC baseline**

```bash
find gpurec -name '*.py' | xargs wc -l | tail -1    # expect ~17240 total
```

- [ ] **Step 3: Record which tests currently pass (the regression oracle)**

Run on the GPU box:
```bash
pytest tests/ -q | tee /tmp/gpurec_baseline_tests.txt
```
Expected: a green/known-failing baseline. Any test that is red *before* a refactor stays out of that refactor's gate. Every task below means "no test that was green in this file goes red."

---

## Task P1a: Relocate the 11 gate/parity scripts out of the package

**Removes ~1,744 LOC from the shipped `gpurec` package.** These are runnable FD/parity "gate" scripts with `__main__` entry points; production never imports them (verified). One real test (`tests/test_optim_golden.py`) *does* import `_parity_kbench`, so this is a move-with-import-fix, not a delete.

**Risk:** Low–medium. Only risk is the pytest package-import path for the moved cluster.

**Files:**
- Move (11 files) from `gpurec/optim/` → `tests/gates/`:
  `_verify_hvp.py` (117), `_verify_hvp_recv.py` (322), `_verify_map.py` (111),
  `_verify_recv_grad.py` (163), `_verify_s3_fwd_tangent.py` (200),
  `_verify_s5_wave_col.py` (111), `_verify_s7_turnon.py` (112),
  `_verify_s9_curvature.py` (350), `_parity_kbench.py` (131),
  `_fit_kbench.py` (51), `_test_first_order_recv.py` (76)
- Create: `tests/gates/__init__.py` (empty), and `tests/__init__.py` if absent
- Modify (rewrite intra-cluster imports `gpurec.optim.X` → `tests.gates.X`):
  - `tests/gates/_test_first_order_recv.py:20` (`_verify_recv_grad`)
  - `tests/gates/_verify_s5_wave_col.py:23` (`_verify_hvp_recv`)
  - `tests/gates/_fit_kbench.py:18` (`_parity_kbench`)
  - `tests/gates/_verify_s7_turnon.py:30` (`_verify_hvp_recv`)
  - `tests/gates/_verify_s9_curvature.py:153` (`_verify_hvp_recv`)
  - `tests/gates/_verify_hvp.py:52` (`_parity_kbench`)
  - `tests/gates/_verify_s3_fwd_tangent.py:45` (`_verify_hvp_recv`)
  - `tests/test_optim_golden.py:65` (`_parity_kbench`)

**Interfaces:**
- The moved scripts keep importing PRODUCTION modules unchanged (e.g. `from gpurec.optim.receiver_curvature import …` in `_verify_s9_curvature.py:39` stays as-is — receiver_curvature is NOT moving).
- Only the 8 imports listed above change, all of the form `from gpurec.optim.<gatefile> import …` → `from tests.gates.<gatefile> import …`.

- [ ] **Step 1: Confirm pytest will not auto-collect the gate scripts as tests**

Default `python_files = test_*.py`; all moved files are `_verify_*` / `_parity_*` / `_fit_*` / `_test_first_order_*` (leading underscore → not matched). Verify no config overrides it:
```bash
grep -nE "python_files|python_classes|python_functions" pyproject.toml setup.cfg pytest.ini 2>/dev/null || echo "  (default collection — safe)"
```
Expected: default collection.

- [ ] **Step 2: Move the files with git (preserves history)**

```bash
cd /home/enzo/Documents/git/gpurec/consolidate-release
mkdir -p tests/gates
git mv gpurec/optim/_verify_hvp.py gpurec/optim/_verify_hvp_recv.py \
       gpurec/optim/_verify_map.py gpurec/optim/_verify_recv_grad.py \
       gpurec/optim/_verify_s3_fwd_tangent.py gpurec/optim/_verify_s5_wave_col.py \
       gpurec/optim/_verify_s7_turnon.py gpurec/optim/_verify_s9_curvature.py \
       gpurec/optim/_parity_kbench.py gpurec/optim/_fit_kbench.py \
       gpurec/optim/_test_first_order_recv.py \
       tests/gates/
touch tests/gates/__init__.py
[ -f tests/__init__.py ] || touch tests/__init__.py
```

- [ ] **Step 3: Rewrite the 8 intra-cluster imports**

Apply to the 8 sites listed under **Files** (7 in `tests/gates/*`, 1 in `tests/test_optim_golden.py`). Each is a literal substring swap:
`from gpurec.optim._verify_hvp_recv` → `from tests.gates._verify_hvp_recv`,
`from gpurec.optim._verify_recv_grad` → `from tests.gates._verify_recv_grad`,
`from gpurec.optim._parity_kbench` → `from tests.gates._parity_kbench`.

- [ ] **Step 4: Confirm no dangling references remain**

```bash
grep -rnE "gpurec\.optim\.(_verify|_parity_kbench|_fit_kbench|_test_first_order)" gpurec/ tests/
```
Expected: **no output**. (All such imports now point at `tests.gates`.)

- [ ] **Step 5: Verify the one real test that consumes the cluster still passes**

```bash
pytest tests/test_optim_golden.py -q
```
Expected: PASS (same as baseline). Then smoke-import the cluster:
```bash
python -c "import tests.gates._verify_hvp_recv, tests.gates._parity_kbench; print('gates import OK')"
```
Expected: `gates import OK`.

- [ ] **Step 6: Full suite + commit**

```bash
pytest tests/ -q                                  # compare against /tmp/gpurec_baseline_tests.txt
git add -A
git commit -m "refactor(optim): move FD/parity gate scripts out of package into tests/gates"
```

---

## Task P1b: Delete the dead `optim/diagnostics/` scripts

**Removes 558 LOC.** All 7 files import `from newton.baselines` / `newton.hvp_exact` / `newton.optimize` / `newton.vg` — a package that **does not exist in this repo** (leftover from the pre-port `newton/` package). They are unimportable dead code; nothing in `gpurec/` or `tests/` imports them.

**Risk:** Very low (already broken; git history preserves them).

**Files:**
- Delete: `gpurec/optim/diagnostics/` (all 7: `basin_compare.py`, `basin_connectivity.py`, `basin_interp.py`, `basin_search.py`, `convergence_audit.py`, `gauge_audit.py`, `theta_diagnostics.py`) + `__init__.py` if present.

- [ ] **Step 1: Confirm nothing imports the diagnostics package**

```bash
grep -rnE "optim\.diagnostics|from \.diagnostics|import diagnostics" gpurec/ tests/ || echo "  (no importers — safe to delete)"
```
Expected: no importers.

- [ ] **Step 2: Confirm the modules are genuinely broken (stale `newton` imports)**

```bash
grep -rnE "from newton|import newton" gpurec/optim/diagnostics/
```
Expected: the stale `newton.*` imports — proof this code cannot run here.

- [ ] **Step 3: Delete and commit**

```bash
git rm -r gpurec/optim/diagnostics/
git commit -m "chore(optim): remove dead diagnostics scripts (broken newton.* imports, superseded by tests)"
```

> **Decision point for the user:** if you want to keep these as a research reference, replace Step 3 with `git mv gpurec/optim/diagnostics scripts/basin_diagnostics` and fix the `newton.*` imports to `gpurec.optim.*` (`newton.vg` → `gpurec.optim.value_and_grad`, `newton.baselines` → `gpurec.optim.baselines`, `newton.hvp_exact` → `gpurec.optim.hvp_exact`, `newton.optimize` → `gpurec.optim.optimize`). Otherwise delete.

---

## Task P2: Unify the three `*_curvature.py` modules

**The biggest structural win and the highest risk.** `receiver_curvature.py` (437), `origination_curvature.py` (345), and `genewise_curvature.py` (564) implement the *same pipeline* — build joint exact-HVP → gauge-project the softmax null space(s) → Lanczos PD certificate → damped Newton — for different active parameter blocks. Confirmed duplicate function names:

| receiver (θ,α) | origination (θ,α,ω) | genewise (per-family θ) |
|---|---|---|
| `proj_z`, `build_joint_hvp`, `make_gauge_operator`, `certify_joint_min`, `newton_joint`, `receiver_information` | `proj_z`, `build_joint_hvp`, `make_gauge_operator`, `certify_joint_min`, `newton_joint`, `origination_information` | `proj_z_genewise`, `make_gauge_operator_genewise`, `certify_joint_min_genewise`, `newton_joint_genewise`, `_assemble_dense_arrowhead`, … |

`origination_curvature` already **supersedes** `receiver_curvature` (adds the ω block; `origination_curvature` imports from `receiver_curvature`). Target: one gauge/HVP framework parametrized by `active=("theta","alpha","omega")`, with the genewise per-family batching as a thin wrapper.

**Production blast radius (why risk is HIGH):**
- `receiver_curvature` ← imported by `newton_cg.py` (production driver), `origination_curvature.py`, `tests/gates/_verify_s9_curvature.py`.
- `origination_curvature` ← imported by `hvp_exact.py` (production, imported 12×), `genewise_curvature.py`.
- `genewise_curvature` ← imported by `tests/test_genewise_hvp.py` only.

**Risk:** HIGH — touches the two most-depended-on optim modules. Do NOT attempt without P2's characterization gate green first. Estimated net saving ~600 LOC.

**Files:**
- Create: `gpurec/optim/curvature.py` — unified `build_joint_hvp(static, …, active)`, `make_gauge_operator(active)`, `certify_joint_min(…, active)`, `newton_joint(…, active)`, `proj_z(…, active)`, plus `receiver_information` / `origination_information` as thin `active=`-specialized calls.
- Modify: `gpurec/optim/receiver_curvature.py` → shim re-exporting the unified names with `active=("theta","alpha")` defaults (keeps `newton_cg.py` + gate imports working) OR delete after updating importers (user's choice — see decision point).
- Modify: `gpurec/optim/origination_curvature.py` → shim/`active=("theta","alpha","omega")` or delete-and-update.
- Modify: `gpurec/optim/genewise_curvature.py` → keep the genewise batching wrappers (`newton_joint_genewise`, `make_multibatch_joint_hvp_genewise`, `multibatch_joint_vg_genewise`); delegate the gauge/certify core to `curvature.py`.
- Modify importers: `gpurec/optim/newton_cg.py`, `gpurec/optim/hvp_exact.py` (only if names move; a re-export shim avoids editing these).

**Interfaces:**
- Consumes: the exact-HVP from `hvp_exact.py` (`build_point_cache`, `make_exact_hvp`) — unchanged.
- Produces: `build_joint_hvp`, `make_gauge_operator`, `certify_joint_min`, `newton_joint`, `proj_z` (all taking an `active` tuple), `receiver_information`, `origination_information`. These MUST be import-compatible with current call sites in `newton_cg.py`, `hvp_exact.py`, `tests/gates/_verify_s9_curvature.py`, `tests/test_genewise_hvp.py`.

- [ ] **Step 1: Characterization gate — capture current curvature outputs as the oracle**

Before changing anything, run the FD/curvature gates and save their numeric output. On the GPU box:
```bash
pytest tests/test_genewise_hvp.py -q | tee /tmp/p2_genewise_before.txt
python -m tests.gates._verify_s9_curvature   | tee /tmp/p2_s9_before.txt   # after P1a; else gpurec.optim._verify_s9_curvature
```
Expected: PASS + recorded PD-certificate / eigenvalue numbers. These are the bit-for-bit oracle for Step 4.

- [ ] **Step 2: Extract the shared core into `gpurec/optim/curvature.py`**

Move `build_joint_hvp` / `make_gauge_operator` / `certify_joint_min` / `newton_joint` / `proj_z` from `origination_curvature.py` (the superset, θ,α,ω) into `curvature.py`, adding an `active: tuple[str,...] = ("theta","alpha","omega")` parameter that selects which gauge null spaces to project and which blocks to assemble. `("theta","alpha")` reproduces the receiver case.

- [ ] **Step 3: Convert `receiver_curvature.py` and `origination_curvature.py` to thin shims**

```python
# gpurec/optim/receiver_curvature.py
from gpurec.optim.curvature import (
    build_joint_hvp, make_gauge_operator, certify_joint_min, newton_joint, proj_z,
)
# receiver = (theta, alpha) specialization
def receiver_information(*a, **k):
    return _receiver_information_impl(*a, **k)   # or partial(active=("theta","alpha"))
```
Keep `_penalty_hvp`, `_tree_edges`, `_alpha_leak`, `softmax_recipient` wherever their single owner is (receiver-specific helpers). This preserves `newton_cg.py` and gate imports with zero edits to those callers.

- [ ] **Step 4: Gate — outputs identical to Step 1 oracle**

```bash
pytest tests/test_genewise_hvp.py -q | tee /tmp/p2_genewise_after.txt
python -m tests.gates._verify_s9_curvature | tee /tmp/p2_s9_after.txt
diff /tmp/p2_genewise_before.txt /tmp/p2_genewise_after.txt   # test pass/fail unchanged
diff /tmp/p2_s9_before.txt /tmp/p2_s9_after.txt               # numbers bit-for-bit identical
```
Expected: no meaningful diff (PD certificate + eigenvalues identical). If the numbers move, the extraction changed math — revert and redo.

- [ ] **Step 5: Full suite + commit**

```bash
pytest tests/ -q
git add -A
git commit -m "refactor(optim): unify receiver/origination/genewise curvature into curvature.py (active-block parametrized)"
```

> **Decision point:** shims (Step 3) are the low-risk path — they keep `receiver_curvature`/`origination_curvature` as 5-line re-export files (saves ~600 LOC of body, keeps import names). The aggressive path deletes both and rewrites the 3 production/gate importers to `from gpurec.optim.curvature import …`. Recommend shims first; delete in a follow-up once green.

---

## Task P3: Deduplicate the implicit-VJP backward

**`ggn.py::vjp_root_to_theta` is a self-described "faithful copy of `implicit_grad_loglik_vjp_wave`"** (`api/_implicit_grad.py`); both files also define `_e_adjoint_and_theta_vjp`. Estimated saving ~150–200 LOC.

**Risk:** Medium–high — `_implicit_grad.py` is THE production gradient; any drift breaks every `.backward()`. Gate on gradient bit-for-bit.

**Files:**
- Modify: `gpurec/optim/ggn.py` — replace the copied `vjp_root_to_theta` + `_e_adjoint_and_theta_vjp` bodies with a call into the canonical implementation, parametrized by the `seed_root` / `drop_norm` hooks that made the copy diverge.
- Modify: `gpurec/api/_implicit_grad.py` — expose the shared core with `seed_root` / `drop_norm` parameters (default values reproduce today's `implicit_grad_loglik_vjp_wave` behavior exactly).
- Importers to keep working: `newton_cg.py`, `hvp_exact.py` (both import `ggn`).

**Interfaces:**
- Produces: `make_ggn_hvp` (unchanged public surface of `ggn.py`) now backed by the shared VJP.
- The canonical `implicit_grad_loglik_vjp_wave(..., seed_root=None, drop_norm=False)` must return **bit-for-bit** the current gradient at default args.

- [ ] **Step 1: Diff the two implementations to confirm the delta is only `seed_root`/`drop_norm`**

```bash
python - <<'PY'
import difflib, pathlib
a = pathlib.Path("gpurec/api/_implicit_grad.py").read_text().splitlines()
b = pathlib.Path("gpurec/optim/ggn.py").read_text().splitlines()
print("\n".join(difflib.unified_diff(a, b, lineterm="", n=1)[:200]))
PY
```
Expected: the two VJP bodies differ only in the seed/normalization hooks. If they differ in the linear-solve or adjoint math, STOP — they are not safe to merge.

- [ ] **Step 2: Capture the production gradient oracle**

```bash
pytest tests/test_genewise_hvp.py tests/test_optim_golden.py -q | tee /tmp/p3_before.txt
```
(These exercise both the direct backward and the GGN HVP.)

- [ ] **Step 3: Parametrize the canonical VJP and delegate from ggn.py**

Add `seed_root`/`drop_norm` params to `implicit_grad_loglik_vjp_wave` (defaults reproduce current behavior). In `ggn.py`, replace the copied body with `from gpurec.api._implicit_grad import implicit_grad_loglik_vjp_wave` and call it with the GGN-specific `seed_root`/`drop_norm`.

- [ ] **Step 4: Gate — gradients bit-for-bit identical**

```bash
pytest tests/test_genewise_hvp.py tests/test_optim_golden.py -q | tee /tmp/p3_after.txt
diff /tmp/p3_before.txt /tmp/p3_after.txt
```
Expected: identical pass set. Any HVP/gradient numeric change → revert.

- [ ] **Step 5: Full suite + commit**

```bash
pytest tests/ -q
git add -A
git commit -m "refactor: dedup implicit-VJP — ggn.vjp_root_to_theta delegates to api._implicit_grad"
```

---

## Task P4: Split `core/kernels/wave_backward.py` (2,446 LOC)

**The single largest file, 2× the next.** Not duplication — just too big to hold in context. Net LOC change ≈ 0; the win is readability and reviewability. Same package, no external import changes if re-exported.

**Risk:** Low–medium (pure move within a package; risk is a missed symbol or a Triton `@jit` closure that must stay co-located).

**Files:**
- Create: `gpurec/core/kernels/wave_backward_kernels.py` (the `@triton.jit` kernels + launch wrappers) and keep `gpurec/core/kernels/wave_backward.py` as the Python-facing VJP orchestration that imports them. (Exact split boundary = kernel definitions vs. the setup/layout/adjoint-accumulation Python.)
- Modify: `gpurec/core/kernels/wave_backward.py` — re-export moved public names so `from gpurec.core.kernels.wave_backward import <name>` keeps working for its 3 importers.

**Interfaces:**
- Produces: every public name currently importable from `wave_backward` remains importable from `wave_backward` (via re-export). Callers: `core/inference/`, `api/_implicit_grad.py`, `optim/*` — none should need edits.

- [ ] **Step 1: List the public symbols and their importers (the contract to preserve)**

```bash
grep -noE "^(def|class)\s+\w+" gpurec/core/kernels/wave_backward.py
grep -rnE "from gpurec\.core\.kernels\.wave_backward import|wave_backward\." gpurec/ tests/
```
Expected: the set of names other modules rely on — these must stay importable from `wave_backward`.

- [ ] **Step 2: Move the `@triton.jit` kernels into `wave_backward_kernels.py`**

Cut the kernel definitions + their direct launch helpers into the new file; keep Triton autotune/heuristic decorators attached. Leave the orchestration/adjoint-accumulation Python in `wave_backward.py`, which now does `from gpurec.core.kernels.wave_backward_kernels import *` (or explicit names).

- [ ] **Step 3: Verify the import contract is intact**

```bash
python -c "import gpurec.core.kernels.wave_backward as wb; print([n for n in dir(wb) if not n.startswith('__')])"
grep -rnE "from gpurec\.core\.kernels\.wave_backward import" gpurec/ tests/   # every name still resolvable
```

- [ ] **Step 4: Gate — the backward still matches (any gradient test)**

```bash
pytest tests/test_genewise_hvp.py -q          # exercises wave backward
pytest tests/ -q
```
Expected: unchanged pass set.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor(kernels): split wave_backward.py into kernels + orchestration"
```

---

## Task P5: Fold the test-only `genewise_hessian_blocks` out of production

**`genewise_curvature.py::genewise_hessian_blocks` is docstring-flagged TEST-ONLY** ("the production genewise fit never assembles blocks") and is imported only by `tests/test_genewise_hvp.py:68`. Small (~22 LOC of the 564-line file). Cleanest folded into P2; standalone otherwise.

**Risk:** Low.

**Files:**
- Modify: `gpurec/optim/genewise_curvature.py` — remove `genewise_hessian_blocks` (and any helper used *only* by it, e.g. verify with grep).
- Modify: `tests/test_genewise_hvp.py` — move the `genewise_hessian_blocks` implementation into the test module (it is the fp32-vs-fp64 golden fixture) OR into `tests/gates/`.

**Interfaces:**
- After the move, `tests/test_genewise_hvp.py` defines/imports `genewise_hessian_blocks` locally; no production module references it.

- [ ] **Step 1: Confirm the only consumer is the test**

```bash
grep -rnE "genewise_hessian_blocks" gpurec/ tests/
```
Expected: the `def` in `genewise_curvature.py` + two references in `tests/test_genewise_hvp.py`, nothing else.

- [ ] **Step 2: Move the function into the test module**

Cut `genewise_hessian_blocks` (and `_assemble_dense_arrowhead` **only if** it has no other caller — check with `grep -n _assemble_dense_arrowhead gpurec/optim/genewise_curvature.py`) into `tests/test_genewise_hvp.py`; drop the `from gpurec.optim.genewise_curvature import genewise_hessian_blocks` line.

- [ ] **Step 3: Gate + commit**

```bash
pytest tests/test_genewise_hvp.py -q
pytest tests/ -q
git add -A
git commit -m "test(genewise): move test-only genewise_hessian_blocks into its test module"
```

---

## Housekeeping (optional, trivial — fold into any commit)

- [ ] Remove the two dead locals in `gpurec/core/kernels/wave_tangent.py:247` (`colw = pi_w  # unused`) and `:428` (`dummy = Pi_in  # unused placeholder`). Gate: `pytest tests/ -q` unchanged.
- [ ] Fix stale `newton/` / `kbench` module references in `optim/*` docstrings and `docs/optim/*.md` (find-replace `newton.` → `gpurec.optim.`, `python -m newton.` → `python -m gpurec.optim.`). Docs-only, no gate needed.

---

## Summary: effort / risk / payoff

| Task | LOC impact | Risk | Production code touched? | Depends on |
|---|---|---|---|---|
| **P1a** move gate scripts | −1,744 from pkg | Low | No (tests only) | — |
| **P1b** delete diagnostics | −558 | Very low | No (dead) | — |
| **P2** unify curvature | ~−600 | **High** | Yes (`newton_cg`, `hvp_exact`) | Task 0 gate |
| **P3** dedup implicit-VJP | ~−150–200 | Med–high | Yes (`_implicit_grad`, `ggn`) | Task 0 gate |
| **P4** split wave_backward | ~0 (readability) | Low–med | Yes (mechanical, re-export) | — |
| **P5** move test-only fn | small | Low | Yes (trivial) | fold into P2 |

**Recommended order if doing all:** P1b → P1a → P4 → P5 → P3 → P2 (cheapest/safest first; the two high-risk math dedups last, each behind its bit-for-bit gate).

## Self-review notes
- Coverage: every P1–P5 item from the architecture overview maps to a task above; housekeeping items included.
- No placeholders: all file paths, the 8 import-rewrite sites, and the exact `git mv`/`git rm` commands are spelled out.
- Type/name consistency: P2's produced names (`build_joint_hvp`, `make_gauge_operator`, `certify_joint_min`, `newton_joint`, `proj_z`) match the current call sites in `newton_cg.py`/`hvp_exact.py`/gates; P3 keeps `make_ggn_hvp` as ggn.py's public surface; P4 preserves every `wave_backward` import name via re-export.
