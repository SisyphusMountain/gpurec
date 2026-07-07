# gpurec P2 + P3: Curvature & Implicit-VJP De-duplication Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans (inline, bit-for-bit gates) or superpowers:subagent-driven-development. Steps use `- [ ]` checkboxes.

**Goal:** Collapse the two remaining duplications identified in the architecture overview — the three-way `*_curvature.py` pipeline (P2) and the copied implicit-VJP backward (P3) — with **zero change to production numerics** (bit-for-bit gradients, HVPs, and PD certificates).

**Architecture:** These touch the most-depended-on optim/api modules, so unlike the top-4 (mechanical moves) these are **numerics-preserving refactors gated on identical output**, not just "tests still green." Each task captures an fp64 numeric oracle *before* touching code and requires the *same numbers* after. Prefer **re-export shims** (extract shared core, leave thin forwarding modules) over deleting-and-rewriting-callers: shims keep `newton_cg.py`, `hvp_exact.py`, and the gate scripts importing the same names, shrinking blast radius.

**Tech Stack:** Python, PyTorch (fp32 kernels, fp64 gates), Triton, pytest. GPU box (RTX 4090) required — most gates need CUDA.

## Global Constraints (same as the top-4 work)

- **Git safety:** never `checkout`/`restore`/`reset` a file with uncommitted changes; the working tree has untracked experiment dirs (`experiments/`, `ghost_experiments/`, `docs/gergely_comparison/`, …) that must **not** be staged. Stage only the exact paths each task names — never `git add -A`.
- **Branch:** continue on `refactor/optim-cleanup` (holds P1a/P1b/P4/P5 + docs) unless it has been merged, in which case cut `refactor/curvature-vjp-dedup` from the merge target.
- **No behavior change / no clamp / no scope creep** — do exactly the dedup; leave adjacent code, public names, and kernel math alone.
- **Numeric oracle protocol:** each task saves `before.json`/`before.txt` from the fp64 gates, makes the change, re-runs, and `diff`s. The refactor is accepted **only if the certificate/eigenvalue/gradient numbers are identical** (fp64 gates are deterministic here). If any number moves, the extraction changed math → revert.
- **Verification command:** `pytest tests/ -q` on the GPU box; per-task fast subsets named below. (Baseline note: `tests/test_cli.py` vs `tests/gpurax/test_cli.py` share a basename → 1 pre-existing collection error; ignore with `--ignore=tests/gpurax`. Whole-suite collects 135 tests clean otherwise.)

---

## Task 0: Numeric baseline (once, before P2/P3)

**Files:** none.

- [ ] **Step 1: Branch state**
```bash
cd /home/enzo/Documents/git/gpurec/consolidate-release
git status                       # only untracked experiment files; no tracked mods
git log --oneline -6             # confirm the P1a/P1b/P4/P5 + docs commits are present
```

- [ ] **Step 2: Capture the fp64 numeric oracles that P2 and P3 must preserve**
On the GPU box, save raw numbers (not just pass/fail):
```bash
python -m gates._verify_s9_curvature --synthetic 2>&1 | tee /tmp/dedup_s9_before.txt   # receiver (theta,alpha) PD cert
pytest tests/test_genewise_hvp.py tests/test_optim_golden.py -q 2>&1 | tee /tmp/dedup_tests_before.txt
```
Expected: PASS + recorded certificate/eigenvalue/rel-error numbers. These are the P2 (certificate) and P3 (gradient/HVP) oracles.

---

## Task P2: Unify `receiver_curvature` + `origination_curvature` (+ genewise core)

**Duplication (confirmed):** `receiver_curvature.py` (437 LOC, block `(θ,α)`) and `origination_curvature.py` (345 LOC, block `(θ,α,ω)`) implement the same five-function pipeline with signatures differing only by the extra `omega`/`S` arguments:

| function | receiver sig | origination sig |
|---|---|---|
| `proj_z` | `(u, theta_numel)` | `(u, theta_numel, S)` |
| `build_joint_hvp` | `(static, theta, alpha, *, …)` | `(static, theta, alpha, omega, *, …)` |
| `make_gauge_operator` | `(hvp, theta_numel, *, …)` | `(hvp, theta_numel, S, *, …)` |
| `certify_joint_min` | `(static, theta, alpha, *, …)` | `(static, theta, alpha, omega, *, …)` |
| `newton_joint` | `(static, theta0, alpha0, *, …)` | `(static, theta0, alpha0, omega0, *, …)` |

`origination` is the strict **(θ,α,ω) generalization**: with `omega=None` it must reproduce the receiver `(θ,α)` case (one softmax gauge-null instead of two). `genewise_curvature.py` holds per-family `_genewise` variants (`proj_z_genewise`, `make_gauge_operator_genewise`, `certify_joint_min_genewise`, `newton_joint_genewise`) plus the batching wrappers.

**Importers (blast radius):**
- `receiver_curvature` ← `newton_cg.py` (production), `origination_curvature.py`, `gates/_verify_s9_curvature.py`. Also its helpers `_penalty_hvp`, `_tree_edges` are imported directly by `gates/_verify_s9_curvature.py`.
- `origination_curvature` ← `hvp_exact.py` (production, imported 12×), `genewise_curvature.py`.
- `genewise_curvature` ← `tests/test_genewise_hvp.py`.

**Strategy (shim-first, lowest risk):** promote origination's `(θ,α,ω)` pipeline to a canonical `gpurec/optim/curvature.py`, with `omega=None ⇒ (θ,α)` reproducing receiver exactly. Convert `receiver_curvature.py` and `origination_curvature.py` to thin re-export shims so every current importer keeps working unchanged. Genewise keeps its per-family wrappers but delegates the gauge/certify core.

**Risk:** HIGH (production `newton_cg` + `hvp_exact`). Estimated net saving ~450–650 LOC.

**Files:**
- Create: `gpurec/optim/curvature.py` — canonical `build_joint_hvp(static, theta, alpha, omega=None, *, …)`, `make_gauge_operator(hvp, theta_numel, S=None, *, …)`, `certify_joint_min(…, omega=None, …)`, `newton_joint(…, omega0=None, …)`, `proj_z(u, theta_numel, S=None)`. `omega=None`/`S=None` selects the receiver `(θ,α)` behavior (single alpha gauge-null); non-None adds the omega block + second null.
- Modify → shim: `gpurec/optim/receiver_curvature.py` re-exports the canonical names bound to the `(θ,α)` specialization, and keeps the alpha-only helpers `_penalty_hvp`, `_tree_edges`, `proj_alpha`, `softmax_recipient`, `_alpha_leak`, `receiver_information` (imported by `gates/_verify_s9`).
- Modify → shim: `gpurec/optim/origination_curvature.py` re-exports the canonical names (full `(θ,α,ω)`) + keeps `softmax_origination`, `origination_information`.
- Modify: `gpurec/optim/genewise_curvature.py` — genewise gauge/certify helpers delegate to `curvature.py` where they duplicate it; batching wrappers stay.
- **Do NOT edit** `newton_cg.py`, `hvp_exact.py`, `gates/_verify_s9_curvature.py` — the shims keep their imports valid. (If a shim can't preserve a name, that's a signal to stop and reconsider, not to edit the caller.)

**Interfaces:**
- Produces: `curvature.build_joint_hvp / make_gauge_operator / certify_joint_min / newton_joint / proj_z`, all with `omega`/`S` optional. `receiver_curvature.*` and `origination_curvature.*` names remain importable and behavior-identical.
- Consumes: `hvp_exact.build_point_cache` / `make_exact_hvp` (unchanged), `cg.lanczos_*` (unchanged).

- [ ] **Step 1: Pin the (θ,α) receiver certificate oracle**
```bash
python -m gates._verify_s9_curvature --synthetic 2>&1 | tee /tmp/p2_recv_before.txt   # A: no-GPU synthetic PD cert
python -m gates._verify_s9_curvature --live       2>&1 | tee /tmp/p2_recv_live_before.txt
```
Record the certificate min-eigenvalue / gauge-null dimensions.

- [ ] **Step 2: Write `curvature.py` = origination's pipeline with `omega=None` guards**
Move `build_joint_hvp/make_gauge_operator/certify_joint_min/newton_joint/proj_z` from `origination_curvature.py`; add `if omega is None:` branches that skip the omega block and second gauge-null, so `(θ,α)` == today's receiver path. Where receiver and origination differ only by omega threading, the `None` branch is the receiver body verbatim.

- [ ] **Step 3: Convert `origination_curvature.py` to a shim**
```python
from gpurec.optim.curvature import (
    build_joint_hvp, make_gauge_operator, certify_joint_min, newton_joint, proj_z,
)
from gpurec.optim.origination_gauge import softmax_origination, origination_information  # or keep inline
```
Keep any origination-only helper bodies that are not part of the shared pipeline.

- [ ] **Step 4: Convert `receiver_curvature.py` to a shim (θ,α specialization)**
Re-export the canonical names (partial-applied / documented as the `omega=None` case) and **retain** `_penalty_hvp`, `_tree_edges`, `proj_alpha`, `softmax_recipient`, `_alpha_leak`, `receiver_information` (their bodies, unchanged — `gates/_verify_s9` imports `_penalty_hvp`, `_tree_edges`, `build_joint_hvp` from here).

- [ ] **Step 5: Gate — certificate numbers bit-for-bit**
```bash
python -m gates._verify_s9_curvature --synthetic 2>&1 | tee /tmp/p2_recv_after.txt
python -m gates._verify_s9_curvature --live       2>&1 | tee /tmp/p2_recv_live_after.txt
diff /tmp/p2_recv_before.txt      /tmp/p2_recv_after.txt
diff /tmp/p2_recv_live_before.txt /tmp/p2_recv_live_after.txt
pytest tests/test_genewise_hvp.py -q     # origination/genewise HVP path via hvp_exact
```
Expected: **zero numeric diff** on the certificates; genewise gate green. Any eigenvalue drift ⇒ the `omega=None` branch isn't the receiver body ⇒ revert Step 2.

- [ ] **Step 6: Delegate genewise core (optional, same commit or follow-up)**
Point `proj_z_genewise`/`make_gauge_operator_genewise`/`certify_joint_min_genewise` at `curvature.py` helpers where identical; re-run `pytest tests/test_genewise_hvp.py -q` (bit-for-bit).

- [ ] **Step 7: Full suite + commit**
```bash
pytest tests/ -q
git add gpurec/optim/curvature.py gpurec/optim/receiver_curvature.py \
        gpurec/optim/origination_curvature.py gpurec/optim/genewise_curvature.py
git diff --cached --name-only | grep -vE '^gpurec/optim/' && echo "!! unexpected" || echo clean
git commit -m "refactor(optim): unify receiver/origination/genewise curvature into curvature.py (omega-optional)"
```

> **Decision point (needs your call before Step 3):** *shims* keep `receiver_curvature`/`origination_curvature` as ~10-line re-export files (saves the ~600 LOC of duplicated bodies, zero caller edits — recommended). The *aggressive* variant deletes both modules and rewrites the 3 importers to `from gpurec.optim.curvature import …` (cleaner tree, but edits production `newton_cg`/`hvp_exact` and the gate). Recommend shims now; delete in a separate follow-up once green.

---

## Task P3: De-duplicate the implicit-VJP backward

**Duplication (confirmed):** `ggn.py::vjp_root_to_theta` is — per its own docstring — the production `implicit_grad_loglik_vjp_wave` (`api/_implicit_grad.py`) generalized with two knobs: an arbitrary root cotangent `seed_root` and `drop_norm`. The docstring states verbatim: *"With `seed_root=None` the loss seed `-softmax2(Pi_root)` is used and `drop_norm` should be False to reproduce the real gradient."* Both files also carry a `_e_adjoint_and_theta_vjp` (the shared E-adjoint + wave-adjoint core), which has **drifted** between the two copies.

**Importers (blast radius):** `ggn.py` ← `newton_cg.py`, `hvp_exact.py` (both production). `api/_implicit_grad.py` is THE `.backward()` gradient — every fit and HVP flows through it. Highest-stakes change in the whole cleanup.

**Strategy:** make the production `implicit_grad_loglik_vjp_wave` the single implementation, adding `seed_root=None` and `drop_norm=False` parameters whose defaults reproduce today's behavior bit-for-bit. `ggn.vjp_root_to_theta` becomes a thin call into it with the GN-specific `seed_root`/`drop_norm`. Delete the drifted `_e_adjoint_and_theta_vjp` copy in `ggn.py`; keep the one in `_implicit_grad.py`.

**Risk:** MED–HIGH. Do **after** P2 and only with the gradient oracle green.

**Files:**
- Modify: `gpurec/api/_implicit_grad.py` — add `seed_root=None`, `drop_norm=False` params to `implicit_grad_loglik_vjp_wave` (and its `_e_adjoint_and_theta_vjp` helper as needed); defaults = current behavior.
- Modify: `gpurec/optim/ggn.py` — replace `vjp_root_to_theta`'s body with a call into the canonical function; delete the local `_e_adjoint_and_theta_vjp` copy. Keep `make_ggn_hvp` (ggn's public surface) intact.
- **Do NOT edit** `newton_cg.py`, `hvp_exact.py` — they import `make_ggn_hvp`/`vjp_root_to_theta` by name, preserved.

**Interfaces:**
- Produces: `implicit_grad_loglik_vjp_wave(..., seed_root=None, drop_norm=False)` returns **bit-for-bit** the current gradient at default args; `ggn.vjp_root_to_theta` / `make_ggn_hvp` keep their signatures.

- [ ] **Step 1: Confirm the delta is only `seed_root`/`drop_norm`**
Diff the two VJP bodies (and the two `_e_adjoint_and_theta_vjp`); confirm they differ only in the seed selection (`-softmax2(Pi_root)` vs arbitrary `seed_root`) and the normalization-term drop. If they differ in the E-adjoint linear solve or wave-adjoint accumulation, **STOP** — they are not safe to merge as-is; reconcile the drift first and report it.

- [ ] **Step 2: Capture the gradient + HVP oracle**
```bash
pytest tests/test_optim_golden.py tests/test_genewise_hvp.py -q 2>&1 | tee /tmp/p3_before.txt
# test_optim_golden pins loss+grad bit-for-bit vs a checkpoint; test_genewise_hvp pins the GGN/exact HVP
```

- [ ] **Step 3: Parametrize the canonical VJP; delegate from ggn**
Add the two params to `implicit_grad_loglik_vjp_wave` (defaults reproduce current output). In `ggn.py`: `from gpurec.api._implicit_grad import implicit_grad_loglik_vjp_wave`; `vjp_root_to_theta(...)` forwards with its `seed_root`/`drop_norm`. Remove the duplicated `_e_adjoint_and_theta_vjp`.

- [ ] **Step 4: Gate — gradient + HVP bit-for-bit**
```bash
pytest tests/test_optim_golden.py tests/test_genewise_hvp.py -q 2>&1 | tee /tmp/p3_after.txt
diff /tmp/p3_before.txt /tmp/p3_after.txt
```
Expected: identical pass set **and** identical printed rel-errors. Any HVP/gradient numeric change ⇒ revert (the `drop_norm=False`/`seed_root=None` defaults didn't reproduce the original path).

- [ ] **Step 5: Full suite + commit**
```bash
pytest tests/ -q
git add gpurec/api/_implicit_grad.py gpurec/optim/ggn.py
git diff --cached --name-only | grep -vE '^gpurec/(api/_implicit_grad.py|optim/ggn.py)$' && echo "!! unexpected" || echo clean
git commit -m "refactor: dedup implicit-VJP — ggn.vjp_root_to_theta delegates to api._implicit_grad (seed_root/drop_norm)"
```

---

## Sequencing & risk

| Task | LOC saved | Risk | Production touched | Gate |
|---|---|---|---|---|
| **P2** unify curvature | ~450–650 | High | `newton_cg`, `hvp_exact` (via shims: none directly) | `_verify_s9` cert bit-for-bit + genewise HVP |
| **P3** dedup VJP | ~150–200 | Med–High | `_implicit_grad` (the gradient), `ggn` | `test_optim_golden` grad + `test_genewise_hvp` HVP bit-for-bit |

**Order:** P2 first (isolated to the curvature cluster), then P3 (touches the core gradient — do last, alone, behind its bit-for-bit gate). Each is a single commit; if a gate shows any numeric drift, revert that task and reassess — do not iterate fixes on a diverging refactor (after two failed attempts, re-diagnose from scratch).

**Not running the full 178-test GPU suite each step** is acceptable during iteration (use the named subsets), but Step "Full suite" of each task must run it once before committing.
