# Analytic HVP as Default Genewise Hessian Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the forward-difference (FD) Hessian in `fit_genewise`'s trust-region Newton step (and its `certify` diagnostic) with the already-validated analytic-HVP construction, and apply the same swap to the separate benchmark driver script.

**Architecture:** One shared `_analytic_hessian(m, theta, pi_cur)` helper per file (duplicated, not cross-imported, since the two files are deliberately independent — see spec), computing the per-family `[G,3,3]` Hessian via 3 broadcast unit-theta-component analytic-HVP probes (each opted into the tangent-adjoint warm-start via `probe_id=j`), replacing the FD block at both call sites (optimization loop, certify block) in each file.

**Tech Stack:** Python, PyTorch, `gpurec.solver.hvp_exact.make_exact_hvp` (already implemented, tested, and merged into `dev` by the immediately preceding HVP-warm-start plan).

## Global Constraints

- `probe_id=j` must be passed on every probe — this is what activates the tangent-adjoint warm-start; omitting it silently loses the validated speedup (correctness is unaffected either way, so this can't be caught by a correctness test — it must be checked by direct code review against the reference implementation below).
- `certify`'s diagnostic Hessian also switches to analytic (no FD anywhere in either file afterward) — user's explicit call, not kept as an independent cross-check.
- `fd_eps` is removed entirely from `fit_genewise`'s signature and `GENEWISE_REFERENCE` — a breaking change to the public signature, not deprecated/kept-for-compat.
- `mu` (eigenvalue-floor convexification) is unchanged in both files — needed regardless of Hessian source.
- Reference spec: `docs/superpowers/specs/2026-07-21-analytic-hvp-default-design.md`.
- Do not touch `gpurec/solver/hvp_exact.py`, `gpurec/solver/genewise_curvature.py`, or any CG-based joint Newton solver (`newton_joint_genewise`, `origination_curvature.newton_joint`) — those are out of scope.

---

## File Map

- `gpurec/fit/genewise_fit.py` — the production recipe. Add `_analytic_hessian`, replace both FD blocks, remove `fd_eps`, update the module docstring's step-2 description, add imports.
- `tests/test_genewise_hvp.py` — new direct correctness test for `fit_genewise` (doesn't exist today).
- `tests/test_reference_defaults.py`, `tests/test_config_rates.py`, `tests/test_config_wiring.py` — verify none break from the `fd_eps` removal (update if they do).
- `experiments/sanderson_cv/bench_genewise_warm_rebatch.py` — the parallel benchmark driver. Same swap, independently (no cross-import).

---

### Task 1: Replace FD Hessian with analytic HVP in `gpurec/fit/genewise_fit.py`

**Files:**
- Modify: `gpurec/fit/genewise_fit.py`
- Modify (only if verification in Step 6 finds a break): `tests/test_reference_defaults.py`, `tests/test_config_rates.py`, `tests/test_config_wiring.py`
- Test: `tests/test_genewise_hvp.py` (new test added here)

**Interfaces:**
- Produces: `_analytic_hessian(m, theta, pi_cur) -> torch.Tensor` (module-level, in `genewise_fit.py`), a `[G,3,3]` symmetric Hessian. `fit_genewise`'s signature no longer has an `fd_eps` parameter; `GENEWISE_REFERENCE` no longer has an `fd_eps` key.
- Consumes: `gpurec.solver.value_and_grad.forward_solve`, `gpurec.solver.hvp_exact.make_exact_hvp` (both already implemented on `dev`).

- [ ] **Step 1: Write the regression test for `fit_genewise` (run it against today's FD code first — this is a behavior-preserving refactor, not new functionality, so the test proves "still converges correctly" both before and after)**

Add to `tests/test_genewise_hvp.py` (uses the same `_D = "tests/data/alerax/test_trees_200"` fixture already defined at the top of this file):

```python
@pytest.mark.gpu
def test_fit_genewise_converges_on_small_fixture():
    """Direct behavior test for fit_genewise itself (previously only covered indirectly via
    signature/config-wiring checks). Runs the real recipe end-to-end on a handful of families
    and checks it actually reaches a converged, finite optimum -- this must pass both before and
    after swapping the Hessian source, since the swap must not change the recipe's behavior
    contract, only its internal Hessian construction."""
    from gpurec.fit.genewise_fit import fit_genewise

    res = fit_genewise(
        f"{_D}/sp.nwk", [f"{_D}/g.nwk"] * 4,
        device="cuda", dtype=torch.float32,
        adam_steps=5, pi_tiers=(16,), neu_opt=16, neu_cert=16,
        max_iter=60, certify=True, verbose=False,
    )
    assert res["n_families"] == 4
    assert torch.isfinite(res["theta"]).all()
    assert res["converged"] == 4
    assert res["pg_max"] < 1e-2
    assert math.isfinite(res["loss_bits"])
```

- [ ] **Step 2: Run the test to confirm it passes against today's FD-Hessian code (baseline)**

Run: `GPUREC_PREPROCESS_PATH=$(pwd)/crates/gpurec-preprocess/target/debug/libgpurec_preprocess.so .venv/bin/python -m pytest tests/test_genewise_hvp.py::test_fit_genewise_converges_on_small_fixture -v -m gpu`

Expected: PASS (this confirms the test itself is correct and the fixture converges, before touching any production code — the swap must not break this).

- [ ] **Step 3: Add imports and the `_analytic_hessian` helper**

In `gpurec/fit/genewise_fit.py`, add to the imports block (after the existing `from gpurec.optimization import ...` line):

```python
from gpurec.solver.value_and_grad import forward_solve
from gpurec.solver.hvp_exact import make_exact_hvp
```

Add the helper function right after `_resolve_gene_trees` (before `fit_genewise`'s definition):

```python
def _analytic_hessian(m, theta, pi_cur):
    """Per-family [G,3,3] curvature via 3 broadcast analytic-HVP probes, warm-started via
    probe_id. Mirrors hvp_exact.py's _make_exact_hvp_streaming genewise gather/scatter for
    single-batch (which the top-level make_exact_hvp does NOT do for you), and lets it handle
    multi-batch itself."""
    G = theta.shape[0]
    dev, dtype = theta.device, theta.dtype
    rw = m.receiver_weights.detach()
    if len(m.batch_statics) > 1:
        hvp = make_exact_hvp(m.batch_statics, theta, rw, None, tangent_self_iters=pi_cur)
        cols = []
        for j in range(3):
            u = torch.zeros(G, 3, device=dev, dtype=dtype); u[:, j] = 1.0
            cols.append(hvp(u.reshape(-1), probe_id=j)[: G * 3].reshape(G, 3))
        H = torch.stack(cols, dim=-1)
    else:
        static = m.batch_statics[0]
        fam = static.family_index_tensor.to(dev)
        theta_b = theta.index_select(0, fam).contiguous()
        _l, sv = forward_solve(m.batch_statics, theta, rw)
        hvp = make_exact_hvp(m.batch_statics, theta_b, rw, sv, tangent_self_iters=pi_cur)
        cols = []
        for j in range(3):
            u_b = torch.zeros(G, 3, device=dev, dtype=dtype); u_b[:, j] = 1.0
            out_b = hvp(u_b.reshape(-1), probe_id=j)[: G * 3].reshape(G, 3)
            col = torch.zeros(G, 3, device=dev, dtype=dtype)
            col.index_add_(0, fam, out_b)
            cols.append(col)
        H = torch.stack(cols, dim=-1)
    return 0.5 * (H + H.transpose(1, 2))
```

- [ ] **Step 4: Replace the optimization loop's FD Hessian block**

Currently reads:

```python
                if it % 5 == 0 or Hd is None or Hd.shape[0] != sub.shape[0]:
                    H = torch.zeros(sub.shape[0], 3, 3, device=dev, dtype=dtype)
                    for j in range(3):
                        tp = sub.clone(); tp[:, j] += fd_eps; _, gp = lg(m, tp)
                        H[:, :, j] = (gp - g) / fd_eps            # forward difference (reuse base g) -> 3 evals
                    H = 0.5 * (H + H.transpose(1, 2))
                    e, V = torch.linalg.eigh(H)
                    Hd = V @ torch.diag_embed(e.clamp(min=mu)) @ V.transpose(1, 2)   # convexify -> PD
```

Replace with:

```python
                if it % 5 == 0 or Hd is None or Hd.shape[0] != sub.shape[0]:
                    H = _analytic_hessian(m, sub, pi_cur)
                    e, V = torch.linalg.eigh(H)
                    Hd = V @ torch.diag_embed(e.clamp(min=mu)) @ V.transpose(1, 2)   # convexify -> PD
```

(`pi_cur` is already in scope — it's the enclosing `for pi_idx, pi_cur in enumerate(pis):` loop variable. `g`, the base gradient, is still computed via `lg(m, sub)` earlier in this same iteration and is unaffected by this change — it's only consumed afterward by the Newton right-hand-side, not by the Hessian construction itself.)

- [ ] **Step 5: Replace the certify block's FD Hessian**

Currently reads:

```python
            H = torch.zeros(F_all, 3, 3, device=dev, dtype=dtype)
            for j in range(3):
                tp = theta.clone(); tp[:, j] += fd_eps; _, gp = lg(mfull, tp)
                tm = theta.clone(); tm[:, j] -= fd_eps; _, gm = lg(mfull, tm)
                H[:, :, j] = (gp - gm) / (2 * fd_eps)
            H = 0.5 * (H + H.transpose(1, 2))
```

Replace with:

```python
            H = _analytic_hessian(mfull, theta, cert_pi)
```

(`cert_pi = max(pis)` is computed earlier in the function, and `mfull` is built at that same tier a few lines above this block — both already in scope.)

- [ ] **Step 6: Remove `fd_eps` from the signature, `GENEWISE_REFERENCE`, and the module docstring**

In the `GENEWISE_REFERENCE` dict, change:
```python
    fd_eps=1e-2, mu=1e-2, fwd_tol=1e-3, improve_frac=0.8, verify_drop=True, eager_defer=True,
```
to:
```python
    mu=1e-2, fwd_tol=1e-3, improve_frac=0.8, verify_drop=True, eager_defer=True,
```

In `fit_genewise`'s signature, remove the line:
```python
    fd_eps: float = 1e-2,
```

In the module docstring, change:
```
  2. **Box-constrained trust-region Newton** on the per-family 3x3 **forward-difference** Hessian
     (3 evals, reusing the base gradient; eigenvalue-clamped to ``mu`` -> PD), converging on the
     per-family projected gradient ``|Pg| < tol``.
```
to:
```
  2. **Box-constrained trust-region Newton** on the per-family 3x3 **analytic-HVP** Hessian (3
     broadcast unit-theta-component probes, warm-started across repeated rebuilds; eigenvalue-clamped
     to ``mu`` -> PD), converging on the per-family projected gradient ``|Pg| < tol``.
```

- [ ] **Step 7: Run the regression test to confirm it still passes post-swap**

Run: `GPUREC_PREPROCESS_PATH=$(pwd)/crates/gpurec-preprocess/target/debug/libgpurec_preprocess.so .venv/bin/python -m pytest tests/test_genewise_hvp.py::test_fit_genewise_converges_on_small_fixture -v -m gpu`

Expected: PASS (same assertions as Step 2 — this is the whole point of a behavior-preserving refactor: identical pass/fail outcome, different internal implementation).

- [ ] **Step 8: Verify no other test references the removed `fd_eps`, fix if any do**

Run: `grep -rn "fd_eps" tests/`

If this returns any hits, read the surrounding test and update it: `tests/test_reference_defaults.py` likely checks `fit_genewise`'s signature defaults against `GENEWISE_REFERENCE` generically (by iterating both and asserting equality per-key) — if so, removing `fd_eps` from both together should keep it passing with no edit needed; only edit it if it explicitly names `fd_eps`. `tests/test_config_rates.py::test_fit_genewise_signature_defaults_come_from_genewise_preset` and `::test_fit_genewise_still_uses_1e6_2p0_box` check `min_rate`/`max_rate`, not `fd_eps` — expected to be unaffected, but confirm by reading them if the grep above shows a hit in this file. `tests/test_config_wiring.py`'s stubbed-model tests use `adam_steps=0` — read `_run_fit_genewise_capture` to confirm it stubs out `GeneReconModel` before the Hessian-construction code ever runs (so it should be unaffected regardless), and update only if the grep shows a hit here.

If the grep returns nothing, this step needs no changes — just confirms the removal was clean.

- [ ] **Step 9: Run the full genewise HVP test suite**

Run: `GPUREC_PREPROCESS_PATH=$(pwd)/crates/gpurec-preprocess/target/debug/libgpurec_preprocess.so .venv/bin/python -m pytest tests/test_genewise_hvp.py tests/test_reference_defaults.py tests/test_config_rates.py tests/test_config_wiring.py -v -m gpu`

Expected: all pass, including the new test and every pre-existing one.

- [ ] **Step 10: Commit**

```bash
git add gpurec/fit/genewise_fit.py tests/test_genewise_hvp.py
git commit -m "Make analytic HVP the default genewise Hessian, remove FD"
```

(If Step 8 required edits to other test files, `git add` those too in this same commit.)

---

### Task 2: Apply the same swap to `experiments/sanderson_cv/bench_genewise_warm_rebatch.py`

**Files:**
- Modify: `experiments/sanderson_cv/bench_genewise_warm_rebatch.py`

**Interfaces:**
- Consumes: `gpurec.solver.value_and_grad.forward_solve`, `gpurec.solver.hvp_exact.make_exact_hvp` (same as Task 1).
- Produces: a module-level `_analytic_hessian(m, theta, pi_cur)` in this file too (duplicated from Task 1's, not imported — this script is deliberately standalone, confirmed it imports nothing from `genewise_fit.py` today).

This file has no existing automated test (it's a standalone CLI benchmark script, run manually against real datasets) — validation for this task is a manual smoke run at modest scale, not a pytest step.

- [ ] **Step 1: Add imports**

Add after the existing `from gpurec.core.inference.solver import solve_forward_residual` line:

```python
from gpurec.solver.value_and_grad import forward_solve
from gpurec.solver.hvp_exact import make_exact_hvp
```

- [ ] **Step 2: Add the `_analytic_hessian` helper**

Add it after the existing `def forward_resid(m, th, pi):` helper (matching this file's existing convention of plain top-level functions using module globals `DEV`/`DT` directly, not parameters):

```python
def _analytic_hessian(m, theta, pi_cur):
    """Per-family [G,3,3] curvature via 3 broadcast analytic-HVP probes, warm-started via
    probe_id. Duplicated from gpurec/fit/genewise_fit.py's helper of the same name (not imported
    -- this script is a standalone reimplementation, no cross-import by design)."""
    G = theta.shape[0]
    rw = m.receiver_weights.detach()
    if len(m.batch_statics) > 1:
        hvp = make_exact_hvp(m.batch_statics, theta, rw, None, tangent_self_iters=pi_cur)
        cols = []
        for j in range(3):
            u = torch.zeros(G, 3, device=DEV, dtype=DT); u[:, j] = 1.0
            cols.append(hvp(u.reshape(-1), probe_id=j)[: G * 3].reshape(G, 3))
        H = torch.stack(cols, dim=-1)
    else:
        static = m.batch_statics[0]
        fam = static.family_index_tensor.to(DEV)
        theta_b = theta.index_select(0, fam).contiguous()
        _l, sv = forward_solve(m.batch_statics, theta, rw)
        hvp = make_exact_hvp(m.batch_statics, theta_b, rw, sv, tangent_self_iters=pi_cur)
        cols = []
        for j in range(3):
            u_b = torch.zeros(G, 3, device=DEV, dtype=DT); u_b[:, j] = 1.0
            out_b = hvp(u_b.reshape(-1), probe_id=j)[: G * 3].reshape(G, 3)
            col = torch.zeros(G, 3, device=DEV, dtype=DT)
            col.index_add_(0, fam, out_b)
            cols.append(col)
        H = torch.stack(cols, dim=-1)
    return 0.5 * (H + H.transpose(1, 2))
```

- [ ] **Step 3: Replace the optimization loop's FD block**

Currently (lines ~201-208):

```python
        if it % HESS_EVERY == 0 or Hd is None or Hd.shape[0] != sub.shape[0]:
            H = torch.zeros(sub.shape[0], 3, 3, device=DEV, dtype=DT)
            for j in range(3):
                tp = sub.clone(); tp[:, j] += FD_EPS; _, gp = lg(m, tp)
                tm = sub.clone(); tm[:, j] -= FD_EPS; _, gm = lg(m, tm)
                H[:, :, j] = (gp - gm) / (2 * FD_EPS)
            H = 0.5 * (H + H.transpose(1, 2)); e, V = torch.linalg.eigh(H)
            Hd = V @ torch.diag_embed(e.clamp(min=MU)) @ V.transpose(1, 2)
```

Replace with:

```python
        if it % HESS_EVERY == 0 or Hd is None or Hd.shape[0] != sub.shape[0]:
            H = _analytic_hessian(m, sub, PI_CUR)
            e, V = torch.linalg.eigh(H)
            Hd = V @ torch.diag_embed(e.clamp(min=MU)) @ V.transpose(1, 2)
```

(`PI_CUR` is the existing module-level loop variable already in scope at this point in the script — confirmed via `rebatch_log.append(dict(pi=PI_CUR, ...))` and the `print(f"  [pi{PI_CUR}] stalled...")` line a few lines above this block, both already present in the file today.)

- [ ] **Step 4: Replace the final cert block's FD Hessian**

Currently (lines ~223-228):

```python
H = torch.zeros(F_all, 3, 3, device=DEV, dtype=DT)
for j in range(3):
    tp = theta.clone(); tp[:, j] += FD_EPS; _, gp = lg(mfull, tp)
    tm = theta.clone(); tm[:, j] -= FD_EPS; _, gm = lg(mfull, tm)
    H[:, :, j] = (gp - gm) / (2 * FD_EPS)
H = 0.5 * (H + H.transpose(1, 2)); lam_min = torch.linalg.eigvalsh(H)[:, 0]
```

Replace with:

```python
H = _analytic_hessian(mfull, theta, CERT_PI)
lam_min = torch.linalg.eigvalsh(H)[:, 0]
```

(`CERT_PI` is the existing module-level variable used a few lines above to build `mfull` — already in scope.)

- [ ] **Step 5: Remove the now-unused `FD_EPS` constant**

Currently:
```python
TOL = float(os.environ.get("TOL", "1e-3")); FD_EPS = 1e-2; MU = 1e-2; TRUST = 2.0
```
Change to:
```python
TOL = float(os.environ.get("TOL", "1e-3")); MU = 1e-2; TRUST = 2.0
```

- [ ] **Step 6: Manual smoke run at modest scale**

Run (uses the `archaea` dataset already wired via `run_cv.DATASETS`, same as the earlier session's benchmarks):

```bash
GPUREC_PREPROCESS_PATH=$(pwd)/crates/gpurec-preprocess/target/debug/libgpurec_preprocess.so \
GPUREC_ARCHAEA_ROOT=$(pwd)/../gpurec-data/benchmarks/large_dataset_capacity/datasets/alerax_archaea_davin2017 \
DATASET=archaea FAMILIES=200 \
.venv/bin/python -u experiments/sanderson_cv/bench_genewise_warm_rebatch.py
```

Expected: completes without error, prints a final convergence summary with `premature=0` (or a small number consistent with what FD gave historically — this script prints its own summary block, read it and confirm it looks sane: no NaN/inf, converged count close to family count).

- [ ] **Step 7: Commit**

```bash
git add experiments/sanderson_cv/bench_genewise_warm_rebatch.py
git commit -m "Apply analytic-HVP-default swap to bench_genewise_warm_rebatch.py"
```

---

### Task 3: Re-validate at archaea60 scale through the real merged code

**Files:** none (verification only — no code changes expected unless Step 1 or 2 surfaces a regression)

This re-runs the exact benchmark from the HVP-warm-start plan's Task 7, but through the actual
production `fit_genewise` (post Task 1's merge) instead of the scratchpad prototype, to catch
anything that differs between "prototype in isolation" and "merged into the real file alongside
its other logic" (rebatching, tiering, drop/defer).

- [ ] **Step 1: 200-family sanity check through the real `fit_genewise`**

```python
# scratch script, e.g. /tmp/.../verify_default_swap_200.py
import os, sys
sys.path.insert(0, os.path.join(os.getcwd(), "experiments", "sanderson_cv"))
from run_cv import DATASETS
from gpurec.fit.genewise_fit import fit_genewise, GENEWISE_REFERENCE

fam_paths = DATASETS["archaea"]["families"](200)
sp_tree = str(DATASETS["archaea"]["species_tree"])
res = fit_genewise(sp_tree, fam_paths, device="cuda", **{**GENEWISE_REFERENCE, "certify": True, "verbose": True})
print(f"wall={res['opt_seconds']:.1f}s n_steps={res['n_steps']} n_builds={res['n_builds']} "
      f"converged={res['converged']}/{res['n_families']} pg_max={res['pg_max']:.3e} "
      f"loss_bits={res['loss_bits']:.4f}")
```

Run with `GPUREC_PREPROCESS_PATH`/`GPUREC_ARCHAEA_ROOT` set as in prior benchmarks this session.
Expected: 200/200 converged, finite loss, roughly comparable wall-clock to the scratchpad
prototype's earlier-measured 22.8s (report the actual number, do not assume it matches exactly —
the merged file's surrounding logic is the same but re-measure).

- [ ] **Step 2: Full 5446-family (26-batch) archaea60 run**

Same script with `DATASETS["archaea"]["families"](None)` (all families). This takes several
minutes (the FD baseline was 353.9s, warm-started analytic prototype was 322.4s) — run in the
background and report the actual measured wall-clock, `converged` count, and `loss_bits` once
complete. Compare directly against the scratchpad prototype's numbers (322.4s, 5446/5446
converged, loss_bits=360167.917) — flag any meaningful discrepancy rather than assuming a match.

- [ ] **Step 3: Full test suite**

```bash
GPUREC_PREPROCESS_PATH=$(pwd)/crates/gpurec-preprocess/target/debug/libgpurec_preprocess.so \
.venv/bin/python -m pytest tests/ -m gpu -q --ignore=tests/data
```

Expected: no failures, same skip reasons as the last full-suite baseline this session (357
passed, 10 skipped, 0 failed) plus the one new test from Task 1.

- [ ] **Step 4: Report results**

Summarize: did the merged `fit_genewise` reproduce the scratchpad prototype's validated numbers
(within reasonable run-to-run variance)? Did the bench-script smoke run (Task 2, Step 6) look
correct? Did the full suite pass clean? This is the final go/no-go signal for the whole plan —
report actual measured numbers, not a restatement of the prototype's earlier numbers.
