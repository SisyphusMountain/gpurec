# Genewise joint analytic HVP — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A single analytic (forward-over-reverse) Hessian/HVP for genewise fits over the joint parameter `z = [θ(G×3); ω(G×S); α(S)]`, plus an arrowhead Newton solve wired into a genewise fit with a PD certificate.

**Architecture:** One joint HVP operator — seed `u=[u_θ;u_ω;u_α]` into one forward-over-reverse sweep; all blocks and cross-terms fall out (never assembled to compute `H·u`). `θ,α` enter the E/Pi fixed points (tangent Triton kernels, already present); `ω` is head-only (autograd double-backward over the NLL head). The Hessian is **arrowhead**: block-diagonal `(θ_g,ω_g)` per family + a global `α`. Newton = per-family Schur (Woodbury on the diag+low-rank `H_ωω`) → dense `S×S` α solve → back-substitute.

**Tech Stack:** Python, PyTorch (fp64/fp32), Triton kernels, pytest. GPU (CUDA) required for every gate.

**Design doc:** `gpurec/docs/optim/genewise_joint_hvp_plan.md` — read it before starting (esp. §4 "one sweep, not blocks" and §2 "what P0 actually was").

## Global Constraints

- **No clamps on values** — use compensated/log-space forms (`survival_from_E`, `safe_log2`, `logsumexp2`); the survival/receiver normalizers are cancellation-free (`b6faad9b`). Index/iteration/step-size clamps are fine.
- **HVP truncation must match the primal forward** — pass `tangent_self_iters == solver_options.pi_iters`, else the gate shows a truncation bias unrelated to correctness.
- **Gates run fp64 + converged solver:** `SolverOptions(e_max_iter=2000, e_tol=1e-10, pi_iters=128, neumann_terms=64, self_loop_solver="neumann", bicgstab_max_iter=500, bicgstab_tol=1e-10, bicgstab_breakdown_tol=1e-30, adjoint_pruning_threshold=1e-6, use_adjoint_pruning=True, pibar_side_threshold=0.0)`, `tangent_self_iters=128`. Acceptance `rel < 5e-4`, symmetry `rel_asym < 5e-3`.
- **Prior-agnostic** — build no prior on `ω`. Conditioning is the caller's (a diagonal on `H_ωω`, or the Newton damping `μ`).
- **Single collated batch** — every gate builds a model whose families fit in one batch (`assert len(m.batch_statics) == 1`). Multi-batch is Task 9 (uniform outer loop), not the default path.
- **Per-family reductions are the risk** — P0's bugs were per-family/per-species reductions, not derivations. Every new gate exists to catch that class; gate on `static.genewise` so specieswise/global stay bit-for-bit.

---

## File structure

- `tests/test_genewise_hvp.py` — all gates + the shared `build_genewise_model` / `make_joint_value_and_grad` helpers. Exists (θ gate green).
- `gpurec/optim/genewise_curvature.py` *(new)* — genewise-facing API: `genewise_hessian_blocks(...)`, `newton_step_joint(...)`. Owns broadcast-probe assembly, the diag+low-rank `H_ωω`, and the Schur/Woodbury arrowhead solve.
- `gpurec/api/model.py` — per-family origination `[G,S]` behind a `per_family_origination` flag (Task 2).
- `gpurec/api/_execution.py` — per-family `ω_g` selection into the aggregation (Task 3), mirroring `theta_for_static`.
- `gpurec/optim/hvp_exact.py` — make `_head_seed_tangents` + the `hvp()` ω tail per-family `[G,S]` (Task 4).
- `gpurec/optim/genewise_fit.py` — swap the FD certificate Hessian for the analytic arrowhead block (Task 7).

---

### Task 1: `genewise_hessian_blocks` (θ-only) + fp32 golden — close out P0

**Files:**
- Create: `gpurec/optim/genewise_curvature.py`
- Test: `tests/test_genewise_hvp.py` (add `test_theta_blocks_fp32_vs_fp64`)

**Interfaces:**
- Consumes: `forward_solve`, `make_exact_hvp` (genewise θ path, green), `build_genewise_model` (in the test file).
- Produces: `genewise_hessian_blocks(static, theta, receiver_weights, sv, *, omega=None, active=("theta",)) -> {"H_tt": Tensor[G,3,3]}`. Later tasks extend `active` and the dict.

- [ ] **Step 1: Write the failing golden test** (append to `tests/test_genewise_hvp.py`)

```python
@pytest.mark.gpu
def test_theta_blocks_fp32_vs_fp64():
    from gpurec.optim.genewise_curvature import genewise_hessian_blocks
    def blocks(dtype):
        m = build_genewise_model(dtype=dtype); st = m.batch_statics[0]
        G = len(m.families); S = int(m.species_helpers["S"])
        th = torch.full((G, 3), math.log2(0.1), device="cuda", dtype=dtype)
        rw = torch.zeros(S, device="cuda", dtype=dtype)
        _l, sv = forward_solve([st], th, rw)
        return genewise_hessian_blocks(st, th, rw, sv)["H_tt"]
    H32, H64 = blocks(torch.float32).double(), blocks(torch.float64)
    assert torch.isfinite(H32).all()
    torch.testing.assert_close(H32, H64, rtol=1e-3, atol=1e-2)
    assert float((H64 - H64.transpose(1, 2)).abs().max()) < 1e-6  # each 3x3 block symmetric
```

- [ ] **Step 2: Run — expect FAIL** (ImportError)

Run: `.venv/bin/python -m pytest tests/test_genewise_hvp.py::test_theta_blocks_fp32_vs_fp64 -q`
Expected: FAIL, `ModuleNotFoundError: gpurec.optim.genewise_curvature`.

- [ ] **Step 3: Implement `genewise_hessian_blocks` (θ-only)** — fire the 3 broadcast probes, assemble `[G,3,3]`, symmetrize.

```python
import torch
from gpurec.optim.hvp_exact import make_exact_hvp


def genewise_hessian_blocks(static, theta, receiver_weights, sv, *, omega=None, active=("theta",)):
    """Structured curvature for the genewise arrowhead Newton solve. P0: theta-only.

    Returns per-family blocks; omega/alpha entries are added in Tasks 4-5. The HVP itself
    is one joint operator (make_exact_hvp) -- these blocks are materialized ONLY for the
    Newton solve, never to compute H@u.
    """
    assert tuple(active) == ("theta",), "P0: theta-only; omega/alpha added in Tasks 4-5"
    G = int(theta.shape[0])
    tsi = int(static.solver_options.pi_iters)
    hvp = make_exact_hvp([static], theta, receiver_weights, sv, tangent_self_iters=tsi)
    cols = []
    for j in range(3):  # broadcast e_j across all families -> column j of every 3x3 block at once
        u = torch.zeros(G, 3, device=theta.device, dtype=theta.dtype); u[:, j] = 1.0
        cols.append(hvp(u.reshape(-1))[: G * 3].reshape(G, 3))
    H = torch.stack(cols, dim=-1)  # [G,3,3], H[:,:,j] = col j
    return {"H_tt": 0.5 * (H + H.transpose(1, 2))}
```

- [ ] **Step 4: Run — expect PASS**

Run: `.venv/bin/python -m pytest tests/test_genewise_hvp.py::test_theta_blocks_fp32_vs_fp64 -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add gpurec/optim/genewise_curvature.py tests/test_genewise_hvp.py
git commit -m "feat(optim): genewise_hessian_blocks (theta-only, 3 broadcast probes) + fp32 golden"
```

**► P0 fully done: reusable analytic genewise θ certificate. Tasks 2–8 add ω and the α arrowhead.**

---

### Task 2: Per-family origination parameter `[G,S]`

**Files:**
- Modify: `gpurec/api/model.py` (origination parameter at `:70-79`; today global `[S]`).
- Test: `tests/test_genewise_hvp.py` (add `test_per_family_origination_shape`; extend `build_genewise_model`).

**Interfaces:**
- Produces: `GeneReconModel(..., mode="genewise", per_family_origination=True)` with `model.origination_weights` of shape `[G, S]` (default flag off ⇒ `[S]`, all existing behaviour unchanged).

- [ ] **Step 1: Read FIRST** `model.py:33-95` (how `theta_shape` + `self.theta` + `self.origination_weights` are built) so origination mirrors the θ per-family pattern exactly.

- [ ] **Step 2: Extend `build_genewise_model` and write the failing test**

```python
def build_genewise_model(n_fam=2, dtype=torch.float64, device="cuda", per_family_origination=False):
    so = SolverOptions(**_SO); so.validate()
    m = GeneReconModel(f"{_D}/sp.nwk", [f"{_D}/g.nwk"] * n_fam, mode="genewise",
                       device=device, dtype=dtype, solver_options=so,
                       per_family_origination=per_family_origination)
    assert len(m.batch_statics) == 1, f"expected 1 batch, got {len(m.batch_statics)}"
    return m

@pytest.mark.gpu
def test_per_family_origination_shape():
    m = build_genewise_model(per_family_origination=True)
    G, S = len(m.families), int(m.species_helpers["S"])
    assert tuple(m.origination_weights.shape) == (G, S)
    m0 = build_genewise_model(per_family_origination=False)   # default unchanged
    assert tuple(m0.origination_weights.shape) == (S,)
```

- [ ] **Step 3: Run — expect FAIL** (`TypeError: unexpected keyword 'per_family_origination'`)

Run: `.venv/bin/python -m pytest tests/test_genewise_hvp.py::test_per_family_origination_shape -q`

- [ ] **Step 4: Implement** — add a `per_family_origination: bool = False` constructor arg; when true (genewise only), allocate `origination_weights` as `torch.full((G, S), 0.0, ...)`; else keep the `[S]` default. Store `self.per_family_origination`. Guard: raise `ValueError` if `per_family_origination and not genewise`.

- [ ] **Step 5: Run — expect PASS.** **Step 6: Commit** (`feat(api): per-family origination [G,S] for genewise (flagged, default off)`).

---

### Task 3: Per-family weighted NLL + origination gradient

**Files:**
- Modify: `gpurec/api/_execution.py` (`theta_for_static` pattern at `:18-19`; `_origination_log_probs` at `:22-28`; the origination-grad call at `:106-108`).
- Test: `tests/test_genewise_hvp.py` (add `test_origination_grad_matches_fd`).

**Interfaces:**
- Consumes: per-family `origination_weights[G,S]` (Task 2); `origination_log_probs_from_weights` (`extract_parameters`); `nll_vector_from_root_rows(..., origination_log_probs, origination_probs)` and `origination_grad_from_root_rows(root_rows, E, origination_weights)` (`solver.py`).
- Produces: a genewise loss+grad that feeds each family its own `ω_g` row and returns a per-family origination gradient `[G,S]`; `make_joint_value_and_grad(static, theta_shape, S, G)` test helper (below) that packs `[g_θ(3G); g_ω(GS); g_α(S)]` into one flat vector for FD.

- [ ] **Step 1: Read FIRST** `_execution.py:22-111` and `solver.py` `nll_vector_from_root_rows` / `origination_grad_from_root_rows`. Note: `_origination_log_probs` currently assumes a **global** `[S]` weight; it must select `ω_g` per family (index by `static.family_index_tensor`, exactly like `theta_for_static`) when the weight is `[G,S]`.

- [ ] **Step 2: Write the failing test** — analytic `∂NLL/∂ω` vs fp64 central-difference of the summed loss.

```python
@pytest.mark.gpu
def test_origination_grad_matches_fd():
    from gpurec.api._execution import evaluate_static_loss_grad
    m = build_genewise_model(per_family_origination=True); st = m.batch_statics[0]
    G, S = len(m.families), int(m.species_helpers["S"])
    th = torch.full((G, 3), math.log2(0.1), device="cuda", dtype=torch.float64)
    rw = torch.zeros(S, device="cuda", dtype=torch.float64)
    om = torch.randn(G, S, device="cuda", dtype=torch.float64) * 0.1
    def loss_only(omega):
        l, *_ = evaluate_static_loss_grad(st, th, rw, omega, need_origination_grad=False)
        return float(l)
    _l, _gt, _gr, g_om = evaluate_static_loss_grad(st, th, rw, om, need_origination_grad=True)
    assert tuple(g_om.shape) == (G, S)
    eps = 1e-5; g, k = 0, S // 3          # probe one (family, species) entry
    d = torch.zeros(G, S, device="cuda", dtype=torch.float64); d[g, k] = eps
    fd = (loss_only(om + d) - loss_only(om - d)) / (2 * eps)
    rel = abs(float(g_om[g, k]) - fd) / max(abs(fd), 1e-30)
    assert rel < 1e-3, f"origination grad rel={rel:.2e}"
```

- [ ] **Step 3: Run — expect FAIL** (per-family `ω` not selected; `g_om` shape or value wrong).

- [ ] **Step 4: Implement** — in `_origination_log_probs` (and its callers), when `origination_weights.ndim == 2`, select the per-family row via `origination_weights.index_select(0, static.family_index_tensor)` before `origination_log_probs_from_weights`; make `origination_grad_from_root_rows` return the `[G,S]` per-family gradient (today it collapses to `[S]` for the global case — keep both paths, gated on ndim). Feed each family its own `ω_g` into `nll_vector_from_root_rows`.

- [ ] **Step 5: Run — expect PASS.** **Step 6: Commit** (`feat: per-family weighted origination NLL + [G,S] gradient`).

---

### Task 4: ω per-family in the HVP head → joint θ+ω gate (P2)

**Files:**
- Modify: `gpurec/optim/hvp_exact.py` — `_head_seed_tangents` (`:81-104`) and the `hvp()` ω tail (`:286-289`, `:537-540`), to carry `ω`/`u_ω` as per-family `[G,S]` instead of global `[S]`.
- Test: `tests/test_genewise_hvp.py` (add `make_joint_value_and_grad` + `test_joint_theta_omega_hvp_matches_fd`).

**Interfaces:**
- Consumes: `make_exact_hvp([static], theta, rw, sv, ..., origination_weights=omega[G,S])`; `evaluate_static_loss_grad` (Task 3) for the FD reference.
- Produces: `hvp(u_vec)` accepts `u = [u_θ(3G); u_α(S); u_ω(GS)]` and returns `[out_θ(3G); out_α(S); out_ω(GS)]`; `Hv_omega` is `[G,S]`.

- [ ] **Step 1: Read FIRST** `_head_seed_tangents` (`hvp_exact.py:81-104`). It already double-backwards the NLL head `⟨∇NLL, [t_root; dE; u_ω]⟩` and returns `(ds_root, ds_E, Hv_om)`. Identify where `omega`/`u_omega` are treated as global `[S]` (the `om`/`u_omega` shapes and the `.sum()` that scalarises the per-family NLL). In `hvp()`, find the ω tail slice (`u_omega = u_vec[theta_numel+S:theta_numel+2*S]`) and the `has_omega` return (`:537-540`).

- [ ] **Step 2: Write the failing joint gate** (add helper + test)

```python
def make_joint_value_and_grad(static, theta_shape, S, G):
    """(loss, flat-grad) over z=[theta(3G); omega(GS); alpha(S)] for fp64 FD. Packs the three
    grads from evaluate_static_loss_grad in the SAME order the joint hvp returns them."""
    from gpurec.api._execution import evaluate_static_loss_grad
    nt = int(torch.tensor(theta_shape).prod())
    def vg(x, warm=None):
        th = x[:nt].reshape(theta_shape)
        om = x[nt:nt + G * S].reshape(G, S)
        al = x[nt + G * S:nt + G * S + S]
        l, g_th, g_al, g_om = evaluate_static_loss_grad(static, th, al, om, need_origination_grad=True)
        g = torch.cat([g_th.reshape(-1), g_om.reshape(-1), g_al.reshape(-1)]).double()
        return float(l), g, None, None
    return vg

@pytest.mark.gpu
def test_joint_theta_omega_hvp_matches_fd():
    from gpurec.optim.newton_cg import _fd_hessian_hvp
    m = build_genewise_model(per_family_origination=True); st = m.batch_statics[0]
    G, S = len(m.families), int(m.species_helpers["S"])
    th = torch.full((G, 3), math.log2(0.1), device="cuda", dtype=torch.float64)
    rw = torch.zeros(S, device="cuda", dtype=torch.float64)
    om = torch.zeros(G, S, device="cuda", dtype=torch.float64)
    _l, sv = forward_solve([st], th, rw)
    hvp = make_exact_hvp([st], th, rw, sv, tangent_self_iters=128, origination_weights=om)
    x0 = torch.cat([th.reshape(-1), om.reshape(-1), rw.reshape(-1)])
    fd = _fd_hessian_hvp(make_joint_value_and_grad(st, (G, 3), S, G), x0, None, eps=1e-5)
    for name, u in [("theta_e0", _dir_theta(G, S, 0)), ("omega_k", _dir_omega(G, S, S // 3))]:
        Ha, Hf = hvp(u).double(), fd(u).double()
        rel = float((Ha - Hf).abs().max()) / max(float(Hf.abs().max()), 1e-30)
        assert torch.isfinite(Ha).all() and rel < 5e-4, f"{name}: rel={rel:.2e}"

def _dir_theta(G, S, j):
    u = torch.zeros(3 * G + G * S + S, device="cuda", dtype=torch.float64)
    u.view(-1)[j:3 * G:3] = 1.0  # broadcast e_j across families' theta block
    return u
def _dir_omega(G, S, k):
    u = torch.zeros(3 * G + G * S + S, device="cuda", dtype=torch.float64)
    u[3 * G:3 * G + G * S].reshape(G, S)[:, k] = 1.0  # broadcast omega e_k across families
    return u
```

- [ ] **Step 3: Run — expect FAIL** (ω tail summed to `[S]`: wrong shape / `rel` large on the omega direction).

Run: `.venv/bin/python -m pytest tests/test_genewise_hvp.py::test_joint_theta_omega_hvp_matches_fd -q`

- [ ] **Step 4: Implement** — thread `origination_weights[G,S]` into `_head_seed_tangents` as `om` (per family); the head NLL is already `nll_vector_from_root_rows` (per family), so the double-backward w.r.t. per-family `om` yields `Hv_om` `[G,S]` without summing over families. In `hvp()`, adopt the **canonical joint layout `[θ(3G); ω(G·S); α(S)]`** — ω BEFORE α (the current code emits `[θ;α;ω]`; reorder it). Tail detection by `n_tail = u.numel() - 3G`: `0` ⇒ θ-only `[θ]`; `== S` ⇒ `[θ;α]` (the existing recv path, `u_ω=0`, output `[θ;α]` — bit-for-bit unchanged, `_verify_hvp_recv` depends on it); `== G*S + S` ⇒ full, parse `u_omega = tail[:G*S].reshape(G,S)`, `u_alpha = tail[G*S:]`, return `torch.cat([out_theta, Hv_omega.reshape(-1), out_col])`. Gate all per-family ω shape changes on `static.genewise` (specieswise/global keep the global-`[S]` ω path).

- [ ] **Step 5: Run the joint θ+ω gate — expect PASS.**

- [ ] **Step 6: Regression** — specieswise + receiver HVP unaffected.

Run: `.venv/bin/python -m gpurec.optim._verify_hvp 8 && .venv/bin/python -m gpurec.optim._verify_hvp_recv`
Expected: `[hvp gate] ALL PASS` and `[recv-hvp S8 gate] ... OVERALL=True`.

- [ ] **Step 7: Commit** (`feat(hvp): per-family omega head Hessian + theta-omega coupling ([G,S])`).

---

### Task 5: Joint θ+ω+α gate — verify/fix the α path under genewise θ (P3)

**Files:**
- Modify (only if the gate fails): `gpurec/optim/hvp_exact.py` (the genewise α-cotangent reductions, same class as P0), `gpurec/core/kernels/wave_so.py` / `dts_so.py` if a genewise per-family α reduction surfaces.
- Test: `tests/test_genewise_hvp.py` (add `test_joint_theta_omega_alpha_hvp_matches_fd`).

**Interfaces:**
- Consumes: the joint `hvp` (Task 4) with a non-uniform `α`; `make_joint_value_and_grad` (Task 3/4).
- Produces: the full joint `H·u` validated for genewise across θ, ω, α, and mixed directions + symmetry.

- [ ] **Step 1: Read FIRST** the α path in `hvp()` — `use_receiver_weights`, `u_alpha`, `dcol`, `out_col`, and `d_gcol` accumulation. The α HVP is validated only with **specieswise** θ (`_verify_hvp_recv`); under genewise θ the per-family cotangent reductions (`d_cot_col`, `grad_col`) have never been exercised. Expect a P0-style per-family reduction bug and locate it before patching.

- [ ] **Step 2: Write the failing full gate** (non-uniform α; θ, ω, α, mixed directions + symmetry)

```python
@pytest.mark.gpu
def test_joint_theta_omega_alpha_hvp_matches_fd():
    from gpurec.optim.newton_cg import _fd_hessian_hvp
    m = build_genewise_model(per_family_origination=True); st = m.batch_statics[0]
    G, S = len(m.families), int(m.species_helpers["S"])
    th = torch.full((G, 3), math.log2(0.1), device="cuda", dtype=torch.float64)
    al = torch.randn(S, device="cuda", dtype=torch.float64) * 0.1   # NON-uniform alpha
    om = torch.zeros(G, S, device="cuda", dtype=torch.float64)
    _l, sv = forward_solve([st], th, al)
    hvp = make_exact_hvp([st], th, al, sv, tangent_self_iters=128, origination_weights=om)
    x0 = torch.cat([th.reshape(-1), om.reshape(-1), al.reshape(-1)])
    fd = _fd_hessian_hvp(make_joint_value_and_grad(st, (G, 3), S, G), x0, None, eps=1e-5)
    P = 3 * G + G * S + S
    dirs = {"theta": _dir_theta(G, S, 1), "omega": _dir_omega(G, S, S // 4)}
    da = torch.zeros(P, device="cuda", dtype=torch.float64); da[3 * G + G * S:] = torch.randn(S, device="cuda", dtype=torch.float64)
    dirs["alpha"] = da
    dirs["mixed"] = _dir_theta(G, S, 0) + _dir_omega(G, S, S // 2) + da
    Hs = {}
    for name, u in dirs.items():
        Ha, Hf = hvp(u).double(), fd(u).double()
        rel = float((Ha - Hf).abs().max()) / max(float(Hf.abs().max()), 1e-30)
        assert torch.isfinite(Ha).all() and rel < 5e-4, f"{name}: rel={rel:.2e}"
        Hs[name] = Ha
    # symmetry across parameter groups: u^T H w == w^T H u
    u, w = dirs["theta"], dirs["alpha"]
    sym = abs(float(torch.dot(u, Hs["alpha"])) - float(torch.dot(w, Hs["theta"])))
    scale = max((abs(float(torch.dot(u, Hs["theta"]))) * abs(float(torch.dot(w, Hs["alpha"])))) ** 0.5, 1e-30)
    assert sym / scale < 5e-3, f"asym={sym/scale:.2e}"
```

- [ ] **Step 3: Run — expect FAIL or PASS.** If PASS, the α path is already genewise-correct — skip to Step 5. If FAIL, diagnose the per-family α reduction (Step 4).

Run: `.venv/bin/python -m pytest tests/test_genewise_hvp.py::test_joint_theta_omega_alpha_hvp_matches_fd -q`

- [ ] **Step 4: Fix (only if red)** — apply the P0 recipe: bisect the forward-over-reverse chain (forward tangent → e-step SO → head contraction → wave/DTS SO) against fp64 FD; the defect will be a genewise per-family/per-species reduction in a `grad_col`/`d_cot_col` path. Gate the fix on `static.genewise`; re-run `_verify_hvp_recv` to prove specieswise α is bit-for-bit unchanged.

- [ ] **Step 5: Commit** (`test(hvp): genewise joint theta+omega+alpha gate` — plus `fix(...)` if Step 4 fired).

**► P3 done: the full joint genewise `H·u` is validated. Tasks 6–7 add the Newton solve + fit wiring.**

---

### Task 6: `H_ωω` diag+low-rank + arrowhead `newton_step_joint`

**Files:**
- Modify: `gpurec/optim/genewise_curvature.py` (extend `genewise_hessian_blocks` to `active=("theta","omega","alpha")`; add `newton_step_joint`).
- Test: `tests/test_genewise_hvp.py` (add `test_newton_step_matches_dense`).

**Interfaces:**
- Consumes: the joint `hvp` (Tasks 4-5); the closed-form `H_ωω` head Hessian from `(root_rows_g, E_g)` (design §6).
- Produces: `genewise_hessian_blocks(..., active=("theta","omega","alpha")) -> {"H_tt":[G,3,3], "H_to":[G,3,S], "H_oo_diag":[G,S], "H_oo_lr":[G,S,r], "H_aa":[S,S], "H_za":[G,3+S,S]}` (dense arrow; matrix-free is P4/optional) and `newton_step_joint(blocks, g_theta[G,3], g_omega[G,S], g_alpha[S], mu) -> (dtheta[G,3], domega[G,S], dalpha[S])`. `newton_step_joint` takes the three grads as separate tensors, so it is flat-order-agnostic; only the gate vectors follow `[θ;ω;α]`.

- [ ] **Step 1: Write the failing test** — on a *synthetic* SPD arrowhead (random `H_tt`, `H_to`, diag+low-rank `H_oo`, dense `H_aa`, couplings), `newton_step_joint` reproduces the dense assembled solve (`torch.linalg.solve` on the full `(3G+GS+S)` matrix) to `rtol 1e-5`.

```python
@pytest.mark.gpu
def test_newton_step_matches_dense():
    from gpurec.optim.genewise_curvature import newton_step_joint, _assemble_dense_arrowhead
    torch.manual_seed(0); G, S, r, mu = 2, 5, 2, 1e-2
    dev, dt = "cuda", torch.float64
    def spd(n):
        A = torch.randn(n, n, device=dev, dtype=dt); return A @ A.T + n * torch.eye(n, device=dev, dtype=dt)
    blocks = dict(
        H_tt=torch.stack([spd(3) for _ in range(G)]),
        H_to=torch.randn(G, 3, S, device=dev, dtype=dt) * 0.1,
        H_oo_diag=torch.rand(G, S, device=dev, dtype=dt) + 1.0,
        H_oo_lr=torch.randn(G, S, r, device=dev, dtype=dt) * 0.1,
        H_aa=spd(S), H_za=torch.randn(G, 3 + S, S, device=dev, dtype=dt) * 0.1)
    g_th = torch.randn(G, 3, device=dev, dtype=dt); g_om = torch.randn(G, S, device=dev, dtype=dt)
    g_al = torch.randn(S, device=dev, dtype=dt)
    dth, dom, dal = newton_step_joint(blocks, g_th, g_om, g_al, mu)
    Hd, gd = _assemble_dense_arrowhead(blocks, mu)          # dense [(3G+GS+S)]^2 + stacked g
    ref = torch.linalg.solve(Hd, -gd)
    got = torch.cat([dth.reshape(-1), dom.reshape(-1), dal.reshape(-1)])
    torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-8)
```

- [ ] **Step 2: Run — expect FAIL** (functions undefined).

- [ ] **Step 3: Implement** `_assemble_dense_arrowhead` (test-only oracle: build `H_oo = diag+lr@lrᵀ`, assemble the full matrix + `μI`) and `newton_step_joint`: per family form `B_g=[[H_tt+μI, H_to],[H_toᵀ, H_oo+μI]]`, invert `H_oo` via **Woodbury** (`(D+UUᵀ)⁻¹`), Schur-eliminate the core to a dense `S×S` α system `(H_aa+μI − Σ_g H_az,g B_g⁻¹ H_za,g)`, solve, back-substitute per family. (`H_aa_op`/`H_za_op` may be dense here; matrix-free is P4/optional.)

- [ ] **Step 4: Run — expect PASS.**

- [ ] **Step 5: Extend `genewise_hessian_blocks`** to emit the real `H_tt/H_to/H_oo_diag/H_oo_lr` (θ,θω via the broadcast probes; `H_oo` closed-form per design §6) and the α couplings, and add `test_blocks_match_hvp`: for random `u`, `blocks`-assembled `H@u` equals the matrix-free `hvp(u)` to `rel 5e-4` on the real 2-family model. **Commit** (`feat(optim): arrowhead newton_step (Schur+Woodbury) + full genewise blocks`).

---

### Task 7: Wire the analytic joint curvature into the genewise fit + PD certificate

**Files:**
- Modify: `gpurec/optim/genewise_fit.py` (certificate Hessian; the inner Newton step).
- Test: `tests/test_genewise_hvp.py` (add `test_genewise_joint_fit_certifies`).

**Interfaces:**
- Consumes: `newton_step_joint`, `genewise_hessian_blocks` (Task 6); the matrix-free `hvp` for a Lanczos `lam_min` certificate.
- Produces: `fit_genewise(..., per_family_origination=True, optimize=("theta","omega","alpha"), certify=True)` completing with finite curvature and an `interior_pd` flag.

- [ ] **Step 1: Read FIRST** `gpurec/optim/genewise_fit.py` — the current FD certificate Hessian and inner step; and `lanczos_extremes` (`newton_cg.py`) for the matrix-free `lam_min`.

- [ ] **Step 2: Write the failing e2e test** — a few joint Newton steps from `θ=log2(0.1), ω=0, α=0` on the 2-family model reduce the joint gradient norm and report a finite `lam_min` (analytic Lanczos on `hvp`) matching an fp64 FD `lam_min` to `rtol 1e-2`.

```python
@pytest.mark.gpu
def test_genewise_joint_fit_certifies():
    from gpurec.optim.genewise_fit import fit_genewise
    m = build_genewise_model(per_family_origination=True)
    res = fit_genewise(m, optimize=("theta", "omega", "alpha"), certify=True,
                       max_newton=3, mu=1e-2)
    assert math.isfinite(res.lam_min) and res.grad_norm < res.grad_norm0
```

- [ ] **Step 3: Run — expect FAIL.**

- [ ] **Step 4: Implement** the joint Newton loop in `fit_genewise`: assemble blocks → `newton_step_joint` (caller adds ω prior/`μ` here) → line search on the joint value-and-grad → Lanczos `lam_min` on the matrix-free `hvp` for `interior_pd`. Prior-agnostic: expose `omega_ridge=0.0` that the caller sets (adds to `H_oo_diag`).

- [ ] **Step 5: Run — expect PASS.** **Step 6: Commit** (`feat(optim): analytic joint genewise Newton fit + PD certificate`).

---

### Task 8: Newton cross-check — structured solve vs matrix-free Newton-CG

**Files:**
- Test: `tests/test_genewise_hvp.py` (add `test_newton_step_matches_cg`).

**Interfaces:**
- Consumes: `newton_step_joint` (Task 6), the matrix-free joint `hvp` (Tasks 4-5).

- [ ] **Step 1: Write the test** — on the real 2-family model at a damped point, the structured `newton_step_joint(blocks, g, μ)` solves `(H+μI)δ = −g` to the same `δ` a matrix-free CG on `v ↦ hvp(v)+μv` produces, `rel 1e-3`. This closes the loop: the assembled blocks and the one-sweep operator agree, and the Schur/Woodbury solve is correct.

```python
@pytest.mark.gpu
def test_newton_step_matches_cg():
    from gpurec.optim.genewise_curvature import newton_step_joint, genewise_hessian_blocks
    from gpurec.optim.newton_cg import _cg_solve   # matrix-free CG (read its signature first)
    m = build_genewise_model(per_family_origination=True); st = m.batch_statics[0]
    G, S = len(m.families), int(m.species_helpers["S"]); mu = 1e-2
    th = torch.full((G, 3), math.log2(0.1), device="cuda", dtype=torch.float64)
    al = torch.zeros(S, device="cuda", dtype=torch.float64); om = torch.zeros(G, S, device="cuda", dtype=torch.float64)
    _l, sv = forward_solve([st], th, al)
    hvp = make_exact_hvp([st], th, al, sv, tangent_self_iters=128, origination_weights=om)
    g = torch.randn(3 * G + G * S + S, device="cuda", dtype=torch.float64)
    blocks = genewise_hessian_blocks(st, th, al, sv, omega=om, active=("theta", "omega", "alpha"))
    g_th, g_om, g_al = g[:3*G].reshape(G,3), g[3*G:3*G+G*S].reshape(G,S), g[3*G+G*S:]
    dth, dom, dal = newton_step_joint(blocks, g_th, g_om, g_al, mu)
    d_struct = torch.cat([dth.reshape(-1), dom.reshape(-1), dal.reshape(-1)])
    d_cg = _cg_solve(lambda v: hvp(v) + mu * v, -g, tol=1e-8, max_iter=2000)
    rel = float((d_struct - d_cg).abs().max()) / max(float(d_cg.abs().max()), 1e-30)
    assert rel < 1e-3, f"structured vs CG rel={rel:.2e}"
```

- [ ] **Step 2: Run — expect PASS** (if Task 6 correct). **Step 3: Commit** (`test(optim): arrowhead Newton structured-vs-CG cross-check`).

---

### Task 9 (follow-on): Multi-batch α accumulation

**Files:**
- Modify: `gpurec/optim/hvp_exact.py` (`_single_static` → outer loop) and `genewise_curvature.py` (accumulate the α Schur pieces across batches).
- Test: `tests/test_genewise_hvp.py` (build a model with `family_chunk_size` small enough to force ≥2 batches; the joint gate must still pass).

**Interfaces:**
- Produces: joint `hvp`/blocks over `len(batch_statics) > 1`; θ/ω concatenated across batches, α (`H_αα`, α-row/col, `Σ_g H_αz B⁻¹ H_zα`) summed across batches before the dense `S×S` solve.

- [ ] **Step 1: Write the failing multi-batch gate** — same as Task 5 but `build_genewise_model(n_fam=4, family_chunk_size=2)` (assert `len(m.batch_statics) == 2`); joint `hvp` vs FD `rel < 5e-4`.
- [ ] **Step 2: Run — expect FAIL** (`NotImplementedError` from `_single_static`).
- [ ] **Step 3: Implement** the outer batch loop: per-batch tangent sweeps for θ/ω (block-diagonal, concatenate), accumulate α contributions across batches. Keep single-batch bit-for-bit.
- [ ] **Step 4: Run — expect PASS.** **Step 5: Commit** (`feat(hvp): multi-batch global-alpha accumulation (arrowhead)`).

---

## Self-review notes

- **Spec coverage:** design §3 arrowhead → Tasks 4-6; §4 one-sweep → Tasks 4-5 (HVP), never assembled for `H·u`; §5 per-param → θ(Task 1, done P0)/ω(Tasks 2-4)/α(Task 5); §6 diag+low-rank `H_ωω` → Task 6; §7 arrowhead solve → Task 6; §8 batching → single-batch default + Task 9; §9 prior-agnostic → `omega_ridge` hook (Task 7), no prior built; §10 verification → gates in Tasks 1,3,4,5,6,8; §11 phasing → Tasks map P0..P5.
- **Investigation-gated tasks:** Tasks 4 and 5 modify adjoint/kernel internals whose exact diff depends on reading the cited functions — each has a "read FIRST" step and a fully-specified gate (the tests are the contract). Task 5 may be a no-op fix (α already genewise-correct) — its gate decides.
- **Type consistency:** `genewise_hessian_blocks` returns a dict grown across Tasks 1→6 (`H_tt`, then `+H_to/H_oo_diag/H_oo_lr/H_aa/H_za`); `newton_step_joint(blocks, g_theta[G,3], g_omega[G,S], g_alpha[S], mu)` consumes exactly that dict. The joint direction/return order is `[θ(3G); ω(GS); α(S)]` everywhere (`make_joint_value_and_grad`, the `hvp` return, `_dir_*` helpers, `newton_step_joint`).
- **Prior-agnostic:** no task adds a prior; Task 7 exposes `omega_ridge` (caller-set, adds to `H_oo_diag`) purely as the conditioning hook.
```
