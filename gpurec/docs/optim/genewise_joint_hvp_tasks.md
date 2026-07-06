# Genewise analytic HVP — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** An analytic (forward-over-reverse) Hessian/HVP for genewise fits over per-family DTL `θ[G,3]` and per-family origination `ω[G,S]`, block-diagonal per family, verified against fp64 finite differences.

**Architecture:** Reuse the existing `C×S` tangent/SO Triton kernels unchanged; add a genewise **θ seed/projection** (broadcast a per-family 3-vector to per-species rate tangents on the way in; sum per-species rate-cotangents back to `[G,3]` on the way out). Origination is a *head* parameter (never in the fixed points), so `H_ωω` is a closed-form diagonal+low-rank head Hessian and `H_θω` falls out of the θ sweeps. Per-family Newton via Schur + Woodbury.

**Tech Stack:** Python, PyTorch (fp64/fp32), Triton kernels, pytest. GPU (CUDA) required for the gates.

**Design doc:** `gpurec/docs/optim/genewise_joint_hvp_plan.md` (read it before starting).

## Global Constraints

- **No clamps on values** — use compensated/log-space forms (`survival_from_E`, `safe_log2`, `logsumexp2`); the survival/receiver normalizers are already cancellation-free (commit `b6faad9b`). Index/iteration/step-size clamps are fine.
- **HVP truncation must match the primal forward** — pass `tangent_self_iters == solver_options.pi_iters` (`hvp_exact.py:122-143`), else the gate shows a truncation bias unrelated to correctness.
- **Gates run fp64 + converged solver** (`pi_iters=128, neumann_terms=64, e_tol=1e-10, tangent_self_iters=128`) so neither analytic nor FD side is truncation-limited; acceptance `rel < 5e-4` and symmetry `rel_asym < 5e-3`.
- **Block-diagonal per family** — no cross-family coupling, no multi-batch accumulation. θ and ω are both per-family.
- Follow existing patterns in `gpurec/optim/hvp_exact.py`, `value_and_grad.py`, `_verify_hvp.py`. Reuse `make_exact_hvp`/`forward_solve` rather than reimplementing the sweep.

---

## File structure

- `gpurec/optim/hvp_exact.py` — add the genewise θ seed/projection to the parameter-side of `make_exact_hvp` (or a `genewise=True` branch). Core sweep unchanged.
- `gpurec/optim/genewise_curvature.py` *(new)* — genewise-facing API: `genewise_hessian_blocks(...)`, `newton_step_genewise(...)`; owns the broadcast-probe assembly, the ω head Hessian, and the Schur/Woodbury solve.
- `gpurec/api/model.py` — per-family origination parameter `[G,S]` (P1).
- `gpurec/core/inference/solver.py` — per-family ω indexing in the weighted NLL (P1); already has the weighted branch.
- `gpurec/optim/genewise_fit.py` — swap the FD certificate Hessian (`:264-270`) for the analytic block (P3).
- `tests/test_genewise_hvp.py` *(new)* — the gate (analytic vs fp64 FD + symmetry) and golden tests.

---

### Task 1: Genewise HVP gate (red) — θ-only

**Files:**
- Create: `tests/test_genewise_hvp.py`

**Interfaces:**
- Consumes: `forward_solve`, `make_exact_hvp` (`gpurec.optim`), `make_value_and_grad`, `_fd_hessian_hvp`.
- Produces: `build_genewise_model(n_fam, dtype, device)` helper reused by later tasks; the `theta_gate` test.

- [ ] **Step 1: Write the failing gate test** (complete code)

```python
import math, pytest, torch
from gpurec import GeneReconModel, SolverOptions
from gpurec.optim.hvp_exact import build_point_cache, make_exact_hvp
from gpurec.optim.newton_cg import _fd_hessian_hvp
from gpurec.optim.value_and_grad import forward_solve, make_value_and_grad

_D = "tests/data/alerax/test_trees_200"
_SO = dict(e_max_iter=2000, e_tol=1e-10, pi_iters=128, neumann_terms=64, self_loop_solver="neumann",
           bicgstab_max_iter=500, bicgstab_tol=1e-10, bicgstab_breakdown_tol=1e-30,
           adjoint_pruning_threshold=1e-6, use_adjoint_pruning=True, pibar_side_threshold=0.0)

def build_genewise_model(n_fam=2, dtype=torch.float64, device="cuda"):
    so = SolverOptions(**_SO); so.validate()
    m = GeneReconModel(f"{_D}/sp.nwk", [f"{_D}/g.nwk"] * n_fam, mode="genewise",
                       device=device, dtype=dtype, solver_options=so)
    assert len(m.batch_statics) == 1
    return m

@pytest.mark.gpu
def test_genewise_theta_hvp_matches_fd():
    m = build_genewise_model(); static = m.batch_statics[0]
    F = len(m.families); S = int(m.species_helpers["S"])
    theta = torch.full((F, 3), math.log2(0.1), device="cuda", dtype=torch.float64)
    rw = torch.zeros(S, device="cuda", dtype=torch.float64)
    _l, sv = forward_solve([static], theta, rw)
    hvp = make_exact_hvp([static], theta, rw, sv, tangent_self_iters=128)  # genewise path (Task 2)
    fd = _fd_hessian_hvp(make_value_and_grad([static], rw, theta_shape=(F, 3)),
                         theta.reshape(-1).contiguous(), None, eps=1e-5)
    Hs = []
    for j in range(3):  # broadcast e_j across all families
        u = torch.zeros(F, 3, device="cuda", dtype=torch.float64); u[:, j] = 1.0; u = u.reshape(-1)
        Ha, Hf = hvp(u).double(), fd(u).double()
        rel = float((Ha - Hf).abs().max()) / max(float(Hf.abs().max()), 1e-30)
        assert torch.isfinite(Ha).all() and rel < 5e-4, f"e_{j}: rel={rel:.2e}"
        Hs.append((u, Ha))
    (u, Hu), (w, Hw) = Hs[0], Hs[1]
    sym = abs(float(torch.dot(u, Hw)) - float(torch.dot(w, Hu)))
    scale = max((abs(float(torch.dot(u, Hu))) * abs(float(torch.dot(w, Hw)))) ** 0.5, 1e-30)
    assert sym / scale < 5e-3, f"asym={sym/scale:.2e}"
```

- [ ] **Step 2: Run it — expect FAIL** (`rel ~4e2`, non-symmetric — the current `(S,3)` projection)

Run: `pytest tests/test_genewise_hvp.py::test_genewise_theta_hvp_matches_fd -v`
Expected: FAIL, `rel` ~ hundreds, symmetry assert also fails. This is the documented starting state.

- [ ] **Step 3: Commit the red gate**

```bash
git add tests/test_genewise_hvp.py
git commit -m "test(hvp): genewise theta HVP gate vs fp64 FD (red: (S,3) projection is wrong)"
```

---

### Task 2: Genewise θ seed/projection (green the θ gate)

**Files:**
- Modify: `gpurec/optim/hvp_exact.py` — the parameter-side of `make_exact_hvp` (the θ tangent **seed** feeding `jvp_root_scores`, and the reverse **projection** that produces `out_theta`, around `hvp_exact.py:259-518`).
- Test: `tests/test_genewise_hvp.py::test_genewise_theta_hvp_matches_fd` (from Task 1).

**Interfaces:**
- Consumes: `static.genewise` (already threaded into `make_exact_hvp`), `extract_parameters_uniform` (`extract_parameters.py:25-38`) for the θ→rates map.
- Produces: `make_exact_hvp(..., static.genewise=True)` returns a correct block-diagonal genewise HVP; `hvp(u_flat[3F]) -> [3F]`.

- [ ] **Step 1: Read the specieswise parameter projection FIRST.** In `make_exact_hvp` identify (a) how the θ tangent `u` (`hvp_exact.py:265`, `u = u_vec[:theta_numel].reshape(theta.shape)`) is turned into the per-species rate tangents that seed `jvp_root_scores`, and (b) where the reverse produces `out_theta` (`~:507-518`). Note the smooth head via `phi1`/`phi2` (`:256`, `:498-522`). The specieswise map is `theta[s] ↔ species s` (identity in S); genewise needs broadcast in / sum out.

- [ ] **Step 2: Implement the genewise seed.** For genewise, `theta` is `[G,3]`; `extract_parameters_uniform` maps each family's 3-logit row through `log_softmax([0, θ_g])` (`extract_parameters.py:30-38`) to `(log_pS,log_pD,log_pL,log_pT)` **broadcast across all S species**. The tangent seed = push `u_θ[G,3]` through that per-family log-softmax Jacobian → per-family per-species rate tangents (same value for every species within a family). Gate on `static.genewise` so the specieswise path is bit-for-bit unchanged.

- [ ] **Step 3: Implement the genewise projection.** The reverse currently emits a per-species `(S,3)`-shaped rate-cotangent. For genewise, **sum the per-species rate-cotangents within each family back to the 3 logit slots** (apply the transpose of the log-softmax Jacobian, then sum over S): `out_theta[g] = Σ_s Jᵀ · cotangent[g,s]` → `[G,3]`. This is the fix for the ~400×/asymmetry.

- [ ] **Step 4: Run the θ gate — expect PASS**

Run: `pytest tests/test_genewise_hvp.py::test_genewise_theta_hvp_matches_fd -v`
Expected: PASS (`rel < 5e-4`, symmetric).

- [ ] **Step 5: Regression — specieswise gate still passes.** Re-run the existing specieswise HVP verification to prove no regression.

Run: `python -m gpurec.optim._verify_hvp 8`
Expected: `[hvp gate] ALL PASS`.

- [ ] **Step 6: Commit**

```bash
git add gpurec/optim/hvp_exact.py
git commit -m "feat(hvp): genewise theta seed/projection (broadcast in, sum-over-species out)"
```

---

### Task 3: `genewise_hessian_blocks` (θ-only) + golden fp32 test

**Files:**
- Create: `gpurec/optim/genewise_curvature.py`
- Test: `tests/test_genewise_hvp.py` (add `test_theta_blocks_fp32_vs_fp64`)

**Interfaces:**
- Consumes: `forward_solve`, `make_exact_hvp` (genewise, from Task 2).
- Produces: `genewise_hessian_blocks(static, theta, receiver_weights, sv, *, active=("theta",)) -> {"H_tt": Tensor[F,3,3]}`. Later tasks extend `active` and the returned dict.

- [ ] **Step 1: Write the failing golden test** (fp32 blocks vs fp64, on the same 2-family fixture)

```python
@pytest.mark.gpu
def test_theta_blocks_fp32_vs_fp64():
    from gpurec.optim.genewise_curvature import genewise_hessian_blocks
    from gpurec.optim.value_and_grad import forward_solve
    def blocks(dtype):
        m = build_genewise_model(dtype=dtype); st = m.batch_statics[0]
        F = len(m.families); S = int(m.species_helpers["S"])
        th = torch.full((F, 3), math.log2(0.1), device="cuda", dtype=dtype)
        rw = torch.zeros(S, device="cuda", dtype=dtype)
        _l, sv = forward_solve([st], th, rw)
        return genewise_hessian_blocks(st, th, rw, sv)["H_tt"]
    H32, H64 = blocks(torch.float32).double(), blocks(torch.float64)
    assert torch.isfinite(H32).all()
    torch.testing.assert_close(H32, H64, rtol=1e-3, atol=1e-2)
    # symmetry of each 3x3 block
    assert float((H64 - H64.transpose(1, 2)).abs().max()) < 1e-6
```

- [ ] **Step 2: Run — expect FAIL** (`genewise_hessian_blocks` undefined)

Run: `pytest tests/test_genewise_hvp.py::test_theta_blocks_fp32_vs_fp64 -v`
Expected: FAIL, ImportError.

- [ ] **Step 3: Implement `genewise_hessian_blocks` (θ-only)** — fire the 3 broadcast probes, assemble `[F,3,3]`, symmetrize.

```python
import torch
from gpurec.optim.hvp_exact import make_exact_hvp

def genewise_hessian_blocks(static, theta, receiver_weights, sv, *, active=("theta",)):
    assert tuple(active) == ("theta",), "P0: theta-only; omega added in P2"
    F = theta.shape[0]
    hvp = make_exact_hvp([static], theta, receiver_weights, sv,
                         tangent_self_iters=int(static.solver_options.pi_iters))
    cols = []
    for j in range(3):
        u = torch.zeros(F, 3, device=theta.device, dtype=theta.dtype); u[:, j] = 1.0
        cols.append(hvp(u.reshape(-1)).reshape(F, 3))          # column j of every block
    H = torch.stack(cols, dim=-1)                              # [F,3,3], H[:,:,j] = col j
    return {"H_tt": 0.5 * (H + H.transpose(1, 2))}
```

- [ ] **Step 4: Run — expect PASS**

Run: `pytest tests/test_genewise_hvp.py::test_theta_blocks_fp32_vs_fp64 -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add gpurec/optim/genewise_curvature.py tests/test_genewise_hvp.py
git commit -m "feat(optim): genewise_hessian_blocks (theta-only, 3 broadcast probes)"
```

**► P0 done: the analytic genewise certificate (theta-only) is available. P1–P3 add per-family origination.**

---

### Task 4: Per-family origination parameter `[G,S]`

**Files:**
- Modify: `gpurec/api/model.py` (origination parameter; currently global `[S]` at `model.py:79`).
- Test: `tests/test_genewise_hvp.py` (add `test_per_family_origination_shape`).

**Interfaces:**
- Produces: `GeneReconModel(..., mode="genewise", per_family_origination=True)` with `model.origination_weights` of shape `[G, S]`; a `_origination_for_static(static, omega)` selector mirroring `_theta_for_static` (`_execution.py:18-19`).

- [ ] **Step 1: Read `model.py:70-81` and `_execution.py:18-19`** (how `theta_shape` and `theta_for_static` are built) so origination follows the identical per-family pattern.

- [ ] **Step 2: Write the failing test**

```python
@pytest.mark.gpu
def test_per_family_origination_shape():
    m = build_genewise_model()  # extend build_genewise_model to pass per_family_origination=True
    assert tuple(m.origination_weights.shape) == (len(m.families), int(m.species_helpers["S"]))
```

- [ ] **Step 3: Implement** the `[G,S]` origination parameter behind a `per_family_origination` flag (default off, so all existing behaviour is unchanged); add `_origination_for_static`.

- [ ] **Step 4: Run — expect PASS**; **Step 5: Commit** (`feat(api): per-family origination [G,S] for genewise`).

---

### Task 5: Per-family weighted NLL + gradient

**Files:**
- Modify: `gpurec/api/_execution.py` (thread per-family `ω_g` into the aggregation), `gpurec/core/inference/solver.py` (weighted branch already exists, `:165-169`).
- Test: `tests/test_genewise_hvp.py` (add `test_origination_grad_matches_fd`).

**Interfaces:**
- Consumes: `_origination_for_static` (Task 4), `nll_vector_from_root_rows(..., origination_log_probs, origination_probs)` (`solver.py:158-169`), `origination_grad_from_root_rows` (`solver.py:184-199`).
- Produces: genewise loss+grad that returns a per-family origination gradient `[G,S]`.

- [ ] **Step 1: Write the failing test** — analytic `∂NLL/∂ω_g` vs central-difference of the loss, per family (`rtol 1e-3`, fp64).
- [ ] **Step 2: Run — expect FAIL.**
- [ ] **Step 3: Implement** per-family `ω_g` selection into each family's `logsumexp2(root_rows + log_softmax2(ω_g))` and weighted `survival_from_E(E_g, softmax2(ω_g))`; return the `[G,S]` origination gradient (today discarded).
- [ ] **Step 4: Run — expect PASS.**  **Step 5: Commit** (`feat: per-family weighted origination NLL + gradient`).

---

### Task 6: `H_ωω` head Hessian + `H_θω` coupling — full `(3+S)` gate

**Files:**
- Modify: `gpurec/optim/hvp_exact.py` (`_head_seed_tangents`, `:81-104`) — per-family instead of global-sum.
- Modify: `gpurec/optim/genewise_curvature.py` — extend `genewise_hessian_blocks` to `active=("theta","omega")`.
- Test: `tests/test_genewise_hvp.py` (add `test_joint_block_hvp_matches_fd`).

**Interfaces:**
- Produces: `genewise_hessian_blocks(..., active=("theta","omega")) -> {"H_tt":[F,3,3], "H_to":[F,3,S], "H_oo_diag":[F,S], "H_oo_lr":[F,S,r]}` and a joint `hvp(u; active)` over `[u_θ(3F); u_ω(SF)]`.

- [ ] **Step 1: Write the failing joint gate** — build the `[G,S]` origination model; compare joint `hvp([u_θ;u_ω])` to fp64 FD of the joint value-and-grad for broadcast `e_j` (θ) and broadcast `e_k` (ω, a few random `k`) plus random directions; assert `rel < 5e-4` and symmetry.
- [ ] **Step 2: Run — expect FAIL** (ω not wired into the HVP).
- [ ] **Step 3a: Read `_head_seed_tangents` (`:81-104`)** — it already double-backwards the origination NLL head for a *global* ω; identify where it sums to `[S]`.
- [ ] **Step 3b: Implement** per-family `H_ωω` (closed-form head Hessian from `root_rows_g, E_g`; keep it as `diag + low-rank` — origination-prior `LSE(ω+r)−LSE(ω)` gives diag+rank-2, survival gives diag+rank-1 scaled by `1/survival`, `1/survival²` rank-1) and `H_θω` (contract the θ-sweep JVP of `(root_rows,E)` with the head's `∂²/∂(root_rows,E)∂ω`). Seed `u_ω` into the head; project the head cotangent to `[G,S]`.
- [ ] **Step 4: Run — expect PASS.**  **Step 5: Commit** (`feat(hvp): per-family omega head Hessian + theta-omega coupling`).

---

### Task 7: Per-family Newton (Schur + Woodbury) + wire the certificate

**Files:**
- Modify: `gpurec/optim/genewise_curvature.py` (add `newton_step_genewise`).
- Modify: `gpurec/optim/genewise_fit.py` — certificate Hessian (`:264-270`) uses the analytic block.
- Test: `tests/test_genewise_hvp.py` (add `test_newton_step_solves_block`), plus an e2e cert check.

**Interfaces:**
- Consumes: the blocks dict from Task 6.
- Produces: `newton_step_genewise(blocks, g_theta[F,3], g_omega[F,S], mu) -> (dtheta[F,3], domega[F,S])`.

- [ ] **Step 1: Write the failing test** — for a random SPD-ified block and gradient, `newton_step_genewise` reproduces the dense per-family solve (`torch.linalg.solve` on the assembled `(3+S)` block) to `rtol 1e-5`.
- [ ] **Step 2: Run — expect FAIL.**
- [ ] **Step 3: Implement** Woodbury inverse of `H_oo = diag + low-rank`, Schur-complement to the `3×3` θ system, back-substitute `δ_ω`; eigenvalue-floor to `mu` for PD (on the reduced system / via a PD guard consistent with `genewise_fit`'s `mu`).
- [ ] **Step 4: Run — expect PASS.**
- [ ] **Step 5: e2e** — `fit_genewise(..., per_family_origination=True, certify=True)` completes with finite curvature and reports `interior_pd`; the analytic `lam_min` matches an fp64 central-difference `lam_min` to `rtol 1e-2`.
- [ ] **Step 6: Commit** (`feat(optim): per-family Newton (Schur+Woodbury) + analytic genewise certificate`).

---

## P4 (optional optimization — not tasked in detail)

- Batch the width-3 θ direction axis into the tangent/SO kernels → `H_tt`/`H_to` in **one** batched forward-over-reverse instead of 3 (add a size-3 leading axis to the tangent buffers; kernels loop/vectorize it).
- Apply `H_oo` matrix-free (Woodbury) in Newton-CG rather than forming `diag+low-rank` explicitly.
- Gate: unchanged (must not move `rel`); measure wall-clock before/after per the profiling rule.

## Self-review notes

- **Spec coverage:** P0 = Tasks 1–3; P1 = Tasks 4–5; P2 = Task 6; P3 = Task 7; P4 noted. §8 identifiability caveat is a *modeling* decision (a prior on `ω_g`) — out of scope for these tasks but flagged in the e2e (Task 7 conditioning via `mu`).
- **Investigation-gated tasks:** Tasks 2 and 6 modify Triton/adjoint internals whose exact diff depends on reading the cited functions first — each has an explicit "read FIRST" step and a complete acceptance gate; the *tests* are fully specified, which is the real contract.
- **Type consistency:** `genewise_hessian_blocks` returns a dict grown across Tasks 3→6 (`H_tt`, then `+H_to/H_oo_diag/H_oo_lr`); `newton_step_genewise` consumes that exact dict.
