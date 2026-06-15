# Merge plan: kernel-bench optimization layer → a gpurec branch (definitive)

**Decision (this supersedes the earlier analysis):** stop iterating on kernel-bench in isolation;
**merge all of our kernel-bench work onto one gpurec branch**, organized under a single `gpurec/optim/`
package, so the optimization / MAP / cross-validation research lives in the real app and can run on the
full hogenom dataset. Phased so the **CV goal lands early (low risk)** and the **second-order Triton
kernels are isolated last (high risk)**. End state: nothing important left stranded in kernel-bench.

## 0. Decisions (resolved)

- **Merge base = branch off the COMMIT `f71a38dae`** (HEAD of `codex/restore-batched-lbfgs-notebook`),
  NOT `main` (9abbece31). VERIFIED committed in f71a38dae (not in the dirty working tree): the per-family
  loss `nll_vector_from_root_rows` (solver.py:132) + `evaluate_static_loss_vector_grad` (_execution.py:146)
  — the CV fold unit — and the E-step backward cross-warp race fix (f71a38dae is literally that commit).
  Branching off the commit ignores gpurec's unrelated uncommitted working-tree edits, so the dirty tree
  is a non-issue. (NB: the per-family loss is the real reason for f71a38dae; the race fix is minor —
  one-line, latent in fp32 (our regime), and kernel-bench has it too.)
- **New code lands in a new `gpurec/optim/` package** (isolated; non-optimization users unaffected).
- **Do NOT port the solver/first-order kernels** — kbench/ is a faithful copy of gpurec's; use gpurec's.
- **Rename contract** (apply throughout the port): `item↔family`, `col↔receiver`,
  `state↔species`, `node↔sp`, `solve_e_pi↔solve_resident_e_pi`,
  `extract_parameters_weighted_cols↔extract_parameters_weighted_receivers`,
  `as_item_param↔as_family_param`, `rate_item_idx↔rate_family_idx`.

## 1. What moves where (manifest)

| kernel-bench source | → gpurec destination | phase | notes |
|---|---|---|---|
| `newton/vg.py` | `gpurec/optim/value_and_grad.py` | 1 | re-point onto `gpurec.api._execution.evaluate_static_loss_grad` / `_vector_grad`; add per-family mask |
| `newton/cg.py` | `gpurec/optim/cg.py` | 1 | pure algorithm; copy as-is |
| `newton/optimize.py` | `gpurec/optim/optimize.py` | 1 | `first_order` (Adam), `ridge_anneal`; import fixes |
| `newton/baselines.py` | `gpurec/optim/baselines.py` | 1 | `lbfgs_scipy` (maxcor); import fixes |
| `newton/newton_cg.py` | `gpurec/optim/newton_cg.py` | 3 | needs HVP |
| `newton/specieswise_fit.py` | `gpurec/optim/map_fit.py` | 3 | end-to-end certified-PD recipe; needs HVP |
| **(new)** CV harness | `gpurec/optim/map_cv.py` | 2 | k-fold over families + λ-homotopy |
| `newton/hvp_exact.py` | `gpurec/optim/hvp_exact.py` | 3 | exact HVP; needs SO kernels |
| `newton/forward_tangent.py` | `gpurec/optim/forward_tangent.py` | 3 | JVP; feeds HVP |
| `newton/ggn.py` | `gpurec/optim/ggn.py` | 3 | optional (GGN) |
| `kbench/core/kernels/{e_step_so,e_step_tangent,wave_so,wave_tangent,dts_so,dts_tangent}.py` | `gpurec/core/kernels/` | 3 | **the high-risk part**: Triton SO/tangent kernels |
| `newton/_map_cv_plan.md`, `_specieswise_basin_findings.md`, `finding-the-minimum.md` | `gpurec/docs/optim/` | 4 | findings |
| `newton/{basin_*,gauge_audit,convergence_audit,theta_diagnostics}.py` | `gpurec/optim/diagnostics/` | 4 | research scripts (adapt on demand) |

## 2. Phases

### Phase 0 — branch + scaffold (cheap)
- Branch the merge off f71a38dae (handle its dirty tree first). Create empty `gpurec/optim/__init__.py`.
- Decide the public surface: add to `gpurec/api/_execution.py` a thin
  `make_value_and_grad(static, receiver_weights, *, family_mask=None, grad_avg_K=1)` returning
  `f(theta_vec) -> (loss, grad, saved, warm)` — the single contract the whole optim layer sits on.

### Phase 1 — optimization core, NO HVP (low risk; unblocks CV)
- Port `value_and_grad.py`, `cg.py`, `optimize.py`, `baselines.py` with the rename.
- **Verify:** reproduce a kernel-bench MAP fit through gpurec — `first_order` (Adam) → `lbfgs_scipy`
  reaches the same NLL on an equivalent problem (mint a 666x80-equivalent via
  `scripts/mint_kernel_bench_fixture.py`, or run on a small live hogenom batch). value+grad must match
  kernel-bench to the gradient-noise floor (~2e-4).

### Phase 2 — MAP+CV harness (THE GOAL; low risk — needs only Phase 1)
- **Per-family train/test mask:** objective `Σ_{train} nll_vector_i + (λ/2)‖θ−θ_ref‖²`; seed the backward
  on train families only (zeroing test families' per-family seed masks both `root_rows` and the shared-E
  contribution — verify the masked gradient vs a finite-difference of the masked loss on a tiny subset).
  gpurec's `evaluate_static_loss_vector_grad` already returns the per-family vector; **multi-batch is
  handled natively by `GeneReconModel`** (θ shared across batches), so this scales to the full dataset
  without per-batch fixtures.
- **`map_cv.py`:** k-fold over families (k=5), λ-grid with **warm-start homotopy** (largest λ first,
  decrease warm-started). `CV(λ) = mean held-out predictive NLL`; pick `λ* = argmin`; refit on all
  families at λ*.
- **Run** on hogenom-1055 → the Sanderson CV curve. (Smoke test first: a single 80/20 split at one λ,
  confirm held-out NLL is finite and decreases vs λ=0.)

### Phase 3 — exact HVP + second-order kernels (HIGH risk; for certificate / exact-Newton)
- Port the SO/tangent Triton kernels into `gpurec/core/kernels/` with the rename; get them to **compile**
  in gpurec's build and match numerically.
- Port `hvp_exact.py`, `forward_tangent.py`, `ggn.py`, `newton_cg.py`, `map_fit.py`.
- **Verify:** the exact-HVP FD gate (Hu vs finite-difference of the gradient) reproduces, and
  `lanczos_min_eigpair`-based `λ_min(H+λI) > 0` certifies the MAP minimum at λ* from Phase 2.
- This is the only part that can stall on Triton; it is **off the CV critical path** — if it fights us,
  Phases 1–2 already deliver the CV result; certification can use a cruder bound meanwhile.

### Phase 4 — consolidate docs + diagnostics
- Move the findings docs + basin/audit scripts under gpurec; update paths. Delete/redirect the
  kernel-bench scratch so there's one home for the work.

## 3. Verification gates (don't advance without these)
1. Phase 1: gpurec value+grad ≡ kernel-bench (≤2e-4), and an Adam→L-BFGS fit reaches the known NLL.
2. Phase 2: masked gradient ≡ FD of the masked loss (tiny subset); 80/20 CV smoke test finite + sane.
3. Phase 3: HVP ≡ FD; PD certificate reproduces at λ*.

## 4. Risks (ranked)
1. **SO-kernel Triton compilation in gpurec** (Phase 3) — isolated last, off the CV path.
2. **Per-family backward masking of the shared-E term** (Phase 2) — must FD-verify.
3. **Terminology rename** — pervasive; mechanical; lean on the verification gates.
4. ~~Base-branch dirty tree~~ — RESOLVED: CV-critical pieces are committed in f71a38dae; branch off the
   commit and ignore the unrelated dirty working tree.
5. **Scale/memory at 12k families** — gpurec's batching + adaptive solver already handle it; start at 1055.

## 5. First action on approval
Phase 0 + the Phase-1 import-only port of `cg.py`/`baselines.py` (zero-risk) + the
`make_value_and_grad` contract in `gpurec/api/_execution.py`, then the Phase-1 verification fit.
