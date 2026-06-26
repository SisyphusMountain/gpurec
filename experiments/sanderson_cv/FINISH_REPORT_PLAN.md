# Execution Plan — Finish the gpurec Progress Report on the local RTX 4090

*(Authored by a scoping/synthesis subagent workflow, 2026-06; CV data independently re-verified.)*

Verified on disk:
- CV data exists: coarse `runs/cv_1055/state.pt` (lam*=0.1), refined `runs/cv_1055_refined/state.pt`
  (**lam*=0.03**, min held-out NLL 106955.75 vs unpenalized MLE 107043.80, −88 NLL).
- §4.4 joint (theta,alpha) code is on worktree `receiver-weights-hvp` (branch `recv-weights-s9-curvature`),
  NOT the paper worktree. Driver `converge_bounded_joint_archaea.py`. Without-w baseline checkpoint on disk:
  `bounded_archaea_full_lam0.03_fp64_CERTIFIED.pt` (loss 359591.92).
- **Paper error:** §4.5 claims κ*≈1 and saturation 0.64→0.05, but the data selects κ*=0.03 and Hogenom
  frac_extreme runs 0.43 (κ=0) → 0.21 (κ=1) — 0.64→0.05 is the archaea loose-box number, not Hogenom.

## Feasibility
| Experiment | Verdict | Reason |
|---|---|---|
| §4.5 CV curve | ready-now (CPU) | both Hogenom CV state.pt exist; pure matplotlib |
| §4.6 ablation rows 1–2 (baseline, +smoothing) | ready-now (CPU) | already in cv_1055; transcribe |
| §4.4 receiver weights | local-feasible (GPU) | joint driver + FD-validated HVP on recv worktree; warm-start on disk |
| §4.6 ablation rows 3–4 (+transfer, +both) | needs-other-code | no CV+held-out joint-fit driver exists; must write `run_cv_joint.py` |

## Ordered run list (serial; CPU first, one GPU job at a time)
1. **§4.5** write `plot_fig_cv.py` → `paper/figures/fig_cv.pdf` (CPU, ~2s).
2. **§4.6 rows 1–2** transcribe baseline 107043.80 / +smoothing 106955.75 (κ*=0.03) from cv_1055_refined.
3. **Paper edits**: fix §4.5 κ*→0.03, drop the wrong 0.64→0.05 + borrowed λ_min; insert fig_cv; fill
   `tab:ablation` rows 1–2; surgical §4.2 relabel of "cross-validated κ*≈1" (κ=1 is a stronger penalty, not the CV optimum).
4. **§4.4 import + .so check** on the recv worktree (CPU, ~5s).
5. **§4.4 validate 256-fam joint fit** (GPU, ~3–8 min) — gate: must certify before the full fit.
6. **§4.4 full-archaea joint fit** (GPU, ~30–90 min) → w[119], se_w, F, cert.
7. **§4.4 held-out WITH-vs-WITHOUT** (GPU) or in-sample AIC/BIC fallback.
8. **§4.4 writeup** (top transfer sinks + likelihood comparison).
9. **Deferred** (+transfer/+both ablation): write+validate `run_cv_joint.py` first; until then rows 3–4 stay `(pending)`.

## Cannot finish locally this cycle
§4.6 rows 3–4: no cross-validated joint-fit driver on Hogenom. `run_cv.py` optimizes theta with FIXED rw;
the only joint driver is archaea-only/single-fit/no-CV. Minimal unblock = write `run_cv_joint.py`
(per-fold optimize z=[theta;beta] via `make_value_and_grad(optimize_receiver=True)` + `make_reparam`,
then held-out NLL with the optimized w), FD-check, validate convergence at small scale before any GPU run.

## Env prefix
```
WT=/home/enzo/Documents/git/gpurec/agent-worktrees/kernel-bench-mapcv-merge
RW=/home/enzo/Documents/git/gpurec/agent-worktrees/receiver-weights-hvp
PY=/home/enzo/miniforge3/bin/python
PYTHONNOUSERSITE=1 PYTHONPATH=$WT GPUREC_PREPROCESS_PATH=$WT/crates/gpurec-preprocess/target/release/libgpurec_preprocess.so GPUREC_MEMORY_POLICY_RESERVE_GIB=0 SADDLE_DTYPE=float32 $PY ...
```
