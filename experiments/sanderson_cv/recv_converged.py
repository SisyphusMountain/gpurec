"""Re-evaluate the CV held-out NLL at the CONVERGED per-fold minima, not the L-BFGS endpoints.

`run_cv.py` selects lam* from held-out NLL evaluated at each fold's Adam->L-BFGS *training endpoint*.
But at light lam those endpoints are saddles / soft-direction-floored points (see docs/optim/
newton_polish.md), so the held-out NLL is read off a not-fully-converged theta. This script redoes the
evaluation rigorously: for every (fold, lam) cell it

  1. rebuilds the exact train/test split (run_cv.kfold_indices, same n/k/seed),
  2. loads the cell's saved L-BFGS theta as a warm start,
  3. escape + line-searched Newton-polishes it on the TRAIN penalized objective to a certified min
     (experiments/sanderson_cv/saddle_escape.run -- the single Newton-with-line-search path),
  4. re-evaluates the held-out NLL on the TEST fold at the converged theta (fp64),

and compares to the held-out NLL at the original theta (also recomputed in fp64, so the delta isolates
the theta-convergence effect, not an fp32->fp64 shift). Resumable per cell.

Held-out NLL is the pure predictive data NLL (no penalty), exactly as run_cv.heldout_nll. The receiver
weights `rw`, parent map and S come from the FULL model (global), matching run_cv.

Env:
  DATASET   hogenom|archaea            CKPT_DIR  dir holding fold{fi}_lam{li}.pt (lam read from each file)
  FAMILIES  n (must match the CV run)  OUT_DIR   where to write conv state + per-cell thetas
  K=5  SEED=0  FULL_HESSIAN=auto (full eigh if p<=1200 else Lanczos+CG)  FOLDS=all (e.g. "0" for a spot-check)
"""
from __future__ import annotations
import os, sys, time
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import saddle_escape
from run_cv import DATASETS, _CV_SO, kfold_indices, heldout_nll
from gpurec import GeneReconModel, SolverOptions

DEV = "cuda"
GTOL = 1e-3  # a train fit counts as converged when ||grad|| < GTOL (1e-3 is enough; fp64)


def build(paths, so, sp_tree):
    return GeneReconModel(str(sp_tree), [str(p) for p in paths], mode="specieswise",
                          device=DEV, solver_options=so)


def main():
    ds_name = os.environ.get("DATASET", "archaea")
    n = int(os.environ["FAMILIES"])
    k = int(os.environ.get("K", "5"))
    seed = int(os.environ.get("SEED", "0"))
    ckpt_dir = os.environ["CKPT_DIR"]
    out_dir = os.environ.get("OUT_DIR", "recv_out")
    os.makedirs(out_dir, exist_ok=True)
    fenv = os.environ.get("FULL_HESSIAN")
    full = None if fenv is None else bool(int(fenv))
    only_folds = os.environ.get("FOLDS")
    only_folds = set(int(x) for x in only_folds.split(",")) if only_folds else None

    ds = DATASETS[ds_name]
    sp_tree = ds["species_tree"]
    so = SolverOptions(**_CV_SO); so.validate()
    paths = ds["families"](n)
    assert len(paths) == n, f"{len(paths)} != {n}"
    folds = kfold_indices(n, k, seed)

    # full model: source of the GLOBAL rw / sp_parent / S (exactly as run_cv)
    full_model = build(paths, so, sp_tree)
    S = int(full_model.species_helpers["S"])
    rw = full_model.receiver_weights.detach().clone()
    rw_d = rw.to(DEV).double()
    sp_parent = full_model.species_helpers["sp_parent"].detach().clone()
    del full_model; torch.cuda.empty_cache()
    print(f"[recv] dataset={ds_name} n={n} k={k} seed={seed} S={S} p={3*S} full_hessian={full}", flush=True)

    state_path = os.path.join(out_dir, "recv_state.pt")
    state = torch.load(state_path, weights_only=False) if os.path.exists(state_path) else {"cells": {}}

    def save_state():
        tmp = state_path + ".tmp"; torch.save(state, tmp); os.replace(tmp, state_path)

    for fi, (tr, te) in enumerate(folds):
        if only_folds is not None and fi not in only_folds:
            continue
        # which lambdas exist for this fold
        cells = []
        for li in range(20):
            ck = os.path.join(ckpt_dir, f"fold{fi}_lam{li}.pt")
            if os.path.exists(ck):
                cells.append((li, ck))
        if not cells:
            continue
        if all(f"{fi},{li}" in state["cells"] for li, _ in cells):
            print(f"[fold {fi}] all {len(cells)} cells done -> skip", flush=True); continue
        print(f"\n[fold {fi}] train={len(tr)} test={len(te)}  ({len(cells)} lambdas)", flush=True)
        train = build([paths[i] for i in tr], so, sp_tree)
        test = build([paths[i] for i in te], so, sp_tree)
        for li, ck in cells:
            key = f"{fi},{li}"
            if key in state["cells"]:
                print(f"  [{key}] done -> skip", flush=True); continue
            blob = torch.load(ck, map_location=DEV, weights_only=False)
            lam = float(blob["lam"]); theta_orig = blob["theta"].to(DEV).double()
            t0 = time.time()
            ho_orig = heldout_nll(test.batch_statics, theta_orig.reshape(S, 3), rw_d)
            conv_path = os.path.join(out_dir, f"conv_fold{fi}_lam{li}.pt")
            # CV mode: drive the TRAIN fit to |g|<1e-3 (escape saddle + line-searched Newton); skip the
            # redundant full-Hessian PD certificate (saddle was already detected/escaped at the start).
            res = saddle_escape.run(train.batch_statics, rw, sp_parent, S, ck, lam,
                                    full=full, out_path=conv_path, final_cert=False, polish_tol=1e-3,
                                    meta=dict(fold=fi, li=li, n_test=len(te), n_train=len(tr)))
            theta_star = res["theta_newton"].to(DEV).double()
            converged = res["gnorm_newton"] < GTOL
            if not converged:
                print(f"  !! [{key}] NOT CONVERGED: |g|={res['gnorm_newton']:.3e} > {GTOL:g} "
                      f"(saddle={res['is_saddle']}) -- held-out at this theta is unreliable", flush=True)
            ho_star = heldout_nll(test.batch_statics, theta_star.reshape(S, 3), rw_d)
            state["cells"][key] = dict(
                fold=fi, li=li, lam=lam, n_test=len(te), n_train=len(tr),
                heldout_conv=ho_star, heldout_orig_fp64=ho_orig,
                delta_heldout=ho_star - ho_orig,
                lam_min_saddle=res["lam_min_saddle"], lam_min_newton=res["lam_min_newton"],
                gnorm_saddle=res["gnorm_saddle"], gnorm_newton=res["gnorm_newton"],
                is_saddle=res["is_saddle"], certified=res["certified"], converged=converged,
                loss_saddle=res["loss_saddle"], loss_newton=res["loss_newton"],
                wall_s=time.time() - t0)
            save_state()
            print(f"  [{key}] lam={lam:<5g} heldout {ho_orig:.4f}->{ho_star:.4f} "
                  f"(d={ho_star-ho_orig:+.4f})  train-min: saddle={res['is_saddle']} "
                  f"lam_min {res['lam_min_saddle']:+.4e}->{res['lam_min_newton']:+.4e} "
                  f"|g|->{res['gnorm_newton']:.2e} cert={res['certified']} ({time.time()-t0:.0f}s)", flush=True)
        del train, test; torch.cuda.empty_cache()

    # ---- converged CV curve ----
    lams = sorted({c["lam"] for c in state["cells"].values()}, reverse=True)
    print("\n================= CONVERGED CV CURVE (held-out NLL at certified minima) =================", flush=True)
    print(f"{'lambda':>8} {'n_folds':>7} | {'CONV total':>12} {'ORIG total':>12} {'delta':>9} | {'CONV/fam':>9} | per-fold-fit health", flush=True)
    curve = {}
    for lam in lams:
        cs = [c for c in state["cells"].values() if abs(c["lam"] - lam) < 1e-12]
        conv = sum(c["heldout_conv"] for c in cs); orig = sum(c["heldout_orig_fp64"] for c in cs)
        nfam = sum(c["n_test"] for c in cs); nsad = sum(1 for c in cs if c["is_saddle"])
        nconv = sum(1 for c in cs if c.get("gnorm_newton", 9.9) < GTOL)
        curve[lam] = (conv, orig, len(cs), nfam)
        flag = "" if nconv == len(cs) else f"  !! {len(cs)-nconv} NOT CONVERGED"
        print(f"{lam:>8g} {len(cs):>7} | {conv:12.3f} {orig:12.3f} {conv-orig:+9.3f} | {conv/max(1,nfam):9.4f} | "
              f"saddles {nsad}/{len(cs)}  converged(|g|<{GTOL:g}) {nconv}/{len(cs)}{flag}", flush=True)
    full_lams = [lam for lam, v in curve.items() if v[2] == k]
    if full_lams:
        best_conv = min(full_lams, key=lambda L: curve[L][0])
        best_orig = min(full_lams, key=lambda L: curve[L][1])
        print(f"\nlam* (CONVERGED) = {best_conv}    lam* (orig L-BFGS) = {best_orig}", flush=True)
    print("[recv] DONE", flush=True)


if __name__ == "__main__":
    main()
