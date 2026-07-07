"""Held-out CV for the JOINT (theta, alpha) model on Hogenom -- fills ablation rows 3-4.

Extends run_cv's per-fold theta-only gbm_fit to a joint fit over z=[theta_flat; alpha] (alpha = per-
species receiver LOGITS; softmax alpha->w inside the kernel), evaluated held-out with the FITTED alpha
(log-weight convention -- passing probabilities collapses to ~uniform). Same kfold seed=0 as run_cv so
the folds match rows 1-2.

CRITICAL: L-BFGS-B keeps a PD Hessian model and CANNOT escape saddles (saddle_escape.py). So after the
Adam->L-BFGS endgame, each fold runs the PROVEN saddle-escape loop: deflated-gauge Lanczos (the exact
joint HVP) -> if lam_min<0, line-search along the most-negative-curvature eigenvector and L-BFGS
re-converge -> re-check, iterate. A fold is accepted only at lam_min >= -tol (no negative curvature).

Env: FAMILIES(1055) K(5) LAMS("0.0,0.03") ADAM(40) LBFGS_ITERS(120) ESCAPE_M(120) MAX_ESCAPES(4)
     SEED(0) SMOKE(0) OUT.
"""
import os, sys, math, time
import numpy as np
import torch
RW = "/home/enzo/Documents/git/gpurec/agent-worktrees/receiver-weights-hvp"
sys.path.insert(0, RW)
import run_cv
from run_cv import DATASETS, build_model, kfold_indices, heldout_nll, _CV_SO
from converge_bounded_joint_archaea import build_joint_hvp_multibatch, make_tree_lap
from gpurec import SolverOptions
from gpurec.solver.value_and_grad import make_value_and_grad
from gpurec.fit.optimize import Schedule
from gpurec.solver.receiver_curvature import certify_joint_min

DEV = "cuda"


def gbm_fit_joint(batch_statics, theta0, alpha0, sp_parent, child, parent, *, lam_tree,
                  adam_steps=40, adam_lr=1.0, lbfgs_iters=120, maxcor=50,
                  escape_m=120, max_escapes=4, sad_tol=1e-3, box=None):
    """argmin_{theta,alpha} NLL + (lam/2)||grad theta||^2, with saddle escape.
    Adam -> L-BFGS-B -> [deflated-gauge Lanczos: if lam_min<0, descend v_min + re-L-BFGS]*. ``box``=
    (min_rate,max_rate) bounds the log2-rate theta block (alpha free) -- this reconditions the joint
    problem the same way the box does in Sec.4.4, preventing the boundary/saddle pathology. Returns
    (theta_hat, alpha_hat, stats incl lam_min and n_escapes)."""
    import math
    from scipy.optimize import minimize
    S3 = tuple(theta0.shape); S = S3[0]; tn = 3 * S
    lo = math.log2(box[0]) if box else None
    hi = math.log2(box[1]) if box else None
    bnds = ([(lo, hi)] * tn + [(None, None)] * S) if box else None
    f = make_value_and_grad(batch_statics, alpha0, theta_shape=S3, optimize_receiver=True,
                            tree_penalty=(lam_tree, sp_parent))
    lap = make_tree_lap(child, parent, lam_tree)
    t0 = time.perf_counter(); n_solves = [0]

    def clamp_th(z):
        if box: z[:tn] = z[:tn].clamp(lo, hi)
        return z

    def val(z):                                            # loss only (line search)
        n_solves[0] += 1
        return float(f(clamp_th(z.clone()).to(DEV).float())[0])

    def lbfgs(z0):
        def fun(x_np):
            n_solves[0] += 1
            loss, g, _s, _c = f(torch.tensor(x_np, device=DEV, dtype=torch.float32))
            return float(loss), g.double().cpu().numpy().astype(np.float64)
        r = minimize(fun, clamp_th(z0.clone()).double().cpu().numpy().astype(np.float64), jac=True,
                     method="L-BFGS-B", bounds=bnds,
                     options={"maxiter": lbfgs_iters, "maxfun": lbfgs_iters * 2, "maxcor": maxcor,
                              "ftol": 1e-13, "gtol": 1e-9})
        return torch.tensor(r.x, device=DEV, dtype=torch.float64), float(r.fun), float(np.linalg.norm(r.jac))

    z = torch.cat([theta0.reshape(-1).float(), alpha0.reshape(-1).float()]).double()
    if adam_steps > 0:                                     # basin entry
        leaf = clamp_th(z.clone()).float().clone().requires_grad_(True)
        opt = torch.optim.Adam([leaf], lr=adam_lr); sched = Schedule("adaptive", adam_lr, t_max=adam_steps)
        for it in range(int(adam_steps)):
            loss, g, _s, _c = f(leaf.detach()); n_solves[0] += 1
            opt.param_groups[0]["lr"] = sched.update(loss, g); leaf.grad = g; opt.step()
            with torch.no_grad():
                if box: leaf[:tn] = leaf[:tn].clamp(lo, hi)
        z = leaf.detach().double()
    z, F, gnorm = lbfgs(z)

    # ---- PROVEN saddle-escape loop (L-BFGS cannot escape saddles on its own) ----
    n_escapes = 0; lam_min = float("nan")
    for _ in range(int(max_escapes)):
        th2d = z[:tn].reshape(S3).double(); al = z[tn:].double()
        torch.cuda.empty_cache()                                      # release allocator-held mem so the exact-HVP memory gate passes
        try:
            Av = build_joint_hvp_multibatch(batch_statics, th2d, al, lap, tn, S)
            cert = certify_joint_min(batch_statics, th2d, al, hvp=Av, theta_numel=tn, S=S, m=escape_m,
                                     verbose=False)                   # default proj = gauge (mean-zero alpha)
        except ValueError:                                            # alpha drifted ~uniform -> receiver curv degenerate
            lam_min = float("nan"); break
        except RuntimeError as e:                                     # e.g. memory gate -> skip escape, accept L-BFGS point
            print(f"    [escape skipped: {str(e)[:80]}]", flush=True); lam_min = float("nan"); break
        lam_min = float(cert["lam_min_gauge"]); v = cert["v_min"].double()
        if lam_min > -sad_tol:
            break                                                     # PD / near-min: no saddle to escape
        a_best = min(((val(z + a * v), a) for a in (-4., -2., -1., 1., 2., 4.)), key=lambda t: t[0])[1]
        z, F, gnorm = lbfgs(z + a_best * v)                           # descend v_min, re-converge
        n_escapes += 1

    theta_hat = z[:tn].reshape(S3).float(); alpha_hat = z[tn:].float()
    stats = dict(final_loss=F, final_gnorm=gnorm, lam_min=lam_min, n_escapes=n_escapes,
                 n_solves=n_solves[0], wall_s=time.perf_counter() - t0,
                 alpha_absmax=float(alpha_hat.abs().max()))
    return theta_hat, alpha_hat, stats


def main():
    smoke = os.environ.get("SMOKE", "0") == "1"
    n_fam = int(os.environ.get("FAMILIES", "40" if smoke else "1055"))
    K = int(os.environ.get("K", "2" if smoke else "5"))
    lams = [float(x) for x in os.environ.get("LAMS", "0.03" if smoke else "0.0,0.03").split(",")]
    adam_steps = int(os.environ.get("ADAM", "20" if smoke else "40"))
    lbfgs_iters = int(os.environ.get("LBFGS_ITERS", "15" if smoke else "120"))
    escape_m = int(os.environ.get("ESCAPE_M", "40" if smoke else "120"))
    max_escapes = int(os.environ.get("MAX_ESCAPES", "2" if smoke else "4"))
    seed = int(os.environ.get("SEED", "0"))
    out = os.environ.get("OUT", f"{RW}/experiments/sanderson_cv/runs/cv_joint_hogenom.pt")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    ds = DATASETS["hogenom"]; run_cv._SP_TREE = ds["species_tree"]
    so = SolverOptions(**_CV_SO); so.validate()
    paths = ds["families"](n_fam); n = len(paths)
    print(f"[joint-cv] hogenom n={n} K={K} lams={lams} adam={adam_steps} lbfgs={lbfgs_iters} "
          f"escape_m={escape_m} smoke={smoke}", flush=True)

    full = build_model(paths, so); S = int(full.species_helpers["S"])
    sp_parent = full.species_helpers["sp_parent"].detach().clone()
    spp = sp_parent.to(DEV).long().reshape(-1)
    child = (spp >= 0).nonzero(as_tuple=True)[0].contiguous(); parent = spp[child].contiguous()
    theta0 = torch.zeros((S, 3), device=DEV, dtype=torch.float32)
    g = torch.Generator(device=DEV).manual_seed(seed)
    alpha0 = 0.05 * torch.randn(S, generator=g, device=DEV, dtype=torch.float32)  # non-uniform: wake receiver grad
    print(f"[joint-cv] S={S} species  -> +{S-1} receiver params  p_joint={4*S}", flush=True)

    folds = kfold_indices(n, K, seed)
    cells = {}; t_start = time.time()
    for lam in lams:
        for fi, (tr, te) in enumerate(folds):
            train = build_model([paths[i] for i in tr], so)
            test = build_model([paths[i] for i in te], so)
            th, al, st = gbm_fit_joint(train.batch_statics, theta0, alpha0, sp_parent, child, parent,
                                       lam_tree=lam, adam_steps=adam_steps, lbfgs_iters=lbfgs_iters,
                                       escape_m=escape_m, max_escapes=max_escapes)
            ho = float(heldout_nll(test.batch_statics, th, al))
            cells[(lam, fi)] = dict(heldout=ho, per_fam=ho / max(1, len(te)), test=len(te), **st)
            flag = "" if st["lam_min"] > -1e-3 else "  <<SADDLE-NOT-ESCAPED"
            print(f"  [lam={lam:g} fold {fi}/{K}] heldout={ho:.2f} |g|={st['final_gnorm']:.2e} "
                  f"lam_min={st['lam_min']:+.3e} escapes={st['n_escapes']} a_absmax={st['alpha_absmax']:.2f} "
                  f"({st['wall_s']:.0f}s, total {time.time()-t_start:.0f}s){flag}", flush=True)
            torch.save(dict(cells={f"{k[0]},{k[1]}": v for k, v in cells.items()}, S=S, n=n, K=K,
                            lams=lams, seed=seed), out)
    print("\n=== JOINT held-out NLL (mean over folds) ===", flush=True)
    for lam in lams:
        vals = [cells[(lam, fi)]["heldout"] for fi in range(K) if (lam, fi) in cells]
        esc = sum(cells[(lam, fi)]["n_escapes"] for fi in range(K) if (lam, fi) in cells)
        if vals:
            print(f"  lam={lam:g}: mean heldout {sum(vals)/len(vals):.1f}  ({len(vals)} folds, "
                  f"{esc} saddle-escapes)  per-fold: {', '.join(f'{v:.0f}' for v in vals)}", flush=True)
    print(f"[saved] {out}  ({time.time()-t_start:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
