"""Archaeal held-out validation of the transfer-weight model (GPT 5.5 Pro's #1 ask).

The transfer-weight claim is MADE on archaea (S=119 branches, 118 recipient params over 5,446 families
-- a defensible param/data ratio, unlike Hogenom S=1331). AIC/BIC are in-sample; this gives the held-out
counterpart. Single 80/20 family split (seed=0, fold 0 of K=5). For lam in {0.03, 0}, fit two models on
the 80% train and evaluate DATA NLL on the held-out 20%, BOTH inside the box [0.05,2.0] that reconditions
the problem (Sec.4.4) -- without it the joint fit hits unescapable saddles:

    uniform recipients : box theta-only fit,             held-out NLL(test, theta, uniform)
    recipient weights  : box joint (theta,alpha) + SADDLE ESCAPE, held-out NLL(test, theta, alpha)

-> the 2x2 ablation {no-smooth, smooth} x {uniform, weights} on held-out archaeal likelihood.
Recipient arg to heldout_nll is ALPHA = log-weights. Env: FAMILIES(0=all) K(5) FOLD(0) LAMS("0.03,0.0")
MIN_RATE(0.05) MAX_RATE(2.0) SMOKE(0) OUT.
"""
import os, sys, time, math
import numpy as np
import torch
RW = "/home/enzo/Documents/git/gpurec/agent-worktrees/receiver-weights-hvp"
sys.path.insert(0, RW)
import run_cv
from run_cv import DATASETS, build_model, kfold_indices, heldout_nll, _CV_SO
from run_cv_joint import gbm_fit_joint
from gpurec import SolverOptions
from gpurec.optim.value_and_grad import make_value_and_grad
DEV = "cuda"


def gbm_fit_theta_box(batch_statics, theta0, rw, sp_parent, *, lam_tree, box, lbfgs_iters=150, maxcor=50):
    """argmin_theta NLL(theta, fixed rw) + (lam/2)||grad theta||^2, theta in the box. L-BFGS-B w/ bounds."""
    from scipy.optimize import minimize
    S3 = tuple(theta0.shape); tn = theta0.numel()
    lo, hi = math.log2(box[0]), math.log2(box[1])
    f = make_value_and_grad(batch_statics, rw, theta_shape=S3, tree_penalty=(lam_tree, sp_parent))
    t0 = time.perf_counter(); ns = [0]
    def fun(x_np):
        ns[0] += 1
        loss, g, _s, _c = f(torch.tensor(x_np, device=DEV, dtype=torch.float32))
        return float(loss), g.double().cpu().numpy().astype(np.float64)
    x0 = theta0.reshape(-1).clamp(lo, hi).double().cpu().numpy().astype(np.float64)
    r = minimize(fun, x0, jac=True, method="L-BFGS-B", bounds=[(lo, hi)] * tn,
                 options={"maxiter": lbfgs_iters, "maxfun": lbfgs_iters * 2, "maxcor": maxcor,
                          "ftol": 1e-13, "gtol": 1e-9})
    th = torch.tensor(r.x, device=DEV, dtype=torch.float32).reshape(S3)
    return th, dict(final_loss=float(r.fun), final_gnorm=float(np.linalg.norm(r.jac)),
                    n_solves=ns[0], wall_s=time.perf_counter() - t0)


def main():
    smoke = os.environ.get("SMOKE", "0") == "1"
    n_fam = int(os.environ.get("FAMILIES", "256" if smoke else "0"))
    K = int(os.environ.get("K", "5")); fold = int(os.environ.get("FOLD", "0"))
    lams = [float(x) for x in os.environ.get("LAMS", "0.03" if smoke else "0.03,0.0").split(",")]
    box = (float(os.environ.get("MIN_RATE", "0.05")), float(os.environ.get("MAX_RATE", "2.0")))
    seed = int(os.environ.get("SEED", "0"))
    adam = int(os.environ.get("ADAM", "20" if smoke else "40"))
    lbfgs = int(os.environ.get("LBFGS_ITERS", "30" if smoke else "150"))
    escape_m = int(os.environ.get("ESCAPE_M", "60" if smoke else "120"))
    out = os.environ.get("OUT", f"{RW}/experiments/sanderson_cv/runs/archaea_heldout.pt")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    ds = DATASETS["archaea"]; run_cv._SP_TREE = ds["species_tree"]
    so = SolverOptions(**_CV_SO); so.validate()
    paths = ds["families"](None if n_fam <= 0 else n_fam); n = len(paths)
    folds = kfold_indices(n, K, seed); tr, te = folds[fold]
    print(f"[archaea-heldout] n={n} split fold {fold}/{K}: train={len(tr)} test={len(te)} "
          f"lams={lams} box={box} smoke={smoke}", flush=True)

    full = build_model(paths, so); S = int(full.species_helpers["S"])
    rw_uniform = full.receiver_weights.detach().clone()
    sp_parent = full.species_helpers["sp_parent"].detach().clone()
    spp = sp_parent.to(DEV).long().reshape(-1)
    child = (spp >= 0).nonzero(as_tuple=True)[0].contiguous(); parent = spp[child].contiguous()
    theta0 = torch.full((S, 3), math.log2(0.1), device=DEV, dtype=torch.float32)
    g = torch.Generator(device=DEV).manual_seed(seed)
    alpha0 = 0.05 * torch.randn(S, generator=g, device=DEV, dtype=torch.float32)
    print(f"[archaea-heldout] S={S} branches -> +{S-1} recipient params (118/5446 = good ratio)", flush=True)

    train = build_model([paths[i] for i in tr], so)
    test = build_model([paths[i] for i in te], so)
    cells = {}; t0 = time.time()
    def save():
        torch.save(dict(cells={f"{k[0]},{k[1]}": v for k, v in cells.items()}, n=n, S=S, box=box,
                        train=len(tr), test=len(te), fold=fold, K=K, seed=seed), out)
    for lam in lams:
        th_u, st_u = gbm_fit_theta_box(train.batch_statics, theta0, rw_uniform, sp_parent,
                                       lam_tree=lam, box=box, lbfgs_iters=lbfgs)
        ho_u = float(heldout_nll(test.batch_statics, th_u, rw_uniform))
        cells[(lam, "uniform")] = dict(heldout=ho_u, gnorm=st_u["final_gnorm"], wall=st_u["wall_s"])
        print(f"  [lam={lam:g} UNIFORM ] held-out NLL={ho_u:.2f}  |g|={st_u['final_gnorm']:.2e} "
              f"({st_u['wall_s']:.0f}s, total {time.time()-t0:.0f}s)", flush=True); save()
        th_w, al_w, st_w = gbm_fit_joint(train.batch_statics, theta0, alpha0, sp_parent, child, parent,
                                         lam_tree=lam, adam_steps=adam, lbfgs_iters=lbfgs,
                                         escape_m=escape_m, max_escapes=4, box=box)
        ho_w = float(heldout_nll(test.batch_statics, th_w, al_w))
        flag = "" if st_w["lam_min"] > -1e-3 else "  <<SADDLE-NOT-ESCAPED"
        cells[(lam, "weights")] = dict(heldout=ho_w, gnorm=st_w["final_gnorm"], lam_min=st_w["lam_min"],
                                       escapes=st_w["n_escapes"], wall=st_w["wall_s"])
        print(f"  [lam={lam:g} WEIGHTS ] held-out NLL={ho_w:.2f}  |g|={st_w['final_gnorm']:.2e} "
              f"lam_min={st_w['lam_min']:+.2e} escapes={st_w['n_escapes']} "
              f"d(weights-uniform)={ho_w-ho_u:+.2f} ({st_w['wall_s']:.0f}s, total {time.time()-t0:.0f}s){flag}", flush=True); save()

    print("\n=== 2x2 held-out NLL (lower=better; held-out 20% archaea) ===", flush=True)
    for lam in lams:
        u = cells.get((lam, "uniform"), {}).get("heldout"); w = cells.get((lam, "weights"), {}).get("heldout")
        if u is not None and w is not None:
            tag = "smoothing" if lam > 0 else "no smoothing"
            print(f"  lam={lam:g} ({tag}): uniform {u:.1f} | weights {w:.1f} | weights-uniform {w-u:+.1f} "
                  f"({'WEIGHTS HELP' if w<u else 'weights hurt'})", flush=True)
    print(f"[saved] {out}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
