"""Genewise bounded convergence with BatchedLBFGS (L-BFGS-B: limited-memory BFGS + native box bounds).

Same bounded problem as bench_genewise_bounded.py (rate = 2^theta in [MIN_RATE, MAX_RATE] = [1e-6, 2] ->
theta in [-19.93, 1.0]), but the optimizer is gpurec's BatchedLBFGS with lower_bound/upper_bound (one
independent bounded L-BFGS per family, dim 0 = family). Converge on the PROJECTED gradient (KKT residual
under the box). Cert: FD 3x3 + projected |g| + bound-active report.

Env: DATASET=hogenom|archaea  FAMILIES=all|N  PI=64  MIN_RATE=1e-6 MAX_RATE=2  TOL=1e-3
     ADAM=20 LBFGS=120  MAXCOR=20  LS=armijo|strong_wolfe  WARM_THETA=  OUT_JSON=
"""
from __future__ import annotations
import os, sys, time, json
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_cv import DATASETS, _CV_SO
from gpurec import GeneReconModel, SolverOptions
from gpurec.batched_lbfgs import BatchedLBFGS
from gpurec.optimization import clamp_log_rate_, project_rate_gradient_, log2_rate_bounds

DEV = "cuda"; DT = torch.float32
DATASET = os.environ.get("DATASET", "hogenom")
_FAM = os.environ.get("FAMILIES", "all"); N_FAM = None if _FAM in ("all", "0", "") else int(_FAM)
PI = int(os.environ.get("PI", "64"))
MIN_RATE = float(os.environ.get("MIN_RATE", "1e-6")); MAX_RATE = float(os.environ.get("MAX_RATE", "2"))
TOL = float(os.environ.get("TOL", "1e-3")); FD_EPS = float(os.environ.get("FD_EPS", "1e-2"))
ADAM = int(os.environ.get("ADAM", "20")); LBFGS = int(os.environ.get("LBFGS", "120"))
MAXCOR = int(os.environ.get("MAXCOR", "20")); LS = os.environ.get("LS", "armijo")
TH_LO, TH_HI = log2_rate_bounds(MIN_RATE, MAX_RATE)
R = dict(meta=dict(dataset=DATASET, families=_FAM, pi=PI, optimizer=f"BatchedLBFGS/{LS}",
                   min_rate=MIN_RATE, max_rate=MAX_RATE, theta_bounds=[TH_LO, TH_HI], tol=TOL))
print(f"=== genewise L-BFGS-B {DATASET} fam={_FAM} pi={PI} rate in [{MIN_RATE},{MAX_RATE}] "
      f"-> theta in [{TH_LO:.2f},{TH_HI:.2f}]  ls={LS} ===", flush=True)


def sopts(pi):
    return SolverOptions(**{**_CV_SO, "pi_iters": pi, "neumann_terms": pi})


def lg(m, th):
    lv, g, _ = m.genewise_loss_vector_and_grad(theta=th, need_grad=True)
    return lv.to(DT), g.to(DT)


def proj_gmax(th, g):
    gp = project_rate_gradient_(th, g.clone(), min_rate=MIN_RATE, max_rate=MAX_RATE)
    return gp.abs().amax(dim=1)


t0 = time.perf_counter()
m = GeneReconModel(str(DATASETS[DATASET]["species_tree"]), [str(x) for x in DATASETS[DATASET]["families"](N_FAM)],
                   mode="genewise", device=DEV, solver_options=sopts(PI), clade_budget=80000)
m.receiver_weights.requires_grad_(False)
F = int(m.theta.shape[0]); S = int(m.species_helpers["S"])
print(f"[build] F={F} S={S} batches={len(m.batch_statics)} ({time.perf_counter()-t0:.1f}s)", flush=True)

WARM = os.environ.get("WARM_THETA")
theta = (torch.load(WARM, map_location=DEV, weights_only=False)["theta"].to(DEV).to(DT).reshape(F, 3)
         if WARM else torch.zeros(F, 3, device=DEV, dtype=DT))
clamp_log_rate_(theta, min_rate=MIN_RATE, max_rate=MAX_RATE)

# ---- Adam warmup (clamped) for basin entry ------------------------------------------------------
t1 = time.perf_counter()
leaf = theta.clone().requires_grad_(True); adam = torch.optim.Adam([leaf], lr=0.05)
for it in range(ADAM):
    _, g = lg(m, leaf.detach()); leaf.grad = g; adam.step()
    with torch.no_grad():
        clamp_log_rate_(leaf, min_rate=MIN_RATE, max_rate=MAX_RATE)
    if it % 10 == 0 or it == ADAM - 1:
        gp = proj_gmax(leaf.detach(), lg(m, leaf.detach())[1])
        print(f"  [adam {it:3d}] |Pg|max={float(gp.max()):.3e} conv={int((gp<TOL).sum())}/{F} ({time.perf_counter()-t1:.1f}s)", flush=True)
theta = leaf.detach().clone()

# ---- BatchedLBFGS (L-BFGS-B), native box bounds -------------------------------------------------
theta_p = theta.clone().requires_grad_(True)
opt = BatchedLBFGS([theta_p], lr=1.0, max_iter=1, history_size=MAXCOR, max_ls=20, line_search_fn=LS,
                   tolerance_grad=1e-14, tolerance_change=1e-18,
                   lower_bound=float(TH_LO), upper_bound=float(TH_HI))

def closure():
    lv, g = lg(m, theta_p.detach()); theta_p.grad = g
    return lv

def loss_closure():
    return m.genewise_loss_vector(theta=theta_p.detach()).to(DT)

t2 = time.perf_counter(); n_lb = 0
for it in range(LBFGS):
    opt.step(closure, loss_closure=loss_closure); n_lb += 1
    _, g = lg(m, theta_p.detach()); gp = proj_gmax(theta_p.detach(), g)
    nconv = int((gp < TOL).sum())
    if it % 10 == 0 or it == LBFGS - 1 or nconv == F:
        nb = int(((theta_p.detach() <= TH_LO + 1e-6) | (theta_p.detach() >= TH_HI - 1e-6)).any(dim=1).sum())
        print(f"  [lbfgsb {it:3d}] |Pg|max={float(gp.max()):.3e} |Pg|med={float(gp.median()):.3e} "
              f"conv={nconv}/{F} bound-active={nb} ({time.perf_counter()-t2:.1f}s)", flush=True)
    if nconv == F:
        print(f"  [lbfgsb] ALL {F} families converged at iter {it}", flush=True); break
theta = theta_p.detach().clone()
opt_s = time.perf_counter() - t1

# ---- cert ---------------------------------------------------------------------------------------
_, g = lg(m, theta); gp = proj_gmax(theta, g)
H = torch.zeros(F, 3, 3, device=DEV, dtype=DT)
for j in range(3):
    tp = theta.clone(); tp[:, j] += FD_EPS; _, gpv = lg(m, tp)
    tm = theta.clone(); tm[:, j] -= FD_EPS; _, gmv = lg(m, tm)
    H[:, :, j] = (gpv - gmv) / (2 * FD_EPS)
H = 0.5 * (H + H.transpose(1, 2)); lam_min = torch.linalg.eigvalsh(H)[:, 0]
at_lo = (theta <= TH_LO + 1e-6); at_hi = (theta >= TH_HI - 1e-6); bound_active = (at_lo | at_hi).any(dim=1)
conv = gp < TOL; pd = lam_min > TOL; total = time.perf_counter() - t0
print(f"\n[cert pi={PI}] BatchedLBFGS/{LS} bounded:", flush=True)
print(f"  CONVERGED (|Pg|<{TOL}) = {int(conv.sum())}/{F}   |Pg|max={float(gp.max()):.3e}", flush=True)
print(f"  interior PD (no active bound) = {int((conv & pd & ~bound_active).sum())}   "
      f"bound-active = {int(bound_active.sum())} [max={int(at_hi.any(dim=1).sum())}, min={int(at_lo.any(dim=1).sum())}]", flush=True)
print(f"  rate range: theta in [{float(theta.min()):.2f},{float(theta.max()):.2f}] -> "
      f"rate in [{2**float(theta.min()):.2e},{2**float(theta.max()):.2f}]", flush=True)
print(f"  TOTAL = {total:.0f}s  (opt {opt_s:.0f}s, {n_lb} lbfgs iters)", flush=True)
R.update(dict(F=F, total_s=total, opt_s=opt_s, lbfgs_iters=n_lb, n_conv=int(conv.sum()),
              n_interior_pd=int((conv & pd & ~bound_active).sum()), n_bound_active=int(bound_active.sum()),
              pg_max=float(gp.max())))
OUT = os.environ.get("OUT_JSON")
if OUT:
    with open(OUT, "w") as fh:
        json.dump(R, fh, indent=2, default=lambda o: float(o) if hasattr(o, "__float__") else str(o))
    print(f"  saved -> {OUT}", flush=True)
