"""Genewise convergence + per-family 3x3 FD cert with BOUNDED rates (4090).

theta is a log2 RELATIVE-RATE (rate = 2^theta = P(category)/P(speciation)); extract_parameters softmaxes
it. Unbounded MLE lets the non-identifiable families run the rate to +/-inf (theta -> +/-10..20, the softmax
boundary). Here we BOX-CONSTRAIN rate in [MIN_RATE, MAX_RATE] (default [1e-6, 2]) -> theta in
[log2(MIN_RATE), log2(MAX_RATE)] = [-19.93, 1.0], using gpurec's bounded-rate primitives
(clamp_log_rate_ projects theta; project_rate_gradient_ zeros the gradient at active bounds = KKT).

Optimizer: PROJECTED trust-region Newton (reused 3x3 Hessian, no line search) -- clamp theta each step,
converge on the PROJECTED gradient. A runaway family parks AT the rate bound (projected |g|->0 = a
constrained KKT minimum) instead of diverging. Cert: FD 3x3 + projected |g|; bound-active families are
reported separately (their reduced/interior curvature is what's certifiable).

Env: DATASET=hogenom|archaea  FAMILIES=all|N  PI=64  MIN_RATE=1e-6 MAX_RATE=2  TOL=1e-3
     ADAM=30 NEWTON=60 HESS_EVERY=5  WARM_THETA=  OUT_JSON=
"""
from __future__ import annotations
import os, sys, time, json, math
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_cv import DATASETS, _CV_SO
from gpurec import GeneReconModel, SolverOptions
from gpurec.optimization import clamp_log_rate_, project_rate_gradient_, log2_rate_bounds

DEV = "cuda"; DT = torch.float32
DATASET = os.environ.get("DATASET", "hogenom")
_FAM = os.environ.get("FAMILIES", "all"); N_FAM = None if _FAM in ("all", "0", "") else int(_FAM)
PI = int(os.environ.get("PI", "64"))
MIN_RATE = float(os.environ.get("MIN_RATE", "1e-6")); MAX_RATE = float(os.environ.get("MAX_RATE", "2"))
TOL = float(os.environ.get("TOL", "1e-3")); FD_EPS = float(os.environ.get("FD_EPS", "1e-2"))
ADAM = int(os.environ.get("ADAM", "30")); NEWTON = int(os.environ.get("NEWTON", "60"))
HESS_EVERY = int(os.environ.get("HESS_EVERY", "5")); MU = 1e-2; TRUST = 2.0
TH_LO, TH_HI = log2_rate_bounds(MIN_RATE, MAX_RATE)
R = dict(meta=dict(dataset=DATASET, families=_FAM, pi=PI, min_rate=MIN_RATE, max_rate=MAX_RATE,
                   theta_bounds=[TH_LO, TH_HI], tol=TOL))
print(f"=== genewise BOUNDED {DATASET} fam={_FAM} pi={PI} rate in [{MIN_RATE},{MAX_RATE}] "
      f"-> theta in [{TH_LO:.2f},{TH_HI:.2f}] ===", flush=True)


def sopts(pi):
    return SolverOptions(**{**_CV_SO, "pi_iters": pi, "neumann_terms": pi})


def lg(m, th):
    lv, g, _ = m.genewise_loss_vector_and_grad(theta=th, need_grad=True)
    return lv.to(DT), g.to(DT)


def proj_gmax(th, g):
    """Per-family inf-norm of the PROJECTED gradient (KKT residual under the box)."""
    gp = project_rate_gradient_(th, g.clone(), min_rate=MIN_RATE, max_rate=MAX_RATE)
    return gp.abs().amax(dim=1)


def clamp_(th):
    clamp_log_rate_(th, min_rate=MIN_RATE, max_rate=MAX_RATE); return th


t0 = time.perf_counter()
m = GeneReconModel(str(DATASETS[DATASET]["species_tree"]), [str(x) for x in DATASETS[DATASET]["families"](N_FAM)],
                   mode="genewise", device=DEV, solver_options=sopts(PI), clade_budget=80000)
m.receiver_weights.requires_grad_(False)
F = int(m.theta.shape[0]); S = int(m.species_helpers["S"])
print(f"[build] F={F} S={S} batches={len(m.batch_statics)} ({time.perf_counter()-t0:.1f}s)", flush=True)

WARM = os.environ.get("WARM_THETA")
if WARM:
    theta = torch.load(WARM, map_location=DEV, weights_only=False)["theta"].to(DEV).to(DT).reshape(F, 3)
else:
    theta = torch.zeros(F, 3, device=DEV, dtype=DT)
clamp_(theta)                                            # start feasible

# ---- Adam warmup (clamped) ----------------------------------------------------------------------
t1 = time.perf_counter()
leaf = theta.clone().requires_grad_(True); adam = torch.optim.Adam([leaf], lr=0.05)
for it in range(ADAM):
    _, g = lg(m, leaf.detach()); leaf.grad = g; adam.step()
    with torch.no_grad():
        clamp_(leaf)
    if it % 10 == 0 or it == ADAM - 1:
        gp = proj_gmax(leaf.detach(), lg(m, leaf.detach())[1])
        print(f"  [adam {it:3d}] |Pg|max={float(gp.max()):.3e} conv={int((gp<TOL).sum())}/{F} ({time.perf_counter()-t1:.1f}s)", flush=True)
theta = leaf.detach().clone()

# ---- projected trust-region Newton (reused Hessian, clamp each step) -----------------------------
Hd = None
for it in range(NEWTON):
    lv, g = lg(m, theta)
    if it % HESS_EVERY == 0:
        H = torch.zeros(F, 3, 3, device=DEV, dtype=DT)
        for j in range(3):
            tp = theta.clone(); tp[:, j] += FD_EPS; _, gpv = lg(m, tp)
            tm = theta.clone(); tm[:, j] -= FD_EPS; _, gmv = lg(m, tm)
            H[:, :, j] = (gpv - gmv) / (2 * FD_EPS)
        H = 0.5 * (H + H.transpose(1, 2)); e, V = torch.linalg.eigh(H)
        Hd = V @ torch.diag_embed(e.clamp(min=MU)) @ V.transpose(1, 2)
    # ACTIVE-SET reduced-Hessian Newton: solve only on FREE coords (a coord is KKT-fixed if at a bound
    # with the gradient pushing further out). Without this, a clamped coord coupled to a free one
    # (D-L collinearity) corrupts the free-coord step -> diverges uphill at the bound.
    fixed = ((theta >= TH_HI - 1e-6) & (g < 0)) | ((theta <= TH_LO + 1e-6) & (g > 0))
    free = (~fixed).float()
    g_red = g * free
    Hred = Hd * free.unsqueeze(1) * free.unsqueeze(2) + torch.diag_embed(1.0 - free)
    delta = -torch.linalg.solve(Hred, g_red.unsqueeze(-1)).squeeze(-1)
    dn = delta.norm(dim=1, keepdim=True); delta = delta * (TRUST / dn.clamp(min=TRUST))
    theta = clamp_(theta + delta)                       # projected step
    gp = proj_gmax(theta, g)
    if it % 5 == 0 or it == NEWTON - 1:
        nb = int(((theta <= TH_LO + 1e-6) | (theta >= TH_HI - 1e-6)).any(dim=1).sum())
        print(f"  [newton {it:3d}] sumNLL={float(lv.sum()):.1f} |Pg|max={float(gp.max()):.3e} |Pg|med={float(gp.median()):.3e} "
              f"conv={int((gp<TOL).sum())}/{F} bound-active={nb} ({time.perf_counter()-t1:.1f}s)", flush=True)
opt_s = time.perf_counter() - t1

# ---- cert: FD 3x3 + projected |g| + bound-active --------------------------------------------------
_, g = lg(m, theta); gp = proj_gmax(theta, g)
H = torch.zeros(F, 3, 3, device=DEV, dtype=DT)
for j in range(3):
    tp = theta.clone(); tp[:, j] += FD_EPS; _, gpv = lg(m, tp)
    tm = theta.clone(); tm[:, j] -= FD_EPS; _, gmv = lg(m, tm)
    H[:, :, j] = (gpv - gmv) / (2 * FD_EPS)
H = 0.5 * (H + H.transpose(1, 2)); lam_min = torch.linalg.eigvalsh(H)[:, 0]
at_lo = (theta <= TH_LO + 1e-6); at_hi = (theta >= TH_HI - 1e-6)
bound_active = (at_lo | at_hi).any(dim=1)
conv = gp < TOL; pd = lam_min > TOL
total = time.perf_counter() - t0
print(f"\n[cert pi={PI}] FD 3x3 (eps={FD_EPS}) + projected-gradient KKT:", flush=True)
print(f"  CONVERGED (|Pg|<{TOL}) = {int(conv.sum())}/{F}   |Pg|max={float(gp.max()):.3e}", flush=True)
print(f"  of those: interior PD (lam_min>{TOL}, no active bound) = {int((conv & pd & ~bound_active).sum())}", flush=True)
print(f"  bound-active families (a rate pinned at {MIN_RATE} or {MAX_RATE}) = {int(bound_active.sum())}  "
      f"[at max={int(at_hi.any(dim=1).sum())}, at min={int(at_lo.any(dim=1).sum())}]", flush=True)
print(f"  rate range now: theta in [{float(theta.min()):.2f},{float(theta.max()):.2f}] "
      f"-> rate in [{2**float(theta.min()):.2e}, {2**float(theta.max()):.2f}]", flush=True)
print(f"  TOTAL = {total:.0f}s  (opt {opt_s:.0f}s)", flush=True)
R.update(dict(F=F, total_s=total, opt_s=opt_s, n_conv=int(conv.sum()), n_interior_pd=int((conv & pd & ~bound_active).sum()),
              n_bound_active=int(bound_active.sum()), n_at_max=int(at_hi.any(dim=1).sum()), n_at_min=int(at_lo.any(dim=1).sum()),
              pg_max=float(gp.max())))
OUT = os.environ.get("OUT_JSON")
if OUT:
    torch.save(dict(theta=theta.cpu(), lam_min=lam_min.cpu(), pg=gp.cpu(), bound_active=bound_active.cpu()),
               OUT.replace(".json", "_theta.pt"))
    with open(OUT, "w") as fh:
        json.dump(R, fh, indent=2, default=lambda o: float(o) if hasattr(o, "__float__") else str(o))
    print(f"  saved -> {OUT}", flush=True)
