"""Diagnose the genewise families that did NOT converge / certified indefinite.

Loads a converged-ish genewise theta + its per-family (|g|, lam_min) certified at some pi, isolates the
PROBLEM families (|g|>=TOL OR lam_min<-TOL), then for that small subset:
  1. characterize them (|g|, |theta|, lam_min, the DTL rates) -- boundary-pinned? extreme rates?
  2. PI SWEEP (64->128->256, neumann too) at the CURRENT theta: does |g| drop as pi rises? (truncation
     test -- if |g| falls with pi, the solver was under-resolved; if it floors, it's not truncation.)
  3. FD-NEWTON the subset HARD at the highest pi (trust-region, reused 3x3 Hessian, no line search) to
     drive |g|->0 as far as it goes.
  4. RE-CERT at high pi with two FD eps (1e-2 AND 1e-3) -> distinguish real indefinite curvature from
     FD noise on near-singular blocks.
  5. CLASSIFY each problem family:
       TRUNCATION  -> converged+PD only after raising pi (the solver was the problem)
       NEWTON_FIX  -> converged+PD after the high-pi Newton polish (optimizer hadn't finished)
       NON_IDENT   -> |g| floors >TOL and/or 3x3 near-singular and |theta| large (boundary; data can't fix)
       INDEFINITE  -> negative 3x3 eig persists at high pi + small eps (a genuine non-minimum -- investigate)

Env: DATASET=hogenom  THETA=<..._theta.pt>  TOL=1e-3  PIS=64,128,256  NEWTON_PI=256  NEWTON_STEPS=40
     HESS_EVERY=5  OUT=<..._diag.pt>
Run: GPUREC_PREPROCESS_PATH=... PYTHONPATH=<wt> python -u .../diagnose_genewise_nonconv.py
"""
from __future__ import annotations
import os, sys, time
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_cv import DATASETS, _CV_SO
from gpurec import GeneReconModel, SolverOptions

DEV = "cuda"; DT = torch.float32
DATASET = os.environ.get("DATASET", "hogenom")
TOL = float(os.environ.get("TOL", "1e-3"))
PIS = [int(x) for x in os.environ.get("PIS", "64,128,256").split(",")]
NEWTON_PI = int(os.environ.get("NEWTON_PI", "256")); NEWTON_STEPS = int(os.environ.get("NEWTON_STEPS", "40"))
HESS_EVERY = int(os.environ.get("HESS_EVERY", "5")); MU = 1e-2; TRUST = 2.0


def sopts(pi):
    return SolverOptions(**{**_CV_SO, "pi_iters": pi, "neumann_terms": pi})


def build(paths, pi):
    m = GeneReconModel(str(DATASETS[DATASET]["species_tree"]), [str(x) for x in paths],
                       mode="genewise", device=DEV, solver_options=sopts(pi), clade_budget=80000)
    m.receiver_weights.requires_grad_(False)
    return m


def lg(m, th):
    lv, g, _ = m.genewise_loss_vector_and_grad(theta=th, need_grad=True)
    return lv.to(DT), g.to(DT)


def gmax(g):
    return g.abs().amax(dim=1)


def fd_hess(m, th, F, eps):
    H = torch.zeros(F, 3, 3, device=DEV, dtype=DT)
    for j in range(3):
        tp = th.clone(); tp[:, j] += eps; _, gp = lg(m, tp)
        tm = th.clone(); tm[:, j] -= eps; _, gm = lg(m, tm)
        H[:, :, j] = (gp - gm) / (2 * eps)
    return 0.5 * (H + H.transpose(1, 2))


# ---- load the full solution + its per-family cert ------------------------------------------------
THETA = os.environ["THETA"]
d = torch.load(THETA, map_location=DEV, weights_only=False)
theta_full = d["theta"].to(DEV).to(DT); gpf0 = d["gpf"].to(DEV); lmin0 = d["lam_min"].to(DEV)
F_all = theta_full.shape[0]
fam_paths = DATASETS[DATASET]["families"](None)
assert len(fam_paths) == F_all, f"{len(fam_paths)} paths vs {F_all} theta rows"

unconv = gpf0 >= TOL; indef = lmin0 < -TOL
problem = (unconv | indef).nonzero(as_tuple=True)[0]
print(f"=== diagnose {DATASET}: {F_all} families, {int(unconv.sum())} unconverged + {int(indef.sum())} indefinite "
      f"-> {problem.numel()} problem families ===", flush=True)
absmax = theta_full.abs().amax(dim=1)
print(f"  problem-family |theta|: <5 (moderate)={int((absmax[problem]<5).sum())}  "
      f">=5 (boundary)={int((absmax[problem]>=5).sum())}  >=10={int((absmax[problem]>=10).sum())}", flush=True)
print(f"  problem-family |g|@cert: med={float(gpf0[problem].median()):.3e} max={float(gpf0[problem].max()):.3e}", flush=True)

sub_paths = [fam_paths[i] for i in problem.tolist()]
theta = theta_full.index_select(0, problem).clone()
Fp = problem.numel()

# ---- PI SWEEP at the CURRENT theta (truncation test) --------------------------------------------
print("\n[1] PI sweep at the current theta (does |g| fall as pi rises? -> truncation):", flush=True)
for pi in PIS:
    m = build(sub_paths, pi)
    _, g = lg(m, theta); gp = gmax(g)
    print(f"  pi={pi:4d}: |g|med={float(gp.median()):.3e} |g|max={float(gp.max()):.3e}  "
          f"conv={int((gp<TOL).sum())}/{Fp}", flush=True)
    del m; torch.cuda.empty_cache()

# ---- FD-NEWTON the subset HARD at NEWTON_PI ------------------------------------------------------
print(f"\n[2] FD-Newton the {Fp} problem families at pi={NEWTON_PI} (trust-region, no line search):", flush=True)
m = build(sub_paths, NEWTON_PI)
leaf = theta.clone(); Hd = None; t0 = time.perf_counter()
for it in range(NEWTON_STEPS):
    lv, g = lg(m, leaf)
    if it % HESS_EVERY == 0:
        Hm = fd_hess(m, leaf, Fp, 1e-2)
        e, V = torch.linalg.eigh(Hm); Hd = V @ torch.diag_embed(e.clamp(min=MU)) @ V.transpose(1, 2)
    delta = -torch.linalg.solve(Hd, g.unsqueeze(-1)).squeeze(-1)
    dn = delta.norm(dim=1, keepdim=True); delta = delta * (TRUST / dn.clamp(min=TRUST))
    leaf = leaf + delta
    gp = gmax(g)
    if it % 5 == 0 or it == NEWTON_STEPS - 1:
        print(f"  [newton {it:3d}] |g|med={float(gp.median()):.3e} |g|max={float(gp.max()):.3e} "
              f"conv={int((gp<TOL).sum())}/{Fp} ({time.perf_counter()-t0:.0f}s)", flush=True)
theta = leaf.detach()
_, g = lg(m, theta); gp = gmax(g)

# ---- RE-CERT at NEWTON_PI with two FD eps (real indefinite vs FD noise) --------------------------
print(f"\n[3] re-cert at pi={NEWTON_PI}, eps=1e-2 and 1e-3:", flush=True)
res = {}
for eps in (1e-2, 1e-3):
    H = fd_hess(m, theta, Fp, eps); lam = torch.linalg.eigvalsh(H)
    res[eps] = lam[:, 0]
    print(f"  eps={eps:.0e}: lam_min min={float(lam[:,0].min()):+.3e}  "
          f"PD={int((lam[:,0]>TOL).sum())}  near-sing={int((lam[:,0].abs()<=TOL).sum())}  "
          f"indef={int((lam[:,0]<-TOL).sum())}", flush=True)

# ---- CLASSIFY -----------------------------------------------------------------------------------
lam_min = res[1e-3]; absmax = theta.abs().amax(dim=1)
conv = gp < TOL; pd = lam_min > TOL; near = lam_min.abs() <= TOL; ind = lam_min < -TOL
# truncation: was unconverged at the original cert pi but converges+PD now at high pi
trunc_fixed = conv & pd
non_ident = (~conv | near) & (absmax >= 5)            # |g| floors / near-singular AND boundary-pinned
real_indef = ind & (absmax < 5)                       # negative curvature, not at the boundary
print(f"\n[4] CLASSIFICATION of the {Fp} problem families (after high-pi Newton):", flush=True)
print(f"  CONVERGED+PD now (truncation/optimizer was the issue) : {int(trunc_fixed.sum())}", flush=True)
print(f"  NON-IDENTIFIABLE (|g| floors / near-singular, |theta|>=5): {int(non_ident.sum())}", flush=True)
print(f"  INDEFINITE persists (|theta|<5, real neg curvature)      : {int(real_indef.sum())}", flush=True)
print(f"  still |g|>={TOL}: {int((~conv).sum())}  (|g|max={float(gp.max()):.3e})", flush=True)
# show the worst few still-unconverged
worst = (~conv).nonzero(as_tuple=True)[0]
if worst.numel():
    order = gp[worst].argsort(descending=True)[:8]
    print("  worst still-unconverged (sub-idx: |g|, |theta|max, lam_min, rates D/T/L softmax-logits):", flush=True)
    for k in order.tolist():
        i = int(worst[k]); th = theta[i]
        print(f"    fam {problem[i].item():4d}: |g|={float(gp[i]):.2e} |th|max={float(absmax[i]):.1f} "
              f"lam_min={float(lam_min[i]):+.2e}  theta={[round(float(x),2) for x in th]}", flush=True)

OUT = os.environ.get("OUT")
if OUT:
    torch.save(dict(problem_idx=problem.cpu(), theta_polished=theta.cpu(), gp=gp.cpu(),
                    lam_min_eps1e3=res[1e-3].cpu(), lam_min_eps1e2=res[1e-2].cpu()), OUT)
    print(f"\n  saved -> {OUT}", flush=True)
