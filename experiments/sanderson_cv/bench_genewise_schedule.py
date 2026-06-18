"""Genewise archaea: SCHEDULED fast recipe to certified per-family convergence (4090).

Implements the recipe:
  Phase 1  WARMUP at pi=8 (cheapest solver): Adam basin-entry + reused-Hessian Newton.
  Phase 2  MAIN at pi=16: re-form the Hessian (gradient scale changed) and keep Newton-stepping until
           |g|max plateaus ("apparent end"). Optimizer state is fresh + the Newton metric is rebuilt at
           the switch (the analogue of "reset Adam + LR-ramp" so the first post-switch step is sane).
  Phase 3  REBATCH the UNCONVERGED families only: take the families still |g|>=tol, build a sub-model
           over JUST them at pi=PI_STIFF (>=16 -- the stiff ones genuinely need more pi), and tune only
           their rows to convergence. Cheap because it's a small subset.
  Cert     FD per-family 3x3 Hessian for ALL families at pi=PI_CERT (accurate) -> batched eigh.

Newton step: reused 3x3 Hessian (refresh every HESS_EVERY steps, NOT every iter), eigenvalue-floored to
stay PD, per-family trust clamp + per-family backtracking (never worsen a family's NLL).

Env: FAMILIES=all|N  PI_WARM=8 PI_MAIN=16 PI_STIFF=64 PI_CERT=16  TOL=1e-3  FD_EPS=1e-2  DTYPE=float32
     WARM_ADAM=25 WARM_NEWT=8  MAIN_NEWT=40  STIFF_ADAM=10 STIFF_NEWT=40  HESS_EVERY=5  OUT_JSON=
"""
from __future__ import annotations
import os, sys, time, json
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_cv import DATASETS, _CV_SO
from gpurec import GeneReconModel, SolverOptions

DEV = "cuda"
DATASET = os.environ.get("DATASET", "archaea")
_FAM = os.environ.get("FAMILIES", "all")
N_FAM = None if _FAM in ("all", "0", "") else int(_FAM)
DT = torch.float64 if os.environ.get("DTYPE", "float32") == "float64" else torch.float32
TOL = float(os.environ.get("TOL", "1e-3")); FD_EPS = float(os.environ.get("FD_EPS", "1e-2"))
HESS_EVERY = int(os.environ.get("HESS_EVERY", "5")); MAXBT = int(os.environ.get("MAXBT", "2"))
MU = float(os.environ.get("NEWTON_FLOOR", "1e-2")); TRUST = float(os.environ.get("NEWTON_TRUST", "2.0"))
PI_WARM = int(os.environ.get("PI_WARM", "8")); PI_MAIN = int(os.environ.get("PI_MAIN", "16"))
PI_STIFF = int(os.environ.get("PI_STIFF", "64")); PI_CERT = int(os.environ.get("PI_CERT", "16"))
MB = 1024 ** 2
R = dict(meta=dict(dataset=DATASET, families=_FAM, mode="genewise_scheduled", dtype=str(DT),
                   pi_warm=PI_WARM, pi_main=PI_MAIN, pi_stiff=PI_STIFF, pi_cert=PI_CERT))


def sopts(pi):
    return SolverOptions(**{**_CV_SO, "pi_iters": pi, "neumann_terms": pi})


def build(paths, pi):
    m = GeneReconModel(str(DATASETS[DATASET]["species_tree"]), [str(x) for x in paths],
                       mode="genewise", device=DEV, solver_options=sopts(pi),
                       clade_budget=int(os.environ.get("CLADE_BUDGET", "80000")))
    m.receiver_weights.requires_grad_(False)
    return m


def loss_grad(m, th):
    lv, g, _ = m.genewise_loss_vector_and_grad(theta=th, need_grad=True)
    return lv.to(DT), g.to(DT)


def loss_only(m, th):
    return m.genewise_loss_vector(theta=th).to(DT)


def gmax(g):
    return g.abs().amax(dim=1)


def fd_hessian(m, th, F):
    H = torch.zeros(F, 3, 3, device=DEV, dtype=DT)
    for j in range(3):
        tp = th.clone(); tp[:, j] += FD_EPS; _, gp = loss_grad(m, tp)
        tm = th.clone(); tm[:, j] -= FD_EPS; _, gm = loss_grad(m, tm)
        H[:, :, j] = (gp - gm) / (2 * FD_EPS)
    H = 0.5 * (H + H.transpose(1, 2))
    e, V = torch.linalg.eigh(H)
    return V @ torch.diag_embed(e.clamp(min=MU)) @ V.transpose(1, 2)


def newton_phase(m, theta, F, steps, hess_every, label, t0, adam_warm=0, adam_lr=0.05, plateau=12):
    """Adam (optional) + reused-Hessian Newton. Returns (theta, final_g, n_steps)."""
    leaf = theta.clone()
    if adam_warm:                                   # fresh Adam state + LR ramp at phase entry
        lf = leaf.clone().requires_grad_(True); ad = torch.optim.Adam([lf], lr=adam_lr)
        for it in range(adam_warm):
            lr = adam_lr * min(1.0, (it + 1) / 5.0)  # 5-step LR ramp so the first step is gentle
            ad.param_groups[0]["lr"] = lr
            _, g = loss_grad(m, lf.detach()); lf.grad = g; ad.step()
        leaf = lf.detach().clone()
    Hd = None; best = float("inf"); since = 0; n = 0
    for it in range(steps):
        lv, g = loss_grad(m, leaf); n += 1
        if it % hess_every == 0:
            Hd = fd_hessian(m, leaf, F)
        delta = -torch.linalg.solve(Hd, g.unsqueeze(-1)).squeeze(-1)
        dn = delta.norm(dim=1, keepdim=True); delta = delta * (TRUST / dn.clamp(min=TRUST))
        # Backtracking is capped (MAXBT) because the BATCHED loss eval re-solves ALL families each probe,
        # so an uncapped loop is straggler-dominated. The eigenvalue-floor + trust-clamp already bound the
        # step, so a couple of probes suffice; MAXBT=0 => no line search (pure trust-region Newton, cheap).
        alpha = torch.ones(F, 1, device=DEV, dtype=DT)
        for _ in range(MAXBT):
            worse = loss_only(m, leaf + alpha * delta) > lv + 1e-9
            if not bool(worse.any()):
                break
            alpha = torch.where(worse.unsqueeze(1), alpha * 0.5, alpha)
        leaf = leaf + alpha * delta
        gp = gmax(g); gm = float(gp.max()); nconv = int((gp < TOL).sum())
        if it % 4 == 0 or it == steps - 1 or nconv == F:
            print(f"    [{label} {it:3d}] sumNLL={float(lv.sum()):.2f} |g|max={gm:.3e} |g|med={float(gp.median()):.3e} "
                  f"conv={nconv}/{F} ({time.perf_counter()-t0:.1f}s)", flush=True)
        if nconv == F:
            break
        if gm < best - 1e-9:
            best = gm; since = 0
        else:
            since += 1
            if since >= plateau:
                print(f"    [{label}] plateau ({gm:.3e}) -> stop", flush=True); break
    _, g = loss_grad(m, leaf)
    return leaf, g, n


# ================================================================= run ========================
print(f"=== genewise SCHEDULE dataset={DATASET} families={_FAM} pi:{PI_WARM}->{PI_MAIN} stiff={PI_STIFF} dtype={DT} ===", flush=True)
torch.cuda.reset_peak_memory_stats()
t0 = time.perf_counter()
fam_paths = DATASETS[DATASET]["families"](N_FAM)
m = build(fam_paths, PI_WARM)
F = int(m.theta.shape[0]); S = int(m.species_helpers["S"])
print(f"[build] F={F} S={S} batches={len(m.batch_statics)} ({time.perf_counter()-t0:.1f}s)", flush=True)
theta = torch.zeros(F, 3, device=DEV, dtype=DT)

# Phase 1: warmup at pi=8
print(f"[P1 warmup pi={PI_WARM}]", flush=True)
tP1 = time.perf_counter()
theta, g, n1 = newton_phase(m, theta, F, int(os.environ.get("WARM_NEWT", "8")), HESS_EVERY, "warm", t0,
                            adam_warm=int(os.environ.get("WARM_ADAM", "25")))
P1 = time.perf_counter() - tP1

# Phase 2: main at pi=16 (re-form Hessian at the new scale; no Adam -- Newton from a good point)
print(f"[P2 main pi={PI_MAIN}]  (switch pi, rebuild Newton metric)", flush=True)
m.solver_options = sopts(PI_MAIN)        # setter propagates to all batch_statics instantly
tP2 = time.perf_counter()
theta, g, n2 = newton_phase(m, theta, F, int(os.environ.get("MAIN_NEWT", "40")), HESS_EVERY, "main", t0,
                            plateau=int(os.environ.get("PLATEAU", "12")))
P2 = time.perf_counter() - tP2
gp = gmax(g); nconv_main = int((gp < TOL).sum())
print(f"[P2 done] conv={nconv_main}/{F}  |g|max={float(gp.max()):.3e}", flush=True)

# Phase 3: rebatch -- tune ONLY the unconverged families at pi=PI_STIFF
unconv = (gp >= TOL).nonzero(as_tuple=True)[0]
P3 = 0.0; n3 = 0; sub_built_s = 0.0
print(f"[P3 rebatch] {unconv.numel()} unconverged families -> sub-model at pi={PI_STIFF}", flush=True)
if unconv.numel() > 0:
    tb = time.perf_counter()
    sub_paths = [fam_paths[i] for i in unconv.tolist()]
    m_sub = build(sub_paths, PI_STIFF)
    Fs = int(m_sub.theta.shape[0]); sub_built_s = time.perf_counter() - tb
    print(f"  [P3] sub-model F={Fs} batches={len(m_sub.batch_statics)} built {sub_built_s:.1f}s", flush=True)
    tP3 = time.perf_counter()
    sub_theta = theta.index_select(0, unconv).clone()
    sub_theta, gs, n3 = newton_phase(m_sub, sub_theta, Fs, int(os.environ.get("STIFF_NEWT", "40")), HESS_EVERY,
                                     "stiff", t0, adam_warm=int(os.environ.get("STIFF_ADAM", "10")),
                                     plateau=int(os.environ.get("PLATEAU", "12")))
    theta.index_copy_(0, unconv, sub_theta)        # write the polished rows back
    P3 = time.perf_counter() - tP3
    print(f"  [P3 done] stiff conv={int((gmax(gs)<TOL).sum())}/{Fs} |g|max={float(gmax(gs).max()):.3e}", flush=True)

# Cert: FD 3x3 for ALL families at pi=PI_CERT (accurate)
print(f"[cert pi={PI_CERT}] FD 3x3 per family ...", flush=True)
m.solver_options = sopts(PI_CERT)
tC = time.perf_counter()
_, g = loss_grad(m, theta); gp = gmax(g)
H = torch.zeros(F, 3, 3, device=DEV, dtype=DT)
for j in range(3):
    tp = theta.clone(); tp[:, j] += FD_EPS; _, gpv = loss_grad(m, tp)
    tm = theta.clone(); tm[:, j] -= FD_EPS; _, gmv = loss_grad(m, tm)
    H[:, :, j] = (gpv - gmv) / (2 * FD_EPS)
H = 0.5 * (H + H.transpose(1, 2))
lam = torch.linalg.eigvalsh(H); lam_min = lam[:, 0]
cert_s = time.perf_counter() - tC
conv = gp < TOL; pd = lam_min > TOL; near = lam_min.abs() <= TOL; indef = lam_min < -TOL
absmax = theta.abs().amax(dim=1)
total = time.perf_counter() - t0
print(f"[cert] {cert_s:.1f}s  PD={int(pd.sum())}/{F}  near-sing={int(near.sum())}  indef={int(indef.sum())}  "
      f"conv(|g|<{TOL})={int(conv.sum())}  CONVERGED+PD={int((conv&pd).sum())}", flush=True)
print(f"  boundary |theta|>5={int((absmax>5).sum())}/{F}", flush=True)

print("\n" + "=" * 76, flush=True)
print(f"GENEWISE SCHEDULED  ({DATASET} F={F}, pi {PI_WARM}->{PI_MAIN}, stiff@{PI_STIFF})", flush=True)
print("=" * 76, flush=True)
print(f"  P1 warmup(pi{PI_WARM})={P1:.0f}s  P2 main(pi{PI_MAIN})={P2:.0f}s  P3 rebatch({unconv.numel()} fam @pi{PI_STIFF})={P3:.0f}s "
      f"(+{sub_built_s:.0f}s build)  cert={cert_s:.0f}s", flush=True)
print(f"  CONVERGED+PD={int((conv&pd).sum())}/{F}  (PD={int(pd.sum())}, conv={int(conv.sum())})  "
      f"|g|max={float(gp.max()):.2e}", flush=True)
print(f"  TOTAL time-to-certified-convergence = {total:.0f}s", flush=True)
R.update(dict(F=F, P1_s=P1, P2_s=P2, P3_s=P3, sub_build_s=sub_built_s, cert_s=cert_s, total_s=total,
              n_unconverged=int(unconv.numel()), n_pd=int(pd.sum()), n_conv=int(conv.sum()),
              n_cert_pd=int((conv & pd).sum()), n_near=int(near.sum()), n_indef=int(indef.sum()),
              peak_reserved_mb=torch.cuda.max_memory_reserved() / MB))
OUT = os.environ.get("OUT_JSON")
if OUT:
    torch.save(dict(theta=theta.cpu(), lam_min=lam_min.cpu(), gpf=gp.cpu()), OUT.replace(".json", "_theta.pt"))
    with open(OUT, "w") as fh:
        json.dump(R, fh, indent=2, default=lambda o: float(o) if hasattr(o, "__float__") else str(o))
    print(f"\n  saved -> {OUT}", flush=True)
