"""Genewise convergence by ADAPTIVE pi-tier rebatching (+ bounded rates).

The non-convergence diagnosis showed families fall into difficulty tiers: most converge at low pi; a
minority are TRUNCATION-stiff (their pi=low gradient is biased small -> they look converged but |g| grows
with pi); and some are NON-IDENTIFIABLE (rates run to the boundary). Paying high pi for ALL families is
wasteful; paying low pi misses the stiff ones. So: give each family only as much pi as it needs.

ALGORITHM (adaptive rebatch by hardness):
  bounds: rate in [MIN_RATE, MAX_RATE] -> theta box (runaway families park at the bound, projected |g|->0).
  active = all families, at tier 0 (pi = PIS[0]).
  for each tier i (pi = PIS[i], ascending):
     - build a sub-model over the ACTIVE families at pi=PIS[i]; bounded projected trust-region Newton.
     - converge them; then VERIFY at the NEXT pi (PIS[i+1]): a family GRADUATES only if projected
       |g| < tol at BOTH pi=PIS[i] AND pi=PIS[i+1] (stable -> not truncation-biased). The rest are
       harder than this tier -> they get REBATCHED into tier i+1 (higher pi, smaller batch).
  Families still unconverged after the top tier: classify boundary (non-identifiable) vs hard-interior.
Cost is dominated by tier 0 (all families, cheap pi); each higher tier runs only the shrinking hard set.

Env: DATASET=hogenom|archaea FAMILIES=all|N  PIS=16,32,64,128,256  MIN_RATE=1e-6 MAX_RATE=2  TOL=1e-3
     TIER_NEWTON=30 ADAM=20 HESS_EVERY=5  SEED=0  OUT_JSON=

REPRODUCE (the committed hogenom run -> 1046/1055 converged, 785 interior-PD, 97 bound-active, 9 stragglers
at |Pg|~1e-3; ~12 min on an RTX 4090). From the worktree root, miniforge3 python:

  WT=$(git rev-parse --show-toplevel)
  GPUREC_PREPROCESS_PATH=$WT/crates/gpurec-preprocess/target/release/libgpurec_preprocess.so \
  PYTHONPATH=$WT \
  DATASET=hogenom FAMILIES=all PIS=16,32,64,128,256 MIN_RATE=1e-6 MAX_RATE=2 TOL=1e-3 \
  ADAM=20 TIER_NEWTON=35 HESS_EVERY=5 SEED=0 \
  OUT_JSON=experiments/sanderson_cv/_artifacts/genewise_adaptive/hogenom_1055.json \
  python -u experiments/sanderson_cv/bench_genewise_adaptive.py

DETERMINISM: theta=0 init + (Adam, FD-Hessian, reduced-Newton) are deterministic, BUT the genewise
backward uses atomic accumulation (run-to-run gradient noise ~2e-4), so the per-family counts reproduce
to within a few families, not bit-exactly. The ~9 stragglers sit AT that FD/atomic floor (|Pg|~1-8e-3).
"""
from __future__ import annotations
import os, sys, time, json
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_cv import DATASETS, _CV_SO
from gpurec import GeneReconModel, SolverOptions
from gpurec.optimization import clamp_log_rate_, project_rate_gradient_, log2_rate_bounds

DEV = "cuda"; DT = torch.float32
DATASET = os.environ.get("DATASET", "hogenom")
_FAM = os.environ.get("FAMILIES", "all"); N_FAM = None if _FAM in ("all", "0", "") else int(_FAM)
PIS = [int(x) for x in os.environ.get("PIS", "16,32,64,128,256").split(",")]
MIN_RATE = float(os.environ.get("MIN_RATE", "1e-6")); MAX_RATE = float(os.environ.get("MAX_RATE", "2"))
TOL = float(os.environ.get("TOL", "1e-3")); FD_EPS = float(os.environ.get("FD_EPS", "1e-2"))
ADAM = int(os.environ.get("ADAM", "20")); TIER_NEWTON = int(os.environ.get("TIER_NEWTON", "30"))
HESS_EVERY = int(os.environ.get("HESS_EVERY", "5")); MU = 1e-2; TRUST = 2.0
# Backward (adjoint) neumann is DECOUPLED from pi: the optimization runs cheap NEU_OPT with adjoint
# WARM-START (GPUREC_WARM_ADJOINT, scoped to each tier's newton_bounded), which recovers NEU_CERT-quality
# gradients at NEU_OPT cost (measured on hogenom-5k: |Pg| 15 at neu16-warm vs 12 at neu64, 29 at neu16-cold;
# ~1.37x faster). Certificates (graduation verify + final) run NEU_CERT COLD -> authoritative & no warm cache.
NEU_OPT = int(os.environ.get("NEU_OPT", "16")); NEU_CERT = int(os.environ.get("NEU_CERT", "64"))
CHECK_EVERY = int(os.environ.get("CHECK_EVERY", "3")); PATIENCE = int(os.environ.get("PATIENCE", "3"))
# warm-start cache ~ tier_clades*S*4 bytes; on the big low-pi tier-0 that can exceed GPU memory (full hogenom
# tier-0 = 12408 fam -> ~20GB). Gate it: warm ON only when the tier's batch <= WARM_MAX_FAM. Tier-0's easy
# families converge fine cold at low pi anyway; warm matters for the small stiff high-pi tiers, where it fits.
WARM_MAX_FAM = int(os.environ.get("WARM_MAX_FAM", "1000000000"))
TH_LO, TH_HI = log2_rate_bounds(MIN_RATE, MAX_RATE)
R = dict(meta=dict(dataset=DATASET, families=_FAM, pis=PIS, min_rate=MIN_RATE, max_rate=MAX_RATE, tol=TOL))
print(f"=== genewise ADAPTIVE-REBATCH {DATASET} fam={_FAM}  pi-tiers={PIS}  rate in [{MIN_RATE},{MAX_RATE}] ===", flush=True)


def sopts(pi, neu):
    return SolverOptions(**{**_CV_SO, "pi_iters": pi, "neumann_terms": neu})


def build(paths, pi, neu):
    m = GeneReconModel(str(DATASETS[DATASET]["species_tree"]), [str(x) for x in paths],
                       mode="genewise", device=DEV, solver_options=sopts(pi, neu), clade_budget=80000)
    m.receiver_weights.requires_grad_(False)
    return m


def lg(m, th):
    lv, g, _ = m.genewise_loss_vector_and_grad(theta=th, need_grad=True)
    return lv.to(DT), g.to(DT)


def proj_gmax(m, th):
    _, g = lg(m, th)
    gp = project_rate_gradient_(th, g.clone(), min_rate=MIN_RATE, max_rate=MAX_RATE)
    return gp.abs().amax(dim=1)


def clamp_(th):
    clamp_log_rate_(th, min_rate=MIN_RATE, max_rate=MAX_RATE); return th


def newton_bounded(m, theta, steps, F, t0, adam=0, tag=""):
    """Bounded projected trust-region Newton (clamp each step, no line search). Returns (theta, n_steps)."""
    leaf = theta.clone()
    if adam:
        lf = leaf.clone().requires_grad_(True); ad = torch.optim.Adam([lf], lr=0.05)
        for it in range(adam):
            _, g = lg(m, lf.detach()); lf.grad = g; ad.step()
            with torch.no_grad():
                clamp_(lf)
        leaf = lf.detach().clone()
    Hd = None; best_nconv = -1; since = 0; n_steps = 0
    for it in range(steps):
        lv, g = lg(m, leaf); n_steps += 1
        # convergence check every CHECK_EVERY iters (projected gradient is free -- we already have g).
        # Key on the CONVERGED COUNT, not |Pg|max: a tier's job is to converge the families it CAN at this
        # pi; once no NEW family is crossing tol (count plateaus for PATIENCE checks), the rest are stiff
        # and get escalated -- so stop. (|Pg|max never drops in tier 0 because the to-be-promoted families
        # keep it high, so it's the wrong signal.)
        if it % CHECK_EVERY == 0:
            pgm = project_rate_gradient_(leaf, g.clone(), min_rate=MIN_RATE, max_rate=MAX_RATE).abs().amax(dim=1)
            nconv = int((pgm < TOL).sum())
            print(f"      [{tag} it{it:2d}] conv={nconv}/{F} |Pg|max={float(pgm.max()):.2e}", flush=True)
            if nconv == F:
                break
            if nconv > best_nconv:
                best_nconv = nconv; since = 0
            else:
                since += 1
                if since >= PATIENCE:
                    break
        if it % HESS_EVERY == 0:
            H = torch.zeros(F, 3, 3, device=DEV, dtype=DT)
            for j in range(3):
                tp = leaf.clone(); tp[:, j] += FD_EPS; _, gp = lg(m, tp)
                tm = leaf.clone(); tm[:, j] -= FD_EPS; _, gm = lg(m, tm)
                H[:, :, j] = (gp - gm) / (2 * FD_EPS)
            H = 0.5 * (H + H.transpose(1, 2)); e, V = torch.linalg.eigh(H)
            Hd = V @ torch.diag_embed(e.clamp(min=MU)) @ V.transpose(1, 2)
        # ACTIVE-SET reduced-Hessian Newton: solve only on the FREE coords (a coord is KKT-fixed if it is
        # at a bound with the gradient pushing further out). Without this, a clamped coord coupled to a
        # free one (D-L collinearity) corrupts the free-coord step -> diverges uphill at the bound.
        fixed = ((leaf >= TH_HI - 1e-6) & (g < 0)) | ((leaf <= TH_LO + 1e-6) & (g > 0))
        free = (~fixed).float()
        g_red = g * free
        Hred = Hd * free.unsqueeze(1) * free.unsqueeze(2) + torch.diag_embed(1.0 - free)
        delta = -torch.linalg.solve(Hred, g_red.unsqueeze(-1)).squeeze(-1)
        dn = delta.norm(dim=1, keepdim=True); delta = delta * (TRUST / dn.clamp(min=TRUST))
        leaf = clamp_(leaf + delta)
    return leaf.detach(), n_steps


t0 = time.perf_counter()
fam_paths = DATASETS[DATASET]["families"](N_FAM)
F_all = len(fam_paths)
theta = torch.zeros(F_all, 3, device=DEV, dtype=DT); clamp_(theta)
active = torch.arange(F_all, device=DEV)                         # global indices still being escalated
graduated_tier = torch.full((F_all,), -1, dtype=torch.long, device=DEV)   # which pi-tier each family graduated at
tier_log = []

for i, pi in enumerate(PIS):
    if active.numel() == 0:
        break
    paths = [fam_paths[j] for j in active.tolist()]
    tb = time.perf_counter()
    m = build(paths, pi, NEU_OPT); n = active.numel()
    sub_theta = theta.index_select(0, active).clone()
    if n <= WARM_MAX_FAM:                      # warm-start this tier's adjoint only if the cache fits (else cold)
        os.environ["GPUREC_WARM_ADJOINT"] = "1"
    sub_theta, n_steps = newton_bounded(m, sub_theta, TIER_NEWTON, n, t0, adam=ADAM if i == 0 else 0, tag=f"pi{pi}")
    os.environ.pop("GPUREC_WARM_ADJOINT", None)
    theta.index_copy_(0, active, sub_theta)
    # certify at NEU_CERT COLD (authoritative, no warm cache): reconfigure m's backward, verify next-pi stability
    m.solver_options = sopts(pi, NEU_CERT)
    pg_here = proj_gmax(m, sub_theta)
    if i + 1 < len(PIS):
        m2 = build(paths, PIS[i + 1], NEU_CERT)
        pg_next = proj_gmax(m2, sub_theta)
        ok = (pg_here < TOL) & (pg_next < TOL)
        del m2
    else:
        ok = pg_here < TOL
    grad_idx = active[ok]; graduated_tier[grad_idx] = pi
    promoted = int((~ok).sum())
    dt = time.perf_counter() - tb
    print(f"[tier {i} pi={pi:4d}] {n} families, {n_steps} newton steps (early-stop of {TIER_NEWTON}) -> "
          f"graduated {int(ok.sum())} (stable<= pi{PIS[i+1] if i+1<len(PIS) else pi}), "
          f"promote {promoted}  |Pg|max_here={float(pg_here.max()):.2e}  {dt:.0f}s", flush=True)
    tier_log.append(dict(tier=i, pi=pi, n_in=int(n), newton_steps=n_steps, graduated=int(ok.sum()), promoted=promoted, secs=dt))
    active = active[~ok]
    del m; torch.cuda.empty_cache()

# ---- final cert at the TOP pi (authoritative) ---------------------------------------------------
TOP = PIS[-1]
print(f"\n[final cert pi={TOP}] FD 3x3 + projected gradient over all {F_all} families ...", flush=True)
tc = time.perf_counter()
mfull = build(fam_paths, TOP, NEU_CERT)
pg = proj_gmax(mfull, theta)
H = torch.zeros(F_all, 3, 3, device=DEV, dtype=DT)
for j in range(3):
    tp = theta.clone(); tp[:, j] += FD_EPS; _, gp = lg(mfull, tp)
    tm = theta.clone(); tm[:, j] -= FD_EPS; _, gm = lg(mfull, tm)
    H[:, :, j] = (gp - gm) / (2 * FD_EPS)
H = 0.5 * (H + H.transpose(1, 2)); lam_min = torch.linalg.eigvalsh(H)[:, 0]
at_lo = (theta <= TH_LO + 1e-6); at_hi = (theta >= TH_HI - 1e-6); bound_active = (at_lo | at_hi).any(dim=1)
conv = pg < TOL; pd = lam_min > TOL; total = time.perf_counter() - t0
still = (~conv).nonzero(as_tuple=True)[0]
print(f"[final cert] {time.perf_counter()-tc:.0f}s", flush=True)
print(f"\n{'='*72}\nGENEWISE ADAPTIVE-REBATCH  ({DATASET} F={F_all}, tiers {PIS}, rate in [{MIN_RATE},{MAX_RATE}])\n{'='*72}", flush=True)
for r in tier_log:
    print(f"  tier pi={r['pi']:4d}: in={r['n_in']:5d} graduated={r['graduated']:5d} promoted={r['promoted']:5d} ({r['secs']:.0f}s)", flush=True)
print(f"  --- @pi={TOP} authoritative cert ---", flush=True)
print(f"  CONVERGED (|Pg|<{TOL}) = {int(conv.sum())}/{F_all}   |Pg|max={float(pg.max()):.2e}", flush=True)
print(f"  interior PD = {int((conv & pd & ~bound_active).sum())}   bound-active(rate@2 or 1e-6) = {int(bound_active.sum())}   "
      f"still-unconverged = {int((~conv).sum())}", flush=True)
print(f"  graduated-by-tier: " + "  ".join(f"pi{p}={int((graduated_tier==p).sum())}" for p in PIS), flush=True)
print(f"  rate range: rate in [{2**float(theta.min()):.2e}, {2**float(theta.max()):.2f}]", flush=True)
print(f"  TOTAL = {total:.0f}s", flush=True)
if still.numel():
    order = pg[still].argsort(descending=True)[:6]
    print("  worst still-unconverged (|Pg|, |theta|max, bound?):", flush=True)
    for k in order.tolist():
        gi = int(still[k]); print(f"    fam {gi:5d}: |Pg|={float(pg[gi]):.2e} |th|={float(theta[gi].abs().max()):.1f} "
                                   f"bound={bool(bound_active[gi])}", flush=True)
R.update(dict(F=F_all, total_s=total, tiers=tier_log, n_conv=int(conv.sum()),
              n_interior_pd=int((conv & pd & ~bound_active).sum()), n_bound_active=int(bound_active.sum()),
              n_unconv=int((~conv).sum()), graduated_by_tier={int(p): int((graduated_tier == p).sum()) for p in PIS}))
OUT = os.environ.get("OUT_JSON")
if OUT:
    torch.save(dict(theta=theta.cpu(), lam_min=lam_min.cpu(), pg=pg.cpu(), graduated_tier=graduated_tier.cpu()),
               OUT.replace(".json", "_theta.pt"))
    with open(OUT, "w") as fh:
        json.dump(R, fh, indent=2, default=lambda o: float(o) if hasattr(o, "__float__") else str(o))
    print(f"  saved -> {OUT}", flush=True)
