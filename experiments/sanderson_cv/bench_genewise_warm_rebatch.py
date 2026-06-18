"""Genewise convergence by WARM-START + CONVERGENCE-BASED rebatching.

One fixed (pi, neumann=NEU_OPT) config with adjoint WARM-START (recovers NEU_CERT-quality gradients at
NEU_OPT cost). Optimize ALL families with bounded active-set trust-region Newton. Every CHECK iters, detect
families whose LIKELIHOOD has plateaued (|loss(it) - loss(it-CHECK)| < CONV_TOL); once more than FRAC of the
ACTIVE batch has plateaued, FREEZE those families (park their theta) and DROP them -> rebuild the model over
only the survivors. The per-step cost shrinks as families finish, so the long tail of hard families runs on a
small batch. Final certificate at NEU_CERT COLD over ALL families (authoritative |Pg| + 3x3 lam_min); this
also flags any family that was dropped PREMATURELY (loss plateaued but |Pg| still > tol).

Why this beats pi-tier escalation here: warm-start already gives the stiff families an accurate gradient at
neu16, so the only adaptivity left worth paying for is shrinking the batch as families converge.

Env: DATASET=hogenom|archaea FAMILIES=all|N PI=64 NEU_OPT=16 NEU_CERT=64 MIN_RATE=1e-6 MAX_RATE=2 TOL=1e-3
     CONV_TOL=1e-2 CHECK=4 FRAC=0.30 ADAM=20 MAXIT=120 HESS_EVERY=5 CLADE_BUDGET=80000 OUT_JSON=

  WT=$(git rev-parse --show-toplevel)
  GPUREC_PREPROCESS_PATH=$WT/crates/gpurec-preprocess/target/release/libgpurec_preprocess.so PYTHONPATH=$WT \
  DATASET=hogenom FAMILIES=all python -u experiments/sanderson_cv/bench_genewise_warm_rebatch.py
"""
from __future__ import annotations
import os, sys, time, json
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from run_cv import DATASETS, _CV_SO
from gpurec import GeneReconModel, SolverOptions
from gpurec.optimization import clamp_log_rate_, project_rate_gradient_, log2_rate_bounds

DEV = "cuda"; DT = torch.float32
DATASET = os.environ.get("DATASET", "hogenom")
_FAM = os.environ.get("FAMILIES", "all"); N_FAM = None if _FAM in ("all", "0", "") else int(_FAM)
PI = int(os.environ.get("PI", "64")); NEU_OPT = int(os.environ.get("NEU_OPT", "16")); NEU_CERT = int(os.environ.get("NEU_CERT", "64"))
MIN_RATE = float(os.environ.get("MIN_RATE", "1e-6")); MAX_RATE = float(os.environ.get("MAX_RATE", "2"))
TOL = float(os.environ.get("TOL", "1e-3")); FD_EPS = 1e-2; MU = 1e-2; TRUST = 2.0
CONV_TOL = float(os.environ.get("CONV_TOL", "1e-2"))   # per-family loss-plateau threshold (DROP_BY=loss only)
CHECK = int(os.environ.get("CHECK", "4")); FRAC = float(os.environ.get("FRAC", "0.30"))
# A family is "converged" (droppable) by its projected GRADIENT |Pg|<TOL (DEFAULT, reliable & free -- g is
# already computed each Newton step) or by LIKELIHOOD plateau |loss(it)-loss(it-CHECK)|<CONV_TOL. Measured:
# loss-plateau drops ~22% of hogenom families PREMATURELY (flat-loss region != minimum); grad does not.
DROP_BY = os.environ.get("DROP_BY", "grad")
# Warm-start cache is ~active_clades*S*4 bytes; on a big initial batch that can exceed GPU memory (full
# hogenom: 12408 fam -> 19.5GB). Gate it: warm ON only once the active batch <= WARM_MAX_FAM. The big
# early batch runs cold (its easy families are NOT stiff, and cold |Pg| is conservatively high so they
# never drop prematurely); warm engages for the small stiff tail, where it matters and the cache is small.
WARM_MAX_FAM = int(os.environ.get("WARM_MAX_FAM", "1000000000"))
# The opt |Pg| uses NEU_OPT (cold neu16 on the big batch), which can be biased BELOW tol -> premature drops
# (measured: 383/12408 on full hogenom). VERIFY_DROP re-checks the converged subset at NEU_CERT cold before
# freezing, so a family is dropped only if it is genuinely converged by the authoritative backward.
VERIFY_DROP = os.environ.get("VERIFY_DROP", "1") != "0"
ADAM = int(os.environ.get("ADAM", "20")); MAXIT = int(os.environ.get("MAXIT", "120")); HESS_EVERY = int(os.environ.get("HESS_EVERY", "5"))
CLADE_BUDGET = int(os.environ.get("CLADE_BUDGET", "80000"))
TH_LO, TH_HI = log2_rate_bounds(MIN_RATE, MAX_RATE)
print(f"=== genewise WARM + CONVERGENCE-REBATCH {DATASET} fam={_FAM}  pi={PI} neu_opt={NEU_OPT}(warm) "
      f"neu_cert={NEU_CERT}  drop>|loss|<{CONV_TOL}/{CHECK}it when >{FRAC*100:.0f}% plateau ===", flush=True)


def sopts(neu): return SolverOptions(**{**_CV_SO, "pi_iters": PI, "neumann_terms": neu})
def build(paths, neu):
    m = GeneReconModel(str(DATASETS[DATASET]["species_tree"]), [str(x) for x in paths],
                       mode="genewise", device=DEV, solver_options=sopts(neu), clade_budget=CLADE_BUDGET)
    m.receiver_weights.requires_grad_(False); return m
def lg(m, th):
    lv, g, _ = m.genewise_loss_vector_and_grad(theta=th, need_grad=True); return lv.to(DT), g.to(DT)
def pgmax(th, g): return project_rate_gradient_(th, g.clone(), min_rate=MIN_RATE, max_rate=MAX_RATE).abs().amax(dim=1)
def clamp_(th): clamp_log_rate_(th, min_rate=MIN_RATE, max_rate=MAX_RATE); return th
def set_warm(n):                                       # enable warm-start only when the active batch fits in memory
    if n <= WARM_MAX_FAM: os.environ["GPUREC_WARM_ADJOINT"] = "1"
    else: os.environ.pop("GPUREC_WARM_ADJOINT", None)


t0 = time.perf_counter()
fam_paths = DATASETS[DATASET]["families"](N_FAM); F_all = len(fam_paths)
theta = torch.zeros(F_all, 3, device=DEV, dtype=DT); clamp_(theta)
active = torch.arange(F_all, device=DEV)
was_dropped = torch.zeros(F_all, dtype=torch.bool, device=DEV)   # dropped mid-run (vs optimized to the end)

set_warm(active.numel())                                        # warm-start the optimization adjoint (gated by size)
m = build([fam_paths[j] for j in active.tolist()], NEU_OPT)
sub = theta.index_select(0, active).clone()
lf = sub.clone().requires_grad_(True); ad = torch.optim.Adam([lf], lr=0.05)
for _ in range(ADAM):                                            # Adam warmup on the full batch
    _, g = lg(m, lf.detach()); lf.grad = g; ad.step()
    with torch.no_grad(): clamp_(lf)
sub = lf.detach().clone()

Hd = None; loss_ref = None; rebatch_log = []
for it in range(MAXIT):
    if active.numel() == 0: break
    lv, g = lg(m, sub)
    if it % CHECK == 0:                                          # convergence-based drop check
        pgm = pgmax(sub, g)
        if DROP_BY == "loss":                                    # likelihood plateau (the proposed proxy; loose)
            conv = ((loss_ref - lv).abs() < CONV_TOL) if (loss_ref is not None and loss_ref.shape[0] == lv.shape[0]) \
                else torch.zeros_like(pgm, dtype=torch.bool)
        else:                                                    # projected gradient |Pg|<TOL (reliable, free)
            conv = pgm < TOL
        frac = float(conv.float().mean())
        print(f"  [it{it:3d}] active={active.numel():5d} converged={frac*100:4.0f}% |Pg|max={float(pgm.max()):.2e} "
              f"t={time.perf_counter()-t0:.0f}s", flush=True)
        do_drop = frac > FRAC and conv.any() and not conv.all()
        if do_drop and VERIFY_DROP:                             # re-verify the converged subset at NEU_CERT cold
            _w = os.environ.pop("GPUREC_WARM_ADJOINT", None)
            m.solver_options = sopts(NEU_CERT)
            conv = conv & (pgmax(sub, lg(m, sub)[1]) < TOL)
            m.solver_options = sopts(NEU_OPT)
            if _w: os.environ["GPUREC_WARM_ADJOINT"] = _w
            do_drop = bool(conv.any()) and not bool(conv.all())
        if do_drop:
            theta.index_copy_(0, active[conv], sub[conv]); was_dropped[active[conv]] = True
            active = active[~conv]; sub = sub[~conv].clone()
            rebatch_log.append(dict(it=it, dropped=int(conv.sum()), remain=int(active.numel())))
            del m; torch.cuda.empty_cache()
            set_warm(active.numel())                             # re-gate warm-start for the (smaller) survivor batch
            m = build([fam_paths[j] for j in active.tolist()], NEU_OPT)   # rebatch survivors (warm cache resets)
            Hd = None; loss_ref = None; continue
        loss_ref = lv.clone()
    if it % HESS_EVERY == 0 or Hd is None or Hd.shape[0] != sub.shape[0]:
        H = torch.zeros(sub.shape[0], 3, 3, device=DEV, dtype=DT)
        for j in range(3):
            tp = sub.clone(); tp[:, j] += FD_EPS; _, gp = lg(m, tp)
            tm = sub.clone(); tm[:, j] -= FD_EPS; _, gm = lg(m, tm)
            H[:, :, j] = (gp - gm) / (2 * FD_EPS)
        H = 0.5 * (H + H.transpose(1, 2)); e, V = torch.linalg.eigh(H)
        Hd = V @ torch.diag_embed(e.clamp(min=MU)) @ V.transpose(1, 2)
    fixed = ((sub >= TH_HI - 1e-6) & (g < 0)) | ((sub <= TH_LO + 1e-6) & (g > 0)); free = (~fixed).float()
    Hred = Hd * free.unsqueeze(1) * free.unsqueeze(2) + torch.diag_embed(1.0 - free)
    delta = -torch.linalg.solve(Hred, (g * free).unsqueeze(-1)).squeeze(-1)
    dn = delta.norm(dim=1, keepdim=True); sub = clamp_(sub + delta * (TRUST / dn.clamp(min=TRUST)))
if active.numel() > 0: theta.index_copy_(0, active, sub)
os.environ.pop("GPUREC_WARM_ADJOINT", None)
opt_s = time.perf_counter() - t0

# ---- final cert at NEU_CERT COLD over ALL families ----------------------------------------------
tc = time.perf_counter()
mfull = build(fam_paths, NEU_CERT)
_, g = lg(mfull, theta); pg = pgmax(theta, g)
H = torch.zeros(F_all, 3, 3, device=DEV, dtype=DT)
for j in range(3):
    tp = theta.clone(); tp[:, j] += FD_EPS; _, gp = lg(mfull, tp)
    tm = theta.clone(); tm[:, j] -= FD_EPS; _, gm = lg(mfull, tm)
    H[:, :, j] = (gp - gm) / (2 * FD_EPS)
H = 0.5 * (H + H.transpose(1, 2)); lam_min = torch.linalg.eigvalsh(H)[:, 0]
at_lo = (theta <= TH_LO + 1e-6); at_hi = (theta >= TH_HI - 1e-6); bound_active = (at_lo | at_hi).any(dim=1)
conv = pg < TOL; pd = lam_min > TOL; total = time.perf_counter() - t0
premature = int((was_dropped & ~conv).sum())
print(f"\n{'='*72}\nWARM + CONVERGENCE-REBATCH  ({DATASET} F={F_all}, pi={PI} neu_opt={NEU_OPT}-warm neu_cert={NEU_CERT})\n{'='*72}", flush=True)
for r in rebatch_log:
    print(f"  rebatch @it{r['it']:3d}: dropped {r['dropped']:5d} -> {r['remain']:5d} remain", flush=True)
print(f"  drops={len(rebatch_log)}  total dropped mid-run={int(was_dropped.sum())}  optimize={opt_s:.0f}s  cert={time.perf_counter()-tc:.0f}s", flush=True)
print(f"  CONVERGED (|Pg|<{TOL}) = {int(conv.sum())}/{F_all}   |Pg|max={float(pg.max()):.2e}", flush=True)
print(f"  interior PD = {int((conv & pd & ~bound_active).sum())}   bound-active = {int(bound_active.sum())}   "
      f"unconverged = {int((~conv).sum())}   (of which dropped-prematurely = {premature})", flush=True)
print(f"  TOTAL = {total:.0f}s", flush=True)
R = dict(dataset=DATASET, F=F_all, pi=PI, neu_opt=NEU_OPT, neu_cert=NEU_CERT, conv_tol=CONV_TOL, frac=FRAC,
         opt_s=opt_s, total_s=total, n_conv=int(conv.sum()), n_interior_pd=int((conv & pd & ~bound_active).sum()),
         n_bound_active=int(bound_active.sum()), n_unconv=int((~conv).sum()), n_dropped=int(was_dropped.sum()),
         n_premature=premature, rebatches=rebatch_log)
OUT = os.environ.get("OUT_JSON")
if OUT:
    torch.save(dict(theta=theta.cpu(), lam_min=lam_min.cpu(), pg=pg.cpu(), was_dropped=was_dropped.cpu()), OUT.replace(".json", "_theta.pt"))
    with open(OUT, "w") as fh: json.dump(R, fh, indent=2, default=lambda o: float(o) if hasattr(o, "__float__") else str(o))
    print(f"  saved -> {OUT}", flush=True)
