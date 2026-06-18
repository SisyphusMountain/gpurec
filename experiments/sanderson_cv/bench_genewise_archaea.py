"""Genewise archaea convergence benchmark + per-family 3x3 FD-Hessian PD certificate (4090).

GENEWISE mode: theta is [F, 3] -- each gene family has its OWN 3 rate-logits (D/T/L vs speciation),
and a family's NLL depends ONLY on its own row. So the full Hessian is BLOCK-DIAGONAL: F independent
3x3 blocks. Convergence is therefore PER-FAMILY and certifiable cheaply WITHOUT an HVP: finite-difference
the analytic per-family gradient (6 grad evals total, columns 0/1/2 perturbed +/- for ALL families at
once) -> F 3x3 Hessians -> batched eigh -> per-family lam_min. A family is a certified local min iff its
3 eigenvalues are all > 0 (and |g| small).

Phases (all timed):
  0 build genewise model + wave/static profile + GPU mem.
  1 optimize theta[F,3] from theta=0: Adam warmup (basin entry) -> BatchedLBFGS (per-family, strong-Wolfe)
    to per-family |g|inf < TOL. Per-step: max/median per-family |g|, #converged, wall.
  2 certify: FD per-family 3x3 Hessian (central diff of analytic per-family grad) -> batched eigh.
    Report PD / indefinite / near-singular counts, lam_min distribution, and which families are
    boundary-pinned (|theta| large => non-identifiable, the genewise analogue of the specieswise ridge).

Env: FAMILIES=256|all  TOL=1e-3  ADAM=80 ADAM_LR=0.05  LBFGS=200  MAXCOR=20
     FD_EPS=1e-2  DTYPE=float32|float64  SOLVER=pi64  OUT_JSON=...  SCHEDULE=  (optional pi ramp)
Run (worktree, 4090):
  GPUREC_PREPROCESS_PATH=<wt>/crates/gpurec-preprocess/target/release/libgpurec_preprocess.so \
  PYTHONPATH=<wt> python -u experiments/sanderson_cv/bench_genewise_archaea.py
"""
from __future__ import annotations
import os, sys, time, json
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_cv import DATASETS, _CV_SO
from gpurec import GeneReconModel, SolverOptions
from gpurec.batched_lbfgs import BatchedLBFGS

DEV = "cuda"
DATASET = os.environ.get("DATASET", "archaea")
_FAM = os.environ.get("FAMILIES", "256")
N_FAM = None if _FAM in ("all", "0", "") else int(_FAM)
DT = torch.float64 if os.environ.get("DTYPE", "float32") == "float64" else torch.float32
TOL = float(os.environ.get("TOL", "1e-3"))
FD_EPS = float(os.environ.get("FD_EPS", "1e-2"))
GB, MB = 1024 ** 3, 1024 ** 2
R = dict(meta=dict(dataset=DATASET, families=_FAM, mode="genewise", dtype=str(DT), tol=TOL, fd_eps=FD_EPS))


def mem(tag):
    torch.cuda.synchronize()
    free_b, tot_b = torch.cuda.mem_get_info()
    d = dict(peak_alloc_mb=torch.cuda.max_memory_allocated() / MB,
             peak_reserved_mb=torch.cuda.max_memory_reserved() / MB, free_gb=free_b / GB)
    print(f"  [mem {tag}] peak_alloc={d['peak_alloc_mb']:.0f}MB peak_reserved={d['peak_reserved_mb']:.0f}MB "
          f"free={d['free_gb']:.2f}GB", flush=True)
    return d


# ----------------------------------------------------------------- PHASE 0: build ------------
print(f"=== bench_genewise dataset={DATASET} families={_FAM} dtype={DT} ===", flush=True)
torch.cuda.reset_peak_memory_stats()
t0 = time.perf_counter()
# pi_iters/neumann_terms override: the solver's inner self-loop count dominates per-iteration cost.
# PI=16 is ~4x cheaper/iter than the _CV_SO default 64 (lever: cheap solver for the trajectory). The
# tradeoff is gradient bias from truncation -- re-check |g| and the FD Hessian honestly at this PI.
_PI = os.environ.get("PI")
_so_kw = dict(_CV_SO)
if _PI:
    _so_kw["pi_iters"] = int(_PI); _so_kw["neumann_terms"] = int(_PI)
so = SolverOptions(**_so_kw); so.validate()
fam_paths = DATASETS[DATASET]["families"](N_FAM)
m = GeneReconModel(str(DATASETS[DATASET]["species_tree"]), [str(x) for x in fam_paths],
                   mode="genewise", device=DEV, solver_options=so,
                   clade_budget=int(os.environ.get("CLADE_BUDGET", "80000")))
m.receiver_weights.requires_grad_(False)
F = int(m.theta.shape[0]); S = int(m.species_helpers["S"])
torch.cuda.synchronize()
build_s = time.perf_counter() - t0
# wave/static profile
itemsize = torch.empty(0, dtype=DT).element_size()
nbatch = len(m.batch_statics); pi_iters = int(m.batch_statics[0].solver_options.pi_iters)
tot_C = sum(int(st.wave_layout["leaf_species_index"].numel()) for st in m.batch_statics)
max_C = max(int(st.wave_layout["leaf_species_index"].numel()) for st in m.batch_statics)
fam_per_batch = [len(st.family_indices) for st in m.batch_statics]
print(f"[0 build] genewise F={F} families  S={S}  p=3F={3*F}  batches={nbatch}  build={build_s:.1f}s", flush=True)
print(f"  [waves] clades total={tot_C} max_batch={max_C}  pi_iters={pi_iters}  "
      f"families/batch: {min(fam_per_batch)}..{max(fam_per_batch)}  peak[C,S]buf={2*max_C*S*itemsize/MB:.1f}MB", flush=True)
R["build"] = dict(build_s=build_s, F=F, S=S, p=3 * F, batches=nbatch, total_clades=tot_C, pi_iters=pi_iters,
                  mem=mem("after build"))

theta = torch.zeros(F, 3, device=DEV, dtype=DT)   # theta=0 -> uniform DTL/spec softmax (neutral init)
WARM = os.environ.get("WARM_THETA")               # load a converged theta (e.g. a pi=16 solution) to
if WARM:                                          # re-evaluate |g| + FD cert at THIS pi (stiff-family check)
    theta = torch.load(WARM, map_location=DEV, weights_only=False)["theta"].to(DEV).to(DT).reshape(F, 3)
    print(f"  WARM-START theta from {WARM}  (set ADAM=0 STEPS=0 to cert only)", flush=True)


def loss_grad(th):
    """Per-family NLL[F] and per-family grad[F,3] (block-independent), at the converged solver."""
    lv, g, _ = m.genewise_loss_vector_and_grad(theta=th, need_grad=True)
    return lv.to(DT), g.to(DT)


def gmax_per_family(g):
    return g.abs().amax(dim=1)   # [F] per-family |g|inf


# ----------------------------------------------------------------- PHASE 1: optimize ---------
# Genewise families are independent 3-param problems. Rprop (sign-based, NO line search) is the proven
# fast genewise optimizer (archaea_experiments.md); BatchedLBFGS+strong_wolfe is ~25x slower here because
# the BATCHED line search runs until the SLOWEST of F families satisfies Wolfe (stragglers dominate).
# One grad eval per iter (we log from the same grad we step with -- no redundant forward).
OPT = os.environ.get("OPT", "rprop")          # rprop | adam | lbfgs_armijo
ADAM = int(os.environ.get("ADAM", "40")); STEPS = int(os.environ.get("STEPS", "250"))
ADAM_LR = float(os.environ.get("ADAM_LR", "0.05")); RPROP_LR = float(os.environ.get("RPROP_LR", "0.02"))
PLATEAU = int(os.environ.get("PLATEAU", "30"))   # early-stop: |g|max no better for this many iters
torch.cuda.reset_peak_memory_stats()
opt_rows = []
t1 = time.perf_counter()
leaf = theta.clone().requires_grad_(True)

# short Adam basin-entry warmup (cheap, robust), then the configured endgame optimizer
adam = torch.optim.Adam([leaf], lr=ADAM_LR)
for it in range(ADAM):
    lv, g = loss_grad(leaf.detach()); leaf.grad = g; adam.step()
    if it % 20 == 0 or it == ADAM - 1:
        gpf = gmax_per_family(g)
        print(f"  [adam {it:3d}] sumNLL={float(lv.sum()):.3f} |g|max={float(gpf.max()):.3e} "
              f"conv={int((gpf<TOL).sum())}/{F} ({time.perf_counter()-t1:.1f}s)", flush=True)

if OPT == "rprop":
    endopt = torch.optim.Rprop([leaf], lr=RPROP_LR, etas=(0.5, 1.2), step_sizes=(1e-6, 5.0))
elif OPT == "adam":
    endopt = torch.optim.Adam([leaf], lr=ADAM_LR)
else:
    endopt = None  # lbfgs_armijo handled below

best_gmax = float("inf"); since_best = 0; n_end = 0
if OPT == "newton":
    # Batched per-family damped Newton. The Hessian is block-diagonal 3x3, so this is F independent
    # 3x3 solves. Re-form the FD Hessian every HESS_EVERY steps (6 grad evals) and reuse it; eigenvalue-
    # FLOOR it to mu (keeps the local model PD so near-singular/indefinite boundary families take a
    # bounded step instead of exploding); per-family trust clamp + per-family backtracking so no family's
    # NLL ever increases (lever: don't let an aggressive step wreck the likelihood).
    HESS_EVERY = int(os.environ.get("HESS_EVERY", "3"))
    MU = float(os.environ.get("NEWTON_FLOOR", "1e-2"))
    TRUST = float(os.environ.get("NEWTON_TRUST", "2.0"))
    Hd = None
    for it in range(STEPS):
        th = leaf.detach(); lv, g = loss_grad(th); n_end += 1
        if it % HESS_EVERY == 0:
            Hm = torch.zeros(F, 3, 3, device=DEV, dtype=DT)
            for j in range(3):
                tp = th.clone(); tp[:, j] += FD_EPS; _, gp = loss_grad(tp)
                tm = th.clone(); tm[:, j] -= FD_EPS; _, gm = loss_grad(tm)
                Hm[:, :, j] = (gp - gm) / (2 * FD_EPS)
            Hm = 0.5 * (Hm + Hm.transpose(1, 2))
            e, V = torch.linalg.eigh(Hm)
            Hd = V @ torch.diag_embed(e.clamp(min=MU)) @ V.transpose(1, 2)   # PD-floored local model
        delta = -torch.linalg.solve(Hd, g.unsqueeze(-1)).squeeze(-1)         # [F,3] per-family Newton dir
        dn = delta.norm(dim=1, keepdim=True)
        delta = delta * (TRUST / dn.clamp(min=TRUST))                        # per-family trust clamp
        alpha = torch.ones(F, 1, device=DEV, dtype=DT)
        for _ in range(8):                                                   # per-family backtracking
            worse = m.genewise_loss_vector(theta=th + alpha * delta).to(DT) > lv + 1e-9
            if not bool(worse.any()):
                break
            alpha = torch.where(worse.unsqueeze(1), alpha * 0.5, alpha)
        with torch.no_grad():
            leaf.copy_(th + alpha * delta)
        gpf = gmax_per_family(g); gmax = float(gpf.max()); nconv = int((gpf < TOL).sum())
        opt_rows.append(dict(it=it, sumNLL=float(lv.sum()), gmax=gmax, gmed=float(gpf.median()),
                             nconv=nconv, t=time.perf_counter() - t1))
        if it % 2 == 0 or it == STEPS - 1 or nconv == F:
            print(f"  [newton {it:3d}] sumNLL={float(lv.sum()):.3f} |g|max={gmax:.3e} |g|med={float(gpf.median()):.3e} "
                  f"conv={nconv}/{F} ({time.perf_counter()-t1:.1f}s)", flush=True)
        if nconv == F:
            print(f"  [newton] ALL {F} families |g|inf<{TOL} at iter {it}", flush=True); break
        if gmax < best_gmax - 1e-9:
            best_gmax = gmax; since_best = 0
        else:
            since_best += 1
            if since_best >= PLATEAU:
                print(f"  [newton] |g|max plateaued ({gmax:.3e}) -> stop (boundary/non-identifiable)", flush=True); break
elif endopt is not None:
    for it in range(STEPS):
        lv, g = loss_grad(leaf.detach()); leaf.grad = g; endopt.step(); n_end += 1
        gpf = gmax_per_family(g); gmax = float(gpf.max()); nconv = int((gpf < TOL).sum())
        opt_rows.append(dict(it=it, sumNLL=float(lv.sum()), gmax=gmax, gmed=float(gpf.median()),
                             nconv=nconv, t=time.perf_counter() - t1))
        if it % 20 == 0 or it == STEPS - 1 or nconv == F:
            print(f"  [{OPT} {it:3d}] sumNLL={float(lv.sum()):.3f} |g|max={gmax:.3e} |g|med={float(gpf.median()):.3e} "
                  f"conv={nconv}/{F} ({time.perf_counter()-t1:.1f}s)", flush=True)
        if nconv == F:
            print(f"  [{OPT}] ALL {F} families |g|inf<{TOL} at iter {it}", flush=True); break
        if gmax < best_gmax - 1e-9:
            best_gmax = gmax; since_best = 0
        else:
            since_best += 1
            if since_best >= PLATEAU:
                print(f"  [{OPT}] |g|max plateaued ({gmax:.3e}) for {PLATEAU} iters -> stop (stuck families "
                      f"are boundary/non-identifiable)", flush=True); break
else:  # BatchedLBFGS with cheap armijo line search
    blbfgs = BatchedLBFGS([leaf], lr=1.0, max_iter=1, history_size=int(os.environ.get("MAXCOR", "20")),
                          max_ls=10, line_search_fn="armijo", tolerance_grad=1e-12, tolerance_change=1e-16)
    def closure():
        lv, g = loss_grad(leaf.detach()); leaf.grad = g; return lv
    for it in range(STEPS):
        blbfgs.step(closure); n_end += 1
        lv, g = loss_grad(leaf.detach()); gpf = gmax_per_family(g)
        gmax = float(gpf.max()); nconv = int((gpf < TOL).sum())
        opt_rows.append(dict(it=it, sumNLL=float(lv.sum()), gmax=gmax, nconv=nconv, t=time.perf_counter() - t1))
        if it % 10 == 0 or nconv == F:
            print(f"  [lbfgs_armijo {it:3d}] sumNLL={float(lv.sum()):.3f} |g|max={gmax:.3e} conv={nconv}/{F} "
                  f"({time.perf_counter()-t1:.1f}s)", flush=True)
        if nconv == F:
            break

theta = leaf.detach().clone()
opt_s = time.perf_counter() - t1
lv, g = loss_grad(theta); gpf = gmax_per_family(g)
print(f"[1 opt] DONE  opt={OPT} adam={ADAM} end={n_end}  sumNLL={float(lv.sum()):.4f}  |g|max={float(gpf.max()):.3e} "
      f"conv={int((gpf<TOL).sum())}/{F}  opt={opt_s:.1f}s", flush=True)
R["opt"] = dict(opt_s=opt_s, optimizer=OPT, adam=ADAM, end_steps=n_end, sumNLL=float(lv.sum()),
                gmax=float(gpf.max()), nconv=int((gpf < TOL).sum()), rows=opt_rows, mem=mem("after opt"))


# ----------------------------------------------------- PHASE 2: per-family 3x3 FD Hessian cert
print(f"[2 cert] FD per-family 3x3 Hessians (eps={FD_EPS}, 6 grad evals) ...", flush=True)
torch.cuda.reset_peak_memory_stats()
t2 = time.perf_counter()
H = torch.zeros(F, 3, 3, device=DEV, dtype=DT)
for j in range(3):
    tp = theta.clone(); tp[:, j] += FD_EPS
    tm = theta.clone(); tm[:, j] -= FD_EPS
    _, gp = loss_grad(tp)
    _, gm = loss_grad(tm)
    H[:, :, j] = (gp - gm) / (2 * FD_EPS)         # d g / d theta_j  -> column j of each 3x3
H = 0.5 * (H + H.transpose(1, 2))                  # symmetrize each block
asym = float((H - H.transpose(1, 2)).abs().max())  # ~0 after symmetrize; pre-sym diagnostic below
evals = torch.linalg.eigvalsh(H)                   # [F,3] ascending
lam_min = evals[:, 0]; lam_max = evals[:, 2]
cert_s = time.perf_counter() - t2

# classification
absmax_theta = theta.abs().amax(dim=1)             # [F] boundary-pinned if large
pd = (lam_min > TOL)
near_sing = (lam_min.abs() <= TOL)
indef = (lam_min < -TOL)
gpf = gmax_per_family(g)
conv = gpf < TOL
cert_pd = pd & conv                                # converged AND positive-definite
print(f"[2 cert] {cert_s:.1f}s   lam_min: min={float(lam_min.min()):+.4e} med={float(lam_min.median()):+.4e} "
      f"max={float(lam_min.max()):+.4e}", flush=True)
print(f"  families: PD(lam_min>{TOL})={int(pd.sum())}/{F}  near-singular(|lam_min|<={TOL})={int(near_sing.sum())}  "
      f"indefinite(lam_min<-{TOL})={int(indef.sum())}", flush=True)
print(f"  |g|inf<{TOL}: {int(conv.sum())}/{F}   CONVERGED+PD (true per-family minima): {int(cert_pd.sum())}/{F}", flush=True)
print(f"  boundary-pinned |theta|>5: {int((absmax_theta>5).sum())}/{F}  |theta|>10: {int((absmax_theta>10).sum())}  "
      f"(non-identifiable families)", flush=True)
print(f"  cond number (lam_max/lam_min) over PD families: med={float((lam_max[pd]/lam_min[pd]).median()):.1f}", flush=True)
R["cert"] = dict(cert_s=cert_s, n_pd=int(pd.sum()), n_near_sing=int(near_sing.sum()), n_indef=int(indef.sum()),
                 n_conv=int(conv.sum()), n_cert_pd=int(cert_pd.sum()), F=F,
                 lam_min_min=float(lam_min.min()), lam_min_med=float(lam_min.median()),
                 n_boundary5=int((absmax_theta > 5).sum()), n_boundary10=int((absmax_theta > 10).sum()),
                 mem=mem("after cert"))

# ----------------------------------------------------------------- REPORT --------------------
print("\n" + "=" * 76, flush=True)
print(f"GENEWISE BENCHMARK  ({DATASET} families={_FAM}, dtype={DT})", flush=True)
print("=" * 76, flush=True)
print(f"  build={build_s:.0f}s  F={F} p=3F={3*F}  batches={nbatch}", flush=True)
print(f"  optimize={opt_s:.0f}s (opt={OPT} adam={ADAM}+end={n_end})  sumNLL={float(lv.sum()):.2f}  |g|max={float(gpf.max()):.2e}", flush=True)
print(f"  cert={cert_s:.0f}s (FD 3x3 per family)  CONVERGED+PD={int(cert_pd.sum())}/{F}  "
      f"(PD={int(pd.sum())}, near-sing={int(near_sing.sum())}, indef={int(indef.sum())})", flush=True)
print(f"  TOTAL time-to-certified-convergence = {build_s+opt_s+cert_s:.0f}s", flush=True)

OUT = os.environ.get("OUT_JSON")
if OUT:
    torch.save(dict(theta=theta.cpu(), lam_min=lam_min.cpu(), gpf=gpf.cpu(), evals=evals.cpu()),
               OUT.replace(".json", "_theta.pt"))
    with open(OUT, "w") as fh:
        json.dump(R, fh, indent=2, default=lambda o: float(o) if hasattr(o, "__float__") else str(o))
    print(f"\n  saved -> {OUT}", flush=True)
