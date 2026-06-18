"""End-to-end: does adjoint warm-start let us converge to the genewise minimum FASTER on archaea?
Fixed pi=64, bounded active-set Newton from theta=0; three configs, same families/start, loss & |Pg| vs
WALL-CLOCK: (cold neu=64 = accurate reference) vs (cold neu=16 = cheap, maybe biased) vs (warm neu=16)."""
import os, sys, time, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from run_cv import DATASETS, _CV_SO
from gpurec import GeneReconModel, SolverOptions
from gpurec.optimization import clamp_log_rate_, project_rate_gradient_, log2_rate_bounds
DEV = "cuda"; DT = torch.float32; MINR, MAXR = 1e-6, 2.0; TOL = 1e-3; FD = 1e-2; MU = 1e-2; TRUST = 2.0
PI = int(os.environ.get("PI", "64")); STEPS = int(os.environ.get("STEPS", "35")); ADAM = int(os.environ.get("ADAM", "20"))
NF = int(os.environ.get("FAMILIES", "800")); HESS_EVERY = 5
DATASET = os.environ.get("DATASET", "archaea")
LO, HI = log2_rate_bounds(MINR, MAXR)
if DATASET == "hogenom5k":
    fam = [l.strip() for l in open("/tmp/hogenom_5000_paths.txt") if l.strip()][:NF]
    SPTREE = str(DATASETS["hogenom"]["species_tree"])
else:
    fam = DATASETS[DATASET]["families"](None)[:NF]
    SPTREE = str(DATASETS[DATASET]["species_tree"])

CLADE_BUDGET = int(os.environ.get("CLADE_BUDGET", "80000"))
def build(neu):
    m = GeneReconModel(SPTREE, [str(x) for x in fam], mode="genewise", device=DEV,
                       solver_options=SolverOptions(**{**_CV_SO, "pi_iters": PI, "neumann_terms": neu}), clade_budget=CLADE_BUDGET)
    m.receiver_weights.requires_grad_(False); return m

def lg(m, th):
    lv, g, _ = m.genewise_loss_vector_and_grad(theta=th, need_grad=True); return lv.to(DT), g.to(DT)
def pgmax(th, g):
    return project_rate_gradient_(th, g.clone(), min_rate=MINR, max_rate=MAXR).abs().amax(dim=1)

def run(label, neu, warm):
    os.environ.pop("GPUREC_WARM_ADJOINT", None)
    if warm: os.environ["GPUREC_WARM_ADJOINT"] = "1"
    m = build(neu); F = len(fam); t0 = time.perf_counter()
    th = torch.zeros(F, 3, device=DEV, dtype=DT); clamp_log_rate_(th, min_rate=MINR, max_rate=MAXR)
    leaf = th.clone().requires_grad_(True); ad = torch.optim.Adam([leaf], lr=0.05)
    for it in range(ADAM):
        _, g = lg(m, leaf.detach()); leaf.grad = g; ad.step()
        with torch.no_grad(): clamp_log_rate_(leaf, min_rate=MINR, max_rate=MAXR)
    th = leaf.detach().clone(); Hd = None
    print(f"\n--- {label} (pi={PI} neumann={neu} warm={warm}) ---", flush=True)
    for it in range(STEPS):
        torch.cuda.empty_cache()                       # return cached blocks to the driver (scratch-budget headroom)
        lv, g = lg(m, th)
        if it % HESS_EVERY == 0:
            H = torch.zeros(F, 3, 3, device=DEV, dtype=DT)
            for j in range(3):
                tp = th.clone(); tp[:, j] += FD; _, gp = lg(m, tp); tm = th.clone(); tm[:, j] -= FD; _, gm = lg(m, tm)
                H[:, :, j] = (gp - gm) / (2 * FD)
            H = 0.5 * (H + H.transpose(1, 2)); e, V = torch.linalg.eigh(H); Hd = V @ torch.diag_embed(e.clamp(min=MU)) @ V.transpose(1, 2)
        fixed = ((th >= HI - 1e-9) & (g < 0)) | ((th <= LO + 1e-9) & (g > 0)); free = (~fixed).to(DT)
        delta = -torch.linalg.solve(Hd * free.unsqueeze(1) * free.unsqueeze(2) + torch.diag_embed(1.0 - free), (g * free).unsqueeze(-1)).squeeze(-1)
        dn = delta.norm(dim=1, keepdim=True); th = clamp_log_rate_(th + delta * (TRUST / dn.clamp(min=TRUST)), min_rate=MINR, max_rate=MAXR)
        if it % 5 == 0 or it == STEPS - 1:
            pg = pgmax(th, lg(m, th)[1]); torch.cuda.synchronize()
            print(f"  it{it:2d} t={time.perf_counter()-t0:6.1f}s sumNLL={float(lv.sum()):.2f} |Pg|max={float(pg.max()):.2e} conv={int((pg<TOL).sum())}/{F}", flush=True)
    torch.cuda.synchronize(); tot = time.perf_counter() - t0
    lv, g = lg(m, th); pg = pgmax(th, g)
    print(f"  [{label}] TOTAL={tot:.1f}s  sumNLL={float(lv.sum()):.3f}  |Pg|max={float(pg.max()):.2e} conv={int((pg<TOL).sum())}/{F}", flush=True)
    del m; torch.cuda.empty_cache(); return float(lv.sum()), tot

print(f"{DATASET} genewise, {len(fam)} families, pi={PI}, {STEPS} Newton steps + {ADAM} Adam warmup", flush=True)
ONLY = os.environ.get("ONLY", "")            # "warm" -> run only the warm config (cold results already measured)
if ONLY != "warm":
    run("cold neu=64 (reference)", 64, False)
    run("cold neu=16", 16, False)
run("warm neu=16", 16, True)
