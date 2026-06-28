"""Cross-validated exposure-sensitivity table for the archaeal transfer weights (GPT round-3 ask).

Upgrades the in-sample genome-size diagnostic (paper Table 2) to a RE-OPTIMIZED, HELD-OUT, MULTI-SPLIT
comparison: for each of SPLITS random 80/20 family splits, fit each recipient model on the 80% train and
score data NLL on the 20% held-out, all inside the box [0.05,2.0] (which reconditions the joint problem,
Sec.4.4). Three recipient models:

  uniform             alpha = 0                                     (null)
  genome-size offset  alpha_s = gamma * x_s, x = centered log       (exposure control, ONE recipient dof)
                      genome size for the 60 extant leaves, 0 for
                      ancestral; gamma fit by ML on the train fold
  free weights        full joint (theta, alpha) [+ saddle escape]   (current claim; nests the offset)

Answers the reviewer question "does the recipient effect survive re-fitting under exposure controls?":
if free beats genome-size-offset on HELD-OUT data (and both beat uniform), the recipient heterogeneity is
not merely genome-size exposure. Reports per-model held-out NLL (mean +- sd), Delta vs uniform, and the
top-10 sink overlap of each split's free fit against the full-data joint fit (sink-rank stability).

Recipient arg to the model is ALPHA (per-branch logits; the model applies log_softmax). The genome->branch
map is rebuilt in-driver by the verified subtree-structure alignment; genome sizes come from
genome_sizes.json (build_genome_sizes.py). Env: SPLITS(3) FRAC(0.2) FAMILIES(0=all) GAMMA_PTS(5)
ADAM LBFGS ESCAPE_M SMOKE(0) OUT SEED0(100).
"""
import os, sys, time, json, math
import numpy as np
import torch
from ete3 import Tree
from scipy.stats import spearmanr

RW = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # gpurec repo root
sys.path.insert(0, RW)
sys.path.insert(0, f"{RW}/experiments/sanderson_cv")
import run_cv
from run_cv import DATASETS, build_model, heldout_nll, _CV_SO
from run_cv_joint import gbm_fit_joint
from archaea_heldout import gbm_fit_theta_box
from gpurec import SolverOptions

HERE = os.path.dirname(os.path.abspath(__file__))
DEV = "cuda"
FULL_FIT = f"{RW}/experiments/sanderson_cv/runs/bounded_joint_archaea_full_fp32.pt"


# ---------------------------------------------------------------- genome <-> branch alignment
def build_alignment(model, S):
    """Map each gpurec branch index to its species-tree genome by aligning the gpurec subtree structure
    (sp_child1/2, leaves encoded c1==S) to the named Newick recursively by subtree leaf-count. Verified."""
    sh = model.species_helpers
    c1 = sh["sp_child1"].cpu().numpy(); c2 = sh["sp_child2"].cpu().numpy()
    root = int(np.where(sh["sp_parent"].cpu().numpy() < 0)[0][0])
    is_leaf = (c1 == S)
    _lc = {}
    def gsub(i):
        if i in _lc: return _lc[i]
        v = 1 if is_leaf[i] else gsub(int(c1[i])) + gsub(int(c2[i]))
        _lc[i] = v; return v
    nwk = Tree(str(DATASETS["archaea"]["species_tree"]), format=1)
    idx2name = {}
    def align(i, node):
        if is_leaf[i]:
            assert node.is_leaf(), f"leaf/internal mismatch at gpurec {i}"
            idx2name[i] = node.name; return
        a, b = int(c1[i]), int(c2[i]); ka, kb = nwk_n(node.children[0]), nwk_n(node.children[1])
        if (gsub(a), gsub(b)) == (ka, kb):
            align(a, node.children[0]); align(b, node.children[1])
        elif (gsub(a), gsub(b)) == (kb, ka):
            align(a, node.children[1]); align(b, node.children[0])
        else:
            raise AssertionError(f"subtree-size mismatch at gpurec {i}")
    def nwk_n(node): return len(node.get_leaves())
    align(root, nwk)
    return idx2name


def kfold_random_splits(n, n_splits, frac, seed0):
    """n_splits independent random hold-outs of size frac*n (distinct seeds), each (train_idx, test_idx)."""
    out = []
    for s in range(n_splits):
        rng = np.random.default_rng(seed0 + s)
        perm = rng.permutation(n); cut = int(round(frac * n))
        te = np.sort(perm[:cut]); tr = np.sort(perm[cut:])
        out.append((tr.tolist(), te.tolist()))
    return out


def main():
    smoke = os.environ.get("SMOKE", "0") == "1"
    n_fam = int(os.environ.get("FAMILIES", "256" if smoke else "0"))
    n_splits = int(os.environ.get("SPLITS", "1" if smoke else "3"))
    frac = float(os.environ.get("FRAC", "0.2"))
    gamma_pts = int(os.environ.get("GAMMA_PTS", "3" if smoke else "5"))
    box = (float(os.environ.get("MIN_RATE", "0.05")), float(os.environ.get("MAX_RATE", "2.0")))
    lam = float(os.environ.get("LAM", "0.03"))
    adam = int(os.environ.get("ADAM", "20" if smoke else "40"))
    lbfgs = int(os.environ.get("LBFGS_ITERS", "30" if smoke else "150"))
    escape_m = int(os.environ.get("ESCAPE_M", "60" if smoke else "120"))
    seed0 = int(os.environ.get("SEED0", "100"))
    out = os.environ.get("OUT", os.path.join(HERE, "exposure_cv_result.json"))

    ds = DATASETS["archaea"]; run_cv._SP_TREE = ds["species_tree"]
    so = SolverOptions(**_CV_SO); so.validate()
    paths = ds["families"](None if n_fam <= 0 else n_fam); n = len(paths)

    full = build_model(paths, so); S = int(full.species_helpers["S"])
    sp_parent = full.species_helpers["sp_parent"].detach().clone()
    spp = sp_parent.to(DEV).long().reshape(-1)
    child = (spp >= 0).nonzero(as_tuple=True)[0].contiguous(); parent = spp[child].contiguous()

    # genome-size covariate x[S]: centered log genome size on the extant leaves, 0 on ancestral lineages
    gsize = json.load(open(os.path.join(HERE, "genome_sizes.json")))
    idx2name = build_alignment(full, S)
    ext = sorted(i for i in idx2name if idx2name[i] in gsize)
    x = np.zeros(S, dtype=np.float64)
    logsz = np.array([math.log(gsize[idx2name[i]]) for i in ext])
    x[ext] = logsz - logsz.mean()
    x_t = torch.tensor(x, device=DEV, dtype=torch.float32)
    print(f"[exp-cv] n={n} fam, S={S} branches, {len(ext)} extant leaves mapped to genomes; "
          f"box={box} lam={lam} splits={n_splits} gamma_pts={gamma_pts} smoke={smoke}", flush=True)

    # full-data free fit's top-10 sinks (reference ranking for stability overlap)
    dfit = torch.load(FULL_FIT, map_location="cpu", weights_only=False)
    full_top10 = set(np.argsort(-dfit["w"].numpy())[:10].tolist())
    # gamma_ref: regress full-fit log-weights on x over extant -> a sensible center for the ML grid
    wfull = dfit["w"].numpy(); gamma_ref = float(np.polyfit(x[ext], np.log(wfull[ext]), 1)[0])
    gammas = [gamma_ref * c for c in np.linspace(0.5, 2.0, gamma_pts)]   # brackets the ML optimum (> ref)
    print(f"[exp-cv] gamma_ref={gamma_ref:.3f}; ML grid {[round(g,3) for g in gammas]}", flush=True)

    theta0 = torch.full((S, 3), math.log2(0.1), device=DEV, dtype=torch.float32)
    zeros = torch.zeros(S, device=DEV, dtype=torch.float32)
    splits = kfold_random_splits(n, n_splits, frac, seed0)
    rows = {"uniform": [], "offset": [], "free": []}
    overlaps = []; gammas_fit = []; t0 = time.time()

    def dump():
        agg = {m: dict(mean=float(np.mean(v)), sd=float(np.std(v)), vals=[float(z) for z in v])
               for m, v in rows.items() if v}
        json.dump(dict(S=S, n=n, box=box, lam=lam, n_splits=n_splits, frac=frac,
                       gamma_ref=gamma_ref, gammas_fit=gammas_fit, full_top10=sorted(full_top10),
                       top10_overlap=overlaps, agg=agg, rows=rows), open(out, "w"), indent=2)

    for s, (tr, te) in enumerate(splits):
        train = build_model([paths[i] for i in tr], so)
        test = build_model([paths[i] for i in te], so)
        tb = train.batch_statics; eb = test.batch_statics
        print(f"\n[split {s}] train={len(tr)} test={len(te)}", flush=True)

        # --- uniform ---
        th_u, st_u = gbm_fit_theta_box(tb, theta0, zeros, sp_parent, lam_tree=lam, box=box, lbfgs_iters=lbfgs)
        ho_u = float(heldout_nll(eb, th_u, zeros)); rows["uniform"].append(ho_u)
        print(f"  uniform : held-out {ho_u:.1f}  ({st_u['wall_s']:.0f}s)", flush=True); dump()

        # --- genome-size offset: ML-fit gamma over the grid (warm-started theta), then score held-out ---
        best = None; th_warm = th_u.clone()
        for g in gammas:
            rw_g = (g * x_t)
            th_g, st_g = gbm_fit_theta_box(tb, th_warm, rw_g, sp_parent, lam_tree=lam, box=box, lbfgs_iters=lbfgs)
            th_warm = th_g
            if best is None or st_g["final_loss"] < best[2]:
                best = (g, th_g.clone(), st_g["final_loss"])
        g_star, th_o, tr_loss = best
        ho_o = float(heldout_nll(eb, th_o, g_star * x_t)); rows["offset"].append(ho_o); gammas_fit.append(g_star)
        print(f"  offset  : held-out {ho_o:.1f}  (gamma*={g_star:.3f}, train NLL {tr_loss:.0f})", flush=True); dump()

        # --- free joint (theta, alpha) ---
        gseed = torch.Generator(device=DEV).manual_seed(seed0 + s)
        alpha0 = 0.05 * torch.randn(S, generator=gseed, device=DEV, dtype=torch.float32)
        th_w, al_w, st_w = gbm_fit_joint(tb, theta0, alpha0, sp_parent, child, parent, lam_tree=lam,
                                         adam_steps=adam, lbfgs_iters=lbfgs, escape_m=escape_m,
                                         max_escapes=4, box=box)
        ho_w = float(heldout_nll(eb, th_w, al_w)); rows["free"].append(ho_w)
        # recipient weights of this split (softmax of alpha), top-10 overlap with the full-data fit
        w_split = torch.softmax(al_w.double(), dim=0).cpu().numpy()
        top10 = set(np.argsort(-w_split)[:10].tolist())
        ov = len(top10 & full_top10); overlaps.append(ov)
        flag = "" if st_w["lam_min"] > -1e-3 else "  <<SADDLE-NOT-ESCAPED"
        print(f"  free    : held-out {ho_w:.1f}  lam_min={st_w['lam_min']:+.2e} escapes={st_w['n_escapes']}"
              f"  top10-overlap(vs full)={ov}/10  ({st_w['wall_s']:.0f}s){flag}", flush=True); dump()
        print(f"  [split {s}] Delta vs uniform:  offset {ho_o-ho_u:+.1f}   free {ho_w-ho_u:+.1f}   "
              f"(free-offset {ho_w-ho_o:+.1f})  [total {time.time()-t0:.0f}s]", flush=True)

    # --- summary ---
    print("\n=== Cross-validated exposure-sensitivity (held-out data NLL, lower=better) ===", flush=True)
    mu = {m: (np.mean(v), np.std(v)) for m, v in rows.items() if v}
    base = mu["uniform"][0]
    for m, label in [("uniform", "uniform recipients   "), ("offset", "genome-size offset   "),
                     ("free", "free recipient wts   ")]:
        if m in mu:
            me, sd = mu[m]
            d = "" if m == "uniform" else f"   Delta vs uniform {me-base:+.1f}"
            ov = f"   top10-overlap {np.mean(overlaps):.1f}/10" if m == "free" else ""
            print(f"  {label}: {me:10.1f} +- {sd:4.1f}{d}{ov}", flush=True)
    if "offset" in mu and "free" in mu:
        go, gf = base - mu["offset"][0], base - mu["free"][0]
        print(f"\n  -> genome-size offset captures {100*go/gf:.0f}% of the free held-out gain; "
              f"the free residual adds {gf-go:+.0f} nats beyond exposure.", flush=True)
    dump(); print(f"[saved] {out}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
