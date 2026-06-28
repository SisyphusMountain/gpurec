"""Coleman (bacteria) D-L identifiability generality check.

Tests whether the duplication-loss weak-identifiability (turnover D+L soft, net D-L stiff, transfer
decoupled) -- demonstrated on archaea -- REPLICATES on a second, independent dataset: Coleman et al.
(2021) 1,007-genome bacteria (S=2,013 branches), 335 random-COG gene families. Fits species-wise rates
(box [0.05,2.0], tree-smoothing lam=0.03), then profile-likelihood slices along the global turnover/net/
transfer directions. If turnover is flat and net is stiff (as on archaea), the identifiability result is
general, not archaea-specific.

Env: FAMILIES(335) NPTS(21) TMAX(1.5) LBFGS_ITERS(120) OUT.
"""
import os, sys, glob, math
import numpy as np
import torch
RW = "/home/enzo/Documents/git/gpurec/agent-worktrees/receiver-weights-hvp"
sys.path.insert(0, RW)
from run_cv import _CV_SO
from archaea_heldout import gbm_fit_theta_box
from gpurec import GeneReconModel, SolverOptions
from gpurec.optim.value_and_grad import make_value_and_grad
DEV = "cuda"
ST = "/home/enzo/Documents/git/gpurec/gpurec/tests/data/coleman/Section01.SpeciesTree/ReferenceTree.nwk"
GT = ("/home/enzo/Documents/git/gpurec/gpurec/benchmarks/large_dataset_capacity/datasets/"
      "bacteria_1007_randomsample/EDCluster_randomsample_step1_iqtree_lg_g4")


def main():
    n = int(os.environ.get("FAMILIES", "335")); npts = int(os.environ.get("NPTS", "21"))
    tmax = float(os.environ.get("TMAX", "1.5")); lbfgs = int(os.environ.get("LBFGS_ITERS", "120"))
    box = (0.05, 2.0); lam = 0.03
    out = os.environ.get("OUT", f"{RW}/experiments/sanderson_cv/runs/coleman_dl.pt")
    paths = sorted(glob.glob(f"{GT}/*.treefile"))[:n]
    print(f"[coleman] {len(paths)} families; building model...", flush=True)
    so = SolverOptions(**_CV_SO); so.validate()
    model = GeneReconModel(str(ST), [str(p) for p in paths], mode="specieswise", device=DEV, solver_options=so)
    bs = model.batch_statics; S = int(model.species_helpers["S"])
    rw = model.receiver_weights.detach().clone()
    sp_parent = model.species_helpers["sp_parent"].detach().clone()
    print(f"[coleman] S={S} branches (1007 genomes), p={3*S} rates", flush=True)

    theta0 = torch.full((S, 3), math.log2(0.1), device=DEV, dtype=torch.float32)
    th, st = gbm_fit_theta_box(bs, theta0, rw, sp_parent, lam_tree=lam, box=box, lbfgs_iters=lbfgs)
    print(f"[coleman] rate fit: final NLL+pen={st['final_loss']:.1f} |g|={st['final_gnorm']:.2e} "
          f"({st['wall_s']:.0f}s)", flush=True)

    f = make_value_and_grad(bs, rw, theta_shape=(S, 3))                 # data NLL (no penalty)
    def loss(theta): return float(f(theta.reshape(-1))[0])
    L0 = loss(th)
    ts = torch.linspace(-tmax, tmax, npts)
    def slice_along(vec):
        v = vec / vec.norm()
        return [loss(th + float(t) * v) - L0 for t in ts]
    g = {}
    g_turn = torch.zeros(S, 3, device=DEV); g_turn[:, 0] = 1; g_turn[:, 1] = 1
    g_net = torch.zeros(S, 3, device=DEV); g_net[:, 0] = 1; g_net[:, 1] = -1
    g_tau = torch.zeros(S, 3, device=DEV); g_tau[:, 2] = 1
    for name, v in [("turnover", g_turn), ("net", g_net), ("transfer", g_tau)]:
        g[name] = slice_along(v)
    # CLEAN curvature via the exact theta-HVP (fp32-safe; HVP accumulated in fp64 -> no cancellation):
    # v^T H_data v along each global direction. (The small-eps FD fails by fp32 cancellation, L~1e6.)
    import saddle_escape, gc
    gc.collect(); torch.cuda.empty_cache()                              # release the fit's ~18 GB before the big HVP
    p = 3 * S
    lap0 = (lambda v: torch.zeros(p, dtype=torch.float64, device=DEV))   # data Hessian only (no penalty)
    Av = saddle_escape.build_hvp_once(bs, th.reshape(S, 3), rw, lap0, p)
    def curv(vec):
        v = (vec / vec.norm()).reshape(-1).float()   # model dtype
        Hv = Av(v)                                    # fp64 accumulator
        return float(v.double() @ Hv)
    cv = {name: curv(v) for name, v in [("turnover", g_turn), ("net", g_net), ("transfer", g_tau)]}
    torch.save(dict(ts=ts, L0=L0, curves=g, curv=cv, eps=eps, S=S, n=len(paths), dataset="coleman"), out)
    print(f"\n=== Coleman CLEAN local curvature (central 2nd-diff eps={eps}; gradient-cancelled) ===", flush=True)
    for k in ["turnover", "net", "transfer"]:
        print(f"  {k:9s}: {cv[k]:.1f}", flush=True)
    print(f"  net/turnover = {cv['net']/max(cv['turnover'],1e-9):.2f}x ; "
          f"transfer/turnover = {cv['transfer']/max(cv['turnover'],1e-9):.2f}x "
          f"({'REPLICATES archaea: net stiffest, turnover softest' if cv['net']>cv['turnover'] and cv['turnover']<cv['transfer'] else 'check'})", flush=True)

    # soft/stiff ratio at +tmax (max curvature side)
    turn_p = max(abs(g["turnover"][-1]), abs(g["turnover"][0]))
    net_p = max(abs(g["net"][-1]), abs(g["net"][0]))
    tau_p = max(abs(g["transfer"][-1]), abs(g["transfer"][0]))
    print(f"\n=== Coleman D-L profile (data NLL=L0={L0:.1f}) ===", flush=True)
    print(f"  turnover (D+L) [expect SOFT]:  max dNLL={turn_p:.1f}", flush=True)
    print(f"  net (D-L)     [expect STIFF]:  max dNLL={net_p:.1f}", flush=True)
    print(f"  transfer (T)  [decoupled]   :  max dNLL={tau_p:.1f}", flush=True)
    print(f"  net/turnover curvature ratio = {net_p/max(turn_p,1e-9):.1f}x  "
          f"({'REPLICATES archaea (net stiffer)' if net_p>turn_p else 'does NOT replicate'})", flush=True)
    print(f"[saved] {out}", flush=True)


if __name__ == "__main__":
    main()
