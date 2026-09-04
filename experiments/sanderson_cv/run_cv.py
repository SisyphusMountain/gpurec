"""Sanderson-style penalized-likelihood cross-validation on the 1055-family hogenom set.

Penalty = autocorrelated-rates (GBM) tree roughness  R(theta) = 1/2 sum_edges ||theta_c - theta_p||^2
(graph Laplacian over the species tree; no arbitrary center -- lam->inf gives the clock with the
common rate set by the data). The CV objective is the held-out predictive NLL; we sweep lam over a
descending homotopy (large lam -> small lam, warm-started), k-fold over families, pick lam* = argmin
mean held-out NLL, then refit on all families along the same homotopy. Each refit is certified a true
local minimum post-hoc by certify.py (Lanczos min-eig of H + lam*L via the ANALYTIC exact-HVP summed
over batches -- NOT an FD HVP, which cannot resolve the near-zero bottom eigenvalue).

Design decisions (see docs/optim/sanderson_cv.md):
  * init theta = 0  -> all DTL probs 0.25 (the empirically better basin).
  * converged solver pi=64/neumann=64 (pi=16 gradient is biased ~5%, would corrupt the optima).
  * homotopy high->low lam, warm-started, so each fit starts inside the previous (more convex) basin.
  * scipy L-BFGS-B: by default unconstrained (bounds=None) -- the prior alone regularizes; pass
    --min-rate/--max-rate to ALSO box theta=log2(rate) (rho in [min,max]), i.e. the bounded CV that
    pairs the lambda penalty with the box on theta of Sec. "certified optimality" (rho in [0.05,2.0]).
  * robustness: lam-level theta checkpoints + state.pt + JSONL event log -> resumable; wandb timing.

Run:
  GPUREC_PREPROCESS_PATH=<.../libgpurec_preprocess.so> \
  python experiments/sanderson_cv/run_cv.py --families 1055 --k 5 [--no-wandb] [--smoke N]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from gpurec import GeneReconModel, SolverOptions
from gpurec.fit.optimize import Schedule
from gpurec.solver.value_and_grad import make_value_and_grad

HERE = Path(__file__).resolve().parent
DATA = Path(
    os.environ.get(
        "GPUREC_DATA_ROOT",
        HERE.parents[1]
        / "data"
        / "external"
        / "benchmarks"
        / "large_dataset_capacity"
        / "datasets",
    )
)
HOGENOM_ROOT = Path(os.environ.get("GPUREC_HOGENOM_ROOT", DATA / "alerax_hogenom_core/hogenom"))
# archaea data root is env-overridable so the same code runs on the cluster (data shipped to /work)
ARCHAEA_ROOT = Path(os.environ.get("GPUREC_ARCHAEA_ROOT", DATA / "alerax_archaea_davin2017"))

# The linear extinction-adjoint solve used to be BiCGSTAB, configured by bicgstab_max_iter /
# bicgstab_tol / bicgstab_breakdown_tol. It is now a Neumann series (the E-step self-map is a
# contraction, so the series converges and there is no breakdown to guard against), configured by
# e_adjoint_max_iter / e_adjoint_tol -- so the three old names no longer exist on SolverOptions and
# this dict raised TypeError on construction, i.e. run_cv.py did not import at all.
# The 500-iteration budget carries over unchanged. The old 1e-7 tolerance does NOT: SolverOptions
# documents it as sitting below float32's own achievable floor, which made the solve raise on an
# essentially converged iterate, and ``None`` means "use the dtype's own resolution" (1e-6 in
# float32, 1e-12 in float64), which is the robust setting.
_CV_SO = dict(
    e_max_iter=2000, e_tol=1e-8, pi_iters=64, neumann_terms=64,
    e_adjoint_max_iter=500, e_adjoint_tol=None,
    adjoint_pruning_threshold=1e-6,
    use_adjoint_pruning=True, pibar_side_threshold=0.0,
)


# ----------------------------------------------------------------------------------------------
# datasets: each maps to a species tree + a family-path resolver. Add a dataset here to run CV on it.
# ----------------------------------------------------------------------------------------------
def _hogenom_families(n):
    fams = [ln.strip() for ln in open(HERE / "families_1055.txt") if ln.strip()]
    if n is not None:
        fams = fams[:n]
    return [str(HOGENOM_ROOT / "families" / f / "gene_trees" / "ufboot1000.MFP.geneTree.newick") for f in fams]


def _hogenom_full_families(n):
    # full hogenom family list (12408, >=4 species); shipped as a tracked input (cf. families_1055.txt)
    fams = [ln.strip() for ln in open(HERE / "families_hogenom_full.txt") if ln.strip()]
    if n is not None:
        fams = fams[:n]
    return [str(HOGENOM_ROOT / "families" / f / "gene_trees" / "ufboot1000.MFP.geneTree.newick") for f in fams]


def _archaea_families(n):
    import glob
    fs = sorted(glob.glob(str(ARCHAEA_ROOT / "ale_gene_tree_distributions/main_families_ge4seq/*.ale")))
    return fs[:n] if n is not None else fs


DATASETS = {
    "hogenom": dict(species_tree=HOGENOM_ROOT / "runs/MFP/true_start_ufboot1000/"
                    "run_--gene-tree-samples_100_--per-family-rates_1/alegenerax/species_trees/"
                    "starting_species_tree.newick", families=_hogenom_families),
    "hogenom_full": dict(species_tree=HOGENOM_ROOT / "runs/MFP/true_start_ufboot1000/"
                    "run_--gene-tree-samples_100_--per-family-rates_1/alegenerax/species_trees/"
                    "starting_species_tree.newick", families=_hogenom_full_families),
    "archaea": dict(species_tree=ARCHAEA_ROOT / "species_reference/reference_species_tree.newick",
                    families=_archaea_families),
}

_SP_TREE = None  # species tree path; set by run_cv() from the chosen dataset


def build_model(paths, so):
    return GeneReconModel(str(_SP_TREE), [str(p) for p in paths], mode="specieswise",
                          device="cuda", solver_options=so)


def kfold_indices(n, k, seed=0):
    g = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(n, generator=g).tolist()
    folds = [perm[i::k] for i in range(k)]            # round-robin -> balanced sizes
    return [(sorted(j for f in range(k) if f != i for j in folds[f]), sorted(folds[i]))
            for i in range(k)]


# ----------------------------------------------------------------------------------------------
# fit / eval / certify
# ----------------------------------------------------------------------------------------------
def theta_stats(theta, bounds=None):
    t = theta.detach().reshape(-1).float()
    d = dict(mean=float(t.mean()), std=float(t.std()), absmax=float(t.abs().max()),
             frac_extreme=float((t.abs() > 5).float().mean()))  # boundary-saturation indicator
    if bounds is not None:                                       # box-constrained run: report active set
        lo, hi = bounds
        tol = 1e-3 * max(hi - lo, 1e-12)
        d["frac_active"] = float(((t <= lo + tol) | (t >= hi - tol)).float().mean())  # rates pinned at the box
    return d


def gbm_fit(batch_statics, theta0, rw, sp_parent, *, lam_tree, adam_steps=40, adam_lr=1.0,
            lbfgs_iters=120, maxcor=50, log=None, tag="", solve_dtype=torch.float32,
            theta_bounds=None):
    """argmin_theta  sum NLL_i + (lam_tree/2) sum_edges ||theta_c - theta_p||^2,  from theta0.

    Adam (basin entry) -> scipy L-BFGS-B (penalized endgame). The SOLVER runs in
    ``solve_dtype`` (fp32 default -- on a consumer 4090 fp64 is ~27x slower with bit-identical loss;
    use fp64 only on an A100). scipy's quasi-Newton bookkeeping stays fp64 on the CPU. Returns
    (theta_hat, stats). ``log(d)`` is called per logged iteration with a scalar dict.

    ``theta_bounds=(lo, hi)`` (in theta=log2(rate) space) imposes the box rho in [2^lo, 2^hi] as a
    HARD constraint -- the Adam warmup is clamped to the box each step and L-BFGS-B runs box-bounded.
    The CV objective then selects lambda for the penalized AND bounded problem (lambda penalty + the
    bound on theta together). ``None`` (default) is the original unconstrained behaviour."""
    from scipy.optimize import minimize

    S3 = tuple(theta0.shape)
    f = make_value_and_grad(batch_statics, rw, theta_shape=S3, tree_penalty=(lam_tree, sp_parent))
    dev = theta0.device
    t_start = time.perf_counter()
    n_solves = 0

    theta = theta0.detach().reshape(S3).float().clone()
    if theta_bounds is not None:
        theta.clamp_(theta_bounds[0], theta_bounds[1])
    if adam_steps > 0:
        leaf = theta.clone().requires_grad_(True)
        opt = torch.optim.Adam([leaf], lr=adam_lr)
        sched = Schedule("adaptive", adam_lr, t_max=adam_steps)
        for it in range(int(adam_steps)):
            loss, g, _sv, _ = f(leaf.detach().reshape(-1)); n_solves += 1
            lr = sched.update(loss, g)
            opt.param_groups[0]["lr"] = lr
            leaf.grad = g.reshape(S3)
            opt.step()
            if theta_bounds is not None:                        # project the warmup back into the box
                with torch.no_grad():
                    leaf.clamp_(theta_bounds[0], theta_bounds[1])
            if log and (it % 5 == 0 or it == adam_steps - 1):
                log(dict(phase="adam", it=it, loss=loss, gnorm=float(g.norm()), lr=lr,
                         t=time.perf_counter() - t_start, **{f"theta_{k}": v for k, v in
                         theta_stats(leaf, theta_bounds).items()}), tag=tag)
        theta = leaf.detach()

    state = {"loss": math.nan, "gnorm": math.nan}

    def fun(x_np):
        nonlocal n_solves
        x = torch.tensor(x_np, device=dev, dtype=solve_dtype)  # solver in fp32 (fp64 = 27x on 4090)
        loss, g, _sv, _ = f(x); n_solves += 1
        state["loss"], state["gnorm"] = float(loss), float(g.norm())
        return float(loss), g.double().cpu().numpy().astype(np.float64)  # scipy bookkeeping fp64

    it_box = {"n": 0}

    def cb(xk):
        it_box["n"] += 1
        if log and (it_box["n"] % 3 == 0):
            log(dict(phase="lbfgs", it=it_box["n"], loss=state["loss"], gnorm=state["gnorm"],
                     t=time.perf_counter() - t_start,
                     **{f"theta_{k}": v for k, v in
                        theta_stats(torch.tensor(xk).reshape(S3), theta_bounds).items()}), tag=tag)

    x0 = theta.reshape(-1).double().cpu().numpy().astype(np.float64)
    bnds = None if theta_bounds is None else [(theta_bounds[0], theta_bounds[1])] * x0.size
    res = minimize(fun, x0, jac=True, method="L-BFGS-B", bounds=bnds, callback=cb,
                   options={"maxiter": lbfgs_iters, "maxfun": lbfgs_iters * 2,
                            "maxcor": maxcor, "ftol": 1e-12, "gtol": 1e-8})
    theta_hat = torch.tensor(res.x, device=dev, dtype=torch.float32).reshape(S3)
    stats = dict(final_loss=float(res.fun), final_gnorm=float(np.linalg.norm(res.jac)),
                 nit=int(res.nit), n_solves=n_solves, wall_s=time.perf_counter() - t_start,
                 **theta_stats(theta_hat, theta_bounds))
    return theta_hat, stats


def heldout_nll(batch_statics, theta, rw):
    """Pure predictive NLL sum_i NLL_i(theta) over the families in batch_statics (no penalty, no grad)."""
    from gpurec.api._execution import stream_batches
    # stream_batches gained a fourth positional argument, the per-species ORIGINATION logits, and a
    # fourth return value, their gradient. All-zero logits are the uniform origination prior this
    # CV has always assumed, so passing zeros reproduces the original behaviour exactly.
    loss, _g, _gr, _go = stream_batches(batch_statics, theta, rw, torch.zeros_like(rw),
                                        genewise=False, need_grad=False)
    return float(loss)


# PD certificate (smallest eigenvalue of H + lam*L) is done by the ANALYTIC exact-HVP summed over
# batches -- see certify.py:make_exact_multibatch_hvp. We do NOT use an FD-of-gradient HVP: at a
# GBM-penalized min the bottom eigenvalue is near-zero (~0.02) and FD's ~0.5%*||H|| truncation floor
# cannot resolve it (it sign-flips). run_cv defers certification to certify.py (run on the saved
# refit thetas, fp64 on the A100 for the authoritative pass); here we only log the gradient norm
# (the cheap first-order necessary condition) on every fit.


# ----------------------------------------------------------------------------------------------
# driver with checkpoint/resume + wandb
# ----------------------------------------------------------------------------------------------
class Run:
    def __init__(self, outdir, use_wandb, wandb_cfg):
        self.outdir = Path(outdir)
        (self.outdir / "ckpt").mkdir(parents=True, exist_ok=True)
        self.state_path = self.outdir / "state.pt"
        self.log_path = self.outdir / "events.jsonl"
        self.state = torch.load(self.state_path, weights_only=False) if self.state_path.exists() else {
            "cells": {}, "folds_done": [], "global_step": 0, "refit": {}}
        self.wandb = None
        if use_wandb:
            try:
                import wandb
                self.wandb = wandb.init(project="gpurec-sanderson-cv", config=wandb_cfg,
                                        dir=str(self.outdir), resume="allow",
                                        id=wandb_cfg.get("run_id"))
            except Exception as exc:  # noqa: BLE001 -- never let wandb kill a multi-hour run
                print(f"[wandb] init failed ({exc}); continuing with events.jsonl only")

    def log(self, d, tag=""):
        self.state["global_step"] += 1
        step = self.state["global_step"]
        rec = {"step": step, "tag": tag, "wall_clock": time.time(), **d}
        with open(self.log_path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        if self.wandb:
            self.wandb.log({f"{tag}/{k}" if tag else k: v for k, v in d.items()
                            if isinstance(v, (int, float))}, step=step)
        if d.get("phase") in (None, "summary") or d.get("it", 1) == 0:
            print(f"  [{tag}] " + "  ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                                            for k, v in d.items() if k != "wall_clock"))

    def save(self):
        tmp = self.state_path.with_suffix(".pt.tmp")
        torch.save(self.state, tmp)
        tmp.replace(self.state_path)  # atomic


def run_cv(*, n_families, k, lambdas, seed, outdir, dataset="hogenom", use_wandb=True, adam_steps=40,
           lbfgs_iters=120, theta_bounds=None):
    global _SP_TREE
    ds = DATASETS[dataset]
    _SP_TREE = ds["species_tree"]
    so = SolverOptions(**_CV_SO); so.validate()
    paths = ds["families"](n_families)
    n = len(paths)
    lambdas = sorted({float(x) for x in lambdas}, reverse=True)  # descending homotopy
    bnd_tag = "" if theta_bounds is None else f"_box{theta_bounds[0]:.3g},{theta_bounds[1]:.3g}"
    cfg = dict(dataset=dataset, n_families=n, k=k, lambdas=lambdas, seed=seed, adam_steps=adam_steps,
               lbfgs_iters=lbfgs_iters, solver="pi64_neu64",
               penalty="gbm_tree_laplacian", init="theta0",
               theta_bounds=theta_bounds, bounded=theta_bounds is not None,
               run_id=f"sandcv_{dataset}_n{n}_k{k}_s{seed}_nl{len(lambdas)}_it{lbfgs_iters}{bnd_tag}")
    run = Run(outdir, use_wandb, cfg)
    print(f"[run_cv] dataset={dataset}  n={n} families, k={k}, lambdas(desc)={lambdas}\n  outdir={outdir}  "
          f"resume: {len(run.state['folds_done'])} folds + {len(run.state['refit'])} refit-lams done")

    # full model: source of sp_parent / rw / S and the final refit; built once.
    full = build_model(paths, so)
    S = int(full.species_helpers["S"])
    rw = full.receiver_weights.detach().clone()
    sp_parent = full.species_helpers["sp_parent"].detach().clone()
    theta0 = torch.zeros((S, 3), device="cuda", dtype=torch.float32)
    folds = kfold_indices(n, k, seed)

    # ---- k-fold CV ----
    for fi, (tr, te) in enumerate(folds):
        if fi in run.state["folds_done"]:
            print(f"[fold {fi}] already done -> skip"); continue
        print(f"[fold {fi}/{k}] train={len(tr)} test={len(te)}")
        train = build_model([paths[i] for i in tr], so)
        test = build_model([paths[i] for i in te], so)
        # resume within fold: continue homotopy from the last checkpointed lam
        theta = theta0.clone()
        start_li = 0
        for li in range(len(lambdas)):
            ck = run.outdir / "ckpt" / f"fold{fi}_lam{li}.pt"
            if ck.exists() and f"{fi},{li}" in run.state["cells"]:
                theta = torch.load(ck, weights_only=False)["theta"].cuda()
                start_li = li + 1
        for li in range(start_li, len(lambdas)):
            lam = lambdas[li]
            t0 = time.perf_counter()
            theta, st = gbm_fit(train.batch_statics, theta, rw, sp_parent, lam_tree=lam,
                                adam_steps=adam_steps, lbfgs_iters=lbfgs_iters,
                                log=run.log, tag=f"fold{fi}/lam{lam:g}", theta_bounds=theta_bounds)
            ho = heldout_nll(test.batch_statics, theta, rw)
            run.state["cells"][f"{fi},{li}"] = dict(fold=fi, lam=lam, heldout=ho, per_fam=ho/max(1, len(te)),
                                                    **st)
            torch.save({"theta": theta.cpu(), "lam": lam, "fold": fi}, run.outdir/"ckpt"/f"fold{fi}_lam{li}.pt")
            run.log(dict(phase="summary", fold=fi, lam=lam, heldout=ho, per_fam=ho/max(1, len(te)),
                         final_loss=st["final_loss"], final_gnorm=st["final_gnorm"],
                         frac_extreme=st["frac_extreme"], frac_active=st.get("frac_active"),
                         n_solves=st["n_solves"],
                         fit_wall_s=time.perf_counter()-t0), tag=f"fold{fi}/lam{lam:g}")
            run.save()
        run.state["folds_done"].append(fi)
        run.save()
        del train, test; torch.cuda.empty_cache()

    # ---- CV curve + lam* ----
    cv = {}
    for li, lam in enumerate(lambdas):
        vals = [c["heldout"] for c in run.state["cells"].values() if c["lam"] == lam and math.isfinite(c["heldout"])]
        if len(vals) == k:
            cv[lam] = sum(vals) / k
    lam_star = min(cv, key=cv.get) if cv else None
    run.state["cv"] = cv; run.state["lam_star"] = lam_star; run.save()
    print("\n=== CV curve (mean held-out NLL) ===")
    for lam in sorted(cv, reverse=True):
        print(f"  lam={lam:<10.4g} CV={cv[lam]:.4f}{'   <- lam*' if lam == lam_star else ''}")

    # ---- all-data refit along the homotopy (PD certificate is run post-hoc by certify.py, which
    #      uses the ANALYTIC exact-HVP summed over batches -- see the note above) ----
    print(f"\n=== all-data refit (lam* = {lam_star}); certify with certify.py afterwards ===")
    theta = theta0.clone()
    for li, lam in enumerate(lambdas):
        if str(lam) in run.state["refit"]:
            theta = torch.load(run.outdir/"ckpt"/f"refit_lam{li}.pt", weights_only=False)["theta"].cuda()
            continue
        theta, st = gbm_fit(full.batch_statics, theta, rw, sp_parent, lam_tree=lam,
                            adam_steps=adam_steps, lbfgs_iters=lbfgs_iters, log=run.log,
                            tag=f"refit/lam{lam:g}", theta_bounds=theta_bounds)
        torch.save({"theta": theta.cpu(), "lam": lam}, run.outdir/"ckpt"/f"refit_lam{li}.pt")
        run.state["refit"][str(lam)] = dict(lam=lam, final_loss=st["final_loss"],
            final_gnorm=st["final_gnorm"], frac_extreme=st["frac_extreme"], frac_active=st.get("frac_active"),
            lam_min=None, ritz_resid=None, certified_pd=None)  # filled by certify.py
        run.log(dict(phase="summary", lam=lam, final_loss=st["final_loss"],
                     final_gnorm=st["final_gnorm"], frac_extreme=st["frac_extreme"],
                     frac_active=st.get("frac_active")),
                tag=f"refit/lam{lam:g}")
        run.save()
        act = "" if st.get("frac_active") is None else f" frac_active={st['frac_active']:.2f}"
        print(f"  lam={lam:<10.4g} F={st['final_loss']:.2f} |g|={st['final_gnorm']:.2e} "
              f"frac|theta|>5={st['frac_extreme']:.2f}{act}  (PD cert deferred to certify.py)")

    print(f"\n[run_cv] DONE. lam*={lam_star}  state={run.state_path}")
    if run.wandb:
        run.wandb.finish()
    return run.state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(DATASETS), default="hogenom")
    ap.add_argument("--families", type=int, default=None, help="number of families (default: all in the dataset)")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--lambdas", type=float, nargs="+", default=[1000, 100, 10, 1, 0.1, 0.0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--adam-steps", type=int, default=40)
    ap.add_argument("--lbfgs-iters", type=int, default=120)
    ap.add_argument("--min-rate", type=float, default=None,
                    help="lower box on rate rho: theta>=log2(min_rate). Giving --min-rate/--max-rate "
                         "switches CV to the BOUNDED problem (lambda penalty + box on theta).")
    ap.add_argument("--max-rate", type=float, default=None, help="upper box on rate rho: theta<=log2(max_rate).")
    ap.add_argument("--cert-m", type=int, default=0, help="ignored; certification is post-hoc via certify.py")
    ap.add_argument("--smoke", type=int, default=0, help="override n_families for a quick smoke")
    args = ap.parse_args()
    n = args.smoke if args.smoke else args.families
    theta_bounds = None
    if args.min_rate is not None or args.max_rate is not None:
        lo = math.log2(args.min_rate) if args.min_rate is not None else -float("inf")
        hi = math.log2(args.max_rate) if args.max_rate is not None else float("inf")
        theta_bounds = (lo, hi)
    suffix = "_bounded" if theta_bounds is not None else ""
    outdir = args.outdir or str(HERE / "runs" / f"cv_{args.dataset}_n{n or 'all'}{suffix}")
    run_cv(n_families=n, k=args.k, lambdas=args.lambdas, seed=args.seed, outdir=outdir,
           dataset=args.dataset, use_wandb=not args.no_wandb, adam_steps=args.adam_steps,
           lbfgs_iters=args.lbfgs_iters, theta_bounds=theta_bounds)


if __name__ == "__main__":
    main()
