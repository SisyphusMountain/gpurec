"""MAP + cross-validated ridge prior (Sanderson-2002-style penalized likelihood) for gpurec.

k-fold cross-validation over FAMILIES with a lambda-homotopy. The CV fold unit is the per-family
NLL. Train/test masking is done by DATA SUBSETTING: in specieswise mode E (survival) depends only
on theta + the species tree, so a train-only batch reproduces ``sum_{i in train} NLL_i`` and its
gradient EXACTLY (verified: per-family NLL/grad additivity holds to ~1e-8 / ~1e-6). This reuses the
parity-verified :func:`gpurec.api._execution.stream_batches` and avoids any backward-seed surgery.

Per fold/lambda the MAP objective fit on the train families is
    ``sum_{i in train} NLL_i(theta) + (lambda/2) ||theta - theta_ref||^2``
(the ``prior`` term of :func:`gpurec.solver.value_and_grad.make_value_and_grad`).
``CV(lambda) = mean over folds of held-out predictive NLL``; pick ``lambda* = argmin`` and refit on
all families at ``lambda*``.

    python -m gpurec.fit.map_cv            # small live-hogenom smoke test (finite + sane CV curve)
"""

from __future__ import annotations

import math

import torch

from gpurec import GeneReconModel, SolverOptions
from gpurec.api._execution import stream_batches
from gpurec.fit.optimize import Schedule
from gpurec.solver.value_and_grad import make_value_and_grad

# solver settings matching the kernel-bench fixture mint (production truncation, pi=16/neumann=16);
# used by the parity tests. NOT for the CV fit: at pi=16 the gradient is biased (FD disagrees ~5%),
# which would corrupt the per-fold optima.
_DEFAULT_SO = dict(
    e_max_iter=128, e_tol=1e-8, pi_iters=16, neumann_terms=16,
    self_loop_solver="neumann", bicgstab_max_iter=128, bicgstab_tol=None,
    bicgstab_breakdown_tol=None, adjoint_pruning_threshold=1e-6,
    use_adjoint_pruning=True, pibar_side_threshold=0.0,
)

# CV default: CONVERGED solver (pi>=64, neumann>=32-64) so the fitted theta is a true minimum and
# the gradient matches a fp64 finite-difference oracle (verified in _verify_map). Required for a
# scientifically valid CV curve.
_CV_SO = {**_DEFAULT_SO, "pi_iters": 64, "neumann_terms": 64}


def kfold_indices(n, k, seed=0):
    """Return ``k`` (train_idx, test_idx) splits of ``range(n)`` (shuffled by ``seed``)."""
    g = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(n, generator=g).tolist()
    folds = [perm[i::k] for i in range(k)]  # round-robin -> balanced sizes
    out = []
    for i in range(k):
        test = sorted(folds[i])
        train = sorted(j for f in range(k) if f != i for j in folds[f])
        out.append((train, test))
    return out


def _build(species_tree, paths, *, mode, device, solver_options):
    return GeneReconModel(
        str(species_tree), [str(p) for p in paths], mode=mode, device=device,
        solver_options=solver_options,
    )


def heldout_nll(batch_statics, theta, receiver_weights):
    """Total predictive NLL ``sum_{i} NLL_i(theta)`` over the families in ``batch_statics`` (no grad)."""
    loss, _g, _gr, _go = stream_batches(batch_statics, theta, receiver_weights,
                                        torch.zeros_like(receiver_weights),
                                        genewise=False, need_grad=False)
    return float(loss)


def fit_map(batch_statics, theta0, receiver_weights, *, lam, theta_ref,
            adam_steps=60, adam_lr=1.0, lbfgs_iters=80, maxcor=50, verbose=False):
    """Fit ``argmin_theta sum_train NLL_i + (lam/2)||theta-theta_ref||^2`` from ``theta0``.

    Adam (basin entry) -> scipy L-BFGS-B (penalized convex-ish endgame), both on the prior-enabled
    value-and-grad. Returns ``theta_hat`` of ``theta0``'s shape.
    """
    import numpy as np
    from scipy.optimize import minimize

    theta_shape = tuple(theta0.shape)
    f = make_value_and_grad(batch_statics, receiver_weights, theta_shape=theta_shape,
                            prior=(lam, theta_ref))
    dev = theta0.device

    theta = theta0.detach().reshape(theta_shape).float().clone()
    if adam_steps > 0:
        leaf = theta.clone().requires_grad_(True)
        opt = torch.optim.Adam([leaf], lr=adam_lr)
        sched = Schedule("adaptive", adam_lr, t_max=adam_steps)
        warm = None
        for _ in range(int(adam_steps)):
            loss, g, _sv, warm = f(leaf.detach().reshape(-1), warm_E=warm)
            opt.param_groups[0]["lr"] = sched.update(loss, g)
            leaf.grad = g.reshape(theta_shape)
            opt.step()
        theta = leaf.detach()

    state = {"warm": None}

    def fun(x_np):
        x = torch.tensor(x_np, device=dev, dtype=torch.float64)
        loss, g, _sv, warm = f(x, warm_E=state["warm"])
        state["warm"] = warm
        return float(loss), g.double().cpu().numpy().astype(np.float64)

    x0 = theta.reshape(-1).double().cpu().numpy().astype(np.float64)
    res = minimize(fun, x0, jac=True, method="L-BFGS-B",
                   options={"maxiter": lbfgs_iters, "maxfun": lbfgs_iters * 2,
                            "maxcor": maxcor, "ftol": 1e-12, "gtol": 1e-8})
    if verbose:
        print(f"    [fit_map lam={lam:.4g}] final F={res.fun:.4f} nit={res.nit}")
    return torch.tensor(res.x, device=dev, dtype=torch.float32).reshape(theta_shape)


# Reference CV tuning; clone-override per dataset. See docs/config_convention.md.
MAP_CV_REFERENCE = dict(
    k=5, lambdas=(0.0, 1.0, 10.0, 100.0, 1000.0), mode="specieswise", init_rate=0.1, seed=0,
    adam_steps=60, lbfgs_iters=80, maxcor=50,
)


def map_cv(species_tree, gene_trees, *, k=5, lambdas=(0.0, 1.0, 10.0, 100.0, 1000.0),
           mode="specieswise", init_rate=0.1, seed=0, solver_options=None, device="cuda",
           adam_steps=60, lbfgs_iters=80, maxcor=50, verbose=True):
    """Run k-fold-over-families CV with a lambda-homotopy. Returns a results dict with the CV curve,
    ``lam_star``, and the final all-families refit ``theta_final``.

    ``theta_ref`` (the ridge center) is the constant ``log2(init_rate)`` per-species rate; lambda
    shrinks the per-species rates toward that common rate (the Sanderson smoothing flavor).
    """
    so = SolverOptions(**(solver_options or _CV_SO))
    so.validate()
    gene_trees = list(gene_trees)
    n = len(gene_trees)

    # one full model: gives S / receiver_weights / theta_ref shape and the final refit.
    full = _build(species_tree, gene_trees, mode=mode, device=device, solver_options=so)
    S = int(full.species_helpers["S"])
    rw = full.receiver_weights.detach().clone()  # uniform (zeros); held fixed
    theta_ref = torch.full((S, 3), math.log2(init_rate), device=device, dtype=torch.float32)

    lambdas_desc = sorted({float(x) for x in lambdas}, reverse=True)
    cv_sum = {lam: 0.0 for lam in lambdas_desc}
    cv_count = {lam: 0 for lam in lambdas_desc}
    folds = kfold_indices(n, k, seed)
    for fi, (tr, te) in enumerate(folds):
        if verbose:
            print(f"[fold {fi+1}/{k}] train={len(tr)} test={len(te)}")
        train = _build(species_tree, [gene_trees[i] for i in tr], mode=mode, device=device,
                       solver_options=so)
        test = _build(species_tree, [gene_trees[i] for i in te], mode=mode, device=device,
                      solver_options=so)
        theta = theta_ref.clone()  # warm-start carried down the homotopy
        for lam in lambdas_desc:   # large lambda first, warm-started downward
            theta = fit_map(train.batch_statics, theta, rw, lam=lam, theta_ref=theta_ref,
                            adam_steps=adam_steps, lbfgs_iters=lbfgs_iters, maxcor=maxcor,
                            verbose=verbose)
            ho = heldout_nll(test.batch_statics, theta, rw)
            cv_sum[lam] += ho
            cv_count[lam] += 1
            if verbose:
                print(f"    lam={lam:<8.4g} held-out NLL={ho:.4f}  (per-family {ho/max(1,len(te)):.4f})")

    cv = {lam: cv_sum[lam] / max(1, cv_count[lam]) for lam in lambdas_desc}
    finite = {lam: v for lam, v in cv.items() if math.isfinite(v)}
    lam_star = min(finite, key=finite.get) if finite else None
    theta_final = None
    if lam_star is not None:
        theta_final = fit_map(full.batch_statics, theta_ref.clone(), rw, lam=lam_star,
                              theta_ref=theta_ref, adam_steps=adam_steps, lbfgs_iters=lbfgs_iters,
                              maxcor=maxcor, verbose=verbose)
    if verbose:
        print("\n=== CV curve (mean held-out NLL per fold) ===")
        for lam in sorted(cv):
            star = "  <- lam*" if lam == lam_star else ""
            print(f"  lam={lam:<10.4g} CV={cv[lam]:.4f}{star}")
    return {"cv": cv, "lam_star": lam_star, "theta_final": theta_final, "S": S, "n": n,
            "k": k, "lambdas": lambdas_desc, "folds": [(len(t), len(e)) for t, e in folds]}


def _smoke(n_families=120, k=3, device="cuda"):
    """Small live-hogenom smoke: CV curve finite + a regularized lambda generalizes <= lambda=0."""
    import glob
    root = "/home/enzo/Documents/git/gpurec/gpurec/tests/data/alerax_hogenom_core/hogenom"
    sp = (f"{root}/runs/MFP/true_start_ufboot1000/"
          "run_--gene-tree-samples_100_--per-family-rates_1/alegenerax/species_trees/"
          "starting_species_tree.newick")
    trees = sorted(glob.glob(f"{root}/families/*/gene_trees/ufboot1000.MFP.geneTree.newick"))[:n_families]
    res = map_cv(sp, trees, k=k, lambdas=(0.0, 1.0, 10.0, 100.0, 1000.0), device=device,
                 adam_steps=40, lbfgs_iters=60, verbose=True)
    cv = res["cv"]
    all_finite = all(math.isfinite(v) for v in cv.values())
    best_pos = min((cv[l] for l in cv if l > 0), default=math.inf)
    helps = best_pos <= cv.get(0.0, math.inf) + 1e-6
    print(f"\n[map_cv smoke] all_finite={all_finite}  lam*={res['lam_star']}  "
          f"best_lam>0_CV={best_pos:.4f}  lam0_CV={cv.get(0.0):.4f}  reg_helps={helps}")
    ok = all_finite and res["lam_star"] is not None
    print(f"  -> {'PASS' if ok else 'FAIL'} (finite + well-defined lam*; reg_helps is the science signal)")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if _smoke() else 1)
