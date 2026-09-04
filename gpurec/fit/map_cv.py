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
import os
from pathlib import Path

import torch

from gpurec import GeneReconModel, SolverOptions
from gpurec.api._execution import stream_batches
from gpurec.config import GpurecConfig
from gpurec.core.scheduling.batching import parse_families
from gpurec.fit.specieswise_fit import fit_specieswise

# solver settings matching the kernel-bench fixture mint (production truncation, pi=16/neumann=16);
# used by the parity tests. NOT for the CV fit: at pi=16 the gradient is biased (FD disagrees ~5%),
# which would corrupt the per-fold optima. This is a distinct recipe from ``map_cv_reference`` (which
# is the CONVERGED pi=64 config below) and has no corresponding ``GpurecConfig`` factory, so it stays
# a plain literal dict.
_DEFAULT_SO = dict(
    e_max_iter=128, e_tol=1e-8, pi_iters=16, neumann_terms=16,
    e_adjoint_max_iter=128, e_adjoint_tol=None,
    adjoint_pruning_threshold=1e-6,
    use_adjoint_pruning=True, pibar_side_threshold=0.0,
)

# CV default: CONVERGED solver (pi>=64, neumann>=32-64) so the fitted theta is a true minimum and
# the gradient matches a fp64 finite-difference oracle (verified in _verify_map). Required for a
# scientifically valid CV curve. Single-sourced from ``GpurecConfig.map_cv_reference().solver``
# (task-10 brief) -- edit the values there, not here. Keys mirror ``_DEFAULT_SO`` (same solver-field
# set, pi_iters/neumann_terms overridden to the converged 64/64 tier).
_CV_SO = {k: getattr(GpurecConfig.map_cv_reference().solver, k) for k in _DEFAULT_SO}


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


def _build(species_tree, parsed, indices, all_paths, *, mode, device, solver_options):
    """Build a model over ``indices`` (positions into ``all_paths``) of an ALREADY PARSED dataset.

    Every fold and every lambda re-uses the same families in a different SUBSET, so the gene-tree
    files are read and parsed ONCE by :func:`parse_families` in :func:`map_cv` and each model here
    only re-plans its batches over the requested subset. Before this, a k-fold run parsed every
    file k+1 times (once for the full model, then once per fold as either train or test).
    """
    idx = [int(i) for i in indices]
    return GeneReconModel(
        str(species_tree), [str(all_paths[i]) for i in idx], mode=mode, device=device,
        solver_options=solver_options, parsed_families=parsed, family_indices=idx,
    )


def heldout_nll(batch_statics, theta, receiver_weights):
    """Total predictive NLL ``sum_{i} NLL_i(theta)`` over the families in ``batch_statics`` (no grad)."""
    loss, _g, _gr, _go = stream_batches(batch_statics, theta, receiver_weights,
                                        torch.zeros_like(receiver_weights),
                                        genewise=False, need_grad=False)
    return float(loss)


# Reference CV tuning; clone-override per dataset. See docs/config_convention.md.
MAP_CV_REFERENCE = dict(
    k=5, lambdas=(0.0, 1.0, 10.0, 100.0, 1000.0), mode="specieswise", init_rate=0.1, seed=0,
    adam_steps=10, max_newton=8,
)


def map_cv(species_tree, gene_trees, *, k=5, lambdas=(0.0, 1.0, 10.0, 100.0, 1000.0),
           mode="specieswise", init_rate=0.1, seed=0, solver_options=None, device="cuda",
           adam_steps=10, max_newton=8, verbose=True,
           config: GpurecConfig | None = None):
    """Run k-fold-over-families CV with a lambda-homotopy. Returns a results dict with the CV curve,
    ``lam_star``, and the final all-families refit ``theta_final``.

    ``theta_ref`` (the ridge center) is the constant ``log2(init_rate)`` per-species rate; lambda
    shrinks the per-species rates toward that common rate (the Sanderson smoothing flavor).

    ``config`` (a top-level :class:`GpurecConfig`) threads ``config.solver`` (the same key subset as
    ``_CV_SO``) into the base solver dict when no explicit ``solver_options`` is passed (explicit
    always wins), and ``config.regularizer.lambdas`` into the ``lambdas`` kwarg when it is still at
    its signature default (the reference-defaults ``==preset`` pattern -- see ``fit_genewise``'s
    ``min_rate``/``max_rate``; documented edge case: explicitly repassing the preset value AND a
    differing config resolves to the config value). ``lam_margin``/``lam_floor`` belong to
    ``fit/map_fit.py`` (out of scope) and are NOT threaded here. ``config=None`` (the default)
    reproduces today's behavior exactly.

    IMPORTANT -- ``config`` is AUTHORITATIVE: passing any non-default ``config`` replaces this
    recipe's CV-tuned solver defaults (``_CV_SO``) with ``config.solver``'s values. To keep the CV
    tuning and change only a few knobs, start from ``GpurecConfig.map_cv_reference()`` and modify it.

    NOT threaded: ``config.rates``/``config.newton``/``config.memory`` (unused/inapplicable to this
    recipe).
    """
    # REGULARIZER: reference-defaults invariant -- only substitute config.regularizer.lambdas when
    # `lambdas` is still at its signature-default CV grid, so an explicit lambdas= always wins.
    if config is not None and lambdas == (0.0, 1.0, 10.0, 100.0, 1000.0):
        lambdas = config.regularizer.lambdas
    # SOLVER: config.solver supplies the base dict (same key subset as _CV_SO) only when no
    # explicit solver_options is given; an explicit solver_options still wins.
    so_base = dict(_CV_SO)
    if config is not None and solver_options is None:
        so_base = {k: getattr(config.solver, k) for k in _CV_SO}
    so = SolverOptions(**(solver_options or so_base))
    so.validate()
    gene_trees = list(gene_trees)
    n = len(gene_trees)

    # Parse the species tree and every family ONCE; every fold/lambda model below is rebuilt
    # from this resident handle over a subset of family indices, so no file is read twice.
    parsed = parse_families(species_tree, gene_trees)

    # one full model: gives S / receiver_weights / theta_ref shape and the final refit.
    full = _build(species_tree, parsed, range(n), gene_trees, mode=mode, device=device,
                  solver_options=so)
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
        train = _build(species_tree, parsed, tr, gene_trees, mode=mode, device=device,
                       solver_options=so)
        test = _build(species_tree, parsed, te, gene_trees, mode=mode, device=device,
                      solver_options=so)
        theta = theta_ref.clone()  # warm-start carried down the homotopy
        for lam in lambdas_desc:   # large lambda first, warm-started downward
            theta = fit_specieswise(train.batch_statics, theta, rw, lam=lam, theta_ref=theta_ref,
                                    adam_steps=adam_steps, max_newton=max_newton,
                                    verbose=verbose)["theta"].to(theta.device)
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
        theta_final = fit_specieswise(full.batch_statics, theta_ref.clone(), rw, lam=lam_star,
                                      theta_ref=theta_ref, adam_steps=adam_steps,
                                      max_newton=max_newton,
                                      verbose=verbose)["theta"].to(theta_ref.device)
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
    repo_root = Path(__file__).resolve().parents[2]
    data_root = Path(os.environ.get(
        "GPUREC_DATA_ROOT",
        repo_root / "data" / "external" / "benchmarks" / "large_dataset_capacity" / "datasets",
    ))
    root = str(data_root / "alerax_hogenom_core" / "hogenom")
    sp = (f"{root}/runs/MFP/true_start_ufboot1000/"
          "run_--gene-tree-samples_100_--per-family-rates_1/alegenerax/species_trees/"
          "starting_species_tree.newick")
    trees = sorted(glob.glob(f"{root}/families/*/gene_trees/ufboot1000.MFP.geneTree.newick"))[:n_families]
    res = map_cv(sp, trees, k=k, lambdas=(0.0, 1.0, 10.0, 100.0, 1000.0), device=device,
                 adam_steps=10, max_newton=8, verbose=True)
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
