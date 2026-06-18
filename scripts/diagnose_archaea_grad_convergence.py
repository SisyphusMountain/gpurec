"""Diagnose whether the archaea specieswise fit can be driven to (near-)zero gradient.

Hypothesis under test (from the notebook investigation): the |g| plateau seen with
first-order optimizers (adam/rprop) is the *optimizer*, not the solver. The truncated
Neumann backward (neumann_terms=16) contributes only ~3.5e-4 relative gradient error at
the operating point, far below the |g|~30 plateau. A curvature/line-search optimizer
(L-BFGS) should drive |g| down by orders of magnitude.

This script:
  1. builds a specieswise GeneReconModel on a subset of real archaea families,
  2. fits with a first-order baseline (rprop) and with L-BFGS (strong Wolfe),
     tracking |g|inf / |g|rms per step from the SAME init,
  3. at the L-BFGS endpoint, RE-computes the full gradient at a high-accuracy solver
     (pi_iters / neumann_terms cranked up) to prove a small |g| is a true stationary
     point, not an artifact of the cheap training solver.

Run:  python scripts/diagnose_archaea_grad_convergence.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Use the repo gpurec, not the stale .venv-installed build. When run as
# `python scripts/diagnose_...py`, sys.path[0] is scripts/ (not the repo root),
# so a bare `import gpurec` would resolve to the OLD installed package whose
# native preprocessor has a buggy parser ("unexpected trailing Newick text" on
# larger family batches). The notebook does the same insert.
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import importlib.util  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from gpurec import GeneReconModel, SolverOptions  # noqa: E402

# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_FAMILIES = 80          # subset of archaea
FAMILY_ORDER = "name"      # representative mix; "largest" picks 100MB+ families that OOM together
MIN_LEAVES = 4
PENALTY_LAMBDA = 1.0       # Sanderson roughness penalty weight
INIT_RATE = 0.1            # theta0 = log2(INIT_RATE) for every entry
SEED = 0

# training solver (cheap; what the notebook uses)
PI_ITERS = 16
NEUMANN_TERMS = 16
# high-accuracy solver for the honest endpoint gradient check
PI_ITERS_HI = 400
NEUMANN_TERMS_HI = 128

SP_TREE = REPO / "tests/data/alerax_archaea_davin2017/species_reference/reference_species_tree.newick"
FAM_DIR = REPO / "tests/data/alerax_archaea_davin2017/ale_gene_tree_distributions/main_families_ge4seq"

# reuse the repo's penalty / selection / norm helpers (same code the notebook imports)
_spec = importlib.util.spec_from_file_location(
    "archaea_opt", str(REPO / "scripts/optimize_alerax_archaea_genewise_adam.py"))
archaea_opt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(archaea_opt)


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------
def build_model(solver_options: SolverOptions) -> tuple[GeneReconModel, list[str]]:
    families, sel = archaea_opt.select_families(
        FAM_DIR, max_families=MAX_FAMILIES, family_order=FAMILY_ORDER,
        min_leaves=MIN_LEAVES, recursive=False)
    fams = [str(f) for f in families]
    # build once (serial parse is reliable; no in-process retry to avoid state poisoning)
    model = GeneReconModel(SP_TREE, fams, mode="specieswise",
                           device=DEVICE, solver_options=solver_options)
    model.receiver_weights.requires_grad_(False)
    model.theta.requires_grad_(True)
    return model, fams


def reset_theta(model: GeneReconModel) -> None:
    torch.manual_seed(SEED)
    with torch.no_grad():
        model.theta.copy_(torch.full_like(model.theta, float(np.log2(INIT_RATE))))
    model.clear_warm_starts()


def total_loss_and_grad(model: GeneReconModel) -> tuple[float, float, float, float]:
    """Compute data NLL + Sanderson penalty, backprop, return (loss, data, penalty, |g|inf)."""
    if model.theta.grad is not None:
        model.theta.grad = None
    data_loss = model(theta=model.theta)
    penalty = PENALTY_LAMBDA * archaea_opt.roughness_penalty(model.theta, model.species_helpers)
    loss = data_loss + penalty
    loss.backward()
    g = model.theta.grad.detach()
    return float(loss.detach()), float(data_loss.detach()), float(penalty.detach()), float(g.abs().max())


# ---------------------------------------------------------------------------
# optimization drivers
# ---------------------------------------------------------------------------
def run_first_order(model: GeneReconModel, steps: int, lr: float) -> list[dict]:
    reset_theta(model)
    opt = torch.optim.Rprop([model.theta], lr=lr)
    hist = []
    t0 = time.time()
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        loss, data, pen, g_inf = total_loss_and_grad(model)
        opt.step()
        hist.append({"step": step, "loss": loss, "data": data, "penalty": pen, "g_inf": g_inf})
        if step % 10 == 0 or step == steps - 1:
            print(f"  rprop  step {step:4d}  loss={loss:12.3f}  |g|inf={g_inf:.3e}")
    print(f"  rprop done: {steps} steps in {time.time()-t0:.1f}s")
    return hist


def run_lbfgs(model: GeneReconModel, steps: int, lr: float, max_iter: int) -> list[dict]:
    reset_theta(model)
    opt = torch.optim.LBFGS([model.theta], lr=lr, max_iter=max_iter,
                            history_size=10, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-12, tolerance_change=1e-16)
    hist = []
    last = {}

    def closure():
        opt.zero_grad(set_to_none=True)
        data_loss = model(theta=model.theta)
        penalty = PENALTY_LAMBDA * archaea_opt.roughness_penalty(model.theta, model.species_helpers)
        loss = data_loss + penalty
        loss.backward()
        last["loss"] = float(loss.detach())
        last["data"] = float(data_loss.detach())
        last["penalty"] = float(penalty.detach())
        return loss

    t0 = time.time()
    for step in range(steps):
        opt.step(closure)
        g_inf = float(model.theta.grad.detach().abs().max())
        hist.append({"step": step, **last, "g_inf": g_inf})
        print(f"  lbfgs  step {step:4d}  loss={last['loss']:12.3f}  |g|inf={g_inf:.3e}")
    print(f"  lbfgs done: {steps} steps in {time.time()-t0:.1f}s")
    return hist


# ---------------------------------------------------------------------------
# honest endpoint check: is a small |g| real or a truncated-solver artifact?
# ---------------------------------------------------------------------------
def endpoint_gradient_at_high_accuracy(model: GeneReconModel) -> dict:
    """Recompute |g| at the current theta with the high-accuracy solver."""
    theta_fixed = model.theta.detach().clone()

    def grad_at(opts: SolverOptions) -> torch.Tensor:
        hi = build_at(model, opts)
        with torch.no_grad():
            hi.theta.copy_(theta_fixed)
        hi.theta.requires_grad_(True)
        hi.clear_warm_starts()
        if hi.theta.grad is not None:
            hi.theta.grad = None
        loss = hi(theta=hi.theta) + PENALTY_LAMBDA * archaea_opt.roughness_penalty(hi.theta, hi.species_helpers)
        loss.backward()
        return hi.theta.grad.detach().clone()

    g_train = grad_at(SolverOptions(e_max_iter=2000, e_tol=1e-8,
                                    pi_iters=PI_ITERS, neumann_terms=NEUMANN_TERMS,
                                    self_loop_solver="neumann"))
    g_hi = grad_at(SolverOptions(e_max_iter=4000, e_tol=1e-10,
                                 pi_iters=PI_ITERS_HI, neumann_terms=NEUMANN_TERMS_HI,
                                 self_loop_solver="neumann"))
    diff = (g_train - g_hi).abs().max()
    rel = diff / g_hi.abs().max().clamp_min(torch.finfo(torch.float32).tiny)
    return {
        "g_inf_train": float(g_train.abs().max()),
        "g_inf_hi": float(g_hi.abs().max()),
        "solver_diff_inf": float(diff),
        "solver_rel": float(rel),
    }


def build_at(template: GeneReconModel, opts: SolverOptions) -> GeneReconModel:
    """Rebuild a model on the same families with a different solver (template carries paths)."""
    return GeneReconModel(SP_TREE, template._family_paths_for_rebuild, mode="specieswise",
                          device=DEVICE, solver_options=opts)


# ---------------------------------------------------------------------------
def main() -> None:
    torch.manual_seed(SEED)
    print(f"device={DEVICE}  families<= {MAX_FAMILIES} ({FAMILY_ORDER})  lambda={PENALTY_LAMBDA}")
    train_opts = SolverOptions(e_max_iter=2000, e_tol=1e-8,
                               pi_iters=PI_ITERS, neumann_terms=NEUMANN_TERMS,
                               self_loop_solver="neumann")
    model, fams = build_model(train_opts)
    model._family_paths_for_rebuild = fams  # stash for endpoint rebuild
    S = model.theta.shape[0]
    print(f"S(species nodes)={S}  families used={len(fams)}  theta={tuple(model.theta.shape)}\n")

    print("[1] first-order baseline (rprop):")
    h_first = run_first_order(model, steps=80, lr=0.1)

    print("\n[2] L-BFGS (strong Wolfe):")
    h_lbfgs = run_lbfgs(model, steps=40, lr=1.0, max_iter=20)

    print("\n[3] honest endpoint gradient check (L-BFGS final theta, high-accuracy solver):")
    chk = endpoint_gradient_at_high_accuracy(model)

    g0 = h_first[0]["g_inf"]
    print("\n" + "=" * 64)
    print("DIAGNOSIS")
    print("=" * 64)
    print(f"  init                |g|inf = {g0:.3e}")
    print(f"  rprop  (80 steps)   |g|inf = {h_first[-1]['g_inf']:.3e}   "
          f"loss={h_first[-1]['loss']:.3f}")
    print(f"  lbfgs  (40 steps)   |g|inf = {h_lbfgs[-1]['g_inf']:.3e}   "
          f"loss={h_lbfgs[-1]['loss']:.3f}")
    print(f"  lbfgs reduction vs rprop plateau: {h_first[-1]['g_inf'] / max(h_lbfgs[-1]['g_inf'], 1e-30):.1f}x lower")
    print()
    print("  endpoint solver sanity (is the small |g| real?):")
    print(f"    |g|inf @ train solver (pi={PI_ITERS},nt={NEUMANN_TERMS})   = {chk['g_inf_train']:.3e}")
    print(f"    |g|inf @ hi    solver (pi={PI_ITERS_HI},nt={NEUMANN_TERMS_HI}) = {chk['g_inf_hi']:.3e}")
    print(f"    solver-induced |g| diff (inf-norm)             = {chk['solver_diff_inf']:.3e}")
    print(f"    solver-induced relative |g| error              = {chk['solver_rel']:.3e}")
    print()
    converged = h_lbfgs[-1]["g_inf"] < 0.1 * g0
    real = chk["g_inf_hi"] < 0.1 * g0
    verdict = "YES" if (converged and real) else "NO / PARTIAL"
    print(f"  CAN WE DRIVE |g| -> ~0 ?  {verdict}")
    print(f"    (L-BFGS lowered |g| by >=10x: {converged};  small |g| confirmed at hi solver: {real})")


if __name__ == "__main__":
    main()
