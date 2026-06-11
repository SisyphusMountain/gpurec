"""Why does |g| converge with a fixed solver but NOT in the notebook?

The notebook runs ``adapt_solver_to_convergence`` every step (ADAPT_EVERY=1). That
rebatches families and gives stiff (tier-1) families a *different* solver
(pi=32/nt=32) than the base (16/16), so the gradient OPERATOR changes between
optimizer steps. L-BFGS builds a curvature history y_k = g(x_{k+1}) - g(x_k) and a
line search that both assume a FIXED objective; if g is computed by a different
solver at consecutive points, the history is inconsistent and it stalls.

This reproduces the notebook's exact batched-LBFGS (B=1) path and compares:
  C1  batched-LBFGS, NO adapt  (fixed pi=16/nt=16)        -> expect |g| -> floor
  C2  batched-LBFGS, adapt EVERY step (notebook setting)  -> expect stall / oscillation

Run:  python scripts/diagnose_notebook_nonconvergence.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# repo gpurec, not the stale .venv build (see project memory)
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import importlib.util  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from gpurec import GeneReconModel, SolverOptions  # noqa: E402
from gpurec.batched_lbfgs import BatchedLBFGS  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_FAMILIES = 80
FAMILY_ORDER = "name"
MIN_LEAVES = 4
PENALTY_LAMBDA = 1.0
INIT_RATE = 0.1
SEED = 0
STEPS = 60
LR = 0.1
BATCHED_LBFGS_MAX_ITER = 4   # notebook setting
ADAPT_PI_ITERS_HIGH = 400

SP_TREE = REPO / "tests/data/alerax_archaea_davin2017/species_reference/reference_species_tree.newick"
FAM_DIR = REPO / "tests/data/alerax_archaea_davin2017/ale_gene_tree_distributions/main_families_ge4seq"

_spec = importlib.util.spec_from_file_location(
    "archaea_opt", str(REPO / "scripts/optimize_alerax_archaea_genewise_adam.py"))
archaea_opt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(archaea_opt)


def build_model() -> tuple[GeneReconModel, int]:
    families, _ = archaea_opt.select_families(
        FAM_DIR, max_families=MAX_FAMILIES, family_order=FAMILY_ORDER,
        min_leaves=MIN_LEAVES, recursive=False)
    opts = SolverOptions(e_init=-1000.0, e_max_iter=2000, e_tol=1e-8,
                         pi_iters=16, neumann_terms=16, self_loop_solver="neumann")
    model = GeneReconModel(SP_TREE, [str(f) for f in families], mode="specieswise",
                           device=DEVICE, solver_options=opts)
    model.receiver_weights.requires_grad_(False)
    return model, len(families)


def reset(model: GeneReconModel) -> None:
    torch.manual_seed(SEED)
    with torch.no_grad():
        model.theta.copy_(torch.full_like(model.theta, float(np.log2(INIT_RATE))))
    # restore the base single-batch fixed solver (undo any prior adapt rebatching)
    model.solver_options = SolverOptions(e_init=-1000.0, e_max_iter=2000, e_tol=1e-8,
                                         pi_iters=16, neumann_terms=16, self_loop_solver="neumann")
    model.replan_batches(family_group_assignments=None)
    model.clear_warm_starts()


def run(model: GeneReconModel, *, adapt: bool, label: str) -> list[dict]:
    reset(model)
    # B=1 batched-LBFGS over the whole shared theta (notebook's specieswise option)
    theta_b = model.theta.detach().reshape(1, *model.theta.shape).clone().requires_grad_(True)
    opt = BatchedLBFGS([theta_b], lr=LR, max_iter=BATCHED_LBFGS_MAX_ITER,
                       history_size=10, max_ls=20, line_search_fn="strong_wolfe",
                       tolerance_grad=1e-7, tolerance_change=1e-9)

    def closure():
        opt.zero_grad(set_to_none=True)
        th = theta_b[0]
        data_loss = model(theta=th)
        penalty = PENALTY_LAMBDA * archaea_opt.roughness_penalty(th, model.species_helpers)
        loss = data_loss + penalty
        loss.backward()
        return loss.reshape(1)

    hist = []
    t0 = time.time()
    for step in range(STEPS):
        opt.step(closure)
        with torch.no_grad():
            model.theta.copy_(theta_b[0])
        g_inf = float(theta_b.grad.abs().max())
        n1 = n2 = 0
        if adapt:
            res = model.adapt_solver_to_convergence(pi_iters_high=ADAPT_PI_ITERS_HIGH)
            c = res["counts"]
            n1, n2 = int(c.get(1, 0)), int(c.get(2, 0))
        with torch.no_grad():
            loss_now = float(model(theta=model.theta))
        hist.append({"step": step, "g_inf": g_inf, "loss": loss_now, "n_tier1": n1, "n_tier2": n2})
        if step % 5 == 0 or step == STEPS - 1:
            extra = f"  tier1/2={n1}/{n2}" if adapt else ""
            print(f"  [{label}] step {step:3d}  |g|inf={g_inf:.3e}  loss={loss_now:11.3f}{extra}")
    print(f"  [{label}] {STEPS} steps in {time.time()-t0:.1f}s")
    return hist


def main() -> None:
    print(f"device={DEVICE}  families<= {MAX_FAMILIES}  lambda={PENALTY_LAMBDA}  optimizer=batched_lbfgs(B=1)\n")
    model, F = build_model()
    print(f"families used={F}  S={model.theta.shape[0]}\n")

    print("[C1] batched-LBFGS, NO adapt (fixed pi=16/nt=16):")
    h1 = run(model, adapt=False, label="no-adapt")
    print("\n[C2] batched-LBFGS, adapt EVERY step (notebook ADAPT_EVERY=1):")
    h2 = run(model, adapt=True, label="adapt")

    def tail_stats(h):
        g = np.array([r["g_inf"] for r in h[-20:]])
        return g.min(), g.max(), g.mean()

    g1min, g1max, g1mean = tail_stats(h1)
    g2min, g2max, g2mean = tail_stats(h2)
    flips = sum(1 for r in h2 if r["n_tier1"] > 0)
    print("\n" + "=" * 64)
    print("WHY THE NOTEBOOK DIDN'T CONVERGE")
    print("=" * 64)
    print(f"  C1 no-adapt : final |g|inf={h1[-1]['g_inf']:.3e}   last-20 |g| min/mean/max={g1min:.2e}/{g1mean:.2e}/{g1max:.2e}")
    print(f"  C2 adapt    : final |g|inf={h2[-1]['g_inf']:.3e}   last-20 |g| min/mean/max={g2min:.2e}/{g2mean:.2e}/{g2max:.2e}")
    print(f"  C2 steps with tier-1 firing: {flips}/{STEPS}  (oscillation = solver flips between steps)")
    print()
    converged_fixed = h1[-1]["g_inf"] < 0.1
    stalled_adapt = h2[-1]["g_inf"] > 5 * h1[-1]["g_inf"]
    if converged_fixed and stalled_adapt:
        print("  VERDICT: per-step adapt is the cause. Fixed solver -> |g| reaches the floor;")
        print("           per-step adapt changes the gradient operator between L-BFGS steps,")
        print("           corrupting its curvature history -> it stalls ~{:.0f}x higher.".format(
            h2[-1]["g_inf"] / max(h1[-1]["g_inf"], 1e-30)))
    else:
        print("  VERDICT: adapt is NOT the dominant cause here — investigate further.")
        print(f"    (converged_fixed={converged_fixed}, stalled_adapt={stalled_adapt})")


if __name__ == "__main__":
    main()
