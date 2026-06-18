"""Is the |g|inf=1.659 floor the float32 limit, or a batched-LBFGS stall?

At the converged theta the TRUE gradient is |g|inf=1.659 (stable across solvers).
batched-LBFGS froze there. Test decisively WITHOUT a slow optimizer:
  1. line-probe: evaluate loss at theta* - alpha*g for a range of alpha. If ANY
     alpha reduces the loss, theta* is not stationary => optimizer stall (a stronger
     optimizer goes lower). If none do, 1.659 is the genuine float32 floor.
  2. also run a few plain GD / Adam steps (no line search) to see if |g| breaks 1.659.

Run:  python -u scripts/diagnose_floor_probe.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import importlib.util  # noqa: E402
import torch  # noqa: E402

from gpurec import GeneReconModel, SolverOptions  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PENALTY_LAMBDA = 1.0
CONVERGED_PATH = Path("/tmp/full_converged_theta.pt")

SP_TREE = REPO / "tests/data/alerax_archaea_davin2017/species_reference/reference_species_tree.newick"
FAM_DIR = REPO / "tests/data/alerax_archaea_davin2017/ale_gene_tree_distributions/main_families_ge4seq"

_spec = importlib.util.spec_from_file_location(
    "archaea_opt", str(REPO / "scripts/optimize_alerax_archaea_genewise_adam.py"))
archaea_opt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(archaea_opt)


def sopts(pi, nt):
    return SolverOptions(e_max_iter=2000, e_tol=1e-8,
                         pi_iters=pi, neumann_terms=nt, self_loop_solver="neumann")


def penalty(th, model):
    return PENALTY_LAMBDA * archaea_opt.roughness_penalty(th, model.species_helpers)


def main():
    fams, _ = archaea_opt.select_families(FAM_DIR, max_families=0, family_order="largest",
                                          min_leaves=4, recursive=False)
    model = GeneReconModel(SP_TREE, [str(f) for f in fams], mode="specieswise",
                           device=DEVICE, solver_options=sopts(64, 64))
    model.receiver_weights.requires_grad_(False)
    theta_star = torch.load(CONVERGED_PATH).to(DEVICE)
    print(f"loaded converged theta  S={model.theta.shape[0]}  F={len(fams)}")

    def loss_grad(theta):
        model.clear_warm_starts()
        with torch.no_grad():
            model.theta.copy_(theta)
        model.theta.requires_grad_(True)
        if model.theta.grad is not None:
            model.theta.grad = None
        loss = model() + penalty(model.theta, model)
        loss.backward()
        return float(loss.detach()), model.theta.grad.detach().clone()

    def loss_only(theta):
        model.clear_warm_starts()
        with torch.no_grad():
            l = float(model(theta=theta) + penalty(theta, model))
        return l

    L0, g = loss_grad(theta_star)
    gi = float(g.abs().max())
    print(f"loss(theta*) = {L0:.4f}   |g|inf = {gi:.4e}   |g|2 = {float(g.norm()):.4e}\n")

    print("[1] line-probe along -g (does ANY step reduce the loss?):")
    best = (0.0, L0)
    for alpha in (1e-5, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1):
        L = loss_only(theta_star - alpha * g)
        d = L - L0
        flag = "  <-- reduces loss" if d < 0 else ""
        print(f"   alpha={alpha:.0e}:  loss={L:.4f}  dloss={d:+.4f}{flag}")
        if L < best[1]:
            best = (alpha, L)
    print(f"   best alpha={best[0]:.0e}  loss={best[1]:.4f}  (dloss={best[1]-L0:+.4f})\n")

    print("[2] a few plain GD steps (no line search) from theta* -- does |g| break 1.659?")
    th = theta_star.clone()
    for step in range(6):
        L, g = loss_grad(th)
        gi = float(g.abs().max())
        print(f"   gd step {step}: loss={L:.4f}  |g|inf={gi:.4e}")
        th = (th - 1e-2 * g).detach()

    print("\n" + "=" * 60)
    print("FLOOR VERDICT")
    print("=" * 60)
    if best[1] < L0 - 0.05:
        print(f"  loss IS reducible (best dloss={best[1]-L0:+.4f}) => theta* not stationary")
        print("  => 1.659 was a batched-LBFGS STALL; a stronger optimizer descends further.")
    else:
        print(f"  loss NOT reducible along -g (best dloss={best[1]-L0:+.4f})")
        print("  => 1.659 is the genuine float32 floor (loss is flatter than fp32 can resolve).")


if __name__ == "__main__":
    main()
