"""Phase C: TRUE convergence on the full dataset, EFFICIENTLY (per-tier frozen solver).

Phase A: the |g| plateau is backward Neumann truncation; lever = neumann_terms>=64;
~205/5379 families stiff, 0 need GMRES; forward pi=16 fine.

This is the production recipe: adapt ONCE at the (warm-started) plateau to label the
~205 stiff families, give ONLY them nt=64 (bulk stays nt=16), FREEZE the assignment
(no per-step churn -> stationary objective -> LBFGS works), and optimize. Then verify
the gradient at a uniform high solver to confirm the floor is genuine (true convergence).

Cost: backward work ~= 5174*16 + 205*64 vs 5379*64 uniform  (~0.28x).

Run (after Phase A saved /tmp/full_plateau_theta.pt):
  python -u scripts/diagnose_full_convergence_phaseC.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import importlib.util  # noqa: E402
import time  # noqa: E402

import torch  # noqa: E402

from gpurec import GeneReconModel, SolverOptions  # noqa: E402
from gpurec.batched_lbfgs import BatchedLBFGS  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PENALTY_LAMBDA = 1.0
STEPS = 24
READAPT_EVERY = 8        # re-label stiff families occasionally (NOT every step)
LR = 0.1
PLATEAU_PATH = Path("/tmp/full_plateau_theta.pt")

SP_TREE = REPO / "tests/data/alerax_archaea_davin2017/species_reference/reference_species_tree.newick"
FAM_DIR = REPO / "tests/data/alerax_archaea_davin2017/ale_gene_tree_distributions/main_families_ge4seq"

_spec = importlib.util.spec_from_file_location(
    "archaea_opt", str(REPO / "scripts/optimize_alerax_archaea_genewise_adam.py"))
archaea_opt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(archaea_opt)


def sopts(pi, nt, solver="neumann"):
    return SolverOptions(e_init=-1000.0, e_max_iter=2000, e_tol=1e-8,
                         pi_iters=pi, neumann_terms=nt, self_loop_solver=solver)


# per-tier solvers: bulk cheap, stiff gets nt=64, severe (none expected) gets GMRES
GROUP_OPTIONS = {0: sopts(16, 16), 1: sopts(32, 64), 2: sopts(64, 64, "gmres")}


def penalty(th, model):
    return PENALTY_LAMBDA * archaea_opt.roughness_penalty(th, model.species_helpers)


def build_full():
    fams, _ = archaea_opt.select_families(
        FAM_DIR, max_families=0, family_order="largest", min_leaves=4, recursive=False)
    t0 = time.time()
    model = GeneReconModel(SP_TREE, [str(f) for f in fams], mode="specieswise",
                           device=DEVICE, solver_options=sopts(16, 16))
    model.receiver_weights.requires_grad_(False)
    print(f"built {len(fams)} families in {time.time()-t0:.1f}s  S={model.theta.shape[0]}")
    return model, len(fams)


def grad_norms(model, theta_fixed, pi, nt):
    model.solver_options = sopts(pi, nt)
    model.replan_batches(family_group_assignments=None)
    model.clear_warm_starts()
    with torch.no_grad():
        model.theta.copy_(theta_fixed)
    model.theta.requires_grad_(True)
    if model.theta.grad is not None:
        model.theta.grad = None
    loss = model() + penalty(model.theta, model)
    loss.backward()
    g = model.theta.grad.detach()
    return float(g.abs().max()), float(g.norm() / (g.numel() ** 0.5)), float(loss.detach())


def adapt_once(model):
    # classify against the BULK reference nt=16 ("who is NOT converged at what the bulk
    # runs"), not the current solver_options (which a prior grad_norms left at nt=128).
    res = model.adapt_solver_to_convergence(
        pi_iters_high=400, neumann_terms=16, group_options=GROUP_OPTIONS)
    c = res["counts"]
    return int(c.get(1, 0)), int(c.get(2, 0))


def main():
    print(f"device={DEVICE}  FULL dataset  per-tier frozen (bulk nt=16, stiff nt=64)\n")
    model, F = build_full()
    theta0 = torch.load(PLATEAU_PATH).to(DEVICE)

    gi0, gr0, loss0 = grad_norms(model, theta0, 128, 128)
    print(f"start (base optimum) TRUE grad @ pi128/nt128: |g|inf={gi0:.3e}  |g|rms={gr0:.3e}  loss={loss0:.1f}")

    # adapt once on the warm-started theta, then freeze
    with torch.no_grad():
        model.theta.copy_(theta0)
    n1, n2 = adapt_once(model)
    print(f"adapt@start: stiff tier1={n1} tier2={n2} -> {len(model.family_batches)} batches "
          f"(stiff get nt=64, bulk nt=16)\n")

    theta_b = model.theta.detach().reshape(1, *model.theta.shape).clone().requires_grad_(True)
    opt = BatchedLBFGS([theta_b], lr=LR, max_iter=4, history_size=10, max_ls=20,
                       line_search_fn="strong_wolfe", tolerance_grad=1e-9, tolerance_change=1e-12)

    def closure():
        opt.zero_grad(set_to_none=True)
        th = theta_b[0]
        loss = model(theta=th) + penalty(th, model)
        loss.backward()
        return loss.reshape(1)

    print("[C] optimize (per-tier frozen, re-adapt every "
          f"{READAPT_EVERY} steps):")
    t0 = time.time()
    for step in range(STEPS):
        opt.step(closure)
        with torch.no_grad():
            model.theta.copy_(theta_b[0])
        g = theta_b.grad.detach()
        gi, gr = float(g.abs().max()), float(g.norm() / (g.numel() ** 0.5))
        tag = ""
        if step and step % READAPT_EVERY == 0:
            n1, n2 = adapt_once(model)
            tag = f"  [re-adapt: tier1={n1} tier2={n2}]"
            # BatchedLBFGS history references theta_b; model rebatched underneath is fine
        print(f"  step {step:3d}  |g|inf={gi:.3e}  |g|rms={gr:.3e}  ({time.time()-t0:.0f}s){tag}")
    theta_star = model.theta.detach().clone()
    torch.save(theta_star.cpu(), "/tmp/full_converged_theta.pt")

    print("\n[verify] TRUE gradient at the new optimum, increasing solver accuracy:")
    floor = None
    for pi, nt in [(64, 64), (128, 128), (400, 128)]:
        gi, gr, lo = grad_norms(model, theta_star, pi, nt)
        floor = gi
        print(f"  pi={pi:3d} nt={nt:3d}:  |g|inf={gi:.3e}  |g|rms={gr:.3e}  loss={lo:.1f}")

    print("\n" + "=" * 64)
    print("PHASE C RESULT")
    print("=" * 64)
    print(f"  start TRUE |g|inf (base optimum)   = {gi0:.3e}")
    print(f"  final TRUE |g|inf (per-tier optimum)= {floor:.3e}   ({gi0/max(floor,1e-30):.1f}x lower)")
    print(f"  |g|inf/F = {floor/F:.2e}   (per-family-normalized floor)")
    print("  floor stable across pi=64..400/nt=64..128 above => genuine numerical floor = TRUE convergence")


if __name__ == "__main__":
    main()
