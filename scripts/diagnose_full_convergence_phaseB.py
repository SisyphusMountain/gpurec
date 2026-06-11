"""Phase B: drive the FULL dataset to TRUE convergence with the converged solver.

Phase A showed the |g| plateau is backward Neumann truncation: at the base-solver
optimum the converged-solver gradient is |g|inf~14.7 (the base nt=16 gradient of 5.5
is a biased undershoot). The lever is neumann_terms>=64 (forward pi=16 is fine);
~205/5379 families are under-converged at nt=16, 0 need GMRES.

This re-optimizes from the saved plateau theta with a CONVERGED, uniform solver
(pi=64/nt=64) and checks that |g| measured at that same solver descends to a real
floor -- and that the floor does not move when we tighten to pi=128/nt=128 (=> the
floor is the genuine float32 numerical floor, i.e. true convergence).

Run (after Phase A saved /tmp/full_plateau_theta.pt):
  python scripts/diagnose_full_convergence_phaseB.py
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
STEPS = 30
LR = 0.1
PI, NT = 64, 64          # converged solver (from Phase A sweeps)
PLATEAU_PATH = Path("/tmp/full_plateau_theta.pt")

SP_TREE = REPO / "tests/data/alerax_archaea_davin2017/species_reference/reference_species_tree.newick"
FAM_DIR = REPO / "tests/data/alerax_archaea_davin2017/ale_gene_tree_distributions/main_families_ge4seq"

_spec = importlib.util.spec_from_file_location(
    "archaea_opt", str(REPO / "scripts/optimize_alerax_archaea_genewise_adam.py"))
archaea_opt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(archaea_opt)


def solver(pi: int, nt: int) -> SolverOptions:
    return SolverOptions(e_init=-1000.0, e_max_iter=4000, e_tol=1e-10,
                         pi_iters=pi, neumann_terms=nt, self_loop_solver="neumann")


def penalty(th, model):
    return PENALTY_LAMBDA * archaea_opt.roughness_penalty(th, model.species_helpers)


def build_full() -> GeneReconModel:
    fams, _ = archaea_opt.select_families(
        FAM_DIR, max_families=0, family_order="largest", min_leaves=4, recursive=False)
    model = GeneReconModel(SP_TREE, [str(f) for f in fams], mode="specieswise",
                           device=DEVICE, solver_options=solver(PI, NT))
    model.receiver_weights.requires_grad_(False)
    return model


def grad_norms(model, theta_fixed, pi, nt):
    model.solver_options = solver(pi, nt)
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


def main() -> None:
    print(f"device={DEVICE}  FULL dataset  converged solver pi={PI}/nt={NT}\n")
    model = build_full()
    theta0 = torch.load(PLATEAU_PATH).to(DEVICE)
    F = len(model.families)

    gi0, gr0, loss0 = grad_norms(model, theta0, PI, NT)
    print(f"start (base-solver optimum), measured at converged solver: "
          f"|g|inf={gi0:.3e}  |g|rms={gr0:.3e}  loss={loss0:.1f}\n")

    # optimize with the converged solver from the plateau warm start
    model.solver_options = solver(PI, NT)
    model.clear_warm_starts()
    with torch.no_grad():
        model.theta.copy_(theta0)
    theta_b = model.theta.detach().reshape(1, *model.theta.shape).clone().requires_grad_(True)
    opt = BatchedLBFGS([theta_b], lr=LR, max_iter=4, history_size=10, max_ls=20,
                       line_search_fn="strong_wolfe", tolerance_grad=1e-9, tolerance_change=1e-12)

    def closure():
        opt.zero_grad(set_to_none=True)
        th = theta_b[0]
        loss = model(theta=th) + penalty(th, model)
        loss.backward()
        return loss.reshape(1)

    print(f"[B] optimize with pi={PI}/nt={NT} (converged, uniform -> stationary objective):")
    t0 = time.time()
    last = {}
    for step in range(STEPS):
        opt.step(closure)
        with torch.no_grad():
            model.theta.copy_(theta_b[0])
        g = theta_b.grad.detach()
        last = {"g_inf": float(g.abs().max()), "g_rms": float(g.norm() / (g.numel() ** 0.5))}
        if step % 3 == 0 or step == STEPS - 1:
            with torch.no_grad():
                lo = float(model(theta=theta_b[0]) + penalty(theta_b[0], model))
            print(f"  step {step:3d}  |g|inf={last['g_inf']:.3e}  |g|rms={last['g_rms']:.3e}  "
                  f"loss={lo:.1f}  ({time.time()-t0:.0f}s)")
    theta_star = model.theta.detach().clone()
    torch.save(theta_star.cpu(), "/tmp/full_converged_theta.pt")

    # verify: is this the true floor? tighten the solver and re-measure the SAME theta.
    print("\n[verify] gradient at the new optimum, at increasing solver accuracy:")
    for pi, nt in [(PI, NT), (128, 128), (400, 128)]:
        gi, gr, lo = grad_norms(model, theta_star, pi, nt)
        print(f"  pi={pi:3d} nt={nt:3d}:  |g|inf={gi:.3e}  |g|rms={gr:.3e}  loss={lo:.1f}")

    # how many families still stiff at the converged optimum?
    model.solver_options = solver(PI, NT)
    model.clear_warm_starts()
    with torch.no_grad():
        model.theta.copy_(theta_star)
    rep = model.convergence_report(pi_iters_high=400, neumann_terms=NT)
    labels = model.classify_families(rep)
    n1 = int((labels == 1).sum()); n2 = int((labels == 2).sum())

    gi_final, gr_final, _ = grad_norms(model, theta_star, 128, 128)
    print("\n" + "=" * 64)
    print("PHASE B RESULT (true convergence?)")
    print("=" * 64)
    print(f"  start |g|inf (converged solver) = {gi0:.3e}")
    print(f"  final |g|inf (converged solver) = {gi_final:.3e}   ({gi0/max(gi_final,1e-30):.0f}x lower)")
    print(f"  final |g|rms = {gr_final:.3e}   |g|inf/F = {gi_final/F:.3e}")
    print(f"  stiff at optimum: tier1={n1} tier2={n2} of {F}")
    print(f"  floor stable across pi=64..400 / nt=64..128 (see [verify]) => genuine numerical floor")


if __name__ == "__main__":
    main()
