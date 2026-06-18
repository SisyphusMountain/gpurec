"""FULL-DATASET convergence investigation (all ~5379 archaea families, specieswise).

Goal: drive gradient descent to true convergence near the minimum and identify the
exact levers. Hypothesis (from the |g|inf~10 plateau with adapt OFF + LBFGS): the
plateau is SOLVER TRUNCATION on stiff families (pi_iters/neumann_terms too small),
not the optimizer and not a numerical floor.

Phase A (this script):
  1. build the full dataset (largest order, all families)
  2. optimize with the base solver (pi=16/nt=16) via batched-LBFGS to a |g| plateau
  3. AT the plateau (theta fixed), sweep solver accuracy and watch |g|inf:
       - joint (pi,nt) sweep            -> does the floor drop at all?
       - forward-only (pi up, nt fixed) -> is the FORWARD Pi fixed point the lever?
       - backward-only (nt up, pi fixed)-> is the Neumann series the lever?
  4. convergence_report: how many / which families are under-converged at base
  saves the plateaued theta to /tmp/full_plateau_theta.pt for Phase B (the fix).

Run:  python scripts/diagnose_full_convergence.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import importlib.util  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from gpurec import GeneReconModel, SolverOptions  # noqa: E402
from gpurec.batched_lbfgs import BatchedLBFGS  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PENALTY_LAMBDA = 1.0
INIT_RATE = 0.1
SEED = 0
STEPS = 40
LR = 0.1
PLATEAU_PATH = Path("/tmp/full_plateau_theta.pt")

SP_TREE = REPO / "tests/data/alerax_archaea_davin2017/species_reference/reference_species_tree.newick"
FAM_DIR = REPO / "tests/data/alerax_archaea_davin2017/ale_gene_tree_distributions/main_families_ge4seq"

_spec = importlib.util.spec_from_file_location(
    "archaea_opt", str(REPO / "scripts/optimize_alerax_archaea_genewise_adam.py"))
archaea_opt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(archaea_opt)


def base_solver() -> SolverOptions:
    return SolverOptions(e_max_iter=2000, e_tol=1e-8,
                         pi_iters=16, neumann_terms=16, self_loop_solver="neumann")


def build_full() -> tuple[GeneReconModel, int]:
    fams, sel = archaea_opt.select_families(
        FAM_DIR, max_families=0, family_order="largest", min_leaves=4, recursive=False)
    t0 = time.time()
    model = GeneReconModel(SP_TREE, [str(f) for f in fams], mode="specieswise",
                           device=DEVICE, solver_options=base_solver())
    model.receiver_weights.requires_grad_(False)
    print(f"built {len(fams)} families in {time.time()-t0:.1f}s  S={model.theta.shape[0]}  "
          f"batches={len(model.family_batches)}")
    return model, len(fams)


def penalty(th, model):
    return PENALTY_LAMBDA * archaea_opt.roughness_penalty(th, model.species_helpers)


def grad_at(model: GeneReconModel, theta_fixed: torch.Tensor, pi: int, nt: int,
            solver: str = "neumann") -> tuple[float, float, float]:
    """Gradient norms at a FIXED theta with a given (pi_iters, neumann_terms) solver."""
    model.solver_options = SolverOptions(e_max_iter=4000, e_tol=1e-10,
                                         pi_iters=pi, neumann_terms=nt, self_loop_solver=solver)
    model.clear_warm_starts()
    with torch.no_grad():
        model.theta.copy_(theta_fixed)
    model.theta.requires_grad_(True)
    if model.theta.grad is not None:
        model.theta.grad = None
    loss = model() + penalty(model.theta, model)
    loss.backward()
    g = model.theta.grad.detach()
    g_inf = float(g.abs().max())
    g_rms = float(g.norm() / (g.numel() ** 0.5))
    return g_inf, g_rms, float(loss.detach())


def optimize_base(model: GeneReconModel) -> tuple[torch.Tensor, list[dict]]:
    model.solver_options = base_solver()
    model.replan_batches(family_group_assignments=None)
    model.clear_warm_starts()
    torch.manual_seed(SEED)
    with torch.no_grad():
        model.theta.copy_(torch.full_like(model.theta, float(np.log2(INIT_RATE))))
    theta_b = model.theta.detach().reshape(1, *model.theta.shape).clone().requires_grad_(True)
    opt = BatchedLBFGS([theta_b], lr=LR, max_iter=4, history_size=10, max_ls=20,
                       line_search_fn="strong_wolfe", tolerance_grad=1e-9, tolerance_change=1e-12)

    def closure():
        opt.zero_grad(set_to_none=True)
        th = theta_b[0]
        loss = model(theta=th) + penalty(th, model)
        loss.backward()
        return loss.reshape(1)

    hist = []
    t0 = time.time()
    for step in range(STEPS):
        opt.step(closure)
        with torch.no_grad():
            model.theta.copy_(theta_b[0])
        g = theta_b.grad.detach()
        rec = {"step": step, "g_inf": float(g.abs().max()),
               "g_rms": float(g.norm() / (g.numel() ** 0.5))}
        hist.append(rec)
        if step % 4 == 0 or step == STEPS - 1:
            print(f"  base-LBFGS step {step:3d}  |g|inf={rec['g_inf']:.3e}  |g|rms={rec['g_rms']:.3e}  "
                  f"({time.time()-t0:.0f}s)")
    return model.theta.detach().clone(), hist


def main() -> None:
    print(f"device={DEVICE}  FULL dataset  lambda={PENALTY_LAMBDA}  optimizer=batched_lbfgs(B=1)\n")
    model, F = build_full()

    print("\n[1] optimize with BASE solver (pi=16/nt=16) to the plateau:")
    theta_plateau, hist = optimize_base(model)
    torch.save(theta_plateau.cpu(), PLATEAU_PATH)
    print(f"  plateau saved to {PLATEAU_PATH}  (final |g|inf={hist[-1]['g_inf']:.3e})")

    print("\n[2] AT the plateau (theta fixed) -- does a more accurate solver lower |g|?")
    print("    joint (pi, nt) sweep:")
    joint = [(16, 16), (32, 32), (64, 32), (128, 64), (256, 96), (400, 128)]
    base_ginf = None
    for pi, nt in joint:
        gi, gr, _ = grad_at(model, theta_plateau, pi, nt)
        if base_ginf is None:
            base_ginf = gi
        print(f"      pi={pi:3d} nt={nt:3d}:  |g|inf={gi:.3e}  |g|rms={gr:.3e}")

    print("    forward-only (vary pi, nt fixed high=128):")
    for pi in (16, 32, 64, 128, 256, 400):
        gi, gr, _ = grad_at(model, theta_plateau, pi, 128)
        print(f"      pi={pi:3d} nt=128:  |g|inf={gi:.3e}  |g|rms={gr:.3e}")

    print("    backward-only (vary nt, pi fixed high=400):")
    for nt in (4, 8, 16, 32, 64, 128):
        gi, gr, _ = grad_at(model, theta_plateau, 400, nt)
        print(f"      pi=400 nt={nt:3d}:  |g|inf={gi:.3e}  |g|rms={gr:.3e}")

    print("\n[3] convergence_report at the plateau (base solver) -- how many families are stiff?")
    model.solver_options = base_solver()
    model.replan_batches(family_group_assignments=None)
    model.clear_warm_starts()
    with torch.no_grad():
        model.theta.copy_(theta_plateau)
    rep = model.convergence_report(pi_iters_high=400)
    labels = model.classify_families(rep)
    n1 = int((labels == 1).sum()); n2 = int((labels == 2).sum())
    fwd = rep["forward_resid"]; bwd = rep["backward_relres"]
    print(f"  stiff: tier1(more-iters)={n1}  tier2(GMRES)={n2}  of {F} families")
    print(f"  forward_resid: max={float(fwd.max()):.2e}  #>1e-3={int((fwd>1e-3).sum())}")
    print(f"  backward_relres: max={float(bwd.max()):.2e}  #>1e-3={int((bwd>1e-3).sum())}")

    print("\n" + "=" * 64)
    print("PHASE A CONCLUSION")
    print("=" * 64)
    print(f"  base plateau |g|inf = {hist[-1]['g_inf']:.3e}")
    print("  see sweeps above: if |g|inf falls as pi/nt rise, the plateau is SOLVER")
    print("  truncation and the lever is whichever sweep moves it. Phase B re-optimizes")
    print("  from this theta with the sufficient uniform solver to reach the true floor.")


if __name__ == "__main__":
    main()
