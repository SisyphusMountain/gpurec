"""Drive the FULL archaea dataset (all ~5379 families, specieswise) to the float32
gradient floor (|g|inf ~= 1.66), the converged operating point.

Recipe (see docs/full_dataset_convergence.md for the full investigation):
  1. build the full dataset (largest order, all families) with the BASE solver
     pi_iters=16/neumann_terms=16;
  2. optimize with batched-LBFGS(B=1) to the base-solver plateau -- but that gradient
     is BIASED: 205/5379 stiff families have an unconverged backward Neumann adjoint at
     nt=16, so the optimizer settles at the WRONG point (true |g|inf ~= 14.7 there,
     under-reported as ~5.5 by the truncated solver);
  3. fix it with the EXACT lever -- give ONLY the stiff families nt=64, grouped into
     their own batch, set once and FROZEN (re-checked every few steps, never per-step,
     which would make the objective non-stationary and break LBFGS). Bulk stays nt=16.
  4. continue LBFGS -> true |g|inf descends 14.7 -> ~1.66, then stalls: that 1.66 is the
     genuine float32 floor (the loss, a float32 sum ~358567, has ULP 0.031; near the min
     the remaining descent is < 1 ULP and the line search can no longer see improvement).
  5. verify: the gradient at the converged theta is identical at pi=64/128/400 and
     nt=64/128 -> the floor is solver-accurate, not truncation.

Levers: neumann_terms>=64 for stiff families (per-tier frozen) is the big one (14.7->1.66);
the float32 loss/gradient quantization sets the 1.66 floor (lower it with float64/Kahan
accumulation of the family-sum -- see the markdown).

Usage:
  python -u scripts/converge_full_archaea.py                 # full self-contained run
  python -u scripts/converge_full_archaea.py --warm-start X  # resume from a saved theta .pt
  python -u scripts/converge_full_archaea.py --steps-base 35 --steps-finish 10
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))  # use the repo gpurec, not the stale .venv build

import importlib.util  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from gpurec import GeneReconModel, SolverOptions  # noqa: E402
from gpurec.batched_lbfgs import BatchedLBFGS  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PENALTY_LAMBDA = 1.0
INIT_RATE = 0.1
SEED = 0
LR = 0.1
READAPT_EVERY = 8

SP_TREE = REPO / "tests/data/alerax_archaea_davin2017/species_reference/reference_species_tree.newick"
FAM_DIR = REPO / "tests/data/alerax_archaea_davin2017/ale_gene_tree_distributions/main_families_ge4seq"

_spec = importlib.util.spec_from_file_location(
    "archaea_opt", str(REPO / "scripts/optimize_alerax_archaea_genewise_adam.py"))
archaea_opt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(archaea_opt)


def sopts(pi, nt, solver="neumann"):
    return SolverOptions(e_init=-1000.0, e_max_iter=2000, e_tol=1e-8,
                         pi_iters=pi, neumann_terms=nt, self_loop_solver=solver)


# per-tier solvers: bulk cheap (nt=16), stiff gets nt=64, severe (none here) -> GMRES
GROUP_OPTIONS = {0: sopts(16, 16), 1: sopts(32, 64), 2: sopts(64, 64, "gmres")}


def penalty(th, model):
    return PENALTY_LAMBDA * archaea_opt.roughness_penalty(th, model.species_helpers)


def build_full():
    fams, _ = archaea_opt.select_families(FAM_DIR, max_families=0, family_order="largest",
                                          min_leaves=4, recursive=False)
    t0 = time.time()
    model = GeneReconModel(SP_TREE, [str(f) for f in fams], mode="specieswise",
                           device=DEVICE, solver_options=sopts(16, 16))
    model.receiver_weights.requires_grad_(False)
    print(f"built {len(fams)} families in {time.time()-t0:.1f}s  S={model.theta.shape[0]}  "
          f"batches={len(model.family_batches)}")
    return model, len(fams)


def lbfgs_optimize(model, steps, *, adapt_first, label):
    """Optimize the shared specieswise theta with batched-LBFGS(B=1)."""
    theta_b = model.theta.detach().reshape(1, *model.theta.shape).clone().requires_grad_(True)
    opt = BatchedLBFGS([theta_b], lr=LR, max_iter=4, history_size=10, max_ls=20,
                       line_search_fn="strong_wolfe", tolerance_grad=1e-9, tolerance_change=1e-12)

    def closure():
        opt.zero_grad(set_to_none=True)
        th = theta_b[0]
        loss = model(theta=th) + penalty(th, model)
        loss.backward()
        return loss.reshape(1)

    for step in range(steps):
        opt.step(closure)
        with torch.no_grad():
            model.theta.copy_(theta_b[0])
        g = theta_b.grad.detach()
        gi, gr = float(g.abs().max()), float(g.norm() / g.numel() ** 0.5)
        tag = ""
        if adapt_first and step and step % READAPT_EVERY == 0:
            n1, n2 = adapt_once(model)
            tag = f"  [re-adapt tier1={n1} tier2={n2}]"
        print(f"  [{label}] step {step:3d}  |g|inf={gi:.3e}  |g|rms={gr:.3e}{tag}")
    return model.theta.detach().clone()


def adapt_once(model):
    # classify against the BULK reference nt=16 ("who is NOT converged at the bulk's nt"),
    # group stiff families into their own batch at nt=64, freeze. Returns (tier1, tier2).
    res = model.adapt_solver_to_convergence(
        pi_iters_high=400, neumann_terms=16, group_options=GROUP_OPTIONS)
    c = res["counts"]
    return int(c.get(1, 0)), int(c.get(2, 0))


def grad_at(model, theta_fixed, pi, nt):
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
    return float(g.abs().max()), float(g.norm() / g.numel() ** 0.5), float(loss.detach())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps-base", type=int, default=35)
    ap.add_argument("--steps-finish", type=int, default=10)
    ap.add_argument("--warm-start", type=str, default=None, help="resume from a saved theta .pt")
    ap.add_argument("--out", type=str, default="/tmp/full_converged_theta.pt")
    args = ap.parse_args()

    print(f"device={DEVICE}  FULL dataset  lambda={PENALTY_LAMBDA}\n")
    model, F = build_full()

    # init / warm start
    model.solver_options = sopts(16, 16)
    model.replan_batches(family_group_assignments=None)
    model.clear_warm_starts()
    if args.warm_start:
        with torch.no_grad():
            model.theta.copy_(torch.load(args.warm_start).to(DEVICE))
        print(f"warm-started from {args.warm_start}")
    else:
        torch.manual_seed(SEED)
        with torch.no_grad():
            model.theta.copy_(torch.full_like(model.theta, float(np.log2(INIT_RATE))))
        print("\n[1] base solver (pi=16/nt=16) -> biased plateau:")
        lbfgs_optimize(model, args.steps_base, adapt_first=False, label="base")

    g_base = grad_at(model, model.theta.detach(), 128, 128)
    print(f"\nTRUE gradient at this theta (pi128/nt128): |g|inf={g_base[0]:.3e}  loss={g_base[2]:.1f}")

    print("\n[2] adapt ONCE -> stiff families get nt=64 (frozen), then finish with LBFGS:")
    n1, n2 = adapt_once(model)
    print(f"  stiff: tier1={n1} tier2={n2} of {F}  ({len(model.family_batches)} batches)")
    theta_star = lbfgs_optimize(model, args.steps_finish, adapt_first=True, label="finish")
    torch.save(theta_star.cpu(), args.out)

    print("\n[3] verify the floor is solver-accurate (gradient stable across solvers):")
    gi_final = None
    for pi, nt in [(64, 64), (128, 128), (400, 128)]:
        gi, gr, lo = grad_at(model, theta_star, pi, nt)
        gi_final = gi
        print(f"  pi={pi:3d} nt={nt:3d}:  |g|inf={gi:.3e}  |g|rms={gr:.3e}  loss={lo:.1f}")

    print("\n" + "=" * 60)
    ok = 1.4 <= gi_final <= 1.9
    print(f"RESULT: TRUE |g|inf {g_base[0]:.2f} -> {gi_final:.3f}   "
          f"({'ATTAINED the ~1.66 float32 floor' if ok else 'unexpected'})")
    print(f"  |g|inf/loss = {gi_final/g_base[2]:.2e}   saved theta -> {args.out}")


if __name__ == "__main__":
    main()
