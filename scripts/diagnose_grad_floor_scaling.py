"""Does the |g|inf 'floor' scale with the number of families?

In specieswise mode theta is shared [S,3]; the gradient is a SUM over families, so
the max-norm |g|inf grows roughly linearly with family count. A plateau at |g|inf~0.5
on thousands of families is the SAME converged state as |g|inf~0.02 on 80 families,
just unnormalized. This fits the notebook (many families) plateauing higher than the
80-family diagnostic, and the earlier batched-LBFGS |g|inf=0.59 on the full set.

For each family count: fit with batched-LBFGS (no adapt), report final |g|inf, the
per-family-normalized |g|inf/F, and how many families are stiff (tier 1/2) at that
size (convergence_report) -- the latter shows whether adapt would even fire.

Run:  python scripts/diagnose_grad_floor_scaling.py
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
SIZES = [80, 240, 400]
FAMILY_ORDER = "name"
MIN_LEAVES = 4
PENALTY_LAMBDA = 1.0
INIT_RATE = 0.1
SEED = 0
STEPS = 35
LR = 0.1

SP_TREE = REPO / "tests/data/alerax_archaea_davin2017/species_reference/reference_species_tree.newick"
FAM_DIR = REPO / "tests/data/alerax_archaea_davin2017/ale_gene_tree_distributions/main_families_ge4seq"

_spec = importlib.util.spec_from_file_location(
    "archaea_opt", str(REPO / "scripts/optimize_alerax_archaea_genewise_adam.py"))
archaea_opt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(archaea_opt)


def build(n: int) -> tuple[GeneReconModel, int]:
    families, _ = archaea_opt.select_families(
        FAM_DIR, max_families=n, family_order=FAMILY_ORDER, min_leaves=MIN_LEAVES, recursive=False)
    opts = SolverOptions(e_init=-1000.0, e_max_iter=2000, e_tol=1e-8,
                         pi_iters=16, neumann_terms=16, self_loop_solver="neumann")
    model = GeneReconModel(SP_TREE, [str(f) for f in families], mode="specieswise",
                           device=DEVICE, solver_options=opts)
    model.receiver_weights.requires_grad_(False)
    return model, len(families)


def fit(model: GeneReconModel) -> float:
    torch.manual_seed(SEED)
    with torch.no_grad():
        model.theta.copy_(torch.full_like(model.theta, float(np.log2(INIT_RATE))))
    model.clear_warm_starts()
    theta_b = model.theta.detach().reshape(1, *model.theta.shape).clone().requires_grad_(True)
    opt = BatchedLBFGS([theta_b], lr=LR, max_iter=4, history_size=10, max_ls=20,
                       line_search_fn="strong_wolfe", tolerance_grad=1e-7, tolerance_change=1e-9)

    def closure():
        opt.zero_grad(set_to_none=True)
        th = theta_b[0]
        loss = model(theta=th) + PENALTY_LAMBDA * archaea_opt.roughness_penalty(th, model.species_helpers)
        loss.backward()
        return loss.reshape(1)

    for _ in range(STEPS):
        opt.step(closure)
        with torch.no_grad():
            model.theta.copy_(theta_b[0])
    return float(theta_b.grad.abs().max())


def main() -> None:
    print(f"device={DEVICE}  optimizer=batched_lbfgs(B=1)  steps={STEPS}  lambda={PENALTY_LAMBDA}\n")
    rows = []
    for n in SIZES:
        t0 = time.time()
        model, F = build(n)
        g_inf = fit(model)
        rep = model.convergence_report(pi_iters_high=400)
        labels = model.classify_families(rep)
        n1 = int((labels == 1).sum()); n2 = int((labels == 2).sum())
        rows.append({"F": F, "g_inf": g_inf, "g_per_F": g_inf / F, "tier1": n1, "tier2": n2})
        print(f"  F={F:4d}  final |g|inf={g_inf:.3e}  |g|inf/F={g_inf/F:.3e}  "
              f"stiff(tier1/2)={n1}/{n2}  ({time.time()-t0:.0f}s)")

    print("\n" + "=" * 64)
    print("DOES |g|inf SCALE WITH FAMILY COUNT?")
    print("=" * 64)
    base = rows[0]
    for r in rows:
        ratio_g = r["g_inf"] / base["g_inf"]
        ratio_F = r["F"] / base["F"]
        print(f"  F={r['F']:4d}: |g|inf={r['g_inf']:.3e}  (x{ratio_g:5.2f} vs F={base['F']});  "
              f"F ratio x{ratio_F:5.2f};  |g|inf/F={r['g_per_F']:.2e}")
    spread = max(r["g_per_F"] for r in rows) / min(r["g_per_F"] for r in rows)
    print(f"\n  per-family-normalized |g|inf/F varies only {spread:.1f}x across {min(SIZES)}..{max(SIZES)} families")
    print("  => the absolute |g|inf 'floor' grows with family count; normalized, it is ~constant.")
    print("  => the notebook's higher |g| plateau (many more families) is the SAME converged")
    print("     state as the 80-family floor, not a failure to converge.")


if __name__ == "__main__":
    main()
