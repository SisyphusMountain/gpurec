"""Reproduce the notebook regime (largest/stiff families + per-step adapt) and test
whether per-step adaptive rebatching is what stalls |g|.

The notebook used FAMILY_ORDER='largest', MAX_FAMILIES=0 (all), ADAPT_EVERY=1. Stiff
(deep, large-leaf) families are where tier-1 actually fires. We pick the largest-by-leaf
families that still fit (skip the >5MB / ~1500-leaf monsters) and compare batched-LBFGS:
  C1  adapt OFF (fixed pi=16/nt=16)
  C2  adapt EVERY step (notebook ADAPT_EVERY=1)
recording per-step |g|inf and tier-1/2 counts. If per-step adapt is the culprit, C2
stalls / oscillates (tier-1 fire-evict-fire) while C1 descends to a lower floor.

Run:  python scripts/diagnose_stiff_adapt_reproduce.py
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
N_FAMILIES = 100          # largest-by-leaf families that fit
MAX_BYTES = 5_000_000     # skip the >5MB monsters (~1500 leaves) that OOM
PENALTY_LAMBDA = 1.0
INIT_RATE = 0.1
SEED = 0
STEPS = 40
LR = 0.1
ADAPT_PI_ITERS_HIGH = 400

SP_TREE = REPO / "tests/data/alerax_archaea_davin2017/species_reference/reference_species_tree.newick"
FAM_DIR = REPO / "tests/data/alerax_archaea_davin2017/ale_gene_tree_distributions/main_families_ge4seq"

_spec = importlib.util.spec_from_file_location(
    "archaea_opt", str(REPO / "scripts/optimize_alerax_archaea_genewise_adam.py"))
archaea_opt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(archaea_opt)


def select_stiff(n: int) -> tuple[list[str], list[int]]:
    # file size is an instant stiffness proxy (bigger file = more Dip_counts = deeper
    # family); counting leaves reads each whole multi-MB file (minutes over ~5400 files).
    cands = [(p, p.stat().st_size) for p in FAM_DIR.glob("*.ale")]
    cands = [(p, sz) for p, sz in cands if sz <= MAX_BYTES]
    cands.sort(key=lambda ps: (-ps[1], ps[0].name))
    chosen = cands[:n]
    sizes_kb = [sz // 1024 for _, sz in chosen]
    return [str(p) for p, _ in chosen], sizes_kb


def build() -> tuple[GeneReconModel, int, list[int]]:
    fams, lvs = select_stiff(N_FAMILIES)
    opts = SolverOptions(e_max_iter=2000, e_tol=1e-8,
                         pi_iters=16, neumann_terms=16, self_loop_solver="neumann")
    # drop any unparsable straggler one at a time
    while True:
        try:
            model = GeneReconModel(SP_TREE, fams, mode="specieswise", device=DEVICE, solver_options=opts)
            break
        except ValueError as exc:
            bad = next((f for f in fams if Path(f).name in str(exc)), None)
            if bad is None:
                raise
            fams.remove(bad)
    model.receiver_weights.requires_grad_(False)
    return model, len(fams), lvs


def reset(model: GeneReconModel) -> None:
    torch.manual_seed(SEED)
    with torch.no_grad():
        model.theta.copy_(torch.full_like(model.theta, float(np.log2(INIT_RATE))))
    model.solver_options = SolverOptions(e_max_iter=2000, e_tol=1e-8,
                                         pi_iters=16, neumann_terms=16, self_loop_solver="neumann")
    model.replan_batches(family_group_assignments=None)
    model.clear_warm_starts()


def run(model: GeneReconModel, *, adapt: bool, label: str) -> list[dict]:
    reset(model)
    theta_b = model.theta.detach().reshape(1, *model.theta.shape).clone().requires_grad_(True)
    opt = BatchedLBFGS([theta_b], lr=LR, max_iter=4, history_size=10, max_ls=20,
                       line_search_fn="strong_wolfe", tolerance_grad=1e-7, tolerance_change=1e-9)

    def closure():
        opt.zero_grad(set_to_none=True)
        th = theta_b[0]
        loss = model(theta=th) + PENALTY_LAMBDA * archaea_opt.roughness_penalty(th, model.species_helpers)
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
            c = res["counts"]; n1, n2 = int(c.get(1, 0)), int(c.get(2, 0))
        hist.append({"step": step, "g_inf": g_inf, "n1": n1, "n2": n2})
        if step % 4 == 0 or step == STEPS - 1:
            extra = f"  tier1/2={n1}/{n2}" if adapt else ""
            print(f"  [{label}] step {step:3d}  |g|inf={g_inf:.3e}{extra}")
    print(f"  [{label}] {STEPS} steps in {time.time()-t0:.1f}s")
    return hist


def main() -> None:
    print(f"device={DEVICE}  stiff largest-leaf families (<= {MAX_BYTES//10**6}MB)  lambda={PENALTY_LAMBDA}\n")
    model, F, sizes_kb = build()
    print(f"families used={F}  S={model.theta.shape[0]}  file-size range={min(sizes_kb)}..{max(sizes_kb)} KB "
          f"(median {int(np.median(sizes_kb))} KB)\n")

    print("[C1] batched-LBFGS, adapt OFF (fixed pi=16/nt=16):")
    h1 = run(model, adapt=False, label="no-adapt")
    print("\n[C2] batched-LBFGS, adapt EVERY step (notebook ADAPT_EVERY=1):")
    h2 = run(model, adapt=True, label="adapt")

    g1 = np.array([r["g_inf"] for r in h1]); g2 = np.array([r["g_inf"] for r in h2])
    fired = sum(1 for r in h2 if r["n1"] > 0 or r["n2"] > 0)
    print("\n" + "=" * 64)
    print("STIFF REGIME: does per-step adapt stall |g|?")
    print("=" * 64)
    print(f"  C1 no-adapt: final |g|inf={g1[-1]:.3e}   last-10 mean={g1[-10:].mean():.3e}")
    print(f"  C2 adapt   : final |g|inf={g2[-1]:.3e}   last-10 mean={g2[-10:].mean():.3e}")
    print(f"  C2 steps with tier-1/2 firing: {fired}/{STEPS}")
    if fired == 0:
        print("  (tier-1 never fired even on stiff families -> adapt is a no-op here too;")
        print("   the notebook's tier-1 firing needs the full largest/all set or stiffer theta.)")
    else:
        worse = g2[-10:].mean() / max(g1[-10:].mean(), 1e-30)
        print(f"  adapt floor is {worse:.2f}x the no-adapt floor "
              f"({'WORSE -> per-step adapt stalls it' if worse > 1.5 else 'comparable'})")


if __name__ == "__main__":
    main()
