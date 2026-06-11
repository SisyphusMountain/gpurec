"""Finite-difference gradient check at theta*: is the true |g| ~1.6 (float32 adjoint)
or ~60 (float64 adjoint)?

The float32 and float64 adjoints disagree 37x at the checkpoint. Central finite
differences on the loss are the independent referee: dL/dtheta_i = (L(+e) - L(-e))/(2e),
evaluated in float64 (the more trustworthy forward). Check the top few |g| coordinates
of BOTH the float32 and float64 gradients, across several step sizes e.

Run:  python -u scripts/diagnose_fd_gradcheck.py
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
CKPT = Path("/tmp/full_converged_theta.pt")

SP_TREE = REPO / "tests/data/alerax_archaea_davin2017/species_reference/reference_species_tree.newick"
FAM_DIR = REPO / "tests/data/alerax_archaea_davin2017/ale_gene_tree_distributions/main_families_ge4seq"

_spec = importlib.util.spec_from_file_location(
    "archaea_opt", str(REPO / "scripts/optimize_alerax_archaea_genewise_adam.py"))
archaea_opt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(archaea_opt)


def sopts(pi, nt):
    return SolverOptions(e_init=-1000.0, e_max_iter=2000, e_tol=1e-8,
                         pi_iters=pi, neumann_terms=nt, self_loop_solver="neumann")


def main():
    fams, _ = archaea_opt.select_families(FAM_DIR, max_families=0, family_order="largest",
                                          min_leaves=4, recursive=False)
    model = GeneReconModel(SP_TREE, [str(f) for f in fams], mode="specieswise",
                           device=DEVICE, solver_options=sopts(64, 64))
    model.receiver_weights.requires_grad_(False)
    theta = torch.load(CKPT).to(DEVICE)

    def loss64(th):
        with torch.no_grad():
            return float(model(theta=th.double())
                         + PENALTY_LAMBDA * archaea_opt.roughness_penalty(th.double(), model.species_helpers))

    def loss32(th):
        with torch.no_grad():
            return float(model(theta=th.float())
                         + PENALTY_LAMBDA * archaea_opt.roughness_penalty(th.float(), model.species_helpers))

    # analytic gradients
    th32 = theta.float().detach().requires_grad_(True)
    (model(theta=th32) + PENALTY_LAMBDA * archaea_opt.roughness_penalty(th32, model.species_helpers)).backward()
    g32 = th32.grad.detach()
    th64 = theta.double().detach().requires_grad_(True)
    (model(theta=th64) + PENALTY_LAMBDA * archaea_opt.roughness_penalty(th64, model.species_helpers)).backward()
    g64 = th64.grad.detach()

    print(f"analytic |g|inf: float32={float(g32.abs().max()):.4f}   float64={float(g64.abs().max()):.4f}\n")

    # check the coordinates that dominate each gradient
    flat32, flat64 = g32.reshape(-1), g64.reshape(-1)
    coords = sorted(set(torch.topk(flat64.abs(), 3).indices.tolist()
                        + torch.topk(flat32.abs(), 3).indices.tolist()))
    names = ["D", "L", "T"]
    print("central finite differences (float64 loss) vs analytic g32 / g64:")
    for c in coords:
        s, ev = divmod(c, 3)
        print(f"  coord species={s} event={names[ev]}:  g32={float(flat32[c]):+.4f}  g64={float(flat64[c]):+.4f}")
        for eps in (1e-1, 1e-2, 1e-3):
            tp = theta.double().clone(); tp.reshape(-1)[c] += eps
            tm = theta.double().clone(); tm.reshape(-1)[c] -= eps
            fd = (loss64(tp) - loss64(tm)) / (2 * eps)
            print(f"      eps={eps:.0e}:  FD(fp64 loss)={fd:+.4f}")


if __name__ == "__main__":
    main()
