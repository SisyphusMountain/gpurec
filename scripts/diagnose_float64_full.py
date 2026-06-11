"""Does the model compute in TRUE float64 (kernels), or cast to float32 internally?

Part 0 of diagnose_float64_step showed model(theta.double()) runs and returns a float64
loss. Decisive test: run the full float64 forward+backward at theta* and compare the
gradient to the float32 one. If |g64| differs from |g32|=1.614 and is reproducible, the
kernels do real float64 work -> a true float64 LBFGS is the lever. If |g64| ~= 1.614, the
kernels cast to float32 internally and only an in-kernel float64 rewrite can help.

Then: one line-searched step along -g64 using the full-float64 loss -> does |g| go down?

Run:  python -u scripts/diagnose_float64_full.py
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


def loss_grad(model, theta_value, dtype):
    th = theta_value.to(dtype).detach().requires_grad_(True)
    loss = model(theta=th) + PENALTY_LAMBDA * archaea_opt.roughness_penalty(th, model.species_helpers)
    loss.backward()
    return loss.detach(), th.grad.detach()


def loss_only(model, theta_value, dtype):
    with torch.no_grad():
        return float(model(theta=theta_value.to(dtype))
                     + PENALTY_LAMBDA * archaea_opt.roughness_penalty(theta_value.to(dtype), model.species_helpers))


def main():
    fams, _ = archaea_opt.select_families(FAM_DIR, max_families=0, family_order="largest",
                                          min_leaves=4, recursive=False)
    model = GeneReconModel(SP_TREE, [str(f) for f in fams], mode="specieswise",
                           device=DEVICE, solver_options=sopts(64, 64))
    model.receiver_weights.requires_grad_(False)
    theta = torch.load(CKPT).to(DEVICE)
    print(f"loaded checkpoint  S={model.theta.shape[0]}  F={len(fams)}")

    print("\n[1] full-pipeline gradient at theta*, float32 vs float64:")
    L32, g32 = loss_grad(model, theta, torch.float32)
    L64, g64 = loss_grad(model, theta, torch.float64)
    print(f"   float32: loss={float(L32):.6f}  |g|inf={float(g32.abs().max()):.6e}  dtype={g32.dtype}")
    print(f"   float64: loss={float(L64):.6f}  |g|inf={float(g64.abs().max()):.6e}  dtype={g64.dtype}")
    diff = float((g64 - g32.double()).abs().max())
    print(f"   |g64 - g32|inf = {diff:.3e}")
    if diff < 1e-2:
        print("   => kernels effectively cast to float32 (gradient ~unchanged): in-kernel float64 needed.")
    else:
        print("   => float64 changes the gradient: the kernels do real float64 work.")

    print("\n[2] one step along -g64 using the full-float64 loss (does |g| go down?):")
    L0 = float(L64)
    best = (0.0, L0)
    for alpha in (1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2):
        L = loss_only(model, theta.double() - alpha * g64, torch.float64)
        dl = L - L0
        mark = "  <-- improves" if dl < 0 else ""
        print(f"   alpha={alpha:.0e}:  dloss(fp64)={dl:+.6f}{mark}")
        if L < best[1]:
            best = (alpha, L)
    a = best[0]
    if a > 0:
        theta_new = theta.double() - a * g64
        Ln, gn = loss_grad(model, theta_new, torch.float64)
        print(f"\n[result] best alpha={a:.0e}")
        print(f"   loss:   {L0:.6f} -> {float(Ln):.6f}   (dloss={float(Ln)-L0:+.6f})")
        print(f"   |g|inf: {float(g64.abs().max()):.6e} -> {float(gn.abs().max()):.6e}")
        print(f"   |g|rms: {float(g64.norm()/g64.numel()**0.5):.6e} -> {float(gn.norm()/gn.numel()**0.5):.6e}")
    else:
        print("\n[result] no alpha reduced the full-float64 loss either.")


if __name__ == "__main__":
    main()
