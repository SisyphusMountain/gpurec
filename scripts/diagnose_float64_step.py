"""One BFGS/GD step from the 1.66 checkpoint with float64 accumulation of the family-sum.

The 1.66 floor is float32 quantization of the loss (a single float32 scalar ~358567,
ULP 0.031) and the gradient (float32 family reduction). The kernels are float32, so a
true end-to-end float64 step is not available without a kernel rewrite. The feasible,
faithful proxy: accumulate the per-batch losses and gradients in float64 (each batch
loss ~14000 has ULP ~0.001, so a float64 sum of 25 of them resolves ~30x finer than the
single float32 scalar). Then line-search along -g and measure how much the loss improves
and the gradient drops.

  - part 0: does the model even run in full float64? (likely no -- float32 kernels)
  - part 1: float32 vs float64-accumulated loss/gradient at theta*
  - part 2: line search along -g (float64 loss), take the step, re-measure |g|

Run:  python -u scripts/diagnose_float64_step.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import importlib.util  # noqa: E402
import torch  # noqa: E402

from gpurec import GeneReconModel, SolverOptions  # noqa: E402
from gpurec.api._execution import evaluate_static_loss_grad, theta_for_static  # noqa: E402

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
    return SolverOptions(e_max_iter=2000, e_tol=1e-8,
                         pi_iters=pi, neumann_terms=nt, self_loop_solver="neumann")


def pen_val(th, model, dtype):
    # Sanderson penalty; compute in the requested dtype (it is cheap, pure-torch, exact)
    return PENALTY_LAMBDA * archaea_opt.roughness_penalty(th.to(dtype), model.species_helpers)


def loss_grad_accum(model, theta, *, acc_dtype, need_grad):
    """Loss (and gradient) accumulated over batches in acc_dtype (float32 or float64).

    Each batch's loss/grad come from the float32 kernels; only the cross-batch
    reduction (and the final scalar) are done in acc_dtype. This isolates how much
    the family-sum precision (vs one float32 scalar) matters.
    """
    S3 = model.theta.numel()
    g = torch.zeros(model.theta.shape, dtype=acc_dtype, device=theta.device) if need_grad else None
    loss = torch.zeros((), dtype=acc_dtype, device=theta.device)
    for static in model.batch_statics:
        th = theta_for_static(static, theta, genewise=model.genewise)
        loss_i, grad_i, _ = evaluate_static_loss_grad(static, th, model.receiver_weights, need_grad=need_grad)
        loss = loss + loss_i.to(acc_dtype)
        if need_grad:
            g = g + grad_i.to(acc_dtype)
    # add penalty (data + penalty) in acc_dtype
    th_pen = theta.detach().to(acc_dtype).requires_grad_(True)
    p = PENALTY_LAMBDA * archaea_opt.roughness_penalty(th_pen, model.species_helpers)
    if need_grad:
        (p.backward() if p.requires_grad else None)
        if th_pen.grad is not None:
            g = g + th_pen.grad.to(acc_dtype)
    loss = loss + p.detach().to(acc_dtype)
    return loss, g


def main():
    fams, _ = archaea_opt.select_families(FAM_DIR, max_families=0, family_order="largest",
                                          min_leaves=4, recursive=False)
    model = GeneReconModel(SP_TREE, [str(f) for f in fams], mode="specieswise",
                           device=DEVICE, solver_options=sopts(64, 64))
    model.receiver_weights.requires_grad_(False)
    theta0 = torch.load(CKPT).to(DEVICE).float()
    print(f"loaded checkpoint  S={model.theta.shape[0]}  F={len(fams)}  batches={len(model.batch_statics)}")

    # --- part 0: does full float64 run? ---
    print("\n[0] try full float64 model forward:")
    try:
        model.theta.requires_grad_(True)
        with torch.no_grad():
            model.theta.copy_(theta0)
        l64 = model(theta=model.theta.double())
        print(f"   ran; loss dtype={l64.dtype}")
    except Exception as e:
        print(f"   NOT supported (float32 kernels): {type(e).__name__}: {str(e)[:80]}")

    # --- part 1: float32 vs float64-accumulated loss/grad at theta* ---
    print("\n[1] loss/gradient at theta*, accumulated in float32 vs float64:")
    L32, g32 = loss_grad_accum(model, theta0, acc_dtype=torch.float32, need_grad=True)
    L64, g64 = loss_grad_accum(model, theta0, acc_dtype=torch.float64, need_grad=True)
    print(f"   float32 acc: loss={float(L32):.6f}  |g|inf={float(g32.abs().max()):.6e}")
    print(f"   float64 acc: loss={float(L64):.6f}  |g|inf={float(g64.abs().max()):.6e}")
    print(f"   loss ULP(float32 scalar @ {float(L64):.0f}) ~= {2**(int(torch.log2(L64).item())-23):.4f}")
    print(f"   |g64 - g32|inf = {float((g64 - g32.double()).abs().max()):.3e}  (cross-batch fp64 effect on gradient)")

    # --- part 2: one line-searched step along -g64, measured with the float64 loss ---
    print("\n[2] one BFGS/GD step along -g (line search on the float64-accumulated loss):")
    d = -g64
    best = (0.0, L64.item(), None)
    for alpha in (3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2):
        L, _ = loss_grad_accum(model, (theta0.double() + alpha * d).float(), acc_dtype=torch.float64, need_grad=False)
        dl = L.item() - L64.item()
        mark = "  <-- improves" if dl < 0 else ""
        print(f"   alpha={alpha:.0e}:  dloss(fp64)={dl:+.5f}{mark}")
        if L.item() < best[1]:
            best = (alpha, L.item(), None)
    a = best[0]
    print(f"   best alpha={a:.0e}  dloss(fp64)={best[1]-L64.item():+.5f}")

    if a > 0:
        theta_new = (theta0.double() + a * d).float()
        Ln, gn = loss_grad_accum(model, theta_new, acc_dtype=torch.float64, need_grad=True)
        print("\n[result] after the step:")
        print(f"   loss:   {float(L64):.5f} -> {float(Ln):.5f}   (dloss={float(Ln-L64):+.5f})")
        print(f"   |g|inf: {float(g64.abs().max()):.6e} -> {float(gn.abs().max()):.6e}")
        print(f"   |g|rms: {float(g64.norm()/g64.numel()**0.5):.6e} -> {float(gn.norm()/gn.numel()**0.5):.6e}")
    else:
        print("\n[result] no alpha reduced even the float64 loss -> within-batch float32 kernel is the wall")

    print("\nNOTE: only the CROSS-BATCH reduction is float64 here; the within-batch family")
    print("reduction is still float32 (in the kernels). A full test needs float64 in-kernel.")


if __name__ == "__main__":
    main()
