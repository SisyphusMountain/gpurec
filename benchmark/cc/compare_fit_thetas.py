"""Per-family NLL of two fitted thetas, scored by ONE common solver.

When two fits of the same families end at different total NLL, this says whether the gap is spread
over the whole set (a systematic difference) or carried by a handful of families that took
different optimizer trajectories. Both thetas are scored by the same solver settings, so the
comparison is about where the fits LANDED, not about how they were evaluated.

Usage:
  python benchmark/cc/compare_fit_thetas.py --species S --families LIST \
      --left $CC_RUNS/results/exact500.pt --right $CC_RUNS/results/exactadj500.pt \
      --clade-budget 315000 --pi-iters 64 --neumann-terms 64 --forward-self-loop exact --top 10
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_exact_forward import _fitted_theta  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--species", required=True)
    parser.add_argument("--families", required=True)
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--clade-budget", required=True, type=int)
    parser.add_argument("--pi-iters", required=True, type=int)
    parser.add_argument("--neumann-terms", required=True, type=int)
    parser.add_argument("--forward-self-loop", required=True, choices=("log", "exact"))
    parser.add_argument("--top", required=True, type=int)
    args = parser.parse_args()

    from gpurec.api.model import GeneReconModel
    from gpurec.api.solver_options import SolverOptions
    from gpurec.fit.genewise_fit import _BASE_SOLVER

    left_payload = torch.load(args.left, map_location="cpu")
    paths = [str(p) for p in left_payload["paths"]]
    options = SolverOptions(**{
        **_BASE_SOLVER,
        "pi_iters": args.pi_iters,
        "neumann_terms": args.neumann_terms,
        "forward_self_loop": args.forward_self_loop,
        "adjoint_self_loop": "series",
    })
    start = time.perf_counter()
    model = GeneReconModel(
        args.species, paths, mode="genewise", device="cuda", dtype=torch.float32,
        solver_options=options, clade_budget=args.clade_budget,
    )
    model.receiver_weights.requires_grad_(False)
    print(f"[cmp] build {time.perf_counter() - start:.1f}s families={len(paths)}", flush=True)

    left = _fitted_theta(args.left, paths, "cuda", torch.float32)
    right = _fitted_theta(args.right, paths, "cuda", torch.float32)
    with torch.no_grad():
        left_nll = model.genewise_loss_vector_and_grad(theta=left, need_grad=False)[0].detach()
        right_nll = model.genewise_loss_vector_and_grad(theta=right, need_grad=False)[0].detach()
    difference = right_nll - left_nll
    # Distribution of the per-family differences, and the raw vectors for later inspection.
    d = difference.detach().double().cpu()
    for thr in (0.01, 0.1, 0.5, 1.0, 2.0):
        print(f"[cmp] |d(NLL)| > {thr:g} bits: {int((d.abs() > thr).sum())} families "
              f"(worse: {int((d > thr).sum())}, better: {int((d < -thr).sum())})", flush=True)
    print(f"[cmp] sum of worsenings {float(d[d > 0].sum()):.3f} bits, sum of improvements {float(d[d < 0].sum()):.3f} bits, "
          f"max worsening {float(d.max()):.3f}, max improvement {float(d.min()):.3f}", flush=True)
    torch.save({"left_nll": left_nll.detach().cpu(), "right_nll": right_nll.detach().cpu(),
                "left": args.left, "right": args.right}, args.right + ".cmp_nll.pt")
    total = float(difference.sum().item())
    order = torch.argsort(difference.abs(), descending=True)[: args.top]
    print(
        f"[cmp] total NLL difference (right - left) = {total:.4f} bits over {len(paths)} families; "
        f"left sum {float(left_nll.sum().item()):.4f}, right sum {float(right_nll.sum().item()):.4f}",
        flush=True,
    )
    carried = 0.0
    for rank, index in enumerate(order.tolist()):
        carried += float(difference[index].item())
        print(
            f"[cmp] rank {rank}: family row {index}  d(NLL) = {float(difference[index].item()):+.4f} bits  "
            f"({100.0 * carried / total:.2f}% of the total once ranks 0..{rank} are counted)  "
            f"left theta = {[round(float(x), 4) for x in left[index].tolist()]}  "
            f"right theta = {[round(float(x), 4) for x in right[index].tolist()]}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
