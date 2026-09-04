"""Analytic Hessian-vector products vs central finite differences of the analytic gradient (fp64).

The gradient twin of this probe is ``corner_fd_grad.py``; this one goes one derivative further.
For a single family at one rate corner it builds the same fp64 genewise model, evaluates the three
analytic Hessian-vector products the genewise recipe uses (``_analytic_hessian`` in
gpurec/fit/genewise_fit.py: one probe per unit rate direction, D then L then T, through
``make_exact_hvp_single``), and compares each with the central finite difference of the ANALYTIC
gradient along the same direction:

    column j of the finite-difference matrix = (grad(theta + h e_j) - grad(theta - h e_j)) / (2 h)

so entry (i, j) of either matrix is d2 NLL / d theta_i d theta_j. Both are printed as 3x3 blocks,
along with the largest relative disagreement and how far each is from symmetric.

Why finite-difference the analytic gradient rather than the loss twice: a second difference of the
loss loses half the working precision, whereas one difference of a gradient that is itself good to
about 1e-4 leaves the comparison limited only by the step's own truncation error.

Usage: corner_fd_hvp.py --species S --families LIST --corner=D,L,T --step 1e-3 --adjoint exact|series
       [--time-probe 1]

``--time-probe 1`` additionally times one Hessian-vector probe on its own (median of a few repeats,
after one warm-up) and reports its peak CUDA allocation, which is what the Newton fit pays three of
per curvature refresh.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time

import torch


def _format_matrix(name, matrix):
    lines = [f"[hvp] {name}"]
    for row in range(3):
        lines.append(
            "[hvp]   "
            + "  ".join(f"{float(matrix[row, col]):+.9e}" for col in range(3))
        )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", required=True)
    ap.add_argument("--families", required=True)
    ap.add_argument("--corner", required=True)
    ap.add_argument("--step", required=True, type=float)
    ap.add_argument("--adjoint", required=True, choices=("exact", "series"))
    ap.add_argument("--time-probe", type=int, choices=(0, 1), default=0,
                    help="also time one Hessian-vector probe (0/1)")
    ap.add_argument("--time-repeats", type=int, default=5,
                    help="how many timed probes to take the median of")
    args = ap.parse_args()

    from gpurec.api.model import GeneReconModel
    from gpurec.api.solver_options import SolverOptions
    from gpurec.fit.genewise_fit import _BASE_SOLVER
    from gpurec.solver.hvp.exact import make_exact_hvp_single
    from gpurec.solver.value_and_grad import forward_solve

    paths = [ln.strip() for ln in open(args.families) if ln.strip() and not ln.startswith("#")]
    # Same solver recipe as corner_fd_grad.py, so the two probes measure the same code path.
    so = SolverOptions(**{**_BASE_SOLVER, "pi_iters": 16, "neumann_terms": 512,
                          "forward_self_loop": "exact", "adjoint_self_loop": args.adjoint})
    m = GeneReconModel(args.species, paths, mode="genewise", device="cuda", dtype=torch.float64,
                       solver_options=so, clade_budget=None)
    m.receiver_weights.requires_grad_(False)
    d, l, t = (float(x) for x in args.corner.split(","))
    G = len(paths)
    theta0 = torch.tensor([d, l, t], dtype=torch.float64, device="cuda").repeat(G, 1).contiguous()
    receiver_weights = m.receiver_weights.detach()

    loss, grad, _ = m.genewise_loss_vector_and_grad(theta=theta0, need_grad=True)
    print(f"[hvp] corner D={d} L={l} T={t} adjoint={args.adjoint} families={G} "
          f"NLL={float(loss.sum()):.6f}", flush=True)
    print("[hvp] analytic gradient  "
          + "  ".join(f"{float(grad[0, k]):+.9e}" for k in range(3)), flush=True)

    # --- analytic Hessian-vector products, exactly as _analytic_hessian drives them ---
    # ``pi_iters`` is the tangent self-loop iteration count the fit passes as ``pi_cur``.
    pi_iters = int(so.pi_iters)
    analytic = torch.zeros(3, 3, dtype=torch.float64, device="cuda")
    for static in m.batch_statics:
        fam = static.family_index_tensor.to(theta0.device)
        G_b = int(fam.numel())
        theta_b = theta0.index_select(0, fam).contiguous()
        _loss_b, sv = forward_solve([static], theta0, receiver_weights)
        hvp = make_exact_hvp_single(static, theta_b, receiver_weights, sv,
                                    tangent_self_iters=pi_iters)
        for j in range(3):
            u_b = torch.zeros(G_b, 3, dtype=torch.float64, device=theta0.device)
            u_b[:, j] = 1.0
            out_b = hvp(u_b.reshape(-1), probe_id=j)[: G_b * 3].reshape(G_b, 3)
            analytic[:, j] += out_b.sum(dim=0).double()
        del hvp, sv

    # --- central finite differences of the analytic gradient ---
    finite = torch.zeros(3, 3, dtype=torch.float64, device="cuda")
    for j in range(3):
        e = torch.zeros_like(theta0)
        e[:, j] = args.step
        _, grad_plus, _ = m.genewise_loss_vector_and_grad(theta=theta0 + e, need_grad=True)
        _, grad_minus, _ = m.genewise_loss_vector_and_grad(theta=theta0 - e, need_grad=True)
        finite[:, j] = ((grad_plus - grad_minus).double().sum(dim=0)) / (2 * args.step)

    print(_format_matrix("analytic Hessian-vector products (columns: probe along D, L, T)",
                         analytic), flush=True)
    print(_format_matrix("central finite differences of the analytic gradient", finite),
          flush=True)

    scale = finite.abs().max().clamp_min(1e-300)
    worst = float(((analytic - finite).abs() / scale).max())
    analytic_asymmetry = float((analytic - analytic.T).abs().max() / analytic.abs().max().clamp_min(1e-300))
    finite_asymmetry = float((finite - finite.T).abs().max() / scale)
    print(f"[hvp] largest entry difference relative to the largest finite-difference entry: "
          f"{worst:.3e}", flush=True)
    print(f"[hvp] asymmetry (|M - M^T| max, relative to |M| max): analytic {analytic_asymmetry:.3e}  "
          f"finite differences {finite_asymmetry:.3e}", flush=True)

    if args.time_probe:
        static = m.batch_statics[0]
        fam = static.family_index_tensor.to(theta0.device)
        G_b = int(fam.numel())
        theta_b = theta0.index_select(0, fam).contiguous()
        _loss_b, sv = forward_solve([static], theta0, receiver_weights)
        hvp = make_exact_hvp_single(static, theta_b, receiver_weights, sv,
                                    tangent_self_iters=pi_iters)
        u_b = torch.zeros(G_b, 3, dtype=torch.float64, device=theta0.device)
        u_b[:, 0] = 1.0
        hvp(u_b.reshape(-1), probe_id=0)  # warm up: compile the kernels, fill the caches
        torch.cuda.synchronize()
        seconds = []
        torch.cuda.reset_peak_memory_stats()
        for _ in range(int(args.time_repeats)):
            start = time.perf_counter()
            hvp(u_b.reshape(-1), probe_id=0)
            torch.cuda.synchronize()
            seconds.append(time.perf_counter() - start)
        peak_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print(f"[hvp] one Hessian-vector probe: median {statistics.median(seconds):.4f} s "
              f"over {len(seconds)} runs (min {min(seconds):.4f}, max {max(seconds):.4f}); "
              f"peak CUDA allocated {peak_gib:.3f} GiB",
              flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
