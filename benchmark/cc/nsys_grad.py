"""Nsight-Systems capture of exactly one genewise loss+gradient call and one analytic-Hessian call.

Builds the model over ``--limit`` families with ``--clade-budget`` (same solver settings as the
production genewise recipe's first tier: pi_iters=16, neumann_terms=16, warm adjoint on, receiver
weights frozen), runs one warm-up loss+grad so Triton kernels are already compiled, then opens the
CUDA profiler and records NVTX ranges:

  "grad"             -- one ``genewise_loss_vector_and_grad(need_grad=True)``
  "hessian"          -- one ``gpurec.fit.genewise_fit._analytic_hessian`` (3 analytic-HVP probes),
                        selected by ``--hessian library``
  "hessian_streamed" -- the same curvature computed one batch at a time (forward solve + adjoint
                        cache built once per batch and shared by the 3 probes, tangent warm-start
                        cache dropped between batches), selected by ``--hessian streamed``. This is
                        the variant that fits on a many-batch model: the library one exhausts the
                        GPU at 500 families / clade_budget 315000 (measured).

Run under:
  nsys profile --trace=cuda,nvtx,osrt --sample=none --capture-range=cudaProfilerApi \
      --capture-range-end=stop -o OUT python -u benchmark/cc/nsys_grad.py \
      --species S.nwk --families LIST.txt --limit 500 --clade-budget 315000
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch


def _streamed_hessian(m, theta, pi_cur):
    """Per-batch streamed curvature: build the forward solve + adjoint point cache ONCE per batch and
    reuse it for the 3 probes, freeing between batches. Same [G,3,3] result as the library's
    ``_analytic_hessian``, but never holding more than one batch's HVP state at a time."""
    from gpurec.solver.hvp.exact import make_exact_hvp_single
    from gpurec.solver.value_and_grad import forward_solve, free_cuda_cache_if_tight

    G = theta.shape[0]
    dev, dtype = theta.device, theta.dtype
    rw = m.receiver_weights.detach()
    cols = [torch.zeros(G, 3, device=dev, dtype=dtype) for _ in range(3)]
    for static in m.batch_statics:
        fam = static.family_index_tensor.to(dev)
        G_b = int(fam.numel())
        theta_b = theta.index_select(0, fam).contiguous()
        _l, sv = forward_solve([static], theta, rw)
        hvp = make_exact_hvp_single(static, theta_b, rw, sv, tangent_self_iters=pi_cur)
        for j in range(3):
            u_b = torch.zeros(G_b, 3, device=dev, dtype=dtype)
            u_b[:, j] = 1.0
            out_b = hvp(u_b.reshape(-1))[: G_b * 3].reshape(G_b, 3)
            cols[j].index_add_(0, fam, out_b)
        del hvp, sv
        free_cuda_cache_if_tight()
    H = torch.stack(cols, dim=-1)
    return 0.5 * (H + H.transpose(1, 2))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", required=True)
    ap.add_argument("--families", required=True, help="listfile: one .ale path per line")
    ap.add_argument("--limit", required=True, type=int, help="0 = all families; N = first N of the list")
    ap.add_argument("--clade-budget", required=True, type=int, help="clades per batch (model clade_budget)")
    ap.add_argument("--hessian", required=True, choices=("library", "streamed"),
                    help="which curvature implementation to capture in the NVTX range")
    args = ap.parse_args()

    from gpurec.api.model import GeneReconModel
    from gpurec.api.solver_options import SolverOptions
    from gpurec.fit.genewise_fit import _BASE_SOLVER, _analytic_hessian

    paths = [ln.strip() for ln in open(args.families) if ln.strip() and not ln.startswith("#")]
    if args.limit > 0:
        paths = paths[: args.limit]

    pi_cur = 16
    solver_options = SolverOptions(**{**dict(_BASE_SOLVER), "pi_iters": pi_cur, "neumann_terms": 16})
    dev = torch.device("cuda")

    t0 = time.perf_counter()
    m = GeneReconModel(
        str(args.species), [str(p) for p in paths], mode="genewise",
        device=dev, dtype=torch.float32, solver_options=solver_options,
        clade_budget=args.clade_budget,
    )
    m.receiver_weights.requires_grad_(False)
    torch.cuda.synchronize()
    print(f"[nsys_grad] build {time.perf_counter() - t0:.2f}s  batches={len(m.batch_statics)} "
          f"families={len(m.families)}", flush=True)

    theta = torch.zeros(len(m.families), 3, device=dev, dtype=torch.float32)

    # Warm-up: compiles Triton kernels and populates warm starts, so the captured call is steady state.
    t0 = time.perf_counter()
    lv, g, _ = m.genewise_loss_vector_and_grad(theta=theta, need_grad=True)
    torch.cuda.synchronize()
    print(f"[nsys_grad] warm-up loss+grad {time.perf_counter() - t0:.2f}s", flush=True)
    del lv, g

    torch.cuda.cudart().cudaProfilerStart()

    torch.cuda.nvtx.range_push("grad")
    t0 = time.perf_counter()
    lv, g, _ = m.genewise_loss_vector_and_grad(theta=theta, need_grad=True)
    torch.cuda.synchronize()
    t_grad = time.perf_counter() - t0
    torch.cuda.nvtx.range_pop()
    print(f"[nsys_grad] grad {t_grad:.3f}s loss_sum={float(lv.sum()):.4f} "
          f"grad_absmax={float(g.abs().max()):.4f}", flush=True)
    del lv, g

    t_hess = None
    if args.hessian == "library":
        torch.cuda.nvtx.range_push("hessian")
        t0 = time.perf_counter()
        try:
            H = _analytic_hessian(m, theta, pi_cur)
            torch.cuda.synchronize()
            t_hess = time.perf_counter() - t0
            print(f"[nsys_grad] hessian {t_hess:.3f}s H000={float(H[0, 0, 0]):.4f}", flush=True)
            del H
        except torch.OutOfMemoryError as exc:
            print(f"[nsys_grad] hessian OUT OF MEMORY: {str(exc).splitlines()[0]}", flush=True)
        torch.cuda.nvtx.range_pop()
    else:
        torch.cuda.nvtx.range_push("hessian_streamed")
        t0 = time.perf_counter()
        try:
            H = _streamed_hessian(m, theta, pi_cur)
            torch.cuda.synchronize()
            t_hess = time.perf_counter() - t0
            print(f"[nsys_grad] hessian_streamed {t_hess:.3f}s H000={float(H[0, 0, 0]):.4f}", flush=True)
            del H
        except torch.OutOfMemoryError as exc:
            print(f"[nsys_grad] hessian_streamed OUT OF MEMORY: {str(exc).splitlines()[0]}", flush=True)
        torch.cuda.nvtx.range_pop()

    torch.cuda.cudart().cudaProfilerStop()
    torch.cuda.synchronize()

    print(f"[nsys_grad] captured grad={t_grad:.3f}s hessian[{args.hessian}]={t_hess}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
