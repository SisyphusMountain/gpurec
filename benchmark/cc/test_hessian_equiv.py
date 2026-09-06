"""Check that the restructured per-family Hessian (batch-outer, 3 probes per batch) matches the
previous implementation (probe-outer, streaming HVP rebuilding each batch per probe) on a
multi-batch model. GPU required.

Usage: python benchmark/cc/test_hessian_equiv.py --species S --families LIST --limit N --clade-budget B
(pick a small clade budget so that N families give several batches)
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch


def _old_analytic_hessian(m, theta, pi_cur):
    """Verbatim copy of the previous implementation (commit 817007e6)."""
    from gpurec.solver.hvp.exact import make_exact_hvp
    from gpurec.solver.value_and_grad import forward_solve
    G = theta.shape[0]
    dev, dtype = theta.device, theta.dtype
    rw = m.receiver_weights.detach()
    if len(m.batch_statics) > 1:
        hvp = make_exact_hvp(m.batch_statics, theta, rw, None, tangent_self_iters=pi_cur)
        cols = []
        for j in range(3):
            u = torch.zeros(G, 3, device=dev, dtype=dtype); u[:, j] = 1.0
            cols.append(hvp(u.reshape(-1))[: G * 3].reshape(G, 3))
        H = torch.stack(cols, dim=-1)
    else:
        static = m.batch_statics[0]
        fam = static.family_index_tensor.to(dev)
        theta_b = theta.index_select(0, fam).contiguous()
        _l, sv = forward_solve(m.batch_statics, theta, rw)
        hvp = make_exact_hvp(m.batch_statics, theta_b, rw, sv, tangent_self_iters=pi_cur)
        cols = []
        for j in range(3):
            u_b = torch.zeros(G, 3, device=dev, dtype=dtype); u_b[:, j] = 1.0
            out_b = hvp(u_b.reshape(-1))[: G * 3].reshape(G, 3)
            col = torch.zeros(G, 3, device=dev, dtype=dtype)
            col.index_add_(0, fam, out_b)
            cols.append(col)
        H = torch.stack(cols, dim=-1)
    return 0.5 * (H + H.transpose(1, 2))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", required=True)
    ap.add_argument("--families", required=True)
    ap.add_argument("--limit", required=True, type=int)
    ap.add_argument("--clade-budget", required=True, type=int)
    args = ap.parse_args()
    paths = [ln.strip() for ln in open(args.families) if ln.strip() and not ln.startswith("#")]
    if args.limit > 0:
        paths = paths[: args.limit]

    from gpurec.api.model import GeneReconModel
    from gpurec.api.solver_options import SolverOptions
    from gpurec.fit.genewise_fit import _BASE_SOLVER, _analytic_hessian

    so = SolverOptions(**{**_BASE_SOLVER, "pi_iters": 16, "neumann_terms": 16})
    m = GeneReconModel(args.species, paths, mode="genewise", device="cuda", dtype=torch.float32,
                       solver_options=so, clade_budget=args.clade_budget)
    m.receiver_weights.requires_grad_(False)
    print(f"[test] {len(paths)} families in {len(m.batch_statics)} batches", flush=True)
    G = len(paths)
    torch.manual_seed(0)
    theta = (torch.full((G, 3), -6.0, device="cuda") + 0.5 * torch.randn(G, 3, device="cuda")).float()

    def timed(fn):
        torch.cuda.synchronize(); t = time.perf_counter(); out = fn(); torch.cuda.synchronize()
        return out, time.perf_counter() - t

    H_old, t_old0 = timed(lambda: _old_analytic_hessian(m, theta, 16))   # warm-up (compiles)
    H_new, t_new0 = timed(lambda: _analytic_hessian(m, theta, 16))
    H_old2, t_old = timed(lambda: _old_analytic_hessian(m, theta, 16))
    H_new2, t_new = timed(lambda: _analytic_hessian(m, theta, 16))
    d_impl = (H_old2 - H_new2).abs().max().item()
    d_rep_old = (H_old - H_old2).abs().max().item()
    d_rep_new = (H_new - H_new2).abs().max().item()
    scale = H_old2.abs().max().item()
    print(f"[test] max|H| = {scale:.4e}")
    print(f"[test] max|H_old - H_new| = {d_impl:.4e}  (relative {d_impl / scale:.2e})")
    print(f"[test] run-to-run: old {d_rep_old:.4e}, new {d_rep_new:.4e}")
    print(f"[test] time old {t_old:.2f}s, new {t_new:.2f}s  (first calls incl. compile: old {t_old0:.1f}s new {t_new0:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
