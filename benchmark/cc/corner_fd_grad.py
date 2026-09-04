"""Exact-path gradient vs central finite differences of the exact-path NLL (fp64), one corner.

Usage: corner_fd_grad.py --species S --families LIST --corner=D,L,T --step 1e-3 --adjoint exact|series
"""
from __future__ import annotations

import argparse
import sys

import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", required=True); ap.add_argument("--families", required=True)
    ap.add_argument("--corner", required=True); ap.add_argument("--step", required=True, type=float)
    ap.add_argument("--adjoint", required=True, choices=("exact", "series"))
    args = ap.parse_args()
    from gpurec.api.model import GeneReconModel
    from gpurec.api.solver_options import SolverOptions
    from gpurec.fit.genewise_fit import _BASE_SOLVER
    paths = [ln.strip() for ln in open(args.families) if ln.strip() and not ln.startswith("#")]
    so = SolverOptions(**{**_BASE_SOLVER, "pi_iters": 16, "neumann_terms": 512,
                          "forward_self_loop": "exact", "adjoint_self_loop": args.adjoint})
    m = GeneReconModel(args.species, paths, mode="genewise", device="cuda", dtype=torch.float64,
                       solver_options=so, clade_budget=None)
    m.receiver_weights.requires_grad_(False)
    d, l, t = (float(x) for x in args.corner.split(","))
    theta0 = torch.tensor([d, l, t], dtype=torch.float64, device="cuda").repeat(len(paths), 1).contiguous()
    loss, grad, _ = m.genewise_loss_vector_and_grad(theta=theta0, need_grad=True)
    loss, grad = loss.detach().double(), grad.detach().double()
    print(f"[fd] corner D={d} L={l} T={t} adjoint={args.adjoint} families={len(paths)} NLL={float(loss.sum()):.6f}", flush=True)
    for k, name in enumerate(("D", "L", "T")):
        e = torch.zeros_like(theta0); e[:, k] = args.step
        lp = m.genewise_loss_vector_and_grad(theta=theta0 + e, need_grad=False)[0].detach().double()
        lm = m.genewise_loss_vector_and_grad(theta=theta0 - e, need_grad=False)[0].detach().double()
        fd = (lp - lm) / (2 * args.step)
        for f in range(len(paths)):
            print(f"[fd]   family {f} d/d{name}: analytic {float(grad[f, k]):+.9e}  central FD {float(fd[f]):+.9e}  "
                  f"rel diff {float((grad[f, k] - fd[f]).abs() / fd[f].abs().clamp_min(1e-300)):.3e}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
