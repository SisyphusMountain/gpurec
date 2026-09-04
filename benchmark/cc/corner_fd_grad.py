"""Exact-path gradient vs central finite differences of the exact-path NLL (fp64), one corner.

Usage: corner_fd_grad.py --species S --families LIST --corner=D,L,T --step 1e-3 --adjoint exact|series
       [--check-receiver-weights 1]

With ``--check-receiver-weights 1`` it also finite-differences the RECEIVER-WEIGHT gradient: the
weights are one shared log2 logit per species, so the check nudges one species' logit by --step
and compares the central difference of the summed NLL with that species' analytic gradient entry.
The species checked are the ones with the largest analytic gradient magnitude, where the finite
difference is furthest above its own rounding noise. That gradient comes out of a different kernel
from the rate gradient (the transfer-receiver VJP), so a rate gradient that agrees says nothing
about it.
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
    # Off unless asked for: each checked species costs two more whole-dataset forward solves.
    ap.add_argument("--check-receiver-weights", type=int, choices=(0, 1), default=0,
                    help="also finite-difference the receiver-weight gradient (0/1)")
    ap.add_argument("--receiver-species", type=int, default=2,
                    help="how many species to check with --check-receiver-weights")
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
    loss, grad, grad_receiver = m.genewise_loss_vector_and_grad(theta=theta0, need_grad=True)
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

    if args.check_receiver_weights:
        if grad_receiver is None:
            raise SystemExit("[fd] the model returned no receiver-weight gradient to check")
        grad_receiver = grad_receiver.detach().double()
        weights = m.receiver_weights.detach().double()
        # The summed NLL is what a shared parameter's gradient differentiates.
        checked = torch.argsort(grad_receiver.abs(), descending=True)[: args.receiver_species]
        for species in checked.tolist():
            step_vector = torch.zeros_like(weights)
            step_vector[species] = args.step
            lp = m.genewise_loss_vector_and_grad(
                theta=theta0, receiver_weights=weights + step_vector, need_grad=False
            )[0].detach().double().sum()
            lm = m.genewise_loss_vector_and_grad(
                theta=theta0, receiver_weights=weights - step_vector, need_grad=False
            )[0].detach().double().sum()
            fd_receiver = (lp - lm) / (2 * args.step)
            analytic = float(grad_receiver[species])
            print(f"[fd]   receiver weight of species {species}: analytic {analytic:+.9e}  "
                  f"central FD {float(fd_receiver):+.9e}  rel diff "
                  f"{float((grad_receiver[species] - fd_receiver).abs() / fd_receiver.abs().clamp_min(1e-300)):.3e}",
                  flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
