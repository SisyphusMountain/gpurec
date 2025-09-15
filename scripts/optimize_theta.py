#!/usr/bin/env python3
"""
Optimize reconciliation parameters theta using Adam with implicit gradients.

This script uses the custom autograd op to compute the negative log-likelihood
and its implicit gradient w.r.t. theta, and performs Adam optimization with
warm-started fixed points between iterations.

Usage examples:
  - Single dataset directory containing sp.nwk and g.nwk:
      python scripts/optimize_theta.py -d tests/data/test_trees_1 --steps 200 --lr 0.2

  - Explicit species/gene tree paths:
      python scripts/optimize_theta.py -s tests/data/test_trees_1/sp.nwk -g tests/data/test_trees_1/g.nwk
"""

from __future__ import annotations

import argparse
from pathlib import Path
import math
import sys

import torch

from src.optimization.fixed_point_autograd import FixedPointProblem, fixed_point_nll
torch.set_float32_matmul_precision('high')

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Optimize theta with Adam using implicit gradients.")
    p.add_argument("--dataset", "-d", type=str, help="Directory containing sp.nwk and g.nwk", default=None)
    p.add_argument("--species", "-s", type=str, help="Species tree file (.nwk)", default=None)
    p.add_argument("--gene", "-g", type=str, help="Gene tree file (.nwk)", default=None)

    p.add_argument("--optimizer", choices=["adam", "lbfgs"], default="adam", help="Optimizer to use")
    p.add_argument("--steps", type=int, default=200, help="Max outer steps")
    p.add_argument("--lr", type=float, default=0.2, help="Learning rate (Adam/LBFGS initial step)")
    p.add_argument("--adam-eps", type=float, default=1e-8, help="Adam epsilon")
    p.add_argument("--tol-theta", type=float, default=1e-3, help="Infinity norm stopping tolerance on theta")
    p.add_argument("--seed", type=int, default=0, help="Random seed")

    p.add_argument("--init-delta", type=float, default=1e-3, help="Initial duplication rate")
    p.add_argument("--init-tau", type=float, default=1e-3, help="Initial transfer rate")
    p.add_argument("--init-lambda", dest="init_lambda", type=float, default=1e-3, help="Initial loss rate")

    p.add_argument("--e-max-iters", type=int, default=3000)
    p.add_argument("--pi-max-iters", type=int, default=3000)
    p.add_argument("--e-tol", type=float, default=1e-3)
    p.add_argument("--pi-tol", type=float, default=1e-3)

    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.dataset:
        d = Path(args.dataset)
        sp = str(d / "sp.nwk")
        g = str(d / "g.nwk")
    else:
        if not args.species or not args.gene:
            print("Error: provide either --dataset or both --species and --gene paths.")
            return 2
        sp = args.species
        g = args.gene

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    # Initialize problem and theta
    problem = FixedPointProblem(sp, g, device=device, dtype=dtype)
    init_rates = [max(args.init_delta, 1e-12), max(args.init_tau, 1e-12), max(args.init_lambda, 1e-12)]
    theta0 = torch.tensor([math.log(r) for r in init_rates], dtype=dtype, device=device)
    problem.init_from_theta(theta0)

    theta = torch.nn.Parameter(theta0.clone())
    last_theta = theta.detach().clone()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    if args.optimizer == "adam":
        opt = torch.optim.Adam([theta], lr=args.lr, eps=args.adam_eps)
        for it in range(1, args.steps + 1):
            opt.zero_grad(set_to_none=True)
            nlll = fixed_point_nll(theta, problem,
                                   e_max_iters=args.e_max_iters, pi_max_iters=args.pi_max_iters,
                                   e_tol=args.e_tol, pi_tol=args.pi_tol)
            nlll.backward()
            opt.step()

            diff = torch.max(torch.abs(theta.detach() - last_theta)).item()
            last_theta = theta.detach().clone()

            rates = torch.exp(theta.detach())
            print(f"[iter {it:03d}] nLL={float(nlll):.6f} | ||Δθ||∞={diff:.3e} | theta={theta.detach().tolist()} | rates={[float(x) for x in rates]}")
            if diff < args.tol_theta:
                break

        final_nlll = fixed_point_nll(theta, problem,
                                     e_max_iters=args.e_max_iters, pi_max_iters=args.pi_max_iters,
                                     e_tol=args.e_tol, pi_tol=args.pi_tol).item()
    else:
        # LBFGS with strong Wolfe line search; multiple closure calls per step
        opt = torch.optim.LBFGS([theta], lr=args.lr, max_iter=20, history_size=10, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad(set_to_none=True)
            loss = fixed_point_nll(theta, problem,
                                   e_max_iters=args.e_max_iters, pi_max_iters=args.pi_max_iters,
                                   e_tol=args.e_tol, pi_tol=args.pi_tol)
            loss.backward()
            return loss

        for it in range(1, args.steps + 1):
            nlll = opt.step(closure)
            diff = torch.max(torch.abs(theta.detach() - last_theta)).item()
            last_theta = theta.detach().clone()
            rates = torch.exp(theta.detach())
            print(f"[iter {it:03d}] nLL={float(nlll):.6f} | ||Δθ||∞={diff:.3e} | theta={theta.detach().tolist()} | rates={[float(x) for x in rates]}")
            if diff < args.tol_theta:
                break

        final_nlll = fixed_point_nll(theta, problem,
                                     e_max_iters=args.e_max_iters, pi_max_iters=args.pi_max_iters,
                                     e_tol=args.e_tol, pi_tol=args.pi_tol).item()
    torch.cuda.synchronize()
    end.record()
    total_time = start.elapsed_time(end) / 1000.0  # milliseconds to seconds
    print(f"Optimization done in {total_time:.1f} seconds.")
    print("\nFinal:")
    print(f"  theta={theta.detach().tolist()}")
    print(f"  rates={[float(x) for x in torch.exp(theta.detach())]}")
    print(f"  negative log-likelihood={final_nlll:.6f}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
