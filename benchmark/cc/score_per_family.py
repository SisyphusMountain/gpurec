"""Per-family NLL of ONE fitted theta, scored by the converged reference solver.

`compare_fit_thetas.py` already does this for a PAIR of fits. This script is the single-fit version,
for the runs where only one code was fitted (the full 10,869-family HOGENOM set) and the per-family
numbers are still needed for the AleRax cross-check. Same solver settings as `compare_fit_thetas.py`
so the two scripts' outputs are directly comparable.

Writes <out>.pt with `{"nll_bits": [F], "paths": [F], "theta": path}` -- NLL in bits (log2), which
`xcheck_alerax.py` turns into a log-likelihood in nats.

Usage:
  python benchmark/cc/score_per_family.py --species S --families LIST --theta FIT.pt \
      --clade-budget 900000 --pi-iters 64 --neumann-terms 64 --forward-self-loop exact --out OUT.pt
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
    parser.add_argument("--theta", required=True, help="a run_genewise.py .pt payload")
    parser.add_argument("--clade-budget", required=True, type=int)
    parser.add_argument("--pi-iters", required=True, type=int)
    parser.add_argument("--neumann-terms", required=True, type=int)
    parser.add_argument("--forward-self-loop", required=True, choices=("log", "linear", "exact"))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from gpurec.api.model import GeneReconModel
    from gpurec.api.solver_options import SolverOptions
    from gpurec.fit.genewise_fit import _BASE_SOLVER

    paths = [line.strip() for line in open(args.families)
             if line.strip() and not line.startswith("#")]
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
    print(f"[score] build {time.perf_counter() - start:.1f}s families={len(paths)}", flush=True)

    theta = _fitted_theta(args.theta, paths, "cuda", torch.float32)
    with torch.no_grad():
        nll = model.genewise_loss_vector_and_grad(theta=theta, need_grad=False)[0].detach()
    total_bits = float(nll.double().sum())
    print(f"[score] {args.theta}: total NLL = {total_bits:.4f} bits over {len(paths)} families "
          f"(score wall {time.perf_counter() - start:.1f}s)", flush=True)
    torch.save({"nll_bits": nll.cpu(), "paths": paths, "theta": args.theta}, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
