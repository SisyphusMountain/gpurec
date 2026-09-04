"""What is actually ill-conditioned at a rate corner where the exact path loses digits?

Reports, for one theta, the distribution of the two quantities the exact forward divides by -- the
row's smallest pivot and its ``1 - loop gain`` -- so a corner that loses digits without flagging a
single row can be attributed rather than guessed at.

Usage:
  python benchmark/cc/probe_exact_margins.py --species S --families LIST --limit 20 \
      --clade-budget 315000 --pi-iters 16 --neumann-terms 16 --rates 1,1,1
"""
from __future__ import annotations

import argparse
import sys

import torch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--species", required=True)
    parser.add_argument("--families", required=True)
    parser.add_argument("--limit", required=True, type=int)
    parser.add_argument("--clade-budget", required=True, type=int)
    parser.add_argument("--pi-iters", required=True, type=int)
    parser.add_argument("--neumann-terms", required=True, type=int)
    parser.add_argument("--rates", required=True, help="comma-separated log2 D,L,T")
    args = parser.parse_args()

    from gpurec.api.model import GeneReconModel
    from gpurec.api.solver_options import SolverOptions
    from gpurec.core.inference import solver as solver_module
    from gpurec.core.kernels.pi_forward import exact_conditioning_floor
    from gpurec.fit.genewise_fit import _BASE_SOLVER

    paths = [
        line.strip() for line in open(args.families) if line.strip() and not line.startswith("#")
    ][: args.limit]
    options = SolverOptions(**{
        **_BASE_SOLVER, "pi_iters": args.pi_iters, "neumann_terms": args.neumann_terms,
        "forward_self_loop": "exact", "adjoint_self_loop": "exact",
    })
    model = GeneReconModel(
        args.species, paths, mode="genewise", device="cuda", dtype=torch.float32,
        solver_options=options, clade_budget=args.clade_budget,
    )
    model.receiver_weights.requires_grad_(False)
    rates = [float(v) for v in args.rates.split(",")]
    theta = torch.tensor(rates, device="cuda", dtype=torch.float32)
    theta = theta.expand(len(paths), 3).contiguous()

    original = solver_module.pi_wave_forward
    captured: list[torch.Tensor] = []

    def probed(**kwargs):
        rows = int(kwargs["wave_layout"]["leaf_species_index"].numel())
        trips = torch.zeros((rows, 4), device=kwargs["e"].device, dtype=kwargs["e"].dtype)
        kwargs["exact_guard_trips_out"] = trips
        result = original(**kwargs)
        captured.append(trips)
        return result

    solver_module.pi_wave_forward = probed
    try:
        for static in model.batch_statics:
            with torch.no_grad():
                solver_module.solve_resident_e_pi(
                    static, model._theta_for_static(static, theta),
                    model.receiver_weights.detach(), warm_start_E=None, pi_iters=None,
                    pi_residual_out=None,
                )
    finally:
        solver_module.pi_wave_forward = original

    trips = torch.cat(captured, dim=0).double()
    floor = exact_conditioning_floor(torch.float32)
    print(f"[probe] rates D,L,T = {rates}, {int(trips.shape[0])} clade rows, "
          f"conditioning floor = {floor:.3e}", flush=True)
    for column, label in ((2, "smallest pivot"), (3, "1 - loop gain")):
        values = trips[:, column]
        # A row the kernel returned from early never wrote these and stays at the 0.0 it was
        # allocated with; the rows that did write are what a distribution should describe.
        wrote = values != 0.0
        subset = values[wrote]
        if subset.numel() == 0:
            print(f"[probe] {label}: no row reached the closure", flush=True)
            continue
        quantiles = torch.tensor([0.0, 0.001, 0.01, 0.5], dtype=torch.float64, device=values.device)
        q = torch.quantile(subset, quantiles).tolist()
        print(
            f"[probe] {label}: rows={int(wrote.sum())} min={q[0]:.4e} "
            f"0.1%={q[1]:.4e} 1%={q[2]:.4e} median={q[3]:.4e} "
            f"under floor={int((subset < floor).sum())}",
            flush=True,
        )
    print(f"[probe] non-positive pivot count total = {int(trips[:, 0].sum())}, "
          f"non-positive loop gain rows = {int(trips[:, 1].sum())}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
