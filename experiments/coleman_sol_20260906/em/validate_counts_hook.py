"""Same-run validation and timing for the intermediate-adjoint count hook."""
from __future__ import annotations

import argparse
from dataclasses import replace
import math
import time

import torch

from counts_hook import counts_and_gradient
from mstep import m_step
from gpurec.api.model import GeneReconModel
from gpurec.config import GpurecConfig
from gpurec.core.scheduling.batching import parse_families


def read_paths(path: str, limit: int) -> list[str]:
    with open(path) as handle:
        rows = [line.strip() for line in handle if line.strip() and not line.startswith("#")]
    return rows if limit == 0 else rows[:limit]


def timed(call):
    torch.cuda.synchronize()
    started = time.perf_counter()
    result = call()
    torch.cuda.synchronize()
    return result, time.perf_counter() - started


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--species", required=True)
    parser.add_argument("--families", required=True)
    parser.add_argument("--limit", required=True, type=int)
    parser.add_argument("--clade-budget", required=True, type=int)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    paths = read_paths(args.families, args.limit)
    config = GpurecConfig.genewise_reference()
    solver = replace(config.solver, pi_iters=16, neumann_terms=16)
    parsed = parse_families(args.species, paths)
    model = GeneReconModel(
        args.species, paths, mode="genewise", device="cuda", dtype=torch.float32,
        solver_options=solver, config=config, clade_budget=args.clade_budget,
        parsed_families=parsed, family_indices=list(range(len(paths))),
    )
    model.receiver_weights.requires_grad_(False)
    theta = torch.tensor(
        (math.log2(0.01), math.log2(0.1), math.log2(0.01)), dtype=torch.float64,
    ).reshape(1, 3).repeat(len(paths), 1)
    report = {}
    for point in ("start", "em1"):
        # Warm kernels once, then compare independent same-source evaluations.
        model.genewise_loss_vector_and_grad(
            theta=theta.to(device="cuda", dtype=torch.float32), need_grad=True,
        )
        (plain_nll, plain_g, _), plain_seconds = timed(lambda: model.genewise_loss_vector_and_grad(
            theta=theta.to(device="cuda", dtype=torch.float32), need_grad=True,
        ))
        (count_nll, count_g, counts), count_seconds = timed(lambda: counts_and_gradient(model, theta))
        plain_nll, plain_g = plain_nll.detach().double().cpu(), plain_g.detach().double().cpu()
        delta = count_g - plain_g
        row = {
            "plain_seconds": plain_seconds,
            "count_seconds": count_seconds,
            "time_ratio": count_seconds / plain_seconds,
            "gradient_max_abs": float(delta.abs().max()),
            "gradient_median_relative_l2": float((delta.norm(dim=1) / plain_g.norm(dim=1)).median()),
            "nll_max_abs": float((count_nll - plain_nll).abs().max()),
            "nonpositive_count_families": int((counts <= 0).any(dim=1).sum()),
            "count_min": float(counts.min()),
        }
        report[point] = row
        print(f"[hook] {point}: {row}", flush=True)
        theta = m_step(counts)[0]
    torch.save(report, args.out)
    print(f"[hook] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
