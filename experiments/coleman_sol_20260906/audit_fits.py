"""Compare two fitted Coleman parameter tensors using one fresh model.

Audit time is recorded separately from each fit's timing. The default uses the
same pruning setting as production; --pruning-threshold 0 requests an unpruned
audit without changing either optimizer's reported stopping rule.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import torch

from gpurec.api.model import GeneReconModel
from gpurec.config import GpurecConfig
from gpurec.optimization import project_rate_gradient_


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--species", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--candidate-repeat", help="independent repeated candidate fit")
    parser.add_argument("--repeat-baseline", action="store_true",
                        help="repeat the same baseline theta after the candidates to measure noise")
    parser.add_argument("--clade-budget", type=int, default=315000)
    parser.add_argument("--pruning-threshold", type=float, default=1e-6)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    baseline = torch.load(args.baseline, map_location="cpu", weights_only=False)
    candidate = torch.load(args.candidate, map_location="cpu", weights_only=False)
    paths = baseline["paths"]
    fits = [("baseline", baseline), ("candidate", candidate)]
    if args.candidate_repeat:
        fits.append(("candidate_repeat", torch.load(args.candidate_repeat, map_location="cpu", weights_only=False)))
    for name, fit in fits:
        if [str(Path(p).resolve()) for p in paths] != [str(Path(p).resolve()) for p in fit["paths"]]:
            raise ValueError(f"baseline and {name} family paths/orders differ")
        if fit["theta"].shape != (len(paths), 3) or not bool(torch.isfinite(fit["theta"]).all()):
            raise ValueError(f"invalid {name} theta shape or values")
    if args.repeat_baseline:
        fits.append(("baseline_repeat", baseline))
    config = GpurecConfig.genewise_reference()
    solver = replace(config.solver, pi_iters=16, neumann_terms=16,
                     adjoint_pruning_threshold=args.pruning_threshold)
    started = time.perf_counter()
    model = GeneReconModel(args.species, paths, mode="genewise", device="cuda",
                           config=config, solver_options=solver,
                           clade_budget=args.clade_budget)
    model.receiver_weights.requires_grad_(False)
    vectors = {}
    metrics = {}
    for name, fit in fits:
        theta = fit["theta"].to(device=model.theta.device, dtype=model.theta.dtype)
        values, gradient, _ = model.genewise_loss_vector_and_grad(theta=theta)
        projected = project_rate_gradient_(theta, gradient.clone(), bounds=config.rates)
        pg = projected.abs().amax(dim=1)
        if not bool(torch.isfinite(values).all()) or not bool(torch.isfinite(gradient).all()):
            raise RuntimeError(f"non-finite {name} audit likelihood or gradient")
        vectors[name] = {"nll": values.detach().double().cpu(),
                         "pg": pg.detach().double().cpu()}
        metrics[name] = {"nll_bits": float(values.double().sum()),
                         "cold_certified": int((pg < 1e-3).sum()),
                         "cold_pg_max": float(pg.max())}
    difference = vectors["candidate"]["nll"] - vectors["baseline"]["nll"]
    changed = (difference.abs() > 0.01).nonzero(as_tuple=True)[0].tolist()
    record = {
        "n_families": len(paths), "pruning_threshold": args.pruning_threshold,
        "baseline_path": args.baseline, "candidate_path": args.candidate,
        "metrics": metrics,
        "nll_difference_bits": float(difference.sum()),
        "max_regression_bits": max(0.0, float(difference.max())),
        "max_improvement_bits": max(0.0, float(-difference.min())),
        "regressions_above_0_01_bits": int((difference > 0.01).sum()),
        "improvements_above_0_01_bits": int((difference < -0.01).sum()),
        "changed_families": [
            {"index": i, "name": Path(paths[i]).name,
             "nll_difference_bits": float(difference[i])} for i in changed
        ],
        "audit_seconds": time.perf_counter() - started,
        "solver": {"pi_iters": solver.pi_iters, "neumann_terms": solver.neumann_terms,
                   "e_max_iter": solver.e_max_iter, "e_tol": solver.e_tol,
                   "e_adjoint_tol": solver.e_adjoint_tol},
        "clade_budget": args.clade_budget, "theta_dtype": str(model.theta.dtype),
        "accumulator_dtype": str(model.accumulator_dtype),
        "torch_version": torch.__version__, "gpu": torch.cuda.get_device_name(),
    }
    if "baseline_repeat" in vectors:
        noise = vectors["baseline_repeat"]["nll"] - vectors["baseline"]["nll"]
        record["baseline_same_theta_repeat"] = {
            "nll_difference_bits": float(noise.sum()), "max_family_abs_bits": float(noise.abs().max()),
            "max_pg_change": float((vectors["baseline_repeat"]["pg"] - vectors["baseline"]["pg"]).abs().max()),
        }
    if "candidate_repeat" in vectors:
        repeat_difference = vectors["candidate_repeat"]["nll"] - vectors["candidate"]["nll"]
        record["candidate_fit_repeat"] = {
            "path": args.candidate_repeat, "nll_difference_bits": float(repeat_difference.sum()),
            "max_family_abs_bits": float(repeat_difference.abs().max()),
            "families_above_0_01_bits": int((repeat_difference.abs() > 0.01).sum()),
        }
    Path(args.out).write_text(json.dumps(record, indent=2) + "\n")
    torch.save({"paths": paths, "vectors": vectors}, str(Path(args.out).with_suffix(".pt")))
    print(json.dumps(record), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
