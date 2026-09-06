"""Fresh, common-model audit of all matched post-EM continuation arms."""
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fits", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    saved = torch.load(args.fits, map_location="cpu", weights_only=False)
    shared = torch.load(saved["artifact"], map_location="cpu", weights_only=False)
    arms = saved["arms"]
    n = len(arms["native"]["theta"])
    paths = shared["paths"][:n]
    config = GpurecConfig.genewise_reference()
    solver = replace(config.solver, pi_iters=16, neumann_terms=16)
    started = time.perf_counter()
    model = GeneReconModel(
        shared["metadata"]["species"], paths, mode="genewise", device="cuda",
        config=config, solver_options=solver,
        clade_budget=int(shared["metadata"]["clade_budget"]),
    )
    model.receiver_weights.requires_grad_(False)
    vectors, metrics = {}, {}
    for name, arm in [*arms.items(), ("native_repeat", arms["native"])]:
        theta = arm["theta"].to(device=model.theta.device, dtype=model.theta.dtype)
        if theta.shape != (n, 3) or not bool(torch.isfinite(theta).all()):
            raise ValueError(f"invalid parameters for {name}")
        values, gradient, _ = model.genewise_loss_vector_and_grad(theta=theta)
        projected = project_rate_gradient_(theta, gradient.clone(), bounds=config.rates)
        pg = projected.abs().amax(dim=1)
        if not bool(torch.isfinite(values).all() & torch.isfinite(gradient).all()):
            raise RuntimeError(f"non-finite audit for {name}")
        vectors[name] = {"nll": values.detach().double().cpu(),
                         "pg": pg.detach().double().cpu()}
        metrics[name] = {"nll_bits": float(values.double().sum()),
                         "fresh_certified": int((pg < 1e-3).sum()),
                         "fresh_pg_max": float(pg.max())}
    differences = {}
    for name in vectors:
        if name == "native":
            continue
        delta = vectors[name]["nll"] - vectors["native"]["nll"]
        changed = (delta.abs() > 0.01).nonzero(as_tuple=True)[0].tolist()
        differences[name] = {
            "nll_change_bits": float(delta.sum()),
            "max_family_abs_bits": float(delta.abs().max()),
            "regressions_above_0_01": int((delta > 0.01).sum()),
            "improvements_above_0_01": int((delta < -0.01).sum()),
            "max_pg_change": float((vectors[name]["pg"] - vectors["native"]["pg"]).abs().max()),
            "changed_families": [{"index": i, "name": Path(paths[i]).name,
                                  "nll_change_bits": float(delta[i])} for i in changed],
        }
    torch.cuda.synchronize()
    report = {"fits": args.fits, "n_families": n,
              "pruning_threshold": solver.adjoint_pruning_threshold,
              "metrics": metrics, "differences_from_native": differences,
              "audit_seconds": time.perf_counter() - started,
              "note": "Separate fresh audit, not part of any fit time; original pruned FP32 recipe."}
    destination = Path(args.out)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    torch.save({"paths": paths, "vectors": vectors}, destination.with_suffix(".pt"))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
