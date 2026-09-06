"""Run native and hierarchical continuations from the point-consistent shared EM2 artifact."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from gpurec.config import GpurecConfig
from gpurec.fit.genewise_fit import TRUST_TEST_OFF

from source_adapter import (
    PRODUCTION_SOURCE_SHA256,
    fit_genewise_hierarchical,
    fit_genewise_native,
)


def _summary(result: dict, wall_seconds: float, paths: list[str]) -> dict:
    keys = (
        "loss_bits", "n_steps", "n_builds", "n_hessians", "n_rebuilds", "converged",
        "unconverged", "bound_active", "pg_max", "newton_grad_seconds", "hessian_seconds",
        "rebuild_seconds", "certify_seconds",
    )
    row = {key: result.get(key) for key in keys}
    row["wall_seconds"] = wall_seconds
    row["gradient_work"] = result["gradient_work"]
    row["gradient_clades"] = sum(int(item["clades"]) for item in result["gradient_work"])
    row["theta_finite"] = bool(torch.isfinite(result["theta"]).all())
    row["curvature_finite"] = bool(torch.isfinite(result["curvature"]).all())
    row["theta_min"] = float(result["theta"].min())
    row["theta_max"] = float(result["theta"].max())
    diagnostics = result.get("hierarchical_ray_diagnostics")
    if diagnostics is not None:
        zero = diagnostics["zero_step_count"].detach().cpu()
        nearzero = diagnostics["nearzero_step_count"].detach().cpu()
        boundary = diagnostics["boundary_zero_count"].detach().cpu()
        zero_direction = diagnostics["zero_direction_nonconverged_count"].detach().cpu()
        native_zero = diagnostics["actual_native_zero_nonconverged_count"].detach().cpu()
        row["ray_diagnostics"] = {
            "zero_steps": int(zero.sum()),
            "nearzero_steps": int(nearzero.sum()),
            "boundary_zero_steps": int(boundary.sum()),
            "zero_direction_nonconverged_steps": int(zero_direction.sum()),
            "actual_native_zero_nonconverged_steps": int(native_zero.sum()),
            "zero_family_ids": [paths[i] for i in (zero > 0).nonzero(as_tuple=True)[0].tolist()],
            "nearzero_family_ids": [paths[i] for i in (nearzero > 0).nonzero(as_tuple=True)[0].tolist()],
            "boundary_zero_family_ids": [
                paths[i] for i in (boundary > 0).nonzero(as_tuple=True)[0].tolist()
            ],
            "zero_direction_family_ids": [
                paths[i] for i in (zero_direction > 0).nonzero(as_tuple=True)[0].tolist()
            ],
            "actual_native_zero_family_ids": [
                paths[i] for i in (native_zero > 0).nonzero(as_tuple=True)[0].tolist()
            ],
            "min_ray_fraction": float(diagnostics["min_ray_fraction"].min()),
        }
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", default="experiments/coleman_sol_20260906/em/hybrid_shared_200_v2.pt")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--max-iter", type=int, default=6)
    parser.add_argument("--check-every", type=int, default=2)
    parser.add_argument("--pi-tiers", default="16,64")
    parser.add_argument("--certify", action="store_true")
    parser.add_argument("--out", required=True)
    parser.add_argument("--tensor-out")
    parser.add_argument(
        "--arms", default="native,hierarchical_native_metric,hierarchical_coordinate_metric",
        help="comma-separated arm names; order is execution order",
    )
    args = parser.parse_args()

    artifact = torch.load(args.artifact, map_location="cpu", weights_only=False)
    if artifact.get("schema") != "gpurec.em_hybrid_shared.v2":
        raise ValueError(f"point-consistent V2 artifact required, got {artifact.get('schema')!r}")
    n = min(args.limit, len(artifact["paths"]))
    paths = artifact["paths"][:n]
    species = artifact["metadata"]["species"]
    theta2 = artifact["theta_native"]["theta2"][:n].float()
    native_seed = artifact["native_control"]["consistent_evaluated_points"]["seed"][:n].float()
    hierarchical_seed = artifact["hierarchical"]["scaled_bfgs_seed_z"][:n].float()
    lo, hi = artifact["metadata"]["rate_box_log2"]
    budget = int(artifact["metadata"]["clade_budget"])
    config = GpurecConfig.genewise_reference()
    pi_tiers = tuple(int(item) for item in args.pi_tiers.split(","))

    common = dict(
        device="cuda", dtype="float32", min_rate=2.0 ** lo, max_rate=2.0 ** hi,
        adam_steps=0, adam_lr=1.0, warmup_method="adam", em_steps=2,
        grad_clip=10.0, pi_tiers=pi_tiers, neu_opt=16, neu_cert=64, clade_budget=budget,
        tol=1e-3, max_iter=args.max_iter, check_every=args.check_every, drop_frac=0.05,
        min_drop=32, rebuild_frac=0.25, hessian_refresh=15, curvature_update="bfgs",
        trust=2.0, trust_max=8.0, mu=1e-4, fwd_tol=1e-3, improve_frac=0.8,
        verify_drop=True, eager_defer=True, certify=args.certify, certify_curvature=False,
        init_log2_rates=theta2, stall_patience=120, step_extrapolation=1.0,
        step_model="quadratic", stop_nll_bits=0.0, approach_pruning_threshold=0.0,
        targeted_hessian=(0, 0.0), coordinate_staging=(0, 0), trust_test=TRUST_TEST_OFF,
        solver_options=None, config=config, verbose=True,
    )
    # Dict construction is written explicitly so CLI order, not declaration order, controls the
    # warmed comparison.
    available_arms = {
        "native": (fit_genewise_native, native_seed, {}),
        "hierarchical_native_metric": (
            fit_genewise_hierarchical, hierarchical_seed, {"hierarchical_trust_metric": "native"}),
        "hierarchical_coordinate_metric": (
            fit_genewise_hierarchical, hierarchical_seed, {"hierarchical_trust_metric": "coordinate"}),
    }
    arm_names = args.arms.split(",")
    unknown = [name for name in arm_names if name not in available_arms]
    if unknown:
        raise ValueError(f"unknown arms: {unknown}; choose from {list(available_arms)}")
    shared_em_work = [
        {"phase": "em", "step": step, "families": n,
         "clades": int(artifact["family_clades"][:n].sum())}
        for step in range(int(artifact["gradient_calls"]))
    ]
    output = {
        "schema": "gpurec.hierarchical_shared_endpoint.v1",
        "artifact": args.artifact,
        "n_families": n,
        "max_iter": args.max_iter,
        "certify": args.certify,
        "production_source_sha256": PRODUCTION_SOURCE_SHA256,
        "charged_shared_em_gradient_work": shared_em_work,
        "common_em_timing_full_artifact": artifact["timing"],
        "arms": {},
    }
    arm_tensors = {}
    for name in arm_names:
        fit, seed, extra = available_arms[name]
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = fit(species, paths, init_curvature=seed, **common, **extra)
        torch.cuda.synchronize()
        output["arms"][name] = _summary(result, time.perf_counter() - started, paths)
        output["arms"][name]["gradient_clades_with_shared_em"] = (
            output["arms"][name]["gradient_clades"]
            + sum(item["clades"] for item in shared_em_work)
        )
        output["arms"][name]["gradient_full_clade_equivalents_with_shared_em"] = (
            output["arms"][name]["gradient_clades_with_shared_em"]
            / int(artifact["family_clades"][:n].sum())
        )
        arm_tensors[name] = {
            "theta": result["theta"].detach().cpu(),
            "curvature": result["curvature"].detach().cpu(),
            "history": result["history"],
            "gradient_work": result["gradient_work"],
            "ray_diagnostics": {
                key: value.detach().cpu()
                for key, value in result.get("hierarchical_ray_diagnostics", {}).items()
            },
        }
        print(json.dumps({name: output["arms"][name]}, indent=2, default=str), flush=True)

    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2, default=str) + "\n")
    tensor_destination = Path(args.tensor_out) if args.tensor_out else destination.with_suffix(".pt")
    torch.save({
        "schema": output["schema"], "artifact": args.artifact,
        "production_source_sha256": PRODUCTION_SOURCE_SHA256, "arms": arm_tensors,
    }, tensor_destination)
    print(f"wrote {destination}")
    print(f"wrote {tensor_destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
