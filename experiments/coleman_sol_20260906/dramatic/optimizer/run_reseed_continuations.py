"""Matched shared-EM2 continuations with/without first-gradient latest-pair reseeding."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import sys
import time
from pathlib import Path

import torch

from gpurec.api.model import GeneReconModel
from gpurec.config import GpurecConfig
from gpurec.fit.genewise_fit import TRUST_TEST_OFF, fit_genewise


THIS = Path(__file__).resolve().parent
REPO = THIS.parents[3]
HYBRID = REPO / "experiments/coleman_sol_20260906/geometry/hybrid"
sys.path.insert(0, str(HYBRID))

from fresh_seed import FreshReseeder  # noqa: E402
from reseed_adapter import compile_hierarchical_reseed, compile_native_reseed  # noqa: E402
from source_adapter import (  # noqa: E402
    PRODUCTION_SOURCE_SHA256,
    fit_genewise_hierarchical,
)


def _quantiles(value: torch.Tensor) -> dict[str, float]:
    value = value.detach().double().flatten()
    return {
        name: float(torch.quantile(value, point))
        for name, point in (("min", 0.0), ("p10", 0.1), ("median", 0.5),
                            ("p90", 0.9), ("max", 1.0))
    }


def _cpu_result(result: dict) -> dict:
    keep = (
        "theta", "curvature", "history", "gradient_work", "loss_bits", "n_steps",
        "n_builds", "n_verify_builds", "n_hessians", "n_rebuilds", "converged",
        "unconverged", "bound_active", "pg_max", "newton_grad_seconds", "hessian_seconds",
        "verify_seconds", "rebuild_seconds", "certify_seconds", "opt_seconds",
        "hierarchical_ray_diagnostics",
    )
    out = {}
    for key in keep:
        if key not in result:
            continue
        value = result[key]
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu()
        elif key == "hierarchical_ray_diagnostics":
            value = {name: tensor.detach().cpu() for name, tensor in value.items()}
        out[key] = value
    return out


def _reseed_summary(reseeder: FreshReseeder | None) -> dict | None:
    if reseeder is None:
        return None
    if reseeder.calls != 1 or reseeder.diagnostic is None:
        raise RuntimeError(f"expected exactly one first-gradient reseed, got {reseeder.calls}")
    row = reseeder.diagnostic
    return {
        "calls": reseeder.calls,
        "scale_valid": int(row["scale_valid"].sum()),
        "scale": _quantiles(row["scale"]),
        "step_norm": _quantiles(row["step"].norm(dim=1)),
        "gradient_change_norm": _quantiles(row["gradient_change"].norm(dim=1)),
        "secant_relative_residual": _quantiles(row["secant_relative_residual"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact", type=Path,
        default=REPO / "experiments/coleman_sol_20260906/em/hybrid_shared_200_v2.pt",
    )
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--certify", action="store_true")
    parser.add_argument("--audit", action="store_true")
    parser.add_argument(
        "--arms", default="native_old,native_reseed,hierarchical_old,hierarchical_reseed",
    )
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    artifact = torch.load(args.artifact, map_location="cpu", weights_only=False)
    if artifact.get("schema") != "gpurec.em_hybrid_shared.v2":
        raise ValueError("point-consistent V2 shared artifact required")
    n = min(args.limit, len(artifact["paths"]))
    if n <= 0:
        raise ValueError("limit must select at least one family")
    paths = artifact["paths"][:n]
    species = artifact["metadata"]["species"]
    theta1 = artifact["theta_native"]["theta1"][:n]
    theta2 = artifact["theta_native"]["theta2"][:n].float()
    gradient1 = artifact["gradient_theta"]["g1"][:n]
    counts1 = artifact["event_counts"]["N1"][:n]
    native_seed = artifact["native_control"]["consistent_evaluated_points"]["seed"][:n].float()
    hierarchical_seed = artifact["hierarchical"]["scaled_bfgs_seed_z"][:n].float()
    lo, hi = artifact["metadata"]["rate_box_log2"]
    config = GpurecConfig.genewise_reference()

    native_reseeder = FreshReseeder(
        coordinate="native", theta1=theta1, gradient1=gradient1, counts1=counts1,
        lo=lo, hi=hi,
    )
    hierarchical_reseeder = FreshReseeder(
        coordinate="hierarchical", theta1=theta1, gradient1=gradient1, counts1=counts1,
        lo=lo, hi=hi,
    )
    native_reseed_fit, native_hash = compile_native_reseed(native_reseeder)
    hierarchical_reseed_fit, hierarchy_hash = compile_hierarchical_reseed(
        hierarchical_reseeder,
    )
    if native_hash != hierarchy_hash or native_hash != PRODUCTION_SOURCE_SHA256:
        raise RuntimeError("all arms must originate from the same production source revision")

    available = {
        "native_old": (fit_genewise, native_seed, None),
        "native_reseed": (native_reseed_fit, native_seed, native_reseeder),
        "hierarchical_old": (fit_genewise_hierarchical, hierarchical_seed, None),
        "hierarchical_reseed": (
            hierarchical_reseed_fit, hierarchical_seed, hierarchical_reseeder,
        ),
    }
    selected = [item.strip() for item in args.arms.split(",") if item.strip()]
    unknown = set(selected) - set(available)
    if unknown:
        raise ValueError(f"unknown arms: {sorted(unknown)}")

    common = dict(
        device="cuda", dtype="float32", min_rate=2.0 ** lo, max_rate=2.0 ** hi,
        adam_steps=0, adam_lr=1.0, warmup_method="adam", em_steps=2, grad_clip=10.0,
        pi_tiers=(16, 64), neu_opt=16, neu_cert=64,
        clade_budget=int(artifact["metadata"]["clade_budget"]), tol=1e-3,
        max_iter=args.max_iter, check_every=2, drop_frac=0.05, min_drop=32,
        rebuild_frac=0.25, hessian_refresh=15, curvature_update="bfgs",
        trust=2.0, trust_max=8.0, mu=1e-4, fwd_tol=1e-3, improve_frac=0.8,
        verify_drop=True, eager_defer=True, certify=args.certify, certify_curvature=False,
        init_log2_rates=theta2, stall_patience=120, step_extrapolation=1.0,
        step_model="quadratic", stop_nll_bits=0.0, approach_pruning_threshold=0.0,
        targeted_hessian=(0, 0.0), coordinate_staging=(0, 0),
        trust_test=TRUST_TEST_OFF, solver_options=None, config=config, verbose=True,
    )
    shared_clades = int(artifact["family_clades"][:n].sum())
    shared_calls = int(artifact["gradient_calls"])
    saved = {
        "schema": "gpurec.em2_first_gradient_reseed.v1",
        "artifact": str(args.artifact),
        "production_source_sha256": native_hash,
        "n_families": n,
        "shared_em": {
            "gradient_calls": shared_calls,
            "clades_per_call": shared_clades,
            "gradient_clades": shared_calls * shared_clades,
            "timing_full_200_artifact": artifact["timing"],
        },
        "arms": {},
    }
    result_thetas = {}
    for name in selected:
        fit, seed, reseeder = available[name]
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = fit(species, paths, init_curvature=seed, **common)
        torch.cuda.synchronize()
        wall = time.perf_counter() - started
        continuation_clades = sum(int(row["clades"]) for row in result["gradient_work"])
        result_thetas[name] = result["theta"].detach().cpu()
        saved["arms"][name] = {
            "wall_seconds_continuation": wall,
            "prototype_wall_with_shared_em_seconds": (
                wall + float(artifact["timing"]["total_seconds"])
            ),
            "continuation_gradient_clades": continuation_clades,
            "gradient_clades_with_shared_em": continuation_clades + shared_calls * shared_clades,
            "full_clade_equivalents_with_shared_em": (
                continuation_clades / shared_clades + shared_calls
            ),
            "reseed": _reseed_summary(reseeder),
            "result": _cpu_result(result),
        }
        headline = saved["arms"][name]
        print(json.dumps({name: {
            key: value for key, value in headline.items() if key != "result"
        }}, indent=2), flush=True)

    if args.audit:
        solver = replace(config.solver, pi_iters=64, neumann_terms=64)
        torch.cuda.synchronize()
        audit_started = time.perf_counter()
        model = GeneReconModel(
            species, paths, mode="genewise", device="cuda", dtype=torch.float32,
            solver_options=solver, config=config,
            clade_budget=int(artifact["metadata"]["clade_budget"]),
        )
        model.receiver_weights.requires_grad_(False)
        audit = {}
        with torch.no_grad():
            for name, theta in result_thetas.items():
                values = model.genewise_loss_vector(theta=theta.cuda()).double().cpu()
                audit[name] = {"nll_per_family_bits": values, "nll_bits": float(values.sum())}
        torch.cuda.synchronize()
        saved["fresh_audit"] = {
            "seconds": time.perf_counter() - audit_started,
            "note": "Outside arm fit timings; one common pi64/Neumann64 model.",
            "arms": audit,
        }
        del model
        torch.cuda.empty_cache()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(saved, args.out)
    summary_path = args.out.with_suffix(".json")
    summary = {
        "schema": saved["schema"], "artifact": saved["artifact"],
        "production_source_sha256": native_hash, "n_families": n,
        "shared_em": saved["shared_em"], "arms": {},
    }
    for name, row in saved["arms"].items():
        result = row["result"]
        summary["arms"][name] = {
            key: value for key, value in row.items() if key != "result"
        }
        summary["arms"][name]["fit"] = {
            key: result.get(key) for key in (
                "loss_bits", "n_steps", "n_builds", "n_verify_builds", "n_hessians",
                "n_rebuilds", "converged", "unconverged", "bound_active", "pg_max",
                "newton_grad_seconds", "hessian_seconds", "verify_seconds",
                "rebuild_seconds", "certify_seconds", "opt_seconds",
            )
        }
    if "fresh_audit" in saved:
        summary["fresh_audit"] = {
            "seconds": saved["fresh_audit"]["seconds"],
            "note": saved["fresh_audit"]["note"],
            "nll_bits": {
                name: row["nll_bits"] for name, row in saved["fresh_audit"]["arms"].items()
            },
        }
    summary_path.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(f"wrote {args.out}")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
