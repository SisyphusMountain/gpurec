"""Run two callback-traced continuations from the shared point-consistent EM2 endpoint."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from gpurec.config import GpurecConfig
from gpurec.fit.genewise_fit import TRUST_TEST_OFF


THIS = Path(__file__).resolve().parent
HYBRID = THIS.parent
if str(HYBRID) not in sys.path:
    sys.path.insert(0, str(HYBRID))

from traced_adapter import compile_traced_hierarchical_native_metric, compile_traced_native


def _cpu_result(result: dict) -> dict:
    keep = (
        "theta", "curvature", "history", "gradient_work", "loss_bits", "n_steps", "n_builds",
        "n_hessians", "n_rebuilds", "converged", "unconverged", "bound_active", "pg_max",
        "newton_grad_seconds", "hessian_seconds", "rebuild_seconds", "certify_seconds",
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", default="experiments/coleman_sol_20260906/em/hybrid_shared_200_v2.pt")
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()

    artifact = torch.load(args.artifact, map_location="cpu", weights_only=False)
    if artifact.get("schema") != "gpurec.em_hybrid_shared.v2" or len(artifact["paths"]) != 200:
        raise ValueError("the traced gate requires the 200-family point-consistent V2 artifact")
    paths = artifact["paths"]
    species = artifact["metadata"]["species"]
    theta2 = artifact["theta_native"]["theta2"].float()
    native_seed = artifact["native_control"]["consistent_evaluated_points"]["seed"].float()
    hierarchical_seed = artifact["hierarchical"]["scaled_bfgs_seed_z"].float()
    lo, hi = artifact["metadata"]["rate_box_log2"]
    config = GpurecConfig.genewise_reference()
    common = dict(
        device="cuda", dtype="float32", min_rate=2.0 ** lo, max_rate=2.0 ** hi,
        adam_steps=0, adam_lr=1.0, warmup_method="adam", em_steps=2, grad_clip=10.0,
        pi_tiers=(16, 64), neu_opt=16, neu_cert=64,
        clade_budget=int(artifact["metadata"]["clade_budget"]), tol=1e-3, max_iter=200,
        check_every=2, drop_frac=0.05, min_drop=32, rebuild_frac=0.25,
        hessian_refresh=15, curvature_update="bfgs", trust=2.0, trust_max=8.0, mu=1e-4,
        fwd_tol=1e-3, improve_frac=0.8, verify_drop=True, eager_defer=True,
        certify=True, certify_curvature=False, init_log2_rates=theta2, stall_patience=120,
        step_extrapolation=1.0, step_model="quadratic", stop_nll_bits=0.0,
        approach_pruning_threshold=0.0, targeted_hessian=(0, 0.0),
        coordinate_staging=(0, 0), trust_test=TRUST_TEST_OFF,
        solver_options=None, config=config, verbose=True,
    )
    native_fit, native_trace, native_hash = compile_traced_native()
    hierarchical_fit, hierarchical_trace, hierarchical_hash = compile_traced_hierarchical_native_metric()
    if native_hash != hierarchical_hash:
        raise RuntimeError("native and hierarchical traces did not originate from one source revision")
    arms = (
        ("native", native_fit, native_trace, native_seed),
        ("hierarchical_native_metric", hierarchical_fit, hierarchical_trace, hierarchical_seed),
    )
    saved = {
        "schema": "gpurec.hybrid_diagnostic_trace.v1",
        "artifact": args.artifact,
        "production_source_sha256": native_hash,
        "paths": paths,
        "family_clades": artifact["family_clades"],
        "shared_em_gradient_work": artifact["gradient_work"],
        "shared_em_timing": artifact["timing"],
        "arms": {},
    }
    summary = {"schema": saved["schema"], "production_source_sha256": native_hash, "arms": {}}
    for name, fit, trace, seed in arms:
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = fit(species, paths, init_curvature=seed, **common)
        torch.cuda.synchronize()
        wall = time.perf_counter() - started
        saved["arms"][name] = {"result": _cpu_result(result), "trace": trace.state_dict()}
        continuation_clades = sum(int(row["clades"]) for row in result["gradient_work"])
        summary["arms"][name] = {
            "wall_seconds_traced": wall,
            "loss_bits": float(result["loss_bits"]),
            "converged": int(result["converged"]),
            "pg_max": float(result["pg_max"]),
            "n_steps": int(result["n_steps"]),
            "n_builds": int(result["n_builds"]),
            "n_hessians": int(result["n_hessians"]),
            "continuation_gradient_clades": continuation_clades,
            "evaluations_traced": len(trace.evaluations),
            "proposals_traced": len(trace.proposals),
        }
        print(json.dumps({name: summary["arms"][name]}, indent=2), flush=True)
    torch.save(saved, args.out)
    Path(args.summary).write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {args.out}")
    print(f"wrote {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
