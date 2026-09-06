"""Derive a point-consistent V2 hybrid artifact from the GPU-generated V1 data."""
from __future__ import annotations

import argparse
import copy
import hashlib
from pathlib import Path

import torch

from gpurec.fit.genewise_fit import _bfgs_update

from hybrid_geometry import (
    calibrated_bfgs_seed,
    complete_information_theta,
    complete_information_z,
    transform_gradient,
    z_from_theta,
)


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _free_at_both(first: torch.Tensor, second: torch.Tensor, lo: float, hi: float, eps: float) -> torch.Tensor:
    return (
        (first > lo + eps)
        & (first < hi - eps)
        & (second > lo + eps)
        & (second < hi - eps)
    ).to(first.dtype)


def _native_seed(
    endpoint: torch.Tensor,
    counts: torch.Tensor,
    first: torch.Tensor,
    second: torch.Tensor,
    gradient_first: torch.Tensor,
    gradient_second: torch.Tensor,
    lo: float,
    hi: float,
    eps: float,
) -> dict[str, torch.Tensor]:
    information = complete_information_theta(endpoint, counts)
    step = second - first
    gradient_change = gradient_second - gradient_first
    information_step = torch.einsum("fij,fj->fi", information, step)
    sy = (step * gradient_change).sum(dim=-1)
    sbs = (step * information_step).sum(dim=-1)
    scale_valid = (sy > 0.0) & (sbs > 0.0) & torch.isfinite(sy) & torch.isfinite(sbs)
    scale = torch.where(
        scale_valid,
        sy / torch.where(scale_valid, sbs, torch.ones_like(sbs)),
        torch.ones_like(sy),
    )
    scaled = information * scale[:, None, None]
    free = _free_at_both(first, second, lo, hi, eps)
    seed = _bfgs_update(scaled, step, gradient_change, free)
    return {
        "endpoint": endpoint,
        "information": information,
        "step": step,
        "gradient_change": gradient_change,
        "free_at_both": free,
        "scale": scale,
        "scale_valid": scale_valid,
        "scaled_information": scaled,
        "seed": seed,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    v1 = torch.load(args.input, map_location="cpu", weights_only=False)
    if v1.get("schema") != "gpurec.em_hybrid_shared.v1":
        raise ValueError(f"expected V1 artifact, got {v1.get('schema')!r}")
    artifact = copy.deepcopy(v1)

    raw_theta0 = v1["theta_native"]["theta0"].double()
    raw_theta1 = v1["theta_native"]["theta1"].double()
    raw_theta2 = v1["theta_native"]["theta2"].double()
    # These are exactly the native points presented to the float32 production model / optimizer.
    actual_theta0 = raw_theta0.float().double()
    actual_theta1 = raw_theta1.float().double()
    actual_theta2 = raw_theta2.float().double()
    g0 = v1["gradient_theta"]["g0"].double()
    g1 = v1["gradient_theta"]["g1"].double()
    n1 = v1["event_counts"]["N1"].double()
    lo, hi = v1["metadata"]["rate_box_log2"]
    eps = 1.0e-6

    z0 = z_from_theta(actual_theta0)
    z1 = z_from_theta(actual_theta1)
    z2 = z_from_theta(actual_theta2)
    g_z0 = transform_gradient(z0, g0)
    g_z1 = transform_gradient(z1, g1)
    step_z = z1 - z0
    gradient_change_z = g_z1 - g_z0
    information_z = complete_information_z(z2, n1)
    seed_z, z_details = calibrated_bfgs_seed(information_z, step_z, gradient_change_z)
    v1_seed_z = v1["hierarchical"]["scaled_bfgs_seed_z"].double()
    z_seed_difference = seed_z - v1_seed_z
    z_seed_relative = z_seed_difference.flatten(1).norm(dim=-1) / torch.clamp_min(
        v1_seed_z.flatten(1).norm(dim=-1), 1.0e-300
    )

    native_consistent = _native_seed(
        actual_theta2, n1, actual_theta0, actual_theta1, g0, g1, lo, hi, eps
    )
    # Reproduce the current inline EM implementation exactly: its CPU state retains raw float64
    # M-step theta1/theta2 even though the gradients and eventual optimizer endpoint use fp32 casts.
    native_legacy = _native_seed(
        raw_theta2, n1, actual_theta0, raw_theta1, g0, g1, lo, hi, eps
    )
    seed_difference = native_consistent["seed"] - native_legacy["seed"]
    legacy_norm = native_legacy["seed"].flatten(1).norm(dim=-1)
    relative = seed_difference.flatten(1).norm(dim=-1) / torch.clamp_min(legacy_norm, 1.0e-300)
    consistent_float = native_consistent["seed"].float()
    legacy_float = native_legacy["seed"].float()
    float_relative = (consistent_float - legacy_float).flatten(1).norm(dim=-1) / torch.clamp_min(
        legacy_float.flatten(1).norm(dim=-1), 1.0e-30
    )

    artifact["schema"] = "gpurec.em_hybrid_shared.v2"
    artifact["description"] = (
        "Point-consistent CPU derivation from V1: all transformed points and gradients use the "
        "native float32-evaluated theta trajectory; no fit or fitted-optimum inputs."
    )
    artifact["theta_native_raw_mstep"] = {"theta1": raw_theta1, "theta2": raw_theta2}
    artifact["theta_native"] = {
        "theta0": actual_theta0,
        "theta1": actual_theta1,
        "theta2": actual_theta2,
    }
    artifact["hierarchical"] = {
        "z0": z0,
        "z1": z1,
        "z2": z2,
        "g_z0": g_z0,
        "g_z1": g_z1,
        "step_z10": step_z,
        "gradient_change_z10": gradient_change_z,
        "information_z2_N1": information_z,
        "scaled_bfgs_seed_z": seed_z,
        "comparison_to_v1_inconsistent_points": {
            "seed_absolute_max": float(z_seed_difference.abs().max()),
            "seed_relative_frobenius_median": float(z_seed_relative.median()),
            "seed_relative_frobenius_p90": float(z_seed_relative.quantile(0.9)),
            "seed_relative_frobenius_max": float(z_seed_relative.max()),
            "z1_absolute_max": float((z1 - v1["hierarchical"]["z1"]).abs().max()),
            "z2_absolute_max": float((z2 - v1["hierarchical"]["z2"]).abs().max()),
            "g_z1_absolute_max": float((g_z1 - v1["hierarchical"]["g_z1"]).abs().max()),
        },
        **z_details,
    }
    artifact["native_control"] = {
        "consistent_evaluated_points": native_consistent,
        "legacy_inline_raw_cpu_points": native_legacy,
        "seed_difference": seed_difference,
        "seed_relative_frobenius_per_family": relative,
        "comparison": {
            "absolute_max": float(seed_difference.abs().max()),
            "relative_frobenius_median": float(relative.median()),
            "relative_frobenius_p90": float(relative.quantile(0.9)),
            "relative_frobenius_max": float(relative.max()),
            "float32_absolute_max": float((consistent_float - legacy_float).abs().max()),
            "float32_relative_frobenius_median": float(float_relative.median()),
            "float32_relative_frobenius_p90": float(float_relative.quantile(0.9)),
            "float32_relative_frobenius_max": float(float_relative.max()),
            "float32_bit_identical": bool(torch.equal(consistent_float, legacy_float)),
            "endpoint_absolute_max": float((actual_theta2 - raw_theta2).abs().max()),
            "pair_point_absolute_max": float((actual_theta1 - raw_theta1).abs().max()),
        },
    }
    artifact["metadata"]["lineage"] = {
        "source_schema": v1["schema"],
        "source_path": args.input,
        "source_sha256": _sha256(args.input),
    }
    artifact["metadata"].pop("theta1_note", None)
    artifact["metadata"]["theta_note"] = (
        "theta_native contains the consistently float32-evaluated points used for z/J/g_z and "
        "the native control; theta_native_raw_mstep retains exact float64 M-step outputs."
    )
    artifact["metadata"]["curvature_note"] = (
        "The hierarchical seed is direct in z. Native control seeds separately record the "
        "consistent evaluated-point construction and faithful current inline-EM raw-CPU construction."
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, output)
    print(f"wrote {output}")
    print(artifact["native_control"]["comparison"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
