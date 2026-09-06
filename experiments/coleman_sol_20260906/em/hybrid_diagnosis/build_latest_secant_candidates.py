"""Build CPU-only first-step candidates using the free EM1->EM2 gradient secant.

The first Newton evaluation at theta2 has already paid for g2.  Existing EM2 warm-up seeds use the
theta0->theta1 pair and deliberately suppress the ordinary first BFGS carry, so they do not use the
available theta1->theta2 / g1->g2 pair before proposing the first Newton step.  This script changes
only that seed, then applies the existing native or hierarchical step control unchanged.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch


REPO = Path(__file__).resolve().parents[4]
EM = REPO / "experiments/coleman_sol_20260906/em"
HYBRID = REPO / "experiments/coleman_sol_20260906/geometry/hybrid"
sys.path.insert(0, str(EM))
sys.path.insert(0, str(HYBRID))

from gpurec.fit.genewise_fit import _bfgs_update  # noqa: E402
from hybrid_geometry import (  # noqa: E402
    calibrated_bfgs_seed,
    complete_information_theta,
    complete_information_z,
    transform_gradient,
    z_from_theta,
)
from hierarchical_adapter import (  # noqa: E402
    predicted_decrease,
    retract_feasible_ray_cpu64,
    transform_gradient as transform_gradient_adapter,
    working_set_tangent_direction,
)


def q(value: torch.Tensor) -> dict[str, float]:
    value = value.detach().double().flatten()
    return {
        name: float(torch.quantile(value, point))
        for name, point in (("min", 0.0), ("p10", 0.1), ("median", 0.5),
                            ("p90", 0.9), ("max", 1.0))
    }


def native_latest_seed(
    theta1: torch.Tensor,
    theta2: torch.Tensor,
    g1: torch.Tensor,
    g2: torch.Tensor,
    counts1: torch.Tensor,
    lo: float,
    hi: float,
    eps: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    information = complete_information_theta(theta2, counts1)
    step = theta2 - theta1
    gradient_change = g2 - g1
    information_step = torch.einsum("fij,fj->fi", information, step)
    sy = (step * gradient_change).sum(dim=1)
    sbs = (step * information_step).sum(dim=1)
    valid = (sy > 0) & (sbs > 0) & torch.isfinite(sy) & torch.isfinite(sbs)
    scale = torch.where(valid, sy / torch.where(valid, sbs, torch.ones_like(sbs)),
                        torch.ones_like(sy))
    scaled = information * scale[:, None, None]
    free = (
        (theta1 > lo + eps) & (theta1 < hi - eps)
        & (theta2 > lo + eps) & (theta2 < hi - eps)
    ).to(theta2.dtype)
    seed = _bfgs_update(scaled, step, gradient_change, free)
    return seed, {
        "information": information, "step": step, "gradient_change": gradient_change,
        "sy": sy, "s_information_s": sbs, "scale": scale, "scale_valid": valid,
        "free_at_both": free,
    }


def native_step(
    theta: torch.Tensor,
    gradient: torch.Tensor,
    seed: torch.Tensor,
    radius: float,
    mu: float,
    lo: float,
    hi: float,
    eps: float,
) -> dict[str, torch.Tensor]:
    fixed = ((theta >= hi - eps) & (gradient < 0)) | ((theta <= lo + eps) & (gradient > 0))
    free = (~fixed).to(theta.dtype)
    values, vectors = torch.linalg.eigh(seed)
    gradient_eigen = (
        vectors.transpose(1, 2) @ (gradient * free).unsqueeze(-1)
    ).squeeze(-1)
    radii = torch.full((len(theta), 1), radius, dtype=theta.dtype)
    adjusted = torch.maximum(
        torch.maximum(values, torch.full_like(values, mu)), gradient_eigen.abs() / radii,
    )
    regularized = vectors @ torch.diag_embed(adjusted) @ vectors.transpose(1, 2)
    regularized = (
        regularized * free.unsqueeze(1) * free.unsqueeze(2)
        + torch.diag_embed(1.0 - free)
    )
    proposed = -torch.linalg.solve(
        regularized, (gradient * free).unsqueeze(-1),
    ).squeeze(-1)
    norm = proposed.norm(dim=1, keepdim=True)
    capped = norm > radius
    applied = proposed * torch.where(
        capped, radii / torch.where(capped, norm, torch.ones_like(norm)), torch.ones_like(norm),
    )
    endpoint = (theta + applied).clamp(min=lo, max=hi)
    applied = endpoint - theta
    predicted = -(
        (gradient * applied).sum(dim=1)
        + 0.5 * torch.einsum("fi,fij,fj->f", applied, regularized, applied)
    )
    return {
        "theta": endpoint, "predicted": predicted, "proposed_direction": proposed,
        "applied_tangent": applied, "fixed": fixed, "working_fixed": fixed,
        "regularized_curvature": regularized, "radius": torch.full((len(theta),), radius),
    }


def hierarchical_step(
    theta: torch.Tensor,
    gradient_theta: torch.Tensor,
    seed: torch.Tensor,
    radius: float,
    mu: float,
    lo: float,
    hi: float,
    eps: float,
) -> dict[str, torch.Tensor]:
    phi = z_from_theta(theta)
    gradient_phi = transform_gradient_adapter(phi, gradient_theta)
    fixed = (
        ((theta >= hi - eps) & (gradient_theta < 0))
        | ((theta <= lo + eps) & (gradient_theta > 0))
    )
    radii = torch.full((len(theta),), radius, dtype=theta.dtype)
    direction, regularized, projected, working, _limited = working_set_tangent_direction(
        phi, theta, gradient_theta, gradient_phi, seed, fixed, radii, mu,
        lo, hi, eps, metric="native",
    )
    phi_new, endpoint, tangent, chord, capped = retract_feasible_ray_cpu64(
        phi, theta, direction, working, lo, hi, radii, metric="native",
    )
    return {
        "theta": endpoint, "predicted": predicted_decrease(projected, regularized, tangent),
        "proposed_direction": direction, "applied_tangent": tangent,
        "lifted_chord": chord, "phi": phi, "phi_new": phi_new,
        "gradient_phi": gradient_phi, "fixed": fixed, "working_fixed": working,
        "regularized_curvature": regularized, "radius": radii, "radius_capped": capped,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact", type=Path,
        default=EM / "hybrid_shared_200_v2.pt",
    )
    parser.add_argument(
        "--trace", type=Path,
        default=REPO / "experiments/coleman_sol_20260906/geometry/hybrid/diagnosis/trace200.pt",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path)
    args = parser.parse_args()

    artifact = torch.load(args.artifact, map_location="cpu", weights_only=False)
    trace = torch.load(args.trace, map_location="cpu", weights_only=False)
    native_trace = trace["arms"]["native"]["trace"]["proposals"][0]
    hierarchy_trace = trace["arms"]["hierarchical_native_metric"]["trace"]["proposals"][0]
    theta2 = artifact["theta_native"]["theta2"].double()
    theta1 = artifact["theta_native"]["theta1"].double()
    g1_theta = artifact["gradient_theta"]["g1"].double()
    counts1 = artifact["event_counts"]["N1"].double()
    z1 = artifact["hierarchical"]["z1"].double()
    z2 = artifact["hierarchical"]["z2"].double()
    lo, hi = artifact["metadata"]["rate_box_log2"]
    eps, radius, mu = 1e-6, 2.0, 1e-4

    g2_native = native_trace["gradient_theta"].double()
    g2_hierarchy = hierarchy_trace["gradient_theta"].double()
    g1_z = transform_gradient(z1, g1_theta)
    g2_z = transform_gradient(z2, g2_hierarchy)

    newest_native_seed, newest_native_detail = native_latest_seed(
        theta1, theta2, g1_theta, g2_native, counts1, lo, hi, eps,
    )
    newest_hierarchy_information = complete_information_z(z2, counts1)
    newest_hierarchy_seed, newest_hierarchy_detail = calibrated_bfgs_seed(
        newest_hierarchy_information, z2 - z1, g2_z - g1_z,
    )
    # The narrowest alternative: preserve the old theta0->theta1 seed and perform the ordinary
    # first BFGS carry with the now available theta1->theta2 pair instead of resetting from Ic.
    carried_native_seed = _bfgs_update(
        artifact["native_control"]["consistent_evaluated_points"]["seed"].double(),
        newest_native_detail["step"], newest_native_detail["gradient_change"],
        newest_native_detail["free_at_both"],
    )
    carried_hierarchy_seed = _bfgs_update(
        artifact["hierarchical"]["scaled_bfgs_seed_z"].double(),
        z2 - z1, g2_z - g1_z, torch.ones_like(z2),
    )

    theta_start = theta2.float()
    latest_native = native_step(
        theta_start, g2_native.float(), newest_native_seed.float(), radius, mu, lo, hi, eps,
    )
    latest_hierarchy = hierarchical_step(
        theta_start, g2_hierarchy.float(), newest_hierarchy_seed.float(),
        radius, mu, lo, hi, eps,
    )
    carried_native = native_step(
        theta_start, g2_native.float(), carried_native_seed.float(), radius, mu, lo, hi, eps,
    )
    carried_hierarchy = hierarchical_step(
        theta_start, g2_hierarchy.float(), carried_hierarchy_seed.float(),
        radius, mu, lo, hi, eps,
    )
    old_hierarchy_half = hierarchical_step(
        theta_start, g2_hierarchy.float(),
        artifact["hierarchical"]["scaled_bfgs_seed_z"].float(),
        0.5, mu, lo, hi, eps,
    )

    candidates = {
        "old_native_trace": {
            "theta": native_trace["applied_theta"].float(),
            "predicted": native_trace["predicted"].float(),
            "gradient_theta": native_trace["gradient_theta"].float(),
            "seed": artifact["native_control"]["consistent_evaluated_points"]["seed"].float(),
            "source": "recorded traced first proposal",
        },
        "old_hierarchical_trace": {
            "theta": hierarchy_trace["applied_theta"].float(),
            "predicted": hierarchy_trace["predicted"].float(),
            "gradient_theta": hierarchy_trace["gradient_theta"].float(),
            "seed": artifact["hierarchical"]["scaled_bfgs_seed_z"].float(),
            "source": "recorded traced first proposal",
        },
        "latest_native": {
            **latest_native, "gradient_theta": g2_native.float(),
            "seed": newest_native_seed.float(),
            "source": "theta1->theta2 / g1->g2 contemporaneous seed; existing native step",
        },
        "latest_hierarchical": {
            **latest_hierarchy, "gradient_theta": g2_hierarchy.float(),
            "seed": newest_hierarchy_seed.float(),
            "source": "z1->z2 / gz1->gz2 contemporaneous seed; existing hierarchy step",
        },
        "carry_native": {
            **carried_native, "gradient_theta": g2_native.float(),
            "seed": carried_native_seed.float(),
            "source": "old seed plus ordinary theta1->theta2 BFGS carry; existing native step",
        },
        "carry_hierarchical": {
            **carried_hierarchy, "gradient_theta": g2_hierarchy.float(),
            "seed": carried_hierarchy_seed.float(),
            "source": "old seed plus ordinary z1->z2 BFGS carry; existing hierarchy step",
        },
        "old_hierarchical_radius_0_5": {
            **old_hierarchy_half, "gradient_theta": g2_hierarchy.float(),
            "seed": artifact["hierarchical"]["scaled_bfgs_seed_z"].float(),
            "source": "old hierarchy seed; existing hierarchy step with radius 0.5",
        },
    }
    output = {
        "schema": "gpurec.em_hybrid_latest_secant_first_step.v1",
        "artifact": str(args.artifact),
        "trace": str(args.trace),
        "theta_start": theta_start,
        "global_family_ids": native_trace["global_family_ids"].long(),
        "candidates": candidates,
        "latest_pair": {
            "theta1": theta1, "theta2": theta2, "g1_theta": g1_theta,
            "g2_native_theta": g2_native, "g2_hierarchical_theta": g2_hierarchy,
            "z1": z1, "z2": z2, "g1_z": g1_z, "g2_z": g2_z,
            "counts_N1": counts1,
            "native": newest_native_detail,
            "hierarchical": {
                "information": newest_hierarchy_information,
                "step": z2 - z1, "gradient_change": g2_z - g1_z,
                **newest_hierarchy_detail,
            },
            "carried_native_seed": carried_native_seed,
            "carried_hierarchical_seed": carried_hierarchy_seed,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.out)

    summary = {
        "schema": output["schema"], "artifact": str(args.artifact), "trace": str(args.trace),
        "g2_native_vs_hierarchical_absolute_max": float((g2_native - g2_hierarchy).abs().max()),
        "native_latest_scale": q(newest_native_detail["scale"]),
        "hierarchical_latest_scale": q(newest_hierarchy_detail["scale"]),
        "native_latest_scale_valid": int(newest_native_detail["scale_valid"].sum()),
        "hierarchical_latest_scale_valid": int(newest_hierarchy_detail["scale_valid"].sum()),
        "hierarchical_latest_bfgs_valid": int(newest_hierarchy_detail["bfgs_valid"].sum()),
        "candidates": {},
    }
    for name, candidate in candidates.items():
        displacement = candidate["theta"].double() - theta_start.double()
        summary["candidates"][name] = {
            "native_displacement_norm": q(displacement.norm(dim=1)),
            "predicted_bits": q(candidate["predicted"]),
            "theta_min": float(candidate["theta"].min()),
            "theta_max": float(candidate["theta"].max()),
        }
    summary_path = args.summary_out or args.out.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.out}")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
