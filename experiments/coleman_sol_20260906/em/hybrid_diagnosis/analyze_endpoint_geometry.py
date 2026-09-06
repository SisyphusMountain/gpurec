"""CPU-only structural diagnostics for the shared EM2 hierarchical endpoint.

The actual endpoint gradient was not stored in the shared artifact.  Consequently the trust-step
comparison below deliberately uses the last observed gradient (at z1) as a labelled proxy at z2.
Metric and seed diagnostics do not make that approximation.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


REPO = Path(__file__).resolve().parents[4]
HYBRID = REPO / "experiments/coleman_sol_20260906/geometry/hybrid"
sys.path.insert(0, str(HYBRID))

from hierarchical_adapter import (  # noqa: E402
    jacobian,
    phi_to_theta,
    regularized_tangent_direction,
)


QUANTILES = (0.0, 0.1, 0.5, 0.9, 1.0)


def quantiles(value: torch.Tensor) -> dict[str, float]:
    value = value.detach().double().flatten()
    return {
        f"p{int(100 * q):02d}": float(torch.quantile(value, q))
        for q in QUANTILES
    }


def exact_linear_native_trust_step(
    hessian: torch.Tensor,
    gradient: torch.Tensor,
    metric: torch.Tensor,
    radius: float,
    mu: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve the convexified 3D generalized trust problem family by family.

    Whitening by G^(1/2) turns d'Gd <= r^2 into a Euclidean ball.  The generalized curvature is
    floored at ``mu`` and the single trust multiplier is found by bisection.  The second return is
    that convexified curvature back in z coordinates, for a common-model gain comparison.
    """
    directions, convexified = [], []
    identity = torch.eye(3, dtype=torch.float64)
    for h, g, gram in zip(hessian, gradient, metric):
        values_g, vectors_g = torch.linalg.eigh(gram)
        gram_half = (vectors_g * values_g.sqrt().unsqueeze(0)) @ vectors_g.T
        gram_inverse_half = (vectors_g * values_g.rsqrt().unsqueeze(0)) @ vectors_g.T
        whitened_h = gram_inverse_half @ h @ gram_inverse_half
        values_h, vectors_h = torch.linalg.eigh(whitened_h)
        values_h = values_h.clamp_min(mu)
        whitened_h = (vectors_h * values_h.unsqueeze(0)) @ vectors_h.T
        whitened_g = gram_inverse_half @ g

        def solve(multiplier: float) -> torch.Tensor:
            return -torch.linalg.solve(
                whitened_h + multiplier * identity, whitened_g,
            )

        whitened_step = solve(0.0)
        if float(whitened_step.norm()) > radius:
            low, high = 0.0, 1.0
            while float(solve(high).norm()) > radius:
                high *= 2.0
            for _ in range(80):
                middle = 0.5 * (low + high)
                if float(solve(middle).norm()) > radius:
                    low = middle
                else:
                    high = middle
            whitened_step = solve(high)
        directions.append(gram_inverse_half @ whitened_step)
        convexified.append(gram_half @ whitened_h @ gram_half)
    return torch.stack(directions), torch.stack(convexified)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact",
        default=REPO / "experiments/coleman_sol_20260906/em/hybrid_shared_200_v2.pt",
        type=Path,
    )
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    artifact = torch.load(args.artifact, map_location="cpu", weights_only=False)
    hierarchy = artifact["hierarchical"]
    z2 = hierarchy["z2"].double()
    theta2 = artifact["theta_native"]["theta2"].double()
    information = hierarchy["information_z2_N1"].double()
    seed = hierarchy["scaled_bfgs_seed_z"].double()
    step = hierarchy["step_z10"].double()
    gradient_change = hierarchy["gradient_change_z10"].double()
    proxy_gradient = hierarchy["g_z1"].double()
    native = artifact["native_control"]["consistent_evaluated_points"]

    j = jacobian(z2)
    metric = j.transpose(-1, -2) @ j
    metric_values = torch.linalg.eigvalsh(metric)
    h_values, h_vectors = torch.linalg.eigh(seed)
    metric_in_h_basis = h_vectors.transpose(-1, -2) @ metric @ h_vectors
    metric_diagonal = torch.diagonal(metric_in_h_basis, dim1=-2, dim2=-1)
    normalized_metric = metric_in_h_basis / torch.sqrt(
        metric_diagonal.unsqueeze(-1) * metric_diagonal.unsqueeze(-2)
    )
    off_diagonal_mask = ~torch.eye(3, dtype=torch.bool)
    max_metric_correlation = normalized_metric[:, off_diagonal_mask].abs().amax(dim=1)
    off_diagonal = metric_in_h_basis - torch.diag_embed(metric_diagonal)
    relative_off_diagonal = (
        off_diagonal.norm(dim=(-2, -1))
        / torch.diag_embed(metric_diagonal).norm(dim=(-2, -1))
    )

    information_step = torch.einsum("fij,fj->fi", information, step)
    scaled_information = information * hierarchy["scale"][:, None, None]
    scaled_prediction = torch.einsum("fij,fj->fi", scaled_information, step)
    seed_prediction = torch.einsum("fij,fj->fi", seed, step)
    direct_cosine = (information_step * gradient_change).sum(dim=1) / (
        information_step.norm(dim=1) * gradient_change.norm(dim=1)
    ).clamp_min(1e-30)
    direct_relative_residual = (
        (scaled_prediction - gradient_change).norm(dim=1)
        / gradient_change.norm(dim=1).clamp_min(1e-30)
    )
    seed_relative_residual = (
        (seed_prediction - gradient_change).norm(dim=1)
        / gradient_change.norm(dim=1).clamp_min(1e-30)
    )
    bfgs_relative_update = (
        (seed - scaled_information).norm(dim=(-2, -1))
        / scaled_information.norm(dim=(-2, -1)).clamp_min(1e-30)
    )

    native_pullback_information = (
        j.transpose(-1, -2) @ native["information"].double() @ j
    )
    direct_information_pullback_error = (
        (information - native_pullback_information).norm(dim=(-2, -1))
        / information.norm(dim=(-2, -1)).clamp_min(1e-30)
    )
    native_pullback_seed = j.transpose(-1, -2) @ native["seed"].double() @ j
    seed_pullback_difference = (
        (seed - native_pullback_seed).norm(dim=(-2, -1))
        / native_pullback_seed.norm(dim=(-2, -1)).clamp_min(1e-30)
    )
    best_scale = (seed * native_pullback_seed).sum(dim=(-2, -1)) / (
        native_pullback_seed.square().sum(dim=(-2, -1)).clamp_min(1e-30)
    )
    seed_shape_difference = (
        (seed - best_scale[:, None, None] * native_pullback_seed).norm(dim=(-2, -1))
        / seed.norm(dim=(-2, -1)).clamp_min(1e-30)
    )

    lo, hi = artifact["metadata"]["rate_box_log2"]
    endpoint_faces = (theta2 <= lo + 1e-6) | (theta2 >= hi - 1e-6)

    # A structural, not measured-fit, comparison. g(z1) is the closest saved real gradient but is
    # not the unknown g(z2); retain it only to estimate the possible size of the metric omission.
    radius = 2.0
    radii = torch.full((len(z2),), radius, dtype=torch.float64)
    current_direction, *_ = regularized_tangent_direction(
        z2, torch.zeros_like(z2), proxy_gradient, seed,
        torch.zeros_like(z2, dtype=torch.bool), radii, 1e-4, metric="native",
    )
    current_norm_before_ray = torch.sqrt(torch.einsum(
        "fi,fij,fj->f", current_direction, metric, current_direction,
    ))
    ray_scale = torch.minimum(
        torch.ones_like(current_norm_before_ray),
        radii / current_norm_before_ray.clamp_min(1e-30),
    )
    current_direction = current_direction * ray_scale[:, None]
    exact_direction, common_convexified = exact_linear_native_trust_step(
        seed, proxy_gradient, metric, radius=radius, mu=1e-4,
    )
    current_norm = torch.sqrt(torch.einsum(
        "fi,fij,fj->f", current_direction, metric, current_direction,
    ))
    exact_norm = torch.sqrt(torch.einsum(
        "fi,fij,fj->f", exact_direction, metric, exact_direction,
    ))
    metric_cosine = torch.einsum(
        "fi,fij,fj->f", current_direction, metric, exact_direction,
    ) / (current_norm * exact_norm).clamp_min(1e-30)

    def gain(direction: torch.Tensor) -> torch.Tensor:
        return -(
            (proxy_gradient * direction).sum(dim=1)
            + 0.5 * torch.einsum(
                "fi,fij,fj->f", direction, common_convexified, direction,
            )
        )

    exact_gain = gain(exact_direction)
    gain_ratio = gain(current_direction) / exact_gain.clamp_min(1e-30)
    nonlinear_displacement = (phi_to_theta(z2 + current_direction) - theta2).norm(dim=1)
    nonlinear_to_linear_norm = nonlinear_displacement / current_norm.clamp_min(1e-30)

    output = {
        "schema": "gpurec.em_hybrid_endpoint_diagnosis.v1",
        "artifact": str(args.artifact),
        "n_families": len(z2),
        "metric_at_em2_endpoint": {
            "lambda_min_G": quantiles(metric_values[:, 0]),
            "lambda_max_G": quantiles(metric_values[:, -1]),
            "condition_G": quantiles(metric_values[:, -1] / metric_values[:, 0]),
            "max_normalized_offdiagonal_in_seed_eigenbasis": quantiles(max_metric_correlation),
            "relative_offdiagonal_frobenius_in_seed_eigenbasis": quantiles(relative_off_diagonal),
            "endpoint_families_at_native_face": int(endpoint_faces.any(dim=1).sum()),
            "endpoint_coordinates_at_native_face": endpoint_faces.sum(dim=0).tolist(),
        },
        "seed_at_em2_endpoint": {
            "em_z_step_norm": quantiles(step.norm(dim=1)),
            "hierarchical_scalar_calibration": quantiles(hierarchy["scale"]),
            "native_scalar_calibration": quantiles(native["scale"]),
            "hierarchical_to_native_scale_ratio": quantiles(
                hierarchy["scale"] / native["scale"]
            ),
            "direct_information_secant_cosine": quantiles(direct_cosine),
            "scaled_direct_information_secant_relative_residual": quantiles(
                direct_relative_residual
            ),
            "bfgs_seed_secant_relative_residual": quantiles(seed_relative_residual),
            "bfgs_update_relative_to_scaled_information": quantiles(bfgs_relative_update),
            "direct_information_vs_native_information_pullback_relative_error": quantiles(
                direct_information_pullback_error
            ),
            "hierarchical_seed_vs_native_seed_pullback_relative_difference": quantiles(
                seed_pullback_difference
            ),
            "seed_shape_difference_after_best_scalar_rescaling": quantiles(
                seed_shape_difference
            ),
            "scale_valid": int(hierarchy["scale_valid"].sum()),
            "bfgs_valid": int(hierarchy["bfgs_valid"].sum()),
        },
        "proxy_trust_comparison": {
            "warning": "Uses measured g(z1) as a proxy at z2; this is structural expectation, not fit evidence.",
            "radius": radius,
            "current_direction_linear_native_norm_before_ray": quantiles(
                current_norm_before_ray
            ),
            "families_requiring_final_combined_ray_cap": int(
                (current_norm_before_ray > radius).sum()
            ),
            "physical_metric_cosine_current_vs_exact": quantiles(metric_cosine),
            "quadratic_gain_ratio_current_vs_exact": quantiles(gain_ratio),
            "current_nonlinear_to_linear_native_norm_ratio": quantiles(
                nonlinear_to_linear_norm
            ),
            "current_nonlinear_displacement_above_radius": int(
                (nonlinear_displacement > radius).sum()
            ),
            "worst_gain_family": int(gain_ratio.argmin()),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
