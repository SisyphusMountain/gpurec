"""CPU-only validation and screening for hierarchical event coordinates.

The three coordinates are

    u = log2((D + T) / (1 + L)), v = log2(T / D), w = log2(L).

They are the logits of a binary event tree.  This script validates the analytic
map and derivatives, checks the expected-count gradient and complete-information
identity, describes the exact observed Hessians in the new coordinates, and
screens inexpensive post-warm-up curvature seeds.  The fitted solution is used
only to evaluate trial-point distance; no seed or step depends on it.

The CPU trial step projects through the original theta box.  It is therefore a
geometric screen, not an accepted likelihood step.  Production-like experiments
must evaluate the actual NLL and retain the theta-space projected-gradient test.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch.func import jacrev, vmap


torch.set_default_dtype(torch.float64)
LN2 = math.log(2.0)
LO = math.log2(1.0e-6)
HI = math.log2(2.0)
BOUND_EPS = 1.0e-6
CURVATURE_FLOOR = 1.0e-4
BFGS_FLOOR = 1.0e-10


def softplus2(x: torch.Tensor) -> torch.Tensor:
    """F(x) = log2(1 + 2**x), evaluated without overflow."""
    return torch.logaddexp(torch.zeros_like(x), LN2 * x) / LN2


def sigmoid2(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(LN2 * x)


def theta_to_phi(theta: torch.Tensor) -> torch.Tensor:
    d, loss, transfer = theta.unbind(dim=-1)
    u = torch.logaddexp(LN2 * d, LN2 * transfer) / LN2 - softplus2(loss)
    v = transfer - d
    return torch.stack((u, v, loss), dim=-1)


def phi_to_theta(phi: torch.Tensor) -> torch.Tensor:
    u, v, w = phi.unbind(dim=-1)
    d = u + softplus2(w) - softplus2(v)
    transfer = d + v
    return torch.stack((d, w, transfer), dim=-1)


def inverse_jacobian(phi: torch.Tensor) -> torch.Tensor:
    """J[..., k, i] = d theta_k / d phi_i, theta order D,L,T."""
    _u, v, w = phi.unbind(dim=-1)
    t, loss_share = sigmoid2(v), sigmoid2(w)
    one = torch.ones_like(t)
    zero = torch.zeros_like(t)
    row_d = torch.stack((one, -t, loss_share), dim=-1)
    row_l = torch.stack((zero, zero, one), dim=-1)
    row_t = torch.stack((one, one - t, loss_share), dim=-1)
    return torch.stack((row_d, row_l, row_t), dim=-2)


def probabilities(phi: torch.Tensor) -> torch.Tensor:
    """Return probabilities in S,D,L,T order."""
    u, v, w = phi.unbind(dim=-1)
    b, transfer_share, loss_share = sigmoid2(u), sigmoid2(v), sigmoid2(w)
    return torch.stack(
        ((1.0 - b) * (1.0 - loss_share), b * (1.0 - transfer_share),
         (1.0 - b) * loss_share, b * transfer_share), dim=-1
    )


def count_gradient(phi: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    """Gradient of expected complete NLL in u,v,w; counts order S,D,L,T."""
    ns, nd, nl, nt = counts.unbind(dim=-1)
    ntot = counts.sum(dim=-1)
    ndt = nd + nt
    nsl = ns + nl
    u, v, w = phi.unbind(dim=-1)
    return torch.stack(
        (ntot * sigmoid2(u) - ndt,
         ndt * sigmoid2(v) - nt,
         nsl * sigmoid2(w) - nl), dim=-1
    )


def complete_information(phi: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    """Expected complete-data curvature, exactly diagonal in u,v,w."""
    ns, nd, nl, nt = counts.unbind(dim=-1)
    ntot = counts.sum(dim=-1)
    ndt = nd + nt
    nsl = ns + nl
    u, v, w = phi.unbind(dim=-1)
    b, transfer_share, loss_share = sigmoid2(u), sigmoid2(v), sigmoid2(w)
    diagonal = LN2 * torch.stack(
        (ntot * b * (1.0 - b),
         ndt * transfer_share * (1.0 - transfer_share),
         nsl * loss_share * (1.0 - loss_share)), dim=-1
    )
    return torch.diag_embed(diagonal)


def transform_gradient(phi: torch.Tensor, gradient_theta: torch.Tensor) -> torch.Tensor:
    jac = inverse_jacobian(phi)
    return torch.einsum("...ki,...k->...i", jac, gradient_theta)


def transform_hessian(
    phi: torch.Tensor, gradient_theta: torch.Tensor, hessian_theta: torch.Tensor
) -> torch.Tensor:
    """Exact coordinate Hessian, including the gradient-times-map-curvature term."""
    jac = inverse_jacobian(phi)
    answer = torch.einsum("...ki,...kl,...lj->...ij", jac, hessian_theta, jac)
    _u, v, w = phi.unbind(dim=-1)
    group_gradient = gradient_theta[..., 0] + gradient_theta[..., 2]
    extra_v = -LN2 * sigmoid2(v) * (1.0 - sigmoid2(v)) * group_gradient
    extra_w = +LN2 * sigmoid2(w) * (1.0 - sigmoid2(w)) * group_gradient
    extra = torch.stack((torch.zeros_like(extra_v), extra_v, extra_w), dim=-1)
    answer = answer + torch.diag_embed(extra)
    return 0.5 * (answer + answer.transpose(-1, -2))


def multinomial_theta_information(theta: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    rates = torch.exp2(theta)
    p = rates / (1.0 + rates.sum(dim=-1, keepdim=True))
    covariance = torch.diag_embed(p) - p.unsqueeze(-1) * p.unsqueeze(-2)
    return LN2 * counts.sum(dim=-1)[..., None, None] * covariance


def projected_gradient_max(theta: torch.Tensor, gradient: torch.Tensor) -> torch.Tensor:
    outward = ((theta <= LO + BOUND_EPS) & (gradient > 0.0)) | (
        (theta >= HI - BOUND_EPS) & (gradient < 0.0)
    )
    return torch.where(outward, torch.zeros_like(gradient), gradient).abs().amax(dim=-1)


def bfgs_update(
    matrix: torch.Tensor, step: torch.Tensor, gradient_change: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    bs = torch.einsum("fij,fj->fi", matrix, step)
    sbs = (step * bs).sum(dim=-1)
    sy = (step * gradient_change).sum(dim=-1)
    good = (sy > BFGS_FLOOR * step.norm(dim=-1) * gradient_change.norm(dim=-1)) & (sbs > 0.0)
    safe_sy = torch.where(good, sy, torch.ones_like(sy))[:, None, None]
    safe_sbs = torch.where(good, sbs, torch.ones_like(sbs))[:, None, None]
    update = gradient_change.unsqueeze(-1) * gradient_change.unsqueeze(-2) / safe_sy
    update -= bs.unsqueeze(-1) * bs.unsqueeze(-2) / safe_sbs
    good = good & torch.isfinite(update).flatten(1).all(dim=-1)
    return matrix + torch.where(good[:, None, None], update, torch.zeros_like(update)), good


def theta_box_projected_step(
    theta: torch.Tensor,
    gradient_theta: torch.Tensor,
    curvature_phi: torch.Tensor,
    radius: float | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convexified phi step, projected via the real theta box and theta radius.

    This deliberately never constructs a uvw box.  A GPU replay must accept/reject
    the resulting point using its actual NLL.
    """
    phi = theta_to_phi(theta)
    jac = inverse_jacobian(phi)
    gradient_phi = transform_gradient(phi, gradient_theta)
    values, vectors = torch.linalg.eigh(curvature_phi)
    gradient_eigen = torch.einsum("fji,fj->fi", vectors, gradient_phi)
    theta_scale = torch.einsum("fki,fij->fkj", jac, vectors).norm(dim=1)
    lower = CURVATURE_FLOOR * theta_scale.square()
    radius_vector = (
        radius.to(dtype=theta.dtype, device=theta.device)
        if isinstance(radius, torch.Tensor)
        else torch.full((theta.shape[0],), float(radius), dtype=theta.dtype, device=theta.device)
    )
    trust = gradient_eigen.abs() * theta_scale / radius_vector[:, None]
    adjusted = torch.maximum(torch.maximum(values, lower), trust)
    direction = -torch.einsum("fij,fj->fi", vectors, gradient_eigen / adjusted)

    def land(alpha: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw_theta = phi_to_theta(phi + alpha[:, None] * direction)
        bounded_theta = torch.minimum(
            torch.maximum(raw_theta, torch.full_like(raw_theta, LO)), torch.full_like(raw_theta, HI)
        )
        return bounded_theta, theta_to_phi(bounded_theta)

    full_theta, _ = land(torch.ones(theta.shape[0]))
    capped = (full_theta - theta).norm(dim=-1) > radius_vector
    low = torch.zeros(theta.shape[0])
    high = torch.ones(theta.shape[0])
    for _ in range(50):
        middle = 0.5 * (low + high)
        middle_theta, _ = land(middle)
        over = (middle_theta - theta).norm(dim=-1) > radius_vector
        high = torch.where(over, middle, high)
        low = torch.where(over, low, middle)
    alpha = torch.where(capped, low, torch.ones_like(low))
    theta_new, phi_new = land(alpha)
    applied = phi_new - phi
    predicted = -(
        (gradient_phi * applied).sum(dim=-1)
        + 0.5 * torch.einsum("fi,fij,fj->f", applied, curvature_phi, applied)
    )
    return theta_new, capped, predicted


def quantile(value: torch.Tensor, probability: float) -> float:
    return float(value.quantile(probability))


def tensor_stats(value: torch.Tensor) -> dict[str, float]:
    return {
        "median": quantile(value, 0.5),
        "p90": quantile(value, 0.9),
        "max": float(value.max()),
    }


def weighted_quantile(value: torch.Tensor, weight: torch.Tensor, probability: float) -> float:
    order = torch.argsort(value)
    ordered_weight = weight[order]
    cumulative = ordered_weight.cumsum(dim=0)
    index = torch.searchsorted(cumulative, probability * ordered_weight.sum())
    index = index.minimum(torch.tensor(value.numel() - 1))
    return float(value[order[index]])


def validate_analytic_identities(data: dict[str, torch.Tensor], counts_data: dict[str, torch.Tensor]) -> dict:
    generator = torch.Generator().manual_seed(20260906)
    phi_random = -5.0 + 10.0 * torch.rand(32, 3, generator=generator)
    theta_random = phi_to_theta(phi_random)
    map_roundtrip = (theta_to_phi(theta_random) - phi_random).abs().max()

    def inverse_one(point: torch.Tensor) -> torch.Tensor:
        return phi_to_theta(point)

    jac_auto = vmap(jacrev(inverse_one))(phi_random)
    jac_error = (jac_auto - inverse_jacobian(phi_random)).abs().max()
    random_g = torch.randn(32, 3, generator=generator)
    random_h = torch.randn(32, 3, 3, generator=generator)
    random_h = 0.5 * (random_h + random_h.transpose(-1, -2))

    def scalar_one(point: torch.Tensor, gradient: torch.Tensor, hessian: torch.Tensor) -> torch.Tensor:
        theta_point = inverse_one(point)
        # The Hessian at theta=0 of this synthetic quadratic is exactly `hessian`,
        # while its gradient there is `gradient`.
        theta_origin = inverse_one(point).detach()
        displacement = theta_point - theta_origin
        return (gradient * theta_point).sum() + 0.5 * displacement @ hessian @ displacement

    h_auto = vmap(jacrev(jacrev(scalar_one)))(phi_random, random_g, random_h)
    h_formula = transform_hessian(phi_random, random_g, random_h)
    hessian_error = (h_auto - h_formula).abs().max()

    count_checks = {}
    complete_checks = {}
    for tag in ("start", "adam", "opt"):
        theta = data[f"theta_{tag}"]
        phi = theta_to_phi(theta)
        counts = counts_data[f"counts_{tag}"]
        transformed_stored = transform_gradient(phi, data[f"g_{tag}"])
        from_counts = count_gradient(phi, counts)
        count_checks[tag] = {
            "gradient_abs_max": float((transformed_stored - from_counts).abs().max()),
            "gradient_rel_median": quantile(
                (transformed_stored - from_counts).norm(dim=-1)
                / torch.maximum(transformed_stored.norm(dim=-1), torch.full((theta.shape[0],), 1.0e-12)), 0.5
            ),
        }
        theta_info = multinomial_theta_information(theta, counts)
        rates = torch.exp2(theta)
        event_probabilities = rates / (1.0 + rates.sum(dim=-1, keepdim=True))
        theta_count_gradient = event_probabilities * counts.sum(dim=-1, keepdim=True) - counts[:, 1:]
        transformed_info = transform_hessian(theta_to_phi(theta), theta_count_gradient, theta_info)
        diagonal_info = complete_information(phi, counts)
        complete_checks[tag] = {
            "offdiag_abs_max": float(
                (diagonal_info - torch.diag_embed(torch.diagonal(diagonal_info, dim1=-2, dim2=-1))).abs().max()
            ),
            "transformed_identity_abs_max": float((transformed_info - diagonal_info).abs().max()),
        }
    return {
        "random_map_roundtrip_abs_max": float(map_roundtrip),
        "random_inverse_jacobian_abs_max": float(jac_error),
        "random_hessian_transform_abs_max": float(hessian_error),
        "count_gradient": count_checks,
        "complete_information": complete_checks,
    }


def geometry_summary(data: dict[str, torch.Tensor], counts_data: dict[str, torch.Tensor]) -> dict:
    result = {}
    transformed = {}
    for tag in ("start", "adam", "opt"):
        phi = theta_to_phi(data[f"theta_{tag}"])
        hessian = transform_hessian(phi, data[f"g_{tag}"], data[f"H_{tag}"])
        information = complete_information(phi, counts_data[f"counts_{tag}"])
        eigen = torch.linalg.eigvalsh(hessian)
        information_root_inv = torch.diag_embed(torch.diagonal(information, dim1=-2, dim2=-1).rsqrt())
        normalized = information_root_inv @ hessian @ information_root_inv
        normalized_eigen = torch.linalg.eigvalsh(0.5 * (normalized + normalized.transpose(-1, -2)))
        positive = eigen[:, 0] > 0.0
        condition = eigen[:, 2] / eigen[:, 0]
        transformed[tag] = hessian
        result[tag] = {
            "indefinite_fraction": float((eigen[:, 0] < 0.0).double().mean()),
            "smallest_eigenvalue": tensor_stats(eigen[:, 0]),
            "condition_positive_median": quantile(condition[positive], 0.5) if bool(positive.any()) else math.inf,
            "observed_over_complete_eigenvalue_median": [
                quantile(normalized_eigen[:, index], 0.5) for index in range(3)
            ],
            "missing_fraction_largest_median": quantile(1.0 - normalized_eigen[:, 0], 0.5),
        }
    result["adam_to_opt_relative_drift"] = tensor_stats(
        (transformed["adam"] - transformed["opt"]).flatten(1).norm(dim=-1)
        / transformed["opt"].flatten(1).norm(dim=-1)
    )
    return result


def seed_screen(data: dict[str, torch.Tensor], counts_data: dict[str, torch.Tensor], radius: float) -> dict:
    theta_start, theta_adam = data["theta_start"], data["theta_adam"]
    phi_start, phi_adam = theta_to_phi(theta_start), theta_to_phi(theta_adam)
    gradient_start = transform_gradient(phi_start, data["g_start"])
    gradient_adam = transform_gradient(phi_adam, data["g_adam"])
    step = phi_adam - phi_start
    change = gradient_adam - gradient_start
    identity = torch.eye(3).expand(theta_adam.shape[0], 3, 3).clone()
    ic_adam = complete_information(phi_adam, counts_data["counts_adam"])

    secant_numerator = (step * change).sum(dim=-1)
    ic_step = torch.einsum("fij,fj->fi", ic_adam, step)
    secant_denominator = (step * ic_step).sum(dim=-1)
    valid_scale = (secant_numerator > 0.0) & (secant_denominator > 0.0)
    scale = torch.where(valid_scale, secant_numerator / secant_denominator, torch.ones_like(secant_numerator))
    scale = torch.minimum(torch.maximum(scale, torch.full_like(scale, 1.0e-3)), torch.full_like(scale, 1.0e3))
    ic_scaled = scale[:, None, None] * ic_adam
    ic_bfgs, ic_update_ok = bfgs_update(ic_adam, step, change)
    scaled_bfgs, scaled_update_ok = bfgs_update(ic_scaled, step, change)

    sy = secant_numerator
    yy = change.square().sum(dim=-1)
    bb_valid = sy > BFGS_FLOOR * step.norm(dim=-1) * change.norm(dim=-1)
    bb_scale = torch.where(bb_valid, yy / torch.where(bb_valid, sy, torch.ones_like(sy)), torch.ones_like(sy))
    bb_identity = bb_scale[:, None, None] * identity
    bb_bfgs, bb_update_ok = bfgs_update(bb_identity, step, change)

    carried = transform_hessian(phi_adam, data["g_adam"], data["curvature_adam_bfgs"])
    exact = transform_hessian(phi_adam, data["g_adam"], data["H_adam"])
    seeds = {
        "complete_diagonal": ic_adam,
        "secant_scaled_complete": ic_scaled,
        "complete_plus_one_bfgs": ic_bfgs,
        "scaled_complete_plus_one_bfgs": scaled_bfgs,
        "bb_identity_plus_one_bfgs": bb_bfgs,
        "pulled_back_production": carried,
        "exact_local_hessian_diagnostic": exact,
    }
    optimum = data["theta_opt"]
    base_distance = (theta_adam - optimum).abs().amax(dim=-1)
    clades = data["clades"]
    result = {
        "radius": radius,
        "base_distance": tensor_stats(base_distance),
        "secant_positive_fraction": float(valid_scale.double().mean()),
        "secant_scale": tensor_stats(scale),
        "bfgs_update_fraction": {
            "complete": float(ic_update_ok.double().mean()),
            "scaled_complete": float(scaled_update_ok.double().mean()),
            "bb": float(bb_update_ok.double().mean()),
        },
        "seeds": {},
    }
    exact_norm = exact.flatten(1).norm(dim=-1)
    for name, seed in seeds.items():
        theta_trial, capped, predicted = theta_box_projected_step(
            theta_adam, data["g_adam"], seed, radius
        )
        distance = (theta_trial - optimum).abs().amax(dim=-1)
        eigen = torch.linalg.eigvalsh(seed)
        result["seeds"][name] = {
            "relative_exact_hessian_error": tensor_stats(
                (seed - exact).flatten(1).norm(dim=-1) / exact_norm
            ),
            "indefinite_fraction": float((eigen[:, 0] < 0.0).double().mean()),
            "trial_distance": tensor_stats(distance),
            "trial_distance_clade_weighted_median": weighted_quantile(distance, clades, 0.5),
            "clade_weighted_distance_improvement_mean": float(
                ((base_distance - distance) * clades).sum() / clades.sum()
            ),
            "trial_below_one_fraction": float((distance < 1.0).double().mean()),
            "trial_below_one_clade_fraction": float(clades[distance < 1.0].sum() / clades.sum()),
            "capped_fraction": float(capped.double().mean()),
            "predicted_decrease_positive_fraction": float((predicted > 0.0).double().mean()),
            "predicted_decrease_sum_bits": float(predicted.sum()),
        }
    return result


def load_inputs(curvature_path: Path, gradient_path: Path, counts_path: Path) -> tuple[dict, dict]:
    curvature = torch.load(curvature_path, map_location="cpu", weights_only=False)
    gradient = torch.load(gradient_path, map_location="cpu", weights_only=False)
    counts = torch.load(counts_path, map_location="cpu", weights_only=False)
    data = {}
    for tag in ("start", "adam", "opt"):
        data[f"theta_{tag}"] = curvature[f"theta_{tag}"].double()
        data[f"H_{tag}"] = curvature[f"H_{tag}"].double()
        data[f"g_{tag}"] = gradient[f"g_{tag}"].double()
        data[f"nll_{tag}"] = gradient[f"nll_{tag}"].double()
    data["curvature_adam_bfgs"] = curvature["curvature_adam_bfgs"].double()
    data["clades"] = torch.as_tensor(curvature["features"]["clades"], dtype=torch.float64)
    counts_data = {
        f"counts_{tag}": -counts[f"a_total_{tag}"].double() for tag in ("start", "adam", "opt")
    }
    path_names = lambda paths: [Path(path).name for path in paths]
    if (path_names(curvature["paths"]) != path_names(gradient["paths"])
            or path_names(curvature["paths"]) != path_names(counts["paths"])):
        raise ValueError("family order differs across saved inputs")
    return data, counts_data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curvature", required=True, type=Path)
    parser.add_argument("--gradients", required=True, type=Path)
    parser.add_argument("--counts", required=True, type=Path)
    parser.add_argument("--radius", required=True, type=float)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    data, counts_data = load_inputs(arguments.curvature, arguments.gradients, arguments.counts)
    output = {
        "inputs": {
            "curvature": str(arguments.curvature),
            "gradients": str(arguments.gradients),
            "counts": str(arguments.counts),
        },
        "validation": validate_analytic_identities(data, counts_data),
        "geometry": geometry_summary(data, counts_data),
        "seed_screen": seed_screen(data, counts_data, arguments.radius),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
