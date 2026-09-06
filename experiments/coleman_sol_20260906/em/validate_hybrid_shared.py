"""CPU validation for point-consistent ``hybrid_shared_200_v2.pt`` seeds."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch.func import grad, hessian, jacrev, vmap

from gpurec.fit.em_warmup import boxed_em_m_step
from gpurec.fit.genewise_fit import _bfgs_update

from hybrid_geometry import (
    calibrated_bfgs_seed,
    complete_gradient_theta,
    complete_information_z,
    exact_transform_complete_hessian,
    fixed_count_surrogate_nll_z,
    jacobian_theta_wrt_z,
    theta_from_z,
    transform_gradient,
    z_from_theta,
)


torch.set_default_dtype(torch.float64)


def _maximum(value: torch.Tensor) -> float:
    return float(value.detach().abs().max())


def _off_diagonal(matrix: torch.Tensor) -> torch.Tensor:
    return matrix - torch.diag_embed(torch.diagonal(matrix, dim1=-2, dim2=-1))


def _autograd_hessians(z: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    single = hessian(lambda zz, nn: fixed_count_surrogate_nll_z(zz, nn), argnums=0)
    return vmap(single)(z, counts)


def _native_seed_reference(
    endpoint: torch.Tensor,
    counts: torch.Tensor,
    first: torch.Tensor,
    second: torch.Tensor,
    gradient_first: torch.Tensor,
    gradient_second: torch.Tensor,
    lo: float,
    hi: float,
) -> torch.Tensor:
    from hybrid_geometry import complete_information_theta

    information = complete_information_theta(endpoint, counts)
    step = second - first
    change = gradient_second - gradient_first
    information_step = torch.einsum("fij,fj->fi", information, step)
    sy = (step * change).sum(dim=-1)
    sbs = (step * information_step).sum(dim=-1)
    good = (sy > 0.0) & (sbs > 0.0) & torch.isfinite(sy) & torch.isfinite(sbs)
    scale = torch.where(good, sy / torch.where(good, sbs, torch.ones_like(sbs)), torch.ones_like(sy))
    free = (
        (first > lo + 1.0e-6)
        & (first < hi - 1.0e-6)
        & (second > lo + 1.0e-6)
        & (second < hi - 1.0e-6)
    ).to(first.dtype)
    return _bfgs_update(information * scale[:, None, None], step, change, free)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    artifact = torch.load(args.artifact, map_location="cpu", weights_only=False)
    if artifact.get("schema") != "gpurec.em_hybrid_shared.v2":
        raise ValueError(f"expected point-consistent V2 artifact, got {artifact.get('schema')!r}")
    theta0 = artifact["theta_native"]["theta0"].double()
    theta1 = artifact["theta_native"]["theta1"].double()
    theta2 = artifact["theta_native"]["theta2"].double()
    raw_theta1 = artifact["theta_native_raw_mstep"]["theta1"].double()
    raw_theta2 = artifact["theta_native_raw_mstep"]["theta2"].double()
    g0 = artifact["gradient_theta"]["g0"].double()
    g1 = artifact["gradient_theta"]["g1"].double()
    n0 = artifact["event_counts"]["N0"].double()
    n1 = artifact["event_counts"]["N1"].double()
    hierarchical = artifact["hierarchical"]
    lo, hi = artifact["metadata"]["rate_box_log2"]

    assert theta0.shape == theta1.shape == theta2.shape == g0.shape == g1.shape == (200, 3)
    assert n0.shape == n1.shape == (200, 4)
    assert all(bool(torch.isfinite(value).all()) for value in (theta0, theta1, theta2, g0, g1, n0, n1))
    assert bool((n0 > 0.0).all()) and bool((n1 > 0.0).all())
    assert bool((theta1 >= lo).all() and (theta1 <= hi).all())
    assert bool((theta2 >= lo).all() and (theta2 <= hi).all())

    theta1_recomputed = boxed_em_m_step(n0, lo, hi)
    theta2_recomputed = boxed_em_m_step(n1, lo, hi)
    z0, z1, z2 = z_from_theta(theta0), z_from_theta(theta1), z_from_theta(theta2)
    jacobian_formula = jacobian_theta_wrt_z(z2)
    jacobian_autograd = vmap(jacrev(theta_from_z))(z2)

    # Differentiate theta(z).g directly: this tests g_z=J.T g without reusing the formula.
    gradient_pullback = vmap(grad(lambda zz, gg: (theta_from_z(zz) * gg).sum(), argnums=0))(z1, g1)
    g_z0 = transform_gradient(z0, g0)
    g_z1 = transform_gradient(z1, g1)

    information_z = complete_information_z(z2, n1)
    hessian_autograd = _autograd_hessians(z2, n1)
    hessian_chain, raw_pullback = exact_transform_complete_hessian(theta2, n1)
    seed, seed_details = calibrated_bfgs_seed(information_z, z1 - z0, g_z1 - g_z0)
    seed_eigenvalues = torch.linalg.eigvalsh(seed)
    secant_residual = torch.einsum("fij,fj->fi", seed, z1 - z0) - (g_z1 - g_z0)

    # Fisher identity check uses the exact points evaluated by the float32 production API.
    complete_g0 = complete_gradient_theta(theta0, n0)
    complete_g1 = complete_gradient_theta(theta1, n1)

    native_control = artifact["native_control"]
    native_consistent_recomputed = _native_seed_reference(theta2, n1, theta0, theta1, g0, g1, lo, hi)
    native_legacy_recomputed = _native_seed_reference(raw_theta2, n1, theta0, raw_theta1, g0, g1, lo, hi)

    # Deliberately extreme counts produce true constrained M-step endpoints on both box faces.
    # They test the chain-rule correction where the fixed-count native gradient is nonzero.
    boundary_counts = torch.tensor(
        [
            [100.0, 1.0e-9, 1.0, 1.0],
            [1.0, 1000.0, 1.0, 1.0],
            [1.0, 1.0, 1000.0, 1.0],
            [1.0, 1.0, 1.0, 1000.0],
            [1.0, 1000.0, 1000.0, 1.0],
            [1.0, 1000.0, 1.0, 1000.0],
            [0.1, 100.0, 1000.0, 10.0],
        ],
        dtype=torch.float64,
    )
    boundary_theta = boxed_em_m_step(boundary_counts, lo, hi)
    boundary_z = z_from_theta(boundary_theta)
    boundary_direct = complete_information_z(boundary_z, boundary_counts)
    boundary_autograd = _autograd_hessians(boundary_z, boundary_counts)
    boundary_chain, boundary_raw = exact_transform_complete_hessian(boundary_theta, boundary_counts)
    boundary_state = torch.where(
        boundary_theta <= lo + 1.0e-10,
        torch.full_like(boundary_theta, -1, dtype=torch.int64),
        torch.where(
            boundary_theta >= hi - 1.0e-10,
            torch.full_like(boundary_theta, 1, dtype=torch.int64),
            torch.zeros_like(boundary_theta, dtype=torch.int64),
        ),
    )

    result = {
        "artifact": args.artifact,
        "n_families": 200,
        "trajectory": {
            "raw_theta1_mstep_abs_max": _maximum(raw_theta1 - theta1_recomputed),
            "raw_theta2_mstep_abs_max": _maximum(raw_theta2 - theta2_recomputed),
            "theta1_actual_cast_abs_max": _maximum(theta1 - raw_theta1.float().double()),
            "theta2_actual_cast_abs_max": _maximum(theta2 - raw_theta2.float().double()),
            "theta1_fp32_rounding_abs_max": _maximum(theta1 - raw_theta1),
            "theta2_fp32_rounding_abs_max": _maximum(theta2 - raw_theta2),
            "nll_drop_0_to_1_bits": float(
                artifact["nll_per_family_bits"]["nll0"].sum()
                - artifact["nll_per_family_bits"]["nll1"].sum()
            ),
            "counts_strictly_positive": bool((n0 > 0).all() and (n1 > 0).all()),
            "count_gradient_g0_abs_max": _maximum(complete_g0 - g0),
            "count_gradient_g1_abs_max": _maximum(complete_g1 - g1),
        },
        "coordinate_checks": {
            "roundtrip_abs_max": _maximum(theta_from_z(z2) - theta2),
            "jacobian_abs_max": _maximum(jacobian_formula - jacobian_autograd),
            "gradient_pullback_abs_max": _maximum(g_z1 - gradient_pullback),
            "stored_z_abs_max": max(
                _maximum(z0 - hierarchical["z0"]),
                _maximum(z1 - hierarchical["z1"]),
                _maximum(z2 - hierarchical["z2"]),
            ),
        },
        "endpoint_information": {
            "direct_vs_autograd_abs_max": _maximum(information_z - hessian_autograd),
            "chain_rule_vs_autograd_abs_max": _maximum(hessian_chain - hessian_autograd),
            "raw_pullback_off_diagonal_abs_max": _maximum(_off_diagonal(raw_pullback)),
            "raw_pullback_diagonal_error_abs_max": _maximum(raw_pullback - information_z),
        },
        "seed": {
            "stored_seed_abs_max": _maximum(seed - hierarchical["scaled_bfgs_seed_z"]),
            "scale_valid": int(seed_details["scale_valid"].sum()),
            "bfgs_valid": int(seed_details["bfgs_valid"].sum()),
            "scale_min": float(seed_details["scale"].min()),
            "scale_median": float(seed_details["scale"].median()),
            "scale_p90": float(seed_details["scale"].quantile(0.9)),
            "scale_max": float(seed_details["scale"].max()),
            "minimum_eigenvalue": float(seed_eigenvalues.min()),
            "secant_residual_abs_max": _maximum(secant_residual),
        },
        "native_control": {
            "consistent_seed_recompute_abs_max": _maximum(
                native_consistent_recomputed
                - native_control["consistent_evaluated_points"]["seed"]
            ),
            "legacy_seed_recompute_abs_max": _maximum(
                native_legacy_recomputed
                - native_control["legacy_inline_raw_cpu_points"]["seed"]
            ),
            **native_control["comparison"],
        },
        "boundary_cases": {
            "count": int(boundary_counts.shape[0]),
            "active_coordinates": int((boundary_state != 0).sum()),
            "active_state": boundary_state.tolist(),
            "direct_vs_autograd_abs_max": _maximum(boundary_direct - boundary_autograd),
            "chain_rule_vs_autograd_abs_max": _maximum(boundary_chain - boundary_autograd),
            "raw_pullback_off_diagonal_abs_max": _maximum(_off_diagonal(boundary_raw)),
            "raw_pullback_diagonal_error_abs_max": _maximum(boundary_raw - boundary_direct),
        },
        "conclusion": (
            "The fixed-count surrogate Hessian is diagonal in z everywhere, including constrained "
            "native-box M-step endpoints. The nonlinear chain-rule correction is diagonal too, so "
            "nonstationarity changes diagonal entries but cannot create cross terms for this map."
        ),
    }

    # Tight analytic checks. Production count-gradient agreement is intentionally looser because
    # g and counts arrive through independent float32 atomic accumulations in the same reverse pass.
    assert result["trajectory"]["raw_theta1_mstep_abs_max"] == 0.0
    assert result["trajectory"]["raw_theta2_mstep_abs_max"] == 0.0
    assert result["trajectory"]["theta1_actual_cast_abs_max"] == 0.0
    assert result["trajectory"]["theta2_actual_cast_abs_max"] == 0.0
    assert result["coordinate_checks"]["roundtrip_abs_max"] < 5.0e-15
    assert result["coordinate_checks"]["jacobian_abs_max"] < 5.0e-15
    assert result["coordinate_checks"]["gradient_pullback_abs_max"] < 5.0e-12
    assert result["endpoint_information"]["direct_vs_autograd_abs_max"] < 5.0e-10
    assert result["endpoint_information"]["chain_rule_vs_autograd_abs_max"] < 5.0e-10
    assert result["boundary_cases"]["direct_vs_autograd_abs_max"] < 5.0e-10
    assert result["boundary_cases"]["chain_rule_vs_autograd_abs_max"] < 5.0e-10
    assert result["boundary_cases"]["active_coordinates"] >= 6
    assert result["seed"]["minimum_eigenvalue"] > 0.0
    assert result["seed"]["scale_valid"] == 200
    assert result["seed"]["bfgs_valid"] == 200
    assert result["native_control"]["consistent_seed_recompute_abs_max"] == 0.0
    assert result["native_control"]["legacy_seed_recompute_abs_max"] == 0.0

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
