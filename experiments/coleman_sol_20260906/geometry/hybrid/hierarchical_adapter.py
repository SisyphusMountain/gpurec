"""Pure tensor geometry for the experiment-only post-EM hierarchical fit."""
from __future__ import annotations

import math

import torch


LN2 = math.log(2.0)


def softplus2(x: torch.Tensor) -> torch.Tensor:
    return torch.logaddexp(torch.zeros_like(x), x * LN2) / LN2


def theta_to_phi(theta: torch.Tensor) -> torch.Tensor:
    d, l, t = theta.unbind(dim=-1)
    u = torch.logaddexp(d * LN2, t * LN2) / LN2 - softplus2(l)
    return torch.stack((u, t - d, l), dim=-1)


def phi_to_theta(phi: torch.Tensor) -> torch.Tensor:
    u, v, w = phi.unbind(dim=-1)
    fw, fv = softplus2(w), softplus2(v)
    return torch.stack((u + fw - fv, w, u + fw + v - fv), dim=-1)


def jacobian(phi: torch.Tensor) -> torch.Tensor:
    _u, v, w = phi.unbind(dim=-1)
    t = torch.sigmoid(v * LN2)
    l = torch.sigmoid(w * LN2)
    one, zero = torch.ones_like(t), torch.zeros_like(t)
    return torch.stack((
        torch.stack((one, -t, l), dim=-1),
        torch.stack((zero, zero, one), dim=-1),
        torch.stack((one, one - t, l), dim=-1),
    ), dim=-2)


def map_hessians(phi: torch.Tensor) -> torch.Tensor:
    """Return d2 theta_k / d phi2 as ``[..., theta_k, phi, phi]``."""
    _u, v, w = phi.unbind(dim=-1)
    t = torch.sigmoid(v * LN2)
    l = torch.sigmoid(w * LN2)
    hv = -LN2 * t * (1.0 - t)
    hw = +LN2 * l * (1.0 - l)
    out = phi.new_zeros((*phi.shape[:-1], 3, 3, 3))
    out[..., 0, 1, 1] = hv
    out[..., 0, 2, 2] = hw
    out[..., 2, 1, 1] = hv
    out[..., 2, 2, 2] = hw
    return out


def transform_gradient(phi: torch.Tensor, gradient_theta: torch.Tensor) -> torch.Tensor:
    return torch.einsum("...ji,...j->...i", jacobian(phi), gradient_theta)


def transform_hessian(
    phi: torch.Tensor,
    gradient_theta: torch.Tensor,
    hessian_theta: torch.Tensor,
) -> torch.Tensor:
    j = jacobian(phi)
    congruence = torch.einsum("...ki,...kl,...lj->...ij", j, hessian_theta, j)
    correction = torch.einsum("...k,...kij->...ij", gradient_theta, map_hessians(phi))
    result = congruence + correction
    return 0.5 * (result + result.transpose(-1, -2))


def complete_information(phi: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    ns, nd, nl, nt = counts.unbind(dim=-1)
    u, v, w = phi.unbind(dim=-1)
    b = torch.sigmoid(u * LN2)
    t = torch.sigmoid(v * LN2)
    l = torch.sigmoid(w * LN2)
    diagonal = LN2 * torch.stack((
        (ns + nd + nl + nt) * b * (1.0 - b),
        (nd + nt) * t * (1.0 - t),
        (ns + nl) * l * (1.0 - l),
    ), dim=-1)
    return torch.diag_embed(diagonal)


def blocking_faces(
    theta: torch.Tensor,
    gradient_theta: torch.Tensor,
    lo: float,
    hi: float,
    eps: float,
) -> torch.Tensor:
    return (((theta >= hi - eps) & (gradient_theta < 0))
            | ((theta <= lo + eps) & (gradient_theta > 0)))


def tangent_projector(phi: torch.Tensor, fixed: torch.Tensor) -> torch.Tensor:
    """Euclidean projector onto ``{d: J[fixed] d = 0}``, batched over families."""
    j = jacobian(phi)
    identity = torch.eye(3, dtype=phi.dtype, device=phi.device).expand(*phi.shape[:-1], 3, 3)
    active = fixed.to(phi.dtype)
    a = j * active.unsqueeze(-1)
    # Inactive zero rows receive a unit diagonal, making the 3x3 Gram solve nonsingular without
    # changing A.T @ solve(G,A). This avoids mask-dependent GPU branching and synchronization.
    gram = a @ a.transpose(-1, -2) + torch.diag_embed(1.0 - active)
    result = identity - a.transpose(-1, -2) @ torch.linalg.solve(gram, a)
    return 0.5 * (result + result.transpose(-1, -2))


def face_step_model(
    phi: torch.Tensor,
    gradient_theta: torch.Tensor,
    gradient_phi: torch.Tensor,
    curvature_phi: torch.Tensor,
    fixed: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    p = tangent_projector(phi, fixed)
    mapping_correction = torch.einsum(
        "...k,...kij->...ij", gradient_theta * fixed.to(gradient_theta.dtype), map_hessians(phi),
    )
    corrected = curvature_phi - mapping_correction
    gradient_face = torch.einsum("...ij,...j->...i", p, gradient_phi)
    return p, 0.5 * (corrected + corrected.transpose(-1, -2)), gradient_face


def regularized_tangent_direction(
    phi: torch.Tensor,
    gradient_theta: torch.Tensor,
    gradient_phi: torch.Tensor,
    curvature_phi: torch.Tensor,
    fixed: torch.Tensor,
    radius: torch.Tensor,
    mu: float,
    metric: str = "native",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return direction, regularized matrix, projected gradient, and eigen-limit indicator."""
    p, corrected, gradient_face = face_step_model(
        phi, gradient_theta, gradient_phi, curvature_phi, fixed,
    )
    # Match production's order: regularize the full curvature first, then reduce the step to the
    # active tangent. Projecting before eigh changes even the free block when B couples it to a
    # blocked direction and would make the native identity adapter an invalid control.
    eigenvalues, vectors = torch.linalg.eigh(corrected)
    gradient_eigen = torch.einsum("...ji,...j->...i", vectors, gradient_face)
    if metric == "native":
        direction_scale = torch.einsum(
            "...ki,...ij->...kj", jacobian(phi), vectors,
        ).norm(dim=-2)
    elif metric == "coordinate":
        direction_scale = torch.ones_like(eigenvalues)
    else:
        raise ValueError(f"metric must be 'native' or 'coordinate', got {metric!r}")
    radius_column = radius.unsqueeze(-1)
    adjusted = torch.maximum(
        torch.maximum(eigenvalues, float(mu) * direction_scale.square()),
        gradient_eigen.abs() * direction_scale / radius_column,
    )
    full_regularized = vectors @ torch.diag_embed(adjusted) @ vectors.transpose(-1, -2)
    identity = torch.eye(3, dtype=phi.dtype, device=phi.device).expand_as(full_regularized)
    regularized = p @ full_regularized @ p + identity - p
    direction = -torch.linalg.solve(regularized, gradient_face.unsqueeze(-1)).squeeze(-1)
    direction = torch.einsum("...ij,...j->...i", p, direction)
    # Per-eigendirection limiting can still produce a combined native norm above radius; retraction
    # below applies the authoritative norm gate. This flag is completed there.
    eigen_limited = (adjusted > torch.maximum(
        eigenvalues, float(mu) * direction_scale.square())).any(dim=-1)
    return direction, regularized, gradient_face, eigen_limited


def working_set_tangent_direction(
    phi: torch.Tensor,
    theta: torch.Tensor,
    gradient_theta: torch.Tensor,
    gradient_phi: torch.Tensor,
    curvature_phi: torch.Tensor,
    fixed: torch.Tensor,
    radius: torch.Tensor,
    mu: float,
    lo: float,
    hi: float,
    bound_eps: float,
    metric: str = "native",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Add currently violated bound faces to the step-only working set, then solve.

    ``fixed`` remains the authoritative outward-gradient/KKT mask. The returned ``working`` mask
    additionally holds a current native face when coupling makes ``J d`` point out of the box.
    This secondary mask affects only this step model and retraction, never projected-gradient
    stopping, freezing, or certification.
    """
    working = fixed.clone()
    eigen_limited = torch.zeros(phi.shape[:-1], dtype=torch.bool, device=phi.device)
    # Three native coordinates can be added. A fourth solve uses the final promoted mask.
    for iteration in range(4):
        direction, regularized, gradient_face, limited = regularized_tangent_direction(
            phi, gradient_theta, gradient_phi, curvature_phi, working, radius, mu, metric,
        )
        eigen_limited = eigen_limited | limited
        if iteration == 3:
            break
        native_linear = torch.einsum("...ki,...i->...k", jacobian(phi), direction)
        violates = (((theta <= lo + bound_eps) & (native_linear < 0))
                    | ((theta >= hi - bound_eps) & (native_linear > 0)))
        working = working | violates
    return direction, regularized, gradient_face, working, eigen_limited


def retract_feasible_ray_cpu64(
    phi: torch.Tensor,
    theta: torch.Tensor,
    direction: torch.Tensor,
    fixed: torch.Tensor,
    lo: float,
    hi: float,
    radius: torch.Tensor,
    iterations: int = 32,
    metric: str = "native",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Bisect a feasible ray endpoint inside the native box and physical trust radius.

    Only endpoint feasibility is tested: because the nonlinear native image of a phi ray need not
    be monotone, this deliberately makes no global "first face hit" claim.
    """
    original_device, original_dtype = phi.device, phi.dtype
    th = theta.detach().double().cpu()
    # The model evaluated ``theta`` (often float32), so its widened native value is authoritative.
    # Re-encoding it makes alpha=0 exactly that endpoint instead of a separately rounded phi map.
    z = theta_to_phi(th)
    d = direction.detach().double().cpu()
    fx = fixed.detach().cpu()
    rad = radius.detach().double().cpu()
    # Match the scalar bounds after conversion to the model dtype (the production clamp's actual
    # faces), then permit only float64 evaluation noise in endpoint feasibility tests.
    lo_effective = float(torch.tensor(lo, dtype=original_dtype).double())
    hi_effective = float(torch.tensor(hi, dtype=original_dtype).double())
    feasibility_tol = 8.0 * torch.finfo(torch.float64).eps * max(
        1.0, abs(lo_effective), abs(hi_effective),
    )

    def endpoint(alpha: torch.Tensor) -> torch.Tensor:
        raw = phi_to_theta(z + alpha[:, None] * d)
        return torch.where(fx, th, raw)

    trial_one = endpoint(torch.ones(z.shape[0], dtype=torch.float64))
    if metric == "native":
        norm_one = (trial_one - th).norm(dim=1)
        norm_at = lambda alpha, trial: (trial - th).norm(dim=1)
    elif metric == "coordinate":
        norm_one = d.norm(dim=1)
        norm_at = lambda alpha, trial: (alpha[:, None] * d).norm(dim=1)
    else:
        raise ValueError(f"metric must be 'native' or 'coordinate', got {metric!r}")
    feasible_one = (((trial_one >= lo_effective - feasibility_tol)
                     & (trial_one <= hi_effective + feasibility_tol)).all(dim=1)
                    & (norm_one <= rad + feasibility_tol))
    low = torch.where(feasible_one, torch.ones_like(rad), torch.zeros_like(rad))
    high = torch.ones_like(rad)
    for _ in range(iterations):
        middle = 0.5 * (low + high)
        trial = endpoint(middle)
        feasible = (((trial >= lo_effective - feasibility_tol)
                     & (trial <= hi_effective + feasibility_tol)).all(dim=1)
                    & (norm_at(middle, trial) <= rad + feasibility_tol))
        low = torch.where(feasible, middle, low)
        high = torch.where(feasible, high, middle)

    theta_new = endpoint(low).clamp(min=lo_effective, max=hi_effective)
    theta_new = torch.where(fx, th, theta_new)
    phi_new = theta_to_phi(theta_new)
    applied_tangent = low[:, None] * d
    chord = phi_new - z
    trust_capped = norm_one > rad
    return tuple(x.to(device=original_device, dtype=original_dtype) for x in (
        phi_new, theta_new, applied_tangent, chord,
    )) + (trust_capped.to(device=original_device),)


def predicted_decrease(
    gradient_face: torch.Tensor,
    regularized: torch.Tensor,
    applied_tangent: torch.Tensor,
) -> torch.Tensor:
    return -(
        (gradient_face * applied_tangent).sum(dim=-1)
        + 0.5 * torch.einsum("...i,...ij,...j->...", applied_tangent, regularized, applied_tangent)
    )
