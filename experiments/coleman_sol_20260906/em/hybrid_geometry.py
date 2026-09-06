"""Hierarchical-coordinate seed calculations for an EM2 warm trajectory.

The native coordinates are ``theta=(log2 D, log2 L, log2 T)``.  The hierarchical
coordinates ``z=(u,v,w)`` are the binary event-tree logits

    u = log2((D+T)/(1+L)),  v = log2(T/D),  w = log2(L).

This module deliberately constructs curvature directly in z coordinates.  It
never transforms or reuses a native-coordinate optimizer matrix.
"""
from __future__ import annotations

import math

import torch


LN2 = math.log(2.0)
BFGS_CURVATURE_FLOOR = 1.0e-10


def softplus2(value: torch.Tensor) -> torch.Tensor:
    """Stable ``log2(1 + 2**value)``."""
    return torch.logaddexp(torch.zeros_like(value), LN2 * value) / LN2


def sigmoid2(value: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(LN2 * value)


def z_from_theta(theta: torch.Tensor) -> torch.Tensor:
    d, loss, transfer = theta.unbind(dim=-1)
    u = torch.logaddexp(LN2 * d, LN2 * transfer) / LN2 - softplus2(loss)
    return torch.stack((u, transfer - d, loss), dim=-1)


def theta_from_z(z: torch.Tensor) -> torch.Tensor:
    u, v, w = z.unbind(dim=-1)
    d = u + softplus2(w) - softplus2(v)
    return torch.stack((d, w, d + v), dim=-1)


def jacobian_theta_wrt_z(z: torch.Tensor) -> torch.Tensor:
    """Return ``J[...,k,i] = d theta_k / d z_i`` in D,L,T / u,v,w order."""
    _u, v, w = z.unbind(dim=-1)
    transfer_share = sigmoid2(v)
    loss_share = sigmoid2(w)
    one = torch.ones_like(transfer_share)
    zero = torch.zeros_like(transfer_share)
    return torch.stack(
        (
            torch.stack((one, -transfer_share, loss_share), dim=-1),
            torch.stack((zero, zero, one), dim=-1),
            torch.stack((one, one - transfer_share, loss_share), dim=-1),
        ),
        dim=-2,
    )


def transform_gradient(z: torch.Tensor, gradient_theta: torch.Tensor) -> torch.Tensor:
    """Apply ``g_z = J.T g_theta`` without transforming any curvature matrix."""
    return torch.einsum("...ki,...k->...i", jacobian_theta_wrt_z(z), gradient_theta)


def event_probabilities_z(z: torch.Tensor) -> torch.Tensor:
    """Return probabilities in S,D,L,T order."""
    u, v, w = z.unbind(dim=-1)
    branch = sigmoid2(u)
    transfer_share = sigmoid2(v)
    loss_share = sigmoid2(w)
    return torch.stack(
        (
            (1.0 - branch) * (1.0 - loss_share),
            branch * (1.0 - transfer_share),
            (1.0 - branch) * loss_share,
            branch * transfer_share,
        ),
        dim=-1,
    )


def fixed_count_surrogate_nll_z(z: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    """Complete-data negative log likelihood in bits, up to a constant."""
    ns, nd, nl, nt = counts.unbind(dim=-1)
    ntotal = ns + nd + nl + nt
    ndt = nd + nt
    nsl = ns + nl
    u, v, w = z.unbind(dim=-1)
    return (
        ntotal * softplus2(u) - ndt * u
        + ndt * softplus2(v) - nt * v
        + nsl * softplus2(w) - nl * w
    )


def complete_information_z(z: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    """Exact fixed-count Hessian, diagonal everywhere in hierarchical coordinates."""
    ns, nd, nl, nt = counts.unbind(dim=-1)
    ntotal = ns + nd + nl + nt
    ndt = nd + nt
    nsl = ns + nl
    u, v, w = z.unbind(dim=-1)
    branch = sigmoid2(u)
    transfer_share = sigmoid2(v)
    loss_share = sigmoid2(w)
    diagonal = LN2 * torch.stack(
        (
            ntotal * branch * (1.0 - branch),
            ndt * transfer_share * (1.0 - transfer_share),
            nsl * loss_share * (1.0 - loss_share),
        ),
        dim=-1,
    )
    return torch.diag_embed(diagonal)


def complete_gradient_theta(theta: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    rates = torch.exp2(theta)
    probabilities = rates / (1.0 + rates.sum(dim=-1, keepdim=True))
    return counts.sum(dim=-1, keepdim=True) * probabilities - counts[..., 1:]


def complete_information_theta(theta: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    rates = torch.exp2(theta)
    probabilities = rates / (1.0 + rates.sum(dim=-1, keepdim=True))
    covariance = torch.diag_embed(probabilities) - probabilities.unsqueeze(-1) * probabilities.unsqueeze(-2)
    return LN2 * counts.sum(dim=-1)[..., None, None] * covariance


def exact_transform_complete_hessian(
    theta: torch.Tensor, counts: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return exact z Hessian and the incomplete raw pullback ``J.T H J``.

    The raw pullback is already diagonal for this factorized map. Away from an
    unconstrained stationary point, the diagonal gradient-times-map-curvature
    chain-rule term changes its diagonal entries. It does not cancel or create
    cross terms, including at constrained native-box M-step endpoints.
    """
    z = z_from_theta(theta)
    jacobian = jacobian_theta_wrt_z(z)
    gradient_theta = complete_gradient_theta(theta, counts)
    information_theta = complete_information_theta(theta, counts)
    raw = torch.einsum("...ki,...kl,...lj->...ij", jacobian, information_theta, jacobian)
    _u, v, w = z.unbind(dim=-1)
    grouped_gradient = gradient_theta[..., 0] + gradient_theta[..., 2]
    correction = torch.diag_embed(
        torch.stack(
            (
                torch.zeros_like(grouped_gradient),
                -LN2 * sigmoid2(v) * (1.0 - sigmoid2(v)) * grouped_gradient,
                +LN2 * sigmoid2(w) * (1.0 - sigmoid2(w)) * grouped_gradient,
            ),
            dim=-1,
        )
    )
    exact = raw + correction
    return 0.5 * (exact + exact.transpose(-1, -2)), raw


def calibrated_bfgs_seed(
    information: torch.Tensor,
    step: torch.Tensor,
    gradient_change: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Scalar-calibrate z information on the latest z secant, then BFGS-fold it."""
    information_step = torch.einsum("fij,fj->fi", information, step)
    sy = (step * gradient_change).sum(dim=-1)
    sbs = (step * information_step).sum(dim=-1)
    scale_valid = (sy > 0.0) & (sbs > 0.0) & torch.isfinite(sy) & torch.isfinite(sbs)
    scale = torch.where(scale_valid, sy / torch.where(scale_valid, sbs, torch.ones_like(sbs)), torch.ones_like(sy))
    scaled = information * scale[:, None, None]

    scaled_step = torch.einsum("fij,fj->fi", scaled, step)
    scaled_sbs = (step * scaled_step).sum(dim=-1)
    curvature_floor = BFGS_CURVATURE_FLOOR * step.norm(dim=-1) * gradient_change.norm(dim=-1)
    update_valid = (
        (sy > curvature_floor)
        & (scaled_sbs > 0.0)
        & torch.isfinite(sy)
        & torch.isfinite(scaled_sbs)
    )
    safe_sy = torch.where(update_valid, sy, torch.ones_like(sy))[:, None, None]
    safe_sbs = torch.where(update_valid, scaled_sbs, torch.ones_like(scaled_sbs))[:, None, None]
    update = (
        gradient_change.unsqueeze(-1) * gradient_change.unsqueeze(-2) / safe_sy
        - scaled_step.unsqueeze(-1) * scaled_step.unsqueeze(-2) / safe_sbs
    )
    update_finite = torch.isfinite(update).flatten(1).all(dim=-1)
    update_valid = update_valid & update_finite
    seed = scaled + torch.where(update_valid[:, None, None], update, torch.zeros_like(update))
    seed = 0.5 * (seed + seed.transpose(-1, -2))
    return seed, {
        "scale": scale,
        "scale_valid": scale_valid,
        "bfgs_valid": update_valid,
        "sy": sy,
        "s_information_s": sbs,
        "s_scaled_information_s": scaled_sbs,
    }
