"""First-Newton reseeding from the already-paid EM1 -> EM2 gradient secant."""
from __future__ import annotations

import math

import torch

from gpurec.fit.genewise_fit import _bfgs_update


LN2 = math.log(2.0)


def _softplus2(value: torch.Tensor) -> torch.Tensor:
    return torch.logaddexp(torch.zeros_like(value), value * LN2) / LN2


def theta_to_phi(theta: torch.Tensor) -> torch.Tensor:
    d, loss, transfer = theta.unbind(dim=-1)
    return torch.stack((
        torch.logaddexp(d * LN2, transfer * LN2) / LN2 - _softplus2(loss),
        transfer - d,
        loss,
    ), dim=-1)


def jacobian(phi: torch.Tensor) -> torch.Tensor:
    _u, v, w = phi.unbind(dim=-1)
    transfer_share = torch.sigmoid(v * LN2)
    loss_share = torch.sigmoid(w * LN2)
    one, zero = torch.ones_like(v), torch.zeros_like(v)
    return torch.stack((
        torch.stack((one, -transfer_share, loss_share), dim=-1),
        torch.stack((zero, zero, one), dim=-1),
        torch.stack((one, 1.0 - transfer_share, loss_share), dim=-1),
    ), dim=-2)


def transform_gradient(phi: torch.Tensor, gradient_theta: torch.Tensor) -> torch.Tensor:
    return torch.einsum("...ki,...k->...i", jacobian(phi), gradient_theta)


def complete_information_native(theta: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    rates = torch.exp2(theta)
    probability = rates / (1.0 + rates.sum(dim=-1, keepdim=True))
    covariance = (
        torch.diag_embed(probability)
        - probability.unsqueeze(-1) * probability.unsqueeze(-2)
    )
    return LN2 * counts.sum(dim=-1)[..., None, None] * covariance


def complete_information_hierarchical(phi: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    ns, nd, nl, nt = counts.unbind(dim=-1)
    u, v, w = phi.unbind(dim=-1)
    branch = torch.sigmoid(u * LN2)
    transfer_share = torch.sigmoid(v * LN2)
    loss_share = torch.sigmoid(w * LN2)
    diagonal = LN2 * torch.stack((
        (ns + nd + nl + nt) * branch * (1.0 - branch),
        (nd + nt) * transfer_share * (1.0 - transfer_share),
        (ns + nl) * loss_share * (1.0 - loss_share),
    ), dim=-1)
    return torch.diag_embed(diagonal)


def _calibrate_and_update(
    information: torch.Tensor,
    step: torch.Tensor,
    gradient_change: torch.Tensor,
    free: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    information_step = torch.einsum("fij,fj->fi", information, step)
    sy = (step * gradient_change).sum(dim=1)
    sbs = (step * information_step).sum(dim=1)
    valid = (sy > 0) & (sbs > 0) & torch.isfinite(sy) & torch.isfinite(sbs)
    scale = torch.where(
        valid, sy / torch.where(valid, sbs, torch.ones_like(sbs)), torch.ones_like(sy),
    )
    scaled = information * scale[:, None, None]
    seed = _bfgs_update(scaled, step, gradient_change, free)
    residual = torch.einsum("fij,fj->fi", seed, step) - gradient_change
    relative_residual = residual.norm(dim=1) / gradient_change.norm(dim=1).clamp_min(1e-300)
    return seed, {
        "step": step,
        "gradient_change": gradient_change,
        "information": information,
        "scale": scale,
        "scale_valid": valid,
        "free_at_both": free,
        "secant_relative_residual": relative_residual,
    }


class FreshReseeder:
    """Callable captured by a fail-closed source adapter at the first Newton gradient."""

    def __init__(
        self,
        *,
        coordinate: str,
        theta1: torch.Tensor,
        gradient1: torch.Tensor,
        counts1: torch.Tensor,
        lo: float,
        hi: float,
        bound_eps: float = 1e-6,
    ) -> None:
        if coordinate not in ("native", "hierarchical"):
            raise ValueError(coordinate)
        self.coordinate = coordinate
        self.theta1 = theta1.detach().double().cpu()
        self.gradient1 = gradient1.detach().double().cpu()
        self.counts1 = counts1.detach().double().cpu()
        self.lo = float(lo)
        self.hi = float(hi)
        self.bound_eps = float(bound_eps)
        self.calls = 0
        self.diagnostic: dict[str, torch.Tensor] | None = None

    def __call__(self, theta2: torch.Tensor, gradient2: torch.Tensor) -> torch.Tensor:
        if self.calls:
            raise RuntimeError("fresh reseed must be constructed exactly once")
        self.calls += 1
        device, dtype = theta2.device, theta2.dtype
        theta2_cpu = theta2.detach().double().cpu()
        gradient2_cpu = gradient2.detach().double().cpu()
        if theta2_cpu.shape != self.theta1.shape:
            raise ValueError("first Newton batch must contain every shared family in artifact order")

        if self.coordinate == "native":
            information = complete_information_native(theta2_cpu, self.counts1)
            step = theta2_cpu - self.theta1
            gradient_change = gradient2_cpu - self.gradient1
            free = (
                (self.theta1 > self.lo + self.bound_eps)
                & (self.theta1 < self.hi - self.bound_eps)
                & (theta2_cpu > self.lo + self.bound_eps)
                & (theta2_cpu < self.hi - self.bound_eps)
            ).to(torch.float64)
        else:
            phi1 = theta_to_phi(self.theta1)
            phi2 = theta_to_phi(theta2_cpu)
            information = complete_information_hierarchical(phi2, self.counts1)
            step = phi2 - phi1
            gradient_change = (
                transform_gradient(phi2, gradient2_cpu)
                - transform_gradient(phi1, self.gradient1)
            )
            free = torch.ones_like(step)

        seed, detail = _calibrate_and_update(information, step, gradient_change, free)
        self.diagnostic = {
            "coordinate": self.coordinate,
            "theta2": theta2_cpu,
            "gradient2": gradient2_cpu,
            "seed": seed,
            **detail,
        }
        return seed.to(device=device, dtype=dtype)

