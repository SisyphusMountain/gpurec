"""
Autograd integration for fixed-point reconciliation with implicit gradients.

Provides a stateful problem wrapper (FixedPointProblem) to manage dataset
structures and warm-started fixed points, and a custom autograd.Function
(FixedPointNLL) that computes the negative log-likelihood in forward and the
implicit gradient w.r.t. theta in backward.
"""

from __future__ import annotations

from typing import Optional, Dict

import torch
from torch.autograd import Function

from src.core.ccp import (
    build_ccp_from_single_tree,
    build_ccp_helpers,
    build_clade_species_mapping,
    get_root_clade_id,
)
from src.core.tree_helpers import build_species_helpers
from src.core.likelihood import (
    E_fixed_point,
    Pi_fixed_point,
    compute_log_likelihood,
)
from src.optimization.implicit_grad import compute_implicit_grad_loglik


class FixedPointProblem:
    """
    Holds dataset structures and current fixed points, and updates them with
    warm starts when theta changes.
    """

    def __init__(
        self,
        species_tree_path: str,
        gene_tree_path: str,
        *,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float64,
        use_anderson_E: bool = False,
        use_anderson_Pi: bool = False,
        anderson_m: int = 5,
        anderson_beta: float = 1.0,
        anderson_lam: float = 1e-4,
    ) -> None:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.dtype = dtype

        # Build persistent dataset structures
        self.ccp = build_ccp_from_single_tree(gene_tree_path)
        self.species_helpers = build_species_helpers(species_tree_path, device, dtype)
        self.clade_species_map = build_clade_species_mapping(self.ccp, self.species_helpers, device, dtype)
        self.clade_species_map = torch.log(self.clade_species_map + 1e-45)
        self.clade_species_map[self.clade_species_map == float('-inf')] = float('-inf')
        self.ccp_helpers = build_ccp_helpers(self.ccp, device, dtype)
        self.root_clade_id = get_root_clade_id(self.ccp)

        # Anderson settings
        self.use_anderson_E = use_anderson_E
        self.use_anderson_Pi = use_anderson_Pi
        self.anderson_m = anderson_m
        self.anderson_beta = anderson_beta
        self.anderson_lam = anderson_lam

        # Current state (initialized in init_from_theta)
        self.theta: Optional[torch.Tensor] = None
        self.E: Optional[torch.Tensor] = None
        self.Pi: Optional[torch.Tensor] = None

    def init_from_theta(
        self,
        theta: torch.Tensor,
        *,
        e_max_iters: int = 200,
        pi_max_iters: int = 200,
        e_tol: float = 1e-12,
        pi_tol: float = 1e-12,
    ) -> None:
        theta = theta.to(device=self.device, dtype=self.dtype)
        self.theta = theta

        # Converge E* from cold start, then Pi* from cold start
        E_out = E_fixed_point(
            species_helpers=self.species_helpers,
            theta=theta,
            max_iters=e_max_iters,
            tolerance=e_tol,
            return_components=True,
            warm_start_E=None,
        )
        E = E_out['E']
        E_s1 = E_out['E_s1']
        E_s2 = E_out['E_s2']
        E_bar = E_out['E_bar']

        Pi_out = Pi_fixed_point(
            ccp_helpers=self.ccp_helpers,
            species_helpers=self.species_helpers,
            clade_species_map=self.clade_species_map,
            E=E,
            Ebar=E_bar,
            E_s1=E_s1,
            E_s2=E_s2,
            theta=theta,
            max_iters=pi_max_iters,
            tolerance=pi_tol,
            warm_start_Pi=None,
        )
        self.E = E
        self.Pi = Pi_out['Pi']

    def update_theta(
        self,
        theta: torch.Tensor,
        *,
        e_max_iters: int = 200,
        pi_max_iters: int = 200,
        e_tol: float = 1e-12,
        pi_tol: float = 1e-12,
    ) -> None:
        """Warm-started convergence of E and Pi for the new theta."""
        theta = theta.to(device=self.device, dtype=self.dtype)
        if self.E is None or self.Pi is None:
            # If not initialized yet, fall back to cold start
            return self.init_from_theta(theta, e_max_iters=e_max_iters, pi_max_iters=pi_max_iters, e_tol=e_tol, pi_tol=pi_tol)

        self.theta = theta

        # Warm-start E
        E_out = E_fixed_point(
            species_helpers=self.species_helpers,
            theta=theta,
            max_iters=e_max_iters,
            tolerance=e_tol,
            return_components=True,
            warm_start_E=self.E,
        )
        self.E = E_out['E']
        E_s1 = E_out['E_s1']
        E_s2 = E_out['E_s2']
        E_bar = E_out['E_bar']

        # Warm-start Pi
        Pi_out = Pi_fixed_point(
            ccp_helpers=self.ccp_helpers,
            species_helpers=self.species_helpers,
            clade_species_map=self.clade_species_map,
            E=self.E,
            Ebar=E_bar,
            E_s1=E_s1,
            E_s2=E_s2,
            theta=theta,
            max_iters=pi_max_iters,
            tolerance=pi_tol,
            warm_start_Pi=self.Pi,
        )
        self.Pi = Pi_out['Pi']

    def nlll(self) -> torch.Tensor:
        assert self.Pi is not None
        ll = compute_log_likelihood(self.Pi, self.root_clade_id)
        return -ll


class FixedPointNLL(Function):
    """Custom autograd op that returns NLL and uses implicit grads in backward."""

    @staticmethod
    def forward(ctx, theta: torch.Tensor, problem: FixedPointProblem,
                e_max_iters: int = 200, pi_max_iters: int = 200,
                e_tol: float = 1e-12, pi_tol: float = 1e-12) -> torch.Tensor:
        # Update problem’s fixed points (warm start) at the given theta
        problem.update_theta(theta.detach(), e_max_iters=e_max_iters, pi_max_iters=pi_max_iters, e_tol=e_tol, pi_tol=pi_tol)

        # Compute negative log-likelihood
        nlll = problem.nlll()

        # Save tensors and helpers for backward
        ctx.save_for_backward(theta.detach(), problem.Pi.detach(), problem.E.detach())
        ctx.ccp_helpers = problem.ccp_helpers
        ctx.species_helpers = problem.species_helpers
        ctx.clade_species_map = problem.clade_species_map
        ctx.root_clade_id = problem.root_clade_id

        return nlll

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (theta, Pi_star, E_star) = ctx.saved_tensors

        # Compute d/dtheta log-likelihood using implicit gradient
        grad_loglik = compute_implicit_grad_loglik(
            Pi_star=Pi_star,
            E_star=E_star,
            theta=theta,
            ccp_helpers=ctx.ccp_helpers,
            species_helpers=ctx.species_helpers,
            clade_species_map=ctx.clade_species_map,
            root_clade_id=ctx.root_clade_id,
        )
        # We need gradient of NLL, so negate
        grad_theta = -grad_loglik

        # Chain with upstream grad_output
        grad_theta = grad_output * grad_theta

        # Return grads for (theta, problem)
        return grad_theta, None, None, None, None, None


def fixed_point_nll(theta: torch.Tensor, problem: FixedPointProblem,
                    *, e_max_iters: int = 200, pi_max_iters: int = 200,
                    e_tol: float = 1e-12, pi_tol: float = 1e-12) -> torch.Tensor:
    """Convenience wrapper to call the custom autograd op."""
    return FixedPointNLL.apply(theta, problem, e_max_iters, pi_max_iters, e_tol, pi_tol)
