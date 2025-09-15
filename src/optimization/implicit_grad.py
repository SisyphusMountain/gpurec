"""
Implicit gradient of the log-likelihood with respect to theta at fixed points.

This module provides minimal, clean utilities to compute the gradient of the
log-likelihood L(Pi*) with respect to the unconstrained log-parameters
theta = [log_delta, log_tau, log_lambda], given converged fixed points
E* (extinction probabilities) and Pi* (clade-species likelihoods), along with
the helper structures to evaluate Pi_step and E_step.

No fixed-point iteration or dataset orchestration is performed here.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import torch

# Local imports for building closures and likelihood
from src.reconciliation.likelihood import (
    E_step, Pi_step, compute_log_likelihood, get_log_params,
    gather_Pi_children, dup_both_survive, NEG_INF
)
from src.reconciliation.reconcile import setup_fixed_points


@torch.no_grad()
def solve_adjoint_fixedpoint(
    vjp_apply: Callable[[torch.Tensor], torch.Tensor],
    rhs: torch.Tensor,
    *,
    max_iter: int = 200,
    tol: float = 1e-10,
    damping: float = 1.0,
) -> torch.Tensor:
    """
    Solve (I - J^T) x = rhs using a matrix-free fixed-point iteration:

        x_{k+1} = rhs + damping * (J^T x_k)

    where vjp_apply(x) returns (J^T x). For contraction mappings, spectral
    radius(J^T) < 1, so the iteration converges (with damping=1 matching the
    exact system). A smaller damping may help numerics but solves a modified
    system (I - damping*J^T) x = rhs; keep damping=1 unless convergence issues.
    """
    x = torch.zeros_like(rhs)
    for _ in range(max_iter):
        x_next = rhs + damping * vjp_apply(x)
        if torch.norm(x_next - x) <= tol * (1 + torch.norm(x)):
            return x_next
        x = x_next
    return x


def compute_implicit_grad_loglik(
    *,
    Pi_star: torch.Tensor,
    E_star: torch.Tensor,
    theta: torch.Tensor,
    ccp_helpers: Dict,
    species_helpers: Dict,
    clade_species_map: torch.Tensor,
    root_clade_id: int,
    max_iter: int = 300,
    tol: float = 1e-10,
    damping: float = 1.0,
) -> torch.Tensor:
    """
    Compute the gradient of log-likelihood L(Pi*) wrt theta at fixed points.

    Inputs are converged Pi_star, E_star for the given theta and dataset helpers.
    Returns a 3-vector: [dL/dlog_delta, dL/dlog_tau, dL/dlog_lambda].
    """
    from torch import func as tfunc

    # Build closures for G (E update) and F (Pi update)
    sp_c1_idx = species_helpers['s_C1_indexes']
    sp_c2_idx = species_helpers['s_C2_indexes']
    sp_internal_mask = species_helpers['sp_internal_mask']
    Recipients_mat = species_helpers['Recipients_mat']

    def Gfun(E: torch.Tensor, th: torch.Tensor) -> torch.Tensor:
        return E_step(E, sp_c1_idx, sp_c2_idx, sp_internal_mask, Recipients_mat, th)

    def Ffun(Pi: torch.Tensor, E: torch.Tensor, th: torch.Tensor) -> torch.Tensor:
        # Recompute components from E to capture ∂F/∂E via autograd
        _, E_s1, E_s2, Ebar = E_step(E, sp_c1_idx, sp_c2_idx, sp_internal_mask, Recipients_mat, th, return_components=True)
        return Pi_step(Pi, ccp_helpers, species_helpers, clade_species_map, E, Ebar, E_s1, E_s2, th)

    # Readout L(Pi) = logsumexp(Pi[root,:])
    def Lfun(Pi: torch.Tensor) -> torch.Tensor:
        return compute_log_likelihood(Pi, root_clade_id)

    # Build autograd graph for VJPs inside an enabled grad context
    with torch.enable_grad():
        # Seed adjoint: phi = ∂L/∂Pi
        Pi_req = Pi_star.detach().requires_grad_(True)
        loss = Lfun(Pi_req)
        (phi,) = torch.autograd.grad(loss, Pi_req)
        del Pi_req, loss

        # Persistent VJP closures at fixed points
        _, vjpF = tfunc.vjp(Ffun, Pi_star, E_star, theta)
        _, vjpG = tfunc.vjp(Gfun, E_star, theta)

    # Solve (I - F_Pi^T) v = phi
    v = solve_adjoint_fixedpoint(lambda x: vjpF(x)[0], phi, max_iter=max_iter, tol=tol, damping=damping)

    # q = (∂F/∂E)^T v and parameter part from F
    _gPi_v, q, g_theta_F = vjpF(v)

    # Solve (I - G_E^T) w = q
    w = solve_adjoint_fixedpoint(lambda x: vjpG(x)[0], q, max_iter=max_iter, tol=tol, damping=damping)

    # Parameter part from G
    _gE_w, g_theta_G = vjpG(w)

    grad_theta = g_theta_F + g_theta_G  # 3-vector
    return grad_theta


def optimize_theta_with_warm_starts(
    species_tree_path: str,
    gene_tree_path: str,
    *,
    theta_init: torch.Tensor,
    step_size: float = 0.2,
    max_outer_iters: int = 200,
    tol_theta: float = 1e-9,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float64,
    e_max_iters: int = 200,
    pi_max_iters: int = 200,
    e_tol: float = 1e-12,
    pi_tol: float = 1e-12,
    verbose: bool = True,
) -> Dict:
    """
    Optimize theta using implicit gradients and warm-started fixed points.

    Procedure (per request):
      1) Converge E*, then Pi* from initial theta.
      2) Compute gradient of nlll wrt theta via implicit VJP (gradient of -L).
      3) Take an optimizer step updating theta.
      4) Warm-start E_fixed_point and Pi_fixed_point from previous E*, Pi* to
         converge at the new theta.
      5) If ||theta_new - theta_old||_inf < tol_theta, stop; else repeat (2).

    Returns a dict containing final theta, E, Pi, log-likelihood history, and traces.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    theta = theta_init.to(device=device, dtype=dtype).clone()

    # Initial fixed points and helpers via reconcile.setup_fixed_points
    fix = setup_fixed_points(
        species_tree_path,
        gene_tree_path,
        max_iters_E=e_max_iters,
        max_iters_Pi=pi_max_iters,
        tol_E=e_tol,
        tol_Pi=pi_tol,
        device=device,
        dtype=dtype,
        debug=False,
        use_theta=True,
        theta=theta,
    )
    E = fix['E']
    Pi = fix['Pi']
    species_helpers = fix['species_helpers']
    ccp_helpers = fix['ccp_helpers']
    clade_species_map = fix['clade_species_map']

    from src.core.ccp import get_root_clade_id
    root_clade_id = get_root_clade_id(fix['ccp'])

    # Build closures once (helpers are constant across iterations)
    Ffun, Gfun = _build_F_G_wrappers(
        ccp_helpers=ccp_helpers, species_helpers=species_helpers, clade_species_map=clade_species_map
    )
    Lfun = lambda Pi_: compute_log_likelihood(Pi_, root_clade_id)

    L_history = []
    theta_history = [theta.clone()]
    rate_history = [torch.exp(theta).clone()]
    E_history = []
    Pi_history = []

    for it in range(1, max_outer_iters + 1):
        # Current likelihood
        L_cur = float(Lfun(Pi))
        L_history.append(L_cur)
        E_history.append(E.detach().clone())
        Pi_history.append(Pi.detach().clone())

        # 2) implicit grad of negative log-likelihood: grad_nlll = - dL/dtheta
        g_d, g_t, g_l, _v, _w = implicit_grad_L_vjp(
            Ffun, Gfun, Lfun, Pi_star=Pi, E_star=E, theta=theta
        )
        grad_logL = torch.stack([g_d, g_t, g_l])
        grad_nlll = -grad_logL

        # 3) update theta (plain gradient descent on nlll)
        theta_new = theta - step_size * grad_nlll  # equals theta + step_size * grad_logL

        # 4) Warm-started fixed points for new theta
        E_out = E_fixed_point(
            species_helpers=species_helpers,
            theta=theta_new,
            max_iters=e_max_iters,
            tolerance=e_tol,
            return_components=True,
            warm_start_E=E,
        )
        E = E_out['E']
        E_s1 = E_out['E_s1']
        E_s2 = E_out['E_s2']
        E_bar = E_out['E_bar']

        Pi_out = Pi_fixed_point(
            ccp_helpers=ccp_helpers,
            species_helpers=species_helpers,
            clade_species_map=clade_species_map,
            E=E,
            Ebar=E_bar,
            E_s1=E_s1,
            E_s2=E_s2,
            theta=theta_new,
            max_iters=pi_max_iters,
            tolerance=pi_tol,
            warm_start_Pi=Pi,
        )
        Pi = Pi_out['Pi']

        # 5) Convergence check on theta (inf-norm)
        diff = torch.max(torch.abs(theta_new - theta)).item()
        theta = theta_new
        theta_history.append(theta.clone())
        rate_history.append(torch.exp(theta).clone())
        if verbose:
            print(
                f"[outer {it:03d}] L={L_cur:.6f} | ||Δθ||_∞={diff:.3e} | θ={theta.tolist()} | rates={torch.exp(theta).tolist()}"
            )
        if diff < tol_theta:
            break

    # Final likelihood at converged Pi
    L_final = float(Lfun(Pi))

    # After convergence: dump inputs to ScatterLogSumExp and Triton-segmented versions
    try:
        import os
        os.makedirs('results', exist_ok=True)

        # Recompute components used in Pi_step to build log_combined_splits
        exp_theta = torch.exp(theta)
        log_pS, log_pD, log_pT, log_pL = get_log_params(torch.log(exp_theta))
        split_parents = ccp_helpers['split_parents']
        split_lefts = ccp_helpers['split_lefts']
        split_rights = ccp_helpers['split_rights']

        # Gather children
        Pi_s1_ws = torch.full_like(Pi, NEG_INF)
        Pi_s2_ws = torch.full_like(Pi, NEG_INF)
        Pi_s1_ws = gather_Pi_children(Pi, species_helpers['s_P_indexes'], species_helpers['s_C1_indexes'], Pi_s1_ws)
        Pi_s2_ws = gather_Pi_children(Pi, species_helpers['s_P_indexes'], species_helpers['s_C2_indexes'], Pi_s2_ws)

        # Duplication/speciation/transfer split-space terms
        Pi_left = torch.index_select(Pi, 0, split_lefts)
        Pi_right = torch.index_select(Pi, 0, split_rights)
        log_split_probs = ccp_helpers['log_split_probs']
        log_D_splits = dup_both_survive(Pi_left, Pi_right, log_split_probs, log_pD)
        Pi_s1_left = torch.index_select(Pi_s1_ws, 0, split_lefts)
        Pi_s2_right = torch.index_select(Pi_s2_ws, 0, split_rights)
        Pi_s1_right = torch.index_select(Pi_s1_ws, 0, split_rights)
        Pi_s2_left = torch.index_select(Pi_s2_ws, 0, split_lefts)
        log_spec1 = log_split_probs.unsqueeze(1) + log_pS + Pi_s1_left + Pi_s2_right
        log_spec2 = log_split_probs.unsqueeze(1) + log_pS + Pi_s1_right + Pi_s2_left
        # Transfer terms as in Pi_step (reuse current Pi and Ebar/E here)
        Pi_max = torch.max(Pi, dim=1, keepdim=True).values
        Pi_linear = torch.exp(Pi - Pi_max)
        Pibar_linear = Pi_linear.mm(species_helpers['Recipients_mat'].T)
        Pibar = torch.log(Pibar_linear) + Pi_max
        Pibar_left = torch.index_select(Pibar, 0, split_lefts)
        Pibar_right = torch.index_select(Pibar, 0, split_rights)
        log_trans1 = log_split_probs.unsqueeze(1) + log_pT + Pi_left + Pibar_right
        log_trans2 = log_split_probs.unsqueeze(1) + log_pT + Pi_right + Pibar_left

        # Combine split-space terms
        no_L_contribs = torch.stack([log_D_splits, log_spec1, log_spec2, log_trans1, log_trans2], dim=0)
        log_combined_splits = torch.logsumexp(no_L_contribs, dim=0)  # [N_splits, S]

        # Scatter path inputs
        leaves_mask = torch.isfinite(clade_species_map).any(dim=1)  # [C]
        C = ccp_helpers['C']

        # Triton segmented path inputs (stable ascending parent order)
        order = torch.argsort(split_parents, stable=True)
        counts = torch.bincount(split_parents, minlength=C)
        ptr = torch.empty(C + 1, dtype=torch.long, device=Pi.device)
        ptr[0] = 0
        ptr[1:] = torch.cumsum(counts, dim=0)
        x_sorted = log_combined_splits.index_select(0, order)

        dump = {
            'log_combined_splits': log_combined_splits.detach().cpu(),
            'split_parents': split_parents.detach().cpu(),
            'C': C,
            'leaves_mask': leaves_mask.detach().cpu(),
            'order': order.detach().cpu(),
            'ptr': ptr.detach().cpu(),
            'x_sorted': x_sorted.detach().cpu(),
            'dtype': str(Pi.dtype),
            'device': str(Pi.device),
            'shapes': {
                'log_combined_splits': tuple(log_combined_splits.shape),
                'x_sorted': tuple(x_sorted.shape),
            }
        }
        torch.save(dump, 'results/scatter_inputs.pt')
    except Exception as _e:
        # Non-fatal; proceed without failing optimization
        pass

    return {
        'theta': theta,
        'rates': torch.exp(theta),
        'log_likelihood': L_final,
        'E': E,
        'Pi': Pi,
        'theta_history': theta_history,
        'rate_history': rate_history,
        'L_history': L_history,
        'species_helpers': species_helpers,
        'ccp_helpers': ccp_helpers,
        'clade_species_map': clade_species_map,
        'root_clade_id': root_clade_id,
    }
