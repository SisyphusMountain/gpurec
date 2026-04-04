# -*- coding: utf-8 -*-
"""
Implicit-gradient optimisation of reconciliation parameters (θ).

Key changes vs. your previous version:
- Fixed points (E*, Π*) are computed under torch.no_grad() -> small memory.
- Gradients use implicit differentiation with ONLY VJP calls (backward):
    (I - F_Π^T) v = φ,  φ = ∂ logL / ∂Π
    (I - G_E^T) w = q,  q = (∂F/∂E)^T v
    ∇θ logL = (∂F/∂θ)^T v + (∂G/∂θ)^T w
- Linear solves default to CG; detect non-SPD and fall back to GMRES.

You can increase Adam's LR safely to take bigger θ-steps without
backprop-through-FP memory blowups.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Callable

import math
import torch
from torch import Tensor
from torch import func as tfunc

from src.core.likelihood import (
    E_fixed_point,
    Pi_wave_forward,
    Pi_wave_backward,
    compute_log_likelihood,
    E_step,
    _safe_log2,
)
from src.core.batching import build_wave_layout
from src.core.scheduling import compute_clade_waves
from src.core.extract_parameters import extract_parameters, extract_parameters_uniform

# -------------------------------------------------------------------------
# Dataclasses for logging
# -------------------------------------------------------------------------

@dataclass
class FixedPointInfo:
    iterations_E: int
    iterations_Pi: int

@dataclass
class LinearSolveStats:
    method: str
    iters: int
    rel_residual: float
    fallback_used: bool

@dataclass
class StepRecord:
    iteration: int
    theta: torch.Tensor
    rates: torch.Tensor
    negative_log_likelihood: float
    log_likelihood: float
    theta_step_inf: float
    grad_infinity_norm: float
    fp_info: FixedPointInfo
    gradient: torch.Tensor
    solve_stats_F: LinearSolveStats
    solve_stats_G: LinearSolveStats


# -------------------------------------------------------------------------
# Utilities: matrix-free Krylov solvers (CG + GMRES fallback)
# -------------------------------------------------------------------------

@torch.no_grad()
def _cg(
    Av: Callable[[Tensor], Tensor],
    b: Tensor,
    *,
    M: Optional[Callable[[Tensor], Tensor]] = None,
    tol: float = 1e-8,
    maxiter: int = 500,
    x0: Optional[Tensor] = None,
) -> Tuple[Tensor, LinearSolveStats, bool]:
    """
    Conjugate Gradients for SPD A. Uses only Av (no A^T).
    Returns (x, stats, success). If breakdown detected, success=False.
    """
    device, dtype = b.device, b.dtype
    x = torch.zeros_like(b) if x0 is None else x0.clone()
    r = b - Av(x)
    z = r if M is None else M(r)
    p = z.clone()
    rz_old = float(torch.dot(r, z))
    bnorm = float(b.norm()) if b.numel() > 0 else 1.0
    bnorm = max(bnorm, 1.0)

    success = True
    iters = 0
    for k in range(1, maxiter + 1):
        Ap = Av(p)
        pAp = float(torch.dot(p, Ap))
        if pAp <= 0.0 or not math.isfinite(pAp):  # non-SPD or numerical breakdown
            success = False
            iters = k - 1
            break
        alpha = rz_old / pAp
        x = x + alpha * p
        r = r - alpha * Ap
        rel_res = float(r.norm()) / bnorm
        if rel_res <= tol:
            iters = k
            return x, LinearSolveStats("CG", iters, rel_res, fallback_used=False), True
        z = r if M is None else M(r)
        rz_new = float(torch.dot(r, z))
        beta = rz_new / max(rz_old, 1e-30)
        p = z + beta * p
        rz_old = rz_new
        iters = k

    rel_res = float(r.norm()) / bnorm
    return x, LinearSolveStats("CG", iters, rel_res, fallback_used=False), success


@torch.no_grad()
def _gmres(
    Av: Callable[[Tensor], Tensor],
    b: Tensor,
    *,
    M: Optional[Callable[[Tensor], Tensor]] = None,  # right preconditioner
    tol: float = 1e-8,
    restart: int = 40,
    maxiter: int = 500,
    x0: Optional[Tensor] = None,
) -> Tuple[Tensor, LinearSolveStats]:
    """
    Restarted GMRES with right preconditioning. Uses only Av.
    """
    device, dtype = b.device, b.dtype
    x = torch.zeros_like(b) if x0 is None else x0.clone()
    Id = (lambda v: v)
    M = Id if M is None else M

    iters = 0
    bnorm = float(b.norm())
    bnorm = max(bnorm, 1.0)

    while iters < maxiter:
        r = b - Av(x)
        beta = float(r.norm())
        if beta / bnorm <= tol:
            return x, LinearSolveStats("GMRES", iters, beta / bnorm, fallback_used=False)

        m = min(restart, maxiter - iters)
        V: List[Tensor] = []
        Z: List[Tensor] = []
        H = torch.zeros((m + 1, m), device=device, dtype=dtype)

        v1 = r / (beta + 1e-30)
        V.append(v1)
        happy = False
        used = 0

        for j in range(m):
            z_j = M(V[j]); Z.append(z_j)
            w = Av(z_j)
            # Modified Gram-Schmidt
            for i in range(j + 1):
                H[i, j] = torch.dot(V[i], w)
                w = w - H[i, j] * V[i]
            H[j + 1, j] = w.norm()
            if float(H[j + 1, j]) < 1e-14:
                happy = True
                used = j + 1
                break
            V.append(w / (H[j + 1, j] + 1e-30))
            used = j + 1

        e1 = torch.zeros(used + 1, device=device, dtype=dtype); e1[0] = 1.0
        y = torch.linalg.lstsq(H[:used + 1, :used], beta * e1[:used + 1]).solution
        x = x + sum(y[i] * Z[i] for i in range(used))
        iters += used
        if happy:
            r = b - Av(x)
            rel = float(r.norm()) / bnorm
            return x, LinearSolveStats("GMRES", iters, rel, fallback_used=False)

    r = b - Av(x)
    rel = float(r.norm()) / bnorm
    return x, LinearSolveStats("GMRES", iters, rel, fallback_used=False)



# -------------------------------------------------------------------------
# Implicit gradient: build VJP closures & solve the two transpose systems
# -------------------------------------------------------------------------

@torch.no_grad()
def implicit_grad_loglik_vjp_wave(
    wave_layout,
    species_helpers,
    *,
    Pi_star_wave: torch.Tensor,
    Pibar_star_wave: torch.Tensor,
    E_star: torch.Tensor,
    E_s1: torch.Tensor,
    E_s2: torch.Tensor,
    Ebar: torch.Tensor,
    log_pS: torch.Tensor,
    log_pD: torch.Tensor,
    log_pL: torch.Tensor,
    max_transfer_mat: torch.Tensor,
    root_clade_ids_perm: torch.Tensor,
    theta: torch.Tensor,
    unnorm_row_max: torch.Tensor,
    specieswise: bool,
    device: torch.device,
    dtype: torch.dtype,
    neumann_terms: int = 3,
    use_pruning: bool = True,
    pruning_threshold: float = 1e-6,
    cg_tol: float = 1e-8,
    cg_maxiter: int = 500,
    gmres_restart: int = 40,
    pibar_mode: str = 'uniform_approx',
    transfer_mat: Optional[torch.Tensor] = None,
    transfer_mat_unnormalized: Optional[torch.Tensor] = None,
    ancestors_T: Optional[torch.Tensor] = None,
):
    """Compute ∇θ logL using wave-decomposed backward pass + E adjoint.

    Steps:
    1. Pi backward: wave-by-wave Neumann series (root→leaves)
    2. E adjoint: solve (I - G_E^T) w = q via CG/GMRES
    3. θ gradient: VJP through extract_parameters

    Returns (grad_theta, pi_backward_info).
    """
    # --- Step 1: Pi backward (can be pre-computed for batched mode) ---
    pi_bwd = Pi_wave_backward(
        wave_layout=wave_layout,
        Pi_star_wave=Pi_star_wave,
        Pibar_star_wave=Pibar_star_wave,
        E=E_star, Ebar=Ebar, E_s1=E_s1, E_s2=E_s2,
        log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
        max_transfer_mat=max_transfer_mat,
        species_helpers=species_helpers,
        root_clade_ids_perm=root_clade_ids_perm,
        device=device, dtype=dtype,
        neumann_terms=neumann_terms,
        use_pruning=use_pruning,
        pruning_threshold=pruning_threshold,
        pibar_mode=pibar_mode,
        transfer_mat=transfer_mat,
        ancestors_T=ancestors_T,
    )

    return _e_adjoint_and_theta_vjp(
        pi_bwd, E_star, Ebar, E_s1, E_s2,
        log_pS, log_pD, log_pL,
        max_transfer_mat, species_helpers, root_clade_ids_perm,
        theta, unnorm_row_max, specieswise,
        device, dtype,
        cg_tol=cg_tol, cg_maxiter=cg_maxiter, gmres_restart=gmres_restart,
        pibar_mode=pibar_mode, transfer_mat=transfer_mat,
        transfer_mat_unnormalized=transfer_mat_unnormalized,
        ancestors_T=ancestors_T,
    )


def _e_adjoint_and_theta_vjp(
    pi_bwd,
    E_star, Ebar, E_s1, E_s2,
    log_pS, log_pD, log_pL,
    max_transfer_mat, species_helpers, root_clade_ids_perm,
    theta, unnorm_row_max, specieswise,
    device, dtype,
    *,
    cg_tol=1e-8, cg_maxiter=500, gmres_restart=40,
    pibar_mode='uniform_approx',
    transfer_mat=None, transfer_mat_unnormalized=None, ancestors_T=None,
):
    """E adjoint solve + theta VJP from pre-computed Pi backward result.

    Takes pi_bwd dict (from Pi_wave_backward) and completes the gradient
    computation through E adjoint solve and extract_parameters VJP.
    """
    # --- Step 2: E adjoint ---
    sp_P_idx = species_helpers['s_P_indexes']
    sp_c12_idx = species_helpers['s_C12_indexes']

    # Direct dNLL/dE from likelihood denominator
    n_fam = root_clade_ids_perm.numel()
    E_req_d = E_star.detach().requires_grad_(True)
    with torch.enable_grad():
        mean_E_exp = torch.exp2(E_req_d).mean(dim=-1)
        denom = torch.log2(1.0 - mean_E_exp)
        direct_dNLL_dE = torch.autograd.grad(n_fam * denom, E_req_d)[0]
    q_E = pi_bwd['grad_E'].clone() + direct_dNLL_dE

    # Chain Ebar gradient through E_step's Ebar computation
    if pi_bwd['grad_Ebar'].abs().max() > 0:
        E_req2 = E_star.detach().requires_grad_(True)
        with torch.enable_grad():
            mt_sq = max_transfer_mat.squeeze(-1) if max_transfer_mat.ndim > 1 else max_transfer_mat
            if pibar_mode in ('dense', 'topk') and transfer_mat is not None:
                # Ebar[i] = log2(sum_j(transfer_mat[i,j] * exp2(E[j]))) + max_E + mt[i]
                max_E = E_req2.max(dim=-1, keepdim=True).values
                expE = torch.exp2(E_req2 - max_E)
                Ebar_recomp = _safe_log2(torch.einsum("ij,j->i", transfer_mat, expE)) + max_E.squeeze(-1) + mt_sq
            elif pibar_mode == 'uniform' and ancestors_T is not None:
                # Ebar[s] = log2(sum(exp2(E)) - sum_{j: s is ancestor of j}(exp2(E[j]))) + max_E + mt[s]
                max_E = E_req2.max(dim=-1, keepdim=True).values
                expE = torch.exp2(E_req2 - max_E)
                expE_2d = expE.unsqueeze(0)
                row_sum = expE_2d.sum(dim=-1, keepdim=True)
                ancestor_sum = expE_2d @ ancestors_T
                Ebar_recomp = _safe_log2((row_sum - ancestor_sum).squeeze(0)) + max_E.squeeze(-1) + mt_sq
            else:
                # uniform: Ebar[s] = logsumexp2(E) + mt[s]
                lse_val = torch.log2(torch.exp2(E_req2).sum(dim=-1, keepdim=True))
                Ebar_recomp = lse_val + mt_sq
            ebar_to_e = torch.autograd.grad(
                Ebar_recomp, E_req2,
                grad_outputs=pi_bwd['grad_Ebar'],
                retain_graph=False,
            )[0]
        q_E = q_E + ebar_to_e

    # Chain E_s1, E_s2 gradients through gather_E_children
    if pi_bwd['grad_E_s1'].abs().max() > 0 or pi_bwd['grad_E_s2'].abs().max() > 0:
        E_req3 = E_star.detach().requires_grad_(True)
        with torch.enable_grad():
            from src.core.terms import gather_E_children
            E_s12 = gather_E_children(E_req3, sp_P_idx, sp_c12_idx)
            E_s1_r, E_s2_r = torch.chunk(E_s12, 2, dim=-1)
            E_s1_r = E_s1_r.view(E_req3.shape)
            E_s2_r = E_s2_r.view(E_req3.shape)
            total = (E_s1_r * pi_bwd['grad_E_s1']).sum() + (E_s2_r * pi_bwd['grad_E_s2']).sum()
            es_to_e = torch.autograd.grad(total, E_req3, retain_graph=False)[0]
        q_E = q_E + es_to_e

    # Solve (I - G_E^T) w = q_E via CG/GMRES
    def G_E_fun(E_in):
        """E_step as a function of E only."""
        return E_step(
            E_in, sp_P_idx, sp_c12_idx,
            log_pS, log_pD, log_pL,
            transfer_mat, max_transfer_mat, pibar_mode=pibar_mode,
            ancestors_T=ancestors_T,
        )[0]

    # Build VJP for G_E
    E_req_g = E_star.detach().requires_grad_(True)
    with torch.enable_grad():
        _, vjpG = tfunc.vjp(G_E_fun, E_req_g)

    nE = E_star.numel()
    E_shape = E_star.shape
    q_flat = q_E.reshape(-1)

    def AG_flat(w_flat):
        wE = w_flat.view(E_shape).contiguous()
        gE, = vjpG(wE.clone())
        return (wE - gE).reshape(-1)

    w_flat, statsG, okG = _cg(AG_flat, q_flat, tol=cg_tol, maxiter=cg_maxiter)
    if not okG:
        w_flat, statsG = _gmres(AG_flat, q_flat, tol=cg_tol, restart=gmres_restart, maxiter=cg_maxiter)
        statsG.fallback_used = True

    wE = w_flat.view(E_shape)

    # --- Step 3: theta gradient through extract_parameters ---
    grad_mt_total = pi_bwd['grad_max_transfer_mat'] + pi_bwd['grad_Ebar']

    theta_req = theta.detach().requires_grad_(True)
    with torch.enable_grad():
        if pibar_mode in ('dense', 'topk') and transfer_mat_unnormalized is not None:
            log_pS_r, log_pD_r, log_pL_r, transfer_mat_r, mt_r_raw = extract_parameters(
                theta_req, transfer_mat_unnormalized,
                genewise=False, specieswise=specieswise, pairwise=False,
            )
            mt_r = mt_r_raw.squeeze(-1) if mt_r_raw.ndim == 2 else mt_r_raw
            param_loss = (
                (log_pS_r * pi_bwd['grad_log_pS']).sum() +
                (log_pD_r * pi_bwd['grad_log_pD']).sum() +
                (mt_r * grad_mt_total).sum()
            )
            # Add transfer_mat gradient contribution (dense/topk)
            grad_transfer_mat = pi_bwd.get('grad_transfer_mat')
            if grad_transfer_mat is not None:
                param_loss = param_loss + (transfer_mat_r * grad_transfer_mat).sum()
        else:
            # uniform and uniform_approx: no theta-dependent transfer_mat
            log_pS_r, log_pD_r, log_pL_r, _, mt_r = extract_parameters_uniform(
                theta_req, unnorm_row_max, specieswise=specieswise,
            )
            param_loss = (
                (log_pS_r * pi_bwd['grad_log_pS']).sum() +
                (log_pD_r * pi_bwd['grad_log_pD']).sum() +
                (mt_r * grad_mt_total).sum()
            )
        grad_theta_pi = torch.autograd.grad(param_loss, theta_req, retain_graph=False)[0]

    # E adjoint contribution to theta
    theta_req2 = theta.detach().requires_grad_(True)
    with torch.enable_grad():
        if pibar_mode in ('dense', 'topk') and transfer_mat_unnormalized is not None:
            log_pS_r2, log_pD_r2, log_pL_r2, transfer_mat_r2, mt_r2_raw = extract_parameters(
                theta_req2, transfer_mat_unnormalized,
                genewise=False, specieswise=specieswise, pairwise=False,
            )
            mt_r2 = mt_r2_raw.squeeze(-1) if mt_r2_raw.ndim == 2 else mt_r2_raw

            def G_E_theta(th_pS, th_pD, th_pL, th_tm, th_mt):
                return E_step(
                    E_star.detach(), sp_P_idx, sp_c12_idx,
                    th_pS, th_pD, th_pL, th_tm, th_mt, pibar_mode='dense',
                )[0]

            E_from_theta = G_E_theta(log_pS_r2, log_pD_r2, log_pL_r2, transfer_mat_r2, mt_r2)
        else:
            if pibar_mode == 'uniform' and ancestors_T is not None:
                log_pS_r2, log_pD_r2, log_pL_r2, _, mt_r2 = extract_parameters_uniform(
                    theta_req2, unnorm_row_max, specieswise=specieswise,
                )

                def G_E_theta(th_pS, th_pD, th_pL, th_mt):
                    return E_step(
                        E_star.detach(), sp_P_idx, sp_c12_idx,
                        th_pS, th_pD, th_pL, None, th_mt,
                        pibar_mode='uniform', ancestors_T=ancestors_T,
                    )[0]

                E_from_theta = G_E_theta(log_pS_r2, log_pD_r2, log_pL_r2, mt_r2)
            else:
                log_pS_r2, log_pD_r2, log_pL_r2, _, mt_r2 = extract_parameters_uniform(
                    theta_req2, unnorm_row_max, specieswise=specieswise,
                )

                def G_E_theta(th_pS, th_pD, th_pL, th_mt):
                    return E_step(
                        E_star.detach(), sp_P_idx, sp_c12_idx,
                        th_pS, th_pD, th_pL, None, th_mt, pibar_mode='uniform_approx',
                    )[0]

                E_from_theta = G_E_theta(log_pS_r2, log_pD_r2, log_pL_r2, mt_r2)

        gtheta_E = torch.autograd.grad(
            E_from_theta, theta_req2,
            grad_outputs=wE,
            retain_graph=False,
        )[0]

    grad_theta = (grad_theta_pi + gtheta_E).detach()
    return grad_theta, statsG


@torch.no_grad()
def implicit_grad_loglik_vjp_wave_genewise(
    families,
    species_helpers,
    *,
    E_all: Tensor,
    E_s1_all: Tensor,
    E_s2_all: Tensor,
    Ebar_all: Tensor,
    log_pS_all: Tensor,
    log_pD_all: Tensor,
    log_pL_all: Tensor,
    mt_all: Tensor,
    theta_stack: Tensor,
    unnorm_row_max: Tensor,
    specieswise: bool,
    device: torch.device,
    dtype: torch.dtype,
    pibar_mode: str = 'uniform_approx',
    neumann_terms: int = 3,
    use_pruning: bool = True,
    pruning_threshold: float = 1e-6,
    cg_tol: float = 1e-8,
    cg_maxiter: int = 500,
    gmres_restart: int = 40,
    local_iters: int = 2000,
    local_tolerance: float = 1e-3,
):
    """Genewise implicit gradient via per-family Pi forward + backward loop.

    Each family gets its own wave layout, Pi forward pass, and backward pass
    with per-gene scalar/[S] parameters. The E adjoint is solved per-family.

    Parameters
    ----------
    families : list of dict
        Per-family data, each with 'ccp_helpers', 'leaf_row_index',
        'leaf_col_index', 'root_clade_id'.
    species_helpers : dict
        Shared species tree helpers.
    E_all, E_s1_all, E_s2_all, Ebar_all : Tensor [G, S]
        Per-gene converged E quantities.
    log_pS_all, log_pD_all, log_pL_all : Tensor [G] or [G, S]
        Per-gene event probabilities.
    mt_all : Tensor [G, S]
        Per-gene max_transfer_mat.
    theta_stack : Tensor [G, 3] or [G, S, 3]
        Per-gene rate parameters.
    unnorm_row_max : Tensor [S]
        Row maxima of unnormalized transfer matrix.
    specieswise : bool
        Whether rates are per-species.
    local_iters, local_tolerance : int, float
        Pi forward convergence parameters.

    Returns
    -------
    (grad_theta_stack, stats_list) : (Tensor [G, ...], list[LinearSolveStats])
    """
    from src.core.likelihood import Pi_wave_forward
    from src.core.batching import collate_gene_families, build_wave_layout
    from src.core.scheduling import compute_clade_waves

    G = len(families)
    grad_thetas = []
    all_stats = []

    for g in range(G):
        fam = families[g]

        # Slice per-gene quantities → [S]
        E_g = E_all[g]
        E_s1_g = E_s1_all[g]
        E_s2_g = E_s2_all[g]
        Ebar_g = Ebar_all[g]
        mt_g = mt_all[g]
        theta_g = theta_stack[g]
        pS_g = log_pS_all[g]
        pD_g = log_pD_all[g]
        pL_g = log_pL_all[g]

        # Build single-family batch for wave_layout
        single_item = {
            'ccp': fam['ccp_helpers'],
            'leaf_row_index': fam['leaf_row_index'],
            'leaf_col_index': fam['leaf_col_index'],
            'root_clade_id': int(fam['root_clade_id']),
        }
        single_batched = collate_gene_families([single_item], dtype=dtype, device=device)

        waves_g, phases_g = compute_clade_waves(fam['ccp_helpers'])
        wave_layout_g = build_wave_layout(
            waves=waves_g, phases=phases_g,
            ccp_helpers=single_batched['ccp'],
            leaf_row_index=single_batched['leaf_row_index'],
            leaf_col_index=single_batched['leaf_col_index'],
            root_clade_ids=single_batched['root_clade_ids'],
            device=device, dtype=dtype,
        )

        # Pi forward for this family (needed for backward)
        Pi_out_g = Pi_wave_forward(
            wave_layout=wave_layout_g, species_helpers=species_helpers,
            E=E_g, Ebar=Ebar_g, E_s1=E_s1_g, E_s2=E_s2_g,
            log_pS=pS_g, log_pD=pD_g, log_pL=pL_g,
            transfer_mat=None, max_transfer_mat=mt_g,
            device=device, dtype=dtype,
            local_iters=local_iters, local_tolerance=local_tolerance,
            pibar_mode=pibar_mode,
        )

        # Backward: per-family gradient
        grad_theta_g, statsG = implicit_grad_loglik_vjp_wave(
            wave_layout_g, species_helpers,
            Pi_star_wave=Pi_out_g['Pi_wave_ordered'],
            Pibar_star_wave=Pi_out_g['Pibar_wave_ordered'],
            E_star=E_g, E_s1=E_s1_g, E_s2=E_s2_g, Ebar=Ebar_g,
            log_pS=pS_g, log_pD=pD_g, log_pL=pL_g,
            max_transfer_mat=mt_g,
            root_clade_ids_perm=wave_layout_g['root_clade_ids'],
            theta=theta_g,
            unnorm_row_max=unnorm_row_max,
            specieswise=specieswise,
            device=device, dtype=dtype,
            neumann_terms=neumann_terms,
            use_pruning=use_pruning,
            pruning_threshold=pruning_threshold,
            cg_tol=cg_tol, cg_maxiter=cg_maxiter,
            gmres_restart=gmres_restart,
            pibar_mode=pibar_mode,
        )

        grad_thetas.append(grad_theta_g)
        all_stats.append(statsG)

    return torch.stack(grad_thetas, dim=0), all_stats


# -------------------------------------------------------------------------
# Wave-based optimizer (uses wave forward + wave backward)
# -------------------------------------------------------------------------

def optimize_theta_wave(
    wave_layout,
    species_helpers,
    root_clade_ids,
    unnorm_row_max,
    theta_init,
    *,
    transfer_mat_unnormalized=None,
    steps: int = 200,
    lr: float = 0.2,
    tol_theta: float = 1e-3,
    e_max_iters: int = 2000,
    e_tol: float = 1e-8,
    neumann_terms: int = 4,
    use_pruning: bool = True,
    pruning_threshold: float = 1e-6,
    cg_tol: float = 1e-8,
    cg_maxiter: int = 500,
    gmres_restart: int = 40,
    specieswise: bool = False,
    device=None,
    dtype=torch.float64,
    pibar_mode: str = 'uniform_approx',
    optimizer: str = 'adam',
    momentum: float = 0.9,
):
    """Optimize theta using wave forward/backward + implicit gradient.

    Parameters
    ----------
    wave_layout : dict
        From build_wave_layout().
    species_helpers : dict
        Species tree helpers.
    root_clade_ids : Tensor
        Root clade IDs (original ordering) for likelihood computation.
    unnorm_row_max : Tensor [S]
        Row maxima of unnormalized transfer matrix.
    theta_init : Tensor [3] or [S, 3]
        Initial rate parameters (log-space).
    steps : int
        Maximum optimization iterations (ignored for 'lbfgs').
    lr : float
        Learning rate (ignored for 'lbfgs').
    optimizer : str
        'adam', 'sgd', or 'lbfgs'.
    momentum : float
        Momentum for SGD (default 0.9, ignored for adam/lbfgs).

    Returns
    -------
    dict with 'theta', 'rates', 'log_likelihood', 'negative_log_likelihood', 'history'.
    """
    if device is None:
        device = theta_init.device

    _THETA_MIN = math.log2(1e-10)

    # Precompute ancestors_T for uniform mode
    _ancestors_T = None
    if pibar_mode == 'uniform':
        anc_dense = species_helpers['ancestors_dense'].to(device=device, dtype=dtype)
        _ancestors_T = anc_dense.T.to_sparse_coo()

    def _extract(theta_d):
        if pibar_mode in ('dense', 'topk') and transfer_mat_unnormalized is not None:
            log_pS, log_pD, log_pL, transfer_mat, mt_raw = extract_parameters(
                theta_d, transfer_mat_unnormalized,
                genewise=False, specieswise=specieswise, pairwise=False,
            )
            mt = mt_raw.squeeze(-1) if mt_raw.ndim == 2 else mt_raw
        else:
            log_pS, log_pD, log_pL, transfer_mat, mt = extract_parameters_uniform(
                theta_d, unnorm_row_max, specieswise=specieswise,
            )
        return log_pS, log_pD, log_pL, transfer_mat, mt

    def _forward_backward(theta_d, warm_E):
        """Full forward + backward: returns (nll, grad_theta, history_record, warm_E)."""
        with torch.no_grad():
            log_pS, log_pD, log_pL, transfer_mat, mt = _extract(theta_d)

            E_out = E_fixed_point(
                species_helpers=species_helpers,
                log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
                transfer_mat=transfer_mat, max_transfer_mat=mt,
                max_iters=e_max_iters, tolerance=e_tol,
                warm_start_E=warm_E,
                dtype=dtype, device=device, pibar_mode=pibar_mode,
                ancestors_T=_ancestors_T,
            )

            Pi_out = Pi_wave_forward(
                wave_layout=wave_layout, species_helpers=species_helpers,
                E=E_out['E'], Ebar=E_out['E_bar'],
                E_s1=E_out['E_s1'], E_s2=E_out['E_s2'],
                log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
                transfer_mat=transfer_mat, max_transfer_mat=mt,
                device=device, dtype=dtype, pibar_mode=pibar_mode,
            )

            logL = compute_log_likelihood(Pi_out['Pi'], E_out['E'], root_clade_ids)
            nll = float(logL.sum().item())

        grad_theta, statsG = implicit_grad_loglik_vjp_wave(
            wave_layout, species_helpers,
            Pi_star_wave=Pi_out['Pi_wave_ordered'],
            Pibar_star_wave=Pi_out['Pibar_wave_ordered'],
            E_star=E_out['E'], E_s1=E_out['E_s1'],
            E_s2=E_out['E_s2'], Ebar=E_out['E_bar'],
            log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
            max_transfer_mat=mt,
            root_clade_ids_perm=wave_layout['root_clade_ids'],
            theta=theta_d,
            unnorm_row_max=unnorm_row_max,
            specieswise=specieswise,
            device=device, dtype=dtype,
            neumann_terms=neumann_terms,
            use_pruning=use_pruning,
            pruning_threshold=pruning_threshold,
            cg_tol=cg_tol, cg_maxiter=cg_maxiter,
            gmres_restart=gmres_restart,
            pibar_mode=pibar_mode,
            transfer_mat=transfer_mat,
            transfer_mat_unnormalized=transfer_mat_unnormalized,
            ancestors_T=_ancestors_T,
        )

        return nll, grad_theta, statsG, E_out

    # --- L-BFGS-B path (scipy) ---
    if optimizer == 'lbfgs':
        import numpy as np
        from scipy.optimize import minimize as scipy_minimize

        theta_t = theta_init.to(device=device, dtype=dtype).clone()
        theta_shape = theta_t.shape
        history: List[StepRecord] = []
        warm_E_ref = [None]
        eval_count = [0]

        def forward_and_grad(theta_flat_np):
            theta_flat = torch.from_numpy(theta_flat_np).to(device=device, dtype=dtype)
            theta_d = theta_flat.reshape(theta_shape).clamp(min=_THETA_MIN)

            nll, grad_theta, statsG, E_out = _forward_backward(theta_d, warm_E_ref[0])
            warm_E_ref[0] = E_out['E'].detach()

            eval_count[0] += 1
            fp_info = FixedPointInfo(iterations_E=int(E_out['iterations']),
                                     iterations_Pi=0)
            history.append(StepRecord(
                iteration=eval_count[0], theta=theta_d.detach().cpu(),
                rates=torch.exp2(theta_d.detach()).cpu(),
                negative_log_likelihood=nll, log_likelihood=-nll,
                theta_step_inf=0.0,
                grad_infinity_norm=float(grad_theta.abs().max().item()),
                fp_info=fp_info, gradient=grad_theta.cpu(),
                solve_stats_F=LinearSolveStats("wave_neumann", neumann_terms, 0.0, False),
                solve_stats_G=statsG,
            ))

            grad_np = grad_theta.reshape(-1).cpu().to(torch.float64).numpy()
            np.nan_to_num(grad_np, copy=False, nan=0.0)
            return float(nll), grad_np

        x0 = theta_t.reshape(-1).cpu().to(torch.float64).numpy()
        bounds = [(_THETA_MIN, None)] * len(x0)
        result = scipy_minimize(
            forward_and_grad, x0, method='L-BFGS-B', jac=True,
            bounds=bounds,
            options={'maxiter': steps, 'ftol': 1e-12, 'gtol': 1e-6},
        )

        theta_final = torch.from_numpy(result.x).to(device=device, dtype=dtype).reshape(theta_shape)
        return {
            "theta": theta_final.cpu(),
            "rates": torch.exp2(theta_final).cpu(),
            "negative_log_likelihood": result.fun,
            "log_likelihood": -result.fun,
            "history": history,
            "scipy_result": result,
        }

    # --- Iterative optimizers (adam, sgd) ---
    theta = torch.nn.Parameter(theta_init.to(device=device, dtype=dtype).clone())
    if optimizer == 'sgd':
        opt = torch.optim.SGD([theta], lr=lr, momentum=momentum)
    else:
        opt = torch.optim.Adam([theta], lr=lr)

    history: List[StepRecord] = []
    prev_theta = theta.detach().clone()
    warm_E = None

    for it in range(1, steps + 1):
        theta_d = theta.detach()

        nll, grad_theta, statsG, E_out = _forward_backward(theta_d, warm_E)
        warm_E = E_out['E'].detach()
        iters_E = int(E_out['iterations'])

        # Optimizer step (minimize NLL)
        opt.zero_grad(set_to_none=True)
        grad_clean = grad_theta.clone()
        grad_clean.nan_to_num_(nan=0.0)
        theta.grad = grad_clean
        opt.step()

        with torch.no_grad():
            theta.clamp_(min=_THETA_MIN)

        # Bookkeeping
        theta_detached = theta.detach()
        diff = float(torch.max(torch.abs(theta_detached - prev_theta)).item())
        prev_theta = theta_detached.clone()
        rates = torch.exp2(theta_detached)
        grad_inf = float(grad_theta.abs().max().item())

        fp_info = FixedPointInfo(iterations_E=iters_E, iterations_Pi=0)
        statsF_dummy = LinearSolveStats("wave_neumann", neumann_terms, 0.0, False)

        history.append(
            StepRecord(
                iteration=it,
                theta=theta_detached.cpu(),
                rates=rates.cpu(),
                negative_log_likelihood=nll,
                log_likelihood=-nll,
                theta_step_inf=diff,
                grad_infinity_norm=grad_inf,
                fp_info=fp_info,
                gradient=grad_theta.cpu(),
                solve_stats_F=statsF_dummy,
                solve_stats_G=statsG,
            )
        )

        if diff < tol_theta and it > 1:
            break

    return {
        "theta": theta.detach().cpu(),
        "rates": torch.exp2(theta.detach()).cpu(),
        "negative_log_likelihood": history[-1].negative_log_likelihood if history else float('nan'),
        "log_likelihood": history[-1].log_likelihood if history else float('nan'),
        "history": history,
    }


# -------------------------------------------------------------------------
# Genewise L-BFGS optimizer
# -------------------------------------------------------------------------

@torch.no_grad()
def _lbfgs_two_loop(grad, S_hist, Y_hist, history_len):
    """Standard L-BFGS two-loop recursion, vectorized over G genes.

    Parameters
    ----------
    grad : Tensor [G, P]
    S_hist, Y_hist : Tensor [G, m, P]
    history_len : int

    Returns
    -------
    direction : Tensor [G, P]  (negative direction, i.e. descent)
    """
    q = grad.clone()
    alphas = []
    for i in range(history_len - 1, -1, -1):
        sy = (Y_hist[:, i] * S_hist[:, i]).sum(-1)  # [G]
        rho_i = torch.where(sy > 1e-10, 1.0 / sy, torch.zeros_like(sy))
        a_i = rho_i * (S_hist[:, i] * q).sum(-1)    # [G]
        alphas.append(a_i)
        q = q - a_i.unsqueeze(-1) * Y_hist[:, i]
    # H0 scaling
    s_last = S_hist[:, history_len - 1]
    y_last = Y_hist[:, history_len - 1]
    sy_last = (s_last * y_last).sum(-1)
    yy_last = (y_last * y_last).sum(-1)
    gamma = torch.where(
        (sy_last > 1e-10) & (yy_last > 1e-10),
        sy_last / yy_last,
        torch.ones_like(yy_last),
    )
    r = gamma.unsqueeze(-1) * q
    for i, a_i in zip(range(history_len), reversed(alphas)):
        sy = (Y_hist[:, i] * S_hist[:, i]).sum(-1)
        rho_i = torch.where(sy > 1e-10, 1.0 / sy, torch.zeros_like(sy))
        beta_i = rho_i * (Y_hist[:, i] * r).sum(-1)
        r = r + (a_i - beta_i).unsqueeze(-1) * S_hist[:, i]
    return -r


def optimize_theta_genewise(
    families,
    species_helpers,
    unnorm_row_max,
    theta_init,
    *,
    max_steps=30,
    lbfgs_m=10,
    grad_tol=1e-5,
    e_max_iters=2000,
    e_tol=1e-8,
    neumann_terms=3,
    pruning_threshold=1e-6,
    cg_tol=1e-8,
    cg_maxiter=500,
    device=None,
    dtype=torch.float32,
    pibar_mode='uniform_approx',
    specieswise=False,
    local_iters=2000,
    local_tolerance=1e-3,
):
    """Genewise L-BFGS: independently optimize (D_g, L_g, T_g) per gene family.

    Each gene has its own E (so Pi forward is per-family sequential), but
    E_fixed_point is batched across all G genes for efficiency.
    Uses cross-family batched backward with per-clade pruning.

    Parameters
    ----------
    families : list[dict]
        G family dicts with 'ccp_helpers', 'leaf_row_index', 'leaf_col_index',
        'root_clade_id'.
    species_helpers : dict
        Shared species tree helpers.
    unnorm_row_max : Tensor [S]
        Row maxima of unnormalized transfer matrix.
    theta_init : Tensor [G, 3]
        Initial rate parameters.
    max_steps : int
        Maximum L-BFGS iterations.
    lbfgs_m : int
        L-BFGS history depth.
    grad_tol : float
        Convergence tolerance on gradient inf-norm.
    e_max_iters, e_tol : int, float
        E fixed-point solver parameters.
    neumann_terms : int
        Neumann series terms for Pi backward.
    cg_tol, cg_maxiter : float, int
        CG solver parameters for E adjoint.
    device, dtype : device, dtype
    pibar_mode : str
        Pibar computation mode.
    specieswise : bool
        Whether rates are per-species.
    local_iters, local_tolerance : int, float
        Pi forward convergence parameters.

    Returns
    -------
    dict with 'theta' [G,3], 'nll' [G], 'rates' [G,3], 'history' list[dict].
    """
    from src.core.batching import collate_gene_families, build_wave_layout
    from src.core.scheduling import compute_clade_waves

    if device is None:
        device = theta_init.device

    _THETA_MIN = math.log2(1e-10)

    G = len(families)
    P = theta_init.shape[-1] if theta_init.ndim == 2 else theta_init.shape[-1] * theta_init.shape[-2]
    unnorm_row_max = unnorm_row_max.to(device=device, dtype=dtype)

    # Move species_helpers tensors to device (skip large [S,S] matrices for uniform modes)
    _skip_keys = set()
    if pibar_mode == 'uniform_approx':
        _skip_keys = {'ancestors_dense', 'Recipients_mat'}
    elif pibar_mode == 'uniform':
        _skip_keys = {'Recipients_mat'}
    def _move_tensor(t):
        if t.dtype.is_floating_point:
            return t.to(device=device, dtype=dtype)
        return t.to(device=device)
    species_helpers = {
        k: (_move_tensor(v) if torch.is_tensor(v) and k not in _skip_keys else v)
        for k, v in species_helpers.items()
    }

    from src.core.batching import collate_wave

    # --- Phase A: precompute wave layouts (once) ---
    wave_layouts = []
    root_clade_ids_list = []
    per_family_waves = []
    per_family_phases = []
    batch_items = []
    for g in range(G):
        fam = families[g]
        single_item = {
            'ccp': fam['ccp_helpers'],
            'leaf_row_index': fam['leaf_row_index'],
            'leaf_col_index': fam['leaf_col_index'],
            'root_clade_id': int(fam['root_clade_id']),
        }
        single_batched = collate_gene_families([single_item], dtype=dtype, device=device)
        waves_g, phases_g = compute_clade_waves(fam['ccp_helpers'])
        wl_g = build_wave_layout(
            waves=waves_g, phases=phases_g,
            ccp_helpers=single_batched['ccp'],
            leaf_row_index=single_batched['leaf_row_index'],
            leaf_col_index=single_batched['leaf_col_index'],
            root_clade_ids=single_batched['root_clade_ids'],
            device=device, dtype=dtype,
        )
        wave_layouts.append(wl_g)
        root_clade_ids_list.append(single_batched['root_clade_ids'])
        per_family_waves.append(waves_g)
        per_family_phases.append(phases_g)
        batch_items.append(single_item)

    # Build merged cross-family layout for batched backward
    all_batched = collate_gene_families(batch_items, dtype=dtype, device=device)
    family_meta = all_batched['family_meta']
    offsets = [m['clade_offset'] for m in family_meta]
    sizes = [m['C'] for m in family_meta]
    cross_waves = collate_wave(per_family_waves, offsets)
    cross_phases = [max(per_family_phases[g][k] for g in range(G) if k < len(per_family_phases[g]))
                    for k in range(len(cross_waves))]
    merged_layout = build_wave_layout(
        waves=cross_waves, phases=cross_phases,
        ccp_helpers=all_batched['ccp'],
        leaf_row_index=all_batched['leaf_row_index'],
        leaf_col_index=all_batched['leaf_col_index'],
        root_clade_ids=all_batched['root_clade_ids'],
        device=device, dtype=dtype,
        family_clade_counts=sizes,
        family_clade_offsets=offsets,
    )
    C_total = int(all_batched['ccp']['C'])
    S = species_helpers['S']

    # Precompute ancestors_T for uniform mode
    _ancestors_T = None
    if pibar_mode == 'uniform':
        anc_dense = species_helpers['ancestors_dense'].to(device=device, dtype=dtype)
        _ancestors_T = anc_dense.T.to_sparse_coo()

    # --- Phase B: define eval functions ---

    def _eval_E(theta_t, warm_E):
        """Vectorized E solve for all G genes."""
        log_pS, log_pD, log_pL, _, mt = extract_parameters_uniform(
            theta_t, unnorm_row_max, specieswise=specieswise, genewise=True,
        )
        E_out = E_fixed_point(
            species_helpers=species_helpers,
            log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
            transfer_mat=None, max_transfer_mat=mt,
            max_iters=e_max_iters, tolerance=e_tol,
            warm_start_E=warm_E,
            dtype=dtype, device=device, pibar_mode=pibar_mode,
            ancestors_T=_ancestors_T,
        )
        return log_pS, log_pD, log_pL, mt, E_out

    def _nll_and_grad(theta_t, warm_E, active_mask):
        """Full forward + batched backward for active genes."""
        log_pS, log_pD, log_pL, mt, E_out = _eval_E(theta_t, warm_E)

        nll = torch.full((G,), float('nan'), device=device, dtype=dtype)
        grad = torch.zeros_like(theta_t)
        active_genes = active_mask.nonzero(as_tuple=False).squeeze(-1).tolist()

        # --- Per-family forward + NLL ---
        Pi_orig = {}  # g → Pi in original (unpermuted) space
        Pibar_orig = {}
        for g in active_genes:
            E_g = E_out['E'][g]
            Ebar_g = E_out['E_bar'][g]
            E_s1_g = E_out['E_s1'][g]
            E_s2_g = E_out['E_s2'][g]
            mt_g = mt[g]
            pS_g = log_pS[g] if log_pS.ndim >= 1 else log_pS
            pD_g = log_pD[g] if log_pD.ndim >= 1 else log_pD
            pL_g = log_pL[g] if log_pL.ndim >= 1 else log_pL

            Pi_out_g = Pi_wave_forward(
                wave_layout=wave_layouts[g], species_helpers=species_helpers,
                E=E_g, Ebar=Ebar_g, E_s1=E_s1_g, E_s2=E_s2_g,
                log_pS=pS_g, log_pD=pD_g, log_pL=pL_g,
                transfer_mat=None, max_transfer_mat=mt_g,
                device=device, dtype=dtype,
                local_iters=local_iters, local_tolerance=local_tolerance,
                pibar_mode=pibar_mode,
            )
            logL_g = compute_log_likelihood(Pi_out_g['Pi'], E_g, root_clade_ids_list[g])
            nll[g] = logL_g.sum()

            # Un-permute to original clade space
            # perm[orig] = wave_pos, so Pi_wave[perm] gives original order
            perm_g = wave_layouts[g]['perm']
            Pi_orig[g] = Pi_out_g['Pi_wave_ordered'][perm_g]
            Pibar_orig[g] = Pi_out_g['Pibar_wave_ordered'][perm_g]

        # --- Merge into batched tensor in merged layout order ---
        merged_perm = merged_layout['perm']
        Pi_star_merged = torch.full((C_total, S), float('-inf'), device=device, dtype=dtype)
        Pibar_star_merged = torch.full((C_total, S), float('-inf'), device=device, dtype=dtype)
        for g in active_genes:
            o, c = offsets[g], sizes[g]
            Pi_star_merged[merged_perm[o:o+c]] = Pi_orig[g]
            Pibar_star_merged[merged_perm[o:o+c]] = Pibar_orig[g]

        # --- Single batched backward ---
        pi_bwd = Pi_wave_backward(
            wave_layout=merged_layout,
            Pi_star_wave=Pi_star_merged,
            Pibar_star_wave=Pibar_star_merged,
            E=E_out['E'], Ebar=E_out['E_bar'],
            E_s1=E_out['E_s1'], E_s2=E_out['E_s2'],
            log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
            max_transfer_mat=mt,
            species_helpers=species_helpers,
            root_clade_ids_perm=merged_layout['root_clade_ids'],
            device=device, dtype=dtype,
            neumann_terms=neumann_terms,
            use_pruning=True,
            pruning_threshold=pruning_threshold,
            pibar_mode=pibar_mode,
            ancestors_T=_ancestors_T,
            family_idx=merged_layout['family_idx'],
        )

        # --- Per-family E adjoint + theta VJP ---
        for g in active_genes:
            E_g = E_out['E'][g]
            Ebar_g = E_out['E_bar'][g]
            mt_g = mt[g]
            pS_g = log_pS[g] if log_pS.ndim >= 1 else log_pS
            pD_g = log_pD[g] if log_pD.ndim >= 1 else log_pD
            pL_g = log_pL[g] if log_pL.ndim >= 1 else log_pL

            # Slice per-family gradients from batched backward
            pi_bwd_g = {
                'grad_E': pi_bwd['grad_E'][g],
                'grad_Ebar': pi_bwd['grad_Ebar'][g],
                'grad_E_s1': pi_bwd['grad_E_s1'][g],
                'grad_E_s2': pi_bwd['grad_E_s2'][g],
                'grad_log_pD': pi_bwd['grad_log_pD'][g],
                'grad_log_pS': pi_bwd['grad_log_pS'][g],
                'grad_max_transfer_mat': pi_bwd['grad_max_transfer_mat'][g],
            }

            grad_theta_g, _ = _e_adjoint_and_theta_vjp(
                pi_bwd_g, E_g, Ebar_g, E_out['E_s1'][g], E_out['E_s2'][g],
                pS_g, pD_g, pL_g, mt_g,
                species_helpers, wave_layouts[g]['root_clade_ids'],
                theta_t[g], unnorm_row_max, specieswise,
                device, dtype,
                cg_tol=cg_tol, cg_maxiter=cg_maxiter,
                pibar_mode=pibar_mode,
                ancestors_T=_ancestors_T,
            )
            grad[g] = grad_theta_g

        return nll, grad, E_out

    def _nll_only(theta_t, warm_E, active_mask):
        """Forward-only NLL (no backward). For line search probes."""
        log_pS, log_pD, log_pL, mt, E_out = _eval_E(theta_t, warm_E)

        nll = torch.full((G,), float('nan'), device=device, dtype=dtype)

        for g in active_mask.nonzero(as_tuple=False).squeeze(-1).tolist():
            E_g = E_out['E'][g]
            mt_g = mt[g]
            pS_g = log_pS[g] if log_pS.ndim >= 1 else log_pS
            pD_g = log_pD[g] if log_pD.ndim >= 1 else log_pD
            pL_g = log_pL[g] if log_pL.ndim >= 1 else log_pL

            Pi_out_g = Pi_wave_forward(
                wave_layout=wave_layouts[g], species_helpers=species_helpers,
                E=E_g, Ebar=E_out['E_bar'][g],
                E_s1=E_out['E_s1'][g], E_s2=E_out['E_s2'][g],
                log_pS=pS_g, log_pD=pD_g, log_pL=pL_g,
                transfer_mat=None, max_transfer_mat=mt_g,
                device=device, dtype=dtype,
                local_iters=local_iters, local_tolerance=local_tolerance,
                pibar_mode=pibar_mode,
            )

            logL_g = compute_log_likelihood(
                Pi_out_g['Pi'], E_g, root_clade_ids_list[g],
            )
            nll[g] = logL_g.sum()

        return nll, E_out

    # --- Phase C: L-BFGS loop ---
    theta = theta_init.to(device=device, dtype=dtype).clone()
    active = torch.ones(G, dtype=torch.bool, device=device)

    # Initial evaluation
    nll, grad, E_out = _nll_and_grad(theta, None, active)

    # L-BFGS history buffers
    theta_shape = theta.shape  # [G, P] or [G, S, 3]
    flat_P = theta[0].numel()
    S_hist = torch.zeros(G, lbfgs_m, flat_P, device=device, dtype=dtype)
    Y_hist = torch.zeros(G, lbfgs_m, flat_P, device=device, dtype=dtype)
    history_len = 0

    history = []
    history.append({
        'step': 0,
        'nll': nll.detach().cpu().clone(),
        'grad_inf': float(grad.abs().max().item()),
        'n_active': int(active.sum().item()),
    })

    for step in range(1, max_steps + 1):
        # Convergence masking
        grad_flat = grad.reshape(G, -1)
        active = active & (grad_flat.abs().max(dim=-1).values >= grad_tol)
        if not active.any():
            break

        # L-BFGS direction
        if history_len == 0:
            # Steepest descent for first step
            direction = -grad_flat
        else:
            direction = _lbfgs_two_loop(grad_flat, S_hist, Y_hist, history_len)
        direction[~active] = 0

        # Fall back to steepest descent for genes with non-descent direction
        slope = (grad_flat * direction).sum(-1)  # [G]
        bad_dir = active & (slope >= 0)
        if bad_dir.any():
            direction[bad_dir] = -grad_flat[bad_dir]
            slope[bad_dir] = (grad_flat[bad_dir] * direction[bad_dir]).sum(-1)

        # Per-gene Armijo backtracking line search
        alpha = torch.ones(G, device=device, dtype=dtype)
        ls_pending = active.clone()

        E_ls = None
        for _ in range(10):
            theta_try = (theta.reshape(G, -1) + alpha.unsqueeze(-1) * direction).reshape(theta_shape)
            theta_try = torch.where(theta_try < _THETA_MIN, _THETA_MIN, theta_try)
            nll_try, E_try = _nll_only(theta_try, E_out['E'].detach(), ls_pending)
            E_ls = E_try

            # Per-gene Armijo check (nll_try is nan for non-pending genes)
            armijo_ok = ls_pending & (nll_try <= nll + 1e-4 * alpha * slope)
            ls_pending = ls_pending & ~armijo_ok
            if not ls_pending.any():
                break
            alpha[ls_pending] *= 0.5

        # If line search failed (Armijo never satisfied), reject the step
        alpha[ls_pending] = 0.0

        # Construct final theta from per-gene alpha
        theta_accepted = (theta.reshape(G, -1) + alpha.unsqueeze(-1) * direction).reshape(theta_shape)
        theta_accepted = torch.where(theta_accepted < _THETA_MIN, _THETA_MIN, theta_accepted)

        # Full eval at accepted point (forward + backward)
        nll_new, grad_new, E_new = _nll_and_grad(
            theta_accepted, E_ls['E'].detach(), active,
        )

        # Update L-BFGS history (sanitize bad curvature entries)
        s_k = (theta_accepted - theta).reshape(G, -1)
        y_k = (grad_new - grad).reshape(G, -1)
        bad_curv = (s_k * y_k).sum(-1) <= 1e-10  # [G]
        s_k[bad_curv] = 0
        y_k[bad_curv] = 0
        if history_len < lbfgs_m:
            S_hist[:, history_len] = s_k
            Y_hist[:, history_len] = y_k
            history_len += 1
        else:
            S_hist[:, :-1] = S_hist[:, 1:].clone()
            S_hist[:, -1] = s_k
            Y_hist[:, :-1] = Y_hist[:, 1:].clone()
            Y_hist[:, -1] = y_k

        # Preserve NLL for converged (inactive) genes
        nll_new[~active] = nll[~active]
        theta, nll, grad, E_out = theta_accepted, nll_new, grad_new, E_new

        history.append({
            'step': step,
            'nll': nll.detach().cpu().clone(),
            'grad_inf': float(grad.abs().max().item()),
            'n_active': int(active.sum().item()),
            'alpha': alpha.detach().cpu().clone(),
        })

    # Compute final rates
    rates = torch.exp2(theta.detach())

    return {
        'theta': theta.detach().cpu(),
        'nll': nll.detach().cpu(),
        'rates': rates.detach().cpu(),
        'history': history,
    }
