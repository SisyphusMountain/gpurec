"""Implicit gradient: build VJP closures & solve the two transpose systems."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import torch
from torch import func as tfunc

from gpurec.core.likelihood import E_step, _uniform_ancestor_sum
from gpurec.core.backward import Pi_wave_backward
from gpurec.core.log2_utils import _safe_log2_internal as _safe_log2
from gpurec.core.extract_parameters import extract_parameters_uniform


@dataclass
class _SolveStats:
    method: str
    iters: int
    rel_res: float
    success: bool = True


@torch.no_grad()
def _cg(Av, b: torch.Tensor, *, tol: float = 1e-8, maxiter: int = 500):
    x = torch.zeros_like(b)
    r = b - Av(x)
    p = r.clone()
    rr_old = float(torch.dot(r, r))
    bnorm = max(float(b.norm()) if b.numel() > 0 else 1.0, 1.0)
    rel_res = float(r.norm()) / bnorm
    if rel_res <= tol:
        return x, _SolveStats("CG", 0, rel_res)

    success = True
    iters = 0
    for k in range(1, maxiter + 1):
        Ap = Av(p)
        pAp = float(torch.dot(p, Ap))
        if pAp <= 0.0 or not math.isfinite(pAp):
            success = False
            iters = k - 1
            break
        alpha = rr_old / pAp
        x = x + alpha * p
        r = r - alpha * Ap
        rel_res = float(r.norm()) / bnorm
        iters = k
        if rel_res <= tol:
            return x, _SolveStats("CG", iters, rel_res)
        rr_new = float(torch.dot(r, r))
        p = r + (rr_new / max(rr_old, 1e-30)) * p
        rr_old = rr_new

    return x, _SolveStats("CG", iters, rel_res, success=success)


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
    ancestors_T: Optional[torch.Tensor] = None,
    uniform_pibar_row_max: Optional[torch.Tensor] = None,
):
    """Compute ∇θ logL using wave-decomposed backward pass + E adjoint.

    Steps:
    1. Pi backward: wave-by-wave Neumann series (root→leaves)
    2. E adjoint: solve (I - G_E^T) w = q via the retained CG solve
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
        ancestors_T=ancestors_T,
        uniform_pibar_row_max=uniform_pibar_row_max,
    )

    grad_theta, statsG = _e_adjoint_and_theta_vjp(
        pi_bwd, E_star, Ebar, E_s1, E_s2,
        log_pS, log_pD, log_pL,
        max_transfer_mat, species_helpers, root_clade_ids_perm,
        theta, unnorm_row_max, specieswise,
        device, dtype,
        ancestors_T=ancestors_T,
    )
    return grad_theta, statsG


def _e_adjoint_and_theta_vjp(
    pi_bwd,
    E_star, Ebar, E_s1, E_s2,
    log_pS, log_pD, log_pL,
    max_transfer_mat, species_helpers, root_clade_ids_perm,
    theta, unnorm_row_max, specieswise,
    device, dtype,
    *,
    genewise=False,
    ancestors_T=None,
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
        # Shared-E mode: denominator contributes once per family => n_fam * denom.
        # Genewise mode: E is per-family [G, S], so each row contributes once.
        if E_req_d.ndim > 1 and E_req_d.shape[0] == n_fam:
            direct_obj = denom.sum()
        else:
            direct_obj = (n_fam * denom).sum() if denom.ndim > 0 else (n_fam * denom)
        direct_dNLL_dE = torch.autograd.grad(direct_obj, E_req_d)[0]
    q_E = pi_bwd['grad_E'].clone() + direct_dNLL_dE

    # Chain Ebar gradient through E_step's Ebar computation
    if pi_bwd['grad_Ebar'].abs().max() > 0:
        E_req2 = E_star.detach().requires_grad_(True)
        with torch.enable_grad():
            mt_sq = max_transfer_mat.squeeze(-1) if max_transfer_mat.ndim > 1 else max_transfer_mat
            max_E = E_req2.max(dim=-1, keepdim=True).values
            expE = torch.exp2(E_req2 - max_E)
            if expE.ndim == 1:
                expE_2d = expE.unsqueeze(0)
                row_sum = expE_2d.sum(dim=-1, keepdim=True)
                ancestor_sum = _uniform_ancestor_sum(expE_2d, ancestors_T)
                Ebar_recomp = _safe_log2((row_sum - ancestor_sum).squeeze(0)) + max_E.squeeze(-1) + mt_sq
            else:
                row_sum = expE.sum(dim=-1, keepdim=True)
                ancestor_sum = _uniform_ancestor_sum(expE, ancestors_T)
                Ebar_recomp = _safe_log2(row_sum - ancestor_sum) + max_E + mt_sq
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
            from gpurec.core.terms import gather_E_children
            E_s12 = gather_E_children(E_req3, sp_P_idx, sp_c12_idx)
            E_s1_r, E_s2_r = torch.chunk(E_s12, 2, dim=-1)
            E_s1_r = E_s1_r.view(E_req3.shape)
            E_s2_r = E_s2_r.view(E_req3.shape)
            total = (E_s1_r * pi_bwd['grad_E_s1']).sum() + (E_s2_r * pi_bwd['grad_E_s2']).sum()
            es_to_e = torch.autograd.grad(total, E_req3, retain_graph=False)[0]
        q_E = q_E + es_to_e

    def G_E_fun(E_in):
        """E_step as a function of E only."""
        return E_step(
            E_in, sp_P_idx, sp_c12_idx,
            log_pS, log_pD, log_pL,
            max_transfer_mat,
            ancestors_T=ancestors_T,
        )[0]

    # Build VJP for G_E
    E_req_g = E_star.detach().requires_grad_(True)
    with torch.enable_grad():
        _, vjpG = tfunc.vjp(G_E_fun, E_req_g)

    E_shape = E_star.shape
    q_flat = q_E.reshape(-1)

    def AG_flat(w_flat):
        wE = w_flat.view(E_shape).contiguous()
        gE, = vjpG(wE.clone())
        return (wE - gE).reshape(-1)

    w_flat, statsG = _cg(AG_flat, q_flat)

    wE = w_flat.view(E_shape)

    # --- Step 3: theta gradient through extract_parameters ---
    grad_mt_total = pi_bwd['grad_max_transfer_mat'] + pi_bwd['grad_Ebar']

    theta_req = theta.detach().requires_grad_(True)
    with torch.enable_grad():
        log_pS_r, log_pD_r, log_pL_r, _, mt_r = extract_parameters_uniform(
            theta_req, unnorm_row_max, specieswise=specieswise, genewise=genewise,
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
        log_pS_r2, log_pD_r2, log_pL_r2, _, mt_r2 = extract_parameters_uniform(
            theta_req2, unnorm_row_max, specieswise=specieswise, genewise=genewise,
        )

        def G_E_theta(th_pS, th_pD, th_pL, th_mt):
            return E_step(
                E_star.detach(), sp_P_idx, sp_c12_idx,
                th_pS, th_pD, th_pL, th_mt,
                ancestors_T=ancestors_T,
            )[0]

        E_from_theta = G_E_theta(log_pS_r2, log_pD_r2, log_pL_r2, mt_r2)

        gtheta_E = torch.autograd.grad(
            E_from_theta, theta_req2,
            grad_outputs=wE,
            retain_graph=False,
        )[0]

    grad_theta = (grad_theta_pi + gtheta_E).detach()
    return grad_theta, statsG
