"""Implicit gradient: build VJP closures & solve the two transpose systems."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import func as tfunc

from gpurec.core.likelihood import (
    E_step,
    _uniform_ancestor_sum,
    compute_origination_denominator,
    prepare_origination_probs,
)
from gpurec.core.backward import Pi_wave_backward
from gpurec.core.log2_utils import _safe_log2_internal as _safe_log2
from gpurec.core.extract_parameters import extract_parameters_uniform


@dataclass
class _SolveStats:
    method: str
    iters: int
    rel_res: float
    success: bool = True
    neumann_terms: int | None = None
    gradient_convergence_delta: float | None = None
    gradient_convergence_threshold: float | None = None
    gradient_converged: bool | None = None
    pi_adjoint_residual_absmax: float | None = None
    pi_adjoint_residual_relmax: float | None = None
    pi_adjoint_residual_wave_count: int | None = None


def _as_float(value: torch.Tensor) -> float:
    return float(value.detach().cpu())


def _change_metrics(previous: torch.Tensor, current: torch.Tensor) -> tuple[float, float]:
    diff = torch.abs(current - previous)
    max_delta = _as_float(diff.max())
    scale = max(_as_float(previous.abs().max()), _as_float(current.abs().max()), 1.0)
    return max_delta, scale


def _iteration_schedule(max_terms: int, interval: int) -> list[int]:
    if max_terms < 1:
        raise ValueError("max_terms must be positive")
    if interval < 1:
        raise ValueError("interval must be positive")
    terms = list(range(interval, max_terms + 1, interval))
    if not terms or terms[-1] != max_terms:
        terms.append(max_terms)
    return terms


@torch.no_grad()
def _bicgstab(Av, b: torch.Tensor, *, tol: float = 1e-7, maxiter: int = 500):
    """Solve a nonsymmetric linear system with BiCGSTAB.

    Nonconvergence is reported through ``_SolveStats(success=False)`` and the
    current best iterate is still returned.  Callers choose whether that failed
    solve is fatal; the E-adjoint gradient path currently treats it as
    diagnostic telemetry.
    """
    x = torch.zeros_like(b)
    r = b - Av(x)
    bnorm = max(_as_float(torch.linalg.vector_norm(b)), 1.0)
    rel_res = _as_float(torch.linalg.vector_norm(r)) / bnorm
    if rel_res <= tol:
        return x, _SolveStats("BiCGSTAB", 0, rel_res)

    r_hat = r.clone()
    rho_old = torch.ones((), dtype=b.dtype, device=b.device)
    alpha = torch.ones((), dtype=b.dtype, device=b.device)
    omega = torch.ones((), dtype=b.dtype, device=b.device)
    v = torch.zeros_like(b)
    p = torch.zeros_like(b)

    success = False
    iters = 0
    for k in range(1, maxiter + 1):
        rho = torch.dot(r_hat, r)
        if _as_float(rho.abs()) <= 1e-30:
            break

        beta = (rho / rho_old) * (alpha / omega)
        p = r + beta * (p - omega * v)
        v = Av(p)
        denom = torch.dot(r_hat, v)
        if _as_float(denom.abs()) <= 1e-30:
            break

        alpha = rho / denom
        s = r - alpha * v
        rel_s = _as_float(torch.linalg.vector_norm(s)) / bnorm
        if rel_s <= tol:
            x = x + alpha * p
            rel_res = rel_s
            success = True
            iters = k
            break

        t = Av(s)
        tt = torch.dot(t, t)
        if _as_float(tt.abs()) <= 1e-30:
            break

        omega = torch.dot(t, s) / tt
        x = x + alpha * p + omega * s
        r = s - omega * t
        rel_res = _as_float(torch.linalg.vector_norm(r)) / bnorm
        iters = k
        if rel_res <= tol:
            success = True
            break
        if _as_float(omega.abs()) <= 1e-30:
            break
        rho_old = rho

    return x, _SolveStats("BiCGSTAB", iters, rel_res, success=success)


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
    origination_probs: Optional[torch.Tensor] = None,
    origination_probs_prepared: bool = False,
    genewise: bool = False,
    family_idx: Optional[torch.Tensor] = None,
    gradient_convergence_tol: float = -1.0,
    gradient_convergence_rtol: float = 0.0,
    gradient_convergence_check_interval: int = 4,
    pi_adjoint_initial_guess: Optional[torch.Tensor] = None,
    return_aux: bool = False,
    record_pi_adjoint_residual: bool = False,
    pi_fixed_point_relaxation: float = 1.0,
):
    """Internal API bridge for wave-decomposed ∇θ logL computation.

    This function is called by ``gpurec.api.model`` and
    ``gpurec.api.autograd`` to connect model/autograd state to the retained
    optimization internals.  It is intentionally not exported from
    ``gpurec.optimization.__all__``; external callers should use
    ``GeneReconModel`` or ``UniformChunkedReconModel`` public methods instead.

    Steps:
    1. Pi backward: wave-by-wave Neumann series (root→leaves)
    2. E adjoint: solve (I - G_E^T) w = q via the retained CG solve
    3. θ gradient: VJP through extract_parameters

    Returns (grad_theta, pi_backward_info).
    """
    def compute_with_terms(terms: int):
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
            neumann_terms=terms,
            use_pruning=use_pruning,
            pruning_threshold=pruning_threshold,
            family_idx=family_idx,
            uniform_pibar_row_max=uniform_pibar_row_max,
            origination_probs=origination_probs,
            origination_probs_prepared=origination_probs_prepared,
            initial_v_pi=pi_adjoint_initial_guess,
            return_residual_stats=record_pi_adjoint_residual,
            fixed_point_relaxation=pi_fixed_point_relaxation,
        )

        grad_theta, statsG, aux = _e_adjoint_and_theta_vjp(
            pi_bwd, E_star, Ebar, E_s1, E_s2,
            log_pS, log_pD, log_pL,
            max_transfer_mat, species_helpers, root_clade_ids_perm,
            theta, unnorm_row_max, specieswise,
            device, dtype,
            genewise=genewise,
            ancestors_T=ancestors_T,
            origination_probs=origination_probs,
            origination_probs_prepared=origination_probs_prepared,
            return_aux=True,
        )
        statsG.neumann_terms = int(terms)
        for key in (
            "pi_adjoint_residual_absmax",
            "pi_adjoint_residual_relmax",
            "pi_adjoint_residual_wave_count",
        ):
            if key in pi_bwd:
                setattr(statsG, key, pi_bwd[key])
        aux = dict(aux)
        aux["pi_adjoint"] = pi_bwd["v_Pi"].detach()
        aux["used_pi_initial_guess"] = bool(pi_bwd.get("used_pi_initial_guess", False))
        return grad_theta, statsG, aux

    if gradient_convergence_tol < 0.0:
        grad_theta, statsG, aux = compute_with_terms(neumann_terms)
        if return_aux:
            return grad_theta, statsG, aux
        return grad_theta, statsG

    previous_grad = None
    final_grad = None
    final_stats = None
    final_aux = None
    for terms in _iteration_schedule(
        neumann_terms,
        gradient_convergence_check_interval,
    ):
        grad_theta, statsG, aux = compute_with_terms(terms)
        final_grad = grad_theta
        final_stats = statsG
        final_aux = aux
        if previous_grad is not None:
            delta, scale = _change_metrics(previous_grad, grad_theta)
            threshold = gradient_convergence_tol + gradient_convergence_rtol * scale
            statsG.gradient_convergence_delta = delta
            statsG.gradient_convergence_threshold = threshold
            statsG.gradient_converged = delta <= threshold
            if delta <= threshold:
                break
        previous_grad = grad_theta.detach()

    if final_grad is None or final_stats is None or final_aux is None:
        raise RuntimeError("internal error: gradient convergence loop did not run")
    if return_aux:
        return final_grad, final_stats, final_aux
    return final_grad, final_stats


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
    origination_probs=None,
    origination_probs_prepared: bool = False,
    return_aux: bool = False,
):
    """E adjoint solve + theta VJP from pre-computed Pi backward result.

    Takes pi_bwd dict (from Pi_wave_backward) and completes the gradient
    computation through E adjoint solve and extract_parameters VJP.
    BiCGSTAB nonconvergence is diagnostic-only in this retained gradient path:
    the best returned E-adjoint iterate is consumed and ``_SolveStats.success``
    is forwarded so workflow history can surface failed batches.
    """
    # --- Step 2: E adjoint ---
    sp_P_idx = species_helpers['s_P_indexes']
    sp_c12_idx = species_helpers['s_C12_indexes']

    # Direct dNLL/dE from likelihood denominator
    n_fam = root_clade_ids_perm.numel()
    origin_probs = prepare_origination_probs(
        origination_probs,
        S=int(E_star.shape[-1]),
        device=device,
        dtype=dtype,
        family_count=int(n_fam) if origination_probs is not None else None,
        assume_prepared=origination_probs_prepared,
    )
    E_req_d = E_star.detach().requires_grad_(True)
    with torch.enable_grad():
        denom = compute_origination_denominator(
            E_req_d,
            origin_probs,
            origination_probs_prepared=True,
        )
        # Shared-E mode: denominator contributes once per family => n_fam * denom.
        # Genewise mode: E is per-family [G, S], so each row contributes once.
        family_specific_origin = origin_probs is not None and origin_probs.ndim == 2
        if family_specific_origin:
            direct_obj = denom.sum()
        elif E_req_d.ndim > 1 and E_req_d.shape[0] == n_fam:
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

    solve_tol = 1e-7 if dtype == torch.float32 else 1e-10
    w_flat, statsG = _bicgstab(AG_flat, q_flat, tol=solve_tol)

    wE = w_flat.view(E_shape)

    # --- Step 3: theta gradient through extract_parameters ---
    grad_mt_total = pi_bwd['grad_max_transfer_mat'] + pi_bwd['grad_Ebar']

    theta_req = theta.detach().requires_grad_(True)
    with torch.enable_grad():
        log_pS_r, log_pD_r, log_pL_r, mt_r = extract_parameters_uniform(
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
        log_pS_r2, log_pD_r2, log_pL_r2, mt_r2 = extract_parameters_uniform(
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
    if return_aux:
        return grad_theta, statsG, {}
    return grad_theta, statsG
