"""Genewise L-BFGS optimizer for per-gene rate parameters."""
from __future__ import annotations

import math

import torch
from torch import Tensor

from src.core.likelihood import (
    E_fixed_point,
    Pi_wave_forward,
    Pi_wave_backward,
    compute_log_likelihood,
)
from src.core.extract_parameters import extract_parameters_uniform

from .implicit_grad import _e_adjoint_and_theta_vjp


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
    pibar_mode='uniform',
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
    from src.core.batching import collate_gene_families, build_wave_layout, collate_wave
    from src.core.scheduling import compute_clade_waves

    if device is None:
        device = theta_init.device

    _THETA_MIN = math.log2(1e-10)

    G = len(families)
    P = theta_init.shape[-1] if theta_init.ndim == 2 else theta_init.shape[-1] * theta_init.shape[-2]
    unnorm_row_max = unnorm_row_max.to(device=device, dtype=dtype)

    # Move species_helpers tensors to device (skip large [S,S] matrices for uniform modes)
    _skip_keys = set()
    if pibar_mode == 'uniform':
        _skip_keys = {'Recipients_mat'}
    def _move_tensor(t):
        if t.dtype.is_floating_point:
            return t.to(device=device, dtype=dtype)
        return t.to(device=device)
    species_helpers = {
        k: (_move_tensor(v) if torch.is_tensor(v) and k not in _skip_keys else v)
        for k, v in species_helpers.items()
    }

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
