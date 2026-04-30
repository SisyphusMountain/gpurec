"""Genewise L-BFGS optimizer for per-gene rate parameters."""
from __future__ import annotations

import math
import time

import torch

from gpurec.core.likelihood import E_fixed_point, compute_log_likelihood
from gpurec.core.forward import Pi_wave_forward
from gpurec.core.backward import Pi_wave_backward
from gpurec.core.extract_parameters import extract_parameters_uniform

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
    gene_batch_size=None,
    verbose=False,
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
    gene_batch_size : int | None
        Number of genes to process per mini-batch for Pi forward/backward.
        ``None`` or <=0 means full-batch over all genes.

    Returns
    -------
    dict with 'theta' [G,3], 'nll' [G], 'rates' [G,3], 'history' list[dict].
    """
    from gpurec.core.batching import collate_gene_families, build_wave_layout, collate_wave
    from gpurec.core.scheduling import compute_clade_waves

    if device is None:
        device = theta_init.device

    _THETA_MIN = math.log2(1e-10)

    G = len(families)
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

    # --- Phase A: precompute per-family scheduling primitives (once) ---
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
        waves_g, phases_g = compute_clade_waves(fam['ccp_helpers'])
        per_family_waves.append(waves_g)
        per_family_phases.append(phases_g)
        batch_items.append(single_item)

    if gene_batch_size is None or gene_batch_size <= 0:
        gene_batch_size = G
    gene_batch_size = int(gene_batch_size)
    if gene_batch_size < 1:
        raise ValueError(f"gene_batch_size must be >=1, got {gene_batch_size}")

    full_gene_ids = list(range(G))
    full_merged_layout = None
    layout_cache = {}
    _MAX_LAYOUT_CACHE = 512

    def _gene_batches(mask: torch.Tensor):
        """Yield lists of global gene indices for active genes in mini-batches."""
        active_idx = mask.nonzero(as_tuple=False).squeeze(-1)
        if active_idx.numel() == 0:
            return
        ids = active_idx.tolist()
        for i in range(0, len(ids), gene_batch_size):
            yield ids[i:i + gene_batch_size]

    def _build_merged_layout_for_gene_ids(gene_ids):
        """Build merged cross-family wave layout for a subset of genes."""
        if len(gene_ids) == 0:
            return None

        key = tuple(gene_ids)
        cached = layout_cache.get(key)
        if cached is not None:
            return cached

        nonlocal full_merged_layout
        if gene_ids == full_gene_ids:
            if full_merged_layout is None:
                chunk_items = [batch_items[g] for g in gene_ids]
                chunk_waves = [per_family_waves[g] for g in gene_ids]
                chunk_phases = [per_family_phases[g] for g in gene_ids]

                chunk_batched = collate_gene_families(chunk_items, dtype=dtype, device=device)
                family_meta = chunk_batched['family_meta']
                offsets = [m['clade_offset'] for m in family_meta]
                sizes = [m['C'] for m in family_meta]

                cross_waves = collate_wave(chunk_waves, offsets)
                cross_phases = []
                for k in range(len(cross_waves)):
                    phase_k = max(fp[k] for fp in chunk_phases if k < len(fp))
                    cross_phases.append(phase_k)

                full_merged_layout = build_wave_layout(
                    waves=cross_waves,
                    phases=cross_phases,
                    ccp_helpers=chunk_batched['ccp'],
                    leaf_row_index=chunk_batched['leaf_row_index'],
                    leaf_col_index=chunk_batched['leaf_col_index'],
                    root_clade_ids=chunk_batched['root_clade_ids'],
                    device=device,
                    dtype=dtype,
                    family_clade_counts=sizes,
                    family_clade_offsets=offsets,
                )
                layout_cache[key] = full_merged_layout
            return full_merged_layout

        chunk_items = [batch_items[g] for g in gene_ids]
        chunk_waves = [per_family_waves[g] for g in gene_ids]
        chunk_phases = [per_family_phases[g] for g in gene_ids]

        chunk_batched = collate_gene_families(chunk_items, dtype=dtype, device=device)
        family_meta = chunk_batched['family_meta']
        offsets = [m['clade_offset'] for m in family_meta]
        sizes = [m['C'] for m in family_meta]

        cross_waves = collate_wave(chunk_waves, offsets)
        cross_phases = []
        for k in range(len(cross_waves)):
            phase_k = max(fp[k] for fp in chunk_phases if k < len(fp))
            cross_phases.append(phase_k)

        layout = build_wave_layout(
            waves=cross_waves,
            phases=cross_phases,
            ccp_helpers=chunk_batched['ccp'],
            leaf_row_index=chunk_batched['leaf_row_index'],
            leaf_col_index=chunk_batched['leaf_col_index'],
            root_clade_ids=chunk_batched['root_clade_ids'],
            device=device,
            dtype=dtype,
            family_clade_counts=sizes,
            family_clade_offsets=offsets,
        )
        layout_cache[key] = layout
        if len(layout_cache) > _MAX_LAYOUT_CACHE:
            oldest_key = next(iter(layout_cache))
            if oldest_key != key:
                layout_cache.pop(oldest_key)
        return layout

    # Precompute ancestors_T for uniform mode
    _ancestors_T = None
    if pibar_mode == 'uniform':
        anc_dense = species_helpers['ancestors_dense'].to(device=device, dtype=dtype)
        _ancestors_T = anc_dense.T.to_sparse_coo()

    S = species_helpers['S']
    if torch.is_tensor(S):
        S = int(S.item())
    else:
        S = int(S)

    # --- Phase B: define eval functions ---

    def _eval_E_chunk(theta_t_chunk, warm_E_chunk):
        """E solve for a gene mini-batch."""
        log_pS, log_pD, log_pL, _, mt = extract_parameters_uniform(
            theta_t_chunk, unnorm_row_max, specieswise=specieswise, genewise=True,
        )
        E_out = E_fixed_point(
            species_helpers=species_helpers,
            log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
            transfer_mat=None, max_transfer_mat=mt,
            max_iters=e_max_iters, tolerance=e_tol,
            warm_start_E=warm_E_chunk,
            dtype=dtype, device=device, pibar_mode=pibar_mode,
            ancestors_T=_ancestors_T,
        )
        return log_pS, log_pD, log_pL, mt, E_out

    def _nll_and_grad(theta_t, warm_E, active_mask):
        """Mini-batched forward/backward for active genes."""

        nll = torch.full((G,), float('nan'), device=device, dtype=dtype)
        grad = torch.zeros_like(theta_t)
        E_full = torch.zeros((G, S), device=device, dtype=dtype)
        Ebar_full = torch.zeros((G, S), device=device, dtype=dtype)
        E_s1_full = torch.zeros((G, S), device=device, dtype=dtype)
        E_s2_full = torch.zeros((G, S), device=device, dtype=dtype)
        e_iters_max = 0

        if warm_E is not None:
            E_full.copy_(warm_E)

        for gene_ids in _gene_batches(active_mask):
            idx = torch.as_tensor(gene_ids, device=device, dtype=torch.long)
            merged_layout = _build_merged_layout_for_gene_ids(gene_ids)

            warm_E_chunk = None
            if warm_E is not None:
                warm_E_chunk = warm_E[idx]

            theta_chunk = theta_t[idx]
            log_pS_chunk, log_pD_chunk, log_pL_chunk, mt_chunk, E_chunk_out = _eval_E_chunk(
                theta_chunk, warm_E_chunk,
            )
            e_iters_max = max(e_iters_max, int(E_chunk_out['iterations']))

            E_chunk = E_chunk_out['E']
            Ebar_chunk = E_chunk_out['E_bar']
            E_s1_chunk = E_chunk_out['E_s1']
            E_s2_chunk = E_chunk_out['E_s2']

            E_full[idx] = E_chunk
            Ebar_full[idx] = Ebar_chunk
            E_s1_full[idx] = E_s1_chunk
            E_s2_full[idx] = E_s2_chunk

            # --- Mini-batched forward + per-family NLL ---
            Pi_out = Pi_wave_forward(
                wave_layout=merged_layout,
                species_helpers=species_helpers,
                E=E_chunk,
                Ebar=Ebar_chunk,
                E_s1=E_s1_chunk,
                E_s2=E_s2_chunk,
                log_pS=log_pS_chunk,
                log_pD=log_pD_chunk,
                log_pL=log_pL_chunk,
                transfer_mat=None,
                max_transfer_mat=mt_chunk,
                device=device,
                dtype=dtype,
                local_iters=local_iters,
                local_tolerance=local_tolerance,
                pibar_mode=pibar_mode,
                family_idx=merged_layout['family_idx'],
            )

            nll_chunk = compute_log_likelihood(
                Pi_out['Pi_wave_ordered'], E_chunk, merged_layout['root_clade_ids'],
            )
            nll[idx] = nll_chunk

            # --- Mini-batched backward ---
            pi_bwd = Pi_wave_backward(
                wave_layout=merged_layout,
                Pi_star_wave=Pi_out['Pi_wave_ordered'],
                Pibar_star_wave=Pi_out['Pibar_wave_ordered'],
                E=E_chunk,
                Ebar=Ebar_chunk,
                E_s1=E_s1_chunk,
                E_s2=E_s2_chunk,
                log_pS=log_pS_chunk,
                log_pD=log_pD_chunk,
                log_pL=log_pL_chunk,
                max_transfer_mat=mt_chunk,
                species_helpers=species_helpers,
                root_clade_ids_perm=merged_layout['root_clade_ids'],
                device=device,
                dtype=dtype,
                neumann_terms=neumann_terms,
                use_pruning=True,
                pruning_threshold=pruning_threshold,
                pibar_mode=pibar_mode,
                ancestors_T=_ancestors_T,
                family_idx=merged_layout['family_idx'],
            )

            # --- Mini-batched E adjoint + theta VJP ---
            grad_theta_chunk, _ = _e_adjoint_and_theta_vjp(
                pi_bwd,
                E_chunk,
                Ebar_chunk,
                E_s1_chunk,
                E_s2_chunk,
                log_pS_chunk,
                log_pD_chunk,
                log_pL_chunk,
                mt_chunk,
                species_helpers,
                merged_layout['root_clade_ids'],
                theta_chunk,
                unnorm_row_max,
                specieswise,
                device,
                dtype,
                genewise=True,
                cg_tol=cg_tol,
                cg_maxiter=cg_maxiter,
                pibar_mode=pibar_mode,
                ancestors_T=_ancestors_T,
            )
            grad[idx] = grad_theta_chunk

        E_out = {
            'E': E_full,
            'E_bar': Ebar_full,
            'E_s1': E_s1_full,
            'E_s2': E_s2_full,
            'iterations': e_iters_max,
        }
        return nll, grad, E_out

    def _nll_only(theta_t, warm_E, active_mask):
        """Forward-only NLL (no backward). For line search probes."""
        nll = torch.full((G,), float('nan'), device=device, dtype=dtype)
        if warm_E is not None:
            E_full = warm_E.clone()
        else:
            E_full = torch.zeros((G, S), device=device, dtype=dtype)
        e_iters_max = 0

        for gene_ids in _gene_batches(active_mask):
            idx = torch.as_tensor(gene_ids, device=device, dtype=torch.long)
            merged_layout = _build_merged_layout_for_gene_ids(gene_ids)

            warm_E_chunk = None
            if warm_E is not None:
                warm_E_chunk = warm_E[idx]

            theta_chunk = theta_t[idx]
            log_pS_chunk, log_pD_chunk, log_pL_chunk, mt_chunk, E_chunk_out = _eval_E_chunk(
                theta_chunk, warm_E_chunk,
            )
            e_iters_max = max(e_iters_max, int(E_chunk_out['iterations']))

            E_chunk = E_chunk_out['E']
            Ebar_chunk = E_chunk_out['E_bar']
            E_s1_chunk = E_chunk_out['E_s1']
            E_s2_chunk = E_chunk_out['E_s2']
            E_full[idx] = E_chunk

            Pi_out = Pi_wave_forward(
                wave_layout=merged_layout,
                species_helpers=species_helpers,
                E=E_chunk,
                Ebar=Ebar_chunk,
                E_s1=E_s1_chunk,
                E_s2=E_s2_chunk,
                log_pS=log_pS_chunk,
                log_pD=log_pD_chunk,
                log_pL=log_pL_chunk,
                transfer_mat=None,
                max_transfer_mat=mt_chunk,
                device=device,
                dtype=dtype,
                local_iters=local_iters,
                local_tolerance=local_tolerance,
                pibar_mode=pibar_mode,
                family_idx=merged_layout['family_idx'],
            )
            nll_chunk = compute_log_likelihood(
                Pi_out['Pi_wave_ordered'], E_chunk, merged_layout['root_clade_ids'],
            )
            nll[idx] = nll_chunk

        return nll, {'E': E_full, 'iterations': e_iters_max}

    # --- Phase C: L-BFGS loop ---
    theta = theta_init.to(device=device, dtype=dtype).clone()
    active = torch.ones(G, dtype=torch.bool, device=device)

    # Initial evaluation
    t0_init = time.perf_counter()
    nll, grad, E_out = _nll_and_grad(theta, None, active)
    init_time = time.perf_counter() - t0_init

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
        'e_iters': int(E_out['iterations']),
        'step_time_s': init_time,
    })
    if verbose:
        print(f"  step   0/{max_steps}  NLL={float(nll.sum()):.4f}  |g|={float(grad.abs().max()):.3e}"
              f"  active={G}/{G}  E_iters={int(E_out['iterations'])}  t={init_time:.2f}s",
              flush=True)

    for step in range(1, max_steps + 1):
        t0_step = time.perf_counter()
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
            armijo_rhs = nll + 1e-4 * alpha * slope
            armijo_ok = (
                ls_pending
                & torch.isfinite(nll_try)
                & torch.isfinite(armijo_rhs)
                & (nll_try <= armijo_rhs)
            )
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

        # Fail fast on per-gene non-finite evaluations.
        grad_new_flat = grad_new.reshape(G, -1)
        bad_eval = active & (
            ~torch.isfinite(nll_new)
            | ~torch.isfinite(grad_new_flat).all(dim=-1)
        )
        if bad_eval.any():
            bad_idx = bad_eval.nonzero(as_tuple=False).squeeze(-1)
            n_bad = int(bad_idx.numel())
            sample_idx = bad_idx[:10].tolist()
            nll_finite = int(torch.isfinite(nll_new[bad_eval]).sum().item())
            grad_finite = int(torch.isfinite(grad_new_flat[bad_eval]).all(dim=-1).sum().item())
            raise FloatingPointError(
                "Non-finite evaluation in optimize_theta_genewise "
                f"at step {step}: {n_bad} active genes produced non-finite values; "
                f"sample_gene_ids={sample_idx}; finite_nll={nll_finite}/{n_bad}; "
                f"finite_grad={grad_finite}/{n_bad}."
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

        step_time = time.perf_counter() - t0_step
        n_active = int(active.sum().item())
        e_it = int(E_new['iterations'])
        history.append({
            'step': step,
            'nll': nll.detach().cpu().clone(),
            'grad_inf': float(grad.abs().max().item()),
            'n_active': n_active,
            'alpha': alpha.detach().cpu().clone(),
            'e_iters': e_it,
            'step_time_s': step_time,
        })
        if verbose:
            print(f"  step {step:3d}/{max_steps}  NLL={float(nll.sum()):.4f}  |g|={float(grad.abs().max()):.3e}"
                  f"  active={n_active}/{G}  E_iters={e_it}  t={step_time:.2f}s",
                  flush=True)

    # Compute final rates
    rates = torch.exp2(theta.detach())

    return {
        'theta': theta.detach().cpu(),
        'nll': nll.detach().cpu(),
        'rates': rates.detach().cpu(),
        'history': history,
    }
