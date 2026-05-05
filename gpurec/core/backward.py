"""Backward pass: Pi_wave_backward and helpers (VJP / Neumann / GMRES)."""
import os

import torch

from .log2_utils import logsumexp2, logaddexp2, _safe_log2_internal as _safe_log2
from ._helpers import _safe_exp2_ratio  # noqa: F401

NEG_INF = float("-inf")


# ---------------------------------------------------------------------------
# Differentiable self-loop step (for torch.func.vjp tracing)
# ---------------------------------------------------------------------------

def _self_loop_differentiable(
    Pi_W, mt_squeezed, DL_const, Ebar, E, SL1_const, SL2_const,
    sp_child1, sp_child2, leaf_wt, dts_r, S,
    pibar_mode='uniform', transfer_mat_T=None, ancestors_T=None,
):
    """Pure-PyTorch differentiable self-loop step (Pibar + DTS_L).

    Computes one iteration of g_k(Pi_W) = logaddexp2(dts_r, DTS_L(Pi_W, Pibar(Pi_W))).
    Used by the backward pass to build VJP closures via torch.func.vjp.

    Args:
        Pi_W: [W, S] log2-space, requires_grad
        mt_squeezed: [S] or [W, S] max transfer mat
        DL_const, Ebar, E, SL1_const, SL2_const: [S] or [W, S] precomputed constants
        sp_child1, sp_child2: [S] species child indices
        leaf_wt: [W, S] leaf term
        dts_r: [W, S] or None, cross-clade DTS (frozen)
        S: int
        pibar_mode: 'dense', 'uniform', or 'topk'
        transfer_mat_T: [S, S] for dense/topk mode (may require grad for theta VJP)
        ancestors_T: [S, S] sparse COO, ancestors.T (for uniform mode)

    Returns:
        Pi_new: [W, S] log2-space
    """
    W = Pi_W.shape[0]

    def _expand(t):
        return t.unsqueeze(0).expand(W, -1) if t.ndim == 1 else t

    mt_w = _expand(mt_squeezed)
    DL_w = _expand(DL_const)
    Ebar_w = _expand(Ebar)
    E_w = _expand(E)
    SL1_w = _expand(SL1_const)
    SL2_w = _expand(SL2_const)

    # --- Pibar ---
    Pi_max = Pi_W.max(dim=1, keepdim=True).values
    Pi_exp = torch.exp2(Pi_W - Pi_max)
    if pibar_mode == 'uniform':
        row_sum = Pi_exp.sum(dim=1, keepdim=True)
        ancestor_sum = Pi_exp @ ancestors_T
        Pibar_W = _safe_log2(row_sum - ancestor_sum) + Pi_max + mt_w
    else:  # dense or topk
        Pibar_W = _safe_log2(Pi_exp @ transfer_mat_T) + Pi_max + mt_w

    # --- Species children of Pi_W ---
    Pi_W_pad = torch.cat([Pi_W, torch.full((W, 1), NEG_INF, device=Pi_W.device, dtype=Pi_W.dtype)], dim=1)
    Pi_s1 = Pi_W_pad[:, sp_child1.long()]
    Pi_s2 = Pi_W_pad[:, sp_child2.long()]

    # --- DTS_L: 6 terms ---
    DTS_L = torch.stack([
        DL_w + Pi_W,
        Pi_W + Ebar_w,
        Pibar_W + E_w,
        SL1_w + Pi_s1,
        SL2_w + Pi_s2,
        leaf_wt,
    ], dim=0)

    DTS_L_term = logsumexp2(DTS_L, dim=0)

    if dts_r is not None:
        return logaddexp2(dts_r, DTS_L_term)
    else:
        return DTS_L_term


# ---------------------------------------------------------------------------
# Differentiable cross-clade DTS
# ---------------------------------------------------------------------------

def _dts_cross_differentiable(
    Pi, Pibar, meta, sp_child1, sp_child2, log_pD, log_pS, S, device, dtype,
):
    """Differentiable DTS cross-clade computation for one wave.

    Same as _compute_dts_cross but uses pure PyTorch ops (no Triton)
    so that torch.func.vjp can trace through it.

    Args:
        Pi, Pibar: [C, S] full tensors (Pi requires_grad for children)
        meta: wave metadata dict
        sp_child1, sp_child2: [S] species child indices
        log_pD, log_pS: scalar or [S] event probabilities
        S: int

    Returns:
        dts_r: [W, S] reduced DTS cross-clade terms
    """
    sl = meta['sl']
    sr = meta['sr']
    sl_long = sl.long()
    sr_long = sr.long()
    wlsp = meta['log_split_probs']
    W = meta['W']
    n_ws = sl.shape[0]

    Pi_l = Pi[sl_long]
    Pi_r = Pi[sr_long]
    Pibar_l = Pibar[sl_long]
    Pibar_r = Pibar[sr_long]

    Pi_pad = torch.cat([Pi, torch.full((Pi.shape[0], 1), NEG_INF, device=device, dtype=dtype)], dim=1)
    Pi_l_s1 = Pi_pad[sl_long][:, sp_child1.long()]
    Pi_l_s2 = Pi_pad[sl_long][:, sp_child2.long()]
    Pi_r_s1 = Pi_pad[sr_long][:, sp_child1.long()]
    Pi_r_s2 = Pi_pad[sr_long][:, sp_child2.long()]

    DTS = torch.stack([
        log_pD + Pi_l + Pi_r,
        Pi_l + Pibar_r,
        Pi_r + Pibar_l,
        log_pS + Pi_l_s1 + Pi_r_s2,
        log_pS + Pi_r_s1 + Pi_l_s2,
    ], dim=0)

    dts_term = wlsp + logsumexp2(DTS, dim=0)

    reduce_idx = meta['reduce_idx'].long()
    reduce_expand = reduce_idx.unsqueeze(1).expand(n_ws, S)

    seg_max = torch.full((W, S), NEG_INF, device=device, dtype=dtype)
    seg_max.scatter_reduce_(0, reduce_expand, dts_term.detach(), reduce='amax',
                            include_self=True)

    # Avoid -inf - (-inf) -> NaN when a whole segment/species slice is unreachable.
    # In that case seg_max is -inf and all corresponding dts_term are -inf, so using
    # a finite shift (0) yields exp2(-inf)=0 and the reduced result remains -inf.
    seg_max_safe = torch.where(seg_max == NEG_INF, torch.zeros_like(seg_max), seg_max)
    shifted = torch.exp2(dts_term - seg_max_safe[reduce_idx])
    seg_sum = torch.zeros((W, S), device=device, dtype=dtype)
    seg_sum.scatter_add_(0, reduce_expand, shifted)
    dts_r = _safe_log2(seg_sum) + seg_max

    return dts_r


# ---------------------------------------------------------------------------
# VJP precompute
# ---------------------------------------------------------------------------

def _self_loop_vjp_precompute(
    Pi_star, Pibar_star, dts_r,
    mt_w, DL_w, Ebar_w, E_w, SL1_w, SL2_w,
    sp_child1, sp_child2, leaf_wt, S,
    pibar_mode, transfer_mat_T, ancestors_T,
):
    """Precompute softmax weights and Pibar VJP ingredients for one wave.

    Evaluates the self-loop g(Pi) at Pi=Pi_star and caches all quantities
    needed by _self_loop_Jt_apply (Neumann VJP) and param VJP.
    Called ONCE per wave.
    """
    W = Pi_star.shape[0]
    device, dtype = Pi_star.device, Pi_star.dtype

    def _expand(t):
        return t.unsqueeze(0).expand(W, -1) if t.ndim == 1 else t

    mt = _expand(mt_w)
    DL = _expand(DL_w)
    Ebar = _expand(Ebar_w)
    E = _expand(E_w)
    SL1 = _expand(SL1_w)
    SL2 = _expand(SL2_w)

    Pi_max = Pi_star.max(dim=1, keepdim=True).values
    p_prime = torch.exp2(Pi_star - Pi_max)

    if pibar_mode == 'uniform':
        row_sum = p_prime.sum(dim=1, keepdim=True)
        anc_sum = p_prime @ ancestors_T
        pibar_denom = row_sum - anc_sum
    else:
        pibar_matmul = p_prime @ transfer_mat_T

    Pi_pad = torch.cat([Pi_star, torch.full((W, 1), NEG_INF, device=device, dtype=dtype)], dim=1)
    Pi_s1 = Pi_pad[:, sp_child1.long()]
    Pi_s2 = Pi_pad[:, sp_child2.long()]

    terms = torch.stack([
        DL + Pi_star,
        Pi_star + Ebar,
        Pibar_star + E,
        SL1 + Pi_s1,
        SL2 + Pi_s2,
        leaf_wt,
    ], dim=0)

    DTS_L = logsumexp2(terms, dim=0)

    if dts_r is not None:
        Pi_new = logaddexp2(dts_r, DTS_L)
        w_L = _safe_exp2_ratio(DTS_L, Pi_new)
    else:
        w_L = torch.ones(W, S, device=device, dtype=dtype)

    w_terms = _safe_exp2_ratio(terms, DTS_L.unsqueeze(0))

    sc1 = sp_child1.long()
    sc2 = sp_child2.long()
    valid1 = sc1 < S
    valid2 = sc2 < S

    result = {
        'w_L': w_L,
        'w_terms': w_terms,
        'p_prime': p_prime,
    }
    if pibar_mode == 'uniform':
        pos = pibar_denom > 0
        inv_denom = torch.where(pos, 1.0 / torch.where(pos, pibar_denom, torch.ones_like(pibar_denom)),
                                torch.zeros_like(pibar_denom))
        result['pibar_inv_denom'] = inv_denom
    else:
        pos = pibar_matmul > 0
        inv_matmul = torch.where(pos, 1.0 / torch.where(pos, pibar_matmul, torch.ones_like(pibar_matmul)),
                                 torch.zeros_like(pibar_matmul))
        result['pibar_inv_matmul'] = inv_matmul
        result['pibar_matmul'] = pibar_matmul
    if valid1.any():
        result['sc1_valid'] = valid1
        result['sc1_idx'] = sc1[valid1].unsqueeze(0)
    if valid2.any():
        result['sc2_valid'] = valid2
        result['sc2_idx'] = sc2[valid2].unsqueeze(0)
    return result


# ---------------------------------------------------------------------------
# GMRES solver for (I - J^T) v = rhs
# ---------------------------------------------------------------------------

def _gmres_self_loop_solve(
    rhs, ingredients, sp_child1, sp_child2, S, W,
    pibar_mode, transfer_mat_T, ancestors_T,
    max_iters=30, tol=1e-5,
):
    """Solve (I - J_self^T) v = rhs via GMRES.

    Used when spectral radius of J_self^T is close to 1 (e.g., pibar_mode='uniform'),
    making the Neumann series diverge. Returns v [W, S].
    """
    n = W * S
    b = rhs.reshape(n)
    beta = b.norm()
    if beta < 1e-30:
        return rhs.clone()

    V = [b / beta]
    H = torch.zeros(max_iters + 1, max_iters, device=rhs.device, dtype=rhs.dtype)

    cs = torch.zeros(max_iters, device=rhs.device, dtype=rhs.dtype)
    sn = torch.zeros(max_iters, device=rhs.device, dtype=rhs.dtype)
    g = torch.zeros(max_iters + 1, device=rhs.device, dtype=rhs.dtype)
    g[0] = beta

    converged_j = 0
    for j in range(max_iters):
        vj_2d = V[j].reshape(W, S)
        Jt_vj = _self_loop_Jt_apply(
            vj_2d, ingredients, sp_child1, sp_child2, S, W,
            pibar_mode, transfer_mat_T, ancestors_T,
        )
        w = (vj_2d - Jt_vj).reshape(n)

        for i in range(j + 1):
            H[i, j] = w.dot(V[i])
            w = w - H[i, j] * V[i]
        H[j + 1, j] = w.norm()

        if H[j + 1, j] > 1e-14:
            V.append(w / H[j + 1, j])
        else:
            V.append(torch.zeros_like(w))

        for i in range(j):
            temp = cs[i] * H[i, j] + sn[i] * H[i + 1, j]
            H[i + 1, j] = -sn[i] * H[i, j] + cs[i] * H[i + 1, j]
            H[i, j] = temp

        denom = (H[j, j] ** 2 + H[j + 1, j] ** 2).sqrt()
        if denom > 1e-14:
            cs[j] = H[j, j] / denom
            sn[j] = H[j + 1, j] / denom
        else:
            cs[j] = 1.0
            sn[j] = 0.0

        H[j, j] = cs[j] * H[j, j] + sn[j] * H[j + 1, j]
        H[j + 1, j] = 0.0
        temp = cs[j] * g[j] + sn[j] * g[j + 1]
        g[j + 1] = -sn[j] * g[j] + cs[j] * g[j + 1]
        g[j] = temp

        converged_j = j + 1
        if abs(float(g[j + 1])) / float(beta) < tol:
            break

    m = converged_j
    y = torch.zeros(m, device=rhs.device, dtype=rhs.dtype)
    for i in range(m - 1, -1, -1):
        y[i] = (g[i] - H[i, i + 1:m] @ y[i + 1:m]) / H[i, i] if H[i, i].abs() > 1e-14 else 0.0

    v = torch.zeros(n, device=rhs.device, dtype=rhs.dtype)
    for i in range(m):
        v = v + float(y[i]) * V[i]

    return v.reshape(W, S)


# ---------------------------------------------------------------------------
# Analytical J^T application
# ---------------------------------------------------------------------------

def _self_loop_Jt_apply(
    v, ingredients, sp_child1, sp_child2, S, W,
    pibar_mode, transfer_mat_T, ancestors_T,
):
    """Apply J_self^T @ v analytically using precomputed ingredients.

    This is the VJP of one self-loop step g(Pi) = logaddexp2(dts_r, DTS_L(Pi)).
    The Jacobian J = dg/dPi is block-diagonal per clade (no cross-clade coupling
    in the self-loop). Each block captures:
      - diagonal: d(DTS_L)/d(Pi) through DL+Pi and Pi+Ebar terms
      - Pibar path: d(DTS_L)/d(Pibar) * d(Pibar)/d(Pi) through Pibar+E term
      - speciation: d(DTS_L)/d(Pi_s1) * d(Pi_s1)/d(Pi) scatter through SL terms
    """
    w_L = ingredients['w_L']
    w_terms = ingredients['w_terms']
    p_prime = ingredients['p_prime']

    alpha = v * w_L

    result = alpha * (w_terms[0] + w_terms[1])

    v_Pibar = alpha * w_terms[2]

    if pibar_mode == 'uniform':
        u_d = v_Pibar * ingredients['pibar_inv_denom']
        A = u_d.sum(dim=1, keepdim=True)
        correction = (ancestors_T @ u_d.T).T
        result = result + p_prime * (A - correction)
    else:  # dense / topk
        u_mr = v_Pibar * ingredients['pibar_inv_matmul']
        result = result + p_prime * (u_mr @ transfer_mat_T.T)

    sc1_valid = ingredients.get('sc1_valid')
    sc2_valid = ingredients.get('sc2_valid')
    sc1_idx = ingredients.get('sc1_idx')
    sc2_idx = ingredients.get('sc2_idx')

    if sc1_valid is not None:
        src = alpha * w_terms[3]
        idx = sc1_idx.expand(W, -1) if sc1_idx.shape[0] == 1 else sc1_idx
        result.scatter_add_(1, idx, src[:, sc1_valid])
    if sc2_valid is not None:
        src = alpha * w_terms[4]
        idx = sc2_idx.expand(W, -1) if sc2_idx.shape[0] == 1 else sc2_idx
        result.scatter_add_(1, idx, src[:, sc2_valid])

    return result


# ---------------------------------------------------------------------------
# Pi wave backward
# ---------------------------------------------------------------------------

@torch.no_grad()
def Pi_wave_backward(
    wave_layout,
    Pi_star_wave,
    Pibar_star_wave,
    E, Ebar, E_s1, E_s2,
    log_pS, log_pD, log_pL,
    max_transfer_mat,
    species_helpers,
    root_clade_ids_perm,
    device, dtype,
    *,
    neumann_terms=3,
    pruning_threshold=1e-6,
    use_pruning=True,
    pibar_mode='uniform',
    transfer_mat=None,
    ancestors_T=None,
    family_idx=None,
    uniform_pibar_row_max=None,
):
    """Wave-decomposed backward pass for implicit gradient computation.

    Computes dL/dPi via Neumann series per wave (root→leaves), then
    accumulates parameter gradients.  Always operates in batched mode
    internally; a single gene tree (family_idx=None) is handled as G=1.

    Args:
        wave_layout: dict from build_wave_layout()
        Pi_star_wave: [C, S] converged Pi in wave-ordered space
        Pibar_star_wave: [C, S] converged Pibar in wave-ordered space
        E, Ebar, E_s1, E_s2: [S] or [G, S] species extinction
        log_pS, log_pD, log_pL: scalar/[S] or [G]/[G, S] event probabilities
        max_transfer_mat: [S] or [G, S] log2-space
        species_helpers: species tree helpers
        root_clade_ids_perm: Long[F] root clade IDs in wave-ordered space
        device, dtype: target device/dtype
        neumann_terms: number of Neumann series terms (default 3)
        pruning_threshold: linear-space adjoint magnitude threshold for pruning
        use_pruning: whether to prune waves with negligible adjoint gradient
        pibar_mode: 'dense', 'uniform', or 'topk'
        transfer_mat: [S, S] linear-space transfer matrix (for dense mode)
        ancestors_T: [S, S] sparse CSR = ancestors.T (for uniform mode)
        family_idx: Long[C] clade→family mapping. None → auto-wrapped as G=1.
        uniform_pibar_row_max: optional [C] final forward-side row max values
            for uniform Pibar. Used only by opt-in fused cross-Pibar VJP paths.

    Returns:
        dict with:
            'v_Pi': [C, S] adjoint vector for Pi (wave-ordered)
            'grad_E': [S] or [G, S] gradient contribution from Pi adjoint to E
            'grad_log_pS': [S] or [G, S] gradient wrt log_pS
            'grad_log_pD': [S] or [G, S] gradient wrt log_pD
            'grad_max_transfer_mat': [S] or [G, S] gradient wrt max_transfer_mat
            'grad_transfer_mat': [S, S] gradient wrt transfer_mat (dense mode only)
    """
    # Fused Triton backward kernels (optional)
    try:
        from .kernels.wave_backward import (
            wave_backward_uniform_fused,
            dts_cross_backward_fused,
            dts_cross_backward_accum_fused,
            dts_cross_backward_accum_grouped_fused,
            dts_cross_backward_accum_parent_tiled_fused,
            dts_cross_backward_accum_parent_ragged_fused,
            _build_parent_ragged_ge2_worklist,
            uniform_cross_pibar_vjp_fused,
            uniform_cross_pibar_vjp_tree_fused,
            uniform_cross_pibar_vjp_tree_from_ud_fused,
            uniform_cross_pibar_vjp_tree_prefix_fused,
            uniform_cross_pibar_vjp_tree_grouped_fused,
            uniform_cross_pibar_vjp_grouped_tree_fused,
            pibar_row_stats_fused,
            active_mask_from_rhs_absmax_fused,
        )
        _HAS_FUSED_BACKWARD = True
    except ImportError:
        _HAS_FUSED_BACKWARD = False

    wave_metas = wave_layout['wave_metas']
    C, S = Pi_star_wave.shape
    K = len(wave_metas)

    target_device = torch.device(device)
    if target_device.type == 'cuda' and target_device.index is None:
        target_device = torch.device('cuda', torch.cuda.current_device())
    species_cache = species_helpers.get('_wave_forward_species_cache')
    sp_child1_cpu = sp_child2_cpu = None
    sp_child1 = sp_child2 = None
    p_cpu = c_cpu = mask_c1 = None
    if species_cache is not None and int(species_cache.get('S', -1)) == int(S):
        cached_child1 = species_cache.get('sp_child1')
        cached_child2 = species_cache.get('sp_child2')
        cached_child1_cpu = species_cache.get('sp_child1_cpu')
        cached_child2_cpu = species_cache.get('sp_child2_cpu')
        if (
            torch.is_tensor(cached_child1)
            and torch.is_tensor(cached_child2)
            and cached_child1.device == target_device
            and cached_child2.device == target_device
            and torch.is_tensor(cached_child1_cpu)
            and torch.is_tensor(cached_child2_cpu)
        ):
            sp_child1 = cached_child1
            sp_child2 = cached_child2
            sp_child1_cpu = cached_child1_cpu
            sp_child2_cpu = cached_child2_cpu

    if sp_child1 is None or sp_child2 is None:
        sp_P_idx = species_helpers['s_P_indexes']
        sp_c12_idx = species_helpers['s_C12_indexes']
        p_cpu = sp_P_idx.cpu().long()
        c_cpu = sp_c12_idx.cpu().long()
        mask_c1 = p_cpu < S
        sp_child1_cpu = torch.full((S,), S, dtype=torch.long)
        sp_child2_cpu = torch.full((S,), S, dtype=torch.long)
        sp_child1_cpu[p_cpu[mask_c1]] = c_cpu[mask_c1]
        sp_child2_cpu[p_cpu[~mask_c1] - S] = c_cpu[~mask_c1]
        sp_child1 = sp_child1_cpu.to(target_device)
        sp_child2 = sp_child2_cpu.to(target_device)
        if species_cache is not None and int(species_cache.get('S', -1)) == int(S):
            species_cache['sp_child1'] = sp_child1
            species_cache['sp_child2'] = sp_child2
            species_cache['sp_child1_cpu'] = sp_child1_cpu
            species_cache['sp_child2_cpu'] = sp_child2_cpu

    fused_cross_pibar_vjp_enabled = (
        os.environ.get("GPUREC_FUSED_CROSS_PIBAR_VJP", "1") != "0"
        and _HAS_FUSED_BACKWARD
        and pibar_mode == 'uniform'
        and dtype in (torch.float32, torch.float64)
        and device.type == 'cuda'
    )
    fused_cross_pibar_vjp_impl = os.environ.get(
        "GPUREC_FUSED_CROSS_PIBAR_VJP_IMPL", "tree"
    ).lower()
    prefix_cross_pibar_vjp_impl = fused_cross_pibar_vjp_impl in ("prefix", "tree_prefix")
    tree_cross_pibar_vjp_impl = fused_cross_pibar_vjp_impl in (
        "tree",
        "prefix",
        "tree_prefix",
    )
    grouped_cross_pibar_vjp_enabled = (
        os.environ.get("GPUREC_GROUPED_CROSS_PIBAR_VJP", "0") != "0"
    )
    grouped_cross_pibar_reduce_impl = os.environ.get(
        "GPUREC_GROUPED_CROSS_PIBAR_REDUCE_IMPL", "torch"
    ).lower()
    grouped_cross_pibar_use_active = (
        os.environ.get("GPUREC_GROUPED_CROSS_PIBAR_USE_ACTIVE", "1") != "0"
    )
    dts_pibar_ud_fusion_enabled = (
        os.environ.get("GPUREC_DTS_PIBAR_UD_FUSION", "1") != "0"
    )
    dts_pibar_ud_side_threshold = float(
        os.environ.get(
            "GPUREC_DTS_PIBAR_UD_SIDE_THRESHOLD",
            os.environ.get("GPUREC_DTS_PIBAR_UD_SIDE_BUDGET", "0"),
        )
    )
    if dts_pibar_ud_side_threshold < 0.0:
        raise ValueError("GPUREC_DTS_PIBAR_UD_SIDE_THRESHOLD must be non-negative")
    dts_pibar_ud_side_threshold_arg = (
        torch.tensor([dts_pibar_ud_side_threshold], device=target_device, dtype=dtype)
        if dts_pibar_ud_side_threshold > 0.0
        else 0.0
    )
    dts_pibar_ud_skip_zero_sides_enabled = (
        os.environ.get("GPUREC_DTS_PIBAR_UD_SKIP_ZERO_SIDES", "1") != "0"
        or os.environ.get("GPUREC_DTS_PIBAR_UD_WORKLIST", "0") != "0"
        or dts_pibar_ud_side_threshold > 0.0
    )
    dts_pibar_ud_compact_levels_enabled = (
        os.environ.get("GPUREC_DTS_PIBAR_UD_COMPACT_LEVELS", "1") != "0"
    )
    dts_pibar_ud_euler_prefix_enabled = (
        os.environ.get("GPUREC_DTS_PIBAR_UD_EULER_PREFIX", "0") != "0"
    )
    dts_pibar_ud_min_splits = int(
        os.environ.get("GPUREC_DTS_PIBAR_UD_MIN_SPLITS", "0")
    )
    cross_pibar_row_stats_enabled = (
        os.environ.get("GPUREC_CROSS_PIBAR_ROW_STATS", "0") != "0"
    )
    kernelized_active_mask_enabled = (
        os.environ.get("GPUREC_KERNELIZED_ACTIVE_MASK", "1") != "0"
        and _HAS_FUSED_BACKWARD
        and pibar_mode == 'uniform'
        and device.type == 'cuda'
        and dtype in (torch.float32, torch.float64)
    )
    kernelized_backward_dts_enabled = (
        os.environ.get("GPUREC_KERNELIZED_BACKWARD_DTS", "1") != "0"
        and device.type == 'cuda'
    )
    dts_parent_reduced_env = os.environ.get("GPUREC_BACKWARD_PARENT_REDUCED_DTS")
    if dts_parent_reduced_env is None:
        dts_parent_reduced_env = os.environ.get("GPUREC_DTS_PARENT_REDUCED")
    if dts_parent_reduced_env is None:
        dts_parent_reduced_env = "tiled"
    parent_reduced_backward_dts_enabled = (
        dts_parent_reduced_env.strip().lower()
        not in ("", "0", "false", "off", "no")
        and kernelized_backward_dts_enabled
        and pibar_mode == 'uniform'
        and device.type == 'cuda'
        and dtype in (torch.float32, torch.float64)
    )
    parent_reduced_backward_dts_min_splits = int(
        os.environ.get(
            "GPUREC_BACKWARD_PARENT_REDUCED_DTS_MIN_SPLITS",
            os.environ.get("GPUREC_DTS_PARENT_REDUCED_MIN_SPLITS", "8192"),
        )
    )
    parent_reduced_backward_dts_impl = dts_parent_reduced_env.strip().lower()
    if parent_reduced_backward_dts_impl in ("1", "true", "yes", "on"):
        parent_reduced_backward_dts_impl = os.environ.get(
            "GPUREC_BACKWARD_PARENT_REDUCED_DTS_IMPL",
            os.environ.get("GPUREC_DTS_PARENT_REDUCED_IMPL", "tiled"),
        ).strip().lower()
    parent_reduced_backward_dts_tile_splits = int(
        os.environ.get(
            "GPUREC_BACKWARD_PARENT_REDUCED_DTS_TILE_SPLITS",
            os.environ.get("GPUREC_DTS_PARENT_REDUCED_TILE_SPLITS", "64"),
        )
    )
    fused_dts_backward_accum_enabled = (
        os.environ.get("GPUREC_FUSED_DTS_BACKWARD_ACCUM", "1") != "0"
    )
    dts_backward_accum_impl = os.environ.get(
        "GPUREC_DTS_BACKWARD_ACCUM_IMPL", "direct"
    ).lower()
    parent_tiled_dts_backward_accum_enabled = (
        os.environ.get("GPUREC_PARENT_TILED_DTS_BACKWARD_ACCUM", "0") != "0"
        or dts_backward_accum_impl in (
            "parent_tiled",
            "parent_tiled_all",
            "parent-tiled",
            "parent-tiled-all",
        )
    )
    parent_ragged_dts_backward_accum_enabled = (
        os.environ.get("GPUREC_PARENT_RAGGED_DTS_BACKWARD_ACCUM", "0") != "0"
        or dts_backward_accum_impl in (
            "parent_ragged",
            "parent_ragged_all",
            "parent-ragged",
            "parent-ragged-all",
        )
    )
    grouped_dts_backward_accum_enabled = (
        os.environ.get("GPUREC_GROUPED_DTS_BACKWARD_ACCUM", "0") != "0"
        or dts_backward_accum_impl in (
            "grouped",
            "grouped_all",
            "child_grouped",
            "child_grouped_all",
        )
    )
    noatomic_dts_backward_accum_enabled = (
        os.environ.get("GPUREC_NOATOMIC_DTS_BACKWARD_ACCUM", "0") != "0"
        or dts_backward_accum_impl in (
            "noatomic",
            "noatomic_all",
            "nonatomic",
            "nonatomic_all",
        )
    )
    merged_dts_backward_accum_enabled = (
        os.environ.get("GPUREC_MERGED_DTS_BACKWARD_ACCUM", "1") != "0"
        or dts_backward_accum_impl in (
            "merged",
            "merged_s",
            "merge_s",
            "merged_s_term",
        )
    )
    grouped_dts_backward_accum_all = dts_backward_accum_impl in (
        "grouped_all",
        "child_grouped_all",
    )
    noatomic_dts_backward_accum_all = dts_backward_accum_impl in (
        "noatomic_all",
        "nonatomic_all",
    )
    parent_tiled_dts_backward_accum_all = dts_backward_accum_impl in (
        "parent_tiled_all",
        "parent-tiled-all",
    )
    parent_ragged_dts_backward_accum_all = dts_backward_accum_impl in (
        "parent_ragged_all",
        "parent-ragged-all",
    )
    grouped_dts_backward_min_splits = int(
        os.environ.get("GPUREC_GROUPED_DTS_BACKWARD_MIN_SPLITS", "8192")
    )
    grouped_dts_backward_min_fanout = float(
        os.environ.get("GPUREC_GROUPED_DTS_BACKWARD_MIN_FANOUT", "64")
    )
    parent_tiled_dts_backward_min_splits = int(
        os.environ.get("GPUREC_PARENT_TILED_DTS_BACKWARD_MIN_SPLITS", "8192")
    )
    parent_tiled_dts_backward_min_fanout = float(
        os.environ.get("GPUREC_PARENT_TILED_DTS_BACKWARD_MIN_FANOUT", "64")
    )
    parent_tiled_dts_backward_tile_splits = int(
        os.environ.get("GPUREC_PARENT_TILED_DTS_BACKWARD_TILE_SPLITS", "16")
    )
    parent_ragged_dts_backward_tile_splits = int(
        os.environ.get(
            "GPUREC_PARENT_RAGGED_DTS_BACKWARD_TILE_SPLITS",
            str(parent_tiled_dts_backward_tile_splits),
        )
    )
    fused_uniform_backward_enabled = (
        os.environ.get("GPUREC_FUSED_UNIFORM_BACKWARD", "1") != "0"
    )
    cuda_self_loop_nosplit_enabled = (
        os.environ.get(
            "GPUREC_CUDA_SELF_LOOP_NOSPLIT",
            os.environ.get("GPUREC_CUDA_WAVE_NOSPLIT", "0"),
        )
        != "0"
    )
    cuda_self_loop_nosplit_correction = os.environ.get(
        "GPUREC_CUDA_SELF_LOOP_NOSPLIT_CORRECTION", "self"
    )
    fused_uniform_backward_view_rhs = (
        os.environ.get("GPUREC_FUSED_UNIFORM_BACKWARD_VIEW_RHS", "1") != "0"
    )
    skip_inactive_zero_stores_enabled = (
        os.environ.get("GPUREC_BACKWARD_SKIP_INACTIVE_ZERO_STORES", "0") != "0"
    )
    backward_pruning_row_stats_enabled = (
        os.environ.get("GPUREC_BACKWARD_PRUNING_ROW_STATS", "0") != "0"
    )
    hybrid_row_pruning_env = os.environ.get("GPUREC_BACKWARD_HYBRID_ROW_PRUNING")
    if hybrid_row_pruning_env is None:
        hybrid_row_pruning_env = os.environ.get("GPUREC_BACKWARD_HYBRID_ROW_MASK")
    if hybrid_row_pruning_env is None:
        hybrid_row_pruning_env = os.environ.get("GPUREC_FUSED_ROW_ACTIVE_MASK")
    if hybrid_row_pruning_env is None:
        hybrid_row_pruning_env = "1"
    hybrid_row_pruning_enabled = (
        (hybrid_row_pruning_env != "0")
        and _HAS_FUSED_BACKWARD
        and pibar_mode == 'uniform'
        and device.type == 'cuda'
        and dtype in (torch.float32, torch.float64)
    )
    hybrid_row_pruning_targets = os.environ.get(
        "GPUREC_BACKWARD_HYBRID_ROW_PRUNING_TARGETS",
        os.environ.get("GPUREC_BACKWARD_ROW_MASK_TARGETS", "all"),
    ).strip().lower()
    if hybrid_row_pruning_targets in ("", "1", "true", "yes", "on"):
        hybrid_row_pruning_targets = "all"
    hybrid_prune_self = hybrid_row_pruning_targets in ("all", "self", "wave")
    hybrid_prune_splits = hybrid_row_pruning_targets in (
        "all",
        "split",
        "splits",
        "dts",
        "dts_pibar",
        "pibar",
    )
    hybrid_row_pruning_require_partial = (
        os.environ.get("GPUREC_BACKWARD_HYBRID_ROW_PRUNING_REQUIRE_PARTIAL", "0")
        != "0"
    )
    hybrid_row_pruning_min_inactive_frac = float(
        os.environ.get("GPUREC_BACKWARD_HYBRID_ROW_PRUNING_MIN_INACTIVE_FRAC", "0")
    )
    family_chunk_pruning_diag_enabled = (
        os.environ.get(
            "GPUREC_BACKWARD_FAMILY_CHUNK_DIAG",
            os.environ.get("GPUREC_BACKWARD_FAMILY_PRUNING_DIAG", "0"),
        )
        != "0"
    )
    family_chunk_rows = 256
    if family_chunk_pruning_diag_enabled:
        family_chunk_rows = max(
            1,
            int(
                os.environ.get(
                    "GPUREC_BACKWARD_FAMILY_CHUNK_ROWS",
                    os.environ.get("GPUREC_BACKWARD_FAMILY_CHUNK_SIZE", "256"),
                )
            ),
        )
    wave_topology_int32_enabled = (
        os.environ.get("GPUREC_WAVE_TOPOLOGY_INT32", "1") != "0"
        and device.type == 'cuda'
        and dtype in (torch.float32, torch.float64)
    )
    dts_reduction_accum_impl = os.environ.get(
        "GPUREC_DTS_BACKWARD_REDUCTION_ACCUM", "scalar"
    ).strip().lower()
    dts_reduction_accum_scalar_enabled = dts_reduction_accum_impl in (
        "1",
        "true",
        "yes",
        "on",
        "scalar",
        "scalars",
        "all",
        "full",
    )
    dts_reduction_accum_mt_enabled = dts_reduction_accum_impl in (
        "mt",
        "grad_mt",
        "all",
        "full",
    )
    dts_reduction_accum_min_splits = int(
        os.environ.get("GPUREC_DTS_BACKWARD_REDUCTION_ACCUM_MIN_SPLITS", "8192")
    )
    dts_grad_mt_two_stage_enabled = (
        os.environ.get("GPUREC_DTS_GRAD_MT_TWO_STAGE", "0") != "0"
    )
    dts_grad_mt_two_stage_tile_splits = int(
        os.environ.get("GPUREC_DTS_GRAD_MT_TWO_STAGE_TILE_SPLITS", "128")
    )
    _compute_dts_cross_kernelized = None
    if kernelized_backward_dts_enabled:
        from .forward import _compute_dts_cross as _compute_dts_cross_kernelized

    ancestor_cols = None
    level_parents = None
    compact_level_ptr = None
    compact_level_parents = None
    compact_level_child1 = None
    compact_level_child2 = None
    subtree_interval_start = None
    subtree_interval_end = None
    sp_parent = None
    max_ancestor_depth = None
    depth_nodes = None
    if fused_cross_pibar_vjp_enabled:
        cache = species_helpers.get('_wave_forward_species_cache')
        if cache is not None and int(cache.get('S', -1)) == int(S):
            cached_max_ancestor_depth = cache.get('max_ancestor_depth')
            if cached_max_ancestor_depth is not None:
                max_ancestor_depth = int(cached_max_ancestor_depth)
            cached_ancestor_cols = cache.get('ancestor_cols')
            if torch.is_tensor(cached_ancestor_cols) and cached_ancestor_cols.device == target_device:
                ancestor_cols = cached_ancestor_cols
            cached_level_parents = cache.get('level_parents')
            if torch.is_tensor(cached_level_parents) and cached_level_parents.device == target_device:
                level_parents = cached_level_parents
            cached_compact_level_ptr = cache.get('compact_level_ptr')
            cached_compact_level_parents = cache.get('compact_level_parents')
            cached_compact_level_child1 = cache.get('compact_level_child1')
            cached_compact_level_child2 = cache.get('compact_level_child2')
            if (
                torch.is_tensor(cached_compact_level_ptr)
                and torch.is_tensor(cached_compact_level_parents)
                and torch.is_tensor(cached_compact_level_child1)
                and torch.is_tensor(cached_compact_level_child2)
                and cached_compact_level_ptr.device == target_device
                and cached_compact_level_parents.device == target_device
                and cached_compact_level_child1.device == target_device
                and cached_compact_level_child2.device == target_device
            ):
                compact_level_ptr = cached_compact_level_ptr
                compact_level_parents = cached_compact_level_parents
                compact_level_child1 = cached_compact_level_child1
                compact_level_child2 = cached_compact_level_child2
            cached_subtree_interval_start = cache.get('subtree_interval_start')
            cached_subtree_interval_end = cache.get('subtree_interval_end')
            if (
                torch.is_tensor(cached_subtree_interval_start)
                and torch.is_tensor(cached_subtree_interval_end)
                and cached_subtree_interval_start.device == target_device
                and cached_subtree_interval_end.device == target_device
            ):
                subtree_interval_start = cached_subtree_interval_start
                subtree_interval_end = cached_subtree_interval_end
            cached_sp_parent = cache.get('sp_parent')
            if torch.is_tensor(cached_sp_parent) and cached_sp_parent.device == target_device:
                sp_parent = cached_sp_parent
            cached_depth_nodes = cache.get('depth_nodes')
            if torch.is_tensor(cached_depth_nodes) and cached_depth_nodes.device == target_device:
                depth_nodes = cached_depth_nodes

        if (
            ancestor_cols is None
            or (tree_cross_pibar_vjp_impl and level_parents is None)
            or (
                dts_pibar_ud_compact_levels_enabled
                and tree_cross_pibar_vjp_impl
                and (
                    compact_level_ptr is None
                    or compact_level_parents is None
                    or compact_level_child1 is None
                    or compact_level_child2 is None
                )
            )
            or (
                dts_pibar_ud_euler_prefix_enabled
                and (
                    subtree_interval_start is None
                    or subtree_interval_end is None
                )
            )
            or (prefix_cross_pibar_vjp_impl and (sp_parent is None or depth_nodes is None))
        ):
            if p_cpu is None or c_cpu is None or mask_c1 is None:
                sp_P_idx = species_helpers['s_P_indexes']
                sp_c12_idx = species_helpers['s_C12_indexes']
                p_cpu = sp_P_idx.cpu().long()
                c_cpu = sp_c12_idx.cpu().long()
                mask_c1 = p_cpu < S
            sp_parent_cpu = torch.full((S,), -1, dtype=torch.long)
            sp_parent_cpu[c_cpu[mask_c1]] = p_cpu[mask_c1]
            sp_parent_cpu[c_cpu[~mask_c1]] = p_cpu[~mask_c1] - S
            if sp_parent is None:
                sp_parent = sp_parent_cpu.to(target_device)

            parent_values = sp_parent_cpu.tolist()
            ancestor_lists = []
            max_ancestor_depth = 0
            for s_idx in range(S):
                cur = s_idx
                depth = 0
                ancestors = []
                while cur >= 0:
                    ancestors.append(cur)
                    depth += 1
                    if depth > S:
                        raise RuntimeError("Cycle detected in species parent pointers")
                    cur = parent_values[cur]
                ancestor_lists.append(ancestors)
                max_ancestor_depth = max(max_ancestor_depth, depth)

            ancestor_cols_cpu = torch.full((S, max_ancestor_depth), -1, dtype=torch.long)
            for s_idx, ancestors in enumerate(ancestor_lists):
                ancestor_cols_cpu[s_idx, :len(ancestors)] = torch.tensor(ancestors, dtype=torch.long)
            if ancestor_cols is None:
                ancestor_cols = ancestor_cols_cpu.T.contiguous().to(target_device)
            max_ancestor_depth = int(max_ancestor_depth)

            need_compact_levels = (
                dts_pibar_ud_compact_levels_enabled
                and tree_cross_pibar_vjp_impl
                and (
                    compact_level_ptr is None
                    or compact_level_parents is None
                    or compact_level_child1 is None
                    or compact_level_child2 is None
                )
            )
            if tree_cross_pibar_vjp_impl and (level_parents is None or need_compact_levels):
                child1_values = sp_child1_cpu.tolist()
                child2_values = sp_child2_cpu.tolist()
                levels = [-1] * S

                for s_idx in range(S):
                    if levels[s_idx] >= 0:
                        continue
                    stack = [(s_idx, False)]
                    while stack:
                        node, expanded = stack.pop()
                        if levels[node] >= 0:
                            continue
                        c1 = child1_values[node]
                        c2 = child2_values[node]
                        if not expanded:
                            stack.append((node, True))
                            if c2 < S and levels[c2] < 0:
                                stack.append((c2, False))
                            if c1 < S and levels[c1] < 0:
                                stack.append((c1, False))
                            continue
                        child_levels = []
                        if c1 < S:
                            child_levels.append(levels[c1])
                        if c2 < S:
                            child_levels.append(levels[c2])
                        levels[node] = (max(child_levels) + 1) if child_levels else 0

                max_level = max(levels) if levels else 0
                level_lists = []
                max_level_width = 1
                for level in range(1, max_level + 1):
                    parents = [
                        s_idx for s_idx, node_level in enumerate(levels)
                        if node_level == level
                        and (child1_values[s_idx] < S or child2_values[s_idx] < S)
                    ]
                    if parents:
                        level_lists.append(parents)
                        max_level_width = max(max_level_width, len(parents))

                if level_parents is None:
                    level_parents_cpu = torch.full(
                        (max(len(level_lists), 1), max_level_width),
                        -1,
                        dtype=torch.long,
                    )
                    for level, parents in enumerate(level_lists):
                        level_parents_cpu[level, :len(parents)] = torch.tensor(parents, dtype=torch.long)
                    level_parents = level_parents_cpu.contiguous().to(target_device)
                if need_compact_levels:
                    ptr_values = [0]
                    flat_parents = []
                    flat_child1 = []
                    flat_child2 = []
                    for parents in level_lists:
                        flat_parents.extend(parents)
                        flat_child1.extend(child1_values[parent] for parent in parents)
                        flat_child2.extend(child2_values[parent] for parent in parents)
                        ptr_values.append(len(flat_parents))
                    if len(ptr_values) == 1:
                        ptr_values.append(0)
                    compact_level_ptr_cpu = torch.tensor(ptr_values, dtype=torch.long)
                    compact_level_parents_cpu = torch.tensor(flat_parents, dtype=torch.int32)
                    compact_level_child1_cpu = torch.tensor(flat_child1, dtype=torch.int32)
                    compact_level_child2_cpu = torch.tensor(flat_child2, dtype=torch.int32)
                    compact_level_ptr = compact_level_ptr_cpu.contiguous().to(target_device)
                    compact_level_parents = compact_level_parents_cpu.contiguous().to(target_device)
                    compact_level_child1 = compact_level_child1_cpu.contiguous().to(target_device)
                    compact_level_child2 = compact_level_child2_cpu.contiguous().to(target_device)

            if (
                dts_pibar_ud_euler_prefix_enabled
                and (
                    subtree_interval_start is None
                    or subtree_interval_end is None
                )
            ):
                from .species_euler_layout import species_euler_layout_report

                report = species_euler_layout_report(
                    sp_child1=sp_child1_cpu,
                    sp_child2=sp_child2_cpu,
                    S=S,
                )
                if not report.all_subtrees_contiguous:
                    raise RuntimeError(
                        "GPUREC_DTS_PIBAR_UD_EULER_PREFIX requires every "
                        "species subtree to be a contiguous current-order "
                        "interval"
                    )
                subtree_interval_start_cpu = torch.tensor(
                    report.current_interval_start,
                    dtype=torch.int32,
                )
                subtree_interval_end_cpu = torch.tensor(
                    report.current_interval_end,
                    dtype=torch.int32,
                )
                subtree_interval_start = subtree_interval_start_cpu.contiguous().to(target_device)
                subtree_interval_end = subtree_interval_end_cpu.contiguous().to(target_device)

            if prefix_cross_pibar_vjp_impl and depth_nodes is None:
                depths = [-1] * S

                def _species_depth(s_idx):
                    cached_depth = depths[s_idx]
                    if cached_depth >= 0:
                        return cached_depth
                    parent = parent_values[s_idx]
                    depth = 0 if parent < 0 else _species_depth(parent) + 1
                    depths[s_idx] = depth
                    return depth

                for s_idx in range(S):
                    _species_depth(s_idx)
                max_depth = max(depths) if depths else 0
                depth_lists = [
                    [s_idx for s_idx, depth in enumerate(depths) if depth == level]
                    for level in range(max_depth + 1)
                ]
                max_depth_width = max((len(nodes) for nodes in depth_lists), default=1)
                depth_nodes_cpu = torch.full(
                    (max(len(depth_lists), 1), max_depth_width),
                    -1,
                    dtype=torch.long,
                )
                for level, nodes in enumerate(depth_lists):
                    if nodes:
                        depth_nodes_cpu[level, :len(nodes)] = torch.tensor(nodes, dtype=torch.long)
                depth_nodes = depth_nodes_cpu.contiguous().to(target_device)

            if cache is not None and int(cache.get('S', -1)) == int(S):
                if level_parents is not None:
                    cache['level_parents'] = level_parents
                if (
                    compact_level_ptr is not None
                    and compact_level_parents is not None
                    and compact_level_child1 is not None
                    and compact_level_child2 is not None
                ):
                    cache['compact_level_ptr'] = compact_level_ptr
                    cache['compact_level_parents'] = compact_level_parents
                    cache['compact_level_child1'] = compact_level_child1
                    cache['compact_level_child2'] = compact_level_child2
                if subtree_interval_start is not None and subtree_interval_end is not None:
                    cache['subtree_interval_start'] = subtree_interval_start
                    cache['subtree_interval_end'] = subtree_interval_end
                if sp_parent is not None:
                    cache['sp_parent'] = sp_parent
                if max_ancestor_depth is not None:
                    cache['max_ancestor_depth'] = int(max_ancestor_depth)
                if depth_nodes is not None:
                    cache['depth_nodes'] = depth_nodes

        if max_ancestor_depth is None and torch.is_tensor(ancestor_cols):
            max_ancestor_depth = int(ancestor_cols.shape[0])

    sp_parent_wave = (
        sp_parent.to(dtype=torch.int32).contiguous()
        if wave_topology_int32_enabled and torch.is_tensor(sp_parent)
        else sp_parent
    )

    # Auto-wrap single-family inputs into batched format (G=1).
    _auto_wrapped = family_idx is None
    if _auto_wrapped:
        family_idx = torch.zeros(C, dtype=torch.long, device=device)
        E = E.unsqueeze(0)
        Ebar = Ebar.unsqueeze(0)
        E_s1 = E_s1.unsqueeze(0)
        E_s2 = E_s2.unsqueeze(0)
        log_pS = log_pS.unsqueeze(0)
        log_pD = log_pD.unsqueeze(0)
        log_pL = log_pL.unsqueeze(0)
        max_transfer_mat = max_transfer_mat.unsqueeze(0)
    else:
        family_idx = family_idx.to(device=device, dtype=torch.long).contiguous()

    family_chunk_diag_wave_info = None
    family_chunk_diag_values = None
    family_chunk_diag_family_count = 1
    if family_chunk_pruning_diag_enabled:
        diag_family_idx = None
        layout_family_idx = wave_layout.get('family_idx')
        if torch.is_tensor(layout_family_idx) and int(layout_family_idx.numel()) == C:
            diag_family_idx = layout_family_idx.to(device=device, dtype=torch.long)
        elif torch.is_tensor(family_idx) and int(family_idx.numel()) == C:
            diag_family_idx = family_idx.to(device=device, dtype=torch.long)
        else:
            diag_family_idx = torch.zeros(C, dtype=torch.long, device=device)

        diag_family_cpu = diag_family_idx.detach().cpu().tolist()
        family_chunk_diag_family_count = (
            max((int(f) for f in diag_family_cpu), default=0) + 1
        )
        family_chunk_diag_wave_info = []
        for meta in wave_metas:
            ws_i = int(meta['start'])
            W_i = int(meta['W'])
            chunk_ids_cpu = torch.empty(W_i, dtype=torch.long)
            chunk_sizes = []
            chunk_families = []
            start_i = 0
            chunk_i = 0
            while start_i < W_i:
                fam_i = int(diag_family_cpu[ws_i + start_i])
                end_i = start_i + 1
                while (
                    end_i < W_i
                    and int(diag_family_cpu[ws_i + end_i]) == fam_i
                    and (end_i - start_i) < family_chunk_rows
                ):
                    end_i += 1
                chunk_ids_cpu[start_i:end_i] = chunk_i
                chunk_sizes.append(end_i - start_i)
                chunk_families.append(fam_i)
                start_i = end_i
                chunk_i += 1

            family_chunk_diag_wave_info.append({
                'chunk_ids': chunk_ids_cpu.to(device=device),
                'chunk_sizes': torch.tensor(chunk_sizes, dtype=torch.long, device=device),
                'chunk_families': torch.tensor(chunk_families, dtype=torch.long, device=device),
                'n_chunks': int(chunk_i),
                'n_family_slots': len(set(chunk_families)),
            })

        family_chunk_diag_values = {
            'waves': 0,
            'rows_total': [],
            'rows_current': [],
            'rows_active': [],
            'rows_chunk_scheduled': [],
            'chunks_total': [],
            'chunks_active': [],
            'family_slots_total': [],
            'family_slots_active': [],
            'splits_current': [],
            'splits_chunk_scheduled': [],
            'splits_active_parent': [],
        }

    mt_squeezed = max_transfer_mat.squeeze(-1) if max_transfer_mat.ndim > 2 else max_transfer_mat

    transfer_mat_T = None
    if pibar_mode in ('dense', 'topk') and transfer_mat is not None:
        transfer_mat_T = transfer_mat.T.contiguous()

    G = log_pD.shape[0]

    def _family_param_shape_ok(p):
        return (
            torch.is_tensor(p)
            and p.ndim in (1, 2)
            and int(p.shape[0]) == int(G)
            and (p.ndim == 1 or int(p.shape[1]) in (1, S))
        )

    def _family_species_shape_ok(p):
        return (
            torch.is_tensor(p)
            and p.ndim == 2
            and int(p.shape[0]) == int(G)
            and int(p.shape[1]) == int(S)
        )

    fused_genewise_self_loop_env = os.environ.get(
        "GPUREC_FUSED_GENEWISE_BACKWARD_SELF_LOOP",
        os.environ.get("GPUREC_FUSED_GENEWISE_BACKWARD", "1"),
    ).strip().lower()
    fused_genewise_self_loop_requested = fused_genewise_self_loop_env not in (
        "", "0", "false", "off", "no"
    )
    family_indexed_self_loop_supported = (
        (not _auto_wrapped)
        and fused_genewise_self_loop_requested
        and _family_species_shape_ok(E)
        and _family_species_shape_ok(Ebar)
        and _family_species_shape_ok(E_s1)
        and _family_species_shape_ok(E_s2)
        and _family_species_shape_ok(mt_squeezed)
        and _family_param_shape_ok(log_pD)
        and _family_param_shape_ok(log_pS)
    )
    can_use_fused_uniform_backward = (
        fused_uniform_backward_enabled
        and _HAS_FUSED_BACKWARD
        and (_auto_wrapped or family_indexed_self_loop_supported)
        and pibar_mode == 'uniform'
        and dtype in (torch.float32, torch.float64)
        and device.type == 'cuda'
        and S > 256
    )
    scratch_pool_requested = (
        os.environ.get("GPUREC_BACKWARD_SCRATCH_POOL", "0") != "0"
    )
    scratch_pool_scope_env = os.environ.get(
        "GPUREC_BACKWARD_SCRATCH_POOL_SCOPES",
        os.environ.get("GPUREC_BACKWARD_SCRATCH_POOL_SCOPE", "all"),
    ).strip().lower()
    if scratch_pool_scope_env in ("", "1", "true", "yes", "on"):
        scratch_pool_scope_env = "all"
    scratch_pool_scopes = {
        scope.strip()
        for scope in scratch_pool_scope_env.replace(";", ",").split(",")
        if scope.strip()
    }
    scratch_pool_all_scopes = (
        "all" in scratch_pool_scopes or "*" in scratch_pool_scopes
    )

    def _scratch_scope_enabled(*names):
        return scratch_pool_all_scopes or any(name in scratch_pool_scopes for name in names)

    scratch_pool_enabled = (
        scratch_pool_requested
        and can_use_fused_uniform_backward
        and _auto_wrapped
        and "none" not in scratch_pool_scopes
    )
    scratch_pool = None
    if scratch_pool_enabled:
        max_wave_W = max((int(meta.get('W', 0)) for meta in wave_metas), default=0)
        max_dts_splits = max(
            (
                int(meta['sl'].numel())
                for meta in wave_metas
                if meta.get('has_splits') and torch.is_tensor(meta.get('sl'))
            ),
            default=0,
        )
        scratch_pool = {}
        if max_wave_W > 0 and _scratch_scope_enabled("self", "wave", "self_loop"):
            scratch_pool["wave"] = {
                name: torch.empty((max_wave_W, S), device=device, dtype=dtype)
                for name in (
                    "v_k",
                    "aw0",
                    "aw1",
                    "aw345",
                    "aw4",
                    "spec_buf",
                    "term_buf",
                    "pibar_corr",
                )
            }
        dts_scratch = {}
        if max_dts_splits > 0 and _scratch_scope_enabled("dts", "dts_ud", "pibar_ud"):
            dts_scratch["pibar_ud"] = torch.empty(
                (2 * max_dts_splits, S), device=device, dtype=dtype
            )
            dts_scratch["pibar_A"] = torch.empty(
                (2 * max_dts_splits,), device=device, dtype=dtype
            )
            dts_scratch["pibar_side_active"] = torch.empty(
                (2 * max_dts_splits,), device=device, dtype=torch.bool
            )
        if (
            max_dts_splits > 0
            and dts_grad_mt_two_stage_enabled
            and _scratch_scope_enabled("dts", "grad_mt", "dts_grad_mt")
        ):
            mt_tile_splits = max(1, int(dts_grad_mt_two_stage_tile_splits))
            max_grad_mt_tiles = (max_dts_splits + mt_tile_splits - 1) // mt_tile_splits
            dts_scratch["grad_mt_partial"] = torch.empty(
                (max_grad_mt_tiles, S), device=device, dtype=dtype
            )
        if dts_scratch:
            scratch_pool["dts"] = dts_scratch
        if not scratch_pool:
            scratch_pool = None

    # Shared/global mode has one parameter/E row for every clade.  Keeping the
    # constants as [S] avoids materializing several [C, S] copies before the
    # wave loop, which is both slow and the main source of backward OOMs.
    if _auto_wrapped:
        mt_shared = mt_squeezed[0]
        E_shared = E[0]
        Ebar_shared = Ebar[0]
        E_s1_shared = E_s1[0]
        E_s2_shared = E_s2[0]
        log_pD_shared = log_pD[0]
        log_pS_shared = log_pS[0]
        DL_shared = 1.0 + log_pD_shared + E_shared
        SL1_shared = log_pS_shared + E_s2_shared
        SL2_shared = log_pS_shared + E_s1_shared
        mt_family = E_family = Ebar_family = None
        DL_family = SL1_family = SL2_family = None
    else:
        mt_shared = E_shared = Ebar_shared = E_s1_shared = E_s2_shared = None
        log_pD_shared = log_pS_shared = None
        DL_shared = SL1_shared = SL2_shared = None
        mt_family = mt_squeezed.contiguous()
        E_family = E.contiguous()
        Ebar_family = Ebar.contiguous()
        _pD_family = log_pD.unsqueeze(-1) if log_pD.ndim == 1 else log_pD
        _pS_family = log_pS.unsqueeze(-1) if log_pS.ndim == 1 else log_pS
        DL_family = (1.0 + _pD_family + E).contiguous()
        SL1_family = (_pS_family + E_s2).contiguous()
        SL2_family = (_pS_family + E_s1).contiguous()

    def _wave_consts(ws, we, *, family_indexed):
        """Return constants for the current self-loop wave."""
        if _auto_wrapped:
            return mt_shared, DL_shared, E_shared, Ebar_shared, SL1_shared, SL2_shared
        if family_indexed:
            return mt_family, DL_family, E_family, Ebar_family, SL1_family, SL2_family
        fi_w = family_idx[ws:we]
        return (
            mt_family[fi_w],
            DL_family[fi_w],
            E_family[fi_w],
            Ebar_family[fi_w],
            SL1_family[fi_w],
            SL2_family[fi_w],
        )

    leaf_row_index = wave_layout['leaf_row_index']
    leaf_col_index = wave_layout['leaf_col_index']
    leaf_species_index = wave_layout.get('leaf_species_index')

    use_uniform_leaf_index = bool(
        os.environ.get("GPUREC_BACKWARD_LEAF_INDEX", "1") != "0"
        and can_use_fused_uniform_backward
        and pibar_mode == 'uniform'
        and device.type == 'cuda'
        and leaf_species_index is not None
    )
    uniform_leaf_logp = None
    if use_uniform_leaf_index:
        if _auto_wrapped:
            use_scalar_leaf_logp = (
                os.environ.get("GPUREC_SCALAR_LEAF_LOGP", "0") != "0"
                and log_pS_shared.ndim == 0
            )
            uniform_leaf_logp = (
                log_pS_shared.contiguous()
                if use_scalar_leaf_logp
                else (
                    log_pS_shared.expand(S).contiguous()
                    if log_pS_shared.ndim == 0
                    else log_pS_shared.contiguous()
                )
            )
        else:
            uniform_leaf_logp = log_pS.contiguous()
    fused_wave_param_accum_enabled = (
        os.environ.get("GPUREC_FUSED_WAVE_PARAM_ACCUM", "1") != "0"
    )
    no_cpu_pruning = (
        os.environ.get("GPUREC_BACKWARD_NO_CPU_PRUNING", "0") != "0"
    )
    device_pruning_requested = (
        os.environ.get("GPUREC_DEVICE_PRUNING", "0") != "0"
    )

    if wave_topology_int32_enabled:
        cached_child1_i32 = None
        cached_child2_i32 = None
        if species_cache is not None and int(species_cache.get('S', -1)) == int(S):
            cached_child1_i32 = species_cache.get('sp_child1_int32')
            cached_child2_i32 = species_cache.get('sp_child2_int32')
        if (
            torch.is_tensor(cached_child1_i32)
            and torch.is_tensor(cached_child2_i32)
            and cached_child1_i32.device == target_device
            and cached_child2_i32.device == target_device
        ):
            sp_child1_wave = cached_child1_i32
            sp_child2_wave = cached_child2_i32
        else:
            sp_child1_wave = sp_child1_cpu.to(device=target_device, dtype=torch.int32)
            sp_child2_wave = sp_child2_cpu.to(device=target_device, dtype=torch.int32)
            if species_cache is not None and int(species_cache.get('S', -1)) == int(S):
                species_cache['sp_child1_int32'] = sp_child1_wave
                species_cache['sp_child2_int32'] = sp_child2_wave
        leaf_species_index_wave = (
            leaf_species_index.to(device=device, dtype=torch.int32).contiguous()
            if torch.is_tensor(leaf_species_index)
            else leaf_species_index
        )
    else:
        sp_child1_wave = sp_child1
        sp_child2_wave = sp_child2
        leaf_species_index_wave = leaf_species_index
    dense_leaf_mask_from_index = (
        os.environ.get("GPUREC_DENSE_LEAF_MASK_FROM_INDEX", "0") != "0"
        and leaf_species_index is not None
        and not (can_use_fused_uniform_backward and use_uniform_leaf_index)
    )
    leaf_species_lanes = (
        torch.arange(S, device=device)
        if dense_leaf_mask_from_index else None
    )
    leaf_zero = (
        torch.tensor(0.0, device=device, dtype=dtype)
        if dense_leaf_mask_from_index else None
    )
    leaf_neg_inf = (
        torch.tensor(NEG_INF, device=device, dtype=dtype)
        if dense_leaf_mask_from_index else None
    )

    def _get_leaf_mask(ws, we):
        W = we - ws
        if dense_leaf_mask_from_index:
            return torch.where(
                leaf_species_index[ws:we].unsqueeze(1) == leaf_species_lanes.unsqueeze(0),
                leaf_zero,
                leaf_neg_inf,
            )

        lwt = torch.full((W, S), NEG_INF, device=device, dtype=dtype)
        mask = (leaf_row_index >= ws) & (leaf_row_index < we)
        if mask.any():
            lwt[leaf_row_index[mask] - ws, leaf_col_index[mask]] = 0.0
        return lwt

    def _get_leaf_wt(ws, we):
        leaf_mask = _get_leaf_mask(ws, we)
        if _auto_wrapped:
            return log_pS_shared + leaf_mask
        log_pS_w = log_pS[family_idx[ws:we]]
        if log_pS_w.ndim == 1:
            log_pS_w = log_pS_w.unsqueeze(-1)
        return log_pS_w + leaf_mask

    n_waves_total = K
    n_waves_skipped = 0
    n_clades_total = C
    n_clades_skipped = 0
    device_pruning_clades_total = 0
    device_pruning_waves_total = 0
    device_pruning_active_counts = []
    device_pruning_wave_active_flags = []

    accumulated_rhs = torch.zeros(C, S, device=device, dtype=dtype)
    for r in root_clade_ids_perm:
        r = int(r)
        root_Pi = Pi_star_wave[r]
        lse = logsumexp2(root_Pi, dim=0)
        accumulated_rhs[r] = -_safe_exp2_ratio(root_Pi, lse)

    grad_log_pD = torch.zeros_like(log_pD)
    grad_log_pS = torch.zeros_like(log_pS)
    grad_mt = torch.zeros_like(mt_squeezed)
    grad_E_acc = torch.zeros_like(E)
    grad_Ebar_acc = torch.zeros_like(Ebar)
    grad_transfer_mat_acc = torch.zeros(S, S, device=device, dtype=dtype) if pibar_mode in ('dense', 'topk') else None
    grad_E_s1_acc = torch.zeros_like(E_s1)
    grad_E_s2_acc = torch.zeros_like(E_s2)
    cross_pibar_row_stats = None
    forward_pibar_row_max = None
    reuse_forward_pibar_stats_enabled = (
        os.environ.get("GPUREC_REUSE_FORWARD_PIBAR_STATS", "0") != "0"
        and uniform_pibar_row_max is not None
        and torch.is_tensor(uniform_pibar_row_max)
        and uniform_pibar_row_max.numel() == C
        and fused_cross_pibar_vjp_enabled
        and pibar_mode == 'uniform'
        and _auto_wrapped
        and mt_shared is not None
        and mt_shared.ndim == 1
        and dtype in (torch.float32, torch.float64)
        and device.type == 'cuda'
        and S > 256
    )
    if reuse_forward_pibar_stats_enabled:
        forward_pibar_row_max = uniform_pibar_row_max.contiguous()
    elif (
        dts_pibar_ud_fusion_enabled
        and uniform_pibar_row_max is not None
        and torch.is_tensor(uniform_pibar_row_max)
        and uniform_pibar_row_max.numel() == C
        and pibar_mode == 'uniform'
        and device.type == 'cuda'
        and dtype in (torch.float32, torch.float64)
    ):
        forward_pibar_row_max = uniform_pibar_row_max.contiguous()
    if (
        cross_pibar_row_stats_enabled
        and not reuse_forward_pibar_stats_enabled
        and fused_cross_pibar_vjp_enabled
        and pibar_mode == 'uniform'
        and dtype in (torch.float32, torch.float64)
        and device.type == 'cuda'
        and S > 256
    ):
        cross_pibar_row_stats = pibar_row_stats_fused(Pi_star_wave)

    def _compute_active_mask(rhs):
        if kernelized_active_mask_enabled:
            threshold = pruning_threshold if use_pruning else 0.0
            return active_mask_from_rhs_absmax_fused(
                rhs, threshold, use_pruning=use_pruning
            )
        clade_max = rhs.abs().max(dim=1).values
        if use_pruning:
            return clade_max >= pruning_threshold
        return clade_max > 0

    def _record_family_chunk_diag(k, active_mask, current_wave_scheduled):
        if family_chunk_diag_values is None or active_mask is None:
            return

        info = family_chunk_diag_wave_info[k]
        meta = wave_metas[k]
        W = int(meta['W'])
        device_i = active_mask.device
        active_i64 = active_mask.to(torch.int64)
        chunk_ids = info['chunk_ids']
        chunk_sizes = info['chunk_sizes']
        chunk_families = info['chunk_families']
        n_chunks = int(info['n_chunks'])

        chunk_active_counts = torch.zeros(
            n_chunks, dtype=torch.int64, device=device_i
        )
        chunk_active_counts.scatter_add_(0, chunk_ids, active_i64)
        active_chunks = chunk_active_counts > 0

        active_rows = active_i64.sum()
        active_chunk_rows = chunk_sizes[active_chunks].sum()
        active_chunk_count = active_chunks.to(torch.int64).sum()

        family_active_counts = torch.zeros(
            family_chunk_diag_family_count, dtype=torch.int64, device=device_i
        )
        family_active_counts.scatter_add_(
            0, chunk_families, active_chunks.to(torch.int64)
        )
        active_family_slots = (family_active_counts > 0).to(torch.int64).sum()

        current_rows = W if current_wave_scheduled else 0
        current_splits = 0
        chunk_scheduled_splits = torch.zeros((), dtype=torch.int64, device=device_i)
        active_parent_splits = torch.zeros((), dtype=torch.int64, device=device_i)
        if meta.get('has_splits') and current_wave_scheduled:
            reduce_idx = meta['reduce_idx']
            reduce_idx_long = reduce_idx.long()
            current_splits = int(reduce_idx.numel())
            parent_chunk_ids = chunk_ids[reduce_idx_long]
            chunk_scheduled_splits = active_chunks[parent_chunk_ids].to(torch.int64).sum()
            active_parent_splits = active_i64[reduce_idx_long].sum()

        family_chunk_diag_values['waves'] += 1
        family_chunk_diag_values['rows_total'].append(
            torch.tensor(W, dtype=torch.int64, device=device_i)
        )
        family_chunk_diag_values['rows_current'].append(
            torch.tensor(current_rows, dtype=torch.int64, device=device_i)
        )
        family_chunk_diag_values['rows_active'].append(active_rows)
        family_chunk_diag_values['rows_chunk_scheduled'].append(active_chunk_rows)
        family_chunk_diag_values['chunks_total'].append(
            torch.tensor(n_chunks, dtype=torch.int64, device=device_i)
        )
        family_chunk_diag_values['chunks_active'].append(active_chunk_count)
        family_chunk_diag_values['family_slots_total'].append(
            torch.tensor(
                int(info['n_family_slots']), dtype=torch.int64, device=device_i
            )
        )
        family_chunk_diag_values['family_slots_active'].append(active_family_slots)
        family_chunk_diag_values['splits_current'].append(
            torch.tensor(current_splits, dtype=torch.int64, device=device_i)
        )
        family_chunk_diag_values['splits_chunk_scheduled'].append(
            chunk_scheduled_splits
        )
        family_chunk_diag_values['splits_active_parent'].append(
            active_parent_splits
        )

    for k in range(K - 1, -1, -1):
        meta = wave_metas[k]
        ws = meta['start']
        we = meta['end']
        W = meta['W']

        use_fused = can_use_fused_uniform_backward
        rhs_slice = accumulated_rhs[ws:we]
        # The fused uniform kernel treats rhs as read-only, and this wave's
        # later cross-DTS/Pibar adjoints accumulate into child rows.
        rhs_k = (
            rhs_slice
            if (use_fused and fused_uniform_backward_view_rhs)
            else rhs_slice.clone()
        )
        no_cpu_pruning_wave = no_cpu_pruning and use_fused
        device_pruning_wave = device_pruning_requested and use_fused
        kernel_pruning_wave = no_cpu_pruning_wave or device_pruning_wave
        active_mask = None
        active_mask_for_dts_forward = None
        active_mask_for_wave_kernel = None
        active_mask_for_split_kernels = None

        if kernel_pruning_wave:
            if use_pruning or device_pruning_wave:
                active_mask = _compute_active_mask(rhs_k)
            active_mask_for_existing_policy = active_mask is not None
            if family_chunk_pruning_diag_enabled and active_mask is None:
                active_mask = _compute_active_mask(rhs_k)
            if active_mask_for_existing_policy:
                active_mask = active_mask.contiguous()
                active_mask_for_dts_forward = active_mask
                active_mask_for_wave_kernel = active_mask
                active_mask_for_split_kernels = active_mask
            if active_mask is not None:
                _record_family_chunk_diag(
                    k, active_mask, current_wave_scheduled=True
                )
            if device_pruning_wave:
                device_pruning_clades_total += W
                device_pruning_waves_total += 1
                device_pruning_active_counts.append(active_mask.sum())
                device_pruning_wave_active_flags.append(active_mask.any())
            n_active = W
            use_compact = False
            active_idx = None
            rhs_active = rhs_k
        else:
            active_mask = _compute_active_mask(rhs_k)
            wave_active = bool(active_mask.any())
            _record_family_chunk_diag(
                k, active_mask, current_wave_scheduled=wave_active
            )

            if not wave_active:
                n_waves_skipped += 1
                n_clades_skipped += W
                continue

            if use_fused:
                apply_hybrid_row_pruning = hybrid_row_pruning_enabled
                n_active_for_policy = None
                if apply_hybrid_row_pruning and (
                    hybrid_row_pruning_require_partial
                    or hybrid_row_pruning_min_inactive_frac > 0.0
                ):
                    n_active_for_policy = int(active_mask.sum().item())
                    n_inactive_for_policy = W - n_active_for_policy
                    if (
                        hybrid_row_pruning_require_partial
                        and n_inactive_for_policy == 0
                    ):
                        apply_hybrid_row_pruning = False
                    if hybrid_row_pruning_min_inactive_frac > 0.0:
                        inactive_frac = n_inactive_for_policy / W
                        apply_hybrid_row_pruning = (
                            apply_hybrid_row_pruning
                            and inactive_frac >= hybrid_row_pruning_min_inactive_frac
                        )
                if apply_hybrid_row_pruning:
                    active_mask = active_mask.contiguous()
                    if hybrid_prune_self:
                        active_mask_for_dts_forward = active_mask
                        active_mask_for_wave_kernel = active_mask
                    if hybrid_prune_splits:
                        active_mask_for_split_kernels = active_mask
                # The fused Triton path does not consume compact row indices.
                # Keep optional row statistics out of the production path
                # because sum().item() synchronizes the wave loop.
                if backward_pruning_row_stats_enabled:
                    n_active = (
                        n_active_for_policy
                        if n_active_for_policy is not None
                        else int(active_mask.sum().item())
                    )
                    n_clades_skipped += (W - n_active)
                else:
                    n_active = W
            else:
                n_active = int(active_mask.sum().item())
                n_clades_skipped += (W - n_active)

        Pi_W_star = Pi_star_wave[ws:we].detach()

        leaf_wt = None if (use_fused and use_uniform_leaf_index) else _get_leaf_wt(ws, we)

        if meta['has_splits']:
            reduce_idx = meta['reduce_idx']
            if _auto_wrapped:
                log_pD_dts = log_pD_shared
                log_pS_dts = log_pS_shared
            else:
                reduce_idx_long = reduce_idx.long()
                fi_splits_dts = family_idx[ws + reduce_idx_long]
                log_pD_dts = log_pD[fi_splits_dts]
                log_pS_dts = log_pS[fi_splits_dts]
                if log_pD_dts.ndim == 1:
                    log_pD_dts = log_pD_dts.unsqueeze(-1)
                if log_pS_dts.ndim == 1:
                    log_pS_dts = log_pS_dts.unsqueeze(-1)
            with torch.no_grad():
                if _compute_dts_cross_kernelized is not None:
                    dts_r = _compute_dts_cross_kernelized(
                        Pi_star_wave.detach(), Pibar_star_wave.detach(), meta,
                        sp_child1, sp_child2, log_pD_dts, log_pS_dts, S, device, dtype,
                        active_mask=active_mask_for_dts_forward,
                        parent_reduced=parent_reduced_backward_dts_enabled,
                        parent_reduced_min_splits=parent_reduced_backward_dts_min_splits,
                        parent_reduced_impl=parent_reduced_backward_dts_impl,
                        parent_reduced_tile_splits=parent_reduced_backward_dts_tile_splits,
                    )
                else:
                    dts_r = _dts_cross_differentiable(
                        Pi_star_wave.detach(), Pibar_star_wave.detach(), meta,
                        sp_child1, sp_child2, log_pD_dts, log_pS_dts, S, device, dtype,
                    )
        else:
            dts_r = None

        use_family_indexed_self_loop = bool(use_fused and not _auto_wrapped)
        mt_w, DL_w, E_w, Ebar_w, SL1_w, SL2_w = _wave_consts(
            ws, we, family_indexed=use_family_indexed_self_loop
        )

        if not kernel_pruning_wave and not use_fused:
            use_compact = (n_active < W)
            if use_compact:
                active_idx = active_mask.nonzero(as_tuple=True)[0]
                rhs_active = rhs_k[active_idx]
            else:
                active_idx = None
                rhs_active = rhs_k
        elif not kernel_pruning_wave:
            use_compact = False
            active_idx = None
            rhs_active = rhs_k

        # Per-wave family indices for scatter accumulation.
        fi_w = family_idx[ws:we]
        fi_expand = fi_w.unsqueeze(1).expand(W, S)

        def _scatter_accum(acc, contrib):
            if G == 1:
                if acc.ndim == 1:
                    acc[0] += contrib.sum()
                else:
                    acc[0] += contrib.sum(dim=0)
                return
            if acc.ndim == 1:
                acc.scatter_add_(0, fi_w, contrib.sum(dim=1))
            else:
                acc.scatter_add_(0, fi_expand, contrib)

        if use_fused:
            accum_param_grads = None
            if fused_wave_param_accum_enabled:
                if _auto_wrapped:
                    accum_param_grads = (
                        grad_log_pD,
                        grad_log_pS,
                        grad_E_acc[0],
                        grad_Ebar_acc[0],
                        grad_E_s1_acc[0],
                        grad_E_s2_acc[0],
                        grad_mt[0],
                    )
                elif use_family_indexed_self_loop:
                    accum_param_grads = (
                        grad_log_pD,
                        grad_log_pS,
                        grad_E_acc,
                        grad_Ebar_acc,
                        grad_E_s1_acc,
                        grad_E_s2_acc,
                        grad_mt,
                    )
            use_cuda_nosplit = (
                cuda_self_loop_nosplit_enabled
                and _auto_wrapped
                and dts_r is None
                and accum_param_grads is not None
                and dtype == torch.float32
                and pibar_mode == 'uniform'
                and use_uniform_leaf_index
                and torch.is_tensor(uniform_leaf_logp)
                and uniform_leaf_logp.numel() == S
                and torch.is_tensor(sp_parent_wave)
                and torch.is_tensor(forward_pibar_row_max)
                and compact_level_ptr is not None
                and compact_level_parents is not None
                and compact_level_child1 is not None
                and compact_level_child2 is not None
                and sp_child1_wave.dtype == torch.int32
                and sp_child2_wave.dtype == torch.int32
                and sp_parent_wave.dtype == torch.int32
                and grad_log_pD.numel() == 1
                and grad_log_pS.numel() == 1
            )
            if use_cuda_nosplit:
                from .kernels.wave_backward_cuda import wave_backward_uniform_nosplit_cuda

                v_k, aw0, aw1, aw2, aw345, aw3, aw4 = wave_backward_uniform_nosplit_cuda(
                    Pi_star_wave,
                    Pibar_star_wave,
                    ws,
                    W,
                    S,
                    rhs_k,
                    mt_w,
                    DL_w,
                    Ebar_w,
                    E_w,
                    SL1_w,
                    SL2_w,
                    sp_child1_wave,
                    sp_child2_wave,
                    sp_parent_wave,
                    leaf_species_index_wave,
                    uniform_leaf_logp,
                    forward_pibar_row_max,
                    compact_level_ptr,
                    compact_level_parents,
                    compact_level_child1,
                    compact_level_child2,
                    accum_param_grads,
                    active_mask=active_mask_for_wave_kernel,
                    neumann_terms=neumann_terms,
                    correction_mode=cuda_self_loop_nosplit_correction,
                )
            else:
                skip_wave_inactive_zero_stores = (
                    skip_inactive_zero_stores_enabled
                    and active_mask_for_wave_kernel is not None
                    and accum_param_grads is not None
                    and (
                        not meta['has_splits']
                        or active_mask_for_split_kernels is active_mask_for_wave_kernel
                    )
                )
                v_k, aw0, aw1, aw2, aw345, aw3, aw4 = wave_backward_uniform_fused(
                    Pi_star_wave, Pibar_star_wave, ws, W, S,
                    dts_r, rhs_k,
                    mt_w, DL_w, Ebar_w, E_w, SL1_w, SL2_w,
                    sp_child1_wave, sp_child2_wave, leaf_wt,
                    neumann_terms=neumann_terms,
                    leaf_species_idx=leaf_species_index_wave if use_uniform_leaf_index else None,
                    leaf_logp=uniform_leaf_logp if use_uniform_leaf_index else None,
                    accum_param_grads=accum_param_grads,
                    active_mask=active_mask_for_wave_kernel,
                    sp_parent=sp_parent_wave,
                    max_ancestor_depth=max_ancestor_depth,
                    pibar_row_max=forward_pibar_row_max,
                    skip_inactive_zero_stores=skip_wave_inactive_zero_stores,
                    scratch=scratch_pool.get("wave") if scratch_pool is not None else None,
                    family_idx=family_idx if use_family_indexed_self_loop else None,
                    family_indexed_consts=use_family_indexed_self_loop,
                )

            if accum_param_grads is None:
                _scatter_accum(grad_log_pD, aw0)
                _scatter_accum(grad_log_pS, aw345)
                _scatter_accum(grad_E_acc, aw0 + aw2)
                _scatter_accum(grad_Ebar_acc, aw1)
                _scatter_accum(grad_E_s1_acc, aw4)
                _scatter_accum(grad_E_s2_acc, aw3)
                _scatter_accum(grad_mt, aw2)

        else:
            Pibar_W_star = Pibar_star_wave[ws:we]
            ingredients = _self_loop_vjp_precompute(
                Pi_W_star, Pibar_W_star, dts_r,
                mt_w, DL_w, Ebar_w, E_w, SL1_w, SL2_w,
                sp_child1, sp_child2, leaf_wt, S,
                pibar_mode, transfer_mat_T, ancestors_T,
            )

            if use_compact:
                compact_ing = {
                    'w_L': ingredients['w_L'][active_idx],
                    'w_terms': ingredients['w_terms'][:, active_idx],
                    'p_prime': ingredients['p_prime'][active_idx],
                }
                if 'pibar_inv_denom' in ingredients:
                    compact_ing['pibar_inv_denom'] = ingredients['pibar_inv_denom'][active_idx]
                if 'pibar_inv_matmul' in ingredients:
                    compact_ing['pibar_inv_matmul'] = ingredients['pibar_inv_matmul'][active_idx]
                if 'pibar_matmul' in ingredients:
                    compact_ing['pibar_matmul'] = ingredients['pibar_matmul'][active_idx]
                for key in ('sc1_valid', 'sc1_idx', 'sc2_valid', 'sc2_idx'):
                    if key in ingredients:
                        compact_ing[key] = ingredients[key]
                solve_ing = compact_ing
                solve_W = n_active
            else:
                solve_ing = ingredients
                solve_W = W

            if pibar_mode == 'uniform':
                v_k = _gmres_self_loop_solve(
                    rhs_active, solve_ing, sp_child1, sp_child2, S, solve_W,
                    pibar_mode, transfer_mat_T, ancestors_T,
                    max_iters=5, tol=1e-8,
                )
            else:
                v_k = rhs_active.clone()
                term = rhs_active
                for _n in range(neumann_terms):
                    term = _self_loop_Jt_apply(
                        term, solve_ing, sp_child1, sp_child2, S, solve_W,
                        pibar_mode, transfer_mat_T, ancestors_T,
                    )
                    v_k = v_k + term

            if use_compact:
                v_k_full = torch.zeros(W, S, device=device, dtype=dtype)
                v_k_full[active_idx] = v_k
                v_k = v_k_full

            alpha_full = v_k * ingredients['w_L']
            wt = ingredients['w_terms']

            aw0 = alpha_full * wt[0]
            aw1 = alpha_full * wt[1]
            aw2 = alpha_full * wt[2]
            aw3 = alpha_full * wt[3]
            aw4 = alpha_full * wt[4]
            aw5 = alpha_full * wt[5]

            _scatter_accum(grad_log_pD, aw0)
            _scatter_accum(grad_log_pS, aw3 + aw4 + aw5)
            _scatter_accum(grad_E_acc, aw0 + aw2)
            _scatter_accum(grad_Ebar_acc, aw1)
            _scatter_accum(grad_E_s1_acc, aw4)
            _scatter_accum(grad_E_s2_acc, aw3)
            _scatter_accum(grad_mt, aw2)

            if pibar_mode in ('dense', 'topk') and grad_transfer_mat_acc is not None:
                v_Pibar_full = alpha_full * wt[2]
                matmul_r = ingredients['pibar_matmul']
                mr_safe = torch.where(matmul_r > 0, matmul_r, torch.ones_like(matmul_r))
                u_mr = torch.where(matmul_r > 0, v_Pibar_full / mr_safe, torch.zeros_like(v_Pibar_full))
                grad_transfer_mat_acc = grad_transfer_mat_acc + u_mr.T @ ingredients['p_prime']

        if meta['has_splits'] and dts_r is not None:
            sl = meta['sl']
            sr = meta['sr']
            wlsp = meta['log_split_probs']
            reduce_idx = meta['reduce_idx']
            n_ws = sl.shape[0]

            fused_scalar_params = (log_pD.numel() == 1 and log_pS.numel() == 1)
            used_fused_pibar_vjp = False
            used_fused_direct_pi_accum = False
            used_dts_mt_reduction_accum = False
            used_dts_pibar_ud_fusion = False
            pibar_side_active = None

            if use_fused and fused_scalar_params:
                # G=1: pass shared params to fused kernel.
                if fused_dts_backward_accum_enabled:
                    dts_accum_threshold_match = (
                        n_ws >= grouped_dts_backward_min_splits
                        and (n_ws / max(W, 1)) >= grouped_dts_backward_min_fanout
                    )
                    use_noatomic_dts_accum = (
                        noatomic_dts_backward_accum_enabled
                        and (noatomic_dts_backward_accum_all or dts_accum_threshold_match)
                    )
                    use_grouped_dts_accum = (
                        grouped_dts_backward_accum_enabled
                        and (grouped_dts_backward_accum_all or dts_accum_threshold_match)
                    )
                    group_children = group_inverse = None
                    if use_noatomic_dts_accum or use_grouped_dts_accum:
                        group_children = meta.get('_dts_accum_group_children')
                        group_inverse = meta.get('_dts_accum_group_inverse')
                        if (
                            not torch.is_tensor(group_children)
                            or not torch.is_tensor(group_inverse)
                            or group_children.device != sl.device
                            or group_inverse.device != sl.device
                        ):
                            all_children = torch.cat((sl, sr), dim=0)
                            group_children, group_inverse = torch.unique(
                                all_children,
                                sorted=True,
                                return_inverse=True,
                            )
                            group_children = group_children.contiguous()
                            group_inverse = group_inverse.contiguous()
                            meta['_dts_accum_group_children'] = group_children
                            meta['_dts_accum_group_inverse'] = group_inverse
                    child_rows_unique = (
                        torch.is_tensor(group_children)
                        and int(group_children.shape[0]) == int(2 * n_ws)
                    )
                    if use_noatomic_dts_accum and child_rows_unique:
                        (grad_Pibar_l, grad_Pibar_r,
                         param_pD, param_pS) = dts_cross_backward_accum_fused(
                            Pi_star_wave, Pibar_star_wave, v_k, ws,
                            sl, sr, reduce_idx, wlsp,
                            log_pD, log_pS,
                            sp_child1, sp_child2, accumulated_rhs, S,
                            active_mask=active_mask_for_split_kernels,
                            use_atomics=False,
                        )
                    elif use_grouped_dts_accum:
                        (grad_Pibar_l, grad_Pibar_r,
                         param_pD, param_pS) = dts_cross_backward_accum_grouped_fused(
                            Pi_star_wave, Pibar_star_wave, v_k, ws,
                            sl, sr, reduce_idx, wlsp,
                            log_pD, log_pS,
                            sp_child1, sp_child2, accumulated_rhs, S,
                            active_mask=active_mask_for_split_kernels,
                            group_children=group_children,
                            group_inverse=group_inverse,
                        )
                    else:
                        reduction_threshold_match = (
                            n_ws >= dts_reduction_accum_min_splits
                        )
                        pibar_ud_fusion_match = (
                            dts_pibar_ud_fusion_enabled
                            and fused_cross_pibar_vjp_enabled
                            and fused_cross_pibar_vjp_impl == "tree"
                            and level_parents is not None
                            and forward_pibar_row_max is not None
                            and mt_shared is not None
                            and mt_shared.ndim == 1
                            and n_ws >= dts_pibar_ud_min_splits
                        )
                        use_dts_reduction_accum_scalar = (
                            dts_reduction_accum_scalar_enabled
                            and reduction_threshold_match
                        )
                        use_dts_reduction_accum_mt = (
                            (
                                dts_reduction_accum_mt_enabled
                                and reduction_threshold_match
                            )
                            or pibar_ud_fusion_match
                        )
                        used_dts_mt_reduction_accum = use_dts_reduction_accum_mt
                        used_dts_pibar_ud_fusion = pibar_ud_fusion_match
                        ge2_ptr = meta.get('ge2_ptr')
                        ge2_parent_ids = meta.get('ge2_parent_ids')
                        ge2_fanout = float(meta.get('ge2_mean_fanout', 0.0))
                        parent_tiled_threshold_match = (
                            n_ws >= parent_tiled_dts_backward_min_splits
                            and ge2_fanout >= parent_tiled_dts_backward_min_fanout
                        )
                        use_parent_tiled_dts_accum = (
                            parent_tiled_dts_backward_accum_enabled
                            and (
                                parent_tiled_dts_backward_accum_all
                                or parent_tiled_threshold_match
                            )
                            and pibar_ud_fusion_match
                            and merged_dts_backward_accum_enabled
                            and use_dts_reduction_accum_scalar
                            and torch.is_tensor(ge2_ptr)
                            and torch.is_tensor(ge2_parent_ids)
                            and int(meta.get('n_ge2_clades', 0)) > 0
                        )
                        use_parent_ragged_dts_accum = (
                            parent_ragged_dts_backward_accum_enabled
                            and (
                                parent_ragged_dts_backward_accum_all
                                or parent_tiled_threshold_match
                            )
                            and pibar_ud_fusion_match
                            and merged_dts_backward_accum_enabled
                            and use_dts_reduction_accum_scalar
                            and torch.is_tensor(ge2_ptr)
                            and torch.is_tensor(ge2_parent_ids)
                            and int(meta.get('n_ge2_clades', 0)) > 0
                        )
                        if use_parent_ragged_dts_accum:
                            ragged_key = (
                                "_dts_accum_parent_ragged_worklist_"
                                f"{int(parent_ragged_dts_backward_tile_splits)}"
                            )
                            ragged_worklist = meta.get(ragged_key)
                            if (
                                not isinstance(ragged_worklist, tuple)
                                or len(ragged_worklist) != 3
                                or not all(torch.is_tensor(t) for t in ragged_worklist)
                                or any(t.device != ge2_ptr.device for t in ragged_worklist)
                            ):
                                ragged_worklist = _build_parent_ragged_ge2_worklist(
                                    int(meta.get('n_eq1', 0)),
                                    ge2_ptr,
                                    ge2_parent_ids,
                                    parent_ragged_dts_backward_tile_splits,
                                )
                                meta[ragged_key] = ragged_worklist
                            (grad_Pibar_l, grad_Pibar_r,
                             param_pD, param_pS) = dts_cross_backward_accum_parent_ragged_fused(
                                Pi_star_wave, Pibar_star_wave, v_k, ws,
                                sl, sr, reduce_idx, wlsp,
                                int(meta.get('n_eq1', 0)),
                                ge2_ptr,
                                ge2_parent_ids,
                                log_pD, log_pS,
                                sp_child1, sp_child2, accumulated_rhs, S,
                                active_mask=active_mask_for_split_kernels,
                                grad_log_pD=grad_log_pD,
                                grad_log_pS=grad_log_pS,
                                grad_mt=(
                                    grad_mt
                                    if grad_mt.ndim == 1
                                    else grad_mt[0]
                                ),
                                accum_mt_reduction=use_dts_reduction_accum_mt,
                                mt_squeezed=mt_shared,
                                pibar_row_max=forward_pibar_row_max,
                                tile_splits=parent_ragged_dts_backward_tile_splits,
                                ge2_tile_split_starts=ragged_worklist[0],
                                ge2_tile_split_ends=ragged_worklist[1],
                                ge2_tile_parent_ids=ragged_worklist[2],
                            )
                        elif use_parent_tiled_dts_accum:
                            (grad_Pibar_l, grad_Pibar_r,
                             param_pD, param_pS) = dts_cross_backward_accum_parent_tiled_fused(
                                Pi_star_wave, Pibar_star_wave, v_k, ws,
                                sl, sr, reduce_idx, wlsp,
                                int(meta.get('n_eq1', 0)),
                                ge2_ptr,
                                ge2_parent_ids,
                                log_pD, log_pS,
                                sp_child1, sp_child2, accumulated_rhs, S,
                                active_mask=active_mask_for_split_kernels,
                                grad_log_pD=grad_log_pD,
                                grad_log_pS=grad_log_pS,
                                grad_mt=(
                                    grad_mt
                                    if grad_mt.ndim == 1
                                    else grad_mt[0]
                                ),
                                accum_mt_reduction=use_dts_reduction_accum_mt,
                                mt_squeezed=mt_shared,
                                pibar_row_max=forward_pibar_row_max,
                                tile_splits=parent_tiled_dts_backward_tile_splits,
                                ge2_max_fanout=meta.get('ge2_max_fanout'),
                            )
                        else:
                            dts_accum_result = dts_cross_backward_accum_fused(
                                Pi_star_wave, Pibar_star_wave, v_k, ws,
                                sl, sr, reduce_idx, wlsp,
                                log_pD, log_pS,
                                sp_child1, sp_child2, accumulated_rhs, S,
                                active_mask=active_mask_for_split_kernels,
                                merge_s_term=merged_dts_backward_accum_enabled,
                                grad_log_pD=grad_log_pD,
                                grad_log_pS=grad_log_pS,
                                grad_mt=(
                                    grad_mt
                                    if grad_mt.ndim == 1
                                    else grad_mt[0]
                                ),
                                accum_param_reductions=use_dts_reduction_accum_scalar,
                                accum_mt_reduction=use_dts_reduction_accum_mt,
                                output_pibar_ud=pibar_ud_fusion_match,
                                output_pibar_side_active=(
                                    pibar_ud_fusion_match
                                    and dts_pibar_ud_skip_zero_sides_enabled
                                ),
                                pibar_side_threshold=dts_pibar_ud_side_threshold_arg,
                                mt_squeezed=mt_shared,
                                pibar_row_max=forward_pibar_row_max,
                                grad_mt_two_stage=(
                                    dts_grad_mt_two_stage_enabled
                                    and pibar_ud_fusion_match
                                ),
                                grad_mt_two_stage_tile_splits=(
                                    dts_grad_mt_two_stage_tile_splits
                                ),
                                skip_inactive_pibar_output_zero=(
                                    skip_inactive_zero_stores_enabled
                                    and pibar_ud_fusion_match
                                    and active_mask_for_split_kernels is not None
                                ),
                                scratch=(
                                    scratch_pool.get("dts")
                                    if scratch_pool is not None
                                    else None
                                ),
                            )
                            if (
                                pibar_ud_fusion_match
                                and dts_pibar_ud_skip_zero_sides_enabled
                            ):
                                (grad_Pibar_l, grad_Pibar_r, pibar_side_active,
                                 param_pD, param_pS) = dts_accum_result
                            else:
                                (grad_Pibar_l, grad_Pibar_r,
                                 param_pD, param_pS) = dts_accum_result
                    used_fused_direct_pi_accum = True
                    grad_Pi_l = grad_Pi_r = None
                else:
                    (grad_Pi_l, grad_Pi_r, grad_Pibar_l, grad_Pibar_r,
                     param_pD, param_pS) = dts_cross_backward_fused(
                        Pi_star_wave, Pibar_star_wave, v_k, ws,
                        sl, sr, reduce_idx, wlsp,
                        log_pD, log_pS,
                        sp_child1, sp_child2, S,
                        active_mask=active_mask_for_split_kernels,
                    )

                # Accumulate into G=1 row. The direct DTS accumulation path can
                # optionally do these reductions in-kernel.
                if not (
                    used_fused_direct_pi_accum
                    and param_pD is None
                    and param_pS is None
                ):
                    grad_log_pD[0] += param_pD.sum()
                    grad_log_pS[0] += param_pS.sum()
                if not (
                    used_fused_direct_pi_accum
                    and used_dts_mt_reduction_accum
                ):
                    mt_contrib = grad_Pibar_l.sum(dim=0) + grad_Pibar_r.sum(dim=0)
                    if grad_mt.ndim == 1:
                        grad_mt[0] += mt_contrib.sum()
                    else:
                        grad_mt[0] += mt_contrib

            elif use_fused and fused_dts_backward_accum_enabled:
                dts_log_pD = log_pD_shared if _auto_wrapped else log_pD
                dts_log_pS = log_pS_shared if _auto_wrapped else log_pS
                dts_grad_log_pD = grad_log_pD[0] if _auto_wrapped else grad_log_pD
                dts_grad_log_pS = grad_log_pS[0] if _auto_wrapped else grad_log_pS
                dts_grad_mt = grad_mt[0] if _auto_wrapped else grad_mt
                dts_mt = mt_shared if _auto_wrapped else mt_squeezed
                dts_family_idx = None if _auto_wrapped else family_idx
                pibar_ud_fusion_match = (
                    dts_pibar_ud_fusion_enabled
                    and fused_cross_pibar_vjp_enabled
                    and fused_cross_pibar_vjp_impl == "tree"
                    and level_parents is not None
                    and forward_pibar_row_max is not None
                    and torch.is_tensor(dts_mt)
                    and n_ws >= dts_pibar_ud_min_splits
                )
                used_dts_mt_reduction_accum = True
                used_dts_pibar_ud_fusion = pibar_ud_fusion_match
                dts_accum_result = dts_cross_backward_accum_fused(
                    Pi_star_wave, Pibar_star_wave, v_k, ws,
                    sl, sr, reduce_idx, wlsp,
                    dts_log_pD, dts_log_pS,
                    sp_child1, sp_child2, accumulated_rhs, S,
                    active_mask=active_mask_for_split_kernels,
                    merge_s_term=merged_dts_backward_accum_enabled,
                    grad_log_pD=dts_grad_log_pD,
                    grad_log_pS=dts_grad_log_pS,
                    grad_mt=dts_grad_mt,
                    accum_param_reductions=True,
                    accum_mt_reduction=True,
                    output_pibar_ud=pibar_ud_fusion_match,
                    output_pibar_side_active=(
                        pibar_ud_fusion_match
                        and dts_pibar_ud_skip_zero_sides_enabled
                    ),
                    pibar_side_threshold=dts_pibar_ud_side_threshold_arg,
                    mt_squeezed=dts_mt,
                    pibar_row_max=forward_pibar_row_max,
                    grad_mt_two_stage=False,
                    skip_inactive_pibar_output_zero=(
                        skip_inactive_zero_stores_enabled
                        and pibar_ud_fusion_match
                        and active_mask_for_split_kernels is not None
                    ),
                    scratch=(
                        scratch_pool.get("dts")
                        if scratch_pool is not None
                        else None
                    ),
                    family_idx=dts_family_idx,
                )
                if (
                    pibar_ud_fusion_match
                    and dts_pibar_ud_skip_zero_sides_enabled
                ):
                    (grad_Pibar_l, grad_Pibar_r, pibar_side_active,
                     param_pD, param_pS) = dts_accum_result
                else:
                    (grad_Pibar_l, grad_Pibar_r,
                     param_pD, param_pS) = dts_accum_result
                used_fused_direct_pi_accum = True
                grad_Pi_l = grad_Pi_r = None

            else:
                sl_long = sl.long()
                sr_long = sr.long()
                reduce_idx_long = reduce_idx.long()

                Pi_l = Pi_star_wave[sl_long]
                Pi_r = Pi_star_wave[sr_long]
                Pibar_l = Pibar_star_wave[sl_long]
                Pibar_r = Pibar_star_wave[sr_long]
                neg_inf_col = torch.full((Pi_star_wave.shape[0], 1), NEG_INF, device=device, dtype=dtype)
                Pi_col_pad = torch.cat([Pi_star_wave, neg_inf_col], dim=1)
                Pi_l_s1 = Pi_col_pad[sl_long][:, sp_child1.long()]
                Pi_l_s2 = Pi_col_pad[sl_long][:, sp_child2.long()]
                Pi_r_s1 = Pi_col_pad[sr_long][:, sp_child1.long()]
                Pi_r_s2 = Pi_col_pad[sr_long][:, sp_child2.long()]

                fi_splits = family_idx[ws + reduce_idx_long]
                _pD_s = log_pD[fi_splits]
                if _pD_s.ndim == 1:
                    _pD_s = _pD_s.unsqueeze(-1)
                _pS_s = log_pS[fi_splits]
                if _pS_s.ndim == 1:
                    _pS_s = _pS_s.unsqueeze(-1)

                DTS_5 = torch.stack([
                    _pD_s + Pi_l + Pi_r,
                    Pi_l + Pibar_r,
                    Pi_r + Pibar_l,
                    _pS_s + Pi_l_s1 + Pi_r_s2,
                    _pS_s + Pi_r_s1 + Pi_l_s2,
                ], dim=0)

                Pi_parent = Pi_W_star[reduce_idx_long]
                combined = wlsp + DTS_5
                v_k_parent = v_k[reduce_idx_long]
                grad_DTS_5 = v_k_parent.unsqueeze(0) * _safe_exp2_ratio(
                    combined, Pi_parent.unsqueeze(0))

                fi_split_expand = fi_splits.unsqueeze(1).expand(n_ws, S)
                if grad_log_pD.ndim == 1:
                    grad_log_pD.scatter_add_(0, fi_splits, grad_DTS_5[0].sum(dim=1))
                    grad_log_pS.scatter_add_(0, fi_splits, (grad_DTS_5[3] + grad_DTS_5[4]).sum(dim=1))
                else:
                    grad_log_pD.scatter_add_(0, fi_split_expand, grad_DTS_5[0])
                    grad_log_pS.scatter_add_(0, fi_split_expand, grad_DTS_5[3] + grad_DTS_5[4])
                child_ids_dts = torch.cat([sl_long, sr_long])
                fi_ch = family_idx[child_ids_dts]
                fi_ch_expand = fi_ch.unsqueeze(1).expand(2 * n_ws, S)
                grad_mt.scatter_add_(0, fi_ch_expand,
                                     torch.cat([grad_DTS_5[2], grad_DTS_5[1]], dim=0))

                if pibar_mode in ('dense', 'topk') and grad_transfer_mat_acc is not None:
                    v_Pibar_ch = torch.cat([grad_DTS_5[2], grad_DTS_5[1]], dim=0)
                    child_ids = torch.cat([sl_long, sr_long])
                    Pi_ch = Pi_star_wave[child_ids]
                    Pi_max_ch = Pi_ch.max(dim=1, keepdim=True).values
                    p_prime_ch = torch.exp2(Pi_ch - Pi_max_ch)
                    matmul_ch = p_prime_ch @ transfer_mat_T
                    mc_safe = torch.where(matmul_ch > 0, matmul_ch, torch.ones_like(matmul_ch))
                    u_mc = torch.where(matmul_ch > 0, v_Pibar_ch / mc_safe, torch.zeros_like(v_Pibar_ch))
                    grad_transfer_mat_acc = grad_transfer_mat_acc + u_mc.T @ p_prime_ch

                grad_Pi_l = grad_DTS_5[0] + grad_DTS_5[1]
                grad_Pi_r = grad_DTS_5[0] + grad_DTS_5[2]
                grad_Pibar_l = grad_DTS_5[2]
                grad_Pibar_r = grad_DTS_5[1]

                sc1 = sp_child1.long()
                sc2 = sp_child2.long()
                valid1 = sc1 < S
                valid2 = sc2 < S
                if valid1.any():
                    idx1 = sc1[valid1]
                    grad_Pi_l.scatter_add_(1, idx1.unsqueeze(0).expand(n_ws, -1), grad_DTS_5[3][:, valid1])
                    grad_Pi_r.scatter_add_(1, idx1.unsqueeze(0).expand(n_ws, -1), grad_DTS_5[4][:, valid1])
                if valid2.any():
                    idx2 = sc2[valid2]
                    grad_Pi_r.scatter_add_(1, idx2.unsqueeze(0).expand(n_ws, -1), grad_DTS_5[3][:, valid2])
                    grad_Pi_l.scatter_add_(1, idx2.unsqueeze(0).expand(n_ws, -1), grad_DTS_5[4][:, valid2])

            if not used_fused_direct_pi_accum:
                sl_long = sl.long()
                sr_long = sr.long()
                accumulated_rhs.index_add_(0, sl_long, grad_Pi_l)
                accumulated_rhs.index_add_(0, sr_long, grad_Pi_r)

            if (
                use_fused
                and (fused_scalar_params or used_dts_pibar_ud_fusion)
                and fused_cross_pibar_vjp_enabled
                and ancestor_cols is not None
            ):
                if used_dts_pibar_ud_fusion:
                    uniform_cross_pibar_vjp_tree_from_ud_fused(
                        Pi_star_wave,
                        grad_Pibar_l,
                        grad_Pibar_r,
                        sl,
                        sr,
                        sp_child1,
                        sp_child2,
                        level_parents,
                        accumulated_rhs,
                        S,
                        active_mask=active_mask_for_split_kernels,
                        reduce_idx=reduce_idx,
                        pibar_row_max=forward_pibar_row_max,
                        skip_zero_sides=dts_pibar_ud_skip_zero_sides_enabled,
                        side_active=pibar_side_active,
                        compact_level_ptr=(
                            compact_level_ptr
                            if dts_pibar_ud_compact_levels_enabled
                            else None
                        ),
                        compact_level_parents=(
                            compact_level_parents
                            if dts_pibar_ud_compact_levels_enabled
                            else None
                        ),
                        compact_level_child1=(
                            compact_level_child1
                            if dts_pibar_ud_compact_levels_enabled
                            else None
                        ),
                        compact_level_child2=(
                            compact_level_child2
                            if dts_pibar_ud_compact_levels_enabled
                            else None
                        ),
                        subtree_interval_start=(
                            subtree_interval_start
                            if dts_pibar_ud_euler_prefix_enabled
                            else None
                        ),
                        subtree_interval_end=(
                            subtree_interval_end
                            if dts_pibar_ud_euler_prefix_enabled
                            else None
                        ),
                        side_active_threshold=dts_pibar_ud_side_threshold_arg,
                    )
                elif (
                    grouped_cross_pibar_vjp_enabled
                    and fused_cross_pibar_vjp_impl == "tree"
                    and level_parents is not None
                    and active_mask_for_split_kernels is None
                ):
                    if grouped_cross_pibar_reduce_impl in ("triton", "kernel"):
                        group_children = meta.get('_cross_pibar_group_children')
                        group_inverse = meta.get('_cross_pibar_group_inverse')
                        if (
                            not torch.is_tensor(group_children)
                            or not torch.is_tensor(group_inverse)
                            or group_children.device != sl.device
                            or group_inverse.device != sl.device
                        ):
                            all_children = torch.cat((sl, sr), dim=0)
                            group_children, group_inverse = torch.unique(
                                all_children,
                                sorted=True,
                                return_inverse=True,
                            )
                            group_children = group_children.contiguous()
                            group_inverse = group_inverse.contiguous()
                            meta['_cross_pibar_group_children'] = group_children
                            meta['_cross_pibar_group_inverse'] = group_inverse
                        uniform_cross_pibar_vjp_tree_grouped_fused(
                            Pi_star_wave,
                            grad_Pibar_l,
                            grad_Pibar_r,
                            group_children,
                            group_inverse,
                            ancestor_cols,
                            sp_child1,
                            sp_child2,
                            level_parents,
                            accumulated_rhs,
                            S,
                            active_mask=(
                                active_mask
                                if grouped_cross_pibar_use_active
                                else active_mask_for_split_kernels
                            ),
                            reduce_idx=reduce_idx,
                        )
                    else:
                        all_children = torch.cat([sl.long(), sr.long()])
                        all_pibar_grad = torch.cat([grad_Pibar_l, grad_Pibar_r])
                        unique_children, inverse = torch.unique(
                            all_children, sorted=True, return_inverse=True
                        )
                        grouped_pibar_grad = torch.zeros(
                            (unique_children.shape[0], S),
                            device=device,
                            dtype=dtype,
                        )
                        grouped_pibar_grad.index_add_(0, inverse, all_pibar_grad)
                        uniform_cross_pibar_vjp_grouped_tree_fused(
                            Pi_star_wave,
                            unique_children,
                            grouped_pibar_grad,
                            ancestor_cols,
                            sp_child1,
                            sp_child2,
                            level_parents,
                            accumulated_rhs,
                            S,
                            row_stats=cross_pibar_row_stats,
                        )
                elif (
                    prefix_cross_pibar_vjp_impl
                    and level_parents is not None
                    and depth_nodes is not None
                    and sp_parent is not None
                ):
                    uniform_cross_pibar_vjp_tree_prefix_fused(
                        Pi_star_wave,
                        grad_Pibar_l,
                        grad_Pibar_r,
                        sl,
                        sr,
                        sp_parent,
                        sp_child1,
                        sp_child2,
                        depth_nodes,
                        level_parents,
                        accumulated_rhs,
                        S,
                        active_mask=active_mask_for_split_kernels,
                        reduce_idx=reduce_idx,
                        row_stats=cross_pibar_row_stats,
                    )
                elif fused_cross_pibar_vjp_impl == "tree" and level_parents is not None:
                    uniform_cross_pibar_vjp_tree_fused(
                        Pi_star_wave,
                        grad_Pibar_l,
                        grad_Pibar_r,
                        sl,
                        sr,
                        ancestor_cols,
                        sp_child1,
                        sp_child2,
                        level_parents,
                        accumulated_rhs,
                        S,
                        active_mask=active_mask_for_split_kernels,
                        reduce_idx=reduce_idx,
                        row_stats=cross_pibar_row_stats,
                        Pibar_star=Pibar_star_wave,
                        mt_squeezed=mt_shared,
                        pibar_row_max=forward_pibar_row_max,
                    )
                else:
                    uniform_cross_pibar_vjp_fused(
                        Pi_star_wave,
                        grad_Pibar_l,
                        grad_Pibar_r,
                        sl,
                        sr,
                        ancestor_cols,
                        accumulated_rhs,
                        S,
                        active_mask=active_mask_for_split_kernels,
                        reduce_idx=reduce_idx,
                        row_stats=cross_pibar_row_stats,
                    )
                used_fused_pibar_vjp = True

            if not used_fused_pibar_vjp:
                all_children = torch.cat([sl.long(), sr.long()])
                all_pibar_grad = torch.cat([grad_Pibar_l, grad_Pibar_r])

                nz = all_pibar_grad.abs().sum(dim=1) > 0
                if nz.any():
                    nz_children = all_children[nz]
                    u = all_pibar_grad[nz]
                    Pi_ch = Pi_star_wave[nz_children]
                    Pi_max_p = Pi_ch.max(dim=1, keepdim=True).values
                    p_prime = torch.exp2(Pi_ch - Pi_max_p)

                    if pibar_mode == 'uniform':
                        anc_sum = p_prime @ ancestors_T
                        denom = p_prime.sum(dim=1, keepdim=True) - anc_sum
                        denom_safe = torch.where(denom > 0, denom, torch.ones_like(denom))
                        u_d = torch.where(denom > 0, u / denom_safe, torch.zeros_like(u))
                        A = u_d.sum(dim=1, keepdim=True)
                        correction = (ancestors_T @ u_d.T).T
                        pi_from_pibar = p_prime * (A - correction)
                    else:
                        matmul_r = p_prime @ transfer_mat_T
                        mr_safe = torch.where(matmul_r > 0, matmul_r, torch.ones_like(matmul_r))
                        u_mr = torch.where(matmul_r > 0, u / mr_safe, torch.zeros_like(u))
                        pi_from_pibar = p_prime * (u_mr @ transfer_mat_T.T)

                    accumulated_rhs.index_add_(0, nz_children, pi_from_pibar)

    if device_pruning_active_counts:
        n_device_active = int(torch.stack(device_pruning_active_counts).sum().item())
        n_device_waves_active = int(torch.stack(device_pruning_wave_active_flags).sum().item())
        n_clades_skipped += device_pruning_clades_total - n_device_active
        n_waves_skipped += device_pruning_waves_total - n_device_waves_active

    family_chunk_diag_result = None
    if family_chunk_diag_values is not None:
        def _diag_sum(name):
            vals = family_chunk_diag_values[name]
            if not vals:
                return 0
            return int(torch.stack(vals).sum().item())

        rows_total_diag = _diag_sum('rows_total')
        rows_current_diag = _diag_sum('rows_current')
        rows_active_diag = _diag_sum('rows_active')
        rows_chunk_scheduled_diag = _diag_sum('rows_chunk_scheduled')
        chunks_total_diag = _diag_sum('chunks_total')
        chunks_active_diag = _diag_sum('chunks_active')
        family_slots_total_diag = _diag_sum('family_slots_total')
        family_slots_active_diag = _diag_sum('family_slots_active')
        splits_current_diag = _diag_sum('splits_current')
        splits_chunk_scheduled_diag = _diag_sum('splits_chunk_scheduled')
        splits_active_parent_diag = _diag_sum('splits_active_parent')

        family_chunk_diag_result = {
            'enabled': True,
            'chunk_rows': family_chunk_rows,
            'n_families': family_chunk_diag_family_count,
            'waves_observed': int(family_chunk_diag_values['waves']),
            'rows_total_observed': rows_total_diag,
            'rows_current_whole_wave_scheduled': rows_current_diag,
            'rows_active': rows_active_diag,
            'rows_chunk_scheduled': rows_chunk_scheduled_diag,
            'rows_skippable_by_chunk': max(
                0, rows_current_diag - rows_chunk_scheduled_diag
            ),
            'rows_skippable_by_row_mask': max(
                0, rows_current_diag - rows_active_diag
            ),
            'rows_in_active_chunks_but_inactive': max(
                0, rows_chunk_scheduled_diag - rows_active_diag
            ),
            'chunks_total': chunks_total_diag,
            'chunks_active': chunks_active_diag,
            'chunks_inactive': max(0, chunks_total_diag - chunks_active_diag),
            'family_slots_total': family_slots_total_diag,
            'family_slots_active': family_slots_active_diag,
            'family_slots_inactive': max(
                0, family_slots_total_diag - family_slots_active_diag
            ),
            'splits_current_whole_wave_scheduled': splits_current_diag,
            'splits_chunk_scheduled': splits_chunk_scheduled_diag,
            'splits_active_parent': splits_active_parent_diag,
            'splits_skippable_by_chunk': max(
                0, splits_current_diag - splits_chunk_scheduled_diag
            ),
            'splits_skippable_by_active_parent': max(
                0, splits_current_diag - splits_active_parent_diag
            ),
        }
        if rows_current_diag > 0:
            family_chunk_diag_result['row_chunk_scheduled_fraction'] = (
                rows_chunk_scheduled_diag / rows_current_diag
            )
            family_chunk_diag_result['row_active_fraction'] = (
                rows_active_diag / rows_current_diag
            )
        if splits_current_diag > 0:
            family_chunk_diag_result['split_chunk_scheduled_fraction'] = (
                splits_chunk_scheduled_diag / splits_current_diag
            )
            family_chunk_diag_result['split_active_parent_fraction'] = (
                splits_active_parent_diag / splits_current_diag
            )

    result = {
        'v_Pi': accumulated_rhs,
        'grad_E': grad_E_acc,
        'grad_Ebar': grad_Ebar_acc,
        'grad_E_s1': grad_E_s1_acc,
        'grad_E_s2': grad_E_s2_acc,
        'grad_log_pD': grad_log_pD,
        'grad_log_pS': grad_log_pS,
        'grad_max_transfer_mat': grad_mt,
        'n_waves_total': n_waves_total,
        'n_waves_skipped': n_waves_skipped,
        'n_waves_processed': n_waves_total - n_waves_skipped,
        'n_clades_total': n_clades_total,
        'n_clades_skipped': n_clades_skipped,
        'n_clades_active': n_clades_total - n_clades_skipped,
    }
    if grad_transfer_mat_acc is not None:
        result['grad_transfer_mat'] = grad_transfer_mat_acc
    if family_chunk_diag_result is not None:
        result['family_chunk_pruning_diag'] = family_chunk_diag_result

    # Unwrap G=1 results back to original shapes.
    if _auto_wrapped:
        for key in ('grad_E', 'grad_Ebar', 'grad_E_s1', 'grad_E_s2',
                     'grad_log_pD', 'grad_log_pS', 'grad_max_transfer_mat'):
            result[key] = result[key][0]

    return result
