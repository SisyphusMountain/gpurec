"""Forward pass: Pi_step, Pi_fixed_point, Pi_wave_forward and helpers."""
import torch

from .log2_utils import logsumexp2, logaddexp2
from .log2_utils import _safe_log2_internal as _safe_log2
from .kernels.scatter_lse import seg_logsumexp
from .kernels.wave_step import wave_step_fused, wave_pibar_step_fused, wave_step_uniform_fused
from .kernels.dts_fused import dts_fused
from ._logmatmul_compat import (
    HAS_LOGMATMUL as _HAS_LOGMATMUL,
    LogspaceMatmulFn,
    streaming_topk as _streaming_topk,
    logspace_matmul_compressed as _logspace_matmul_compressed,
)
from ._helpers import _safe_exp2_ratio, _seg_logsumexp_host, _nvtx_range, _nvtx_here  # noqa: F401

NEG_INF = float("-inf")

# Re-export legacy code so existing ``from .forward import Pi_step`` still works.
from .legacy import Pi_step, Pi_fixed_point  # noqa: F401


# ---------------------------------------------------------------------------
# Cross-clade DTS
# ---------------------------------------------------------------------------

def _compute_dts_cross(Pi, Pibar, meta, sp_child1, sp_child2, log_pD, log_pS,
                       S, device, dtype):
    """Compute DTS cross-clade terms and reduce to [W, S] for one wave."""
    sl = meta['sl']
    sr = meta['sr']
    wlsp = meta['log_split_probs']
    W = meta['W']

    dts_term = dts_fused(
        Pi, Pibar, sl, sr,
        sp_child1, sp_child2,
        log_pD, log_pS, wlsp,
    )

    NEG_INF = float('-inf')
    dts_r = torch.full((W, S), NEG_INF, device=device, dtype=dtype)

    n_eq1 = meta.get('n_eq1', 0)
    n_ge2_clades = meta.get('n_ge2_clades', 0)

    if n_eq1 > 0:
        dts_r[meta['eq1_reduce_idx']] = dts_term[:n_eq1]

    if n_ge2_clades > 0:
        ge2_term = dts_term[n_eq1:].contiguous()
        y_ge2 = seg_logsumexp(ge2_term, meta['ge2_ptr'])
        dts_r[meta['ge2_parent_ids']] = y_ge2

    return dts_r


# ---------------------------------------------------------------------------
# Pibar inline
# ---------------------------------------------------------------------------

def _compute_Pibar_inline(Pi_W, transfer_mat_T, mt_squeezed, pibar_mode,
                          ancestors_T=None, topk_k=16):
    """Compute Pibar for a wave, either via dense matmul or uniform approximation."""
    Pi_max = Pi_W.max(dim=1, keepdim=True).values
    if pibar_mode == 'uniform_approx':
        Pi_exp = torch.exp2(Pi_W - Pi_max)
        row_sum = Pi_exp.sum(dim=1, keepdim=True)
        Pibar_W = torch.log2(row_sum - Pi_exp) + Pi_max + mt_squeezed
    elif pibar_mode == 'uniform':
        Pi_exp = torch.exp2(Pi_W - Pi_max)
        row_sum = Pi_exp.sum(dim=1, keepdim=True)
        ancestor_sum = Pi_exp @ ancestors_T
        Pibar_W = torch.log2(row_sum - ancestor_sum) + Pi_max + mt_squeezed
    elif pibar_mode == 'topk':
        W, S = Pi_W.shape
        topk_vals, topk_idx = torch.topk(Pi_W, topk_k, dim=1)
        Pi_max_topk = topk_vals[:, :1]
        Pi_exp_topk = torch.exp2(topk_vals - Pi_max_topk)
        if W * topk_k * S * Pi_W.element_size() < 512 * 1024 * 1024:
            tf_gathered = transfer_mat_T[topk_idx.reshape(-1)].reshape(W, topk_k, S)
            Pibar_linear = torch.bmm(Pi_exp_topk.unsqueeze(1), tf_gathered).squeeze(1)
        else:
            Pibar_linear = torch.zeros(W, S, device=Pi_W.device, dtype=Pi_W.dtype)
            for j in range(topk_k):
                Pibar_linear.addcmul_(Pi_exp_topk[:, j:j+1], transfer_mat_T[topk_idx[:, j]])
        Pibar_W = _safe_log2(Pibar_linear) + Pi_max_topk + mt_squeezed
    else:
        Pibar_W = torch.log2(torch.exp2(Pi_W - Pi_max) @ transfer_mat_T) + Pi_max + mt_squeezed
    return Pibar_W


# ---------------------------------------------------------------------------
# DTS reduced (small-S path)
# ---------------------------------------------------------------------------

def _compute_DTS_reduced(Pi, Pibar, lr, n_ws, S, W, log_pD, log_pS,
                          sp_child1, sp_child2, wlsp, reduce_idx, device, dtype):
    """Compute DTS cross-clade terms and reduce to per-wave-clade."""
    Pi_lr = Pi[lr]
    Pibar_lr = Pibar[lr]
    neg_inf_col = torch.full((2 * n_ws, 1), NEG_INF, device=device, dtype=dtype)
    Pi_lr_pad = torch.cat([Pi_lr, neg_inf_col], dim=1)
    DTS = torch.empty((5, n_ws, S), device=device, dtype=dtype)
    DTS[0] = log_pD + Pi_lr[:n_ws] + Pi_lr[n_ws:]
    DTS[1] = Pi_lr[:n_ws] + Pibar_lr[n_ws:]
    DTS[2] = Pi_lr[n_ws:] + Pibar_lr[:n_ws]
    DTS[3] = log_pS + Pi_lr_pad[:n_ws, sp_child1] + Pi_lr_pad[n_ws:, sp_child2]
    DTS[4] = log_pS + Pi_lr_pad[n_ws:, sp_child1] + Pi_lr_pad[:n_ws, sp_child2]
    DTS_term = wlsp + logsumexp2(DTS, dim=0)
    reduce_exp = reduce_idx.unsqueeze(1).expand_as(DTS_term)
    DTS_reduced = torch.full((W, S), NEG_INF, device=device, dtype=dtype)
    DTS_reduced.scatter_reduce_(0, reduce_exp, DTS_term, reduce='amax', include_self=True)
    DTS_max = DTS_reduced.clone()
    DTS_sum = torch.zeros((W, S), device=device, dtype=dtype)
    DTS_sum.scatter_add_(0, reduce_exp, torch.exp2(DTS_term - DTS_max[reduce_idx]))
    return torch.log2(DTS_sum) + DTS_max


# ---------------------------------------------------------------------------
# Split parent reconstruction
# ---------------------------------------------------------------------------

def _reconstruct_split_parents(ccp_helpers):
    """Reconstruct split_parents_sorted from seg_parent_ids and segment structure."""
    seg_parent_ids = ccp_helpers['seg_parent_ids']
    num_ge2 = int(ccp_helpers['num_segs_ge2'])
    num_eq1 = int(ccp_helpers['num_segs_eq1'])
    end_rows_ge2 = int(ccp_helpers['end_rows_ge2'])
    ptr_ge2 = ccp_helpers['ptr_ge2']
    N_splits = int(ccp_helpers['N_splits'])

    split_parents = torch.empty(N_splits, dtype=torch.long, device=seg_parent_ids.device)

    for i in range(num_ge2):
        start = int(ptr_ge2[i].item())
        end = int(ptr_ge2[i + 1].item())
        parent_id = seg_parent_ids[i]
        split_parents[start:end] = parent_id

    for i in range(num_eq1):
        row = end_rows_ge2 + i
        parent_id = seg_parent_ids[num_ge2 + i]
        split_parents[row] = parent_id

    return split_parents


# ---------------------------------------------------------------------------
# Gradient bounds (pruning)
# ---------------------------------------------------------------------------

def compute_gradient_bounds(
    Pi,
    ccp_helpers,
    root_clade_idx,
    threshold=-20.0,
    wave_metas=None,
):
    """Compute gradient bounds for pruning clades from the backward pass."""
    C = Pi.shape[0]
    device = Pi.device
    dtype = Pi.dtype

    s = Pi.max(dim=1).values

    grad_bound = torch.full((C,), NEG_INF, device=device, dtype=dtype)
    if isinstance(root_clade_idx, int):
        grad_bound[root_clade_idx] = 0.0
    else:
        grad_bound[root_clade_idx] = 0.0

    if wave_metas is not None:
        for k in range(len(wave_metas) - 1, -1, -1):
            meta = wave_metas[k]
            if not meta['has_splits']:
                continue
            sl = meta['sl']
            sr = meta['sr']
            reduce_idx = meta['reduce_idx']
            ws = meta['start']

            gb_parents = grad_bound[reduce_idx + ws]
            new_l = gb_parents + s[sr]
            new_r = gb_parents + s[sl]

            grad_bound[sl] = logaddexp2(grad_bound[sl], new_l)
            grad_bound[sr] = logaddexp2(grad_bound[sr], new_r)
    else:
        N_splits = int(ccp_helpers['N_splits'])
        split_leftrights = ccp_helpers['split_leftrights_sorted']
        lefts = split_leftrights[:N_splits]
        rights = split_leftrights[N_splits:]

        split_parents = ccp_helpers.get('split_parents_sorted', None)
        if split_parents is None:
            split_parents = _reconstruct_split_parents(ccp_helpers)

        for i in range(N_splits):
            p = int(split_parents[i].item()) if torch.is_tensor(split_parents[i]) else int(split_parents[i])
            l_idx = int(lefts[i].item()) if torch.is_tensor(lefts[i]) else int(lefts[i])
            r_idx = int(rights[i].item()) if torch.is_tensor(rights[i]) else int(rights[i])

            if grad_bound[p].item() < threshold:
                continue

            gb_p = grad_bound[p]
            new_l = gb_p + s[l_idx]
            new_r = gb_p + s[r_idx]

            grad_bound[l_idx] = logaddexp2(grad_bound[l_idx], new_r)
            grad_bound[r_idx] = logaddexp2(grad_bound[r_idx], new_l)

    pruned_mask = grad_bound < threshold
    return grad_bound, pruned_mask


# ---------------------------------------------------------------------------
# Pi wave forward
# ---------------------------------------------------------------------------

def Pi_wave_forward(
    wave_layout,
    species_helpers,
    E,
    Ebar,
    E_s1,
    E_s2,
    log_pS,
    log_pD,
    log_pL,
    transfer_mat,
    max_transfer_mat,
    device,
    dtype,
    *,
    local_iters: int = 50,
    local_tolerance: float = 1e-3,
    fixed_iters: int | None = None,
    overlap_streams: bool = False,
    pibar_mode: str = 'dense',
    topk_k: int = 16,
):
    """Wave-based Pi forward pass with wave-ordered layout (v2).

    Clades are permuted so each wave occupies a contiguous block of Pi[ws:we].
    The self-loop uses zero-copy views instead of gather/scatter.

    Args:
        wave_layout: dict from build_wave_layout() containing permuted indices
                     and precomputed per-wave metadata
        species_helpers: species tree helpers dict
        E, Ebar, E_s1, E_s2: converged E vectors [S]
        log_pS, log_pD, log_pL: event probabilities [S]
        transfer_mat: [S, S] linear-space transfer matrix (None when pibar_mode='uniform_approx')
        max_transfer_mat: [S] log2-space column maxima
        device, dtype: target device and float dtype
        local_iters: max iterations per wave self-loop
        local_tolerance: convergence threshold
        fixed_iters: if set, use fixed iteration count (no convergence check / GPU sync)
        overlap_streams: if True, overlap DTS preparation for wave k+1 with
                         self-loop of wave k via a secondary CUDA stream
        pibar_mode: 'dense', 'uniform_approx', 'uniform', or 'topk'
        topk_k: number of top-k entries per clade for pibar_mode='topk' (default 16)

    Returns:
        dict with 'Pi' (in original clade order), 'clade_species_map', 'iterations'
    """
    ccp_helpers = wave_layout['ccp_helpers']
    leaf_row_index = wave_layout['leaf_row_index']
    leaf_col_index = wave_layout['leaf_col_index']
    wave_metas = wave_layout['wave_metas']
    wave_starts = wave_layout['wave_starts']

    C = int(ccp_helpers['C'])
    S = int(species_helpers['S'])

    _PI_INIT = torch.finfo(dtype).min
    Pi = torch.full((C, S), _PI_INIT, dtype=dtype, device=device)
    Pi[leaf_row_index.to(device), leaf_col_index.to(device)] = 0.0
    Pibar = torch.full((C, S), NEG_INF, dtype=dtype, device=device)

    sp_P_idx = species_helpers['s_P_indexes']
    sp_c12_idx = species_helpers['s_C12_indexes']
    p_cpu = sp_P_idx.cpu().long()
    c_cpu = sp_c12_idx.cpu().long()
    mask_c1 = p_cpu < S
    sp_child1_cpu = torch.full((S,), S, dtype=torch.long)
    sp_child2_cpu = torch.full((S,), S, dtype=torch.long)
    sp_child1_cpu[p_cpu[mask_c1]] = c_cpu[mask_c1]
    sp_child2_cpu[p_cpu[~mask_c1] - S] = c_cpu[~mask_c1]
    sp_child1 = sp_child1_cpu.to(device)
    sp_child2 = sp_child2_cpu.to(device)

    # Precompute constant DTS_L terms
    DL_const = 1.0 + log_pD + E
    SL1_const = log_pS + E_s2
    SL2_const = log_pS + E_s1

    ancestors_T_mat = None
    if pibar_mode in ('uniform_approx', 'topk'):
        clade_species_map = None
        leaf_term = None
        transfer_mat_T = transfer_mat.T.contiguous() if pibar_mode == 'topk' else None
        transfer_mat_c = None
    elif pibar_mode == 'uniform':
        anc_dense = species_helpers['ancestors_dense'].to(device=device, dtype=dtype)
        ancestors_T_mat = anc_dense.T.to_sparse_coo()
        clade_species_map = torch.full((C, S), NEG_INF, device=device, dtype=dtype)
        clade_species_map[leaf_row_index, leaf_col_index] = 0.0
        leaf_term = log_pS + clade_species_map
        transfer_mat_T = None
        transfer_mat_c = None
    else:
        clade_species_map = torch.full((C, S), NEG_INF, device=device, dtype=dtype)
        clade_species_map[leaf_row_index, leaf_col_index] = 0.0
        leaf_term = log_pS + clade_species_map
        transfer_mat_T = transfer_mat.T.contiguous()
        transfer_mat_c = transfer_mat.contiguous()

    mt_squeezed = max_transfer_mat.squeeze(-1) if max_transfer_mat.ndim > 1 else max_transfer_mat

    def _get_leaf_wt(ws, we):
        if leaf_term is not None:
            return leaf_term[ws:we]
        W = we - ws
        lwt = torch.full((W, S), NEG_INF, device=device, dtype=dtype)
        mask = (leaf_row_index >= ws) & (leaf_row_index < we)
        if mask.any():
            lwt[leaf_row_index[mask] - ws, leaf_col_index[mask]] = 0.0
        return log_pS + lwt

    use_global_pibar = (S > 256) or pibar_mode in ('uniform_approx', 'uniform', 'topk')
    n_waves = len(wave_metas)

    use_fixed = fixed_iters is not None
    n_iters = fixed_iters if use_fixed else local_iters
    min_warmup = 0 if use_fixed else 3

    total_iters = 0

    with _nvtx_range("Pi wave forward v2"):
        if use_global_pibar:
            prev_tf32 = torch.backends.cuda.matmul.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = True

            if overlap_streams and n_waves > 1:
                stream_main = torch.cuda.current_stream(device)
                stream_prep = torch.cuda.Stream(device=device)

                meta0 = wave_metas[0]
                if meta0['has_splits']:
                    dts_r_current = _compute_dts_cross(
                        Pi, Pibar, meta0, sp_child1, sp_child2,
                        log_pD, log_pS, S, device, dtype)
                else:
                    dts_r_current = None

                dts_r_next = None
                event_prep_done = None

                for wi in range(n_waves):
                    meta = wave_metas[wi]
                    ws = meta['start']
                    we = meta['end']
                    W = meta['W']

                    if wi > 0 and event_prep_done is not None:
                        stream_main.wait_event(event_prep_done)
                        dts_r_current = dts_r_next
                        dts_r_next = None

                    dts_r = dts_r_current
                    leaf_wt = _get_leaf_wt(ws, we)

                    if pibar_mode == 'uniform_approx':
                        for local_iter in range(n_iters):
                            total_iters += 1
                            Pi_new, max_diff = wave_step_uniform_fused(
                                Pi, Pibar, ws, W, S,
                                mt_squeezed, DL_const, Ebar, E, SL1_const, SL2_const,
                                sp_child1, sp_child2, leaf_wt, dts_r,
                            )
                            Pi[ws:we] = Pi_new
                            if not use_fixed and local_iter >= min_warmup:
                                if max_diff < local_tolerance:
                                    break
                    else:
                        for local_iter in range(n_iters):
                            total_iters += 1
                            Pi_W = Pi[ws:we]

                            Pibar_W = _compute_Pibar_inline(Pi_W, transfer_mat_T, mt_squeezed, pibar_mode,
                                                            ancestors_T=ancestors_T_mat,
                                                            topk_k=topk_k)

                            Pi_new = wave_step_fused(
                                Pi_W, Pibar_W,
                                DL_const, Ebar, E, SL1_const, SL2_const,
                                sp_child1, sp_child2, leaf_wt, dts_r,
                            )

                            if not use_fixed and local_iter >= min_warmup:
                                significant = Pi_new > -100.0
                                if not significant.any() or torch.abs(Pi_new - Pi_W)[significant].max().item() < local_tolerance:
                                    Pi[ws:we] = Pi_new
                                    Pibar[ws:we] = Pibar_W
                                    break

                            Pi[ws:we] = Pi_new
                            Pibar[ws:we] = Pibar_W

                    if wi + 1 < n_waves:
                        meta_next = wave_metas[wi + 1]
                        if meta_next['has_splits']:
                            event_self_done = torch.cuda.Event()
                            event_self_done.record(stream_main)
                            with torch.cuda.stream(stream_prep):
                                stream_prep.wait_event(event_self_done)
                                dts_r_next = _compute_dts_cross(
                                    Pi, Pibar, meta_next, sp_child1, sp_child2,
                                    log_pD, log_pS, S, device, dtype)
                                event_prep_done = torch.cuda.Event()
                                event_prep_done.record(stream_prep)
                        else:
                            dts_r_next = None
                            event_prep_done = None
            else:
                for wi in range(n_waves):
                    meta = wave_metas[wi]
                    ws = meta['start']
                    we = meta['end']
                    W = meta['W']

                    if meta['has_splits']:
                        dts_r = _compute_dts_cross(
                            Pi, Pibar, meta, sp_child1, sp_child2,
                            log_pD, log_pS, S, device, dtype)
                    else:
                        dts_r = None

                    leaf_wt = _get_leaf_wt(ws, we)

                    if pibar_mode == 'uniform_approx':
                        for local_iter in range(n_iters):
                            total_iters += 1
                            Pi_new, max_diff = wave_step_uniform_fused(
                                Pi, Pibar, ws, W, S,
                                mt_squeezed, DL_const, Ebar, E, SL1_const, SL2_const,
                                sp_child1, sp_child2, leaf_wt, dts_r,
                            )
                            Pi[ws:we] = Pi_new
                            if not use_fixed and local_iter >= min_warmup:
                                if max_diff < local_tolerance:
                                    break
                    else:
                        for local_iter in range(n_iters):
                            total_iters += 1
                            Pi_W = Pi[ws:we]

                            Pibar_W = _compute_Pibar_inline(Pi_W, transfer_mat_T, mt_squeezed, pibar_mode,
                                                            ancestors_T=ancestors_T_mat,
                                                            topk_k=topk_k)

                            Pi_new = wave_step_fused(
                                Pi_W, Pibar_W,
                                DL_const, Ebar, E, SL1_const, SL2_const,
                                sp_child1, sp_child2, leaf_wt, dts_r,
                            )

                            if not use_fixed and local_iter >= min_warmup:
                                significant = Pi_new > -100.0
                                if not significant.any() or torch.abs(Pi_new - Pi_W)[significant].max().item() < local_tolerance:
                                    Pi[ws:we] = Pi_new
                                    Pibar[ws:we] = Pibar_W
                                    break

                            Pi[ws:we] = Pi_new
                            Pibar[ws:we] = Pibar_W

            torch.backends.cuda.matmul.allow_tf32 = prev_tf32
        else:
            for wi in range(n_waves):
                meta = wave_metas[wi]
                ws = meta['start']
                we = meta['end']
                W = meta['W']

                if meta['has_splits']:
                    sl = meta['sl']
                    sr = meta['sr']
                    lr = torch.cat([sl, sr])
                    dts_r = _compute_DTS_reduced(
                        Pi, Pibar, lr, meta['n_ws'], S, W, log_pD, log_pS,
                        sp_child1, sp_child2, meta['log_split_probs'],
                        meta['reduce_idx'], device, dtype)
                else:
                    dts_r = None

                leaf_wt = _get_leaf_wt(ws, we).contiguous()

                Pibar_W_buf = torch.empty(W, S, device=device, dtype=dtype)

                for local_iter in range(n_iters):
                    total_iters += 1
                    Pi_W = Pi[ws:we]

                    Pi_new = wave_pibar_step_fused(
                        Pi_W, transfer_mat_c, mt_squeezed,
                        DL_const, Ebar, E, SL1_const, SL2_const,
                        sp_child1, sp_child2, leaf_wt,
                        Pibar_W_buf, dts_r,
                    )

                    if not use_fixed and local_iter >= min_warmup:
                        significant = Pi_new > -100.0
                        if not significant.any() or torch.abs(Pi_new - Pi_W)[significant].max().item() < local_tolerance:
                            Pi[ws:we] = Pi_new
                            Pibar[ws:we] = Pibar_W_buf
                            break

                    Pi[ws:we] = Pi_new
                    Pibar[ws:we] = Pibar_W_buf

    perm = wave_layout['perm']
    Pi_orig = Pi[perm]
    clade_species_map_orig = clade_species_map[perm] if clade_species_map is not None else None

    return {
        'Pi': Pi_orig,
        'clade_species_map': clade_species_map_orig,
        'iterations': total_iters,
        'Pi_wave_ordered': Pi,
        'Pibar_wave_ordered': Pibar,
    }
