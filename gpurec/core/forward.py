"""Forward pass: Pi_wave_forward and helpers."""
import os

import torch

from .log2_utils import logsumexp2, logaddexp2
from .log2_utils import _safe_log2_internal as _safe_log2
from .kernels.scatter_lse import seg_logsumexp
from .kernels.wave_step import (
    build_uniform_linear_operator,
    wave_step_fused,
    wave_pibar_step_fused,
    wave_step_uniform_fused,
    wave_step_uniform_fused_into,
    wave_step_uniform_ancestor_fused,
    wave_step_uniform_csr_fused,
    wave_step_uniform_two_kernel_fused,
    wave_step_uniform_linear_fused,
    wave_pibar_uniform_fused,
    wave_pibar_uniform_parent_fused,
)
from .kernels.dts_fused import dts_fused, dts_fused_parent_reduced
from ._logmatmul_compat import (
    HAS_LOGMATMUL as _HAS_LOGMATMUL,
    LogspaceMatmulFn,
    streaming_topk as _streaming_topk,
    logspace_matmul_compressed as _logspace_matmul_compressed,
)
from ._helpers import _safe_exp2_ratio, _seg_logsumexp_host, _nvtx_range, _nvtx_here
from .species import uniform_ancestors_t_from_topology

NEG_INF = float("-inf")


# ---------------------------------------------------------------------------
# Cross-clade DTS
# ---------------------------------------------------------------------------

def _compute_dts_cross(Pi, Pibar, meta, sp_child1, sp_child2, log_pD, log_pS,
                       S, device, dtype, active_mask=None,
                       parent_reduced=False, parent_reduced_min_splits=8192,
                       parent_reduced_impl="tiled",
                       parent_reduced_tile_splits=64):
    """Compute DTS cross-clade terms and reduce to [W, S] for one wave."""
    sl = meta['sl']
    sr = meta['sr']
    wlsp = meta['log_split_probs']
    W = meta['W']
    n_eq1 = meta.get('n_eq1', 0)
    n_ge2_clades = meta.get('n_ge2_clades', 0)

    use_parent_reduced = (
        bool(parent_reduced)
        and (n_eq1 > 0 or n_ge2_clades > 0)
        and sl.numel() >= int(parent_reduced_min_splits)
    )
    if use_parent_reduced:
        empty_ge2_ptr = None
        if n_ge2_clades == 0:
            empty_ge2_ptr = torch.zeros((1,), dtype=torch.long, device=device)
        return dts_fused_parent_reduced(
            Pi, Pibar, sl, sr,
            sp_child1, sp_child2,
            log_pD, log_pS, wlsp,
            W,
            n_eq1,
            meta.get('eq1_reduce_idx', sl[:0]),
            meta.get('ge2_ptr', empty_ge2_ptr),
            meta.get('ge2_parent_ids', sl[:0]),
            active_mask=active_mask,
            impl=parent_reduced_impl,
            tile_splits=parent_reduced_tile_splits,
            ge2_max_fanout=meta.get('ge2_max_fanout'),
        )

    dts_term = dts_fused(
        Pi, Pibar, sl, sr,
        sp_child1, sp_child2,
        log_pD, log_pS, wlsp,
        active_mask=active_mask,
        reduce_idx=meta['reduce_idx'] if active_mask is not None else None,
    )

    NEG_INF = float('-inf')
    dts_r = torch.full((W, S), NEG_INF, device=device, dtype=dtype)

    if n_eq1 > 0:
        dts_r[meta['eq1_reduce_idx'].long()] = dts_term[:n_eq1]

    if n_ge2_clades > 0:
        ge2_term = dts_term[n_eq1:].contiguous()
        y_ge2 = seg_logsumexp(ge2_term, meta['ge2_ptr'])
        dts_r[meta['ge2_parent_ids'].long()] = y_ge2

    return dts_r


# ---------------------------------------------------------------------------
# Pibar inline
# ---------------------------------------------------------------------------

def _compute_Pibar_inline(Pi_W, transfer_mat_T, mt_squeezed, pibar_mode,
                          ancestors_T=None, topk_k=16, family_ids=None):
    """Compute Pibar for a wave, either via dense matmul or uniform approximation.

    Args:
        family_ids: Long[W] wave-local family indices. When provided with a [G, S, S]
            transfer_mat_T, loops over unique families to avoid [W, S, S] allocation.
    """
    Pi_max = Pi_W.max(dim=1, keepdim=True).values
    if pibar_mode == 'uniform':
        Pi_exp = torch.exp2(Pi_W - Pi_max)
        row_sum = Pi_exp.sum(dim=1, keepdim=True)
        # Sparse COO matmul returns non-contiguous output for 2D input — make
        # it contiguous before the subtraction so row_sum - ancestor_sum is exact.
        ancestor_sum = (Pi_exp @ ancestors_T).contiguous()
        # Use safe log2: float32 cancellation can make row_sum - ancestor_sum
        # slightly negative for species with many ancestors; safe log2 returns
        # -inf (zero transfer probability) instead of NaN.
        Pibar_W = _safe_log2(row_sum - ancestor_sum) + Pi_max + mt_squeezed
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
    elif family_ids is not None and transfer_mat_T.ndim == 3:
        # Batched dense: transfer_mat_T is [G, S, S].
        # Loop over unique families to avoid materializing [W, S, S].
        Pi_exp = torch.exp2(Pi_W - Pi_max)
        Pibar_W = torch.empty_like(Pi_W)
        for g in family_ids.unique():
            mask_g = family_ids == g
            Pibar_W[mask_g] = torch.log2(Pi_exp[mask_g] @ transfer_mat_T[g]) + Pi_max[mask_g] + mt_squeezed[mask_g]
    else:
        Pibar_W = torch.log2(torch.exp2(Pi_W - Pi_max) @ transfer_mat_T) + Pi_max + mt_squeezed
    return Pibar_W.contiguous()


def _compute_Pibar_uniform_spmm(Pi_W, ancestors_sparse, mt_squeezed):
    """Compute uniform Pibar via sparse ancestors @ dense Pi_exp.T."""
    Pi_max = Pi_W.max(dim=1, keepdim=True).values
    Pi_exp = torch.exp2(Pi_W - Pi_max)
    row_sum = Pi_exp.sum(dim=1, keepdim=True)
    # cuSPARSE handles the [S, W] RHS much better when it is physically
    # contiguous. For large waves this copy is cheaper than strided CSRMM.
    ancestor_sum = torch.sparse.mm(ancestors_sparse, Pi_exp.T.contiguous()).T.contiguous()
    Pibar_W = _safe_log2(row_sum - ancestor_sum) + Pi_max + mt_squeezed
    return Pibar_W.contiguous()


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
    DTS_max_safe = torch.where(DTS_max == NEG_INF, torch.zeros_like(DTS_max), DTS_max)
    DTS_sum = torch.zeros((W, S), device=device, dtype=dtype)
    DTS_sum.scatter_add_(0, reduce_exp, torch.exp2(DTS_term - DTS_max_safe[reduce_idx]))
    return torch.log2(DTS_sum) + DTS_max


def _get_species_wave_helpers(species_helpers, S, device, use_uniform_fused):
    """Return cached species child/parent helpers for wave kernels."""
    target_device = torch.device(device)
    if target_device.type == 'cuda' and target_device.index is None:
        target_device = torch.device('cuda', torch.cuda.current_device())
    use_topology_int32 = bool(
        target_device.type == 'cuda'
        and int(S) < 2 ** 31
        and os.environ.get("GPUREC_FORWARD_TOPOLOGY_INT32", "1") != "0"
    )
    index_dtype = torch.int32 if use_topology_int32 else torch.long
    cache = species_helpers.get('_wave_forward_species_cache')
    if (
        cache is not None
        and int(cache.get('S', -1)) == int(S)
        and cache.get('index_dtype') == str(index_dtype)
    ):
        sp_child1 = cache.get('sp_child1')
        sp_child2 = cache.get('sp_child2')
        sp_parent = cache.get('sp_parent')
        ancestor_cols = cache.get('ancestor_cols')
        ancestor_csr_indptr = cache.get('ancestor_csr_indptr')
        ancestor_csr_indices = cache.get('ancestor_csr_indices')
        cache_ok = (
            torch.is_tensor(sp_child1)
            and torch.is_tensor(sp_child2)
            and sp_child1.device == target_device
            and sp_child2.device == target_device
        )
        if use_uniform_fused:
            cache_ok = (
                cache_ok
                and torch.is_tensor(sp_parent)
                and sp_parent.device == target_device
                and torch.is_tensor(ancestor_cols)
                and ancestor_cols.device == target_device
                and torch.is_tensor(ancestor_csr_indptr)
                and ancestor_csr_indptr.device == target_device
                and torch.is_tensor(ancestor_csr_indices)
                and ancestor_csr_indices.device == target_device
            )
        if cache_ok:
            return (
                sp_child1,
                sp_child2,
                sp_parent if use_uniform_fused else None,
                int(cache.get('max_ancestor_depth', 0)),
                cache.get('sp_child1_cpu'),
                cache.get('sp_child2_cpu'),
                cache.get('sp_parent_cpu'),
                ancestor_cols if use_uniform_fused else None,
                ancestor_csr_indptr if use_uniform_fused else None,
                ancestor_csr_indices if use_uniform_fused else None,
            )

    sp_P_idx = species_helpers['s_P_indexes']
    sp_c12_idx = species_helpers['s_C12_indexes']
    p_cpu = sp_P_idx.cpu().long()
    c_cpu = sp_c12_idx.cpu().long()
    mask_c1 = p_cpu < S

    sp_child1_cpu = torch.full((S,), S, dtype=torch.long)
    sp_child2_cpu = torch.full((S,), S, dtype=torch.long)
    sp_child1_cpu[p_cpu[mask_c1]] = c_cpu[mask_c1]
    sp_child2_cpu[p_cpu[~mask_c1] - S] = c_cpu[~mask_c1]

    sp_parent_cpu = None
    sp_parent = None
    ancestor_cols = None
    ancestor_cols_cpu = None
    ancestor_csr_indptr = None
    ancestor_csr_indices = None
    ancestor_csr_indptr_cpu = None
    ancestor_csr_indices_cpu = None
    max_ancestor_depth = 0
    if use_uniform_fused:
        sp_parent_cpu = torch.full((S,), -1, dtype=torch.long)
        sp_parent_cpu[c_cpu[mask_c1]] = p_cpu[mask_c1]
        sp_parent_cpu[c_cpu[~mask_c1]] = p_cpu[~mask_c1] - S

        parent_values = sp_parent_cpu.tolist()
        ancestor_lists = []
        for s_idx in range(S):
            depth = 0
            cur = s_idx
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
        csr_indptr = [0]
        csr_indices = []
        for ancestors in ancestor_lists:
            csr_indices.extend(ancestors)
            csr_indptr.append(len(csr_indices))
        ancestor_csr_indptr_cpu = torch.tensor(csr_indptr, dtype=torch.int32)
        ancestor_csr_indices_cpu = torch.tensor(csr_indices, dtype=torch.int32)
        sp_parent = sp_parent_cpu.to(device=target_device, dtype=index_dtype)
        ancestor_cols = ancestor_cols_cpu.T.contiguous().to(device=target_device, dtype=index_dtype)
        ancestor_csr_indptr = ancestor_csr_indptr_cpu.to(target_device)
        ancestor_csr_indices = ancestor_csr_indices_cpu.to(target_device)

    sp_child1 = sp_child1_cpu.to(device=target_device, dtype=index_dtype)
    sp_child2 = sp_child2_cpu.to(device=target_device, dtype=index_dtype)
    species_helpers['_wave_forward_species_cache'] = {
        'S': int(S),
        'index_dtype': str(index_dtype),
        'sp_child1': sp_child1,
        'sp_child2': sp_child2,
        'sp_parent': sp_parent,
        'ancestor_cols': ancestor_cols,
        'ancestor_csr_indptr': ancestor_csr_indptr,
        'ancestor_csr_indices': ancestor_csr_indices,
        'max_ancestor_depth': int(max_ancestor_depth),
        'sp_child1_cpu': sp_child1_cpu,
        'sp_child2_cpu': sp_child2_cpu,
        'sp_parent_cpu': sp_parent_cpu,
    }
    return (
        sp_child1,
        sp_child2,
        sp_parent,
        int(max_ancestor_depth),
        sp_child1_cpu,
        sp_child2_cpu,
        sp_parent_cpu,
        ancestor_cols,
        ancestor_csr_indptr,
        ancestor_csr_indices,
    )


def _ensure_dts_ready_after(wave_layout, wave_metas):
    """Return per-wave cross-DTS child readiness diagnostics.

    dts_ready_after[j] is the largest wave index containing a split child of
    wave j.  A future wave's cross-DTS may be scheduled on another stream only
    after that wave has completed and written final Pi/Pibar rows.
    """
    cached = wave_layout.get('dts_ready_after')
    if (
        isinstance(cached, list)
        and len(cached) == len(wave_metas)
        and all('dts_ready_after' in m for m in wave_metas)
    ):
        return cached

    starts = [int(m['start']) for m in wave_metas]
    if wave_metas:
        starts.append(int(wave_metas[-1]['end']))
    wave_ends_cpu = torch.tensor(starts[1:], dtype=torch.long)
    ready_after = []
    overlap_gap = []
    for wi, meta in enumerate(wave_metas):
        if meta.get('has_splits', False):
            sl_cpu = meta['sl'].detach().cpu().long()
            sr_cpu = meta['sr'].detach().cpu().long()
            left_waves = torch.searchsorted(wave_ends_cpu, sl_cpu, right=True)
            right_waves = torch.searchsorted(wave_ends_cpu, sr_cpu, right=True)
            ready = int(torch.maximum(left_waves, right_waves).max().item())
            if ready >= wi:
                ready = wi - 1
        else:
            ready = -1
        gap = max(0, wi - ready - 1) if ready >= 0 else 0
        meta['dts_ready_after'] = ready
        meta['dts_overlap_gap'] = gap
        ready_after.append(ready)
        overlap_gap.append(gap)
    wave_layout['dts_ready_after'] = ready_after
    wave_layout['dts_overlap_gap'] = overlap_gap
    return ready_after


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
    """Compute gradient bounds for pruning clades from the backward pass.

    Deprecated: no longer called from backward pass. Kept for test coverage.
    """
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
            sl_long = sl.long()
            sr_long = sr.long()
            reduce_idx_long = reduce_idx.long()
            ws = meta['start']

            gb_parents = grad_bound[reduce_idx_long + ws]
            new_l = gb_parents + s[sr_long]
            new_r = gb_parents + s[sl_long]

            grad_bound[sl_long] = logaddexp2(grad_bound[sl_long], new_l)
            grad_bound[sr_long] = logaddexp2(grad_bound[sr_long], new_r)
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
    family_idx: torch.Tensor | None = None,
    return_original: bool = True,
    need_pibar: bool = True,
    return_root_rows: bool = False,
):
    """Wave-based Pi forward pass with wave-ordered layout (v2).

    Clades are permuted so each wave occupies a contiguous block of Pi[ws:we].
    The self-loop uses zero-copy views instead of gather/scatter.

    Args:
        wave_layout: dict from build_wave_layout() containing permuted indices
                     and precomputed per-wave metadata
        species_helpers: species tree helpers dict
        E, Ebar, E_s1, E_s2: converged E vectors [S] or [G, S]
        log_pS, log_pD, log_pL: event probabilities (scalar, [S], [G], or [G, S])
        transfer_mat: [S, S] or [G, S, S] linear-space transfer matrix
                      (None when pibar_mode='uniform')
        max_transfer_mat: [S] or [G, S] log2-space column maxima
        device, dtype: target device and float dtype
        local_iters: max iterations per wave self-loop
        local_tolerance: convergence threshold
        fixed_iters: if set, use fixed iteration count (no convergence check / GPU sync)
        overlap_streams: if True, overlap DTS preparation for wave k+1 with
                         self-loop of wave k via a secondary CUDA stream
        pibar_mode: 'dense', 'uniform', or 'topk'
        topk_k: number of top-k entries per clade for pibar_mode='topk' (default 16)
        family_idx: Long[C] clade→family mapping in wave-ordered space.
                    When provided, parameters are [G, ...] and indexed per-clade.
        need_pibar: if False, do not return final Pibar rows. In fixed even
                    uniform ping-pong mode, root-wave final Pibar recomputation
                    is skipped because no later cross-DTS can consume those rows.
        return_root_rows: if True, gather and return only final root rows as
                          ``Pi_root_rows`` and drop the full wave-ordered Pi
                          reference from the output. This is for inference-only
                          likelihood callers; backward needs the default full
                          wave-ordered Pi/Pibar outputs.

    Returns:
        dict with 'Pi' (in original clade order when requested),
        'Pi_root_rows' when requested, 'clade_species_map', and 'iterations'
    """
    ccp_helpers = wave_layout['ccp_helpers']
    leaf_row_index = wave_layout['leaf_row_index']
    leaf_col_index = wave_layout['leaf_col_index']
    leaf_species_index = wave_layout.get('leaf_species_index')
    wave_metas = wave_layout['wave_metas']
    wave_starts = wave_layout['wave_starts']

    C = int(ccp_helpers['C'])
    S = int(species_helpers['S'])

    with _nvtx_range("Pi setup tensors"):
        _PI_INIT = torch.finfo(dtype).min
        Pi = torch.full((C, S), _PI_INIT, dtype=dtype, device=device)
        Pi[leaf_row_index.to(device), leaf_col_index.to(device)] = 0.0
        Pibar = torch.full((C, S), NEG_INF, dtype=dtype, device=device)

    batched = family_idx is not None
    use_uniform_fused = (pibar_mode == 'uniform' and torch.device(device).type == 'cuda')
    uniform_impl = os.environ.get("GPUREC_UNIFORM_IMPL", "").lower().replace("-", "_")
    use_uniform_csr = (
        use_uniform_fused
        and not batched
        and uniform_impl in (
            "spmm", "sparse", "sparse_mm", "sparse_matmul",
            "csr", "csr_spmm", "triton_spmm", "fused_spmm",
        )
    )
    use_uniform_spmm = (
        use_uniform_fused
        and not batched
        and not use_uniform_csr
        and uniform_impl in ("torch_spmm", "torch_sparse", "torch_sparse_mm")
    )
    use_uniform_linear = (
        use_uniform_fused
        and not use_uniform_csr
        and not use_uniform_spmm
        and not batched
        and (
            os.environ.get("GPUREC_UNIFORM_LINEAR_OPERATOR", "0") == "1"
            or uniform_impl == "linear"
        )
    )
    use_uniform_two_kernel = (
        use_uniform_fused
        and not use_uniform_linear
        and not use_uniform_csr
        and not use_uniform_spmm
        and (
            os.environ.get("GPUREC_UNIFORM_TWO_KERNEL", "0") == "1"
            or uniform_impl in ("two", "two_kernel", "two_kernels")
        )
    )
    use_uniform_ancestor = (
        use_uniform_fused
        and not use_uniform_linear
        and not use_uniform_two_kernel
        and not use_uniform_csr
        and not use_uniform_spmm
        and uniform_impl in ("ancestor", "ancestors", "ancestor_list", "ancestor_table")
    )
    use_uniform_leaf_index = bool(
        use_uniform_fused
        and os.environ.get("GPUREC_FORWARD_LEAF_INDEX", "1") != "0"
        and not use_uniform_linear
        and not use_uniform_two_kernel
        and not use_uniform_spmm
        and leaf_species_index is not None
    )
    reuse_forward_pibar_stats = bool(
        need_pibar
        and
        use_uniform_fused
        and not batched
        and (
            os.environ.get("GPUREC_REUSE_FORWARD_PIBAR_STATS", "0") != "0"
            or os.environ.get("GPUREC_DTS_PIBAR_UD_FUSION", "1") != "0"
        )
    )
    uniform_pibar_row_max = (
        torch.empty((C,), dtype=dtype, device=device)
        if reuse_forward_pibar_stats else None
    )

    sp_parent = None
    ancestor_cols = None
    ancestor_csr_indptr = None
    ancestor_csr_indices = None
    max_ancestor_depth = 0
    with _nvtx_range("Pi setup species helpers"):
        (
            sp_child1,
            sp_child2,
            sp_parent,
            max_ancestor_depth,
            sp_child1_cpu,
            sp_child2_cpu,
            sp_parent_cpu,
            ancestor_cols,
            ancestor_csr_indptr,
            ancestor_csr_indices,
        ) = _get_species_wave_helpers(species_helpers, S, device, use_uniform_fused)

    # Precompute constant DTS_L terms — [S] when shared, [G, S] when batched.
    # When genewise non-specieswise, log_pD/log_pS are [G] (1D) but E is [G, S].
    # Unsqueeze to [G, 1] so broadcasting produces [G, S].
    with _nvtx_range("Pi setup DTS constants"):
        _pD = log_pD.unsqueeze(-1) if batched and log_pD.ndim == 1 else log_pD
        _pS = log_pS.unsqueeze(-1) if batched and log_pS.ndim == 1 else log_pS
        DL_const = 1.0 + _pD + E
        SL1_const = _pS + E_s2
        SL2_const = _pS + E_s1

    ancestors_T_mat = None
    ancestors_spmm_mat = None
    with _nvtx_range("Pi setup pibar mode"):
        if pibar_mode == 'topk':
            clade_species_map = None
            leaf_term = None
            transfer_mat_T = transfer_mat.T.contiguous()
            transfer_mat_c = None
        elif pibar_mode == 'uniform':
            if use_uniform_spmm:
                ancestors_T_mat = uniform_ancestors_t_from_topology(
                    species_helpers,
                    device=device,
                    dtype=dtype,
                )
                ancestors_spmm_mat = ancestors_T_mat.transpose(0, 1).to_sparse_csr()
            elif not use_uniform_fused:
                ancestors_T_mat = uniform_ancestors_t_from_topology(
                    species_helpers,
                    device=device,
                    dtype=dtype,
                )
            if use_uniform_leaf_index:
                clade_species_map = None
                leaf_term = None
            else:
                clade_species_map = torch.full((C, S), NEG_INF, device=device, dtype=dtype)
                clade_species_map[leaf_row_index, leaf_col_index] = 0.0
                leaf_term = None if batched else log_pS + clade_species_map
            transfer_mat_T = None
            transfer_mat_c = None
        else:
            clade_species_map = torch.full((C, S), NEG_INF, device=device, dtype=dtype)
            clade_species_map[leaf_row_index, leaf_col_index] = 0.0
            leaf_term = None if batched else log_pS + clade_species_map
            if transfer_mat is not None:
                if transfer_mat.ndim == 2:
                    transfer_mat_T = transfer_mat.T.contiguous()
                    transfer_mat_c = transfer_mat.contiguous()
                else:
                    # [G, S, S] — precompute transposed version for per-family Pibar
                    transfer_mat_T = transfer_mat.transpose(1, 2).contiguous()
                    transfer_mat_c = None
            else:
                transfer_mat_T = None
                transfer_mat_c = None

    with _nvtx_range("Pi setup uniform linear"):
        mt_squeezed = max_transfer_mat.squeeze(-1) if max_transfer_mat.ndim > 1 else max_transfer_mat
        uniform_leaf_logp = None
        if use_uniform_leaf_index:
            uniform_leaf_logp = log_pS.expand(S).contiguous() if log_pS.ndim == 0 else log_pS.contiguous()
        uniform_linear_op = None
        uniform_linear_debug_guard = os.environ.get("GPUREC_DEBUG_UNIFORM_LINEAR_GUARD", "1") != "0"
        if use_uniform_linear:
            uniform_linear_op = build_uniform_linear_operator(
                DL_const, Ebar, E, SL1_const, SL2_const, mt_squeezed,
                sp_parent_cpu, sp_child1_cpu, sp_child2_cpu,
                device=device, dtype=dtype,
            )

    def _get_leaf_mask(ws, we):
        """Return [W, S] mask with 0.0 at leaf positions, NEG_INF elsewhere."""
        W = we - ws
        lwt = torch.full((W, S), NEG_INF, device=device, dtype=dtype)
        m = (leaf_row_index >= ws) & (leaf_row_index < we)
        if m.any():
            lwt[leaf_row_index[m] - ws, leaf_col_index[m]] = 0.0
        return lwt

    def _get_leaf_wt(ws, we):
        if leaf_term is not None:
            return leaf_term[ws:we]
        lwt = _get_leaf_mask(ws, we)
        if batched:
            # log_pS is [G, S] or [G]; gather per-clade then add leaf mask
            fi_w = family_idx[ws:we]
            log_pS_w = log_pS[fi_w]                     # [W] or [W, S]
            if log_pS_w.ndim == 1:
                log_pS_w = log_pS_w.unsqueeze(-1)       # [W, 1] for broadcast
            return log_pS_w + lwt
        return log_pS + lwt

    def _wave_consts(ws, we):
        """Return per-wave constants: (DL, SL1, SL2, Ebar_w, E_w, mt_w).

        When batched: gathers [W, S] from [G, S] via family_idx.
        When shared: returns the [S] tensors as-is.
        """
        if not batched:
            return DL_const, SL1_const, SL2_const, Ebar, E, mt_squeezed
        fi_w = family_idx[ws:we]
        return (DL_const[fi_w], SL1_const[fi_w], SL2_const[fi_w],
                Ebar[fi_w], E[fi_w], mt_squeezed[fi_w])

    def _wave_dts_params(meta):
        """Return per-split (log_pD, log_pS) for DTS cross-clade computation.

        When batched: gathers per-split params from [G, ...] via family_idx,
        expanding [N_splits] scalars to [N_splits, S] for the kernel.
        When shared: returns the original params.
        """
        if not batched:
            return log_pD, log_pS
        # Each split's parent clade determines the family.
        # reduce_idx maps splits to wave-local clade indices.
        ws = meta['start']
        reduce_idx = meta['reduce_idx'].long()
        fi_splits = family_idx[ws + reduce_idx]
        pD = log_pD[fi_splits]  # [N_splits] or [N_splits, S]
        pS = log_pS[fi_splits]
        # Kernel needs [S] or [N, S]; expand [N] → [N, S]
        if pD.ndim == 1:
            pD = pD.unsqueeze(-1).expand(-1, S).contiguous()
        if pS.ndim == 1:
            pS = pS.unsqueeze(-1).expand(-1, S).contiguous()
        return pD, pS

    use_global_pibar = (S > 256) or pibar_mode in ('uniform', 'topk')
    # Fused pibar+step kernel can't handle per-family [G, S, S] transfer matrices
    if batched and transfer_mat is not None and transfer_mat.ndim == 3:
        use_global_pibar = True
    n_waves = len(wave_metas)

    use_fixed = fixed_iters is not None
    n_iters = fixed_iters if use_fixed else local_iters
    min_warmup = 0 if use_fixed else 3
    forward_dts_overlap_mode = os.environ.get(
        "GPUREC_FORWARD_DTS_OVERLAP_MODE",
        "next" if overlap_streams else "off",
    ).strip().lower()
    if not overlap_streams:
        forward_dts_overlap_mode = "off"
    if forward_dts_overlap_mode in ("0", "false", "none"):
        forward_dts_overlap_mode = "off"
    if forward_dts_overlap_mode in ("1", "true"):
        forward_dts_overlap_mode = "next"
    if forward_dts_overlap_mode in ("lookahead", "readiness"):
        forward_dts_overlap_mode = "ready"
    forward_dts_overlap_max_pending = max(
        1, int(os.environ.get("GPUREC_FORWARD_DTS_OVERLAP_MAX_PENDING", "2"))
    )
    use_uniform_pingpong = bool(
        use_uniform_fused
        and use_uniform_leaf_index
        and os.environ.get("GPUREC_UNIFORM_PINGPONG", "1") != "0"
        and use_fixed
        and n_iters % 2 == 0
        and not use_uniform_linear
        and not use_uniform_two_kernel
        and not use_uniform_ancestor
        and not use_uniform_csr
        and not use_uniform_spmm
    )
    root_clade_ids_for_skip = None
    if use_uniform_pingpong and not need_pibar:
        root_clade_ids_for_skip = wave_layout.get('root_clade_ids_cpu')
        if root_clade_ids_for_skip is None:
            root_clade_ids_for_skip = [
                int(r) for r in wave_layout['root_clade_ids'].detach().cpu().tolist()
            ]

    def _can_skip_final_pibar(ws: int, we: int, W: int) -> bool:
        if root_clade_ids_for_skip is None:
            return False
        roots_in_wave = 0
        for root_id in root_clade_ids_for_skip:
            if ws <= root_id < we:
                roots_in_wave += 1
        return roots_in_wave == W

    use_forward_parent_reduced_dts = bool(
        use_uniform_fused
        and os.environ.get("GPUREC_FORWARD_PARENT_REDUCED_DTS", "1") != "0"
    )
    forward_parent_reduced_dts_min_splits = int(
        os.environ.get("GPUREC_FORWARD_PARENT_REDUCED_DTS_MIN_SPLITS", "0")
    )
    forward_parent_reduced_dts_impl = os.environ.get(
        "GPUREC_FORWARD_PARENT_REDUCED_DTS_IMPL", "tiled"
    )
    forward_parent_reduced_dts_tile_splits = int(
        os.environ.get("GPUREC_FORWARD_PARENT_REDUCED_DTS_TILE_SPLITS", "64")
    )
    forward_parent_reduced_dts_ge2_only = (
        os.environ.get("GPUREC_FORWARD_PARENT_REDUCED_DTS_GE2_ONLY", "1") != "0"
    )

    def _compute_wave_dts(meta, pD_dts, pS_dts):
        parent_reduced_wave = (
            use_forward_parent_reduced_dts
            and (
                not forward_parent_reduced_dts_ge2_only
                or meta.get('n_ge2_clades', 0) > 0
            )
        )
        return _compute_dts_cross(
            Pi, Pibar, meta, sp_child1, sp_child2,
            pD_dts, pS_dts, S, device, dtype,
            parent_reduced=parent_reduced_wave,
            parent_reduced_min_splits=forward_parent_reduced_dts_min_splits,
            parent_reduced_impl=forward_parent_reduced_dts_impl,
            parent_reduced_tile_splits=forward_parent_reduced_dts_tile_splits,
        )

    total_iters = 0

    def _run_wave_self_loop(meta, dts_r, leaf_wt, DL_w, SL1_w, SL2_w,
                            Ebar_w, E_w, mt_w, tm_T_w, fi_w_dense):
        nonlocal total_iters
        ws = meta['start']
        we = meta['end']
        W = meta['W']
        for local_iter in range(n_iters):
            total_iters += 1
            Pi_W = Pi[ws:we]

            if use_uniform_linear:
                compute_diff = not use_fixed and local_iter >= min_warmup
                Pi_new, max_diff = wave_step_uniform_linear_fused(
                    Pi, ws, W, S,
                    uniform_linear_op['op_cols'],
                    uniform_linear_op['op_vals'],
                    uniform_linear_op['v_scaled'],
                    uniform_linear_op['row_scale'],
                    leaf_wt, dts_r,
                    compute_diff=compute_diff,
                    debug_guard=uniform_linear_debug_guard,
                )

                if compute_diff and max_diff.item() < local_tolerance:
                    Pi[ws:we] = Pi_new
                    break

                Pi[ws:we] = Pi_new
            elif use_uniform_two_kernel:
                compute_diff = not use_fixed and local_iter >= min_warmup
                Pi_new, max_diff = wave_step_uniform_two_kernel_fused(
                    Pi, Pibar, ws, W, S,
                    mt_w, DL_w, Ebar_w, E_w, SL1_w, SL2_w,
                    sp_child1, sp_child2, sp_parent, max_ancestor_depth,
                    leaf_wt, dts_r,
                    compute_diff=compute_diff,
                )

                if compute_diff and max_diff.item() < local_tolerance:
                    Pi[ws:we] = Pi_new
                    break

                Pi[ws:we] = Pi_new
            elif use_uniform_ancestor:
                compute_diff = not use_fixed and local_iter >= min_warmup
                Pi_new, max_diff = wave_step_uniform_ancestor_fused(
                    Pi, Pibar, ws, W, S,
                    mt_w, DL_w, Ebar_w, E_w, SL1_w, SL2_w,
                    sp_child1, sp_child2, ancestor_cols,
                    leaf_wt, dts_r,
                    compute_diff=compute_diff,
                    leaf_species_idx=leaf_species_index,
                    leaf_logp=uniform_leaf_logp,
                    family_idx=family_idx if batched else None,
                    pibar_row_max=uniform_pibar_row_max,
                )

                if compute_diff and max_diff.item() < local_tolerance:
                    Pi[ws:we] = Pi_new
                    break

                Pi[ws:we] = Pi_new
            elif use_uniform_csr:
                compute_diff = not use_fixed and local_iter >= min_warmup
                Pi_new, max_diff = wave_step_uniform_csr_fused(
                    Pi, Pibar, ws, W, S,
                    mt_w, DL_w, Ebar_w, E_w, SL1_w, SL2_w,
                    sp_child1, sp_child2, ancestor_csr_indptr,
                    ancestor_csr_indices, max_ancestor_depth,
                    leaf_wt, dts_r,
                    compute_diff=compute_diff,
                    leaf_species_idx=leaf_species_index,
                    leaf_logp=uniform_leaf_logp,
                    family_idx=family_idx if batched else None,
                    pibar_row_max=uniform_pibar_row_max,
                )

                if compute_diff and max_diff.item() < local_tolerance:
                    Pi[ws:we] = Pi_new
                    break

                Pi[ws:we] = Pi_new
            elif use_uniform_spmm:
                Pibar_W = _compute_Pibar_uniform_spmm(
                    Pi_W,
                    ancestors_spmm_mat,
                    mt_w,
                )

                Pi_new = wave_step_fused(
                    Pi_W, Pibar_W,
                    DL_w, Ebar_w, E_w, SL1_w, SL2_w,
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
            elif use_uniform_fused:
                if use_uniform_pingpong:
                    pi_in = Pi if (local_iter % 2 == 0) else Pibar
                    pi_out = Pibar if (local_iter % 2 == 0) else Pi
                    wave_step_uniform_fused_into(
                        pi_in, pi_out, Pibar, ws, W, S,
                        mt_w, DL_w, Ebar_w, E_w, SL1_w, SL2_w,
                        sp_child1, sp_child2, sp_parent, max_ancestor_depth,
                        leaf_wt, dts_r,
                        leaf_species_idx=leaf_species_index,
                        leaf_logp=uniform_leaf_logp,
                        family_idx=family_idx if batched else None,
                    )
                    if (
                        local_iter == n_iters - 1
                        and not _can_skip_final_pibar(ws, we, W)
                    ):
                        wave_pibar_uniform_parent_fused(
                            Pi, Pibar, ws, W, S,
                            mt_w, sp_parent, max_ancestor_depth,
                            row_max_out=uniform_pibar_row_max,
                        )
                else:
                    compute_diff = not use_fixed and local_iter >= min_warmup
                    Pi_new, max_diff = wave_step_uniform_fused(
                        Pi, Pibar, ws, W, S,
                        mt_w, DL_w, Ebar_w, E_w, SL1_w, SL2_w,
                        sp_child1, sp_child2, sp_parent, max_ancestor_depth,
                        leaf_wt, dts_r,
                        compute_diff=compute_diff,
                        leaf_species_idx=leaf_species_index,
                        leaf_logp=uniform_leaf_logp,
                        family_idx=family_idx if batched else None,
                        pibar_row_max=uniform_pibar_row_max,
                    )

                    if compute_diff and max_diff.item() < local_tolerance:
                        Pi[ws:we] = Pi_new
                        break

                    Pi[ws:we] = Pi_new
            else:
                Pibar_W = _compute_Pibar_inline(Pi_W, tm_T_w, mt_w, pibar_mode,
                                                ancestors_T=ancestors_T_mat,
                                                topk_k=topk_k,
                                                family_ids=fi_w_dense)

                Pi_new = wave_step_fused(
                    Pi_W, Pibar_W,
                    DL_w, Ebar_w, E_w, SL1_w, SL2_w,
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

        if use_uniform_linear:
            wave_pibar_uniform_fused(
                Pi, Pibar, ws, W, S,
                mt_w,
                uniform_linear_op['ancestor_cols'],
                row_max_out=uniform_pibar_row_max,
            )

    with _nvtx_range("Pi wave forward v2"):
        if use_global_pibar:
            prev_tf32 = torch.backends.cuda.matmul.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = True

            if (
                overlap_streams
                and forward_dts_overlap_mode == "ready"
                and n_waves > 1
                and use_fixed
            ):
                stream_main = torch.cuda.current_stream(device)
                stream_prep = torch.cuda.Stream(device=device)
                dts_ready_after = _ensure_dts_ready_after(wave_layout, wave_metas)
                dts_results = [None] * n_waves
                dts_events = [None] * n_waves
                dts_scheduled = [False] * n_waves

                def _pending_dts_count():
                    return sum(x is not None for x in dts_results)

                def _schedule_ready_dts(completed_wave_idx: int, current_wave_idx: int):
                    if completed_wave_idx < 0:
                        return
                    pending = _pending_dts_count()
                    if pending >= forward_dts_overlap_max_pending:
                        return
                    ready_event = torch.cuda.Event()
                    ready_event.record(stream_main)
                    for future_wi in range(current_wave_idx + 1, n_waves):
                        if pending >= forward_dts_overlap_max_pending:
                            break
                        if dts_scheduled[future_wi]:
                            continue
                        meta_future = wave_metas[future_wi]
                        if not meta_future.get('has_splits', False):
                            continue
                        if int(dts_ready_after[future_wi]) > completed_wave_idx:
                            continue
                        with torch.cuda.stream(stream_prep):
                            stream_prep.wait_event(ready_event)
                            pD_future, pS_future = _wave_dts_params(meta_future)
                            dts_results[future_wi] = _compute_wave_dts(
                                meta_future, pD_future, pS_future)
                            done = torch.cuda.Event()
                            done.record(stream_prep)
                            dts_events[future_wi] = done
                        dts_scheduled[future_wi] = True
                        pending += 1

                for wi in range(n_waves):
                    meta = wave_metas[wi]
                    ws = meta['start']
                    we = meta['end']

                    if meta['has_splits']:
                        if dts_scheduled[wi]:
                            stream_main.wait_event(dts_events[wi])
                            dts_r = dts_results[wi]
                            dts_events[wi] = None
                        else:
                            pD_dts, pS_dts = _wave_dts_params(meta)
                            dts_r = _compute_wave_dts(meta, pD_dts, pS_dts)
                    else:
                        dts_r = None

                    _schedule_ready_dts(wi - 1, wi)

                    leaf_wt = uniform_leaf_logp if use_uniform_leaf_index else _get_leaf_wt(ws, we)
                    DL_w, SL1_w, SL2_w, Ebar_w, E_w, mt_w = _wave_consts(ws, we)

                    tm_T_w = transfer_mat_T
                    fi_w_dense = None
                    if batched and transfer_mat_T is not None and transfer_mat_T.ndim == 3:
                        fi_w_dense = family_idx[ws:we]

                    _run_wave_self_loop(
                        meta, dts_r, leaf_wt, DL_w, SL1_w, SL2_w,
                        Ebar_w, E_w, mt_w, tm_T_w, fi_w_dense,
                    )
                    dts_results[wi] = None
                    dts_events[wi] = None
                    _schedule_ready_dts(wi, wi)
            elif (
                overlap_streams
                and forward_dts_overlap_mode != "off"
                and n_waves > 1
            ):
                stream_main = torch.cuda.current_stream(device)
                stream_prep = torch.cuda.Stream(device=device)

                meta0 = wave_metas[0]
                if meta0['has_splits']:
                    pD_dts0, pS_dts0 = _wave_dts_params(meta0)
                    dts_r_current = _compute_wave_dts(meta0, pD_dts0, pS_dts0)
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
                    leaf_wt = uniform_leaf_logp if use_uniform_leaf_index else _get_leaf_wt(ws, we)
                    DL_w, SL1_w, SL2_w, Ebar_w, E_w, mt_w = _wave_consts(ws, we)

                    tm_T_w = transfer_mat_T
                    fi_w_dense = None
                    if batched and transfer_mat_T is not None and transfer_mat_T.ndim == 3:
                        fi_w_dense = family_idx[ws:we]

                    for local_iter in range(n_iters):
                        total_iters += 1
                        Pi_W = Pi[ws:we]

                        if use_uniform_linear:
                            compute_diff = not use_fixed and local_iter >= min_warmup
                            Pi_new, max_diff = wave_step_uniform_linear_fused(
                                Pi, ws, W, S,
                                uniform_linear_op['op_cols'],
                                uniform_linear_op['op_vals'],
                                uniform_linear_op['v_scaled'],
                                uniform_linear_op['row_scale'],
                                leaf_wt, dts_r,
                                compute_diff=compute_diff,
                                debug_guard=uniform_linear_debug_guard,
                            )

                            if compute_diff and max_diff.item() < local_tolerance:
                                Pi[ws:we] = Pi_new
                                break

                            Pi[ws:we] = Pi_new
                        elif use_uniform_two_kernel:
                            compute_diff = not use_fixed and local_iter >= min_warmup
                            Pi_new, max_diff = wave_step_uniform_two_kernel_fused(
                                Pi, Pibar, ws, W, S,
                                mt_w, DL_w, Ebar_w, E_w, SL1_w, SL2_w,
                                sp_child1, sp_child2, sp_parent, max_ancestor_depth,
                                leaf_wt, dts_r,
                                compute_diff=compute_diff,
                            )

                            if compute_diff and max_diff.item() < local_tolerance:
                                Pi[ws:we] = Pi_new
                                break

                            Pi[ws:we] = Pi_new
                        elif use_uniform_ancestor:
                            compute_diff = not use_fixed and local_iter >= min_warmup
                            Pi_new, max_diff = wave_step_uniform_ancestor_fused(
                                Pi, Pibar, ws, W, S,
                                mt_w, DL_w, Ebar_w, E_w, SL1_w, SL2_w,
                                sp_child1, sp_child2, ancestor_cols,
                                leaf_wt, dts_r,
                                compute_diff=compute_diff,
                                leaf_species_idx=leaf_species_index,
                                leaf_logp=uniform_leaf_logp,
                                family_idx=family_idx if batched else None,
                                pibar_row_max=uniform_pibar_row_max,
                            )

                            if compute_diff and max_diff.item() < local_tolerance:
                                Pi[ws:we] = Pi_new
                                break

                            Pi[ws:we] = Pi_new
                        elif use_uniform_csr:
                            compute_diff = not use_fixed and local_iter >= min_warmup
                            Pi_new, max_diff = wave_step_uniform_csr_fused(
                                Pi, Pibar, ws, W, S,
                                mt_w, DL_w, Ebar_w, E_w, SL1_w, SL2_w,
                                sp_child1, sp_child2, ancestor_csr_indptr,
                                ancestor_csr_indices, max_ancestor_depth,
                                leaf_wt, dts_r,
                                compute_diff=compute_diff,
                                leaf_species_idx=leaf_species_index,
                                leaf_logp=uniform_leaf_logp,
                                family_idx=family_idx if batched else None,
                                pibar_row_max=uniform_pibar_row_max,
                            )

                            if compute_diff and max_diff.item() < local_tolerance:
                                Pi[ws:we] = Pi_new
                                break

                            Pi[ws:we] = Pi_new
                        elif use_uniform_spmm:
                            Pibar_W = _compute_Pibar_uniform_spmm(
                                Pi_W,
                                ancestors_spmm_mat,
                                mt_w,
                            )

                            Pi_new = wave_step_fused(
                                Pi_W, Pibar_W,
                                DL_w, Ebar_w, E_w, SL1_w, SL2_w,
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
                        elif use_uniform_fused:
                            if use_uniform_pingpong:
                                pi_in = Pi if (local_iter % 2 == 0) else Pibar
                                pi_out = Pibar if (local_iter % 2 == 0) else Pi
                                wave_step_uniform_fused_into(
                                    pi_in, pi_out, Pibar, ws, W, S,
                                    mt_w, DL_w, Ebar_w, E_w, SL1_w, SL2_w,
                                    sp_child1, sp_child2, sp_parent, max_ancestor_depth,
                                    leaf_wt, dts_r,
                                    leaf_species_idx=leaf_species_index,
                                    leaf_logp=uniform_leaf_logp,
                                    family_idx=family_idx if batched else None,
                                )
                                if (
                                    local_iter == n_iters - 1
                                    and not _can_skip_final_pibar(ws, we, W)
                                ):
                                    wave_pibar_uniform_parent_fused(
                                        Pi, Pibar, ws, W, S,
                                        mt_w, sp_parent, max_ancestor_depth,
                                        row_max_out=uniform_pibar_row_max,
                                    )
                            else:
                                compute_diff = not use_fixed and local_iter >= min_warmup
                                Pi_new, max_diff = wave_step_uniform_fused(
                                    Pi, Pibar, ws, W, S,
                                    mt_w, DL_w, Ebar_w, E_w, SL1_w, SL2_w,
                                    sp_child1, sp_child2, sp_parent, max_ancestor_depth,
                                    leaf_wt, dts_r,
                                    compute_diff=compute_diff,
                                    leaf_species_idx=leaf_species_index,
                                    leaf_logp=uniform_leaf_logp,
                                    family_idx=family_idx if batched else None,
                                    pibar_row_max=uniform_pibar_row_max,
                                )

                                if compute_diff and max_diff.item() < local_tolerance:
                                    Pi[ws:we] = Pi_new
                                    break

                                Pi[ws:we] = Pi_new
                        else:
                            Pibar_W = _compute_Pibar_inline(Pi_W, tm_T_w, mt_w, pibar_mode,
                                                            ancestors_T=ancestors_T_mat,
                                                            topk_k=topk_k,
                                                            family_ids=fi_w_dense)

                            Pi_new = wave_step_fused(
                                Pi_W, Pibar_W,
                                DL_w, Ebar_w, E_w, SL1_w, SL2_w,
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

                    if use_uniform_linear:
                        wave_pibar_uniform_fused(
                            Pi, Pibar, ws, W, S,
                            mt_w,
                            uniform_linear_op['ancestor_cols'],
                            row_max_out=uniform_pibar_row_max,
                        )

                    if wi + 1 < n_waves:
                        meta_next = wave_metas[wi + 1]
                        if meta_next['has_splits']:
                            pD_next, pS_next = _wave_dts_params(meta_next)
                            event_self_done = torch.cuda.Event()
                            event_self_done.record(stream_main)
                            with torch.cuda.stream(stream_prep):
                                stream_prep.wait_event(event_self_done)
                                dts_r_next = _compute_wave_dts(
                                    meta_next, pD_next, pS_next)
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
                        pD_dts, pS_dts = _wave_dts_params(meta)
                        dts_r = _compute_wave_dts(meta, pD_dts, pS_dts)
                    else:
                        dts_r = None

                    leaf_wt = uniform_leaf_logp if use_uniform_leaf_index else _get_leaf_wt(ws, we)
                    DL_w, SL1_w, SL2_w, Ebar_w, E_w, mt_w = _wave_consts(ws, we)

                    tm_T_w = transfer_mat_T
                    fi_w_dense = None
                    if batched and transfer_mat_T is not None and transfer_mat_T.ndim == 3:
                        fi_w_dense = family_idx[ws:we]

                    for local_iter in range(n_iters):
                        total_iters += 1
                        Pi_W = Pi[ws:we]

                        if use_uniform_linear:
                            compute_diff = not use_fixed and local_iter >= min_warmup
                            Pi_new, max_diff = wave_step_uniform_linear_fused(
                                Pi, ws, W, S,
                                uniform_linear_op['op_cols'],
                                uniform_linear_op['op_vals'],
                                uniform_linear_op['v_scaled'],
                                uniform_linear_op['row_scale'],
                                leaf_wt, dts_r,
                                compute_diff=compute_diff,
                                debug_guard=uniform_linear_debug_guard,
                            )

                            if compute_diff and max_diff.item() < local_tolerance:
                                Pi[ws:we] = Pi_new
                                break

                            Pi[ws:we] = Pi_new
                        elif use_uniform_two_kernel:
                            compute_diff = not use_fixed and local_iter >= min_warmup
                            Pi_new, max_diff = wave_step_uniform_two_kernel_fused(
                                Pi, Pibar, ws, W, S,
                                mt_w, DL_w, Ebar_w, E_w, SL1_w, SL2_w,
                                sp_child1, sp_child2, sp_parent, max_ancestor_depth,
                                leaf_wt, dts_r,
                                compute_diff=compute_diff,
                            )

                            if compute_diff and max_diff.item() < local_tolerance:
                                Pi[ws:we] = Pi_new
                                break

                            Pi[ws:we] = Pi_new
                        elif use_uniform_ancestor:
                            compute_diff = not use_fixed and local_iter >= min_warmup
                            Pi_new, max_diff = wave_step_uniform_ancestor_fused(
                                Pi, Pibar, ws, W, S,
                                mt_w, DL_w, Ebar_w, E_w, SL1_w, SL2_w,
                                sp_child1, sp_child2, ancestor_cols,
                                leaf_wt, dts_r,
                                compute_diff=compute_diff,
                                leaf_species_idx=leaf_species_index,
                                leaf_logp=uniform_leaf_logp,
                                family_idx=family_idx if batched else None,
                                pibar_row_max=uniform_pibar_row_max,
                            )

                            if compute_diff and max_diff.item() < local_tolerance:
                                Pi[ws:we] = Pi_new
                                break

                            Pi[ws:we] = Pi_new
                        elif use_uniform_csr:
                            compute_diff = not use_fixed and local_iter >= min_warmup
                            Pi_new, max_diff = wave_step_uniform_csr_fused(
                                Pi, Pibar, ws, W, S,
                                mt_w, DL_w, Ebar_w, E_w, SL1_w, SL2_w,
                                sp_child1, sp_child2, ancestor_csr_indptr,
                                ancestor_csr_indices, max_ancestor_depth,
                                leaf_wt, dts_r,
                                compute_diff=compute_diff,
                                leaf_species_idx=leaf_species_index,
                                leaf_logp=uniform_leaf_logp,
                                family_idx=family_idx if batched else None,
                                pibar_row_max=uniform_pibar_row_max,
                            )

                            if compute_diff and max_diff.item() < local_tolerance:
                                Pi[ws:we] = Pi_new
                                break

                            Pi[ws:we] = Pi_new
                        elif use_uniform_spmm:
                            Pibar_W = _compute_Pibar_uniform_spmm(
                                Pi_W,
                                ancestors_spmm_mat,
                                mt_w,
                            )

                            Pi_new = wave_step_fused(
                                Pi_W, Pibar_W,
                                DL_w, Ebar_w, E_w, SL1_w, SL2_w,
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
                        elif use_uniform_fused:
                            if use_uniform_pingpong:
                                pi_in = Pi if (local_iter % 2 == 0) else Pibar
                                pi_out = Pibar if (local_iter % 2 == 0) else Pi
                                wave_step_uniform_fused_into(
                                    pi_in, pi_out, Pibar, ws, W, S,
                                    mt_w, DL_w, Ebar_w, E_w, SL1_w, SL2_w,
                                    sp_child1, sp_child2, sp_parent, max_ancestor_depth,
                                    leaf_wt, dts_r,
                                    leaf_species_idx=leaf_species_index,
                                    leaf_logp=uniform_leaf_logp,
                                    family_idx=family_idx if batched else None,
                                )
                                if (
                                    local_iter == n_iters - 1
                                    and not _can_skip_final_pibar(ws, we, W)
                                ):
                                    wave_pibar_uniform_parent_fused(
                                        Pi, Pibar, ws, W, S,
                                        mt_w, sp_parent, max_ancestor_depth,
                                        row_max_out=uniform_pibar_row_max,
                                    )
                            else:
                                compute_diff = not use_fixed and local_iter >= min_warmup
                                Pi_new, max_diff = wave_step_uniform_fused(
                                    Pi, Pibar, ws, W, S,
                                    mt_w, DL_w, Ebar_w, E_w, SL1_w, SL2_w,
                                    sp_child1, sp_child2, sp_parent, max_ancestor_depth,
                                    leaf_wt, dts_r,
                                    compute_diff=compute_diff,
                                    leaf_species_idx=leaf_species_index,
                                    leaf_logp=uniform_leaf_logp,
                                    family_idx=family_idx if batched else None,
                                    pibar_row_max=uniform_pibar_row_max,
                                )

                                if compute_diff and max_diff.item() < local_tolerance:
                                    Pi[ws:we] = Pi_new
                                    break

                                Pi[ws:we] = Pi_new
                        else:
                            Pibar_W = _compute_Pibar_inline(Pi_W, tm_T_w, mt_w, pibar_mode,
                                                            ancestors_T=ancestors_T_mat,
                                                            topk_k=topk_k,
                                                            family_ids=fi_w_dense)

                            Pi_new = wave_step_fused(
                                Pi_W, Pibar_W,
                                DL_w, Ebar_w, E_w, SL1_w, SL2_w,
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

                    if use_uniform_linear:
                        wave_pibar_uniform_fused(
                            Pi, Pibar, ws, W, S,
                            mt_w,
                            uniform_linear_op['ancestor_cols'],
                            row_max_out=uniform_pibar_row_max,
                        )

            torch.backends.cuda.matmul.allow_tf32 = prev_tf32
        else:
            for wi in range(n_waves):
                meta = wave_metas[wi]
                ws = meta['start']
                we = meta['end']
                W = meta['W']

                DL_w, SL1_w, SL2_w, Ebar_w, E_w, mt_w = _wave_consts(ws, we)

                if meta['has_splits']:
                    sl = meta['sl']
                    sr = meta['sr']
                    sl_long = sl.long()
                    sr_long = sr.long()
                    lr = torch.cat([sl_long, sr_long])
                    pD_dts, pS_dts = _wave_dts_params(meta)
                    dts_r = _compute_DTS_reduced(
                        Pi, Pibar, lr, meta['n_ws'], S, W, pD_dts, pS_dts,
                        sp_child1, sp_child2, meta['log_split_probs'],
                        meta['reduce_idx'].long(), device, dtype)
                else:
                    dts_r = None

                leaf_wt = _get_leaf_wt(ws, we).contiguous()

                # For dense+batched: the fused pibar+step kernel doesn't support
                # per-family transfer matrices, so fall back to separate Pibar + step.
                tm_c_w = transfer_mat_c

                Pibar_W_buf = torch.empty(W, S, device=device, dtype=dtype)

                for local_iter in range(n_iters):
                    total_iters += 1
                    Pi_W = Pi[ws:we]

                    Pi_new = wave_pibar_step_fused(
                        Pi_W, tm_c_w, mt_w,
                        DL_w, Ebar_w, E_w, SL1_w, SL2_w,
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

    with _nvtx_range("Pi finalize permute"):
        if return_root_rows:
            Pi_root_rows = Pi[wave_layout['root_clade_ids']]
        else:
            Pi_root_rows = None

        if return_original:
            perm = wave_layout['perm']
            Pi_orig = Pi[perm]
            clade_species_map_orig = clade_species_map[perm] if clade_species_map is not None else None
        else:
            Pi_orig = None
            clade_species_map_orig = None

    Pi_wave_ordered = None if return_root_rows else Pi

    return {
        'Pi': Pi_orig,
        'Pi_root_rows': Pi_root_rows,
        'clade_species_map': clade_species_map_orig,
        'iterations': total_iters,
        'Pi_wave_ordered': Pi_wave_ordered,
        'Pibar_wave_ordered': Pibar if need_pibar else None,
        'uniform_pibar_row_max': uniform_pibar_row_max if need_pibar else None,
    }
