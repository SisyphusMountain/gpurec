"""Forward pass: Pi_wave_forward and helpers."""
import math
import os

import torch

from .kernels.wave_step import (
    wave_step_uniform_fused,
    wave_step_uniform_fused_into,
    wave_pibar_uniform_parent_fused,
)
from .kernels.dts_fused import dts_fused_parent_reduced
from ._helpers import _nvtx_range
from .extract_parameters import as_family_param, as_family_species

NEG_INF = float("-inf")


# ---------------------------------------------------------------------------
# Cross-clade DTS
# ---------------------------------------------------------------------------

def _compute_dts_cross(Pi, Pibar, meta, sp_child1, sp_child2, log_pD, log_pS,
                       S, device, dtype, active_mask=None,
                       family_idx=None,
                       family_offset=0):
    """Compute DTS cross-clade terms and reduce to [W, S] for one wave."""
    sl = meta['sl']
    sr = meta['sr']
    wlsp = meta['log_split_probs']
    W = meta['W']
    n_eq1 = meta.get('n_eq1', 0)
    n_ge2_clades = meta.get('n_ge2_clades', 0)

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
        family_idx=family_idx,
        family_offset=family_offset,
        tile_splits=64,
        ge2_max_fanout=meta.get('ge2_max_fanout'),
    )


def _get_species_wave_helpers(species_helpers, S, device):
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
                sp_parent,
                int(cache.get('max_ancestor_depth', 0)),
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

    sp_parent_cpu = torch.full((S,), -1, dtype=torch.long)
    sp_parent_cpu[c_cpu[mask_c1]] = p_cpu[mask_c1]
    sp_parent_cpu[c_cpu[~mask_c1]] = p_cpu[~mask_c1] - S

    parent_values = sp_parent_cpu.tolist()
    ancestor_lists = []
    max_ancestor_depth = 0
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

    sp_child1 = sp_child1_cpu.to(device=target_device, dtype=index_dtype)
    sp_child2 = sp_child2_cpu.to(device=target_device, dtype=index_dtype)
    sp_parent = sp_parent_cpu.to(device=target_device, dtype=index_dtype)
    ancestor_cols = ancestor_cols_cpu.T.contiguous().to(device=target_device, dtype=index_dtype)
    ancestor_csr_indptr = torch.tensor(csr_indptr, dtype=torch.int32, device=target_device)
    ancestor_csr_indices = torch.tensor(csr_indices, dtype=torch.int32, device=target_device)
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
    }
    return (
        sp_child1,
        sp_child2,
        sp_parent,
        int(max_ancestor_depth),
    )


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
        transfer_mat: unused in the lean uniform path
        max_transfer_mat: [S] or [G, S] log2-space uniform transfer maxima
        device, dtype: target device and float dtype
        local_iters: max iterations per wave self-loop
        local_tolerance: convergence threshold
        fixed_iters: if set, use fixed iteration count (no convergence check / GPU sync)
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
    target_device = torch.device(device)
    if target_device.type != "cuda":
        raise ValueError("The lean forward path requires CUDA.")
    if leaf_species_index is None:
        raise ValueError("The lean forward path requires leaf_species_index in the wave layout.")

    with _nvtx_range("Pi setup tensors"):
        _PI_INIT = torch.finfo(dtype).min
        Pi = torch.full((C, S), _PI_INIT, dtype=dtype, device=device)
        Pi[leaf_row_index.to(device), leaf_col_index.to(device)] = 0.0
        Pibar = torch.full((C, S), NEG_INF, dtype=dtype, device=device)

    batched = family_idx is not None
    if batched:
        family_idx = family_idx.to(device=device, dtype=torch.long).contiguous()

    if batched:
        family_rows = int(E.shape[0]) if E.ndim == 2 else None
        param_kwargs = dict(S=S, device=target_device, dtype=dtype, family_rows=family_rows)
        E_family = as_family_species(E, name="E", **param_kwargs)
        Ebar_family = as_family_species(Ebar, name="Ebar", **param_kwargs)
        E_s1_family = as_family_species(E_s1, name="E_s1", **param_kwargs)
        E_s2_family = as_family_species(E_s2, name="E_s2", **param_kwargs)
        mt_family = as_family_species(
            max_transfer_mat.squeeze(-1), name="max_transfer_mat", **param_kwargs
        )
        log_pD_param = as_family_param(log_pD, name="log_pD", **param_kwargs)
        log_pS_param = as_family_param(log_pS, name="log_pS", **param_kwargs)
        log_pD_family = as_family_species(log_pD, name="log_pD", **param_kwargs)
        log_pS_family = as_family_species(log_pS, name="log_pS", **param_kwargs)
    else:
        E_family = Ebar_family = E_s1_family = E_s2_family = None
        mt_family = None
        log_pD_param = log_pD
        log_pS_param = log_pS
        log_pD_family = log_pD
        log_pS_family = log_pS

    uniform_pibar_row_max = (
        torch.empty((C,), dtype=dtype, device=device)
        if need_pibar else None
    )

    with _nvtx_range("Pi setup species helpers"):
        (
            sp_child1,
            sp_child2,
            sp_parent,
            max_ancestor_depth,
        ) = _get_species_wave_helpers(species_helpers, S, device)

    with _nvtx_range("Pi setup DTS constants"):
        dl_loss_multiplier = float(os.environ.get("GPUREC_DL_LOSS_MULTIPLIER", "2.0"))
        if batched:
            DL_const = math.log2(dl_loss_multiplier) + log_pD_family + E_family
            SL1_const = log_pS_family + E_s2_family
            SL2_const = log_pS_family + E_s1_family
        else:
            DL_const = math.log2(dl_loss_multiplier) + log_pD + E
            SL1_const = log_pS + E_s2
            SL2_const = log_pS + E_s1

    with _nvtx_range("Pi setup uniform tensors"):
        mt_squeezed = max_transfer_mat.squeeze(-1) if max_transfer_mat.ndim > 1 else max_transfer_mat
        if batched:
            DL_const = DL_const.contiguous()
            SL1_const = SL1_const.contiguous()
            SL2_const = SL2_const.contiguous()
            Ebar_family = Ebar_family.contiguous()
            E_family = E_family.contiguous()
            mt_family = mt_family.contiguous()
            uniform_leaf_logp = log_pS_family.contiguous()
        else:
            uniform_leaf_logp = log_pS.expand(S).contiguous() if log_pS.ndim == 0 else log_pS.contiguous()

    def _wave_consts(ws, we):
        """Return per-wave constants: (DL, SL1, SL2, Ebar_w, E_w, mt_w).

        Batched constants stay in [P, S] form and Triton addresses them
        through family_idx. Shared constants stay in [S] form.
        """
        if batched:
            return DL_const, SL1_const, SL2_const, Ebar_family, E_family, mt_family
        return DL_const, SL1_const, SL2_const, Ebar, E, mt_squeezed

    def _wave_dts_params(meta):
        """Return DTS parameters; genewise scalar params keep zero species stride."""
        return log_pD_param, log_pS_param

    n_waves = len(wave_metas)

    use_fixed = fixed_iters is not None
    n_iters = fixed_iters if use_fixed else local_iters
    min_warmup = 0 if use_fixed else 3
    use_uniform_pingpong = bool(use_fixed and n_iters % 2 == 0)
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

    def _compute_wave_dts(meta, pD_dts, pS_dts):
        return _compute_dts_cross(
            Pi, Pibar, meta, sp_child1, sp_child2,
            pD_dts, pS_dts, S, device, dtype,
            family_idx=family_idx if batched else None,
            family_offset=meta['start'],
        )

    total_iters = 0

    def _run_wave_self_loop(meta, dts_r, leaf_wt, DL_w, SL1_w, SL2_w,
                            Ebar_w, E_w, mt_w):
        nonlocal total_iters
        ws = meta['start']
        we = meta['end']
        W = meta['W']
        for local_iter in range(n_iters):
            total_iters += 1
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
                    family_indexed_consts=batched,
                )
                if (
                    local_iter == n_iters - 1
                    and not _can_skip_final_pibar(ws, we, W)
                ):
                    wave_pibar_uniform_parent_fused(
                        Pi, Pibar, ws, W, S,
                        mt_w, sp_parent, max_ancestor_depth,
                        row_max_out=uniform_pibar_row_max,
                        family_idx=family_idx if batched else None,
                        family_indexed_consts=batched,
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
                    family_indexed_consts=batched,
                    pibar_row_max=uniform_pibar_row_max,
                )

                if compute_diff and max_diff.item() < local_tolerance:
                    Pi[ws:we] = Pi_new
                    break

                Pi[ws:we] = Pi_new

    with _nvtx_range("Pi wave forward v2"):
        for meta in wave_metas:
            if meta['has_splits']:
                pD_dts, pS_dts = _wave_dts_params(meta)
                dts_r = _compute_wave_dts(meta, pD_dts, pS_dts)
            else:
                dts_r = None

            DL_w, SL1_w, SL2_w, Ebar_w, E_w, mt_w = _wave_consts(meta['start'], meta['end'])
            _run_wave_self_loop(
                meta, dts_r, uniform_leaf_logp, DL_w, SL1_w, SL2_w,
                Ebar_w, E_w, mt_w,
            )

    with _nvtx_range("Pi finalize permute"):
        if return_root_rows:
            Pi_root_rows = Pi[wave_layout['root_clade_ids']]
        else:
            Pi_root_rows = None

        if return_original:
            perm = wave_layout['perm']
            Pi_orig = Pi[perm]
        else:
            Pi_orig = None

    Pi_wave_ordered = None if return_root_rows else Pi

    return {
        'Pi': Pi_orig,
        'Pi_root_rows': Pi_root_rows,
        'clade_species_map': None,
        'iterations': total_iters,
        'Pi_wave_ordered': Pi_wave_ordered,
        'Pibar_wave_ordered': Pibar if need_pibar else None,
        'uniform_pibar_row_max': uniform_pibar_row_max if need_pibar else None,
    }
