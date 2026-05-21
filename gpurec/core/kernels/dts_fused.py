"""Fused DTS computation: gather + 5 terms + logsumexp in one Triton kernel."""

import os

import torch
import triton
import triton.language as tl


def _tl_float_dtype(dtype):
    return tl.float64 if dtype == torch.float64 else tl.float32


def _prepare_param(p, n_splits, S, *, family_indexed=False):
    """Normalize direct DTS forward parameters and return addressing metadata.

    With family indexing enabled, a 1-D tensor with ``numel() == S`` is treated
    as a shared species vector.  Public model callers normalize genewise scalar
    rows to ``[G, 1]`` before this helper.  Direct callers should do the same
    when ``G == S`` and use ``[G, S]`` for family/species rows.
    """
    if family_indexed:
        if p.dim() == 0:
            return p.reshape(1), 4, 0, 0
        if p.dim() == 1:
            if p.numel() == S:
                return p, 4, 0, int(p.stride(0))
            return p, 4, int(p.stride(0)), 0
        if p.dim() == 2:
            if p.shape[1] == 1:
                return p, 4, int(p.stride(0)), 0
            if p.shape[1] == S:
                row_stride = 0 if int(p.shape[0]) == 1 else int(p.stride(0))
                return p, 4, row_stride, int(p.stride(1))
        raise ValueError(
            "family-indexed DTS parameters must be [G], [G, 1], or [G, S]; "
            f"got shape {tuple(p.shape)} with S={S}"
        )
    if p.dim() == 0:
        return p.expand(S).contiguous(), 0, 0, 1
    if p.dim() == 1:
        if p.numel() == S:
            return p.contiguous(), 0, 0, 1
    raise ValueError(
        "DTS parameters must be scalar, [S], [G], [G, 1], or [G, S]; "
        f"got shape {tuple(p.shape)} with N={n_splits}, S={S}"
    )


@triton.jit
def _load_dts_param(param_ptr, n, s_offs, family, S: tl.constexpr, mask,
                    mode: tl.constexpr, ROW_STRIDE: tl.constexpr,
                    SPECIES_STRIDE: tl.constexpr,
                    BLOCK_S: tl.constexpr, DTYPE: tl.constexpr):
    # Modes: 0 shared [S], 4 family-indexed with explicit row/species strides.
    if mode == 4:
        return tl.load(param_ptr + family * ROW_STRIDE + s_offs * SPECIES_STRIDE, mask=mask, other=-1e30)
    return tl.load(param_ptr + s_offs, mask=mask, other=-1e30)



@triton.jit
def _dts_eq1_to_rows_kernel(
    Pi_ptr, Pibar_ptr,
    lefts_ptr, rights_ptr,
    sp_child1_ptr, sp_child2_ptr,
    log_pD_ptr,
    log_pS_ptr,
    log_split_probs_ptr,
    eq1_parent_ids_ptr,
    active_mask_ptr,
    out_ptr,
    family_idx_ptr,
    family_offset,
    n_eq1: tl.constexpr,
    S: tl.constexpr,
    BLOCK_S: tl.constexpr,
    mode_pD: tl.constexpr = 0,
    mode_pS: tl.constexpr = 0,
    ROW_STRIDE_D: tl.constexpr = 0,
    SPECIES_STRIDE_D: tl.constexpr = 1,
    ROW_STRIDE_S: tl.constexpr = 0,
    SPECIES_STRIDE_S: tl.constexpr = 1,
    USE_ACTIVE_MASK: tl.constexpr = False,
    DTYPE: tl.constexpr = tl.float32,
):
    n = tl.program_id(0)
    s_block = tl.program_id(1)
    s_offs = s_block * BLOCK_S + tl.arange(0, BLOCK_S)
    mask = s_offs < S

    parent_w = tl.load(eq1_parent_ids_ptr + n).to(tl.int64)
    out_base = parent_w * S
    family = tl.full((), 0, dtype=tl.int64)
    if mode_pD == 4 or mode_pS == 4:
        family = tl.load(family_idx_ptr + family_offset + parent_w).to(tl.int64)
    if USE_ACTIVE_MASK:
        parent_active = tl.load(active_mask_ptr + parent_w)
        if parent_active == 0:
            tl.store(
                out_ptr + out_base + s_offs,
                tl.full([BLOCK_S], value=-1e30, dtype=DTYPE),
                mask=mask,
            )
            return

    left_idx = tl.load(lefts_ptr + n).to(tl.int64)
    right_idx = tl.load(rights_ptr + n).to(tl.int64)
    base_l = left_idx * S
    base_r = right_idx * S

    pi_l = tl.load(Pi_ptr + base_l + s_offs, mask=mask, other=-1e30)
    pi_r = tl.load(Pi_ptr + base_r + s_offs, mask=mask, other=-1e30)
    pibar_l = tl.load(Pibar_ptr + base_l + s_offs, mask=mask, other=-1e30)
    pibar_r = tl.load(Pibar_ptr + base_r + s_offs, mask=mask, other=-1e30)

    log_pD_s = _load_dts_param(log_pD_ptr, n, s_offs, family, S, mask, mode_pD, ROW_STRIDE_D, SPECIES_STRIDE_D, BLOCK_S, DTYPE)
    log_pS_s = _load_dts_param(log_pS_ptr, n, s_offs, family, S, mask, mode_pS, ROW_STRIDE_S, SPECIES_STRIDE_S, BLOCK_S, DTYPE)

    c1 = tl.load(sp_child1_ptr + s_offs, mask=mask, other=S)
    c2 = tl.load(sp_child2_ptr + s_offs, mask=mask, other=S)
    c1_valid = c1 < S
    c2_valid = c2 < S

    t0 = log_pD_s + pi_l + pi_r
    t1 = pi_l + pibar_r
    t2 = pi_r + pibar_l
    t3 = (
        log_pS_s
        + tl.load(Pi_ptr + base_l + c1, mask=mask & c1_valid, other=-1e30)
        + tl.load(Pi_ptr + base_r + c2, mask=mask & c2_valid, other=-1e30)
    )
    t4 = (
        log_pS_s
        + tl.load(Pi_ptr + base_r + c1, mask=mask & c1_valid, other=-1e30)
        + tl.load(Pi_ptr + base_l + c2, mask=mask & c2_valid, other=-1e30)
    )

    m = tl.maximum(t0, t1)
    m = tl.maximum(m, t2)
    m = tl.maximum(m, t3)
    m = tl.maximum(m, t4)
    m_safe = tl.where(m > -1e29, m, tl.zeros_like(m))
    s = (
        tl.exp2(t0 - m_safe)
        + tl.exp2(t1 - m_safe)
        + tl.exp2(t2 - m_safe)
        + tl.exp2(t3 - m_safe)
        + tl.exp2(t4 - m_safe)
    )
    lsp = tl.load(log_split_probs_ptr + n)
    result = tl.log2(s) + m + lsp
    tl.store(out_ptr + out_base + s_offs, result, mask=mask)


@triton.jit
def _dts_parent_reduced_ge2_stage1_kernel(
    Pi_ptr, Pibar_ptr,
    lefts_ptr, rights_ptr,
    sp_child1_ptr, sp_child2_ptr,
    log_pD_ptr,
    log_pS_ptr,
    log_split_probs_ptr,
    ge2_ptr,
    ge2_parent_ids_ptr,
    active_mask_ptr,
    partial_max_ptr,
    partial_sum_ptr,
    family_idx_ptr,
    family_offset,
    split_offset: tl.constexpr,
    n_groups: tl.constexpr,
    S: tl.constexpr,
    MAX_TILES: tl.constexpr,
    TILE_SPLITS: tl.constexpr,
    BLOCK_S: tl.constexpr,
    mode_pD: tl.constexpr = 0,
    mode_pS: tl.constexpr = 0,
    ROW_STRIDE_D: tl.constexpr = 0,
    SPECIES_STRIDE_D: tl.constexpr = 1,
    ROW_STRIDE_S: tl.constexpr = 0,
    SPECIES_STRIDE_S: tl.constexpr = 1,
    USE_ACTIVE_MASK: tl.constexpr = False,
    DTYPE: tl.constexpr = tl.float32,
):
    group = tl.program_id(0)
    tile_id = tl.program_id(1)
    s_block = tl.program_id(2)
    s_offs = s_block * BLOCK_S + tl.arange(0, BLOCK_S)
    mask = s_offs < S

    parent_w = tl.load(ge2_parent_ids_ptr + group).to(tl.int64)
    family = tl.full((), 0, dtype=tl.int64)
    if mode_pD == 4 or mode_pS == 4:
        family = tl.load(family_idx_ptr + family_offset + parent_w).to(tl.int64)
    if USE_ACTIVE_MASK:
        parent_active = tl.load(active_mask_ptr + parent_w)
        if parent_active == 0:
            return

    start = tl.load(ge2_ptr + group)
    end = tl.load(ge2_ptr + group + 1)
    tile_start = start + tile_id * TILE_SPLITS
    if tile_start >= end:
        return
    tile_end = tl.minimum(tile_start + TILE_SPLITS, end)

    m = tl.full([BLOCK_S], value=-1e30, dtype=DTYPE)
    acc = tl.zeros([BLOCK_S], dtype=DTYPE)
    split_rel = tile_start
    while split_rel < tile_end:
        split_i = split_offset + split_rel
        left_idx = tl.load(lefts_ptr + split_i).to(tl.int64)
        right_idx = tl.load(rights_ptr + split_i).to(tl.int64)
        base_l = left_idx * S
        base_r = right_idx * S

        pi_l = tl.load(Pi_ptr + base_l + s_offs, mask=mask, other=-1e30)
        pi_r = tl.load(Pi_ptr + base_r + s_offs, mask=mask, other=-1e30)
        pibar_l = tl.load(Pibar_ptr + base_l + s_offs, mask=mask, other=-1e30)
        pibar_r = tl.load(Pibar_ptr + base_r + s_offs, mask=mask, other=-1e30)

        log_pD_s = _load_dts_param(log_pD_ptr, split_i, s_offs, family, S, mask, mode_pD, ROW_STRIDE_D, SPECIES_STRIDE_D, BLOCK_S, DTYPE)
        log_pS_s = _load_dts_param(log_pS_ptr, split_i, s_offs, family, S, mask, mode_pS, ROW_STRIDE_S, SPECIES_STRIDE_S, BLOCK_S, DTYPE)

        c1 = tl.load(sp_child1_ptr + s_offs, mask=mask, other=S)
        c2 = tl.load(sp_child2_ptr + s_offs, mask=mask, other=S)
        c1_valid = c1 < S
        c2_valid = c2 < S

        pi_l_c1 = tl.load(Pi_ptr + base_l + c1, mask=mask & c1_valid, other=-1e30)
        pi_r_c2 = tl.load(Pi_ptr + base_r + c2, mask=mask & c2_valid, other=-1e30)
        pi_r_c1 = tl.load(Pi_ptr + base_r + c1, mask=mask & c1_valid, other=-1e30)
        pi_l_c2 = tl.load(Pi_ptr + base_l + c2, mask=mask & c2_valid, other=-1e30)

        lsp = tl.load(log_split_probs_ptr + split_i)
        v0 = lsp + log_pD_s + pi_l + pi_r
        v1 = lsp + pi_l + pibar_r
        v2 = lsp + pi_r + pibar_l
        v3 = lsp + log_pS_s + pi_l_c1 + pi_r_c2
        v4 = lsp + log_pS_s + pi_r_c1 + pi_l_c2

        split_m = tl.maximum(v0, v1)
        split_m = tl.maximum(split_m, v2)
        split_m = tl.maximum(split_m, v3)
        split_m = tl.maximum(split_m, v4)
        split_m_safe = tl.where(split_m > -1e29, split_m, tl.zeros_like(split_m))
        split_sum = (
            tl.exp2(v0 - split_m_safe)
            + tl.exp2(v1 - split_m_safe)
            + tl.exp2(v2 - split_m_safe)
            + tl.exp2(v3 - split_m_safe)
            + tl.exp2(v4 - split_m_safe)
        )

        new_m = tl.maximum(m, split_m)
        new_m_safe = tl.where(new_m > -1e29, new_m, tl.zeros_like(new_m))
        old_term = tl.where(m > -1e29, acc * tl.exp2(m - new_m_safe), tl.zeros_like(acc))
        split_term = split_sum * tl.exp2(split_m_safe - new_m_safe)
        acc = old_term + split_term
        m = new_m
        split_rel += 1

    partial_row = group * MAX_TILES + tile_id
    tl.store(partial_max_ptr + partial_row * S + s_offs, m, mask=mask)
    tl.store(partial_sum_ptr + partial_row * S + s_offs, acc, mask=mask)


@triton.jit
def _dts_parent_reduced_ge2_stage2_kernel(
    ge2_ptr,
    ge2_parent_ids_ptr,
    active_mask_ptr,
    partial_max_ptr,
    partial_sum_ptr,
    out_ptr,
    n_groups: tl.constexpr,
    S: tl.constexpr,
    MAX_TILES: tl.constexpr,
    TILE_SPLITS: tl.constexpr,
    BLOCK_S: tl.constexpr,
    USE_ACTIVE_MASK: tl.constexpr = False,
    DTYPE: tl.constexpr = tl.float32,
):
    group = tl.program_id(0)
    s_block = tl.program_id(1)
    s_offs = s_block * BLOCK_S + tl.arange(0, BLOCK_S)
    mask = s_offs < S

    parent_w = tl.load(ge2_parent_ids_ptr + group).to(tl.int64)
    out_base = parent_w * S
    if USE_ACTIVE_MASK:
        parent_active = tl.load(active_mask_ptr + parent_w)
        if parent_active == 0:
            tl.store(
                out_ptr + out_base + s_offs,
                tl.full([BLOCK_S], value=-1e30, dtype=DTYPE),
                mask=mask,
            )
            return

    start = tl.load(ge2_ptr + group)
    end = tl.load(ge2_ptr + group + 1)
    n_tiles = tl.cdiv(end - start, TILE_SPLITS)
    m = tl.full([BLOCK_S], value=-1e30, dtype=DTYPE)
    acc = tl.zeros([BLOCK_S], dtype=DTYPE)
    tile_id = 0
    while tile_id < n_tiles:
        partial_row = group * MAX_TILES + tile_id
        pm = tl.load(partial_max_ptr + partial_row * S + s_offs, mask=mask, other=-1e30)
        ps = tl.load(partial_sum_ptr + partial_row * S + s_offs, mask=mask, other=0.0)
        new_m = tl.maximum(m, pm)
        new_m_safe = tl.where(new_m > -1e29, new_m, tl.zeros_like(new_m))
        old_term = tl.where(m > -1e29, acc * tl.exp2(m - new_m_safe), tl.zeros_like(acc))
        tile_term = tl.where(
            pm > -1e29,
            ps * tl.exp2(pm - new_m_safe),
            tl.zeros_like(acc),
        )
        acc = old_term + tile_term
        m = new_m
        tile_id += 1

    result = tl.log2(acc) + m
    tl.store(out_ptr + out_base + s_offs, result, mask=mask)


def dts_fused_parent_reduced(
    Pi,
    Pibar,
    lefts,
    rights,
    sp_child1,
    sp_child2,
    log_pD,
    log_pS,
    log_split_probs,
    W,
    n_eq1,
    eq1_reduce_idx,
    ge2_ptr,
    ge2_parent_ids,
    out=None,
    active_mask=None,
    family_idx=None,
    family_offset=0,
    tile_splits=64,
    ge2_max_fanout=None,
):
    """Parent-reduced DTS forward recompute.

    Eq1 splits write directly into their parent rows. Ge2 parents are reduced
    through the retained two-stage tiled partial max/sum path, avoiding the
    full ``[n_splits, S]`` DTS materialization.
    """
    N = lefts.shape[0]
    S = Pi.shape[1]
    if out is None:
        out = torch.full((W, S), float("-inf"), device=Pi.device, dtype=Pi.dtype)
    else:
        out.fill_(float("-inf"))

    lsp = log_split_probs.reshape(N).contiguous()
    family_indexed = family_idx is not None

    if n_eq1 > 0:
        log_pD_vec, mode_pD, row_stride_D, species_stride_D = _prepare_param(
            log_pD, N, S, family_indexed=family_indexed
        )
        log_pS_vec, mode_pS, row_stride_S, species_stride_S = _prepare_param(
            log_pS, N, S, family_indexed=family_indexed
        )
        block_s_env = os.environ.get("GPUREC_DTS_PARENT_BLOCK_S")
        if block_s_env is None:
            BLOCK_S = min(256, triton.next_power_of_2(S))
        else:
            BLOCK_S = min(
                max(1, triton.next_power_of_2(int(block_s_env))),
                triton.next_power_of_2(S),
            )
        parent_num_warps = int(os.environ.get("GPUREC_DTS_PARENT_NUM_WARPS", "0"))
        launch_options = {}
        if parent_num_warps > 0:
            launch_options["num_warps"] = parent_num_warps
        grid_eq1 = (n_eq1, triton.cdiv(S, BLOCK_S))
        _dts_eq1_to_rows_kernel[grid_eq1](
            Pi.contiguous(),
            Pibar.contiguous(),
            lefts,
            rights,
            sp_child1,
            sp_child2,
            log_pD_vec,
            log_pS_vec,
            lsp,
            eq1_reduce_idx,
            active_mask if active_mask is not None else eq1_reduce_idx,
            out,
            family_idx if family_idx is not None else eq1_reduce_idx,
            int(family_offset),
            n_eq1,
            S,
            BLOCK_S=BLOCK_S,
            mode_pD=mode_pD,
            mode_pS=mode_pS,
            ROW_STRIDE_D=row_stride_D,
            SPECIES_STRIDE_D=species_stride_D,
            ROW_STRIDE_S=row_stride_S,
            SPECIES_STRIDE_S=species_stride_S,
            USE_ACTIVE_MASK=bool(active_mask is not None),
            DTYPE=_tl_float_dtype(Pi.dtype),
            **launch_options,
        )

    n_groups = ge2_parent_ids.numel()
    if n_groups == 0:
        return out

    log_pD_vec, mode_pD, row_stride_D, species_stride_D = _prepare_param(
        log_pD, N, S, family_indexed=family_indexed
    )
    log_pS_vec, mode_pS, row_stride_S, species_stride_S = _prepare_param(
        log_pS, N, S, family_indexed=family_indexed
    )
    block_s_env = os.environ.get("GPUREC_DTS_PARENT_BLOCK_S")
    if block_s_env is None:
        BLOCK_S = min(256, triton.next_power_of_2(S))
    else:
        BLOCK_S = min(
            max(1, triton.next_power_of_2(int(block_s_env))),
            triton.next_power_of_2(S),
        )
    tile_splits_env = os.environ.get("GPUREC_DTS_PARENT_TILE_SPLITS")
    if tile_splits_env is not None:
        tile_splits = int(tile_splits_env)
    parent_num_warps = int(os.environ.get("GPUREC_DTS_PARENT_NUM_WARPS", "0"))
    launch_options = {}
    if parent_num_warps > 0:
        launch_options["num_warps"] = parent_num_warps
    if ge2_max_fanout is None:
        ge2_max_fanout = int((ge2_ptr[1:] - ge2_ptr[:-1]).max().item())
    tile_splits = max(1, int(tile_splits))
    max_tiles = max(1, triton.cdiv(int(ge2_max_fanout), tile_splits))
    partial_shape = (n_groups * max_tiles, S)
    partial_max = torch.empty(partial_shape, device=Pi.device, dtype=Pi.dtype)
    partial_sum = torch.empty(partial_shape, device=Pi.device, dtype=Pi.dtype)
    grid_stage1 = (n_groups, max_tiles, triton.cdiv(S, BLOCK_S))
    _dts_parent_reduced_ge2_stage1_kernel[grid_stage1](
        Pi.contiguous(),
        Pibar.contiguous(),
        lefts,
        rights,
        sp_child1,
        sp_child2,
        log_pD_vec,
        log_pS_vec,
        lsp,
        ge2_ptr,
        ge2_parent_ids,
        active_mask if active_mask is not None else ge2_parent_ids,
        partial_max,
        partial_sum,
        family_idx if family_idx is not None else ge2_parent_ids,
        int(family_offset),
        split_offset=n_eq1,
        n_groups=n_groups,
        S=S,
        MAX_TILES=max_tiles,
        TILE_SPLITS=tile_splits,
        BLOCK_S=BLOCK_S,
        mode_pD=mode_pD,
        mode_pS=mode_pS,
        ROW_STRIDE_D=row_stride_D,
        SPECIES_STRIDE_D=species_stride_D,
        ROW_STRIDE_S=row_stride_S,
        SPECIES_STRIDE_S=species_stride_S,
        USE_ACTIVE_MASK=bool(active_mask is not None),
        DTYPE=_tl_float_dtype(Pi.dtype),
        **launch_options,
    )
    grid_stage2 = (n_groups, triton.cdiv(S, BLOCK_S))
    _dts_parent_reduced_ge2_stage2_kernel[grid_stage2](
        ge2_ptr,
        ge2_parent_ids,
        active_mask if active_mask is not None else ge2_parent_ids,
        partial_max,
        partial_sum,
        out,
        n_groups=n_groups,
        S=S,
        MAX_TILES=max_tiles,
        TILE_SPLITS=tile_splits,
        BLOCK_S=BLOCK_S,
        USE_ACTIVE_MASK=bool(active_mask is not None),
        DTYPE=_tl_float_dtype(Pi.dtype),
        **launch_options,
    )
    return out
