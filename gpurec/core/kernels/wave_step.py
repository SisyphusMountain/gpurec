"""Fused Triton kernels for wave-step computation."""

import torch
import triton
import triton.language as tl


def _uniform_block_s(S: int, *, default_cap: int = 256) -> int:
    """Return the species tile width for uniform forward kernels."""
    return int(min(default_cap, triton.next_power_of_2(S)))


def _uniform_num_warps() -> int:
    return 4


def _leaf_logp_mode(use_leaf_index: bool, leaf_logp, family_idx, S: int) -> int:
    """Return the Triton leaf-logp addressing mode for uniform leaf-index hits."""
    if not use_leaf_index or family_idx is None:
        return 0
    if leaf_logp.ndim == 1:
        if int(leaf_logp.numel()) == int(S):
            return 0
        return 1
    if leaf_logp.ndim == 2:
        return 2
    raise ValueError("batched leaf_logp must have shape [G] or [G, S]")


def _uniform_const_layout(const_tensor, family_idx, family_indexed: bool) -> int:
    """Return addressing mode for uniform per-species constants.

    Modes:
      0: shared [S]
      2: family-indexed [G, S] addressed through family_idx[C]
    """
    if family_indexed:
        if family_idx is None:
            raise ValueError("family-indexed constants require family_idx")
        if const_tensor.ndim != 2:
            raise ValueError("family-indexed constants require [G, S] tensors")
        return 2
    if const_tensor.ndim == 2:
        raise ValueError("row-expanded forward constants are not part of the lean path")
    return 0


# --- Fused uniform-Pibar kernel ---
# One program per clade row. Two passes:
#   Pass 1: online max+sum over Pi row (for uniform Pibar stats)
#   Pass 2: compute Pibar inline, DTS_L terms, logsumexp, convergence diff

@triton.jit
def _wave_step_uniform_kernel(
    # Global Pi tensor [C, S] — read from rows [ws : ws+W]
    Pi_ptr,
    ws,                  # wave start (clade offset)
    # Constants: [S] each
    mt_ptr,
    DL_const_ptr, Ebar_ptr, E_ptr, SL1_const_ptr, SL2_const_ptr,
    # Species child indices: [S] long each
    sp_child1_ptr, sp_child2_ptr,
    # Species parent index: [S] long, -1 for root
    sp_parent_ptr,
    # Per-wave arrays: [W, S]
    leaf_term_ptr,
    leaf_species_ptr,
    leaf_logp_ptr,
    family_idx_ptr,
    DTS_reduced_ptr,
    has_splits: tl.constexpr,
    # Outputs
    Pi_new_ptr,          # [W, S]
    Pibar_out_ptr,       # [C, S] — write Pibar to rows [ws : ws+W]
    max_diff_ptr,        # [W] — per-row max |Pi_new - Pi_old| for convergence
    pibar_row_max_ptr,   # optional [C] — final row max for backward Pibar VJP
    # Dimensions
    S: tl.constexpr,
    stride: tl.constexpr,
    CONST_ROW_STRIDE: tl.constexpr,
    CONST_SPECIES_STRIDE: tl.constexpr,
    BLOCK_S: tl.constexpr,
    MAX_ANCESTOR_DEPTH: tl.constexpr,
    COMPUTE_DIFF: tl.constexpr,
    USE_LEAF_INDEX: tl.constexpr,
    LEAF_LOGP_MODE: tl.constexpr,
    STORE_PIBAR: tl.constexpr,
    STORE_PIBAR_ROW_MAX: tl.constexpr,
    OUTPUT_GLOBAL: tl.constexpr,
    FP64: tl.constexpr,
    TOPOLOGY_INT32: tl.constexpr,
    CONST_LAYOUT: tl.constexpr = 0,
):
    """Fused kernel: uniform Pibar + DTS_L + logsumexp + convergence diff.

    Each program handles one full clade row, processing S elements in tiles.
    Pass 1 uses the online max+sum trick (single scan) for row statistics.
    Pass 2 computes Pibar inline and all DTS_L terms in one scan.

    CONST_LAYOUT:
      0 = shared [S] constants
      2 = per-family [G, S] constants addressed by family_idx[ws + w]
    """
    DTYPE = tl.float64 if FP64 else tl.float32
    NEG_LARGE = -1e300 if FP64 else -1e30

    w = tl.program_id(0)
    pi_base = (ws + w) * stride      # offset into global Pi/Pibar
    if OUTPUT_GLOBAL:
        out_base = pi_base            # offset into global output rows
    else:
        out_base = w * stride         # offset into [W, S] outputs
    const_base = 0
    if CONST_LAYOUT == 2:
        family_const = tl.load(family_idx_ptr + ws + w)
        const_base = family_const * CONST_ROW_STRIDE

    # === Pass 1: Online max + sum over the Pi row ===
    row_max = tl.full([1], value=NEG_LARGE, dtype=DTYPE)
    row_sum = tl.full([1], value=0.0, dtype=DTYPE)

    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        mask = s_offs < S
        pi_val = tl.load(Pi_ptr + pi_base + s_offs, mask=mask, other=NEG_LARGE)
        tile_max = tl.max(pi_val, axis=0)
        new_max = tl.maximum(row_max, tile_max)
        # Rescale running sum to new max, add this tile's contribution
        row_sum = row_sum * tl.exp2(row_max - new_max) + tl.sum(tl.exp2(pi_val - new_max), axis=0)
        row_max = new_max

    if STORE_PIBAR_ROW_MAX:
        tl.store(pibar_row_max_ptr + ws + w, tl.max(row_max, axis=0))

    # === Pass 2: Pibar + DTS_L terms + logsumexp ===
    if COMPUTE_DIFF:
        local_max_diff = tl.full([1], value=0.0, dtype=DTYPE)
    M_SAFE_THRESH = -1e299 if FP64 else -1e29

    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        mask = s_offs < S

        # Load Pi[w, s]
        pi_w = tl.load(Pi_ptr + pi_base + s_offs, mask=mask, other=NEG_LARGE)

        ancestor_sum = tl.zeros([BLOCK_S], dtype=DTYPE)
        # Uniform Pibar: log2(row_sum - ancestor_sum) + max + mt. Walk the
        # species parent chain starting at s and sum exp2(Pi[ancestor] - max).
        if TOPOLOGY_INT32:
            cur = s_offs
        else:
            cur = s_offs.to(tl.int64)
        for _ in range(0, MAX_ANCESTOR_DEPTH):
            cur_valid = mask & (cur >= 0) & (cur < S)
            pi_anc = tl.load(Pi_ptr + pi_base + cur, mask=cur_valid, other=NEG_LARGE)
            ancestor_sum += tl.where(cur_valid, tl.exp2(pi_anc - row_max), tl.zeros([BLOCK_S], dtype=DTYPE))
            cur = tl.load(sp_parent_ptr + cur, mask=cur_valid, other=-1)

        const_offsets = const_base + s_offs * CONST_SPECIES_STRIDE
        mt = tl.load(mt_ptr + const_offsets, mask=mask, other=0.0)
        denom = row_sum - ancestor_sum
        pibar_w = tl.where(denom > 0.0, tl.log2(denom) + row_max + mt, NEG_LARGE)

        # Store Pibar to global tensor when this invocation is producing the
        # final Pibar rows. Fixed-iteration ping-pong uses Pibar as Pi scratch
        # and recomputes/stores final Pibar after the last iteration.
        if STORE_PIBAR:
            tl.store(Pibar_out_ptr + pi_base + s_offs, pibar_w, mask=mask)

        dl_const = tl.load(DL_const_ptr + const_offsets, mask=mask, other=NEG_LARGE)
        ebar = tl.load(Ebar_ptr + const_offsets, mask=mask, other=NEG_LARGE)
        e_val = tl.load(E_ptr + const_offsets, mask=mask, other=NEG_LARGE)
        sl1_const = tl.load(SL1_const_ptr + const_offsets, mask=mask, other=NEG_LARGE)
        sl2_const = tl.load(SL2_const_ptr + const_offsets, mask=mask, other=NEG_LARGE)

        # Gather species children
        c1 = tl.load(sp_child1_ptr + s_offs, mask=mask, other=0)
        c2 = tl.load(sp_child2_ptr + s_offs, mask=mask, other=0)
        c1_valid = c1 < S
        c2_valid = c2 < S
        pi_s1 = tl.load(Pi_ptr + pi_base + c1, mask=mask & c1_valid, other=NEG_LARGE)
        pi_s2 = tl.load(Pi_ptr + pi_base + c2, mask=mask & c2_valid, other=NEG_LARGE)

        # 6 DTS_L terms
        t0 = dl_const + pi_w
        t1 = pi_w + ebar
        t2 = pibar_w + e_val
        t3 = sl1_const + pi_s1
        t4 = sl2_const + pi_s2
        if USE_LEAF_INDEX:
            leaf_species = tl.load(leaf_species_ptr + ws + w)
            leaf_hit = mask & (leaf_species == s_offs)
            if LEAF_LOGP_MODE == 2:
                family = tl.load(family_idx_ptr + ws + w)
                leaf_logp = tl.load(leaf_logp_ptr + family * S + s_offs, mask=mask, other=NEG_LARGE)
                t5 = tl.where(leaf_hit, leaf_logp, NEG_LARGE)
            elif LEAF_LOGP_MODE == 1:
                family = tl.load(family_idx_ptr + ws + w)
                leaf_logp = tl.load(leaf_logp_ptr + family)
                t5 = tl.where(leaf_hit, leaf_logp, NEG_LARGE)
            else:
                leaf_logp = tl.load(leaf_logp_ptr + s_offs, mask=mask, other=NEG_LARGE)
                t5 = tl.where(leaf_hit, leaf_logp, NEG_LARGE)
        else:
            t5 = tl.load(leaf_term_ptr + out_base + s_offs, mask=mask, other=NEG_LARGE)

        m = tl.maximum(t0, t1)
        m = tl.maximum(m, t2)
        m = tl.maximum(m, t3)
        m = tl.maximum(m, t4)
        m = tl.maximum(m, t5)

        if has_splits:
            dts_r = tl.load(DTS_reduced_ptr + out_base + s_offs, mask=mask, other=NEG_LARGE)
            m = tl.maximum(m, dts_r)

        m_safe = tl.where(m > M_SAFE_THRESH, m, tl.zeros_like(m))
        s = tl.exp2(t0 - m_safe) + tl.exp2(t1 - m_safe) + tl.exp2(t2 - m_safe)
        s += tl.exp2(t3 - m_safe) + tl.exp2(t4 - m_safe) + tl.exp2(t5 - m_safe)
        if has_splits:
            s += tl.exp2(dts_r - m_safe)

        result = tl.log2(s) + m
        tl.store(Pi_new_ptr + out_base + s_offs, result, mask=mask)

        if COMPUTE_DIFF:
            significant = result > -100.0
            diff = tl.where(significant & mask, tl.abs(result - pi_w), tl.zeros_like(result))
            local_max_diff = tl.maximum(local_max_diff, tl.max(diff, axis=0))

    if COMPUTE_DIFF:
        tl.store(max_diff_ptr + w, tl.max(local_max_diff, axis=0))


def wave_step_uniform_fused_into(Pi_in, Pi_out, Pibar, ws, W, S,
                                 mt_squeezed, DL_const, Ebar, E, SL1_const, SL2_const,
                                 sp_child1, sp_child2, sp_parent, max_ancestor_depth,
                                 leaf_term_wt, DTS_reduced=None,
                                 leaf_species_idx=None, leaf_logp=None,
                                 family_idx=None,
                                 family_indexed_consts=False):
    """Fused uniform wave step writing Pi output directly into global rows."""
    fp64 = Pi_in.dtype == torch.float64
    has_splits = DTS_reduced is not None
    const_layout = _uniform_const_layout(DL_const, family_idx, family_indexed_consts)
    const_row_stride = 0
    const_species_stride = 1
    if const_layout == 2:
        const_row_stride = 0 if int(DL_const.shape[0]) == 1 else int(DL_const.stride(0))
        const_species_stride = int(DL_const.stride(1)) if DL_const.ndim == 2 else 1
    use_leaf_index = leaf_species_idx is not None and leaf_logp is not None
    leaf_species_arg = leaf_species_idx if use_leaf_index else sp_parent
    leaf_logp_arg = leaf_logp if use_leaf_index else leaf_term_wt
    leaf_logp_mode = _leaf_logp_mode(use_leaf_index, leaf_logp, family_idx, S)
    family_idx_arg = family_idx if family_idx is not None else sp_parent
    max_diff_buf = Pi_out

    BLOCK_S = _uniform_block_s(S)
    num_warps = _uniform_num_warps()
    grid = (W,)
    Pi_out_rows = Pi_out.narrow(0, int(ws), int(W))

    _wave_step_uniform_kernel[grid](
        Pi_in, ws,
        mt_squeezed,
        DL_const, Ebar, E, SL1_const, SL2_const,
        sp_child1, sp_child2,
        sp_parent,
        leaf_term_wt,
        leaf_species_arg,
        leaf_logp_arg,
        family_idx_arg,
        DTS_reduced if has_splits else leaf_term_wt,
        has_splits,
        Pi_out_rows, Pibar, max_diff_buf, Pibar,
        S,
        stride=S,
        CONST_ROW_STRIDE=const_row_stride,
        CONST_SPECIES_STRIDE=const_species_stride,
        BLOCK_S=BLOCK_S,
        MAX_ANCESTOR_DEPTH=int(max_ancestor_depth),
        COMPUTE_DIFF=False,
        USE_LEAF_INDEX=use_leaf_index,
        LEAF_LOGP_MODE=leaf_logp_mode,
        STORE_PIBAR=False,
        STORE_PIBAR_ROW_MAX=False,
        OUTPUT_GLOBAL=False,
        FP64=fp64,
        TOPOLOGY_INT32=sp_parent.dtype == torch.int32,
        CONST_LAYOUT=const_layout,
        num_warps=num_warps,
    )


@triton.jit
def _wave_pibar_uniform_parent_kernel(
    Pi_ptr,
    ws,
    mt_ptr,
    sp_parent_ptr,
    family_idx_ptr,
    Pibar_out_ptr,
    row_max_out_ptr,
    S: tl.constexpr,
    stride: tl.constexpr,
    CONST_ROW_STRIDE: tl.constexpr,
    CONST_SPECIES_STRIDE: tl.constexpr,
    BLOCK_S: tl.constexpr,
    MAX_ANCESTOR_DEPTH: tl.constexpr,
    STORE_ROW_MAX: tl.constexpr,
    FP64: tl.constexpr,
    TOPOLOGY_INT32: tl.constexpr,
    CONST_LAYOUT: tl.constexpr = 0,
):
    DTYPE = tl.float64 if FP64 else tl.float32
    NEG_LARGE = -1e300 if FP64 else -1e30

    w = tl.program_id(0)
    pi_base = (ws + w) * stride
    const_base = 0
    if CONST_LAYOUT == 2:
        family_const = tl.load(family_idx_ptr + ws + w)
        const_base = family_const * CONST_ROW_STRIDE

    row_max = tl.full([1], value=NEG_LARGE, dtype=DTYPE)
    row_sum = tl.full([1], value=0.0, dtype=DTYPE)
    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        mask = s_offs < S
        pi_val = tl.load(Pi_ptr + pi_base + s_offs, mask=mask, other=NEG_LARGE)
        tile_max = tl.max(pi_val, axis=0)
        new_max = tl.maximum(row_max, tile_max)
        row_sum = row_sum * tl.exp2(row_max - new_max) + tl.sum(tl.exp2(pi_val - new_max), axis=0)
        row_max = new_max

    if STORE_ROW_MAX:
        tl.store(row_max_out_ptr + ws + w, tl.max(row_max, axis=0))

    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        mask = s_offs < S

        if TOPOLOGY_INT32:
            cur = s_offs
        else:
            cur = s_offs.to(tl.int64)
        ancestor_sum = tl.zeros([BLOCK_S], dtype=DTYPE)
        for _ in range(0, MAX_ANCESTOR_DEPTH):
            cur_valid = mask & (cur >= 0) & (cur < S)
            pi_anc = tl.load(Pi_ptr + pi_base + cur, mask=cur_valid, other=NEG_LARGE)
            ancestor_sum += tl.where(cur_valid, tl.exp2(pi_anc - row_max), tl.zeros([BLOCK_S], dtype=DTYPE))
            cur = tl.load(sp_parent_ptr + cur, mask=cur_valid, other=-1)

        mt = tl.load(mt_ptr + const_base + s_offs * CONST_SPECIES_STRIDE, mask=mask, other=0.0)
        denom = row_sum - ancestor_sum
        pibar_w = tl.where(denom > 0.0, tl.log2(denom) + row_max + mt, NEG_LARGE)
        tl.store(Pibar_out_ptr + pi_base + s_offs, pibar_w, mask=mask)


def wave_pibar_uniform_parent_fused(Pi, Pibar, ws, W, S,
                                    mt_squeezed, sp_parent, max_ancestor_depth,
                                    row_max_out=None, family_idx=None,
                                    family_indexed_consts=False):
    """Compute uniform Pibar rows by walking species parent pointers."""
    fp64 = Pi.dtype == torch.float64
    const_layout = _uniform_const_layout(mt_squeezed, family_idx, family_indexed_consts)
    const_row_stride = 0
    const_species_stride = 1
    if const_layout == 2:
        const_row_stride = 0 if int(mt_squeezed.shape[0]) == 1 else int(mt_squeezed.stride(0))
        const_species_stride = int(mt_squeezed.stride(1)) if mt_squeezed.ndim == 2 else 1
    family_idx_arg = family_idx if family_idx is not None else sp_parent
    BLOCK_S = _uniform_block_s(S)
    num_warps = _uniform_num_warps()
    grid = (W,)

    _wave_pibar_uniform_parent_kernel[grid](
        Pi,
        ws,
        mt_squeezed,
        sp_parent,
        family_idx_arg,
        Pibar,
        row_max_out if row_max_out is not None else Pibar,
        S,
        stride=S,
        CONST_ROW_STRIDE=const_row_stride,
        CONST_SPECIES_STRIDE=const_species_stride,
        BLOCK_S=BLOCK_S,
        MAX_ANCESTOR_DEPTH=int(max_ancestor_depth),
        STORE_ROW_MAX=bool(row_max_out is not None),
        FP64=fp64,
        TOPOLOGY_INT32=sp_parent.dtype == torch.int32,
        CONST_LAYOUT=const_layout,
        num_warps=num_warps,
    )
