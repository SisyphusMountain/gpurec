"""Fused Triton kernels for wave-step computation."""

import torch
import triton
import triton.language as tl


@triton.jit
def _wave_pibar_step_kernel(
    # Pi[wt]: [W, S] log2-space (input, the current wave's Pi)
    Pi_W_ptr,
    # Transfer matrix: [S, S] (M[e, f]) — NOT transposed
    transfer_mat_ptr,
    # max_transfer_mat: [S] (mt[e])
    mt_ptr,
    # Precomputed constants: [S] each
    DL_const_ptr, Ebar_ptr, E_ptr, SL1_const_ptr, SL2_const_ptr,
    # Species child indices: [S] each (index into columns, S = padding/leaf)
    sp_child1_ptr, sp_child2_ptr,
    # Leaf term: [W, S]
    leaf_term_ptr,
    # DTS_reduced: [W, S] or None (if no splits)
    DTS_reduced_ptr,
    has_splits: tl.constexpr,
    # Outputs: Pi_new [W, S], Pibar [W, S]
    Pi_new_ptr, Pibar_ptr,
    # Dimensions
    S: tl.constexpr,
    # Strides
    stride_w: tl.constexpr,       # stride for [W, S] tensors (== S)
    stride_m_row: tl.constexpr,   # stride for transfer_mat rows (== S)
    # Block sizes
    BLOCK_S: tl.constexpr,
    BLOCK_F: tl.constexpr,  # tile size for reduction over f in matmul
):
    """Fully fused kernel: Pibar computation + DTS_L terms + logsumexp.

    Each program instance handles one (wave_clade, species_block) pair.
    For each output species e in the block:
      1. Compute Pibar[w,e] = log2(sum_f exp2(Pi[w,f] - max_f) * M[e,f]) + max_f + mt[e]
      2. Compute 6 DTS_L terms using Pi[w,e], Pibar[w,e], children
      3. logsumexp all terms (+ DTS_reduced if present)
    """
    w = tl.program_id(0)
    s_block = tl.program_id(1)

    s_offs = s_block * BLOCK_S + tl.arange(0, BLOCK_S)
    mask = s_offs < S

    base_w = w * stride_w

    # --- Step 1: Compute Pibar[w, s_offs] via log-space matmul ---
    # First pass: find max over f for stabilization
    pi_max = tl.full([1], value=-1e30, dtype=tl.float32)
    for f_start in range(0, S, BLOCK_F):
        f_offs = f_start + tl.arange(0, BLOCK_F)
        f_mask = f_offs < S
        pi_f = tl.load(Pi_W_ptr + base_w + f_offs, mask=f_mask, other=-1e30)
        pi_max = tl.maximum(pi_max, tl.max(pi_f, axis=0))

    # Second pass: accumulate exp2(pi_f - max) * M[e, f] for each e in s_offs
    acc = tl.zeros([BLOCK_S], dtype=tl.float32)
    for f_start in range(0, S, BLOCK_F):
        f_offs = f_start + tl.arange(0, BLOCK_F)
        f_mask = f_offs < S
        pi_f = tl.load(Pi_W_ptr + base_w + f_offs, mask=f_mask, other=-1e30)
        exp_f = tl.exp2(pi_f - pi_max)  # [BLOCK_F]

        # Load M[s_offs, f_offs] — transfer_mat[e, f]
        # M has shape [S, S], stride_m_row = S
        # For each e in s_offs, load M[e, f_offs]
        m_ptrs = transfer_mat_ptr + s_offs[:, None] * stride_m_row + f_offs[None, :]
        m_mask = mask[:, None] & f_mask[None, :]
        m_vals = tl.load(m_ptrs, mask=m_mask, other=0.0)  # [BLOCK_S, BLOCK_F]

        acc += tl.sum(m_vals * exp_f[None, :], axis=1)  # [BLOCK_S]

    mt = tl.load(mt_ptr + s_offs, mask=mask, other=0.0)
    pibar_w = tl.log2(acc) + pi_max + mt  # [BLOCK_S]

    # Store Pibar
    tl.store(Pibar_ptr + base_w + s_offs, pibar_w, mask=mask)

    # --- Step 2: Load Pi[w, s_offs] and children ---
    pi_w = tl.load(Pi_W_ptr + base_w + s_offs, mask=mask, other=-1e30)

    # Load constants
    dl_const = tl.load(DL_const_ptr + s_offs, mask=mask, other=-1e30)
    ebar = tl.load(Ebar_ptr + s_offs, mask=mask, other=-1e30)
    e_val = tl.load(E_ptr + s_offs, mask=mask, other=-1e30)
    sl1_const = tl.load(SL1_const_ptr + s_offs, mask=mask, other=-1e30)
    sl2_const = tl.load(SL2_const_ptr + s_offs, mask=mask, other=-1e30)

    # Gather species children
    c1 = tl.load(sp_child1_ptr + s_offs, mask=mask, other=0)
    c2 = tl.load(sp_child2_ptr + s_offs, mask=mask, other=0)
    c1_valid = c1 < S
    c2_valid = c2 < S
    pi_s1 = tl.load(Pi_W_ptr + base_w + c1, mask=mask & c1_valid, other=-1e30)
    pi_s2 = tl.load(Pi_W_ptr + base_w + c2, mask=mask & c2_valid, other=-1e30)

    # --- Step 3: 6 DTS_L terms + logsumexp ---
    t0 = dl_const + pi_w
    t1 = pi_w + ebar
    t2 = pibar_w + e_val
    t3 = sl1_const + pi_s1
    t4 = sl2_const + pi_s2
    t5 = tl.load(leaf_term_ptr + base_w + s_offs, mask=mask, other=-1e30)

    m = tl.maximum(t0, t1)
    m = tl.maximum(m, t2)
    m = tl.maximum(m, t3)
    m = tl.maximum(m, t4)
    m = tl.maximum(m, t5)

    if has_splits:
        dts_r = tl.load(DTS_reduced_ptr + base_w + s_offs, mask=mask, other=-1e30)
        m = tl.maximum(m, dts_r)

    m_safe = tl.where(m > -1e29, m, tl.zeros_like(m))
    s = tl.exp2(t0 - m_safe) + tl.exp2(t1 - m_safe) + tl.exp2(t2 - m_safe)
    s += tl.exp2(t3 - m_safe) + tl.exp2(t4 - m_safe) + tl.exp2(t5 - m_safe)
    if has_splits:
        s += tl.exp2(dts_r - m_safe)

    result = tl.log2(s) + m
    tl.store(Pi_new_ptr + base_w + s_offs, result, mask=mask)


def wave_pibar_step_fused(Pi_W, transfer_mat, mt_squeezed,
                          DL_const, Ebar, E, SL1_const, SL2_const,
                          sp_child1, sp_child2, leaf_term_wt,
                          Pibar_out, DTS_reduced=None):
    """Fully fused wave step: Pibar + DTS_L + logsumexp in one kernel.

    Args:
        Pi_W: [W, S] contiguous, log2-space
        transfer_mat: [S, S] contiguous
        mt_squeezed: [S]
        DL_const, Ebar, E, SL1_const, SL2_const: [S] each
        sp_child1, sp_child2: [S] long
        leaf_term_wt: [W, S] contiguous
        Pibar_out: [W, S] output buffer for Pibar (written to)
        DTS_reduced: [W, S] or None

    Returns:
        Pi_new: [W, S] log2-space
    """
    W, S = Pi_W.shape
    Pi_new = torch.empty_like(Pi_W)
    has_splits = DTS_reduced is not None

    BLOCK_S = 32
    BLOCK_F = min(64, triton.next_power_of_2(S))
    grid = (W, (S + BLOCK_S - 1) // BLOCK_S)

    _wave_pibar_step_kernel[grid](
        Pi_W,
        transfer_mat,
        mt_squeezed,
        DL_const, Ebar, E, SL1_const, SL2_const,
        sp_child1, sp_child2,
        leaf_term_wt,
        DTS_reduced if has_splits else Pi_W,  # dummy
        has_splits,
        Pi_new, Pibar_out,
        S,
        stride_w=S,
        stride_m_row=S,
        BLOCK_S=BLOCK_S,
        BLOCK_F=BLOCK_F,
    )
    return Pi_new


# Keep the old kernel for compatibility
@triton.jit
def _wave_step_kernel(
    Pi_W_ptr, Pibar_W_ptr,
    DL_const_ptr, Ebar_ptr, E_ptr, SL1_const_ptr, SL2_const_ptr,
    sp_child1_ptr, sp_child2_ptr,
    leaf_term_ptr,
    DTS_reduced_ptr,
    has_splits: tl.constexpr,
    Pi_new_ptr,
    W: tl.constexpr, S: tl.constexpr,
    stride_ws: tl.constexpr,
):
    """Fused kernel: given Pi_W, Pibar_W, compute Pi_new = logsumexp2(all_terms, dim=0)."""
    w = tl.program_id(0)
    s_block = tl.program_id(1)
    BLOCK_S: tl.constexpr = 32

    s_offs = s_block * BLOCK_S + tl.arange(0, BLOCK_S)
    mask = s_offs < S

    base = w * stride_ws
    pi_w = tl.load(Pi_W_ptr + base + s_offs, mask=mask, other=-1e30)
    pibar_w = tl.load(Pibar_W_ptr + base + s_offs, mask=mask, other=-1e30)

    dl_const = tl.load(DL_const_ptr + s_offs, mask=mask, other=-1e30)
    ebar = tl.load(Ebar_ptr + s_offs, mask=mask, other=-1e30)
    e_val = tl.load(E_ptr + s_offs, mask=mask, other=-1e30)
    sl1_const = tl.load(SL1_const_ptr + s_offs, mask=mask, other=-1e30)
    sl2_const = tl.load(SL2_const_ptr + s_offs, mask=mask, other=-1e30)

    c1 = tl.load(sp_child1_ptr + s_offs, mask=mask, other=0)
    c2 = tl.load(sp_child2_ptr + s_offs, mask=mask, other=0)
    c1_valid = c1 < S
    c2_valid = c2 < S
    pi_s1 = tl.load(Pi_W_ptr + base + c1, mask=mask & c1_valid, other=-1e30)
    pi_s2 = tl.load(Pi_W_ptr + base + c2, mask=mask & c2_valid, other=-1e30)

    t0 = dl_const + pi_w
    t1 = pi_w + ebar
    t2 = pibar_w + e_val
    t3 = sl1_const + pi_s1
    t4 = sl2_const + pi_s2
    t5 = tl.load(leaf_term_ptr + base + s_offs, mask=mask, other=-1e30)

    m = tl.maximum(t0, t1)
    m = tl.maximum(m, t2)
    m = tl.maximum(m, t3)
    m = tl.maximum(m, t4)
    m = tl.maximum(m, t5)

    if has_splits:
        dts_r = tl.load(DTS_reduced_ptr + w * stride_ws + s_offs, mask=mask, other=-1e30)
        m = tl.maximum(m, dts_r)

    m_safe = tl.where(m > -1e29, m, tl.zeros_like(m))
    s = tl.exp2(t0 - m_safe) + tl.exp2(t1 - m_safe) + tl.exp2(t2 - m_safe)
    s += tl.exp2(t3 - m_safe) + tl.exp2(t4 - m_safe) + tl.exp2(t5 - m_safe)
    if has_splits:
        s += tl.exp2(dts_r - m_safe)

    result = tl.log2(s) + m
    tl.store(Pi_new_ptr + base + s_offs, result, mask=mask)


def wave_step_fused(Pi_W, Pibar_W, DL_const, Ebar, E, SL1_const, SL2_const,
                    sp_child1, sp_child2, leaf_term_wt, DTS_reduced=None):
    """Fused wave step without Pibar computation."""
    W, S = Pi_W.shape
    Pi_new = torch.empty_like(Pi_W)
    has_splits = DTS_reduced is not None

    BLOCK_S = 32
    grid = (W, (S + BLOCK_S - 1) // BLOCK_S)

    _wave_step_kernel[grid](
        Pi_W, Pibar_W,
        DL_const, Ebar, E, SL1_const, SL2_const,
        sp_child1, sp_child2,
        leaf_term_wt,
        DTS_reduced if has_splits else Pi_W,
        has_splits,
        Pi_new,
        W, S,
        stride_ws=S,
    )
    return Pi_new


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
    # Per-wave arrays: [W, S]
    leaf_term_ptr,
    DTS_reduced_ptr,
    has_splits: tl.constexpr,
    # Outputs
    Pi_new_ptr,          # [W, S]
    Pibar_out_ptr,       # [C, S] — write Pibar to rows [ws : ws+W]
    max_diff_ptr,        # [W] — per-row max |Pi_new - Pi_old| for convergence
    # Dimensions
    S: tl.constexpr,
    stride: tl.constexpr,
    BLOCK_S: tl.constexpr,
    FP64: tl.constexpr,
):
    """Fused kernel: uniform Pibar + DTS_L + logsumexp + convergence diff.

    Each program handles one full clade row, processing S elements in tiles.
    Pass 1 uses the online max+sum trick (single scan) for row statistics.
    Pass 2 computes Pibar inline and all DTS_L terms in one scan.
    """
    DTYPE = tl.float64 if FP64 else tl.float32
    NEG_LARGE = -1e300 if FP64 else -1e30

    w = tl.program_id(0)
    pi_base = (ws + w) * stride      # offset into global Pi/Pibar
    out_base = w * stride             # offset into [W, S] outputs

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

    # === Pass 2: Pibar + DTS_L terms + logsumexp ===
    local_max_diff = tl.full([1], value=0.0, dtype=DTYPE)
    M_SAFE_THRESH = -1e299 if FP64 else -1e29

    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        mask = s_offs < S

        # Load Pi[w, s]
        pi_w = tl.load(Pi_ptr + pi_base + s_offs, mask=mask, other=NEG_LARGE)

        # Uniform Pibar: log2(row_sum - exp2(pi - max)) + max + mt
        pi_exp = tl.exp2(pi_w - row_max)
        mt = tl.load(mt_ptr + s_offs, mask=mask, other=0.0)
        pibar_w = tl.log2(row_sum - pi_exp) + row_max + mt

        # Store Pibar to global tensor
        tl.store(Pibar_out_ptr + pi_base + s_offs, pibar_w, mask=mask)

        # Load constants
        dl_const = tl.load(DL_const_ptr + s_offs, mask=mask, other=NEG_LARGE)
        ebar = tl.load(Ebar_ptr + s_offs, mask=mask, other=NEG_LARGE)
        e_val = tl.load(E_ptr + s_offs, mask=mask, other=NEG_LARGE)
        sl1_const = tl.load(SL1_const_ptr + s_offs, mask=mask, other=NEG_LARGE)
        sl2_const = tl.load(SL2_const_ptr + s_offs, mask=mask, other=NEG_LARGE)

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

        # Convergence: max |result - pi_old| for significant entries (result > -100)
        significant = result > -100.0
        diff = tl.where(significant & mask, tl.abs(result - pi_w), tl.zeros_like(result))
        local_max_diff = tl.maximum(local_max_diff, tl.max(diff, axis=0))

    tl.store(max_diff_ptr + w, tl.max(local_max_diff, axis=0))


def wave_step_uniform_fused(Pi, Pibar, ws, W, S,
                            mt_squeezed, DL_const, Ebar, E, SL1_const, SL2_const,
                            sp_child1, sp_child2, leaf_term_wt,
                            DTS_reduced=None):
    """Fused uniform-Pibar + wave step + convergence in one kernel.

    Computes Pibar inline using the uniform transfer matrix approximation,
    then DTS_L terms and logsumexp, plus per-row convergence diff.
    Single kernel launch per iteration, eliminating the [W,S] Pibar intermediate.

    Args:
        Pi: [C, S] global Pi tensor (reads rows [ws:ws+W])
        Pibar: [C, S] global Pibar tensor (writes rows [ws:ws+W])
        ws: wave start index
        W: wave size (number of clades)
        S: number of species
        mt_squeezed: [S] max_transfer_mat
        DL_const, Ebar, E, SL1_const, SL2_const: [S] precomputed constants
        sp_child1, sp_child2: [S] long species child indices
        leaf_term_wt: [W, S]
        DTS_reduced: [W, S] or None

    Returns:
        Pi_new: [W, S] new Pi values
        max_diff: scalar, max |Pi_new - Pi_old| across all significant entries
    """
    fp64 = Pi.dtype == torch.float64
    Pi_new = torch.empty((W, S), dtype=Pi.dtype, device=Pi.device)
    max_diff_buf = torch.empty(W, dtype=Pi.dtype, device=Pi.device)
    has_splits = DTS_reduced is not None

    BLOCK_S = min(256, triton.next_power_of_2(S))
    grid = (W,)

    _wave_step_uniform_kernel[grid](
        Pi, ws,
        mt_squeezed,
        DL_const, Ebar, E, SL1_const, SL2_const,
        sp_child1, sp_child2,
        leaf_term_wt,
        DTS_reduced if has_splits else leaf_term_wt,  # dummy
        has_splits,
        Pi_new, Pibar, max_diff_buf,
        S,
        stride=S,
        BLOCK_S=BLOCK_S,
        FP64=fp64,
        num_warps=4,
    )
    max_diff = max_diff_buf.max().item()
    return Pi_new, max_diff
