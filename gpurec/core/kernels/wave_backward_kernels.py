"""Triton @triton.jit device kernels for the wave-step VJP (backward pass).

Split out of wave_backward.py (mechanical AST extraction, 2026-07-07): these
are the compiled device kernels. The host-side orchestration that launches
them lives in wave_backward.py, which re-imports every name defined here so
callers and tests keep reaching them as ``wave_backward.<name>``.
"""
import torch
import triton
import triton.language as tl

@triton.jit
def _active_mask_from_rhs_absmax_kernel(
    rhs_ptr,          # [W, S]
    active_mask_ptr,  # [W] bool
    threshold,
    S: tl.constexpr,
    stride: tl.constexpr,
    BLOCK_S: tl.constexpr,
    STRICT_GT: tl.constexpr,
    DTYPE: tl.constexpr,
):
    w = tl.program_id(0)
    row_base = w * stride
    row_max = tl.full([1], value=0.0, dtype=DTYPE)

    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        mask = s_offs < S
        rhs_val = tl.load(rhs_ptr + row_base + s_offs, mask=mask, other=0.0)
        tile_max = tl.max(tl.abs(rhs_val), axis=0)
        row_max = tl.maximum(row_max, tile_max)

    if STRICT_GT:
        active = row_max > threshold
    else:
        active = row_max >= threshold
    lane = tl.arange(0, 1)
    tl.store(active_mask_ptr + w + lane, active)


@triton.jit
def _wave_backward_uniform_2d_precompute_kernel(
    Pi_star_ptr,
    Pibar_star_ptr,
    Pi_offset_ptr,
    Pibar_offset_ptr,
    Pibar_row_max_ptr,
    dts_r_ptr,
    dts_offset_ptr,
    has_splits: tl.constexpr,
    rhs_ptr,
    active_mask_ptr,
    mt_ptr, DL_const_ptr, Ebar_ptr, E_ptr, SL1_const_ptr, SL2_const_ptr,
    receiver_log_probs_ptr,
    sp_child1_ptr, sp_child2_ptr, sp_parent_ptr,
    leaf_term_ptr,
    leaf_species_ptr,
    leaf_logp_ptr,
    family_idx_ptr,
    v_k_ptr,
    diag_ptr,
    pibar_coeff_ptr,
    p_prime_ptr,
    sl1_ptr,
    sl2_ptr,
    ws,
    W,
    S: tl.constexpr,
    stride: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_S: tl.constexpr,
    MAX_ANCESTOR_DEPTH: tl.constexpr,
    USE_LEAF_INDEX: tl.constexpr,
    HAS_LEAF_TERM: tl.constexpr,
    LEAF_LOGP_MODE: tl.constexpr,
    USE_ACTIVE_MASK: tl.constexpr,
    SKIP_INACTIVE_SCRATCH_ZERO: tl.constexpr,
    CONST_LAYOUT: tl.constexpr,
    DTYPE: tl.constexpr,
    USE_CHILD_EDGE_SELF_LOOP: tl.constexpr,
    USE_RECEIVER_WEIGHTS: tl.constexpr,
):
    """Precompute self-loop J^T coefficients for a block of rows and all species."""
    NEG_LARGE: tl.constexpr = -float("inf")

    block = tl.program_id(0)
    rows = block * BLOCK_W + tl.arange(0, BLOCK_W)
    s_offs = tl.arange(0, BLOCK_S)
    row_valid = rows < W
    species_valid = s_offs < S
    if USE_ACTIVE_MASK:
        row_active = tl.load(active_mask_ptr + rows, mask=row_valid, other=0) != 0
    else:
        row_active = row_valid
    row_mask = row_valid & row_active
    mask = species_valid[:, None] & row_mask[None, :]
    if SKIP_INACTIVE_SCRATCH_ZERO:
        store_mask = mask
    else:
        store_mask = species_valid[:, None] & row_valid[None, :]

    row_global = ws + rows
    pi_offsets = row_global[None, :] * stride + s_offs[:, None]
    out_offsets = rows[None, :] * S + s_offs[:, None]

    pi_row_offset = tl.load(
        Pi_offset_ptr + row_global, mask=row_valid, other=0.0
    )
    pibar_row_offset = tl.load(
        Pibar_offset_ptr + row_global, mask=row_valid, other=0.0
    )
    pibar_offset_corr = (pibar_row_offset - pi_row_offset).to(DTYPE)
    leaf_offset_corr = (-pi_row_offset).to(DTYPE)
    if has_splits:
        dts_row_offset = tl.load(
            dts_offset_ptr + rows, mask=row_valid, other=0.0
        )
        dts_offset_corr = (dts_row_offset - pi_row_offset).to(DTYPE)
    else:
        dts_offset_corr = tl.zeros([BLOCK_W], dtype=DTYPE)

    row_max = tl.load(Pibar_row_max_ptr + row_global, mask=row_valid, other=NEG_LARGE).to(DTYPE)
    pi_w = tl.load(Pi_star_ptr + pi_offsets, mask=mask, other=NEG_LARGE).to(DTYPE)
    pibar_w = tl.load(Pibar_star_ptr + pi_offsets, mask=mask, other=NEG_LARGE).to(DTYPE)
    row_max_safe = tl.where(row_max != NEG_LARGE, row_max, tl.zeros_like(row_max))
    if USE_RECEIVER_WEIGHTS:
        receiver_logp = tl.load(receiver_log_probs_ptr + s_offs, mask=species_valid, other=NEG_LARGE).to(DTYPE)
        p_prime = tl.exp2(receiver_logp[:, None] + pi_w - row_max_safe[None, :])
    else:
        p_prime = tl.exp2(pi_w - row_max_safe[None, :])
    row_sum = tl.sum(tl.where(mask, p_prime, tl.zeros([BLOCK_S, BLOCK_W], dtype=DTYPE)), axis=0)

    family = tl.full([BLOCK_W], value=0, dtype=tl.int64)
    const_base = tl.zeros([BLOCK_W], dtype=tl.int64)
    if CONST_LAYOUT == 1:
        const_offsets = out_offsets
    elif CONST_LAYOUT == 2:
        family = tl.load(family_idx_ptr + row_global, mask=row_valid, other=0).to(tl.int64)
        const_base = family * stride
        const_offsets = const_base[None, :] + s_offs[:, None]
    else:
        const_offsets = s_offs[:, None]

    if CONST_LAYOUT == 0:
        const_mask = species_valid[:, None]
    else:
        const_mask = mask
    dl_c = tl.load(DL_const_ptr + const_offsets, mask=const_mask, other=NEG_LARGE).to(DTYPE)
    ebar = tl.load(Ebar_ptr + const_offsets, mask=const_mask, other=NEG_LARGE).to(DTYPE)
    e_val = tl.load(E_ptr + const_offsets, mask=const_mask, other=NEG_LARGE).to(DTYPE)
    sl1_c = tl.load(SL1_const_ptr + const_offsets, mask=const_mask, other=NEG_LARGE).to(DTYPE)
    sl2_c = tl.load(SL2_const_ptr + const_offsets, mask=const_mask, other=NEG_LARGE).to(DTYPE)

    c1 = tl.load(sp_child1_ptr + s_offs, mask=species_valid, other=0)
    c2 = tl.load(sp_child2_ptr + s_offs, mask=species_valid, other=0)
    c1_valid = c1 < S
    c2_valid = c2 < S
    pi_s1 = tl.load(
        Pi_star_ptr + row_global[None, :] * stride + c1[:, None],
        mask=(species_valid & c1_valid)[:, None] & row_mask[None, :],
        other=NEG_LARGE,
    ).to(DTYPE)
    pi_s2 = tl.load(
        Pi_star_ptr + row_global[None, :] * stride + c2[:, None],
        mask=(species_valid & c2_valid)[:, None] & row_mask[None, :],
        other=NEG_LARGE,
    ).to(DTYPE)

    t0 = dl_c + pi_w
    t1 = pi_w + ebar
    t2 = pibar_w + e_val + pibar_offset_corr[None, :]
    t3 = sl1_c + pi_s1
    t4 = sl2_c + pi_s2
    if USE_LEAF_INDEX:
        leaf_species = tl.load(leaf_species_ptr + row_global, mask=row_valid, other=-1)
        leaf_hit = mask & (leaf_species[None, :] == s_offs[:, None])
        if LEAF_LOGP_MODE == 3:
            leaf_logp = tl.load(leaf_logp_ptr).to(DTYPE)
            t5 = tl.where(leaf_hit, leaf_logp, NEG_LARGE)
        elif LEAF_LOGP_MODE == 1:
            leaf_logp = tl.load(leaf_logp_ptr + family, mask=row_valid, other=NEG_LARGE).to(DTYPE)
            t5 = tl.where(leaf_hit, leaf_logp[None, :], NEG_LARGE)
        elif LEAF_LOGP_MODE == 2:
            leaf_logp = tl.load(
                leaf_logp_ptr + const_base[None, :] + s_offs[:, None],
                mask=leaf_hit,
                other=NEG_LARGE,
            ).to(DTYPE)
            t5 = tl.where(leaf_hit, leaf_logp, NEG_LARGE)
        else:
            leaf_logp = tl.load(leaf_logp_ptr + s_offs, mask=species_valid, other=NEG_LARGE).to(DTYPE)
            t5 = tl.where(leaf_hit, leaf_logp[:, None], NEG_LARGE)
    elif HAS_LEAF_TERM:
        t5 = tl.load(leaf_term_ptr + out_offsets, mask=mask, other=NEG_LARGE).to(DTYPE)
    else:
        t5 = tl.full([BLOCK_S, BLOCK_W], value=NEG_LARGE, dtype=DTYPE)
    if USE_LEAF_INDEX or HAS_LEAF_TERM:
        t5 += leaf_offset_corr[None, :]

    m = tl.maximum(t0, t1)
    m = tl.maximum(m, t2)
    m = tl.maximum(m, t3)
    m = tl.maximum(m, t4)
    m = tl.maximum(m, t5)
    m_safe = tl.where(m != NEG_LARGE, m, tl.zeros([BLOCK_S, BLOCK_W], dtype=DTYPE))
    e0 = tl.exp2(t0 - m_safe)
    e1 = tl.exp2(t1 - m_safe)
    e2 = tl.exp2(t2 - m_safe)
    e3 = tl.exp2(t3 - m_safe)
    e4 = tl.exp2(t4 - m_safe)
    e5 = tl.exp2(t5 - m_safe)
    dts_l_sum = e0 + e1 + e2 + e3 + e4 + e5
    inv_sum = tl.where(dts_l_sum > 0.0, 1.0 / dts_l_sum, tl.zeros_like(dts_l_sum))

    if has_splits:
        dts_r = tl.load(dts_r_ptr + out_offsets, mask=mask, other=NEG_LARGE).to(DTYPE)
        dts_r += dts_offset_corr[None, :]
        dts_l = tl.log2(dts_l_sum) + m
        pi_new_m = tl.maximum(dts_l, dts_r)
        pi_new_ms = tl.where(pi_new_m != NEG_LARGE, pi_new_m, tl.zeros_like(pi_new_m))
        pi_new = tl.log2(tl.exp2(dts_l - pi_new_ms) + tl.exp2(dts_r - pi_new_ms)) + pi_new_m
        w_L = tl.where(dts_l != NEG_LARGE, tl.exp2(dts_l - pi_new), tl.zeros_like(dts_l))
    else:
        w_L = tl.full([BLOCK_S, BLOCK_W], value=1.0, dtype=DTYPE)

    ancestor_sum = tl.zeros([BLOCK_S, BLOCK_W], dtype=DTYPE)
    cur = s_offs
    for _depth in range(MAX_ANCESTOR_DEPTH):
        cur_valid = species_valid & (cur >= 0) & (cur < S)
        pi_anc = tl.load(
            Pi_star_ptr + row_global[None, :] * stride + cur[:, None],
            mask=cur_valid[:, None] & row_mask[None, :],
            other=NEG_LARGE,
        ).to(DTYPE)
        if USE_RECEIVER_WEIGHTS:
            receiver_logp_anc = tl.load(
                receiver_log_probs_ptr + cur,
                mask=cur_valid,
                other=NEG_LARGE,
            ).to(DTYPE)
            ancestor_sum += tl.where(
                cur_valid[:, None] & row_mask[None, :],
                tl.exp2(receiver_logp_anc[:, None] + pi_anc - row_max_safe[None, :]),
                tl.zeros([BLOCK_S, BLOCK_W], dtype=DTYPE),
            )
        else:
            ancestor_sum += tl.where(
                cur_valid[:, None] & row_mask[None, :],
                tl.exp2(pi_anc - row_max_safe[None, :]),
                tl.zeros([BLOCK_S, BLOCK_W], dtype=DTYPE),
            )
        cur = tl.load(sp_parent_ptr + cur, mask=cur_valid, other=-1)
    denom = row_sum[None, :] - ancestor_sum
    inv_denom = tl.where(denom > 0.0, 1.0 / denom, tl.zeros_like(denom))

    diag_wt = w_L * (e0 + e1) * inv_sum
    pibar_u_coeff = w_L * e2 * inv_sum * inv_denom
    sl1_wt = w_L * e3 * inv_sum
    sl2_wt = w_L * e4 * inv_sum

    zero = tl.zeros([BLOCK_S, BLOCK_W], dtype=DTYPE)
    rhs_val = tl.load(rhs_ptr + out_offsets, mask=mask, other=0.0).to(DTYPE)
    tl.store(v_k_ptr + out_offsets, tl.where(mask, rhs_val, zero), mask=store_mask)
    tl.store(diag_ptr + out_offsets, tl.where(mask, diag_wt, zero), mask=store_mask)
    tl.store(pibar_coeff_ptr + out_offsets, tl.where(mask, pibar_u_coeff, zero), mask=store_mask)
    tl.store(p_prime_ptr + out_offsets, tl.where(mask, p_prime, zero), mask=store_mask)
    if USE_CHILD_EDGE_SELF_LOOP:
        child1_offsets = rows[None, :] * S + c1[:, None]
        child2_offsets = rows[None, :] * S + c2[:, None]
        child1_mask = (species_valid & c1_valid)[:, None] & row_mask[None, :]
        child2_mask = (species_valid & c2_valid)[:, None] & row_mask[None, :]
        tl.store(sl1_ptr + child1_offsets, sl1_wt, mask=child1_mask)
        tl.store(sl1_ptr + child2_offsets, sl2_wt, mask=child2_mask)
    else:
        tl.store(sl1_ptr + out_offsets, tl.where(mask, sl1_wt, zero), mask=store_mask)
        tl.store(sl2_ptr + out_offsets, tl.where(mask, sl2_wt, zero), mask=store_mask)


@triton.jit
def _wave_backward_uniform_2d_jt_kernel(
    term_in_ptr,
    term_out_ptr,
    rhs_update_ptr,
    active_mask_ptr,
    diag_ptr,
    pibar_coeff_ptr,
    p_prime_ptr,
    sl1_ptr,
    sl2_ptr,
    sp_child1_ptr,
    sp_child2_ptr,
    sp_parent_ptr,
    compact_level_ptr,
    compact_level_parent_ptr,
    compact_level_child1_ptr,
    compact_level_child2_ptr,
    pibar_corr_ptr,
    v_k_ptr,
    W,
    S: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_NODES: tl.constexpr,
    N_LEVELS: tl.constexpr,
    USE_ACTIVE_MASK: tl.constexpr,
    SKIP_INACTIVE_SCRATCH_ZERO: tl.constexpr,
    FIXED_POINT_UPDATE: tl.constexpr,
    DTYPE: tl.constexpr,
    USE_CHILD_EDGE_SELF_LOOP: tl.constexpr,
    OUTPUT_A: tl.constexpr,
    ACCUMULATE_V: tl.constexpr,
):
    """Apply one self-loop J^T term using in-program bottom-up tree reduction."""
    block = tl.program_id(0)
    rows = block * BLOCK_W + tl.arange(0, BLOCK_W)
    s_offs = tl.arange(0, BLOCK_S)
    row_valid = rows < W
    species_valid = s_offs < S
    if USE_ACTIVE_MASK:
        row_active = tl.load(active_mask_ptr + rows, mask=row_valid, other=0) != 0
    else:
        row_active = row_valid
    row_mask = row_valid & row_active
    mask = species_valid[:, None] & row_mask[None, :]
    if SKIP_INACTIVE_SCRATCH_ZERO:
        store_mask = mask
    else:
        store_mask = species_valid[:, None] & row_valid[None, :]
    offsets = rows[None, :] * S + s_offs[:, None]

    term_val = tl.load(term_in_ptr + offsets, mask=mask, other=0.0).to(DTYPE)
    pibar_u_coeff = tl.load(pibar_coeff_ptr + offsets, mask=mask, other=0.0).to(DTYPE)
    u_d = term_val * pibar_u_coeff
    A = tl.sum(tl.where(mask, u_d, tl.zeros([BLOCK_S, BLOCK_W], dtype=DTYPE)), axis=0)
    tl.store(pibar_corr_ptr + offsets, tl.where(mask, u_d, tl.zeros_like(u_d)), mask=store_mask)

    tl.debug_barrier()

    for level in range(0, N_LEVELS):
        level_start = tl.load(compact_level_ptr + level)
        level_end = tl.load(compact_level_ptr + level + 1)
        node_start = level_start
        while node_start < level_end:
            node_offs = node_start + tl.arange(0, BLOCK_NODES)
            node_mask = node_offs < level_end
            parent = tl.load(compact_level_parent_ptr + node_offs, mask=node_mask, other=0)
            c1 = tl.load(compact_level_child1_ptr + node_offs, mask=node_mask, other=S)
            c2 = tl.load(compact_level_child2_ptr + node_offs, mask=node_mask, other=S)
            reduce_mask = node_mask[:, None] & row_mask[None, :]
            row_base = rows[None, :] * S
            parent_val = tl.load(
                pibar_corr_ptr + row_base + parent[:, None],
                mask=reduce_mask,
                other=0.0,
            ).to(DTYPE)
            c1_val = tl.load(
                pibar_corr_ptr + row_base + c1[:, None],
                mask=reduce_mask & (c1 < S)[:, None],
                other=0.0,
            ).to(DTYPE)
            c2_val = tl.load(
                pibar_corr_ptr + row_base + c2[:, None],
                mask=reduce_mask & (c2 < S)[:, None],
                other=0.0,
            ).to(DTYPE)
            tl.store(
                pibar_corr_ptr + row_base + parent[:, None],
                parent_val + c1_val + c2_val,
                mask=reduce_mask,
            )
            node_start += BLOCK_NODES
        tl.debug_barrier()

    tl.debug_barrier()

    corr = tl.load(pibar_corr_ptr + offsets, mask=mask, other=0.0).to(DTYPE)
    diag_wt = tl.load(diag_ptr + offsets, mask=mask, other=0.0).to(DTYPE)
    p_prime = tl.load(p_prime_ptr + offsets, mask=mask, other=0.0).to(DTYPE)
    base = term_val * diag_wt + p_prime * (A[None, :] - corr)

    if USE_CHILD_EDGE_SELF_LOOP:
        parent = tl.load(sp_parent_ptr + s_offs, mask=species_valid, other=-1)
        parent_valid = species_valid & (parent >= 0) & (parent < S)
        row_base = rows[None, :] * S
        parent_mask = parent_valid[:, None] & row_mask[None, :]
        parent_term = tl.load(
            term_in_ptr + row_base + parent[:, None],
            mask=parent_mask,
            other=0.0,
        ).to(DTYPE)
        edge_wt = tl.load(sl1_ptr + offsets, mask=parent_mask, other=0.0).to(DTYPE)
        result = base + parent_term * edge_wt
    else:
        tl.store(term_out_ptr + offsets, tl.where(mask, base, tl.zeros_like(base)), mask=store_mask)

        tl.debug_barrier()

        c1 = tl.load(sp_child1_ptr + s_offs, mask=species_valid, other=S)
        c2 = tl.load(sp_child2_ptr + s_offs, mask=species_valid, other=S)
        sl1_wt = tl.load(sl1_ptr + offsets, mask=mask, other=0.0).to(DTYPE)
        sl2_wt = tl.load(sl2_ptr + offsets, mask=mask, other=0.0).to(DTYPE)
        row_base = rows[None, :] * S
        c1_mask = (species_valid & (c1 < S))[:, None] & row_mask[None, :]
        c2_mask = (species_valid & (c2 < S))[:, None] & row_mask[None, :]
        c1_cur = tl.load(term_out_ptr + row_base + c1[:, None], mask=c1_mask, other=0.0).to(DTYPE)
        c2_cur = tl.load(term_out_ptr + row_base + c2[:, None], mask=c2_mask, other=0.0).to(DTYPE)
        tl.store(term_out_ptr + row_base + c1[:, None], c1_cur + term_val * sl1_wt, mask=c1_mask)
        tl.store(term_out_ptr + row_base + c2[:, None], c2_cur + term_val * sl2_wt, mask=c2_mask)

        tl.debug_barrier()

        result = tl.load(term_out_ptr + offsets, mask=mask, other=0.0).to(DTYPE)

    out_val = term_val - result if OUTPUT_A else result
    tl.store(
        term_out_ptr + offsets,
        tl.where(mask, out_val, tl.zeros_like(out_val)),
        mask=store_mask,
    )

    if FIXED_POINT_UPDATE:
        rhs_val = tl.load(rhs_update_ptr + offsets, mask=mask, other=0.0).to(DTYPE)
        tl.store(
            v_k_ptr + offsets,
            tl.where(mask, rhs_val + result, tl.zeros_like(result)),
            mask=store_mask,
        )
    elif ACCUMULATE_V:
        v_prev = tl.load(v_k_ptr + offsets, mask=mask, other=0.0).to(DTYPE)
        tl.store(v_k_ptr + offsets, v_prev + result, mask=mask)


@triton.jit
def _receiver_grad_from_pibar_self_loop_kernel(
    v_k_ptr,
    active_mask_ptr,
    pibar_coeff_ptr,
    p_prime_ptr,
    compact_level_ptr,
    compact_level_parent_ptr,
    compact_level_child1_ptr,
    compact_level_child2_ptr,
    pibar_corr_ptr,
    grad_receiver_log_probs_ptr,
    W,
    S: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_NODES: tl.constexpr,
    N_LEVELS: tl.constexpr,
    USE_ACTIVE_MASK: tl.constexpr,
    DTYPE: tl.constexpr,
):
    block = tl.program_id(0)
    rows = block * BLOCK_W + tl.arange(0, BLOCK_W)
    s_offs = tl.arange(0, BLOCK_S)
    row_valid = rows < W
    species_valid = s_offs < S
    if USE_ACTIVE_MASK:
        row_active = tl.load(active_mask_ptr + rows, mask=row_valid, other=0) != 0
    else:
        row_active = row_valid
    row_mask = row_valid & row_active
    mask = species_valid[:, None] & row_mask[None, :]
    store_mask = species_valid[:, None] & row_valid[None, :]
    offsets = rows[None, :] * S + s_offs[:, None]

    term_val = tl.load(v_k_ptr + offsets, mask=mask, other=0.0).to(DTYPE)
    pibar_u_coeff = tl.load(pibar_coeff_ptr + offsets, mask=mask, other=0.0).to(DTYPE)
    u_d = term_val * pibar_u_coeff
    zero = tl.zeros([BLOCK_S, BLOCK_W], dtype=DTYPE)
    A = tl.sum(tl.where(mask, u_d, zero), axis=0)
    tl.store(pibar_corr_ptr + offsets, tl.where(mask, u_d, zero), mask=store_mask)

    tl.debug_barrier()

    for level in range(0, N_LEVELS):
        level_start = tl.load(compact_level_ptr + level)
        level_end = tl.load(compact_level_ptr + level + 1)
        node_start = level_start
        while node_start < level_end:
            node_offs = node_start + tl.arange(0, BLOCK_NODES)
            node_mask = node_offs < level_end
            parent = tl.load(compact_level_parent_ptr + node_offs, mask=node_mask, other=0)
            c1 = tl.load(compact_level_child1_ptr + node_offs, mask=node_mask, other=S)
            c2 = tl.load(compact_level_child2_ptr + node_offs, mask=node_mask, other=S)
            reduce_mask = node_mask[:, None] & row_mask[None, :]
            row_base = rows[None, :] * S
            parent_val = tl.load(
                pibar_corr_ptr + row_base + parent[:, None],
                mask=reduce_mask,
                other=0.0,
            ).to(DTYPE)
            c1_val = tl.load(
                pibar_corr_ptr + row_base + c1[:, None],
                mask=reduce_mask & (c1 < S)[:, None],
                other=0.0,
            ).to(DTYPE)
            c2_val = tl.load(
                pibar_corr_ptr + row_base + c2[:, None],
                mask=reduce_mask & (c2 < S)[:, None],
                other=0.0,
            ).to(DTYPE)
            tl.store(
                pibar_corr_ptr + row_base + parent[:, None],
                parent_val + c1_val + c2_val,
                mask=reduce_mask,
            )
            node_start += BLOCK_NODES
        tl.debug_barrier()

    corr = tl.load(pibar_corr_ptr + offsets, mask=mask, other=0.0).to(DTYPE)
    p_prime = tl.load(p_prime_ptr + offsets, mask=mask, other=0.0).to(DTYPE)
    contrib = p_prime * (A[None, :] - corr)
    species_contrib = tl.sum(tl.where(mask, contrib, zero), axis=1)
    tl.atomic_add(
        grad_receiver_log_probs_ptr + s_offs,
        species_contrib,
        sem="relaxed",
        mask=species_valid,
    )


@triton.jit
def _wave_backward_uniform_param_store_kernel(
    Pi_star_ptr,
    Pibar_star_ptr,
    Pi_offset_ptr,
    Pibar_offset_ptr,
    dts_r_ptr,
    dts_offset_ptr,
    has_splits: tl.constexpr,
    v_k_ptr,
    active_mask_ptr,
    mt_ptr, DL_const_ptr, Ebar_ptr, E_ptr, SL1_const_ptr, SL2_const_ptr,
    sp_child1_ptr, sp_child2_ptr,
    leaf_term_ptr,
    leaf_species_ptr,
    leaf_logp_ptr,
    family_idx_ptr,
    grad_log_pD_ptr,
    grad_log_pS_ptr,
    grad_E_ptr,
    grad_Ebar_ptr,
    grad_E_s1_ptr,
    grad_E_s2_ptr,
    grad_mt_ptr,
    aw0_ptr,
    aw1_ptr,
    aw2_ptr,
    aw345_ptr,
    aw3_ptr,
    aw4_ptr,
    ws,
    W,
    S: tl.constexpr,
    stride: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_S: tl.constexpr,
    USE_LEAF_INDEX: tl.constexpr,
    HAS_LEAF_TERM: tl.constexpr,
    LEAF_LOGP_MODE: tl.constexpr,
    USE_ACTIVE_MASK: tl.constexpr,
    CONST_LAYOUT: tl.constexpr,
    ACCUM_GRADS: tl.constexpr,
    PARAM_GRAD_VECTOR: tl.constexpr,
    DTYPE: tl.constexpr,
):
    """Store per-element self-loop parameter VJP contributions after Neumann."""
    NEG_LARGE: tl.constexpr = -float("inf")

    block = tl.program_id(0)
    rows = block * BLOCK_W + tl.arange(0, BLOCK_W)
    s_offs = tl.arange(0, BLOCK_S)
    row_valid = rows < W
    species_valid = s_offs < S
    if USE_ACTIVE_MASK:
        row_active = tl.load(active_mask_ptr + rows, mask=row_valid, other=0) != 0
    else:
        row_active = row_valid
    row_mask = row_valid & row_active
    mask = species_valid[:, None] & row_mask[None, :]
    store_mask = species_valid[:, None] & row_valid[None, :]
    row_global = ws + rows
    pi_offsets = row_global[None, :] * stride + s_offs[:, None]
    out_offsets = rows[None, :] * S + s_offs[:, None]

    pi_row_offset = tl.load(
        Pi_offset_ptr + row_global, mask=row_valid, other=0.0
    )
    pibar_row_offset = tl.load(
        Pibar_offset_ptr + row_global, mask=row_valid, other=0.0
    )
    pibar_offset_corr = (pibar_row_offset - pi_row_offset).to(DTYPE)
    leaf_offset_corr = (-pi_row_offset).to(DTYPE)
    if has_splits:
        dts_row_offset = tl.load(
            dts_offset_ptr + rows, mask=row_valid, other=0.0
        )
        dts_offset_corr = (dts_row_offset - pi_row_offset).to(DTYPE)
    else:
        dts_offset_corr = tl.zeros([BLOCK_W], dtype=DTYPE)

    family = tl.full([BLOCK_W], value=0, dtype=tl.int64)
    const_base = tl.zeros([BLOCK_W], dtype=tl.int64)
    if CONST_LAYOUT == 1:
        const_offsets = out_offsets
    elif CONST_LAYOUT == 2:
        family = tl.load(family_idx_ptr + row_global, mask=row_valid, other=0).to(tl.int64)
        const_base = family * stride
        const_offsets = const_base[None, :] + s_offs[:, None]
    else:
        const_offsets = s_offs[:, None]

    const_mask = species_valid[:, None] if CONST_LAYOUT == 0 else mask
    pi_w = tl.load(Pi_star_ptr + pi_offsets, mask=mask, other=NEG_LARGE).to(DTYPE)
    pibar_w = tl.load(Pibar_star_ptr + pi_offsets, mask=mask, other=NEG_LARGE).to(DTYPE)
    v_k_val = tl.load(v_k_ptr + out_offsets, mask=mask, other=0.0).to(DTYPE)
    dl_c = tl.load(DL_const_ptr + const_offsets, mask=const_mask, other=NEG_LARGE).to(DTYPE)
    ebar = tl.load(Ebar_ptr + const_offsets, mask=const_mask, other=NEG_LARGE).to(DTYPE)
    e_val = tl.load(E_ptr + const_offsets, mask=const_mask, other=NEG_LARGE).to(DTYPE)
    sl1_c = tl.load(SL1_const_ptr + const_offsets, mask=const_mask, other=NEG_LARGE).to(DTYPE)
    sl2_c = tl.load(SL2_const_ptr + const_offsets, mask=const_mask, other=NEG_LARGE).to(DTYPE)

    c1 = tl.load(sp_child1_ptr + s_offs, mask=species_valid, other=0)
    c2 = tl.load(sp_child2_ptr + s_offs, mask=species_valid, other=0)
    c1_valid = c1 < S
    c2_valid = c2 < S
    pi_s1 = tl.load(
        Pi_star_ptr + row_global[None, :] * stride + c1[:, None],
        mask=(species_valid & c1_valid)[:, None] & row_mask[None, :],
        other=NEG_LARGE,
    ).to(DTYPE)
    pi_s2 = tl.load(
        Pi_star_ptr + row_global[None, :] * stride + c2[:, None],
        mask=(species_valid & c2_valid)[:, None] & row_mask[None, :],
        other=NEG_LARGE,
    ).to(DTYPE)

    t0 = dl_c + pi_w
    t1 = pi_w + ebar
    t2 = pibar_w + e_val + pibar_offset_corr[None, :]
    t3 = sl1_c + pi_s1
    t4 = sl2_c + pi_s2
    if USE_LEAF_INDEX:
        leaf_species = tl.load(leaf_species_ptr + row_global, mask=row_valid, other=-1)
        leaf_hit = mask & (leaf_species[None, :] == s_offs[:, None])
        if LEAF_LOGP_MODE == 3:
            leaf_logp = tl.load(leaf_logp_ptr).to(DTYPE)
            t5 = tl.where(leaf_hit, leaf_logp, NEG_LARGE)
        elif LEAF_LOGP_MODE == 1:
            leaf_logp = tl.load(leaf_logp_ptr + family, mask=row_valid, other=NEG_LARGE).to(DTYPE)
            t5 = tl.where(leaf_hit, leaf_logp[None, :], NEG_LARGE)
        elif LEAF_LOGP_MODE == 2:
            leaf_logp = tl.load(
                leaf_logp_ptr + const_base[None, :] + s_offs[:, None],
                mask=leaf_hit,
                other=NEG_LARGE,
            ).to(DTYPE)
            t5 = tl.where(leaf_hit, leaf_logp, NEG_LARGE)
        else:
            leaf_logp = tl.load(leaf_logp_ptr + s_offs, mask=species_valid, other=NEG_LARGE).to(DTYPE)
            t5 = tl.where(leaf_hit, leaf_logp[:, None], NEG_LARGE)
    elif HAS_LEAF_TERM:
        t5 = tl.load(leaf_term_ptr + out_offsets, mask=mask, other=NEG_LARGE).to(DTYPE)
    else:
        t5 = tl.full([BLOCK_S, BLOCK_W], value=NEG_LARGE, dtype=DTYPE)
    if USE_LEAF_INDEX or HAS_LEAF_TERM:
        t5 += leaf_offset_corr[None, :]

    m = tl.maximum(t0, t1)
    m = tl.maximum(m, t2)
    m = tl.maximum(m, t3)
    m = tl.maximum(m, t4)
    m = tl.maximum(m, t5)
    m_safe = tl.where(m != NEG_LARGE, m, tl.zeros([BLOCK_S, BLOCK_W], dtype=DTYPE))
    e0 = tl.exp2(t0 - m_safe)
    e1 = tl.exp2(t1 - m_safe)
    e2 = tl.exp2(t2 - m_safe)
    e3 = tl.exp2(t3 - m_safe)
    e4 = tl.exp2(t4 - m_safe)
    e5 = tl.exp2(t5 - m_safe)
    dts_l_sum = e0 + e1 + e2 + e3 + e4 + e5
    inv_sum = tl.where(dts_l_sum > 0.0, 1.0 / dts_l_sum, tl.zeros_like(dts_l_sum))

    if has_splits:
        dts_r = tl.load(dts_r_ptr + out_offsets, mask=mask, other=NEG_LARGE).to(DTYPE)
        dts_r += dts_offset_corr[None, :]
        dts_l = tl.log2(dts_l_sum) + m
        pi_new_m = tl.maximum(dts_l, dts_r)
        pi_new_ms = tl.where(pi_new_m != NEG_LARGE, pi_new_m, tl.zeros_like(pi_new_m))
        pi_new = tl.log2(tl.exp2(dts_l - pi_new_ms) + tl.exp2(dts_r - pi_new_ms)) + pi_new_m
        w_L = tl.where(dts_l != NEG_LARGE, tl.exp2(dts_l - pi_new), tl.zeros_like(dts_l))
    else:
        w_L = tl.full([BLOCK_S, BLOCK_W], value=1.0, dtype=DTYPE)

    alpha = v_k_val * w_L
    _aw0 = alpha * e0 * inv_sum
    _aw1 = alpha * e1 * inv_sum
    _aw2 = alpha * e2 * inv_sum
    _aw3 = alpha * e3 * inv_sum
    _aw4 = alpha * e4 * inv_sum
    _aw5 = alpha * e5 * inv_sum
    _aw345 = _aw3 + _aw4 + _aw5
    zero = tl.zeros([BLOCK_S, BLOCK_W], dtype=DTYPE)
    if ACCUM_GRADS:
        aw0_s = tl.sum(tl.where(mask, _aw0, zero), axis=1)
        aw1_s = tl.sum(tl.where(mask, _aw1, zero), axis=1)
        aw2_s = tl.sum(tl.where(mask, _aw2, zero), axis=1)
        aw345_s = tl.sum(tl.where(mask, _aw345, zero), axis=1)
        aw3_s = tl.sum(tl.where(mask, _aw3, zero), axis=1)
        aw4_s = tl.sum(tl.where(mask, _aw4, zero), axis=1)
        if PARAM_GRAD_VECTOR:
            tl.atomic_add(
                grad_log_pD_ptr + s_offs,
                aw0_s,
                sem="relaxed",
                mask=species_valid,
            )
            tl.atomic_add(
                grad_log_pS_ptr + s_offs,
                aw345_s,
                sem="relaxed",
                mask=species_valid,
            )
        else:
            tl.atomic_add(grad_log_pD_ptr, tl.sum(aw0_s, axis=0), sem="relaxed")
            tl.atomic_add(grad_log_pS_ptr, tl.sum(aw345_s, axis=0), sem="relaxed")
        tl.atomic_add(
            grad_E_ptr + s_offs,
            aw0_s + aw2_s,
            sem="relaxed",
            mask=species_valid,
        )
        tl.atomic_add(
            grad_Ebar_ptr + s_offs,
            aw1_s,
            sem="relaxed",
            mask=species_valid,
        )
        tl.atomic_add(
            grad_E_s1_ptr + s_offs,
            aw4_s,
            sem="relaxed",
            mask=species_valid,
        )
        tl.atomic_add(
            grad_E_s2_ptr + s_offs,
            aw3_s,
            sem="relaxed",
            mask=species_valid,
        )
        tl.atomic_add(
            grad_mt_ptr + s_offs,
            aw2_s,
            sem="relaxed",
            mask=species_valid,
        )
    else:
        tl.store(aw0_ptr + out_offsets, tl.where(mask, _aw0, zero), mask=store_mask)
        tl.store(aw1_ptr + out_offsets, tl.where(mask, _aw1, zero), mask=store_mask)
        tl.store(aw2_ptr + out_offsets, tl.where(mask, _aw2, zero), mask=store_mask)
        tl.store(aw345_ptr + out_offsets, tl.where(mask, _aw345, zero), mask=store_mask)
        tl.store(aw3_ptr + out_offsets, tl.where(mask, _aw3, zero), mask=store_mask)
        tl.store(aw4_ptr + out_offsets, tl.where(mask, _aw4, zero), mask=store_mask)


@triton.jit
def _dts_cross_backward_accum_kernel(
    # Converged values [C, S]
    Pi_star_ptr,
    Pibar_star_ptr,
    Pi_offset_ptr,
    Pibar_offset_ptr,
    # Neumann-solved adjoint [W, S]
    v_k_ptr,
    active_mask_ptr,   # optional [W] bool parent row activity mask
    # Split metadata
    sl_ptr,            # [n_ws] int64 — left child global clade index
    sr_ptr,            # [n_ws] int64 — right child global clade index
    reduce_idx_ptr,    # [n_ws] int64 — wave-local parent index
    wlsp_ptr,          # [n_ws] float — log split probability (squeezed)
    # Params: scalar [1], shared species [S], family scalar [G], or [G, S]
    log_pD_arg,        # [1] scalar tensor or Python float
    log_pS_arg,        # [1] scalar tensor or Python float
    family_idx_ptr,    # optional [C] clade -> family id
    # Species children [S] int64
    sp_child1_ptr,
    sp_child2_ptr,
    # Outputs
    accumulated_rhs_ptr,  # [C, S], direct Pi adjoints updated atomically
    grad_Pibar_l_ptr,     # [n_ws, S]
    grad_Pibar_r_ptr,     # [n_ws, S]
    param_pD_ptr,         # [n_ws]
    param_pS_ptr,         # [n_ws]
    grad_log_pD_ptr,      # optional scalar accumulation target
    grad_log_pS_ptr,      # optional scalar accumulation target
    grad_mt_ptr,          # optional scalar/[S] accumulation target
    grad_mt_partial_ptr,  # optional [ceil(n_ws/tile_splits), S] two-stage vector accumulation
    pibar_ud_ptr,         # optional [2 * n_ws, S] initial Pibar VJP subtree values
    pibar_A_ptr,          # optional [2 * n_ws] row sums of pibar_ud
    pibar_side_active_ptr, # optional [2 * n_ws] exact nonzero u_d row mask
    mt_ptr,               # optional [S] max transfer mat for Pibar denom reuse
    pibar_row_max_ptr,    # optional [C] Pi-row max from forward uniform Pibar
    side_active_threshold_ptr,
    # Dimensions
    ws,                # wave start offset (parent row = ws + reduce_idx)
    S: tl.constexpr,
    stride_C: tl.constexpr,
    BLOCK_S: tl.constexpr,
    USE_ACTIVE_MASK: tl.constexpr,
    USE_ATOMICS: tl.constexpr,
    MERGE_S_TERM: tl.constexpr,
    DEVICE_SCALAR_PARAMS: tl.constexpr,
    PARAM_LAYOUT: tl.constexpr,
    PARAM_GRAD_LAYOUT: tl.constexpr,
    MT_LAYOUT: tl.constexpr,
    GRAD_MT_LAYOUT: tl.constexpr,
    ACCUM_PARAM_REDUCTIONS: tl.constexpr,
    ACCUM_MT_REDUCTION: tl.constexpr,
    GRAD_MT_SCALAR: tl.constexpr,
    GRAD_MT_TWO_STAGE: tl.constexpr,
    GRAD_MT_TILE_SPLITS: tl.constexpr,
    OUTPUT_PIBAR_UD: tl.constexpr,
    OUTPUT_SIDE_ACTIVE: tl.constexpr,
    SIDE_ACTIVE_THRESHOLD_ENABLED: tl.constexpr,
    SKIP_INACTIVE_PIBAR_OUTPUT_ZERO: tl.constexpr,
    DTYPE: tl.constexpr,
):
    """DTS cross-clade backward with direct accumulation of Pi adjoints.

    It writes direct Pi contributions into accumulated_rhs instead of materializing
    grad_Pi_l/grad_Pi_r and relying on two PyTorch index_add_ calls.
    Pibar adjoints are still materialized because they feed the uniform Pibar
    VJP kernel.
    """
    NEG_LARGE: tl.constexpr = -float("inf")

    i = tl.program_id(0)

    sl = tl.load(sl_ptr + i).to(tl.int64)
    sr = tl.load(sr_ptr + i).to(tl.int64)
    parent_w = tl.load(reduce_idx_ptr + i).to(tl.int64)
    wlsp = tl.load(wlsp_ptr + i).to(DTYPE)
    if USE_ACTIVE_MASK:
        parent_active = tl.load(active_mask_ptr + parent_w)
        if parent_active == 0:
            out_base = i * S
            ud_l_base = i * S
            ud_r_base = (tl.program_id(0) + 0 + tl.num_programs(0)) * S
            zero_scalar = tl.zeros((1,), dtype=DTYPE)
            _scalar_off = tl.arange(0, 1)
            if not ACCUM_PARAM_REDUCTIONS:
                tl.store(param_pD_ptr + i + _scalar_off, zero_scalar)
                tl.store(param_pS_ptr + i + _scalar_off, zero_scalar)
            if OUTPUT_PIBAR_UD:
                if OUTPUT_SIDE_ACTIVE:
                    tl.store(pibar_side_active_ptr + i + _scalar_off, 0)
                    tl.store(pibar_side_active_ptr + tl.num_programs(0) + i + _scalar_off, 0)
                if SKIP_INACTIVE_PIBAR_OUTPUT_ZERO:
                    return
                tl.store(pibar_A_ptr + i + _scalar_off, zero_scalar)
                tl.store(pibar_A_ptr + tl.num_programs(0) + i + _scalar_off, zero_scalar)
            for s_start in range(0, S, BLOCK_S):
                s_offs = s_start + tl.arange(0, BLOCK_S)
                mask = s_offs < S
                zero = tl.zeros([BLOCK_S], dtype=DTYPE)
                if OUTPUT_PIBAR_UD:
                    tl.store(pibar_ud_ptr + ud_l_base + s_offs, zero, mask=mask)
                    tl.store(pibar_ud_ptr + ud_r_base + s_offs, zero, mask=mask)
                else:
                    tl.store(grad_Pibar_l_ptr + out_base + s_offs, zero, mask=mask)
                    tl.store(grad_Pibar_r_ptr + out_base + s_offs, zero, mask=mask)
            return
    else:
        parent_active = True

    parent_global = ws + parent_w
    if (
        PARAM_LAYOUT == 2
        or PARAM_LAYOUT == 3
        or PARAM_GRAD_LAYOUT == 2
        or PARAM_GRAD_LAYOUT == 3
    ):
        parent_family = tl.load(family_idx_ptr + parent_global).to(tl.int64)
    else:
        parent_family = 0

    if MT_LAYOUT == 1 or GRAD_MT_LAYOUT == 1:
        family_l = tl.load(family_idx_ptr + sl).to(tl.int64)
        family_r = tl.load(family_idx_ptr + sr).to(tl.int64)
    else:
        family_l = 0
        family_r = 0

    if PARAM_LAYOUT == 0 and DEVICE_SCALAR_PARAMS:
        log_pD = tl.load(log_pD_arg).to(DTYPE)
        log_pS = tl.load(log_pS_arg).to(DTYPE)
    elif PARAM_LAYOUT == 0:
        log_pD = log_pD_arg
        log_pS = log_pS_arg
    elif PARAM_LAYOUT == 2:
        log_pD = tl.load(log_pD_arg + parent_family).to(DTYPE)
        log_pS = tl.load(log_pS_arg + parent_family).to(DTYPE)
    else:
        log_pD = tl.zeros((1,), dtype=DTYPE)
        log_pS = tl.zeros((1,), dtype=DTYPE)

    pi_l_base = sl * stride_C
    pi_r_base = sr * stride_C
    pibar_l_base = sl * stride_C
    pibar_r_base = sr * stride_C
    parent_pi_base = (ws + parent_w) * stride_C
    parent_vk_base = parent_w * S
    out_base = i * S

    pi_offset_l = tl.load(Pi_offset_ptr + sl)
    pi_offset_r = tl.load(Pi_offset_ptr + sr)
    pi_offset_parent = tl.load(Pi_offset_ptr + parent_global)
    pibar_offset_l = tl.load(Pibar_offset_ptr + sl)
    pibar_offset_r = tl.load(Pibar_offset_ptr + sr)
    corr0 = (pi_offset_l + pi_offset_r - pi_offset_parent).to(DTYPE)
    corr1 = (pi_offset_l + pibar_offset_r - pi_offset_parent).to(DTYPE)
    corr2 = (pi_offset_r + pibar_offset_l - pi_offset_parent).to(DTYPE)
    inv_denom_corr_l = (pi_offset_l - pibar_offset_l).to(DTYPE)
    inv_denom_corr_r = (pi_offset_r - pibar_offset_r).to(DTYPE)

    sum_pD = tl.zeros((1,), dtype=DTYPE)
    sum_pS = tl.zeros((1,), dtype=DTYPE)
    sum_ud_l = tl.zeros((1,), dtype=DTYPE)
    sum_ud_r = tl.zeros((1,), dtype=DTYPE)
    _scalar_off = tl.arange(0, 1)
    if OUTPUT_PIBAR_UD:
        row_max_l = tl.load(pibar_row_max_ptr + sl).to(DTYPE)
        row_max_r = tl.load(pibar_row_max_ptr + sr).to(DTYPE)
        side_nonzero_l = tl.full((1,), value=0, dtype=tl.int32)
        side_nonzero_r = tl.full((1,), value=0, dtype=tl.int32)
        side_abs_bound_l = tl.zeros((1,), dtype=DTYPE)
        side_abs_bound_r = tl.zeros((1,), dtype=DTYPE)

    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        valid_mask = s_offs < S
        mask = valid_mask & parent_active

        Pi_l = tl.load(Pi_star_ptr + pi_l_base + s_offs, mask=mask, other=NEG_LARGE).to(DTYPE)
        Pi_r = tl.load(Pi_star_ptr + pi_r_base + s_offs, mask=mask, other=NEG_LARGE).to(DTYPE)
        Pibar_l = tl.load(Pibar_star_ptr + pibar_l_base + s_offs, mask=mask, other=NEG_LARGE).to(DTYPE)
        Pibar_r = tl.load(Pibar_star_ptr + pibar_r_base + s_offs, mask=mask, other=NEG_LARGE).to(DTYPE)

        c1 = tl.load(sp_child1_ptr + s_offs, mask=mask, other=0)
        c2 = tl.load(sp_child2_ptr + s_offs, mask=mask, other=0)
        c1_valid = (c1 < S) & mask
        c2_valid = (c2 < S) & mask
        Pi_l_s1 = tl.load(Pi_star_ptr + pi_l_base + c1, mask=c1_valid, other=NEG_LARGE).to(DTYPE)
        Pi_l_s2 = tl.load(Pi_star_ptr + pi_l_base + c2, mask=c2_valid, other=NEG_LARGE).to(DTYPE)
        Pi_r_s1 = tl.load(Pi_star_ptr + pi_r_base + c1, mask=c1_valid, other=NEG_LARGE).to(DTYPE)
        Pi_r_s2 = tl.load(Pi_star_ptr + pi_r_base + c2, mask=c2_valid, other=NEG_LARGE).to(DTYPE)

        Pi_parent = tl.load(Pi_star_ptr + parent_pi_base + s_offs, mask=mask, other=NEG_LARGE).to(DTYPE)
        v_k_val = tl.load(v_k_ptr + parent_vk_base + s_offs, mask=mask, other=0.0).to(DTYPE)

        if PARAM_LAYOUT == 1:
            log_pD_s = tl.load(log_pD_arg + s_offs, mask=valid_mask, other=NEG_LARGE).to(DTYPE)
            log_pS_s = tl.load(log_pS_arg + s_offs, mask=valid_mask, other=NEG_LARGE).to(DTYPE)
        elif PARAM_LAYOUT == 3:
            param_base = parent_family * S
            log_pD_s = tl.load(log_pD_arg + param_base + s_offs, mask=valid_mask, other=NEG_LARGE).to(DTYPE)
            log_pS_s = tl.load(log_pS_arg + param_base + s_offs, mask=valid_mask, other=NEG_LARGE).to(DTYPE)
        else:
            log_pD_s = log_pD
            log_pS_s = log_pS

        d0 = log_pD_s + Pi_l + Pi_r + corr0
        d1 = Pi_l + Pibar_r + corr1
        d2 = Pi_r + Pibar_l + corr2
        d3 = log_pS_s + Pi_l_s1 + Pi_r_s2 + corr0
        d4 = log_pS_s + Pi_r_s1 + Pi_l_s2 + corr0

        parent_valid = Pi_parent != NEG_LARGE
        w0 = tl.where(parent_valid, tl.exp2(wlsp + d0 - Pi_parent), tl.zeros_like(d0))
        w1 = tl.where(parent_valid, tl.exp2(wlsp + d1 - Pi_parent), tl.zeros_like(d1))
        w2 = tl.where(parent_valid, tl.exp2(wlsp + d2 - Pi_parent), tl.zeros_like(d2))
        w3 = tl.where(parent_valid, tl.exp2(wlsp + d3 - Pi_parent), tl.zeros_like(d3))
        w4 = tl.where(parent_valid, tl.exp2(wlsp + d4 - Pi_parent), tl.zeros_like(d4))

        vd0 = v_k_val * w0
        vd1 = v_k_val * w1
        vd2 = v_k_val * w2
        vd3 = v_k_val * w3
        vd4 = v_k_val * w4

        pi_l_out = accumulated_rhs_ptr + pi_l_base + s_offs
        pi_r_out = accumulated_rhs_ptr + pi_r_base + s_offs
        if USE_ATOMICS:
            tl.atomic_add(pi_l_out, vd0 + vd1, sem="relaxed", mask=mask)
            tl.atomic_add(pi_r_out, vd0 + vd2, sem="relaxed", mask=mask)
        else:
            pi_l_cur = tl.load(pi_l_out, mask=mask, other=0.0).to(DTYPE)
            pi_r_cur = tl.load(pi_r_out, mask=mask, other=0.0).to(DTYPE)
            tl.store(pi_l_out, pi_l_cur + vd0 + vd1, mask=mask)
            tl.store(pi_r_out, pi_r_cur + vd0 + vd2, mask=mask)
        if OUTPUT_PIBAR_UD:
            if MT_LAYOUT == 1:
                mt_l = tl.load(mt_ptr + family_l * S + s_offs, mask=valid_mask, other=0.0).to(DTYPE)
                mt_r = tl.load(mt_ptr + family_r * S + s_offs, mask=valid_mask, other=0.0).to(DTYPE)
            else:
                mt = tl.load(mt_ptr + s_offs, mask=valid_mask, other=0.0).to(DTYPE)
                mt_l = mt
                mt_r = mt
            finite_l = (Pibar_l != NEG_LARGE) & mask
            finite_r = (Pibar_r != NEG_LARGE) & mask
            inv_denom_l = tl.where(
                finite_l,
                tl.exp2(row_max_l + mt_l - Pibar_l + inv_denom_corr_l),
                tl.zeros([BLOCK_S], dtype=DTYPE),
            )
            inv_denom_r = tl.where(
                finite_r,
                tl.exp2(row_max_r + mt_r - Pibar_r + inv_denom_corr_r),
                tl.zeros([BLOCK_S], dtype=DTYPE),
            )
            ud_l = vd2 * inv_denom_l
            ud_r = vd1 * inv_denom_r
            tl.store(pibar_ud_ptr + i * S + s_offs, ud_l, mask=valid_mask)
            tl.store(pibar_ud_ptr + (tl.num_programs(0) + i) * S + s_offs, ud_r, mask=valid_mask)
            sum_ud_l += tl.sum(tl.where(mask, ud_l, 0.0), axis=0)
            sum_ud_r += tl.sum(tl.where(mask, ud_r, 0.0), axis=0)
            if OUTPUT_SIDE_ACTIVE:
                if SIDE_ACTIVE_THRESHOLD_ENABLED:
                    side_abs_bound_l += tl.sum(tl.where(mask, tl.abs(ud_l), 0.0), axis=0)
                    side_abs_bound_r += tl.sum(tl.where(mask, tl.abs(ud_r), 0.0), axis=0)
                else:
                    side_nonzero_l += tl.where(tl.max(tl.abs(ud_l), axis=0) != 0.0, 1, 0)
                    side_nonzero_r += tl.where(tl.max(tl.abs(ud_r), axis=0) != 0.0, 1, 0)
        else:
            tl.store(grad_Pibar_l_ptr + out_base + s_offs, vd2, mask=valid_mask)
            tl.store(grad_Pibar_r_ptr + out_base + s_offs, vd1, mask=valid_mask)

        if ACCUM_PARAM_REDUCTIONS and PARAM_GRAD_LAYOUT == 1:
            tl.atomic_add(grad_log_pD_ptr + s_offs, vd0, sem="relaxed", mask=mask)
            tl.atomic_add(grad_log_pS_ptr + s_offs, vd3 + vd4, sem="relaxed", mask=mask)
        elif ACCUM_PARAM_REDUCTIONS and PARAM_GRAD_LAYOUT == 3:
            grad_param_base = parent_family * S
            tl.atomic_add(grad_log_pD_ptr + grad_param_base + s_offs, vd0, sem="relaxed", mask=mask)
            tl.atomic_add(grad_log_pS_ptr + grad_param_base + s_offs, vd3 + vd4, sem="relaxed", mask=mask)
        else:
            sum_pD += tl.sum(vd0, axis=0)
            sum_pS += tl.sum(vd3 + vd4, axis=0)
        if ACCUM_MT_REDUCTION:
            mt_contrib = vd1 + vd2
            if GRAD_MT_LAYOUT == 1:
                tl.atomic_add(
                    grad_mt_ptr + family_l * S + s_offs,
                    vd2,
                    sem="relaxed",
                    mask=mask,
                )
                tl.atomic_add(
                    grad_mt_ptr + family_r * S + s_offs,
                    vd1,
                    sem="relaxed",
                    mask=mask,
                )
            elif GRAD_MT_SCALAR:
                tl.atomic_add(
                    grad_mt_ptr + _scalar_off,
                    tl.sum(tl.where(mask, mt_contrib, 0.0), axis=0),
                    sem="relaxed",
                )
            elif GRAD_MT_TWO_STAGE:
                mt_tile = i // GRAD_MT_TILE_SPLITS
                tl.atomic_add(
                    grad_mt_partial_ptr + mt_tile * S + s_offs,
                    mt_contrib,
                    sem="relaxed",
                    mask=mask,
                )
            else:
                tl.atomic_add(
                    grad_mt_ptr + s_offs,
                    mt_contrib,
                    sem="relaxed",
                    mask=mask,
                )

        if MERGE_S_TERM:
            pi_l_c1_out = accumulated_rhs_ptr + pi_l_base + c1
            pi_r_c1_out = accumulated_rhs_ptr + pi_r_base + c1
            pi_r_c2_out = accumulated_rhs_ptr + pi_r_base + c2
            pi_l_c2_out = accumulated_rhs_ptr + pi_l_base + c2
            if USE_ATOMICS:
                tl.atomic_add(pi_l_c1_out, vd3, sem="relaxed", mask=c1_valid)
                tl.atomic_add(pi_r_c1_out, vd4, sem="relaxed", mask=c1_valid)
                tl.atomic_add(pi_r_c2_out, vd3, sem="relaxed", mask=c2_valid)
                tl.atomic_add(pi_l_c2_out, vd4, sem="relaxed", mask=c2_valid)
            else:
                pi_l_c1_cur = tl.load(pi_l_c1_out, mask=c1_valid, other=0.0)
                pi_r_c1_cur = tl.load(pi_r_c1_out, mask=c1_valid, other=0.0)
                pi_r_c2_cur = tl.load(pi_r_c2_out, mask=c2_valid, other=0.0)
                pi_l_c2_cur = tl.load(pi_l_c2_out, mask=c2_valid, other=0.0)
                tl.store(pi_l_c1_out, pi_l_c1_cur + vd3, mask=c1_valid)
                tl.store(pi_r_c1_out, pi_r_c1_cur + vd4, mask=c1_valid)
                tl.store(pi_r_c2_out, pi_r_c2_cur + vd3, mask=c2_valid)
                tl.store(pi_l_c2_out, pi_l_c2_cur + vd4, mask=c2_valid)

    if ACCUM_PARAM_REDUCTIONS:
        if PARAM_GRAD_LAYOUT == 0:
            tl.atomic_add(grad_log_pD_ptr + _scalar_off, sum_pD, sem="relaxed")
            tl.atomic_add(grad_log_pS_ptr + _scalar_off, sum_pS, sem="relaxed")
        elif PARAM_GRAD_LAYOUT == 2:
            tl.atomic_add(
                grad_log_pD_ptr + parent_family + _scalar_off,
                sum_pD,
                sem="relaxed",
            )
            tl.atomic_add(
                grad_log_pS_ptr + parent_family + _scalar_off,
                sum_pS,
                sem="relaxed",
            )
    else:
        tl.store(param_pD_ptr + i + _scalar_off, sum_pD)
        tl.store(param_pS_ptr + i + _scalar_off, sum_pS)
    if OUTPUT_PIBAR_UD:
        tl.store(pibar_A_ptr + i + _scalar_off, sum_ud_l)
        tl.store(pibar_A_ptr + tl.num_programs(0) + i + _scalar_off, sum_ud_r)
        if OUTPUT_SIDE_ACTIVE:
            if SIDE_ACTIVE_THRESHOLD_ENABLED:
                threshold = tl.load(side_active_threshold_ptr).to(DTYPE)
                bound_l = side_abs_bound_l
                bound_r = side_abs_bound_r
                tl.store(pibar_side_active_ptr + i + _scalar_off, bound_l > threshold)
                tl.store(
                    pibar_side_active_ptr + tl.num_programs(0) + i + _scalar_off,
                    bound_r > threshold,
                )
            else:
                tl.store(pibar_side_active_ptr + i + _scalar_off, side_nonzero_l != 0)
                tl.store(
                    pibar_side_active_ptr + tl.num_programs(0) + i + _scalar_off,
                    side_nonzero_r != 0,
                )

    if not MERGE_S_TERM:
        for s_start in range(0, S, BLOCK_S):
            s_offs = s_start + tl.arange(0, BLOCK_S)
            valid_mask = s_offs < S
            mask = valid_mask & parent_active

            c1 = tl.load(sp_child1_ptr + s_offs, mask=mask, other=0)
            c2 = tl.load(sp_child2_ptr + s_offs, mask=mask, other=0)
            c1_valid = (c1 < S) & mask
            c2_valid = (c2 < S) & mask

            Pi_l_s1 = tl.load(Pi_star_ptr + pi_l_base + c1, mask=c1_valid, other=NEG_LARGE).to(DTYPE)
            Pi_l_s2 = tl.load(Pi_star_ptr + pi_l_base + c2, mask=c2_valid, other=NEG_LARGE).to(DTYPE)
            Pi_r_s1 = tl.load(Pi_star_ptr + pi_r_base + c1, mask=c1_valid, other=NEG_LARGE).to(DTYPE)
            Pi_r_s2 = tl.load(Pi_star_ptr + pi_r_base + c2, mask=c2_valid, other=NEG_LARGE).to(DTYPE)

            Pi_parent = tl.load(Pi_star_ptr + parent_pi_base + s_offs, mask=mask, other=NEG_LARGE).to(DTYPE)
            v_k_val = tl.load(v_k_ptr + parent_vk_base + s_offs, mask=mask, other=0.0).to(DTYPE)

            if PARAM_LAYOUT == 1:
                log_pS_s = tl.load(log_pS_arg + s_offs, mask=valid_mask, other=NEG_LARGE).to(DTYPE)
            elif PARAM_LAYOUT == 3:
                log_pS_s = tl.load(log_pS_arg + parent_family * S + s_offs, mask=valid_mask, other=NEG_LARGE).to(DTYPE)
            else:
                log_pS_s = log_pS

            d3 = log_pS_s + Pi_l_s1 + Pi_r_s2 + corr0
            d4 = log_pS_s + Pi_r_s1 + Pi_l_s2 + corr0

            parent_valid = Pi_parent != NEG_LARGE
            w3 = tl.where(parent_valid, tl.exp2(wlsp + d3 - Pi_parent), tl.zeros_like(d3))
            w4 = tl.where(parent_valid, tl.exp2(wlsp + d4 - Pi_parent), tl.zeros_like(d4))
            vd3 = v_k_val * w3
            vd4 = v_k_val * w4

            pi_l_c1_out = accumulated_rhs_ptr + pi_l_base + c1
            pi_r_c1_out = accumulated_rhs_ptr + pi_r_base + c1
            pi_r_c2_out = accumulated_rhs_ptr + pi_r_base + c2
            pi_l_c2_out = accumulated_rhs_ptr + pi_l_base + c2
            if USE_ATOMICS:
                tl.atomic_add(pi_l_c1_out, vd3, sem="relaxed", mask=c1_valid)
                tl.atomic_add(pi_r_c1_out, vd4, sem="relaxed", mask=c1_valid)
                tl.atomic_add(pi_r_c2_out, vd3, sem="relaxed", mask=c2_valid)
                tl.atomic_add(pi_l_c2_out, vd4, sem="relaxed", mask=c2_valid)
            else:
                pi_l_c1_cur = tl.load(pi_l_c1_out, mask=c1_valid, other=0.0).to(DTYPE)
                pi_r_c1_cur = tl.load(pi_r_c1_out, mask=c1_valid, other=0.0).to(DTYPE)
                pi_r_c2_cur = tl.load(pi_r_c2_out, mask=c2_valid, other=0.0).to(DTYPE)
                pi_l_c2_cur = tl.load(pi_l_c2_out, mask=c2_valid, other=0.0).to(DTYPE)
                tl.store(pi_l_c1_out, pi_l_c1_cur + vd3, mask=c1_valid)
                tl.store(pi_r_c1_out, pi_r_c1_cur + vd4, mask=c1_valid)
                tl.store(pi_r_c2_out, pi_r_c2_cur + vd3, mask=c2_valid)
                tl.store(pi_l_c2_out, pi_l_c2_cur + vd4, mask=c2_valid)


@triton.jit
def _dts_grad_mt_two_stage_reduce_kernel(
    partial_ptr,   # [n_tiles, S]
    grad_mt_ptr,   # [S]
    n_tiles: tl.constexpr,
    S: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
    BLOCK_S: tl.constexpr,
    DTYPE: tl.constexpr,
):
    """Reduce split-tile DTS grad_mt partials by species."""
    s_block = tl.program_id(0)
    s_offs = s_block * BLOCK_S + tl.arange(0, BLOCK_S)
    valid_s = s_offs < S
    acc = tl.zeros([BLOCK_S], dtype=DTYPE)

    tile_start = 0
    while tile_start < n_tiles:
        tile_offs = tile_start + tl.arange(0, BLOCK_TILES)
        mask = (tile_offs[:, None] < n_tiles) & valid_s[None, :]
        vals = tl.load(
            partial_ptr + tile_offs[:, None] * S + s_offs[None, :],
            mask=mask,
            other=0.0,
        )
        acc += tl.sum(vals, axis=0)
        tile_start += BLOCK_TILES

    current = tl.load(grad_mt_ptr + s_offs, mask=valid_s, other=0.0)
    tl.store(grad_mt_ptr + s_offs, current + acc, mask=valid_s)


@triton.jit
def _pibar_ud_side_active_kernel(
    pibar_ud_ptr,        # [n_rows, S]
    side_active_ptr,     # [n_rows] bool
    side_active_threshold_ptr,
    S: tl.constexpr,
    BLOCK_S: tl.constexpr,
    SIDE_ACTIVE_THRESHOLD_ENABLED: tl.constexpr,
    DTYPE: tl.constexpr,
):
    """Mark split-side rows whose staged u_d should run Pibar tree work."""
    row = tl.program_id(0)
    row_base = row * S
    row_absmax = tl.full([1], value=0.0, dtype=DTYPE)
    row_abssum = tl.full([1], value=0.0, dtype=DTYPE)

    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        mask = s_offs < S
        vals = tl.load(pibar_ud_ptr + row_base + s_offs, mask=mask, other=0.0)
        abs_vals = tl.abs(vals)
        row_absmax = tl.maximum(row_absmax, tl.max(abs_vals, axis=0))
        row_abssum += tl.sum(tl.where(mask, abs_vals, 0.0), axis=0)

    lane = tl.arange(0, 1)
    if SIDE_ACTIVE_THRESHOLD_ENABLED:
        threshold = tl.load(side_active_threshold_ptr).to(DTYPE)
        tl.store(side_active_ptr + row + lane, row_abssum > threshold)
    else:
        tl.store(side_active_ptr + row + lane, row_absmax != 0.0)


@triton.jit
def _uniform_cross_pibar_vjp_tree_from_ud_compact_kernel(
    Pi_star_ptr,          # [C, S]
    receiver_log_probs_ptr, # [S]
    pibar_ud_ptr,         # [2 * n_ws, S], initial subtree values u_d
    pibar_A_ptr,          # [2 * n_ws], sum_s u_d[s] per split side
    side_active_ptr,      # optional [2 * n_ws] bool exact-zero side skip mask
    sl_ptr,               # [n_ws]
    sr_ptr,               # [n_ws]
    reduce_idx_ptr,       # [n_ws], used with active_mask_ptr when enabled
    active_mask_ptr,      # optional [W] bool parent row activity mask
    pibar_row_max_ptr,    # [C], Pi-row max from forward uniform Pibar
    compact_level_ptr,    # [N_LEVELS + 1]
    compact_level_parent_ptr, # [total internal nodes across levels]
    compact_level_child1_ptr, # [total internal nodes across levels]
    compact_level_child2_ptr, # [total internal nodes across levels]
    accumulated_rhs_ptr,  # [C, S], updated atomically
    grad_receiver_log_probs_ptr, # optional [S], updated atomically
    n_ws: tl.constexpr,
    S: tl.constexpr,
    stride_C: tl.constexpr,
    BLOCK_S: tl.constexpr,
    N_LEVELS: tl.constexpr,
    USE_ACTIVE_MASK: tl.constexpr,
    USE_SIDE_ACTIVE: tl.constexpr,
    ACCUM_RECEIVER_GRAD: tl.constexpr,
    USE_RECEIVER_WEIGHTS: tl.constexpr,
    DTYPE: tl.constexpr,
):
    """Uniform Pibar from-u_d tree correction using compact per-level nodes."""
    NEG_LARGE: tl.constexpr = -float("inf")

    row = tl.program_id(0)
    split_i = tl.where(row < n_ws, row, row - n_ws)
    is_right = row >= n_ws
    if USE_SIDE_ACTIVE:
        side_active = tl.load(side_active_ptr + row)
        if side_active == 0:
            return

    child_l = tl.load(sl_ptr + split_i).to(tl.int64)
    child_r = tl.load(sr_ptr + split_i).to(tl.int64)
    child = tl.where(is_right, child_r, child_l)
    if USE_ACTIVE_MASK:
        parent_w = tl.load(reduce_idx_ptr + split_i).to(tl.int64)
        row_active = tl.load(active_mask_ptr + parent_w)
        if row_active == 0:
            return
    else:
        row_active = True

    pi_base = child * stride_C
    row_base = row * S
    row_max = tl.load(pibar_row_max_ptr + child).to(DTYPE)
    row_max_safe = tl.where(row_max != NEG_LARGE, row_max, tl.zeros_like(row_max))
    A = tl.load(pibar_A_ptr + row).to(DTYPE)

    tl.debug_barrier()
    for level in range(0, N_LEVELS):
        level_start = tl.load(compact_level_ptr + level)
        level_end = tl.load(compact_level_ptr + level + 1)
        p_start = level_start
        while p_start < level_end:
            node_offs = p_start + tl.arange(0, BLOCK_S)
            node_mask = node_offs < level_end
            parent = tl.load(compact_level_parent_ptr + node_offs, mask=node_mask, other=-1)
            c1 = tl.load(compact_level_child1_ptr + node_offs, mask=node_mask, other=S)
            c2 = tl.load(compact_level_child2_ptr + node_offs, mask=node_mask, other=S)
            parent_valid = node_mask & (parent >= 0) & (parent < S) & row_active
            c1_valid = node_mask & (c1 >= 0) & (c1 < S) & row_active
            c2_valid = node_mask & (c2 >= 0) & (c2 < S) & row_active

            parent_val = tl.load(pibar_ud_ptr + row_base + parent, mask=parent_valid, other=0.0)
            c1_val = tl.load(pibar_ud_ptr + row_base + c1, mask=c1_valid, other=0.0)
            c2_val = tl.load(pibar_ud_ptr + row_base + c2, mask=c2_valid, other=0.0)
            tl.store(pibar_ud_ptr + row_base + parent, parent_val + c1_val + c2_val, mask=parent_valid)
            p_start += BLOCK_S
        tl.debug_barrier()

    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        valid_mask = s_offs < S
        mask = valid_mask & row_active
        pi_val = tl.load(Pi_star_ptr + pi_base + s_offs, mask=mask, other=NEG_LARGE)
        if USE_RECEIVER_WEIGHTS:
            receiver_logp = tl.load(receiver_log_probs_ptr + s_offs, mask=valid_mask, other=NEG_LARGE)
            p_prime = tl.exp2(receiver_logp + pi_val - row_max_safe)
        else:
            p_prime = tl.exp2(pi_val - row_max_safe)
        subtree_sum = tl.load(pibar_ud_ptr + row_base + s_offs, mask=mask, other=0.0)
        contrib = p_prime * (A - subtree_sum)
        tl.atomic_add(accumulated_rhs_ptr + pi_base + s_offs, contrib, sem="relaxed", mask=mask)
        if ACCUM_RECEIVER_GRAD:
            tl.atomic_add(
                grad_receiver_log_probs_ptr + s_offs,
                contrib,
                sem="relaxed",
                mask=mask,
            )
