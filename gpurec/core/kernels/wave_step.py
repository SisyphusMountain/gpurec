import torch
import triton
import triton.language as tl


def _tl_float_dtype(dtype):
    return tl.float64 if dtype == torch.float64 else tl.float32


def _prepare_wave_launch(S: int, const_tensor) -> tuple[int, int]:
    const_row_stride = 0 if int(const_tensor.shape[0]) == 1 else int(const_tensor.stride(0))
    return int(min(256, triton.next_power_of_2(S))), const_row_stride


@triton.jit
def _row_logsumexp(
    Pi_ptr,
    receiver_log_probs_ptr,
    base,
    S: tl.constexpr,
    BLOCK_S: tl.constexpr,
    USE_RECEIVER_WEIGHTS: tl.constexpr,
    DTYPE: tl.constexpr,
):
    NEG_INF: tl.constexpr = -float("inf")
    row_max = tl.full([1], value=NEG_INF, dtype=DTYPE)
    row_sum = tl.full([1], value=0.0, dtype=DTYPE)
    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        mask = s_offs < S
        pi_val = tl.load(Pi_ptr + base + s_offs, mask=mask, other=NEG_INF)
        if USE_RECEIVER_WEIGHTS:
            receiver_logp = tl.load(receiver_log_probs_ptr + s_offs, mask=mask, other=NEG_INF)
            weighted_pi = receiver_logp + pi_val
        else:
            weighted_pi = pi_val
        new_max = tl.maximum(row_max, tl.max(weighted_pi, axis=0))
        new_max_safe = tl.where(new_max != NEG_INF, new_max, tl.zeros_like(new_max))
        previous = tl.where(
            row_max != NEG_INF,
            row_sum * tl.exp2(row_max - new_max_safe),
            tl.zeros_like(row_sum),
        )
        current = tl.sum(tl.exp2(weighted_pi - new_max_safe), axis=0)
        row_sum = previous + current
        row_max = new_max
    return row_max, row_sum


@triton.jit
def _pibar_tile(
    Pi_ptr,
    receiver_log_probs_ptr,
    base,
    s_offs,
    mask,
    row_max,
    row_sum,
    max_transfer,
    sp_parent_ptr,
    S: tl.constexpr,
    BLOCK_S: tl.constexpr,
    MAX_ANCESTOR_DEPTH: tl.constexpr,
    USE_RECEIVER_WEIGHTS: tl.constexpr,
    DTYPE: tl.constexpr,
):
    NEG_INF: tl.constexpr = -float("inf")
    ancestor_sum = tl.zeros([BLOCK_S], dtype=DTYPE)
    row_max_safe = tl.where(row_max != NEG_INF, row_max, tl.zeros_like(row_max))
    cur = s_offs.to(tl.int64)
    for _ in range(0, MAX_ANCESTOR_DEPTH):
        cur_valid = mask & (cur >= 0) & (cur < S)
        pi_anc = tl.load(Pi_ptr + base + cur, mask=cur_valid, other=NEG_INF)
        if USE_RECEIVER_WEIGHTS:
            receiver_logp_anc = tl.load(receiver_log_probs_ptr + cur, mask=cur_valid, other=NEG_INF)
            ancestor_sum += tl.where(
                cur_valid,
                tl.exp2(receiver_logp_anc + pi_anc - row_max_safe),
                tl.zeros([BLOCK_S], dtype=DTYPE),
            )
        else:
            ancestor_sum += tl.where(
                cur_valid,
                tl.exp2(pi_anc - row_max_safe),
                tl.zeros([BLOCK_S], dtype=DTYPE),
            )
        cur = tl.load(sp_parent_ptr + cur, mask=cur_valid, other=-1).to(tl.int64)
    denom = row_sum - ancestor_sum
    return tl.where(denom > 0.0, tl.log2(denom) + row_max + max_transfer, NEG_INF)


@triton.jit
def _wave_step_kernel(
    Pi_ptr,
    ws,
    pi_ws,
    max_transfer_ptr,
    DL_const_ptr, Ebar_ptr, E_ptr, SL1_const_ptr, SL2_const_ptr,
    receiver_log_probs_ptr,
    sp_child1_ptr, sp_child2_ptr,
    sp_parent_ptr,
    leaf_species_ptr,
    leaf_logp_ptr,
    family_idx_ptr,
    DTS_reduced_ptr,
    has_splits: tl.constexpr,
    Pi_new_ptr,
    Pibar_out_ptr,
    pibar_row_max_ptr,
    S: tl.constexpr,
    stride: tl.constexpr,
    CONST_ROW_STRIDE: tl.constexpr,
    BLOCK_S: tl.constexpr,
    MAX_ANCESTOR_DEPTH: tl.constexpr,
    USE_LEAF_INDEX: tl.constexpr,
    STORE_FINAL_PIBAR: tl.constexpr,
    USE_RECEIVER_WEIGHTS: tl.constexpr,
    DTYPE: tl.constexpr,
):
    NEG_LARGE = -float("inf")

    w = tl.program_id(0)
    pi_base = (pi_ws + w) * stride
    global_base = (ws + w) * stride
    out_base = w * stride
    family_const = tl.load(family_idx_ptr + ws + w)
    const_base = family_const * CONST_ROW_STRIDE

    row_max, row_sum = _row_logsumexp(
        Pi_ptr, receiver_log_probs_ptr, pi_base, S, BLOCK_S, USE_RECEIVER_WEIGHTS, DTYPE
    )

    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        mask = s_offs < S

        pi_w = tl.load(Pi_ptr + pi_base + s_offs, mask=mask, other=NEG_LARGE)

        const_offsets = const_base + s_offs
        max_transfer = tl.load(max_transfer_ptr + const_offsets, mask=mask, other=0.0)
        pibar_w = _pibar_tile(
            Pi_ptr, receiver_log_probs_ptr, pi_base, s_offs, mask, row_max, row_sum,
            max_transfer, sp_parent_ptr, S, BLOCK_S, MAX_ANCESTOR_DEPTH, USE_RECEIVER_WEIGHTS, DTYPE,
        )

        dl_const = tl.load(DL_const_ptr + const_offsets, mask=mask, other=NEG_LARGE)
        ebar = tl.load(Ebar_ptr + const_offsets, mask=mask, other=NEG_LARGE)
        e_val = tl.load(E_ptr + const_offsets, mask=mask, other=NEG_LARGE)
        sl1_const = tl.load(SL1_const_ptr + const_offsets, mask=mask, other=NEG_LARGE)
        sl2_const = tl.load(SL2_const_ptr + const_offsets, mask=mask, other=NEG_LARGE)

        c1 = tl.load(sp_child1_ptr + s_offs, mask=mask, other=0)
        c2 = tl.load(sp_child2_ptr + s_offs, mask=mask, other=0)
        c1_valid = c1 < S
        c2_valid = c2 < S
        pi_s1 = tl.load(Pi_ptr + pi_base + c1, mask=mask & c1_valid, other=NEG_LARGE)
        pi_s2 = tl.load(Pi_ptr + pi_base + c2, mask=mask & c2_valid, other=NEG_LARGE)

        t0 = dl_const + pi_w
        t1 = pi_w + ebar
        t2 = pibar_w + e_val
        t3 = sl1_const + pi_s1
        t4 = sl2_const + pi_s2
        if USE_LEAF_INDEX:
            leaf_species = tl.load(leaf_species_ptr + ws + w)
            leaf_hit = mask & (leaf_species == s_offs)
            leaf_logp = tl.load(leaf_logp_ptr + family_const * S + s_offs, mask=mask, other=NEG_LARGE)
            t5 = tl.where(leaf_hit, leaf_logp, NEG_LARGE)
        else:
            t5 = tl.full([BLOCK_S], value=NEG_LARGE, dtype=DTYPE)

        m = tl.maximum(t0, t1)
        m = tl.maximum(m, t2)
        m = tl.maximum(m, t3)
        m = tl.maximum(m, t4)
        m = tl.maximum(m, t5)
        if has_splits:
            dts_r = tl.load(DTS_reduced_ptr + out_base + s_offs, mask=mask, other=NEG_LARGE)
            m = tl.maximum(m, dts_r)

        m_safe = tl.where(m != NEG_LARGE, m, tl.zeros_like(m))
        s = tl.exp2(t0 - m_safe) + tl.exp2(t1 - m_safe) + tl.exp2(t2 - m_safe)
        s += tl.exp2(t3 - m_safe) + tl.exp2(t4 - m_safe) + tl.exp2(t5 - m_safe)
        if has_splits:
            s += tl.exp2(dts_r - m_safe)

        result = tl.log2(s) + m
        tl.store(Pi_new_ptr + out_base + s_offs, result, mask=mask)

    if STORE_FINAL_PIBAR:
        final_row_max, final_row_sum = _row_logsumexp(
            Pi_new_ptr, receiver_log_probs_ptr, out_base, S, BLOCK_S, USE_RECEIVER_WEIGHTS, DTYPE
        )
        tl.store(pibar_row_max_ptr + ws + w, tl.max(final_row_max, axis=0))

        for s_start in range(0, S, BLOCK_S):
            s_offs = s_start + tl.arange(0, BLOCK_S)
            mask = s_offs < S
            const_offsets = const_base + s_offs
            max_transfer = tl.load(max_transfer_ptr + const_offsets, mask=mask, other=0.0)
            pibar_w = _pibar_tile(
                Pi_new_ptr, receiver_log_probs_ptr, out_base, s_offs, mask, final_row_max, final_row_sum,
                max_transfer, sp_parent_ptr, S, BLOCK_S, MAX_ANCESTOR_DEPTH, USE_RECEIVER_WEIGHTS, DTYPE,
            )
            tl.store(Pibar_out_ptr + global_base + s_offs, pibar_w, mask=mask)


@triton.jit
def _leaf_initial_wave_step_kernel(
    Pi_new_ptr,
    ws,
    max_transfer_ptr,
    DL_const_ptr,
    Ebar_ptr,
    E_ptr,
    SL1_const_ptr,
    SL2_const_ptr,
    receiver_log_probs_ptr,
    sp_child1_ptr,
    sp_child2_ptr,
    sp_subtree_start_ptr,
    sp_subtree_end_ptr,
    leaf_species_ptr,
    leaf_logp_ptr,
    family_idx_ptr,
    S: tl.constexpr,
    stride: tl.constexpr,
    CONST_ROW_STRIDE: tl.constexpr,
    BLOCK_S: tl.constexpr,
    USE_RECEIVER_WEIGHTS: tl.constexpr,
    DTYPE: tl.constexpr,
):
    NEG_LARGE = -float("inf")

    w = tl.program_id(0)
    s_start = tl.program_id(1) * BLOCK_S
    s_offs = s_start + tl.arange(0, BLOCK_S)
    mask = s_offs < S
    out_base = w * stride

    family = tl.load(family_idx_ptr + ws + w)
    const_base = family * CONST_ROW_STRIDE

    leaf_species = tl.load(leaf_species_ptr + ws + w)
    leaf_start = tl.load(sp_subtree_start_ptr + leaf_species)
    leaf_end = tl.load(sp_subtree_end_ptr + leaf_species)
    species_start = tl.load(sp_subtree_start_ptr + s_offs, mask=mask, other=-1)
    descendant = (species_start >= leaf_start) & (species_start < leaf_end)
    leaf_hit = mask & (s_offs == leaf_species)

    const_offsets = const_base + s_offs
    max_transfer = tl.load(max_transfer_ptr + const_offsets, mask=mask, other=0.0)
    if USE_RECEIVER_WEIGHTS:
        leaf_receiver_logp = tl.load(receiver_log_probs_ptr + leaf_species).to(DTYPE)
    else:
        leaf_receiver_logp = tl.zeros((), dtype=DTYPE)
    dl_const = tl.load(DL_const_ptr + const_offsets, mask=mask, other=NEG_LARGE)
    ebar = tl.load(Ebar_ptr + const_offsets, mask=mask, other=NEG_LARGE)
    e_val = tl.load(E_ptr + const_offsets, mask=mask, other=NEG_LARGE)
    sl1_const = tl.load(SL1_const_ptr + const_offsets, mask=mask, other=NEG_LARGE)
    sl2_const = tl.load(SL2_const_ptr + const_offsets, mask=mask, other=NEG_LARGE)

    pi_w = tl.where(leaf_hit, tl.zeros([BLOCK_S], dtype=DTYPE), NEG_LARGE)
    pibar_w = tl.where(~descendant, max_transfer + leaf_receiver_logp, NEG_LARGE)

    c1 = tl.load(sp_child1_ptr + s_offs, mask=mask, other=S)
    c2 = tl.load(sp_child2_ptr + s_offs, mask=mask, other=S)
    pi_s1 = tl.where(mask & (c1 == leaf_species), 0.0, NEG_LARGE)
    pi_s2 = tl.where(mask & (c2 == leaf_species), 0.0, NEG_LARGE)

    t0 = dl_const + pi_w
    t1 = pi_w + ebar
    t2 = pibar_w + e_val
    t3 = sl1_const + pi_s1
    t4 = sl2_const + pi_s2
    leaf_logp = tl.load(leaf_logp_ptr + family * S + s_offs, mask=mask, other=NEG_LARGE)
    t5 = tl.where(leaf_hit, leaf_logp, NEG_LARGE)

    m = tl.maximum(t0, t1)
    m = tl.maximum(m, t2)
    m = tl.maximum(m, t3)
    m = tl.maximum(m, t4)
    m = tl.maximum(m, t5)
    m_safe = tl.where(m != NEG_LARGE, m, tl.zeros_like(m))
    total = (
        tl.exp2(t0 - m_safe)
        + tl.exp2(t1 - m_safe)
        + tl.exp2(t2 - m_safe)
        + tl.exp2(t3 - m_safe)
        + tl.exp2(t4 - m_safe)
        + tl.exp2(t5 - m_safe)
    )
    result = tl.log2(total) + m
    tl.store(Pi_new_ptr + out_base + s_offs, result, mask=mask)


def compute_leaf_initial_wave_step(
    Pi_out,
    ws,
    W,
    S,
    max_transfer_mat,
    DL_const,
    Ebar,
    E,
    SL1_const,
    SL2_const,
    receiver_log_probs,
    sp_child1,
    sp_child2,
    sp_subtree_start,
    sp_subtree_end,
    leaf_species_idx,
    leaf_logp,
    family_idx,
    use_receiver_weights=True,
):
    block_s, const_row_stride = _prepare_wave_launch(S, DL_const)
    grid = (W, triton.cdiv(S, block_s))
    Pi_out_rows = Pi_out.narrow(0, int(ws), int(W))
    _leaf_initial_wave_step_kernel[grid](
        Pi_out_rows,
        ws,
        max_transfer_mat,
        DL_const,
        Ebar,
        E,
        SL1_const,
        SL2_const,
        receiver_log_probs,
        sp_child1,
        sp_child2,
        sp_subtree_start,
        sp_subtree_end,
        leaf_species_idx,
        leaf_logp,
        family_idx,
        S,
        stride=S,
        CONST_ROW_STRIDE=const_row_stride,
        BLOCK_S=block_s,
        USE_RECEIVER_WEIGHTS=bool(use_receiver_weights),
        DTYPE=_tl_float_dtype(Pi_out.dtype),
        num_warps=8,
    )

def compute_wave_step(Pi_in, Pi_out, Pibar, ws, W, S,
                     max_transfer_mat, DL_const, Ebar, E, SL1_const, SL2_const,
                     receiver_log_probs,
                     sp_child1, sp_child2, sp_parent, max_ancestor_depth,
                     DTS_reduced=None,
                     *,
                     leaf_species_idx, leaf_logp,
                     family_idx,
                     pibar_row_max,
                     store_final_pibar=False,
                     has_leaf_term=True,
                     input_ws=None,
                     use_receiver_weights=True):
    has_splits = DTS_reduced is not None
    block_s, const_row_stride = _prepare_wave_launch(S, DL_const)
    use_leaf_index = bool(has_leaf_term)

    grid = (W,)
    Pi_out_rows = Pi_out.narrow(0, int(ws), int(W))

    _wave_step_kernel[grid](
        Pi_in, ws, ws if input_ws is None else int(input_ws),
        max_transfer_mat,
        DL_const, Ebar, E, SL1_const, SL2_const,
        receiver_log_probs,
        sp_child1, sp_child2,
        sp_parent,
        leaf_species_idx,
        leaf_logp,
        family_idx,
        DTS_reduced if has_splits else Pi_in,
        has_splits,
        Pi_out_rows, Pibar, pibar_row_max,
        S,
        stride=S,
        CONST_ROW_STRIDE=const_row_stride,
        BLOCK_S=block_s,
        MAX_ANCESTOR_DEPTH=int(max_ancestor_depth),
        USE_LEAF_INDEX=use_leaf_index,
        STORE_FINAL_PIBAR=bool(store_final_pibar),
        USE_RECEIVER_WEIGHTS=bool(use_receiver_weights),
        DTYPE=_tl_float_dtype(Pi_in.dtype),
        num_warps=8,
    )
