import torch
import triton
import triton.language as tl

__all__ = [
    "compute_dts_forward",
    "compute_leaf_initial_wave_step",
    "compute_wave_step",
    "_load_rate",
    "_prepare_wave_launch",
    "_tl_float_dtype",
]


def _tl_float_dtype(dtype):
    return tl.float64 if dtype == torch.float64 else tl.float32


def _prepare_wave_launch(S: int, const_tensor) -> tuple[int, int]:
    const_row_stride = 0 if int(const_tensor.shape[0]) == 1 else int(const_tensor.stride(0))
    return int(min(256, triton.next_power_of_2(S))), const_row_stride


@triton.jit
def _load_rate(
    param,
    family,
    s_offs,
    mask,
    S: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    BY_SPECIES: tl.constexpr,
    BLOCK_S: tl.constexpr,
    DTYPE: tl.constexpr,
):
    NEG_INF: tl.constexpr = -float("inf")
    if BY_SPECIES:
        return tl.load(param + family * ROW_STRIDE + s_offs, mask=mask, other=NEG_INF)
    return tl.load(param + family * ROW_STRIDE) + tl.zeros([BLOCK_S], dtype=DTYPE)


@triton.jit
def _row_logsumexp(
    Pi_ptr,
    receiver_log_probs_ptr,
    base,
    S: tl.constexpr,
    BLOCK_S: tl.constexpr,
    USE_RECEIVER_WEIGHTS: tl.constexpr,
    TRACK_RAW_MAX: tl.constexpr,
    DTYPE: tl.constexpr,
):
    NEG_INF: tl.constexpr = -float("inf")
    row_max = tl.full([1], value=NEG_INF, dtype=DTYPE)
    row_sum = tl.full([1], value=0.0, dtype=DTYPE)
    raw_row_max = tl.full([1], value=NEG_INF, dtype=DTYPE)
    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        mask = s_offs < S
        pi_val = tl.load(Pi_ptr + base + s_offs, mask=mask, other=NEG_INF)
        if TRACK_RAW_MAX:
            raw_row_max = tl.maximum(raw_row_max, tl.max(pi_val, axis=0))
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
    return row_max, row_sum, raw_row_max


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
def _leaf_initial_wave_step_kernel(
    Pi_new_ptr,
    Pi_new_offset_ptr,
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
    NEG_LARGE: tl.constexpr = -float("inf")

    w = tl.program_id(0)
    global_row = ws + w
    family = tl.load(family_idx_ptr + global_row)
    const_base = family * CONST_ROW_STRIDE
    leaf_species = tl.load(leaf_species_ptr + global_row)
    leaf_start = tl.load(sp_subtree_start_ptr + leaf_species)
    leaf_end = tl.load(sp_subtree_end_ptr + leaf_species)
    if USE_RECEIVER_WEIGHTS:
        leaf_receiver_logp = tl.load(receiver_log_probs_ptr + leaf_species).to(DTYPE)
    else:
        leaf_receiver_logp = tl.zeros((), dtype=DTYPE)
    leaf_obs_logp = tl.load(leaf_logp_ptr + family * S + leaf_species).to(DTYPE)

    row_max = tl.full((), value=NEG_LARGE, dtype=DTYPE)
    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        mask = s_offs < S
        species_start = tl.load(sp_subtree_start_ptr + s_offs, mask=mask, other=-1)
        descendant = (species_start >= leaf_start) & (species_start < leaf_end)
        leaf_hit = mask & (s_offs == leaf_species)
        const_offsets = const_base + s_offs
        max_transfer = tl.load(max_transfer_ptr + const_offsets, mask=mask, other=0.0)
        dl_const = tl.load(DL_const_ptr + const_offsets, mask=mask, other=NEG_LARGE)
        ebar = tl.load(Ebar_ptr + const_offsets, mask=mask, other=NEG_LARGE)
        e_val = tl.load(E_ptr + const_offsets, mask=mask, other=NEG_LARGE)
        sl1_const = tl.load(SL1_const_ptr + const_offsets, mask=mask, other=NEG_LARGE)
        sl2_const = tl.load(SL2_const_ptr + const_offsets, mask=mask, other=NEG_LARGE)

        pi_w = tl.where(leaf_hit, leaf_obs_logp, NEG_LARGE)
        pibar_w = tl.where(
            ~descendant,
            max_transfer + leaf_receiver_logp + leaf_obs_logp,
            NEG_LARGE,
        )
        c1 = tl.load(sp_child1_ptr + s_offs, mask=mask, other=S)
        c2 = tl.load(sp_child2_ptr + s_offs, mask=mask, other=S)
        pi_s1 = tl.where(mask & (c1 == leaf_species), leaf_obs_logp, NEG_LARGE)
        pi_s2 = tl.where(mask & (c2 == leaf_species), leaf_obs_logp, NEG_LARGE)

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
        row_max = tl.maximum(row_max, tl.max(tl.where(mask, result, NEG_LARGE), axis=0))

    row_max_safe = tl.where(row_max != NEG_LARGE, row_max, 0.0)
    tl.store(Pi_new_offset_ptr + global_row, row_max_safe.to(tl.float64))
    out_global_base = global_row * stride
    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        mask = s_offs < S
        species_start = tl.load(sp_subtree_start_ptr + s_offs, mask=mask, other=-1)
        descendant = (species_start >= leaf_start) & (species_start < leaf_end)
        leaf_hit = mask & (s_offs == leaf_species)
        const_offsets = const_base + s_offs
        max_transfer = tl.load(max_transfer_ptr + const_offsets, mask=mask, other=0.0)
        dl_const = tl.load(DL_const_ptr + const_offsets, mask=mask, other=NEG_LARGE)
        ebar = tl.load(Ebar_ptr + const_offsets, mask=mask, other=NEG_LARGE)
        e_val = tl.load(E_ptr + const_offsets, mask=mask, other=NEG_LARGE)
        sl1_const = tl.load(SL1_const_ptr + const_offsets, mask=mask, other=NEG_LARGE)
        sl2_const = tl.load(SL2_const_ptr + const_offsets, mask=mask, other=NEG_LARGE)
        pi_w = tl.where(leaf_hit, leaf_obs_logp, NEG_LARGE)
        pibar_w = tl.where(
            ~descendant,
            max_transfer + leaf_receiver_logp + leaf_obs_logp,
            NEG_LARGE,
        )
        c1 = tl.load(sp_child1_ptr + s_offs, mask=mask, other=S)
        c2 = tl.load(sp_child2_ptr + s_offs, mask=mask, other=S)
        pi_s1 = tl.where(mask & (c1 == leaf_species), leaf_obs_logp, NEG_LARGE)
        pi_s2 = tl.where(mask & (c2 == leaf_species), leaf_obs_logp, NEG_LARGE)
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
        result = tl.log2(total) + m - row_max_safe
        tl.store(Pi_new_ptr + out_global_base + s_offs, result, mask=mask)


@triton.jit
def _wave_step_kernel(
    Pi_ptr,
    Pi_offset_ptr,
    ws,
    pi_ws,
    max_transfer_ptr,
    DL_const_ptr,
    Ebar_ptr,
    E_ptr,
    SL1_const_ptr,
    SL2_const_ptr,
    receiver_log_probs_ptr,
    sp_child1_ptr,
    sp_child2_ptr,
    sp_parent_ptr,
    leaf_species_ptr,
    leaf_logp_ptr,
    family_idx_ptr,
    DTS_reduced_ptr,
    DTS_offset_ptr,
    DTS_center_offset_ptr,
    has_splits: tl.constexpr,
    INPUT_IS_DTS: tl.constexpr,
    Pi_new_ptr,
    Pi_new_offset_ptr,
    Pibar_out_ptr,
    Pibar_offset_ptr,
    pibar_row_max_ptr,
    pi_residual_out_ptr,
    S: tl.constexpr,
    stride: tl.constexpr,
    CONST_ROW_STRIDE: tl.constexpr,
    BLOCK_S: tl.constexpr,
    MAX_ANCESTOR_DEPTH: tl.constexpr,
    USE_LEAF_INDEX: tl.constexpr,
    STORE_FINAL_PIBAR: tl.constexpr,
    COMPUTE_DIFF: tl.constexpr,
    USE_RECEIVER_WEIGHTS: tl.constexpr,
    DTYPE: tl.constexpr,
):
    NEG_LARGE: tl.constexpr = -float("inf")

    w = tl.program_id(0)
    pi_row = pi_ws + w
    global_row = ws + w
    pi_base = pi_row * stride
    global_base = global_row * stride
    dts_base = w * stride
    family_const = tl.load(family_idx_ptr + global_row)
    const_base = family_const * CONST_ROW_STRIDE
    pi_offset = tl.load(Pi_offset_ptr + pi_row)

    row_max, row_sum, raw_row_max = _row_logsumexp(
        Pi_ptr,
        receiver_log_probs_ptr,
        pi_base,
        S,
        BLOCK_S,
        USE_RECEIVER_WEIGHTS,
        INPUT_IS_DTS and USE_RECEIVER_WEIGHTS,
        DTYPE,
    )
    # ``row_max`` is already required for Pibar. Absorb it lazily into the
    # fp64 row frame so the recurrence consumes near-zero residuals without an
    # exact recenter/store pass over the input row.
    if INPUT_IS_DTS:
        if USE_RECEIVER_WEIGHTS:
            shift_source = raw_row_max
        else:
            # Without receiver weights ``row_max`` is already the raw maximum;
            # compile out the second tile reduction entirely.
            shift_source = row_max
        row_shift = tl.max(
            tl.where(
                shift_source != NEG_LARGE,
                shift_source,
                tl.zeros_like(shift_source),
            ),
            axis=0,
        ).to(DTYPE)
    else:
        # Leaf initialization and the first virtually gauged DTS iteration already
        # put ordinary Pi iterates in their local frame. Avoid repeating four
        # vector shifts on every later fixed-point iteration.
        row_shift = tl.zeros((), dtype=DTYPE)
    effective_pi_offset = pi_offset + row_shift.to(tl.float64)

    term_base = effective_pi_offset
    if has_splits:
        dts_offset = tl.load(DTS_offset_ptr + w)
        if INPUT_IS_DTS:
            dts_center_offset = dts_offset + row_shift.to(tl.float64)
        else:
            dts_center_offset = tl.load(DTS_center_offset_ptr + w)
        term_base = tl.maximum(term_base, dts_center_offset)
    else:
        dts_offset = term_base
    if USE_LEAF_INDEX:
        leaf_species = tl.load(leaf_species_ptr + global_row)
        leaf_obs_logp = tl.load(
            leaf_logp_ptr + family_const * S + leaf_species
        ).to(DTYPE)
        # The leaf source represents ``leaf_obs_logp``, not a zero-frame
        # value. Using 0 here forced every negative HOGENOM row back into the
        # absolute frame after the exactly gauged leaf initializer.
        term_base = tl.maximum(term_base, leaf_obs_logp.to(tl.float64))
    pi_corr = (effective_pi_offset - term_base).to(DTYPE)
    # DTS storage remains in its original gauge. The virtual row offset
    # participates only in base selection; one fp64 subtraction folds its
    # shift into the same correction the recurrence already applies.
    dts_corr = (dts_offset - term_base).to(DTYPE)
    leaf_corr = (0.0 - term_base).to(DTYPE)
    if STORE_FINAL_PIBAR:
        pi_has_finite = tl.full((), value=0, dtype=tl.int32)

    if COMPUTE_DIFF:
        row_max_diff = tl.zeros([1], dtype=tl.float32)

    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        mask = s_offs < S
        pi_w = tl.load(Pi_ptr + pi_base + s_offs, mask=mask, other=NEG_LARGE)
        pi_w = pi_w - row_shift
        const_offsets = const_base + s_offs
        max_transfer = tl.load(max_transfer_ptr + const_offsets, mask=mask, other=0.0)
        pibar_w = _pibar_tile(
            Pi_ptr,
            receiver_log_probs_ptr,
            pi_base,
            s_offs,
            mask,
            row_max,
            row_sum,
            max_transfer,
            sp_parent_ptr,
            S,
            BLOCK_S,
            MAX_ANCESTOR_DEPTH,
            USE_RECEIVER_WEIGHTS,
            DTYPE,
        )
        pibar_w = pibar_w - row_shift
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
        pi_s1 = pi_s1 - row_shift
        pi_s2 = pi_s2 - row_shift

        t0 = dl_const + pi_w + pi_corr
        t1 = pi_w + ebar + pi_corr
        t2 = pibar_w + e_val + pi_corr
        t3 = sl1_const + pi_s1 + pi_corr
        t4 = sl2_const + pi_s2 + pi_corr
        if USE_LEAF_INDEX:
            leaf_hit = mask & (leaf_species == s_offs)
            t5 = tl.where(leaf_hit, leaf_obs_logp + leaf_corr, NEG_LARGE)
        else:
            t5 = tl.full([BLOCK_S], value=NEG_LARGE, dtype=DTYPE)

        m = tl.maximum(t0, t1)
        m = tl.maximum(m, t2)
        m = tl.maximum(m, t3)
        m = tl.maximum(m, t4)
        m = tl.maximum(m, t5)
        if has_splits:
            dts_r = tl.load(DTS_reduced_ptr + dts_base + s_offs, mask=mask, other=NEG_LARGE)
            dts_r = dts_r + dts_corr
            m = tl.maximum(m, dts_r)
        m_safe = tl.where(m != NEG_LARGE, m, tl.zeros_like(m))
        total = tl.exp2(t0 - m_safe) + tl.exp2(t1 - m_safe) + tl.exp2(t2 - m_safe)
        total += tl.exp2(t3 - m_safe) + tl.exp2(t4 - m_safe) + tl.exp2(t5 - m_safe)
        if has_splits:
            total += tl.exp2(dts_r - m_safe)
        result = tl.log2(total) + m
        tl.store(Pi_new_ptr + global_base + s_offs, result, mask=mask)
        if STORE_FINAL_PIBAR:
            pi_has_finite = tl.maximum(
                pi_has_finite,
                tl.max(tl.where(mask & (result != NEG_LARGE), 1, 0), axis=0),
            )

        if COMPUTE_DIFF:
            # Compare represented absolute values in fp64 without materializing
            # either large absolute fp32 row.
            finite = mask & (result != NEG_LARGE) & (pi_w != NEG_LARGE)
            diff = tl.where(
                finite,
                tl.abs(
                    result.to(tl.float64)
                    - pi_w.to(tl.float64)
                    + term_base
                    - effective_pi_offset
                ),
                tl.zeros([BLOCK_S], dtype=tl.float64),
            )
            row_max_diff = tl.maximum(row_max_diff, tl.max(diff, axis=0).to(tl.float32))

    # Only the final iterate is published as row-gauged state; earlier iterates
    # are internal gauge-equivalent scratch. Canonicalize the published row in
    # its existing traversal without charging every fixed-point iteration.
    if STORE_FINAL_PIBAR:
        pi_new_offset = tl.where(pi_has_finite != 0, term_base, 0.0)
    else:
        pi_new_offset = term_base
    tl.store(Pi_new_offset_ptr + global_row, pi_new_offset)

    if COMPUTE_DIFF:
        tl.store(pi_residual_out_ptr + global_row, tl.max(row_max_diff, axis=0))

    if has_splits and INPUT_IS_DTS:
        tl.store(DTS_center_offset_ptr + w, dts_center_offset)

    if STORE_FINAL_PIBAR:
        final_row_max, final_row_sum, _ = _row_logsumexp(
            Pi_new_ptr,
            receiver_log_probs_ptr,
            global_base,
            S,
            BLOCK_S,
            USE_RECEIVER_WEIGHTS,
            False,
            DTYPE,
        )
        tl.store(pibar_row_max_ptr + global_row, tl.max(final_row_max, axis=0))
        # ``Pi_new`` is stored in ``term_base``'s frame, so the Pibar values
        # produced from it are already in that same frame. Keep that cheap
        # gauge instead of traversing the species row once to find an exact
        # Pibar maximum and again to store the recentered values.
        pibar_has_finite = tl.full((), value=0, dtype=tl.int32)
        for s_start in range(0, S, BLOCK_S):
            s_offs = s_start + tl.arange(0, BLOCK_S)
            mask = s_offs < S
            const_offsets = const_base + s_offs
            max_transfer = tl.load(max_transfer_ptr + const_offsets, mask=mask, other=0.0)
            pibar_w = _pibar_tile(
                Pi_new_ptr,
                receiver_log_probs_ptr,
                global_base,
                s_offs,
                mask,
                final_row_max,
                final_row_sum,
                max_transfer,
                sp_parent_ptr,
                S,
                BLOCK_S,
                MAX_ANCESTOR_DEPTH,
                USE_RECEIVER_WEIGHTS,
                DTYPE,
            )
            pibar_has_finite = tl.maximum(
                pibar_has_finite,
                tl.max(tl.where(mask & (pibar_w != NEG_LARGE), 1, 0), axis=0),
            )
            tl.store(Pibar_out_ptr + global_base + s_offs, pibar_w, mask=mask)
        # An all-impossible Pibar row must remain the canonical ``(-inf, 0)``
        # pair even when its Pi row used a nonzero heuristic gauge.
        pibar_offset = tl.where(pibar_has_finite != 0, term_base, 0.0)
        tl.store(Pibar_offset_ptr + global_row, pibar_offset)


@triton.jit
def _dts_eq1_kernel(
    Pi,
    Pi_offset,
    Pibar,
    Pibar_offset,
    lefts,
    rights,
    sp_child1,
    sp_child2,
    log_pD,
    log_pS,
    log_split_probs,
    eq1_reduce_idx,
    active_rows,
    out,
    out_offset,
    family_idx,
    family_offset,
    S: tl.constexpr,
    BLOCK_S: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    BY_SPECIES: tl.constexpr,
    USE_ACTIVE: tl.constexpr,
    DTYPE: tl.constexpr,
):
    NEG_INF: tl.constexpr = -float("inf")
    n = tl.program_id(0)
    s_block = tl.program_id(1)
    s_offs = s_block * BLOCK_S + tl.arange(0, BLOCK_S)
    mask = s_offs < S
    parent_w = tl.load(eq1_reduce_idx + n).to(tl.int64)
    if USE_ACTIVE:
        if tl.load(active_rows + parent_w) == 0:
            tl.store(out + parent_w * S + s_offs, tl.full([BLOCK_S], NEG_INF, dtype=DTYPE), mask=mask)
            if s_block == 0:
                tl.store(out_offset + parent_w, 0.0)
            return

    family = tl.load(family_idx + family_offset + parent_w).to(tl.int64)
    left = tl.load(lefts + n).to(tl.int64)
    right = tl.load(rights + n).to(tl.int64)
    base_l = left * S
    base_r = right * S
    pi_l = tl.load(Pi + base_l + s_offs, mask=mask, other=NEG_INF)
    pi_r = tl.load(Pi + base_r + s_offs, mask=mask, other=NEG_INF)
    pibar_l = tl.load(Pibar + base_l + s_offs, mask=mask, other=NEG_INF)
    pibar_r = tl.load(Pibar + base_r + s_offs, mask=mask, other=NEG_INF)
    log_d = _load_rate(log_pD, family, s_offs, mask, S, ROW_STRIDE, BY_SPECIES, BLOCK_S, DTYPE)
    log_s = _load_rate(log_pS, family, s_offs, mask, S, ROW_STRIDE, BY_SPECIES, BLOCK_S, DTYPE)
    c1 = tl.load(sp_child1 + s_offs, mask=mask, other=S)
    c2 = tl.load(sp_child2 + s_offs, mask=mask, other=S)
    c1_valid = c1 < S
    c2_valid = c2 < S
    lsp = tl.load(log_split_probs + n)

    pi_off_l = tl.load(Pi_offset + left)
    pi_off_r = tl.load(Pi_offset + right)
    pibar_off_l = tl.load(Pibar_offset + left)
    pibar_off_r = tl.load(Pibar_offset + right)
    base = tl.maximum(pi_off_l + pi_off_r, pi_off_l + pibar_off_r)
    base = tl.maximum(base, pi_off_r + pibar_off_l)
    corr0 = (pi_off_l + pi_off_r - base).to(DTYPE)
    corr1 = (pi_off_l + pibar_off_r - base).to(DTYPE)
    corr2 = (pi_off_r + pibar_off_l - base).to(DTYPE)

    t0 = lsp + log_d + pi_l + pi_r + corr0
    t1 = lsp + pi_l + pibar_r + corr1
    t2 = lsp + pi_r + pibar_l + corr2
    t3 = (
        lsp
        + log_s
        + tl.load(Pi + base_l + c1, mask=mask & c1_valid, other=NEG_INF)
        + tl.load(Pi + base_r + c2, mask=mask & c2_valid, other=NEG_INF)
        + corr0
    )
    t4 = (
        lsp
        + log_s
        + tl.load(Pi + base_r + c1, mask=mask & c1_valid, other=NEG_INF)
        + tl.load(Pi + base_l + c2, mask=mask & c2_valid, other=NEG_INF)
        + corr0
    )
    m = tl.maximum(tl.maximum(tl.maximum(t0, t1), tl.maximum(t2, t3)), t4)
    m_safe = tl.where(m != NEG_INF, m, tl.zeros_like(m))
    acc = (
        tl.exp2(t0 - m_safe)
        + tl.exp2(t1 - m_safe)
        + tl.exp2(t2 - m_safe)
        + tl.exp2(t3 - m_safe)
        + tl.exp2(t4 - m_safe)
    )
    result = tl.log2(acc) + m
    tl.store(out + parent_w * S + s_offs, result, mask=mask)
    # ``out_offset`` starts at zero. Every species tile for this eq1 row has
    # the same candidate base, so any tile containing a finite lane may publish
    # it safely. If every lane is impossible, no tile writes and the canonical
    # all--inf row keeps offset zero without an extra row pass.
    tile_has_finite = tl.max(tl.where(mask & (result != NEG_INF), 1, 0), axis=0) != 0
    tl.store(out_offset + parent_w, base, mask=tile_has_finite)


@triton.jit
def _dts_ge2_stage1_kernel(
    Pi,
    Pi_offset,
    Pibar,
    Pibar_offset,
    lefts,
    rights,
    sp_child1,
    sp_child2,
    log_pD,
    log_pS,
    log_split_probs,
    ge2_ptr,
    ge2_parent_ids,
    active_rows,
    partial_max,
    partial_sum,
    partial_offset,
    family_idx,
    family_offset,
    split_offset,
    MAX_TILES: tl.constexpr,
    S: tl.constexpr,
    TILE_SPLITS: tl.constexpr,
    BLOCK_S: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    BY_SPECIES: tl.constexpr,
    USE_ACTIVE: tl.constexpr,
    DTYPE: tl.constexpr,
):
    NEG_INF: tl.constexpr = -float("inf")
    group = tl.program_id(0)
    tile_id = tl.program_id(1)
    s_block = tl.program_id(2)
    s_offs = s_block * BLOCK_S + tl.arange(0, BLOCK_S)
    mask = s_offs < S
    parent_w = tl.load(ge2_parent_ids + group).to(tl.int64)
    if USE_ACTIVE:
        if tl.load(active_rows + parent_w) == 0:
            return

    family = tl.load(family_idx + family_offset + parent_w).to(tl.int64)
    start = tl.load(ge2_ptr + group)
    end = tl.load(ge2_ptr + group + 1)
    tile_start = start + tile_id * TILE_SPLITS
    if tile_start >= end:
        return
    tile_end = tl.minimum(tile_start + TILE_SPLITS, end)
    m = tl.full([BLOCK_S], NEG_INF, dtype=DTYPE)
    acc = tl.zeros([BLOCK_S], dtype=DTYPE)
    # Use a tile-local fp64 frame and advance it as split offsets increase.
    # This folds the old group-wide offset prepass into work stage 1 already
    # performs, without rescanning every split once per species block.
    tile_base_offset = tl.full((), value=NEG_INF, dtype=tl.float64)
    split_rel = tile_start
    while split_rel < tile_end:
        split_i = split_offset + split_rel
        left = tl.load(lefts + split_i).to(tl.int64)
        right = tl.load(rights + split_i).to(tl.int64)
        base_l = left * S
        base_r = right * S
        pi_l = tl.load(Pi + base_l + s_offs, mask=mask, other=NEG_INF)
        pi_r = tl.load(Pi + base_r + s_offs, mask=mask, other=NEG_INF)
        pibar_l = tl.load(Pibar + base_l + s_offs, mask=mask, other=NEG_INF)
        pibar_r = tl.load(Pibar + base_r + s_offs, mask=mask, other=NEG_INF)
        log_d = _load_rate(log_pD, family, s_offs, mask, S, ROW_STRIDE, BY_SPECIES, BLOCK_S, DTYPE)
        log_s = _load_rate(log_pS, family, s_offs, mask, S, ROW_STRIDE, BY_SPECIES, BLOCK_S, DTYPE)
        c1 = tl.load(sp_child1 + s_offs, mask=mask, other=S)
        c2 = tl.load(sp_child2 + s_offs, mask=mask, other=S)
        c1_valid = c1 < S
        c2_valid = c2 < S
        lsp = tl.load(log_split_probs + split_i)

        pi_off_l = tl.load(Pi_offset + left)
        pi_off_r = tl.load(Pi_offset + right)
        pibar_off_l = tl.load(Pibar_offset + left)
        pibar_off_r = tl.load(Pibar_offset + right)
        split_base_offset = tl.maximum(pi_off_l + pi_off_r, pi_off_l + pibar_off_r)
        split_base_offset = tl.maximum(split_base_offset, pi_off_r + pibar_off_l)
        new_base_offset = tl.maximum(tile_base_offset, split_base_offset)
        frame_shift = tl.where(
            tile_base_offset != NEG_INF,
            tile_base_offset - new_base_offset,
            0.0,
        ).to(DTYPE)
        m = tl.where(m != NEG_INF, m + frame_shift, m)
        corr0 = (pi_off_l + pi_off_r - new_base_offset).to(DTYPE)
        corr1 = (pi_off_l + pibar_off_r - new_base_offset).to(DTYPE)
        corr2 = (pi_off_r + pibar_off_l - new_base_offset).to(DTYPE)

        v0 = lsp + log_d + pi_l + pi_r + corr0
        v1 = lsp + pi_l + pibar_r + corr1
        v2 = lsp + pi_r + pibar_l + corr2
        v3 = (
            lsp
            + log_s
            + tl.load(Pi + base_l + c1, mask=mask & c1_valid, other=NEG_INF)
            + tl.load(Pi + base_r + c2, mask=mask & c2_valid, other=NEG_INF)
            + corr0
        )
        v4 = (
            lsp
            + log_s
            + tl.load(Pi + base_r + c1, mask=mask & c1_valid, other=NEG_INF)
            + tl.load(Pi + base_l + c2, mask=mask & c2_valid, other=NEG_INF)
            + corr0
        )
        split_m = tl.maximum(tl.maximum(tl.maximum(v0, v1), tl.maximum(v2, v3)), v4)
        split_m_safe = tl.where(split_m != NEG_INF, split_m, tl.zeros_like(split_m))
        split_sum = (
            tl.exp2(v0 - split_m_safe)
            + tl.exp2(v1 - split_m_safe)
            + tl.exp2(v2 - split_m_safe)
            + tl.exp2(v3 - split_m_safe)
            + tl.exp2(v4 - split_m_safe)
        )

        new_m = tl.maximum(m, split_m)
        new_m_safe = tl.where(new_m != NEG_INF, new_m, tl.zeros_like(new_m))
        acc = (
            tl.where(m != NEG_INF, acc * tl.exp2(m - new_m_safe), tl.zeros_like(acc))
            + split_sum * tl.exp2(split_m_safe - new_m_safe)
        )
        m = new_m
        tile_base_offset = new_base_offset
        split_rel += 1

    partial_row = group * MAX_TILES + tile_id
    tl.store(partial_max + partial_row * S + s_offs, m, mask=mask)
    tl.store(partial_sum + partial_row * S + s_offs, acc, mask=mask)
    # Species blocks race only to publish the same scalar tile frame.
    tl.store(partial_offset + partial_row, tile_base_offset)


@triton.jit
def _dts_ge2_stage2_kernel(
    ge2_ptr,
    ge2_parent_ids,
    active_rows,
    partial_max,
    partial_sum,
    partial_offset,
    out,
    out_offset,
    MAX_TILES: tl.constexpr,
    S: tl.constexpr,
    TILE_SPLITS: tl.constexpr,
    BLOCK_S: tl.constexpr,
    USE_ACTIVE: tl.constexpr,
    DTYPE: tl.constexpr,
):
    NEG_INF: tl.constexpr = -float("inf")
    group = tl.program_id(0)
    s_block = tl.program_id(1)
    s_offs = s_block * BLOCK_S + tl.arange(0, BLOCK_S)
    mask = s_offs < S
    parent_w = tl.load(ge2_parent_ids + group).to(tl.int64)
    if USE_ACTIVE:
        if tl.load(active_rows + parent_w) == 0:
            tl.store(out + parent_w * S + s_offs, tl.full([BLOCK_S], NEG_INF, dtype=DTYPE), mask=mask)
            return

    start = tl.load(ge2_ptr + group)
    end = tl.load(ge2_ptr + group + 1)
    n_tiles = tl.cdiv(end - start, TILE_SPLITS)
    m = tl.full([BLOCK_S], NEG_INF, dtype=DTYPE)
    acc = tl.zeros([BLOCK_S], dtype=DTYPE)
    row_base_offset = tl.full((), value=NEG_INF, dtype=tl.float64)
    tile_id = 0
    while tile_id < n_tiles:
        partial_row = group * MAX_TILES + tile_id
        pm = tl.load(partial_max + partial_row * S + s_offs, mask=mask, other=NEG_INF)
        ps = tl.load(partial_sum + partial_row * S + s_offs, mask=mask, other=0.0)
        tile_base_offset = tl.load(partial_offset + partial_row)
        new_base_offset = tl.maximum(row_base_offset, tile_base_offset)
        m = tl.where(
            m != NEG_INF,
            m + (row_base_offset - new_base_offset).to(DTYPE),
            m,
        )
        pm = tl.where(
            pm != NEG_INF,
            pm + (tile_base_offset - new_base_offset).to(DTYPE),
            pm,
        )
        new_m = tl.maximum(m, pm)
        new_m_safe = tl.where(new_m != NEG_INF, new_m, tl.zeros_like(new_m))
        acc = tl.where(m != NEG_INF, acc * tl.exp2(m - new_m_safe), tl.zeros_like(acc)) + tl.where(
            pm != NEG_INF, ps * tl.exp2(pm - new_m_safe), tl.zeros_like(acc)
        )
        m = new_m
        row_base_offset = new_base_offset
        tile_id += 1

    result = tl.log2(acc) + m
    tl.store(out + parent_w * S + s_offs, result, mask=mask)
    tile_has_finite = tl.max(tl.where(mask & (result != NEG_INF), 1, 0), axis=0) != 0
    tl.store(out_offset + parent_w, row_base_offset, mask=tile_has_finite)


def compute_leaf_initial_wave_step(
    Pi_out,
    Pi_out_offset,
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
    _leaf_initial_wave_step_kernel[(W,)](
        Pi_out,
        Pi_out_offset,
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


def compute_wave_step(
    Pi_in,
    Pi_in_offset,
    Pi_out,
    Pi_out_offset,
    Pibar,
    Pibar_offset,
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
    sp_parent,
    max_ancestor_depth,
    DTS_reduced=None,
    DTS_offset=None,
    DTS_center_offset=None,
    *,
    leaf_species_idx,
    leaf_logp,
    family_idx,
    pibar_row_max,
    store_final_pibar=False,
    has_leaf_term=True,
    input_ws=None,
    use_receiver_weights=True,
    pi_residual_out=None,
):
    has_splits = DTS_reduced is not None
    if has_splits and DTS_offset is None:
        raise ValueError("DTS_offset is required with row-gauged split DTS input")
    if has_splits and DTS_center_offset is None:
        # Direct callers that are not the first DTS-input launch have no
        # virtual shift to publish; their storage offset is the correct base.
        # The first input launch writes every freshly allocated sidecar lane.
        DTS_center_offset = (
            torch.empty_like(DTS_offset) if input_ws is not None else DTS_offset
        )
    if not has_splits:
        DTS_reduced = Pi_in
        DTS_offset = Pi_in_offset
        DTS_center_offset = Pi_in_offset
    if input_ws is not None and (
        not has_splits
        or int(input_ws) != 0
        or Pi_in.data_ptr() != DTS_reduced.data_ptr()
        or Pi_in_offset.data_ptr() != DTS_offset.data_ptr()
        or Pi_out.data_ptr() == DTS_reduced.data_ptr()
        or DTS_center_offset.data_ptr() == DTS_offset.data_ptr()
        or DTS_center_offset.data_ptr() == Pi_out_offset.data_ptr()
        or DTS_center_offset.data_ptr() == Pibar_offset.data_ptr()
    ):
        raise ValueError(
            "split virtual framing requires wave-local aliased Pi/DTS inputs "
            "and a distinct output buffer"
        )
    compute_diff = pi_residual_out is not None
    block_s, const_row_stride = _prepare_wave_launch(S, DL_const)
    _wave_step_kernel[(W,)](
        Pi_in,
        Pi_in_offset,
        ws,
        ws if input_ws is None else int(input_ws),
        max_transfer_mat,
        DL_const,
        Ebar,
        E,
        SL1_const,
        SL2_const,
        receiver_log_probs,
        sp_child1,
        sp_child2,
        sp_parent,
        leaf_species_idx,
        leaf_logp,
        family_idx,
        DTS_reduced,
        DTS_offset,
        DTS_center_offset,
        has_splits,
        bool(input_ws is not None),
        Pi_out,
        Pi_out_offset,
        Pibar,
        Pibar_offset,
        pibar_row_max,
        pi_residual_out if compute_diff else pibar_row_max,
        S,
        stride=S,
        CONST_ROW_STRIDE=const_row_stride,
        BLOCK_S=block_s,
        MAX_ANCESTOR_DEPTH=int(max_ancestor_depth),
        USE_LEAF_INDEX=bool(has_leaf_term),
        STORE_FINAL_PIBAR=bool(store_final_pibar),
        COMPUTE_DIFF=compute_diff,
        USE_RECEIVER_WEIGHTS=bool(use_receiver_weights),
        DTYPE=_tl_float_dtype(Pi_in.dtype),
        num_warps=8,
    )


def compute_dts_forward(
    Pi,
    Pi_offset,
    Pibar,
    Pibar_offset,
    lefts,
    rights,
    sp_child1,
    sp_child2,
    W,
    reduce_idx,
    log_pD_vec,
    log_pS_vec,
    family_idx,
    *,
    log_split_probs=None,
    n_eq1=None,
    eq1_reduce_idx=None,
    ge2_ptr=None,
    ge2_parent_ids=None,
    ge2_max_fanout=None,
    active_parent_rows=None,
    family_offset=0,
):
    N = int(lefts.shape[0])
    S = int(Pi.shape[1])
    out = torch.full((W, S), float("-inf"), device=Pi.device, dtype=Pi.dtype)
    out_offset = torch.zeros((W,), device=Pi.device, dtype=torch.float64)
    if N == 0:
        return out, out_offset
    Pi_offset = Pi_offset.to(device=Pi.device, dtype=torch.float64).contiguous()
    Pibar_offset = Pibar_offset.to(device=Pi.device, dtype=torch.float64).contiguous()
    if log_split_probs is None:
        log_split_probs = torch.zeros((N,), device=Pi.device, dtype=Pi.dtype)
    else:
        # Preprocessing owns split statics in fp64. Match the normal DTS boundary:
        # fp32 residual kernels must not acquire an fp64 loop-carried accumulator.
        log_split_probs = log_split_probs.reshape(N).to(Pi.dtype).contiguous()
    if n_eq1 is None:
        n_eq1 = N
        eq1_reduce_idx = reduce_idx
        ge2_parent_ids = reduce_idx[:0]
        ge2_ptr = reduce_idx.new_zeros((1,), dtype=torch.long)
        ge2_max_fanout = 0

    by_species = log_pD_vec.ndim == 2 and int(log_pD_vec.shape[1]) != 1
    row_stride = 0 if int(log_pD_vec.shape[0]) == 1 else int(log_pD_vec.stride(0))
    block_s = min(512, triton.next_power_of_2(S))
    active = active_parent_rows if active_parent_rows is not None else reduce_idx

    if int(n_eq1) > 0:
        _dts_eq1_kernel[(int(n_eq1), triton.cdiv(S, block_s))](
            Pi,
            Pi_offset,
            Pibar,
            Pibar_offset,
            lefts,
            rights,
            sp_child1,
            sp_child2,
            log_pD_vec,
            log_pS_vec,
            log_split_probs,
            eq1_reduce_idx,
            active,
            out,
            out_offset,
            family_idx,
            int(family_offset),
            S,
            BLOCK_S=block_s,
            ROW_STRIDE=row_stride,
            BY_SPECIES=bool(by_species),
            USE_ACTIVE=bool(active_parent_rows is not None),
            DTYPE=_tl_float_dtype(Pi.dtype),
        )

    if ge2_parent_ids is None or int(ge2_parent_ids.numel()) == 0:
        return out, out_offset
    tile_splits = 64
    if ge2_max_fanout is None:
        ge2_max_fanout = int((ge2_ptr[1:] - ge2_ptr[:-1]).max().item())
    max_tiles = max(1, triton.cdiv(int(ge2_max_fanout), tile_splits))
    n_groups = int(ge2_parent_ids.numel())
    partial_shape = (n_groups * max_tiles, S)
    partial_max = torch.empty(partial_shape, device=Pi.device, dtype=Pi.dtype)
    partial_sum = torch.empty(partial_shape, device=Pi.device, dtype=Pi.dtype)
    partial_offset = torch.empty(
        (n_groups * max_tiles,), device=Pi.device, dtype=torch.float64
    )
    _dts_ge2_stage1_kernel[(n_groups, max_tiles, triton.cdiv(S, block_s))](
        Pi,
        Pi_offset,
        Pibar,
        Pibar_offset,
        lefts,
        rights,
        sp_child1,
        sp_child2,
        log_pD_vec,
        log_pS_vec,
        log_split_probs,
        ge2_ptr,
        ge2_parent_ids,
        active,
        partial_max,
        partial_sum,
        partial_offset,
        family_idx,
        int(family_offset),
        split_offset=int(n_eq1),
        MAX_TILES=max_tiles,
        S=S,
        TILE_SPLITS=tile_splits,
        BLOCK_S=block_s,
        ROW_STRIDE=row_stride,
        BY_SPECIES=bool(by_species),
        USE_ACTIVE=bool(active_parent_rows is not None),
        DTYPE=_tl_float_dtype(Pi.dtype),
    )
    _dts_ge2_stage2_kernel[(n_groups, triton.cdiv(S, block_s))](
        ge2_ptr,
        ge2_parent_ids,
        active,
        partial_max,
        partial_sum,
        partial_offset,
        out,
        out_offset,
        MAX_TILES=max_tiles,
        S=S,
        TILE_SPLITS=tile_splits,
        BLOCK_S=block_s,
        USE_ACTIVE=bool(active_parent_rows is not None),
        DTYPE=_tl_float_dtype(Pi.dtype),
    )
    return out, out_offset
