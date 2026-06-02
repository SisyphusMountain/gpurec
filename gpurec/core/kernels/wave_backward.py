import torch
import triton
import triton.language as tl


@triton.jit
def _self_loop_coefficients_kernel(
    Pi_star_ptr,
    Pibar_star_ptr,
    Pibar_row_max_ptr,
    dts_r_ptr,
    has_splits: tl.constexpr,
    rhs_ptr,
    active_parent_rows_ptr,
    DL_const_ptr, Ebar_ptr, E_ptr, SL1_const_ptr, SL2_const_ptr,
    sp_child1_ptr, sp_child2_ptr, sp_parent_ptr,
    leaf_species_ptr,
    leaf_logp_ptr,
    family_idx_ptr,
    v_k_ptr,
    diag_ptr,
    pibar_coeff_ptr,
    p_prime_ptr,
    sl1_ptr,
    ws,
    S: tl.constexpr,
    stride: tl.constexpr,
    BLOCK_S: tl.constexpr,
    MAX_ANCESTOR_DEPTH: tl.constexpr,
    USE_LEAF_INDEX: tl.constexpr,
):
    NEG_LARGE: tl.constexpr = -1e30
    M_SAFE: tl.constexpr = -1e29

    w = tl.program_id(0)
    s_offs = tl.arange(0, BLOCK_S)
    species_valid = s_offs < S
    row_active = tl.load(active_parent_rows_ptr + w) != 0
    mask = species_valid & row_active

    row_global = ws + w
    pi_offsets = row_global * stride + s_offs
    out_offsets = w * S + s_offs

    row_max = tl.load(Pibar_row_max_ptr + row_global).to(tl.float32)
    pi_w = tl.load(Pi_star_ptr + pi_offsets, mask=mask, other=NEG_LARGE).to(tl.float32)
    pibar_w = tl.load(Pibar_star_ptr + pi_offsets, mask=mask, other=NEG_LARGE).to(tl.float32)
    p_prime = tl.exp2(pi_w - row_max)
    row_sum = tl.sum(tl.where(mask, p_prime, tl.zeros([BLOCK_S], dtype=tl.float32)), axis=0)

    family = tl.load(family_idx_ptr + row_global).to(tl.int64)
    const_base = family * stride
    const_offsets = const_base + s_offs
    dl_c = tl.load(DL_const_ptr + const_offsets, mask=mask, other=NEG_LARGE).to(tl.float32)
    ebar = tl.load(Ebar_ptr + const_offsets, mask=mask, other=NEG_LARGE).to(tl.float32)
    e_val = tl.load(E_ptr + const_offsets, mask=mask, other=NEG_LARGE).to(tl.float32)
    sl1_c = tl.load(SL1_const_ptr + const_offsets, mask=mask, other=NEG_LARGE).to(tl.float32)
    sl2_c = tl.load(SL2_const_ptr + const_offsets, mask=mask, other=NEG_LARGE).to(tl.float32)

    c1 = tl.load(sp_child1_ptr + s_offs, mask=species_valid, other=0)
    c2 = tl.load(sp_child2_ptr + s_offs, mask=species_valid, other=0)
    c1_valid = c1 < S
    c2_valid = c2 < S
    pi_s1 = tl.load(
        Pi_star_ptr + row_global * stride + c1,
        mask=mask & c1_valid,
        other=NEG_LARGE,
    ).to(tl.float32)
    pi_s2 = tl.load(
        Pi_star_ptr + row_global * stride + c2,
        mask=mask & c2_valid,
        other=NEG_LARGE,
    ).to(tl.float32)

    t0 = dl_c + pi_w
    t1 = pi_w + ebar
    t2 = pibar_w + e_val
    t3 = sl1_c + pi_s1
    t4 = sl2_c + pi_s2
    if USE_LEAF_INDEX:
        leaf_species = tl.load(leaf_species_ptr + row_global)
        leaf_hit = mask & (leaf_species == s_offs)
        leaf_logp = tl.load(
            leaf_logp_ptr + const_base + s_offs,
            mask=leaf_hit,
            other=NEG_LARGE,
        ).to(tl.float32)
        t5 = tl.where(leaf_hit, leaf_logp, NEG_LARGE)
    else:
        t5 = tl.full([BLOCK_S], value=NEG_LARGE, dtype=tl.float32)

    m = tl.maximum(t0, t1)
    m = tl.maximum(m, t2)
    m = tl.maximum(m, t3)
    m = tl.maximum(m, t4)
    m = tl.maximum(m, t5)
    m_safe = tl.where(m > M_SAFE, m, tl.zeros([BLOCK_S], dtype=tl.float32))
    e0 = tl.exp2(t0 - m_safe)
    e1 = tl.exp2(t1 - m_safe)
    e2 = tl.exp2(t2 - m_safe)
    e3 = tl.exp2(t3 - m_safe)
    e4 = tl.exp2(t4 - m_safe)
    e5 = tl.exp2(t5 - m_safe)
    dts_l_sum = e0 + e1 + e2 + e3 + e4 + e5
    inv_sum = tl.where(dts_l_sum > 0.0, 1.0 / dts_l_sum, tl.zeros_like(dts_l_sum))

    if has_splits:
        dts_r = tl.load(dts_r_ptr + out_offsets, mask=mask, other=NEG_LARGE).to(tl.float32)
        dts_l = tl.log2(dts_l_sum) + m
        pi_new_m = tl.maximum(dts_l, dts_r)
        pi_new_ms = tl.where(pi_new_m > M_SAFE, pi_new_m, tl.zeros_like(pi_new_m))
        pi_new = tl.log2(tl.exp2(dts_l - pi_new_ms) + tl.exp2(dts_r - pi_new_ms)) + pi_new_m
        w_L = tl.where(dts_l > M_SAFE, tl.exp2(dts_l - pi_new), tl.zeros_like(dts_l))
    else:
        w_L = tl.full([BLOCK_S], value=1.0, dtype=tl.float32)

    ancestor_sum = tl.zeros([BLOCK_S], dtype=tl.float32)
    cur = s_offs
    for _depth in range(MAX_ANCESTOR_DEPTH):
        cur_valid = species_valid & (cur >= 0) & (cur < S)
        pi_anc = tl.load(
            Pi_star_ptr + row_global * stride + cur,
            mask=cur_valid & row_active,
            other=NEG_LARGE,
        ).to(tl.float32)
        ancestor_sum += tl.where(
            cur_valid & row_active,
            tl.exp2(pi_anc - row_max),
            tl.zeros([BLOCK_S], dtype=tl.float32),
        )
        cur = tl.load(sp_parent_ptr + cur, mask=cur_valid, other=-1)
    denom = row_sum - ancestor_sum
    inv_denom = tl.where(denom > 0.0, 1.0 / denom, tl.zeros_like(denom))

    diag_wt = w_L * (e0 + e1) * inv_sum
    pibar_u_coeff = w_L * e2 * inv_sum * inv_denom
    sl1_wt = w_L * e3 * inv_sum
    sl2_wt = w_L * e4 * inv_sum

    rhs_val = tl.load(rhs_ptr + out_offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(v_k_ptr + out_offsets, rhs_val, mask=mask)
    tl.store(diag_ptr + out_offsets, diag_wt, mask=mask)
    tl.store(pibar_coeff_ptr + out_offsets, pibar_u_coeff, mask=mask)
    tl.store(p_prime_ptr + out_offsets, p_prime, mask=mask)
    child1_offsets = w * S + c1
    child2_offsets = w * S + c2
    child1_mask = species_valid & c1_valid & row_active
    child2_mask = species_valid & c2_valid & row_active
    tl.store(sl1_ptr + child1_offsets, sl1_wt, mask=child1_mask)
    tl.store(sl1_ptr + child2_offsets, sl2_wt, mask=child2_mask)


@triton.jit
def _self_loop_adjoint_update_kernel(
    term_in_ptr,
    term_out_ptr,
    active_parent_rows_ptr,
    diag_ptr,
    pibar_coeff_ptr,
    p_prime_ptr,
    sl1_ptr,
    sp_parent_ptr,
    compact_level_ptr,
    compact_level_parent_ptr,
    compact_level_child1_ptr,
    compact_level_child2_ptr,
    pibar_corr_ptr,
    v_k_ptr,
    S: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_NODES: tl.constexpr,
    N_LEVELS: tl.constexpr,
):
    w = tl.program_id(0)
    s_offs = tl.arange(0, BLOCK_S)
    species_valid = s_offs < S
    row_active = tl.load(active_parent_rows_ptr + w) != 0
    mask = species_valid & row_active
    row_base = w * S
    offsets = row_base + s_offs

    term_val = tl.load(term_in_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    pibar_u_coeff = tl.load(pibar_coeff_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    u_d = term_val * pibar_u_coeff
    A = tl.sum(tl.where(mask, u_d, tl.zeros([BLOCK_S], dtype=tl.float32)), axis=0)
    tl.store(pibar_corr_ptr + offsets, u_d, mask=mask)

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
            parent_val = tl.load(
                pibar_corr_ptr + row_base + parent,
                mask=node_mask & row_active,
                other=0.0,
            ).to(tl.float32)
            c1_val = tl.load(
                pibar_corr_ptr + row_base + c1,
                mask=node_mask & row_active & (c1 < S),
                other=0.0,
            ).to(tl.float32)
            c2_val = tl.load(
                pibar_corr_ptr + row_base + c2,
                mask=node_mask & row_active & (c2 < S),
                other=0.0,
            ).to(tl.float32)
            tl.store(
                pibar_corr_ptr + row_base + parent,
                parent_val + c1_val + c2_val,
                mask=node_mask & row_active,
            )
            node_start += BLOCK_NODES
        tl.debug_barrier()

    tl.debug_barrier()

    corr = tl.load(pibar_corr_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    diag_wt = tl.load(diag_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    p_prime = tl.load(p_prime_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    base = term_val * diag_wt + p_prime * (A - corr)

    parent = tl.load(sp_parent_ptr + s_offs, mask=species_valid, other=-1)
    parent_valid = species_valid & (parent >= 0) & (parent < S)
    parent_mask = parent_valid & row_active
    parent_term = tl.load(
        term_in_ptr + row_base + parent,
        mask=parent_mask,
        other=0.0,
    ).to(tl.float32)
    edge_wt = tl.load(sl1_ptr + offsets, mask=parent_mask, other=0.0).to(tl.float32)
    result = base + parent_term * edge_wt
    tl.store(term_out_ptr + offsets, result, mask=mask)

    v_prev = tl.load(v_k_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(v_k_ptr + offsets, v_prev + result, mask=mask)


@triton.jit
def _self_loop_parameter_gradient_kernel(
    Pi_star_ptr,
    Pibar_star_ptr,
    dts_r_ptr,
    has_splits: tl.constexpr,
    v_k_ptr,
    active_parent_rows_ptr,
    DL_const_ptr, Ebar_ptr, E_ptr, SL1_const_ptr, SL2_const_ptr,
    sp_child1_ptr, sp_child2_ptr,
    leaf_species_ptr,
    leaf_logp_ptr,
    family_idx_ptr,
    grad_log_pD_ptr,
    grad_log_pS_ptr,
    grad_E_ptr,
    grad_Ebar_ptr,
    grad_E_s1_ptr,
    grad_E_s2_ptr,
    grad_max_transfer_ptr,
    ws,
    S: tl.constexpr,
    stride: tl.constexpr,
    BLOCK_S: tl.constexpr,
    USE_LEAF_INDEX: tl.constexpr,
):
    NEG_LARGE: tl.constexpr = -1e30
    M_SAFE: tl.constexpr = -1e29

    w = tl.program_id(0)
    s_offs = tl.arange(0, BLOCK_S)
    species_valid = s_offs < S
    row_active = tl.load(active_parent_rows_ptr + w) != 0
    mask = species_valid & row_active
    row_global = ws + w
    pi_offsets = row_global * stride + s_offs
    out_offsets = w * S + s_offs

    family = tl.load(family_idx_ptr + row_global).to(tl.int64)
    const_base = family * stride
    const_offsets = const_base + s_offs
    pi_w = tl.load(Pi_star_ptr + pi_offsets, mask=mask, other=NEG_LARGE).to(tl.float32)
    pibar_w = tl.load(Pibar_star_ptr + pi_offsets, mask=mask, other=NEG_LARGE).to(tl.float32)
    v_k_val = tl.load(v_k_ptr + out_offsets, mask=mask, other=0.0).to(tl.float32)
    dl_c = tl.load(DL_const_ptr + const_offsets, mask=mask, other=NEG_LARGE).to(tl.float32)
    ebar = tl.load(Ebar_ptr + const_offsets, mask=mask, other=NEG_LARGE).to(tl.float32)
    e_val = tl.load(E_ptr + const_offsets, mask=mask, other=NEG_LARGE).to(tl.float32)
    sl1_c = tl.load(SL1_const_ptr + const_offsets, mask=mask, other=NEG_LARGE).to(tl.float32)
    sl2_c = tl.load(SL2_const_ptr + const_offsets, mask=mask, other=NEG_LARGE).to(tl.float32)

    c1 = tl.load(sp_child1_ptr + s_offs, mask=species_valid, other=0)
    c2 = tl.load(sp_child2_ptr + s_offs, mask=species_valid, other=0)
    c1_valid = c1 < S
    c2_valid = c2 < S
    pi_s1 = tl.load(
        Pi_star_ptr + row_global * stride + c1,
        mask=mask & c1_valid,
        other=NEG_LARGE,
    ).to(tl.float32)
    pi_s2 = tl.load(
        Pi_star_ptr + row_global * stride + c2,
        mask=mask & c2_valid,
        other=NEG_LARGE,
    ).to(tl.float32)

    t0 = dl_c + pi_w
    t1 = pi_w + ebar
    t2 = pibar_w + e_val
    t3 = sl1_c + pi_s1
    t4 = sl2_c + pi_s2
    if USE_LEAF_INDEX:
        leaf_species = tl.load(leaf_species_ptr + row_global)
        leaf_hit = mask & (leaf_species == s_offs)
        leaf_logp = tl.load(
            leaf_logp_ptr + const_base + s_offs,
            mask=leaf_hit,
            other=NEG_LARGE,
        ).to(tl.float32)
        t5 = tl.where(leaf_hit, leaf_logp, NEG_LARGE)
    else:
        t5 = tl.full([BLOCK_S], value=NEG_LARGE, dtype=tl.float32)

    m = tl.maximum(t0, t1)
    m = tl.maximum(m, t2)
    m = tl.maximum(m, t3)
    m = tl.maximum(m, t4)
    m = tl.maximum(m, t5)
    m_safe = tl.where(m > M_SAFE, m, tl.zeros([BLOCK_S], dtype=tl.float32))
    e0 = tl.exp2(t0 - m_safe)
    e1 = tl.exp2(t1 - m_safe)
    e2 = tl.exp2(t2 - m_safe)
    e3 = tl.exp2(t3 - m_safe)
    e4 = tl.exp2(t4 - m_safe)
    e5 = tl.exp2(t5 - m_safe)
    dts_l_sum = e0 + e1 + e2 + e3 + e4 + e5
    inv_sum = tl.where(dts_l_sum > 0.0, 1.0 / dts_l_sum, tl.zeros_like(dts_l_sum))

    if has_splits:
        dts_r = tl.load(dts_r_ptr + out_offsets, mask=mask, other=NEG_LARGE).to(tl.float32)
        dts_l = tl.log2(dts_l_sum) + m
        pi_new_m = tl.maximum(dts_l, dts_r)
        pi_new_ms = tl.where(pi_new_m > M_SAFE, pi_new_m, tl.zeros_like(pi_new_m))
        pi_new = tl.log2(tl.exp2(dts_l - pi_new_ms) + tl.exp2(dts_r - pi_new_ms)) + pi_new_m
        w_L = tl.where(dts_l > M_SAFE, tl.exp2(dts_l - pi_new), tl.zeros_like(dts_l))
    else:
        w_L = tl.full([BLOCK_S], value=1.0, dtype=tl.float32)

    alpha = v_k_val * w_L
    aw0_s = tl.where(mask, alpha * e0 * inv_sum, 0.0)
    aw1_s = tl.where(mask, alpha * e1 * inv_sum, 0.0)
    aw2_s = tl.where(mask, alpha * e2 * inv_sum, 0.0)
    aw345_s = tl.where(mask, (alpha * e3 * inv_sum) + (alpha * e4 * inv_sum) + (alpha * e5 * inv_sum), 0.0)
    aw3_s = tl.where(mask, alpha * e3 * inv_sum, 0.0)
    aw4_s = tl.where(mask, alpha * e4 * inv_sum, 0.0)
    grad_species_offset = family * S + s_offs
    tl.atomic_add(grad_log_pD_ptr + family, tl.sum(aw0_s, axis=0), sem="relaxed")
    tl.atomic_add(grad_log_pS_ptr + family, tl.sum(aw345_s, axis=0), sem="relaxed")
    tl.atomic_add(grad_E_ptr + grad_species_offset, aw0_s + aw2_s, sem="relaxed", mask=species_valid & row_active)
    tl.atomic_add(grad_Ebar_ptr + grad_species_offset, aw1_s, sem="relaxed", mask=species_valid & row_active)
    tl.atomic_add(grad_E_s1_ptr + grad_species_offset, aw4_s, sem="relaxed", mask=species_valid & row_active)
    tl.atomic_add(grad_E_s2_ptr + grad_species_offset, aw3_s, sem="relaxed", mask=species_valid & row_active)
    tl.atomic_add(grad_max_transfer_ptr + grad_species_offset, aw2_s, sem="relaxed", mask=species_valid & row_active)


def compute_wave_adjoint(
    Pi_star, Pibar_star, ws, W, S,
    dts_r,
    rhs,
    DL_const, Ebar, E, SL1_const, SL2_const,
    sp_child1, sp_child2,
    *,
    family_idx,
    leaf_species_idx,
    leaf_logp,
    has_leaf_term,
    active_parent_rows,
    sp_parent,
    max_ancestor_depth,
    pibar_row_max,
    compact_level_ptr,
    compact_level_parents,
    compact_level_child1,
    compact_level_child2,
    self_loop_grad_targets,
):
    block_s = triton.next_power_of_2(S)
    v_k, aw0, aw1, aw2, aw3, spec_buf, term_buf, pibar_corr = (
        torch.empty_like(rhs) for _ in range(8)
    )
    has_splits = dts_r is not None
    dts_arg = dts_r if has_splits else Pi_star

    _self_loop_coefficients_kernel[(W,)](
        Pi_star,
        Pibar_star,
        pibar_row_max,
        dts_arg,
        has_splits,
        rhs,
        active_parent_rows,
        DL_const,
        Ebar,
        E,
        SL1_const,
        SL2_const,
        sp_child1,
        sp_child2,
        sp_parent,
        leaf_species_idx,
        leaf_logp,
        family_idx,
        v_k,
        aw0,
        aw1,
        aw2,
        aw3,
        ws,
        S,
        Pi_star.stride(0),
        block_s,
        max(1, int(max_ancestor_depth)),
        USE_LEAF_INDEX=bool(has_leaf_term),
        num_warps=8,
    )

    def _launch_self_loop_adjoint_update(term_in, term_out):
        _self_loop_adjoint_update_kernel[(W,)](
            term_in,
            term_out,
            active_parent_rows,
            aw0,
            aw1,
            aw2,
            aw3,
            sp_parent,
            compact_level_ptr,
            compact_level_parents,
            compact_level_child1,
            compact_level_child2,
            pibar_corr,
            v_k,
            S,
            block_s,
            128,
            compact_level_ptr.numel() - 1,
            num_warps=2,
        )

    for n in range(3):
        term_in = rhs if n == 0 else (spec_buf if n % 2 == 1 else term_buf)
        term_out = spec_buf if n % 2 == 0 else term_buf
        _launch_self_loop_adjoint_update(term_in, term_out)

    (
        grad_log_pD_ptr,
        grad_log_pS_ptr,
        grad_E_ptr,
        grad_Ebar_ptr,
        grad_E_s1_ptr,
        grad_E_s2_ptr,
        grad_max_transfer_ptr,
    ) = self_loop_grad_targets

    _self_loop_parameter_gradient_kernel[(W,)](
        Pi_star,
        Pibar_star,
        dts_arg,
        has_splits,
        v_k,
        active_parent_rows,
        DL_const,
        Ebar,
        E,
        SL1_const,
        SL2_const,
        sp_child1,
        sp_child2,
        leaf_species_idx,
        leaf_logp,
        family_idx,
        grad_log_pD_ptr,
        grad_log_pS_ptr,
        grad_E_ptr,
        grad_Ebar_ptr,
        grad_E_s1_ptr,
        grad_E_s2_ptr,
        grad_max_transfer_ptr,
        ws,
        S,
        Pi_star.stride(0),
        block_s,
        USE_LEAF_INDEX=bool(has_leaf_term),
        num_warps=8,
    )

    return v_k

@triton.jit
def _split_dts_vjp_kernel(
    Pi_star_ptr,
    Pibar_star_ptr,
    v_k_ptr,
    active_parent_rows_ptr,
    sl_ptr,
    sr_ptr,
    reduce_idx_ptr,
    log_pD_arg,
    log_pS_arg,
    family_idx_ptr,
    sp_child1_ptr,
    sp_child2_ptr,
    accumulated_rhs_ptr,
    grad_log_pD_ptr,
    grad_log_pS_ptr,
    grad_max_transfer_ptr,
    pibar_ud_ptr,
    pibar_A_ptr,
    max_transfer_ptr,
    pibar_row_max_ptr,
    ws,
    S: tl.constexpr,
    stride_C: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    NEG_LARGE: tl.constexpr = -1e30

    i = tl.program_id(0)

    sl = tl.load(sl_ptr + i).to(tl.int64)
    sr = tl.load(sr_ptr + i).to(tl.int64)
    parent_w = tl.load(reduce_idx_ptr + i).to(tl.int64)
    if tl.load(active_parent_rows_ptr + parent_w) == 0:
        return
    _scalar_off = tl.arange(0, 1)

    parent_global = ws + parent_w
    parent_family = tl.load(family_idx_ptr + parent_global).to(tl.int64)
    log_pD = tl.load(log_pD_arg + parent_family).to(tl.float32)
    log_pS = tl.load(log_pS_arg + parent_family).to(tl.float32)

    pi_l_base = sl * stride_C
    pi_r_base = sr * stride_C
    parent_pi_base = (ws + parent_w) * stride_C
    parent_vk_base = parent_w * S

    sum_pD = tl.zeros((1,), dtype=tl.float32)
    sum_pS = tl.zeros((1,), dtype=tl.float32)
    sum_ud_l = tl.zeros((1,), dtype=tl.float32)
    sum_ud_r = tl.zeros((1,), dtype=tl.float32)
    row_max_l = tl.load(pibar_row_max_ptr + sl).to(tl.float32)
    row_max_r = tl.load(pibar_row_max_ptr + sr).to(tl.float32)

    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        valid_mask = s_offs < S

        Pi_l = tl.load(Pi_star_ptr + pi_l_base + s_offs, mask=valid_mask, other=NEG_LARGE).to(tl.float32)
        Pi_r = tl.load(Pi_star_ptr + pi_r_base + s_offs, mask=valid_mask, other=NEG_LARGE).to(tl.float32)
        Pibar_l = tl.load(Pibar_star_ptr + pi_l_base + s_offs, mask=valid_mask, other=NEG_LARGE).to(tl.float32)
        Pibar_r = tl.load(Pibar_star_ptr + pi_r_base + s_offs, mask=valid_mask, other=NEG_LARGE).to(tl.float32)

        c1 = tl.load(sp_child1_ptr + s_offs, mask=valid_mask, other=0)
        c2 = tl.load(sp_child2_ptr + s_offs, mask=valid_mask, other=0)
        c1_valid = (c1 < S) & valid_mask
        c2_valid = (c2 < S) & valid_mask
        Pi_l_s1 = tl.load(Pi_star_ptr + pi_l_base + c1, mask=c1_valid, other=NEG_LARGE).to(tl.float32)
        Pi_l_s2 = tl.load(Pi_star_ptr + pi_l_base + c2, mask=c2_valid, other=NEG_LARGE).to(tl.float32)
        Pi_r_s1 = tl.load(Pi_star_ptr + pi_r_base + c1, mask=c1_valid, other=NEG_LARGE).to(tl.float32)
        Pi_r_s2 = tl.load(Pi_star_ptr + pi_r_base + c2, mask=c2_valid, other=NEG_LARGE).to(tl.float32)

        Pi_parent = tl.load(Pi_star_ptr + parent_pi_base + s_offs, mask=valid_mask, other=NEG_LARGE).to(tl.float32)
        v_k_val = tl.load(v_k_ptr + parent_vk_base + s_offs, mask=valid_mask, other=0.0).to(tl.float32)

        d0 = log_pD + Pi_l + Pi_r
        d1 = Pi_l + Pibar_r
        d2 = Pi_r + Pibar_l
        d3 = log_pS + Pi_l_s1 + Pi_r_s2
        d4 = log_pS + Pi_r_s1 + Pi_l_s2

        parent_valid = Pi_parent > NEG_LARGE
        w0 = tl.where(parent_valid, tl.exp2(d0 - Pi_parent), tl.zeros_like(d0))
        w1 = tl.where(parent_valid, tl.exp2(d1 - Pi_parent), tl.zeros_like(d1))
        w2 = tl.where(parent_valid, tl.exp2(d2 - Pi_parent), tl.zeros_like(d2))
        w3 = tl.where(parent_valid, tl.exp2(d3 - Pi_parent), tl.zeros_like(d3))
        w4 = tl.where(parent_valid, tl.exp2(d4 - Pi_parent), tl.zeros_like(d4))

        vd0 = v_k_val * w0
        vd1 = v_k_val * w1
        vd2 = v_k_val * w2
        vd3 = v_k_val * w3
        vd4 = v_k_val * w4

        tl.atomic_add(accumulated_rhs_ptr + pi_l_base + s_offs, vd0 + vd1, sem="relaxed", mask=valid_mask)
        tl.atomic_add(accumulated_rhs_ptr + pi_r_base + s_offs, vd0 + vd2, sem="relaxed", mask=valid_mask)
        max_transfer_l = tl.load(max_transfer_ptr + parent_family * S + s_offs, mask=valid_mask, other=0.0).to(tl.float32)
        max_transfer_r = tl.load(max_transfer_ptr + parent_family * S + s_offs, mask=valid_mask, other=0.0).to(tl.float32)
        finite_l = Pibar_l > -1e29
        finite_r = Pibar_r > -1e29
        inv_denom_l = tl.where(
            finite_l,
            tl.exp2(row_max_l + max_transfer_l - Pibar_l),
            tl.zeros([BLOCK_S], dtype=tl.float32),
        )
        inv_denom_r = tl.where(
            finite_r,
            tl.exp2(row_max_r + max_transfer_r - Pibar_r),
            tl.zeros([BLOCK_S], dtype=tl.float32),
        )
        ud_l = vd2 * inv_denom_l
        ud_r = vd1 * inv_denom_r
        tl.store(pibar_ud_ptr + i * S + s_offs, ud_l, mask=valid_mask)
        tl.store(pibar_ud_ptr + (tl.num_programs(0) + i) * S + s_offs, ud_r, mask=valid_mask)
        sum_ud_l += tl.sum(tl.where(valid_mask, ud_l, 0.0), axis=0)
        sum_ud_r += tl.sum(tl.where(valid_mask, ud_r, 0.0), axis=0)

        sum_pD += tl.sum(vd0, axis=0)
        sum_pS += tl.sum(vd3 + vd4, axis=0)
        tl.atomic_add(grad_max_transfer_ptr + parent_family * S + s_offs, vd2 + vd1, sem="relaxed", mask=valid_mask)

        tl.atomic_add(accumulated_rhs_ptr + pi_l_base + c1, vd3, sem="relaxed", mask=c1_valid)
        tl.atomic_add(accumulated_rhs_ptr + pi_r_base + c1, vd4, sem="relaxed", mask=c1_valid)
        tl.atomic_add(accumulated_rhs_ptr + pi_r_base + c2, vd3, sem="relaxed", mask=c2_valid)
        tl.atomic_add(accumulated_rhs_ptr + pi_l_base + c2, vd4, sem="relaxed", mask=c2_valid)

    tl.atomic_add(grad_log_pD_ptr + parent_family + _scalar_off, sum_pD, sem="relaxed")
    tl.atomic_add(grad_log_pS_ptr + parent_family + _scalar_off, sum_pS, sem="relaxed")
    tl.store(pibar_A_ptr + i + _scalar_off, sum_ud_l)
    tl.store(pibar_A_ptr + tl.num_programs(0) + i + _scalar_off, sum_ud_r)


def accumulate_split_dts_vjp(
    Pi_star, Pibar_star, v_k, ws,
    sl, sr, reduce_idx,
    log_pD, log_pS,
    sp_child1, sp_child2,
    accumulated_rhs,
    S,
    *,
    family_idx,
    active_parent_rows,
    grad_log_pD,
    grad_log_pS,
    grad_max_transfer_mat,
    max_transfer_mat,
    pibar_row_max,
):
    n_ws = sl.shape[0]
    device = Pi_star.device
    dtype = Pi_star.dtype

    pibar_ud = torch.empty((2 * n_ws, S), device=device, dtype=dtype)
    pibar_A = torch.empty((2 * n_ws,), device=device, dtype=dtype)

    stride_C = Pi_star.stride(0)
    BLOCK_S = min(256, triton.next_power_of_2(S))
    _split_dts_vjp_kernel[(n_ws,)](
        Pi_star, Pibar_star,
        v_k,
        active_parent_rows,
        sl, sr, reduce_idx,
        log_pD, log_pS, family_idx,
        sp_child1, sp_child2,
        accumulated_rhs,
        grad_log_pD, grad_log_pS, grad_max_transfer_mat,
        pibar_ud, pibar_A, max_transfer_mat, pibar_row_max,
        ws, S, stride_C, BLOCK_S,
        num_warps=8,
    )

    return pibar_ud, pibar_A


@triton.jit
def _pibar_vjp_kernel(
    Pi_star_ptr,
    pibar_ud_ptr,
    pibar_A_ptr,
    sl_ptr,
    sr_ptr,
    reduce_idx_ptr,
    active_parent_rows_ptr,
    pibar_row_max_ptr,
    compact_level_ptr,
    compact_level_parent_ptr,
    compact_level_child1_ptr,
    compact_level_child2_ptr,
    accumulated_rhs_ptr,
    n_ws: tl.constexpr,
    S: tl.constexpr,
    stride_C: tl.constexpr,
    BLOCK_S: tl.constexpr,
    N_LEVELS: tl.constexpr,
):
    NEG_LARGE: tl.constexpr = -1e30

    row = tl.program_id(0)
    split_i = tl.where(row < n_ws, row, row - n_ws)
    is_right = row >= n_ws

    child_l = tl.load(sl_ptr + split_i).to(tl.int64)
    child_r = tl.load(sr_ptr + split_i).to(tl.int64)
    child = tl.where(is_right, child_r, child_l)
    parent_w = tl.load(reduce_idx_ptr + split_i).to(tl.int64)
    if tl.load(active_parent_rows_ptr + parent_w) == 0:
        return

    pi_base = child * stride_C
    row_base = row * S
    row_max = tl.load(pibar_row_max_ptr + child).to(tl.float32)
    A = tl.load(pibar_A_ptr + row).to(tl.float32)

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
            parent_valid = node_mask & (parent >= 0) & (parent < S)
            c1_valid = node_mask & (c1 >= 0) & (c1 < S)
            c2_valid = node_mask & (c2 >= 0) & (c2 < S)

            parent_val = tl.load(pibar_ud_ptr + row_base + parent, mask=parent_valid, other=0.0)
            c1_val = tl.load(pibar_ud_ptr + row_base + c1, mask=c1_valid, other=0.0)
            c2_val = tl.load(pibar_ud_ptr + row_base + c2, mask=c2_valid, other=0.0)
            tl.store(pibar_ud_ptr + row_base + parent, parent_val + c1_val + c2_val, mask=parent_valid)
            p_start += BLOCK_S
        tl.debug_barrier()

    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        valid_mask = s_offs < S
        pi_val = tl.load(Pi_star_ptr + pi_base + s_offs, mask=valid_mask, other=NEG_LARGE)
        p_prime = tl.exp2(pi_val - row_max)
        subtree_sum = tl.load(pibar_ud_ptr + row_base + s_offs, mask=valid_mask, other=0.0)
        contrib = p_prime * (A - subtree_sum)
        tl.atomic_add(accumulated_rhs_ptr + pi_base + s_offs, contrib, sem="relaxed", mask=valid_mask)


def accumulate_split_pibar_vjp(
    Pi_star,
    pibar_ud,
    pibar_A,
    sl,
    sr,
    accumulated_rhs,
    S,
    active_parent_rows,
    reduce_idx,
    pibar_row_max,
    compact_level_ptr,
    compact_level_parents,
    compact_level_child1,
    compact_level_child2,
):
    n_ws = sl.shape[0]
    BLOCK_S = min(256, triton.next_power_of_2(S))
    stride_C = Pi_star.stride(0)

    _pibar_vjp_kernel[(2 * n_ws,)](
        Pi_star,
        pibar_ud,
        pibar_A,
        sl,
        sr,
        reduce_idx,
        active_parent_rows,
        pibar_row_max,
        compact_level_ptr,
        compact_level_parents,
        compact_level_child1,
        compact_level_child2,
        accumulated_rhs,
        n_ws,
        S,
        stride_C,
        BLOCK_S,
        N_LEVELS=compact_level_ptr.numel() - 1,
        num_warps=4,
    )
