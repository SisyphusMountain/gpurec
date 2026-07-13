"""DTS second-order kernels; see ``docs/latex/kernel_mathematics.tex``."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from gpurec.core.kernels.pi_forward import (
    _load_rate,
    _tl_float_dtype,
    _validate_offset_tensor,
)


@triton.jit
def _dts_split_so_kernel(
    Pi, dPi, Pibar, dPibar, Pi_offset, Pibar_offset, v_ptr,
    split_left_rows, split_right_rows, species_child1, species_child2,
    log_pD, log_pS, dlog_pD, dlog_pS, mt_ptr, dmt_ptr,
    log_split_probs, reduce_idx, item_idx, item_offset, ws,
    pibar_row_max_ptr,
    d_rhs_ptr,
    ud_l_ptr, ud_r_ptr, dud_l_ptr, dud_r_ptr,
    d_grad_pD_ptr, d_grad_pS_ptr, d_grad_mt_ptr,
    S: tl.constexpr, BLOCK_S: tl.constexpr, ROW_STRIDE: tl.constexpr,
    BY_STATE: tl.constexpr, MT_ROW_STRIDE: tl.constexpr, DTYPE: tl.constexpr,
):
    LN2 = 0.6931471805599453
    NEG_INF = -float("inf")
    n = tl.program_id(0)
    s_block = tl.program_id(1)
    s_offs = s_block * BLOCK_S + tl.arange(0, BLOCK_S)
    mask = s_offs < S
    zero = tl.zeros([BLOCK_S], dtype=DTYPE)
    # Metadata remains int32 in memory; flattened address arithmetic is widened
    # locally so row * S cannot overflow.
    parent_w = tl.load(reduce_idx + n).to(tl.int64)
    item = tl.load(item_idx + item_offset + parent_w).to(tl.int64)
    left = tl.load(split_left_rows + n).to(tl.int64)
    right = tl.load(split_right_rows + n).to(tl.int64)
    base_l = left * S
    base_r = right * S
    base_p = (ws + parent_w) * S

    pi_offset_l = tl.load(Pi_offset + left)
    pi_offset_r = tl.load(Pi_offset + right)
    pi_offset_parent = tl.load(Pi_offset + ws + parent_w)
    pibar_offset_l = tl.load(Pibar_offset + left)
    pibar_offset_r = tl.load(Pibar_offset + right)
    # Offsets may use wider accumulation precision. Event probabilities belong
    # to the residual recurrence, so frame shifts are narrowed exactly once at
    # this boundary before they are combined with Pi/Pibar values.
    child_frame_shift = (pi_offset_l + pi_offset_r - pi_offset_parent).to(DTYPE)
    left_transfer_frame_shift = (
        pi_offset_l + pibar_offset_r - pi_offset_parent
    ).to(DTYPE)
    right_transfer_frame_shift = (
        pi_offset_r + pibar_offset_l - pi_offset_parent
    ).to(DTYPE)
    left_exclusion_frame_shift = (pi_offset_l - pibar_offset_l).to(DTYPE)
    right_exclusion_frame_shift = (pi_offset_r - pibar_offset_r).to(DTYPE)

    pi_l = tl.load(Pi + base_l + s_offs, mask=mask, other=NEG_INF)
    pi_r = tl.load(Pi + base_r + s_offs, mask=mask, other=NEG_INF)
    dpi_l = tl.load(dPi + base_l + s_offs, mask=mask, other=0.0)
    dpi_r = tl.load(dPi + base_r + s_offs, mask=mask, other=0.0)
    pibar_l = tl.load(Pibar + base_l + s_offs, mask=mask, other=NEG_INF)
    pibar_r = tl.load(Pibar + base_r + s_offs, mask=mask, other=NEG_INF)
    dpibar_l = tl.load(dPibar + base_l + s_offs, mask=mask, other=0.0)
    dpibar_r = tl.load(dPibar + base_r + s_offs, mask=mask, other=0.0)
    pi_p = tl.load(Pi + base_p + s_offs, mask=mask, other=NEG_INF)
    dpi_p = tl.load(dPi + base_p + s_offs, mask=mask, other=0.0)
    v = tl.load(v_ptr + parent_w * S + s_offs, mask=mask, other=0.0)

    log_d = _load_rate(log_pD, item, s_offs, mask, S, ROW_STRIDE, BY_STATE, BLOCK_S, DTYPE)
    log_s = _load_rate(log_pS, item, s_offs, mask, S, ROW_STRIDE, BY_STATE, BLOCK_S, DTYPE)
    dlog_d = _load_rate(dlog_pD, item, s_offs, mask, S, ROW_STRIDE, BY_STATE, BLOCK_S, DTYPE)
    dlog_s = _load_rate(dlog_pS, item, s_offs, mask, S, ROW_STRIDE, BY_STATE, BLOCK_S, DTYPE)
    mt = tl.load(mt_ptr + item * MT_ROW_STRIDE + s_offs, mask=mask, other=0.0)
    dmt = tl.load(dmt_ptr + item * MT_ROW_STRIDE + s_offs, mask=mask, other=0.0)

    c1 = tl.load(species_child1 + s_offs, mask=mask, other=S)
    c2 = tl.load(species_child2 + s_offs, mask=mask, other=S)
    c1v = mask & (c1 < S)
    c2v = mask & (c2 < S)
    pi_l_c1 = tl.load(Pi + base_l + c1, mask=c1v, other=NEG_INF)
    pi_r_c2 = tl.load(Pi + base_r + c2, mask=c2v, other=NEG_INF)
    pi_r_c1 = tl.load(Pi + base_r + c1, mask=c1v, other=NEG_INF)
    pi_l_c2 = tl.load(Pi + base_l + c2, mask=c2v, other=NEG_INF)
    dpi_l_c1 = tl.load(dPi + base_l + c1, mask=c1v, other=0.0)
    dpi_r_c2 = tl.load(dPi + base_r + c2, mask=c2v, other=0.0)
    dpi_r_c1 = tl.load(dPi + base_r + c1, mask=c1v, other=0.0)
    dpi_l_c2 = tl.load(dPi + base_l + c2, mask=c2v, other=0.0)
    lsp = tl.load(log_split_probs + n)
    duplication_log_weight = lsp + log_d + pi_l + pi_r + child_frame_shift
    left_transfer_log_weight = lsp + pi_l + pibar_r + left_transfer_frame_shift
    right_transfer_log_weight = lsp + pi_r + pibar_l + right_transfer_frame_shift
    speciation_lr_log_weight = (
        lsp + log_s + pi_l_c1 + pi_r_c2 + child_frame_shift
    )
    speciation_rl_log_weight = (
        lsp + log_s + pi_r_c1 + pi_l_c2 + child_frame_shift
    )
    d_duplication = dlog_d + dpi_l + dpi_r
    d_left_transfer = dpi_l + dpibar_r
    d_right_transfer = dpi_r + dpibar_l
    d_speciation_lr = dlog_s + dpi_l_c1 + dpi_r_c2
    d_speciation_rl = dlog_s + dpi_r_c1 + dpi_l_c2

    is_finite = mask & (pi_p != NEG_INF)
    duplication_probability = tl.where(
        is_finite, tl.exp2(duplication_log_weight - pi_p), zero
    )
    left_transfer_probability = tl.where(
        is_finite, tl.exp2(left_transfer_log_weight - pi_p), zero
    )
    right_transfer_probability = tl.where(
        is_finite, tl.exp2(right_transfer_log_weight - pi_p), zero
    )
    speciation_lr_probability = tl.where(
        is_finite, tl.exp2(speciation_lr_log_weight - pi_p), zero
    )
    speciation_rl_probability = tl.where(
        is_finite, tl.exp2(speciation_rl_log_weight - pi_p), zero
    )
    left_transfer_adjoint = v * left_transfer_probability
    right_transfer_adjoint = v * right_transfer_probability
    d_duplication_adjoint = (
        v * LN2 * duplication_probability * (d_duplication - dpi_p)
    )
    d_left_transfer_adjoint = (
        v * LN2 * left_transfer_probability * (d_left_transfer - dpi_p)
    )
    d_right_transfer_adjoint = (
        v * LN2 * right_transfer_probability * (d_right_transfer - dpi_p)
    )
    d_speciation_lr_adjoint = (
        v * LN2 * speciation_lr_probability * (d_speciation_lr - dpi_p)
    )
    d_speciation_rl_adjoint = (
        v * LN2 * speciation_rl_probability * (d_speciation_rl - dpi_p)
    )

    # tangent of the rhs scatters (same targets as the primal)
    tl.atomic_add(d_rhs_ptr + base_l + s_offs, d_duplication_adjoint + d_left_transfer_adjoint, sem="relaxed", mask=mask)
    tl.atomic_add(d_rhs_ptr + base_r + s_offs, d_duplication_adjoint + d_right_transfer_adjoint, sem="relaxed", mask=mask)
    tl.atomic_add(d_rhs_ptr + base_l + c1, d_speciation_lr_adjoint, sem="relaxed", mask=c1v)
    tl.atomic_add(d_rhs_ptr + base_r + c1, d_speciation_rl_adjoint, sem="relaxed", mask=c1v)
    tl.atomic_add(d_rhs_ptr + base_r + c2, d_speciation_lr_adjoint, sem="relaxed", mask=c2v)
    tl.atomic_add(d_rhs_ptr + base_l + c2, d_speciation_rl_adjoint, sem="relaxed", mask=c2v)

    # pibar staging: ud = vd * 2^{rm + mt - Pibar} (rm frozen), d(ud) = dvd*f + vd*ln2*f*(dmt - dPibar)
    rm_l = tl.load(pibar_row_max_ptr + left).to(DTYPE)
    rm_r = tl.load(pibar_row_max_ptr + right).to(DTYPE)
    fl_ok = mask & (pibar_l != NEG_INF)
    fr_ok = mask & (pibar_r != NEG_INF)
    f_l = tl.where(fl_ok, tl.exp2(rm_l + mt - pibar_l + left_exclusion_frame_shift), zero)
    f_r = tl.where(fr_ok, tl.exp2(rm_r + mt - pibar_r + right_exclusion_frame_shift), zero)
    ud_l = right_transfer_adjoint * f_l
    ud_r = left_transfer_adjoint * f_r
    dud_l = d_right_transfer_adjoint * f_l + right_transfer_adjoint * LN2 * f_l * (dmt - dpibar_l)
    dud_r = d_left_transfer_adjoint * f_r + left_transfer_adjoint * LN2 * f_r * (dmt - dpibar_r)
    tl.store(ud_l_ptr + n * S + s_offs, ud_l, mask=mask)
    tl.store(ud_r_ptr + n * S + s_offs, ud_r, mask=mask)
    tl.store(dud_l_ptr + n * S + s_offs, dud_l, mask=mask)
    tl.store(dud_r_ptr + n * S + s_offs, dud_r, mask=mask)

    # parameter tangents (same buckets as the primal accumulations)
    tl.atomic_add(d_grad_pD_ptr + item * S + s_offs, d_duplication_adjoint, sem="relaxed", mask=mask)
    tl.atomic_add(d_grad_pS_ptr + item * S + s_offs, d_speciation_lr_adjoint + d_speciation_rl_adjoint, sem="relaxed", mask=mask)
    tl.atomic_add(d_grad_mt_ptr + item * S + s_offs, d_left_transfer_adjoint + d_right_transfer_adjoint, sem="relaxed", mask=mask)


@triton.jit
def _dts_tree_so_kernel(
    Pi_ptr, dPi_ptr, col_log_probs_ptr, dcol_ptr,
    receiver_adjoint_ptr, d_receiver_adjoint_ptr,
    split_left_rows_ptr, split_right_rows_ptr,
    pibar_row_max_ptr,
    level_offsets_ptr, level_parents_ptr,
    level_child1_ptr, level_child2_ptr,
    d_rhs_ptr, d_grad_col_ptr,
    n_ws: tl.constexpr, S: tl.constexpr, stride_C: tl.constexpr,
    BLOCK_S: tl.constexpr, N_LEVELS: tl.constexpr,
    USE_COL_WEIGHTS: tl.constexpr, DTYPE: tl.constexpr,
):
    """Evaluate the DTS transfer-tree curvature term documented in LaTeX."""
    LN2 = 0.6931471805599453
    NEG = -float("inf")
    row = tl.program_id(0)
    split_i = tl.where(row < n_ws, row, row - n_ws)
    is_right = row >= n_ws
    child_l = tl.load(split_left_rows_ptr + split_i).to(tl.int64)
    child_r = tl.load(split_right_rows_ptr + split_i).to(tl.int64)
    child = tl.where(is_right, child_r, child_l)

    pi_base = child * stride_C
    row_base = row * S
    rm = tl.load(pibar_row_max_ptr + child).to(DTYPE)
    rm_safe = tl.where(rm != NEG, rm, tl.zeros_like(rm))

    # row totals A = sum_s ud, dA = sum_s dud (this program owns the full row): computed here,
    # BEFORE the in-place level walk overwrites ud/dud with subtree sums.
    A = tl.zeros((), dtype=DTYPE)
    dA = tl.zeros((), dtype=DTYPE)
    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        sm = s_offs < S
        A += tl.sum(tl.load(receiver_adjoint_ptr + row_base + s_offs, mask=sm, other=0.0))
        dA += tl.sum(tl.load(d_receiver_adjoint_ptr + row_base + s_offs, mask=sm, other=0.0))
    # All warps must finish the original row totals before any warp overwrites
    # an internal node with its subtree sum. For example, without this barrier
    # one warp could replace u[parent] by u[parent]+u[c1]+u[c2] while another
    # warp is still reducing U, causing c1 and c2 to be counted twice.
    tl.debug_barrier()
    for level in range(0, N_LEVELS):
        level_start = tl.load(level_offsets_ptr + level)
        level_end = tl.load(level_offsets_ptr + level + 1)
        p_start = level_start
        while p_start < level_end:
            node_offs = p_start + tl.arange(0, BLOCK_S)
            node_mask = node_offs < level_end
            parent = tl.load(level_parents_ptr + node_offs, mask=node_mask, other=-1)
            c1 = tl.load(level_child1_ptr + node_offs, mask=node_mask, other=S)
            c2 = tl.load(level_child2_ptr + node_offs, mask=node_mask, other=S)
            pv = node_mask & (parent >= 0) & (parent < S)
            c1_mask = node_mask & (c1 >= 0) & (c1 < S)
            c2_mask = node_mask & (c2 >= 0) & (c2 < S)
            pval = tl.load(receiver_adjoint_ptr + row_base + parent, mask=pv, other=0.0)
            c1val = tl.load(receiver_adjoint_ptr + row_base + c1, mask=c1_mask, other=0.0)
            c2val = tl.load(receiver_adjoint_ptr + row_base + c2, mask=c2_mask, other=0.0)
            tl.store(receiver_adjoint_ptr + row_base + parent, pval + c1val + c2val, mask=pv)
            dpval = tl.load(d_receiver_adjoint_ptr + row_base + parent, mask=pv, other=0.0)
            dc1 = tl.load(d_receiver_adjoint_ptr + row_base + c1, mask=c1_mask, other=0.0)
            dc2 = tl.load(d_receiver_adjoint_ptr + row_base + c2, mask=c2_mask, other=0.0)
            tl.store(d_receiver_adjoint_ptr + row_base + parent, dpval + dc1 + dc2, mask=pv)
            p_start += BLOCK_S
        tl.debug_barrier()

    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        mask = s_offs < S
        pi_val = tl.load(Pi_ptr + pi_base + s_offs, mask=mask, other=NEG)
        dpi_val = tl.load(dPi_ptr + pi_base + s_offs, mask=mask, other=0.0)
        if USE_COL_WEIGHTS:
            col_logp = tl.load(col_log_probs_ptr + s_offs, mask=mask, other=NEG)
            dcol_val = tl.load(dcol_ptr + s_offs, mask=mask, other=0.0)
            p_prime = tl.exp2(col_logp + pi_val - rm_safe)
            p_prime = tl.where(pi_val != NEG, p_prime, tl.zeros_like(p_prime))
            # col is a variable: p' = exp2(col + pi - rm), so dp' = ln2 p' (dpi + dcol) (rm frozen).
            # The contrib at :228 then carries +ln2 p' dcol into BOTH d_rhs and d_grad_col scatters.
            dp_prime = LN2 * p_prime * (dpi_val + dcol_val)
        else:
            p_prime = tl.exp2(pi_val - rm_safe)
            p_prime = tl.where(pi_val != NEG, p_prime, tl.zeros_like(p_prime))
            dp_prime = LN2 * p_prime * dpi_val
        sub = tl.load(receiver_adjoint_ptr + row_base + s_offs, mask=mask, other=0.0)
        dsub = tl.load(d_receiver_adjoint_ptr + row_base + s_offs, mask=mask, other=0.0)
        contrib = dp_prime * (A - sub) + p_prime * (dA - dsub)
        tl.atomic_add(d_rhs_ptr + pi_base + s_offs, contrib, sem="relaxed", mask=mask)
        tl.atomic_add(d_grad_col_ptr + s_offs, contrib, sem="relaxed", mask=mask)


def dts_backward_so(
    Pi, dPi, Pibar, dPibar, v, ws, meta, S,
    log_pD_param, log_pS_param, dlog_pD_param, dlog_pS_param, mc_item, dmc_item,
    col_log_probs, node_child1, node_child2, pibar_row_max, item_idx,
    d_rhs, d_grad_pD, d_grad_pS, d_grad_mt, d_grad_col,
    *, compact_level_ptr=None, compact_level_parents=None,
    compact_level_child1=None, compact_level_child2=None,
    use_col_weights=False, dcol=None, pi_offset, pibar_offset,
):
    """Accumulate the DTS second-order contraction documented in LaTeX."""
    sl, sr = meta["sl"], meta["sr"]
    N = int(sl.numel())
    dev, dt = Pi.device, Pi.dtype
    expected_rows = int(Pi.shape[0])
    pi_offset = _validate_offset_tensor(
        "pi_offset",
        pi_offset,
        rows=expected_rows,
        device=dev,
        residual_dtype=Pi.dtype,
    )
    pibar_offset = _validate_offset_tensor(
        "pibar_offset",
        pibar_offset,
        rows=expected_rows,
        device=dev,
        dtype=pi_offset.dtype,
    )
    if N == 0:
        return
    lsp = meta.get("log_split_probs")
    if lsp is None:
        lsp = torch.zeros((N,), device=Pi.device, dtype=Pi.dtype)
    else:
        # Scheduling retains split priors at accumulator precision. DTS event
        # probabilities are residual-state quantities, so convert once at this
        # mathematical boundary; preprocessing cannot choose the model dtype.
        lsp = lsp.reshape(N).to(Pi.dtype).contiguous()
    by_state = log_pD_param.ndim == 2 and int(log_pD_param.shape[1]) != 1
    row_stride = 0 if int(log_pD_param.shape[0]) == 1 else int(log_pD_param.stride(0))
    mt_row_stride = 0 if int(mc_item.shape[0]) == 1 else int(mc_item.stride(0))
    block_s = min(512, triton.next_power_of_2(S))

    # The split kernel accumulates the second-order log_pD/log_pS cotangents per SPECIES via the
    # d_grad_p*_ptr + item*S + s layout (identical to d_grad_mt). That matches a [rows, S] buffer,
    # but genewise/global pass a species-REDUCED d_grad_pD/pS ([G,1] / [1,1]) because the rate is a
    # per-family (or global) scalar and the first-order path reduces the species axis internally.
    # Writing item*S+s into a [*,1] buffer runs off the end (only s==0/1 land; the rest silently
    # corrupt adjacent pool memory). So when the caller's buffer is species-reduced (shape[1]==1),
    # hand the kernel a [rows, S] scratch (rows = d_grad_mt's family-row count -- the layout `item`
    # indexes) and sum the species axis back into the caller's buffer afterward. Specieswise ([1,S])
    # already matches the kernel layout and writes straight through, bit-for-bit unchanged.
    pd_reduced = int(d_grad_pD.shape[1]) == 1
    ps_reduced = int(d_grad_pS.shape[1]) == 1
    rows = int(d_grad_mt.shape[0])
    dgpD_k = torch.zeros((rows, S), device=dev, dtype=dt) if pd_reduced else d_grad_pD
    dgpS_k = torch.zeros((rows, S), device=dev, dtype=dt) if ps_reduced else d_grad_pS

    # stacked staging: rows [0:N) = left side, [N:2N) = right side (contiguous views, so the
    # split kernel writes them via the same n*S offsets); the tree kernel walks all 2N rows.
    ud = torch.empty((2 * N, S), device=dev, dtype=dt)
    dud = torch.empty((2 * N, S), device=dev, dtype=dt)
    ud_l, ud_r = ud[:N], ud[N:]
    dud_l, dud_r = dud[:N], dud[N:]

    _dts_split_so_kernel[(N, triton.cdiv(S, block_s))](
        Pi, dPi, Pibar, dPibar,
        pi_offset,
        pibar_offset,
        v,
        sl, sr, node_child1, node_child2,
        log_pD_param, log_pS_param, dlog_pD_param, dlog_pS_param, mc_item, dmc_item,
        lsp, meta["reduce_idx"], item_idx, int(meta["start"]), int(ws),
        pibar_row_max,
        d_rhs, ud_l, ud_r, dud_l, dud_r,
        dgpD_k, dgpS_k, d_grad_mt,
        S, BLOCK_S=block_s, ROW_STRIDE=row_stride, BY_STATE=bool(by_state),
        MT_ROW_STRIDE=mt_row_stride,
        DTYPE=_tl_float_dtype(Pi.dtype),
    )
    if pd_reduced:
        d_grad_pD += dgpD_k.sum(dim=-1, keepdim=True)
    if ps_reduced:
        d_grad_pS += dgpS_k.sum(dim=-1, keepdim=True)

    # Compact level tables pack only internal species-tree nodes by bottom-up
    # depth. They evaluate every subtree sum after its children while omitting
    # leaves, whose initial staged values are already complete subtree sums.
    if compact_level_ptr is None:
        raise ValueError("dts_backward_so requires compact_level_* tables for the tree kernel")
    n_levels = int(compact_level_ptr.numel()) - 1
    _dts_tree_so_kernel[(2 * N,)](
        Pi, dPi, col_log_probs, dcol if dcol is not None else Pi,
        ud, dud, sl, sr,
        pibar_row_max,
        compact_level_ptr.contiguous(), compact_level_parents.contiguous(),
        compact_level_child1.contiguous(), compact_level_child2.contiguous(),
        d_rhs, d_grad_col,
        n_ws=N, S=S, stride_C=int(Pi.stride(0)),
        BLOCK_S=block_s, N_LEVELS=n_levels,
        USE_COL_WEIGHTS=bool(use_col_weights), DTYPE=_tl_float_dtype(Pi.dtype),
        # num_warps=8 trims _dts_tree_so ~8% vs 4 on 666x80 (back-to-back wall 997->989ms;
        # nsys kernel 12%->11% of HVP). Each program owns a full side-row walked in BLOCK_S
        # chunks -> more warps hide the dependent level-walk loads. split kernel unaffected (kept
        # at Triton's default). Bit-identical (dts_so/hvp gates unchanged).
        num_warps=8,
    )
