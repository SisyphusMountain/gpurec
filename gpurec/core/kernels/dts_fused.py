import torch
import triton
import triton.language as tl

@triton.jit
def _single_split_dts_parent_rows_kernel(
    Pi_ptr,
    Pibar_ptr,
    lefts_ptr,
    rights_ptr,
    sp_child1_ptr,
    sp_child2_ptr,
    log_pD_ptr,
    log_pS_ptr,
    parent_ids_ptr,
    active_parent_rows_ptr,
    out_ptr,
    family_idx_ptr,
    family_offset,
    S: tl.constexpr,
    BLOCK_S: tl.constexpr,
    ROW_STRIDE: tl.constexpr = 0,
    USE_ACTIVE_PARENT_ROWS: tl.constexpr = False,
):
    n = tl.program_id(0)
    s_block = tl.program_id(1)
    s_offs = s_block * BLOCK_S + tl.arange(0, BLOCK_S)
    mask = s_offs < S

    parent_w = tl.load(parent_ids_ptr + n).to(tl.int64)
    out_base = parent_w * S
    family = tl.load(family_idx_ptr + family_offset + parent_w).to(tl.int64)
    if USE_ACTIVE_PARENT_ROWS:
        parent_active = tl.load(active_parent_rows_ptr + parent_w)
        if parent_active == 0:
            tl.store(out_ptr + out_base + s_offs, tl.full([BLOCK_S], value=-1e30, dtype=tl.float32), mask=mask)
            return

    left_idx = tl.load(lefts_ptr + n).to(tl.int64)
    right_idx = tl.load(rights_ptr + n).to(tl.int64)
    base_l = left_idx * S
    base_r = right_idx * S

    pi_l = tl.load(Pi_ptr + base_l + s_offs, mask=mask, other=-1e30)
    pi_r = tl.load(Pi_ptr + base_r + s_offs, mask=mask, other=-1e30)
    pibar_l = tl.load(Pibar_ptr + base_l + s_offs, mask=mask, other=-1e30)
    pibar_r = tl.load(Pibar_ptr + base_r + s_offs, mask=mask, other=-1e30)

    log_pD_s = tl.load(log_pD_ptr + family * ROW_STRIDE)
    log_pS_s = tl.load(log_pS_ptr + family * ROW_STRIDE)

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
    result = tl.log2(s) + m
    tl.store(out_ptr + out_base + s_offs, result, mask=mask)


def compute_dts_forward(
    Pi,
    Pibar,
    lefts,
    rights,
    sp_child1,
    sp_child2,
    W,
    reduce_idx,
    log_pD_vec,
    log_pS_vec,
    family_idx,
    active_parent_rows=None,
    family_offset=0,
):
    N = lefts.shape[0]
    S = Pi.shape[1]
    needs_fill = int(N) != int(W)
    out = torch.full((W, S), float("-inf"), device=Pi.device, dtype=Pi.dtype) if needs_fill else torch.empty((W, S), device=Pi.device, dtype=Pi.dtype)

    row_stride = 0 if int(log_pD_vec.shape[0]) == 1 else int(log_pD_vec.stride(0))
    block_s = min(512, triton.next_power_of_2(S))
    _single_split_dts_parent_rows_kernel[(N, triton.cdiv(S, block_s))](
        Pi,
        Pibar,
        lefts,
        rights,
        sp_child1,
        sp_child2,
        log_pD_vec,
        log_pS_vec,
        reduce_idx,
        active_parent_rows if active_parent_rows is not None else reduce_idx,
        out,
        family_idx,
        int(family_offset),
        S,
        BLOCK_S=block_s,
        ROW_STRIDE=row_stride,
        USE_ACTIVE_PARENT_ROWS=bool(active_parent_rows is not None),
    )
    return out
