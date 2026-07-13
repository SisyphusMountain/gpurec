"""DTS tangent kernels; see ``docs/latex/kernel_mathematics.tex``."""

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
def _dts_tangent_kernel(
    Pi, Pibar, dPi, dPibar, split_left_rows, split_right_rows,
    species_child1, species_child2,
    log_pD, log_pS, dlog_pD, dlog_pS, log_split_probs, reduce_idx,
    dts_r_ptr, d_out_ptr, item_idx, item_offset,
    Pi_offset, Pibar_offset, dts_offset,
    S: tl.constexpr, BLOCK_S: tl.constexpr, ROW_STRIDE: tl.constexpr,
    BY_STATE: tl.constexpr, DTYPE: tl.constexpr,
):
    """Evaluate the DTS JVP defined in the kernel mathematics reference."""
    NEG_INF = -float("inf")
    n = tl.program_id(0)
    s_block = tl.program_id(1)
    s_offs = s_block * BLOCK_S + tl.arange(0, BLOCK_S)
    mask = s_offs < S
    zero = tl.zeros([BLOCK_S], dtype=DTYPE)
    # Split metadata is stored as int32 to reduce bandwidth. Widen only values
    # used in flattened addresses, whose products may exceed the int32 range.
    parent_w = tl.load(reduce_idx + n).to(tl.int64)
    item = tl.load(item_idx + item_offset + parent_w).to(tl.int64)
    left = tl.load(split_left_rows + n).to(tl.int64)
    right = tl.load(split_right_rows + n).to(tl.int64)
    base_l = left * S
    base_r = right * S

    pi_l = tl.load(Pi + base_l + s_offs, mask=mask, other=NEG_INF)
    pi_r = tl.load(Pi + base_r + s_offs, mask=mask, other=NEG_INF)
    pibar_l = tl.load(Pibar + base_l + s_offs, mask=mask, other=NEG_INF)
    pibar_r = tl.load(Pibar + base_r + s_offs, mask=mask, other=NEG_INF)
    dpi_l = tl.load(dPi + base_l + s_offs, mask=mask, other=0.0)
    dpi_r = tl.load(dPi + base_r + s_offs, mask=mask, other=0.0)
    dpibar_l = tl.load(dPibar + base_l + s_offs, mask=mask, other=0.0)
    dpibar_r = tl.load(dPibar + base_r + s_offs, mask=mask, other=0.0)

    log_d = _load_rate(log_pD, item, s_offs, mask, S, ROW_STRIDE, BY_STATE, BLOCK_S, DTYPE)
    log_s = _load_rate(log_pS, item, s_offs, mask, S, ROW_STRIDE, BY_STATE, BLOCK_S, DTYPE)
    dlog_d = _load_rate(dlog_pD, item, s_offs, mask, S, ROW_STRIDE, BY_STATE, BLOCK_S, DTYPE)
    dlog_s = _load_rate(dlog_pS, item, s_offs, mask, S, ROW_STRIDE, BY_STATE, BLOCK_S, DTYPE)

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

    # The DTS normalizer is a residual in dts_offset[parent_w]'s frame. Align
    # each child combination to that frame before forming its softmax weight.
    # Offsets have no tangent: dpi/dpibar differentiate the represented rows.
    out_offset = tl.load(dts_offset + parent_w)
    pi_off_l = tl.load(Pi_offset + left)
    pi_off_r = tl.load(Pi_offset + right)
    pibar_off_l = tl.load(Pibar_offset + left)
    pibar_off_r = tl.load(Pibar_offset + right)
    child_frame_shift = (pi_off_l + pi_off_r - out_offset).to(DTYPE)
    left_transfer_frame_shift = (pi_off_l + pibar_off_r - out_offset).to(DTYPE)
    right_transfer_frame_shift = (pi_off_r + pibar_off_l - out_offset).to(DTYPE)
    duplication_log_weight = lsp + log_d + pi_l + pi_r + child_frame_shift
    left_transfer_log_weight = lsp + pi_l + pibar_r + left_transfer_frame_shift
    right_transfer_log_weight = lsp + pi_r + pibar_l + right_transfer_frame_shift
    speciation_lr_log_weight = lsp + log_s + pi_l_c1 + pi_r_c2 + child_frame_shift
    speciation_rl_log_weight = lsp + log_s + pi_r_c1 + pi_l_c2 + child_frame_shift
    d_duplication = dlog_d + dpi_l + dpi_r
    d_left_transfer = dpi_l + dpibar_r
    d_right_transfer = dpi_r + dpibar_l
    d_speciation_lr = dlog_s + dpi_l_c1 + dpi_r_c2
    d_speciation_rl = dlog_s + dpi_r_c1 + dpi_l_c2

    dts_out = tl.load(dts_r_ptr + parent_w * S + s_offs, mask=mask, other=NEG_INF)
    active = mask & (dts_out != NEG_INF)
    duplication_probability = tl.exp2(duplication_log_weight - dts_out)
    left_transfer_probability = tl.exp2(left_transfer_log_weight - dts_out)
    right_transfer_probability = tl.exp2(right_transfer_log_weight - dts_out)
    speciation_lr_probability = tl.exp2(speciation_lr_log_weight - dts_out)
    speciation_rl_probability = tl.exp2(speciation_rl_log_weight - dts_out)
    contrib = (
        duplication_probability * d_duplication
        + left_transfer_probability * d_left_transfer
        + right_transfer_probability * d_right_transfer
        + speciation_lr_probability * d_speciation_lr
        + speciation_rl_probability * d_speciation_rl
    )
    contrib = tl.where(active, contrib, zero)
    tl.atomic_add(d_out_ptr + parent_w * S + s_offs, contrib, sem="relaxed", mask=mask)


def compute_dts_tangent(
    Pi, Pibar, dPi, dPibar, split_left_rows, split_right_rows,
    species_child1, species_child2, W, reduce_idx,
    log_pD_vec, log_pS_vec, dlog_pD_vec, dlog_pS_vec, dts_r, item_idx,
    *, log_split_probs=None, item_offset=0,
    pi_offset, pibar_offset, dts_offset,
):
    """Return the DTS JVP; see ``docs/latex/kernel_mathematics.tex``."""
    N = int(split_left_rows.shape[0])
    S = int(Pi.shape[1])
    d_out = torch.zeros((W, S), device=Pi.device, dtype=Pi.dtype)
    C = int(Pi.shape[0])
    pi_offset = _validate_offset_tensor(
        "pi_offset",
        pi_offset,
        rows=C,
        device=Pi.device,
        residual_dtype=Pi.dtype,
    )
    accumulator_dtype = pi_offset.dtype
    pibar_offset = _validate_offset_tensor(
        "pibar_offset",
        pibar_offset,
        rows=C,
        device=Pi.device,
        dtype=accumulator_dtype,
    )
    dts_offset = _validate_offset_tensor(
        "dts_offset",
        dts_offset,
        rows=W,
        device=Pi.device,
        dtype=accumulator_dtype,
    )
    if N == 0:
        return d_out
    if log_split_probs is None:
        log_split_probs = torch.zeros((N,), device=Pi.device, dtype=Pi.dtype)
    else:
        # Batch static -> compute dtype at the canonical forward boundary.
        log_split_probs = log_split_probs.reshape(N).to(Pi.dtype).contiguous()
    by_state = log_pD_vec.ndim == 2 and int(log_pD_vec.shape[1]) != 1
    row_stride = 0 if int(log_pD_vec.shape[0]) == 1 else int(log_pD_vec.stride(0))
    block_s = min(512, triton.next_power_of_2(S))
    _dts_tangent_kernel[(N, triton.cdiv(S, block_s))](
        Pi, Pibar, dPi, dPibar, split_left_rows, split_right_rows,
        species_child1, species_child2,
        log_pD_vec, log_pS_vec, dlog_pD_vec, dlog_pS_vec, log_split_probs, reduce_idx,
        dts_r, d_out, item_idx, int(item_offset),
        pi_offset, pibar_offset, dts_offset,
        S, BLOCK_S=block_s, ROW_STRIDE=row_stride, BY_STATE=bool(by_state),
        DTYPE=_tl_float_dtype(Pi.dtype),
    )
    return d_out
