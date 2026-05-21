"""Fused Triton kernels for the retained wave-backward fast path."""

import os

import torch
import triton
import triton.language as tl

from gpurec.core._helpers import _env_flag_enabled, _env_mode_enabled_required
from gpurec.core.memory_policy import proposal0_memory_gate

_cuda_pibar_from_ud_fallback_warned = False
_SUPPORTED_FLOAT_DTYPES = (torch.float32, torch.float64, torch.bfloat16)


def _tl_float_dtype(dtype):
    return tl.float64 if dtype == torch.float64 else tl.float32


def _cuda_pibar_from_ud_options():
    """Return CUDA Pibar prototype mode, enablement, and selected-path strictness.

    ``auto`` is silent best-effort, ``enabled`` is best-effort with the caller's
    warning-on-fallback path, and ``GPUREC_CUDA_PIBAR_FROM_UD_STRICT`` makes an
    otherwise best-effort selected Pibar prototype re-raise failures.
    """
    mode, enabled, required = _env_mode_enabled_required(
        "GPUREC_CUDA_PIBAR_FROM_UD",
        "auto",
    )
    strict_required = _env_flag_enabled("GPUREC_CUDA_PIBAR_FROM_UD_STRICT", "0")
    return mode, enabled, enabled and (required or strict_required)


def _device_scalar_param(param, *, device, dtype):
    """Return a one-element device tensor without extracting CUDA scalars."""
    if torch.is_tensor(param):
        if param.numel() != 1:
            raise ValueError("fused DTS backward scalar parameters must have one element")
        if param.device != device or param.dtype != dtype:
            param = param.to(device=device, dtype=dtype)
        return param.reshape(1).contiguous()
    return torch.tensor([param], device=device, dtype=dtype)


def _dts_layout_param_args(log_pD, log_pS, *, family_idx, S, device, dtype):
    """Return DTS parameter tensors plus a Triton addressing layout.

    Layouts:
      0: shared scalar, tensor [1]
      1: shared species vector, tensor [S]
      2: family scalar, tensor [G] addressed by family_idx[parent]
      3: family species, tensor [G, S] addressed by family_idx[parent], s
    """

    def _normalize(param):
        if not torch.is_tensor(param):
            return _device_scalar_param(param, device=device, dtype=dtype), 0
        if param.device != device or param.dtype != dtype:
            param = param.to(device=device, dtype=dtype)
        if param.numel() == 1:
            return param.reshape(1).contiguous(), 0
        if family_idx is not None and param.ndim == 1:
            return param.contiguous(), 2
        if param.ndim == 1 and int(param.shape[0]) == int(S):
            return param.contiguous(), 1
        if family_idx is not None:
            if param.ndim == 1:
                return param.contiguous(), 2
            if param.ndim == 2 and int(param.shape[1]) == 1:
                return param.reshape(int(param.shape[0])).contiguous(), 2
            if param.ndim == 2 and int(param.shape[1]) == int(S):
                return param.contiguous(), 3
        raise ValueError(
            "DTS parameters must be scalar, [S], [G], or [G, S] for "
            "the fused DTS backward path"
        )

    pD, layout_D = _normalize(log_pD)
    pS, layout_S = _normalize(log_pS)
    if layout_D != layout_S:
        raise ValueError("log_pD/log_pS must use the same DTS parameter layout")
    return pD, pS, layout_D


def _dts_grad_layout(grad, *, family_idx, S):
    """Return gradient addressing layout matching _dts_layout_param_args."""
    if grad.numel() == 1:
        return 0
    if family_idx is not None and grad.ndim == 1:
        return 2
    if grad.ndim == 1 and int(grad.shape[0]) == int(S):
        return 1
    if family_idx is not None:
        if grad.ndim == 1:
            return 2
        if grad.ndim == 2 and int(grad.shape[1]) == 1:
            return 2
        if grad.ndim == 2 and int(grad.shape[1]) == int(S):
            return 3
    raise ValueError("unsupported DTS gradient layout")


def _uniform_backward_const_layout(const_tensor, family_idx, family_indexed):
    """Return addressing mode for self-loop constants.

    Modes:
      0: shared [S]
      1: row-expanded [W, S]
      2: family-indexed [G, S] addressed through family_idx[C]
    """
    if family_indexed:
        if family_idx is None:
            raise ValueError("family-indexed backward constants require family_idx")
        if const_tensor.ndim != 2:
            raise ValueError("family-indexed backward constants require [G, S] tensors")
        return 2
    if const_tensor.ndim == 2:
        return 1
    return 0


def _uniform_backward_leaf_logp_mode(use_leaf_index, leaf_logp, family_idx, family_indexed):
    """Return addressing mode for leaf log-probabilities in the self-loop."""
    if not use_leaf_index:
        return 0
    if family_indexed:
        if family_idx is None:
            raise ValueError("family-indexed leaf log-probabilities require family_idx")
        if leaf_logp.ndim == 1:
            return 1
        if leaf_logp.ndim == 2:
            if int(leaf_logp.shape[1]) == 1:
                raise ValueError("family-indexed [G, 1] leaf_logp should be expanded to [G, S]")
            return 2
        raise ValueError("family-indexed leaf_logp must have shape [G] or [G, S]")
    if leaf_logp.numel() == 1:
        return 3
    return 0


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


def active_mask_from_rhs_absmax_fused(rhs, threshold, *, use_pruning=True):
    """Build the row activity mask for backward pruning in one Triton launch."""
    if rhs.ndim != 2:
        raise ValueError("rhs must be a 2D tensor")
    if rhs.device.type != "cuda":
        raise ValueError("active_mask_from_rhs_absmax_fused requires a CUDA tensor")
    if rhs.dtype not in _SUPPORTED_FLOAT_DTYPES:
        raise ValueError(
            "active_mask_from_rhs_absmax_fused supports fp32/fp64/bf16 tensors"
        )

    W, S = rhs.shape
    active_mask = torch.empty((W,), device=rhs.device, dtype=torch.bool)
    if W == 0:
        return active_mask

    BLOCK_S = min(256, triton.next_power_of_2(S))
    _active_mask_from_rhs_absmax_kernel[(W,)](
        rhs,
        active_mask,
        float(threshold),
        S,
        rhs.stride(0),
        BLOCK_S,
        STRICT_GT=bool(not use_pruning),
        DTYPE=_tl_float_dtype(rhs.dtype),
    )
    return active_mask

@triton.jit
def _wave_backward_uniform_2d_precompute_kernel(
    Pi_star_ptr,
    Pibar_star_ptr,
    Pibar_row_max_ptr,
    dts_r_ptr,
    has_splits: tl.constexpr,
    rhs_ptr,
    active_mask_ptr,
    mt_ptr, DL_const_ptr, Ebar_ptr, E_ptr, SL1_const_ptr, SL2_const_ptr,
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
):
    """Precompute self-loop J^T coefficients for a block of rows and all species."""
    NEG_LARGE: tl.constexpr = -1e30
    M_SAFE: tl.constexpr = -1e29

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

    row_max = tl.load(Pibar_row_max_ptr + row_global, mask=row_valid, other=NEG_LARGE).to(DTYPE)
    pi_w = tl.load(Pi_star_ptr + pi_offsets, mask=mask, other=NEG_LARGE).to(DTYPE)
    pibar_w = tl.load(Pibar_star_ptr + pi_offsets, mask=mask, other=NEG_LARGE).to(DTYPE)
    p_prime = tl.exp2(pi_w - row_max[None, :])
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
    t2 = pibar_w + e_val
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

    m = tl.maximum(t0, t1)
    m = tl.maximum(m, t2)
    m = tl.maximum(m, t3)
    m = tl.maximum(m, t4)
    m = tl.maximum(m, t5)
    m_safe = tl.where(m > M_SAFE, m, tl.zeros([BLOCK_S, BLOCK_W], dtype=DTYPE))
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
        dts_l = tl.log2(dts_l_sum) + m
        pi_new_m = tl.maximum(dts_l, dts_r)
        pi_new_ms = tl.where(pi_new_m > M_SAFE, pi_new_m, tl.zeros_like(pi_new_m))
        pi_new = tl.log2(tl.exp2(dts_l - pi_new_ms) + tl.exp2(dts_r - pi_new_ms)) + pi_new_m
        w_L = tl.where(dts_l > M_SAFE, tl.exp2(dts_l - pi_new), tl.zeros_like(dts_l))
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
        ancestor_sum += tl.where(
            cur_valid[:, None] & row_mask[None, :],
            tl.exp2(pi_anc - row_max[None, :]),
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
    tl.store(sl1_ptr + out_offsets, tl.where(mask, sl1_wt, zero), mask=store_mask)
    tl.store(sl2_ptr + out_offsets, tl.where(mask, sl2_wt, zero), mask=store_mask)


@triton.jit
def _wave_backward_uniform_2d_jt_kernel(
    term_in_ptr,
    term_out_ptr,
    active_mask_ptr,
    diag_ptr,
    pibar_coeff_ptr,
    p_prime_ptr,
    sl1_ptr,
    sl2_ptr,
    sp_child1_ptr,
    sp_child2_ptr,
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
    DTYPE: tl.constexpr,
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
    v_prev = tl.load(v_k_ptr + offsets, mask=mask, other=0.0).to(DTYPE)
    tl.store(v_k_ptr + offsets, v_prev + result, mask=mask)


@triton.jit
def _wave_backward_uniform_param_store_kernel(
    Pi_star_ptr,
    Pibar_star_ptr,
    dts_r_ptr,
    has_splits: tl.constexpr,
    v_k_ptr,
    active_mask_ptr,
    mt_ptr, DL_const_ptr, Ebar_ptr, E_ptr, SL1_const_ptr, SL2_const_ptr,
    sp_child1_ptr, sp_child2_ptr,
    leaf_term_ptr,
    leaf_species_ptr,
    leaf_logp_ptr,
    family_idx_ptr,
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
    DTYPE: tl.constexpr,
):
    """Store per-element self-loop parameter VJP contributions after Neumann."""
    NEG_LARGE: tl.constexpr = -1e30
    M_SAFE: tl.constexpr = -1e29

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
    t2 = pibar_w + e_val
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

    m = tl.maximum(t0, t1)
    m = tl.maximum(m, t2)
    m = tl.maximum(m, t3)
    m = tl.maximum(m, t4)
    m = tl.maximum(m, t5)
    m_safe = tl.where(m > M_SAFE, m, tl.zeros([BLOCK_S, BLOCK_W], dtype=DTYPE))
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
        dts_l = tl.log2(dts_l_sum) + m
        pi_new_m = tl.maximum(dts_l, dts_r)
        pi_new_ms = tl.where(pi_new_m > M_SAFE, pi_new_m, tl.zeros_like(pi_new_m))
        pi_new = tl.log2(tl.exp2(dts_l - pi_new_ms) + tl.exp2(dts_r - pi_new_ms)) + pi_new_m
        w_L = tl.where(dts_l > M_SAFE, tl.exp2(dts_l - pi_new), tl.zeros_like(dts_l))
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
    tl.store(aw0_ptr + out_offsets, tl.where(mask, _aw0, zero), mask=store_mask)
    tl.store(aw1_ptr + out_offsets, tl.where(mask, _aw1, zero), mask=store_mask)
    tl.store(aw2_ptr + out_offsets, tl.where(mask, _aw2, zero), mask=store_mask)
    tl.store(aw345_ptr + out_offsets, tl.where(mask, _aw345, zero), mask=store_mask)
    tl.store(aw3_ptr + out_offsets, tl.where(mask, _aw3, zero), mask=store_mask)
    tl.store(aw4_ptr + out_offsets, tl.where(mask, _aw4, zero), mask=store_mask)


def _wave_backward_uniform_2d(
    Pi_star, Pibar_star, ws, W, S,
    dts_r,
    rhs,
    mt_squeezed, DL_const, Ebar, E, SL1_const, SL2_const,
    sp_child1, sp_child2, leaf_term_wt,
    *,
    neumann_terms,
    leaf_species_idx,
    leaf_logp,
    has_leaf_term,
    active_mask,
    sp_parent,
    max_ancestor_depth,
    pibar_row_max,
    family_idx,
    const_layout,
    leaf_logp_mode,
    use_leaf_index,
    compact_level_ptr,
    compact_level_parents,
    compact_level_child1,
    compact_level_child2,
):
    """Retained 2D row-block/full-species tree-reduction self-loop."""
    if Pi_star.device.type != "cuda":
        raise RuntimeError("GPUREC self-loop 2D fast path requires CUDA tensors")
    if Pi_star.dtype not in (torch.float32, torch.float64):
        raise RuntimeError(
            "2D self-loop fast path currently supports fp32/fp64 only",
        )
    ok, required_bytes, budget_bytes = proposal0_memory_gate(
        W,
        S,
        Pi_star.dtype,
        device=Pi_star.device,
    )
    if not ok:
        raise RuntimeError(
            "2D self-loop fast path estimated scratch "
            f"{required_bytes / (1024 ** 3):.2f} GiB above memory budget "
            f"{(budget_bytes or 0) / (1024 ** 3):.2f} GiB",
        )
    if const_layout not in (0, 1, 2):
        raise RuntimeError("unsupported self-loop constant layout")
    if use_leaf_index and leaf_logp_mode not in (0, 1, 2, 3):
        raise RuntimeError("unsupported leaf log-probability layout")

    device = Pi_star.device
    dtype = Pi_star.dtype
    if sp_parent is None:
        raise ValueError("sp_parent is required for the retained 2D self-loop path")
    sp_parent = sp_parent.to(device=device, dtype=torch.int32).contiguous()
    if max_ancestor_depth is None:
        raise ValueError("max_ancestor_depth is required for the retained 2D self-loop path")
    max_ancestor_depth = max(1, int(max_ancestor_depth))

    if (
        compact_level_ptr is None
        or compact_level_parents is None
        or compact_level_child1 is None
        or compact_level_child2 is None
    ):
        raise ValueError("compact species levels are required for the retained 2D self-loop path")
    compact_level_ptr = compact_level_ptr.to(device=device, dtype=torch.long).contiguous()
    compact_level_parents = compact_level_parents.to(device=device, dtype=torch.int32).contiguous()
    compact_level_child1 = compact_level_child1.to(device=device, dtype=torch.int32).contiguous()
    compact_level_child2 = compact_level_child2.to(device=device, dtype=torch.int32).contiguous()

    block_w = triton.next_power_of_2(
        max(1, int(os.environ.get("GPUREC_SELF_LOOP_2D_BLOCK_W", "1")))
    )
    block_s = triton.next_power_of_2(S)
    block_nodes = triton.next_power_of_2(
        max(1, int(os.environ.get("GPUREC_SELF_LOOP_2D_BLOCK_NODES", "128")))
    )
    n_row_blocks = triton.cdiv(W, block_w)
    scratch_shape = (W, S)

    v_k = torch.empty(scratch_shape, device=device, dtype=dtype)
    aw0 = torch.empty(scratch_shape, device=device, dtype=dtype)
    aw1 = torch.empty(scratch_shape, device=device, dtype=dtype)
    aw2 = torch.empty(scratch_shape, device=device, dtype=dtype)
    aw345 = torch.empty(scratch_shape, device=device, dtype=dtype)
    aw3 = torch.empty(scratch_shape, device=device, dtype=dtype)
    aw4 = torch.empty(scratch_shape, device=device, dtype=dtype)
    spec_buf = torch.empty(scratch_shape, device=device, dtype=dtype)
    term_buf = torch.empty(scratch_shape, device=device, dtype=dtype)
    pibar_corr = torch.empty(scratch_shape, device=device, dtype=dtype)

    if pibar_row_max is None:
        raise ValueError("pibar_row_max is required for the retained 2D self-loop path")
    pibar_row_max = pibar_row_max.to(device=device, dtype=dtype).contiguous()
    skip_inactive_scratch_zero = (
        os.environ.get(
            "GPUREC_SELF_LOOP_2D_SKIP_INACTIVE_SCRATCH_ZERO",
            "1",
        ).strip().lower()
        not in ("", "0", "false", "no", "off")
    )
    if family_idx is not None:
        family_idx = family_idx.to(device=device, dtype=torch.long).contiguous()
    else:
        family_idx = sp_parent
    requested_has_leaf_term = bool(has_leaf_term)
    use_leaf_index = bool(use_leaf_index and requested_has_leaf_term)
    has_materialized_leaf_term = leaf_term_wt is not None
    if leaf_term_wt is None:
        leaf_term_wt = leaf_logp if use_leaf_index else Pi_star
    has_leaf_term = bool(
        requested_has_leaf_term
        and (use_leaf_index or has_materialized_leaf_term)
    )
    leaf_species_arg = leaf_species_idx if use_leaf_index else sp_child1
    leaf_logp_arg = leaf_logp if use_leaf_index else leaf_term_wt

    precompute_warps = int(os.environ.get("GPUREC_SELF_LOOP_2D_NUM_WARPS", "8"))
    launch_options = {}
    if precompute_warps > 0:
        launch_options["num_warps"] = precompute_warps

    _wave_backward_uniform_2d_precompute_kernel[(n_row_blocks,)](
        Pi_star,
        Pibar_star,
        pibar_row_max,
        dts_r if dts_r is not None else Pi_star,
        dts_r is not None,
        rhs,
        active_mask if active_mask is not None else rhs,
        mt_squeezed,
        DL_const,
        Ebar,
        E,
        SL1_const,
        SL2_const,
        sp_child1,
        sp_child2,
        sp_parent,
        leaf_term_wt,
        leaf_species_arg,
        leaf_logp_arg,
        family_idx,
        v_k,
        aw0,
        aw1,
        aw2,
        aw3,
        aw4,
        ws,
        W,
        S,
        Pi_star.stride(0),
        block_w,
        block_s,
        max_ancestor_depth,
        USE_LEAF_INDEX=bool(use_leaf_index),
        HAS_LEAF_TERM=bool(has_leaf_term),
        LEAF_LOGP_MODE=int(leaf_logp_mode),
        USE_ACTIVE_MASK=bool(active_mask is not None),
        SKIP_INACTIVE_SCRATCH_ZERO=bool(skip_inactive_scratch_zero),
        CONST_LAYOUT=int(const_layout),
        DTYPE=_tl_float_dtype(dtype),
        **launch_options,
    )

    jt_warps = int(os.environ.get("GPUREC_SELF_LOOP_2D_JT_NUM_WARPS", "2"))
    jt_options = {}
    if jt_warps > 0:
        jt_options["num_warps"] = jt_warps
    for n in range(int(neumann_terms)):
        term_in = rhs if n == 0 else (spec_buf if n % 2 == 1 else term_buf)
        term_out = spec_buf if n % 2 == 0 else term_buf
        _wave_backward_uniform_2d_jt_kernel[(n_row_blocks,)](
            term_in,
            term_out,
            active_mask if active_mask is not None else rhs,
            aw0,
            aw1,
            aw2,
            aw3,
            aw4,
            sp_child1,
            sp_child2,
            compact_level_ptr,
            compact_level_parents,
            compact_level_child1,
            compact_level_child2,
            pibar_corr,
            v_k,
            W,
            S,
            block_w,
            block_s,
            block_nodes,
            compact_level_ptr.numel() - 1,
            USE_ACTIVE_MASK=bool(active_mask is not None),
            SKIP_INACTIVE_SCRATCH_ZERO=bool(skip_inactive_scratch_zero),
            DTYPE=_tl_float_dtype(dtype),
            **jt_options,
        )

    _wave_backward_uniform_param_store_kernel[(n_row_blocks,)](
        Pi_star,
        Pibar_star,
        dts_r if dts_r is not None else Pi_star,
        dts_r is not None,
        v_k,
        active_mask if active_mask is not None else rhs,
        mt_squeezed,
        DL_const,
        Ebar,
        E,
        SL1_const,
        SL2_const,
        sp_child1,
        sp_child2,
        leaf_term_wt,
        leaf_species_arg,
        leaf_logp_arg,
        family_idx,
        aw0,
        aw1,
        aw2,
        aw345,
        aw3,
        aw4,
        ws,
        W,
        S,
        Pi_star.stride(0),
        block_w,
        block_s,
        USE_LEAF_INDEX=bool(use_leaf_index),
        HAS_LEAF_TERM=bool(has_leaf_term),
        LEAF_LOGP_MODE=int(leaf_logp_mode),
        USE_ACTIVE_MASK=bool(active_mask is not None),
        CONST_LAYOUT=int(const_layout),
        DTYPE=_tl_float_dtype(dtype),
        **launch_options,
    )

    return v_k, aw0, aw1, aw2, aw345, aw3, aw4


def wave_backward_uniform_fused(
    Pi_star, Pibar_star, ws, W, S,
    dts_r,
    rhs,
    mt_squeezed, DL_const, Ebar, E, SL1_const, SL2_const,
    sp_child1, sp_child2, leaf_term_wt,
    neumann_terms=3,
    leaf_species_idx=None,
    leaf_logp=None,
    has_leaf_term=True,
    active_mask=None,
    sp_parent=None,
    max_ancestor_depth=None,
    pibar_row_max=None,
    family_idx=None,
    family_indexed_consts=False,
    compact_level_ptr=None,
    compact_level_parents=None,
    compact_level_child1=None,
    compact_level_child2=None,
):
    """Fused backward: precompute + Neumann + param VJP in one kernel per wave.

    Args:
        Pi_star: [C, S] converged Pi
        Pibar_star: [C, S] converged Pibar
        ws: wave start offset
        W: wave size
        S: number of species
        dts_r: [W, S] or None
        rhs: [W, S] incoming adjoint
        mt_squeezed, DL_const, Ebar, E, SL1_const, SL2_const:
            [S], [W, S], or [G, S] when family_indexed_consts=True
        sp_child1, sp_child2: [S] long
        leaf_term_wt: [W, S]
        neumann_terms: int
        leaf_species_idx: optional [C] row -> species leaf index, -1 for non-leaves
        leaf_logp: optional [S], [G], or [G, S] log_pS values used with
            leaf_species_idx

    Returns:
        v_k: [W, S] Neumann-solved adjoint
        aw0, aw1, aw2, aw345, aw3, aw4: [W, S] per-element param grad contributions
    """
    requested_has_leaf_term = bool(has_leaf_term)
    use_leaf_index = (
        requested_has_leaf_term
        and leaf_species_idx is not None
        and leaf_logp is not None
    )
    const_layout = _uniform_backward_const_layout(
        DL_const, family_idx, bool(family_indexed_consts)
    )
    if bool(family_indexed_consts) and use_leaf_index:
        if leaf_logp.ndim == 1:
            leaf_logp = leaf_logp.unsqueeze(-1).expand(-1, S).contiguous()
        elif leaf_logp.ndim == 2 and int(leaf_logp.shape[1]) == 1:
            leaf_logp = leaf_logp.expand(-1, S).contiguous()
        else:
            leaf_logp = leaf_logp.contiguous()
    leaf_logp_mode = _uniform_backward_leaf_logp_mode(
        use_leaf_index, leaf_logp, family_idx, bool(family_indexed_consts)
    )
    if family_idx is not None:
        family_idx = family_idx.to(device=Pi_star.device, dtype=torch.long).contiguous()
    if sp_parent is None:
        raise ValueError("sp_parent is required for the retained backward fast path")
    sp_parent = sp_parent.to(device=Pi_star.device).contiguous()
    if Pi_star.device.type == "cuda" and sp_parent.dtype != torch.int32:
        sp_parent = sp_parent.to(dtype=torch.int32)
    if max_ancestor_depth is None:
        raise ValueError("max_ancestor_depth is required for the retained backward fast path")
    max_ancestor_depth = max(1, int(max_ancestor_depth))
    if pibar_row_max is None:
        raise ValueError("pibar_row_max is required for the retained backward fast path")
    pibar_row_max = pibar_row_max.to(device=Pi_star.device, dtype=Pi_star.dtype).contiguous()

    return _wave_backward_uniform_2d(
        Pi_star,
        Pibar_star,
        ws,
        W,
        S,
        dts_r,
        rhs,
        mt_squeezed,
        DL_const,
        Ebar,
        E,
        SL1_const,
        SL2_const,
        sp_child1,
        sp_child2,
        leaf_term_wt,
        neumann_terms=neumann_terms,
        leaf_species_idx=leaf_species_idx,
        leaf_logp=leaf_logp,
        has_leaf_term=requested_has_leaf_term,
        active_mask=active_mask,
        sp_parent=sp_parent,
        max_ancestor_depth=max_ancestor_depth,
        pibar_row_max=pibar_row_max,
        family_idx=family_idx,
        const_layout=const_layout,
        leaf_logp_mode=leaf_logp_mode,
        use_leaf_index=use_leaf_index,
        compact_level_ptr=compact_level_ptr,
        compact_level_parents=compact_level_parents,
        compact_level_child1=compact_level_child1,
        compact_level_child2=compact_level_child2,
    )


# =========================================================================
# Cross-clade DTS backward kernel
# =========================================================================

@triton.jit
def _dts_cross_backward_accum_kernel(
    # Converged values [C, S]
    Pi_star_ptr,
    Pibar_star_ptr,
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
    NEG_LARGE: tl.constexpr = -1e30

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

        d0 = log_pD_s + Pi_l + Pi_r
        d1 = Pi_l + Pibar_r
        d2 = Pi_r + Pibar_l
        d3 = log_pS_s + Pi_l_s1 + Pi_r_s2
        d4 = log_pS_s + Pi_r_s1 + Pi_l_s2

        parent_valid = Pi_parent > NEG_LARGE
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
            finite_l = (Pibar_l > -1e29) & mask
            finite_r = (Pibar_r > -1e29) & mask
            inv_denom_l = tl.where(
                finite_l,
                tl.exp2(row_max_l + mt_l - Pibar_l),
                tl.zeros([BLOCK_S], dtype=DTYPE),
            )
            inv_denom_r = tl.where(
                finite_r,
                tl.exp2(row_max_r + mt_r - Pibar_r),
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

            d3 = log_pS_s + Pi_l_s1 + Pi_r_s2
            d4 = log_pS_s + Pi_r_s1 + Pi_l_s2

            parent_valid = Pi_parent > NEG_LARGE
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


def dts_cross_backward_accum_fused(
    Pi_star, Pibar_star, v_k, ws,
    sl, sr, reduce_idx, wlsp,
    log_pD, log_pS,
    sp_child1, sp_child2,
    accumulated_rhs,
    S,
    active_mask=None,
    use_atomics=True,
    merge_s_term=False,
    grad_log_pD=None,
    grad_log_pS=None,
    grad_mt=None,
    accum_param_reductions=False,
    accum_mt_reduction=False,
    output_pibar_ud=False,
    output_pibar_side_active=False,
    pibar_side_threshold=0.0,
    mt_squeezed=None,
    pibar_row_max=None,
    grad_mt_two_stage=False,
    grad_mt_two_stage_tile_splits=128,
    skip_inactive_pibar_output_zero=False,
    family_idx=None,
):
    """Fused DTS backward with direct Pi-adjoint accumulation."""
    n_ws = sl.shape[0]
    device = Pi_star.device
    dtype = Pi_star.dtype

    wlsp_flat = wlsp.squeeze(-1) if wlsp.ndim > 1 else wlsp
    family_idx_arg = None
    if family_idx is not None:
        family_idx_arg = family_idx.to(device=device, dtype=torch.long).contiguous()
    log_pD_arg, log_pS_arg, param_layout = _dts_layout_param_args(
        log_pD, log_pS, family_idx=family_idx_arg, S=S, device=device, dtype=dtype
    )
    device_scalar_params = False
    if param_layout == 0:
        device_scalar_params = True

    if accum_param_reductions and (grad_log_pD is None or grad_log_pS is None):
        raise ValueError("grad_log_pD/grad_log_pS are required when accumulating DTS scalar reductions")
    if accum_param_reductions:
        param_grad_layout = _dts_grad_layout(grad_log_pD, family_idx=family_idx_arg, S=S)
        if param_grad_layout != _dts_grad_layout(grad_log_pS, family_idx=family_idx_arg, S=S):
            raise ValueError("grad_log_pD/grad_log_pS must use the same DTS gradient layout")
    else:
        param_grad_layout = 0
    if accum_mt_reduction and grad_mt is None:
        raise ValueError("grad_mt is required when accumulating DTS mt reductions")
    if accum_mt_reduction:
        if grad_mt.numel() == 1 or (grad_mt.ndim == 1 and int(grad_mt.shape[0]) == int(S)):
            grad_mt_layout = 0
        elif family_idx_arg is not None and grad_mt.ndim == 2 and int(grad_mt.shape[1]) == int(S):
            grad_mt_layout = 1
        else:
            raise ValueError("DTS mt reduction target must be scalar, [S], or [G, S]")
    else:
        grad_mt_layout = 0
    if output_pibar_ud and (mt_squeezed is None or pibar_row_max is None):
        raise ValueError("mt_squeezed and pibar_row_max are required when outputting Pibar u_d")
    if output_pibar_ud:
        if mt_squeezed.ndim == 1 and int(mt_squeezed.shape[0]) == int(S):
            mt_layout = 0
        elif family_idx_arg is not None and mt_squeezed.ndim == 2 and int(mt_squeezed.shape[1]) == int(S):
            mt_layout = 1
        else:
            raise ValueError("mt_squeezed must have shape [S] or [G, S] when outputting Pibar u_d")
    else:
        mt_layout = 0
    if output_pibar_ud and pibar_row_max.numel() < Pi_star.shape[0]:
        raise ValueError("pibar_row_max must contain one row-max value per Pi row")
    if output_pibar_side_active and not output_pibar_ud:
        raise ValueError("output_pibar_side_active requires output_pibar_ud")
    if torch.is_tensor(pibar_side_threshold):
        if pibar_side_threshold.numel() != 1:
            raise ValueError("pibar_side_threshold tensor must contain one value")
        side_threshold_enabled = bool(output_pibar_side_active)
        side_active_threshold_arg = _device_scalar_param(
            pibar_side_threshold, device=device, dtype=dtype
        )
    else:
        pibar_side_threshold = float(pibar_side_threshold)
        if pibar_side_threshold < 0.0:
            raise ValueError("pibar_side_threshold must be non-negative")
        side_threshold_enabled = bool(output_pibar_side_active and pibar_side_threshold > 0.0)
        side_active_threshold_arg = (
            torch.tensor([pibar_side_threshold], device=device, dtype=dtype)
            if side_threshold_enabled
            else None
        )

    if output_pibar_ud:
        grad_Pibar_l = None
        grad_Pibar_r = None
    else:
        grad_Pibar_l = torch.empty((n_ws, S), device=device, dtype=dtype)
        grad_Pibar_r = torch.empty((n_ws, S), device=device, dtype=dtype)
    if output_pibar_ud:
        pibar_ud = torch.empty((2 * n_ws, S), device=device, dtype=dtype)
        pibar_A = torch.empty((2 * n_ws,), device=device, dtype=dtype)
    else:
        pibar_ud = None
        pibar_A = None
    pibar_side_active = (
        torch.empty((2 * n_ws,), device=device, dtype=torch.bool)
        if output_pibar_side_active
        else None
    )
    if accum_param_reductions:
        param_pD = None
        param_pS = None
    else:
        param_pD = torch.empty(n_ws, device=device, dtype=dtype)
        param_pS = torch.empty(n_ws, device=device, dtype=dtype)
    param_pD_arg = grad_log_pD if accum_param_reductions else param_pD
    param_pS_arg = grad_log_pS if accum_param_reductions else param_pS
    dummy = pibar_ud if output_pibar_ud else grad_Pibar_l
    grad_log_pD_arg = grad_log_pD if accum_param_reductions else dummy
    grad_log_pS_arg = grad_log_pS if accum_param_reductions else dummy
    grad_mt_arg = grad_mt if accum_mt_reduction else dummy
    pibar_ud_arg = pibar_ud if output_pibar_ud else dummy
    pibar_A_arg = pibar_A if output_pibar_ud else dummy
    pibar_side_active_arg = pibar_side_active if output_pibar_side_active else dummy
    mt_arg = mt_squeezed.contiguous() if output_pibar_ud and not mt_squeezed.is_contiguous() else mt_squeezed
    pibar_row_max_arg = (
        pibar_row_max.contiguous()
        if output_pibar_ud and not pibar_row_max.is_contiguous()
        else pibar_row_max
    )
    mt_arg = mt_arg if output_pibar_ud else dummy
    pibar_row_max_arg = pibar_row_max_arg if output_pibar_ud else dummy
    side_active_threshold_arg = side_active_threshold_arg if side_threshold_enabled else dummy
    family_idx_kernel_arg = family_idx_arg if family_idx_arg is not None else sl
    grad_mt_scalar = bool(accum_mt_reduction and grad_mt.numel() == 1)
    use_grad_mt_two_stage = bool(
        grad_mt_two_stage
        and accum_mt_reduction
        and grad_mt_layout == 0
        and not grad_mt_scalar
        and grad_mt.numel() == S
    )
    env_tile_splits = os.environ.get("GPUREC_DTS_GRAD_MT_TILE_SPLITS")
    if env_tile_splits is not None:
        grad_mt_two_stage_tile_splits = int(env_tile_splits)
    grad_mt_two_stage_tile_splits = max(1, int(grad_mt_two_stage_tile_splits))
    n_grad_mt_tiles = triton.cdiv(n_ws, grad_mt_two_stage_tile_splits)
    if use_grad_mt_two_stage:
        grad_mt_partial = torch.empty((n_grad_mt_tiles, S), device=device, dtype=dtype)
        grad_mt_partial.zero_()
    else:
        grad_mt_partial = dummy

    stride_C = Pi_star.stride(0)
    block_s_env = os.environ.get("GPUREC_DTS_BLOCK_S")
    if block_s_env is None:
        BLOCK_S = min(256, triton.next_power_of_2(S))
    else:
        BLOCK_S = min(
            max(1, triton.next_power_of_2(int(block_s_env))),
            triton.next_power_of_2(S),
        )
    dts_num_warps = int(os.environ.get("GPUREC_DTS_NUM_WARPS", "8"))
    launch_options = {}
    if dts_num_warps > 0:
        launch_options["num_warps"] = dts_num_warps

    _dts_cross_backward_accum_kernel[(n_ws,)](
        Pi_star, Pibar_star,
        v_k,
        active_mask if active_mask is not None else v_k,
        sl, sr, reduce_idx, wlsp_flat,
        log_pD_arg, log_pS_arg, family_idx_kernel_arg,
        sp_child1, sp_child2,
        accumulated_rhs,
        grad_Pibar_l if grad_Pibar_l is not None else pibar_ud,
        grad_Pibar_r if grad_Pibar_r is not None else pibar_ud,
        param_pD_arg, param_pS_arg,
        grad_log_pD_arg, grad_log_pS_arg, grad_mt_arg,
        grad_mt_partial,
        pibar_ud_arg, pibar_A_arg, pibar_side_active_arg, mt_arg, pibar_row_max_arg,
        side_active_threshold_arg,
        ws, S, stride_C, BLOCK_S,
        USE_ACTIVE_MASK=bool(active_mask is not None),
        USE_ATOMICS=bool(use_atomics),
        MERGE_S_TERM=bool(merge_s_term),
        DEVICE_SCALAR_PARAMS=bool(device_scalar_params),
        PARAM_LAYOUT=int(param_layout),
        PARAM_GRAD_LAYOUT=int(param_grad_layout),
        MT_LAYOUT=int(mt_layout),
        GRAD_MT_LAYOUT=int(grad_mt_layout),
        ACCUM_PARAM_REDUCTIONS=bool(accum_param_reductions),
        ACCUM_MT_REDUCTION=bool(accum_mt_reduction),
        GRAD_MT_SCALAR=bool(grad_mt_scalar),
        GRAD_MT_TWO_STAGE=bool(use_grad_mt_two_stage),
        GRAD_MT_TILE_SPLITS=grad_mt_two_stage_tile_splits,
        OUTPUT_PIBAR_UD=bool(output_pibar_ud),
        OUTPUT_SIDE_ACTIVE=bool(output_pibar_side_active),
        SIDE_ACTIVE_THRESHOLD_ENABLED=side_threshold_enabled,
        SKIP_INACTIVE_PIBAR_OUTPUT_ZERO=bool(skip_inactive_pibar_output_zero),
        DTYPE=_tl_float_dtype(dtype),
        **launch_options,
    )

    if use_grad_mt_two_stage:
        mt_block_s = min(64, triton.next_power_of_2(S))
        mt_block_tiles = 16
        _dts_grad_mt_two_stage_reduce_kernel[(triton.cdiv(S, mt_block_s),)](
            grad_mt_partial,
            grad_mt,
            n_grad_mt_tiles,
            S,
            mt_block_tiles,
            mt_block_s,
            DTYPE=_tl_float_dtype(dtype),
            num_warps=4,
        )

    if output_pibar_ud:
        if output_pibar_side_active:
            return pibar_ud, pibar_A, pibar_side_active, param_pD, param_pS
        return pibar_ud, pibar_A, param_pD, param_pS
    return grad_Pibar_l, grad_Pibar_r, param_pD, param_pS


# =========================================================================
# Uniform Pibar VJP for cross-clade gradients
# =========================================================================

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
    n_ws: tl.constexpr,
    S: tl.constexpr,
    stride_C: tl.constexpr,
    BLOCK_S: tl.constexpr,
    N_LEVELS: tl.constexpr,
    USE_ACTIVE_MASK: tl.constexpr,
    USE_SIDE_ACTIVE: tl.constexpr,
    DTYPE: tl.constexpr,
):
    """Uniform Pibar from-u_d tree correction using compact per-level nodes."""
    NEG_LARGE: tl.constexpr = -1e30

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
        p_prime = tl.exp2(pi_val - row_max)
        subtree_sum = tl.load(pibar_ud_ptr + row_base + s_offs, mask=mask, other=0.0)
        contrib = p_prime * (A - subtree_sum)
        tl.atomic_add(accumulated_rhs_ptr + pi_base + s_offs, contrib, sem="relaxed", mask=mask)


def uniform_cross_pibar_vjp_tree_from_ud_fused(
    Pi_star,
    pibar_ud,
    pibar_A,
    sl,
    sr,
    accumulated_rhs,
    S,
    active_mask=None,
    reduce_idx=None,
    pibar_row_max=None,
    skip_zero_sides=False,
    side_active=None,
    compact_level_ptr=None,
    compact_level_parents=None,
    compact_level_child1=None,
    compact_level_child2=None,
    side_active_threshold=0.0,
):
    """Uniform-Pibar VJP tree correction from DTS-staged u_d."""
    n_ws = sl.shape[0]
    if n_ws == 0:
        return
    if active_mask is not None and reduce_idx is None:
        raise ValueError("reduce_idx is required when active_mask is provided")
    if pibar_row_max is None:
        raise ValueError("pibar_row_max is required for DTS-staged Pibar VJP")
    if torch.is_tensor(side_active_threshold):
        if side_active_threshold.numel() != 1:
            raise ValueError("side_active_threshold tensor must contain one value")
        side_active_threshold_enabled = True
        side_active_threshold_arg = _device_scalar_param(
            side_active_threshold, device=Pi_star.device, dtype=Pi_star.dtype
        )
    else:
        side_active_threshold = float(side_active_threshold)
        if side_active_threshold < 0.0:
            raise ValueError("side_active_threshold must be non-negative")
        side_active_threshold_enabled = side_active_threshold > 0.0
        side_active_threshold_arg = (
            torch.tensor([side_active_threshold], device=Pi_star.device, dtype=Pi_star.dtype)
            if side_active_threshold_enabled
            else None
        )

    pibar_ud_block_s_env = os.environ.get("GPUREC_PIBAR_UD_BLOCK_S")
    if pibar_ud_block_s_env is None:
        BLOCK_S = min(256, triton.next_power_of_2(S))
    else:
        BLOCK_S = min(
            max(1, triton.next_power_of_2(int(pibar_ud_block_s_env))),
            triton.next_power_of_2(S),
        )
    pibar_ud_num_warps = int(os.environ.get("GPUREC_PIBAR_UD_NUM_WARPS", "4"))
    launch_options = {}
    if pibar_ud_num_warps > 0:
        launch_options["num_warps"] = pibar_ud_num_warps
    stride_C = Pi_star.stride(0)
    if side_active is not None:
        if side_active.numel() != 2 * n_ws:
            raise ValueError("side_active must have one entry per split side")
        side_active = side_active.contiguous()
    elif skip_zero_sides:
        side_active = torch.empty((2 * n_ws,), device=Pi_star.device, dtype=torch.bool)
        _pibar_ud_side_active_kernel[(2 * n_ws,)](
            pibar_ud,
            side_active,
            side_active_threshold_arg if side_active_threshold_enabled else pibar_ud,
            S,
            BLOCK_S,
            SIDE_ACTIVE_THRESHOLD_ENABLED=bool(side_active_threshold_enabled),
            DTYPE=_tl_float_dtype(Pi_star.dtype),
            **launch_options,
        )

    (
        cuda_pibar_from_ud_mode,
        cuda_pibar_from_ud_enabled,
        cuda_pibar_from_ud_required,
    ) = _cuda_pibar_from_ud_options()
    if (
        cuda_pibar_from_ud_enabled
        and Pi_star.dtype == torch.float32
        and Pi_star.device.type == "cuda"
    ):
        try:
            from .pibar_vjp_cuda import uniform_cross_pibar_vjp_tree_from_ud_cuda

            return uniform_cross_pibar_vjp_tree_from_ud_cuda(
                Pi_star,
                pibar_ud,
                pibar_A,
                sl,
                sr,
                accumulated_rhs,
                S,
                active_mask=active_mask,
                reduce_idx=reduce_idx,
                pibar_row_max=pibar_row_max,
                side_active=side_active,
                compact_level_ptr=compact_level_ptr,
                compact_level_parents=compact_level_parents,
                compact_level_child1=compact_level_child1,
                compact_level_child2=compact_level_child2,
            )
        except Exception as exc:
            if cuda_pibar_from_ud_required:
                raise

            if cuda_pibar_from_ud_mode not in ("auto", ""):
                import warnings

                global _cuda_pibar_from_ud_fallback_warned
                if not _cuda_pibar_from_ud_fallback_warned:
                    warnings.warn(
                        "GPUREC_CUDA_PIBAR_FROM_UD requested, but the CUDA "
                        f"prototype was unavailable ({exc}); falling back to Triton.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    _cuda_pibar_from_ud_fallback_warned = True

    if (
        compact_level_ptr is None
        or compact_level_parents is None
        or compact_level_child1 is None
        or compact_level_child2 is None
    ):
        raise ValueError("compact species levels are required for Pibar VJP")
    if compact_level_ptr.numel() < 2:
        raise ValueError("compact_level_ptr must contain at least start and end offsets")
    compact_level_ptr = compact_level_ptr.contiguous()
    compact_level_parents = compact_level_parents.contiguous()
    compact_level_child1 = compact_level_child1.contiguous()
    compact_level_child2 = compact_level_child2.contiguous()
    _uniform_cross_pibar_vjp_tree_from_ud_compact_kernel[(2 * n_ws,)](
        Pi_star,
        pibar_ud,
        pibar_A,
        side_active if side_active is not None else pibar_A,
        sl,
        sr,
        reduce_idx if reduce_idx is not None else sl,
        active_mask if active_mask is not None else pibar_ud,
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
        USE_ACTIVE_MASK=bool(active_mask is not None),
        USE_SIDE_ACTIVE=bool(side_active is not None),
        DTYPE=_tl_float_dtype(Pi_star.dtype),
        **launch_options,
    )
    return side_active
