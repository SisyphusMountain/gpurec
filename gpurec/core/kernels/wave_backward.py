"""Fused Triton kernels for wave-backward.

Two kernels:
1. _wave_backward_uniform_kernel: self-loop backward (Neumann VJP + param VJP)
2. _dts_cross_backward_kernel: cross-clade DTS backward (adjoint propagation + param VJP)

Both use one CTA per work-item, multi-pass over species dimension.
"""

import os

import torch
import triton
import triton.language as tl

_cuda_pibar_from_ud_fallback_warned = False


def _tl_float_dtype(dtype):
    return tl.float64 if dtype == torch.float64 else tl.float32


def _device_scalar_param(param, *, device, dtype):
    """Return a one-element device tensor without extracting CUDA scalars."""
    if torch.is_tensor(param):
        if param.numel() != 1:
            raise ValueError("fused DTS backward scalar parameters must have one element")
        if param.device != device or param.dtype != dtype:
            param = param.to(device=device, dtype=dtype)
        return param.reshape(1).contiguous()
    return torch.tensor([param], device=device, dtype=dtype)


def _dts_scalar_param_args(log_pD, log_pS, *, device, dtype):
    use_device_scalars = (
        os.environ.get("GPUREC_DTS_BACKWARD_DEVICE_SCALARS", "1") != "0"
    )
    if use_device_scalars:
        return (
            _device_scalar_param(log_pD, device=device, dtype=dtype),
            _device_scalar_param(log_pS, device=device, dtype=dtype),
            True,
        )

    def _extract(param):
        if torch.is_tensor(param):
            return float(param) if param.ndim == 0 else float(param.item())
        return float(param)

    return _extract(log_pD), _extract(log_pS), False


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


def _scratch_view(scratch, name, shape, *, device, dtype):
    """Return a contiguous leading slice from caller-owned scratch if valid."""
    if not isinstance(scratch, dict):
        return None
    buf = scratch.get(name)
    if not torch.is_tensor(buf):
        return None
    if buf.device != device or buf.dtype != dtype or buf.ndim != len(shape):
        return None
    if any(int(buf.shape[i]) < int(shape[i]) for i in range(len(shape))):
        return None
    view = buf[tuple(slice(0, int(dim)) for dim in shape)]
    return view if view.is_contiguous() else None


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


def _grad_param_is_scalar(grad):
    """Whether a log_pD/log_pS gradient tensor is family-scalar."""
    return grad.ndim == 1 or (grad.ndim == 2 and int(grad.shape[1]) == 1)


def _parent_from_children(sp_child1, sp_child2, S):
    """Build species parent pointers from child arrays for direct kernel callers."""
    device = sp_child1.device
    parent = torch.full((S,), -1, device=device, dtype=sp_child1.dtype)
    species = torch.arange(S, device=device, dtype=sp_child1.dtype)
    c1 = sp_child1.to(dtype=sp_child1.dtype)
    c2 = sp_child2.to(dtype=sp_child1.dtype)
    mask1 = c1 < S
    mask2 = c2 < S
    parent[c1[mask1].long()] = species[mask1]
    parent[c2[mask2].long()] = species[mask2]
    return parent.contiguous()


def _max_ancestor_depth_from_parent(sp_parent, S):
    """Return root-path length including self."""
    parent_values = sp_parent.detach().cpu().long().tolist()
    max_depth = 1
    for s_idx in range(S):
        cur = s_idx
        depth = 0
        while cur >= 0:
            depth += 1
            if depth > S:
                raise RuntimeError("Cycle detected in species parent pointers")
            cur = parent_values[cur]
        max_depth = max(max_depth, depth)
    return max_depth


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
    if rhs.dtype not in (torch.float32, torch.float64):
        raise ValueError("active_mask_from_rhs_absmax_fused supports fp32/fp64 tensors")

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
def _wave_backward_uniform_kernel(
    # Converged values from forward pass
    Pi_star_ptr,      # [C, S] — read rows [ws:ws+W]
    Pibar_star_ptr,   # [C, S] — read rows [ws:ws+W]
    Pibar_row_max_ptr, # optional [C] forward Pi row maxima for uniform Pibar
    dts_r_ptr,        # [W, S] or None — cross-clade DTS
    has_splits: tl.constexpr,
    # Incoming adjoint
    rhs_ptr,          # [W, S]
    active_mask_ptr,  # optional [W] bool row activity mask
    # Constants [S]
    mt_ptr, DL_const_ptr, Ebar_ptr, E_ptr, SL1_const_ptr, SL2_const_ptr,
    # Species children [S] long
    sp_child1_ptr, sp_child2_ptr,
    sp_parent_ptr,
    # Leaf term [W, S]
    leaf_term_ptr,
    leaf_species_ptr,
    leaf_logp_ptr,
    family_idx_ptr,
    # Outputs
    v_k_ptr,          # [W, S] — Neumann-solved adjoint
    # Per-element param grad contributions [W, S] each — reduced by caller
    aw0_ptr,          # grad contribution to log_pD, E (from term 0)
    aw1_ptr,          # grad contribution to Ebar (from term 1)
    aw2_ptr,          # grad contribution to E, mt (from term 2)
    aw345_ptr,        # grad contribution to log_pS (from terms 3+4+5)
    aw3_ptr,          # grad contribution to E_s2 (from term 3)
    aw4_ptr,          # grad contribution to E_s1 (from term 4)
    # Scratch buffer for speciation scatter [W, S]
    spec_buf_ptr,
    term_buf_ptr,
    pibar_corr_ptr,
    # Optional in-kernel accumulation targets for global-mode param grads.
    grad_log_pD_ptr,
    grad_log_pS_ptr,
    grad_E_ptr,
    grad_Ebar_ptr,
    grad_E_s1_ptr,
    grad_E_s2_ptr,
    grad_mt_ptr,
    # Dimensions
    ws,               # wave start offset into [C, S]
    S: tl.constexpr,
    stride: tl.constexpr,
    BLOCK_S: tl.constexpr,
    NEUMANN_TERMS: tl.constexpr,
    MAX_ANCESTOR_DEPTH: tl.constexpr,
    USE_LEAF_INDEX: tl.constexpr,
    ACCUM_PARAM_GRADS: tl.constexpr,
    PARAM_GRAD_TWO_STAGE: tl.constexpr,
    FAST_NOSPLIT_PARAM_GRADS: tl.constexpr,
    COMPACT_PIBAR_SCRATCH: tl.constexpr,
    RECOMPUTE_PIBAR_DENOM: tl.constexpr,
    LEAF_HIT_ONLY_LOGP: tl.constexpr,
    LEAF_LOGP_MODE: tl.constexpr,
    USE_PIBAR_ROW_MAX: tl.constexpr,
    SPEC_GATHER: tl.constexpr,
    USE_ACTIVE_MASK: tl.constexpr,
    SKIP_INACTIVE_ZERO_STORES: tl.constexpr,
    CONST_LAYOUT: tl.constexpr,
    LOG_PD_GRAD_SCALAR: tl.constexpr,
    LOG_PS_GRAD_SCALAR: tl.constexpr,
    DTYPE: tl.constexpr,
):
    """Fused backward kernel for uniform Pibar mode.

    Per clade w, computes:
    1. Softmax weights (w_L, w_terms) from converged Pi/Pibar
    2. Neumann series: v_k = (I + J^T + (J^T)^2 + ...) @ rhs
    3. Param VJP element-wise contributions

    The Neumann J^T application needs A = sum_s(u_d[s]) — a full-row reduction.
    Uniform Pibar also subtracts, for each species, the subtree sum over all
    descendants whose ancestor list contains that species.
    Each iteration uses 2 sub-passes:
      Pass A: compute u_d[s], accumulate A, scatter u_d to ancestor correction
      Pass B: compute result[s] using A and correction, read spec contribution
    """
    NEG_LARGE = tl.full([1], value=-1e30, dtype=DTYPE)

    w = tl.program_id(0)
    pi_base = (ws + w) * stride      # offset into [C, S]
    out_base = w * stride             # offset into [W, S]
    family = tl.full([1], value=0, dtype=tl.int64)
    const_base = 0
    grad_family_base = 0
    if CONST_LAYOUT == 1:
        const_base = out_base
    elif CONST_LAYOUT == 2:
        family = tl.load(family_idx_ptr + ws + w).to(tl.int64) + tl.zeros([1], dtype=tl.int64)
        const_base = family * stride
        grad_family_base = family * stride
    if USE_ACTIVE_MASK:
        row_active = tl.load(active_mask_ptr + w)
        if row_active == 0:
            if SKIP_INACTIVE_ZERO_STORES:
                return
            for s_start in range(0, S, BLOCK_S):
                s_offs = s_start + tl.arange(0, BLOCK_S)
                mask = s_offs < S
                zero = tl.zeros([BLOCK_S], dtype=DTYPE)
                tl.store(v_k_ptr + out_base + s_offs, zero, mask=mask)
                if (not ACCUM_PARAM_GRADS) and (not PARAM_GRAD_TWO_STAGE):
                    off = out_base + s_offs
                    tl.store(aw0_ptr + off, zero, mask=mask)
                    tl.store(aw1_ptr + off, zero, mask=mask)
                    tl.store(aw2_ptr + off, zero, mask=mask)
                    tl.store(aw345_ptr + off, zero, mask=mask)
                    tl.store(aw3_ptr + off, zero, mask=mask)
                    tl.store(aw4_ptr + off, zero, mask=mask)
            return
    else:
        row_active = True

    # ================================================================
    # Pass 1: Row statistics for uniform Pibar (same as forward)
    # ================================================================
    if USE_PIBAR_ROW_MAX:
        row_max = tl.load(Pibar_row_max_ptr + ws + w)
        row_sum = tl.full([1], value=0.0, dtype=DTYPE)
        for s_start in range(0, S, BLOCK_S):
            s_offs = s_start + tl.arange(0, BLOCK_S)
            valid_mask = s_offs < S
            mask = valid_mask & row_active
            pi_val = tl.load(Pi_star_ptr + pi_base + s_offs, mask=mask, other=-1e30)
            row_sum += tl.sum(tl.exp2(pi_val - row_max), axis=0)
    else:
        row_max = tl.full([1], value=-1e30, dtype=DTYPE)
        row_sum = tl.full([1], value=0.0, dtype=DTYPE)

        for s_start in range(0, S, BLOCK_S):
            s_offs = s_start + tl.arange(0, BLOCK_S)
            valid_mask = s_offs < S
            mask = valid_mask & row_active
            pi_val = tl.load(Pi_star_ptr + pi_base + s_offs, mask=mask, other=-1e30)
            tile_max = tl.max(pi_val, axis=0)
            new_max = tl.maximum(row_max, tile_max)
            row_sum = row_sum * tl.exp2(row_max - new_max) + tl.sum(tl.exp2(pi_val - new_max), axis=0)
            row_max = new_max

    # ================================================================
    # Pass 2: Compute softmax weights and store to [W, S] buffers
    # We need w_L[s], w_terms[0..5][s], inv_denom[s], p_prime[s]
    # These are consumed by the Neumann loop. Since S doesn't fit in
    # registers across passes, we store per-element to global memory.
    #
    # Actually, we interleave: compute weights and immediately use them.
    # But Neumann needs full-row A, so we can't do it in one pass.
    #
    # Strategy: store weights to reusable buffers, then iterate Neumann.
    # We reuse the output buffers (aw0..aw4) as scratch during Neumann,
    # then overwrite with final param contributions.
    #
    # Stored per-element (to aw* buffers temporarily):
    #   aw0 = w_L * (w_terms[0] + w_terms[1])  — diagonal weight
    #   aw1 = w_L * w_terms[2]                  — Pibar weight
    #   aw2 = inv_denom                         — for Pibar VJP
    #   aw3 = p_prime                            — for Pibar VJP
    #   aw4 = w_L * w_terms[3]                  — SL1 weight
    #   aw345 = w_L * w_terms[4]                — SL2 weight
    # ================================================================
    M_SAFE = tl.full([1], value=-1e29, dtype=DTYPE)

    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        valid_mask = s_offs < S
        mask = valid_mask & row_active
        off = out_base + s_offs

        # Load Pi*, Pibar*
        pi_w = tl.load(Pi_star_ptr + pi_base + s_offs, mask=mask, other=-1e30)
        pibar_w = tl.load(Pibar_star_ptr + pi_base + s_offs, mask=mask, other=-1e30)

        # Load constants
        dl_c = tl.load(DL_const_ptr + const_base + s_offs, mask=mask, other=-1e30)
        ebar = tl.load(Ebar_ptr + const_base + s_offs, mask=mask, other=-1e30)
        e_val = tl.load(E_ptr + const_base + s_offs, mask=mask, other=-1e30)
        sl1_c = tl.load(SL1_const_ptr + const_base + s_offs, mask=mask, other=-1e30)
        sl2_c = tl.load(SL2_const_ptr + const_base + s_offs, mask=mask, other=-1e30)

        # Gather species children
        c1 = tl.load(sp_child1_ptr + s_offs, mask=mask, other=0)
        c2 = tl.load(sp_child2_ptr + s_offs, mask=mask, other=0)
        c1_valid = c1 < S
        c2_valid = c2 < S
        pi_s1 = tl.load(Pi_star_ptr + pi_base + c1, mask=mask & c1_valid, other=-1e30)
        pi_s2 = tl.load(Pi_star_ptr + pi_base + c2, mask=mask & c2_valid, other=-1e30)

        # 6 DTS_L terms
        t0 = dl_c + pi_w
        t1 = pi_w + ebar
        t2 = pibar_w + e_val
        t3 = sl1_c + pi_s1
        t4 = sl2_c + pi_s2
        if USE_LEAF_INDEX:
            leaf_species = tl.load(leaf_species_ptr + ws + w)
            leaf_hit = mask & (leaf_species == s_offs)
            if LEAF_LOGP_MODE == 3:
                leaf_logp = tl.load(leaf_logp_ptr).to(DTYPE) + tl.zeros([BLOCK_S], dtype=DTYPE)
                t5 = tl.where(leaf_hit, leaf_logp, NEG_LARGE)
            elif LEAF_LOGP_MODE == 2:
                leaf_logp = tl.load(
                    leaf_logp_ptr + family * stride + s_offs,
                    mask=leaf_hit,
                    other=-1e30,
                )
                t5 = tl.where(leaf_hit, leaf_logp, NEG_LARGE)
            elif LEAF_LOGP_MODE == 1:
                leaf_family = tl.load(family_idx_ptr + ws + w).to(tl.int64)
                leaf_logp = (
                    tl.load(leaf_logp_ptr + leaf_family).to(DTYPE)
                    + tl.zeros([BLOCK_S], dtype=DTYPE)
                )
                t5 = tl.where(leaf_hit, leaf_logp, NEG_LARGE)
            elif LEAF_HIT_ONLY_LOGP:
                t5 = tl.load(leaf_logp_ptr + s_offs, mask=leaf_hit, other=-1e30)
            else:
                leaf_logp = tl.load(leaf_logp_ptr + s_offs, mask=mask, other=-1e30)
                t5 = tl.where(leaf_hit, leaf_logp, NEG_LARGE)
        else:
            t5 = tl.load(leaf_term_ptr + off, mask=mask, other=-1e30)

        # Logsumexp over 6 terms → DTS_L
        m = tl.maximum(t0, t1)
        m = tl.maximum(m, t2)
        m = tl.maximum(m, t3)
        m = tl.maximum(m, t4)
        m = tl.maximum(m, t5)
        m_safe = tl.where(m > M_SAFE, m, tl.zeros_like(m))
        e0 = tl.exp2(t0 - m_safe)
        e1 = tl.exp2(t1 - m_safe)
        e2 = tl.exp2(t2 - m_safe)
        e3 = tl.exp2(t3 - m_safe)
        e4 = tl.exp2(t4 - m_safe)
        e5 = tl.exp2(t5 - m_safe)
        dts_l_sum = e0 + e1 + e2 + e3 + e4 + e5
        dts_l = tl.log2(dts_l_sum) + m

        # w_L = exp2(DTS_L - Pi_new), w_terms[i] = exp2(terms[i] - DTS_L) = e_i / dts_l_sum
        if has_splits:
            dts_r = tl.load(dts_r_ptr + off, mask=mask, other=-1e30)
            pi_new_m = tl.maximum(dts_l, dts_r)
            pi_new_ms = tl.where(pi_new_m > M_SAFE, pi_new_m, tl.zeros_like(pi_new_m))
            pi_new = tl.log2(tl.exp2(dts_l - pi_new_ms) + tl.exp2(dts_r - pi_new_ms)) + pi_new_m
            # w_L: safe when DTS_L = -inf → w_L = 0
            w_L = tl.where(dts_l > M_SAFE, tl.exp2(dts_l - pi_new), tl.zeros_like(dts_l))
        else:
            w_L = tl.full(s_offs.shape, value=1.0, dtype=DTYPE)

        # Per-term softmax weights (divide by dts_l_sum, not dts_l_sum + dts_r)
        inv_sum = tl.where(dts_l_sum > 0, 1.0 / dts_l_sum, tl.zeros_like(dts_l_sum))
        wt0 = e0 * inv_sum
        wt1 = e1 * inv_sum
        wt2 = e2 * inv_sum
        wt3 = e3 * inv_sum
        wt4 = e4 * inv_sum
        # wt5 = e5 * inv_sum  (only needed for param VJP log_pS, computed later)

        # Pibar VJP ingredients:
        #   denom[s] = sum_j exp(Pi[j]-row_max)
        #              - sum_{a in ancestors(s)} exp(Pi[a]-row_max)
        p_prime = tl.exp2(pi_w - row_max)
        ancestor_sum = tl.zeros([BLOCK_S], dtype=DTYPE)
        cur = s_offs
        for _depth in range(MAX_ANCESTOR_DEPTH):
            anc_mask = mask & (cur >= 0) & (cur < S)
            pi_anc = tl.load(Pi_star_ptr + pi_base + cur, mask=anc_mask, other=-1e30)
            ancestor_sum += tl.where(
                anc_mask,
                tl.exp2(pi_anc - row_max),
                tl.zeros([BLOCK_S], dtype=DTYPE),
            )
            cur = tl.load(sp_parent_ptr + cur, mask=anc_mask, other=-1)
        denom = row_sum - ancestor_sum
        inv_denom = tl.where(denom > 0, 1.0 / denom, tl.zeros_like(denom))

        # Store precomputed weights to scratch buffers
        diag_wt = w_L * (wt0 + wt1)        # diagonal J^T weight
        pibar_wt = w_L * wt2               # Pibar path weight
        sl1_wt = w_L * wt3                 # SL1 speciation weight
        sl2_wt = w_L * wt4                 # SL2 speciation weight

        tl.store(aw0_ptr + off, diag_wt, mask=mask)
        if COMPACT_PIBAR_SCRATCH:
            tl.store(aw1_ptr + off, pibar_wt * inv_denom, mask=mask)
        else:
            tl.store(aw1_ptr + off, pibar_wt, mask=mask)
        if (not RECOMPUTE_PIBAR_DENOM) and (not COMPACT_PIBAR_SCRATCH):
            tl.store(aw2_ptr + off, inv_denom, mask=mask)
            tl.store(aw3_ptr + off, p_prime, mask=mask)
        tl.store(aw4_ptr + off, sl1_wt, mask=mask)
        tl.store(aw345_ptr + off, sl2_wt, mask=mask)

    # ================================================================
    # Neumann series: v = rhs + J^T(rhs) + (J^T)^2(rhs) + ...
    #
    # Each J^T application on vector `term` requires:
    #   Pass A: compute u_d = term * pibar_wt * inv_denom, accumulate A = sum(u_d),
    #           scatter u_d to ancestors, and scatter speciation contributions.
    #   Pass B: result[s] = term[s] * diag_wt[s] + p_prime[s] * (A - correction[s])
    #                        + speciation contribution.
    #
    # SPEC_GATHER replaces the scatter/zero/read speciation path with:
    #   spec[s] = term[parent[s]] * sl_weight[parent[s] -> s]
    # This removes one full zero pass and the speciation scatter stores, but
    # adds parent-index gathers from the current term and scratch weights.
    # ================================================================
    # Copy rhs → v_k (v_k accumulates the Neumann sum)
    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        valid_mask = s_offs < S
        mask = valid_mask & row_active
        rhs_val = tl.load(rhs_ptr + out_base + s_offs, mask=mask, other=0.0)
        tl.store(v_k_ptr + out_base + s_offs, rhs_val, mask=valid_mask)

    # Buffer ping-pong: iteration 0 reads rhs_ptr, even iterations write
    # spec_buf, and odd iterations write term_buf. Output buffer is zeroed
    # at the start of each iteration to avoid stale data at non-child positions.

    for _n in range(NEUMANN_TERMS):
        # Zero the correction buffer before ancestor scatters.
        for s_start in range(0, S, BLOCK_S):
            s_offs = s_start + tl.arange(0, BLOCK_S)
            valid_mask = s_offs < S
            mask = valid_mask & row_active
            zero = tl.zeros(s_offs.shape, dtype=DTYPE)
            tl.store(pibar_corr_ptr + out_base + s_offs, zero, mask=mask)

        if not SPEC_GATHER:
            # The scatter speciation path needs its ping-pong output buffer for
            # child contributions.
            for s_start in range(0, S, BLOCK_S):
                s_offs = s_start + tl.arange(0, BLOCK_S)
                valid_mask = s_offs < S
                mask = valid_mask & row_active
                zero = tl.zeros(s_offs.shape, dtype=DTYPE)
                # Zero the output buffer before speciation scatter writes.
                # Sub-pass A only writes child positions; sub-pass B reads all
                # positions, so non-child positions must not retain stale terms.
                if _n % 2 == 0:
                    tl.store(spec_buf_ptr + out_base + s_offs, zero, mask=mask)
                else:
                    tl.store(term_buf_ptr + out_base + s_offs, zero, mask=mask)

        tl.debug_barrier()

        # --- Sub-pass A: accumulate A = sum_s(term * pibar_wt * inv_denom) ---
        # Also write ancestor-correction and speciation scatter contributions.
        A_acc = tl.full([1], value=0.0, dtype=DTYPE)

        for s_start in range(0, S, BLOCK_S):
            s_offs = s_start + tl.arange(0, BLOCK_S)
            valid_mask = s_offs < S
            mask = valid_mask & row_active
            off = out_base + s_offs

            # Load term from appropriate buffer
            if _n == 0:
                term_val = tl.load(rhs_ptr + off, mask=mask, other=0.0)
            elif _n % 2 == 1:
                term_val = tl.load(spec_buf_ptr + off, mask=mask, other=0.0)
            else:
                term_val = tl.load(term_buf_ptr + off, mask=mask, other=0.0)

            if COMPACT_PIBAR_SCRATCH:
                pibar_u_coeff = tl.load(aw1_ptr + off, mask=mask, other=0.0)
                u_d = term_val * pibar_u_coeff
            else:
                pibar_wt = tl.load(aw1_ptr + off, mask=mask, other=0.0)
                if RECOMPUTE_PIBAR_DENOM:
                    pi_w = tl.load(Pi_star_ptr + pi_base + s_offs, mask=mask, other=-1e30)
                    p_prime = tl.exp2(pi_w - row_max)
                    denom = row_sum - p_prime
                    inv_denom = tl.where(denom > 0, 1.0 / denom, tl.zeros_like(denom))
                else:
                    inv_denom = tl.load(aw2_ptr + off, mask=mask, other=0.0)

                u_d = term_val * pibar_wt * inv_denom

            A_acc += tl.sum(u_d, axis=0)

            cur = s_offs
            for _depth in range(MAX_ANCESTOR_DEPTH):
                anc_mask = mask & (cur >= 0) & (cur < S)
                tl.atomic_add(pibar_corr_ptr + out_base + cur, u_d, mask=anc_mask)
                cur = tl.load(sp_parent_ptr + cur, mask=anc_mask, other=-1)

            if not SPEC_GATHER:
                # Speciation scatter: write term * sl_wt to child index
                sl1_wt = tl.load(aw4_ptr + off, mask=mask, other=0.0)
                sl2_wt = tl.load(aw345_ptr + off, mask=mask, other=0.0)
                c1 = tl.load(sp_child1_ptr + s_offs, mask=mask, other=0)
                c2 = tl.load(sp_child2_ptr + s_offs, mask=mask, other=0)
                c1_valid = (c1 < S) & mask
                c2_valid = (c2 < S) & mask

                # No conflict: each child appears as target of exactly one parent.
                src1 = term_val * sl1_wt
                src2 = term_val * sl2_wt
                # Write to output buffer at child index (using the OTHER buffer)
                if _n % 2 == 0:
                    # Writing to spec_buf
                    tl.store(spec_buf_ptr + out_base + c1, src1, mask=c1_valid)
                    tl.store(spec_buf_ptr + out_base + c2, src2, mask=c2_valid)
                else:
                    tl.store(term_buf_ptr + out_base + c1, src1, mask=c1_valid)
                    tl.store(term_buf_ptr + out_base + c2, src2, mask=c2_valid)

        tl.debug_barrier()

        # --- Sub-pass B: compute J^T result using A ---
        for s_start in range(0, S, BLOCK_S):
            s_offs = s_start + tl.arange(0, BLOCK_S)
            valid_mask = s_offs < S
            mask = valid_mask & row_active
            off = out_base + s_offs

            # Reload term and weights
            if _n == 0:
                term_val = tl.load(rhs_ptr + off, mask=mask, other=0.0)
            elif _n % 2 == 1:
                term_val = tl.load(spec_buf_ptr + off, mask=mask, other=0.0)
            else:
                term_val = tl.load(term_buf_ptr + off, mask=mask, other=0.0)

            diag_wt = tl.load(aw0_ptr + off, mask=mask, other=0.0)
            if COMPACT_PIBAR_SCRATCH:
                pibar_u_coeff = tl.load(aw1_ptr + off, mask=mask, other=0.0)
                pi_w = tl.load(Pi_star_ptr + pi_base + s_offs, mask=mask, other=-1e30)
                p_prime = tl.exp2(pi_w - row_max)
                u_d = term_val * pibar_u_coeff
            else:
                pibar_wt = tl.load(aw1_ptr + off, mask=mask, other=0.0)
                if RECOMPUTE_PIBAR_DENOM:
                    pi_w = tl.load(Pi_star_ptr + pi_base + s_offs, mask=mask, other=-1e30)
                    p_prime = tl.exp2(pi_w - row_max)
                    denom = row_sum - p_prime
                    inv_denom = tl.where(denom > 0, 1.0 / denom, tl.zeros_like(denom))
                else:
                    inv_denom = tl.load(aw2_ptr + off, mask=mask, other=0.0)
                    p_prime = tl.load(aw3_ptr + off, mask=mask, other=0.0)

                u_d = term_val * pibar_wt * inv_denom
            correction = tl.load(pibar_corr_ptr + off, mask=mask, other=0.0)
            result = term_val * diag_wt + p_prime * (A_acc - correction)

            # Add speciation contribution.
            if SPEC_GATHER:
                parent = tl.load(sp_parent_ptr + s_offs, mask=mask, other=-1)
                parent_valid = parent >= 0
                parent_off = out_base + parent
                if _n == 0:
                    parent_term = tl.load(rhs_ptr + parent_off, mask=mask & parent_valid, other=0.0)
                elif _n % 2 == 1:
                    parent_term = tl.load(spec_buf_ptr + parent_off, mask=mask & parent_valid, other=0.0)
                else:
                    parent_term = tl.load(term_buf_ptr + parent_off, mask=mask & parent_valid, other=0.0)
                parent_sl1 = tl.load(aw4_ptr + parent_off, mask=mask & parent_valid, other=0.0)
                parent_sl2 = tl.load(aw345_ptr + parent_off, mask=mask & parent_valid, other=0.0)
                parent_c1 = tl.load(sp_child1_ptr + parent, mask=mask & parent_valid, other=S)
                parent_wt = tl.where(parent_c1 == s_offs, parent_sl1, parent_sl2)
                spec_val = parent_term * parent_wt
            elif _n % 2 == 0:
                spec_val = tl.load(spec_buf_ptr + off, mask=mask, other=0.0)
            else:
                spec_val = tl.load(term_buf_ptr + off, mask=mask, other=0.0)
            result = result + spec_val

            # Store result to output buffer
            if _n % 2 == 0:
                tl.store(spec_buf_ptr + off, result, mask=mask)
            else:
                tl.store(term_buf_ptr + off, result, mask=mask)

            # Accumulate into v_k
            v_k_val = tl.load(v_k_ptr + off, mask=mask, other=0.0)
            tl.store(v_k_ptr + off, v_k_val + result, mask=mask)

    # ================================================================
    # Pass final: Param VJP contributions
    # Recompute alpha = v_k * w_L and weighted terms.
    # Store per-element contributions for the caller to reduce.
    # ================================================================
    if PARAM_GRAD_TWO_STAGE:
        return

    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        valid_mask = s_offs < S
        mask = valid_mask & row_active
        off = out_base + s_offs

        v_k_val = tl.load(v_k_ptr + off, mask=mask, other=0.0)

        if ACCUM_PARAM_GRADS and FAST_NOSPLIT_PARAM_GRADS and not has_splits and not COMPACT_PIBAR_SCRATCH:
            diag_wt = tl.load(aw0_ptr + off, mask=mask, other=0.0)
            pibar_wt = tl.load(aw1_ptr + off, mask=mask, other=0.0)
            sl1_wt = tl.load(aw4_ptr + off, mask=mask, other=0.0)
            sl2_wt = tl.load(aw345_ptr + off, mask=mask, other=0.0)
            leaf_wt = 1.0 - diag_wt - pibar_wt - sl1_wt - sl2_wt

            dl_c = tl.load(DL_const_ptr + const_base + s_offs, mask=mask, other=-1e30)
            ebar = tl.load(Ebar_ptr + const_base + s_offs, mask=mask, other=-1e30)
            m01 = tl.maximum(dl_c, ebar)
            e0_01 = tl.exp2(dl_c - m01)
            e1_01 = tl.exp2(ebar - m01)
            frac_d = e0_01 / (e0_01 + e1_01)

            _aw0 = v_k_val * diag_wt * frac_d
            _aw1 = v_k_val * diag_wt * (1.0 - frac_d)
            _aw2 = v_k_val * pibar_wt
            _aw3 = v_k_val * sl1_wt
            _aw4 = v_k_val * sl2_wt
            _aw345 = v_k_val * (sl1_wt + sl2_wt + leaf_wt)

            if LOG_PD_GRAD_SCALAR:
                tl.atomic_add(
                    grad_log_pD_ptr + family,
                    tl.sum(tl.where(mask, _aw0, 0.0), axis=0),
                    sem="relaxed",
                )
            else:
                tl.atomic_add(
                    grad_log_pD_ptr + grad_family_base + s_offs,
                    _aw0,
                    sem="relaxed",
                    mask=mask,
                )
            if LOG_PS_GRAD_SCALAR:
                tl.atomic_add(
                    grad_log_pS_ptr + family,
                    tl.sum(tl.where(mask, _aw345, 0.0), axis=0),
                    sem="relaxed",
                )
            else:
                tl.atomic_add(
                    grad_log_pS_ptr + grad_family_base + s_offs,
                    _aw345,
                    sem="relaxed",
                    mask=mask,
                )
            tl.atomic_add(grad_E_ptr + grad_family_base + s_offs, _aw0 + _aw2, sem="relaxed", mask=mask)
            tl.atomic_add(grad_Ebar_ptr + grad_family_base + s_offs, _aw1, sem="relaxed", mask=mask)
            tl.atomic_add(grad_E_s1_ptr + grad_family_base + s_offs, _aw4, sem="relaxed", mask=mask)
            tl.atomic_add(grad_E_s2_ptr + grad_family_base + s_offs, _aw3, sem="relaxed", mask=mask)
            tl.atomic_add(grad_mt_ptr + grad_family_base + s_offs, _aw2, sem="relaxed", mask=mask)
        else:
            # Reload Pi and Pibar to recompute weights
            # (we overwrote aw* buffers with Jt scratch data)
            pi_w = tl.load(Pi_star_ptr + pi_base + s_offs, mask=mask, other=-1e30)
            pibar_w = tl.load(Pibar_star_ptr + pi_base + s_offs, mask=mask, other=-1e30)
            dl_c = tl.load(DL_const_ptr + const_base + s_offs, mask=mask, other=-1e30)
            ebar = tl.load(Ebar_ptr + const_base + s_offs, mask=mask, other=-1e30)
            e_val = tl.load(E_ptr + const_base + s_offs, mask=mask, other=-1e30)
            sl1_c = tl.load(SL1_const_ptr + const_base + s_offs, mask=mask, other=-1e30)
            sl2_c = tl.load(SL2_const_ptr + const_base + s_offs, mask=mask, other=-1e30)
            c1 = tl.load(sp_child1_ptr + s_offs, mask=mask, other=0)
            c2 = tl.load(sp_child2_ptr + s_offs, mask=mask, other=0)
            c1_valid = c1 < S
            c2_valid = c2 < S
            pi_s1 = tl.load(Pi_star_ptr + pi_base + c1, mask=mask & c1_valid, other=-1e30)
            pi_s2 = tl.load(Pi_star_ptr + pi_base + c2, mask=mask & c2_valid, other=-1e30)
            if USE_LEAF_INDEX:
                leaf_species = tl.load(leaf_species_ptr + ws + w)
                leaf_hit = mask & (leaf_species == s_offs)
                if LEAF_LOGP_MODE == 3:
                    leaf_logp = tl.load(leaf_logp_ptr).to(DTYPE) + tl.zeros([BLOCK_S], dtype=DTYPE)
                    t5 = tl.where(leaf_hit, leaf_logp, -1e30)
                elif LEAF_LOGP_MODE == 2:
                    leaf_logp = tl.load(
                        leaf_logp_ptr + family * stride + s_offs,
                        mask=leaf_hit,
                        other=-1e30,
                    )
                    t5 = tl.where(leaf_hit, leaf_logp, -1e30)
                elif LEAF_LOGP_MODE == 1:
                    leaf_family = tl.load(family_idx_ptr + ws + w).to(tl.int64)
                    leaf_logp = (
                        tl.load(leaf_logp_ptr + leaf_family).to(DTYPE)
                        + tl.zeros([BLOCK_S], dtype=DTYPE)
                    )
                    t5 = tl.where(leaf_hit, leaf_logp, -1e30)
                elif LEAF_HIT_ONLY_LOGP:
                    t5 = tl.load(leaf_logp_ptr + s_offs, mask=leaf_hit, other=-1e30)
                else:
                    leaf_logp = tl.load(leaf_logp_ptr + s_offs, mask=mask, other=-1e30)
                    t5 = tl.where(leaf_hit, leaf_logp, -1e30)
            else:
                t5 = tl.load(leaf_term_ptr + off, mask=mask, other=-1e30)

            # Recompute DTS_L terms and softmax weights
            t0 = dl_c + pi_w
            t1 = pi_w + ebar
            t2 = pibar_w + e_val
            t3 = sl1_c + pi_s1
            t4 = sl2_c + pi_s2
            m = tl.maximum(tl.maximum(tl.maximum(t0, t1), tl.maximum(t2, t3)), tl.maximum(t4, t5))
            m_safe = tl.where(m > -1e29, m, tl.zeros_like(m))
            e0 = tl.exp2(t0 - m_safe)
            e1 = tl.exp2(t1 - m_safe)
            e2 = tl.exp2(t2 - m_safe)
            e3 = tl.exp2(t3 - m_safe)
            e4 = tl.exp2(t4 - m_safe)
            e5 = tl.exp2(t5 - m_safe)
            dts_l_sum = e0 + e1 + e2 + e3 + e4 + e5
            inv_sum = tl.where(dts_l_sum > 0, 1.0 / dts_l_sum, tl.zeros_like(dts_l_sum))

            if has_splits:
                dts_r = tl.load(dts_r_ptr + off, mask=mask, other=-1e30)
                dts_l = tl.log2(dts_l_sum) + m
                pi_new_m = tl.maximum(dts_l, dts_r)
                pi_new_ms = tl.where(pi_new_m > -1e29, pi_new_m, tl.zeros_like(pi_new_m))
                pi_new = tl.log2(tl.exp2(dts_l - pi_new_ms) + tl.exp2(dts_r - pi_new_ms)) + pi_new_m
                w_L = tl.where(dts_l > -1e29, tl.exp2(dts_l - pi_new), tl.zeros_like(dts_l))
            else:
                w_L = tl.full(s_offs.shape, value=1.0, dtype=DTYPE)

            alpha = v_k_val * w_L

            # Per-element param contributions
            _aw0 = alpha * e0 * inv_sum   # → log_pD, E
            _aw1 = alpha * e1 * inv_sum   # → Ebar
            _aw2 = alpha * e2 * inv_sum   # → E, mt
            _aw3 = alpha * e3 * inv_sum   # → log_pS, E_s2
            _aw4 = alpha * e4 * inv_sum   # → log_pS, E_s1
            _aw5 = alpha * e5 * inv_sum   # → log_pS

            _aw345 = _aw3 + _aw4 + _aw5
            if ACCUM_PARAM_GRADS:
                if LOG_PD_GRAD_SCALAR:
                    tl.atomic_add(
                        grad_log_pD_ptr + family,
                        tl.sum(tl.where(mask, _aw0, 0.0), axis=0),
                        sem="relaxed",
                    )
                else:
                    tl.atomic_add(
                        grad_log_pD_ptr + grad_family_base + s_offs,
                        _aw0,
                        sem="relaxed",
                        mask=mask,
                    )
                if LOG_PS_GRAD_SCALAR:
                    tl.atomic_add(
                        grad_log_pS_ptr + family,
                        tl.sum(tl.where(mask, _aw345, 0.0), axis=0),
                        sem="relaxed",
                    )
                else:
                    tl.atomic_add(
                        grad_log_pS_ptr + grad_family_base + s_offs,
                        _aw345,
                        sem="relaxed",
                        mask=mask,
                    )
                tl.atomic_add(grad_E_ptr + grad_family_base + s_offs, _aw0 + _aw2, sem="relaxed", mask=mask)
                tl.atomic_add(grad_Ebar_ptr + grad_family_base + s_offs, _aw1, sem="relaxed", mask=mask)
                tl.atomic_add(grad_E_s1_ptr + grad_family_base + s_offs, _aw4, sem="relaxed", mask=mask)
                tl.atomic_add(grad_E_s2_ptr + grad_family_base + s_offs, _aw3, sem="relaxed", mask=mask)
                tl.atomic_add(grad_mt_ptr + grad_family_base + s_offs, _aw2, sem="relaxed", mask=mask)
            else:
                tl.store(aw0_ptr + off, _aw0, mask=valid_mask)
                tl.store(aw1_ptr + off, _aw1, mask=valid_mask)
                tl.store(aw2_ptr + off, _aw2, mask=valid_mask)
                tl.store(aw345_ptr + off, _aw345, mask=valid_mask)
                tl.store(aw3_ptr + off, _aw3, mask=valid_mask)
                tl.store(aw4_ptr + off, _aw4, mask=valid_mask)


@triton.jit
def _wave_backward_uniform_param_stage1_kernel(
    Pi_star_ptr,       # [C, S]
    Pibar_star_ptr,    # [C, S]
    v_k_ptr,           # [W, S]
    active_mask_ptr,   # optional [W]
    mt_ptr, DL_const_ptr, Ebar_ptr, E_ptr, SL1_const_ptr, SL2_const_ptr,
    sp_child1_ptr, sp_child2_ptr,
    leaf_term_ptr,
    leaf_species_ptr,
    leaf_logp_ptr,
    partial_vec_ptr,      # [5, N_TILES, S]
    partial_scalar_ptr,   # [2, N_TILES, N_S_BLOCKS]
    ws,
    W: tl.constexpr,
    S: tl.constexpr,
    stride: tl.constexpr,
    N_TILES: tl.constexpr,
    N_S_BLOCKS: tl.constexpr,
    TILE_ROWS: tl.constexpr,
    BLOCK_S: tl.constexpr,
    USE_LEAF_INDEX: tl.constexpr,
    LEAF_HIT_ONLY_LOGP: tl.constexpr,
    LEAF_LOGP_SCALAR: tl.constexpr,
    USE_ACTIVE_MASK: tl.constexpr,
    DTYPE: tl.constexpr,
):
    """Reduce no-split self-loop parameter VJPs into row-tile partials."""
    NEG_LARGE: tl.constexpr = -1e30
    M_SAFE: tl.constexpr = -1e29

    tile = tl.program_id(0)
    s_block = tl.program_id(1)
    rows = tile * TILE_ROWS + tl.arange(0, TILE_ROWS)
    s_offs = s_block * BLOCK_S + tl.arange(0, BLOCK_S)
    row_mask = rows < W
    valid_s = s_offs < S

    if USE_ACTIVE_MASK:
        active = tl.load(active_mask_ptr + rows, mask=row_mask, other=0)
        row_mask = row_mask & (active != 0)

    mask = row_mask[:, None] & valid_s[None, :]
    row_global = ws + rows
    pi_base = row_global[:, None] * stride
    row_base = rows[:, None] * S
    s_matrix = s_offs[None, :]

    pi_w = tl.load(Pi_star_ptr + pi_base + s_matrix, mask=mask, other=NEG_LARGE)
    pibar_w = tl.load(Pibar_star_ptr + pi_base + s_matrix, mask=mask, other=NEG_LARGE)
    v_k_val = tl.load(v_k_ptr + row_base + s_matrix, mask=mask, other=0.0)

    dl_c = tl.load(DL_const_ptr + s_offs, mask=valid_s, other=NEG_LARGE)
    ebar = tl.load(Ebar_ptr + s_offs, mask=valid_s, other=NEG_LARGE)
    e_val = tl.load(E_ptr + s_offs, mask=valid_s, other=NEG_LARGE)
    sl1_c = tl.load(SL1_const_ptr + s_offs, mask=valid_s, other=NEG_LARGE)
    sl2_c = tl.load(SL2_const_ptr + s_offs, mask=valid_s, other=NEG_LARGE)

    c1 = tl.load(sp_child1_ptr + s_offs, mask=valid_s, other=0)
    c2 = tl.load(sp_child2_ptr + s_offs, mask=valid_s, other=0)
    c1_valid = c1 < S
    c2_valid = c2 < S
    pi_s1 = tl.load(
        Pi_star_ptr + pi_base + c1[None, :],
        mask=mask & c1_valid[None, :],
        other=NEG_LARGE,
    )
    pi_s2 = tl.load(
        Pi_star_ptr + pi_base + c2[None, :],
        mask=mask & c2_valid[None, :],
        other=NEG_LARGE,
    )

    t0 = dl_c[None, :] + pi_w
    t1 = pi_w + ebar[None, :]
    t2 = pibar_w + e_val[None, :]
    t3 = sl1_c[None, :] + pi_s1
    t4 = sl2_c[None, :] + pi_s2
    if USE_LEAF_INDEX:
        leaf_species = tl.load(leaf_species_ptr + row_global, mask=row_mask, other=-1)
        leaf_hit = mask & (leaf_species[:, None] == s_matrix)
        if LEAF_LOGP_SCALAR:
            leaf_logp = tl.load(leaf_logp_ptr)
            t5 = tl.where(leaf_hit, leaf_logp, NEG_LARGE)
        elif LEAF_HIT_ONLY_LOGP:
            leaf_logp = tl.load(leaf_logp_ptr + s_offs, mask=valid_s, other=NEG_LARGE)
            t5 = tl.where(leaf_hit, leaf_logp[None, :], NEG_LARGE)
        else:
            leaf_logp = tl.load(leaf_logp_ptr + s_offs, mask=valid_s, other=NEG_LARGE)
            t5 = tl.where(leaf_hit, leaf_logp[None, :], NEG_LARGE)
    else:
        t5 = tl.load(leaf_term_ptr + row_base + s_matrix, mask=mask, other=NEG_LARGE)

    m = tl.maximum(t0, t1)
    m = tl.maximum(m, t2)
    m = tl.maximum(m, t3)
    m = tl.maximum(m, t4)
    m = tl.maximum(m, t5)
    m_safe = tl.where(m > M_SAFE, m, tl.zeros_like(m))
    e0 = tl.exp2(t0 - m_safe)
    e1 = tl.exp2(t1 - m_safe)
    e2 = tl.exp2(t2 - m_safe)
    e3 = tl.exp2(t3 - m_safe)
    e4 = tl.exp2(t4 - m_safe)
    e5 = tl.exp2(t5 - m_safe)
    dts_l_sum = e0 + e1 + e2 + e3 + e4 + e5
    inv_sum = tl.where(dts_l_sum > 0.0, 1.0 / dts_l_sum, tl.zeros_like(dts_l_sum))

    aw0 = v_k_val * e0 * inv_sum
    aw1 = v_k_val * e1 * inv_sum
    aw2 = v_k_val * e2 * inv_sum
    aw3 = v_k_val * e3 * inv_sum
    aw4 = v_k_val * e4 * inv_sum
    aw5 = v_k_val * e5 * inv_sum

    zero = tl.zeros([TILE_ROWS, BLOCK_S], dtype=DTYPE)
    sum_aw0 = tl.sum(tl.where(mask, aw0, zero), axis=0)
    sum_aw1 = tl.sum(tl.where(mask, aw1, zero), axis=0)
    sum_aw2 = tl.sum(tl.where(mask, aw2, zero), axis=0)
    sum_aw3 = tl.sum(tl.where(mask, aw3, zero), axis=0)
    sum_aw4 = tl.sum(tl.where(mask, aw4, zero), axis=0)
    sum_aw5 = tl.sum(tl.where(mask, aw5, zero), axis=0)

    partial_base = tile * S + s_offs
    partial_stride = N_TILES * S
    tl.store(partial_vec_ptr + partial_base, sum_aw0 + sum_aw2, mask=valid_s)
    tl.store(partial_vec_ptr + partial_stride + partial_base, sum_aw1, mask=valid_s)
    tl.store(partial_vec_ptr + 2 * partial_stride + partial_base, sum_aw4, mask=valid_s)
    tl.store(partial_vec_ptr + 3 * partial_stride + partial_base, sum_aw3, mask=valid_s)
    tl.store(partial_vec_ptr + 4 * partial_stride + partial_base, sum_aw2, mask=valid_s)

    scalar_idx = tile * N_S_BLOCKS + s_block
    scalar_count = N_TILES * N_S_BLOCKS
    grad_pd = tl.sum(tl.where(valid_s, sum_aw0, tl.zeros([BLOCK_S], dtype=DTYPE)), axis=0)
    grad_ps = tl.sum(
        tl.where(valid_s, sum_aw3 + sum_aw4 + sum_aw5, tl.zeros([BLOCK_S], dtype=DTYPE)),
        axis=0,
    )
    tl.store(partial_scalar_ptr + scalar_idx, grad_pd)
    tl.store(partial_scalar_ptr + scalar_count + scalar_idx, grad_ps)


@triton.jit
def _wave_backward_uniform_param_stage2_kernel(
    partial_vec_ptr,      # [5, N_TILES, S]
    partial_scalar_ptr,   # [2, N_TILES, N_S_BLOCKS]
    grad_log_pD_ptr,
    grad_log_pS_ptr,
    grad_E_ptr,
    grad_Ebar_ptr,
    grad_E_s1_ptr,
    grad_E_s2_ptr,
    grad_mt_ptr,
    S: tl.constexpr,
    N_TILES: tl.constexpr,
    N_S_BLOCKS: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_SCALAR_PARTIALS: tl.constexpr,
    DTYPE: tl.constexpr,
):
    """Reduce compact no-split self-loop parameter partials into gradients."""
    s_block = tl.program_id(0)
    s_offs = s_block * BLOCK_S + tl.arange(0, BLOCK_S)
    valid_s = s_offs < S

    acc_E = tl.zeros([BLOCK_S], dtype=DTYPE)
    acc_Ebar = tl.zeros([BLOCK_S], dtype=DTYPE)
    acc_E_s1 = tl.zeros([BLOCK_S], dtype=DTYPE)
    acc_E_s2 = tl.zeros([BLOCK_S], dtype=DTYPE)
    acc_mt = tl.zeros([BLOCK_S], dtype=DTYPE)
    partial_stride = N_TILES * S

    tile_start = 0
    while tile_start < N_TILES:
        tile_offs = tile_start + tl.arange(0, BLOCK_TILES)
        mask = (tile_offs[:, None] < N_TILES) & valid_s[None, :]
        partial_offs = tile_offs[:, None] * S + s_offs[None, :]
        acc_E += tl.sum(tl.load(partial_vec_ptr + partial_offs, mask=mask, other=0.0), axis=0)
        acc_Ebar += tl.sum(
            tl.load(partial_vec_ptr + partial_stride + partial_offs, mask=mask, other=0.0),
            axis=0,
        )
        acc_E_s1 += tl.sum(
            tl.load(partial_vec_ptr + 2 * partial_stride + partial_offs, mask=mask, other=0.0),
            axis=0,
        )
        acc_E_s2 += tl.sum(
            tl.load(partial_vec_ptr + 3 * partial_stride + partial_offs, mask=mask, other=0.0),
            axis=0,
        )
        acc_mt += tl.sum(
            tl.load(partial_vec_ptr + 4 * partial_stride + partial_offs, mask=mask, other=0.0),
            axis=0,
        )
        tile_start += BLOCK_TILES

    tl.store(
        grad_E_ptr + s_offs,
        tl.load(grad_E_ptr + s_offs, mask=valid_s, other=0.0) + acc_E,
        mask=valid_s,
    )
    tl.store(
        grad_Ebar_ptr + s_offs,
        tl.load(grad_Ebar_ptr + s_offs, mask=valid_s, other=0.0) + acc_Ebar,
        mask=valid_s,
    )
    tl.store(
        grad_E_s1_ptr + s_offs,
        tl.load(grad_E_s1_ptr + s_offs, mask=valid_s, other=0.0) + acc_E_s1,
        mask=valid_s,
    )
    tl.store(
        grad_E_s2_ptr + s_offs,
        tl.load(grad_E_s2_ptr + s_offs, mask=valid_s, other=0.0) + acc_E_s2,
        mask=valid_s,
    )
    tl.store(
        grad_mt_ptr + s_offs,
        tl.load(grad_mt_ptr + s_offs, mask=valid_s, other=0.0) + acc_mt,
        mask=valid_s,
    )

    if s_block == 0:
        scalar_count = N_TILES * N_S_BLOCKS
        acc_pD = tl.zeros([1], dtype=DTYPE)
        acc_pS = tl.zeros([1], dtype=DTYPE)
        scalar_start = 0
        while scalar_start < scalar_count:
            scalar_offs = scalar_start + tl.arange(0, BLOCK_SCALAR_PARTIALS)
            scalar_mask = scalar_offs < scalar_count
            pD_vals = tl.load(partial_scalar_ptr + scalar_offs, mask=scalar_mask, other=0.0)
            pS_vals = tl.load(
                partial_scalar_ptr + scalar_count + scalar_offs,
                mask=scalar_mask,
                other=0.0,
            )
            acc_pD += tl.sum(pD_vals, axis=0)
            acc_pS += tl.sum(pS_vals, axis=0)
            scalar_start += BLOCK_SCALAR_PARTIALS

        scalar = tl.arange(0, 1)
        tl.store(grad_log_pD_ptr + scalar, tl.load(grad_log_pD_ptr + scalar) + acc_pD)
        tl.store(grad_log_pS_ptr + scalar, tl.load(grad_log_pS_ptr + scalar) + acc_pS)


def _wave_backward_uniform_param_two_stage(
    Pi_star,
    Pibar_star,
    ws,
    W,
    S,
    v_k,
    mt_squeezed,
    DL_const,
    Ebar,
    E,
    SL1_const,
    SL2_const,
    sp_child1,
    sp_child2,
    leaf_term_wt,
    leaf_species_idx,
    leaf_logp,
    accum_param_grads,
    *,
    active_mask=None,
    use_leaf_index=False,
    leaf_hit_only_logp=False,
    leaf_logp_scalar=False,
    scratch=None,
):
    tile_rows_raw = max(
        1,
        int(os.environ.get("GPUREC_SELF_LOOP_PARAM_TWO_STAGE_TILE_ROWS", "16")),
    )
    tile_rows = triton.next_power_of_2(tile_rows_raw)
    block_s_default = min(256, triton.next_power_of_2(S))
    block_s_raw = max(
        1,
        int(os.environ.get("GPUREC_SELF_LOOP_PARAM_TWO_STAGE_BLOCK_S", str(block_s_default))),
    )
    block_s = min(triton.next_power_of_2(block_s_raw), triton.next_power_of_2(S))
    reduce_block_tiles_raw = max(
        1,
        int(os.environ.get("GPUREC_SELF_LOOP_PARAM_TWO_STAGE_REDUCE_TILES", "16")),
    )
    reduce_block_tiles = triton.next_power_of_2(reduce_block_tiles_raw)
    scalar_block_raw = max(
        1,
        int(os.environ.get("GPUREC_SELF_LOOP_PARAM_TWO_STAGE_SCALAR_BLOCK", "1024")),
    )
    scalar_block = triton.next_power_of_2(scalar_block_raw)
    n_tiles = triton.cdiv(W, tile_rows)
    n_s_blocks = triton.cdiv(S, block_s)

    partial_vec = _scratch_view(
        scratch, "param_two_stage_vec", (5, n_tiles, S),
        device=Pi_star.device, dtype=Pi_star.dtype,
    )
    if partial_vec is None:
        partial_vec = torch.empty((5, n_tiles, S), device=Pi_star.device, dtype=Pi_star.dtype)
    partial_scalar = _scratch_view(
        scratch, "param_two_stage_scalar", (2, n_tiles, n_s_blocks),
        device=Pi_star.device, dtype=Pi_star.dtype,
    )
    if partial_scalar is None:
        partial_scalar = torch.empty(
            (2, n_tiles, n_s_blocks), device=Pi_star.device, dtype=Pi_star.dtype
        )

    (
        grad_log_pD,
        grad_log_pS,
        grad_E,
        grad_Ebar,
        grad_E_s1,
        grad_E_s2,
        grad_mt,
    ) = accum_param_grads

    stage1_warps = int(
        os.environ.get(
            "GPUREC_SELF_LOOP_PARAM_TWO_STAGE_NUM_WARPS",
            "8" if tile_rows * block_s >= 1024 else "4",
        )
    )
    stage1_options = {}
    if stage1_warps > 0:
        stage1_options["num_warps"] = stage1_warps

    _wave_backward_uniform_param_stage1_kernel[(n_tiles, n_s_blocks)](
        Pi_star,
        Pibar_star,
        v_k,
        active_mask if active_mask is not None else v_k,
        mt_squeezed,
        DL_const,
        Ebar,
        E,
        SL1_const,
        SL2_const,
        sp_child1,
        sp_child2,
        leaf_term_wt,
        leaf_species_idx,
        leaf_logp,
        partial_vec,
        partial_scalar,
        ws,
        W,
        S,
        Pi_star.stride(0),
        n_tiles,
        n_s_blocks,
        tile_rows,
        block_s,
        USE_LEAF_INDEX=bool(use_leaf_index),
        LEAF_HIT_ONLY_LOGP=bool(leaf_hit_only_logp),
        LEAF_LOGP_SCALAR=bool(leaf_logp_scalar),
        USE_ACTIVE_MASK=bool(active_mask is not None),
        DTYPE=_tl_float_dtype(Pi_star.dtype),
        **stage1_options,
    )

    stage2_warps = int(os.environ.get("GPUREC_SELF_LOOP_PARAM_TWO_STAGE_REDUCE_WARPS", "4"))
    stage2_options = {}
    if stage2_warps > 0:
        stage2_options["num_warps"] = stage2_warps
    _wave_backward_uniform_param_stage2_kernel[(n_s_blocks,)](
        partial_vec,
        partial_scalar,
        grad_log_pD,
        grad_log_pS,
        grad_E,
        grad_Ebar,
        grad_E_s1,
        grad_E_s2,
        grad_mt,
        S,
        n_tiles,
        n_s_blocks,
        reduce_block_tiles,
        block_s,
        scalar_block,
        DTYPE=_tl_float_dtype(Pi_star.dtype),
        **stage2_options,
    )


def wave_backward_uniform_fused(
    Pi_star, Pibar_star, ws, W, S,
    dts_r,
    rhs,
    mt_squeezed, DL_const, Ebar, E, SL1_const, SL2_const,
    sp_child1, sp_child2, leaf_term_wt,
    neumann_terms=3,
    leaf_species_idx=None,
    leaf_logp=None,
    accum_param_grads=None,
    active_mask=None,
    sp_parent=None,
    max_ancestor_depth=None,
    pibar_row_max=None,
    skip_inactive_zero_stores=False,
    scratch=None,
    family_idx=None,
    family_indexed_consts=False,
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
        accum_param_grads: optional tuple of seven tensors
            (grad_log_pD, grad_log_pS, grad_E, grad_Ebar,
             grad_E_s1, grad_E_s2, grad_mt). When provided, the kernel
            atomically accumulates param VJP results and returns None for
            the per-element contribution tensors.

    Returns:
        v_k: [W, S] Neumann-solved adjoint
        aw0, aw1, aw2, aw345, aw3, aw4: [W, S] per-element param grad contributions
    """
    device = Pi_star.device
    dtype = Pi_star.dtype

    accum_enabled = accum_param_grads is not None
    has_splits = dts_r is not None
    use_leaf_index = leaf_species_idx is not None and leaf_logp is not None
    const_layout = _uniform_backward_const_layout(
        DL_const, family_idx, bool(family_indexed_consts)
    )
    if accum_enabled and const_layout == 1:
        raise ValueError(
            "in-kernel parameter accumulation is not supported for row-expanded "
            "backward constants"
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
        family_idx = family_idx.to(device=device, dtype=torch.long).contiguous()
    if sp_parent is None:
        sp_parent = _parent_from_children(sp_child1, sp_child2, S)
    else:
        sp_parent = sp_parent.to(device=device).contiguous()
    if device.type == "cuda" and sp_parent.dtype != torch.int32:
        sp_parent = sp_parent.to(dtype=torch.int32)
    if max_ancestor_depth is None:
        max_ancestor_depth = _max_ancestor_depth_from_parent(sp_parent, S)
    max_ancestor_depth = max(1, int(max_ancestor_depth))
    fast_nosplit_param_grads = (
        os.environ.get("GPUREC_FAST_NOSPLIT_PARAM_ACCUM", "0") != "0"
    )
    recompute_pibar_denom = (
        os.environ.get("GPUREC_RECOMPUTE_PIBAR_DENOM", "0") != "0"
    )
    # The old recompute path used a self-only denominator. Keep one exact
    # denominator pass and store/reuse it instead of allowing that approximation.
    recompute_pibar_denom = False
    compact_pibar_scratch_mode = (
        os.environ.get("GPUREC_COMPACT_PIBAR_SCRATCH", "1")
        .strip()
        .lower()
    )
    compact_pibar_scratch = compact_pibar_scratch_mode not in (
        "0", "false", "no", "off", ""
    )
    if compact_pibar_scratch_mode in ("leaf", "nosplit", "no_split"):
        compact_pibar_scratch = not has_splits
    if compact_pibar_scratch:
        recompute_pibar_denom = False
    leaf_hit_only_logp = (
        os.environ.get("GPUREC_LEAF_HIT_ONLY_LOGP", "0") != "0"
    )
    leaf_logp_scalar = leaf_logp_mode == 3
    # Parent-gather speciation was an optimization for the old self-only
    # correction. With the full ancestor correction it is not exact enough on
    # all waves, so keep the scatter path as the only production path.
    spec_gather = False
    if pibar_row_max is None:
        pibar_row_max = Pi_star.max(dim=1).values.contiguous()
    else:
        pibar_row_max = pibar_row_max.to(device=device, dtype=dtype).contiguous()
    use_pibar_row_max = True
    param_two_stage_mode = (
        os.environ.get("GPUREC_SELF_LOOP_PARAM_TWO_STAGE", "0").strip().lower()
    )
    param_two_stage_requested = param_two_stage_mode not in (
        "", "0", "false", "no", "off"
    )
    param_two_stage_enabled = False
    if (
        param_two_stage_requested
        and accum_enabled
        and not has_splits
        and const_layout == 0
    ):
        (
            grad_log_pD_arg,
            grad_log_pS_arg,
            grad_E_arg,
            grad_Ebar_arg,
            grad_E_s1_arg,
            grad_E_s2_arg,
            grad_mt_arg,
        ) = accum_param_grads
        param_two_stage_enabled = (
            grad_log_pD_arg.numel() == 1
            and grad_log_pS_arg.numel() == 1
            and grad_E_arg.numel() == S
            and grad_Ebar_arg.numel() == S
            and grad_E_s1_arg.numel() == S
            and grad_E_s2_arg.numel() == S
            and grad_mt_arg.numel() == S
        )

    scratch_shape = (W, S)

    v_k = _scratch_view(scratch, "v_k", scratch_shape, device=device, dtype=dtype)
    if v_k is None:
        v_k = torch.empty(scratch_shape, device=device, dtype=dtype)
    aw0 = _scratch_view(scratch, "aw0", scratch_shape, device=device, dtype=dtype)
    if aw0 is None:
        aw0 = torch.empty(scratch_shape, device=device, dtype=dtype)
    aw1 = _scratch_view(scratch, "aw1", scratch_shape, device=device, dtype=dtype)
    if aw1 is None:
        aw1 = torch.empty(scratch_shape, device=device, dtype=dtype)
    need_pibar_denom_scratch = not (
        accum_enabled and (compact_pibar_scratch or recompute_pibar_denom)
    )
    if need_pibar_denom_scratch:
        aw2 = _scratch_view(scratch, "aw2", scratch_shape, device=device, dtype=dtype)
        if aw2 is None:
            aw2 = torch.empty(scratch_shape, device=device, dtype=dtype)
    else:
        aw2 = aw0
    aw345 = _scratch_view(scratch, "aw345", scratch_shape, device=device, dtype=dtype)
    if aw345 is None:
        aw345 = torch.empty(scratch_shape, device=device, dtype=dtype)
    if need_pibar_denom_scratch:
        aw3 = _scratch_view(scratch, "aw3", scratch_shape, device=device, dtype=dtype)
        if aw3 is None:
            aw3 = torch.empty(scratch_shape, device=device, dtype=dtype)
    else:
        aw3 = aw0
    aw4 = _scratch_view(scratch, "aw4", scratch_shape, device=device, dtype=dtype)
    if aw4 is None:
        aw4 = torch.empty(scratch_shape, device=device, dtype=dtype)
    spec_buf = _scratch_view(scratch, "spec_buf", scratch_shape, device=device, dtype=dtype)
    if spec_buf is None:
        spec_buf = torch.empty(scratch_shape, device=device, dtype=dtype)
    term_buf = _scratch_view(scratch, "term_buf", scratch_shape, device=device, dtype=dtype)
    if term_buf is None:
        term_buf = torch.empty(scratch_shape, device=device, dtype=dtype)
    pibar_corr_buf = _scratch_view(
        scratch, "pibar_corr", scratch_shape, device=device, dtype=dtype
    )
    if pibar_corr_buf is None:
        pibar_corr_buf = torch.empty(scratch_shape, device=device, dtype=dtype)

    if leaf_term_wt is None:
        if not use_leaf_index:
            raise ValueError("leaf_term_wt is required when leaf_species_idx/leaf_logp are not provided")
        leaf_term_wt = leaf_logp
    leaf_species_arg = leaf_species_idx if use_leaf_index else sp_child1
    leaf_logp_arg = leaf_logp if use_leaf_index else leaf_term_wt
    if accum_enabled:
        (
            grad_log_pD_arg,
            grad_log_pS_arg,
            grad_E_arg,
            grad_Ebar_arg,
            grad_E_s1_arg,
            grad_E_s2_arg,
            grad_mt_arg,
        ) = accum_param_grads
    else:
        grad_log_pD_arg = grad_log_pS_arg = aw0
        grad_E_arg = grad_Ebar_arg = grad_E_s1_arg = grad_E_s2_arg = grad_mt_arg = aw0
    log_pD_grad_scalar = _grad_param_is_scalar(grad_log_pD_arg)
    log_pS_grad_scalar = _grad_param_is_scalar(grad_log_pS_arg)
    family_idx_arg = family_idx if family_idx is not None else sp_parent

    block_s_env = os.environ.get("GPUREC_WAVE_BLOCK_S", "").strip()
    if block_s_env:
        BLOCK_S = int(block_s_env)
    else:
        BLOCK_S = min(256, triton.next_power_of_2(S))
    num_warps_env = os.environ.get(
        "GPUREC_WAVE_NUM_WARPS",
        "8" if spec_gather else "",
    ).strip()
    launch_options = {}
    if num_warps_env:
        num_warps = int(num_warps_env)
        if num_warps > 0:
            launch_options["num_warps"] = num_warps

    grid = (W,)
    _wave_backward_uniform_kernel[grid](
        Pi_star, Pibar_star,
        pibar_row_max if use_pibar_row_max else Pi_star,
        dts_r if has_splits else Pi_star,  # dummy ptr when no splits
        has_splits,
        rhs,
        active_mask if active_mask is not None else rhs,
        mt_squeezed, DL_const, Ebar, E, SL1_const, SL2_const,
        sp_child1, sp_child2,
        sp_parent,
        leaf_term_wt,
        leaf_species_arg,
        leaf_logp_arg,
        family_idx_arg,
        v_k,
        aw0, aw1, aw2, aw345, aw3, aw4,
        spec_buf,
        term_buf,
        pibar_corr_buf,
        grad_log_pD_arg,
        grad_log_pS_arg,
        grad_E_arg,
        grad_Ebar_arg,
        grad_E_s1_arg,
        grad_E_s2_arg,
        grad_mt_arg,
        ws, S, S, BLOCK_S,
        neumann_terms,
        max_ancestor_depth,
        USE_LEAF_INDEX=bool(use_leaf_index),
        ACCUM_PARAM_GRADS=bool(accum_enabled),
        PARAM_GRAD_TWO_STAGE=bool(param_two_stage_enabled),
        FAST_NOSPLIT_PARAM_GRADS=bool(fast_nosplit_param_grads),
        COMPACT_PIBAR_SCRATCH=bool(compact_pibar_scratch),
        RECOMPUTE_PIBAR_DENOM=bool(recompute_pibar_denom),
        LEAF_HIT_ONLY_LOGP=bool(leaf_hit_only_logp),
        LEAF_LOGP_MODE=int(leaf_logp_mode),
        USE_PIBAR_ROW_MAX=bool(use_pibar_row_max),
        SPEC_GATHER=bool(spec_gather),
        USE_ACTIVE_MASK=bool(active_mask is not None),
        SKIP_INACTIVE_ZERO_STORES=bool(skip_inactive_zero_stores),
        CONST_LAYOUT=int(const_layout),
        LOG_PD_GRAD_SCALAR=bool(log_pD_grad_scalar),
        LOG_PS_GRAD_SCALAR=bool(log_pS_grad_scalar),
        DTYPE=_tl_float_dtype(dtype),
        **launch_options,
    )

    if param_two_stage_enabled:
        _wave_backward_uniform_param_two_stage(
            Pi_star,
            Pibar_star,
            ws,
            W,
            S,
            v_k,
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
            accum_param_grads,
            active_mask=active_mask,
            use_leaf_index=use_leaf_index,
            leaf_hit_only_logp=leaf_hit_only_logp,
            leaf_logp_scalar=leaf_logp_scalar,
            scratch=scratch,
        )

    if accum_enabled:
        return v_k, None, None, None, None, None, None
    return v_k, aw0, aw1, aw2, aw345, aw3, aw4


# =========================================================================
# Cross-clade DTS backward kernel
# =========================================================================

@triton.jit
def _dts_cross_backward_kernel(
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
    # Scalar params
    log_pD_arg,
    log_pS_arg,
    # Species children [S] int64
    sp_child1_ptr,
    sp_child2_ptr,
    # Outputs [n_ws, S]
    grad_Pi_l_ptr,
    grad_Pi_r_ptr,
    grad_Pibar_l_ptr,
    grad_Pibar_r_ptr,
    # Per-split param sums [n_ws]
    param_pD_ptr,
    param_pS_ptr,
    # Dimensions
    ws,                # wave start offset (parent row = ws + reduce_idx)
    S: tl.constexpr,
    stride_C: tl.constexpr,
    BLOCK_S: tl.constexpr,
    USE_ACTIVE_MASK: tl.constexpr,
    DEVICE_SCALAR_PARAMS: tl.constexpr,
    DTYPE: tl.constexpr,
):
    """Fused cross-clade DTS backward for uniform Pibar mode.

    For each split i, computes the VJP of the 5 cross-clade DTS terms.

    Key simplification: the chain rule through segment-logsumexp, 5-term
    logsumexp, and the DTS_L/dts_r mixing collapses to a single weight:

        v_DTS_5[t, i, s] = v_k[parent, s] * exp2(wlsp[i] + DTS_5[t,i,s] - Pi_parent[s])

    This avoids materializing intermediate [5, n_ws, S] tensors.

    Pass 1: compute direct Pi/Pibar gradients, accumulate param sums.
    Pass 2: scatter speciation gradients to child species positions.
    """
    NEG_LARGE: tl.constexpr = -1e30

    i = tl.program_id(0)  # split index

    # Load split metadata (scalar per CTA)
    sl = tl.load(sl_ptr + i).to(tl.int64)
    sr = tl.load(sr_ptr + i).to(tl.int64)
    parent_w = tl.load(reduce_idx_ptr + i).to(tl.int64)
    wlsp = tl.load(wlsp_ptr + i)
    if USE_ACTIVE_MASK:
        parent_active = tl.load(active_mask_ptr + parent_w)
        if parent_active == 0:
            out_base = i * S
            zero_scalar = tl.zeros((1,), dtype=DTYPE)
            _scalar_off = tl.arange(0, 1)
            tl.store(param_pD_ptr + i + _scalar_off, zero_scalar)
            tl.store(param_pS_ptr + i + _scalar_off, zero_scalar)
            for s_start in range(0, S, BLOCK_S):
                s_offs = s_start + tl.arange(0, BLOCK_S)
                mask = s_offs < S
                zero = tl.zeros([BLOCK_S], dtype=DTYPE)
                tl.store(grad_Pi_l_ptr + out_base + s_offs, zero, mask=mask)
                tl.store(grad_Pi_r_ptr + out_base + s_offs, zero, mask=mask)
                tl.store(grad_Pibar_l_ptr + out_base + s_offs, zero, mask=mask)
                tl.store(grad_Pibar_r_ptr + out_base + s_offs, zero, mask=mask)
            return
    else:
        parent_active = True

    if DEVICE_SCALAR_PARAMS:
        log_pD = tl.load(log_pD_arg).to(DTYPE)
        log_pS = tl.load(log_pS_arg).to(DTYPE)
    else:
        log_pD = log_pD_arg
        log_pS = log_pS_arg

    parent_global = ws + parent_w
    # Base offsets into [C, S] for child clades
    pi_l_base = sl * stride_C
    pi_r_base = sr * stride_C
    pibar_l_base = sl * stride_C
    pibar_r_base = sr * stride_C
    # Parent clade in Pi_star: row (ws + parent_w)
    parent_pi_base = parent_global * stride_C
    # v_k is [W, S] contiguous, indexed by parent_w
    parent_vk_base = parent_w * S
    # Output row
    out_base = i * S

    # Accumulators for per-split param sums (1-element blocks for Triton compatibility)
    sum_pD = tl.zeros((1,), dtype=DTYPE)
    sum_pS = tl.zeros((1,), dtype=DTYPE)
    _scalar_off = tl.arange(0, 1)  # for storing 1-element blocks

    # ================================================================
    # Pass 1: Direct contributions + param sums
    # ================================================================
    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        valid_mask = s_offs < S
        mask = valid_mask & parent_active

        # Load child Pi/Pibar
        Pi_l = tl.load(Pi_star_ptr + pi_l_base + s_offs, mask=mask, other=NEG_LARGE)
        Pi_r = tl.load(Pi_star_ptr + pi_r_base + s_offs, mask=mask, other=NEG_LARGE)
        Pibar_l = tl.load(Pibar_star_ptr + pibar_l_base + s_offs, mask=mask, other=NEG_LARGE)
        Pibar_r = tl.load(Pibar_star_ptr + pibar_r_base + s_offs, mask=mask, other=NEG_LARGE)

        # Species child gathers (for speciation terms)
        c1 = tl.load(sp_child1_ptr + s_offs, mask=mask, other=0)
        c2 = tl.load(sp_child2_ptr + s_offs, mask=mask, other=0)
        c1_valid = (c1 < S) & mask
        c2_valid = (c2 < S) & mask
        Pi_l_s1 = tl.load(Pi_star_ptr + pi_l_base + c1, mask=c1_valid, other=NEG_LARGE)
        Pi_l_s2 = tl.load(Pi_star_ptr + pi_l_base + c2, mask=c2_valid, other=NEG_LARGE)
        Pi_r_s1 = tl.load(Pi_star_ptr + pi_r_base + c1, mask=c1_valid, other=NEG_LARGE)
        Pi_r_s2 = tl.load(Pi_star_ptr + pi_r_base + c2, mask=c2_valid, other=NEG_LARGE)

        # Load parent's Pi and v_k
        Pi_parent = tl.load(Pi_star_ptr + parent_pi_base + s_offs, mask=mask, other=NEG_LARGE)
        v_k_val = tl.load(v_k_ptr + parent_vk_base + s_offs, mask=mask, other=0.0)

        # DTS_5 terms
        d0 = log_pD + Pi_l + Pi_r           # D: duplication
        d1 = Pi_l + Pibar_r                 # T: transfer l→r
        d2 = Pi_r + Pibar_l                 # T: transfer r→l
        d3 = log_pS + Pi_l_s1 + Pi_r_s2    # S: speciation (c1 in l, c2 in r)
        d4 = log_pS + Pi_r_s1 + Pi_l_s2    # S: speciation (c1 in r, c2 in l)

        # Simplified weight: v_DTS_5[t] = v_k * exp2(wlsp + DTS_5[t] - Pi_parent)
        # Pi_parent >= wlsp + DTS_5[t], so exponent <= 0 and result in [0, 1].
        # Guard: when Pi_parent = -inf, all weights are 0.
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

        # Direct contributions to Pi gradients (D + T terms only; S terms via scatter in pass 2)
        tl.store(grad_Pi_l_ptr + out_base + s_offs, vd0 + vd1, mask=valid_mask)
        tl.store(grad_Pi_r_ptr + out_base + s_offs, vd0 + vd2, mask=valid_mask)
        tl.store(grad_Pibar_l_ptr + out_base + s_offs, vd2, mask=valid_mask)
        tl.store(grad_Pibar_r_ptr + out_base + s_offs, vd1, mask=valid_mask)

        # Accumulate param sums
        sum_pD += tl.sum(vd0, axis=0)
        sum_pS += tl.sum(vd3 + vd4, axis=0)

    # Store per-split param sums (use block pointer for compatibility)
    tl.store(param_pD_ptr + i + _scalar_off, sum_pD)
    tl.store(param_pS_ptr + i + _scalar_off, sum_pS)

    # ================================================================
    # Pass 2: Scatter speciation contributions to child species positions
    #
    # DTS[3] reads Pi_l[child1[s]] and Pi_r[child2[s]], so:
    #   grad_Pi_l[child1[s]] += vd3[s],  grad_Pi_r[child2[s]] += vd3[s]
    # DTS[4] reads Pi_r[child1[s]] and Pi_l[child2[s]], so:
    #   grad_Pi_r[child1[s]] += vd4[s],  grad_Pi_l[child2[s]] += vd4[s]
    #
    # Each CTA owns its output row. sp_child1/sp_child2 are injective
    # (each child has one parent), so read-modify-write is race-free.
    # ================================================================
    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        valid_mask = s_offs < S
        mask = valid_mask & parent_active

        # Recompute vd3, vd4 (speciation terms only)
        c1 = tl.load(sp_child1_ptr + s_offs, mask=mask, other=0)
        c2 = tl.load(sp_child2_ptr + s_offs, mask=mask, other=0)
        c1_valid = (c1 < S) & mask
        c2_valid = (c2 < S) & mask

        Pi_l_s1 = tl.load(Pi_star_ptr + pi_l_base + c1, mask=c1_valid, other=NEG_LARGE)
        Pi_l_s2 = tl.load(Pi_star_ptr + pi_l_base + c2, mask=c2_valid, other=NEG_LARGE)
        Pi_r_s1 = tl.load(Pi_star_ptr + pi_r_base + c1, mask=c1_valid, other=NEG_LARGE)
        Pi_r_s2 = tl.load(Pi_star_ptr + pi_r_base + c2, mask=c2_valid, other=NEG_LARGE)

        Pi_parent = tl.load(Pi_star_ptr + parent_pi_base + s_offs, mask=mask, other=NEG_LARGE)
        v_k_val = tl.load(v_k_ptr + parent_vk_base + s_offs, mask=mask, other=0.0)

        d3 = log_pS + Pi_l_s1 + Pi_r_s2
        d4 = log_pS + Pi_r_s1 + Pi_l_s2

        parent_valid = Pi_parent > NEG_LARGE
        w3 = tl.where(parent_valid, tl.exp2(wlsp + d3 - Pi_parent), tl.zeros_like(d3))
        w4 = tl.where(parent_valid, tl.exp2(wlsp + d4 - Pi_parent), tl.zeros_like(d4))
        vd3 = v_k_val * w3
        vd4 = v_k_val * w4

        # Scatter to child1 positions
        # grad_Pi_l[child1[s]] += vd3[s]
        cur = tl.load(grad_Pi_l_ptr + out_base + c1, mask=c1_valid, other=0.0)
        tl.store(grad_Pi_l_ptr + out_base + c1, cur + vd3, mask=c1_valid)
        # grad_Pi_r[child1[s]] += vd4[s]
        cur = tl.load(grad_Pi_r_ptr + out_base + c1, mask=c1_valid, other=0.0)
        tl.store(grad_Pi_r_ptr + out_base + c1, cur + vd4, mask=c1_valid)

        # Scatter to child2 positions
        # grad_Pi_r[child2[s]] += vd3[s]
        cur = tl.load(grad_Pi_r_ptr + out_base + c2, mask=c2_valid, other=0.0)
        tl.store(grad_Pi_r_ptr + out_base + c2, cur + vd3, mask=c2_valid)
        # grad_Pi_l[child2[s]] += vd4[s]
        cur = tl.load(grad_Pi_l_ptr + out_base + c2, mask=c2_valid, other=0.0)
        tl.store(grad_Pi_l_ptr + out_base + c2, cur + vd4, mask=c2_valid)


def dts_cross_backward_fused(
    Pi_star, Pibar_star, v_k, ws,
    sl, sr, reduce_idx, wlsp,
    log_pD, log_pS,
    sp_child1, sp_child2,
    S,
    active_mask=None,
):
    """Fused DTS cross-clade backward: replaces both param-grad and adjoint blocks.

    Args:
        Pi_star: [C, S] converged Pi (full, not just wave slice)
        Pibar_star: [C, S] converged Pibar
        v_k: [W, S] Neumann-solved adjoint for this wave
        ws: wave start offset (int)
        sl: [n_ws] int64 — left child clade indices
        sr: [n_ws] int64 — right child clade indices
        reduce_idx: [n_ws] int64 — wave-local parent indices
        wlsp: [n_ws, 1] or [n_ws] — log split probabilities
        log_pD: scalar float — log2 duplication probability
        log_pS: scalar float — log2 speciation probability
        sp_child1, sp_child2: [S] int64 — species child indices
        S: int — number of species

    Returns:
        grad_Pi_l: [n_ws, S] gradient to Pi at left child clades (includes speciation scatter)
        grad_Pi_r: [n_ws, S] gradient to Pi at right child clades (includes speciation scatter)
        grad_Pibar_l: [n_ws, S] gradient to Pibar at left child clades
        grad_Pibar_r: [n_ws, S] gradient to Pibar at right child clades
        param_pD: [n_ws] per-split sum of v_DTS_5[0] (duplication param grad)
        param_pS: [n_ws] per-split sum of v_DTS_5[3]+v_DTS_5[4] (speciation param grad)
    """
    n_ws = sl.shape[0]
    device = Pi_star.device
    dtype = Pi_star.dtype

    # Squeeze wlsp to [n_ws]
    wlsp_flat = wlsp.squeeze(-1) if wlsp.ndim > 1 else wlsp

    log_pD_arg, log_pS_arg, device_scalar_params = _dts_scalar_param_args(
        log_pD, log_pS, device=device, dtype=dtype
    )

    # Allocate outputs
    grad_Pi_l = torch.empty((n_ws, S), device=device, dtype=dtype)
    grad_Pi_r = torch.empty((n_ws, S), device=device, dtype=dtype)
    grad_Pibar_l = torch.empty((n_ws, S), device=device, dtype=dtype)
    grad_Pibar_r = torch.empty((n_ws, S), device=device, dtype=dtype)
    param_pD = torch.empty(n_ws, device=device, dtype=dtype)
    param_pS = torch.empty(n_ws, device=device, dtype=dtype)

    stride_C = Pi_star.stride(0)
    BLOCK_S = min(256, triton.next_power_of_2(S))

    grid = (n_ws,)
    _dts_cross_backward_kernel[grid](
        Pi_star, Pibar_star,
        v_k,
        active_mask if active_mask is not None else v_k,
        sl, sr, reduce_idx, wlsp_flat,
        log_pD_arg, log_pS_arg,
        sp_child1, sp_child2,
        grad_Pi_l, grad_Pi_r, grad_Pibar_l, grad_Pibar_r,
        param_pD, param_pS,
        ws, S, stride_C, BLOCK_S,
        USE_ACTIVE_MASK=bool(active_mask is not None),
        DEVICE_SCALAR_PARAMS=bool(device_scalar_params),
        DTYPE=_tl_float_dtype(dtype),
    )

    return grad_Pi_l, grad_Pi_r, grad_Pibar_l, grad_Pibar_r, param_pD, param_pS


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

    This is the same VJP as _dts_cross_backward_kernel, but it writes direct
    Pi contributions into accumulated_rhs instead of materializing
    grad_Pi_l/grad_Pi_r and relying on two PyTorch index_add_ calls.
    Pibar adjoints are still materialized because they feed the uniform Pibar
    VJP kernel.
    """
    NEG_LARGE: tl.constexpr = -1e30

    i = tl.program_id(0)

    sl = tl.load(sl_ptr + i).to(tl.int64)
    sr = tl.load(sr_ptr + i).to(tl.int64)
    parent_w = tl.load(reduce_idx_ptr + i).to(tl.int64)
    wlsp = tl.load(wlsp_ptr + i)
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

        Pi_l = tl.load(Pi_star_ptr + pi_l_base + s_offs, mask=mask, other=NEG_LARGE)
        Pi_r = tl.load(Pi_star_ptr + pi_r_base + s_offs, mask=mask, other=NEG_LARGE)
        Pibar_l = tl.load(Pibar_star_ptr + pibar_l_base + s_offs, mask=mask, other=NEG_LARGE)
        Pibar_r = tl.load(Pibar_star_ptr + pibar_r_base + s_offs, mask=mask, other=NEG_LARGE)

        c1 = tl.load(sp_child1_ptr + s_offs, mask=mask, other=0)
        c2 = tl.load(sp_child2_ptr + s_offs, mask=mask, other=0)
        c1_valid = (c1 < S) & mask
        c2_valid = (c2 < S) & mask
        Pi_l_s1 = tl.load(Pi_star_ptr + pi_l_base + c1, mask=c1_valid, other=NEG_LARGE)
        Pi_l_s2 = tl.load(Pi_star_ptr + pi_l_base + c2, mask=c2_valid, other=NEG_LARGE)
        Pi_r_s1 = tl.load(Pi_star_ptr + pi_r_base + c1, mask=c1_valid, other=NEG_LARGE)
        Pi_r_s2 = tl.load(Pi_star_ptr + pi_r_base + c2, mask=c2_valid, other=NEG_LARGE)

        Pi_parent = tl.load(Pi_star_ptr + parent_pi_base + s_offs, mask=mask, other=NEG_LARGE)
        v_k_val = tl.load(v_k_ptr + parent_vk_base + s_offs, mask=mask, other=0.0)

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
            pi_l_cur = tl.load(pi_l_out, mask=mask, other=0.0)
            pi_r_cur = tl.load(pi_r_out, mask=mask, other=0.0)
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

            Pi_l_s1 = tl.load(Pi_star_ptr + pi_l_base + c1, mask=c1_valid, other=NEG_LARGE)
            Pi_l_s2 = tl.load(Pi_star_ptr + pi_l_base + c2, mask=c2_valid, other=NEG_LARGE)
            Pi_r_s1 = tl.load(Pi_star_ptr + pi_r_base + c1, mask=c1_valid, other=NEG_LARGE)
            Pi_r_s2 = tl.load(Pi_star_ptr + pi_r_base + c2, mask=c2_valid, other=NEG_LARGE)

            Pi_parent = tl.load(Pi_star_ptr + parent_pi_base + s_offs, mask=mask, other=NEG_LARGE)
            v_k_val = tl.load(v_k_ptr + parent_vk_base + s_offs, mask=mask, other=0.0)

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
                pi_l_c1_cur = tl.load(pi_l_c1_out, mask=c1_valid, other=0.0)
                pi_r_c1_cur = tl.load(pi_r_c1_out, mask=c1_valid, other=0.0)
                pi_r_c2_cur = tl.load(pi_r_c2_out, mask=c2_valid, other=0.0)
                pi_l_c2_cur = tl.load(pi_l_c2_out, mask=c2_valid, other=0.0)
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
    scratch=None,
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
        use_device_scalars = (
            os.environ.get("GPUREC_DTS_BACKWARD_DEVICE_SCALARS", "1") != "0"
        )
        if use_device_scalars:
            device_scalar_params = True
        else:
            log_pD_arg = float(log_pD_arg.item())
            log_pS_arg = float(log_pS_arg.item())

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
        grad_Pibar_l = _scratch_view(
            scratch, "grad_Pibar_l", (n_ws, S), device=device, dtype=dtype
        )
        if grad_Pibar_l is None:
            grad_Pibar_l = torch.empty((n_ws, S), device=device, dtype=dtype)
        grad_Pibar_r = _scratch_view(
            scratch, "grad_Pibar_r", (n_ws, S), device=device, dtype=dtype
        )
        if grad_Pibar_r is None:
            grad_Pibar_r = torch.empty((n_ws, S), device=device, dtype=dtype)
    if output_pibar_ud:
        pibar_ud = _scratch_view(
            scratch, "pibar_ud", (2 * n_ws, S), device=device, dtype=dtype
        )
        if pibar_ud is None:
            pibar_ud = torch.empty((2 * n_ws, S), device=device, dtype=dtype)
        pibar_A = _scratch_view(
            scratch, "pibar_A", (2 * n_ws,), device=device, dtype=dtype
        )
        if pibar_A is None:
            pibar_A = torch.empty((2 * n_ws,), device=device, dtype=dtype)
    else:
        pibar_ud = None
        pibar_A = None
    pibar_side_active = (
        _scratch_view(
            scratch, "pibar_side_active", (2 * n_ws,),
            device=device, dtype=torch.bool
        )
        if output_pibar_side_active
        else None
    )
    if output_pibar_side_active and pibar_side_active is None:
        pibar_side_active = torch.empty((2 * n_ws,), device=device, dtype=torch.bool)
    if accum_param_reductions:
        param_pD = None
        param_pS = None
    else:
        param_pD = _scratch_view(scratch, "param_pD", (n_ws,), device=device, dtype=dtype)
        if param_pD is None:
            param_pD = torch.empty(n_ws, device=device, dtype=dtype)
        param_pS = _scratch_view(scratch, "param_pS", (n_ws,), device=device, dtype=dtype)
        if param_pS is None:
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
    grad_mt_two_stage_tile_splits = max(1, int(grad_mt_two_stage_tile_splits))
    n_grad_mt_tiles = triton.cdiv(n_ws, grad_mt_two_stage_tile_splits)
    if use_grad_mt_two_stage:
        grad_mt_partial = _scratch_view(
            scratch, "grad_mt_partial", (n_grad_mt_tiles, S),
            device=device, dtype=dtype
        )
        if grad_mt_partial is None:
            grad_mt_partial = torch.empty((n_grad_mt_tiles, S), device=device, dtype=dtype)
        grad_mt_partial.zero_()
    else:
        grad_mt_partial = dummy

    stride_C = Pi_star.stride(0)
    BLOCK_S = min(256, triton.next_power_of_2(S))

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


@triton.jit
def _dts_cross_backward_accum_parent_tiled_ge2_kernel(
    # Converged values [C, S]
    Pi_star_ptr,
    Pibar_star_ptr,
    # Neumann-solved adjoint [W, S]
    v_k_ptr,
    active_mask_ptr,       # optional [W] bool parent row activity mask
    # Split metadata
    sl_ptr,                # [n_ws] int64, eq1 first then ge2 grouped by parent
    sr_ptr,
    reduce_idx_ptr,        # [n_ws] int64 wave-local parent index
    wlsp_ptr,              # [n_ws] float
    ge2_ptr,               # [n_ge2_groups + 1] CSR offsets for split ids after N_EQ1
    ge2_parent_ids_ptr,    # [n_ge2_groups] wave-local parent ids
    # Scalar params
    log_pD_arg,
    log_pS_arg,
    # Species children [S] int64
    sp_child1_ptr,
    sp_child2_ptr,
    # Outputs
    accumulated_rhs_ptr,   # [C, S], direct Pi adjoints updated atomically
    grad_log_pD_ptr,       # scalar accumulation target
    grad_log_pS_ptr,       # scalar accumulation target
    grad_mt_ptr,           # optional scalar/[S] accumulation target
    pibar_ud_ptr,          # [2 * N_WS, S]
    pibar_A_ptr,           # [2 * N_WS]
    mt_ptr,                # [S]
    pibar_row_max_ptr,     # [C]
    # Dimensions
    ws,
    S: tl.constexpr,
    stride_C: tl.constexpr,
    N_WS: tl.constexpr,
    N_EQ1: tl.constexpr,
    TILE_SPLITS: tl.constexpr,
    BLOCK_S: tl.constexpr,
    USE_ACTIVE_MASK: tl.constexpr,
    DEVICE_SCALAR_PARAMS: tl.constexpr,
    ACCUM_MT_REDUCTION: tl.constexpr,
    GRAD_MT_SCALAR: tl.constexpr,
    DTYPE: tl.constexpr,
):
    """Parent-tiled staged DTS backward for high-fanout waves."""
    NEG_LARGE: tl.constexpr = -1e30

    group = tl.program_id(0)
    tile = tl.program_id(1)
    block = tl.program_id(2)
    group_i64 = group.to(tl.int64)
    tile_i64 = tile.to(tl.int64)

    if group < N_EQ1:
        if tile != 0:
            return
        split_start = group_i64
        split_end = group_i64 + 1
        parent_w = tl.load(reduce_idx_ptr + group).to(tl.int64)
    else:
        ge2_group = group - N_EQ1
        start_ge2 = tl.load(ge2_ptr + ge2_group)
        end_ge2 = tl.load(ge2_ptr + ge2_group + 1)
        split_start = N_EQ1 + start_ge2 + tile_i64 * TILE_SPLITS
        split_end = N_EQ1 + tl.minimum(end_ge2, start_ge2 + (tile_i64 + 1) * TILE_SPLITS)
        parent_w = tl.load(ge2_parent_ids_ptr + ge2_group).to(tl.int64)

    if split_start >= split_end:
        return

    s_offs = block * BLOCK_S + tl.arange(0, BLOCK_S)
    valid_mask = s_offs < S
    scalar = tl.arange(0, 1)

    if USE_ACTIVE_MASK:
        parent_active = tl.load(active_mask_ptr + parent_w)
        if parent_active == 0:
            zero = tl.zeros([BLOCK_S], dtype=DTYPE)
            for j in tl.static_range(0, TILE_SPLITS):
                split_i = split_start + j
                split_valid = split_i < split_end
                tl.store(pibar_ud_ptr + split_i * S + s_offs, zero, mask=valid_mask & split_valid)
                tl.store(pibar_ud_ptr + (N_WS + split_i) * S + s_offs, zero, mask=valid_mask & split_valid)
            return

    if DEVICE_SCALAR_PARAMS:
        log_pD = tl.load(log_pD_arg).to(DTYPE)
        log_pS = tl.load(log_pS_arg).to(DTYPE)
    else:
        log_pD = log_pD_arg
        log_pS = log_pS_arg

    parent_pi_base = (ws + parent_w) * stride_C
    parent_vk_base = parent_w * S
    Pi_parent = tl.load(Pi_star_ptr + parent_pi_base + s_offs, mask=valid_mask, other=NEG_LARGE)
    v_k_val = tl.load(v_k_ptr + parent_vk_base + s_offs, mask=valid_mask, other=0.0)
    parent_valid = Pi_parent > NEG_LARGE
    mt = tl.load(mt_ptr + s_offs, mask=valid_mask, other=0.0).to(DTYPE)
    c1 = tl.load(sp_child1_ptr + s_offs, mask=valid_mask, other=0)
    c2 = tl.load(sp_child2_ptr + s_offs, mask=valid_mask, other=0)

    for j in tl.static_range(0, TILE_SPLITS):
        split_i = split_start + j
        split_valid = split_i < split_end
        mask = valid_mask & split_valid

        sl = tl.load(sl_ptr + split_i, mask=split_valid, other=0).to(tl.int64)
        sr = tl.load(sr_ptr + split_i, mask=split_valid, other=0).to(tl.int64)
        wlsp = tl.load(wlsp_ptr + split_i, mask=split_valid, other=0.0)

        pi_l_base = sl * stride_C
        pi_r_base = sr * stride_C
        Pi_l = tl.load(Pi_star_ptr + pi_l_base + s_offs, mask=mask, other=NEG_LARGE)
        Pi_r = tl.load(Pi_star_ptr + pi_r_base + s_offs, mask=mask, other=NEG_LARGE)
        Pibar_l = tl.load(Pibar_star_ptr + pi_l_base + s_offs, mask=mask, other=NEG_LARGE)
        Pibar_r = tl.load(Pibar_star_ptr + pi_r_base + s_offs, mask=mask, other=NEG_LARGE)

        c1_valid = (c1 < S) & mask
        c2_valid = (c2 < S) & mask
        Pi_l_s1 = tl.load(Pi_star_ptr + pi_l_base + c1, mask=c1_valid, other=NEG_LARGE)
        Pi_l_s2 = tl.load(Pi_star_ptr + pi_l_base + c2, mask=c2_valid, other=NEG_LARGE)
        Pi_r_s1 = tl.load(Pi_star_ptr + pi_r_base + c1, mask=c1_valid, other=NEG_LARGE)
        Pi_r_s2 = tl.load(Pi_star_ptr + pi_r_base + c2, mask=c2_valid, other=NEG_LARGE)

        d0 = log_pD + Pi_l + Pi_r
        d1 = Pi_l + Pibar_r
        d2 = Pi_r + Pibar_l
        d3 = log_pS + Pi_l_s1 + Pi_r_s2
        d4 = log_pS + Pi_r_s1 + Pi_l_s2

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

        tl.atomic_add(accumulated_rhs_ptr + pi_l_base + s_offs, vd0 + vd1, sem="relaxed", mask=mask)
        tl.atomic_add(accumulated_rhs_ptr + pi_r_base + s_offs, vd0 + vd2, sem="relaxed", mask=mask)

        row_max_l = tl.load(pibar_row_max_ptr + sl, mask=split_valid, other=0.0).to(DTYPE)
        row_max_r = tl.load(pibar_row_max_ptr + sr, mask=split_valid, other=0.0).to(DTYPE)
        finite_l = (Pibar_l > -1e29) & mask
        finite_r = (Pibar_r > -1e29) & mask
        inv_denom_l = tl.where(
            finite_l,
            tl.exp2(row_max_l + mt - Pibar_l),
            tl.zeros([BLOCK_S], dtype=DTYPE),
        )
        inv_denom_r = tl.where(
            finite_r,
            tl.exp2(row_max_r + mt - Pibar_r),
            tl.zeros([BLOCK_S], dtype=DTYPE),
        )
        ud_l = vd2 * inv_denom_l
        ud_r = vd1 * inv_denom_r
        tl.store(pibar_ud_ptr + split_i * S + s_offs, ud_l, mask=mask)
        tl.store(pibar_ud_ptr + (N_WS + split_i) * S + s_offs, ud_r, mask=mask)
        tl.atomic_add(
            pibar_A_ptr + split_i + scalar,
            tl.sum(tl.where(mask, ud_l, 0.0), axis=0),
            sem="relaxed",
            mask=split_valid,
        )
        tl.atomic_add(
            pibar_A_ptr + N_WS + split_i + scalar,
            tl.sum(tl.where(mask, ud_r, 0.0), axis=0),
            sem="relaxed",
            mask=split_valid,
        )

        tl.atomic_add(
            grad_log_pD_ptr + scalar,
            tl.sum(tl.where(mask, vd0, 0.0), axis=0),
            sem="relaxed",
            mask=split_valid,
        )
        tl.atomic_add(
            grad_log_pS_ptr + scalar,
            tl.sum(tl.where(mask, vd3 + vd4, 0.0), axis=0),
            sem="relaxed",
            mask=split_valid,
        )

        if ACCUM_MT_REDUCTION:
            mt_contrib = vd1 + vd2
            if GRAD_MT_SCALAR:
                tl.atomic_add(
                    grad_mt_ptr + scalar,
                    tl.sum(tl.where(mask, mt_contrib, 0.0), axis=0),
                    sem="relaxed",
                    mask=split_valid,
                )
            else:
                tl.atomic_add(grad_mt_ptr + s_offs, mt_contrib, sem="relaxed", mask=mask)

        tl.atomic_add(accumulated_rhs_ptr + pi_l_base + c1, vd3, sem="relaxed", mask=c1_valid)
        tl.atomic_add(accumulated_rhs_ptr + pi_r_base + c1, vd4, sem="relaxed", mask=c1_valid)
        tl.atomic_add(accumulated_rhs_ptr + pi_r_base + c2, vd3, sem="relaxed", mask=c2_valid)
        tl.atomic_add(accumulated_rhs_ptr + pi_l_base + c2, vd4, sem="relaxed", mask=c2_valid)


def dts_cross_backward_accum_parent_tiled_fused(
    Pi_star, Pibar_star, v_k, ws,
    sl, sr, reduce_idx, wlsp,
    n_eq1, ge2_ptr, ge2_parent_ids,
    log_pD, log_pS,
    sp_child1, sp_child2,
    accumulated_rhs,
    S,
    active_mask=None,
    grad_log_pD=None,
    grad_log_pS=None,
    grad_mt=None,
    accum_mt_reduction=False,
    mt_squeezed=None,
    pibar_row_max=None,
    tile_splits=16,
    ge2_max_fanout=None,
):
    """Parent-tiled staged DTS backward accumulation.

    This opt-in path matches the default high-fanout staged path: merged
    S-term, direct atomic Pi accumulation, staged Pibar u_d output, and scalar
    parameter accumulation into caller-owned reduction targets.
    """
    n_ws = sl.shape[0]
    device = Pi_star.device
    dtype = Pi_star.dtype

    if grad_log_pD is None or grad_log_pS is None:
        raise ValueError("parent-tiled DTS accumulation requires scalar reduction targets")
    if grad_log_pD.numel() != 1 or grad_log_pS.numel() != 1:
        raise ValueError("parent-tiled DTS scalar reduction targets must have one element")
    if accum_mt_reduction and grad_mt is None:
        raise ValueError("grad_mt is required when accumulating DTS mt reductions")
    if accum_mt_reduction and grad_mt.numel() not in (1, S):
        raise ValueError("DTS mt reduction target must have one element or S elements")
    if mt_squeezed is None or pibar_row_max is None:
        raise ValueError("mt_squeezed and pibar_row_max are required for parent-tiled DTS")
    if mt_squeezed.numel() != S:
        raise ValueError("mt_squeezed must have S elements for parent-tiled DTS")
    if pibar_row_max.numel() < Pi_star.shape[0]:
        raise ValueError("pibar_row_max must contain one row-max value per Pi row")
    if ge2_ptr is None or ge2_parent_ids is None:
        raise ValueError("ge2_ptr/ge2_parent_ids are required for parent-tiled DTS")

    tile_splits = max(1, int(tile_splits))
    n_eq1 = int(n_eq1)
    n_ge2_groups = int(ge2_parent_ids.numel())
    if n_ge2_groups > 0 and ge2_max_fanout is None:
        raise ValueError("ge2_max_fanout is required to avoid synchronizing in parent-tiled DTS")
    ge2_max_fanout = 0 if ge2_max_fanout is None else int(ge2_max_fanout)
    max_tiles = max(1, triton.cdiv(max(1, ge2_max_fanout), tile_splits))
    n_groups = n_eq1 + n_ge2_groups

    wlsp_flat = wlsp.squeeze(-1) if wlsp.ndim > 1 else wlsp
    log_pD_arg, log_pS_arg, device_scalar_params = _dts_scalar_param_args(
        log_pD, log_pS, device=device, dtype=dtype
    )
    mt_arg = mt_squeezed.contiguous() if not mt_squeezed.is_contiguous() else mt_squeezed
    pibar_row_max_arg = (
        pibar_row_max.contiguous()
        if not pibar_row_max.is_contiguous()
        else pibar_row_max
    )
    grad_mt_arg = grad_mt if accum_mt_reduction else grad_log_pD
    grad_mt_scalar = bool(accum_mt_reduction and grad_mt.numel() == 1)

    pibar_ud = torch.empty((2 * n_ws, S), device=device, dtype=dtype)
    pibar_A = torch.zeros((2 * n_ws,), device=device, dtype=dtype)
    if n_groups == 0:
        return pibar_ud, pibar_A, None, None

    BLOCK_S = min(256, triton.next_power_of_2(S))
    _dts_cross_backward_accum_parent_tiled_ge2_kernel[
        (n_groups, max_tiles, triton.cdiv(S, BLOCK_S))
    ](
        Pi_star,
        Pibar_star,
        v_k,
        active_mask if active_mask is not None else v_k,
        sl,
        sr,
        reduce_idx,
        wlsp_flat,
        ge2_ptr,
        ge2_parent_ids,
        log_pD_arg,
        log_pS_arg,
        sp_child1,
        sp_child2,
        accumulated_rhs,
        grad_log_pD,
        grad_log_pS,
        grad_mt_arg,
        pibar_ud,
        pibar_A,
        mt_arg,
        pibar_row_max_arg,
        ws,
        S,
        Pi_star.stride(0),
        n_ws,
        n_eq1,
        tile_splits,
        BLOCK_S,
        USE_ACTIVE_MASK=bool(active_mask is not None),
        DEVICE_SCALAR_PARAMS=bool(device_scalar_params),
        ACCUM_MT_REDUCTION=bool(accum_mt_reduction),
        GRAD_MT_SCALAR=bool(grad_mt_scalar),
        DTYPE=_tl_float_dtype(dtype),
        num_warps=4,
    )

    return pibar_ud, pibar_A, None, None


def _build_parent_ragged_ge2_worklist(n_eq1, ge2_ptr, ge2_parent_ids, tile_splits):
    """Build compact ge2 parent-tile metadata for the ragged DTS path."""
    device = ge2_ptr.device
    n_eq1 = int(n_eq1)
    tile_splits = max(1, int(tile_splits))
    if ge2_parent_ids.numel() == 0:
        empty = torch.empty((0,), device=device, dtype=torch.int32)
        return empty, empty, empty

    ptr_cpu = ge2_ptr.detach().cpu().tolist()
    parent_cpu = ge2_parent_ids.detach().cpu().tolist()
    tile_split_starts = []
    tile_split_ends = []
    tile_parent_ids = []
    for group, parent in enumerate(parent_cpu):
        group_start = int(ptr_cpu[group])
        group_end = int(ptr_cpu[group + 1])
        for split_start in range(group_start, group_end, tile_splits):
            tile_split_starts.append(n_eq1 + split_start)
            tile_split_ends.append(n_eq1 + min(group_end, split_start + tile_splits))
            tile_parent_ids.append(int(parent))

    if not tile_split_starts:
        empty = torch.empty((0,), device=device, dtype=torch.int32)
        return empty, empty, empty

    return (
        torch.tensor(tile_split_starts, device=device, dtype=torch.int32),
        torch.tensor(tile_split_ends, device=device, dtype=torch.int32),
        torch.tensor(tile_parent_ids, device=device, dtype=torch.int32),
    )


@triton.jit
def _dts_cross_backward_accum_parent_ragged_kernel(
    # Converged values [C, S]
    Pi_star_ptr,
    Pibar_star_ptr,
    # Neumann-solved adjoint [W, S]
    v_k_ptr,
    active_mask_ptr,       # optional [W] bool parent row activity mask
    # Split metadata
    sl_ptr,                # [n_ws] int64, eq1 first then ge2 grouped by parent
    sr_ptr,
    reduce_idx_ptr,        # [n_ws] int64 wave-local parent index
    wlsp_ptr,              # [n_ws] float
    ge2_tile_split_start_ptr,  # [n_ge2_tiles] absolute split start
    ge2_tile_split_end_ptr,    # [n_ge2_tiles] absolute split end
    ge2_tile_parent_ids_ptr,   # [n_ge2_tiles] wave-local parent ids
    # Scalar params
    log_pD_arg,
    log_pS_arg,
    # Species children [S] int64
    sp_child1_ptr,
    sp_child2_ptr,
    # Outputs
    accumulated_rhs_ptr,   # [C, S], direct Pi adjoints updated atomically
    grad_log_pD_ptr,       # scalar accumulation target
    grad_log_pS_ptr,       # scalar accumulation target
    grad_mt_ptr,           # optional scalar/[S] accumulation target
    pibar_ud_ptr,          # [2 * N_WS, S]
    pibar_A_ptr,           # [2 * N_WS]
    mt_ptr,                # [S]
    pibar_row_max_ptr,     # [C]
    # Dimensions
    ws,
    S: tl.constexpr,
    stride_C: tl.constexpr,
    N_WS: tl.constexpr,
    N_EQ1: tl.constexpr,
    TILE_SPLITS: tl.constexpr,
    BLOCK_S: tl.constexpr,
    USE_ACTIVE_MASK: tl.constexpr,
    DEVICE_SCALAR_PARAMS: tl.constexpr,
    ACCUM_MT_REDUCTION: tl.constexpr,
    GRAD_MT_SCALAR: tl.constexpr,
    DTYPE: tl.constexpr,
):
    """Parent-tiled staged DTS backward using a compact real-tile worklist."""
    NEG_LARGE: tl.constexpr = -1e30

    work = tl.program_id(0)
    block = tl.program_id(1)
    scalar = tl.arange(0, 1)

    if work < N_EQ1:
        split_start = work.to(tl.int64)
        split_end = split_start + 1
        parent_w = tl.load(reduce_idx_ptr + work).to(tl.int64)
    else:
        tile = work - N_EQ1
        split_start = tl.load(ge2_tile_split_start_ptr + tile).to(tl.int64)
        split_end = tl.load(ge2_tile_split_end_ptr + tile).to(tl.int64)
        parent_w = tl.load(ge2_tile_parent_ids_ptr + tile).to(tl.int64)

    s_offs = block * BLOCK_S + tl.arange(0, BLOCK_S)
    valid_mask = s_offs < S

    if USE_ACTIVE_MASK:
        parent_active = tl.load(active_mask_ptr + parent_w)
        if parent_active == 0:
            zero = tl.zeros([BLOCK_S], dtype=DTYPE)
            for j in tl.static_range(0, TILE_SPLITS):
                split_i = split_start + j
                split_valid = split_i < split_end
                tl.store(pibar_ud_ptr + split_i * S + s_offs, zero, mask=valid_mask & split_valid)
                tl.store(pibar_ud_ptr + (N_WS + split_i) * S + s_offs, zero, mask=valid_mask & split_valid)
            return

    if DEVICE_SCALAR_PARAMS:
        log_pD = tl.load(log_pD_arg).to(DTYPE)
        log_pS = tl.load(log_pS_arg).to(DTYPE)
    else:
        log_pD = log_pD_arg
        log_pS = log_pS_arg

    parent_pi_base = (ws + parent_w) * stride_C
    parent_vk_base = parent_w * S
    Pi_parent = tl.load(Pi_star_ptr + parent_pi_base + s_offs, mask=valid_mask, other=NEG_LARGE)
    v_k_val = tl.load(v_k_ptr + parent_vk_base + s_offs, mask=valid_mask, other=0.0)
    parent_valid = Pi_parent > NEG_LARGE
    mt = tl.load(mt_ptr + s_offs, mask=valid_mask, other=0.0).to(DTYPE)
    c1 = tl.load(sp_child1_ptr + s_offs, mask=valid_mask, other=0)
    c2 = tl.load(sp_child2_ptr + s_offs, mask=valid_mask, other=0)

    for j in tl.static_range(0, TILE_SPLITS):
        split_i = split_start + j
        split_valid = split_i < split_end
        mask = valid_mask & split_valid

        sl = tl.load(sl_ptr + split_i, mask=split_valid, other=0).to(tl.int64)
        sr = tl.load(sr_ptr + split_i, mask=split_valid, other=0).to(tl.int64)
        wlsp = tl.load(wlsp_ptr + split_i, mask=split_valid, other=0.0)

        pi_l_base = sl * stride_C
        pi_r_base = sr * stride_C
        Pi_l = tl.load(Pi_star_ptr + pi_l_base + s_offs, mask=mask, other=NEG_LARGE)
        Pi_r = tl.load(Pi_star_ptr + pi_r_base + s_offs, mask=mask, other=NEG_LARGE)
        Pibar_l = tl.load(Pibar_star_ptr + pi_l_base + s_offs, mask=mask, other=NEG_LARGE)
        Pibar_r = tl.load(Pibar_star_ptr + pi_r_base + s_offs, mask=mask, other=NEG_LARGE)

        c1_valid = (c1 < S) & mask
        c2_valid = (c2 < S) & mask
        Pi_l_s1 = tl.load(Pi_star_ptr + pi_l_base + c1, mask=c1_valid, other=NEG_LARGE)
        Pi_l_s2 = tl.load(Pi_star_ptr + pi_l_base + c2, mask=c2_valid, other=NEG_LARGE)
        Pi_r_s1 = tl.load(Pi_star_ptr + pi_r_base + c1, mask=c1_valid, other=NEG_LARGE)
        Pi_r_s2 = tl.load(Pi_star_ptr + pi_r_base + c2, mask=c2_valid, other=NEG_LARGE)

        d0 = log_pD + Pi_l + Pi_r
        d1 = Pi_l + Pibar_r
        d2 = Pi_r + Pibar_l
        d3 = log_pS + Pi_l_s1 + Pi_r_s2
        d4 = log_pS + Pi_r_s1 + Pi_l_s2

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

        tl.atomic_add(accumulated_rhs_ptr + pi_l_base + s_offs, vd0 + vd1, sem="relaxed", mask=mask)
        tl.atomic_add(accumulated_rhs_ptr + pi_r_base + s_offs, vd0 + vd2, sem="relaxed", mask=mask)

        row_max_l = tl.load(pibar_row_max_ptr + sl, mask=split_valid, other=0.0).to(DTYPE)
        row_max_r = tl.load(pibar_row_max_ptr + sr, mask=split_valid, other=0.0).to(DTYPE)
        finite_l = (Pibar_l > -1e29) & mask
        finite_r = (Pibar_r > -1e29) & mask
        inv_denom_l = tl.where(
            finite_l,
            tl.exp2(row_max_l + mt - Pibar_l),
            tl.zeros([BLOCK_S], dtype=DTYPE),
        )
        inv_denom_r = tl.where(
            finite_r,
            tl.exp2(row_max_r + mt - Pibar_r),
            tl.zeros([BLOCK_S], dtype=DTYPE),
        )
        ud_l = vd2 * inv_denom_l
        ud_r = vd1 * inv_denom_r
        tl.store(pibar_ud_ptr + split_i * S + s_offs, ud_l, mask=mask)
        tl.store(pibar_ud_ptr + (N_WS + split_i) * S + s_offs, ud_r, mask=mask)
        tl.atomic_add(
            pibar_A_ptr + split_i + scalar,
            tl.sum(tl.where(mask, ud_l, 0.0), axis=0),
            sem="relaxed",
            mask=split_valid,
        )
        tl.atomic_add(
            pibar_A_ptr + N_WS + split_i + scalar,
            tl.sum(tl.where(mask, ud_r, 0.0), axis=0),
            sem="relaxed",
            mask=split_valid,
        )

        tl.atomic_add(
            grad_log_pD_ptr + scalar,
            tl.sum(tl.where(mask, vd0, 0.0), axis=0),
            sem="relaxed",
            mask=split_valid,
        )
        tl.atomic_add(
            grad_log_pS_ptr + scalar,
            tl.sum(tl.where(mask, vd3 + vd4, 0.0), axis=0),
            sem="relaxed",
            mask=split_valid,
        )

        if ACCUM_MT_REDUCTION:
            mt_contrib = vd1 + vd2
            if GRAD_MT_SCALAR:
                tl.atomic_add(
                    grad_mt_ptr + scalar,
                    tl.sum(tl.where(mask, mt_contrib, 0.0), axis=0),
                    sem="relaxed",
                    mask=split_valid,
                )
            else:
                tl.atomic_add(grad_mt_ptr + s_offs, mt_contrib, sem="relaxed", mask=mask)

        tl.atomic_add(accumulated_rhs_ptr + pi_l_base + c1, vd3, sem="relaxed", mask=c1_valid)
        tl.atomic_add(accumulated_rhs_ptr + pi_r_base + c1, vd4, sem="relaxed", mask=c1_valid)
        tl.atomic_add(accumulated_rhs_ptr + pi_r_base + c2, vd3, sem="relaxed", mask=c2_valid)
        tl.atomic_add(accumulated_rhs_ptr + pi_l_base + c2, vd4, sem="relaxed", mask=c2_valid)


def dts_cross_backward_accum_parent_ragged_fused(
    Pi_star, Pibar_star, v_k, ws,
    sl, sr, reduce_idx, wlsp,
    n_eq1, ge2_ptr, ge2_parent_ids,
    log_pD, log_pS,
    sp_child1, sp_child2,
    accumulated_rhs,
    S,
    active_mask=None,
    grad_log_pD=None,
    grad_log_pS=None,
    grad_mt=None,
    accum_mt_reduction=False,
    mt_squeezed=None,
    pibar_row_max=None,
    tile_splits=16,
    ge2_max_fanout=None,
    ge2_tile_split_starts=None,
    ge2_tile_split_ends=None,
    ge2_tile_parent_ids=None,
):
    """Parent-tiled staged DTS accumulation with a ragged real-tile worklist."""
    del ge2_max_fanout
    n_ws = sl.shape[0]
    device = Pi_star.device
    dtype = Pi_star.dtype

    if grad_log_pD is None or grad_log_pS is None:
        raise ValueError("parent-ragged DTS accumulation requires scalar reduction targets")
    if grad_log_pD.numel() != 1 or grad_log_pS.numel() != 1:
        raise ValueError("parent-ragged DTS scalar reduction targets must have one element")
    if accum_mt_reduction and grad_mt is None:
        raise ValueError("grad_mt is required when accumulating DTS mt reductions")
    if accum_mt_reduction and grad_mt.numel() not in (1, S):
        raise ValueError("DTS mt reduction target must have one element or S elements")
    if mt_squeezed is None or pibar_row_max is None:
        raise ValueError("mt_squeezed and pibar_row_max are required for parent-ragged DTS")
    if mt_squeezed.numel() != S:
        raise ValueError("mt_squeezed must have S elements for parent-ragged DTS")
    if pibar_row_max.numel() < Pi_star.shape[0]:
        raise ValueError("pibar_row_max must contain one row-max value per Pi row")
    if ge2_ptr is None or ge2_parent_ids is None:
        raise ValueError("ge2_ptr/ge2_parent_ids are required for parent-ragged DTS")

    tile_splits = max(1, int(tile_splits))
    n_eq1 = int(n_eq1)
    if (
        ge2_tile_split_starts is None
        or ge2_tile_split_ends is None
        or ge2_tile_parent_ids is None
    ):
        (
            ge2_tile_split_starts,
            ge2_tile_split_ends,
            ge2_tile_parent_ids,
        ) = _build_parent_ragged_ge2_worklist(
            n_eq1,
            ge2_ptr,
            ge2_parent_ids,
            tile_splits,
        )
    else:
        ge2_tile_split_starts = ge2_tile_split_starts.contiguous()
        ge2_tile_split_ends = ge2_tile_split_ends.contiguous()
        ge2_tile_parent_ids = ge2_tile_parent_ids.contiguous()
    if (
        ge2_tile_split_starts.numel() != ge2_tile_split_ends.numel()
        or ge2_tile_split_starts.numel() != ge2_tile_parent_ids.numel()
    ):
        raise ValueError("parent-ragged tile worklist tensors must have matching lengths")

    n_work_tiles = n_eq1 + int(ge2_tile_split_starts.numel())

    wlsp_flat = wlsp.squeeze(-1) if wlsp.ndim > 1 else wlsp
    log_pD_arg, log_pS_arg, device_scalar_params = _dts_scalar_param_args(
        log_pD, log_pS, device=device, dtype=dtype
    )
    mt_arg = mt_squeezed.contiguous() if not mt_squeezed.is_contiguous() else mt_squeezed
    pibar_row_max_arg = (
        pibar_row_max.contiguous()
        if not pibar_row_max.is_contiguous()
        else pibar_row_max
    )
    grad_mt_arg = grad_mt if accum_mt_reduction else grad_log_pD
    grad_mt_scalar = bool(accum_mt_reduction and grad_mt.numel() == 1)

    pibar_ud = torch.empty((2 * n_ws, S), device=device, dtype=dtype)
    pibar_A = torch.zeros((2 * n_ws,), device=device, dtype=dtype)
    if n_work_tiles == 0:
        return pibar_ud, pibar_A, None, None

    BLOCK_S = min(256, triton.next_power_of_2(S))
    _dts_cross_backward_accum_parent_ragged_kernel[
        (n_work_tiles, triton.cdiv(S, BLOCK_S))
    ](
        Pi_star,
        Pibar_star,
        v_k,
        active_mask if active_mask is not None else v_k,
        sl,
        sr,
        reduce_idx,
        wlsp_flat,
        ge2_tile_split_starts,
        ge2_tile_split_ends,
        ge2_tile_parent_ids,
        log_pD_arg,
        log_pS_arg,
        sp_child1,
        sp_child2,
        accumulated_rhs,
        grad_log_pD,
        grad_log_pS,
        grad_mt_arg,
        pibar_ud,
        pibar_A,
        mt_arg,
        pibar_row_max_arg,
        ws,
        S,
        Pi_star.stride(0),
        n_ws,
        n_eq1,
        tile_splits,
        BLOCK_S,
        USE_ACTIVE_MASK=bool(active_mask is not None),
        DEVICE_SCALAR_PARAMS=bool(device_scalar_params),
        ACCUM_MT_REDUCTION=bool(accum_mt_reduction),
        GRAD_MT_SCALAR=bool(grad_mt_scalar),
        DTYPE=_tl_float_dtype(dtype),
        num_warps=4,
    )

    return pibar_ud, pibar_A, None, None


@triton.jit
def _add_grouped_dts_pi_accum_kernel(
    group_children_ptr,   # [n_groups]
    grouped_grad_ptr,     # [n_groups, S]
    accumulated_rhs_ptr,  # [C, S]
    S: tl.constexpr,
    stride_C: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    group = tl.program_id(0)
    block = tl.program_id(1)

    child = tl.load(group_children_ptr + group).to(tl.int64)
    s_offs = block * BLOCK_S + tl.arange(0, BLOCK_S)
    mask = s_offs < S
    grad = tl.load(grouped_grad_ptr + group * S + s_offs, mask=mask, other=0.0)
    out = accumulated_rhs_ptr + child * stride_C + s_offs
    cur = tl.load(out, mask=mask, other=0.0)
    tl.store(out, cur + grad, mask=mask)


def dts_cross_backward_accum_grouped_fused(
    Pi_star, Pibar_star, v_k, ws,
    sl, sr, reduce_idx, wlsp,
    log_pD, log_pS,
    sp_child1, sp_child2,
    accumulated_rhs,
    S,
    active_mask=None,
    group_children=None,
    group_inverse=None,
):
    """Two-stage DTS backward accumulation grouped by child clade.

    Stage 1 reuses the per-split fused DTS VJP to build local Pi/Pibar
    adjoints. Stage 2 reduces left/right Pi adjoints by child row in a compact
    scratch buffer, then adds each reduced child row once to accumulated_rhs.
    This is an opt-in high-fanout alternative to direct atomics.
    """
    n_ws = sl.shape[0]
    if active_mask is not None and reduce_idx is None:
        raise ValueError("reduce_idx is required when active_mask is provided")

    (grad_Pi_l, grad_Pi_r, grad_Pibar_l, grad_Pibar_r,
     param_pD, param_pS) = dts_cross_backward_fused(
        Pi_star, Pibar_star, v_k, ws,
        sl, sr, reduce_idx, wlsp,
        log_pD, log_pS,
        sp_child1, sp_child2, S,
        active_mask=active_mask,
    )

    if n_ws == 0:
        return grad_Pibar_l, grad_Pibar_r, param_pD, param_pS

    if group_children is None or group_inverse is None:
        all_children = torch.cat((sl, sr), dim=0)
        group_children, group_inverse = torch.unique(
            all_children,
            sorted=True,
            return_inverse=True,
        )

    group_children = group_children.contiguous()
    group_inverse = group_inverse.contiguous()
    n_groups = group_children.shape[0]
    if n_groups == 0:
        return grad_Pibar_l, grad_Pibar_r, param_pD, param_pS

    BLOCK_S = min(256, triton.next_power_of_2(S))
    grouped_grad = torch.zeros((n_groups, S), device=Pi_star.device, dtype=Pi_star.dtype)
    use_active_mask = active_mask is not None

    _group_cross_pibar_grad_kernel[(2 * n_ws, triton.cdiv(S, BLOCK_S))](
        grad_Pi_l,
        grad_Pi_r,
        group_inverse,
        reduce_idx if reduce_idx is not None else group_inverse,
        active_mask if active_mask is not None else grouped_grad,
        grouped_grad,
        grouped_grad,
        n_ws,
        S,
        BLOCK_S,
        USE_ACTIVE_MASK=use_active_mask,
        TRACK_GROUP_ACTIVE=False,
        num_warps=4,
    )

    _add_grouped_dts_pi_accum_kernel[(n_groups, triton.cdiv(S, BLOCK_S))](
        group_children,
        grouped_grad,
        accumulated_rhs,
        S,
        accumulated_rhs.stride(0),
        BLOCK_S,
        num_warps=4,
    )

    return grad_Pibar_l, grad_Pibar_r, param_pD, param_pS


# =========================================================================
# Uniform Pibar VJP for cross-clade gradients
# =========================================================================

@triton.jit
def _pibar_row_stats_kernel(
    Pi_star_ptr,
    row_max_ptr,
    row_sum_ptr,
    C: tl.constexpr,
    S: tl.constexpr,
    stride_C: tl.constexpr,
    BLOCK_S: tl.constexpr,
    DTYPE: tl.constexpr,
):
    """Precompute row_max and shifted row_sum for uniform Pibar VJP."""
    NEG_LARGE: tl.constexpr = -1e30
    row = tl.program_id(0)

    row_max = tl.full([1], value=NEG_LARGE, dtype=DTYPE)
    row_sum = tl.full([1], value=0.0, dtype=DTYPE)
    pi_base = row * stride_C
    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        mask = s_offs < S
        pi_val = tl.load(Pi_star_ptr + pi_base + s_offs, mask=mask, other=NEG_LARGE)
        tile_max = tl.max(pi_val, axis=0)
        new_max = tl.maximum(row_max, tile_max)
        row_sum = row_sum * tl.exp2(row_max - new_max) + tl.sum(tl.exp2(pi_val - new_max), axis=0)
        row_max = new_max

    scalar = tl.arange(0, 1)
    tl.store(row_max_ptr + row + scalar, row_max)
    tl.store(row_sum_ptr + row + scalar, row_sum)


def pibar_row_stats_fused(Pi_star):
    """Compute compact row stats used by uniform Pibar VJP kernels."""
    C, S = Pi_star.shape
    row_max = torch.empty((C,), device=Pi_star.device, dtype=Pi_star.dtype)
    row_sum = torch.empty((C,), device=Pi_star.device, dtype=Pi_star.dtype)
    BLOCK_S = min(256, triton.next_power_of_2(S))
    _pibar_row_stats_kernel[(C,)](
        Pi_star,
        row_max,
        row_sum,
        C,
        S,
        Pi_star.stride(0),
        BLOCK_S,
        DTYPE=_tl_float_dtype(Pi_star.dtype),
        num_warps=4,
    )
    return row_max, row_sum


@triton.jit
def _uniform_cross_pibar_vjp_kernel(
    Pi_star_ptr,          # [C, S]
    grad_Pibar_l_ptr,     # [n_ws, S]
    grad_Pibar_r_ptr,     # [n_ws, S]
    sl_ptr,               # [n_ws]
    sr_ptr,               # [n_ws]
    reduce_idx_ptr,       # [n_ws], used with active_mask_ptr when enabled
    active_mask_ptr,      # optional [W] bool parent row activity mask
    ancestor_cols_ptr,    # [MAX_ANCESTOR_DEPTH, S]
    row_max_ptr,          # optional [C]
    row_sum_ptr,          # optional [C]
    accumulated_rhs_ptr,  # [C, S], updated atomically
    correction_buf_ptr,   # [2 * n_ws, S]
    n_ws: tl.constexpr,
    S: tl.constexpr,
    stride_C: tl.constexpr,
    BLOCK_S: tl.constexpr,
    MAX_ANCESTOR_DEPTH: tl.constexpr,
    USE_ACTIVE_MASK: tl.constexpr,
    USE_ROW_STATS: tl.constexpr,
    DTYPE: tl.constexpr,
):
    """Apply the VJP of uniform Pibar for one cross-DTS child side.

    For a child row with incoming Pibar adjoint u:

        p = exp2(Pi - row_max)
        denom[s] = sum_f p[f] - sum_{a in ancestors(s)} p[a]
        u_d[s] = u[s] / denom[s]
        grad_Pi[f] = p[f] * (sum_s u_d[s] -
                             sum_{s: f in ancestors(s)} u_d[s])

    Each program handles either the left or right child of one split and
    atomically accumulates grad_Pi into accumulated_rhs[child].
    """
    NEG_LARGE: tl.constexpr = -1e30

    row = tl.program_id(0)
    split_i = tl.where(row < n_ws, row, row - n_ws)
    is_right = row >= n_ws

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
    grad_base = split_i * S
    corr_base = row * S

    # Clear the correction row owned by this program.
    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        valid_mask = s_offs < S
        mask = valid_mask & row_active
        tl.store(correction_buf_ptr + corr_base + s_offs,
                 tl.zeros([BLOCK_S], dtype=DTYPE), mask=mask)

    if USE_ROW_STATS:
        row_max = tl.load(row_max_ptr + child)
        row_sum = tl.load(row_sum_ptr + child)
    else:
        # Row max and shifted row sum for p = exp2(Pi - row_max).
        row_max = tl.full([1], value=NEG_LARGE, dtype=DTYPE)
        row_sum = tl.full([1], value=0.0, dtype=DTYPE)
        for s_start in range(0, S, BLOCK_S):
            s_offs = s_start + tl.arange(0, BLOCK_S)
            valid_mask = s_offs < S
            mask = valid_mask & row_active
            pi_val = tl.load(Pi_star_ptr + pi_base + s_offs, mask=mask, other=NEG_LARGE)
            tile_max = tl.max(pi_val, axis=0)
            new_max = tl.maximum(row_max, tile_max)
            row_sum = row_sum * tl.exp2(row_max - new_max) + tl.sum(tl.exp2(pi_val - new_max), axis=0)
            row_max = new_max

    # Build correction[f] = sum_{s: f ancestor of s} u_d[s].
    A = tl.full([1], value=0.0, dtype=DTYPE)
    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        valid_mask = s_offs < S
        mask = valid_mask & row_active

        ancestor_sum = tl.zeros([BLOCK_S], dtype=DTYPE)
        for k in range(0, MAX_ANCESTOR_DEPTH):
            anc = tl.load(ancestor_cols_ptr + k * S + s_offs, mask=mask, other=-1)
            anc_valid = mask & (anc >= 0) & (anc < S)
            pi_anc = tl.load(Pi_star_ptr + pi_base + anc, mask=anc_valid, other=NEG_LARGE)
            ancestor_sum += tl.where(
                anc_valid,
                tl.exp2(pi_anc - row_max),
                tl.zeros([BLOCK_S], dtype=DTYPE),
            )

        denom = row_sum - ancestor_sum

        grad_l = tl.load(grad_Pibar_l_ptr + grad_base + s_offs, mask=mask, other=0.0)
        grad_r = tl.load(grad_Pibar_r_ptr + grad_base + s_offs, mask=mask, other=0.0)
        grad_u = tl.where(is_right, grad_r, grad_l)
        u_d = tl.where(denom > 0.0, grad_u / denom, tl.zeros([BLOCK_S], dtype=DTYPE))
        A += tl.sum(u_d, axis=0)

        for k in range(0, MAX_ANCESTOR_DEPTH):
            anc = tl.load(ancestor_cols_ptr + k * S + s_offs, mask=mask, other=-1)
            anc_valid = mask & (anc >= 0) & (anc < S)
            tl.atomic_add(correction_buf_ptr + corr_base + anc, u_d, sem="relaxed", mask=anc_valid)

    # Add p[f] * (A - correction[f]) into the child row's Pi adjoint.
    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        valid_mask = s_offs < S
        mask = valid_mask & row_active
        pi_val = tl.load(Pi_star_ptr + pi_base + s_offs, mask=mask, other=NEG_LARGE)
        p_prime = tl.exp2(pi_val - row_max)
        correction = tl.load(correction_buf_ptr + corr_base + s_offs, mask=mask, other=0.0)
        contrib = p_prime * (A - correction)
        tl.atomic_add(accumulated_rhs_ptr + pi_base + s_offs, contrib, sem="relaxed", mask=mask)


def uniform_cross_pibar_vjp_fused(
    Pi_star,
    grad_Pibar_l,
    grad_Pibar_r,
    sl,
    sr,
    ancestor_cols,
    accumulated_rhs,
    S,
    active_mask=None,
    reduce_idx=None,
    row_stats=None,
):
    """Fused uniform-Pibar VJP for cross-DTS child gradients.

    This replaces:
      cat(left/right Pibar grads) -> p_prime @ ancestors_T ->
      ancestors_T @ u_d.T -> index_add into accumulated_rhs.

    The operation is in-place on accumulated_rhs.
    """
    n_ws = sl.shape[0]
    if n_ws == 0:
        return
    if active_mask is not None and reduce_idx is None:
        raise ValueError("reduce_idx is required when active_mask is provided")

    correction_buf = torch.empty((2 * n_ws, S), device=Pi_star.device, dtype=Pi_star.dtype)
    BLOCK_S = min(256, triton.next_power_of_2(S))
    stride_C = Pi_star.stride(0)

    _uniform_cross_pibar_vjp_kernel[(2 * n_ws,)](
        Pi_star,
        grad_Pibar_l,
        grad_Pibar_r,
        sl,
        sr,
        reduce_idx if reduce_idx is not None else sl,
        active_mask if active_mask is not None else grad_Pibar_l,
        ancestor_cols,
        row_stats[0] if row_stats is not None else Pi_star,
        row_stats[1] if row_stats is not None else Pi_star,
        accumulated_rhs,
        correction_buf,
        n_ws,
        S,
        stride_C,
        BLOCK_S,
        MAX_ANCESTOR_DEPTH=ancestor_cols.shape[0],
        USE_ACTIVE_MASK=bool(active_mask is not None),
        USE_ROW_STATS=bool(row_stats is not None),
        DTYPE=_tl_float_dtype(Pi_star.dtype),
        num_warps=4,
    )


@triton.jit
def _group_cross_pibar_grad_kernel(
    grad_Pibar_l_ptr,     # [n_ws, S]
    grad_Pibar_r_ptr,     # [n_ws, S]
    group_inverse_ptr,    # [2 * n_ws]
    reduce_idx_ptr,       # [n_ws], used with active_mask_ptr when enabled
    active_mask_ptr,      # optional [W] bool parent row activity mask
    grouped_grad_ptr,     # [n_groups, S]
    group_active_ptr,     # optional [n_groups] bool
    n_ws: tl.constexpr,
    S: tl.constexpr,
    BLOCK_S: tl.constexpr,
    USE_ACTIVE_MASK: tl.constexpr,
    TRACK_GROUP_ACTIVE: tl.constexpr,
):
    side = tl.program_id(0)
    block = tl.program_id(1)
    split_i = tl.where(side < n_ws, side, side - n_ws)
    is_right = side >= n_ws

    if USE_ACTIVE_MASK:
        parent_w = tl.load(reduce_idx_ptr + split_i).to(tl.int64)
        row_active = tl.load(active_mask_ptr + parent_w)
        if row_active == 0:
            return

    group = tl.load(group_inverse_ptr + side).to(tl.int64)
    if TRACK_GROUP_ACTIVE:
        tl.store(group_active_ptr + group, 1)

    s_offs = block * BLOCK_S + tl.arange(0, BLOCK_S)
    mask = s_offs < S
    grad_base = split_i * S
    grad_l = tl.load(grad_Pibar_l_ptr + grad_base + s_offs, mask=mask, other=0.0)
    grad_r = tl.load(grad_Pibar_r_ptr + grad_base + s_offs, mask=mask, other=0.0)
    grad = tl.where(is_right, grad_r, grad_l)
    tl.atomic_add(grouped_grad_ptr + group * S + s_offs, grad, sem="relaxed", mask=mask)


@triton.jit
def _uniform_cross_pibar_vjp_tree_grouped_kernel(
    Pi_star_ptr,          # [C, S]
    grouped_grad_ptr,     # [n_groups, S]
    group_children_ptr,   # [n_groups]
    ancestor_cols_ptr,    # [MAX_ANCESTOR_DEPTH, S]
    sp_child1_ptr,        # [S]
    sp_child2_ptr,        # [S]
    level_parents_ptr,    # [N_LEVELS, MAX_LEVEL_WIDTH]
    accumulated_rhs_ptr,  # [C, S], updated atomically
    subtree_buf_ptr,      # [n_groups, S]
    group_active_ptr,     # optional [n_groups] bool
    n_groups: tl.constexpr,
    S: tl.constexpr,
    stride_C: tl.constexpr,
    BLOCK_S: tl.constexpr,
    MAX_ANCESTOR_DEPTH: tl.constexpr,
    N_LEVELS: tl.constexpr,
    MAX_LEVEL_WIDTH: tl.constexpr,
    USE_GROUP_ACTIVE: tl.constexpr,
    DTYPE: tl.constexpr,
):
    """Uniform Pibar VJP after reducing split-side adjoints by child clade."""
    NEG_LARGE: tl.constexpr = -1e30

    row = tl.program_id(0)
    if USE_GROUP_ACTIVE:
        group_active = tl.load(group_active_ptr + row)
        if group_active == 0:
            return

    child = tl.load(group_children_ptr + row).to(tl.int64)
    pi_base = child * stride_C
    grad_base = row * S
    subtree_base = row * S

    row_max = tl.full([1], value=NEG_LARGE, dtype=DTYPE)
    row_sum = tl.full([1], value=0.0, dtype=DTYPE)
    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        mask = s_offs < S
        pi_val = tl.load(Pi_star_ptr + pi_base + s_offs, mask=mask, other=NEG_LARGE)
        tile_max = tl.max(pi_val, axis=0)
        new_max = tl.maximum(row_max, tile_max)
        row_sum = row_sum * tl.exp2(row_max - new_max) + tl.sum(tl.exp2(pi_val - new_max), axis=0)
        row_max = new_max

    A = tl.full([1], value=0.0, dtype=DTYPE)
    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        mask = s_offs < S

        ancestor_sum = tl.zeros([BLOCK_S], dtype=DTYPE)
        for k in range(0, MAX_ANCESTOR_DEPTH):
            anc = tl.load(ancestor_cols_ptr + k * S + s_offs, mask=mask, other=-1)
            anc_valid = mask & (anc >= 0) & (anc < S)
            pi_anc = tl.load(Pi_star_ptr + pi_base + anc, mask=anc_valid, other=NEG_LARGE)
            ancestor_sum += tl.where(
                anc_valid,
                tl.exp2(pi_anc - row_max),
                tl.zeros([BLOCK_S], dtype=DTYPE),
            )

        denom = row_sum - ancestor_sum
        grad_u = tl.load(grouped_grad_ptr + grad_base + s_offs, mask=mask, other=0.0)
        u_d = tl.where(denom > 0.0, grad_u / denom, tl.zeros([BLOCK_S], dtype=DTYPE))
        A += tl.sum(u_d, axis=0)
        tl.store(subtree_buf_ptr + subtree_base + s_offs, u_d, mask=mask)

    tl.debug_barrier()

    for level in range(0, N_LEVELS):
        for p_start in range(0, MAX_LEVEL_WIDTH, BLOCK_S):
            p_offs = p_start + tl.arange(0, BLOCK_S)
            parent = tl.load(
                level_parents_ptr + level * MAX_LEVEL_WIDTH + p_offs,
                mask=p_offs < MAX_LEVEL_WIDTH,
                other=-1,
            )
            parent_valid = (parent >= 0) & (parent < S)
            c1 = tl.load(sp_child1_ptr + parent, mask=parent_valid, other=S)
            c2 = tl.load(sp_child2_ptr + parent, mask=parent_valid, other=S)
            c1_valid = parent_valid & (c1 < S)
            c2_valid = parent_valid & (c2 < S)

            parent_val = tl.load(subtree_buf_ptr + subtree_base + parent, mask=parent_valid, other=0.0)
            c1_val = tl.load(subtree_buf_ptr + subtree_base + c1, mask=c1_valid, other=0.0)
            c2_val = tl.load(subtree_buf_ptr + subtree_base + c2, mask=c2_valid, other=0.0)
            tl.store(subtree_buf_ptr + subtree_base + parent, parent_val + c1_val + c2_val, mask=parent_valid)
        tl.debug_barrier()

    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        mask = s_offs < S
        pi_val = tl.load(Pi_star_ptr + pi_base + s_offs, mask=mask, other=NEG_LARGE)
        p_prime = tl.exp2(pi_val - row_max)
        subtree_sum = tl.load(subtree_buf_ptr + subtree_base + s_offs, mask=mask, other=0.0)
        contrib = p_prime * (A - subtree_sum)
        tl.atomic_add(accumulated_rhs_ptr + pi_base + s_offs, contrib, sem="relaxed", mask=mask)


@triton.jit
def _uniform_cross_pibar_vjp_tree_kernel(
    Pi_star_ptr,          # [C, S]
    Pibar_star_ptr,       # [C, S], used when reusing forward Pibar denominators
    mt_ptr,               # [S], used when reusing forward Pibar denominators
    pibar_row_max_ptr,    # [C], used when reusing forward Pibar denominators
    grad_Pibar_l_ptr,     # [n_ws, S]
    grad_Pibar_r_ptr,     # [n_ws, S]
    sl_ptr,               # [n_ws]
    sr_ptr,               # [n_ws]
    reduce_idx_ptr,       # [n_ws], used with active_mask_ptr when enabled
    active_mask_ptr,      # optional [W] bool parent row activity mask
    ancestor_cols_ptr,    # [MAX_ANCESTOR_DEPTH, S]
    row_max_ptr,          # optional [C]
    row_sum_ptr,          # optional [C]
    sp_child1_ptr,        # [S]
    sp_child2_ptr,        # [S]
    level_parents_ptr,    # [N_LEVELS, MAX_LEVEL_WIDTH]
    accumulated_rhs_ptr,  # [C, S], updated atomically
    subtree_buf_ptr,      # [2 * n_ws, S]
    n_ws: tl.constexpr,
    S: tl.constexpr,
    stride_C: tl.constexpr,
    BLOCK_S: tl.constexpr,
    MAX_ANCESTOR_DEPTH: tl.constexpr,
    N_LEVELS: tl.constexpr,
    MAX_LEVEL_WIDTH: tl.constexpr,
    USE_ACTIVE_MASK: tl.constexpr,
    USE_ROW_STATS: tl.constexpr,
    USE_PIBAR_DENOM_STATS: tl.constexpr,
    DTYPE: tl.constexpr,
):
    """Uniform Pibar VJP using a descendant/subtree gather.

    This computes the same correction term as _uniform_cross_pibar_vjp_kernel,
    but avoids scattering every descendant into all of its ancestors.  Instead
    it writes u_d into subtree_buf and performs a bottom-up tree reduction:

        subtree_sum[parent] = u_d[parent] + subtree_sum[child1] + subtree_sum[child2]

    After that, correction[f] is subtree_sum[f].
    """
    NEG_LARGE: tl.constexpr = -1e30

    row = tl.program_id(0)
    split_i = tl.where(row < n_ws, row, row - n_ws)
    is_right = row >= n_ws

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
    grad_base = split_i * S
    subtree_base = row * S

    if USE_PIBAR_DENOM_STATS:
        row_max = tl.load(pibar_row_max_ptr + child)
        row_sum = tl.full([1], value=0.0, dtype=DTYPE)
    elif USE_ROW_STATS:
        row_max = tl.load(row_max_ptr + child)
        row_sum = tl.load(row_sum_ptr + child)
    else:
        row_max = tl.full([1], value=NEG_LARGE, dtype=DTYPE)
        row_sum = tl.full([1], value=0.0, dtype=DTYPE)
        for s_start in range(0, S, BLOCK_S):
            s_offs = s_start + tl.arange(0, BLOCK_S)
            valid_mask = s_offs < S
            mask = valid_mask & row_active
            pi_val = tl.load(Pi_star_ptr + pi_base + s_offs, mask=mask, other=NEG_LARGE)
            tile_max = tl.max(pi_val, axis=0)
            new_max = tl.maximum(row_max, tile_max)
            row_sum = row_sum * tl.exp2(row_max - new_max) + tl.sum(tl.exp2(pi_val - new_max), axis=0)
            row_max = new_max

    # Initial subtree_buf contains u_d for every species.  Leaves are already
    # final subtree sums; internal nodes are completed by the bottom-up pass.
    A = tl.full([1], value=0.0, dtype=DTYPE)
    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        valid_mask = s_offs < S
        mask = valid_mask & row_active

        grad_l = tl.load(grad_Pibar_l_ptr + grad_base + s_offs, mask=mask, other=0.0)
        grad_r = tl.load(grad_Pibar_r_ptr + grad_base + s_offs, mask=mask, other=0.0)
        grad_u = tl.where(is_right, grad_r, grad_l)
        if USE_PIBAR_DENOM_STATS:
            pibar_val = tl.load(Pibar_star_ptr + pi_base + s_offs, mask=mask, other=NEG_LARGE)
            mt = tl.load(mt_ptr + s_offs, mask=mask, other=0.0)
            finite_pibar = pibar_val > -1e29
            inv_denom = tl.where(
                finite_pibar,
                tl.exp2(row_max + mt - pibar_val),
                tl.zeros([BLOCK_S], dtype=DTYPE),
            )
            u_d = grad_u * inv_denom
        else:
            ancestor_sum = tl.zeros([BLOCK_S], dtype=DTYPE)
            for k in range(0, MAX_ANCESTOR_DEPTH):
                anc = tl.load(ancestor_cols_ptr + k * S + s_offs, mask=mask, other=-1)
                anc_valid = mask & (anc >= 0) & (anc < S)
                pi_anc = tl.load(Pi_star_ptr + pi_base + anc, mask=anc_valid, other=NEG_LARGE)
                ancestor_sum += tl.where(
                    anc_valid,
                    tl.exp2(pi_anc - row_max),
                    tl.zeros([BLOCK_S], dtype=DTYPE),
                )

            denom = row_sum - ancestor_sum
            u_d = tl.where(denom > 0.0, grad_u / denom, tl.zeros([BLOCK_S], dtype=DTYPE))
        A += tl.sum(u_d, axis=0)
        tl.store(subtree_buf_ptr + subtree_base + s_offs, u_d, mask=mask)

    tl.debug_barrier()

    # Bottom-up subtree reduction.  level_parents is ordered from parents of
    # leaves up to the root, so every child subtree has already been computed.
    for level in range(0, N_LEVELS):
        for p_start in range(0, MAX_LEVEL_WIDTH, BLOCK_S):
            p_offs = p_start + tl.arange(0, BLOCK_S)
            parent = tl.load(
                level_parents_ptr + level * MAX_LEVEL_WIDTH + p_offs,
                mask=p_offs < MAX_LEVEL_WIDTH,
                other=-1,
            )
            parent_valid = (parent >= 0) & (parent < S) & row_active
            c1 = tl.load(sp_child1_ptr + parent, mask=parent_valid, other=S)
            c2 = tl.load(sp_child2_ptr + parent, mask=parent_valid, other=S)
            c1_valid = parent_valid & (c1 < S)
            c2_valid = parent_valid & (c2 < S)

            parent_val = tl.load(subtree_buf_ptr + subtree_base + parent, mask=parent_valid, other=0.0)
            c1_val = tl.load(subtree_buf_ptr + subtree_base + c1, mask=c1_valid, other=0.0)
            c2_val = tl.load(subtree_buf_ptr + subtree_base + c2, mask=c2_valid, other=0.0)
            tl.store(subtree_buf_ptr + subtree_base + parent, parent_val + c1_val + c2_val, mask=parent_valid)
        tl.debug_barrier()

    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        valid_mask = s_offs < S
        mask = valid_mask & row_active
        pi_val = tl.load(Pi_star_ptr + pi_base + s_offs, mask=mask, other=NEG_LARGE)
        p_prime = tl.exp2(pi_val - row_max)
        subtree_sum = tl.load(subtree_buf_ptr + subtree_base + s_offs, mask=mask, other=0.0)
        contrib = p_prime * (A - subtree_sum)
        # Duplicate child clades across splits still require an atomic add into
        # accumulated_rhs.  The subtree correction itself is atomic-free.
        tl.atomic_add(accumulated_rhs_ptr + pi_base + s_offs, contrib, sem="relaxed", mask=mask)


@triton.jit
def _uniform_cross_pibar_vjp_tree_prefix_kernel(
    Pi_star_ptr,          # [C, S]
    grad_Pibar_l_ptr,     # [n_ws, S]
    grad_Pibar_r_ptr,     # [n_ws, S]
    sl_ptr,               # [n_ws]
    sr_ptr,               # [n_ws]
    reduce_idx_ptr,       # [n_ws], used with active_mask_ptr when enabled
    active_mask_ptr,      # optional [W] bool parent row activity mask
    row_max_ptr,          # optional [C]
    row_sum_ptr,          # optional [C]
    sp_parent_ptr,        # [S]
    sp_child1_ptr,        # [S]
    sp_child2_ptr,        # [S]
    depth_nodes_ptr,      # [N_DEPTHS, MAX_DEPTH_WIDTH], top-down
    level_parents_ptr,    # [N_LEVELS, MAX_LEVEL_WIDTH], bottom-up
    accumulated_rhs_ptr,  # [C, S], updated atomically
    subtree_buf_ptr,      # [2 * n_ws, S], reused as prefix then subtree
    n_ws: tl.constexpr,
    S: tl.constexpr,
    stride_C: tl.constexpr,
    BLOCK_S: tl.constexpr,
    N_DEPTHS: tl.constexpr,
    MAX_DEPTH_WIDTH: tl.constexpr,
    N_LEVELS: tl.constexpr,
    MAX_LEVEL_WIDTH: tl.constexpr,
    USE_ACTIVE_MASK: tl.constexpr,
    USE_ROW_STATS: tl.constexpr,
    DTYPE: tl.constexpr,
):
    """Uniform Pibar VJP with top-down species-tree denominator prefixes."""
    NEG_LARGE: tl.constexpr = -1e30

    row = tl.program_id(0)
    split_i = tl.where(row < n_ws, row, row - n_ws)
    is_right = row >= n_ws

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
    grad_base = split_i * S
    row_base = row * S

    if USE_ROW_STATS:
        row_max = tl.load(row_max_ptr + child)
        row_sum = tl.load(row_sum_ptr + child)
    else:
        row_max = tl.full([1], value=NEG_LARGE, dtype=DTYPE)
        row_sum = tl.full([1], value=0.0, dtype=DTYPE)
        for s_start in range(0, S, BLOCK_S):
            s_offs = s_start + tl.arange(0, BLOCK_S)
            valid_mask = s_offs < S
            mask = valid_mask & row_active
            pi_val = tl.load(Pi_star_ptr + pi_base + s_offs, mask=mask, other=NEG_LARGE)
            tile_max = tl.max(pi_val, axis=0)
            new_max = tl.maximum(row_max, tile_max)
            row_sum = row_sum * tl.exp2(row_max - new_max) + tl.sum(tl.exp2(pi_val - new_max), axis=0)
            row_max = new_max

    # subtree_buf first holds prefix[s] = sum_{a in ancestors(s)} p[a].
    for depth in range(0, N_DEPTHS):
        for n_start in range(0, MAX_DEPTH_WIDTH, BLOCK_S):
            n_offs = n_start + tl.arange(0, BLOCK_S)
            node = tl.load(
                depth_nodes_ptr + depth * MAX_DEPTH_WIDTH + n_offs,
                mask=n_offs < MAX_DEPTH_WIDTH,
                other=-1,
            )
            valid = (node >= 0) & (node < S) & row_active
            pi_node = tl.load(Pi_star_ptr + pi_base + node, mask=valid, other=NEG_LARGE)
            p_node = tl.exp2(pi_node - row_max)
            parent = tl.load(sp_parent_ptr + node, mask=valid, other=-1)
            has_parent = valid & (parent >= 0) & (parent < S)
            parent_prefix = tl.load(subtree_buf_ptr + row_base + parent, mask=has_parent, other=0.0)
            tl.store(subtree_buf_ptr + row_base + node, parent_prefix + p_node, mask=valid)
        tl.debug_barrier()

    A = tl.full([1], value=0.0, dtype=DTYPE)
    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        valid_mask = s_offs < S
        mask = valid_mask & row_active

        ancestor_sum = tl.load(subtree_buf_ptr + row_base + s_offs, mask=mask, other=0.0)
        denom = row_sum - ancestor_sum
        grad_l = tl.load(grad_Pibar_l_ptr + grad_base + s_offs, mask=mask, other=0.0)
        grad_r = tl.load(grad_Pibar_r_ptr + grad_base + s_offs, mask=mask, other=0.0)
        grad_u = tl.where(is_right, grad_r, grad_l)
        u_d = tl.where(denom > 0.0, grad_u / denom, tl.zeros([BLOCK_S], dtype=DTYPE))
        A += tl.sum(u_d, axis=0)
        tl.store(subtree_buf_ptr + row_base + s_offs, u_d, mask=mask)

    tl.debug_barrier()

    for level in range(0, N_LEVELS):
        for p_start in range(0, MAX_LEVEL_WIDTH, BLOCK_S):
            p_offs = p_start + tl.arange(0, BLOCK_S)
            parent = tl.load(
                level_parents_ptr + level * MAX_LEVEL_WIDTH + p_offs,
                mask=p_offs < MAX_LEVEL_WIDTH,
                other=-1,
            )
            parent_valid = (parent >= 0) & (parent < S) & row_active
            c1 = tl.load(sp_child1_ptr + parent, mask=parent_valid, other=S)
            c2 = tl.load(sp_child2_ptr + parent, mask=parent_valid, other=S)
            c1_valid = parent_valid & (c1 < S)
            c2_valid = parent_valid & (c2 < S)

            parent_val = tl.load(subtree_buf_ptr + row_base + parent, mask=parent_valid, other=0.0)
            c1_val = tl.load(subtree_buf_ptr + row_base + c1, mask=c1_valid, other=0.0)
            c2_val = tl.load(subtree_buf_ptr + row_base + c2, mask=c2_valid, other=0.0)
            tl.store(subtree_buf_ptr + row_base + parent, parent_val + c1_val + c2_val, mask=parent_valid)
        tl.debug_barrier()

    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        valid_mask = s_offs < S
        mask = valid_mask & row_active
        pi_val = tl.load(Pi_star_ptr + pi_base + s_offs, mask=mask, other=NEG_LARGE)
        p_prime = tl.exp2(pi_val - row_max)
        subtree_sum = tl.load(subtree_buf_ptr + row_base + s_offs, mask=mask, other=0.0)
        contrib = p_prime * (A - subtree_sum)
        tl.atomic_add(accumulated_rhs_ptr + pi_base + s_offs, contrib, sem="relaxed", mask=mask)


def uniform_cross_pibar_vjp_tree_fused(
    Pi_star,
    grad_Pibar_l,
    grad_Pibar_r,
    sl,
    sr,
    ancestor_cols,
    sp_child1,
    sp_child2,
    level_parents,
    accumulated_rhs,
    S,
    active_mask=None,
    reduce_idx=None,
    row_stats=None,
    Pibar_star=None,
    mt_squeezed=None,
    pibar_row_max=None,
):
    """Uniform-Pibar VJP using bottom-up descendant/subtree gathering."""
    n_ws = sl.shape[0]
    if n_ws == 0:
        return
    if active_mask is not None and reduce_idx is None:
        raise ValueError("reduce_idx is required when active_mask is provided")

    subtree_buf = torch.empty((2 * n_ws, S), device=Pi_star.device, dtype=Pi_star.dtype)
    BLOCK_S = min(256, triton.next_power_of_2(S))
    stride_C = Pi_star.stride(0)

    n_levels = level_parents.shape[0]
    max_level_width = level_parents.shape[1]
    use_pibar_denom_stats = (
        Pibar_star is not None
        and pibar_row_max is not None
        and mt_squeezed is not None
    )
    _uniform_cross_pibar_vjp_tree_kernel[(2 * n_ws,)](
        Pi_star,
        Pibar_star if use_pibar_denom_stats else Pi_star,
        mt_squeezed if use_pibar_denom_stats else grad_Pibar_l,
        pibar_row_max if use_pibar_denom_stats else sl,
        grad_Pibar_l,
        grad_Pibar_r,
        sl,
        sr,
        reduce_idx if reduce_idx is not None else sl,
        active_mask if active_mask is not None else grad_Pibar_l,
        ancestor_cols,
        row_stats[0] if row_stats is not None else Pi_star,
        row_stats[1] if row_stats is not None else Pi_star,
        sp_child1,
        sp_child2,
        level_parents,
        accumulated_rhs,
        subtree_buf,
        n_ws,
        S,
        stride_C,
        BLOCK_S,
        MAX_ANCESTOR_DEPTH=ancestor_cols.shape[0],
        N_LEVELS=n_levels,
        MAX_LEVEL_WIDTH=max_level_width,
        USE_ACTIVE_MASK=bool(active_mask is not None),
        USE_ROW_STATS=bool(row_stats is not None),
        USE_PIBAR_DENOM_STATS=bool(use_pibar_denom_stats),
        DTYPE=_tl_float_dtype(Pi_star.dtype),
        num_warps=4,
    )


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
def _uniform_cross_pibar_vjp_tree_from_ud_kernel(
    Pi_star_ptr,          # [C, S]
    pibar_ud_ptr,         # [2 * n_ws, S], initial subtree values u_d
    pibar_A_ptr,          # [2 * n_ws], sum_s u_d[s] per split side
    side_active_ptr,      # optional [2 * n_ws] bool exact-zero side skip mask
    sl_ptr,               # [n_ws]
    sr_ptr,               # [n_ws]
    reduce_idx_ptr,       # [n_ws], used with active_mask_ptr when enabled
    active_mask_ptr,      # optional [W] bool parent row activity mask
    pibar_row_max_ptr,    # [C], Pi-row max from forward uniform Pibar
    sp_child1_ptr,        # [S]
    sp_child2_ptr,        # [S]
    level_parents_ptr,    # [N_LEVELS, MAX_LEVEL_WIDTH]
    accumulated_rhs_ptr,  # [C, S], updated atomically
    n_ws: tl.constexpr,
    S: tl.constexpr,
    stride_C: tl.constexpr,
    BLOCK_S: tl.constexpr,
    N_LEVELS: tl.constexpr,
    MAX_LEVEL_WIDTH: tl.constexpr,
    USE_ACTIVE_MASK: tl.constexpr,
    USE_SIDE_ACTIVE: tl.constexpr,
    DTYPE: tl.constexpr,
):
    """Uniform Pibar VJP tree correction when DTS has already staged u_d."""
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

    # pibar_ud is intentionally reused in-place as subtree_buf.  It already
    # contains u_d for each species from the DTS kernel.
    tl.debug_barrier()
    for level in range(0, N_LEVELS):
        for p_start in range(0, MAX_LEVEL_WIDTH, BLOCK_S):
            p_offs = p_start + tl.arange(0, BLOCK_S)
            parent = tl.load(
                level_parents_ptr + level * MAX_LEVEL_WIDTH + p_offs,
                mask=p_offs < MAX_LEVEL_WIDTH,
                other=-1,
            )
            parent_valid = (parent >= 0) & (parent < S) & row_active
            c1 = tl.load(sp_child1_ptr + parent, mask=parent_valid, other=S)
            c2 = tl.load(sp_child2_ptr + parent, mask=parent_valid, other=S)
            c1_valid = parent_valid & (c1 < S)
            c2_valid = parent_valid & (c2 < S)

            parent_val = tl.load(pibar_ud_ptr + row_base + parent, mask=parent_valid, other=0.0)
            c1_val = tl.load(pibar_ud_ptr + row_base + c1, mask=c1_valid, other=0.0)
            c2_val = tl.load(pibar_ud_ptr + row_base + c2, mask=c2_valid, other=0.0)
            tl.store(pibar_ud_ptr + row_base + parent, parent_val + c1_val + c2_val, mask=parent_valid)
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


@triton.jit
def _uniform_cross_pibar_vjp_tree_from_ud_euler_prefix_kernel(
    Pi_star_ptr,          # [C, S]
    pibar_ud_ptr,         # [2 * n_ws, S], initial u_d; reused as prefix scratch
    pibar_A_ptr,          # [2 * n_ws], sum_s u_d[s] per split side
    side_active_ptr,      # optional [2 * n_ws] bool exact-zero side skip mask
    sl_ptr,               # [n_ws]
    sr_ptr,               # [n_ws]
    reduce_idx_ptr,       # [n_ws], used with active_mask_ptr when enabled
    active_mask_ptr,      # optional [W] bool parent row activity mask
    pibar_row_max_ptr,    # [C], Pi-row max from forward uniform Pibar
    subtree_start_ptr,    # [S], inclusive current-order descendant interval start
    subtree_end_ptr,      # [S], exclusive current-order descendant interval end
    accumulated_rhs_ptr,  # [C, S], updated atomically
    n_ws: tl.constexpr,
    S: tl.constexpr,
    stride_C: tl.constexpr,
    BLOCK_S: tl.constexpr,
    USE_ACTIVE_MASK: tl.constexpr,
    USE_SIDE_ACTIVE: tl.constexpr,
    DTYPE: tl.constexpr,
):
    """Uniform Pibar from-u_d correction using one current-order prefix scan.

    This path relies on species ids already being an Euler/postorder layout:
    every subtree is a single current-order interval.  `pibar_ud` is scratch at
    this point in the backward wave, so the kernel overwrites it with per-row
    inclusive prefix sums before reading interval sums back from global memory.
    """
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

    offs = tl.arange(0, BLOCK_S)
    valid = offs < S
    row_base = row * S
    pi_base = child * stride_C

    u = tl.load(pibar_ud_ptr + row_base + offs, mask=valid, other=0.0).to(DTYPE)
    prefix = tl.cumsum(u, axis=0)
    tl.store(pibar_ud_ptr + row_base + offs, prefix, mask=valid)
    tl.debug_barrier()

    start = tl.load(subtree_start_ptr + offs, mask=valid, other=0)
    end_exclusive = tl.load(subtree_end_ptr + offs, mask=valid, other=0)
    end_inclusive = end_exclusive - 1

    prefix_end = tl.load(
        pibar_ud_ptr + row_base + end_inclusive,
        mask=valid & (end_exclusive > start),
        other=0.0,
    ).to(DTYPE)
    prefix_before = tl.load(
        pibar_ud_ptr + row_base + start - 1,
        mask=valid & (start > 0),
        other=0.0,
    ).to(DTYPE)
    subtree_sum = prefix_end - prefix_before

    row_max = tl.load(pibar_row_max_ptr + child).to(DTYPE)
    A = tl.load(pibar_A_ptr + row).to(DTYPE)
    pi_val = tl.load(Pi_star_ptr + pi_base + offs, mask=valid & row_active, other=NEG_LARGE)
    p_prime = tl.exp2(pi_val - row_max)
    contrib = p_prime * (A - subtree_sum)
    tl.atomic_add(
        accumulated_rhs_ptr + pi_base + offs,
        contrib,
        sem="relaxed",
        mask=valid & row_active,
    )


def uniform_cross_pibar_vjp_tree_from_ud_fused(
    Pi_star,
    pibar_ud,
    pibar_A,
    sl,
    sr,
    sp_child1,
    sp_child2,
    level_parents,
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
    subtree_interval_start=None,
    subtree_interval_end=None,
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

    BLOCK_S = min(256, triton.next_power_of_2(S))
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
            num_warps=4,
        )

    use_compact_levels = (
        compact_level_ptr is not None
        and compact_level_parents is not None
        and compact_level_child1 is not None
        and compact_level_child2 is not None
    )
    use_euler_prefix = (
        subtree_interval_start is not None
        and subtree_interval_end is not None
        and os.environ.get("GPUREC_DTS_PIBAR_UD_EULER_PREFIX", "0") != "0"
    )
    cuda_pibar_from_ud_enabled = (
        os.environ.get("GPUREC_CUDA_PIBAR_FROM_UD", "0") != "0"
    )
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
                level_parents=level_parents,
                sp_child1=sp_child1,
                sp_child2=sp_child2,
                compact_level_ptr=compact_level_ptr,
                compact_level_parents=compact_level_parents,
                compact_level_child1=compact_level_child1,
                compact_level_child2=compact_level_child2,
            )
        except Exception as exc:
            if os.environ.get("GPUREC_CUDA_PIBAR_FROM_UD_STRICT", "0") != "0":
                raise
            import warnings

            global _cuda_pibar_from_ud_fallback_warned
            if not _cuda_pibar_from_ud_fallback_warned:
                warnings.warn(
                    "GPUREC_CUDA_PIBAR_FROM_UD=1 requested, but the CUDA "
                    f"prototype was unavailable ({exc}); falling back to Triton.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                _cuda_pibar_from_ud_fallback_warned = True

    if use_euler_prefix:
        euler_max_s = int(os.environ.get("GPUREC_DTS_PIBAR_UD_EULER_PREFIX_MAX_S", "4096"))
        if S > euler_max_s:
            raise ValueError(
                "Euler-prefix Pibar VJP requested for S="
                f"{S}, above GPUREC_DTS_PIBAR_UD_EULER_PREFIX_MAX_S={euler_max_s}"
            )
        BLOCK_EULER = triton.next_power_of_2(S)
        if BLOCK_EULER < 1:
            BLOCK_EULER = 1
        if subtree_interval_start.numel() != S or subtree_interval_end.numel() != S:
            raise ValueError("subtree interval arrays must have one entry per species")
        subtree_interval_start = subtree_interval_start.contiguous()
        subtree_interval_end = subtree_interval_end.contiguous()
        _uniform_cross_pibar_vjp_tree_from_ud_euler_prefix_kernel[(2 * n_ws,)](
            Pi_star,
            pibar_ud,
            pibar_A,
            side_active if side_active is not None else pibar_A,
            sl,
            sr,
            reduce_idx if reduce_idx is not None else sl,
            active_mask if active_mask is not None else pibar_ud,
            pibar_row_max,
            subtree_interval_start,
            subtree_interval_end,
            accumulated_rhs,
            n_ws,
            S,
            stride_C,
            BLOCK_EULER,
            USE_ACTIVE_MASK=bool(active_mask is not None),
            USE_SIDE_ACTIVE=bool(side_active is not None),
            DTYPE=_tl_float_dtype(Pi_star.dtype),
            num_warps=8,
        )
        return side_active

    if use_compact_levels:
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
            num_warps=4,
        )
        return side_active

    _uniform_cross_pibar_vjp_tree_from_ud_kernel[(2 * n_ws,)](
        Pi_star,
        pibar_ud,
        pibar_A,
        side_active if side_active is not None else pibar_A,
        sl,
        sr,
        reduce_idx if reduce_idx is not None else sl,
        active_mask if active_mask is not None else pibar_ud,
        pibar_row_max,
        sp_child1,
        sp_child2,
        level_parents,
        accumulated_rhs,
        n_ws,
        S,
        stride_C,
        BLOCK_S,
        N_LEVELS=level_parents.shape[0],
        MAX_LEVEL_WIDTH=level_parents.shape[1],
        USE_ACTIVE_MASK=bool(active_mask is not None),
        USE_SIDE_ACTIVE=bool(side_active is not None),
        DTYPE=_tl_float_dtype(Pi_star.dtype),
        num_warps=4,
    )
    return side_active


def uniform_cross_pibar_vjp_tree_prefix_fused(
    Pi_star,
    grad_Pibar_l,
    grad_Pibar_r,
    sl,
    sr,
    sp_parent,
    sp_child1,
    sp_child2,
    depth_nodes,
    level_parents,
    accumulated_rhs,
    S,
    active_mask=None,
    reduce_idx=None,
    row_stats=None,
):
    """Uniform-Pibar VJP using top-down denominator prefixes."""
    n_ws = sl.shape[0]
    if n_ws == 0:
        return
    if active_mask is not None and reduce_idx is None:
        raise ValueError("reduce_idx is required when active_mask is provided")

    subtree_buf = torch.empty((2 * n_ws, S), device=Pi_star.device, dtype=Pi_star.dtype)
    BLOCK_S = min(256, triton.next_power_of_2(S))
    stride_C = Pi_star.stride(0)

    _uniform_cross_pibar_vjp_tree_prefix_kernel[(2 * n_ws,)](
        Pi_star,
        grad_Pibar_l,
        grad_Pibar_r,
        sl,
        sr,
        reduce_idx if reduce_idx is not None else sl,
        active_mask if active_mask is not None else grad_Pibar_l,
        row_stats[0] if row_stats is not None else Pi_star,
        row_stats[1] if row_stats is not None else Pi_star,
        sp_parent,
        sp_child1,
        sp_child2,
        depth_nodes,
        level_parents,
        accumulated_rhs,
        subtree_buf,
        n_ws,
        S,
        stride_C,
        BLOCK_S,
        N_DEPTHS=depth_nodes.shape[0],
        MAX_DEPTH_WIDTH=depth_nodes.shape[1],
        N_LEVELS=level_parents.shape[0],
        MAX_LEVEL_WIDTH=level_parents.shape[1],
        USE_ACTIVE_MASK=bool(active_mask is not None),
        USE_ROW_STATS=bool(row_stats is not None),
        DTYPE=_tl_float_dtype(Pi_star.dtype),
        num_warps=4,
    )


def uniform_cross_pibar_vjp_tree_grouped_fused(
    Pi_star,
    grad_Pibar_l,
    grad_Pibar_r,
    group_children,
    group_inverse,
    ancestor_cols,
    sp_child1,
    sp_child2,
    level_parents,
    accumulated_rhs,
    S,
    active_mask=None,
    reduce_idx=None,
):
    """Reduce cross-DTS Pibar adjoints by child, then run one VJP per child."""
    n_ws = grad_Pibar_l.shape[0]
    n_groups = group_children.shape[0]
    if n_ws == 0 or n_groups == 0:
        return
    if active_mask is not None and reduce_idx is None:
        raise ValueError("reduce_idx is required when active_mask is provided")

    BLOCK_S = min(256, triton.next_power_of_2(S))
    grouped_grad = torch.zeros((n_groups, S), device=Pi_star.device, dtype=Pi_star.dtype)
    track_group_active = active_mask is not None
    group_active = (
        torch.zeros((n_groups,), device=Pi_star.device, dtype=torch.bool)
        if track_group_active
        else grouped_grad
    )

    _group_cross_pibar_grad_kernel[(2 * n_ws, triton.cdiv(S, BLOCK_S))](
        grad_Pibar_l,
        grad_Pibar_r,
        group_inverse,
        reduce_idx if reduce_idx is not None else group_inverse,
        active_mask if active_mask is not None else grouped_grad,
        grouped_grad,
        group_active,
        n_ws,
        S,
        BLOCK_S,
        USE_ACTIVE_MASK=track_group_active,
        TRACK_GROUP_ACTIVE=track_group_active,
        num_warps=4,
    )

    subtree_buf = torch.empty((n_groups, S), device=Pi_star.device, dtype=Pi_star.dtype)
    stride_C = Pi_star.stride(0)
    _uniform_cross_pibar_vjp_tree_grouped_kernel[(n_groups,)](
        Pi_star,
        grouped_grad,
        group_children,
        ancestor_cols,
        sp_child1,
        sp_child2,
        level_parents,
        accumulated_rhs,
        subtree_buf,
        group_active,
        n_groups,
        S,
        stride_C,
        BLOCK_S,
        MAX_ANCESTOR_DEPTH=ancestor_cols.shape[0],
        N_LEVELS=level_parents.shape[0],
        MAX_LEVEL_WIDTH=level_parents.shape[1],
        USE_GROUP_ACTIVE=track_group_active,
        DTYPE=_tl_float_dtype(Pi_star.dtype),
        num_warps=4,
    )


@triton.jit
def _uniform_cross_pibar_vjp_grouped_tree_kernel(
    Pi_star_ptr,          # [C, S]
    child_ids_ptr,        # [n_child_rows]
    grad_u_ptr,           # [n_child_rows, S], already reduced by child clade
    ancestor_cols_ptr,    # [MAX_ANCESTOR_DEPTH, S]
    row_max_ptr,          # optional [C]
    row_sum_ptr,          # optional [C]
    sp_child1_ptr,        # [S]
    sp_child2_ptr,        # [S]
    level_parents_ptr,    # [N_LEVELS, MAX_LEVEL_WIDTH]
    accumulated_rhs_ptr,  # [C, S], updated atomically
    subtree_buf_ptr,      # [n_child_rows, S]
    n_child_rows: tl.constexpr,
    S: tl.constexpr,
    stride_C: tl.constexpr,
    BLOCK_S: tl.constexpr,
    MAX_ANCESTOR_DEPTH: tl.constexpr,
    N_LEVELS: tl.constexpr,
    MAX_LEVEL_WIDTH: tl.constexpr,
    USE_ROW_STATS: tl.constexpr,
    DTYPE: tl.constexpr,
):
    """Uniform Pibar VJP after reducing incoming Pibar adjoints by child row."""
    NEG_LARGE: tl.constexpr = -1e30

    row = tl.program_id(0)
    child = tl.load(child_ids_ptr + row).to(tl.int64)
    pi_base = child * stride_C
    row_base = row * S

    if USE_ROW_STATS:
        row_max = tl.load(row_max_ptr + child)
        row_sum = tl.load(row_sum_ptr + child)
    else:
        row_max = tl.full([1], value=NEG_LARGE, dtype=DTYPE)
        row_sum = tl.full([1], value=0.0, dtype=DTYPE)
        for s_start in range(0, S, BLOCK_S):
            s_offs = s_start + tl.arange(0, BLOCK_S)
            mask = s_offs < S
            pi_val = tl.load(Pi_star_ptr + pi_base + s_offs, mask=mask, other=NEG_LARGE)
            tile_max = tl.max(pi_val, axis=0)
            new_max = tl.maximum(row_max, tile_max)
            row_sum = row_sum * tl.exp2(row_max - new_max) + tl.sum(tl.exp2(pi_val - new_max), axis=0)
            row_max = new_max

    A = tl.full([1], value=0.0, dtype=DTYPE)
    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        mask = s_offs < S

        ancestor_sum = tl.zeros([BLOCK_S], dtype=DTYPE)
        for k in range(0, MAX_ANCESTOR_DEPTH):
            anc = tl.load(ancestor_cols_ptr + k * S + s_offs, mask=mask, other=-1)
            anc_valid = mask & (anc >= 0) & (anc < S)
            pi_anc = tl.load(Pi_star_ptr + pi_base + anc, mask=anc_valid, other=NEG_LARGE)
            ancestor_sum += tl.where(
                anc_valid,
                tl.exp2(pi_anc - row_max),
                tl.zeros([BLOCK_S], dtype=DTYPE),
            )

        denom = row_sum - ancestor_sum
        grad_u = tl.load(grad_u_ptr + row_base + s_offs, mask=mask, other=0.0)
        u_d = tl.where(denom > 0.0, grad_u / denom, tl.zeros([BLOCK_S], dtype=DTYPE))
        A += tl.sum(u_d, axis=0)
        tl.store(subtree_buf_ptr + row_base + s_offs, u_d, mask=mask)

    tl.debug_barrier()

    for level in range(0, N_LEVELS):
        for p_start in range(0, MAX_LEVEL_WIDTH, BLOCK_S):
            p_offs = p_start + tl.arange(0, BLOCK_S)
            parent = tl.load(
                level_parents_ptr + level * MAX_LEVEL_WIDTH + p_offs,
                mask=p_offs < MAX_LEVEL_WIDTH,
                other=-1,
            )
            parent_valid = (parent >= 0) & (parent < S)
            c1 = tl.load(sp_child1_ptr + parent, mask=parent_valid, other=S)
            c2 = tl.load(sp_child2_ptr + parent, mask=parent_valid, other=S)
            c1_valid = parent_valid & (c1 < S)
            c2_valid = parent_valid & (c2 < S)

            parent_val = tl.load(subtree_buf_ptr + row_base + parent, mask=parent_valid, other=0.0)
            c1_val = tl.load(subtree_buf_ptr + row_base + c1, mask=c1_valid, other=0.0)
            c2_val = tl.load(subtree_buf_ptr + row_base + c2, mask=c2_valid, other=0.0)
            tl.store(subtree_buf_ptr + row_base + parent, parent_val + c1_val + c2_val, mask=parent_valid)
        tl.debug_barrier()

    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        mask = s_offs < S
        pi_val = tl.load(Pi_star_ptr + pi_base + s_offs, mask=mask, other=NEG_LARGE)
        p_prime = tl.exp2(pi_val - row_max)
        subtree_sum = tl.load(subtree_buf_ptr + row_base + s_offs, mask=mask, other=0.0)
        contrib = p_prime * (A - subtree_sum)
        tl.atomic_add(accumulated_rhs_ptr + pi_base + s_offs, contrib, sem="relaxed", mask=mask)


def uniform_cross_pibar_vjp_grouped_tree_fused(
    Pi_star,
    child_ids,
    grad_u,
    ancestor_cols,
    sp_child1,
    sp_child2,
    level_parents,
    accumulated_rhs,
    S,
    row_stats=None,
):
    """Uniform-Pibar VJP for pre-reduced unique child rows."""
    n_child_rows = child_ids.shape[0]
    if n_child_rows == 0:
        return

    subtree_buf = torch.empty((n_child_rows, S), device=Pi_star.device, dtype=Pi_star.dtype)
    BLOCK_S = min(256, triton.next_power_of_2(S))
    stride_C = Pi_star.stride(0)

    _uniform_cross_pibar_vjp_grouped_tree_kernel[(n_child_rows,)](
        Pi_star,
        child_ids,
        grad_u,
        ancestor_cols,
        row_stats[0] if row_stats is not None else Pi_star,
        row_stats[1] if row_stats is not None else Pi_star,
        sp_child1,
        sp_child2,
        level_parents,
        accumulated_rhs,
        subtree_buf,
        n_child_rows,
        S,
        stride_C,
        BLOCK_S,
        MAX_ANCESTOR_DEPTH=ancestor_cols.shape[0],
        N_LEVELS=level_parents.shape[0],
        MAX_LEVEL_WIDTH=level_parents.shape[1],
        USE_ROW_STATS=bool(row_stats is not None),
        DTYPE=_tl_float_dtype(Pi_star.dtype),
        num_warps=4,
    )
