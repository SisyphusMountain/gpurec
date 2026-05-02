"""Fused Triton kernels for wave-backward.

Two kernels:
1. _wave_backward_uniform_kernel: self-loop backward (Neumann VJP + param VJP)
2. _dts_cross_backward_kernel: cross-clade DTS backward (adjoint propagation + param VJP)

Both use one CTA per work-item, multi-pass over species dimension.
"""

import torch
import triton
import triton.language as tl


def _tl_float_dtype(dtype):
    return tl.float64 if dtype == torch.float64 else tl.float32


@triton.jit
def _wave_backward_uniform_kernel(
    # Converged values from forward pass
    Pi_star_ptr,      # [C, S] — read rows [ws:ws+W]
    Pibar_star_ptr,   # [C, S] — read rows [ws:ws+W]
    dts_r_ptr,        # [W, S] or None — cross-clade DTS
    has_splits: tl.constexpr,
    # Incoming adjoint
    rhs_ptr,          # [W, S]
    # Constants [S]
    mt_ptr, DL_const_ptr, Ebar_ptr, E_ptr, SL1_const_ptr, SL2_const_ptr,
    # Species children [S] long
    sp_child1_ptr, sp_child2_ptr,
    # Leaf term [W, S]
    leaf_term_ptr,
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
    # Dimensions
    ws,               # wave start offset into [C, S]
    S: tl.constexpr,
    stride: tl.constexpr,
    BLOCK_S: tl.constexpr,
    NEUMANN_TERMS: tl.constexpr,
    DTYPE: tl.constexpr,
):
    """Fused backward kernel for uniform Pibar mode.

    Per clade w, computes:
    1. Softmax weights (w_L, w_terms) from converged Pi/Pibar
    2. Neumann series: v_k = (I + J^T + (J^T)^2 + ...) @ rhs
    3. Param VJP element-wise contributions

    The Neumann J^T application needs A = sum_s(u_d[s]) — a full-row reduction.
    Each iteration uses 2 sub-passes:
      Pass A: compute u_d[s], accumulate A, write spec scatter to buffer
      Pass B: compute result[s] using A, read spec scatter from buffer
    """
    NEG_LARGE = tl.full([1], value=-1e30, dtype=DTYPE)

    w = tl.program_id(0)
    pi_base = (ws + w) * stride      # offset into [C, S]
    out_base = w * stride             # offset into [W, S]

    # ================================================================
    # Pass 1: Row statistics for uniform Pibar (same as forward)
    # ================================================================
    row_max = tl.full([1], value=-1e30, dtype=DTYPE)
    row_sum = tl.full([1], value=0.0, dtype=DTYPE)

    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        mask = s_offs < S
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
        mask = s_offs < S
        off = out_base + s_offs

        # Load Pi*, Pibar*
        pi_w = tl.load(Pi_star_ptr + pi_base + s_offs, mask=mask, other=-1e30)
        pibar_w = tl.load(Pibar_star_ptr + pi_base + s_offs, mask=mask, other=-1e30)

        # Load constants
        dl_c = tl.load(DL_const_ptr + s_offs, mask=mask, other=-1e30)
        ebar = tl.load(Ebar_ptr + s_offs, mask=mask, other=-1e30)
        e_val = tl.load(E_ptr + s_offs, mask=mask, other=-1e30)
        sl1_c = tl.load(SL1_const_ptr + s_offs, mask=mask, other=-1e30)
        sl2_c = tl.load(SL2_const_ptr + s_offs, mask=mask, other=-1e30)

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

        # Pibar VJP ingredients: inv_denom = 1 / (row_sum - p_prime)
        p_prime = tl.exp2(pi_w - row_max)
        denom = row_sum - p_prime
        inv_denom = tl.where(denom > 0, 1.0 / denom, tl.zeros_like(denom))

        # Store precomputed weights to scratch buffers
        diag_wt = w_L * (wt0 + wt1)        # diagonal J^T weight
        pibar_wt = w_L * wt2               # Pibar path weight
        sl1_wt = w_L * wt3                 # SL1 speciation weight
        sl2_wt = w_L * wt4                 # SL2 speciation weight

        tl.store(aw0_ptr + off, diag_wt, mask=mask)
        tl.store(aw1_ptr + off, pibar_wt, mask=mask)
        tl.store(aw2_ptr + off, inv_denom, mask=mask)
        tl.store(aw3_ptr + off, p_prime, mask=mask)
        tl.store(aw4_ptr + off, sl1_wt, mask=mask)
        tl.store(aw345_ptr + off, sl2_wt, mask=mask)

    # ================================================================
    # Neumann series: v = rhs + J^T(rhs) + (J^T)^2(rhs) + ...
    #
    # Each J^T application on vector `term` requires:
    #   Pass A: compute u_d = term * pibar_wt * inv_denom, accumulate A = sum(u_d)
    #           also scatter speciation: spec_buf[child[s]] = term[s] * sl_wt[s]
    #   Pass B: result[s] = term[s] * diag_wt[s] + p_prime[s] * (A - u_d[s])
    #                        + spec_buf[s] (read back speciation contribution)
    # ================================================================
    # Copy rhs → v_k (v_k accumulates the Neumann sum)
    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        mask = s_offs < S
        rhs_val = tl.load(rhs_ptr + out_base + s_offs, mask=mask, other=0.0)
        tl.store(v_k_ptr + out_base + s_offs, rhs_val, mask=mask)

    # Buffer ping-pong: even iterations read rhs_ptr, write spec_buf;
    # odd iterations read spec_buf, write rhs_ptr. Output buffer is zeroed
    # at the start of each iteration to avoid stale data at non-child positions.

    for _n in range(NEUMANN_TERMS):
        # Zero the output buffer before scatter writes.
        # Sub-pass A only writes to child positions (scatter); sub-pass B reads ALL positions.
        # Without zeroing, non-child positions would have stale data from prior iterations.
        for s_start in range(0, S, BLOCK_S):
            s_offs = s_start + tl.arange(0, BLOCK_S)
            mask = s_offs < S
            if _n % 2 == 0:
                tl.store(spec_buf_ptr + out_base + s_offs,
                         tl.zeros(s_offs.shape, dtype=DTYPE), mask=mask)
            else:
                tl.store(rhs_ptr + out_base + s_offs,
                         tl.zeros(s_offs.shape, dtype=DTYPE), mask=mask)

        # --- Sub-pass A: accumulate A = sum_s(term * pibar_wt * inv_denom) ---
        # Also write speciation scatter contributions.
        A_acc = tl.full([1], value=0.0, dtype=DTYPE)

        for s_start in range(0, S, BLOCK_S):
            s_offs = s_start + tl.arange(0, BLOCK_S)
            mask = s_offs < S
            off = out_base + s_offs

            # Load term from appropriate buffer
            if _n == 0:
                term_val = tl.load(rhs_ptr + off, mask=mask, other=0.0)
            elif _n % 2 == 1:
                term_val = tl.load(spec_buf_ptr + off, mask=mask, other=0.0)
            else:
                term_val = tl.load(rhs_ptr + off, mask=mask, other=0.0)

            pibar_wt = tl.load(aw1_ptr + off, mask=mask, other=0.0)
            inv_denom = tl.load(aw2_ptr + off, mask=mask, other=0.0)

            u_d = term_val * pibar_wt * inv_denom
            A_acc += tl.sum(u_d, axis=0)

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
                tl.store(rhs_ptr + out_base + c1, src1, mask=c1_valid)
                tl.store(rhs_ptr + out_base + c2, src2, mask=c2_valid)

        # --- Sub-pass B: compute J^T result using A ---
        for s_start in range(0, S, BLOCK_S):
            s_offs = s_start + tl.arange(0, BLOCK_S)
            mask = s_offs < S
            off = out_base + s_offs

            # Reload term and weights
            if _n == 0:
                term_val = tl.load(rhs_ptr + off, mask=mask, other=0.0)
            elif _n % 2 == 1:
                term_val = tl.load(spec_buf_ptr + off, mask=mask, other=0.0)
            else:
                term_val = tl.load(rhs_ptr + off, mask=mask, other=0.0)

            diag_wt = tl.load(aw0_ptr + off, mask=mask, other=0.0)
            pibar_wt = tl.load(aw1_ptr + off, mask=mask, other=0.0)
            inv_denom = tl.load(aw2_ptr + off, mask=mask, other=0.0)
            p_prime = tl.load(aw3_ptr + off, mask=mask, other=0.0)

            u_d = term_val * pibar_wt * inv_denom
            result = term_val * diag_wt + p_prime * (A_acc - u_d)

            # Add speciation contribution (written to output buffer in sub-pass A)
            if _n % 2 == 0:
                spec_val = tl.load(spec_buf_ptr + off, mask=mask, other=0.0)
            else:
                spec_val = tl.load(rhs_ptr + off, mask=mask, other=0.0)
            result = result + spec_val

            # Store result to output buffer
            if _n % 2 == 0:
                tl.store(spec_buf_ptr + off, result, mask=mask)
            else:
                tl.store(rhs_ptr + off, result, mask=mask)

            # Accumulate into v_k
            v_k_val = tl.load(v_k_ptr + off, mask=mask, other=0.0)
            tl.store(v_k_ptr + off, v_k_val + result, mask=mask)

    # ================================================================
    # Pass final: Param VJP contributions
    # Recompute alpha = v_k * w_L and weighted terms.
    # Store per-element contributions for the caller to reduce.
    # ================================================================
    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        mask = s_offs < S
        off = out_base + s_offs

        v_k_val = tl.load(v_k_ptr + off, mask=mask, other=0.0)

        # Reload Pi and Pibar to recompute weights
        # (we overwrote aw* buffers with Jt scratch data)
        pi_w = tl.load(Pi_star_ptr + pi_base + s_offs, mask=mask, other=-1e30)
        pibar_w = tl.load(Pibar_star_ptr + pi_base + s_offs, mask=mask, other=-1e30)
        dl_c = tl.load(DL_const_ptr + s_offs, mask=mask, other=-1e30)
        ebar = tl.load(Ebar_ptr + s_offs, mask=mask, other=-1e30)
        e_val = tl.load(E_ptr + s_offs, mask=mask, other=-1e30)
        sl1_c = tl.load(SL1_const_ptr + s_offs, mask=mask, other=-1e30)
        sl2_c = tl.load(SL2_const_ptr + s_offs, mask=mask, other=-1e30)
        c1 = tl.load(sp_child1_ptr + s_offs, mask=mask, other=0)
        c2 = tl.load(sp_child2_ptr + s_offs, mask=mask, other=0)
        c1_valid = c1 < S
        c2_valid = c2 < S
        pi_s1 = tl.load(Pi_star_ptr + pi_base + c1, mask=mask & c1_valid, other=-1e30)
        pi_s2 = tl.load(Pi_star_ptr + pi_base + c2, mask=mask & c2_valid, other=-1e30)
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

        tl.store(aw0_ptr + off, _aw0, mask=mask)
        tl.store(aw1_ptr + off, _aw1, mask=mask)
        tl.store(aw2_ptr + off, _aw2, mask=mask)
        tl.store(aw345_ptr + off, _aw3 + _aw4 + _aw5, mask=mask)
        tl.store(aw3_ptr + off, _aw3, mask=mask)
        tl.store(aw4_ptr + off, _aw4, mask=mask)


def wave_backward_uniform_fused(
    Pi_star, Pibar_star, ws, W, S,
    dts_r,
    rhs,
    mt_squeezed, DL_const, Ebar, E, SL1_const, SL2_const,
    sp_child1, sp_child2, leaf_term_wt,
    neumann_terms=3,
):
    """Fused backward: precompute + Neumann + param VJP in one kernel per wave.

    Args:
        Pi_star: [C, S] converged Pi
        Pibar_star: [C, S] converged Pibar
        ws: wave start offset
        W: wave size
        S: number of species
        dts_r: [W, S] or None
        rhs: [W, S] incoming adjoint (WILL BE OVERWRITTEN as scratch)
        mt_squeezed, DL_const, Ebar, E, SL1_const, SL2_const: [S]
        sp_child1, sp_child2: [S] long
        leaf_term_wt: [W, S]
        neumann_terms: int

    Returns:
        v_k: [W, S] Neumann-solved adjoint
        aw0, aw1, aw2, aw345, aw3, aw4: [W, S] per-element param grad contributions
    """
    device = Pi_star.device
    dtype = Pi_star.dtype

    v_k = torch.empty((W, S), device=device, dtype=dtype)
    aw0 = torch.empty((W, S), device=device, dtype=dtype)
    aw1 = torch.empty((W, S), device=device, dtype=dtype)
    aw2 = torch.empty((W, S), device=device, dtype=dtype)
    aw345 = torch.empty((W, S), device=device, dtype=dtype)
    aw3 = torch.empty((W, S), device=device, dtype=dtype)
    aw4 = torch.empty((W, S), device=device, dtype=dtype)
    spec_buf = torch.zeros((W, S), device=device, dtype=dtype)

    has_splits = dts_r is not None

    BLOCK_S = min(256, triton.next_power_of_2(S))

    grid = (W,)
    _wave_backward_uniform_kernel[grid](
        Pi_star, Pibar_star,
        dts_r if has_splits else Pi_star,  # dummy ptr when no splits
        has_splits,
        rhs,
        mt_squeezed, DL_const, Ebar, E, SL1_const, SL2_const,
        sp_child1, sp_child2,
        leaf_term_wt,
        v_k,
        aw0, aw1, aw2, aw345, aw3, aw4,
        spec_buf,
        ws, S, S, BLOCK_S,
        neumann_terms,
        DTYPE=_tl_float_dtype(dtype),
    )

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
    # Split metadata
    sl_ptr,            # [n_ws] int64 — left child global clade index
    sr_ptr,            # [n_ws] int64 — right child global clade index
    reduce_idx_ptr,    # [n_ws] int64 — wave-local parent index
    wlsp_ptr,          # [n_ws] float — log split probability (squeezed)
    # Scalar params
    log_pD,            # float
    log_pS,            # float
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
    sl = tl.load(sl_ptr + i)
    sr = tl.load(sr_ptr + i)
    parent_w = tl.load(reduce_idx_ptr + i)
    wlsp = tl.load(wlsp_ptr + i)

    # Base offsets into [C, S] for child clades
    pi_l_base = sl * stride_C
    pi_r_base = sr * stride_C
    pibar_l_base = sl * stride_C
    pibar_r_base = sr * stride_C
    # Parent clade in Pi_star: row (ws + parent_w)
    parent_pi_base = (ws + parent_w) * stride_C
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
        mask = s_offs < S

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
        tl.store(grad_Pi_l_ptr + out_base + s_offs, vd0 + vd1, mask=mask)
        tl.store(grad_Pi_r_ptr + out_base + s_offs, vd0 + vd2, mask=mask)
        tl.store(grad_Pibar_l_ptr + out_base + s_offs, vd2, mask=mask)
        tl.store(grad_Pibar_r_ptr + out_base + s_offs, vd1, mask=mask)

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
        mask = s_offs < S

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

    # Extract scalar param values
    pD_val = float(log_pD) if log_pD.ndim == 0 else float(log_pD.item())
    pS_val = float(log_pS) if log_pS.ndim == 0 else float(log_pS.item())

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
        sl, sr, reduce_idx, wlsp_flat,
        pD_val, pS_val,
        sp_child1, sp_child2,
        grad_Pi_l, grad_Pi_r, grad_Pibar_l, grad_Pibar_r,
        param_pD, param_pS,
        ws, S, stride_C, BLOCK_S,
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
    # Split metadata
    sl_ptr,            # [n_ws] int64 — left child global clade index
    sr_ptr,            # [n_ws] int64 — right child global clade index
    reduce_idx_ptr,    # [n_ws] int64 — wave-local parent index
    wlsp_ptr,          # [n_ws] float — log split probability (squeezed)
    # Scalar params
    log_pD,            # float
    log_pS,            # float
    # Species children [S] int64
    sp_child1_ptr,
    sp_child2_ptr,
    # Outputs
    accumulated_rhs_ptr,  # [C, S], direct Pi adjoints updated atomically
    grad_Pibar_l_ptr,     # [n_ws, S]
    grad_Pibar_r_ptr,     # [n_ws, S]
    param_pD_ptr,         # [n_ws]
    param_pS_ptr,         # [n_ws]
    # Dimensions
    ws,                # wave start offset (parent row = ws + reduce_idx)
    S: tl.constexpr,
    stride_C: tl.constexpr,
    BLOCK_S: tl.constexpr,
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

    sl = tl.load(sl_ptr + i)
    sr = tl.load(sr_ptr + i)
    parent_w = tl.load(reduce_idx_ptr + i)
    wlsp = tl.load(wlsp_ptr + i)

    pi_l_base = sl * stride_C
    pi_r_base = sr * stride_C
    pibar_l_base = sl * stride_C
    pibar_r_base = sr * stride_C
    parent_pi_base = (ws + parent_w) * stride_C
    parent_vk_base = parent_w * S
    out_base = i * S

    sum_pD = tl.zeros((1,), dtype=DTYPE)
    sum_pS = tl.zeros((1,), dtype=DTYPE)
    _scalar_off = tl.arange(0, 1)

    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        mask = s_offs < S

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

        d0 = log_pD + Pi_l + Pi_r
        d1 = Pi_l + Pibar_r
        d2 = Pi_r + Pibar_l
        d3 = log_pS + Pi_l_s1 + Pi_r_s2
        d4 = log_pS + Pi_r_s1 + Pi_l_s2

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

        tl.atomic_add(accumulated_rhs_ptr + pi_l_base + s_offs, vd0 + vd1, sem="relaxed", mask=mask)
        tl.atomic_add(accumulated_rhs_ptr + pi_r_base + s_offs, vd0 + vd2, sem="relaxed", mask=mask)
        tl.store(grad_Pibar_l_ptr + out_base + s_offs, vd2, mask=mask)
        tl.store(grad_Pibar_r_ptr + out_base + s_offs, vd1, mask=mask)

        sum_pD += tl.sum(vd0, axis=0)
        sum_pS += tl.sum(vd3 + vd4, axis=0)

    tl.store(param_pD_ptr + i + _scalar_off, sum_pD)
    tl.store(param_pS_ptr + i + _scalar_off, sum_pS)

    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        mask = s_offs < S

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

        tl.atomic_add(accumulated_rhs_ptr + pi_l_base + c1, vd3, sem="relaxed", mask=c1_valid)
        tl.atomic_add(accumulated_rhs_ptr + pi_r_base + c1, vd4, sem="relaxed", mask=c1_valid)
        tl.atomic_add(accumulated_rhs_ptr + pi_r_base + c2, vd3, sem="relaxed", mask=c2_valid)
        tl.atomic_add(accumulated_rhs_ptr + pi_l_base + c2, vd4, sem="relaxed", mask=c2_valid)


def dts_cross_backward_accum_fused(
    Pi_star, Pibar_star, v_k, ws,
    sl, sr, reduce_idx, wlsp,
    log_pD, log_pS,
    sp_child1, sp_child2,
    accumulated_rhs,
    S,
):
    """Fused DTS backward that atomically accumulates direct Pi adjoints."""
    n_ws = sl.shape[0]
    device = Pi_star.device
    dtype = Pi_star.dtype

    wlsp_flat = wlsp.squeeze(-1) if wlsp.ndim > 1 else wlsp
    pD_val = float(log_pD) if log_pD.ndim == 0 else float(log_pD.item())
    pS_val = float(log_pS) if log_pS.ndim == 0 else float(log_pS.item())

    grad_Pibar_l = torch.empty((n_ws, S), device=device, dtype=dtype)
    grad_Pibar_r = torch.empty((n_ws, S), device=device, dtype=dtype)
    param_pD = torch.empty(n_ws, device=device, dtype=dtype)
    param_pS = torch.empty(n_ws, device=device, dtype=dtype)

    stride_C = Pi_star.stride(0)
    BLOCK_S = min(256, triton.next_power_of_2(S))

    _dts_cross_backward_accum_kernel[(n_ws,)](
        Pi_star, Pibar_star,
        v_k,
        sl, sr, reduce_idx, wlsp_flat,
        pD_val, pS_val,
        sp_child1, sp_child2,
        accumulated_rhs,
        grad_Pibar_l, grad_Pibar_r,
        param_pD, param_pS,
        ws, S, stride_C, BLOCK_S,
        DTYPE=_tl_float_dtype(dtype),
    )

    return grad_Pibar_l, grad_Pibar_r, param_pD, param_pS


# =========================================================================
# Uniform Pibar VJP for cross-clade gradients
# =========================================================================

@triton.jit
def _uniform_cross_pibar_vjp_kernel(
    Pi_star_ptr,          # [C, S]
    grad_Pibar_l_ptr,     # [n_ws, S]
    grad_Pibar_r_ptr,     # [n_ws, S]
    sl_ptr,               # [n_ws]
    sr_ptr,               # [n_ws]
    ancestor_cols_ptr,    # [MAX_ANCESTOR_DEPTH, S]
    accumulated_rhs_ptr,  # [C, S], updated atomically
    correction_buf_ptr,   # [2 * n_ws, S]
    n_ws: tl.constexpr,
    S: tl.constexpr,
    stride_C: tl.constexpr,
    BLOCK_S: tl.constexpr,
    MAX_ANCESTOR_DEPTH: tl.constexpr,
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

    child_l = tl.load(sl_ptr + split_i)
    child_r = tl.load(sr_ptr + split_i)
    child = tl.where(is_right, child_r, child_l)

    pi_base = child * stride_C
    grad_base = split_i * S
    corr_base = row * S

    # Clear the correction row owned by this program.
    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        mask = s_offs < S
        tl.store(correction_buf_ptr + corr_base + s_offs,
                 tl.zeros([BLOCK_S], dtype=DTYPE), mask=mask)

    # Row max and shifted row sum for p = exp2(Pi - row_max).
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

    # Build correction[f] = sum_{s: f ancestor of s} u_d[s].
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
        mask = s_offs < S
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

    correction_buf = torch.empty((2 * n_ws, S), device=Pi_star.device, dtype=Pi_star.dtype)
    BLOCK_S = min(256, triton.next_power_of_2(S))
    stride_C = Pi_star.stride(0)

    _uniform_cross_pibar_vjp_kernel[(2 * n_ws,)](
        Pi_star,
        grad_Pibar_l,
        grad_Pibar_r,
        sl,
        sr,
        ancestor_cols,
        accumulated_rhs,
        correction_buf,
        n_ws,
        S,
        stride_C,
        BLOCK_S,
        MAX_ANCESTOR_DEPTH=ancestor_cols.shape[0],
        DTYPE=_tl_float_dtype(Pi_star.dtype),
        num_warps=4,
    )


@triton.jit
def _uniform_cross_pibar_vjp_tree_kernel(
    Pi_star_ptr,          # [C, S]
    grad_Pibar_l_ptr,     # [n_ws, S]
    grad_Pibar_r_ptr,     # [n_ws, S]
    sl_ptr,               # [n_ws]
    sr_ptr,               # [n_ws]
    ancestor_cols_ptr,    # [MAX_ANCESTOR_DEPTH, S]
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

    child_l = tl.load(sl_ptr + split_i)
    child_r = tl.load(sr_ptr + split_i)
    child = tl.where(is_right, child_r, child_l)

    pi_base = child * stride_C
    grad_base = split_i * S
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

    # Initial subtree_buf contains u_d for every species.  Leaves are already
    # final subtree sums; internal nodes are completed by the bottom-up pass.
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
        grad_l = tl.load(grad_Pibar_l_ptr + grad_base + s_offs, mask=mask, other=0.0)
        grad_r = tl.load(grad_Pibar_r_ptr + grad_base + s_offs, mask=mask, other=0.0)
        grad_u = tl.where(is_right, grad_r, grad_l)
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
        # Duplicate child clades across splits still require an atomic add into
        # accumulated_rhs.  The subtree correction itself is atomic-free.
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
):
    """Uniform-Pibar VJP using bottom-up descendant/subtree gathering."""
    n_ws = sl.shape[0]
    if n_ws == 0:
        return

    subtree_buf = torch.empty((2 * n_ws, S), device=Pi_star.device, dtype=Pi_star.dtype)
    BLOCK_S = min(256, triton.next_power_of_2(S))
    stride_C = Pi_star.stride(0)

    n_levels = level_parents.shape[0]
    max_level_width = level_parents.shape[1]
    _uniform_cross_pibar_vjp_tree_kernel[(2 * n_ws,)](
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
        subtree_buf,
        n_ws,
        S,
        stride_C,
        BLOCK_S,
        MAX_ANCESTOR_DEPTH=ancestor_cols.shape[0],
        N_LEVELS=n_levels,
        MAX_LEVEL_WIDTH=max_level_width,
        DTYPE=_tl_float_dtype(Pi_star.dtype),
        num_warps=4,
    )
