"""Fused Triton kernel for wave-backward (Neumann VJP + param VJP).

Mirrors the forward wave_step_uniform_fused kernel structure:
one CTA per clade, multi-pass over species dimension.
"""

import torch
import triton
import triton.language as tl


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
):
    """Fused backward kernel for uniform_approx Pibar mode.

    Per clade w, computes:
    1. Softmax weights (w_L, w_terms) from converged Pi/Pibar
    2. Neumann series: v_k = (I + J^T + (J^T)^2 + ...) @ rhs
    3. Param VJP element-wise contributions

    The Neumann J^T application needs A = sum_s(u_d[s]) — a full-row reduction.
    Each iteration uses 2 sub-passes:
      Pass A: compute u_d[s], accumulate A, write spec scatter to buffer
      Pass B: compute result[s] using A, read spec scatter from buffer
    """
    NEG_LARGE = tl.full([1], value=-1e30, dtype=tl.float32)

    w = tl.program_id(0)
    pi_base = (ws + w) * stride      # offset into [C, S]
    out_base = w * stride             # offset into [W, S]

    # ================================================================
    # Pass 1: Row statistics for uniform Pibar (same as forward)
    # ================================================================
    row_max = tl.full([1], value=-1e30, dtype=tl.float32)
    row_sum = tl.full([1], value=0.0, dtype=tl.float32)

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
    M_SAFE = tl.full([1], value=-1e29, dtype=tl.float32)

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
            w_L = tl.full(s_offs.shape, value=1.0, dtype=tl.float32)

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
                         tl.zeros(s_offs.shape, dtype=tl.float32), mask=mask)
            else:
                tl.store(rhs_ptr + out_base + s_offs,
                         tl.zeros(s_offs.shape, dtype=tl.float32), mask=mask)

        # --- Sub-pass A: accumulate A = sum_s(term * pibar_wt * inv_denom) ---
        # Also write speciation scatter contributions.
        A_acc = tl.full([1], value=0.0, dtype=tl.float32)

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
            w_L = tl.full(s_offs.shape, value=1.0, dtype=tl.float32)

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
    )

    return v_k, aw0, aw1, aw2, aw345, aw3, aw4
