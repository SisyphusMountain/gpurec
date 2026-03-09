"""Fused Triton kernel for wave-step: computes Pi_new from Pibar_W, Pi_W, and constant terms."""

import torch
import triton
import triton.language as tl


@triton.jit
def _wave_step_kernel(
    # Pi_W: [W, S], Pibar_W: [W, S]
    Pi_W_ptr, Pibar_W_ptr,
    # Precomputed constants: [S] each
    DL_const_ptr, Ebar_ptr, E_ptr, SL1_const_ptr, SL2_const_ptr,
    # Species child indices: [S] each (index into columns, S = padding/leaf)
    sp_child1_ptr, sp_child2_ptr,
    # Leaf term: [W, S]
    leaf_term_ptr,
    # DTS_reduced: [W, S] or None (if no splits)
    DTS_reduced_ptr,
    has_splits: tl.constexpr,
    # Output: [W, S]
    Pi_new_ptr,
    # Dimensions
    W: tl.constexpr, S: tl.constexpr,
    # Strides
    stride_ws: tl.constexpr,  # stride for [W, S] tensors (== S for contiguous)
):
    """Fused kernel: given Pi_W, Pibar_W, compute Pi_new = logsumexp2(all_terms, dim=0)."""
    w = tl.program_id(0)
    s_block = tl.program_id(1)
    BLOCK_S: tl.constexpr = 32

    s_offs = s_block * BLOCK_S + tl.arange(0, BLOCK_S)
    mask = s_offs < S

    # Load Pi_W[w, s] and Pibar_W[w, s]
    base = w * stride_ws
    pi_w = tl.load(Pi_W_ptr + base + s_offs, mask=mask, other=-1e30)
    pibar_w = tl.load(Pibar_W_ptr + base + s_offs, mask=mask, other=-1e30)

    # Load constants
    dl_const = tl.load(DL_const_ptr + s_offs, mask=mask, other=-1e30)
    ebar = tl.load(Ebar_ptr + s_offs, mask=mask, other=-1e30)
    e_val = tl.load(E_ptr + s_offs, mask=mask, other=-1e30)
    sl1_const = tl.load(SL1_const_ptr + s_offs, mask=mask, other=-1e30)
    sl2_const = tl.load(SL2_const_ptr + s_offs, mask=mask, other=-1e30)

    # Gather species children: Pi[w, child1[s]], Pi[w, child2[s]]
    c1 = tl.load(sp_child1_ptr + s_offs, mask=mask, other=0)
    c2 = tl.load(sp_child2_ptr + s_offs, mask=mask, other=0)
    # Clamp child indices to valid range (S is the padding column = -inf)
    c1_valid = c1 < S
    c2_valid = c2 < S
    pi_s1 = tl.load(Pi_W_ptr + base + c1, mask=mask & c1_valid, other=-1e30)
    pi_s2 = tl.load(Pi_W_ptr + base + c2, mask=mask & c2_valid, other=-1e30)

    # 6 DTS_L terms
    t0 = dl_const + pi_w       # DL
    t1 = pi_w + ebar           # TL
    t2 = pibar_w + e_val       # TL (transfer)
    t3 = sl1_const + pi_s1     # SL child1
    t4 = sl2_const + pi_s2     # SL child2
    t5 = tl.load(leaf_term_ptr + base + s_offs, mask=mask, other=-1e30)  # leaf

    # Max of 6 terms
    m = tl.maximum(t0, t1)
    m = tl.maximum(m, t2)
    m = tl.maximum(m, t3)
    m = tl.maximum(m, t4)
    m = tl.maximum(m, t5)

    if has_splits:
        dts_r = tl.load(DTS_reduced_ptr + w * stride_ws + s_offs, mask=mask, other=-1e30)
        m = tl.maximum(m, dts_r)

    # Stabilized logsumexp2
    m_safe = tl.where(m > -1e29, m, tl.zeros_like(m))
    s = tl.exp2(t0 - m_safe) + tl.exp2(t1 - m_safe) + tl.exp2(t2 - m_safe)
    s += tl.exp2(t3 - m_safe) + tl.exp2(t4 - m_safe) + tl.exp2(t5 - m_safe)
    if has_splits:
        s += tl.exp2(dts_r - m_safe)

    result = tl.log2(s) + m
    tl.store(Pi_new_ptr + base + s_offs, result, mask=mask)


def wave_step_fused(Pi_W, Pibar_W, DL_const, Ebar, E, SL1_const, SL2_const,
                    sp_child1, sp_child2, leaf_term_wt, DTS_reduced=None):
    """Fused wave step: computes Pi_new from all terms in a single kernel.

    Args:
        Pi_W: [W, S] log2-space
        Pibar_W: [W, S] log2-space
        DL_const, Ebar, E, SL1_const, SL2_const: [S] each
        sp_child1, sp_child2: [S] long, species child indices (S = padding)
        leaf_term_wt: [W, S] log2-space
        DTS_reduced: [W, S] or None if no splits

    Returns:
        Pi_new: [W, S] log2-space
    """
    W, S = Pi_W.shape
    Pi_new = torch.empty_like(Pi_W)
    has_splits = DTS_reduced is not None

    BLOCK_S = 32
    grid = (W, (S + BLOCK_S - 1) // BLOCK_S)

    _wave_step_kernel[grid](
        Pi_W, Pibar_W,
        DL_const, Ebar, E, SL1_const, SL2_const,
        sp_child1, sp_child2,
        leaf_term_wt,
        DTS_reduced if has_splits else Pi_W,  # dummy ptr when no splits
        has_splits,
        Pi_new,
        W, S,
        stride_ws=S,
    )
    return Pi_new
