"""Fused DTS computation: 5 terms + logsumexp in one Triton kernel."""

import torch
import triton
import triton.language as tl


@triton.jit
def _dts_fused_kernel(
    # Inputs: Pi and Pibar for left/right children [N, S]
    Pi_l_ptr, Pi_r_ptr, Pibar_l_ptr, Pibar_r_ptr,
    # Padded Pi for species children: [N, S+1] (last col = -inf)
    Pi_l_pad_ptr, Pi_r_pad_ptr,
    # Species child indices: [S] each
    sp_child1_ptr, sp_child2_ptr,
    # Parameters: scalar values
    log_pD_val,
    log_pS_val,
    # Split probs: [N]
    log_split_probs_ptr,
    # Output: [N, S]
    out_ptr,
    # Dimensions
    N: tl.constexpr,
    S: tl.constexpr,
    stride_pad: tl.constexpr,  # S + 1 for padded tensors
    BLOCK_S: tl.constexpr,
):
    """Compute DTS_term[i, s] = log_split_probs[i] + logsumexp2(5 DTS terms)[s]."""
    n = tl.program_id(0)
    s_block = tl.program_id(1)
    s_offs = s_block * BLOCK_S + tl.arange(0, BLOCK_S)
    mask = s_offs < S

    base = n * S
    base_pad = n * stride_pad

    # Load Pi/Pibar for this split
    pi_l = tl.load(Pi_l_ptr + base + s_offs, mask=mask, other=-1e30)
    pi_r = tl.load(Pi_r_ptr + base + s_offs, mask=mask, other=-1e30)
    pibar_l = tl.load(Pibar_l_ptr + base + s_offs, mask=mask, other=-1e30)
    pibar_r = tl.load(Pibar_r_ptr + base + s_offs, mask=mask, other=-1e30)

    # Compute 5 DTS terms
    t0 = log_pD_val + pi_l + pi_r                        # D
    t1 = pi_l + pibar_r                                  # T (l->r)
    t2 = pi_r + pibar_l                                  # T (r->l)

    # Species children for S terms: index into padded Pi
    c1 = tl.load(sp_child1_ptr + s_offs, mask=mask, other=0)
    c2 = tl.load(sp_child2_ptr + s_offs, mask=mask, other=0)
    pi_l_c1 = tl.load(Pi_l_pad_ptr + base_pad + c1, mask=mask, other=-1e30)
    pi_r_c2 = tl.load(Pi_r_pad_ptr + base_pad + c2, mask=mask, other=-1e30)
    pi_r_c1 = tl.load(Pi_r_pad_ptr + base_pad + c1, mask=mask, other=-1e30)
    pi_l_c2 = tl.load(Pi_l_pad_ptr + base_pad + c2, mask=mask, other=-1e30)

    t3 = log_pS_val + pi_l_c1 + pi_r_c2                  # S
    t4 = log_pS_val + pi_r_c1 + pi_l_c2                  # S (swapped)

    # Fused logsumexp2 over 5 terms
    m = tl.maximum(t0, t1)
    m = tl.maximum(m, t2)
    m = tl.maximum(m, t3)
    m = tl.maximum(m, t4)
    m_safe = tl.where(m > -1e29, m, tl.zeros_like(m))

    s = (tl.exp2(t0 - m_safe) + tl.exp2(t1 - m_safe) + tl.exp2(t2 - m_safe)
         + tl.exp2(t3 - m_safe) + tl.exp2(t4 - m_safe))

    # Add log_split_probs
    lsp = tl.load(log_split_probs_ptr + n)
    result = tl.log2(s) + m + lsp

    tl.store(out_ptr + base + s_offs, result, mask=mask)


def dts_fused(Pi_l, Pi_r, Pibar_l, Pibar_r,
              Pi_l_pad, Pi_r_pad,
              sp_child1, sp_child2,
              log_pD, log_pS, log_split_probs,
              out=None):
    """Fused DTS: 5 terms + logsumexp + split_probs in one kernel.

    Args:
        Pi_l, Pi_r: [N, S] contiguous
        Pibar_l, Pibar_r: [N, S] contiguous
        Pi_l_pad, Pi_r_pad: [N, S+1] contiguous (last col = -inf)
        sp_child1, sp_child2: [S] long
        log_pD, log_pS: scalar or [S]
        log_split_probs: [N, 1] or [N]
        out: optional [N, S] output buffer

    Returns:
        DTS_term: [N, S] = log_split_probs + logsumexp2(5 DTS terms)
    """
    N, S = Pi_l.shape
    if out is None:
        out = torch.empty_like(Pi_l)

    # Flatten log_split_probs to [N]
    lsp = log_split_probs.reshape(N).contiguous()

    # Extract scalar values for kernel
    pD_val = float(log_pD.item()) if log_pD.dim() == 0 else float(log_pD.mean().item())
    pS_val = float(log_pS.item()) if log_pS.dim() == 0 else float(log_pS.mean().item())

    BLOCK_S = 32
    grid = (N, (S + BLOCK_S - 1) // BLOCK_S)

    _dts_fused_kernel[grid](
        Pi_l, Pi_r, Pibar_l, Pibar_r,
        Pi_l_pad, Pi_r_pad,
        sp_child1, sp_child2,
        pD_val, pS_val,
        lsp,
        out,
        N, S,
        stride_pad=S + 1,
        BLOCK_S=BLOCK_S,
    )
    return out
