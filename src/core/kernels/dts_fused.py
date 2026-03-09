"""Fused DTS computation: gather + 5 terms + logsumexp in one Triton kernel."""

import torch
import triton
import triton.language as tl


@triton.jit
def _dts_fused_kernel(
    # Full Pi and Pibar tensors: [C, S]
    Pi_ptr, Pibar_ptr,
    # Split left/right child indices: [N] each (clade indices into Pi)
    lefts_ptr, rights_ptr,
    # Species child indices: [S] each (S = sentinel for "no child")
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
    BLOCK_S: tl.constexpr,
):
    """Compute DTS_term[i, s] = log_split_probs[i] + logsumexp2(5 DTS terms)[s].

    Fuses the gather of Pi[left], Pi[right], Pibar[left], Pibar[right]
    directly from the full [C, S] tensors, avoiding materialization of
    [N, S] intermediates and [N, S+1] padded tensors.
    """
    n = tl.program_id(0)
    s_block = tl.program_id(1)
    s_offs = s_block * BLOCK_S + tl.arange(0, BLOCK_S)
    mask = s_offs < S

    # Load left/right clade indices for this split
    left_idx = tl.load(lefts_ptr + n)
    right_idx = tl.load(rights_ptr + n)

    base_l = left_idx * S
    base_r = right_idx * S

    # Load Pi/Pibar for left and right children directly from [C, S] tensors
    pi_l = tl.load(Pi_ptr + base_l + s_offs, mask=mask, other=-1e30)
    pi_r = tl.load(Pi_ptr + base_r + s_offs, mask=mask, other=-1e30)
    pibar_l = tl.load(Pibar_ptr + base_l + s_offs, mask=mask, other=-1e30)
    pibar_r = tl.load(Pibar_ptr + base_r + s_offs, mask=mask, other=-1e30)

    # Compute first 3 DTS terms
    t0 = log_pD_val + pi_l + pi_r                        # D
    t1 = pi_l + pibar_r                                  # T (l->r)
    t2 = pi_r + pibar_l                                  # T (r->l)

    # Species children for S terms: load from Pi directly with sentinel check
    c1 = tl.load(sp_child1_ptr + s_offs, mask=mask, other=S)
    c2 = tl.load(sp_child2_ptr + s_offs, mask=mask, other=S)
    c1_valid = c1 < S
    c2_valid = c2 < S

    # Load Pi[left, c1], Pi[right, c2], etc. with sentinel masking
    # When c1==S (leaf species), result is -inf (no speciation possible)
    pi_l_c1 = tl.load(Pi_ptr + base_l + c1, mask=mask & c1_valid, other=-1e30)
    pi_r_c2 = tl.load(Pi_ptr + base_r + c2, mask=mask & c2_valid, other=-1e30)
    pi_r_c1 = tl.load(Pi_ptr + base_r + c1, mask=mask & c1_valid, other=-1e30)
    pi_l_c2 = tl.load(Pi_ptr + base_l + c2, mask=mask & c2_valid, other=-1e30)

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

    out_base = n * S
    tl.store(out_ptr + out_base + s_offs, result, mask=mask)


def dts_fused(Pi, Pibar, lefts, rights,
              sp_child1, sp_child2,
              log_pD, log_pS, log_split_probs,
              out=None):
    """Fused DTS: gather + 5 terms + logsumexp + split_probs in one kernel.

    Args:
        Pi: [C, S] contiguous — full Pi tensor
        Pibar: [C, S] contiguous — full Pibar tensor
        lefts: [N] long — left child clade indices per split
        rights: [N] long — right child clade indices per split
        sp_child1, sp_child2: [S] long — species tree child indices (S=sentinel)
        log_pD, log_pS: scalar
        log_split_probs: [N, 1] or [N]
        out: optional [N, S] output buffer

    Returns:
        DTS_term: [N, S] = log_split_probs + logsumexp2(5 DTS terms)
    """
    N = lefts.shape[0]
    S = Pi.shape[1]
    if out is None:
        out = torch.empty((N, S), device=Pi.device, dtype=Pi.dtype)

    # Flatten log_split_probs to [N]
    lsp = log_split_probs.reshape(N).contiguous()

    # Extract scalar values for kernel
    pD_val = float(log_pD.item()) if log_pD.dim() == 0 else float(log_pD.mean().item())
    pS_val = float(log_pS.item()) if log_pS.dim() == 0 else float(log_pS.mean().item())

    BLOCK_S = 128
    grid = (N, (S + BLOCK_S - 1) // BLOCK_S)

    _dts_fused_kernel[grid](
        Pi.contiguous(), Pibar.contiguous(),
        lefts, rights,
        sp_child1, sp_child2,
        pD_val, pS_val,
        lsp,
        out,
        N, S,
        BLOCK_S=BLOCK_S,
    )
    return out
