"""DTS tangent kernels; see ``docs/latex/kernel_mathematics.tex``."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from gpurec.core.kernels.pi_forward import (
    _load_event_log_probability,
    _tl_float_dtype,
    _validate_offset_tensor,
    _validate_residual_tensors,
)


@triton.jit
def _gene_split_reduction_jvp_kernel(
    Pi, Pibar, dPi, dPibar, split_left_rows, split_right_rows,
    species_child1, species_child2,
    log_pD, log_pS, dlog_pD, dlog_pS, log_split_probs, reduce_idx,
    gene_split_log_likelihood_ptr,
    d_gene_split_log_likelihood_ptr,
    family_idx,
    family_offset,
    Pi_offset,
    Pibar_offset,
    gene_split_offset,
    S: tl.constexpr, BLOCK_S: tl.constexpr, ROW_STRIDE: tl.constexpr,
    BY_SPECIES: tl.constexpr, DTYPE: tl.constexpr,
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
    parent_wave_row = tl.load(reduce_idx + n).to(tl.int64)
    family = tl.load(family_idx + family_offset + parent_wave_row).to(tl.int64)
    left_clade_row = tl.load(split_left_rows + n).to(tl.int64)
    right_clade_row = tl.load(split_right_rows + n).to(tl.int64)
    left_base = left_clade_row * S
    right_base = right_clade_row * S

    left_pi = tl.load(Pi + left_base + s_offs, mask=mask, other=NEG_INF)
    right_pi = tl.load(Pi + right_base + s_offs, mask=mask, other=NEG_INF)
    left_pibar = tl.load(Pibar + left_base + s_offs, mask=mask, other=NEG_INF)
    right_pibar = tl.load(Pibar + right_base + s_offs, mask=mask, other=NEG_INF)
    d_left_pi = tl.load(dPi + left_base + s_offs, mask=mask, other=0.0)
    d_right_pi = tl.load(dPi + right_base + s_offs, mask=mask, other=0.0)
    d_left_pibar = tl.load(dPibar + left_base + s_offs, mask=mask, other=0.0)
    d_right_pibar = tl.load(dPibar + right_base + s_offs, mask=mask, other=0.0)

    duplication_log_probability = _load_event_log_probability(
        log_pD, family, s_offs, mask, S, ROW_STRIDE, BY_SPECIES, BLOCK_S, DTYPE
    )
    speciation_log_probability = _load_event_log_probability(
        log_pS, family, s_offs, mask, S, ROW_STRIDE, BY_SPECIES, BLOCK_S, DTYPE
    )
    d_duplication_log_probability = _load_event_log_probability(
        dlog_pD,
        family,
        s_offs,
        mask,
        S,
        ROW_STRIDE,
        BY_SPECIES,
        BLOCK_S,
        DTYPE,
    )
    d_speciation_log_probability = _load_event_log_probability(
        dlog_pS,
        family,
        s_offs,
        mask,
        S,
        ROW_STRIDE,
        BY_SPECIES,
        BLOCK_S,
        DTYPE,
    )

    c1 = tl.load(species_child1 + s_offs, mask=mask, other=S)
    c2 = tl.load(species_child2 + s_offs, mask=mask, other=S)
    c1_valid = mask & (c1 < S)
    c2_valid = mask & (c2 < S)
    left_pi_child1 = tl.load(Pi + left_base + c1, mask=c1_valid, other=NEG_INF)
    right_pi_child2 = tl.load(Pi + right_base + c2, mask=c2_valid, other=NEG_INF)
    right_pi_child1 = tl.load(Pi + right_base + c1, mask=c1_valid, other=NEG_INF)
    left_pi_child2 = tl.load(Pi + left_base + c2, mask=c2_valid, other=NEG_INF)
    d_left_pi_child1 = tl.load(dPi + left_base + c1, mask=c1_valid, other=0.0)
    d_right_pi_child2 = tl.load(dPi + right_base + c2, mask=c2_valid, other=0.0)
    d_right_pi_child1 = tl.load(dPi + right_base + c1, mask=c1_valid, other=0.0)
    d_left_pi_child2 = tl.load(dPi + left_base + c2, mask=c2_valid, other=0.0)
    split_log_prior = tl.load(log_split_probs + n)

    # The split reduction is a residual in gene_split_offset[parent_wave_row]'s frame. Align
    # each child combination to that frame before forming its softmax weight.
    # Offsets have no tangent: dpi/dpibar differentiate the represented rows.
    gene_split_row_offset = tl.load(gene_split_offset + parent_wave_row)
    left_pi_offset = tl.load(Pi_offset + left_clade_row)
    right_pi_offset = tl.load(Pi_offset + right_clade_row)
    left_pibar_offset = tl.load(Pibar_offset + left_clade_row)
    right_pibar_offset = tl.load(Pibar_offset + right_clade_row)
    child_frame_shift = (left_pi_offset + right_pi_offset - gene_split_row_offset).to(DTYPE)
    left_transfer_frame_shift = (left_pi_offset + right_pibar_offset - gene_split_row_offset).to(DTYPE)
    right_transfer_frame_shift = (right_pi_offset + left_pibar_offset - gene_split_row_offset).to(DTYPE)
    duplication_log_term = (
        split_log_prior
        + duplication_log_probability
        + left_pi
        + right_pi
        + child_frame_shift
    )
    transfer_left_retained_log_term = (
        split_log_prior + left_pi + right_pibar + left_transfer_frame_shift
    )
    transfer_right_retained_log_term = (
        split_log_prior + right_pi + left_pibar + right_transfer_frame_shift
    )
    speciation_lr_log_term = (
        split_log_prior
        + speciation_log_probability
        + left_pi_child1
        + right_pi_child2
        + child_frame_shift
    )
    speciation_rl_log_term = (
        split_log_prior
        + speciation_log_probability
        + right_pi_child1
        + left_pi_child2
        + child_frame_shift
    )
    d_duplication_log_term = d_duplication_log_probability + d_left_pi + d_right_pi
    d_transfer_left_retained_log_term = d_left_pi + d_right_pibar
    d_transfer_right_retained_log_term = d_right_pi + d_left_pibar
    d_speciation_lr_log_term = d_speciation_log_probability + d_left_pi_child1 + d_right_pi_child2
    d_speciation_rl_log_term = d_speciation_log_probability + d_right_pi_child1 + d_left_pi_child2

    gene_split_log_likelihood = tl.load(
        gene_split_log_likelihood_ptr + parent_wave_row * S + s_offs,
        mask=mask,
        other=NEG_INF,
    )
    parent_possible = mask & (gene_split_log_likelihood != NEG_INF)
    duplication_probability = tl.exp2(
        duplication_log_term - gene_split_log_likelihood
    )
    transfer_left_retained_probability = tl.exp2(
        transfer_left_retained_log_term - gene_split_log_likelihood
    )
    transfer_right_retained_probability = tl.exp2(
        transfer_right_retained_log_term - gene_split_log_likelihood
    )
    speciation_lr_probability = tl.exp2(
        speciation_lr_log_term - gene_split_log_likelihood
    )
    speciation_rl_probability = tl.exp2(
        speciation_rl_log_term - gene_split_log_likelihood
    )
    split_tangent_contribution = (
        duplication_probability * d_duplication_log_term
        + transfer_left_retained_probability * d_transfer_left_retained_log_term
        + transfer_right_retained_probability * d_transfer_right_retained_log_term
        + speciation_lr_probability * d_speciation_lr_log_term
        + speciation_rl_probability * d_speciation_rl_log_term
    )
    split_tangent_contribution = tl.where(
        parent_possible, split_tangent_contribution, zero
    )
    tl.atomic_add(
        d_gene_split_log_likelihood_ptr + parent_wave_row * S + s_offs,
        split_tangent_contribution,
        sem="relaxed",
        mask=mask,
    )


def compute_dts_tangent(
    Pi, Pibar, dPi, dPibar, split_left_rows, split_right_rows,
    species_child1, species_child2, W, reduce_idx,
    log_pD_vec, log_pS_vec, dlog_pD_vec, dlog_pS_vec,
    gene_split_log_likelihood, family_idx,
    *, log_split_probs=None, family_offset=0,
    pi_offset, pibar_offset, gene_split_offset,
):
    """Return the DTS JVP; see ``docs/latex/kernel_mathematics.tex``."""
    N = int(split_left_rows.shape[0])
    S = int(Pi.shape[1])
    _validate_residual_tensors(
        Pi,
        Pibar=Pibar,
        dPi=dPi,
        dPibar=dPibar,
        log_pD_vec=log_pD_vec,
        log_pS_vec=log_pS_vec,
        dlog_pD_vec=dlog_pD_vec,
        dlog_pS_vec=dlog_pS_vec,
        gene_split_log_likelihood=gene_split_log_likelihood,
        log_split_probs=log_split_probs,
    )
    d_gene_split_log_likelihood = torch.zeros(
        (W, S), device=Pi.device, dtype=Pi.dtype
    )
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
    gene_split_offset = _validate_offset_tensor(
        "gene_split_offset",
        gene_split_offset,
        rows=W,
        device=Pi.device,
        dtype=accumulator_dtype,
    )
    if N == 0:
        return d_gene_split_log_likelihood
    if log_split_probs is None:
        log_split_probs = torch.zeros((N,), device=Pi.device, dtype=Pi.dtype)
    else:
        log_split_probs = log_split_probs.reshape(N).contiguous()
    by_species = log_pD_vec.ndim == 2 and int(log_pD_vec.shape[1]) != 1
    row_stride = 0 if int(log_pD_vec.shape[0]) == 1 else int(log_pD_vec.stride(0))
    block_s = min(512, triton.next_power_of_2(S))
    _gene_split_reduction_jvp_kernel[(N, triton.cdiv(S, block_s))](
        Pi, Pibar, dPi, dPibar, split_left_rows, split_right_rows,
        species_child1, species_child2,
        log_pD_vec, log_pS_vec, dlog_pD_vec, dlog_pS_vec, log_split_probs, reduce_idx,
        gene_split_log_likelihood,
        d_gene_split_log_likelihood,
        family_idx,
        int(family_offset),
        pi_offset,
        pibar_offset,
        gene_split_offset,
        S, BLOCK_S=block_s, ROW_STRIDE=row_stride, BY_SPECIES=bool(by_species),
        DTYPE=_tl_float_dtype(Pi.dtype),
    )
    return d_gene_split_log_likelihood
