"""Wave-step second-order kernels; see ``docs/latex/kernel_mathematics.tex``."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from gpurec.core.kernels.pi_forward import (
    _prepare_wave_launch,
    _tl_float_dtype,
    _validate_offset_tensor,
    _validate_residual_tensors,
)


@triton.jit
def _reconciliation_vjp_directional_derivative_kernel(
    Pi_ptr, dPi_ptr, Pibar_ptr, dPibar_ptr,
    Pi_offset_ptr, Pibar_offset_ptr,
    v_ptr,
    ws, S: tl.constexpr, stride: tl.constexpr,
    pibar_row_max_ptr,
    duplication_loss_const_ptr, d_duplication_loss_const_ptr,
    Ebar_ptr, dEbar_ptr, E_ptr, dE_ptr,
    speciation_child1_const_ptr, d_speciation_child1_const_ptr,
    speciation_child2_const_ptr, d_speciation_child2_const_ptr,
    receiver_log_probs_ptr, dreceiver_log_probs_ptr,
    species_child1_ptr, species_child2_ptr, species_parent_ptr,
    leaf_species_ptr, leaf_logp_ptr, d_leaf_logp_ptr,
    family_idx_ptr,
    gene_split_log_likelihood_ptr,
    d_gene_split_log_likelihood_ptr,
    gene_split_offset_ptr,
    has_splits: tl.constexpr,
    d_self_loop_vjp_ptr, d_rhs_ptr, d_grad_receiver_log_probs_ptr,
    d_duplication_loss_event_vjp_ptr,
    d_transfer_loss_event_vjp_ptr,
    d_transfer_event_vjp_ptr,
    d_speciation_leaf_event_vjp_ptr,
    d_speciation_child1_event_vjp_ptr,
    d_speciation_child2_event_vjp_ptr,
    subtree_donor_adjoint_ptr, d_subtree_donor_adjoint_ptr,
    CONST_ROW_STRIDE: tl.constexpr,
    BLOCK_S: tl.constexpr,
    MAX_ANCESTOR_DEPTH: tl.constexpr,
    USE_LEAF_INDEX: tl.constexpr,
    USE_RECEIVER_WEIGHTS: tl.constexpr,
    FOLD_RHS: tl.constexpr,
    DTYPE: tl.constexpr,
):
    LN2 = 0.6931471805599453
    NEG = -float("inf")
    w = tl.program_id(0)
    pi_base = (ws + w) * stride
    out_base = w * stride
    family = tl.load(family_idx_ptr + ws + w)
    const_base = family * CONST_ROW_STRIDE

    s_offs = tl.arange(0, BLOCK_S)
    mask = s_offs < S
    zero = tl.zeros([BLOCK_S], dtype=DTYPE)

    reconciliation_log_likelihood = tl.load(Pi_ptr + pi_base + s_offs, mask=mask, other=NEG)
    d_reconciliation_log_likelihood = tl.load(dPi_ptr + pi_base + s_offs, mask=mask, other=0.0)
    transfer_complement_log_likelihood = tl.load(Pibar_ptr + pi_base + s_offs, mask=mask, other=NEG)
    d_transfer_complement_log_likelihood = tl.load(dPibar_ptr + pi_base + s_offs, mask=mask, other=0.0)
    v = tl.load(v_ptr + out_base + s_offs, mask=mask, other=0.0)
    receiver_mass_log_scale = tl.load(pibar_row_max_ptr + ws + w)
    receiver_mass_log_scale_safe = tl.where(receiver_mass_log_scale != NEG, receiver_mass_log_scale, tl.zeros((), dtype=DTYPE))

    pi_offset = tl.load(Pi_offset_ptr + ws + w)
    pibar_offset = tl.load(Pibar_offset_ptr + ws + w)
    transfer_complement_frame_shift = (pibar_offset - pi_offset).to(DTYPE)
    leaf_frame_shift = (-pi_offset).to(DTYPE)
    if has_splits:
        gene_split_offset = tl.load(gene_split_offset_ptr + w)
        gene_split_frame_shift = (gene_split_offset - pi_offset).to(DTYPE)
    else:
        gene_split_frame_shift = tl.zeros((), dtype=DTYPE)

    if USE_RECEIVER_WEIGHTS:
        receiver_log_probability = tl.load(
            receiver_log_probs_ptr + s_offs, mask=mask, other=NEG
        )
        d_receiver_log_probability = tl.load(
            dreceiver_log_probs_ptr + s_offs, mask=mask, other=0.0
        )
        receiver_mass = tl.where(mask, tl.exp2(receiver_log_probability + reconciliation_log_likelihood - receiver_mass_log_scale_safe), zero)
        d_receiver_mass = LN2 * receiver_mass * (d_reconciliation_log_likelihood + d_receiver_log_probability)
    else:
        receiver_mass = tl.where(mask, tl.exp2(reconciliation_log_likelihood - receiver_mass_log_scale_safe), zero)
        d_receiver_mass = LN2 * receiver_mass * d_reconciliation_log_likelihood
    total_receiver_mass = tl.sum(receiver_mass, axis=0)
    d_total_receiver_mass = tl.sum(d_receiver_mass, axis=0)

    # The stabilizing row maximum is frozen: the represented transfer
    # complement is invariant to this gauge.
    ancestor_species = s_offs
    excluded_ancestor_mass = zero
    d_excluded_ancestor_mass = zero
    for _ in range(0, MAX_ANCESTOR_DEPTH):
        ancestor_valid = mask & (ancestor_species >= 0) & (ancestor_species < S)
        ancestor_reconciliation_log_likelihood = tl.load(Pi_ptr + pi_base + ancestor_species, mask=ancestor_valid, other=NEG)
        d_ancestor_reconciliation_log_likelihood = tl.load(dPi_ptr + pi_base + ancestor_species, mask=ancestor_valid, other=0.0)
        if USE_RECEIVER_WEIGHTS:
            ancestor_receiver_log_probability = tl.load(
                receiver_log_probs_ptr + ancestor_species, mask=ancestor_valid, other=NEG
            )
            d_ancestor_receiver_log_probability = tl.load(
                dreceiver_log_probs_ptr + ancestor_species, mask=ancestor_valid, other=0.0
            )
            ancestor_receiver_mass = tl.where(
                ancestor_valid, tl.exp2(ancestor_receiver_log_probability + ancestor_reconciliation_log_likelihood - receiver_mass_log_scale_safe), zero
            )
            d_excluded_ancestor_mass += (
                LN2
                * ancestor_receiver_mass
                * (d_ancestor_reconciliation_log_likelihood + d_ancestor_receiver_log_probability)
            )
        else:
            ancestor_receiver_mass = tl.where(ancestor_valid, tl.exp2(ancestor_reconciliation_log_likelihood - receiver_mass_log_scale_safe), zero)
            d_excluded_ancestor_mass += LN2 * ancestor_receiver_mass * d_ancestor_reconciliation_log_likelihood
        excluded_ancestor_mass += ancestor_receiver_mass
        ancestor_species = tl.load(species_parent_ptr + ancestor_species, mask=ancestor_valid, other=-1).to(tl.int32)

    valid_receiver_mass = total_receiver_mass - excluded_ancestor_mass
    has_valid_receiver_mass = mask & (valid_receiver_mass > 0.0)
    safe_valid_receiver_mass = tl.where(
        has_valid_receiver_mass,
        valid_receiver_mass,
        tl.full([BLOCK_S], 1.0, DTYPE),
    )
    inverse_valid_receiver_mass = tl.where(
        has_valid_receiver_mass, 1.0 / safe_valid_receiver_mass, zero
    )
    d_valid_receiver_mass = d_total_receiver_mass - d_excluded_ancestor_mass

    # Event terms and probabilities at the saved primal fixed point.
    const_offsets = const_base + s_offs
    duplication_loss_const = tl.load(
        duplication_loss_const_ptr + const_offsets, mask=mask, other=NEG
    )
    d_duplication_loss_const = tl.load(
        d_duplication_loss_const_ptr + const_offsets, mask=mask, other=0.0
    )
    extinction_complement_log_probability = tl.load(Ebar_ptr + const_offsets, mask=mask, other=NEG)
    d_extinction_complement_log_probability = tl.load(dEbar_ptr + const_offsets, mask=mask, other=0.0)
    extinction_log_probability = tl.load(E_ptr + const_offsets, mask=mask, other=NEG)
    d_extinction_log_probability = tl.load(dE_ptr + const_offsets, mask=mask, other=0.0)
    speciation_child1_const = tl.load(
        speciation_child1_const_ptr + const_offsets, mask=mask, other=NEG
    )
    d_speciation_child1_const = tl.load(
        d_speciation_child1_const_ptr + const_offsets, mask=mask, other=0.0
    )
    speciation_child2_const = tl.load(
        speciation_child2_const_ptr + const_offsets, mask=mask, other=NEG
    )
    d_speciation_child2_const = tl.load(
        d_speciation_child2_const_ptr + const_offsets, mask=mask, other=0.0
    )

    c1 = tl.load(species_child1_ptr + s_offs, mask=mask, other=S)
    c2 = tl.load(species_child2_ptr + s_offs, mask=mask, other=S)
    child1_valid = mask & (c1 < S)
    child2_valid = mask & (c2 < S)
    reconciliation_child1_log_likelihood = tl.where(
        child1_valid,
        tl.gather(
            reconciliation_log_likelihood,
            tl.where(child1_valid, c1, 0),
            axis=0,
        ),
        NEG,
    )
    reconciliation_child2_log_likelihood = tl.where(
        child2_valid,
        tl.gather(
            reconciliation_log_likelihood,
            tl.where(child2_valid, c2, 0),
            axis=0,
        ),
        NEG,
    )
    d_reconciliation_child1_log_likelihood = tl.where(
        child1_valid,
        tl.gather(
            d_reconciliation_log_likelihood,
            tl.where(child1_valid, c1, 0),
            axis=0,
        ),
        zero,
    )
    d_reconciliation_child2_log_likelihood = tl.where(
        child2_valid,
        tl.gather(
            d_reconciliation_log_likelihood,
            tl.where(child2_valid, c2, 0),
            axis=0,
        ),
        zero,
    )

    duplication_loss_log_term = duplication_loss_const + reconciliation_log_likelihood
    transfer_loss_log_term = reconciliation_log_likelihood + extinction_complement_log_probability
    transfer_log_term = transfer_complement_log_likelihood + extinction_log_probability + transfer_complement_frame_shift
    speciation_child1_log_term = speciation_child1_const + reconciliation_child1_log_likelihood
    speciation_child2_log_term = speciation_child2_const + reconciliation_child2_log_likelihood
    d_duplication_loss_log_term = d_duplication_loss_const + d_reconciliation_log_likelihood
    d_transfer_loss_log_term = d_reconciliation_log_likelihood + d_extinction_complement_log_probability
    d_transfer_log_term = d_transfer_complement_log_likelihood + d_extinction_log_probability
    d_speciation_child1_log_term = d_speciation_child1_const + d_reconciliation_child1_log_likelihood
    d_speciation_child2_log_term = d_speciation_child2_const + d_reconciliation_child2_log_likelihood
    if USE_LEAF_INDEX:
        leaf_species = tl.load(leaf_species_ptr + ws + w)
        leaf_hit = mask & (leaf_species == s_offs)
        leaf_logp = tl.load(leaf_logp_ptr + family * S + s_offs, mask=mask, other=NEG)
        d_leaf_logp = tl.load(
            d_leaf_logp_ptr + family * S + s_offs, mask=mask, other=0.0
        )
        leaf_observation_log_term = tl.where(leaf_hit, leaf_logp + leaf_frame_shift, NEG)
        d_leaf_observation_log_term = tl.where(leaf_hit, d_leaf_logp, zero)
    else:
        leaf_observation_log_term = tl.full([BLOCK_S], NEG, dtype=DTYPE)
        d_leaf_observation_log_term = zero

    logsumexp_max = tl.maximum(
        tl.maximum(tl.maximum(duplication_loss_log_term, transfer_loss_log_term),
                   tl.maximum(transfer_log_term, speciation_child1_log_term)),
        tl.maximum(speciation_child2_log_term, leaf_observation_log_term),
    )
    logsumexp_max_safe = tl.where(logsumexp_max != NEG, logsumexp_max, zero)
    duplication_loss_mass = tl.exp2(duplication_loss_log_term - logsumexp_max_safe)
    transfer_loss_mass = tl.exp2(transfer_loss_log_term - logsumexp_max_safe)
    transfer_mass = tl.exp2(transfer_log_term - logsumexp_max_safe)
    speciation_child1_mass = tl.exp2(speciation_child1_log_term - logsumexp_max_safe)
    speciation_child2_mass = tl.exp2(speciation_child2_log_term - logsumexp_max_safe)
    leaf_observation_mass = tl.exp2(leaf_observation_log_term - logsumexp_max_safe)
    local_event_scaled_mass = (
        duplication_loss_mass + transfer_loss_mass + transfer_mass
        + speciation_child1_mass + speciation_child2_mass + leaf_observation_mass
    )
    inverse_local_event_scaled_mass = tl.where(
        local_event_scaled_mass > 0.0, 1.0 / local_event_scaled_mass, zero
    )
    duplication_loss_probability = duplication_loss_mass * inverse_local_event_scaled_mass
    transfer_loss_probability = transfer_loss_mass * inverse_local_event_scaled_mass
    transfer_probability = transfer_mass * inverse_local_event_scaled_mass
    speciation_child1_probability = speciation_child1_mass * inverse_local_event_scaled_mass
    speciation_child2_probability = speciation_child2_mass * inverse_local_event_scaled_mass
    leaf_observation_probability = leaf_observation_mass * inverse_local_event_scaled_mass

    d_local_event_log_likelihood = (
        duplication_loss_probability * d_duplication_loss_log_term
        + transfer_loss_probability * d_transfer_loss_log_term
        + transfer_probability * d_transfer_log_term
        + speciation_child1_probability * d_speciation_child1_log_term
        + speciation_child2_probability * d_speciation_child2_log_term
        + leaf_observation_probability * d_leaf_observation_log_term
    )
    d_duplication_loss_probability = LN2 * duplication_loss_probability * (
        d_duplication_loss_log_term - d_local_event_log_likelihood
    )
    d_transfer_loss_probability = LN2 * transfer_loss_probability * (
        d_transfer_loss_log_term - d_local_event_log_likelihood
    )
    d_transfer_probability = LN2 * transfer_probability * (
        d_transfer_log_term - d_local_event_log_likelihood
    )
    d_speciation_child1_probability = LN2 * speciation_child1_probability * (
        d_speciation_child1_log_term - d_local_event_log_likelihood
    )
    d_speciation_child2_probability = LN2 * speciation_child2_probability * (
        d_speciation_child2_log_term - d_local_event_log_likelihood
    )
    d_leaf_observation_probability = LN2 * leaf_observation_probability * (
        d_leaf_observation_log_term - d_local_event_log_likelihood
    )

    if has_splits:
        gene_split_log_likelihood = tl.load(
            gene_split_log_likelihood_ptr + out_base + s_offs,
            mask=mask,
            other=NEG,
        ) + gene_split_frame_shift
        d_gene_split_log_likelihood = tl.load(
            d_gene_split_log_likelihood_ptr + out_base + s_offs,
            mask=mask,
            other=0.0,
        )
        local_events_are_possible = mask & (local_event_scaled_mass > 0.0)
        safe_local_event_mass_sum = tl.where(
            local_events_are_possible,
            local_event_scaled_mass,
            tl.full([BLOCK_S], 1.0, DTYPE),
        )
        within_wave_log_likelihood = tl.where(
            local_events_are_possible,
            tl.log2(safe_local_event_mass_sum) + logsumexp_max,
            tl.full([BLOCK_S], NEG, DTYPE),
        )
        recurrence_log_term_max = tl.maximum(within_wave_log_likelihood, gene_split_log_likelihood)
        recurrence_log_term_max_safe = tl.where(
            recurrence_log_term_max != NEG, recurrence_log_term_max, zero
        )
        updated_reconciliation_log_likelihood = (
            tl.log2(
                tl.exp2(
                    within_wave_log_likelihood - recurrence_log_term_max_safe
                )
                + tl.exp2(
                    gene_split_log_likelihood - recurrence_log_term_max_safe
                )
            )
            + recurrence_log_term_max
        )
        within_wave_probability = tl.where(
            within_wave_log_likelihood != NEG,
            tl.exp2(within_wave_log_likelihood - updated_reconciliation_log_likelihood),
            zero,
        )
        d_within_wave_probability = tl.where(
            mask & (gene_split_log_likelihood != NEG) & (within_wave_log_likelihood != NEG),
            LN2 * within_wave_probability * (1.0 - within_wave_probability)
            * (d_local_event_log_likelihood - d_gene_split_log_likelihood),
            zero,
        )
    else:
        within_wave_probability = tl.where(mask, tl.full([BLOCK_S], 1.0, DTYPE), zero)
        d_within_wave_probability = zero

    d_duplication_loss_event_vjp = v * (
        d_within_wave_probability * duplication_loss_probability
        + within_wave_probability * d_duplication_loss_probability
    )
    d_transfer_loss_event_vjp = v * (
        d_within_wave_probability * transfer_loss_probability
        + within_wave_probability * d_transfer_loss_probability
    )
    d_transfer_event_vjp = v * (
        d_within_wave_probability * transfer_probability
        + within_wave_probability * d_transfer_probability
    )
    d_speciation_child1_event_vjp = v * (
        d_within_wave_probability * speciation_child1_probability
        + within_wave_probability * d_speciation_child1_probability
    )
    d_speciation_child2_event_vjp = v * (
        d_within_wave_probability * speciation_child2_probability
        + within_wave_probability * d_speciation_child2_probability
    )
    d_leaf_observation_event_vjp = v * (
        d_within_wave_probability * leaf_observation_probability
        + within_wave_probability * d_leaf_observation_probability
    )
    tl.store(d_duplication_loss_event_vjp_ptr + out_base + s_offs, d_duplication_loss_event_vjp, mask=mask)
    tl.store(d_transfer_loss_event_vjp_ptr + out_base + s_offs, d_transfer_loss_event_vjp, mask=mask)
    tl.store(d_transfer_event_vjp_ptr + out_base + s_offs, d_transfer_event_vjp, mask=mask)
    tl.store(
        d_speciation_leaf_event_vjp_ptr + out_base + s_offs,
        d_speciation_child1_event_vjp + d_speciation_child2_event_vjp + d_leaf_observation_event_vjp,
        mask=mask,
    )
    tl.store(d_speciation_child1_event_vjp_ptr + out_base + s_offs, d_speciation_child1_event_vjp, mask=mask)
    tl.store(d_speciation_child2_event_vjp_ptr + out_base + s_offs, d_speciation_child2_event_vjp, mask=mask)

    donor_adjoint_coefficient = (
        within_wave_probability
        * transfer_probability
        * inverse_valid_receiver_mass
    )
    d_donor_adjoint_coefficient = (
        d_within_wave_probability * transfer_probability
        + within_wave_probability * d_transfer_probability
    ) * inverse_valid_receiver_mass - (
        donor_adjoint_coefficient
        * inverse_valid_receiver_mass
        * d_valid_receiver_mass
    )
    donor_adjoint = v * donor_adjoint_coefficient
    d_donor_adjoint = v * d_donor_adjoint_coefficient
    total_donor_adjoint = tl.sum(donor_adjoint, axis=0)
    d_total_donor_adjoint = tl.sum(d_donor_adjoint, axis=0)

    tl.store(subtree_donor_adjoint_ptr + out_base + s_offs, zero, mask=mask)
    tl.store(
        d_subtree_donor_adjoint_ptr + out_base + s_offs, zero, mask=mask
    )
    tl.debug_barrier()
    ancestor_species = s_offs
    for _ in range(0, MAX_ANCESTOR_DEPTH):
        ancestor_valid = mask & (ancestor_species >= 0) & (ancestor_species < S)
        tl.atomic_add(
            subtree_donor_adjoint_ptr + out_base + ancestor_species,
            donor_adjoint,
            sem="relaxed",
            mask=ancestor_valid,
        )
        tl.atomic_add(
            d_subtree_donor_adjoint_ptr + out_base + ancestor_species,
            d_donor_adjoint,
            sem="relaxed",
            mask=ancestor_valid,
        )
        ancestor_species = tl.load(species_parent_ptr + ancestor_species, mask=ancestor_valid, other=-1).to(tl.int32)
    tl.debug_barrier()
    subtree_donor_adjoint = tl.load(
        subtree_donor_adjoint_ptr + out_base + s_offs,
        mask=mask,
        other=0.0,
    )
    d_subtree_donor_adjoint = tl.load(
        d_subtree_donor_adjoint_ptr + out_base + s_offs,
        mask=mask,
        other=0.0,
    )

    d_self_loop_diagonal = (
        d_within_wave_probability * (duplication_loss_probability + transfer_loss_probability)
        + within_wave_probability
        * (d_duplication_loss_probability + d_transfer_loss_probability)
    )
    # Keep this addition association stable for the uniform-path numerical
    # regression: floating-point addition is not associative.
    d_self_loop_vjp = v * d_self_loop_diagonal + d_receiver_mass * (
        total_donor_adjoint - subtree_donor_adjoint
    ) + receiver_mass * (
        d_total_donor_adjoint - d_subtree_donor_adjoint
    )
    if USE_RECEIVER_WEIGHTS:
        d_transfer_complement_vjp = d_receiver_mass * (
            total_donor_adjoint - subtree_donor_adjoint
        ) + receiver_mass * (
            d_total_donor_adjoint - d_subtree_donor_adjoint
        )
        tl.atomic_add(
            d_grad_receiver_log_probs_ptr + s_offs,
            d_transfer_complement_vjp,
            sem="relaxed",
            mask=mask,
        )
    if FOLD_RHS:
        d_self_loop_vjp += tl.load(
            d_rhs_ptr + (ws + w) * S + s_offs, mask=mask, other=0.0
        )
    tl.store(
        d_self_loop_vjp_ptr + out_base + s_offs,
        d_self_loop_vjp,
        mask=mask,
    )
    tl.debug_barrier()

    d_speciation_child1_probability = (
        d_within_wave_probability * speciation_child1_probability
        + within_wave_probability * d_speciation_child1_probability
    )
    d_speciation_child2_probability = (
        d_within_wave_probability * speciation_child2_probability
        + within_wave_probability * d_speciation_child2_probability
    )
    tl.atomic_add(
        d_self_loop_vjp_ptr + out_base + c1,
        v * d_speciation_child1_probability,
        sem="relaxed",
        mask=child1_valid,
    )
    tl.atomic_add(
        d_self_loop_vjp_ptr + out_base + c2,
        v * d_speciation_child2_probability,
        sem="relaxed",
        mask=child2_valid,
    )


def wave_backward_so(
    Pi_star, dPi, Pibar_star, dPibar, v, ws, W, S,
    pibar_row_max,
    duplication_loss_const, d_duplication_loss_const,
    Ebar, dEbar, E, dE,
    speciation_child1_const, d_speciation_child1_const,
    speciation_child2_const, d_speciation_child2_const,
    receiver_log_probs, species_child1, species_child2, species_parent, max_ancestor_depth,
    gene_split_log_likelihood=None, d_gene_split_log_likelihood=None,
    *, leaf_species_idx, leaf_logp, d_leaf_logp, family_idx, has_leaf_term=True,
    use_receiver_weights=False, d_rhs=None, dreceiver_log_probs=None,
    pi_offset, pibar_offset, gene_split_offset=None,
):
    """Return the wave second-order contraction documented in LaTeX."""
    has_splits = gene_split_log_likelihood is not None
    fold_rhs = d_rhs is not None
    _, const_row_stride = _prepare_wave_launch(S, duplication_loss_const)
    block_s = int(triton.next_power_of_2(S))
    device, dtype = Pi_star.device, Pi_star.dtype
    _validate_residual_tensors(
        Pi_star,
        dPi=dPi,
        Pibar_star=Pibar_star,
        dPibar=dPibar,
        v=v,
        pibar_row_max=pibar_row_max,
        duplication_loss_const=duplication_loss_const,
        d_duplication_loss_const=d_duplication_loss_const,
        Ebar=Ebar,
        dEbar=dEbar,
        E=E,
        dE=dE,
        speciation_child1_const=speciation_child1_const,
        d_speciation_child1_const=d_speciation_child1_const,
        speciation_child2_const=speciation_child2_const,
        d_speciation_child2_const=d_speciation_child2_const,
        receiver_log_probs=receiver_log_probs,
        dreceiver_log_probs=dreceiver_log_probs,
        leaf_logp=leaf_logp,
        d_leaf_logp=d_leaf_logp,
        gene_split_log_likelihood=gene_split_log_likelihood,
        d_gene_split_log_likelihood=d_gene_split_log_likelihood,
        d_rhs=d_rhs,
    )
    expected_rows = int(Pi_star.shape[0])
    pi_offset = _validate_offset_tensor(
        "pi_offset",
        pi_offset,
        rows=expected_rows,
        device=device,
        residual_dtype=Pi_star.dtype,
    )
    accumulator_dtype = pi_offset.dtype
    pibar_offset = _validate_offset_tensor(
        "pibar_offset",
        pibar_offset,
        rows=expected_rows,
        device=device,
        dtype=accumulator_dtype,
    )
    if has_splits:
        if gene_split_offset is None:
            raise ValueError("gene_split_offset is required for split-wave second order")
        gene_split_offset = _validate_offset_tensor(
            "gene_split_offset",
            gene_split_offset,
            rows=W,
            device=device,
            dtype=accumulator_dtype,
        )
    elif gene_split_offset is not None:
        raise ValueError("gene_split_offset is only valid for a split wave")
    d_self_loop_vjp = torch.empty((W, S), device=device, dtype=dtype)
    d_local_event_vjps = tuple(
        torch.empty((W, S), device=device, dtype=dtype) for _ in range(6)
    )
    subtree_donor_adjoint = torch.empty((W, S), device=device, dtype=dtype)
    d_subtree_donor_adjoint = torch.empty(
        (W, S), device=device, dtype=dtype
    )
    d_grad_receiver_log_probs = torch.zeros((S,), device=device, dtype=dtype)
    dummy = Pi_star
    dreceiver_log_probs_arg = (
        torch.zeros_like(receiver_log_probs)
        if dreceiver_log_probs is None
        else dreceiver_log_probs
    )
    _reconciliation_vjp_directional_derivative_kernel[(int(W),)](
        Pi_star, dPi, Pibar_star, dPibar,
        pi_offset,
        pibar_offset,
        v,
        ws, S, S,
        pibar_row_max,
        duplication_loss_const, d_duplication_loss_const,
        Ebar, dEbar, E, dE,
        speciation_child1_const, d_speciation_child1_const,
        speciation_child2_const, d_speciation_child2_const,
        receiver_log_probs, dreceiver_log_probs_arg,
        species_child1, species_child2, species_parent,
        leaf_species_idx, leaf_logp, d_leaf_logp,
        family_idx,
        gene_split_log_likelihood if has_splits else dummy,
        d_gene_split_log_likelihood if has_splits else dummy,
        gene_split_offset if has_splits else dummy,
        has_splits,
        d_self_loop_vjp, d_rhs if fold_rhs else dummy,
        d_grad_receiver_log_probs,
        *d_local_event_vjps,
        subtree_donor_adjoint, d_subtree_donor_adjoint,
        CONST_ROW_STRIDE=const_row_stride,
        BLOCK_S=block_s,
        MAX_ANCESTOR_DEPTH=int(max_ancestor_depth),
        USE_LEAF_INDEX=bool(has_leaf_term),
        USE_RECEIVER_WEIGHTS=bool(use_receiver_weights),
        FOLD_RHS=fold_rhs,
        DTYPE=_tl_float_dtype(Pi_star.dtype),
        num_warps=8,
    )
    return (
        d_self_loop_vjp,
        *d_local_event_vjps,
        d_grad_receiver_log_probs,
    )
