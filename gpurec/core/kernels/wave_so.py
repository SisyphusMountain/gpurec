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
# The same additive species-tree sums the forward and the tangent use, so the second-order
# contraction differentiates the primal the other two actually compute.
from gpurec.core.kernels.species_tree_sums import (
    off_subtree_sum,
    species_neighbourhood,
    valid_receiver_sum,
)


# Warps per program for the wave second-order contraction
# (``_reconciliation_vjp_directional_derivative_kernel``). One program holds one clade row's whole
# species tile, so the warp count decides how many species lanes each thread carries, and this
# kernel needs the full 255 registers per thread. Measured on the RTX 4090 at S=2013, one probe's
# 190 launches: 57.4 ms at 4 warps, 35.9 at 8 (the value it was launched with before), 31.1 at 16,
# 30.2 at 32; the answer moves by 3e-7 relative, a warp reduction order, far inside float32 noise.
# Measured on the 4090 only.
_NUM_WARPS_WAVE_SO = 32


# ``ws`` is the wave's start row and changes every launch; keeping it out of the
# specialization key avoids one JIT compile per divisibility state (see README.md).
@triton.jit(do_not_specialize=["ws"])
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
    species_child1_ptr, species_child2_ptr, species_parent_ptr, species_height_ptr,
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
    active_mask_ptr,   # optional [W] bool row-activity mask (the adjoint pruner's)
    CONST_ROW_STRIDE: tl.constexpr,
    BLOCK_S: tl.constexpr,
    N_LEVELS: tl.constexpr,
    USE_LEAF_INDEX: tl.constexpr,
    USE_RECEIVER_WEIGHTS: tl.constexpr,
    FOLD_RHS: tl.constexpr,
    USE_ACTIVE_MASK: tl.constexpr,
    DTYPE: tl.constexpr,
):
    LN2 = 0.6931471805599453
    NEG = -float("inf")
    # int64: w ranges over the whole batch's clade rows, so (ws+w)*stride can
    # overflow int32 once total_clades * S exceeds 2^31.
    w = tl.program_id(0).to(tl.int64)
    pi_base = (ws + w) * stride
    out_base = w * stride
    family = tl.load(family_idx_ptr + ws + w)
    const_base = family * CONST_ROW_STRIDE

    s_offs = tl.arange(0, BLOCK_S)
    mask = s_offs < S
    zero = tl.zeros([BLOCK_S], dtype=DTYPE)

    if USE_ACTIVE_MASK:
        if tl.load(active_mask_ptr + w) == 0:
            # Every quantity this kernel produces is written as ``v * (...)`` -- read the six
            # event stores and the donor adjoint below -- and the first-order adjoint pruner has
            # already set this row's ``v`` to zero, so the whole contraction is exactly zero. The
            # seven output buffers are uninitialised (torch.empty) and the caller sums them over
            # rows, so the zeros still have to be written; the solve seed keeps the FOLD_RHS
            # pass-through, which is what the full computation would have stored. 53 % of the
            # clade rows are pruned on the 200-family Coleman batch.
            if FOLD_RHS:
                tl.store(
                    d_self_loop_vjp_ptr + out_base + s_offs,
                    tl.load(d_rhs_ptr + (ws + w) * S + s_offs, mask=mask, other=0.0),
                    mask=mask,
                )
            else:
                tl.store(d_self_loop_vjp_ptr + out_base + s_offs, zero, mask=mask)
            tl.store(d_duplication_loss_event_vjp_ptr + out_base + s_offs, zero, mask=mask)
            tl.store(d_transfer_loss_event_vjp_ptr + out_base + s_offs, zero, mask=mask)
            tl.store(d_transfer_event_vjp_ptr + out_base + s_offs, zero, mask=mask)
            tl.store(d_speciation_leaf_event_vjp_ptr + out_base + s_offs, zero, mask=mask)
            tl.store(d_speciation_child1_event_vjp_ptr + out_base + s_offs, zero, mask=mask)
            tl.store(d_speciation_child2_event_vjp_ptr + out_base + s_offs, zero, mask=mask)
            return

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
    # The stabilizing row maximum is frozen: the represented transfer
    # complement is invariant to this gauge.
    (
        species_height, child1_valid, c1, child2_valid, c2,
        has_parent, parent_safe, parent_height, has_sibling, sibling_safe,
    ) = species_neighbourhood(
        species_child1_ptr, species_child2_ptr, species_parent_ptr, species_height_ptr,
        s_offs, mask, S,
    )
    # A donor may transfer into every species that is neither itself nor one of its ancestors.
    # Both this mass and its tangent are built by ADDITION -- subtree sums bottom-up, off-chain
    # sums top-down -- exactly as the forward and the tangent build them, never as the row total
    # minus the ancestor chain (see gpurec/core/kernels/species_tree_sums.py for why that
    # subtraction is rounding noise for the species under the lane holding the row's mass).
    valid_receiver_mass = valid_receiver_sum(
        receiver_mass, mask, zero, species_height,
        child1_valid, c1, child2_valid, c2,
        has_parent, parent_safe, parent_height, has_sibling, sibling_safe, N_LEVELS,
    )
    d_valid_receiver_mass = valid_receiver_sum(
        d_receiver_mass, mask, zero, species_height,
        child1_valid, c1, child2_valid, c2,
        has_parent, parent_safe, parent_height, has_sibling, sibling_safe, N_LEVELS,
    )
    has_valid_receiver_mass = mask & (valid_receiver_mass > 0.0)
    safe_valid_receiver_mass = tl.where(
        has_valid_receiver_mass,
        valid_receiver_mass,
        tl.full([BLOCK_S], 1.0, DTYPE),
    )
    inverse_valid_receiver_mass = tl.where(
        has_valid_receiver_mass, 1.0 / safe_valid_receiver_mass, zero
    )

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

    # c1/c2/child*_valid already came back from species_neighbourhood above; the indices are
    # already 0 where the child is missing, so the gathers below can stay unconditional.
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

    # A receiver s takes mass from every donor OUTSIDE s's own subtree, so what the transfer term
    # needs per lane is that off-subtree sum of the donor adjoints (and of its tangent). This was
    # the row total minus the lane's subtree sum, built with an ancestor walk of scattered atomic
    # adds. Each donor adjoint divides by that donor's own valid receiver mass, so for a species
    # hanging under the lane holding the row's mass it is astronomically large, the total is
    # dominated by those terms, and for the dominant lane the difference cancels to rounding noise
    # of that same size -- the first-order gradient came out 1e8 times too large on a 1007-species
    # Coleman family at the loss-rate cap. Built by ADDITION instead, in registers: subtree sums
    # bottom-up, then off-subtree(child) = off-subtree(parent) + parent's own term + sibling's
    # subtree, top-down. No scratch buffer and no atomics.
    off_subtree_donor_adjoint = off_subtree_sum(
        donor_adjoint, mask, zero, species_height,
        child1_valid, c1, child2_valid, c2,
        has_parent, parent_safe, parent_height, has_sibling, sibling_safe, N_LEVELS,
    )
    d_off_subtree_donor_adjoint = off_subtree_sum(
        d_donor_adjoint, mask, zero, species_height,
        child1_valid, c1, child2_valid, c2,
        has_parent, parent_safe, parent_height, has_sibling, sibling_safe, N_LEVELS,
    )

    d_self_loop_diagonal = (
        d_within_wave_probability * (duplication_loss_probability + transfer_loss_probability)
        + within_wave_probability
        * (d_duplication_loss_probability + d_transfer_loss_probability)
    )
    # Keep this addition association stable for the uniform-path numerical
    # regression: floating-point addition is not associative.
    d_self_loop_vjp = (
        v * d_self_loop_diagonal
        + d_receiver_mass * off_subtree_donor_adjoint
        + receiver_mass * d_off_subtree_donor_adjoint
    )
    if USE_RECEIVER_WEIGHTS:
        d_transfer_complement_vjp = (
            d_receiver_mass * off_subtree_donor_adjoint
            + receiver_mass * d_off_subtree_donor_adjoint
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
    receiver_log_probs, species_child1, species_child2, species_parent,
    gene_split_log_likelihood=None, d_gene_split_log_likelihood=None,
    *, species_height, species_levels,
    leaf_species_idx, leaf_logp, d_leaf_logp, family_idx, has_leaf_term=True,
    use_receiver_weights=False, d_rhs=None, dreceiver_log_probs=None,
    pi_offset, pibar_offset, gene_split_offset=None, active_mask,
):
    """Return the wave second-order contraction documented in LaTeX.

    ``active_mask`` is the adjoint pruner's per-row activity mask for this wave (``None`` runs
    every row, which is what this function did before). Every output is a product with the row's
    first-order adjoint ``v``, so a pruned row -- where ``v`` is zero -- writes zeros and returns.

    ``species_height`` (0 at a leaf, 1 + the taller child above) and ``species_levels`` (the tree's
    height, so the number of bottom-up passes) drive the additive valid-receiver and off-subtree
    sums; the old ancestor-chain walk and its scratch buffers are gone.
    """
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
        species_child1, species_child2, species_parent, species_height,
        leaf_species_idx, leaf_logp, d_leaf_logp,
        family_idx,
        gene_split_log_likelihood if has_splits else dummy,
        d_gene_split_log_likelihood if has_splits else dummy,
        gene_split_offset if has_splits else dummy,
        has_splits,
        d_self_loop_vjp, d_rhs if fold_rhs else dummy,
        d_grad_receiver_log_probs,
        *d_local_event_vjps,
        active_mask if active_mask is not None else dummy,
        CONST_ROW_STRIDE=const_row_stride,
        BLOCK_S=block_s,
        N_LEVELS=int(species_levels),
        USE_LEAF_INDEX=bool(has_leaf_term),
        USE_RECEIVER_WEIGHTS=bool(use_receiver_weights),
        FOLD_RHS=fold_rhs,
        USE_ACTIVE_MASK=bool(active_mask is not None),
        DTYPE=_tl_float_dtype(Pi_star.dtype),
        num_warps=_NUM_WARPS_WAVE_SO,
    )
    return (
        d_self_loop_vjp,
        *d_local_event_vjps,
        d_grad_receiver_log_probs,
    )
