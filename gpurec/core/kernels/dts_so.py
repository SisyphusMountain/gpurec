"""DTS second-order kernels; see ``docs/latex/kernel_mathematics.tex``."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from gpurec.core.kernels.pi_forward import (
    _load_event_log_probability,
    _select_log_split_probs,
    _tl_float_dtype,
    _validate_offset_tensor,
    _validate_residual_tensors,
)


# Warps per program for ``_transfer_subtree_vjp_directional_derivative_kernel``, and the
# species-tree nodes its two level walks put in one tile.
#
# A CUDA warp is 32 threads on every NVIDIA GPU. That is not a setting: it is a fact about the
# hardware, and it is named here only so the tile width reads as "one node per thread". The tile
# MUST hold one node per thread, because a Triton program cannot idle a thread: all 256 issue
# every instruction of the walk, and with the species tile's 512 nodes per tile each instruction
# was issued twice whether or not the second node existed. Which is what the walks used before,
# having no width of their own.
_NUM_WARPS_TRANSFER_SUBTREE_SO = 8
_THREADS_PER_WARP = 32
_BLOCK_NODES_TRANSFER_SUBTREE_SO = _NUM_WARPS_TRANSFER_SUBTREE_SO * _THREADS_PER_WARP


# ``family_offset``/``ws`` are wave start rows, and the right-side staging views start at
# row ``n_splits`` of a shared buffer, so their byte offset is only sometimes a multiple
# of 16. Both would otherwise recompile the kernel per wave (see README.md).
@triton.jit(
    do_not_specialize=["family_offset", "ws"],
    do_not_specialize_on_alignment=["right_donor_adjoint_ptr", "d_right_donor_adjoint_ptr"],
)
def _gene_split_event_vjp_directional_derivative_kernel(
    Pi, dPi, Pibar, dPibar, Pi_offset, Pibar_offset, v_ptr,
    split_left_rows, split_right_rows, species_child1, species_child2,
    log_pD, log_pS, dlog_pD, dlog_pS, max_transfer_ptr, d_max_transfer_ptr,
    log_split_probs, reduce_idx, family_idx, family_offset, ws,
    pibar_row_max_ptr,
    d_rhs_ptr,
    left_donor_adjoint_ptr, right_donor_adjoint_ptr,
    d_left_donor_adjoint_ptr, d_right_donor_adjoint_ptr,
    d_grad_pD_ptr, d_grad_pS_ptr, d_grad_max_transfer_ptr,
    S: tl.constexpr, BLOCK_S: tl.constexpr, ROW_STRIDE: tl.constexpr,
    BY_SPECIES: tl.constexpr, MAX_TRANSFER_ROW_STRIDE: tl.constexpr,
    DTYPE: tl.constexpr,
):
    LN2 = 0.6931471805599453
    NEG_INF = -float("inf")
    # int64: n ranges over the batch's split count, so the raw n*S stores below
    # (left/right donor adjoint) can overflow int32 once n_splits * S exceeds 2^31.
    n = tl.program_id(0).to(tl.int64)
    s_block = tl.program_id(1)
    s_offs = s_block * BLOCK_S + tl.arange(0, BLOCK_S)
    mask = s_offs < S
    zero = tl.zeros([BLOCK_S], dtype=DTYPE)
    # Metadata remains int32 in memory; flattened address arithmetic is widened
    # locally so row * S cannot overflow.
    parent_wave_row = tl.load(reduce_idx + n).to(tl.int64)
    family = tl.load(family_idx + family_offset + parent_wave_row).to(tl.int64)
    left_clade_row = tl.load(split_left_rows + n).to(tl.int64)
    right_clade_row = tl.load(split_right_rows + n).to(tl.int64)
    left_base = left_clade_row * S
    right_base = right_clade_row * S
    parent_base = (ws + parent_wave_row) * S

    left_pi_offset = tl.load(Pi_offset + left_clade_row)
    right_pi_offset = tl.load(Pi_offset + right_clade_row)
    parent_pi_offset = tl.load(Pi_offset + ws + parent_wave_row)
    left_pibar_offset = tl.load(Pibar_offset + left_clade_row)
    right_pibar_offset = tl.load(Pibar_offset + right_clade_row)
    # Offsets may use wider accumulation precision. Event probabilities belong
    # to the residual recurrence, so frame shifts are narrowed exactly once at
    # this boundary before they are combined with Pi/Pibar values.
    child_frame_shift = (
        left_pi_offset + right_pi_offset - parent_pi_offset
    ).to(DTYPE)
    left_transfer_frame_shift = (
        left_pi_offset + right_pibar_offset - parent_pi_offset
    ).to(DTYPE)
    right_transfer_frame_shift = (
        right_pi_offset + left_pibar_offset - parent_pi_offset
    ).to(DTYPE)
    left_exclusion_frame_shift = (left_pi_offset - left_pibar_offset).to(DTYPE)
    right_exclusion_frame_shift = (right_pi_offset - right_pibar_offset).to(DTYPE)

    left_pi = tl.load(Pi + left_base + s_offs, mask=mask, other=NEG_INF)
    right_pi = tl.load(Pi + right_base + s_offs, mask=mask, other=NEG_INF)
    d_left_pi = tl.load(dPi + left_base + s_offs, mask=mask, other=0.0)
    d_right_pi = tl.load(dPi + right_base + s_offs, mask=mask, other=0.0)
    left_pibar = tl.load(Pibar + left_base + s_offs, mask=mask, other=NEG_INF)
    right_pibar = tl.load(Pibar + right_base + s_offs, mask=mask, other=NEG_INF)
    d_left_pibar = tl.load(dPibar + left_base + s_offs, mask=mask, other=0.0)
    d_right_pibar = tl.load(dPibar + right_base + s_offs, mask=mask, other=0.0)
    parent_reconciliation_log_likelihood = tl.load(
        Pi + parent_base + s_offs, mask=mask, other=NEG_INF
    )
    d_parent_reconciliation_log_likelihood = tl.load(
        dPi + parent_base + s_offs, mask=mask, other=0.0
    )
    parent_adjoint = tl.load(
        v_ptr + parent_wave_row * S + s_offs, mask=mask, other=0.0
    )

    duplication_log_probability = _load_event_log_probability(
        log_pD, family, s_offs, mask, S, ROW_STRIDE, BY_SPECIES, BLOCK_S, DTYPE
    )
    speciation_log_probability = _load_event_log_probability(
        log_pS, family, s_offs, mask, S, ROW_STRIDE, BY_SPECIES, BLOCK_S, DTYPE
    )
    d_duplication_log_probability = _load_event_log_probability(
        dlog_pD, family, s_offs, mask, S, ROW_STRIDE, BY_SPECIES, BLOCK_S, DTYPE
    )
    d_speciation_log_probability = _load_event_log_probability(
        dlog_pS, family, s_offs, mask, S, ROW_STRIDE, BY_SPECIES, BLOCK_S, DTYPE
    )
    max_transfer = tl.load(
        max_transfer_ptr + family * MAX_TRANSFER_ROW_STRIDE + s_offs,
        mask=mask,
        other=0.0,
    )
    d_max_transfer = tl.load(
        d_max_transfer_ptr + family * MAX_TRANSFER_ROW_STRIDE + s_offs,
        mask=mask,
        other=0.0,
    )

    c1 = tl.load(species_child1 + s_offs, mask=mask, other=S)
    c2 = tl.load(species_child2 + s_offs, mask=mask, other=S)
    child1_valid = mask & (c1 < S)
    child2_valid = mask & (c2 < S)
    left_pi_child1 = tl.load(Pi + left_base + c1, mask=child1_valid, other=NEG_INF)
    right_pi_child2 = tl.load(Pi + right_base + c2, mask=child2_valid, other=NEG_INF)
    right_pi_child1 = tl.load(Pi + right_base + c1, mask=child1_valid, other=NEG_INF)
    left_pi_child2 = tl.load(Pi + left_base + c2, mask=child2_valid, other=NEG_INF)
    d_left_pi_child1 = tl.load(dPi + left_base + c1, mask=child1_valid, other=0.0)
    d_right_pi_child2 = tl.load(dPi + right_base + c2, mask=child2_valid, other=0.0)
    d_right_pi_child1 = tl.load(dPi + right_base + c1, mask=child1_valid, other=0.0)
    d_left_pi_child2 = tl.load(dPi + left_base + c2, mask=child2_valid, other=0.0)
    split_log_prior = tl.load(log_split_probs + n)
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
        split_log_prior + speciation_log_probability + left_pi_child1 + right_pi_child2
        + child_frame_shift
    )
    speciation_rl_log_term = (
        split_log_prior + speciation_log_probability + right_pi_child1 + left_pi_child2
        + child_frame_shift
    )
    d_duplication_log_term = (
        d_duplication_log_probability + d_left_pi + d_right_pi
    )
    d_transfer_left_retained_log_term = d_left_pi + d_right_pibar
    d_transfer_right_retained_log_term = d_right_pi + d_left_pibar
    d_speciation_lr_log_term = (
        d_speciation_log_probability + d_left_pi_child1 + d_right_pi_child2
    )
    d_speciation_rl_log_term = (
        d_speciation_log_probability + d_right_pi_child1 + d_left_pi_child2
    )

    is_finite = mask & (parent_reconciliation_log_likelihood != NEG_INF)
    duplication_probability = tl.where(
        is_finite,
        tl.exp2(duplication_log_term - parent_reconciliation_log_likelihood),
        zero,
    )
    transfer_left_retained_probability = tl.where(
        is_finite,
        tl.exp2(
            transfer_left_retained_log_term
            - parent_reconciliation_log_likelihood
        ),
        zero,
    )
    transfer_right_retained_probability = tl.where(
        is_finite,
        tl.exp2(
            transfer_right_retained_log_term
            - parent_reconciliation_log_likelihood
        ),
        zero,
    )
    speciation_lr_probability = tl.where(
        is_finite,
        tl.exp2(speciation_lr_log_term - parent_reconciliation_log_likelihood),
        zero,
    )
    speciation_rl_probability = tl.where(
        is_finite,
        tl.exp2(speciation_rl_log_term - parent_reconciliation_log_likelihood),
        zero,
    )
    transfer_left_retained_event_vjp = (
        parent_adjoint * transfer_left_retained_probability
    )
    transfer_right_retained_event_vjp = (
        parent_adjoint * transfer_right_retained_probability
    )
    d_duplication_event_vjp = (
        parent_adjoint * LN2 * duplication_probability
        * (
            d_duplication_log_term
            - d_parent_reconciliation_log_likelihood
        )
    )
    d_transfer_left_retained_event_vjp = (
        parent_adjoint * LN2 * transfer_left_retained_probability
        * (
            d_transfer_left_retained_log_term
            - d_parent_reconciliation_log_likelihood
        )
    )
    d_transfer_right_retained_event_vjp = (
        parent_adjoint * LN2 * transfer_right_retained_probability
        * (
            d_transfer_right_retained_log_term
            - d_parent_reconciliation_log_likelihood
        )
    )
    d_speciation_lr_event_vjp = (
        parent_adjoint * LN2 * speciation_lr_probability
        * (d_speciation_lr_log_term - d_parent_reconciliation_log_likelihood)
    )
    d_speciation_rl_event_vjp = (
        parent_adjoint * LN2 * speciation_rl_probability
        * (d_speciation_rl_log_term - d_parent_reconciliation_log_likelihood)
    )

    # tangent of the rhs scatters (same targets as the primal)
    tl.atomic_add(d_rhs_ptr + left_base + s_offs, d_duplication_event_vjp + d_transfer_left_retained_event_vjp, sem="relaxed", mask=mask)
    tl.atomic_add(d_rhs_ptr + right_base + s_offs, d_duplication_event_vjp + d_transfer_right_retained_event_vjp, sem="relaxed", mask=mask)
    tl.atomic_add(d_rhs_ptr + left_base + c1, d_speciation_lr_event_vjp, sem="relaxed", mask=child1_valid)
    tl.atomic_add(d_rhs_ptr + right_base + c1, d_speciation_rl_event_vjp, sem="relaxed", mask=child1_valid)
    tl.atomic_add(d_rhs_ptr + right_base + c2, d_speciation_lr_event_vjp, sem="relaxed", mask=child2_valid)
    tl.atomic_add(d_rhs_ptr + left_base + c2, d_speciation_rl_event_vjp, sem="relaxed", mask=child2_valid)

    # Convert each transfer-complement event VJP into the donor-adjoint
    # coefficient used by the subtree formula. The stabilizing row maximum is
    # frozen, so only max_transfer and Pibar contribute to the scale tangent.
    left_pibar_row_max = tl.load(pibar_row_max_ptr + left_clade_row)
    right_pibar_row_max = tl.load(pibar_row_max_ptr + right_clade_row)
    left_exclusion_is_finite = mask & (left_pibar != NEG_INF)
    right_exclusion_is_finite = mask & (right_pibar != NEG_INF)
    left_exclusion_scale = tl.where(
        left_exclusion_is_finite,
        tl.exp2(
            left_pibar_row_max + max_transfer - left_pibar
            + left_exclusion_frame_shift
        ),
        zero,
    )
    right_exclusion_scale = tl.where(
        right_exclusion_is_finite,
        tl.exp2(
            right_pibar_row_max + max_transfer - right_pibar
            + right_exclusion_frame_shift
        ),
        zero,
    )
    left_donor_adjoint = (
        transfer_right_retained_event_vjp * left_exclusion_scale
    )
    right_donor_adjoint = (
        transfer_left_retained_event_vjp * right_exclusion_scale
    )
    d_left_donor_adjoint = (
        d_transfer_right_retained_event_vjp * left_exclusion_scale
        + transfer_right_retained_event_vjp * LN2 * left_exclusion_scale
        * (d_max_transfer - d_left_pibar)
    )
    d_right_donor_adjoint = (
        d_transfer_left_retained_event_vjp * right_exclusion_scale
        + transfer_left_retained_event_vjp * LN2 * right_exclusion_scale
        * (d_max_transfer - d_right_pibar)
    )
    tl.store(left_donor_adjoint_ptr + n * S + s_offs, left_donor_adjoint, mask=mask)
    tl.store(right_donor_adjoint_ptr + n * S + s_offs, right_donor_adjoint, mask=mask)
    tl.store(d_left_donor_adjoint_ptr + n * S + s_offs, d_left_donor_adjoint, mask=mask)
    tl.store(d_right_donor_adjoint_ptr + n * S + s_offs, d_right_donor_adjoint, mask=mask)

    # parameter tangents (same buckets as the primal accumulations)
    tl.atomic_add(d_grad_pD_ptr + family * S + s_offs, d_duplication_event_vjp, sem="relaxed", mask=mask)
    tl.atomic_add(d_grad_pS_ptr + family * S + s_offs, d_speciation_lr_event_vjp + d_speciation_rl_event_vjp, sem="relaxed", mask=mask)
    tl.atomic_add(d_grad_max_transfer_ptr + family * S + s_offs, d_transfer_left_retained_event_vjp + d_transfer_right_retained_event_vjp, sem="relaxed", mask=mask)


# ``n_ws`` is the wave's split count and changes every launch; keeping it out of the
# specialization key avoids one JIT compile per divisibility state (see README.md).
@triton.jit(do_not_specialize=["n_ws"])
def _transfer_subtree_vjp_directional_derivative_kernel(
    Pi_ptr, dPi_ptr, receiver_log_probs_ptr, dreceiver_log_probs_ptr,
    donor_adjoint_ptr, d_donor_adjoint_ptr,
    internal_node_own_ptr, d_internal_node_own_ptr,
    split_left_rows_ptr, split_right_rows_ptr,
    pibar_row_max_ptr,
    species_parent_ptr,
    level_offsets_ptr, level_parents_ptr,
    level_child1_ptr, level_child2_ptr,
    d_rhs_ptr, d_grad_receiver_log_probs_ptr,
    n_ws,  # runtime int (per-wave split count; constexpr caused one JIT compile per wave shape)
    S: tl.constexpr, stride_C: tl.constexpr,
    BLOCK_S: tl.constexpr, BLOCK_NODES: tl.constexpr,
    N_LEVELS: tl.constexpr, N_COMPACT_NODES: tl.constexpr,
    USE_RECEIVER_WEIGHTS: tl.constexpr, DTYPE: tl.constexpr,
):
    """Evaluate the DTS transfer-tree curvature term documented in LaTeX.

    Receiver s takes donor mass from every donor OUTSIDE s's own subtree, so this kernel needs,
    per species s, this split side's donor adjoint -- and its tangent -- summed off s's subtree.
    That used to be ``row total - subtree sum``. Each donor's adjoint carries the reciprocal of
    that donor's own valid receiver mass, so for species hanging under the lane holding the row's
    mass it is astronomically large; the row total is dominated by those terms and, for the
    dominant lane whose subtree holds them all, the difference cancels to rounding noise of that
    same size. The first-order twin of this kernel measured a gradient 1e8 times too large on a
    1007-species Coleman family at the loss-rate cap.

    Built by ADDITION only, exactly as
    :func:`gpurec.core.kernels.wave_backward_kernels._accumulate_transfer_subtree_vjp_kernel`
    does it: subtree sums bottom-up over the compact level tables, then a top-down walk over the
    same tables, ``off-subtree(child) = off-subtree(parent) + parent's own term + sibling's
    subtree sum``, each child's off-subtree sum overwriting its no-longer-needed subtree sum. The
    bottom-up pass destroys each internal node's own term, so it is parked first in
    ``internal_node_own_ptr`` / ``d_internal_node_own_ptr``, indexed exactly like the level
    tables; recovering it as ``subtree(parent) - subtree(c1) - subtree(c2)`` would be the same
    cancelling subtraction one level down.
    """
    LN2 = 0.6931471805599453
    NEG = -float("inf")
    # int64: row ranges over 2*n_splits, so row_base below can overflow int32
    # once that count * S exceeds 2^31.
    row = tl.program_id(0).to(tl.int64)
    split_i = tl.where(row < n_ws, row, row - n_ws)
    is_right = row >= n_ws
    child_l = tl.load(split_left_rows_ptr + split_i).to(tl.int64)
    child_r = tl.load(split_right_rows_ptr + split_i).to(tl.int64)
    child = tl.where(is_right, child_r, child_l)

    pi_base = child * stride_C
    row_base = row * S
    receiver_mass_log_scale = tl.load(pibar_row_max_ptr + child)
    receiver_mass_log_scale_safe = tl.where(
        receiver_mass_log_scale != NEG,
        receiver_mass_log_scale,
        tl.zeros_like(receiver_mass_log_scale),
    )

    own_base = internal_node_own_ptr + row * N_COMPACT_NODES
    d_own_base = d_internal_node_own_ptr + row * N_COMPACT_NODES

    # All warps must reach the same point before any warp overwrites an internal node with its
    # subtree sum: without this barrier one warp could replace u[parent] by u[parent]+u[c1]+u[c2]
    # while another warp is still reading u[parent] as its own term.
    tl.debug_barrier()
    for level in range(0, N_LEVELS):
        level_start = tl.load(level_offsets_ptr + level)
        level_end = tl.load(level_offsets_ptr + level + 1)
        p_start = level_start
        while p_start < level_end:
            node_offs = p_start + tl.arange(0, BLOCK_NODES)
            node_mask = node_offs < level_end
            parent = tl.load(level_parents_ptr + node_offs, mask=node_mask, other=-1)
            c1 = tl.load(level_child1_ptr + node_offs, mask=node_mask, other=S)
            c2 = tl.load(level_child2_ptr + node_offs, mask=node_mask, other=S)
            parent_valid = node_mask & (parent >= 0) & (parent < S)
            c1_mask = node_mask & (c1 >= 0) & (c1 < S)
            c2_mask = node_mask & (c2 >= 0) & (c2 < S)
            parent_adjoint = tl.load(
                donor_adjoint_ptr + row_base + parent,
                mask=parent_valid,
                other=0.0,
            )
            child1_adjoint = tl.load(
                donor_adjoint_ptr + row_base + c1, mask=c1_mask, other=0.0
            )
            child2_adjoint = tl.load(
                donor_adjoint_ptr + row_base + c2, mask=c2_mask, other=0.0
            )
            tl.store(own_base + node_offs, parent_adjoint, mask=parent_valid)
            tl.store(
                donor_adjoint_ptr + row_base + parent,
                parent_adjoint + child1_adjoint + child2_adjoint,
                mask=parent_valid,
            )
            d_parent_adjoint = tl.load(
                d_donor_adjoint_ptr + row_base + parent,
                mask=parent_valid,
                other=0.0,
            )
            d_child1_adjoint = tl.load(
                d_donor_adjoint_ptr + row_base + c1,
                mask=c1_mask,
                other=0.0,
            )
            d_child2_adjoint = tl.load(
                d_donor_adjoint_ptr + row_base + c2,
                mask=c2_mask,
                other=0.0,
            )
            tl.store(d_own_base + node_offs, d_parent_adjoint, mask=parent_valid)
            tl.store(
                d_donor_adjoint_ptr + row_base + parent,
                d_parent_adjoint + d_child1_adjoint + d_child2_adjoint,
                mask=parent_valid,
            )
            p_start += BLOCK_NODES
        tl.debug_barrier()

    # The root's subtree is the whole tree: nothing lies off it. Every other species is some
    # internal node's child, so the top-down walk below writes it exactly once.
    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        species_mask = s_offs < S
        species_parent = tl.load(species_parent_ptr + s_offs, mask=species_mask, other=0)
        is_root_lane = species_mask & (species_parent < 0)
        zero_lane = tl.zeros([BLOCK_S], dtype=DTYPE)
        tl.store(donor_adjoint_ptr + row_base + s_offs, zero_lane, mask=is_root_lane)
        tl.store(d_donor_adjoint_ptr + row_base + s_offs, zero_lane, mask=is_root_lane)
    tl.debug_barrier()
    for level_index in range(0, N_LEVELS):
        level = N_LEVELS - 1 - level_index
        level_start = tl.load(level_offsets_ptr + level)
        level_end = tl.load(level_offsets_ptr + level + 1)
        p_start = level_start
        while p_start < level_end:
            node_offs = p_start + tl.arange(0, BLOCK_NODES)
            node_mask = node_offs < level_end
            parent = tl.load(level_parents_ptr + node_offs, mask=node_mask, other=-1)
            c1 = tl.load(level_child1_ptr + node_offs, mask=node_mask, other=S)
            c2 = tl.load(level_child2_ptr + node_offs, mask=node_mask, other=S)
            parent_valid = node_mask & (parent >= 0) & (parent < S)
            c1_mask = parent_valid & (c1 >= 0) & (c1 < S)
            c2_mask = parent_valid & (c2 >= 0) & (c2 < S)

            parent_off_subtree = tl.load(
                donor_adjoint_ptr + row_base + parent, mask=parent_valid, other=0.0
            )
            parent_own = tl.load(own_base + node_offs, mask=parent_valid, other=0.0)
            c1_subtree = tl.load(
                donor_adjoint_ptr + row_base + c1, mask=c1_mask, other=0.0
            )
            c2_subtree = tl.load(
                donor_adjoint_ptr + row_base + c2, mask=c2_mask, other=0.0
            )
            tl.store(
                donor_adjoint_ptr + row_base + c1,
                parent_off_subtree + parent_own + c2_subtree,
                mask=c1_mask,
            )
            tl.store(
                donor_adjoint_ptr + row_base + c2,
                parent_off_subtree + parent_own + c1_subtree,
                mask=c2_mask,
            )

            d_parent_off_subtree = tl.load(
                d_donor_adjoint_ptr + row_base + parent, mask=parent_valid, other=0.0
            )
            d_parent_own = tl.load(d_own_base + node_offs, mask=parent_valid, other=0.0)
            d_c1_subtree = tl.load(
                d_donor_adjoint_ptr + row_base + c1, mask=c1_mask, other=0.0
            )
            d_c2_subtree = tl.load(
                d_donor_adjoint_ptr + row_base + c2, mask=c2_mask, other=0.0
            )
            tl.store(
                d_donor_adjoint_ptr + row_base + c1,
                d_parent_off_subtree + d_parent_own + d_c2_subtree,
                mask=c1_mask,
            )
            tl.store(
                d_donor_adjoint_ptr + row_base + c2,
                d_parent_off_subtree + d_parent_own + d_c1_subtree,
                mask=c2_mask,
            )
            p_start += BLOCK_NODES
        tl.debug_barrier()

    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        mask = s_offs < S
        pi_val = tl.load(Pi_ptr + pi_base + s_offs, mask=mask, other=NEG)
        dpi_val = tl.load(dPi_ptr + pi_base + s_offs, mask=mask, other=0.0)
        if USE_RECEIVER_WEIGHTS:
            receiver_log_probability = tl.load(
                receiver_log_probs_ptr + s_offs, mask=mask, other=NEG
            )
            d_receiver_log_probability = tl.load(
                dreceiver_log_probs_ptr + s_offs, mask=mask, other=0.0
            )
            receiver_mass = tl.exp2(
                receiver_log_probability + pi_val - receiver_mass_log_scale_safe
            )
            receiver_mass = tl.where(
                pi_val != NEG, receiver_mass, tl.zeros_like(receiver_mass)
            )
            d_receiver_mass = (
                LN2 * receiver_mass
                * (dpi_val + d_receiver_log_probability)
            )
        else:
            receiver_mass = tl.exp2(pi_val - receiver_mass_log_scale_safe)
            receiver_mass = tl.where(
                pi_val != NEG, receiver_mass, tl.zeros_like(receiver_mass)
            )
            d_receiver_mass = LN2 * receiver_mass * dpi_val
        off_subtree_donor_adjoint = tl.load(
            donor_adjoint_ptr + row_base + s_offs, mask=mask, other=0.0
        )
        d_off_subtree_donor_adjoint = tl.load(
            d_donor_adjoint_ptr + row_base + s_offs,
            mask=mask,
            other=0.0,
        )
        d_transfer_complement_vjp = (
            d_receiver_mass * off_subtree_donor_adjoint
            + receiver_mass * d_off_subtree_donor_adjoint
        )
        tl.atomic_add(
            d_rhs_ptr + pi_base + s_offs,
            d_transfer_complement_vjp,
            sem="relaxed",
            mask=mask,
        )
        if USE_RECEIVER_WEIGHTS:
            tl.atomic_add(
                d_grad_receiver_log_probs_ptr + s_offs,
                d_transfer_complement_vjp,
                sem="relaxed",
                mask=mask,
            )


def dts_backward_so(
    Pi, dPi, Pibar, dPibar, v, ws, meta, S,
    log_pD_param, log_pS_param, dlog_pD_param, dlog_pS_param,
    max_transfer, d_max_transfer,
    receiver_log_probs, species_child1, species_child2, pibar_row_max, family_idx,
    d_rhs, d_grad_pD, d_grad_pS, d_grad_max_transfer,
    d_grad_receiver_log_probs,
    *, species_parent, compact_level_ptr=None, compact_level_parents=None,
    compact_level_child1=None, compact_level_child2=None,
    use_receiver_weights=False, dreceiver_log_probs=None, pi_offset, pibar_offset,
):
    """Accumulate the DTS second-order contraction documented in LaTeX."""
    split_left_rows = meta["sl"]
    split_right_rows = meta["sr"]
    n_splits = int(split_left_rows.numel())
    device, dtype = Pi.device, Pi.dtype
    split_log_priors = _select_log_split_probs(meta, Pi.dtype)
    _validate_residual_tensors(
        Pi,
        dPi=dPi,
        Pibar=Pibar,
        dPibar=dPibar,
        v=v,
        log_pD_param=log_pD_param,
        log_pS_param=log_pS_param,
        dlog_pD_param=dlog_pD_param,
        dlog_pS_param=dlog_pS_param,
        max_transfer=max_transfer,
        d_max_transfer=d_max_transfer,
        receiver_log_probs=receiver_log_probs,
        pibar_row_max=pibar_row_max,
        d_rhs=d_rhs,
        d_grad_pD=d_grad_pD,
        d_grad_pS=d_grad_pS,
        d_grad_max_transfer=d_grad_max_transfer,
        d_grad_receiver_log_probs=d_grad_receiver_log_probs,
        dreceiver_log_probs=dreceiver_log_probs,
        log_split_probs=split_log_priors,
    )
    expected_rows = int(Pi.shape[0])
    pi_offset = _validate_offset_tensor(
        "pi_offset",
        pi_offset,
        rows=expected_rows,
        device=device,
        residual_dtype=Pi.dtype,
    )
    pibar_offset = _validate_offset_tensor(
        "pibar_offset",
        pibar_offset,
        rows=expected_rows,
        device=device,
        dtype=pi_offset.dtype,
    )
    if n_splits == 0:
        return
    if split_log_priors is None:
        split_log_priors = torch.zeros(
            (n_splits,), device=Pi.device, dtype=Pi.dtype
        )
    else:
        split_log_priors = (
            split_log_priors.reshape(n_splits).contiguous()
        )
    by_species = log_pD_param.ndim == 2 and int(log_pD_param.shape[1]) != 1
    row_stride = 0 if int(log_pD_param.shape[0]) == 1 else int(log_pD_param.stride(0))
    max_transfer_row_stride = 0 if int(max_transfer.shape[0]) == 1 else int(max_transfer.stride(0))
    block_s = min(512, triton.next_power_of_2(S))

    # The split kernel accumulates the second-order log_pD/log_pS cotangents per SPECIES via the
    # d_grad_p*_ptr + family*S + s layout (identical to d_grad_max_transfer). That matches a [rows, S] buffer,
    # but genewise/global pass a species-REDUCED d_grad_pD/pS ([G,1] / [1,1]) because the rate is a
    # per-family (or global) scalar and the first-order path reduces the species axis internally.
    # Writing family*S+s into a [*,1] buffer runs off the end (only s==0/1 land; the rest silently
    # corrupt adjacent pool memory). So when the caller's buffer is species-reduced (shape[1]==1),
    # hand the kernel a [rows, S] scratch (rows = d_grad_max_transfer's family-row count -- the layout `family`
    # indexes) and sum the species axis back into the caller's buffer afterward. Specieswise ([1,S])
    # already matches the kernel layout and writes straight through, bit-for-bit unchanged.
    pd_reduced = int(d_grad_pD.shape[1]) == 1
    ps_reduced = int(d_grad_pS.shape[1]) == 1
    rows = int(d_grad_max_transfer.shape[0])
    d_grad_pD_kernel = (
        torch.zeros((rows, S), device=device, dtype=dtype)
        if pd_reduced
        else d_grad_pD
    )
    d_grad_pS_kernel = (
        torch.zeros((rows, S), device=device, dtype=dtype)
        if ps_reduced
        else d_grad_pS
    )

    # Stacked staging: the first n_splits rows are left sides and the second
    # n_splits rows are right sides. The tree kernel walks all split sides.
    donor_adjoint = torch.empty(
        (2 * n_splits, S), device=device, dtype=dtype
    )
    d_donor_adjoint = torch.empty(
        (2 * n_splits, S), device=device, dtype=dtype
    )
    left_donor_adjoint, right_donor_adjoint = (
        donor_adjoint[:n_splits],
        donor_adjoint[n_splits:],
    )
    d_left_donor_adjoint, d_right_donor_adjoint = (
        d_donor_adjoint[:n_splits],
        d_donor_adjoint[n_splits:],
    )

    _gene_split_event_vjp_directional_derivative_kernel[(n_splits, triton.cdiv(S, block_s))](
        Pi, dPi, Pibar, dPibar,
        pi_offset,
        pibar_offset,
        v,
        split_left_rows, split_right_rows, species_child1, species_child2,
        log_pD_param, log_pS_param, dlog_pD_param, dlog_pS_param, max_transfer, d_max_transfer,
        split_log_priors, meta["reduce_idx"], family_idx,
        int(meta["start"]), int(ws),
        pibar_row_max,
        d_rhs,
        left_donor_adjoint, right_donor_adjoint,
        d_left_donor_adjoint, d_right_donor_adjoint,
        d_grad_pD_kernel, d_grad_pS_kernel, d_grad_max_transfer,
        S, BLOCK_S=block_s, ROW_STRIDE=row_stride, BY_SPECIES=bool(by_species),
        MAX_TRANSFER_ROW_STRIDE=max_transfer_row_stride,
        DTYPE=_tl_float_dtype(Pi.dtype),
    )
    if pd_reduced:
        d_grad_pD += d_grad_pD_kernel.sum(dim=-1, keepdim=True)
    if ps_reduced:
        d_grad_pS += d_grad_pS_kernel.sum(dim=-1, keepdim=True)

    # Compact level tables pack only internal species-tree nodes by bottom-up
    # depth. They evaluate every subtree sum after its children while omitting
    # leaves, whose initial staged values are already complete subtree sums.
    if compact_level_ptr is None:
        raise ValueError("dts_backward_so requires compact_level_* tables for the tree kernel")
    n_levels = int(compact_level_ptr.numel()) - 1
    dreceiver_log_probs_arg = (
        torch.zeros_like(receiver_log_probs)
        if dreceiver_log_probs is None
        else dreceiver_log_probs.to(device=device, dtype=dtype)
        .reshape(S)
        .contiguous()
    )
    # The bottom-up pass overwrites each internal node's own donor adjoint with its subtree sum,
    # and the top-down pass that follows needs that own term back (see the kernel's docstring).
    # One slot per compact level-table entry -- that is, per internal species node -- per split
    # side, indexed exactly as the level tables are; one buffer for the adjoint, one for its
    # tangent.
    compact_level_parents = compact_level_parents.contiguous()
    n_compact_nodes = int(compact_level_parents.numel())
    internal_node_own_donor_adjoint = torch.empty(
        (2 * n_splits, n_compact_nodes), device=device, dtype=dtype
    )
    d_internal_node_own_donor_adjoint = torch.empty(
        (2 * n_splits, n_compact_nodes), device=device, dtype=dtype
    )
    _transfer_subtree_vjp_directional_derivative_kernel[(2 * n_splits,)](
        Pi, dPi, receiver_log_probs, dreceiver_log_probs_arg,
        donor_adjoint, d_donor_adjoint,
        internal_node_own_donor_adjoint, d_internal_node_own_donor_adjoint,
        split_left_rows, split_right_rows,
        pibar_row_max,
        species_parent.to(device=device, dtype=torch.int32).contiguous(),
        compact_level_ptr.contiguous(), compact_level_parents,
        compact_level_child1.contiguous(), compact_level_child2.contiguous(),
        d_rhs, d_grad_receiver_log_probs,
        n_ws=n_splits, S=S, stride_C=int(Pi.stride(0)),
        BLOCK_S=block_s, BLOCK_NODES=_BLOCK_NODES_TRANSFER_SUBTREE_SO,
        N_LEVELS=n_levels, N_COMPACT_NODES=n_compact_nodes,
        USE_RECEIVER_WEIGHTS=bool(use_receiver_weights), DTYPE=_tl_float_dtype(Pi.dtype),
        # num_warps=8 trims _dts_tree_so ~8% vs 4 on 666x80 (back-to-back wall 997->989ms;
        # nsys kernel 12%->11% of HVP). Each program owns a full side-row walked in BLOCK_S
        # chunks -> more warps hide the dependent level-walk loads. split kernel unaffected (kept
        # at Triton's default). Bit-identical (dts_so/hvp gates unchanged).
        num_warps=_NUM_WARPS_TRANSFER_SUBTREE_SO,
    )
