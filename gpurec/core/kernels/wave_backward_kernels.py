"""Triton @triton.jit device kernels for the wave-step VJP (backward pass).

Split out of wave_backward.py (mechanical AST extraction, 2026-07-07): these
are the compiled device kernels. The host-side orchestration that launches
them lives in wave_backward.py, which re-imports every name defined here so
callers and tests keep reaching them as ``wave_backward.<name>``.
"""
import torch
import triton
import triton.language as tl

@triton.jit
def _select_active_adjoint_rows_kernel(
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


@triton.jit
def _prepare_reconciliation_self_loop_vjp_kernel(
    Pi_star_ptr,
    Pibar_star_ptr,
    Pi_offset_ptr,
    Pibar_offset_ptr,
    Pibar_row_max_ptr,
    gene_split_log_likelihood_ptr,
    gene_split_offset_ptr,
    has_splits: tl.constexpr,
    rhs_ptr,
    active_mask_ptr,
    max_transfer_ptr, duplication_loss_const_ptr, Ebar_ptr, E_ptr, speciation_child1_const_ptr, speciation_child2_const_ptr,
    receiver_log_probs_ptr,
    species_child1_ptr, species_child2_ptr, species_parent_ptr,
    leaf_term_ptr,
    leaf_species_ptr,
    leaf_logp_ptr,
    family_idx_ptr,
    v_k_ptr,
    self_loop_diagonal_ptr,
    donor_adjoint_coefficient_ptr,
    receiver_mass_ptr,
    speciation_child1_probability_ptr,
    speciation_child2_probability_ptr,
    ws,
    W,
    S: tl.constexpr,
    stride: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_S: tl.constexpr,
    MAX_ANCESTOR_DEPTH: tl.constexpr,
    USE_LEAF_INDEX: tl.constexpr,
    HAS_LEAF_TERM: tl.constexpr,
    LEAF_LOGP_MODE: tl.constexpr,
    USE_ACTIVE_MASK: tl.constexpr,
    SKIP_INACTIVE_SCRATCH_ZERO: tl.constexpr,
    CONST_LAYOUT: tl.constexpr,
    DTYPE: tl.constexpr,
    USE_CHILD_EDGE_SELF_LOOP: tl.constexpr,
    USE_RECEIVER_WEIGHTS: tl.constexpr,
):
    """Precompute self-loop J^T coefficients for a block of rows and all species."""
    NEG_LARGE: tl.constexpr = -float("inf")

    block = tl.program_id(0)
    rows = block * BLOCK_W + tl.arange(0, BLOCK_W)
    s_offs = tl.arange(0, BLOCK_S)
    row_valid = rows < W
    species_valid = s_offs < S
    if USE_ACTIVE_MASK:
        row_active = tl.load(active_mask_ptr + rows, mask=row_valid, other=0) != 0
    else:
        row_active = row_valid
    row_mask = row_valid & row_active
    mask = species_valid[:, None] & row_mask[None, :]
    if SKIP_INACTIVE_SCRATCH_ZERO:
        store_mask = mask
    else:
        store_mask = species_valid[:, None] & row_valid[None, :]

    row_global = ws + rows
    pi_offsets = row_global[None, :] * stride + s_offs[:, None]
    out_offsets = rows[None, :] * S + s_offs[:, None]

    pi_row_offset = tl.load(
        Pi_offset_ptr + row_global, mask=row_valid, other=0.0
    )
    pibar_row_offset = tl.load(
        Pibar_offset_ptr + row_global, mask=row_valid, other=0.0
    )
    pibar_offset_corr = (pibar_row_offset - pi_row_offset).to(DTYPE)
    leaf_offset_corr = (-pi_row_offset).to(DTYPE)
    if has_splits:
        gene_split_row_offset = tl.load(
            gene_split_offset_ptr + rows, mask=row_valid, other=0.0
        )
        gene_split_frame_shift = (gene_split_row_offset - pi_row_offset).to(DTYPE)
    else:
        gene_split_frame_shift = tl.zeros([BLOCK_W], dtype=DTYPE)

    row_max = tl.load(Pibar_row_max_ptr + row_global, mask=row_valid, other=NEG_LARGE).to(DTYPE)
    reconciliation_log_likelihood = tl.load(Pi_star_ptr + pi_offsets, mask=mask, other=NEG_LARGE).to(DTYPE)
    transfer_complement_log_likelihood = tl.load(Pibar_star_ptr + pi_offsets, mask=mask, other=NEG_LARGE).to(DTYPE)
    row_max_safe = tl.where(row_max != NEG_LARGE, row_max, tl.zeros_like(row_max))
    if USE_RECEIVER_WEIGHTS:
        receiver_log_probability = tl.load(receiver_log_probs_ptr + s_offs, mask=species_valid, other=NEG_LARGE).to(DTYPE)
        receiver_mass = tl.exp2(receiver_log_probability[:, None] + reconciliation_log_likelihood - row_max_safe[None, :])
    else:
        receiver_mass = tl.exp2(reconciliation_log_likelihood - row_max_safe[None, :])
    total_receiver_mass = tl.sum(tl.where(mask, receiver_mass, tl.zeros([BLOCK_S, BLOCK_W], dtype=DTYPE)), axis=0)

    family = tl.full([BLOCK_W], value=0, dtype=tl.int64)
    const_base = tl.zeros([BLOCK_W], dtype=tl.int64)
    if CONST_LAYOUT == 1:
        const_offsets = out_offsets
    elif CONST_LAYOUT == 2:
        family = tl.load(family_idx_ptr + row_global, mask=row_valid, other=0).to(tl.int64)
        const_base = family * stride
        const_offsets = const_base[None, :] + s_offs[:, None]
    else:
        const_offsets = s_offs[:, None]

    if CONST_LAYOUT == 0:
        const_mask = species_valid[:, None]
    else:
        const_mask = mask
    duplication_loss_const = tl.load(duplication_loss_const_ptr + const_offsets, mask=const_mask, other=NEG_LARGE).to(DTYPE)
    extinction_complement_log_probability = tl.load(
        Ebar_ptr + const_offsets, mask=const_mask, other=NEG_LARGE
    ).to(DTYPE)
    extinction_log_probability = tl.load(
        E_ptr + const_offsets, mask=const_mask, other=NEG_LARGE
    ).to(DTYPE)
    speciation_child1_const = tl.load(speciation_child1_const_ptr + const_offsets, mask=const_mask, other=NEG_LARGE).to(DTYPE)
    speciation_child2_const = tl.load(speciation_child2_const_ptr + const_offsets, mask=const_mask, other=NEG_LARGE).to(DTYPE)

    c1 = tl.load(species_child1_ptr + s_offs, mask=species_valid, other=0)
    c2 = tl.load(species_child2_ptr + s_offs, mask=species_valid, other=0)
    c1_valid = c1 < S
    c2_valid = c2 < S
    reconciliation_child1_log_likelihood = tl.load(
        Pi_star_ptr + row_global[None, :] * stride + c1[:, None],
        mask=(species_valid & c1_valid)[:, None] & row_mask[None, :],
        other=NEG_LARGE,
    ).to(DTYPE)
    reconciliation_child2_log_likelihood = tl.load(
        Pi_star_ptr + row_global[None, :] * stride + c2[:, None],
        mask=(species_valid & c2_valid)[:, None] & row_mask[None, :],
        other=NEG_LARGE,
    ).to(DTYPE)

    duplication_loss_log_term = duplication_loss_const + reconciliation_log_likelihood
    transfer_loss_log_term = (
        reconciliation_log_likelihood + extinction_complement_log_probability
    )
    transfer_log_term = (
        transfer_complement_log_likelihood
        + extinction_log_probability
        + pibar_offset_corr[None, :]
    )
    speciation_child1_log_term = speciation_child1_const + reconciliation_child1_log_likelihood
    speciation_child2_log_term = speciation_child2_const + reconciliation_child2_log_likelihood
    if USE_LEAF_INDEX:
        leaf_species = tl.load(leaf_species_ptr + row_global, mask=row_valid, other=-1)
        leaf_hit = mask & (leaf_species[None, :] == s_offs[:, None])
        if LEAF_LOGP_MODE == 3:
            leaf_logp = tl.load(leaf_logp_ptr).to(DTYPE)
            leaf_observation_log_term = tl.where(leaf_hit, leaf_logp, NEG_LARGE)
        elif LEAF_LOGP_MODE == 1:
            leaf_logp = tl.load(leaf_logp_ptr + family, mask=row_valid, other=NEG_LARGE).to(DTYPE)
            leaf_observation_log_term = tl.where(leaf_hit, leaf_logp[None, :], NEG_LARGE)
        elif LEAF_LOGP_MODE == 2:
            leaf_logp = tl.load(
                leaf_logp_ptr + const_base[None, :] + s_offs[:, None],
                mask=leaf_hit,
                other=NEG_LARGE,
            ).to(DTYPE)
            leaf_observation_log_term = tl.where(leaf_hit, leaf_logp, NEG_LARGE)
        else:
            leaf_logp = tl.load(leaf_logp_ptr + s_offs, mask=species_valid, other=NEG_LARGE).to(DTYPE)
            leaf_observation_log_term = tl.where(leaf_hit, leaf_logp[:, None], NEG_LARGE)
    elif HAS_LEAF_TERM:
        leaf_observation_log_term = tl.load(leaf_term_ptr + out_offsets, mask=mask, other=NEG_LARGE).to(DTYPE)
    else:
        leaf_observation_log_term = tl.full([BLOCK_S, BLOCK_W], value=NEG_LARGE, dtype=DTYPE)
    if USE_LEAF_INDEX or HAS_LEAF_TERM:
        leaf_observation_log_term += leaf_offset_corr[None, :]

    local_event_max = tl.maximum(duplication_loss_log_term, transfer_loss_log_term)
    local_event_max = tl.maximum(local_event_max, transfer_log_term)
    local_event_max = tl.maximum(local_event_max, speciation_child1_log_term)
    local_event_max = tl.maximum(local_event_max, speciation_child2_log_term)
    local_event_max = tl.maximum(local_event_max, leaf_observation_log_term)
    local_event_max_safe = tl.where(local_event_max != NEG_LARGE, local_event_max, tl.zeros([BLOCK_S, BLOCK_W], dtype=DTYPE))
    duplication_loss_mass = tl.exp2(duplication_loss_log_term - local_event_max_safe)
    transfer_loss_mass = tl.exp2(transfer_loss_log_term - local_event_max_safe)
    transfer_mass = tl.exp2(transfer_log_term - local_event_max_safe)
    speciation_child1_mass = tl.exp2(speciation_child1_log_term - local_event_max_safe)
    speciation_child2_mass = tl.exp2(speciation_child2_log_term - local_event_max_safe)
    leaf_observation_mass = tl.exp2(leaf_observation_log_term - local_event_max_safe)
    local_event_scaled_mass = (
        duplication_loss_mass + transfer_loss_mass + transfer_mass
        + speciation_child1_mass + speciation_child2_mass + leaf_observation_mass
    )
    inverse_local_event_scaled_mass = tl.where(local_event_scaled_mass > 0.0, 1.0 / local_event_scaled_mass, tl.zeros_like(local_event_scaled_mass))

    if has_splits:
        gene_split_log_likelihood = tl.load(gene_split_log_likelihood_ptr + out_offsets, mask=mask, other=NEG_LARGE).to(DTYPE)
        gene_split_log_likelihood += gene_split_frame_shift[None, :]
        local_event_log_likelihood = tl.log2(local_event_scaled_mass) + local_event_max
        updated_reconciliation_max = tl.maximum(local_event_log_likelihood, gene_split_log_likelihood)
        updated_reconciliation_max_safe = tl.where(updated_reconciliation_max != NEG_LARGE, updated_reconciliation_max, tl.zeros_like(updated_reconciliation_max))
        updated_reconciliation_log_likelihood = tl.log2(tl.exp2(local_event_log_likelihood - updated_reconciliation_max_safe) + tl.exp2(gene_split_log_likelihood - updated_reconciliation_max_safe)) + updated_reconciliation_max
        within_wave_probability = tl.where(local_event_log_likelihood != NEG_LARGE, tl.exp2(local_event_log_likelihood - updated_reconciliation_log_likelihood), tl.zeros_like(local_event_log_likelihood))
    else:
        within_wave_probability = tl.full([BLOCK_S, BLOCK_W], value=1.0, dtype=DTYPE)

    ancestor_sum = tl.zeros([BLOCK_S, BLOCK_W], dtype=DTYPE)
    ancestor_species = s_offs
    for _depth in range(MAX_ANCESTOR_DEPTH):
        ancestor_valid = (
            species_valid & (ancestor_species >= 0) & (ancestor_species < S)
        )
        ancestor_reconciliation_log_likelihood = tl.load(
            Pi_star_ptr
            + row_global[None, :] * stride
            + ancestor_species[:, None],
            mask=ancestor_valid[:, None] & row_mask[None, :],
            other=NEG_LARGE,
        ).to(DTYPE)
        if USE_RECEIVER_WEIGHTS:
            ancestor_receiver_log_probability = tl.load(
                receiver_log_probs_ptr + ancestor_species,
                mask=ancestor_valid,
                other=NEG_LARGE,
            ).to(DTYPE)
            ancestor_sum += tl.where(
                ancestor_valid[:, None] & row_mask[None, :],
                tl.exp2(
                    ancestor_receiver_log_probability[:, None]
                    + ancestor_reconciliation_log_likelihood
                    - row_max_safe[None, :]
                ),
                tl.zeros([BLOCK_S, BLOCK_W], dtype=DTYPE),
            )
        else:
            ancestor_sum += tl.where(
                ancestor_valid[:, None] & row_mask[None, :],
                tl.exp2(
                    ancestor_reconciliation_log_likelihood
                    - row_max_safe[None, :]
                ),
                tl.zeros([BLOCK_S, BLOCK_W], dtype=DTYPE),
            )
        ancestor_species = tl.load(
            species_parent_ptr + ancestor_species,
            mask=ancestor_valid,
            other=-1,
        )
    valid_receiver_mass = total_receiver_mass[None, :] - ancestor_sum
    inverse_valid_receiver_mass = tl.where(valid_receiver_mass > 0.0, 1.0 / valid_receiver_mass, tl.zeros_like(valid_receiver_mass))

    self_loop_diagonal = within_wave_probability * (duplication_loss_mass + transfer_loss_mass) * inverse_local_event_scaled_mass
    donor_adjoint_coefficient = within_wave_probability * transfer_mass * inverse_local_event_scaled_mass * inverse_valid_receiver_mass
    speciation_child1_probability = within_wave_probability * speciation_child1_mass * inverse_local_event_scaled_mass
    speciation_child2_probability = within_wave_probability * speciation_child2_mass * inverse_local_event_scaled_mass

    zero = tl.zeros([BLOCK_S, BLOCK_W], dtype=DTYPE)
    rhs_val = tl.load(rhs_ptr + out_offsets, mask=mask, other=0.0).to(DTYPE)
    tl.store(v_k_ptr + out_offsets, tl.where(mask, rhs_val, zero), mask=store_mask)
    tl.store(self_loop_diagonal_ptr + out_offsets, tl.where(mask, self_loop_diagonal, zero), mask=store_mask)
    tl.store(donor_adjoint_coefficient_ptr + out_offsets, tl.where(mask, donor_adjoint_coefficient, zero), mask=store_mask)
    tl.store(receiver_mass_ptr + out_offsets, tl.where(mask, receiver_mass, zero), mask=store_mask)
    if USE_CHILD_EDGE_SELF_LOOP:
        child1_offsets = rows[None, :] * S + c1[:, None]
        child2_offsets = rows[None, :] * S + c2[:, None]
        child1_mask = (species_valid & c1_valid)[:, None] & row_mask[None, :]
        child2_mask = (species_valid & c2_valid)[:, None] & row_mask[None, :]
        tl.store(speciation_child1_probability_ptr + child1_offsets, speciation_child1_probability, mask=child1_mask)
        tl.store(speciation_child1_probability_ptr + child2_offsets, speciation_child2_probability, mask=child2_mask)
    else:
        tl.store(speciation_child1_probability_ptr + out_offsets, tl.where(mask, speciation_child1_probability, zero), mask=store_mask)
        tl.store(speciation_child2_probability_ptr + out_offsets, tl.where(mask, speciation_child2_probability, zero), mask=store_mask)


@triton.jit
def _apply_reconciliation_self_loop_transpose_kernel(
    term_in_ptr,
    term_out_ptr,
    rhs_update_ptr,
    active_mask_ptr,
    self_loop_diagonal_ptr,
    donor_adjoint_coefficient_ptr,
    receiver_mass_ptr,
    speciation_child1_probability_ptr,
    speciation_child2_probability_ptr,
    species_child1_ptr,
    species_child2_ptr,
    species_parent_ptr,
    compact_level_ptr,
    compact_level_parent_ptr,
    compact_level_child1_ptr,
    compact_level_child2_ptr,
    subtree_donor_adjoint_ptr,
    v_k_ptr,
    W,
    S: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_NODES: tl.constexpr,
    N_LEVELS: tl.constexpr,
    USE_ACTIVE_MASK: tl.constexpr,
    SKIP_INACTIVE_SCRATCH_ZERO: tl.constexpr,
    FIXED_POINT_UPDATE: tl.constexpr,
    DTYPE: tl.constexpr,
    USE_CHILD_EDGE_SELF_LOOP: tl.constexpr,
    OUTPUT_A: tl.constexpr,
    ACCUMULATE_V: tl.constexpr,
):
    """Apply one self-loop J^T term using in-program bottom-up tree reduction."""
    block = tl.program_id(0)
    rows = block * BLOCK_W + tl.arange(0, BLOCK_W)
    s_offs = tl.arange(0, BLOCK_S)
    row_valid = rows < W
    species_valid = s_offs < S
    if USE_ACTIVE_MASK:
        row_active = tl.load(active_mask_ptr + rows, mask=row_valid, other=0) != 0
    else:
        row_active = row_valid
    row_mask = row_valid & row_active
    mask = species_valid[:, None] & row_mask[None, :]
    if SKIP_INACTIVE_SCRATCH_ZERO:
        store_mask = mask
    else:
        store_mask = species_valid[:, None] & row_valid[None, :]
    offsets = rows[None, :] * S + s_offs[:, None]

    input_adjoint = tl.load(term_in_ptr + offsets, mask=mask, other=0.0).to(DTYPE)
    donor_adjoint_coefficient = tl.load(donor_adjoint_coefficient_ptr + offsets, mask=mask, other=0.0).to(DTYPE)
    donor_adjoint = input_adjoint * donor_adjoint_coefficient
    total_donor_adjoint = tl.sum(tl.where(mask, donor_adjoint, tl.zeros([BLOCK_S, BLOCK_W], dtype=DTYPE)), axis=0)
    tl.store(subtree_donor_adjoint_ptr + offsets, tl.where(mask, donor_adjoint, tl.zeros_like(donor_adjoint)), mask=store_mask)

    tl.debug_barrier()

    for level in range(0, N_LEVELS):
        level_start = tl.load(compact_level_ptr + level)
        level_end = tl.load(compact_level_ptr + level + 1)
        node_start = level_start
        while node_start < level_end:
            node_offs = node_start + tl.arange(0, BLOCK_NODES)
            node_mask = node_offs < level_end
            parent = tl.load(compact_level_parent_ptr + node_offs, mask=node_mask, other=0)
            c1 = tl.load(compact_level_child1_ptr + node_offs, mask=node_mask, other=S)
            c2 = tl.load(compact_level_child2_ptr + node_offs, mask=node_mask, other=S)
            reduce_mask = node_mask[:, None] & row_mask[None, :]
            row_base = rows[None, :] * S
            parent_val = tl.load(
                subtree_donor_adjoint_ptr + row_base + parent[:, None],
                mask=reduce_mask,
                other=0.0,
            ).to(DTYPE)
            c1_val = tl.load(
                subtree_donor_adjoint_ptr + row_base + c1[:, None],
                mask=reduce_mask & (c1 < S)[:, None],
                other=0.0,
            ).to(DTYPE)
            c2_val = tl.load(
                subtree_donor_adjoint_ptr + row_base + c2[:, None],
                mask=reduce_mask & (c2 < S)[:, None],
                other=0.0,
            ).to(DTYPE)
            tl.store(
                subtree_donor_adjoint_ptr + row_base + parent[:, None],
                parent_val + c1_val + c2_val,
                mask=reduce_mask,
            )
            node_start += BLOCK_NODES
        tl.debug_barrier()

    tl.debug_barrier()

    subtree_donor_adjoint = tl.load(subtree_donor_adjoint_ptr + offsets, mask=mask, other=0.0).to(DTYPE)
    self_loop_diagonal = tl.load(self_loop_diagonal_ptr + offsets, mask=mask, other=0.0).to(DTYPE)
    receiver_mass = tl.load(receiver_mass_ptr + offsets, mask=mask, other=0.0).to(DTYPE)
    self_loop_vjp_without_child_edges = (
        input_adjoint * self_loop_diagonal
        + receiver_mass
        * (total_donor_adjoint[None, :] - subtree_donor_adjoint)
    )

    if USE_CHILD_EDGE_SELF_LOOP:
        parent = tl.load(species_parent_ptr + s_offs, mask=species_valid, other=-1)
        parent_valid = species_valid & (parent >= 0) & (parent < S)
        row_base = rows[None, :] * S
        parent_mask = parent_valid[:, None] & row_mask[None, :]
        parent_input_adjoint = tl.load(
            term_in_ptr + row_base + parent[:, None],
            mask=parent_mask,
            other=0.0,
        ).to(DTYPE)
        speciation_parent_to_child_probability = tl.load(
            speciation_child1_probability_ptr + offsets,
            mask=parent_mask,
            other=0.0,
        ).to(DTYPE)
        self_loop_vjp = (
            self_loop_vjp_without_child_edges
            + parent_input_adjoint * speciation_parent_to_child_probability
        )
    else:
        tl.store(
            term_out_ptr + offsets,
            tl.where(
                mask,
                self_loop_vjp_without_child_edges,
                tl.zeros_like(self_loop_vjp_without_child_edges),
            ),
            mask=store_mask,
        )

        tl.debug_barrier()

        c1 = tl.load(species_child1_ptr + s_offs, mask=species_valid, other=S)
        c2 = tl.load(species_child2_ptr + s_offs, mask=species_valid, other=S)
        speciation_child1_probability = tl.load(speciation_child1_probability_ptr + offsets, mask=mask, other=0.0).to(DTYPE)
        speciation_child2_probability = tl.load(speciation_child2_probability_ptr + offsets, mask=mask, other=0.0).to(DTYPE)
        row_base = rows[None, :] * S
        c1_mask = (species_valid & (c1 < S))[:, None] & row_mask[None, :]
        c2_mask = (species_valid & (c2 < S))[:, None] & row_mask[None, :]
        current_child1_vjp = tl.load(
            term_out_ptr + row_base + c1[:, None],
            mask=c1_mask,
            other=0.0,
        ).to(DTYPE)
        current_child2_vjp = tl.load(
            term_out_ptr + row_base + c2[:, None],
            mask=c2_mask,
            other=0.0,
        ).to(DTYPE)
        tl.store(
            term_out_ptr + row_base + c1[:, None],
            current_child1_vjp
            + input_adjoint * speciation_child1_probability,
            mask=c1_mask,
        )
        tl.store(
            term_out_ptr + row_base + c2[:, None],
            current_child2_vjp
            + input_adjoint * speciation_child2_probability,
            mask=c2_mask,
        )

        tl.debug_barrier()

        self_loop_vjp = tl.load(
            term_out_ptr + offsets, mask=mask, other=0.0
        ).to(DTYPE)

    operator_output = (
        input_adjoint - self_loop_vjp if OUTPUT_A else self_loop_vjp
    )
    tl.store(
        term_out_ptr + offsets,
        tl.where(mask, operator_output, tl.zeros_like(operator_output)),
        mask=store_mask,
    )

    if FIXED_POINT_UPDATE:
        rhs_val = tl.load(rhs_update_ptr + offsets, mask=mask, other=0.0).to(DTYPE)
        tl.store(
            v_k_ptr + offsets,
            tl.where(mask, rhs_val + self_loop_vjp, tl.zeros_like(self_loop_vjp)),
            mask=store_mask,
        )
    elif ACCUMULATE_V:
        v_prev = tl.load(v_k_ptr + offsets, mask=mask, other=0.0).to(DTYPE)
        tl.store(v_k_ptr + offsets, v_prev + self_loop_vjp, mask=mask)


@triton.jit
def _accumulate_transfer_receiver_log_probability_vjp_kernel(
    v_k_ptr,
    active_mask_ptr,
    donor_adjoint_coefficient_ptr,
    receiver_mass_ptr,
    compact_level_ptr,
    compact_level_parent_ptr,
    compact_level_child1_ptr,
    compact_level_child2_ptr,
    subtree_donor_adjoint_ptr,
    grad_receiver_log_probs_ptr,
    W,
    S: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_NODES: tl.constexpr,
    N_LEVELS: tl.constexpr,
    USE_ACTIVE_MASK: tl.constexpr,
    DTYPE: tl.constexpr,
):
    block = tl.program_id(0)
    rows = block * BLOCK_W + tl.arange(0, BLOCK_W)
    s_offs = tl.arange(0, BLOCK_S)
    row_valid = rows < W
    species_valid = s_offs < S
    if USE_ACTIVE_MASK:
        row_active = tl.load(active_mask_ptr + rows, mask=row_valid, other=0) != 0
    else:
        row_active = row_valid
    row_mask = row_valid & row_active
    mask = species_valid[:, None] & row_mask[None, :]
    store_mask = species_valid[:, None] & row_valid[None, :]
    offsets = rows[None, :] * S + s_offs[:, None]

    input_adjoint = tl.load(v_k_ptr + offsets, mask=mask, other=0.0).to(DTYPE)
    donor_adjoint_coefficient = tl.load(donor_adjoint_coefficient_ptr + offsets, mask=mask, other=0.0).to(DTYPE)
    donor_adjoint = input_adjoint * donor_adjoint_coefficient
    zero = tl.zeros([BLOCK_S, BLOCK_W], dtype=DTYPE)
    total_donor_adjoint = tl.sum(tl.where(mask, donor_adjoint, zero), axis=0)
    tl.store(subtree_donor_adjoint_ptr + offsets, tl.where(mask, donor_adjoint, zero), mask=store_mask)

    tl.debug_barrier()

    for level in range(0, N_LEVELS):
        level_start = tl.load(compact_level_ptr + level)
        level_end = tl.load(compact_level_ptr + level + 1)
        node_start = level_start
        while node_start < level_end:
            node_offs = node_start + tl.arange(0, BLOCK_NODES)
            node_mask = node_offs < level_end
            parent = tl.load(compact_level_parent_ptr + node_offs, mask=node_mask, other=0)
            c1 = tl.load(compact_level_child1_ptr + node_offs, mask=node_mask, other=S)
            c2 = tl.load(compact_level_child2_ptr + node_offs, mask=node_mask, other=S)
            reduce_mask = node_mask[:, None] & row_mask[None, :]
            row_base = rows[None, :] * S
            parent_val = tl.load(
                subtree_donor_adjoint_ptr + row_base + parent[:, None],
                mask=reduce_mask,
                other=0.0,
            ).to(DTYPE)
            c1_val = tl.load(
                subtree_donor_adjoint_ptr + row_base + c1[:, None],
                mask=reduce_mask & (c1 < S)[:, None],
                other=0.0,
            ).to(DTYPE)
            c2_val = tl.load(
                subtree_donor_adjoint_ptr + row_base + c2[:, None],
                mask=reduce_mask & (c2 < S)[:, None],
                other=0.0,
            ).to(DTYPE)
            tl.store(
                subtree_donor_adjoint_ptr + row_base + parent[:, None],
                parent_val + c1_val + c2_val,
                mask=reduce_mask,
            )
            node_start += BLOCK_NODES
        tl.debug_barrier()

    subtree_donor_adjoint = tl.load(subtree_donor_adjoint_ptr + offsets, mask=mask, other=0.0).to(DTYPE)
    receiver_mass = tl.load(receiver_mass_ptr + offsets, mask=mask, other=0.0).to(DTYPE)
    transfer_complement_vjp = receiver_mass * (total_donor_adjoint[None, :] - subtree_donor_adjoint)
    species_contrib = tl.sum(tl.where(mask, transfer_complement_vjp, zero), axis=1)
    tl.atomic_add(
        grad_receiver_log_probs_ptr + s_offs,
        species_contrib,
        sem="relaxed",
        mask=species_valid,
    )


@triton.jit
def _accumulate_reconciliation_event_vjp_kernel(
    Pi_star_ptr,
    Pibar_star_ptr,
    Pi_offset_ptr,
    Pibar_offset_ptr,
    gene_split_log_likelihood_ptr,
    gene_split_offset_ptr,
    has_splits: tl.constexpr,
    v_k_ptr,
    active_mask_ptr,
    max_transfer_ptr, duplication_loss_const_ptr, Ebar_ptr, E_ptr, speciation_child1_const_ptr, speciation_child2_const_ptr,
    species_child1_ptr, species_child2_ptr,
    leaf_term_ptr,
    leaf_species_ptr,
    leaf_logp_ptr,
    family_idx_ptr,
    grad_log_pD_ptr,
    grad_log_pS_ptr,
    grad_E_ptr,
    grad_Ebar_ptr,
    grad_E_s1_ptr,
    grad_E_s2_ptr,
    grad_max_transfer_ptr,
    duplication_loss_event_vjp_ptr,
    transfer_loss_event_vjp_ptr,
    transfer_event_vjp_ptr,
    speciation_leaf_event_vjp_ptr,
    speciation_child1_event_vjp_ptr,
    speciation_child2_event_vjp_ptr,
    ws,
    W,
    S: tl.constexpr,
    stride: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_S: tl.constexpr,
    USE_LEAF_INDEX: tl.constexpr,
    HAS_LEAF_TERM: tl.constexpr,
    LEAF_LOGP_MODE: tl.constexpr,
    USE_ACTIVE_MASK: tl.constexpr,
    CONST_LAYOUT: tl.constexpr,
    ACCUM_GRADS: tl.constexpr,
    PARAM_GRAD_VECTOR: tl.constexpr,
    DTYPE: tl.constexpr,
):
    """Store per-element self-loop parameter VJP contributions after Neumann."""
    NEG_LARGE: tl.constexpr = -float("inf")

    block = tl.program_id(0)
    rows = block * BLOCK_W + tl.arange(0, BLOCK_W)
    s_offs = tl.arange(0, BLOCK_S)
    row_valid = rows < W
    species_valid = s_offs < S
    if USE_ACTIVE_MASK:
        row_active = tl.load(active_mask_ptr + rows, mask=row_valid, other=0) != 0
    else:
        row_active = row_valid
    row_mask = row_valid & row_active
    mask = species_valid[:, None] & row_mask[None, :]
    store_mask = species_valid[:, None] & row_valid[None, :]
    row_global = ws + rows
    pi_offsets = row_global[None, :] * stride + s_offs[:, None]
    out_offsets = rows[None, :] * S + s_offs[:, None]

    pi_row_offset = tl.load(
        Pi_offset_ptr + row_global, mask=row_valid, other=0.0
    )
    pibar_row_offset = tl.load(
        Pibar_offset_ptr + row_global, mask=row_valid, other=0.0
    )
    pibar_offset_corr = (pibar_row_offset - pi_row_offset).to(DTYPE)
    leaf_offset_corr = (-pi_row_offset).to(DTYPE)
    if has_splits:
        gene_split_row_offset = tl.load(
            gene_split_offset_ptr + rows, mask=row_valid, other=0.0
        )
        gene_split_frame_shift = (gene_split_row_offset - pi_row_offset).to(DTYPE)
    else:
        gene_split_frame_shift = tl.zeros([BLOCK_W], dtype=DTYPE)

    family = tl.full([BLOCK_W], value=0, dtype=tl.int64)
    const_base = tl.zeros([BLOCK_W], dtype=tl.int64)
    if CONST_LAYOUT == 1:
        const_offsets = out_offsets
    elif CONST_LAYOUT == 2:
        family = tl.load(family_idx_ptr + row_global, mask=row_valid, other=0).to(tl.int64)
        const_base = family * stride
        const_offsets = const_base[None, :] + s_offs[:, None]
    else:
        const_offsets = s_offs[:, None]

    const_mask = species_valid[:, None] if CONST_LAYOUT == 0 else mask
    reconciliation_log_likelihood = tl.load(Pi_star_ptr + pi_offsets, mask=mask, other=NEG_LARGE).to(DTYPE)
    transfer_complement_log_likelihood = tl.load(Pibar_star_ptr + pi_offsets, mask=mask, other=NEG_LARGE).to(DTYPE)
    parent_adjoint = tl.load(v_k_ptr + out_offsets, mask=mask, other=0.0).to(DTYPE)
    duplication_loss_const = tl.load(duplication_loss_const_ptr + const_offsets, mask=const_mask, other=NEG_LARGE).to(DTYPE)
    extinction_complement_log_probability = tl.load(
        Ebar_ptr + const_offsets, mask=const_mask, other=NEG_LARGE
    ).to(DTYPE)
    extinction_log_probability = tl.load(
        E_ptr + const_offsets, mask=const_mask, other=NEG_LARGE
    ).to(DTYPE)
    speciation_child1_const = tl.load(speciation_child1_const_ptr + const_offsets, mask=const_mask, other=NEG_LARGE).to(DTYPE)
    speciation_child2_const = tl.load(speciation_child2_const_ptr + const_offsets, mask=const_mask, other=NEG_LARGE).to(DTYPE)

    c1 = tl.load(species_child1_ptr + s_offs, mask=species_valid, other=0)
    c2 = tl.load(species_child2_ptr + s_offs, mask=species_valid, other=0)
    c1_valid = c1 < S
    c2_valid = c2 < S
    reconciliation_child1_log_likelihood = tl.load(
        Pi_star_ptr + row_global[None, :] * stride + c1[:, None],
        mask=(species_valid & c1_valid)[:, None] & row_mask[None, :],
        other=NEG_LARGE,
    ).to(DTYPE)
    reconciliation_child2_log_likelihood = tl.load(
        Pi_star_ptr + row_global[None, :] * stride + c2[:, None],
        mask=(species_valid & c2_valid)[:, None] & row_mask[None, :],
        other=NEG_LARGE,
    ).to(DTYPE)

    duplication_loss_log_term = duplication_loss_const + reconciliation_log_likelihood
    transfer_loss_log_term = (
        reconciliation_log_likelihood + extinction_complement_log_probability
    )
    transfer_log_term = (
        transfer_complement_log_likelihood
        + extinction_log_probability
        + pibar_offset_corr[None, :]
    )
    speciation_child1_log_term = speciation_child1_const + reconciliation_child1_log_likelihood
    speciation_child2_log_term = speciation_child2_const + reconciliation_child2_log_likelihood
    if USE_LEAF_INDEX:
        leaf_species = tl.load(leaf_species_ptr + row_global, mask=row_valid, other=-1)
        leaf_hit = mask & (leaf_species[None, :] == s_offs[:, None])
        if LEAF_LOGP_MODE == 3:
            leaf_logp = tl.load(leaf_logp_ptr).to(DTYPE)
            leaf_observation_log_term = tl.where(leaf_hit, leaf_logp, NEG_LARGE)
        elif LEAF_LOGP_MODE == 1:
            leaf_logp = tl.load(leaf_logp_ptr + family, mask=row_valid, other=NEG_LARGE).to(DTYPE)
            leaf_observation_log_term = tl.where(leaf_hit, leaf_logp[None, :], NEG_LARGE)
        elif LEAF_LOGP_MODE == 2:
            leaf_logp = tl.load(
                leaf_logp_ptr + const_base[None, :] + s_offs[:, None],
                mask=leaf_hit,
                other=NEG_LARGE,
            ).to(DTYPE)
            leaf_observation_log_term = tl.where(leaf_hit, leaf_logp, NEG_LARGE)
        else:
            leaf_logp = tl.load(leaf_logp_ptr + s_offs, mask=species_valid, other=NEG_LARGE).to(DTYPE)
            leaf_observation_log_term = tl.where(leaf_hit, leaf_logp[:, None], NEG_LARGE)
    elif HAS_LEAF_TERM:
        leaf_observation_log_term = tl.load(leaf_term_ptr + out_offsets, mask=mask, other=NEG_LARGE).to(DTYPE)
    else:
        leaf_observation_log_term = tl.full([BLOCK_S, BLOCK_W], value=NEG_LARGE, dtype=DTYPE)
    if USE_LEAF_INDEX or HAS_LEAF_TERM:
        leaf_observation_log_term += leaf_offset_corr[None, :]

    local_event_max = tl.maximum(duplication_loss_log_term, transfer_loss_log_term)
    local_event_max = tl.maximum(local_event_max, transfer_log_term)
    local_event_max = tl.maximum(local_event_max, speciation_child1_log_term)
    local_event_max = tl.maximum(local_event_max, speciation_child2_log_term)
    local_event_max = tl.maximum(local_event_max, leaf_observation_log_term)
    local_event_max_safe = tl.where(local_event_max != NEG_LARGE, local_event_max, tl.zeros([BLOCK_S, BLOCK_W], dtype=DTYPE))
    duplication_loss_mass = tl.exp2(duplication_loss_log_term - local_event_max_safe)
    transfer_loss_mass = tl.exp2(transfer_loss_log_term - local_event_max_safe)
    transfer_mass = tl.exp2(transfer_log_term - local_event_max_safe)
    speciation_child1_mass = tl.exp2(speciation_child1_log_term - local_event_max_safe)
    speciation_child2_mass = tl.exp2(speciation_child2_log_term - local_event_max_safe)
    leaf_observation_mass = tl.exp2(leaf_observation_log_term - local_event_max_safe)
    local_event_scaled_mass = (
        duplication_loss_mass + transfer_loss_mass + transfer_mass
        + speciation_child1_mass + speciation_child2_mass + leaf_observation_mass
    )
    inverse_local_event_scaled_mass = tl.where(local_event_scaled_mass > 0.0, 1.0 / local_event_scaled_mass, tl.zeros_like(local_event_scaled_mass))

    if has_splits:
        gene_split_log_likelihood = tl.load(gene_split_log_likelihood_ptr + out_offsets, mask=mask, other=NEG_LARGE).to(DTYPE)
        gene_split_log_likelihood += gene_split_frame_shift[None, :]
        local_event_log_likelihood = tl.log2(local_event_scaled_mass) + local_event_max
        updated_reconciliation_max = tl.maximum(local_event_log_likelihood, gene_split_log_likelihood)
        updated_reconciliation_max_safe = tl.where(updated_reconciliation_max != NEG_LARGE, updated_reconciliation_max, tl.zeros_like(updated_reconciliation_max))
        updated_reconciliation_log_likelihood = tl.log2(tl.exp2(local_event_log_likelihood - updated_reconciliation_max_safe) + tl.exp2(gene_split_log_likelihood - updated_reconciliation_max_safe)) + updated_reconciliation_max
        within_wave_probability = tl.where(local_event_log_likelihood != NEG_LARGE, tl.exp2(local_event_log_likelihood - updated_reconciliation_log_likelihood), tl.zeros_like(local_event_log_likelihood))
    else:
        within_wave_probability = tl.full([BLOCK_S, BLOCK_W], value=1.0, dtype=DTYPE)

    within_wave_adjoint = parent_adjoint * within_wave_probability
    duplication_loss_event_vjp = within_wave_adjoint * duplication_loss_mass * inverse_local_event_scaled_mass
    transfer_loss_event_vjp = within_wave_adjoint * transfer_loss_mass * inverse_local_event_scaled_mass
    transfer_event_vjp = within_wave_adjoint * transfer_mass * inverse_local_event_scaled_mass
    speciation_child1_event_vjp = within_wave_adjoint * speciation_child1_mass * inverse_local_event_scaled_mass
    speciation_child2_event_vjp = within_wave_adjoint * speciation_child2_mass * inverse_local_event_scaled_mass
    leaf_observation_event_vjp = within_wave_adjoint * leaf_observation_mass * inverse_local_event_scaled_mass
    speciation_leaf_event_vjp = (
        speciation_child1_event_vjp + speciation_child2_event_vjp + leaf_observation_event_vjp
    )
    zero = tl.zeros([BLOCK_S, BLOCK_W], dtype=DTYPE)
    if ACCUM_GRADS:
        duplication_loss_species_vjp = tl.sum(tl.where(mask, duplication_loss_event_vjp, zero), axis=1)
        transfer_loss_species_vjp = tl.sum(tl.where(mask, transfer_loss_event_vjp, zero), axis=1)
        transfer_species_vjp = tl.sum(tl.where(mask, transfer_event_vjp, zero), axis=1)
        speciation_leaf_species_vjp = tl.sum(tl.where(mask, speciation_leaf_event_vjp, zero), axis=1)
        speciation_child1_species_vjp = tl.sum(tl.where(mask, speciation_child1_event_vjp, zero), axis=1)
        speciation_child2_species_vjp = tl.sum(tl.where(mask, speciation_child2_event_vjp, zero), axis=1)
        if PARAM_GRAD_VECTOR:
            tl.atomic_add(
                grad_log_pD_ptr + s_offs,
                duplication_loss_species_vjp,
                sem="relaxed",
                mask=species_valid,
            )
            tl.atomic_add(
                grad_log_pS_ptr + s_offs,
                speciation_leaf_species_vjp,
                sem="relaxed",
                mask=species_valid,
            )
        else:
            tl.atomic_add(grad_log_pD_ptr, tl.sum(duplication_loss_species_vjp, axis=0), sem="relaxed")
            tl.atomic_add(grad_log_pS_ptr, tl.sum(speciation_leaf_species_vjp, axis=0), sem="relaxed")
        tl.atomic_add(
            grad_E_ptr + s_offs,
            duplication_loss_species_vjp + transfer_species_vjp,
            sem="relaxed",
            mask=species_valid,
        )
        tl.atomic_add(
            grad_Ebar_ptr + s_offs,
            transfer_loss_species_vjp,
            sem="relaxed",
            mask=species_valid,
        )
        tl.atomic_add(
            grad_E_s1_ptr + s_offs,
            speciation_child2_species_vjp,
            sem="relaxed",
            mask=species_valid,
        )
        tl.atomic_add(
            grad_E_s2_ptr + s_offs,
            speciation_child1_species_vjp,
            sem="relaxed",
            mask=species_valid,
        )
        tl.atomic_add(
            grad_max_transfer_ptr + s_offs,
            transfer_species_vjp,
            sem="relaxed",
            mask=species_valid,
        )
    else:
        tl.store(duplication_loss_event_vjp_ptr + out_offsets, tl.where(mask, duplication_loss_event_vjp, zero), mask=store_mask)
        tl.store(transfer_loss_event_vjp_ptr + out_offsets, tl.where(mask, transfer_loss_event_vjp, zero), mask=store_mask)
        tl.store(transfer_event_vjp_ptr + out_offsets, tl.where(mask, transfer_event_vjp, zero), mask=store_mask)
        tl.store(speciation_leaf_event_vjp_ptr + out_offsets, tl.where(mask, speciation_leaf_event_vjp, zero), mask=store_mask)
        tl.store(speciation_child1_event_vjp_ptr + out_offsets, tl.where(mask, speciation_child1_event_vjp, zero), mask=store_mask)
        tl.store(speciation_child2_event_vjp_ptr + out_offsets, tl.where(mask, speciation_child2_event_vjp, zero), mask=store_mask)


@triton.jit
def _accumulate_gene_split_event_vjp_kernel(
    # Converged values [C, S]
    Pi_star_ptr,
    Pibar_star_ptr,
    Pi_offset_ptr,
    Pibar_offset_ptr,
    # Neumann-solved adjoint [W, S]
    v_k_ptr,
    active_mask_ptr,   # optional [W] bool parent row activity mask
    # Split metadata
    split_left_rows_ptr,            # [n_ws] int64 — left child global clade index
    split_right_rows_ptr,            # [n_ws] int64 — right child global clade index
    reduce_idx_ptr,    # [n_ws] int64 — wave-local parent index
    log_split_probs_ptr,          # [n_ws] float — log split probability (squeezed)
    # Params: scalar [1], shared species [S], family scalar [G], or [G, S]
    log_pD_arg,        # [1] scalar tensor or Python float
    log_pS_arg,        # [1] scalar tensor or Python float
    family_idx_ptr,    # optional [C] clade -> family id
    # Species children [S] int64
    species_child1_ptr,
    species_child2_ptr,
    # Outputs
    accumulated_rhs_ptr,  # [C, S], direct Pi adjoints updated atomically
    left_transfer_complement_vjp_ptr,     # [n_ws, S]
    right_transfer_complement_vjp_ptr,     # [n_ws, S]
    duplication_parameter_vjp_ptr,         # [n_ws]
    speciation_parameter_vjp_ptr,         # [n_ws]
    grad_log_pD_ptr,      # optional scalar accumulation target
    grad_log_pS_ptr,      # optional scalar accumulation target
    grad_max_transfer_ptr,          # optional scalar/[S] accumulation target
    grad_max_transfer_partial_ptr,  # optional [ceil(n_ws/tile_splits), S] two-stage vector accumulation
    donor_adjoint_ptr,         # optional [2 * n_ws, S] initial Pibar VJP subtree values
    total_donor_adjoint_ptr,          # optional [2 * n_ws] row sums of pibar_ud
    active_donor_side_ptr, # optional [2 * n_ws] exact nonzero donor_adjoint row mask
    max_transfer_ptr,               # optional [S] max transfer mat for Pibar valid_receiver_mass reuse
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
    MAX_TRANSFER_LAYOUT: tl.constexpr,
    GRAD_MAX_TRANSFER_LAYOUT: tl.constexpr,
    ACCUM_PARAM_REDUCTIONS: tl.constexpr,
    ACCUM_MAX_TRANSFER_REDUCTION: tl.constexpr,
    GRAD_MAX_TRANSFER_SCALAR: tl.constexpr,
    GRAD_MAX_TRANSFER_TWO_STAGE: tl.constexpr,
    GRAD_MAX_TRANSFER_TILE_SPLITS: tl.constexpr,
    OUTPUT_DONOR_ADJOINT: tl.constexpr,
    OUTPUT_SIDE_ACTIVE: tl.constexpr,
    SIDE_ACTIVE_THRESHOLD_ENABLED: tl.constexpr,
    SKIP_INACTIVE_PIBAR_OUTPUT_ZERO: tl.constexpr,
    DTYPE: tl.constexpr,
):
    """DTS cross-clade backward with direct accumulation of Pi adjoints.

    It writes direct Pi contributions into accumulated_rhs instead of materializing
    grad_Pi_l/grad_Pi_r and relying on two PyTorch index_add_ calls.
    Transfer-complement adjoints are still staged because they feed the
    complementary-subtree VJP kernel.
    """
    NEG_LARGE: tl.constexpr = -float("inf")

    split_index = tl.program_id(0)

    left_clade_row = tl.load(split_left_rows_ptr + split_index).to(tl.int64)
    right_clade_row = tl.load(split_right_rows_ptr + split_index).to(tl.int64)
    parent_wave_row = tl.load(reduce_idx_ptr + split_index).to(tl.int64)
    split_log_prior = tl.load(log_split_probs_ptr + split_index).to(DTYPE)
    if USE_ACTIVE_MASK:
        parent_active = tl.load(active_mask_ptr + parent_wave_row)
        if parent_active == 0:
            out_base = split_index * S
            left_donor_adjoint_base = split_index * S
            right_donor_adjoint_base = (tl.program_id(0) + 0 + tl.num_programs(0)) * S
            zero_scalar = tl.zeros((1,), dtype=DTYPE)
            scalar_lane_offset = tl.arange(0, 1)
            if not ACCUM_PARAM_REDUCTIONS:
                tl.store(duplication_parameter_vjp_ptr + split_index + scalar_lane_offset, zero_scalar)
                tl.store(speciation_parameter_vjp_ptr + split_index + scalar_lane_offset, zero_scalar)
            if OUTPUT_DONOR_ADJOINT:
                if OUTPUT_SIDE_ACTIVE:
                    tl.store(active_donor_side_ptr + split_index + scalar_lane_offset, 0)
                    tl.store(active_donor_side_ptr + tl.num_programs(0) + split_index + scalar_lane_offset, 0)
                if SKIP_INACTIVE_PIBAR_OUTPUT_ZERO:
                    return
                tl.store(total_donor_adjoint_ptr + split_index + scalar_lane_offset, zero_scalar)
                tl.store(total_donor_adjoint_ptr + tl.num_programs(0) + split_index + scalar_lane_offset, zero_scalar)
            for s_start in range(0, S, BLOCK_S):
                s_offs = s_start + tl.arange(0, BLOCK_S)
                mask = s_offs < S
                zero = tl.zeros([BLOCK_S], dtype=DTYPE)
                if OUTPUT_DONOR_ADJOINT:
                    tl.store(donor_adjoint_ptr + left_donor_adjoint_base + s_offs, zero, mask=mask)
                    tl.store(donor_adjoint_ptr + right_donor_adjoint_base + s_offs, zero, mask=mask)
                else:
                    tl.store(left_transfer_complement_vjp_ptr + out_base + s_offs, zero, mask=mask)
                    tl.store(right_transfer_complement_vjp_ptr + out_base + s_offs, zero, mask=mask)
            return
    else:
        parent_active = True

    parent_clade_row = ws + parent_wave_row
    if (
        PARAM_LAYOUT == 2
        or PARAM_LAYOUT == 3
        or PARAM_GRAD_LAYOUT == 2
        or PARAM_GRAD_LAYOUT == 3
    ):
        parent_family = tl.load(family_idx_ptr + parent_clade_row).to(tl.int64)
    else:
        parent_family = 0

    if MAX_TRANSFER_LAYOUT == 1 or GRAD_MAX_TRANSFER_LAYOUT == 1:
        left_family = tl.load(family_idx_ptr + left_clade_row).to(tl.int64)
        right_family = tl.load(family_idx_ptr + right_clade_row).to(tl.int64)
    else:
        left_family = 0
        right_family = 0

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

    left_clade_base = left_clade_row * stride_C
    right_clade_base = right_clade_row * stride_C
    left_transfer_complement_base = left_clade_row * stride_C
    right_transfer_complement_base = right_clade_row * stride_C
    parent_clade_base = (ws + parent_wave_row) * stride_C
    parent_adjoint_base = parent_wave_row * S
    out_base = split_index * S

    left_pi_offset = tl.load(Pi_offset_ptr + left_clade_row)
    right_pi_offset = tl.load(Pi_offset_ptr + right_clade_row)
    parent_pi_offset = tl.load(Pi_offset_ptr + parent_clade_row)
    left_pibar_offset = tl.load(Pibar_offset_ptr + left_clade_row)
    right_pibar_offset = tl.load(Pibar_offset_ptr + right_clade_row)
    child_pair_frame_shift = (left_pi_offset + right_pi_offset - parent_pi_offset).to(DTYPE)
    left_transfer_frame_shift = (left_pi_offset + right_pibar_offset - parent_pi_offset).to(DTYPE)
    right_transfer_frame_shift = (right_pi_offset + left_pibar_offset - parent_pi_offset).to(DTYPE)
    left_exclusion_frame_shift = (left_pi_offset - left_pibar_offset).to(DTYPE)
    right_exclusion_frame_shift = (right_pi_offset - right_pibar_offset).to(DTYPE)

    duplication_parameter_vjp_sum = tl.zeros((1,), dtype=DTYPE)
    speciation_parameter_vjp_sum = tl.zeros((1,), dtype=DTYPE)
    left_total_donor_adjoint = tl.zeros((1,), dtype=DTYPE)
    right_total_donor_adjoint = tl.zeros((1,), dtype=DTYPE)
    scalar_lane_offset = tl.arange(0, 1)
    if OUTPUT_DONOR_ADJOINT:
        left_pibar_row_max = tl.load(pibar_row_max_ptr + left_clade_row).to(DTYPE)
        right_pibar_row_max = tl.load(pibar_row_max_ptr + right_clade_row).to(DTYPE)
        left_donor_side_nonzero = tl.full((1,), value=0, dtype=tl.int32)
        right_donor_side_nonzero = tl.full((1,), value=0, dtype=tl.int32)
        left_donor_adjoint_abs_sum = tl.zeros((1,), dtype=DTYPE)
        right_donor_adjoint_abs_sum = tl.zeros((1,), dtype=DTYPE)

    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        valid_mask = s_offs < S
        mask = valid_mask & parent_active

        left_reconciliation_log_likelihood = tl.load(Pi_star_ptr + left_clade_base + s_offs, mask=mask, other=NEG_LARGE).to(DTYPE)
        right_reconciliation_log_likelihood = tl.load(Pi_star_ptr + right_clade_base + s_offs, mask=mask, other=NEG_LARGE).to(DTYPE)
        left_transfer_complement_log_likelihood = tl.load(Pibar_star_ptr + left_transfer_complement_base + s_offs, mask=mask, other=NEG_LARGE).to(DTYPE)
        right_transfer_complement_log_likelihood = tl.load(Pibar_star_ptr + right_transfer_complement_base + s_offs, mask=mask, other=NEG_LARGE).to(DTYPE)

        c1 = tl.load(species_child1_ptr + s_offs, mask=mask, other=0)
        c2 = tl.load(species_child2_ptr + s_offs, mask=mask, other=0)
        c1_valid = (c1 < S) & mask
        c2_valid = (c2 < S) & mask
        left_child1_reconciliation_log_likelihood = tl.load(Pi_star_ptr + left_clade_base + c1, mask=c1_valid, other=NEG_LARGE).to(DTYPE)
        left_child2_reconciliation_log_likelihood = tl.load(Pi_star_ptr + left_clade_base + c2, mask=c2_valid, other=NEG_LARGE).to(DTYPE)
        right_child1_reconciliation_log_likelihood = tl.load(Pi_star_ptr + right_clade_base + c1, mask=c1_valid, other=NEG_LARGE).to(DTYPE)
        right_child2_reconciliation_log_likelihood = tl.load(Pi_star_ptr + right_clade_base + c2, mask=c2_valid, other=NEG_LARGE).to(DTYPE)

        parent_reconciliation_log_likelihood = tl.load(Pi_star_ptr + parent_clade_base + s_offs, mask=mask, other=NEG_LARGE).to(DTYPE)
        parent_adjoint = tl.load(v_k_ptr + parent_adjoint_base + s_offs, mask=mask, other=0.0).to(DTYPE)

        if PARAM_LAYOUT == 1:
            duplication_log_probability = tl.load(log_pD_arg + s_offs, mask=valid_mask, other=NEG_LARGE).to(DTYPE)
            speciation_log_probability = tl.load(log_pS_arg + s_offs, mask=valid_mask, other=NEG_LARGE).to(DTYPE)
        elif PARAM_LAYOUT == 3:
            param_base = parent_family * S
            duplication_log_probability = tl.load(log_pD_arg + param_base + s_offs, mask=valid_mask, other=NEG_LARGE).to(DTYPE)
            speciation_log_probability = tl.load(log_pS_arg + param_base + s_offs, mask=valid_mask, other=NEG_LARGE).to(DTYPE)
        else:
            duplication_log_probability = log_pD
            speciation_log_probability = log_pS

        duplication_log_term = duplication_log_probability + left_reconciliation_log_likelihood + right_reconciliation_log_likelihood + child_pair_frame_shift
        transfer_left_retained_log_term = left_reconciliation_log_likelihood + right_transfer_complement_log_likelihood + left_transfer_frame_shift
        transfer_right_retained_log_term = right_reconciliation_log_likelihood + left_transfer_complement_log_likelihood + right_transfer_frame_shift
        speciation_lr_log_term = speciation_log_probability + left_child1_reconciliation_log_likelihood + right_child2_reconciliation_log_likelihood + child_pair_frame_shift
        speciation_rl_log_term = speciation_log_probability + right_child1_reconciliation_log_likelihood + left_child2_reconciliation_log_likelihood + child_pair_frame_shift

        parent_valid = parent_reconciliation_log_likelihood != NEG_LARGE
        duplication_probability = tl.where(
            parent_valid,
            tl.exp2(split_log_prior + duplication_log_term - parent_reconciliation_log_likelihood),
            tl.zeros_like(duplication_log_term),
        )
        transfer_left_retained_probability = tl.where(
            parent_valid,
            tl.exp2(split_log_prior + transfer_left_retained_log_term - parent_reconciliation_log_likelihood),
            tl.zeros_like(transfer_left_retained_log_term),
        )
        transfer_right_retained_probability = tl.where(
            parent_valid,
            tl.exp2(split_log_prior + transfer_right_retained_log_term - parent_reconciliation_log_likelihood),
            tl.zeros_like(transfer_right_retained_log_term),
        )
        speciation_lr_probability = tl.where(
            parent_valid,
            tl.exp2(split_log_prior + speciation_lr_log_term - parent_reconciliation_log_likelihood),
            tl.zeros_like(speciation_lr_log_term),
        )
        speciation_rl_probability = tl.where(
            parent_valid,
            tl.exp2(split_log_prior + speciation_rl_log_term - parent_reconciliation_log_likelihood),
            tl.zeros_like(speciation_rl_log_term),
        )

        duplication_event_vjp = parent_adjoint * duplication_probability
        transfer_left_retained_event_vjp = parent_adjoint * transfer_left_retained_probability
        transfer_right_retained_event_vjp = parent_adjoint * transfer_right_retained_probability
        speciation_lr_event_vjp = parent_adjoint * speciation_lr_probability
        speciation_rl_event_vjp = parent_adjoint * speciation_rl_probability

        left_reconciliation_vjp_ptr = accumulated_rhs_ptr + left_clade_base + s_offs
        right_reconciliation_vjp_ptr = accumulated_rhs_ptr + right_clade_base + s_offs
        if USE_ATOMICS:
            tl.atomic_add(
                left_reconciliation_vjp_ptr,
                duplication_event_vjp + transfer_left_retained_event_vjp,
                sem="relaxed",
                mask=mask,
            )
            tl.atomic_add(
                right_reconciliation_vjp_ptr,
                duplication_event_vjp + transfer_right_retained_event_vjp,
                sem="relaxed",
                mask=mask,
            )
        else:
            left_reconciliation_vjp = tl.load(left_reconciliation_vjp_ptr, mask=mask, other=0.0).to(DTYPE)
            right_reconciliation_vjp = tl.load(right_reconciliation_vjp_ptr, mask=mask, other=0.0).to(DTYPE)
            tl.store(
                left_reconciliation_vjp_ptr,
                left_reconciliation_vjp + duplication_event_vjp + transfer_left_retained_event_vjp,
                mask=mask,
            )
            tl.store(
                right_reconciliation_vjp_ptr,
                right_reconciliation_vjp + duplication_event_vjp + transfer_right_retained_event_vjp,
                mask=mask,
            )
        if OUTPUT_DONOR_ADJOINT:
            if MAX_TRANSFER_LAYOUT == 1:
                left_max_transfer = tl.load(max_transfer_ptr + left_family * S + s_offs, mask=valid_mask, other=0.0).to(DTYPE)
                right_max_transfer = tl.load(max_transfer_ptr + right_family * S + s_offs, mask=valid_mask, other=0.0).to(DTYPE)
            else:
                max_transfer = tl.load(max_transfer_ptr + s_offs, mask=valid_mask, other=0.0).to(DTYPE)
                left_max_transfer = max_transfer
                right_max_transfer = max_transfer
            left_transfer_complement_is_finite = (left_transfer_complement_log_likelihood != NEG_LARGE) & mask
            right_transfer_complement_is_finite = (right_transfer_complement_log_likelihood != NEG_LARGE) & mask
            left_exclusion_scale = tl.where(
                left_transfer_complement_is_finite,
                tl.exp2(left_pibar_row_max + left_max_transfer - left_transfer_complement_log_likelihood + left_exclusion_frame_shift),
                tl.zeros([BLOCK_S], dtype=DTYPE),
            )
            right_exclusion_scale = tl.where(
                right_transfer_complement_is_finite,
                tl.exp2(right_pibar_row_max + right_max_transfer - right_transfer_complement_log_likelihood + right_exclusion_frame_shift),
                tl.zeros([BLOCK_S], dtype=DTYPE),
            )
            left_donor_adjoint = transfer_right_retained_event_vjp * left_exclusion_scale
            right_donor_adjoint = transfer_left_retained_event_vjp * right_exclusion_scale
            tl.store(donor_adjoint_ptr + split_index * S + s_offs, left_donor_adjoint, mask=valid_mask)
            tl.store(donor_adjoint_ptr + (tl.num_programs(0) + split_index) * S + s_offs, right_donor_adjoint, mask=valid_mask)
            left_total_donor_adjoint += tl.sum(tl.where(mask, left_donor_adjoint, 0.0), axis=0)
            right_total_donor_adjoint += tl.sum(tl.where(mask, right_donor_adjoint, 0.0), axis=0)
            if OUTPUT_SIDE_ACTIVE:
                if SIDE_ACTIVE_THRESHOLD_ENABLED:
                    left_donor_adjoint_abs_sum += tl.sum(tl.where(mask, tl.abs(left_donor_adjoint), 0.0), axis=0)
                    right_donor_adjoint_abs_sum += tl.sum(tl.where(mask, tl.abs(right_donor_adjoint), 0.0), axis=0)
                else:
                    left_donor_side_nonzero += tl.where(tl.max(tl.abs(left_donor_adjoint), axis=0) != 0.0, 1, 0)
                    right_donor_side_nonzero += tl.where(tl.max(tl.abs(right_donor_adjoint), axis=0) != 0.0, 1, 0)
        else:
            tl.store(
                left_transfer_complement_vjp_ptr + out_base + s_offs,
                transfer_right_retained_event_vjp,
                mask=valid_mask,
            )
            tl.store(
                right_transfer_complement_vjp_ptr + out_base + s_offs,
                transfer_left_retained_event_vjp,
                mask=valid_mask,
            )

        if ACCUM_PARAM_REDUCTIONS and PARAM_GRAD_LAYOUT == 1:
            tl.atomic_add(
                grad_log_pD_ptr + s_offs,
                duplication_event_vjp,
                sem="relaxed",
                mask=mask,
            )
            tl.atomic_add(
                grad_log_pS_ptr + s_offs,
                speciation_lr_event_vjp + speciation_rl_event_vjp,
                sem="relaxed",
                mask=mask,
            )
        elif ACCUM_PARAM_REDUCTIONS and PARAM_GRAD_LAYOUT == 3:
            grad_param_base = parent_family * S
            tl.atomic_add(grad_log_pD_ptr + grad_param_base + s_offs, duplication_event_vjp, sem="relaxed", mask=mask)
            tl.atomic_add(grad_log_pS_ptr + grad_param_base + s_offs, speciation_lr_event_vjp + speciation_rl_event_vjp, sem="relaxed", mask=mask)
        else:
            duplication_parameter_vjp_sum += tl.sum(duplication_event_vjp, axis=0)
            speciation_parameter_vjp_sum += tl.sum(speciation_lr_event_vjp + speciation_rl_event_vjp, axis=0)
        if ACCUM_MAX_TRANSFER_REDUCTION:
            max_transfer_vjp = transfer_left_retained_event_vjp + transfer_right_retained_event_vjp
            if GRAD_MAX_TRANSFER_LAYOUT == 1:
                tl.atomic_add(
                    grad_max_transfer_ptr + left_family * S + s_offs,
                    transfer_right_retained_event_vjp,
                    sem="relaxed",
                    mask=mask,
                )
                tl.atomic_add(
                    grad_max_transfer_ptr + right_family * S + s_offs,
                    transfer_left_retained_event_vjp,
                    sem="relaxed",
                    mask=mask,
                )
            elif GRAD_MAX_TRANSFER_SCALAR:
                tl.atomic_add(
                    grad_max_transfer_ptr + scalar_lane_offset,
                    tl.sum(tl.where(mask, max_transfer_vjp, 0.0), axis=0),
                    sem="relaxed",
                )
            elif GRAD_MAX_TRANSFER_TWO_STAGE:
                max_transfer_tile = split_index // GRAD_MAX_TRANSFER_TILE_SPLITS
                tl.atomic_add(
                    grad_max_transfer_partial_ptr + max_transfer_tile * S + s_offs,
                    max_transfer_vjp,
                    sem="relaxed",
                    mask=mask,
                )
            else:
                tl.atomic_add(
                    grad_max_transfer_ptr + s_offs,
                    max_transfer_vjp,
                    sem="relaxed",
                    mask=mask,
                )

        if MERGE_S_TERM:
            pi_l_c1_out = accumulated_rhs_ptr + left_clade_base + c1
            pi_r_c1_out = accumulated_rhs_ptr + right_clade_base + c1
            pi_r_c2_out = accumulated_rhs_ptr + right_clade_base + c2
            pi_l_c2_out = accumulated_rhs_ptr + left_clade_base + c2
            if USE_ATOMICS:
                tl.atomic_add(pi_l_c1_out, speciation_lr_event_vjp, sem="relaxed", mask=c1_valid)
                tl.atomic_add(pi_r_c1_out, speciation_rl_event_vjp, sem="relaxed", mask=c1_valid)
                tl.atomic_add(pi_r_c2_out, speciation_lr_event_vjp, sem="relaxed", mask=c2_valid)
                tl.atomic_add(pi_l_c2_out, speciation_rl_event_vjp, sem="relaxed", mask=c2_valid)
            else:
                pi_l_c1_cur = tl.load(pi_l_c1_out, mask=c1_valid, other=0.0)
                pi_r_c1_cur = tl.load(pi_r_c1_out, mask=c1_valid, other=0.0)
                pi_r_c2_cur = tl.load(pi_r_c2_out, mask=c2_valid, other=0.0)
                pi_l_c2_cur = tl.load(pi_l_c2_out, mask=c2_valid, other=0.0)
                tl.store(pi_l_c1_out, pi_l_c1_cur + speciation_lr_event_vjp, mask=c1_valid)
                tl.store(pi_r_c1_out, pi_r_c1_cur + speciation_rl_event_vjp, mask=c1_valid)
                tl.store(pi_r_c2_out, pi_r_c2_cur + speciation_lr_event_vjp, mask=c2_valid)
                tl.store(pi_l_c2_out, pi_l_c2_cur + speciation_rl_event_vjp, mask=c2_valid)

    if ACCUM_PARAM_REDUCTIONS:
        if PARAM_GRAD_LAYOUT == 0:
            tl.atomic_add(grad_log_pD_ptr + scalar_lane_offset, duplication_parameter_vjp_sum, sem="relaxed")
            tl.atomic_add(grad_log_pS_ptr + scalar_lane_offset, speciation_parameter_vjp_sum, sem="relaxed")
        elif PARAM_GRAD_LAYOUT == 2:
            tl.atomic_add(
                grad_log_pD_ptr + parent_family + scalar_lane_offset,
                duplication_parameter_vjp_sum,
                sem="relaxed",
            )
            tl.atomic_add(
                grad_log_pS_ptr + parent_family + scalar_lane_offset,
                speciation_parameter_vjp_sum,
                sem="relaxed",
            )
    else:
        tl.store(duplication_parameter_vjp_ptr + split_index + scalar_lane_offset, duplication_parameter_vjp_sum)
        tl.store(speciation_parameter_vjp_ptr + split_index + scalar_lane_offset, speciation_parameter_vjp_sum)
    if OUTPUT_DONOR_ADJOINT:
        tl.store(total_donor_adjoint_ptr + split_index + scalar_lane_offset, left_total_donor_adjoint)
        tl.store(total_donor_adjoint_ptr + tl.num_programs(0) + split_index + scalar_lane_offset, right_total_donor_adjoint)
        if OUTPUT_SIDE_ACTIVE:
            if SIDE_ACTIVE_THRESHOLD_ENABLED:
                threshold = tl.load(side_active_threshold_ptr).to(DTYPE)
                left_donor_adjoint_bound = left_donor_adjoint_abs_sum
                right_donor_adjoint_bound = right_donor_adjoint_abs_sum
                tl.store(active_donor_side_ptr + split_index + scalar_lane_offset, left_donor_adjoint_bound > threshold)
                tl.store(
                    active_donor_side_ptr + tl.num_programs(0) + split_index + scalar_lane_offset,
                    right_donor_adjoint_bound > threshold,
                )
            else:
                tl.store(active_donor_side_ptr + split_index + scalar_lane_offset, left_donor_side_nonzero != 0)
                tl.store(
                    active_donor_side_ptr + tl.num_programs(0) + split_index + scalar_lane_offset,
                    right_donor_side_nonzero != 0,
                )

    if not MERGE_S_TERM:
        for s_start in range(0, S, BLOCK_S):
            s_offs = s_start + tl.arange(0, BLOCK_S)
            valid_mask = s_offs < S
            mask = valid_mask & parent_active

            c1 = tl.load(species_child1_ptr + s_offs, mask=mask, other=0)
            c2 = tl.load(species_child2_ptr + s_offs, mask=mask, other=0)
            c1_valid = (c1 < S) & mask
            c2_valid = (c2 < S) & mask

            left_child1_reconciliation_log_likelihood = tl.load(Pi_star_ptr + left_clade_base + c1, mask=c1_valid, other=NEG_LARGE).to(DTYPE)
            left_child2_reconciliation_log_likelihood = tl.load(Pi_star_ptr + left_clade_base + c2, mask=c2_valid, other=NEG_LARGE).to(DTYPE)
            right_child1_reconciliation_log_likelihood = tl.load(Pi_star_ptr + right_clade_base + c1, mask=c1_valid, other=NEG_LARGE).to(DTYPE)
            right_child2_reconciliation_log_likelihood = tl.load(Pi_star_ptr + right_clade_base + c2, mask=c2_valid, other=NEG_LARGE).to(DTYPE)

            parent_reconciliation_log_likelihood = tl.load(Pi_star_ptr + parent_clade_base + s_offs, mask=mask, other=NEG_LARGE).to(DTYPE)
            parent_adjoint = tl.load(v_k_ptr + parent_adjoint_base + s_offs, mask=mask, other=0.0).to(DTYPE)

            if PARAM_LAYOUT == 1:
                speciation_log_probability = tl.load(log_pS_arg + s_offs, mask=valid_mask, other=NEG_LARGE).to(DTYPE)
            elif PARAM_LAYOUT == 3:
                speciation_log_probability = tl.load(log_pS_arg + parent_family * S + s_offs, mask=valid_mask, other=NEG_LARGE).to(DTYPE)
            else:
                speciation_log_probability = log_pS

            speciation_lr_log_term = speciation_log_probability + left_child1_reconciliation_log_likelihood + right_child2_reconciliation_log_likelihood + child_pair_frame_shift
            speciation_rl_log_term = speciation_log_probability + right_child1_reconciliation_log_likelihood + left_child2_reconciliation_log_likelihood + child_pair_frame_shift

            parent_valid = parent_reconciliation_log_likelihood != NEG_LARGE
            speciation_lr_probability = tl.where(
                parent_valid,
                tl.exp2(split_log_prior + speciation_lr_log_term - parent_reconciliation_log_likelihood),
                tl.zeros_like(speciation_lr_log_term),
            )
            speciation_rl_probability = tl.where(
                parent_valid,
                tl.exp2(split_log_prior + speciation_rl_log_term - parent_reconciliation_log_likelihood),
                tl.zeros_like(speciation_rl_log_term),
            )
            speciation_lr_event_vjp = parent_adjoint * speciation_lr_probability
            speciation_rl_event_vjp = parent_adjoint * speciation_rl_probability

            pi_l_c1_out = accumulated_rhs_ptr + left_clade_base + c1
            pi_r_c1_out = accumulated_rhs_ptr + right_clade_base + c1
            pi_r_c2_out = accumulated_rhs_ptr + right_clade_base + c2
            pi_l_c2_out = accumulated_rhs_ptr + left_clade_base + c2
            if USE_ATOMICS:
                tl.atomic_add(pi_l_c1_out, speciation_lr_event_vjp, sem="relaxed", mask=c1_valid)
                tl.atomic_add(pi_r_c1_out, speciation_rl_event_vjp, sem="relaxed", mask=c1_valid)
                tl.atomic_add(pi_r_c2_out, speciation_lr_event_vjp, sem="relaxed", mask=c2_valid)
                tl.atomic_add(pi_l_c2_out, speciation_rl_event_vjp, sem="relaxed", mask=c2_valid)
            else:
                pi_l_c1_cur = tl.load(pi_l_c1_out, mask=c1_valid, other=0.0).to(DTYPE)
                pi_r_c1_cur = tl.load(pi_r_c1_out, mask=c1_valid, other=0.0).to(DTYPE)
                pi_r_c2_cur = tl.load(pi_r_c2_out, mask=c2_valid, other=0.0).to(DTYPE)
                pi_l_c2_cur = tl.load(pi_l_c2_out, mask=c2_valid, other=0.0).to(DTYPE)
                tl.store(pi_l_c1_out, pi_l_c1_cur + speciation_lr_event_vjp, mask=c1_valid)
                tl.store(pi_r_c1_out, pi_r_c1_cur + speciation_rl_event_vjp, mask=c1_valid)
                tl.store(pi_r_c2_out, pi_r_c2_cur + speciation_lr_event_vjp, mask=c2_valid)
                tl.store(pi_l_c2_out, pi_l_c2_cur + speciation_rl_event_vjp, mask=c2_valid)


@triton.jit
def _reduce_max_transfer_vjp_kernel(
    partial_ptr,   # [n_tiles, S]
    grad_max_transfer_ptr,   # [S]
    n_tiles: tl.constexpr,
    S: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
    BLOCK_S: tl.constexpr,
    DTYPE: tl.constexpr,
):
    """Reduce split-tile transfer-parameter VJP partials by species."""
    s_block = tl.program_id(0)
    s_offs = s_block * BLOCK_S + tl.arange(0, BLOCK_S)
    valid_s = s_offs < S
    max_transfer_vjp_sum = tl.zeros([BLOCK_S], dtype=DTYPE)

    tile_start = 0
    while tile_start < n_tiles:
        tile_offs = tile_start + tl.arange(0, BLOCK_TILES)
        mask = (tile_offs[:, None] < n_tiles) & valid_s[None, :]
        max_transfer_vjp_partial = tl.load(
            partial_ptr + tile_offs[:, None] * S + s_offs[None, :],
            mask=mask,
            other=0.0,
        )
        max_transfer_vjp_sum += tl.sum(max_transfer_vjp_partial, axis=0)
        tile_start += BLOCK_TILES

    current = tl.load(grad_max_transfer_ptr + s_offs, mask=valid_s, other=0.0)
    tl.store(
        grad_max_transfer_ptr + s_offs,
        current + max_transfer_vjp_sum,
        mask=valid_s,
    )


@triton.jit
def _select_active_transfer_donor_sides_kernel(
    donor_adjoint_ptr,        # [n_rows, S]
    side_active_ptr,     # [n_rows] bool
    side_active_threshold_ptr,
    S: tl.constexpr,
    BLOCK_S: tl.constexpr,
    SIDE_ACTIVE_THRESHOLD_ENABLED: tl.constexpr,
    DTYPE: tl.constexpr,
):
    """Mark split-side rows whose staged donor_adjoint should run Pibar tree work."""
    row = tl.program_id(0)
    row_base = row * S
    row_absmax = tl.full([1], value=0.0, dtype=DTYPE)
    row_abssum = tl.full([1], value=0.0, dtype=DTYPE)

    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        mask = s_offs < S
        donor_adjoint = tl.load(
            donor_adjoint_ptr + row_base + s_offs, mask=mask, other=0.0
        )
        donor_adjoint_magnitude = tl.abs(donor_adjoint)
        row_absmax = tl.maximum(
            row_absmax, tl.max(donor_adjoint_magnitude, axis=0)
        )
        row_abssum += tl.sum(
            tl.where(mask, donor_adjoint_magnitude, 0.0), axis=0
        )

    lane = tl.arange(0, 1)
    if SIDE_ACTIVE_THRESHOLD_ENABLED:
        threshold = tl.load(side_active_threshold_ptr).to(DTYPE)
        tl.store(side_active_ptr + row + lane, row_abssum > threshold)
    else:
        tl.store(side_active_ptr + row + lane, row_absmax != 0.0)


@triton.jit
def _accumulate_transfer_subtree_vjp_kernel(
    Pi_star_ptr,          # [C, S]
    receiver_log_probs_ptr, # [S]
    donor_adjoint_ptr,         # [2 * n_ws, S], initial subtree values donor_adjoint
    total_donor_adjoint_ptr,          # [2 * n_ws], sum_s donor_adjoint[s] per split side
    side_active_ptr,      # optional [2 * n_ws] bool exact-zero side skip mask
    split_left_rows_ptr,               # [n_ws]
    split_right_rows_ptr,               # [n_ws]
    reduce_idx_ptr,       # [n_ws], used with active_mask_ptr when enabled
    active_mask_ptr,      # optional [W] bool parent row activity mask
    pibar_row_max_ptr,    # [C], Pi-row max from forward uniform Pibar
    compact_level_ptr,    # [N_LEVELS + 1]
    compact_level_parent_ptr, # [total internal nodes across levels]
    compact_level_child1_ptr, # [total internal nodes across levels]
    compact_level_child2_ptr, # [total internal nodes across levels]
    accumulated_rhs_ptr,  # [C, S], updated atomically
    grad_receiver_log_probs_ptr, # optional [S], updated atomically
    n_ws: tl.constexpr,
    S: tl.constexpr,
    stride_C: tl.constexpr,
    BLOCK_S: tl.constexpr,
    N_LEVELS: tl.constexpr,
    USE_ACTIVE_MASK: tl.constexpr,
    USE_SIDE_ACTIVE: tl.constexpr,
    ACCUM_RECEIVER_GRAD: tl.constexpr,
    USE_RECEIVER_WEIGHTS: tl.constexpr,
    DTYPE: tl.constexpr,
):
    """Apply the transfer-complement VJP using compact subtree reductions."""
    NEG_LARGE: tl.constexpr = -float("inf")

    row = tl.program_id(0)
    split_i = tl.where(row < n_ws, row, row - n_ws)
    is_right = row >= n_ws
    if USE_SIDE_ACTIVE:
        side_active = tl.load(side_active_ptr + row)
        if side_active == 0:
            return

    child_l = tl.load(split_left_rows_ptr + split_i).to(tl.int64)
    child_r = tl.load(split_right_rows_ptr + split_i).to(tl.int64)
    child = tl.where(is_right, child_r, child_l)
    if USE_ACTIVE_MASK:
        parent_wave_row = tl.load(reduce_idx_ptr + split_i).to(tl.int64)
        row_active = tl.load(active_mask_ptr + parent_wave_row)
        if row_active == 0:
            return
    else:
        row_active = True

    pi_base = child * stride_C
    row_base = row * S
    row_max = tl.load(pibar_row_max_ptr + child).to(DTYPE)
    row_max_safe = tl.where(row_max != NEG_LARGE, row_max, tl.zeros_like(row_max))
    total_donor_adjoint = tl.load(total_donor_adjoint_ptr + row).to(DTYPE)

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

            parent_val = tl.load(donor_adjoint_ptr + row_base + parent, mask=parent_valid, other=0.0)
            c1_val = tl.load(donor_adjoint_ptr + row_base + c1, mask=c1_valid, other=0.0)
            c2_val = tl.load(donor_adjoint_ptr + row_base + c2, mask=c2_valid, other=0.0)
            tl.store(donor_adjoint_ptr + row_base + parent, parent_val + c1_val + c2_val, mask=parent_valid)
            p_start += BLOCK_S
        tl.debug_barrier()

    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        valid_mask = s_offs < S
        mask = valid_mask & row_active
        pi_val = tl.load(Pi_star_ptr + pi_base + s_offs, mask=mask, other=NEG_LARGE)
        if USE_RECEIVER_WEIGHTS:
            receiver_log_probability = tl.load(receiver_log_probs_ptr + s_offs, mask=valid_mask, other=NEG_LARGE)
            receiver_mass = tl.exp2(receiver_log_probability + pi_val - row_max_safe)
        else:
            receiver_mass = tl.exp2(pi_val - row_max_safe)
        subtree_donor_adjoint = tl.load(
            donor_adjoint_ptr + row_base + s_offs, mask=mask, other=0.0
        )
        transfer_complement_vjp = receiver_mass * (
            total_donor_adjoint - subtree_donor_adjoint
        )
        tl.atomic_add(accumulated_rhs_ptr + pi_base + s_offs, transfer_complement_vjp, sem="relaxed", mask=mask)
        if ACCUM_RECEIVER_GRAD:
            tl.atomic_add(
                grad_receiver_log_probs_ptr + s_offs,
                transfer_complement_vjp,
                sem="relaxed",
                mask=mask,
            )
