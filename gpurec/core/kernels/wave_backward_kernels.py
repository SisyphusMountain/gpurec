"""Triton @triton.jit device kernels for the wave-step VJP (backward pass).

Split out of wave_backward.py (mechanical AST extraction, 2026-07-07): these
are the compiled device kernels. The host-side orchestration that launches
them lives in wave_backward.py, which re-imports every name defined here so
callers and tests keep reaching them as ``wave_backward.<name>``.
"""
import torch
import triton
import triton.language as tl

# The species-tree neighbourhood gathers a register-resident, one-program-per-row kernel needs;
# shared verbatim with the wave tangent and E-step kernels so every path walks the same tree.
from gpurec.core.kernels.species_tree_sums import species_neighbourhood

# ``rhs`` is the wave's slice of the [clades, species] adjoint buffer, so its 16-byte
# alignment changes with the wave start; specializing on it recompiles the kernel for
# half the waves (see README.md).
@triton.jit(do_not_specialize_on_alignment=["rhs_ptr"])
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
    # int64: w ranges over the whole batch's clade rows, so row_base below can
    # overflow int32 once total_clades * S exceeds 2^31.
    w = tl.program_id(0).to(tl.int64)
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


# ``ws``/``W`` are the wave's start row and width, and ``rhs`` is the wave's slice of the
# [clades, species] adjoint buffer, so its 16-byte alignment changes with the wave start.
# All three would otherwise recompile the kernel per wave (see README.md).
@triton.jit(do_not_specialize=["ws", "W"], do_not_specialize_on_alignment=["rhs_ptr"])
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
    species_child1_ptr, species_child2_ptr,
    not_open_source_ptr, closed_source_ptr, not_open_index_ptr, closed_index_ptr,
    valid_receiver_scratch_ptr,
    leaf_term_ptr,
    leaf_species_ptr,
    leaf_logp_ptr,
    leaf_fm_log_ptr,
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
    USE_LEAF_INDEX: tl.constexpr,
    HAS_LEAF_TERM: tl.constexpr,
    LEAF_LOGP_MODE: tl.constexpr,
    USE_FRACTION_MISSING: tl.constexpr,
    USE_ACTIVE_MASK: tl.constexpr,
    SKIP_INACTIVE_SCRATCH_ZERO: tl.constexpr,
    CONST_LAYOUT: tl.constexpr,
    DTYPE: tl.constexpr,
    USE_CHILD_EDGE_SELF_LOOP: tl.constexpr,
    USE_RECEIVER_WEIGHTS: tl.constexpr,
):
    """Precompute self-loop J^T coefficients for a block of rows and all species."""
    NEG_LARGE: tl.constexpr = -float("inf")

    # int64: rows range over the whole batch's clade rows, so the *stride/*S
    # address arithmetic below can overflow int32 once total_clades * S
    # exceeds 2^31.
    block = tl.program_id(0).to(tl.int64)
    rows = block * BLOCK_W + tl.arange(0, BLOCK_W)
    s_offs = tl.arange(0, BLOCK_S)
    row_valid = rows < W
    species_valid = s_offs < S
    if USE_ACTIVE_MASK:
        row_active = tl.load(active_mask_ptr + rows, mask=row_valid, other=0) != 0
    else:
        row_active = row_valid
    row_mask = row_valid & row_active
    if SKIP_INACTIVE_SCRATCH_ZERO:
        # Nothing to write, so nothing to compute. On the exact adjoint path this kernel is
        # launched only for the rows the elimination could not take -- almost always none of them
        # -- so without this every launch would still read the whole primal row and walk the
        # species tree with every store masked off. Only legal when the inactive rows are not
        # supposed to be written: with SKIP_INACTIVE_SCRATCH_ZERO off the caller is asking this
        # kernel to zero them, which returning early would skip.
        if tl.sum(row_mask.to(tl.int32), axis=0) == 0:
            return
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

    row_max = tl.load(Pibar_row_max_ptr + row_global, mask=row_valid, other=NEG_LARGE)
    reconciliation_log_likelihood = tl.load(Pi_star_ptr + pi_offsets, mask=mask, other=NEG_LARGE)
    transfer_complement_log_likelihood = tl.load(Pibar_star_ptr + pi_offsets, mask=mask, other=NEG_LARGE)
    row_max_safe = tl.where(row_max != NEG_LARGE, row_max, tl.zeros_like(row_max))
    if USE_RECEIVER_WEIGHTS:
        receiver_log_probability = tl.load(receiver_log_probs_ptr + s_offs, mask=species_valid, other=NEG_LARGE)
        receiver_mass = tl.exp2(receiver_log_probability[:, None] + reconciliation_log_likelihood - row_max_safe[None, :])
    else:
        receiver_mass = tl.exp2(reconciliation_log_likelihood - row_max_safe[None, :])
    # Publish the receiver-mass row now: the two running sums below gather it at species the
    # host's scan orders name, i.e. at lanes other warps of this block hold. There is no row-wide
    # reduction here any more -- the total the old subtraction needed is what has gone away.
    zero = tl.zeros([BLOCK_S, BLOCK_W], dtype=DTYPE)
    tl.store(receiver_mass_ptr + out_offsets, tl.where(mask, receiver_mass, zero), mask=store_mask)

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
    duplication_loss_const = tl.load(duplication_loss_const_ptr + const_offsets, mask=const_mask, other=NEG_LARGE)
    extinction_complement_log_probability = tl.load(
        Ebar_ptr + const_offsets, mask=const_mask, other=NEG_LARGE
    )
    extinction_log_probability = tl.load(
        E_ptr + const_offsets, mask=const_mask, other=NEG_LARGE
    )
    speciation_child1_const = tl.load(speciation_child1_const_ptr + const_offsets, mask=const_mask, other=NEG_LARGE)
    speciation_child2_const = tl.load(speciation_child2_const_ptr + const_offsets, mask=const_mask, other=NEG_LARGE)

    c1 = tl.load(species_child1_ptr + s_offs, mask=species_valid, other=0)
    c2 = tl.load(species_child2_ptr + s_offs, mask=species_valid, other=0)
    c1_valid = c1 < S
    c2_valid = c2 < S
    reconciliation_child1_log_likelihood = tl.load(
        Pi_star_ptr + row_global[None, :] * stride + c1[:, None],
        mask=(species_valid & c1_valid)[:, None] & row_mask[None, :],
        other=NEG_LARGE,
    )
    reconciliation_child2_log_likelihood = tl.load(
        Pi_star_ptr + row_global[None, :] * stride + c2[:, None],
        mask=(species_valid & c2_valid)[:, None] & row_mask[None, :],
        other=NEG_LARGE,
    )

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
            leaf_logp = tl.load(leaf_logp_ptr)
            leaf_observation_log_term = tl.where(leaf_hit, leaf_logp, NEG_LARGE)
        elif LEAF_LOGP_MODE == 1:
            leaf_logp = tl.load(leaf_logp_ptr + family, mask=row_valid, other=NEG_LARGE)
            leaf_observation_log_term = tl.where(leaf_hit, leaf_logp[None, :], NEG_LARGE)
        elif LEAF_LOGP_MODE == 2:
            if USE_FRACTION_MISSING:
                # Off-hit leaf-species columns carry the "present-but-unobserved"
                # baseline log_pS[s] + log2(fm_s); non-leaf/observed columns stay
                # -inf (fm_col is -inf there). Mirrors the Pi forward.
                leaf_logp = tl.load(
                    leaf_logp_ptr + const_base[None, :] + s_offs[:, None],
                    mask=mask,
                    other=NEG_LARGE,
                )
                fm_col = tl.load(leaf_fm_log_ptr + s_offs, mask=species_valid, other=NEG_LARGE)
                baseline = leaf_logp + fm_col[:, None]
                leaf_observation_log_term = tl.where(leaf_hit, leaf_logp, baseline)
            else:
                leaf_logp = tl.load(
                    leaf_logp_ptr + const_base[None, :] + s_offs[:, None],
                    mask=leaf_hit,
                    other=NEG_LARGE,
                )
                leaf_observation_log_term = tl.where(leaf_hit, leaf_logp, NEG_LARGE)
        else:
            leaf_logp = tl.load(leaf_logp_ptr + s_offs, mask=species_valid, other=NEG_LARGE)
            if USE_FRACTION_MISSING:
                fm_col = tl.load(leaf_fm_log_ptr + s_offs, mask=species_valid, other=NEG_LARGE)
                baseline = (leaf_logp + fm_col)[:, None]
                leaf_observation_log_term = tl.where(leaf_hit, leaf_logp[:, None], baseline)
            else:
                leaf_observation_log_term = tl.where(leaf_hit, leaf_logp[:, None], NEG_LARGE)
    elif HAS_LEAF_TERM:
        leaf_observation_log_term = tl.load(leaf_term_ptr + out_offsets, mask=mask, other=NEG_LARGE)
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
        gene_split_log_likelihood = tl.load(gene_split_log_likelihood_ptr + out_offsets, mask=mask, other=NEG_LARGE)
        gene_split_log_likelihood += gene_split_frame_shift[None, :]
        local_event_log_likelihood = tl.log2(local_event_scaled_mass) + local_event_max
        updated_reconciliation_max = tl.maximum(local_event_log_likelihood, gene_split_log_likelihood)
        updated_reconciliation_max_safe = tl.where(updated_reconciliation_max != NEG_LARGE, updated_reconciliation_max, tl.zeros_like(updated_reconciliation_max))
        updated_reconciliation_log_likelihood = tl.log2(tl.exp2(local_event_log_likelihood - updated_reconciliation_max_safe) + tl.exp2(gene_split_log_likelihood - updated_reconciliation_max_safe)) + updated_reconciliation_max
        within_wave_probability = tl.where(local_event_log_likelihood != NEG_LARGE, tl.exp2(local_event_log_likelihood - updated_reconciliation_log_likelihood), tl.zeros_like(local_event_log_likelihood))
    else:
        within_wave_probability = tl.full([BLOCK_S, BLOCK_W], value=1.0, dtype=DTYPE)

    # A donor may transfer to every species that is neither itself nor one of its ancestors. This
    # used to be ``total row mass - mass on the donor's lineage``, which walked 34 ancestors per
    # lane and then subtracted two nearly equal numbers: at a high transfer rate the row's mass
    # sits on the donor's own lineage, the two agree past float32's 24 bits and the difference is
    # noise. Built additively instead, exactly as the forward self-loop builds it -- see
    # :func:`gpurec.core.valid_receivers.valid_receiver_index_tables` for the depth-first interval
    # argument and :func:`gpurec.core.kernels.pi_forward._write_valid_receiver_prefix_sums` for the
    # same two passes on the forward side. The allowed recipients split into the species whose
    # subtree has not opened yet and those whose subtree already closed, two disjoint groups whose
    # masses are running sums of non-negative terms and so cannot cancel.
    tl.debug_barrier()
    scratch_row_base = rows[None, :] * S
    # Distance between the two running-sum slots, in elements. int64 for the same overflow reason
    # as ``out_offsets``, and Triton hands ``W`` in as a plain Python int when it specializes the
    # argument, so widen an int64 value we already have and add W to that.
    prefix_slot_stride = (block * 0 + W) * S
    for pass_id in tl.static_range(2):
        if pass_id == 0:
            source_ptr = not_open_source_ptr
            output_base = valid_receiver_scratch_ptr
        else:
            source_ptr = closed_source_ptr
            output_base = valid_receiver_scratch_ptr + prefix_slot_stride
        scan_species = tl.load(source_ptr + s_offs, mask=species_valid, other=S)
        # ``S`` is the one-position shift's sentinel: it contributes nothing.
        contributes = species_valid & (scan_species < S)
        scan_value = tl.load(
            receiver_mass_ptr + scratch_row_base + scan_species[:, None],
            mask=contributes[:, None] & row_mask[None, :],
            other=0.0,
        )
        tl.store(
            output_base + scratch_row_base + s_offs[:, None],
            tl.cumsum(tl.where(contributes[:, None], scan_value, zero), axis=0),
            mask=species_valid[:, None] & row_valid[None, :],
        )
    # The lookups below read lanes other warps in this block just wrote.
    tl.debug_barrier()
    not_open_index = tl.load(not_open_index_ptr + s_offs, mask=species_valid, other=0).to(tl.int64)
    closed_index = tl.load(closed_index_ptr + s_offs, mask=species_valid, other=0).to(tl.int64)
    not_yet_open = tl.load(
        valid_receiver_scratch_ptr + scratch_row_base + not_open_index[:, None],
        mask=mask,
        other=0.0,
    )
    already_closed = tl.load(
        valid_receiver_scratch_ptr + prefix_slot_stride + scratch_row_base + closed_index[:, None],
        mask=mask,
        other=0.0,
    )
    valid_receiver_mass = not_yet_open + already_closed
    inverse_valid_receiver_mass = tl.where(valid_receiver_mass > 0.0, 1.0 / valid_receiver_mass, tl.zeros_like(valid_receiver_mass))

    self_loop_diagonal = within_wave_probability * (duplication_loss_mass + transfer_loss_mass) * inverse_local_event_scaled_mass
    donor_adjoint_coefficient = within_wave_probability * transfer_mass * inverse_local_event_scaled_mass * inverse_valid_receiver_mass
    speciation_child1_probability = within_wave_probability * speciation_child1_mass * inverse_local_event_scaled_mass
    speciation_child2_probability = within_wave_probability * speciation_child2_mass * inverse_local_event_scaled_mass

    rhs_val = tl.load(rhs_ptr + out_offsets, mask=mask, other=0.0)
    tl.store(v_k_ptr + out_offsets, tl.where(mask, rhs_val, zero), mask=store_mask)
    tl.store(self_loop_diagonal_ptr + out_offsets, tl.where(mask, self_loop_diagonal, zero), mask=store_mask)
    tl.store(donor_adjoint_coefficient_ptr + out_offsets, tl.where(mask, donor_adjoint_coefficient, zero), mask=store_mask)
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
def _reconciliation_self_loop_transpose_term(
    term_in_ptr,
    term_out_ptr,
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
    rows,
    s_offs,
    offsets,
    mask,
    store_mask,
    row_mask,
    species_valid,
    S: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_NODES: tl.constexpr,
    N_LEVELS: tl.constexpr,
    DTYPE: tl.constexpr,
    USE_CHILD_EDGE_SELF_LOOP: tl.constexpr,
):
    """One J^T application on this program's row block.

    Shared verbatim by the single-term kernel and by the fused Neumann-series
    kernel so both do the same arithmetic in the same order. Returns the tile
    read from ``term_in_ptr`` and the tile ``J^T @ that``.
    """
    input_adjoint = tl.load(term_in_ptr + offsets, mask=mask, other=0.0)
    donor_adjoint_coefficient = tl.load(donor_adjoint_coefficient_ptr + offsets, mask=mask, other=0.0)
    donor_adjoint = input_adjoint * donor_adjoint_coefficient
    tl.store(subtree_donor_adjoint_ptr + offsets, tl.where(mask, donor_adjoint, tl.zeros_like(donor_adjoint)), mask=store_mask)

    tl.debug_barrier()

    # A donor t's transfer term collects g[s] v[s] over every receiver s OUTSIDE t's subtree
    # (s is neither t nor a descendant of t). That used to be ``row total - subtree sum``. The
    # receiver coefficient g[s] divides by s's own valid receiver mass, so for a species hanging
    # under the lane that holds the row's mass it is astronomically large (2**depth); the row
    # total is then dominated by those terms, and for the dominant lane itself -- whose subtree
    # holds all of them -- the subtraction cancels to rounding noise of that astronomical size.
    # Measured on a 1007-species Coleman family at the loss-rate cap, the gradient came out 1e8
    # times too large. So the off-subtree sum is built top-down by ADDITION instead: what lies
    # off a child's subtree is what lies off its parent's, plus the parent's own term, plus the
    # sibling's whole subtree. Subtree sums first (bottom-up), then that walk (top-down), each
    # child's off-subtree sum overwriting its no-longer-needed subtree sum.
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
            )
            c1_val = tl.load(
                subtree_donor_adjoint_ptr + row_base + c1[:, None],
                mask=reduce_mask & (c1 < S)[:, None],
                other=0.0,
            )
            c2_val = tl.load(
                subtree_donor_adjoint_ptr + row_base + c2[:, None],
                mask=reduce_mask & (c2 < S)[:, None],
                other=0.0,
            )
            tl.store(
                subtree_donor_adjoint_ptr + row_base + parent[:, None],
                parent_val + c1_val + c2_val,
                mask=reduce_mask,
            )
            node_start += BLOCK_NODES
        tl.debug_barrier()

    # The root's chain is empty: nothing lies off it. Every other lane is written by its parent.
    species_parent_of_lane = tl.load(species_parent_ptr + s_offs, mask=species_valid, other=0)
    is_root_lane = (species_parent_of_lane < 0)[:, None] & mask
    tl.store(
        subtree_donor_adjoint_ptr + offsets,
        tl.zeros([BLOCK_S, BLOCK_W], dtype=DTYPE),
        mask=is_root_lane,
    )
    tl.debug_barrier()
    for level_index in range(0, N_LEVELS):
        level = N_LEVELS - 1 - level_index
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
            c1_mask = reduce_mask & (c1 < S)[:, None]
            c2_mask = reduce_mask & (c2 < S)[:, None]
            row_base = rows[None, :] * S
            parent_off_subtree = tl.load(
                subtree_donor_adjoint_ptr + row_base + parent[:, None],
                mask=reduce_mask,
                other=0.0,
            )
            parent_own = tl.load(
                donor_adjoint_coefficient_ptr + row_base + parent[:, None],
                mask=reduce_mask,
                other=0.0,
            ) * tl.load(term_in_ptr + row_base + parent[:, None], mask=reduce_mask, other=0.0)
            c1_subtree = tl.load(
                subtree_donor_adjoint_ptr + row_base + c1[:, None], mask=c1_mask, other=0.0
            )
            c2_subtree = tl.load(
                subtree_donor_adjoint_ptr + row_base + c2[:, None], mask=c2_mask, other=0.0
            )
            tl.store(
                subtree_donor_adjoint_ptr + row_base + c1[:, None],
                parent_off_subtree + parent_own + c2_subtree,
                mask=c1_mask,
            )
            tl.store(
                subtree_donor_adjoint_ptr + row_base + c2[:, None],
                parent_off_subtree + parent_own + c1_subtree,
                mask=c2_mask,
            )
            node_start += BLOCK_NODES
        tl.debug_barrier()

    tl.debug_barrier()

    off_subtree_donor_adjoint = tl.load(subtree_donor_adjoint_ptr + offsets, mask=mask, other=0.0)
    self_loop_diagonal = tl.load(self_loop_diagonal_ptr + offsets, mask=mask, other=0.0)
    receiver_mass = tl.load(receiver_mass_ptr + offsets, mask=mask, other=0.0)
    self_loop_vjp_without_child_edges = (
        input_adjoint * self_loop_diagonal + receiver_mass * off_subtree_donor_adjoint
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
        )
        speciation_parent_to_child_probability = tl.load(
            speciation_child1_probability_ptr + offsets,
            mask=parent_mask,
            other=0.0,
        )
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
        speciation_child1_probability = tl.load(speciation_child1_probability_ptr + offsets, mask=mask, other=0.0)
        speciation_child2_probability = tl.load(speciation_child2_probability_ptr + offsets, mask=mask, other=0.0)
        row_base = rows[None, :] * S
        c1_mask = (species_valid & (c1 < S))[:, None] & row_mask[None, :]
        c2_mask = (species_valid & (c2 < S))[:, None] & row_mask[None, :]
        current_child1_vjp = tl.load(
            term_out_ptr + row_base + c1[:, None],
            mask=c1_mask,
            other=0.0,
        )
        current_child2_vjp = tl.load(
            term_out_ptr + row_base + c2[:, None],
            mask=c2_mask,
            other=0.0,
        )
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
        )

    return input_adjoint, self_loop_vjp


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
    # int64: rows range over the whole batch's clade rows, so the *S address
    # arithmetic below can overflow int32 once total_clades * S exceeds 2^31.
    block = tl.program_id(0).to(tl.int64)
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

    input_adjoint, self_loop_vjp = _reconciliation_self_loop_transpose_term(
        term_in_ptr,
        term_out_ptr,
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
        rows,
        s_offs,
        offsets,
        mask,
        store_mask,
        row_mask,
        species_valid,
        S,
        BLOCK_W,
        BLOCK_S,
        BLOCK_NODES,
        N_LEVELS,
        DTYPE,
        USE_CHILD_EDGE_SELF_LOOP,
    )

    operator_output = (
        input_adjoint - self_loop_vjp if OUTPUT_A else self_loop_vjp
    )
    tl.store(
        term_out_ptr + offsets,
        tl.where(mask, operator_output, tl.zeros_like(operator_output)),
        mask=store_mask,
    )

    if FIXED_POINT_UPDATE:
        rhs_val = tl.load(rhs_update_ptr + offsets, mask=mask, other=0.0)
        tl.store(
            v_k_ptr + offsets,
            tl.where(mask, rhs_val + self_loop_vjp, tl.zeros_like(self_loop_vjp)),
            mask=store_mask,
        )
    elif ACCUMULATE_V:
        v_prev = tl.load(v_k_ptr + offsets, mask=mask, other=0.0)
        tl.store(v_k_ptr + offsets, v_prev + self_loop_vjp, mask=mask)


# ``W`` is the wave's width, and ``rhs`` is the wave's slice of the [clades, species]
# adjoint buffer, so its 16-byte alignment changes with the wave start. Both would
# otherwise recompile the kernel per wave (see README.md).
@triton.jit(do_not_specialize=["W"], do_not_specialize_on_alignment=["rhs_ptr"])
def _reconciliation_self_loop_transpose_series_kernel(
    rhs_ptr,
    term_pair_ptr,
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
    terms_taken_ptr,
    term_tol,
    max_terms,
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
    COLLECT_TERMS: tl.constexpr,
    WRITE_LAST_TERM: tl.constexpr,
):
    """Whole Neumann series for one row block in a single launch.

    Every term calls the same device function the single-term kernel calls, so
    the arithmetic and its per-element order are unchanged; only the loop moved
    from the host into the kernel. A program stops as soon as its row block's
    largest remaining term is at or below ``term_tol * (block max |v_k|)``, or
    after ``max_terms`` terms.

    The test is RELATIVE to the block's own adjoint, with no absolute floor. An
    earlier version used ``term_tol * max(1, block max |v_k|)``; measured on the
    40-family Coleman fit that floor turned the test absolute for the many rows
    whose adjoint is far below 1, dropped terms those rows still needed, and cost
    the Newton loop 42 steps instead of 25 (same final NLL, but 4 families left
    uncertified). Purely relative, the dropped tail is a fixed fraction of each
    row's own adjoint whatever its scale.

    With ``term_tol == 0`` the test can only fire on an exactly-zero term, which
    adds nothing to ``v_k`` and (the operator being linear) makes every later term
    zero too, so the result is bit-identical to ``max_terms`` separate launches of
    the single-term kernel.

    ``term_pair_ptr`` points at a ``[2, W, S]`` scratch holding the two term
    buffers. The ping-pong is an integer offset of 0 or ``W * S`` added to that one
    base pointer, because Triton can carry an integer through a ``while`` loop but
    not a swapped pointer.
    """
    # int64: rows range over the whole batch's clade rows, so the *S address
    # arithmetic below can overflow int32 once total_clades * S exceeds 2^31.
    block = tl.program_id(0).to(tl.int64)
    rows = block * BLOCK_W + tl.arange(0, BLOCK_W)
    s_offs = tl.arange(0, BLOCK_S)
    row_valid = rows < W
    species_valid = s_offs < S
    if USE_ACTIVE_MASK:
        row_active = tl.load(active_mask_ptr + rows, mask=row_valid, other=0) != 0
    else:
        row_active = row_valid
    row_mask = row_valid & row_active
    if SKIP_INACTIVE_SCRATCH_ZERO:
        # Nothing to write, so nothing to compute. Without this the series still walks the whole
        # species tree once per program with every load and store masked off -- and on the exact
        # adjoint path that is EVERY launch, because the elimination almost never spills a row:
        # 66 ms of one 200-family Coleman gradient (2.7 %) spent producing nothing. Only legal
        # when the inactive rows are not supposed to be written: with SKIP_INACTIVE_SCRATCH_ZERO
        # off the caller is asking this kernel to zero them, which returning early would skip.
        if tl.sum(row_mask.to(tl.int32), axis=0) == 0:
            return
    mask = species_valid[:, None] & row_mask[None, :]
    if SKIP_INACTIVE_SCRATCH_ZERO:
        store_mask = mask
    else:
        store_mask = species_valid[:, None] & row_valid[None, :]
    offsets = rows[None, :] * S + s_offs[:, None]
    # Distance between the two term buffers, in elements. It must be int64 for the
    # same overflow reason as `offsets` above, and Triton hands `W` in as a plain
    # Python int when it specializes the argument (no `.to()` on it), so widen an
    # int64 value we already have and add W to that.
    zero_offset = block * 0
    buffer_stride = (zero_offset + W) * S

    n_taken = 0
    out_off = zero_offset
    if FIXED_POINT_UPDATE:
        # Warm branch: v_k is both the iterate and the operator input, so there is
        # no ping-pong -- the first term buffer only holds the raw operator output.
        running = 1
        while running == 1:
            input_adjoint, self_loop_vjp = _reconciliation_self_loop_transpose_term(
                v_k_ptr,
                term_pair_ptr,
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
                rows,
                s_offs,
                offsets,
                mask,
                store_mask,
                row_mask,
                species_valid,
                S,
                BLOCK_W,
                BLOCK_S,
                BLOCK_NODES,
                N_LEVELS,
                DTYPE,
                USE_CHILD_EDGE_SELF_LOOP,
            )
            tl.store(
                term_pair_ptr + offsets,
                tl.where(mask, self_loop_vjp, tl.zeros_like(self_loop_vjp)),
                mask=store_mask,
            )
            rhs_val = tl.load(rhs_ptr + offsets, mask=mask, other=0.0)
            v_new = rhs_val + self_loop_vjp
            tl.store(
                v_k_ptr + offsets,
                tl.where(mask, v_new, tl.zeros_like(self_loop_vjp)),
                mask=store_mask,
            )
            # Warm-branch analogue of a Neumann term: how far the fixed-point
            # iterate moved. No movement means the iteration has stopped.
            increment = tl.where(mask, v_new - input_adjoint, tl.zeros_like(v_new))
            increment_max = tl.max(tl.abs(increment))
            v_max = tl.max(tl.abs(tl.where(mask, v_new, tl.zeros_like(v_new))))
            n_taken += 1
            if (n_taken >= max_terms) or (increment_max <= term_tol * v_max):
                running = 0
            tl.debug_barrier()
    else:
        # Cold branch. Term 0 reads the incoming adjoint and is peeled out of the
        # loop so the loop never has to choose between two different base pointers;
        # later terms ping-pong between the two halves of term_pair.
        input_adjoint, self_loop_vjp = _reconciliation_self_loop_transpose_term(
            rhs_ptr,
            term_pair_ptr,
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
            rows,
            s_offs,
            offsets,
            mask,
            store_mask,
            row_mask,
            species_valid,
            S,
            BLOCK_W,
            BLOCK_S,
            BLOCK_NODES,
            N_LEVELS,
            DTYPE,
            USE_CHILD_EDGE_SELF_LOOP,
        )
        tl.store(
            term_pair_ptr + offsets,
            tl.where(mask, self_loop_vjp, tl.zeros_like(self_loop_vjp)),
            mask=store_mask,
        )
        v_prev = tl.load(v_k_ptr + offsets, mask=mask, other=0.0)
        v_new = v_prev + self_loop_vjp
        tl.store(v_k_ptr + offsets, v_new, mask=mask)
        term_max = tl.max(
            tl.abs(tl.where(mask, self_loop_vjp, tl.zeros_like(self_loop_vjp)))
        )
        v_max = tl.max(tl.abs(tl.where(mask, v_new, tl.zeros_like(v_new))))
        n_taken = 1
        running = 1
        if (n_taken >= max_terms) or (term_max <= term_tol * v_max):
            running = 0
        tl.debug_barrier()
        # out_off always names the half holding the newest term (buffer 0 so far),
        # so it is still right if the loop below never runs.
        in_off = zero_offset
        while running == 1:
            in_off = out_off
            out_off = buffer_stride - out_off
            input_adjoint, self_loop_vjp = _reconciliation_self_loop_transpose_term(
                term_pair_ptr + in_off,
                term_pair_ptr + out_off,
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
                rows,
                s_offs,
                offsets,
                mask,
                store_mask,
                row_mask,
                species_valid,
                S,
                BLOCK_W,
                BLOCK_S,
                BLOCK_NODES,
                N_LEVELS,
                DTYPE,
                USE_CHILD_EDGE_SELF_LOOP,
            )
            tl.store(
                term_pair_ptr + out_off + offsets,
                tl.where(mask, self_loop_vjp, tl.zeros_like(self_loop_vjp)),
                mask=store_mask,
            )
            v_prev = tl.load(v_k_ptr + offsets, mask=mask, other=0.0)
            v_new = v_prev + self_loop_vjp
            tl.store(v_k_ptr + offsets, v_new, mask=mask)
            term_max = tl.max(
                tl.abs(tl.where(mask, self_loop_vjp, tl.zeros_like(self_loop_vjp)))
            )
            v_max = tl.max(tl.abs(tl.where(mask, v_new, tl.zeros_like(v_new))))
            n_taken += 1
            if (n_taken >= max_terms) or (term_max <= term_tol * v_max):
                running = 0
            tl.debug_barrier()

    if WRITE_LAST_TERM:
        # The stiffness diagnostic on the host reads the FIRST half of term_pair,
        # so mirror the final term there when the ping-pong left it in the second.
        if out_off != zero_offset:
            last_term = tl.load(
                term_pair_ptr + out_off + offsets, mask=mask, other=0.0
            )
            tl.store(
                term_pair_ptr + offsets,
                tl.where(mask, last_term, tl.zeros_like(last_term)),
                mask=store_mask,
            )
    if COLLECT_TERMS:
        tl.store(
            terms_taken_ptr + rows,
            n_taken + tl.zeros([BLOCK_W], dtype=tl.int32),
            mask=row_valid,
        )



@triton.jit
def _adjoint_children_subtrees(
    constant1, parent_weight1, gain1,
    constant2, parent_weight2, gain2,
    donor_coefficient,
):
    """Both children's subtree adjoints as affine functions of the parent's ``v`` and ``Off``.

    Child ``i`` arrives with ``D[c_i] = P_i + Q_i v[t] + R_i Off[c_i]`` and sees
    ``Off[c_i] = Off[t] + g[t] v[t] + D[c_other]``: the two children are coupled through each
    other's subtree, a 2x2 system whose solution is

        D[c_i] = A_i + B_i v[t] + C_i Off[t]

    with ``den = 1 - R_1 R_2``. Returns ``(A_1, B_1, C_1, A_2, B_2, C_2, den)``. ``R`` is a gain
    built from primal masses only, well below 1, so ``den`` is one minus a small product.
    """
    den = 1.0 - gain1 * gain2
    a1 = (constant1 + gain1 * constant2) / den
    b1 = (parent_weight1 + gain1 * parent_weight2 + gain1 * (1.0 + gain2) * donor_coefficient) / den
    c1 = gain1 * (1.0 + gain2) / den
    a2 = (constant2 + gain2 * constant1) / den
    b2 = (parent_weight2 + gain2 * parent_weight1 + gain2 * (1.0 + gain1) * donor_coefficient) / den
    c2 = gain2 * (1.0 + gain1) / den
    return a1, b1, c1, a2, b2, c2, den





# ``ws``/``W`` are the wave's start row and width, and ``rhs`` is the wave's slice of the
# [clades, species] adjoint buffer, so its 16-byte alignment changes with the wave start. All
# three would otherwise recompile the kernel per wave (see README.md).
@triton.jit(do_not_specialize=["ws", "W"], do_not_specialize_on_alignment=["rhs_ptr"])
def _solve_reconciliation_self_loop_transpose_row_kernel(
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
    duplication_loss_const_ptr, Ebar_ptr, E_ptr,
    speciation_child1_const_ptr, speciation_child2_const_ptr,
    receiver_log_probs_ptr,
    species_child1_ptr, species_child2_ptr, species_parent_ptr, species_height_ptr,
    not_open_source_ptr, closed_source_ptr, not_open_index_ptr, closed_index_ptr,
    leaf_term_ptr,
    leaf_species_ptr,
    leaf_logp_ptr,
    leaf_fm_log_ptr,
    family_idx_ptr,
    v_k_ptr,
    donor_adjoint_coefficient_ptr,
    receiver_mass_ptr,
    guard_trips_ptr,
    spill_ptr,
    spill_count_ptr,
    conditioning_floor,
    ws,
    W,
    S: tl.constexpr,
    stride: tl.constexpr,
    BLOCK_S: tl.constexpr,
    N_LEVELS: tl.constexpr,
    USE_LEAF_INDEX: tl.constexpr,
    HAS_LEAF_TERM: tl.constexpr,
    LEAF_LOGP_MODE: tl.constexpr,
    USE_FRACTION_MISSING: tl.constexpr,
    CONST_LAYOUT: tl.constexpr,
    DTYPE: tl.constexpr,
    USE_RECEIVER_WEIGHTS: tl.constexpr,
    PUBLISH_DONOR_TERMS: tl.constexpr,
    WRITE_GUARD_TRIPS: tl.constexpr,
    COUNT_SPILLS: tl.constexpr,
):
    """Solve one clade row's transposed self-loop EXACTLY, with the whole row in registers.

    Replaces the Neumann series (:func:`_reconciliation_self_loop_transpose_series_kernel`) with a
    direct solve of the same system. Same entry and exit contract: ``rhs`` in, the solution of

        (I - J^T) v = rhs

    written into ``v_k``. The series reaches that by summing ``(J^T)^k rhs`` (cold) or by iterating
    ``v <- rhs + J^T v`` (warm); both converge to the same v, so the exact solve makes the warm and
    cold branches identical and ``initial_v`` irrelevant.

    THE SYSTEM. ``J`` is the wave update's Jacobian, whose transpose the series applies term by
    term in :func:`_reconciliation_self_loop_transpose_term`. Per species ``t``, with

        d[t]  self-loop diagonal        -- stay in t (duplicate-and-lose, or transfer-and-lose)
        g[t]  donor adjoint coefficient -- t's transfer term per unit of donor mass reaching it
        m[t]  receiver mass             -- t's own weight as a transfer donor
        sp[t] parent-to-child probability -- speciate at parent(t), follow the edge into t

    that transpose reads

        v[t] = rhs[t] + d[t] v[t] + m[t] Off[t] + sp[t] v[parent(t)]
        Off[t] = sum of g[s] v[s] over every s OUTSIDE t's subtree (s neither t nor below it)

    -- the mirror image of the forward system: the forward's children terms became the single
    parent term ``sp[t] v[parent(t)]``, and the forward's "every species except my ancestors and
    me" became "every species except my descendants and me". ``Off`` passes DOWN the tree by
    addition: with ``D[t] = g[t] v[t] + D[c1] + D[c2]`` the adjoint of ``t``'s whole subtree,

        Off[root] = 0,      Off[c1] = Off[t] + g[t] v[t] + D[c2],      Off[c2] = Off[t] + g[t] v[t] + D[c1].

    Why not ``G - D[t]`` with ``G`` the row total, which is the same number on paper. ``g[s]``
    divides by ``s``'s own valid receiver mass, so for a species hanging under the lane that holds
    the row's mass it is astronomically large -- about 2**(that lane's depth below the row
    maximum) -- and ``G`` is dominated by those terms. For the dominant lane itself, whose subtree
    holds all of them, ``G - D`` is then a difference of two astronomically large, nearly equal
    numbers: rounding noise of that size, injected straight into the adjoint. Measured on a
    1007-species Coleman family at the loss-rate cap (benchmark/cc/corner_fd_grad.py, float64),
    the gradient came out 1e8 times larger than finite differences of the likelihood. The forward
    kernel removed the same subtraction from its own solve for the same reason.

    So ``t`` couples upward through ``v[parent(t)]`` and ``Off[t]`` only; nothing below ``t``
    enters its own row. Two O(S) walks solve it:

    1. Bottom-up, shallowest subtree first. Each lane's row is

           v[t] = a[t] + b[t] v[parent(t)] + c[t] Off[t]
           a = rhs / (1 - d)      b = sp / (1 - d)      c = m / (1 - d)

       and each node's SUBTREE adjoint becomes an affine function of the two numbers that reach
       the subtree from above,

           D[t] = P[t] + Q[t] v[parent(t)] + R[t] Off[t].

       A leaf has ``P = g a``, ``Q = g b``, ``R = g c``. At an internal node the two children are
       coupled through each other (child 1's ``Off`` contains child 2's ``D`` and vice versa);
       :func:`_adjoint_children_subtrees` resolves that 2x2 system into ``D[c_i] = A_i + B_i v[t]
       + C_i Off[t]``, and then

           D[t] = g v[t] + D[c1] + D[c2]     with   v[t] = a + b v[parent] + c Off[t]

       gives ``P = (A1 + A2) + (g + B1 + B2) a``, ``Q = (g + B1 + B2) b``,
       ``R = (g + B1 + B2) c + C1 + C2``. ``R`` is a gain built from primal masses only (the
       receiver's share of transfer-in times its donor mass over its valid receiver mass), well
       below 1, so ``1 - R1 R2`` is one minus a small product and never cancels.

    2. The root has no parent and nothing off its subtree: ``v[root] = a[root]``. Top-down, the
       same levels reversed: a node with settled ``v[t]`` and ``Off[t]`` evaluates its children's
       ``D`` from their triples, hands each child ``Off[c] = Off[t] + g[t] v[t] + D[sibling]`` and
       ``v[c] = a[c] + b[c] v[t] + c[c] Off[c]``. Additions only: ``Off`` of a lane on the
       dominant lane's chain never touches the astronomical terms under it.

    Unlike the forward solve there is no sign structure to exploit: ``rhs`` is a gradient, so
    ``a``, ``P`` and ``v`` carry signs and a subtree whose adjoint changes sign cancels. That
    cancellation is in the system, not in this method -- the series sums the same signed terms.

    WHERE THE NUMBERS LIVE. The two kernels this one replaces passed eight
    ``[wave clades, species]`` arrays through global memory: five coefficients a prepare kernel
    wrote and a block-tiled elimination read back (``self_loop_diagonal``,
    ``donor_adjoint_coefficient``, ``receiver_mass`` and the two speciation edge probabilities),
    two running sums for the valid receiver mass, and three working arrays that elimination's own
    walks wrote and re-read at every species-tree level. On the 200-family Coleman batch each of
    those is 12 GB per gradient. Here one program owns one clade row, the whole species row sits
    in registers as ``BLOCK_S`` lanes spread over several warps, and the coefficients are computed
    from the primal row on the spot, so none of those eight arrays is written or read.

    HOW MUCH THAT WAS WORTH, measured rather than assumed. The round trip really does go: Nsight
    Compute on one wave puts this kernel's DRAM throughput at 1.5 % of peak against the old pair's
    36 %. But the two kernels were never bandwidth-bound -- they were latency-bound at 16 %
    occupancy -- so the gradient only got about 2 % faster (325.4 ms of 1926 ms in the two kernels
    before, 318.5 ms of 1912 ms in this one plus the residual prepare launch after). ``tl.gather``
    stages through shared memory, and it pays back there what the arrays cost in DRAM: the top
    stall is now ``mio_throttle`` at 2.43 warps per issue, L1 throughput 53 %, SM throughput 7.6 %,
    112 registers per thread at 16 warps for 33 % occupancy. Re-launching one captured 254-row
    wave says where the time goes: 58.3 us in all, of which the coefficient setup above is 18.4 us
    and the two walks below are 40 us, 1.21 us per species-tree level. The saving is the setup
    half; the walks cost what the block-tiled walks through global memory cost. See
    docs/genewise_h100_runtime.md, round five, for what would pay next.

    The two walks change shape to match. The block-tiled elimination walked one level's node list
    at a time out of the ``compact_level_*`` tables; a register-resident program cannot scatter into
    another lane's register, so each level here is a WHOLE-ROW update masked to the lanes at that
    height (``species_height``, 0 at a leaf), with children, parent and sibling reached by
    ``tl.gather``. Every lane computes every level's update and only the lanes at that height keep
    it; their children sit at strictly lower heights and are therefore already final. This is the
    scheme :func:`gpurec.core.kernels.wave_tangent._solve_reconciliation_self_loop_jvp_exact_kernel`
    already uses on the forward side.

    The top-down walk is re-pointed as well. The block-tiled form stood at a node and wrote
    BOTH of its children; here each lane settles ITSELF from its parent, which is the same
    assignment seen from the other end: lane ``s`` gathers its parent's ``v`` and ``Off`` and its
    SIBLING's ``(P, Q, R)`` triple, hands (sibling, self) to :func:`_adjoint_children_subtrees` in
    that order -- so the triple that comes back is the sibling's subtree adjoint, which is exactly
    the term ``Off[s]`` is missing -- and lands on the same two numbers, in the same order of
    operations, as the parent-side form. ``Off`` gets its own register vector instead of
    overwriting ``P``; the block-tiled form overwrote ``P`` only to save one
    ``[clades, species]`` array, and here there is no array to save.

    WHAT IS UNCHANGED, DELIBERATELY. Every arithmetic expression is the one those two kernels
    used, in the same order, so the gradient does not move:

    * the six event log-terms, their shared maximum, the six masses and the split wave's
      ``within_wave_probability`` factor -- copied term for term from the prepare kernel;
    * the valid receiver mass, still the depth-first "not yet open" plus "already closed" pair of
      running sums (:func:`gpurec.core.valid_receivers.valid_receiver_index_tables`), still built
      by ``tl.cumsum`` over the same ``BLOCK_S`` lanes in the same order -- only the gathers around
      it read registers instead of a scratch array. NOT the additive tree walk of
      :func:`gpurec.core.kernels.species_tree_sums.valid_receiver_sum`, which is the same number to
      a different rounding and would have moved the answer for no reason;
    * ``sp[t]``, "speciate at parent(t), follow the edge into t". The prepare kernel wrote this by
      SCATTERING each node's two speciation probabilities into its children's slots; a
      register-resident program cannot scatter, so each lane GATHERS the one its parent computed
      for it, picking the first or the second child's according to which one it is. Same number.
      A lane with no parent gets 0 -- the block-tiled path left the root's slot unwritten and its
      ``Q[root]`` is dead either way, so this changes nothing but is defined;
    * the conditioning spill: a row whose smallest pivot ``1 - d`` or smallest sibling divisor
      ``1 - R1 R2`` is below ``conditioning_floor`` writes ``rhs`` into ``v_k``, raises its flag in
      ``spill_ptr`` and stops, and the Neumann series takes it from there.

    ``donor_adjoint_coefficient`` and ``receiver_mass`` are still PUBLISHED, under
    ``PUBLISH_DONOR_TERMS``, because
    :func:`_accumulate_transfer_receiver_log_probability_vjp_kernel` reads them when the receiver
    weights carry a gradient. When they do not -- the genewise recipe -- nothing downstream reads
    them and the flag is off. The other three coefficient arrays have no reader left at all:
    :func:`_accumulate_reconciliation_event_vjp_kernel` rebuilds every mass it needs from
    ``Pi_star`` itself.

    ``guard_trips_ptr`` is an optional ``[W, 2]`` int32 diagnostic: column 0 counts the row's
    non-positive pivots ``1 - d``, column 1 flags a non-positive ``1 - R1 R2`` at any node.
    Neither is substituted or clipped -- a non-positive one divides through to an infinity or a
    NaN.
    """
    NEG_LARGE: tl.constexpr = -float("inf")
    POS_LARGE: tl.constexpr = float("inf")

    # int64: rows range over the whole batch's clade rows, so the *stride/*S address arithmetic
    # below can overflow int32 once total_clades * S exceeds 2^31.
    w = tl.program_id(0).to(tl.int64)
    if tl.load(active_mask_ptr + w) == 0:
        # Nothing to write, so nothing to compute: this row is either pruned, or one the forward
        # already handed to its log sweeps and the series will take here.
        return

    row_global = ws + w
    pi_base = row_global * stride
    out_base = w * S
    s_offs = tl.arange(0, BLOCK_S)
    mask = s_offs < S
    zero = tl.zeros([BLOCK_S], dtype=DTYPE)
    lane = tl.arange(0, 1)

    pi_row_offset = tl.load(Pi_offset_ptr + row_global)
    pibar_row_offset = tl.load(Pibar_offset_ptr + row_global)
    pibar_offset_corr = (pibar_row_offset - pi_row_offset).to(DTYPE)
    leaf_offset_corr = (-pi_row_offset).to(DTYPE)

    # ---- the primal row, and every donor's own weight as a transfer source.
    row_max = tl.load(Pibar_row_max_ptr + row_global)
    reconciliation_log_likelihood = tl.load(
        Pi_star_ptr + pi_base + s_offs, mask=mask, other=NEG_LARGE
    )
    transfer_complement_log_likelihood = tl.load(
        Pibar_star_ptr + pi_base + s_offs, mask=mask, other=NEG_LARGE
    )
    row_max_safe = tl.where(row_max != NEG_LARGE, row_max, tl.zeros_like(row_max))
    if USE_RECEIVER_WEIGHTS:
        receiver_log_probability = tl.load(
            receiver_log_probs_ptr + s_offs, mask=mask, other=NEG_LARGE
        )
        receiver_mass = tl.exp2(
            receiver_log_probability + reconciliation_log_likelihood - row_max_safe
        )
    else:
        receiver_mass = tl.exp2(reconciliation_log_likelihood - row_max_safe)
    receiver_mass = tl.where(mask, receiver_mass, zero)

    # ---- the species-tree neighbourhood of every lane, gathered once and reused by both walks.
    (
        species_height, c1_valid, c1_safe, c2_valid, c2_safe,
        has_parent, parent_safe, parent_height, has_sibling, sibling_safe,
    ) = species_neighbourhood(
        species_child1_ptr, species_child2_ptr, species_parent_ptr, species_height_ptr,
        s_offs, mask, S,
    )
    is_root = mask & ~has_parent

    # ---- the per-species event constants, in whichever layout the caller keeps them.
    if CONST_LAYOUT == 1:
        family = 0
        const_base = 0
        const_offsets = out_base + s_offs
    elif CONST_LAYOUT == 2:
        family = tl.load(family_idx_ptr + row_global).to(tl.int64)
        const_base = family * stride
        const_offsets = const_base + s_offs
    else:
        family = 0
        const_base = 0
        const_offsets = s_offs
    duplication_loss_const = tl.load(
        duplication_loss_const_ptr + const_offsets, mask=mask, other=NEG_LARGE
    )
    extinction_complement_log_probability = tl.load(
        Ebar_ptr + const_offsets, mask=mask, other=NEG_LARGE
    )
    extinction_log_probability = tl.load(E_ptr + const_offsets, mask=mask, other=NEG_LARGE)
    speciation_child1_const = tl.load(
        speciation_child1_const_ptr + const_offsets, mask=mask, other=NEG_LARGE
    )
    speciation_child2_const = tl.load(
        speciation_child2_const_ptr + const_offsets, mask=mask, other=NEG_LARGE
    )

    # The two children's primal likelihoods come out of the row already in registers.
    reconciliation_child1_log_likelihood = tl.where(
        c1_valid, tl.gather(reconciliation_log_likelihood, c1_safe, axis=0), NEG_LARGE
    )
    reconciliation_child2_log_likelihood = tl.where(
        c2_valid, tl.gather(reconciliation_log_likelihood, c2_safe, axis=0), NEG_LARGE
    )

    duplication_loss_log_term = duplication_loss_const + reconciliation_log_likelihood
    transfer_loss_log_term = reconciliation_log_likelihood + extinction_complement_log_probability
    transfer_log_term = (
        transfer_complement_log_likelihood + extinction_log_probability + pibar_offset_corr
    )
    speciation_child1_log_term = speciation_child1_const + reconciliation_child1_log_likelihood
    speciation_child2_log_term = speciation_child2_const + reconciliation_child2_log_likelihood
    if USE_LEAF_INDEX:
        leaf_species = tl.load(leaf_species_ptr + row_global)
        leaf_hit = mask & (leaf_species == s_offs)
        if LEAF_LOGP_MODE == 3:
            leaf_logp = tl.load(leaf_logp_ptr)
            leaf_observation_log_term = tl.where(leaf_hit, leaf_logp, NEG_LARGE)
        elif LEAF_LOGP_MODE == 1:
            leaf_logp = tl.load(leaf_logp_ptr + family)
            leaf_observation_log_term = tl.where(leaf_hit, leaf_logp, NEG_LARGE)
        elif LEAF_LOGP_MODE == 2:
            if USE_FRACTION_MISSING:
                # Off-hit leaf-species columns carry the "present-but-unobserved" baseline
                # log_pS[s] + log2(fm_s); non-leaf/observed columns stay -inf (fm_col is -inf
                # there). Mirrors the Pi forward.
                leaf_logp = tl.load(
                    leaf_logp_ptr + const_base + s_offs, mask=mask, other=NEG_LARGE
                )
                fm_col = tl.load(leaf_fm_log_ptr + s_offs, mask=mask, other=NEG_LARGE)
                baseline = leaf_logp + fm_col
                leaf_observation_log_term = tl.where(leaf_hit, leaf_logp, baseline)
            else:
                leaf_logp = tl.load(
                    leaf_logp_ptr + const_base + s_offs, mask=leaf_hit, other=NEG_LARGE
                )
                leaf_observation_log_term = tl.where(leaf_hit, leaf_logp, NEG_LARGE)
        else:
            leaf_logp = tl.load(leaf_logp_ptr + s_offs, mask=mask, other=NEG_LARGE)
            if USE_FRACTION_MISSING:
                fm_col = tl.load(leaf_fm_log_ptr + s_offs, mask=mask, other=NEG_LARGE)
                baseline = leaf_logp + fm_col
                leaf_observation_log_term = tl.where(leaf_hit, leaf_logp, baseline)
            else:
                leaf_observation_log_term = tl.where(leaf_hit, leaf_logp, NEG_LARGE)
    elif HAS_LEAF_TERM:
        leaf_observation_log_term = tl.load(
            leaf_term_ptr + out_base + s_offs, mask=mask, other=NEG_LARGE
        )
    else:
        leaf_observation_log_term = tl.full([BLOCK_S], value=NEG_LARGE, dtype=DTYPE)
    if USE_LEAF_INDEX or HAS_LEAF_TERM:
        leaf_observation_log_term += leaf_offset_corr

    local_event_max = tl.maximum(duplication_loss_log_term, transfer_loss_log_term)
    local_event_max = tl.maximum(local_event_max, transfer_log_term)
    local_event_max = tl.maximum(local_event_max, speciation_child1_log_term)
    local_event_max = tl.maximum(local_event_max, speciation_child2_log_term)
    local_event_max = tl.maximum(local_event_max, leaf_observation_log_term)
    local_event_max_safe = tl.where(local_event_max != NEG_LARGE, local_event_max, zero)
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
    inverse_local_event_scaled_mass = tl.where(
        local_event_scaled_mass > 0.0,
        1.0 / local_event_scaled_mass,
        tl.zeros_like(local_event_scaled_mass),
    )

    if has_splits:
        gene_split_row_offset = tl.load(gene_split_offset_ptr + w)
        gene_split_frame_shift = (gene_split_row_offset - pi_row_offset).to(DTYPE)
        gene_split_log_likelihood = tl.load(
            gene_split_log_likelihood_ptr + out_base + s_offs, mask=mask, other=NEG_LARGE
        )
        gene_split_log_likelihood += gene_split_frame_shift
        local_event_log_likelihood = tl.log2(local_event_scaled_mass) + local_event_max
        updated_reconciliation_max = tl.maximum(
            local_event_log_likelihood, gene_split_log_likelihood
        )
        updated_reconciliation_max_safe = tl.where(
            updated_reconciliation_max != NEG_LARGE,
            updated_reconciliation_max,
            tl.zeros_like(updated_reconciliation_max),
        )
        updated_reconciliation_log_likelihood = tl.log2(
            tl.exp2(local_event_log_likelihood - updated_reconciliation_max_safe)
            + tl.exp2(gene_split_log_likelihood - updated_reconciliation_max_safe)
        ) + updated_reconciliation_max
        within_wave_probability = tl.where(
            local_event_log_likelihood != NEG_LARGE,
            tl.exp2(local_event_log_likelihood - updated_reconciliation_log_likelihood),
            tl.zeros_like(local_event_log_likelihood),
        )
    else:
        within_wave_probability = tl.full([BLOCK_S], value=1.0, dtype=DTYPE)

    # ---- each donor's valid receiver mass: every species that is neither the donor itself nor one
    # of its ancestors, split into the ones whose subtree has not opened yet and the ones whose
    # subtree already closed. Two disjoint groups, two running sums of non-negative terms, and so
    # no subtraction of nearly equal numbers -- see the prepare kernel's comment for the
    # cancellation this replaced.
    not_open_scan = tl.load(not_open_source_ptr + s_offs, mask=mask, other=S)
    not_open_contributes = mask & (not_open_scan < S)
    not_open_prefix = tl.cumsum(
        tl.where(
            not_open_contributes,
            tl.gather(receiver_mass, tl.where(not_open_contributes, not_open_scan, 0), axis=0),
            zero,
        ),
        axis=0,
    )
    closed_scan = tl.load(closed_source_ptr + s_offs, mask=mask, other=S)
    closed_contributes = mask & (closed_scan < S)
    closed_prefix = tl.cumsum(
        tl.where(
            closed_contributes,
            tl.gather(receiver_mass, tl.where(closed_contributes, closed_scan, 0), axis=0),
            zero,
        ),
        axis=0,
    )
    not_open_index = tl.load(not_open_index_ptr + s_offs, mask=mask, other=0)
    closed_index = tl.load(closed_index_ptr + s_offs, mask=mask, other=0)
    not_yet_open = tl.where(mask, tl.gather(not_open_prefix, not_open_index, axis=0), zero)
    already_closed = tl.where(mask, tl.gather(closed_prefix, closed_index, axis=0), zero)
    valid_receiver_mass = not_yet_open + already_closed
    inverse_valid_receiver_mass = tl.where(
        valid_receiver_mass > 0.0,
        1.0 / valid_receiver_mass,
        tl.zeros_like(valid_receiver_mass),
    )

    # ---- the four coefficients of the transposed row, per species: d (stay here), g (transfer in,
    # per unit of donor mass), m (this lane's own weight as a donor) and sp (speciate at the parent
    # and follow the edge into this lane).
    self_loop_diagonal = tl.where(
        mask,
        within_wave_probability
        * (duplication_loss_mass + transfer_loss_mass)
        * inverse_local_event_scaled_mass,
        zero,
    )
    donor_adjoint_coefficient = tl.where(
        mask,
        within_wave_probability
        * transfer_mass
        * inverse_local_event_scaled_mass
        * inverse_valid_receiver_mass,
        zero,
    )
    speciation_child1_probability = (
        within_wave_probability * speciation_child1_mass * inverse_local_event_scaled_mass
    )
    speciation_child2_probability = (
        within_wave_probability * speciation_child2_mass * inverse_local_event_scaled_mass
    )
    is_first_child = has_parent & (tl.gather(c1_safe, parent_safe, axis=0) == s_offs)
    parent_to_child_probability = tl.where(
        has_parent,
        tl.where(
            is_first_child,
            tl.gather(speciation_child1_probability, parent_safe, axis=0),
            tl.gather(speciation_child2_probability, parent_safe, axis=0),
        ),
        zero,
    )

    if PUBLISH_DONOR_TERMS:
        # The transfer receiver-weight VJP is the only kernel downstream that still reads these.
        tl.store(
            donor_adjoint_coefficient_ptr + out_base + s_offs,
            donor_adjoint_coefficient,
            mask=mask,
        )
        tl.store(receiver_mass_ptr + out_base + s_offs, receiver_mass, mask=mask)

    # ---- one lane's solved row as an affine function of what is above it:
    # v[t] = a[t] + b[t] v[parent(t)] + c[t] Off[t]. Nothing below t enters its own row, so the
    # pivot is just 1 - d[t]; a masked-off lane has d = 0, hence pivot exactly 1 and a safe
    # division.
    right_hand_side = tl.load(rhs_ptr + out_base + s_offs, mask=mask, other=0.0)
    pivot = 1.0 - self_loop_diagonal
    lane_a = right_hand_side / pivot
    lane_b = parent_to_child_probability / pivot
    lane_c = receiver_mass / pivot

    # ---- walk 1: bottom-up. Every lane is seeded with the leaf case P = g a, Q = g b, R = g c;
    # pass ``level`` then rewrites the lanes of that height, whose children are already final.
    subtree_donor_constant = donor_adjoint_coefficient * lane_a
    subtree_donor_weight = donor_adjoint_coefficient * lane_b
    subtree_donor_gain = donor_adjoint_coefficient * lane_c
    smallest_pivot = tl.min(
        tl.where(mask, pivot, tl.full([BLOCK_S], value=POS_LARGE, dtype=DTYPE)), axis=0
    )
    nonpositive_pivots = tl.sum(tl.where(mask & (pivot <= 0.0), 1, 0).to(tl.int32), axis=0)
    # Rank-0 running minimum / counter, seeded through the same reductions the loop uses so their
    # types line up without a scalar literal Triton would have to widen.
    smallest_den = tl.min(tl.full([BLOCK_S], value=POS_LARGE, dtype=DTYPE), axis=0)
    nonpositive_den = tl.sum(tl.zeros([BLOCK_S], dtype=tl.int32), axis=0)
    for level in range(1, N_LEVELS + 1):
        child1_constant = tl.where(
            c1_valid, tl.gather(subtree_donor_constant, c1_safe, axis=0), zero
        )
        child1_weight = tl.where(c1_valid, tl.gather(subtree_donor_weight, c1_safe, axis=0), zero)
        child1_gain = tl.where(c1_valid, tl.gather(subtree_donor_gain, c1_safe, axis=0), zero)
        child2_constant = tl.where(
            c2_valid, tl.gather(subtree_donor_constant, c2_safe, axis=0), zero
        )
        child2_weight = tl.where(c2_valid, tl.gather(subtree_donor_weight, c2_safe, axis=0), zero)
        child2_gain = tl.where(c2_valid, tl.gather(subtree_donor_gain, c2_safe, axis=0), zero)
        a1, b1, c1, a2, b2, c2, den = _adjoint_children_subtrees(
            child1_constant, child1_weight, child1_gain,
            child2_constant, child2_weight, child2_gain,
            donor_adjoint_coefficient,
        )
        # D[t] = g v[t] + D[c1] + D[c2], with v[t] = a + b v[parent] + c Off[t].
        per_unit_v = donor_adjoint_coefficient + b1 + b2
        at_level = mask & (species_height == level)
        subtree_donor_constant = tl.where(
            at_level, a1 + a2 + per_unit_v * lane_a, subtree_donor_constant
        )
        subtree_donor_weight = tl.where(at_level, per_unit_v * lane_b, subtree_donor_weight)
        subtree_donor_gain = tl.where(
            at_level, per_unit_v * lane_c + c1 + c2, subtree_donor_gain
        )
        smallest_den = tl.minimum(
            smallest_den,
            tl.min(
                tl.where(at_level, den, tl.full([BLOCK_S], value=POS_LARGE, dtype=DTYPE)), axis=0
            ),
        )
        nonpositive_den += tl.sum(tl.where(at_level & (den <= 0.0), 1, 0).to(tl.int32), axis=0)

    if WRITE_GUARD_TRIPS:
        tl.store(guard_trips_ptr + w * 2 + lane, nonpositive_pivots)
        tl.store(guard_trips_ptr + w * 2 + 1 + lane, tl.where(nonpositive_den > 0, 1, 0))

    # ---- the conditioning decision. Every lane divides by ``1 - d`` and every node by
    # ``1 - R1 R2``; a margin m costs about eps/m in relative error, so a row whose smallest margin
    # is under ``conditioning_floor`` goes to the Neumann series instead, which has no such
    # division. Its ``v_k`` is left holding the rhs, which is what the series' cold branch starts
    # from.
    if (smallest_pivot < conditioning_floor) | (smallest_den < conditioning_floor):
        tl.store(spill_ptr + w + lane, tl.full([1], value=1, dtype=tl.int8))
        if COUNT_SPILLS:
            # One number per wave saying whether ANY row was handed to the series, so the host can
            # skip the series launch entirely. Spilling needs a badly conditioned pivot and is
            # rare, so this atomic almost never fires.
            tl.atomic_add(spill_count_ptr + lane, tl.full([1], value=1, dtype=tl.int32))
        tl.store(v_k_ptr + out_base + s_offs, right_hand_side, mask=mask)
        return

    # ---- walk 2: top-down. The root has no parent and nothing off its subtree, so v[root] =
    # a[root] and Off[root] = 0; every other lane is settled in the pass whose level is its
    # PARENT's height, out of the parent's settled v and Off and the sibling's final triple.
    # Walk 1 has settled every triple and nothing below rewrites them, so the sibling coupling is
    # the SAME number at every level of this walk. Solve each lane's 2x2 once here, outside the
    # loop, and the loop is left with two gathers and eight flops: (sibling, self) in that order,
    # so the triple that comes back is the SIBLING's subtree adjoint as an affine function of the
    # parent's v and Off -- the one term ``Off[self]`` is missing.
    parent_donor = tl.gather(donor_adjoint_coefficient, parent_safe, axis=0)
    sibling_a, sibling_b, sibling_c, _a2, _b2, _c2, _den = _adjoint_children_subtrees(
        tl.where(has_sibling, tl.gather(subtree_donor_constant, sibling_safe, axis=0), zero),
        tl.where(has_sibling, tl.gather(subtree_donor_weight, sibling_safe, axis=0), zero),
        tl.where(has_sibling, tl.gather(subtree_donor_gain, sibling_safe, axis=0), zero),
        subtree_donor_constant, subtree_donor_weight, subtree_donor_gain,
        parent_donor,
    )
    reconciliation_adjoint = tl.where(is_root, lane_a, zero)
    off_subtree_donor_adjoint = zero
    for level_index in range(0, N_LEVELS):
        level = N_LEVELS - level_index
        at_level = has_parent & (parent_height == level)
        parent_adjoint = tl.gather(reconciliation_adjoint, parent_safe, axis=0)
        parent_off = tl.gather(off_subtree_donor_adjoint, parent_safe, axis=0)
        sibling_subtree = sibling_a + sibling_b * parent_adjoint + sibling_c * parent_off
        shared = parent_off + parent_donor * parent_adjoint
        level_off = shared + sibling_subtree
        level_adjoint = lane_a + lane_b * parent_adjoint + lane_c * level_off
        reconciliation_adjoint = tl.where(at_level, level_adjoint, reconciliation_adjoint)
        off_subtree_donor_adjoint = tl.where(at_level, level_off, off_subtree_donor_adjoint)

    tl.store(v_k_ptr + out_base + s_offs, reconciliation_adjoint, mask=mask)


# ``W`` is the wave's width and changes every launch; keeping it out of the specialization
# key avoids one JIT compile per new "== 1" / divisible-by-16 state (see README.md).
@triton.jit(do_not_specialize=["W"])
def _accumulate_transfer_receiver_log_probability_vjp_kernel(
    v_k_ptr,
    active_mask_ptr,
    donor_adjoint_coefficient_ptr,
    receiver_mass_ptr,
    species_parent_ptr,
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
    # int64: rows range over the whole batch's clade rows, so the *S address
    # arithmetic below can overflow int32 once total_clades * S exceeds 2^31.
    block = tl.program_id(0).to(tl.int64)
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

    input_adjoint = tl.load(v_k_ptr + offsets, mask=mask, other=0.0)
    donor_adjoint_coefficient = tl.load(donor_adjoint_coefficient_ptr + offsets, mask=mask, other=0.0)
    donor_adjoint = input_adjoint * donor_adjoint_coefficient
    zero = tl.zeros([BLOCK_S, BLOCK_W], dtype=DTYPE)
    tl.store(subtree_donor_adjoint_ptr + offsets, tl.where(mask, donor_adjoint, zero), mask=store_mask)

    tl.debug_barrier()

    # Receiver s's weight moves the mass of every donor t that may transfer INTO s, that is every
    # donor OUTSIDE s's own subtree, so this kernel needs, per species s, the donor adjoint summed
    # off s's subtree. That used to be ``row total - subtree sum``. Each donor's adjoint divides by
    # that donor's own valid receiver mass, so for species hanging under the lane holding the row's
    # mass it is astronomically large (2**depth); the row total is then dominated by those terms,
    # and for the dominant lane -- whose subtree holds all of them -- the subtraction cancels to
    # rounding noise of that astronomical size. Measured on a 1007-species Coleman family at the
    # loss-rate cap, the gradient came out 1e8 times too large. So the off-subtree sum is built
    # top-down by ADDITION instead: what lies off a child's subtree is what lies off its parent's,
    # plus the parent's own term, plus the sibling's whole subtree. Subtree sums first (bottom-up),
    # then that walk (top-down), each child's off-subtree sum overwriting its subtree sum.
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
            )
            c1_val = tl.load(
                subtree_donor_adjoint_ptr + row_base + c1[:, None],
                mask=reduce_mask & (c1 < S)[:, None],
                other=0.0,
            )
            c2_val = tl.load(
                subtree_donor_adjoint_ptr + row_base + c2[:, None],
                mask=reduce_mask & (c2 < S)[:, None],
                other=0.0,
            )
            tl.store(
                subtree_donor_adjoint_ptr + row_base + parent[:, None],
                parent_val + c1_val + c2_val,
                mask=reduce_mask,
            )
            node_start += BLOCK_NODES
        tl.debug_barrier()

    # The root's subtree is the whole tree: nothing lies off it. Every other lane is written by
    # its parent below.
    species_parent_of_lane = tl.load(species_parent_ptr + s_offs, mask=species_valid, other=0)
    is_root_lane = (species_parent_of_lane < 0)[:, None] & mask
    tl.store(subtree_donor_adjoint_ptr + offsets, zero, mask=is_root_lane)
    tl.debug_barrier()
    for level_index in range(0, N_LEVELS):
        level = N_LEVELS - 1 - level_index
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
            c1_mask = reduce_mask & (c1 < S)[:, None]
            c2_mask = reduce_mask & (c2 < S)[:, None]
            row_base = rows[None, :] * S
            parent_off_subtree = tl.load(
                subtree_donor_adjoint_ptr + row_base + parent[:, None],
                mask=reduce_mask,
                other=0.0,
            )
            parent_own = tl.load(
                donor_adjoint_coefficient_ptr + row_base + parent[:, None],
                mask=reduce_mask,
                other=0.0,
            ) * tl.load(v_k_ptr + row_base + parent[:, None], mask=reduce_mask, other=0.0)
            c1_subtree = tl.load(
                subtree_donor_adjoint_ptr + row_base + c1[:, None], mask=c1_mask, other=0.0
            )
            c2_subtree = tl.load(
                subtree_donor_adjoint_ptr + row_base + c2[:, None], mask=c2_mask, other=0.0
            )
            tl.store(
                subtree_donor_adjoint_ptr + row_base + c1[:, None],
                parent_off_subtree + parent_own + c2_subtree,
                mask=c1_mask,
            )
            tl.store(
                subtree_donor_adjoint_ptr + row_base + c2[:, None],
                parent_off_subtree + parent_own + c1_subtree,
                mask=c2_mask,
            )
            node_start += BLOCK_NODES
        tl.debug_barrier()

    off_subtree_donor_adjoint = tl.load(subtree_donor_adjoint_ptr + offsets, mask=mask, other=0.0)
    receiver_mass = tl.load(receiver_mass_ptr + offsets, mask=mask, other=0.0)
    transfer_complement_vjp = receiver_mass * off_subtree_donor_adjoint
    species_contrib = tl.sum(tl.where(mask, transfer_complement_vjp, zero), axis=1)
    tl.atomic_add(
        grad_receiver_log_probs_ptr + s_offs,
        species_contrib,
        sem="relaxed",
        mask=species_valid,
    )


# ``ws``/``W`` are the wave's start row and width and change every launch; keeping them
# out of the specialization key avoids one JIT compile per new "== 1" /
# divisible-by-16 state (see README.md).
@triton.jit(do_not_specialize=["ws", "W"])
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
    leaf_fm_log_ptr,
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
    USE_FRACTION_MISSING: tl.constexpr,
    USE_ACTIVE_MASK: tl.constexpr,
    CONST_LAYOUT: tl.constexpr,
    ACCUM_GRADS: tl.constexpr,
    PARAM_GRAD_VECTOR: tl.constexpr,
    DTYPE: tl.constexpr,
):
    """Store per-element self-loop parameter VJP contributions after Neumann."""
    NEG_LARGE: tl.constexpr = -float("inf")

    # int64: rows range over the whole batch's clade rows, so the *stride/*S
    # address arithmetic below can overflow int32 once total_clades * S
    # exceeds 2^31.
    block = tl.program_id(0).to(tl.int64)
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
    reconciliation_log_likelihood = tl.load(Pi_star_ptr + pi_offsets, mask=mask, other=NEG_LARGE)
    transfer_complement_log_likelihood = tl.load(Pibar_star_ptr + pi_offsets, mask=mask, other=NEG_LARGE)
    parent_adjoint = tl.load(v_k_ptr + out_offsets, mask=mask, other=0.0)
    duplication_loss_const = tl.load(duplication_loss_const_ptr + const_offsets, mask=const_mask, other=NEG_LARGE)
    extinction_complement_log_probability = tl.load(
        Ebar_ptr + const_offsets, mask=const_mask, other=NEG_LARGE
    )
    extinction_log_probability = tl.load(
        E_ptr + const_offsets, mask=const_mask, other=NEG_LARGE
    )
    speciation_child1_const = tl.load(speciation_child1_const_ptr + const_offsets, mask=const_mask, other=NEG_LARGE)
    speciation_child2_const = tl.load(speciation_child2_const_ptr + const_offsets, mask=const_mask, other=NEG_LARGE)

    c1 = tl.load(species_child1_ptr + s_offs, mask=species_valid, other=0)
    c2 = tl.load(species_child2_ptr + s_offs, mask=species_valid, other=0)
    c1_valid = c1 < S
    c2_valid = c2 < S
    reconciliation_child1_log_likelihood = tl.load(
        Pi_star_ptr + row_global[None, :] * stride + c1[:, None],
        mask=(species_valid & c1_valid)[:, None] & row_mask[None, :],
        other=NEG_LARGE,
    )
    reconciliation_child2_log_likelihood = tl.load(
        Pi_star_ptr + row_global[None, :] * stride + c2[:, None],
        mask=(species_valid & c2_valid)[:, None] & row_mask[None, :],
        other=NEG_LARGE,
    )

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
            leaf_logp = tl.load(leaf_logp_ptr)
            leaf_observation_log_term = tl.where(leaf_hit, leaf_logp, NEG_LARGE)
        elif LEAF_LOGP_MODE == 1:
            leaf_logp = tl.load(leaf_logp_ptr + family, mask=row_valid, other=NEG_LARGE)
            leaf_observation_log_term = tl.where(leaf_hit, leaf_logp[None, :], NEG_LARGE)
        elif LEAF_LOGP_MODE == 2:
            if USE_FRACTION_MISSING:
                # Off-hit leaf-species columns carry the "present-but-unobserved"
                # baseline log_pS[s] + log2(fm_s); non-leaf/observed columns stay
                # -inf (fm_col is -inf there). Mirrors the Pi forward.
                leaf_logp = tl.load(
                    leaf_logp_ptr + const_base[None, :] + s_offs[:, None],
                    mask=mask,
                    other=NEG_LARGE,
                )
                fm_col = tl.load(leaf_fm_log_ptr + s_offs, mask=species_valid, other=NEG_LARGE)
                baseline = leaf_logp + fm_col[:, None]
                leaf_observation_log_term = tl.where(leaf_hit, leaf_logp, baseline)
            else:
                leaf_logp = tl.load(
                    leaf_logp_ptr + const_base[None, :] + s_offs[:, None],
                    mask=leaf_hit,
                    other=NEG_LARGE,
                )
                leaf_observation_log_term = tl.where(leaf_hit, leaf_logp, NEG_LARGE)
        else:
            leaf_logp = tl.load(leaf_logp_ptr + s_offs, mask=species_valid, other=NEG_LARGE)
            if USE_FRACTION_MISSING:
                fm_col = tl.load(leaf_fm_log_ptr + s_offs, mask=species_valid, other=NEG_LARGE)
                baseline = (leaf_logp + fm_col)[:, None]
                leaf_observation_log_term = tl.where(leaf_hit, leaf_logp[:, None], baseline)
            else:
                leaf_observation_log_term = tl.where(leaf_hit, leaf_logp[:, None], NEG_LARGE)
    elif HAS_LEAF_TERM:
        leaf_observation_log_term = tl.load(leaf_term_ptr + out_offsets, mask=mask, other=NEG_LARGE)
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
        gene_split_log_likelihood = tl.load(gene_split_log_likelihood_ptr + out_offsets, mask=mask, other=NEG_LARGE)
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


# ``ws`` is the wave's start row and changes every launch; keeping it out of the
# specialization key avoids one JIT compile per divisibility state (see README.md).
@triton.jit(do_not_specialize=["ws"])
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

    # int64: split_index ranges over the whole batch's split count, so the *S
    # address arithmetic below can overflow int32 once n_splits * S exceeds 2^31.
    split_index = tl.program_id(0).to(tl.int64)

    left_clade_row = tl.load(split_left_rows_ptr + split_index).to(tl.int64)
    right_clade_row = tl.load(split_right_rows_ptr + split_index).to(tl.int64)
    parent_wave_row = tl.load(reduce_idx_ptr + split_index).to(tl.int64)
    split_log_prior = tl.load(log_split_probs_ptr + split_index)
    if USE_ACTIVE_MASK:
        parent_active = tl.load(active_mask_ptr + parent_wave_row)
        if parent_active == 0:
            out_base = split_index * S
            left_donor_adjoint_base = split_index * S
            right_donor_adjoint_base = (split_index + tl.num_programs(0)) * S
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
        log_pD = tl.load(log_pD_arg)
        log_pS = tl.load(log_pS_arg)
    elif PARAM_LAYOUT == 0:
        log_pD = log_pD_arg
        log_pS = log_pS_arg
    elif PARAM_LAYOUT == 2:
        log_pD = tl.load(log_pD_arg + parent_family)
        log_pS = tl.load(log_pS_arg + parent_family)
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
    scalar_lane_offset = tl.arange(0, 1)
    if OUTPUT_DONOR_ADJOINT:
        left_pibar_row_max = tl.load(pibar_row_max_ptr + left_clade_row)
        right_pibar_row_max = tl.load(pibar_row_max_ptr + right_clade_row)
        left_donor_side_nonzero = tl.full((1,), value=0, dtype=tl.int32)
        right_donor_side_nonzero = tl.full((1,), value=0, dtype=tl.int32)
        left_donor_adjoint_abs_sum = tl.zeros((1,), dtype=DTYPE)
        right_donor_adjoint_abs_sum = tl.zeros((1,), dtype=DTYPE)

    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        valid_mask = s_offs < S
        mask = valid_mask & parent_active

        left_reconciliation_log_likelihood = tl.load(Pi_star_ptr + left_clade_base + s_offs, mask=mask, other=NEG_LARGE)
        right_reconciliation_log_likelihood = tl.load(Pi_star_ptr + right_clade_base + s_offs, mask=mask, other=NEG_LARGE)
        left_transfer_complement_log_likelihood = tl.load(Pibar_star_ptr + left_transfer_complement_base + s_offs, mask=mask, other=NEG_LARGE)
        right_transfer_complement_log_likelihood = tl.load(Pibar_star_ptr + right_transfer_complement_base + s_offs, mask=mask, other=NEG_LARGE)

        c1 = tl.load(species_child1_ptr + s_offs, mask=mask, other=0)
        c2 = tl.load(species_child2_ptr + s_offs, mask=mask, other=0)
        c1_valid = (c1 < S) & mask
        c2_valid = (c2 < S) & mask
        left_child1_reconciliation_log_likelihood = tl.load(Pi_star_ptr + left_clade_base + c1, mask=c1_valid, other=NEG_LARGE)
        left_child2_reconciliation_log_likelihood = tl.load(Pi_star_ptr + left_clade_base + c2, mask=c2_valid, other=NEG_LARGE)
        right_child1_reconciliation_log_likelihood = tl.load(Pi_star_ptr + right_clade_base + c1, mask=c1_valid, other=NEG_LARGE)
        right_child2_reconciliation_log_likelihood = tl.load(Pi_star_ptr + right_clade_base + c2, mask=c2_valid, other=NEG_LARGE)

        parent_reconciliation_log_likelihood = tl.load(Pi_star_ptr + parent_clade_base + s_offs, mask=mask, other=NEG_LARGE)
        parent_adjoint = tl.load(v_k_ptr + parent_adjoint_base + s_offs, mask=mask, other=0.0)

        if PARAM_LAYOUT == 1:
            duplication_log_probability = tl.load(log_pD_arg + s_offs, mask=valid_mask, other=NEG_LARGE)
            speciation_log_probability = tl.load(log_pS_arg + s_offs, mask=valid_mask, other=NEG_LARGE)
        elif PARAM_LAYOUT == 3:
            param_base = parent_family * S
            duplication_log_probability = tl.load(log_pD_arg + param_base + s_offs, mask=valid_mask, other=NEG_LARGE)
            speciation_log_probability = tl.load(log_pS_arg + param_base + s_offs, mask=valid_mask, other=NEG_LARGE)
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
            left_reconciliation_vjp = tl.load(left_reconciliation_vjp_ptr, mask=mask, other=0.0)
            right_reconciliation_vjp = tl.load(right_reconciliation_vjp_ptr, mask=mask, other=0.0)
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
                left_max_transfer = tl.load(max_transfer_ptr + left_family * S + s_offs, mask=valid_mask, other=0.0)
                right_max_transfer = tl.load(max_transfer_ptr + right_family * S + s_offs, mask=valid_mask, other=0.0)
            else:
                max_transfer = tl.load(max_transfer_ptr + s_offs, mask=valid_mask, other=0.0)
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
        if OUTPUT_SIDE_ACTIVE:
            if SIDE_ACTIVE_THRESHOLD_ENABLED:
                threshold = tl.load(side_active_threshold_ptr)
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

            left_child1_reconciliation_log_likelihood = tl.load(Pi_star_ptr + left_clade_base + c1, mask=c1_valid, other=NEG_LARGE)
            left_child2_reconciliation_log_likelihood = tl.load(Pi_star_ptr + left_clade_base + c2, mask=c2_valid, other=NEG_LARGE)
            right_child1_reconciliation_log_likelihood = tl.load(Pi_star_ptr + right_clade_base + c1, mask=c1_valid, other=NEG_LARGE)
            right_child2_reconciliation_log_likelihood = tl.load(Pi_star_ptr + right_clade_base + c2, mask=c2_valid, other=NEG_LARGE)

            parent_reconciliation_log_likelihood = tl.load(Pi_star_ptr + parent_clade_base + s_offs, mask=mask, other=NEG_LARGE)
            parent_adjoint = tl.load(v_k_ptr + parent_adjoint_base + s_offs, mask=mask, other=0.0)

            if PARAM_LAYOUT == 1:
                speciation_log_probability = tl.load(log_pS_arg + s_offs, mask=valid_mask, other=NEG_LARGE)
            elif PARAM_LAYOUT == 3:
                speciation_log_probability = tl.load(log_pS_arg + parent_family * S + s_offs, mask=valid_mask, other=NEG_LARGE)
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
                pi_l_c1_cur = tl.load(pi_l_c1_out, mask=c1_valid, other=0.0)
                pi_r_c1_cur = tl.load(pi_r_c1_out, mask=c1_valid, other=0.0)
                pi_r_c2_cur = tl.load(pi_r_c2_out, mask=c2_valid, other=0.0)
                pi_l_c2_cur = tl.load(pi_l_c2_out, mask=c2_valid, other=0.0)
                tl.store(pi_l_c1_out, pi_l_c1_cur + speciation_lr_event_vjp, mask=c1_valid)
                tl.store(pi_r_c1_out, pi_r_c1_cur + speciation_rl_event_vjp, mask=c1_valid)
                tl.store(pi_r_c2_out, pi_r_c2_cur + speciation_lr_event_vjp, mask=c2_valid)
                tl.store(pi_l_c2_out, pi_l_c2_cur + speciation_rl_event_vjp, mask=c2_valid)


@triton.jit
def _reduce_max_transfer_vjp_kernel(
    partial_ptr,   # [n_tiles, S]
    grad_max_transfer_ptr,   # [S]
    n_tiles,  # runtime int (per-wave partial-tile count; constexpr caused one JIT compile per value)
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
    # int64: row ranges over 2*n_ws, so row_base below can overflow int32 once
    # that count * S exceeds 2^31.
    row = tl.program_id(0).to(tl.int64)
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
        threshold = tl.load(side_active_threshold_ptr)
        tl.store(side_active_ptr + row + lane, row_abssum > threshold)
    else:
        tl.store(side_active_ptr + row + lane, row_absmax != 0.0)


# ``n_ws`` is the wave's split count and changes every launch; keeping it out of the
# specialization key avoids one JIT compile per divisibility state (see README.md).
@triton.jit(do_not_specialize=["n_ws"])
def _accumulate_transfer_subtree_vjp_kernel(
    Pi_star_ptr,          # [C, S]
    receiver_log_probs_ptr, # [S]
    donor_adjoint_ptr,         # [2 * n_ws, S], initial subtree values donor_adjoint
    internal_node_own_ptr, # [2 * n_ws, N_COMPACT_NODES] scratch: each internal node's own term
    side_active_ptr,      # optional [2 * n_ws] bool exact-zero side skip mask
    split_left_rows_ptr,               # [n_ws]
    split_right_rows_ptr,               # [n_ws]
    reduce_idx_ptr,       # [n_ws], used with active_mask_ptr when enabled
    active_mask_ptr,      # optional [W] bool parent row activity mask
    pibar_row_max_ptr,    # [C], Pi-row max from forward uniform Pibar
    species_parent_ptr,   # [S] int32, each species' parent, negative at the root
    compact_level_ptr,    # [N_LEVELS + 1]
    compact_level_parent_ptr, # [total internal nodes across levels]
    compact_level_child1_ptr, # [total internal nodes across levels]
    compact_level_child2_ptr, # [total internal nodes across levels]
    accumulated_rhs_ptr,  # [C, S], updated atomically
    grad_receiver_log_probs_ptr, # optional [S], updated atomically
    n_ws,  # runtime int, NOT constexpr: it is the wave's split count and differs for every wave; a
           # constexpr here forced one Triton compile per distinct value (~20k cached variants,
           # minutes of JIT per gradient at 5000 families). Only used for index arithmetic.
    S: tl.constexpr,
    stride_C: tl.constexpr,
    BLOCK_S: tl.constexpr,
    N_LEVELS: tl.constexpr,
    N_COMPACT_NODES: tl.constexpr,
    USE_ACTIVE_MASK: tl.constexpr,
    USE_SIDE_ACTIVE: tl.constexpr,
    ACCUM_RECEIVER_GRAD: tl.constexpr,
    USE_RECEIVER_WEIGHTS: tl.constexpr,
    DTYPE: tl.constexpr,
):
    """Apply the transfer-complement VJP using compact subtree reductions.

    Receiver s takes donor mass from every donor OUTSIDE s's own subtree, so this kernel needs,
    per species s, this split side's donor adjoint summed off s's subtree. That used to be
    ``row total - subtree sum``. Each donor's adjoint carries the reciprocal of that donor's own
    valid receiver mass, so for species hanging under the lane holding the row's mass it is
    astronomically large (2**depth); the row total is then dominated by those terms, and for the
    dominant lane -- whose subtree holds all of them -- the subtraction cancels to rounding noise
    of that astronomical size. Measured on a 1007-species Coleman family at the loss-rate cap, the
    gradient came out 1e8 times too large. So the off-subtree sum is built by ADDITION only:
    subtree sums bottom-up as before, then a top-down walk over the same level tables,
    off-subtree(child) = off-subtree(parent) + parent's own term + sibling's subtree sum, each
    child's off-subtree sum overwriting its no-longer-needed subtree sum. The bottom-up walk
    overwrites each internal node's own term with its subtree sum, so it parks that own term in
    ``internal_node_own_ptr`` first, indexed exactly like the compact level tables: recovering it
    as ``subtree(parent) - subtree(c1) - subtree(c2)`` would be the same cancelling subtraction.
    """
    NEG_LARGE: tl.constexpr = -float("inf")

    # int64: row ranges over 2*n_ws, so row_base below can overflow int32 once
    # that count * S exceeds 2^31.
    row = tl.program_id(0).to(tl.int64)
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
    row_max = tl.load(pibar_row_max_ptr + child)
    row_max_safe = tl.where(row_max != NEG_LARGE, row_max, tl.zeros_like(row_max))
    own_base = internal_node_own_ptr + row * N_COMPACT_NODES

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
            tl.store(own_base + node_offs, parent_val, mask=parent_valid)
            tl.store(donor_adjoint_ptr + row_base + parent, parent_val + c1_val + c2_val, mask=parent_valid)
            p_start += BLOCK_S
        tl.debug_barrier()

    # The root's subtree is the whole tree: nothing lies off it. Every other species is some
    # internal node's child, so the walk below writes it exactly once.
    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        valid_mask = s_offs < S
        species_parent = tl.load(species_parent_ptr + s_offs, mask=valid_mask, other=0)
        is_root_lane = valid_mask & (species_parent < 0) & row_active
        tl.store(
            donor_adjoint_ptr + row_base + s_offs,
            tl.zeros([BLOCK_S], dtype=DTYPE),
            mask=is_root_lane,
        )
    tl.debug_barrier()
    for level_index in range(0, N_LEVELS):
        level = N_LEVELS - 1 - level_index
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
            c1_valid = parent_valid & (c1 >= 0) & (c1 < S)
            c2_valid = parent_valid & (c2 >= 0) & (c2 < S)

            parent_off_subtree = tl.load(
                donor_adjoint_ptr + row_base + parent, mask=parent_valid, other=0.0
            )
            parent_own = tl.load(own_base + node_offs, mask=parent_valid, other=0.0)
            c1_subtree = tl.load(donor_adjoint_ptr + row_base + c1, mask=c1_valid, other=0.0)
            c2_subtree = tl.load(donor_adjoint_ptr + row_base + c2, mask=c2_valid, other=0.0)
            tl.store(
                donor_adjoint_ptr + row_base + c1,
                parent_off_subtree + parent_own + c2_subtree,
                mask=c1_valid,
            )
            tl.store(
                donor_adjoint_ptr + row_base + c2,
                parent_off_subtree + parent_own + c1_subtree,
                mask=c2_valid,
            )
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
        off_subtree_donor_adjoint = tl.load(
            donor_adjoint_ptr + row_base + s_offs, mask=mask, other=0.0
        )
        transfer_complement_vjp = receiver_mass * off_subtree_donor_adjoint
        tl.atomic_add(accumulated_rhs_ptr + pi_base + s_offs, transfer_complement_vjp, sem="relaxed", mask=mask)
        if ACCUM_RECEIVER_GRAD:
            tl.atomic_add(
                grad_receiver_log_probs_ptr + s_offs,
                transfer_complement_vjp,
                sem="relaxed",
                mask=mask,
            )
