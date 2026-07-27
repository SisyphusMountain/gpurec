"""Wave-step tangent kernels; see ``docs/latex/kernel_mathematics.tex``."""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

from gpurec.core.kernels.pi_forward import (
    _prepare_wave_launch,
    _tl_float_dtype,
    _validate_offset_tensor,
    _validate_residual_tensors,
)

# Launch tuning for the reconciliation-likelihood JVP. Tuned on the representative 666x80 fixture
# (S=1331, BLOCK_S=2048): num_warps=4 is ~7.4% faster on the reconciliation_event_scaled_mass HVP than the old default of 8
# (sweep {2,4,8,16,32}; 2 spills catastrophically, 8/16/32 slower). The win is NOT from occupancy —
# both 4 and 8 pin to 8 active warps/SM (16.67%), register+shared-limited — but from more elements
# per thread (16 vs 8 → better ILP) and 2 resident blocks/SM (vs 1) hiding the cold-DRAM latency.
# Bit-identical (hvp FD gate unchanged); neutral on small (S=119). Override per run via env.
_WST_NUM_WARPS = int(os.environ.get("NEWTON_WST_NUM_WARPS", "4"))


@triton.jit
def _update_reconciliation_likelihood_jvp_kernel(
    Pi_ptr, dPi_ptr,
    Pi_offset_ptr,
    ws, pi_ws,
    max_transfer_ptr, dmax_transfer_ptr,
    duplication_loss_const_ptr, d_duplication_loss_const_ptr,
    Ebar_ptr, dEbar_ptr,
    E_ptr, dE_ptr,
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
    dPi_new_ptr,
    dPibar_out_ptr,
    S: tl.constexpr,
    stride: tl.constexpr,
    CONST_ROW_STRIDE: tl.constexpr,
    BLOCK_S: tl.constexpr,
    MAX_ANCESTOR_DEPTH: tl.constexpr,
    USE_LEAF_INDEX: tl.constexpr,
    STORE_PIBAR: tl.constexpr,
    USE_RECEIVER_WEIGHTS: tl.constexpr,
    DTYPE: tl.constexpr,
):
    NEG = -float("inf")
    # int64: w ranges over the whole batch's clade rows, so the *stride
    # multiplies below can overflow int32 once total_clades * S exceeds 2^31.
    w = tl.program_id(0).to(tl.int64)
    pi_base = (pi_ws + w) * stride
    out_base = w * stride
    global_base = (ws + w) * stride
    family = tl.load(family_idx_ptr + ws + w)
    const_base = family * CONST_ROW_STRIDE

    s_offs = tl.arange(0, BLOCK_S)
    mask = s_offs < S
    zero = tl.zeros([BLOCK_S], dtype=DTYPE)

    # Offsets are gauges, not differentiable state. Primal weights are formed
    # in the input Pi row's frame; dPi/dPibar remain derivatives of the
    # represented absolute rows and therefore need no offset tangent.
    pi_offset = tl.load(Pi_offset_ptr + pi_ws + w)

    reconciliation_log_likelihood = tl.load(Pi_ptr + pi_base + s_offs, mask=mask, other=NEG)
    d_reconciliation_log_likelihood = tl.load(dPi_ptr + pi_base + s_offs, mask=mask, other=0.0)
    if USE_RECEIVER_WEIGHTS:
        receiver_log_probability = tl.load(receiver_log_probs_ptr + s_offs, mask=mask, other=NEG)
        d_receiver_log_probability = tl.load(
            dreceiver_log_probs_ptr + s_offs, mask=mask, other=0.0
        )
        receiver_weighted_reconciliation_log_likelihood = receiver_log_probability + reconciliation_log_likelihood
        d_receiver_weighted_reconciliation_log_likelihood = d_reconciliation_log_likelihood + d_receiver_log_probability
    else:
        receiver_weighted_reconciliation_log_likelihood = reconciliation_log_likelihood
        d_receiver_weighted_reconciliation_log_likelihood = d_reconciliation_log_likelihood
    row_max = tl.max(receiver_weighted_reconciliation_log_likelihood, axis=0)
    row_max_safe = tl.where(row_max != NEG, row_max, tl.zeros([1], dtype=DTYPE))
    receiver_mass = tl.where(
        mask, tl.exp2(receiver_weighted_reconciliation_log_likelihood - row_max_safe), zero
    )
    total_receiver_mass = tl.sum(receiver_mass, axis=0)
    total_receiver_tangent_numerator = tl.sum(
        tl.where(mask, receiver_mass * d_receiver_weighted_reconciliation_log_likelihood, zero), axis=0
    )

    ancestor_species = s_offs
    excluded_ancestor_mass = zero
    excluded_ancestor_tangent_numerator = zero
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
                ancestor_valid, tl.exp2(ancestor_receiver_log_probability + ancestor_reconciliation_log_likelihood - row_max_safe), zero
            )
            d_ancestor_weighted_reconciliation_log_likelihood = d_ancestor_reconciliation_log_likelihood + d_ancestor_receiver_log_probability
        else:
            ancestor_receiver_mass = tl.where(
                ancestor_valid, tl.exp2(ancestor_reconciliation_log_likelihood - row_max_safe), zero
            )
            d_ancestor_weighted_reconciliation_log_likelihood = d_ancestor_reconciliation_log_likelihood
        excluded_ancestor_mass += ancestor_receiver_mass
        excluded_ancestor_tangent_numerator += (
            ancestor_receiver_mass * d_ancestor_weighted_reconciliation_log_likelihood
        )
        ancestor_species = tl.load(species_parent_ptr + ancestor_species, mask=ancestor_valid, other=-1).to(tl.int32)

    const_offsets = const_base + s_offs
    max_transfer = tl.load(max_transfer_ptr + const_offsets, mask=mask, other=0.0)
    d_max_transfer = tl.load(dmax_transfer_ptr + const_offsets, mask=mask, other=0.0)
    valid_receiver_mass = total_receiver_mass - excluded_ancestor_mass
    has_valid_receiver_mass = valid_receiver_mass > 0.0
    safe_valid_receiver_mass = tl.where(
        has_valid_receiver_mass,
        valid_receiver_mass,
        tl.full([BLOCK_S], 1.0, DTYPE),
    )
    transfer_complement_log_likelihood = tl.where(
        has_valid_receiver_mass,
        tl.log2(safe_valid_receiver_mass) + row_max + max_transfer,
        NEG,
    )
    d_transfer_complement_log_likelihood = tl.where(
        has_valid_receiver_mass,
        (
            total_receiver_tangent_numerator
            - excluded_ancestor_tangent_numerator
        )
        / safe_valid_receiver_mass
        + d_max_transfer,
        zero,
    )

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
    c1_valid = mask & (c1 < S)
    c2_valid = mask & (c2 < S)
    reconciliation_child1_log_likelihood = tl.where(
        c1_valid, tl.gather(reconciliation_log_likelihood, tl.where(c1_valid, c1, 0), axis=0), NEG
    )
    reconciliation_child2_log_likelihood = tl.where(
        c2_valid, tl.gather(reconciliation_log_likelihood, tl.where(c2_valid, c2, 0), axis=0), NEG
    )
    d_reconciliation_child1_log_likelihood = tl.where(
        c1_valid, tl.gather(d_reconciliation_log_likelihood, tl.where(c1_valid, c1, 0), axis=0), zero
    )
    d_reconciliation_child2_log_likelihood = tl.where(
        c2_valid, tl.gather(d_reconciliation_log_likelihood, tl.where(c2_valid, c2, 0), axis=0), zero
    )

    duplication_loss_log_term = duplication_loss_const + reconciliation_log_likelihood
    d_duplication_loss_log_term = d_duplication_loss_const + d_reconciliation_log_likelihood
    transfer_loss_log_term = reconciliation_log_likelihood + extinction_complement_log_probability
    d_transfer_loss_log_term = d_reconciliation_log_likelihood + d_extinction_complement_log_probability
    transfer_log_term = transfer_complement_log_likelihood + extinction_log_probability
    d_transfer_log_term = d_transfer_complement_log_likelihood + d_extinction_log_probability
    speciation_child1_log_term = speciation_child1_const + reconciliation_child1_log_likelihood
    d_speciation_child1_log_term = d_speciation_child1_const + d_reconciliation_child1_log_likelihood
    speciation_child2_log_term = speciation_child2_const + reconciliation_child2_log_likelihood
    d_speciation_child2_log_term = d_speciation_child2_const + d_reconciliation_child2_log_likelihood
    if USE_LEAF_INDEX:
        leaf_species = tl.load(leaf_species_ptr + ws + w)
        leaf_hit = mask & (leaf_species == s_offs)
        leaf_logp = tl.load(leaf_logp_ptr + family * S + s_offs, mask=mask, other=NEG)
        d_leaf_logp = tl.load(
            d_leaf_logp_ptr + family * S + s_offs, mask=mask, other=0.0
        )
        leaf_logp = leaf_logp + (0.0 - pi_offset).to(DTYPE)
        leaf_observation_log_term = tl.where(leaf_hit, leaf_logp, NEG)
        d_leaf_observation_log_term = tl.where(leaf_hit, d_leaf_logp, zero)
    else:
        leaf_observation_log_term = tl.full([BLOCK_S], NEG, dtype=DTYPE)
        d_leaf_observation_log_term = zero

    logsumexp_max = tl.maximum(
        tl.maximum(tl.maximum(duplication_loss_log_term, transfer_loss_log_term),
                   tl.maximum(transfer_log_term, speciation_child1_log_term)),
        tl.maximum(speciation_child2_log_term, leaf_observation_log_term),
    )
    if has_splits:
        gene_split_log_likelihood = tl.load(
            gene_split_log_likelihood_ptr + out_base + s_offs,
            mask=mask,
            other=NEG,
        )
        gene_split_offset = tl.load(gene_split_offset_ptr + w)
        gene_split_log_likelihood = gene_split_log_likelihood + (gene_split_offset - pi_offset).to(DTYPE)
        d_gene_split_log_likelihood = tl.load(
            d_gene_split_log_likelihood_ptr + out_base + s_offs,
            mask=mask,
            other=0.0,
        )
        logsumexp_max = tl.maximum(logsumexp_max, gene_split_log_likelihood)
    logsumexp_max_safe = tl.where(logsumexp_max != NEG, logsumexp_max, zero)
    duplication_loss_mass = tl.exp2(duplication_loss_log_term - logsumexp_max_safe)
    transfer_loss_mass = tl.exp2(transfer_loss_log_term - logsumexp_max_safe)
    transfer_mass = tl.exp2(transfer_log_term - logsumexp_max_safe)
    speciation_child1_mass = tl.exp2(speciation_child1_log_term - logsumexp_max_safe)
    speciation_child2_mass = tl.exp2(speciation_child2_log_term - logsumexp_max_safe)
    leaf_observation_mass = tl.exp2(leaf_observation_log_term - logsumexp_max_safe)
    reconciliation_event_scaled_mass = (
        duplication_loss_mass + transfer_loss_mass + transfer_mass
        + speciation_child1_mass + speciation_child2_mass + leaf_observation_mass
    )
    event_tangent_numerator = (
        duplication_loss_mass * d_duplication_loss_log_term
        + transfer_loss_mass * d_transfer_loss_log_term
        + transfer_mass * d_transfer_log_term
        + speciation_child1_mass * d_speciation_child1_log_term
        + speciation_child2_mass * d_speciation_child2_log_term
        + leaf_observation_mass * d_leaf_observation_log_term
    )
    if has_splits:
        gene_split_mass = tl.exp2(gene_split_log_likelihood - logsumexp_max_safe)
        reconciliation_event_scaled_mass += gene_split_mass
        event_tangent_numerator += gene_split_mass * d_gene_split_log_likelihood
    updated_reconciliation_log_likelihood = tl.log2(reconciliation_event_scaled_mass) + logsumexp_max
    inverse_reconciliation_event_scaled_mass = tl.where(
        reconciliation_event_scaled_mass > 0.0, 1.0 / reconciliation_event_scaled_mass, zero
    )
    d_updated_reconciliation_log_likelihood = tl.where(
        mask & (updated_reconciliation_log_likelihood != NEG),
        event_tangent_numerator * inverse_reconciliation_event_scaled_mass,
        zero,
    )
    tl.store(dPi_new_ptr + out_base + s_offs, d_updated_reconciliation_log_likelihood, mask=mask)

    if STORE_PIBAR:
        tl.store(dPibar_out_ptr + global_base + s_offs, d_transfer_complement_log_likelihood, mask=mask)


@triton.jit
def _apply_reconciliation_self_loop_jvp_iterations_kernel(
    Pi_ptr, dPi_ptr,
    Pi_offset_ptr,
    ws, pi_ws,
    max_transfer_ptr, dmax_transfer_ptr,
    duplication_loss_const_ptr, d_duplication_loss_const_ptr,
    Ebar_ptr, dEbar_ptr,
    E_ptr, dE_ptr,
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
    dPi_new_ptr,
    dPibar_out_ptr,
    n_iters,
    S: tl.constexpr,
    stride: tl.constexpr,
    CONST_ROW_STRIDE: tl.constexpr,
    BLOCK_S: tl.constexpr,
    MAX_ANCESTOR_DEPTH: tl.constexpr,
    USE_LEAF_INDEX: tl.constexpr,
    STORE_PIBAR: tl.constexpr,
    USE_RECEIVER_WEIGHTS: tl.constexpr,
    DTYPE: tl.constexpr,
):
    """Fuse fixed-count wave-tangent self-loop iterations."""
    NEG = -float("inf")
    # int64: w ranges over the whole batch's clade rows, so the *stride
    # multiplies below can overflow int32 once total_clades * S exceeds 2^31.
    w = tl.program_id(0).to(tl.int64)
    pi_base = (pi_ws + w) * stride
    out_base = w * stride
    global_base = (ws + w) * stride
    family = tl.load(family_idx_ptr + ws + w)
    const_base = family * CONST_ROW_STRIDE

    s_offs = tl.arange(0, BLOCK_S)
    mask = s_offs < S
    zero = tl.zeros([BLOCK_S], dtype=DTYPE)

    pi_offset = tl.load(Pi_offset_ptr + pi_ws + w)

    # ---- invariant setup (computed once) ----
    reconciliation_log_likelihood = tl.load(Pi_ptr + pi_base + s_offs, mask=mask, other=NEG)
    if USE_RECEIVER_WEIGHTS:
        receiver_log_probability = tl.load(receiver_log_probs_ptr + s_offs, mask=mask, other=NEG)
        d_receiver_log_probability = tl.load(
            dreceiver_log_probs_ptr + s_offs, mask=mask, other=0.0
        )
        receiver_weighted_reconciliation_log_likelihood = receiver_log_probability + reconciliation_log_likelihood
    else:
        d_receiver_log_probability = zero
        receiver_weighted_reconciliation_log_likelihood = reconciliation_log_likelihood
    row_max = tl.max(receiver_weighted_reconciliation_log_likelihood, axis=0)
    row_max_safe = tl.where(row_max != NEG, row_max, tl.zeros([1], dtype=DTYPE))
    receiver_mass = tl.where(mask, tl.exp2(receiver_weighted_reconciliation_log_likelihood - row_max_safe), zero)
    total_receiver_mass = tl.sum(receiver_mass, axis=0)

    # primal ancestor sum (invariant) -> valid_receiver_mass
    ancestor_species = s_offs
    excluded_ancestor_mass = zero
    for _ in range(0, MAX_ANCESTOR_DEPTH):
        ancestor_valid = mask & (ancestor_species >= 0) & (ancestor_species < S)
        ancestor_receiver_mass = tl.where(
            ancestor_valid, tl.gather(receiver_mass, tl.where(ancestor_valid, ancestor_species, 0), axis=0), zero
        )
        excluded_ancestor_mass += ancestor_receiver_mass
        ancestor_species = tl.load(species_parent_ptr + ancestor_species, mask=ancestor_valid, other=-1).to(tl.int32)

    const_offsets = const_base + s_offs
    max_transfer = tl.load(max_transfer_ptr + const_offsets, mask=mask, other=0.0)
    d_max_transfer = tl.load(dmax_transfer_ptr + const_offsets, mask=mask, other=0.0)
    valid_receiver_mass = total_receiver_mass - excluded_ancestor_mass
    has_valid_receiver_mass = valid_receiver_mass > 0.0
    safe_valid_receiver_mass = tl.where(
        has_valid_receiver_mass,
        valid_receiver_mass,
        tl.full([BLOCK_S], 1.0, DTYPE),
    )
    inverse_valid_receiver_mass = tl.where(
        has_valid_receiver_mass, 1.0 / safe_valid_receiver_mass, zero
    )
    transfer_complement_log_likelihood = tl.where(
        has_valid_receiver_mass,
        tl.log2(safe_valid_receiver_mass) + row_max + max_transfer,
        NEG,
    )

    duplication_loss_const = tl.load(duplication_loss_const_ptr + const_offsets, mask=mask, other=NEG)
    d_duplication_loss_const = tl.load(d_duplication_loss_const_ptr + const_offsets, mask=mask, other=0.0)
    extinction_complement_log_probability = tl.load(Ebar_ptr + const_offsets, mask=mask, other=NEG)
    d_extinction_complement_log_probability = tl.load(dEbar_ptr + const_offsets, mask=mask, other=0.0)
    extinction_log_probability = tl.load(E_ptr + const_offsets, mask=mask, other=NEG)
    d_extinction_log_probability = tl.load(dE_ptr + const_offsets, mask=mask, other=0.0)
    speciation_child1_const = tl.load(speciation_child1_const_ptr + const_offsets, mask=mask, other=NEG)
    d_speciation_child1_const = tl.load(d_speciation_child1_const_ptr + const_offsets, mask=mask, other=0.0)
    speciation_child2_const = tl.load(speciation_child2_const_ptr + const_offsets, mask=mask, other=NEG)
    d_speciation_child2_const = tl.load(d_speciation_child2_const_ptr + const_offsets, mask=mask, other=0.0)

    c1 = tl.load(species_child1_ptr + s_offs, mask=mask, other=S)
    c2 = tl.load(species_child2_ptr + s_offs, mask=mask, other=S)
    c1_valid = mask & (c1 < S)
    c2_valid = mask & (c2 < S)
    reconciliation_child1_log_likelihood = tl.where(c1_valid, tl.gather(reconciliation_log_likelihood, tl.where(c1_valid, c1, 0), axis=0), NEG)
    reconciliation_child2_log_likelihood = tl.where(c2_valid, tl.gather(reconciliation_log_likelihood, tl.where(c2_valid, c2, 0), axis=0), NEG)

    duplication_loss_log_term = duplication_loss_const + reconciliation_log_likelihood
    transfer_loss_log_term = reconciliation_log_likelihood + extinction_complement_log_probability
    transfer_log_term = transfer_complement_log_likelihood + extinction_log_probability
    speciation_child1_log_term = speciation_child1_const + reconciliation_child1_log_likelihood
    speciation_child2_log_term = speciation_child2_const + reconciliation_child2_log_likelihood
    if USE_LEAF_INDEX:
        leaf_species = tl.load(leaf_species_ptr + ws + w)
        leaf_hit = mask & (leaf_species == s_offs)
        leaf_logp = tl.load(leaf_logp_ptr + family * S + s_offs, mask=mask, other=NEG)
        d_leaf_logp = tl.load(
            d_leaf_logp_ptr + family * S + s_offs, mask=mask, other=0.0
        )
        leaf_logp = leaf_logp + (0.0 - pi_offset).to(DTYPE)
        leaf_observation_log_term = tl.where(leaf_hit, leaf_logp, NEG)
        d_leaf_observation_log_term = tl.where(leaf_hit, d_leaf_logp, zero)
    else:
        leaf_observation_log_term = tl.full([BLOCK_S], NEG, dtype=DTYPE)
        d_leaf_observation_log_term = zero

    logsumexp_max = tl.maximum(
        tl.maximum(tl.maximum(duplication_loss_log_term, transfer_loss_log_term),
                   tl.maximum(transfer_log_term, speciation_child1_log_term)),
        tl.maximum(speciation_child2_log_term, leaf_observation_log_term),
    )
    if has_splits:
        gene_split_log_likelihood = tl.load(gene_split_log_likelihood_ptr + out_base + s_offs, mask=mask, other=NEG)
        gene_split_offset = tl.load(gene_split_offset_ptr + w)
        gene_split_log_likelihood = gene_split_log_likelihood + (gene_split_offset - pi_offset).to(DTYPE)
        d_gene_split_log_likelihood = tl.load(d_gene_split_log_likelihood_ptr + out_base + s_offs, mask=mask, other=0.0)
        logsumexp_max = tl.maximum(logsumexp_max, gene_split_log_likelihood)
    logsumexp_max_safe = tl.where(logsumexp_max != NEG, logsumexp_max, zero)
    duplication_loss_mass = tl.exp2(duplication_loss_log_term - logsumexp_max_safe)
    transfer_loss_mass = tl.exp2(transfer_loss_log_term - logsumexp_max_safe)
    transfer_mass = tl.exp2(transfer_log_term - logsumexp_max_safe)
    speciation_child1_mass = tl.exp2(speciation_child1_log_term - logsumexp_max_safe)
    speciation_child2_mass = tl.exp2(speciation_child2_log_term - logsumexp_max_safe)
    leaf_observation_mass = tl.exp2(leaf_observation_log_term - logsumexp_max_safe)
    reconciliation_event_scaled_mass = (
        duplication_loss_mass + transfer_loss_mass + transfer_mass
        + speciation_child1_mass + speciation_child2_mass + leaf_observation_mass
    )
    gene_split_tangent_numerator = zero
    if has_splits:
        gene_split_mass = tl.exp2(gene_split_log_likelihood - logsumexp_max_safe)
        reconciliation_event_scaled_mass += gene_split_mass
        gene_split_tangent_numerator = gene_split_mass * d_gene_split_log_likelihood
    updated_reconciliation_log_likelihood = tl.log2(reconciliation_event_scaled_mass) + logsumexp_max
    inverse_reconciliation_event_scaled_mass = tl.where(reconciliation_event_scaled_mass > 0.0, 1.0 / reconciliation_event_scaled_mass, zero)
    valid = mask & (updated_reconciliation_log_likelihood != NEG)

    # ---- tangent self-loop (register-resident; only d_reconciliation_log_likelihood varies) ----
    d_reconciliation_log_likelihood = tl.load(dPi_ptr + pi_base + s_offs, mask=mask, other=0.0)
    d_transfer_complement_log_likelihood = zero
    for _it in range(0, n_iters):
        d_receiver_weighted_reconciliation_log_likelihood = d_reconciliation_log_likelihood + d_receiver_log_probability
        total_receiver_tangent_numerator = tl.sum(
            tl.where(mask, receiver_mass * d_receiver_weighted_reconciliation_log_likelihood, zero), axis=0
        )
        ancestor_species = s_offs
        excluded_ancestor_tangent_numerator = zero
        for _ in range(0, MAX_ANCESTOR_DEPTH):
            ancestor_valid = mask & (ancestor_species >= 0) & (ancestor_species < S)
            ancestor_receiver_mass = tl.where(
                ancestor_valid, tl.gather(receiver_mass, tl.where(ancestor_valid, ancestor_species, 0), axis=0), zero
            )
            d_ancestor_reconciliation_log_likelihood = tl.where(ancestor_valid, tl.gather(d_reconciliation_log_likelihood, tl.where(ancestor_valid, ancestor_species, 0), axis=0), zero)
            d_ancestor_receiver_log_probability = tl.where(ancestor_valid, tl.gather(d_receiver_log_probability, tl.where(ancestor_valid, ancestor_species, 0), axis=0), zero)
            excluded_ancestor_tangent_numerator += ancestor_receiver_mass * (d_ancestor_reconciliation_log_likelihood + d_ancestor_receiver_log_probability)
            ancestor_species = tl.load(species_parent_ptr + ancestor_species, mask=ancestor_valid, other=-1).to(tl.int32)
        d_transfer_complement_log_likelihood = tl.where(
            has_valid_receiver_mass,
            (
                total_receiver_tangent_numerator
                - excluded_ancestor_tangent_numerator
            )
            * inverse_valid_receiver_mass
            + d_max_transfer,
            zero,
        )
        d_reconciliation_child1_log_likelihood = tl.where(c1_valid, tl.gather(d_reconciliation_log_likelihood, tl.where(c1_valid, c1, 0), axis=0), zero)
        d_reconciliation_child2_log_likelihood = tl.where(c2_valid, tl.gather(d_reconciliation_log_likelihood, tl.where(c2_valid, c2, 0), axis=0), zero)
        d_duplication_loss_log_term = d_duplication_loss_const + d_reconciliation_log_likelihood
        d_transfer_loss_log_term = d_reconciliation_log_likelihood + d_extinction_complement_log_probability
        d_transfer_log_term = d_transfer_complement_log_likelihood + d_extinction_log_probability
        d_speciation_child1_log_term = d_speciation_child1_const + d_reconciliation_child1_log_likelihood
        d_speciation_child2_log_term = d_speciation_child2_const + d_reconciliation_child2_log_likelihood
        event_tangent_numerator = (
            duplication_loss_mass * d_duplication_loss_log_term
            + transfer_loss_mass * d_transfer_loss_log_term
            + transfer_mass * d_transfer_log_term
            + speciation_child1_mass * d_speciation_child1_log_term
            + speciation_child2_mass * d_speciation_child2_log_term
            + leaf_observation_mass * d_leaf_observation_log_term
            + gene_split_tangent_numerator
        )
        d_reconciliation_log_likelihood = tl.where(valid, event_tangent_numerator * inverse_reconciliation_event_scaled_mass, zero)

    tl.store(dPi_new_ptr + out_base + s_offs, d_reconciliation_log_likelihood, mask=mask)
    if STORE_PIBAR:
        tl.store(dPibar_out_ptr + global_base + s_offs, d_transfer_complement_log_likelihood, mask=mask)


def _prepare_wave_offsets(Pi_in, pi_offset, gene_split_offset, has_splits, W):
    """Validate the reconciliation and gene-split gauges."""
    pi_offset = _validate_offset_tensor(
        "pi_offset",
        pi_offset,
        rows=Pi_in.shape[0],
        device=Pi_in.device,
        residual_dtype=Pi_in.dtype,
    )
    if has_splits:
        if gene_split_offset is None:
            raise ValueError("gene_split_offset is required for a split wave")
        gene_split_offset = _validate_offset_tensor(
            "gene_split_offset",
            gene_split_offset,
            rows=W,
            device=Pi_in.device,
            dtype=pi_offset.dtype,
        )
    elif gene_split_offset is not None:
        raise ValueError("gene_split_offset is only valid for a split wave")
    return pi_offset, (gene_split_offset if has_splits else pi_offset)


def _validate_wave_tangent_inputs(
    Pi_in,
    *,
    dPi,
    max_transfer_mat,
    dmax_transfer,
    duplication_loss_const,
    d_duplication_loss_const,
    Ebar,
    dEbar,
    E,
    dE,
    speciation_child1_const,
    d_speciation_child1_const,
    speciation_child2_const,
    d_speciation_child2_const,
    receiver_log_probs,
    leaf_logp,
    d_leaf_logp,
    gene_split_log_likelihood,
    d_gene_split_log_likelihood,
    dreceiver_log_probs,
    dPibar_out,
):
    _validate_residual_tensors(
        Pi_in,
        dPi=dPi,
        max_transfer_mat=max_transfer_mat,
        dmax_transfer=dmax_transfer,
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
        leaf_logp=leaf_logp,
        d_leaf_logp=d_leaf_logp,
        gene_split_log_likelihood=gene_split_log_likelihood,
        d_gene_split_log_likelihood=d_gene_split_log_likelihood,
        dreceiver_log_probs=dreceiver_log_probs,
        dPibar_out=dPibar_out,
    )


def compute_wave_step_tangent_selfloop(
    Pi_in, dPi_io, ws, W, S, n_iters,
    max_transfer_mat, dmax_transfer,
    duplication_loss_const, d_duplication_loss_const,
    Ebar, dEbar, E, dE,
    speciation_child1_const, d_speciation_child1_const,
    speciation_child2_const, d_speciation_child2_const,
    receiver_log_probs, species_child1, species_child2, species_parent, max_ancestor_depth,
    gene_split_log_likelihood=None, d_gene_split_log_likelihood=None,
    *, leaf_species_idx, leaf_logp, d_leaf_logp, family_idx,
    dPibar_out=None, has_leaf_term=True, use_receiver_weights=True,
    dreceiver_log_probs=None,
    pi_offset, gene_split_offset=None,
):
    """Run fixed-count wave-tangent iterations; see the LaTeX reference."""
    has_splits = gene_split_log_likelihood is not None
    pi_offset_arg, gene_split_offset_arg = _prepare_wave_offsets(
        Pi_in, pi_offset, gene_split_offset, has_splits, W
    )
    _, const_row_stride = _prepare_wave_launch(S, duplication_loss_const)
    block_s = int(triton.next_power_of_2(S))
    store_pibar = dPibar_out is not None
    dPi_out_rows = dPi_io.narrow(0, int(ws), int(W))
    dummy = Pi_in
    dreceiver_log_probs = (
        torch.zeros_like(receiver_log_probs)
        if dreceiver_log_probs is None
        else dreceiver_log_probs.to(device=Pi_in.device, dtype=Pi_in.dtype)
        .reshape(S)
        .contiguous()
    )
    _validate_wave_tangent_inputs(
        Pi_in,
        dPi=dPi_io,
        max_transfer_mat=max_transfer_mat,
        dmax_transfer=dmax_transfer,
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
        leaf_logp=leaf_logp,
        d_leaf_logp=d_leaf_logp,
        gene_split_log_likelihood=gene_split_log_likelihood,
        d_gene_split_log_likelihood=d_gene_split_log_likelihood,
        dreceiver_log_probs=dreceiver_log_probs,
        dPibar_out=dPibar_out,
    )
    _apply_reconciliation_self_loop_jvp_iterations_kernel[(int(W),)](
        Pi_in, dPi_io,
        pi_offset_arg,
        ws, ws,
        max_transfer_mat, dmax_transfer,
        duplication_loss_const, d_duplication_loss_const,
        Ebar, dEbar, E, dE,
        speciation_child1_const, d_speciation_child1_const,
        speciation_child2_const, d_speciation_child2_const,
        receiver_log_probs, dreceiver_log_probs,
        species_child1, species_child2, species_parent,
        leaf_species_idx, leaf_logp, d_leaf_logp,
        family_idx,
        gene_split_log_likelihood if has_splits else dummy,
        d_gene_split_log_likelihood if has_splits else dummy,
        gene_split_offset_arg,
        has_splits,
        dPi_out_rows,
        dPibar_out if store_pibar else dummy,
        int(max(int(n_iters), 1)),
        S,
        stride=S,
        CONST_ROW_STRIDE=const_row_stride,
        BLOCK_S=block_s,
        MAX_ANCESTOR_DEPTH=int(max_ancestor_depth),
        USE_LEAF_INDEX=bool(has_leaf_term),
        STORE_PIBAR=bool(store_pibar),
        USE_RECEIVER_WEIGHTS=bool(use_receiver_weights),
        DTYPE=_tl_float_dtype(Pi_in.dtype),
        num_warps=_WST_NUM_WARPS,
    )


def compute_wave_step_tangent(
    Pi_in, dPi_in, dPi_out, ws, W, S,
    max_transfer_mat, dmax_transfer,
    duplication_loss_const, d_duplication_loss_const,
    Ebar, dEbar, E, dE,
    speciation_child1_const, d_speciation_child1_const,
    speciation_child2_const, d_speciation_child2_const,
    receiver_log_probs, species_child1, species_child2, species_parent, max_ancestor_depth,
    gene_split_log_likelihood=None, d_gene_split_log_likelihood=None,
    *, leaf_species_idx, leaf_logp, d_leaf_logp, family_idx,
    dPibar_out=None, has_leaf_term=True, input_ws=None,
    use_receiver_weights=True, dreceiver_log_probs=None,
    pi_offset, gene_split_offset=None,
):
    """Apply the wave-step JVP documented in the LaTeX reference."""
    has_splits = gene_split_log_likelihood is not None
    pi_offset_arg, gene_split_offset_arg = _prepare_wave_offsets(
        Pi_in, pi_offset, gene_split_offset, has_splits, W
    )
    _, const_row_stride = _prepare_wave_launch(S, duplication_loss_const)
    block_s = int(triton.next_power_of_2(S))
    store_pibar = dPibar_out is not None
    dPi_out_rows = dPi_out.narrow(0, int(ws), int(W))
    dummy = Pi_in  # unused placeholder for None pointers
    dreceiver_log_probs = (
        torch.zeros_like(receiver_log_probs)
        if dreceiver_log_probs is None
        else dreceiver_log_probs.to(device=Pi_in.device, dtype=Pi_in.dtype)
        .reshape(S)
        .contiguous()
    )
    _validate_residual_tensors(Pi_in, dPi_out=dPi_out)
    _validate_wave_tangent_inputs(
        Pi_in,
        dPi=dPi_in,
        max_transfer_mat=max_transfer_mat,
        dmax_transfer=dmax_transfer,
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
        leaf_logp=leaf_logp,
        d_leaf_logp=d_leaf_logp,
        gene_split_log_likelihood=gene_split_log_likelihood,
        d_gene_split_log_likelihood=d_gene_split_log_likelihood,
        dreceiver_log_probs=dreceiver_log_probs,
        dPibar_out=dPibar_out,
    )
    _update_reconciliation_likelihood_jvp_kernel[(int(W),)](
        Pi_in, dPi_in,
        pi_offset_arg,
        ws, ws if input_ws is None else int(input_ws),
        max_transfer_mat, dmax_transfer,
        duplication_loss_const, d_duplication_loss_const,
        Ebar, dEbar, E, dE,
        speciation_child1_const, d_speciation_child1_const,
        speciation_child2_const, d_speciation_child2_const,
        receiver_log_probs, dreceiver_log_probs,
        species_child1, species_child2, species_parent,
        leaf_species_idx, leaf_logp, d_leaf_logp,
        family_idx,
        gene_split_log_likelihood if has_splits else dummy,
        d_gene_split_log_likelihood if has_splits else dummy,
        gene_split_offset_arg,
        has_splits,
        dPi_out_rows,
        dPibar_out if store_pibar else dummy,
        S,
        stride=S,
        CONST_ROW_STRIDE=const_row_stride,
        BLOCK_S=block_s,
        MAX_ANCESTOR_DEPTH=int(max_ancestor_depth),
        USE_LEAF_INDEX=bool(has_leaf_term),
        STORE_PIBAR=bool(store_pibar),
        USE_RECEIVER_WEIGHTS=bool(use_receiver_weights),
        DTYPE=_tl_float_dtype(Pi_in.dtype),
        num_warps=_WST_NUM_WARPS,
    )
