"""E-step second-order kernels; see ``docs/latex/kernel_mathematics.tex``."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from gpurec.core.kernels.e_step import _tl_float_dtype


@triton.jit
def _stage_extinction_vjp_directional_derivative_kernel(
    E_ptr, dE_ptr, E_new_ptr, dE_new_ptr, E_s1_ptr, dE_s1_ptr, E_s2_ptr, dE_s2_ptr,
    Ebar_ptr, dEbar_ptr,
    log_pS_ptr, dlog_pS_ptr, log_pD_ptr, dlog_pD_ptr, log_pL_ptr, dlog_pL_ptr,
    receiver_log_probs_ptr, dreceiver_log_probs_ptr,
    species_parent_ptr, species_child1_ptr, species_child2_ptr,
    extinction_update_adjoint_ptr, transfer_complement_output_adjoint_ptr,
    d_grad_E_ptr, d_grad_pS_ptr, d_grad_pD_ptr, d_grad_pL_ptr,
    d_grad_max_transfer_ptr,
    receiver_mass_ptr, d_receiver_mass_ptr,
    excluded_donor_adjoint_ptr, d_excluded_donor_adjoint_ptr,
    total_donor_adjoint_ptr, d_total_donor_adjoint_ptr,
    S: tl.constexpr, BLOCK_S: tl.constexpr, MAX_ANCESTOR_DEPTH: tl.constexpr,
    USE_RECEIVER_WEIGHTS: tl.constexpr, DTYPE: tl.constexpr,
):
    LN2 = 0.6931471805599453
    g = tl.program_id(0)
    base = g * S
    offs = tl.arange(0, BLOCK_S)
    mask = offs < S
    NEG_INF = -float("inf")
    zero = tl.zeros([BLOCK_S], dtype=DTYPE)

    E = tl.load(E_ptr + base + offs, mask=mask, other=NEG_INF)
    dE = tl.load(dE_ptr + base + offs, mask=mask, other=0.0)
    E_new = tl.load(E_new_ptr + base + offs, mask=mask, other=NEG_INF)
    dE_new = tl.load(dE_new_ptr + base + offs, mask=mask, other=0.0)
    E_s1 = tl.load(E_s1_ptr + base + offs, mask=mask, other=NEG_INF)
    dE_s1 = tl.load(dE_s1_ptr + base + offs, mask=mask, other=0.0)
    E_s2 = tl.load(E_s2_ptr + base + offs, mask=mask, other=NEG_INF)
    dE_s2 = tl.load(dE_s2_ptr + base + offs, mask=mask, other=0.0)
    Ebar = tl.load(Ebar_ptr + base + offs, mask=mask, other=NEG_INF)
    dEbar = tl.load(dEbar_ptr + base + offs, mask=mask, other=0.0)
    speciation_log_probability = tl.load(log_pS_ptr + base + offs, mask=mask, other=NEG_INF)
    d_speciation_log_probability = tl.load(dlog_pS_ptr + base + offs, mask=mask, other=0.0)
    duplication_log_probability = tl.load(log_pD_ptr + base + offs, mask=mask, other=NEG_INF)
    d_duplication_log_probability = tl.load(dlog_pD_ptr + base + offs, mask=mask, other=0.0)
    loss_log_probability = tl.load(log_pL_ptr + base + offs, mask=mask, other=NEG_INF)
    d_loss_log_probability = tl.load(dlog_pL_ptr + base + offs, mask=mask, other=0.0)
    extinction_update_adjoint = tl.load(extinction_update_adjoint_ptr + base + offs, mask=mask, other=0.0)
    transfer_complement_output_adjoint = tl.load(transfer_complement_output_adjoint_ptr + base + offs, mask=mask, other=0.0)
    # Uniform and weighted receiver measures are distinct model semantics. The
    # unweighted branch represents equal receiver mass, not missing data.
    if USE_RECEIVER_WEIGHTS:
        receiver_log_probability = tl.load(
            receiver_log_probs_ptr + offs, mask=mask, other=NEG_INF
        )
        d_receiver_log_probability = tl.load(
            dreceiver_log_probs_ptr + offs, mask=mask, other=0.0
        )
        receiver_weighted_extinction_log_probability = receiver_log_probability + E
        d_receiver_weighted_extinction_log_probability = d_receiver_log_probability + dE
    else:
        receiver_weighted_extinction_log_probability = E
        d_receiver_weighted_extinction_log_probability = dE
    row_max = tl.max(receiver_weighted_extinction_log_probability, axis=0)
    row_max_safe = tl.where(row_max != NEG_INF, row_max, tl.zeros([1], dtype=DTYPE))

    # term tangents (q_k linear in extinction_update_adjoint, nonlinear in primals)
    speciation_log_term = speciation_log_probability + E_s1 + E_s2
    duplication_log_term = duplication_log_probability + 2.0 * E
    transfer_log_term = E + Ebar
    loss_log_term = loss_log_probability
    speciation_event_vjp = tl.where(
        mask, extinction_update_adjoint * tl.exp2(speciation_log_term - E_new), zero
    )
    duplication_event_vjp = tl.where(
        mask, extinction_update_adjoint * tl.exp2(duplication_log_term - E_new), zero
    )
    transfer_event_vjp = tl.where(
        mask, extinction_update_adjoint * tl.exp2(transfer_log_term - E_new), zero
    )
    loss_event_vjp = tl.where(mask, extinction_update_adjoint * tl.exp2(loss_log_term - E_new), zero)
    d_speciation_log_term = d_speciation_log_probability + dE_s1 + dE_s2
    d_duplication_log_term = d_duplication_log_probability + 2.0 * dE
    d_transfer_log_term = dE + dEbar
    d_loss_log_term = d_loss_log_probability
    d_speciation_event_vjp = LN2 * speciation_event_vjp * (
        d_speciation_log_term - dE_new
    )
    d_duplication_event_vjp = LN2 * duplication_event_vjp * (
        d_duplication_log_term - dE_new
    )
    d_transfer_event_vjp = LN2 * transfer_event_vjp * (d_transfer_log_term - dE_new)
    d_loss_event_vjp = LN2 * loss_event_vjp * (d_loss_log_term - dE_new)

    d_transfer_complement_vjp = d_transfer_event_vjp
    tl.store(d_grad_pS_ptr + base + offs, d_speciation_event_vjp, mask=mask)
    tl.store(d_grad_pD_ptr + base + offs, d_duplication_event_vjp, mask=mask)
    tl.store(d_grad_pL_ptr + base + offs, d_loss_event_vjp, mask=mask)
    tl.store(
        d_grad_max_transfer_ptr + base + offs,
        d_transfer_complement_vjp,
        mask=mask,
    )
    tl.store(
        d_grad_E_ptr + base + offs,
        2.0 * d_duplication_event_vjp + d_transfer_event_vjp,
        mask=mask,
    )

    receiver_mass = tl.where(
        mask, tl.exp2(receiver_weighted_extinction_log_probability - row_max_safe), zero
    )
    d_receiver_mass = (
        LN2 * receiver_mass * d_receiver_weighted_extinction_log_probability
    )  # row maximum frozen
    tl.store(receiver_mass_ptr + base + offs, receiver_mass, mask=mask)
    tl.store(d_receiver_mass_ptr + base + offs, d_receiver_mass, mask=mask)
    tl.store(excluded_donor_adjoint_ptr + base + offs, zero, mask=mask)
    tl.store(d_excluded_donor_adjoint_ptr + base + offs, zero, mask=mask)

    # order plain stores before overlapping atomics (same discipline as the primal backward)
    tl.debug_barrier()

    c1 = tl.load(species_child1_ptr + offs, mask=mask, other=-1)
    c2 = tl.load(species_child2_ptr + offs, mask=mask, other=-1)
    child1_valid = mask & (c1 >= 0) & (c1 < S)
    child2_valid = mask & (c2 >= 0) & (c2 < S)
    tl.atomic_add(
        d_grad_E_ptr + base + c1,
        d_speciation_event_vjp,
        sem="relaxed",
        mask=child1_valid,
    )
    tl.atomic_add(
        d_grad_E_ptr + base + c2,
        d_speciation_event_vjp,
        sem="relaxed",
        mask=child2_valid,
    )

    total_receiver_mass = tl.sum(receiver_mass, axis=0)
    d_total_receiver_mass = tl.sum(d_receiver_mass, axis=0)
    ancestor_species = offs
    excluded_ancestor_mass = zero
    d_excluded_ancestor_mass = zero
    for _ in range(0, MAX_ANCESTOR_DEPTH):
        valid = mask & (ancestor_species >= 0) & (ancestor_species < S)
        ancestor_extinction_log_probability = tl.load(E_ptr + base + ancestor_species, mask=valid, other=NEG_INF)
        d_ancestor_extinction_log_probability = tl.load(dE_ptr + base + ancestor_species, mask=valid, other=0.0)
        if USE_RECEIVER_WEIGHTS:
            ancestor_receiver_log_probability = tl.load(
                receiver_log_probs_ptr + ancestor_species, mask=valid, other=NEG_INF
            )
            d_ancestor_receiver_log_probability = tl.load(
                dreceiver_log_probs_ptr + ancestor_species, mask=valid, other=0.0
            )
            ancestor_receiver_mass = tl.where(
                valid, tl.exp2(ancestor_receiver_log_probability + ancestor_extinction_log_probability - row_max_safe), zero
            )
            d_ancestor_receiver_mass = (
                LN2 * ancestor_receiver_mass * (d_ancestor_receiver_log_probability + d_ancestor_extinction_log_probability)
            )
        else:
            ancestor_receiver_mass = tl.where(
                valid, tl.exp2(ancestor_extinction_log_probability - row_max_safe), zero
            )
            d_ancestor_receiver_mass = LN2 * ancestor_receiver_mass * d_ancestor_extinction_log_probability
        excluded_ancestor_mass += ancestor_receiver_mass
        d_excluded_ancestor_mass += d_ancestor_receiver_mass
        ancestor_species = tl.load(species_parent_ptr + ancestor_species, mask=valid, other=-1).to(tl.int32)

    valid_receiver_mass = total_receiver_mass - excluded_ancestor_mass
    d_valid_receiver_mass = (
        d_total_receiver_mass - d_excluded_ancestor_mass
    )
    has_valid_receiver_mass = mask & (valid_receiver_mass > 0.0)
    safe_valid_receiver_mass = tl.where(
        has_valid_receiver_mass,
        valid_receiver_mass,
        tl.full([BLOCK_S], 1.0, DTYPE),
    )
    transfer_complement_vjp = transfer_event_vjp + transfer_complement_output_adjoint
    donor_adjoint = tl.where(
        has_valid_receiver_mass,
        transfer_complement_vjp / safe_valid_receiver_mass,
        zero,
    )
    d_donor_adjoint = tl.where(
        has_valid_receiver_mass,
        (
            d_transfer_complement_vjp
            - donor_adjoint * d_valid_receiver_mass
        )
        / safe_valid_receiver_mass,
        zero,
    )
    tl.store(total_donor_adjoint_ptr + g, tl.sum(donor_adjoint, axis=0))
    tl.store(
        d_total_donor_adjoint_ptr + g,
        tl.sum(d_donor_adjoint, axis=0),
    )

    ancestor_species = offs
    for _ in range(0, MAX_ANCESTOR_DEPTH):
        valid = mask & (ancestor_species >= 0) & (ancestor_species < S)
        tl.atomic_add(
            excluded_donor_adjoint_ptr + base + ancestor_species,
            donor_adjoint,
            sem="relaxed",
            mask=valid,
        )
        tl.atomic_add(
            d_excluded_donor_adjoint_ptr + base + ancestor_species,
            d_donor_adjoint,
            sem="relaxed",
            mask=valid,
        )
        ancestor_species = tl.load(species_parent_ptr + ancestor_species, mask=valid, other=-1).to(tl.int32)


@triton.jit
def _finalize_extinction_vjp_directional_derivative_kernel(
    d_grad_E_ptr, d_grad_receiver_log_probs_ptr,
    receiver_mass_ptr, d_receiver_mass_ptr,
    excluded_donor_adjoint_ptr, d_excluded_donor_adjoint_ptr,
    total_donor_adjoint_ptr, d_total_donor_adjoint_ptr,
    S: tl.constexpr, BLOCK_S: tl.constexpr,
    USE_RECEIVER_WEIGHTS: tl.constexpr,
):
    g = tl.program_id(0)
    base = g * S
    offs = tl.arange(0, BLOCK_S)
    mask = offs < S
    receiver_mass = tl.load(
        receiver_mass_ptr + base + offs, mask=mask, other=0.0
    )
    d_receiver_mass = tl.load(
        d_receiver_mass_ptr + base + offs, mask=mask, other=0.0
    )
    excluded_donor_adjoint = tl.load(
        excluded_donor_adjoint_ptr + base + offs, mask=mask, other=0.0
    )
    d_excluded_donor_adjoint = tl.load(
        d_excluded_donor_adjoint_ptr + base + offs, mask=mask, other=0.0
    )
    total_donor_adjoint = tl.load(total_donor_adjoint_ptr + g)
    d_total_donor_adjoint = tl.load(d_total_donor_adjoint_ptr + g)
    current_d_grad_E = tl.load(
        d_grad_E_ptr + base + offs, mask=mask, other=0.0
    )
    d_transfer_complement_vjp = d_receiver_mass * (
        total_donor_adjoint - excluded_donor_adjoint
    ) + receiver_mass * (
        d_total_donor_adjoint - d_excluded_donor_adjoint
    )
    tl.store(
        d_grad_E_ptr + base + offs,
        current_d_grad_E + d_transfer_complement_vjp,
        mask=mask,
    )
    if USE_RECEIVER_WEIGHTS:
        tl.atomic_add(
            d_grad_receiver_log_probs_ptr + offs,
            d_transfer_complement_vjp,
            sem="relaxed",
            mask=mask,
        )


def e_step_backward_so(
    E, E_new, E_s1, E_s2, Ebar, log_pS, log_pD, log_pL, receiver_log_probs,
    species_parent, species_child1, species_child2, max_ancestor_depth,
    extinction_update_adjoint, transfer_complement_output_adjoint,
    dE, dE_new, dE_s1, dE_s2, dEbar, dlog_pS, dlog_pD, dlog_pL, dreceiver_log_probs,
    *, use_receiver_weights=False,
):
    """Return the E-step second-order contraction documented in LaTeX."""
    G, S = int(E.shape[0]), int(E.shape[1])
    block_s = int(triton.next_power_of_2(S))
    (
        d_grad_E,
        d_grad_pS,
        d_grad_pD,
        d_grad_pL,
        d_grad_max_transfer,
        receiver_mass,
        d_receiver_mass,
        excluded_donor_adjoint,
        d_excluded_donor_adjoint,
    ) = (
        torch.empty_like(E) for _ in range(9)
    )
    d_grad_receiver_log_probs = torch.zeros_like(receiver_log_probs)
    total_donor_adjoint = torch.empty((G,), dtype=E.dtype, device=E.device)
    d_total_donor_adjoint = torch.empty((G,), dtype=E.dtype, device=E.device)
    dreceiver_log_probs_arg = (
        dreceiver_log_probs
        if dreceiver_log_probs is not None
        else torch.zeros_like(receiver_log_probs)
    )
    _stage_extinction_vjp_directional_derivative_kernel[(G,)](
        E, dE, E_new, dE_new, E_s1, dE_s1, E_s2, dE_s2, Ebar, dEbar,
        log_pS, dlog_pS, log_pD, dlog_pD, log_pL, dlog_pL,
        receiver_log_probs, dreceiver_log_probs_arg,
        species_parent, species_child1, species_child2,
        extinction_update_adjoint, transfer_complement_output_adjoint,
        d_grad_E, d_grad_pS, d_grad_pD, d_grad_pL, d_grad_max_transfer,
        receiver_mass, d_receiver_mass,
        excluded_donor_adjoint, d_excluded_donor_adjoint,
        total_donor_adjoint, d_total_donor_adjoint,
        S, BLOCK_S=block_s, MAX_ANCESTOR_DEPTH=int(max_ancestor_depth),
        USE_RECEIVER_WEIGHTS=bool(use_receiver_weights), DTYPE=_tl_float_dtype(E.dtype),
        num_warps=8,
    )
    _finalize_extinction_vjp_directional_derivative_kernel[(G,)](
        d_grad_E, d_grad_receiver_log_probs,
        receiver_mass, d_receiver_mass,
        excluded_donor_adjoint, d_excluded_donor_adjoint,
        total_donor_adjoint, d_total_donor_adjoint,
        S,
        BLOCK_S=block_s,
        USE_RECEIVER_WEIGHTS=bool(use_receiver_weights),
        num_warps=8,
    )
    return (
        d_grad_E,
        d_grad_pS,
        d_grad_pD,
        d_grad_pL,
        d_grad_max_transfer,
        d_grad_receiver_log_probs,
    )
