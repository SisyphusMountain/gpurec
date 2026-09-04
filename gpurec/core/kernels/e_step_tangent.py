"""E-step tangent kernels; see ``docs/latex/kernel_mathematics.tex``."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from gpurec.api.solver_options import SolverOptions
from gpurec.core.parameters.extract_parameters import as_family_species
from gpurec.core.kernels.e_step import _tl_float_dtype
# The same additive species-tree sum the E-step forward uses, so this tangent linearizes the
# primal the forward actually computes.
from gpurec.core.kernels.species_tree_sums import (
    species_neighbourhood,
    valid_receiver_sum,
)


@triton.jit
def _update_extinction_log_probabilities_jvp_kernel(
    E_ptr,
    dE_ptr,
    dE_new_ptr,
    dE_s1_out_ptr,
    dE_s2_out_ptr,
    dEbar_out_ptr,
    max_diff_out_ptr,
    log_pS_ptr,
    log_pD_ptr,
    log_pL_ptr,
    max_transfer_ptr,
    receiver_log_probs_ptr,
    dreceiver_log_probs_ptr,
    dlog_pS_ptr,
    dlog_pD_ptr,
    dlog_pL_ptr,
    dmax_transfer_ptr,
    species_parent_ptr,
    species_child1_ptr,
    species_child2_ptr,
    species_height_ptr,
    leaf_fm_log_ptr,
    S: tl.constexpr,
    BLOCK_S: tl.constexpr,
    N_LEVELS: tl.constexpr,
    COMPUTE_DIFF: tl.constexpr,
    USE_RECEIVER_WEIGHTS: tl.constexpr,
    USE_FRACTION_MISSING: tl.constexpr,
    DTYPE: tl.constexpr,
):
    g = tl.program_id(0)
    base = g * S
    offs = tl.arange(0, BLOCK_S)
    mask = offs < S
    neg_inf = -float("inf")
    zero = tl.zeros([BLOCK_S], dtype=DTYPE)

    E = tl.load(E_ptr + base + offs, mask=mask, other=neg_inf)
    dE = tl.load(dE_ptr + base + offs, mask=mask, other=0.0)
    if USE_RECEIVER_WEIGHTS:
        receiver_log_probability = tl.load(
            receiver_log_probs_ptr + offs, mask=mask, other=neg_inf
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
    row_max_safe = tl.where(row_max != neg_inf, row_max, tl.zeros([1], dtype=DTYPE))
    receiver_mass = tl.where(
        mask, tl.exp2(receiver_weighted_extinction_log_probability - row_max_safe), zero
    )
    (
        species_height, c1_valid, c1, c2_valid, c2,
        has_parent, parent_safe, parent_height, has_sibling, sibling_safe,
    ) = species_neighbourhood(
        species_child1_ptr, species_child2_ptr, species_parent_ptr, species_height_ptr,
        offs, mask, S,
    )
    # The forward's valid receiver mass and its tangent numerator, both by ADDITION over the tree
    # and never as the row total minus the ancestor chain -- see
    # gpurec/core/kernels/species_tree_sums.py. The tangent must linearize the primal the forward
    # actually computes, so the two use the same walk.
    valid_receiver_mass = valid_receiver_sum(
        receiver_mass, mask, zero, species_height,
        c1_valid, c1, c2_valid, c2,
        has_parent, parent_safe, parent_height, has_sibling, sibling_safe, N_LEVELS,
    )
    valid_receiver_tangent_numerator = valid_receiver_sum(
        receiver_mass * d_receiver_weighted_extinction_log_probability,
        mask, zero, species_height,
        c1_valid, c1, c2_valid, c2,
        has_parent, parent_safe, parent_height, has_sibling, sibling_safe, N_LEVELS,
    )

    E_s1 = tl.load(E_ptr + base + c1, mask=c1_valid, other=neg_inf)
    E_s2 = tl.load(E_ptr + base + c2, mask=c2_valid, other=neg_inf)
    dE_s1 = tl.load(dE_ptr + base + c1, mask=c1_valid, other=0.0)
    dE_s2 = tl.load(dE_ptr + base + c2, mask=c2_valid, other=0.0)
    if USE_FRACTION_MISSING:
        # Mirror the forward E-step (e_step.py): at a leaf species the terminal
        # speciation term is the single factor p^S * fm_l, so the PRIMAL boundary is
        # E_s1=log2(fm_l), E_s2=0. fm_l is a fixed input, so its tangent is 0; dE_s1/dE_s2
        # already loaded 0.0 at the leaf child sentinels, so leave them untouched.
        fm_log = tl.load(leaf_fm_log_ptr + offs, mask=mask, other=neg_inf)
        is_missing_leaf = mask & (fm_log > neg_inf)
        E_s1 = tl.where(is_missing_leaf, fm_log, E_s1)
        E_s2 = tl.where(is_missing_leaf, tl.zeros([BLOCK_S], dtype=DTYPE), E_s2)

    max_transfer = tl.load(max_transfer_ptr + base + offs, mask=mask, other=0.0)
    d_max_transfer = tl.load(
        dmax_transfer_ptr + base + offs, mask=mask, other=0.0
    )
    has_valid_receiver_mass = valid_receiver_mass > 0.0
    safe_valid_receiver_mass = tl.where(
        has_valid_receiver_mass,
        valid_receiver_mass,
        tl.full([BLOCK_S], 1.0, DTYPE),
    )
    Ebar = tl.where(
        has_valid_receiver_mass,
        tl.log2(safe_valid_receiver_mass) + row_max + max_transfer,
        neg_inf,
    )
    dEbar = tl.where(
        has_valid_receiver_mass,
        valid_receiver_tangent_numerator / safe_valid_receiver_mass + d_max_transfer,
        zero,
    )

    log_pS = tl.load(log_pS_ptr + base + offs, mask=mask, other=neg_inf)
    log_pD = tl.load(log_pD_ptr + base + offs, mask=mask, other=neg_inf)
    log_pL = tl.load(log_pL_ptr + base + offs, mask=mask, other=neg_inf)
    dlog_pS = tl.load(dlog_pS_ptr + base + offs, mask=mask, other=0.0)
    dlog_pD = tl.load(dlog_pD_ptr + base + offs, mask=mask, other=0.0)
    dlog_pL = tl.load(dlog_pL_ptr + base + offs, mask=mask, other=0.0)

    speciation_log_term = log_pS + E_s1 + E_s2
    duplication_log_term = log_pD + 2.0 * E
    transfer_log_term = E + Ebar
    loss_log_term = log_pL
    logsumexp_max = tl.maximum(
        tl.maximum(speciation_log_term, duplication_log_term),
        tl.maximum(transfer_log_term, loss_log_term),
    )
    logsumexp_max_safe = tl.where(logsumexp_max == neg_inf, zero, logsumexp_max)
    speciation_mass = tl.exp2(speciation_log_term - logsumexp_max_safe)
    duplication_mass = tl.exp2(duplication_log_term - logsumexp_max_safe)
    transfer_mass = tl.exp2(transfer_log_term - logsumexp_max_safe)
    loss_mass = tl.exp2(loss_log_term - logsumexp_max_safe)
    extinction_event_scaled_mass = (
        speciation_mass + duplication_mass + transfer_mass + loss_mass
    )
    E_new = tl.log2(extinction_event_scaled_mass) + logsumexp_max
    inverse_extinction_event_scaled_mass = tl.where(
        extinction_event_scaled_mass > 0.0,
        1.0 / extinction_event_scaled_mass,
        zero,
    )
    speciation_probability = (
        speciation_mass * inverse_extinction_event_scaled_mass
    )
    duplication_probability = (
        duplication_mass * inverse_extinction_event_scaled_mass
    )
    transfer_probability = transfer_mass * inverse_extinction_event_scaled_mass
    loss_probability = loss_mass * inverse_extinction_event_scaled_mass

    d_speciation_log_term = dlog_pS + dE_s1 + dE_s2
    d_duplication_log_term = dlog_pD + 2.0 * dE
    d_transfer_log_term = dE + dEbar
    d_loss_log_term = dlog_pL
    dE_new = (
        speciation_probability * d_speciation_log_term
        + duplication_probability * d_duplication_log_term
        + transfer_probability * d_transfer_log_term
        + loss_probability * d_loss_log_term
    )
    dE_new = tl.where(mask & (E_new != neg_inf), dE_new, zero)

    tl.store(dE_new_ptr + base + offs, dE_new, mask=mask)
    tl.store(dE_s1_out_ptr + base + offs, dE_s1, mask=mask)
    tl.store(dE_s2_out_ptr + base + offs, dE_s2, mask=mask)
    tl.store(dEbar_out_ptr + base + offs, dEbar, mask=mask)
    if COMPUTE_DIFF:
        diff = tl.where(mask, tl.abs(dE_new - dE), zero)
        tl.store(max_diff_out_ptr + g, tl.max(diff, axis=0))


def _launch_e_step_tangent_2d(
    E,
    dE,
    log_pS_mat,
    log_pD_mat,
    log_pL_mat,
    max_transfer_mat,
    receiver_log_probs,
    dreceiver_log_probs,
    dlog_pS_mat,
    dlog_pD_mat,
    dlog_pL_mat,
    dmax_transfer_mat,
    species_parent,
    species_child1,
    species_child2,
    species_height,
    species_levels,
    *,
    out=None,
    max_diff_out=None,
    use_receiver_weights=True,
    leaf_fm_log=None,
):
    G = int(E.shape[0])
    S = int(E.shape[1])
    block_s = int(triton.next_power_of_2(S))
    dE_new, dE_s1, dE_s2, dEbar = (
        (torch.empty_like(E) for _ in range(4)) if out is None else out
    )
    use_fraction_missing = leaf_fm_log is not None
    # When there is no fraction-missing tensor the constexpr short-circuits the
    # kernel load, so a valid-but-unused 1-element placeholder is enough (mirror e_step.py).
    leaf_fm_log_arg = (
        leaf_fm_log.contiguous()
        if use_fraction_missing
        else torch.empty(1, device=E.device, dtype=E.dtype)
    )
    _update_extinction_log_probabilities_jvp_kernel[(G,)](
        E, dE, dE_new, dE_s1, dE_s2, dEbar,
        dE_new if max_diff_out is None else max_diff_out,
        log_pS_mat,
        log_pD_mat,
        log_pL_mat,
        max_transfer_mat,
        receiver_log_probs,
        dreceiver_log_probs,
        dlog_pS_mat, dlog_pD_mat, dlog_pL_mat, dmax_transfer_mat,
        species_parent, species_child1, species_child2, species_height,
        leaf_fm_log_arg,
        S,
        BLOCK_S=block_s,
        N_LEVELS=int(species_levels),
        COMPUTE_DIFF=max_diff_out is not None,
        USE_RECEIVER_WEIGHTS=bool(use_receiver_weights),
        USE_FRACTION_MISSING=use_fraction_missing,
        DTYPE=_tl_float_dtype(E.dtype),
        num_warps=8,
    )
    return dE_new, dE_s1, dE_s2, dEbar


def e_tangent_fixed_point(
    E_star,
    dlog_pS, dlog_pD, dlog_pL, dmax_transfer,
    log_pS, log_pD, log_pL, max_transfer, receiver_log_probs,
    species_parent, species_child1, species_child2,
    *,
    species_height,
    species_levels,
    max_iter=None,
    tol=None,
    use_receiver_weights=True,
    dE0=None,
    dreceiver_log_probs=None,
    leaf_fm_log=None,
):
    """Solve the tangent fixed point documented in the LaTeX reference.

    ``species_height`` (0 at a leaf, 1 + the taller child above) and ``species_levels`` (the tree's
    height) drive the additive valid-receiver-mass walk this tangent shares with the forward."""
    if max_iter is None:
        max_iter = SolverOptions().e_max_iter
    if tol is None:
        tol = SolverOptions().e_tangent_tol
    E_a = E_star.contiguous()
    S = int(E_a.shape[1])
    family_rows = int(E_a.shape[0])
    mats = tuple(
        as_family_species(parameter, S, family_rows)
        for parameter in (log_pS, log_pD, log_pL, max_transfer)
    )
    dmats = tuple(
        as_family_species(parameter, S, family_rows)
        for parameter in (dlog_pS, dlog_pD, dlog_pL, dmax_transfer)
    )
    receiver_log_probs = receiver_log_probs.contiguous()
    dreceiver_log_probs = (
        torch.zeros_like(receiver_log_probs)
        if dreceiver_log_probs is None
        else dreceiver_log_probs.to(
            device=receiver_log_probs.device, dtype=receiver_log_probs.dtype
        )
        .reshape(S)
        .contiguous()
    )
    args = (
        *mats,
        receiver_log_probs,
        dreceiver_log_probs,
        *dmats,
        species_parent,
        species_child1,
        species_child2,
        species_height,
        int(species_levels),
    )

    dE_a = torch.zeros_like(E_a) if dE0 is None else dE0.contiguous().clone()
    dE_b, dE_s1, dE_s2, dEbar = (torch.empty_like(E_a) for _ in range(4))
    max_diff_out = torch.empty((family_rows,), dtype=E_a.dtype, device=E_a.device)

    for _ in range(int(max_iter)):
        _launch_e_step_tangent_2d(
            E_a, dE_a, *args, out=(dE_b, dE_s1, dE_s2, dEbar),
            max_diff_out=max_diff_out, use_receiver_weights=bool(use_receiver_weights),
            leaf_fm_log=leaf_fm_log,
        )
        dE_a, dE_b = dE_b, dE_a
        max_diff = float(max_diff_out.max().item())
        scale = float(dE_a.abs().max().item())
        if max_diff <= tol * max(1.0, scale):
            break

    _, dE_s1, dE_s2, dEbar = _launch_e_step_tangent_2d(
        E_a, dE_a, *args, use_receiver_weights=bool(use_receiver_weights),
        leaf_fm_log=leaf_fm_log,
    )
    return dE_a, dE_s1, dE_s2, dEbar
