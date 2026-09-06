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
# The two additive species-tree sums these kernels linearize; shared verbatim with the
# wave second-order and E-step kernels so every path builds the same primal.
from gpurec.core.kernels.species_tree_sums import (
    species_neighbourhood as _species_neighbourhood,
    valid_receiver_sum as _valid_receiver_sum,
)

# Triton warps per program for the fused self-loop JVP iteration kernel
# used only for the exact tangent's automatic numerical fallback. Override per run via env.
_NUM_WARPS_SELF_LOOP_JVP_ITERS = int(os.environ.get("NEWTON_WST_NUM_WARPS", "4"))

# Triton warps per program for the EXACT tangent solve
# (``_solve_reconciliation_self_loop_jvp_exact_kernel``). It read the iteration kernel's constant
# until it was measured on its own, and it is a different kernel: Nsight Compute on the 200-family
# Coleman batch (S=2013, BLOCK_S=2048, largest wave 4694 rows) shows it using the full 255
# registers per thread, spilling on 100 % of its warps (8.1e7 local-memory LOAD requests in one
# launch) and running at 16.4 % occupancy. Splitting the row over more warps gives each thread
# fewer species lanes to hold, which is what removes the spills, and the kernel gets monotonically
# faster all the way to the widest launch Triton allows: on the RTX 4090 at S=2013, one probe's
# 190 launches cost 71.3 ms at 4 warps, 60.7 at 8, 56.3 at 16 and 48.0 at 32 (median of 5
# round-robin rounds in one process). The answer moves by 5.7e-7 relative -- a different warp
# reduction order, far inside float32 noise. Measured on the 4090 only; the H100 was not available
# for this sweep, and the transposed solve next door also wants 32 warps there.
_NUM_WARPS_SELF_LOOP_JVP_EXACT = 32

# Debug instrumentation for the exact tangent solve, off in production. When switched on with
# ``set_exact_tangent_guard_trip_collection(True)`` every wave appends a [W, 2] int32 tensor to
# ``_EXACT_TANGENT_GUARD_TRIPS``: column 0 counts a row's non-positive elimination pivots,
# column 1 flags a non-positive ``1 - donor tangent loop gain``. A module-level switch (not an
# argument) so the production call path keeps its exact signature and costs nothing.
_COLLECT_EXACT_TANGENT_GUARD_TRIPS = False
_EXACT_TANGENT_GUARD_TRIPS = []


def set_exact_tangent_guard_trip_collection(enabled):
    """Turn the exact tangent solve's per-row pivot-guard counters on or off."""
    global _COLLECT_EXACT_TANGENT_GUARD_TRIPS
    _COLLECT_EXACT_TANGENT_GUARD_TRIPS = bool(enabled)
    if not _COLLECT_EXACT_TANGENT_GUARD_TRIPS:
        _EXACT_TANGENT_GUARD_TRIPS.clear()


@triton.jit(do_not_specialize=["ws", "pi_ws"], do_not_specialize_on_alignment=["dPi_new_ptr"])
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
    species_child1_ptr, species_child2_ptr, species_parent_ptr, species_height_ptr,
    leaf_species_ptr, leaf_logp_ptr, d_leaf_logp_ptr,
    family_idx_ptr,
    gene_split_log_likelihood_ptr,
    d_gene_split_log_likelihood_ptr,
    gene_split_offset_ptr,
    has_splits: tl.constexpr,
    dPi_new_ptr,
    dPibar_out_ptr,
    row_mask_ptr,
    n_iters,
    S: tl.constexpr,
    stride: tl.constexpr,
    CONST_ROW_STRIDE: tl.constexpr,
    BLOCK_S: tl.constexpr,
    N_LEVELS: tl.constexpr,
    USE_LEAF_INDEX: tl.constexpr,
    STORE_PIBAR: tl.constexpr,
    USE_RECEIVER_WEIGHTS: tl.constexpr,
    DTYPE: tl.constexpr,
):
    """Fuse fixed-count wave-tangent self-loop iterations.

    ``row_mask_ptr`` restricts the sweeps to the clade rows the exact tangent handed back -- the
    ones the forward could not hold under one row scale. Every other row returns immediately,
    leaving whatever the exact elimination published for it.
    """
    NEG = -float("inf")
    # int64: w ranges over the whole batch's clade rows, so the *stride
    # multiplies below can overflow int32 once total_clades * S exceeds 2^31.
    w = tl.program_id(0).to(tl.int64)
    # Selective sweep: every row the exact tangent could solve already holds its answer, so
    # returning here makes this launch a no-op on it.
    if tl.load(row_mask_ptr + ws + w) == 0:
        return
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
    (
        species_height, c1_valid, c1_safe, c2_valid, c2_safe,
        has_parent, parent_safe, parent_height, has_sibling, sibling_safe,
    ) = _species_neighbourhood(
        species_child1_ptr, species_child2_ptr, species_parent_ptr, species_height_ptr,
        s_offs, mask, S,
    )
    c1 = c1_safe
    c2 = c2_safe
    # primal valid receiver mass (invariant), by addition only
    valid_receiver_mass = _valid_receiver_sum(
        receiver_mass, mask, zero, species_height,
        c1_valid, c1_safe, c2_valid, c2_safe,
        has_parent, parent_safe, parent_height, has_sibling, sibling_safe, N_LEVELS,
    )

    const_offsets = const_base + s_offs
    max_transfer = tl.load(max_transfer_ptr + const_offsets, mask=mask, other=0.0)
    d_max_transfer = tl.load(dmax_transfer_ptr + const_offsets, mask=mask, other=0.0)
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
        valid_receiver_tangent_numerator = _valid_receiver_sum(
            receiver_mass * d_receiver_weighted_reconciliation_log_likelihood, mask, zero, species_height,
            c1_valid, c1_safe, c2_valid, c2_safe,
            has_parent, parent_safe, parent_height, has_sibling, sibling_safe, N_LEVELS,
        )
        d_transfer_complement_log_likelihood = tl.where(
            has_valid_receiver_mass,
            valid_receiver_tangent_numerator * inverse_valid_receiver_mass + d_max_transfer,
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


@triton.jit(do_not_specialize=["ws", "pi_ws"], do_not_specialize_on_alignment=["dPi_new_ptr"])
def _solve_reconciliation_self_loop_jvp_exact_kernel(
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
    species_child1_ptr, species_child2_ptr, species_parent_ptr, species_height_ptr,
    leaf_species_ptr, leaf_logp_ptr, d_leaf_logp_ptr,
    family_idx_ptr,
    gene_split_log_likelihood_ptr,
    d_gene_split_log_likelihood_ptr,
    gene_split_offset_ptr,
    has_splits: tl.constexpr,
    dPi_new_ptr,
    dPibar_out_ptr,
    guard_trips_ptr,
    row_mask_ptr,
    S: tl.constexpr,
    stride: tl.constexpr,
    CONST_ROW_STRIDE: tl.constexpr,
    BLOCK_S: tl.constexpr,
    N_LEVELS: tl.constexpr,
    USE_LEAF_INDEX: tl.constexpr,
    STORE_PIBAR: tl.constexpr,
    WRITE_GUARD_TRIPS: tl.constexpr,
    USE_RECEIVER_WEIGHTS: tl.constexpr,
    USE_ROW_MASK: tl.constexpr,
    DTYPE: tl.constexpr,
):
    """Solve the wave-tangent self-loop EXACTLY, by elimination on the species tree.

    Replaces the fixed ``n_iters`` Jacobi sweeps of
    :func:`_apply_reconciliation_self_loop_jvp_iterations_kernel` with a direct solve of the same
    system, and publishes the same two outputs (``dPi_new``, and ``dPibar`` when asked).

    THE SYSTEM. The tangent iterates ``dv <- b + J dv`` where ``J`` is the SAME wave-update
    Jacobian the backward transposes -- the tangent and the adjoint are the two sides of one
    operator. Per species, in the notation of that kernel,

        d[s]  = (duplication_loss_mass + transfer_loss_mass) / event_mass    stay in s
        g[s]  = transfer_mass / event_mass / valid_receiver_mass             transfer into s
        p1/p2 = speciation_child_mass / event_mass                           speciate into s
        m[s]  = receiver_mass[s]                                             s's weight as a donor
        b[s]  = every term that does NOT depend on dv (the tangents of the event constants,
                the leaf observation and the gene-split input), over event_mass

    and, with ``M`` the row's total donor tangent and ``X[s]`` the part of it held by ``s`` and its
    ancestors,

        dv[s] = b[s] + d[s] dv[s] + g[s] (M - X[s]) + p1[s] dv[c1] + p2[s] dv[c2]
        M     = sum_t m[t] (dv[t] + drecv[t])       X[s] = the same sum over s and its ancestors.

    That is the FORWARD tree system of the primal solve with a different right-hand side, so the
    forward elimination applies unchanged. Writing ``u[s] = M - X[parent(s)]`` for the donor
    tangent still available to ``s`` (``u[root] = M``, ``u[c] = u[s] - m[s](dv[s] + drecv[s])``),

        diag[s] dv[s] - g[s] u[s] - p1[s] dv[c1] - p2[s] dv[c2] = b[s]
        diag[s] = 1 - d[s] + g[s] m[s]

    1. Bottom-up: ``dv[s] = alpha[s] + gamma[s] u[s]``. A leaf divides by ``diag``; an internal
       node folds in its children, with ``G = p1 gamma[c1] + p2 gamma[c2]`` and
       ``pivot = diag[s] + G m[s]`` -- again the diagonal PLUS the children's feedback.
    2. In the same bottom-up walk, each subtree's donor tangent becomes an affine function of the
       donor tangent entering it, ``sum over r in subtree(s) of m[r](dv[r] + drecv[r]) = A[s] +
       G[s] u[s]``: a leaf has ``A = m (alpha + drecv)``, ``G = m gamma``; an internal node, whose
       children both see ``u[c] = u[s] - m[s](dv[s] + drecv[s])``, has

           G[s] = m gamma + (G[c1] + G[c2]) (1 - m gamma)
           A[s] = m (alpha + drecv) (1 - G[c1] - G[c2]) + A[c1] + A[c2].

       ``G`` is built from primal masses only, so it is a gain in ``[0, 1)`` like the forward's;
       ``A`` carries the signs of the right-hand side.
    3. The root closes the loop: ``M = A[root] / (1 - G[root])``. Then top-down, by ADDITION
       only, with ``R[s]`` the donor tangent hanging off ``s``'s ancestor chain (``R[root] = 0``):

           v     = (R[s] + A[c1] + A[c2]) / (1 - G[c1] - G[c2])      what both children see
           u[c1] = u[c2] = v
           R[c1] = R[s] + A[c2] + G[c2] v        R[c2] = R[s] + A[c1] + G[c1] v

       ``v`` is also the tangent of ``s``'s own valid receiver mass, so ``dPibar`` comes out of
       it directly. This is the forward kernel's scheme (see its docstring); the earlier form
       ``u[c] = u[s] - m[s](dv[s] + drecv[s])`` is a difference of nearly equal numbers for every
       species under the lane holding the row's donor mass and is rounding noise there.

    THE PRIMAL MASSES. Each lane's primal valid receiver mass, ``valid[s] = sum of m over every
    species that is neither s nor an ancestor of s``, is built the same additive way -- subtree
    masses bottom-up, off-chain masses top-down, ``valid[s] = R[s] + M[c1] + M[c2]`` -- and never
    as ``total - ancestor chain``: that subtraction is exactly the rounding floor the forward
    kernel removed, and the tangent must linearize the primal the forward actually computed.

    HOW THE WALKS RUN HERE. This kernel is register-resident -- one program per clade row, the
    whole species row in one tile, children and parents reached with ``tl.gather`` -- and there is
    no scatter to a register tile. So each level is a WHOLE-ROW update masked to the lanes at that
    height, using ``species_height`` (0 at a leaf), rather than a walk over one level's node list:

        alpha = where(height == level, (b + p1 alpha[c1] + p2 alpha[c2]) / pivot, alpha)

    Every lane computes the update; only the lanes at this level keep it, and their children sit at
    strictly lower heights and are therefore already final. ``2 * N_LEVELS`` such passes, each two
    gathers wide, replace ``n_iters`` sweeps each carrying an ancestor-depth-deep ancestor
    gather -- and unlike them, the answer does not depend on how many were run.

    Lanes whose primal reconciliation is impossible have zero event mass, hence zero ``b``, ``d``,
    ``g``, ``p1`` and ``p2``: their pivot is exactly 1 and their tangent exactly 0, which is what
    the iteration's ``valid`` mask produces too.

    ``guard_trips_ptr`` is an optional ``[rows, 2]`` int32 diagnostic: column 0 counts a row's
    non-positive pivots, column 1 flags a non-positive ``1 - M1``. Neither is substituted.
    """
    NEG = -float("inf")
    # int64: w ranges over the whole batch's clade rows, so the *stride
    # multiplies below can overflow int32 once total_clades * S exceeds 2^31.
    w = tl.program_id(0).to(tl.int64)
    if USE_ROW_MASK:
        # The forward could not hold these rows under one row scale and solved them in log space;
        # the tangent's own elimination underflows on exactly the same lanes, so it leaves them to
        # the sweeps. The two launches cover every row of the wave between them.
        if tl.load(row_mask_ptr + ws + w) != 0:
            return
    pi_base = (pi_ws + w) * stride
    out_base = w * stride
    global_base = (ws + w) * stride
    family = tl.load(family_idx_ptr + ws + w)
    const_base = family * CONST_ROW_STRIDE

    s_offs = tl.arange(0, BLOCK_S)
    mask = s_offs < S
    zero = tl.zeros([BLOCK_S], dtype=DTYPE)

    pi_offset = tl.load(Pi_offset_ptr + pi_ws + w)

    # ---- invariant setup: the primal masses the tangent multiplies. Identical to
    # ``_apply_reconciliation_self_loop_jvp_iterations_kernel``'s, term for term.
    reconciliation_log_likelihood = tl.load(Pi_ptr + pi_base + s_offs, mask=mask, other=NEG)
    if USE_RECEIVER_WEIGHTS:
        receiver_log_probability = tl.load(receiver_log_probs_ptr + s_offs, mask=mask, other=NEG)
        d_receiver_log_probability = tl.load(
            dreceiver_log_probs_ptr + s_offs, mask=mask, other=0.0
        )
        receiver_weighted_reconciliation_log_likelihood = (
            receiver_log_probability + reconciliation_log_likelihood
        )
    else:
        d_receiver_log_probability = zero
        receiver_weighted_reconciliation_log_likelihood = reconciliation_log_likelihood
    row_max = tl.max(receiver_weighted_reconciliation_log_likelihood, axis=0)
    row_max_safe = tl.where(row_max != NEG, row_max, tl.zeros([1], dtype=DTYPE))
    receiver_mass = tl.where(
        mask, tl.exp2(receiver_weighted_reconciliation_log_likelihood - row_max_safe), zero
    )
    total_receiver_mass = tl.sum(receiver_mass, axis=0)

    # Species-tree neighbourhood of every lane, and the primal valid receiver mass by addition
    # only (docstring, THE PRIMAL MASSES); both shared with the sweep kernels above.
    (
        species_height, c1_valid, c1_safe, c2_valid, c2_safe,
        has_parent, parent_safe, parent_height, has_sibling, sibling_safe,
    ) = _species_neighbourhood(
        species_child1_ptr, species_child2_ptr, species_parent_ptr, species_height_ptr,
        s_offs, mask, S,
    )
    is_root = mask & ~has_parent
    valid_receiver_mass = _valid_receiver_sum(
        receiver_mass, mask, zero, species_height,
        c1_valid, c1_safe, c2_valid, c2_safe,
        has_parent, parent_safe, parent_height, has_sibling, sibling_safe, N_LEVELS,
    )

    const_offsets = const_base + s_offs
    max_transfer = tl.load(max_transfer_ptr + const_offsets, mask=mask, other=0.0)
    d_max_transfer = tl.load(dmax_transfer_ptr + const_offsets, mask=mask, other=0.0)
    has_valid_receiver_mass = valid_receiver_mass > 0.0
    safe_valid_receiver_mass = tl.where(
        has_valid_receiver_mass, valid_receiver_mass, tl.full([BLOCK_S], 1.0, DTYPE)
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

    reconciliation_child1_log_likelihood = tl.where(
        c1_valid, tl.gather(reconciliation_log_likelihood, c1_safe, axis=0), NEG
    )
    reconciliation_child2_log_likelihood = tl.where(
        c2_valid, tl.gather(reconciliation_log_likelihood, c2_safe, axis=0), NEG
    )

    duplication_loss_log_term = duplication_loss_const + reconciliation_log_likelihood
    transfer_loss_log_term = reconciliation_log_likelihood + extinction_complement_log_probability
    transfer_log_term = transfer_complement_log_likelihood + extinction_log_probability
    speciation_child1_log_term = speciation_child1_const + reconciliation_child1_log_likelihood
    speciation_child2_log_term = speciation_child2_const + reconciliation_child2_log_likelihood
    if USE_LEAF_INDEX:
        leaf_species = tl.load(leaf_species_ptr + ws + w)
        leaf_hit = mask & (leaf_species == s_offs)
        leaf_logp = tl.load(leaf_logp_ptr + family * S + s_offs, mask=mask, other=NEG)
        d_leaf_logp = tl.load(d_leaf_logp_ptr + family * S + s_offs, mask=mask, other=0.0)
        leaf_logp = leaf_logp + (0.0 - pi_offset).to(DTYPE)
        leaf_observation_log_term = tl.where(leaf_hit, leaf_logp, NEG)
        d_leaf_observation_log_term = tl.where(leaf_hit, d_leaf_logp, zero)
    else:
        leaf_observation_log_term = tl.full([BLOCK_S], NEG, dtype=DTYPE)
        d_leaf_observation_log_term = zero

    logsumexp_max = tl.maximum(
        tl.maximum(
            tl.maximum(duplication_loss_log_term, transfer_loss_log_term),
            tl.maximum(transfer_log_term, speciation_child1_log_term),
        ),
        tl.maximum(speciation_child2_log_term, leaf_observation_log_term),
    )
    if has_splits:
        gene_split_log_likelihood = tl.load(
            gene_split_log_likelihood_ptr + out_base + s_offs, mask=mask, other=NEG
        )
        gene_split_offset = tl.load(gene_split_offset_ptr + w)
        gene_split_log_likelihood = gene_split_log_likelihood + (gene_split_offset - pi_offset).to(DTYPE)
        d_gene_split_log_likelihood = tl.load(
            d_gene_split_log_likelihood_ptr + out_base + s_offs, mask=mask, other=0.0
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
    gene_split_tangent_numerator = zero
    if has_splits:
        gene_split_mass = tl.exp2(gene_split_log_likelihood - logsumexp_max_safe)
        reconciliation_event_scaled_mass += gene_split_mass
        gene_split_tangent_numerator = gene_split_mass * d_gene_split_log_likelihood
    updated_reconciliation_log_likelihood = tl.log2(reconciliation_event_scaled_mass) + logsumexp_max
    inverse_reconciliation_event_scaled_mass = tl.where(
        reconciliation_event_scaled_mass > 0.0, 1.0 / reconciliation_event_scaled_mass, zero
    )
    valid = mask & (updated_reconciliation_log_likelihood != NEG)

    # ---- the tangent system's five coefficients, per species.
    self_coefficient = (
        (duplication_loss_mass + transfer_loss_mass) * inverse_reconciliation_event_scaled_mass
    )
    donor_coefficient = (
        transfer_mass * inverse_reconciliation_event_scaled_mass * inverse_valid_receiver_mass
    )
    speciation_child1_coefficient = (
        speciation_child1_mass * inverse_reconciliation_event_scaled_mass
    )
    speciation_child2_coefficient = (
        speciation_child2_mass * inverse_reconciliation_event_scaled_mass
    )
    # Everything the tangent picks up that does NOT depend on the row's own tangent. The transfer
    # term's own d_max_transfer rides here; the iteration drops it wherever there is no valid
    # receiver mass, and so does this, because transfer_mass is zero exactly there.
    source = (
        duplication_loss_mass * d_duplication_loss_const
        + transfer_loss_mass * d_extinction_complement_log_probability
        + transfer_mass
        * (
            d_extinction_log_probability
            + tl.where(has_valid_receiver_mass, d_max_transfer, zero)
        )
        + speciation_child1_mass * d_speciation_child1_const
        + speciation_child2_mass * d_speciation_child2_const
        + leaf_observation_mass * d_leaf_observation_log_term
        + gene_split_tangent_numerator
    ) * inverse_reconciliation_event_scaled_mass
    diagonal = 1.0 - self_coefficient + donor_coefficient * receiver_mass
    # What s contributes to the row's donor tangent is m[s] (dv[s] + drecv[s]). Only the dv[s]
    # half is an unknown; the drecv[s] half is a known number that s hands DOWN, because every
    # node below s sees the donor tangent already reduced by it:
    #
    #     u[c] = u[s] - m[s] (dv[s] + drecv[s]).
    #
    # So it leaves the diagonal twice over -- once out of s's own donor term, and once out of
    # every child term, whose u[c] carries it. Collecting both against the coefficient that
    # multiplies u, which is exactly gamma[s], the whole correction is one product:
    #
    #     alpha[s] = (source[s] + children) / pivot[s]  -  gamma[s] m[s] drecv[s].
    #
    # Dropping the children's half is what broke the joint (theta, alpha, omega) Hessian's
    # symmetry: it is zero at a leaf, where there are no children, so the error only appears at
    # internal nodes, and it is zero for every node when the receiver weights are uniform, which
    # is why no genewise fit could see it.
    receiver_tangent_offset = receiver_mass * d_receiver_log_probability

    # ---- walk 1, bottom-up. Every lane starts at the leaf case; level ``level`` then rewrites the
    # lanes of that height, whose children are already final. Alongside alpha and gamma, each
    # lane's subtree donor tangent as an affine function of what enters it (docstring, step 2).
    gamma = donor_coefficient / diagonal
    alpha = source / diagonal - gamma * receiver_tangent_offset
    subtree_tangent_gain = receiver_mass * gamma
    subtree_tangent_constant = receiver_mass * (alpha + d_receiver_log_probability)
    guard_trips = tl.sum(tl.where(mask & (diagonal <= 0.0), 1, 0).to(tl.int32), axis=0)
    smallest_gain_margin = tl.full([1], 1.0, DTYPE)
    for level in range(1, N_LEVELS + 1):
        child_source = (
            speciation_child1_coefficient
            * tl.where(c1_valid, tl.gather(alpha, c1_safe, axis=0), zero)
            + speciation_child2_coefficient
            * tl.where(c2_valid, tl.gather(alpha, c2_safe, axis=0), zero)
        )
        child_donor_gain = (
            speciation_child1_coefficient
            * tl.where(c1_valid, tl.gather(gamma, c1_safe, axis=0), zero)
            + speciation_child2_coefficient
            * tl.where(c2_valid, tl.gather(gamma, c2_safe, axis=0), zero)
        )
        pivot = diagonal + child_donor_gain * receiver_mass
        level_gamma = (donor_coefficient + child_donor_gain) / pivot
        level_alpha = (source + child_source) / pivot - level_gamma * receiver_tangent_offset
        at_level = mask & (species_height == level)
        alpha = tl.where(at_level, level_alpha, alpha)
        gamma = tl.where(at_level, level_gamma, gamma)
        guard_trips += tl.sum(tl.where(at_level & (pivot <= 0.0), 1, 0).to(tl.int32), axis=0)
        children_gain = tl.where(
            c1_valid, tl.gather(subtree_tangent_gain, c1_safe, axis=0), zero
        ) + tl.where(c2_valid, tl.gather(subtree_tangent_gain, c2_safe, axis=0), zero)
        children_constant = tl.where(
            c1_valid, tl.gather(subtree_tangent_constant, c1_safe, axis=0), zero
        ) + tl.where(c2_valid, tl.gather(subtree_tangent_constant, c2_safe, axis=0), zero)
        children_gain_margin = 1.0 - children_gain
        own_gain = receiver_mass * level_gamma
        subtree_tangent_gain = tl.where(
            at_level, own_gain + children_gain * (1.0 - own_gain), subtree_tangent_gain
        )
        subtree_tangent_constant = tl.where(
            at_level,
            receiver_mass * (level_alpha + d_receiver_log_probability) * children_gain_margin
            + children_constant,
            subtree_tangent_constant,
        )
        guard_trips += tl.sum(
            tl.where(at_level & (children_gain_margin <= 0.0), 1, 0).to(tl.int32), axis=0
        )
        smallest_gain_margin = tl.minimum(
            smallest_gain_margin,
            tl.min(tl.where(at_level, children_gain_margin, tl.full([BLOCK_S], 1.0, DTYPE)), axis=0),
        )

    # ---- close the loop at the root: M = A[root] + G[root] M (docstring, step 3).
    root_tangent_constant = tl.sum(tl.where(is_root, subtree_tangent_constant, zero), axis=0)
    root_tangent_gain = tl.sum(tl.where(is_root, subtree_tangent_gain, zero), axis=0)
    loop_denominator = 1.0 - root_tangent_gain
    total_donor_tangent = root_tangent_constant / loop_denominator
    smallest_gain_margin = tl.minimum(smallest_gain_margin, loop_denominator)

    # ---- walk 2, top-down, additions only. The root holds u = M and nothing hangs off its
    # chain; every other lane is settled in the pass whose level is its PARENT's height, from
    # the parent's settled R and the sibling's final subtree coefficients.
    available_donor_tangent = tl.where(is_root, total_donor_tangent, zero)
    off_chain_donor_tangent = zero
    for level_index in range(0, N_LEVELS):
        level = N_LEVELS - level_index
        at_level = has_parent & (parent_height == level)
        parent_off_chain = tl.gather(off_chain_donor_tangent, parent_safe, axis=0)
        sibling_constant = tl.where(
            has_sibling, tl.gather(subtree_tangent_constant, sibling_safe, axis=0), zero
        )
        sibling_gain = tl.where(
            has_sibling, tl.gather(subtree_tangent_gain, sibling_safe, axis=0), zero
        )
        level_available = (parent_off_chain + subtree_tangent_constant + sibling_constant) / (
            1.0 - subtree_tangent_gain - sibling_gain
        )
        available_donor_tangent = tl.where(at_level, level_available, available_donor_tangent)
        off_chain_donor_tangent = tl.where(
            at_level,
            parent_off_chain + sibling_constant + sibling_gain * level_available,
            off_chain_donor_tangent,
        )

    d_reconciliation_log_likelihood = tl.where(
        valid, alpha + gamma * available_donor_tangent, zero
    )
    tl.store(dPi_new_ptr + out_base + s_offs, d_reconciliation_log_likelihood, mask=mask)
    if STORE_PIBAR:
        # The tangent of s's valid receiver mass is what both of its children see: the mass off
        # s's chain plus its children's subtree tangents, again a sum.
        children_gain = tl.where(
            c1_valid, tl.gather(subtree_tangent_gain, c1_safe, axis=0), zero
        ) + tl.where(c2_valid, tl.gather(subtree_tangent_gain, c2_safe, axis=0), zero)
        children_constant = tl.where(
            c1_valid, tl.gather(subtree_tangent_constant, c1_safe, axis=0), zero
        ) + tl.where(c2_valid, tl.gather(subtree_tangent_constant, c2_safe, axis=0), zero)
        d_valid_receiver_mass = (off_chain_donor_tangent + children_constant) / (
            1.0 - children_gain
        )
        d_transfer_complement_log_likelihood = tl.where(
            has_valid_receiver_mass,
            d_valid_receiver_mass * inverse_valid_receiver_mass + d_max_transfer,
            zero,
        )
        tl.store(
            dPibar_out_ptr + global_base + s_offs,
            d_transfer_complement_log_likelihood,
            mask=mask,
        )
    if WRITE_GUARD_TRIPS:
        tl.store(guard_trips_ptr + w * 3, guard_trips.to(DTYPE))
        tl.store(
            guard_trips_ptr + w * 3 + 1,
            tl.where(smallest_gain_margin <= 0.0, 1.0, 0.0).to(DTYPE),
        )
        # How far the closure and the per-node ``1 - gain`` divisors stayed from singular. The
        # solve divides by each of them, so a row whose smallest margin is near zero is a row
        # whose tangent carries the reciprocal of that margin in relative error -- the number to
        # look at when the answer disagrees with the sweeps but no guard actually tripped.
        tl.store(guard_trips_ptr + w * 3 + 2, smallest_gain_margin.to(DTYPE))


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


def compute_wave_step_tangent(
    Pi_in, dPi_io, ws, W, S, n_iters,
    max_transfer_mat, dmax_transfer,
    duplication_loss_const, d_duplication_loss_const,
    Ebar, dEbar, E, dE,
    speciation_child1_const, d_speciation_child1_const,
    speciation_child2_const, d_speciation_child2_const,
    receiver_log_probs, species_child1, species_child2, species_parent,
    gene_split_log_likelihood=None, d_gene_split_log_likelihood=None,
    *, leaf_species_idx, leaf_logp, d_leaf_logp, family_idx,
    dPibar_out=None, has_leaf_term=True, use_receiver_weights=True,
    dreceiver_log_probs=None,
    pi_offset, gene_split_offset=None,
    species_height, species_levels, wide_row=None,
):
    """Solve the wave-tangent self-loop exactly; see the LaTeX reference.

    The tree elimination handles normal rows. ``n_iters`` bounds the iterative numerical fallback
    for rows that the primal marked as too wide for one scale. The solve needs
    ``species_height`` (``species_helpers["sp_height"]``) and
    ``species_levels``, the species tree's height, which is ``compact_level_ptr.numel() - 1``;
    that is a shape, so the caller reads it without a device-to-host reduction.

    ``wide_row`` (``int8``, one entry per clade row) is the forward's own record of which rows it
    could not hold under a single row scale. Those rows go to the fallback sweeps; ``None`` means
    the forward flagged nothing and only the elimination runs.
    """
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
    species_height = species_height.to(device=Pi_in.device, dtype=torch.int32).contiguous()
    n_levels = int(species_levels)
    collect_guard_trips = _COLLECT_EXACT_TANGENT_GUARD_TRIPS
    # [rows, 3]: non-positive pivot count, non-positive ``1 - gain`` flag, and the smallest
    # ``1 - gain`` margin (per node or at the closure). Floats throughout keep stores uniform.
    guard_trips = (
        torch.zeros((int(W), 3), device=Pi_in.device, dtype=Pi_in.dtype)
        if collect_guard_trips
        else Pi_in
    )
    _solve_reconciliation_self_loop_jvp_exact_kernel[(int(W),)](
        Pi_in, dPi_io,
        pi_offset_arg,
        ws, ws,
        max_transfer_mat, dmax_transfer,
        duplication_loss_const, d_duplication_loss_const,
        Ebar, dEbar, E, dE,
        speciation_child1_const, d_speciation_child1_const,
        speciation_child2_const, d_speciation_child2_const,
        receiver_log_probs, dreceiver_log_probs,
        species_child1, species_child2, species_parent, species_height,
        leaf_species_idx, leaf_logp, d_leaf_logp,
        family_idx,
        gene_split_log_likelihood if has_splits else dummy,
        d_gene_split_log_likelihood if has_splits else dummy,
        gene_split_offset_arg,
        has_splits,
        dPi_out_rows,
        dPibar_out if store_pibar else dummy,
        guard_trips,
        wide_row if wide_row is not None else species_child1,
        S,
        stride=S,
        CONST_ROW_STRIDE=const_row_stride,
        BLOCK_S=block_s,
        N_LEVELS=n_levels,
        USE_LEAF_INDEX=bool(has_leaf_term),
        STORE_PIBAR=bool(store_pibar),
        WRITE_GUARD_TRIPS=bool(collect_guard_trips),
        USE_RECEIVER_WEIGHTS=bool(use_receiver_weights),
        USE_ROW_MASK=bool(wide_row is not None),
        DTYPE=_tl_float_dtype(Pi_in.dtype),
        num_warps=_NUM_WARPS_SELF_LOOP_JVP_EXACT,
    )
    if collect_guard_trips:
        _EXACT_TANGENT_GUARD_TRIPS.append(guard_trips)
    if wide_row is None:
        return
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
        species_child1, species_child2, species_parent, species_height,
        leaf_species_idx, leaf_logp, d_leaf_logp,
        family_idx,
        gene_split_log_likelihood if has_splits else dummy,
        d_gene_split_log_likelihood if has_splits else dummy,
        gene_split_offset_arg,
        has_splits,
        dPi_out_rows,
        dPibar_out if store_pibar else dummy,
        wide_row,
        int(max(int(n_iters), 1)),
        S,
        stride=S,
        CONST_ROW_STRIDE=const_row_stride,
        BLOCK_S=block_s,
        N_LEVELS=n_levels,
        USE_LEAF_INDEX=bool(has_leaf_term),
        STORE_PIBAR=bool(store_pibar),
        USE_RECEIVER_WEIGHTS=bool(use_receiver_weights),
        DTYPE=_tl_float_dtype(Pi_in.dtype),
        num_warps=_NUM_WARPS_SELF_LOOP_JVP_ITERS,
    )
