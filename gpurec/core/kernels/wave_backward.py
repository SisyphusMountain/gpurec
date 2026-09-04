"""Reconciliation-wave adjoint kernels and launch wrappers.

See ``docs/latex/kernel_mathematics.tex`` for the self-loop transpose, event
VJPs, and transfer-subtree equations implemented here.
"""

import torch
import triton
import triton.language as tl

from gpurec.core.kernels._dts_layout_contract import dts_backward_param_layout
from gpurec.core.kernels.pi_forward import (
    _validate_offset_tensor,
    _validate_residual_tensors,
    exact_conditioning_floor,
)
from gpurec.core.memory_policy import proposal0_memory_gate
from gpurec.core.valid_receivers import valid_receiver_index_tables
from gpurec.api.solver_options import dtype_scaled_self_loop_tol

from gpurec.core.kernels.wave_backward_kernels import (
    _select_active_adjoint_rows_kernel,
    _prepare_reconciliation_self_loop_vjp_kernel,
    _apply_reconciliation_self_loop_transpose_kernel,
    _reconciliation_self_loop_transpose_series_kernel,
    _exact_tree_self_loop_transpose_kernel,
    _accumulate_transfer_receiver_log_probability_vjp_kernel,
    _accumulate_reconciliation_event_vjp_kernel,
    _accumulate_gene_split_event_vjp_kernel,
    _reduce_max_transfer_vjp_kernel,
    _select_active_transfer_donor_sides_kernel,
    _accumulate_transfer_subtree_vjp_kernel,
)

_SUPPORTED_FLOAT_DTYPES = (torch.float32, torch.float64, torch.bfloat16)

# Triton warps per program for the three adjoint kernels that dominate the genewise gradient.
# Launch-shape tuning only: block sizes and every other constexpr are unchanged, so the
# arithmetic is identical. Not user settings -- they are properties of the GPU the kernels run
# on, measured by benchmark/cc/sweep_num_warps.py and benchmark/cc/bwd_kernels.py --mode warps.
#
# The transfer-subtree kernel sat at 4 because 8, though 1.05x faster, used to move the gradient
# by 3.5 -- 2800x the run-to-run atomics noise. That was not the warp count's doing: the self-loop
# VJP kernel built each donor's valid receiver mass by subtracting two nearly equal numbers, and
# any reordering anywhere in the wave came back magnified through that cancellation. With the mass
# built additively (see ``_prepare_reconciliation_self_loop_vjp_kernel``) the same 8 warps move it
# by 5.1e-3 against a noise floor of 8.4e-3, i.e. no longer distinguishable from running the same
# code twice, and the 1.052x stands. Measured on an H100 NVL at 500 families, fitted theta.
_NUM_WARPS_PREPARE_SELF_LOOP_VJP = 8
_NUM_WARPS_SELF_LOOP_TRANSPOSE = 2
_NUM_WARPS_TRANSFER_SUBTREE_VJP = 8

# Attribute name under which the four valid-receiver scan tables are memoised on the caller's
# ``sp_subtree_start`` tensor. They depend only on the species tree, and a gradient launches the
# self-loop VJP kernel once per wave -- thousands of times -- so they must be built once.
_VALID_RECEIVER_TABLE_ATTR = "_gpurec_backward_valid_receiver_tables"


def _valid_receiver_tables(species_subtree_start, species_subtree_end, S, *, device):
    """The four scan tables of :func:`gpurec.core.valid_receivers.valid_receiver_index_tables`.

    Memoised on the ``sp_subtree_start`` tensor, which is one object per species tree for the whole
    run. Built from the caller's own tensors rather than from device/dtype-converted copies, so a
    conversion that allocates cannot hand every wave a fresh object and so a fresh table.
    """
    cached = getattr(species_subtree_start, _VALID_RECEIVER_TABLE_ATTR, None)
    if (
        cached is not None
        and int(cached[0].numel()) == int(S)
        and cached[0].device == device
    ):
        return cached
    tables = valid_receiver_index_tables(
        species_subtree_start.to(device=device),
        species_subtree_end.to(device=device),
        S,
    )
    setattr(species_subtree_start, _VALID_RECEIVER_TABLE_ATTR, tables)
    return tables


# Debug instrumentation for the fused Neumann-series kernel, off in production.
# When switched on with ``set_neumann_term_stat_collection(True)`` every wave's
# self-loop solve appends a [W] int32 tensor to ``_NEUMANN_TERM_COUNTS`` holding
# the number of Neumann terms each row block actually ran before its early exit.
# A module-level switch (not a function argument or an environment variable) so
# the production call path keeps its exact signature and costs nothing.
_COLLECT_NEUMANN_TERM_STATS = False
_NEUMANN_TERM_COUNTS = []

# A/B switch between the fused Neumann-series kernel (production) and the historical
# one-launch-per-term reference loop. Only benchmark/cc/test_neumann_exit.py flips it,
# to show the two agree bit for bit at neumann_term_tol=0.
_USE_FUSED_NEUMANN_SERIES = True

# The two adjoint self-loop implementations selectable through ``SolverOptions.adjoint_self_loop``.
# "series": sum the Neumann terms of (I - J^T)^-1 rhs, stopping early per row block.
# "exact":  solve (I - J^T) v = rhs directly by elimination on the species tree
#           (``_exact_tree_self_loop_transpose_kernel``), which is what the series converges to.
ADJOINT_SELF_LOOP_MODES = ("series", "exact")

# Debug instrumentation for the exact adjoint solve, off in production. When switched on with
# ``set_exact_adjoint_guard_trip_collection(True)`` every wave appends a [W, 2] int32 tensor to
# ``_EXACT_ADJOINT_GUARD_TRIPS``: column 0 counts a row's non-positive elimination pivots,
# column 1 flags a non-positive ``1 - transfer loop gain``. A module-level switch (not an
# argument) so the production call path keeps its exact signature and costs nothing.
_COLLECT_EXACT_ADJOINT_GUARD_TRIPS = False
_EXACT_ADJOINT_GUARD_TRIPS = []


def set_exact_adjoint_guard_trip_collection(enabled):
    """Turn the exact adjoint solve's per-row pivot-guard counters on or off."""
    global _COLLECT_EXACT_ADJOINT_GUARD_TRIPS
    _COLLECT_EXACT_ADJOINT_GUARD_TRIPS = bool(enabled)
    if not _COLLECT_EXACT_ADJOINT_GUARD_TRIPS:
        _EXACT_ADJOINT_GUARD_TRIPS.clear()


def set_fused_neumann_series(enabled):
    """Choose the fused series kernel (True) or the per-term reference loop (False)."""
    global _USE_FUSED_NEUMANN_SERIES
    _USE_FUSED_NEUMANN_SERIES = bool(enabled)


def set_neumann_term_stat_collection(enabled):
    """Turn per-row Neumann-term counting on or off and clear any counts held."""
    global _COLLECT_NEUMANN_TERM_STATS
    _COLLECT_NEUMANN_TERM_STATS = bool(enabled)
    _NEUMANN_TERM_COUNTS.clear()


def take_neumann_term_counts():
    """Return and clear the per-wave [W] int32 term counts collected so far."""
    counts = list(_NEUMANN_TERM_COUNTS)
    _NEUMANN_TERM_COUNTS.clear()
    return counts


def _tl_float_dtype(dtype):
    """Map a supported backward-state dtype to its Triton compute dtype."""
    if dtype == torch.float64:
        return tl.float64
    if dtype in (torch.float32, torch.bfloat16):
        return tl.float32
    raise TypeError(
        "backward kernel state must use torch.bfloat16, torch.float32, or "
        f"torch.float64, got {dtype}"
    )


def _device_scalar_param(param, *, device, dtype):
    """Return a one-element device tensor without extracting CUDA scalars."""
    if torch.is_tensor(param):
        if param.numel() != 1:
            raise ValueError("DTS backward scalar parameters must have one element")
        if param.device != device or param.dtype != dtype:
            param = param.to(device=device, dtype=dtype)
        return param.reshape(1).contiguous()
    return torch.tensor([param], device=device, dtype=dtype)


def _dts_layout_param_args(log_pD, log_pS, *, family_idx, S, device, dtype):
    """Return DTS parameter tensors plus a Triton addressing layout.

    With ``family_idx`` present, retained backward treats a one-dimensional
    tensor as family scalar rows before considering a shared ``[S]`` species
    vector.  Direct callers that need forward/backward parity when ``G == S``
    should use ``[G, 1]`` for family scalar rows or ``[G, S]`` for
    family/species rows.

    Layouts:
      0: shared scalar, tensor [1]
      1: shared species vector, tensor [S]
      2: family scalar, tensor [G] addressed by family_idx[parent]
      3: family species, tensor [G, S] addressed by family_idx[parent], s
    """

    def _normalize(param):
        if not torch.is_tensor(param):
            return _device_scalar_param(param, device=device, dtype=dtype), 0
        if param.device != device or param.dtype != dtype:
            param = param.to(device=device, dtype=dtype)
        try:
            layout = dts_backward_param_layout(
                param,
                S=S,
                family_indexed=family_idx is not None,
            )
        except ValueError as exc:
            raise ValueError(
                "DTS parameters must be scalar, [S], [G], [G, 1], or [G, S] "
                "for the DTS backward path"
            ) from exc
        layout_code = int(layout.code)
        if layout_code == 0:
            return param.reshape(1).contiguous(), 0
        if layout_code == 1:
            return param.contiguous(), 1
        if layout_code == 2:
            if param.ndim == 2:
                return param.reshape(int(param.shape[0])).contiguous(), 2
            return param.contiguous(), 2
        if layout_code == 3:
            return param.contiguous(), 3
        raise AssertionError("validated DTS backward layout reached unreachable branch")

    pD, layout_D = _normalize(log_pD)
    pS, layout_S = _normalize(log_pS)
    if layout_D != layout_S:
        raise ValueError("log_pD/log_pS must use the same DTS parameter layout")
    return pD, pS, layout_D


def _dts_grad_layout(grad, *, family_idx, S):
    """Return gradient addressing layout matching _dts_layout_param_args."""
    try:
        return int(
            dts_backward_param_layout(
                grad,
                S=S,
                family_indexed=family_idx is not None,
            ).code
        )
    except ValueError as exc:
        raise ValueError("unsupported DTS gradient layout") from exc


def _self_loop_constant_layout(const_tensor, family_idx, family_indexed):
    """Return addressing mode for self-loop constants.

    Modes:
      0: shared [S]
      1: row-expanded [W, S]
      2: family-indexed [G, S] addressed through family_idx[C]
    """
    if family_indexed:
        if family_idx is None:
            raise ValueError("family-indexed backward constants require family_idx")
        if const_tensor.ndim != 2:
            raise ValueError("family-indexed backward constants require [G, S] tensors")
        return 2
    if const_tensor.ndim == 2:
        return 1
    return 0


def _self_loop_leaf_log_probability_layout(
    use_leaf_index, leaf_logp, family_idx, family_indexed
):
    """Return addressing mode for leaf log-probabilities in the self-loop."""
    if not use_leaf_index:
        return 0
    if family_indexed:
        if family_idx is None:
            raise ValueError("family-indexed leaf log-probabilities require family_idx")
        if leaf_logp.ndim == 1:
            return 1
        if leaf_logp.ndim == 2:
            if int(leaf_logp.shape[1]) == 1:
                raise ValueError("family-indexed [G, 1] leaf_logp should be expanded to [G, S]")
            return 2
        raise ValueError("family-indexed leaf_logp must have shape [G] or [G, S]")
    if leaf_logp.numel() == 1:
        return 3
    return 0


def compute_active_adjoint_row_mask(rhs, threshold, *, use_pruning=True):
    """Build the row activity mask for backward pruning in one Triton launch.

    This is a private retained-kernel helper, not a public dtype policy.  The
    helper accepts fp32/fp64/bf16 CUDA tensors for standalone mask experiments;
    the public ``Pi_wave_backward`` path still supports only fp32/fp64 and
    rejects bf16 before calling this helper.
    """
    if rhs.ndim != 2:
        raise ValueError("rhs must be a 2D tensor")
    if rhs.device.type != "cuda":
        raise ValueError("compute_active_adjoint_row_mask requires a CUDA tensor")
    if rhs.dtype not in _SUPPORTED_FLOAT_DTYPES:
        raise ValueError(
            "compute_active_adjoint_row_mask supports fp32/fp64/bf16 tensors"
        )

    W, S = rhs.shape
    active_mask = torch.empty((W,), device=rhs.device, dtype=torch.bool)
    if W == 0:
        return active_mask

    BLOCK_S = min(256, triton.next_power_of_2(S))
    _select_active_adjoint_rows_kernel[(W,)](
        rhs,
        active_mask,
        float(threshold),
        S,
        rhs.stride(0),
        BLOCK_S,
        STRICT_GT=bool(not use_pruning),
        DTYPE=_tl_float_dtype(rhs.dtype),
    )
    return active_mask


def _solve_reconciliation_wave_vjp_2d(
    Pi_star, Pibar_star, ws, W, S,
    gene_split_log_likelihood,
    rhs,
    max_transfer, duplication_loss_const, Ebar, E, speciation_child1_const, speciation_child2_const,
    receiver_log_probs,
    species_child1, species_child2, leaf_term_wt,
    *,
    neumann_terms,
    neumann_term_tol,
    adjoint_self_loop,
    wide_row,
    leaf_species_idx,
    leaf_logp,
    leaf_fm_log,
    has_leaf_term,
    active_mask,
    species_parent,
    species_subtree_start,
    species_subtree_end,
    pibar_row_max,
    family_idx,
    const_layout,
    leaf_logp_mode,
    use_leaf_index,
    compact_level_ptr,
    compact_level_parents,
    compact_level_child1,
    compact_level_child2,
    grad_receiver_log_probs=None,
    use_receiver_weights=True,
    self_loop_grad_targets=None,
    initial_v=None,
    return_last_increment=False,
    reserved_scratch_bytes=None,
    pi_offset,
    pibar_offset,
    gene_split_offset=None,
):
    """Retained 2D row-block/full-species tree-reduction self-loop."""
    if Pi_star.device.type != "cuda":
        raise RuntimeError("GPUREC self-loop 2D fast path requires CUDA tensors")
    if Pi_star.dtype not in (torch.float32, torch.float64):
        raise RuntimeError(
            "2D self-loop fast path currently supports fp32/fp64 only",
        )
    fits_memory_budget, required_bytes, budget_bytes = proposal0_memory_gate(
        W,
        S,
        Pi_star.dtype,
        device=Pi_star.device,
        reserved_scratch_bytes=reserved_scratch_bytes,
    )
    if not fits_memory_budget:
        raise RuntimeError(
            "2D self-loop fast path estimated scratch "
            f"{required_bytes / (1024 ** 3):.2f} GiB above memory budget "
            f"{(budget_bytes or 0) / (1024 ** 3):.2f} GiB",
        )
    if adjoint_self_loop not in ADJOINT_SELF_LOOP_MODES:
        raise ValueError(
            f"adjoint_self_loop must be one of {ADJOINT_SELF_LOOP_MODES}, got {adjoint_self_loop!r}"
        )
    if const_layout not in (0, 1, 2):
        raise RuntimeError("unsupported self-loop constant layout")
    if use_leaf_index and leaf_logp_mode not in (0, 1, 2, 3):
        raise RuntimeError("unsupported leaf log-probability layout")

    device = Pi_star.device
    dtype = Pi_star.dtype
    grad_targets = {}
    if self_loop_grad_targets is not None:
        target_names = (
            "grad_log_pD",
            "grad_log_pS",
            "grad_E",
            "grad_Ebar",
            "grad_E_s1",
            "grad_E_s2",
            "grad_max_transfer",
        )
        grad_targets = dict(zip(target_names, self_loop_grad_targets[:7]))
    _validate_residual_tensors(
        Pi_star,
        Pibar_star=Pibar_star,
        gene_split_log_likelihood=gene_split_log_likelihood,
        rhs=rhs,
        max_transfer=max_transfer,
        duplication_loss_const=duplication_loss_const,
        Ebar=Ebar,
        E=E,
        speciation_child1_const=speciation_child1_const,
        speciation_child2_const=speciation_child2_const,
        receiver_log_probs=receiver_log_probs,
        leaf_term_wt=leaf_term_wt,
        leaf_logp=leaf_logp,
        pibar_row_max=pibar_row_max,
        grad_receiver_log_probs=grad_receiver_log_probs,
        initial_v=initial_v,
        **grad_targets,
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
    if gene_split_log_likelihood is not None:
        if gene_split_offset is None:
            raise ValueError("gene_split_offset is required for split-wave backward")
        gene_split_offset = _validate_offset_tensor(
            "gene_split_offset",
            gene_split_offset,
            rows=W,
            device=device,
            dtype=accumulator_dtype,
        )
    elif gene_split_offset is not None:
        raise ValueError("gene_split_offset is only valid for a split wave")
    receiver_log_probs = receiver_log_probs.contiguous()
    if species_parent is None:
        raise ValueError("species_parent is required for the retained 2D self-loop path")
    species_parent = species_parent.to(device=device, dtype=torch.int32).contiguous()
    if species_subtree_start is None or species_subtree_end is None:
        raise ValueError(
            "species_subtree_start and species_subtree_end are required for the retained 2D "
            "self-loop path: the valid receiver mass is built from the depth-first interval order"
        )
    (
        not_open_source,
        closed_source,
        not_open_index,
        closed_index,
    ) = _valid_receiver_tables(species_subtree_start, species_subtree_end, S, device=device)

    if (
        compact_level_ptr is None
        or compact_level_parents is None
        or compact_level_child1 is None
        or compact_level_child2 is None
    ):
        raise ValueError("compact species levels are required for the retained 2D self-loop path")
    compact_level_ptr = compact_level_ptr.to(device=device, dtype=torch.long).contiguous()
    compact_level_parents = compact_level_parents.to(device=device, dtype=torch.int32).contiguous()
    compact_level_child1 = compact_level_child1.to(device=device, dtype=torch.int32).contiguous()
    compact_level_child2 = compact_level_child2.to(device=device, dtype=torch.int32).contiguous()

    block_w = 1
    block_s = triton.next_power_of_2(S)
    block_nodes = 128
    n_row_blocks = triton.cdiv(W, block_w)
    scratch_shape = (W, S)

    # The buffers below are what this function RETURNS. Every kernel here writes only the rows
    # the adjoint pruner marked active (SKIP_INACTIVE_SCRATCH_ZERO), so with pruning on a pruned
    # row is never written at all. Left as torch.empty it hands back whatever the caching
    # allocator last left in that block, and the exact-HVP consumer reads those rows -- the
    # first-order gradient masks them away, which is why only the HVP noticed. That made
    # tests/test_fraction_missing_hvp.py (two identical computations, asserted equal bit for bit)
    # pass or fail on whether the recycled block happened to hold the same bytes both times.
    # An explicit zero is also what "pruned" means: the row's adjoint is below
    # adjoint_pruning_threshold and contributes nothing. Active rows are always fully written, so
    # their values are bit-for-bit unchanged; without pruning every row is written, so the
    # memsets are skipped entirely.
    returned_buffer = torch.zeros if active_mask is not None else torch.empty

    v_k = returned_buffer(scratch_shape, device=device, dtype=dtype)
    self_loop_diagonal = returned_buffer(scratch_shape, device=device, dtype=dtype)
    donor_adjoint_coefficient = returned_buffer(scratch_shape, device=device, dtype=dtype)
    receiver_mass = returned_buffer(scratch_shape, device=device, dtype=dtype)
    accum_self_loop_grads = self_loop_grad_targets is not None
    speciation_leaf_event_vjp = (
        None if accum_self_loop_grads else returned_buffer(scratch_shape, device=device, dtype=dtype)
    )
    speciation_child1_probability = returned_buffer(scratch_shape, device=device, dtype=dtype)
    speciation_child2_probability = returned_buffer(scratch_shape, device=device, dtype=dtype)
    use_exact_adjoint = adjoint_self_loop == "exact"
    # The forward flagged the rows whose lanes it could not hold under one row scale and solved
    # them in log space instead. The transposed solve underflows on exactly those rows, so they
    # go to the Neumann series and the exact elimination skips them. ``wide_row`` is None
    # whenever the forward flagged nothing, which is the ordinary case and leaves both paths
    # exactly as they were.
    wide_rows_in_wave = None
    if use_exact_adjoint and wide_row is not None:
        wide_rows_in_wave = wide_row[int(ws): int(ws) + int(W)].to(torch.bool)
    split_by_range = wide_rows_in_wave is not None
    # One [2, W, S] allocation instead of two [W, S] ones (same total bytes): the
    # fused series kernel ping-pongs the two Neumann term buffers by adding an
    # integer offset of 0 or W*S to a single base pointer, which Triton can carry
    # through a while loop (a pointer swap it cannot).
    #
    # The exact solve has no terms to ping-pong. It spends the same two [W, S] arrays on the two
    # of its three per-node numbers that need their own storage; the third shares
    # ``subtree_donor_adjoint``, which is pure scratch until the receiver-gradient kernel below
    # rewrites it. So the exact path allocates exactly what the series path does.
    term_pair = torch.empty(
        (2, 1, 1) if use_exact_adjoint else (2, W, S), device=device, dtype=dtype
    )
    spec_buf = term_pair[0]
    term_buf = term_pair[1]
    subtree_donor_adjoint = torch.empty(scratch_shape, device=device, dtype=dtype)
    if use_exact_adjoint:
        elimination_pair = torch.empty((2, W, S), device=device, dtype=dtype)
        subtree_donor_weight = elimination_pair[0]
        subtree_donor_constant = elimination_pair[1]
    # The prepare kernel builds each donor's valid receiver mass from two running sums, which need
    # one [W, S] slot each. No new allocation: on the exact path the elimination pair is pure
    # scratch until the tree solve seeds every species of it, and on the series path the term pair
    # is pure scratch until the first J^T application writes into it -- and both of those happen
    # after the prepare kernel has finished with them. Same trick, and same reason, as
    # ``subtree_donor_adjoint`` being scratch until the receiver-gradient kernel rewrites it.
    valid_receiver_scratch = elimination_pair if use_exact_adjoint else term_pair

    if pibar_row_max is None:
        raise ValueError("pibar_row_max is required for the retained 2D self-loop path")
    pibar_row_max = pibar_row_max.contiguous()
    skip_inactive_scratch_zero = True
    if family_idx is not None:
        family_idx = family_idx.to(device=device, dtype=torch.long).contiguous()
    else:
        family_idx = species_parent
    requested_has_leaf_term = bool(has_leaf_term)
    use_leaf_index = bool(use_leaf_index and requested_has_leaf_term)
    has_materialized_leaf_term = leaf_term_wt is not None
    if leaf_term_wt is None:
        leaf_term_wt = leaf_logp if use_leaf_index else Pi_star
    has_leaf_term = bool(
        requested_has_leaf_term
        and (use_leaf_index or has_materialized_leaf_term)
    )
    leaf_species_arg = leaf_species_idx if use_leaf_index else species_child1
    leaf_logp_arg = leaf_logp if use_leaf_index else leaf_term_wt
    use_fraction_missing = leaf_fm_log is not None
    # When there is no fraction-missing tensor the constexpr short-circuits the
    # off-hit load, so a valid-but-unused 1-element placeholder is enough.
    leaf_fm_log_arg = (
        leaf_fm_log.to(device=device, dtype=dtype).contiguous()
        if use_fraction_missing
        else torch.empty(1, device=device, dtype=dtype)
    )
    use_child_edge_self_loop = True

    launch_options = {"num_warps": 8}

    _prepare_reconciliation_self_loop_vjp_kernel[(n_row_blocks,)](
        Pi_star,
        Pibar_star,
        pi_offset,
        pibar_offset,
        pibar_row_max,
        gene_split_log_likelihood if gene_split_log_likelihood is not None else Pi_star,
        gene_split_offset if gene_split_offset is not None else Pi_star,
        gene_split_log_likelihood is not None,
        rhs,
        active_mask if active_mask is not None else rhs,
        max_transfer,
        duplication_loss_const,
        Ebar,
        E,
        speciation_child1_const,
        speciation_child2_const,
        receiver_log_probs,
        species_child1,
        species_child2,
        not_open_source,
        closed_source,
        not_open_index,
        closed_index,
        valid_receiver_scratch,
        leaf_term_wt,
        leaf_species_arg,
        leaf_logp_arg,
        leaf_fm_log_arg,
        family_idx,
        v_k,
        self_loop_diagonal,
        donor_adjoint_coefficient,
        receiver_mass,
        speciation_child1_probability,
        speciation_child2_probability,
        ws,
        W,
        S,
        Pi_star.stride(0),
        block_w,
        block_s,
        USE_LEAF_INDEX=bool(use_leaf_index),
        HAS_LEAF_TERM=bool(has_leaf_term),
        LEAF_LOGP_MODE=int(leaf_logp_mode),
        USE_FRACTION_MISSING=bool(use_fraction_missing),
        USE_ACTIVE_MASK=bool(active_mask is not None),
        SKIP_INACTIVE_SCRATCH_ZERO=bool(skip_inactive_scratch_zero),
        CONST_LAYOUT=int(const_layout),
        DTYPE=_tl_float_dtype(dtype),
        USE_CHILD_EDGE_SELF_LOOP=bool(use_child_edge_self_loop),
        USE_RECEIVER_WEIGHTS=bool(use_receiver_weights),
        num_warps=_NUM_WARPS_PREPARE_SELF_LOOP_VJP,
    )

    if initial_v is not None and not use_exact_adjoint:
        # Warm start: the series then iterates the fixed point v <- rhs + J^T v
        # from this v instead of summing Neumann terms from scratch. The exact solve
        # lands on that fixed point in one shot, so a starting point buys it nothing.
        if tuple(initial_v.shape) != scratch_shape:
            raise ValueError(
                f"initial_v shape {tuple(initial_v.shape)} does not match "
                f"wave scratch shape {scratch_shape}"
            )
        v_k.copy_(initial_v.contiguous())

    jt_options = {"num_warps": 2}
    if use_exact_adjoint:
        # One launch, no terms: solve (I - J^T) v = rhs by elimination on the species tree.
        collect_guard_trips = _COLLECT_EXACT_ADJOINT_GUARD_TRIPS
        guard_trips = (
            torch.zeros((W, 2), device=device, dtype=torch.int32)
            if collect_guard_trips
            else compact_level_ptr
        )
        # Three row sets, not two. The forward already handed some rows to the log sweeps, and
        # those go to the series here. The elimination takes the rest -- but it can also refuse a
        # row of its own accord, because the TRANSPOSE is a different operator with its own pivots
        # and its own loop gain, and it spills such a row into the same series mask. All of it
        # stays inside the pruner's active mask, so a pruned row is skipped by every launch.
        active_rows = (
            active_mask.reshape(W, -1).ne(0).any(dim=1)
            if active_mask is not None
            else torch.ones(W, device=device, dtype=torch.bool)
        )
        if split_by_range:
            exact_rows = (active_rows & ~wide_rows_in_wave).to(torch.int8)
            series_rows = (active_rows & wide_rows_in_wave).to(torch.int8)
        else:
            exact_rows = active_rows.to(torch.int8)
            series_rows = torch.zeros(W, device=device, dtype=torch.int8)
        _exact_tree_self_loop_transpose_kernel[(n_row_blocks,)](
            rhs,
            exact_rows,
            self_loop_diagonal,
            donor_adjoint_coefficient,
            receiver_mass,
            # In the child-edge layout the prepare kernel writes BOTH children's speciation
            # probabilities into this one array, indexed by the child, so each entry already
            # reads "speciate at my parent and follow the edge into me".
            speciation_child1_probability,
            species_parent,
            compact_level_ptr,
            compact_level_parents,
            compact_level_child1,
            compact_level_child2,
            subtree_donor_adjoint,
            subtree_donor_constant,
            subtree_donor_weight,
            v_k,
            guard_trips,
            series_rows,
            exact_conditioning_floor(dtype),
            W,
            S,
            block_w,
            block_s,
            block_nodes,
            compact_level_ptr.numel() - 1,
            USE_ACTIVE_MASK=True,
            WRITE_GUARD_TRIPS=bool(collect_guard_trips),
            DTYPE=_tl_float_dtype(dtype),
            num_warps=_NUM_WARPS_SELF_LOOP_TRANSPOSE,
        )
        if collect_guard_trips:
            _EXACT_ADJOINT_GUARD_TRIPS.append(guard_trips)
        # Always: the mask is empty in the ordinary case, and then every program returns on its
        # first load. Reading it back to skip the launch would cost a device-to-host copy per
        # wave, which is more than the empty launch. ``elimination_pair`` is [2, W, S] and dead
        # once the tree solve above has finished with it, so the series ping-pongs in it and this
        # path allocates nothing of its own. v_k still holds rhs for every masked row -- the
        # prepare kernel wrote it and the elimination either skipped or refused the row -- which
        # is exactly what the series' cold branch starts from.
        _reconciliation_self_loop_transpose_series_kernel[(n_row_blocks,)](
            rhs,
            elimination_pair,
            series_rows,
            self_loop_diagonal,
            donor_adjoint_coefficient,
            receiver_mass,
            speciation_child1_probability,
            speciation_child2_probability,
            species_child1,
            species_child2,
            species_parent,
            compact_level_ptr,
            compact_level_parents,
            compact_level_child1,
            compact_level_child2,
            subtree_donor_adjoint,
            v_k,
            compact_level_ptr,
            float(dtype_scaled_self_loop_tol(neumann_term_tol, dtype)),
            int(neumann_terms),
            W,
            S,
            block_w,
            block_s,
            block_nodes,
            compact_level_ptr.numel() - 1,
            USE_ACTIVE_MASK=True,
            SKIP_INACTIVE_SCRATCH_ZERO=bool(skip_inactive_scratch_zero),
            FIXED_POINT_UPDATE=False,
            DTYPE=_tl_float_dtype(dtype),
            USE_CHILD_EDGE_SELF_LOOP=bool(use_child_edge_self_loop),
            COLLECT_TERMS=False,
            WRITE_LAST_TERM=False,
            num_warps=_NUM_WARPS_SELF_LOOP_TRANSPOSE,
        )
    elif int(neumann_terms) > 0 and not _USE_FUSED_NEUMANN_SERIES:
        # Reference path: one launch per Neumann term, no early exit. Kept only so
        # benchmark/cc/test_neumann_exit.py can show the fused kernel below repro-
        # duces it bit for bit; production never takes this branch.
        for n in range(int(neumann_terms)):
            if initial_v is not None:
                term_in = v_k
                term_out = spec_buf
            else:
                term_in = rhs if n == 0 else (spec_buf if n % 2 == 1 else term_buf)
                term_out = spec_buf if n % 2 == 0 else term_buf
            _apply_reconciliation_self_loop_transpose_kernel[(n_row_blocks,)](
                term_in,
                term_out,
                rhs,
                active_mask if active_mask is not None else rhs,
                self_loop_diagonal,
                donor_adjoint_coefficient,
                receiver_mass,
                speciation_child1_probability,
                speciation_child2_probability,
                species_child1,
                species_child2,
                species_parent,
                compact_level_ptr,
                compact_level_parents,
                compact_level_child1,
                compact_level_child2,
                subtree_donor_adjoint,
                v_k,
                W,
                S,
                block_w,
                block_s,
                block_nodes,
                compact_level_ptr.numel() - 1,
                USE_ACTIVE_MASK=bool(active_mask is not None),
                SKIP_INACTIVE_SCRATCH_ZERO=bool(skip_inactive_scratch_zero),
                FIXED_POINT_UPDATE=bool(initial_v is not None),
                DTYPE=_tl_float_dtype(dtype),
                USE_CHILD_EDGE_SELF_LOOP=bool(use_child_edge_self_loop),
                OUTPUT_A=False,
                ACCUMULATE_V=True,
                num_warps=_NUM_WARPS_SELF_LOOP_TRANSPOSE,
            )
        if return_last_increment and initial_v is None:
            # The reference path leaves the final term in whichever buffer the
            # launch parity picked; mirror it where the diagnostic below looks.
            if (int(neumann_terms) - 1) % 2 != 0:
                spec_buf.copy_(term_buf)
    elif int(neumann_terms) > 0:
        # One launch for the WHOLE Neumann series (was one launch per term). Each
        # program runs its row block's terms in-kernel and stops as soon as the
        # block's largest remaining term is at or below
        # neumann_term_tol (rescaled to this solve's dtype) * (block max |v_k|).
        # The self-loop operator is a strong contraction (spectral radius ~0.04), so
        # the tail terms are below the working dtype's resolution and the launches
        # that carried them did no useful work.
        collect_terms = _COLLECT_NEUMANN_TERM_STATS
        terms_taken = (
            torch.zeros((W,), device=device, dtype=torch.int32)
            if collect_terms
            else compact_level_ptr
        )
        write_last_term = bool(return_last_increment and initial_v is None)
        _reconciliation_self_loop_transpose_series_kernel[(n_row_blocks,)](
            rhs,
            term_pair,
            active_mask if active_mask is not None else rhs,
            self_loop_diagonal,
            donor_adjoint_coefficient,
            receiver_mass,
            speciation_child1_probability,
            speciation_child2_probability,
            species_child1,
            species_child2,
            species_parent,
            compact_level_ptr,
            compact_level_parents,
            compact_level_child1,
            compact_level_child2,
            subtree_donor_adjoint,
            v_k,
            terms_taken,
            dtype_scaled_self_loop_tol(neumann_term_tol, dtype),
            int(neumann_terms),
            W,
            S,
            block_w,
            block_s,
            block_nodes,
            compact_level_ptr.numel() - 1,
            USE_ACTIVE_MASK=bool(active_mask is not None),
            SKIP_INACTIVE_SCRATCH_ZERO=bool(skip_inactive_scratch_zero),
            FIXED_POINT_UPDATE=bool(initial_v is not None),
            DTYPE=_tl_float_dtype(dtype),
            USE_CHILD_EDGE_SELF_LOOP=bool(use_child_edge_self_loop),
            COLLECT_TERMS=bool(collect_terms),
            WRITE_LAST_TERM=write_last_term,
            num_warps=_NUM_WARPS_SELF_LOOP_TRANSPOSE,
        )
        if collect_terms:
            _NEUMANN_TERM_COUNTS.append(terms_taken)
    # Per-row relative size of the last Neumann increment = validated stiffness predictor.
    # Computed before the param-store kernel runs (which may reuse scratch buffers).
    last_increment_relres = None
    if return_last_increment and use_exact_adjoint:
        # There is no remaining increment: the solve is exact, so the residual it would measure
        # is zero up to rounding. Reported as such rather than left None, which every caller of
        # this diagnostic would then have to special-case.
        last_increment_relres = torch.zeros(W, device=device, dtype=torch.float32)
    elif (
        return_last_increment
        and initial_v is None
        and int(neumann_terms) > 0
    ):
        # The series kernel mirrors each row's final term into term_pair[0]
        # (= spec_buf) whatever term it stopped at, so no parity bookkeeping.
        last_buf = spec_buf
        positive_floor = torch.finfo(torch.float32).tiny
        last_increment_norm = last_buf.float().norm(dim=1)
        adjoint_norm = v_k.float().norm(dim=1).clamp_min(positive_floor)
        relres = last_increment_norm / adjoint_norm
        # Inactive (pruned) rows hold uninitialized scratch — their adjoint is negligible,
        # so treat them as converged (0) rather than letting garbage pollute the per-family max.
        if active_mask is not None:
            row_active = active_mask.reshape(active_mask.shape[0], -1).ne(0).any(dim=1)
            relres = torch.where(row_active, relres, torch.zeros_like(relres))
        last_increment_relres = relres

    if accum_self_loop_grads:
        (
            grad_log_pD_ptr,
            grad_log_pS_ptr,
            grad_E_ptr,
            grad_Ebar_ptr,
            grad_E_s1_ptr,
            grad_E_s2_ptr,
            grad_mt_ptr,
            param_grad_vector,
        ) = self_loop_grad_targets
        speciation_leaf_event_vjp_ptr = self_loop_diagonal
    else:
        grad_log_pD_ptr = self_loop_diagonal
        grad_log_pS_ptr = self_loop_diagonal
        grad_E_ptr = self_loop_diagonal
        grad_Ebar_ptr = self_loop_diagonal
        grad_E_s1_ptr = self_loop_diagonal
        grad_E_s2_ptr = self_loop_diagonal
        grad_mt_ptr = self_loop_diagonal
        param_grad_vector = False
        speciation_leaf_event_vjp_ptr = speciation_leaf_event_vjp

    if grad_receiver_log_probs is not None:
        _accumulate_transfer_receiver_log_probability_vjp_kernel[(n_row_blocks,)](
            v_k,
            active_mask if active_mask is not None else rhs,
            donor_adjoint_coefficient,
            receiver_mass,
            compact_level_ptr,
            compact_level_parents,
            compact_level_child1,
            compact_level_child2,
            subtree_donor_adjoint,
            grad_receiver_log_probs,
            W,
            S,
            block_w,
            block_s,
            block_nodes,
            compact_level_ptr.numel() - 1,
            USE_ACTIVE_MASK=bool(active_mask is not None),
            DTYPE=_tl_float_dtype(dtype),
            **jt_options,
        )

    _accumulate_reconciliation_event_vjp_kernel[(n_row_blocks,)](
        Pi_star,
        Pibar_star,
        pi_offset,
        pibar_offset,
        gene_split_log_likelihood if gene_split_log_likelihood is not None else Pi_star,
        gene_split_offset if gene_split_offset is not None else Pi_star,
        gene_split_log_likelihood is not None,
        v_k,
        active_mask if active_mask is not None else rhs,
        max_transfer,
        duplication_loss_const,
        Ebar,
        E,
        speciation_child1_const,
        speciation_child2_const,
        species_child1,
        species_child2,
        leaf_term_wt,
        leaf_species_arg,
        leaf_logp_arg,
        leaf_fm_log_arg,
        family_idx,
        grad_log_pD_ptr,
        grad_log_pS_ptr,
        grad_E_ptr,
        grad_Ebar_ptr,
        grad_E_s1_ptr,
        grad_E_s2_ptr,
        grad_mt_ptr,
        self_loop_diagonal,
        donor_adjoint_coefficient,
        receiver_mass,
        speciation_leaf_event_vjp_ptr,
        speciation_child1_probability,
        speciation_child2_probability,
        ws,
        W,
        S,
        Pi_star.stride(0),
        block_w,
        block_s,
        USE_LEAF_INDEX=bool(use_leaf_index),
        HAS_LEAF_TERM=bool(has_leaf_term),
        LEAF_LOGP_MODE=int(leaf_logp_mode),
        USE_FRACTION_MISSING=bool(use_fraction_missing),
        USE_ACTIVE_MASK=bool(active_mask is not None),
        CONST_LAYOUT=int(const_layout),
        ACCUM_GRADS=bool(accum_self_loop_grads),
        PARAM_GRAD_VECTOR=bool(param_grad_vector),
        DTYPE=_tl_float_dtype(dtype),
        **launch_options,
    )

    if accum_self_loop_grads:
        base = (v_k, None, None, None, None, None, None)
    else:
        duplication_loss_event_vjp = self_loop_diagonal
        transfer_loss_event_vjp = donor_adjoint_coefficient
        transfer_event_vjp = receiver_mass
        speciation_child1_event_vjp = speciation_child1_probability
        speciation_child2_event_vjp = speciation_child2_probability
        base = (
            v_k,
            duplication_loss_event_vjp,
            transfer_loss_event_vjp,
            transfer_event_vjp,
            speciation_leaf_event_vjp,
            speciation_child1_event_vjp,
            speciation_child2_event_vjp,
        )
    if return_last_increment:
        return (*base, last_increment_relres)
    return base


def solve_reconciliation_wave_vjp(
    Pi_star, Pibar_star, ws, W, S,
    gene_split_log_likelihood,
    rhs,
    max_transfer, duplication_loss_const, Ebar, E, speciation_child1_const, speciation_child2_const,
    receiver_log_probs,
    species_child1, species_child2, leaf_term_wt,
    neumann_terms=3,
    leaf_species_idx=None,
    leaf_logp=None,
    leaf_fm_log=None,
    has_leaf_term=True,
    active_mask=None,
    species_parent=None,
    species_subtree_start=None,
    species_subtree_end=None,
    pibar_row_max=None,
    family_idx=None,
    family_indexed_consts=False,
    compact_level_ptr=None,
    compact_level_parents=None,
    compact_level_child1=None,
    compact_level_child2=None,
    grad_receiver_log_probs=None,
    use_receiver_weights=True,
    self_loop_grad_targets=None,
    initial_v=None,
    return_last_increment=False,
    reserved_scratch_bytes=None,
    *,
    neumann_term_tol,
    adjoint_self_loop,
    wide_row,
    pi_offset,
    pibar_offset,
    gene_split_offset=None,
):
    """Fused backward: precompute + Neumann + param VJP in one kernel per wave.

    Args:
        Pi_star: [C, S] converged Pi
        Pibar_star: [C, S] converged Pibar
        ws: wave start offset
        W: wave size
        S: number of species
        gene_split_log_likelihood: [W, S] or None
        rhs: [W, S] incoming adjoint
        max_transfer, duplication_loss_const, Ebar, E, speciation_child1_const, speciation_child2_const:
            [S], [W, S], or [G, S] when family_indexed_consts=True
        species_child1, species_child2: [S] long
        leaf_term_wt: [W, S]
        neumann_terms: int, the maximum number of Neumann terms
        neumann_term_tol: stop a row block's series once its largest remaining
            term is <= neumann_term_tol * (that block's max |v_k|).
            0.0 disables the early exit (full ``neumann_terms`` every row).
        adjoint_self_loop: "series" to sum Neumann terms, "exact" to solve
            (I - J^T) v = rhs directly by elimination on the species tree. "exact"
            ignores ``neumann_terms``, ``neumann_term_tol`` and ``initial_v``: it
            returns the v the series converges to.
        leaf_species_idx: optional [C] row -> species leaf index, -1 for non-leaves
        leaf_logp: optional [S], [G], or [G, S] log_pS values used with
            leaf_species_idx

    Returns:
        v_k: [W, S] Neumann-solved adjoint
        Per-event VJP tensors for duplication-loss, transfer-loss, transfer,
        combined speciation/leaf, and the two speciation child assignments.
    """
    requested_has_leaf_term = bool(has_leaf_term)
    use_leaf_index = (
        requested_has_leaf_term
        and leaf_species_idx is not None
        and leaf_logp is not None
    )
    const_layout = _self_loop_constant_layout(
        duplication_loss_const, family_idx, bool(family_indexed_consts)
    )
    if bool(family_indexed_consts) and use_leaf_index:
        if leaf_logp.ndim == 1:
            leaf_logp = leaf_logp.unsqueeze(-1).expand(-1, S).contiguous()
        elif leaf_logp.ndim == 2 and int(leaf_logp.shape[1]) == 1:
            leaf_logp = leaf_logp.expand(-1, S).contiguous()
        else:
            leaf_logp = leaf_logp.contiguous()
    leaf_logp_mode = _self_loop_leaf_log_probability_layout(
        use_leaf_index, leaf_logp, family_idx, bool(family_indexed_consts)
    )
    if leaf_fm_log is not None and leaf_logp_mode in (1, 3):
        # The off-hit leaf baseline is log_pS[s] + log2(fm_s), which needs a
        # per-species log_pS column. Modes 1/3 carry only a family-scalar
        # log_pS, so the per-column baseline is undefined there.
        raise NotImplementedError(
            "fraction_missing requires a per-species log_pS layout in the Pi "
            "backward (leaf_logp_mode 0 [S] or 2 [G, S]); got mode "
            f"{leaf_logp_mode} (family-scalar log_pS)."
        )
    if family_idx is not None:
        family_idx = family_idx.to(device=Pi_star.device, dtype=torch.long).contiguous()
    if species_parent is None:
        raise ValueError("species_parent is required for the retained backward fast path")
    species_parent = species_parent.to(device=Pi_star.device).contiguous()
    if Pi_star.device.type == "cuda" and species_parent.dtype != torch.int32:
        species_parent = species_parent.to(dtype=torch.int32)
    if pibar_row_max is None:
        raise ValueError("pibar_row_max is required for the retained backward fast path")
    pibar_row_max = pibar_row_max.contiguous()

    return _solve_reconciliation_wave_vjp_2d(
        Pi_star,
        Pibar_star,
        ws,
        W,
        S,
        gene_split_log_likelihood,
        rhs,
        max_transfer,
        duplication_loss_const,
        Ebar,
        E,
        speciation_child1_const,
        speciation_child2_const,
        receiver_log_probs,
        species_child1,
        species_child2,
        leaf_term_wt,
        neumann_terms=neumann_terms,
        neumann_term_tol=neumann_term_tol,
        adjoint_self_loop=adjoint_self_loop,
        wide_row=wide_row,
        leaf_species_idx=leaf_species_idx,
        leaf_logp=leaf_logp,
        leaf_fm_log=leaf_fm_log,
        has_leaf_term=requested_has_leaf_term,
        active_mask=active_mask,
        species_parent=species_parent,
        species_subtree_start=species_subtree_start,
        species_subtree_end=species_subtree_end,
        pibar_row_max=pibar_row_max,
        family_idx=family_idx,
        const_layout=const_layout,
        leaf_logp_mode=leaf_logp_mode,
        use_leaf_index=use_leaf_index,
        compact_level_ptr=compact_level_ptr,
        compact_level_parents=compact_level_parents,
        compact_level_child1=compact_level_child1,
        compact_level_child2=compact_level_child2,
        grad_receiver_log_probs=grad_receiver_log_probs,
        use_receiver_weights=use_receiver_weights,
        self_loop_grad_targets=self_loop_grad_targets,
        initial_v=initial_v,
        return_last_increment=return_last_increment,
        reserved_scratch_bytes=reserved_scratch_bytes,
        pi_offset=pi_offset,
        pibar_offset=pibar_offset,
        gene_split_offset=gene_split_offset,
    )


# =========================================================================
# Cross-clade DTS backward kernel
# =========================================================================


def accumulate_gene_split_event_vjp(
    Pi_star, Pibar_star, v_k, ws,
    split_left_rows, split_right_rows, reduce_idx, log_split_probs,
    log_pD, log_pS,
    species_child1, species_child2,
    accumulated_rhs,
    S,
    active_mask=None,
    use_atomics=True,
    merge_s_term=False,
    grad_log_pD=None,
    grad_log_pS=None,
    grad_max_transfer=None,
    accum_param_reductions=False,
    accum_max_transfer_reduction=False,
    output_donor_adjoint=False,
    output_active_donor_sides=False,
    pibar_side_threshold=0.0,
    max_transfer=None,
    pibar_row_max=None,
    grad_max_transfer_two_stage=False,
    grad_max_transfer_two_stage_tile_splits=128,
    skip_inactive_pibar_output_zero=False,
    family_idx=None,
    *,
    pi_offset,
    pibar_offset,
):
    """Fused DTS backward with direct Pi-adjoint accumulation."""
    n_splits = split_left_rows.shape[0]
    device = Pi_star.device
    dtype = Pi_star.dtype
    expected_rows = int(Pi_star.shape[0])
    pi_offset = _validate_offset_tensor(
        "pi_offset",
        pi_offset,
        rows=expected_rows,
        device=device,
        residual_dtype=Pi_star.dtype,
    )
    pibar_offset = _validate_offset_tensor(
        "pibar_offset",
        pibar_offset,
        rows=expected_rows,
        device=device,
        dtype=pi_offset.dtype,
    )

    split_log_priors = (log_split_probs.squeeze(-1) if log_split_probs.ndim > 1 else log_split_probs).contiguous()
    family_idx_arg = None
    if family_idx is not None:
        family_idx_arg = family_idx.to(device=device, dtype=torch.long).contiguous()
    log_pD_arg, log_pS_arg, param_layout = _dts_layout_param_args(
        log_pD, log_pS, family_idx=family_idx_arg, S=S, device=device, dtype=dtype
    )
    device_scalar_params = False
    if param_layout == 0:
        device_scalar_params = True

    if accum_param_reductions and (grad_log_pD is None or grad_log_pS is None):
        raise ValueError("grad_log_pD/grad_log_pS are required when accumulating DTS scalar reductions")
    if accum_param_reductions:
        param_grad_layout = _dts_grad_layout(grad_log_pD, family_idx=family_idx_arg, S=S)
        if param_grad_layout != _dts_grad_layout(grad_log_pS, family_idx=family_idx_arg, S=S):
            raise ValueError("grad_log_pD/grad_log_pS must use the same DTS gradient layout")
    else:
        param_grad_layout = 0
    if accum_max_transfer_reduction and grad_max_transfer is None:
        raise ValueError(
            "grad_max_transfer is required when accumulating gene-split transfer reductions"
        )
    if accum_max_transfer_reduction:
        if grad_max_transfer.numel() == 1 or (grad_max_transfer.ndim == 1 and int(grad_max_transfer.shape[0]) == int(S)):
            grad_max_transfer_layout = 0
        elif family_idx_arg is not None and grad_max_transfer.ndim == 2 and int(grad_max_transfer.shape[1]) == int(S):
            grad_max_transfer_layout = 1
        else:
            raise ValueError(
                "gene-split max_transfer reduction target must be scalar, [S], or [G, S]"
            )
    else:
        grad_max_transfer_layout = 0
    if output_donor_adjoint and (max_transfer is None or pibar_row_max is None):
        raise ValueError(
            "max_transfer and pibar_row_max are required when outputting donor adjoints"
        )
    if output_donor_adjoint:
        if max_transfer.ndim == 1 and int(max_transfer.shape[0]) == int(S):
            max_transfer_layout = 0
        elif family_idx_arg is not None and max_transfer.ndim == 2 and int(max_transfer.shape[1]) == int(S):
            max_transfer_layout = 1
        else:
            raise ValueError(
                "max_transfer must have shape [S] or [G, S] when outputting donor adjoints"
            )
    else:
        max_transfer_layout = 0
    _validate_residual_tensors(
        Pi_star,
        Pibar_star=Pibar_star,
        v_k=v_k,
        log_split_probs=split_log_priors,
        log_pD=log_pD_arg,
        log_pS=log_pS_arg,
        accumulated_rhs=accumulated_rhs,
        grad_log_pD=grad_log_pD,
        grad_log_pS=grad_log_pS,
        grad_max_transfer=grad_max_transfer,
        max_transfer=max_transfer if output_donor_adjoint else None,
        pibar_row_max=pibar_row_max if output_donor_adjoint else None,
    )
    if output_donor_adjoint and pibar_row_max.numel() < Pi_star.shape[0]:
        raise ValueError("pibar_row_max must contain one row-max value per Pi row")
    if output_active_donor_sides and not output_donor_adjoint:
        raise ValueError("output_active_donor_sides requires output_donor_adjoint")
    if torch.is_tensor(pibar_side_threshold):
        if pibar_side_threshold.numel() != 1:
            raise ValueError("pibar_side_threshold tensor must contain one value")
        side_threshold_enabled = bool(output_active_donor_sides)
        side_active_threshold_arg = _device_scalar_param(
            pibar_side_threshold, device=device, dtype=dtype
        )
    else:
        pibar_side_threshold = float(pibar_side_threshold)
        if pibar_side_threshold < 0.0:
            raise ValueError("pibar_side_threshold must be non-negative")
        side_threshold_enabled = bool(output_active_donor_sides and pibar_side_threshold > 0.0)
        side_active_threshold_arg = (
            torch.tensor([pibar_side_threshold], device=device, dtype=dtype)
            if side_threshold_enabled
            else None
        )

    if output_donor_adjoint:
        left_transfer_complement_vjp = None
        right_transfer_complement_vjp = None
    else:
        left_transfer_complement_vjp = torch.empty((n_splits, S), device=device, dtype=dtype)
        right_transfer_complement_vjp = torch.empty((n_splits, S), device=device, dtype=dtype)
    if output_donor_adjoint:
        donor_adjoint = torch.empty((2 * n_splits, S), device=device, dtype=dtype)
        total_donor_adjoint = torch.empty((2 * n_splits,), device=device, dtype=dtype)
    else:
        donor_adjoint = None
        total_donor_adjoint = None
    active_donor_side = (
        torch.empty((2 * n_splits,), device=device, dtype=torch.bool)
        if output_active_donor_sides
        else None
    )
    if accum_param_reductions:
        duplication_parameter_vjp = None
        speciation_parameter_vjp = None
    else:
        duplication_parameter_vjp = torch.empty(n_splits, device=device, dtype=dtype)
        speciation_parameter_vjp = torch.empty(n_splits, device=device, dtype=dtype)
    duplication_parameter_vjp_arg = grad_log_pD if accum_param_reductions else duplication_parameter_vjp
    speciation_parameter_vjp_arg = grad_log_pS if accum_param_reductions else speciation_parameter_vjp
    dummy = donor_adjoint if output_donor_adjoint else left_transfer_complement_vjp
    grad_log_pD_arg = grad_log_pD if accum_param_reductions else dummy
    grad_log_pS_arg = grad_log_pS if accum_param_reductions else dummy
    grad_max_transfer_arg = grad_max_transfer if accum_max_transfer_reduction else dummy
    donor_adjoint_arg = donor_adjoint if output_donor_adjoint else dummy
    total_donor_adjoint_arg = total_donor_adjoint if output_donor_adjoint else dummy
    active_donor_side_arg = active_donor_side if output_active_donor_sides else dummy
    max_transfer_arg = max_transfer.contiguous() if output_donor_adjoint and not max_transfer.is_contiguous() else max_transfer
    pibar_row_max_arg = (
        pibar_row_max.contiguous()
        if output_donor_adjoint and not pibar_row_max.is_contiguous()
        else pibar_row_max
    )
    max_transfer_arg = max_transfer_arg if output_donor_adjoint else dummy
    pibar_row_max_arg = pibar_row_max_arg if output_donor_adjoint else dummy
    side_active_threshold_arg = side_active_threshold_arg if side_threshold_enabled else dummy
    family_idx_kernel_arg = family_idx_arg if family_idx_arg is not None else split_left_rows
    grad_max_transfer_scalar = bool(accum_max_transfer_reduction and grad_max_transfer.numel() == 1)
    use_grad_max_transfer_two_stage = bool(
        grad_max_transfer_two_stage
        and accum_max_transfer_reduction
        and grad_max_transfer_layout == 0
        and not grad_max_transfer_scalar
        and grad_max_transfer.numel() == S
    )
    grad_max_transfer_two_stage_tile_splits = max(1, int(grad_max_transfer_two_stage_tile_splits))
    n_grad_max_transfer_tiles = triton.cdiv(n_splits, grad_max_transfer_two_stage_tile_splits)
    if use_grad_max_transfer_two_stage:
        grad_max_transfer_partial = torch.empty((n_grad_max_transfer_tiles, S), device=device, dtype=dtype)
        grad_max_transfer_partial.zero_()
    else:
        grad_max_transfer_partial = dummy

    stride_C = Pi_star.stride(0)
    BLOCK_S = min(256, triton.next_power_of_2(S))
    launch_options = {"num_warps": 8}

    _accumulate_gene_split_event_vjp_kernel[(n_splits,)](
        Pi_star, Pibar_star,
        pi_offset,
        pibar_offset,
        v_k,
        active_mask if active_mask is not None else v_k,
        split_left_rows, split_right_rows, reduce_idx, split_log_priors,
        log_pD_arg, log_pS_arg, family_idx_kernel_arg,
        species_child1, species_child2,
        accumulated_rhs,
        left_transfer_complement_vjp if left_transfer_complement_vjp is not None else donor_adjoint,
        right_transfer_complement_vjp if right_transfer_complement_vjp is not None else donor_adjoint,
        duplication_parameter_vjp_arg, speciation_parameter_vjp_arg,
        grad_log_pD_arg, grad_log_pS_arg, grad_max_transfer_arg,
        grad_max_transfer_partial,
        donor_adjoint_arg, total_donor_adjoint_arg, active_donor_side_arg, max_transfer_arg, pibar_row_max_arg,
        side_active_threshold_arg,
        ws, S, stride_C, BLOCK_S,
        USE_ACTIVE_MASK=bool(active_mask is not None),
        USE_ATOMICS=bool(use_atomics),
        MERGE_S_TERM=bool(merge_s_term),
        DEVICE_SCALAR_PARAMS=bool(device_scalar_params),
        PARAM_LAYOUT=int(param_layout),
        PARAM_GRAD_LAYOUT=int(param_grad_layout),
        MAX_TRANSFER_LAYOUT=int(max_transfer_layout),
        GRAD_MAX_TRANSFER_LAYOUT=int(grad_max_transfer_layout),
        ACCUM_PARAM_REDUCTIONS=bool(accum_param_reductions),
        ACCUM_MAX_TRANSFER_REDUCTION=bool(accum_max_transfer_reduction),
        GRAD_MAX_TRANSFER_SCALAR=bool(grad_max_transfer_scalar),
        GRAD_MAX_TRANSFER_TWO_STAGE=bool(use_grad_max_transfer_two_stage),
        GRAD_MAX_TRANSFER_TILE_SPLITS=grad_max_transfer_two_stage_tile_splits,
        OUTPUT_DONOR_ADJOINT=bool(output_donor_adjoint),
        OUTPUT_SIDE_ACTIVE=bool(output_active_donor_sides),
        SIDE_ACTIVE_THRESHOLD_ENABLED=side_threshold_enabled,
        SKIP_INACTIVE_PIBAR_OUTPUT_ZERO=bool(skip_inactive_pibar_output_zero),
        DTYPE=_tl_float_dtype(dtype),
        **launch_options,
    )

    if use_grad_max_transfer_two_stage:
        max_transfer_block_s = min(64, triton.next_power_of_2(S))
        max_transfer_block_tiles = 16
        _reduce_max_transfer_vjp_kernel[(triton.cdiv(S, max_transfer_block_s),)](
            grad_max_transfer_partial,
            grad_max_transfer,
            n_grad_max_transfer_tiles,
            S,
            max_transfer_block_tiles,
            max_transfer_block_s,
            DTYPE=_tl_float_dtype(dtype),
            num_warps=4,
        )

    if output_donor_adjoint:
        if output_active_donor_sides:
            return donor_adjoint, total_donor_adjoint, active_donor_side, duplication_parameter_vjp, speciation_parameter_vjp
        return donor_adjoint, total_donor_adjoint, duplication_parameter_vjp, speciation_parameter_vjp
    return left_transfer_complement_vjp, right_transfer_complement_vjp, duplication_parameter_vjp, speciation_parameter_vjp


# =========================================================================
# Transfer-complement VJP for cross-clade gradients
# =========================================================================


def accumulate_transfer_complement_vjp_from_donor_adjoint(
    Pi_star,
    receiver_log_probs,
    donor_adjoint,
    total_donor_adjoint,
    split_left_rows,
    split_right_rows,
    accumulated_rhs,
    S,
    active_mask=None,
    reduce_idx=None,
    pibar_row_max=None,
    skip_zero_donor_sides=False,
    active_donor_side=None,
    compact_level_ptr=None,
    compact_level_parents=None,
    compact_level_child1=None,
    compact_level_child2=None,
    grad_receiver_log_probs=None,
    use_receiver_weights=True,
    side_active_threshold=0.0,
):
    """Accumulate the transfer-complement VJP from staged donor adjoints."""
    n_splits = split_left_rows.shape[0]
    if n_splits == 0:
        return
    if active_mask is not None and reduce_idx is None:
        raise ValueError("reduce_idx is required when active_mask is provided")
    if pibar_row_max is None:
        raise ValueError("pibar_row_max is required for DTS-staged Pibar VJP")
    if torch.is_tensor(side_active_threshold):
        if side_active_threshold.numel() != 1:
            raise ValueError("side_active_threshold tensor must contain one value")
        side_active_threshold_enabled = True
        side_active_threshold_arg = _device_scalar_param(
            side_active_threshold, device=Pi_star.device, dtype=Pi_star.dtype
        )
    else:
        side_active_threshold = float(side_active_threshold)
        if side_active_threshold < 0.0:
            raise ValueError("side_active_threshold must be non-negative")
        side_active_threshold_enabled = side_active_threshold > 0.0
        side_active_threshold_arg = (
            torch.tensor([side_active_threshold], device=Pi_star.device, dtype=Pi_star.dtype)
            if side_active_threshold_enabled
            else None
        )

    BLOCK_S = min(256, triton.next_power_of_2(S))
    launch_options = {"num_warps": 4}
    stride_C = Pi_star.stride(0)
    if active_donor_side is not None:
        if active_donor_side.numel() != 2 * n_splits:
            raise ValueError("active_donor_side must have one entry per split side")
        active_donor_side = active_donor_side.contiguous()
    elif skip_zero_donor_sides:
        active_donor_side = torch.empty((2 * n_splits,), device=Pi_star.device, dtype=torch.bool)
        _select_active_transfer_donor_sides_kernel[(2 * n_splits,)](
            donor_adjoint,
            active_donor_side,
            side_active_threshold_arg if side_active_threshold_enabled else donor_adjoint,
            S,
            BLOCK_S,
            SIDE_ACTIVE_THRESHOLD_ENABLED=bool(side_active_threshold_enabled),
            DTYPE=_tl_float_dtype(Pi_star.dtype),
            **launch_options,
        )

    if (
        compact_level_ptr is None
        or compact_level_parents is None
        or compact_level_child1 is None
        or compact_level_child2 is None
    ):
        raise ValueError("compact species levels are required for Pibar VJP")
    if compact_level_ptr.numel() < 2:
        raise ValueError("compact_level_ptr must contain at least start and end offsets")
    compact_level_ptr = compact_level_ptr.contiguous()
    compact_level_parents = compact_level_parents.contiguous()
    compact_level_child1 = compact_level_child1.contiguous()
    compact_level_child2 = compact_level_child2.contiguous()
    _validate_residual_tensors(
        Pi_star,
        receiver_log_probs=receiver_log_probs,
        donor_adjoint=donor_adjoint,
        total_donor_adjoint=total_donor_adjoint,
        pibar_row_max=pibar_row_max,
        accumulated_rhs=accumulated_rhs,
        grad_receiver_log_probs=grad_receiver_log_probs,
    )
    receiver_log_probs = receiver_log_probs.contiguous()
    receiver_grad_arg = (
        grad_receiver_log_probs
        if grad_receiver_log_probs is not None
        else total_donor_adjoint
    )
    _accumulate_transfer_subtree_vjp_kernel[(2 * n_splits,)](
        Pi_star,
        receiver_log_probs,
        donor_adjoint,
        total_donor_adjoint,
        active_donor_side if active_donor_side is not None else total_donor_adjoint,
        split_left_rows,
        split_right_rows,
        reduce_idx if reduce_idx is not None else split_left_rows,
        active_mask if active_mask is not None else donor_adjoint,
        pibar_row_max,
        compact_level_ptr,
        compact_level_parents,
        compact_level_child1,
        compact_level_child2,
        accumulated_rhs,
        receiver_grad_arg,
        n_splits,
        S,
        stride_C,
        BLOCK_S,
        N_LEVELS=compact_level_ptr.numel() - 1,
        USE_ACTIVE_MASK=bool(active_mask is not None),
        USE_SIDE_ACTIVE=bool(active_donor_side is not None),
        ACCUM_RECEIVER_GRAD=bool(grad_receiver_log_probs is not None),
        USE_RECEIVER_WEIGHTS=bool(use_receiver_weights),
        DTYPE=_tl_float_dtype(Pi_star.dtype),
        num_warps=_NUM_WARPS_TRANSFER_SUBTREE_VJP,
    )
    return active_donor_side
