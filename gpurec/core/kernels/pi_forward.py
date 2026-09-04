import math

import torch
import triton
import triton.language as tl

__all__ = [
    "compute_dts_forward",
    "compute_exact_tree_self_loop",
    "compute_leaf_initial_wave_step",
    "compute_wave_step",
    "_load_event_log_probability",
    "_prepare_wave_launch",
    "_select_log_split_probs",
    "_tl_float_dtype",
    "_validate_offset_tensor",
    "_validate_residual_tensors",
]


_SUPPORTED_FLOAT_DTYPES = (torch.float32, torch.float64)

# Triton warps per program for the reconciliation-likelihood wave update
# (``_update_reconciliation_likelihood_kernel``). Launch-shape tuning only: BLOCK_S and every
# other constexpr are unchanged, so the arithmetic is identical. Not a user setting -- it is a
# property of the GPU the kernel runs on, measured by benchmark/cc/sweep_num_warps.py.
_NUM_WARPS_UPDATE_RECONCILIATION = 8

# Species lanes per tile inside ``_exact_tree_pi_self_loop_kernel``. Launch-shape tuning only:
# the tile loop covers the whole species row either way, so the same terms are summed. Not a user
# setting -- it is a property of the GPU the kernel runs on. Measured on the fused linear-space
# self-loop that used to share this tile loop, since the two ran the same row sweep: H100 NVL,
# S=2013, 500 families, one loss+gradient call, 256 -> 7.35 s, 512 -> 7.68 s, 1024 -> 7.98 s,
# 2048 -> 8.60 s. Wider tiles lose because each thread then has more of the 34-deep ancestor
# gather in flight at once. 256 also keeps the row summation order closest to the log-space
# kernel's, which uses the same tile width, so the two paths agree to fp32 rounding rather than a
# few times it.
_BLOCK_SPECIES_SELF_LOOP = 256

# Species-tree nodes per tile inside ``_exact_tree_pi_self_loop_kernel``'s two level walks, and
# that kernel's warps per program. Launch-shape tuning only: the walks visit every node of a level
# either way. 128 matches the backward pass's own level walk
# (``gpurec/core/kernels/wave_backward.py``'s ``block_nodes``), which reads the same
# ``compact_level_*`` tables.
_BLOCK_NODES_EXACT_TREE = 128
_NUM_WARPS_EXACT_TREE = 8

# Per-row species arrays the exact tree solve keeps live at once: the two affine coefficients
# alpha and gamma, plus two working arrays that each carry three roles in turn (see the kernel
# docstring's slot table).
_EXACT_TREE_SCRATCH_SLOTS = 4

# The reference dtype ``SolverOptions.exact_range_log2`` is written in: float32, whose smallest
# normal is 2**-126, so a 100-order default leaves 26 binary orders of margin for what the solve
# adds on top of the range it measured at entry. NOT a setting -- it is the property of the format
# the configured number is quoted against.
_EXACT_RANGE_REFERENCE_DTYPE = torch.float32


def exact_conditioning_floor(dtype) -> float:
    """Smallest divisor the exact elimination is allowed to use, for ``dtype``.

    Every lane divides by its pivot and the whole row divides once by ``1 - loop gain``. Both are
    one minus a probability, so they are order-1 numbers and a margin can be compared against them
    directly. A margin ``m`` costs about ``eps/m`` in relative error, so ``sqrt(eps)`` is the point
    where half the dtype's digits are gone: float32 -> 3.5e-4, float64 -> 1.5e-8. Beyond it the row
    goes to the log-space path, which has no such division.

    NOT a setting: it is a property of the arithmetic, not of a dataset.
    """
    return float(torch.finfo(dtype).eps) ** 0.5


def exact_range_for_dtype(range_log2, dtype) -> float:
    """Rescale the configured exact-solve range limit to the exponent range of ``dtype``.

    The configured number says "this many binary orders below the row maximum still fit", quoted
    against float32. Another dtype fits a different number of them, in proportion to its exponent
    range, so the factor is the ratio of smallest-normal exponents:

      * float32 -> factor exactly 1.0, i.e. the configured value;
      * float64 -> factor 1022/126 = 8.11, so the default 100 becomes 811, well inside float64's
        own 1022 and keeping the same 26/126 fraction of headroom.

    Without this a float64 solve would hand the log path every row float32 could not hold, and
    never use the range float64 actually has.
    """
    reference_orders = -math.log2(torch.finfo(_EXACT_RANGE_REFERENCE_DTYPE).tiny)
    return float(range_log2) * (-math.log2(torch.finfo(dtype).tiny) / reference_orders)


def _tl_float_dtype(dtype):
    """Map a supported PyTorch model dtype to its Triton scalar dtype."""
    if dtype == torch.float32:
        return tl.float32
    if dtype == torch.float64:
        return tl.float64
    raise TypeError(f"kernel state must use torch.float32 or torch.float64, got {dtype}")


def _validate_offset_tensor(
    name, value, *, rows, device, dtype=None, residual_dtype=None
):
    """Return a contiguous row-offset tensor after structural dtype checks."""
    if value.ndim != 1 or int(value.shape[0]) != int(rows):
        raise ValueError(f"{name} must have shape [{int(rows)}]")
    if value.dtype not in _SUPPORTED_FLOAT_DTYPES:
        raise TypeError(f"{name} must use torch.float32 or torch.float64")
    if residual_dtype == torch.float64 and value.dtype != torch.float64:
        raise TypeError(f"{name} must not be narrower than the residual dtype")
    if dtype is not None and value.dtype != dtype:
        raise TypeError(f"{name} must match accumulator dtype {dtype}")
    if value.device != device:
        raise ValueError(f"{name} must be on {device}")
    return value.contiguous()


def _validate_residual_tensors(reference, /, **tensors) -> None:
    """Require dense kernel tensors to share the model dtype and device.

    Centered offsets are intentionally excluded: they use the accumulator
    dtype and are validated separately by :func:`_validate_offset_tensor`.
    """
    if not torch.is_tensor(reference):
        raise TypeError("residual reference must be a tensor")
    if reference.dtype not in _SUPPORTED_FLOAT_DTYPES:
        raise TypeError("residual tensors must use torch.float32 or torch.float64")
    for name, value in tensors.items():
        if value is None:
            continue
        if not torch.is_tensor(value):
            raise TypeError(f"{name} must be a tensor")
        if value.dtype != reference.dtype:
            raise TypeError(
                f"{name} must match residual dtype {reference.dtype}, got {value.dtype}"
            )
        if value.device != reference.device:
            raise ValueError(f"{name} must be on {reference.device}")


def _select_log_split_probs(meta, dtype):
    """Return preprocessing-owned split probabilities for a kernel dtype."""
    variants = meta.get("_log_split_probs_by_dtype")
    if variants is not None:
        try:
            return variants[dtype]
        except KeyError as exc:
            raise TypeError(f"no split-probability tensor for residual dtype {dtype}") from exc
    value = meta.get("log_split_probs")
    if value is None and "sl" in meta:
        value = torch.zeros(
            int(meta["sl"].numel()), device=meta["sl"].device, dtype=dtype
        )
    return value


def _prepare_wave_launch(S: int, const_tensor) -> tuple[int, int]:
    const_row_stride = 0 if int(const_tensor.shape[0]) == 1 else int(const_tensor.stride(0))
    return int(min(256, triton.next_power_of_2(S))), const_row_stride


@triton.jit
def _load_event_log_probability(
    param,
    family,
    s_offs,
    mask,
    S: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    BY_SPECIES: tl.constexpr,
    BLOCK_S: tl.constexpr,
    DTYPE: tl.constexpr,
):
    """Load and broadcast one configured rate layout."""
    NEG_INF: tl.constexpr = -float("inf")
    if BY_SPECIES:
        return tl.load(param + family * ROW_STRIDE + s_offs, mask=mask, other=NEG_INF)
    family_rate = tl.load(param + family * ROW_STRIDE)
    return family_rate + tl.zeros([BLOCK_S], dtype=DTYPE)


@triton.jit
def _compute_total_receiver_mass(
    Pi_ptr,
    receiver_log_probs_ptr,
    row_base,
    S: tl.constexpr,
    BLOCK_S: tl.constexpr,
    USE_RECEIVER_WEIGHTS: tl.constexpr,
    TRACK_RAW_MAX: tl.constexpr,
    DTYPE: tl.constexpr,
):
    NEG_INF: tl.constexpr = -float("inf")
    row_max = tl.full([1], value=NEG_INF, dtype=DTYPE)
    total_receiver_mass = tl.full([1], value=0.0, dtype=DTYPE)
    reconciliation_row_max = tl.full([1], value=NEG_INF, dtype=DTYPE)
    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        mask = s_offs < S
        pi_val = tl.load(Pi_ptr + row_base + s_offs, mask=mask, other=NEG_INF)
        if TRACK_RAW_MAX:
            reconciliation_row_max = tl.maximum(reconciliation_row_max, tl.max(pi_val, axis=0))
        if USE_RECEIVER_WEIGHTS:
            receiver_log_probability = tl.load(receiver_log_probs_ptr + s_offs, mask=mask, other=NEG_INF)
            receiver_weighted_reconciliation_log_likelihood = receiver_log_probability + pi_val
        else:
            receiver_weighted_reconciliation_log_likelihood = pi_val
        new_max = tl.maximum(row_max, tl.max(receiver_weighted_reconciliation_log_likelihood, axis=0))
        new_max_safe = tl.where(new_max != NEG_INF, new_max, tl.zeros_like(new_max))
        previous = tl.where(
            row_max != NEG_INF,
            total_receiver_mass * tl.exp2(row_max - new_max_safe),
            tl.zeros_like(total_receiver_mass),
        )
        current = tl.sum(tl.exp2(receiver_weighted_reconciliation_log_likelihood - new_max_safe), axis=0)
        total_receiver_mass = previous + current
        row_max = new_max
    return row_max, total_receiver_mass, reconciliation_row_max


@triton.jit
def _compute_transfer_complement(
    Pi_ptr,
    receiver_log_probs_ptr,
    row_base,
    s_offs,
    mask,
    row_max,
    total_receiver_mass,
    max_transfer,
    species_parent_ptr,
    S: tl.constexpr,
    BLOCK_S: tl.constexpr,
    MAX_ANCESTOR_DEPTH: tl.constexpr,
    USE_RECEIVER_WEIGHTS: tl.constexpr,
    DTYPE: tl.constexpr,
):
    NEG_INF: tl.constexpr = -float("inf")
    excluded_ancestor_mass = tl.zeros([BLOCK_S], dtype=DTYPE)
    row_max_safe = tl.where(row_max != NEG_INF, row_max, tl.zeros_like(row_max))
    ancestor_species = s_offs.to(tl.int64)
    for _ in range(0, MAX_ANCESTOR_DEPTH):
        ancestor_valid = mask & (ancestor_species >= 0) & (ancestor_species < S)
        ancestor_reconciliation_log_likelihood = tl.load(
            Pi_ptr + row_base + ancestor_species, mask=ancestor_valid, other=NEG_INF
        )
        if USE_RECEIVER_WEIGHTS:
            ancestor_receiver_log_probability = tl.load(receiver_log_probs_ptr + ancestor_species, mask=ancestor_valid, other=NEG_INF)
            excluded_ancestor_mass += tl.where(
                ancestor_valid,
                tl.exp2(
                    ancestor_receiver_log_probability
                    + ancestor_reconciliation_log_likelihood
                    - row_max_safe
                ),
                tl.zeros([BLOCK_S], dtype=DTYPE),
            )
        else:
            excluded_ancestor_mass += tl.where(
                ancestor_valid,
                tl.exp2(ancestor_reconciliation_log_likelihood - row_max_safe),
                tl.zeros([BLOCK_S], dtype=DTYPE),
            )
        ancestor_species = tl.load(species_parent_ptr + ancestor_species, mask=ancestor_valid, other=-1).to(tl.int64)
    valid_receiver_mass = total_receiver_mass - excluded_ancestor_mass
    return tl.where(valid_receiver_mass > 0.0, tl.log2(valid_receiver_mass) + row_max + max_transfer, NEG_INF)


@triton.jit
def _initialize_leaf_reconciliation_likelihood_kernel(
    Pi_new_ptr,
    Pi_new_offset_ptr,
    ws,
    max_transfer_ptr,
    duplication_loss_const_ptr,
    Ebar_ptr,
    E_ptr,
    speciation_child1_const_ptr,
    speciation_child2_const_ptr,
    receiver_log_probs_ptr,
    species_child1_ptr,
    species_child2_ptr,
    species_subtree_start_ptr,
    species_subtree_end_ptr,
    leaf_species_ptr,
    leaf_logp_ptr,
    family_idx_ptr,
    S: tl.constexpr,
    stride: tl.constexpr,
    CONST_ROW_STRIDE: tl.constexpr,
    BLOCK_S: tl.constexpr,
    USE_RECEIVER_WEIGHTS: tl.constexpr,
    DTYPE: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
):
    NEG_LARGE: tl.constexpr = -float("inf")

    # int64: w ranges over the whole batch's clade rows, so global_row*stride
    # below can overflow int32 once total_clades * S exceeds 2^31.
    w = tl.program_id(0).to(tl.int64)
    global_row = ws + w
    family = tl.load(family_idx_ptr + global_row)
    const_base = family * CONST_ROW_STRIDE
    leaf_species = tl.load(leaf_species_ptr + global_row)
    leaf_start = tl.load(species_subtree_start_ptr + leaf_species)
    leaf_end = tl.load(species_subtree_end_ptr + leaf_species)
    if USE_RECEIVER_WEIGHTS:
        leaf_receiver_log_probability = tl.load(receiver_log_probs_ptr + leaf_species)
    else:
        leaf_receiver_log_probability = tl.zeros((), dtype=DTYPE)
    leaf_observation_log_probability = tl.load(leaf_logp_ptr + family * S + leaf_species)

    row_max = tl.full((), value=NEG_LARGE, dtype=DTYPE)
    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        mask = s_offs < S
        species_start = tl.load(species_subtree_start_ptr + s_offs, mask=mask, other=-1)
        descendant = (species_start >= leaf_start) & (species_start < leaf_end)
        leaf_hit = mask & (s_offs == leaf_species)
        const_offsets = const_base + s_offs
        max_transfer = tl.load(max_transfer_ptr + const_offsets, mask=mask, other=0.0)
        duplication_loss_const = tl.load(duplication_loss_const_ptr + const_offsets, mask=mask, other=NEG_LARGE)
        extinction_complement_log_probability = tl.load(
            Ebar_ptr + const_offsets, mask=mask, other=NEG_LARGE
        )
        extinction_log_probability = tl.load(
            E_ptr + const_offsets, mask=mask, other=NEG_LARGE
        )
        speciation_child1_const = tl.load(speciation_child1_const_ptr + const_offsets, mask=mask, other=NEG_LARGE)
        speciation_child2_const = tl.load(speciation_child2_const_ptr + const_offsets, mask=mask, other=NEG_LARGE)

        reconciliation_log_likelihood = tl.where(leaf_hit, leaf_observation_log_probability, NEG_LARGE)
        transfer_complement_log_likelihood = tl.where(
            ~descendant,
            max_transfer + leaf_receiver_log_probability + leaf_observation_log_probability,
            NEG_LARGE,
        )
        c1 = tl.load(species_child1_ptr + s_offs, mask=mask, other=S)
        c2 = tl.load(species_child2_ptr + s_offs, mask=mask, other=S)
        reconciliation_child1_log_likelihood = tl.where(mask & (c1 == leaf_species), leaf_observation_log_probability, NEG_LARGE)
        reconciliation_child2_log_likelihood = tl.where(mask & (c2 == leaf_species), leaf_observation_log_probability, NEG_LARGE)

        duplication_loss_log_term = duplication_loss_const + reconciliation_log_likelihood
        transfer_loss_log_term = (
            reconciliation_log_likelihood + extinction_complement_log_probability
        )
        transfer_log_term = (
            transfer_complement_log_likelihood + extinction_log_probability
        )
        speciation_child1_log_term = speciation_child1_const + reconciliation_child1_log_likelihood
        speciation_child2_log_term = speciation_child2_const + reconciliation_child2_log_likelihood
        leaf_logp = tl.load(leaf_logp_ptr + family * S + s_offs, mask=mask, other=NEG_LARGE)
        leaf_observation_log_term = tl.where(leaf_hit, leaf_logp, NEG_LARGE)
        logsumexp_max = tl.maximum(duplication_loss_log_term, transfer_loss_log_term)
        logsumexp_max = tl.maximum(logsumexp_max, transfer_log_term)
        logsumexp_max = tl.maximum(logsumexp_max, speciation_child1_log_term)
        logsumexp_max = tl.maximum(logsumexp_max, speciation_child2_log_term)
        logsumexp_max = tl.maximum(logsumexp_max, leaf_observation_log_term)
        logsumexp_max_safe = tl.where(logsumexp_max != NEG_LARGE, logsumexp_max, tl.zeros_like(logsumexp_max))
        local_event_scaled_mass = (
            tl.exp2(duplication_loss_log_term - logsumexp_max_safe)
            + tl.exp2(transfer_loss_log_term - logsumexp_max_safe)
            + tl.exp2(transfer_log_term - logsumexp_max_safe)
            + tl.exp2(speciation_child1_log_term - logsumexp_max_safe)
            + tl.exp2(speciation_child2_log_term - logsumexp_max_safe)
            + tl.exp2(leaf_observation_log_term - logsumexp_max_safe)
        )
        reconciliation_log_likelihood = (
            tl.log2(local_event_scaled_mass) + logsumexp_max
        )
        row_max = tl.maximum(
            row_max,
            tl.max(
                tl.where(mask, reconciliation_log_likelihood, NEG_LARGE),
                axis=0,
            ),
        )

    row_max_safe = tl.where(row_max != NEG_LARGE, row_max, 0.0)
    tl.store(Pi_new_offset_ptr + global_row, row_max_safe.to(ACC_DTYPE))
    out_global_base = global_row * stride
    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        mask = s_offs < S
        species_start = tl.load(species_subtree_start_ptr + s_offs, mask=mask, other=-1)
        descendant = (species_start >= leaf_start) & (species_start < leaf_end)
        leaf_hit = mask & (s_offs == leaf_species)
        const_offsets = const_base + s_offs
        max_transfer = tl.load(max_transfer_ptr + const_offsets, mask=mask, other=0.0)
        duplication_loss_const = tl.load(duplication_loss_const_ptr + const_offsets, mask=mask, other=NEG_LARGE)
        extinction_complement_log_probability = tl.load(
            Ebar_ptr + const_offsets, mask=mask, other=NEG_LARGE
        )
        extinction_log_probability = tl.load(
            E_ptr + const_offsets, mask=mask, other=NEG_LARGE
        )
        speciation_child1_const = tl.load(speciation_child1_const_ptr + const_offsets, mask=mask, other=NEG_LARGE)
        speciation_child2_const = tl.load(speciation_child2_const_ptr + const_offsets, mask=mask, other=NEG_LARGE)
        reconciliation_log_likelihood = tl.where(leaf_hit, leaf_observation_log_probability, NEG_LARGE)
        transfer_complement_log_likelihood = tl.where(
            ~descendant,
            max_transfer + leaf_receiver_log_probability + leaf_observation_log_probability,
            NEG_LARGE,
        )
        c1 = tl.load(species_child1_ptr + s_offs, mask=mask, other=S)
        c2 = tl.load(species_child2_ptr + s_offs, mask=mask, other=S)
        reconciliation_child1_log_likelihood = tl.where(mask & (c1 == leaf_species), leaf_observation_log_probability, NEG_LARGE)
        reconciliation_child2_log_likelihood = tl.where(mask & (c2 == leaf_species), leaf_observation_log_probability, NEG_LARGE)
        duplication_loss_log_term = duplication_loss_const + reconciliation_log_likelihood
        transfer_loss_log_term = (
            reconciliation_log_likelihood + extinction_complement_log_probability
        )
        transfer_log_term = (
            transfer_complement_log_likelihood + extinction_log_probability
        )
        speciation_child1_log_term = speciation_child1_const + reconciliation_child1_log_likelihood
        speciation_child2_log_term = speciation_child2_const + reconciliation_child2_log_likelihood
        leaf_logp = tl.load(leaf_logp_ptr + family * S + s_offs, mask=mask, other=NEG_LARGE)
        leaf_observation_log_term = tl.where(leaf_hit, leaf_logp, NEG_LARGE)
        logsumexp_max = tl.maximum(duplication_loss_log_term, transfer_loss_log_term)
        logsumexp_max = tl.maximum(logsumexp_max, transfer_log_term)
        logsumexp_max = tl.maximum(logsumexp_max, speciation_child1_log_term)
        logsumexp_max = tl.maximum(logsumexp_max, speciation_child2_log_term)
        logsumexp_max = tl.maximum(logsumexp_max, leaf_observation_log_term)
        logsumexp_max_safe = tl.where(logsumexp_max != NEG_LARGE, logsumexp_max, tl.zeros_like(logsumexp_max))
        local_event_scaled_mass = (
            tl.exp2(duplication_loss_log_term - logsumexp_max_safe)
            + tl.exp2(transfer_loss_log_term - logsumexp_max_safe)
            + tl.exp2(transfer_log_term - logsumexp_max_safe)
            + tl.exp2(speciation_child1_log_term - logsumexp_max_safe)
            + tl.exp2(speciation_child2_log_term - logsumexp_max_safe)
            + tl.exp2(leaf_observation_log_term - logsumexp_max_safe)
        )
        reconciliation_log_likelihood = (
            tl.log2(local_event_scaled_mass) + logsumexp_max - row_max_safe
        )
        tl.store(
            Pi_new_ptr + out_global_base + s_offs,
            reconciliation_log_likelihood,
            mask=mask,
        )


# ``ws``/``pi_ws`` are the wave's start rows and change every launch. Triton keys its
# compile cache on each integer's "== 1" / divisible-by-16 state, so leaving them
# specialized recompiles this kernel for no gain (see README.md).
@triton.jit(do_not_specialize=["ws", "pi_ws"])
def _update_reconciliation_likelihood_kernel(
    Pi_ptr,
    Pi_offset_ptr,
    ws,
    pi_ws,
    max_transfer_ptr,
    duplication_loss_const_ptr,
    Ebar_ptr,
    E_ptr,
    speciation_child1_const_ptr,
    speciation_child2_const_ptr,
    receiver_log_probs_ptr,
    species_child1_ptr,
    species_child2_ptr,
    species_parent_ptr,
    leaf_species_ptr,
    leaf_logp_ptr,
    leaf_fm_log_ptr,
    family_idx_ptr,
    gene_split_log_likelihood_ptr,
    gene_split_offset_ptr,
    gene_split_center_offset_ptr,
    has_splits: tl.constexpr,
    INPUT_IS_GENE_SPLIT: tl.constexpr,
    Pi_new_ptr,
    Pi_new_offset_ptr,
    Pibar_out_ptr,
    Pibar_offset_ptr,
    pibar_row_max_ptr,
    pi_residual_out_ptr,
    row_mask_ptr,
    S: tl.constexpr,
    stride: tl.constexpr,
    CONST_ROW_STRIDE: tl.constexpr,
    BLOCK_S: tl.constexpr,
    MAX_ANCESTOR_DEPTH: tl.constexpr,
    USE_LEAF_INDEX: tl.constexpr,
    USE_FRACTION_MISSING: tl.constexpr,
    STORE_FINAL_PIBAR: tl.constexpr,
    COMPUTE_DIFF: tl.constexpr,
    USE_RECEIVER_WEIGHTS: tl.constexpr,
    USE_ROW_MASK: tl.constexpr,
    DTYPE: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
):
    NEG_LARGE: tl.constexpr = -float("inf")

    # int64: w ranges over the whole batch's clade rows, so the *stride
    # multiplies below can overflow int32 once total_clades * S exceeds 2^31.
    w = tl.program_id(0).to(tl.int64)
    pi_row = pi_ws + w
    global_row = ws + w
    if USE_ROW_MASK:
        # Selective sweep: the exact solve handed back only the rows it could not carry in scaled
        # linear space (see ``_exact_tree_pi_self_loop_kernel``'s range check), and every other
        # row already holds its published answer. Returning here leaves those rows untouched, so
        # a masked sweep is bit-for-bit a no-op on them.
        if tl.load(row_mask_ptr + global_row) == 0:
            return
    pi_base = pi_row * stride
    global_base = global_row * stride
    gene_split_base = w * stride
    family_const = tl.load(family_idx_ptr + global_row)
    const_base = family_const * CONST_ROW_STRIDE
    pi_offset = tl.load(Pi_offset_ptr + pi_row)

    row_max, total_receiver_mass, reconciliation_row_max = _compute_total_receiver_mass(
        Pi_ptr,
        receiver_log_probs_ptr,
        pi_base,
        S,
        BLOCK_S,
        USE_RECEIVER_WEIGHTS,
        INPUT_IS_GENE_SPLIT and USE_RECEIVER_WEIGHTS,
        DTYPE,
    )
    # ``row_max`` is already required for Pibar. Absorb it lazily into the
    # accumulator-dtype row frame so the recurrence consumes near-zero residuals without an
    # exact recenter/store pass over the input row.
    if INPUT_IS_GENE_SPLIT:
        if USE_RECEIVER_WEIGHTS:
            shift_source = reconciliation_row_max
        else:
            # Without receiver weights ``row_max`` is already the raw maximum;
            # Compile out the second tile reduction entirely.
            shift_source = row_max
        reconciliation_residual_shift = tl.max(
            tl.where(
                shift_source != NEG_LARGE,
                shift_source,
                tl.zeros_like(shift_source),
            ),
            axis=0,
        )
    else:
        # Leaf initialization and the first virtually gauged DTS iteration already
        # put ordinary Pi iterates in their local frame. Avoid repeating four
        # vector shifts on every later fixed-point iteration.
        reconciliation_residual_shift = tl.zeros((), dtype=DTYPE)
    effective_pi_offset = pi_offset + reconciliation_residual_shift.to(ACC_DTYPE)

    output_frame_offset = effective_pi_offset
    if has_splits:
        gene_split_row_offset = tl.load(gene_split_offset_ptr + w)
        if INPUT_IS_GENE_SPLIT:
            gene_split_center_offset = gene_split_row_offset + reconciliation_residual_shift.to(ACC_DTYPE)
        else:
            gene_split_center_offset = tl.load(gene_split_center_offset_ptr + w)
        output_frame_offset = tl.maximum(output_frame_offset, gene_split_center_offset)
    else:
        gene_split_row_offset = output_frame_offset
    if USE_LEAF_INDEX:
        leaf_species = tl.load(leaf_species_ptr + global_row)
        leaf_observation_log_probability = tl.load(
            leaf_logp_ptr + family_const * S + leaf_species
        )
        # The leaf source represents ``leaf_observation_log_probability``, not a zero-frame
        # value. Using 0 here forced every negative HOGENOM row back into the
        # absolute frame after the exactly gauged leaf initializer.
        output_frame_offset = tl.maximum(output_frame_offset, leaf_observation_log_probability.to(ACC_DTYPE))
    reconciliation_frame_shift = (effective_pi_offset - output_frame_offset).to(DTYPE)
    # DTS storage remains in its original gauge. The virtual row offset
    # participates only in base selection; one accumulator subtraction folds its
    # shift into the same correction the recurrence already applies.
    gene_split_frame_shift = (gene_split_row_offset - output_frame_offset).to(DTYPE)
    leaf_frame_shift = (0.0 - output_frame_offset).to(DTYPE)
    if STORE_FINAL_PIBAR:
        pi_has_finite = tl.full((), value=0, dtype=tl.int32)

    if COMPUTE_DIFF:
        row_max_diff = tl.zeros([1], dtype=tl.float32)

    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        mask = s_offs < S
        reconciliation_log_likelihood = tl.load(Pi_ptr + pi_base + s_offs, mask=mask, other=NEG_LARGE)
        reconciliation_log_likelihood = reconciliation_log_likelihood - reconciliation_residual_shift
        const_offsets = const_base + s_offs
        max_transfer = tl.load(max_transfer_ptr + const_offsets, mask=mask, other=0.0)
        transfer_complement_log_likelihood = _compute_transfer_complement(
            Pi_ptr,
            receiver_log_probs_ptr,
            pi_base,
            s_offs,
            mask,
            row_max,
            total_receiver_mass,
            max_transfer,
            species_parent_ptr,
            S,
            BLOCK_S,
            MAX_ANCESTOR_DEPTH,
            USE_RECEIVER_WEIGHTS,
            DTYPE,
        )
        transfer_complement_log_likelihood = transfer_complement_log_likelihood - reconciliation_residual_shift
        duplication_loss_const = tl.load(duplication_loss_const_ptr + const_offsets, mask=mask, other=NEG_LARGE)
        extinction_complement_log_probability = tl.load(
            Ebar_ptr + const_offsets, mask=mask, other=NEG_LARGE
        )
        extinction_log_probability = tl.load(
            E_ptr + const_offsets, mask=mask, other=NEG_LARGE
        )
        speciation_child1_const = tl.load(speciation_child1_const_ptr + const_offsets, mask=mask, other=NEG_LARGE)
        speciation_child2_const = tl.load(speciation_child2_const_ptr + const_offsets, mask=mask, other=NEG_LARGE)

        c1 = tl.load(species_child1_ptr + s_offs, mask=mask, other=0)
        c2 = tl.load(species_child2_ptr + s_offs, mask=mask, other=0)
        c1_valid = c1 < S
        c2_valid = c2 < S
        reconciliation_child1_log_likelihood = tl.load(Pi_ptr + pi_base + c1, mask=mask & c1_valid, other=NEG_LARGE)
        reconciliation_child2_log_likelihood = tl.load(Pi_ptr + pi_base + c2, mask=mask & c2_valid, other=NEG_LARGE)
        reconciliation_child1_log_likelihood = reconciliation_child1_log_likelihood - reconciliation_residual_shift
        reconciliation_child2_log_likelihood = reconciliation_child2_log_likelihood - reconciliation_residual_shift

        duplication_loss_log_term = duplication_loss_const + reconciliation_log_likelihood + reconciliation_frame_shift
        transfer_loss_log_term = (
            reconciliation_log_likelihood
            + extinction_complement_log_probability
            + reconciliation_frame_shift
        )
        transfer_log_term = (
            transfer_complement_log_likelihood
            + extinction_log_probability
            + reconciliation_frame_shift
        )
        speciation_child1_log_term = speciation_child1_const + reconciliation_child1_log_likelihood + reconciliation_frame_shift
        speciation_child2_log_term = speciation_child2_const + reconciliation_child2_log_likelihood + reconciliation_frame_shift
        if USE_LEAF_INDEX:
            leaf_hit = mask & (leaf_species == s_offs)
            mapped_term = leaf_observation_log_probability + leaf_frame_shift
            if USE_FRACTION_MISSING:
                # Off-hit leaf-species columns carry the "present-but-unobserved"
                # baseline log_pS[s] + log2(fm_s); non-leaf/observed columns stay -inf.
                leaf_logp_col = tl.load(leaf_logp_ptr + family_const * S + s_offs, mask=mask, other=NEG_LARGE)
                fm_col = tl.load(leaf_fm_log_ptr + s_offs, mask=mask, other=NEG_LARGE)
                baseline = leaf_logp_col + fm_col + leaf_frame_shift  # -inf where fm_col is -inf
                leaf_observation_log_term = tl.where(leaf_hit, mapped_term, baseline)
            else:
                leaf_observation_log_term = tl.where(leaf_hit, mapped_term, NEG_LARGE)
        else:
            leaf_observation_log_term = tl.full(
                [BLOCK_S], value=NEG_LARGE, dtype=DTYPE
            )

        logsumexp_max = tl.maximum(duplication_loss_log_term, transfer_loss_log_term)
        logsumexp_max = tl.maximum(logsumexp_max, transfer_log_term)
        logsumexp_max = tl.maximum(logsumexp_max, speciation_child1_log_term)
        logsumexp_max = tl.maximum(logsumexp_max, speciation_child2_log_term)
        logsumexp_max = tl.maximum(logsumexp_max, leaf_observation_log_term)
        if has_splits:
            gene_split_log_likelihood = tl.load(gene_split_log_likelihood_ptr + gene_split_base + s_offs, mask=mask, other=NEG_LARGE)
            gene_split_log_likelihood = gene_split_log_likelihood + gene_split_frame_shift
            logsumexp_max = tl.maximum(logsumexp_max, gene_split_log_likelihood)
        logsumexp_max_safe = tl.where(logsumexp_max != NEG_LARGE, logsumexp_max, tl.zeros_like(logsumexp_max))
        local_event_scaled_mass = (
            tl.exp2(duplication_loss_log_term - logsumexp_max_safe)
            + tl.exp2(transfer_loss_log_term - logsumexp_max_safe)
            + tl.exp2(transfer_log_term - logsumexp_max_safe)
        )
        local_event_scaled_mass += (
            tl.exp2(speciation_child1_log_term - logsumexp_max_safe)
            + tl.exp2(speciation_child2_log_term - logsumexp_max_safe)
            + tl.exp2(leaf_observation_log_term - logsumexp_max_safe)
        )
        if has_splits:
            local_event_scaled_mass += tl.exp2(gene_split_log_likelihood - logsumexp_max_safe)
        updated_reconciliation_log_likelihood = (
            tl.log2(local_event_scaled_mass) + logsumexp_max
        )
        tl.store(
            Pi_new_ptr + global_base + s_offs,
            updated_reconciliation_log_likelihood,
            mask=mask,
        )
        if STORE_FINAL_PIBAR:
            pi_has_finite = tl.maximum(
                pi_has_finite,
                tl.max(
                    tl.where(
                        mask
                        & (updated_reconciliation_log_likelihood != NEG_LARGE),
                        1,
                        0,
                    ),
                    axis=0,
                ),
            )

        if COMPUTE_DIFF:
            # Compare represented absolute values in the accumulator dtype
            # without materializing either large absolute row.
            finite = (
                mask
                & (updated_reconciliation_log_likelihood != NEG_LARGE)
                & (reconciliation_log_likelihood != NEG_LARGE)
            )
            diff = tl.where(
                finite,
                tl.abs(
                    updated_reconciliation_log_likelihood.to(ACC_DTYPE)
                    - reconciliation_log_likelihood.to(ACC_DTYPE)
                    + output_frame_offset
                    - effective_pi_offset
                ),
                tl.zeros([BLOCK_S], dtype=ACC_DTYPE),
            )
            row_max_diff = tl.maximum(row_max_diff, tl.max(diff, axis=0).to(tl.float32))

    # Only the final iterate is published as row-gauged state; earlier iterates
    # are internal gauge-equivalent scratch. Canonicalize the published row in
    # its existing traversal without charging every fixed-point iteration.
    if STORE_FINAL_PIBAR:
        pi_new_offset = tl.where(pi_has_finite != 0, output_frame_offset, 0.0)
    else:
        pi_new_offset = output_frame_offset
    tl.store(Pi_new_offset_ptr + global_row, pi_new_offset)

    if COMPUTE_DIFF:
        tl.store(pi_residual_out_ptr + global_row, tl.max(row_max_diff, axis=0))

    if has_splits and INPUT_IS_GENE_SPLIT:
        tl.store(gene_split_center_offset_ptr + w, gene_split_center_offset)

    if STORE_FINAL_PIBAR:
        final_row_max, final_row_sum, _ = _compute_total_receiver_mass(
            Pi_new_ptr,
            receiver_log_probs_ptr,
            global_base,
            S,
            BLOCK_S,
            USE_RECEIVER_WEIGHTS,
            False,
            DTYPE,
        )
        tl.store(pibar_row_max_ptr + global_row, tl.max(final_row_max, axis=0))
        # ``Pi_new`` is stored in ``output_frame_offset``'s frame, so the Pibar values
        # produced from it are already in that same frame. Keep that cheap
        # gauge instead of traversing the species row once to find an exact
        # Pibar maximum and again to store the recentered values.
        pibar_has_finite = tl.full((), value=0, dtype=tl.int32)
        for s_start in range(0, S, BLOCK_S):
            s_offs = s_start + tl.arange(0, BLOCK_S)
            mask = s_offs < S
            const_offsets = const_base + s_offs
            max_transfer = tl.load(max_transfer_ptr + const_offsets, mask=mask, other=0.0)
            transfer_complement_log_likelihood = _compute_transfer_complement(
                Pi_new_ptr,
                receiver_log_probs_ptr,
                global_base,
                s_offs,
                mask,
                final_row_max,
                final_row_sum,
                max_transfer,
                species_parent_ptr,
                S,
                BLOCK_S,
                MAX_ANCESTOR_DEPTH,
                USE_RECEIVER_WEIGHTS,
                DTYPE,
            )
            pibar_has_finite = tl.maximum(
                pibar_has_finite,
                tl.max(tl.where(mask & (transfer_complement_log_likelihood != NEG_LARGE), 1, 0), axis=0),
            )
            tl.store(Pibar_out_ptr + global_base + s_offs, transfer_complement_log_likelihood, mask=mask)
        # An all-impossible Pibar row must remain the canonical ``(-inf, 0)``
        # pair even when its Pi row used a nonzero heuristic gauge.
        pibar_offset = tl.where(pibar_has_finite != 0, output_frame_offset, 0.0)
        tl.store(Pibar_offset_ptr + global_row, pibar_offset)


@triton.jit
def _lookup_valid_receiver_mass(
    scratch_ptr,
    not_open_base,
    closed_base,
    not_open_index_ptr,
    closed_index_ptr,
    s_offs,
    mask,
    S: tl.constexpr,
    BLOCK_S: tl.constexpr,
    DTYPE: tl.constexpr,
):
    """Read each donor lane's valid receiver mass out of the two running sums.

    Both index tables come from
    :func:`gpurec.core.valid_receivers.valid_receiver_index_tables`: one says where this
    donor's "not yet opened" prefix ends, the other where its "already closed" prefix ends. Adding
    the two is the whole mass, with no subtraction anywhere.
    """
    not_open_index = tl.load(not_open_index_ptr + s_offs, mask=mask, other=0).to(tl.int64)
    closed_index = tl.load(closed_index_ptr + s_offs, mask=mask, other=0).to(tl.int64)
    not_yet_open = tl.load(scratch_ptr + not_open_base + not_open_index, mask=mask, other=0.0)
    already_closed = tl.load(scratch_ptr + closed_base + closed_index, mask=mask, other=0.0)
    return tl.where(mask, not_yet_open + already_closed, tl.zeros([BLOCK_S], dtype=DTYPE))


@triton.jit
def _write_valid_receiver_prefix_sums(
    scratch_ptr,
    value_base,
    not_open_base,
    closed_base,
    receiver_lin_ptr,
    not_open_source_ptr,
    closed_source_ptr,
    S: tl.constexpr,
    BLOCK_S: tl.constexpr,
    USE_RECEIVER_WEIGHTS: tl.constexpr,
    DTYPE: tl.constexpr,
):
    """Write the two running sums a donor's valid receiver mass is built from, without any subtraction.

    A donor ``s`` may transfer to every species that is neither ``s`` itself nor one of its
    ancestors. :func:`_compute_transfer_complement` gets that mass as
    ``total_receiver_mass - excluded_ancestor_mass``, two nearly equal numbers whose difference is
    the answer: at a high transfer rate the row's whole mass sits on the donor's own lineage, the
    two agree to more than fp32's 24 bits, and the difference is noise -- it even changes sign, so
    lanes flip between a finite transfer complement and ``-inf``.

    With the depth-first interval numbering (``start`` is a permutation of ``0..S-1`` and each
    subtree owns ``[start, end)``), ``a`` is an ancestor-or-self of ``s`` exactly when
    ``start[a] <= start[s] < end[a]``. So the ALLOWED recipients split into two disjoint groups,

        not yet opened   ``start[a] > start[s]``
        already closed   ``end[a] <= start[s]``

    and their masses are two running sums of non-negative terms, which cannot cancel. This writes
    both, each as a plain forward scan over a species order the host prepared
    (:func:`gpurec.core.valid_receivers.valid_receiver_index_tables`); the sources are shifted
    by one position, so an inclusive scan already gives the exclusive prefix the lookup wants.
    """
    # Both scans gather lanes of the value row at arbitrary positions, i.e. lanes other warps of
    # this block wrote (the entry conversion, or the previous iteration's store). Only a barrier
    # makes those writes visible here; the row-wide reductions that used to sit in front of these
    # gathers are gone.
    tl.debug_barrier()
    for pass_id in tl.static_range(2):
        if pass_id == 0:
            source_ptr = not_open_source_ptr
            output_base = not_open_base
        else:
            source_ptr = closed_source_ptr
            output_base = closed_base
        running_total = tl.full((), value=0.0, dtype=DTYPE)
        for s_start in range(0, S, BLOCK_S):
            positions = s_start + tl.arange(0, BLOCK_S)
            mask = positions < S
            species = tl.load(source_ptr + positions, mask=mask, other=S).to(tl.int64)
            # ``S`` is the one-position shift's sentinel: it contributes nothing.
            contributes = mask & (species < S)
            value = tl.load(scratch_ptr + value_base + species, mask=contributes, other=0.0)
            if USE_RECEIVER_WEIGHTS:
                value = value * tl.load(
                    receiver_lin_ptr + species, mask=contributes, other=0.0
                )
            value = tl.where(contributes, value, tl.zeros([BLOCK_S], dtype=DTYPE))
            tl.store(
                scratch_ptr + output_base + positions,
                running_total + tl.cumsum(value, axis=0),
                mask=mask,
            )
            running_total += tl.sum(value, axis=0)
    # The lookups below read lanes other warps in this block just wrote.
    tl.debug_barrier()


# ``ws`` is the wave's start row and ``slot_span`` the scratch stride; both change per
# wave or per batch. Keeping them out of the specialization key avoids one JIT compile
# per new "== 1" / divisible-by-16 state (see README.md).
@triton.jit(do_not_specialize=["ws", "slot_span", "n_levels"])
def _exact_tree_pi_self_loop_kernel(
    Pi_in_ptr,
    Pi_in_offset_ptr,
    scratch_ptr,
    Pi_out_ptr,
    Pi_out_offset_ptr,
    Pibar_out_ptr,
    Pibar_offset_ptr,
    pibar_row_max_ptr,
    pi_residual_out_ptr,
    guard_trips_ptr,
    self_diagonal_lin_ptr,
    transfer_coefficient_lin_ptr,
    speciation_child1_lin_ptr,
    speciation_child2_lin_ptr,
    max_transfer_lin_ptr,
    receiver_lin_ptr,
    species_child1_ptr,
    species_child2_ptr,
    level_ptr_ptr,
    level_parents_ptr,
    level_child1_ptr,
    level_child2_ptr,
    leaf_species_ptr,
    leaf_logp_ptr,
    leaf_fm_log_ptr,
    family_idx_ptr,
    gene_split_log_likelihood_ptr,
    gene_split_offset_ptr,
    gene_split_center_offset_ptr,
    wide_row_ptr,
    wide_row_count_ptr,
    ws,
    slot_span,
    n_levels,
    range_limit,
    conditioning_floor,
    wave_index,
    S: tl.constexpr,
    stride: tl.constexpr,
    CONST_ROW_STRIDE: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_NODES: tl.constexpr,
    HAS_SPLITS: tl.constexpr,
    USE_LEAF_INDEX: tl.constexpr,
    USE_FRACTION_MISSING: tl.constexpr,
    COMPUTE_DIFF: tl.constexpr,
    WRITE_GUARD_TRIPS: tl.constexpr,
    USE_RECEIVER_WEIGHTS: tl.constexpr,
    DTYPE: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
):
    """Solve one clade row's Pi self-loop EXACTLY, by elimination on the species tree.

    Published state is the usual log2 residual + row offset (docs/centered_state_contract.md).
    Instead of iterating the fixed point this kernel SOLVES it in one launch, so it returns the
    CONVERGED row -- the fixed point the log-space path
    (:func:`_update_reconciliation_likelihood_kernel`) approaches as ``pi_iters`` grows -- and
    ``pi_iters`` therefore plays no part in it.

    Why an exact solve is possible. In scaled linear space ``p[s] = 2**(Pi_abs[c, s] - scale)``
    the self-loop update is

        A[s]     = recv[s] * p[s]                    (recv = 1 when receiver weights are off)
        T        = sum over all species of A
        X[s]     = sum of A over s itself and its species-tree ancestors
        p_new[s] = src[s] + dl[s] p[s] + ebar[s] p[s] + e[s] mt[s] max(T - X[s], 0)
                   + sl1[s] p[child1(s)] + sl2[s] p[child2(s)]

    with ``src = leaf observation + gene-split (DTS) source``. Every ``p`` is a likelihood, so
    ``p >= 0`` and ``X[s] <= T``: the ``max`` never clips and the fixed point is a LINEAR system.

    Each per-species multiplier is the linear-space copy of a log2 event constant, ``2**(its log2
    value)``, built once per forward solve (``gpurec.core.inference.forward``'s
    ``_linear_event_multipliers`` and ``_exact_tree_coefficients``):

        sl1[s]  = 2**(log_pS[s] + E_child2[s])   speciate, follow child 1, child 2 lost
        sl2[s]  = 2**(log_pS[s] + E_child1[s])   speciate, follow child 2, child 1 lost
        mt[s]   = 2**max_transfer[s]             per-donor transfer normalizer
        recv[s] = 2**receiver_log_prob[s]        receiver weight (1 when weights are off)
        dl[s]   = 2**(1 + log_pD[s] + E[s])      duplicate, then lose one copy
        ebar[s] = 2**Ebar[s]                     stay in place, transferred copy lost
        e[s]    = 2**E[s]                        transfer out, donor copy lost

    The first four arrive as their own pointers; ``dl``, ``ebar`` and ``e`` reach the kernel only
    folded into ``self_diagonal_lin`` and ``transfer_coefficient_lin`` below.

    The whole solve is written in terms of

        u[s] = T - X[parent(s)],

    the transfer mass still available to species ``s`` -- the row total minus what its ANCESTORS
    have taken, ``s`` itself not yet subtracted. It is the natural variable because it is what the
    equation multiplies, it is non-negative, and it passes down the tree by a single subtraction:
    a child sees ``u[c] = u[s] - recv[s] p[s]``, and ``u[root] = T``. With ``X[s] = (T - u[s]) +
    recv[s] p[s]`` the per-species equation reads

        diag[s] p[s] - q[s] u[s] - sl1[s] p[c1] - sl2[s] p[c2] = src[s]
        q[s]    = e[s] * mt[s]                                  (``transfer_coefficient_lin``)
        diag[s] = 1 - dl[s] - ebar[s] + q[s] recv[s]            (``self_diagonal_lin``)

    so species ``s`` couples to everything above it through the single number ``u[s]`` and to
    everything below it through ``p[c1]``, ``p[c2]``. Four O(S) walks over the species tree solve
    it: two to eliminate, and two more that rebuild ``u`` out of sums instead of differences.

    1. Bottom-up (leaves first, then the ``compact_level_*`` tables level by level). Each node's
       solution becomes an affine function of what comes from above,

           p[s] = alpha[s] + gamma[s] u[s].

       A species-tree leaf has no children: dividing its equation by ``diag[s]`` gives
       ``alpha = src/diag``, ``gamma = q/diag``. At an internal node the children already have
       their own affine form and see ``u[c] = u[s] - recv[s] p[s]``; substituting and collecting
       the ``p[s]`` terms gives, with

           Aa    = sl1 alpha[c1] + sl2 alpha[c2]
           G     = sl1 gamma[c1] + sl2 gamma[c2]
           pivot = diag[s] + G recv[s]

           alpha[s] = (src[s] + Aa) / pivot
           gamma[s] = (q[s] + G)   / pivot.

       Every quantity here is non-negative and nothing is subtracted: the pivot is the diagonal
       plus the children's feedback, never minus it, so it is bounded BELOW by ``diag[s]``.

    2. Top-down (root first, same levels reversed). ``u`` is not known until ``T`` is, so this
       pass carries ``u[s] = U0[s] + U1[s] T`` and with it ``p[s] = P0[s] + P1[s] T``, where
       ``P0 = alpha + gamma U0`` and ``P1 = gamma U1``. The root has ``u = T``, so
       ``U0[root] = 0``, ``U1[root] = 1``, and a child of ``s`` gets ``U0[c] = U0[s] - recv[s]
       P0[s]``, ``U1[c] = U1[s] - recv[s] P1[s]``. Then ``T = sum_s recv[s] p[s]`` is one scalar
       equation ``T = T0 + T1 T`` with ``T0 = sum recv (alpha + gamma U0)`` and
       ``T1 = sum recv gamma U1``, so ``T = T0 / (1 - T1)``; ``T1`` is the total gain of the
       transfer loop, well below 1.

    3. Why two more walks. ``u = U0 + U1 T`` is mathematically ``T - (what the ancestors took)``,
       and for a deep species whose ancestors hold nearly all of the row's transfer mass that is a
       difference of two nearly equal numbers. In float32 it loses every significant digit, and
       since ``p = alpha + gamma u`` those lanes then come out orders of magnitude too small or
       even negative. (Measured on the fitted-rate benchmark: in float64 the two-walk solve
       matches a converged log-space forward to 2.5e-7 log2 -- the elimination is right -- while
       in float32 ~1e-5 of the entries were off by more than 5 log2.) So ``u`` is rebuilt from
       ADDITIONS only, using the first-pass ``p`` (call it ``p1``) purely to weigh masses:

           M[s] = recv[s] p1[s] + M[c1] + M[c2]          bottom-up: mass of s's whole subtree
           R[root] = 0,  R[c1] = R[s] + M[c2],           top-down: mass hanging off the
                         R[c2] = R[s] + M[c1]            ancestor chain, s's subtree excluded
           u[s] = R[s] + M[s]
           T - X[s] = R[s] + M[c1] + M[c2]

       Every step is a sum of non-negative terms, so ``u > 0`` always and ``p = alpha + gamma u``
       is non-negative by construction -- no lane can come out negative and be published as
       ``-inf``, which is what the two-walk form did to 16010 of 6.2e8 entries on the benchmark.
       ``M`` and ``R`` are dominated by the row's largest lanes, which ``p1`` already gets right,
       so one correction round is all this buys and all it needs.

    What remains is not this solve's to fix. Scaled linear space carries a row over about 24 log2
    units of range in float32: a lane whose share of the row's transfer mass is below float32's
    unit roundoff gets that roundoff instead of its true value, and comes out too LARGE. That
    floor belongs to scaled linear space itself rather than to the elimination -- the fused
    linear-space iteration this kernel replaced landed on the same wrong value, 12.5 log2 above
    the log path, for the same entry on the fitted-rate benchmark. Only the log-space path is
    free of it.

    Pivots and ``1 - T1`` are the only divisions. Both are positive for any parameter set the
    fixed point converges for (``dl + ebar < 1`` per species, loop gain < 1), and NEITHER is
    substituted or clipped: a non-positive one divides through to an infinity or a NaN that shows
    up immediately in the likelihood. ``guard_trips_ptr`` counts them per row so a run can say so.

    Scratch: ``[4, rows, S]`` in the model dtype, four per-row species arrays, two of them reused
    as each role finishes:

        slot 0  alpha[s]
        slot 1  gamma[s]
        slot 2  src[s]  -> U0[s] -> R[s]
        slot 3  U1[s]   -> M[s]

    The species-tree walks read values written by other warps of the same program, so every level
    ends in a ``tl.debug_barrier()``, exactly as ``_reconciliation_self_loop_transpose_term`` does
    for the same level tables in the backward pass.
    """
    NEG_LARGE: tl.constexpr = -float("inf")

    # int64: w ranges over the whole batch's clade rows, so the *stride multiplies
    # below can overflow int32 once total_clades * S exceeds 2^31.
    w = tl.program_id(0).to(tl.int64)
    global_row = ws + w
    global_base = global_row * stride
    row_base = w * stride
    family_const = tl.load(family_idx_ptr + global_row)
    const_base = family_const * CONST_ROW_STRIDE
    span = slot_span.to(tl.int64)
    alpha_base = row_base
    gamma_base = span + row_base
    # slot 2: the source term, then U0, then the off-chain mass R.
    mass_off_chain_base = 2 * span + row_base
    # slot 3: U1, then the subtree mass M.
    mass_subtree_base = 3 * span + row_base

    # ---- entry pass 1: exact absolute row maximum, which becomes the linear gauge, and the
    # smallest finite lane, which says whether this row FITS in that single gauge at all.
    POS_LARGE: tl.constexpr = float("inf")
    entry_residual_max = tl.full((), value=NEG_LARGE, dtype=DTYPE)
    entry_residual_min = tl.full((), value=POS_LARGE, dtype=DTYPE)
    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        mask = s_offs < S
        value = tl.load(Pi_in_ptr + global_base + s_offs, mask=mask, other=NEG_LARGE)
        entry_residual_max = tl.maximum(
            entry_residual_max, tl.max(tl.where(mask, value, NEG_LARGE), axis=0)
        )
        entry_residual_min = tl.minimum(
            entry_residual_min,
            tl.min(tl.where(mask & (value != NEG_LARGE), value, POS_LARGE), axis=0),
        )
    pi_in_offset = tl.load(Pi_in_offset_ptr + global_row)
    scale = tl.where(
        entry_residual_max != NEG_LARGE,
        entry_residual_max.to(ACC_DTYPE) + pi_in_offset,
        tl.full((), value=NEG_LARGE, dtype=ACC_DTYPE),
    )
    if HAS_SPLITS:
        # Two different gauges, exactly as in the log-space kernel: the DTS row is STORED
        # against ``gene_split_offset`` (the sum of its two child gauges), while
        # ``gene_split_center_offset`` is that row's absolute maximum and only picks the frame.
        gene_split_row_offset = tl.load(gene_split_offset_ptr + w)
        gene_split_center_offset = tl.load(gene_split_center_offset_ptr + w)
        scale = tl.maximum(scale, gene_split_center_offset)
    if USE_LEAF_INDEX:
        leaf_species = tl.load(leaf_species_ptr + global_row)
        leaf_observation_log_probability = tl.load(
            leaf_logp_ptr + family_const * S + leaf_species
        )
        scale = tl.maximum(scale, leaf_observation_log_probability.to(ACC_DTYPE))
    # A wholly impossible row keeps the canonical zero gauge.
    scale = tl.where(scale != NEG_LARGE, scale, tl.zeros_like(scale))

    # ---- entry pass 2: the iteration-invariant source term src[s], in this row's frame.
    # ``scale`` is at least the maximum of every log2 term entering it, so each exponent is <= 0.
    # The same pass records how far the SOURCE reaches below the gauge, because a row's range is
    # set by everything that enters it, not only by the iterate it was handed.
    source_span_min = tl.full((), value=POS_LARGE, dtype=ACC_DTYPE)
    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        mask = s_offs < S
        source = tl.zeros([BLOCK_S], dtype=DTYPE)
        if USE_LEAF_INDEX:
            leaf_hit = mask & (leaf_species == s_offs)
            mapped_term = tl.exp2(
                (leaf_observation_log_probability.to(ACC_DTYPE) - scale).to(DTYPE)
            )
            if USE_FRACTION_MISSING:
                # Off-hit leaf-species columns carry the "present-but-unobserved"
                # baseline log_pS[s] + log2(fm_s); non-leaf/observed columns stay zero.
                leaf_logp_col = tl.load(
                    leaf_logp_ptr + family_const * S + s_offs, mask=mask, other=NEG_LARGE
                )
                fm_col = tl.load(leaf_fm_log_ptr + s_offs, mask=mask, other=NEG_LARGE)
                baseline = tl.exp2(leaf_logp_col + fm_col - scale.to(DTYPE))
                source += tl.where(leaf_hit, mapped_term, baseline)
            else:
                source += tl.where(leaf_hit, mapped_term, tl.zeros([BLOCK_S], dtype=DTYPE))
        if HAS_SPLITS:
            gene_split_log_likelihood = tl.load(
                gene_split_log_likelihood_ptr + row_base + s_offs, mask=mask, other=NEG_LARGE
            )
            source += tl.exp2(
                gene_split_log_likelihood + (gene_split_row_offset - scale).to(DTYPE)
            )
            gene_split_min = tl.min(
                tl.where(
                    mask & (gene_split_log_likelihood != NEG_LARGE),
                    gene_split_log_likelihood,
                    POS_LARGE,
                ),
                axis=0,
            )
            source_span_min = tl.minimum(
                source_span_min, gene_split_min.to(ACC_DTYPE) + gene_split_row_offset - scale
            )
        tl.store(
            scratch_ptr + mass_off_chain_base + s_offs,
            tl.where(mask, source, tl.zeros([BLOCK_S], dtype=DTYPE)),
            mask=mask,
        )

    if USE_LEAF_INDEX:
        # The leaf observation sits at one lane, and with fraction-missing every other lane of
        # this row's leaf column carries the present-but-unobserved baseline; both are sources.
        source_span_min = tl.minimum(
            source_span_min, leaf_observation_log_probability.to(ACC_DTYPE) - scale
        )
        if USE_FRACTION_MISSING:
            for s_start in range(0, S, BLOCK_S):
                s_offs = s_start + tl.arange(0, BLOCK_S)
                mask = s_offs < S
                leaf_logp_col = tl.load(
                    leaf_logp_ptr + family_const * S + s_offs, mask=mask, other=NEG_LARGE
                )
                fm_col = tl.load(leaf_fm_log_ptr + s_offs, mask=mask, other=NEG_LARGE)
                baseline = leaf_logp_col + fm_col
                source_span_min = tl.minimum(
                    source_span_min,
                    tl.min(
                        tl.where(mask & (baseline != NEG_LARGE), baseline, POS_LARGE), axis=0
                    ).to(ACC_DTYPE)
                    - scale,
                )

    # ---- the range decision. Scaled linear space carries ONE exponent for the whole row, so a
    # lane this far under the gauge is an exact zero here while the log path, which keeps an
    # exponent per lane, still holds it. Hand such a row back untouched: the caller sweeps it in
    # log space instead, and the row's published state is left exactly as this kernel found it.
    row_span_min = tl.minimum(
        source_span_min,
        tl.where(
            entry_residual_min != POS_LARGE,
            entry_residual_min.to(ACC_DTYPE) + pi_in_offset - scale,
            tl.full((), value=POS_LARGE, dtype=ACC_DTYPE),
        ),
    )
    if row_span_min < -range_limit:
        tl.store(wide_row_ptr + global_row, 1)
        tl.atomic_add(wide_row_count_ptr + wave_index, 1)
        return

    guard_trips = tl.full((), value=0, dtype=tl.int32)
    smallest_pivot = tl.full((), value=POS_LARGE, dtype=DTYPE)

    # ---- walk 1a: the species-tree leaves, which have no children to fold in. They are not in
    # the ``compact_level_*`` tables (those hold internal nodes only), so they are done in one
    # contiguous sweep, masked to the lanes whose first child is the S sentinel.
    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        mask = s_offs < S
        const_offsets = const_base + s_offs
        child1 = tl.load(species_child1_ptr + s_offs, mask=mask, other=0)
        is_leaf = mask & (child1 >= S)
        diagonal = tl.load(self_diagonal_lin_ptr + const_offsets, mask=mask, other=1.0)
        transfer_coefficient = tl.load(
            transfer_coefficient_lin_ptr + const_offsets, mask=is_leaf, other=0.0
        )
        source = tl.load(scratch_ptr + mass_off_chain_base + s_offs, mask=is_leaf, other=0.0)
        tl.store(scratch_ptr + alpha_base + s_offs, source / diagonal, mask=is_leaf)
        tl.store(scratch_ptr + gamma_base + s_offs, transfer_coefficient / diagonal, mask=is_leaf)
        guard_trips += tl.sum(
            tl.where(is_leaf & (diagonal <= 0.0), 1, 0).to(tl.int32), axis=0
        )
        # Every internal node's pivot is its diagonal PLUS a non-negative children term, so the
        # smallest diagonal in the row bounds every divisor this solve will use from below.
        smallest_pivot = tl.minimum(
            smallest_pivot, tl.min(tl.where(mask, diagonal, POS_LARGE), axis=0)
        )
    tl.debug_barrier()

    # ---- walk 1b: internal nodes, shallowest-subtree level first. Every node in level ``level``
    # has both children at strictly lower levels (or at leaves), so their alpha and gamma are
    # already final when this level reads them.
    for level in range(0, n_levels):
        level_start = tl.load(level_ptr_ptr + level)
        level_end = tl.load(level_ptr_ptr + level + 1)
        node_start = level_start
        while node_start < level_end:
            node_offs = node_start + tl.arange(0, BLOCK_NODES)
            node_mask = node_offs < level_end
            parent = tl.load(level_parents_ptr + node_offs, mask=node_mask, other=0).to(tl.int64)
            child1 = tl.load(level_child1_ptr + node_offs, mask=node_mask, other=S).to(tl.int64)
            child2 = tl.load(level_child2_ptr + node_offs, mask=node_mask, other=S).to(tl.int64)
            child1_mask = node_mask & (child1 < S)
            child2_mask = node_mask & (child2 < S)
            const_offsets = const_base + parent

            speciation_child1 = tl.load(
                speciation_child1_lin_ptr + const_offsets, mask=node_mask, other=0.0
            )
            speciation_child2 = tl.load(
                speciation_child2_lin_ptr + const_offsets, mask=node_mask, other=0.0
            )
            alpha_child1 = tl.load(scratch_ptr + alpha_base + child1, mask=child1_mask, other=0.0)
            gamma_child1 = tl.load(scratch_ptr + gamma_base + child1, mask=child1_mask, other=0.0)
            alpha_child2 = tl.load(scratch_ptr + alpha_base + child2, mask=child2_mask, other=0.0)
            gamma_child2 = tl.load(scratch_ptr + gamma_base + child2, mask=child2_mask, other=0.0)
            child_constant = speciation_child1 * alpha_child1 + speciation_child2 * alpha_child2
            child_transfer_gain = (
                speciation_child1 * gamma_child1 + speciation_child2 * gamma_child2
            )

            diagonal = tl.load(self_diagonal_lin_ptr + const_offsets, mask=node_mask, other=1.0)
            transfer_coefficient = tl.load(
                transfer_coefficient_lin_ptr + const_offsets, mask=node_mask, other=0.0
            )
            if USE_RECEIVER_WEIGHTS:
                receiver_weight = tl.load(receiver_lin_ptr + parent, mask=node_mask, other=0.0)
            else:
                receiver_weight = tl.full([BLOCK_NODES], value=1.0, dtype=DTYPE)
            source = tl.load(scratch_ptr + mass_off_chain_base + parent, mask=node_mask, other=0.0)

            pivot = diagonal + child_transfer_gain * receiver_weight
            tl.store(
                scratch_ptr + alpha_base + parent,
                (source + child_constant) / pivot,
                mask=node_mask,
            )
            tl.store(
                scratch_ptr + gamma_base + parent,
                (transfer_coefficient + child_transfer_gain) / pivot,
                mask=node_mask,
            )
            guard_trips += tl.sum(
                tl.where(node_mask & (pivot <= 0.0), 1, 0).to(tl.int32), axis=0
            )
            node_start += BLOCK_NODES
        tl.debug_barrier()

    # ---- seed walk 2 with the ROOT's value of u, namely u = T (U0 = 0, U1 = 1). The seed is
    # written to every species: the root keeps it (it has no ancestors to take mass), and every
    # other node's entry is overwritten by its parent before anything reads it. This also retires
    # the source term, whose slot U0 takes over.
    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        mask = s_offs < S
        tl.store(
            scratch_ptr + mass_off_chain_base + s_offs, tl.zeros([BLOCK_S], dtype=DTYPE), mask=mask
        )
        tl.store(
            scratch_ptr + mass_subtree_base + s_offs,
            tl.full([BLOCK_S], value=1.0, dtype=DTYPE),
            mask=mask,
        )
    tl.debug_barrier()

    # ---- walk 2: the same levels, deepest-subtree level (the root) first. A node's U0 / U1 are
    # written exactly once, by its parent.
    for level_index in range(0, n_levels):
        level = n_levels - 1 - level_index
        level_start = tl.load(level_ptr_ptr + level)
        level_end = tl.load(level_ptr_ptr + level + 1)
        node_start = level_start
        while node_start < level_end:
            node_offs = node_start + tl.arange(0, BLOCK_NODES)
            node_mask = node_offs < level_end
            parent = tl.load(level_parents_ptr + node_offs, mask=node_mask, other=0).to(tl.int64)
            child1 = tl.load(level_child1_ptr + node_offs, mask=node_mask, other=S).to(tl.int64)
            child2 = tl.load(level_child2_ptr + node_offs, mask=node_mask, other=S).to(tl.int64)
            child1_mask = node_mask & (child1 < S)
            child2_mask = node_mask & (child2 < S)

            parent_alpha = tl.load(scratch_ptr + alpha_base + parent, mask=node_mask, other=0.0)
            parent_gamma = tl.load(scratch_ptr + gamma_base + parent, mask=node_mask, other=0.0)
            parent_u0 = tl.load(scratch_ptr + mass_off_chain_base + parent, mask=node_mask, other=0.0)
            parent_u1 = tl.load(scratch_ptr + mass_subtree_base + parent, mask=node_mask, other=0.0)
            if USE_RECEIVER_WEIGHTS:
                receiver_weight = tl.load(receiver_lin_ptr + parent, mask=node_mask, other=0.0)
            else:
                receiver_weight = tl.full([BLOCK_NODES], value=1.0, dtype=DTYPE)
            # Both children see the same remaining mass: the parent's, minus the parent's own.
            child_u0 = parent_u0 - receiver_weight * (parent_alpha + parent_gamma * parent_u0)
            child_u1 = parent_u1 - receiver_weight * (parent_gamma * parent_u1)

            tl.store(scratch_ptr + mass_off_chain_base + child1, child_u0, mask=child1_mask)
            tl.store(scratch_ptr + mass_subtree_base + child1, child_u1, mask=child1_mask)
            tl.store(scratch_ptr + mass_off_chain_base + child2, child_u0, mask=child2_mask)
            tl.store(scratch_ptr + mass_subtree_base + child2, child_u1, mask=child2_mask)
            node_start += BLOCK_NODES
        tl.debug_barrier()

    # ---- close the loop: T = sum recv p = T0 + T1 T, so T = T0 / (1 - T1).
    transfer_constant = tl.full((), value=0.0, dtype=DTYPE)
    transfer_gain = tl.full((), value=0.0, dtype=DTYPE)
    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        mask = s_offs < S
        alpha = tl.load(scratch_ptr + alpha_base + s_offs, mask=mask, other=0.0)
        gamma = tl.load(scratch_ptr + gamma_base + s_offs, mask=mask, other=0.0)
        u0 = tl.load(scratch_ptr + mass_off_chain_base + s_offs, mask=mask, other=0.0)
        u1 = tl.load(scratch_ptr + mass_subtree_base + s_offs, mask=mask, other=0.0)
        p0 = alpha + gamma * u0
        p1 = gamma * u1
        if USE_RECEIVER_WEIGHTS:
            receiver_weight = tl.load(receiver_lin_ptr + s_offs, mask=mask, other=0.0)
            p0 = p0 * receiver_weight
            p1 = p1 * receiver_weight
        transfer_constant += tl.sum(tl.where(mask, p0, 0.0), axis=0)
        transfer_gain += tl.sum(tl.where(mask, p1, 0.0), axis=0)
    loop_denominator = 1.0 - transfer_gain
    total_receiver_mass = transfer_constant / loop_denominator
    if WRITE_GUARD_TRIPS:
        tl.store(guard_trips_ptr + 4 * global_row, guard_trips.to(DTYPE))
        tl.store(
            guard_trips_ptr + 4 * global_row + 1,
            tl.where(loop_denominator <= 0.0, 1.0, 0.0).to(DTYPE),
        )
        # The two conditioning margins themselves, so a row that came out wrong without tripping
        # a sign can still be explained.
        tl.store(guard_trips_ptr + 4 * global_row + 2, smallest_pivot)
        tl.store(guard_trips_ptr + 4 * global_row + 3, loop_denominator)

    # ---- the conditioning decision, and the second reason to hand a row over. Range was the
    # first: a row too tall for one scale. This is the other way the elimination fails -- every
    # lane divides by a pivot, and the whole row divides once by ``1 - loop gain``, so a row whose
    # smallest divisor is within a rounding of zero loses digits in proportion. Both margins are
    # order-1 quantities (one minus probabilities), so ``conditioning_floor`` compares against
    # them directly. Deciding here, before walks 3 and 4 and before anything is published, leaves
    # the row exactly as this kernel found it, same as the range check does.
    if (smallest_pivot < conditioning_floor) or (loop_denominator < conditioning_floor):
        tl.store(wide_row_ptr + global_row, 1)
        tl.atomic_add(wide_row_count_ptr + wave_index, 1)
        return
    tl.debug_barrier()

    # ---- seed walk 3 with each species' OWN receiver mass, from the first-pass solution
    # p1 = alpha + gamma (U0 + U1 T). A first-pass lane that came out negative is a lane whose
    # true value is far below the row maximum and whose two nearly-equal terms cancelled; it
    # carries no mass, so it enters the mass sums as zero. This retires U0 and U1.
    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        mask = s_offs < S
        alpha = tl.load(scratch_ptr + alpha_base + s_offs, mask=mask, other=0.0)
        gamma = tl.load(scratch_ptr + gamma_base + s_offs, mask=mask, other=0.0)
        u0 = tl.load(scratch_ptr + mass_off_chain_base + s_offs, mask=mask, other=0.0)
        u1 = tl.load(scratch_ptr + mass_subtree_base + s_offs, mask=mask, other=0.0)
        first_pass = alpha + gamma * (u0 + u1 * total_receiver_mass)
        own_mass = tl.where(
            mask & (first_pass > 0.0), first_pass, tl.zeros([BLOCK_S], dtype=DTYPE)
        )
        if USE_RECEIVER_WEIGHTS:
            receiver_weight = tl.load(receiver_lin_ptr + s_offs, mask=mask, other=0.0)
            own_mass = own_mass * receiver_weight
        tl.store(scratch_ptr + mass_subtree_base + s_offs, own_mass, mask=mask)
    tl.debug_barrier()

    # ---- walk 3: M[s] += M[c1] + M[c2], leaves upward. Each M[s] ends as the total receiver mass
    # of s's whole subtree, built by addition only.
    for level in range(0, n_levels):
        level_start = tl.load(level_ptr_ptr + level)
        level_end = tl.load(level_ptr_ptr + level + 1)
        node_start = level_start
        while node_start < level_end:
            node_offs = node_start + tl.arange(0, BLOCK_NODES)
            node_mask = node_offs < level_end
            parent = tl.load(level_parents_ptr + node_offs, mask=node_mask, other=0).to(tl.int64)
            child1 = tl.load(level_child1_ptr + node_offs, mask=node_mask, other=S).to(tl.int64)
            child2 = tl.load(level_child2_ptr + node_offs, mask=node_mask, other=S).to(tl.int64)
            parent_mass = tl.load(scratch_ptr + mass_subtree_base + parent, mask=node_mask, other=0.0)
            child1_mass = tl.load(
                scratch_ptr + mass_subtree_base + child1, mask=node_mask & (child1 < S), other=0.0
            )
            child2_mass = tl.load(
                scratch_ptr + mass_subtree_base + child2, mask=node_mask & (child2 < S), other=0.0
            )
            tl.store(
                scratch_ptr + mass_subtree_base + parent,
                parent_mass + child1_mass + child2_mass,
                mask=node_mask,
            )
            node_start += BLOCK_NODES
        tl.debug_barrier()

    # ---- seed walk 4: R[root] = 0, the mass hanging off an empty ancestor chain. Written to
    # every species for the same reason as the walk-2 seed; this retires U0's slot.
    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        mask = s_offs < S
        tl.store(
            scratch_ptr + mass_off_chain_base + s_offs, tl.zeros([BLOCK_S], dtype=DTYPE), mask=mask
        )
    tl.debug_barrier()

    # ---- walk 4: R[c1] = R[s] + M[c2], R[c2] = R[s] + M[c1], root downward. Each R[s] ends as
    # the receiver mass sitting outside s's subtree and outside s's ancestors -- again addition
    # only, so u[s] = R[s] + M[s] never cancels.
    for level_index in range(0, n_levels):
        level = n_levels - 1 - level_index
        level_start = tl.load(level_ptr_ptr + level)
        level_end = tl.load(level_ptr_ptr + level + 1)
        node_start = level_start
        while node_start < level_end:
            node_offs = node_start + tl.arange(0, BLOCK_NODES)
            node_mask = node_offs < level_end
            parent = tl.load(level_parents_ptr + node_offs, mask=node_mask, other=0).to(tl.int64)
            child1 = tl.load(level_child1_ptr + node_offs, mask=node_mask, other=S).to(tl.int64)
            child2 = tl.load(level_child2_ptr + node_offs, mask=node_mask, other=S).to(tl.int64)
            child1_mask = node_mask & (child1 < S)
            child2_mask = node_mask & (child2 < S)
            parent_off_chain = tl.load(
                scratch_ptr + mass_off_chain_base + parent, mask=node_mask, other=0.0
            )
            child1_mass = tl.load(scratch_ptr + mass_subtree_base + child1, mask=child1_mask, other=0.0)
            child2_mass = tl.load(scratch_ptr + mass_subtree_base + child2, mask=child2_mask, other=0.0)
            tl.store(
                scratch_ptr + mass_off_chain_base + child1,
                parent_off_chain + child2_mass,
                mask=child1_mask,
            )
            tl.store(
                scratch_ptr + mass_off_chain_base + child2,
                parent_off_chain + child1_mass,
                mask=child2_mask,
            )
            node_start += BLOCK_NODES
        tl.debug_barrier()

    # ---- publish: p = alpha + gamma (R + M) and Pibar = mt (R + M[c1] + M[c2]), both sums of
    # non-negative terms, then the log2 residual + offset pair for each row.
    final_receiver_max = tl.full((), value=0.0, dtype=DTYPE)
    max_log2_change = tl.full((), value=0.0, dtype=tl.float32)
    pi_has_finite = tl.full((), value=0, dtype=tl.int32)
    pibar_has_finite = tl.full((), value=0, dtype=tl.int32)
    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        mask = s_offs < S
        const_offsets = const_base + s_offs
        if COMPUTE_DIFF:
            # Read BEFORE this loop's stores: for a leaf wave ``Pi_in`` is the same tensor as
            # ``Pibar_out``, and for a split wave it is the same tensor as ``Pi_out``.
            entry_residual = tl.load(
                Pi_in_ptr + global_base + s_offs, mask=mask, other=NEG_LARGE
            )
        alpha = tl.load(scratch_ptr + alpha_base + s_offs, mask=mask, other=0.0)
        gamma = tl.load(scratch_ptr + gamma_base + s_offs, mask=mask, other=0.0)
        off_chain_mass = tl.load(scratch_ptr + mass_off_chain_base + s_offs, mask=mask, other=0.0)
        subtree_mass = tl.load(scratch_ptr + mass_subtree_base + s_offs, mask=mask, other=0.0)
        child1 = tl.load(species_child1_ptr + s_offs, mask=mask, other=S).to(tl.int64)
        child2 = tl.load(species_child2_ptr + s_offs, mask=mask, other=S).to(tl.int64)
        child1_mass = tl.load(
            scratch_ptr + mass_subtree_base + child1, mask=mask & (child1 < S), other=0.0
        )
        child2_mass = tl.load(
            scratch_ptr + mass_subtree_base + child2, mask=mask & (child2 < S), other=0.0
        )

        available_mass = off_chain_mass + subtree_mass
        final_likelihood = tl.where(
            mask, alpha + gamma * available_mass, tl.zeros([BLOCK_S], dtype=DTYPE)
        )
        # T - X[s]: what is left once s's OWN subtree mass is dropped from ``available_mass``.
        valid_receiver_mass = off_chain_mass + child1_mass + child2_mass
        max_transfer = tl.load(max_transfer_lin_ptr + const_offsets, mask=mask, other=0.0)
        transfer_complement_likelihood = max_transfer * tl.where(
            valid_receiver_mass > 0.0, valid_receiver_mass, tl.zeros([BLOCK_S], dtype=DTYPE)
        )

        weighted = final_likelihood
        if USE_RECEIVER_WEIGHTS:
            receiver_weight = tl.load(receiver_lin_ptr + s_offs, mask=mask, other=0.0)
            weighted = weighted * receiver_weight
        final_receiver_max = tl.maximum(
            final_receiver_max,
            tl.max(tl.where(mask, weighted, tl.zeros([BLOCK_S], dtype=DTYPE)), axis=0),
        )

        final_residual = tl.where(
            mask & (final_likelihood > 0.0), tl.log2(final_likelihood), NEG_LARGE
        )
        pibar_residual = tl.where(
            mask & (transfer_complement_likelihood > 0.0),
            tl.log2(transfer_complement_likelihood),
            NEG_LARGE,
        )
        tl.store(Pi_out_ptr + global_base + s_offs, final_residual, mask=mask)
        tl.store(Pibar_out_ptr + global_base + s_offs, pibar_residual, mask=mask)
        pi_has_finite = tl.maximum(
            pi_has_finite, tl.max(tl.where(final_residual != NEG_LARGE, 1, 0), axis=0)
        )
        pibar_has_finite = tl.maximum(
            pibar_has_finite, tl.max(tl.where(pibar_residual != NEG_LARGE, 1, 0), axis=0)
        )
        if COMPUTE_DIFF:
            # Distance from the iterate this solve was handed to the solved row, in absolute
            # log2 units: the two live in different frames, so both offsets are added back.
            both_finite = mask & (final_residual != NEG_LARGE) & (entry_residual != NEG_LARGE)
            change = tl.where(
                both_finite,
                tl.abs(
                    final_residual.to(ACC_DTYPE)
                    + scale
                    - (entry_residual.to(ACC_DTYPE) + pi_in_offset)
                ),
                tl.zeros([BLOCK_S], dtype=ACC_DTYPE),
            )
            max_log2_change = tl.maximum(max_log2_change, tl.max(change, axis=0).to(tl.float32))

    tl.store(
        pibar_row_max_ptr + global_row,
        tl.where(final_receiver_max > 0.0, tl.log2(final_receiver_max), NEG_LARGE),
    )
    tl.store(
        Pi_out_offset_ptr + global_row,
        tl.where(pi_has_finite != 0, scale, tl.zeros_like(scale)),
    )
    tl.store(
        Pibar_offset_ptr + global_row,
        tl.where(pibar_has_finite != 0, scale, tl.zeros_like(scale)),
    )
    if COMPUTE_DIFF:
        tl.store(pi_residual_out_ptr + global_row, max_log2_change)


# ``family_offset`` is the wave's start row and changes every launch; keeping it out of
# the specialization key avoids one JIT compile per divisibility state (see README.md).
@triton.jit(do_not_specialize=["family_offset"])
def _reduce_single_gene_split_events_kernel(
    Pi,
    Pi_offset,
    Pibar,
    Pibar_offset,
    split_left_rows,
    split_right_rows,
    species_child1,
    species_child2,
    log_pD,
    log_pS,
    log_split_probs,
    single_split_parent_rows,
    active_rows,
    gene_split_log_likelihood,
    gene_split_offset,
    family_idx,
    family_offset,
    S: tl.constexpr,
    BLOCK_S: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    BY_SPECIES: tl.constexpr,
    USE_ACTIVE: tl.constexpr,
    DTYPE: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
):
    NEG_INF: tl.constexpr = -float("inf")
    n = tl.program_id(0)
    s_block = tl.program_id(1)
    s_offs = s_block * BLOCK_S + tl.arange(0, BLOCK_S)
    mask = s_offs < S
    parent_wave_row = tl.load(single_split_parent_rows + n).to(tl.int64)
    if USE_ACTIVE:
        if tl.load(active_rows + parent_wave_row) == 0:
            tl.store(gene_split_log_likelihood + parent_wave_row * S + s_offs, tl.full([BLOCK_S], NEG_INF, dtype=DTYPE), mask=mask)
            if s_block == 0:
                tl.store(gene_split_offset + parent_wave_row, 0.0)
            return

    family = tl.load(family_idx + family_offset + parent_wave_row).to(tl.int64)
    left_clade_row = tl.load(split_left_rows + n).to(tl.int64)
    right_clade_row = tl.load(split_right_rows + n).to(tl.int64)
    left_base = left_clade_row * S
    right_base = right_clade_row * S
    left_pi = tl.load(Pi + left_base + s_offs, mask=mask, other=NEG_INF)
    right_pi = tl.load(Pi + right_base + s_offs, mask=mask, other=NEG_INF)
    left_pibar = tl.load(Pibar + left_base + s_offs, mask=mask, other=NEG_INF)
    right_pibar = tl.load(Pibar + right_base + s_offs, mask=mask, other=NEG_INF)
    duplication_log_probability = _load_event_log_probability(log_pD, family, s_offs, mask, S, ROW_STRIDE, BY_SPECIES, BLOCK_S, DTYPE)
    speciation_log_probability = _load_event_log_probability(log_pS, family, s_offs, mask, S, ROW_STRIDE, BY_SPECIES, BLOCK_S, DTYPE)
    c1 = tl.load(species_child1 + s_offs, mask=mask, other=S)
    c2 = tl.load(species_child2 + s_offs, mask=mask, other=S)
    c1_valid = c1 < S
    c2_valid = c2 < S
    split_log_prior = tl.load(log_split_probs + n)

    left_pi_offset = tl.load(Pi_offset + left_clade_row)
    right_pi_offset = tl.load(Pi_offset + right_clade_row)
    left_pibar_offset = tl.load(Pibar_offset + left_clade_row)
    right_pibar_offset = tl.load(Pibar_offset + right_clade_row)
    split_frame_offset = tl.maximum(
        left_pi_offset + right_pi_offset,
        left_pi_offset + right_pibar_offset,
    )
    split_frame_offset = tl.maximum(
        split_frame_offset, right_pi_offset + left_pibar_offset
    ).to(ACC_DTYPE)
    child_pair_frame_shift = (
        left_pi_offset + right_pi_offset - split_frame_offset
    ).to(DTYPE)
    left_transfer_frame_shift = (
        left_pi_offset + right_pibar_offset - split_frame_offset
    ).to(DTYPE)
    right_transfer_frame_shift = (
        right_pi_offset + left_pibar_offset - split_frame_offset
    ).to(DTYPE)

    duplication_log_term = split_log_prior + duplication_log_probability + left_pi + right_pi + child_pair_frame_shift
    transfer_left_retained_log_term = split_log_prior + left_pi + right_pibar + left_transfer_frame_shift
    transfer_right_retained_log_term = split_log_prior + right_pi + left_pibar + right_transfer_frame_shift
    speciation_lr_log_term = (
        split_log_prior
        + speciation_log_probability
        + tl.load(Pi + left_base + c1, mask=mask & c1_valid, other=NEG_INF)
        + tl.load(Pi + right_base + c2, mask=mask & c2_valid, other=NEG_INF)
        + child_pair_frame_shift
    )
    speciation_rl_log_term = (
        split_log_prior
        + speciation_log_probability
        + tl.load(Pi + right_base + c1, mask=mask & c1_valid, other=NEG_INF)
        + tl.load(Pi + left_base + c2, mask=mask & c2_valid, other=NEG_INF)
        + child_pair_frame_shift
    )
    logsumexp_max = tl.maximum(
        tl.maximum(
            tl.maximum(duplication_log_term, transfer_left_retained_log_term),
            tl.maximum(transfer_right_retained_log_term, speciation_lr_log_term),
        ),
        speciation_rl_log_term,
    )
    logsumexp_max_safe = tl.where(logsumexp_max != NEG_INF, logsumexp_max, tl.zeros_like(logsumexp_max))
    gene_split_scaled_mass = (
        tl.exp2(duplication_log_term - logsumexp_max_safe)
        + tl.exp2(transfer_left_retained_log_term - logsumexp_max_safe)
        + tl.exp2(transfer_right_retained_log_term - logsumexp_max_safe)
        + tl.exp2(speciation_lr_log_term - logsumexp_max_safe)
        + tl.exp2(speciation_rl_log_term - logsumexp_max_safe)
    )
    reduced_gene_split_log_likelihood = (
        tl.log2(gene_split_scaled_mass) + logsumexp_max
    )
    tl.store(
        gene_split_log_likelihood + parent_wave_row * S + s_offs,
        reduced_gene_split_log_likelihood,
        mask=mask,
    )
    # ``gene_split_offset`` starts at zero. Every species tile for this
    # single-split parent row has
    # the same candidate base, so any tile containing a finite lane may publish
    # it safely. If every lane is impossible, no tile writes and the canonical
    # all--inf row keeps offset zero without an extra row pass.
    tile_has_finite = (
        tl.max(
            tl.where(
                mask & (reduced_gene_split_log_likelihood != NEG_INF), 1, 0
            ),
            axis=0,
        )
        != 0
    )
    tl.store(
        gene_split_offset + parent_wave_row,
        split_frame_offset,
        mask=tile_has_finite,
    )


# ``family_offset`` (wave start row), ``split_offset`` (single-split cursor) and
# ``MAX_TILES`` all change per wave; keeping them out of the specialization key avoids
# one JIT compile per new "== 1" / divisible-by-16 state (see README.md).
@triton.jit(do_not_specialize=["family_offset", "split_offset", "MAX_TILES"])
def _stage_multiple_gene_split_event_reduction_kernel(
    Pi,
    Pi_offset,
    Pibar,
    Pibar_offset,
    split_left_rows,
    split_right_rows,
    species_child1,
    species_child2,
    log_pD,
    log_pS,
    log_split_probs,
    multiple_split_group_ptr,
    multiple_split_parent_rows,
    active_rows,
    partial_event_max_ptr,
    partial_event_scaled_mass_ptr,
    partial_frame_offset_ptr,
    family_idx,
    family_offset,
    split_offset,
    MAX_TILES,  # runtime int (per-wave tile count; constexpr caused one JIT compile per value)
    S: tl.constexpr,
    TILE_SPLITS: tl.constexpr,
    BLOCK_S: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    BY_SPECIES: tl.constexpr,
    USE_ACTIVE: tl.constexpr,
    DTYPE: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
):
    NEG_INF: tl.constexpr = -float("inf")
    # int64: group ranges over the batch's multi-split parent count, so
    # partial_row*S below can overflow int32 once that count * S exceeds 2^31.
    group = tl.program_id(0).to(tl.int64)
    tile_id = tl.program_id(1)
    s_block = tl.program_id(2)
    s_offs = s_block * BLOCK_S + tl.arange(0, BLOCK_S)
    mask = s_offs < S
    parent_wave_row = tl.load(multiple_split_parent_rows + group).to(tl.int64)
    if USE_ACTIVE:
        if tl.load(active_rows + parent_wave_row) == 0:
            return

    family = tl.load(family_idx + family_offset + parent_wave_row).to(tl.int64)
    group_start = tl.load(multiple_split_group_ptr + group)
    group_end = tl.load(multiple_split_group_ptr + group + 1)
    tile_start = group_start + tile_id * TILE_SPLITS
    if tile_start >= group_end:
        return
    tile_end = tl.minimum(tile_start + TILE_SPLITS, group_end)
    logsumexp_max = tl.full([BLOCK_S], NEG_INF, dtype=DTYPE)
    gene_split_scaled_mass = tl.zeros([BLOCK_S], dtype=DTYPE)
    # Use a tile-local accumulator frame and advance it as split offsets increase.
    # This folds the old group-wide offset prepass into work stage 1 already
    # performs, without rescanning every split once per species block.
    tile_base_offset = tl.full((), value=NEG_INF, dtype=ACC_DTYPE)
    split_rel = tile_start
    while split_rel < tile_end:
        split_i = split_offset + split_rel
        left_clade_row = tl.load(split_left_rows + split_i).to(tl.int64)
        right_clade_row = tl.load(split_right_rows + split_i).to(tl.int64)
        left_base = left_clade_row * S
        right_base = right_clade_row * S
        left_pi = tl.load(Pi + left_base + s_offs, mask=mask, other=NEG_INF)
        right_pi = tl.load(Pi + right_base + s_offs, mask=mask, other=NEG_INF)
        left_pibar = tl.load(Pibar + left_base + s_offs, mask=mask, other=NEG_INF)
        right_pibar = tl.load(Pibar + right_base + s_offs, mask=mask, other=NEG_INF)
        duplication_log_probability = _load_event_log_probability(log_pD, family, s_offs, mask, S, ROW_STRIDE, BY_SPECIES, BLOCK_S, DTYPE)
        speciation_log_probability = _load_event_log_probability(log_pS, family, s_offs, mask, S, ROW_STRIDE, BY_SPECIES, BLOCK_S, DTYPE)
        c1 = tl.load(species_child1 + s_offs, mask=mask, other=S)
        c2 = tl.load(species_child2 + s_offs, mask=mask, other=S)
        c1_valid = c1 < S
        c2_valid = c2 < S
        split_log_prior = tl.load(log_split_probs + split_i)

        left_pi_offset = tl.load(Pi_offset + left_clade_row)
        right_pi_offset = tl.load(Pi_offset + right_clade_row)
        left_pibar_offset = tl.load(Pibar_offset + left_clade_row)
        right_pibar_offset = tl.load(Pibar_offset + right_clade_row)
        split_base_offset = tl.maximum(left_pi_offset + right_pi_offset, left_pi_offset + right_pibar_offset)
        split_base_offset = tl.maximum(split_base_offset, right_pi_offset + left_pibar_offset)
        new_base_offset = tl.maximum(tile_base_offset, split_base_offset)
        frame_shift = tl.where(
            tile_base_offset != NEG_INF,
            tile_base_offset - new_base_offset,
            0.0,
        ).to(DTYPE)
        logsumexp_max = tl.where(logsumexp_max != NEG_INF, logsumexp_max + frame_shift, logsumexp_max)
        child_pair_frame_shift = (left_pi_offset + right_pi_offset - new_base_offset).to(DTYPE)
        left_transfer_frame_shift = (left_pi_offset + right_pibar_offset - new_base_offset).to(DTYPE)
        right_transfer_frame_shift = (right_pi_offset + left_pibar_offset - new_base_offset).to(DTYPE)

        duplication_log_term = split_log_prior + duplication_log_probability + left_pi + right_pi + child_pair_frame_shift
        transfer_left_retained_log_term = split_log_prior + left_pi + right_pibar + left_transfer_frame_shift
        transfer_right_retained_log_term = split_log_prior + right_pi + left_pibar + right_transfer_frame_shift
        speciation_lr_log_term = (
            split_log_prior
            + speciation_log_probability
            + tl.load(Pi + left_base + c1, mask=mask & c1_valid, other=NEG_INF)
            + tl.load(Pi + right_base + c2, mask=mask & c2_valid, other=NEG_INF)
            + child_pair_frame_shift
        )
        speciation_rl_log_term = (
            split_log_prior
            + speciation_log_probability
            + tl.load(Pi + right_base + c1, mask=mask & c1_valid, other=NEG_INF)
            + tl.load(Pi + left_base + c2, mask=mask & c2_valid, other=NEG_INF)
            + child_pair_frame_shift
        )
        split_event_max = tl.maximum(
            tl.maximum(
                tl.maximum(duplication_log_term, transfer_left_retained_log_term),
                tl.maximum(transfer_right_retained_log_term, speciation_lr_log_term),
            ),
            speciation_rl_log_term,
        )
        split_event_max_safe = tl.where(split_event_max != NEG_INF, split_event_max, tl.zeros_like(split_event_max))
        split_event_scaled_mass = (
            tl.exp2(duplication_log_term - split_event_max_safe)
            + tl.exp2(transfer_left_retained_log_term - split_event_max_safe)
            + tl.exp2(transfer_right_retained_log_term - split_event_max_safe)
            + tl.exp2(speciation_lr_log_term - split_event_max_safe)
            + tl.exp2(speciation_rl_log_term - split_event_max_safe)
        )

        merged_event_max = tl.maximum(logsumexp_max, split_event_max)
        merged_event_max_safe = tl.where(merged_event_max != NEG_INF, merged_event_max, tl.zeros_like(merged_event_max))
        gene_split_scaled_mass = (
            tl.where(logsumexp_max != NEG_INF, gene_split_scaled_mass * tl.exp2(logsumexp_max - merged_event_max_safe), tl.zeros_like(gene_split_scaled_mass))
            + split_event_scaled_mass * tl.exp2(split_event_max_safe - merged_event_max_safe)
        )
        logsumexp_max = merged_event_max
        tile_base_offset = new_base_offset
        split_rel += 1

    partial_row = group * MAX_TILES + tile_id
    tl.store(
        partial_event_max_ptr + partial_row * S + s_offs,
        logsumexp_max,
        mask=mask,
    )
    tl.store(
        partial_event_scaled_mass_ptr + partial_row * S + s_offs,
        gene_split_scaled_mass,
        mask=mask,
    )
    # Species blocks race only to publish the same scalar tile frame.
    tl.store(partial_frame_offset_ptr + partial_row, tile_base_offset)


# ``MAX_TILES`` is the wave's tile count and changes every launch; keeping it out of the
# specialization key avoids one JIT compile per new value class (see README.md).
@triton.jit(do_not_specialize=["MAX_TILES"])
def _finalize_multiple_gene_split_event_reduction_kernel(
    multiple_split_group_ptr,
    multiple_split_parent_rows,
    active_rows,
    partial_event_max_ptr,
    partial_event_scaled_mass_ptr,
    partial_frame_offset_ptr,
    gene_split_log_likelihood,
    gene_split_offset,
    MAX_TILES,  # runtime int (per-wave tile count; constexpr caused one JIT compile per value)
    S: tl.constexpr,
    TILE_SPLITS: tl.constexpr,
    BLOCK_S: tl.constexpr,
    USE_ACTIVE: tl.constexpr,
    DTYPE: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
):
    NEG_INF: tl.constexpr = -float("inf")
    # int64: group ranges over the batch's multi-split parent count, so
    # partial_row*S below can overflow int32 once that count * S exceeds 2^31.
    group = tl.program_id(0).to(tl.int64)
    s_block = tl.program_id(1)
    s_offs = s_block * BLOCK_S + tl.arange(0, BLOCK_S)
    mask = s_offs < S
    parent_wave_row = tl.load(multiple_split_parent_rows + group).to(tl.int64)
    if USE_ACTIVE:
        if tl.load(active_rows + parent_wave_row) == 0:
            tl.store(gene_split_log_likelihood + parent_wave_row * S + s_offs, tl.full([BLOCK_S], NEG_INF, dtype=DTYPE), mask=mask)
            return

    group_start = tl.load(multiple_split_group_ptr + group)
    group_end = tl.load(multiple_split_group_ptr + group + 1)
    n_tiles = tl.cdiv(group_end - group_start, TILE_SPLITS)
    logsumexp_max = tl.full([BLOCK_S], NEG_INF, dtype=DTYPE)
    gene_split_scaled_mass = tl.zeros([BLOCK_S], dtype=DTYPE)
    row_base_offset = tl.full((), value=NEG_INF, dtype=ACC_DTYPE)
    tile_id = 0
    while tile_id < n_tiles:
        partial_row = group * MAX_TILES + tile_id
        partial_event_max = tl.load(
            partial_event_max_ptr + partial_row * S + s_offs,
            mask=mask,
            other=NEG_INF,
        )
        partial_event_scaled_mass = tl.load(
            partial_event_scaled_mass_ptr + partial_row * S + s_offs,
            mask=mask,
            other=0.0,
        )
        tile_base_offset = tl.load(partial_frame_offset_ptr + partial_row)
        new_base_offset = tl.maximum(row_base_offset, tile_base_offset)
        logsumexp_max = tl.where(
            logsumexp_max != NEG_INF,
            logsumexp_max + (row_base_offset - new_base_offset).to(DTYPE),
            logsumexp_max,
        )
        partial_event_max = tl.where(
            partial_event_max != NEG_INF,
            partial_event_max + (tile_base_offset - new_base_offset).to(DTYPE),
            partial_event_max,
        )
        merged_event_max = tl.maximum(logsumexp_max, partial_event_max)
        merged_event_max_safe = tl.where(merged_event_max != NEG_INF, merged_event_max, tl.zeros_like(merged_event_max))
        gene_split_scaled_mass = tl.where(logsumexp_max != NEG_INF, gene_split_scaled_mass * tl.exp2(logsumexp_max - merged_event_max_safe), tl.zeros_like(gene_split_scaled_mass)) + tl.where(
            partial_event_max != NEG_INF, partial_event_scaled_mass * tl.exp2(partial_event_max - merged_event_max_safe), tl.zeros_like(gene_split_scaled_mass)
        )
        logsumexp_max = merged_event_max
        row_base_offset = new_base_offset
        tile_id += 1

    reduced_gene_split_log_likelihood = (
        tl.log2(gene_split_scaled_mass) + logsumexp_max
    )
    tl.store(
        gene_split_log_likelihood + parent_wave_row * S + s_offs,
        reduced_gene_split_log_likelihood,
        mask=mask,
    )
    tile_has_finite = (
        tl.max(
            tl.where(
                mask & (reduced_gene_split_log_likelihood != NEG_INF), 1, 0
            ),
            axis=0,
        )
        != 0
    )
    tl.store(gene_split_offset + parent_wave_row, row_base_offset, mask=tile_has_finite)


def compute_leaf_initial_wave_step(
    Pi_out,
    Pi_out_offset,
    ws,
    W,
    S,
    max_transfer_mat,
    duplication_loss_const,
    Ebar,
    E,
    speciation_child1_const,
    speciation_child2_const,
    receiver_log_probs,
    species_child1,
    species_child2,
    species_subtree_start,
    species_subtree_end,
    leaf_species_idx,
    leaf_logp,
    family_idx,
    use_receiver_weights=True,
    leaf_fm_log=None,
):
    # ``leaf_fm_log`` is accepted for signature symmetry with ``compute_wave_step``
    # and forwarded by ``pi_wave_forward``. The leaf initializer only seeds
    # iterate 0 of the fixed point; the per-column off-hit baseline is carried by
    # the main wave-step recurrence, so this kwarg is intentionally unused here.
    del leaf_fm_log
    _validate_residual_tensors(
        Pi_out,
        max_transfer_mat=max_transfer_mat,
        duplication_loss_const=duplication_loss_const,
        Ebar=Ebar,
        E=E,
        speciation_child1_const=speciation_child1_const,
        speciation_child2_const=speciation_child2_const,
        receiver_log_probs=receiver_log_probs,
        leaf_logp=leaf_logp,
    )
    Pi_out_offset = _validate_offset_tensor(
        "Pi_out_offset",
        Pi_out_offset,
        rows=Pi_out.shape[0],
        device=Pi_out.device,
        residual_dtype=Pi_out.dtype,
    )
    block_s, const_row_stride = _prepare_wave_launch(S, duplication_loss_const)
    _initialize_leaf_reconciliation_likelihood_kernel[(W,)](
        Pi_out,
        Pi_out_offset,
        ws,
        max_transfer_mat,
        duplication_loss_const,
        Ebar,
        E,
        speciation_child1_const,
        speciation_child2_const,
        receiver_log_probs,
        species_child1,
        species_child2,
        species_subtree_start,
        species_subtree_end,
        leaf_species_idx,
        leaf_logp,
        family_idx,
        S,
        stride=S,
        CONST_ROW_STRIDE=const_row_stride,
        BLOCK_S=block_s,
        USE_RECEIVER_WEIGHTS=bool(use_receiver_weights),
        DTYPE=_tl_float_dtype(Pi_out.dtype),
        ACC_DTYPE=_tl_float_dtype(Pi_out_offset.dtype),
        num_warps=8,
    )


def compute_wave_step(
    Pi_in,
    Pi_in_offset,
    Pi_out,
    Pi_out_offset,
    Pibar,
    Pibar_offset,
    ws,
    W,
    S,
    max_transfer_mat,
    duplication_loss_const,
    Ebar,
    E,
    speciation_child1_const,
    speciation_child2_const,
    receiver_log_probs,
    species_child1,
    species_child2,
    species_parent,
    max_ancestor_depth,
    gene_split_log_likelihood=None,
    gene_split_offset=None,
    gene_split_center_offset=None,
    *,
    leaf_species_idx,
    leaf_logp,
    family_idx,
    pibar_row_max,
    store_final_pibar=False,
    has_leaf_term=True,
    input_ws=None,
    use_receiver_weights=True,
    pi_residual_out=None,
    leaf_fm_log=None,
    row_mask,
):
    _validate_residual_tensors(
        Pi_in,
        Pi_out=Pi_out,
        Pibar=Pibar,
        max_transfer_mat=max_transfer_mat,
        duplication_loss_const=duplication_loss_const,
        Ebar=Ebar,
        E=E,
        speciation_child1_const=speciation_child1_const,
        speciation_child2_const=speciation_child2_const,
        receiver_log_probs=receiver_log_probs,
        leaf_logp=leaf_logp,
        pibar_row_max=pibar_row_max,
        gene_split_log_likelihood=gene_split_log_likelihood,
    )
    Pi_in_offset = _validate_offset_tensor(
        "Pi_in_offset",
        Pi_in_offset,
        rows=Pi_in.shape[0],
        device=Pi_in.device,
        residual_dtype=Pi_in.dtype,
    )
    accumulator_dtype = Pi_in_offset.dtype
    Pi_out_offset = _validate_offset_tensor(
        "Pi_out_offset",
        Pi_out_offset,
        rows=Pi_out.shape[0],
        device=Pi_in.device,
        dtype=accumulator_dtype,
    )
    Pibar_offset = _validate_offset_tensor(
        "Pibar_offset",
        Pibar_offset,
        rows=Pibar.shape[0],
        device=Pi_in.device,
        dtype=accumulator_dtype,
    )
    has_splits = gene_split_log_likelihood is not None
    if has_splits and gene_split_offset is None:
        raise ValueError("gene_split_offset is required with row-gauged split DTS input")
    if has_splits and gene_split_center_offset is None:
        # Direct callers that are not the first DTS-input launch have no
        # virtual shift to publish; their storage offset is the correct base.
        # The first input launch writes every freshly allocated sidecar lane.
        gene_split_center_offset = (
            torch.empty_like(gene_split_offset) if input_ws is not None else gene_split_offset
        )
    if not has_splits:
        gene_split_log_likelihood = Pi_in
        gene_split_offset = Pi_in_offset
        gene_split_center_offset = Pi_in_offset
    else:
        gene_split_offset = _validate_offset_tensor(
            "gene_split_offset",
            gene_split_offset,
            rows=W,
            device=Pi_in.device,
            dtype=accumulator_dtype,
        )
        gene_split_center_offset = _validate_offset_tensor(
            "gene_split_center_offset",
            gene_split_center_offset,
            rows=W,
            device=Pi_in.device,
            dtype=accumulator_dtype,
        )
    if input_ws is not None and (
        not has_splits
        or int(input_ws) != 0
        or Pi_in.data_ptr() != gene_split_log_likelihood.data_ptr()
        or Pi_in_offset.data_ptr() != gene_split_offset.data_ptr()
        or Pi_out.data_ptr() == gene_split_log_likelihood.data_ptr()
        or gene_split_center_offset.data_ptr() == gene_split_offset.data_ptr()
        or gene_split_center_offset.data_ptr() == Pi_out_offset.data_ptr()
        or gene_split_center_offset.data_ptr() == Pibar_offset.data_ptr()
    ):
        raise ValueError(
            "split virtual framing requires wave-local aliased Pi/DTS inputs "
            "and a distinct output buffer"
        )
    compute_diff = pi_residual_out is not None
    # ``row_mask`` restricts the sweep to the clade rows the exact solve handed back; None sweeps
    # every row, which is what the "log" mode itself does.
    use_row_mask = row_mask is not None
    use_fraction_missing = leaf_fm_log is not None
    # When there is no fraction-missing tensor the constexpr short-circuits the
    # off-hit load, so a valid-but-unused 1-element placeholder is enough.
    leaf_fm_log_arg = (
        leaf_fm_log.contiguous()
        if use_fraction_missing
        else torch.empty(1, device=Pi_in.device, dtype=Pi_in.dtype)
    )
    block_s, const_row_stride = _prepare_wave_launch(S, duplication_loss_const)
    _update_reconciliation_likelihood_kernel[(W,)](
        Pi_in,
        Pi_in_offset,
        ws,
        ws if input_ws is None else int(input_ws),
        max_transfer_mat,
        duplication_loss_const,
        Ebar,
        E,
        speciation_child1_const,
        speciation_child2_const,
        receiver_log_probs,
        species_child1,
        species_child2,
        species_parent,
        leaf_species_idx,
        leaf_logp,
        leaf_fm_log_arg,
        family_idx,
        gene_split_log_likelihood,
        gene_split_offset,
        gene_split_center_offset,
        has_splits,
        bool(input_ws is not None),
        Pi_out,
        Pi_out_offset,
        Pibar,
        Pibar_offset,
        pibar_row_max,
        pi_residual_out if compute_diff else pibar_row_max,
        row_mask if use_row_mask else leaf_species_idx,
        S,
        stride=S,
        CONST_ROW_STRIDE=const_row_stride,
        BLOCK_S=block_s,
        MAX_ANCESTOR_DEPTH=int(max_ancestor_depth),
        USE_LEAF_INDEX=bool(has_leaf_term),
        USE_FRACTION_MISSING=use_fraction_missing,
        STORE_FINAL_PIBAR=bool(store_final_pibar),
        COMPUTE_DIFF=compute_diff,
        USE_RECEIVER_WEIGHTS=bool(use_receiver_weights),
        USE_ROW_MASK=use_row_mask,
        DTYPE=_tl_float_dtype(Pi_in.dtype),
        ACC_DTYPE=_tl_float_dtype(accumulator_dtype),
        num_warps=_NUM_WARPS_UPDATE_RECONCILIATION,
    )


def compute_exact_tree_self_loop(
    Pi_in,
    Pi_in_offset,
    Pi_out,
    Pi_out_offset,
    Pibar,
    Pibar_offset,
    scratch,
    ws,
    W,
    S,
    self_diagonal_lin,
    transfer_coefficient_lin,
    speciation_child1_lin,
    speciation_child2_lin,
    max_transfer_lin,
    receiver_lin,
    species_child1,
    species_child2,
    compact_level_ptr,
    compact_level_parents,
    compact_level_child1,
    compact_level_child2,
    gene_split_log_likelihood,
    gene_split_offset,
    gene_split_center_offset,
    *,
    leaf_species_idx,
    leaf_logp,
    family_idx,
    pibar_row_max,
    has_leaf_term,
    use_receiver_weights,
    pi_residual_out,
    guard_trips_out,
    leaf_fm_log,
    wide_row,
    wide_row_count,
    wave_index,
    range_log2,
):
    """Launch :func:`_exact_tree_pi_self_loop_kernel` for one wave.

    One launch per wave. The wave's current iterate arrives in ``Pi_in``/``Pi_in_offset`` and
    the answer is published into ``Pi_out``/``Pi_out_offset`` with its Pibar in
    ``Pibar``/``Pibar_offset``, exactly where the final log-space iteration would have put them.
    The answer is the EXACT fixed point rather than an iterate, so there is no iteration count and
    no tolerance; ``Pi_in`` only supplies the row gauge and the ``pi_residual_out`` reference
    point.

    ``scratch`` is the four-slot per-row working buffer, shape ``[4, rows, S]`` with
    ``rows >= W``; ``rows * S`` is passed as the slot stride so the offset arithmetic is built in
    Python rather than in int32 device arithmetic.

    ``guard_trips_out`` is an optional tensor of shape ``[clade rows, 4]`` in the model dtype:
    how many of this row's pivots were non-positive, whether ``1 - loop gain`` was, and then the
    two margins themselves -- the row's smallest pivot and its ``1 - loop gain``. The counts are
    small integers, exact in either float type, and one dtype keeps the kernel's stores uniform.
    The margins are what explains a row that came out wrong without tripping a sign. Diagnostic
    only; pass ``None`` in production.

    ``wide_row`` (``int8``, ``[clade rows]``) and ``wide_row_count`` (``int32``, one entry per
    wave, indexed by ``wave_index``) carry the range fallback. A row whose lanes reach more than
    ``range_log2`` binary orders below its own gauge cannot be held in scaled linear space at
    this dtype: the kernel writes its flag, bumps the wave's count, and returns WITHOUT touching
    the row, leaving it exactly as it was handed in for the caller to sweep in log space. The
    limit is quoted against float32 and rescaled by :func:`exact_range_for_dtype`.
    """
    _validate_residual_tensors(
        Pi_in,
        Pi_out=Pi_out,
        Pibar=Pibar,
        scratch=scratch,
        self_diagonal_lin=self_diagonal_lin,
        transfer_coefficient_lin=transfer_coefficient_lin,
        speciation_child1_lin=speciation_child1_lin,
        speciation_child2_lin=speciation_child2_lin,
        max_transfer_lin=max_transfer_lin,
        receiver_lin=receiver_lin,
        leaf_logp=leaf_logp,
        pibar_row_max=pibar_row_max,
        gene_split_log_likelihood=gene_split_log_likelihood,
    )
    if (
        scratch.ndim != 3
        or int(scratch.shape[0]) != _EXACT_TREE_SCRATCH_SLOTS
        or int(scratch.shape[2]) != int(S)
    ):
        raise ValueError(
            f"exact tree working buffer must have shape [{_EXACT_TREE_SCRATCH_SLOTS}, rows, S]"
        )
    slot_rows = int(scratch.shape[1])
    if slot_rows < int(W):
        raise ValueError("exact tree working buffer has fewer rows than the wave")
    # One entry per species-tree height, holding that height's internal nodes; a tree with no
    # internal node at all leaves both walks empty, which is the right answer for it.
    n_levels = int(compact_level_ptr.numel()) - 1
    if n_levels < 0:
        raise ValueError("compact_level_ptr must have at least one entry")
    if wide_row.dtype != torch.int8 or wide_row.numel() != int(Pi_out.shape[0]):
        raise ValueError("wide_row must be an int8 tensor with one entry per clade row")
    if wide_row_count.dtype != torch.int32 or int(wave_index) >= wide_row_count.numel():
        raise ValueError("wide_row_count must be an int32 tensor with one entry per wave")
    if float(range_log2) <= 0.0:
        raise ValueError("range_log2 must be positive")
    Pi_in_offset = _validate_offset_tensor(
        "Pi_in_offset",
        Pi_in_offset,
        rows=Pi_in.shape[0],
        device=Pi_in.device,
        residual_dtype=Pi_in.dtype,
    )
    accumulator_dtype = Pi_in_offset.dtype
    Pi_out_offset = _validate_offset_tensor(
        "Pi_out_offset",
        Pi_out_offset,
        rows=Pi_out.shape[0],
        device=Pi_in.device,
        dtype=accumulator_dtype,
    )
    Pibar_offset = _validate_offset_tensor(
        "Pibar_offset",
        Pibar_offset,
        rows=Pibar.shape[0],
        device=Pi_in.device,
        dtype=accumulator_dtype,
    )
    has_splits = gene_split_log_likelihood is not None
    if has_splits:
        if gene_split_offset is None or gene_split_center_offset is None:
            raise ValueError(
                "gene_split_offset and gene_split_center_offset are required with split DTS input"
            )
        gene_split_offset = _validate_offset_tensor(
            "gene_split_offset",
            gene_split_offset,
            rows=W,
            device=Pi_in.device,
            dtype=accumulator_dtype,
        )
        gene_split_center_offset = _validate_offset_tensor(
            "gene_split_center_offset",
            gene_split_center_offset,
            rows=W,
            device=Pi_in.device,
            dtype=accumulator_dtype,
        )
    else:
        # Unused behind the ``HAS_SPLITS`` constexpr; a valid pointer is still required.
        gene_split_log_likelihood = Pi_in
        gene_split_offset = Pi_in_offset
        gene_split_center_offset = Pi_in_offset
    compute_diff = pi_residual_out is not None
    write_guard_trips = guard_trips_out is not None
    if write_guard_trips and tuple(guard_trips_out.shape) != (int(Pi_out.shape[0]), 4):
        raise ValueError("guard_trips_out must have shape [clade rows, 4]")
    use_fraction_missing = leaf_fm_log is not None
    # When there is no fraction-missing tensor the constexpr short-circuits the
    # off-hit load, so a valid-but-unused 1-element placeholder is enough.
    leaf_fm_log_arg = (
        leaf_fm_log.contiguous()
        if use_fraction_missing
        else torch.empty(1, device=Pi_in.device, dtype=Pi_in.dtype)
    )
    _, const_row_stride = _prepare_wave_launch(S, self_diagonal_lin)
    block_s = min(_BLOCK_SPECIES_SELF_LOOP, triton.next_power_of_2(S))
    _exact_tree_pi_self_loop_kernel[(W,)](
        Pi_in,
        Pi_in_offset,
        scratch,
        Pi_out,
        Pi_out_offset,
        Pibar,
        Pibar_offset,
        pibar_row_max,
        pi_residual_out if compute_diff else pibar_row_max,
        guard_trips_out if write_guard_trips else Pi_in,
        self_diagonal_lin,
        transfer_coefficient_lin,
        speciation_child1_lin,
        speciation_child2_lin,
        max_transfer_lin,
        receiver_lin,
        species_child1,
        species_child2,
        compact_level_ptr,
        compact_level_parents,
        compact_level_child1,
        compact_level_child2,
        leaf_species_idx,
        leaf_logp,
        leaf_fm_log_arg,
        family_idx,
        gene_split_log_likelihood,
        gene_split_offset,
        gene_split_center_offset,
        wide_row,
        wide_row_count,
        ws,
        slot_rows * int(S),
        n_levels,
        exact_range_for_dtype(range_log2, Pi_in.dtype),
        exact_conditioning_floor(Pi_in.dtype),
        int(wave_index),
        S,
        stride=S,
        CONST_ROW_STRIDE=const_row_stride,
        BLOCK_S=block_s,
        BLOCK_NODES=_BLOCK_NODES_EXACT_TREE,
        HAS_SPLITS=has_splits,
        USE_LEAF_INDEX=bool(has_leaf_term),
        USE_FRACTION_MISSING=use_fraction_missing,
        COMPUTE_DIFF=compute_diff,
        WRITE_GUARD_TRIPS=write_guard_trips,
        USE_RECEIVER_WEIGHTS=bool(use_receiver_weights),
        DTYPE=_tl_float_dtype(Pi_in.dtype),
        ACC_DTYPE=_tl_float_dtype(accumulator_dtype),
        num_warps=_NUM_WARPS_EXACT_TREE,
    )


def compute_dts_forward(
    Pi,
    Pi_offset,
    Pibar,
    Pibar_offset,
    split_left_rows,
    split_right_rows,
    species_child1,
    species_child2,
    W,
    reduce_idx,
    log_pD_vec,
    log_pS_vec,
    family_idx,
    *,
    log_split_probs=None,
    n_single_split_parents=None,
    single_split_parent_rows=None,
    multiple_split_group_ptr=None,
    multiple_split_parent_rows=None,
    max_splits_per_multiple_parent=None,
    active_parent_rows=None,
    family_offset=0,
):
    N = int(split_left_rows.shape[0])
    S = int(Pi.shape[1])
    _validate_residual_tensors(
        Pi,
        Pibar=Pibar,
        log_pD_vec=log_pD_vec,
        log_pS_vec=log_pS_vec,
        log_split_probs=log_split_probs,
    )
    Pi_offset = _validate_offset_tensor(
        "Pi_offset",
        Pi_offset,
        rows=Pi.shape[0],
        device=Pi.device,
        residual_dtype=Pi.dtype,
    )
    accumulator_dtype = Pi_offset.dtype
    Pibar_offset = _validate_offset_tensor(
        "Pibar_offset",
        Pibar_offset,
        rows=Pibar.shape[0],
        device=Pi.device,
        dtype=accumulator_dtype,
    )
    gene_split_log_likelihood = torch.full((W, S), float("-inf"), device=Pi.device, dtype=Pi.dtype)
    gene_split_offset = torch.zeros((W,), device=Pi.device, dtype=accumulator_dtype)
    if N == 0:
        return gene_split_log_likelihood, gene_split_offset
    if log_split_probs is None:
        log_split_probs = torch.zeros((N,), device=Pi.device, dtype=Pi.dtype)
    else:
        log_split_probs = log_split_probs.reshape(N).contiguous()
    if n_single_split_parents is None:
        n_single_split_parents = N
        single_split_parent_rows = reduce_idx
        multiple_split_parent_rows = reduce_idx[:0]
        multiple_split_group_ptr = reduce_idx.new_zeros((1,), dtype=torch.long)
        max_splits_per_multiple_parent = 0

    by_species = log_pD_vec.ndim == 2 and int(log_pD_vec.shape[1]) != 1
    row_stride = 0 if int(log_pD_vec.shape[0]) == 1 else int(log_pD_vec.stride(0))
    block_s = min(512, triton.next_power_of_2(S))
    active = active_parent_rows if active_parent_rows is not None else reduce_idx

    if int(n_single_split_parents) > 0:
        _reduce_single_gene_split_events_kernel[(int(n_single_split_parents), triton.cdiv(S, block_s))](
            Pi,
            Pi_offset,
            Pibar,
            Pibar_offset,
            split_left_rows,
            split_right_rows,
            species_child1,
            species_child2,
            log_pD_vec,
            log_pS_vec,
            log_split_probs,
            single_split_parent_rows,
            active,
            gene_split_log_likelihood,
            gene_split_offset,
            family_idx,
            int(family_offset),
            S,
            BLOCK_S=block_s,
            ROW_STRIDE=row_stride,
            BY_SPECIES=bool(by_species),
            USE_ACTIVE=bool(active_parent_rows is not None),
            DTYPE=_tl_float_dtype(Pi.dtype),
            ACC_DTYPE=_tl_float_dtype(accumulator_dtype),
        )

    if multiple_split_parent_rows is None or int(multiple_split_parent_rows.numel()) == 0:
        return gene_split_log_likelihood, gene_split_offset
    tile_splits = 64
    if max_splits_per_multiple_parent is None:
        max_splits_per_multiple_parent = int((multiple_split_group_ptr[1:] - multiple_split_group_ptr[:-1]).max().item())
    max_tiles = max(1, triton.cdiv(int(max_splits_per_multiple_parent), tile_splits))
    n_groups = int(multiple_split_parent_rows.numel())
    partial_shape = (n_groups * max_tiles, S)
    partial_event_max = torch.empty(partial_shape, device=Pi.device, dtype=Pi.dtype)
    partial_event_scaled_mass = torch.empty(partial_shape, device=Pi.device, dtype=Pi.dtype)
    partial_frame_offset = torch.empty(
        (n_groups * max_tiles,), device=Pi.device, dtype=accumulator_dtype
    )
    _stage_multiple_gene_split_event_reduction_kernel[(n_groups, max_tiles, triton.cdiv(S, block_s))](
        Pi,
        Pi_offset,
        Pibar,
        Pibar_offset,
        split_left_rows,
        split_right_rows,
        species_child1,
        species_child2,
        log_pD_vec,
        log_pS_vec,
        log_split_probs,
        multiple_split_group_ptr,
        multiple_split_parent_rows,
        active,
        partial_event_max,
        partial_event_scaled_mass,
        partial_frame_offset,
        family_idx,
        int(family_offset),
        split_offset=int(n_single_split_parents),
        MAX_TILES=max_tiles,
        S=S,
        TILE_SPLITS=tile_splits,
        BLOCK_S=block_s,
        ROW_STRIDE=row_stride,
        BY_SPECIES=bool(by_species),
        USE_ACTIVE=bool(active_parent_rows is not None),
        DTYPE=_tl_float_dtype(Pi.dtype),
        ACC_DTYPE=_tl_float_dtype(accumulator_dtype),
    )
    _finalize_multiple_gene_split_event_reduction_kernel[(n_groups, triton.cdiv(S, block_s))](
        multiple_split_group_ptr,
        multiple_split_parent_rows,
        active,
        partial_event_max,
        partial_event_scaled_mass,
        partial_frame_offset,
        gene_split_log_likelihood,
        gene_split_offset,
        MAX_TILES=max_tiles,
        S=S,
        TILE_SPLITS=tile_splits,
        BLOCK_S=block_s,
        USE_ACTIVE=bool(active_parent_rows is not None),
        DTYPE=_tl_float_dtype(Pi.dtype),
        ACC_DTYPE=_tl_float_dtype(accumulator_dtype),
    )
    return gene_split_log_likelihood, gene_split_offset
