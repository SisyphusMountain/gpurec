import torch
import triton
import triton.language as tl

__all__ = [
    "compute_dts_forward",
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


@triton.jit
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
    DTYPE: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
):
    NEG_LARGE: tl.constexpr = -float("inf")

    # int64: w ranges over the whole batch's clade rows, so the *stride
    # multiplies below can overflow int32 once total_clades * S exceeds 2^31.
    w = tl.program_id(0).to(tl.int64)
    pi_row = pi_ws + w
    global_row = ws + w
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


@triton.jit
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


@triton.jit
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
        DTYPE=_tl_float_dtype(Pi_in.dtype),
        ACC_DTYPE=_tl_float_dtype(accumulator_dtype),
        num_warps=8,
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
