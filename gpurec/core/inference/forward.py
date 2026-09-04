import torch

from ..pi_state import PiState
from ..kernels.pi_forward import (
    compute_dts_forward,
    compute_exact_tree_self_loop,
    compute_leaf_initial_wave_step,
    compute_wave_step,
    _EXACT_TREE_SCRATCH_SLOTS,
    _VALID_RECEIVER_SCRATCH_SLOTS,
    _select_log_split_probs,
)
from ..valid_receivers import valid_receiver_index_tables
from ..parameters.extract_parameters import as_family_param, as_family_species

# The two self-loop implementations selectable through ``SolverOptions.forward_self_loop``.
# "log": one ``compute_wave_step`` launch per fixed-point iteration, arithmetic in log2 space.
# "exact": one ``compute_exact_tree_self_loop`` launch SOLVES the fixed point instead of
# iterating it (the ``max`` in the transfer complement is never active, so the fixed point is a
# linear system on the species tree). Its answer is what "log" converges to as ``pi_iters`` grows,
# so ``pi_iters`` has no effect on it beyond the shared log-space prologue below.
SELF_LOOP_MODES = ("log", "exact")

# How the exact forward decides, per wave, whether to run the log-space range fallback at all.
#   "sync":   read that wave's flagged-row count back to the host and run the sweeps only when it
#             is nonzero -- one device-to-host copy per wave, and no launches at all in the
#             overwhelmingly common case of nothing flagged.
#   "always": skip the read and launch the masked sweeps regardless. They return immediately on
#             every row, so this trades the copy for ``pi_iters`` empty launches per wave, and
#             needs one whole-mask reduction at the end to learn the total.
# A module-level switch rather than a setting: which one wins is a property of the GPU and the
# wave count, not something a run should have to state.
EXACT_RANGE_FALLBACK_DECISIONS = ("sync", "always")
_EXACT_RANGE_FALLBACK_DECISION = "sync"


def set_exact_range_fallback_decision(mode):
    """Choose how the per-wave range-fallback decision is taken; see the constant above."""
    global _EXACT_RANGE_FALLBACK_DECISION
    if mode not in EXACT_RANGE_FALLBACK_DECISIONS:
        raise ValueError(
            f"decision must be one of {EXACT_RANGE_FALLBACK_DECISIONS}, got {mode!r}"
        )
    _EXACT_RANGE_FALLBACK_DECISION = mode


def _linear_event_multipliers(
    speciation_child1_const,
    speciation_child2_const,
    max_transfer,
    receiver_log_probs,
):
    """Linear-space copies of the per-species event constants the exact solve multiplies by.

    Each is ``2**(its log2 value)``. They are gauge-free (no row offset enters them), so one
    conversion per forward solve serves every wave; see
    :func:`gpurec.core.kernels.pi_forward._exact_tree_pi_self_loop_kernel` for how they combine.
    The duplication, extinction-complement and extinction constants are NOT here: the exact solve
    sees them only folded into the two coefficients :func:`_exact_tree_coefficients` builds.
    """
    return (
        torch.exp2(speciation_child1_const),
        torch.exp2(speciation_child2_const),
        torch.exp2(max_transfer),
        torch.exp2(receiver_log_probs),
    )


def _exact_tree_coefficients(
    duplication_loss_const,
    extinction_complement,
    extinction,
    max_transfer,
    receiver_log_probs,
    use_receiver_weights,
    accumulator_dtype,
):
    """The two extra per-species linear coefficients the exact tree solve needs.

    Both are per ``[family, species]``, the same layout as the multipliers above, so one build per
    forward solve serves every wave.

    ``transfer_coefficient[s] = 2**E[s] * 2**max_transfer[s]`` is what multiplies the transfer
    mass still available to species ``s`` (the row total minus s and its ancestors).

    ``self_diagonal[s] = 1 - 2**(1 + log_pD[s] + E[s]) - 2**Ebar[s]
                           + transfer_coefficient[s] * recv[s]``
    is the coefficient of ``p[s]`` in the fixed-point equation once the transfer term has been
    written as ``transfer_coefficient[s] * (T - y[s] - recv[s] p[s])``. The two subtracted terms
    are the probability of the gene staying in species ``s``, so ``1 - (them)`` is the only
    cancellation in the whole solve; it is evaluated in the accumulator dtype and only then
    rounded to the model dtype, which keeps the subtraction's relative error at the accumulator's
    resolution rather than the model dtype's.
    """
    dl = torch.exp2(duplication_loss_const.to(accumulator_dtype))
    ebar = torch.exp2(extinction_complement.to(accumulator_dtype))
    transfer_coefficient = torch.exp2(extinction.to(accumulator_dtype)) * torch.exp2(
        max_transfer.to(accumulator_dtype)
    )
    if use_receiver_weights:
        weight = torch.exp2(receiver_log_probs.to(accumulator_dtype))
    else:
        weight = torch.ones_like(receiver_log_probs, dtype=accumulator_dtype)
    self_diagonal = 1.0 - dl - ebar + transfer_coefficient * weight
    dtype = duplication_loss_const.dtype
    return (
        self_diagonal.to(dtype).contiguous(),
        transfer_coefficient.to(dtype).contiguous(),
    )


def pi_wave_forward(
    wave_layout,
    species_helpers,
    e,
    e_bar,
    e_s1,
    e_s2,
    log_p_s,
    log_p_d,
    max_transfer_mat,
    receiver_log_probs,
    use_receiver_weights: bool = True,
    *,
    family_idx: torch.Tensor,
    pi_iters: int = 6,
    pi_residual_out: torch.Tensor | None = None,
    accumulator_dtype: torch.dtype | None = None,
    leaf_fm_log: torch.Tensor | None = None,
    self_loop_mode: str,
    exact_range_log2: float,
    exact_guard_trips_out: torch.Tensor | None,
    # ``None``: each split wave's gene-split (DTS) rows are freed as soon as that wave is done,
    # and whoever needs them later recomputes them. A dict: every split wave's rows and row
    # offsets are stored in it under the wave's start row, so the backward can read the numbers
    # this forward already produced instead of running the three reduction kernels again. The
    # caller decides, because the caller is the one who knows both whether a backward is coming
    # and whether the memory is there (gpurec/core/memory_policy.forward_gene_split_cache_fits).
    gene_split_out: dict | None,
):
    pi_iters = int(pi_iters)
    if pi_iters < 2 or pi_iters % 2 != 0:
        raise ValueError("pi_iters must be an even integer at least 2")
    if self_loop_mode not in SELF_LOOP_MODES:
        raise ValueError(f"self_loop_mode must be one of {SELF_LOOP_MODES}, got {self_loop_mode!r}")
    if e.device.type != "cuda":
        raise RuntimeError("Pi forward requires CUDA")
    if e.dtype not in (torch.float32, torch.float64):
        raise RuntimeError("Pi forward requires fp32 or fp64 residual storage")
    if accumulator_dtype is None:
        accumulator_dtype = e.dtype
    if accumulator_dtype not in (torch.float32, torch.float64):
        raise TypeError("accumulator_dtype must be torch.float32 or torch.float64")
    if e.dtype == torch.float64 and accumulator_dtype != torch.float64:
        raise TypeError("accumulator_dtype must not be narrower than the Pi residual dtype")

    C = int(wave_layout["leaf_species_index"].numel())
    S = int(species_helpers["S"])
    device = e.device
    dtype = e.dtype

    if gene_split_out is not None:
        # Start from empty: a caller that hands the same dict to two solves must get the second
        # solve's rows, never a mixture with the first's.
        gene_split_out.clear()

    pi = torch.empty((C, S), dtype=dtype, device=device)
    pibar = torch.empty((C, S), dtype=dtype, device=device)
    pi_offset = torch.zeros((C,), dtype=accumulator_dtype, device=device)
    pibar_offset = torch.zeros((C,), dtype=accumulator_dtype, device=device)

    family_rows = int(e.shape[0])
    e_family = as_family_species(e, S, family_rows)
    e_bar_family = as_family_species(e_bar, S, family_rows)
    e_s1_family = as_family_species(e_s1, S, family_rows)
    e_s2_family = as_family_species(e_s2, S, family_rows)
    max_transfer_family = as_family_species(max_transfer_mat.squeeze(-1), S, family_rows)
    log_p_d_param = as_family_param(log_p_d, family_rows, S)
    log_p_s_param = as_family_param(log_p_s, family_rows, S)
    log_p_d_family = as_family_species(log_p_d, S, family_rows)
    log_p_s_family = as_family_species(log_p_s, S, family_rows)
    uniform_pibar_row_max = torch.empty((C,), dtype=dtype, device=device)

    sp_child1 = species_helpers["sp_child1"]
    sp_child2 = species_helpers["sp_child2"]
    sp_subtree_start = species_helpers["sp_subtree_start"]
    sp_subtree_end = species_helpers["sp_subtree_end"]

    # The log-space sweep builds every lane's available transfer mass by ADDITION, as two
    # running sums over fixed species orders, never as "the row total minus the donor's own
    # lineage" (see ``_write_valid_receiver_prefix_sums`` for why that subtraction is not
    # usable). The orders depend only on the species tree, so one build serves every wave.
    (
        not_open_source,
        closed_source,
        not_open_index,
        closed_index,
    ) = valid_receiver_index_tables(sp_subtree_start, sp_subtree_end, S)
    max_wave_rows = max(
        (int(meta["W"]) for meta in wave_layout["wave_metas"]), default=1
    )
    # Two slots: one running sum per receiver group, one row per clade row of the widest wave.
    valid_receiver_scratch = torch.empty(
        (_VALID_RECEIVER_SCRATCH_SLOTS, max_wave_rows, S), dtype=dtype, device=device
    )

    dl_const = 1.0 + log_p_d_family + e_family
    sl1_const = log_p_s_family + e_s2_family
    sl2_const = log_p_s_family + e_s1_family

    use_exact_self_loop = self_loop_mode == "exact"
    if use_exact_self_loop:
        (
            sl1_lin,
            sl2_lin,
            max_transfer_lin,
            receiver_lin,
        ) = _linear_event_multipliers(
            sl1_const,
            sl2_const,
            max_transfer_family,
            receiver_log_probs,
        )
        self_diag_lin, transfer_coefficient_lin = _exact_tree_coefficients(
            dl_const,
            e_bar_family,
            e_family,
            max_transfer_family,
            receiver_log_probs,
            use_receiver_weights,
            accumulator_dtype,
        )
        # Four slots: the elimination's two affine coefficients per species plus two working
        # arrays (see ``_exact_tree_pi_self_loop_kernel``'s slot table).
        exact_scratch = torch.empty(
            (_EXACT_TREE_SCRATCH_SLOTS, max_wave_rows, S), dtype=dtype, device=device
        )
        compact_level_ptr = species_helpers["compact_level_ptr"]
        compact_level_parents = species_helpers["compact_level_parents"]
        compact_level_child1 = species_helpers["compact_level_child1"]
        compact_level_child2 = species_helpers["compact_level_child2"]
        # One flag per clade row, and one count per wave so the host learns whether ANY row needs
        # the log-space fallback without reading the [C] mask back. The mask is part of the
        # forward's published state: the adjoint and the tangent of this solve must make the same
        # per-row decision it did.
        wide_row = torch.zeros((C,), dtype=torch.int8, device=device)
        wide_row_count = torch.zeros(
            (len(wave_layout["wave_metas"]),), dtype=torch.int32, device=device
        )
    wave_index = 0
    wide_row_total = 0

    # The log-space prologue never publishes the wave's final state when a one-launch self-loop
    # takes over afterwards; ``-1`` is a local iteration index no prologue step can reach.
    final_log_iter = -1 if use_exact_self_loop else pi_iters - 1

    for meta in wave_layout["wave_metas"]:
        ws = meta["start"]
        W = meta["W"]
        if "sl" in meta:
            dts_r, dts_offset = compute_dts_forward(
                pi,
                pi_offset,
                pibar,
                pibar_offset,
                meta["sl"],
                meta["sr"],
                sp_child1,
                sp_child2,
                W,
                meta["reduce_idx"],
                log_p_d_param,
                log_p_s_param,
                family_idx=family_idx,
                log_split_probs=_select_log_split_probs(meta, pi.dtype),
                n_single_split_parents=meta.get("n_eq1"),
                single_split_parent_rows=meta.get("eq1_reduce_idx"),
                multiple_split_group_ptr=meta.get("ge2_ptr"),
                multiple_split_parent_rows=meta.get("ge2_parent_ids"),
                max_splits_per_multiple_parent=meta.get("ge2_max_fanout"),
                family_offset=ws,
            )
            # Virtual gauge used only by the forward wave consumer. DTS
            # residual/offset inputs remain immutable for deterministic reads.
            dts_center_offset = torch.empty_like(dts_offset)
            if gene_split_out is not None:
                # Keeping the reference is all it takes: nothing below writes into these two
                # (the wave step and the exact solve read them and publish their virtual gauge
                # into ``dts_center_offset``), so the block the backward reads is bit-for-bit
                # what this wave used.
                gene_split_out[int(ws)] = (dts_r, dts_offset)
        else:
            dts_r = None
            dts_offset = None
            dts_center_offset = None
        has_leaf_term = "sl" not in meta
        # Log-space prologue, run by both modes: the leaf initializer (iteration 0) for a leaf
        # wave, plus the gene-split-input step (iteration 1) for a split wave, which is also what
        # publishes ``dts_center_offset``.
        prologue_iters = pi_iters
        if use_exact_self_loop:
            # The exact solve needs only that prologue: the leaf initializer for a leaf wave, and
            # for a split wave the gene-split-input step, which publishes ``dts_center_offset``
            # (the DTS row's absolute maximum, part of the exact solve's entry gauge).
            # ``pi_iters`` plays no other role here -- the solve is not an iteration.
            prologue_iters = 1 if has_leaf_term else 2
        for local_iter in range(prologue_iters):
            pi_in = pi if (local_iter % 2 == 0) else pibar
            pi_in_offset = pi_offset if (local_iter % 2 == 0) else pibar_offset
            pi_out = pibar if (local_iter % 2 == 0) else pi
            pi_out_offset = pibar_offset if (local_iter % 2 == 0) else pi_offset
            if local_iter == 0 and not has_leaf_term:
                continue
            elif local_iter == 0:
                compute_leaf_initial_wave_step(
                    pi_out,
                    pi_out_offset,
                    ws,
                    W,
                    S,
                    max_transfer_family,
                    dl_const,
                    e_bar_family,
                    e_family,
                    sl1_const,
                    sl2_const,
                    receiver_log_probs,
                    sp_child1,
                    sp_child2,
                    sp_subtree_start,
                    sp_subtree_end,
                    wave_layout["leaf_species_index"],
                    log_p_s_family,
                    family_idx=family_idx,
                    use_receiver_weights=use_receiver_weights,
                    leaf_fm_log=leaf_fm_log,
                )
            else:
                step_input_ws = 0 if local_iter == 1 and not has_leaf_term else None
                compute_wave_step(
                    dts_r if step_input_ws == 0 else pi_in,
                    dts_offset if step_input_ws == 0 else pi_in_offset,
                    pi_out,
                    pi_out_offset,
                    pibar,
                    pibar_offset,
                    ws,
                    W,
                    S,
                    max_transfer_family,
                    dl_const,
                    e_bar_family,
                    e_family,
                    sl1_const,
                    sl2_const,
                    receiver_log_probs,
                    sp_child1,
                    sp_child2,
                    dts_r,
                    dts_offset,
                    dts_center_offset,
                    valid_receiver_scratch=valid_receiver_scratch,
                    not_open_source=not_open_source,
                    closed_source=closed_source,
                    not_open_index=not_open_index,
                    closed_index=closed_index,
                    leaf_species_idx=wave_layout["leaf_species_index"],
                    leaf_logp=log_p_s_family,
                    family_idx=family_idx,
                    pibar_row_max=uniform_pibar_row_max,
                    store_final_pibar=local_iter == final_log_iter,
                    has_leaf_term=has_leaf_term,
                    input_ws=step_input_ws,
                    use_receiver_weights=use_receiver_weights,
                    pi_residual_out=(
                        pi_residual_out if local_iter == final_log_iter else None
                    ),
                    leaf_fm_log=leaf_fm_log,
                    row_mask=None,
                )
        if use_exact_self_loop:
            # The prologue's last write lands in ``pibar`` for a leaf wave (iteration 0) and in
            # ``pi`` for a split wave (iteration 1).
            exact_input = pibar if has_leaf_term else pi
            exact_input_offset = pibar_offset if has_leaf_term else pi_offset
            compute_exact_tree_self_loop(
                exact_input,
                exact_input_offset,
                pi,
                pi_offset,
                pibar,
                pibar_offset,
                exact_scratch,
                ws,
                W,
                S,
                self_diag_lin,
                transfer_coefficient_lin,
                sl1_lin,
                sl2_lin,
                max_transfer_lin,
                receiver_lin,
                sp_child1,
                sp_child2,
                compact_level_ptr,
                compact_level_parents,
                compact_level_child1,
                compact_level_child2,
                dts_r,
                dts_offset,
                dts_center_offset,
                leaf_species_idx=wave_layout["leaf_species_index"],
                leaf_logp=log_p_s_family,
                family_idx=family_idx,
                pibar_row_max=uniform_pibar_row_max,
                has_leaf_term=has_leaf_term,
                use_receiver_weights=use_receiver_weights,
                pi_residual_out=pi_residual_out,
                guard_trips_out=exact_guard_trips_out,
                leaf_fm_log=leaf_fm_log,
                wide_row=wide_row,
                wide_row_count=wide_row_count,
                wave_index=wave_index,
                range_log2=exact_range_log2,
            )
            # Two ways to spend the per-wave decision, measured against each other by
            # benchmark/cc/test_exact_range_cost.py; see ``EXACT_RANGE_FALLBACK_DECISIONS``.
            # Either way the exact kernel left every flagged row exactly as it found it, so the
            # sweeps below pick up from the prologue's output with the same buffer parity the
            # "log" mode uses, and the masked kernel returns immediately on every other row.
            if _EXACT_RANGE_FALLBACK_DECISION == "sync":
                wave_wide_rows = int(wide_row_count[wave_index].item())
                wide_row_total += wave_wide_rows
                run_fallback_sweeps = wave_wide_rows > 0
            else:
                run_fallback_sweeps = True
            # How many log-space iterations are left for the flagged rows after the prologue.
            # Normally that is ``pi_iters`` minus the prologue, exactly what "log" mode would
            # run. But a split wave's prologue is already 2 steps, so at the smallest legal
            # ``pi_iters`` (2) the range below is EMPTY: the flagged rows would get no sweep at
            # all, and since the exact kernel returned without touching them, their Pibar row and
            # its row maximum would never be written -- the caller would read back whatever the
            # freshly allocated buffer happened to hold. One full Jacobi pair is added in that
            # case so every flagged row is published. Two, not one, to keep the alternation the
            # loop relies on: the final write must land in ``pi``, which needs an odd last index.
            # For every ``pi_iters`` that leaves the prologue room -- 4 and up, so every real
            # configuration, including the production 16 -- this is exactly ``pi_iters`` and
            # nothing changes.
            fallback_end = (
                pi_iters if pi_iters > prologue_iters else prologue_iters + 2
            )
            if run_fallback_sweeps:
                for local_iter in range(prologue_iters, fallback_end):
                    pi_in = pi if (local_iter % 2 == 0) else pibar
                    pi_in_offset = pi_offset if (local_iter % 2 == 0) else pibar_offset
                    pi_out = pibar if (local_iter % 2 == 0) else pi
                    pi_out_offset = pibar_offset if (local_iter % 2 == 0) else pi_offset
                    compute_wave_step(
                        pi_in,
                        pi_in_offset,
                        pi_out,
                        pi_out_offset,
                        pibar,
                        pibar_offset,
                        ws,
                        W,
                        S,
                        max_transfer_family,
                        dl_const,
                        e_bar_family,
                        e_family,
                        sl1_const,
                        sl2_const,
                        receiver_log_probs,
                        sp_child1,
                        sp_child2,
                        dts_r,
                        dts_offset,
                        dts_center_offset,
                        valid_receiver_scratch=valid_receiver_scratch,
                        not_open_source=not_open_source,
                        closed_source=closed_source,
                        not_open_index=not_open_index,
                        closed_index=closed_index,
                        leaf_species_idx=wave_layout["leaf_species_index"],
                        leaf_logp=log_p_s_family,
                        family_idx=family_idx,
                        pibar_row_max=uniform_pibar_row_max,
                        store_final_pibar=local_iter == fallback_end - 1,
                        has_leaf_term=has_leaf_term,
                        input_ws=None,
                        use_receiver_weights=use_receiver_weights,
                        pi_residual_out=(
                            pi_residual_out if local_iter == fallback_end - 1 else None
                        ),
                        leaf_fm_log=leaf_fm_log,
                        row_mask=wide_row,
                    )
        wave_index += 1

    if use_exact_self_loop and _EXACT_RANGE_FALLBACK_DECISION != "sync":
        # The per-wave counts were never read back, so the total costs one reduction here.
        wide_row_total = int(wide_row.sum().item())
    state = PiState(
        pi_offset=pi_offset,
        pibar_offset=pibar_offset,
        wide_row=wide_row if use_exact_self_loop else None,
        wide_row_total=wide_row_total,
    )
    state.validate(pi, pibar, uniform_pibar_row_max, check_values=False)
    root_ids = wave_layout["root_clade_ids"]
    root_rows = PiState.reconstruct_rows(
        pi.index_select(0, root_ids),
        pi_offset.index_select(0, root_ids),
    )
    return root_rows, pi, pibar, uniform_pibar_row_max, state
