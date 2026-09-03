import torch

from ..pi_state import PiState
from ..kernels.pi_forward import (
    compute_dts_forward,
    compute_exact_tree_self_loop,
    compute_fused_linear_self_loop,
    compute_leaf_initial_wave_step,
    compute_wave_step,
    _EXACT_TREE_SCRATCH_SLOTS,
    _select_log_split_probs,
)
from ..parameters.extract_parameters import as_family_param, as_family_species
from ..valid_receivers import valid_receiver_index_tables
from gpurec.api.solver_options import dtype_scaled_self_loop_tol

# The three self-loop implementations selectable through ``SolverOptions.forward_self_loop``.
# "log": one ``compute_wave_step`` launch per fixed-point iteration, arithmetic in log2 space.
# "linear": the same recurrence, but one ``compute_fused_linear_self_loop`` launch runs every
# remaining iteration per row in scaled linear space with a per-row early exit.
# "exact": one ``compute_exact_tree_self_loop`` launch SOLVES the fixed point instead of
# iterating it (the ``max`` in the transfer complement is never active, so the fixed point is a
# linear system on the species tree). Its answer is what "log" converges to as ``pi_iters`` grows,
# so ``pi_iters`` has no effect on it beyond the shared log-space prologue below.
SELF_LOOP_MODES = ("log", "linear", "exact")


def _linear_event_multipliers(
    duplication_loss_const,
    extinction_complement,
    extinction,
    speciation_child1_const,
    speciation_child2_const,
    max_transfer,
    receiver_log_probs,
):
    """Linear-space copies of the per-species event constants the self-loop multiplies by.

    Each is ``2**(its log2 value)``. They are gauge-free (no row offset enters them), so one
    conversion per forward solve serves every wave and every fixed-point iteration; see
    :func:`gpurec.core.kernels.pi_forward._fused_linear_pi_self_loop_kernel` for how they combine.
    """
    return (
        torch.exp2(duplication_loss_const),
        torch.exp2(extinction_complement),
        torch.exp2(extinction),
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
    linear_tol: float,
    linear_iterations_out: torch.Tensor | None,
    exact_guard_trips_out: torch.Tensor | None,
):
    pi_iters = int(pi_iters)
    if pi_iters < 2 or pi_iters % 2 != 0:
        raise ValueError("pi_iters must be an even integer at least 2")
    if self_loop_mode not in SELF_LOOP_MODES:
        raise ValueError(f"self_loop_mode must be one of {SELF_LOOP_MODES}, got {self_loop_mode!r}")
    linear_tol = float(linear_tol)
    if linear_tol < 0.0:
        raise ValueError("linear_tol must be non-negative")
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
    # linear_tol is written in units of float32 precision; carry that meaning to
    # the dtype this solve actually runs in, so a float64 solve is not stopped at
    # float32 resolution (see gpurec.api.solver_options.dtype_scaled_self_loop_tol).
    linear_tol = dtype_scaled_self_loop_tol(linear_tol, dtype)

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
    sp_parent = species_helpers["sp_parent"]
    sp_subtree_start = species_helpers["sp_subtree_start"]
    sp_subtree_end = species_helpers["sp_subtree_end"]
    max_ancestor_depth = int(species_helpers["max_ancestor_depth"])

    dl_const = 1.0 + log_p_d_family + e_family
    sl1_const = log_p_s_family + e_s2_family
    sl2_const = log_p_s_family + e_s1_family

    use_linear_self_loop = self_loop_mode == "linear"
    use_exact_self_loop = self_loop_mode == "exact"
    if use_linear_self_loop or use_exact_self_loop:
        (
            dl_lin,
            e_bar_lin,
            e_lin,
            sl1_lin,
            sl2_lin,
            max_transfer_lin,
            receiver_lin,
        ) = _linear_event_multipliers(
            dl_const,
            e_bar_family,
            e_family,
            sl1_const,
            sl2_const,
            max_transfer_family,
            receiver_log_probs,
        )
        valid_receiver_tables = valid_receiver_index_tables(sp_subtree_start, sp_subtree_end, S)
        max_wave_rows = max(
            (int(meta["W"]) for meta in wave_layout["wave_metas"]), default=1
        )
    if use_linear_self_loop:
        # Four slots, one allocation reused by every wave: 0 and 1 ping-pong the iterate (the
        # update is Jacobi, so a source and a destination row are both live), 2 and 3 hold the two
        # running sums the valid receiver mass is built from.
        linear_scratch = torch.empty((4, max_wave_rows, S), dtype=dtype, device=device)
    if use_exact_self_loop:
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
        else:
            dts_r = None
            dts_offset = None
            dts_center_offset = None
        has_leaf_term = "sl" not in meta
        # Log-space prologue: the leaf initializer (iteration 0) for a leaf wave, plus the
        # gene-split-input step (iteration 1) for a split wave, which is also what publishes
        # ``dts_center_offset``. The linear kernel takes over from there.
        prologue_iters = pi_iters
        fused_iters = 0
        if use_linear_self_loop:
            candidate_prologue = 1 if has_leaf_term else 2
            if pi_iters - candidate_prologue >= 1:
                prologue_iters = candidate_prologue
                fused_iters = pi_iters - candidate_prologue
        elif use_exact_self_loop:
            # Same prologue as "linear": the leaf initializer for a leaf wave, and for a split
            # wave the gene-split-input step, which is what publishes ``dts_center_offset`` (the
            # DTS row's absolute maximum, part of the exact solve's entry gauge). ``pi_iters``
            # plays no other role here -- the solve is not an iteration.
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
                    sp_parent,
                    max_ancestor_depth,
                    dts_r,
                    dts_offset,
                    dts_center_offset,
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
                )
        if fused_iters > 0:
            # The prologue's last write lands in ``pibar`` for a leaf wave (iteration 0) and in
            # ``pi`` for a split wave (iteration 1); the fused kernel always publishes into
            # ``pi``/``pibar``.
            fused_input = pibar if has_leaf_term else pi
            fused_input_offset = pibar_offset if has_leaf_term else pi_offset
            compute_fused_linear_self_loop(
                fused_input,
                fused_input_offset,
                pi,
                pi_offset,
                pibar,
                pibar_offset,
                linear_scratch,
                ws,
                W,
                S,
                fused_iters,
                linear_tol,
                dl_lin,
                e_bar_lin,
                e_lin,
                sl1_lin,
                sl2_lin,
                max_transfer_lin,
                receiver_lin,
                sp_child1,
                sp_child2,
                valid_receiver_tables,
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
                iterations_used=linear_iterations_out,
                leaf_fm_log=leaf_fm_log,
            )
        if use_exact_self_loop:
            # Same handover as the linear path: the prologue's last write lands in ``pibar`` for a
            # leaf wave (iteration 0) and in ``pi`` for a split wave (iteration 1).
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
            )

    state = PiState(
        pi_offset=pi_offset,
        pibar_offset=pibar_offset,
    )
    state.validate(pi, pibar, uniform_pibar_row_max, check_values=False)
    root_ids = wave_layout["root_clade_ids"]
    root_rows = PiState.reconstruct_rows(
        pi.index_select(0, root_ids),
        pi_offset.index_select(0, root_ids),
    )
    return root_rows, pi, pibar, uniform_pibar_row_max, state
