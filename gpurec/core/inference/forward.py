import torch

from ..pi_state import PiState
from ..kernels.pi_forward import (
    compute_dts_forward,
    compute_leaf_initial_wave_step,
    compute_wave_step,
)
from ..parameters.extract_parameters import as_family_param, as_family_species


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
):
    pi_iters = int(pi_iters)
    if pi_iters < 2 or pi_iters % 2 != 0:
        raise ValueError("pi_iters must be an even integer at least 2")
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
                log_split_probs=meta.get("log_split_probs"),
                n_eq1=meta.get("n_eq1"),
                eq1_reduce_idx=meta.get("eq1_reduce_idx"),
                ge2_ptr=meta.get("ge2_ptr"),
                ge2_parent_ids=meta.get("ge2_parent_ids"),
                ge2_max_fanout=meta.get("ge2_max_fanout"),
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
        for local_iter in range(pi_iters):
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
                    store_final_pibar=local_iter == pi_iters - 1,
                    has_leaf_term=has_leaf_term,
                    input_ws=step_input_ws,
                    use_receiver_weights=use_receiver_weights,
                    pi_residual_out=(
                        pi_residual_out if local_iter == pi_iters - 1 else None
                    ),
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
