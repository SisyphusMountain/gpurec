import torch

from ..kernels.dts_fused import compute_dts_forward
from ..kernels.wave_step import compute_leaf_initial_wave_step, compute_wave_step
from ..parameters.extract_parameters import as_family_param, as_family_species


def Pi_wave_forward(
    wave_layout,
    species_helpers,
    E,
    Ebar,
    E_s1,
    E_s2,
    log_pS,
    log_pD,
    max_transfer_mat,
    *,
    family_idx: torch.Tensor,
):
    C = int(wave_layout["leaf_species_index"].numel())
    S = int(species_helpers["S"])
    device = E.device
    dtype = E.dtype

    Pi = torch.empty((C, S), dtype=dtype, device=device)
    Pibar = torch.empty((C, S), dtype=dtype, device=device)

    family_rows = int(E.shape[0])
    E_family = as_family_species(E, S, family_rows)
    Ebar_family = as_family_species(Ebar, S, family_rows)
    E_s1_family = as_family_species(E_s1, S, family_rows)
    E_s2_family = as_family_species(E_s2, S, family_rows)
    max_transfer_family = as_family_species(max_transfer_mat.squeeze(-1), S, family_rows)
    log_pD_param = as_family_param(log_pD, family_rows)
    log_pS_param = as_family_param(log_pS, family_rows)
    log_pD_family = as_family_species(log_pD, S, family_rows)
    log_pS_family = as_family_species(log_pS, S, family_rows)

    uniform_pibar_row_max = torch.empty((C,), dtype=dtype, device=device)

    sp_child1 = species_helpers["sp_child1"]
    sp_child2 = species_helpers["sp_child2"]
    sp_parent = species_helpers["sp_parent"]
    sp_subtree_start = species_helpers["sp_subtree_start"]
    sp_subtree_end = species_helpers["sp_subtree_end"]
    max_ancestor_depth = int(species_helpers["max_ancestor_depth"])

    DL_const = 1.0 + log_pD_family + E_family
    SL1_const = log_pS_family + E_s2_family
    SL2_const = log_pS_family + E_s1_family

    for meta in wave_layout["wave_metas"]:
        ws = meta["start"]
        W = meta["W"]
        dts_r = (
            compute_dts_forward(
                Pi, Pibar, meta["sl"], meta["sr"], sp_child1, sp_child2,
                W, meta["reduce_idx"], log_pD_param, log_pS_param,
                family_idx=family_idx,
                family_offset=ws,
            )
            if "sl" in meta
            else None
        )
        has_leaf_term = "sl" not in meta
        for local_iter in range(6):
            pi_in = Pi if (local_iter % 2 == 0) else Pibar
            pi_out = Pibar if (local_iter % 2 == 0) else Pi
            if local_iter == 0 and not has_leaf_term:
                continue
            elif local_iter == 0:
                compute_leaf_initial_wave_step(
                    pi_out, ws, W, S,
                    max_transfer_family, DL_const, Ebar_family, E_family, SL1_const, SL2_const,
                    sp_child1, sp_child2, sp_subtree_start, sp_subtree_end,
                    wave_layout["leaf_species_index"],
                    log_pS_family,
                    family_idx=family_idx,
                )
            else:
                step_input_ws = 0 if local_iter == 1 and not has_leaf_term else None
                compute_wave_step(
                    dts_r if step_input_ws == 0 else pi_in, pi_out, Pibar, ws, W, S,
                    max_transfer_family, DL_const, Ebar_family, E_family, SL1_const, SL2_const,
                    sp_child1, sp_child2, sp_parent, max_ancestor_depth,
                    dts_r,
                    leaf_species_idx=wave_layout["leaf_species_index"],
                    leaf_logp=log_pS_family,
                    family_idx=family_idx,
                    pibar_row_max=uniform_pibar_row_max,
                    store_final_pibar=local_iter == 5,
                    has_leaf_term=has_leaf_term,
                    input_ws=step_input_ws,
                )

    return Pi[wave_layout["root_clade_ids"]], Pi, Pibar, uniform_pibar_row_max
