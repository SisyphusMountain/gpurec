"""Diagnostic: verify accumulated_rhs at each wave by FD.

For each wave k, pick a clade c and species s=13. Perturb Pi^*[c, 13],
then re-run all waves k+1..K-1 from scratch. Compare FD with accumulated_rhs
from the backward pass.

This isolates whether the cross-clade backpropagation is correct.
"""
import math, torch
from pathlib import Path

from gpurec.core.preprocess_cpp import _load_extension
from gpurec.core.extract_parameters import extract_parameters_uniform
from gpurec.core.likelihood import E_fixed_point, compute_log_likelihood
from gpurec.core.forward import Pi_wave_forward, _compute_Pibar_inline, _compute_dts_cross
from gpurec.core.backward import (
    Pi_wave_backward,
    _self_loop_vjp_precompute,
    _self_loop_Jt_apply,
    _gmres_self_loop_solve,
    _dts_cross_differentiable,
)
from gpurec.core.scheduling import compute_clade_waves
from gpurec.core.batching import collate_gene_families, collate_wave, build_wave_layout
from gpurec.core.log2_utils import logsumexp2
from gpurec.core._helpers import _safe_exp2_ratio

_INV = 1.0 / math.log(2.0)
D, L, T = 0.05, 0.05, 0.05
FI = 1
S_IDX = 13
eps = 1e-4
NEG_INF = float("-inf")


def load_data():
    device = torch.device("cuda")
    dtype = torch.float64
    ext = _load_extension()
    root = Path(__file__).resolve().parent.parent
    data_dir = root / "data" / "test_trees_20"
    sp_path = str(data_dir / "sp.nwk")
    gene_paths = sorted(data_dir.glob("g_*.nwk"))[:3]
    batch_items = []
    sr = None
    for gp in gene_paths:
        raw = ext.preprocess(sp_path, [str(gp)])
        if sr is None:
            sr = raw["species"]
        cr = raw["ccp"]
        ch = {
            "split_leftrights_sorted": cr["split_leftrights_sorted"],
            "log_split_probs_sorted": cr["log_split_probs_sorted"].to(dtype=dtype) * _INV,
            "seg_parent_ids": cr["seg_parent_ids"],
            "ptr_ge2": cr["ptr_ge2"],
            "num_segs_ge2": int(cr["num_segs_ge2"]),
            "num_segs_eq1": int(cr["num_segs_eq1"]),
            "end_rows_ge2": int(cr["end_rows_ge2"]),
            "C": int(cr["C"]),
            "N_splits": int(cr["N_splits"]),
        }
        if "split_parents_sorted" in cr:
            ch["split_parents_sorted"] = cr["split_parents_sorted"]
        batch_items.append(
            {
                "ccp": ch,
                "leaf_row_index": raw["leaf_row_index"].long(),
                "leaf_col_index": raw["leaf_col_index"].long(),
                "root_clade_id": int(cr["root_clade_id"]),
            }
        )
    sh = {
        "S": int(sr["S"]),
        "s_P_indexes": sr["s_P_indexes"].to(device=device),
        "s_C12_indexes": sr["s_C12_indexes"].to(device=device),
        "Recipients_mat": sr["Recipients_mat"].to(dtype=dtype, device=device),
    }
    if "ancestors_dense" in sr:
        sh["ancestors_dense"] = sr["ancestors_dense"].to(dtype=dtype, device=device)
    S = sh["S"]
    unnorm_row_max = (
        torch.log2(sh["Recipients_mat"]).to(device=device, dtype=dtype).max(dim=-1).values
    )
    ancestors_T = (
        sh["ancestors_dense"].T.to_sparse_coo() if "ancestors_dense" in sh else None
    )
    return batch_items, sh, S, unnorm_row_max, ancestors_T, device, dtype


def rerun_forward_from_wave(Pi, Pibar, wave_metas, from_wave, sp_child1, sp_child2,
                            log_pD, log_pS, mt_squeezed, DL_const, SL1_const, SL2_const,
                            Ebar, E, leaf_row_index, leaf_col_index, S, device, dtype,
                            ancestors_T, log_pS_scalar):
    """Re-run the forward from wave `from_wave` to K-1, updating Pi/Pibar in place."""
    K = len(wave_metas)
    leaf_term_global = None  # We'll compute per-wave

    for wi in range(from_wave, K):
        meta = wave_metas[wi]
        ws = meta['start']
        we = meta['end']
        W = meta['W']

        # Compute DTS_cross
        if meta['has_splits']:
            dts_r = _compute_dts_cross(
                Pi, Pibar, meta, sp_child1, sp_child2,
                log_pD, log_pS, S, device, dtype,
            )
        else:
            dts_r = None

        # Compute leaf weights
        lwt = torch.full((W, S), NEG_INF, device=device, dtype=dtype)
        m = (leaf_row_index >= ws) & (leaf_row_index < we)
        if m.any():
            lwt[leaf_row_index[m] - ws, leaf_col_index[m]] = 0.0
        leaf_wt = log_pS_scalar + lwt

        # Self-loop to convergence
        from gpurec.core.kernels.wave_step import wave_step_fused
        for local_iter in range(200):
            Pi_W = Pi[ws:we]
            Pibar_W = _compute_Pibar_inline(
                Pi_W, None, mt_squeezed, 'uniform', ancestors_T=ancestors_T
            )

            Pi_new = wave_step_fused(
                Pi_W, Pibar_W,
                DL_const, Ebar, E, SL1_const, SL2_const,
                sp_child1, sp_child2, leaf_wt, dts_r,
            )

            significant = Pi_new > -100.0
            if local_iter >= 3:
                if not significant.any() or torch.abs(Pi_new - Pi_W)[significant].max().item() < 1e-10:
                    Pi[ws:we] = Pi_new
                    Pibar[ws:we] = Pibar_W
                    break

            Pi[ws:we] = Pi_new
            Pibar[ws:we] = Pibar_W


def compute_logL(Pi, E, root_clade_ids, S):
    rp = Pi[root_clade_ids, :]
    return -(logsumexp2(rp, dim=-1) - math.log2(S)).sum().item()


def main():
    batch_items, sh, S, unnorm_row_max, ancestors_T, device, dtype = load_data()
    theta = (
        torch.log2(torch.tensor([D, L, T], dtype=dtype, device=device))
        .unsqueeze(0)
        .expand(S, -1)
        .contiguous()
    )
    log_pS, log_pD, log_pL, _, mt = extract_parameters_uniform(
        theta, unnorm_row_max, specieswise=True
    )

    E_out = E_fixed_point(
        species_helpers=sh, log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
        transfer_mat=None, max_transfer_mat=mt,
        max_iters=2000, tolerance=1e-10, warm_start_E=None,
        dtype=dtype, device=device, pibar_mode='uniform', ancestors_T=ancestors_T,
    )

    bi = batch_items[FI]
    sb = collate_gene_families([bi], dtype=dtype, device=device)
    w, p = compute_clade_waves(bi["ccp"])
    cw = collate_wave([w], [0])
    wl = build_wave_layout(
        waves=cw, phases=p, ccp_helpers=sb["ccp"],
        leaf_row_index=sb["leaf_row_index"], leaf_col_index=sb["leaf_col_index"],
        root_clade_ids=sb["root_clade_ids"], device=device, dtype=dtype,
    )

    wave_metas = wl['wave_metas']
    K = len(wave_metas)
    leaf_row_index = wl['leaf_row_index']
    leaf_col_index = wl['leaf_col_index']

    # Run forward
    Po = Pi_wave_forward(
        wave_layout=wl, species_helpers=sh,
        E=E_out["E"], Ebar=E_out["E_bar"],
        E_s1=E_out["E_s1"], E_s2=E_out["E_s2"],
        log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
        transfer_mat=None, max_transfer_mat=mt,
        device=device, dtype=dtype, pibar_mode='uniform',
        local_tolerance=1e-10, local_iters=5000,
    )
    Pi_star = Po['Pi_wave_ordered']
    Pibar_star = Po['Pibar_wave_ordered']
    root_clade_ids = sb['root_clade_ids']
    perm = wl['perm']

    # Setup species children
    sp_P_idx = sh['s_P_indexes']
    sp_c12_idx = sh['s_C12_indexes']
    p_cpu = sp_P_idx.cpu().long()
    c_cpu = sp_c12_idx.cpu().long()
    mask_c1 = p_cpu < S
    sp_child1 = torch.full((S,), S, dtype=torch.long, device=device)
    sp_child2 = torch.full((S,), S, dtype=torch.long, device=device)
    sp_child1[p_cpu[mask_c1]] = c_cpu[mask_c1].to(device)
    sp_child2[p_cpu[~mask_c1] - S] = c_cpu[~mask_c1].to(device)

    mt_squeezed = mt.squeeze(-1) if mt.ndim > 1 else mt
    _pD = log_pD
    _pS = log_pS
    DL_const = 1.0 + _pD + E_out["E"]
    SL1_const = _pS + E_out["E_s2"]
    SL2_const = _pS + E_out["E_s1"]

    # Run backward
    pi_bwd = Pi_wave_backward(
        wave_layout=wl,
        Pi_star_wave=Pi_star,
        Pibar_star_wave=Pibar_star,
        E=E_out["E"], Ebar=E_out["E_bar"],
        E_s1=E_out["E_s1"], E_s2=E_out["E_s2"],
        log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
        max_transfer_mat=mt,
        species_helpers=sh,
        root_clade_ids_perm=wl["root_clade_ids"],
        device=device, dtype=dtype,
        neumann_terms=4, use_pruning=False,
        pibar_mode='uniform', ancestors_T=ancestors_T,
    )

    # ═══ Now replicate the backward to capture accumulated_rhs ═══
    C = Pi_star.shape[0]
    accumulated_rhs = torch.zeros(C, S, device=device, dtype=dtype)
    for r in wl["root_clade_ids"]:
        r = int(r)
        root_Pi = Pi_star[r]
        lse = logsumexp2(root_Pi, dim=0)
        accumulated_rhs[r] = -_safe_exp2_ratio(root_Pi, lse)

    # Save accumulated_rhs snapshots BEFORE each wave is processed
    rhs_snapshots = {}  # wave_idx -> accumulated_rhs[ws:we].clone()

    for k in range(K - 1, -1, -1):
        meta = wave_metas[k]
        ws = meta['start']
        we = meta['end']
        W = meta['W']

        rhs_snapshots[k] = accumulated_rhs[ws:we].clone()

        rhs_k = accumulated_rhs[ws:we].clone()
        Pi_W_star = Pi_star[ws:we].detach()

        # Compute leaf weights (same as backward)
        lwt = torch.full((W, S), NEG_INF, device=device, dtype=dtype)
        m_leaf = (leaf_row_index >= ws) & (leaf_row_index < we)
        if m_leaf.any():
            lwt[leaf_row_index[m_leaf] - ws, leaf_col_index[m_leaf]] = 0.0
        leaf_wt = log_pS + lwt

        if meta['has_splits']:
            with torch.no_grad():
                dts_r = _dts_cross_differentiable(
                    Pi_star.detach(), Pibar_star.detach(), meta,
                    sp_child1, sp_child2,
                    log_pD, log_pS, S, device, dtype,
                )
        else:
            dts_r = None

        Pibar_W_star = Pibar_star[ws:we]
        mt_w = mt_squeezed
        DL_w = DL_const
        Ebar_w = E_out["E_bar"]
        E_w = E_out["E"]
        SL1_w = SL1_const
        SL2_w = SL2_const

        ingredients = _self_loop_vjp_precompute(
            Pi_W_star, Pibar_W_star, dts_r,
            mt_w, DL_w, Ebar_w, E_w, SL1_w, SL2_w,
            sp_child1, sp_child2, leaf_wt, S,
            'uniform', None, ancestors_T,
        )

        v_k = _gmres_self_loop_solve(
            rhs_k, ingredients, sp_child1, sp_child2, S, W,
            'uniform', None, ancestors_T,
            max_iters=50, tol=1e-8,
        )

        # Cross-clade backprop
        if meta['has_splits'] and dts_r is not None:
            sl = meta['sl']
            sr = meta['sr']
            wlsp = meta['log_split_probs']
            reduce_idx = meta['reduce_idx']
            n_ws = sl.shape[0]

            Pi_l = Pi_star[sl]
            Pi_r = Pi_star[sr]
            Pibar_l = Pibar_star[sl]
            Pibar_r = Pibar_star[sr]
            neg_inf_col = torch.full((Pi_star.shape[0], 1), NEG_INF, device=device, dtype=dtype)
            Pi_col_pad = torch.cat([Pi_star, neg_inf_col], dim=1)
            Pi_l_s1 = Pi_col_pad[sl][:, sp_child1.long()]
            Pi_l_s2 = Pi_col_pad[sl][:, sp_child2.long()]
            Pi_r_s1 = Pi_col_pad[sr][:, sp_child1.long()]
            Pi_r_s2 = Pi_col_pad[sr][:, sp_child2.long()]

            # For single-family: log_pD is [S], expand to [1, S] for broadcast with [n_ws, S]
            _pD_s = log_pD.unsqueeze(0) if log_pD.ndim == 1 else log_pD
            _pS_s = log_pS.unsqueeze(0) if log_pS.ndim == 1 else log_pS

            DTS_5 = torch.stack([
                _pD_s + Pi_l + Pi_r,
                Pi_l + Pibar_r,
                Pi_r + Pibar_l,
                _pS_s + Pi_l_s1 + Pi_r_s2,
                _pS_s + Pi_r_s1 + Pi_l_s2,
            ], dim=0)

            Pi_parent = Pi_W_star[reduce_idx]
            combined = wlsp + DTS_5
            v_k_parent = v_k[reduce_idx]
            grad_DTS_5 = v_k_parent.unsqueeze(0) * _safe_exp2_ratio(
                combined, Pi_parent.unsqueeze(0))

            sc1 = sp_child1.long()
            sc2 = sp_child2.long()
            valid1 = sc1 < S
            valid2 = sc2 < S

            grad_Pi_l = grad_DTS_5[0] + grad_DTS_5[1]
            grad_Pi_r = grad_DTS_5[0] + grad_DTS_5[2]
            grad_Pibar_l = grad_DTS_5[2]
            grad_Pibar_r = grad_DTS_5[1]

            if valid1.any():
                idx1 = sc1[valid1]
                grad_Pi_l.scatter_add_(1, idx1.unsqueeze(0).expand(n_ws, -1), grad_DTS_5[3][:, valid1])
                grad_Pi_r.scatter_add_(1, idx1.unsqueeze(0).expand(n_ws, -1), grad_DTS_5[4][:, valid1])
            if valid2.any():
                idx2 = sc2[valid2]
                grad_Pi_r.scatter_add_(1, idx2.unsqueeze(0).expand(n_ws, -1), grad_DTS_5[3][:, valid2])
                grad_Pi_l.scatter_add_(1, idx2.unsqueeze(0).expand(n_ws, -1), grad_DTS_5[4][:, valid2])

            accumulated_rhs.index_add_(0, sl, grad_Pi_l)
            accumulated_rhs.index_add_(0, sr, grad_Pi_r)

            all_children = torch.cat([sl, sr])
            all_pibar_grad = torch.cat([grad_Pibar_l, grad_Pibar_r])

            nz = all_pibar_grad.abs().sum(dim=1) > 0
            if nz.any():
                nz_children = all_children[nz]
                u = all_pibar_grad[nz]
                Pi_ch = Pi_star[nz_children]
                Pi_max_p = Pi_ch.max(dim=1, keepdim=True).values
                p_prime = torch.exp2(Pi_ch - Pi_max_p)

                anc_sum = p_prime @ ancestors_T
                denom = p_prime.sum(dim=1, keepdim=True) - anc_sum
                denom_safe = torch.where(denom > 0, denom, torch.ones_like(denom))
                u_d = torch.where(denom > 0, u / denom_safe, torch.zeros_like(u))
                A = u_d.sum(dim=1, keepdim=True)
                correction = (ancestors_T @ u_d.T).T
                pi_from_pibar = p_prime * (A - correction)

                accumulated_rhs.index_add_(0, nz_children, pi_from_pibar)

    # ═══ Now verify accumulated_rhs by FD ═══
    print(f"Family {FI} (C={bi['ccp']['C']}), K={K} waves")
    print(f"{'='*70}")

    anc_dense = sh['ancestors_dense'].to(device=device, dtype=dtype)
    anc_T_dense = anc_dense.T.to_sparse_coo()

    base_logL = compute_logL(Po['Pi'], E_out['E'], root_clade_ids, S)

    for k_check in range(K - 1, -1, -1):
        meta = wave_metas[k_check]
        ws = meta['start']
        we = meta['end']
        W = meta['W']

        # Pick the clade in this wave with the largest |accumulated_rhs[:, S_IDX]|
        rhs_wave = rhs_snapshots[k_check]
        c_local = rhs_wave[:, S_IDX].abs().argmax().item()
        c_global = ws + c_local
        rhs_val = rhs_wave[c_local, S_IDX].item()

        # FD: perturb Pi^*[c_global, S_IDX], re-run from wave k_check+1
        def fd_perturb(delta):
            Pi_copy = Pi_star.clone()
            Pibar_copy = Pibar_star.clone()
            Pi_copy[c_global, S_IDX] += delta

            # Recompute Pibar at perturbed clade
            Pi_ch_row = Pi_copy[c_global:c_global+1]
            Pibar_new = _compute_Pibar_inline(
                Pi_ch_row, None, mt_squeezed, 'uniform', ancestors_T=anc_T_dense
            )
            Pibar_copy[c_global] = Pibar_new[0]

            # Re-run forward from wave k_check+1 to K-1
            rerun_forward_from_wave(
                Pi_copy, Pibar_copy, wave_metas, k_check + 1,
                sp_child1, sp_child2, log_pD, log_pS, mt_squeezed,
                DL_const, SL1_const, SL2_const, E_out["E_bar"], E_out["E"],
                leaf_row_index, leaf_col_index, S, device, dtype,
                anc_T_dense, log_pS,
            )

            # Compute logL in original clade order
            Pi_orig = Pi_copy[wl['perm']]
            return compute_logL(Pi_orig, E_out['E'], root_clade_ids, S)

        logL_plus = fd_perturb(+eps)
        logL_minus = fd_perturb(-eps)
        fd_val = (logL_plus - logL_minus) / (2 * eps)

        err = abs(rhs_val - fd_val) / max(abs(fd_val), 1e-12)
        marker = "***" if err > 0.05 else ""
        print(f"  Wave {k_check:2d} (W={W:3d}): clade {c_global:3d} (local {c_local:2d}) "
              f"rhs={rhs_val:+.6e}  FD={fd_val:+.6e}  err={err:.4e} {marker}")


if __name__ == "__main__":
    main()
