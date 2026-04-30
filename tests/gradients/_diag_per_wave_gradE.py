"""Diagnostic: verify per-wave grad_E[13] by FD.

For each wave k:
  1. Perturb E[13] by ±eps
  2. Recompute DL_const and terms that depend on E (Pibar+E term)
  3. Re-run wave k's self-loop to convergence (holding DTS_cross fixed)
  4. Compute the contribution to logL change: accumulated_rhs[k] · delta(Pi^*_k)
  5. Compare with the backward's per-wave grad_E[13] contribution

This verifies that v_k · dg_k/dE is computed correctly.
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
from gpurec.core.kernels.wave_step import wave_step_fused

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
        batch_items.append({
            "ccp": ch,
            "leaf_row_index": raw["leaf_row_index"].long(),
            "leaf_col_index": raw["leaf_col_index"].long(),
            "root_clade_id": int(cr["root_clade_id"]),
        })
    sh = {
        "S": int(sr["S"]),
        "s_P_indexes": sr["s_P_indexes"].to(device=device),
        "s_C12_indexes": sr["s_C12_indexes"].to(device=device),
        "Recipients_mat": sr["Recipients_mat"].to(dtype=dtype, device=device),
    }
    if "ancestors_dense" in sr:
        sh["ancestors_dense"] = sr["ancestors_dense"].to(dtype=dtype, device=device)
    S = sh["S"]
    unnorm_row_max = torch.log2(sh["Recipients_mat"]).to(device=device, dtype=dtype).max(dim=-1).values
    ancestors_T = sh["ancestors_dense"].T.to_sparse_coo() if "ancestors_dense" in sh else None
    return batch_items, sh, S, unnorm_row_max, ancestors_T, device, dtype


def reconverge_wave(Pi, Pibar, meta, sp_child1, sp_child2,
                    log_pD, log_pS, mt_squeezed, DL_const, SL1_const, SL2_const,
                    Ebar, E, leaf_row_index, leaf_col_index, S, device, dtype,
                    ancestors_T, dts_r_fixed):
    """Re-run self-loop for one wave to convergence with fixed DTS_cross."""
    ws = meta['start']
    we = meta['end']
    W = meta['W']

    lwt = torch.full((W, S), NEG_INF, device=device, dtype=dtype)
    m = (leaf_row_index >= ws) & (leaf_row_index < we)
    if m.any():
        lwt[leaf_row_index[m] - ws, leaf_col_index[m]] = 0.0
    leaf_wt = log_pS + lwt

    for local_iter in range(5000):
        Pi_W = Pi[ws:we]
        Pibar_W = _compute_Pibar_inline(
            Pi_W, None, mt_squeezed, 'uniform', ancestors_T=ancestors_T
        )
        Pi_new = wave_step_fused(
            Pi_W, Pibar_W, DL_const, Ebar, E, SL1_const, SL2_const,
            sp_child1, sp_child2, leaf_wt, dts_r_fixed,
        )

        significant = Pi_new > -100.0
        if local_iter >= 3:
            if not significant.any() or torch.abs(Pi_new - Pi_W)[significant].max().item() < 1e-10:
                Pi[ws:we] = Pi_new
                Pibar[ws:we] = Pibar_W
                return
        Pi[ws:we] = Pi_new
        Pibar[ws:we] = Pibar_W


def main():
    batch_items, sh, S, unnorm_row_max, ancestors_T, device, dtype = load_data()
    theta = torch.log2(torch.tensor([D, L, T], dtype=dtype, device=device)).unsqueeze(0).expand(S, -1).contiguous()
    log_pS, log_pD, log_pL, _, mt = extract_parameters_uniform(theta, unnorm_row_max, specieswise=True)

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

    # Run forward with tight tolerance
    Po = Pi_wave_forward(
        wave_layout=wl, species_helpers=sh,
        E=E_out["E"], Ebar=E_out["E_bar"],
        E_s1=E_out["E_s1"], E_s2=E_out["E_s2"],
        log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
        transfer_mat=None, max_transfer_mat=mt,
        device=device, dtype=dtype, pibar_mode='uniform',
        local_tolerance=1e-10, local_iters=5000,
    )
    Pi_star = Po['Pi_wave_ordered'].clone()
    Pibar_star = Po['Pibar_wave_ordered'].clone()
    root_clade_ids = wl['root_clade_ids']

    # Run backward (to get the per-wave decomposition)
    pi_bwd = Pi_wave_backward(
        wave_layout=wl,
        Pi_star_wave=Pi_star, Pibar_star_wave=Pibar_star,
        E=E_out["E"], Ebar=E_out["E_bar"],
        E_s1=E_out["E_s1"], E_s2=E_out["E_s2"],
        log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
        max_transfer_mat=mt, species_helpers=sh,
        root_clade_ids_perm=root_clade_ids,
        device=device, dtype=dtype,
        neumann_terms=4, use_pruning=False,
        pibar_mode='uniform', ancestors_T=ancestors_T,
    )
    print(f"Backward grad_E[{S_IDX}] = {pi_bwd['grad_E'][S_IDX]:.8e}")

    # ═══ Replicate backward to get per-wave grad_E contributions ═══
    C = Pi_star.shape[0]
    accumulated_rhs = torch.zeros(C, S, device=device, dtype=dtype)
    for r in root_clade_ids:
        r = int(r)
        root_Pi = Pi_star[r]
        lse = logsumexp2(root_Pi, dim=0)
        accumulated_rhs[r] = -_safe_exp2_ratio(root_Pi, lse)

    per_wave_gE = torch.zeros(S, device=device, dtype=dtype)

    for k in range(K - 1, -1, -1):
        meta = wave_metas[k]
        ws = meta['start']
        we = meta['end']
        W = meta['W']

        rhs_k = accumulated_rhs[ws:we].clone()

        Pi_W_star = Pi_star[ws:we]
        Pibar_W_star = Pibar_star[ws:we]

        lwt = torch.full((W, S), NEG_INF, device=device, dtype=dtype)
        m_leaf = (leaf_row_index >= ws) & (leaf_row_index < we)
        if m_leaf.any():
            lwt[leaf_row_index[m_leaf] - ws, leaf_col_index[m_leaf]] = 0.0
        leaf_wt = log_pS + lwt

        DL_const = 1.0 + log_pD + E_out["E"]
        SL1_const = log_pS + E_out["E_s2"]
        SL2_const = log_pS + E_out["E_s1"]

        if meta['has_splits']:
            with torch.no_grad():
                dts_r = _dts_cross_differentiable(
                    Pi_star.detach(), Pibar_star.detach(), meta,
                    sp_child1, sp_child2, log_pD, log_pS, S, device, dtype,
                )
        else:
            dts_r = None

        ingredients = _self_loop_vjp_precompute(
            Pi_W_star, Pibar_W_star, dts_r,
            mt_squeezed, DL_const, E_out["E_bar"], E_out["E"],
            SL1_const, SL2_const,
            sp_child1, sp_child2, leaf_wt, S,
            'uniform', None, ancestors_T,
        )

        v_k = _gmres_self_loop_solve(
            rhs_k, ingredients, sp_child1, sp_child2, S, W,
            'uniform', None, ancestors_T,
            max_iters=50, tol=1e-8,
        )

        alpha_full = v_k * ingredients['w_L']
        wt = ingredients['w_terms']
        aw0 = alpha_full * wt[0]  # DL_const + Pi → grad_E
        aw2 = alpha_full * wt[2]  # Pibar + E → grad_E

        wave_gE = (aw0 + aw2).sum(dim=0)  # sum over clades in this wave
        per_wave_gE += wave_gE

        # ═══ FD for this wave's grad_E ═══
        # Perturb E[13], recompute DL_const, reconverge this wave, measure Pi change
        # Then: contribution to dL = accumulated_rhs[wave k] · delta(Pi at wave k)
        def wave_with_E_delta(e_delta):
            E_pert = E_out["E"].clone()
            E_pert[S_IDX] += e_delta
            DL_pert = 1.0 + log_pD + E_pert
            SL1_pert = log_pS + E_out["E_s2"]
            SL2_pert = log_pS + E_out["E_s1"]

            Pi_copy = Pi_star.clone()
            Pibar_copy = Pibar_star.clone()

            reconverge_wave(
                Pi_copy, Pibar_copy, meta, sp_child1, sp_child2,
                log_pD, log_pS, mt_squeezed,
                DL_pert, SL1_pert, SL2_pert, E_out["E_bar"], E_pert,
                leaf_row_index, leaf_col_index, S, device, dtype,
                ancestors_T, dts_r,
            )

            delta_Pi = Pi_copy[ws:we] - Pi_star[ws:we]
            return (rhs_k * delta_Pi).sum().item()

        fd_contrib = (wave_with_E_delta(+eps) - wave_with_E_delta(-eps)) / (2 * eps)

        ana_s13 = wave_gE[S_IDX].item()
        err = abs(ana_s13 - fd_contrib) / max(abs(fd_contrib), 1e-12) if abs(fd_contrib) > 1e-15 else 0
        print(f"  Wave {k:2d} (W={W:3d}): ana_gE[13]={ana_s13:+.6e}  FD_contrib={fd_contrib:+.6e}  err={err:.4e}")

        # Cross-clade backprop (same as backward)
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
            neg_inf_col = torch.full((C, 1), NEG_INF, device=device, dtype=dtype)
            Pi_col_pad = torch.cat([Pi_star, neg_inf_col], dim=1)
            Pi_l_s1 = Pi_col_pad[sl][:, sp_child1.long()]
            Pi_l_s2 = Pi_col_pad[sl][:, sp_child2.long()]
            Pi_r_s1 = Pi_col_pad[sr][:, sp_child1.long()]
            Pi_r_s2 = Pi_col_pad[sr][:, sp_child2.long()]

            _pD_s = log_pD.unsqueeze(0)
            _pS_s = log_pS.unsqueeze(0)

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

    print(f"\n  Total per-wave gE[13] = {per_wave_gE[S_IDX]:.8e}")
    print(f"  Backward grad_E[13]  = {pi_bwd['grad_E'][S_IDX]:.8e}")

    # Full FD: perturb E[13], re-run ALL waves, measure logL change
    def full_fd(e_delta):
        E_pert = E_out["E"].clone()
        E_pert[S_IDX] += e_delta
        Po_pert = Pi_wave_forward(
            wave_layout=wl, species_helpers=sh,
            E=E_pert, Ebar=E_out["E_bar"],
            E_s1=E_out["E_s1"], E_s2=E_out["E_s2"],
            log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
            transfer_mat=None, max_transfer_mat=mt,
            device=device, dtype=dtype, pibar_mode='uniform',
            local_tolerance=1e-10, local_iters=5000,
        )
        rp = Po_pert["Pi"][sb["root_clade_ids"], :]
        return -(logsumexp2(rp, dim=-1) - math.log2(S)).sum().item()

    fd_full = (full_fd(+eps) - full_fd(-eps)) / (2 * eps)
    print(f"  FD (full, all waves) = {fd_full:.8e}")
    print(f"  Missing = {fd_full - per_wave_gE[S_IDX]:.8e}")


if __name__ == "__main__":
    main()
