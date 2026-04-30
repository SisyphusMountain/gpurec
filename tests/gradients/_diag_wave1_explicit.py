"""Diagnostic: explicit Jacobian + direct solve for wave 1.

Build the full (W*S × W*S) Jacobian for wave 1 by FD, compute
v_k = (I-J^T)^{-1} * rhs via direct solve, and compare with GMRES.
"""
import math, torch
from pathlib import Path

from gpurec.core.preprocess_cpp import _load_extension
from gpurec.core.extract_parameters import extract_parameters_uniform
from gpurec.core.likelihood import E_fixed_point
from gpurec.core.forward import Pi_wave_forward, _compute_Pibar_inline, _compute_dts_cross
from gpurec.core.backward import (
    Pi_wave_backward,
    _self_loop_vjp_precompute,
    _self_loop_Jt_apply,
    _gmres_self_loop_solve,
    _dts_cross_differentiable,
    _self_loop_differentiable,
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
eps_jac = 1e-6
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


def main():
    batch_items, sh, S, unnorm_row_max, ancestors_T, device, dtype = load_data()
    theta = torch.log2(torch.tensor([D, L, T], dtype=dtype, device=device)).unsqueeze(0).expand(S, -1).contiguous()
    log_pS, log_pD, log_pL, _, mt = extract_parameters_uniform(theta, unnorm_row_max, specieswise=True)
    mt_squeezed = mt.squeeze(-1) if mt.ndim > 1 else mt

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

    # Forward
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

    # Focus on WAVE 1
    WAVE_K = 1
    meta = wave_metas[WAVE_K]
    ws = meta['start']
    we = meta['end']
    W = meta['W']
    print(f"Wave {WAVE_K}: ws={ws}, we={we}, W={W}")
    n = W * S  # total dimension of the Jacobian

    # Compute DTS_cross (fixed during self-loop)
    if meta['has_splits']:
        with torch.no_grad():
            dts_r = _dts_cross_differentiable(
                Pi_star.detach(), Pibar_star.detach(), meta,
                sp_child1, sp_child2, log_pD, log_pS, S, device, dtype,
            )
    else:
        dts_r = None

    # Leaf weights
    lwt = torch.full((W, S), NEG_INF, device=device, dtype=dtype)
    m_leaf = (leaf_row_index >= ws) & (leaf_row_index < we)
    if m_leaf.any():
        lwt[leaf_row_index[m_leaf] - ws, leaf_col_index[m_leaf]] = 0.0
    leaf_wt = log_pS + lwt

    DL_const = 1.0 + log_pD + E_out["E"]
    SL1_const = log_pS + E_out["E_s2"]
    SL2_const = log_pS + E_out["E_s1"]

    Pi_W_star = Pi_star[ws:we]
    Pibar_W_star = Pibar_star[ws:we]

    # ═══ Build explicit Jacobian by FD ═══
    print(f"Building {n}×{n} Jacobian by FD...")
    J_fd = torch.zeros(n, n, device=device, dtype=dtype)

    def one_step(Pi_flat):
        Pi_W = Pi_flat.reshape(W, S)
        Pibar_W = _compute_Pibar_inline(Pi_W, None, mt_squeezed, 'uniform', ancestors_T=ancestors_T)
        Pi_new = wave_step_fused(
            Pi_W, Pibar_W, DL_const, E_out["E_bar"], E_out["E"],
            SL1_const, SL2_const, sp_child1, sp_child2, leaf_wt, dts_r,
        )
        return Pi_new.reshape(-1)

    Pi_base_flat = Pi_W_star.reshape(-1)
    g_base = one_step(Pi_base_flat)

    for j in range(n):
        Pi_pert = Pi_base_flat.clone()
        Pi_pert[j] += eps_jac
        g_pert = one_step(Pi_pert)
        J_fd[:, j] = (g_pert - g_base) / eps_jac

    # ═══ Analytical J^T application ═══
    ingredients = _self_loop_vjp_precompute(
        Pi_W_star, Pibar_W_star, dts_r,
        mt_squeezed, DL_const, E_out["E_bar"], E_out["E"],
        SL1_const, SL2_const,
        sp_child1, sp_child2, leaf_wt, S,
        'uniform', None, ancestors_T,
    )

    # Compare J^T analytical vs FD
    test_v = torch.randn(W, S, device=device, dtype=dtype)
    Jt_ana = _self_loop_Jt_apply(
        test_v, ingredients, sp_child1, sp_child2, S, W,
        'uniform', None, ancestors_T,
    )
    Jt_fd = (J_fd.T @ test_v.reshape(-1)).reshape(W, S)
    jt_err = (Jt_ana - Jt_fd).abs().max() / Jt_fd.abs().max()
    print(f"J^T analytical vs FD: max_rel_err = {jt_err:.4e}")

    # ═══ Compute accumulated_rhs for wave 1 (replicate backward) ═══
    C = Pi_star.shape[0]
    accumulated_rhs = torch.zeros(C, S, device=device, dtype=dtype)
    root_clade_ids = wl['root_clade_ids']
    for r in root_clade_ids:
        r = int(r)
        root_Pi = Pi_star[r]
        lse = logsumexp2(root_Pi, dim=0)
        accumulated_rhs[r] = -_safe_exp2_ratio(root_Pi, lse)

    # Process waves K-1 down to WAVE_K+1, propagating accumulated_rhs
    for k in range(K - 1, WAVE_K, -1):
        mk = wave_metas[k]
        wsk = mk['start']
        wek = mk['end']
        Wk = mk['W']

        rhs_k = accumulated_rhs[wsk:wek].clone()
        Pi_Wk = Pi_star[wsk:wek]
        Pibar_Wk = Pibar_star[wsk:wek]

        lwt_k = torch.full((Wk, S), NEG_INF, device=device, dtype=dtype)
        m_k = (leaf_row_index >= wsk) & (leaf_row_index < wek)
        if m_k.any():
            lwt_k[leaf_row_index[m_k] - wsk, leaf_col_index[m_k]] = 0.0
        leaf_wt_k = log_pS + lwt_k

        if mk['has_splits']:
            with torch.no_grad():
                dts_r_k = _dts_cross_differentiable(
                    Pi_star.detach(), Pibar_star.detach(), mk,
                    sp_child1, sp_child2, log_pD, log_pS, S, device, dtype,
                )
        else:
            dts_r_k = None

        ing_k = _self_loop_vjp_precompute(
            Pi_Wk, Pibar_Wk, dts_r_k,
            mt_squeezed, DL_const, E_out["E_bar"], E_out["E"],
            SL1_const, SL2_const,
            sp_child1, sp_child2, leaf_wt_k, S,
            'uniform', None, ancestors_T,
        )

        v_k_solve = _gmres_self_loop_solve(
            rhs_k, ing_k, sp_child1, sp_child2, S, Wk,
            'uniform', None, ancestors_T,
            max_iters=50, tol=1e-8,
        )

        # Cross-clade backprop
        if mk['has_splits'] and dts_r_k is not None:
            sl = mk['sl']
            sr = mk['sr']
            wlsp = mk['log_split_probs']
            reduce_idx = mk['reduce_idx']
            n_ws = sl.shape[0]

            Pi_l = Pi_star[sl]; Pi_r = Pi_star[sr]
            Pibar_l = Pibar_star[sl]; Pibar_r = Pibar_star[sr]
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
            Pi_parent = Pi_Wk[reduce_idx]
            combined = wlsp + DTS_5
            v_k_parent = v_k_solve[reduce_idx]
            grad_DTS_5 = v_k_parent.unsqueeze(0) * _safe_exp2_ratio(combined, Pi_parent.unsqueeze(0))

            sc1 = sp_child1.long(); sc2 = sp_child2.long()
            valid1 = sc1 < S; valid2 = sc2 < S
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
                nz_ch = all_children[nz]
                u = all_pibar_grad[nz]
                Pi_ch = Pi_star[nz_ch]
                Pi_max_p = Pi_ch.max(dim=1, keepdim=True).values
                p_prime = torch.exp2(Pi_ch - Pi_max_p)
                anc_sum = p_prime @ ancestors_T
                denom = p_prime.sum(dim=1, keepdim=True) - anc_sum
                denom_safe = torch.where(denom > 0, denom, torch.ones_like(denom))
                u_d = torch.where(denom > 0, u / denom_safe, torch.zeros_like(u))
                A = u_d.sum(dim=1, keepdim=True)
                correction = (ancestors_T @ u_d.T).T
                pi_from_pibar = p_prime * (A - correction)
                accumulated_rhs.index_add_(0, nz_ch, pi_from_pibar)

    rhs_wave1 = accumulated_rhs[ws:we].clone()
    print(f"accumulated_rhs[wave1, :, 13] = {rhs_wave1[:, S_IDX]}")
    print(f"|rhs|_max = {rhs_wave1.abs().max():.6e}")

    # ═══ GMRES solve for wave 1 ═══
    v_gmres = _gmres_self_loop_solve(
        rhs_wave1, ingredients, sp_child1, sp_child2, S, W,
        'uniform', None, ancestors_T,
        max_iters=50, tol=1e-8,
    )

    # ═══ Direct solve using explicit Jacobian ═══
    I_n = torch.eye(n, device=device, dtype=dtype)
    A_mat = I_n - J_fd.T
    rhs_flat = rhs_wave1.reshape(-1)
    v_direct = torch.linalg.solve(A_mat, rhs_flat).reshape(W, S)

    # Compare GMRES vs direct
    gmres_direct_err = (v_gmres - v_direct).abs().max() / v_direct.abs().max()
    print(f"\nGMRES vs direct solve: max_rel_err = {gmres_direct_err:.4e}")

    # Check GMRES residual
    Jt_v = _self_loop_Jt_apply(
        v_gmres, ingredients, sp_child1, sp_child2, S, W,
        'uniform', None, ancestors_T,
    )
    resid_gmres = (v_gmres - Jt_v - rhs_wave1).norm() / rhs_wave1.norm()
    print(f"GMRES residual (I-J^T)v - rhs: {resid_gmres:.4e}")

    # Check direct residual
    resid_direct = (A_mat @ v_direct.reshape(-1) - rhs_flat).norm() / rhs_flat.norm()
    print(f"Direct residual: {resid_direct:.4e}")

    # ═══ Compute grad_E[13] from both ═══
    wt = ingredients['w_terms']
    w_L = ingredients['w_L']
    dg_dE = w_L * (wt[0] + wt[2])  # [W, S]

    gE_gmres = (v_gmres * dg_dE).sum(dim=0)[S_IDX].item()
    gE_direct = (v_direct * dg_dE).sum(dim=0)[S_IDX].item()

    # Compare v_k at species 13
    print(f"\nv_gmres[:, 13] = {v_gmres[:, S_IDX]}")
    print(f"v_direct[:, 13] = {v_direct[:, S_IDX]}")

    print(f"\ngrad_E[13] from GMRES:  {gE_gmres:.8e}")
    print(f"grad_E[13] from direct: {gE_direct:.8e}")

    # ═══ Also compute the FD-based grad_E contribution ═══
    def reconverge_wave1(E_pert):
        Pi_W_copy = Pi_W_star.clone()
        DL_pert = 1.0 + log_pD + E_pert
        for it in range(5000):
            Pibar_W = _compute_Pibar_inline(Pi_W_copy, None, mt_squeezed, 'uniform', ancestors_T=ancestors_T)
            Pi_new = wave_step_fused(
                Pi_W_copy, Pibar_W, DL_pert, E_out["E_bar"], E_pert,
                SL1_const, SL2_const, sp_child1, sp_child2, leaf_wt, dts_r,
            )
            significant = Pi_new > -100.0
            if it >= 3 and (not significant.any() or torch.abs(Pi_new - Pi_W_copy)[significant].max().item() < 1e-10):
                return Pi_new
            Pi_W_copy = Pi_new
        return Pi_new

    E_p = E_out["E"].clone(); E_p[S_IDX] += 1e-4
    E_m = E_out["E"].clone(); E_m[S_IDX] -= 1e-4
    Pi_p = reconverge_wave1(E_p)
    Pi_m = reconverge_wave1(E_m)
    delta_Pi = (Pi_p - Pi_m) / (2 * 1e-4)

    fd_gE = (rhs_wave1 * delta_Pi).sum().item()
    print(f"FD grad_E contrib (wave 1): {fd_gE:.8e}")

    # Per-clade breakdown
    print(f"\nPer-clade breakdown for species 13:")
    for c in range(W):
        gE_c_ana = (v_gmres[c, :] * dg_dE[c, :]).sum().item()
        gE_c_direct = (v_direct[c, :] * dg_dE[c, :]).sum().item()
        fd_c = (rhs_wave1[c, :] * delta_Pi[c, :]).sum().item()
        print(f"  Clade {ws+c}: ana={gE_c_ana:.6e}  direct={gE_c_direct:.6e}  FD={fd_c:.6e}")

    # Check: does delta_Pi match (I-J)^{-1} * dg/dE ?
    dg_dE_vec = torch.zeros(W, S, device=device, dtype=dtype)
    dg_dE_vec[:, S_IDX] = dg_dE[:, S_IDX]  # Only species 13
    predicted_delta_Pi = torch.linalg.solve(I_n - J_fd, dg_dE_vec.reshape(-1)).reshape(W, S)
    pred_err = (predicted_delta_Pi - delta_Pi).abs().max() / delta_Pi.abs().max()
    print(f"\n(I-J)^{{-1}} * dg/dE vs FD delta_Pi: max_rel_err = {pred_err:.4e}")


if __name__ == "__main__":
    main()
