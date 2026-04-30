"""Diagnostic: decompose J^T error for wave 1 into its components.

Compare analytical vs FD for each Jacobian component separately:
  - Diagonal (DL+Pi, Pi+Ebar terms)
  - Pibar path (Pibar+E term → dPibar/dPi chain rule)
  - Speciation (SL1+Pi[s1], SL2+Pi[s2] scatter terms)
"""
import math, torch
from pathlib import Path

from gpurec.core.preprocess_cpp import _load_extension
from gpurec.core.extract_parameters import extract_parameters_uniform
from gpurec.core.likelihood import E_fixed_point
from gpurec.core.forward import Pi_wave_forward, _compute_Pibar_inline
from gpurec.core.backward import (
    _self_loop_vjp_precompute,
    _self_loop_Jt_apply,
    _dts_cross_differentiable,
)
from gpurec.core.scheduling import compute_clade_waves
from gpurec.core.batching import collate_gene_families, collate_wave, build_wave_layout
from gpurec.core.log2_utils import logsumexp2, logaddexp2
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
    n = W * S
    print(f"Wave {WAVE_K}: ws={ws}, we={we}, W={W}, n={n}")

    # DTS cross
    if meta['has_splits']:
        with torch.no_grad():
            dts_r = _dts_cross_differentiable(
                Pi_star.detach(), Pibar_star.detach(), meta,
                sp_child1, sp_child2, log_pD, log_pS, S, device, dtype,
            )
    else:
        dts_r = None

    lwt = torch.full((W, S), NEG_INF, device=device, dtype=dtype)
    m_leaf = (leaf_row_index >= ws) & (leaf_row_index < we)
    if m_leaf.any():
        lwt[leaf_row_index[m_leaf] - ws, leaf_col_index[m_leaf]] = 0.0
    leaf_wt = log_pS + lwt

    DL_const = 1.0 + log_pD + E_out["E"]
    SL1_const = log_pS + E_out["E_s2"]
    SL2_const = log_pS + E_out["E_s1"]

    Pi_W = Pi_star[ws:we]
    Pibar_W = Pibar_star[ws:we]

    # ═══ Manually compute terms and weights ═══
    def _expand(t):
        return t.unsqueeze(0).expand(W, -1) if t.ndim == 1 else t

    DL = _expand(DL_const)
    Ebar_exp = _expand(E_out["E_bar"])
    E_exp = _expand(E_out["E"])
    SL1 = _expand(SL1_const)
    SL2 = _expand(SL2_const)

    Pi_max = Pi_W.max(dim=1, keepdim=True).values
    p_prime = torch.exp2(Pi_W - Pi_max)

    Pi_pad = torch.cat([Pi_W, torch.full((W, 1), NEG_INF, device=device, dtype=dtype)], dim=1)
    Pi_s1 = Pi_pad[:, sp_child1.long()]
    Pi_s2 = Pi_pad[:, sp_child2.long()]

    terms = torch.stack([
        DL + Pi_W,
        Pi_W + Ebar_exp,
        Pibar_W + E_exp,
        SL1 + Pi_s1,
        SL2 + Pi_s2,
        leaf_wt,
    ], dim=0)

    DTS_L = logsumexp2(terms, dim=0)
    Pi_new = logaddexp2(dts_r, DTS_L)
    w_L = _safe_exp2_ratio(DTS_L, Pi_new)
    w_terms = _safe_exp2_ratio(terms, DTS_L.unsqueeze(0))

    print(f"\nw_L stats: min={w_L.min():.4e}, max={w_L.max():.4e}, mean={w_L.mean():.4e}")
    print(f"w_terms[0] (DL+Pi): max={w_terms[0].max():.4e}")
    print(f"w_terms[1] (Pi+Ebar): max={w_terms[1].max():.4e}")
    print(f"w_terms[2] (Pibar+E): max={w_terms[2].max():.4e}")
    print(f"w_terms[3] (SL1+Pi_s1): max={w_terms[3].max():.4e}")
    print(f"w_terms[4] (SL2+Pi_s2): max={w_terms[4].max():.4e}")
    print(f"w_terms[5] (leaf): max={w_terms[5].max():.4e}")

    # ═══ Build FD Jacobian ═══
    def one_step(Pi_flat):
        Pi_W_local = Pi_flat.reshape(W, S)
        Pibar_local = _compute_Pibar_inline(Pi_W_local, None, mt_squeezed, 'uniform', ancestors_T=ancestors_T)
        Pi_new_local = wave_step_fused(
            Pi_W_local, Pibar_local, DL_const, E_out["E_bar"], E_out["E"],
            SL1_const, SL2_const, sp_child1, sp_child2, leaf_wt, dts_r,
        )
        return Pi_new_local.reshape(-1)

    Pi_base = Pi_W.reshape(-1)
    g_base = one_step(Pi_base)

    # Build J column by column (FD)
    print("\nBuilding FD Jacobian...")
    J_fd = torch.zeros(n, n, device=device, dtype=dtype)
    for j in range(n):
        Pi_pert = Pi_base.clone()
        Pi_pert[j] += eps_jac
        g_pert = one_step(Pi_pert)
        J_fd[:, j] = (g_pert - g_base) / eps_jac

    # ═══ Build analytical Jacobian component by component ═══
    # Component 1: Diagonal (terms 0, 1)
    # J_diag[c*S+s, c*S+s'] = w_L[c,s] * (wt0[c,s] + wt1[c,s]) * delta(s=s')
    J_diag = torch.zeros(n, n, device=device, dtype=dtype)
    for c in range(W):
        for s in range(S):
            J_diag[c*S+s, c*S+s] = (w_L[c,s] * (w_terms[0][c,s] + w_terms[1][c,s])).item()

    # Component 2: Pibar path (term 2)
    # J_pibar[c*S+s, c*S+s'] = w_L[c,s] * wt2[c,s] * dPibar[c,s]/dPi[c,s']
    J_pibar = torch.zeros(n, n, device=device, dtype=dtype)
    row_sum = p_prime.sum(dim=1, keepdim=True)
    anc_sum = p_prime @ ancestors_T
    pibar_denom = row_sum - anc_sum  # [W, S]
    anc_dense = sh['ancestors_dense'].to(device=device, dtype=dtype)
    for c in range(W):
        for s in range(S):
            for s_prime in range(S):
                # dPibar[c,s]/dPi[c,s'] = p_prime[c,s'] / pibar_denom[c,s] if s' not desc of s
                # s is NOT ancestor of s' → ancestors_dense[s, s'] = 0
                if anc_dense[s, s_prime] == 0:  # s is not ancestor of s'
                    dPibar = p_prime[c, s_prime] / pibar_denom[c, s]
                    J_pibar[c*S+s, c*S+s_prime] = (w_L[c,s] * w_terms[2][c,s] * dPibar).item()

    # Component 3: Speciation
    J_spec = torch.zeros(n, n, device=device, dtype=dtype)
    sc1 = sp_child1.long()
    sc2 = sp_child2.long()
    for c in range(W):
        for s in range(S):
            # SL1 term: d(SL1[s] + Pi[s1(s)])/dPi[s'] = delta(s' = s1(s))
            s1 = sc1[s].item()
            if s1 < S:
                J_spec[c*S+s, c*S+s1] += (w_L[c,s] * w_terms[3][c,s]).item()
            # SL2 term: d(SL2[s] + Pi[s2(s)])/dPi[s'] = delta(s' = s2(s))
            s2 = sc2[s].item()
            if s2 < S:
                J_spec[c*S+s, c*S+s2] += (w_L[c,s] * w_terms[4][c,s]).item()

    J_ana = J_diag + J_pibar + J_spec

    # ═══ Compare ═══
    total_err = (J_ana - J_fd).abs().max() / J_fd.abs().max()
    diag_err = (J_diag - J_fd * (J_diag != 0).float()).abs().max()  # only diagonal entries
    print(f"\nTotal J error (ana vs FD): {total_err:.4e}")

    # Check each component
    # Extract diagonal of J_fd
    J_fd_diag = torch.zeros_like(J_fd)
    for i in range(n):
        J_fd_diag[i, i] = J_fd[i, i]

    print(f"\nComponent errors (Frobenius norm):")
    print(f"  Diagonal: ana={J_diag.norm():.6e}, fd_diag={J_fd_diag.norm():.6e}")

    # Check by applying J^T to a test vector
    test_v = torch.randn(W, S, device=device, dtype=dtype)
    Jt_ana_vec = (J_ana.T @ test_v.reshape(-1)).reshape(W, S)
    Jt_fd_vec = (J_fd.T @ test_v.reshape(-1)).reshape(W, S)

    # Also get the ingredients-based J^T
    ingredients = _self_loop_vjp_precompute(
        Pi_W, Pibar_W, dts_r,
        mt_squeezed, DL_const, E_out["E_bar"], E_out["E"],
        SL1_const, SL2_const,
        sp_child1, sp_child2, leaf_wt, S,
        'uniform', None, ancestors_T,
    )
    Jt_ing_vec = _self_loop_Jt_apply(
        test_v, ingredients, sp_child1, sp_child2, S, W,
        'uniform', None, ancestors_T,
    )

    err_manual_fd = (Jt_ana_vec - Jt_fd_vec).abs().max() / Jt_fd_vec.abs().max()
    err_ing_fd = (Jt_ing_vec - Jt_fd_vec).abs().max() / Jt_fd_vec.abs().max()
    err_manual_ing = (Jt_ana_vec - Jt_ing_vec).abs().max() / Jt_ing_vec.abs().max()

    print(f"\nJ^T * v errors:")
    print(f"  Manual analytical vs FD:     {err_manual_fd:.4e}")
    print(f"  Ingredients-based vs FD:     {err_ing_fd:.4e}")
    print(f"  Manual vs ingredients-based: {err_manual_ing:.4e}")

    # Which component has the error?
    Jt_diag_vec = (J_diag.T @ test_v.reshape(-1)).reshape(W, S)
    Jt_pibar_vec = (J_pibar.T @ test_v.reshape(-1)).reshape(W, S)
    Jt_spec_vec = (J_spec.T @ test_v.reshape(-1)).reshape(W, S)

    # FD components (extract from full J)
    Jt_pibar_fd = Jt_fd_vec - Jt_diag_vec - Jt_spec_vec  # residual = pibar component
    pibar_comp_err = (Jt_pibar_vec - Jt_pibar_fd).abs().max() / max(Jt_pibar_fd.abs().max().item(), 1e-15)
    print(f"\n  Pibar component error: {pibar_comp_err:.4e}")

    # Check if DTS_L matches wave_step_fused
    print(f"\n  g_base vs Pi_W (convergence check): {(g_base - Pi_W.reshape(-1)).abs().max():.4e}")

    # Spectral radius
    eig = torch.linalg.eigvals(J_fd)
    print(f"\n  Spectral radius (FD): {eig.abs().max():.6e}")
    eig_ana = torch.linalg.eigvals(J_ana)
    print(f"  Spectral radius (analytical): {eig_ana.abs().max():.6e}")

    # Check if ancestors_T is consistent
    # anc_dense[s, s'] = 1 if s is ancestor of s'
    # Check: root species should be ancestor of all
    # ancestors_dense should have diagonal = 1 (each species is ancestor of itself)
    diag_check = torch.diag(anc_dense).sum().item()
    print(f"\n  ancestors_dense diagonal sum: {diag_check} (should be {S})")
    print(f"  ancestors_dense total: {anc_dense.sum().item()}")


if __name__ == "__main__":
    main()
