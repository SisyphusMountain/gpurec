"""Diagnostic: verify ONE diagonal Jacobian entry at wave 1, clade 0, species 0.

Manually compute all intermediate values to find where the formula diverges from FD.
"""
import math, torch
from pathlib import Path

from gpurec.core.preprocess_cpp import _load_extension
from gpurec.core.extract_parameters import extract_parameters_uniform
from gpurec.core.likelihood import E_fixed_point
from gpurec.core.forward import Pi_wave_forward, _compute_Pibar_inline
from gpurec.core.backward import _dts_cross_differentiable
from gpurec.core.scheduling import compute_clade_waves
from gpurec.core.batching import collate_gene_families, collate_wave, build_wave_layout
from gpurec.core.log2_utils import logsumexp2, logaddexp2
from gpurec.core._helpers import _safe_exp2_ratio
from gpurec.core.kernels.wave_step import wave_step_fused

_INV = 1.0 / math.log(2.0)
D, L, T = 0.05, 0.05, 0.05
FI = 1
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

    WAVE_K = 1
    meta = wave_metas[WAVE_K]
    ws = meta['start']
    we = meta['end']
    W = meta['W']

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

    # Check a few (clade, species) entries
    for c in range(min(W, 3)):
        for s in [0, 13, 20]:
            pi = Pi_W[c, s].item()
            pibar = Pibar_W[c, s].item()
            dl = DL_const[s].item()
            ebar = E_out["E_bar"][s].item()
            e = E_out["E"][s].item()
            sl1 = SL1_const[s].item()
            sl2 = SL2_const[s].item()
            s1 = sp_child1[s].item()
            s2 = sp_child2[s].item()
            pi_s1 = Pi_W[c, s1].item() if s1 < S else NEG_INF
            pi_s2 = Pi_W[c, s2].item() if s2 < S else NEG_INF
            lw = leaf_wt[c, s].item()
            dts = dts_r[c, s].item() if dts_r is not None else NEG_INF

            t0 = dl + pi
            t1 = pi + ebar
            t2 = pibar + e
            t3 = sl1 + pi_s1
            t4 = sl2 + pi_s2
            t5 = lw

            # Manual logsumexp of all 7 terms
            all_terms = [t0, t1, t2, t3, t4, t5]
            if dts is not None and dts > NEG_INF:
                all_terms.append(dts)
            m = max(all_terms)
            s_sum = sum(2**(t - m) for t in all_terms)
            g_manual = math.log2(s_sum) + m

            # Two-level computation (as in VJP precompute)
            dtsl_terms = [t0, t1, t2, t3, t4, t5]
            m_dtsl = max(dtsl_terms)
            s_dtsl = sum(2**(t - m_dtsl) for t in dtsl_terms)
            DTS_L_manual = math.log2(s_dtsl) + m_dtsl

            if dts > NEG_INF:
                Pi_new_manual = math.log2(2**DTS_L_manual + 2**dts) if abs(DTS_L_manual - dts) < 50 else max(DTS_L_manual, dts) + math.log2(1 + 2**(-abs(DTS_L_manual - dts)))
            else:
                Pi_new_manual = DTS_L_manual

            w_L_manual = 2**(DTS_L_manual - Pi_new_manual)
            wt0_manual = 2**(t0 - DTS_L_manual)
            wt1_manual = 2**(t1 - DTS_L_manual)

            # Direct diagonal Jacobian: 2^(t0 - g) + 2^(t1 - g) where g = Pi*
            J_diag_direct = 2**(t0 - g_manual) + 2**(t1 - g_manual)
            J_diag_two_level = w_L_manual * (wt0_manual + wt1_manual)

            # FD check
            eps = 1e-6
            pi_pert = pi + eps
            t0_p = dl + pi_pert
            t1_p = pi_pert + ebar
            # Pibar doesn't change for self-species (s excluded from own Pibar)
            t2_p = t2  # unchanged
            # Speciation: Pi_s1, Pi_s2 don't change when only Pi[c,s] changes
            t3_p = t3; t4_p = t4; t5_p = t5

            all_p = [t0_p, t1_p, t2_p, t3_p, t4_p, t5_p]
            if dts > NEG_INF:
                all_p.append(dts)
            m_p = max(all_p)
            s_p = sum(2**(t - m_p) for t in all_p)
            g_p = math.log2(s_p) + m_p

            J_fd = (g_p - g_manual) / eps

            print(f"  Clade {c}, Species {s}:")
            print(f"    Pi={pi:.4f}, Pibar={pibar:.4f}")
            print(f"    t0={t0:.4f}, t1={t1:.4f}, t2={t2:.4f}, t3={t3:.4f}, t4={t4:.4f}, t5={t5:.4f}, dts_r={dts:.4f}")
            print(f"    g_manual={g_manual:.6f}, Pi_star={pi:.6f}, diff={g_manual-pi:.4e}")
            print(f"    DTS_L={DTS_L_manual:.6f}, w_L={w_L_manual:.6e}")
            print(f"    wt0={wt0_manual:.6e}, wt1={wt1_manual:.6e}")
            print(f"    J_diag(direct) = {J_diag_direct:.6e}")
            print(f"    J_diag(2-level) = {J_diag_two_level:.6e}")
            print(f"    J_diag(FD)      = {J_fd:.6e}")
            print()


if __name__ == "__main__":
    main()
