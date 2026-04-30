"""Diagnostic: verify pi_bwd grad_log_pD by Pi-only FD (no E re-solve)."""
import math, torch
from pathlib import Path

from gpurec.core.preprocess_cpp import _load_extension
from gpurec.core.extract_parameters import extract_parameters_uniform
from gpurec.core.likelihood import E_fixed_point, compute_log_likelihood
from gpurec.core.forward import Pi_wave_forward
from gpurec.core.backward import Pi_wave_backward
from gpurec.core.scheduling import compute_clade_waves
from gpurec.core.batching import collate_gene_families, collate_wave, build_wave_layout

_INV = 1.0 / math.log(2.0)
D, L, T = 0.05, 0.05, 0.05

def load_data():
    device = torch.device("cuda")
    dtype = torch.float64
    ext = _load_extension()
    root = Path(__file__).resolve().parent.parent
    data_dir = root / "data" / "test_trees_20"
    sp_path = str(data_dir / "sp.nwk")
    gene_paths = sorted(data_dir.glob("g_*.nwk"))[:3]
    families, batch_items = [], []
    sr = None
    for gp in gene_paths:
        raw = ext.preprocess(sp_path, [str(gp)])
        if sr is None: sr = raw["species"]
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
        item = {
            "ccp": ch,
            "leaf_row_index": raw["leaf_row_index"].long(),
            "leaf_col_index": raw["leaf_col_index"].long(),
            "root_clade_id": int(cr["root_clade_id"]),
        }
        batch_items.append(item)
    sh = {
        "S": int(sr["S"]),
        "names": sr["names"],
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

    # Compute E ONCE at the base theta
    E_out = E_fixed_point(
        species_helpers=sh, log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
        transfer_mat=None, max_transfer_mat=mt,
        max_iters=2000, tolerance=1e-10, warm_start_E=None,
        dtype=dtype, device=device, pibar_mode='uniform', ancestors_T=ancestors_T)

    eps = 1e-4
    species_idx = 13

    for fi in range(3):
        bi = batch_items[fi]
        sb = collate_gene_families([bi], dtype=dtype, device=device)
        w, p = compute_clade_waves(bi["ccp"])
        cw = collate_wave([w], [0])
        wl = build_wave_layout(
            waves=cw, phases=p, ccp_helpers=sb["ccp"],
            leaf_row_index=sb["leaf_row_index"], leaf_col_index=sb["leaf_col_index"],
            root_clade_ids=sb["root_clade_ids"], device=device, dtype=dtype)

        # Base Pi forward
        Po = Pi_wave_forward(
            wave_layout=wl, species_helpers=sh,
            E=E_out["E"], Ebar=E_out["E_bar"],
            E_s1=E_out["E_s1"], E_s2=E_out["E_s2"],
            log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
            transfer_mat=None, max_transfer_mat=mt,
            device=device, dtype=dtype, pibar_mode='uniform')
        logL_base = compute_log_likelihood(Po["Pi"], E_out["E"], sb["root_clade_ids"]).sum().item()

        # Pi backward
        pi_bwd = Pi_wave_backward(
            wave_layout=wl,
            Pi_star_wave=Po["Pi_wave_ordered"],
            Pibar_star_wave=Po["Pibar_wave_ordered"],
            E=E_out["E"], Ebar=E_out["E_bar"],
            E_s1=E_out["E_s1"], E_s2=E_out["E_s2"],
            log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
            max_transfer_mat=mt,
            species_helpers=sh,
            root_clade_ids_perm=wl["root_clade_ids"],
            device=device, dtype=dtype,
            neumann_terms=4, use_pruning=False,
            pibar_mode='uniform', ancestors_T=ancestors_T)

        # --- Pi-only FD for log_pD[13] ---
        log_pD_p = log_pD.clone(); log_pD_p[species_idx] += eps
        Po_p = Pi_wave_forward(
            wave_layout=wl, species_helpers=sh,
            E=E_out["E"], Ebar=E_out["E_bar"],
            E_s1=E_out["E_s1"], E_s2=E_out["E_s2"],
            log_pS=log_pS, log_pD=log_pD_p, log_pL=log_pL,
            transfer_mat=None, max_transfer_mat=mt,
            device=device, dtype=dtype, pibar_mode='uniform')
        logL_p = compute_log_likelihood(Po_p["Pi"], E_out["E"], sb["root_clade_ids"]).sum().item()

        log_pD_m = log_pD.clone(); log_pD_m[species_idx] -= eps
        Po_m = Pi_wave_forward(
            wave_layout=wl, species_helpers=sh,
            E=E_out["E"], Ebar=E_out["E_bar"],
            E_s1=E_out["E_s1"], E_s2=E_out["E_s2"],
            log_pS=log_pS, log_pD=log_pD_m, log_pL=log_pL,
            transfer_mat=None, max_transfer_mat=mt,
            device=device, dtype=dtype, pibar_mode='uniform')
        logL_m = compute_log_likelihood(Po_m["Pi"], E_out["E"], sb["root_clade_ids"]).sum().item()

        fd_pD = (logL_p - logL_m) / (2 * eps)
        ana_pD = pi_bwd['grad_log_pD'][species_idx].item()

        # --- Pi-only FD for mt[13] ---
        mt_p = mt.clone(); mt_p[species_idx] += eps
        Po_p2 = Pi_wave_forward(
            wave_layout=wl, species_helpers=sh,
            E=E_out["E"], Ebar=E_out["E_bar"],
            E_s1=E_out["E_s1"], E_s2=E_out["E_s2"],
            log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
            transfer_mat=None, max_transfer_mat=mt_p,
            device=device, dtype=dtype, pibar_mode='uniform')
        logL_p2 = compute_log_likelihood(Po_p2["Pi"], E_out["E"], sb["root_clade_ids"]).sum().item()

        mt_m = mt.clone(); mt_m[species_idx] -= eps
        Po_m2 = Pi_wave_forward(
            wave_layout=wl, species_helpers=sh,
            E=E_out["E"], Ebar=E_out["E_bar"],
            E_s1=E_out["E_s1"], E_s2=E_out["E_s2"],
            log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
            transfer_mat=None, max_transfer_mat=mt_m,
            device=device, dtype=dtype, pibar_mode='uniform')
        logL_m2 = compute_log_likelihood(Po_m2["Pi"], E_out["E"], sb["root_clade_ids"]).sum().item()

        fd_mt = (logL_p2 - logL_m2) / (2 * eps)
        # pi_bwd['grad_max_transfer_mat'] captures grad from both self-loop and cross-clade Pibar paths
        # pi_bwd['grad_Ebar'] captures the Ebar→mt path (since Ebar = f(E) + mt)
        # But mt also enters through Ebar: we need to verify separately
        ana_mt_pibar = pi_bwd['grad_max_transfer_mat'][species_idx].item()
        ana_ebar = pi_bwd['grad_Ebar'][species_idx].item()
        ana_mt_total = ana_mt_pibar + ana_ebar

        # --- Pi-only FD for log_pS[13] ---
        log_pS_p = log_pS.clone(); log_pS_p[species_idx] += eps
        Po_p3 = Pi_wave_forward(
            wave_layout=wl, species_helpers=sh,
            E=E_out["E"], Ebar=E_out["E_bar"],
            E_s1=E_out["E_s1"], E_s2=E_out["E_s2"],
            log_pS=log_pS_p, log_pD=log_pD, log_pL=log_pL,
            transfer_mat=None, max_transfer_mat=mt,
            device=device, dtype=dtype, pibar_mode='uniform')
        logL_p3 = compute_log_likelihood(Po_p3["Pi"], E_out["E"], sb["root_clade_ids"]).sum().item()

        log_pS_m = log_pS.clone(); log_pS_m[species_idx] -= eps
        Po_m3 = Pi_wave_forward(
            wave_layout=wl, species_helpers=sh,
            E=E_out["E"], Ebar=E_out["E_bar"],
            E_s1=E_out["E_s1"], E_s2=E_out["E_s2"],
            log_pS=log_pS_m, log_pD=log_pD, log_pL=log_pL,
            transfer_mat=None, max_transfer_mat=mt,
            device=device, dtype=dtype, pibar_mode='uniform')
        logL_m3 = compute_log_likelihood(Po_m3["Pi"], E_out["E"], sb["root_clade_ids"]).sum().item()

        fd_pS = (logL_p3 - logL_m3) / (2 * eps)
        ana_pS = pi_bwd['grad_log_pS'][species_idx].item()

        # --- Pi-only FD for E[13] ---
        E_p = E_out["E"].clone(); E_p[species_idx] += eps
        # Need recomputed Ebar for perturbed E
        from gpurec.core.likelihood import E_step
        E_ebar_p = E_step(
            E_p, sh['s_P_indexes'], sh['s_C12_indexes'],
            log_pS, log_pD, log_pL, None, mt,
            pibar_mode='uniform', ancestors_T=ancestors_T)
        Ebar_p = E_ebar_p[3]  # Ebar from E_step output

        Po_Ep = Pi_wave_forward(
            wave_layout=wl, species_helpers=sh,
            E=E_p, Ebar=Ebar_p,
            E_s1=E_out["E_s1"], E_s2=E_out["E_s2"],  # don't change E_s1/E_s2
            log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
            transfer_mat=None, max_transfer_mat=mt,
            device=device, dtype=dtype, pibar_mode='uniform')
        num_Ep = compute_log_likelihood(Po_Ep["Pi"], E_p, sb["root_clade_ids"]).sum().item()

        E_m = E_out["E"].clone(); E_m[species_idx] -= eps
        E_ebar_m = E_step(
            E_m, sh['s_P_indexes'], sh['s_C12_indexes'],
            log_pS, log_pD, log_pL, None, mt,
            pibar_mode='uniform', ancestors_T=ancestors_T)
        Ebar_m = E_ebar_m[3]
        Po_Em = Pi_wave_forward(
            wave_layout=wl, species_helpers=sh,
            E=E_m, Ebar=Ebar_m,
            E_s1=E_out["E_s1"], E_s2=E_out["E_s2"],
            log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
            transfer_mat=None, max_transfer_mat=mt,
            device=device, dtype=dtype, pibar_mode='uniform')
        num_Em = compute_log_likelihood(Po_Em["Pi"], E_m, sb["root_clade_ids"]).sum().item()

        fd_E_full = (num_Ep - num_Em) / (2 * eps)
        ana_E = pi_bwd['grad_E'][species_idx].item()
        ana_Ebar = pi_bwd['grad_Ebar'][species_idx].item()

        print(f"\nFamily {fi} (C={batch_items[fi]['ccp']['C']}):")
        print(f"  grad_log_pD[{species_idx}]: analytic={ana_pD:.8e}, FD={fd_pD:.8e}, err={abs(ana_pD-fd_pD)/max(abs(fd_pD), 1e-8):.4e}")
        print(f"  grad_log_pS[{species_idx}]: analytic={ana_pS:.8e}, FD={fd_pS:.8e}, err={abs(ana_pS-fd_pS)/max(abs(fd_pS), 1e-8):.4e}")
        print(f"  grad_mt[{species_idx}]:     pibar={ana_mt_pibar:.8e}, Ebar={ana_ebar:.8e}, total={ana_mt_total:.8e}, FD={fd_mt:.8e}, err={abs(ana_mt_total-fd_mt)/max(abs(fd_mt), 1e-8):.4e}")
        print(f"  grad_E[{species_idx}]+Ebar: E={ana_E:.8e}, Ebar={ana_Ebar:.8e}, sum={ana_E+ana_Ebar:.8e}, FD(E+Ebar)={fd_E_full:.8e}, err={abs(ana_E+ana_Ebar-fd_E_full)/max(abs(fd_E_full), 1e-8):.4e}")


if __name__ == "__main__":
    main()
