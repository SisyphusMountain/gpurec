"""Check grad_E[13] (E direct, numerator) for all 3 families."""
import math, torch
from pathlib import Path

from gpurec.core.preprocess_cpp import _load_extension
from gpurec.core.extract_parameters import extract_parameters_uniform
from gpurec.core.likelihood import E_fixed_point, compute_log_likelihood
from gpurec.core.forward import Pi_wave_forward
from gpurec.core.backward import Pi_wave_backward
from gpurec.core.scheduling import compute_clade_waves
from gpurec.core.batching import collate_gene_families, collate_wave, build_wave_layout
from gpurec.core.log2_utils import logsumexp2

_INV = 1.0 / math.log(2.0)
D, L, T = 0.05, 0.05, 0.05
S_IDX = 13
eps = 1e-4


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

    E_out = E_fixed_point(
        species_helpers=sh, log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
        transfer_mat=None, max_transfer_mat=mt,
        max_iters=2000, tolerance=1e-10, warm_start_E=None,
        dtype=dtype, device=device, pibar_mode='uniform', ancestors_T=ancestors_T)
    E_star = E_out["E"]
    Ebar = E_out["E_bar"]
    E_s1 = E_out["E_s1"]
    E_s2 = E_out["E_s2"]

    E_p = E_star.clone(); E_p[S_IDX] += eps
    E_m = E_star.clone(); E_m[S_IDX] -= eps

    for fi in range(3):
        bi = batch_items[fi]
        sb = collate_gene_families([bi], dtype=dtype, device=device)
        w_s, p = compute_clade_waves(bi["ccp"])
        cw = collate_wave([w_s], [0])
        wl = build_wave_layout(
            waves=cw, phases=p, ccp_helpers=sb["ccp"],
            leaf_row_index=sb["leaf_row_index"], leaf_col_index=sb["leaf_col_index"],
            root_clade_ids=sb["root_clade_ids"], device=device, dtype=dtype)

        def num_only(E_in):
            Po = Pi_wave_forward(
                wave_layout=wl, species_helpers=sh,
                E=E_in, Ebar=Ebar, E_s1=E_s1, E_s2=E_s2,
                log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
                transfer_mat=None, max_transfer_mat=mt,
                device=device, dtype=dtype, pibar_mode='uniform')
            root_probs = Po["Pi"][sb["root_clade_ids"], :]
            return -(logsumexp2(root_probs, dim=-1) - math.log2(S)).sum().item()

        fd_E_direct = (num_only(E_p) - num_only(E_m)) / (2 * eps)

        Po = Pi_wave_forward(
            wave_layout=wl, species_helpers=sh,
            E=E_star, Ebar=Ebar, E_s1=E_s1, E_s2=E_s2,
            log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
            transfer_mat=None, max_transfer_mat=mt,
            device=device, dtype=dtype, pibar_mode='uniform')

        pi_bwd = Pi_wave_backward(
            wave_layout=wl,
            Pi_star_wave=Po["Pi_wave_ordered"],
            Pibar_star_wave=Po["Pibar_wave_ordered"],
            E=E_star, Ebar=Ebar, E_s1=E_s1, E_s2=E_s2,
            log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
            max_transfer_mat=mt,
            species_helpers=sh,
            root_clade_ids_perm=wl["root_clade_ids"],
            device=device, dtype=dtype,
            neumann_terms=4, use_pruning=False,
            pibar_mode='uniform', ancestors_T=ancestors_T)

        ana = pi_bwd['grad_E'][S_IDX].item()
        err = abs(ana - fd_E_direct) / max(abs(fd_E_direct), 1e-8)

        # Also do dense mode for comparison
        from gpurec.core.extract_parameters import extract_parameters
        tm_unnorm = torch.log2(sh["Recipients_mat"]).to(device=device, dtype=dtype)
        pS_d, pD_d, pL_d, tm_d, mt_d = extract_parameters(
            theta, tm_unnorm, genewise=False, specieswise=True, pairwise=False)
        mt_d = mt_d.squeeze(-1) if mt_d.ndim == 2 else mt_d

        E_out_d = E_fixed_point(
            species_helpers=sh, log_pS=pS_d, log_pD=pD_d, log_pL=pL_d,
            transfer_mat=tm_d, max_transfer_mat=mt_d,
            max_iters=2000, tolerance=1e-10, warm_start_E=None,
            dtype=dtype, device=device, pibar_mode='dense')

        E_d_p = E_out_d["E"].clone(); E_d_p[S_IDX] += eps
        E_d_m = E_out_d["E"].clone(); E_d_m[S_IDX] -= eps

        def num_only_dense(E_in):
            Po = Pi_wave_forward(
                wave_layout=wl, species_helpers=sh,
                E=E_in, Ebar=E_out_d["E_bar"], E_s1=E_out_d["E_s1"], E_s2=E_out_d["E_s2"],
                log_pS=pS_d, log_pD=pD_d, log_pL=pL_d,
                transfer_mat=tm_d, max_transfer_mat=mt_d,
                device=device, dtype=dtype, pibar_mode='dense')
            root_probs = Po["Pi"][sb["root_clade_ids"], :]
            return -(logsumexp2(root_probs, dim=-1) - math.log2(S)).sum().item()

        fd_E_dense = (num_only_dense(E_d_p) - num_only_dense(E_d_m)) / (2 * eps)

        Po_d = Pi_wave_forward(
            wave_layout=wl, species_helpers=sh,
            E=E_out_d["E"], Ebar=E_out_d["E_bar"],
            E_s1=E_out_d["E_s1"], E_s2=E_out_d["E_s2"],
            log_pS=pS_d, log_pD=pD_d, log_pL=pL_d,
            transfer_mat=tm_d, max_transfer_mat=mt_d,
            device=device, dtype=dtype, pibar_mode='dense')

        pi_bwd_d = Pi_wave_backward(
            wave_layout=wl,
            Pi_star_wave=Po_d["Pi_wave_ordered"],
            Pibar_star_wave=Po_d["Pibar_wave_ordered"],
            E=E_out_d["E"], Ebar=E_out_d["E_bar"],
            E_s1=E_out_d["E_s1"], E_s2=E_out_d["E_s2"],
            log_pS=pS_d, log_pD=pD_d, log_pL=pL_d,
            max_transfer_mat=mt_d,
            species_helpers=sh,
            root_clade_ids_perm=wl["root_clade_ids"],
            device=device, dtype=dtype,
            neumann_terms=4, use_pruning=False,
            pibar_mode='dense', transfer_mat=tm_d)

        ana_d = pi_bwd_d['grad_E'][S_IDX].item()
        err_d = abs(ana_d - fd_E_dense) / max(abs(fd_E_dense), 1e-8)

        print(f"Family {fi} (C={bi['ccp']['C']}):")
        print(f"  UNIFORM: grad_E[{S_IDX}] analytic={ana:.6e}, FD={fd_E_direct:.6e}, err={err:.4e}")
        print(f"  DENSE:   grad_E[{S_IDX}] analytic={ana_d:.6e}, FD={fd_E_dense:.6e}, err={err_d:.4e}")
        print(f"  ratio uniform/dense: analytic={ana/ana_d:.4f}, FD={fd_E_direct/fd_E_dense:.4f}")
        print()


if __name__ == "__main__":
    main()
