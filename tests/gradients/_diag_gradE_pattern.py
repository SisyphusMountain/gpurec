"""Check grad_E pattern for Family 1: which species are wrong, and is it
the self-loop or cross-clade backward?"""
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
FI = 1
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

    bi = batch_items[FI]
    sb = collate_gene_families([bi], dtype=dtype, device=device)
    w_s, p = compute_clade_waves(bi["ccp"])
    cw = collate_wave([w_s], [0])
    wl = build_wave_layout(
        waves=cw, phases=p, ccp_helpers=sb["ccp"],
        leaf_row_index=sb["leaf_row_index"], leaf_col_index=sb["leaf_col_index"],
        root_clade_ids=sb["root_clade_ids"], device=device, dtype=dtype)

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

    # FD grad_E for all species (numerator only, E direct)
    def num_only(E_in):
        Po2 = Pi_wave_forward(
            wave_layout=wl, species_helpers=sh,
            E=E_in, Ebar=Ebar, E_s1=E_s1, E_s2=E_s2,
            log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
            transfer_mat=None, max_transfer_mat=mt,
            device=device, dtype=dtype, pibar_mode='uniform')
        root_probs = Po2["Pi"][sb["root_clade_ids"], :]
        return -(logsumexp2(root_probs, dim=-1) - math.log2(S)).sum().item()

    fd_gradE = torch.zeros(S, dtype=dtype, device=device)
    print(f"Computing FD grad_E for all {S} species...")
    for s in range(S):
        E_p = E_star.clone(); E_p[s] += eps
        E_m = E_star.clone(); E_m[s] -= eps
        fd_gradE[s] = (num_only(E_p) - num_only(E_m)) / (2 * eps)

    ana_gradE = pi_bwd['grad_E']

    # Compare
    rel_err = (ana_gradE - fd_gradE).abs() / fd_gradE.abs().clamp(min=1e-10)
    print(f"\nFamily {FI}: grad_E comparison (uniform mode)")
    print(f"  Max absolute error: {(ana_gradE - fd_gradE).abs().max():.4e}")
    print(f"  Max relative error: {rel_err.max():.4e} at species {rel_err.argmax().item()}")
    print(f"  Mean relative error: {rel_err.mean():.4e}")
    print(f"  Median relative error: {rel_err.median():.4e}")

    print(f"\n  Top 10 species by relative error:")
    topk = rel_err.topk(10)
    for i, (err, idx) in enumerate(zip(topk.values, topk.indices)):
        s = idx.item()
        print(f"    species {s}: analytic={ana_gradE[s]:.6e}, FD={fd_gradE[s]:.6e}, rel_err={err:.4e}")

    print(f"\n  Species with < 5% error:")
    good = (rel_err < 0.05).sum().item()
    print(f"    {good}/{S} species")

    # Check the ratio ana/FD for all species
    ratio = ana_gradE / fd_gradE.clamp(min=1e-20)
    print(f"\n  Ratio analytic/FD (should be ~1):")
    print(f"    Mean: {ratio.mean():.4f}")
    print(f"    Std:  {ratio.std():.4f}")
    print(f"    Min:  {ratio.min():.4f} at species {ratio.argmin().item()}")
    print(f"    Max:  {ratio.max():.4f} at species {ratio.argmax().item()}")


if __name__ == "__main__":
    main()
