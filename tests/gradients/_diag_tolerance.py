"""Check if forward convergence tolerance affects the FD result."""
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

    bi = batch_items[FI]
    sb = collate_gene_families([bi], dtype=dtype, device=device)
    w_s, p = compute_clade_waves(bi["ccp"])
    cw = collate_wave([w_s], [0])
    wl = build_wave_layout(
        waves=cw, phases=p, ccp_helpers=sb["ccp"],
        leaf_row_index=sb["leaf_row_index"], leaf_col_index=sb["leaf_col_index"],
        root_clade_ids=sb["root_clade_ids"], device=device, dtype=dtype)

    E_p = E_out["E"].clone(); E_p[S_IDX] += eps
    E_m = E_out["E"].clone(); E_m[S_IDX] -= eps

    for tol_name, tol_val in [("1e-3 (default)", 1e-3), ("1e-6", 1e-6), ("1e-10", 1e-10)]:
        def num(E_in):
            Po = Pi_wave_forward(
                wave_layout=wl, species_helpers=sh,
                E=E_in, Ebar=E_out["E_bar"], E_s1=E_out["E_s1"], E_s2=E_out["E_s2"],
                log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
                transfer_mat=None, max_transfer_mat=mt,
                device=device, dtype=dtype, pibar_mode='uniform',
                local_tolerance=tol_val, local_iters=5000)
            rp = Po["Pi"][sb["root_clade_ids"], :]
            return -(logsumexp2(rp, dim=-1) - math.log2(S)).sum().item(), Po['iterations']

        base, iters_base = num(E_out["E"])
        plus, iters_p = num(E_p)
        minus, iters_m = num(E_m)
        fd = (plus - minus) / (2 * eps)
        print(f"tol={tol_name}: FD={fd:.8e}, iters={iters_base}/{iters_p}/{iters_m}")

    # Also run the backward with tight-tolerance forward
    Po_tight = Pi_wave_forward(
        wave_layout=wl, species_helpers=sh,
        E=E_out["E"], Ebar=E_out["E_bar"], E_s1=E_out["E_s1"], E_s2=E_out["E_s2"],
        log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
        transfer_mat=None, max_transfer_mat=mt,
        device=device, dtype=dtype, pibar_mode='uniform',
        local_tolerance=1e-10, local_iters=5000)

    bwd_tight = Pi_wave_backward(
        wave_layout=wl,
        Pi_star_wave=Po_tight["Pi_wave_ordered"],
        Pibar_star_wave=Po_tight["Pibar_wave_ordered"],
        E=E_out["E"], Ebar=E_out["E_bar"],
        E_s1=E_out["E_s1"], E_s2=E_out["E_s2"],
        log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
        max_transfer_mat=mt,
        species_helpers=sh,
        root_clade_ids_perm=wl["root_clade_ids"],
        device=device, dtype=dtype,
        neumann_terms=4, use_pruning=False,
        pibar_mode='uniform', ancestors_T=ancestors_T)

    print(f"\nBackward (tight tol forward): grad_E[{S_IDX}] = {bwd_tight['grad_E'][S_IDX]:.8e}")

    # Check max Pi diff between default and tight tolerance
    Po_default = Pi_wave_forward(
        wave_layout=wl, species_helpers=sh,
        E=E_out["E"], Ebar=E_out["E_bar"], E_s1=E_out["E_s1"], E_s2=E_out["E_s2"],
        log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
        transfer_mat=None, max_transfer_mat=mt,
        device=device, dtype=dtype, pibar_mode='uniform')

    diff = (Po_default["Pi_wave_ordered"] - Po_tight["Pi_wave_ordered"]).abs()
    print(f"Pi diff (default vs tight): max={diff.max():.4e}, mean={diff.mean():.4e}")


if __name__ == "__main__":
    main()
