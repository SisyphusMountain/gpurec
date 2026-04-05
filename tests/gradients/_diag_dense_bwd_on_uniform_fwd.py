"""Run DENSE backward on UNIFORM forward's Pi. If grad_E is correct,
the bug is in the uniform backward, not the forward or chain rule."""
import math, torch
from pathlib import Path

from gpurec.core.preprocess_cpp import _load_extension
from gpurec.core.extract_parameters import extract_parameters_uniform, extract_parameters
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
    tm_unnorm = torch.log2(sh["Recipients_mat"]).to(device=device, dtype=dtype)
    unnorm_row_max = tm_unnorm.max(dim=-1).values
    ancestors_T = sh["ancestors_dense"].T.to_sparse_coo() if "ancestors_dense" in sh else None
    return batch_items, sh, S, tm_unnorm, unnorm_row_max, ancestors_T, device, dtype


def main():
    batch_items, sh, S, tm_unnorm, unnorm_row_max, ancestors_T, device, dtype = load_data()
    theta = torch.log2(torch.tensor([D, L, T], dtype=dtype, device=device)).unsqueeze(0).expand(S, -1).contiguous()

    # Uniform params & E
    log_pS, log_pD, log_pL, _, mt = extract_parameters_uniform(theta, unnorm_row_max, specieswise=True)
    E_out = E_fixed_point(
        species_helpers=sh, log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
        transfer_mat=None, max_transfer_mat=mt,
        max_iters=2000, tolerance=1e-10, warm_start_E=None,
        dtype=dtype, device=device, pibar_mode='uniform', ancestors_T=ancestors_T)

    # Dense params (for transfer_mat)
    pS_d, pD_d, pL_d, tm_d, mt_d = extract_parameters(
        theta, tm_unnorm, genewise=False, specieswise=True, pairwise=False)
    mt_d = mt_d.squeeze(-1) if mt_d.ndim == 2 else mt_d

    bi = batch_items[FI]
    sb = collate_gene_families([bi], dtype=dtype, device=device)
    w_s, p = compute_clade_waves(bi["ccp"])
    cw = collate_wave([w_s], [0])
    wl = build_wave_layout(
        waves=cw, phases=p, ccp_helpers=sb["ccp"],
        leaf_row_index=sb["leaf_row_index"], leaf_col_index=sb["leaf_col_index"],
        root_clade_ids=sb["root_clade_ids"], device=device, dtype=dtype)

    # UNIFORM forward
    Po_u = Pi_wave_forward(
        wave_layout=wl, species_helpers=sh,
        E=E_out["E"], Ebar=E_out["E_bar"],
        E_s1=E_out["E_s1"], E_s2=E_out["E_s2"],
        log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
        transfer_mat=None, max_transfer_mat=mt,
        device=device, dtype=dtype, pibar_mode='uniform')

    # FD of grad_E (direct E, numerator only)
    E_p = E_out["E"].clone(); E_p[S_IDX] += eps
    E_m = E_out["E"].clone(); E_m[S_IDX] -= eps
    def num_u(E_in):
        Po = Pi_wave_forward(
            wave_layout=wl, species_helpers=sh,
            E=E_in, Ebar=E_out["E_bar"], E_s1=E_out["E_s1"], E_s2=E_out["E_s2"],
            log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
            transfer_mat=None, max_transfer_mat=mt,
            device=device, dtype=dtype, pibar_mode='uniform')
        rp = Po["Pi"][sb["root_clade_ids"], :]
        return -(logsumexp2(rp, dim=-1) - math.log2(S)).sum().item()

    fd_u = (num_u(E_p) - num_u(E_m)) / (2 * eps)

    # 1. UNIFORM backward on UNIFORM forward's Pi
    bwd_uu = Pi_wave_backward(
        wave_layout=wl,
        Pi_star_wave=Po_u["Pi_wave_ordered"],
        Pibar_star_wave=Po_u["Pibar_wave_ordered"],
        E=E_out["E"], Ebar=E_out["E_bar"],
        E_s1=E_out["E_s1"], E_s2=E_out["E_s2"],
        log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
        max_transfer_mat=mt,
        species_helpers=sh,
        root_clade_ids_perm=wl["root_clade_ids"],
        device=device, dtype=dtype,
        neumann_terms=4, use_pruning=False,
        pibar_mode='uniform', ancestors_T=ancestors_T)

    # 2. DENSE backward on UNIFORM forward's Pi
    # Need to recompute Pibar using dense mode from the uniform Pi
    # (Pibar_star_wave from uniform is NOT the same as dense Pibar)
    # Actually, let me just use the uniform Pibar_star_wave — the backward
    # uses it only for the DTS_L terms, and both modes should give similar DTS_L
    # at the same Pi.
    # But the key is: the backward uses the PIBAR_MODE to decide J^T and
    # cross-clade VJP formulas. By passing pibar_mode='dense', we use the
    # dense J^T on the uniform Pi.
    #
    # BUT: dense needs transfer_mat. And mt from dense != mt from uniform.
    # Actually, for specieswise non-pairwise, mt_d = log_pT + unnorm_row_max
    # and mt from uniform = log_pT + unnorm_row_max. So they should be identical
    # (since the softmax gives the same log_pT).
    # And log_pS, log_pD, log_pL from dense == from uniform (same softmax).
    # But transfer_mat from dense != None.

    # Recompute Pibar for the uniform Pi using dense formula
    from gpurec.core.forward import _compute_Pibar_inline
    C = Po_u["Pi_wave_ordered"].shape[0]
    Pi_wave = Po_u["Pi_wave_ordered"]
    tm_T = tm_d.T.contiguous()
    Pibar_dense = torch.zeros_like(Pi_wave)
    for meta in wl["wave_metas"]:
        ws, we = meta["start"], meta["end"]
        Pi_W = Pi_wave[ws:we]
        Pibar_dense[ws:we] = _compute_Pibar_inline(Pi_W, tm_T, mt, 'dense')

    bwd_du = Pi_wave_backward(
        wave_layout=wl,
        Pi_star_wave=Po_u["Pi_wave_ordered"],
        Pibar_star_wave=Pibar_dense,  # Use dense Pibar on uniform Pi
        E=E_out["E"], Ebar=E_out["E_bar"],
        E_s1=E_out["E_s1"], E_s2=E_out["E_s2"],
        log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
        max_transfer_mat=mt,
        species_helpers=sh,
        root_clade_ids_perm=wl["root_clade_ids"],
        device=device, dtype=dtype,
        neumann_terms=4, use_pruning=False,
        pibar_mode='dense', transfer_mat=tm_d)

    ana_uu = bwd_uu['grad_E'][S_IDX].item()
    ana_du = bwd_du['grad_E'][S_IDX].item()

    err_uu = abs(ana_uu - fd_u) / max(abs(fd_u), 1e-8)
    err_du = abs(ana_du - fd_u) / max(abs(fd_u), 1e-8)

    print(f"Family {FI}, species {S_IDX}")
    print(f"  FD (uniform fwd)        = {fd_u:.8e}")
    print(f"  Uniform bwd on uniform  = {ana_uu:.8e}  err={err_uu:.4e}")
    print(f"  Dense bwd on uniform    = {ana_du:.8e}  err={err_du:.4e}")

    # Also check Pibar_star_wave: how different are uniform and dense Pibar at the same Pi?
    pibar_u = Po_u["Pibar_wave_ordered"]
    diff = (pibar_u - Pibar_dense).abs()
    print(f"\n  Pibar uniform vs dense at same Pi:")
    print(f"    Max diff:  {diff.max():.4e}")
    print(f"    Mean diff: {diff.mean():.4e}")

    # Check grad_log_pD too (should be correct for both)
    ana_pD_uu = bwd_uu['grad_log_pD'][S_IDX].item()
    ana_pD_du = bwd_du['grad_log_pD'][S_IDX].item()
    print(f"\n  grad_log_pD[{S_IDX}]:")
    print(f"    Uniform bwd: {ana_pD_uu:.8e}")
    print(f"    Dense bwd:   {ana_pD_du:.8e}")


if __name__ == "__main__":
    main()
