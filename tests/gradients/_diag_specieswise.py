"""Diagnostic: decompose analytic gradient for specieswise uniform at theta[13,1]."""
import math, torch
from pathlib import Path

from gpurec.core.preprocess_cpp import _load_extension
from gpurec.core.extract_parameters import extract_parameters_uniform
from gpurec.core.likelihood import E_fixed_point, compute_log_likelihood
from gpurec.core.forward import Pi_wave_forward
from gpurec.core.backward import Pi_wave_backward
from gpurec.core.scheduling import compute_clade_waves
from gpurec.core.batching import collate_gene_families, collate_wave, build_wave_layout
from gpurec.optimization.implicit_grad import _e_adjoint_and_theta_vjp

_INV = 1.0 / math.log(2.0)
D, L, T = 0.05, 0.05, 0.05
IDX = (13, 1)  # the failing theta index

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
        item = {
            "ccp": ch,
            "leaf_row_index": raw["leaf_row_index"].long(),
            "leaf_col_index": raw["leaf_col_index"].long(),
            "root_clade_id": int(cr["root_clade_id"]),
        }
        batch_items.append(item)
        families.append({
            "ccp_helpers": ch,
            "leaf_row_index": raw["leaf_row_index"].long(),
            "leaf_col_index": raw["leaf_col_index"].long(),
            "root_clade_id": int(cr["root_clade_id"]),
        })

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
    tm_unnorm = torch.log2(sh["Recipients_mat"]).to(device=device, dtype=dtype)
    unnorm_row_max = tm_unnorm.max(dim=-1).values
    ancestors_T = sh["ancestors_dense"].T.to_sparse_coo() if "ancestors_dense" in sh else None

    return families, batch_items, sh, S, unnorm_row_max, ancestors_T, device, dtype


def make_theta(S, device, dtype):
    base = torch.log2(torch.tensor([D, L, T], dtype=dtype, device=device))
    return base.unsqueeze(0).expand(S, -1).contiguous()


def forward_single_family(theta, bi, sh, unnorm_row_max, ancestors_T, device, dtype):
    """Single-family forward returning intermediates."""
    log_pS, log_pD, log_pL, _, mt = extract_parameters_uniform(
        theta, unnorm_row_max, specieswise=True)

    E_out = E_fixed_point(
        species_helpers=sh,
        log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
        transfer_mat=None, max_transfer_mat=mt,
        max_iters=2000, tolerance=1e-10, warm_start_E=None,
        dtype=dtype, device=device,
        pibar_mode='uniform', ancestors_T=ancestors_T)

    sb = collate_gene_families([bi], dtype=dtype, device=device)
    w, p = compute_clade_waves(bi["ccp"])
    cw = collate_wave([w], [0])
    wl = build_wave_layout(
        waves=cw, phases=p, ccp_helpers=sb["ccp"],
        leaf_row_index=sb["leaf_row_index"], leaf_col_index=sb["leaf_col_index"],
        root_clade_ids=sb["root_clade_ids"], device=device, dtype=dtype)

    Po = Pi_wave_forward(
        wave_layout=wl, species_helpers=sh,
        E=E_out["E"], Ebar=E_out["E_bar"],
        E_s1=E_out["E_s1"], E_s2=E_out["E_s2"],
        log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
        transfer_mat=None, max_transfer_mat=mt,
        device=device, dtype=dtype, pibar_mode='uniform')

    logL = compute_log_likelihood(Po["Pi"], E_out["E"], sb["root_clade_ids"])
    return logL.sum().item(), wl, Po, E_out, sb


def main():
    families, batch_items, sh, S, unnorm_row_max, ancestors_T, device, dtype = load_data()
    theta = make_theta(S, device, dtype)

    # ── FD for theta[13, 1] ──
    eps = 1e-4

    # Per-family FD
    print("=== Per-family FD for theta[13,1] ===")
    fd_total = 0.0
    for fi, bi in enumerate(batch_items):
        tp = theta.clone(); tp[IDX] += eps
        lp, *_ = forward_single_family(tp, bi, sh, unnorm_row_max, ancestors_T, device, dtype)
        tm = theta.clone(); tm[IDX] -= eps
        lm, *_ = forward_single_family(tm, bi, sh, unnorm_row_max, ancestors_T, device, dtype)
        fd = (lp - lm) / (2 * eps)
        fd_total += fd
        print(f"  Family {fi}: FD = {fd:.8e}")
    print(f"  Total FD = {fd_total:.8e}")

    # ── Per-family analytic gradient ──
    print("\n=== Per-family analytic gradient for theta[13,1] ===")
    ana_total = 0.0
    for fi, bi in enumerate(batch_items):
        logL_val, wl, Po, E_out, sb = forward_single_family(theta, bi, sh, unnorm_row_max, ancestors_T, device, dtype)

        # Run Pi backward
        pi_bwd = Pi_wave_backward(
            wave_layout=wl,
            Pi_star_wave=Po["Pi_wave_ordered"],
            Pibar_star_wave=Po["Pibar_wave_ordered"],
            E=E_out["E"], Ebar=E_out["E_bar"],
            E_s1=E_out["E_s1"], E_s2=E_out["E_s2"],
            log_pS=extract_parameters_uniform(theta, unnorm_row_max, specieswise=True)[0],
            log_pD=extract_parameters_uniform(theta, unnorm_row_max, specieswise=True)[1],
            log_pL=extract_parameters_uniform(theta, unnorm_row_max, specieswise=True)[2],
            max_transfer_mat=extract_parameters_uniform(theta, unnorm_row_max, specieswise=True)[4],
            species_helpers=sh,
            root_clade_ids_perm=wl["root_clade_ids"],
            device=device, dtype=dtype,
            neumann_terms=4, use_pruning=False,
            pibar_mode='uniform',
            ancestors_T=ancestors_T)

        # Separate the two gradient components
        # 1. grad_theta_pi (Pi backward → theta)
        log_pS, log_pD, log_pL, _, mt = extract_parameters_uniform(theta, unnorm_row_max, specieswise=True)
        grad_mt_total = pi_bwd['grad_max_transfer_mat'] + pi_bwd['grad_Ebar']

        theta_req = theta.detach().requires_grad_(True)
        with torch.enable_grad():
            log_pS_r, log_pD_r, log_pL_r, _, mt_r = extract_parameters_uniform(
                theta_req, unnorm_row_max, specieswise=True)
            param_loss = (
                (log_pS_r * pi_bwd['grad_log_pS']).sum() +
                (log_pD_r * pi_bwd['grad_log_pD']).sum() +
                (mt_r * grad_mt_total).sum()
            )
            grad_theta_pi = torch.autograd.grad(param_loss, theta_req)[0]

        # 2. gtheta_E (E adjoint → theta)
        grad_theta_full, _ = _e_adjoint_and_theta_vjp(
            pi_bwd, E_out["E"], E_out["E_bar"], E_out["E_s1"], E_out["E_s2"],
            log_pS, log_pD, log_pL,
            mt, sh, wl["root_clade_ids"],
            theta, unnorm_row_max, specieswise=True,
            device=device, dtype=dtype,
            cg_tol=1e-10, cg_maxiter=1000,
            pibar_mode='uniform', ancestors_T=ancestors_T)

        gtheta_E_val = (grad_theta_full - grad_theta_pi)[IDX].item()
        gpi_val = grad_theta_pi[IDX].item()
        total_val = grad_theta_full[IDX].item()

        print(f"  Family {fi}: grad_theta_pi={gpi_val:.8e}, gtheta_E={gtheta_E_val:.8e}, total={total_val:.8e}")
        ana_total += total_val

    print(f"  Total analytic = {ana_total:.8e}")
    print(f"\n  Rel error = {abs(ana_total - fd_total) / max(abs(fd_total), 1e-8):.4e}")

    # ── Check if any family has multiple root clades ──
    print("\n=== Root clade checks ===")
    for fi, bi in enumerate(batch_items):
        sb = collate_gene_families([bi], dtype=dtype, device=device)
        print(f"  Family {fi}: root_clade_ids = {sb['root_clade_ids'].tolist()}, C = {bi['ccp']['C']}")

    # ── Detailed pi_bwd inspection for family 1 ──
    print("\n=== Detailed pi_bwd for family 1 ===")
    fi = 1
    bi = batch_items[fi]
    _, wl, Po, E_out, sb = forward_single_family(theta, bi, sh, unnorm_row_max, ancestors_T, device, dtype)
    log_pS, log_pD, log_pL, _, mt = extract_parameters_uniform(theta, unnorm_row_max, specieswise=True)

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

    for key in ('grad_E', 'grad_Ebar', 'grad_E_s1', 'grad_E_s2',
                'grad_log_pD', 'grad_log_pS', 'grad_max_transfer_mat'):
        v = pi_bwd[key]
        if v.ndim >= 1 and v.shape[-1] > 13:
            print(f"  {key}[13] = {v[..., 13].item():.8e}")
        else:
            print(f"  {key} = {v}")

    # Now also run dense mode for the same family to compare
    from gpurec.core.extract_parameters import extract_parameters
    tm_unnorm = torch.log2(sh["Recipients_mat"]).to(device=device, dtype=dtype)
    log_pS_d, log_pD_d, log_pL_d, tm_d, mt_d = extract_parameters(
        theta, tm_unnorm, genewise=False, specieswise=True, pairwise=False)
    if mt_d.ndim == 2:
        mt_d = mt_d.squeeze(-1)

    E_out_d = E_fixed_point(
        species_helpers=sh,
        log_pS=log_pS_d, log_pD=log_pD_d, log_pL=log_pL_d,
        transfer_mat=tm_d, max_transfer_mat=mt_d,
        max_iters=2000, tolerance=1e-10, warm_start_E=None,
        dtype=dtype, device=device, pibar_mode='dense')

    Po_d = Pi_wave_forward(
        wave_layout=wl, species_helpers=sh,
        E=E_out_d["E"], Ebar=E_out_d["E_bar"],
        E_s1=E_out_d["E_s1"], E_s2=E_out_d["E_s2"],
        log_pS=log_pS_d, log_pD=log_pD_d, log_pL=log_pL_d,
        transfer_mat=tm_d, max_transfer_mat=mt_d,
        device=device, dtype=dtype, pibar_mode='dense')

    pi_bwd_d = Pi_wave_backward(
        wave_layout=wl,
        Pi_star_wave=Po_d["Pi_wave_ordered"],
        Pibar_star_wave=Po_d["Pibar_wave_ordered"],
        E=E_out_d["E"], Ebar=E_out_d["E_bar"],
        E_s1=E_out_d["E_s1"], E_s2=E_out_d["E_s2"],
        log_pS=log_pS_d, log_pD=log_pD_d, log_pL=log_pL_d,
        max_transfer_mat=mt_d,
        species_helpers=sh,
        root_clade_ids_perm=wl["root_clade_ids"],
        device=device, dtype=dtype,
        neumann_terms=4, use_pruning=False,
        pibar_mode='dense', transfer_mat=tm_d)

    print("\n=== Dense mode pi_bwd for family 1 (comparison) ===")
    for key in ('grad_E', 'grad_Ebar', 'grad_E_s1', 'grad_E_s2',
                'grad_log_pD', 'grad_log_pS', 'grad_max_transfer_mat'):
        v = pi_bwd_d[key]
        if v.ndim >= 1 and v.shape[-1] > 13:
            print(f"  {key}[13] = {v[..., 13].item():.8e}")
        else:
            print(f"  {key} = {v}")


if __name__ == "__main__":
    main()
