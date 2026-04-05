"""Diagnostic: decompose grad_theta into Pi path vs E path, verify each via FD."""
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
IDX = (13, 1)
S_IDX = 13

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
    eps = 1e-4

    # Compute E at base theta
    E_out = E_fixed_point(
        species_helpers=sh, log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
        transfer_mat=None, max_transfer_mat=mt,
        max_iters=2000, tolerance=1e-10, warm_start_E=None,
        dtype=dtype, device=device, pibar_mode='uniform', ancestors_T=ancestors_T)

    for fi in range(3):
        bi = batch_items[fi]
        sb = collate_gene_families([bi], dtype=dtype, device=device)
        w, p = compute_clade_waves(bi["ccp"])
        cw = collate_wave([w], [0])
        wl = build_wave_layout(
            waves=cw, phases=p, ccp_helpers=sb["ccp"],
            leaf_row_index=sb["leaf_row_index"], leaf_col_index=sb["leaf_col_index"],
            root_clade_ids=sb["root_clade_ids"], device=device, dtype=dtype)

        def logL_pi_only(th):
            """Run extract_parameters + Pi_wave_forward + logL with FIXED E."""
            pS, pD, pL, _, m = extract_parameters_uniform(th, unnorm_row_max, specieswise=True)
            Po = Pi_wave_forward(
                wave_layout=wl, species_helpers=sh,
                E=E_out["E"], Ebar=E_out["E_bar"],
                E_s1=E_out["E_s1"], E_s2=E_out["E_s2"],
                log_pS=pS, log_pD=pD, log_pL=pL,
                transfer_mat=None, max_transfer_mat=m,
                device=device, dtype=dtype, pibar_mode='uniform')
            return compute_log_likelihood(Po["Pi"], E_out["E"], sb["root_clade_ids"]).sum().item()

        def logL_full(th):
            """Run extract_parameters + E_fixed_point + Pi_wave_forward + logL."""
            pS, pD, pL, _, m = extract_parameters_uniform(th, unnorm_row_max, specieswise=True)
            E_o = E_fixed_point(
                species_helpers=sh, log_pS=pS, log_pD=pD, log_pL=pL,
                transfer_mat=None, max_transfer_mat=m,
                max_iters=2000, tolerance=1e-10, warm_start_E=None,
                dtype=dtype, device=device, pibar_mode='uniform', ancestors_T=ancestors_T)
            Po = Pi_wave_forward(
                wave_layout=wl, species_helpers=sh,
                E=E_o["E"], Ebar=E_o["E_bar"],
                E_s1=E_o["E_s1"], E_s2=E_o["E_s2"],
                log_pS=pS, log_pD=pD, log_pL=pL,
                transfer_mat=None, max_transfer_mat=m,
                device=device, dtype=dtype, pibar_mode='uniform')
            return compute_log_likelihood(Po["Pi"], E_o["E"], sb["root_clade_ids"]).sum().item()

        # FD: theta[13,1], Pi-only (E fixed)
        tp = theta.clone(); tp[IDX] += eps
        tm = theta.clone(); tm[IDX] -= eps
        fd_pi_only = (logL_pi_only(tp) - logL_pi_only(tm)) / (2 * eps)

        # FD: theta[13,1], full (including E re-solve)
        fd_full = (logL_full(tp) - logL_full(tm)) / (2 * eps)

        # Analytic
        Po = Pi_wave_forward(
            wave_layout=wl, species_helpers=sh,
            E=E_out["E"], Ebar=E_out["E_bar"],
            E_s1=E_out["E_s1"], E_s2=E_out["E_s2"],
            log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
            transfer_mat=None, max_transfer_mat=mt,
            device=device, dtype=dtype, pibar_mode='uniform')

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

        # Compute grad_theta_pi from pi_bwd
        grad_mt_total = pi_bwd['grad_max_transfer_mat'] + pi_bwd['grad_Ebar']
        theta_req = theta.detach().requires_grad_(True)
        with torch.enable_grad():
            pS_r, pD_r, pL_r, _, mt_r = extract_parameters_uniform(
                theta_req, unnorm_row_max, specieswise=True)
            param_loss = (
                (pS_r * pi_bwd['grad_log_pS']).sum() +
                (pD_r * pi_bwd['grad_log_pD']).sum() +
                (mt_r * grad_mt_total).sum()
            )
            grad_theta_pi = torch.autograd.grad(param_loss, theta_req)[0]

        # Full analytic (Pi + E adjoint)
        grad_theta_full, _ = _e_adjoint_and_theta_vjp(
            pi_bwd, E_out["E"], E_out["E_bar"], E_out["E_s1"], E_out["E_s2"],
            log_pS, log_pD, log_pL,
            mt, sh, wl["root_clade_ids"],
            theta, unnorm_row_max, specieswise=True,
            device=device, dtype=dtype,
            cg_tol=1e-10, cg_maxiter=1000,
            pibar_mode='uniform', ancestors_T=ancestors_T)

        gtheta_E = (grad_theta_full - grad_theta_pi)[IDX].item()
        gpi = grad_theta_pi[IDX].item()
        gfull = grad_theta_full[IDX].item()

        print(f"\nFamily {fi} (C={bi['ccp']['C']}):")
        print(f"  FD (Pi-only)   = {fd_pi_only:.8e}")
        print(f"  FD (full)      = {fd_full:.8e}")
        print(f"  FD (E-path)    = {fd_full - fd_pi_only:.8e}")
        print(f"  analytic Pi    = {gpi:.8e}  err vs FD(Pi-only) = {abs(gpi - fd_pi_only) / max(abs(fd_pi_only), 1e-8):.4e}")
        print(f"  analytic E     = {gtheta_E:.8e}  err vs FD(E-path) = {abs(gtheta_E - (fd_full - fd_pi_only)) / max(abs(fd_full - fd_pi_only), 1e-8):.4e}")
        print(f"  analytic total = {gfull:.8e}  err vs FD(full) = {abs(gfull - fd_full) / max(abs(fd_full), 1e-8):.4e}")


if __name__ == "__main__":
    main()
