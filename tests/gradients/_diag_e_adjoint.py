"""Diagnostic: instrument E adjoint for Family 1, specieswise uniform."""
import math, torch
from pathlib import Path
from torch import func as tfunc

from gpurec.core.preprocess_cpp import _load_extension
from gpurec.core.extract_parameters import extract_parameters_uniform
from gpurec.core.likelihood import E_fixed_point, E_step, compute_log_likelihood
from gpurec.core.forward import Pi_wave_forward
from gpurec.core.backward import Pi_wave_backward
from gpurec.core.scheduling import compute_clade_waves
from gpurec.core.batching import collate_gene_families, collate_wave, build_wave_layout
from gpurec.core.log2_utils import _safe_log2_internal as _safe_log2
from gpurec.core.terms import gather_E_children
from gpurec.optimization.linear_solvers import _cg, _gmres

_INV = 1.0 / math.log(2.0)
D, L, T = 0.05, 0.05, 0.05
S_IDX = 13


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
    sp_P_idx = sh['s_P_indexes']
    sp_c12_idx = sh['s_C12_indexes']

    E_out = E_fixed_point(
        species_helpers=sh, log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
        transfer_mat=None, max_transfer_mat=mt,
        max_iters=2000, tolerance=1e-10, warm_start_E=None,
        dtype=dtype, device=device, pibar_mode='uniform', ancestors_T=ancestors_T)
    E_star = E_out["E"]

    for fi in [0, 1, 2]:
        bi = batch_items[fi]
        sb = collate_gene_families([bi], dtype=dtype, device=device)
        w, p = compute_clade_waves(bi["ccp"])
        cw = collate_wave([w], [0])
        wl = build_wave_layout(
            waves=cw, phases=p, ccp_helpers=sb["ccp"],
            leaf_row_index=sb["leaf_row_index"], leaf_col_index=sb["leaf_col_index"],
            root_clade_ids=sb["root_clade_ids"], device=device, dtype=dtype)

        Po = Pi_wave_forward(
            wave_layout=wl, species_helpers=sh,
            E=E_star, Ebar=E_out["E_bar"],
            E_s1=E_out["E_s1"], E_s2=E_out["E_s2"],
            log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
            transfer_mat=None, max_transfer_mat=mt,
            device=device, dtype=dtype, pibar_mode='uniform')

        pi_bwd = Pi_wave_backward(
            wave_layout=wl,
            Pi_star_wave=Po["Pi_wave_ordered"],
            Pibar_star_wave=Po["Pibar_wave_ordered"],
            E=E_star, Ebar=E_out["E_bar"],
            E_s1=E_out["E_s1"], E_s2=E_out["E_s2"],
            log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
            max_transfer_mat=mt,
            species_helpers=sh,
            root_clade_ids_perm=wl["root_clade_ids"],
            device=device, dtype=dtype,
            neumann_terms=4, use_pruning=False,
            pibar_mode='uniform', ancestors_T=ancestors_T)

        # ═══ Manually compute each q_E component ═══
        n_fam = wl["root_clade_ids"].numel()  # = 1

        # 1. direct_dNLL_dE
        E_req_d = E_star.detach().requires_grad_(True)
        with torch.enable_grad():
            mean_E_exp = torch.exp2(E_req_d).mean(dim=-1)
            denom = torch.log2(1.0 - mean_E_exp)
            direct_dNLL_dE = torch.autograd.grad(n_fam * denom, E_req_d)[0]

        # 2. ebar_to_e
        E_req2 = E_star.detach().requires_grad_(True)
        with torch.enable_grad():
            mt_sq = mt.squeeze(-1) if mt.ndim > 1 else mt
            max_E = E_req2.max(dim=-1, keepdim=True).values
            expE = torch.exp2(E_req2 - max_E)
            expE_2d = expE.unsqueeze(0)
            row_sum = expE_2d.sum(dim=-1, keepdim=True)
            ancestor_sum = expE_2d @ ancestors_T
            Ebar_recomp = _safe_log2((row_sum - ancestor_sum).squeeze(0)) + max_E.squeeze(-1) + mt_sq
            ebar_to_e = torch.autograd.grad(
                Ebar_recomp, E_req2,
                grad_outputs=pi_bwd['grad_Ebar'],
            )[0]

        # 3. es_to_e
        E_req3 = E_star.detach().requires_grad_(True)
        with torch.enable_grad():
            E_s12 = gather_E_children(E_req3, sp_P_idx, sp_c12_idx)
            E_s1_r, E_s2_r = torch.chunk(E_s12, 2, dim=-1)
            E_s1_r = E_s1_r.view(E_req3.shape)
            E_s2_r = E_s2_r.view(E_req3.shape)
            total = (E_s1_r * pi_bwd['grad_E_s1']).sum() + (E_s2_r * pi_bwd['grad_E_s2']).sum()
            es_to_e = torch.autograd.grad(total, E_req3)[0]

        q_E = pi_bwd['grad_E'].clone() + direct_dNLL_dE + ebar_to_e + es_to_e

        print(f"\n{'='*60}")
        print(f"Family {fi} (C={bi['ccp']['C']})")
        print(f"{'='*60}")
        print(f"  q_E components at species {S_IDX}:")
        print(f"    grad_E       = {pi_bwd['grad_E'][S_IDX]:.8e}")
        print(f"    direct_dNLL  = {direct_dNLL_dE[S_IDX]:.8e}")
        print(f"    ebar_to_e    = {ebar_to_e[S_IDX]:.8e}")
        print(f"    es_to_e      = {es_to_e[S_IDX]:.8e}")
        print(f"    q_E total    = {q_E[S_IDX]:.8e}")
        print(f"    |q_E|_max    = {q_E.abs().max():.8e}")

        # ═══ Solve (I - G_E^T) w = q_E ═══
        def G_E_fun(E_in):
            return E_step(
                E_in, sp_P_idx, sp_c12_idx,
                log_pS, log_pD, log_pL, None, mt,
                pibar_mode='uniform', ancestors_T=ancestors_T,
            )[0]

        E_req_g = E_star.detach().requires_grad_(True)
        with torch.enable_grad():
            _, vjpG = tfunc.vjp(G_E_fun, E_req_g)

        nE = E_star.numel()
        q_flat = q_E.reshape(-1)

        def AG_flat(w_flat):
            wE = w_flat.view(E_star.shape).contiguous()
            gE, = vjpG(wE.clone())
            return (wE - gE).reshape(-1)

        w_flat, statsG, okG = _cg(AG_flat, q_flat, tol=1e-10, maxiter=1000)
        if not okG:
            w_flat, statsG = _gmres(AG_flat, q_flat, tol=1e-10, restart=40, maxiter=1000)
            print(f"  CG failed, used GMRES")

        wE = w_flat.view(E_star.shape)
        residual = AG_flat(w_flat) - q_flat
        print(f"  CG converged: {okG}, residual = {residual.norm():.4e}, w[{S_IDX}] = {wE[S_IDX]:.8e}")

        # ═══ Verify w via FD of total dNLL/dE ═══
        # FD: perturb E[s], re-run everything (E_fixed_point output + Pi + logL)
        eps = 1e-4
        E_p = E_star.clone(); E_p[S_IDX] += eps
        E_m = E_star.clone(); E_m[S_IDX] -= eps
        # Need to recompute Ebar, E_s1, E_s2 from perturbed E
        def logL_with_E(E_test):
            E_test_step = E_step(E_test, sp_P_idx, sp_c12_idx, log_pS, log_pD, log_pL,
                                 None, mt, pibar_mode='uniform', ancestors_T=ancestors_T)
            Ebar_t = E_test_step[3]
            E_s12_t = gather_E_children(E_test, sp_P_idx, sp_c12_idx)
            E_s1_t, E_s2_t = torch.chunk(E_s12_t, 2, dim=-1)
            E_s1_t = E_s1_t.view(E_test.shape)
            E_s2_t = E_s2_t.view(E_test.shape)
            Po_t = Pi_wave_forward(
                wave_layout=wl, species_helpers=sh,
                E=E_test, Ebar=Ebar_t,
                E_s1=E_s1_t, E_s2=E_s2_t,
                log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
                transfer_mat=None, max_transfer_mat=mt,
                device=device, dtype=dtype, pibar_mode='uniform')
            return compute_log_likelihood(Po_t["Pi"], E_test, sb["root_clade_ids"]).sum().item()

        fd_dNLL_dE_s = (logL_with_E(E_p) - logL_with_E(E_m)) / (2 * eps)
        print(f"  FD dNLL/dE[{S_IDX}] (single step)    = {fd_dNLL_dE_s:.8e}")
        print(f"  q_E[{S_IDX}]                          = {q_E[S_IDX]:.8e}")
        print(f"  q_E should ≈ FD for direct effect (not including E fixed-point)")

        # ═══ Compute full gtheta_E ═══
        theta_req2 = theta.detach().requires_grad_(True)
        with torch.enable_grad():
            pS2, pD2, pL2, _, mt2 = extract_parameters_uniform(
                theta_req2, unnorm_row_max, specieswise=True)
            E_from_theta = E_step(
                E_star.detach(), sp_P_idx, sp_c12_idx,
                pS2, pD2, pL2, None, mt2,
                pibar_mode='uniform', ancestors_T=ancestors_T,
            )[0]
            gtheta_E = torch.autograd.grad(E_from_theta, theta_req2, grad_outputs=wE)[0]

        print(f"  gtheta_E[13,1] = {gtheta_E[13, 1]:.8e}")

        # Also compute grad_theta_pi for reference
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

        print(f"  grad_theta_pi[13,1] = {grad_theta_pi[13, 1]:.8e}")
        print(f"  total[13,1] = {(grad_theta_pi + gtheta_E)[13, 1]:.8e}")


if __name__ == "__main__":
    main()
