"""Diagnostic: FD-verify each component of q_E for Family 1."""
import math, torch
from pathlib import Path

from gpurec.core.preprocess_cpp import _load_extension
from gpurec.core.extract_parameters import extract_parameters_uniform
from gpurec.core.likelihood import E_fixed_point, E_step, compute_log_likelihood
from gpurec.core.forward import Pi_wave_forward
from gpurec.core.backward import Pi_wave_backward
from gpurec.core.scheduling import compute_clade_waves
from gpurec.core.batching import collate_gene_families, collate_wave, build_wave_layout
from gpurec.core.terms import gather_E_children
from gpurec.core.log2_utils import logsumexp2

_INV = 1.0 / math.log(2.0)
D, L, T = 0.05, 0.05, 0.05
FI = 1  # family index
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
    Ebar = E_out["E_bar"]
    E_s1 = E_out["E_s1"]
    E_s2 = E_out["E_s2"]

    for FI_local in range(3):
        bi = batch_items[FI_local]
    sb = collate_gene_families([bi], dtype=dtype, device=device)
    w_sched, p = compute_clade_waves(bi["ccp"])
    cw = collate_wave([w_sched], [0])
    wl = build_wave_layout(
        waves=cw, phases=p, ccp_helpers=sb["ccp"],
        leaf_row_index=sb["leaf_row_index"], leaf_col_index=sb["leaf_col_index"],
        root_clade_ids=sb["root_clade_ids"], device=device, dtype=dtype)

    def run_pi_and_logL(E_in, Ebar_in, E_s1_in, E_s2_in):
        """Run Pi forward + logL."""
        Po = Pi_wave_forward(
            wave_layout=wl, species_helpers=sh,
            E=E_in, Ebar=Ebar_in, E_s1=E_s1_in, E_s2=E_s2_in,
            log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
            transfer_mat=None, max_transfer_mat=mt,
            device=device, dtype=dtype, pibar_mode='uniform')
        return compute_log_likelihood(Po["Pi"], E_in, sb["root_clade_ids"]).sum().item()

    def run_pi_numerator_only(E_in, Ebar_in, E_s1_in, E_s2_in):
        """Run Pi forward, return numerator only (no denominator)."""
        Po = Pi_wave_forward(
            wave_layout=wl, species_helpers=sh,
            E=E_in, Ebar=Ebar_in, E_s1=E_s1_in, E_s2=E_s2_in,
            log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
            transfer_mat=None, max_transfer_mat=mt,
            device=device, dtype=dtype, pibar_mode='uniform')
        root_probs = Po["Pi"][sb["root_clade_ids"], :]
        return -(logsumexp2(root_probs, dim=-1) - math.log2(S)).sum().item()

    base_nll = run_pi_and_logL(E_star, Ebar, E_s1, E_s2)
    base_num = run_pi_numerator_only(E_star, Ebar, E_s1, E_s2)

    # Pi backward
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

    print(f"Family {FI} (C={bi['ccp']['C']}), species {S_IDX}")
    print(f"Base NLL = {base_nll:.8f}, numerator = {base_num:.8f}")

    # ══ Component 1: grad_E (direct E in DL and TL2, numerator only) ══
    # FD: perturb E[13], keep Ebar/E_s1/E_s2 fixed
    E_p = E_star.clone(); E_p[S_IDX] += eps
    E_m = E_star.clone(); E_m[S_IDX] -= eps

    fd_E_num = (run_pi_numerator_only(E_p, Ebar, E_s1, E_s2) -
                run_pi_numerator_only(E_m, Ebar, E_s1, E_s2)) / (2 * eps)
    ana_E = pi_bwd['grad_E'][S_IDX].item()
    print(f"\n1. grad_E[{S_IDX}] (E direct, numerator only):")
    print(f"   analytic = {ana_E:.8e}")
    print(f"   FD       = {fd_E_num:.8e}")
    print(f"   err      = {abs(ana_E - fd_E_num) / max(abs(fd_E_num), 1e-8):.4e}")

    # ══ Component 2: grad_Ebar (Ebar in TL1, numerator only) ══
    # FD: perturb Ebar[13], keep E/E_s1/E_s2 fixed
    Ebar_p = Ebar.clone(); Ebar_p[S_IDX] += eps
    Ebar_m = Ebar.clone(); Ebar_m[S_IDX] -= eps
    fd_Ebar_num = (run_pi_numerator_only(E_star, Ebar_p, E_s1, E_s2) -
                   run_pi_numerator_only(E_star, Ebar_m, E_s1, E_s2)) / (2 * eps)
    ana_Ebar = pi_bwd['grad_Ebar'][S_IDX].item()
    print(f"\n2. grad_Ebar[{S_IDX}] (Ebar in TL1, numerator only):")
    print(f"   analytic = {ana_Ebar:.8e}")
    print(f"   FD       = {fd_Ebar_num:.8e}")
    print(f"   err      = {abs(ana_Ebar - fd_Ebar_num) / max(abs(fd_Ebar_num), 1e-8):.4e}")

    # ══ Component 3: grad_E_s1/E_s2 (species children, numerator only) ══
    # FD: perturb E_s1[13]
    E_s1_p = E_s1.clone(); E_s1_p[S_IDX] += eps
    E_s1_m = E_s1.clone(); E_s1_m[S_IDX] -= eps
    fd_Es1_num = (run_pi_numerator_only(E_star, Ebar, E_s1_p, E_s2) -
                  run_pi_numerator_only(E_star, Ebar, E_s1_m, E_s2)) / (2 * eps)
    ana_Es1 = pi_bwd['grad_E_s1'][S_IDX].item()
    # FD: perturb E_s2[13]
    E_s2_p = E_s2.clone(); E_s2_p[S_IDX] += eps
    E_s2_m = E_s2.clone(); E_s2_m[S_IDX] -= eps
    fd_Es2_num = (run_pi_numerator_only(E_star, Ebar, E_s1, E_s2_p) -
                  run_pi_numerator_only(E_star, Ebar, E_s1, E_s2_m)) / (2 * eps)
    ana_Es2 = pi_bwd['grad_E_s2'][S_IDX].item()
    print(f"\n3. grad_E_s1[{S_IDX}] and grad_E_s2[{S_IDX}]:")
    print(f"   E_s1: analytic = {ana_Es1:.8e}, FD = {fd_Es1_num:.8e}, err = {abs(ana_Es1 - fd_Es1_num) / max(abs(fd_Es1_num), 1e-8):.4e}")
    print(f"   E_s2: analytic = {ana_Es2:.8e}, FD = {fd_Es2_num:.8e}, err = {abs(ana_Es2 - fd_Es2_num) / max(abs(fd_Es2_num), 1e-8):.4e}")

    # ══ Component 4: direct_dNLL_dE (denominator) ══
    # FD: perturb E[13] in denominator only
    denom_base = math.log2(1 - torch.exp2(E_star).mean(dim=-1).item())
    E_p = E_star.clone(); E_p[S_IDX] += eps
    denom_p = math.log2(1 - torch.exp2(E_p).mean(dim=-1).item())
    E_m = E_star.clone(); E_m[S_IDX] -= eps
    denom_m = math.log2(1 - torch.exp2(E_m).mean(dim=-1).item())
    fd_denom = (denom_p - denom_m) / (2 * eps)
    E_req_d = E_star.detach().requires_grad_(True)
    with torch.enable_grad():
        mean_E_exp = torch.exp2(E_req_d).mean(dim=-1)
        denom_t = torch.log2(1.0 - mean_E_exp)
        direct_dNLL_dE = torch.autograd.grad(1 * denom_t, E_req_d)[0]
    print(f"\n4. direct_dNLL_dE[{S_IDX}] (denominator):")
    print(f"   analytic = {direct_dNLL_dE[S_IDX]:.8e}")
    print(f"   FD       = {fd_denom:.8e}")
    print(f"   err      = {abs(direct_dNLL_dE[S_IDX].item() - fd_denom) / max(abs(fd_denom), 1e-8):.4e}")

    # ══ Total q_E vs FD of full NLL ══
    print(f"\n══ Summary ══")
    q_E_s = (pi_bwd['grad_E'][S_IDX].item() +
             direct_dNLL_dE[S_IDX].item())
    print(f"  grad_E + direct_dNLL_dE = {q_E_s:.8e}")

    # FD: total NLL change when E[13] perturbed (with Ebar, E_s1/E_s2 recomputed)
    def logL_full_E_perturb(E_test):
        E_t_step = E_step(E_test, sp_P_idx, sp_c12_idx, log_pS, log_pD, log_pL,
                          None, mt, pibar_mode='uniform', ancestors_T=ancestors_T)
        Ebar_t = E_t_step[3]
        E_s12_t = gather_E_children(E_test, sp_P_idx, sp_c12_idx)
        E_s1_t, E_s2_t = torch.chunk(E_s12_t, 2, dim=-1)
        return run_pi_and_logL(E_test, Ebar_t, E_s1_t.view(E_test.shape), E_s2_t.view(E_test.shape))

    E_p = E_star.clone(); E_p[S_IDX] += eps
    E_m = E_star.clone(); E_m[S_IDX] -= eps
    fd_full = (logL_full_E_perturb(E_p) - logL_full_E_perturb(E_m)) / (2 * eps)

    # Also compute FD separately: only numerator, with Ebar+Es recomputed
    def num_full_E_perturb(E_test):
        E_t_step = E_step(E_test, sp_P_idx, sp_c12_idx, log_pS, log_pD, log_pL,
                          None, mt, pibar_mode='uniform', ancestors_T=ancestors_T)
        Ebar_t = E_t_step[3]
        E_s12_t = gather_E_children(E_test, sp_P_idx, sp_c12_idx)
        E_s1_t, E_s2_t = torch.chunk(E_s12_t, 2, dim=-1)
        return run_pi_numerator_only(E_test, Ebar_t, E_s1_t.view(E_test.shape), E_s2_t.view(E_test.shape))

    fd_num_full = (num_full_E_perturb(E_p) - num_full_E_perturb(E_m)) / (2 * eps)

    print(f"  FD dNLL/dE[{S_IDX}] (full, num+denom)     = {fd_full:.8e}")
    print(f"  FD d(num)/dE[{S_IDX}] (Ebar+Es recomputed) = {fd_num_full:.8e}")
    print(f"  FD d(denom)/dE[{S_IDX}]                    = {fd_denom:.8e}")
    print(f"  FD(full) should ≈ FD(num) + FD(denom) = {fd_num_full + fd_denom:.8e}")

    # ebar_to_e and es_to_e
    from gpurec.core.log2_utils import _safe_log2_internal as _safe_log2
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
            Ebar_recomp, E_req2, grad_outputs=pi_bwd['grad_Ebar'])[0]

    E_req3 = E_star.detach().requires_grad_(True)
    with torch.enable_grad():
        E_s12_r = gather_E_children(E_req3, sp_P_idx, sp_c12_idx)
        E_s1_r, E_s2_r = torch.chunk(E_s12_r, 2, dim=-1)
        E_s1_r = E_s1_r.view(E_req3.shape)
        E_s2_r = E_s2_r.view(E_req3.shape)
        total_es = (E_s1_r * pi_bwd['grad_E_s1']).sum() + (E_s2_r * pi_bwd['grad_E_s2']).sum()
        es_to_e = torch.autograd.grad(total_es, E_req3)[0]

    q_E_total = (pi_bwd['grad_E'][S_IDX].item() + direct_dNLL_dE[S_IDX].item() +
                 ebar_to_e[S_IDX].item() + es_to_e[S_IDX].item())
    print(f"\n  Analytic q_E[{S_IDX}] breakdown:")
    print(f"    grad_E       = {pi_bwd['grad_E'][S_IDX]:.8e}")
    print(f"    direct_dNLL  = {direct_dNLL_dE[S_IDX]:.8e}")
    print(f"    ebar_to_e    = {ebar_to_e[S_IDX]:.8e}")
    print(f"    es_to_e      = {es_to_e[S_IDX]:.8e}")
    print(f"    TOTAL q_E    = {q_E_total:.8e}")
    print(f"    FD (full)    = {fd_full:.8e}")
    print(f"    GAP          = {fd_full - q_E_total:.8e}")

    # ══ The gap should come from the mt→Ebar path ══
    # When E changes, Ebar changes. This affects mt through nothing (Ebar doesn't contain mt).
    # But wait: the FD recomputes Ebar from the perturbed E. The analytic captures this
    # through ebar_to_e. So why is there a gap?

    # Let me also verify: what's the total numerator sensitivity to E[13] with only direct E path?
    fd_E_direct = (run_pi_numerator_only(E_p, Ebar, E_s1, E_s2) -
                   run_pi_numerator_only(E_m, Ebar, E_s1, E_s2)) / (2 * eps)
    print(f"\n  FD d(num)/dE[{S_IDX}] (E direct only, no Ebar/Es update) = {fd_E_direct:.8e}")
    print(f"  pi_bwd['grad_E'][{S_IDX}]                                  = {pi_bwd['grad_E'][S_IDX]:.8e}")
    print(f"  Difference (should capture Ebar+Es indirect paths via FD) = {fd_num_full - fd_E_direct:.8e}")

    # What does ebar_to_e + es_to_e look like?
    ebar_es_sum = ebar_to_e[S_IDX].item() + es_to_e[S_IDX].item()
    print(f"  ebar_to_e + es_to_e = {ebar_es_sum:.8e}")
    print(f"  FD indirect = {fd_num_full - fd_E_direct:.8e}")
    print(f"  err = {abs(ebar_es_sum - (fd_num_full - fd_E_direct)) / max(abs(fd_num_full - fd_E_direct), 1e-8):.4e}")


if __name__ == "__main__":
    main()
