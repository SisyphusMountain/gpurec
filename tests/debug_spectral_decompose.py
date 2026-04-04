"""Decompose J^T into its 4 paths and measure each one's spectral radius.

Goal: understand WHY rho(J^T) ≈ 0.97 when the TL2 weight is tiny.
"""
import torch
from tests.gradients.test_wave_gradient import _setup_uniform
from src.core.likelihood import (
    Pi_wave_forward, _self_loop_vjp_precompute, _self_loop_Jt_apply,
    _compute_Pibar_inline, _compute_dts_cross, NEG_INF,
)
from src.core.extract_parameters import extract_parameters_uniform
from src.core.likelihood import E_fixed_point

d = _setup_uniform("test_trees_20", n_families=1, dtype=torch.float64)
device, dtype = d['device'], d['dtype']
theta = d['theta'].clone()
pibar_mode = d['pibar_mode']
ancestors_T = d['ancestors_T']
sh = d['species_helpers']
wl = d['wave_layout']
S = sh['S']

log_pS, log_pD, log_pL, transfer_mat, mt = extract_parameters_uniform(
    theta, d['unnorm_row_max'], specieswise=False,
)
E_out = E_fixed_point(
    species_helpers=sh, log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
    transfer_mat=transfer_mat, max_transfer_mat=mt,
    max_iters=2000, tolerance=1e-10, warm_start_E=None,
    dtype=dtype, device=device, pibar_mode=pibar_mode, ancestors_T=ancestors_T,
)
Pi_out = Pi_wave_forward(
    wave_layout=wl, species_helpers=sh,
    E=E_out['E'], Ebar=E_out['E_bar'], E_s1=E_out['E_s1'], E_s2=E_out['E_s2'],
    log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
    transfer_mat=transfer_mat, max_transfer_mat=mt,
    device=device, dtype=dtype, pibar_mode=pibar_mode,
    local_tolerance=1e-10, local_iters=500,
)
Pi_star_wave = Pi_out['Pi_wave_ordered']
Pibar_star_wave = Pi_out['Pibar_wave_ordered']

sp_P_idx = sh['s_P_indexes']
sp_c12_idx = sh['s_C12_indexes']
p_cpu = sp_P_idx.cpu().long()
c_cpu = sp_c12_idx.cpu().long()
mask_c1 = p_cpu < S
sp_child1 = torch.full((S,), S, dtype=torch.long, device=device)
sp_child2 = torch.full((S,), S, dtype=torch.long, device=device)
sp_child1[p_cpu[mask_c1].to(device)] = c_cpu[mask_c1].to(device)
sp_child2[(p_cpu[~mask_c1] - S).to(device)] = c_cpu[~mask_c1].to(device)

wave_metas = wl['wave_metas']
wave_starts = wl['wave_starts']
E = E_out['E']
Ebar = E_out['E_bar']
DL_const = log_pD + log_pL
SL1_const = log_pS + E_out['E_s2']
SL2_const = log_pS + E_out['E_s1']

print(f"S = {S}")
print(f"log_pS = {float(log_pS):.6f}  (pS = {2**float(log_pS):.6f})")
print(f"log_pD = {float(log_pD):.6f}  (pD = {2**float(log_pD):.6f})")
print(f"log_pL = {float(log_pL):.6f}  (pL = {2**float(log_pL):.6f})")
print(f"E range: [{E.min():.4f}, {E.max():.4f}]  (exp2 range: [{E.min().exp2():.6f}, {E.max().exp2():.6f}])")
print(f"Ebar range: [{Ebar.min():.4f}, {Ebar.max():.4f}]")
print(f"mt range: [{mt.min():.4f}, {mt.max():.4f}]")
print()

# Pick the wave with highest spectral radius
for wave_idx in range(len(wave_metas)):
    meta = wave_metas[wave_idx]
    ws = wave_starts[wave_idx]
    W = meta['W']
    we = ws + W

    if W * S > 3000:
        continue

    Pi_W = Pi_star_wave[ws:we]
    Pibar_W = Pibar_star_wave[ws:we]

    if meta['has_splits']:
        dts_r = _compute_dts_cross(
            Pi_star_wave, Pibar_star_wave, meta,
            sp_child1, sp_child2, log_pD, log_pS, S, device, dtype
        )
    else:
        dts_r = None

    leaf_wt = torch.full((W, S), NEG_INF, device=device, dtype=dtype)
    leaf_row = wl['leaf_row_index'].to(device)
    leaf_col = wl['leaf_col_index'].to(device)
    mask_lr = (leaf_row >= ws) & (leaf_row < we)
    if mask_lr.any():
        leaf_wt[leaf_row[mask_lr] - ws, leaf_col[mask_lr]] = 0.0

    mt_w = mt.unsqueeze(0).expand(W, -1)
    DL_w = DL_const.unsqueeze(0).expand(W, -1)
    E_w = E.unsqueeze(0).expand(W, -1)
    Ebar_w = Ebar.unsqueeze(0).expand(W, -1)
    SL1_w = SL1_const.unsqueeze(0).expand(W, -1)
    SL2_w = SL2_const.unsqueeze(0).expand(W, -1)

    ingredients = _self_loop_vjp_precompute(
        Pi_W, Pibar_W, dts_r,
        mt_w, DL_w, Ebar_w, E_w, SL1_w, SL2_w,
        sp_child1, sp_child2, leaf_wt, S,
        pibar_mode, None, ancestors_T,
    )

    w_L = ingredients['w_L']
    w_terms = ingredients['w_terms']
    p_prime = ingredients['p_prime']

    # --- Per-path weight statistics ---
    # w_L is how much DTS_L contributes vs DTS_cross
    # w_terms[i] is how much term i contributes within DTS_L
    # Effective weight of Pibar path = w_L * w_terms[2]
    eff_pibar = w_L * w_terms[2]

    # --- Power iteration for full J^T ---
    n = W * S
    torch.manual_seed(42)
    v = torch.randn(W, S, device=device, dtype=dtype)
    for _ in range(300):
        v_new = _self_loop_Jt_apply(v, ingredients, sp_child1, sp_child2, S, W,
                                     pibar_mode, None, ancestors_T)
        rho_full = v_new.norm() / max(v.norm().item(), 1e-30)
        v = v_new / max(v_new.norm().item(), 1e-30)

    # --- Decompose: apply EACH path separately ---
    # Path 0+1 (diagonal): alpha * (w_terms[0] + w_terms[1])
    def apply_diag(v):
        alpha = v * w_L
        return alpha * (w_terms[0] + w_terms[1])

    # Path 2 (Pibar): alpha * w_terms[2] → Pibar VJP
    def apply_pibar(v):
        alpha = v * w_L
        v_Pibar = alpha * w_terms[2]
        u_d = v_Pibar * ingredients['pibar_inv_denom']
        A = u_d.sum(dim=1, keepdim=True)
        correction = (ancestors_T @ u_d.T).T
        return p_prime * (A - correction)

    # Path 3+4 (speciation scatter)
    def apply_spec(v):
        alpha = v * w_L
        result = torch.zeros_like(v)
        sc1_valid = ingredients.get('sc1_valid')
        sc2_valid = ingredients.get('sc2_valid')
        sc1_idx = ingredients.get('sc1_idx')
        sc2_idx = ingredients.get('sc2_idx')
        if sc1_valid is not None:
            src = alpha * w_terms[3]
            idx = sc1_idx.expand(W, -1) if sc1_idx.shape[0] == 1 else sc1_idx
            result.scatter_add_(1, idx, src[:, sc1_valid])
        if sc2_valid is not None:
            src = alpha * w_terms[4]
            idx = sc2_idx.expand(W, -1) if sc2_idx.shape[0] == 1 else sc2_idx
            result.scatter_add_(1, idx, src[:, sc2_valid])
        return result

    # Power iteration per path
    def power_iter(apply_fn, name, iters=300):
        torch.manual_seed(42)
        v = torch.randn(W, S, device=device, dtype=dtype)
        for _ in range(iters):
            v_new = apply_fn(v)
            rho = v_new.norm() / max(v.norm().item(), 1e-30)
            v = v_new / max(v_new.norm().item(), 1e-30)
        return rho

    rho_diag = power_iter(apply_diag, "diag")
    rho_pibar = power_iter(apply_pibar, "pibar")
    rho_spec = power_iter(apply_spec, "spec")

    # Also: what's the dominant diagonal weight?
    diag_weight = w_L * (w_terms[0] + w_terms[1])

    print(f"Wave {wave_idx}: W={W:4d}, has_splits={meta['has_splits']}")
    print(f"  w_L:           mean={w_L.mean():.6f}, max={w_L.max():.6f}, min={w_L.min():.6f}")
    print(f"  w_terms[0] DL: mean={w_terms[0].mean():.6f}, max={w_terms[0].max():.6f}")
    print(f"  w_terms[1] TL1:mean={w_terms[1].mean():.6f}, max={w_terms[1].max():.6f}")
    print(f"  w_terms[2] TL2:mean={w_terms[2].mean():.6f}, max={w_terms[2].max():.6f}")
    print(f"  w_terms[3] SL1:mean={w_terms[3].mean():.6f}, max={w_terms[3].max():.6f}")
    print(f"  w_terms[4] SL2:mean={w_terms[4].mean():.6f}, max={w_terms[4].max():.6f}")
    print(f"  w_terms[5] lf: mean={w_terms[5].mean():.6f}, max={w_terms[5].max():.6f}")
    print(f"  eff Pibar (w_L*w2): mean={eff_pibar.mean():.6f}, max={eff_pibar.max():.6f}")
    print(f"  diag weight (w_L*(w0+w1)): mean={diag_weight.mean():.6f}, max={diag_weight.max():.6f}")
    print(f"  ρ(full J^T)  = {rho_full:.6f}")
    print(f"  ρ(diag only) = {rho_diag:.6f}")
    print(f"  ρ(pibar only)= {rho_pibar:.6f}")
    print(f"  ρ(spec only) = {rho_spec:.6f}")

    if n <= 500:
        # Build full Jacobian to check eigenvalues
        J_full = torch.zeros(n, n, device=device, dtype=dtype)
        J_diag = torch.zeros(n, n, device=device, dtype=dtype)
        J_pibar = torch.zeros(n, n, device=device, dtype=dtype)
        J_spec = torch.zeros(n, n, device=device, dtype=dtype)
        for i in range(n):
            e_i = torch.zeros(W, S, device=device, dtype=dtype)
            e_i.view(-1)[i] = 1.0
            J_full[:, i] = _self_loop_Jt_apply(e_i, ingredients, sp_child1, sp_child2, S, W,
                                                 pibar_mode, None, ancestors_T).reshape(-1)
            J_diag[:, i] = apply_diag(e_i).reshape(-1)
            J_pibar[:, i] = apply_pibar(e_i).reshape(-1)
            J_spec[:, i] = apply_spec(e_i).reshape(-1)
        eigs_full = torch.linalg.eigvals(J_full).abs()
        eigs_diag = torch.linalg.eigvals(J_diag).abs()
        eigs_pibar = torch.linalg.eigvals(J_pibar).abs()
        eigs_spec = torch.linalg.eigvals(J_spec).abs()
        print(f"  [EXACT] ρ(full)  = {eigs_full.max():.6f}, top-5: {eigs_full.topk(min(5,n)).values.tolist()}")
        print(f"  [EXACT] ρ(diag)  = {eigs_diag.max():.6f}, top-5: {eigs_diag.topk(min(5,n)).values.tolist()}")
        print(f"  [EXACT] ρ(pibar) = {eigs_pibar.max():.6f}, top-5: {eigs_pibar.topk(min(5,n)).values.tolist()}")
        print(f"  [EXACT] ρ(spec)  = {eigs_spec.max():.6f}, top-5: {eigs_spec.topk(min(5,n)).values.tolist()}")
        # Check: J_full should equal J_diag + J_pibar + J_spec
        recon = J_diag + J_pibar + J_spec
        print(f"  Reconstruction error: {(J_full - recon).abs().max():.2e}")
    print()
