"""Check w_L (self-loop weight) per wave, and verify the spectral radius
of the ACTUAL operator J^T used in the Neumann/GMRES solve matches what
the forward sees."""
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

for wave_idx in range(len(wave_metas)):
    meta = wave_metas[wave_idx]
    ws = wave_starts[wave_idx]
    W = meta['W']
    we = ws + W

    Pi_W_star = Pi_star_wave[ws:we]
    Pibar_W_star = Pibar_star_wave[ws:we]

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
        Pi_W_star, Pibar_W_star, dts_r,
        mt_w, DL_w, Ebar_w, E_w, SL1_w, SL2_w,
        sp_child1, sp_child2, leaf_wt, S,
        pibar_mode, None, ancestors_T,
    )

    w_L = ingredients['w_L']
    has_dts_cross = dts_r is not None
    n_leaves = mask_lr.sum().item()

    print(f"Wave {wave_idx}: W={W:4d}, has_splits={has_dts_cross}, n_leaves={n_leaves}")
    print(f"  w_L: mean={w_L.mean():.6f}, max={w_L.max():.6f}, min={w_L.min():.6f}")
    
    # Also check w_terms[2] (the Pibar+E weight, which drives the high spectral radius)
    w2 = ingredients['w_terms'][2]
    print(f"  w_terms[2] (Pibar+E): mean={w2.mean():.6f}, max={w2.max():.6f}")
    
    # Product w_L * w_terms[2] is the effective Pibar contribution
    eff = w_L * w2
    print(f"  w_L * w_terms[2]: mean={eff.mean():.6f}, max={eff.max():.6f}")

    # Spectral radius of J^T via power iteration
    if W * S <= 5000:
        torch.manual_seed(42)
        v = torch.randn(W, S, device=device, dtype=dtype)
        for _ in range(200):
            v_new = _self_loop_Jt_apply(
                v, ingredients, sp_child1, sp_child2, S, W,
                pibar_mode, None, ancestors_T,
            )
            rho_est = v_new.norm() / max(v.norm().item(), 1e-30)
            v = v_new / max(v_new.norm().item(), 1e-30)
        print(f"  ρ(J^T) = {rho_est:.6f}")
    print()
