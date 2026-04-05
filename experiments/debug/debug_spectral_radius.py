"""Identify per-wave gradient error contribution for uniform mode.

Compare analytical backward's v_k (per-wave adjoint solution) against
a differentiable forward + autograd backward per wave.
"""
import torch
from tests.gradients.test_wave_gradient import _setup_uniform
from gpurec.core.forward import (
    Pi_wave_forward, _compute_dts_cross, _compute_Pibar_inline, NEG_INF,
)
from gpurec.core.backward import (
    Pi_wave_backward, _self_loop_differentiable, _dts_cross_differentiable,
    _self_loop_vjp_precompute, _self_loop_Jt_apply,
)
from gpurec.core.likelihood import E_fixed_point, compute_log_likelihood
from gpurec.core.extract_parameters import extract_parameters_uniform

d = _setup_uniform("test_trees_20", n_families=1, dtype=torch.float64)
device, dtype = d['device'], d['dtype']
theta = d['theta'].clone()
pibar_mode = d['pibar_mode']
ancestors_T = d['ancestors_T']
sh = d['species_helpers']
wl = d['wave_layout']
root_clade_ids = d['root_clade_ids']
S = sh['S']

# Forward pass
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

# Build species child indices
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

# Run the Neumann solve per wave and compute spectral radius
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
    mask = (leaf_row >= ws) & (leaf_row < we)
    if mask.any():
        leaf_wt[leaf_row[mask] - ws, leaf_col[mask]] = 0.0
    
    mt_w = mt.unsqueeze(0).expand(W, -1) if mt.ndim == 1 else mt
    DL_w = (log_pD + log_pL).unsqueeze(0).expand(W, -1) if log_pD.ndim == 0 else (log_pD + log_pL)
    E_w = E_out['E'].unsqueeze(0).expand(W, -1) if E_out['E'].ndim == 1 else E_out['E']
    Ebar_w = E_out['E_bar'].unsqueeze(0).expand(W, -1) if E_out['E_bar'].ndim == 1 else E_out['E_bar']
    SL1_w = (log_pS + E_out['E_s2']).unsqueeze(0).expand(W, -1) if (log_pS + E_out['E_s2']).ndim == 1 else (log_pS + E_out['E_s2'])
    SL2_w = (log_pS + E_out['E_s1']).unsqueeze(0).expand(W, -1) if (log_pS + E_out['E_s1']).ndim == 1 else (log_pS + E_out['E_s1'])
    
    ingredients = _self_loop_vjp_precompute(
        Pi_W_star, Pibar_W_star, dts_r,
        mt_w, DL_w, Ebar_w, E_w, SL1_w, SL2_w,
        sp_child1, sp_child2, leaf_wt, S,
        pibar_mode, transfer_mat, ancestors_T,
    )
    
    # Estimate spectral radius using power iteration
    torch.manual_seed(42)
    v = torch.randn(W, S, device=device, dtype=dtype)
    for _ in range(100):
        v_new = _self_loop_Jt_apply(
            v, ingredients, sp_child1, sp_child2, S, W,
            pibar_mode, transfer_mat, ancestors_T,
        )
        rho_est = v_new.norm() / max(v.norm().item(), 1e-30)
        v = v_new / max(v_new.norm().item(), 1e-30)
    
    print(f"Wave {wave_idx}: W={W:4d}, ρ ≈ {rho_est:.6f}", end="")
    if rho_est > 0.9:
        print(f"  *** HIGH ***", end="")
    print()
