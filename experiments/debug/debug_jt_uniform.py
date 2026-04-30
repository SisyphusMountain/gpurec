"""Verify _self_loop_Jt_apply for uniform mode against torch.func.vjp (autograd)."""
import torch
from tests.gradients.test_wave_gradient import _setup_uniform
from gpurec.core.forward import Pi_wave_forward, NEG_INF
from gpurec.core.backward import (
    _self_loop_vjp_precompute, _self_loop_Jt_apply, _self_loop_differentiable,
)
from gpurec.core.likelihood import E_fixed_point
from gpurec.core.extract_parameters import extract_parameters_uniform

d = _setup_uniform("test_trees_20", n_families=1, dtype=torch.float64)
device, dtype = d['device'], d['dtype']
theta = d['theta'].clone()
pibar_mode = d['pibar_mode']
ancestors_T = d['ancestors_T']
sh = d['species_helpers']
wl = d['wave_layout']
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
sp_child1_cpu = torch.full((S,), S, dtype=torch.long)
sp_child2_cpu = torch.full((S,), S, dtype=torch.long)
sp_child1_cpu[p_cpu[mask_c1]] = c_cpu[mask_c1]
sp_child2_cpu[p_cpu[~mask_c1] - S] = c_cpu[~mask_c1]
sp_child1 = sp_child1_cpu.to(device)
sp_child2 = sp_child2_cpu.to(device)

# Test on a small wave
wave_metas = wl['wave_metas']
wave_starts = wl['wave_starts']

for wave_idx in range(min(len(wave_metas), 5)):
    meta = wave_metas[wave_idx]
    ws = wave_starts[wave_idx]
    W = meta['W']
    we = ws + W
    
    if W > 50:
        print(f"Wave {wave_idx}: W={W} (too large, skipping)")
        continue
    
    Pi_W_star = Pi_star_wave[ws:we]
    Pibar_W_star = Pibar_star_wave[ws:we]
    
    # Get DTS_r for this wave
    if meta['has_splits']:
        # Need to compute dts_r...
        from gpurec.core.forward import _compute_dts_cross
        dts_r = _compute_dts_cross(
            Pi_star_wave, Pibar_star_wave, meta,
            sp_child1, sp_child2, log_pD, log_pS, S, device, dtype
        )
    else:
        dts_r = None
    
    # Precompute
    leaf_wt = torch.full((W, S), NEG_INF, device=device, dtype=dtype)
    leaf_row = wl['leaf_row_index'].to(device)
    leaf_col = wl['leaf_col_index'].to(device)
    mask = (leaf_row >= ws) & (leaf_row < we)
    if mask.any():
        leaf_wt[leaf_row[mask] - ws, leaf_col[mask]] = 0.0
    
    mt_w = mt.unsqueeze(0).expand(W, -1) if mt.ndim == 1 else mt[ws:we] if mt.ndim == 2 else mt
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
    
    # Random cotangent vector
    torch.manual_seed(42 + wave_idx)
    v = torch.randn(W, S, device=device, dtype=dtype)
    
    # Analytical J^T @ v
    Jt_v_analytical = _self_loop_Jt_apply(
        v, ingredients, sp_child1, sp_child2, S, W,
        pibar_mode, transfer_mat, ancestors_T,
    )
    
    # Autograd J^T @ v via torch.func.vjp
    def self_loop_fn(Pi):
        return _self_loop_differentiable(
            Pi, mt if mt.ndim == 1 else mt_w,
            DL_w[0] if DL_w.shape[0] == W else DL_w,
            Ebar_w[0] if Ebar_w.shape[0] == W else Ebar_w,
            E_w[0] if E_w.shape[0] == W else E_w,
            SL1_w[0] if SL1_w.shape[0] == W else SL1_w,
            SL2_w[0] if SL2_w.shape[0] == W else SL2_w,
            sp_child1, sp_child2, leaf_wt, dts_r, S,
            pibar_mode=pibar_mode,
            transfer_mat_T=transfer_mat,
            ancestors_T=ancestors_T,
        )
    
    _, vjp_fn = torch.func.vjp(self_loop_fn, Pi_W_star)
    Jt_v_autograd = vjp_fn(v)[0]
    
    rel_err = (Jt_v_analytical - Jt_v_autograd).norm() / max(Jt_v_autograd.norm().item(), 1e-30)
    max_abs_err = (Jt_v_analytical - Jt_v_autograd).abs().max().item()
    print(f"Wave {wave_idx}: W={W}, ||Jt_analytical - Jt_autograd||/||Jt_autograd|| = {rel_err:.6e}, "
          f"max_abs = {max_abs_err:.6e}")
    
    if rel_err > 1e-3:
        # Debug: check per-term
        # Check which species have the largest discrepancy
        diff = (Jt_v_analytical - Jt_v_autograd).abs()
        max_per_s = diff.max(dim=0).values
        top5 = torch.topk(max_per_s, min(5, S)).indices
        print(f"  Top-5 species with largest error: {top5.tolist()}")
        for s_idx in top5:
            print(f"    s={s_idx.item()}: analytical={Jt_v_analytical[:,s_idx].tolist()[:3]}..., "
                  f"autograd={Jt_v_autograd[:,s_idx].tolist()[:3]}...")
