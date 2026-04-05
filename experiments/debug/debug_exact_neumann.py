"""Verify Neumann solution for a single wave by comparing with exact (I-J^T)^{-1}.

Use torch.func.jacobian to get the full Jacobian of the self-loop,
then compute exact (I-J^T)^{-1} @ rhs.
"""
import torch
from tests.gradients.test_wave_gradient import _setup_uniform
from gpurec.core.forward import Pi_wave_forward, _compute_dts_cross, NEG_INF
from gpurec.core.backward import (
    Pi_wave_backward, _self_loop_differentiable,
    _self_loop_vjp_precompute, _self_loop_Jt_apply,
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
root_clade_ids = d['root_clade_ids']
S = sh['S']

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

# Forward
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

wave_metas = wl['wave_metas']
wave_starts = wl['wave_starts']

# Test waves with moderate ρ
for wave_idx in [6, 7]:  # These had ρ ≈ 0.96, 0.97
    meta = wave_metas[wave_idx]
    ws = wave_starts[wave_idx]
    W = meta['W']
    we = ws + W
    n = W * S
    
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
    leaf_wt = log_pS + leaf_wt
    
    # Precompute
    mt_w = mt.unsqueeze(0).expand(W, -1)
    DL_w = (1.0 + log_pD + E_out['E']).unsqueeze(0).expand(W, -1)
    E_w = E_out['E'].unsqueeze(0).expand(W, -1)
    Ebar_w = E_out['E_bar'].unsqueeze(0).expand(W, -1)
    SL1_w = (log_pS + E_out['E_s2']).unsqueeze(0).expand(W, -1)
    SL2_w = (log_pS + E_out['E_s1']).unsqueeze(0).expand(W, -1)
    
    ingredients = _self_loop_vjp_precompute(
        Pi_W_star, Pibar_W_star, dts_r,
        mt_w, DL_w, Ebar_w, E_w, SL1_w, SL2_w,
        sp_child1, sp_child2, leaf_wt, S,
        pibar_mode, transfer_mat, ancestors_T,
    )
    
    # Get full Jacobian via autograd
    ancestors_T_dense = ancestors_T.to_dense()
    def self_loop_fn(Pi_flat):
        Pi = Pi_flat.reshape(W, S)
        out = _self_loop_differentiable(
            Pi, mt, 1.0 + log_pD + E_out['E'], E_out['E_bar'], E_out['E'],
            log_pS + E_out['E_s2'], log_pS + E_out['E_s1'],
            sp_child1, sp_child2, leaf_wt, dts_r, S,
            pibar_mode=pibar_mode,
            transfer_mat_T=transfer_mat,
            ancestors_T=ancestors_T_dense,
        )
        return out.reshape(-1)
    
    J = torch.func.jacrev(self_loop_fn)(Pi_W_star.reshape(-1))  # [n, n]
    
    # Spectral radius
    eigvals = torch.linalg.eigvals(J)
    rho = eigvals.abs().max().item()
    print(f"\nWave {wave_idx}: W={W}, n={n}, ρ(J) = {rho:.6f}")
    
    # Random RHS
    torch.manual_seed(42)
    rhs = torch.randn(W, S, device=device, dtype=dtype)
    
    # Exact solution: (I - J^T)^{-1} @ rhs
    I_minus_JT = torch.eye(n, device=device, dtype=dtype) - J.T
    v_exact = torch.linalg.solve(I_minus_JT, rhs.reshape(-1)).reshape(W, S)
    
    # Neumann solution (50 terms)
    v_neumann = rhs.clone()
    term = rhs.clone()
    for i in range(50):
        term = _self_loop_Jt_apply(
            term, ingredients, sp_child1, sp_child2, S, W,
            pibar_mode, transfer_mat, ancestors_T,
        )
        v_neumann = v_neumann + term
    
    rel_err = (v_neumann - v_exact).norm() / v_exact.norm()
    print(f"  Neumann(50) vs exact: rel_err = {rel_err:.6e}")
    print(f"  ||v_exact|| = {v_exact.norm():.4e}, ||v_neumann|| = {v_neumann.norm():.4e}")
    
    # Also check: Neumann with 200 terms
    v_neumann200 = rhs.clone()
    term = rhs.clone()
    for i in range(200):
        term = _self_loop_Jt_apply(
            term, ingredients, sp_child1, sp_child2, S, W,
            pibar_mode, transfer_mat, ancestors_T,
        )
        v_neumann200 = v_neumann200 + term
    
    rel_err200 = (v_neumann200 - v_exact).norm() / v_exact.norm()
    print(f"  Neumann(200) vs exact: rel_err = {rel_err200:.6e}")
    
    # GMRES solution
    from gpurec.core.backward import _gmres_self_loop_solve
    v_gmres = _gmres_self_loop_solve(
        rhs, ingredients, sp_child1, sp_child2, S, W,
        pibar_mode, transfer_mat, ancestors_T,
        max_iters=100, tol=1e-12,
    )
    rel_err_gmres = (v_gmres - v_exact).norm() / v_exact.norm()
    print(f"  GMRES(100) vs exact: rel_err = {rel_err_gmres:.6e}")
