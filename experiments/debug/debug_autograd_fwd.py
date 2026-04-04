"""Compare Pi_wave_backward grad_log_pD against autograd through differentiable forward.

Key test: use _self_loop_differentiable and _dts_cross_differentiable to build
a fully differentiable forward pass, then compare autograd gradients against
the analytical backward.
"""
import torch
from tests.gradients.test_wave_gradient import _setup_uniform
from src.core.likelihood import (
    Pi_wave_forward, Pi_wave_backward, compute_log_likelihood,
    _self_loop_differentiable, _dts_cross_differentiable,
    _compute_dts_cross, NEG_INF,
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

E = E_out['E']
Ebar = E_out['E_bar']
E_s1 = E_out['E_s1']
E_s2 = E_out['E_s2']

# Make mt require grad to track d(NLL)/d(mt) through autograd
mt_grad = mt.clone().detach().requires_grad_(True)

# Build a differentiable forward pass wave-by-wave
C = wl['ccp_helpers']['C']
Pi = torch.full((C, S), torch.finfo(dtype).min, dtype=dtype, device=device)
Pibar = torch.full((C, S), NEG_INF, dtype=dtype, device=device)

leaf_row = wl['leaf_row_index'].to(device)
leaf_col = wl['leaf_col_index'].to(device)
Pi = Pi.clone()
Pi[leaf_row, leaf_col] = 0.0

for wave_idx in range(len(wave_metas)):
    meta = wave_metas[wave_idx]
    ws = wave_starts[wave_idx]
    W = meta['W']
    we = ws + W
    
    # Leaf weights for this wave
    leaf_wt = torch.full((W, S), NEG_INF, device=device, dtype=dtype)
    mask = (leaf_row >= ws) & (leaf_row < we)
    if mask.any():
        leaf_wt[leaf_row[mask] - ws, leaf_col[mask]] = 0.0
    
    # DTS cross (not differentiable — uses converged Pi/Pibar of parent waves)
    if meta['has_splits']:
        dts_r = _dts_cross_differentiable(
            Pi, Pibar, meta, sp_child1, sp_child2,
            log_pD, log_pS, S, device, dtype,
        )
    else:
        dts_r = None
    
    # Self-loop: run to convergence differentiably
    Pi_W = Pi[ws:we].clone()
    for it in range(200):
        Pi_new = _self_loop_differentiable(
            Pi_W, mt_grad, log_pD + log_pL, Ebar, E, log_pS + E_s2, log_pS + E_s1,
            sp_child1, sp_child2, leaf_wt, dts_r, S,
            pibar_mode=pibar_mode,
            transfer_mat_T=transfer_mat,
            ancestors_T=ancestors_T,
        )
        diff = (Pi_new - Pi_W).abs()
        significant = Pi_new > (Pi_new.max(dim=1, keepdim=True).values - 50)
        if not significant.any() or diff[significant].max().item() < 1e-10:
            break
        Pi_W = Pi_new
    
    # Store converged values (keeping gradient tape)
    Pi = Pi.clone()
    Pi[ws:we] = Pi_W
    
    # Recompute Pibar from converged Pi  
    Pi_max = Pi_W.max(dim=1, keepdim=True).values
    Pi_exp = torch.exp2(Pi_W - Pi_max)
    row_sum = Pi_exp.sum(dim=1, keepdim=True)
    ancestor_sum = Pi_exp @ ancestors_T
    from src.core.likelihood import _safe_log2
    Pibar_W = _safe_log2(row_sum - ancestor_sum) + Pi_max + mt_grad
    Pibar = Pibar.clone()
    Pibar[ws:we] = Pibar_W

# Compute NLL
logL = compute_log_likelihood(Pi, E, root_clade_ids).sum()
nll = logL.item()
print(f"Differentiable forward NLL = {nll:.6f}")

# Autograd backward
logL.backward()
grad_mt_autograd = mt_grad.grad.clone()
print(f"Autograd d(NLL)/d(mt) sum = {grad_mt_autograd.sum().item():.8e}")

# Now compare with analytical backward
Pi_out = Pi_wave_forward(
    wave_layout=wl, species_helpers=sh,
    E=E, Ebar=Ebar, E_s1=E_s1, E_s2=E_s2,
    log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
    transfer_mat=transfer_mat, max_transfer_mat=mt,
    device=device, dtype=dtype, pibar_mode=pibar_mode,
    local_tolerance=1e-10, local_iters=500,
)
bw = Pi_wave_backward(
    wave_layout=wl,
    Pi_star_wave=Pi_out['Pi_wave_ordered'],
    Pibar_star_wave=Pi_out['Pibar_wave_ordered'],
    E=E, Ebar=Ebar, E_s1=E_s1, E_s2=E_s2,
    log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
    max_transfer_mat=mt,
    species_helpers=sh,
    root_clade_ids_perm=wl['root_clade_ids'],
    device=device, dtype=dtype,
    neumann_terms=50,
    use_pruning=False,
    pibar_mode=pibar_mode,
    ancestors_T=ancestors_T,
)

print(f"Analytical d(NLL)/d(mt) sum = {bw['grad_max_transfer_mat'].sum().item():.8e}")
print(f"Analytical grad_log_pD = {float(bw['grad_log_pD']):.8e}")
print(f"Analytical grad_log_pS = {float(bw['grad_log_pS']):.8e}")

# Per-species comparison
print("\nPer-species mt gradient comparison:")
for s in range(min(S, 10)):
    an = float(bw['grad_max_transfer_mat'][s]) if bw['grad_max_transfer_mat'].ndim > 0 else float(bw['grad_max_transfer_mat'])
    au = float(grad_mt_autograd[s])
    err = abs(an - au) / max(abs(au), 1e-30) if abs(au) > 1e-15 else abs(an - au)
    print(f"  s={s}: autograd={au:.6e}, analytical={an:.6e}, rel_err={err:.4e}")
