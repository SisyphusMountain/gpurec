"""Compare Pi backward param grads against autograd through differentiable forward.

Uses _self_loop_differentiable with proper convergence for each wave.
The key difference from the analytical backward is that autograd tracks
gradients through ALL iterations, while the analytical backward uses
the implicit function theorem (IFT) approximation.
"""
import torch
from tests.gradients.test_wave_gradient import _setup_uniform
from src.core.likelihood import (
    Pi_wave_forward, Pi_wave_backward, compute_log_likelihood,
    _self_loop_differentiable, _dts_cross_differentiable,
    _compute_Pibar_inline, _safe_log2, NEG_INF,
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

# Forward pass (non-differentiable) to get E
log_pS, log_pD, log_pL, transfer_mat, mt = extract_parameters_uniform(
    theta, d['unnorm_row_max'], specieswise=False,
)
E_out = E_fixed_point(
    species_helpers=sh, log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
    transfer_mat=transfer_mat, max_transfer_mat=mt,
    max_iters=2000, tolerance=1e-10, warm_start_E=None,
    dtype=dtype, device=device, pibar_mode=pibar_mode, ancestors_T=ancestors_T,
)
E = E_out['E'].detach()
Ebar = E_out['E_bar'].detach()
E_s1 = E_out['E_s1'].detach()
E_s2 = E_out['E_s2'].detach()

wave_metas = wl['wave_metas']
wave_starts = wl['wave_starts']
C = int(wl['ccp_helpers']['C'])

# ====== AUTOGRAD PATH ======
# Make log_pD a leaf variable requiring grad
log_pD_var = log_pD.clone().detach().requires_grad_(True)

# Constants that depend on log_pD_var
DL_const = 1.0 + log_pD_var + E  # [S] — same formula as in the forward

leaf_row = wl['leaf_row_index'].to(device)
leaf_col = wl['leaf_col_index'].to(device)

# Build differentiable forward
Pi = torch.full((C, S), torch.finfo(dtype).min, dtype=dtype, device=device)
Pi[leaf_row, leaf_col] = 0.0

for wave_idx in range(len(wave_metas)):
    meta = wave_metas[wave_idx]
    ws = wave_starts[wave_idx]
    W = meta['W']
    we = ws + W
    
    leaf_wt = torch.full((W, S), NEG_INF, device=device, dtype=dtype)
    mask = (leaf_row >= ws) & (leaf_row < we)
    if mask.any():
        leaf_wt[leaf_row[mask] - ws, leaf_col[mask]] = 0.0
    leaf_wt = log_pS + leaf_wt
    
    # DTS cross (frozen Pi from previous waves already in Pi tensor)
    if meta['has_splits']:
        # Need a differentiable DTS cross computation
        Pi_max = Pi.max(dim=1, keepdim=True).values
        Pi_exp = torch.exp2(Pi - Pi_max)
        row_sum = Pi_exp.sum(dim=1, keepdim=True)
        anc_sum = Pi_exp @ ancestors_T
        Pibar_full = _safe_log2(row_sum - anc_sum) + Pi_max + mt
        
        dts_r = _dts_cross_differentiable(
            Pi, Pibar_full, meta, sp_child1, sp_child2,
            log_pD_var, log_pS, S, device, dtype,
        )
    else:
        dts_r = None
    
    # Self-loop: iterate to convergence
    Pi_W = Pi[ws:we].clone()
    for it in range(300):
        Pi_new = _self_loop_differentiable(
            Pi_W, mt, DL_const, Ebar, E, log_pS + E_s2, log_pS + E_s1,
            sp_child1, sp_child2, leaf_wt, dts_r, S,
            pibar_mode=pibar_mode,
            transfer_mat_T=transfer_mat,
            ancestors_T=ancestors_T,
        )
        with torch.no_grad():
            diff = (Pi_new - Pi_W).abs()
            significant = Pi_new > (Pi_new.max(dim=1, keepdim=True).values - 50)
            converged = not significant.any() or diff[significant].max().item() < 1e-10
        if converged:
            Pi_W = Pi_new
            break
        Pi_W = Pi_new
    
    Pi = torch.cat([Pi[:ws], Pi_W, Pi[we:]], dim=0)

# NLL
logL_auto = compute_log_likelihood(Pi, E, root_clade_ids).sum()
nll_auto = logL_auto.item()

# Autograd backward
logL_auto.backward()
grad_pD_auto = float(log_pD_var.grad)

print(f"Autograd NLL = {nll_auto:.6f}")
print(f"Autograd d(NLL)/d(log_pD) = {grad_pD_auto:.8e}")

# ====== ANALYTICAL PATH ======
Pi_out = Pi_wave_forward(
    wave_layout=wl, species_helpers=sh,
    E=E, Ebar=Ebar, E_s1=E_s1, E_s2=E_s2,
    log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
    transfer_mat=transfer_mat, max_transfer_mat=mt,
    device=device, dtype=dtype, pibar_mode=pibar_mode,
    local_tolerance=1e-10, local_iters=500,
)
nll_analytical = compute_log_likelihood(Pi_out['Pi'], E, root_clade_ids).sum().item()

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
grad_pD_an = float(bw['grad_log_pD'])

print(f"Analytical NLL = {nll_analytical:.6f}")
print(f"Analytical d(NLL)/d(log_pD) = {grad_pD_an:.8e}")

# FD reference
eps = 1e-6
Pi_p = Pi_wave_forward(
    wave_layout=wl, species_helpers=sh,
    E=E, Ebar=Ebar, E_s1=E_s1, E_s2=E_s2,
    log_pS=log_pS, log_pD=log_pD + eps, log_pL=log_pL,
    transfer_mat=transfer_mat, max_transfer_mat=mt,
    device=device, dtype=dtype, pibar_mode=pibar_mode,
    local_tolerance=1e-10, local_iters=500,
)
Pi_m = Pi_wave_forward(
    wave_layout=wl, species_helpers=sh,
    E=E, Ebar=Ebar, E_s1=E_s1, E_s2=E_s2,
    log_pS=log_pS, log_pD=log_pD - eps, log_pL=log_pL,
    transfer_mat=transfer_mat, max_transfer_mat=mt,
    device=device, dtype=dtype, pibar_mode=pibar_mode,
    local_tolerance=1e-10, local_iters=500,
)
fd_pD = (compute_log_likelihood(Pi_p['Pi'], E, root_clade_ids).sum().item() -
         compute_log_likelihood(Pi_m['Pi'], E, root_clade_ids).sum().item()) / (2 * eps)

print(f"FD d(NLL)/d(log_pD) = {fd_pD:.8e}")
print(f"\nErrors:")
print(f"  Analytical vs FD: {abs(grad_pD_an - fd_pD) / abs(fd_pD):.4e}")
print(f"  Autograd vs FD:   {abs(grad_pD_auto - fd_pD) / abs(fd_pD):.4e}")
