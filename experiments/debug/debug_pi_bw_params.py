"""Compare v_Pi (adjoint) from Pi_wave_backward against FD for individual clades.

This isolates whether the wave-decomposed backward produces correct dL/dPi
for the uniform mode.
"""
import torch
from tests.gradients.test_wave_gradient import _setup_uniform
from src.core.likelihood import (
    Pi_wave_forward, Pi_wave_backward, compute_log_likelihood,
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
logL = compute_log_likelihood(Pi_out['Pi'], E_out['E'], root_clade_ids).sum().item()

# Analytical backward 
bw = Pi_wave_backward(
    wave_layout=wl,
    Pi_star_wave=Pi_star_wave,
    Pibar_star_wave=Pibar_star_wave,
    E=E_out['E'], Ebar=E_out['E_bar'], E_s1=E_out['E_s1'], E_s2=E_out['E_s2'],
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
v_Pi = bw['v_Pi']  # [C, S]

# Also do the same with uniform for comparison
E_out_a = E_fixed_point(
    species_helpers=sh, log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
    transfer_mat=transfer_mat, max_transfer_mat=mt,
    max_iters=2000, tolerance=1e-10, warm_start_E=None,
    dtype=dtype, device=device, pibar_mode='uniform', ancestors_T=ancestors_T,
)
Pi_out_a = Pi_wave_forward(
    wave_layout=wl, species_helpers=sh,
    E=E_out_a['E'], Ebar=E_out_a['E_bar'], E_s1=E_out_a['E_s1'], E_s2=E_out_a['E_s2'],
    log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
    transfer_mat=transfer_mat, max_transfer_mat=mt,
    device=device, dtype=dtype, pibar_mode='uniform',
    local_tolerance=1e-10, local_iters=500,
)
bw_a = Pi_wave_backward(
    wave_layout=wl,
    Pi_star_wave=Pi_out_a['Pi_wave_ordered'],
    Pibar_star_wave=Pi_out_a['Pibar_wave_ordered'],
    E=E_out_a['E'], Ebar=E_out_a['E_bar'], E_s1=E_out_a['E_s1'], E_s2=E_out_a['E_s2'],
    log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
    max_transfer_mat=mt,
    species_helpers=sh,
    root_clade_ids_perm=wl['root_clade_ids'],
    device=device, dtype=dtype,
    neumann_terms=50,
    use_pruning=False,
    pibar_mode='uniform',
    ancestors_T=ancestors_T,
)

print(f"Uniform grad_log_pD: {float(bw['grad_log_pD']):.8e}")
print(f"Approx  grad_log_pD: {float(bw_a['grad_log_pD']):.8e}")
print(f"Uniform grad_log_pS: {float(bw['grad_log_pS']):.8e}")
print(f"Approx  grad_log_pS: {float(bw_a['grad_log_pS']):.8e}")
print(f"Uniform grad_mt sum: {float(bw['grad_max_transfer_mat'].sum()):.8e}")
print(f"Approx  grad_mt sum: {float(bw_a['grad_max_transfer_mat'].sum()):.8e}")
print(f"Uniform grad_E sum:  {float(bw['grad_E'].sum()):.8e}")
print(f"Approx  grad_E sum:  {float(bw_a['grad_E'].sum()):.8e}")

# Now let's check the Pi backward param grads against FD at FIXED E/Ebar/E_s1/E_s2
print("\n--- Checking Pi backward param grads at fixed E ---")
E_frozen = E_out['E'].detach().clone()
Ebar_frozen = E_out['E_bar'].detach().clone()
E_s1_frozen = E_out['E_s1'].detach().clone()
E_s2_frozen = E_out['E_s2'].detach().clone()

# Analytical: grad_log_pD from Pi backward (uniform)
an_pD = float(bw['grad_log_pD'])
an_pS = float(bw['grad_log_pS'])
an_mt = float(bw['grad_max_transfer_mat'].sum())

# FD: perturb log_pD, rerun Pi forward at frozen E, compare NLL
eps = 1e-6
# log_pD
Pi_p = Pi_wave_forward(
    wave_layout=wl, species_helpers=sh,
    E=E_frozen, Ebar=Ebar_frozen, E_s1=E_s1_frozen, E_s2=E_s2_frozen,
    log_pS=log_pS, log_pD=log_pD + eps, log_pL=log_pL,
    transfer_mat=transfer_mat, max_transfer_mat=mt,
    device=device, dtype=dtype, pibar_mode=pibar_mode,
    local_tolerance=1e-10, local_iters=500,
)
Pi_m = Pi_wave_forward(
    wave_layout=wl, species_helpers=sh,
    E=E_frozen, Ebar=Ebar_frozen, E_s1=E_s1_frozen, E_s2=E_s2_frozen,
    log_pS=log_pS, log_pD=log_pD - eps, log_pL=log_pL,
    transfer_mat=transfer_mat, max_transfer_mat=mt,
    device=device, dtype=dtype, pibar_mode=pibar_mode,
    local_tolerance=1e-10, local_iters=500,
)
fd_pD = (compute_log_likelihood(Pi_p['Pi'], E_frozen, root_clade_ids).sum().item() -
         compute_log_likelihood(Pi_m['Pi'], E_frozen, root_clade_ids).sum().item()) / (2 * eps)
print(f"log_pD: FD={fd_pD:.8e}, analytical={an_pD:.8e}, rel_err={(abs(fd_pD-an_pD)/max(abs(fd_pD),1e-30)):.4e}")

# log_pS
Pi_p = Pi_wave_forward(
    wave_layout=wl, species_helpers=sh,
    E=E_frozen, Ebar=Ebar_frozen, E_s1=E_s1_frozen, E_s2=E_s2_frozen,
    log_pS=log_pS + eps, log_pD=log_pD, log_pL=log_pL,
    transfer_mat=transfer_mat, max_transfer_mat=mt,
    device=device, dtype=dtype, pibar_mode=pibar_mode,
    local_tolerance=1e-10, local_iters=500,
)
Pi_m = Pi_wave_forward(
    wave_layout=wl, species_helpers=sh,
    E=E_frozen, Ebar=Ebar_frozen, E_s1=E_s1_frozen, E_s2=E_s2_frozen,
    log_pS=log_pS - eps, log_pD=log_pD, log_pL=log_pL,
    transfer_mat=transfer_mat, max_transfer_mat=mt,
    device=device, dtype=dtype, pibar_mode=pibar_mode,
    local_tolerance=1e-10, local_iters=500,
)
fd_pS = (compute_log_likelihood(Pi_p['Pi'], E_frozen, root_clade_ids).sum().item() -
         compute_log_likelihood(Pi_m['Pi'], E_frozen, root_clade_ids).sum().item()) / (2 * eps)
print(f"log_pS: FD={fd_pS:.8e}, analytical={an_pS:.8e}, rel_err={(abs(fd_pS-an_pS)/max(abs(fd_pS),1e-30)):.4e}")

# mt (perturb uniformly)
mt_p = mt + eps
mt_m = mt - eps
Pi_p = Pi_wave_forward(
    wave_layout=wl, species_helpers=sh,
    E=E_frozen, Ebar=Ebar_frozen, E_s1=E_s1_frozen, E_s2=E_s2_frozen,
    log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
    transfer_mat=transfer_mat, max_transfer_mat=mt_p,
    device=device, dtype=dtype, pibar_mode=pibar_mode,
    local_tolerance=1e-10, local_iters=500,
)
Pi_m = Pi_wave_forward(
    wave_layout=wl, species_helpers=sh,
    E=E_frozen, Ebar=Ebar_frozen, E_s1=E_s1_frozen, E_s2=E_s2_frozen,
    log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
    transfer_mat=transfer_mat, max_transfer_mat=mt_m,
    device=device, dtype=dtype, pibar_mode=pibar_mode,
    local_tolerance=1e-10, local_iters=500,
)
fd_mt = (compute_log_likelihood(Pi_p['Pi'], E_frozen, root_clade_ids).sum().item() -
         compute_log_likelihood(Pi_m['Pi'], E_frozen, root_clade_ids).sum().item()) / (2 * eps)
print(f"mt_sum: FD={fd_mt:.8e}, analytical={an_mt:.8e}, rel_err={(abs(fd_mt-an_mt)/max(abs(fd_mt),1e-30)):.4e}")
