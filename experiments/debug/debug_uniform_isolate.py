"""Isolate where the uniform gradient error comes from.

Strategy:
1. Run forward at theta_0 with uniform mode
2. Compute analytical backward
3. Compare analytical backward with autograd backward (differentiable self-loop)
4. Identify which component (self-loop VJP vs cross-clade VJP) is wrong
"""
import math
import torch
from pathlib import Path
from tests.gradients.test_wave_gradient import _setup_uniform, _full_forward
from gpurec.core.likelihood import compute_log_likelihood
from gpurec.core.forward import Pi_wave_forward, NEG_INF
from gpurec.core.backward import (
    Pi_wave_backward, _self_loop_differentiable,
    _self_loop_vjp_precompute, _self_loop_Jt_apply,
)
from gpurec.core.extract_parameters import extract_parameters_uniform
from gpurec.optimization.theta_optimizer import implicit_grad_loglik_vjp_wave

d = _setup_uniform("test_trees_20", n_families=1, dtype=torch.float64)
device, dtype = d['device'], d['dtype']
theta = d['theta'].clone()
pibar_mode = d['pibar_mode']
ancestors_T = d['ancestors_T']
sh = d['species_helpers']
wl = d['wave_layout']
root_clade_ids = d['root_clade_ids']
unnorm_row_max = d['unnorm_row_max']

# Forward pass
log_pS, log_pD, log_pL, transfer_mat, mt = extract_parameters_uniform(
    theta, unnorm_row_max, specieswise=False,
)
from gpurec.core.likelihood import E_fixed_point  # noqa: E402
E_out = E_fixed_point(
    species_helpers=sh, log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
    transfer_mat=transfer_mat, max_transfer_mat=mt,
    max_iters=2000, tolerance=1e-10, warm_start_E=None,
    dtype=dtype, device=device, pibar_mode=pibar_mode,
    ancestors_T=ancestors_T,
)
Pi_out = Pi_wave_forward(
    wave_layout=wl, species_helpers=sh,
    E=E_out['E'], Ebar=E_out['E_bar'], E_s1=E_out['E_s1'], E_s2=E_out['E_s2'],
    log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
    transfer_mat=transfer_mat, max_transfer_mat=mt,
    device=device, dtype=dtype, pibar_mode=pibar_mode,
    local_tolerance=1e-10, local_iters=500,
)
logL = compute_log_likelihood(Pi_out['Pi'], E_out['E'], root_clade_ids)
nll = logL.sum().item()
print(f"NLL = {nll:.6f}")

Pi_star_wave = Pi_out['Pi_wave_ordered']
Pibar_star_wave = Pi_out['Pibar_wave_ordered']

# --- Analytical backward (the function being tested) ---
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

print(f"\nAnalytical grad_log_pD = {float(bw['grad_log_pD']):.6f}")
print(f"Analytical grad_log_pS = {float(bw['grad_log_pS']):.6f}")
print(f"Analytical grad_mt     = {float(bw['grad_max_transfer_mat'].sum()):.6f}")
print(f"Analytical grad_E      = {float(bw['grad_E'].sum()):.6f}")
print(f"Analytical grad_Ebar   = {float(bw['grad_Ebar'].sum()):.6f}")

# --- Now compare with uniform backward ---
# First, run forward with uniform to see if the VJP is correct for that mode
log_pS_a, log_pD_a, log_pL_a, transfer_mat_a, mt_a = extract_parameters_uniform(
    theta, unnorm_row_max, specieswise=False,
)
E_out_a = E_fixed_point(
    species_helpers=sh, log_pS=log_pS_a, log_pD=log_pD_a, log_pL=log_pL_a,
    transfer_mat=transfer_mat_a, max_transfer_mat=mt_a,
    max_iters=2000, tolerance=1e-10, warm_start_E=None,
    dtype=dtype, device=device, pibar_mode='uniform',
    ancestors_T=ancestors_T,
)
Pi_out_a = Pi_wave_forward(
    wave_layout=wl, species_helpers=sh,
    E=E_out_a['E'], Ebar=E_out_a['E_bar'], E_s1=E_out_a['E_s1'], E_s2=E_out_a['E_s2'],
    log_pS=log_pS_a, log_pD=log_pD_a, log_pL=log_pL_a,
    transfer_mat=transfer_mat_a, max_transfer_mat=mt_a,
    device=device, dtype=dtype, pibar_mode='uniform',
    local_tolerance=1e-10, local_iters=500,
)
logL_a = compute_log_likelihood(Pi_out_a['Pi'], E_out_a['E'], root_clade_ids)
nll_a = logL_a.sum().item()
print(f"\nuniform NLL = {nll_a:.6f}")
print(f"uniform NLL = {nll:.6f}")
print(f"Difference = {abs(nll - nll_a):.6f}")

# --- FD reference for the 'uniform' forward ---
eps = 1e-5
fd_grad = torch.zeros_like(theta)
for i in range(theta.numel()):
    theta_p = theta.clone()
    theta_p[i] += eps
    log_pS_p, log_pD_p, log_pL_p, tf_p, mt_p = extract_parameters_uniform(theta_p, unnorm_row_max, specieswise=False)
    E_p = E_fixed_point(
        species_helpers=sh, log_pS=log_pS_p, log_pD=log_pD_p, log_pL=log_pL_p,
        transfer_mat=tf_p, max_transfer_mat=mt_p,
        max_iters=2000, tolerance=1e-10, warm_start_E=None,
        dtype=dtype, device=device, pibar_mode=pibar_mode, ancestors_T=ancestors_T,
    )
    Pi_p = Pi_wave_forward(
        wave_layout=wl, species_helpers=sh,
        E=E_p['E'], Ebar=E_p['E_bar'], E_s1=E_p['E_s1'], E_s2=E_p['E_s2'],
        log_pS=log_pS_p, log_pD=log_pD_p, log_pL=log_pL_p,
        transfer_mat=tf_p, max_transfer_mat=mt_p,
        device=device, dtype=dtype, pibar_mode=pibar_mode,
        local_tolerance=1e-10, local_iters=500,
    )
    logL_p = compute_log_likelihood(Pi_p['Pi'], E_p['E'], root_clade_ids).sum().item()
    
    theta_m = theta.clone()
    theta_m[i] -= eps
    log_pS_m, log_pD_m, log_pL_m, tf_m, mt_m = extract_parameters_uniform(theta_m, unnorm_row_max, specieswise=False)
    E_m = E_fixed_point(
        species_helpers=sh, log_pS=log_pS_m, log_pD=log_pD_m, log_pL=log_pL_m,
        transfer_mat=tf_m, max_transfer_mat=mt_m,
        max_iters=2000, tolerance=1e-10, warm_start_E=None,
        dtype=dtype, device=device, pibar_mode=pibar_mode, ancestors_T=ancestors_T,
    )
    Pi_m = Pi_wave_forward(
        wave_layout=wl, species_helpers=sh,
        E=E_m['E'], Ebar=E_m['E_bar'], E_s1=E_m['E_s1'], E_s2=E_m['E_s2'],
        log_pS=log_pS_m, log_pD=log_pD_m, log_pL=log_pL_m,
        transfer_mat=tf_m, max_transfer_mat=mt_m,
        device=device, dtype=dtype, pibar_mode=pibar_mode,
        local_tolerance=1e-10, local_iters=500,
    )
    logL_m = compute_log_likelihood(Pi_m['Pi'], E_m['E'], root_clade_ids).sum().item()
    fd_grad[i] = (logL_p - logL_m) / (2 * eps)

print(f"\nFD gradient: {fd_grad.tolist()}")

# Now get analytical gradient from the full chain
grad_theta, stats = implicit_grad_loglik_vjp_wave(
    wl, sh,
    Pi_star_wave=Pi_star_wave,
    Pibar_star_wave=Pibar_star_wave,
    E_star=E_out['E'], E_s1=E_out['E_s1'],
    E_s2=E_out['E_s2'], Ebar=E_out['E_bar'],
    log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
    max_transfer_mat=mt,
    root_clade_ids_perm=wl['root_clade_ids'],
    theta=theta,
    unnorm_row_max=unnorm_row_max,
    specieswise=False,
    device=device, dtype=dtype,
    neumann_terms=50, use_pruning=False,
    cg_tol=1e-12, cg_maxiter=1000,
    pibar_mode=pibar_mode,
    ancestors_T=ancestors_T,
)

print(f"Analytical gradient: {grad_theta.tolist()}")
print(f"\nPer-component comparison:")
for i in range(theta.numel()):
    fd_val = float(fd_grad[i])
    an_val = float(grad_theta[i])
    rel_err = abs(an_val - fd_val) / max(abs(fd_val), 1e-30)
    print(f"  theta[{i}]: FD={fd_val:.8e}, analytical={an_val:.8e}, rel_err={rel_err:.4e}")
