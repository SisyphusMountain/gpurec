"""Check how gradient error varies with Pi forward tolerance for uniform mode."""
import math, torch
from pathlib import Path
from tests.gradients.test_wave_gradient import _setup_uniform
from src.core.extract_parameters import extract_parameters_uniform
from src.core.likelihood import (
    E_fixed_point, Pi_wave_forward, compute_log_likelihood,
)
from src.optimization.theta_optimizer import implicit_grad_loglik_vjp_wave

d = _setup_uniform("test_trees_20", n_families=1, dtype=torch.float64)
device, dtype = d['device'], d['dtype']
theta = d['theta'].clone()
pibar_mode = d['pibar_mode']
ancestors_T = d['ancestors_T']
sh = d['species_helpers']
wl = d['wave_layout']
root_clade_ids = d['root_clade_ids']
unnorm_row_max = d['unnorm_row_max']


def _forward_with_tol(theta_val, tol, local_iters=200):
    log_pS, log_pD, log_pL, transfer_mat, mt = extract_parameters_uniform(
        theta_val, unnorm_row_max, specieswise=False,
    )
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
        local_tolerance=tol, local_iters=local_iters,
    )
    logL = compute_log_likelihood(Pi_out['Pi'], E_out['E'], root_clade_ids)
    return logL.sum().item(), Pi_out, E_out, log_pS, log_pD, log_pL, mt


# FD reference with very tight tolerance
eps = 1e-5
fd_grad = torch.zeros_like(theta)
for i in range(theta.numel()):
    theta_p = theta.clone()
    theta_p[i] += eps
    logL_p, _, _, _, _, _, _ = _forward_with_tol(theta_p, tol=1e-10, local_iters=500)
    theta_m = theta.clone()
    theta_m[i] -= eps
    logL_m, _, _, _, _, _, _ = _forward_with_tol(theta_m, tol=1e-10, local_iters=500)
    fd_grad[i] = (logL_p - logL_m) / (2 * eps)

print(f"FD gradient (tight tol): {fd_grad}")

# Sweep forward tolerance
for fwd_tol in [1e-3, 1e-4, 1e-5, 1e-6, 1e-8, 1e-10]:
    logL_base, Pi_out_base, E_out_base, log_pS, log_pD, log_pL, mt = _forward_with_tol(
        theta, tol=fwd_tol, local_iters=500
    )

    grad_theta, stats = implicit_grad_loglik_vjp_wave(
        wl, sh,
        Pi_star_wave=Pi_out_base['Pi_wave_ordered'],
        Pibar_star_wave=Pi_out_base['Pibar_wave_ordered'],
        E_star=E_out_base['E'], E_s1=E_out_base['E_s1'],
        E_s2=E_out_base['E_s2'], Ebar=E_out_base['E_bar'],
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
    rel_errs = [(abs(float(grad_theta[i]) - float(fd_grad[i])) / max(abs(float(fd_grad[i])), 1e-30))
                for i in range(theta.numel())]
    max_err = max(rel_errs)
    print(f"  fwd_tol={fwd_tol:.0e}: grad={[f'{float(g):.6f}' for g in grad_theta]}, "
          f"max_rel_err={max_err:.4e}, all_errs={[f'{e:.4e}' for e in rel_errs]}")
