"""Test: is the Pi backward correct for uniform mode by comparing v_Pi against FD?

This compares dL/dPi_star (the adjoint of the converged Pi) from the analytical
backward with finite differences.
"""
import torch
from tests.gradients.test_wave_gradient import _setup_uniform
from src.core.likelihood import (
    E_fixed_point, Pi_wave_forward, Pi_wave_backward, compute_log_likelihood,
)
from src.core.extract_parameters import extract_parameters_uniform

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
Pi_star = Pi_out['Pi_wave_ordered']
Pibar_star = Pi_out['Pibar_wave_ordered']
logL = compute_log_likelihood(Pi_out['Pi'], E_out['E'], root_clade_ids).sum().item()
print(f"NLL = {logL:.6f}")

# Analytical backward: get v_Pi and parameter gradients
bw = Pi_wave_backward(
    wave_layout=wl,
    Pi_star_wave=Pi_star,
    Pibar_star_wave=Pibar_star,
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

v_Pi_analytical = bw['v_Pi']  # [C, S]
grad_log_pD_an = bw['grad_log_pD']
grad_log_pS_an = bw['grad_log_pS']
grad_mt_an = bw['grad_max_transfer_mat']
grad_E_an = bw['grad_E']
grad_Ebar_an = bw['grad_Ebar']
grad_E_s1_an = bw['grad_E_s1']
grad_E_s2_an = bw['grad_E_s2']

# --- FD: dL/d(log_pD) at fixed E, Pi ---
# Actually, let's compare dL/d(Pi*) using FD, which avoids the E adjoint chain entirely
# dL/d(Pi*[c,s]) = lim_{eps->0} [L(Pi* + eps*e_{c,s}) - L(Pi* - eps*e_{c,s})] / (2*eps)
# But L depends on Pi* only through the root loglik computation.
# Actually, L = -[logsumexp2(Pi*[root,:]) - log2(S)] + log2(1-exp2(E).mean())
# So dL/dPi*[c,s] is zero for non-root clades.
# The v_Pi from backward is dL/dPi propagated through the wave structure.

# A better comparison: test dL/d(log_pD) holding E, Ebar, E_s1, E_s2 fixed
# This isolates the Pi backward from the E adjoint chain.
eps = 1e-6
E_frozen = E_out['E'].clone().detach()
Ebar_frozen = E_out['E_bar'].clone().detach()
E_s1_frozen = E_out['E_s1'].clone().detach()
E_s2_frozen = E_out['E_s2'].clone().detach()

for param_name, param_val, grad_an in [
    ('log_pD', log_pD, grad_log_pD_an),
    ('log_pS', log_pS, grad_log_pS_an),
    ('max_transfer_mat', mt, grad_mt_an),
]:
    # Scalar parameters
    if param_val.ndim == 0 or param_val.numel() == 1:
        p_plus = param_val.clone() + eps
        p_minus = param_val.clone() - eps
        
        # Re-run Pi forward with perturbed param at frozen E
        if param_name == 'log_pD':
            Pi_p = Pi_wave_forward(
                wave_layout=wl, species_helpers=sh,
                E=E_frozen, Ebar=Ebar_frozen, E_s1=E_s1_frozen, E_s2=E_s2_frozen,
                log_pS=log_pS, log_pD=p_plus, log_pL=log_pL,
                transfer_mat=transfer_mat, max_transfer_mat=mt,
                device=device, dtype=dtype, pibar_mode=pibar_mode,
                local_tolerance=1e-10, local_iters=500,
            )
            Pi_m = Pi_wave_forward(
                wave_layout=wl, species_helpers=sh,
                E=E_frozen, Ebar=Ebar_frozen, E_s1=E_s1_frozen, E_s2=E_s2_frozen,
                log_pS=log_pS, log_pD=p_minus, log_pL=log_pL,
                transfer_mat=transfer_mat, max_transfer_mat=mt,
                device=device, dtype=dtype, pibar_mode=pibar_mode,
                local_tolerance=1e-10, local_iters=500,
            )
        elif param_name == 'log_pS':
            Pi_p = Pi_wave_forward(
                wave_layout=wl, species_helpers=sh,
                E=E_frozen, Ebar=Ebar_frozen, E_s1=E_s1_frozen, E_s2=E_s2_frozen,
                log_pS=p_plus, log_pD=log_pD, log_pL=log_pL,
                transfer_mat=transfer_mat, max_transfer_mat=mt,
                device=device, dtype=dtype, pibar_mode=pibar_mode,
                local_tolerance=1e-10, local_iters=500,
            )
            Pi_m = Pi_wave_forward(
                wave_layout=wl, species_helpers=sh,
                E=E_frozen, Ebar=Ebar_frozen, E_s1=E_s1_frozen, E_s2=E_s2_frozen,
                log_pS=p_minus, log_pD=log_pD, log_pL=log_pL,
                transfer_mat=transfer_mat, max_transfer_mat=mt,
                device=device, dtype=dtype, pibar_mode=pibar_mode,
                local_tolerance=1e-10, local_iters=500,
            )
        else:  # max_transfer_mat
            Pi_p = Pi_wave_forward(
                wave_layout=wl, species_helpers=sh,
                E=E_frozen, Ebar=Ebar_frozen, E_s1=E_s1_frozen, E_s2=E_s2_frozen,
                log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
                transfer_mat=transfer_mat, max_transfer_mat=p_plus,
                device=device, dtype=dtype, pibar_mode=pibar_mode,
                local_tolerance=1e-10, local_iters=500,
            )
            Pi_m = Pi_wave_forward(
                wave_layout=wl, species_helpers=sh,
                E=E_frozen, Ebar=Ebar_frozen, E_s1=E_s1_frozen, E_s2=E_s2_frozen,
                log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
                transfer_mat=transfer_mat, max_transfer_mat=p_minus,
                device=device, dtype=dtype, pibar_mode=pibar_mode,
                local_tolerance=1e-10, local_iters=500,
            )
        
        logL_p = compute_log_likelihood(Pi_p['Pi'], E_frozen, root_clade_ids).sum().item()
        logL_m = compute_log_likelihood(Pi_m['Pi'], E_frozen, root_clade_ids).sum().item()
        fd = (logL_p - logL_m) / (2 * eps)
        grad_an_val = float(grad_an.sum()) if grad_an.ndim > 0 else float(grad_an)
        rel_err = abs(fd - grad_an_val) / max(abs(fd), 1e-30)
        print(f"  {param_name}: FD={fd:.8e}, analytical={grad_an_val:.8e}, rel_err={rel_err:.4e}")
    else:
        # Vector parameter: sum over species
        fd_total = 0.0
        for s in range(min(param_val.shape[0], 5)):  # just check a few
            p_plus = param_val.clone()
            p_plus[s] += eps
            p_minus = param_val.clone()
            p_minus[s] -= eps
            
            if param_name == 'max_transfer_mat':
                Pi_p = Pi_wave_forward(
                    wave_layout=wl, species_helpers=sh,
                    E=E_frozen, Ebar=Ebar_frozen, E_s1=E_s1_frozen, E_s2=E_s2_frozen,
                    log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
                    transfer_mat=transfer_mat, max_transfer_mat=p_plus,
                    device=device, dtype=dtype, pibar_mode=pibar_mode,
                    local_tolerance=1e-10, local_iters=500,
                )
                Pi_m = Pi_wave_forward(
                    wave_layout=wl, species_helpers=sh,
                    E=E_frozen, Ebar=Ebar_frozen, E_s1=E_s1_frozen, E_s2=E_s2_frozen,
                    log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
                    transfer_mat=transfer_mat, max_transfer_mat=p_minus,
                    device=device, dtype=dtype, pibar_mode=pibar_mode,
                    local_tolerance=1e-10, local_iters=500,
                )
            
            logL_p = compute_log_likelihood(Pi_p['Pi'], E_frozen, root_clade_ids).sum().item()
            logL_m = compute_log_likelihood(Pi_m['Pi'], E_frozen, root_clade_ids).sum().item()
            fd_s = (logL_p - logL_m) / (2 * eps)
            grad_an_s = float(grad_an[s]) if grad_an.ndim > 0 else float(grad_an)
            rel_err = abs(fd_s - grad_an_s) / max(abs(fd_s), 1e-30)
            print(f"  {param_name}[{s}]: FD={fd_s:.8e}, analytical={grad_an_s:.8e}, rel_err={rel_err:.4e}")
