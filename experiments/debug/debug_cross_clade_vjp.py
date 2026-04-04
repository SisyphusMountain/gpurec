"""Verify the cross-clade Pibar→Pi chain rule for uniform mode against autograd.

The cross-clade backward computes d(L)/d(Pi_child) from d(L)/d(Pibar_child).
For uniform mode, this uses the ancestor correction formula.
"""
import torch
from tests.gradients.test_wave_gradient import _setup_uniform
from src.core.likelihood import (
    Pi_wave_forward, _compute_Pibar_inline,
    _safe_log2, NEG_INF,
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
Pi_star = Pi_out['Pi_wave_ordered']

# Test the Pibar→Pi VJP directly
# Pibar(Pi) for a small batch of clades
N = 10  # test with first 10 clades
Pi_test = Pi_star[:N].clone()
mt_squeezed = mt if mt.ndim == 1 else mt.squeeze()

# Forward: compute Pibar
def pibar_fn(Pi):
    Pi_max = Pi.max(dim=1, keepdim=True).values
    Pi_exp = torch.exp2(Pi - Pi_max)
    row_sum = Pi_exp.sum(dim=1, keepdim=True)
    ancestor_sum = Pi_exp @ ancestors_T
    Pibar = _safe_log2(row_sum - ancestor_sum) + Pi_max + mt_squeezed
    return Pibar

# Autograd VJP
torch.manual_seed(42)
v_Pibar = torch.randn(N, S, device=device, dtype=dtype)

# Method 1: torch.func.vjp
_, vjp_fn = torch.func.vjp(pibar_fn, Pi_test)
pi_from_pibar_autograd = vjp_fn(v_Pibar)[0]

# Method 2: Analytical (the cross-clade code)
Pi_max_p = Pi_test.max(dim=1, keepdim=True).values
p_prime = torch.exp2(Pi_test - Pi_max_p)
anc_sum = p_prime @ ancestors_T
denom = p_prime.sum(dim=1, keepdim=True) - anc_sum
denom_safe = torch.where(denom > 0, denom, torch.ones_like(denom))
u_d = torch.where(denom > 0, v_Pibar / denom_safe, torch.zeros_like(v_Pibar))
A = u_d.sum(dim=1, keepdim=True)
correction = (ancestors_T @ u_d.T).T
pi_from_pibar_analytical = p_prime * (A - correction)

rel_err = (pi_from_pibar_analytical - pi_from_pibar_autograd).norm() / pi_from_pibar_autograd.norm()
max_abs = (pi_from_pibar_analytical - pi_from_pibar_autograd).abs().max().item()
print(f"Cross-clade Pibar→Pi VJP: rel_err={rel_err:.6e}, max_abs={max_abs:.6e}")

# Check per-species
diff = (pi_from_pibar_analytical - pi_from_pibar_autograd).abs()
max_per_s = diff.max(dim=0).values
top10 = torch.topk(max_per_s, min(10, S)).indices
for s_idx in top10[:5]:
    s = s_idx.item()
    an_vals = pi_from_pibar_analytical[:3, s].tolist()
    au_vals = pi_from_pibar_autograd[:3, s].tolist()
    print(f"  s={s}: analytical={[f'{v:.6e}' for v in an_vals]}, autograd={[f'{v:.6e}' for v in au_vals]}")

if rel_err > 1e-3:
    print("\n*** MISMATCH in cross-clade VJP! Investigating... ***")
    
    # Check if the issue is in the pibar_denom or the ancestor correction
    # Let's compare with the simpler uniform formula
    denom_approx = p_prime.sum(dim=1, keepdim=True) - p_prime
    denom_approx_safe = torch.where(denom_approx > 0, denom_approx, torch.ones_like(denom_approx))
    u_d_approx = torch.where(denom_approx > 0, v_Pibar / denom_approx_safe, torch.zeros_like(v_Pibar))
    A_approx = u_d_approx.sum(dim=1, keepdim=True)
    pi_from_pibar_approx = p_prime * (A_approx - u_d_approx)
    
    rel_err_approx = (pi_from_pibar_approx - pi_from_pibar_autograd).norm() / pi_from_pibar_autograd.norm()
    print(f"  uniform VJP vs autograd: rel_err={rel_err_approx:.6e}")
else:
    print("Cross-clade Pibar→Pi VJP is CORRECT!")
