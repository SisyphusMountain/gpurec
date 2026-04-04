"""Compare spectral radii of the FORWARD iteration Jacobian vs the BACKWARD J^T.

The forward fixed-point iteration is:
    Pi_new = f(Pi) = logsumexp2(DL + Pi, Pi + Ebar, Pibar(Pi) + E, SL1 + Pi_s1, SL2 + Pi_s2, leaf, dts_cross)

The backward solves (I - J^T) v = rhs where J = df/dPi.

If the forward converges in 4-10 iters, ρ(J) should be small (< 0.5 or so).
The question is: does _self_loop_Jt_apply compute the SAME J as the forward iteration's Jacobian?

We test this by:
1. Computing J_autograd = autograd Jacobian of one forward step f(Pi) at Pi*
2. Computing J_analytic via _self_loop_Jt_apply on identity vectors
3. Comparing spectral radii of both
"""
import torch
from tests.gradients.test_wave_gradient import _setup_uniform
from src.core.likelihood import (
    Pi_wave_forward, _self_loop_vjp_precompute, _self_loop_Jt_apply,
    _compute_Pibar_inline, NEG_INF,
)
from src.core.extract_parameters import extract_parameters_uniform
from src.core.likelihood import E_fixed_point
from src.core.kernels.wave_step import wave_step_fused

d = _setup_uniform("test_trees_20", n_families=1, dtype=torch.float64)
device, dtype = d['device'], d['dtype']
theta = d['theta'].clone()
pibar_mode = d['pibar_mode']
ancestors_T = d['ancestors_T']
sh = d['species_helpers']
wl = d['wave_layout']
S = sh['S']

log_pS, log_pD, log_pL, transfer_mat, mt = extract_parameters_uniform(
    theta, d['unnorm_row_max'], specieswise=False,
)
# In uniform mode, transfer_mat is None
transfer_mat_T = None

# Need dense ancestors_T for autograd
if ancestors_T.is_sparse:
    ancestors_T_dense = ancestors_T.to_dense()
else:
    ancestors_T_dense = ancestors_T

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

sp_P_idx = sh['s_P_indexes']
sp_c12_idx = sh['s_C12_indexes']
p_cpu = sp_P_idx.cpu().long()
c_cpu = sp_c12_idx.cpu().long()
mask_c1 = p_cpu < S
sp_child1 = torch.full((S,), S, dtype=torch.long, device=device)
sp_child2 = torch.full((S,), S, dtype=torch.long, device=device)
sp_child1[p_cpu[mask_c1].to(device)] = c_cpu[mask_c1].to(device)
sp_child2[(p_cpu[~mask_c1] - S).to(device)] = c_cpu[~mask_c1].to(device)

from src.core.likelihood import _compute_dts_cross

wave_metas = wl['wave_metas']
wave_starts = wl['wave_starts']

E = E_out['E']
Ebar = E_out['E_bar']
E_s1 = E_out['E_s1']
E_s2 = E_out['E_s2']

# Shared constants
DL_const = log_pD + log_pL
SL1_const = log_pS + E_s2
SL2_const = log_pS + E_s1
mt_squeezed = mt

for wave_idx in range(len(wave_metas)):
    meta = wave_metas[wave_idx]
    ws = wave_starts[wave_idx]
    W = meta['W']
    we = ws + W

    if W * S > 5000:
        print(f"Wave {wave_idx}: W={W}, skipping (too large for full Jacobian)")
        continue

    Pi_W_star = Pi_star_wave[ws:we].clone()
    Pibar_W_star = Pibar_star_wave[ws:we].clone()

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
    mask_lr = (leaf_row >= ws) & (leaf_row < we)
    if mask_lr.any():
        leaf_wt[leaf_row[mask_lr] - ws, leaf_col[mask_lr]] = 0.0

    # ========================================================
    # 1. AUTOGRAD JACOBIAN of forward step f(Pi)
    # ========================================================
    def forward_step(Pi_W_flat):
        """One forward self-loop step: Pi_W -> Pi_new."""
        Pi_W = Pi_W_flat.reshape(W, S)
        Pibar_W = _compute_Pibar_inline(Pi_W, transfer_mat_T, mt_squeezed, pibar_mode,
                                         ancestors_T=ancestors_T_dense)
        # Manual wave_step (can't use Triton with autograd)
        DL_w = DL_const.unsqueeze(0).expand(W, -1)
        Ebar_w = Ebar.unsqueeze(0).expand(W, -1)
        E_w = E.unsqueeze(0).expand(W, -1)
        SL1_w = SL1_const.unsqueeze(0).expand(W, -1)
        SL2_w = SL2_const.unsqueeze(0).expand(W, -1)

        t0 = DL_w + Pi_W
        t1 = Pi_W + Ebar_w
        t2 = Pibar_W + E_w
        # Speciation terms: gather children
        t3 = SL1_w + Pi_W[:, sp_child1.clamp(max=S-1)]
        t4 = SL2_w + Pi_W[:, sp_child2.clamp(max=S-1)]
        # Mask invalid children
        c1_valid = sp_child1 < S
        c2_valid = sp_child2 < S
        t3 = torch.where(c1_valid.unsqueeze(0), t3, torch.full_like(t3, NEG_INF))
        t4 = torch.where(c2_valid.unsqueeze(0), t4, torch.full_like(t4, NEG_INF))

        terms = [t0, t1, t2, t3, t4, leaf_wt]
        if dts_r is not None:
            terms.append(dts_r)

        stacked = torch.stack(terms, dim=0)
        m = stacked.max(dim=0).values
        m_safe = torch.where(m > -1e30, m, torch.zeros_like(m))
        result = torch.log2(torch.exp2(stacked - m_safe.unsqueeze(0)).sum(dim=0)) + m
        return result.reshape(-1)

    Pi_flat = Pi_W_star.reshape(-1).clone().requires_grad_(True)
    J_autograd = torch.autograd.functional.jacobian(forward_step, Pi_flat)
    # J_autograd is [W*S, W*S]

    # ========================================================
    # 2. ANALYTIC Jacobian from _self_loop_Jt_apply
    # ========================================================
    mt_w = mt_squeezed.unsqueeze(0).expand(W, -1)
    DL_w = DL_const.unsqueeze(0).expand(W, -1)
    E_w = E.unsqueeze(0).expand(W, -1)
    Ebar_w = Ebar.unsqueeze(0).expand(W, -1)
    SL1_w = SL1_const.unsqueeze(0).expand(W, -1)
    SL2_w = SL2_const.unsqueeze(0).expand(W, -1)

    ingredients = _self_loop_vjp_precompute(
        Pi_W_star, Pibar_W_star, dts_r,
        mt_w, DL_w, Ebar_w, E_w, SL1_w, SL2_w,
        sp_child1, sp_child2, leaf_wt, S,
        pibar_mode, transfer_mat_T, ancestors_T,
    )

    n = W * S
    J_analytic_T = torch.zeros(n, n, device=device, dtype=dtype)
    for i in range(n):
        e_i = torch.zeros(W, S, device=device, dtype=dtype)
        e_i.view(-1)[i] = 1.0
        col = _self_loop_Jt_apply(
            e_i, ingredients, sp_child1, sp_child2, S, W,
            pibar_mode, transfer_mat_T, ancestors_T,
        )
        J_analytic_T[:, i] = col.reshape(-1)
    # J_analytic_T is the matrix that _self_loop_Jt_apply represents, i.e. J^T
    # So J_analytic = J_analytic_T^T
    J_analytic = J_analytic_T.T

    # ========================================================
    # 3. Compare
    # ========================================================
    diff = (J_autograd - J_analytic).abs().max().item()
    rho_autograd = torch.linalg.eigvals(J_autograd).abs().max().item()
    rho_analytic = torch.linalg.eigvals(J_analytic).abs().max().item()

    print(f"Wave {wave_idx}: W={W:4d}, n={n}")
    print(f"  J_autograd vs J_analytic max diff: {diff:.6e}")
    print(f"  ρ(J_autograd)  = {rho_autograd:.6f}")
    print(f"  ρ(J_analytic)  = {rho_analytic:.6f}")
    print(f"  Frobenius norm of J_autograd: {J_autograd.norm():.6f}")
    print(f"  Frobenius norm of J_analytic: {J_analytic.norm():.6f}")
    print()
