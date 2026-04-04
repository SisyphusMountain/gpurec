"""Check how many forward iterations each wave actually takes."""
import torch
from tests.gradients.test_wave_gradient import _setup_uniform
from src.core.likelihood import (
    _compute_Pibar_inline, _compute_dts_cross, NEG_INF,
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
transfer_mat_T = None  # uniform mode

E_out = E_fixed_point(
    species_helpers=sh, log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
    transfer_mat=transfer_mat, max_transfer_mat=mt,
    max_iters=2000, tolerance=1e-10, warm_start_E=None,
    dtype=dtype, device=device, pibar_mode=pibar_mode, ancestors_T=ancestors_T,
)

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
DL_const = log_pD + log_pL
SL1_const = log_pS + E_s2
SL2_const = log_pS + E_s1

if ancestors_T.is_sparse:
    ancestors_T_dense = ancestors_T.to_dense()
else:
    ancestors_T_dense = ancestors_T

# Build Pi/Pibar arrays, iterate wave by wave
C = sum(m['W'] for m in wave_metas)
Pi = torch.full((C, S), NEG_INF, device=device, dtype=dtype)
Pibar = torch.full((C, S), NEG_INF, device=device, dtype=dtype)

# Set leaf entries
leaf_row = wl['leaf_row_index'].to(device)
leaf_col = wl['leaf_col_index'].to(device)

print(f"theta = {theta}")
print(f"log_pS={log_pS:.6f}, log_pD={log_pD:.6f}, log_pL={log_pL:.6f}")
print(f"S = {S}")
print()

for wave_idx in range(len(wave_metas)):
    meta = wave_metas[wave_idx]
    ws = wave_starts[wave_idx]
    W = meta['W']
    we = ws + W

    if meta['has_splits']:
        dts_r = _compute_dts_cross(
            Pi, Pibar, meta,
            sp_child1, sp_child2, log_pD, log_pS, S, device, dtype
        )
    else:
        dts_r = None

    leaf_wt = torch.full((W, S), NEG_INF, device=device, dtype=dtype)
    mask_lr = (leaf_row >= ws) & (leaf_row < we)
    if mask_lr.any():
        leaf_wt[leaf_row[mask_lr] - ws, leaf_col[mask_lr]] = 0.0

    # Manual forward iteration with convergence tracking
    Pi_W = torch.full((W, S), NEG_INF, device=device, dtype=dtype)
    # Initialize leaf entries
    if mask_lr.any():
        Pi_W[leaf_row[mask_lr] - ws, leaf_col[mask_lr]] = 0.0

    for iteration in range(500):
        Pibar_W = _compute_Pibar_inline(Pi_W, transfer_mat_T, mt, pibar_mode,
                                         ancestors_T=ancestors_T_dense)

        # Manual wave step
        t0 = DL_const.unsqueeze(0) + Pi_W
        t1 = Pi_W + Ebar.unsqueeze(0)
        t2 = Pibar_W + E.unsqueeze(0)
        Pi_pad = torch.cat([Pi_W, torch.full((W, 1), NEG_INF, device=device, dtype=dtype)], dim=1)
        t3 = SL1_const.unsqueeze(0) + Pi_pad[:, sp_child1.long()]
        t4 = SL2_const.unsqueeze(0) + Pi_pad[:, sp_child2.long()]
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
        Pi_new = torch.log2(torch.exp2(stacked - m_safe.unsqueeze(0)).sum(dim=0)) + m

        diff = (Pi_new - Pi_W).abs()
        significant = Pi_new > -100.0
        if significant.any():
            max_diff = diff[significant].max().item()
        else:
            max_diff = 0.0

        Pi_W = Pi_new
        if max_diff < 1e-10:
            print(f"Wave {wave_idx}: W={W:4d}, converged in {iteration+1} iters, max_diff={max_diff:.2e}")
            break
    else:
        print(f"Wave {wave_idx}: W={W:4d}, DID NOT CONVERGE in 500 iters, max_diff={max_diff:.2e}")

    Pi[ws:we] = Pi_W
    Pibar_W_final = _compute_Pibar_inline(Pi_W, transfer_mat_T, mt, pibar_mode,
                                           ancestors_T=ancestors_T_dense)
    Pibar[ws:we] = Pibar_W_final
