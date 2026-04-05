"""Full-chain gradient test: pure-CPU, no Triton.

Replicates the uniform full-chain FD test but does everything on CPU using
_self_loop_differentiable instead of Triton kernels.
"""
import torch
import sys
sys.path.insert(0, '/home/enzo/Documents/git/gpurec/gpurec')

from gpurec.core.likelihood import E_fixed_point
from gpurec.core.forward import _compute_dts_cross, NEG_INF
from gpurec.core.backward import (
    Pi_wave_backward, _self_loop_differentiable, _dts_cross_differentiable,
)
from gpurec.core.log2_utils import _safe_log2_internal as _safe_log2, logsumexp2
from gpurec.core.scheduling import compute_clade_waves
from gpurec.core.batching import collate_wave, build_wave_layout
from gpurec.core.extract_parameters import extract_parameters_uniform
from gpurec.core.model import collate_gene_families
from gpurec.optimization.theta_optimizer import implicit_grad_loglik_vjp_wave
from tests.gradients.test_wave_gradient import _load_extension, _ROOT
import gpurec.core.forward as _fwd_mod

# Monkey-patch _compute_dts_cross to use pure PyTorch (no Triton)
_orig_compute_dts_cross = _fwd_mod._compute_dts_cross

def _patched_compute_dts_cross(Pi, Pibar, meta, sp_child1, sp_child2, log_pD, log_pS,
                                S, device, dtype):
    """Pure PyTorch fallback for _compute_dts_cross."""
    return _dts_cross_differentiable(Pi, Pibar, meta, sp_child1, sp_child2,
                                     log_pD, log_pS, S, device, dtype)

_fwd_mod._compute_dts_cross = _patched_compute_dts_cross

dtype = torch.float64
device = torch.device('cpu')
D, L, T = 0.27, 0.27, 0.1
INV_LN2 = 1.0 / 0.6931471805599453

# ─── Load data ───────────────────────────────────────────────────────────────
ext = _load_extension()
data_dir = _ROOT / "data" / "test_trees_20"
sp_path = str(data_dir / "sp.nwk")
gene_paths = sorted(data_dir.glob("g_*.nwk"))[:1]
raw = ext.preprocess(sp_path, [str(gene_paths[0])])
sr = raw['species']
cr = raw['ccp']
ch = {
    "split_leftrights_sorted": cr["split_leftrights_sorted"],
    "log_split_probs_sorted": cr["log_split_probs_sorted"].to(dtype=dtype) * INV_LN2,
    "seg_parent_ids": cr["seg_parent_ids"],
    "ptr_ge2": cr["ptr_ge2"],
    "num_segs_ge2": int(cr["num_segs_ge2"]),
    "num_segs_eq1": int(cr["num_segs_eq1"]),
    "end_rows_ge2": int(cr["end_rows_ge2"]),
    "C": int(cr["C"]),
    "N_splits": int(cr["N_splits"]),
}
if "split_parents_sorted" in cr:
    ch["split_parents_sorted"] = cr["split_parents_sorted"]
batch_items = [{
    "ccp": ch,
    "leaf_row_index": raw["leaf_row_index"].long(),
    "leaf_col_index": raw["leaf_col_index"].long(),
    "root_clade_id": int(cr["root_clade_id"]),
}]
sh = {
    "S": int(sr["S"]),
    "names": sr["names"],
    "s_P_indexes": sr["s_P_indexes"].to(device=device),
    "s_C12_indexes": sr["s_C12_indexes"].to(device=device),
    "Recipients_mat": sr["Recipients_mat"].to(dtype=dtype, device=device),
    "ancestors_dense": sr["ancestors_dense"].to(dtype=dtype, device=device),
}
S = sh["S"]
ancestors_T = sh["ancestors_dense"].T.to_sparse_coo()
pibar_mode = 'uniform'

theta = torch.log2(torch.tensor([D, L, T], dtype=dtype, device=device))
tm_unnorm = torch.log2(sh["Recipients_mat"])
unnorm_row_max = tm_unnorm.max(dim=-1).values

# Build wave layout (once)
batched = collate_gene_families(batch_items, dtype=dtype, device=device)
families_waves, families_phases = [], []
for bi in batch_items:
    w, p = compute_clade_waves(bi['ccp'])
    families_waves.append(w)
    families_phases.append(p)
offsets = [m['clade_offset'] for m in batched['family_meta']]
cross_waves = collate_wave(families_waves, offsets)
max_n = max(len(p) for p in families_phases)
cross_phases = [max(fp[k] if k < len(fp) else 1 for fp in families_phases) for k in range(max_n)]
wave_layout = build_wave_layout(
    waves=cross_waves, phases=cross_phases,
    ccp_helpers=batched['ccp'],
    leaf_row_index=batched['leaf_row_index'],
    leaf_col_index=batched['leaf_col_index'],
    root_clade_ids=batched['root_clade_ids'],
    device=device, dtype=dtype,
)

# Species tree children
sp_P_idx = sh['s_P_indexes']
sp_c12_idx = sh['s_C12_indexes']
p_cpu = sp_P_idx.long()
c_cpu = sp_c12_idx.long()
mask_c1 = p_cpu < S
sp_child1 = torch.full((S,), S, dtype=torch.long)
sp_child2 = torch.full((S,), S, dtype=torch.long)
sp_child1[p_cpu[mask_c1]] = c_cpu[mask_c1]
sp_child2[(p_cpu[~mask_c1] - S)] = c_cpu[~mask_c1]


def full_forward_cpu(theta_val):
    """Run full forward (E + Pi) on CPU, return logL and intermediate results."""
    log_pS, log_pD, log_pL, transfer_mat, mt = extract_parameters_uniform(
        theta_val, unnorm_row_max, specieswise=False,
    )
    E_out = E_fixed_point(
        species_helpers=sh, log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
        transfer_mat=transfer_mat, max_transfer_mat=mt,
        max_iters=2000, tolerance=1e-10, warm_start_E=None,
        dtype=dtype, device=device, pibar_mode=pibar_mode, ancestors_T=ancestors_T,
    )
    E = E_out['E']
    Ebar = E_out['E_bar']
    E_s1 = E_out['E_s1']
    E_s2 = E_out['E_s2']

    # Forward pass: pure PyTorch
    wave_metas = wave_layout['wave_metas']
    C_total = wave_layout['ccp_helpers']['C']
    Pi = torch.full((C_total, S), torch.finfo(dtype).min, dtype=dtype, device=device)
    lr = wave_layout['leaf_row_index'].long()
    lc = wave_layout['leaf_col_index'].long()
    Pi[lr, lc] = 0.0
    Pibar = torch.full((C_total, S), NEG_INF, dtype=dtype, device=device)

    DL_const = 1.0 + log_pD + E
    SL1_const = log_pS + E_s2
    SL2_const = log_pS + E_s1

    for wi in range(len(wave_metas)):
        meta = wave_metas[wi]
        ws = meta['start']
        W = meta['W']
        we = ws + W

        if meta['has_splits']:
            dts_r = _dts_cross_differentiable(
                Pi, Pibar, meta, sp_child1, sp_child2,
                log_pD, log_pS, S, device, dtype,
            )
        else:
            dts_r = None

        leaf_wt = torch.full((W, S), NEG_INF, dtype=dtype, device=device)
        mask_lr = (lr >= ws) & (lr < we)
        if mask_lr.any():
            leaf_wt[lr[mask_lr] - ws, lc[mask_lr]] = 0.0
        leaf_wt = log_pS + leaf_wt

        Pi_W = Pi[ws:we].clone()
        for local_iter in range(500):
            Pi_new = _self_loop_differentiable(
                Pi_W, mt, DL_const, Ebar, E, SL1_const, SL2_const,
                sp_child1, sp_child2, leaf_wt, dts_r, S,
                pibar_mode=pibar_mode, transfer_mat_T=None, ancestors_T=ancestors_T,
            )
            sig = Pi_new > -100.0
            diff = torch.abs(Pi_new - Pi_W)[sig].max().item() if sig.any() else 0
            Pi_W = Pi_new
            if diff < 1e-12:
                break

        Pi[ws:we] = Pi_W.detach()
        Pi_max = Pi_W.max(dim=1, keepdim=True).values
        Pi_exp = torch.exp2(Pi_W - Pi_max)
        row_sum = Pi_exp.sum(dim=1, keepdim=True)
        anc_sum = Pi_exp @ ancestors_T
        Pibar[ws:we] = (_safe_log2(row_sum - anc_sum) + Pi_max + mt.unsqueeze(0)).detach()

    # logL
    root_ids = wave_layout['root_clade_ids']
    root_Pi = Pi[root_ids]
    lse = logsumexp2(root_Pi, dim=-1)
    import math
    numerator = lse - math.log2(S)
    denom = torch.log2(1 - torch.exp2(E).mean(dim=-1))
    logL = -(numerator - denom).sum()

    return logL, Pi, Pibar, E_out, log_pS, log_pD, log_pL, mt


# ─── Base forward ────────────────────────────────────────────────────────────
logL_base, Pi_base, Pibar_base, E_out_base, log_pS, log_pD, log_pL, mt = full_forward_cpu(theta)
print(f"Base logL = {logL_base:.10f}")

# ─── Analytical gradient ─────────────────────────────────────────────────────
grad_theta, statsG = implicit_grad_loglik_vjp_wave(
    wave_layout, sh,
    Pi_star_wave=Pi_base.detach(),
    Pibar_star_wave=Pibar_base.detach(),
    E_star=E_out_base['E'], E_s1=E_out_base['E_s1'],
    E_s2=E_out_base['E_s2'], Ebar=E_out_base['E_bar'],
    log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
    max_transfer_mat=mt,
    root_clade_ids_perm=wave_layout['root_clade_ids'],
    theta=theta,
    unnorm_row_max=unnorm_row_max,
    specieswise=False,
    device=device, dtype=dtype,
    neumann_terms=4, use_pruning=False,
    cg_tol=1e-10, cg_maxiter=1000,
    pibar_mode=pibar_mode,
    ancestors_T=ancestors_T,
)
print(f"Analytical gradient: {grad_theta}")
print(f"E adjoint stats: {statsG}")

# ─── FD gradient ─────────────────────────────────────────────────────────────
eps = 1e-5
print(f"\nFD with eps={eps}:")
for i in range(theta.numel()):
    theta_p = theta.clone(); theta_p[i] += eps
    theta_m = theta.clone(); theta_m[i] -= eps
    logL_p, *_ = full_forward_cpu(theta_p)
    logL_m, *_ = full_forward_cpu(theta_m)
    fd = (logL_p - logL_m) / (2 * eps)
    a = grad_theta[i].item()
    rel_err = abs(a - fd) / (abs(fd) + 1e-30)
    print(f"  theta[{i}]: FD={fd:.10e}, analytic={a:.10e}, rel_err={rel_err:.6e}")
