"""Decompose J^T into its paths and measure each one's spectral radius.
All pure-PyTorch. Works on CPU — no Triton/CUDA needed.
"""
import torch
import sys
sys.path.insert(0, '/home/enzo/Documents/git/gpurec/gpurec')

from gpurec.core.likelihood import E_fixed_point
from gpurec.core.backward import (
    _self_loop_differentiable, _self_loop_vjp_precompute, _self_loop_Jt_apply,
    _dts_cross_differentiable,
)
from gpurec.core.forward import NEG_INF
from gpurec.core.log2_utils import _safe_log2_internal as _safe_log2
from gpurec.core.scheduling import compute_clade_waves
from gpurec.core.batching import collate_wave, build_wave_layout
from gpurec.core.extract_parameters import extract_parameters_uniform
from gpurec.core.model import collate_gene_families
from tests.gradients.test_wave_gradient import _load_extension, _ROOT

# ─── Setup (replicate _setup_uniform on CPU) ────────────────────────────────
dtype = torch.float64
device = torch.device('cpu')
ds_name = "test_trees_20"
n_families = 1
D, L, T = 0.27, 0.27, 0.1

ext = _load_extension()
data_dir = _ROOT / "data" / ds_name
sp_path = str(data_dir / "sp.nwk")
gene_paths = sorted(data_dir.glob("g_*.nwk"))[:n_families]

INV_LN2 = 1.0 / 0.6931471805599453

batch_items = []
sr = None
for gp in gene_paths:
    raw = ext.preprocess(sp_path, [str(gp)])
    if sr is None:
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
    batch_items.append({
        "ccp": ch,
        "leaf_row_index": raw["leaf_row_index"].long(),
        "leaf_col_index": raw["leaf_col_index"].long(),
        "root_clade_id": int(cr["root_clade_id"]),
    })

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

log_pS, log_pD, log_pL, transfer_mat, mt = extract_parameters_uniform(
    theta, unnorm_row_max, specieswise=False,
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

# Build wave layout
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

# ─── Print parameters ────────────────────────────────────────────────────────
pT_val = 1 - 2**float(log_pS) - 2**float(log_pD) - 2**float(log_pL)
print(f"S = {S}")
print(f"pS={2**float(log_pS):.6f}, pD={2**float(log_pD):.6f}, pL={2**float(log_pL):.6f}, pT={pT_val:.6f}")
print(f"mt range: [{mt.min():.4f}, {mt.max():.4f}]  (exp2: [{mt.min().exp2():.6e}, {mt.max().exp2():.6e}])")
print(f"E range: [{E.min():.4f}, {E.max():.4f}]  (exp2: [{E.min().exp2():.6e}, {E.max().exp2():.6e}])")
print(f"Ebar range: [{Ebar.min():.4f}, {Ebar.max():.4f}]  (exp2: [{Ebar.min().exp2():.6e}, {Ebar.max().exp2():.6e}])")
print()

# ─── Forward pass: pure PyTorch ──────────────────────────────────────────────
wave_metas = wave_layout['wave_metas']
n_waves = len(wave_metas)
total_clades = wave_layout['ccp_helpers']['C']

Pi = torch.full((total_clades, S), NEG_INF, dtype=dtype, device=device)
Pibar = torch.full((total_clades, S), NEG_INF, dtype=dtype, device=device)

DL_const = 1.0 + log_pD + E  # matches Pi_wave_forward line 601
SL1_const = log_pS + E_s2
SL2_const = log_pS + E_s1
leaf_row = wave_layout['leaf_row_index'].long()
leaf_col = wave_layout['leaf_col_index'].long()

for wi in range(n_waves):
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
    mask_lr = (leaf_row >= ws) & (leaf_row < we)
    if mask_lr.any():
        leaf_wt[leaf_row[mask_lr] - ws, leaf_col[mask_lr]] = 0.0

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

print(f"Forward done ({n_waves} waves).\n")

# ─── Analyze each wave ───────────────────────────────────────────────────────
for wi in range(n_waves):
    meta = wave_metas[wi]
    ws = meta['start']
    W = meta['W']
    we = ws + W
    n = W * S

    if n > 3000:
        print(f"Wave {wi}: W={W}, dim={n}, skipping (too large)\n")
        continue

    Pi_W = Pi[ws:we]
    Pibar_W = Pibar[ws:we]

    if meta['has_splits']:
        dts_r = _dts_cross_differentiable(
            Pi, Pibar, meta, sp_child1, sp_child2,
            log_pD, log_pS, S, device, dtype,
        )
    else:
        dts_r = None

    leaf_wt = torch.full((W, S), NEG_INF, dtype=dtype, device=device)
    mask_lr = (leaf_row >= ws) & (leaf_row < we)
    if mask_lr.any():
        leaf_wt[leaf_row[mask_lr] - ws, leaf_col[mask_lr]] = 0.0

    mt_w = mt.unsqueeze(0).expand(W, -1)

    def _exp(t):
        return t.unsqueeze(0).expand(W, -1) if t.ndim == 1 else t

    ingredients = _self_loop_vjp_precompute(
        Pi_W, Pibar_W, dts_r,
        mt_w, _exp(DL_const), _exp(Ebar), _exp(E), _exp(SL1_const), _exp(SL2_const),
        sp_child1, sp_child2, leaf_wt, S,
        pibar_mode, None, ancestors_T,
    )

    w_L = ingredients['w_L']
    w_terms = ingredients['w_terms']

    eff_pibar = w_L * w_terms[2]
    diag_weight = w_L * (w_terms[0] + w_terms[1])

    # Build full Jacobian matrix by applying J^T to basis vectors
    J_full = torch.zeros(n, n, dtype=dtype, device=device)
    for i in range(n):
        e_i = torch.zeros(W, S, dtype=dtype, device=device)
        e_i.view(-1)[i] = 1.0
        J_full[:, i] = _self_loop_Jt_apply(
            e_i, ingredients, sp_child1, sp_child2, S, W,
            pibar_mode, None, ancestors_T,
        ).reshape(-1)

    eigs = torch.linalg.eigvals(J_full)
    eigs_abs = eigs.abs()
    top_idx = eigs_abs.argmax()

    names = ['DL  ', 'TL1 ', 'TL2 ', 'SL1 ', 'SL2 ', 'leaf']
    print(f"Wave {wi}: W={W}, has_splits={meta['has_splits']}, dim={n}")
    print(f"  w_L: mean={w_L.mean():.6f}, max={w_L.max():.6f}, min={w_L.min():.6f}")
    for t in range(6):
        wt = w_terms[t]
        eff = (w_L * wt).mean()
        print(f"  w_terms[{t}] {names[t]}: mean={wt.mean():.6f}, max={wt.max():.6f} | eff(w_L*wt)={eff:.6f}")
    print(f"  diag weight w_L*(w0+w1): mean={diag_weight.mean():.6f}, max={diag_weight.max():.6f}")
    print(f"  pibar weight w_L*w2:     mean={eff_pibar.mean():.6f}, max={eff_pibar.max():.6f}")
    print(f"  ρ(J^T)        = {eigs_abs.max():.6f}")
    print(f"  top-5 |λ|     = {[f'{x:.4f}' for x in eigs_abs.topk(min(5,n)).values.tolist()]}")
    print(f"  dominant λ    = {eigs[top_idx].real:.6f} + {eigs[top_idx].imag:.6f}i")
    print()
