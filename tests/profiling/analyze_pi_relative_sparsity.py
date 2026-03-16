"""Analyze Pi sparsity RELATIVE to row max (what matters for the matmul).

The matmul computes exp2(Pi - Pi_max) @ transfer_mat_T.
Entries where (Pi - Pi_max) < -149 underflow to exactly 0 in float32.
How many entries per row are within various gaps of the max?
"""

import math
import sys
import time
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

from src.core.preprocess_cpp import _load_extension
from src.core.extract_parameters import extract_parameters
from src.core.likelihood import E_fixed_point, Pi_wave_forward, compute_log_likelihood
from src.core.scheduling import compute_clade_waves
from src.core.batching import collate_gene_families, collate_wave, build_wave_layout

_INV_LN2 = 1.0 / math.log(2.0)


def main(n_fam=10):
    device = torch.device("cuda")
    dtype = torch.float32
    ext = _load_extension()
    data_dir = _ROOT / "tests" / "data" / "test_trees_1000"
    sp_path = str(data_dir / "sp.nwk")
    gene_paths = sorted(data_dir.glob("g_*.nwk"))[:n_fam]

    batch_items = []
    sr = None
    for gp in gene_paths:
        raw = ext.preprocess(sp_path, [str(gp)])
        sr, cr = raw["species"], raw["ccp"]
        ch = {
            "split_leftrights_sorted": cr["split_leftrights_sorted"],
            "log_split_probs_sorted": cr["log_split_probs_sorted"].to(dtype=dtype) * _INV_LN2,
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
        if "phased_waves" in cr:
            ch["phased_waves"] = cr["phased_waves"]
            ch["phased_phases"] = cr["phased_phases"]
        batch_items.append({
            "ccp": ch,
            "leaf_row_index": raw["leaf_row_index"].long(),
            "leaf_col_index": raw["leaf_col_index"].long(),
            "root_clade_id": int(cr["root_clade_id"]),
        })

    D, L, T = 0.05, 0.05, 0.05
    sh = {
        "S": int(sr["S"]),
        "names": sr["names"],
        "s_P_indexes": sr["s_P_indexes"].to(device=device),
        "s_C12_indexes": sr["s_C12_indexes"].to(device=device),
        "Recipients_mat": sr["Recipients_mat"].to(dtype=dtype, device=device),
    }
    theta = torch.log2(torch.tensor([D, L, T], dtype=dtype, device=device))
    tm = torch.log2(sh["Recipients_mat"])
    pS, pD, pL, tf, mt = extract_parameters(
        theta, tm, genewise=False, specieswise=False, pairwise=False
    )
    mv = mt.squeeze(-1) if mt.ndim == 2 else mt
    Eo = E_fixed_point(
        species_helpers=sh, log_pS=pS, log_pD=pD, log_pL=pL,
        transfer_mat=tf, max_transfer_mat=mv, max_iters=2000,
        tolerance=1e-3, warm_start_E=None, dtype=dtype, device=device,
    )

    batched = collate_gene_families(batch_items, dtype=dtype, device=device)
    ccp = batched["ccp"]
    li = batched["leaf_row_index"]
    lc = batched["leaf_col_index"]
    root_ids = batched["root_clade_ids"]
    family_meta = batched["family_meta"]

    families_waves, families_phases = [], []
    for item in batch_items:
        ch_dev = {k: (v.to(device) if torch.is_tensor(v) else v)
                  for k, v in item["ccp"].items()}
        w, p = compute_clade_waves(ch_dev)
        families_waves.append(w)
        families_phases.append(p)
    offsets = [m["clade_offset"] for m in family_meta]
    cross_waves = collate_wave(families_waves, offsets)
    max_n = max(len(p) for p in families_phases)
    cross_phases = []
    for ki in range(max_n):
        pk = 1
        for fp in families_phases:
            if ki < len(fp):
                pk = max(pk, fp[ki])
        cross_phases.append(pk)

    wave_layout = build_wave_layout(
        waves=cross_waves, phases=cross_phases,
        ccp_helpers=ccp, leaf_row_index=li, leaf_col_index=lc,
        root_clade_ids=root_ids, device=device, dtype=dtype,
    )

    # Run to convergence
    wv = Pi_wave_forward(
        wave_layout=wave_layout, species_helpers=sh,
        E=Eo["E"], Ebar=Eo["E_bar"], E_s1=Eo["E_s1"], E_s2=Eo["E_s2"],
        log_pS=pS, log_pD=pD, log_pL=pL,
        transfer_mat=tf, max_transfer_mat=mv,
        device=device, dtype=dtype,
    )
    Pi = wv["Pi"]  # [C, S]
    C, S = Pi.shape
    print(f"Pi shape: [{C}, {S}]")

    # ---- Relative sparsity: Pi - Pi_max ----
    Pi_max = Pi.max(dim=1, keepdim=True).values  # [C, 1]
    Pi_rel = Pi - Pi_max  # [C, S], all <= 0

    print(f"\nPi_max stats: min={Pi_max.min().item():.2f}, "
          f"max={Pi_max.max().item():.2f}, "
          f"mean={Pi_max.mean().item():.2f}")

    print(f"\n--- Entries within gap of row max (Pi - Pi_max > -gap) ---")
    for gap in [1, 5, 10, 20, 50, 100, 126, 149, 500, 1000]:
        nnz = (Pi_rel > -gap).float().sum(dim=1)
        print(f"  gap={gap:5d}: mean nnz={nnz.mean().item():7.1f} / {S}, "
              f"min={nnz.min().item():.0f}, max={nnz.max().item():.0f}, "
              f"sparsity={(1 - nnz.mean().item()/S)*100:.1f}%")

    # ---- What does exp2(Pi_rel) look like? ----
    print(f"\n--- exp2(Pi - Pi_max) sparsity (float32 underflow at -149) ---")
    Pi_linear = torch.exp2(Pi_rel)  # [C, S], exp2 of relative values
    nnz_linear = (Pi_linear > 0).float().sum(dim=1)
    print(f"  Non-zero entries: mean={nnz_linear.mean().item():.1f} / {S}, "
          f"min={nnz_linear.min().item():.0f}, max={nnz_linear.max().item():.0f}")

    # ---- Mass fraction with relative values ----
    print(f"\n--- Mass fraction of top-k in exp2(Pi - Pi_max) ---")
    Pi_linear_row_sums = Pi_linear.sum(dim=1)  # [C]
    Pi_sorted_rel = Pi_rel.sort(dim=1, descending=True).values
    for k in [1, 2, 4, 8, 16, 32, 64]:
        topk_linear = torch.exp2(Pi_sorted_rel[:, :k])
        topk_sums = topk_linear.sum(dim=1)
        frac = topk_sums / Pi_linear_row_sums.clamp(min=1e-30)
        # Clamp fractions > 1 (can happen due to float precision)
        frac = frac.clamp(max=1.0)
        print(f"  top-{k:3d}: mean={frac.mean().item():.4f}, "
              f"min={frac.min().item():.4f}, "
              f"median={frac.median().item():.4f}, "
              f"p95={frac.quantile(0.95).item():.4f}")

    # ---- Histogram of Pi_rel values (what's the distribution?) ----
    print(f"\n--- Distribution of Pi - Pi_max ---")
    print(f"  (sampling 1000 random rows)")
    idx = torch.randperm(C, device=device)[:min(1000, C)]
    sample_rel = Pi_rel[idx].flatten()
    for pct in [0.01, 0.1, 1, 5, 10, 25, 50, 75, 90, 95, 99]:
        val = sample_rel.quantile(pct / 100).item()
        print(f"  p{pct:5.1f}: {val:.2f}")

    # ---- Per-wave analysis: internal waves only ----
    print(f"\n--- Per-wave relative sparsity (internal waves) ---")
    inv_perm = wave_layout["inv_perm"]
    Pi_wave = Pi[inv_perm]
    wave_metas = wave_layout["wave_metas"]
    for wi, meta in enumerate(wave_metas):
        if meta["phase"] != 2:
            continue
        ws, we = meta["start"], meta["end"]
        W = meta["W"]
        if W < 50:
            continue
        Pi_W = Pi_wave[ws:we]
        Pi_W_max = Pi_W.max(dim=1, keepdim=True).values
        Pi_W_rel = Pi_W - Pi_W_max
        for gap in [10, 50, 149]:
            nnz = (Pi_W_rel > -gap).float().sum(dim=1)
            print(f"  Wave {wi:2d} (W={W:5d}, phase=2): gap={gap:3d} → "
                  f"mean nnz={nnz.mean().item():.1f}")
        # Only print first 5
        if wi >= 10:
            print("  ...")
            break


if __name__ == "__main__":
    main(n_fam=10)
