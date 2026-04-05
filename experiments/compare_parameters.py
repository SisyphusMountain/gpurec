"""
Compare optimized parameters (our optimizer) vs AleRax reference on test_trees_100.

Usage:
    python experiments/compare_parameters.py [--families N]
"""

import argparse
import math
import time
from pathlib import Path

import torch
import numpy as np

DATA_DIR = Path(__file__).resolve().parents[1] / "tests" / "data" / "test_trees_100"
LN2 = math.log(2)


# ──────────────────────────────────────────────────────────────────────────────
# AleRax parameter readers
# ──────────────────────────────────────────────────────────────────────────────

def read_alerax_global():
    """Return (D, L, T) from the global run — same for every node."""
    lines = (DATA_DIR / "output_global/model_parameters/model_parameters.txt").read_text().splitlines()
    _, d, l, t = lines[1].split()
    return float(d), float(l), float(t)


def read_alerax_specieswise():
    """Return dict: node_name -> (D, L, T)."""
    d = {}
    lines = (DATA_DIR / "output_specieswise/model_parameters/model_parameters.txt").read_text().splitlines()
    for line in lines[1:]:
        parts = line.split()
        d[parts[0]] = (float(parts[1]), float(parts[2]), float(parts[3]))
    return d


def read_alerax_genewise(G):
    """Return ndarray [G, 3] of (D, L, T) for families 0..G-1."""
    rates = []
    for g in range(G):
        path = DATA_DIR / f"output_genewise/model_parameters/family_{g:04d}_rates.txt"
        lines = path.read_text().splitlines()
        d, l, t = lines[1].split()
        rates.append((float(d), float(l), float(t)))
    return np.array(rates)


# ──────────────────────────────────────────────────────────────────────────────
# Wave layout builder (same pattern as validate_three_modes.py)
# ──────────────────────────────────────────────────────────────────────────────

def _build_wave_layout(ds, device, dtype):
    from gpurec.core.batching import collate_gene_families, collate_wave, build_wave_layout
    from gpurec.core.scheduling import compute_clade_waves

    items = [{"ccp": fam["ccp_helpers"], "leaf_row_index": fam["leaf_row_index"],
               "leaf_col_index": fam["leaf_col_index"], "root_clade_id": int(fam["root_clade_id"])}
             for fam in ds.families]
    batched = collate_gene_families(items, dtype=dtype, device=device)

    families_waves, families_phases = [], []
    for fam in ds.families:
        w, p = compute_clade_waves(fam["ccp_helpers"])
        families_waves.append(w)
        families_phases.append(p)

    offsets = [m["clade_offset"] for m in batched["family_meta"]]
    cross_waves = collate_wave(families_waves, offsets)

    max_n_waves = max(len(p) for p in families_phases)
    cross_phases = []
    for k in range(max_n_waves):
        phase_k = max((fp[k] if k < len(fp) else 1) for fp in families_phases)
        cross_phases.append(phase_k)

    family_clade_counts  = [m["C"]            for m in batched["family_meta"]]
    family_clade_offsets = [m["clade_offset"] for m in batched["family_meta"]]

    wl = build_wave_layout(
        waves=cross_waves, phases=cross_phases,
        ccp_helpers=batched["ccp"],
        leaf_row_index=batched["leaf_row_index"],
        leaf_col_index=batched["leaf_col_index"],
        root_clade_ids=batched["root_clade_ids"],
        device=device, dtype=dtype,
        family_clade_counts=family_clade_counts,
        family_clade_offsets=family_clade_offsets,
    )
    return wl, batched["root_clade_ids"]


def _sp_helpers_gpu(ds, device, dtype):
    return {k: (v.to(device=device, dtype=dtype) if torch.is_tensor(v) and v.is_floating_point()
                else v.to(device=device) if torch.is_tensor(v) else v)
            for k, v in ds.species_helpers.items()}


def _pct(val, ref):
    return 100.0 * val / ref if ref != 0 else float("inf")


# ──────────────────────────────────────────────────────────────────────────────
# Per-mode comparison helpers
# ──────────────────────────────────────────────────────────────────────────────

def _print_stats(label, diffs_rel_pct):
    arr = np.array(diffs_rel_pct)
    print(f"  {label:30s}  mean={arr.mean():+.2f}%  median={np.median(arr):+.2f}%  "
          f"p90={np.percentile(arr,90):+.2f}%  max|err|={np.abs(arr).max():.2f}%")


def compare_global(our_rates_np, alerax):
    al_D, al_L, al_T = alerax
    print(f"\n{'─'*70}")
    print("  GLOBAL PARAMETERS")
    print(f"  {'':20s}  {'Ours':>12}  {'AleRax':>12}  {'Diff %':>10}")
    for name, ours, ref in [("D", our_rates_np[0], al_D),
                             ("L", our_rates_np[1], al_L),
                             ("T", our_rates_np[2], al_T)]:
        pct = _pct(ours - ref, ref)
        print(f"  {name:20s}  {ours:12.6f}  {ref:12.6f}  {pct:+10.3f}%")


def compare_specieswise(our_rates_tensor, ds, alerax_dict):
    """our_rates_tensor: [S, 3] on CPU (already 2^theta)."""
    n2i = ds.species_helpers["species_name_to_index"]
    names = ds.species_helpers["names"]

    diffs = {"D": [], "L": [], "T": []}
    abs_diffs = {"D": [], "L": [], "T": []}
    n_matched = 0
    for node_name, al_rates in alerax_dict.items():
        if node_name not in n2i:
            continue
        idx = n2i[node_name]
        our = our_rates_tensor[idx].numpy()
        for j, key in enumerate(["D", "L", "T"]):
            ref = al_rates[j]
            rel = _pct(float(our[j]) - ref, ref)
            diffs[key].append(rel)
            abs_diffs[key].append(abs(float(our[j]) - ref))
        n_matched += 1

    print(f"\n{'─'*70}")
    print(f"  SPECIESWISE PARAMETERS  ({n_matched} nodes matched)")
    for key in ["D", "L", "T"]:
        _print_stats(key, diffs[key])
    # Overall
    all_rel = diffs["D"] + diffs["L"] + diffs["T"]
    all_abs = abs_diffs["D"] + abs_diffs["L"] + abs_diffs["T"]
    arr_rel = np.array(all_rel)
    arr_abs = np.array(all_abs)
    print(f"  {'ALL (D+L+T)':30s}  mean|err|={np.abs(arr_rel).mean():.2f}%  "
          f"median|err|={np.median(np.abs(arr_rel)):.2f}%  "
          f"max|err|={np.abs(arr_rel).max():.2f}%")
    print(f"  Absolute  mean={arr_abs.mean():.5f}  max={arr_abs.max():.5f}")


def compare_genewise(our_rates_tensor, alerax_rates_np, nan_mask=None):
    """our_rates_tensor: [G, 3] CPU; alerax_rates_np: [G, 3]."""
    G = our_rates_tensor.shape[0]
    our = our_rates_tensor.numpy()

    diffs = {"D": [], "L": [], "T": []}
    abs_diffs = {"D": [], "L": [], "T": []}
    for g in range(G):
        if nan_mask is not None and nan_mask[g]:
            continue
        for j, key in enumerate(["D", "L", "T"]):
            ref = alerax_rates_np[g, j]
            val = float(our[g, j])
            rel = _pct(val - ref, ref)
            diffs[key].append(rel)
            abs_diffs[key].append(abs(val - ref))

    n_valid = len(diffs["D"])
    print(f"\n{'─'*70}")
    print(f"  GENEWISE PARAMETERS  ({n_valid}/{G} valid families)")
    for key in ["D", "L", "T"]:
        _print_stats(key, diffs[key])
    all_rel = diffs["D"] + diffs["L"] + diffs["T"]
    all_abs = abs_diffs["D"] + abs_diffs["L"] + abs_diffs["T"]
    arr_rel = np.array(all_rel)
    arr_abs = np.array(all_abs)
    print(f"  {'ALL (D+L+T)':30s}  mean|err|={np.abs(arr_rel).mean():.2f}%  "
          f"median|err|={np.median(np.abs(arr_rel)):.2f}%  "
          f"max|err|={np.abs(arr_rel).max():.2f}%")
    print(f"  Absolute  mean={arr_abs.mean():.5f}  max={arr_abs.max():.5f}")
    # Worst families
    per_fam = [max(abs(float(our[g,j]) - alerax_rates_np[g,j]) / max(alerax_rates_np[g,j], 1e-12)
                   for j in range(3))
               for g in range(G)
               if nan_mask is None or not nan_mask[g]]
    worst = sorted(enumerate(per_fam), key=lambda x: -x[1])[:5]
    print(f"  Worst 5 families (max relative error):")
    for rank, (i, err) in enumerate(worst):
        g = [g for g in range(G) if nan_mask is None or not nan_mask[g]][i]
        print(f"    family {g:03d}: {err*100:.1f}%  "
              f"our=({our[g,0]:.4f},{our[g,1]:.4f},{our[g,2]:.4f})  "
              f"alerax=({alerax_rates_np[g,0]:.4f},{alerax_rates_np[g,1]:.4f},{alerax_rates_np[g,2]:.4f})")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--families", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    device = torch.device("cuda")
    dtype  = torch.float32

    from gpurec.core.model import GeneDataset
    from gpurec.optimization.theta_optimizer import optimize_theta_wave, optimize_theta_genewise

    sp_path    = str(DATA_DIR / "sp.nwk")
    gene_paths = sorted(DATA_DIR.glob("g_*.nwk"))
    if args.families > 0:
        gene_paths = gene_paths[:args.families]
    gene_paths = [str(g) for g in gene_paths]
    G = len(gene_paths)
    print(f"Using {G} families, {args.steps} L-BFGS steps")

    # ── Global ────────────────────────────────────────────────────────────────
    print("\n[1/3] Global ...")
    ds_g = GeneDataset(sp_path, gene_paths, genewise=False, specieswise=False,
                       pairwise=False, dtype=dtype, device=device)
    wl, rids = _build_wave_layout(ds_g, device, dtype)
    sp_h = _sp_helpers_gpu(ds_g, device, dtype)
    theta_init = math.log2(0.02) * torch.ones(3, dtype=dtype, device=device)
    t0 = time.time()
    res_g = optimize_theta_wave(wl, sp_h, rids,
                                ds_g.unnorm_row_max.to(device=device, dtype=dtype),
                                theta_init, steps=args.steps, optimizer="lbfgs",
                                specieswise=False, pibar_mode="uniform",
                                device=device, dtype=dtype, verbose=True)
    print(f"  Done in {time.time()-t0:.1f}s  NLL={res_g['negative_log_likelihood']:.3f}")
    global_rates = res_g["rates"].cpu()  # [3], already 2^theta
    compare_global(global_rates.numpy(), read_alerax_global())

    # ── Specieswise ───────────────────────────────────────────────────────────
    print("\n[2/3] Specieswise ...")
    ds_sw = GeneDataset(sp_path, gene_paths, genewise=False, specieswise=True,
                        pairwise=False, dtype=dtype, device=device)
    S = ds_sw.S
    wl_sw, rids_sw = _build_wave_layout(ds_sw, device, dtype)
    sp_h_sw = _sp_helpers_gpu(ds_sw, device, dtype)
    theta_init_sw = math.log2(float(global_rates.mean())) * torch.ones(S, 3, dtype=dtype, device=device)
    t0 = time.time()
    res_sw = optimize_theta_wave(wl_sw, sp_h_sw, rids_sw,
                                 ds_sw.unnorm_row_max.to(device=device, dtype=dtype),
                                 theta_init_sw, steps=args.steps, optimizer="lbfgs",
                                 specieswise=True, pibar_mode="uniform",
                                 device=device, dtype=dtype, verbose=True)
    print(f"  Done in {time.time()-t0:.1f}s  NLL={res_sw['negative_log_likelihood']:.3f}")
    sw_rates = res_sw["rates"].cpu()  # [S, 3]
    compare_specieswise(sw_rates, ds_sw, read_alerax_specieswise())

    # ── Genewise ──────────────────────────────────────────────────────────────
    print("\n[3/3] Genewise ...")
    ds_gw = GeneDataset(sp_path, gene_paths, genewise=True, specieswise=False,
                        pairwise=False, dtype=dtype, device=device)
    theta_init_gw = math.log2(float(global_rates.mean())) * torch.ones(G, 3, dtype=dtype, device=device)
    t0 = time.time()
    res_gw = optimize_theta_genewise(
        families=ds_gw.families, species_helpers=ds_gw.species_helpers,
        unnorm_row_max=ds_gw.unnorm_row_max,
        theta_init=theta_init_gw,
        max_steps=args.steps, lbfgs_m=10, grad_tol=1e-5,
        pibar_mode="uniform", device=device, dtype=dtype,
    )
    print(f"  Done in {time.time()-t0:.1f}s")
    gw_rates = res_gw["rates"].cpu()  # [G, 3]
    nan_mask_np = torch.isnan(res_gw["nll"]).cpu().numpy()
    compare_genewise(gw_rates, read_alerax_genewise(G), nan_mask=nan_mask_np)

    print(f"\n{'═'*70}")


if __name__ == "__main__":
    main()
