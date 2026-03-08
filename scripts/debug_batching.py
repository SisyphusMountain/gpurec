#!/usr/bin/env python3
"""
Debug batched vs single-family likelihood and CCP collation.

Runs on a small dataset by default (tests/data/test_trees_1) to compare:
- Raw CCP helpers (single) vs collated (batch of size 1)
- Pi_single vs corresponding slice of Pi_batched
- Single vs batched log-likelihoods

No code changes are made — this is a pure diagnostic utility.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
sys.path.append("/home/enzo/Documents/git/WP2")
from typing import Dict, Any, List

import torch


def parse_args():
    p = argparse.ArgumentParser()
    repo_root = Path(__file__).resolve().parents[1]
    default_sp = repo_root / "tests" / "data" / "test_mixed_200" / "sp.nwk"
    default_g = repo_root / "tests" / "data" / "test_mixed_200" / "g.nwk"
    p.add_argument("--species", type=str, default=str(default_sp))
    p.add_argument("--gene", type=str, action="append", default=[str(default_g)], help="Repeat to add multiple gene trees")
    p.add_argument("--device", type=str, default=("cpu" if not torch.cuda.is_available() else "cuda"))
    p.add_argument("--dtype", type=str, default="float32", choices=["float32","float64"])  # favor determinism
    p.add_argument("--itersE", type=int, default=500)
    p.add_argument("--itersPi", type=int, default=500)
    return p.parse_args()


def to_dev_dtype(x, device, dtype):
    if torch.is_tensor(x):
        return x.to(device=device, dtype=(dtype if x.dtype.is_floating_point else None))
    return x


def compare_ccp(single: Dict[str, Any], batched: Dict[str, Any]) -> List[str]:
    mism = []
    # scalar keys
    for k in ["C","N_splits","end_rows_ge2","num_segs_ge2","num_segs_eq1"]:
        if int(single[k]) != int(batched[k]):
            mism.append(f"scalar {k}: {single[k]} != {batched[k]}")
    # tensor keys
    for k in ["split_leftrights_sorted","log_split_probs_sorted","seg_parent_ids","ptr_ge2"]:
        a = single[k]
        b = batched[k]
        if a.shape != b.shape:
            mism.append(f"shape {k}: {tuple(a.shape)} != {tuple(b.shape)}")
        else:
            if not torch.allclose(a, b):
                # report a small sample of differences
                diff = (a - b).abs().max().item() if a.dtype.is_floating_point else (a != b).sum().item()
                mism.append(f"values {k}: mismatch (max_abs or count={diff})")
    return mism


def run():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from core.model import GeneDataset
    from core.batching import collate_gene_families
    from core.extract_parameters import extract_parameters
    from core.likelihood import E_fixed_point, Pi_fixed_point, compute_log_likelihood

    device = torch.device(args.device)
    dtype = torch.float32 if args.dtype == "float32" else torch.float64

    print(f"Device={device}, dtype={dtype}")
    sp = args.species
    gs = args.gene
    print(f"Species: {sp}")
    print(f"Genes ({len(gs)}): {gs}")

    # Build dataset
    ds = GeneDataset(sp, gs*2, genewise=False, specieswise=False, pairwise=False, device=device, dtype=dtype)

    fam = ds.families[0]
    # Single-family CCP (move to device/dtype)
    single_ccp = {k: to_dev_dtype(v, device, dtype) if torch.is_tensor(v) else v for k, v in fam['ccp_helpers'].items()}
    single_leaf_row = to_dev_dtype(fam['leaf_row_index'], device, dtype).to(torch.long)
    single_leaf_col = to_dev_dtype(fam['leaf_col_index'], device, dtype).to(torch.long)
    root0 = int(fam['root_clade_id'])

    # Collate batch of size 1
    batched = collate_gene_families([{
        'ccp': fam['ccp_helpers'],
        'leaf_row_index': fam['leaf_row_index'],
        'leaf_col_index': fam['leaf_col_index'],
        'root_clade_id': int(fam['root_clade_id']),
    }], dtype=dtype, device=device)

    bat_ccp = batched['ccp']
    print("CCP single vs collated(batch=1) diffs:")
    diffs = compare_ccp(single_ccp, bat_ccp)
    if not diffs:
        print("  None — exact match")
    else:
        for d in diffs:
            print("  ", d)

    # Parameters (shared)
    theta0 = ds.families[0]['theta'].to(device=device, dtype=dtype)
    log_pS, log_pD, log_pL, T, T_max = extract_parameters(theta0, ds.tr_mat_unnormalized.to(device=device, dtype=dtype), genewise=False, specieswise=False, pairwise=False)

    # Species helpers
    def _mv(t):
        return to_dev_dtype(t, device, dtype) if torch.is_tensor(t) else t
    species = {k: _mv(v) for k, v in ds.species_helpers.items()}

    # E
    E_out = E_fixed_point(species, log_pS, log_pD, log_pL, T, T_max.squeeze(-1), args.itersE, 1e-12, None, dtype, device)
    E = E_out['E']
    E_s1 = E_out['E_s1']
    E_s2 = E_out['E_s2']
    Ebar = E_out['E_bar']

    # Pi single
    Pi_single_out = Pi_fixed_point(
        ccp_helpers=single_ccp,
        species_helpers=species,
        leaf_row_index=single_leaf_row,
        leaf_col_index=single_leaf_col,
        E=E, Ebar=Ebar, E_s1=E_s1, E_s2=E_s2,
        log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
        transfer_mat_T=T.transpose(-1, -2), max_transfer_mat=T_max.squeeze(-1),
        max_iters=args.itersPi, tolerance=1e-12,
        warm_start_Pi=None, device=device, dtype=dtype,
        genewise=False, specieswise=False, pairwise=False,
        clades_per_gene=None, batch_info=None,
    )
    Pi_single = Pi_single_out['Pi']
    ll_single = compute_log_likelihood(Pi_single, E, root0)
    print(f"ll_single: {float(ll_single):.12f}")

    # Pi batched (batch size 1)
    Pi_batch_out = Pi_fixed_point(
        ccp_helpers=bat_ccp,
        species_helpers=species,
        leaf_row_index=batched['leaf_row_index'],
        leaf_col_index=batched['leaf_col_index'],
        E=E, Ebar=Ebar, E_s1=E_s1, E_s2=E_s2,
        log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
        transfer_mat_T=T.transpose(-1, -2), max_transfer_mat=T_max.squeeze(-1),
        max_iters=args.itersPi, tolerance=1e-12,
        warm_start_Pi=None, device=device, dtype=dtype,
        genewise=False, specieswise=False, pairwise=False,
        clades_per_gene=None, batch_info=None,
    )
    Pi_b = Pi_batch_out['Pi']
    # Slice corresponding to family 0
    C0 = int(single_ccp['C'])
    Pi_b0 = Pi_b[:C0]
    max_diff = (Pi_single - Pi_b0).abs().max().item()
    print(f"Pi_single vs Pi_batched(slice) max_abs_diff: {max_diff:.3e}")

    # Now do batched likelihood through GeneDataset API
    ll_list = [ds.compute_likelihood(0, max_iters_E=args.itersE, max_iters_Pi=args.itersPi)['log_likelihood']]
    ll_batch = ds.compute_likelihood_batch(indices=list(range(ds.num_families)), max_iters_E=args.itersE, max_iters_Pi=args.itersPi)
    print(f"compute_likelihood API: single={ll_list[0]:.12f}, batched={ll_batch}")

    # If multiple identical genes provided, collate two and compare per-family slices
    if len(gs) >= 2:
        fam2 = ds.families[1]
        items = []
        for f in (fam, fam2):
            items.append({
                'ccp': f['ccp_helpers'],
                'leaf_row_index': f['leaf_row_index'],
                'leaf_col_index': f['leaf_col_index'],
                'root_clade_id': int(f['root_clade_id']),
            })
        bat2 = collate_gene_families(items, dtype=dtype, device=device)
        C1 = int(fam['ccp_helpers']['C'])
        C2 = int(fam2['ccp_helpers']['C'])
        print(f"Two-family collate: C1={C1}, C2={C2}, total={int(bat2['ccp']['C'])}")
        # Compare per-family slices against singles with clade offsets applied
        for i, f in enumerate((fam, fam2)):
            off = 0 if i == 0 else C1
            # Check keys
            for k in ["split_leftrights_sorted","log_split_probs_sorted"]:
                a = to_dev_dtype(f['ccp_helpers'][k], device, dtype)
                b = bat2['ccp'][k]
                if k == "split_leftrights_sorted":
                    # apply clade offset
                    a = a + off
                a_ge2 = a[: int(f['ccp_helpers']['end_rows_ge2'])]
                b_ge2 = b[int(0 if i == 0 else bat2['ccp']['end_rows_ge2']) - int(f['ccp_helpers']['end_rows_ge2']): int(0 if i == 0 else bat2['ccp']['end_rows_ge2'])]
            # Additional deep checks can be added as needed


if __name__ == "__main__":
    run()
