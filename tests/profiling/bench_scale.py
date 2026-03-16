#!/usr/bin/env python3
"""Benchmark cross-family wave batching at scale.

Tests throughput (families/sec) for different family counts, max wave sizes,
and chunk sizes, using the phased cross-family scheduler.

Usage:
    python tests/profiling/bench_scale.py [--dataset test_trees_1000] [--max-families 100]
    python tests/profiling/bench_scale.py --dataset test_trees_100 --max-families 100 --repeat 10
"""

import argparse
import gc
import math
import pathlib
import sys
import time

import torch

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.preprocess_cpp import _load_extension as _load_cpp_ext
from src.core.extract_parameters import extract_parameters
from src.core.likelihood import E_fixed_point, Pi_wave_forward, compute_log_likelihood
from src.core.scheduling import compute_clade_waves
from src.core.batching import collate_gene_families, collate_wave, collate_wave_cross, build_wave_layout

_INV = 1.0 / math.log(2.0)


def preprocess_families(ext, sp_path, gene_paths, device, dtype):
    """Preprocess all families, return batch items + species helpers."""
    batch_items = []
    sr = None
    for gp in gene_paths:
        raw = ext.preprocess(sp_path, [str(gp)])
        sr_i, cr = raw["species"], raw["ccp"]
        if sr is None:
            sr = sr_i
        ch = {
            "split_leftrights_sorted": cr["split_leftrights_sorted"],
            "log_split_probs_sorted": cr["log_split_probs_sorted"].to(dtype=dtype) * _INV,
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
    sh = {
        "S": int(sr["S"]),
        "names": sr["names"],
        "s_P_indexes": sr["s_P_indexes"].to(device=device),
        "s_C12_indexes": sr["s_C12_indexes"].to(device=device),
        "Recipients_mat": sr["Recipients_mat"].to(dtype=dtype, device=device),
    }
    theta = torch.log2(torch.tensor([0.05, 0.05, 0.05], dtype=dtype, device=device))
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
    return batch_items, sh, pS, pD, pL, tf, mv, Eo


def run_batched_cross(items, sh, pS, pD, pL, tf, mv, Eo, device, dtype,
                      max_wave_size=256, tol=1e-3, chunk_size=None):
    """Run cross-family batched wave forward with phased cross scheduler."""
    if chunk_size is not None and len(items) > chunk_size:
        all_logLs = []
        total_iters = 0
        max_nw = 0
        for start in range(0, len(items), chunk_size):
            chunk = items[start:start + chunk_size]
            logLs, iters, nw = run_batched_cross(
                chunk, sh, pS, pD, pL, tf, mv, Eo, device, dtype,
                max_wave_size=max_wave_size, tol=tol, chunk_size=None
            )
            all_logLs.extend(logLs)
            total_iters += iters
            max_nw = max(max_nw, nw)
        return all_logLs, total_iters, max_nw

    batched = collate_gene_families(items, dtype=dtype, device=device)
    ccp = batched["ccp"]
    meta = batched["family_meta"]

    cross_waves, cross_phases = collate_wave_cross(items, meta, max_wave_size=max_wave_size)

    wave_layout = build_wave_layout(
        waves=cross_waves, phases=cross_phases,
        ccp_helpers=ccp,
        leaf_row_index=batched["leaf_row_index"],
        leaf_col_index=batched["leaf_col_index"],
        root_clade_ids=batched["root_clade_ids"],
        device=device, dtype=dtype,
    )

    wv = Pi_wave_forward(
        wave_layout=wave_layout, species_helpers=sh,
        E=Eo["E"], Ebar=Eo["E_bar"], E_s1=Eo["E_s1"], E_s2=Eo["E_s2"],
        log_pS=pS, log_pD=pD, log_pL=pL,
        transfer_mat=tf, max_transfer_mat=mv,
        device=device, dtype=dtype,
        local_iters=1000, local_tolerance=tol,
    )

    logL_vec = compute_log_likelihood(wv["Pi"], Eo["E"], batched["root_clade_ids"])
    return [float(x) for x in logL_vec], wv["iterations"], len(cross_waves)


def run_batched_naive(items, sh, pS, pD, pL, tf, mv, Eo, device, dtype,
                      tol=1e-3, chunk_size=None):
    """Run cross-family batched wave forward with naive collate_wave (no max_wave_size)."""
    if chunk_size is not None and len(items) > chunk_size:
        all_logLs = []
        total_iters = 0
        max_nw = 0
        for start in range(0, len(items), chunk_size):
            chunk = items[start:start + chunk_size]
            logLs, iters, nw = run_batched_naive(
                chunk, sh, pS, pD, pL, tf, mv, Eo, device, dtype,
                tol=tol, chunk_size=None
            )
            all_logLs.extend(logLs)
            total_iters += iters
            max_nw = max(max_nw, nw)
        return all_logLs, total_iters, max_nw

    batched = collate_gene_families(items, dtype=dtype, device=device)
    ccp = batched["ccp"]
    meta = batched["family_meta"]

    families_waves, families_phases = [], []
    for item in items:
        ch_dev = {k: (v.to(device) if torch.is_tensor(v) else v)
                  for k, v in item["ccp"].items()}
        w, p = compute_clade_waves(ch_dev)
        families_waves.append(w)
        families_phases.append(p)

    offsets = [m["clade_offset"] for m in meta]
    cross_waves = collate_wave(families_waves, offsets)
    max_n = max(len(p) for p in families_phases)
    cross_phases = []
    for k in range(max_n):
        ph = 1
        for fp in families_phases:
            if k < len(fp):
                ph = max(ph, fp[k])
        cross_phases.append(ph)

    wave_layout = build_wave_layout(
        waves=cross_waves, phases=cross_phases,
        ccp_helpers=ccp,
        leaf_row_index=batched["leaf_row_index"],
        leaf_col_index=batched["leaf_col_index"],
        root_clade_ids=batched["root_clade_ids"],
        device=device, dtype=dtype,
    )

    wv = Pi_wave_forward(
        wave_layout=wave_layout, species_helpers=sh,
        E=Eo["E"], Ebar=Eo["E_bar"], E_s1=Eo["E_s1"], E_s2=Eo["E_s2"],
        log_pS=pS, log_pD=pD, log_pL=pL,
        transfer_mat=tf, max_transfer_mat=mv,
        device=device, dtype=dtype,
        local_iters=1000, local_tolerance=tol,
    )

    logL_vec = compute_log_likelihood(wv["Pi"], Eo["E"], batched["root_clade_ids"])
    return [float(x) for x in logL_vec], wv["iterations"], len(cross_waves)


def run_sequential(items, sh, pS, pD, pL, tf, mv, Eo, device, dtype, tol=1e-3):
    """Run wave forward on each family individually (baseline)."""
    logLs = []
    total_iters = 0
    for item in items:
        ch_dev = {k: (v.to(device) if torch.is_tensor(v) else v)
                  for k, v in item["ccp"].items()}
        li = item["leaf_row_index"].to(device)
        lc = item["leaf_col_index"].to(device)
        root_id = item["root_clade_id"]
        waves, phases = compute_clade_waves(ch_dev)
        wave_layout = build_wave_layout(
            waves=waves, phases=phases,
            ccp_helpers=ch_dev,
            leaf_row_index=li, leaf_col_index=lc,
            root_clade_ids=torch.tensor([root_id], dtype=torch.long, device=device),
            device=device, dtype=dtype,
        )
        wv = Pi_wave_forward(
            wave_layout=wave_layout, species_helpers=sh,
            E=Eo["E"], Ebar=Eo["E_bar"], E_s1=Eo["E_s1"], E_s2=Eo["E_s2"],
            log_pS=pS, log_pD=pD, log_pL=pL,
            transfer_mat=tf, max_transfer_mat=mv,
            device=device, dtype=dtype,
            local_iters=1000, local_tolerance=tol,
        )
        logL = float(compute_log_likelihood(wv["Pi"], Eo["E"], item["root_clade_id"]))
        logLs.append(logL)
        total_iters += wv["iterations"]
    return logLs, total_iters


def bench(fn, n_warmup=1, n_runs=3):
    """Benchmark fn(), return (result, median_ms)."""
    for _ in range(n_warmup):
        result = fn()
        torch.cuda.synchronize()

    times = []
    for _ in range(n_runs):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = fn()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    median = sorted(times)[len(times) // 2]
    return result, median


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="test_trees_1000")
    parser.add_argument("--max-families", type=int, default=100,
                        help="Number of families to preprocess")
    parser.add_argument("--repeat", type=int, default=1,
                        help="Repeat preprocessed families to simulate more (e.g. --repeat 10 with 100 families = 1000)")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--wave-sizes", type=str, default="256,512,1024,2048,0",
                        help="Comma-separated max wave sizes for cross scheduler (0=unlimited)")
    parser.add_argument("--family-counts", type=str, default=None,
                        help="Comma-separated family counts")
    parser.add_argument("--chunk-size", type=int, default=None,
                        help="Chunk size for batching (default: all at once, unless OOM)")
    parser.add_argument("--no-naive", action="store_true",
                        help="Skip naive collate_wave benchmark")
    parser.add_argument("--no-sequential", action="store_true",
                        help="Skip sequential per-family benchmark")
    args = parser.parse_args()

    device = torch.device("cuda")
    dtype = torch.float32
    ext = _load_cpp_ext()

    data_dir = PROJECT_ROOT / "tests" / "data" / args.dataset
    if not data_dir.exists():
        print(f"Dataset not found: {data_dir}")
        sys.exit(1)

    sp_path = str(data_dir / "sp.nwk")
    gene_paths = sorted(data_dir.glob("g_*.nwk"))
    n_avail = len(gene_paths)
    n_preprocess = min(args.max_families, n_avail)

    wave_sizes = [int(x) for x in args.wave_sizes.split(",")]
    # 0 means unlimited; we'll use a very large value
    wave_sizes_actual = [ws if ws > 0 else 100000 for ws in wave_sizes]
    wave_labels = [str(ws) if ws > 0 else "inf" for ws in wave_sizes]

    print(f"Dataset: {args.dataset}, preprocessing {n_preprocess} families...")
    t0 = time.perf_counter()
    all_items, sh, pS, pD, pL, tf, mv, Eo = preprocess_families(
        ext, sp_path, gene_paths[:n_preprocess], device, dtype
    )
    t_pre = time.perf_counter() - t0
    S = sh["S"]
    total_C_base = sum(item["ccp"]["C"] for item in all_items)
    print(f"  S={S}, {n_preprocess} families, total_C={total_C_base}, preprocess={t_pre:.1f}s")

    # Repeat to simulate more families
    if args.repeat > 1:
        print(f"  Repeating {args.repeat}x: {n_preprocess * args.repeat} virtual families")
        all_items = all_items * args.repeat

    # Determine family counts
    n_total = len(all_items)
    if args.family_counts:
        family_counts = [int(x) for x in args.family_counts.split(",")]
    else:
        family_counts = []
        n = 10
        while n <= n_total:
            family_counts.append(n)
            n *= 2
        if not family_counts or family_counts[-1] < n_total:
            family_counts.append(n_total)

    chunk_size = args.chunk_size

    # Warmup
    print(f"\nWarming up...")
    _ = run_batched_cross(all_items[:2], sh, pS, pD, pL, tf, mv, Eo, device, dtype,
                          max_wave_size=256)
    torch.cuda.synchronize()

    # Header
    print(f"\n{'mode':<18s} {'n_fam':>5s} {'max_ws':>6s} {'chunk':>5s} {'waves':>6s} "
          f"{'iters':>6s} {'time':>8s} {'ms/fam':>7s} {'fam/s':>7s}")
    print("=" * 90)

    for n_fam in family_counts:
        items = all_items[:n_fam]
        total_C = sum(item["ccp"]["C"] for item in items)
        pi_gb = total_C * S * 4 / 1e9
        print(f"\n--- {n_fam} families, total_C={total_C}, Pi=[{total_C},{S}] = {pi_gb:.1f} GB ---")

        # Decide chunk_size if not specified: auto-detect based on memory
        cs = chunk_size
        if cs is None and pi_gb > 8:  # Pi + Pibar + temporaries ~ 3x Pi
            cs = max(10, int(n_fam * 8 / pi_gb))
            print(f"  Auto chunk_size={cs} (Pi too large for single batch)")
        cs_label = str(cs) if cs else "all"

        # Sequential baseline
        if not args.no_sequential and n_fam <= 50:
            try:
                (logLs, iters), ms = bench(
                    lambda: run_sequential(items, sh, pS, pD, pL, tf, mv, Eo, device, dtype),
                    n_runs=args.runs)
                fam_s = n_fam / (ms / 1000)
                print(f"{'sequential':<18s} {n_fam:5d} {'n/a':>6s} {'n/a':>5s} {'n/a':>6s} "
                      f"{iters:6d} {ms:8.0f} {ms/n_fam:7.1f} {fam_s:7.0f}")
            except Exception as e:
                print(f"{'sequential':<18s} {n_fam:5d}  ERROR: {e}")

        # Naive collate_wave (no max_wave_size)
        if not args.no_naive:
            try:
                (logLs, iters, nw), ms = bench(
                    lambda: run_batched_naive(items, sh, pS, pD, pL, tf, mv, Eo, device, dtype,
                                             chunk_size=cs),
                    n_runs=args.runs)
                fam_s = n_fam / (ms / 1000)
                print(f"{'naive':<18s} {n_fam:5d} {'inf':>6s} {cs_label:>5s} {nw:6d} "
                      f"{iters:6d} {ms:8.0f} {ms/n_fam:7.1f} {fam_s:7.0f}")
            except Exception as e:
                print(f"{'naive':<18s} {n_fam:5d}  ERROR: {e}")
                torch.cuda.empty_cache(); gc.collect()

        # Phased cross with different max wave sizes
        for ws, ws_label in zip(wave_sizes_actual, wave_labels):
            label = f"cross(ws={ws_label})"
            try:
                (logLs, iters, nw), ms = bench(
                    lambda ws=ws: run_batched_cross(
                        items, sh, pS, pD, pL, tf, mv, Eo, device, dtype,
                        max_wave_size=ws, chunk_size=cs),
                    n_runs=args.runs)
                fam_s = n_fam / (ms / 1000)
                print(f"{label:<18s} {n_fam:5d} {ws_label:>6s} {cs_label:>5s} {nw:6d} "
                      f"{iters:6d} {ms:8.0f} {ms/n_fam:7.1f} {fam_s:7.0f}")
            except Exception as e:
                print(f"{label:<18s} {n_fam:5d}  ERROR: {e}")
                torch.cuda.empty_cache(); gc.collect()

    print("\nDone.")


if __name__ == "__main__":
    main()
