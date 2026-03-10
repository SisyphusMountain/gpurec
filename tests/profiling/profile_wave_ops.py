#!/usr/bin/env python3
"""Profile per-wave operation timings for cross-family batched Pi_wave_forward.

All families are collated and processed in parallel: each wave contains clades
from ALL families, so wave sizes reflect the actual GPU workload.

Usage:
    python tests/profiling/profile_wave_ops.py [--families N] [--dataset DIR] [--chunk-size K]
"""

import argparse
import math
import pathlib
import sys
import time

import torch

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.preprocess_cpp import _load_extension as _load_cpp_ext
from src.core.extract_parameters import extract_parameters
from src.core.likelihood import E_fixed_point, compute_log_likelihood
from src.core.scheduling import compute_clade_waves
from src.core.batching import collate_gene_families, collate_wave
from src.core.kernels.wave_step import wave_step_fused
from src.core.kernels.dts_fused import dts_fused
from src.core.log2_utils import logsumexp2

_INV = 1.0 / math.log(2.0)
NEG_INF = float("-inf")


# ── helpers ──────────────────────────────────────────────────────────────────

def _cuda_timer():
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    return s, e


def _elapsed_ms(start, end):
    return start.elapsed_time(end)


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

    # Shared species helpers
    sh = {
        "S": int(sr["S"]),
        "names": sr["names"],
        "s_P_indexes": sr["s_P_indexes"].to(device=device),
        "s_C12_indexes": sr["s_C12_indexes"].to(device=device),
        "Recipients_mat": sr["Recipients_mat"].to(dtype=dtype, device=device),
    }
    theta = torch.log(torch.tensor([0.05, 0.05, 0.05], dtype=dtype, device=device))
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


def build_cross_family_waves(batch_items, batched, device):
    """Compute per-family waves, merge into cross-family waves + phases."""
    families_waves = []
    families_phases = []
    for item in batch_items:
        ch_dev = {k: (v.to(device) if torch.is_tensor(v) else v)
                  for k, v in item["ccp"].items()}
        waves_i, phases_i = compute_clade_waves(ch_dev)
        families_waves.append(waves_i)
        families_phases.append(phases_i)

    offsets = [m["clade_offset"] for m in batched["family_meta"]]
    cross_waves = collate_wave(families_waves, offsets)

    max_n_waves = max(len(p) for p in families_phases)
    cross_phases = []
    for k in range(max_n_waves):
        phase_k = 1
        for fp in families_phases:
            if k < len(fp):
                phase_k = max(phase_k, fp[k])
        cross_phases.append(phase_k)

    return cross_waves, cross_phases


# ── main profiling logic ────────────────────────────────────────────────────

def profile_batched(batch_items, sh, pS, pD, pL, tf, mv, Eo, device, dtype,
                    tol=1e-3, max_iters=1000):
    """Run the large-S wave loop on cross-family batched data with CUDA-event timing.

    Returns a list of dicts (one per wave) with timing fields in ms,
    plus a summary dict.
    """
    # 1. Collate families
    batched = collate_gene_families(batch_items, dtype=dtype, device=device)
    ccp = batched["ccp"]
    li = batched["leaf_row_index"]
    lc = batched["leaf_col_index"]
    root_ids = batched["root_clade_ids"]

    C = int(ccp["C"])
    S = sh["S"]
    N_splits = ccp["N_splits"]

    # 2. Build cross-family waves
    cross_waves, cross_phases = build_cross_family_waves(batch_items, batched, device)

    # 3. Prepare wave-forward data structures (mirrors Pi_wave_forward in likelihood.py)
    E_val = Eo["E"]
    Ebar_val = Eo["E_bar"]
    E_s1 = Eo["E_s1"]
    E_s2 = Eo["E_s2"]

    split_lr = ccp["split_leftrights_sorted"].to(device)
    split_parents = ccp["split_parents_sorted"].to(device)
    log_sp = ccp["log_split_probs_sorted"].to(device).unsqueeze(1).contiguous()
    lefts = split_lr[:N_splits]
    rights = split_lr[N_splits:]

    # Species-tree child indices
    sp_P_idx = sh["s_P_indexes"]
    sp_c12_idx = sh["s_C12_indexes"]
    sp_child1 = torch.full((S,), S, dtype=torch.long, device=device)
    sp_child2 = torch.full((S,), S, dtype=torch.long, device=device)
    for i in range(len(sp_P_idx)):
        p = int(sp_P_idx[i].item())
        c = int(sp_c12_idx[i].item())
        if p < S:
            sp_child1[p] = c
        else:
            sp_child2[p - S] = c

    # Constant DTS_L terms
    DL_const = 1.0 + pD + E_val
    SL1_const = pS + E_s2
    SL2_const = pS + E_s1
    mt_squeezed = mv
    transfer_mat_T = tf.T.contiguous()

    # Clade-species map & leaf term
    clade_species_map = torch.full((C, S), NEG_INF, device=device, dtype=dtype)
    clade_species_map[li, lc] = 0.0
    leaf_term = pS + clade_species_map

    # Init Pi
    _PI_INIT = torch.finfo(dtype).min
    Pi = torch.full((C, S), _PI_INIT, dtype=dtype, device=device)
    Pi[li, lc] = 0.0
    Pibar = torch.full((C, S), NEG_INF, dtype=dtype, device=device)

    # Build parent-to-splits mapping
    parent_to_splits = {}
    for i in range(N_splits):
        p = int(split_parents[i].item())
        parent_to_splits.setdefault(p, []).append(i)

    # Precompute per-wave data
    wave_data = []
    for wave_ids in cross_waves:
        if not wave_ids:
            wave_data.append(None)
            continue
        wt = torch.tensor(wave_ids, dtype=torch.long, device=device)
        W = len(wave_ids)
        wsids = []
        for c_id in wave_ids:
            if c_id in parent_to_splits:
                wsids.extend(parent_to_splits[c_id])
        if wsids:
            wst = torch.tensor(wsids, dtype=torch.long, device=device)
            n_ws = len(wsids)
            sl = lefts[wst]
            sr = rights[wst]
            wlsp = log_sp[wst]
            wsp = split_parents[wst]
            clade_to_wi = torch.empty(C, dtype=torch.long, device=device)
            clade_to_wi[wt] = torch.arange(W, device=device)
            reduce_idx = clade_to_wi[wsp]
            wave_data.append((wt, W, True, n_ws, sl, sr, wlsp, reduce_idx))
        else:
            wave_data.append((wt, W, False, 0, None, None, None, None))

    wave_leaf_terms = []
    for wd in wave_data:
        if wd is None:
            wave_leaf_terms.append(None)
        else:
            wave_leaf_terms.append(leaf_term[wd[0]].contiguous())

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.cuda.synchronize()

    # ── Per-wave profiling ───────────────────────────────────────────────
    results = []

    for wi, wd in enumerate(wave_data):
        if wd is None:
            continue

        wt, W = wd[0], wd[1]
        has_splits = wd[2]
        n_ws = wd[3]
        leaf_wt = wave_leaf_terms[wi]

        rec = {
            "wave": wi, "W": W, "n_splits": n_ws,
            "has_splits": has_splits, "phase": cross_phases[wi],
        }

        # ── DTS cross-clade kernel ───────────────────────────────────────
        ev_dts_s, ev_dts_e = _cuda_timer()
        ev_dts_s.record()
        if has_splits:
            sl, sr, wlsp, reduce_idx = wd[4], wd[5], wd[6], wd[7]
            dts_term = dts_fused(
                Pi, Pibar, sl, sr,
                sp_child1, sp_child2,
                pD, pS, wlsp,
            )
        ev_dts_e.record()

        # ── DTS reduce (scatter_reduce + logsumexp) ──────────────────────
        ev_red_s, ev_red_e = _cuda_timer()
        ev_red_s.record()
        if has_splits:
            reduce_exp = reduce_idx.unsqueeze(1).expand_as(dts_term)
            dts_r = torch.full((W, S), NEG_INF, device=device, dtype=dtype)
            dts_r.scatter_reduce_(0, reduce_exp, dts_term, reduce="amax",
                                  include_self=True)
            dts_max = dts_r.clone()
            dts_sum = torch.zeros((W, S), device=device, dtype=dtype)
            dts_sum.scatter_add_(0, reduce_exp,
                                 torch.exp2(dts_term - dts_max[reduce_idx]))
            dts_r = torch.log2(dts_sum) + dts_max
        else:
            dts_r = None
        ev_red_e.record()

        # ── Self-loop iterations ─────────────────────────────────────────
        t_gather = 0.0
        t_pibar = 0.0
        t_step = 0.0
        t_scatter = 0.0
        n_iters = 0

        for local_iter in range(max_iters):
            n_iters += 1

            # gather
            eg_s, eg_e = _cuda_timer()
            eg_s.record()
            Pi_W = Pi[wt].contiguous()
            eg_e.record()

            # pibar matmul
            ep_s, ep_e = _cuda_timer()
            ep_s.record()
            Pi_max = Pi_W.max(dim=1, keepdim=True).values
            Pibar_W = (torch.log2(torch.exp2(Pi_W - Pi_max) @ transfer_mat_T)
                       + Pi_max + mt_squeezed)
            ep_e.record()

            # wave_step_fused
            es_s, es_e = _cuda_timer()
            es_s.record()
            Pi_new = wave_step_fused(
                Pi_W, Pibar_W,
                DL_const, Ebar_val, E_val, SL1_const, SL2_const,
                sp_child1, sp_child2, leaf_wt, dts_r,
            )
            es_e.record()

            # scatter
            ew_s, ew_e = _cuda_timer()
            ew_s.record()
            Pi[wt] = Pi_new
            Pibar[wt] = Pibar_W
            ew_e.record()

            # sync to read timings
            torch.cuda.synchronize()
            t_gather += _elapsed_ms(eg_s, eg_e)
            t_pibar += _elapsed_ms(ep_s, ep_e)
            t_step += _elapsed_ms(es_s, es_e)
            t_scatter += _elapsed_ms(ew_s, ew_e)

            # Convergence check
            if local_iter >= 3:
                significant = Pi_new > -100.0
                if (not significant.any()
                        or torch.abs(Pi_new - Pi_W)[significant].max().item()
                        < tol):
                    break

        # Record DTS timings
        torch.cuda.synchronize()
        rec["t_dts_kernel"] = _elapsed_ms(ev_dts_s, ev_dts_e)
        rec["t_dts_reduce"] = _elapsed_ms(ev_red_s, ev_red_e)
        rec["t_gather"] = t_gather
        rec["t_pibar"] = t_pibar
        rec["t_step"] = t_step
        rec["t_scatter"] = t_scatter
        rec["n_iters"] = n_iters
        rec["t_total"] = (rec["t_dts_kernel"] + rec["t_dts_reduce"]
                          + t_gather + t_pibar + t_step + t_scatter)
        results.append(rec)

    # Compute log-likelihoods for verification
    logL_vec = compute_log_likelihood(Pi, Eo["E"], root_ids)
    logLs = [float(x) for x in logL_vec]

    return results, logLs


# ── printing ─────────────────────────────────────────────────────────────────

def print_report(results, n_families, logLs=None):
    header = (
        f"{'wave':>5s} {'ph':>2s} {'W':>6s} {'splits':>7s} {'iters':>5s} "
        f"{'dts_k':>7s} {'dts_r':>7s} {'gather':>7s} {'pibar':>7s} "
        f"{'step':>7s} {'scat':>7s} {'total':>7s}"
    )
    print(header)
    print("─" * len(header))

    totals = {k: 0.0 for k in [
        "t_dts_kernel", "t_dts_reduce", "t_gather",
        "t_pibar", "t_step", "t_scatter", "t_total",
    ]}
    total_iters = 0

    for r in results:
        print(
            f"{r['wave']:5d} {r['phase']:2d} {r['W']:6d} {r['n_splits']:7d} "
            f"{r['n_iters']:5d} "
            f"{r['t_dts_kernel']:7.2f} {r['t_dts_reduce']:7.2f} "
            f"{r['t_gather']:7.2f} {r['t_pibar']:7.2f} "
            f"{r['t_step']:7.2f} {r['t_scatter']:7.2f} "
            f"{r['t_total']:7.2f}"
        )
        for k in totals:
            totals[k] += r[k]
        total_iters += r["n_iters"]

    print("─" * len(header))
    gt = totals["t_total"]
    pct = lambda v: f"{100*v/gt:.1f}%" if gt > 0 else "n/a"
    print(
        f"{'SUM':>5s} {'':>2s} {'':>6s} {'':>7s} {total_iters:5d} "
        f"{totals['t_dts_kernel']:7.2f} {totals['t_dts_reduce']:7.2f} "
        f"{totals['t_gather']:7.2f} {totals['t_pibar']:7.2f} "
        f"{totals['t_step']:7.2f} {totals['t_scatter']:7.2f} "
        f"{gt:7.2f}"
    )
    print(
        f"\n  Breakdown:  dts_kernel={pct(totals['t_dts_kernel'])}  "
        f"dts_reduce={pct(totals['t_dts_reduce'])}  "
        f"gather={pct(totals['t_gather'])}  "
        f"pibar={pct(totals['t_pibar'])}  "
        f"step={pct(totals['t_step'])}  "
        f"scatter={pct(totals['t_scatter'])}"
    )
    print(f"  Total: {gt:.1f}ms  ({total_iters} iterations, {n_families} families)")
    print(f"  Per family: {gt/n_families:.1f}ms")

    if logLs:
        finite = [l for l in logLs if math.isfinite(l)]
        print(f"  LogL: {len(finite)}/{len(logLs)} finite, "
              f"mean={sum(finite)/len(finite):.2f}" if finite else "  LogL: none finite")


# ── entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--families", type=int, default=10,
                        help="Number of gene families to profile")
    parser.add_argument("--dataset", type=str, default="test_trees_1000",
                        help="Dataset name under tests/data/")
    parser.add_argument("--chunk-size", type=int, default=None,
                        help="Process families in chunks (for memory). Default: all at once")
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

    n = min(args.families, len(gene_paths))
    print(f"Cross-family batched profiling: {n} families from {args.dataset}")
    print(f"All times in milliseconds.  Columns:")
    print(f"  ph      = phase (1=leaf, 2=internal, 3=root)")
    print(f"  W       = wave size (clades from ALL families combined)")
    print(f"  splits  = total splits across all families for this wave")
    print(f"  iters   = convergence iterations")
    print(f"  dts_k   = DTS cross-clade fused kernel")
    print(f"  dts_r   = DTS scatter-reduce to [W,S]")
    print(f"  gather  = Pi[wt].contiguous()")
    print(f"  pibar   = max + exp2 + cuBLAS TF32 matmul + log2 + add")
    print(f"  step    = wave_step_fused Triton kernel")
    print(f"  scat    = Pi[wt]=Pi_new; Pibar[wt]=Pibar_W\n")

    # Preprocess all families
    gene_subset = gene_paths[:n]
    print(f"Preprocessing {n} families...")
    t0 = time.perf_counter()
    batch_items, sh, pS, pD, pL, tf, mv, Eo = preprocess_families(
        ext, sp_path, gene_subset, device, dtype
    )
    t_preproc = time.perf_counter() - t0
    S = sh["S"]
    total_C = sum(item["ccp"]["C"] for item in batch_items)
    print(f"  S={S}, total_C={total_C}, preprocessing: {t_preproc:.1f}s")

    if args.chunk_size and n > args.chunk_size:
        # Process in chunks, printing per-chunk reports
        for start in range(0, n, args.chunk_size):
            end = min(start + args.chunk_size, n)
            chunk = batch_items[start:end]
            chunk_n = len(chunk)
            print(f"\n── Chunk {start}-{end-1} ({chunk_n} families) ──")

            # Warmup
            if start == 0:
                print("  Warmup run...")
                profile_batched(chunk[:2], sh, pS, pD, pL, tf, mv, Eo, device, dtype)

            results, logLs = profile_batched(
                chunk, sh, pS, pD, pL, tf, mv, Eo, device, dtype
            )
            print_report(results, chunk_n, logLs)
    else:
        # Single batch
        print(f"\nWarmup run...")
        profile_batched(batch_items[:min(2, n)], sh, pS, pD, pL, tf, mv, Eo, device, dtype)

        print(f"\n── All {n} families batched together ──")
        results, logLs = profile_batched(
            batch_items, sh, pS, pD, pL, tf, mv, Eo, device, dtype
        )
        print_report(results, n, logLs)


if __name__ == "__main__":
    main()
