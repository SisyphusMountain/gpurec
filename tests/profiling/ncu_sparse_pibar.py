"""Minimal script for ncu profiling of sparse k=16 Pibar at S=20000.

Run with:
  /usr/local/cuda-12.8/bin/ncu --set full -o sparse_k16 python -m tests.profiling.ncu_sparse_pibar
Or for text report:
  /usr/local/cuda-12.8/bin/ncu --set full python -m tests.profiling.ncu_sparse_pibar
"""

import math
import sys
import time
from pathlib import Path

import torch
import triton
import triton.language as tl

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

from src.core.preprocess_cpp import _load_extension
from src.core.extract_parameters import extract_parameters
from src.core.likelihood import E_fixed_point, compute_log_likelihood
from src.core.scheduling import compute_clade_waves
from src.core.batching import collate_gene_families, collate_wave, build_wave_layout
from src.core.kernels.wave_step import wave_step_fused
from src.core.kernels.dts_fused import dts_fused
from src.core.log2_utils import logsumexp2

_INV_LN2 = 1.0 / math.log(2.0)
NEG_INF = float("-inf")


# ── Import the Triton kernel from the main script ──
from tests.profiling.explore_hybrid_pibar import (
    _sparse_input_pibar_kernel,
    compute_pibar_sparse_input_triton_precomputed,
    compute_pibar_dense,
    compute_pibar_uniform,
    _precompute_species_children,
    _compute_dts_cross,
    load_families_generic,
)


def main():
    device = torch.device("cuda")
    dtype = torch.float32
    k = 16
    ds_name = "test_trees_10000"

    print(f"Loading {ds_name}...")
    wave_layout, sh, pS, pD, pL, tf, mv, Eo, root_ids, device, dtype = \
        load_families_generic(1, ds_name)

    S = int(sh["S"])
    C = int(wave_layout["ccp_helpers"]["C"])
    wave_metas = wave_layout['wave_metas']
    n_waves = len(wave_metas)
    mt = mv.squeeze(-1) if mv.ndim > 1 else mv
    tf_T = tf.T.contiguous()

    print(f"S={S}, C={C}, waves={n_waves}")
    print(f"Wave sizes: {[m['W'] for m in wave_metas[:10]]}...")

    # ── Dense reference for precomputed indices ──
    print("Running dense reference...")
    ccp_helpers = wave_layout['ccp_helpers']
    leaf_row_index = wave_layout['leaf_row_index']
    leaf_col_index = wave_layout['leaf_col_index']

    _PI_INIT = torch.finfo(dtype).min
    Pi = torch.full((C, S), _PI_INIT, dtype=dtype, device=device)
    Pi[leaf_row_index, leaf_col_index] = 0.0
    Pibar = torch.full((C, S), NEG_INF, dtype=dtype, device=device)

    sp_child1, sp_child2 = _precompute_species_children(sh, S, device)
    E_vec = Eo["E"]; Ebar_vec = Eo["E_bar"]
    DL_const = 1.0 + pD + E_vec
    SL1_const = pS + Eo["E_s2"]
    SL2_const = pS + Eo["E_s1"]

    leaf_term = torch.full((C, S), NEG_INF, device=device, dtype=dtype)
    leaf_term[leaf_row_index, leaf_col_index] = 0.0
    leaf_term.add_(pS)

    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True

    # Run dense to convergence
    for wi in range(n_waves):
        meta = wave_metas[wi]
        ws, we, W = meta['start'], meta['end'], meta['W']
        if meta['has_splits']:
            dts_r = _compute_dts_cross(Pi, Pibar, meta, sp_child1, sp_child2,
                                        pD, pS, S, device, dtype)
        else:
            dts_r = None
        leaf_wt = leaf_term[ws:we]
        for li in range(50):
            Pi_W = Pi[ws:we]
            Pi_max = Pi_W.max(dim=1, keepdim=True).values
            Pibar_W = torch.log2(torch.exp2(Pi_W - Pi_max) @ tf_T) + Pi_max + mt
            Pi_new = wave_step_fused(Pi_W, Pibar_W, DL_const, Ebar_vec, E_vec,
                                      SL1_const, SL2_const, sp_child1, sp_child2,
                                      leaf_wt, dts_r)
            if li >= 3:
                sig = Pi_new > -100.0
                if not sig.any() or torch.abs(Pi_new - Pi_W)[sig].max().item() < 1e-3:
                    Pi[ws:we] = Pi_new; Pibar[ws:we] = Pibar_W; break
            Pi[ws:we] = Pi_new; Pibar[ws:we] = Pibar_W
    torch.cuda.synchronize()
    print("Dense done.")

    # Extract precomputed top-k indices
    # Pi is already in wave order (computed via Pi[ws:we] slices), so extract directly.
    precomp = {}
    for wi, meta in enumerate(wave_metas):
        ws, we = meta['start'], meta['end']
        _, idx = Pi[ws:we].topk(k, dim=1)
        precomp[wi] = idx.long().clone()

    # ── Warmup sparse run (compile Triton) ──
    print("Warmup sparse run...")
    Pi.fill_(_PI_INIT)
    Pi[leaf_row_index, leaf_col_index] = 0.0
    Pibar.fill_(NEG_INF)
    for wi in range(n_waves):
        meta = wave_metas[wi]
        ws, we, W = meta['start'], meta['end'], meta['W']
        if meta['has_splits']:
            dts_r = _compute_dts_cross(Pi, Pibar, meta, sp_child1, sp_child2,
                                        pD, pS, S, device, dtype)
        else:
            dts_r = None
        leaf_wt = leaf_term[ws:we]
        for li in range(50):
            Pi_W = Pi[ws:we]
            Pibar_W = compute_pibar_sparse_input_triton_precomputed(
                Pi_W, tf_T, mt, k, precomp[wi])
            Pi_new = wave_step_fused(Pi_W, Pibar_W, DL_const, Ebar_vec, E_vec,
                                      SL1_const, SL2_const, sp_child1, sp_child2,
                                      leaf_wt, dts_r)
            if li >= 3:
                sig = Pi_new > -100.0
                if not sig.any() or torch.abs(Pi_new - Pi_W)[sig].max().item() < 1e-3:
                    Pi[ws:we] = Pi_new; Pibar[ws:we] = Pibar_W; break
            Pi[ws:we] = Pi_new; Pibar[ws:we] = Pibar_W
    torch.cuda.synchronize()
    print("Warmup done.")

    # ── Find waves with DTS splits ──
    split_waves = [wi for wi in range(n_waves) if wave_metas[wi]['has_splits']]
    print(f"Waves with splits: {len(split_waves)} / {n_waves}")
    for wi in split_waves[:5]:
        m = wave_metas[wi]
        print(f"  wave {wi}: W={m['W']}, splits={len(m['sl'])}")

    # ── Profile the root wave ──
    root_wi = n_waves - 1
    root_meta = wave_metas[root_wi]
    print(f"\nRoot wave {root_wi}: W={root_meta['W']}, "
          f"has_splits={root_meta['has_splits']}, "
          f"n_splits={root_meta.get('n_ws', 0)}, "
          f"n_eq1={root_meta.get('n_eq1', 0)}, "
          f"n_ge2_clades={root_meta.get('n_ge2_clades', 0)}")

    # Run ALL waves to convergence, timing each
    print("\nRunning all waves to convergence with per-wave timing...")
    Pi.fill_(_PI_INIT)
    Pi[leaf_row_index, leaf_col_index] = 0.0
    Pibar.fill_(NEG_INF)
    wave_times = []
    wave_iters_list = []
    for wi in range(n_waves):
        meta = wave_metas[wi]
        ws, we, W = meta['start'], meta['end'], meta['W']
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        if meta['has_splits']:
            dts_r = _compute_dts_cross(Pi, Pibar, meta, sp_child1, sp_child2,
                                        pD, pS, S, device, dtype)
        else:
            dts_r = None
        leaf_wt = leaf_term[ws:we]
        n_iters = 0
        for li in range(50):
            Pi_W = Pi[ws:we]
            Pibar_W = compute_pibar_sparse_input_triton_precomputed(
                Pi_W, tf_T, mt, k, precomp[wi])
            Pi_new = wave_step_fused(Pi_W, Pibar_W, DL_const, Ebar_vec, E_vec,
                                      SL1_const, SL2_const, sp_child1, sp_child2,
                                      leaf_wt, dts_r)
            n_iters = li + 1
            if li >= 3:
                sig = Pi_new > -100.0
                if not sig.any() or torch.abs(Pi_new - Pi_W)[sig].max().item() < 1e-3:
                    Pi[ws:we] = Pi_new; Pibar[ws:we] = Pibar_W; break
            Pi[ws:we] = Pi_new; Pibar[ws:we] = Pibar_W
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        wave_times.append(elapsed)
        wave_iters_list.append(n_iters)

    total_time = sum(wave_times)
    print(f"\nTotal Pi convergence: {total_time*1000:.1f} ms across {n_waves} waves")
    print(f"\nTop 10 waves by time:")
    ranked = sorted(range(n_waves), key=lambda i: wave_times[i], reverse=True)
    for rank, wi in enumerate(ranked[:10]):
        m = wave_metas[wi]
        pct = 100 * wave_times[wi] / total_time
        print(f"  #{rank+1} wave {wi:2d}: W={m['W']:5d}, "
              f"{wave_iters_list[wi]:2d} iters, "
              f"{wave_times[wi]*1000:8.1f} ms ({pct:5.1f}%), "
              f"has_splits={m['has_splits']}")
    print(f"\nRoot wave {root_wi}: {wave_times[root_wi]*1000:.1f} ms "
          f"({100*wave_times[root_wi]/total_time:.1f}% of total)")

    # ── nsys-profiled section: root wave only, 2 iters ──
    # Re-run all waves to convergence first (unprofiled), then profile root
    Pi.fill_(_PI_INIT)
    Pi[leaf_row_index, leaf_col_index] = 0.0
    Pibar.fill_(NEG_INF)
    for wi in range(root_wi):
        meta = wave_metas[wi]
        ws, we, W = meta['start'], meta['end'], meta['W']
        if meta['has_splits']:
            dts_r = _compute_dts_cross(Pi, Pibar, meta, sp_child1, sp_child2,
                                        pD, pS, S, device, dtype)
        else:
            dts_r = None
        leaf_wt = leaf_term[ws:we]
        for li in range(50):
            Pi_W = Pi[ws:we]
            Pibar_W = compute_pibar_sparse_input_triton_precomputed(
                Pi_W, tf_T, mt, k, precomp[wi])
            Pi_new = wave_step_fused(Pi_W, Pibar_W, DL_const, Ebar_vec, E_vec,
                                      SL1_const, SL2_const, sp_child1, sp_child2,
                                      leaf_wt, dts_r)
            if li >= 3:
                sig = Pi_new > -100.0
                if not sig.any() or torch.abs(Pi_new - Pi_W)[sig].max().item() < 1e-3:
                    Pi[ws:we] = Pi_new; Pibar[ws:we] = Pibar_W; break
            Pi[ws:we] = Pi_new; Pibar[ws:we] = Pibar_W
    torch.cuda.synchronize()
    print(f"\nPre-converged waves 0..{root_wi-1}, now profiling root wave...")

    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()

    meta = wave_metas[root_wi]
    ws, we, W = meta['start'], meta['end'], meta['W']
    if meta['has_splits']:
        dts_r = _compute_dts_cross(Pi, Pibar, meta, sp_child1, sp_child2,
                                    pD, pS, S, device, dtype)
    else:
        dts_r = None
    leaf_wt = leaf_term[ws:we]
    for li in range(2):
        Pi_W = Pi[ws:we]
        Pibar_W = compute_pibar_sparse_input_triton_precomputed(
            Pi_W, tf_T, mt, k, precomp[root_wi])
        Pi_new = wave_step_fused(
            Pi_W, Pibar_W, DL_const, Ebar_vec, E_vec,
            SL1_const, SL2_const, sp_child1, sp_child2,
            leaf_wt, dts_r)
        Pi[ws:we] = Pi_new
        Pibar[ws:we] = Pibar_W

    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    print("Profiled section done.")

    torch.backends.cuda.matmul.allow_tf32 = prev_tf32


if __name__ == "__main__":
    main()
